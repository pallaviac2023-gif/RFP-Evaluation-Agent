"""
tools/evaluation_agent.py
---------------------------
Evaluation Agent: uses the active criteria to score ONE supplier's
proposal and cites evidence, returning JSON only.

Responsibility (per project brief):
    "Uses active criteria to score one supplier and cites evidence."
    -> LLM with structured JSON output

Important: this agent may JUDGE proposal content, but it must NOT
decide final arithmetic, benchmarks, tie-breaks, or rank — those are
handled deterministically by ranking_tool.py.

This module calls Google's Gemini API (google-genai SDK) and uses
Gemini's native structured-output support (response_schema), which
constrains generation to valid JSON at the API level rather than
relying solely on prompt instructions. The rest of the pipeline
(Validation Tool, Ranking Tool, Orchestrator) is provider-agnostic —
it only ever sees the parsed dict this module returns, so swapping to
a different JSON-capable LLM only requires changing this one file.
"""

from __future__ import annotations
import json
import os
from typing import Any

from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.6-flash"  # any JSON-capable Gemini model; swap freely


# --------------------------------------------------------------------------
# Response schema (Gemini enforces this at generation time)
# --------------------------------------------------------------------------
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "supplier_name": {"type": "string"},
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion_id": {"type": "integer"},
                    "score": {"type": "number"},
                    "max_score": {"type": "number"},
                    "justification": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["criterion_id", "score", "max_score", "justification", "evidence"],
            },
        },
        "risks": {"type": "array", "items": {"type": "string"}},
        "overall_summary": {"type": "string"},
    },
    "required": ["supplier_name", "criteria", "risks", "overall_summary"],
}


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
def build_prompt(criteria: list[dict], supplier_name: str, document_text: str) -> str:
    """
    Build the evaluation prompt. Requirements enforced:
      - use ONLY evidence present in the supplier document
      - return exactly one result for every active criterion
      - stay within the criterion's score range
    (JSON-only output is enforced by RESPONSE_SCHEMA below, not by
    prompt instructions, so no schema example needs to be embedded
    in the prompt itself — Gemini's docs note that duplicating the
    schema in the prompt can actually reduce output quality.)
    """
    criteria_block = "\n".join(
        f'- criterion_id={c["criterion_id"]}, name="{c["name"]}", '
        f'max_score={c["max_score"]}, what to inspect: {c["description"]}'
        for c in criteria
    )

    return f"""You are a procurement evaluation assistant. Score the supplier proposal
below against ONLY the following active evaluation criteria. Do not invent
criteria. Do not perform any weighting, ranking, or comparison to other
suppliers — that is handled separately.

ACTIVE CRITERIA:
{criteria_block}

RULES:
1. Base every score and justification ONLY on evidence actually present in
   the supplier document text below. Do not assume unstated facts.
2. Return EXACTLY ONE result for EVERY criterion listed above — do not skip
   any, and do not add extras.
3. Each "score" must be a number between 0 and that criterion's max_score
   (inclusive). If the document does not address a criterion, score it low
   (e.g. 0-2) and say so in the justification — do not omit it.
4. "evidence" must be a short paraphrase or quote grounded in the document
   text — never fabricated.

SUPPLIER NAME: {supplier_name}

SUPPLIER DOCUMENT TEXT:
\"\"\"
{document_text}
\"\"\"
"""


class LLMEmptyResponseError(Exception):
    """Raised when Gemini returns no usable text — most commonly because
    'thinking' tokens consumed the entire max_output_tokens budget before
    any visible output was written. See call_llm() for the mitigation."""


def call_llm(prompt: str, api_key: str | None = None, max_tokens: int = 8000) -> str:
    """
    Call Gemini and return the raw text response (guaranteed valid JSON
    matching RESPONSE_SCHEMA, per Gemini's structured-output contract).

    Important Gemini-specific gotcha: Gemini models have "thinking"
    enabled by default, and thinking tokens are deducted from the SAME
    max_output_tokens budget as the visible answer — unlike reasoning-
    token accounting on some other providers. On a non-trivial prompt
    this can silently consume the entire budget and return an EMPTY
    response body (finish_reason=MAX_TOKENS, no error raised), which
    then fails JSON parsing downstream and looks like "the LLM didn't
    return anything" rather than a token-budget problem. We avoid this
    two ways:
      1. Set thinking to its lowest level — this is a straightforward
         extraction/scoring task, not one that benefits from extended
         reasoning. NOTE: the config parameter for this differs by
         model generation, and Gemini rejects a request that sets both:
           - Gemini 3.x models (e.g. gemini-3.6-flash, used here):
             thinking_level = "minimal" | "low" | "medium" | "high"
           - Gemini 2.5 and earlier: thinking_budget (an integer token
             count; 0 disables thinking)
         MODEL_NAME above is a 3.x model, so thinking_level is used. If
         you swap MODEL_NAME to a 2.5-era model, switch this to
         thinking_budget=0 instead — don't set both.
      2. Use a generous max_output_tokens ceiling as a safety margin.
    If the response still comes back empty, we raise explicitly instead
    of returning "" and letting a confusing all-fields-missing validation
    warning be the only symptom.
    """
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        ),
    )

    if not response.text:
        finish_reason = None
        try:
            finish_reason = response.candidates[0].finish_reason
        except (IndexError, AttributeError):
            pass
        raise LLMEmptyResponseError(
            f"Gemini returned an empty response (finish_reason={finish_reason}). "
            f"If finish_reason is MAX_TOKENS, raise max_output_tokens further."
        )

    return response.text


def _strip_json_fences(text: str) -> str:
    """Defensive cleanup in case a future model/config wraps JSON in
    ```json fences despite response_mime_type=application/json."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1] if cleaned.count("```") >= 2 else cleaned
        cleaned = cleaned.removeprefix("json").strip()
    return cleaned.strip("`").strip()


def parse_llm_json(raw_text: str) -> dict[str, Any]:
    """
    Parse the LLM's response into a dict. Returns a dict in all cases —
    if parsing fails entirely, returns an empty-ish structure so the
    Validation Tool can still normalize it (rather than crashing the
    orchestrator on one bad supplier).
    """
    cleaned = _strip_json_fences(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-resort: find the outermost {...} block
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {"supplier_name": "", "criteria": [], "risks": [], "overall_summary": ""}


# --------------------------------------------------------------------------
# Public entry point used by the orchestrator
# --------------------------------------------------------------------------
def evaluate_supplier(
    supplier_name: str,
    document_text: str,
    criteria: list[dict],
    api_key: str | None = None,
) -> dict[str, Any]:
    """Build the prompt, call the LLM, and return the parsed (but not yet
    validated/normalized) JSON dict for one supplier."""
    prompt = build_prompt(criteria, supplier_name, document_text)
    raw_response = call_llm(prompt, api_key=api_key)
    return parse_llm_json(raw_response)
