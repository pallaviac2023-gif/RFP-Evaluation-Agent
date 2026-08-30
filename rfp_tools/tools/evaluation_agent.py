from __future__ import annotations
import json
import os
from typing import Any

from langchain_openai import ChatOpenAI

# OpenRouter is OpenAI-compatible, so LangChain's ChatOpenAI can talk to
# it directly by pointing base_url at OpenRouter instead of OpenAI.
# "model" is a string of the form "<provider>/<model-name>"; free-tier
# models carry a ":free" suffix. Swap this for any other OpenRouter model.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
def build_prompt(criteria: list[dict], supplier_name: str, document_text: str) -> str:
    """
    Build the evaluation prompt. Requirements enforced in the prompt text:
      - use ONLY evidence present in the supplier document
      - return exactly one result for every active criterion
      - stay within the criterion's score range
      - output JSON only, no preamble or markdown fences
    """
    criteria_block = "\n".join(
        f'- criterion_id={c["criterion_id"]}, name="{c["name"]}", '
        f'max_score={c["max_score"]}, what to inspect: {c["description"]}'
        for c in criteria
    )

    schema_example = {
        "supplier_name": supplier_name,
        "criteria": [
            {
                "criterion_id": criteria[0]["criterion_id"] if criteria else 1,
                "score": 0,
                "max_score": criteria[0]["max_score"] if criteria else 10,
                "justification": "one or two sentences",
                "evidence": "short quote or paraphrase from the document",
            }
        ],
        "risks": ["short risk statement", "..."],
        "overall_summary": "2-3 sentence neutral summary",
    }

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
5. Output VALID JSON ONLY. No markdown code fences, no preamble, no
   explanation outside the JSON object. Match this exact shape:

{json.dumps(schema_example, indent=2)}

SUPPLIER NAME: {supplier_name}

SUPPLIER DOCUMENT TEXT:
\"\"\"
{document_text}
\"\"\"
"""


# --------------------------------------------------------------------------
# LLM call
# --------------------------------------------------------------------------
def call_llm(
    prompt: str,
    api_key: str | None = None,
    workspace_id: str | None = None,  # kept for call-signature compatibility; unused
    max_tokens: int = 2000,
) -> str:
    """
    Call the LLM via OpenRouter's OpenAI-compatible chat-completions
    endpoint and return the raw text response (expected to be JSON).

    Requires an OpenRouter API key, either passed in directly or set as
    the OPENROUTER_API_KEY environment variable. Get one free at
    https://openrouter.ai/keys
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "No OpenRouter API key found. Pass api_key= or set the "
            "OPENROUTER_API_KEY environment variable."
        )

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,  # deterministic-ish output, helps with strict JSON
        max_tokens=max_tokens,
        openai_api_key=key,
        base_url=OPENROUTER_BASE_URL,
    )
    response = llm.invoke(prompt)
    return response.content


def _strip_json_fences(text: str) -> str:
    """Defensive cleanup in case the model wraps JSON in ```json fences
    despite instructions not to."""
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
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Build the prompt, call the LLM, and return the parsed (but not yet
    validated/normalized) JSON dict for one supplier."""
    prompt = build_prompt(criteria, supplier_name, document_text)
    raw_response = call_llm(prompt, api_key=api_key, workspace_id=workspace_id)
    return parse_llm_json(raw_response)