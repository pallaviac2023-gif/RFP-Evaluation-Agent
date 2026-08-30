"""
tools/validation_tool.py
--------------------------
Validation Tool: checks the LLM's JSON output against a schema, fills
in missing criteria, clips out-of-range scores, and records warnings.

Responsibility (per project brief):
    "Checks schema, fills missing criteria, clips invalid scores,
     and records warnings." -> Pydantic / custom Python

This tool never invents a score for a criterion the LLM addressed —
it only repairs structural problems (missing entries, bad types,
out-of-range values) so downstream deterministic scoring never
crashes or silently uses garbage.
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field, ValidationError, field_validator


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
class CriterionResult(BaseModel):
    criterion_id: int
    score: float
    max_score: float = 10
    justification: str = ""
    evidence: str = ""

    @field_validator("score")
    @classmethod
    def score_must_be_finite(cls, v):
        if v is None:
            raise ValueError("score is None")
        return v


class EvaluationResult(BaseModel):
    supplier_name: str
    criteria: list[CriterionResult] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    overall_summary: str = ""


class ValidationOutcome(BaseModel):
    """Wraps a normalized EvaluationResult plus a list of human-readable
    warnings describing every repair that was made, for auditability."""
    result: EvaluationResult
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Core validation / normalization logic
# --------------------------------------------------------------------------
def validate_and_normalize(
    raw_llm_output: dict[str, Any],
    active_criteria: list[dict],
    supplier_name_fallback: str,
) -> ValidationOutcome:
    """
    Validate raw LLM JSON against the schema and repair it so that
    every active criterion has exactly one valid, in-range result.

    Parameters
    ----------
    raw_llm_output : dict parsed from the LLM's JSON response
    active_criteria : the list of criteria dicts from the DB
                       (criterion_id, name, weight, max_score, ...)
    supplier_name_fallback : supplier name entered by the user, used
                              if the LLM omits/mismatches the name

    Returns
    -------
    ValidationOutcome with a fully-populated EvaluationResult (one
    entry per active criterion, scores clipped to [0, max_score]) and
    a list of warnings describing every correction applied.
    """
    warnings: list[str] = []

    # --- Step 1: structural parse (schema check) ---------------------
    try:
        parsed = EvaluationResult.model_validate(raw_llm_output)
    except ValidationError as e:
        warnings.append(f"Malformed LLM JSON, attempting partial recovery: {e}")
        parsed = _best_effort_partial_parse(raw_llm_output, supplier_name_fallback, warnings)

    if not parsed.supplier_name or not parsed.supplier_name.strip():
        parsed.supplier_name = supplier_name_fallback
        warnings.append("Missing supplier_name in LLM output; used form input instead.")

    # --- Step 2: index existing criterion results by ID ---------------
    by_id: dict[int, CriterionResult] = {c.criterion_id: c for c in parsed.criteria}

    normalized_criteria: list[CriterionResult] = []
    active_ids = {c["criterion_id"] for c in active_criteria}

    for crit in active_criteria:
        cid = crit["criterion_id"]
        max_score = crit["max_score"]

        if cid not in by_id:
            warnings.append(
                f"Criterion '{crit['name']}' (id={cid}) missing from LLM output; "
                f"filled with score 0 and flagged for manual review."
            )
            normalized_criteria.append(
                CriterionResult(
                    criterion_id=cid,
                    score=0,
                    max_score=max_score,
                    justification="MISSING — not returned by LLM.",
                    evidence="",
                )
            )
            continue

        item = by_id[cid]
        score = item.score
        clipped = False

        if score is None:
            score = 0
            clipped = True
        if score < 0:
            score = 0
            clipped = True
        if score > max_score:
            score = max_score
            clipped = True

        if clipped:
            warnings.append(
                f"Criterion '{crit['name']}' (id={cid}) had an out-of-range or "
                f"invalid score ({item.score}); clipped to {score}."
            )

        normalized_criteria.append(
            CriterionResult(
                criterion_id=cid,
                score=score,
                max_score=max_score,
                justification=item.justification or "",
                evidence=item.evidence or "",
            )
        )

    # --- Step 3: flag any extra criteria the LLM invented --------------
    for extra_id in by_id.keys() - active_ids:
        warnings.append(
            f"LLM returned a result for inactive/unknown criterion_id={extra_id}; ignored."
        )

    parsed.criteria = normalized_criteria
    return ValidationOutcome(result=parsed, warnings=warnings)


def _best_effort_partial_parse(
    raw: dict[str, Any], supplier_name_fallback: str, warnings: list[str]
) -> EvaluationResult:
    """Salvage whatever usable fields exist when strict validation fails
    (e.g. the LLM returned extra prose, wrong types, or a truncated list)."""
    supplier_name = raw.get("supplier_name") or supplier_name_fallback

    criteria_raw = raw.get("criteria", [])
    if not isinstance(criteria_raw, list):
        criteria_raw = []
        warnings.append("'criteria' field was not a list; treated as empty.")

    salvaged: list[CriterionResult] = []
    for entry in criteria_raw:
        if not isinstance(entry, dict):
            continue
        try:
            cid = int(entry.get("criterion_id"))
            score = float(entry.get("score", 0))
        except (TypeError, ValueError):
            warnings.append(f"Could not parse a criterion entry: {entry!r}; skipped.")
            continue
        salvaged.append(
            CriterionResult(
                criterion_id=cid,
                score=score,
                max_score=float(entry.get("max_score", 10)),
                justification=str(entry.get("justification", "")),
                evidence=str(entry.get("evidence", "")),
            )
        )

    risks = raw.get("risks", [])
    if not isinstance(risks, list):
        risks = []

    return EvaluationResult(
        supplier_name=supplier_name,
        criteria=salvaged,
        risks=[str(r) for r in risks],
        overall_summary=str(raw.get("overall_summary", "")),
    )
