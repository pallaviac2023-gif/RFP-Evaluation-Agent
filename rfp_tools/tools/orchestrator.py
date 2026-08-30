"""
tools/orchestrator.py
------------------------
Orchestrator Agent: controls the workflow and calls the required tools
in order for an entire RFP evaluation batch.

Responsibility (per project brief):
    "Controls the workflow and calls the required tools in order."
    -> Python functions or an agent framework

Pipeline (matches the "Required Architecture and Data Flow" section):
    1. Load active criteria
    2. For each supplier: extract PDF text (Document Tool)
    3. For each supplier: score via LLM (Evaluation Agent)
    4. For each supplier: validate/normalize (Validation Tool)
    5. Score + benchmark + rank the whole batch (Ranking Tool)
    6. Persist every result under one RFP_RUN_ID, including a snapshot
       of the criteria used and any batch-level warnings, so the run
       can be reloaded later from the database (see "Previous Runs" in
       app.py / database.get_full_run_result).
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Any

from db import database
from tools.document_tool import extract_text_from_pdf, DocumentExtractionError
from tools.evaluation_agent import evaluate_supplier
from tools.validation_tool import validate_and_normalize
from tools.ranking_tool import SupplierInput, score_and_rank_batch


@dataclass
class SupplierSubmission:
    """Raw input collected from the Streamlit upload form for one supplier."""
    supplier_name: str
    submission_date: date
    experience_rating: float
    pdf_bytes: bytes
    filename: str


class OrchestratorError(Exception):
    """Raised for batch-level failures (e.g. no active criteria)."""


def run_batch(
    submissions: list[SupplierSubmission],
    api_key: str | None = None,
    db_path=database.DB_PATH,
) -> dict[str, Any]:
    """
    Run the full agentic pipeline for a batch of supplier submissions
    and persist the results under one RFP_RUN_ID.

    api_key is the OpenRouter API key (see tools/evaluation_agent.py),
    resolved by the caller (app.py) from Streamlit secrets/env — it is
    passed straight through to evaluate_supplier for each supplier.

    Returns a dict with:
        rfp_run_id, leaderboard (ranked results as dicts), any
        batch-level warnings (e.g. suppliers that failed extraction),
        and criteria_used.
    """
    # --- Step 1: load & validate active criteria ----------------------
    criteria = database.get_active_criteria(db_path)
    if not criteria:
        raise OrchestratorError("No active evaluation criteria found in the database.")
    if not database.validate_weights_sum_to_100(criteria):
        raise OrchestratorError(
            "Active criteria weights do not sum to 100%. Fix criteria before running a batch."
        )

    rfp_run_id = database.create_run(db_path)
    batch_warnings: list[str] = []

    validated_inputs: list[SupplierInput] = []
    warnings_by_supplier: dict[str, list[str]] = {}
    risks_by_supplier: dict[str, list[str]] = {}
    summary_by_supplier: dict[str, str] = {}

    for sub in submissions:
        supplier_warnings: list[str] = []

        # --- Step 2: Document Tool -------------------------------------
        try:
            document_text = extract_text_from_pdf(sub.pdf_bytes, filename=sub.filename)
        except DocumentExtractionError as e:
            batch_warnings.append(f"[{sub.supplier_name}] {e} — supplier skipped.")
            continue

        # --- Step 3: Evaluation Agent (LLM) -----------------------------
        try:
            raw_llm_output = evaluate_supplier(
                supplier_name=sub.supplier_name,
                document_text=document_text,
                criteria=criteria,
                api_key=api_key,
            )
        except Exception as e:  # noqa: BLE001 — LLM/network failures must not kill the batch
            supplier_warnings.append(f"LLM call failed ({e}); all criteria scored 0.")
            raw_llm_output = {"supplier_name": sub.supplier_name, "criteria": [], "risks": [], "overall_summary": ""}

        # --- Step 4: Validation Tool ------------------------------------
        outcome = validate_and_normalize(
            raw_llm_output=raw_llm_output,
            active_criteria=criteria,
            supplier_name_fallback=sub.supplier_name,
        )
        supplier_warnings.extend(outcome.warnings)

        warnings_by_supplier[sub.supplier_name] = supplier_warnings
        risks_by_supplier[sub.supplier_name] = outcome.result.risks
        summary_by_supplier[sub.supplier_name] = outcome.result.overall_summary

        validated_inputs.append(
            SupplierInput(
                supplier_name=sub.supplier_name,
                submission_date=sub.submission_date,
                experience_rating=sub.experience_rating,
                criteria_results=[c.model_dump() for c in outcome.result.criteria],
            )
        )

    if not validated_inputs:
        database.finalize_run(
            rfp_run_id,
            criteria_used=criteria,
            batch_warnings=batch_warnings,
            status="failed",
            db_path=db_path,
        )
        raise OrchestratorError(
            "No suppliers could be evaluated in this batch: " + "; ".join(batch_warnings)
        )

    # --- Step 5: Ranking Tool (deterministic) ---------------------------
    ranked = score_and_rank_batch(
        validated_suppliers=validated_inputs,
        criteria_meta=criteria,
        warnings_by_supplier=warnings_by_supplier,
        risks_by_supplier=risks_by_supplier,
        summary_by_supplier=summary_by_supplier,
    )

    # --- Step 6: Persist --------------------------------------------------
    for r in ranked:
        database.save_supplier_result(rfp_run_id, r.to_dict(), db_path)
    database.finalize_run(
        rfp_run_id,
        criteria_used=criteria,
        batch_warnings=batch_warnings,
        status="completed",
        db_path=db_path,
    )

    return {
        "rfp_run_id": rfp_run_id,
        "leaderboard": [r.to_dict() for r in ranked],
        "batch_warnings": batch_warnings,
        "criteria_used": criteria,
    }