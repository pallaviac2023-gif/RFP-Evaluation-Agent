"""
tools/ranking_tool.py
------------------------
Ranking Tool: performs all formulas, peer comparison, tie-breaks, and
final ranking. DETERMINISTIC PYTHON ONLY — no LLM calls here.

Responsibility (per project brief):
    "Performs formulas, peer comparison, tie-breaks, and final
     ranking." -> Deterministic Python only

Formulas implemented (see brief section 5):
    absolute weighted score = sum((criterion_score / max_score) * weight)
    criterion benchmark      = highest valid score observed for that
                                criterion across all suppliers
    criterion gap            = supplier_score - benchmark_score
    relative performance %   = (supplier_score / benchmark_score) * 100
                                (0 if benchmark_score == 0, to avoid /0)
    PPI                      = weighted average of criterion relative%
    tie-break order          = 1) higher PPI
                                2) earlier submission_date
                                3) higher experience_rating
                                4) supplier_name ascending
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date


# --------------------------------------------------------------------------
# Input container: one validated evaluation per supplier
# --------------------------------------------------------------------------
@dataclass
class SupplierInput:
    supplier_name: str
    submission_date: date
    experience_rating: float
    criteria_results: list[dict]  # [{criterion_id, score, max_score}, ...]


@dataclass
class RankedResult:
    supplier_name: str
    submission_date: str
    experience_rating: float
    absolute_score: float
    ppi: float
    final_rank: int | None
    criterion_details: list[dict] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    overall_summary: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "supplier_name": self.supplier_name,
            "submission_date": self.submission_date,
            "experience_rating": self.experience_rating,
            "absolute_score": round(self.absolute_score, 4),
            "ppi": round(self.ppi, 4),
            "final_rank": self.final_rank,
            "criterion_details": self.criterion_details,
            "risks": self.risks,
            "overall_summary": self.overall_summary,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------
# Step 1: absolute weighted score per supplier
# --------------------------------------------------------------------------
def compute_absolute_score(criteria_results: list[dict], criteria_meta: list[dict]) -> float:
    """absolute_score = sum((score / max_score) * weight) across all
    active criteria, expressed on a 0-100 scale (weights sum to 1.0)."""
    weight_by_id = {c["criterion_id"]: c["weight"] for c in criteria_meta}
    total = 0.0
    for cr in criteria_results:
        weight = weight_by_id.get(cr["criterion_id"], 0)
        max_score = cr["max_score"] or 1
        total += (cr["score"] / max_score) * weight
    return total * 100  # 0-100 scale


# --------------------------------------------------------------------------
# Step 2: peer benchmarks across all suppliers in the batch
# --------------------------------------------------------------------------
def compute_benchmarks(all_suppliers_criteria: list[list[dict]]) -> dict[int, float]:
    """For each criterion_id, find the highest score observed for that
    criterion across every supplier in this run."""
    benchmarks: dict[int, float] = {}
    for supplier_criteria in all_suppliers_criteria:
        for cr in supplier_criteria:
            cid = cr["criterion_id"]
            benchmarks[cid] = max(benchmarks.get(cid, 0), cr["score"])
    return benchmarks


def compute_criterion_details(
    criteria_results: list[dict], benchmarks: dict[int, float], criteria_meta: list[dict]
) -> list[dict]:
    """Attach benchmark, gap, and relative-% to every criterion result
    for one supplier, plus the criterion name/weight for display."""
    meta_by_id = {c["criterion_id"]: c for c in criteria_meta}
    details = []
    for cr in criteria_results:
        cid = cr["criterion_id"]
        benchmark = benchmarks.get(cid, 0)
        gap = cr["score"] - benchmark
        relative_pct = (cr["score"] / benchmark) * 100 if benchmark > 0 else 0.0

        details.append(
            {
                "criterion_id": cid,
                "criterion_name": meta_by_id.get(cid, {}).get("name", f"Criterion {cid}"),
                "weight": meta_by_id.get(cid, {}).get("weight", 0),
                "score": cr["score"],
                "max_score": cr["max_score"],
                "benchmark": benchmark,
                "gap": round(gap, 4),
                "relative_pct": round(relative_pct, 4),
                "justification": cr.get("justification", ""),
                "evidence": cr.get("evidence", ""),
            }
        )
    return details


# --------------------------------------------------------------------------
# Step 3: Peer Performance Index (PPI)
# --------------------------------------------------------------------------
def compute_ppi(criterion_details: list[dict]) -> float:
    """PPI = weighted average of each criterion's relative performance %,
    using the same weights as the absolute score."""
    total_weight = sum(d["weight"] for d in criterion_details) or 1
    weighted_sum = sum(d["relative_pct"] * d["weight"] for d in criterion_details)
    return weighted_sum / total_weight


# --------------------------------------------------------------------------
# Step 4: deterministic tie-break ranking
# --------------------------------------------------------------------------
def rank_suppliers(results: list[RankedResult]) -> list[RankedResult]:
    """
    Apply the mandatory tie-break order and assign sequential ranks:
      1) Higher PPI first
      2) Earlier submission date
      3) Higher historical experience rating
      4) Supplier name ascending (alphabetical)
    Stable sort, then rank = position + 1.
    """

    def sort_key(r: RankedResult):
        return (
            -r.ppi,                                  # higher PPI first
            r.submission_date,                        # earlier date first (ISO string sorts correctly)
            -r.experience_rating,                      # higher experience first
            r.supplier_name.lower(),                    # name ascending
        )

    ordered = sorted(results, key=sort_key)
    for idx, r in enumerate(ordered, start=1):
        r.final_rank = idx
    return ordered


# --------------------------------------------------------------------------
# Orchestration entry point used by orchestrator.py
# --------------------------------------------------------------------------
def score_and_rank_batch(
    validated_suppliers: list[SupplierInput],
    criteria_meta: list[dict],
    warnings_by_supplier: dict[str, list[str]] | None = None,
    risks_by_supplier: dict[str, list[str]] | None = None,
    summary_by_supplier: dict[str, str] | None = None,
) -> list[RankedResult]:
    """
    Full deterministic pipeline for one batch/run:
      compute absolute scores -> benchmarks -> gaps/relative% -> PPI ->
      tie-break rank. Returns suppliers in final rank order.
    """
    warnings_by_supplier = warnings_by_supplier or {}
    risks_by_supplier = risks_by_supplier or {}
    summary_by_supplier = summary_by_supplier or {}

    benchmarks = compute_benchmarks([s.criteria_results for s in validated_suppliers])

    results: list[RankedResult] = []
    for supplier in validated_suppliers:
        absolute_score = compute_absolute_score(supplier.criteria_results, criteria_meta)
        criterion_details = compute_criterion_details(
            supplier.criteria_results, benchmarks, criteria_meta
        )
        ppi = compute_ppi(criterion_details)

        results.append(
            RankedResult(
                supplier_name=supplier.supplier_name,
                submission_date=supplier.submission_date.isoformat(),
                experience_rating=supplier.experience_rating,
                absolute_score=absolute_score,
                ppi=ppi,
                final_rank=None,
                criterion_details=criterion_details,
                risks=risks_by_supplier.get(supplier.supplier_name, []),
                overall_summary=summary_by_supplier.get(supplier.supplier_name, ""),
                warnings=warnings_by_supplier.get(supplier.supplier_name, []),
            )
        )

    return rank_suppliers(results)
