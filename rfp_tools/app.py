"""
app.py
--------
Streamlit UI for the Agentic RFP Evaluation project.

Implements the five required screens (brief section 9):
    1. Criteria        - active criteria, weights, max score (+ management)
    2. Supplier input   - multi-PDF upload, metadata, validation, Evaluate
    3. Leaderboard        - rank, supplier, absolute score, PPI, date, experience
    4. Detailed scorecard  - per-criterion score/benchmark/gap/relative%/evidence
    5. Run details          - RFP_RUN_ID, warnings, tie-break explanation, JSON download

This file only handles presentation and input collection. All actual
work (extraction, LLM scoring, validation, ranking, persistence) is
delegated to tools/orchestrator.py and db/database.py.
"""

from __future__ import annotations
import json
import os
from datetime import date

import streamlit as st

from db import database
from tools.orchestrator import run_batch, SupplierSubmission, OrchestratorError

st.set_page_config(page_title="Agentic RFP Evaluation", layout="wide")


def _resolve_secret(name: str) -> str | None:
    """Resolve a named secret from Streamlit secrets first, then the
    environment."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass  # no secrets.toml present locally — fall through to env var
    return os.environ.get(name)


def _resolve_api_key() -> str | None:
    """
    Resolve the Gemini API key from secrets/environment only — it is
    never collected through the UI, so it's never visible in the
    browser, in session state, or in a screen-recorded demo.

    Resolution order:
      1. Streamlit secrets: .streamlit/secrets.toml locally, or the
         "Secrets" panel in Streamlit Community Cloud's app settings.
      2. GEMINI_API_KEY environment variable.
    """
    return _resolve_secret("GEMINI_API_KEY")


API_KEY = _resolve_api_key()

# --------------------------------------------------------------------------
# Session state init
# --------------------------------------------------------------------------
st.session_state.setdefault("run_history", {})   # rfp_run_id -> full result dict
st.session_state.setdefault("current_run_id", None)


def _ensure_db_ready():
    database.init_db()
    database.seed_default_criteria()


_ensure_db_ready()


# --------------------------------------------------------------------------
# Sidebar: setup
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Setup")

    if API_KEY:
        st.success("Gemini API key loaded from secrets.")
    else:
        st.error(
            "No Gemini API key found. Add `GEMINI_API_KEY` to "
            "`.streamlit/secrets.toml` locally, or to your app's Secrets "
            "panel on Streamlit Community Cloud, then restart the app."
        )

    st.caption(f"Database: `{database.DB_PATH.name}`")

    if st.button("Reset database (re-seed default criteria)"):
        if database.DB_PATH.exists():
            os.remove(database.DB_PATH)
        _ensure_db_ready()
        st.session_state["run_history"] = {}
        st.session_state["current_run_id"] = None
        st.success("Database reset and re-seeded.")
        st.rerun()

    st.divider()
    past_runs = database.list_runs()
    if past_runs:
        st.caption(f"{len(past_runs)} run(s) stored in this database.")


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_criteria, tab_input, tab_leaderboard, tab_scorecard, tab_run = st.tabs(
    ["📋 Criteria", "📤 Supplier Input", "🏆 Leaderboard", "🔍 Detailed Scorecard", "🧾 Run Details"]
)

# ==========================================================================
# TAB 1 — Criteria
# ==========================================================================
with tab_criteria:
    st.header("Evaluation Criteria")
    st.caption(
        "Loaded from SQLite. Only ACTIVE criteria with weights summing to "
        "100% are used to evaluate suppliers."
    )

    all_criteria = database.get_all_criteria()
    active_criteria = [c for c in all_criteria if c["is_active"]]
    weights_ok = database.validate_weights_sum_to_100(active_criteria) if active_criteria else False

    active_total = sum(c["weight"] for c in active_criteria) * 100
    if weights_ok:
        st.success(f"Active weights sum to {active_total:.1f}% ✅")
    else:
        st.error(f"Active weights sum to {active_total:.1f}% — must total 100% before running a batch.")

    st.subheader("Active criteria (used for scoring)")
    if active_criteria:
        st.dataframe(
            [
                {
                    "Criterion": c["name"],
                    "Weight": f"{c['weight'] * 100:.0f}%",
                    "Max Score": c["max_score"],
                    "Inspects": c["description"],
                }
                for c in active_criteria
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No active criteria yet.")

    with st.expander("Manage criteria (activate/deactivate, edit weights)"):
        st.caption(
            "Edit weights as decimals (e.g. 0.30 = 30%). Weight for INACTIVE "
            "criteria does not need to fit the 100% total."
        )
        for c in all_criteria:
            cols = st.columns([3, 1, 1, 1, 4])
            with cols[0]:
                new_name = st.text_input("Name", value=c["name"], key=f"name_{c['criterion_id']}")
            with cols[1]:
                new_weight = st.number_input(
                    "Weight", value=float(c["weight"]), min_value=0.0, max_value=1.0,
                    step=0.05, key=f"weight_{c['criterion_id']}"
                )
            with cols[2]:
                new_max = st.number_input(
                    "Max score", value=float(c["max_score"]), min_value=1.0,
                    key=f"max_{c['criterion_id']}"
                )
            with cols[3]:
                new_active = st.checkbox("Active", value=bool(c["is_active"]), key=f"active_{c['criterion_id']}")
            with cols[4]:
                new_desc = st.text_input("Inspects", value=c["description"], key=f"desc_{c['criterion_id']}")

            if st.button("Save", key=f"save_{c['criterion_id']}"):
                database.update_criterion(
                    c["criterion_id"], new_name, new_desc, new_weight, new_max, new_active
                )
                st.success(f"Saved '{new_name}'.")
                st.rerun()

        st.divider()
        st.subheader("Add a new criterion")
        with st.form("add_criterion_form", clear_on_submit=True):
            ac_cols = st.columns([3, 1, 1, 1])
            ac_name = ac_cols[0].text_input("Name")
            ac_weight = ac_cols[1].number_input("Weight", min_value=0.0, max_value=1.0, step=0.05, value=0.10)
            ac_max = ac_cols[2].number_input("Max score", min_value=1.0, value=10.0)
            ac_active = ac_cols[3].checkbox("Active", value=True)
            ac_desc = st.text_area("What should the LLM inspect for this criterion?")
            if st.form_submit_button("Add criterion"):
                if not ac_name.strip():
                    st.error("Name is required.")
                else:
                    database.add_criterion(ac_name, ac_desc, ac_weight, ac_max, ac_active)
                    st.success(f"Added '{ac_name}'.")
                    st.rerun()


# ==========================================================================
# TAB 2 — Supplier Input / Evaluate
# ==========================================================================
with tab_input:
    st.header("Upload Supplier Proposals")
    st.caption(
        "Upload one or more supplier RFP PDFs, enter metadata for each, "
        "then create a batch to evaluate them all under one run."
    )

    uploaded_files = st.file_uploader(
        "Supplier RFP PDFs", type=["pdf"], accept_multiple_files=True
    )

    submissions_ready: list[SupplierSubmission] = []
    input_errors: list[str] = []

    if uploaded_files:
        st.subheader("Supplier metadata")
        for i, f in enumerate(uploaded_files):
            with st.container(border=True):
                st.markdown(f"**{f.name}**")
                cols = st.columns(3)
                supplier_name = cols[0].text_input(
                    "Supplier name", value=f.name.rsplit(".", 1)[0], key=f"supplier_name_{i}"
                )
                submission_date = cols[1].date_input(
                    "Submission date", value=date.today(), key=f"submission_date_{i}"
                )
                experience_rating = cols[2].number_input(
                    "Historical experience rating (0-10)", min_value=0.0, max_value=10.0,
                    value=5.0, step=0.5, key=f"experience_rating_{i}"
                )

                if not supplier_name.strip():
                    input_errors.append(f"Row {i + 1}: supplier name is required.")
                    continue

                submissions_ready.append(
                    SupplierSubmission(
                        supplier_name=supplier_name.strip(),
                        submission_date=submission_date,
                        experience_rating=experience_rating,
                        pdf_bytes=f.getvalue(),
                        filename=f.name,
                    )
                )

        # duplicate name check
        names = [s.supplier_name for s in submissions_ready]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            input_errors.append(f"Duplicate supplier names in this batch: {', '.join(dupes)}")

    for err in input_errors:
        st.warning(err)

    st.divider()

    can_run = bool(submissions_ready) and not input_errors and weights_ok and bool(API_KEY)
    if not weights_ok:
        st.error("Cannot run: active criteria weights don't sum to 100%. Fix this on the Criteria tab.")
    if not API_KEY:
        st.error("Cannot run: no Gemini API key configured (see the sidebar).")

    if st.button("▶️ Create batch & Evaluate", type="primary", disabled=not can_run):
        with st.spinner(f"Evaluating {len(submissions_ready)} supplier(s)... this calls the LLM for each one."):
            try:
                result = run_batch(submissions_ready, api_key=API_KEY)
            except OrchestratorError as e:
                st.error(f"Batch failed: {e}")
            else:
                st.session_state["run_history"][result["rfp_run_id"]] = result
                st.session_state["current_run_id"] = result["rfp_run_id"]
                st.success(
                    f"Run {result['rfp_run_id']} complete — "
                    f"{len(result['leaderboard'])} supplier(s) ranked."
                )
                if result["batch_warnings"]:
                    for w in result["batch_warnings"]:
                        st.warning(w)
                st.info("See the Leaderboard, Detailed Scorecard, and Run Details tabs for results.")


# --------------------------------------------------------------------------
# Shared run picker for the remaining tabs
# --------------------------------------------------------------------------
def _get_selected_run(key_suffix: str) -> dict | None:
    run_ids = list(st.session_state["run_history"].keys())
    if not run_ids:
        return None
    default_idx = (
        run_ids.index(st.session_state["current_run_id"])
        if st.session_state["current_run_id"] in run_ids
        else 0
    )
    selected = st.selectbox(
        "Run (RFP_RUN_ID)", run_ids, index=default_idx, key=f"run_picker_{key_suffix}"
    )
    st.session_state["current_run_id"] = selected
    return st.session_state["run_history"][selected]


# ==========================================================================
# TAB 3 — Leaderboard
# ==========================================================================
with tab_leaderboard:
    st.header("Leaderboard")
    result = _get_selected_run("leaderboard")
    if not result:
        st.info("No completed runs yet. Create a batch on the Supplier Input tab.")
    else:
        leaderboard = result["leaderboard"]
        st.dataframe(
            [
                {
                    "Rank": r["final_rank"],
                    "Supplier": r["supplier_name"],
                    "Absolute Score": round(r["absolute_score"], 2),
                    "PPI": round(r["ppi"], 2),
                    "Submission Date": r["submission_date"],
                    "Experience Rating": r["experience_rating"],
                }
                for r in leaderboard
            ],
            use_container_width=True,
            hide_index=True,
        )

        winner = leaderboard[0]
        st.success(f"🏆 Top-ranked supplier: **{winner['supplier_name']}** (PPI {winner['ppi']:.1f})")

        st.download_button(
            "⬇️ Download leaderboard JSON",
            data=json.dumps(leaderboard, indent=2),
            file_name=f"leaderboard_{result['rfp_run_id']}.json",
            mime="application/json",
        )


# ==========================================================================
# TAB 4 — Detailed Scorecard
# ==========================================================================
with tab_scorecard:
    st.header("Detailed Scorecard")
    result = _get_selected_run("scorecard")
    if not result:
        st.info("No completed runs yet. Create a batch on the Supplier Input tab.")
    else:
        leaderboard = result["leaderboard"]
        supplier_names = [r["supplier_name"] for r in leaderboard]
        chosen = st.selectbox("Supplier", supplier_names)
        supplier_row = next(r for r in leaderboard if r["supplier_name"] == chosen)

        c1, c2, c3 = st.columns(3)
        c1.metric("Final Rank", supplier_row["final_rank"])
        c2.metric("Absolute Score", f"{supplier_row['absolute_score']:.2f}")
        c3.metric("PPI", f"{supplier_row['ppi']:.2f}")

        st.subheader("Criterion breakdown")
        st.dataframe(
            [
                {
                    "Criterion": d["criterion_name"],
                    "Weight": f"{d['weight'] * 100:.0f}%",
                    "Score": f"{d['score']}/{d['max_score']}",
                    "Benchmark": d["benchmark"],
                    "Gap": d["gap"],
                    "Relative %": f"{d['relative_pct']:.1f}%",
                }
                for d in supplier_row["criterion_details"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Evidence & justification")
        for d in supplier_row["criterion_details"]:
            with st.expander(f"{d['criterion_name']} — score {d['score']}/{d['max_score']}"):
                st.markdown(f"**Justification:** {d['justification'] or '_none provided_'}")
                st.markdown(f"**Evidence:** {d['evidence'] or '_none provided_'}")

        if supplier_row["risks"]:
            st.subheader("Risks")
            for risk in supplier_row["risks"]:
                st.markdown(f"- {risk}")

        if supplier_row["overall_summary"]:
            st.subheader("Overall summary")
            st.write(supplier_row["overall_summary"])

        if supplier_row["warnings"]:
            st.subheader("Validation warnings for this supplier")
            for w in supplier_row["warnings"]:
                st.warning(w)


# ==========================================================================
# TAB 5 — Run Details
# ==========================================================================
with tab_run:
    st.header("Run Details")
    result = _get_selected_run("rundetails")
    if not result:
        st.info("No completed runs yet. Create a batch on the Supplier Input tab.")
    else:
        st.subheader("RFP_RUN_ID")
        st.code(result["rfp_run_id"])

        st.subheader("Criteria used in this run")
        st.dataframe(
            [
                {"Criterion": c["name"], "Weight": f"{c['weight'] * 100:.0f}%", "Max Score": c["max_score"]}
                for c in result["criteria_used"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Tie-break rule applied")
        st.markdown(
            "Suppliers are sorted by this rule, in order, until every tie is "
            "broken; ranks are then assigned 1, 2, 3...\n\n"
            "1. **Higher PPI** first\n"
            "2. **Earlier submission date**\n"
            "3. **Higher historical experience rating**\n"
            "4. **Supplier name**, ascending alphabetically"
        )

        st.subheader("Batch-level warnings")
        if result["batch_warnings"]:
            for w in result["batch_warnings"]:
                st.warning(w)
        else:
            st.success("No batch-level issues (e.g. failed PDF extraction).")

        st.subheader("Per-supplier validation warnings")
        any_warnings = False
        for r in result["leaderboard"]:
            if r["warnings"]:
                any_warnings = True
                st.markdown(f"**{r['supplier_name']}**")
                for w in r["warnings"]:
                    st.warning(w)
        if not any_warnings:
            st.success("No per-supplier validation warnings — every LLM response was well-formed.")

        st.subheader("Full run export")
        st.download_button(
            "⬇️ Download complete run JSON",
            data=json.dumps(result, indent=2),
            file_name=f"run_{result['rfp_run_id']}.json",
            mime="application/json",
        )
