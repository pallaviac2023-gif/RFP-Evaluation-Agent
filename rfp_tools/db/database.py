"""
db/database.py
---------------
SQLite schema creation, seeding, and persistence helpers for the
Agentic RFP Evaluation project.

Tables
------
evaluation_criteria : configurable scoring criteria + weights
rfp_runs            : one row per evaluation batch, including a
                       snapshot of the criteria used and any batch-
                       level warnings, so a completed run can be
                       reloaded from the database exactly as it was
                       (see get_full_run_result / "Previous Runs" in
                       app.py) without depending on session state.
supplier_results    : one row per supplier per run, including the
                       full validated JSON result for traceability
"""

import sqlite3
import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "rfp_evaluation.db"


# --------------------------------------------------------------------------
# Connection helper
# --------------------------------------------------------------------------
@contextmanager
def get_connection(db_path: str | Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Schema creation
# --------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_criteria (
    criterion_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    description     TEXT,
    weight          REAL NOT NULL,      -- fraction of 100, e.g. 0.30
    max_score       REAL NOT NULL DEFAULT 10,
    is_active       INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS rfp_runs (
    rfp_run_id      TEXT PRIMARY KEY,   -- UUID string
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'in_progress',  -- in_progress|completed|failed
    criteria_json   TEXT,               -- snapshot of criteria used in this run
    batch_warnings_json TEXT            -- JSON list of batch-level warnings
);

CREATE TABLE IF NOT EXISTS supplier_results (
    result_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    rfp_run_id           TEXT NOT NULL,
    supplier_name         TEXT NOT NULL,
    submission_date       TEXT NOT NULL,
    experience_rating     REAL NOT NULL,
    absolute_score        REAL,
    ppi                    REAL,
    final_rank             INTEGER,
    result_json            TEXT NOT NULL,  -- full traceable payload
    FOREIGN KEY (rfp_run_id) REFERENCES rfp_runs (rfp_run_id)
);
"""


def _migrate_schema(conn: sqlite3.Connection):
    """Add columns introduced after the initial release to any
    pre-existing database file, so upgrading doesn't require deleting
    rfp_evaluation.db. Safe to call on every startup."""
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(rfp_runs)")}
    if "criteria_json" not in existing_cols:
        conn.execute("ALTER TABLE rfp_runs ADD COLUMN criteria_json TEXT")
    if "batch_warnings_json" not in existing_cols:
        conn.execute("ALTER TABLE rfp_runs ADD COLUMN batch_warnings_json TEXT")


def init_db(db_path: str | Path = DB_PATH):
    """Create tables if they do not already exist, and migrate older
    database files to the current schema."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
        _migrate_schema(conn)


def seed_default_criteria(db_path: str | Path = DB_PATH):
    """Seed the default criteria set from the project brief (idempotent)."""
    default_criteria = [
        ("Technical Capability", "Architecture, integrations, scalability, technical fit", 0.30, 10, 1),
        ("Implementation Plan", "Timeline, milestones, staffing, risk plan", 0.20, 10, 1),
        ("Commercial Value", "Pricing clarity, total cost, assumptions", 0.20, 10, 1),
        ("Security & Compliance", "Controls, certifications, privacy, auditability", 0.20, 10, 1),
        ("Support & Experience", "Support model, similar projects, references", 0.10, 10, 1),
    ]
    with get_connection(db_path) as conn:
        existing = conn.execute("SELECT COUNT(*) AS c FROM evaluation_criteria").fetchone()["c"]
        if existing == 0:
            conn.executemany(
                """INSERT INTO evaluation_criteria
                   (name, description, weight, max_score, is_active)
                   VALUES (?, ?, ?, ?, ?)""",
                default_criteria,
            )


# --------------------------------------------------------------------------
# Criteria access
# --------------------------------------------------------------------------
def get_active_criteria(db_path: str | Path = DB_PATH) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM evaluation_criteria WHERE is_active = 1 ORDER BY criterion_id"
        ).fetchall()
        return [dict(r) for r in rows]


def validate_weights_sum_to_100(criteria: list[dict], tolerance: float = 0.01) -> bool:
    total = sum(c["weight"] for c in criteria)
    return abs(total - 1.0) <= tolerance


def get_all_criteria(db_path: str | Path = DB_PATH) -> list[dict]:
    """Return every criterion (active and inactive) for the management UI."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM evaluation_criteria ORDER BY criterion_id"
        ).fetchall()
        return [dict(r) for r in rows]


def update_criterion(
    criterion_id: int,
    name: str,
    description: str,
    weight: float,
    max_score: float,
    is_active: bool,
    db_path: str | Path = DB_PATH,
):
    """Update an existing criterion's fields (used by the Streamlit
    criteria-management screen so weights/activation can change without
    touching prompt code, per the brief)."""
    with get_connection(db_path) as conn:
        conn.execute(
            """UPDATE evaluation_criteria
               SET name = ?, description = ?, weight = ?, max_score = ?, is_active = ?
               WHERE criterion_id = ?""",
            (name, description, weight, max_score, int(is_active), criterion_id),
        )


def add_criterion(
    name: str,
    description: str,
    weight: float,
    max_score: float = 10,
    is_active: bool = True,
    db_path: str | Path = DB_PATH,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO evaluation_criteria (name, description, weight, max_score, is_active)
               VALUES (?, ?, ?, ?, ?)""",
            (name, description, weight, max_score, int(is_active)),
        )
        return cur.lastrowid


def list_runs(db_path: str | Path = DB_PATH) -> list[dict]:
    """Return all runs, most recent first, for the run-picker in the UI."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM rfp_runs ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def list_completed_runs(db_path: str | Path = DB_PATH) -> list[dict]:
    """Return only completed runs, most recent first. Used by the
    Previous Runs picker so in-progress/failed batches aren't offered
    as if they had results to show."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM rfp_runs WHERE status = 'completed' ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# Run + result persistence
# --------------------------------------------------------------------------
def create_run(db_path: str | Path = DB_PATH) -> str:
    """Create a new RFP run and return its ID. One ID is shared by every
    supplier evaluated in the same batch, per the brief's requirement."""
    rfp_run_id = str(uuid.uuid4())
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO rfp_runs (rfp_run_id, created_at, status) VALUES (?, ?, ?)",
            (rfp_run_id, datetime.now(timezone.utc).isoformat(), "in_progress"),
        )
    return rfp_run_id


def save_supplier_result(rfp_run_id: str, ranked_result: dict, db_path: str | Path = DB_PATH):
    """Persist one supplier's complete, traceable result."""
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT INTO supplier_results
               (rfp_run_id, supplier_name, submission_date, experience_rating,
                absolute_score, ppi, final_rank, result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rfp_run_id,
                ranked_result["supplier_name"],
                ranked_result["submission_date"],
                ranked_result["experience_rating"],
                ranked_result["absolute_score"],
                ranked_result["ppi"],
                ranked_result["final_rank"],
                json.dumps(ranked_result),
            ),
        )


def mark_run_status(rfp_run_id: str, status: str, db_path: str | Path = DB_PATH):
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE rfp_runs SET status = ? WHERE rfp_run_id = ?", (status, rfp_run_id)
        )


def finalize_run(
    rfp_run_id: str,
    criteria_used: list[dict],
    batch_warnings: list[str],
    status: str = "completed",
    db_path: str | Path = DB_PATH,
):
    """Snapshot the criteria used and any batch-level warnings onto the
    run row, and mark it complete (or failed). Call this once, at the
    end of a batch, after all supplier_results rows have been written —
    it's what makes get_full_run_result() able to reconstruct the run
    later without session state."""
    with get_connection(db_path) as conn:
        conn.execute(
            """UPDATE rfp_runs
               SET status = ?, criteria_json = ?, batch_warnings_json = ?
               WHERE rfp_run_id = ?""",
            (status, json.dumps(criteria_used), json.dumps(batch_warnings), rfp_run_id),
        )


def get_run_results(rfp_run_id: str, db_path: str | Path = DB_PATH) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM supplier_results WHERE rfp_run_id = ? ORDER BY final_rank",
            (rfp_run_id,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["result_json"] = json.loads(d["result_json"])
            results.append(d)
        return results


def get_full_run_result(rfp_run_id: str, db_path: str | Path = DB_PATH) -> dict | None:
    """
    Reconstruct the same shape of dict that run_batch() returns in
    memory — {rfp_run_id, leaderboard, criteria_used, batch_warnings} —
    entirely from SQLite. This is what lets app.py's Leaderboard,
    Detailed Scorecard, and Run Details tabs render a past run exactly
    like a freshly-completed one, after a restart or in a new session.

    Returns None if the run_id doesn't exist.
    """
    with get_connection(db_path) as conn:
        run_row = conn.execute(
            "SELECT * FROM rfp_runs WHERE rfp_run_id = ?", (rfp_run_id,)
        ).fetchone()
        if run_row is None:
            return None
        run = dict(run_row)

    supplier_rows = get_run_results(rfp_run_id, db_path)
    leaderboard = [row["result_json"] for row in supplier_rows]

    return {
        "rfp_run_id": run["rfp_run_id"],
        "created_at": run["created_at"],
        "status": run["status"],
        "leaderboard": leaderboard,
        "criteria_used": json.loads(run["criteria_json"]) if run["criteria_json"] else [],
        "batch_warnings": json.loads(run["batch_warnings_json"]) if run["batch_warnings_json"] else [],
    }


if __name__ == "__main__":
    init_db()
    seed_default_criteria()
    print(f"Database ready at {DB_PATH}")