# Agentic RFP Evaluation

Full working app: SQLite persistence, the agentic tools layer, and a
Streamlit UI on top, matching every screen in the project brief
(section 9).

## Folder structure

```
rfp_tools/
├── app.py                    # Streamlit UI — all 5 required screens
├── generate_sample_pdfs.py   # Regenerates the 4 synthetic supplier PDFs
├── sample_data/               # The 4 synthetic supplier RFP PDFs (section 7)
│   ├── Apex_Systems_Proposal.pdf
│   ├── BrightPath_Tech_Proposal.pdf
│   ├── NexaWorks_Proposal.pdf
│   └── Orbit_Digital_Proposal.pdf
├── db/
│   └── database.py           # SQLite schema, seeding, persistence
├── tools/
│   ├── document_tool.py      # PDF -> clean text (PyMuPDF, pypdf fallback)
│   ├── evaluation_agent.py   # Prompt building + LLM call -> JSON
│   ├── validation_tool.py    # Pydantic schema check + normalization
│   ├── ranking_tool.py       # Deterministic formulas, benchmarks, PPI, tie-break rank
│   └── orchestrator.py       # Runs all tools in order for a batch
└── requirements.txt
```

## Sample data

`sample_data/` has four fictional, 2-page supplier proposals — Apex
Systems, BrightPath Tech, NexaWorks, and Orbit Digital — all
responding to the same made-up RFP (a cloud OMS for a mid-size
retailer, SAP-integrated, sized for 3x order-volume growth). Each was
written with deliberately different strengths so the leaderboard
produces a genuinely interesting spread:

| Supplier | Profile |
|---|---|
| Apex Systems | Strong technical design & security detail; highest price; 7-month schedule |
| BrightPath Tech | Lowest price, fastest timeline (10 weeks); thin security/compliance detail; limited relevant experience |
| NexaWorks | Most detailed implementation plan with named milestone owners; strongest support-transition model; mid-range price |
| Orbit Digital | Most experience and the most references; vague/unconfirmed integration specifics; mid-range price |

All four were extracted successfully through `tools/document_tool.py`
in testing. Regenerate them anytime with `python generate_sample_pdfs.py`.

## How the pieces map to the brief

| Brief component     | File                        | LLM involved? |
|----------------------|------------------------------|:---:|
| Orchestrator Agent    | `tools/orchestrator.py`      | No |
| Document Tool          | `tools/document_tool.py`      | No |
| Evaluation Agent        | `tools/evaluation_agent.py`    | **Yes** |
| Validation Tool          | `tools/validation_tool.py`      | No |
| Ranking Tool              | `tools/ranking_tool.py`          | No |

The LLM only ever judges *content* (via `evaluation_agent.py`). Every
number that ends up on the leaderboard — weighted score, benchmark,
gap, relative %, PPI, tie-break, rank — is computed by plain
deterministic Python in `ranking_tool.py`, so the same validated
inputs always reproduce the same leaderboard.

## Quick start (local)

```bash
cd rfp_tools
pip install -r requirements.txt
python db/database.py            # creates + seeds rfp_evaluation.db
```

**Set your API key via Streamlit secrets (never through the UI).**
This app calls Anthropic's Claude API for the Evaluation Agent.

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste in your real key
```

`.streamlit/secrets.toml` is gitignored, so it's never pushed to
GitHub. The app reads the key from there automatically — there's no
API key field in the UI, and the key is never typed into the browser,
stored in session state, or visible in a screen-recorded demo.

If your key is identity-linked (personal/service account) with access
to multiple workspaces, Anthropic also requires a workspace ID on
every request. Simplest fix: when creating the key in the Console
(Settings → API keys), scope it to a single specific workspace and
you can skip the workspace-ID secret entirely. Otherwise, uncomment
and set `ANTHROPIC_WORKSPACE_ID` in `secrets.toml`.

```bash
streamlit run app.py
```

The sidebar will show a green "API key loaded from secrets" message
once it's picked up correctly.

## Using the app

1. **Criteria tab** — review the 5 seeded default criteria (weights
   sum to 100% out of the box). Expand "Manage criteria" to edit
   weights, activate/deactivate, or add new criteria — no code
   changes needed, per the brief's requirement.
2. **Supplier Input tab** — upload the 4 PDFs from `sample_data/` (or
   your own), fill in name / submission date / historical experience
   rating for each, then click **Create batch & Evaluate**. This
   calls the full pipeline (extract → LLM score → validate → rank →
   persist) under one `RFP_RUN_ID`.
3. **Leaderboard tab** — ranked table + JSON download.
4. **Detailed Scorecard tab** — per-criterion score, benchmark, gap,
   relative %, evidence, and justification for any supplier in the
   selected run.
5. **Run Details tab** — the run ID, the tie-break rule that was
   applied, every validation/extraction warning, and a full JSON
   export of the run.

All five tabs share a run picker once more than one batch has been
run in the session, so you can compare/re-inspect older runs without
losing state.

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repo (root of the repo = this
   folder, so `app.py` is at the repo root). `.gitignore` already
   excludes `.streamlit/secrets.toml`, so your real key won't be
   pushed even by accident.
2. On [share.streamlit.io](https://share.streamlit.io), create a new
   app pointing at `app.py`.
3. Open the app's **Settings → Secrets** panel in the dashboard and
   paste in:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-real-key-here"
   ```
   This is Streamlit Cloud's own encrypted secrets store — it's
   equivalent to the local `secrets.toml` file, and `app.py` reads it
   the same way (`st.secrets["ANTHROPIC_API_KEY"]`) without any code
   changes.
4. Deploy, then submit the public app URL per the brief's submission
   requirements. Reviewers running your app will need their own key
   in their own secrets panel — the key never travels with the code.

## Programmatic use (without the UI)

```python
from tools.orchestrator import run_batch, SupplierSubmission
from datetime import date

submissions = [
    SupplierSubmission(
        supplier_name="Apex Systems",
        submission_date=date(2026, 7, 10),
        experience_rating=8.0,
        pdf_bytes=uploaded_file.read(),
        filename=uploaded_file.name,
    ),
    # ... one SupplierSubmission per uploaded PDF
]

result = run_batch(submissions)
print(result["rfp_run_id"])
for row in result["leaderboard"]:
    print(row["final_rank"], row["supplier_name"], row["ppi"])
```

## What's already tested

Because this sandbox has no network access and a couple of packages
(`pydantic`, `pymupdf`) aren't preinstalled, I could only exercise the
parts that run on the standard library:

- **`db/database.py`** — full round trip: init schema, seed criteria,
  verify weights sum to 100%, create a run, persist a result, read it
  back. ✅ Passed.
- **`tools/ranking_tool.py`** — 4-supplier scenario including a
  deliberate PPI tie, verifying: absolute weighted score, benchmark =
  max per criterion, gap, relative %, PPI as weighted average, and the
  full 4-level tie-break cascade (PPI → date → experience rating →
  name). ✅ Passed.
- All six modules pass `python -m py_compile` (no syntax/import
  errors at parse time).

**Not exercised in this sandbox** (needs `pip install pydantic
pymupdf streamlit anthropic` and a real `ANTHROPIC_API_KEY`, none
available here — no network access): `document_tool.py`'s PyMuPDF
path (pypdf fallback path is fine), `validation_tool.py`'s Pydantic
models, `evaluation_agent.py`'s actual API call, and `app.py` at
runtime (only syntax-checked with `py_compile`, not run in a live
Streamlit server). Install the requirements and set your API key,
then run `streamlit run app.py` against your four synthetic supplier
PDFs before recording the demo.

## What's still missing to finish the project

This delivers the full working app (tools layer + Streamlit UI +
sample data). You still need to, per the brief:
- Deploy to Streamlit Community Cloud and grab the public URL.
- Take the required screenshots and record the demo video (one
  successful run + at least one validation/error case — e.g. upload
  a non-PDF or a proposal missing a whole section to trigger the
  Validation Tool's "missing criterion" repair path).
- Export a sample completed run's JSON (the Run Details tab's
  download button does this for you) and include it in your
  submission.
