import csv
import io
import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.preprocessor import clean_candidates
from src.scoring.composite import score_candidates
from src.scoring.reasoning import generate_reasoning
from src.utils.output import REQUIRED_HEADER, validate_submission_locally

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🎯", layout="wide")

st.title("🎯 Redrob Candidate Ranker — Sandbox")
st.caption(
    "Runs the exact ranking pipeline end-to-end on a small candidate sample "
    "and produces a ranked CSV in the official submission format — in seconds, "
    "CPU-only, no network calls."
)

with st.expander("How this works", expanded=False):
    st.markdown(
        """
        **Pipeline:** clean & normalize -> score 6 weighted components
        (career, skills, experience, behavioral, location, education) ->
        apply honeypot / behavioral / salary multipliers -> optional TF-IDF
        re-rank -> generate grounded reasoning -> write CSV
        (`candidate_id, rank, score, reasoning`).

        This is the **same code** (`src/`) used by `rank.py` to produce the
        full 100,000-candidate submission - just run here on a smaller sample
        so it completes instantly in the browser.
        """
    )

SAMPLE_PATH = Path(__file__).resolve().parent / "sample_candidates.json"

# ── Input source ────────────────────────────────────────────────────────
st.subheader("1. Provide candidates")

source = st.radio(
    "Choose input source",
    ["Use pre-loaded sample (50 candidates)", "Upload my own JSON"],
    horizontal=True,
)

raw = None
if source.startswith("Use pre-loaded"):
    if SAMPLE_PATH.exists():
        raw = json.loads(SAMPLE_PATH.read_text())
        st.success(f"Loaded {len(raw)} candidates from sample_candidates.json")
    else:
        st.error("sample_candidates.json not found in the app directory.")
else:
    uploaded = st.file_uploader(
        "Upload candidate JSON (array format, max 100 candidates)",
        type=["json"],
    )
    if uploaded:
        raw = json.load(uploaded)
        if not isinstance(raw, list):
            st.error("File must be a JSON array of candidate objects.")
            raw = None
        elif len(raw) > 100:
            st.warning(f"File has {len(raw)} candidates - only the first 100 will be used.")
            raw = raw[:100]
        else:
            st.success(f"Loaded {len(raw)} candidates.")

# ── Options ────────────────────────────────────────────────────────────
st.subheader("2. Run ranking")

col1, col2 = st.columns(2)
with col1:
    use_tfidf = st.checkbox("Enable TF-IDF re-ranking", value=True)
with col2:
    top_n = st.slider(
        "Top N to include in output CSV",
        min_value=5,
        max_value=100,
        value=min(50, len(raw) if raw else 50),
    )

run_clicked = st.button("Run Ranker", type="primary", disabled=(raw is None))
st.caption("If the results don't appear after the first click, click **Run Ranker** again.")

# Run pipeline (results stored in session_state so they survive reruns
# triggered by st.download_button or other widget interactions)
if run_clicked and raw is not None:
    t0 = time.time()
    with st.spinner("Running ranking pipeline..."):
        cleaned = clean_candidates(raw)
        scored = score_candidates(cleaned)

        if use_tfidf and len(scored) > 5:
            from src.retrieval.tfidf_reranker import tfidf_rerank
            scored = tfidf_rerank(scored, top_n=min(len(scored), 500))

        actual_n = min(top_n, len(scored))
        rows = []
        for rank, cs in enumerate(scored[:actual_n], start=1):
            reasoning = generate_reasoning(cs, rank)
            rows.append({
                "candidate_id": cs.candidate_id,
                "rank": rank,
                "score": f"{cs.composite:.6f}",
                "reasoning": reasoning,
            })

    elapsed = time.time() - t0

    # Build CSV in memory
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=REQUIRED_HEADER)
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = buf.getvalue().encode("utf-8")

    # Persist everything needed to render results across reruns
    st.session_state["result"] = {
        "scored": scored,
        "rows": rows,
        "csv_bytes": csv_bytes,
        "elapsed": elapsed,
        "actual_n": actual_n,
        "total_scored": len(scored),
    }

# ── Render results from session_state (if any) ──────────────────────────
result = st.session_state.get("result")

if result:
    scored = result["scored"]
    rows = result["rows"]
    csv_bytes = result["csv_bytes"]
    elapsed = result["elapsed"]
    actual_n = result["actual_n"]
    total_scored = result["total_scored"]

    st.success(
        f"Ranked {total_scored} candidates -> top {actual_n} written to CSV "
        f"in {elapsed:.2f} seconds (budget: 300s / 5 min)."
    )

    # Compute budget visual
    budget_pct = min(1.0, elapsed / 300.0)
    st.progress(budget_pct, text=f"{elapsed:.2f}s used of 300s (5 min) CPU budget")

    # Download button
    st.download_button(
        label="Download ranked CSV",
        data=csv_bytes,
        file_name="submission.csv",
        mime="text/csv",
        type="primary",
        key="download_csv_btn",
    )

    # Validation
    tmp_path = Path("/tmp/sandbox_submission.csv")
    tmp_path.write_bytes(csv_bytes)
    errors = validate_submission_locally(tmp_path)
    real_errors = [
        e for e in errors
        if "100 data rows" not in e and "Missing ranks" not in e
    ]
    if real_errors:
        st.warning("Format check found issues:\n" + "\n".join(f"- {e}" for e in real_errors))
    else:
        st.info("CSV format valid (header, candidate_id pattern, score monotonicity, no duplicates).")
        if actual_n != 100:
            st.caption(f"Note: this sample produced {actual_n} rows. The full 100K run produces exactly 100.")

    # Preview table
    st.subheader("3. Ranked output")
    st.dataframe(rows, use_container_width=True, height=400)

    # Detail cards for top candidates
    st.subheader("4. Top candidate detail")
    for i, cs in enumerate(scored[:min(10, actual_n)]):
        c = cs.candidate
        p = c["profile"]
        with st.expander(
            f"#{i + 1}  {cs.candidate_id}  -  "
            f"{p.get('current_title', '')} @ {p.get('current_company', '')}  "
            f"(score: {cs.composite:.1f})",
            expanded=(i < 2),
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Career", f"{cs.career_score:.0f}")
            c2.metric("Skills", f"{cs.skills_score:.0f}")
            c3.metric("Behavioral", f"{cs.behavioral_score:.0f}")
            c4.metric("Location", f"{cs.location_score:.0f}")

            st.write(
                f"**YoE:** {p.get('years_of_experience', 0):.1f}  |  "
                f"**Location:** {p.get('location', '')}, {p.get('country', '')}"
            )
            st.write(f"**Tier-1 skills:** {', '.join(cs.tier1_skills) or 'none'}")
            st.info(generate_reasoning(cs, i + 1))

            if cs.honeypot_flags:
                st.warning(f"Honeypot flags: {cs.honeypot_flags}")
else:
    st.info("Choose an input source, then click **Run Ranker** to produce the ranked CSV.")
