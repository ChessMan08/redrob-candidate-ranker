import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.preprocessor import clean_candidates
from src.scoring.composite import score_candidates
from src.scoring.reasoning import generate_reasoning

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🎯", layout="wide")

st.title("🎯 Redrob Candidate Ranker — Sandbox")
st.caption(
    "Upload a JSON array of candidate profiles (e.g. sample_candidates.json) "
    "and see them scored & ranked by the structured multi-signal scorer."
)

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown()

uploaded = st.file_uploader(
    "Upload candidate JSON (array format, e.g. sample_candidates.json)",
    type=["json"],
)

col1, col2 = st.columns(2)
with col1:
    top_n = st.slider("Show top N candidates", min_value=5, max_value=50, value=10)
with col2:
    use_tfidf = st.checkbox("Enable TF-IDF re-ranking", value=True)

if uploaded and st.button("Rank Candidates", type="primary"):
    with st.spinner("Loading and scoring..."):
        try:
            raw = json.load(uploaded)
            if not isinstance(raw, list):
                st.error("File must be a JSON array of candidate objects.")
            else:
                cleaned = clean_candidates(raw)
                scored = score_candidates(cleaned)

                if use_tfidf and len(scored) > 5:
                    from src.retrieval.tfidf_reranker import tfidf_rerank
                    scored = tfidf_rerank(scored, top_n=min(len(scored), 100))

                st.success(f"Scored {len(scored)} candidates. Showing top {top_n}.")

                for i, cs in enumerate(scored[:top_n]):
                    c = cs.candidate
                    p = c["profile"]
                    reason = generate_reasoning(cs, i + 1)

                    with st.expander(
                        f"#{i + 1}  {cs.candidate_id}  —  "
                        f"{p.get('current_title', '')} @ {p.get('current_company', '')}  "
                        f"(score: {cs.composite:.1f})",
                        expanded=(i < 3),
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
                        st.write(
                            f"**Tier-1 skills:** {', '.join(cs.tier1_skills) or 'none'}"
                        )
                        st.info(reason)

                        if cs.honeypot_flags:
                            st.warning(f"Honeypot flags: {cs.honeypot_flags}")

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

else:
    st.info("👆 Upload a candidate JSON file and click **Rank Candidates** to begin.")
