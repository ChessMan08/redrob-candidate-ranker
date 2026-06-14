"""
tfidf_reranker.py — TF-IDF re-ranking layer over the structured scorer.

Strategy:
  1. Build a JD text representation from key terms (not raw JD — we encode
     the *intent*, not just the words, to avoid boosting keyword-stuffers).
  2. Build a candidate text representation from:
       - headline + summary (rich free text)
       - career descriptions (most signal-rich)
       - skill names (with proficiency and duration as pseudo-text boosts)
  3. Fit TF-IDF on the candidate + JD texts.
  4. Compute cosine similarity between JD vector and each candidate.
  5. Blend: final = (1 - alpha) * structured_score + alpha * tfidf_similarity
     with alpha = TFIDF_BLEND_WEIGHT = 0.15.

Why this is safe despite keyword-stuffers:
  - Only called for the top 500 candidates from the structured ranker.
  - Those 500 have already passed the career + skill credibility filters.
  - A keyword-stuffer in the top 500 would need to have survived the
    career-trajectory penalty — unlikely given the IT-services + anti-skill
    penalties applied upstream.

The TF-IDF boost helps surface candidates who:
  - Describe real ML work in their summaries using domain vocabulary
  - Use synonyms not in our keyword lists (e.g., "ann", "approximate nearest
    neighbour", "dense passage retrieval", "dual encoder")
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.scoring.composite import CandidateScore
from src.config.settings import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_BLEND_WEIGHT,
    TFIDF_TOP_N_FOR_RERANK,
    DOMAIN_KEYWORDS_IN_DESCRIPTION,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# JD representation
# ─────────────────────────────────────────────────────────────────────────────

# We build a pseudo-document that captures JD intent.
# Terms are weighted by repetition (repeated = higher TF).
JD_TEXT = " ".join([
    # Core requirements — repeated 3x for emphasis
    "embeddings retrieval vector search semantic search " * 3,
    "sentence transformers faiss pinecone qdrant weaviate milvus elasticsearch opensearch " * 3,
    "ranking evaluation ndcg mrr map learning to rank reranking cross encoder " * 3,
    "hybrid search bm25 dense retrieval sparse retrieval " * 3,
    # Important but secondary — repeated 2x
    "python production nlp natural language processing pytorch hugging face " * 2,
    "recommendation systems llm large language model fine tuning lora " * 2,
    "feature engineering mlops model deployment a b testing " * 2,
    "product company startup shipped production deployed " * 2,
    "information retrieval search engineer ml engineer applied scientist " * 2,
    # Nice-to-have — once
    "xgboost lightgbm scikit learn mlflow wandb",
    "transformer bert gpt bi encoder",
    "bangalore bengaluru hyderabad pune noida india",
])


# ─────────────────────────────────────────────────────────────────────────────
# Candidate text builder
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_text(cs: CandidateScore) -> str:
    """
    Build a weighted text representation of a candidate.
    Skills with high duration/endorsements are repeated to boost their TF weight.
    """
    c       = cs.candidate
    profile = c["profile"]
    parts   = []

    # Profile free text
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))

    # Career titles and descriptions
    for role in c.get("career_history", []):
        parts.append(role.get("title", ""))
        # Description carries the most signal — include in full
        desc = role.get("description", "")
        if desc:
            parts.append(desc)

    # Skills — repeat high-credibility skills to boost TF weight
    for sk in c.get("skills", []):
        name = sk.get("name_raw") or sk.get("name", "")
        dur  = sk.get("duration_months", 0)
        end  = sk.get("endorsements", 0)
        # Weight by credibility: 0 → 1, 1 → 2, 12 → 3, 24+ → 4 repetitions
        reps = min(4, 1 + dur // 12 + (1 if end > 10 else 0))
        parts.append((name + " ") * reps)

    # Certifications
    for cert in c.get("certifications", []):
        parts.append(cert.get("name", ""))

    return " ".join(p for p in parts if p).lower()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def tfidf_rerank(
    scored: List[CandidateScore],
    top_n: int = TFIDF_TOP_N_FOR_RERANK,
    alpha: float = TFIDF_BLEND_WEIGHT,
) -> List[CandidateScore]:
    """
    Re-rank the top_n candidates using TF-IDF similarity to the JD.

    Parameters
    ----------
    scored : CandidateScore list, already sorted by composite (descending).
             Only the first top_n entries are re-ranked; the rest are unchanged.
    top_n  : how many candidates to re-rank (default 500)
    alpha  : weight of TF-IDF similarity in the blended score [0, 1]

    Returns
    -------
    Re-sorted CandidateScore list (full list, same length as input).
    """
    if top_n == 0 or alpha == 0.0:
        return scored

    to_rerank = scored[:top_n]
    rest      = scored[top_n:]

    logger.info("TF-IDF re-ranking top %d candidates (alpha=%.2f)...", len(to_rerank), alpha)

    # Build corpus: candidates first, JD last
    texts = [_candidate_text(cs) for cs in to_rerank] + [JD_TEXT]

    try:
        vec = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            sublinear_tf=True,          # apply log(1+tf) to dampen frequency
            min_df=1,
            max_df=0.95,
            strip_accents="unicode",
        )
        matrix     = vec.fit_transform(texts)
        jd_vec     = matrix[-1]
        cand_vecs  = matrix[:-1]
        sims       = cosine_similarity(jd_vec, cand_vecs).flatten()
    except Exception as exc:
        logger.warning("TF-IDF failed, skipping re-rank: %s", exc)
        return scored

    # Blend scores
    # Normalise tfidf sims to [0,1] range (cosine is already [0,1])
    # Normalise structured scores to [0,1] by dividing by 100
    blended_candidates = []
    for i, cs in enumerate(to_rerank):
        structured_norm = cs.composite / 100.0
        tfidf_sim       = float(sims[i])
        blended         = (1.0 - alpha) * structured_norm + alpha * tfidf_sim
        # Store back as a score in [0, 100] range for consistency
        import dataclasses
        cs_new = dataclasses.replace(cs, composite=round(blended * 100.0, 6))
        blended_candidates.append(cs_new)

    # Re-sort by blended score
    blended_candidates.sort(key=lambda s: (-s.composite, s.candidate_id))

    logger.info("TF-IDF re-ranking complete.")
    return blended_candidates + rest


def compute_tfidf_scores(
    scored: List[CandidateScore],
    top_n: int = TFIDF_TOP_N_FOR_RERANK,
) -> List[Tuple[str, float]]:
    """
    Return (candidate_id, tfidf_sim) pairs for the top_n candidates.
    Useful for analysis / debugging without modifying the scored list.
    """
    to_check = scored[:top_n]
    texts    = [_candidate_text(cs) for cs in to_check] + [JD_TEXT]
    try:
        vec    = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            sublinear_tf=True,
        )
        matrix = vec.fit_transform(texts)
        jd_vec = matrix[-1]
        sims   = cosine_similarity(jd_vec, matrix[:-1]).flatten()
    except Exception:
        return [(cs.candidate_id, 0.0) for cs in to_check]
    return [(cs.candidate_id, float(sims[i])) for i, cs in enumerate(to_check)]
