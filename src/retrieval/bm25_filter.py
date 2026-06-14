"""
bm25_filter.py — Optional BM25 pre-filter.

For 100K candidates, the structured scorer runs in ~60s on CPU, so BM25
pre-filtering is NOT required for the 5-minute budget.

However, if you extend this to 1M+ candidates or want a fast first-pass
retrieval layer, this module provides a BM25 index over candidate text.

Usage pattern (not used in default rank.py pipeline):
  index = build_bm25_index(candidates)
  top_ids = bm25_retrieve(index, jd_query, top_k=2000)
  top_candidates = [c for c in candidates if c['candidate_id'] in top_ids]

The BM25 query is built from key JD terms — NOT the full JD text.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

# JD-focused BM25 query terms (high precision, not recall)
BM25_QUERY_TERMS = [
    "embeddings", "retrieval", "vector", "search", "ranking",
    "faiss", "pinecone", "qdrant", "weaviate", "milvus",
    "elasticsearch", "opensearch", "bm25",
    "sentence transformers", "semantic",
    "ndcg", "mrr", "reranking", "learning to rank",
    "recommendation", "nlp", "pytorch",
    "production", "shipped", "deployed",
]


def _build_candidate_tokens(candidate: Dict) -> List[str]:
    """Tokenise candidate into BM25-compatible token list."""
    parts = []
    profile = candidate.get("profile", {})
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    for sk in candidate.get("skills", []):
        dur = sk.get("duration_months", 0)
        if dur > 0:  # only tokenise skills with evidence
            parts.append((sk.get("name_raw") or sk.get("name", "")) * max(1, dur // 12))
    return " ".join(p for p in parts if p).lower().split()


class BM25Index:
    def __init__(self, bm25, ids: List[str]):
        self.bm25 = bm25
        self.ids  = ids

    def retrieve(self, query_tokens: List[str], top_k: int = 2000) -> Set[str]:
        scores = self.bm25.get_scores(query_tokens)
        import numpy as np
        top_idxs = np.argsort(scores)[::-1][:top_k]
        return {self.ids[i] for i in top_idxs}


def build_bm25_index(candidates: List[Dict]) -> "BM25Index | None":
    """
    Build a BM25 index from a list of cleaned candidates.
    Returns None if rank-bm25 is not installed.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank-bm25 not installed; BM25 pre-filter disabled.")
        return None

    ids     = [c["candidate_id"] for c in candidates]
    corpus  = [_build_candidate_tokens(c) for c in candidates]
    bm25    = BM25Okapi(corpus)
    logger.info("Built BM25 index over %d candidates.", len(candidates))
    return BM25Index(bm25, ids)


def bm25_retrieve(
    index: "BM25Index",
    top_k: int = 5000,
) -> Set[str]:
    """Retrieve top_k candidate IDs using the standard JD query."""
    if index is None:
        return set()
    query_tokens = " ".join(BM25_QUERY_TERMS).lower().split()
    return index.retrieve(query_tokens, top_k=top_k)
