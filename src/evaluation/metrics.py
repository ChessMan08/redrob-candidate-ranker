"""
metrics.py — Offline ranking evaluation metrics.

Implements the exact metrics used in the hidden ground truth evaluation:
  - NDCG@K  (primary: @10 and @50)
  - MAP      (Mean Average Precision)
  - P@K      (Precision at K)
  - MRR      (Mean Reciprocal Rank)

Also implements:
  - Honeypot rate in top-K
  - Score distribution diagnostics

Relevance grades (assumed from spec context):
  0 = not relevant (honeypot or clearly unfit)
  1 = borderline relevant
  2 = relevant
  3 = highly relevant (tier 3+ in spec language)

Without the ground truth, we use proxy labels from our scoring system
and from manual annotation of the 50 sample candidates.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Core metric implementations
# ─────────────────────────────────────────────────────────────────────────────

def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    relevances = list(relevances)[:k]
    return sum(
        rel / math.log2(i + 2)  # position is 0-indexed; log(2) for position 1
        for i, rel in enumerate(relevances)
    )


def ndcg_at_k(
    ranked_ids: List[str],
    relevance_map: Dict[str, float],
    k: int,
) -> float:
    """
    NDCG@K.

    Parameters
    ----------
    ranked_ids    : candidate_id list in ranked order (best first)
    relevance_map : {candidate_id: relevance_score} — ground truth grades
    k             : cutoff

    Returns
    -------
    NDCG@K in [0, 1]
    """
    if not ranked_ids or not relevance_map:
        return 0.0

    # Actual DCG
    actual_rels = [relevance_map.get(cid, 0.0) for cid in ranked_ids[:k]]
    actual_dcg  = dcg_at_k(actual_rels, k)

    # Ideal DCG: sort all known relevant items by relevance
    ideal_rels = sorted(relevance_map.values(), reverse=True)
    ideal_dcg  = dcg_at_k(ideal_rels, k)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def precision_at_k(
    ranked_ids: List[str],
    relevant_ids: set,
    k: int,
) -> float:
    """
    P@K: fraction of top-K that are relevant.
    relevant_ids: set of candidate_ids with relevance > threshold (usually >= 3).
    """
    top_k = ranked_ids[:k]
    hits  = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / max(1, len(top_k))


def average_precision(
    ranked_ids: List[str],
    relevant_ids: set,
) -> float:
    """
    Average Precision for a single query (single JD in our case).
    """
    if not relevant_ids:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for i, cid in enumerate(ranked_ids, 1):
        if cid in relevant_ids:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / max(1, len(relevant_ids))


def mean_average_precision(
    ranked_ids: List[str],
    relevant_ids: set,
) -> float:
    """MAP (for single-query scenario this is just AP)."""
    return average_precision(ranked_ids, relevant_ids)


def reciprocal_rank(
    ranked_ids: List[str],
    relevant_ids: set,
) -> float:
    """MRR for a single query."""
    for i, cid in enumerate(ranked_ids, 1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


def composite_score(
    ranked_ids: List[str],
    relevance_map: Dict[str, float],
    relevant_ids_for_precision: Optional[set] = None,
) -> Dict[str, float]:
    """
    Compute the exact composite used in the competition:
      0.50 * NDCG@10 + 0.30 * NDCG@50 + 0.15 * MAP + 0.05 * P@10

    Parameters
    ----------
    ranked_ids                 : candidate IDs in ranked order
    relevance_map              : {cid: grade} — ground truth
    relevant_ids_for_precision : for P@K, which IDs are "relevant" (grade >= 3).
                                 If None, derived from relevance_map >= 3.
    """
    if relevant_ids_for_precision is None:
        relevant_ids_for_precision = {
            cid for cid, g in relevance_map.items() if g >= 3
        }

    ndcg10 = ndcg_at_k(ranked_ids, relevance_map, 10)
    ndcg50 = ndcg_at_k(ranked_ids, relevance_map, 50)
    map_   = mean_average_precision(ranked_ids, relevant_ids_for_precision)
    p10    = precision_at_k(ranked_ids, relevant_ids_for_precision, 10)

    comp = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * map_ + 0.05 * p10

    return {
        "ndcg@10":   round(ndcg10, 4),
        "ndcg@50":   round(ndcg50, 4),
        "map":       round(map_,   4),
        "p@10":      round(p10,    4),
        "composite": round(comp,   4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Honeypot rate
# ─────────────────────────────────────────────────────────────────────────────

def honeypot_rate_in_top_k(
    ranked_ids: List[str],
    honeypot_ids: set,
    k: int = 100,
) -> float:
    """
    Fraction of top-K that are known honeypots.
    Submissions with > 10% are disqualified (per spec).
    """
    top_k = ranked_ids[:k]
    hits  = sum(1 for cid in top_k if cid in honeypot_ids)
    return hits / max(1, len(top_k))


# ─────────────────────────────────────────────────────────────────────────────
# Score distribution diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def score_diagnostics(scores: List[float]) -> Dict[str, float]:
    """Basic stats on the score distribution."""
    if not scores:
        return {}
    import statistics
    return {
        "min":    round(min(scores), 4),
        "max":    round(max(scores), 4),
        "mean":   round(statistics.mean(scores), 4),
        "median": round(statistics.median(scores), 4),
        "stdev":  round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 4),
        "p90":    round(sorted(scores)[int(0.9 * len(scores))], 4),
        "p99":    round(sorted(scores)[int(0.99 * len(scores))], 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Proxy-label generation (weak supervision)
# ─────────────────────────────────────────────────────────────────────────────

def make_proxy_relevance_map(
    scored_candidates: List[Dict],
    threshold_high: float = 70.0,
    threshold_med:  float = 45.0,
    threshold_low:  float = 25.0,
) -> Dict[str, float]:
    """
    Build proxy relevance grades from composite scores (0-3 scale).

    When no ground truth is available:
      composite >= threshold_high → grade 3 (highly relevant)
      composite >= threshold_med  → grade 2 (relevant)
      composite >= threshold_low  → grade 1 (borderline)
      below                       → grade 0

    These are used for offline evaluation / sensitivity analysis.
    NOT used in the final submission.

    Parameters
    ----------
    scored_candidates : list of dicts with 'candidate_id' and 'composite'
    """
    rel_map = {}
    for item in scored_candidates:
        cid   = item["candidate_id"]
        score = item["composite"]
        if score >= threshold_high:
            rel_map[cid] = 3.0
        elif score >= threshold_med:
            rel_map[cid] = 2.0
        elif score >= threshold_low:
            rel_map[cid] = 1.0
        else:
            rel_map[cid] = 0.0
    return rel_map
