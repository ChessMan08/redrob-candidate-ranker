"""
parallel.py — CPU-parallel batch processing for scoring 100K candidates.

Uses ProcessPoolExecutor with chunked work to maximise CPU utilisation
while staying within the 16 GB RAM constraint.

Memory note:
  100K candidates × ~5 KB each ≈ 500 MB loaded.
  After scoring each chunk the full candidate dict is detached from the
  returned CandidateScore (candidate=None) to free memory early.
  The full dict is kept only for the top-N final candidates.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

from src.data.preprocessor import clean_candidate
from src.scoring.composite import score_candidate, CandidateScore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Worker function (must be importable at module level for pickle)
# ─────────────────────────────────────────────────────────────────────────────

def _score_chunk(raw_chunk: List[dict]) -> List[dict]:
    """
    Worker: clean + score a chunk of raw candidate dicts.
    Returns lightweight dicts (no full candidate payload) for fast pickling.
    """
    results = []
    for raw in raw_chunk:
        try:
            c  = clean_candidate(raw)
            cs = score_candidate(c)
            results.append({
                "candidate_id":       cs.candidate_id,
                "composite":          cs.composite,
                "career_score":       cs.career_score,
                "skills_score":       cs.skills_score,
                "experience_score":   cs.experience_score,
                "behavioral_score":   cs.behavioral_score,
                "location_score":     cs.location_score,
                "education_score":    cs.education_score,
                "honeypot_multiplier":cs.honeypot_multiplier,
                "behavioral_gate":    cs.behavioral_gate,
                "salary_multiplier":  cs.salary_multiplier,
                "honeypot_flags":     cs.honeypot_flags,
                "tier1_skills":       cs.tier1_skills,
                "tier2_skills":       cs.tier2_skills,
                "career_breakdown":   cs.career_breakdown,
                "behavioral_breakdown":cs.behavioral_breakdown,
                "location_breakdown": cs.location_breakdown,
                # Attach raw candidate so we can re-hydrate top-N
                "_raw": raw,
            })
        except Exception as exc:
            cid = raw.get("candidate_id", "?")
            logger.warning("Error scoring %s: %s", cid, exc)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parallel_score(
    raw_candidates: List[dict],
    n_workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    top_n_to_keep_full: int = 500,
    show_progress: bool = True,
) -> List[CandidateScore]:
    """
    Score all candidates in parallel, return sorted CandidateScore list.

    Parameters
    ----------
    raw_candidates         : raw (uncleaned) candidate dicts
    n_workers              : CPU workers; default = min(8, cpu_count)
    chunk_size             : records per worker task; default = auto
    top_n_to_keep_full     : re-attach full candidate dict only for top-N
                             (saves RAM for the rest)
    show_progress          : show tqdm bar if installed

    Returns
    -------
    CandidateScore list sorted by composite descending.
    """
    n_workers  = n_workers or min(8, multiprocessing.cpu_count())
    total      = len(raw_candidates)
    chunk_size = chunk_size or max(500, math.ceil(total / (n_workers * 4)))

    chunks = [
        raw_candidates[i: i + chunk_size]
        for i in range(0, total, chunk_size)
    ]

    logger.info(
        "Parallel scoring: %d candidates / %d workers / %d chunks (size≈%d)",
        total, n_workers, len(chunks), chunk_size,
    )

    all_results: List[dict] = []

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(chunks), desc="Scoring", unit="chunk") if show_progress else None
    except ImportError:
        pbar = None

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_score_chunk, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            try:
                all_results.extend(future.result())
            except Exception as exc:
                logger.error("Chunk failed: %s", exc)
            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    # Sort by composite descending, then candidate_id ascending for tie-break
    all_results.sort(key=lambda r: (-r["composite"], r["candidate_id"]))

    # Re-hydrate full candidate dict only for top-N (RAM-efficient)
    scored: List[CandidateScore] = []
    for i, r in enumerate(all_results):
        raw = r.pop("_raw", None)
        if i < top_n_to_keep_full and raw is not None:
            full_candidate = clean_candidate(raw)
        else:
            full_candidate = None

        cs = CandidateScore(
            candidate_id        = r["candidate_id"],
            composite           = r["composite"],
            career_score        = r["career_score"],
            skills_score        = r["skills_score"],
            experience_score    = r["experience_score"],
            behavioral_score    = r["behavioral_score"],
            location_score      = r["location_score"],
            education_score     = r["education_score"],
            honeypot_multiplier = r["honeypot_multiplier"],
            behavioral_gate     = r["behavioral_gate"],
            salary_multiplier   = r["salary_multiplier"],
            honeypot_flags      = r["honeypot_flags"],
            tier1_skills        = r["tier1_skills"],
            tier2_skills        = r["tier2_skills"],
            career_breakdown    = r["career_breakdown"],
            behavioral_breakdown= r["behavioral_breakdown"],
            location_breakdown  = r["location_breakdown"],
            candidate           = full_candidate,
        )
        scored.append(cs)

    logger.info("Scoring complete. Top score=%.2f, #100 score=%.2f",
                scored[0].composite if scored else 0,
                scored[99].composite if len(scored) >= 100 else 0)
    return scored


def sequential_score(
    raw_candidates: List[dict],
    show_progress: bool = True,
) -> List[CandidateScore]:
    """
    Single-process fallback (safer for small datasets or debugging).
    Cleans + scores inline; keeps full candidate dict for all records.
    """
    from src.data.preprocessor import clean_candidates
    from src.scoring.composite import score_candidates

    logger.info("Sequential scoring %d candidates...", len(raw_candidates))
    cleaned = clean_candidates(raw_candidates)
    scored  = score_candidates(cleaned, show_progress=show_progress)
    return scored
