#!/usr/bin/env python3
"""
rank.py — Main inference script for the Redrob hackathon.

Produces a valid submission.csv from candidates.jsonl.gz in < 5 minutes
on CPU with 16 GB RAM and no network access.

Usage:
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv

Advanced:
    python rank.py \\
        --candidates ./candidates.jsonl.gz \\
        --out ./submission.csv \\
        --workers 8 \\
        --tfidf           \\   # enable TF-IDF re-ranking (adds ~10s)
        --no-parallel         # force single-process mode (safer, slightly slower)

Exit codes:
    0 = success, submission written and validated
    1 = validation failed (CSV format errors)
    2 = runtime error
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Ensure project root is on PYTHONPATH when run as a script ────────────────
import os
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import load_candidates, deduplicate
from src.data.preprocessor import clean_candidates
from src.scoring.composite import score_candidates
from src.utils.output import write_submission, validate_submission_locally
from src.evaluation.evaluator import (
    build_evaluation_report,
    print_evaluation_report,
    load_manual_labels,
    load_honeypot_ids,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rank")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def step_load(candidates_path: str, max_records: int | None) -> list:
    t0 = time.time()
    logger.info("Step 1/6 — Loading candidates from %s", candidates_path)
    raw = load_candidates(candidates_path, max_records=max_records)
    raw = deduplicate(raw)
    logger.info("  Loaded %d candidates in %.1fs", len(raw), time.time() - t0)
    return raw


def step_clean(raw: list) -> list:
    t0 = time.time()
    logger.info("Step 2/6 — Cleaning and normalising ...")
    cleaned = clean_candidates(raw)
    logger.info("  Cleaned %d records in %.1fs", len(cleaned), time.time() - t0)
    return cleaned


def step_score(cleaned: list, use_parallel: bool, n_workers: int) -> list:
    t0 = time.time()
    logger.info("Step 3/6 — Scoring candidates (%s, workers=%d) ...",
                "parallel" if use_parallel else "sequential", n_workers)

    if use_parallel and len(cleaned) > 1000:
        from src.utils.parallel import parallel_score
        # parallel_score takes raw dicts; we already cleaned, so pass cleaned as-is
        # (parallel_score calls clean_candidate internally, but we can also
        # use sequential_score on already-cleaned data — cleaner path below)
        scored = score_candidates(cleaned, show_progress=True)
    else:
        scored = score_candidates(cleaned, show_progress=True)

    logger.info("  Scored %d candidates in %.1fs", len(scored), time.time() - t0)
    return scored


def step_tfidf_rerank(scored: list, top_n: int = 500) -> list:
    t0 = time.time()
    logger.info("Step 4/6 — TF-IDF re-ranking top %d candidates ...", top_n)
    try:
        from src.retrieval.tfidf_reranker import tfidf_rerank
        scored = tfidf_rerank(scored, top_n=top_n)
        logger.info("  TF-IDF re-rank done in %.1fs", time.time() - t0)
    except Exception as exc:
        logger.warning("  TF-IDF re-rank failed (%s), using structured scores only.", exc)
    return scored


def step_evaluate(scored: list) -> None:
    logger.info("Step 5/6 — Building evaluation report ...")
    manual_labels = load_manual_labels()
    honeypot_ids  = load_honeypot_ids()
    report = build_evaluation_report(scored, manual_labels, honeypot_ids)
    print_evaluation_report(report)


def step_write(scored: list, out_path: str) -> int:
    t0 = time.time()
    logger.info("Step 6/6 — Writing submission to %s ...", out_path)
    path = write_submission(scored, out_path, top_n=100)

    # Validate locally before returning
    errors = validate_submission_locally(path)
    if errors:
        logger.error("Submission INVALID (%d errors):", len(errors))
        for e in errors:
            logger.error("  - %s", e)
        return 1

    logger.info("  Submission written and validated in %.1fs", time.time() - t0)
    logger.info("  → %s  (run validate_submission.py for official check)", path)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Redrob hackathon candidate ranker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidates.jsonl.gz (or .jsonl / .json for sample data)",
    )
    parser.add_argument(
        "--out", default="submission.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Parallel worker count (0 = use sequential scorer)",
    )
    parser.add_argument(
        "--tfidf", action="store_true",
        help="Enable TF-IDF re-ranking of top-500 candidates",
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Force single-process mode",
    )
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="Load only first N records (for dev/testing)",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip evaluation report (faster)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Set log level to DEBUG",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    wall_start = time.time()

    try:
        # 1. Load
        raw = step_load(args.candidates, args.max_records)
        if not raw:
            logger.error("No candidates loaded. Check the file path.")
            return 2

        # 2. Clean
        cleaned = step_clean(raw)

        # 3. Score
        use_parallel = not args.no_parallel and args.workers > 0
        n_workers    = args.workers if use_parallel else 1
        scored = step_score(cleaned, use_parallel, n_workers)

        if not scored:
            logger.error("No candidates scored.")
            return 2

        # 4. TF-IDF re-rank (optional)
        if args.tfidf:
            scored = step_tfidf_rerank(scored, top_n=500)

        # 5. Evaluate (optional)
        if not args.no_eval:
            step_evaluate(scored)

        # 6. Write
        exit_code = step_write(scored, args.out)

        wall_elapsed = time.time() - wall_start
        logger.info("Total wall time: %.1fs (%.1f min)", wall_elapsed, wall_elapsed / 60)

        return exit_code

    except KeyboardInterrupt:
        logger.info("Interrupted.")
        return 2
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
