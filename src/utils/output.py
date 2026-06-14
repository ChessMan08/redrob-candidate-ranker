"""
output.py — Submission CSV writer.

Produces exactly the format required by validate_submission.py:
  Row 1:    candidate_id,rank,score,reasoning
  Rows 2+:  100 data rows, scores non-increasing, ranks 1-100 unique
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List

from src.scoring.composite import CandidateScore
from src.scoring.reasoning import generate_reasoning

logger = logging.getLogger(__name__)

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]


def write_submission(
    scored: List[CandidateScore],
    out_path: str | Path,
    top_n: int = 100,
) -> Path:
    """
    Write the top_n candidates to a submission CSV.

    Parameters
    ----------
    scored   : CandidateScore list sorted by composite descending
    out_path : output file path (must end in .csv)
    top_n    : number of candidates to write (default 100)

    Returns
    -------
    Path to the written file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    actual_n = min(top_n, len(scored))
    if actual_n < top_n:
        logger.warning(
            "Only %d candidates scored — writing %d rows (need %d for a valid submission).",
            len(scored), actual_n, top_n,
        )

    rows = []
    for rank, cs in enumerate(scored[:actual_n], start=1):
        reasoning = generate_reasoning(cs, rank)
        rows.append({
            "candidate_id": cs.candidate_id,
            "rank":         rank,
            "score":        f"{cs.composite:.6f}",
            "reasoning":    reasoning,
        })

    # Validate score monotonicity before writing
    for i in range(len(rows) - 1):
        s1 = float(rows[i]["score"])
        s2 = float(rows[i + 1]["score"])
        if s1 < s2:
            logger.error(
                "Score not monotonically non-increasing at ranks %d→%d: %.6f < %.6f",
                rows[i]["rank"], rows[i + 1]["rank"], s1, s2,
            )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d rows to %s", len(rows), out_path)
    return out_path


def validate_submission_locally(csv_path: str | Path) -> List[str]:
    """
    Run the same checks as validate_submission.py.
    Returns a list of error strings (empty = valid).
    """
    import re
    path   = Path(csv_path)
    errors = []

    CAND_PATTERN = re.compile(r"^CAND_[0-9]{7}$")

    if path.suffix.lower() != ".csv":
        errors.append(f"Extension must be .csv, got {path.suffix}")
        return errors

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                errors.append("File is empty")
                return errors

            if header != REQUIRED_HEADER:
                errors.append(f"Header mismatch. Expected {REQUIRED_HEADER}, got {header}")

            data_rows = [row for row in reader if any(c.strip() for c in row)]
    except UnicodeDecodeError:
        errors.append("File must be UTF-8 encoded")
        return errors

    if len(data_rows) != 100:
        errors.append(f"Expected 100 data rows, found {len(data_rows)} — OK for dev/sample runs, required for final submission")

    seen_ids   = set()
    seen_ranks = set()
    by_rank    = []

    for i, row in enumerate(data_rows):
        row_num = i + 2
        if len(row) != 4:
            errors.append(f"Row {row_num}: expected 4 columns, got {len(row)}")
            continue

        cid, rank_s, score_s, reasoning = row

        if not CAND_PATTERN.match(cid.strip()):
            errors.append(f"Row {row_num}: invalid candidate_id '{cid}'")
        elif cid in seen_ids:
            errors.append(f"Row {row_num}: duplicate candidate_id '{cid}'")
        else:
            seen_ids.add(cid)

        try:
            rank = int(rank_s.strip())
            if not 1 <= rank <= 100:
                errors.append(f"Row {row_num}: rank {rank} out of [1,100]")
            elif rank in seen_ranks:
                errors.append(f"Row {row_num}: duplicate rank {rank}")
            else:
                seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {row_num}: rank must be integer, got '{rank_s}'")
            rank = None

        try:
            score = float(score_s.strip())
        except ValueError:
            errors.append(f"Row {row_num}: score must be float, got '{score_s}'")
            score = None

        if rank and score is not None:
            by_rank.append((rank, score, cid.strip()))

    missing = set(range(1, 101)) - seen_ranks
    if missing:
        errors.append(f"Missing ranks: {sorted(missing)}")

    by_rank.sort()
    for i in range(len(by_rank) - 1):
        r1, s1, _ = by_rank[i]
        r2, s2, _ = by_rank[i + 1]
        if s1 < s2:
            errors.append(f"Score not non-increasing: rank {r1}={s1} < rank {r2}={s2}")

    return errors
