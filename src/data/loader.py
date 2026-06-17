from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Generator, List, Optional

logger = logging.getLogger(__name__)

CANDIDATE_ID_PREFIX = "CAND_"

# Low-level streaming generators
def _iter_jsonl(path: Path) -> Generator[dict, None, None]:
    """Stream a plain JSONL file line by line."""
    bad = 0
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                bad += 1
                logger.debug("Bad JSON at line %d: %s", lineno, exc)
    if bad:
        logger.warning("%s: skipped %d malformed lines", path.name, bad)


def _iter_jsonl_gz(path: Path) -> Generator[dict, None, None]:
    """Stream a gzip-compressed JSONL file line by line."""
    bad = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                bad += 1
                logger.debug("Bad JSON at line %d: %s", lineno, exc)
    if bad:
        logger.warning("%s: skipped %d malformed lines", path.name, bad)


def _iter_json_array(path: Path) -> Generator[dict, None, None]:
    """Load a JSON array file (e.g. sample_candidates.json)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array, got {type(data).__name__}")
    yield from data

# Public API
def stream_candidates(path: str | Path) -> Generator[dict, None, None]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    suffix = "".join(path.suffixes).lower()  # handles .jsonl.gz correctly

    if suffix in (".jsonl.gz", ".json.gz"):
        yield from _iter_jsonl_gz(path)
    elif suffix == ".jsonl":
        yield from _iter_jsonl(path)
    elif suffix == ".json":
        try:
            yield from _iter_json_array(path)
        except (json.JSONDecodeError, ValueError):
            logger.warning("%s: not a JSON array, trying JSONL", path.name)
            yield from _iter_jsonl(path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")


def load_candidates(
    path: str | Path,
    max_records: Optional[int] = None,
    require_valid_id: bool = True,
) -> List[dict]:
    candidates: List[dict] = []
    skipped_id = 0

    for record in stream_candidates(path):
        cid = record.get("candidate_id", "")
        if require_valid_id:
            if not (isinstance(cid, str) and cid.startswith(CANDIDATE_ID_PREFIX)):
                skipped_id += 1
                continue

        candidates.append(record)

        if max_records is not None and len(candidates) >= max_records:
            break

    if skipped_id:
        logger.warning("Dropped %d records with invalid candidate_id", skipped_id)

    logger.info("Loaded %d candidates from %s", len(candidates), path)
    return candidates

# Duplicate detection
def deduplicate(candidates: List[dict]) -> List[dict]:
    seen: set[str] = set()
    out: List[dict] = []
    dupes = 0
    for c in candidates:
        cid = c.get("candidate_id", "")
        if cid in seen:
            dupes += 1
        else:
            seen.add(cid)
            out.append(c)
    if dupes:
        logger.warning("Removed %d duplicate candidate_ids", dupes)
    return out
