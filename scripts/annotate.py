#!/usr/bin/env python3
"""
annotate.py — Interactive CLI tool to manually label candidate relevance.

Presents candidates from the top-N of your current ranker one by one.
You assign a grade (0-3). Labels are saved to artifacts/manual_labels.json.

This creates the ground truth needed for meaningful offline evaluation.

Usage:
    python scripts/annotate.py --candidates sample_candidates.json --top 50
    python scripts/annotate.py --candidates candidates.jsonl.gz --top 200 --start 51

Grades:
    3 = Highly relevant (would shortlist to recruiter)
    2 = Relevant (would consider)
    1 = Borderline (maybe with more context)
    0 = Not relevant (wrong role, wrong background)
    s = Skip (unsure, review later)
    q = Quit and save
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from src.data.loader import load_candidates, deduplicate
from src.data.preprocessor import clean_candidates
from src.scoring.composite import score_candidates
from src.scoring.reasoning import generate_reasoning
from src.config.settings import ARTIFACTS_DIR, IT_SERVICES_COMPANIES, TODAY


LABELS_PATH = ARTIFACTS_DIR / "manual_labels.json"


def _days_since(d) -> int:
    if d is None:
        return 999
    from datetime import date
    if isinstance(d, str):
        try:
            d = date.fromisoformat(d)
        except Exception:
            return 999
    return max(0, (TODAY - d).days)


def display_candidate(cs, rank: int) -> None:
    """Print a rich candidate summary for annotation."""
    c       = cs.candidate
    profile = c["profile"]
    sig     = c["redrob_signals"]
    career  = c["career_history"]
    skills  = c["skills"]

    print("\n" + "=" * 72)
    print(f"  RANK {rank:3d}  |  {cs.candidate_id}  |  Score: {cs.composite:.1f}")
    print("=" * 72)

    print(f"\n  Title    : {profile.get('current_title','')}")
    print(f"  Company  : {profile.get('current_company','')}  ({profile.get('current_industry','')})")
    print(f"  Location : {profile.get('location','')}, {profile.get('country','')}")
    print(f"  YoE      : {profile.get('years_of_experience',0):.1f} years")

    print(f"\n  Headline : {profile.get('headline','')[:80]}")

    print(f"\n  Career History:")
    for r in career:
        it = any(s in (r.get("company") or "").lower() for s in IT_SERVICES_COMPANIES)
        tag = " [IT-SVC]" if it else ""
        print(f"    {r.get('duration_months',0):3d}m  {r.get('title',''):35s}  @ {r.get('company','')}{tag}")

    t1 = cs.tier1_skills[:5]
    t2 = cs.tier2_skills[:4]
    print(f"\n  Tier-1 Skills : {', '.join(t1) if t1 else '(none)'}")
    print(f"  Tier-2 Skills : {', '.join(t2) if t2 else '(none)'}")

    print(f"\n  Behavioral:")
    open_w = sig.get("open_to_work_flag", False)
    days   = _days_since(sig.get("last_active_date"))
    rrr    = sig.get("recruiter_response_rate", 0)
    notice = sig.get("notice_period_days", 90)
    github = sig.get("github_activity_score", -1)
    print(f"    open_to_work={open_w}  last_active={days}d  rrr={rrr:.0%}  notice={notice}d  github={github}")

    print(f"\n  Scores: career={cs.career_score:.0f}  skills={cs.skills_score:.0f}  "
          f"exp={cs.experience_score:.0f}  behavioral={cs.behavioral_score:.0f}  "
          f"loc={cs.location_score:.0f}")
    if cs.honeypot_flags:
        print(f"  ⚠ Honeypot flags: {cs.honeypot_flags}")

    print(f"\n  Generated reasoning:")
    print(f"  {generate_reasoning(cs, rank)}")


def load_existing_labels() -> dict:
    if LABELS_PATH.exists():
        with open(LABELS_PATH) as f:
            return json.load(f)
    return {}


def save_labels(labels: dict) -> None:
    LABELS_PATH.parent.mkdir(exist_ok=True)
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"\n  Saved {len(labels)} labels to {LABELS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Manual relevance annotation")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--top", type=int, default=100,
                        help="Annotate top-N candidates from the ranker")
    parser.add_argument("--start", type=int, default=1,
                        help="Start from rank N (resume annotation)")
    parser.add_argument("--max-load", type=int, default=None,
                        help="Max records to load from file (for speed)")
    args = parser.parse_args()

    print("Loading candidates...")
    raw     = load_candidates(args.candidates, max_records=args.max_load)
    raw     = deduplicate(raw)
    cleaned = clean_candidates(raw)
    scored  = score_candidates(cleaned, show_progress=True)

    labels   = load_existing_labels()
    start_i  = args.start - 1  # 0-indexed

    print(f"\nAnnotation session: top-{args.top}, starting at rank {args.start}")
    print("Grades: 3=Highly relevant  2=Relevant  1=Borderline  0=Not relevant")
    print("        s=Skip  q=Quit and save\n")

    try:
        for i, cs in enumerate(scored[start_i:args.top], start=start_i + 1):
            if cs.candidate is None:
                print(f"Rank {i}: no candidate dict attached (load with sequential scorer)")
                continue

            display_candidate(cs, i)

            while True:
                choice = input("\n  Grade [0/1/2/3/s/q]: ").strip().lower()
                if choice in ("0", "1", "2", "3"):
                    labels[cs.candidate_id] = int(choice)
                    print(f"  → Labelled {cs.candidate_id} = {choice}")
                    break
                elif choice == "s":
                    print(f"  → Skipped {cs.candidate_id}")
                    break
                elif choice == "q":
                    print("  → Quitting...")
                    save_labels(labels)
                    return
                else:
                    print("  Invalid input. Enter 0, 1, 2, 3, s, or q.")

    except KeyboardInterrupt:
        print("\n\nInterrupted.")

    save_labels(labels)
    print(f"\nDone. {len(labels)} candidates labelled.")
    print(f"Run: python scripts/evaluate.py --candidates {args.candidates}")


if __name__ == "__main__":
    main()
