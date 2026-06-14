#!/usr/bin/env python3
"""
check_honeypots.py — Show exactly which candidates in your top-100
are flagged as honeypots, and why.

Usage:
    python scripts/check_honeypots.py --candidates candidates.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_candidates, deduplicate
from src.data.preprocessor import clean_candidates
from src.scoring.composite import score_candidates
from src.retrieval.tfidf_reranker import tfidf_rerank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--tfidf", action="store_true", default=True)
    args = parser.parse_args()

    print("Loading and scoring...")
    raw     = load_candidates(args.candidates)
    raw     = deduplicate(raw)
    cleaned = clean_candidates(raw)
    scored  = score_candidates(cleaned, show_progress=True)

    if args.tfidf:
        scored = tfidf_rerank(scored, top_n=500)

    top100 = scored[:100]

    # ── Summary ──────────────────────────────────────────────────────────────
    flagged = [cs for cs in top100 if cs.honeypot_flags]
    print(f"\n{'='*65}")
    print(f"HONEYPOT SUMMARY")
    print(f"{'='*65}")
    print(f"Top-100 candidates:        100")
    print(f"Flagged as suspicious:     {len(flagged)}")
    print(f"Disqualification threshold: >10")
    if len(flagged) <= 10:
        print(f"Status:  SAFE  ({len(flagged)}/10)")
    else:
        print(f"Status:  *** DISQUALIFIED *** — reduce flagged count below 10")

    # ── Detail on each flagged candidate ─────────────────────────────────────
    if flagged:
        print(f"\n{'='*65}")
        print("FLAGGED CANDIDATES — DETAIL")
        print(f"{'='*65}")
        for rank, cs in enumerate(top100, 1):
            if not cs.honeypot_flags:
                continue
            c = cs.candidate
            p = c["profile"]
            sig = c["redrob_signals"]

            print(f"\nRank {rank:3d} | {cs.candidate_id}")
            print(f"  Title   : {p.get('current_title')} @ {p.get('current_company')}")
            print(f"  YoE     : {p.get('years_of_experience'):.1f}  "
                  f"Score: {cs.composite:.2f}  hp_mult: {cs.honeypot_multiplier:.3f}")
            print(f"  Flags   : {cs.honeypot_flags}")

            # Show the specific skills causing flags
            assessments = sig.get("skill_assessment_scores", {}) or {}
            print(f"  Skills causing flags:")
            for sk in c["skills"]:
                name = sk.get("name_raw") or sk.get("name", "")
                dur  = sk.get("duration_months", 0)
                end  = sk.get("endorsements", 0)
                prof = sk.get("proficiency", "")
                assess = assessments.get(name)

                is_suspicious = False
                reasons = []
                if end > 15 and dur == 0:
                    is_suspicious = True
                    reasons.append(f"endorsements={end} but duration=0")
                if assess is not None and prof == "expert" and assess < 25:
                    is_suspicious = True
                    reasons.append(f"expert but assessment={assess:.0f}/100")
                if assess is not None and prof == "advanced" and assess < 15:
                    is_suspicious = True
                    reasons.append(f"advanced but assessment={assess:.0f}/100")

                if is_suspicious:
                    print(f"    ⚠  {name:30s} {prof:12s} dur={dur}m  end={end}  {' | '.join(reasons)}")

            print(f"\n  Decision: ", end="")
            if cs.honeypot_multiplier < 0.5:
                print("STRONG honeypot signal — score already penalised by "
                      f"{(1-cs.honeypot_multiplier)*100:.0f}%. Consider removing from top-100.")
            elif cs.honeypot_multiplier < 0.8:
                print("MODERATE signal — score penalised, likely OK to keep if career/skills are genuine.")
            else:
                print("MILD signal — minor flag, safe to keep.")

    # ── Also show what's NOT in top-100 that should be ───────────────────────
    print(f"\n{'='*65}")
    print("CLEAN ML CANDIDATES JUST OUTSIDE TOP-100 (ranks 101-110)")
    print(f"{'='*65}")
    for rank, cs in enumerate(scored[100:110], 101):
        if cs.candidate is None:
            continue
        c = cs.candidate
        p = c["profile"]
        t1 = ", ".join(cs.tier1_skills[:2]) if cs.tier1_skills else "-"
        flag_str = f"  ⚠ {cs.honeypot_flags}" if cs.honeypot_flags else ""
        print(f"  Rank {rank}: {cs.candidate_id}  {p.get('current_title','')[:35]:35s}  "
              f"score={cs.composite:.2f}  skills=[{t1}]{flag_str}")


if __name__ == "__main__":
    main()
