# show_top100.py: Print a human-readable summary of your top-100 candidates.
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_candidates, deduplicate
from src.data.preprocessor import clean_candidates
from src.scoring.composite import score_candidates
from src.retrieval.tfidf_reranker import tfidf_rerank
from src.config.settings import IT_SERVICES_COMPANIES, TODAY
from datetime import date


def days_since(d):
    if d is None:
        return 999
    try:
        return max(0, (TODAY - d).days)
    except Exception:
        return 999


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--csv", default=None,
                        help="If provided, show reasoning from this CSV alongside")
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()

    # Load reasoning from CSV if provided
    csv_reasoning = {}
    if args.csv:
        with open(args.csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                csv_reasoning[row["candidate_id"]] = row["reasoning"]

    print("Scoring all candidates...")
    raw     = load_candidates(args.candidates)
    raw     = deduplicate(raw)
    cleaned = clean_candidates(raw)
    scored  = score_candidates(cleaned, show_progress=True)
    scored  = tfidf_rerank(scored, top_n=500)

    print(f"\n{'='*80}")
    print(f"TOP {args.top} CANDIDATES - MANUAL REVIEW")
    print(f"{'='*80}")
    print(f"{'Rk':>3}  {'ID':12}  {'Title':35}  {'Company':20}  {'YoE':4}  {'Score':6}")
    print(f"{'':3}  {'':12}  {'Car':5} {'Sk':5} {'Beh':5} {'Loc':5}  "
          f"{'Open':5} {'Days':5} {'RRR':5} {'Notice':6}  {'Tier-1 Skills'}")
    print("-"*120)

    for rank, cs in enumerate(scored[:args.top], 1):
        if cs.candidate is None:
            print(f"{rank:3d}  {cs.candidate_id}  [no candidate data]  {cs.composite:.2f}")
            continue

        c   = cs.candidate
        p   = c["profile"]
        sig = c["redrob_signals"]

        title   = p.get("current_title", "")[:35]
        company = p.get("current_company", "")[:20]
        yoe     = p.get("years_of_experience", 0)
        country = p.get("country", "")[:3].upper()
        loc     = p.get("location", "")[:15]

        open_w  = "YES" if sig.get("open_to_work_flag") else "no"
        days    = days_since(sig.get("last_active_date"))
        rrr     = sig.get("recruiter_response_rate", 0)
        notice  = sig.get("notice_period_days", 0)

        t1 = ", ".join(cs.tier1_skills[:3]) if cs.tier1_skills else "(no tier-1)"
        flag_str = " ⚠HONEYPOT" if cs.honeypot_flags else ""

        # Colour-code obvious problems
        problem = ""
        company_lower = company.lower()
        if any(s in company_lower for s in IT_SERVICES_COMPANIES):
            problem = " ← IT SERVICES (should not be here)"
        if not any(t in title.lower() for t in
                   ("ml", "machine learning", "ai", "nlp", "data scientist",
                    "recommendation", "search", "ranking", "applied", "research")):
            problem = " ← NON-ML TITLE (review)"

        print(f"{rank:3d}  {cs.candidate_id}  {title:35}  {company:20}  "
              f"{yoe:4.1f}  {cs.composite:6.2f}{flag_str}{problem}")
        print(f"     {country:3}  {loc:15}  "
              f"car={cs.career_score:5.0f} sk={cs.skills_score:5.0f} "
              f"beh={cs.behavioral_score:5.0f} loc={cs.location_score:5.0f}  "
              f"open={open_w:3} days={days:4d} rrr={rrr:.0%} notice={notice:3d}d  {t1}")

        if cs.candidate_id in csv_reasoning:
            print(f"     REASON: {csv_reasoning[cs.candidate_id][:110]}")

        # Extra blank line every 10 for readability
        if rank % 10 == 0:
            print()

    # ── Quick health check ────────────────────────────────────────────────────
    top10 = scored[:10]
    print(f"\n{'='*80}")
    print("QUICK HEALTH CHECK")
    print(f"{'='*80}")

    ml_titles = sum(1 for cs in top10 if any(
        t in cs.candidate["profile"].get("current_title","").lower()
        for t in ("ml","machine learning","ai","nlp","data scientist",
                  "recommendation","search","ranking","applied","research")
    ) if cs.candidate)
    india_count = sum(1 for cs in top10 if
                      "india" in (cs.candidate["profile"].get("country","") or "").lower()
                      if cs.candidate)
    has_t1 = sum(1 for cs in top10 if cs.tier1_skills)
    flagged = sum(1 for cs in scored[:100] if cs.honeypot_flags)

    print(f"  Top-10 with ML/AI/NLP title:    {ml_titles}/10  {'✓' if ml_titles >= 7 else '✗ REVIEW'}")
    print(f"  Top-10 based in India:           {india_count}/10  {'✓' if india_count >= 6 else '⚠ check'}")
    print(f"  Top-10 with Tier-1 skills:       {has_t1}/10  {'✓' if has_t1 >= 8 else '✗ REVIEW'}")
    print(f"  Honeypot-flagged in top-100:     {flagged}/100  {'✓ SAFE' if flagged <= 10 else '✗ DANGER'}")

    scores = [cs.composite for cs in scored[:10]]
    gap = max(scores) - min(scores)
    print(f"  Top-10 score gap (rank1-rank10): {gap:.2f}  {'✓' if gap >= 5 else '⚠ very tight'}")


if __name__ == "__main__":
    main()
