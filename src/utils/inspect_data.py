from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import date
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TODAY = date.today()

# Helpers
def _days_since(date_str: str) -> int:
    try:
        return max(0, (TODAY - date.fromisoformat(date_str)).days)
    except Exception:
        return -1


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default
        
# Individual inspection functions
def inspect_basic(candidates: List[dict]) -> None:
    print(f"\n{'='*60}")
    print(f"DATASET OVERVIEW  ({len(candidates):,} candidates)")
    print(f"{'='*60}")

    # Required field coverage
    fields = ["candidate_id", "profile", "career_history", "education",
              "skills", "redrob_signals"]
    for f in fields:
        present = sum(1 for c in candidates if c.get(f))
        print(f"  {f:25s}: {present:,}/{len(candidates):,} ({100*present/len(candidates):.0f}%)")

    # Optional fields
    for f in ["certifications", "languages"]:
        present = sum(1 for c in candidates if c.get(f))
        print(f"  {f:25s}: {present:,}/{len(candidates):,} ({100*present/len(candidates):.0f}%)")


def inspect_career(candidates: List[dict], top_n: int = 20) -> None:
    print(f"\n{'='*60}")
    print("CAREER DISTRIBUTION")
    print(f"{'='*60}")

    titles    = [c["profile"].get("current_title", "?") for c in candidates]
    industries= [c["profile"].get("current_industry", "?") for c in candidates]
    companies = [c["profile"].get("current_company", "?") for c in candidates]
    sizes     = [c["profile"].get("current_company_size", "?") for c in candidates]
    yoe_vals  = [_safe_float(c["profile"].get("years_of_experience", 0)) for c in candidates]
    countries = [c["profile"].get("country", "?") for c in candidates]

    print(f"\nTop {top_n} titles:")
    for t, n in Counter(titles).most_common(top_n):
        print(f"  {n:5d}  {t}")

    print(f"\nTop 10 industries:")
    for ind, n in Counter(industries).most_common(10):
        print(f"  {n:5d}  {ind}")

    print(f"\nTop 10 companies:")
    for co, n in Counter(companies).most_common(10):
        print(f"  {n:5d}  {co}")

    print(f"\nCompany size distribution:")
    for sz, n in sorted(Counter(sizes).items()):
        print(f"  {n:5d}  {sz}")

    print(f"\nCountry distribution:")
    for co, n in Counter(countries).most_common(15):
        print(f"  {n:5d}  {co}")

    valid_yoe = [y for y in yoe_vals if y > 0]
    if valid_yoe:
        import statistics
        print(f"\nYoE: min={min(valid_yoe):.1f}  max={max(valid_yoe):.1f}  "
              f"mean={statistics.mean(valid_yoe):.1f}  "
              f"median={statistics.median(valid_yoe):.1f}")


def inspect_skills(candidates: List[dict], top_n: int = 30) -> None:
    print(f"\n{'='*60}")
    print("SKILLS DISTRIBUTION")
    print(f"{'='*60}")

    all_skills    = []
    credentialed  = []  # duration > 0 or endorsements > 0
    for c in candidates:
        for sk in c.get("skills", []):
            name = sk.get("name", "")
            all_skills.append(name)
            if sk.get("duration_months", 0) > 0 or sk.get("endorsements", 0) > 0:
                credentialed.append(name)

    print(f"\nTotal skill mentions: {len(all_skills):,}")
    print(f"Credentialed skill mentions: {len(credentialed):,} ({100*len(credentialed)/max(1,len(all_skills)):.0f}%)")

    print(f"\nTop {top_n} skills (all mentions):")
    for sk, n in Counter(all_skills).most_common(top_n):
        print(f"  {n:5d}  {sk}")

    print(f"\nTop {top_n} CREDENTIALED skills:")
    for sk, n in Counter(credentialed).most_common(top_n):
        print(f"  {n:5d}  {sk}")

    # Proficiency distribution
    profs = [sk.get("proficiency","?")
             for c in candidates for sk in c.get("skills", [])]
    print(f"\nProficiency distribution:")
    for p, n in Counter(profs).most_common():
        print(f"  {n:5d}  {p}")


def inspect_behavioral(candidates: List[dict]) -> None:
    print(f"\n{'='*60}")
    print("BEHAVIORAL SIGNALS")
    print(f"{'='*60}")

    open_work = [c for c in candidates if c["redrob_signals"].get("open_to_work_flag")]
    active_30 = [c for c in candidates
                 if _days_since(c["redrob_signals"].get("last_active_date","2000-01-01")) <= 30]
    active_90 = [c for c in candidates
                 if _days_since(c["redrob_signals"].get("last_active_date","2000-01-01")) <= 90]

    print(f"\n  Open to work: {len(open_work):,}/{len(candidates):,} "
          f"({100*len(open_work)/len(candidates):.0f}%)")
    print(f"  Active in last 30d: {len(active_30):,}/{len(candidates):,} "
          f"({100*len(active_30)/len(candidates):.0f}%)")
    print(f"  Active in last 90d: {len(active_90):,}/{len(candidates):,} "
          f"({100*len(active_90)/len(candidates):.0f}%)")

    # Notice period
    notices = [c["redrob_signals"].get("notice_period_days", 0) for c in candidates]
    from collections import Counter
    buckets = Counter()
    for n in notices:
        if n == 0: buckets["0d (immediate)"] += 1
        elif n <= 30: buckets["1-30d"] += 1
        elif n <= 60: buckets["31-60d"] += 1
        elif n <= 90: buckets["61-90d"] += 1
        else: buckets[">90d"] += 1
    print("\n  Notice period:")
    for bucket in ["0d (immediate)", "1-30d", "31-60d", "61-90d", ">90d"]:
        print(f"    {buckets[bucket]:5d}  {bucket}")

    # RRR distribution
    rrrs = [_safe_float(c["redrob_signals"].get("recruiter_response_rate", 0.5))
            for c in candidates]
    rrr_buckets = Counter()
    for r in rrrs:
        if r < 0.10:   rrr_buckets["<10%"] += 1
        elif r < 0.25: rrr_buckets["10-25%"] += 1
        elif r < 0.50: rrr_buckets["25-50%"] += 1
        elif r < 0.75: rrr_buckets["50-75%"] += 1
        else:          rrr_buckets[">=75%"] += 1
    print("\n  Recruiter response rate:")
    for bucket in ["<10%","10-25%","25-50%","50-75%",">=75%"]:
        print(f"    {rrr_buckets[bucket]:5d}  {bucket}")

    # GitHub
    githubs = [_safe_float(c["redrob_signals"].get("github_activity_score", -1))
               for c in candidates]
    no_github = sum(1 for g in githubs if g < 0)
    with_github = [g for g in githubs if g >= 0]
    print(f"\n  No GitHub linked: {no_github:,}/{len(candidates):,}")
    if with_github:
        import statistics
        print(f"  GitHub score (those linked): "
              f"min={min(with_github):.0f}  "
              f"mean={statistics.mean(with_github):.0f}  "
              f"max={max(with_github):.0f}")


def inspect_data_quality(candidates: List[dict]) -> None:
    print(f"\n{'='*60}")
    print("DATA QUALITY ISSUES")
    print(f"{'='*60}")

    issues = []

    for c in candidates:
        cid = c.get("candidate_id", "?")
        sig = c.get("redrob_signals", {})

        # Salary inversion
        sal = sig.get("expected_salary_range_inr_lpa", {}) or {}
        sal_min = _safe_float(sal.get("min", 0))
        sal_max = _safe_float(sal.get("max", 0))
        if sal_min > sal_max and sal_max > 0:
            issues.append(f"  {cid}: salary inverted (min={sal_min}, max={sal_max})")

        # YoE vs career total
        yoe = _safe_float(c.get("profile", {}).get("years_of_experience", 0))
        hist = c.get("career_history", [])
        career_months = sum(r.get("duration_months", 0) for r in hist)
        if career_months > 0 and abs(yoe - career_months / 12) > 5:
            issues.append(f"  {cid}: YoE={yoe:.1f} but career_total={career_months/12:.1f}yr")

        # All-identical descriptions
        descs = [r.get("description", "")[:80] for r in hist]
        if len(set(descs)) == 1 and len(descs) > 1 and descs[0]:
            issues.append(f"  {cid}: all {len(descs)} career descriptions identical")

    if issues:
        print(f"\nFound {len(issues)} issues:")
        for iss in issues[:30]:
            print(iss)
        if len(issues) > 30:
            print(f"  ... and {len(issues)-30} more")
    else:
        print("  No data quality issues found.")

# Entry point
def run_inspection(candidates: List[dict]) -> None:
    inspect_basic(candidates)
    inspect_career(candidates)
    inspect_skills(candidates)
    inspect_behavioral(candidates)
    inspect_data_quality(candidates)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect candidate dataset")
    parser.add_argument("--candidates", required=True, help="Path to candidates file")
    parser.add_argument("--max", type=int, default=None, help="Max records to inspect")
    args = parser.parse_args()

    from src.data.loader import load_candidates
    candidates = load_candidates(args.candidates, max_records=args.max)
    run_inspection(candidates)
