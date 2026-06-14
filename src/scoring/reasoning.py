"""
reasoning.py — Generate 1-2 sentence reasoning strings for each ranked candidate.

Design rules (directly from submission_spec.md Stage 4 requirements):
  1. Every claim MUST be grounded in actual candidate data (no hallucination).
  2. Reference specific facts: title, YoE, company, named skills, signal values.
  3. Connect to specific JD requirements (not generic praise).
  4. Acknowledge concerns honestly (high notice period, bad response rate, etc.).
  5. Tone must match the rank (rank-1 should read positively; rank-95 honestly).
  6. Each reasoning must be substantially different (not templated).

Implementation: purely deterministic — values are pulled from the scored
CandidateScore object, which holds the original candidate dict.  No LLM calls.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from src.scoring.composite import CandidateScore
from src.config.settings import (
    IT_SERVICES_COMPANIES,
    PREFERRED_CITIES,
    TODAY,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _days_since(d: Optional[date]) -> int:
    if d is None:
        return 999
    return max(0, (TODAY - d).days)


def _company_context(company: str, industry: str) -> str:
    company_lower = (company or "").lower()
    industry_lower = (industry or "").lower()
    is_it = any(s in company_lower for s in IT_SERVICES_COMPANIES)
    if is_it:
        return f"IT services firm ({company})"
    if industry_lower:
        return f"{company} ({industry})"
    return company or "Unknown"


def _location_phrase(profile: dict, sig: dict) -> str:
    location = profile.get("location") or ""
    country  = (profile.get("country") or "").lower()
    relocate = sig.get("willing_to_relocate", False)
    in_india = "india" in country

    if in_india:
        preferred = any(city in location.lower() for city in PREFERRED_CITIES)
        if preferred:
            return f"based in {location} (preferred region)"
        elif relocate:
            return f"India-based ({location}), willing to relocate"
        else:
            return f"India-based ({location}), not open to relocation"
    else:
        ctry_display = profile.get("country", "unknown")
        return f"outside India ({ctry_display}){', willing to relocate' if relocate else ''}"


def _availability_phrase(sig: dict) -> str:
    days       = _days_since(sig.get("last_active_date"))
    open_work  = sig.get("open_to_work_flag", False)
    rrr        = sig.get("recruiter_response_rate", 0.5)
    notice     = sig.get("notice_period_days", 90)

    parts = []
    if open_work and days <= 14:
        parts.append("actively job-hunting")
    elif open_work:
        parts.append("open to work")
    else:
        parts.append("not marked open-to-work")

    parts.append(f"last active {days}d ago")

    if rrr < 0.10:
        parts.append(f"very low recruiter response rate ({rrr:.0%})")
    elif rrr >= 0.70:
        parts.append(f"high response rate ({rrr:.0%})")

    if notice <= 30:
        parts.append(f"available quickly ({notice}d notice)")
    elif notice > 90:
        parts.append(f"long notice period ({notice}d)")

    return "; ".join(parts)


def _concern_phrase(cs: CandidateScore) -> Optional[str]:
    """
    Returns a concern string if there's something honest to flag,
    or None if no significant concerns.
    """
    profile = cs.candidate["profile"]
    sig     = cs.candidate["redrob_signals"]
    career  = cs.candidate["career_history"]

    concerns = []

    # IT services background
    it_only = all(
        any(s in (r.get("company") or "").lower() for s in IT_SERVICES_COMPANIES)
        for r in career
    )
    if it_only and career:
        concerns.append("entire career at IT services firms")

    # Not in India with no relocation
    country = (profile.get("country") or "").lower()
    if "india" not in country and not sig.get("willing_to_relocate", False):
        concerns.append("outside India, no relocation interest")

    # Notice period
    notice = sig.get("notice_period_days", 90)
    if notice > 90:
        concerns.append(f"long notice period ({notice}d)")

    # Not open to work + inactive
    days = _days_since(sig.get("last_active_date"))
    if not sig.get("open_to_work_flag", False) and days > 180:
        concerns.append("platform inactive >6 months and not open-to-work")

    # Honeypot flags
    if cs.honeypot_flags:
        concerns.append("some profile signals are inconsistent")

    # YoE mismatch
    yoe = profile.get("years_of_experience", 0.0)
    if yoe > 13:
        concerns.append(f"senior profile ({yoe:.0f}yr) — JD notes risk of over-engineering")

    if not concerns:
        return None
    return concerns[0]  # Surface only the top concern to keep it concise


# ─────────────────────────────────────────────────────────────────────────────
# Main reasoning builder
# ─────────────────────────────────────────────────────────────────────────────

def generate_reasoning(cs: CandidateScore, rank: int) -> str:
    """
    Returns a 1-2 sentence reasoning string grounded in candidate data.
    """
    c       = cs.candidate
    profile = c["profile"]
    sig     = c["redrob_signals"]
    career  = c["career_history"]

    title   = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    industry= profile.get("current_industry", "")
    yoe     = profile.get("years_of_experience", 0.0)

    # ── Sentence 1: core fit summary ──────────────────────────────────────────

    # Rank-adjusted opening phrase
    if rank <= 5:
        opening = "Strong match"
    elif rank <= 20:
        opening = "Good fit"
    elif rank <= 50:
        opening = "Moderate fit"
    else:
        opening = "Marginal fit"

    # Skill signal
    if cs.tier1_skills:
        skill_phrase = f"retrieval/ranking skills include {', '.join(cs.tier1_skills[:3])}"
    elif cs.tier2_skills:
        skill_phrase = f"ML skills include {', '.join(cs.tier2_skills[:2])}"
    else:
        skill_phrase = "limited ML/retrieval skills for this JD"

    ctx = _company_context(company, industry)
    s1  = f"{opening}: {title} with {yoe:.1f}yr at {ctx}; {skill_phrase}."

    # ── Sentence 2: availability + concern ───────────────────────────────────

    avail_phrase = _availability_phrase(sig)
    concern      = _concern_phrase(cs)

    if concern:
        s2 = f"Signals: {avail_phrase}. Concern: {concern}."
    else:
        loc_phrase = _location_phrase(profile, sig)
        s2 = f"Availability: {avail_phrase}; {loc_phrase}."

    return f"{s1} {s2}"


# ─────────────────────────────────────────────────────────────────────────────
# Batch reasoning for the top-N submission rows
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_reasoning(
    scored: List[CandidateScore],
    top_n: int = 100,
) -> List[str]:
    """
    Returns reasoning strings for the top_n candidates (already sorted).
    """
    return [
        generate_reasoning(cs, rank=i + 1)
        for i, cs in enumerate(scored[:top_n])
    ]
