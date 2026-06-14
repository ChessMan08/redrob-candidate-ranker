"""
behavioral_features.py — Platform availability and engagement scorer.

The JD explicitly says: "a perfect-on-paper candidate who hasn't logged in
for 6 months and has a 5% recruiter response rate is, for hiring purposes,
not actually available. Down-weight them appropriately."

This module encodes that logic precisely.

Two sub-scores are returned:
  1. availability_score  — is this person reachable RIGHT NOW?
  2. engagement_score    — external validation (GitHub, saved by recruiters, etc.)

They're blended 60/40 in the composite.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, Tuple

from src.config.settings import ACTIVITY_SCORE, TODAY


# ─────────────────────────────────────────────────────────────────────────────
# Availability score
# ─────────────────────────────────────────────────────────────────────────────

def _days_since_active(last_active: date | None) -> int:
    if last_active is None:
        return 999
    return max(0, (TODAY - last_active).days)


def _activity_delta(days: int) -> float:
    for threshold, delta in ACTIVITY_SCORE:
        if days <= threshold:
            return float(delta)
    return float(ACTIVITY_SCORE[-1][1])


def score_availability(sig: Dict) -> Tuple[float, Dict]:
    """
    Score [0, 100] capturing how likely the candidate is to respond
    to a recruiter approach today.
    """
    score = 50.0
    breakdown = {}

    # 1. Last active date (major signal — 35 pts range)
    days = _days_since_active(sig.get("last_active_date"))
    activity_d = _activity_delta(days)
    score += activity_d
    breakdown["days_since_active"] = days
    breakdown["activity_delta"]    = activity_d

    # 2. Open-to-work flag (15 pts)
    if sig.get("open_to_work_flag", False):
        score += 15.0
        breakdown["open_to_work"] = True
    else:
        score -= 5.0
        breakdown["open_to_work"] = False

    # 3. Recruiter response rate (20 pts range)
    rrr = sig.get("recruiter_response_rate", 0.5)
    if rrr >= 0.75:
        rrr_d = +15.0
    elif rrr >= 0.50:
        rrr_d = +8.0
    elif rrr >= 0.25:
        rrr_d = +0.0
    elif rrr >= 0.10:
        rrr_d = -10.0
    else:
        rrr_d = -20.0   # ghost-level response rate
    score += rrr_d
    breakdown["recruiter_response_rate"] = rrr
    breakdown["rrr_delta"] = rrr_d

    # 4. Notice period (15 pts range)
    notice = sig.get("notice_period_days", 90)
    if notice <= 30:
        notice_d = +10.0
    elif notice <= 60:
        notice_d = +5.0
    elif notice <= 90:
        notice_d = -3.0
    elif notice <= 120:
        notice_d = -10.0
    else:
        notice_d = -18.0   # 120+ days: severe friction for a startup
    score += notice_d
    breakdown["notice_period_days"] = notice
    breakdown["notice_delta"] = notice_d

    # 5. Profile completeness (soft ±7.5 pts)
    completeness = sig.get("profile_completeness_score", 60.0)
    score += (completeness - 60.0) * 0.25
    breakdown["profile_completeness"] = completeness

    # 6. Avg response time (small signal)
    rt = sig.get("avg_response_time_hours", 48.0)
    if rt <= 4:
        score += 5.0
    elif rt <= 24:
        score += 2.0
    elif rt > 168:   # > 1 week
        score -= 5.0
    breakdown["avg_response_time_hours"] = rt

    return round(min(100.0, max(0.0, score)), 2), breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Engagement / external-validation score
# ─────────────────────────────────────────────────────────────────────────────

def score_engagement(sig: Dict) -> Tuple[float, Dict]:
    """
    Score [0, 100] capturing external validation and engagement quality.
    """
    score = 50.0
    breakdown = {}

    # 1. GitHub activity (JD says "external validation" matters for this role)
    github = sig.get("github_activity_score", -1.0)
    if github == -1.0:
        # No GitHub linked — mild negative (JD: needs external validation)
        github_d = -5.0
    elif github >= 70:
        github_d = +25.0
    elif github >= 45:
        github_d = +15.0
    elif github >= 20:
        github_d = +7.0
    else:
        github_d = +2.0
    score += github_d
    breakdown["github_score"] = github
    breakdown["github_delta"] = github_d

    # 2. Saved by recruiters in last 30d (peer validation)
    saved = sig.get("saved_by_recruiters_30d", 0)
    if saved >= 15:
        saved_d = +15.0
    elif saved >= 8:
        saved_d = +9.0
    elif saved >= 3:
        saved_d = +4.0
    else:
        saved_d = 0.0
    score += saved_d
    breakdown["saved_by_recruiters"] = saved

    # 3. Interview completion rate (reliability)
    icr = sig.get("interview_completion_rate", 0.7)
    if icr >= 0.85:
        icr_d = +10.0
    elif icr >= 0.60:
        icr_d = +3.0
    elif icr < 0.40:
        icr_d = -12.0
    else:
        icr_d = 0.0
    score += icr_d
    breakdown["interview_completion_rate"] = icr

    # 4. Applications submitted (shows active job search)
    apps = sig.get("applications_submitted_30d", 0)
    if apps >= 5:
        score += 5.0
    elif apps >= 2:
        score += 2.0

    # 5. Verified contact info (basic trust signal)
    verified = (sig.get("verified_email", False) and sig.get("verified_phone", False))
    if verified:
        score += 5.0
    elif not sig.get("verified_email", False) and not sig.get("verified_phone", False):
        score -= 5.0
    breakdown["verified_contacts"] = verified

    # 6. LinkedIn connected
    if sig.get("linkedin_connected", False):
        score += 3.0

    return round(min(100.0, max(0.0, score)), 2), breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Combined behavioral score
# ─────────────────────────────────────────────────────────────────────────────

def score_behavioral(sig: Dict) -> Tuple[float, Dict]:
    """
    Blended behavioral score (60% availability, 40% engagement).
    """
    avail, avail_bd  = score_availability(sig)
    engage, engage_bd = score_engagement(sig)
    blended = 0.60 * avail + 0.40 * engage
    return round(blended, 2), {"availability": avail_bd, "engagement": engage_bd}


# ─────────────────────────────────────────────────────────────────────────────
# Hard-gate multiplier (applied AFTER composite scoring)
# ─────────────────────────────────────────────────────────────────────────────

def behavioral_multiplier(sig: Dict) -> float:
    """
    Returns a scalar in [0.5, 1.0] to multiply the composite score.
    Only applied in extreme unavailability cases — the signal is already
    encoded in behavioral_score, but these cases warrant an extra kick.
    """
    multiplier = 1.0

    days = _days_since_active(sig.get("last_active_date"))
    rrr  = sig.get("recruiter_response_rate", 0.5)
    open_work = sig.get("open_to_work_flag", False)

    # Dark for > 1 year AND not actively looking
    if days > 365 and not open_work:
        multiplier *= 0.65

    # Ghost-level response rate
    if rrr < 0.05:
        multiplier *= 0.75

    # Both: severe penalty
    if days > 365 and not open_work and rrr < 0.10:
        multiplier = min(multiplier, 0.50)

    return multiplier
