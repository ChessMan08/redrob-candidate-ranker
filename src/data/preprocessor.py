from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Helpers
def _parse_date(val: Any, field_name: str = "") -> Optional[date]:
    if val is None:
        return None
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(val, fmt).date()
            except ValueError:
                continue
        logger.debug("Cannot parse date %r for field %s", val, field_name)
        return None
    return None


def _safe_str(val: Any, default: str = "") -> str:
    if val is None:
        return default
    return str(val).strip()


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default

# Sub-cleaners
def _clean_profile(profile: Dict) -> Dict:
    return {
        "anonymized_name":    _safe_str(profile.get("anonymized_name")),
        "headline":           _safe_str(profile.get("headline")),
        "summary":            _safe_str(profile.get("summary")),
        "location":           _safe_str(profile.get("location")).lower(),
        "country":            _safe_str(profile.get("country")).lower(),
        "years_of_experience":_safe_float(profile.get("years_of_experience"), 0.0),
        "current_title":      _safe_str(profile.get("current_title")),
        "current_company":    _safe_str(profile.get("current_company")),
        "current_company_size": _safe_str(profile.get("current_company_size"), "unknown"),
        "current_industry":   _safe_str(profile.get("current_industry")),
    }


def _clean_career_history(history: List[Dict]) -> List[Dict]:
    cleaned = []
    for role in (history or []):
        start = _parse_date(role.get("start_date"), "start_date")
        end   = _parse_date(role.get("end_date"),   "end_date")
        duration = _safe_int(role.get("duration_months"), 0)

        # Sanity-check duration vs dates
        if start and end and duration == 0:
            months_approx = (end.year - start.year) * 12 + (end.month - start.month)
            duration = max(0, months_approx)

        cleaned.append({
            "company":      _safe_str(role.get("company")),
            "title":        _safe_str(role.get("title")),
            "start_date":   start,
            "end_date":     end,
            "duration_months": duration,
            "is_current":   bool(role.get("is_current", False)),
            "industry":     _safe_str(role.get("industry")).lower(),
            "company_size": _safe_str(role.get("company_size"), "unknown"),
            "description":  _safe_str(role.get("description")),
        })
    return cleaned


def _clean_education(education: List[Dict]) -> List[Dict]:
    cleaned = []
    for edu in (education or []):
        cleaned.append({
            "institution":    _safe_str(edu.get("institution")),
            "degree":         _safe_str(edu.get("degree")).lower(),
            "field_of_study": _safe_str(edu.get("field_of_study")).lower(),
            "start_year":     _safe_int(edu.get("start_year"), 0),
            "end_year":       _safe_int(edu.get("end_year"), 0),
            "grade":          edu.get("grade"),
            "tier":           _safe_str(edu.get("tier"), "unknown"),
        })
    return cleaned


def _clean_skills(skills: List[Dict]) -> List[Dict]:
    cleaned = []
    seen_names: set[str] = set()
    for sk in (skills or []):
        name = _safe_str(sk.get("name")).lower()
        if not name or name in seen_names:
            continue 
        seen_names.add(name)
        cleaned.append({
            "name":           name,
            "name_raw":       _safe_str(sk.get("name")), 
            "proficiency":    _safe_str(sk.get("proficiency"), "beginner").lower(),
            "endorsements":   max(0, _safe_int(sk.get("endorsements"), 0)),
            "duration_months":max(0, _safe_int(sk.get("duration_months"), 0)),
        })
    return cleaned


def _clean_certifications(certs: List[Dict]) -> List[Dict]:
    return [
        {
            "name":   _safe_str(c.get("name")),
            "issuer": _safe_str(c.get("issuer")),
            "year":   _safe_int(c.get("year"), 0),
        }
        for c in (certs or [])
        if c.get("name")
    ]


def _clean_languages(langs: List[Dict]) -> List[Dict]:
    return [
        {
            "language":    _safe_str(l.get("language")),
            "proficiency": _safe_str(l.get("proficiency"), "basic").lower(),
        }
        for l in (langs or [])
        if l.get("language")
    ]


def _clean_signals(sig: Dict) -> Dict:
    # Fix inverted salary
    sal = sig.get("expected_salary_range_inr_lpa", {}) or {}
    sal_min = _safe_float(sal.get("min"), 0.0)
    sal_max = _safe_float(sal.get("max"), 0.0)
    if sal_min > sal_max and sal_max > 0:
        sal_min, sal_max = sal_max, sal_min

    # github_activity_score: -1 means no GitHub linked
    github = _safe_float(sig.get("github_activity_score"), -1.0)

    # offer_acceptance_rate: -1 means no history
    oar = _safe_float(sig.get("offer_acceptance_rate"), -1.0)

    return {
        "profile_completeness_score": _safe_float(sig.get("profile_completeness_score"), 50.0),
        "signup_date":      _parse_date(sig.get("signup_date"), "signup_date"),
        "last_active_date": _parse_date(sig.get("last_active_date"), "last_active_date"),
        "open_to_work_flag": bool(sig.get("open_to_work_flag", False)),
        "profile_views_received_30d":  max(0, _safe_int(sig.get("profile_views_received_30d"), 0)),
        "applications_submitted_30d":  max(0, _safe_int(sig.get("applications_submitted_30d"), 0)),
        "recruiter_response_rate":     min(1.0, max(0.0, _safe_float(sig.get("recruiter_response_rate"), 0.5))),
        "avg_response_time_hours":     max(0.0, _safe_float(sig.get("avg_response_time_hours"), 48.0)),
        "skill_assessment_scores":     dict(sig.get("skill_assessment_scores") or {}),
        "connection_count":            max(0, _safe_int(sig.get("connection_count"), 0)),
        "endorsements_received":       max(0, _safe_int(sig.get("endorsements_received"), 0)),
        "notice_period_days":          min(180, max(0, _safe_int(sig.get("notice_period_days"), 90))),
        "salary_min_lpa": sal_min,
        "salary_max_lpa": sal_max,
        "preferred_work_mode": _safe_str(sig.get("preferred_work_mode"), "flexible"),
        "willing_to_relocate": bool(sig.get("willing_to_relocate", False)),
        "github_activity_score":      github,
        "search_appearance_30d":      max(0, _safe_int(sig.get("search_appearance_30d"), 0)),
        "saved_by_recruiters_30d":    max(0, _safe_int(sig.get("saved_by_recruiters_30d"), 0)),
        "interview_completion_rate":  min(1.0, max(0.0, _safe_float(sig.get("interview_completion_rate"), 0.7))),
        "offer_acceptance_rate":      oar,
        "verified_email":   bool(sig.get("verified_email", False)),
        "verified_phone":   bool(sig.get("verified_phone", False)),
        "linkedin_connected": bool(sig.get("linkedin_connected", False)),
    }

# Public API
def clean_candidate(raw: Dict) -> Dict:
    return {
        "candidate_id":   _safe_str(raw.get("candidate_id")),
        "profile":        _clean_profile(raw.get("profile") or {}),
        "career_history": _clean_career_history(raw.get("career_history") or []),
        "education":      _clean_education(raw.get("education") or []),
        "skills":         _clean_skills(raw.get("skills") or []),
        "certifications": _clean_certifications(raw.get("certifications") or []),
        "languages":      _clean_languages(raw.get("languages") or []),
        "redrob_signals": _clean_signals(raw.get("redrob_signals") or {}),
    }


def clean_candidates(raws: List[Dict]) -> List[Dict]:
    """Vectorised version of clean_candidate."""
    return [clean_candidate(r) for r in raws]
