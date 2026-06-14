"""
career_features.py — Career trajectory scorer.

This is the single most important feature for this JD.

The JD explicitly disqualifies:
  - Entire careers at IT services / consulting (TCS, Infosys, Wipro, etc.)
  - Candidates who have never shipped a production ML system
  - "Pure architect" types who haven't written code in 18+ months

The JD positively values:
  - Product company experience
  - ML/search/ranking/recommendation titles
  - Smaller startup experience (51-500 headcount) in relevant industries

Algorithm:
  - Score each role [0-100] based on company type, industry, title
  - Weight by that role's share of total career tenure (time-weighted)
  - Return the weighted sum, clipped to [0, 100]
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.config.settings import (
    IT_SERVICES_COMPANIES,
    PRODUCT_COMPANIES,
    GOOD_INDUSTRIES,
    BAD_INDUSTRIES,
    ML_TITLE_TERMS,
    NON_IC_TITLE_TERMS,
    DOMAIN_KEYWORDS_IN_DESCRIPTION,
)


# ─────────────────────────────────────────────────────────────────────────────
# Company classifier
# ─────────────────────────────────────────────────────────────────────────────

def _classify_company(company_lower: str, company_size: str) -> Tuple[float, str]:
    """
    Returns (score_delta, label) for a company name.
    """
    # Exact / substring match against known lists
    if any(s in company_lower for s in IT_SERVICES_COMPANIES):
        return -30.0, "it_services"
    if any(s in company_lower for s in PRODUCT_COMPANIES):
        return +25.0, "product_known"

    # Size-based heuristic for unknowns
    if company_size in ("51-200", "201-500"):
        return +10.0, "product_likely_startup"
    if company_size in ("1-10", "11-50"):
        return +8.0, "product_very_small"
    if company_size in ("1001-5000", "5001-10000", "10001+"):
        return +3.0, "large_unknown"   # large company; could be product or services
    return 0.0, "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Industry classifier
# ─────────────────────────────────────────────────────────────────────────────

def _classify_industry(industry_lower: str) -> float:
    if any(g in industry_lower for g in GOOD_INDUSTRIES):
        return +12.0
    if any(b in industry_lower for b in BAD_INDUSTRIES):
        return -15.0
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Title classifier
# ─────────────────────────────────────────────────────────────────────────────

def _classify_title(title_lower: str) -> Tuple[float, str]:
    if any(t in title_lower for t in ML_TITLE_TERMS):
        return +22.0, "ml_title"
    if any(t in title_lower for t in NON_IC_TITLE_TERMS):
        return -15.0, "non_ic_title"
    # Data engineering / backend — adjacent, not negative
    if any(t in title_lower for t in ("data engineer", "backend", "platform", "software engineer", "full stack")):
        return +4.0, "adjacent_title"
    return 0.0, "neutral_title"


# ─────────────────────────────────────────────────────────────────────────────
# Description keyword signal (supplementary — catches unlabelled ML work)
# ─────────────────────────────────────────────────────────────────────────────

def _description_ml_signal(description: str) -> float:
    """
    Returns a small bonus [0, 10] if the role description mentions
    ML/retrieval/ranking keywords, even if the title doesn't.
    """
    desc_lower = description.lower()
    hits = sum(1 for kw in DOMAIN_KEYWORDS_IN_DESCRIPTION if kw in desc_lower)
    # Cap at 5 keywords for bonus purposes
    return min(10.0, hits * 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def score_career(career_history: List[Dict]) -> Tuple[float, Dict]:
    """
    Compute a weighted career trajectory score in [0, 100].

    Returns
    -------
    (score, breakdown)
      score     : float in [0, 100]
      breakdown : dict with per-role details for debugging
    """
    if not career_history:
        return 20.0, {"reason": "no_career_history"}

    total_months = max(1, sum(r.get("duration_months", 0) for r in career_history))
    weighted_sum = 0.0
    breakdown: List[Dict] = []

    it_services_only = True  # tracks if ALL tenure is at IT services
    ml_title_months  = 0     # months spent in ML/AI roles
    non_ml_title_months = 0  # months in non-ML titles (FE, Java, QA, etc.)

    for role in career_history:
        base = 50.0  # neutral starting point

        company_lower = (role.get("company") or "").lower()
        company_size  = role.get("company_size", "unknown")
        title_lower   = (role.get("title") or "").lower()
        industry_lower= (role.get("industry") or "").lower()
        description   = role.get("description") or ""
        duration      = role.get("duration_months", 0)

        company_delta, company_label = _classify_company(company_lower, company_size)
        industry_delta = _classify_industry(industry_lower)
        title_delta, title_label = _classify_title(title_lower)
        desc_bonus = _description_ml_signal(description)

        if company_label != "it_services":
            it_services_only = False

        # Track ML vs non-ML title tenure
        if title_label == "ml_title":
            ml_title_months += duration
        elif title_label in ("non_ic_title",) or any(
            t in title_lower for t in (
                "frontend", "java developer", "qa engineer", "qa ",
                "mobile developer", ".net developer", "devops",
                "android developer", "ios developer", "react developer",
            )
        ):
            non_ml_title_months += duration

        role_score = base + company_delta + industry_delta + title_delta + desc_bonus
        role_score = max(0.0, min(100.0, role_score))

        weight = duration / total_months
        weighted_sum += role_score * weight

        breakdown.append({
            "company":        role.get("company"),
            "title":          role.get("title"),
            "duration":       duration,
            "role_score":     round(role_score, 1),
            "weight":         round(weight, 3),
            "company_label":  company_label,
            "title_label":    title_label,
        })

    final = max(0.0, min(100.0, weighted_sum))

    # Extra penalty if 100% of career is at IT services firms
    if it_services_only and len(career_history) > 1:
        final *= 0.6
        breakdown.append({"note": "it_services_only_penalty"})

    # Title-coherence penalty: if the majority of tenure is in non-ML titles
    # (Frontend, Java, QA, Mobile, DevOps) with no ML title history,
    # the company-prestige boost is misleading — penalise it.
    if ml_title_months == 0 and non_ml_title_months > total_months * 0.5:
        final *= 0.75
        breakdown.append({"note": f"non_ml_title_majority_penalty "
                                  f"(ml={ml_title_months}m non_ml={non_ml_title_months}m)"})

    return round(final, 2), {"roles": breakdown, "total_months": total_months}


def has_ever_worked_at_product_company(career_history: List[Dict]) -> bool:
    """True if at least one role was at a non-IT-services company."""
    for role in career_history:
        company_lower = (role.get("company") or "").lower()
        _, label = _classify_company(company_lower, role.get("company_size", ""))
        if label != "it_services":
            return True
    return False


def years_in_ml_roles(career_history: List[Dict]) -> float:
    """Total tenure (in years) in ML/AI/search/ranking titles."""
    months = 0
    for role in career_history:
        title_lower = (role.get("title") or "").lower()
        delta, label = _classify_title(title_lower)
        if label == "ml_title":
            months += role.get("duration_months", 0)
    return months / 12.0
