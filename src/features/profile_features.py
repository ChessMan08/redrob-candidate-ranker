from __future__ import annotations

from typing import Dict, List, Tuple

from src.config.settings import (
    PREFERRED_CITIES,
    RELEVANT_EDU_FIELDS,
    EDU_TIER_SCORE,
)

# Experience score
def score_experience(years_of_experience: float) -> Tuple[float, Dict]:
    yoe = years_of_experience

    if   6.0 <= yoe <= 8.0:  score = 100.0
    elif 5.0 <= yoe < 6.0:   score = 88.0
    elif 8.0 < yoe <= 9.0:   score = 88.0
    elif 4.0 <= yoe < 5.0:   score = 73.0
    elif 9.0 < yoe <= 11.0:  score = 73.0
    elif 3.0 <= yoe < 4.0:   score = 57.0
    elif 11.0 < yoe <= 13.0: score = 57.0
    elif yoe > 13.0:          score = 43.0   # over-experienced;
    elif yoe >= 2.0:          score = 28.0
    else:                     score = 12.0

    return round(score, 2), {"years_of_experience": yoe}

# Location score
def score_location(profile: Dict, sig: Dict) -> Tuple[float, Dict]:
    location        = (profile.get("location") or "").lower()
    country         = (profile.get("country")  or "").lower()
    willing_relocate= sig.get("willing_to_relocate", False)

    in_india = "india" in country or country in ("in", "india")

    if in_india:
        city_match = any(city in location for city in PREFERRED_CITIES)
        if city_match:
            score = 100.0
            label = "preferred_city"
        elif willing_relocate:
            score = 72.0
            label = "india_will_relocate"
        else:
            score = 50.0
            label = "india_wrong_city_no_reloc"
    else:
        if willing_relocate:
            score = 30.0
            label = "outside_india_will_relocate"
        else:
            score = 12.0
            label = "outside_india_no_reloc"

    return round(score, 2), {
        "location": profile.get("location"),
        "country":  profile.get("country"),
        "label":    label,
        "willing_to_relocate": willing_relocate,
    }

# Education score
def score_education(education: List[Dict]) -> Tuple[float, Dict]:
    if not education:
        return 50.0, {"reason": "no_education_listed"}

    best_score = 30.0
    best_label  = ""

    for edu in education:
        tier      = edu.get("tier", "unknown")
        field     = (edu.get("field_of_study") or "").lower()
        degree    = (edu.get("degree") or "").lower()

        tier_pts  = EDU_TIER_SCORE.get(tier, 45)

        # Field relevance
        field_match = any(f in field for f in RELEVANT_EDU_FIELDS)
        field_bonus = +15.0 if field_match else -5.0

        # Degree level bonus
        degree_bonus = 0.0
        if any(d in degree for d in ("m.tech", "m.s.", "ms", "mtech", "m.e.", "me")):
            degree_bonus = 8.0 if field_match else 2.0
        elif "phd" in degree or "ph.d" in degree:
            degree_bonus = 10.0 if field_match else -5.0
            # PhD in non-CS (e.g. Mech Eng) is actually a mild negative here
        elif "b.tech" in degree or "b.e." in degree or "bachelor" in degree:
            degree_bonus = 3.0 if field_match else 0.0

        entry_score = tier_pts + field_bonus + degree_bonus
        entry_score = max(0.0, min(100.0, entry_score))

        if entry_score > best_score:
            best_score = entry_score
            best_label = f"{degree}_{tier}_{field[:20]}"

    return round(best_score, 2), {"best_edu": best_label}

# Salary fit (soft signal, small penalty for extreme mismatch)
def salary_fit_multiplier(sig: Dict) -> float:
    sal_max = sig.get("salary_max_lpa", 0.0)
    sal_min = sig.get("salary_min_lpa", 0.0)

    # Use the lower bound for fit calculation
    if sal_min > 80.0:
        return 0.90  # expecting very high salary, Series A may struggle
    if sal_min < 5.0 and sal_min > 0.0:
        return 0.93  # expecting very low -> likely mismatch on seniority

    return 1.0
