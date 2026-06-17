from __future__ import annotations

import math
from typing import Dict, List, Tuple

from src.config.settings import (
    TIER1_SKILLS,
    TIER2_SKILLS,
    TIER3_SKILLS,
    ANTI_SKILLS,
    PROFICIENCY_VALUE,
)

# Credibility helpers
def _duration_factor(months: int) -> float:
    """Saturates at 24 months (2 years of use = full credit)."""
    return min(1.0, months / 24.0) if months > 0 else 0.0


def _endorsement_factor(n: int) -> float:
    """Saturates at 20 endorsements."""
    return min(1.0, n / 20.0) if n > 0 else 0.0


def _credibility(duration: int, endorsements: int, assessment: float | None) -> float:
    d = _duration_factor(duration)
    e = _endorsement_factor(endorsements)

    # If both are zero -> uncredentialed; return 0
    if d == 0.0 and e == 0.0:
        return 0.0

    # Geometric mean avoids inflating a skill endorsed 50 times but used 0 months
    ge = math.sqrt(d * e) if (d > 0 and e > 0) else max(d, e) * 0.5

    if assessment is not None:
        # Blend: 80% duration/endorsement, 20% assessment
        ge = 0.80 * ge + 0.20 * (assessment / 100.0)

    return min(1.0, ge)

# Tier classifier
def _classify_skill(name_lower: str) -> Tuple[int, float]:
    if name_lower in TIER1_SKILLS or any(t in name_lower for t in TIER1_SKILLS):
        return 1, 15.0
    if name_lower in TIER2_SKILLS or any(t in name_lower for t in TIER2_SKILLS):
        return 2, 8.0
    if name_lower in TIER3_SKILLS or any(t in name_lower for t in TIER3_SKILLS):
        return 3, 3.0
    if name_lower in ANTI_SKILLS:
        return -1, 0.0
    return 0, 1.0

# Public API
def score_skills(
    skills: List[Dict],
    assessments: Dict[str, float],
) -> Tuple[float, Dict]:
    tier1_pts = 0.0
    tier2_pts = 0.0
    tier3_pts = 0.0
    anti_count = 0
    credentialed_count = 0
    matched_tier1_names: List[str] = []
    matched_tier2_names: List[str] = []

    for sk in skills:
        name_lower = sk["name"]    
        name_raw   = sk.get("name_raw", sk["name"])
        proficiency_val = PROFICIENCY_VALUE.get(sk["proficiency"], 0.30)
        duration    = sk["duration_months"]
        endorsements= sk["endorsements"]

        # Assessment lookup
        assessment = None
        for assess_key, assess_val in assessments.items():
            if assess_key.lower() == name_raw.lower():
                assessment = float(assess_val)
                break

        # Credibility gate
        cred = _credibility(duration, endorsements, assessment)
        if cred == 0.0:
            if name_lower in ANTI_SKILLS:
                anti_count += 1
            continue

        credentialed_count += 1
        skill_value = proficiency_val * cred

        tier, weight = _classify_skill(name_lower)

        if tier == 1:
            tier1_pts += skill_value * weight
            matched_tier1_names.append(name_raw)
        elif tier == 2:
            tier2_pts += skill_value * weight
            matched_tier2_names.append(name_raw)
        elif tier == 3:
            tier3_pts += skill_value * weight
        elif tier == -1:
            anti_count += 1

    # Cap tiers to prevent runaway scoring
    raw = (
        min(60.0, tier1_pts)
        + min(30.0, tier2_pts) * 0.80
        + min(10.0, tier3_pts) * 0.50
    )

    # Anti-skill penalty: if non-technical skills dominate credentialed skills
    if credentialed_count > 0 and anti_count > credentialed_count * 0.55:
        raw *= 0.40

    final = min(100.0, max(0.0, raw))

    breakdown = {
        "tier1_pts":     round(tier1_pts, 2),
        "tier2_pts":     round(tier2_pts, 2),
        "tier3_pts":     round(tier3_pts, 2),
        "anti_count":    anti_count,
        "credentialed":  credentialed_count,
        "tier1_skills":  matched_tier1_names[:6],
        "tier2_skills":  matched_tier2_names[:4],
    }

    return round(final, 2), breakdown


def top_tier1_skills(
    skills: List[Dict],
    assessments: Dict[str, float],
    n: int = 4,
) -> List[str]:
    """Return the top N credentialed Tier-1 skill names (for reasoning text)."""
    tier1: List[Tuple[float, str]] = []

    for sk in skills:
        name_lower = sk["name"]
        name_raw   = sk.get("name_raw", sk["name"])
        duration    = sk["duration_months"]
        endorsements= sk["endorsements"]

        assessment = None
        for k, v in assessments.items():
            if k.lower() == name_raw.lower():
                assessment = float(v)
                break

        cred = _credibility(duration, endorsements, assessment)
        if cred == 0.0:
            continue

        tier, _ = _classify_skill(name_lower)
        if tier == 1:
            proficiency_val = PROFICIENCY_VALUE.get(sk["proficiency"], 0.30)
            tier1.append((proficiency_val * cred, name_raw))

    tier1.sort(key=lambda x: -x[0])
    return [name for _, name in tier1[:n]]


def top_tier2_skills(
    skills: List[Dict],
    assessments: Dict[str, float],
    n: int = 3,
) -> List[str]:
    """Return the top N credentialed Tier-2 skill names."""
    tier2: List[Tuple[float, str]] = []

    for sk in skills:
        name_lower = sk["name"]
        name_raw   = sk.get("name_raw", sk["name"])
        duration    = sk["duration_months"]
        endorsements= sk["endorsements"]

        assessment = None
        for k, v in assessments.items():
            if k.lower() == name_raw.lower():
                assessment = float(v)
                break

        cred = _credibility(duration, endorsements, assessment)
        if cred == 0.0:
            continue

        tier, _ = _classify_skill(name_lower)
        if tier == 2:
            proficiency_val = PROFICIENCY_VALUE.get(sk["proficiency"], 0.30)
            tier2.append((proficiency_val * cred, name_raw))

    tier2.sort(key=lambda x: -x[0])
    return [name for _, name in tier2[:n]]
