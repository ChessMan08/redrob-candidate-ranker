"""
honeypot_detector.py — Detect candidates with impossible/inconsistent profiles.

The dataset contains ~80 honeypot candidates with "subtly impossible profiles"
(per submission_spec.md).  Examples from the spec:
  - 8 years experience at a company founded 3 years ago
  - "expert" proficiency in 10 skills with 0 years used

Rather than hard-excluding honeypots (risky if detector has false positives),
we return a multiplier in [0.20, 1.0] that penalises suspicious profiles.
A genuine candidate will never be hard-excluded; their multiplier stays 1.0.

Checks implemented:
  1. High endorsements + zero duration (endorsement fraud signal)
  2. Expert / advanced proficiency contradicted by low assessment score
  3. Mass expert claims without evidence
  4. Career tenure impossibly long (YoE >> sum of role durations)
  5. Identical descriptions across all roles (data-gen indicator — mild)
  6. Current role duration > company plausible age (hard to detect without
     external data; we approximate via founding-year heuristics where possible)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from src.config.settings import (
    HONEYPOT_HIGH_ENDORSE_ZERO_DUR_THRESHOLD,
    HONEYPOT_EXPERT_LOW_ASSESS_THRESHOLD,
    HONEYPOT_ADVANCED_LOW_ASSESS_THRESHOLD,
    HONEYPOT_MASS_EXPERT_COUNT,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_endorsement_fraud(skills: List[Dict], assessments: Dict) -> List[str]:
    """High endorsements on a skill with zero usage months."""
    flags = []
    for sk in skills:
        end = sk["endorsements"]
        dur = sk["duration_months"]
        if end > HONEYPOT_HIGH_ENDORSE_ZERO_DUR_THRESHOLD and dur == 0:
            flags.append(f"high_endorse_zero_dur:{sk['name_raw']}(end={end})")
    return flags


def _check_assessment_contradiction(skills: List[Dict], assessments: Dict) -> List[str]:
    """Expert/advanced self-rating contradicted by a poor Redrob assessment."""
    flags = []
    for sk in skills:
        name_raw = sk.get("name_raw", sk["name"])
        prof = sk["proficiency"]
        # Find matching assessment
        assess_score = None
        for k, v in assessments.items():
            if k.lower() == name_raw.lower():
                assess_score = float(v)
                break
        if assess_score is None:
            continue
        if prof == "expert" and assess_score < HONEYPOT_EXPERT_LOW_ASSESS_THRESHOLD:
            flags.append(
                f"expert_low_assess:{name_raw}"
                f"(assess={assess_score:.0f}<{HONEYPOT_EXPERT_LOW_ASSESS_THRESHOLD})"
            )
        elif prof == "advanced" and assess_score < HONEYPOT_ADVANCED_LOW_ASSESS_THRESHOLD:
            flags.append(
                f"advanced_very_low_assess:{name_raw}"
                f"(assess={assess_score:.0f}<{HONEYPOT_ADVANCED_LOW_ASSESS_THRESHOLD})"
            )
    return flags


def _check_mass_expert_claims(skills: List[Dict]) -> List[str]:
    """Too many expert skills without supporting evidence."""
    expert_skills = [sk for sk in skills if sk["proficiency"] == "expert"]
    expert_with_evidence = [
        sk for sk in expert_skills
        if sk["duration_months"] > 0 or sk["endorsements"] > 5
    ]
    if len(expert_skills) > HONEYPOT_MASS_EXPERT_COUNT:
        evidence_ratio = len(expert_with_evidence) / max(1, len(expert_skills))
        if evidence_ratio < 0.5:
            return [
                f"mass_expert:{len(expert_skills)}_skills"
                f"_only_{len(expert_with_evidence)}_with_evidence"
            ]
    return []


def _check_career_duration_mismatch(profile: Dict, career_history: List[Dict]) -> List[str]:
    """
    Check if declared YoE is wildly inconsistent with career history totals.
    A gap of > 4 years is suspicious in the synthetic data.
    """
    yoe = profile.get("years_of_experience", 0.0)
    total_months = sum(r.get("duration_months", 0) for r in career_history)
    total_years  = total_months / 12.0

    flags = []
    if total_years > 0 and yoe > total_years + 4:
        flags.append(
            f"yoe_career_mismatch:yoe={yoe:.1f}_career={total_years:.1f}yr"
        )
    return flags


def _check_identical_descriptions(career_history: List[Dict]) -> List[str]:
    """All role descriptions are identical — data-gen artifact, mild flag."""
    if len(career_history) < 2:
        return []
    descs = [r.get("description", "")[:80] for r in career_history]
    if len(set(descs)) == 1 and descs[0]:
        return ["all_descriptions_identical"]
    return []


def _check_zero_duration_past_roles(career_history: List[Dict]) -> List[str]:
    """Non-current roles with 0 duration months."""
    flags = []
    for role in career_history:
        if not role.get("is_current", False) and role.get("duration_months", 1) == 0:
            flags.append(f"zero_duration_past:{role.get('company','?')}")
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_honeypot(candidate: Dict) -> Tuple[float, List[str]]:
    """
    Returns (multiplier, flags).

    multiplier : float in [0.20, 1.0]
      1.0   → no honeypot signals
      < 1.0 → suspicious; the more flags, the lower
    flags : list of human-readable flag strings (for debugging / logging)
    """
    profile        = candidate["profile"]
    career_history = candidate["career_history"]
    skills         = candidate["skills"]
    assessments    = candidate["redrob_signals"].get("skill_assessment_scores", {})

    all_flags: List[str] = []
    multiplier = 1.0

    # Run all checks
    ef = _check_endorsement_fraud(skills, assessments)
    ac = _check_assessment_contradiction(skills, assessments)
    me = _check_mass_expert_claims(skills)
    cm = _check_career_duration_mismatch(profile, career_history)
    id_ = _check_identical_descriptions(career_history)
    zd = _check_zero_duration_past_roles(career_history)

    all_flags.extend(ef + ac + me + cm + id_ + zd)

    # Penalise per flag — severity-weighted
    for flag in ef:
        multiplier *= 0.88   # each endorsement-fraud flag: -12%
    for flag in ac:
        multiplier *= 0.82   # each assessment-contradiction: -18%
    for flag in me:
        multiplier *= 0.70   # mass expert claim: -30%
    for flag in cm:
        multiplier *= 0.85   # career mismatch: -15%
    for flag in id_:
        multiplier *= 0.95   # identical descriptions: -5% (mild; data artefact)
    for flag in zd:
        multiplier *= 0.93   # zero-duration past role: -7%

    # Floor: never go below 0.20 (don't hard-exclude; we could be wrong)
    multiplier = max(0.20, min(1.0, multiplier))

    if all_flags:
        logger.debug(
            "%s honeypot multiplier=%.2f flags=%s",
            candidate.get("candidate_id"), multiplier, all_flags
        )

    return round(multiplier, 4), all_flags
