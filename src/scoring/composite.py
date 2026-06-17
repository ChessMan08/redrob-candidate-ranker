from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config.settings import WEIGHTS
from src.features.career_features import score_career
from src.features.skill_features import score_skills, top_tier1_skills, top_tier2_skills
from src.features.behavioral_features import score_behavioral, behavioral_multiplier
from src.features.profile_features import (
    score_experience,
    score_location,
    score_education,
    salary_fit_multiplier,
)
from src.features.honeypot_detector import detect_honeypot

# Output dataclass
@dataclass
class CandidateScore:
    candidate_id: str
    composite: float                        # final score [0, 100]
    # Component scores [0, 100]
    career_score: float = 0.0
    skills_score: float = 0.0
    experience_score: float = 0.0
    behavioral_score: float = 0.0
    location_score: float = 0.0
    education_score: float = 0.0
    # Multipliers
    honeypot_multiplier: float = 1.0
    behavioral_gate: float = 1.0
    salary_multiplier: float = 1.0
    # Diagnostics
    honeypot_flags: List[str] = field(default_factory=list)
    # For reasoning generation
    tier1_skills: List[str] = field(default_factory=list)
    tier2_skills: List[str] = field(default_factory=list)
    career_breakdown: Dict = field(default_factory=dict)
    behavioral_breakdown: Dict = field(default_factory=dict)
    location_breakdown: Dict = field(default_factory=dict)
    # Convenience: the full cleaned candidate dict (not stored in output CSV)
    candidate: Optional[Dict] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id":       self.candidate_id,
            "composite":          round(self.composite, 6),
            "career_score":       self.career_score,
            "skills_score":       self.skills_score,
            "experience_score":   self.experience_score,
            "behavioral_score":   self.behavioral_score,
            "location_score":     self.location_score,
            "education_score":    self.education_score,
            "honeypot_multiplier":self.honeypot_multiplier,
            "behavioral_gate":    self.behavioral_gate,
            "salary_multiplier":  self.salary_multiplier,
            "honeypot_flags":     "|".join(self.honeypot_flags),
            "tier1_skills":       ",".join(self.tier1_skills),
        }

# Per-candidate scorer
def score_candidate(candidate: Dict) -> CandidateScore:
    """Full scoring pipeline for a single cleaned candidate."""
    cid    = candidate["candidate_id"]
    profile= candidate["profile"]
    sig    = candidate["redrob_signals"]
    skills = candidate["skills"]
    edu    = candidate["education"]
    career = candidate["career_history"]

    # ── Feature scores ─────────────────────────────────────────────
    career_s,  career_bd  = score_career(career)
    skills_s,  _          = score_skills(skills, sig.get("skill_assessment_scores") or {})
    exp_s,     _          = score_experience(profile.get("years_of_experience", 0.0))
    behav_s,   behav_bd   = score_behavioral(sig)
    loc_s,     loc_bd     = score_location(profile, sig)
    edu_s,     _          = score_education(edu)

    # ── Weighted composite ─────────────────────────────────────────
    composite = (
        WEIGHTS["career"]     * career_s
        + WEIGHTS["skills"]   * skills_s
        + WEIGHTS["experience"]* exp_s
        + WEIGHTS["behavioral"]* behav_s
        + WEIGHTS["location"]  * loc_s
        + WEIGHTS["education"] * edu_s
    )

    # ── Multipliers ────────────────────────────────────────────────
    hp_mult, hp_flags = detect_honeypot(candidate)
    bh_gate           = behavioral_multiplier(sig)
    sal_mult          = salary_fit_multiplier(sig)

    composite *= hp_mult * bh_gate * sal_mult

    # ── Clip to [0, 100] ──────────────────────────────────────────
    composite = min(100.0, max(0.0, composite))
  
    #    Cap composite to 45 if skills_score < 5 (no credentialed ML skills).
    if skills_s < 5.0:
        composite = min(composite, 45.0)

    #    Add a small bonus (max 2.0 pts) to separate closely-ranked candidates.
    github = sig.get("github_activity_score", -1.0)
    if github >= 0:
        composite += (github / 100.0) * 2.0   # max +2.0 pts at github=100

    # ── Collect top skills for reasoning ──────────────────────────
    t1_skills = top_tier1_skills(skills, sig.get("skill_assessment_scores") or {})
    t2_skills = top_tier2_skills(skills, sig.get("skill_assessment_scores") or {})

    return CandidateScore(
        candidate_id        = cid,
        composite           = round(composite, 6),
        career_score        = career_s,
        skills_score        = skills_s,
        experience_score    = exp_s,
        behavioral_score    = behav_s,
        location_score      = loc_s,
        education_score     = edu_s,
        honeypot_multiplier = hp_mult,
        behavioral_gate     = bh_gate,
        salary_multiplier   = sal_mult,
        honeypot_flags      = hp_flags,
        tier1_skills        = t1_skills,
        tier2_skills        = t2_skills,
        career_breakdown    = career_bd,
        behavioral_breakdown= behav_bd,
        location_breakdown  = loc_bd,
        candidate           = candidate,
    )

# Batch scorer (with optional progress)
def score_candidates(
    candidates: List[Dict],
    show_progress: bool = False,
) -> List[CandidateScore]:
    scores: List[CandidateScore] = []

    iterator = candidates
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(candidates, desc="Scoring", unit="cand")
        except ImportError:
            pass

    for c in iterator:
        try:
            scores.append(score_candidate(c))
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Error scoring %s: %s", c.get("candidate_id", "?"), exc
            )

    scores.sort(key=lambda s: (-s.composite, s.candidate_id))
    return scores
