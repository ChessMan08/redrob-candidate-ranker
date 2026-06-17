import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

from src.data.loader import load_candidates, deduplicate
from src.data.preprocessor import clean_candidates
from src.scoring.composite import score_candidates, CandidateScore
from src.evaluation.evaluator import (
    build_evaluation_report,
    print_evaluation_report,
    load_manual_labels,
    load_honeypot_ids,
)
from src.config.settings import ARTIFACTS_DIR


def save_scores(scored: list, path: Path) -> None:
    """Save all scored candidates to JSON for analysis."""
    data = [cs.to_dict() for cs in scored]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %d scores to %s", len(data), path)


def save_top100_detail(scored: list, path: Path) -> None:
    """Save detailed breakdown for top-100."""
    from src.scoring.reasoning import generate_reasoning
    top100 = []
    for rank, cs in enumerate(scored[:100], 1):
        if cs.candidate is None:
            continue
        profile = cs.candidate["profile"]
        sig     = cs.candidate["redrob_signals"]
        top100.append({
            "rank":           rank,
            "candidate_id":   cs.candidate_id,
            "composite":      round(cs.composite, 4),
            "career_score":   cs.career_score,
            "skills_score":   cs.skills_score,
            "experience_score": cs.experience_score,
            "behavioral_score": cs.behavioral_score,
            "location_score": cs.location_score,
            "education_score": cs.education_score,
            "honeypot_mult":  cs.honeypot_multiplier,
            "behavioral_gate":cs.behavioral_gate,
            "honeypot_flags": cs.honeypot_flags,
            "tier1_skills":   cs.tier1_skills,
            "tier2_skills":   cs.tier2_skills,
            "title":          profile.get("current_title"),
            "company":        profile.get("current_company"),
            "industry":       profile.get("current_industry"),
            "yoe":            profile.get("years_of_experience"),
            "location":       profile.get("location"),
            "country":        profile.get("country"),
            "open_to_work":   sig.get("open_to_work_flag"),
            "rrr":            sig.get("recruiter_response_rate"),
            "notice_days":    sig.get("notice_period_days"),
            "github":         sig.get("github_activity_score"),
            "reasoning":      generate_reasoning(cs, rank),
            "career_breakdown":   cs.career_breakdown,
            "behavioral_breakdown": cs.behavioral_breakdown,
        })
    with open(path, "w") as f:
        json.dump(top100, f, indent=2, default=str)
    logger.info("Saved top-100 detail to %s", path)


def sensitivity_analysis(scored: list) -> None:
    from src.config import settings
    import copy

    original_top10 = [cs.candidate_id for cs in scored[:10]]

    perturbations = [
        {"career": 0.40, "skills": 0.25, "experience": 0.13, "behavioral": 0.12,
         "location": 0.06, "education": 0.04},
        {"career": 0.30, "skills": 0.35, "experience": 0.13, "behavioral": 0.12,
         "location": 0.06, "education": 0.04},
        {"career": 0.35, "skills": 0.30, "experience": 0.13, "behavioral": 0.17,
         "location": 0.01, "education": 0.04},
    ]

    print("\n── Sensitivity Analysis (top-10 stability) ──────────────────────")
    for p_idx, weights in enumerate(perturbations):
        # Temporarily patch settings.WEIGHTS
        original_weights = copy.copy(settings.WEIGHTS)
        settings.WEIGHTS.update(weights)

        # Re-score all
        perturbed = []
        for cs in scored:
            composite = (
                weights["career"]     * cs.career_score
                + weights["skills"]   * cs.skills_score
                + weights["experience"]* cs.experience_score
                + weights["behavioral"]* cs.behavioral_score
                + weights["location"]  * cs.location_score
                + weights["education"] * cs.education_score
            )
            composite *= cs.honeypot_multiplier * cs.behavioral_gate * cs.salary_multiplier
            composite = min(100.0, max(0.0, composite))
            import dataclasses
            perturbed.append(dataclasses.replace(cs, composite=composite))

        perturbed.sort(key=lambda s: (-s.composite, s.candidate_id))
        perturbed_top10 = [cs.candidate_id for cs in perturbed[:10]]
        overlap = len(set(original_top10) & set(perturbed_top10))

        settings.WEIGHTS.update(original_weights)

        weight_str = " ".join(f"{k[0]}={v:.2f}" for k, v in weights.items())
        print(f"  Perturbation {p_idx+1} [{weight_str}]: "
              f"top-10 overlap={overlap}/10")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the candidate ranker")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--tfidf", action="store_true",
                        help="Enable TF-IDF re-ranking")
    parser.add_argument("--save-scores", action="store_true",
                        help="Save all scores to artifacts/scores.json")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run sensitivity analysis on weights")
    args = parser.parse_args()

    t0 = time.time()

    logger.info("Loading candidates...")
    raw     = load_candidates(args.candidates, max_records=args.max)
    raw     = deduplicate(raw)
    cleaned = clean_candidates(raw)
    logger.info("Loaded %d candidates", len(cleaned))

    logger.info("Scoring...")
    scored = score_candidates(cleaned, show_progress=True)

    if args.tfidf:
        from src.retrieval.tfidf_reranker import tfidf_rerank
        logger.info("TF-IDF re-ranking...")
        scored = tfidf_rerank(scored, top_n=500)

    manual_labels = load_manual_labels()
    honeypot_ids  = load_honeypot_ids()
    report = build_evaluation_report(scored, manual_labels, honeypot_ids)
    print_evaluation_report(report)

    if args.sensitivity:
        sensitivity_analysis(scored)

    if args.save_scores:
        save_scores(scored, ARTIFACTS_DIR / "scores.json")

    save_top100_detail(scored, ARTIFACTS_DIR / "top100_detail.json")

    logger.info("Total: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
