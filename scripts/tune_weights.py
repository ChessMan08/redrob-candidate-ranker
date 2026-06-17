import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("tune_weights")

from src.data.loader import load_candidates, deduplicate
from src.data.preprocessor import clean_candidates
from src.scoring.composite import score_candidates, CandidateScore
from src.evaluation.metrics import ndcg_at_k, make_proxy_relevance_map
from src.config.settings import ARTIFACTS_DIR


def recompute_composite(scored: list[CandidateScore], weights: dict) -> list[CandidateScore]:
    """Re-compute composite scores from stored components with new weights."""
    import dataclasses
    recomputed = []
    for cs in scored:
        composite = (
            weights["career"]      * cs.career_score
            + weights["skills"]    * cs.skills_score
            + weights["experience"]* cs.experience_score
            + weights["behavioral"]* cs.behavioral_score
            + weights["location"]  * cs.location_score
            + weights["education"] * cs.education_score
        )
        composite *= cs.honeypot_multiplier * cs.behavioral_gate * cs.salary_multiplier
        composite = min(100.0, max(0.0, composite))
        recomputed.append(dataclasses.replace(cs, composite=composite))
    recomputed.sort(key=lambda s: (-s.composite, s.candidate_id))
    return recomputed


def evaluate_weights(
    baseline_scored: list[CandidateScore],
    weights: dict,
    proxy_map: dict,
) -> dict:
    """Return metric dict for a given weight configuration."""
    recomputed = recompute_composite(baseline_scored, weights)
    ranked_ids = [cs.candidate_id for cs in recomputed]
    ndcg10 = ndcg_at_k(ranked_ids, proxy_map, 10)
    ndcg50 = ndcg_at_k(ranked_ids, proxy_map, 50)
    composite_metric = 0.50 * ndcg10 + 0.30 * ndcg50
    return {
        "ndcg@10": round(ndcg10, 4),
        "ndcg@50": round(ndcg50, 4),
        "composite": round(composite_metric, 4),
        "weights": weights,
    }


def grid_search(
    scored: list[CandidateScore],
    proxy_map: dict,
) -> list[dict]:
    # Career and skills are the dominant signals.
    # Education and location are fixed at their small values.
    career_vals   = [0.28, 0.32, 0.35, 0.38, 0.42]
    skills_vals   = [0.22, 0.26, 0.30, 0.34]
    exp_vals      = [0.10, 0.13, 0.16]
    behavioral_vals = [0.10, 0.12, 0.15]

    results = []
    total_combos = len(career_vals) * len(skills_vals) * len(exp_vals) * len(behavioral_vals)
    logger.info("Grid search: %d combinations...", total_combos)

    for car, sk, exp, beh in itertools.product(
        career_vals, skills_vals, exp_vals, behavioral_vals
    ):
        remainder = 1.0 - car - sk - exp - beh
        if remainder < 0 or remainder > 0.20:
            continue   # outside reasonable range for loc+edu

        loc = round(remainder * 0.55, 3)
        edu = round(remainder * 0.45, 3)
        # Fix rounding
        edu = round(1.0 - car - sk - exp - beh - loc, 4)
        if edu < 0:
            continue

        weights = {
            "career":     car,
            "skills":     sk,
            "experience": exp,
            "behavioral": beh,
            "location":   loc,
            "education":  edu,
        }
        result = evaluate_weights(scored, weights, proxy_map)
        results.append(result)

    results.sort(key=lambda r: -r["composite"])
    return results


def main():
    parser = argparse.ArgumentParser(description="Grid-search weight tuning")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=10,
                        help="Show top-N weight configs")
    args = parser.parse_args()

    t0 = time.time()

    logger.info("Loading candidates...")
    raw     = load_candidates(args.candidates, max_records=args.max)
    raw     = deduplicate(raw)
    cleaned = clean_candidates(raw)

    logger.info("Scoring with baseline weights...")
    scored = score_candidates(cleaned, show_progress=True)

    # Build proxy labels from baseline scores
    proxy_map = make_proxy_relevance_map(
        [{"candidate_id": cs.candidate_id, "composite": cs.composite}
         for cs in scored]
    )
    logger.info(
        "Proxy labels: grade-3=%d  grade-2=%d  grade-1=%d  grade-0=%d",
        sum(1 for v in proxy_map.values() if v == 3),
        sum(1 for v in proxy_map.values() if v == 2),
        sum(1 for v in proxy_map.values() if v == 1),
        sum(1 for v in proxy_map.values() if v == 0),
    )

    # Grid search
    results = grid_search(scored, proxy_map)

    # Current baseline for comparison
    from src.config.settings import WEIGHTS as CURRENT_WEIGHTS
    baseline = evaluate_weights(scored, CURRENT_WEIGHTS, proxy_map)

    print(f"\n{'='*70}")
    print("WEIGHT TUNING RESULTS")
    print(f"{'='*70}")
    print(f"\nBaseline (current settings.py):")
    print(f"  NDCG@10={baseline['ndcg@10']:.4f}  NDCG@50={baseline['ndcg@50']:.4f}  "
          f"Composite={baseline['composite']:.4f}")
    print(f"  Weights: {CURRENT_WEIGHTS}")

    print(f"\nTop {args.top_n} weight configurations found:")
    for i, r in enumerate(results[:args.top_n], 1):
        print(f"\n  #{i}  NDCG@10={r['ndcg@10']:.4f}  NDCG@50={r['ndcg@50']:.4f}  "
              f"Composite={r['composite']:.4f}")
        w = r["weights"]
        print(f"       car={w['career']:.2f} sk={w['skills']:.2f} "
              f"exp={w['experience']:.2f} beh={w['behavioral']:.2f} "
              f"loc={w['location']:.2f} edu={w['education']:.2f}")

    # Save results
    out = ARTIFACTS_DIR / "weight_search_results.json"
    with open(out, "w") as f:
        json.dump(results[:50], f, indent=2)
    logger.info("Saved top-50 results to %s", out)
    logger.info("Total: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
