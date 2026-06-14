"""
evaluator.py — Offline evaluation harness.

Without ground-truth labels we do three things:
  1. Proxy evaluation  — use our own composite scores as pseudo-labels
     (lower-bound signal; useful for detecting regressions between runs).
  2. Manual-annotation evaluation — if manual_labels.json exists in
     artifacts/, use those grades as ground truth.
  3. Cross-validation over sub-sampled JD variants — score sensitivity analysis.

Run via:  python scripts/evaluate.py --candidates sample_candidates.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.evaluation.metrics import (
    composite_score,
    score_diagnostics,
    honeypot_rate_in_top_k,
    make_proxy_relevance_map,
    ndcg_at_k,
    precision_at_k,
)
from src.scoring.composite import CandidateScore
from src.config.settings import ARTIFACTS_DIR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Label loading
# ─────────────────────────────────────────────────────────────────────────────

def load_manual_labels(path: Optional[Path] = None) -> Optional[Dict[str, float]]:
    """
    Load manually annotated relevance grades from a JSON file.

    Expected format:
      {
        "CAND_0000031": 3,
        "CAND_0000015": 2,
        "CAND_0000002": 0,
        ...
      }

    Returns None if the file doesn't exist.
    """
    path = path or (ARTIFACTS_DIR / "manual_labels.json")
    if not path.exists():
        logger.info("No manual labels found at %s", path)
        return None
    with open(path) as f:
        raw = json.load(f)
    labels = {k: float(v) for k, v in raw.items()
              if not k.startswith("_")}
    logger.info("Loaded %d manual labels from %s", len(labels), path)
    return labels


def load_honeypot_ids(path: Optional[Path] = None) -> set:
    """
    Load known honeypot candidate IDs from a JSON list file.
    File format: ["CAND_0000042", "CAND_0000099", ...]
    """
    path = path or (ARTIFACTS_DIR / "honeypot_ids.json")
    if not path.exists():
        return set()
    with open(path) as f:
        return set(json.load(f))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation report builder
# ─────────────────────────────────────────────────────────────────────────────

def build_evaluation_report(
    scored: List[CandidateScore],
    manual_labels: Optional[Dict[str, float]] = None,
    honeypot_ids: Optional[set] = None,
    top_n: int = 100,
) -> Dict:
    """
    Build a full evaluation report.

    Returns a dict with all metric values and diagnostic info.
    """
    ranked_ids    = [cs.candidate_id for cs in scored]
    top_n_ids     = ranked_ids[:top_n]
    all_composites= [cs.composite for cs in scored]
    top_composites= [cs.composite for cs in scored[:top_n]]

    report: Dict = {}

    # ── Score diagnostics ────────────────────────────────────────
    report["score_distribution_all"]   = score_diagnostics(all_composites)
    report["score_distribution_top100"]= score_diagnostics(top_composites)

    # ── Gap between rank-1 and rank-10 (good models have large gaps) ──
    if len(scored) >= 10:
        report["rank1_score"]  = round(scored[0].composite, 4)
        report["rank10_score"] = round(scored[9].composite, 4)
        report["rank1_to_rank10_gap"] = round(scored[0].composite - scored[9].composite, 4)

    # ── Component score averages in top-10 vs top-100 ────────────
    def avg_component(sl, attr):
        vals = [getattr(s, attr) for s in sl]
        return round(sum(vals) / max(1, len(vals)), 2)

    for attr in ("career_score", "skills_score", "experience_score",
                 "behavioral_score", "location_score"):
        report[f"avg_{attr}_top10"]  = avg_component(scored[:10],  attr)
        report[f"avg_{attr}_top100"] = avg_component(scored[:100], attr)

    # ── Honeypot analysis ────────────────────────────────────────
    flagged_in_top100 = [cs for cs in scored[:100] if cs.honeypot_flags]
    report["honeypot_flagged_in_top100"] = len(flagged_in_top100)
    if flagged_in_top100:
        report["honeypot_flagged_ids"] = [cs.candidate_id for cs in flagged_in_top100]

    if honeypot_ids:
        report["known_honeypot_rate_top100"] = round(
            honeypot_rate_in_top_k(ranked_ids, honeypot_ids, 100), 4
        )

    # ── Proxy-label metrics (always computable) ──────────────────
    proxy_map = make_proxy_relevance_map(
        [{"candidate_id": cs.candidate_id, "composite": cs.composite} for cs in scored]
    )
    proxy_rel_ids = {cid for cid, g in proxy_map.items() if g >= 3}

    proxy_metrics = composite_score(ranked_ids, proxy_map, proxy_rel_ids)
    report["proxy_metrics"] = proxy_metrics

    # ── Manual-label metrics (if available) ─────────────────────
    if manual_labels:
        manual_rel_ids = {cid for cid, g in manual_labels.items() if g >= 3}
        manual_metrics = composite_score(ranked_ids, manual_labels, manual_rel_ids)
        report["manual_metrics"] = manual_metrics
        report["manual_label_coverage"] = len(
            [cid for cid in top_n_ids if cid in manual_labels]
        )

    # ── Top-10 candidate summary ─────────────────────────────────
    top10_summary = []
    for rank, cs in enumerate(scored[:10], 1):
        c = cs.candidate
        p = c["profile"]
        top10_summary.append({
            "rank":       rank,
            "id":         cs.candidate_id,
            "title":      p.get("current_title", ""),
            "company":    p.get("current_company", ""),
            "yoe":        p.get("years_of_experience", 0),
            "composite":  round(cs.composite, 2),
            "career":     cs.career_score,
            "skills":     cs.skills_score,
            "behavioral": cs.behavioral_score,
            "tier1_skills": cs.tier1_skills,
            "honeypot_flags": cs.honeypot_flags,
        })
    report["top10"] = top10_summary

    return report


def print_evaluation_report(report: Dict) -> None:
    """Pretty-print the evaluation report to stdout."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    print("\n── Score Distribution (all candidates) ─────────────────────────")
    diag = report.get("score_distribution_all", {})
    print(f"  Min={diag.get('min',0):.2f}  Max={diag.get('max',0):.2f}  "
          f"Mean={diag.get('mean',0):.2f}  Median={diag.get('median',0):.2f}  "
          f"Stdev={diag.get('stdev',0):.2f}")

    print(f"\n── Top-10 gap: rank-1={report.get('rank1_score',0):.2f}  "
          f"rank-10={report.get('rank10_score',0):.2f}  "
          f"gap={report.get('rank1_to_rank10_gap',0):.2f}")

    print("\n── Component Averages (top-10 vs top-100) ──────────────────────")
    for attr in ("career_score", "skills_score", "behavioral_score"):
        t10  = report.get(f"avg_{attr}_top10", 0)
        t100 = report.get(f"avg_{attr}_top100", 0)
        print(f"  {attr:22s}  top-10={t10:.1f}  top-100={t100:.1f}")

    print("\n── Proxy Metrics ───────────────────────────────────────────────")
    pm = report.get("proxy_metrics", {})
    print(f"  NDCG@10={pm.get('ndcg@10',0):.4f}  NDCG@50={pm.get('ndcg@50',0):.4f}  "
          f"MAP={pm.get('map',0):.4f}  P@10={pm.get('p@10',0):.4f}  "
          f"Composite={pm.get('composite',0):.4f}")

    if "manual_metrics" in report:
        print("\n── Manual-Label Metrics ────────────────────────────────────────")
        mm = report["manual_metrics"]
        print(f"  NDCG@10={mm.get('ndcg@10',0):.4f}  NDCG@50={mm.get('ndcg@50',0):.4f}  "
              f"MAP={mm.get('map',0):.4f}  P@10={mm.get('p@10',0):.4f}  "
              f"Composite={mm.get('composite',0):.4f}")
        print(f"  Label coverage in top-100: {report.get('manual_label_coverage',0)}")

    print(f"\n── Honeypot ──────────────────────────────────────────────────")
    print(f"  Flagged in top-100: {report.get('honeypot_flagged_in_top100',0)}")
    if report.get("known_honeypot_rate_top100") is not None:
        rate = report["known_honeypot_rate_top100"]
        status = "✓ OK" if rate <= 0.10 else "✗ DISQUALIFIED (>10%)"
        print(f"  Known honeypot rate in top-100: {rate:.1%}  {status}")

    print("\n── Top-10 Candidates ───────────────────────────────────────────")
    for row in report.get("top10", []):
        flags = f" ⚠ {row['honeypot_flags']}" if row["honeypot_flags"] else ""
        print(
            f"  #{row['rank']:2d} {row['id']}  {row['title'][:30]:30s}  "
            f"YoE={row['yoe']:.1f}  score={row['composite']:.1f}  "
            f"car={row['career']:.0f} sk={row['skills']:.0f} be={row['behavioral']:.0f}"
            f"{flags}"
        )
        if row["tier1_skills"]:
            print(f"      Skills: {', '.join(row['tier1_skills'][:3])}")

    print("=" * 70 + "\n")
