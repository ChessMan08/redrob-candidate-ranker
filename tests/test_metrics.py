"""
tests/test_metrics.py — Unit tests for evaluation metrics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.evaluation.metrics import (
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    average_precision,
    reciprocal_rank,
    composite_score,
    honeypot_rate_in_top_k,
    score_diagnostics,
)


class TestDCG:
    def test_perfect_ranking(self):
        rels = [3, 2, 1, 0]
        dcg = dcg_at_k(rels, 4)
        assert dcg > 0

    def test_zero_relevance(self):
        assert dcg_at_k([0, 0, 0], 3) == 0.0

    def test_k_cutoff_respected(self):
        d4 = dcg_at_k([3, 2, 1, 3], 4)
        d2 = dcg_at_k([3, 2, 1, 3], 2)
        assert d4 > d2


class TestNDCG:
    def test_perfect_ranking_is_1(self):
        ids    = ["a", "b", "c", "d"]
        rel    = {"a": 3, "b": 2, "c": 1, "d": 0}
        score  = ndcg_at_k(ids, rel, 4)
        assert abs(score - 1.0) < 1e-9

    def test_reversed_ranking_is_low(self):
        # With k=2, placing the worst item at rank-1 gives near-0 DCG
        ids    = ["d", "c", "b", "a"]
        rel    = {"a": 3, "b": 2, "c": 1, "d": 0}
        score  = ndcg_at_k(ids, rel, 2)
        assert score < 0.20

    def test_missing_ids_get_zero_relevance(self):
        ids   = ["a", "z", "b"]  # z is not in rel_map
        rel   = {"a": 3, "b": 1}
        score = ndcg_at_k(ids, rel, 3)
        assert 0 < score < 1.0

    def test_empty_inputs(self):
        assert ndcg_at_k([], {}, 10) == 0.0
        assert ndcg_at_k(["a"], {}, 10) == 0.0


class TestPrecisionAtK:
    def test_all_relevant(self):
        ids = ["a", "b", "c"]
        rel = {"a", "b", "c"}
        assert precision_at_k(ids, rel, 3) == 1.0

    def test_none_relevant(self):
        ids = ["a", "b", "c"]
        rel = {"x", "y"}
        assert precision_at_k(ids, rel, 3) == 0.0

    def test_k_cutoff(self):
        ids = ["a", "b", "c", "d", "e"]
        rel = {"a", "b"}
        assert precision_at_k(ids, rel, 2) == 1.0
        assert precision_at_k(ids, rel, 5) == 0.4


class TestAveragePrecision:
    def test_perfect_ap(self):
        ids = ["a", "b", "c"]
        rel = {"a", "b", "c"}
        assert average_precision(ids, rel) == 1.0

    def test_no_relevant(self):
        ids = ["a", "b", "c"]
        rel = set()
        assert average_precision(ids, rel) == 0.0

    def test_relevant_at_end_lower(self):
        ids_good = ["a", "b", "c"]  # relevant items at top
        ids_bad  = ["x", "y", "a"]  # relevant item at bottom
        rel      = {"a"}
        ap_good  = average_precision(ids_good, rel)
        ap_bad   = average_precision(ids_bad, rel)
        assert ap_good > ap_bad


class TestReciprocalRank:
    def test_first_relevant_at_rank1(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_first_relevant_at_rank5(self):
        ids = ["x", "x", "x", "x", "a"]
        assert abs(reciprocal_rank(ids, {"a"}) - 0.2) < 1e-9

    def test_no_relevant(self):
        assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


class TestCompositeScore:
    def test_perfect_submission(self):
        rel    = {"a": 3, "b": 2, "c": 1}
        ranked = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
        result = composite_score(ranked, rel)
        assert result["ndcg@10"] > 0.5
        assert result["composite"] > 0.0

    def test_all_metrics_present(self):
        result = composite_score(["a"], {"a": 3})
        assert all(k in result for k in ("ndcg@10", "ndcg@50", "map", "p@10", "composite"))


class TestHoneypotRate:
    def test_no_honeypots(self):
        ranked   = ["a", "b", "c"]
        honeypot = {"x", "y"}
        rate     = honeypot_rate_in_top_k(ranked, honeypot, 3)
        assert rate == 0.0

    def test_all_honeypots(self):
        ranked   = ["a", "b", "c"]
        honeypot = {"a", "b", "c"}
        rate     = honeypot_rate_in_top_k(ranked, honeypot, 3)
        assert rate == 1.0

    def test_10pct_threshold(self):
        ranked   = ["a"] + [f"x{i}" for i in range(9)]
        honeypot = {"a"}
        rate     = honeypot_rate_in_top_k(ranked, honeypot, 10)
        assert rate == 0.10


class TestScoreDiagnostics:
    def test_basic_stats(self):
        scores = [10.0, 20.0, 30.0, 40.0, 50.0]
        diag   = score_diagnostics(scores)
        assert diag["min"] == 10.0
        assert diag["max"] == 50.0
        assert diag["mean"] == 30.0

    def test_empty_list(self):
        assert score_diagnostics([]) == {}
