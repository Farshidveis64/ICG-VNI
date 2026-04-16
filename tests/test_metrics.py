"""
Unit Tests — Metrics
====================
Sanity checks for all evaluation metrics.
Run with:  pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest
from utils.metrics import kendall_tau, jaccard_at_k, monotonicity, evaluate


class TestKendallTau:
    def test_perfect_agreement(self):
        scores = np.array([3.0, 1.0, 2.0, 4.0])
        assert abs(kendall_tau(scores, scores) - 1.0) < 1e-6

    def test_perfect_disagreement(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        true = np.array([4.0, 3.0, 2.0, 1.0])
        assert kendall_tau(pred, true) == pytest.approx(-1.0, abs=1e-6)

    def test_random_in_range(self):
        rng = np.random.default_rng(0)
        pred = rng.random(50)
        true = rng.random(50)
        tau = kendall_tau(pred, true)
        assert -1.0 <= tau <= 1.0


class TestJaccardAtK:
    def test_perfect_overlap(self):
        scores = np.array([4.0, 3.0, 2.0, 1.0])
        assert jaccard_at_k(scores, scores, k=2) == pytest.approx(1.0)

    def test_no_overlap(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        true = np.array([4.0, 3.0, 2.0, 1.0])
        # top-2 pred = {0,1}, top-2 true = {2,3} → JS = 0
        assert jaccard_at_k(pred, true, k=2) == pytest.approx(0.0)

    def test_k_larger_than_n(self):
        scores = np.array([1.0, 2.0])
        # Should not raise; caps k at N
        result = jaccard_at_k(scores, scores, k=100)
        assert result == pytest.approx(1.0)


class TestMonotonicity:
    def test_all_unique(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        assert monotonicity(scores) == pytest.approx(1.0)

    def test_all_tied(self):
        scores = np.array([1.0, 1.0, 1.0, 1.0])
        assert monotonicity(scores) == pytest.approx(0.0)

    def test_partial_ties(self):
        scores = np.array([1.0, 1.0, 2.0, 3.0])
        m = monotonicity(scores)
        assert 0.0 < m < 1.0

    def test_single_node(self):
        assert monotonicity(np.array([5.0])) == pytest.approx(1.0)


class TestEvaluate:
    def test_returns_all_keys(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        true = np.array([1.0, 2.0, 3.0, 4.0])
        result = evaluate(pred, true)
        assert "kendall_tau" in result
        assert "js_at_10" in result
        assert "js_at_50" in result
        assert "monotonicity" in result

    def test_perfect_scores(self):
        scores = np.arange(10, dtype=float)
        result = evaluate(scores, scores)
        assert result["kendall_tau"] == pytest.approx(1.0, abs=1e-6)
        assert result["monotonicity"] == pytest.approx(1.0, abs=1e-6)
