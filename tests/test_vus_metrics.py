"""
test_vus_metrics.py — Property-based and reference-case tests for the
range-aware AUC and VUS implementations.

References echoed by the assertions below:
    Paparrizos et al. (2022) — VUS, PVLDB 15(11).
    Tatbul et al. (2018)    — Range-based recall, NeurIPS.
    Davis & Goadrich (2006) — AUC-PR baseline = base rate.

Tracks TASK-54 in docs/audit/audit_verification_tracker.md.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sl_ads.evaluate.vus_metrics import (  # noqa: E402
    existence_recall,
    extend_anomaly_ranges,
    find_anomaly_ranges,
    range_auc_pr,
    range_auc_roc,
    vus_pr,
    vus_roc,
    vus_summary,
)


# ════════════════════════════════════════════════════════════════════════
# find_anomaly_ranges
# ════════════════════════════════════════════════════════════════════════
class TestFindAnomalyRanges:
    def test_empty_label_vector_returns_empty_list(self):
        assert find_anomaly_ranges([]) == []

    def test_no_anomaly_returns_empty_list(self):
        assert find_anomaly_ranges([0, 0, 0, 0]) == []

    def test_all_anomaly_returns_single_full_range(self):
        assert find_anomaly_ranges([1, 1, 1]) == [(0, 2)]

    def test_two_disjoint_runs(self):
        assert find_anomaly_ranges([0, 1, 1, 0, 1]) == [(1, 2), (4, 4)]

    def test_run_at_boundaries(self):
        # Runs at index 0 and at the last index — boundary handling.
        assert find_anomaly_ranges([1, 1, 0, 1]) == [(0, 1), (3, 3)]


# ════════════════════════════════════════════════════════════════════════
# extend_anomaly_ranges
# ════════════════════════════════════════════════════════════════════════
class TestExtendAnomalyRanges:
    def test_buffer_zero_is_identity(self):
        y = np.array([0, 1, 1, 0, 0, 1, 0])
        out = extend_anomaly_ranges(y, 0)
        assert (out == y).all()

    def test_buffer_widens_each_range(self):
        y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
        out = extend_anomaly_ranges(y, 1).tolist()
        assert out == [0, 1, 1, 1, 1, 0, 1, 1, 1, 0]

    def test_buffer_clipped_at_boundaries(self):
        y = np.array([1, 0, 0, 0, 1])
        out = extend_anomaly_ranges(y, 100)
        assert (out == 1).all()

    def test_negative_buffer_raises(self):
        with pytest.raises(ValueError):
            extend_anomaly_ranges([0, 1, 0], -1)

    def test_dtype_is_uint8(self):
        out = extend_anomaly_ranges([0, 1, 0], 1)
        assert out.dtype == np.uint8

    def test_input_not_mutated(self):
        y = np.array([0, 1, 0, 1, 0])
        snapshot = y.copy()
        _ = extend_anomaly_ranges(y, 1)
        assert (y == snapshot).all()


# ════════════════════════════════════════════════════════════════════════
# Range-AUC reference cases
# ════════════════════════════════════════════════════════════════════════
class TestRangeAUCReferenceCases:
    def test_perfect_detector_l0_returns_one(self):
        y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
        s = y.astype(float)
        assert math.isclose(range_auc_roc(y, s, 0), 1.0, abs_tol=1e-9)
        assert math.isclose(range_auc_pr(y, s, 0), 1.0, abs_tol=1e-9)

    def test_inverted_detector_l0_returns_zero(self):
        y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
        s = 1.0 - y.astype(float)
        assert range_auc_roc(y, s, 0) == 0.0

    def test_all_zero_labels_returns_nan(self):
        y = np.zeros(20, dtype=int)
        s = np.random.default_rng(0).random(20)
        assert math.isnan(range_auc_roc(y, s, 0))
        assert math.isnan(range_auc_pr(y, s, 0))

    def test_buffer_saturating_labels_returns_nan(self):
        # A short sequence with a buffer big enough to make every label
        # positive must return NaN (degenerate ROC).
        y = np.array([0, 0, 1, 0, 0])
        s = np.array([0.1, 0.2, 0.9, 0.2, 0.1])
        assert math.isnan(range_auc_roc(y, s, 5))

    def test_random_detector_auc_near_chance(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 5000)
        s = rng.random(5000)
        auc = range_auc_roc(y, s, 0)
        assert 0.45 < auc < 0.55, auc


# ════════════════════════════════════════════════════════════════════════
# VUS — definitional properties
# ════════════════════════════════════════════════════════════════════════
class TestVUSProperties:
    @pytest.fixture
    def sparse_anomalies(self):
        rng = np.random.default_rng(2)
        y = np.zeros(2000, dtype=np.int8)
        starts = rng.integers(50, 1950, size=20)
        for s in starts:
            y[s: s + 5] = 1
        return y

    def test_vus_in_unit_interval(self, sparse_anomalies):
        rng = np.random.default_rng(3)
        s = rng.random(sparse_anomalies.size)
        for fn in (vus_roc, vus_pr):
            v = fn(sparse_anomalies, s, max_buffer=5, n_steps=6)
            assert 0.0 <= v <= 1.0, (fn.__name__, v)

    def test_vus_roc_random_near_half(self, sparse_anomalies):
        rng = np.random.default_rng(4)
        s = rng.random(sparse_anomalies.size)
        v = vus_roc(sparse_anomalies, s, max_buffer=5, n_steps=6)
        assert 0.40 < v < 0.60, v

    def test_vus_perfect_detector_high(self, sparse_anomalies):
        s = sparse_anomalies.astype(float)
        v_roc = vus_roc(sparse_anomalies, s, max_buffer=3, n_steps=4)
        v_pr = vus_pr(sparse_anomalies, s, max_buffer=3, n_steps=4)
        # The perfect detector reaches AUC=1.0 at L=0 but degrades as the
        # buffer adds buffered-positives that the detector did not flag.
        # An average above 0.75 (well above 0.5 chance) is the property
        # we are after; tighter bounds depend on run-length distribution.
        assert v_roc > 0.75, v_roc
        # AP for the perfect detector at L=0 is 1.0; buffering reduces
        # it slightly but it must remain well above the base rate.
        assert v_pr > 0.5, v_pr

    def test_vus_invalid_n_steps_raises(self, sparse_anomalies):
        s = np.zeros(sparse_anomalies.size)
        with pytest.raises(ValueError):
            vus_roc(sparse_anomalies, s, max_buffer=3, n_steps=1)

    def test_vus_negative_max_buffer_raises(self, sparse_anomalies):
        s = np.zeros(sparse_anomalies.size)
        with pytest.raises(ValueError):
            vus_roc(sparse_anomalies, s, max_buffer=-1, n_steps=3)


# ════════════════════════════════════════════════════════════════════════
# Existence-based recall (Tatbul 2018)
# ════════════════════════════════════════════════════════════════════════
class TestExistenceRecall:
    def test_zero_when_no_overlap(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 1])
        assert existence_recall(y_true, y_pred) == 0.0

    def test_one_when_each_range_hit(self):
        y_true = np.array([0, 1, 1, 0, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 0])
        assert existence_recall(y_true, y_pred) == 1.0

    def test_half_when_one_of_two_ranges_hit(self):
        y_true = np.array([0, 1, 1, 0, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 0, 0])
        assert existence_recall(y_true, y_pred) == 0.5

    def test_no_truth_returns_nan(self):
        assert math.isnan(existence_recall([0, 0, 0], [1, 1, 1]))


# ════════════════════════════════════════════════════════════════════════
# vus_summary — convenience wrapper
# ════════════════════════════════════════════════════════════════════════
class TestVUSSummary:
    def test_summary_returns_expected_keys(self):
        y = np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
        s = np.linspace(0, 1, len(y))
        out = vus_summary(y, s, y_pred=None, max_buffer=2, n_steps=3)
        for k in ("n", "n_anomalies", "n_ranges", "max_buffer",
                  "vus_roc", "vus_pr",
                  "range_auc_roc_at_max", "range_auc_pr_at_max"):
            assert k in out, k
        assert "existence_recall" not in out  # only when y_pred passed

    def test_summary_with_pred_includes_existence_recall(self):
        y = np.array([0, 1, 1, 0, 0, 1, 0])
        s = y.astype(float)
        p = np.array([0, 0, 1, 0, 0, 1, 0])
        out = vus_summary(y, s, y_pred=p, max_buffer=1, n_steps=2)
        assert math.isclose(out["existence_recall"], 1.0, abs_tol=1e-9)

    def test_summary_default_max_buffer_uses_median_run_length(self):
        # Ranges of lengths 3 and 5 → median = 4.
        y = np.zeros(50, dtype=np.int8)
        y[5: 8] = 1
        y[20: 25] = 1
        out = vus_summary(y, np.zeros(50), max_buffer=None, n_steps=2)
        assert out["max_buffer"] == 4

    def test_summary_no_anomaly_falls_back_to_zero_buffer(self):
        out = vus_summary(np.zeros(10), np.zeros(10), max_buffer=None)
        assert out["max_buffer"] == 0
        # All-zero labels → NaN downstream metrics, but no exception.
        assert math.isnan(out["vus_roc"])
        assert math.isnan(out["vus_pr"])
