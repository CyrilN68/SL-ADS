"""
test_axelsson_ppv.py — Unit tests for the Axelsson 2000 base-rate
fallacy / per-attack PPV module.

The module realises the Bayesian PPV formula and an inverse that gives
the FPR ceiling required to reach a target PPV.  These tests exercise
the algebraic invariants (round-trip, monotonicity) plus one textbook
reference case from Axelsson (2000) Section 6 + Table 1.

Tracks TASK-18 in docs/audit/audit_verification_tracker.md.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sl_ads.evaluate.axelsson_ppv import (  # noqa: E402
    _bca_ci_proportion,
    bayesian_ppv,
    format_axelsson_md,
    min_fpr_for_ppv,
    min_tpr_for_ppv,
    per_attack_ppv_table,
)


# ════════════════════════════════════════════════════════════════════════
# Bayesian PPV formula
# ════════════════════════════════════════════════════════════════════════
class TestBayesianPPV:
    def test_axelsson_2000_textbook_case(self):
        # Axelsson 2000 §6 — pi=1e-5, TPR=0.7, FPR=1e-5 ⇒ PPV ≈ 0.4118
        p = bayesian_ppv(0.7, 1e-5, 1e-5)
        assert math.isclose(p, 0.4118, abs_tol=2e-4)

    def test_perfect_detector_zero_fpr_returns_one(self):
        assert bayesian_ppv(1.0, 0.0, 0.5) == 1.0
        assert bayesian_ppv(0.5, 0.0, 0.01) == 1.0

    def test_zero_tpr_with_any_fpr_returns_zero(self):
        assert bayesian_ppv(0.0, 0.5, 0.5) == 0.0

    def test_silent_detector_returns_nan(self):
        # tpr = fpr = 0  ⇒ no positive prediction at all  ⇒ PPV undefined.
        assert math.isnan(bayesian_ppv(0.0, 0.0, 0.5))

    def test_low_base_rate_dominates(self):
        # Same TPR/FPR, smaller base rate → smaller PPV.
        p_high = bayesian_ppv(0.9, 0.05, 0.5)
        p_low = bayesian_ppv(0.9, 0.05, 0.001)
        assert p_low < p_high

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_out_of_range_inputs_raise(self, bad):
        with pytest.raises(ValueError):
            bayesian_ppv(bad, 0.5, 0.5)
        with pytest.raises(ValueError):
            bayesian_ppv(0.5, bad, 0.5)
        with pytest.raises(ValueError):
            bayesian_ppv(0.5, 0.5, bad)


# ════════════════════════════════════════════════════════════════════════
# Inverse: min_fpr_for_ppv
# ════════════════════════════════════════════════════════════════════════
class TestMinFPRForPPV:
    @pytest.mark.parametrize("tpr,pi,target", [
        (0.9, 1e-3, 0.5),
        (0.7, 1e-5, 0.95),
        (0.5, 0.05, 0.30),
    ])
    def test_round_trip_inverse(self, tpr, pi, target):
        fpr = min_fpr_for_ppv(target, tpr, pi)
        # Plugging this FPR back into bayesian_ppv must yield exactly the
        # target (within float tolerance).
        ppv = bayesian_ppv(tpr, fpr, pi)
        assert math.isclose(ppv, target, abs_tol=1e-9)

    def test_target_one_returns_zero_fpr(self):
        assert min_fpr_for_ppv(1.0, 0.9, 0.001) == 0.0

    def test_lower_base_rate_demands_lower_fpr(self):
        f1 = min_fpr_for_ppv(0.5, 0.9, 0.01)
        f2 = min_fpr_for_ppv(0.5, 0.9, 0.001)
        assert f2 < f1

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError):
            min_fpr_for_ppv(0.0, 0.5, 0.5)
        with pytest.raises(ValueError):
            min_fpr_for_ppv(-0.1, 0.5, 0.5)
        with pytest.raises(ValueError):
            min_fpr_for_ppv(1.1, 0.5, 0.5)


class TestMinTPRForPPV:
    def test_round_trip(self):
        tpr = min_tpr_for_ppv(0.5, fpr=1e-3, base_rate=1e-3)
        ppv = bayesian_ppv(tpr, 1e-3, 1e-3)
        assert math.isclose(ppv, 0.5, abs_tol=1e-9)

    def test_infeasible_target_returns_inf(self):
        assert math.isinf(min_tpr_for_ppv(0.99, 0.5, 0.001))


# ════════════════════════════════════════════════════════════════════════
# Wilson CI on a binomial proportion
# ════════════════════════════════════════════════════════════════════════
class TestWilsonCI:
    def test_zero_n_returns_nan(self):
        lo, hi = _bca_ci_proportion(0, 0)
        assert math.isnan(lo) and math.isnan(hi)

    def test_textbook_case_10_of_100(self):
        # Wilson CI for p̂=0.10, n=100  →  ≈ (0.0552, 0.1744)
        lo, hi = _bca_ci_proportion(10, 100)
        assert abs(lo - 0.0552) < 5e-3
        assert abs(hi - 0.1744) < 5e-3

    def test_zero_successes_floored_at_zero(self):
        lo, hi = _bca_ci_proportion(0, 50)
        # Wilson with k=0 has analytic lo = 0; floating-point round-off
        # can leave a sub-ULP positive residue.
        assert lo < 1e-9
        assert 0.0 < hi < 0.10

    def test_full_success_capped_at_one(self):
        lo, hi = _bca_ci_proportion(50, 50)
        assert hi == 1.0
        assert 0.90 < lo < 1.0

    def test_ci_width_shrinks_with_n(self):
        _, hi_small = _bca_ci_proportion(5, 50)
        _, hi_large = _bca_ci_proportion(50, 500)
        # Same point estimate (0.1) but tighter CI with more data.
        assert hi_large < hi_small

    def test_effective_n_widens_wilson_interval(self):
        lo_full, hi_full = _bca_ci_proportion(10, 100)
        lo_eff, hi_eff = _bca_ci_proportion(10, 100, n_eff=25)
        assert lo_eff < lo_full
        assert hi_eff > hi_full


# ════════════════════════════════════════════════════════════════════════
# per_attack_ppv_table
# ════════════════════════════════════════════════════════════════════════
class TestPerAttackTable:
    @pytest.fixture
    def trio(self):
        ts = pd.date_range("2026-01-01", periods=100, freq="h")
        y_true = np.zeros(100, dtype=int)
        y_pred = np.zeros(100, dtype=int)
        # Single attack on hours 10-19; detector flags hours 11-15.
        y_true[10:20] = 1
        y_pred[11:16] = 1
        # One false alarm at hour 30.
        y_pred[30] = 1
        return ts, y_pred, y_true

    def test_basic_metrics_match_hand_computation(self, trio):
        ts, y_pred, y_true = trio
        attacks = [{"name": "TEST_ATTACK", "start": ts[10], "end": ts[20]}]
        table = per_attack_ppv_table(attacks, ts, y_pred, y_true)
        row = table.iloc[0]
        assert row["n_windows"] == 10
        assert math.isclose(row["base_rate"], 0.10)
        assert math.isclose(row["tpr"], 0.5)
        # 1 FP among 90 negatives.
        assert math.isclose(row["fpr_global"], 1 / 90, abs_tol=1e-9)
        expected_ppv = bayesian_ppv(0.5, 1 / 90, 0.10)
        assert math.isclose(row["ppv"], expected_ppv, abs_tol=1e-9)

    def test_zero_window_attack(self, trio):
        ts, y_pred, y_true = trio
        # An attack that does not overlap any timestamp.
        attacks = [{"name": "OUT_OF_RANGE",
                    "start": ts[-1] + pd.Timedelta(days=10),
                    "duration_h": 1}]
        table = per_attack_ppv_table(attacks, ts, y_pred, y_true)
        assert table.iloc[0]["n_windows"] == 0
        assert math.isnan(table.iloc[0]["tpr"])
        assert math.isnan(table.iloc[0]["ppv"])

    def test_duration_h_and_end_are_equivalent(self, trio):
        ts, y_pred, y_true = trio
        atk_a = {"name": "A", "start": ts[10], "duration_h": 10}
        atk_b = {"name": "B", "start": ts[10], "end": ts[20]}
        ta = per_attack_ppv_table([atk_a], ts, y_pred, y_true)
        tb = per_attack_ppv_table([atk_b], ts, y_pred, y_true)
        # Same windows under both schemas.
        assert int(ta.iloc[0]["n_windows"]) == int(tb.iloc[0]["n_windows"])
        assert math.isclose(ta.iloc[0]["tpr"], tb.iloc[0]["tpr"])

    def test_invalid_interval_raises(self, trio):
        ts, y_pred, y_true = trio
        bad = {"name": "BAD", "start": ts[20], "end": ts[10]}
        with pytest.raises(ValueError, match="end > start"):
            per_attack_ppv_table([bad], ts, y_pred, y_true)

    def test_length_mismatch_raises(self, trio):
        ts, y_pred, y_true = trio
        attacks = [{"name": "A", "start": ts[10], "end": ts[20]}]
        with pytest.raises(ValueError, match="length"):
            per_attack_ppv_table(attacks, ts, y_pred[:-1], y_true)

    def test_user_supplied_operating_fpr_overrides_empirical(self, trio):
        ts, y_pred, y_true = trio
        attacks = [{"name": "A", "start": ts[10], "end": ts[20]}]
        table = per_attack_ppv_table(
            attacks, ts, y_pred, y_true, operating_fpr=0.05
        )
        assert table.iloc[0]["fpr_global"] == 0.05

    def test_table_columns(self, trio):
        ts, y_pred, y_true = trio
        attacks = [{"name": "A", "start": ts[10], "end": ts[20]}]
        table = per_attack_ppv_table(attacks, ts, y_pred, y_true)
        for col in ("attack", "n_windows", "base_rate", "tpr",
                    "base_rate_n_eff", "tpr_n_eff", "fpr_global", "ppv",
                    "fpr_required_for_target_ppv"):
            assert col in table.columns


# ════════════════════════════════════════════════════════════════════════
# Markdown rendering
# ════════════════════════════════════════════════════════════════════════
class TestFormatAxelssonMD:
    def test_handles_empty_table(self):
        md = format_axelsson_md(pd.DataFrame(), target_ppv=0.5,
                                operating_fpr=0.01)
        assert "_(empty)_" in md
        assert "Axelsson" in md

    def test_includes_attack_name_and_columns(self):
        ts = pd.date_range("2026-01-01", periods=20, freq="h")
        attacks = [{"name": "MY_ATTACK", "start": ts[5], "end": ts[10]}]
        y_true = np.zeros(20, dtype=int)
        y_pred = np.zeros(20, dtype=int)
        y_true[5:10] = 1
        y_pred[5:10] = 1
        table = per_attack_ppv_table(attacks, ts, y_pred, y_true)
        md = format_axelsson_md(table, 0.5, float(table.iloc[0]["fpr_global"]))
        assert "MY_ATTACK" in md
        assert "fpr_required_for_target_ppv" in md
