"""tests/test_edge_cases.py

Edge-case tests for the SL-ADS pipeline.  Covers:

  * preprocessing utilities on degenerate inputs (empty, all-NaN,
    single-row);
  * statistical utilities (bootstrap CI, McNemar) on degenerate inputs
    (n=1, all-zeros, identical predictions);
  * adapter behaviour when the input file is empty.

Phase H — added 2026-04-29.  Closes the "edge case" gap noted in
``docs/archive/2026-05-11_public_release_cleanup/top_level/RENAMING_LOG_PHASE_H.md``
Phase 7 audit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sl_ads.preprocessing_utils import preprocess_metrics  # noqa: E402
from sl_ads.stats.bootstrap_ci import paired_bootstrap_bca_ci  # noqa: E402
from sl_ads.stats.mcnemar import mcnemar_paired_test  # noqa: E402


# ────────────────────────────────────────────────────────────────────────
# 1. preprocess_metrics — degenerate inputs
# ────────────────────────────────────────────────────────────────────────
class TestPreprocessMetricsEdgeCases:
    def test_empty_dataframe_returns_empty(self):
        """preprocess_metrics on an empty df must not crash and must
        return an empty df with the same schema."""
        df = pd.DataFrame({"ds": pd.to_datetime([]), "x": []})
        out = preprocess_metrics(df.copy(), limit_ffill=10)
        assert len(out) == 0
        assert list(out.columns) == ["ds", "x"]

    def test_single_row_unchanged(self):
        df = pd.DataFrame({
            "ds": [pd.Timestamp("2025-01-01")],
            "x":  [1.5],
        })
        out = preprocess_metrics(df.copy(), limit_ffill=10)
        assert len(out) == 1
        assert out["x"].iloc[0] == 1.5

    def test_all_nan_metric_stays_nan(self):
        """A metric column that is entirely NaN must remain entirely NaN
        after forward-fill (nothing to propagate from)."""
        df = pd.DataFrame({
            "ds": pd.date_range("2025-01-01", periods=3, freq="1min"),
            "x":  [np.nan, np.nan, np.nan],
        })
        out = preprocess_metrics(df.copy(), limit_ffill=10)
        assert out["x"].isna().all()

    def test_limit_ffill_zero_disables_propagation(self):
        """``limit_ffill=0`` enforces the strict policy: NaNs are NEVER
        propagated."""
        df = pd.DataFrame({
            "ds": pd.date_range("2025-01-01", periods=4, freq="1min"),
            "x":  [1.0, np.nan, np.nan, 4.0],
        })
        out = preprocess_metrics(df.copy(), limit_ffill=0)
        # Middle two entries must remain NaN under the strict policy.
        assert out["x"].iloc[0] == 1.0
        assert pd.isna(out["x"].iloc[1])
        assert pd.isna(out["x"].iloc[2])
        assert out["x"].iloc[3] == 4.0

    def test_long_gap_truncated_at_limit(self):
        """A gap longer than ``limit_ffill`` must leave the tail as NaN."""
        df = pd.DataFrame({
            "ds": pd.date_range("2025-01-01", periods=8, freq="1min"),
            "x":  [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8.0],
        })
        out = preprocess_metrics(df.copy(), limit_ffill=2)
        # Indices 1 and 2 receive the propagated 1.0; 3..6 stay NaN.
        assert out["x"].iloc[1] == 1.0
        assert out["x"].iloc[2] == 1.0
        for i in (3, 4, 5, 6):
            assert pd.isna(out["x"].iloc[i]), f"index {i} should stay NaN beyond limit"
        assert out["x"].iloc[7] == 8.0


# ────────────────────────────────────────────────────────────────────────
# 2. bootstrap_ci — degenerate inputs
# ────────────────────────────────────────────────────────────────────────
class TestBootstrapCIEdgeCases:
    def test_identical_predictions_zero_delta(self):
        """When y_pred_a == y_pred_b, the paired delta must be exactly 0."""
        from sklearn.metrics import f1_score
        rng = np.random.default_rng(0)
        n = 200
        y_true = rng.integers(0, 2, size=n)
        y_pred = rng.integers(0, 2, size=n)
        out = paired_bootstrap_bca_ci(y_true, y_pred, y_pred, f1_score,
                                      n_boot=200, seed=42)
        # API: dict with keys ``point``, ``ci_low``, ``ci_high``, ``theta_a``,
        # ``theta_b``, ``significant_at_alpha``.  Identical predictions ⇒
        # point delta = 0 and the CI collapses to [0, 0].
        assert isinstance(out, dict)
        assert abs(float(out["point"]))   < 1e-9
        assert abs(float(out["ci_low"]))  < 1e-9
        assert abs(float(out["ci_high"])) < 1e-9
        assert out["significant_at_alpha"] is False

    def test_zero_size_inputs_raise_or_return_nan(self):
        """An empty input must NOT crash silently."""
        from sklearn.metrics import f1_score
        with pytest.raises((ValueError, IndexError, AssertionError, RuntimeError)):
            paired_bootstrap_bca_ci(np.array([]), np.array([]), np.array([]),
                                    f1_score, n_boot=10, seed=0)


# ────────────────────────────────────────────────────────────────────────
# 3. McNemar — degenerate inputs
# ────────────────────────────────────────────────────────────────────────
class TestMcNemarEdgeCases:
    def test_identical_predictions_returns_tie(self):
        """If both classifiers agree on every sample (n10 = n01 = 0),
        McNemar must return tie (p=1.0)."""
        rng = np.random.default_rng(1)
        n = 100
        y_true = rng.integers(0, 2, size=n)
        y_pred = rng.integers(0, 2, size=n)  # same predictions for both classifiers
        result = mcnemar_paired_test(y_true, y_pred, y_pred)
        # Result is a dict-like; pull p-value
        p = (result.get("p_value") if isinstance(result, dict)
             else getattr(result, "p_value", None))
        assert p is not None
        assert abs(float(p) - 1.0) < 1e-9

    def test_disagreement_p_value_in_unit_interval(self):
        rng = np.random.default_rng(2)
        n = 200
        y_true = rng.integers(0, 2, size=n)
        y_a = rng.integers(0, 2, size=n)
        y_b = rng.integers(0, 2, size=n)
        result = mcnemar_paired_test(y_true, y_a, y_b)
        p = (result.get("p_value") if isinstance(result, dict)
             else getattr(result, "p_value", None))
        assert p is not None
        assert 0.0 <= float(p) <= 1.0


# ────────────────────────────────────────────────────────────────────────
# 4. Adapter input degeneracy (empty CSV)
# ────────────────────────────────────────────────────────────────────────
class TestAdapterEmptyInput:
    def test_rederio_empty_csv_does_not_crash(self, tmp_path):
        from sl_ads.adapters.rederio_adapter import RederioAdapter
        empty = tmp_path / "rederio_empty.csv"
        # Header-only CSV (zero rows)
        pd.DataFrame({"timestamp": pd.to_datetime([]),
                       "bytes":     []}).to_csv(empty, index=False)
        cfg = {"path_raw": str(empty), "seasonality_period": 288}
        a = RederioAdapter("RedeRio", cfg)
        a.load_raw_data()
        # extract_metrics must not crash on an empty input
        a.extract_metrics()
        assert a.standardized_data is not None
        assert len(a.standardized_data) == 0
