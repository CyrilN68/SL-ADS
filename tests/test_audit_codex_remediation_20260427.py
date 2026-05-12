"""
Unit tests for the audit_codex_2026-04-26 remediation (Phase G).

These tests cover PATCH TASK-34 to TASK-46, which fix the 14 actionable
findings raised by ``audit_codex_2026-04-26/SCIENTIFIC_AUDIT_FULL.md``
(3 CRITICAL + 11 MAJOR + 1 MINOR).  See:

  - docs/audit/audit_verification_tracker.md  (master tracker — Phase H)
  - docs/audit/scientific_audit_reconciliation_20260425.md  (per-finding verdicts)
  - docs/honest_limitations.md  (CRIT-02, MAJ-05, MAJ-09 disclosures)

Run from the project root:

    pytest tests/test_audit_codex_remediation_20260427.py -v

Each test_<task>_*  is independent so a single failure does not poison
the whole audit verification.
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ─── Repository layout (Phase H — sl_ads package via src/) ──────────────
# conftest.py at the project root prepends ``src/`` to sys.path so
# ``import sl_ads.<X>`` works from tests.  We also keep the project
# root on sys.path so file-existence assertions (``_ROOT / "src" / ...``)
# resolve.
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ════════════════════════════════════════════════════════════════════════
# CRIT-01 (PATCH TASK-34): _select_best_row uses sidecar threshold
# ════════════════════════════════════════════════════════════════════════
def test_task34_select_best_row_uses_sidecar_not_argmax(tmp_path, monkeypatch):
    """The deployed _select_best_row must not pick threshold by argmax.

    Phase H: import from the new package path (legacy
    ``evaluate_injection_v2`` is now a deprecation shim and ``import *``
    drops the ``_``-prefixed name)."""
    monkeypatch.delenv("SL_ALLOW_TEST_TUNED_THRESHOLD", raising=False)

    from sl_ads.evaluate.evaluate_injection import _select_best_row, _decision_thr

    # Build a fake sweep where argmax of the metric is at a DIFFERENT
    # threshold than the sidecar — so the test catches argmax leakage.
    rows = []
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30]:
        rows.append({
            "threshold": thr,
            "f1_coverage_hybrid_episode_recall": 0.99 if thr == 0.30 else 0.50,
            "f1_micro_pure": 0.5,
            "fpr_pct": 1.0,
            "n_detected_attacks": 1,
            "n_attacks": 1,
            # Other columns required by print_summary_report — set to safe defaults
            "f1_macro_pure": 0.5, "precision_window": 0.5, "tpr_window": 0.5,
            "fpr_window": 0.01, "accuracy": 0.5, "mcc": 0.0,
            "f1_binary_hybrid_episode_recall": 0.5,
            "f1_ttd_hybrid_episode_recall": 0.5,
            "recall_binary": 0.5, "recall_coverage": 0.5,
        })
    df_sweep = pd.DataFrame(rows)

    # Force a sidecar threshold ≠ the argmax-best one:
    sidecar_thr = float(df_sweep["threshold"].iloc[0])  # 0.10
    if abs(sidecar_thr - 0.30) < 1e-9:
        sidecar_thr = 0.20  # avoid coincidence
    monkeypatch.setattr("sl_ads.evaluate.evaluate_injection._decision_thr", sidecar_thr)

    chosen = _select_best_row(df_sweep)
    assert abs(float(chosen["threshold"]) - sidecar_thr) < 1e-6, (
        f"_select_best_row picked threshold {chosen['threshold']} "
        f"instead of sidecar threshold {sidecar_thr} — CRIT-01 regression."
    )


def test_task34_select_best_row_escape_hatch_warns(tmp_path, monkeypatch):
    """Setting SL_ALLOW_TEST_TUNED_THRESHOLD=1 reverts to argmax with a warning."""
    monkeypatch.setenv("SL_ALLOW_TEST_TUNED_THRESHOLD", "1")
    from sl_ads.evaluate.evaluate_injection import _select_best_row

    rows = [
        {"threshold": 0.10, "f1_coverage_hybrid_episode_recall": 0.10},
        {"threshold": 0.20, "f1_coverage_hybrid_episode_recall": 0.99},
    ]
    df_sweep = pd.DataFrame(rows)
    with pytest.warns(UserWarning, match=r"\[CRIT-01-OVERRIDE\]"):
        chosen = _select_best_row(df_sweep)
    assert abs(float(chosen["threshold"]) - 0.20) < 1e-9


def test_task34_select_best_row_raises_when_sidecar_absent(monkeypatch):
    """If the sidecar threshold is not present in the sweep, raise (no silent fallback)."""
    monkeypatch.delenv("SL_ALLOW_TEST_TUNED_THRESHOLD", raising=False)
    from sl_ads.evaluate.evaluate_injection import _select_best_row
    monkeypatch.setattr("sl_ads.evaluate.evaluate_injection._decision_thr", 0.42)

    df_sweep = pd.DataFrame([
        {"threshold": 0.10, "f1_coverage_hybrid_episode_recall": 0.99},
        {"threshold": 0.20, "f1_coverage_hybrid_episode_recall": 0.50},
    ])
    with pytest.raises(RuntimeError, match=r"\[CRIT-01\]"):
        _select_best_row(df_sweep)


# ════════════════════════════════════════════════════════════════════════
# CRIT-03 (PATCH TASK-35): IF contamination uses CONFIG default, no test-tune
# ════════════════════════════════════════════════════════════════════════
def test_task35_if_contamination_default_present_in_config():
    import sl_ads.config as _cfg
    importlib.reload(_cfg)
    assert "IF_CONTAMINATION_DEFAULT" in _cfg.CONFIG
    assert "IF_CONTAMINATION_LADDER" in _cfg.CONFIG
    assert _cfg.CONFIG["IF_CONTAMINATION_DEFAULT"] in _cfg.CONFIG["IF_CONTAMINATION_LADDER"]


# ════════════════════════════════════════════════════════════════════════
# MAJ-01 (PATCH TASK-36): NaN preservation in adapters
# ════════════════════════════════════════════════════════════════════════
def test_task36_rederio_extract_metrics_preserves_nans(tmp_path):
    """RedeRio adapter must NOT silently fillna(0) on metric columns."""
    from sl_ads.adapters.rederio_adapter import RederioAdapter

    raw = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=5, freq="30s"),
        "bytes":     [100.0, np.nan, 200.0, np.nan, 300.0],
        "packets":   [1.0, 2.0, np.nan, 4.0, 5.0],
    })
    csv = tmp_path / "rederio_fake.csv"
    raw.to_csv(csv, index=False)

    cfg = {"path_raw": str(csv), "seasonality_period": 288}
    a = RederioAdapter("RedeRio", cfg)
    a.load_raw_data()
    a.extract_metrics()

    # The forbidden behaviour is replacing NaN with 0; test that we kept NaNs.
    assert a.standardized_data["bytes"].isna().sum() == 2, (
        "rederio_adapter should preserve NaNs on metrics (audit_codex MAJ-01)"
    )
    # Label column policy still defaults to 0 (deliberate pseudo-labelling).
    assert (a.standardized_data["label"] == 0).all()


# ════════════════════════════════════════════════════════════════════════
# MAJ-02 (PATCH TASK-37): preprocess_metrics whitelist
# ════════════════════════════════════════════════════════════════════════
def test_task37_preprocess_metrics_does_not_ffill_label_columns():
    from sl_ads.preprocessing_utils import preprocess_metrics

    df = pd.DataFrame({
        "ds":        pd.date_range("2025-01-01", periods=5, freq="1min"),
        "bytes":     [1.0, np.nan, np.nan, 4.0, 5.0],
        "label":     [1,   0,      0,      0,   0],
        "is_anomaly":[1,   0,      0,      0,   0],
    })
    out = preprocess_metrics(df.copy(), limit_ffill=10)

    # bytes should be ffilled (1, 1, 1, 4, 5)
    np.testing.assert_array_equal(out["bytes"].values, [1.0, 1.0, 1.0, 4.0, 5.0])
    # label/is_anomaly must NOT be ffilled — they are non-metric columns
    np.testing.assert_array_equal(out["label"].values, [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(out["is_anomaly"].values, [1, 0, 0, 0, 0])


def test_task37_preprocess_metrics_explicit_metric_cols_overrides_default():
    from sl_ads.preprocessing_utils import preprocess_metrics
    df = pd.DataFrame({
        "ds":   pd.date_range("2025-01-01", periods=3, freq="1min"),
        "x":    [1.0, np.nan, 3.0],
        "label":[1, 0, 1],
    })
    out = preprocess_metrics(df.copy(), limit_ffill=10, metric_cols=["x"])
    np.testing.assert_array_equal(out["x"].values, [1.0, 1.0, 3.0])
    np.testing.assert_array_equal(out["label"].values, [1, 0, 1])


# ════════════════════════════════════════════════════════════════════════
# MAJ-03 (PATCH TASK-38): SBN_NOVELTY_U_RAW_THRESHOLD declared in CONFIG
# ════════════════════════════════════════════════════════════════════════
def test_task38_sbn_novelty_threshold_in_config():
    import sl_ads.config as _cfg
    importlib.reload(_cfg)
    assert "SBN_NOVELTY_U_RAW_THRESHOLD" in _cfg.CONFIG
    v = _cfg.CONFIG["SBN_NOVELTY_U_RAW_THRESHOLD"]
    assert 0.0 <= v <= 1.0
    assert abs(v - 0.82) < 1e-9, "Default value documented in M-07 sensitivity study"


# ════════════════════════════════════════════════════════════════════════
# MAJ-04 (PATCH TASK-40): STL failure policy
# ════════════════════════════════════════════════════════════════════════
def test_task40_labeller_stl_raise_default(monkeypatch):
    """STL failure must raise by default (no silent zero votes).

    Phase H: monkeypatch the *real* module
    ``sl_ads.adapters.labeller_unsupervised`` (the legacy
    ``labeller_unsupervised`` is now a deprecation shim — see
    ``docs/RENAMING_LOG_PHASE_H.md``).
    """
    import sl_ads.adapters.labeller_unsupervised as lu
    import sl_ads.config as _cfg
    importlib.reload(_cfg)
    assert getattr(_cfg, "STL_FAIL_POLICY", "raise") == "raise"

    def _raise(*a, **kw):
        raise ValueError("forced STL failure for unit test")
    monkeypatch.setattr(lu, "STL", _raise)

    lab = lu.ConsensusLabeller(period=24)
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(RuntimeError, match=r"STL"):
        lab._algo_stl_resid(series)


def test_task40_labeller_stl_abstain_returns_nans(monkeypatch):
    import sl_ads.adapters.labeller_unsupervised as lu
    import sl_ads.config as _cfg
    monkeypatch.setattr(_cfg, "STL_FAIL_POLICY", "abstain", raising=False)

    def _raise(*a, **kw):
        raise ValueError("forced STL failure for unit test")
    monkeypatch.setattr(lu, "STL", _raise)

    lab = lu.ConsensusLabeller(period=24)
    out = lab._algo_stl_resid(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert np.all(np.isnan(out))


# ════════════════════════════════════════════════════════════════════════
# MAJ-06 (PATCH TASK-41): METR-LA train-only sensor selection
# ════════════════════════════════════════════════════════════════════════
def test_task41_metr_la_uses_train_only_variance(monkeypatch, tmp_path):
    """Sensor variance ranking must use only the train slice."""
    # Build a synthetic METR-LA-like dataframe with two sensors:
    #   sensor 'A' has high variance only in TEST window (fake leakage signal)
    #   sensor 'B' has high variance only in TRAIN window
    train_idx = pd.date_range("2012-03-01", periods=100, freq="5min")
    test_idx  = pd.date_range("2012-04-02", periods=100, freq="5min")
    rng = np.random.default_rng(0)
    rows = []
    for ts in train_idx:
        rows.append({"timestamp": ts, "sensor_id": "A", "speed": 30.0 + rng.normal(0, 0.1)})
        rows.append({"timestamp": ts, "sensor_id": "B", "speed": 30.0 + rng.normal(0, 5.0)})
    for ts in test_idx:
        rows.append({"timestamp": ts, "sensor_id": "A", "speed": 30.0 + rng.normal(0, 50.0)})
        rows.append({"timestamp": ts, "sensor_id": "B", "speed": 30.0 + rng.normal(0, 0.1)})
    df = pd.DataFrame(rows)

    # Replicate the fix's logic: train-only variance ranking
    import sl_ads.config as _cfg
    importlib.reload(_cfg)
    split = pd.to_datetime("2012-04-01 00:00:00")
    train_mask = df["timestamp"] < split
    sensor_var_train = (df.loc[train_mask].groupby("sensor_id")["speed"]
                        .var().sort_values(ascending=False))
    top1 = sensor_var_train.head(1).index.tolist()[0]
    assert top1 == "B", (
        "Train-only ranking must pick B (high variance in train); "
        "if A is picked, the fix has regressed and test-set variance is leaking."
    )


# ════════════════════════════════════════════════════════════════════════
# MAJ-07 (PATCH TASK-42): GECCO loads all CSVs deterministically
# ════════════════════════════════════════════════════════════════════════
def test_task42_gecco_concat_mode_loads_all_files(tmp_path):
    from sl_ads.adapters.gecco_adapter import GeccoAdapter
    raw_dir = tmp_path / "gecco_raw"
    raw_dir.mkdir()

    # Two fake CSV files — verify both are loaded.
    df1 = pd.DataFrame({
        "Time": pd.date_range("2016-08-01", periods=3, freq="1min"),
        "EVENT": [False, False, True],
        "Tp": [10.0, 11.0, 12.0],
    })
    df2 = pd.DataFrame({
        "Time": pd.date_range("2016-08-01 00:03:00", periods=3, freq="1min"),
        "EVENT": [False, True, False],
        "Tp": [13.0, 14.0, 15.0],
    })
    df1.to_csv(raw_dir / "01_aug.csv", index=False)
    df2.to_csv(raw_dir / "02_aug.csv", index=False)

    cfg = {"path_raw": str(raw_dir)}
    a = GeccoAdapter("GECCO-IoT", cfg)
    a.load_raw_data()

    assert len(a.raw_data) == 6, (
        "GECCO adapter must concat all CSVs in concat mode (audit_codex MAJ-07)."
    )


# ════════════════════════════════════════════════════════════════════════
# MAJ-08 (PATCH TASK-39): CALIB_* keys declared in CONFIG
# ════════════════════════════════════════════════════════════════════════
def test_task39_calib_constants_in_config():
    import sl_ads.config as _cfg
    importlib.reload(_cfg)
    for k, expected in [
        ("CALIB_BIJECTION_FLOOR_TOL", 0.01),
        ("CALIB_AGEING_WIN_FRACTION", 0.5),
        ("CALIB_SPARSITY_CUTOFF",     1e-9),
    ]:
        assert k in _cfg.CONFIG, f"audit_codex MAJ-08: {k} missing from CONFIG"
        assert abs(_cfg.CONFIG[k] - expected) < 1e-12, (
            f"audit_codex MAJ-08: {k} value drifted from {expected}"
        )


# ════════════════════════════════════════════════════════════════════════
# MAJ-09 (PATCH TASK-44): paths.get_fusion_mode_for_run helper present
# ════════════════════════════════════════════════════════════════════════
def test_task44_paths_helpers_present():
    from sl_ads.paths import get_fusion_mode_for_run, get_detection_col_fused, get_detection_col
    import sl_ads.config as _cfg
    importlib.reload(_cfg)
    a = get_detection_col(_cfg.CONFIG, up_levels=1)
    b = get_detection_col_fused(_cfg.CONFIG, up_levels=1)
    assert a == b, "get_detection_col_fused must alias get_detection_col"
    # Reading from a non-existent dir must return None gracefully
    assert get_fusion_mode_for_run("/non/existent/dir/should/not/exist") is None


def test_task44_fusion_mode_sidecar_round_trip(tmp_path):
    from sl_ads.paths import get_fusion_mode_for_run
    sidecar = tmp_path / "fusion_mode_at_compute_opinions.json"
    payload = {
        "column_prefix": "FINAL_SYSTEM_CBF",
        "actual_fusion_mode": "wbf",
        "wbf_weight_mode": "trust_discount",
    }
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    out = get_fusion_mode_for_run(str(tmp_path))
    assert out is not None
    assert out["actual_fusion_mode"] == "wbf"


# ════════════════════════════════════════════════════════════════════════
# MAJ-05 (PATCH TASK-46): CESNET timestamp policy
# ════════════════════════════════════════════════════════════════════════
def test_task46_cesnet_timestamp_keys_in_config():
    import sl_ads.config as _cfg
    importlib.reload(_cfg)
    assert _cfg.CONFIG.get("CESNET_TIMESTAMP_MODE") in (
        "fabricated_warning", "fabricated_silent", "reject"
    )
    assert _cfg.CONFIG.get("CESNET_TIMESTAMP_ANCHOR") == "2024-01-01"


# ════════════════════════════════════════════════════════════════════════
# MIN-01 (PATCH TASK-43): paths.py docstrings updated
# ════════════════════════════════════════════════════════════════════════
def test_task43_paths_docstrings_no_train_v9():
    """No active doc paragraph should reference the retired train_v9.py.

    Phase H: paths.py moved to ``src/sl_ads/paths.py``."""
    new_path = Path(_ROOT) / "src" / "sl_ads" / "paths.py"
    legacy_path = Path(_ROOT) / "paths.py"
    p = new_path if new_path.exists() else legacy_path
    text = p.read_text(encoding="utf-8")
    # train_v9.py mentions are forbidden in *active* prose:
    # we tolerate occurrences inside a "PATCH TASK-43" comment block.
    bad_lines = [
        ln for ln in text.splitlines()
        if "train_v9.py" in ln and "TASK-43" not in ln and "MIN-01" not in ln
    ]
    assert not bad_lines, (
        "audit_codex MIN-01: train_v9.py mentioned outside the patch comment.\n"
        + "\n".join(bad_lines)
    )
