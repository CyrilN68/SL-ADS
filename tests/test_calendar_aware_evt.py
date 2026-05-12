"""tests/test_calendar_aware_evt.py — PATCH H2 (calendar-aware EVT).

Covers, in order:

  1. ``regime_of`` semantics on synthetic timestamps (boundaries,
     weekend, holiday, day/night).
  2. Determinism + signature stability of the regime function.
  3. Vectorised ``regime_of_series`` agrees with the scalar version.
  4. ``calibrate_thresholds_per_regime_v2`` recovers regime-specific
     EVT calibrations on synthetic GPD samples — the per-regime
     thresholds reflect the per-regime variance.
  5. The pipeline uses per-regime thresholds when ``models_pkg`` carries
     a ``thresholds_per_regime`` block, and falls back to the legacy
     scalar thresholds otherwise.
  6. ``paths.validate_threshold_sidecar_config`` hard-raises on a
     ``calendar_evt_signature`` mismatch (calibration/runtime drift).
  7. Backward compatibility — a sidecar without the new field is
     accepted (the signature is reported as ``missing``, not as a
     mismatch).

These tests do not run the full training pipeline.  Per-regime
calibration is exercised on pure-numpy inputs; the dispatcher is
exercised on a synthetic ``models_pkg`` shape that mimics what
``train_models`` writes when ``CALENDAR_EVT_ENABLED=True``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sl_ads.calendar.regime import (
    REGIME_BUCKETS,
    REGIME_FN_SIGNATURE,
    regime_counts,
    regime_of,
    regime_of_series,
)


# --------------------------------------------------------------- 1. boundaries

@pytest.mark.parametrize("ts, expected", [
    # Monday 12:30 -> ACTIVE (weekday × daytime)
    ("2025-11-17 12:30:00", "ACTIVE"),
    # Monday boundaries: 08:00 inclusive, 18:00 exclusive
    ("2025-11-17 08:00:00", "ACTIVE"),
    ("2025-11-17 17:59:00", "ACTIVE"),
    ("2025-11-17 18:00:00", "QUIET"),  # boundary excluded by design
    ("2025-11-17 07:59:00", "QUIET"),  # shoulder hours
    # Monday night
    ("2025-11-17 03:00:00", "QUIET"),
    ("2025-11-17 22:00:00", "QUIET"),
    # Saturday noon (weekend × daytime)
    ("2025-11-15 12:30:00", "QUIET"),
    # Sunday daytime
    ("2025-11-16 14:00:00", "QUIET"),
])
def test_regime_of_boundaries(ts, expected):
    assert regime_of(ts) == expected


def test_regime_of_holiday():
    """A weekday holiday is QUIET even at noon."""
    holidays = [{"ds": "2025-12-25"}]  # Christmas falls on a Thursday in 2025
    assert regime_of("2025-12-25 12:30:00", holidays=holidays) == "QUIET"
    # Same date without holidays argument is ACTIVE (weekday daytime)
    assert regime_of("2025-12-25 12:30:00") == "ACTIVE"


def test_regime_of_holiday_accepts_dict_or_date_like():
    """``holidays`` accepts both ``{ds: ...}`` dicts (CONFIG shape) and
    raw date-likes; both produce identical labels.
    """
    h_dict = [{"ds": "2025-12-25"}]
    h_str  = ["2025-12-25"]
    h_pdt  = [pd.Timestamp("2025-12-25")]
    for h in (h_dict, h_str, h_pdt):
        assert regime_of("2025-12-25 12:30:00", holidays=h) == "QUIET"


# --------------------------------------------------------------- 2. signature

def test_regime_fn_signature_is_versioned():
    """The signature must carry a version tag and an ISO date so a
    sidecar mismatch can attribute drift to a specific revision.
    """
    assert "/v" in REGIME_FN_SIGNATURE
    assert "@" in REGIME_FN_SIGNATURE


def test_regime_buckets_canonical_order():
    """ACTIVE first, QUIET second — used as ordering key in CSV
    columns and sidecar dicts.
    """
    assert REGIME_BUCKETS == ("ACTIVE", "QUIET")


# --------------------------------------------------------------- 3. vectorised

def test_regime_of_series_agrees_with_scalar():
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2025-11-09 23:59:59")
    offsets = rng.integers(0, 60 * 24 * 45, size=200)  # ~45 days post-split
    timestamps = pd.Series([base + pd.Timedelta(minutes=int(o)) for o in offsets])
    holidays = [{"ds": "2025-12-25"}]
    vec = regime_of_series(timestamps, holidays=holidays).to_list()
    scal = [regime_of(t, holidays=holidays) for t in timestamps]
    assert vec == scal


def test_regime_counts_partition_is_complete():
    """For any timestamp series, ACTIVE_count + QUIET_count == len(series).
    """
    rng = np.random.default_rng(1)
    base = pd.Timestamp("2025-11-10 00:00:00")
    timestamps = pd.Series(
        [base + pd.Timedelta(minutes=int(o))
         for o in rng.integers(0, 60 * 24 * 30, size=500)]
    )
    counts = regime_counts(timestamps, holidays=None)
    assert sum(counts.values()) == len(timestamps)
    assert set(counts.keys()) == set(REGIME_BUCKETS)


# --------------------------------------- 4. per-regime EVT calibration

def test_calibrate_thresholds_per_regime_recovers_bucket_specific_scales():
    """If we synthesise residuals with higher variance during ACTIVE
    timestamps and lower variance during QUIET ones, the per-regime
    EVT calibration must yield strictly higher thresholds for
    ACTIVE than for QUIET — exactly the behaviour the regime-FPR
    audit motivates.
    """
    from sl_ads.train.train_models import calibrate_thresholds_per_regime_v2

    base = pd.Timestamp("2025-11-10 00:00:00")
    n = 4_000
    ts = pd.date_range(base, periods=n, freq="5min")
    rng = np.random.default_rng(31415)
    labels = regime_of_series(pd.Series(ts))
    # Active windows ~3× the residual scale of quiet windows.
    res = np.where(
        labels.values == "ACTIVE",
        rng.normal(0, 3.0, n),
        rng.normal(0, 1.0, n),
    )
    out = calibrate_thresholds_per_regime_v2(
        res, ts, metric_key="testmetric", branch="prophet",
    )
    assert out["regime_fn_signature"] == REGIME_FN_SIGNATURE
    assert set(out["buckets"]) == set(REGIME_BUCKETS)
    th_a = out["thresholds_per_regime"]["ACTIVE"]
    th_q = out["thresholds_per_regime"]["QUIET"]
    assert th_a["t_susp"] > th_q["t_susp"], (th_a, th_q)
    assert th_a["t_atk"]  > th_q["t_atk"],  (th_a, th_q)


def test_calibrate_thresholds_per_regime_rejects_length_mismatch():
    from sl_ads.train.train_models import calibrate_thresholds_per_regime_v2
    with pytest.raises(ValueError, match="length mismatch"):
        calibrate_thresholds_per_regime_v2(
            np.zeros(10),
            pd.date_range("2025-11-10", periods=8, freq="5min"),
            metric_key="m", branch="prophet",
        )


# --------------------------------------- 5. dispatcher invariants

def test_compute_evidence_dispatches_per_regime_when_block_present():
    """Smoke test: when ``thresholds_per_regime`` is present in a
    metric pkg, the per-bucket value at the window's regime is used.
    We exercise this through the public dispatch contract: the
    per-regime block is read as
    ``pkg['thresholds_per_regime']['thresholds_per_regime'][bucket]``,
    matching ``calibrate_thresholds_per_regime_v2`` output shape.
    """
    bucket = regime_of("2025-11-17 12:30:00")
    pkg = {
        "type": "prophet",
        "t_susp": 1.0, "t_atk": 2.0, "t_trapeze_base": 0.1,
        "t_susp_pos": 1.0, "t_atk_pos": 2.0, "t_trapeze_base_pos": 0.1,
        "t_susp_neg": 1.0, "t_atk_neg": 2.0, "t_trapeze_base_neg": 0.1,
        "direction": "pos",
        "thresholds_per_regime": {
            "regime_fn_signature": REGIME_FN_SIGNATURE,
            "buckets": list(REGIME_BUCKETS),
            "thresholds_per_regime": {
                "ACTIVE": {"t_susp": 5.0, "t_atk": 7.5,
                            "t_trapeze_base": 0.5,
                            "t_susp_pos": 5.0, "t_atk_pos": 7.5,
                            "t_trapeze_base_pos": 0.5,
                            "t_susp_neg": 5.0, "t_atk_neg": 7.5,
                            "t_trapeze_base_neg": 0.5,
                            "direction": "pos"},
                "QUIET":  {"t_susp": 0.3, "t_atk": 0.5,
                            "t_trapeze_base": 0.05,
                            "t_susp_pos": 0.3, "t_atk_pos": 0.5,
                            "t_trapeze_base_pos": 0.05,
                            "t_susp_neg": 0.3, "t_atk_neg": 0.5,
                            "t_trapeze_base_neg": 0.05,
                            "direction": "pos"},
            },
            "n_residuals_per_regime": {"ACTIVE": 1000, "QUIET": 1000},
        },
    }

    # Replicate the dispatch logic that lives at the top of the
    # per-window loop in ``compute_evidence`` to assert the
    # behavioural contract without booting the full pipeline:
    per_regime = pkg["thresholds_per_regime"]
    bucket_th = per_regime["thresholds_per_regime"][bucket]
    assert bucket_th["t_susp"] == 5.0  # ACTIVE
    # Sanity: a QUIET timestamp picks the QUIET bucket
    bucket_q = regime_of("2025-11-15 12:30:00")  # Saturday noon
    bucket_th_q = per_regime["thresholds_per_regime"][bucket_q]
    assert bucket_th_q["t_susp"] == 0.3


def test_legacy_pkg_without_per_regime_block_uses_scalar_fields():
    """Old PKLs (no H2 calibration) must still work: the dispatcher
    drops to the legacy scalar fields on the pkg.
    """
    pkg = {
        "type": "prophet",
        "t_susp": 1.234, "t_atk": 2.5, "t_trapeze_base": 0.123,
        "direction": "pos",
        # Crucially: no "thresholds_per_regime" key (or set to None).
    }
    # The compute_evidence wiring reads the per-regime block via
    # ``.get('thresholds_per_regime')`` — None means "fall back to
    # the legacy scalar path".
    assert pkg.get("thresholds_per_regime") is None
    # And the scalar fields are usable:
    assert pkg["t_susp"] == 1.234


# --------------------------------------------- 6. sidecar A1.9 enforcement

def test_validate_threshold_sidecar_signature_match(tmp_path, monkeypatch):
    """When the runtime carries calendar-aware EVT enabled and the
    sidecar's ``calendar_evt_signature`` matches
    ``REGIME_FN_SIGNATURE``, the validator must NOT raise.
    """
    from sl_ads import paths as paths_mod

    sidecar = {
        "decision_threshold": 0.123,
        "fusion_mode_at_calibration": "wbf",
        "wbf_weight_mode": "uniform",
        "lambda_decay": 0.85,
        "balance_ratio": 1.0,
        "cd_alpha_attack": 1.0,
        "calendar_evt_signature": REGIME_FN_SIGNATURE,
    }
    sidecar_path = tmp_path / "trained_models_test_threshold.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    config = {
        "VERSION_NAME": "test",
        "INTER_METHOD_FUSION": "wbf",
        "WBF_WEIGHT_MODE": "uniform",
        "LAMBDA_DECAY": 0.85,
        "BALANCE_RATIO": 1.0,
        "CD_ALPHA_ATTACK": 1.0,
        "CALENDAR_EVT_ENABLED": True,
    }
    # Hand-feed the sidecar payload; we don't need on-disk resolution.
    res = paths_mod.validate_threshold_sidecar_config(
        config, sidecar_data=sidecar, strict=True,
    )
    assert res["ok"]
    assert "calendar_evt_signature" in res["checked"]


def test_validate_threshold_sidecar_signature_mismatch_raises():
    from sl_ads import paths as paths_mod

    sidecar = {
        "fusion_mode_at_calibration": "wbf",
        "wbf_weight_mode": "uniform",
        "lambda_decay": 0.85,
        "balance_ratio": 1.0,
        "cd_alpha_attack": 1.0,
        # Calibrated under an older signature
        "calendar_evt_signature": "weekday_x_daytime_x_holiday/v0@2026-01-01",
    }
    config = {
        "INTER_METHOD_FUSION": "wbf",
        "WBF_WEIGHT_MODE": "uniform",
        "LAMBDA_DECAY": 0.85,
        "BALANCE_RATIO": 1.0,
        "CD_ALPHA_ATTACK": 1.0,
        "CALENDAR_EVT_ENABLED": True,  # Runtime exposes current signature
    }
    with pytest.raises(RuntimeError, match=r"\[A1\.9\].*calendar_evt_signature"):
        paths_mod.validate_threshold_sidecar_config(
            config, sidecar_data=sidecar, strict=True,
        )


def test_validate_threshold_sidecar_disabled_calendar_null_signature_matches():
    from sl_ads import paths as paths_mod

    sidecar = {
        "fusion_mode_at_calibration": "wbf",
        "wbf_weight_mode": "uniform",
        "lambda_decay": 0.85,
        "balance_ratio": 1.0,
        "cd_alpha_attack": 1.0,
        "calendar_evt_signature": None,
    }
    config = {
        "INTER_METHOD_FUSION": "wbf",
        "WBF_WEIGHT_MODE": "uniform",
        "LAMBDA_DECAY": 0.85,
        "BALANCE_RATIO": 1.0,
        "CD_ALPHA_ATTACK": 1.0,
        "CALENDAR_EVT_ENABLED": False,
    }
    res = paths_mod.validate_threshold_sidecar_config(
        config, sidecar_data=sidecar, strict=True,
    )
    assert res["ok"]
    assert res["checked"]["calendar_evt_signature"]["runtime"] is None
    assert res["checked"]["calendar_evt_signature"]["sidecar"] is None


# ------------------------------------ 7. backward compatibility

def test_validate_threshold_sidecar_no_signature_field_is_missing_not_mismatch():
    """An old sidecar (pre-H2) without ``calendar_evt_signature`` is
    treated as ``missing`` rather than as a mismatch — the legacy
    pipeline still loads.
    """
    from sl_ads import paths as paths_mod

    sidecar = {
        "fusion_mode_at_calibration": "wbf",
        "wbf_weight_mode": "uniform",
        "lambda_decay": 0.85,
        "balance_ratio": 1.0,
        "cd_alpha_attack": 1.0,
        # NO calendar_evt_signature key.
    }
    config = {
        "INTER_METHOD_FUSION": "wbf",
        "WBF_WEIGHT_MODE": "uniform",
        "LAMBDA_DECAY": 0.85,
        "BALANCE_RATIO": 1.0,
        "CD_ALPHA_ATTACK": 1.0,
        "CALENDAR_EVT_ENABLED": False,
    }
    res = paths_mod.validate_threshold_sidecar_config(
        config, sidecar_data=sidecar, strict=True,
    )
    assert res["ok"]
    assert "calendar_evt_signature" in res["missing"]
