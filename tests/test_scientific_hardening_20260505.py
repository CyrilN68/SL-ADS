from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
from sklearn.metrics import f1_score

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def test_real_attack_catalog_is_subtracted_from_global_far(monkeypatch):
    from sl_ads.evaluate import evaluate_qualify_sbn as eval_sbn

    ts = pd.date_range("2025-11-12 00:00:00", periods=6, freq="h")
    df = pd.DataFrame({
        "timestamp": ts,
        "gate_open": [False, True, True, True, True, False],
    })
    real_ddos = {
        "name": "REAL_DDOS",
        "type": "DDoS",
        "expected_qualif": "UDP_FLOOD",
        "start": str(ts[1]),
        "duration_h": 2,
    }
    monkeypatch.setattr(eval_sbn, "REAL_ATTACKS", {})
    monkeypatch.setitem(eval_sbn.CONFIG["EVAL"], "REAL_ATTACK_CATALOG", [real_ddos])

    stats = eval_sbn._compute_global_detection_stats(
        df, attack_events=[], outage_events=[]
    )

    assert stats["real_attack_catalog_subtracted_from_far"] is True
    assert stats["real_attack_catalog_events"] == 1
    assert stats["n_attack_windows"] == 2
    assert stats["TP"] == 2
    assert stats["FP"] == 2
    assert stats["TN"] == 2


def test_global_detection_stats_uses_right_open_outage_intervals(monkeypatch):
    from sl_ads.evaluate import evaluate_qualify_sbn as eval_sbn

    ts = pd.date_range("2025-12-16 10:00:00", periods=6, freq="h")
    df = pd.DataFrame({
        "timestamp": ts,
        "gate_open": [False, False, True, True, True, False],
    })
    outage = {
        "name": "NETWORK_OUTAGE_TEST",
        "type": "NETWORK_OUTAGE",
        "expected_qualif": "NETWORK_OUTAGE",
        "start": str(ts[2]),
        "end": str(ts[4]),
    }
    monkeypatch.setattr(eval_sbn, "REAL_ATTACKS", {"NETWORK_OUTAGE_TEST": [outage]})
    monkeypatch.setitem(eval_sbn.CONFIG["EVAL"], "REAL_ATTACK_CATALOG", [])

    stats = eval_sbn._compute_global_detection_stats(
        df, attack_events=[], outage_events=[]
    )

    assert stats["n_outage_windows"] == 2
    assert stats["gate_open_during_outage"] == 2
    assert stats["n_normal_windows"] == 4
    assert stats["FP"] == 1
    assert stats["TN"] == 3


def test_bca_bootstrap_supports_moving_blocks():
    from sl_ads.stats.bootstrap_ci import bootstrap_bca_ci, paired_bootstrap_bca_ci

    y_true = np.array([0, 0, 1, 1] * 30)
    y_pred_a = y_true.copy()
    y_pred_b = np.roll(y_true, 1)

    res = bootstrap_bca_ci(
        y_true, y_pred_a, f1_score, n_boot=80, seed=1, block_length=4
    )
    assert res["resampling"] == "moving_block"
    assert res["block_length"] == 4
    assert res["method"] == "BCa-block"

    paired = paired_bootstrap_bca_ci(
        y_true, y_pred_a, y_pred_b, f1_score, n_boot=80, seed=2, block_length=4
    )
    assert paired["resampling"] == "moving_block"
    assert paired["block_length"] == 4


def test_pwm_gpd_fit_recovers_known_parameters():
    """PWM (Hosking-Wallis 1987) on a synthetic GPD sample must
    recover xi, sigma to within a few percent — verifies the closed-form
    estimator is numerically correct.
    """
    from sl_ads.train.train_models import _pwm_gpd_fit
    rng = np.random.default_rng(7)
    xi_true, sigma_true = 0.20, 1.5
    # Inverse-CDF sampling for GPD: x = (sigma/xi) * ((1-U)^{-xi} - 1)
    u = rng.uniform(size=5000)
    x = (sigma_true / xi_true) * ((1.0 - u) ** (-xi_true) - 1.0)
    xi_hat, sigma_hat = _pwm_gpd_fit(x)
    assert abs(xi_hat - xi_true) < 0.05, (xi_hat, xi_true)
    assert abs(sigma_hat - sigma_true) < 0.15, (sigma_hat, sigma_true)


def test_pwm_gpd_fit_handles_heavy_tail_xi_above_half():
    """Grimshaw MLE is documented as unstable for xi > 0.5; PWM must
    still produce a finite estimate close to the true xi.
    """
    from sl_ads.train.train_models import _pwm_gpd_fit
    rng = np.random.default_rng(11)
    xi_true, sigma_true = 0.70, 2.0
    u = rng.uniform(size=8000)
    x = (sigma_true / xi_true) * ((1.0 - u) ** (-xi_true) - 1.0)
    xi_hat, sigma_hat = _pwm_gpd_fit(x)
    # PWM bias grows with xi but stays bounded; tolerate ±0.10.
    assert abs(xi_hat - xi_true) < 0.10
    assert sigma_hat > 0


def test_a47_cbf_emits_dependence_warning(monkeypatch):
    """A4.7 — switching INTER_METHOD_FUSION to 'cbf' must surface the
    independence-violation warning on RedeRio (cross max |rho|=0.915).

    Note: switching the fusion mode without recalibrating the threshold
    sidecar is exactly the situation the A1.9 hard-raise (PATCH TASK-45,
    enforced 2026-05-06) is meant to block. We therefore opt into the
    documented ablation bypass ``SL_ALLOW_THRESHOLD_FUSION_MISMATCH_FOR_ABLATION=1``
    so the import reaches the warning emission site instead of being
    pre-empted by the sidecar/config-mismatch ``RuntimeError`` raised in
    ``paths.validate_threshold_sidecar_config``.
    """
    import importlib
    import warnings as _warnings
    from sl_ads.config import CONFIG

    monkeypatch.setitem(CONFIG, "INTER_METHOD_FUSION", "cbf")
    # Keep this unit test independent from long-run evidence artifacts.
    # The assertion is about the fusion warning, not injection-file routing.
    monkeypatch.setitem(CONFIG, "ATTACK_CATALOG", [])
    monkeypatch.setenv("SL_ALLOW_THRESHOLD_FUSION_MISMATCH_FOR_ABLATION", "1")
    # Force a fresh import of the module so the top-level warning fires.
    import sl_ads.core.opinions_pipeline as op_pipe

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        importlib.reload(op_pipe)
    msgs = [str(w.message) for w in caught]
    assert any("INTER_METHOD_FUSION='cbf'" in m and "Theorem 12.2" in m
               for m in msgs), msgs


def test_a35_catalog_validator_rejects_pre_split_event(monkeypatch):
    """A3.5 — _validate_catalog must hard-raise when any injected catalog
    event starts on or before CONFIG['split_date']. Otherwise the threshold
    calibrator silently sees the attack as 'normal' and biases delta.
    """
    from sl_ads.inject import evidence_level as ev_mod
    from sl_ads.config import CONFIG

    # Build a catalog entry that lands BEFORE the configured split_date.
    bad_event = {
        "name": "BAD_PRE_SPLIT",
        "type": "DDoS",
        "start": "1900-01-01 00:00:00",
        "duration_h": 1.0,
        "ramp_frac": 0.1,
        "signature": {"prophet_packets": (1, 0, 0)},
    }

    import pytest
    with pytest.raises(ValueError, match="A3.5"):
        ev_mod._validate_catalog([bad_event])


def test_a35_catalog_validator_accepts_post_split_event():
    """Sanity: a well-formed post-split-date entry must not raise."""
    from sl_ads.inject import evidence_level as ev_mod
    from sl_ads.config import CONFIG

    after = (pd.Timestamp(CONFIG["split_date"]) + pd.Timedelta(days=1))
    good_event = {
        "name": "GOOD_POST_SPLIT",
        "type": "DDoS",
        "start": str(after),
        "duration_h": 1.0,
        "ramp_frac": 0.1,
        "signature": {"prophet_packets": (1, 0, 0)},
    }
    ev_mod._validate_catalog([good_event])  # Must not raise.


def test_signature_noise_supports_heavy_tail_distributions():
    """A3.3 follow-up — the perturbation helper must support Cauchy and
    Student-t tails in addition to Gaussian, while preserving the
    bijection invariant (clipped, renormalised triplets summing to 1).
    """
    from sl_ads.ablation.ablation_signature_noise import _draw_noise, _perturb_row

    rng = np.random.default_rng(0)

    # Distribution dispatch yields finite arrays with the right shape.
    g = _draw_noise("gaussian", 0.1, 3.0, np.random.default_rng(0))
    c = _draw_noise("cauchy", 0.1, 3.0, np.random.default_rng(0))
    t = _draw_noise("student_t", 0.1, 3.0, np.random.default_rng(0))
    assert g.shape == c.shape == t.shape == (3,)
    assert np.all(np.isfinite(g))  # Gaussian draws are always finite.

    try:
        _draw_noise("laplace", 0.1, 3.0, rng)
    except ValueError:
        pass
    else:  # pragma: no cover — defensive
        raise AssertionError("Unknown distribution must raise ValueError.")

    # Perturbation preserves the (b_safe + b_susp + b_atk) = 1 bijection
    # invariant after clip + renorm, even with a heavy-tailed Cauchy draw.
    row = pd.Series({
        "x_proj_safe": 0.4,
        "x_proj_susp": 0.4,
        "x_proj_atk": 0.2,
    })
    for distribution in ("gaussian", "cauchy", "student_t"):
        rng_local = np.random.default_rng(42)
        out = _perturb_row(row, ["x"], 0.2, rng_local,
                            distribution=distribution, df_t=3.0)
        triplet = out[["x_proj_safe", "x_proj_susp", "x_proj_atk"]].astype(float).to_numpy()
        assert np.all(triplet >= 0.0), distribution
        assert abs(triplet.sum() - 1.0) < 1e-9, (distribution, triplet)


def test_threshold_sweep_excludes_real_attacks_outage_from_fpr(monkeypatch):
    """A3.2 — outage windows from REAL_ATTACKS must NOT count as FP in
    the threshold-sweep FPR (they are operational network outages, not
    quiet normal traffic).  Without this fix the headline FPR is
    inflated 5×–10× compared to the regime-by-regime audit.
    """
    from sl_ads.evaluate import evaluate_injection as ev

    ts = pd.date_range("2025-12-01 00:00:00", periods=10, freq="h")
    df = pd.DataFrame({
        "timestamp": ts,
        ev.COL_DET: [0.0, 0.0, 0.5, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
    })
    catalog = [
        {"name": "ATK", "start": str(ts[2]), "duration_h": 2, "expected": "ATK"},
    ]
    real_outage = {
        "name": "OUTAGE",
        "type": "NETWORK_OUTAGE",
        "start": str(ts[5]),
        "end": str(ts[7]),
    }
    monkeypatch.setattr(
        ev, "_real_attacks_iter", lambda: iter([real_outage])
    )
    outside = ev.windows_outside_attacks(df, catalog)
    # Catalog window 2-3 + outage [5,7) are excluded (right-open boundary).
    # Indices left: 0, 1, 4, 7, 8, 9.
    assert set(outside.tolist()) == {0, 1, 4, 7, 8, 9}


def test_f1_protocol_comparison_reports_operator_faithful_outages(monkeypatch):
    """TASK-60 — paper outputs must expose both F1 protocols explicitly.

    The catalog protocol keeps outage windows out of the F1 base. The
    operator-faithful protocol counts the same outage as an anomaly positive,
    creating FN when the detector misses outage windows.
    """
    from sl_ads.evaluate import evaluate_injection as ev

    ts = pd.date_range("2025-12-01 00:00:00", periods=6, freq="h")
    df = pd.DataFrame({
        "timestamp": ts,
        ev.COL_DET: [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
    })
    catalog = [
        {"name": "ATK", "start": str(ts[1]), "duration_h": 1, "expected": "ATK"},
    ]
    real_outage = {
        "name": "NETWORK_OUTAGE_DEC1617",
        "type": "NETWORK_OUTAGE",
        "start": str(ts[3]),
        "end": str(ts[5]),
    }
    monkeypatch.setattr(ev, "_real_attacks_iter", lambda: iter([real_outage]))

    out = ev.f1_protocol_comparison(df, catalog, threshold=0.4)
    by_protocol = out.set_index("protocol")

    catalog_row = by_protocol.loc["catalog_outages_separate"]
    anomaly_row = by_protocol.loc["operator_faithful_anomaly"]

    assert int(catalog_row["n_positive"]) == 1
    assert int(catalog_row["fn"]) == 0
    assert float(catalog_row["f1_micro_pure"]) == pytest.approx(1.0)

    assert int(anomaly_row["n_positive"]) == 3
    assert int(anomaly_row["fn"]) == 1
    assert float(anomaly_row["f1_micro_pure"]) < float(catalog_row["f1_micro_pure"])
