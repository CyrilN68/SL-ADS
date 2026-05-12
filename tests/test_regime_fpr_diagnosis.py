"""tests/test_regime_fpr_diagnosis.py — TASK-58 (Phase B Option C).

Validates the helpers of ``sl_ads.audit.regime_fpr_diagnosis`` on
synthetic in-memory data so the test stays fast (no real
``detection_results_INJECTED.csv`` is loaded).

Covered behaviours:

  - ``_per_metric_proj_atk_columns`` filters out directional sub-
    components (``*_dir_pos_*`` / ``*_dir_neg_*``) and the system
    aggregates (``FINAL_SYSTEM_*`` / ``METHODE_*``).
  - ``_benign_only_mask`` excludes catalogue + REAL_ATTACKS windows
    using the same logic as ``evaluate_regime_fpr._excluded_mask``.
  - ``per_metric_exceedance_per_regime`` produces correct counts +
    rates per (regime, metric).
  - ``fused_proj_atk_distribution_per_regime`` returns the expected
    percentile ordering on a controlled distribution.
  - ``joint_exceedance_counts_per_regime`` counts the right
    co-occurrence frequencies.
  - ``synthesise_narrative`` returns ``H_evidence`` / ``H_fusion`` /
    ``H_correlation`` / ``H_inconclusive`` according to the documented
    discriminator thresholds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sl_ads.audit.regime_fpr_diagnosis import (
    _per_metric_proj_atk_columns,
    _benign_only_mask,
    per_metric_exceedance_per_regime,
    fused_proj_atk_distribution_per_regime,
    joint_exceedance_counts_per_regime,
    synthesise_narrative,
)
from sl_ads.calendar.regime import regime_of_series


# ──────────────────────────────────────────── column filtering

def test_per_metric_columns_filter_keeps_leaves_only():
    """Only per-leaf-metric ``*_proj_atk`` columns must survive; the
    directional sub-components and the system aggregates are dropped.
    """
    df = pd.DataFrame(columns=[
        "timestamp",
        "prophet_bytes_proj_atk",
        "prophet_bytes_dir_pos_proj_atk",
        "prophet_bytes_dir_neg_proj_atk",
        "reconst_fin_from_syn_proj_atk",
        "FINAL_SYSTEM_CBF_proj_atk",
        "METHODE_1_PROPHET_proj_atk",
        "prophet_bytes_b_atk",
    ])
    out = _per_metric_proj_atk_columns(df)
    assert "prophet_bytes_proj_atk" in out
    assert "reconst_fin_from_syn_proj_atk" in out
    assert "prophet_bytes_dir_pos_proj_atk" not in out
    assert "prophet_bytes_dir_neg_proj_atk" not in out
    assert "FINAL_SYSTEM_CBF_proj_atk" not in out
    assert "METHODE_1_PROPHET_proj_atk" not in out
    assert "prophet_bytes_b_atk" not in out


# ──────────────────────────────────────────── benign-only mask

def test_benign_only_mask_excludes_real_attacks(monkeypatch):
    """Windows inside catalogue / REAL_ATTACKS / outage events are
    excluded by ``_benign_only_mask``.  Synthesised events are
    monkey-patched into the module-level catalogues to keep the test
    self-contained.
    """
    from sl_ads.audit import regime_fpr_diagnosis as diag

    ts = pd.date_range("2025-12-01 00:00:00", periods=10, freq="h")
    df = pd.DataFrame({"timestamp": ts})

    # Catalogue event covers ts[2..3], REAL_ATTACKS covers ts[6..7].
    monkeypatch.setattr(diag, "ATTACK_CATALOG", [
        {"name": "ATK", "start": str(ts[2]), "duration_h": 2,
         "expected": "ATK"},
    ])
    monkeypatch.setitem(diag.CONFIG.setdefault("EVAL", {}),
                         "REAL_ATTACK_CATALOG", [])
    monkeypatch.setattr(diag, "REAL_ATTACKS", {
        "OUTAGE_TEST": [{
            "name": "OUTAGE",
            "type": "NETWORK_OUTAGE",
            "start": str(ts[6]),
            "end":   str(ts[8]),
        }],
    })

    mask = _benign_only_mask(df)
    benign_idx = df.index[mask].tolist()
    # ts[2,3] excluded by catalogue, ts[6,7] excluded by REAL_ATTACKS.
    assert set(benign_idx) == {0, 1, 4, 5, 8, 9}


# ──────────────────────────────────────────── C.1 per-metric exceedance

def test_per_metric_exceedance_counts_match_hand_computation():
    """C.1 — On synthetic data with two metrics and known regime
    assignment, the exceedance counts and rates must match the
    hand-computed values.
    """
    # 8 windows: 4 ACTIVE + 4 QUIET.
    ts = pd.date_range("2025-11-17 09:00:00", periods=4, freq="h").tolist()
    ts += pd.date_range("2025-11-15 12:00:00", periods=4, freq="h").tolist()
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(ts),
        "prophet_a_proj_atk":  [0.20, 0.05, 0.30, 0.40, 0.05, 0.05, 0.05, 0.05],
        "prophet_b_proj_atk":  [0.10, 0.20, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05],
    })
    regimes = regime_of_series(df["timestamp"])
    out = per_metric_exceedance_per_regime(
        df, ["prophet_a_proj_atk", "prophet_b_proj_atk"], regimes,
        delta=0.15,
    )

    a_active = out[(out["metric"] == "prophet_a_proj_atk")
                   & (out["regime"] == "ACTIVE")].iloc[0]
    a_quiet  = out[(out["metric"] == "prophet_a_proj_atk")
                   & (out["regime"] == "QUIET")].iloc[0]
    b_active = out[(out["metric"] == "prophet_b_proj_atk")
                   & (out["regime"] == "ACTIVE")].iloc[0]

    # On ACTIVE: prophet_a has values [0.20, 0.05, 0.30, 0.40] → 3 above 0.15.
    assert a_active["n_above_delta"] == 3
    assert a_active["rate"] == pytest.approx(3 / 4)

    # On QUIET: prophet_a has [0.05]*4 → 0 above.
    assert a_quiet["n_above_delta"] == 0

    # prophet_b on ACTIVE: [0.10, 0.20, 0.05, 0.10] → 1 above 0.15.
    assert b_active["n_above_delta"] == 1
    assert b_active["rate"] == pytest.approx(1 / 4)


# ──────────────────────────────────────────── C.3 fused distribution

def test_fused_distribution_orders_match_input():
    """C.3 — On a synthetic fused score that is bigger on ACTIVE, the
    ACTIVE p99.9 must be greater than the QUIET p99.9.
    """
    rng = np.random.default_rng(0)
    n_each = 1000
    ts_active = pd.date_range("2025-11-17 09:00:00", periods=n_each, freq="min")
    ts_quiet  = pd.date_range("2025-11-15 09:00:00", periods=n_each, freq="min")
    score_active = rng.gamma(2.0, 0.05, size=n_each)
    score_quiet  = rng.gamma(2.0, 0.01, size=n_each)
    df = pd.DataFrame({
        "timestamp": list(ts_active) + list(ts_quiet),
        "FINAL_SYSTEM_CBF_proj_atk": np.concatenate([score_active, score_quiet]),
    })
    regimes = regime_of_series(df["timestamp"])
    out = fused_proj_atk_distribution_per_regime(
        df, "FINAL_SYSTEM_CBF_proj_atk", regimes, delta=1.0,
    )
    p_active = float(out[out["regime"] == "ACTIVE"]["p99_9"].iloc[0])
    p_quiet  = float(out[out["regime"] == "QUIET"]["p99_9"].iloc[0])
    assert p_active > p_quiet


# ──────────────────────────────────────────── C.4 joint exceedance

def test_joint_exceedance_counts_co_occurrence():
    """C.4 — Two metrics co-occurring above ``δ`` on a single window
    must increment the k=2 count for that window's regime.
    """
    ts = [
        # Two ACTIVE windows: first has both above, second has only one.
        pd.Timestamp("2025-11-17 09:00:00"),
        pd.Timestamp("2025-11-17 10:00:00"),
        # One QUIET window: both above (clustered).
        pd.Timestamp("2025-11-15 12:00:00"),
        # One QUIET window: none above.
        pd.Timestamp("2025-11-15 13:00:00"),
    ]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(ts),
        "prophet_a_proj_atk": [0.30, 0.30, 0.30, 0.05],
        "prophet_b_proj_atk": [0.30, 0.05, 0.30, 0.05],
    })
    regimes = regime_of_series(df["timestamp"])
    out = joint_exceedance_counts_per_regime(
        df, ["prophet_a_proj_atk", "prophet_b_proj_atk"], regimes,
        delta=0.15, k_max=2,
    )
    a_k2 = out[(out["regime"] == "ACTIVE") & (out["k_min_simultaneous"] == 2)].iloc[0]
    q_k2 = out[(out["regime"] == "QUIET")  & (out["k_min_simultaneous"] == 2)].iloc[0]
    # ACTIVE: 1 of 2 windows has both above → fraction = 0.5.
    assert a_k2["fraction"] == pytest.approx(0.5)
    # QUIET: 1 of 2 windows has both above → fraction = 0.5.
    assert q_k2["fraction"] == pytest.approx(0.5)
    # k=1 (at least one alarm): ACTIVE 2/2, QUIET 1/2.
    a_k1 = out[(out["regime"] == "ACTIVE") & (out["k_min_simultaneous"] == 1)].iloc[0]
    q_k1 = out[(out["regime"] == "QUIET")  & (out["k_min_simultaneous"] == 1)].iloc[0]
    assert a_k1["fraction"] == pytest.approx(1.0)
    assert q_k1["fraction"] == pytest.approx(0.5)


# ──────────────────────────────────────────── narrative discriminator

def _build_synthetic_views(per_metric_a, per_metric_q,
                            fused_p999_a, fused_p999_q,
                            joint_k3_a, joint_k3_q):
    """Build minimal C1/C2/C3/C4 dataframes that drive the verdict."""
    c1 = pd.DataFrame([
        {"view": "C1_per_metric_exceedance", "regime": "ACTIVE",
         "metric": "m1", "rate": per_metric_a},
        {"view": "C1_per_metric_exceedance", "regime": "QUIET",
         "metric": "m1", "rate": per_metric_q},
    ])
    c2 = pd.DataFrame()
    c3 = pd.DataFrame([
        {"view": "C3_fused_distribution", "regime": "ACTIVE",
         "p99_9": fused_p999_a},
        {"view": "C3_fused_distribution", "regime": "QUIET",
         "p99_9": fused_p999_q},
    ])
    c4 = pd.DataFrame([
        {"view": "C4_joint_exceedance", "regime": "ACTIVE",
         "k_min_simultaneous": 3, "fraction": joint_k3_a},
        {"view": "C4_joint_exceedance", "regime": "QUIET",
         "k_min_simultaneous": 3, "fraction": joint_k3_q},
    ])
    return c1, c2, c3, c4


def test_synthesise_narrative_emits_H_evidence_when_per_metric_dominates():
    """When per-metric A/Q is large, the evidence hypothesis wins
    regardless of fused / joint ratios.
    """
    c1, c2, c3, c4 = _build_synthetic_views(
        per_metric_a=0.10, per_metric_q=0.01,   # ratio = 10
        fused_p999_a=0.10, fused_p999_q=0.10,
        joint_k3_a=0.01,   joint_k3_q=0.01,
    )
    s = synthesise_narrative(c1, c2, c3, c4)
    assert s["verdict"] == "H_evidence"
    assert s["median_per_metric_ratio_active_over_quiet"] == pytest.approx(10.0)


def test_synthesise_narrative_emits_H_fusion_when_fused_tail_dominates():
    """When per-metric is balanced but the fused p99.9 tail is heavier
    on ACTIVE, the fusion hypothesis wins.
    """
    c1, c2, c3, c4 = _build_synthetic_views(
        per_metric_a=0.01, per_metric_q=0.01,
        fused_p999_a=0.20, fused_p999_q=0.05,    # ratio = 4
        joint_k3_a=0.01,   joint_k3_q=0.01,
    )
    s = synthesise_narrative(c1, c2, c3, c4)
    assert s["verdict"] == "H_fusion"
    assert s["fused_p99_9_ratio_active_over_quiet"] == pytest.approx(4.0)


def test_synthesise_narrative_emits_H_correlation_when_joint_dominates():
    """When the only dominant signal is the joint k=3 ratio, the
    correlation hypothesis wins.
    """
    c1, c2, c3, c4 = _build_synthetic_views(
        per_metric_a=0.01, per_metric_q=0.01,
        fused_p999_a=0.05, fused_p999_q=0.05,
        joint_k3_a=0.10,   joint_k3_q=0.01,    # ratio = 10
    )
    s = synthesise_narrative(c1, c2, c3, c4)
    assert s["verdict"] == "H_correlation"
    assert s["joint_k3_ratio_active_over_quiet"] == pytest.approx(10.0)


def test_synthesise_narrative_emits_H_inconclusive_when_no_dominant_signal():
    """When all three ratios are below 2×, the verdict is
    ``H_inconclusive`` and the explanation invites a multi-seed
    confirmation.
    """
    c1, c2, c3, c4 = _build_synthetic_views(
        per_metric_a=0.01, per_metric_q=0.008,
        fused_p999_a=0.06, fused_p999_q=0.05,
        joint_k3_a=0.010,  joint_k3_q=0.008,
    )
    s = synthesise_narrative(c1, c2, c3, c4)
    assert s["verdict"] == "H_inconclusive"
    assert "multi-seed" in s["explanation"].lower()
