"""tests/test_regime_fpr_alpha_sweep.py — TASK-59 sweep helpers.

Validates the building blocks of ``sl_ads.audit.regime_fpr_alpha_sweep``
on synthetic in-memory data:

  - ``_anomaly_window_mask`` unions catalogue ∪ REAL_ATTACK_CATALOG ∪
    NETWORK_OUTAGE_* and deduplicates the DDOS_ATTACK overlap.
  - ``_outage_only_mask`` returns NETWORK_OUTAGE windows only.
  - ``_apply_attack_discount`` preserves the bijection invariant
    ``b_safe + b_susp + b_atk + u = 1`` and acts only on ``b_atk``.
  - ``_resolve_leaf_keys`` filters columns to those with a complete
    leaf opinion (the seven required suffixes).
  - ``evaluate_alpha`` reports the operator-faithful F1 protocol
    (positives = anomaly_mask, negatives = benign_mask) on a small
    synthetic frame and matches hand-computed counters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sl_ads.audit.regime_fpr_alpha_sweep import (
    _anomaly_window_mask,
    _apply_attack_discount,
    _benign_only_mask,
    _outage_only_mask,
    _resolve_leaf_keys,
    evaluate_alpha,
)
from sl_ads.calendar.regime import regime_of_series
from sl_ads.core.subjective_logic import MultinomialOpinion


# ─────────────────────────────────────── anomaly mask wiring

def test_anomaly_mask_unions_catalog_and_outages(monkeypatch):
    """The operator-faithful anomaly mask unions catalogue +
    REAL_ATTACK_CATALOG + NETWORK_OUTAGE_* and deduplicates the
    DDOS_ATTACK overlap with REAL_ATTACK_CATALOG.
    """
    from sl_ads.audit import regime_fpr_alpha_sweep as sweep_mod

    ts = pd.date_range("2025-12-01 00:00:00", periods=10, freq="h")
    df = pd.DataFrame({"timestamp": ts})

    # Inject minimal synthetic catalogues that mimic the RedeRio shape:
    # - one synthetic attack at ts[1..2]
    # - one REAL_ATTACK_CATALOG event at ts[3..4]  (= "REAL_DDOS")
    # - one DDOS_ATTACK event at ts[3..4] (DUPLICATE of the REAL_DDOS;
    #   must not be double-counted)
    # - one NETWORK_OUTAGE_X event at ts[6..7]
    monkeypatch.setattr(sweep_mod, "ATTACK_CATALOG", [
        {"name": "ATK", "start": str(ts[1]), "duration_h": 2,
         "expected": "ATK"},
    ])
    monkeypatch.setitem(sweep_mod.CONFIG.setdefault("EVAL", {}),
                         "REAL_ATTACK_CATALOG", [
                             {"name": "REAL_DDOS", "start": str(ts[3]),
                              "duration_h": 2}
                         ])
    monkeypatch.setattr(sweep_mod, "REAL_ATTACKS", {
        "DDOS_ATTACK": [{
            "name": "DDOS_ATTACK",
            "start": str(ts[3]), "end": str(ts[5])
        }],
        "NETWORK_OUTAGE_X": [{
            "name": "NETWORK_OUTAGE_X",
            "start": str(ts[6]), "end": str(ts[8])
        }],
    })

    anomaly = _anomaly_window_mask(df)
    outage = _outage_only_mask(df)
    benign = _benign_only_mask(df)

    # Anomaly: ts[1,2] (catalog) + ts[3,4] (REAL_ATTACK_CATALOG /
    # DDOS_ATTACK overlap deduplicated) + ts[6,7] (NETWORK_OUTAGE_X)
    # → indices 1, 2, 3, 4, 6, 7
    assert set(df.index[anomaly].tolist()) == {1, 2, 3, 4, 6, 7}

    # Outage-only is JUST NETWORK_OUTAGE_X
    assert set(df.index[outage].tolist()) == {6, 7}

    # Benign excludes everything anomalous
    assert set(df.index[benign].tolist()) == {0, 5, 8, 9}


# ─────────────────────────────────────── contextual discount math

def test_apply_attack_discount_preserves_bijection_invariant():
    """Mercier-Quost-Denoeux 2008 contextual discount on ``b_atk`` must
    keep the multinomial-opinion invariant
    ``b_safe + b_susp + b_atk + u = 1``.
    """
    op = MultinomialOpinion([0.4, 0.3, 0.2], 0.1, [1 / 3, 1 / 3, 1 / 3])
    for alpha in (1.0, 0.8, 0.5, 0.2, 0.0):
        out = _apply_attack_discount(op, alpha)
        total = float(out.b[0] + out.b[1] + out.b[2] + out.u)
        assert abs(total - 1.0) < 1e-12, (alpha, total, out.b, out.u)
        # Only b_atk should change; b_safe / b_susp untouched.
        assert out.b[0] == op.b[0]
        assert out.b[1] == op.b[1]
        assert abs(out.b[2] - alpha * op.b[2]) < 1e-12
        assert abs(out.u - (op.u + (1 - alpha) * op.b[2])) < 1e-12


def test_apply_attack_discount_alpha_one_is_identity():
    """α=1 must return the same opinion (numerically equal)."""
    op = MultinomialOpinion([0.5, 0.2, 0.2], 0.1, [1 / 3, 1 / 3, 1 / 3])
    out = _apply_attack_discount(op, 1.0)
    assert out.b[0] == op.b[0]
    assert out.b[1] == op.b[1]
    assert out.b[2] == op.b[2]
    assert out.u == op.u


# ─────────────────────────────────────── leaf-key resolution

def test_resolve_leaf_keys_requires_complete_opinion_columns():
    """A leaf is reported only if all seven opinion components
    (b_safe, b_susp, b_atk, u, a_safe, a_susp, a_atk) are present.
    """
    df = pd.DataFrame(columns=[
        "timestamp",
        # complete set
        "prophet_bytes_b_safe", "prophet_bytes_b_susp",
        "prophet_bytes_b_atk", "prophet_bytes_u",
        "prophet_bytes_a_safe", "prophet_bytes_a_susp",
        "prophet_bytes_a_atk",
        # incomplete (missing a_atk) — must be dropped
        "prophet_packets_b_safe", "prophet_packets_b_susp",
        "prophet_packets_b_atk", "prophet_packets_u",
        "prophet_packets_a_safe", "prophet_packets_a_susp",
        # FINAL aggregate — must be filtered
        "FINAL_SYSTEM_CBF_b_safe", "FINAL_SYSTEM_CBF_b_susp",
        "FINAL_SYSTEM_CBF_b_atk", "FINAL_SYSTEM_CBF_u",
        "FINAL_SYSTEM_CBF_a_safe", "FINAL_SYSTEM_CBF_a_susp",
        "FINAL_SYSTEM_CBF_a_atk",
    ])
    keys = _resolve_leaf_keys(df)
    assert keys == ["prophet_bytes"]


# ─────────────────────────────────────── evaluate_alpha protocol

def test_evaluate_alpha_treats_outages_as_positives(monkeypatch):
    """Operator-faithful protocol: an outage that the system flags is
    a TP; an outage missed is a FN; outage windows are NEVER excluded
    from the F1 base.
    """
    # Synthetic frame: 4 anomaly windows (2 catalog + 2 outage),
    # 4 benign windows.
    ts = pd.date_range("2025-11-17 09:00:00", periods=4, freq="h").tolist()
    ts += pd.date_range("2025-11-15 12:00:00", periods=4, freq="h").tolist()
    df = pd.DataFrame({"timestamp": pd.to_datetime(ts)})

    # Index 0,1 = catalog;  index 2,3 = outage;  index 4..7 = benign.
    anomaly_idx = pd.Series(False, index=df.index)
    anomaly_idx.iloc[0:4] = True  # catalog ∪ outage
    benign_idx = pd.Series(False, index=df.index)
    benign_idx.iloc[4:8] = True
    outage_only_idx = pd.Series(False, index=df.index)
    outage_only_idx.iloc[2:4] = True
    regimes = regime_of_series(df["timestamp"])

    # System detects index 0 (a catalog), index 2 (an outage), and one
    # benign (index 5) — i.e. one TP catalog, one TP outage, one FP.
    fused = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0])
    delta = 0.4

    out = evaluate_alpha(
        df, fused, benign_idx, anomaly_idx, outage_only_idx, regimes, delta,
    )
    assert out["protocol"].startswith("operator_faithful")
    # TP = 2 (catalog at 0, outage at 2 — outage IS a positive!)
    assert out["tp"] == 2
    # FP = 1 (benign at index 5 mis-classified)
    assert out["fp"] == 1
    # FN = 2 (catalog at 1 + outage at 3 missed)
    assert out["fn"] == 2
    # TN = 3 (benign 4, 6, 7 correctly negative)
    assert out["tn"] == 3
    # Precision = TP / (TP+FP) = 2/3
    assert out["precision_window"] == pytest.approx(2 / 3)
    # Recall = TP / (TP+FN) = 2/4 = 0.5
    assert out["recall_window"] == pytest.approx(0.5)


def test_evaluate_alpha_drops_f1_when_outages_are_mostly_missed():
    """Operator-faithful protocol: when the system fails to detect
    most outages, F1 is strictly lower than the catalog-only F1
    because every missed outage contributes a FN that didn't exist
    under the legacy A3.2 protocol.

    This is the regime that holds on the RedeRio reference run
    (NETWORK_OUTAGE_DEC1617 at 51 % recall, NOV17 at 0 %), and it is
    the mechanism that drops the published F1 from 0.940 to ≈ 0.83
    when the protocol is switched.  The test fixes the missed-outage
    pattern explicitly so the inequality holds.
    """
    ts = pd.date_range("2025-11-17 09:00:00", periods=4, freq="h").tolist()
    ts += pd.date_range("2025-11-15 12:00:00", periods=10, freq="h").tolist()
    df = pd.DataFrame({"timestamp": pd.to_datetime(ts)})

    catalog_only = pd.Series(False, index=df.index)
    catalog_only.iloc[0:2] = True
    outage = pd.Series(False, index=df.index)
    outage.iloc[2:8] = True   # 6 outage windows
    benign = pd.Series(False, index=df.index)
    benign.iloc[8:14] = True  # 6 benign windows
    anomaly = catalog_only | outage
    regimes = regime_of_series(df["timestamp"])

    # Catch 1/2 catalog AND 0/6 outages (realistic for NOV17-style
    # under-detection).  No FP on benign.
    fused = np.zeros(len(df))
    fused[0] = 0.5  # catalog detected
    delta = 0.4

    op = evaluate_alpha(
        df, fused, benign, anomaly, outage, regimes, delta,
    )
    legacy = evaluate_alpha(
        df, fused, benign, catalog_only,
        pd.Series(False, index=df.index), regimes, delta,
    )
    assert op["f1_window"] < legacy["f1_window"]
    # And the 6 missed outages all show up as FN under operator-faithful
    assert op["fn"] == 1 + 6  # 1 missed catalog + 6 missed outages
    assert legacy["fn"] == 1   # only the catalog miss counts
