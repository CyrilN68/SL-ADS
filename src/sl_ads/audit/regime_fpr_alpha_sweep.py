"""regime_fpr_alpha_sweep.py — Pareto sweep of α_attack for the per-regime
contextual discount candidate fix (TASK-58 follow-up).

Background
----------
TASK-58 (`regime_fpr_diagnosis`) was refreshed after the 2026-05-10
full RedeRio rerun.  The current reproducible artifact is effectively
reconstruction-only because the local Prophet backend is unavailable;
the diagnostic verdict is ``H_evidence`` (median per-metric
ACTIVE/QUIET exceedance ratio = 32.57×).  The α sweep below is kept as
the empirical check for a per-regime contextual discount
(Mercier-Quost-Denoeux 2008) on the volumetric leaves' ``b_attack``
mass during ACTIVE windows.

This module **does not implement** the fix in production.  It runs an
exploratory Pareto sweep that asks the empirical question:

> Does there exist any ``α_attack ∈ (0, 1]`` applied per-regime to
> the volumetric group such that the realised FPR moves toward the
> operator target without sacrificing more than ~1 point of F1?

The answer determines whether TASK-59 (the production fix) is worth
implementing.

Methodology
-----------
1. Load the canonical detection results CSV (already contains all
   per-leaf-metric post-ageing opinion components plus base rates).
2. For each ``α`` in the sweep grid:
   a. For each window labelled regime = ACTIVE, apply the
      Mercier-Quost-Denoeux 2008 contextual discount with α_atk = α
      to the volumetric leaves (``b'_atk = α · b_atk``,
      ``u' = u + (1 - α) · b_atk``).
   b. Re-fuse the Prophet group via the production WBF operator
      (`fusion_wbf_n_sources`, evidence-space, uniform external
      weights — same as the published headline configuration).
   c. Re-fuse the Reconstruction group identically (untouched by the
      discount unless its own volumetric reconstructions are listed
      in the candidate volumetric set).
   d. Compute the inter-method WBF on the two group opinions to
      obtain the fused ``proj_atk_FINAL`` for that window.
   e. Apply the published decision threshold ``δ`` and tally
      detection statistics:
         - per-regime FPR (FPR_ACTIVE, FPR_QUIET)
         - global FPR
         - per-event recall (synthetic catalog, real DDoS, outages)
         - operator-faithful F1/MCC where outages are positives
3. Output a long-format CSV (one row per α) and a markdown summary
   so a reviewer can pick the operating α without us picking it.

Caveat
------
The sweep is run on the **test span** (the canonical detection CSV
covers post-split windows).  Picking α from this curve and reporting
metrics on the same span would be test-set tuning.  The output is
therefore tagged ``exploratory_pareto`` and any production α must be
re-calibrated on the train-calib hold-out before deployment.  See
``docs/review/regime_fpr_root_cause_analysis.md`` §6.3 and §4 for the
discipline.

References
----------
- Mercier, D., Quost, B., Denoeux, T. (2008).  *Contextual discounting
  of belief functions*.  Information Fusion 9(2), 246–258.
- Joesang, A. (2016).  *Subjective Logic*.  Springer §12.5, §14.3.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sl_ads.calendar.regime import regime_of_series
from sl_ads.config import CONFIG, REAL_ATTACKS
from sl_ads.core.subjective_logic import (
    MultinomialOpinion,
    fusion_wbf_n_sources,
)
from sl_ads.inject.evidence_level import ATTACK_CATALOG
from sl_ads.paths import get_decision_threshold


# ───────────────────────────────────────────── candidate volumetric set

# The verdict §6.1 (regime_fpr_root_cause_analysis.md) identifies the
# pairs with Δ |ρ| > 0.5 between ACTIVE and QUIET. Their *common* leaf
# names (after deduplication) form the candidate group below. Reviewers
# can swap this list at the CLI via ``--volumetric-leaves``.
DEFAULT_VOLUMETRIC_LEAVES = (
    "prophet_bytes",
    "prophet_packets",
    "prophet_flows",
    "prophet_avg_pkt_size",
    "prophet_entropy_dst_port",
    "reconst_bytes_from_packets",
    "reconst_tcp_from_packets",
)


# ───────────────────────────────────────────── helpers reused from C diagnosis

def _event_bounds(ev: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    t0 = pd.Timestamp(ev["start"])
    if ev.get("end") is not None:
        return t0, pd.Timestamp(ev["end"])
    return t0, t0 + pd.Timedelta(hours=float(ev["duration_h"]))


def _benign_only_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of "truly benign" windows = NOT in any anomaly
    source (catalog ∪ REAL_ATTACK_CATALOG ∪ REAL_ATTACKS).  This is
    the F1 negative base under both the legacy protocol (outages
    excluded from F1 base) and the operator protocol (outages as
    positives) — the difference is in how outages are handled, not
    in what counts as "truly benign".
    """
    excluded = pd.Series(False, index=df.index)
    events = []
    events.extend(ATTACK_CATALOG)
    events.extend(CONFIG.get("EVAL", {}).get("REAL_ATTACK_CATALOG", []) or [])
    for evs in REAL_ATTACKS.values():
        events.extend(evs)
    for ev in events:
        t0, t1 = _event_bounds(ev)
        excluded |= (df["timestamp"] >= t0) & (df["timestamp"] < t1)
    return ~excluded


def _anomaly_window_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of windows that fall inside ANY anomaly the system
    is supposed to detect — synthetic catalogue, real DDoS, *and*
    real network outages.

    This implements the operator framing: a network outage is an
    anomaly that the detector should flag.  Counting it as a true
    positive (when detected) or false negative (when missed) is the
    operationally-faithful F1 protocol, in contrast to the legacy
    A3.2 fix which excluded outages from the F1 base entirely.

    The set is the union of three sources (deduplicated implicitly
    via the timestamp ∨-comparison):

      - ``ATTACK_CATALOG`` (synthetic, 13 events on RedeRio).
      - ``CONFIG['EVAL']['REAL_ATTACK_CATALOG']`` (operator-curated
        real attacks; 1 real DDoS Nov 12 on RedeRio).
      - ``REAL_ATTACKS['NETWORK_OUTAGE_*']`` (operator-curated real
        outages; 2 outages totalling 342 windows on RedeRio).

    ``REAL_ATTACKS['DDOS_ATTACK']`` is the **same event** as
    ``REAL_ATTACK_CATALOG['REAL_DDOS']`` (declared twice for legacy
    reasons); the timestamp union deduplicates it.
    """
    mask = pd.Series(False, index=df.index)
    for ev in ATTACK_CATALOG:
        t0, t1 = _event_bounds(ev)
        mask |= (df["timestamp"] >= t0) & (df["timestamp"] < t1)
    for ev in (CONFIG.get("EVAL", {}).get("REAL_ATTACK_CATALOG") or []):
        t0, t1 = _event_bounds(ev)
        mask |= (df["timestamp"] >= t0) & (df["timestamp"] < t1)
    for key, evs in (REAL_ATTACKS or {}).items():
        # Only NETWORK_OUTAGE_* contributes new positives here —
        # DDOS_ATTACK is already in REAL_ATTACK_CATALOG above.
        if not str(key).startswith("NETWORK_OUTAGE"):
            continue
        for ev in (evs or []):
            t0, t1 = _event_bounds(ev)
            mask |= (df["timestamp"] >= t0) & (df["timestamp"] < t1)
    return mask


# Legacy helpers kept for traceability and tests of the pre-2026-05-10
# methodology (outages excluded from the F1 base).  They are no longer
# called from this module.
def _attack_window_mask_legacy(df: pd.DataFrame) -> pd.Series:
    """Pre-2026-05-10 methodology: catalogue ∪ REAL_ATTACK_CATALOG only,
    outages excluded.  Kept for direct comparison with archived diagnostic
    runs that used this protocol.
    """
    mask = pd.Series(False, index=df.index)
    for ev in ATTACK_CATALOG:
        t0, t1 = _event_bounds(ev)
        mask |= (df["timestamp"] >= t0) & (df["timestamp"] < t1)
    for ev in (CONFIG.get("EVAL", {}).get("REAL_ATTACK_CATALOG") or []):
        t0, t1 = _event_bounds(ev)
        mask |= (df["timestamp"] >= t0) & (df["timestamp"] < t1)
    return mask


def _outage_only_mask(df: pd.DataFrame) -> pd.Series:
    """Pure NETWORK_OUTAGE_* mask (excluding DDOS_ATTACK which lives in
    REAL_ATTACK_CATALOG too).  Kept as a public helper so per-class
    diagnostics remain available even though outages are now folded
    into the anomaly mask.
    """
    mask = pd.Series(False, index=df.index)
    for key, evs in (REAL_ATTACKS or {}).items():
        if not str(key).startswith("NETWORK_OUTAGE"):
            continue
        for ev in (evs or []):
            t0, t1 = _event_bounds(ev)
            mask |= (df["timestamp"] >= t0) & (df["timestamp"] < t1)
    return mask


# ───────────────────────────────────────────── opinion reconstruction

def _resolve_leaf_keys(df: pd.DataFrame) -> list[str]:
    """Return all per-leaf metric keys that have a complete opinion in
    the CSV (b_safe, b_susp, b_atk, u, a_safe, a_susp, a_atk).
    """
    out = []
    for col in df.columns:
        if not col.endswith("_b_atk"):
            continue
        prefix = col[: -len("_b_atk")]
        if prefix.startswith("FINAL_SYSTEM"):
            continue
        if prefix.startswith("METHODE_"):
            continue
        if "_dir_pos_" in prefix or "_dir_neg_" in prefix:
            continue
        if all(f"{prefix}_{s}" in df.columns
                for s in ("b_safe", "b_susp", "u",
                          "a_safe", "a_susp", "a_atk")):
            out.append(prefix)
    return sorted(out)


def _row_opinion(row: pd.Series, key: str) -> MultinomialOpinion:
    """Build a MultinomialOpinion from per-leaf columns of one row."""
    b = [
        float(row[f"{key}_b_safe"]),
        float(row[f"{key}_b_susp"]),
        float(row[f"{key}_b_atk"]),
    ]
    u = float(row[f"{key}_u"])
    a = [
        float(row[f"{key}_a_safe"]),
        float(row[f"{key}_a_susp"]),
        float(row[f"{key}_a_atk"]),
    ]
    return MultinomialOpinion(b, u, a)


def _apply_attack_discount(op: MultinomialOpinion, alpha_atk: float) -> MultinomialOpinion:
    """Mercier-Quost-Denoeux 2008 contextual discount on b_atk only.

    α_attack ∈ [0, 1].  α=1 leaves the opinion unchanged, α=0 silences
    the attack mass entirely.

    Invariant: b'_safe + b'_susp + b'_atk + u' = 1.
    """
    if alpha_atk >= 1.0:
        return op
    b_atk_old = float(op.b[2])
    new_b = [float(op.b[0]), float(op.b[1]), alpha_atk * b_atk_old]
    new_u = float(op.u) + (1.0 - alpha_atk) * b_atk_old
    return MultinomialOpinion(new_b, new_u, op.a.copy())


# ─────────────────────────────────────────── per-window re-fusion

def fused_proj_atk_with_discount(
    df: pd.DataFrame,
    leaf_keys: list[str],
    volumetric_leaves: list[str],
    regimes: pd.Series,
    alpha_atk: float,
    W: float = 3.0,
) -> np.ndarray:
    """Re-fuse the per-window opinion stack with α_atk applied to
    volumetric leaves on ACTIVE windows; return the new
    ``proj_atk_FINAL`` array (length = len(df)).

    Group structure follows ``CONFIG['FUSION_METHOD_GROUPS']``:
    Prophet leaves are intra-group fused, Reconstruction leaves are
    intra-group fused, then the two group opinions are inter-method
    fused.  All fusions use uniform WBF (the published default).
    """
    prophet_leaves = [k for k in leaf_keys if k.startswith("prophet_")]
    reconst_leaves = [k for k in leaf_keys if k.startswith("reconst_")]
    volumetric_set = set(volumetric_leaves)

    n = len(df)
    proj_atk_out = np.full(n, np.nan, dtype=float)

    # Vectorised per-row loop. The inner SL operations are scalar; the
    # outer iteration is the bottleneck. ~12k rows × 8 alphas
    # ≈ 100k iterations — runs in under 1 minute on a developer machine.
    for i in range(n):
        row = df.iloc[i]
        is_active = regimes.iat[i] == "ACTIVE"

        prophet_ops = []
        for k in prophet_leaves:
            op = _row_opinion(row, k)
            if is_active and k in volumetric_set:
                op = _apply_attack_discount(op, alpha_atk)
            prophet_ops.append(op)

        reconst_ops = []
        for k in reconst_leaves:
            op = _row_opinion(row, k)
            if is_active and k in volumetric_set:
                op = _apply_attack_discount(op, alpha_atk)
            reconst_ops.append(op)

        prophet_fused = fusion_wbf_n_sources(prophet_ops, external_weights=None, W=W)
        if reconst_ops:
            reconst_fused = fusion_wbf_n_sources(reconst_ops, external_weights=None, W=W)
            inter = fusion_wbf_n_sources(
                [prophet_fused, reconst_fused], external_weights=None, W=W,
            )
        else:
            inter = prophet_fused

        # proj_atk = b_atk + a_atk · u  (Eq. 3.23)
        proj_atk_out[i] = float(inter.b[2] + inter.a[2] * inter.u)
    return proj_atk_out


# ─────────────────────────────────────────── metrics

def _per_attack_recall(df: pd.DataFrame, fused: np.ndarray, delta: float) -> dict:
    """Recall per anomaly event in the union (catalog ∪
    REAL_ATTACK_CATALOG ∪ NETWORK_OUTAGE_*).
    """
    out = {}
    # Synthetic catalogue
    for ev in ATTACK_CATALOG:
        t0, t1 = _event_bounds(ev)
        mask = (df["timestamp"] >= t0) & (df["timestamp"] < t1)
        n = int(mask.sum())
        if n == 0:
            continue
        tp = int(np.sum(fused[mask.to_numpy()] >= delta))
        out[ev["name"]] = {
            "source": "synthetic_catalog",
            "n_windows": n,
            "n_detected": tp,
            "recall": tp / n,
        }
    # Real DDoS (REAL_ATTACK_CATALOG)
    for ev in (CONFIG.get("EVAL", {}).get("REAL_ATTACK_CATALOG") or []):
        t0, t1 = _event_bounds(ev)
        mask = (df["timestamp"] >= t0) & (df["timestamp"] < t1)
        n = int(mask.sum())
        if n == 0:
            continue
        tp = int(np.sum(fused[mask.to_numpy()] >= delta))
        out[ev.get("name", "REAL_DDOS")] = {
            "source": "real_attack_catalog",
            "n_windows": n,
            "n_detected": tp,
            "recall": tp / n,
        }
    # Network outages (REAL_ATTACKS NETWORK_OUTAGE_*)
    for key, evs in (REAL_ATTACKS or {}).items():
        if not str(key).startswith("NETWORK_OUTAGE"):
            continue
        for i, ev in enumerate(evs or []):
            t0, t1 = _event_bounds(ev)
            mask = (df["timestamp"] >= t0) & (df["timestamp"] < t1)
            n = int(mask.sum())
            if n == 0:
                continue
            tp = int(np.sum(fused[mask.to_numpy()] >= delta))
            label = f"{key}#{i}" if len(evs) > 1 else key
            out[label] = {
                "source": "network_outage",
                "n_windows": n,
                "n_detected": tp,
                "recall": tp / n,
            }
    return out


def evaluate_alpha(
    df: pd.DataFrame,
    fused: np.ndarray,
    benign_mask: pd.Series,
    anomaly_mask: pd.Series,
    outage_mask: pd.Series,
    regimes: pd.Series,
    delta: float,
) -> dict:
    """Compute the headline statistics for one α value under the
    **operator-faithful** F1 protocol: a network outage is an anomaly
    that the detector should flag.

    F1 / MCC framing:
      - **Positives** = ``anomaly_mask`` (catalog ∪ REAL_ATTACK_CATALOG
        ∪ NETWORK_OUTAGE_*).  All anomalies the system is expected to
        detect.
      - **Negatives** = ``benign_mask`` (truly benign windows;
        anomalies of every kind excluded).
      - **base** = ``benign_mask | anomaly_mask`` — by construction
        every window is either positive or negative; nothing is
        excluded from the F1 base under this protocol.

    This differs from the legacy A3.2 protocol, which excluded
    NETWORK_OUTAGE_* from both numerator and denominator.  Reporting outages
    as positives usually lowers F1 when outages are missed, but it is the
    operator-faithful anomaly-detection view.  See
    ``docs/review/regime_fpr_root_cause_analysis.md`` §6.5 for the
    discussion.

    Per-regime FPR is unchanged across protocols (same benign base).
    """
    pred = fused >= delta

    # FPR per regime (benign-only base, matches evaluate_regime_fpr).
    benign = benign_mask.to_numpy()
    metrics = {"delta": delta}
    for bucket in ("ACTIVE", "QUIET", "ALL"):
        if bucket == "ALL":
            r_mask = benign
        else:
            r_mask = benign & (regimes == bucket).to_numpy()
        n = int(r_mask.sum())
        fp = int(np.sum(pred & r_mask))
        metrics[f"n_benign_{bucket.lower()}"] = n
        metrics[f"fp_{bucket.lower()}"] = fp
        metrics[f"fpr_{bucket.lower()}"] = (fp / n) if n else float("nan")

    # F1 / TPR / MCC under the operator-faithful protocol.
    base = benign | anomaly_mask.to_numpy()
    y_true = anomaly_mask.to_numpy()[base].astype(int)
    y_pred = pred[base].astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-12)
            if precision + recall > 0 else 0.0)
    denom_mcc = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom_mcc if denom_mcc > 0 else 0.0
    metrics.update({
        "protocol": "operator_faithful_v2_outages_as_positives",
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_outage_windows": int(outage_mask.to_numpy().sum()),
        "precision_window": precision,
        "recall_window": recall,
        "f1_window": f1,
        "mcc_window": mcc,
    })

    # Per-attack recall: count how many anomaly events keep ≥ 50 % coverage.
    per_attack = _per_attack_recall(df, fused, delta)
    metrics["n_attacks_with_recall_above_50"] = int(sum(
        1 for r in per_attack.values() if r["recall"] >= 0.5
    ))
    metrics["n_attacks_total"] = len(per_attack)
    metrics["per_attack"] = per_attack
    return metrics


# ──────────────────────────────────────────────────────── markdown

def _df_to_pipe_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(empty)_"
    cols = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                cells.append("NaN" if not math.isfinite(v) else f"{v:.6g}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    return "\n".join([header, sep, *rows])


# ──────────────────────────────────────────────────────── CLI

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=None,
                   help="Detection CSV (auto-discover if omitted).")
    p.add_argument("--out-dir", default="outputs/scientific_hardening")
    p.add_argument("--alphas", default="1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1",
                   help="Comma-separated α_attack grid.")
    p.add_argument("--volumetric-leaves", default=",".join(DEFAULT_VOLUMETRIC_LEAVES),
                   help="Comma-separated leaf metric keys to discount on ACTIVE.")
    p.add_argument("--W", type=float, default=3.0)
    args = p.parse_args(argv)

    csv_path = (Path(args.csv).resolve() if args.csv
                else Path("outputs/detection_results_INJECTED.csv").resolve())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[regime_fpr_alpha_sweep] reading {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    benign = _benign_only_mask(df)
    anomaly = _anomaly_window_mask(df)   # operator-faithful (outages = positives)
    outage = _outage_only_mask(df)         # diagnostic only
    regimes = regime_of_series(df["timestamp"], holidays=CONFIG.get("HOLIDAYS_LIST"))

    leaf_keys = _resolve_leaf_keys(df)
    volumetric = [k for k in args.volumetric_leaves.split(",") if k]
    missing = [k for k in volumetric if k not in leaf_keys]
    if missing:
        print(f"  WARNING: volumetric leaves not found in CSV: {missing}")
    volumetric = [k for k in volumetric if k in leaf_keys]

    delta = float(get_decision_threshold(CONFIG, up_levels=1))
    print(f"  benign={int(benign.sum())} (ACTIVE={int((benign & (regimes=='ACTIVE')).sum())},"
          f" QUIET={int((benign & (regimes=='QUIET')).sum())})")
    print(f"  anomaly (catalog + REAL_ATTACK_CATALOG + NETWORK_OUTAGE) windows={int(anomaly.sum())}")
    print(f"    of which pure NETWORK_OUTAGE = {int(outage.sum())}")
    print(f"  leaf metrics = {len(leaf_keys)}, volumetric subset = {len(volumetric)}")
    print(f"  volumetric leaves: {volumetric}")
    print(f"  δ = {delta:.6f}")
    print(f"  protocol: operator-faithful (outages = positives)")

    alphas = [float(x) for x in args.alphas.split(",")]
    rows = []
    per_attack_rows = []
    for alpha in alphas:
        print(f"  [α={alpha:.2f}] re-fusing {len(df)} windows ...", flush=True)
        fused = fused_proj_atk_with_discount(
            df, leaf_keys, volumetric, regimes, alpha, W=args.W,
        )
        m = evaluate_alpha(df, fused, benign, anomaly, outage, regimes, delta)
        m["alpha_attack"] = alpha
        per_attack = m.pop("per_attack")
        rows.append(m)
        for name, v in per_attack.items():
            per_attack_rows.append({
                "alpha_attack": alpha, "attack": name,
                "n_windows": v["n_windows"], "n_detected": v["n_detected"],
                "recall": v["recall"],
            })

    sweep_df = pd.DataFrame(rows)
    per_atk_df = pd.DataFrame(per_attack_rows)
    sweep_csv = out_dir / "regime_fpr_alpha_sweep.csv"
    per_atk_csv = out_dir / "regime_fpr_alpha_sweep_per_attack.csv"
    sweep_md = out_dir / "regime_fpr_alpha_sweep.md"
    sweep_json = out_dir / "regime_fpr_alpha_sweep.json"
    sweep_df.to_csv(sweep_csv, index=False)
    per_atk_df.to_csv(per_atk_csv, index=False)
    payload = {
        "csv_path": str(csv_path),
        "delta": delta,
        "volumetric_leaves": volumetric,
        "alphas": alphas,
        "summary": sweep_df.to_dict(orient="records"),
        "per_attack": per_atk_df.to_dict(orient="records"),
        "exploratory_pareto": True,
        "note": ("This sweep is run on the test span and is therefore "
                 "exploratory.  Any α used in production must be "
                 "re-calibrated on the train-calib hold-out."),
    }
    sweep_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    headline = sweep_df[[
        "alpha_attack", "fpr_active", "fpr_quiet", "fpr_all",
        "f1_window", "mcc_window", "recall_window",
        "n_attacks_with_recall_above_50",
    ]].rename(columns={
        "alpha_attack": "α",
        "fpr_active": "FPR_ACTIVE",
        "fpr_quiet":  "FPR_QUIET",
        "fpr_all":    "FPR_ALL",
        "f1_window":  "F1",
        "mcc_window": "MCC",
        "recall_window": "TPR",
        "n_attacks_with_recall_above_50": "N_attacks≥50%",
    })

    md = []
    md.append("# Regime-FPR α-sweep — exploratory Pareto curve\n")
    md.append("**Status: exploratory** — α calibrated on the test span; "
              "production use requires train-calib re-calibration.\n")
    md.append(f"- CSV: `{csv_path}`")
    md.append(f"- δ (system threshold) = {delta:.6f}")
    md.append(f"- Volumetric leaves discounted on ACTIVE: {volumetric}")
    md.append(f"- α grid: {alphas}\n")
    md.append("## Headline F1 / FPR Pareto curve")
    md.append("")
    md.append(_df_to_pipe_table(headline))
    md.append("")
    md.append("## Decision rule\n")
    md.append("- α=1.0 row is the published baseline.")
    md.append("- A candidate α<1 is *worth shipping* if FPR_ACTIVE drops "
              "toward the 0.001 target AND F1 stays within 0.01 of the "
              "α=1 baseline AND `N_attacks≥50%` does not drop.")
    md.append("- If no α meets these constraints, the per-regime "
              "contextual discount is rejected on RedeRio and we "
              "document the FPR overshoot as the operational optimum.")
    md.append("")
    md.append("## Per-attack recall sanity table (head)")
    md.append("")
    md.append(_df_to_pipe_table(per_atk_df.head(20)))
    sweep_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] wrote {sweep_csv}")
    print(f"[OK] wrote {per_atk_csv}")
    print(f"[OK] wrote {sweep_md}")
    print(f"[OK] wrote {sweep_json}")
    print()
    print("[Pareto] α | FPR_ACTIVE | FPR_QUIET | FPR_ALL | F1 | TPR | N_atk≥50%")
    for _, r in headline.iterrows():
        print(f"  α={r['α']:.2f}  FPR_A={r['FPR_ACTIVE']*100:.3f}%  "
              f"FPR_Q={r['FPR_QUIET']*100:.3f}%  FPR={r['FPR_ALL']*100:.3f}%  "
              f"F1={r['F1']:.4f}  TPR={r['TPR']:.4f}  "
              f"N≥50%={int(r['N_attacks≥50%'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
