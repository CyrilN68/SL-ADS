"""SBN evidence-scale / u_raw threshold sensitivity grid.

Reads an existing detection_results*.csv and re-runs the SBN qualifier on
attack windows for a grid of:
  - SBN evidence_scale
  - u_raw novelty threshold (passed as autre_anomalie_prior)

Outputs a CSV plus a QP heatmap so the paper can show whether the published
values sit on a broad plateau or on a fragile spike.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sl_ads.config import CONFIG, INJECTED_ATTACK_CATALOG
from sl_ads.paths import get_decision_threshold
from sl_ads.qualify.sbn_qualifier import sbn_qualify_row


def _default_detection_csv() -> Path:
    candidates = [
        Path("outputs/detection_results_INJECTED.csv"),
        Path("outputs/detection_results.csv"),
        Path(CONFIG.get("RESULTS_DIR", "")) / "detection_results_INJECTED.csv",
        Path(CONFIG.get("RESULTS_DIR", "")) / "detection_results.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No detection_results*.csv found; pass --csv.")


def _attack_bounds(atk: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    t0 = pd.Timestamp(atk["start"])
    if atk.get("end") is not None:
        return t0, pd.Timestamp(atk["end"])
    return t0, t0 + pd.Timedelta(hours=float(atk["duration_h"]))


def _known_attacks() -> list[dict]:
    return [
        a for a in INJECTED_ATTACK_CATALOG
        if a.get("expected") and not a.get("is_novelty_control", False)
    ]


def run_grid(
        df: pd.DataFrame,
        evidence_scales: list[float],
        u_thresholds: list[float],
        threshold: float,
) -> pd.DataFrame:
    rows = []
    sbn_cond = CONFIG.get("SBN_COND_OPINIONS", {})
    attacks = _known_attacks()

    for scale in evidence_scales:
        for u_thr in u_thresholds:
            # Aggregate over ALL known attacks for this (scale, u_thr) cell.
            n_total = n_detected = n_qualified_named = n_correct_named = 0
            n_autre = 0
            per_attack_qp = []
            for atk in attacks:
                t0, t1 = _attack_bounds(atk)
                df_atk = df[(df["timestamp"] >= t0) & (df["timestamp"] < t1)]
                if df_atk.empty:
                    continue
                a_total = len(df_atk)
                a_detected = a_qualified_named = a_correct = 0
                a_autre = 0
                expected = atk["expected"]
                for _, row in df_atk.iterrows():
                    r = sbn_qualify_row(
                        row,
                        sbn_cond=sbn_cond,
                        threshold=threshold,
                        apply_temporal=False,
                        apply_um=True,
                        evidence_scale=float(scale),
                        autre_anomalie_prior=float(u_thr),
                    )
                    if not r.get("gate_open"):
                        continue
                    a_detected += 1
                    qstatus = r.get("qual_status")
                    if qstatus == "no_groups":
                        continue
                    if qstatus == "autre_anomalie":
                        a_autre += 1
                        # Autre_anomalie counts as "abstained from typing", so it
                        # is NOT in the named-typing denominator.  This matches
                        # the operational semantics: QP measures "given that we
                        # output a named type, how often is it correct?".
                        continue
                    a_qualified_named += 1
                    if r.get("top1_type") == expected:
                        a_correct += 1
                n_total += a_total
                n_detected += a_detected
                n_qualified_named += a_qualified_named
                n_correct_named += a_correct
                n_autre += a_autre
                if a_qualified_named:
                    per_attack_qp.append(a_correct / a_qualified_named)

            dr = n_detected / max(n_total, 1)
            qp = n_correct_named / max(n_qualified_named, 1)
            # autre_rate denominator is "all detected windows" so it is
            # interpretable as a coverage statistic ("of the alarms, what
            # fraction did the qualifier abstain from?").
            autre_rate = n_autre / max(n_detected, 1)
            rows.append({
                "evidence_scale": scale,
                "u_raw_threshold": u_thr,
                "n_attack_windows": n_total,
                "n_detected": n_detected,
                "n_qualified_named": n_qualified_named,
                "n_correct_named": n_correct_named,
                "n_autre_anomalie": n_autre,
                "dr_micro": dr,
                "qp_micro": qp,
                "qp_macro": float(np.mean(per_attack_qp)) if per_attack_qp else np.nan,
                "autre_rate_over_detected": autre_rate,
                "named_typing_rate_over_detected": (n_qualified_named / max(n_detected, 1)),
            })
    return pd.DataFrame(rows)


def _annotate(ax, pivot, fmt: str, hot_cutoff: float) -> None:
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color="white" if v < hot_cutoff else "black", fontsize=7)


def plot_heatmap(res: pd.DataFrame, out_png: Path) -> None:
    """Three-panel sensitivity figure: QP, autre_rate, named_typing_rate."""
    panels = [
        ("qp_micro", "micro QP (correct named / qualified named)", "{:.2f}"),
        ("autre_rate_over_detected", "autre_anomalie rate / detected", "{:.2f}"),
        ("named_typing_rate_over_detected", "named typing rate / detected", "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    for ax, (col, title, fmt) in zip(axes, panels):
        pivot = res.pivot(index="u_raw_threshold", columns="evidence_scale", values=col)
        im = ax.imshow(pivot.values, origin="lower", aspect="auto",
                       vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x:g}" for x in pivot.columns], rotation=45, fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{y:g}" for y in pivot.index], fontsize=8)
        ax.set_xlabel("SBN evidence_scale")
        ax.set_ylabel("u_raw novelty threshold")
        ax.set_title(title, fontsize=10)
        _annotate(ax, pivot, fmt, hot_cutoff=0.55)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("SBN qualifier sensitivity to (evidence_scale, u_raw_threshold)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=None)
    p.add_argument("--out-dir", default="outputs/scientific_hardening")
    # Wide grid required to expose plateau behaviour around the published
    # operating point (evidence_scale=3.0, u_raw_threshold=0.82).  evidence_scale
    # below ~0.5 forces u_raw above the novelty threshold even on clean
    # synthetic injections, which is the structural failure mode the audit
    # asks us to demonstrate.
    p.add_argument("--scales", default="0.1,0.3,0.5,1.0,1.5,2.0,3.0,5.0,10.0")
    p.add_argument("--u-thresholds", default="0.10,0.30,0.50,0.70,0.82,0.90,0.95,0.99")
    args = p.parse_args()

    csv_path = Path(args.csv) if args.csv else _default_detection_csv()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    threshold = get_decision_threshold(CONFIG, up_levels=1)
    scales = [float(x) for x in args.scales.split(",") if x]
    u_thresholds = [float(x) for x in args.u_thresholds.split(",") if x]
    res = run_grid(df, scales, u_thresholds, threshold)
    out_csv = out_dir / "sbn_param_sensitivity.csv"
    out_png = out_dir / "sbn_param_sensitivity_qp_heatmap.png"
    res.to_csv(out_csv, index=False)
    plot_heatmap(res, out_png)
    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_png}")
    print(res.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
