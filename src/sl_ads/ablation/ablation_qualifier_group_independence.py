"""ablation_qualifier_group_independence.py — Audit A6.1 (Naive-Bayes assumption)
=================================================================================

The SBN qualifier sums per-group log-scores under the Naive-Bayes
assumption that group-level evidence vectors are conditionally
independent given an attack type k:

    P(g_1, g_2, ..., g_G | k) ≈ ∏_g P(g | k)
    ⇒  e(k) = Σ_g e_g(k)

A6.1 in ASSUMPTIONS.md acknowledges this is a working assumption and
flags two known violations: ``volume`` ↔ ``protocol_tcp/udp`` for
volumetric floods and an earlier ``tcp_flags`` ↔ ``reconstruction`` link
that was patched by removing ``reconst_fin_from_syn`` from the
``reconstruction`` group.

This script measures *empirically* the dependence between group
projected-evidence vectors on RedeRio attack windows and reports a
Domingos-Pazzani-style robustness bound:

  - Per group g, build the 3-vector P_g = (P_safe, P_susp, P_atk) per
    window using the same geometric-mean pooling as the qualifier.
  - For each pair (g_1, g_2), compute the Pearson correlation matrix
    between P_{g_1} and P_{g_2} on (i) attack windows, (ii) normal
    windows, and (iii) a pooled view.
  - Report max |rho| per pair and the Domingos-Pazzani robustness band
    (NB optimal under zero-one loss when max correlated dependence is
    "moderate" — see the original 1997 paper).

Outputs:
  - outputs/scientific_hardening/qualifier_group_correlations.csv
  - outputs/scientific_hardening/qualifier_group_correlations_summary.json

References
----------
Domingos & Pazzani (1997) Mach. Learn. 29(2-3): 103-130.
Rish (2001) Workshop on Empirical Methods in AI (IJCAI).
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sl_ads.config import CONFIG, INJECTED_ATTACK_CATALOG, QUALIFY_GROUP_SOURCES


def _default_detection_csv() -> Path:
    candidates = [
        Path("outputs/detection_results_INJECTED.csv"),
        Path(CONFIG.get("RESULTS_DIR", "")) / "detection_results_INJECTED.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No detection_results_INJECTED.csv found.")


def _attack_mask(df: pd.DataFrame) -> np.ndarray:
    """Return a boolean mask for windows inside any catalogued attack."""
    mask = np.zeros(len(df), dtype=bool)
    for atk in INJECTED_ATTACK_CATALOG:
        if not atk.get("expected"):
            continue
        t0 = pd.Timestamp(atk["start"])
        if atk.get("end") is not None:
            t1 = pd.Timestamp(atk["end"])
        else:
            t1 = t0 + pd.Timedelta(hours=float(atk["duration_h"]))
        mask |= ((df["timestamp"] >= t0) & (df["timestamp"] < t1)).to_numpy()
    return mask


def _group_proj(df: pd.DataFrame, group_metrics: list[str]) -> np.ndarray:
    """Per-window 3-vector P_g via geometric mean across group metrics.

    Mirrors the production logic in ``qualify.sbn_qualifier._compute_group_projected``:
    geomean of P_safe / P_susp / P_atk across metrics in the group, then
    renormalise to a simplex.  Missing columns are silently skipped.
    """
    states = ("proj_safe", "proj_susp", "proj_atk")
    log_sums = np.zeros((len(df), 3), dtype=float)
    n_present = 0
    for metric in group_metrics:
        cols = [f"{metric}_{s}" for s in states]
        if not all(c in df.columns for c in cols):
            continue
        vals = df[cols].to_numpy(dtype=float)
        # Numerical floor — log(0) = -inf would propagate to all states.
        vals = np.clip(vals, 1e-12, 1.0)
        log_sums += np.log(vals)
        n_present += 1
    if n_present == 0:
        return np.full((len(df), 3), np.nan)
    geomean = np.exp(log_sums / n_present)
    s = geomean.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return geomean / s


def _max_abs_corr_pair(P_a: np.ndarray, P_b: np.ndarray) -> tuple[float, str]:
    """Max |Pearson correlation| over the 3x3 cross between two state vectors."""
    if np.any(np.isnan(P_a)) or np.any(np.isnan(P_b)):
        return float("nan"), "nan"
    states = ("safe", "susp", "atk")
    best, best_pair = 0.0, ""
    for i, si in enumerate(states):
        for j, sj in enumerate(states):
            a = P_a[:, i]
            b = P_b[:, j]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            if abs(r) > abs(best):
                best, best_pair = r, f"{si}-{sj}"
    return best, best_pair


def _domingos_band(rho_max: float) -> str:
    """Domingos-Pazzani robustness band — NB tolerates moderate dependence.

    The 1997 paper proves NB optimal under zero-one loss whenever the
    pairwise dependence is bounded.  Following Rish (2001), we use the
    standard practical bands:

        |rho| < 0.30 -> NB optimality plausible
        0.30 <= |rho| < 0.60 -> NB acceptable, report dependence caveat
        |rho| >= 0.60 -> NB vulnerable, restructure groups or report
    """
    if not np.isfinite(rho_max):
        return "INSUFFICIENT_DATA"
    a = abs(rho_max)
    if a < 0.30:
        return "LOW"
    if a < 0.60:
        return "MODERATE"
    return "HIGH"


def analyse(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    attack_mask = _attack_mask(df)
    group_proj = {
        g: _group_proj(df, metrics) for g, metrics in QUALIFY_GROUP_SOURCES.items()
    }
    rows = []
    pairs = list(combinations(group_proj.keys(), 2))
    for g1, g2 in pairs:
        P_a = group_proj[g1]
        P_b = group_proj[g2]
        if np.all(np.isnan(P_a)) or np.all(np.isnan(P_b)):
            continue
        rho_atk, pair_atk = _max_abs_corr_pair(P_a[attack_mask], P_b[attack_mask])
        rho_norm, pair_norm = _max_abs_corr_pair(
            P_a[~attack_mask], P_b[~attack_mask]
        )
        rho_pool, pair_pool = _max_abs_corr_pair(P_a, P_b)
        rows.append({
            "group_a": g1,
            "group_b": g2,
            "rho_attack": rho_atk,
            "rho_attack_state_pair": pair_atk,
            "rho_normal": rho_norm,
            "rho_normal_state_pair": pair_norm,
            "rho_pooled": rho_pool,
            "verdict_attack": _domingos_band(rho_atk),
            "verdict_normal": _domingos_band(rho_norm),
        })
    res = pd.DataFrame(rows).sort_values(
        "rho_attack", key=lambda s: s.abs(), ascending=False
    )
    summary = {
        "n_attack_windows": int(attack_mask.sum()),
        "n_normal_windows": int((~attack_mask).sum()),
        "n_groups_audited": len(group_proj),
        "n_pairs_tested": len(rows),
        "n_pairs_high_dependence_attack": int(
            (res["verdict_attack"] == "HIGH").sum()
        ),
        "n_pairs_moderate_dependence_attack": int(
            (res["verdict_attack"] == "MODERATE").sum()
        ),
        "max_abs_rho_attack": float(res["rho_attack"].abs().max())
        if not res.empty else None,
        "domingos_pazzani_verdict": (
            "NB_OPTIMALITY_PLAUSIBLE"
            if res["rho_attack"].abs().max() < 0.30
            else "NB_ACCEPTABLE_WITH_CAVEAT"
            if res["rho_attack"].abs().max() < 0.60
            else "NB_VIOLATED_RECONSIDER_GROUPS"
        ) if not res.empty else "INSUFFICIENT_DATA",
        "top_dependent_pairs": res.head(10).to_dict(orient="records"),
    }
    return res, summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=None)
    p.add_argument("--out-dir", default="outputs/scientific_hardening")
    args = p.parse_args()

    csv_path = Path(args.csv) if args.csv else _default_detection_csv()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    res, summary = analyse(df)
    out_csv = out_dir / "qualifier_group_correlations.csv"
    out_json = out_dir / "qualifier_group_correlations_summary.json"
    res.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_json}")
    print(json.dumps(summary, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
