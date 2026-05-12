"""
analysis_residual_correlation.py
================================

Empirical residual-correlation matrices for fusion-dependence auditing.

Why this exists
---------------
CBF is equivalent to adding evidence and is only defensible when the
fused sources can be treated as independent. WBF/averaging-style fusion is
the conservative default for dependent or partially redundant sources, but
strong residual correlations still matter: they show that the paper must
describe Prophet and Reconstruction as two related views of the same raw
traffic window, not as independent evidence streams.

Two sub-systems contribute opinions:

  Prophet metrics (12 streams):
    bytes, packets, flows, syn, icmp, udp, tcp, fin,
    entropy_src_ip, entropy_src_port, entropy_dst_port, avg_pkt_size

  Reconstruction metrics (5 streams):
    reconst_bytes_from_packets, bytes_from_entropy_src_port,
    udp_from_flows, fin_from_syn, tcp_from_packets

We compute three matrices for the *training-time* residuals (where there
is no attack contamination):

  * R_p     : 12 x 12 Prophet residual correlation
  * R_r     :  5 x  5 Reconstruction residual correlation
  * R_cross : 17 x 17 union (block-diagonal expectation; off-block reveals
              cross-method leakage and bounds the "double-count" concern)

For each matrix we report:

  * mean off-diagonal |rho|
  * max off-diagonal |rho| with the (i, j) pair achieving it
  * spectral radius of the off-diagonal part
  * effective sample size (Newey-West-style autocorr correction)
  * Bartlett's test for sphericity (H0: matrix = identity)
  * variance-inflation factors (VIF) per column
  * a dependence verdict for publication wording:
      LOW         : max |rho| < 0.30 -> weak-dependence wording is plausible
      MODERATE    : 0.30 <= max |rho| < 0.60 -> report dependence caveat
      HIGH        : >= 0.60 -> do not claim independence; keep CBF as
                                sensitivity/legacy only and prefer WBF or
                                hierarchical grouping

Inputs
------
The script accepts either:
  * --residuals-csv path/to/residuals.csv : raw residuals time series
    (columns named exactly as listed above), OR
  * --self-test : generate a synthetic correlation pattern and validate
                  the analysis pipeline.

Outputs
-------
  * results/residual_correlation_prophet.csv   (12 x 12 matrix)
  * results/residual_correlation_reconst.csv   (5 x 5)
  * results/residual_correlation_cross.csv     (17 x 17)
  * results/residual_correlation_summary.json  (verdicts, top pairs)

References
----------
Joesang, A. (2016). *Subjective Logic*. Springer. Belief fusion chapters.

Newey, W. K. & West, K. D. (1987). "A Simple, Positive Semi-definite,
Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
*Econometrica* 55 (3): 703-708.

Bartlett, M. S. (1950). "Tests of significance in factor analysis."
*British Journal of Psychology* 3 (2): 77-85.

Belsley, D., Kuh, E., & Welsch, R. (1980). *Regression Diagnostics:
Identifying Influential Data and Sources of Collinearity.* Wiley.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


PROPHET_METRICS: List[str] = [
    "prophet_bytes",
    "prophet_packets",
    "prophet_flows",
    "prophet_syn",
    "prophet_icmp",
    "prophet_udp",
    "prophet_tcp",
    "prophet_fin",
    "prophet_entropy_src_ip",
    "prophet_entropy_src_port",
    "prophet_entropy_dst_port",
    "prophet_avg_pkt_size",
]

RECONST_METRICS: List[str] = [
    "reconst_bytes_from_packets",
    "reconst_bytes_from_entropy_src_port",
    "reconst_udp_from_flows",
    "reconst_fin_from_syn",
    "reconst_tcp_from_packets",
]

ALL_METRICS = PROPHET_METRICS + RECONST_METRICS


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def newey_west_eff_n(x: np.ndarray, max_lag: int = 10) -> float:
    """Effective sample size accounting for residual autocorrelation."""
    n = len(x)
    if n < 2:
        return float(n)
    x = x - x.mean()
    var = x.var(ddof=1)
    if var < 1e-12:
        return float(n)
    rho_sum = 0.0
    for lag in range(1, min(max_lag + 1, n)):
        cov = (x[:-lag] * x[lag:]).mean()
        w = 1.0 - lag / (max_lag + 1)
        rho_sum += w * cov / var
    n_eff = n / (1.0 + 2.0 * rho_sum)
    return float(max(min(n_eff, n), 1.0))


def variance_inflation_factor(X: np.ndarray) -> np.ndarray:
    """VIF per column (Belsley, Kuh, Welsch 1980).

    VIF_j = 1 / (1 - R_j^2) where R_j^2 is the R-squared of regressing
    column j on all other columns. VIF > 10 indicates severe collinearity;
    VIF > 5 is the conventional warning threshold.
    """
    n, k = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    vifs = np.full(k, np.nan)
    for j in range(k):
        y = Xc[:, j]
        Xrest = np.delete(Xc, j, axis=1)
        # Solve via least-squares.
        beta, _, _, _ = np.linalg.lstsq(Xrest, y, rcond=None)
        y_hat = Xrest @ beta
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = (y ** 2).sum()
        if ss_tot < 1e-12:
            vifs[j] = np.nan
        else:
            r2 = 1.0 - ss_res / ss_tot
            vifs[j] = 1.0 / max(1.0 - r2, 1e-9)
    return vifs


def bartlett_sphericity(corr: np.ndarray, n: int) -> Tuple[float, float]:
    """Bartlett 1950 test: H0 corr = I.

    chi^2 = -(n - 1 - (2k+5)/6) * log|corr|
    df = k(k-1)/2
    """
    k = corr.shape[0]
    sign, logdet = np.linalg.slogdet(corr)
    if sign <= 0:
        # Singular or non-PSD - return absurd p-value to flag.
        return float("inf"), 0.0
    stat = -(n - 1 - (2 * k + 5) / 6) * logdet
    df = k * (k - 1) / 2
    p = float(stats.chi2.sf(stat, df))
    return float(stat), p


def summarise_corr(name: str, R: np.ndarray, n_eff: int, columns: List[str]) -> Dict[str, object]:
    """Return summary stats for a correlation matrix."""
    k = R.shape[0]
    off_mask = ~np.eye(k, dtype=bool)
    abs_off = np.abs(R[off_mask])
    max_idx = np.unravel_index(np.argmax(np.abs(R) * off_mask), R.shape)
    eig = np.linalg.eigvalsh(R - np.eye(k))
    bart_stat, bart_p = bartlett_sphericity(R, n_eff)
    if abs_off.max() < 0.30:
        verdict = "LOW"
    elif abs_off.max() < 0.60:
        verdict = "MODERATE"
    else:
        verdict = "HIGH"
    return {
        "name": name,
        "k": int(k),
        "n_eff": int(n_eff),
        "mean_abs_off": float(abs_off.mean()),
        "max_abs_off": float(abs_off.max()),
        "max_pair": [columns[max_idx[0]], columns[max_idx[1]]],
        "spectral_radius_off": float(np.abs(eig).max()),
        "bartlett_chi2": bart_stat,
        "bartlett_p": bart_p,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Synthetic generator (validation only)
# ---------------------------------------------------------------------------

def make_synthetic_residuals(
    n: int = 5000,
    seed: int = 0,
    cross_leakage: float = 0.4,
) -> pd.DataFrame:
    """Generate residuals with a built-in correlation structure that we can
    rediscover. Useful for self-test."""
    rng = np.random.default_rng(seed)
    # Latent volume signal drives bytes/packets/flows AND
    # bytes_from_packets/tcp_from_packets (cross-leakage).
    z_volume = rng.normal(size=n)
    z_proto  = rng.normal(size=n)   # syn/tcp/udp share factor
    z_entropy = rng.normal(size=n)  # entropy share factor

    cols = {}
    cols["prophet_bytes"]   = z_volume + 0.4 * rng.normal(size=n)
    cols["prophet_packets"] = z_volume + 0.4 * rng.normal(size=n)
    cols["prophet_flows"]   = z_volume + 0.6 * rng.normal(size=n)
    cols["prophet_syn"]     = z_proto + 0.6 * rng.normal(size=n)
    cols["prophet_icmp"]    = rng.normal(size=n)
    cols["prophet_udp"]     = 0.5 * z_proto + 0.7 * rng.normal(size=n)
    cols["prophet_tcp"]     = z_proto + 0.5 * rng.normal(size=n)
    cols["prophet_fin"]     = 0.7 * z_proto + 0.6 * rng.normal(size=n)
    cols["prophet_entropy_src_ip"]   = z_entropy + 0.7 * rng.normal(size=n)
    cols["prophet_entropy_src_port"] = z_entropy + 0.7 * rng.normal(size=n)
    cols["prophet_entropy_dst_port"] = 0.5 * z_entropy + 0.8 * rng.normal(size=n)
    cols["prophet_avg_pkt_size"]     = 0.3 * z_volume + 0.9 * rng.normal(size=n)

    # Reconstruction residuals - structurally tied to their parents.
    cols["reconst_bytes_from_packets"]      = cross_leakage * cols["prophet_bytes"] + 0.7 * rng.normal(size=n)
    cols["reconst_bytes_from_entropy_src_port"] = cross_leakage * cols["prophet_bytes"] + 0.8 * rng.normal(size=n)
    cols["reconst_udp_from_flows"]          = cross_leakage * cols["prophet_udp"] + 0.7 * rng.normal(size=n)
    cols["reconst_fin_from_syn"]            = cross_leakage * cols["prophet_fin"] + 0.7 * rng.normal(size=n)
    cols["reconst_tcp_from_packets"]        = cross_leakage * cols["prophet_tcp"] + 0.7 * rng.normal(size=n)

    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(
    df: pd.DataFrame,
    out_dir: str = "results",
) -> Dict[str, object]:
    """Compute the three matrices, write CSV artefacts, return summary dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    available = [c for c in ALL_METRICS if c in df.columns]
    prophet_cols = [c for c in PROPHET_METRICS if c in df.columns]
    reconst_cols = [c for c in RECONST_METRICS if c in df.columns]

    if len(available) < 2:
        raise ValueError(
            f"Need at least 2 metrics; got {available}. Expected columns: {ALL_METRICS}"
        )

    # Effective sample size (worst-case across all columns).
    n_effs = [newey_west_eff_n(df[c].values) for c in available]
    n_eff = int(min(n_effs))

    # Block matrices.
    summaries = []
    if len(prophet_cols) >= 2:
        Rp = df[prophet_cols].corr().values
        pd.DataFrame(Rp, index=prophet_cols, columns=prophet_cols).to_csv(
            out / "residual_correlation_prophet.csv"
        )
        summaries.append(summarise_corr("prophet", Rp, n_eff, prophet_cols))

    if len(reconst_cols) >= 2:
        Rr = df[reconst_cols].corr().values
        pd.DataFrame(Rr, index=reconst_cols, columns=reconst_cols).to_csv(
            out / "residual_correlation_reconst.csv"
        )
        summaries.append(summarise_corr("reconst", Rr, n_eff, reconst_cols))

    Rc = df[available].corr().values
    pd.DataFrame(Rc, index=available, columns=available).to_csv(
        out / "residual_correlation_cross.csv"
    )
    summaries.append(summarise_corr("cross", Rc, n_eff, available))

    # Variance inflation factors on the *cross* matrix.
    vifs = variance_inflation_factor(df[available].values)
    vif_table = pd.DataFrame({"metric": available, "VIF": vifs})
    vif_table.to_csv(out / "residual_vif.csv", index=False)

    # Top correlated pairs (off-diagonal, |rho| >= 0.30).
    top_pairs = []
    for i in range(Rc.shape[0]):
        for j in range(i + 1, Rc.shape[1]):
            r = Rc[i, j]
            if abs(r) >= 0.30:
                top_pairs.append({
                    "i": available[i],
                    "j": available[j],
                    "rho": float(r),
                    "abs_rho": float(abs(r)),
                })
    top_pairs.sort(key=lambda d: -d["abs_rho"])

    summary = {
        "n_samples": int(len(df)),
        "n_eff": n_eff,
        "matrices": summaries,
        "vif_warnings": [
            {"metric": m, "VIF": float(v)}
            for m, v in zip(available, vifs)
            if not np.isnan(v) and v >= 5.0
        ],
        "top_correlated_pairs": top_pairs[:20],
    }
    with open(out / "residual_correlation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Console pretty print.
    print(f"\n=== Residual correlation analysis (n={len(df)}, n_eff={n_eff}) ===")
    for s in summaries:
        print(f"\n[{s['name']}]")
        print(f"  k = {s['k']}, mean|rho_off| = {s['mean_abs_off']:.3f}, "
              f"max|rho_off| = {s['max_abs_off']:.3f}")
        print(f"  worst pair: {s['max_pair'][0]} <-> {s['max_pair'][1]}")
        print(f"  Bartlett chi^2 = {s['bartlett_chi2']:.1f}  p = {s['bartlett_p']:.3g}")
        print(f"  verdict: {s['verdict']}")
    print(f"\n[VIF warnings] (>= 5):")
    for w in summary["vif_warnings"]:
        print(f"  {w['metric']:40s} VIF = {w['VIF']:.2f}")
    print(f"\n[OK] artefacts written to {out}/")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _self_test() -> int:
    print("[TEST] analysis_residual_correlation.py self-test")
    df = make_synthetic_residuals(n=3000, seed=0, cross_leakage=0.4)
    summary = analyse(df, out_dir="results/_residcorr_selftest")
    # We injected non-trivial correlation -> at least the cross matrix
    # must have verdict != LOW.
    cross = next(s for s in summary["matrices"] if s["name"] == "cross")
    assert cross["verdict"] in {"MODERATE", "HIGH"}, cross
    print(f"\n   [OK] Synthetic cross verdict = {cross['verdict']} (expected MODERATE/HIGH)")
    print(f"   [OK] Top pair found = {cross['max_pair']} rho = {cross['max_abs_off']:.3f}")
    print("[TEST] ALL PASS")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--residuals-csv", type=str, default=None,
        help="CSV with one column per metric (see PROPHET_METRICS / RECONST_METRICS).",
    )
    p.add_argument("--out-dir", default="results")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        return _self_test()
    if args.residuals_csv is None:
        print("ERROR: pass --residuals-csv or --self-test")
        return 2
    df = pd.read_csv(args.residuals_csv)
    analyse(df, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
