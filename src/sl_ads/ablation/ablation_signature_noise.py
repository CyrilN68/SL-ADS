"""Noise robustness ablation on injected per-metric projected triplets.

This is a reviewer-facing stress test for A3.3: if injected signatures are
too clean, small perturbations of the normalized (safe, suspicious, attack)
triplets should not collapse the qualifier.

Three perturbation distributions are supported:

  * **gaussian** (default) — Normal(0, sigma). Light-tailed reference
    matching the original 2026-05-06 ablation.
  * **cauchy** — Cauchy(loc=0, scale=sigma). Heavy-tailed; no finite
    moments (mean/variance undefined). Stress test for catastrophic
    perturbations a real attack might briefly produce.
  * **student_t** — Student-t with ``df`` degrees of freedom, scaled by
    ``sigma``. Default ``df=3`` keeps a finite mean and variance while
    producing tails noticeably heavier than Gaussian (Hyndman et al.
    2003 *J. Econometrics* 113 §5; Lange et al. 1989 *JASA* 84).

After perturbation each triplet is clipped to non-negative values and
renormalised to sum to 1; pathological draws that zero every component
fall back to the original triplet.

References
----------
- Wu, R., Keogh, E. (2021). "Current TSAD Benchmarks are Flawed."
  *IEEE TKDE* 35(3) — flaw #3 (clean signatures).
- Cauchy, A. (1853). "Sur les résultats moyens d'observations de
  même nature." *Comptes Rendus* 37, 198–206.
- Student (Gosset, W. S.) (1908). "The probable error of a mean."
  *Biometrika* 6(1), 1–25.
- Lange, K. L., Little, R. J. A., Taylor, J. M. G. (1989). "Robust
  statistical modeling using the t distribution." *JASA* 84(408),
  881–896.
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
        Path(CONFIG.get("RESULTS_DIR", "")) / "detection_results_INJECTED.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No injected detection CSV found; pass --csv.")


def _known_attacks() -> list[dict]:
    return [
        a for a in INJECTED_ATTACK_CATALOG
        if a.get("expected") and not a.get("is_novelty_control", False)
    ]


def _triplet_prefixes(columns: list[str]) -> list[str]:
    cols = set(columns)
    prefixes = []
    for c in columns:
        for suffix in ("_proj_safe", "_dir_pos_proj_safe", "_dir_neg_proj_safe"):
            if c.endswith(suffix):
                prefix = c[:-len("_proj_safe")] if suffix == "_proj_safe" else c[:-len("_proj_safe")]
                if prefix.startswith("FINAL_SYSTEM_CBF"):
                    continue
                if (f"{prefix}_proj_susp" in cols and f"{prefix}_proj_atk" in cols):
                    prefixes.append(prefix)
    return sorted(set(prefixes))


def _draw_noise(distribution: str, sigma: float, df_t: float,
                rng: np.random.Generator) -> np.ndarray:
    """Draw a length-3 noise vector for a perturbation triplet.

    ``sigma`` is the scale parameter shared across distributions. For the
    Gaussian it equals the standard deviation; for Cauchy it is the half-
    width at half-maximum (the canonical scale parameter); for Student-t
    it scales the standardised variate. Comparing the same ``sigma`` value
    across distributions is therefore a comparison at matched scale, not
    at matched standard deviation (Cauchy has no variance).
    """
    if distribution == "gaussian":
        return rng.normal(0.0, sigma, size=3)
    if distribution == "cauchy":
        return rng.standard_cauchy(size=3) * sigma
    if distribution == "student_t":
        return rng.standard_t(df=df_t, size=3) * sigma
    raise ValueError(
        f"Unknown noise distribution: {distribution!r}. "
        f"Expected one of: gaussian, cauchy, student_t."
    )


def _perturb_row(row: pd.Series, prefixes: list[str], sigma: float,
                 rng: np.random.Generator,
                 distribution: str = "gaussian",
                 df_t: float = 3.0) -> pd.Series:
    if sigma <= 0:
        return row
    out = row.copy()
    for prefix in prefixes:
        cols = [f"{prefix}_proj_safe", f"{prefix}_proj_susp", f"{prefix}_proj_atk"]
        vals = out[cols].astype(float).to_numpy()
        if not np.all(np.isfinite(vals)):
            continue
        noise = _draw_noise(distribution, sigma, df_t, rng)
        noisy = np.clip(vals + noise, 0.0, None)
        total = noisy.sum()
        noisy = vals if total <= 1e-12 else noisy / total
        out.loc[cols] = noisy
    return out


def run_ablation(df: pd.DataFrame, sigmas: list[float], reps: int,
                 threshold: float, seed: int,
                 distributions: list[str] | None = None,
                 df_t: float = 3.0) -> pd.DataFrame:
    sbn_cond = CONFIG.get("SBN_COND_OPINIONS", {})
    prefixes = _triplet_prefixes(list(df.columns))
    distributions = distributions or ["gaussian"]
    rows = []
    attacks = _known_attacks()
    for distribution in distributions:
        # Distribution-specific seed offset keeps the streams independent
        # while remaining deterministic across reruns.
        dist_offset = abs(hash(distribution)) % (2 ** 31)
        for sigma in sigmas:
            for rep in range(reps):
                rng = np.random.default_rng(
                    seed + rep * 1000 + int(round(sigma * 1000)) + dist_offset
                )
                n_windows = n_detected = n_qualified = n_correct = n_autre = 0
                for atk in attacks:
                    t0 = pd.Timestamp(atk["start"])
                    if atk.get("end") is not None:
                        t1 = pd.Timestamp(atk["end"])
                    else:
                        t1 = t0 + pd.Timedelta(hours=float(atk["duration_h"]))
                    df_atk = df[(df["timestamp"] >= t0) & (df["timestamp"] < t1)]
                    expected = atk["expected"]
                    for _, row in df_atk.iterrows():
                        n_windows += 1
                        noisy_row = _perturb_row(
                            row, prefixes, sigma, rng,
                            distribution=distribution, df_t=df_t,
                        )
                        r = sbn_qualify_row(
                            noisy_row,
                            sbn_cond=sbn_cond,
                            threshold=threshold,
                            apply_temporal=False,
                            apply_um=True,
                            evidence_scale=float(CONFIG.get("SBN_EVIDENCE_SCALE", 3.0)),
                            autre_anomalie_prior=float(CONFIG.get("SBN_NOVELTY_U_RAW_THRESHOLD", 0.82)),
                        )
                        if r.get("gate_open"):
                            n_detected += 1
                            if r.get("qual_status") != "no_groups":
                                n_qualified += 1
                                if r.get("top1_type") == expected:
                                    n_correct += 1
                                if r.get("qual_status") == "autre_anomalie":
                                    n_autre += 1
                rows.append({
                    "noise_distribution": distribution,
                    "student_t_df": df_t if distribution == "student_t" else float("nan"),
                    "sigma": sigma,
                    "rep": rep,
                    "n_windows": n_windows,
                    "n_detected": n_detected,
                    "n_qualified": n_qualified,
                    "n_correct": n_correct,
                    "qp": n_correct / max(n_qualified, 1),
                    "dr": n_detected / max(n_windows, 1),
                    "autre_rate": n_autre / max(n_qualified, 1),
                    "n_triplet_prefixes": len(prefixes),
                })
    return pd.DataFrame(rows)


def plot_degradation(res: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    distributions = list(res["noise_distribution"].unique()) if "noise_distribution" in res.columns else ["gaussian"]
    colours = {"gaussian": "#1f77b4", "cauchy": "#d62728", "student_t": "#2ca02c"}
    markers = {"gaussian": "o", "cauchy": "s", "student_t": "^"}
    for dist in distributions:
        sub = res[res["noise_distribution"] == dist] if "noise_distribution" in res.columns else res
        g = sub.groupby("sigma").agg(
            qp_mean=("qp", "mean"), qp_std=("qp", "std"),
            dr_mean=("dr", "mean"), dr_std=("dr", "std"),
        ).reset_index()
        ax.errorbar(g["sigma"], g["qp_mean"], yerr=g["qp_std"].fillna(0),
                    marker=markers.get(dist, "o"), color=colours.get(dist),
                    label=f"QP — {dist}")
        ax.errorbar(g["sigma"], g["dr_mean"], yerr=g["dr_std"].fillna(0),
                    marker=markers.get(dist, "o"), linestyle="--",
                    color=colours.get(dist), alpha=0.6,
                    label=f"DR — {dist}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Noise scale sigma on normalized triplets")
    ax.set_ylabel("score")
    ax.set_title("Injected signature noise robustness "
                 "(QP solid, DR dashed)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=None)
    p.add_argument("--out-dir", default="outputs/scientific_hardening")
    p.add_argument("--sigmas", default="0,0.05,0.10,0.15,0.20")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--distributions",
        default="gaussian,cauchy,student_t",
        help="Comma-separated list of noise distributions. Choices: "
             "gaussian, cauchy, student_t. Defaults to running all three.",
    )
    p.add_argument(
        "--student-t-df",
        type=float,
        default=3.0,
        help="Degrees of freedom for the Student-t variant (default 3, "
             "heavy-tailed but with finite mean and variance).",
    )
    args = p.parse_args()

    csv_path = Path(args.csv) if args.csv else _default_detection_csv()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    threshold = get_decision_threshold(CONFIG, up_levels=1)
    sigmas = [float(x) for x in args.sigmas.split(",") if x]
    distributions = [d.strip() for d in args.distributions.split(",") if d.strip()]
    res = run_ablation(
        df, sigmas, args.reps, threshold, args.seed,
        distributions=distributions, df_t=args.student_t_df,
    )
    out_csv = out_dir / "signature_noise_ablation.csv"
    out_png = out_dir / "signature_noise_ablation.png"
    res.to_csv(out_csv, index=False)
    plot_degradation(res, out_png)
    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_png}")
    print(
        res.groupby(["noise_distribution", "sigma"])[
            ["qp", "dr", "autre_rate"]
        ].mean().to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
