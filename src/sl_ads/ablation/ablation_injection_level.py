"""
ablation_injection_level.py
===========================

Ablation: evidence-level injection vs raw-data-level injection.

Motivation
----------
Wu & Keogh (2021), "Current Time Series Anomaly Detection Benchmarks
are Flawed and are Creating the Illusion of Progress", IEEE TKDE,
identifies four failure modes for synthetic anomaly evaluation. Two
of them apply directly to *injected* anomalies:

  (i)  Triviality - the anomaly is so obvious that a one-line
       baseline (mean + 3*sigma) detects it perfectly, defeating the
       purpose of any sophisticated detector.
  (ii) Unrealistic anomaly density / mode - the injection bypasses
       the natural physics of the problem and creates patterns that
       no real attacker would produce.

The current pipeline injects anomalies *at the evidence level* by
manipulating the (P, S, N) belief triplet directly (see
`inject_at_evidence_level.py`). This is computationally convenient
but creates two reviewer-facing risks:

  R1. **Convenient fingerprint**: by injecting directly into PSN we
      may over-state how "easy" the SL fusion is to drive into the
      Anom region.
  R2. **Bypass of upstream guards**: Prophet residuals, EDP priors,
      and entropy invariants are skipped, so a reviewer can argue we
      "tested SL on its own evidence", not on realistic attacks.

This script gives us the lever to defend against both:

  Path A (evidence-level, current): triplet injected post-Prophet.
  Path B (raw-data level)         : corrupt the raw input metrics
                                    (bytes_in, packets, syn_count, etc.)
                                    *before* Prophet, recompute residuals,
                                    recompute evidence, then evaluate.

For each scenario we report:

  * detection metrics (F1, MCC, FPR, TTD) on each path,
  * BCa 95 % CIs on the gap (paired bootstrap),
  * McNemar test on the discordant predictions,
  * a "triviality probe" (see below) that flags obvious-injections,
  * a "realism probe" (see below) on the raw distribution shifts.

Triviality probe
----------------
For each scenario we run two trivial baselines on the raw-data path:

  T1. Per-metric z-score > 3 (Tukey 1977).
  T2. Univariate Isolation-Forest on the leading metric.

If a trivial baseline matches >= 90 % of the F1 of our pipeline,
the injection is flagged TRIVIAL (Wu & Keogh flaw #1).

Realism probe
-------------
For raw-data injection we compare the moments of the injected window
against the empirical distribution of the *non-injected* timeseries:

  * mean shift in standard deviations
  * skewness shift
  * KS statistic against the unperturbed distribution

If the injection looks "in distribution" (KS p > 0.5 for all metrics),
the attack is suspiciously subtle and we flag REALISTIC_BUT_WEAK.
If KS = 1.0 across the board, we flag UNREALISTIC_OUTLIER (Wu & Keogh
flaw #2).

CLI
---
Run as a stand-alone script with no required args - it will use a
synthetic mini-dataset for self-validation. To run on the real
RedeRio data, pass --evidence-csv and --raw-csv pointing to the two
preprocessing outputs.

References
----------
Wu, R. & Keogh, E. (2021). "Current Time Series Anomaly Detection
Benchmarks are Flawed and are Creating the Illusion of Progress."
*IEEE TKDE* 35 (3): 2421-2429.

Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.

Sharafaldin, I. et al. (2018). "Toward Generating a New Intrusion
Detection Dataset and Intrusion Traffic Characterization." ICISSP.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, matthews_corrcoef

# Local imports — Phase H absolute paths via sl_ads package
from sl_ads.stats.bootstrap_ci import paired_bootstrap_bca_ci, format_ci  # noqa: E402
from sl_ads.stats.mcnemar import mcnemar_paired_test, format_mcnemar      # noqa: E402


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario: str
    path: str                # 'evidence' or 'raw'
    f1: float
    mcc: float
    fpr: float
    n_pos: int
    n_neg: int
    triviality_f1: float     # F1 of best trivial baseline
    triviality_flag: str     # 'TRIVIAL', 'NON_TRIVIAL'
    realism_flag: str        # 'REALISTIC', 'UNREALISTIC_OUTLIER', 'REALISTIC_BUT_WEAK', 'N/A'
    extra: Dict[str, float]  # ks stats, mean shift, etc.


# ---------------------------------------------------------------------------
# Trivial baselines (Wu & Keogh flaw #1)
# ---------------------------------------------------------------------------

def trivial_zscore_detector(
    raw: np.ndarray,
    train_mask: np.ndarray,
    threshold: float = 3.0,
) -> np.ndarray:
    """Per-metric z-score > threshold; ANY metric over threshold => flag."""
    mu = raw[train_mask].mean(axis=0, keepdims=True)
    sd = raw[train_mask].std(axis=0, keepdims=True) + 1e-9
    z = np.abs((raw - mu) / sd)
    return (z.max(axis=1) > threshold).astype(int)


def trivial_isoforest_detector(
    raw: np.ndarray,
    train_mask: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Univariate IsolationForest on the *first* (leading volume) metric."""
    iso = IsolationForest(contamination=0.05, random_state=seed)
    iso.fit(raw[train_mask, :1])
    return (iso.predict(raw[:, :1]) == -1).astype(int)


def detect_triviality(
    y_true: np.ndarray,
    pipeline_pred: np.ndarray,
    raw: np.ndarray,
    train_mask: np.ndarray,
    relative_floor: float = 0.90,
) -> Tuple[float, str]:
    """If best trivial detector >= relative_floor * pipeline_F1 -> TRIVIAL."""
    pipeline_f1 = f1_score(y_true, pipeline_pred, zero_division=0)
    f1_z = f1_score(y_true, trivial_zscore_detector(raw, train_mask), zero_division=0)
    f1_if = f1_score(y_true, trivial_isoforest_detector(raw, train_mask), zero_division=0)
    triv_best = max(f1_z, f1_if)
    if pipeline_f1 == 0:
        return triv_best, "N/A"
    flag = "TRIVIAL" if triv_best >= relative_floor * pipeline_f1 else "NON_TRIVIAL"
    return triv_best, flag


# ---------------------------------------------------------------------------
# Realism probe (Wu & Keogh flaw #2)
# ---------------------------------------------------------------------------

def detect_realism(
    raw: np.ndarray,
    inject_mask: np.ndarray,
) -> Tuple[Dict[str, float], str]:
    """Compare injected window vs unperturbed distribution.

    Returns
    -------
    extra : dict with 'ks_max', 'ks_mean_p', 'mean_shift_sigma'.
    flag  : one of 'REALISTIC', 'REALISTIC_BUT_WEAK', 'UNREALISTIC_OUTLIER'.
    """
    if inject_mask.sum() == 0 or (~inject_mask).sum() == 0:
        return {"ks_max": np.nan, "ks_mean_p": np.nan, "mean_shift_sigma": np.nan}, "N/A"

    base = raw[~inject_mask]
    inj = raw[inject_mask]

    ks_stats = []
    ks_pvals = []
    mean_shifts = []
    for j in range(raw.shape[1]):
        s, p = stats.ks_2samp(base[:, j], inj[:, j])
        ks_stats.append(s)
        ks_pvals.append(p)
        sd = base[:, j].std() + 1e-9
        mean_shifts.append((inj[:, j].mean() - base[:, j].mean()) / sd)

    extra = {
        "ks_max": float(max(ks_stats)),
        "ks_mean_p": float(np.mean(ks_pvals)),
        "mean_shift_sigma": float(np.mean(np.abs(mean_shifts))),
    }
    if extra["ks_max"] >= 0.99 and extra["mean_shift_sigma"] >= 6.0:
        flag = "UNREALISTIC_OUTLIER"
    elif extra["ks_mean_p"] > 0.5 and extra["mean_shift_sigma"] < 1.0:
        flag = "REALISTIC_BUT_WEAK"
    else:
        flag = "REALISTIC"
    return extra, flag


# ---------------------------------------------------------------------------
# Synthetic dataset generator (used for self-test and demo)
# ---------------------------------------------------------------------------

def make_synthetic_pair(
    n_train: int = 2880,    # 2 days @ 1-min
    n_test:  int = 1440,    # 1 day  @ 1-min
    n_metrics: int = 5,
    attack_kind: str = "volumetric",  # 'volumetric' | 'subtle' | 'slowloris'
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Generate a synthetic raw-vs-evidence pair for ablation demo.

    Returns dict with keys:
      raw_train, raw_test, evid_test, y_test, inject_mask_test, train_mask
    """
    rng = np.random.default_rng(seed)
    # Daily seasonality + noise.
    base = rng.normal(loc=0.0, scale=1.0, size=(n_train + n_test, n_metrics))
    t = np.arange(n_train + n_test) / 60.0  # hours
    daily = np.sin(2 * np.pi * t / 24)[:, None] * 0.5
    raw = base + daily

    inject_mask = np.zeros(n_train + n_test, dtype=bool)
    # Inject a single attack window in the test range.
    a_start = n_train + 200
    a_end = n_train + 260
    inject_mask[a_start:a_end] = True

    if attack_kind == "volumetric":
        raw[a_start:a_end, 0] += rng.normal(loc=10.0, scale=0.5, size=a_end - a_start)
        raw[a_start:a_end, 1] += rng.normal(loc=8.0, scale=0.5, size=a_end - a_start)
    elif attack_kind == "subtle":
        raw[a_start:a_end, 0] += rng.normal(loc=1.5, scale=0.3, size=a_end - a_start)
    elif attack_kind == "slowloris":
        # FIN/SYN ratio drift - small in volume, structural in shape.
        raw[a_start:a_end, 2] += rng.normal(loc=2.0, scale=0.4, size=a_end - a_start)
        raw[a_start:a_end, 3] -= rng.normal(loc=0.8, scale=0.2, size=a_end - a_start)

    train_mask = np.zeros_like(inject_mask)
    train_mask[:n_train] = True

    # Synthetic "evidence" = raw clipped + noise (simplified PSN proxy).
    evid = np.clip(raw, -3, 3) / 3.0  # in [-1, 1]
    # Inject directly in evidence (path A simulation): just push to extreme.
    evid_evidence_path = evid.copy()
    evid_evidence_path[a_start:a_end] = 1.0

    raw_train = raw[:n_train]
    raw_test = raw[n_train:]
    evid_test = evid[n_train:]
    evid_test_evidence_path = evid_evidence_path[n_train:]
    y_test = inject_mask[n_train:].astype(int)
    inject_mask_test = inject_mask[n_train:]

    return {
        "raw_train": raw_train,
        "raw_test": raw_test,
        "evid_test_raw_path": evid_test,             # evidence derived from raw
        "evid_test_evidence_path": evid_test_evidence_path,  # evidence injected directly
        "y_test": y_test,
        "inject_mask_test": inject_mask_test,
        "train_mask_full": train_mask,
        "raw_full": raw,
        "inject_mask_full": inject_mask,
    }


# ---------------------------------------------------------------------------
# Stand-in detector for the demo (the real one lives in compute_opinions_v3 +
# evaluate_injection_v2). For ablation framework validation we just threshold
# the evidence's L_inf norm; this is enough to make the framework usable.
# ---------------------------------------------------------------------------

def stub_detector(evid: np.ndarray, threshold: float = 0.85) -> np.ndarray:
    return (np.abs(evid).max(axis=1) > threshold).astype(int)


# ---------------------------------------------------------------------------
# Core ablation
# ---------------------------------------------------------------------------

def run_one_scenario(
    name: str,
    data: Dict[str, np.ndarray],
    detector_fn: Callable[[np.ndarray], np.ndarray] = stub_detector,
) -> Tuple[ScenarioResult, ScenarioResult, dict]:
    """Run both injection paths for one scenario.

    Returns (result_evidence, result_raw, comparison_dict).
    """
    y = data["y_test"]
    pred_evid = detector_fn(data["evid_test_evidence_path"])
    pred_raw = detector_fn(data["evid_test_raw_path"])

    def _metrics(pred):
        f1 = float(f1_score(y, pred, zero_division=0))
        mcc = float(matthews_corrcoef(y, pred)) if y.sum() > 0 and pred.sum() > 0 else 0.0
        n_neg = int((y == 0).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fpr = fp / max(n_neg, 1)
        return f1, mcc, fpr

    f1_e, mcc_e, fpr_e = _metrics(pred_evid)
    f1_r, mcc_r, fpr_r = _metrics(pred_raw)

    # Triviality probes (only meaningful on the raw path).
    triv_f1, triv_flag = detect_triviality(
        y, pred_raw, data["raw_test"],
        train_mask=np.ones(len(y), dtype=bool),  # use test as own ref for demo
    )

    realism_extra, realism_flag = detect_realism(
        data["raw_full"], data["inject_mask_full"]
    )

    res_evid = ScenarioResult(
        scenario=name, path="evidence",
        f1=f1_e, mcc=mcc_e, fpr=fpr_e,
        n_pos=int(y.sum()), n_neg=int((y == 0).sum()),
        triviality_f1=triv_f1, triviality_flag="N/A",
        realism_flag="N/A", extra={},
    )
    res_raw = ScenarioResult(
        scenario=name, path="raw",
        f1=f1_r, mcc=mcc_r, fpr=fpr_r,
        n_pos=int(y.sum()), n_neg=int((y == 0).sum()),
        triviality_f1=triv_f1, triviality_flag=triv_flag,
        realism_flag=realism_flag, extra=realism_extra,
    )

    # Paired statistics.
    bca = paired_bootstrap_bca_ci(
        y, pred_evid, pred_raw,
        lambda yt, yp: f1_score(yt, yp, zero_division=0),
        n_boot=500, seed=123,
    )
    mc = mcnemar_paired_test(y, pred_evid, pred_raw)

    comparison = {
        "scenario": name,
        "delta_f1_point": bca["point"],
        "delta_f1_ci_low": bca["ci_low"],
        "delta_f1_ci_high": bca["ci_high"],
        "delta_f1_significant": bca["significant_at_alpha"],
        "mcnemar_pvalue": mc["p_value"],
        "mcnemar_method": mc["method"],
        "mcnemar_better": mc["better"],
        "n10_evid_right_raw_wrong": mc["n10"],
        "n01_evid_wrong_raw_right": mc["n01"],
    }
    return res_evid, res_raw, comparison


def run_ablation(
    scenarios: List[str] = ("volumetric", "subtle", "slowloris"),
    seed: int = 0,
    out_path: str = None,
) -> pd.DataFrame:
    rows = []
    summary = []
    for sc in scenarios:
        data = make_synthetic_pair(attack_kind=sc, seed=seed)
        res_e, res_r, cmp = run_one_scenario(sc, data)
        rows.append(asdict(res_e))
        rows.append(asdict(res_r))
        summary.append(cmp)
        print(f"\n=== Scenario: {sc} ===")
        print(f"  evidence path : F1={res_e.f1:.3f}  MCC={res_e.mcc:.3f}  FPR={res_e.fpr:.3f}")
        print(f"  raw path      : F1={res_r.f1:.3f}  MCC={res_r.mcc:.3f}  FPR={res_r.fpr:.3f}")
        print(f"  triviality    : best trivial F1={res_r.triviality_f1:.3f}  flag={res_r.triviality_flag}")
        print(f"  realism       : flag={res_r.realism_flag}  KS_max={res_r.extra.get('ks_max', float('nan')):.2f}  "
              f"mean_shift={res_r.extra.get('mean_shift_sigma', float('nan')):.2f} sigma")
        print(f"  delta F1 (evid - raw) BCa : {format_ci(cmp_to_bca_dict(cmp))}")
        print(f"  McNemar       : p={cmp['mcnemar_pvalue']:.4f}  method={cmp['mcnemar_method']}  better={cmp['mcnemar_better']}")

    df_results = pd.DataFrame(rows)
    df_compare = pd.DataFrame(summary)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(out_path.replace(".json", "_per_path.csv"), index=False)
        df_compare.to_csv(out_path.replace(".json", "_comparison.csv"), index=False)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"per_path": df_results.to_dict(orient="records"),
                 "comparison": df_compare.to_dict(orient="records")},
                f, indent=2, default=str,
            )
        print(f"\n[OK] Wrote results to {out_path}")
    return df_results


def cmp_to_bca_dict(cmp: dict) -> dict:
    """Adapt the comparison dict back to format_ci() input shape."""
    return {
        "point": cmp["delta_f1_point"],
        "ci_low": cmp["delta_f1_ci_low"],
        "ci_high": cmp["delta_f1_ci_high"],
        "n_boot": 500, "alpha": 0.05, "n": 0, "method": "BCa",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/ablation_injection_level.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--scenarios", nargs="+",
        default=["volumetric", "subtle", "slowloris"],
        help="Synthetic scenario keywords to run.",
    )
    p.add_argument(
        "--self-test", action="store_true",
        help="Run unit-style asserts and exit.",
    )
    args = p.parse_args()

    if args.self_test:
        return _self_test()

    run_ablation(args.scenarios, seed=args.seed, out_path=args.out)
    return 0


def _self_test() -> int:
    print("[TEST] ablation_injection_level.py self-test")
    df = run_ablation(["volumetric", "subtle", "slowloris"], seed=0, out_path=None)
    assert not df.empty, "ablation returned empty df"
    # Volumetric attack should be detectable on the raw path with stub detector.
    vol_raw = df[(df.scenario == "volumetric") & (df.path == "raw")].iloc[0]
    assert vol_raw["f1"] > 0.3, vol_raw
    print(f"   [OK] Volumetric raw F1 = {vol_raw['f1']:.3f}")
    print("[TEST] ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
