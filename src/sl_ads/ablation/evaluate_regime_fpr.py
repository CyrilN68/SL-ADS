"""Empirical FPR by calendar/time regime for deployment-stationarity audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sl_ads.config import CONFIG, REAL_ATTACKS
from sl_ads.inject.evidence_level import ATTACK_CATALOG
from sl_ads.paths import get_decision_threshold, get_detection_col
from sl_ads.evaluate.axelsson_ppv import _bca_ci_proportion, _effective_n_indicator
from sl_ads.calendar.regime import (  # PATCH H2 — canonical regime function
    REGIME_BUCKETS,
    REGIME_FN_SIGNATURE,
    regime_of_series,
)


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
    raise FileNotFoundError("No detection CSV found; pass --csv.")


def _event_bounds(ev: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    t0 = pd.Timestamp(ev["start"])
    if ev.get("end") is not None:
        return t0, pd.Timestamp(ev["end"])
    return t0, t0 + pd.Timedelta(hours=float(ev["duration_h"]))


def _excluded_mask(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(False, index=df.index)
    events = []
    events.extend(ATTACK_CATALOG)
    events.extend(CONFIG.get("EVAL", {}).get("REAL_ATTACK_CATALOG", []) or [])
    for evs in REAL_ATTACKS.values():
        events.extend(evs)
    for ev in events:
        t0, t1 = _event_bounds(ev)
        out |= (df["timestamp"] >= t0) & (df["timestamp"] < t1)
    return out


def _regime_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Diagnostic regime masks (overlapping by design).

    These rows are kept for backward-compatibility with the
    pre-PATCH-H2 CSV consumers and provide a finer-grained breakdown
    than the canonical 2-bucket partition.  The canonical
    ``ACTIVE``/``QUIET`` partition used by the calibration is
    appended below via ``_canonical_partition_masks``.
    """
    ts = df["timestamp"]
    holiday_dates = {
        pd.Timestamp(h["ds"]).date() for h in CONFIG.get("HOLIDAYS_LIST", [])
        if isinstance(h, dict) and "ds" in h
    }
    weekend = ts.dt.dayofweek >= 5
    holiday = ts.dt.date.isin(holiday_dates)
    night = ts.dt.hour < 6
    day = (ts.dt.hour >= 8) & (ts.dt.hour < 18)
    shoulder = ~(night | day)
    return {
        "all_normal": pd.Series(True, index=df.index),
        "weekday_term_like": (~weekend) & (~holiday),
        "weekend": weekend,
        "holiday_or_closure": holiday,
        "day_08_18": day,
        "night_00_06": night,
        "shoulder_06_08_18_24": shoulder,
    }


def _canonical_partition_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Canonical 2-bucket partition (PATCH H2 ; ``ACTIVE``/``QUIET``).

    These rows are *disjoint* and exhaust every window, mirroring the
    partition consumed by ``calibrate_thresholds_per_regime_v2`` at
    training time.  Reporting them next to the diagnostic regime rows
    lets a reviewer verify that the FPR-per-bucket numbers used by the
    calibration are the same as the audit's empirical estimate.
    """
    labels = regime_of_series(df["timestamp"], holidays=CONFIG.get("HOLIDAYS_LIST"))
    return {
        f"canonical_{bucket}": (labels == bucket).reindex(df.index, fill_value=False)
        for bucket in REGIME_BUCKETS
    }


def analyse(df: pd.DataFrame, threshold: float, det_col: str) -> tuple[pd.DataFrame, dict]:
    excluded = _excluded_mask(df)
    normal = ~excluded
    pred = (df[det_col].fillna(0.0).astype(float) >= threshold).to_numpy()
    rows = []
    # PATCH H2 — emit both the diagnostic regime rows (overlapping) AND
    # the canonical 2-bucket partition rows (disjoint, used by the
    # calibration).  The canonical rows let a reviewer cross-check the
    # per-bucket realised FPR against the per-bucket sidecar entry
    # without re-running the audit on a custom mask.
    masks_diagnostic = _regime_masks(df)
    masks_canonical  = _canonical_partition_masks(df)
    for name, mask in {**masks_diagnostic, **masks_canonical}.items():
        m = (normal & mask).to_numpy()
        n = int(m.sum())
        fp = int((pred & m).sum())
        fpr = fp / n if n else np.nan
        n_eff = _effective_n_indicator(pred[m].astype(int), max_lag=10) if n else np.nan
        lo, hi = _bca_ci_proportion(fp, n, n_eff=n_eff) if n else (np.nan, np.nan)
        rows.append({
            "regime": name,
            "n_normal_windows": n,
            "fp_windows": fp,
            "fpr": fpr,
            "fpr_pct": 100 * fpr if np.isfinite(fpr) else np.nan,
            "fpr_ci_low": lo,
            "fpr_ci_high": hi,
            "n_eff": n_eff,
        })
    duration_days = float(
        (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400
    )
    fpr_target = float(CONFIG.get("FPR_TARGET_DECISION", float("nan")))
    res = pd.DataFrame(rows)
    res["fpr_target"] = fpr_target
    res["fpr_ratio_to_target"] = np.where(
        fpr_target > 0, res["fpr"] / fpr_target, np.nan
    )

    # A1.5 — year-deployment projection.  Assumes the regime distribution
    # observed over the audited span is representative of one academic
    # calendar year.  This is documented as an assumption in the report;
    # in particular it does NOT include term-vs-vacation reweighting.
    annual = _annual_projection(res, fpr_target)

    # Worst-case regime FPR (upper Wilson bound) — informs the "deployable
    # for one year" decision: the probable peak FPR in the worst regime.
    daytime_rows = res[res["regime"].isin(
        ["day_08_18", "weekday_term_like"]
    )]
    worst_fpr = (
        float(daytime_rows["fpr_ci_high"].max())
        if not daytime_rows.empty else float("nan")
    )

    summary = {
        "dataset_start": str(df["timestamp"].min()),
        "dataset_end": str(df["timestamp"].max()),
        "n_windows": int(len(df)),
        "duration_days": duration_days,
        "threshold": threshold,
        "detection_column": det_col,
        "excluded_attack_or_outage_windows": int(excluded.sum()),
        "fpr_target_decision": fpr_target,
        "all_normal_fpr_observed": float(res.loc[res.regime == "all_normal", "fpr"].iloc[0]),
        "all_normal_fpr_ratio_to_target": float(
            res.loc[res.regime == "all_normal", "fpr_ratio_to_target"].iloc[0]
        ),
        "worst_daytime_fpr_upper_ci_95": worst_fpr,
        "annual_projection": annual,
        "deployment_caveat": (
            "Year projection assumes the regime distribution observed on the "
            "audited span (45 days, late November to late December 2025) is "
            "representative of one academic year. Term-vs-vacation residual "
            "variance differences are NOT modelled: A1.5 still applies."
        ),
    }
    return res, summary


def _annual_projection(res: pd.DataFrame, fpr_target: float) -> dict:
    """Project observed regime FPRs to a 365-day deployment.

    Model: each regime contributes ``regime_fraction × FPR_regime`` to the
    expected FP-rate; the global expected annual FAR is the sum across
    regimes.  Five-minute windows give 105,120 windows/year.
    """
    windows_per_year = 12 * 24 * 365
    target_rows = res[res["regime"] != "all_normal"].copy()
    if target_rows.empty:
        return {}
    total_n = float(target_rows["n_normal_windows"].sum())
    if total_n <= 0:
        return {}
    target_rows["regime_fraction"] = target_rows["n_normal_windows"] / total_n
    target_rows["expected_fp_per_year"] = (
        target_rows["regime_fraction"]
        * target_rows["fpr"].fillna(0.0)
        * windows_per_year
    )
    weighted_fpr = float(
        (target_rows["regime_fraction"] * target_rows["fpr"].fillna(0.0)).sum()
    )
    expected_fp = float(target_rows["expected_fp_per_year"].sum())
    return {
        "windows_per_year": windows_per_year,
        "expected_fpr_weighted_by_regime": weighted_fpr,
        "expected_fp_count_per_year": expected_fp,
        "fpr_target": fpr_target,
        "expected_fp_count_under_target": (
            fpr_target * windows_per_year if fpr_target > 0 else None
        ),
        "ratio_realised_to_target": (
            weighted_fpr / fpr_target if fpr_target > 0 else None
        ),
        "regime_fractions": {
            row["regime"]: float(row["regime_fraction"])
            for _, row in target_rows.iterrows()
        },
    }


def plot(res: pd.DataFrame, out_png: Path) -> None:
    r = res.sort_values("fpr_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(r["regime"], r["fpr_pct"], color="#3b82f6")
    ax.set_ylabel("FPR (%)")
    ax.set_title("Empirical FPR by calendar/time regime")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=None)
    p.add_argument("--out-dir", default="outputs/scientific_hardening")
    args = p.parse_args()

    csv_path = Path(args.csv) if args.csv else _default_detection_csv()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    threshold = get_decision_threshold(CONFIG, up_levels=1)
    det_col = get_detection_col(CONFIG, up_levels=1)
    res, summary = analyse(df, threshold, det_col)
    out_csv = out_dir / "regime_fpr.csv"
    out_png = out_dir / "regime_fpr.png"
    out_json = out_dir / "regime_fpr_summary.json"
    res.to_csv(out_csv, index=False)
    plot(res, out_png)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_png}")
    print(f"[OK] wrote {out_json}")
    print(json.dumps(summary, indent=2))
    print(res.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
