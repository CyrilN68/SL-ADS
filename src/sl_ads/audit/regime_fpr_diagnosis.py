"""regime_fpr_diagnosis.py — Root-cause investigation for the regime-FPR
overshoot (TASK-58, Phase B Option C).

Background
----------
The 2026-05-06 regime audit (`evaluate_regime_fpr.py`) measured a 7.02×
overshoot of the operator FPR target on `canonical_ACTIVE` windows
(weekday × not-holiday × hour ∈ [08, 18)) versus 0.79× on
`canonical_QUIET` for the canonical RedeRio reference run.  PATCH H2
(calendar-aware EVT, audit-grade opt-in 2026-05-07) was implemented to
attack this overshoot at the per-metric calibration layer; an empirical
benchmark (`docs/review/calendar_evt_design.md` post-mortem) showed
that per-regime ``t_atk`` thresholds differ from the global value by
only ≈ ±5 % (median ACTIVE/GLOBAL = 1.045, QUIET/GLOBAL = 0.983).
First-order tail estimate: H2 alone reduces 7.02× → ≈ 3.5×, not 1.0×.

Conclusion: the regime-FPR mechanism is **not located at the
per-metric EVT calibration step**.  It must be at the *fusion* layer
(Subjective Logic WBF over per-metric opinions, then inter-method
fusion) or at the *cross-metric correlation* level on ACTIVE windows.

This module produces a reviewer-grade diagnosis that pinpoints the
mechanism, by computing four orthogonal views of the per-regime
benign-window behaviour:

  C.1 — **Per-metric exceedance rates per regime.**
        For each Prophet/RANSAC metric, fraction of benign-only windows
        in regime *r* where ``proj_atk_metric ≥ δ_per_metric``.  If
        per-metric exceedance rates already differ between ACTIVE and
        QUIET (independent of fusion), the imbalance is at the
        evidence-emission layer, not the fusion layer.
  C.2 — **Pairwise correlations of evidence weight per regime.**
        Pearson correlation of ``proj_atk_metric_i`` vs.
        ``proj_atk_metric_j`` restricted to benign-only windows in
        regime *r*.  Heavy positive correlations on ACTIVE imply
        clustered exceedances → WBF fuses them into a single high
        ``proj_atk`` more often than the per-metric union rate would
        suggest.
  C.3 — **Fused proj_atk distribution per regime.**
        Mean, median, p99, p99.9, p99.99 of
        ``FINAL_SYSTEM_CBF_proj_atk`` on benign-only windows in regime
        *r*.  Direct comparison: does the post-fusion distribution
        already separate the two regimes?  If yes, the imbalance is
        post-fusion.
  C.4 — **Joint exceedance counts per regime.**
        For each *k* in {2, 3, …, K_max}, fraction of benign windows
        in regime *r* where ≥ *k* of the available leaf metrics simultaneously have
        ``proj_atk_metric ≥ δ_per_metric``.  Decomposes the union into
        the SL fusion-relevant statistic.

Outputs
-------
The module is run as a script and writes the following under
``outputs/scientific_hardening/``:

  - ``regime_fpr_diagnosis.csv`` — long-format dataframe with one row
    per ``(view, regime, metric_or_pair_or_k)`` triplet.
  - ``regime_fpr_diagnosis.json`` — same content in JSON for the
    reproducibility package.
  - ``regime_fpr_diagnosis.md`` — human-readable summary with the
    headline numbers and a *narrative* identifying the dominant
    mechanism.

The script is **non-destructive**: it only reads the existing
``detection_results_INJECTED.csv`` produced by ``compute_opinions``
plus the canonical attack/outage catalogues from ``CONFIG``.  No
re-training and no re-calibration.

References
----------
- A1.5 (single global EVT threshold per metric), §11.2 of
  ``docs/scientific_deconstruction/ASSUMPTIONS.md``.
- A6.1 (Naive-Bayes group-independence audit), 2026-05-06 measurement
  showing 32 / 66 attack-window pairs with HIGH dependence — but this
  was on ATTACK windows, not on the BENIGN ACTIVE windows that drive
  the FPR overshoot.  This module re-runs the same correlation idea
  but conditioned on regime.
- ``docs/review/calendar_evt_design.md`` §post-mortem — the empirical
  finding that motivated this Option C investigation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from sl_ads.calendar.regime import REGIME_BUCKETS, regime_of_series
from sl_ads.config import CONFIG, REAL_ATTACKS
from sl_ads.inject.evidence_level import ATTACK_CATALOG
from sl_ads.paths import get_decision_threshold


# ───────────────────────────────────────────────────────── helpers

def _resolve_detection_csv(arg: Optional[str]) -> Path:
    if arg:
        return Path(arg).resolve()
    candidates = [
        Path("outputs/detection_results_INJECTED.csv"),
        Path("outputs/detection_results.csv"),
        Path(CONFIG.get("RESULTS_DIR", "")) / "detection_results_INJECTED.csv",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "No detection_results CSV found.  Run compute_opinions first or "
        "pass --csv explicitly."
    )


def _event_bounds(ev: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    t0 = pd.Timestamp(ev["start"])
    if ev.get("end") is not None:
        return t0, pd.Timestamp(ev["end"])
    return t0, t0 + pd.Timedelta(hours=float(ev["duration_h"]))


def _benign_only_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask of windows that are NOT inside any
    catalogue event nor any REAL_ATTACKS / outage interval.

    Mirrors the logic in ``evaluate_regime_fpr._excluded_mask`` so the
    benign base used here is identical to the published regime audit.
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


def _per_metric_proj_atk_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of per-leaf-metric ``*_proj_atk`` columns,
    excluding the directional sub-components and the system aggregates.
    """
    out = []
    for col in df.columns:
        if not col.endswith("_proj_atk"):
            continue
        if col.startswith("FINAL_SYSTEM"):
            continue
        if col.startswith("METHODE_"):
            continue
        if "_dir_pos_" in col or "_dir_neg_" in col:
            continue
        out.append(col)
    return sorted(out)


def _detection_score_column(df: pd.DataFrame) -> str:
    """Resolve the canonical fused detection score column name."""
    for c in ("FINAL_SYSTEM_CBF_proj_atk", "FINAL_SYSTEM_proj_atk"):
        if c in df.columns:
            return c
    raise KeyError(
        "No FINAL_SYSTEM_*_proj_atk column found; cannot run the C.3 view."
    )


# ───────────────────────────────────── C.1 per-metric exceedance

def per_metric_exceedance_per_regime(
    df_benign: pd.DataFrame,
    metric_cols: Iterable[str],
    regimes: pd.Series,
    delta: float,
) -> pd.DataFrame:
    """C.1 — Per-metric exceedance rate of ``δ`` per regime.

    Note: the per-metric ``proj_atk`` is the post-leaf-bijection
    projected attack mass.  Comparing it to the system-level ``δ`` is
    a stress test — operationally we never compare per-metric
    ``proj_atk`` to the system threshold; we use it here as a
    discriminator to see whether one regime already produces more
    "metric-level alarms" than the other before fusion.
    """
    rows = []
    for col in metric_cols:
        for bucket in REGIME_BUCKETS:
            mask = regimes == bucket
            n = int(mask.sum())
            if n == 0:
                continue
            x = pd.to_numeric(df_benign.loc[mask, col], errors="coerce").to_numpy()
            n_finite = int(np.isfinite(x).sum())
            n_above = int(np.sum(x >= delta))
            rate = n_above / n if n else float("nan")
            rows.append({
                "view": "C1_per_metric_exceedance",
                "regime": bucket,
                "metric": col,
                "n_windows": n,
                "n_finite": n_finite,
                "n_above_delta": n_above,
                "rate": rate,
                "delta": delta,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────── C.2 pairwise correlations

def pairwise_proj_atk_correlations_per_regime(
    df_benign: pd.DataFrame,
    metric_cols: list[str],
    regimes: pd.Series,
) -> pd.DataFrame:
    """C.2 — Pairwise Pearson correlation of ``proj_atk`` between
    every pair of metrics, restricted to benign windows in each regime.

    The published A6.1 audit measured the same correlations on attack
    windows and identified 32 / 66 HIGH dependence pairs.  Here we
    measure them on **benign** windows split by regime — the result
    tells us whether the SL fusion is summing correlated noise on
    ACTIVE (which would inflate the joint exceedance rate beyond the
    per-metric rate) or independent noise on QUIET.
    """
    rows = []
    for bucket in REGIME_BUCKETS:
        mask = regimes == bucket
        sub = df_benign.loc[mask, metric_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        if len(sub) < 30:
            continue
        corr = sub.corr(method="pearson")
        names = list(corr.columns)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                rho = float(corr.iat[i, j])
                rows.append({
                    "view": "C2_pairwise_corr",
                    "regime": bucket,
                    "metric_a": names[i],
                    "metric_b": names[j],
                    "rho": rho if math.isfinite(rho) else float("nan"),
                    "abs_rho": abs(rho) if math.isfinite(rho) else float("nan"),
                    "n_windows": int(mask.sum()),
                })
    return pd.DataFrame(rows)


# ─────────────────────────────── C.3 fused proj_atk distribution

def fused_proj_atk_distribution_per_regime(
    df_benign: pd.DataFrame,
    fused_col: str,
    regimes: pd.Series,
    delta: float,
) -> pd.DataFrame:
    """C.3 — Distribution of the fused detection score per regime.

    Headline statistic: the empirical cumulative distribution of
    ``proj_atk_FINAL`` on benign-only windows in each regime, summarised
    at meaningful percentiles.  If the p99.9 of the ACTIVE distribution
    is much closer to ``δ`` than the QUIET p99.9, fusion is the source
    of the regime imbalance.
    """
    rows = []
    for bucket in REGIME_BUCKETS:
        mask = regimes == bucket
        x = pd.to_numeric(df_benign.loc[mask, fused_col], errors="coerce").to_numpy()
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        rows.append({
            "view": "C3_fused_distribution",
            "regime": bucket,
            "n_windows": int(mask.sum()),
            "n_finite": int(x.size),
            "mean": float(x.mean()),
            "median": float(np.median(x)),
            "p90":  float(np.quantile(x, 0.90)),
            "p99":  float(np.quantile(x, 0.99)),
            "p99_9":  float(np.quantile(x, 0.999)),
            "p99_99": float(np.quantile(x, 0.9999)),
            "max":  float(x.max()),
            "frac_above_delta": float(np.mean(x >= delta)),
            "delta": delta,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────── C.4 joint exceedance counts

def joint_exceedance_counts_per_regime(
    df_benign: pd.DataFrame,
    metric_cols: list[str],
    regimes: pd.Series,
    delta: float,
    k_max: int = 8,
) -> pd.DataFrame:
    """C.4 — For each *k*, fraction of benign windows in each regime
    where at least *k* of the per-metric ``proj_atk`` values are above
    ``δ`` simultaneously.

    Under the SL WBF aggregation, the fused ``proj_atk`` is dominated
    by joint per-metric attack mass, not by individual exceedances.
    A regime that produces clusters of simultaneous (correlated)
    per-metric exceedances will have a heavier tail at the fused
    level even if each individual exceedance rate is identical.
    """
    rows = []
    arr_full = df_benign[metric_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    flag_full = arr_full >= delta  # (n_windows, n_metrics)
    sum_per_window = np.nansum(flag_full, axis=1)
    for bucket in REGIME_BUCKETS:
        mask = (regimes == bucket).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        sum_b = sum_per_window[mask]
        for k in range(1, k_max + 1):
            n_above = int(np.sum(sum_b >= k))
            rows.append({
                "view": "C4_joint_exceedance",
                "regime": bucket,
                "k_min_simultaneous": k,
                "n_windows": n,
                "n_above_or_eq": n_above,
                "fraction": n_above / n,
                "delta": delta,
            })
    return pd.DataFrame(rows)


# ────────────────────────────────────── narrative synthesis

def synthesise_narrative(
    c1: pd.DataFrame, c2: pd.DataFrame, c3: pd.DataFrame, c4: pd.DataFrame,
) -> dict:
    """Produce the verdict pointing at the dominant mechanism.

    The verdict picks between three mutually exclusive hypotheses:

      H_evidence  — per-metric exceedance rates already differ
                    significantly between ACTIVE and QUIET (C1 ratio
                    ACTIVE/QUIET > 2× on the median metric).
      H_fusion    — per-metric exceedance rates are similar between
                    regimes, but the fused proj_atk has a heavier tail
                    on ACTIVE (C3 p99.9 ratio > 2×).
      H_correlation — per-metric exceedance rates are similar; fused
                    distribution is similar at low percentiles; but
                    joint exceedance counts diverge for k ≥ 2 (C4
                    ratio > 2× at k=3).
    """
    # Per-metric ratio A/Q on the median metric.
    c1_p = c1.pivot_table(index="metric", columns="regime",
                            values="rate", aggfunc="first")
    if {"ACTIVE", "QUIET"}.issubset(c1_p.columns):
        ratios = c1_p["ACTIVE"] / c1_p["QUIET"].replace(0, np.nan)
        median_evidence_ratio = float(np.nanmedian(ratios))
    else:
        median_evidence_ratio = float("nan")

    # Fused tail ratio.
    c3_a = c3[c3["regime"] == "ACTIVE"]
    c3_q = c3[c3["regime"] == "QUIET"]
    fused_tail_ratio = float("nan")
    if not c3_a.empty and not c3_q.empty:
        a_p99_9 = float(c3_a["p99_9"].iloc[0])
        q_p99_9 = float(c3_q["p99_9"].iloc[0])
        if q_p99_9 > 0:
            fused_tail_ratio = a_p99_9 / q_p99_9

    # Joint exceedance ratio at k=3.
    c4_p = c4.pivot_table(index="k_min_simultaneous", columns="regime",
                            values="fraction", aggfunc="first")
    joint_ratio = float("nan")
    if 3 in c4_p.index and {"ACTIVE", "QUIET"}.issubset(c4_p.columns):
        a3 = float(c4_p.loc[3, "ACTIVE"])
        q3 = float(c4_p.loc[3, "QUIET"])
        if q3 > 0:
            joint_ratio = a3 / q3

    if median_evidence_ratio > 2.0 and not math.isnan(median_evidence_ratio):
        verdict = "H_evidence"
        explanation = (
            f"Per-metric exceedance rates already diverge by a median "
            f"factor of {median_evidence_ratio:.1f}× between ACTIVE and "
            f"QUIET.  The regime imbalance is at the evidence-emission "
            f"layer (Prophet/QR residuals are heteroscedastic across "
            f"regimes).  The targeted fix is calendar-aware EVT "
            f"calibration (PATCH H2) — but the post-mortem benchmark "
            f"showed per-regime thresholds only differ by ±5%, which "
            f"contradicts this hypothesis.  Re-examine whether the "
            f"per-metric proj_atk has a long tail that the EVT "
            f"threshold flattens but the regime-conditional "
            f"distribution retains."
        )
    elif fused_tail_ratio > 2.0 and not math.isnan(fused_tail_ratio):
        verdict = "H_fusion"
        explanation = (
            f"The fused proj_atk p99.9 is {fused_tail_ratio:.1f}× higher "
            f"on ACTIVE than on QUIET, while per-metric exceedance "
            f"rates are similar (median A/Q = "
            f"{median_evidence_ratio:.2f}).  The Subjective Logic WBF "
            f"is amplifying ACTIVE-regime evidence beyond what the "
            f"per-metric rates would predict.  Likely cause: WBF "
            f"weights are confidence-driven; if ACTIVE windows have "
            f"higher per-metric confidence (lower u) the fused tail "
            f"inherits this concentration."
        )
    elif joint_ratio > 2.0 and not math.isnan(joint_ratio):
        verdict = "H_correlation"
        explanation = (
            f"Joint exceedance count (≥ 3 simultaneous per-metric "
            f"alarms) is {joint_ratio:.1f}× higher on ACTIVE than on "
            f"QUIET, while individual per-metric rates and the fused "
            f"low-percentile distribution are similar.  The mechanism "
            f"is cross-metric correlation: in ACTIVE regimes, several "
            f"metrics tend to fire together on the same window "
            f"(common-cause bursts), and SL fusion accumulates these "
            f"correlated alarms into a heavier fused tail.  Targeted "
            f"fix: contextual discounting per regime, or a "
            f"correlation-aware fusion operator."
        )
    else:
        verdict = "H_inconclusive"
        explanation = (
            "None of the three hypotheses cleanly dominates.  All "
            "three diagnostic ratios (per-metric A/Q, fused p99.9 A/Q, "
            "joint k=3 A/Q) are below 2×.  The 7.02× regime-FPR "
            "overshoot may be driven by a combination of small effects "
            "across the layers, or by the aggregation of the legacy "
            "global threshold on a slightly-skewed fused distribution. "
            "Recommend running the same diagnosis on a multi-seed "
            "fold to confirm the verdict."
        )

    return {
        "verdict": verdict,
        "explanation": explanation,
        "median_per_metric_ratio_active_over_quiet": median_evidence_ratio,
        "fused_p99_9_ratio_active_over_quiet": fused_tail_ratio,
        "joint_k3_ratio_active_over_quiet": joint_ratio,
    }


def _df_to_pipe_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GFM pipe-table without depending on
    ``tabulate`` (which is an optional pandas dependency).  Same
    fallback pattern as in ``sl_ads/evaluate/run_multi_seed.py``.
    """
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


def render_markdown(rows: dict, summary: dict, n_active: int, n_quiet: int) -> str:
    """Render a human-readable diagnosis summary."""
    lines = []
    lines.append("# Regime-FPR Root-Cause Diagnosis (Phase B Option C)")
    lines.append("")
    lines.append(f"- ACTIVE benign windows: {n_active}")
    lines.append(f"- QUIET  benign windows: {n_quiet}")
    lines.append(f"- Decision threshold δ : {summary['delta']:.6f}")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{summary['verdict']}**")
    lines.append("")
    lines.append(summary["explanation"])
    lines.append("")
    lines.append("## Diagnostic ratios (ACTIVE / QUIET)")
    lines.append("")
    lines.append("| Ratio | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Median per-metric exceedance rate | "
                 f"{summary['median_per_metric_ratio_active_over_quiet']:.3f} |")
    lines.append(f"| Fused proj_atk p99.9 | "
                 f"{summary['fused_p99_9_ratio_active_over_quiet']:.3f} |")
    lines.append(f"| Joint exceedance k=3 | "
                 f"{summary['joint_k3_ratio_active_over_quiet']:.3f} |")
    lines.append("")
    lines.append("## C.3 Fused proj_atk distribution per regime")
    lines.append("")
    if "c3_table" in rows:
        lines.append(_df_to_pipe_table(rows["c3_table"]))
        lines.append("")
    lines.append("## C.4 Joint exceedance counts per regime")
    lines.append("")
    if "c4_table" in rows:
        lines.append(_df_to_pipe_table(rows["c4_table"]))
        lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────── CLI

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=None,
                   help="Path to detection_results CSV (default: auto-discover).")
    p.add_argument("--out-dir", default="outputs/scientific_hardening")
    p.add_argument("--per-metric-delta", type=float, default=None,
                   help="Per-metric threshold for C.1/C.4. Defaults to "
                        "the system decision threshold; this is a stress "
                        "test, not the operational threshold.")
    p.add_argument("--k-max", type=int, default=8,
                   help="Highest k for the joint-exceedance view.")
    args = p.parse_args(argv)

    csv_path = _resolve_detection_csv(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[regime_fpr_diagnosis] reading {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    benign_mask = _benign_only_mask(df)
    df_benign = df.loc[benign_mask].reset_index(drop=True)
    regimes = regime_of_series(
        df_benign["timestamp"], holidays=CONFIG.get("HOLIDAYS_LIST"),
    )
    n_active = int((regimes == "ACTIVE").sum())
    n_quiet  = int((regimes == "QUIET").sum())
    print(f"  benign windows: {len(df_benign)} "
          f"(ACTIVE={n_active}, QUIET={n_quiet})")

    delta = float(get_decision_threshold(CONFIG, up_levels=1))
    per_metric_delta = (args.per_metric_delta
                          if args.per_metric_delta is not None else delta)
    print(f"  system δ = {delta:.6f}")
    print(f"  per-metric δ used in C.1/C.4 = {per_metric_delta:.6f}")

    metric_cols = _per_metric_proj_atk_columns(df)
    fused_col   = _detection_score_column(df)
    print(f"  per-leaf metric columns: {len(metric_cols)}")
    print(f"  fused-score column     : {fused_col}")

    c1 = per_metric_exceedance_per_regime(df_benign, metric_cols, regimes,
                                            per_metric_delta)
    c2 = pairwise_proj_atk_correlations_per_regime(df_benign, metric_cols,
                                                     regimes)
    c3 = fused_proj_atk_distribution_per_regime(df_benign, fused_col, regimes,
                                                  delta)
    c4 = joint_exceedance_counts_per_regime(df_benign, metric_cols, regimes,
                                              per_metric_delta,
                                              k_max=args.k_max)
    summary = synthesise_narrative(c1, c2, c3, c4)
    summary["delta"] = delta
    summary["per_metric_delta"] = per_metric_delta
    summary["n_active_benign"] = n_active
    summary["n_quiet_benign"]  = n_quiet
    summary["detection_csv"]   = str(csv_path)

    long_df = pd.concat([c1, c2, c3, c4], ignore_index=True, sort=False)
    csv_out  = out_dir / "regime_fpr_diagnosis.csv"
    json_out = out_dir / "regime_fpr_diagnosis.json"
    md_out   = out_dir / "regime_fpr_diagnosis.md"
    long_df.to_csv(csv_out, index=False)
    json_payload = {
        "summary": summary,
        "rows":    long_df.to_dict(orient="records"),
    }
    json_out.write_text(json.dumps(json_payload, indent=2,
                                   default=str), encoding="utf-8")

    md = render_markdown({"c3_table": c3, "c4_table": c4}, summary,
                          n_active, n_quiet)
    md_out.write_text(md, encoding="utf-8")

    print(f"[OK] wrote {csv_out}")
    print(f"[OK] wrote {json_out}")
    print(f"[OK] wrote {md_out}")
    print()
    print(f"[VERDICT] {summary['verdict']}")
    print(f"  median per-metric A/Q   = "
          f"{summary['median_per_metric_ratio_active_over_quiet']:.3f}")
    print(f"  fused p99.9 A/Q         = "
          f"{summary['fused_p99_9_ratio_active_over_quiet']:.3f}")
    print(f"  joint k=3 A/Q           = "
          f"{summary['joint_k3_ratio_active_over_quiet']:.3f}")
    print()
    print(summary["explanation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
