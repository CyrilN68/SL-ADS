"""Leak-free same-evidence comparison: SL-ADS vs non-SL score baselines.

This module answers the narrow publication question:

    "What does the same anomaly detector look like with and without the
    Subjective Logic layer?"

The comparison is deliberately *not* an external SOTA comparison.  Every
non-SL baseline reads the same evidence triplets as SL-ADS, but replaces the
SL bijection, uncertainty, ageing, EDP, and fusion operators with direct scalar
scores over attack evidence N.

Critical anti-leakage rule
--------------------------
Non-SL thresholds are calibrated on the persisted train-calibration residuals
(`models_pkg['_calib_signed_residuals']`) and then evaluated once on the test
span.  Test labels are used only for reporting metrics.

Outputs
-------
`results/resultats_<VERSION>/evaluation_no_sl_fair/`
    no_sl_fair_summary.csv
    no_sl_fair_paired_vs_sl.csv
    no_sl_fair_per_episode.csv
    no_sl_fair_thresholds.csv
    no_sl_fair_report.md
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sl_ads.config import CONFIG, REAL_ATTACKS
from sl_ads.evaluate.vus_metrics import vus_summary
from sl_ads.inject.evidence_level import ATTACK_CATALOG as INJECTED_ATTACK_CATALOG
from sl_ads.paths import (
    get_decision_threshold,
    get_detection_col,
    get_model_path,
    get_results_dir,
    get_version_names,
)
from sl_ads.qualif_filters import is_outage_event
from sl_ads.stats.bootstrap_ci import paired_bootstrap_bca_ci
from sl_ads.stats.mcnemar import mcnemar_paired_test
from sl_ads.train.compute_evidence import compute_instantaneous_evidence


WINDOW_SIZE = int(CONFIG.get("WINDOW_SIZE", 10))
FPR_TARGET = float(CONFIG.get("FPR_TARGET_DECISION", 0.001))
BLOCK_LENGTH = 36
N_BOOT = int(os.environ.get("SL_NO_SL_N_BOOT", "1000"))


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    description: str
    fn: Callable[[np.ndarray, np.ndarray, list[str], list[str]], np.ndarray]


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _event_bounds(ev: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    t0 = pd.Timestamp(ev["start"])
    if ev.get("duration_h") is not None:
        t1 = t0 + pd.Timedelta(hours=float(ev["duration_h"]))
    elif ev.get("end") is not None:
        t1 = pd.Timestamp(ev["end"])
    else:
        raise KeyError(f"event has no duration_h/end: {ev!r}")
    return t0, t1


def _catalog() -> list[dict]:
    out = list(INJECTED_ATTACK_CATALOG)
    if CONFIG.get("EVAL", {}).get("INCLUDE_REAL_ATTACK", True):
        out.extend(CONFIG.get("EVAL", {}).get("REAL_ATTACK_CATALOG", []))
    return out


def _real_events() -> list[dict]:
    events: list[dict] = []
    for name, items in REAL_ATTACKS.items():
        for ev in items:
            e = dict(ev)
            e.setdefault("name", name)
            events.append(e)
    return events


def _mask_from_events(timestamps: pd.Series, events: list[dict]) -> np.ndarray:
    mask = np.zeros(len(timestamps), dtype=bool)
    for ev in events:
        t0, t1 = _event_bounds(ev)
        mask |= ((timestamps >= t0) & (timestamps < t1)).to_numpy()
    return mask


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, Path]:
    version_name, version_modif = get_version_names(CONFIG)
    results_dir = Path(get_results_dir(CONFIG, up_levels=1))
    model_path = Path(get_model_path(CONFIG, up_levels=1))
    model_pkg = joblib.load(model_path)

    evidence_path = results_dir / f"evidence_{version_modif}.csv"
    if not evidence_path.exists():
        evidence_path = results_dir / f"evidence_{version_name}.csv"
    detection_path = results_dir / "detection_results_INJECTED.csv"
    metadata_path = results_dir / f"metadata_{version_name}.csv"

    evidence = pd.read_csv(evidence_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    detection = pd.read_csv(detection_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    metadata = pd.read_csv(metadata_path)
    return evidence, detection, metadata, model_pkg, results_dir


def _metric_keys(metadata: pd.DataFrame) -> tuple[list[str], list[str], np.ndarray]:
    rows = metadata.sort_values("metric_key")
    keys = rows["metric_key"].astype(str).tolist()
    clean = rows["clean_key"].astype(str).tolist()
    weights = rows["r2_weight"].astype(float).clip(lower=0.01).to_numpy()
    return keys, clean, weights


def _residuals_to_calib_n_matrix(model_pkg: dict, metric_keys: list[str]) -> np.ndarray:
    """Map persisted train-calib residuals to window-level N fractions."""
    residuals = model_pkg.get("_calib_signed_residuals") or {}
    if not residuals:
        raise RuntimeError(
            "model package has no _calib_signed_residuals; cannot calibrate "
            "non-SL baselines without leakage."
        )

    per_metric: list[np.ndarray] = []
    n_windows = None
    for key in metric_keys:
        if key not in residuals or key not in model_pkg:
            raise RuntimeError(f"missing calibration residuals/model for {key}")
        arr = np.asarray(residuals[key], dtype=float)
        usable = (arr.size // WINDOW_SIZE) * WINDOW_SIZE
        arr = arr[:usable]
        pkg = model_pkg[key]
        n_vals = np.fromiter(
            (_residual_to_n(float(r), pkg) for r in arr),
            dtype=float,
            count=arr.size,
        )
        win_n = n_vals.reshape(-1, WINDOW_SIZE).sum(axis=1) / float(WINDOW_SIZE)
        per_metric.append(win_n)
        n_windows = win_n.size if n_windows is None else min(n_windows, win_n.size)

    if n_windows is None or n_windows == 0:
        raise RuntimeError("no calibration windows available")
    return np.column_stack([x[:n_windows] for x in per_metric])


def _residual_to_n(r: float, pkg: dict) -> float:
    direction = pkg.get("direction", "both")
    if direction == "both":
        if r >= 0:
            _, _, n = compute_instantaneous_evidence(
                r,
                float(pkg.get("t_susp_pos", pkg["t_susp"])),
                float(pkg.get("t_atk_pos", pkg["t_atk"])),
                float(pkg.get("t_trapeze_base_pos", pkg["t_trapeze_base"])),
                "pos",
            )
        else:
            _, _, n = compute_instantaneous_evidence(
                r,
                float(pkg.get("t_susp_neg", pkg["t_susp"])),
                float(pkg.get("t_atk_neg", pkg["t_atk"])),
                float(pkg.get("t_trapeze_base_neg", pkg["t_trapeze_base"])),
                "neg",
            )
    else:
        _, _, n = compute_instantaneous_evidence(
            r,
            float(pkg["t_susp"]),
            float(pkg["t_atk"]),
            float(pkg["t_trapeze_base"]),
            str(direction),
        )
    return float(n)


def _test_n_matrix(evidence: pd.DataFrame, clean_keys: list[str]) -> np.ndarray:
    cols = []
    for ck in clean_keys:
        n_col = f"{ck}_N"
        p_col = f"{ck}_P"
        s_col = f"{ck}_S"
        if n_col not in evidence.columns:
            raise RuntimeError(f"missing evidence column {n_col}")
        denom = None
        if p_col in evidence.columns and s_col in evidence.columns:
            denom = (
                evidence[p_col].fillna(0.0)
                + evidence[s_col].fillna(0.0)
                + evidence[n_col].fillna(0.0)
            ).to_numpy(dtype=float)
        if denom is None:
            denom = np.full(len(evidence), float(WINDOW_SIZE))
        denom = np.where(denom > 1e-12, denom, float(WINDOW_SIZE))
        cols.append(evidence[n_col].fillna(0.0).to_numpy(dtype=float) / denom)
    return np.column_stack(cols)


def _align_evidence_to_detection(
    evidence: pd.DataFrame,
    detection: pd.DataFrame,
    clean_keys: list[str],
) -> pd.DataFrame:
    """Use detection timestamps as the canonical evaluation timeline.

    `compute_evidence` keeps the physical window center/end timestamp, whereas
    `opinions_pipeline` writes fixed 5-minute window-start timestamps.  They
    differ by a deterministic offset (`00:04:30` evidence -> `00:00:00`
    detection on RedeRio).  A nearest join is unsafe because it can assign the
    same evidence row to two adjacent detection windows on ties.  Flooring the
    evidence timestamp to the configured decision window reproduces the
    pipeline's window-start convention exactly.
    """
    n_cols = [f"{ck}_N" for ck in clean_keys]
    missing = [c for c in n_cols if c not in evidence.columns]
    if missing:
        raise RuntimeError(f"missing evidence columns before alignment: {missing}")

    freq = pd.to_timedelta(CONFIG.get("freq_data", "30s"))
    window = freq * WINDOW_SIZE
    ev = evidence.sort_values("timestamp").reset_index(drop=True).copy()
    ev["timestamp"] = ev["timestamp"].dt.floor(window)
    if ev["timestamp"].duplicated().any():
        dupes = ev.loc[ev["timestamp"].duplicated(), "timestamp"].head(5).astype(str).tolist()
        raise RuntimeError(f"evidence timestamps are duplicated after floor alignment: {dupes}")

    aligned = detection[["timestamp"]].sort_values("timestamp").reset_index(drop=True).merge(
        ev,
        on="timestamp",
        how="left",
    )
    if len(aligned) != len(detection):
        raise RuntimeError(
            f"evidence/detection alignment changed row count: "
            f"{len(aligned)} != {len(detection)}"
        )
    bad = aligned[n_cols].isna().any(axis=1)
    if bool(bad.any()):
        examples = aligned.loc[bad, "timestamp"].head(5).astype(str).tolist()
        raise RuntimeError(
            f"evidence/detection alignment failed for {int(bad.sum())} rows "
            f"(examples: {examples})"
        )
    return aligned


def _score_specs() -> list[ScoreSpec]:
    def mean_n(x, w, keys, types):
        return np.mean(x, axis=1)

    def max_n(x, w, keys, types):
        return np.max(x, axis=1)

    def top3_n(x, w, keys, types):
        k = min(3, x.shape[1])
        return np.sort(x, axis=1)[:, -k:].mean(axis=1)

    def r2_weighted(x, w, keys, types):
        ww = np.asarray(w, dtype=float)
        return (x * ww).sum(axis=1) / ww.sum()

    def prophet_mean(x, w, keys, types):
        mask = np.asarray([t == "prophet" for t in types])
        return np.mean(x[:, mask], axis=1)

    def reconst_mean(x, w, keys, types):
        mask = np.asarray([t == "reconstruction" for t in types])
        return np.mean(x[:, mask], axis=1)

    def hard_vote_90(x, w, keys, types):
        ww = np.asarray(w, dtype=float)
        return ((x >= 0.90).astype(float) * ww).sum(axis=1) / ww.sum()

    return [
        ScoreSpec("no_sl_mean_N", "Mean attack evidence N over all leaves", mean_n),
        ScoreSpec("no_sl_max_N", "Maximum attack evidence N over leaves", max_n),
        ScoreSpec("no_sl_top3_mean_N", "Mean of top-3 attack-evidence leaves", top3_n),
        ScoreSpec("no_sl_r2_weighted_mean_N", "R2-weighted mean attack evidence", r2_weighted),
        ScoreSpec("no_sl_prophet_mean_N", "Mean attack evidence over Prophet leaves only", prophet_mean),
        ScoreSpec("no_sl_reconst_mean_N", "Mean attack evidence over reconstruction leaves only", reconst_mean),
        ScoreSpec("no_sl_hard_vote_90_r2", "R2-weighted fraction of leaves with N>=0.90", hard_vote_90),
    ]


def _calibrate_threshold(scores: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """Most permissive threshold with empirical calibration FPR <= target."""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("empty calibration scores")
    candidates = np.unique(s)
    candidates = np.concatenate(([np.nextafter(float(candidates[-1]), math.inf)], candidates))
    ok: list[tuple[float, float]] = []
    for thr in candidates:
        fpr = float(np.mean(s >= thr))
        if fpr <= target_fpr + 1e-15:
            ok.append((float(thr), fpr))
    if not ok:
        return float(np.nextafter(float(np.max(s)), math.inf)), 0.0
    # Choose the smallest accepted threshold, i.e. highest possible sensitivity.
    return min(ok, key=lambda z: z[0])


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = _safe_div(tp, tp + fp)
    tpr = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "tpr": tpr,
        "fpr": fpr,
        "f1_micro": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
    }


def _episode_metrics(
    timestamps: pd.Series,
    y_pred_full: np.ndarray,
    catalog: list[dict],
    system: str,
    threshold: float,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    for atk in catalog:
        t0, t1 = _event_bounds(atk)
        mask = ((timestamps >= t0) & (timestamps < t1)).to_numpy()
        pred = y_pred_full[mask]
        n = int(mask.sum())
        n_det = int(pred.sum())
        detected = bool(n_det > 0)
        coverage = _safe_div(n_det, n)
        ttd_min = float("nan")
        if detected:
            first_ts = timestamps[mask].iloc[np.flatnonzero(pred)[0]]
            ttd_min = (pd.Timestamp(first_ts) - t0).total_seconds() / 60.0
        rows.append({
            "system": system,
            "attack": atk.get("name", atk.get("type", "UNKNOWN")),
            "n_windows": n,
            "detected": int(detected),
            "coverage": coverage,
            "ttd_minutes": ttd_min,
            "threshold": threshold,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}
    detected = int(df["detected"].sum())
    recall_binary = _safe_div(detected, len(df))
    recall_coverage = float(df["coverage"].mean())
    ttd_med = float(df.loc[df["detected"] == 1, "ttd_minutes"].median()) if detected else float("nan")
    return df, {
        "n_detected_attacks": detected,
        "n_attacks": int(len(df)),
        "recall_binary": recall_binary,
        "recall_coverage": recall_coverage,
        "median_ttd_minutes": ttd_med,
    }


def _auc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["pr_auc"] = float("nan")
    return out


def _f1_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


def _write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    threshold_info: pd.DataFrame,
) -> None:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        cols = [str(c) for c in df.columns]
        rows = [[_fmt_cell(v) for v in row] for row in df.to_numpy()]
        widths = [len(c) for c in cols]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def fmt(vals: list[str]) -> str:
            return "| " + " | ".join(vals[i].ljust(widths[i]) for i in range(len(vals))) + " |"

        sep = "| " + " | ".join("-" * w for w in widths) + " |"
        return "\n".join([fmt(cols), sep] + [fmt(r) for r in rows])

    def _fmt_cell(v) -> str:
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(float(v)):
                return "nan"
            return f"{float(v):.6g}"
        return str(v)

    best_no_sl = summary[
        (summary["family"] == "same_evidence_no_sl")
        & (summary["protocol"] == "catalog_outages_separate")
    ].sort_values("f1_micro", ascending=False).head(3)
    lines = [
        "# Same-Evidence SL vs No-SL Fair Comparison",
        "",
        "Thresholds for non-SL systems are calibrated on train-calib residuals only.",
        "The test span is used once for reporting metrics.",
        "",
        "## Main Catalog Protocol",
        "",
        md_table(summary[summary["protocol"] == "catalog_outages_separate"][
            [
                "system",
                "family",
                "threshold",
                "threshold_source",
                "f1_micro",
                "f1_macro",
                "mcc",
                "precision",
                "tpr",
                "fpr_pct",
                "recall_coverage",
                "n_detected_attacks",
                "n_attacks",
            ]
        ]),
        "",
        "## Best No-SL Rows",
        "",
        md_table(best_no_sl[["system", "f1_micro", "f1_macro", "mcc", "fpr_pct", "recall_coverage"]]),
        "",
        "## Paired Tests vs Full SL-ADS",
        "",
        md_table(paired),
        "",
        "## Calibration Thresholds",
        "",
        md_table(threshold_info),
        "",
        "## Interpretation Guardrail",
        "",
        "A non-SL row is comparable here only because it uses the same evidence CSV "
        "and a threshold calibrated on train-calib.  Test-FPR-matched or best-test "
        "thresholds are intentionally not used for headline rows.",
    ]
    (out_dir / "no_sl_fair_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    evidence, detection, metadata, model_pkg, results_dir = _load_inputs()
    out_dir = results_dir / "evaluation_no_sl_fair"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_keys, clean_keys, weights = _metric_keys(metadata)
    types = metadata.sort_values("metric_key")["type"].astype(str).tolist()
    evidence = _align_evidence_to_detection(evidence, detection, clean_keys)
    calib_matrix = _residuals_to_calib_n_matrix(model_pkg, metric_keys)
    test_matrix = _test_n_matrix(evidence, clean_keys)

    timestamps = detection["timestamp"]
    catalog = _catalog()
    real_events = _real_events()
    outage_events = [ev for ev in real_events if is_outage_event(ev)]

    catalog_mask = _mask_from_events(timestamps, catalog)
    outage_mask = _mask_from_events(timestamps, outage_events)
    anomaly_mask = catalog_mask | _mask_from_events(timestamps, real_events)

    valid_catalog = ~outage_mask
    protocols = {
        "catalog_outages_separate": (catalog_mask.astype(int), valid_catalog),
        "operator_faithful_anomaly": (anomaly_mask.astype(int), np.ones(len(timestamps), dtype=bool)),
    }

    systems: list[dict] = []
    threshold_rows: list[dict] = []

    det_col = get_detection_col(CONFIG, up_levels=1)
    sl_scores = detection[det_col].astype(float).to_numpy()
    if len(sl_scores) != len(evidence):
        raise RuntimeError(
            f"SL score length mismatch after canonical alignment: "
            f"{len(sl_scores)} != {len(evidence)}"
        )
    sl_thr = float(get_decision_threshold(CONFIG, up_levels=1))
    systems.append({
        "system": "Full SL-ADS",
        "family": "sl_reference",
        "score": sl_scores,
        "threshold": sl_thr,
        "threshold_source": "trained sidecar",
        "calib_fpr_emp": float("nan"),
        "description": "SL bijection + EDP + conflict-aware ageing + WBF",
    })

    for spec in _score_specs():
        calib_scores = spec.fn(calib_matrix, weights, metric_keys, types)
        test_scores = spec.fn(test_matrix, weights, metric_keys, types)
        thr, calib_fpr = _calibrate_threshold(calib_scores, FPR_TARGET)
        systems.append({
            "system": spec.name,
            "family": "same_evidence_no_sl",
            "score": test_scores,
            "threshold": thr,
            "threshold_source": "train-calib normal quantile",
            "calib_fpr_emp": calib_fpr,
            "description": spec.description,
        })
        threshold_rows.append({
            "system": spec.name,
            "threshold": thr,
            "target_fpr": FPR_TARGET,
            "empirical_calib_fpr": calib_fpr,
            "n_calib_windows": int(len(calib_scores)),
            "calib_score_p50": float(np.quantile(calib_scores, 0.50)),
            "calib_score_p99": float(np.quantile(calib_scores, 0.99)),
            "calib_score_p999": float(np.quantile(calib_scores, 0.999)),
            "description": spec.description,
        })

    summary_rows: list[dict] = []
    episode_frames: list[pd.DataFrame] = []
    pred_by_system: dict[str, np.ndarray] = {}

    for sys_row in systems:
        score = np.asarray(sys_row["score"], dtype=float)
        pred_full = (score >= float(sys_row["threshold"])).astype(np.int8)
        pred_by_system[sys_row["system"]] = pred_full
        ep_df, ep_metrics = _episode_metrics(
            timestamps, pred_full, catalog, sys_row["system"], float(sys_row["threshold"])
        )
        episode_frames.append(ep_df)
        for protocol, (y_all, valid_mask) in protocols.items():
            y = y_all[valid_mask].astype(np.int8)
            pred = pred_full[valid_mask].astype(np.int8)
            score_valid = score[valid_mask]
            bm = _binary_metrics(y, pred)
            aucs = _auc_metrics(y, score_valid)
            vus = vus_summary(y, score_valid, y_pred=pred)
            precision = bm["precision"]
            rec_bin = ep_metrics.get("recall_binary", float("nan"))
            rec_cov = ep_metrics.get("recall_coverage", float("nan"))
            f1_bin_h = _safe_div(2 * precision * rec_bin, precision + rec_bin)
            f1_cov_h = _safe_div(2 * precision * rec_cov, precision + rec_cov)
            summary_rows.append({
                "system": sys_row["system"],
                "family": sys_row["family"],
                "protocol": protocol,
                "threshold": float(sys_row["threshold"]),
                "threshold_source": sys_row["threshold_source"],
                "calib_fpr_emp": sys_row["calib_fpr_emp"],
                "description": sys_row["description"],
                **bm,
                "fpr_pct": 100.0 * bm["fpr"],
                "roc_auc": aucs["roc_auc"],
                "pr_auc": aucs["pr_auc"],
                "vus_roc": vus.get("vus_roc", float("nan")),
                "vus_pr": vus.get("vus_pr", float("nan")),
                "range_auc_roc_at_max": vus.get("range_auc_roc_at_max", float("nan")),
                "range_auc_pr_at_max": vus.get("range_auc_pr_at_max", float("nan")),
                "n_detected_attacks": ep_metrics.get("n_detected_attacks", 0),
                "n_attacks": ep_metrics.get("n_attacks", 0),
                "recall_binary_episode": rec_bin,
                "recall_coverage": rec_cov,
                "median_ttd_minutes": ep_metrics.get("median_ttd_minutes", float("nan")),
                "f1_binary_hybrid_episode_recall": f1_bin_h,
                "f1_coverage_hybrid_episode_recall": f1_cov_h,
            })

    summary = pd.DataFrame(summary_rows)
    threshold_info = pd.DataFrame(threshold_rows)
    per_episode = pd.concat(episode_frames, ignore_index=True)

    # Paired tests vs Full SL-ADS on the canonical catalog/outages-separate protocol.
    y_ref_all, valid_ref = protocols["catalog_outages_separate"]
    y_ref = y_ref_all[valid_ref].astype(np.int8)
    sl_pred = pred_by_system["Full SL-ADS"][valid_ref].astype(np.int8)
    paired_rows = []
    for sys_row in systems:
        name = sys_row["system"]
        if name == "Full SL-ADS":
            continue
        pred = pred_by_system[name][valid_ref].astype(np.int8)
        mc = mcnemar_paired_test(y_ref, sl_pred, pred)
        try:
            delta = paired_bootstrap_bca_ci(
                y_ref,
                sl_pred,
                pred,
                _f1_metric,
                n_boot=N_BOOT,
                seed=42,
                block_length=BLOCK_LENGTH,
            )
        except Exception as exc:
            delta = {
                "point": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "method": f"failed: {exc}",
                "significant_at_alpha": False,
            }
        paired_rows.append({
            "system_b": name,
            "comparison": "Full SL-ADS - system_b",
            "delta_f1_micro": delta["point"],
            "delta_f1_ci_low": delta["ci_low"],
            "delta_f1_ci_high": delta["ci_high"],
            "delta_f1_method": delta["method"],
            "delta_f1_significant": delta.get("significant_at_alpha", False),
            "mcnemar_p": mc["p_value"],
            "mcnemar_statistic": mc["statistic"],
            "mcnemar_better": mc["better"],
            "n10_sl_right_b_wrong": mc["n10"],
            "n01_sl_wrong_b_right": mc["n01"],
            "n_disc": mc["n_disc"],
        })
    paired = pd.DataFrame(paired_rows)

    summary.to_csv(out_dir / "no_sl_fair_summary.csv", index=False)
    threshold_info.to_csv(out_dir / "no_sl_fair_thresholds.csv", index=False)
    per_episode.to_csv(out_dir / "no_sl_fair_per_episode.csv", index=False)
    paired.to_csv(out_dir / "no_sl_fair_paired_vs_sl.csv", index=False)
    manifest = {
        "threshold_calibration": "train-calib residuals from _calib_signed_residuals",
        "target_fpr": FPR_TARGET,
        "window_size": WINDOW_SIZE,
        "n_boot": N_BOOT,
        "block_length": BLOCK_LENGTH,
        "outputs": [p.name for p in out_dir.iterdir() if p.is_file()],
    }
    (out_dir / "no_sl_fair_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(out_dir, summary, paired, threshold_info)

    print(f"[OK] no-SL fair comparison written to {out_dir}")
    print(summary[summary["protocol"] == "catalog_outages_separate"][
        ["system", "family", "threshold", "f1_micro", "f1_macro", "mcc", "fpr_pct", "n_detected_attacks", "n_attacks"]
    ].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
