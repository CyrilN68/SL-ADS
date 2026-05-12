"""Leak-free raw-data baselines for RedeRio.

This comparison answers a different question from `compare_no_sl_fair`:

* `compare_no_sl_fair`: same ADS evidence, with vs without Subjective Logic.
* this module: external baselines trained directly on raw network metrics.

Important scope guard
---------------------
The 13 catalog attacks are injected at *evidence* level, not into the raw
traffic CSV.  Raw-data baselines therefore cannot be evaluated on those
synthetic episodes as if the raw signal contained them.  This script reports
only label protocols that are meaningful from raw data available today:

1. `pseudo_csv_excluding_synthetic`:
   the standardized CSV `label` column, excluding synthetic injection windows.
2. `real_events_excluding_synthetic`:
   REAL_ATTACKS intervals, including outages, excluding synthetic windows.

All raw baselines fit on pre-split rows only and calibrate their thresholds on
pre-split normal windows only.  Test labels are used only for reporting.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from sl_ads.config import CONFIG, REAL_ATTACKS
from sl_ads.inject.evidence_level import ATTACK_CATALOG as SYNTHETIC_ATTACK_CATALOG
from sl_ads.paths import get_decision_threshold, get_detection_col, get_results_dir, get_version_names
from sl_ads.stats.bootstrap_ci import paired_bootstrap_bca_ci
from sl_ads.stats.mcnemar import mcnemar_paired_test


FPR_TARGET = float(CONFIG.get("FPR_TARGET_DECISION", 0.001))
N_BOOT = int(os.environ.get("SL_RAW_BASELINE_N_BOOT", "1000"))
BLOCK_LENGTH = 36
LOF_FIT_CAP = int(os.environ.get("SL_RAW_BASELINE_LOF_FIT_CAP", "10000"))
OCSVM_FIT_CAP = int(os.environ.get("SL_RAW_BASELINE_OCSVM_FIT_CAP", "5000"))
SGD_OCSVM_FIT_CAP = int(os.environ.get("SL_RAW_BASELINE_SGD_OCSVM_FIT_CAP", "20000"))
SGD_RBF_COMPONENTS = int(os.environ.get("SL_RAW_BASELINE_SGD_RBF_COMPONENTS", "256"))


@dataclass(frozen=True)
class BaselineResult:
    name: str
    description: str
    train_scores: np.ndarray
    test_scores: np.ndarray


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _window_string() -> str:
    freq = pd.to_timedelta(CONFIG.get("freq_data", "30s"))
    window_size = int(CONFIG.get("WINDOW_SIZE", 10))
    return f"{int(freq.total_seconds() * window_size / 60)}min"


def _event_bounds(ev: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    t0 = pd.Timestamp(ev["start"])
    if ev.get("duration_h") is not None:
        t1 = t0 + pd.Timedelta(hours=float(ev["duration_h"]))
    elif ev.get("end") is not None:
        t1 = pd.Timestamp(ev["end"])
    else:
        raise KeyError(f"event has no duration_h/end: {ev!r}")
    return t0, t1


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


def _calibrate_threshold(scores: np.ndarray, target_fpr: float) -> tuple[float, float]:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("empty calibration scores")
    candidates = np.unique(np.sort(scores))
    best_thr = float(candidates[-1])
    best_fpr = float(np.mean(scores >= best_thr))
    for thr in candidates:
        fpr = float(np.mean(scores >= thr))
        if fpr <= target_fpr + 1e-15:
            best_thr = float(thr)
            best_fpr = fpr
            break
    if best_fpr > target_fpr + 1e-15:
        best_thr = float(np.nextafter(candidates[-1], np.inf))
        best_fpr = 0.0
    return best_thr, best_fpr


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    tpr = float(recall_score(y_true, y_pred, zero_division=0))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "tpr": tpr,
        "fpr": _safe_div(fp, fp + tn),
        "f1_micro": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
    }


def _auc_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    out = {"roc_auc": float("nan"), "pr_auc": float("nan")}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        pass
    try:
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    except Exception:
        pass
    return out


def _f1_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


def _raw_path() -> Path:
    p = Path(str(CONFIG.get("file_path")))
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, Path, str]:
    version_name, _ = get_version_names(CONFIG)
    results_dir = Path(get_results_dir(CONFIG, up_levels=1))
    raw = pd.read_csv(_raw_path(), parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    raw_sl_path = results_dir / "opinions_non_injected" / "detection_results_RAW.csv"
    injected_sl_path = results_dir / "detection_results_INJECTED.csv"
    if raw_sl_path.exists():
        detection_path = raw_sl_path
        sl_source = "non_injected_opinions"
    elif os.environ.get("SL_ALLOW_INJECTED_SL_FOR_RAW_BASELINE", "").strip().lower() in ("1", "true", "yes"):
        detection_path = injected_sl_path
        sl_source = "injected_opinions_excluding_synthetic_windows"
    else:
        raise RuntimeError(
            "Raw-baseline comparison requires non-injected SL opinions. Run:\n"
            "  SL_FORCE_NONINJECTED_OPINIONS=1 SL_SKIP_OPINION_PLOTS=1 "
            "python -m sl_ads.core.opinions_pipeline\n"
            "before `python -m sl_ads.compare.compare_raw_baselines_fair`.\n"
            "For diagnostic-only fallback to injected SL scores, set "
            "SL_ALLOW_INJECTED_SL_FOR_RAW_BASELINE=1."
        )
    detection = pd.read_csv(detection_path, parse_dates=["timestamp"])
    detection = detection.sort_values("timestamp").reset_index(drop=True)
    return raw, detection, results_dir, sl_source


def _prepare_raw(raw: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    missing = [c for c in metrics if c not in raw.columns]
    if missing:
        raise RuntimeError(f"missing raw metric columns: {missing}")
    out = raw[["timestamp", *metrics, *(["label"] if "label" in raw.columns else [])]].copy()
    if "label" not in out.columns:
        out["label"] = 0
    out["label"] = (out["label"].fillna(0).astype(float) > 0).astype(np.int8)
    out["synthetic_interval"] = _mask_from_events(out["timestamp"], list(SYNTHETIC_ATTACK_CATALOG)).astype(np.int8)
    out["real_event"] = _mask_from_events(out["timestamp"], _real_events()).astype(np.int8)
    return out


def _fit_raw_baselines(
    train_fit: pd.DataFrame,
    train_calib: pd.DataFrame,
    test: pd.DataFrame,
    metrics: list[str],
) -> list[BaselineResult]:
    seed = int(CONFIG.get("RANDOM_SEED") or 0)
    fill = train_fit[metrics].median(numeric_only=True)
    x_fit = train_fit[metrics].fillna(fill).to_numpy(dtype=float)
    x_calib = train_calib[metrics].fillna(fill).to_numpy(dtype=float)
    x_test = test[metrics].fillna(fill).to_numpy(dtype=float)

    scaler = StandardScaler()
    x_fit_s = scaler.fit_transform(x_fit)
    x_calib_s = scaler.transform(x_calib)
    x_test_s = scaler.transform(x_test)

    def fit_subset(x: np.ndarray, cap: int) -> np.ndarray:
        if cap <= 0 or len(x) <= cap:
            return x
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(x), size=cap, replace=False))
        return x[idx]

    if_model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=seed,
        n_jobs=1,
    )
    if_model.fit(x_fit_s)

    lof_fit_s = fit_subset(x_fit_s, LOF_FIT_CAP)
    lof = LocalOutlierFactor(
        n_neighbors=35,
        contamination="auto",
        novelty=True,
        n_jobs=1,
    )
    lof.fit(lof_fit_s)

    ocsvm_fit_s = fit_subset(x_fit_s, OCSVM_FIT_CAP)
    ocsvm = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.01,
    )
    ocsvm.fit(ocsvm_fit_s)

    sgd_fit_s = fit_subset(x_fit_s, SGD_OCSVM_FIT_CAP)
    rbf = RBFSampler(
        gamma=1.0 / max(1, x_fit_s.shape[1]),
        n_components=SGD_RBF_COMPONENTS,
        random_state=seed,
    )
    z_sgd_fit = rbf.fit_transform(sgd_fit_s)
    sgd_ocsvm = SGDOneClassSVM(
        nu=0.01,
        max_iter=2000,
        tol=1e-3,
        random_state=seed,
        shuffle=True,
        average=True,
    )
    sgd_ocsvm.fit(z_sgd_fit)

    med = np.nanmedian(x_fit, axis=0)
    mad = np.nanmedian(np.abs(x_fit - med), axis=0)
    mad = np.where(mad > 1e-12, mad, 1.0)

    pca = PCA(n_components=0.95, svd_solver="full", random_state=int(CONFIG.get("RANDOM_SEED") or 0))
    pca.fit(x_fit_s)

    def robust_z(x: np.ndarray) -> np.ndarray:
        z = 0.6745 * np.abs(x - med) / mad
        return np.nanmax(z, axis=1)

    def pca_err(x_s: np.ndarray) -> np.ndarray:
        proj = pca.inverse_transform(pca.transform(x_s))
        return np.mean((x_s - proj) ** 2, axis=1)

    return [
        BaselineResult(
            "raw_iforest",
            "IsolationForest on raw ACTIVE_METRICS, score=max window anomaly score",
            -if_model.decision_function(x_calib_s),
            -if_model.decision_function(x_test_s),
        ),
        BaselineResult(
            "raw_lof_novelty",
            f"LocalOutlierFactor novelty on standardized raw ACTIVE_METRICS (train-normal fit cap={len(lof_fit_s)})",
            -lof.score_samples(x_calib_s),
            -lof.score_samples(x_test_s),
        ),
        BaselineResult(
            "raw_ocsvm_rbf",
            f"RBF OneClassSVM on standardized raw ACTIVE_METRICS (train-normal fit cap={len(ocsvm_fit_s)}, nu=0.01)",
            -ocsvm.decision_function(x_calib_s),
            -ocsvm.decision_function(x_test_s),
        ),
        BaselineResult(
            "raw_sgd_ocsvm_rbf",
            (
                "SGDOneClassSVM with RBFSampler on standardized raw ACTIVE_METRICS "
                f"(train-normal fit cap={len(sgd_fit_s)}, n_components={SGD_RBF_COMPONENTS}, nu=0.01)"
            ),
            -sgd_ocsvm.decision_function(rbf.transform(x_calib_s)),
            -sgd_ocsvm.decision_function(rbf.transform(x_test_s)),
        ),
        BaselineResult(
            "raw_robust_z_max",
            "max robust modified z-score over raw ACTIVE_METRICS",
            robust_z(x_calib),
            robust_z(x_test),
        ),
        BaselineResult(
            "raw_pca_reconstruction",
            "PCA reconstruction error on standardized raw ACTIVE_METRICS",
            pca_err(x_calib_s),
            pca_err(x_test_s),
        ),
    ]


def _window_scores(df: pd.DataFrame, scores: np.ndarray, window: str) -> pd.DataFrame:
    tmp = df[["timestamp", "label", "synthetic_interval", "real_event"]].copy()
    tmp["score"] = np.asarray(scores, dtype=float)
    tmp["window_start"] = tmp["timestamp"].dt.floor(window)
    return (
        tmp.groupby("window_start", as_index=False)
        .agg(
            score=("score", "max"),
            pseudo_label=("label", "max"),
            synthetic_interval=("synthetic_interval", "max"),
            real_event=("real_event", "max"),
        )
        .rename(columns={"window_start": "timestamp"})
    )


def _sl_windows(detection: pd.DataFrame, window: str) -> pd.DataFrame:
    det_col = get_detection_col(CONFIG, up_levels=1)
    out = detection[["timestamp", det_col]].copy()
    out["window_start"] = out["timestamp"].dt.floor(window)
    return (
        out.groupby("window_start", as_index=False)
        .agg(sl_score=(det_col, "max"))
        .rename(columns={"window_start": "timestamp"})
    )


def _episode_metrics(timestamps: pd.Series, pred: np.ndarray, events: list[dict]) -> dict:
    detected = 0
    coverages: list[float] = []
    for ev in events:
        t0, t1 = _event_bounds(ev)
        mask = ((timestamps >= t0) & (timestamps < t1)).to_numpy()
        if not bool(mask.any()):
            continue
        n_hit = int(pred[mask].sum())
        detected += int(n_hit > 0)
        coverages.append(n_hit / int(mask.sum()))
    return {
        "n_real_events": len(events),
        "n_detected_real_events": detected,
        "real_event_coverage": float(np.mean(coverages)) if coverages else float("nan"),
    }


def _write_report(out_dir: Path, summary: pd.DataFrame, paired: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    def table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        cols = [str(c) for c in df.columns]
        rows = []
        for row in df.to_numpy():
            vals = []
            for v in row:
                if isinstance(v, (float, np.floating)):
                    vals.append("nan" if not np.isfinite(float(v)) else f"{float(v):.6g}")
                else:
                    vals.append(str(v))
            rows.append(vals)
        widths = [len(c) for c in cols]
        for r in rows:
            widths = [max(w, len(v)) for w, v in zip(widths, r)]
        fmt = lambda r: "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(r)) + " |"
        return "\n".join([fmt(cols), "| " + " | ".join("-" * w for w in widths) + " |", *[fmt(r) for r in rows]])

    lines = [
        "# Raw-Data Baselines Fair Comparison",
        "",
        "Raw baselines are trained on pre-split raw metrics only. Thresholds are calibrated on pre-split normal windows.",
        "Synthetic catalog attacks are excluded from raw-data label protocols because they were injected at evidence level, not raw-traffic level.",
        "",
        "## Summary",
        "",
        table(summary[[
            "system",
            "score_source",
            "protocol",
            "threshold",
            "f1_micro",
            "f1_macro",
            "mcc",
            "precision",
            "tpr",
            "fpr_pct",
            "roc_auc",
            "pr_auc",
        ]]),
        "",
        "## Paired Tests Vs SL",
        "",
        table(paired),
        "",
        "## Thresholds",
        "",
        table(thresholds),
    ]
    (out_dir / "raw_baselines_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    raw, detection, results_dir, sl_source = _load_inputs()
    out_dir = results_dir / "evaluation_raw_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    window = _window_string()
    split = pd.Timestamp(CONFIG["split_date"])
    metrics = list(CONFIG.get("ACTIVE_METRICS", []))
    raw = _prepare_raw(raw, metrics)

    train = raw[raw["timestamp"] <= split].copy()
    test = raw[raw["timestamp"] > split].copy()
    train_clean = train[train["label"] == 0].copy()
    if train_clean.empty:
        raise RuntimeError("no clean pre-split raw rows available for baseline training")

    baselines = _fit_raw_baselines(train_clean, train_clean, test, metrics)
    sl_win = _sl_windows(detection, window)

    train_threshold_rows: list[dict] = []
    systems: list[dict] = [{
        "system": "Full SL-ADS",
        "family": "sl_reference",
        "score_source": sl_source,
        "threshold": float(get_decision_threshold(CONFIG, up_levels=1)),
        "threshold_source": "trained sidecar",
        "score_df": sl_win.rename(columns={"sl_score": "score"}),
        "description": f"SL output from pipeline ({sl_source}); synthetic windows excluded where raw labels cannot contain injected attacks",
    }]

    for b in baselines:
        train_win = _window_scores(train_clean, b.train_scores, window)
        test_win = _window_scores(test, b.test_scores, window)
        thr, emp_fpr = _calibrate_threshold(train_win["score"].to_numpy(), FPR_TARGET)
        systems.append({
            "system": b.name,
            "family": "raw_baseline",
            "score_source": "raw_metrics",
            "threshold": thr,
            "threshold_source": "pre-split normal-window quantile",
            "score_df": test_win,
            "description": b.description,
            "empirical_calib_fpr": emp_fpr,
        })
        train_threshold_rows.append({
            "system": b.name,
            "threshold": thr,
            "target_fpr": FPR_TARGET,
            "empirical_calib_fpr": emp_fpr,
            "n_calib_windows": int(len(train_win)),
            "score_p50": float(np.quantile(train_win["score"], 0.50)),
            "score_p99": float(np.quantile(train_win["score"], 0.99)),
            "score_p999": float(np.quantile(train_win["score"], 0.999)),
            "description": b.description,
        })

    # Labels come from the raw test windows, not from the SL CSV.
    label_win = _window_scores(test, np.zeros(len(test)), window)[
        ["timestamp", "pseudo_label", "synthetic_interval", "real_event"]
    ]

    protocols = {
        "pseudo_csv_excluding_synthetic": ("pseudo_label", label_win["synthetic_interval"].to_numpy() == 0),
        "real_events_excluding_synthetic": ("real_event", label_win["synthetic_interval"].to_numpy() == 0),
    }

    summary_rows: list[dict] = []
    pred_by_protocol: dict[tuple[str, str], np.ndarray] = {}
    y_by_protocol: dict[str, np.ndarray] = {}
    valid_by_protocol: dict[str, np.ndarray] = {}

    for sys_row in systems:
        merged = label_win.merge(sys_row["score_df"][["timestamp", "score"]], on="timestamp", how="left")
        if merged["score"].isna().any():
            missing = int(merged["score"].isna().sum())
            raise RuntimeError(f"{sys_row['system']} missing {missing} score windows after raw alignment")
        score = merged["score"].to_numpy(dtype=float)
        pred = (score >= float(sys_row["threshold"])).astype(np.int8)
        ep = _episode_metrics(merged["timestamp"], pred, _real_events())
        for protocol, (label_col, valid) in protocols.items():
            y = merged[label_col].to_numpy(dtype=np.int8)
            valid = np.asarray(valid, dtype=bool)
            y_valid = y[valid]
            pred_valid = pred[valid]
            score_valid = score[valid]
            bm = _binary_metrics(y_valid, pred_valid)
            auc = _auc_metrics(y_valid, score_valid)
            summary_rows.append({
                "system": sys_row["system"],
                "family": sys_row["family"],
                "score_source": sys_row["score_source"],
                "protocol": protocol,
                "threshold": float(sys_row["threshold"]),
                "threshold_source": sys_row["threshold_source"],
                "description": sys_row["description"],
                "n_windows_valid": int(valid.sum()),
                "n_positive_windows": int(y_valid.sum()),
                **bm,
                "fpr_pct": 100.0 * bm["fpr"],
                **auc,
                **ep,
            })
            pred_by_protocol[(protocol, sys_row["system"])] = pred_valid
            y_by_protocol[protocol] = y_valid
            valid_by_protocol[protocol] = valid

    paired_rows: list[dict] = []
    for protocol in protocols:
        y = y_by_protocol[protocol]
        sl_pred = pred_by_protocol[(protocol, "Full SL-ADS")]
        for sys_row in systems:
            name = sys_row["system"]
            if name == "Full SL-ADS":
                continue
            pred = pred_by_protocol[(protocol, name)]
            mc = mcnemar_paired_test(y, sl_pred, pred)
            try:
                delta = paired_bootstrap_bca_ci(
                    y,
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
                "protocol": protocol,
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

    summary = pd.DataFrame(summary_rows)
    thresholds = pd.DataFrame(train_threshold_rows)
    paired = pd.DataFrame(paired_rows)

    summary.to_csv(out_dir / "raw_baselines_summary.csv", index=False)
    thresholds.to_csv(out_dir / "raw_baselines_thresholds.csv", index=False)
    paired.to_csv(out_dir / "raw_baselines_paired_vs_sl.csv", index=False)
    manifest = {
        "scope_guard": "raw baselines cannot see evidence-level synthetic injections",
        "sl_score_source": sl_source,
        "window": window,
        "split_date": str(split),
        "metrics": metrics,
        "target_fpr": FPR_TARGET,
        "n_boot": N_BOOT,
        "protocols": list(protocols),
        "raw_baseline_fit_caps": {
            "lof_fit_cap": LOF_FIT_CAP,
            "ocsvm_fit_cap": OCSVM_FIT_CAP,
            "sgd_ocsvm_fit_cap": SGD_OCSVM_FIT_CAP,
            "sgd_rbf_components": SGD_RBF_COMPONENTS,
        },
        "note": (
            "LOF / exact OCSVM / SGD-OCSVM caps are deterministic train-normal "
            "subsamples used only for model fitting; threshold calibration still "
            "uses train-normal windows and test labels are reporting-only."
        ),
    }
    (out_dir / "raw_baselines_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(out_dir, summary, paired, thresholds)

    print(f"[OK] raw baseline comparison written to {out_dir}")
    print(summary[[
        "system",
        "protocol",
        "threshold",
        "f1_micro",
        "f1_macro",
        "mcc",
        "fpr_pct",
        "roc_auc",
        "pr_auc",
        "n_positive_windows",
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
