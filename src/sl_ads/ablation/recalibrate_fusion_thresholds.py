"""Recompute mode-specific decision-threshold sidecars from an existing model.

The full training entrypoint can be slow because it refits Prophet/QR models.
For fusion-operator comparisons we only need the normal holdout residuals used
to calibrate ``DECISION_THRESHOLD``.  This module reloads the trained model
package, recomputes those residuals on the calibration split, then writes one
threshold sidecar per requested inter-method fusion mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sl_ads.core import fusion_policy
import sl_ads.core.subjective_logic as sl
from sl_ads.config import CONFIG
from sl_ads.paths import get_model_path, get_threshold_sidecar_path
from sl_ads.preprocessing_utils import preprocess_metrics
from sl_ads.train import train_models as tm


def _effective_cd_alpha(models_pkg: dict) -> float:
    raw = CONFIG.get("RECONST_ATTACK_RELIABILITY", 1.0)
    if str(raw).strip().lower() == "auto":
        raw = models_pkg.get("reconst_attack_reliability", 1.0)
    if raw is None:
        return 1.0
    return float(raw)


def _load_calibration_frame() -> pd.DataFrame:
    df = pd.read_csv(CONFIG["file_path"])
    df["ds"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("ds")

    metric_cols = CONFIG.get("ACTIVE_METRICS", [])
    if isinstance(metric_cols, str):
        metric_cols = []
    for exc in CONFIG.get("TRAIN_EXCLUSIONS", []):
        try:
            start = pd.to_datetime(exc["start"])
            end = pd.to_datetime(exc["end"])
            mask = (df["ds"] >= start) & (df["ds"] <= end)
            if mask.any() and metric_cols:
                df.loc[mask, metric_cols] = np.nan
        except Exception:
            pass

    non_metric_cols = [c for c in df.columns if c not in metric_cols and c != "ds"]
    df[non_metric_cols] = df[non_metric_cols].ffill().fillna(0)
    df = preprocess_metrics(df, limit_ffill=CONFIG.get("NAN_FFILL_LIMIT", 10))

    if df["ds"].duplicated().any():
        df = df.drop_duplicates(subset=["ds"], keep="first")

    holidays = CONFIG.get("HOLIDAYS_LIST", CONFIG.get("RIO_HOLIDAYS", []))
    is_cal_wknd = df["ds"].dt.dayofweek >= 5
    if holidays:
        holidays_df = pd.DataFrame(holidays)
        holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])
        is_holiday = df["ds"].dt.normalize().isin(holidays_df["ds"].dt.normalize())
        df["on_weekend"] = (is_cal_wknd | is_holiday).astype(int)
    else:
        df["on_weekend"] = is_cal_wknd.astype(int)
    df["on_weekday"] = (1 - df["on_weekend"]).astype(int)

    split_date = pd.to_datetime(CONFIG["split_date"])
    df_train = df[df["ds"] <= split_date].copy()
    calib_fraction = float(CONFIG.get("CALIB_SPLIT_FRACTION", 0.0))
    df_train_sorted = df_train.sort_values("ds").reset_index(drop=True)
    n_model = max(10, int(len(df_train_sorted) * (1.0 - calib_fraction)))
    df_train_calib = df_train_sorted.iloc[n_model:].copy()
    if df_train_calib.empty:
        df_train_calib = df_train_sorted.copy()
    return df_train_calib


def _recompute_calibration_residuals(models_pkg: dict,
                                     df_calib: pd.DataFrame) -> dict[str, np.ndarray]:
    residuals: dict[str, np.ndarray] = {}
    for key, pkg in sorted(models_pkg.items()):
        if not isinstance(pkg, dict) or str(key).startswith("_"):
            continue
        typ = str(pkg.get("type", "")).lower()
        model = pkg.get("model")
        if model is None:
            continue

        if typ == "reconstruction":
            try:
                parts = key.replace("reconst_", "").split("_from_")
                target, feature = parts[0], parts[1]
            except Exception:
                print(f"  [WARN] Bad reconstruction key format: {key}")
                continue
            if target not in df_calib.columns or feature not in df_calib.columns:
                print(f"  [WARN] Missing calibration columns for {key}")
                continue
            calib_tmp = tm.apply_exclusions(df_calib.copy(), metric_col=target)
            calib_tmp = tm.apply_exclusions(calib_tmp, metric_col=feature)
            clean = calib_tmp.dropna(subset=[feature, target])
            if clean.empty:
                continue
            y_true = clean[target].to_numpy()
            y_pred = model.predict(clean[[feature]].to_numpy())
            residuals[key] = y_true - y_pred

        elif typ == "prophet":
            target = key.replace("prophet_", "", 1)
            needed = ["ds", target, "on_weekend", "on_weekday"]
            if any(col not in df_calib.columns for col in needed):
                print(f"  [WARN] Missing calibration columns for {key}")
                continue
            calib_m = df_calib[needed].copy()
            calib_m = tm.apply_exclusions(calib_m, metric_col=target)
            calib_p = calib_m.rename(columns={target: "y"}).reset_index(drop=True)
            fcst = model.predict(calib_p)
            mask = ~np.isnan(calib_p["y"])
            if not mask.any():
                continue
            y_true = calib_p.loc[mask, "y"].to_numpy()
            y_pred = fcst.loc[mask, "yhat"].to_numpy()
            residuals[key] = y_true - y_pred

    return residuals


def _write_sidecar(mode: str,
                   result: dict,
                   model_path: str,
                   generic: bool,
                   modes: list[str],
                   models_pkg: dict,
                   calibration_chain: str) -> str:
    sidecar = {
        "decision_threshold": result["decision_threshold"],
        "decision_variable": result["decision_variable"],
        "fpr_target": CONFIG.get("FPR_TARGET_DECISION", 0.01),
        "evt_enabled": CONFIG.get("USE_EVT_THRESHOLDS", True),
        "evt_q_susp": CONFIG.get("EVT_Q_SUSP", 1e-2),
        "evt_q_atk": CONFIG.get("EVT_Q_ATK", 1e-4),
        "calib_strategy": result["calib_strategy"],
        "b_atk_train_n_windows": result["n_windows"],
        "b_atk_train_nonzero": result["nonzero"],
        "fusion_mode_at_calibration": mode,
        "wbf_weight_mode": str(CONFIG.get("WBF_WEIGHT_MODE", "uniform")),
        "lambda_decay": float(CONFIG.get("LAMBDA_DECAY", 0.85)),
        "cd_alpha_attack": _effective_cd_alpha(models_pkg),
        "balance_ratio": tm._effective_balance_ratio(),
        "threshold_sidecar_scope": "legacy_active_mode" if generic else "mode_specific",
        "threshold_calibration_modes": modes,
        "method_groups": CONFIG.get("FUSION_METHOD_GROUPS"),
        "calibration_chain": calibration_chain,
        "calibration_surrogate": (
            "existing model package + recomputed normal holdout residuals + "
            f"{calibration_chain} mode-aware method fusion"
        ),
        "calibration_surrogate_caveat": (
            "Mode-specific threshold sidecar for strict WBF/ABF comparison. "
            "It reuses the trained forecasting/reconstruction models and "
            "recomputes normal holdout residuals; no test/injection data is "
            "used for threshold selection."
        ),
        "source_model_path": os.path.abspath(model_path),
        "patch_id": "TASK-50 (strict mode-specific WBF/ABF recalibration)",
        "iso_date": "2026-05-07",
    }
    path = get_threshold_sidecar_path(
        CONFIG,
        up_levels=1,
        fusion_mode=None if generic else mode,
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[OK] Sidecar {mode}{' generic' if generic else ''}: {path}")
    return path


def _compute_training_proj_atk_with_ageing(models_pkg: dict,
                                           residuals: dict[str, np.ndarray],
                                           window_size: int,
                                           fusion_mode: str) -> np.ndarray:
    """Replay the deployed temporal-ageing chain on normal calibration residuals."""
    W_BIJ = float(CONFIG.get("SL_PARAM_K", 3.0))
    edp = models_pkg.get("empirical_priors") or {}
    metric_keys = [
        k for k in models_pkg
        if isinstance(models_pkg[k], dict)
        and not str(k).startswith("_")
        and k not in ("empirical_priors", "trust_scores")
        and k in residuals
    ]
    if not metric_keys:
        return np.array([])

    metric_meta = {k: models_pkg[k] for k in metric_keys}
    method_groups = fusion_policy.get_method_groups(CONFIG)
    group_keys = fusion_policy.group_metric_keys(metric_meta, CONFIG)
    group_name_by_metric = {
        key: group["name"]
        for group in method_groups
        for key in group_keys.get(group["name"], [])
    }

    def _prior_for_metric(key: str) -> np.ndarray:
        prior = edp.get(key) if isinstance(edp, dict) else None
        if prior:
            a = np.array([prior["a_safe"], prior["a_susp"], prior["a_atk"]], dtype=float)
        else:
            a = np.asarray(CONFIG.get("SL_PRIOR_A", [1 / 3, 1 / 3, 1 / 3]), dtype=float)
        s = float(a.sum())
        return a / s if s > 0 else np.full(3, 1.0 / 3.0)

    state_memory = {
        key: (_prior_for_metric(key) * W_BIJ if key in edp else np.zeros(3))
        for key in metric_keys
    }
    n_windows = min(len(residuals[key]) // window_size for key in metric_keys)
    lam = float(CONFIG.get("LAMBDA_DECAY", 0.85))
    alpha_conf = float(CONFIG.get("CONFLICT_ALPHA", 1.495))
    wbf_weight_mode = str(CONFIG.get("WBF_WEIGHT_MODE", "uniform")).lower()
    c3_mode = str(CONFIG.get("C3_WEIGHT_MODE", "uniform")).lower()

    def _weight_for_metric(key: str) -> float:
        if c3_mode == "uniform":
            return 1.0
        return max(float(models_pkg[key].get("r2_score", 0.01)), 0.01)

    out = []
    for win_idx in range(n_windows):
        ops_by_group = {group["name"]: [] for group in method_groups}
        weights_by_group = {group["name"]: [] for group in method_groups}
        for key in metric_keys:
            pkg = models_pkg[key]
            window = residuals[key][win_idx * window_size:(win_idx + 1) * window_size]
            direction = pkg["direction"]
            t_susp = pkg["t_susp"]; t_atk = pkg["t_atk"]
            t_tb = pkg["t_trapeze_base"]
            t_sp = pkg.get("t_susp_pos", t_susp)
            t_ap = pkg.get("t_atk_pos", t_atk)
            t_tbp = pkg.get("t_trapeze_base_pos", t_tb)
            t_sn = pkg.get("t_susp_neg", t_susp)
            t_an = pkg.get("t_atk_neg", t_atk)
            t_tbn = pkg.get("t_trapeze_base_neg", t_tb)
            P_w = S_w = N_w = 0.0
            for r in window:
                if direction == "pos":
                    if r <= 0.0:
                        P_w += 1.0
                        continue
                    p, s, n = tm._apply_trapezoid_single(r, t_susp, t_atk, t_tb)
                elif direction == "neg":
                    if r >= 0.0:
                        P_w += 1.0
                        continue
                    p, s, n = tm._apply_trapezoid_single(abs(r), t_susp, t_atk, t_tb)
                elif direction == "both":
                    if r >= 0.0:
                        p, s, n = tm._apply_trapezoid_single(r, t_sp, t_ap, t_tbp)
                    else:
                        p, s, n = tm._apply_trapezoid_single(abs(r), t_sn, t_an, t_tbn)
                else:
                    p, s, n = tm._apply_trapezoid_single(abs(r), t_susp, t_atk, t_tb)
                P_w += p; S_w += s; N_w += n

            r_current = np.array([P_w, S_w, N_w], dtype=float)
            r_new, _, _ = sl.temporal_adaptive_ageing(
                r_accumulated=state_memory[key],
                r_current=r_current,
                lam_base=lam,
                W=W_BIJ,
                alpha=alpha_conf,
            )
            state_memory[key] = r_new
            op_leaf = sl.evidence_to_opinion(r_new, W=W_BIJ, a=_prior_for_metric(key))
            if wbf_weight_mode == "trust_discount":
                trust = float(models_pkg.get("trust_scores", {}).get(key, _prior_for_metric(key)[0]))
                op_leaf = sl.apply_trust_discount(op_leaf, trust)
                weight = 1.0
            else:
                weight = _weight_for_metric(key)
            group_name = group_name_by_metric[key]
            ops_by_group[group_name].append(op_leaf)
            weights_by_group[group_name].append(weight)

        method_ops = []
        for group in method_groups:
            group_name = group["name"]
            group_ops = ops_by_group.get(group_name, [])
            if not group_ops:
                continue
            ext_w = None if wbf_weight_mode == "trust_discount" else weights_by_group[group_name]
            op_group = sl.fusion_wbf_n_sources(group_ops, external_weights=ext_w, W=W_BIJ)
            explicit_alpha = None
            if group.get("attack_discount_config_key") == "RECONST_ATTACK_RELIABILITY":
                explicit_alpha = _effective_cd_alpha(models_pkg)
            method_ops.append(
                fusion_policy.apply_method_discount(
                    op_group, group, CONFIG, explicit_alpha=explicit_alpha
                )
            )
        op_final = fusion_policy.fuse_method_opinions(
            method_ops,
            mode=fusion_mode,
            W=W_BIJ,
            balance_ratio_eff=tm._effective_balance_ratio(),
        )
        out.append(float(op_final.projected_prob()[2]))
    return np.asarray(out, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="wbf,abf")
    parser.add_argument("--chain", choices=["ageing", "instant"], default="ageing",
                        help="Calibration chain used for threshold scores.")
    parser.add_argument("--write-generic", action="store_true",
                        help="Also overwrite the generic production sidecar.")
    parser.add_argument("--update-model", action="store_true",
                        help="Persist threshold metadata back into the pkl.")
    args = parser.parse_args()

    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    if not modes:
        raise SystemExit("No modes requested.")

    model_path = get_model_path(CONFIG, up_levels=1)
    print(f"-> Loading model package: {model_path}")
    models_pkg = joblib.load(model_path)

    print("-> Recomputing calibration residuals from existing models...")
    df_calib = _load_calibration_frame()
    residuals = _recompute_calibration_residuals(models_pkg, df_calib)
    print(f"-> Residuals available: {len(residuals)} metrics, "
          f"calib rows={len(df_calib)}")
    if not residuals:
        raise RuntimeError("No calibration residuals could be recomputed.")

    fpr_target = float(CONFIG["FPR_TARGET_DECISION"])
    results = {}
    for mode in modes:
        if args.chain == "ageing":
            scores = _compute_training_proj_atk_with_ageing(
                models_pkg,
                residuals,
                int(CONFIG.get("WINDOW_SIZE", 10)),
                fusion_mode=mode,
            )
        else:
            scores = tm._compute_training_proj_atk(
                models_pkg,
                residuals,
                int(CONFIG.get("WINDOW_SIZE", 10)),
                fusion_mode=mode,
            )
        if scores is None or len(scores) == 0:
            raise RuntimeError(f"No calibration scores for mode={mode!r}")
        results[mode] = tm._calibrate_decision_threshold_from_scores(
            scores,
            fpr_target=fpr_target,
            source_label="calib hors-train (existing model replay)",
            mode_label=mode,
        )

    active = str(CONFIG.get("INTER_METHOD_FUSION", modes[0])).strip().lower()
    if active not in results:
        active = modes[0]
    for mode, result in results.items():
        _write_sidecar(
            mode, result, model_path, generic=False, modes=modes,
            models_pkg=models_pkg, calibration_chain=args.chain,
        )
    if args.write_generic:
        _write_sidecar(
            active, results[active], model_path, generic=True, modes=modes,
            models_pkg=models_pkg, calibration_chain=args.chain,
        )

    if args.update_model:
        models_pkg["_decision_threshold"] = results[active]["decision_threshold"]
        models_pkg["_decision_variable"] = "proj_atk"
        models_pkg["_decision_threshold_by_fusion_mode"] = {
            mode: result["decision_threshold"] for mode, result in results.items()
        }
        models_pkg["_threshold_calibration_results"] = results
        joblib.dump(models_pkg, model_path)
        print(f"[OK] Model package updated: {model_path}")


if __name__ == "__main__":
    main()
