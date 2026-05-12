"""EVT declustering ablation harness.

Two execution modes:

* ``--mode lightweight`` (default) — load the persisted Prophet/Reconst
  residuals (raw_data CSV) and re-run only the EVT/POT step under
  ``EVT_DECLUSTER_RUN`` in ``{-1, 1, 3, 5}``.  This isolates the
  declustering knob without re-fitting Prophet/QR (cheap, < 1 min).

* ``--mode full`` — full pipeline retraining + re-evaluation per
  ``EVT_DECLUSTER_RUN`` value via subprocesses.  Expensive; produces a
  side-by-side detection report rather than just threshold deltas.

The audit (A1.3) only asks "does declustering meaningfully change the
EVT thresholds?".  The lightweight mode answers that question with
quantitative threshold deltas; the full mode is reserved for cases
where the deltas are large enough to demand a downstream re-evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sl_ads.config import CONFIG
from sl_ads.train.train_models import _evt_threshold_pair


DEFAULT_RUNS = [-1, 1, 3, 5]
DEFAULT_COMMANDS = [
    [sys.executable, "-m", "sl_ads.train.train_models"],
    [sys.executable, "-m", "sl_ads.train.compute_evidence"],
    [sys.executable, "-m", "sl_ads.inject.evidence_level"],
    [sys.executable, "-m", "sl_ads.notebooks.compute_opinions"],
    [sys.executable, "-m", "sl_ads.qualify.sbn_qualifier"],
    [sys.executable, "-m", "sl_ads.evaluate.evaluate_injection"],
]


def _suffix_for(run: int) -> str:
    return f"_evt_decl_{'off' if run < 0 else run}"


def build_plan(runs: list[int]) -> list[dict]:
    plan = []
    for run in runs:
        plan.append({
            "EVT_DECLUSTER_RUN": run,
            "SL_EVT_DECLUSTER_RUN_OVERRIDE": str(run),
            "SL_VERSION_SUFFIX": _suffix_for(run),
            "commands": [" ".join(cmd) for cmd in DEFAULT_COMMANDS],
        })
    return plan


def execute_plan(plan: list[dict]) -> list[dict]:
    results = []
    for item in plan:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path("src").resolve())
        env["SL_EVT_DECLUSTER_RUN_OVERRIDE"] = item["SL_EVT_DECLUSTER_RUN_OVERRIDE"]
        env["SL_VERSION_SUFFIX"] = item["SL_VERSION_SUFFIX"]
        for command_str, cmd in zip(item["commands"], DEFAULT_COMMANDS):
            print(f"[RUN][decl={item['EVT_DECLUSTER_RUN']}] {command_str}")
            proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
            results.append({
                "EVT_DECLUSTER_RUN": item["EVT_DECLUSTER_RUN"],
                "command": command_str,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
            })
            if proc.returncode != 0:
                print(proc.stdout[-2000:])
                print(proc.stderr[-2000:])
                raise SystemExit(proc.returncode)
    return results


# ----------------------------------------------------------------------
# Lightweight mode: re-derive EVT thresholds from persisted residuals.
# ----------------------------------------------------------------------
def _load_residuals_from_models_pkl(pkl_path: Path
                                     ) -> tuple[dict[str, np.ndarray], dict]:
    """Preferred path: read calibration residuals from the trained PKL.

    Requires train_models.py to have been run with the 2026-05-06+ change
    that persists ``models_pkg['_calib_signed_residuals']``. Returns
    ``(residuals_dict, info)``; raises if the key is missing.
    """
    import joblib
    pkg = joblib.load(pkl_path)
    raw = pkg.get('_calib_signed_residuals')
    if raw is None or not raw:
        raise KeyError(
            "_calib_signed_residuals not found in PKL — retrain with the "
            "2026-05-06+ train_models.py to enable production-grade "
            "EVT declustering ablation."
        )
    out = {k: np.abs(np.asarray(v, dtype=float)) for k, v in raw.items()}
    info = {
        "source": "trained_models_pkl_calibration_residuals",
        "pkl_path": str(pkl_path),
        "n_metrics": len(out),
        "metrics": list(out.keys()),
    }
    return out, info


def _load_residuals_for_evt(raw_csv: Path,
                             attack_catalog: list[dict] | None = None,
                             ) -> tuple[dict[str, np.ndarray], dict]:
    """Load |residual| arrays per metric, excluding catalogued attack windows.

    The persisted raw_data CSV covers the inference span (split_date+).
    For the EVT-sensitivity question we want a *clean* normal-traffic
    sample: we therefore strip any timestamp inside an injected or real
    attack catalog window.  This is a proxy for the train-calib span
    used in production calibration; the per-metric thresholds will
    differ in absolute level, but their *relative* response to the
    declustering knob is the load-bearing audit signal.
    """
    df = pd.read_csv(raw_csv, parse_dates=["timestamp"])
    if df.empty:
        raise RuntimeError(f"{raw_csv} is empty.")
    if attack_catalog is None:
        # Lazy import — avoids hard dependency in unit-test contexts.
        try:
            from sl_ads.config import REAL_ATTACKS
            from sl_ads.inject.evidence_level import ATTACK_CATALOG
        except Exception:
            ATTACK_CATALOG = []
            REAL_ATTACKS = {}
        attack_catalog = list(ATTACK_CATALOG or [])
        attack_catalog.extend(
            CONFIG.get("EVAL", {}).get("REAL_ATTACK_CATALOG", []) or []
        )
        for evs in (REAL_ATTACKS or {}).values():
            attack_catalog.extend(evs or [])
    excluded = pd.Series(False, index=df.index)
    for ev in attack_catalog:
        t0 = pd.Timestamp(ev["start"])
        if ev.get("end") is not None:
            t1 = pd.Timestamp(ev["end"])
        elif "duration_h" in ev:
            t1 = t0 + pd.Timedelta(hours=float(ev["duration_h"]))
        else:
            continue
        excluded |= (df["timestamp"] >= t0) & (df["timestamp"] < t1)
    df_clean = df[~excluded]
    out: dict[str, np.ndarray] = {}
    for metric, sub in df_clean.groupby("metric_key"):
        residuals = sub["abs_error"].dropna().to_numpy(dtype=float)
        out[str(metric)] = residuals
    info = {
        "raw_csv": str(raw_csv),
        "n_total_rows": int(len(df)),
        "n_excluded_attack_rows": int(excluded.sum()),
        "n_clean_rows": int((~excluded).sum()),
        "metrics": list(out.keys()),
        "calibration_proxy_caveat": (
            "raw_data CSV only covers the inference span (post split_date). "
            "The lightweight EVT ablation therefore uses the inference-span "
            "residuals (catalog excluded) as a proxy for the train-calib span. "
            "The percentage deltas in T_susp/T_atk vs the no-decluster baseline "
            "are interpretable as declustering sensitivity; their absolute "
            "values are NOT the production thresholds."
        ),
    }
    return out, info


def _direction_from_metric(metric_key: str) -> str:
    """Match the production direction policy in the trained PKL.

    For the lightweight mode we only need to choose a direction tag
    accepted by ``_evt_threshold_pair``.  The metric-key prefix tells us
    whether the original code applies pos/neg/both/sym, but for the EVT
    sensitivity question we use ``sym`` (absolute residuals) consistently
    across all knobs — that is the apples-to-apples view.
    """
    return "sym"


def lightweight_threshold_grid(residuals: dict[str, np.ndarray],
                                runs: list[int]) -> pd.DataFrame:
    rows = []
    q_susp_p = CONFIG.get("EVT_Q_SUSP_PROPHET", 0.02)
    q_atk_p = CONFIG.get("EVT_Q_ATK_PROPHET", 0.01)
    q_susp_r = CONFIG.get("EVT_Q_SUSP_RANSAC", 0.01)
    q_atk_r = CONFIG.get("EVT_Q_ATK_RANSAC", 0.001)
    safety = CONFIG.get("THRESHOLD_SAFETY_MARGIN", 1.10)
    saved_run = CONFIG.get("EVT_DECLUSTER_RUN", -1)
    try:
        for metric, res in sorted(residuals.items()):
            if res.size == 0:
                continue
            is_reconst = metric.startswith("reconst_")
            q_susp = q_susp_r if is_reconst else q_susp_p
            q_atk = q_atk_r if is_reconst else q_atk_p
            branch = "ransac" if is_reconst else "prophet"
            for run in runs:
                CONFIG["EVT_DECLUSTER_RUN"] = run
                t_susp, t_atk = _evt_threshold_pair(
                    res,
                    q_susp=q_susp,
                    q_atk=q_atk,
                    safety_margin=safety,
                    metric_key=metric,
                    branch=branch,
                )
                rows.append({
                    "metric": metric,
                    "EVT_DECLUSTER_RUN": run,
                    "n_residuals": int(res.size),
                    "t_susp": float(t_susp),
                    "t_atk": float(t_atk),
                })
    finally:
        CONFIG["EVT_DECLUSTER_RUN"] = saved_run
    return pd.DataFrame(rows)


def summarise_grid(grid: pd.DataFrame) -> pd.DataFrame:
    """Compute per-metric % change in T_susp/T_atk vs. the baseline (run<0)."""
    baseline = grid[grid["EVT_DECLUSTER_RUN"] < 0].set_index("metric")
    out_rows = []
    for (metric, run), sub in grid.groupby(["metric", "EVT_DECLUSTER_RUN"]):
        if metric not in baseline.index:
            continue
        b = baseline.loc[metric]
        row = sub.iloc[0]
        out_rows.append({
            "metric": metric,
            "EVT_DECLUSTER_RUN": run,
            "delta_t_susp_pct": (row["t_susp"] / b["t_susp"] - 1.0) * 100.0
                if b["t_susp"] > 0 else float("nan"),
            "delta_t_atk_pct": (row["t_atk"] / b["t_atk"] - 1.0) * 100.0
                if b["t_atk"] > 0 else float("nan"),
        })
    return pd.DataFrame(out_rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["lightweight", "full"], default="lightweight")
    p.add_argument("--runs", default="-1,1,3,5")
    p.add_argument(
        "--raw-csv",
        default=None,
        help="raw_data CSV produced by compute_evidence (lightweight mode).",
    )
    p.add_argument("--out-dir", default="outputs/scientific_hardening")
    p.add_argument("--execute", action="store_true",
                   help="Full mode only: actually run the subprocesses.")
    args = p.parse_args()

    runs = [int(x) for x in args.runs.split(",") if x]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "full":
        plan = build_plan(runs)
        plan_path = out_dir / "evt_declustering_ablation_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"[OK] wrote {plan_path}")
        if not args.execute:
            print(json.dumps(plan, indent=2))
            print("[DRY-RUN] pass --execute to retrain/re-evaluate all declustering variants.")
            return 0
        results = execute_plan(plan)
        results_path = out_dir / "evt_declustering_ablation_execution.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[OK] wrote {results_path}")
        return 0

    # lightweight mode
    # Preferred: read calibration residuals from the trained PKL.
    # Fallback: reconstruct from raw_data CSV (catalog excluded).
    info: dict
    residuals: dict[str, np.ndarray]
    pkl_path = Path(args.raw_csv) if args.raw_csv and args.raw_csv.endswith(
        ".pkl") else _default_models_pkl()
    if pkl_path is not None and pkl_path.exists():
        try:
            residuals, info = _load_residuals_from_models_pkl(pkl_path)
            print(f"[INFO] loaded calibration residuals from {pkl_path} "
                  f"({len(residuals)} metrics).")
        except (KeyError, Exception) as exc:
            print(f"[INFO] PKL fallback ({exc}) — using raw_data proxy.")
            raw_csv = Path(args.raw_csv) if args.raw_csv else _default_raw_csv()
            residuals, info = _load_residuals_for_evt(raw_csv)
    else:
        raw_csv = Path(args.raw_csv) if args.raw_csv else _default_raw_csv()
        print(f"[INFO] reading residuals from {raw_csv}")
        residuals, info = _load_residuals_for_evt(raw_csv)
        print(f"[INFO] {len(residuals)} metric residual series loaded "
              f"(clean rows: {info['n_clean_rows']} / total: {info['n_total_rows']}).")
    grid = lightweight_threshold_grid(residuals, runs)
    deltas = summarise_grid(grid)
    grid_path = out_dir / "evt_declustering_thresholds.csv"
    deltas_path = out_dir / "evt_declustering_thresholds_delta_pct.csv"
    grid.to_csv(grid_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    print(f"[OK] wrote {grid_path}")
    print(f"[OK] wrote {deltas_path}")
    print("\n--- Threshold deltas (% vs EVT_DECLUSTER_RUN<0 baseline) ---")
    print(deltas.to_string(index=False))
    summary = {
        "mode": "lightweight",
        "runs": runs,
        "n_metrics": int(grid["metric"].nunique()),
        "max_abs_delta_t_susp_pct": float(deltas["delta_t_susp_pct"].abs().max())
            if not deltas.empty else None,
        "max_abs_delta_t_atk_pct": float(deltas["delta_t_atk_pct"].abs().max())
            if not deltas.empty else None,
        "info": info,
    }
    (out_dir / "evt_declustering_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


def _default_models_pkl() -> Path | None:
    """Locate the trained PKL with calibration residuals (if present)."""
    from sl_ads.paths import get_model_path
    try:
        path = Path(get_model_path(CONFIG, up_levels=1))
    except Exception:
        return None
    return path if path.exists() else None


def _default_raw_csv() -> Path:
    candidates = [
        Path("outputs/raw_data_RedeRio_trained_v4s_v4_v2.csv"),
        Path(CONFIG.get("RESULTS_DIR", "")) / "raw_data_RedeRio_trained_v4s_v4_v2.csv",
    ]
    name = (CONFIG.get("VERSION_NAME") or "")
    if name:
        candidates.append(Path("outputs") / f"raw_data_{name}.csv")
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No raw_data_*.csv found; pass --raw-csv.")


if __name__ == "__main__":
    raise SystemExit(main())
