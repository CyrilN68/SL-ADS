"""Strict WBF-vs-ABF comparison with mode-specific calibrated thresholds.

This script intentionally does not set
``SL_ALLOW_THRESHOLD_FUSION_MISMATCH_FOR_ABLATION``.  Each mode must load its
own ``*_threshold_<mode>.json`` sidecar, otherwise evaluation fails loudly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sl_ads.config import CONFIG  # noqa: E402
from sl_ads.paths import get_results_dir, get_threshold_sidecar_path  # noqa: E402


def _sidecar_path(mode: str) -> Path:
    return Path(get_threshold_sidecar_path(CONFIG, up_levels=1, fusion_mode=mode))


def _snapshot_sidecars(modes: list[str]) -> dict[str, bytes | None]:
    snapshots: dict[str, bytes | None] = {}
    for mode in modes:
        path = _sidecar_path(mode)
        snapshots[mode] = path.read_bytes() if path.is_file() else None
    return snapshots


def _restore_sidecars(snapshots: dict[str, bytes | None]) -> None:
    for mode, payload in snapshots.items():
        path = _sidecar_path(mode)
        if payload is None:
            if path.exists():
                path.unlink()
        else:
            path.write_bytes(payload)


def _snapshot_generated_sidecars(modes: list[str], out_dir: Path) -> None:
    dest = out_dir / "threshold_sidecars"
    dest.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        path = _sidecar_path(mode)
        if path.is_file():
            shutil.copy2(path, dest / path.name)


def _env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.setdefault("SL_SKIP_OPINION_PLOTS", "1")
    env.setdefault("SL_SKIP_EVAL_PLOTS", "1")
    env.pop("SL_ALLOW_THRESHOLD_FUSION_MISMATCH_FOR_ABLATION", None)
    if extra:
        env.update(extra)
    return env


def _run(cmd: list[str], env_extra: dict[str, str] | None = None) -> None:
    print("[strict-fusion]", " ".join(cmd))
    subprocess.run(cmd, cwd=BASE_DIR, env=_env(env_extra), check=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing expected evaluation artifact: {path}")
    return pd.read_csv(path)


def _mode_threshold(mode: str) -> float | None:
    path = _sidecar_path(mode)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload["decision_threshold"])


def _summarise_mode(mode: str, snapshot_dir: Path) -> tuple[dict, pd.DataFrame]:
    eval_dir = Path(get_results_dir(CONFIG, up_levels=1)) / "evaluation"
    sweep = _read_csv(eval_dir / "eval_threshold_sweep.csv")
    det = _read_csv(eval_dir / "eval_detection_summary.csv")
    vus_path = eval_dir / "eval_vus_summary.csv"
    vus = pd.read_csv(vus_path) if vus_path.is_file() else pd.DataFrame()

    snapshot_eval = snapshot_dir / mode / "evaluation"
    if snapshot_eval.exists():
        shutil.rmtree(snapshot_eval)
    shutil.copytree(eval_dir, snapshot_eval)

    row = sweep.iloc[0].to_dict()
    def _first(*names: str):
        for name in names:
            val = row.get(name)
            if val is not None and not pd.isna(val):
                return val
        return None

    threshold = _mode_threshold(mode)
    summary = {
        "mode": mode,
        "threshold": threshold if threshold is not None else row.get("threshold"),
        "precision": _first("precision_window", "precision"),
        "recall": _first("recall_binary", "recall_attack", "recall"),
        "recall_coverage": row.get("recall_coverage"),
        "recall_event": row.get("recall_event"),
        "f1": _first("f1_micro_pure", "f1_score", "f1"),
        "f1_coverage_hybrid": row.get("f1_coverage_hybrid_episode_recall"),
        "mcc": row.get("mcc"),
        "fpr_win": _first("fpr_window", "fpr_win"),
        "fpr_pct": row.get("fpr_pct"),
        "vus_pr": None if vus.empty else vus.iloc[0].get("vus_pr"),
        "vus_roc": None if vus.empty else vus.iloc[0].get("vus_roc"),
        "eval_snapshot": str(snapshot_eval),
    }

    family = (
        det.groupby("family", dropna=False)
        .agg(
            attacks=("name", "count"),
            detected=("detected", "sum"),
            coverage_pct_mean=("coverage_pct", "mean"),
            ttd_minutes_median=("ttd_minutes", "median"),
            fpr_pct_mean=("fpr_pct", "mean"),
        )
        .reset_index()
    )
    family.insert(0, "mode", mode)
    return summary, family


def _recommend(headline: pd.DataFrame) -> str:
    rows = {r["mode"]: r for _, r in headline.iterrows()}
    if "wbf" not in rows or "abf" not in rows:
        return "insufficient_modes"
    wbf, abf = rows["wbf"], rows["abf"]
    def _num(row, key: str, default: float = 0.0) -> float:
        val = row.get(key, default)
        if val is None or pd.isna(val):
            return default
        return float(val)
    f1_ok = _num(abf, "f1") >= _num(wbf, "f1") - 1e-9
    mcc_ok = _num(abf, "mcc") >= _num(wbf, "mcc") - 0.005
    fpr_abf = _num(abf, "fpr_win", _num(abf, "fpr_pct"))
    fpr_wbf = _num(wbf, "fpr_win", _num(wbf, "fpr_pct"))
    fpr_ok = fpr_abf <= fpr_wbf + 1e-6
    return "switch_default_to_abf" if (f1_ok and mcc_ok and fpr_ok) else "keep_default_wbf"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="wbf,abf",
                        help="Comma-separated inter-method fusion modes.")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Reuse existing mode-specific threshold sidecars.")
    parser.add_argument("--full-train", action="store_true",
                        help="Refit models before comparing modes. Slow.")
    parser.add_argument("--from-step", default="opinions",
                        help="Pipeline step to start from after training.")
    args = parser.parse_args()

    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    if not modes:
        raise SystemExit("No modes requested.")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = BASE_DIR / "results" / "fusion_mode_recalibrated" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_snapshots = _snapshot_sidecars(modes)

    try:
        if not args.skip_calibration:
            if args.full_train:
                _run(
                    [sys.executable, "-m", "sl_ads.train.train_models"],
                    {"SL_THRESHOLD_CALIBRATION_FUSION_MODES": ",".join(modes)},
                )
            else:
                _run(
                    [
                        sys.executable,
                        "-m",
                        "sl_ads.ablation.recalibrate_fusion_thresholds",
                        "--modes",
                        ",".join(modes),
                    ],
                    {"SL_THRESHOLD_CALIBRATION_FUSION_MODES": ",".join(modes)},
                )
        _snapshot_generated_sidecars(modes, out_dir)

        headline_rows = []
        family_rows = []
        for mode in modes:
            _run(
                [
                    sys.executable,
                    "run_pipeline.py",
                    "--from-step",
                    args.from_step,
                    "--to-step",
                    "eval_injection",
                    "--no-archive",
                ],
                {"SL_INTER_METHOD_FUSION_OVERRIDE": mode},
            )
            summary, family = _summarise_mode(mode, out_dir)
            headline_rows.append(summary)
            family_rows.append(family)

        headline = pd.DataFrame(headline_rows)
        by_family = pd.concat(family_rows, ignore_index=True) if family_rows else pd.DataFrame()
        recommendation = _recommend(headline)

        headline_path = out_dir / "fusion_mode_recalibrated_headline.csv"
        family_path = out_dir / "fusion_mode_recalibrated_by_family.csv"
        decision_path = out_dir / "fusion_mode_recalibrated_decision.json"
        headline.to_csv(headline_path, index=False)
        by_family.to_csv(family_path, index=False)
        decision_path.write_text(json.dumps({
            "modes": modes,
            "recommendation": recommendation,
            "headline_csv": str(headline_path),
            "by_family_csv": str(family_path),
        }, indent=2), encoding="utf-8")

        print("\nStrict recalibrated fusion comparison complete.")
        print(headline.to_string(index=False))
        print(f"Recommendation: {recommendation}")
        print(f"Artifacts: {out_dir}")
    finally:
        _restore_sidecars(sidecar_snapshots)


if __name__ == "__main__":
    main()
