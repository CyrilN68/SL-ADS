"""
ablation_sbn_novelty.py — SBN_NOVELTY_U_RAW_THRESHOLD sensitivity sweep
======================================================================
PATCH M-07 / F10.

Purpose
-------
Re-run the qualification + evaluation stages of the SL-ADS pipeline for
a set of ``SBN_NOVELTY_U_RAW_THRESHOLD`` values and collect the
resulting qualification metrics so that the reviewer can judge how
sensitive the paper's novelty-detection claims are to this threshold.

Why a separate script?
----------------------
``SBN_NOVELTY_U_RAW_THRESHOLD`` only affects the *qualification* stage
(``qualify_anomaly_sbn.py`` — the rule
``qual_status=='autre_anomalie'  iff  u_raw > threshold``).  It does
**not** affect training, evidence computation, opinion fusion, or raw
detection: the upstream artefacts (opinions CSV, injected evidence) can
be reused across the whole sweep.  Therefore:

1. We restrict the pipeline to ``--from-step qualify_sbn --to-step
   eval_qualify``.  Each iteration takes ~1-2 min instead of ~15-25 min
   for a full re-run, making the sweep cheap (≈ 5-10 min for 5 values).
2. The override is injected via the env var
   ``SL_SBN_NOVELTY_U_RAW_THRESHOLD_OVERRIDE``; the tail block in
   ``config.py`` (PATCH M-07/F10) reads it and sets
   ``CONFIG["SBN_NOVELTY_U_RAW_THRESHOLD"]`` before any subprocess runs.
3. After each run, we read the latest
   ``eval_qualify_summary_*.json`` from ``RESULTS_DIR`` and extract
   macro/micro DR / QP / F1 / F2, plus novelty-control signals
   (``lr_mean``, ``novelty_detected`` per control attack) and count
   the number of ``autre_anomalie`` rows from the qualification CSV.

Why these thresholds?
---------------------
``DEFAULT_THRESHOLDS = [0.70, 0.75, 0.82, 0.85, 0.90]`` spans the
useful range:
* 0.70 — aggressive  (≈ 3.7 evidence units triggers novelty)
* 0.75 — moderate    (≈ 3.0 evidence units)
* 0.82 — REFERENCE value used in the paper (≈ 2.4 evidence units)
* 0.85 — conservative (≈ 1.9 evidence units)
* 0.90 — very conservative (≈ 1.2 evidence units — near "no signal")

Reviewer audit directive (CONSOLIDATED_AUDIT_REVIEW.md, M-07/F10):
    "Ajouter au moins un test de sensibilité : varier
    SBN_NOVELTY_U_RAW_THRESHOLD ∈ {0.70, 0.75, 0.82, 0.85, 0.90} et
    reporter dans un tableau annexe."

USAGE
-----
    python ablation_sbn_novelty.py                         # default sweep
    python ablation_sbn_novelty.py --thresholds 0.82 0.90  # custom sweep
    python ablation_sbn_novelty.py --dry-run               # print cmds only
    python ablation_sbn_novelty.py --dataset METR-LA       # override dataset

Output
------
* ``ablation_sbn_novelty_summary.csv`` — one row per threshold with
  macro/micro metrics and novelty-control counts.  Written
  incrementally so a partial sweep still yields a usable (shorter)
  table.

Notes
-----
* The iteration is non-destructive against models and upstream
  artefacts — only the qualify CSV / JSON are rewritten each step.
  Because the qualify step uses a fixed timestamped output filename,
  each run produces a new JSON that we pick up via "latest-by-mtime".
* To keep the comparison fair between thresholds the ``eval_qualify``
  step uses the SAME opinions CSV for all runs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

BASE_DIR = Path(__file__).resolve().parent
LAUNCHER = BASE_DIR / "run_full_sl_ads.py"
OUT_CSV = BASE_DIR / "ablation_sbn_novelty_summary.csv"

# Reviewer-directed sweep (M-07/F10).
DEFAULT_THRESHOLDS: List[float] = [0.70, 0.75, 0.82, 0.85, 0.90]


# ──────────────────────────────────────────────────────────────────────
# argparse
# ──────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SBN_NOVELTY_U_RAW_THRESHOLD ablation sweep "
                    "(PATCH M-07/F10).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--thresholds", type=float, nargs="+",
        default=DEFAULT_THRESHOLDS,
        help=f"Space-separated list of SBN_NOVELTY_U_RAW_THRESHOLD "
             f"values to sweep (default: {DEFAULT_THRESHOLDS}).  "
             f"All values must be in [0.0, 1.0].",
    )
    p.add_argument(
        "--from-step", default="qualify_sbn",
        help="Pipeline start step (default: qualify_sbn — cheapest valid "
             "starting point for a novelty-threshold sweep).",
    )
    p.add_argument(
        "--to-step", default="eval_qualify",
        help="Pipeline end step (default: eval_qualify — produces the "
             "qualification metrics JSON consumed by this script).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands without executing anything.",
    )
    p.add_argument(
        "--out-csv", type=Path, default=OUT_CSV,
        help=f"Summary CSV path (default: {OUT_CSV.name}).",
    )
    p.add_argument(
        "--dataset", default=None,
        help="Override SL_ACTIVE_DATASET (default: from config.py).",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Output discovery: find the latest eval_qualify_summary_*.json that was
# written *after* ``since_ts`` so we pick up exactly the current run's
# JSON (not an older one from a previous sweep iteration).
# ──────────────────────────────────────────────────────────────────────
def _resolve_results_dir(dataset: Optional[str]) -> Path:
    """Return the RESULTS_DIR that evaluate_qualify_sbn.py will write to."""
    # Inherit the dataset hint via env so ``config.py`` picks it up.
    if dataset:
        os.environ["SL_ACTIVE_DATASET"] = dataset
    # Fresh import — config reads the env var at module load.
    import importlib
    import sl_ads.config as _cfg_mod  # Phase H
    importlib.reload(_cfg_mod)
    from sl_ads.paths import get_results_dir  # Phase H
    return Path(get_results_dir(_cfg_mod.CONFIG, up_levels=1))


def _latest_summary_json(results_dir: Path, since_ts: float) -> Optional[Path]:
    """Return the newest ``eval_qualify_summary_*.json`` modified after
    ``since_ts`` (Unix timestamp), or None if nothing matched."""
    if not results_dir.exists():
        return None
    candidates = sorted(results_dir.glob("eval_qualify_summary_*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        if p.stat().st_mtime >= since_ts:
            return p
    return None


def _extract_metrics(summary_path: Path) -> dict:
    """Parse an eval_qualify_summary_*.json and flatten the fields we
    care about for the ablation table.  Missing keys become empty
    strings (so a partial run still writes a row)."""
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"   [WARN] Could not parse {summary_path.name}: {exc!r}")
        return {"summary_parse_error": repr(exc)}

    out: dict = {
        "summary_json":             summary_path.name,
        "macro_DR":                 data.get("macro_DR", ""),
        "macro_QP":                 data.get("macro_QP", ""),
        "macro_F1":                 data.get("macro_F1", ""),
        "macro_F2":                 data.get("macro_F2", ""),
        "micro_DR":                 data.get("micro_DR", ""),
        "micro_QP":                 data.get("micro_QP", ""),
        "micro_F2":                 data.get("micro_F2", ""),
        "n_attack_types_injected":  data.get("n_attack_types_injected", ""),
        "n_attacks_not_qualified":  len(data.get("attacks_not_qualified", [])
                                         or []),
    }

    # Novelty-control aggregation: count how many control (unknown) attacks
    # were flagged as novelty, plus mean lr_mean across controls.  These are
    # the per-scenario signals the threshold is supposed to discriminate.
    nov_list = data.get("novelty_controls", []) or []
    if nov_list:
        n_controls = len(nov_list)
        n_flagged = sum(1 for r in nov_list if bool(r.get("novelty_detected")))
        lr_vals = [r.get("lr_mean") for r in nov_list
                   if isinstance(r.get("lr_mean"), (int, float))]
        out["n_novelty_controls"] = n_controls
        out["n_novelty_flagged"]  = n_flagged
        out["novelty_flag_rate"]  = round(n_flagged / n_controls, 4) \
            if n_controls else ""
        out["mean_lr_controls"]   = round(sum(lr_vals) / len(lr_vals), 4) \
            if lr_vals else ""
    else:
        for k in ("n_novelty_controls", "n_novelty_flagged",
                   "novelty_flag_rate", "mean_lr_controls"):
            out[k] = ""

    # Global detection stats (m-04/F21 after the filter refactor).
    gd = data.get("global_detection", {}) or {}
    out["global_n_windows"]        = gd.get("n_windows", "")
    out["global_n_attack_windows"] = gd.get("n_attack_windows", "")
    out["global_n_outage_windows"] = gd.get("n_outage_windows", "")
    out["global_FAR"]              = gd.get("FAR", "")
    out["global_MCC"]              = gd.get("MCC", "")

    return out


def _count_autre_anomalie(results_dir: Path, since_ts: float) -> Optional[int]:
    """Re-read the latest qualif_types_sbn.csv produced by qualify_sbn
    and count rows with qual_status == 'autre_anomalie'.

    This is the single most threshold-sensitive quantity in the pipeline
    — it is what the sensitivity sweep is supposed to characterise.
    Returns None if the CSV is missing or unreadable."""
    if not results_dir.exists():
        return None
    # qualify_anomaly_sbn.py writes either ``qualif_types_sbn.csv`` or
    # ``qualif_types_sbn_<balance>.csv`` — pick whichever was touched last.
    candidates = sorted(results_dir.glob("qualif_types_sbn*.csv"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        if p.stat().st_mtime < since_ts:
            continue
        try:
            import pandas as _pd
            df = _pd.read_csv(p, usecols=["qual_status"])
            return int((df["qual_status"].astype(str)
                         == "autre_anomalie").sum())
        except Exception as exc:
            print(f"   [WARN] Could not count autre_anomalie in "
                  f"{p.name}: {exc!r}")
            return None
    return None


# ──────────────────────────────────────────────────────────────────────
# Row writing
# ──────────────────────────────────────────────────────────────────────
def _append_csv_row(path: Path, row: dict, header: List[str]) -> None:
    """Append one row (creating the file with header if absent)."""
    import csv
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


# ──────────────────────────────────────────────────────────────────────
# Per-threshold runner
# ──────────────────────────────────────────────────────────────────────
def _run_one(threshold: float, from_step: str, to_step: str,
              dry_run: bool, dataset: Optional[str],
              results_dir: Path) -> dict:
    """Run the qualify_sbn + eval_qualify sub-pipeline once for the
    given SBN_NOVELTY_U_RAW_THRESHOLD value, then aggregate metrics."""
    env = os.environ.copy()
    env["SL_SBN_NOVELTY_U_RAW_THRESHOLD_OVERRIDE"] = f"{threshold:.6f}"
    env["PYTHONUNBUFFERED"] = "1"
    cmd: List[str] = [sys.executable, str(LAUNCHER),
                      "--from-step", from_step, "--to-step", to_step]
    if dataset:
        cmd.extend(["--dataset", dataset])
    print(f"\n{'=' * 72}")
    print(f"  SBN_NOVELTY_U_RAW_THRESHOLD = {threshold:.3f}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'=' * 72}")
    if dry_run:
        return {"SBN_NOVELTY_U_RAW_THRESHOLD": threshold,
                "status": "dry-run"}

    t0 = time.time()
    # Anchor: discovery of summary JSON / qualif CSV must only pick up
    # artefacts NEWER than the start of this iteration so we don't
    # conflate iterations.  Use t0 - 2s for filesystem-mtime slack.
    since_ts = t0 - 2.0
    proc = subprocess.run(cmd, env=env, cwd=str(BASE_DIR))
    elapsed = int(time.time() - t0)

    summary_path = _latest_summary_json(results_dir, since_ts)
    metrics = _extract_metrics(summary_path) if summary_path else {
        "summary_json": "(not found)"}
    n_autre = _count_autre_anomalie(results_dir, since_ts)

    row = {
        "SBN_NOVELTY_U_RAW_THRESHOLD": threshold,
        "returncode":                  proc.returncode,
        "duration_s":                  elapsed,
        "n_autre_anomalie":            n_autre if n_autre is not None else "",
    }
    row.update(metrics)
    return row


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────
def _validate_thresholds(xs: Iterable[float]) -> List[float]:
    """Reject NaN, inf, or values outside [0, 1]."""
    out: List[float] = []
    import math
    for x in xs:
        if (not isinstance(x, (int, float))) or math.isnan(x) \
                or math.isinf(x) or not (0.0 <= x <= 1.0):
            raise SystemExit(
                f"[ABL] Invalid threshold {x!r}: must be finite and in "
                f"[0.0, 1.0].")
        out.append(float(x))
    return out


def main() -> int:
    args = _parse_args()
    thresholds = _validate_thresholds(args.thresholds)

    # The CSV header is fixed and covers both success and (partial) failure
    # rows — missing fields stay empty so the table remains rectangular.
    header = [
        "SBN_NOVELTY_U_RAW_THRESHOLD", "returncode", "duration_s",
        "n_autre_anomalie",
        "macro_DR", "macro_QP", "macro_F1", "macro_F2",
        "micro_DR", "micro_QP", "micro_F2",
        "n_attack_types_injected", "n_attacks_not_qualified",
        "n_novelty_controls", "n_novelty_flagged",
        "novelty_flag_rate", "mean_lr_controls",
        "global_n_windows", "global_n_attack_windows",
        "global_n_outage_windows", "global_FAR", "global_MCC",
        "summary_json",
    ]

    print(f"\n[ABL] SBN_NOVELTY_U_RAW_THRESHOLD sweep — "
          f"thresholds={thresholds}  from={args.from_step}  "
          f"to={args.to_step}  dry_run={args.dry_run}")

    try:
        results_dir = _resolve_results_dir(args.dataset)
        print(f"[ABL] RESULTS_DIR = {results_dir}")
    except Exception as exc:
        print(f"[ABL][WARN] Could not resolve RESULTS_DIR ({exc!r}); "
              f"metric aggregation will be skipped.")
        results_dir = BASE_DIR / "investigations"  # Phase H — was "modèle évaluation"

    for thr in thresholds:
        row = _run_one(
            threshold=thr,
            from_step=args.from_step, to_step=args.to_step,
            dry_run=args.dry_run, dataset=args.dataset,
            results_dir=results_dir,
        )
        if not args.dry_run:
            _append_csv_row(args.out_csv, row, header)
            print(f"   appended to {args.out_csv.name}")

    if not args.dry_run:
        print(f"\n[ABL] Done.  Summary: {args.out_csv}")
    else:
        print(f"\n[ABL] Dry-run complete (no runs executed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
