"""
ablation_nan_ffill.py — NAN_FFILL_LIMIT sensitivity sweep (PATCH m-07 / F25)
===========================================================================

Purpose
-------
Re-run the full SL-ADS pipeline for a set of ``NAN_FFILL_LIMIT`` values
and collect the resulting detection metrics so that the reviewer can
judge how sensitive the paper's headline numbers are to the
forward-fill policy.

Why a separate script?
----------------------
``NAN_FFILL_LIMIT`` affects **training** (``train_v10.py`` computes
Prophet residuals AFTER preprocess_metrics) AND **inference**
(``compute_evidence_v2.py`` applies the same filter).  Changing it
therefore requires rerunning train → evidence → opinions → evaluate in
order to get valid numbers — this is out of scope for
``run_ablation_v2.py`` which only replays post-opinion fusion.

Mechanism
---------
For each candidate ``limit`` in ``LIMITS_TO_SWEEP``:

1. Set env var ``SL_NAN_FFILL_LIMIT_OVERRIDE`` so that ``config.py``
   (tail override block, PATCH m-07/F25) injects the value into
   ``CONFIG["NAN_FFILL_LIMIT"]`` before anything else runs.
2. Invoke ``run_full_sl_ads.py`` with ``--from-step train --to-step
   eval_injection`` via ``subprocess.run``.
3. Parse the ``pipeline_run_summary.json`` and the MANIFEST.md append
   (PATCH m-06/F23 already captures F1/precision/recall/TTD) for the
   run.
4. Aggregate each result row into ``ablation_nan_ffill_summary.csv``.

The generated summary is easy to ingest in the publication: one row
per value of the sweep, columns = the same metrics reported for the
reference run.

Defaults
--------
``LIMITS_TO_SWEEP = [0, 5, 10, 20, 30]`` covers the realistic operating
range:
* 0  — strict "no fill" policy (worst-case, every missing slice → u=1)
* 5  — tight (~2.5 min @ 30s freq) — only micro-dropouts
* 10 — reference value used in the paper
* 20 — loose  (~10 min @ 30s freq) — typical for noisy sensors
* 30 — ceiling (~15 min) before ffill starts hiding real anomalies

USAGE
-----
    python ablation_nan_ffill.py                          # default sweep
    python ablation_nan_ffill.py --limits 5 10 20          # custom sweep
    python ablation_nan_ffill.py --dry-run                # print commands only

Note
----
Each iteration retrains the models, so the full sweep is expensive
(~N × single-run time).  Run on a workstation overnight; the summary
CSV is produced incrementally so even a partial sweep yields a valid
(shorter) table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent
LAUNCHER = BASE_DIR / "run_full_sl_ads.py"
SUMMARY_JSON = BASE_DIR / "pipeline_run_summary.json"
MANIFEST_MD = BASE_DIR / "MANIFEST.md"
OUT_CSV = BASE_DIR / "ablation_nan_ffill_summary.csv"

DEFAULT_LIMITS: List[int] = [0, 5, 10, 20, 30]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NAN_FFILL_LIMIT ablation sweep (PATCH m-07/F25).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--limits", type=int, nargs="+", default=DEFAULT_LIMITS,
        help=f"Space-separated list of NAN_FFILL_LIMIT values to sweep "
             f"(default: {DEFAULT_LIMITS}).",
    )
    p.add_argument(
        "--from-step", default="train",
        help="Pipeline start step (default: train).",
    )
    p.add_argument(
        "--to-step", default="eval_injection",
        help="Pipeline end step (default: eval_injection).",
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


def _load_summary() -> Optional[dict]:
    """Return the dict in pipeline_run_summary.json, or None if missing."""
    if not SUMMARY_JSON.exists():
        return None
    try:
        return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"   [WARN] Could not parse {SUMMARY_JSON.name}: {exc!r}")
        return None


def _extract_last_manifest_metrics() -> dict:
    """Parse the *last* run block in MANIFEST.md and return its metrics.

    This is intentionally tolerant: any missing row → NaN.  The
    manifest format is stable (tables with ``| Metric | Value |``
    header) so a light regex does the job without pandas.
    """
    if not MANIFEST_MD.exists():
        return {}
    txt = MANIFEST_MD.read_text(encoding="utf-8", errors="replace")
    # Take everything after the LAST "## " heading.
    last_sep = txt.rfind("\n## ")
    if last_sep < 0:
        return {}
    tail = txt[last_sep:]
    out: dict = {}
    for line in tail.splitlines():
        # Match rows like "| F1 — binary | `0.872` |"
        if line.startswith("| ") and "| `" in line and line.rstrip().endswith("` |"):
            parts = [s.strip() for s in line.strip("|").split("|")]
            if len(parts) >= 2:
                key = parts[0]
                # Value is wrapped in backticks
                val = parts[1].strip("`")
                out[key] = val
    return out


def _append_csv_row(path: Path, row: dict, header: List[str]) -> None:
    """Append one row (creating the file with header if absent)."""
    import csv
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


def _run_one(limit: int, from_step: str, to_step: str,
              dry_run: bool, dataset: Optional[str]) -> dict:
    """Run the pipeline once for a given NAN_FFILL_LIMIT value."""
    env = os.environ.copy()
    env["SL_NAN_FFILL_LIMIT_OVERRIDE"] = str(limit)
    env["PYTHONUNBUFFERED"] = "1"
    cmd: List[str] = [sys.executable, str(LAUNCHER),
                      "--from-step", from_step, "--to-step", to_step]
    if dataset:
        cmd.extend(["--dataset", dataset])
    print(f"\n{'='*72}")
    print(f"  NAN_FFILL_LIMIT = {limit}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*72}")
    if dry_run:
        return {"NAN_FFILL_LIMIT": limit, "status": "dry-run"}
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(BASE_DIR))
    elapsed = int(time.time() - t0)
    summary = _load_summary() or {}
    manifest_metrics = _extract_last_manifest_metrics()
    return {
        "NAN_FFILL_LIMIT":          limit,
        "returncode":               proc.returncode,
        "duration_s":               elapsed,
        "pipeline_counts":          json.dumps(summary.get("counts", {})),
        "F1 — binary":              manifest_metrics.get("F1 — binary", ""),
        "F1 — coverage-weighted":   manifest_metrics.get("F1 — coverage-weighted", ""),
        "Precision (window)":       manifest_metrics.get("Precision (window)", ""),
        "Recall — binary":          manifest_metrics.get("Recall — binary", ""),
        "Recall — coverage":        manifest_metrics.get("Recall — coverage", ""),
        "FPR (%)":                  manifest_metrics.get("FPR (%)", ""),
        "MCC":                      manifest_metrics.get("MCC", ""),
        "Median TTD (min)":         manifest_metrics.get("Median TTD (min)", ""),
        "Detected attacks":         manifest_metrics.get("Detected attacks", ""),
        "Total attacks":            manifest_metrics.get("Total attacks", ""),
    }


def main() -> int:
    args = _parse_args()
    header = [
        "NAN_FFILL_LIMIT", "returncode", "duration_s", "pipeline_counts",
        "F1 — binary", "F1 — coverage-weighted",
        "Precision (window)", "Recall — binary", "Recall — coverage",
        "FPR (%)", "MCC", "Median TTD (min)",
        "Detected attacks", "Total attacks",
    ]
    print(f"\n[ABL] NAN_FFILL_LIMIT sweep — "
          f"limits={args.limits}  from={args.from_step}  to={args.to_step}  "
          f"dry_run={args.dry_run}")
    for lim in args.limits:
        row = _run_one(
            limit=lim, from_step=args.from_step,
            to_step=args.to_step, dry_run=args.dry_run,
            dataset=args.dataset,
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
