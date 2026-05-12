"""
ablation_fusion_mode.py — INTER_METHOD_FUSION sensitivity (all implemented modes)
================================================================================
PATCH M-11 / fusion-dependence audit.

Purpose
-------
Re-run the opinions + evaluation stages of the SL-ADS pipeline for the
supported inter-method fusion modes (``wbf``, ``abf``, ``hierarchical``,
``cbf``, ``bcf``, ``ccf``, ``minbf``, ``maxbf``) and collect the headline
detection metrics so that the reviewer can judge how sensitive the paper's
results are to the fusion-mode choice.

Why this ablation is mandatory
------------------------------
The paper's early fusion mode was CBF (Cumulative Belief Fusion).  CBF
accumulates evidence and is sensitive to dependent sources.  In this
codebase, the Prophet and Reconstruction branches are derived from the
same raw traffic window; the real RedeRio residual audit now shows
substantial dependence.  Therefore the reference pipeline uses WBF as a
conservative production default, ABF is tested as the literature-aligned
candidate for dependent sources, and CBF/BCF/CCF/MinBF/MaxBF remain
explicit ablation or research modes.

This script produces the fusion-operator ablation table requested by
M-11 and the follow-up SL operator audit.  The paper-side companion
section should be titled along the lines of "Dependence audit between
Prophet and reconstruction branches" and should report:
  (a) empirical residual correlations on normal windows from
      ``sl_ads.stats.residual_correlation``;
  (b) the inter-method fusion sensitivity table from this script.

Mechanism
---------
For each fusion mode in ``MODES_TO_SWEEP``:

1. Set env var ``SL_INTER_METHOD_FUSION_OVERRIDE`` so that ``config.py``
   (tail override block, PATCH M-11/CBF) injects the mode into
   ``CONFIG["INTER_METHOD_FUSION"]`` before any subprocess runs.
   The script also sets
   ``SL_ALLOW_THRESHOLD_FUSION_MISMATCH_FOR_ABLATION=1`` so the same
   calibrated threshold can be used as a fixed reference across modes.
2. Invoke ``run_pipeline.py --from-step opinions --to-step
   eval_injection`` (the fusion mode only affects opinion fusion and
   downstream evaluation; training & evidence are reusable).
3. Parse ``pipeline_run_summary.json`` and the last run block of
   ``MANIFEST.md`` (PATCH m-06/F23 captures F1/precision/recall/TTD).
4. Append one row per mode to ``ablation_fusion_mode_summary.csv``.

Defaults
--------
``MODES_TO_SWEEP = ["wbf", "abf", "hierarchical", "cbf", "bcf", "ccf",
"minbf", "maxbf"]`` — all implemented modes in deterministic order
(WBF default first, so the CSV row-0 matches the reference run).

USAGE
-----
    python ablation_fusion_mode.py                         # default sweep
    python ablation_fusion_mode.py --modes wbf abf ccf      # subset
    python ablation_fusion_mode.py --dry-run                # print cmds only

Notes
-----
* Each iteration re-runs only the ``opinions → eval_injection`` chain
  (a few minutes), not the full training — cheap enough to run
  systematically before each paper revision.
* The CSV is written incrementally so a partial sweep still yields a
  valid (shorter) table.
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

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[2]
LAUNCHER = PROJECT_ROOT / "run_pipeline.py"
SUMMARY_JSON = PROJECT_ROOT / "pipeline_run_summary.json"
MANIFEST_MD = PROJECT_ROOT / "src" / "sl_ads" / "evaluate" / "MANIFEST.md"
OUT_CSV = PROJECT_ROOT / "results" / "ablation_fusion_mode_summary.csv"

VALID_MODES = ("wbf", "abf", "hierarchical", "cbf", "bcf", "ccf", "minbf", "maxbf")
# Default order: WBF first (M-11 default), ABF second (main theoretical
# candidate), then equal-method averaging and legacy/extreme ablations.
DEFAULT_MODES: List[str] = list(VALID_MODES)


# ──────────────────────────────────────────────────────────────────────
# argparse
# ──────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="INTER_METHOD_FUSION ablation sweep "
                    "(PATCH M-11/CBF-indep).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--modes", nargs="+", default=DEFAULT_MODES,
        choices=list(VALID_MODES),
        help=f"Fusion modes to sweep (default: {DEFAULT_MODES}).  "
             f"Must be a subset of {list(VALID_MODES)}.",
    )
    p.add_argument(
        "--from-step", default="opinions",
        help="Pipeline start step (default: opinions — cheapest valid "
             "starting point for a fusion-mode sweep; training and "
             "evidence are invariant w.r.t. INTER_METHOD_FUSION).",
    )
    p.add_argument(
        "--to-step", default="eval_injection",
        help="Pipeline end step (default: eval_injection — produces the "
             "MANIFEST.md row consumed by this script).",
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
# Manifest / summary extraction (mirrors ablation_nan_ffill.py).
# ──────────────────────────────────────────────────────────────────────
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
    Tolerant to missing rows (→ NaN); relies on the stable
    ``| Metric | Value |`` table format (PATCH m-06/F23)."""
    if not MANIFEST_MD.exists():
        return {}
    txt = MANIFEST_MD.read_text(encoding="utf-8", errors="replace")
    last_sep = txt.rfind("\n## ")
    if last_sep < 0:
        return {}
    tail = txt[last_sep:]
    out: dict = {}
    for line in tail.splitlines():
        if (line.startswith("| ") and "| `" in line
                and line.rstrip().endswith("` |")):
            parts = [s.strip() for s in line.strip("|").split("|")]
            if len(parts) >= 2:
                key = parts[0]
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


# ──────────────────────────────────────────────────────────────────────
# Per-mode runner
# ──────────────────────────────────────────────────────────────────────
def _run_one(mode: str, from_step: str, to_step: str,
              dry_run: bool, dataset: Optional[str]) -> dict:
    """Run the sub-pipeline once for the given INTER_METHOD_FUSION mode."""
    env = os.environ.copy()
    env["SL_INTER_METHOD_FUSION_OVERRIDE"] = mode
    env["SL_ALLOW_THRESHOLD_FUSION_MISMATCH_FOR_ABLATION"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    cmd: List[str] = [sys.executable, str(LAUNCHER),
                      "--from-step", from_step, "--to-step", to_step]
    if dataset:
        cmd.extend(["--dataset", dataset])
    print(f"\n{'=' * 72}")
    print(f"  INTER_METHOD_FUSION = {mode}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'=' * 72}")
    if dry_run:
        return {"INTER_METHOD_FUSION": mode, "status": "dry-run"}
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    elapsed = int(time.time() - t0)
    summary = _load_summary() or {}
    m = _extract_last_manifest_metrics() if proc.returncode == 0 else {}
    return {
        "INTER_METHOD_FUSION":      mode,
        "returncode":               proc.returncode,
        "duration_s":               elapsed,
        "pipeline_counts":          json.dumps(summary.get("counts", {})),
        "F1 — binary":              m.get("F1 — binary", ""),
        "F1 — coverage-weighted":   m.get("F1 — coverage-weighted", ""),
        "Precision (window)":       m.get("Precision (window)", ""),
        "Recall — binary":          m.get("Recall — binary", ""),
        "Recall — coverage":        m.get("Recall — coverage", ""),
        "FPR (%)":                  m.get("FPR (%)", ""),
        "MCC":                      m.get("MCC", ""),
        "Median TTD (min)":         m.get("Median TTD (min)", ""),
        "Detected attacks":         m.get("Detected attacks", ""),
        "Total attacks":            m.get("Total attacks", ""),
    }


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    args = _parse_args()
    header = [
        "INTER_METHOD_FUSION", "returncode", "duration_s", "pipeline_counts",
        "F1 — binary", "F1 — coverage-weighted",
        "Precision (window)", "Recall — binary", "Recall — coverage",
        "FPR (%)", "MCC", "Median TTD (min)",
        "Detected attacks", "Total attacks",
    ]
    print(f"\n[ABL] INTER_METHOD_FUSION sweep — "
          f"modes={args.modes}  from={args.from_step}  to={args.to_step}  "
          f"dry_run={args.dry_run}")
    for mode in args.modes:
        row = _run_one(
            mode=mode, from_step=args.from_step,
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
