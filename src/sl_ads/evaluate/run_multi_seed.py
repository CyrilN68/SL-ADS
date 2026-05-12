"""
run_multi_seed.py — Multi-seed evaluation orchestrator and aggregator
(Wu & Keogh 2021, flaw #4: "single-run results").

Purpose
-------
Wu and Keogh (TKDE 2021) identify the absence of multi-seed evaluation
as one of the four critical flaws in current time-series anomaly
detection benchmarks.  This module provides the missing orchestration
layer:

1. **Run mode** (``--mode run``): launch ``run_pipeline.py`` once per
   seed in ``--seeds``, propagating the seed through the
   ``SL_RANDOM_SEED`` environment variable that ``sl_ads.config``
   already honours (see ``config.py`` "PATCH TASK-53" block).
2. **Aggregate mode** (``--mode aggregate``): walk the ``results/``
   directory (or take an explicit ``--run-ids`` list) and compute
   per-metric mean / std / min / max / BCa 95 % CI across seeds.
3. **End-to-end mode** (``--mode all``, default): run then aggregate.

Outputs (always written under ``--output-dir``, default
``results/_multi_seed/<timestamp>``):

* ``multi_seed_per_run.csv``   — one row per seed, per-run snapshot.
* ``multi_seed_aggregate.csv`` — aggregate statistics with CI.
* ``multi_seed_report.md``     — human-readable summary.
* ``multi_seed_manifest.json`` — full provenance (seeds, run_ids,
                                 cwd, python, command line, durations,
                                 git hash if available).

The aggregator is deliberately decoupled from the orchestrator: it can
be re-run on any subset of pre-existing ``results/<run_id>/`` folders
without re-launching the pipeline.

References
----------
- Wu, R., Keogh, E. (2021). "Current Time Series Anomaly Detection
  Benchmarks are Flawed and are Creating the Illusion of Progress."
  *IEEE TKDE* (preprint arXiv:2009.13807).
- Demsar, J. (2006). "Statistical Comparisons of Classifiers over
  Multiple Data Sets." *JMLR* 7:1-30.

CLI
---
Run mode (full pipeline, expensive)::

    python -m sl_ads.evaluate.run_multi_seed --mode run \\
        --seeds 0,1,2,3,4 --dataset RedeRio

Aggregate mode (read existing runs)::

    python -m sl_ads.evaluate.run_multi_seed --mode aggregate \\
        --run-ids da8ab988fddaf681,9c9c2e02bb4ee103,...

Self-test::

    python -m sl_ads.evaluate.run_multi_seed --self-test

Tracks TASK-53 of ``docs/audit/audit_verification_tracker.md``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _utc_now() -> _dt.datetime:
    """Timezone-aware UTC ``now`` (replacement for the deprecated
    ``datetime.utcnow()``)."""
    return _dt.datetime.now(_dt.timezone.utc)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
HEADLINE_METRICS = (
    "F1_micro",
    "F1_macro",
    "MCC",
    "FAR_pct",
    "n_detected",
)

# Mapping from canonical headline name → list of accepted CSV column
# names (the CSV schema has evolved across phases; keep a tolerant
# resolution to ease forward compatibility).
_METRIC_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "F1_micro":  ("f1_micro", "F1_micro", "f1_micro_paper"),
    "F1_macro":  ("f1_macro", "F1_macro"),
    "MCC":       ("mcc", "MCC"),
    "FAR_pct":   ("far_pct", "FAR_pct", "fpr_pct"),
    "n_detected": ("n_detected", "detected_count"),
}


# ──────────────────────────────────────────────────────────────────────
# Orchestration helpers
# ──────────────────────────────────────────────────────────────────────
def _parse_seed_list(s: str) -> List[int]:
    """Parse a comma-separated list of integer seeds."""
    if not s.strip():
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError as exc:
            raise ValueError(f"Invalid seed value: {p!r}") from exc
    if len(out) != len(set(out)):
        raise ValueError(f"Duplicate seeds in {s!r} — would re-run identical pipelines.")
    return out


def _build_pipeline_command(
    dataset: str,
    from_step: Optional[str],
    to_step: Optional[str],
    project_root: Path,
) -> List[str]:
    """Return the ``run_pipeline.py`` invocation as a list (no shell)."""
    cmd = [sys.executable, str(project_root / "run_pipeline.py"), "--dataset", dataset]
    if from_step:
        cmd += ["--from-step", from_step]
    if to_step:
        cmd += ["--to-step", to_step]
    return cmd


def _run_one_seed(
    seed: int,
    dataset: str,
    from_step: Optional[str],
    to_step: Optional[str],
    project_root: Path,
    timeout_s: Optional[int],
    dry_run: bool,
    log_path: Path,
) -> Dict[str, object]:
    """Launch ``run_pipeline.py`` for a single seed and return a record.

    The record contains: seed, returncode, duration_s, run_id (parsed
    from the latest manifest after the run), log_path.
    """
    env = os.environ.copy()
    env["SL_RANDOM_SEED"] = str(seed)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    cmd = _build_pipeline_command(dataset, from_step, to_step, project_root)

    record: Dict[str, object] = {
        "seed": seed,
        "command": " ".join(cmd),
        "env_SL_RANDOM_SEED": str(seed),
        "dry_run": bool(dry_run),
    }

    if dry_run:
        record["returncode"] = None
        record["duration_s"] = 0.0
        record["run_id"] = None
        record["log_path"] = None
        return record

    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(  # noqa: S603 — internal launcher, no shell
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    duration = time.time() - started

    record["returncode"] = int(proc.returncode)
    record["duration_s"] = round(duration, 2)
    record["log_path"] = str(log_path)
    record["run_id"] = _latest_run_id(project_root / "results")
    return record


def _latest_run_id(results_root: Path) -> Optional[str]:
    """Return the directory name of the most recently finished run.

    A run is considered "finished" if ``_run_manifest.json`` exists.
    The manifest's ``finished_utc`` is used to break ties.
    """
    if not results_root.is_dir():
        return None
    candidates: List[Tuple[str, str]] = []  # (finished_utc, run_id)
    for sub in results_root.iterdir():
        if not sub.is_dir():
            continue
        manifest = sub / "_run_manifest.json"
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        finished = data.get("summary", {}).get("finished_utc", "")
        candidates.append((finished, sub.name))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


# ──────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ──────────────────────────────────────────────────────────────────────
def _resolve_metric_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """Return the first matching column name from the alias list."""
    for alias in _METRIC_COLUMN_ALIASES.get(canonical, (canonical,)):
        if alias in df.columns:
            return alias
    return None


def _read_run_metrics(results_root: Path, run_id: str) -> Dict[str, float]:
    """Extract the headline metrics for a given run_id.

    The eval pipeline writes ``evaluation/eval_detection_summary.csv``;
    we pick the row aggregated over all attacks (column ``attack`` ==
    ``__ALL__`` or ``OVERALL`` if present, otherwise the mean across
    rows).  Missing metrics return NaN — never raise — so the
    aggregator stays robust to schema drift.
    """
    out = {m: float("nan") for m in HEADLINE_METRICS}
    csv_path = results_root / run_id / "evaluation" / "eval_detection_summary.csv"
    if not csv_path.is_file():
        return out
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.ParserError, OSError):
        return out
    if df.empty:
        return out

    # Prefer an aggregated row if present.
    aggregated = None
    for tag in ("__ALL__", "OVERALL", "ALL", "overall"):
        if "attack" in df.columns and (df["attack"] == tag).any():
            aggregated = df[df["attack"] == tag].iloc[0]
            break
    for canonical in HEADLINE_METRICS:
        col = _resolve_metric_column(df, canonical)
        if col is None:
            continue
        if aggregated is not None and col in aggregated.index:
            out[canonical] = float(aggregated[col])
        else:
            # Fall back: mean over rows for ratios; sum for n_detected.
            series = pd.to_numeric(df[col], errors="coerce")
            if canonical == "n_detected":
                out[canonical] = float(series.sum())
            else:
                out[canonical] = float(series.mean())
    return out


def _bca_ci(values: np.ndarray, alpha: float = 0.05, n_boot: int = 2000,
            seed: int = 42) -> Tuple[float, float]:
    """Lightweight BCa CI helper (calls ``sl_ads.stats.bootstrap_ci`` if
    available; otherwise falls back to the percentile bootstrap).
    """
    values = values[np.isfinite(values)]
    if values.size < 2:
        return (float("nan"), float("nan"))
    try:
        from sl_ads.stats.bootstrap_ci import bootstrap_bca_ci
        # Reuse bootstrap_bca_ci by treating identity as the metric_fn.
        # bootstrap_bca_ci expects (y_true, y_pred, metric_fn); pass the
        # values both as y_true and y_pred and use a metric that ignores
        # y_pred for a single-sample mean CI.
        res = bootstrap_bca_ci(
            values,
            values,
            lambda a, _b: float(np.mean(a)),
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
        )
        return (float(res["ci_low"]), float(res["ci_high"]))
    except (ImportError, Exception):  # pragma: no cover — fallback
        rng = np.random.default_rng(seed)
        boots = [
            float(np.mean(rng.choice(values, size=values.size, replace=True)))
            for _ in range(n_boot)
        ]
        lo = float(np.quantile(boots, alpha / 2))
        hi = float(np.quantile(boots, 1 - alpha / 2))
        return (lo, hi)


def aggregate_runs(
    per_run: pd.DataFrame,
    metrics: Iterable[str] = HEADLINE_METRICS,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute mean / std / min / max / BCa CI across seeds.

    Parameters
    ----------
    per_run : DataFrame
        One row per seed with columns ``seed`` and the metric columns.
    metrics : iterable of str
        Subset of metric column names to aggregate.
    alpha : float
        Two-sided significance level for the BCa CI.

    Returns
    -------
    DataFrame with columns ``metric``, ``n``, ``mean``, ``std``,
    ``min``, ``max``, ``ci_low``, ``ci_high``.
    """
    rows = []
    for m in metrics:
        if m not in per_run.columns:
            continue
        v = pd.to_numeric(per_run[m], errors="coerce").to_numpy(dtype=float)
        v_finite = v[np.isfinite(v)]
        if v_finite.size == 0:
            rows.append({
                "metric": m, "n": 0,
                "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
            })
            continue
        ci_lo, ci_hi = _bca_ci(v_finite, alpha=alpha)
        rows.append({
            "metric": m,
            "n": int(v_finite.size),
            "mean": float(v_finite.mean()),
            # ddof=1 — sample std with Bessel correction, matches the
            # convention of the BCa fallback and standard reporting.
            "std": float(v_finite.std(ddof=1)) if v_finite.size > 1 else 0.0,
            "min": float(v_finite.min()),
            "max": float(v_finite.max()),
            "ci_low": ci_lo,
            "ci_high": ci_hi,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────
def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GFM table without depending on ``tabulate``.

    pandas' ``to_markdown`` requires the optional ``tabulate`` package
    which is not always installed; this minimalist fallback produces a
    valid pipe-table from any 2-D DataFrame.
    """
    if df.empty:
        return "_(empty)_"
    cols = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([_format_cell(row[c]) for c in df.columns])
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([header, sep, body])


def _format_cell(v) -> str:
    if isinstance(v, float):
        if not np.isfinite(v):
            return "NaN"
        return f"{v:.6g}"
    return str(v)


def _format_aggregate_md(per_run: pd.DataFrame, agg: pd.DataFrame) -> str:
    """Markdown report ready to drop into a paper appendix."""
    lines = []
    lines.append("# Multi-seed evaluation report")
    lines.append("")
    lines.append(f"- Number of seeds          : {len(per_run)}")
    if "seed" in per_run.columns:
        lines.append(f"- Seeds                    : {sorted(per_run['seed'].tolist())}")
    if "run_id" in per_run.columns:
        ids = [r for r in per_run["run_id"].tolist() if isinstance(r, str)]
        if ids:
            lines.append(f"- Run IDs                  : {ids}")
    lines.append(f"- Aggregation timestamp UTC: {_utc_now().isoformat()}")
    lines.append("")
    lines.append("## Per-seed metrics")
    lines.append("")
    lines.append(_df_to_markdown(per_run))
    lines.append("")
    lines.append("## Aggregate (mean ± std, BCa 95 % CI)")
    lines.append("")
    lines.append(_df_to_markdown(agg))
    lines.append("")
    lines.append("## Reference")
    lines.append("")
    lines.append("Wu, R., Keogh, E. (2021). *Current Time Series Anomaly "
                 "Detection Benchmarks are Flawed and are Creating the "
                 "Illusion of Progress.* IEEE TKDE.")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Public entry points
# ──────────────────────────────────────────────────────────────────────
def run(
    seeds: List[int],
    dataset: str,
    project_root: Path,
    output_dir: Path,
    from_step: Optional[str] = None,
    to_step: Optional[str] = None,
    timeout_s: Optional[int] = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Launch ``run_pipeline.py`` once per seed.  Returns the per-seed
    record DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in seeds:
        log_path = output_dir / f"seed_{seed:03d}.log"
        rec = _run_one_seed(
            seed=seed,
            dataset=dataset,
            from_step=from_step,
            to_step=to_step,
            project_root=project_root,
            timeout_s=timeout_s,
            dry_run=dry_run,
            log_path=log_path,
        )
        records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "multi_seed_orchestration.csv", index=False)
    return df


def aggregate(
    run_ids: List[str],
    project_root: Path,
    seeds: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate metrics across the given run_ids.

    Parameters
    ----------
    run_ids : list of str
        Names of folders under ``project_root/results/``.
    seeds : list of int, optional
        If supplied, must be parallel to ``run_ids``; recorded in the
        per-run DataFrame for traceability.
    output_dir : Path, optional
        If provided, write per-run, aggregate, and report files there.

    Returns
    -------
    (per_run_df, aggregate_df)
    """
    if seeds is not None and len(seeds) != len(run_ids):
        raise ValueError(
            f"seeds length ({len(seeds)}) must match run_ids length ({len(run_ids)})"
        )
    rows = []
    results_root = project_root / "results"
    for i, rid in enumerate(run_ids):
        metrics = _read_run_metrics(results_root, rid)
        row = {"run_id": rid, **metrics}
        if seeds is not None:
            row["seed"] = seeds[i]
        rows.append(row)
    per_run = pd.DataFrame(rows)
    agg = aggregate_runs(per_run)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        per_run.to_csv(output_dir / "multi_seed_per_run.csv", index=False)
        agg.to_csv(output_dir / "multi_seed_aggregate.csv", index=False)
        report = _format_aggregate_md(per_run, agg)
        (output_dir / "multi_seed_report.md").write_text(report, encoding="utf-8")
    return per_run, agg


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_multi_seed",
        description="Multi-seed pipeline orchestrator and aggregator (TASK-53).",
    )
    p.add_argument("--mode", choices=("run", "aggregate", "all"), default="all",
                   help="run = launch pipelines; aggregate = read existing run_ids; "
                        "all = run then aggregate (default).")
    p.add_argument("--seeds", default="0,1,2,3,4",
                   help="Comma-separated integer seeds.  Default '0,1,2,3,4'.")
    p.add_argument("--dataset", default="RedeRio",
                   help="Dataset name passed through to run_pipeline.py.")
    p.add_argument("--from-step", default=None,
                   help="Pass-through to run_pipeline.py --from-step.")
    p.add_argument("--to-step", default=None,
                   help="Pass-through to run_pipeline.py --to-step.")
    p.add_argument("--run-ids", default=None,
                   help="Comma-separated run_ids for aggregate-only mode.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write outputs.  Default: "
                        "results/_multi_seed/<UTC timestamp>.")
    p.add_argument("--project-root", default=None,
                   help="Project root containing run_pipeline.py.  "
                        "Default: parent of this file's grand-grand-parent.")
    p.add_argument("--timeout-s", type=int, default=None,
                   help="Per-seed wall-clock timeout in seconds.")
    p.add_argument("--dry-run", action="store_true",
                   help="In run mode: print the commands without launching them.")
    p.add_argument("--self-test", action="store_true",
                   help="Validate the aggregator on synthetic data and exit.")
    return p


def _resolve_project_root(arg: Optional[str]) -> Path:
    if arg:
        return Path(arg).resolve()
    # this file = .../src/sl_ads/evaluate/run_multi_seed.py
    return Path(__file__).resolve().parent.parent.parent.parent


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.self_test:
        return _self_test()

    project_root = _resolve_project_root(args.project_root)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        ts = _utc_now().strftime("%Y%m%dT%H%M%SZ")
        output_dir = (project_root / "results" / "_multi_seed" / ts).resolve()

    seeds = _parse_seed_list(args.seeds) if args.seeds else []

    started = time.time()
    per_run_df: Optional[pd.DataFrame] = None

    # --- run mode ---
    if args.mode in ("run", "all"):
        if not seeds:
            print("[ERR] --seeds is empty in run mode.", file=sys.stderr)
            return 2
        per_run_df = run(
            seeds=seeds,
            dataset=args.dataset,
            project_root=project_root,
            output_dir=output_dir,
            from_step=args.from_step,
            to_step=args.to_step,
            timeout_s=args.timeout_s,
            dry_run=args.dry_run,
        )

    # --- aggregate mode ---
    if args.mode in ("aggregate", "all"):
        if args.run_ids:
            run_ids = [s.strip() for s in args.run_ids.split(",") if s.strip()]
            seeds_for_agg = seeds if (len(seeds) == len(run_ids)) else None
        elif per_run_df is not None:
            run_ids = [r for r in per_run_df.get("run_id", []).tolist()
                       if isinstance(r, str)]
            seeds_for_agg = per_run_df["seed"].tolist()[: len(run_ids)]
        else:
            print("[ERR] aggregate mode needs --run-ids when not chained "
                  "with --mode all.", file=sys.stderr)
            return 2

        if not run_ids:
            print("[WARN] no run_ids resolved; nothing to aggregate.")
            return 0

        per_run, agg = aggregate(
            run_ids=run_ids,
            project_root=project_root,
            seeds=seeds_for_agg,
            output_dir=output_dir,
        )
        print(f"[OK] aggregated {len(per_run)} runs → {output_dir}")
        if not agg.empty:
            print(agg.to_string(index=False))

    # --- manifest ---
    manifest = {
        "task": "TASK-53",
        "mode": args.mode,
        "dataset": args.dataset,
        "seeds": seeds,
        "started_utc": _dt.datetime.fromtimestamp(
            started, tz=_dt.timezone.utc
        ).isoformat(),
        "finished_utc": _utc_now().isoformat(),
        "duration_s": round(time.time() - started, 2),
        "argv": sys.argv,
        "python": sys.version,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "multi_seed_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


# ──────────────────────────────────────────────────────────────────────
# Self-test (aggregator only — orchestration requires a live pipeline)
# ──────────────────────────────────────────────────────────────────────
def _self_test() -> int:
    print("[TEST] run_multi_seed.py — self-test (aggregation only)")

    # 1. _parse_seed_list
    assert _parse_seed_list("0,1,2,3,4") == [0, 1, 2, 3, 4]
    assert _parse_seed_list(" 7 , 11 ") == [7, 11]
    try:
        _parse_seed_list("0,0,1")
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate seeds must raise.")
    print("   [OK] _parse_seed_list")

    # 2. aggregate_runs reproduces hand-computed mean / std / range
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "seed": [0, 1, 2, 3, 4],
        "F1_micro": [0.78, 0.79, 0.78, 0.77, 0.79],
        "F1_macro": [0.88, 0.89, 0.87, 0.88, 0.88],
        "MCC":      [0.77, 0.78, 0.76, 0.77, 0.77],
        "FAR_pct":  [0.30, 0.32, 0.29, 0.31, 0.30],
        "n_detected": [14, 14, 14, 14, 14],
    })
    agg = aggregate_runs(df)
    assert set(agg["metric"]) == set(HEADLINE_METRICS)
    f1_micro_row = agg[agg["metric"] == "F1_micro"].iloc[0]
    expected_mean = float(np.mean([0.78, 0.79, 0.78, 0.77, 0.79]))
    assert abs(f1_micro_row["mean"] - expected_mean) < 1e-9
    assert f1_micro_row["min"] == 0.77
    assert f1_micro_row["max"] == 0.79
    assert f1_micro_row["n"] == 5
    # Sample std with ddof=1.
    assert abs(f1_micro_row["std"] - 0.008366600265340756) < 1e-9
    # CI must straddle the mean for a small, balanced sample.
    assert f1_micro_row["ci_low"] <= f1_micro_row["mean"] <= f1_micro_row["ci_high"]
    print(f"   [OK] aggregate_runs: F1_micro mean={f1_micro_row['mean']:.4f} "
          f"std={f1_micro_row['std']:.4f} "
          f"CI=[{f1_micro_row['ci_low']:.4f}, {f1_micro_row['ci_high']:.4f}]")

    # 3. _read_run_metrics returns NaN-padded dict on missing files
    metrics = _read_run_metrics(Path("/nonexistent_path"), "fake_run_id")
    assert set(metrics.keys()) == set(HEADLINE_METRICS)
    assert all(np.isnan(v) for v in metrics.values())
    print("   [OK] _read_run_metrics graceful on missing files")

    # 4. aggregate end-to-end on synthetic results tree
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        (tmp_root / "results").mkdir()
        for i, seed in enumerate([0, 1, 2]):
            run_id = f"run_{seed:03d}"
            run_dir = tmp_root / "results" / run_id / "evaluation"
            run_dir.mkdir(parents=True)
            pd.DataFrame([
                {"attack": "__ALL__",
                 "f1_micro": 0.78 + 0.005 * i,
                 "f1_macro": 0.88 + 0.005 * i,
                 "mcc": 0.77 + 0.005 * i,
                 "far_pct": 0.30 + 0.005 * i,
                 "n_detected": 14},
            ]).to_csv(run_dir / "eval_detection_summary.csv", index=False)
            (tmp_root / "results" / run_id / "_run_manifest.json").write_text(
                json.dumps({"summary": {"finished_utc": f"2026-01-0{i+1}T00:00:00Z"}}),
                encoding="utf-8",
            )

        per_run, agg = aggregate(
            run_ids=[f"run_{s:03d}" for s in [0, 1, 2]],
            project_root=tmp_root,
            seeds=[0, 1, 2],
            output_dir=tmp_root / "_out",
        )
        assert len(per_run) == 3, per_run
        assert (per_run["seed"].tolist() == [0, 1, 2])
        f1m_row = agg[agg["metric"] == "F1_micro"].iloc[0]
        assert abs(f1m_row["mean"] - 0.785) < 1e-9
        assert (tmp_root / "_out" / "multi_seed_per_run.csv").is_file()
        assert (tmp_root / "_out" / "multi_seed_aggregate.csv").is_file()
        assert (tmp_root / "_out" / "multi_seed_report.md").is_file()
    print("   [OK] aggregate end-to-end on synthetic results tree")

    # 5. _build_pipeline_command shape
    cmd = _build_pipeline_command(
        dataset="RedeRio",
        from_step="opinions",
        to_step="audit",
        project_root=Path("/tmp/proj"),
    )
    assert cmd[1].endswith("run_pipeline.py")
    assert "--dataset" in cmd and "RedeRio" in cmd
    assert "--from-step" in cmd and "opinions" in cmd
    print("   [OK] _build_pipeline_command")

    # 6. Latest-run-id heuristic: tie-break by finished_utc
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp) / "results"
        tmp_root.mkdir()
        for name, ts in [("a", "2026-01-01"), ("b", "2026-01-03"),
                         ("c", "2026-01-02")]:
            (tmp_root / name).mkdir()
            (tmp_root / name / "_run_manifest.json").write_text(
                json.dumps({"summary": {"finished_utc": ts}}),
                encoding="utf-8",
            )
        latest = _latest_run_id(tmp_root)
        assert latest == "b", latest
    print("   [OK] _latest_run_id resolves the most recent finish")

    print("[OK] run_multi_seed.py — ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
