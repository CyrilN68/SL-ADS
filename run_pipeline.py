"""run_pipeline.py — Single-binary entry point for the SL-ADS pipeline.

Phase H (2026-04-29).  Replaces the legacy ``run_full_sl_ads.py``
(now a deprecation shim — see
``docs/archive/2026-05-11_public_release_cleanup/top_level/RENAMING_LOG_PHASE_H.md``).

Usage::

    python run_pipeline.py                                    # defaults
    python run_pipeline.py --dataset RedeRio
    python run_pipeline.py --dataset METR-LA --from-step evidence
    python run_pipeline.py --dataset GECCO-IoT --to-step opinions
    python run_pipeline.py --dataset CESNET-TimeSeries24 --dry-run
    python run_pipeline.py --list-steps

What this launcher does that the old one didn't:

* Steps are dispatched as ``python -m sl_ads.<subpkg>.<module>``
  rather than ``python <legacy_script>.py`` — the sub-processes
  import only the new package layout, no shim is touched.
* ``PYTHONPATH=src`` is exported to every sub-process so the package
  is importable regardless of the cwd.
* On success, the active ``CONFIG['RESULTS_DIR']`` directory produced by
  the pipeline is copied into ``outputs/`` and then snapshotted into
  ``results/<run_id>/`` (deterministic ID — see
  :func:`sl_ads.utils_manifest.compute_run_id`).  ``outputs/`` is the
  *current* run (overwritten); ``results/<run_id>/`` is the
  *historical archive* used to diff experiments across iterations.

Pipeline profiles (per-dataset):

* RedeRio          : train → evidence → inject → opinions
                     → eval_injection → qualify_sbn → eval_qualify
                     → ablation → compare_if → audit
* METR-LA / GECCO-IoT / CESNET-TimeSeries24 : train → evidence
                     → opinions → qualify_sbn → eval_qualify
                     → ablation_labeled → compare_if
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


def _configure_console_encoding() -> None:
    """Keep CLI smoke commands printable on legacy Windows consoles."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_configure_console_encoding()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src")

# Make `sl_ads` importable in the parent process (subprocess inherits via env)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Step status codes (PATCH m-02 / F19, retained verbatim)
# ─────────────────────────────────────────────────────────────────────────────
STATUS_OK      = "ok"
STATUS_FAILED  = "failed"
STATUS_SKIPPED = "skipped"
STATUS_DRYRUN  = "dry-run"
STATUS_ABORTED = "aborted"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline definitions per dataset.  Each tuple is (step_name, dotted_module).
# Sub-processes are launched as ``python -m <dotted_module>``.
# ─────────────────────────────────────────────────────────────────────────────
_STEPS_REDERIO: List[Tuple[str, str]] = [
    ("train",          "sl_ads.train.train_models"),
    ("evidence",       "sl_ads.train.compute_evidence"),
    ("inject",         "sl_ads.inject.evidence_level"),
    ("opinions",       "sl_ads.core.opinions_pipeline"),
    ("eval_injection", "sl_ads.evaluate.evaluate_injection"),
    ("qualify_sbn",    "sl_ads.qualify.sbn_qualifier"),
    ("eval_qualify",   "sl_ads.evaluate.evaluate_qualify_sbn"),
    ("ablation",       "sl_ads.ablation.run_ablation"),
    ("compare_if",     "sl_ads.compare.compare_if_fair"),
    ("audit",          "sl_ads.audit.audit_full_dataset"),
]

_STEPS_METR_LA: List[Tuple[str, str]] = [
    ("train",        "sl_ads.train.train_models"),
    ("evidence",     "sl_ads.train.compute_evidence"),
    ("opinions",     "sl_ads.core.opinions_pipeline"),
    ("qualify_sbn",  "sl_ads.qualify.sbn_qualifier"),
    ("eval_qualify", "sl_ads.evaluate.evaluate_qualify_injected"),
    ("ablation",     "sl_ads.ablation.run_ablation_labeled"),
    ("compare_if",   "sl_ads.compare.compare_if_fair"),
]

_STEPS_GECCO: List[Tuple[str, str]] = [
    ("train",       "sl_ads.train.train_models"),
    ("evidence",    "sl_ads.train.compute_evidence"),
    ("opinions",    "sl_ads.core.opinions_pipeline"),
    ("qualify_sbn", "sl_ads.qualify.sbn_qualifier"),
    ("ablation",    "sl_ads.ablation.run_ablation_labeled"),
    ("compare_if",  "sl_ads.compare.compare_if_fair"),
]

_STEPS_CESNET: List[Tuple[str, str]] = [
    ("train",       "sl_ads.train.train_models"),
    ("evidence",    "sl_ads.train.compute_evidence"),
    ("opinions",    "sl_ads.core.opinions_pipeline"),
    ("qualify_sbn", "sl_ads.qualify.sbn_qualifier"),
    ("ablation",    "sl_ads.ablation.run_ablation_labeled"),
    ("compare_if",  "sl_ads.compare.compare_if_fair"),
]

_PIPELINE_BY_DATASET: Dict[str, List[Tuple[str, str]]] = {
    "RedeRio":             _STEPS_REDERIO,
    "":                    _STEPS_REDERIO,  # empty = RedeRio
    "METR-LA":             _STEPS_METR_LA,
    "GECCO-IoT":           _STEPS_GECCO,
    "CESNET-TimeSeries24": _STEPS_CESNET,
}


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing & helpers
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_dataset(cli_dataset: Optional[str]) -> str:
    if cli_dataset:
        return cli_dataset
    env_ds = os.environ.get("SL_ACTIVE_DATASET", "").strip()
    if env_ds:
        return env_ds
    try:
        from sl_ads.config import CONFIG  # Phase H
        return CONFIG.get("ACTIVE_DATASET", "")
    except Exception:
        return ""


def _get_steps(dataset: str) -> List[Tuple[str, str]]:
    if dataset not in _PIPELINE_BY_DATASET:
        print(f"[WARN] Dataset '{dataset}' inconnu → pipeline RedeRio par défaut.")
    return _PIPELINE_BY_DATASET.get(dataset, _STEPS_REDERIO)


def _slice_steps(steps: List[Tuple[str, str]], from_step: str,
                 to_step: str) -> List[Tuple[str, str]]:
    names = [s for s, _ in steps]
    i = names.index(from_step)
    j = names.index(to_step)
    if i > j:
        raise ValueError(f"--from-step ({from_step}) doit précéder --to-step ({to_step}).")
    return steps[i:j + 1]


def _build_env(dataset: str) -> dict:
    """Build env vars for sub-processes — Phase H makes ``sl_ads``
    importable by exporting ``PYTHONPATH``."""
    env = os.environ.copy()
    env["SL_ACTIVE_DATASET"] = dataset
    env["PYTHONUNBUFFERED"]  = "1"  # immediate stdout flush
    # Prepend src/ to PYTHONPATH so subprocess `python -m sl_ads.X` works.
    existing_pp = env.get("PYTHONPATH", "")
    parts = [SRC_DIR] + ([existing_pp] if existing_pp else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def parse_args(steps: List[Tuple[str, str]]) -> argparse.Namespace:
    step_names = [s for s, _ in steps]
    parser = argparse.ArgumentParser(
        description="Run the SL-ADS pipeline end-to-end (Phase H launcher).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default=None, metavar="DATASET",
                        help="Override CONFIG['ACTIVE_DATASET'].")
    parser.add_argument("--from-step", choices=step_names, default=step_names[0],
                        metavar="STEP")
    parser.add_argument("--to-step",   choices=step_names, default=step_names[-1],
                        metavar="STEP")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--list-steps", action="store_true",
                        help="List the active pipeline and exit.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Don't stop on a step failure (log it).")
    parser.add_argument("--no-archive", action="store_true",
                        help="Skip the post-run snapshot to results/<run_id>/.")
    parser.add_argument("--archive-dir", default=None, metavar="DIR",
                        help="Override the historical archive root "
                             "(default: <project_root>/results).")
    return parser.parse_args()


def _print_context(dataset: str, steps: List[Tuple[str, str]]) -> None:
    """Display the resolved context (version, paths, threshold)."""
    _saved = os.environ.get("SL_ACTIVE_DATASET")
    os.environ["SL_ACTIVE_DATASET"] = dataset
    try:
        import importlib
        import sl_ads.config as _cfg_mod  # Phase H
        importlib.reload(_cfg_mod)
        _cfg = _cfg_mod.CONFIG
        from sl_ads.paths import (  # Phase H
            get_version_names, get_results_dir, get_model_path,
            get_decision_threshold,
        )
        version, version_modif = get_version_names(_cfg)
        results_dir = get_results_dir(_cfg, up_levels=1)
        model_path  = get_model_path(_cfg, up_levels=1)
        thr         = get_decision_threshold(_cfg, up_levels=1)
    except Exception as e:
        version = version_modif = results_dir = model_path = "?"
        thr = 0.0
        print(f"  [WARN] Could not read context: {e}")
    finally:
        if _saved is None:
            os.environ.pop("SL_ACTIVE_DATASET", None)
        else:
            os.environ["SL_ACTIVE_DATASET"] = _saved

    print("=" * 80)
    print("  SL-ADS PIPELINE LAUNCHER  (Phase H)")
    print("=" * 80)
    print(f"  ACTIVE_DATASET    : {dataset or '(RedeRio)'}")
    print(f"  VERSION_NAME      : {version}")
    print(f"  VERSION_NAME_MODIF: {version_modif}")
    print(f"  RESULTS_DIR       : {results_dir}")
    print(f"  MODEL_PATH        : {model_path}")
    print(f"  DECISION_THRESHOLD: {thr:.4f}")
    print(f"  Pipeline          : {' → '.join(s for s, _ in steps)}")
    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# Step execution
# ─────────────────────────────────────────────────────────────────────────────
def run_step(step_name: str, module_name: str, env: dict,
             dry_run: bool = False) -> Tuple[str, Optional[int]]:
    """Launch one pipeline step as ``python -m <module_name>``.

    Returns ``(status, returncode)``.  ``returncode`` is ``None`` when
    the process never started.
    """
    cmd = [sys.executable, "-m", module_name]
    t0 = time.time()
    print(f"\n{'─' * 80}")
    print(f"[STEP] {step_name:20s}  →  python -m {module_name}")
    if dry_run:
        print("       (dry-run — not executed)")
        return STATUS_DRYRUN, None
    proc = subprocess.run(cmd, cwd=BASE_DIR, env=env)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"\n❌ Step '{step_name}' failed (exit {proc.returncode}) — {elapsed:.0f}s")
        return STATUS_FAILED, proc.returncode
    print(f"\n✅ Step '{step_name}' done in {elapsed:.0f}s")
    return STATUS_OK, 0


# ─────────────────────────────────────────────────────────────────────────────
# Output / results dual-write (Phase H)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_run_id(dataset: str) -> str:
    """Compute a deterministic run_id from CONFIG, git SHA, and env."""
    try:
        from sl_ads.utils_manifest import compute_run_id, _get_git_sha
        from sl_ads.config import CONFIG
        return compute_run_id(
            config=CONFIG,
            git_sha=_get_git_sha(BASE_DIR),
            extras={"dataset": dataset, "launcher": "run_pipeline.py"},
        )
    except Exception as exc:
        # Fallback: timestamp-based id.  Loses determinism but the run
        # still gets archived.
        print(f"  [WARN] Could not compute deterministic run_id ({exc}); "
              "falling back to timestamp.")
        return _dt.datetime.utcnow().strftime("ts-%Y%m%dT%H%M%SZ")


def _same_path(left: str, right: str) -> bool:
    """Return True when two paths resolve to the same filesystem target."""
    return (os.path.normcase(os.path.abspath(left))
            == os.path.normcase(os.path.abspath(right)))


def _resolve_active_results_dir(dataset: str) -> str:
    """Return the absolute CONFIG-driven results directory for this run.

    Pipeline steps run with ``cwd=BASE_DIR`` and mostly use paths such as
    ``../results/resultats_<VERSION_NAME>``.  Resolving relative paths against
    ``BASE_DIR`` keeps the launcher aligned with the subprocesses.
    """
    _saved = os.environ.get("SL_ACTIVE_DATASET")
    os.environ["SL_ACTIVE_DATASET"] = dataset
    try:
        import importlib
        import sl_ads.config as _cfg_mod
        importlib.reload(_cfg_mod)
        from sl_ads.paths import get_results_dir

        raw_dir = get_results_dir(_cfg_mod.CONFIG, up_levels=1)
    finally:
        if _saved is None:
            os.environ.pop("SL_ACTIVE_DATASET", None)
        else:
            os.environ["SL_ACTIVE_DATASET"] = _saved

    if os.path.isabs(raw_dir):
        return os.path.abspath(raw_dir)
    return os.path.abspath(os.path.join(BASE_DIR, raw_dir))


def _copytree_replace(src: str, dst: str) -> None:
    """Copy ``src`` to ``dst`` after replacing the old destination tree."""
    if _same_path(src, dst):
        return
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, dirs_exist_ok=False)


def _write_run_manifest(dst: str, run_id: str, dataset: str, summary: Dict[str, Any],
                        source_dir: str, current_outputs_dir: str) -> None:
    """Write the audit manifest used by both outputs/ and archived runs."""
    manifest = {
        "run_id":              run_id,
        "dataset":             dataset,
        "snapshot_utc":        _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source_basename":     os.path.basename(os.path.normpath(source_dir)),
        "source_dir":          os.path.abspath(source_dir),
        "current_outputs_dir": os.path.abspath(current_outputs_dir),
        "summary":             summary,
    }
    with open(os.path.join(dst, "_run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _archive_outputs(run_id: str, dataset: str, archive_dir: str,
                     summary: Dict[str, Any]) -> Optional[str]:
    """Mirror the active results directory, then archive it under run_id.

    Returns the destination path, or ``None`` when there's nothing to
    archive (e.g. no results were produced).
    """
    current_outputs_dir = os.path.join(BASE_DIR, "outputs")
    try:
        src = _resolve_active_results_dir(dataset)
    except Exception as exc:
        print(f"\n[archive][WARN] Could not resolve CONFIG RESULTS_DIR: {exc}")
        src = current_outputs_dir

    if not os.path.isdir(src):
        if os.path.isdir(current_outputs_dir):
            print(f"\n[archive] Active RESULTS_DIR not found ({src}); "
                  f"falling back to outputs/.")
            src = current_outputs_dir
        # Legacy fallback: actual_outputs/ still exists from pre-Phase-H runs.
        else:
            legacy_src = os.path.join(BASE_DIR, "actual_outputs")
            if os.path.isdir(legacy_src):
                print(f"\n[archive] Active RESULTS_DIR not found ({src}); "
                      f"falling back to actual_outputs/ (legacy).")
                src = legacy_src
            else:
                print(f"\n[archive] No active RESULTS_DIR, outputs/, or "
                      f"actual_outputs/ to archive.")
                return None

    summary["artifact_source_dir"] = os.path.abspath(src)
    summary["current_outputs_dir"] = os.path.abspath(current_outputs_dir)

    try:
        _copytree_replace(src, current_outputs_dir)
        _write_run_manifest(
            current_outputs_dir, run_id, dataset, summary,
            source_dir=src, current_outputs_dir=current_outputs_dir,
        )
    except Exception as exc:
        print(f"\n[archive][WARN] Could not refresh outputs/ from {src}: {exc}")
        return None

    dst = os.path.join(archive_dir, run_id)
    if os.path.exists(dst):
        # Same run_id already archived — keep the historical copy
        # untouched, do not silently overwrite.
        print(f"\n[archive] {dst} already exists — not overwriting.")
        return dst

    os.makedirs(archive_dir, exist_ok=True)
    try:
        shutil.copytree(current_outputs_dir, dst, dirs_exist_ok=False)
    except Exception as exc:
        print(f"\n[archive][WARN] Could not snapshot outputs/ -> {dst}: {exc}")
        return None

    try:
        _write_run_manifest(
            dst, run_id, dataset, summary,
            source_dir=src, current_outputs_dir=current_outputs_dir,
        )
    except OSError as exc:
        print(f"[archive][WARN] Could not write _run_manifest.json: {exc}")
    print(f"\n[archive] {os.path.abspath(src)} mirrored to outputs/ and "
          f"snapshotted -> {dst}")
    return dst


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # Pre-parse to know --dataset before constructing the choices list.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--dataset", default=None)
    _pre_args, _ = _pre.parse_known_args()
    dataset = _resolve_dataset(_pre_args.dataset)

    steps = _get_steps(dataset)
    args  = parse_args(steps)
    if args.dataset:
        dataset = args.dataset

    if args.list_steps:
        print(f"\nPipeline pour dataset='{dataset or 'RedeRio'}' :")
        for i, (name, mod) in enumerate(steps, 1):
            print(f"  {i:2d}. {name:22s}  →  python -m {mod}")
        return

    _print_context(dataset, steps)

    selected_steps = _slice_steps(steps, args.from_step, args.to_step)
    env = _build_env(dataset)

    print(f"\nÉtapes sélectionnées : {' → '.join(s for s, _ in selected_steps)}\n")

    results: List[Dict[str, Any]] = []
    pipeline_start = time.time()
    pipeline_start_utc = _dt.datetime.now(_dt.timezone.utc).isoformat()
    aborted = False

    for step_name, module_name in selected_steps:
        if aborted:
            print(f"\n[ABORT] {step_name:20s}  (skipped — previous step failed)")
            results.append({
                "step":         step_name,
                "module":       module_name,
                "status":       STATUS_ABORTED,
                "returncode":   None,
                "duration_s":   0,
                "duration_str": "0s",
            })
            continue

        t_step = time.time()
        status, rc = run_step(step_name, module_name, env=env, dry_run=args.dry_run)
        elapsed_s = int(time.time() - t_step)
        results.append({
            "step":         step_name,
            "module":       module_name,
            "status":       status,
            "returncode":   rc,
            "duration_s":   elapsed_s,
            "duration_str": f"{elapsed_s}s",
        })
        if status == STATUS_FAILED and not args.continue_on_error:
            print(f"\n🛑 Pipeline interrupted at step '{step_name}'.")
            print("   Use --continue-on-error to continue past failures.")
            aborted = True

    # ── Summary ──────────────────────────────────────────────────────────────
    total_s = int(time.time() - pipeline_start)
    mins, secs = divmod(total_s, 60)
    print(f"\n{'=' * 80}")
    print(f"  PIPELINE SUMMARY — dataset={dataset or 'RedeRio'}  ({mins}m{secs:02d}s total)")
    print(f"{'=' * 80}")
    _ICONS = {
        STATUS_OK:      "✅",
        STATUS_FAILED:  "❌",
        STATUS_SKIPPED: "⏭ ",
        STATUS_DRYRUN:  "🧪",
        STATUS_ABORTED: "🛑",
    }
    for rec in results:
        icon = _ICONS.get(rec["status"], "?")
        print(f"  {icon}  {rec['step']:22s}  "
              f"{rec['duration_str']:>6}   [{rec['status']}]")

    counts = {k: 0 for k in (STATUS_OK, STATUS_FAILED, STATUS_SKIPPED,
                              STATUS_DRYRUN, STATUS_ABORTED)}
    for rec in results:
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1

    total_steps = len(results)
    if counts[STATUS_FAILED] == 0 and counts[STATUS_ABORTED] == 0:
        print(
            f"\n✅ Pipeline complete: "
            f"{counts[STATUS_OK]} ok, "
            f"{counts[STATUS_SKIPPED]} skipped, "
            f"{counts[STATUS_DRYRUN]} dry-run "
            f"(total={total_steps})."
        )
    else:
        print(
            f"\n⚠  Pipeline incomplete: "
            f"{counts[STATUS_OK]} ok, "
            f"{counts[STATUS_FAILED]} failed, "
            f"{counts[STATUS_ABORTED]} aborted, "
            f"{counts[STATUS_SKIPPED]} skipped "
            f"(total={total_steps})."
        )

    # ── Exit-summary JSON (PATCH m-02 / F19) ─────────────────────────────────
    exit_summary = {
        "dataset":            dataset or "RedeRio",
        "pipeline":           [s for s, _ in steps],
        "selected_steps":     [s for s, _ in selected_steps],
        "from_step":          args.from_step,
        "to_step":            args.to_step,
        "dry_run":            bool(args.dry_run),
        "continue_on_error":  bool(args.continue_on_error),
        "started_utc":        pipeline_start_utc,
        "finished_utc":       _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "total_duration_s":   total_s,
        "counts":             counts,
        "steps":              results,
        "launcher":           "run_pipeline.py (Phase H)",
    }
    summary_path = os.path.join(BASE_DIR, "pipeline_run_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(exit_summary, f, ensure_ascii=False, indent=2)
        print(f"\n   exit-summary JSON saved: {summary_path}")
    except OSError as _exc:
        print(f"\n   [WARN] Could not write pipeline_run_summary.json: {_exc!r}")

    # ── Phase H dual-write: snapshot outputs/ → results/<run_id>/ ────────────
    pipeline_succeeded = (counts[STATUS_FAILED] == 0
                          and counts[STATUS_ABORTED] == 0
                          and not args.dry_run)
    if pipeline_succeeded and not args.no_archive:
        archive_dir = (
            args.archive_dir
            if args.archive_dir is not None
            else os.path.join(BASE_DIR, "results")
        )
        run_id = _compute_run_id(dataset)
        archived_to = _archive_outputs(run_id, dataset, archive_dir, exit_summary)
        if archived_to:
            exit_summary["archived_to"] = archived_to
            exit_summary["run_id"]      = run_id
            try:
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(exit_summary, f, ensure_ascii=False, indent=2)
            except OSError:
                pass
    elif args.no_archive:
        print("\n[archive] --no-archive: skipping snapshot to results/<run_id>/.")

    if counts[STATUS_FAILED] > 0 or counts[STATUS_ABORTED] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
