"""
utils_manifest.py — Auto-populated run manifest (PATCH m-06 / F23)
==================================================================

Appends a timestamped block to a master ``MANIFEST.md`` file every time
``evaluate_injection_v2.py`` finishes a run. The manifest grows
monotonically (append-only) so the whole experimental history — across
dataset variants, parameter sweeps, and fusion-operator choices — lives
in a single, human-readable markdown file.

Each entry records:
    * UTC timestamp of the run
    * ``VERSION_NAME`` (dataset × split × params × fusion × hash)
    * source detection CSV (absolute path)
    * key evaluation metrics (F1 variants, precision, recall, MCC, FPR,
      TTD summary, number of detected attacks, operational threshold)
    * environment signature: Python version, platform, pinned library
      versions (numpy, pandas, scipy, scikit-learn, statsmodels,
      joblib, prophet, matplotlib)
    * git commit SHA (short) if the pipeline runs inside a git working
      tree — ``"no-git"`` otherwise

The file is UTF-8 and the header is written once, guarded by a
``<!-- manifest:header:v1 -->`` marker so subsequent runs can detect
and skip header rewrites.

Usage (from the evaluation entry point) ::

    from sl_ads.utils_manifest import append_manifest_entry
    append_manifest_entry(
        metrics={"f1_coverage": 0.87, "precision_window": 0.91, ...},
        version_name=VERSION_NAME,
        csv_path=RESULTS_CSV,
        project_root=os.path.dirname(os.path.abspath(__file__)),
    )

Design constraints:
    * never raises: a manifest failure must NOT break the pipeline.
      All errors are caught and logged as warnings.
    * atomic-ish write: re-read then append in one ``open("a")`` call.
    * Windows-cp1252 safe: explicit ``encoding="utf-8"`` everywhere.
"""

from __future__ import annotations

import datetime as _dt
import hashlib as _hashlib
import json as _json
import os as _os
import platform as _platform
import subprocess as _subprocess
import sys as _sys
import warnings as _warnings
from pathlib import Path as _Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

__all__ = ["append_manifest_entry", "compute_run_id", "manifest_header_template"]

# Marker used to detect whether a header is already present. Bumping the
# version suffix triggers a one-time header rewrite on next call.
_HEADER_MARKER = "<!-- manifest:header:v1 -->"

# Libraries reported in the environment signature. Kept in sync with
# requirements.txt (PATCH m-05 / F22).
_TRACKED_LIBS = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "joblib",
    "prophet",
    "matplotlib",
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _get_git_sha(repo_root: Union[str, _Path]) -> str:
    """Return the short git SHA of the current HEAD, or "no-git" on failure.

    We use a subprocess call rather than pulling in GitPython because the
    dependency footprint of the pipeline is intentionally minimal
    (see requirements.txt).  A 1-second timeout keeps the call cheap.
    """
    try:
        out = _subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=_subprocess.DEVNULL,
            timeout=1.0,
        )
        return out.decode("utf-8", errors="replace").strip() or "no-git"
    except Exception:
        return "no-git"


def _get_git_dirty(repo_root: Union[str, _Path]) -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        out = _subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            stderr=_subprocess.DEVNULL,
            timeout=1.0,
        )
        return bool(out.decode("utf-8", errors="replace").strip())
    except Exception:
        return False


def _get_lib_version(name: str) -> str:
    """Return the installed version of ``name`` or ``"n/a"`` if absent."""
    try:
        # importlib.metadata works without importing the package itself
        # (cheaper + avoids triggering heavy side-effects like prophet's
        # compile-on-import).
        from importlib import metadata as _md  # type: ignore
        return _md.version(name)
    except Exception:
        return "n/a"


def _get_env_signature() -> Dict[str, str]:
    """Return a dict of {lib: version} + python/platform."""
    env: Dict[str, str] = {
        "python": _sys.version.split()[0],
        "platform": _platform.platform(),
    }
    for lib in _TRACKED_LIBS:
        env[lib] = _get_lib_version(lib)
    return env


def _hash_file(path: Union[str, _Path], chunk: int = 1 << 20) -> str:
    """
    Return the SHA-256 hex digest of a file's bytes.  Streams in 1 MiB
    chunks to avoid loading large CSVs in memory.  Returns ``"missing"``
    if the file is absent and ``"err"`` on any I/O error.
    """
    p = _Path(path)
    if not p.exists():
        return "missing"
    try:
        h = _hashlib.sha256()
        with p.open("rb") as f:
            while True:
                buf = f.read(chunk)
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest()
    except Exception:
        return "err"


def compute_run_id(
    *,
    config: Mapping[str, Any],
    git_sha: str,
    input_paths: Optional[Iterable[Union[str, _Path]]] = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    PATCH TASK-32 (audit_tmp MAJ-11, 2026-04-26)
    ────────────────────────────────────────────────────────────────────
    Compute a deterministic, reproducible ``run_id`` from:
      1. ``config`` — a JSON-serialisable mapping (CONFIG dict, with
         ``sort_keys=True`` so insertion-order doesn't perturb the hash)
      2. ``git_sha`` — short SHA from :func:`_get_git_sha`
      3. ``input_paths`` — iterable of input CSVs / model files whose
         SHA-256 is folded in (defaults to no input fingerprint)
      4. ``extras`` — any additional key/value pairs (also JSON-sorted)

    The resulting ID is the first 16 hex chars of SHA-256, which gives
    a 64-bit collision space — sufficient for distinguishing every run
    a research project will ever produce while remaining grep-friendly.

    Identical inputs ⇒ identical run_id, on any machine, any day.
    This enables manifest entries to be deduplicated and re-runs to be
    detected at audit time.
    """
    h = _hashlib.sha256()
    try:
        h.update(_json.dumps(dict(config), sort_keys=True, default=str)
                 .encode("utf-8", errors="replace"))
    except Exception:
        h.update(repr(config).encode("utf-8", errors="replace"))
    h.update(b"\x00git\x00")
    h.update((git_sha or "no-git").encode("utf-8"))
    if input_paths:
        h.update(b"\x00inputs\x00")
        for ip in sorted(str(p) for p in input_paths):
            h.update(ip.encode("utf-8"))
            h.update(b":")
            h.update(_hash_file(ip).encode("utf-8"))
            h.update(b"\n")
    if extras:
        h.update(b"\x00extras\x00")
        try:
            h.update(_json.dumps(dict(extras), sort_keys=True, default=str)
                     .encode("utf-8", errors="replace"))
        except Exception:
            h.update(repr(extras).encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


def _fmt_number(x: Any, prec: int = 3) -> str:
    """Format a scalar for markdown; handles None/NaN gracefully."""
    if x is None:
        return "—"
    try:
        import math
        if isinstance(x, float) and math.isnan(x):
            return "—"
    except Exception:
        pass
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


def manifest_header_template() -> str:
    """Return the markdown header written once per file."""
    return (
        f"# IDS-SL Run Manifest\n"
        f"{_HEADER_MARKER}\n\n"
        "This file is appended automatically by `utils_manifest.py` "
        "at the end of every evaluation run (PATCH m-06 / F23).\n"
        "Each section below is a single experimental run: timestamp, "
        "version descriptor, source CSV, key metrics, environment.\n\n"
        "**Do not edit entries manually** — they encode the reproducibility "
        "trail of the paper. New runs append at the bottom.\n\n"
        "---\n\n"
    )


def _ensure_header(path: _Path) -> None:
    """Create the file and write the header if missing."""
    if path.exists():
        try:
            head = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            head = ""
        if _HEADER_MARKER in head:
            return
    # Either file missing or header absent → (re)create header at top.
    # To stay append-only semantically we keep any prior content and
    # prepend only if the file is empty.
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(manifest_header_template(), encoding="utf-8")
    else:
        # Legacy content present — add marker + header at the top once.
        prev = path.read_text(encoding="utf-8", errors="replace")
        path.write_text(
            manifest_header_template()
            + "<!-- legacy content preserved below -->\n\n"
            + prev,
            encoding="utf-8",
        )


def _render_entry(
    *,
    metrics: Mapping[str, Any],
    version_name: str,
    csv_path: Union[str, _Path],
    git_sha: str,
    git_dirty: bool,
    env: Mapping[str, str],
    extras: Optional[Mapping[str, Any]] = None,
    timestamp: Optional[_dt.datetime] = None,
    run_id: Optional[str] = None,
) -> str:
    """Format a single run block in markdown."""
    # Python 3.12+ : datetime.utcnow() is deprecated — use timezone-aware UTC.
    ts = timestamp or _dt.datetime.now(_dt.timezone.utc)
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")

    # --- Metrics table: stable column order for easy diffing ---
    metric_order = [
        ("threshold",          "Operational threshold"),
        ("f1_binary",          "F1 — binary"),
        ("f1_micro_pure",      "F1 — micro (pure)"),
        ("f1_macro_pure",      "F1 — macro (pure)"),
        ("f1_coverage",        "F1 — coverage-weighted"),
        ("f1_ttd",             "F1 — TTD-penalized"),
        ("precision_window",   "Precision (window)"),
        ("recall_binary",      "Recall — binary"),
        ("recall_coverage",    "Recall — coverage"),
        ("fpr_pct",            "FPR (%)"),
        ("mcc",                "MCC"),
        ("tpr_window",         "TPR (window-level)"),
        ("fpr_window",         "FPR (window-level)"),
        ("median_ttd_min",     "Median TTD (min)"),
        ("n_detected_attacks", "Detected attacks"),
        ("n_attacks",          "Total attacks"),
    ]

    rows = ["| Metric | Value |", "|---|---|"]
    for k, label in metric_order:
        if k in metrics:
            rows.append(f"| {label} | `{_fmt_number(metrics[k])}` |")
    # Any additional metrics passed in get appended too, so nothing is lost.
    extra_metric_keys = [k for k in metrics if k not in dict(metric_order)]
    for k in sorted(extra_metric_keys):
        rows.append(f"| {k} | `{_fmt_number(metrics[k])}` |")
    metrics_table = "\n".join(rows)

    # --- Environment block ---
    env_rows = ["| Component | Version |", "|---|---|"]
    for k in ("python", "platform", *_TRACKED_LIBS):
        env_rows.append(f"| {k} | `{env.get(k, 'n/a')}` |")
    env_table = "\n".join(env_rows)

    # --- Extras block (free-form key/value) ---
    extras_block = ""
    if extras:
        extras_rows = ["| Key | Value |", "|---|---|"]
        for k, v in extras.items():
            if isinstance(v, (dict, list)):
                v = _json.dumps(v, ensure_ascii=False, default=str)
            extras_rows.append(f"| {k} | `{v}` |")
        extras_block = "\n### Run-specific details\n\n" + "\n".join(extras_rows) + "\n"

    dirty_flag = " *(dirty working tree)*" if git_dirty else ""
    csv_display = str(csv_path)
    # PATCH TASK-32 (audit_tmp MAJ-11): include deterministic run_id in
    # the entry header for grep-able cross-referencing.
    run_id_line = f"- **Run ID:** `{run_id}`\n" if run_id else ""

    return (
        f"## {ts_str} — `{version_name}`\n\n"
        f"{run_id_line}"
        f"- **Source CSV:** `{csv_display}`\n"
        f"- **Git SHA:** `{git_sha}`{dirty_flag}\n\n"
        f"### Key metrics\n\n"
        f"{metrics_table}\n\n"
        f"### Environment\n\n"
        f"{env_table}\n"
        f"{extras_block}\n"
        f"---\n\n"
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def append_manifest_entry(
    metrics: Mapping[str, Any],
    version_name: str,
    csv_path: Union[str, _Path],
    project_root: Union[str, _Path],
    *,
    extras: Optional[Mapping[str, Any]] = None,
    manifest_name: str = "MANIFEST.md",
    timestamp: Optional[_dt.datetime] = None,
    config: Optional[Mapping[str, Any]] = None,
    input_paths: Optional[Iterable[Union[str, _Path]]] = None,
) -> Optional[_Path]:
    """Append a single run block to ``project_root/manifest_name``.

    Parameters
    ----------
    metrics : Mapping[str, Any]
        Evaluation metrics for this run. Known keys (see
        ``_render_entry`` table) are rendered in a stable column order;
        unknown keys are appended alphabetically.
    version_name : str
        VERSION_NAME descriptor (e.g.
        ``"trained_models_v9_v9_v4s_v3_v5"``).
    csv_path : str | Path
        Absolute path of the detection results CSV that was evaluated.
    project_root : str | Path
        Directory in which ``MANIFEST.md`` lives (normally the script
        directory of ``evaluate_injection_v2.py``).
    extras : Mapping[str, Any] | None, optional
        Free-form key/value pairs appended at the end of the entry
        (e.g. ``{"fusion_op": "CBF", "lambda_decay": 0.85}``).
    manifest_name : str, optional
        File name inside ``project_root``. Default ``"MANIFEST.md"``.
    timestamp : datetime | None, optional
        Override the timestamp (useful for testing). Defaults to UTC now.

    Returns
    -------
    Path | None
        The manifest path on success, ``None`` on any I/O failure.
        Never raises — failures are logged as warnings.
    """
    try:
        root = _Path(project_root).resolve()
        manifest_path = root / manifest_name
        _ensure_header(manifest_path)

        env = _get_env_signature()
        sha = _get_git_sha(root)
        dirty = _get_git_dirty(root) if sha != "no-git" else False

        # PATCH TASK-32 (audit_tmp MAJ-11): compute deterministic run_id
        # iff the caller supplied a config — otherwise leave it absent
        # for backward compatibility.
        run_id: Optional[str] = None
        if config is not None:
            try:
                run_id = compute_run_id(
                    config=config,
                    git_sha=sha,
                    input_paths=input_paths,
                    extras=extras,
                )
            except Exception as _exc_rid:
                _warnings.warn(
                    f"[utils_manifest] run_id computation failed "
                    f"({_exc_rid!r}) — entry will not include a run_id.",
                    stacklevel=2,
                )

        entry = _render_entry(
            metrics=metrics,
            version_name=version_name,
            csv_path=csv_path,
            git_sha=sha,
            git_dirty=dirty,
            env=env,
            extras=extras,
            timestamp=timestamp,
            run_id=run_id,
        )
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(entry)
        return manifest_path
    except Exception as exc:  # pragma: no cover — defensive
        _warnings.warn(
            f"[utils_manifest] manifest write failed ({exc!r}); "
            "continuing without it — pipeline is unaffected.",
            stacklevel=2,
        )
        return None


# ----------------------------------------------------------------------
# Minimal self-test (run ``python utils_manifest.py`` manually)
# ----------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        fake_metrics = {
            "threshold": 0.20,
            "f1_binary": 0.87,
            "f1_coverage": 0.83,
            "precision_window": 0.91,
            "recall_binary": 0.82,
            "fpr_pct": 0.58,
            "mcc": 0.79,
            "n_detected_attacks": 14,
            "n_attacks": 17,
            "median_ttd_min": 6.0,
        }
        p = append_manifest_entry(
            metrics=fake_metrics,
            version_name="trained_models_v9_test",
            csv_path="/tmp/detection_results_INJECTED.csv",
            project_root=tmp,
            extras={"fusion_op": "CBF", "lambda_decay": 0.85,
                    "um_enabled": False},
        )
        print(f"Wrote: {p}")
        # Append a second entry to confirm append-only semantics.
        p2 = append_manifest_entry(
            metrics={**fake_metrics, "f1_binary": 0.88},
            version_name="trained_models_v9_test_run2",
            csv_path="/tmp/detection_results_INJECTED_run2.csv",
            project_root=tmp,
        )
        print(f"Appended: {p2}")
        print("\n--- MANIFEST.md ---\n")
        print(_Path(p2).read_text(encoding="utf-8"))
