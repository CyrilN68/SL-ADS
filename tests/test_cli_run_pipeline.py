"""tests/test_cli_run_pipeline.py

Smoke tests for the ``run_pipeline.py`` launcher CLI.

USENIX SecAE / ACM artifact-evaluation reviewers must be able to
reproduce every published result from the documented entry point.
This file exercises the CLI surface (argparse, --list-steps,
--dry-run, --from-step / --to-step slicing, --no-archive) without
running the heavy sub-processes — fast, deterministic, and proves
that the launcher's contract is intact.

Phase H — added 2026-04-29.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_LAUNCHER = _ROOT / "run_pipeline.py"


def _run_launcher(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Spawn ``python run_pipeline.py <args>`` and return the result.

    UTF-8 IO is forced because the launcher prints ``→`` and ``✅``
    glyphs that fail under cp1252 on Windows when pytest captures stdout.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(_LAUNCHER), *args],
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


# ────────────────────────────────────────────────────────────────────────
# 1. --list-steps
# ────────────────────────────────────────────────────────────────────────
class TestListSteps:
    @pytest.mark.parametrize("dataset,expected_count,expected_substrings", [
        ("RedeRio",             10, ["train", "evidence", "inject", "opinions",
                                     "eval_injection", "qualify_sbn", "eval_qualify",
                                     "ablation", "compare_if", "audit"]),
        ("METR-LA",             7,  ["train", "evidence", "opinions", "qualify_sbn",
                                     "eval_qualify", "ablation", "compare_if"]),
        ("GECCO-IoT",           6,  ["train", "evidence", "opinions", "qualify_sbn",
                                     "ablation", "compare_if"]),
        ("CESNET-TimeSeries24", 6,  ["train", "evidence", "opinions", "qualify_sbn",
                                     "ablation", "compare_if"]),
    ])
    def test_list_steps_per_dataset(self, dataset, expected_count, expected_substrings):
        proc = _run_launcher("--dataset", dataset, "--list-steps")
        assert proc.returncode == 0, f"non-zero exit on {dataset}: {proc.stderr}"
        # Count enumerated lines like "  N. step_name → python -m sl_ads.<X>"
        import re
        n = len(re.findall(r"^\s+\d+\.\s+\w+", proc.stdout, re.MULTILINE))
        assert n == expected_count, (
            f"{dataset}: expected {expected_count} steps, got {n}.\n"
            f"stdout:\n{proc.stdout}"
        )
        for substr in expected_substrings:
            assert substr in proc.stdout, f"{dataset}: '{substr}' missing from --list-steps"

    def test_list_steps_uses_sl_ads_module_paths(self):
        """Every dispatched step must reference the new ``sl_ads.*`` module
        path (no legacy flat path remaining)."""
        proc = _run_launcher("--list-steps")
        assert "python -m sl_ads." in proc.stdout
        # Negative: legacy paths must NOT appear
        assert "train_v10.py" not in proc.stdout
        assert "compute_opinions_v3.py" not in proc.stdout
        assert "evaluate_injection_v2.py" not in proc.stdout


# ────────────────────────────────────────────────────────────────────────
# 2. --dry-run
# ────────────────────────────────────────────────────────────────────────
class TestDryRun:
    def test_full_dry_run_marks_all_steps(self):
        proc = _run_launcher("--dry-run")
        assert proc.returncode == 0, f"non-zero exit: {proc.stderr}"
        assert "Pipeline complete" in proc.stdout
        # 10 dry-run lines for RedeRio (default)
        n_dry = proc.stdout.count("[dry-run]")
        assert n_dry >= 10, f"expected ≥10 dry-run lines, got {n_dry}"

    def test_dry_run_writes_exit_summary_json(self, tmp_path):
        """The launcher must write ``pipeline_run_summary.json`` even in
        dry-run mode — otherwise CI/audit tooling can't introspect."""
        # Use a single step to keep it fast.
        proc = _run_launcher("--dry-run", "--from-step", "train", "--to-step", "train")
        assert proc.returncode == 0
        assert "exit-summary JSON saved" in proc.stdout
        # The summary file must exist after the run.
        summary = _ROOT / "pipeline_run_summary.json"
        assert summary.exists()
        # And contain valid JSON with the expected key set.
        import json
        data = json.loads(summary.read_text(encoding="utf-8"))
        for required in ("dataset", "pipeline", "selected_steps",
                          "from_step", "to_step", "dry_run", "counts", "steps"):
            assert required in data, f"missing key '{required}' in exit-summary JSON"
        assert data["dry_run"] is True


# ────────────────────────────────────────────────────────────────────────
# 3. --from-step / --to-step slicing
# ────────────────────────────────────────────────────────────────────────
class TestStepSlicing:
    def test_from_step_only(self):
        proc = _run_launcher("--dry-run", "--from-step", "qualify_sbn")
        assert proc.returncode == 0
        # Should run qualify_sbn..audit (5 steps for RedeRio)
        n = proc.stdout.count("[dry-run]")
        assert n == 5

    def test_to_step_only(self):
        proc = _run_launcher("--dry-run", "--to-step", "evidence")
        assert proc.returncode == 0
        # Should run train..evidence (2 steps)
        n = proc.stdout.count("[dry-run]")
        assert n == 2

    def test_invalid_from_to_order_raises(self):
        """``--from-step`` must precede ``--to-step``."""
        proc = _run_launcher("--dry-run", "--from-step", "audit", "--to-step", "train")
        # argparse choices accept the values; the slicing logic should reject.
        assert proc.returncode != 0 or "doit précéder" in (proc.stdout + proc.stderr)


# ────────────────────────────────────────────────────────────────────────
# 4. --no-archive flag
# ────────────────────────────────────────────────────────────────────────
class TestNoArchive:
    def test_no_archive_skips_snapshot(self):
        """``--no-archive`` must not create a results/<run_id>/ folder."""
        proc = _run_launcher(
            "--dry-run", "--from-step", "compare_if", "--to-step", "compare_if",
            "--no-archive",
        )
        assert proc.returncode == 0
        # Dry-run + --no-archive ⇒ no archive message anywhere
        assert "[archive] outputs/ snapshotted" not in proc.stdout


class TestArchiveOutputs:
    def test_archive_uses_active_results_dir_and_refreshes_outputs(
        self, tmp_path, monkeypatch
    ):
        """The archive source must be CONFIG RESULTS_DIR, not stale outputs/."""
        import json
        import run_pipeline

        active = tmp_path / "active_results"
        active.mkdir()
        (active / "fresh.csv").write_text("run,fresh\n1,yes\n", encoding="utf-8")
        (active / "evaluation").mkdir()
        (active / "evaluation" / "eval_threshold_sweep.csv").write_text(
            "threshold,f1_micro_pure\n0.1,0.784\n",
            encoding="utf-8",
        )

        outputs = tmp_path / "outputs"
        outputs.mkdir()
        (outputs / "stale.csv").write_text("old,stale\n1,yes\n", encoding="utf-8")

        monkeypatch.setattr(run_pipeline, "BASE_DIR", str(tmp_path))
        monkeypatch.setattr(
            run_pipeline,
            "_resolve_active_results_dir",
            lambda dataset: str(active),
        )

        summary = {"counts": {"failed": 0, "aborted": 0}}
        archive_root = tmp_path / "archives"
        archived = run_pipeline._archive_outputs(
            "abc123", "RedeRio", str(archive_root), summary
        )

        assert archived == str(archive_root / "abc123")
        assert not (outputs / "stale.csv").exists()
        assert (outputs / "fresh.csv").read_text(encoding="utf-8") == "run,fresh\n1,yes\n"
        assert (archive_root / "abc123" / "fresh.csv").is_file()
        assert (archive_root / "abc123" / "evaluation" / "eval_threshold_sweep.csv").is_file()

        manifest = json.loads(
            (archive_root / "abc123" / "_run_manifest.json").read_text(encoding="utf-8")
        )
        outputs_manifest = json.loads(
            (outputs / "_run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["source_dir"] == str(active.resolve())
        assert manifest["current_outputs_dir"] == str(outputs.resolve())
        assert manifest["source_basename"] == "active_results"
        assert outputs_manifest["source_dir"] == str(active.resolve())
        assert outputs_manifest["summary"]["artifact_source_dir"] == str(active.resolve())
        assert summary["artifact_source_dir"] == str(active.resolve())
        assert summary["current_outputs_dir"] == str(outputs.resolve())

    def test_archive_keeps_existing_run_id_immutable(self, tmp_path, monkeypatch):
        """An existing historical run_id must not be overwritten silently."""
        import run_pipeline

        active = tmp_path / "active_results"
        active.mkdir()
        (active / "fresh.csv").write_text("fresh\n", encoding="utf-8")

        archive_root = tmp_path / "archives"
        existing = archive_root / "abc123"
        existing.mkdir(parents=True)
        (existing / "old.csv").write_text("old\n", encoding="utf-8")

        monkeypatch.setattr(run_pipeline, "BASE_DIR", str(tmp_path))
        monkeypatch.setattr(
            run_pipeline,
            "_resolve_active_results_dir",
            lambda dataset: str(active),
        )

        archived = run_pipeline._archive_outputs(
            "abc123", "RedeRio", str(archive_root), {"counts": {}}
        )

        assert archived == str(existing)
        assert (existing / "old.csv").read_text(encoding="utf-8") == "old\n"
        assert not (existing / "fresh.csv").exists()
        assert (tmp_path / "outputs" / "fresh.csv").is_file()
