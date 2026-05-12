"""
test_run_multi_seed.py — Unit tests for the multi-seed orchestrator and
aggregator.

The orchestrator's run loop spawns ``run_pipeline.py`` and is therefore
slow + environment-dependent; it is tested only via the ``--dry-run``
path (which records the command without launching it).  The aggregator
is exercised on synthetic ``results/<run_id>/`` trees that we build
in temp directories.

Tracks TASK-53 in docs/audit/audit_verification_tracker.md.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sl_ads.evaluate.run_multi_seed import (  # noqa: E402
    HEADLINE_METRICS,
    _build_pipeline_command,
    _df_to_markdown,
    _format_aggregate_md,
    _latest_run_id,
    _parse_seed_list,
    _read_run_metrics,
    _resolve_metric_column,
    aggregate,
    aggregate_runs,
    run,
)


# ════════════════════════════════════════════════════════════════════════
# Seed parsing
# ════════════════════════════════════════════════════════════════════════
class TestParseSeedList:
    def test_canonical_csv(self):
        assert _parse_seed_list("0,1,2,3,4") == [0, 1, 2, 3, 4]

    def test_whitespace_tolerated(self):
        assert _parse_seed_list(" 7 , 11 ") == [7, 11]

    def test_empty_string_returns_empty_list(self):
        assert _parse_seed_list("") == []
        assert _parse_seed_list("   ") == []

    def test_duplicates_raise(self):
        with pytest.raises(ValueError, match="Duplicate"):
            _parse_seed_list("0,1,0")

    def test_non_integer_raises(self):
        with pytest.raises(ValueError):
            _parse_seed_list("0,abc,2")


# ════════════════════════════════════════════════════════════════════════
# Aggregator core
# ════════════════════════════════════════════════════════════════════════
class TestAggregateRuns:
    @pytest.fixture
    def per_run_df(self):
        return pd.DataFrame({
            "seed": [0, 1, 2, 3, 4],
            "F1_micro":   [0.78, 0.79, 0.78, 0.77, 0.79],
            "F1_macro":   [0.88, 0.89, 0.87, 0.88, 0.88],
            "MCC":        [0.77, 0.78, 0.76, 0.77, 0.77],
            "FAR_pct":    [0.30, 0.32, 0.29, 0.31, 0.30],
            "n_detected": [14, 14, 14, 14, 14],
        })

    def test_returns_one_row_per_metric(self, per_run_df):
        out = aggregate_runs(per_run_df)
        assert set(out["metric"]) == set(HEADLINE_METRICS)
        assert len(out) == len(HEADLINE_METRICS)

    def test_mean_matches_numpy(self, per_run_df):
        out = aggregate_runs(per_run_df)
        f1m = out[out["metric"] == "F1_micro"].iloc[0]
        expected = float(per_run_df["F1_micro"].mean())
        assert abs(f1m["mean"] - expected) < 1e-12

    def test_std_uses_bessel_correction(self, per_run_df):
        out = aggregate_runs(per_run_df)
        f1m = out[out["metric"] == "F1_micro"].iloc[0]
        expected = float(per_run_df["F1_micro"].std(ddof=1))
        assert abs(f1m["std"] - expected) < 1e-12

    def test_min_max_match(self, per_run_df):
        out = aggregate_runs(per_run_df)
        f1m = out[out["metric"] == "F1_micro"].iloc[0]
        assert f1m["min"] == per_run_df["F1_micro"].min()
        assert f1m["max"] == per_run_df["F1_micro"].max()

    def test_ci_brackets_mean(self, per_run_df):
        out = aggregate_runs(per_run_df)
        f1m = out[out["metric"] == "F1_micro"].iloc[0]
        # The BCa CI on five identical-distribution values must contain
        # the sample mean.
        assert f1m["ci_low"] <= f1m["mean"] <= f1m["ci_high"]
        assert f1m["ci_low"] < f1m["ci_high"]

    def test_constant_column_yields_zero_std(self, per_run_df):
        out = aggregate_runs(per_run_df)
        nd = out[out["metric"] == "n_detected"].iloc[0]
        assert nd["std"] == 0.0
        assert nd["mean"] == 14.0

    def test_missing_metric_column_skipped(self):
        df = pd.DataFrame({"seed": [0, 1], "F1_micro": [0.78, 0.79]})
        out = aggregate_runs(df)
        assert "F1_micro" in set(out["metric"])
        # Other HEADLINE_METRICS that are absent must not appear.
        assert set(out["metric"]) == {"F1_micro"}

    def test_all_nan_metric_returns_nan_row(self):
        df = pd.DataFrame({"seed": [0, 1], "F1_micro": [float("nan"), float("nan")]})
        out = aggregate_runs(df)
        row = out.iloc[0]
        assert row["n"] == 0
        assert math.isnan(row["mean"])

    def test_single_run_yields_zero_std_no_crash(self):
        df = pd.DataFrame({"seed": [0], "F1_micro": [0.78]})
        out = aggregate_runs(df)
        row = out[out["metric"] == "F1_micro"].iloc[0]
        assert row["n"] == 1
        assert row["std"] == 0.0


# ════════════════════════════════════════════════════════════════════════
# Read-from-disk helpers
# ════════════════════════════════════════════════════════════════════════
class TestReadRunMetrics:
    def test_missing_directory_returns_nan_dict(self, tmp_path):
        out = _read_run_metrics(tmp_path / "results", "no_such_run")
        assert set(out) == set(HEADLINE_METRICS)
        assert all(math.isnan(v) for v in out.values())

    def test_missing_csv_returns_nan_dict(self, tmp_path):
        (tmp_path / "results" / "rid").mkdir(parents=True)
        out = _read_run_metrics(tmp_path / "results", "rid")
        assert all(math.isnan(v) for v in out.values())

    def test_picks_aggregated_row_when_present(self, tmp_path):
        run_dir = tmp_path / "results" / "rid" / "evaluation"
        run_dir.mkdir(parents=True)
        pd.DataFrame([
            {"attack": "A", "f1_micro": 0.5, "f1_macro": 0.5, "mcc": 0.5,
             "far_pct": 1.0, "n_detected": 1},
            {"attack": "__ALL__", "f1_micro": 0.78, "f1_macro": 0.88,
             "mcc": 0.77, "far_pct": 0.3, "n_detected": 14},
        ]).to_csv(run_dir / "eval_detection_summary.csv", index=False)
        out = _read_run_metrics(tmp_path / "results", "rid")
        assert out["F1_micro"] == 0.78
        assert out["F1_macro"] == 0.88
        assert out["MCC"] == 0.77
        assert out["n_detected"] == 14.0

    def test_falls_back_to_mean_when_no_aggregated_row(self, tmp_path):
        run_dir = tmp_path / "results" / "rid" / "evaluation"
        run_dir.mkdir(parents=True)
        pd.DataFrame([
            {"attack": "A", "f1_micro": 0.7, "n_detected": 1},
            {"attack": "B", "f1_micro": 0.9, "n_detected": 1},
        ]).to_csv(run_dir / "eval_detection_summary.csv", index=False)
        out = _read_run_metrics(tmp_path / "results", "rid")
        assert out["F1_micro"] == 0.8       # mean
        assert out["n_detected"] == 2.0    # sum (not mean)

    def test_alias_resolution_handles_legacy_names(self):
        df = pd.DataFrame({"fpr_pct": [0.3]})
        # Legacy column "fpr_pct" should map to canonical FAR_pct.
        assert _resolve_metric_column(df, "FAR_pct") == "fpr_pct"


# ════════════════════════════════════════════════════════════════════════
# aggregate() end-to-end on synthetic trees
# ════════════════════════════════════════════════════════════════════════
class TestAggregateE2E:
    def _make_run(self, root: Path, run_id: str, seed: int, f1_micro: float,
                  finished_utc: str = "2026-01-01T00:00:00Z"):
        d = root / "results" / run_id / "evaluation"
        d.mkdir(parents=True)
        pd.DataFrame([{
            "attack": "__ALL__",
            "f1_micro": f1_micro,
            "f1_macro": f1_micro + 0.10,
            "mcc": f1_micro - 0.01,
            "far_pct": 0.30,
            "n_detected": 14,
        }]).to_csv(d / "eval_detection_summary.csv", index=False)
        (root / "results" / run_id / "_run_manifest.json").write_text(
            json.dumps({"summary": {"finished_utc": finished_utc}}),
            encoding="utf-8",
        )

    def test_three_runs_aggregated(self, tmp_path):
        for i, seed in enumerate([0, 1, 2]):
            self._make_run(tmp_path, f"rid{seed}", seed, 0.78 + 0.005 * i)
        per_run, agg = aggregate(
            run_ids=[f"rid{s}" for s in [0, 1, 2]],
            project_root=tmp_path,
            seeds=[0, 1, 2],
            output_dir=tmp_path / "_out",
        )
        assert len(per_run) == 3
        assert per_run["seed"].tolist() == [0, 1, 2]
        f1m = agg[agg["metric"] == "F1_micro"].iloc[0]
        assert abs(f1m["mean"] - 0.785) < 1e-9
        assert (tmp_path / "_out" / "multi_seed_per_run.csv").is_file()
        assert (tmp_path / "_out" / "multi_seed_aggregate.csv").is_file()
        assert (tmp_path / "_out" / "multi_seed_report.md").is_file()

    def test_aggregate_skips_runs_with_missing_csv(self, tmp_path):
        self._make_run(tmp_path, "rid0", 0, 0.78)
        # rid1 directory exists but has no eval CSV.
        (tmp_path / "results" / "rid1").mkdir(parents=True)
        per_run, agg = aggregate(
            run_ids=["rid0", "rid1"],
            project_root=tmp_path,
            seeds=[0, 1],
        )
        # rid0 contributes one finite value, rid1 is all-NaN.
        f1m = agg[agg["metric"] == "F1_micro"].iloc[0]
        assert f1m["n"] == 1
        assert f1m["mean"] == 0.78

    def test_seeds_length_mismatch_raises(self, tmp_path):
        self._make_run(tmp_path, "rid0", 0, 0.78)
        with pytest.raises(ValueError, match="match"):
            aggregate(["rid0"], project_root=tmp_path, seeds=[0, 1])


# ════════════════════════════════════════════════════════════════════════
# Latest-run-id tie break
# ════════════════════════════════════════════════════════════════════════
class TestLatestRunId:
    def test_picks_most_recent_finished_utc(self, tmp_path):
        for name, ts in [("a", "2026-01-01"), ("b", "2026-01-03"),
                         ("c", "2026-01-02")]:
            (tmp_path / name).mkdir()
            (tmp_path / name / "_run_manifest.json").write_text(
                json.dumps({"summary": {"finished_utc": ts}}),
                encoding="utf-8",
            )
        assert _latest_run_id(tmp_path) == "b"

    def test_returns_none_when_no_runs(self, tmp_path):
        assert _latest_run_id(tmp_path) is None

    def test_skips_subdirs_without_manifest(self, tmp_path):
        (tmp_path / "a").mkdir()  # no manifest
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "_run_manifest.json").write_text(
            json.dumps({"summary": {"finished_utc": "2026-01-01"}}),
            encoding="utf-8",
        )
        assert _latest_run_id(tmp_path) == "b"


# ════════════════════════════════════════════════════════════════════════
# Pipeline command construction
# ════════════════════════════════════════════════════════════════════════
class TestBuildPipelineCommand:
    def test_includes_dataset(self):
        cmd = _build_pipeline_command("RedeRio", None, None, Path("/p"))
        assert "--dataset" in cmd and "RedeRio" in cmd

    def test_includes_from_and_to_step_when_provided(self):
        cmd = _build_pipeline_command("RedeRio", "opinions", "audit",
                                      Path("/p"))
        assert "--from-step" in cmd and "opinions" in cmd
        assert "--to-step" in cmd and "audit" in cmd

    def test_omits_step_flags_when_none(self):
        cmd = _build_pipeline_command("RedeRio", None, None, Path("/p"))
        assert "--from-step" not in cmd
        assert "--to-step" not in cmd

    def test_uses_run_pipeline_py(self):
        cmd = _build_pipeline_command("RedeRio", None, None, Path("/p"))
        assert any("run_pipeline.py" in arg for arg in cmd)


# ════════════════════════════════════════════════════════════════════════
# Markdown rendering
# ════════════════════════════════════════════════════════════════════════
class TestMarkdownRendering:
    def test_df_to_markdown_handles_empty(self):
        out = _df_to_markdown(pd.DataFrame())
        assert "empty" in out

    def test_df_to_markdown_renders_pipe_table(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, float("nan")]})
        out = _df_to_markdown(df)
        assert out.count("\n") == 3      # header, separator, two rows
        assert "| a | b |" in out
        assert "NaN" in out               # NaN handled

    def test_format_aggregate_md_includes_required_sections(self):
        per_run = pd.DataFrame({"seed": [0, 1], "F1_micro": [0.78, 0.79]})
        agg = aggregate_runs(per_run)
        md = _format_aggregate_md(per_run, agg)
        assert "# Multi-seed evaluation report" in md
        assert "Per-seed metrics" in md
        assert "Aggregate (mean ± std" in md
        assert "Wu, R., Keogh, E." in md  # citation


# ════════════════════════════════════════════════════════════════════════
# run() orchestration in --dry-run mode
# ════════════════════════════════════════════════════════════════════════
class TestRunDryRun:
    def test_dry_run_does_not_execute_subprocess(self, tmp_path):
        df = run(
            seeds=[0, 1, 2],
            dataset="RedeRio",
            project_root=tmp_path,
            output_dir=tmp_path / "_out",
            dry_run=True,
        )
        assert len(df) == 3
        assert df["seed"].tolist() == [0, 1, 2]
        assert (df["dry_run"] == True).all()  # noqa: E712
        assert (df["returncode"].isna()).all()
        # CSV side-effect.
        assert (tmp_path / "_out" / "multi_seed_orchestration.csv").is_file()

    def test_dry_run_records_env_seed(self, tmp_path):
        df = run(
            seeds=[42, 99],
            dataset="RedeRio",
            project_root=tmp_path,
            output_dir=tmp_path / "_out",
            dry_run=True,
        )
        assert df["env_SL_RANDOM_SEED"].tolist() == ["42", "99"]
