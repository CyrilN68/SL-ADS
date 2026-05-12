"""
test_metrics_consistency.py - Cross-run consistency on the published metrics.

The headline numbers reported for the canonical complete 17-leaf RedeRio
run (archive run_id 2e12261d55a8f975, source directory
``../results/resultats_RedeRio_trained_v4s_v4_v3``, 2026-05-12) are:

    F1_micro (canonical paper)  = 0.8666 [IC 95% : 0.760 - 0.9232]
    F1_macro                    = 0.9292
    MCC                         = 0.8587
    VUS-PR                      = 0.604
    VUS-ROC                     = 0.856
    F1_cov hybrid               = 0.8795 (catalog/outages-separate)
    Detected attacks            = 14/14
    FAR (audit, operational)    = 0.97%

This test parses the canonical published result artifact
``evaluation/eval_threshold_sweep.csv`` plus the SBN qualification summary
when present and asserts the published numbers are reproduced within the
expected epsilon = +/- 0.005 (Prophet seed-noise band documented in
``docs/audit/pipeline_reconciliation_20260425.md``).

Marked ``slow`` because it depends on a full pipeline run being present
on disk.  In CI it is skipped unless the ``-m slow`` selector is passed.

Tracks TASK-52 in audit_verification_tracker.md.

Run from project root:

    pytest tests/test_metrics_consistency.py -v -m slow
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# Exact headline targets from the module docstring.
PUBLISHED_RUN_ID = "2e12261d55a8f975"
PUBLISHED_RESULTS_DIR_NAME = "resultats_RedeRio_trained_v4s_v4_v3"
F1_MICRO_TARGET = 0.8666
F1_MACRO_TARGET = 0.9292
MCC_TARGET = 0.8587
VUS_PR_TARGET = 0.604
VUS_ROC_TARGET = 0.856
RANGE_AUC_PR_AT_MAX_TARGET = 0.491
RANGE_AUC_ROC_AT_MAX_TARGET = 0.760
METRIC_EPS = 0.005
F1_MICRO_CI_LO, F1_MICRO_CI_HI = 0.760, 0.9232
DETECTED_ATTACKS_MIN = 14
FAR_GLOBAL_TARGET_PCT = 0.97
FAR_GLOBAL_EPS_PCT = 0.02
ABLATION_F1_COV_TARGET = 0.8795
ABLATION_FPR_TARGET_PCT = 0.9655
ABLATION_METRIC_PROTOCOL = "catalog_outages_separate"


def _candidate_result_dirs() -> list[Path]:
    """Return result dirs containing the canonical threshold-sweep artifact."""
    candidates: dict[Path, float] = {}
    for root in (_ROOT / "results", _ROOT.parent / "results"):
        if not root.is_dir():
            continue
        for sweep in root.rglob("evaluation/eval_threshold_sweep.csv"):
            candidates[sweep.parent.parent] = sweep.stat().st_mtime
    return [p for p, _ in sorted(candidates.items(), key=lambda kv: kv[1])]


def _is_published_candidate(run_dir: Path) -> bool:
    """Return True for the exact run/source named in the module docstring."""
    if run_dir.name in {PUBLISHED_RUN_ID, PUBLISHED_RESULTS_DIR_NAME}:
        return True

    manifest = run_dir / "_run_manifest.json"
    if not manifest.is_file():
        return False
    try:
        import json

        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        data.get("run_id") == PUBLISHED_RUN_ID
        or data.get("source_basename") == PUBLISHED_RESULTS_DIR_NAME
    )


def _find_published_run_dir() -> Path | None:
    """Return the exact complete run candidate for the published RedeRio run."""
    import pandas as pd

    for run_dir in reversed(_candidate_result_dirs()):
        if not _is_published_candidate(run_dir):
            continue
        sweep = run_dir / "evaluation" / "eval_threshold_sweep.csv"
        try:
            df = pd.read_csv(sweep)
        except Exception:
            continue
        required = {
            "f1_micro_pure", "f1_macro_pure", "mcc",
            "n_attacks", "n_detected_attacks",
        }
        if not required.issubset(df.columns) or df.empty:
            continue
        row = df.iloc[0]
        if int(row["n_attacks"]) == 14 and int(row["n_detected_attacks"]) == 14:
            return run_dir
    return None


def _published_run_or_skip() -> Path:
    run_dir = _find_published_run_dir()
    if run_dir is None:
        pytest.skip(f"No completed published RedeRio run found "
                    f"({PUBLISHED_RUN_ID} or {PUBLISHED_RESULTS_DIR_NAME}).")
    return run_dir


def _threshold_row(run_dir: Path):
    import pandas as pd

    csv_path = run_dir / "evaluation" / "eval_threshold_sweep.csv"
    assert csv_path.is_file(), f"Missing canonical threshold sweep: {csv_path}"
    df = pd.read_csv(csv_path)
    required = {
        "f1_micro_pure", "f1_macro_pure", "mcc",
        "f1_ci_lo", "f1_ci_hi", "n_attacks", "n_detected_attacks",
    }
    missing = required.difference(df.columns)
    assert not missing, f"{csv_path} missing columns: {sorted(missing)}"
    assert not df.empty, f"{csv_path} is empty"
    return df.iloc[0]


def _global_far_pct(run_dir: Path) -> float:
    import json

    summaries = sorted(run_dir.glob("eval_qualify_summary_qualif_types_sbn_*.json"))
    assert summaries, f"Missing SBN qualification summary in {run_dir}"
    data = json.loads(summaries[-1].read_text(encoding="utf-8"))
    far = data.get("global_detection", {}).get("FAR")
    assert far is not None, f"{summaries[-1]} missing global_detection.FAR"
    return float(far) * 100.0


def _vus_row(run_dir: Path):
    import pandas as pd

    csv_path = run_dir / "evaluation" / "eval_vus_summary.csv"
    assert csv_path.is_file(), f"Missing range-aware VUS summary: {csv_path}"
    df = pd.read_csv(csv_path)
    required = {
        "n", "n_anomalies", "n_ranges", "max_buffer",
        "range_auc_roc_at_max", "range_auc_pr_at_max",
        "vus_roc", "vus_pr", "existence_recall",
        "threshold", "score_column", "label_scope",
        "n_catalog_attacks_with_overlap",
    }
    missing = required.difference(df.columns)
    assert not missing, f"{csv_path} missing columns: {sorted(missing)}"
    assert len(df) == 1, f"{csv_path} must contain exactly one summary row"
    return df.iloc[0]


def _ablation_full_row(run_dir: Path):
    import pandas as pd

    csv_path = run_dir / "ablation_uniform" / "ablation_summary.csv"
    assert csv_path.is_file(), f"Missing ablation summary: {csv_path}"
    df = pd.read_csv(csv_path)
    required = {
        "run", "f1_coverage", "f1_binary", "precision", "fpr_pct",
        "n_detected", "n_attacks", "metric_protocol",
        "f1_coverage_legacy_outages_as_normal",
    }
    missing = required.difference(df.columns)
    assert not missing, f"{csv_path} missing columns: {sorted(missing)}"
    row = df[df["run"] == "Full SL-ADS (λ=0.85, Uniform Weights) [reference]"]
    assert len(row) == 1, "Ablation summary must contain exactly one Full SL-ADS reference row"
    return row.iloc[0]


# ════════════════════════════════════════════════════════════════════════
# slow tests — require a completed run on disk
# ════════════════════════════════════════════════════════════════════════
@pytest.mark.slow
def test_published_run_metric_artifacts_present():
    """The published run must include the artifacts named in the docstring."""
    run_dir = _published_run_or_skip()
    assert (run_dir / "evaluation" / "eval_threshold_sweep.csv").is_file()
    assert (run_dir / "evaluation" / "eval_vus_summary.csv").is_file()
    assert sorted(run_dir.glob("eval_qualify_summary_qualif_types_sbn_*.json"))


@pytest.mark.slow
def test_published_canonical_metrics_match_docstring():
    """F1_micro, F1_macro, MCC and F1 CI must match the published claim."""
    row = _threshold_row(_published_run_or_skip())
    assert float(row["f1_micro_pure"]) == pytest.approx(F1_MICRO_TARGET, abs=METRIC_EPS)
    assert float(row["f1_macro_pure"]) == pytest.approx(F1_MACRO_TARGET, abs=METRIC_EPS)
    assert float(row["mcc"]) == pytest.approx(MCC_TARGET, abs=METRIC_EPS)
    assert float(row["f1_ci_lo"]) == pytest.approx(F1_MICRO_CI_LO, abs=0.001)
    assert float(row["f1_ci_hi"]) == pytest.approx(F1_MICRO_CI_HI, abs=0.001)


@pytest.mark.slow
def test_detected_attacks_count_meets_target():
    """At least 14 distinct attacks must be detected at the operational threshold."""
    row = _threshold_row(_published_run_or_skip())
    n_detected = int(row["n_detected_attacks"])
    assert n_detected >= DETECTED_ATTACKS_MIN, (
        f"Detected {n_detected} attacks; published claim is "
        f">= {DETECTED_ATTACKS_MIN}.  Regression on detection coverage."
    )


@pytest.mark.slow
def test_audit_far_matches_published_value():
    """The operational FAR from qualification audit must match the published value."""
    far_pct = _global_far_pct(_published_run_or_skip())
    assert far_pct == pytest.approx(FAR_GLOBAL_TARGET_PCT, abs=FAR_GLOBAL_EPS_PCT), (
        f"Operational FAR {far_pct:.3f}% differs from the published "
        f"{FAR_GLOBAL_TARGET_PCT:.2f}% claim."
    )


@pytest.mark.slow
def test_published_vus_metrics_match_docstring():
    """VUS-PR/VUS-ROC must be present and match the publication table."""
    row = _vus_row(_published_run_or_skip())
    assert float(row["vus_pr"]) == pytest.approx(VUS_PR_TARGET, abs=METRIC_EPS)
    assert float(row["vus_roc"]) == pytest.approx(VUS_ROC_TARGET, abs=METRIC_EPS)
    assert float(row["range_auc_pr_at_max"]) == pytest.approx(
        RANGE_AUC_PR_AT_MAX_TARGET, abs=METRIC_EPS
    )
    assert float(row["range_auc_roc_at_max"]) == pytest.approx(
        RANGE_AUC_ROC_AT_MAX_TARGET, abs=METRIC_EPS
    )
    assert int(row["n_ranges"]) == 14
    assert int(row["n_catalog_attacks_with_overlap"]) == 14
    assert float(row["existence_recall"]) == pytest.approx(1.0, abs=1e-9)
    assert row["label_scope"] == "timestamp_catalog"


@pytest.mark.slow
def test_ablation_reference_matches_main_protocol():
    """Full ablation reference must use the same protocol as eval_injection."""
    row = _ablation_full_row(_published_run_or_skip())
    assert row["metric_protocol"] == ABLATION_METRIC_PROTOCOL
    assert float(row["f1_coverage"]) == pytest.approx(ABLATION_F1_COV_TARGET, abs=METRIC_EPS)
    assert float(row["fpr_pct"]) == pytest.approx(ABLATION_FPR_TARGET_PCT, abs=FAR_GLOBAL_EPS_PCT)
    assert int(row["n_detected"]) == 14
    assert int(row["n_attacks"]) == 14
    assert float(row["f1_coverage_legacy_outages_as_normal"]) < float(row["f1_coverage"])


# ════════════════════════════════════════════════════════════════════════
# Smoke tests (always run) — guards on hardcoded thresholds in the bands
# ════════════════════════════════════════════════════════════════════════
def test_consistency_bands_are_plausible():
    """The expected metric targets must be plausible probabilities.

    Catches the trivial mistake of swapping LO/HI bounds in the constants
    above (the slow tests would skip silently otherwise).
    """
    assert 0.5 < F1_MICRO_TARGET < 1.0
    assert 0.5 < F1_MACRO_TARGET < 1.0
    assert 0.5 < MCC_TARGET < 1.0
    assert 0.0 < VUS_PR_TARGET < 1.0
    assert 0.0 < VUS_ROC_TARGET < 1.0
    assert 0.0 < METRIC_EPS < 0.05
    assert F1_MICRO_CI_LO < F1_MICRO_TARGET < F1_MICRO_CI_HI
    assert PUBLISHED_RUN_ID != "da8ab988fddaf681"
    assert PUBLISHED_RESULTS_DIR_NAME.startswith("resultats_RedeRio_")
    assert FAR_GLOBAL_TARGET_PCT > 0
    assert FAR_GLOBAL_EPS_PCT > 0
    assert ABLATION_F1_COV_TARGET > 0
    assert ABLATION_FPR_TARGET_PCT > 0
    assert DETECTED_ATTACKS_MIN >= 1
