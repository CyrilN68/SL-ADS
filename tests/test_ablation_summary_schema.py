"""
test_ablation_summary_schema.py — Schema regression for the ablation table.

The publication ablation table (`run_ablation.py`, output
`ablation_summary.csv`) is the artefact §9bis in
`docs/review/PUBLICATION_TABLES.md`.  Its column schema is therefore part
of the publication contract: any silent rename or omission would break
the cross-validation chain in §10.4 of PUBLICATION_TABLES.md.

This test does NOT rerun the full ablation (which would take hours).
It exercises ``run_ablation.to_summary()`` and ``run_ablation.best_row()``
on a synthetic threshold sweep that reproduces the column schema, and
asserts the contract.

Tracks TASK-49 in audit_verification_tracker.md.

Run from project root:

    pytest tests/test_ablation_summary_schema.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# Columns required by §9bis of docs/review/PUBLICATION_TABLES.md.
# Any silent rename here breaks the cross-validation table at §10.4.
EXPECTED_SUMMARY_COLUMNS = [
    "run",
    "best_threshold",
    "f1_binary",
    "f1_coverage",
    "f1_ttd",
    "precision",
    "recall_binary",
    "recall_coverage",
    "fpr_pct",
    "n_detected",
    "n_attacks",
    "metric_protocol",
    "normal_windows",
    "outage_windows_excluded",
    "f1_coverage_legacy_outages_as_normal",
    "precision_legacy_outages_as_normal",
    "fpr_pct_legacy_outages_as_normal",
]


def _make_synthetic_sweep(seed: int = 42) -> pd.DataFrame:
    """Reproduce the column schema produced by run_ablation.run_threshold_sweep().

    Mirrors the dict appended at run_ablation.py:808-820 — keep this in
    sync with that function.  A schema drift here will surface as a test
    failure and force the maintainer to update §9bis cross-validation.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for thr in [0.05, 0.10, 0.15, 0.20, 0.25]:
        prec = float(rng.uniform(0.5, 0.9))
        r_bin = float(rng.uniform(0.5, 0.9))
        r_cov = float(rng.uniform(0.4, 0.8))
        rows.append({
            "threshold":       thr,
            "n_attacks":       14,
            "n_detected":      int(rng.integers(10, 14, endpoint=True)),
            "recall_binary":   round(r_bin,  4),
            "recall_coverage": round(r_cov,  4),
            "recall_ttd":      round(float(rng.uniform(2.0, 6.0)), 4),
            "precision":       round(prec,   4),
            "f1_binary":       round(2 * prec * r_bin / (prec + r_bin), 4),
            "f1_coverage":     round(2 * prec * r_cov / (prec + r_cov), 4),
            "f1_ttd":          round(float(rng.uniform(0.10, 0.40)),    4),
            "fp_windows":      int(rng.integers(0, 200)),
            "fpr_pct":         round(float(rng.uniform(0.5, 3.0)), 5),
            "metric_protocol": "catalog_outages_separate",
            "normal_windows": 12015,
            "outage_windows_excluded": 342,
            "f1_coverage_legacy_outages_as_normal": round(float(rng.uniform(0.4, 0.8)), 4),
            "precision_legacy_outages_as_normal": round(float(rng.uniform(0.4, 0.8)), 4),
            "fpr_pct_legacy_outages_as_normal": round(float(rng.uniform(0.5, 4.0)), 5),
        })
    return pd.DataFrame(rows)


def test_to_summary_returns_expected_schema():
    """to_summary() must produce exactly the publication-table column set."""
    from sl_ads.ablation.run_ablation import to_summary

    sweep_df = _make_synthetic_sweep()
    summary = to_summary("synthetic_run", sweep_df)
    assert summary is not None, "to_summary returned None for non-empty sweep"
    assert isinstance(summary, dict)
    assert set(summary.keys()) == set(EXPECTED_SUMMARY_COLUMNS), (
        "to_summary() schema drift.  Got "
        f"{sorted(summary.keys())!r}, expected {sorted(EXPECTED_SUMMARY_COLUMNS)!r}.  "
        "Update §9bis cross-validation in PUBLICATION_TABLES.md if this is intentional."
    )


def test_to_summary_preserves_value_types():
    """Each column must have the documented type contract.

    `run`              : str
    `best_threshold`   : float
    `f1_*`, `precision`, `recall_*`, `fpr_pct` : float
    `n_detected`, `n_attacks`                  : int
    """
    from sl_ads.ablation.run_ablation import to_summary

    sweep_df = _make_synthetic_sweep()
    summary = to_summary("type_check_run", sweep_df)

    assert isinstance(summary["run"], str)
    assert isinstance(summary["best_threshold"], float)
    for k in ("f1_binary", "f1_coverage", "f1_ttd",
              "precision", "recall_binary", "recall_coverage", "fpr_pct",
              "f1_coverage_legacy_outages_as_normal",
              "precision_legacy_outages_as_normal",
              "fpr_pct_legacy_outages_as_normal"):
        assert isinstance(summary[k], float), \
            f"{k}: expected float, got {type(summary[k]).__name__}"
    assert isinstance(summary["metric_protocol"], str)
    for k in ("n_detected", "n_attacks", "normal_windows", "outage_windows_excluded"):
        assert isinstance(summary[k], int), \
            f"{k}: expected int, got {type(summary[k]).__name__}"


def test_best_row_picks_max_f1_coverage():
    """best_row() must select the threshold maximising f1_coverage.

    This is the §9bis publication metric (Tatbul et al. 2018, coverage-weighted F1).
    Selecting on a different column would silently swap the table's reference.
    """
    from sl_ads.ablation.run_ablation import best_row

    df = pd.DataFrame([
        {"threshold": 0.10, "f1_coverage": 0.50, "f1_binary": 0.99,
         "f1_ttd": 0.10, "precision": 0.50, "recall_binary": 0.50,
         "recall_coverage": 0.50, "fpr_pct": 1.0, "n_detected": 10, "n_attacks": 14},
        {"threshold": 0.20, "f1_coverage": 0.80, "f1_binary": 0.40,
         "f1_ttd": 0.20, "precision": 0.50, "recall_binary": 0.50,
         "recall_coverage": 0.50, "fpr_pct": 1.0, "n_detected": 12, "n_attacks": 14},
    ])
    br = best_row(df)
    assert abs(float(br["threshold"]) - 0.20) < 1e-9, (
        "best_row() must pick the row with highest f1_coverage, "
        f"not f1_binary.  Picked threshold={br['threshold']!r}."
    )


def test_to_summary_returns_none_for_empty_sweep():
    """Defensive: empty sweep must not crash, must return None."""
    from sl_ads.ablation.run_ablation import to_summary
    assert to_summary("empty", pd.DataFrame()) is None


def test_summary_csv_columns_match_dataframe_round_trip(tmp_path):
    """Round-trip a summary dict through pandas → CSV → DataFrame and
    verify the column ordering and type contract are preserved.

    This is the protocol used at run_ablation.py:1344-1345 to write
    ablation_summary.csv.  A round-trip mismatch would mean reviewers
    open a CSV with a different schema than the dict they see in logs.
    """
    from sl_ads.ablation.run_ablation import to_summary

    sweep_df = _make_synthetic_sweep()
    summary = to_summary("round_trip_run", sweep_df)
    df = pd.DataFrame([summary])

    csv_path = tmp_path / "ablation_summary.csv"
    df.to_csv(csv_path, index=False)
    loaded = pd.read_csv(csv_path)

    assert list(loaded.columns) == list(df.columns), (
        "Column ordering changed across CSV round-trip: "
        f"in-memory={list(df.columns)} vs on-disk={list(loaded.columns)}."
    )
    # n_detected / n_attacks must come back as ints; pandas infers them
    # automatically when there's no missing data.
    for col in ("n_detected", "n_attacks", "normal_windows", "outage_windows_excluded"):
        assert pd.api.types.is_integer_dtype(loaded[col]), (
            f"Column {col!r} lost integer dtype on CSV round-trip."
        )
