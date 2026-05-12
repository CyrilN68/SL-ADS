"""tests/test_config_and_sidecar.py

Configuration schema + threshold-sidecar I/O round-trip tests.

This file enforces two invariants required by the USENIX Security
Artifact Evaluation "Functional" criterion (every key result must be
reproducible from the artifact):

  1. ``sl_ads.config.CONFIG`` exposes the *complete* set of keys the
     pipeline reads at runtime, with values in the documented ranges.
  2. The threshold sidecar JSON written by ``train_v10`` and read by
     ``paths.get_decision_threshold`` round-trips correctly under all
     fallback paths (sidecar present / missing / malformed).

Phase H — added 2026-04-29.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sl_ads.config import CONFIG  # noqa: E402
from sl_ads.paths import (  # noqa: E402
    get_decision_threshold,
    get_decision_variable,
    get_detection_col,
    get_detection_col_fused,
    get_threshold_sidecar_path,
    resolve_threshold_sidecar_path,
    get_fusion_mode_for_run,
    validate_threshold_sidecar_config,
)


# ────────────────────────────────────────────────────────────────────────
# 1. CONFIG schema invariants
# ────────────────────────────────────────────────────────────────────────
class TestConfigSchema:
    """Every key the pipeline relies on must be present and well-typed."""

    @pytest.mark.parametrize("key,expected_type", [
        # Core SL parameters
        ("LAMBDA_DECAY",                   (int, float)),
        ("SL_PARAM_K",                     (int, float)),
        ("SL_PRIOR_A",                     (int, float, list)),
        # EVT / threshold calibration
        ("FPR_TARGET_DECISION",            (int, float)),
        ("USE_EVT_THRESHOLDS",             bool),
        # Fusion mode (audit_codex MAJ-09)
        ("INTER_METHOD_FUSION",            str),
        ("FUSION_METHOD_GROUPS",           list),
        ("THRESHOLD_CALIBRATION_FUSION_MODES", list),
        ("WBF_WEIGHT_MODE",                str),
        # Ageing + conflict
        ("CONFLICT_ALPHA",                 (int, float)),
        # Audit_codex Phase G additions
        ("SBN_NOVELTY_U_RAW_THRESHOLD",    (int, float)),
        ("CALIB_BIJECTION_FLOOR_TOL",      (int, float)),
        ("CALIB_AGEING_WIN_FRACTION",      (int, float)),
        ("CALIB_SPARSITY_CUTOFF",          (int, float)),
        ("IF_CONTAMINATION_DEFAULT",       (int, float)),
        ("IF_CONTAMINATION_LADDER",        list),
        ("CESNET_TIMESTAMP_MODE",          str),
        ("CESNET_TIMESTAMP_ANCHOR",        str),
        # Window / dataset
        ("WINDOW_SIZE",                    int),
        ("VERSION_NAME",                   str),
    ])
    def test_required_key_present_and_typed(self, key, expected_type):
        assert key in CONFIG, f"CONFIG missing required key: {key}"
        v = CONFIG[key]
        assert isinstance(v, expected_type), (
            f"CONFIG[{key!r}] = {v!r} (type {type(v).__name__}); "
            f"expected one of {expected_type}"
        )

    @pytest.mark.parametrize("key,lo,hi", [
        ("LAMBDA_DECAY",                0.0, 1.0),
        ("FPR_TARGET_DECISION",         0.0, 1.0),
        ("SBN_NOVELTY_U_RAW_THRESHOLD", 0.0, 1.0),
        ("CALIB_AGEING_WIN_FRACTION",   0.0, 1.0),
        ("IF_CONTAMINATION_DEFAULT",    0.0, 1.0),
        ("CALIB_BIJECTION_FLOOR_TOL",   0.0, 1.0),
    ])
    def test_probability_keys_in_unit_interval(self, key, lo, hi):
        v = CONFIG.get(key)
        assert v is not None and lo <= float(v) <= hi, (
            f"CONFIG[{key!r}] = {v!r} not in [{lo}, {hi}]"
        )

    @pytest.mark.parametrize("key,lo,hi", [
        ("EVT_Q_SUSP", 0.0, 1.0),
        ("EVT_Q_ATK",  0.0, 1.0),
    ])
    def test_optional_evt_keys_when_set(self, key, lo, hi):
        """EVT_Q_* keys may be None (deferred to library default).
        When non-None they must be probabilities."""
        v = CONFIG.get(key)
        if v is None:
            return  # OK — optional
        assert lo <= float(v) <= hi, f"CONFIG[{key!r}]={v!r} not in [{lo},{hi}]"

    def test_inter_method_fusion_in_allowed_set(self):
        assert CONFIG["INTER_METHOD_FUSION"] in (
            "wbf", "abf", "cbf", "bcf", "ccf", "minbf", "maxbf", "hierarchical"
        )

    def test_threshold_calibration_modes_include_wbf_and_abf(self):
        modes = set(CONFIG["THRESHOLD_CALIBRATION_FUSION_MODES"])
        assert {"wbf", "abf"}.issubset(modes)

    def test_method_groups_are_extensible(self):
        groups = CONFIG["FUSION_METHOD_GROUPS"]
        names = {g["name"] for g in groups}
        assert {"prophet", "reconstruction"}.issubset(names)
        assert all("metric_types" in g and "output_key" in g for g in groups)

    def test_cesnet_timestamp_mode_in_allowed_set(self):
        assert CONFIG["CESNET_TIMESTAMP_MODE"] in (
            "fabricated_warning", "fabricated_silent", "reject"
        )

    def test_if_contamination_default_inside_ladder(self):
        """Headline contamination must be a member of the sensitivity ladder
        (audit_codex CRIT-03)."""
        ladder = CONFIG["IF_CONTAMINATION_LADDER"]
        default = CONFIG["IF_CONTAMINATION_DEFAULT"]
        assert default in ladder, (
            f"IF_CONTAMINATION_DEFAULT={default} not in IF_CONTAMINATION_LADDER={ladder}"
        )

    def test_lambda_decay_canonical_value(self):
        """Default λ=0.85 is the canonical value for the published results."""
        assert abs(CONFIG["LAMBDA_DECAY"] - 0.85) < 1e-9


# ────────────────────────────────────────────────────────────────────────
# 2. Threshold sidecar I/O round-trip
# ────────────────────────────────────────────────────────────────────────
class TestThresholdSidecarRoundTrip:
    """Write a fake threshold sidecar JSON, then read it back via the
    public helpers in ``sl_ads.paths``.  The round-trip must preserve
    threshold value and decision variable; missing/malformed sidecars
    must fall back gracefully without crashing the pipeline."""

    def _build_minimal_config(self, tmp_path, version="test_unique_xyz"):
        return {
            "VERSION_NAME": version,
            "EVAL": {"DECISION_THRESHOLD": 0.20},  # fallback threshold
        }

    def test_sidecar_path_construction(self, tmp_path, monkeypatch):
        cfg = self._build_minimal_config(tmp_path)
        # We expect the path to contain the version name + "_threshold.json"
        monkeypatch.chdir(tmp_path)
        path = get_threshold_sidecar_path(cfg, up_levels=0)
        assert "test_unique_xyz" in path
        assert path.endswith("_threshold.json")

    def test_round_trip_threshold_value(self, tmp_path, monkeypatch):
        cfg = self._build_minimal_config(tmp_path, "test_rt_xyz")
        monkeypatch.chdir(tmp_path)
        sidecar_path = get_threshold_sidecar_path(cfg, up_levels=0)
        Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "decision_threshold": 0.42,
            "decision_variable":  "proj_atk",
            "fpr_target":         0.001,
        }
        Path(sidecar_path).write_text(json.dumps(payload), encoding="utf-8")

        thr = get_decision_threshold(cfg, up_levels=0)
        var = get_decision_variable(cfg, up_levels=0)
        col = get_detection_col(cfg, up_levels=0)

        assert abs(thr - 0.42) < 1e-9
        assert var == "proj_atk"
        assert col == "FINAL_SYSTEM_CBF_proj_atk"

    def test_mode_specific_sidecar_preferred_over_legacy(self, tmp_path, monkeypatch):
        cfg = {
            "VERSION_NAME": "test_mode_specific_xyz",
            "EVAL": {"DECISION_THRESHOLD": 0.20},
            "INTER_METHOD_FUSION": "abf",
            "WBF_WEIGHT_MODE": "uniform",
            "LAMBDA_DECAY": 0.85,
            "BALANCE_RATIO": 1.0,
        }
        monkeypatch.chdir(tmp_path)
        legacy_path = get_threshold_sidecar_path(cfg, up_levels=0)
        abf_path = get_threshold_sidecar_path(cfg, up_levels=0, fusion_mode="abf")
        Path(legacy_path).parent.mkdir(parents=True, exist_ok=True)
        Path(legacy_path).write_text(json.dumps({
            "decision_threshold": 0.11,
            "decision_variable": "proj_atk",
            "fusion_mode_at_calibration": "wbf",
            "wbf_weight_mode": "uniform",
            "lambda_decay": 0.85,
            "balance_ratio": 1.0,
            "cd_alpha_attack": 1.0,
        }), encoding="utf-8")
        Path(abf_path).write_text(json.dumps({
            "decision_threshold": 0.22,
            "decision_variable": "proj_atk",
            "fusion_mode_at_calibration": "abf",
            "wbf_weight_mode": "uniform",
            "lambda_decay": 0.85,
            "balance_ratio": 1.0,
            "cd_alpha_attack": 1.0,
        }), encoding="utf-8")

        assert resolve_threshold_sidecar_path(cfg, up_levels=0) == abf_path
        assert get_decision_threshold(cfg, up_levels=0) == 0.22

        cfg["INTER_METHOD_FUSION"] = "wbf"
        wbf_path = get_threshold_sidecar_path(cfg, up_levels=0, fusion_mode="wbf")
        Path(wbf_path).write_text(json.dumps({
            "decision_threshold": 0.33,
            "decision_variable": "proj_atk",
            "fusion_mode_at_calibration": "wbf",
            "wbf_weight_mode": "uniform",
            "lambda_decay": 0.85,
            "balance_ratio": 1.0,
            "cd_alpha_attack": 1.0,
        }), encoding="utf-8")
        assert resolve_threshold_sidecar_path(cfg, up_levels=0) == wbf_path
        assert get_decision_threshold(cfg, up_levels=0) == 0.33

    def test_round_trip_b_atk_variable(self, tmp_path, monkeypatch):
        cfg = self._build_minimal_config(tmp_path, "test_batk_xyz")
        monkeypatch.chdir(tmp_path)
        sidecar_path = get_threshold_sidecar_path(cfg, up_levels=0)
        Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)
        Path(sidecar_path).write_text(json.dumps({
            "decision_threshold": 0.10,
            "decision_variable":  "b_atk",
        }), encoding="utf-8")
        assert abs(get_decision_threshold(cfg, up_levels=0) - 0.10) < 1e-9
        assert get_decision_variable(cfg, up_levels=0) == "b_atk"
        assert get_detection_col(cfg, up_levels=0) == "FINAL_SYSTEM_CBF_b_atk"

    def test_missing_sidecar_falls_back_to_config(self, tmp_path, monkeypatch):
        """No sidecar on disk → must fall back to CONFIG['EVAL']['DECISION_THRESHOLD']."""
        cfg = self._build_minimal_config(tmp_path, "test_missing_xyz")
        monkeypatch.chdir(tmp_path)
        sidecar_path = get_threshold_sidecar_path(cfg, up_levels=0)
        # Make sure no such file exists
        Path(sidecar_path).unlink(missing_ok=True)
        thr = get_decision_threshold(cfg, up_levels=0)
        assert abs(thr - 0.20) < 1e-9, (
            f"expected fallback to CONFIG['EVAL']['DECISION_THRESHOLD']=0.20, got {thr}"
        )

    def test_malformed_sidecar_falls_back_to_config(self, tmp_path, monkeypatch, capsys):
        """A malformed JSON sidecar must NOT crash the pipeline."""
        cfg = self._build_minimal_config(tmp_path, "test_malformed_xyz")
        monkeypatch.chdir(tmp_path)
        sidecar_path = get_threshold_sidecar_path(cfg, up_levels=0)
        Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)
        Path(sidecar_path).write_text("{ this is not json", encoding="utf-8")

        thr = get_decision_threshold(cfg, up_levels=0)
        # Falls back to CONFIG['EVAL']['DECISION_THRESHOLD']
        assert abs(thr - 0.20) < 1e-9
        # Fallback message printed to stdout
        captured = capsys.readouterr()
        assert "WARN" in captured.out or "fallback" in captured.out.lower()

    def test_sensitive_sidecar_config_match_passes(self, tmp_path, monkeypatch):
        cfg = {
            "VERSION_NAME": "test_sensitive_match_xyz",
            "EVAL": {"DECISION_THRESHOLD": 0.20},
            "INTER_METHOD_FUSION": "wbf",
            "WBF_WEIGHT_MODE": "uniform",
            "LAMBDA_DECAY": 0.85,
            "BALANCE_RATIO": 1.0,
        }
        monkeypatch.chdir(tmp_path)
        payload = {
            "decision_threshold": 0.33,
            "decision_variable": "proj_atk",
            "fusion_mode_at_calibration": "wbf",
            "wbf_weight_mode": "uniform",
            "lambda_decay": 0.85,
            "balance_ratio": 1.0,
            "cd_alpha_attack": 1.0,
        }
        sidecar_path = get_threshold_sidecar_path(cfg, up_levels=0)
        Path(sidecar_path).write_text(json.dumps(payload), encoding="utf-8")

        status = validate_threshold_sidecar_config(cfg, up_levels=0)
        assert status["ok"] is True
        assert get_decision_threshold(cfg, up_levels=0) == 0.33

    def test_sensitive_sidecar_config_mismatch_raises(self, tmp_path, monkeypatch):
        cfg = {
            "VERSION_NAME": "test_sensitive_mismatch_xyz",
            "EVAL": {"DECISION_THRESHOLD": 0.20},
            "INTER_METHOD_FUSION": "wbf",
            "WBF_WEIGHT_MODE": "uniform",
            "LAMBDA_DECAY": 0.90,
            "BALANCE_RATIO": 1.0,
        }
        monkeypatch.chdir(tmp_path)
        payload = {
            "decision_threshold": 0.33,
            "decision_variable": "proj_atk",
            "fusion_mode_at_calibration": "wbf",
            "wbf_weight_mode": "uniform",
            "lambda_decay": 0.85,
            "balance_ratio": 1.0,
            "cd_alpha_attack": 1.0,
        }
        sidecar_path = get_threshold_sidecar_path(cfg, up_levels=0)
        Path(sidecar_path).write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(RuntimeError, match="A1.9"):
            get_decision_threshold(cfg, up_levels=0)

    def test_sensitive_sidecar_mismatch_can_be_allowed_for_ablation(self, tmp_path, monkeypatch):
        cfg = {
            "VERSION_NAME": "test_sensitive_ablation_xyz",
            "EVAL": {"DECISION_THRESHOLD": 0.20},
            "INTER_METHOD_FUSION": "abf",
            "WBF_WEIGHT_MODE": "uniform",
            "LAMBDA_DECAY": 0.85,
            "BALANCE_RATIO": 1.0,
        }
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SL_ALLOW_THRESHOLD_FUSION_MISMATCH_FOR_ABLATION", "1")
        payload = {
            "decision_threshold": 0.33,
            "decision_variable": "proj_atk",
            "fusion_mode_at_calibration": "wbf",
            "wbf_weight_mode": "uniform",
            "lambda_decay": 0.85,
            "balance_ratio": 1.0,
            "cd_alpha_attack": 1.0,
        }
        sidecar_path = get_threshold_sidecar_path(cfg, up_levels=0)
        Path(sidecar_path).write_text(json.dumps(payload), encoding="utf-8")

        assert get_decision_threshold(cfg, up_levels=0) == 0.33


# ────────────────────────────────────────────────────────────────────────
# 3. Fusion-mode sidecar (audit_codex MAJ-09 / TASK-44)
# ────────────────────────────────────────────────────────────────────────
class TestFusionModeSidecar:
    def test_get_fusion_mode_returns_none_when_missing(self, tmp_path):
        out = get_fusion_mode_for_run(str(tmp_path))
        assert out is None

    def test_get_fusion_mode_round_trip(self, tmp_path):
        sidecar = tmp_path / "fusion_mode_at_compute_opinions.json"
        payload = {
            "column_prefix":      "FINAL_SYSTEM_CBF",
            "actual_fusion_mode": "wbf",
            "wbf_weight_mode":    "uniform",
            "balance_ratio":      1.0,
            "lambda_decay":       0.85,
        }
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
        out = get_fusion_mode_for_run(str(tmp_path))
        assert out is not None
        assert out["actual_fusion_mode"] == "wbf"
        assert out["wbf_weight_mode"] == "uniform"

    def test_get_fusion_mode_swallows_malformed_json(self, tmp_path, capsys):
        sidecar = tmp_path / "fusion_mode_at_compute_opinions.json"
        sidecar.write_text("{ broken json", encoding="utf-8")
        out = get_fusion_mode_for_run(str(tmp_path))
        assert out is None
        captured = capsys.readouterr()
        assert "MAJ-09" in captured.out or "WARN" in captured.out


# ────────────────────────────────────────────────────────────────────────
# 4. Detection-column alias (audit_codex MAJ-09 / TASK-44)
# ────────────────────────────────────────────────────────────────────────
class TestDetectionColumnAlias:
    def test_get_detection_col_fused_aliases_get_detection_col(self, tmp_path, monkeypatch):
        cfg = {"VERSION_NAME": "test_alias_xyz",
               "EVAL": {"DECISION_THRESHOLD": 0.2}}
        monkeypatch.chdir(tmp_path)
        a = get_detection_col(cfg, up_levels=0)
        b = get_detection_col_fused(cfg, up_levels=0)
        assert a == b
