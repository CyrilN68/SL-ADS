from __future__ import annotations

import pytest

from sl_ads.train.model_guardrails import (
    expected_model_keys_from_config,
    validate_required_model_coverage,
)


def _config(require_all: bool = True):
    return {
        "ACTIVE_METRICS": ["bytes", "packets"],
        "RECONST_RULES": [
            {"target": "bytes", "feature": "packets"},
            {"target": "tcp", "feature": "packets"},
        ],
        "REQUIRE_ALL_MODELS": require_all,
    }


def test_expected_model_keys_include_prophet_and_reconstruction():
    assert expected_model_keys_from_config(_config()) == [
        "prophet_bytes",
        "prophet_packets",
        "reconst_bytes_from_packets",
        "reconst_tcp_from_packets",
    ]


def test_validate_required_model_coverage_rejects_partial_artifact():
    models_pkg = {
        "prophet_bytes": {"type": "prophet"},
        "reconst_bytes_from_packets": {"type": "reconstruction"},
        "reconst_tcp_from_packets": {"type": "reconstruction"},
    }
    with pytest.raises(RuntimeError, match="prophet_packets"):
        validate_required_model_coverage(models_pkg, _config())


def test_validate_required_model_coverage_can_be_disabled_for_debug_only():
    models_pkg = {
        "reconst_bytes_from_packets": {"type": "reconstruction"},
    }
    validate_required_model_coverage(models_pkg, _config(require_all=False))
