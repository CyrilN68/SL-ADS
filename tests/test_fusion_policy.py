from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sl_ads.core import fusion_policy  # noqa: E402
from sl_ads.core.subjective_logic import MultinomialOpinion  # noqa: E402


def _op(b_safe: float, b_susp: float, b_atk: float, u: float):
    return MultinomialOpinion([b_safe, b_susp, b_atk], u, np.full(3, 1 / 3))


def test_group_metric_keys_accepts_future_method_group():
    cfg = {
        "FUSION_METHOD_GROUPS": [
            {"name": "prophet", "metric_types": ["prophet"], "output_key": "M1"},
            {"name": "reconstruction", "metric_types": ["reconstruction"], "output_key": "M2"},
            {"name": "spectral", "metric_types": ["spectral"], "output_key": "M3"},
        ],
        "FUSION_UNKNOWN_METHOD_POLICY": "raise",
    }
    meta = {
        "prophet_bytes": {"type": "prophet"},
        "reconst_udp": {"type": "reconstruction"},
        "fft_energy": {"type": "spectral"},
    }
    grouped = fusion_policy.group_metric_keys(meta, cfg)
    assert grouped["prophet"] == ["prophet_bytes"]
    assert grouped["reconstruction"] == ["reconst_udp"]
    assert grouped["spectral"] == ["fft_energy"]


def test_group_metric_keys_raises_on_unknown_type():
    cfg = {"FUSION_METHOD_GROUPS": [{"name": "prophet", "metric_types": ["prophet"]}]}
    with pytest.raises(ValueError, match="unknown method type"):
        fusion_policy.group_metric_keys({"x": {"type": "new_method"}}, cfg)


@pytest.mark.parametrize("mode", ["wbf", "abf", "hierarchical", "cbf"])
def test_fuse_method_opinions_accepts_three_methods(mode):
    opinions = [
        _op(0.60, 0.10, 0.10, 0.20),
        _op(0.10, 0.10, 0.60, 0.20),
        _op(0.20, 0.20, 0.20, 0.40),
    ]
    out = fusion_policy.fuse_method_opinions(opinions, mode=mode, W=3.0)
    assert abs(float(out.b.sum()) + float(out.u) - 1.0) < 1e-9
    assert np.all(np.isfinite(out.b))
    assert np.isfinite(out.u)


def test_apply_method_discount_uses_group_config_key():
    op = _op(0.10, 0.10, 0.60, 0.20)
    group = {"attack_discount_config_key": "RECONST_ATTACK_RELIABILITY"}
    out = fusion_policy.apply_method_discount(
        op,
        group,
        {"RECONST_ATTACK_RELIABILITY": 0.5},
    )
    assert out.b[2] < op.b[2]
    assert out.u > op.u
