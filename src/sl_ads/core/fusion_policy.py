"""Fusion policy helpers shared by training calibration and inference.

The goal is to keep SL-ADS extensible at the method level.  Leaf metrics are
first grouped into semantic methods (Prophet, Reconstruction, and future
families), each method is internally pooled with WBF, then method-level
opinions are fused with the configured inter-method operator.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

import numpy as np

import sl_ads.core.subjective_logic as sl


DEFAULT_METHOD_GROUPS = (
    {
        "name": "prophet",
        "metric_types": ("prophet",),
        "output_key": "METHODE_1_PROPHET",
    },
    {
        "name": "reconstruction",
        "metric_types": ("reconstruction",),
        "output_key": "METHODE_2_RECONST",
        "attack_discount_config_key": "RECONST_ATTACK_RELIABILITY",
    },
)


def _normalise_group(raw: Mapping[str, Any], index: int) -> dict:
    name = str(raw.get("name") or f"method_{index + 1}").strip().lower()
    metric_types = raw.get("metric_types", raw.get("types", (name,)))
    if isinstance(metric_types, str):
        metric_types = (metric_types,)
    metric_types = tuple(str(t).strip().lower() for t in metric_types if str(t).strip())
    if not metric_types:
        metric_types = (name,)
    return {
        "name": name,
        "metric_types": metric_types,
        "output_key": str(raw.get("output_key") or f"METHODE_{index + 1}_{name.upper()}"),
        "attack_discount_config_key": raw.get("attack_discount_config_key"),
    }


def get_method_groups(config: Mapping[str, Any]) -> list[dict]:
    """Return configured method groups, with Prophet/Reconstruction defaults."""
    raw_groups = config.get("FUSION_METHOD_GROUPS") or DEFAULT_METHOD_GROUPS
    return [_normalise_group(g, i) for i, g in enumerate(raw_groups)]


def group_metric_keys(metric_meta: Mapping[str, Mapping[str, Any]],
                      config: Mapping[str, Any]) -> OrderedDict[str, list[str]]:
    """Map metric keys to method-group names according to metadata ``type``."""
    groups = get_method_groups(config)
    out: OrderedDict[str, list[str]] = OrderedDict((g["name"], []) for g in groups)
    type_to_group = {
        metric_type: group["name"]
        for group in groups
        for metric_type in group["metric_types"]
    }

    unknown_policy = str(config.get("FUSION_UNKNOWN_METHOD_POLICY", "raise")).lower()
    for key, meta in metric_meta.items():
        metric_type = str(meta.get("type", "")).strip().lower()
        group_name = type_to_group.get(metric_type)
        if group_name is None:
            if unknown_policy == "ignore":
                continue
            raise ValueError(
                f"Metric {key!r} has unknown method type {metric_type!r}. "
                "Add it to CONFIG['FUSION_METHOD_GROUPS'] or set "
                "FUSION_UNKNOWN_METHOD_POLICY='ignore'."
            )
        out[group_name].append(key)
    return out


def apply_method_discount(op, group: Mapping[str, Any],
                          config: Mapping[str, Any],
                          explicit_alpha: float | None = None):
    """Apply the optional attack-hypothesis discount configured for a group."""
    raw_alpha = explicit_alpha
    key = group.get("attack_discount_config_key")
    if raw_alpha is None and key:
        raw_alpha = config.get(str(key), 1.0)
    if raw_alpha is None or str(raw_alpha).strip().lower() == "auto":
        raw_alpha = 1.0
    alpha = float(raw_alpha)
    if alpha < 1.0 - 1e-6:
        return sl.apply_contextual_discount(op, alpha=[1.0, 1.0, alpha])
    return op


def fuse_method_opinions(method_opinions: list,
                         mode: str,
                         W: float,
                         balance_ratio_eff: float = 1.0):
    """Fuse method-level opinions with the configured inter-method operator."""
    method_opinions = list(method_opinions)
    if not method_opinions:
        return sl.MultinomialOpinion([0, 0, 0], 1.0)
    if len(method_opinions) == 1:
        op = method_opinions[0]
        return sl.MultinomialOpinion(op.b.copy(), op.u, op.a.copy())

    mode = str(mode).strip().lower()
    if mode == "hierarchical":
        weights = [1.0 / len(method_opinions)] * len(method_opinions)
        return sl.fusion_evidence_average_n_sources(method_opinions, external_weights=weights, W=W)

    if mode == "cbf" and len(method_opinions) == 2 and not np.isclose(balance_ratio_eff, 1.0):
        left, right = method_opinions
        if balance_ratio_eff > 1.0:
            left = sl.boost_opinion_evidence(left, 1.0 / balance_ratio_eff, W=W)
        else:
            right = sl.boost_opinion_evidence(right, balance_ratio_eff, W=W)
        return sl.fusion_cbf(left, right)

    return sl.fusion_by_mode(method_opinions, mode=mode, W=W)

