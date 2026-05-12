"""Model-artifact completeness guardrails.

The production RedeRio pipeline is defined as 12 Prophet leaves plus
5 reconstruction leaves.  A partial artifact is scientifically invalid
for a "full pipeline" run: downstream metrics would silently become a
different experiment.  These helpers centralise the expected-key logic
so both training and inference fail fast on incomplete artifacts.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping


def expected_model_keys_from_config(config: Mapping[str, Any]) -> list[str]:
    """Return the model keys required by the active configuration.

    ``ACTIVE_METRICS='auto'`` is resolved inside ``train_models`` before
    training.  If a consumer calls this helper before resolution, there
    is no safe static list of Prophet keys, so the Prophet portion is
    intentionally empty.
    """
    active_metrics = config.get("ACTIVE_METRICS", [])
    prophet_metrics: Iterable[str]
    if isinstance(active_metrics, str):
        prophet_metrics = [] if active_metrics == "auto" else [active_metrics]
    else:
        prophet_metrics = active_metrics or []

    prophet_keys = [f"prophet_{metric}" for metric in prophet_metrics]
    reconst_keys = [
        f"reconst_{rule['target']}_from_{rule['feature']}"
        for rule in (config.get("RECONST_RULES") or [])
        if rule.get("target") and rule.get("feature")
    ]
    return prophet_keys + reconst_keys


def validate_required_model_coverage(
    models_pkg: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    expected_keys: Iterable[str] | None = None,
    failures: Iterable[Mapping[str, Any]] | None = None,
) -> None:
    """Raise if an expected model leaf is missing from ``models_pkg``.

    The guard is enabled by default.  ``REQUIRE_ALL_MODELS=False`` is an
    explicit escape hatch for debugging-only experiments; production and
    paper runs should leave it enabled.
    """
    if not bool(config.get("REQUIRE_ALL_MODELS", True)):
        return

    expected = list(expected_keys or expected_model_keys_from_config(config))
    missing = [
        key for key in expected
        if not isinstance(models_pkg.get(key), dict)
    ]
    if not missing:
        return

    failure_bits: list[str] = []
    for failure in failures or []:
        expected_key = failure.get("expected_key") or failure.get("metric")
        if expected_key in missing:
            reason = failure.get("error") or failure.get("reason") or "unknown"
            failure_bits.append(f"{expected_key}: {reason}")

    details = ""
    if failure_bits:
        details = " Recorded failures: " + "; ".join(failure_bits[:8])
        if len(failure_bits) > 8:
            details += f"; ... (+{len(failure_bits) - 8} more)"

    raise RuntimeError(
        "Incomplete model artifact: missing required model leaves "
        f"{missing}. A full pipeline run must train every configured "
        "Prophet/reconstruction leaf before computing evidence."
        f"{details}"
    )
