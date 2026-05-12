"""SL-ADS — Subjective-Logic Anomaly Detection System.

Top-level package.  Re-exports the centre-of-gravity ``CONFIG`` dict
so callers can write ``from sl_ads import CONFIG`` directly.

Sub-packages:
    core       — Subjective-Logic maths and the opinion-fusion pipeline.
    train      — model training and evidence computation.
    inject     — synthetic-attack injection at the evidence level.
    qualify    — cause-attribution qualifiers (SBN, argmax baseline).
    evaluate   — evaluation scripts (injection, qualification).
    compare    — baseline comparisons (Isolation Forest, label-vs-SL…).
    ablation   — controlled-variable ablation studies.
    stats      — bootstrap CI, McNemar, residual-correlation utilities.
    audit      — pipeline audit tooling.
    adapters   — per-dataset adapters (RedeRio, METR-LA, GECCO, CESNET).
    notebooks  — Marimo interactive notebooks.

The reorganisation that produced this layout is documented in
``docs/RENAMING_LOG_PHASE_H.md`` (2026-04-27, Phase H).
"""
__version__ = "1.0.0-phase-h"

# Convenience re-export so consumers can do ``from sl_ads import CONFIG``.
try:
    from sl_ads.config import CONFIG  # noqa: F401
except Exception:
    # During Phase H the legacy top-level config.py is still authoritative;
    # the in-package config.py is added in Phase 2.  This try/except keeps
    # `import sl_ads` cheap and side-effect-free until then.
    pass
