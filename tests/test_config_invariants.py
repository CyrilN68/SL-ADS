"""
test_config_invariants.py — Regression guards on the published configuration.

Verifies that the configuration shipped in ``src/sl_ads/config.py`` preserves
the methodological invariants required by the audit/closure documents:

  - ``WBF_WEIGHT_MODE = "uniform"`` (PATCH "uniform-as-reference" 2026-04-29)
  - ``INTER_METHOD_FUSION = "wbf"`` (PATCH M-11/CBF, audit_codex MAJ-09)
  - ``LR_NOVELTY_THR = None`` (PATCH-C2, no test-derived threshold —
    Varma & Simon 2006; Japkowicz & Shah 2011)
  - ``full_sl`` ablation matches the production reference (uniform)
  - The pathological alternative ``trust_discount_legacy`` exists in the
    ablation registry as the *only* run with ``wbf_weight_mode="trust_discount"``.

Tests covering these as TASK-47 / TASK-51 in audit_verification_tracker.md.

Run from project root:

    pytest tests/test_config_invariants.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ════════════════════════════════════════════════════════════════════════
# TASK-47 — uniform-as-reference invariants
# ════════════════════════════════════════════════════════════════════════
def test_wbf_weight_mode_is_uniform_in_production():
    """The shipped production config must use uniform weighting.

    Trust-discount has a documented R²-pathology (5/12 Prophet models with
    R² < 0 → poids inversés sur les signaux discriminants).  See
    docs/audit/trust_discount_r2_analysis.md and §5.3.3 honest_limitations.md.
    """
    from sl_ads.config import CONFIG
    assert CONFIG["WBF_WEIGHT_MODE"] == "uniform", (
        "Production config must ship WBF_WEIGHT_MODE='uniform' "
        "(PATCH uniform-as-reference 2026-04-29). Got "
        f"{CONFIG['WBF_WEIGHT_MODE']!r}.  Trust-discount has a documented "
        "R²-pathology (F1 0.811→0.566). "
        "See docs/audit/trust_discount_r2_analysis.md."
    )


def test_inter_method_fusion_is_wbf():
    """PATCH M-11 (CBF independence concern): inter-method fusion must default to WBF."""
    from sl_ads.config import CONFIG
    assert CONFIG["INTER_METHOD_FUSION"] == "wbf", (
        "Production config must ship INTER_METHOD_FUSION='wbf' "
        "(PATCH M-11, audit_codex MAJ-09).  Got "
        f"{CONFIG['INTER_METHOD_FUSION']!r}."
    )


def test_full_sl_ablation_matches_production_reference():
    """The 'full_sl' ablation run must match the production WBF mode.

    Otherwise the ablation table mislabels its reference and any 'Δ vs
    reference' comparison is meaningless.
    """
    from sl_ads.config import CONFIG
    runs = CONFIG.get("ABLATION", {}).get("RUNS", {})
    full_sl = runs.get("full_sl")
    assert full_sl is not None, "'full_sl' run missing from ABLATION.RUNS"
    assert full_sl.get("wbf_weight_mode") == "uniform", (
        "'full_sl' ablation reference must use 'uniform' weighting "
        "to match the production config.  Got "
        f"{full_sl.get('wbf_weight_mode')!r}."
    )
    assert full_sl.get("wbf_uniform") is True, (
        "'full_sl' ablation must have wbf_uniform=True.  Got "
        f"{full_sl.get('wbf_uniform')!r}."
    )
    assert full_sl.get("inter_method_fusion", CONFIG.get("INTER_METHOD_FUSION")) == CONFIG["INTER_METHOD_FUSION"], (
        "'full_sl' ablation reference must match production INTER_METHOD_FUSION.  Got "
        f"{full_sl.get('inter_method_fusion')!r} vs production "
        f"{CONFIG['INTER_METHOD_FUSION']!r}."
    )


def test_validate_full_sl_ads_defaults_matches_uniform_reference():
    """The production invariant helper must agree with uniform-as-reference.

    This helper is consumed by audit notebooks/scripts; keeping a stale
    trust_discount invariant there would make the reference run appear invalid.
    """
    from sl_ads.config import CONFIG
    from sl_ads.paths import validate_full_sl_ads_defaults

    invariants = validate_full_sl_ads_defaults(CONFIG)
    assert invariants["wbf_mode_uniform"] is True
    assert "wbf_mode_trust_discount" not in invariants


def test_trust_discount_legacy_exists_for_pathology_demo():
    """A dedicated 'trust_discount_legacy' run must expose the F1=0.566 pathology.

    Without this run, the ablation table cannot demonstrate why
    uniform is the reference rather than trust_discount.
    """
    from sl_ads.config import CONFIG
    runs = CONFIG.get("ABLATION", {}).get("RUNS", {})
    legacy = runs.get("trust_discount_legacy")
    assert legacy is not None, (
        "Ablation registry must include 'trust_discount_legacy' to "
        "demonstrate the R²-pathology against the uniform reference."
    )
    assert legacy.get("wbf_weight_mode") == "trust_discount", (
        "'trust_discount_legacy' must actually use trust_discount.  Got "
        f"{legacy.get('wbf_weight_mode')!r}."
    )
    assert legacy.get("wbf_uniform") is False, (
        "'trust_discount_legacy' must have wbf_uniform=False so the "
        "trust-discount path is exercised.  Got "
        f"{legacy.get('wbf_uniform')!r}."
    )


def test_no_other_run_uses_trust_discount():
    """Only 'trust_discount_legacy' should use wbf_weight_mode='trust_discount'.

    All other 'isolated' / 'sensitivity' / 'cd_alpha' runs were migrated to
    uniform on 2026-04-29 to derive cleanly from the new full_sl reference.
    """
    from sl_ads.config import CONFIG
    runs = CONFIG.get("ABLATION", {}).get("RUNS", {})
    offenders = [
        name for name, cfg in runs.items()
        if cfg.get("wbf_weight_mode") == "trust_discount"
        and name != "trust_discount_legacy"
    ]
    assert not offenders, (
        f"Only 'trust_discount_legacy' may use wbf_weight_mode='trust_discount'. "
        f"Found unexpected offenders: {offenders}."
    )


# ════════════════════════════════════════════════════════════════════════
# TASK-51 — LR_NOVELTY_THR=None regression guard (PATCH-C2)
# ════════════════════════════════════════════════════════════════════════
def test_lr_novelty_thr_is_none():
    """PATCH-C2: the published config must not ship a test-derived
    LR_NOVELTY_THR.

    Setting this to a numerical value would re-introduce the
    test-set-leakage flagged by Varma & Simon (2006) and
    Japkowicz & Shah (2011).  Reporting AUC and J in-sample is fine
    (cf. §5.3.12 honest_limitations.md), but binarising on a learned
    threshold is not.
    """
    from sl_ads.config import CONFIG
    eval_cfg = CONFIG.get("EVAL", {})
    # The threshold is read at three possible levels; check all.
    candidates = [
        ("CONFIG['LR_NOVELTY_THR']",                   CONFIG.get("LR_NOVELTY_THR")),
        ("CONFIG['EVAL']['LR_NOVELTY_THR']",           eval_cfg.get("LR_NOVELTY_THR")),
        ("CONFIG['EVAL']['NOVELTY_LR_THRESHOLD']",     eval_cfg.get("NOVELTY_LR_THRESHOLD")),
    ]
    for name, val in candidates:
        assert val is None, (
            f"PATCH-C2 regression: {name} must be None to avoid "
            f"test-derived threshold leakage (Varma & Simon 2006, "
            f"Japkowicz & Shah 2011).  Got {val!r}."
        )


# ════════════════════════════════════════════════════════════════════════
# Sanity: documentation pointer files still exist
# ════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("relpath", [
    "docs/honest_limitations.md",
    "docs/audit/trust_discount_r2_analysis.md",
    "docs/audit/audit_verification_tracker.md",
    "docs/review/PUBLICATION_TABLES.md",
    "docs/review/regime_fpr_root_cause_analysis.md",
    "docs/archive/2026-05-11_public_release_cleanup/review/SCIENTIFIC_HARDENING_20260506.md",
    "docs/archive/2026-05-07_audit_cleanup/review/SCIENTIFIC_HARDENING_20260504.md",
])
def test_disclosure_documents_present(relpath):
    """The disclosure documents reviewers consult for the audit narrative must
    remain reachable. The 2026-05-04 hardening note was archived during the
    2026-05-07 cleanup; the 2026-05-06 hardening note was archived during the
    public-release cleanup. Active paper-facing figures now live in
    ``PUBLICATION_TABLES.md`` and ``regime_fpr_root_cause_analysis.md``.
    """
    p = _ROOT / relpath
    assert p.is_file(), f"Required disclosure document missing: {relpath}"


def test_sbn_qualifier_declares_legacy_terminology():
    """The qualifier must not advertise itself as a strict SBN."""
    text = (_ROOT / "src" / "sl_ads" / "qualify" / "sbn_qualifier.py").read_text(
        encoding="utf-8", errors="replace"
    )
    head = text[:2500].lower()
    assert "not a canonical subjective bayesian network" in head
    assert "sl-template qualifier" in head
    assert "architecture sbn rigoureuse" not in head
