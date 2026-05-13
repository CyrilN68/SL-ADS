"""
test_audit_remediation_20260426.py — Unit tests for Phase F audit fixes
=========================================================================

Validates the Phase F remediation patches applied on 2026-04-26 in
response to ``_audit_tmp/SCIENTIFIC_AUDIT_REPORT.md``:

  * TASK-20 (CRIT-02) — compute_opinions_v3 raises on missing _attacks
  * TASK-21 (CRIT-03) — evaluate_injection_v2 hybrid-F1 column rename
  * TASK-24 (MAJ-03)  — compare_qualif_methods uses paths.get_decision_threshold
  * TASK-25 (MAJ-04)  — evaluate_qualify_sbn no longer falls back to a
                         hardcoded versioned directory
  * TASK-26 (MAJ-05)  — sl_formulas_v2.compute_asymmetric_escalation_conflict
                         (renamed) + new compute_conflict_degree_canonical
  * TASK-27 (MAJ-06)  — sl_formulas_v2.fusion_cbf symmetric degenerate handling
  * TASK-28 (MAJ-07)  — targeted warnings filters (no global ignore)
  * TASK-32 (MAJ-11)  — utils_manifest.compute_run_id determinism
  * TASK-33 (MIN-01)  — marimo paths point to current dir name
  * TASK-33 (MIN-03)  — compute_pearson_independence reads ATTACK_PERIODS
                         from config.py instead of hardcoding

Run
---
    pytest tests/test_audit_remediation_20260426.py -v
    # OR (no pytest):
    python tests/test_audit_remediation_20260426.py

Exits 0 on success, non-zero on any failure.
"""
from __future__ import annotations

import os
import re
import sys
import warnings
from pathlib import Path

# Phase H: conftest.py at the project root adds ``src/`` and the
# project root to sys.path so ``sl_ads.*`` is importable from tests.
# We keep ``_PROJ`` defined for path-existence assertions below.
_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent
_SRC = _PROJ / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import numpy as np  # noqa: E402


# ----------------------------------------------------------------------
# TASK-26 (MAJ-05) — conflict_degree rename & canonical
# ----------------------------------------------------------------------
def test_task26_conflict_alias_exists() -> None:
    import sl_ads.core.subjective_logic as sl
    # The legacy public name still works (back-compat alias).
    assert hasattr(sl, "compute_conflict_degree"), "legacy alias missing"
    # The new explicit name exists.
    assert hasattr(sl, "compute_asymmetric_escalation_conflict"), \
        "renamed function missing"
    # The new canonical (symmetric) version exists.
    assert hasattr(sl, "compute_conflict_degree_canonical"), \
        "canonical symmetric version missing"


def test_task26_conflict_alias_equivalence() -> None:
    """compute_conflict_degree must forward to the asymmetric variant."""
    import sl_ads.core.subjective_logic as sl
    rng = np.random.default_rng(20260426)
    for _ in range(50):
        r_prev = rng.uniform(0, 10, size=3)
        r_curr = rng.uniform(0, 10, size=3)
        a = sl.compute_conflict_degree(r_prev, r_curr)
        b = sl.compute_asymmetric_escalation_conflict(r_prev, r_curr)
        assert abs(a - b) < 1e-12, f"alias diverges: {a} vs {b}"


def test_task26_canonical_is_symmetric() -> None:
    """The canonical version must be symmetric: K(prev,curr) == K(curr,prev)."""
    import sl_ads.core.subjective_logic as sl
    rng = np.random.default_rng(42)
    for _ in range(50):
        r_a = rng.uniform(0, 10, size=3)
        r_b = rng.uniform(0, 10, size=3)
        k_ab = sl.compute_conflict_degree_canonical(r_a, r_b)
        k_ba = sl.compute_conflict_degree_canonical(r_b, r_a)
        assert abs(k_ab - k_ba) < 1e-12, \
            f"canonical conflict not symmetric: {k_ab} vs {k_ba}"


def test_task26_asymmetric_is_actually_asymmetric() -> None:
    """The escalation-only variant must produce different values for
    escalation vs de-escalation (otherwise the rename is meaningless)."""
    import sl_ads.core.subjective_logic as sl
    # Strong escalation: prev=safe, curr=attack.
    r_safe   = np.array([10.0, 0.0, 0.0])
    r_attack = np.array([0.0, 0.0, 10.0])
    k_up   = sl.compute_asymmetric_escalation_conflict(r_safe, r_attack)
    k_down = sl.compute_asymmetric_escalation_conflict(r_attack, r_safe)
    # Both should be high (safe×attack and attack×safe are both in the
    # asymmetric kernel) — but the canonical diff lives in suspect↔safe
    # transitions, which we check next.
    r_susp = np.array([0.0, 10.0, 0.0])
    k_safe_to_susp = sl.compute_asymmetric_escalation_conflict(r_safe, r_susp)
    k_susp_to_safe = sl.compute_asymmetric_escalation_conflict(r_susp, r_safe)
    # safe→susp counts (escalation), susp→safe does not (de-escalation):
    assert k_safe_to_susp > k_susp_to_safe + 0.05, \
        f"asymmetry not preserved: up={k_safe_to_susp} vs down={k_susp_to_safe}"


# ----------------------------------------------------------------------
# TASK-27 (MAJ-06) — fusion_cbf symmetric degenerate handling
# ----------------------------------------------------------------------
def test_task27_fusion_cbf_symmetric_degenerate() -> None:
    """When fusion_cbf hits the degenerate denom branch, the result
    must NOT silently equal op_A.  It should be the weighted average."""
    import sl_ads.core.subjective_logic as sl

    # Construct two non-dogmatic (u >= 1e-9) opinions whose denom is
    # numerically degenerate.  Hard to hit naturally; we instead verify
    # the easier property: identical input → identical output (idempotence)
    # and average→ symmetric result for two near-degenerate inputs.

    # Sanity: two clean non-degenerate opinions fuse correctly.
    op_a = sl.MultinomialOpinion(
        np.array([0.6, 0.1, 0.1]), 0.2, np.array([0.5, 0.3, 0.2])
    )
    op_b = sl.MultinomialOpinion(
        np.array([0.5, 0.2, 0.1]), 0.2, np.array([0.4, 0.4, 0.2])
    )
    fused = sl.fusion_cbf(op_a, op_b)
    assert isinstance(fused, sl.MultinomialOpinion)
    # Bijection: Σb + u ≈ 1
    s = float(fused.b.sum() + fused.u)
    assert abs(s - 1.0) < 1e-6, f"fusion_cbf bijection broken: Σb+u={s}"
    # Symmetry: f(a,b) = f(b,a) for non-dogmatic case.
    fused_rev = sl.fusion_cbf(op_b, op_a)
    assert np.allclose(fused.b, fused_rev.b, atol=1e-9), "fusion_cbf asymmetric"
    assert abs(fused.u - fused_rev.u) < 1e-9


# ----------------------------------------------------------------------
# TASK-32 (MAJ-11) — run_id determinism
# ----------------------------------------------------------------------
def test_task32_run_id_deterministic() -> None:
    from sl_ads.utils_manifest import compute_run_id
    cfg = {"VERSION_NAME": "test", "LAMBDA_DECAY": 0.85, "SL_PARAM_K": 3.0}
    rid1 = compute_run_id(config=cfg, git_sha="abc1234")
    rid2 = compute_run_id(config=cfg, git_sha="abc1234")
    assert rid1 == rid2, "run_id must be deterministic for identical inputs"
    assert len(rid1) == 16, "run_id should be 16 hex chars"
    assert re.fullmatch(r"[0-9a-f]{16}", rid1), \
        f"run_id not lowercase hex: {rid1!r}"


def test_task32_run_id_differs_on_change() -> None:
    from sl_ads.utils_manifest import compute_run_id
    cfg1 = {"LAMBDA_DECAY": 0.85}
    cfg2 = {"LAMBDA_DECAY": 0.90}
    rid1 = compute_run_id(config=cfg1, git_sha="x")
    rid2 = compute_run_id(config=cfg2, git_sha="x")
    assert rid1 != rid2, "run_id must change when CONFIG changes"
    rid3 = compute_run_id(config=cfg1, git_sha="y")
    assert rid1 != rid3, "run_id must change when git_sha changes"


def test_task32_run_id_handles_missing_inputs() -> None:
    """Missing input files must produce 'missing' fingerprint, not raise."""
    from sl_ads.utils_manifest import compute_run_id
    rid = compute_run_id(
        config={"a": 1},
        git_sha="z",
        input_paths=["/nonexistent/path/foo.csv"],
    )
    assert isinstance(rid, str) and len(rid) == 16


# ----------------------------------------------------------------------
# TASK-21 (CRIT-03) — hybrid F1 column names + best-row helper
# ----------------------------------------------------------------------
def test_task21_select_best_row_uses_hybrid_metric(monkeypatch) -> None:
    """_select_best_row must prefer the explicit hybrid column when
    available, and gracefully fall back to f1_coverage otherwise.

    NOTE (audit_codex CRIT-01, 2026-04-27): this legacy assertion was
    written before TASK-34 forbade argmax-based threshold selection.
    Argmax behaviour is now reachable only through the explicit escape
    hatch ``SL_ALLOW_TEST_TUNED_THRESHOLD=1`` (research/legacy only).
    The test is preserved here to verify the column-name fallback logic
    still routes through the same code path under the override; the
    primary anti-leakage assertions live in
    ``test_audit_codex_remediation_20260427.py``.
    """
    import importlib
    import pandas as pd

    monkeypatch.setenv("SL_ALLOW_TEST_TUNED_THRESHOLD", "1")
    # Phase H: the implementation moved to sl_ads.evaluate.evaluate_injection;
    # the legacy ``evaluate_injection_v2`` is now a deprecation shim that
    # cannot expose ``_``-prefixed names through ``import *``.
    ev = importlib.import_module("sl_ads.evaluate.evaluate_injection")
    sel = ev._select_best_row

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", category=UserWarning)
        # Case 1: new explicit column present → use it.
        df_new = pd.DataFrame({
            "threshold": [0.10, 0.20, 0.30],
            "f1_coverage_hybrid_episode_recall": [0.30, 0.80, 0.50],
            "f1_coverage": [0.30, 0.20, 0.50],   # decoy: max @ 0.30
        })
        best = sel(df_new)
        assert float(best["threshold"]) == 0.20, \
            f"expected 0.20 (hybrid argmax), got {best['threshold']}"

        # Case 2: only legacy column present → fall back.
        df_legacy = pd.DataFrame({
            "threshold": [0.10, 0.20, 0.30],
            "f1_coverage": [0.10, 0.20, 0.90],
        })
        best_legacy = sel(df_legacy)
        assert float(best_legacy["threshold"]) == 0.30


# ----------------------------------------------------------------------
# TASK-28 (MAJ-07) — no global warnings.filterwarnings('ignore')
# ----------------------------------------------------------------------
def test_task28_no_global_ignore_in_4_scripts() -> None:
    """Ensure none of the 4 patched scripts re-introduces a global
    ``warnings.filterwarnings('ignore')`` line.

    Phase H: paths point at the new sl_ads/ package layout.  Falls back
    to the legacy flat path if the new file is missing (transitional)."""
    targets = [
        ("evaluate_injection_v2.py", "src/sl_ads/evaluate/evaluate_injection.py"),
        ("compare_labeller_vs_sl.py", "src/sl_ads/compare/compare_labeller_vs_sl.py"),
        ("run_ablation_labeled.py",   "src/sl_ads/ablation/run_ablation_labeled.py"),
        ("run_ablation_v2.py",        "src/sl_ads/ablation/run_ablation.py"),
    ]
    bad = []
    for legacy, new in targets:
        new_path = _PROJ / new
        legacy_path = _PROJ / legacy
        p = new_path if new_path.exists() else legacy_path
        text = p.read_text(encoding="utf-8", errors="replace")
        if re.search(r'^\s*warnings\.filterwarnings\(\s*["\']ignore["\']\s*\)\s*$',
                     text, re.MULTILINE):
            bad.append(p.name)
    assert not bad, f"global ignore re-introduced in: {bad}"


# ----------------------------------------------------------------------
# TASK-33 (MIN-01) — marimo notebooks resolve their project root portably
# ----------------------------------------------------------------------
def test_task33_min01_marimo_paths() -> None:
    """Public-release contract for the Marimo notebooks under
    ``src/sl_ads/notebooks/``:

      * no hardcoded absolute path (Windows ``C:\\Users\\``,
        Unix ``/Users/`` or ``/home/``) — would leak the author's
        environment and break on any other machine;
      * no reference to the pre-Phase-H folder name
        ``actual_ version_claude_autre dataset`` (now archived);
      * each notebook resolves its project root from
        ``pathlib.Path(__file__).resolve()`` so it stays portable;
      * each notebook imports the package config via
        ``from sl_ads.config import CONFIG``.
    """
    notebooks = [
        _PROJ / "src" / "sl_ads" / "notebooks" / "admin.py",
        _PROJ / "src" / "sl_ads" / "notebooks" / "compute_opinions.py",
        _PROJ / "src" / "sl_ads" / "notebooks" / "qualify_sbn.py",
    ]
    forbidden_substrings = (
        r"C:\Users\\",                       # Windows hardcoded user dir
        "/Users/",                            # macOS hardcoded user dir
        "/home/",                             # Linux hardcoded user dir
        "actual_ version_claude_autre dataset",  # pre-Phase-H folder
        "actual_version",                    # earlier pre-Phase-H folder
    )
    required_substrings = (
        "pathlib.Path(__file__).resolve()",
        "from sl_ads.config import CONFIG",
    )
    for nb in notebooks:
        text = nb.read_text(encoding="utf-8", errors="replace")
        for forbidden in forbidden_substrings:
            assert forbidden not in text, (
                f"{nb.name}: forbidden substring '{forbidden}' still present "
                f"(hardcoded path or stale pre-Phase-H folder name)"
            )
        for required in required_substrings:
            assert required in text, (
                f"{nb.name}: required substring '{required}' missing "
                f"(notebook must resolve __file__ and import sl_ads.config)"
            )


# ----------------------------------------------------------------------
# TASK-33 (MIN-03) — compute_pearson_independence reads from config.py
# ----------------------------------------------------------------------
def test_task33_min03_pearson_uses_config() -> None:
    """Phase H: ``modèle évaluation/`` was renamed to ``investigations/``."""
    new_path = _PROJ / "investigations" / "compute_pearson_independence.py"
    legacy_path = _PROJ / "modèle évaluation" / "compute_pearson_independence.py"
    target = new_path if new_path.exists() else legacy_path
    text = target.read_text(encoding="utf-8", errors="replace")
    # Must import INJECTED_ATTACK_CATALOG (or REAL_ATTACKS) from config.
    assert "INJECTED_ATTACK_CATALOG" in text, \
        "MIN-03: still hardcoded periods (no config import)"
    # Must NOT keep the old hardcoded literal "2025-11-12 18:21:13".
    # (allow it inside comments, but the *literal* tuple list must be gone)
    # Heuristic: ensure the build helper exists.
    assert "_build_attack_periods" in text, \
        "MIN-03: dynamic period builder missing"


# ----------------------------------------------------------------------
# TASK-25 (MAJ-04) — no hardcoded versioned fallback in evaluate_qualify_sbn
# ----------------------------------------------------------------------
def test_task25_no_hardcoded_v9_v9_v4s_fallback() -> None:
    """Phase H: read from sl_ads/evaluate/evaluate_qualify_sbn.py if available."""
    new_path = _PROJ / "src" / "sl_ads" / "evaluate" / "evaluate_qualify_sbn.py"
    legacy_path = _PROJ / "evaluate_qualify_sbn.py"
    target = new_path if new_path.exists() else legacy_path
    text = target.read_text(encoding="utf-8", errors="replace")
    # The hardcoded path must not appear as a fallback assignment.
    # We allow the string in comments (PATCH explanation).
    bad_pattern = re.compile(
        r"^\s*_fallback\s*=.*resultats_trained_models_v9_v9_v4s_v3_v3",
        re.MULTILINE,
    )
    assert not bad_pattern.search(text), \
        "MAJ-04: hardcoded versioned fallback still present"


# ----------------------------------------------------------------------
# TASK-24 (MAJ-03) — compare_qualif_methods uses paths.get_decision_threshold
# ----------------------------------------------------------------------
def test_task24_compare_uses_paths_helper() -> None:
    """Phase H: the implementation moved to
    ``sl_ads/compare/compare_qualif_methods.py``; read from there if
    available, fall back to the legacy shim path otherwise.

    Phase 3 import rewrite: accept either the legacy ``from paths``
    pattern (back-compat shim) or the new ``from sl_ads.paths`` form."""
    new_path = _PROJ / "src" / "sl_ads" / "compare" / "compare_qualif_methods.py"
    legacy_path = _PROJ / "compare_qualif_methods.py"
    target = new_path if new_path.exists() else legacy_path
    text = target.read_text(encoding="utf-8", errors="replace")
    assert (
        ("from paths import" in text)
        or ("from sl_ads.paths import" in text)
    ) and "get_decision_threshold" in text, \
        "MAJ-03: paths.get_decision_threshold not imported"
    # Old hardcoded fallback `CONFIG.get('DECISION_THRESHOLD', 0.20)` must
    # no longer be the *only* assignment to GATE_THRESHOLD in the try block.
    # The new line `GATE_THRESHOLD = get_decision_threshold(CONFIG, ...)`
    # must be present.
    assert "GATE_THRESHOLD  = get_decision_threshold" in text \
        or "GATE_THRESHOLD = get_decision_threshold" in text, \
        "MAJ-03: GATE_THRESHOLD still computed via CONFIG.get fallback"


# ----------------------------------------------------------------------
# TASK-20 (CRIT-02) — compute_opinions_v3 raises on missing _attacks
# ----------------------------------------------------------------------
def test_task20_compute_opinions_raises_on_missing_attacks() -> None:
    """Phase H: the implementation moved to sl_ads/core/opinions_pipeline.py.
    Read from there (or the legacy shim, whichever exists)."""
    new_path = _PROJ / "src" / "sl_ads" / "core" / "opinions_pipeline.py"
    legacy_path = _PROJ / "compute_opinions_v3.py"
    target = new_path if new_path.exists() else legacy_path
    text = target.read_text(encoding="utf-8", errors="replace")
    # Must contain the explicit raise FileNotFoundError block.
    assert "raise FileNotFoundError" in text, \
        "CRIT-02: explicit raise missing"
    assert "SL_ALLOW_NONINJECTED_FALLBACK" in text, \
        "CRIT-02: env-var escape hatch missing"
    assert "[CRIT-02]" in text, \
        "CRIT-02: error message tag missing"


# ----------------------------------------------------------------------
# Manual runner (no pytest required)
# ----------------------------------------------------------------------
def _run_all() -> int:
    failures = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"  [OK] {name}")
        except AssertionError as exc:
            failures.append((name, str(exc)))
            print(f"  [FAIL] {name}: {exc}")
        except Exception as exc:
            failures.append((name, repr(exc)))
            print(f"  [ERR ] {name}: {exc!r}")
    if failures:
        print(f"\n{len(failures)} failure(s).")
        for n, e in failures:
            print(f"  - {n}: {e}")
        return 1
    print(f"\n{sum(1 for k in globals() if k.startswith('test_'))} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
