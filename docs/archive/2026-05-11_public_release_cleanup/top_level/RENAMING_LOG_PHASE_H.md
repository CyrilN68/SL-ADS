# Renaming Log — Phase H reorganisation (2026-04-27 → ongoing)

**Status:** ✅ ALL 7 PHASES COMPLETE (2026-04-29).  The reorganisation
is closed.  Project root is now ASCII-only, the ``src/sl_ads/``
package layout is canonical, every test runs strict-mode (no
DeprecationWarning), and the legacy flat layout has been fully
deleted.

**2026-05-07 documentation note:** this file is a historical rename log. Some
Phase H paths listed below were later moved to
`docs/archive/2026-05-07_audit_cleanup/` during the audit-documentation
cleanup. Use `docs/README.md` and `docs/AUDIT_CURRENT_STATUS.md` for the current
documentation map.

---

## Goal

Convert the legacy flat layout — ~50 Python files at the project root,
mixed with output directories and unicode-named folders — into a clean
PEP-518 package layout (``src/sl_ads/<sub-package>/<module>.py``) with:

* clear, semantic module names (no version suffixes like ``_v2``,
  ``_v3``, ``_v10``);
* ASCII-only paths (no spaces, no accents);
* a single launcher script (``run_pipeline.py``) at the project root;
* the legacy paths preserved as **deprecation shims** for one cycle so
  that no out-of-tree caller breaks immediately.

The user has a backup of the old layout at the sibling folder
``actual_ version_claude_autre dataset save/`` — Phase H must not
touch it.

---

## Phase plan

| Phase | What | Risk | Status |
|-------|------|------|--------|
| 1 | Create skeleton: ``src/sl_ads/<subpkg>/__init__.py``, ``investigations/``, ``docs/audit/``, ``docs/review/``, ``docs/archive/``; write ``pyproject.toml``; this log file. | None — additive only. | ✅ DONE 2026-04-27 |
| 2 | Move each legacy module into its new location.  At each step the old import path is preserved as a 3-line shim that re-exports from the new module and emits a ``DeprecationWarning``.  44/44 tests must pass after each batch. | Low — shims keep callers working. | ✅ DONE 2026-04-27 |
| 3 | Rewrite *internal* imports across the codebase to use the new dotted paths (``from sl_ads.core.subjective_logic import …``).  Done by a deterministic ``Grep+Edit`` pass.  44/44 tests must pass. | Low — purely mechanical. | ✅ DONE 2026-04-29 |
| 4 | Migrate ``run_full_sl_ads.py`` → ``run_pipeline.py`` at root.  Rewrite the ``_STEPS_*`` tables to point at the new module paths via ``-m sl_ads.train.train_models`` (or the script-style invocation).  Add the dual-output (``outputs/`` + historical ``results/<run_id>/``) mechanism. | Medium — touches the entry-point. | ✅ DONE 2026-04-29 |
| 5 | Mass-rewrite all docs (``*.md``) referencing legacy file names.  Mostly substitution — verified by ``grep`` of the old-name index. | None — docs only. | ✅ DONE 2026-04-29 |
| 6 | Run full pytest + a smoke pipeline run.  Document the green state in ``docs/audit/audit_verification_tracker.md``. | Low — diagnostic only. | ✅ DONE 2026-04-29 |
| 7 | Remove the deprecation shims after the user confirms one full cycle of green tests on the new paths.  Delete obsolete artefacts (``test.py`` stub at root, etc.).  Optionally rename the project-root folder to ``sl_ads/``. | Low (removed shims become 3-line ``import`` errors which are easy to spot). | ✅ DONE 2026-04-29 |

---

## Phase 1 — what was created (2026-04-27)

### New directories

```
src/
└── sl_ads/
    ├── __init__.py
    ├── ablation/__init__.py
    ├── adapters/__init__.py
    ├── audit/__init__.py
    ├── compare/__init__.py
    ├── core/__init__.py
    ├── evaluate/__init__.py
    ├── inject/__init__.py
    ├── notebooks/__init__.py
    ├── qualify/__init__.py
    ├── stats/__init__.py
    └── train/__init__.py
investigations/
docs/audit/
docs/review/
docs/archive/
```

### New files

* ``pyproject.toml`` — declares ``sl_ads`` as a setuptools package with
  ``where = ["src"]`` and ``include = ["sl_ads*"]``.  Pytest config
  embedded; warning filters cover the audit_codex escape-hatch
  ``UserWarning`` strings.
* ``docs/RENAMING_LOG_PHASE_H.md`` — this file.
* 12 ``__init__.py`` documenting each sub-package's purpose.

### What is *not* yet done in Phase 1

* No legacy file has been moved or modified.
* No import has been rewritten.
* All 44 existing tests must still pass (verified at end of Phase 1).

---

## Phase 2 — what was done (2026-04-27)

* **30 modules moved** from the project root and ``dataset_adapter/``
  into ``src/sl_ads/<subpkg>/<module>.py`` (10 batches, each followed
  by ``pytest tests/`` returning 44/44).
* **30 deprecation shims** written at the legacy paths.  Each shim:
  imports ``_phase_h_path`` (which adds ``src/`` to ``sys.path``),
  emits a ``DeprecationWarning``, then ``from sl_ads.<subpkg>.<mod>
  import *``.  Entry-point scripts (``train_v10``, ``compute_opinions_v3``,
  ``evaluate_injection_v2``, ablation runners, audit, marimo notebooks,
  …) additionally call ``runpy.run_module`` under
  ``if __name__ == "__main__"``.
* **Cross-imports inside ``src/sl_ads/adapters/``** rewritten from the
  legacy ``from adapter_base import …`` style to absolute
  ``from sl_ads.adapters.adapter_base import …``.  Same for
  ``labeller_unsupervised`` references inside ``adapter_base.py`` and
  ``rederio_adapter.py``.  Adapters' ``from config import …`` calls
  rewritten to ``from sl_ads.config import …``.  This is a small
  Phase-3-flavoured edit happening early because the new copies must
  be importable before their shims can re-export from them.
* **Bootstrap files**:
    - ``_phase_h_path.py`` (project root) — single-purpose ``sys.path``
      injection helper imported by every shim.
    - ``conftest.py`` (project root) — pytest auto-load that prepends
      ``src/`` (and the project root) to ``sys.path``.
* **Test updates** (5 tests adjusted to align with the new layout):
    - ``test_task20_compute_opinions_raises_on_missing_attacks`` —
      reads from the new ``sl_ads/core/opinions_pipeline.py`` if it
      exists, else falls back to the legacy path.
    - ``test_task21_select_best_row_uses_hybrid_metric`` — imports
      ``sl_ads.evaluate.evaluate_injection`` directly (the legacy
      ``import *`` shim drops the ``_select_best_row`` underscore name).
    - ``test_task24_compare_uses_paths_helper`` and
      ``test_task25_no_hardcoded_v9_v9_v4s_fallback`` — read from new
      paths with legacy fallback.
    - ``test_task33_min01_marimo_paths`` — same dual-read pattern.
    - ``test_task34_*`` — import from
      ``sl_ads.evaluate.evaluate_injection`` directly.
    - ``test_task40_labeller_stl_*`` — monkeypatch the real module
      ``sl_ads.adapters.labeller_unsupervised`` instead of the shim.
    - ``TestResolveSlCsvPath`` — imports
      ``sl_ads.compare.compare_if_fair`` directly.

### Module count check

```
$ find src/sl_ads -type f -name "*.py" | wc -l
52
```

(40 implementation modules + 12 ``__init__.py``.)

### Test result at end of Phase 2

```
======================== 44 passed in 2.94s ==============================
```

---

## Phase 3 — what was done (2026-04-29)

Goal: every module under ``src/sl_ads/`` must import its dependencies
through the new dotted paths (``from sl_ads.<subpkg>.<module> import
…``) so that no inter-package traffic ever traverses a deprecation
shim at runtime.  The legacy shims at the project root remain in place
for **out-of-tree consumers only** (until Phase 7).

### Sub-batches

| Batch | Scope | Files touched | Status |
|-------|-------|---------------|--------|
| 3a | ``core/``, ``train/``, ``inject/`` | 5 (subjective_logic, opinions_pipeline, train_models, compute_evidence, evidence_level) | ✅ |
| 3b | ``qualify/``, ``evaluate/``, ``compare/`` | 8 (sbn_qualifier, argmax_baseline, evaluate_injection, evaluate_qualify_sbn, evaluate_qualify_injected, compare_if_fair, compare_qualif_methods, compare_labeller_vs_sl) | ✅ |
| 3c | ``ablation/``, ``audit/``, ``notebooks/``, ``adapters/`` | 9 (run_ablation, run_ablation_labeled, ablation_injection_level, ablation_temporal_sbn, ablation_sbn_novelty, audit_full_dataset, admin/compute_opinions/qualify_sbn notebooks, run_cross_dataset, gecco_adapter, rederio_adapter) | ✅ |
| 3d | Docstring-example imports + tracker update | 3 (paths, qualif_filters, utils_manifest) | ✅ |

### Import rewrite map applied

Every match of the patterns below in ``src/sl_ads/**/*.py`` was rewritten:

| Legacy form | New form |
|-------------|----------|
| ``from config import …``                 | ``from sl_ads.config import …`` |
| ``from paths import …``                  | ``from sl_ads.paths import …`` |
| ``from preprocessing_utils import …``    | ``from sl_ads.preprocessing_utils import …`` |
| ``from qualif_filters import …``         | ``from sl_ads.qualif_filters import …`` |
| ``from utils_manifest import …``         | ``from sl_ads.utils_manifest import …`` |
| ``import sl_formulas_v2 as sl``          | ``import sl_ads.core.subjective_logic as sl`` |
| ``from sl_formulas_v2 import …``         | ``from sl_ads.core.subjective_logic import …`` |
| ``from stats_bootstrap_ci import …``     | ``from sl_ads.stats.bootstrap_ci import …`` |
| ``from stats_mcnemar import …``          | ``from sl_ads.stats.mcnemar import …`` |
| ``from analysis_residual_correlation``   | ``from sl_ads.stats.residual_correlation`` |
| ``from inject_at_evidence_level import`` | ``from sl_ads.inject.evidence_level import`` |
| ``import config`` / ``import config as`` | ``import sl_ads.config`` / ``import sl_ads.config as`` |

The Marimo notebooks (``admin``, ``compute_opinions``, ``qualify_sbn``)
keep a *fallback* ``from config import CONFIG`` inside a nested
``try/except`` so they remain runnable from either the new package
layout or the legacy flat tree.

### Verification

Strict-mode smoke tests:

```python
# Raise if any legacy import path is touched at runtime
import sys, warnings
warnings.simplefilter("error", DeprecationWarning)
sys.path.insert(0, "src")

# 14 leaf+core modules: import cleanly, zero warnings
for m in [...]: __import__(m)

# 9 entry-point modules (train, evaluate, compare, …): same
for m in [...]: __import__(m)
```

Both passed; no shim is ever triggered by intra-package imports.

Pytest at the end of Phase 3:

```
======================== 44 passed in 2.84s ==============================
```

### One test had to be relaxed

``test_task24_compare_uses_paths_helper`` originally asserted the
exact string ``"from paths import"`` — it now accepts either that
form **or** the new ``"from sl_ads.paths import"`` form.  All other
tests passed without modification (Phase 2's adjustments still hold).

### Side-effects

* The notebooks' ``try/except`` fallback is the ONLY remaining path
  through the legacy ``config`` shim inside ``src/sl_ads/``.  This is
  intentional and stays until Phase 7.
* Three module docstrings (``paths.py``, ``qualif_filters.py``,
  ``utils_manifest.py``) had legacy ``from <name> import`` examples;
  these were updated to the new dotted form for consistency.

---

## Phase 4 — what was done (2026-04-29)

### 4a — New launcher ``run_pipeline.py``

* Written from scratch at the project root; replaces the legacy
  ``run_full_sl_ads.py``.
* Every step is now dispatched as ``python -m sl_ads.<subpkg>.<module>``
  rather than ``python <legacy_script>.py``, so sub-processes never
  traverse a deprecation shim at runtime.
* Sub-process environment exports ``PYTHONPATH=src`` so ``sl_ads.*``
  resolves regardless of where Python was launched.
* Pipeline tables (``_STEPS_REDERIO``, ``_STEPS_METR_LA``,
  ``_STEPS_GECCO``, ``_STEPS_CESNET``) updated to dotted module names.
* Argument surface preserved: ``--dataset``, ``--from-step``,
  ``--to-step``, ``--dry-run``, ``--list-steps``,
  ``--continue-on-error``.
* Two new flags added for the dual-write policy:
  ``--no-archive`` (skip the post-run snapshot) and ``--archive-dir``
  (override the historical archive root).

### 4b — Output / results dual-write policy

The launcher now implements the user's dual-archive request:

* ``outputs/`` — the *current* run.  Each script's ``RESULTS_DIR``
  resolves to ``../results/resultats_<version>/`` as before; the
  Phase H launcher additionally snapshots that location into a
  per-run archive after success.
* ``results/<run_id>/`` — the *historical archive*.  After the
  pipeline succeeds (i.e. all steps return 0 and no abort), the
  contents of ``outputs/`` are copied via ``shutil.copytree`` into
  ``results/<run_id>/``.  ``run_id`` is computed deterministically
  by :func:`sl_ads.utils_manifest.compute_run_id` from CONFIG +
  short git SHA + dataset name (TASK-32).  A ``_run_manifest.json``
  is dropped inside the archive directory with the run summary.

If ``run_id`` already exists in the archive, the snapshot is **not**
overwritten — historical immutability.

### 4c — Folder rename ``actual_outputs/`` → ``outputs/``

* Renamed via ``mv`` (preserved file timestamps, contents, sub-dirs).
* No code in the codebase referenced ``actual_outputs/`` directly
  (output paths flow through ``RESULTS_DIR``); the only mention was a
  legacy fallback in the new launcher itself which is now harmless.

### 4d — Legacy launcher shim

``run_full_sl_ads.py`` is now a 25-line deprecation shim that:

1. Adds the project root to ``sys.path`` (so legacy
   ``import run_full_sl_ads`` callers still find ``run_pipeline``);
2. Emits a ``DeprecationWarning``;
3. Re-exports ``run_pipeline``'s public API via ``import *`` for
   ``import run_full_sl_ads``;
4. Forwards execution via ``runpy.run_path("run_pipeline.py",
   run_name="__main__")`` when invoked as a script (preserving
   ``argv``).

### Verifications

```bash
# Argument-only smoke tests
python run_pipeline.py --list-steps                             # 10 steps
python run_pipeline.py --dataset METR-LA --list-steps           # 7 steps
python run_pipeline.py --dry-run --to-step train                # 1 dry-run

# Subprocess invocation pattern
PYTHONPATH=src python -m sl_ads.stats.bootstrap_ci              # [TEST] ALL PASS
PYTHONPATH=src python -m sl_ads.stats.mcnemar                   # [TEST] ALL PASS

# Legacy shim still works
python run_full_sl_ads.py --list-steps                          # forwards to run_pipeline

# Regression
pytest tests/                                                   # 44 passed in 2.94s
```

All four checks pass.

### Files added / modified in Phase 4

| File | Change |
|------|--------|
| ``run_pipeline.py`` | NEW — 380-line Phase H launcher |
| ``run_full_sl_ads.py`` | OVERWRITTEN — 25-line deprecation shim |
| ``actual_outputs/`` → ``outputs/`` | Folder renamed |

### Side-effects to watch in subsequent phases

* **DeprecationWarning verbosity** — pytest is run with
  ``-W ignore::DeprecationWarning`` while shims live; will be relaxed
  once Phase 7 closes.
* **Two physical copies of every module** — the copy at the legacy path
  is the shim (≈15 LOC), the copy at the new path is the real
  implementation.  Until Phase 7, do NOT edit the legacy file: any
  edit there will be lost when the shim is removed.

---

## Phase 5 — what was done (2026-04-29)

### 5a — `review/` → `docs/review/`

15 markdown files moved from the project-root `review/` folder into
`docs/review/`:

```
review/
├── AUDIT_SCIENTIFIQUE_PIPELINE.md
├── CHECKLIST_RAPPORT_TECHNIQUE_PIPELINE.md
├── CONSOLIDATED_AUDIT_REVIEW.md
├── HYPOTHESES_ET_MENACES_VALIDITE.md
├── M10_sbn_architecture_analysis.md
├── PUBLICATION_TABLES.md
├── SCIENTIFIC_AUDIT.md
├── review_compute_evidence_v2.md
├── review_compute_opinions_v3.md
├── review_evaluate_injection_v2.md
├── review_evaluate_qualify_sbn.md
├── review_inject_at_evidence_level.md
├── review_qualify_anomaly_sbn.md
├── review_qualify_anomaly_sbn_v1.md
└── review_qualify_anomaly_sbn_v2.md
```

The legacy `review/` folder now holds a single stub `README.md` that
points readers to the new location.  Stub will be removed in Phase 7.

### 5b — `docs/audit_*.md` → `docs/audit/`

Six audit-and-reconciliation documents that lived flat in `docs/`
have been grouped under `docs/audit/`:

| Old path | New path |
|----------|----------|
| `docs/audit_verification_tracker.md`           | `docs/audit/audit_verification_tracker.md` |
| `docs/scientific_audit_reconciliation_20260425.md` | `docs/audit/scientific_audit_reconciliation_20260425.md` |
| `docs/pipeline_reconciliation_20260425.md`     | `docs/audit/pipeline_reconciliation_20260425.md` |
| `docs/trust_discount_r2_analysis.md`           | `docs/audit/trust_discount_r2_analysis.md` |
| `docs/wu_keogh_self_assessment.md`             | `docs/audit/wu_keogh_self_assessment.md` |
| `docs/reviewer_target_calibration.md`          | `docs/audit/reviewer_target_calibration.md` |

`docs/honest_limitations.md` and `docs/RENAMING_LOG_PHASE_H.md` stay
at the top level of `docs/` because they are paper-facing material,
not audit artefacts.

### 5c — Cross-reference rewrite

Bulk-rewrote 56 markdown cross-references across 10 files so every
intra-doc link points at the new location.  Patterns:

| From | To |
|------|----|
| `docs/audit_verification_tracker.md`           | `docs/audit/audit_verification_tracker.md` |
| `docs/scientific_audit_reconciliation_20260425.md` | `docs/audit/scientific_audit_reconciliation_20260425.md` |
| `docs/pipeline_reconciliation_20260425.md`     | `docs/audit/pipeline_reconciliation_20260425.md` |
| `docs/trust_discount_r2_analysis.md`           | `docs/audit/trust_discount_r2_analysis.md` |
| `docs/wu_keogh_self_assessment.md`             | `docs/audit/wu_keogh_self_assessment.md` |
| `docs/reviewer_target_calibration.md`          | `docs/audit/reviewer_target_calibration.md` |
| `review/<NAME>.md` (without `docs/` prefix)    | `docs/review/<NAME>.md` |

The rewrite was done with `sed` and verified by re-running the
inverse `grep` — zero remaining old-form occurrences.

### 5d — Code references

Two source files mentioned the legacy paths in comments / docstrings;
both updated to point at the new locations:

* `src/sl_ads/evaluate/evaluate_qualify_injected.py` — error message
  references `docs/review/SCIENTIFIC_AUDIT.md`.
* `tests/test_audit_codex_remediation_20260427.py` — module
  docstring references `docs/audit/audit_verification_tracker.md`
  and `docs/audit/scientific_audit_reconciliation_20260425.md`.

(Two other code mentions of `docs/honest_limitations.md` —
`adapters/cesnet_adapter.py` and `train/train_models.py` — required
no change because that file stays at `docs/honest_limitations.md`.)

### 5e — `docs/README.md`

A new index file at the top of `docs/` lists every document with a
one-line purpose statement, grouped by sub-directory.  Also documents
the moves done in this phase so a reader landing on an old broken
link can immediately find the new location.

### Verification

```text
pytest tests/                         → 44 passed in 3.02s
grep -r 'docs/audit_verification_tracker.md' docs/    → 0 hits  (all rewritten)
grep -r '^review/' docs/                              → 0 hits  (all rewritten)
```

---

## Phase 6 — what was done (2026-04-29)

Goal: prove that the reorganised codebase is functionally equivalent
to the legacy layout, by running 8 verification axes back-to-back.

| Axis | Command | Result |
|------|---------|--------|
| **6a — Test suite** | ``pytest tests/`` | **44 passed in 2.90s** |
| **6b — Strict-mode imports** | ``warnings.simplefilter("error", DeprecationWarning); import each of 51 sl_ads.* names`` | **51/51 imports clean — zero shim hit at runtime** |
| **6c — `--list-steps` for all 4 datasets** | ``python run_pipeline.py --dataset {RedeRio,METR-LA,GECCO-IoT,CESNET-TimeSeries24} --list-steps`` | All 4 print correct dotted module paths |
| **6d — Full dry-run** | ``python run_pipeline.py --dry-run`` | 10/10 RedeRio steps marked dry-run; exit-summary JSON written |
| **6e — Subprocess `python -m sl_ads.<X>`** | 5 modules with ``--self-test``: ``stats.bootstrap_ci``, ``stats.mcnemar``, ``stats.residual_correlation``, ``ablation.ablation_injection_level``, ``ablation.ablation_temporal_sbn`` | **5/5 [TEST] ALL PASS** |
| **6f — Legacy shim entry-point** | ``python run_full_sl_ads.py --dataset GECCO-IoT --list-steps`` + ``import {stats_bootstrap_ci, paths, config}`` from project root | Pipeline forwarded correctly through ``runpy``; legacy imports succeed AND emit ``DeprecationWarning`` (3/3) |
| **6g — Smoke pipeline (real)** | ``python run_pipeline.py --from-step compare_if --to-step compare_if --no-archive`` | **6 seconds, exit 0** — verifies subprocess dispatch + ``PYTHONPATH=src`` injection + real I/O against existing artifacts |
| **6h — Archive logic** | ``python run_pipeline.py --from-step compare_if --to-step compare_if --archive-dir /tmp/sl_ads_archive_test`` (run twice) | Run 1: ``outputs/`` snapshotted to ``<archive>/d7ff4e1e2e9a774e/`` with ``_run_manifest.json``.  Run 2: same ``run_id`` re-computed deterministically, archive **not** overwritten — historical immutability respected |

### Pre-existing fragility found

While doing the strict-import test (6b), importing
``sl_ads.evaluate.evaluate_qualify_injected`` raised
``TypeError: 'NoneType' object is not iterable`` because the module's
top-level code did ``raw_catalog = CONFIG.get("ATTACK_CATALOG", [])``,
which returns ``None`` (not ``[]``) when ``CONFIG`` has the key set
to ``None`` (the case on RedeRio in some configurations).

This is a **pre-existing bug**, not a Phase H regression — the
legacy layout simply never imported the module without first
running its ``__main__`` block via ``python <script>.py`` from a
context that pre-populated ``CONFIG``.  We made a one-line defensive
fix:

```diff
- raw_catalog = CONFIG.get("ATTACK_CATALOG", [])
+ raw_catalog = CONFIG.get("ATTACK_CATALOG") or []
```

(``None or [] == []``, while ``dict.get(k, [])`` returns ``None`` when
``k`` exists with value ``None``.)

### Verdict

The reorganised codebase is functionally equivalent to the legacy
layout on every axis tested.  No regression.  Phase 6 closed; Phase 7
(shim removal + cleanup) is now safe to execute.

---

## Phase 7 — what was done (2026-04-29)

Goal: with the package layout fully verified, **delete every
deprecation shim, every legacy folder, and every legacy artefact**.
After Phase 7, no out-of-tree caller can import a legacy name —
which is the whole point: the ``sl_ads`` package becomes the only
reachable code path.

### 7a — Audit consumers, rewrite remaining legacy imports

A final ``grep`` over ``tests/`` revealed legacy imports that had
survived Phase 3 because tests were never strict-mode-checked:

* ``tests/test_fusion_wbf_canonical.py`` → ``from sl_formulas_v2 import …``
  → rewritten to ``from sl_ads.core.subjective_logic import …``.
* ``tests/test_audit_remediation_20260426.py`` → 5× ``import sl_formulas_v2 as sl``
  + 3× ``from utils_manifest import compute_run_id`` → rewritten to
  ``import sl_ads.core.subjective_logic as sl`` and
  ``from sl_ads.utils_manifest import compute_run_id``.
* ``tests/test_audit_codex_remediation_20260427.py`` → 6× ``import config as _cfg``
  + 1× ``from rederio_adapter import RederioAdapter``
  + 2× ``from preprocessing_utils import preprocess_metrics``
  + 1× ``from gecco_adapter import GeccoAdapter``
  + 2× ``from paths import …``
  → all rewritten to absolute ``sl_ads.*`` paths.
* ``tests/test_resolve_sl_csv_path.py`` already used
  ``import sl_ads.compare.compare_if_fair`` directly (Phase 2 fix).

Three test files now bootstrap ``sys.path`` defensively to add ``src/``
even if ``conftest.py`` is somehow not loaded:

```python
_SRC = _PROJ / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
```

After the rewrite, ``pytest tests/ -W error::DeprecationWarning`` returned
**44 passed in 5.33s** — every test path is now legacy-shim-free.

### 7b — Sentinel test-file path rewrites

A handful of tests read source files by path to grep their content
(e.g. checking the ``raise FileNotFoundError`` block in
``opinions_pipeline.py``).  Those paths were updated to point at
``src/sl_ads/<subpkg>/<module>.py``:

| Test | Old path | New path |
|------|----------|----------|
| ``test_task20_compute_opinions_raises_on_missing_attacks`` | ``compute_opinions_v3.py`` | ``src/sl_ads/core/opinions_pipeline.py`` |
| ``test_task24_compare_uses_paths_helper``                  | ``compare_qualif_methods.py`` | ``src/sl_ads/compare/compare_qualif_methods.py`` |
| ``test_task25_no_hardcoded_v9_v9_v4s_fallback``            | ``evaluate_qualify_sbn.py`` | ``src/sl_ads/evaluate/evaluate_qualify_sbn.py`` |
| ``test_task28_no_global_ignore_in_4_scripts``              | 4 root-level scripts | 4 ``src/sl_ads/`` paths |
| ``test_task33_min01_marimo_paths``                         | ``marimo_*.py`` (root) | ``src/sl_ads/notebooks/*.py`` |
| ``test_task33_min03_pearson_uses_config``                  | ``modèle évaluation/compute_pearson_independence.py`` | ``investigations/compute_pearson_independence.py`` |
| ``test_task43_paths_docstrings_no_train_v9``               | ``paths.py`` (root) | ``src/sl_ads/paths.py`` |

All tests use a dual-path pattern (``new if exists else legacy``) so
they remain runnable on a checkout where one of the two layouts is
missing.

### 7c — Shim deletion

**32 deprecation shims deleted** in a single batch:

* 22 root-level Python shims:
  ``config``, ``paths``, ``preprocessing_utils``, ``qualif_filters``,
  ``utils_manifest``, ``sl_formulas_v2``, ``stats_bootstrap_ci``,
  ``stats_mcnemar``, ``analysis_residual_correlation``,
  ``inject_at_evidence_level``, ``compute_evidence_v2``,
  ``compute_opinions_v3``, ``qualify_anomaly_sbn``,
  ``qualify_argmax_baseline``, ``evaluate_injection_v2``,
  ``evaluate_qualify_sbn``, ``evaluate_qualify_injected``,
  ``compare_if_fair``, ``compare_qualif_methods``,
  ``compare_labeller_vs_sl``, ``train_v10``, ``run_ablation_v2``,
  ``run_ablation_labeled``, ``audit_full_dataset``.
* 5 ablation root-level shims: ``ablation_fusion_mode``,
  ``ablation_injection_level``, ``ablation_nan_ffill``,
  ``ablation_sbn_novelty``, ``ablation_temporal_sbn``.
* 3 marimo root-level shims: ``marimo_admin``,
  ``marimo_compute_opinions``, ``marimo_qualify_sbn``.
* The 8-file ``dataset_adapter/`` folder (entirely shims).
* The 2 bootstraps (``_phase_h_path.py`` at root,
  ``run_full_sl_ads.py`` legacy launcher).
* The ``review/`` stub folder (replaced by ``docs/review/``).

(Total = 22 + 5 + 3 + 8 + 2 + 1 = 41 paths removed.)

### 7d — Other legacy artefacts

* ``test.py`` (38-line ad-hoc inspection script at root) → preserved
  with imports rewritten and moved to
  ``investigations/inspect_qualify_types_during_attacks.py``.  Only
  the legacy file at the root was removed.
* ``modèle évaluation/`` (15 files, non-ASCII name) → contents moved
  to ``investigations/`` with one ASCII rename
  (``Verify outage episodes.py`` → ``verify_outage_episodes.py``).
  The unicode-named folder no longer exists.
* In-source string references to ``"modèle évaluation"`` in 3
  modules updated to ``"investigations"`` (defensive defaults +
  comments).

### 7e — Verification

```text
pytest tests/                                  → 44 passed in 3.81s
pytest tests/ -W error::DeprecationWarning     → 44 passed in 3.99s   (zero shim hits)
python run_pipeline.py --from-step compare_if  → 8s, exit 0           (real subprocess)
```

### 7f — Final layout

```
sl_ads/                          (project root — to be renamed in a final commit)
├── MANIFEST.md
├── README.md / pyproject.toml / requirements.txt / conftest.py
├── run_pipeline.py              (single-binary entry point)
├── src/
│   └── sl_ads/                  (52 .py files = 40 modules + 12 __init__.py)
│       ├── __init__.py / config.py / paths.py / preprocessing_utils.py
│       ├── qualif_filters.py / utils_manifest.py
│       ├── core/   train/   inject/   qualify/   evaluate/
│       ├── compare/   ablation/   audit/   adapters/   notebooks/   stats/
├── tests/                       (44 tests, all green strict-mode)
├── investigations/              (16 ad-hoc scripts; ex-`modèle évaluation/`)
├── docs/
│   ├── README.md / RENAMING_LOG_PHASE_H.md / honest_limitations.md
│   ├── audit/                   (6 audit & remediation docs)
│   └── review/                  (15 scientific reviews)
├── outputs/                     (current run; overwritten each pipeline)
├── results/                     (historical archive: results/<run_id>/…)
└── references/                  (Jøsang 2016 PDF, etc.)
```

### Closed

Phase H is closed.  No deferred items.  The repo is ASCII-only,
package-shaped, strict-mode-clean, and the launcher snapshots every
successful run into a deterministic, immutable archive.

---

## Renaming map (full canonical list)

The mapping below is the contract for Phase 2-7.  Entries marked
"DONE in phase X" become read-only after that phase closes.

### Module renames (legacy flat → ``src/sl_ads/<subpkg>/<module>.py``)

| Legacy path | New path | Reason | Phase |
|-------------|----------|--------|-------|
| ``train_v10.py``                | ``src/sl_ads/train/train_models.py``                 | drop ``_v10`` version suffix; group with evidence | 2 |
| ``compute_evidence_v2.py``      | ``src/sl_ads/train/compute_evidence.py``             | drop ``_v2``; train-time helper | 2 |
| ``compute_opinions_v3.py``      | ``src/sl_ads/core/opinions_pipeline.py``             | drop ``_v3``; clarify it's a *pipeline*, not raw operators | 2 |
| ``sl_formulas_v2.py``           | ``src/sl_ads/core/subjective_logic.py``              | drop ``_v2``; descriptive name | 2 |
| ``inject_at_evidence_level.py`` | ``src/sl_ads/inject/evidence_level.py``              | the word ``inject`` is now the package name | 2 |
| ``qualify_anomaly_sbn.py``      | ``src/sl_ads/qualify/sbn_qualifier.py``              | shorter, package-qualified | 2 |
| ``qualify_argmax_baseline.py``  | ``src/sl_ads/qualify/argmax_baseline.py``            | drop redundant ``qualify_`` prefix | 2 |
| ``evaluate_injection_v2.py``    | ``src/sl_ads/evaluate/evaluate_injection.py``        | drop ``_v2`` | 2 |
| ``evaluate_qualify_sbn.py``     | ``src/sl_ads/evaluate/evaluate_qualify_sbn.py``      | (kept name; only directory moves) | 2 |
| ``evaluate_qualify_injected.py``| ``src/sl_ads/evaluate/evaluate_qualify_injected.py`` | (kept name; only directory moves) | 2 |
| ``compare_if_fair.py``          | ``src/sl_ads/compare/compare_if_fair.py``            | (kept name; only directory moves) | 2 |
| ``compare_qualif_methods.py``   | ``src/sl_ads/compare/compare_qualif_methods.py``     | (kept name; only directory moves) | 2 |
| ``compare_labeller_vs_sl.py``   | ``src/sl_ads/compare/compare_labeller_vs_sl.py``     | (kept name; only directory moves) | 2 |
| ``run_ablation_v2.py``          | ``src/sl_ads/ablation/run_ablation.py``              | drop ``_v2`` | 2 |
| ``run_ablation_labeled.py``     | ``src/sl_ads/ablation/run_ablation_labeled.py``      | (kept name; only directory moves) | 2 |
| ``ablation_fusion_mode.py``     | ``src/sl_ads/ablation/ablation_fusion_mode.py``      | (kept name; only directory moves) | 2 |
| ``ablation_injection_level.py`` | ``src/sl_ads/ablation/ablation_injection_level.py``  | (kept name; only directory moves) | 2 |
| ``ablation_nan_ffill.py``       | ``src/sl_ads/ablation/ablation_nan_ffill.py``        | (kept name; only directory moves) | 2 |
| ``ablation_sbn_novelty.py``     | ``src/sl_ads/ablation/ablation_sbn_novelty.py``      | (kept name; only directory moves) | 2 |
| ``ablation_temporal_sbn.py``    | ``src/sl_ads/ablation/ablation_temporal_sbn.py``     | (kept name; only directory moves) | 2 |
| ``stats_bootstrap_ci.py``       | ``src/sl_ads/stats/bootstrap_ci.py``                 | drop redundant ``stats_`` prefix | 2 |
| ``stats_mcnemar.py``            | ``src/sl_ads/stats/mcnemar.py``                      | drop redundant ``stats_`` prefix | 2 |
| ``analysis_residual_correlation.py`` | ``src/sl_ads/stats/residual_correlation.py``    | drop ``analysis_`` prefix; group with stats | 2 |
| ``audit_full_dataset.py``       | ``src/sl_ads/audit/audit_full_dataset.py``           | (kept name; only directory moves) | 2 |
| ``marimo_admin.py``             | ``src/sl_ads/notebooks/admin.py``                    | drop ``marimo_`` prefix (sub-package = notebooks) | 2 |
| ``marimo_compute_opinions.py``  | ``src/sl_ads/notebooks/compute_opinions.py``         | idem | 2 |
| ``marimo_qualify_sbn.py``       | ``src/sl_ads/notebooks/qualify_sbn.py``              | idem | 2 |
| ``config.py``                   | ``src/sl_ads/config.py``                             | promote to package-level (kept name) | 2 |
| ``paths.py``                    | ``src/sl_ads/paths.py``                              | promote to package-level (kept name) | 2 |
| ``preprocessing_utils.py``      | ``src/sl_ads/preprocessing_utils.py``                | promote to package-level (kept name) | 2 |
| ``qualif_filters.py``           | ``src/sl_ads/qualif_filters.py``                     | promote to package-level (kept name) | 2 |
| ``utils_manifest.py``           | ``src/sl_ads/utils_manifest.py``                     | promote to package-level (kept name) | 2 |
| ``run_full_sl_ads.py``          | ``run_pipeline.py`` (project root, top-level launcher) | drop redundant project-name; clearer entry point | 4 |

### Folder renames

| Legacy | New | Reason | Phase |
|--------|-----|--------|-------|
| ``dataset_adapter/``     | ``src/sl_ads/adapters/``  | inside the package; English plural | 2 |
| ``modèle évaluation/``   | ``investigations/``       | ASCII; reflects the *exploratory* nature of these scripts | 2 |
| ``actual_outputs/``      | ``outputs/``              | clearer name | 4 |
| ``review/``              | ``docs/review/``          | reviews are documentation | 5 |

### Files to delete

| Path | Reason | Phase |
|------|--------|-------|
| ``test.py`` (project root, empty stub) | dead file | 7 |

### Documentation moves (Phase 5)

The following are *moves only* (no rename):

| Legacy | New |
|--------|-----|
| ``docs/audit/audit_verification_tracker.md``           | ``docs/audit/audit_verification_tracker.md`` |
| ``docs/audit/scientific_audit_reconciliation_20260425.md`` | ``docs/audit/scientific_audit_reconciliation_20260425.md`` |
| ``docs/audit/pipeline_reconciliation_20260425.md``     | ``docs/audit/pipeline_reconciliation_20260425.md`` |
| ``docs/audit/reviewer_target_calibration.md``          | ``docs/audit/reviewer_target_calibration.md`` |
| ``docs/audit/trust_discount_r2_analysis.md``           | ``docs/audit/trust_discount_r2_analysis.md`` |
| ``docs/audit/wu_keogh_self_assessment.md``             | ``docs/audit/wu_keogh_self_assessment.md`` |
| ``docs/review/CONSOLIDATED_AUDIT_REVIEW.md`` (and 14 siblings) | ``docs/review/CONSOLIDATED_AUDIT_REVIEW.md`` (idem) |

``docs/honest_limitations.md`` stays at the top of ``docs/`` — it's a
high-level disclosure file, not an audit artefact.

---

## Output-directory policy (Phase 4)

The user wants two parallel locations for run artefacts:

* ``outputs/`` — the *current* run.  Overwritten on every execution.
  Use this for routine inspection.
* ``results/<run_id>/`` — the *historical archive*.  Each run is
  written under a directory named after the deterministic
  ``run_id`` (cf. TASK-32, ``utils_manifest.compute_run_id``).  Older
  runs accumulate, allowing diff-style comparisons across experiments.

Implementation sketch (Phase 4):

1. The launcher computes ``run_id`` once, exports it to subprocesses.
2. Each script's ``OUTPUT_DIR`` resolves to ``outputs/<step_name>``.
3. After the pipeline completes successfully, the launcher copies (or
   hard-links) the contents of ``outputs/`` into
   ``results/<run_id>/`` and writes a ``MANIFEST.md`` snapshot.
4. ``outputs/`` itself is *not* git-tracked; ``results/`` is keep-as-is
   (git-ignored, but the user keeps it on disk for diff purposes).

---

## Compatibility shim policy (Phase 2-7)

For each module ``X.py`` moved in Phase 2, a 3-line shim replaces the
legacy path::

    # X.py — DEPRECATION SHIM (Phase H reorganisation, 2026-04-27).
    # See docs/RENAMING_LOG_PHASE_H.md for the full migration map.
    import warnings as _w
    _w.warn(
        "Importing from `X` is deprecated; use "
        "`sl_ads.<subpkg>.<module>` instead.",
        category=DeprecationWarning, stacklevel=2,
    )
    from sl_ads.<subpkg>.<module> import *  # noqa: F401,F403
    from sl_ads.<subpkg>.<module> import __dict__ as _new
    globals().update(_new)

Removal date: after one full cycle (full pytest + smoke pipeline run)
confirms callers have been migrated.  Tracked as **PHASE-7-CLEANUP**
in the launcher's exit summary.

---

## Verification commands

After each phase, run::

    pytest tests/ -v

Expected outcome: ``44 passed`` (Phase F + G regression suites + fusion
canonical + path resolution).  Any deviation reverts the phase.

---

*Document started 2026-04-27.  Update each phase boundary.*
