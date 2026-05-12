# Artifact Appendix — SL-ADS

**Submission template** : USENIX Security AE / IEEE S&P AE / ACM CCS AE
(based on the SecArtifacts and ACM artifact-review guidelines, 2026).
**Length** : ≤ 3 pages.

**Status note (2026-05-12).** Values and artifact paths below are tied to the
complete RedeRio 17-leaf run `2e12261d55a8f975`. Historical placeholder claims
from earlier reconstruction-only or partial runs are superseded.

## A.1 Abstract

This artifact accompanies *"Subjective-Logic Anomaly Detection for
Network Telemetry"*.  It is a self-contained Python 3.10+ pipeline
that reproduces every quantitative claim in the paper from the raw
RedeRio / METR-LA / GECCO-IoT / CESNET-TimeSeries24 captures.  The
pipeline produces opinion vectors, calibrated decision thresholds,
F1 / MCC / AUC metrics with bootstrap CIs, an SBN cause-attribution
qualifier output, same-evidence no-SL comparators, and raw-data baseline
comparisons.  Output artifacts are deterministically archived under
``results/<run_id>/`` where ``run_id`` is the first 16 hex chars of
SHA-256(CONFIG ∥ git_sha ∥ dataset).

## A.2 Description & requirements

### A.2.1 Hardware

* CPU : x86_64, ≥ 8 logical cores recommended (Prophet training is
  the bottleneck).
* RAM : ≥ 16 GB.
* Disk : ≥ 4 GB free under the project root (artifacts + per-run
  archive).

### A.2.2 Software

* OS : Linux / Windows / macOS (tested on Windows 11, Python 3.13).
* Python : ≥ 3.10 (3.13 used to produce the published figures).
* Frozen dependencies : ``requirements.txt`` (numpy 2.4.4,
  pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0, statsmodels 0.14.6,
  prophet 1.3.0, matplotlib 3.10.9, joblib 1.5.3).

### A.2.3 Data

Data files are not committed to git.  Place the raw files under
``../data/<dataset>/raw/`` (paths resolved by
``DATASETS_CONFIG`` in ``src/sl_ads/config.py``).  Sources :

* **RedeRio** : UFRJ/RedeRio capture (30 s aggregation; provenance/licence
  paragraph to be confirmed before public release).
* **METR-LA** : Li et al. 2018 DCRNN (5-min sensor speed).
* **GECCO-IoT** : Beck & Jawale 2018 (1-min water-quality).
* **CESNET-TimeSeries24** : Čejka et al. 2023 Nature SD (10-min,
  synthetic timestamps — disclosed in
  ``docs/honest_limitations.md`` MAJ-05).

## A.3 Security, privacy, and ethical concerns

The pipeline runs offline on archived telemetry; it never opens
network sockets and does not require root privileges.  The injected
attacks are synthetic perturbations of evidence vectors (no real
attack traffic generated).  Ethical use : intended for defensive
anomaly-detection research; redistribution of derivative
training-data must comply with the source dataset licences.

## A.4 Installation

```bash
git clone <anonymised-repository-URL>     # in submission, Zenodo DOI
cd sl_ads
python -m venv .venv && source .venv/bin/activate   # or .\.venv\Scripts\activate
pip install -r requirements.txt
pytest tests/                              # full suite should PASS on a clean machine
```

## A.5 Major claims

The paper makes 5 quantitative claims; each is mapped to a
reproducibility experiment.

| Claim | Experiment | Estimated time | Hardware |
|-------|------------|----------------|----------|
| C1 - RedeRio detection metrics at the calibrated threshold | E1 (§A.6.1) | several hours | 8-core CPU |
| C2 - Attack/event coverage under both F1 protocols | E1 (§A.6.1) | included in E1 | - |
| C3 - Same-evidence SL vs no-SL comparison is leak-free and reported with paired tests | E3 (§A.6.3) | ~1 min | CPU |
| C4 - Raw-data IF/LOF/OCSVM/SGD-OCSVM/PCA/Robust-Z baselines are reported only on raw-valid protocols | E3 (§A.6.3) | ~3 min | CPU |
| C5 - Audit tracker current; high-priority findings are resolved or explicitly disclosed | E4 (§A.6.4) | ~6 s | - |

## A.6 Evaluation

### A.6.1 Experiment E1 — main result

```bash
python run_pipeline.py --dataset RedeRio
```

* **Expected output**: ``outputs/`` populated with 43 files (CSV +
  PNG); ``results/<run_id>/_run_manifest.json`` with the full
  exit-summary; metrics printed to stdout.
* **Compare to paper** : use the metrics from the current complete run only.
  Record the commit hash, threshold sidecar, both F1 protocol rows
  (`catalog_outages_separate` and `operator_faithful_anomaly`), and output
  archive alongside any copied value.

### A.6.2 Experiment E2 — ablation

The ablation step is included in E1.  The headline ablation
comparison appears in
``outputs/ablation_summary.csv`` using the calibrated operational threshold.
Alternative threshold rows are separated into
``outputs/ablation_threshold_sensitivity.csv`` so they cannot be mistaken for
headline results.  The ranking table is printed under *"SUMMARY @ THRESHOLD"*.

### A.6.3 Experiment E3 — comparison baselines

The complete pipeline runs the legacy `compare_if` step. For paper-facing
SL-vs-no-SL and raw-baseline tables, also run:

```powershell
python -m sl_ads.compare.compare_no_sl_fair
$env:SL_FORCE_NONINJECTED_OPINIONS = "1"
$env:SL_SKIP_OPINION_PLOTS = "1"
python -m sl_ads.core.opinions_pipeline
$env:SL_FORCE_NONINJECTED_OPINIONS = $null
$env:SL_SKIP_OPINION_PLOTS = $null
python -m sl_ads.compare.compare_raw_baselines_fair
```

Expected outputs:

* `evaluation_no_sl_fair/no_sl_fair_summary.csv`
* `evaluation_no_sl_fair/no_sl_fair_paired_vs_sl.csv`
* `opinions_non_injected/detection_results_RAW.csv`
* `evaluation_raw_baselines/raw_baselines_summary.csv`
* `evaluation_raw_baselines/raw_baselines_paired_vs_sl.csv`

The first script is the direct same-evidence "with vs without SL" comparator.
The forced non-injected opinions run creates a raw-only SL score without
overwriting the injected detection CSV. The second comparison script trains IF,
LOF, exact RBF One-Class SVM, SGDOneClassSVM, robust-z, and PCA baselines on
raw metrics only. It deliberately excludes synthetic catalog windows because
the catalog attacks are injected at evidence level and have no raw-traffic
counterpart.

### A.6.4 Experiment E4 — test suite

```bash
pytest tests/ -W error::DeprecationWarning
```

Expected: the full suite passes on a clean machine. The exact count evolves as
new audit guards are added; use `docs/audit/audit_verification_tracker.md` as
the authoritative list of required checks. This validates:

* WBF canonical fusion property (bijection, idempotence,
  asymmetric-confidence, dogmatic-limit) — 8 tests
* Extended inter-method fusion operators (WBF, ABF, CBF, BCF,
  projected CCF, MinBF, MaxBF, hierarchical) plus method-group policy
  guards
* CBF + ageing + trust + contextual + conflict — 22 tests
* Adapter contract (4 adapters × schema invariants) — 14 tests
* CONFIG schema + threshold-sidecar round-trip + fusion-mode
  sidecar — 39 tests
* CLI launcher (``--list-steps``, ``--dry-run``, slicing,
  ``--no-archive``) — 11 tests
* Edge cases (empty / single-row / all-NaN / degenerate stats) —
  10 tests
* Audit-finding non-regression (Phase F/G/I guards and fusion sidecars)

## A.7 Customization

* Switch dataset : ``--dataset {RedeRio, METR-LA, GECCO-IoT,
  CESNET-TimeSeries24}`` or set ``ACTIVE_DATASET`` in
  ``src/sl_ads/config.py``.
* Switch fusion mode : ``CONFIG['INTER_METHOD_FUSION']`` in
  ``{wbf, abf, cbf, bcf, ccf, minbf, maxbf, hierarchical}`` or env-var
  ``SL_INTER_METHOD_FUSION_OVERRIDE``.
* Extend method groups : edit ``CONFIG['FUSION_METHOD_GROUPS']`` to add a
  third forecasting/reconstruction family without changing the fusion
  pipeline code.
* Sweep λ_decay, IF contamination, SBN novelty threshold, NaN ffill
  limit : ``src/sl_ads/ablation/run_ablation.py`` orchestrates all
  documented sensitivity tables.

## A.8 Notes on reproducibility

* Determinism : every successful run produces the same ``run_id``.
  Re-running on the same checkout + same dataset reuses the existing
  archive directory (immutability respected).
* Threshold calibration : the generic threshold sidecar
  ``trained_models_*_threshold.json`` is regenerated at every ``train`` step
  with EVT/FPR-target = 0.001. Mode-specific sidecars
  ``trained_models_*_threshold_<mode>.json`` may also be emitted for strict
  WBF/ABF ablations; the default WBF reference remains the generic production
  sidecar unless a validated WBF-specific sidecar is present.
* Strict WBF/ABF comparison : historical diagnostic evidence is preserved at
  ``results/fusion_mode_recalibrated/20260507_110115/``. This run
  recalibrated both modes independently and kept WBF as the default
  (`F1=0.7057`, `MCC=0.7087`, `FPR=4.31 %`) because ABF was slightly worse
  (`F1=0.7046`, `MCC=0.7077`, `FPR=4.34 %`).
* Anti-leakage guards : two CRITICAL findings of the audit_codex
  (CRIT-01 — argmax-on-test for threshold; CRIT-03 — best-of IF
  contamination) are now **structurally impossible** in the public
  code path; they are reachable only via the explicit escape
  hatches ``SL_ALLOW_TEST_TUNED_THRESHOLD=1`` and
  ``SL_ALLOW_TEST_TUNED_IF=1`` (with documented ``UserWarning``).
* ``docs/AUDIT_CURRENT_STATUS.md``, ``docs/honest_limitations.md`` and
  ``docs/audit/audit_verification_tracker.md`` give the current disclosure of
  every known limitation, resolved item, and partial fix. Superseded 2026-04
  reconciliation drafts are preserved under
  ``docs/archive/2026-05-07_audit_cleanup/``.

## A.9 Version

Software v1.0.0 (Phase H, 2026-04-29).  Historical reorganisation notes are
archived under ``docs/archive/``.
