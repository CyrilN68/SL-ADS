# Current Audit Status - SL-ADS

**Status date:** 2026-05-13 (deep consistency check against complete 17-leaf run artifacts)
**Scope:** current scientific, methodological, reproducibility, and audit state
for `current_version/`.

This is the canonical entry point for the audit trail. Historical audit drafts
are preserved under `docs/archive/`; they should not be used as current status
unless this file or `audit/audit_verification_tracker.md` explicitly points to
them.

**2026-05-12 reference-run note.** The complete 17-leaf RedeRio pipeline was
rerun end-to-end from `run_pipeline.py` and finished successfully:
run id `2e12261d55a8f975`, 10/10 steps OK, total runtime 582m28s. The previous
2026-05-10 reconstruction-only metrics are diagnostic only and are superseded
for paper-facing detection, FPR, outage, and qualifier numbers.

**2026-05-13 verification note.** The headline claims below were rechecked
against `_run_manifest.json`, `evaluation/eval_f1_protocol_comparison.csv`,
`evaluation/eval_vus_summary.csv`,
`eval_qualify_summary_qualif_types_sbn_20260512_003447.json`,
`ablation_uniform/ablation_summary.csv`,
`evaluation_no_sl_fair/*.csv`, `evaluation_raw_baselines/*.csv`, and
`outputs/scientific_hardening/regime_fpr*.{csv,json}`. The modern TSAD harness
is a plan/probe scaffold only; it is not a completed paper-facing baseline.

## Executive Verdict

SL-ADS is in a reviewer-defensible state for a serious paper **if the claims are
kept inside the documented scope**:

- RedeRio is the only fully reported dataset.
- Synthetic evidence-level injection is the primary controlled evaluation.
- WBF with uniform weights remains the reference inter-method fusion.
- CBF is legacy/ablation only because Prophet and Reconstruction are dependent.
- ABF is implemented and theoretically plausible for dependent methods, but the
  strict 2026-05-07 recalibration did **not** justify switching the default.
- The qualifier is an expert-template-driven Subjective Logic qualifier, not a
  canonical Bayesian network.
- Realised FPR can exceed the nominal calibration target; this is documented and
  must be discussed, not hidden.
- Trust-discount alternatives have been formally evaluated (PATCH D5,
  2026-05-07): both `WBF_WEIGHT_MODE='trust_discount'` (R^2) and `'mase'`
  (Hyndman-Koehler 2006) exhibit symmetric pathologies on RedeRio at 30 s
  sampling; `'uniform'` remains the published default.
- The previous `NETWORK_OUTAGE_NOV17` 0/3 diagnostic is superseded by the
  complete run: `NETWORK_OUTAGE_NOV17` is detected on 1/3 windows and
  `NETWORK_OUTAGE_DEC1617` on 188/339 windows. Outages must be reported
  explicitly, and both F1 protocols must be shown.
- The original "ADS with SL vs same ADS without SL" question is now answered
  by `compare_no_sl_fair.py`: Full SL-ADS beats the leak-free all-leaf
  `no_sl_mean_N` comparator on F1 micro (0.8666 vs 0.8268), MCC (0.8587 vs
  0.8185), and FPR (0.965% vs 1.665%), with paired block-bootstrap
  delta F1 = +0.040 [0.011, 0.085]. Reconstruction-only remains a strong diagnostic
  no-SL baseline and must not be hidden.
- Interpretation guardrail: this is a modest positive gain, not a large enough
  F1 jump to justify SL complexity by itself. The paper should frame the added
  value as uncertainty-aware fusion, auditable semantics, outage/event
  robustness, and compatibility with the qualifier.
- Raw-data IF / LOF / OCSVM / SGD-OCSVM / PCA / robust-z baselines are now reported separately by
  `compare_raw_baselines_fair.py`. They are valid for raw pseudo-label and
  real-event protocols, but not for the synthetic catalog task because those
  attacks are injected at evidence level rather than into raw traffic. The SL
  row uses the non-injected output `opinions_non_injected/detection_results_RAW.csv`.

Headline complete-run values from `current_version/results/2e12261d55a8f975/`
and `../results/resultats_RedeRio_trained_v4s_v4_v3/`:

| Metric | Value |
|---|---:|
| Catalog/outages-separate F1 micro | 0.8666 |
| Catalog/outages-separate F1 macro | 0.9292 |
| Operator-faithful anomaly F1 micro | 0.8257 |
| MCC | 0.8587 |
| Window TPR | 0.8877 |
| Realised global FPR | 0.965% |
| FPR target ratio | 9.65x target |
| Attacks detected | 14/14 |
| VUS-PR / VUS-ROC | 0.6038 / 0.8562 |
| SBN qualifier macro DR / QP / F1 | 91.1% / 67.6% / 65.0% |

## Current Reference Configuration

| Item | Current status |
|---|---|
| Dataset | RedeRio reference run |
| Reference run id | `2e12261d55a8f975` |
| Main pipeline | `run_pipeline.py` |
| Detection score | `FINAL_SYSTEM_CBF_proj_atk`; prefix is historical |
| Default fusion | `CONFIG['INTER_METHOD_FUSION'] = 'wbf'` |
| Fusion alternatives | `wbf`, `abf`, `cbf`, `bcf`, `ccf`, `minbf`, `maxbf`, `hierarchical` |
| Method grouping | `CONFIG['FUSION_METHOD_GROUPS']`, extensible to future method families |
| Published threshold source | generic production threshold sidecar (`delta = 0.102614`) |
| ABF status | implemented; not default on RedeRio |
| Trust-discount status | implemented; opt-in only; documented pathology |
| Calendar-aware EVT | implemented audit-grade opt-in; default OFF |
| Modern TSAD baselines | plan/probe scaffold only; no paper-facing execution yet |

## Authoritative Active Files

| Need | Canonical file |
|---|---|
| Current audit state | `docs/AUDIT_CURRENT_STATUS.md` |
| Finding-by-finding tracker | `docs/audit/audit_verification_tracker.md` |
| Documentation rules | `docs/DOCS_GOVERNANCE.md` |
| Paper limitations | `docs/honest_limitations.md` |
| Reproducibility checklist | `docs/REPRODUCIBILITY_CHECKLIST.md` |
| Artifact appendix | `docs/ARTIFACT_APPENDIX.md` |
| Formal assumptions | `docs/scientific_deconstruction/ASSUMPTIONS.md` |
| Formal methods inventory | `docs/scientific_deconstruction/METHODS.md` |
| Pipeline logic | `docs/scientific_deconstruction/PIPELINE_LOGIC.md` |
| Theory graph | `docs/scientific_deconstruction/THEORY_GRAPH.md` |
| References and citation discipline | `docs/scientific_deconstruction/REFERENCES.md` |
| Fusion-operator ablation | `docs/review/FUSION_OPERATOR_ABLATION_20260506.md` |
| Publication tables | `docs/review/PUBLICATION_TABLES.md` |
| SBN terminology analysis | `docs/review/M10_sbn_architecture_analysis.md` |
| Trust-discount pathology | `docs/audit/trust_discount_r2_analysis.md` |
| Wu and Keogh self-assessment | `docs/audit/wu_keogh_self_assessment.md` |

## Consolidated Scientific Risks

| ID | Risk | Current treatment |
|---|---|---|
| A1.1 | Training span may contain unlabelled anomalies | Defensive train-span audit exists; top suspects require manual review. |
| A1.5 | Single global EVT thresholds are regime-sensitive | Measured on the complete run; canonical ACTIVE FPR overshoots target. Calendar-aware EVT is implemented as audit-grade opt-in and remains default OFF for the paper reference. |
| A1.9 | Threshold calibration can drift from deployed score | Runtime sidecar checks exist; mode-specific WBF/ABF sidecars exist for ablation; realised FPR drift remains disclosed. |
| A3.2 | "Normal" windows may include real incidents | REAL_DDOS and outage windows are excluded from detector/qualifier FAR bases. |
| A3.3 | Synthetic signatures may be too clean | Gaussian, Cauchy, and Student-t signature-noise ablations are measured; raw-traffic synthetic injection and real heavy-tailed collection noise remain future work. |
| A4.7 | CBF independence assumption is violated | WBF default; CBF warns and is ablation-only. |
| A6.1 | Qualifier naive-Bayes dependence is violated | Correlations measured; argmax robust enough for current scope, ROC-style claims constrained. |
| A6.3 | Unknown-attack handling is weak | LOAO exposes template-orphan failures; novelty claims remain conservative. |
| A7.3/A7.5 | iid statistical intervals are too narrow | Newey-West effective n and moving-block bootstrap are implemented. |

## Current WBF vs ABF Decision

Strict run:
`current_version/results/fusion_mode_recalibrated/20260507_110115/`

| Mode | Threshold | F1 | MCC | FPR | Decision |
|---|---:|---:|---:|---:|---|
| WBF | 0.059663 | 0.7057 | 0.7087 | 4.31% | keep default |
| ABF | 0.059518 | 0.7046 | 0.7077 | 4.34% | do not switch on RedeRio |

Interpretation: ABF remains scientifically interesting for future datasets or
future method families, especially when dependence is dominant. On current
RedeRio evidence, WBF is at least as good and preserves the practical advantage
of confidence weighting.

## Open Work Before Journal Submission

The 2026-05-12 refresh closed several previously-open items. The remaining open
work and recent closures are tabulated below.

### Still open

Additional open comparability items after the 2026-05-12 baseline refresh:

- High: cross-dataset SL vs no-SL replication is still needed before claiming
  that the reconstruction-only RedeRio finding generalises.
- Medium: raw-level synthetic attack injection would be required before raw IF
  / LOF / PCA / OCSVM can be evaluated on the 13 synthetic catalog attacks.

| Priority | Item | Why it matters | Status |
|---|---|---|---|
| High | Multi-seed evaluation (5 seeds) on F1 / MCC / VUS | Wu & Keogh flaw #4 mitigation. | DEFERRED - `run_multi_seed.py` ready; about 8-10 h compute pending |
| Medium | Calendar-aware EVT (audit-grade) | Opt-in mode for datasets with stronger heteroscedasticity than RedeRio. | _CODE SHIPPED_ Phase B PATCH H2 (TASK-57); audit-grade only, default OFF. See `docs/review/calendar_evt_design.md` post-mortem. |
| High | Modern TSAD baselines under the same protocol | Needed for TKDE/VLDB-style comparison claims beyond classical baselines. | Classical raw IF / LOF / OCSVM / SGD-OCSVM / PCA / robust-z baselines shipped; `compare_sota_tsad.py --mode plan/probe` scaffold exists; reviewer-grade USAD/TranAD/AnomalyTransformer/TimesNet execution remains deferred. |
| Medium | DATA-03 RedeRio licence/provenance paragraph in paper | Required for publication hygiene. | NEEDED - operator-provided provenance info pending |
| Medium | Paper-side citations and tables from the tracker | Several code-level findings are resolved but still need paper text. | Phase C consolidation, after Phase B numbers stabilise |

### Recently closed / refreshed

| Item | Closure summary |
|---|---|
| Complete 17-leaf rerun refresh | `run_pipeline.py` completed 10/10 steps on 2026-05-12 (`2e12261d55a8f975`). Final metrics are now paper-facing: catalog F1 micro 0.8666, operator-faithful anomaly F1 micro 0.8257, realised global FPR 0.965%, canonical ACTIVE FPR 2.903%, 14/14 attacks detected. |
| Ablation protocol harmonisation | `run_ablation.py` now uses the same `catalog_outages_separate` normal-window protocol as `eval_injection` and the `full_sl` reference explicitly matches production `INTER_METHOD_FUSION="wbf"`. Full SL-ADS ablation at threshold 0.103 now reports F1-cov 0.879, F1-bin 0.917, precision 0.847, FPR 0.965%, 14/14 detected. `ablation_summary.csv` is one calibrated-threshold row per run; threshold-grid rows are isolated in `ablation_threshold_sensitivity.csv`, and figures use a readable selected subset. Legacy outages-as-normal values remain in explicit CSV columns only. |
| Same-evidence no-SL and raw-baseline comparators | `compare_no_sl_fair.py` adds leak-free same-evidence ADS-without-SL baselines and paired tests. `compare_raw_baselines_fair.py` adds raw IF / LOF / OCSVM / SGD-OCSVM / robust-z / PCA baselines on raw-valid protocols, explicitly excluding synthetic injection windows. Paper-facing tables are in `docs/review/PUBLICATION_TABLES.md` sections 8bis-8ter. |
| TASK-58 regime-FPR refresh | `evaluate_regime_fpr.py` and `regime_fpr_diagnosis.py` were rerun on the complete outputs. Verdict: `H_correlation` remains the mechanism; joint k=3 exceedances are 4.524x higher on ACTIVE than QUIET. |
| TASK-60 F1 protocol values | `eval_f1_protocol_comparison.csv` now contains final complete-run values. Paper should report both `catalog_outages_separate` and `operator_faithful_anomaly`. |
| Heavy-tailed signature-noise ablation | Cauchy + Student-t variants implemented and run; QP degrades smoothly to 0.253 (Cauchy sigma=0.20) without collapse. Evidence in `outputs/scientific_hardening/signature_noise_ablation.csv`. |
| MASE-based trust mode (Hyndman-Koehler 2006) | TASK-56. Implemented and tested; ablation `mase_legacy` added. On 30 s data Naive-1 dominates most Prophet metrics; at the fixed uniform production threshold, the MASE-trust score crosses no attack windows (0/14 detected). This is an opt-in pathology / non-drop-in ablation, not a separately recalibrated MASE detector. `uniform` reaffirmed as default. Detail: `docs/audit/trust_discount_r2_analysis.md` section 4.1. |
| `NETWORK_OUTAGE_NOV17` cold-start diagnosis | The cold-start explanation remains retracted. Complete-run metrics now replace the reconstruction-only post-mortem: NOV17 is detected 1/3 windows, DEC1617 188/339 windows, and outage-aware F1 is reported through the `operator_faithful_anomaly` protocol. |
| Documentation hygiene (Phase A LOT 1) | Tracker, current status, scientific deconstruction, REPRODUCIBILITY_CHECKLIST, and ARTIFACT_APPENDIX consolidated. Superseded hardening/renaming documents are now archived. |
| Reviewer-grade ablation artefacts (Phase A LOT 2) | Hardening ablation scripts and artifacts are preserved as diagnostic evidence under `outputs/scientific_hardening/`; paper-facing ablation numbers now come from the complete 17-leaf run and the harmonised `ablation_uniform/` outputs. |
| TASK-59 per-regime contextual discount on volumetric metrics | Exploratory diagnostic / future work only. No alpha value is shipped or claimed as a contribution. If this becomes a future contribution, alpha must be selected on train-calib with complete model coverage, locked, and evaluated once on test. Production/paper reference remains alpha=1.0. See `docs/review/regime_fpr_root_cause_analysis.md` section 6. |
| TASK-60 F1 protocol decision | RESOLVED for the current paper: report both `catalog_outages_separate` and `operator_faithful_anomaly` F1 side by side. The complete run writes the machine-readable comparison to `results/.../evaluation/eval_f1_protocol_comparison.csv`. |

## Archived Material Policy

Archived reports are preserved for traceability, not deleted. If an archived
report contains a claim that conflicts with this file, the current file wins.
If the archived report contains a useful unresolved issue, that issue must be
copied into `audit/audit_verification_tracker.md` before the report is treated
as retired.

## Documentation Cleanup - 2026-05-11

Historical files are stored under `docs/archive/`. The 2026-05-07 cleanup
preserved old module reviews and reconciliation drafts; the 2026-05-11 cleanup
moved the Phase H renaming log and the dated 2026-05-06 hardening report out of
the active corpus. The current scientific state is owned by this file, the
tracker, and the `scientific_deconstruction/` documents.
