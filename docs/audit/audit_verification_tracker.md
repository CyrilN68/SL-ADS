# Audit Verification Tracker

**Status:** 2026-05-12, complete 17-leaf rerun reviewed
**Purpose:** a single living document that lets us (or any future
auditor) verify *a posteriori* that every claim raised in the audit
process has been either (a) addressed in code, (b) addressed in the
paper, (c) explicitly deferred with rationale, or (d) rejected with
rationale. Each row is a self-contained verification target.

The structure is intentionally machine-readable: the columns
**ID / Source / Type / Status / Verification command / Verifier**
should let a future reviewer regenerate the full evidence in under
30 minutes on a clean checkout. For the current high-level verdict, start with
`docs/AUDIT_CURRENT_STATUS.md`; this tracker owns row-level evidence.

---

## Conventions

- **ID** is stable across audit cycles. Do not re-number; mark as
  RESOLVED / DEFERRED instead.
- **Source** : where the concern was raised:
  - "C-01" = ID in archived `CONSOLIDATED_AUDIT_REVIEW.md`
  - "Wu&K #1" = Wu & Keogh 2021 flaw number
  - "Phase C" = the user-driven concerns in this session
  - "Self" = identified by Claude during code reading
- **Type** : `code` | `paper` | `data` | `process`
- **Status** : `RESOLVED`, `IN_PROGRESS`, `DEFERRED`, `REJECTED`,
  `PENDING_DATA`
- **Verification command** : a single bash line (or pseudo-line)
  that produces evidence
- Legacy filenames in `Source` or `Title` are preserved when they were part of
  the original finding. Verification commands should prefer active
  `src/sl_ads/...` paths when practical.
- **Verifier** : machine timestamp + initials

---

## A. Code-level verification targets

| ID | Source | Title | Status | Verification command | Verifier |
|----|--------|-------|--------|----------------------|----------|
| TASK-01 | C-01 | Canonical 2-source WBF (Joesang Eq. 12.22) | RESOLVED | `pytest tests/test_fusion_wbf_canonical.py -v` | 2026-04-23, claude |
| TASK-02 | M-01 | Bijection b+u=1 preserved by canonical WBF | RESOLVED | `pytest tests/test_fusion_wbf_canonical.py::test_bijection_preserved` | 2026-04-23, claude |
| TASK-03 | M-10 | SBN feasibility study | RESOLVED | open `docs/review/M10_sbn_architecture_analysis.md` | 2026-04-23, claude |
| TASK-04 | Phase C | BCa bootstrap CI module | RESOLVED | `python -W ignore stats_bootstrap_ci.py` | 2026-04-25, claude |
| TASK-05 | Phase C | McNemar paired test module | RESOLVED | `python -W ignore stats_mcnemar.py` | 2026-04-25, claude |
| TASK-06 | Phase C | Evidence-vs-raw ablation | RESOLVED | `python -W ignore ablation_injection_level.py --self-test` | 2026-04-25, claude |
| TASK-07 | Phase C | Temporal-SBN ablation | RESOLVED | `python -W ignore ablation_temporal_sbn.py --self-test` | 2026-04-25, claude |
| TASK-08 | Phase C | Residual correlation matrices (12x12, 5x5, 17x17) | RESOLVED | `python -W ignore analysis_residual_correlation.py --self-test` | 2026-04-25, claude |
| TASK-09 | Phase C | trust_discount/R^2 pathology - real data confirmation | RESOLVED | open `docs/audit/trust_discount_r2_analysis.md` (RedeRio real-data analysis; production default is `WBF_WEIGHT_MODE='uniform'`) | 2026-05-01, claude |
| TASK-56 | Phase A | MASE-based trust map (Hyndman-Koehler 2006) as third `WBF_WEIGHT_MODE` option (`mase`); empirical evaluation on RedeRio 30s shows persistence dominates most Prophet metrics, and the MASE-trust ablation silences the detector (0/14 attacks detected) → confirms `uniform` as published default. | RESOLVED | `pytest tests/test_mase_weighting.py -v`; `PYTHONPATH=src python -m sl_ads.train.compute_mase_postfit --dry-run`; `ablation_summary.csv` row `MASE-Trust legacy`. | 2026-05-12, codex |
| TASK-57 | Phase B | Calendar-aware EVT thresholds (PATCH H2, A1.5 audit-grade). New module `sl_ads/calendar/regime.py` (canonical 2-bucket partition ACTIVE/QUIET, versioned signature `weekday_x_daytime_x_holiday/v1@2026-05-07`); `train_models.calibrate_thresholds_per_regime_v2`; regime dispatch in `train.compute_evidence`; sidecar A1.9 strict on `calendar_evt_signature` (drift between calibration and runtime regime function hard-raises `RuntimeError("[A1.9] ...")`). **Audit-grade opt-in**: default `CONFIG['CALENDAR_EVT_ENABLED'] = False`. The complete-run TASK-58 refresh shows the main mechanism is correlation-level (`H_correlation`), not per-metric regime EVT alone. | RESOLVED | `pytest tests/test_calendar_aware_evt.py -v`; `python -m sl_ads.ablation.evaluate_regime_fpr`; design + post-mortem: `docs/review/calendar_evt_design.md`. | 2026-05-12, codex |
| TASK-58 | Phase B Option C | Regime-FPR root-cause investigation refreshed on complete run `2e12261d55a8f975`. Verdict remains `H_correlation`: median per-metric ACTIVE/QUIET = 0.571, fused p99.9 = 1.522, joint k=3 = 4.524. Realised global FPR = 0.965%, canonical ACTIVE FPR = 2.903%. | RESOLVED | `pytest tests/test_regime_fpr_diagnosis.py -v`; `PYTHONIOENCODING=utf-8 python -m sl_ads.audit.regime_fpr_diagnosis`; `python -m sl_ads.ablation.evaluate_regime_fpr`. | 2026-05-12, codex |
| TASK-59 | Phase B Option C follow-up | Per-regime contextual discount on volumetric metrics. Existing α-sweeps are exploratory/diagnostic and cannot justify a shipped setting. The current paper does **not** adopt α=0.90 or any other tuned value; production remains α=1.0. If this becomes a future contribution, α must be selected on train-calib with complete model coverage, locked, and evaluated once on test. | DEFERRED / EXPLORATORY | `pytest tests/test_regime_fpr_alpha_sweep.py -v` verifies the diagnostic machinery. Do not rerun as a paper claim unless a train-calib selection protocol is added first. | 2026-05-11, codex |
| TASK-60 | Phase B Option C follow-up | F1 protocol decision finalized on complete run. Report both: `catalog_outages_separate` F1 micro = 0.8666 and `operator_faithful_anomaly` F1 micro = 0.8257. | RESOLVED | `python -m sl_ads.evaluate.evaluate_injection`; inspect `evaluation/eval_f1_protocol_comparison.csv`. | 2026-05-12, codex |
| TASK-10 | Phase C | Wire BCa CI into `compare_if_fair.py` | RESOLVED | `python -m sl_ads.compare.compare_if_fair` produit `bootstrap_ci_f1.csv` (BCa par système) + `bootstrap_ci_paired_sl_vs_if.csv` (ΔF1 SL−IF apparié) ; colonnes `f1_ci_low/f1_ci_high` ajoutées à `fair_if_vs_sl_summary.csv` | 2026-05-01, claude |
| TASK-11 | Phase C | Wire McNemar into `compare_if_fair.py` | RESOLVED | `python -m sl_ads.compare.compare_if_fair` produit `mcnemar_sl_vs_if.csv` (test exact, statsmodels) ; déjà câblé dans la version courante | 2026-04-29, claude |
| TASK-12 | Phase C | Multi-seed evaluation (Wu&K #4) | DEFERRED | `PYTHONPATH=src python -m sl_ads.evaluate.run_multi_seed --seeds 0,1,2,3,4` (runner exists; full publication run still pending) | - |
| TASK-13 | Self | C-02 dadza typo - audit references stale entry | RESOLVED | `grep -i dadza compute_evidence_v2.py` (no hit) | 2026-04-21, prev. session |
| TASK-14 | Self | trust_discount activation warning comment | RESOLVED | `python -W error::RuntimeWarning -c "from sl_ads.config import CONFIG; CONFIG['WBF_WEIGHT_MODE']='trust_discount'; import sl_ads.core.opinions_pipeline"` lève RuntimeWarning citant §5.3.3 | 2026-05-01, claude |
| TASK-15 | Self | Brazilian Prophet holidays list populated | RESOLVED | `pytest tests/test_holidays_brazil.py -v` (5 tests, vérifient via paquet ``holidays.Brazil()`` que tous les fériés nationaux dans la fenêtre 2025-10-13 → 2026-01-01 sont couverts ; design ``University_Closed`` parcimonieux documenté en `config.py` §HOLIDAYS_LIST) | 2026-05-01, claude |
| TASK-16 | Phase C | Kitsune baseline integration | DEFERRED | `python compare_kitsune.py` (not yet impl.) | - |
| TASK-17 | Phase C | VUS-PR / VUS-ROC metrics | RESOLVED | Superseded by TASK-54: `python -m sl_ads.evaluate.vus_metrics --self-test` + `pytest tests/test_vus_metrics.py -v` | 2026-05-01, claude |
| TASK-18 | Phase C | Axelsson 2000 per-attack PPV table | RESOLVED | `python -m sl_ads.evaluate.axelsson_ppv --self-test` (10 self-tests + 29 unit tests dans `tests/test_axelsson_ppv.py` ; reproduit le cas Axelsson 2000 §6 PPV=0.4118 ; round-trip Bayes inverse vérifié) | 2026-05-01, claude |
| TASK-19 | Phase D | Full pipeline re-execution + reconciliation 2026-04-25 | RESOLVED | open `docs/archive/2026-05-07_audit_cleanup/audit/pipeline_reconciliation_20260425.md` | 2026-04-25, claude |
| TASK-20 | _audit_tmp CRIT-02 | `compute_opinions_v3.py` raise FileNotFoundError when injection active but `_attacks` missing | RESOLVED | `pytest tests/test_audit_remediation_20260426.py -v -k task20` | 2026-04-26, claude |
| TASK-21 | _audit_tmp CRIT-03 | Rename `f1_binary`/`f1_coverage` -> `*_hybrid_episode_recall` + deprecation | RESOLVED | `pytest tests/test_audit_remediation_20260426.py -v -k task21` | 2026-04-26, claude |
| TASK-22 | _audit_tmp MAJ-01 | Recalibrate t_susp/t_atk on independent calibration split in `train_v10.py` (legacy name; active module `src/sl_ads/train/train_models.py`) | RESOLVED | inspect `src/sl_ads/train/train_models.py` for `len(_calib_clean_pre) >= 30` branch using `y_cp - reg.predict(X_cp)` | 2026-04-26, claude |
| TASK-23 | _audit_tmp MAJ-02 | Externalize magic numbers `0.5`, `1e-9`, `_bijection_floor` in active training config | RESOLVED | `Select-String -Path src/sl_ads/config.py,src/sl_ads/train/train_models.py -Pattern 'CALIB_BIJECTION_FLOOR_TOL|CALIB_AGEING_WIN_FRACTION|CALIB_SPARSITY_CUTOFF'` | 2026-04-26, claude |
| TASK-24 | _audit_tmp MAJ-03 | `compare_qualif_methods.py` use `paths.get_decision_threshold()` | RESOLVED | `pytest tests/test_audit_remediation_20260426.py -v -k task24` | 2026-04-26, claude |
| TASK-25 | _audit_tmp MAJ-04 | Remove hardcoded fallback path in `evaluate_qualify_sbn.py:949-970` | RESOLVED | `pytest tests/test_audit_remediation_20260426.py -v -k task25` | 2026-04-26, claude |
| TASK-26 | _audit_tmp MAJ-05 | Rename `compute_conflict_degree` -> `compute_asymmetric_escalation_conflict` + add canonical | RESOLVED | `pytest tests/test_audit_remediation_20260426.py -v -k task26` | 2026-04-26, claude |
| TASK-27 | _audit_tmp MAJ-06 | `fusion_cbf` symmetric handling of degenerate case | RESOLVED | `pytest tests/test_audit_remediation_20260426.py -v -k task27` | 2026-04-26, claude |
| TASK-28 | _audit_tmp MAJ-07 | Targeted warnings filters instead of global `filterwarnings("ignore")` | RESOLVED | `pytest tests/test_audit_remediation_20260426.py -v -k task28` | 2026-04-26, claude |
| TASK-29 | _audit_tmp MAJ-08 | METR-LA sensor variance computed only on training period | DEFERRED | `dataset_adapter/metr_la_adapter.py:90-104` ; relevant only if METR-LA in main results | - |
| TASK-30 | _audit_tmp MAJ-09 | RedeRio pseudo-label vote threshold sweep `{3,4,5,6,7}` | DEFERRED | `dataset_adapter/rederio_adapter.py:116-123` ; pseudo-labels are descriptive only | - |
| TASK-31 | _audit_tmp MAJ-10 | `apply_pseudo_labels()` abstract or true multi-metric in adapter_base | DEFERRED | `dataset_adapter/adapter_base.py:34-52` ; affects cross-domain only | - |
| TASK-32 | _audit_tmp MAJ-11 | Deterministic run_id from hash(config + git + inputs) | RESOLVED | `pytest tests/test_audit_remediation_20260426.py -v -k task32` | 2026-04-26, claude |
| TASK-33 | _audit_tmp MIN-01..04 | 4 minor fixes (notebooks path, benchmark constants, pearson periods, audit message) | RESOLVED | `pytest tests/test_audit_remediation_20260426.py -v -k task33` | 2026-04-26, claude |
| TASK-34 | audit_codex CRIT-01 | `_select_best_row()` selects threshold from sidecar (no argmax leakage); `SL_ALLOW_TEST_TUNED_THRESHOLD=1` escape hatch with warning | RESOLVED | `pytest tests/test_audit_codex_remediation_20260427.py -v -k task34` | 2026-04-27, claude |
| TASK-35 | audit_codex CRIT-03 | IF FPR-matched headline contamination is `IF_CONTAMINATION_DEFAULT` (a-priori); ladder `IF_CONTAMINATION_LADDER` for sensitivity only; `SL_ALLOW_TEST_TUNED_IF` escape hatch | RESOLVED | `pytest tests/test_audit_codex_remediation_20260427.py -v -k task35` | 2026-04-27, claude |
| TASK-36 | audit_codex MAJ-01 | NaN preserved on metrics in `rederio_adapter.py:62` and `cesnet_adapter.py:69`; downstream `preprocess_metrics()` does bounded ffill | RESOLVED | `pytest tests/test_audit_codex_remediation_20260427.py -v -k task36` | 2026-04-27, claude |
| TASK-37 | audit_codex MAJ-02 | `preprocessing_utils.preprocess_metrics()` accepts explicit `metric_cols`; default uses `NON_METRIC_COLUMNS` whitelist (label/flag/mask never ffilled); `strict=True` mode raises on schema drift | RESOLVED | `pytest tests/test_audit_codex_remediation_20260427.py -v -k task37` | 2026-04-27, claude |
| TASK-38 | audit_codex MAJ-03 | `CONFIG['SBN_NOVELTY_U_RAW_THRESHOLD'] = 0.82` declared as base default in `config.py` (was only set inside env-var override branch) | RESOLVED | `pytest tests/test_audit_codex_remediation_20260427.py -v -k task38` | 2026-04-27, claude |
| TASK-39 | audit_codex MAJ-08 | `CALIB_BIJECTION_FLOOR_TOL=0.01`, `CALIB_AGEING_WIN_FRACTION=0.5`, `CALIB_SPARSITY_CUTOFF=1e-9` declared in `config.py` (were only consumed via `CONFIG.get(..., default)`) | RESOLVED | `pytest tests/test_audit_codex_remediation_20260427.py -v -k task39` | 2026-04-27, claude |
| TASK-40 | audit_codex MAJ-04 | `STL_FAIL_POLICY='raise'` (was silent zero votes); `'abstain'` opt-in returns NaN votes; `REDERIO_METRIC_VOTE_THRESHOLD` moved to `config.py`; `np.nansum` consensus aggregation | RESOLVED | `pytest tests/test_audit_codex_remediation_20260427.py -v -k task40` | 2026-04-27, claude |
| TASK-41 | audit_codex MAJ-06 | METR-LA top-5 sensor variance ranking computed on TRAIN slice only (`timestamp < SELECTED_SPLIT`); raises if train slice empty; warns if split unknown | RESOLVED | `pytest tests/test_audit_codex_remediation_20260427.py -v -k task41` | 2026-04-27, claude |
| TASK-42 | audit_codex MAJ-07 | GECCO adapter uses `GECCO_LOAD_MODE='concat'` (default, no silent file loss) or `'single'` (assert exactly one CSV); files sorted lexicographically for determinism | RESOLVED | `pytest tests/test_audit_codex_remediation_20260427.py -v -k task42` | 2026-04-27, claude |
| TASK-43 | audit_codex MIN-01 | `paths.py` docstrings reference `train_v10.py` (no longer mention retired `train_v9.py`) | RESOLVED | `pytest tests/test_audit_codex_remediation_20260427.py -v -k task43` | 2026-04-27, claude |
| TASK-44 | audit_codex MAJ-09 | `compute_opinions_v3.py` writes `fusion_mode_at_compute_opinions.json` (actual_fusion_mode, wbf_weight_mode, balance_ratio, …); `paths.get_fusion_mode_for_run(OUTPUT_DIR)` reads it; alias `get_detection_col_fused()` added | RESOLVED | `pytest tests/test_audit_codex_remediation_20260427.py -v -k task44` | 2026-04-27, claude |
| TASK-45 | audit_codex CRIT-02 | Threshold sidecar persists sensitive calibration config; runtime sidecar/config mismatch now hard-fails. Full deployed-chain calibration is available for WBF/ABF strict ablation via TASK-55 sidecars, while generic production sidecar caveat remains documented. | RESOLVED | `pytest tests/test_config_and_sidecar.py -v`; see TASK-55 for mode-specific fusion sidecars | 2026-05-07, codex |
| TASK-46 | audit_codex MAJ-05 | CESNET adapter exposes `CESNET_TIMESTAMP_MODE` ∈ {fabricated_warning, fabricated_silent, reject} with `CESNET_TIMESTAMP_ANCHOR`; warning emitted at every load by default | RESOLVED | `pytest tests/test_audit_codex_remediation_20260427.py -v -k task46` | 2026-04-27, claude |
| TASK-47 | Phase H+ | "Uniform-as-reference" — `WBF_WEIGHT_MODE="uniform"` propagé partout; `full_sl` ablation matche désormais la production WBF. `trust_discount_legacy` expose la pathologie sur le run complet: Full `F1-cov=0.879` vs legacy R² trust-discount `F1-cov=0.628`, `FPR=4.39%`, `12/14` attaques. PUBLICATION_TABLES.md + honest_limitations.md §5.3.3 mis à jour. | RESOLVED | `grep -nE "wbf_weight_mode.*\"trust_discount\"" src/sl_ads/config.py` (1 hit attendu = trust_discount_legacy) | 2026-05-12, codex |
| TASK-48 | Phase H+ | Disclosures honest_limitations.md §5.3.7 (EVT calibration limits + R² négatif), §5.3.8 (qualifier confusion matrix BOTNET/DNS_TUN/DNS_AMP), §5.3.9 (compare_if vs eval_injection methodology gap), §5.3.10 (Slowloris persistent gap), §5.3.11 (NETWORK_OUTAGE_NOV17 calibration-boundary case), §5.3.12 (AUC novelty_lr in-sample reporting-only) | RESOLVED | open `docs/honest_limitations.md` and search "5.3.7", "5.3.8", "5.3.9", "5.3.10", "5.3.11", "5.3.12" | 2026-04-29, claude |
| TASK-49 | Phase H+ | Test ablation production — `tests/test_ablation_summary_schema.py` vérifie que `run_ablation.py` produit `ablation_summary.csv` avec colonnes attendues (run/best_threshold/f1_binary/f1_coverage/precision/recall_binary/fpr_pct/n_detected/n_attacks) | RESOLVED | `pytest tests/test_ablation_summary_schema.py -v` (5 passed) | 2026-05-01, claude |
| TASK-50 | Phase H+ | Test EVT-fallback — `tests/test_evt_fallback.py` couvre les 4 chemins de fallback (Grimshaw fail → scipy → empirical, σ̃≤0 short-circuit, evt_short_data) | RESOLVED | `pytest tests/test_evt_fallback.py -v` (11 passed) | 2026-05-01, claude |
| TASK-51 | Phase H+ | Test LR_NOVELTY_THR=None regression guard — `tests/test_config_invariants.py::test_lr_novelty_thr_none` vérifie que la config publiée a toujours `LR_NOVELTY_THR=None` (PATCH-C2) | RESOLVED | `pytest tests/test_config_invariants.py -v` (10 passed) | 2026-05-01, claude |
| TASK-52 | Phase H+ | Test cross-run consistency — `tests/test_metrics_consistency.py` (slow marker) now targets complete run `2e12261d55a8f975`: F1_micro≈0.8666, MCC≈0.8587, 14/14 attacks detected, FAR≈0.97%. | RESOLVED | `pytest tests/test_metrics_consistency.py -v -m slow` when completed outputs are present. | 2026-05-12, codex |
| TASK-53 | Phase H+ | Multi-seed evaluation infrastructure — `RANDOM_SEED` configurable + `run_multi_seed.py` runner pour k=5 runs avec graines distinctes (Wu&K #4) | RESOLVED | `python -m sl_ads.evaluate.run_multi_seed --self-test` + `pytest tests/test_run_multi_seed.py -v` (34 tests : parsing, agrégation, BCa CI sur seeds, latest-run-id resolution, dry-run orchestration) ; orchestration via SL_RANDOM_SEED env var déjà câblée dans `config.py` §"PATCH TASK-53" | 2026-05-01, claude |
| TASK-54 | Phase H+ | VUS-PR / VUS-ROC infrastructure (Paparrizos 2022) — `src/sl_ads/evaluate/vus_metrics.py` avec implémentation range-based + tests unitaires sur cas connus | RESOLVED | `python -m sl_ads.evaluate.vus_metrics --self-test` (10 self-tests) + `pytest tests/test_vus_metrics.py -v` (29 tests : extension de runs, AUC sur cas perfect/inverted/random, VUS bornes [0,1], existence-recall Tatbul 2018) | 2026-05-01, claude |
| TASK-55 | Phase I | Inter-method fusion operators + strict WBF/ABF recalibration. Implements WBF/ABF/CBF/BCF/projected-CCF/MinBF/MaxBF/hierarchical dispatch, method-group config for future third methods, mode-specific threshold sidecars, and strict sidecar-backed WBF vs ABF ablation. ABF was **not** adopted as default on RedeRio because WBF was slightly better after per-mode recalibration (`F1=0.7057`, `MCC=0.7087`, `FPR=4.31 %` vs ABF `F1=0.7046`, `MCC=0.7077`, `FPR=4.34 %`). | RESOLVED | `PYTHONPATH=src python -m sl_ads.ablation.compare_recalibrated_fusion_modes --modes wbf,abf`; open `results/fusion_mode_recalibrated/20260507_110115/fusion_mode_recalibrated_decision.json` and confirm `keep_default_wbf` | 2026-05-07, codex |

## B. Paper-level verification targets

| ID | Source | Title | Status | Where in paper | Note |
|----|--------|-------|--------|-----------------|------|
| L-01 | C-04 | Replace "exact Joesang Theorem 12.2" claim with "evidence-space WBF (Joesang 2016 Sec. 12.5.4)" | RESOLVED in code; PAPER PENDING | Section 3.5 | Wording in paper must match docstring of `fusion_wbf_n_sources` |
| L-02 | C-05 | Rename "SBN" -> "Expert-template-driven SL qualifier" everywhere | RESOLVED in `docs/review/M10_sbn_architecture_analysis.md` (decision recorded); PAPER PENDING | Sections 3.6, 3.7 | Search for "SBN" in `paper/main.tex` and replace |
| L-03 | Phase C | Add "Honest limitations" section | RESOLVED in `docs/honest_limitations.md`; PAPER PENDING | Section 5.3 | Drop the doc into the paper template |
| L-04 | Phase C | Cite Wu & Keogh 2021 explicitly + self-assessment | RESOLVED in `docs/audit/wu_keogh_self_assessment.md`; PAPER PENDING | Section 4.0 (eval methodology) | Paper text must include the four-flaw checklist |
| L-05 | Phase C | Cite Paparrizos 2022 (VUS-PR/ROC) | NOT YET CITED | Section 4.x | Add to bibliography + Discussion |
| L-06 | Phase C | Cite Baldan 2025 multimodal benchmark | NOT YET CITED | Section 6 (related work) | Single sentence + future-work pointer |
| L-07 | Phase C | Add Axelsson 2000 base-rate fallacy discussion | NOT YET DONE | Section 4.4 | Per-attack PPV table required |
| L-08 | Phase C | Trust-discount/R^2 pathology disclosure | RESOLVED in `docs/audit/trust_discount_r2_analysis.md`; PAPER PENDING | Appendix C | Full ablation + decision rationale |
| L-09 | Phase C | Bootstrap CI on every reported metric | NOT YET INTEGRATED INTO PAPER | All result tables | Use `stats_bootstrap_ci.py.format_ci()` for consistency |
| L-10 | Phase C | McNemar p-value on every paired comparison | NOT YET INTEGRATED INTO PAPER | Comparison tables | Use `stats_mcnemar.py.format_mcnemar()` |
| L-11 | Phase C | Hutchins 2011 Kill-Chain disclosure (temporal SBN limits) | RESOLVED in `ablation_temporal_sbn.py` doctring; PAPER PENDING | Section 3.7 | One paragraph + ablation table |
| L-12 | Phase C | Joesang Def. 14.6 trust transitivity citation | RESOLVED in `sl_formulas_v2.py:833`; PAPER PENDING | Section 3.5 | Verify exact definition number |
| L-13 | Phase C | Reviewer-target calibration disclosure | INTERNAL only | (not in paper) | Strategic venue-risk note archived at `docs/archive/2026-05-07_audit_cleanup/audit/reviewer_target_calibration.md`; actionable open work has been consolidated into `docs/AUDIT_CURRENT_STATUS.md` "Open Work Before Journal Submission". |
| L-14 | Phase H+ | "Uniform-as-reference" — Section 3.5 fusion baseline = uniform; trust-discount = pathology section dédiée §5.3.3 | RESOLVED in `docs/honest_limitations.md` §5.3.3 + `docs/review/PUBLICATION_TABLES.md`; PAPER PENDING (insertion LaTeX) | Section 3.5 + 5.3.3 | Paper tables must use complete-run ablation values: Full `F1-cov=0.879`; legacy R² trust-discount `F1-cov=0.628`; MASE-trust `0/14` attacks. |
| L-15 | Phase H+ | EVT calibration limits — section dédiée expliquant lien R² négatif → EVT instable → fallback empirique | RESOLVED in `docs/honest_limitations.md` §5.3.7 ; PAPER PENDING | Section 3.4 + 5.3.7 | Table FPR_susp_emp / FPR_atk_emp à intégrer en figure |
| L-16 | Phase H+ | Qualifier confusion matrix — discussion BOTNET_CC↔PORT_SCAN, DNS_TUNNELING multi-classe, DNS_AMP↔NTP_AMP collision | RESOLVED in `docs/honest_limitations.md` §5.3.8 ; PAPER PENDING | Section 4.x (qualification) | Matrice 13×13 à figurer en annexe |
| L-17 | Phase H+ | compare_if vs eval_injection methodological clarification | RESOLVED in `docs/honest_limitations.md` §5.3.9 ; PAPER PENDING | Section 4.0 (methodology) | Disclaimer obligatoire sur la différence catalog vs pseudo-labels |
| L-18 | Phase H+ | Slowloris persistent gap (65.6% recall, 90 min TTD) | RESOLVED in `docs/honest_limitations.md` §5.3.10 ; PAPER PENDING | Section 5.3 (limitations) | Single paragraph + note opérationnelle |
| L-19 | Phase H+ | Network outages as operator-relevant anomalies | RESOLVED in `docs/honest_limitations.md` §5.3.11 ; PAPER PENDING | Section 5.3 (limitations) | NOV17 = 1/3, DEC1617 = 188/339, and operator-faithful F1 = 0.8257. |
| L-20 | Phase H+ | AUC novelty_lr in-sample = 0.654 reporting-only | RESOLVED in `docs/honest_limitations.md` §5.3.12 ; PAPER PENDING | Section 4.5 + 5.3.12 | Cite Hanley & McNeil 1982, Youden 1950 ; PATCH-C2 conforme |

## C. Process-level verification targets

| ID | Source | Title | Status | Note |
|----|--------|-------|--------|------|
| PROC-01 | Phase C | All scripts have a `--self-test` mode | RESOLVED for the 4 new scripts | Maintainers must extend this discipline to any new artifact |
| PROC-02 | Phase C | All claims numbered with stable IDs | RESOLVED | This document is the canonical mapping |
| PROC-03 | Phase C | Every statistical comparison reports BCa CI + McNemar | IN_PROGRESS | TASK-10, TASK-11 must close before submission |
| PROC-04 | Phase C | Reproducibility checklist (NeurIPS-style) | RESOLVED | See `docs/REPRODUCIBILITY_CHECKLIST.md`; keep it updated with every artifact change |
| PROC-05 | Phase C | Ablation results stored as JSON + CSV next to plots | RESOLVED | All four ablation scripts emit both |

## D. Data verification targets

| ID | Source | Title | Status | Note |
|----|--------|-------|--------|------|
| DATA-01 | Self | Train/test temporal split documented | RESOLVED | `src/sl_ads/train/train_models.py` + paper Section 3.2 |
| DATA-02 | Phase C | Synthetic injection catalog reproducibility | RESOLVED | `config.py:946-989` with `RANDOM_SEED` |
| DATA-03 | Phase C | RedeRio data licence + provenance disclosure | NEEDED | Paper Section 3.1 |
| DATA-04 | Phase C | Holidays list populated for Prophet | RESOLVED | See TASK-15; Brazilian national holidays in the 2025-10-13 to 2026-01-01 window are covered |

---

## Quick-verify sequence (clean checkout)

The following commands, run in order on a clean checkout, regenerate
all RESOLVED artefacts. Total runtime ~ 2 minutes on a developer
workstation.

```bash
# Phase H package layout — full test suite (recommended one-liner)
pytest tests/ -W ignore::DeprecationWarning

# Phase H+ TASK-53 / 54 / 18 / 15 / 49 / 50 / 51 / 52 module self-tests
PYTHONPATH=src python -m sl_ads.evaluate.vus_metrics       --self-test
PYTHONPATH=src python -m sl_ads.evaluate.run_multi_seed    --self-test
PYTHONPATH=src python -m sl_ads.evaluate.axelsson_ppv      --self-test
PYTHONPATH=src python -m sl_ads.stats.bootstrap_ci         # implicit self-test
PYTHONPATH=src python -m sl_ads.stats.mcnemar              # implicit self-test
```

Expected outcome (clean checkout, Phase H+):

* ``pytest`` should pass on a clean checkout. The exact pass count evolves as
  new audit guards are added; this tracker, ``ARTIFACT_APPENDIX.md`` and
  ``REPRODUCIBILITY_CHECKLIST.md`` should be re-aligned whenever tests are
  added or retired.
* Every module self-test prints `[OK] ... ALL PASS`.

---

## Failure mode

If any line above fails on a future audit, mark the corresponding TASK
as `IN_PROGRESS` with a new row added below documenting the regression.
The history of regressions is itself part of the audit trail.

| Date | TASK | Failure mode | Fix commit | Re-verifier |
|------|------|--------------|------------|-------------|
| -    | -    | (no regressions yet) | - | - |
| 2026-04-25 | TASK-04..08 | (no failure) - re-verified after full pipeline re-run; all 5 ablation/stat modules pass; pipeline produces values inside published BCa CIs; deltas all attributable to documented commit 9993c24. See archived reconciliation `docs/archive/2026-05-07_audit_cleanup/audit/pipeline_reconciliation_20260425.md`. | 9993c24 (already in main) | claude |
| 2026-04-26 | TASK-20..28, TASK-32, TASK-33 | (no failure) - bulk audit remediation closing CRIT-02, CRIT-03, MAJ-01..07, MAJ-11, MIN-01..04. All 15 unit tests in `tests/test_audit_remediation_20260426.py` PASS; legacy `evaluate_injection_v2.py` end-to-end smoke run produced 14/14 attacks detected with canonical (CRIT-03 compliant) F1_micro=0.781, F1_macro=0.884; MANIFEST.md updated with deterministic run_id. Pre-existing tests `test_fusion_wbf_canonical.py` (8/8) and `test_resolve_sl_csv_path.py` (4/4) still PASS. See archived reconciliation `docs/archive/2026-05-07_audit_cleanup/audit/scientific_audit_reconciliation_20260425.md` for per-finding traceability. | (uncommitted local fixes) | claude |
| 2026-04-27 | TASK-34..46 | (no failure) — audit_codex_2026-04-26 closeout (Phase G). 13 actionable findings (3 CRIT, 9 MAJ, 1 MIN; CRIT-02 partial then closed by TASK-55). All 17 unit tests in `tests/test_audit_codex_remediation_20260427.py` PASS. Combined regression suite (Phase F + Phase G + fusion + path tests) = **44 passed** in 3.13s. New artifacts: `fusion_mode_at_compute_opinions.json`, `CESNET_TIMESTAMP_MODE`, `IF_CONTAMINATION_LADDER`, `STL_FAIL_POLICY`, `GECCO_LOAD_MODE`. See archived reconciliation and `docs/AUDIT_CURRENT_STATUS.md` for current synthesis. | (uncommitted local fixes) | claude |
| 2026-04-29 | (Phase H reorganisation) | (no regression) — full codebase reorganisation into ``src/sl_ads/`` package layout. Historical reorganisation notes are archived under ``docs/archive/2026-05-11_public_release_cleanup/top_level/RENAMING_LOG_PHASE_H.md``. | (uncommitted; Phase 7 removed the shims) | claude |
| 2026-04-29 | (Phase H closeout — Phase 7) | (no regression) — legacy shims and artefacts deleted; final public layout is package-first (`src`, `tests`, `docs`, `investigations`, `outputs`, `results`, `references`). Historical detail is archived under ``docs/archive/2026-05-11_public_release_cleanup/top_level/RENAMING_LOG_PHASE_H.md``. | (uncommitted) | claude |
