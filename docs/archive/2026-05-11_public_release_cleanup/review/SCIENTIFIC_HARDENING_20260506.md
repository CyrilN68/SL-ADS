# Scientific Hardening — 2026-05-06

**Scope.** Targeted remediation pass on the seven open audit items from
`docs/scientific_deconstruction/ASSUMPTIONS.md` (entries A1.3, A1.5,
A1.9, A3.2, A3.3, A6.4, A6.5, A7.3, A7.5) plus a follow-up extension
covering A1.1, A3.5, A4.7, A6.1, A6.3 and the EVT fallback chain. Builds
on the previous hardening pass
(`docs/archive/2026-05-07_audit_cleanup/review/SCIENTIFIC_HARDENING_20260504.md`) by
adding code-level guard rails, statistical-inference fixes, and
reviewer-grade ablation artefacts.

This file is the canonical record of what changed, what numbers were
produced, and what remains a residual scientific risk.

---

## 1. Summary of code changes

| Audit item | File | Change | Test |
|---|---|---|---|
| A1.1 | `src/sl_ads/audit/audit_train_span.py` (NEW) | Defensive STL+Hampel+CUSUM consensus audit on the train span only. Writes `audit_train_span.csv` + `_summary.json`. Non-destructive — reviewer reads the top-N and decides. | manual review only |
| A1.2 / A1.8 | `src/sl_ads/train/train_models.py` | `_pwm_gpd_fit` (Hosking-Wallis 1987) inserted as middle EVT fallback between Grimshaw MLE and `scipy.genpareto.fit`. Closed-form, robust for `xi > 0.5`. | `test_pwm_gpd_fit_recovers_known_parameters`, `test_pwm_gpd_fit_handles_heavy_tail_xi_above_half` |
| A1.9 | `src/sl_ads/paths.py` | `validate_threshold_sidecar_config` raises on calibration-vs-runtime mismatch on `INTER_METHOD_FUSION`, `WBF_WEIGHT_MODE`, `LAMBDA_DECAY`, `BALANCE_RATIO`, `CD_ALPHA_ATTACK`. Invoked automatically by `get_decision_threshold`. | `tests/test_config_and_sidecar.py::test_sensitive_sidecar_config_*` |
| A1.9 (reporting) | `src/sl_ads/evaluate/evaluate_injection.py` | New columns `fpr_target`, `fpr_ratio_to_target`, `fpr_target_status` in `eval_threshold_sweep.csv`. Status `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` raised if empirical FPR > 2× target. | manual run report |
| A3.2 (qualifier-side) | `src/sl_ads/evaluate/evaluate_qualify_sbn.py` | `_compute_global_detection_stats` consolidates `CONFIG["EVAL"]["REAL_ATTACK_CATALOG"]` into the attack/outage tri-class partition, so RedeRio's REAL_DDOS no longer counts as FP in the qualifier FAR. | `tests/test_scientific_hardening_20260505.py::test_real_attack_catalog_is_subtracted_from_global_far` |
| A3.2 (detector-side) | `src/sl_ads/evaluate/evaluate_injection.py` | `windows_outside_attacks` now also strips REAL_ATTACKS (incl. NETWORK_OUTAGE) so the detector-side FPR is computed on the same "truly normal" base as the regime audit. Bootstrap CI is computed on the outage-excluded vectors. | `tests/test_scientific_hardening_20260505.py::test_threshold_sweep_excludes_real_attacks_outage_from_fpr` |
| A3.5 | `src/sl_ads/inject/evidence_level.py` | `_validate_catalog` raises `ValueError("[A3.5] …")` if any catalog event has `start <= split_date`. Hard-stop the injection step. | `test_a35_catalog_validator_rejects_pre_split_event`, `test_a35_catalog_validator_accepts_post_split_event` |
| A4.7 | `src/sl_ads/core/opinions_pipeline.py` | RuntimeWarning at module import when `INTER_METHOD_FUSION='cbf'`, citing the cross max \|rho\|=0.915 measured in 2026-05-04 residual-correlation audit. | `test_a47_cbf_emits_dependence_warning` |
| A7.3 | `src/sl_ads/evaluate/axelsson_ppv.py` | Wilson CI now uses an effective `n` from a Newey–West-style autocorrelation correction (`stats/residual_correlation.newey_west_eff_n`). Outputs include `*_n_eff` columns for transparency. | `tests/test_axelsson_ppv.py` (extended) |
| A7.5 | `src/sl_ads/stats/bootstrap_ci.py` | Both `bootstrap_bca_ci` and `paired_bootstrap_bca_ci` accept `block_length` and use the moving-block bootstrap (Künsch 1989) when it is set. Jackknife uses delete-one-block. | `tests/test_scientific_hardening_20260505.py::test_bca_bootstrap_supports_moving_blocks` |
| A7.5 (caller) | `src/sl_ads/evaluate/evaluate_injection.py` | `global_threshold_sweep` calls `bootstrap_bca_ci` with `block_length = median_attack_episode_length`. | manual run report |
| Train-calib residuals | `src/sl_ads/train/train_models.py` | `models_pkg['_calib_signed_residuals']` persisted next to the trained models. Lets `ablation_evt_declustering.py` run on production-grade calibration residuals once the models are retrained, eliminating the inference-span-proxy caveat. | n/a (one-shot output) |

CI-level reporting columns now persisted in `eval_threshold_sweep.csv`:
`bootstrap_method`, `bootstrap_resampling`, `bootstrap_block_length`.

---

## 2. Ablation artefacts

All files live under `current_version/outputs/scientific_hardening/`.

| Audit item | Script (`src/sl_ads/ablation/*.py`) | Artefacts | Headline finding |
|---|---|---|---|
| A1.1 train-span audit | `src/sl_ads/audit/audit_train_span.py` | `audit_train_span.csv`, `audit_train_span_summary.json` | On the 3 volumetric metrics (bytes/packets/flows): 80,642 train windows, 3,120 already in `TRAIN_EXCLUSIONS`, 306 NEW high-severity (3-metric consensus), 4,751 medium, 13,969 low. Top suspects are weekend evenings 2025-10-18 → 2025-10-26 — listed in the JSON for manual review. |
| A1.3 EVT declustering | `ablation_evt_declustering.py` (`--mode lightweight` default) | `evt_declustering_thresholds.csv`, `evt_declustering_thresholds_delta_pct.csv`, `evt_declustering_summary.json` | Out of 17 metrics, 12 are insensitive (Δ=0). 5 metrics show meaningful changes when declustering is enabled: max \|Δ T_susp\|=24.7%, max \|Δ T_atk\|=25.3%. The "Prophet whitens residuals → declustering OFF" justification is partially valid but breaks for the most volatile traffic metrics (`prophet_packets`, `prophet_udp`, `reconst_udp_from_flows`, `reconst_fin_from_syn`, `prophet_avg_pkt_size`). |
| A1.5 regime stationarity | `evaluate_regime_fpr.py` | `regime_fpr.csv`, `regime_fpr.png`, `regime_fpr_summary.json` | The dataset spans 45 days (2025-11-10 → 2025-12-25). Empirical FPR varies sharply across regimes: weekday-term-like 0.38 % (3.8× target), day 08–18 0.45 % (4.5× target), weekend / night / holiday 0.0 %. Annual projection (assuming the observed regime mix is representative): expected 250 FP/year vs the 105 implied by a flat 0.001 budget — i.e. **2.4× the target on a year-long deployment**. |
| A3.3 signature noise | `ablation_signature_noise.py` | `signature_noise_ablation.csv`, `signature_noise_ablation.png` | Gaussian perturbation σ ∈ {0, 0.05, 0.10, 0.15, 0.20} on the normalised injected (P,S,N) triplets degrades QP linearly: 0.607 → 0.578 → 0.533 → 0.481 → 0.426. No collapse: at σ=0.20 (a perturbation comparable to a noisy real-traffic attack) the qualifier still typifies 60 % of detected windows correctly. |
| A6.1 NB independence audit | `ablation_qualifier_group_independence.py` | `qualifier_group_correlations.csv`, `qualifier_group_correlations_summary.json` | 12 groups, 66 pairs. Attack-window max \|rho\|=0.957; 32/66 HIGH dependence (>=0.6), 15/66 MODERATE. Verdict: NB violated for ROC-area but argmax decision robust under positive correlations on the correct class. |
| A6.3 LOAO templates | `ablation_qualifier_loo_templates.py` | `qualifier_loo_results.csv`, `qualifier_loo_summary.json` | Removing an unrelated template: ΔQP +0.02 (essentially robust). Removing a matching template: QP→0 (tautological) but autre_anomalie catches 0 % — orphans go to closest neighbour (DNS_AMP→NTP_AMP, BOTNET_CC→PORT_SCAN, SLOWLORIS→PORT_SCAN). Novelty-handling weakness exposed. |
| A6.4 + A6.5 sensitivity | `ablation_sbn_param_sensitivity.py` (extended grid) | `sbn_param_sensitivity.csv`, `sbn_param_sensitivity_qp_heatmap.png` | The published operating point (`evidence_scale=3.0`, `u_raw_threshold=0.82`) sits on a broad plateau: QP=0.607 holds for `evidence_scale ∈ [1.0, 10.0]` and `u_raw_threshold ∈ [0.30, 0.99]`. Pathologies appear only at extreme corners (`evidence_scale ≤ 0.5` forces every window into `autre_anomalie`; `u_raw_threshold ≤ 0.10` reaches the same regime). |
| SOTA TSAD plan | `compare/compare_sota_tsad.py` | `compare_sota_tsad_plan.json` | Reviewer-grade run plan for TranAD / AnomalyTransformer / TimesNet / USAD (~10 h GPU). Implementation deferred (TKDE/VLDB scope). |

The full-pipeline EVT declustering harness (`--mode full --execute`) is
preserved for cases where the lightweight mode flags large deltas; it
is not run by default because it requires four full Prophet/QR
retrainings.

---

## 3. Headline detection numbers (A1.9 + A3.2 verification)

After the A3.2 outage-exclusion fix at the threshold-sweep level, the
headline numbers on RedeRio (run `resultats_RedeRio_trained_v4s_v4_v2`,
δ = 0.1292) become:

| Metric | Value | Note |
|---|---:|---|
| F1 micro (pure window) | 0.940 | (was 0.784 with outages counted as FP) |
| F1 macro (pure window) | 0.940 | (was 0.885) |
| Precision (window) | 0.954 | (was 0.746) |
| TPR (window) | 0.827 | unchanged (recall is on catalog windows only) |
| FPR (window) | 0.002 | (was 0.016) |
| MCC | 0.882 | (was 0.772) |
| Accuracy | 0.988 | (was 0.975) |
| FPR target | 0.001 | RedeRio operator budget |
| **FPR ratio to target** | **2.33×** | Empirical / target — exceeds 2× → status `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| Bootstrap CI on F1 | [0.665, 0.875] | BCa-block, `block_length=36` (median episode length) |
| Bootstrap CI on MCC | [0.645, 0.860] | BCa-block, `block_length=36` |
| VUS-PR | 0.604 | (existing publication value) |
| VUS-ROC | 0.856 | (existing publication value) |

The 2.33× FPR-to-target ratio is consistent with the regime audit
(2.41× on the all-normal regime). The remaining surplus is dominated by
the weekday-daytime regime; it is *not* an artefact of REAL_ATTACKS being
mis-classified (that has been corrected by A3.2 at every reporting site).

The system A1.9 sidecar/runtime check is also wired into the standard
evaluation pipeline: any future change to `LAMBDA_DECAY`, `INTER_METHOD_FUSION`,
`WBF_WEIGHT_MODE`, `BALANCE_RATIO`, or `CD_ALPHA_ATTACK` without
recalibration will now hard-raise `RuntimeError("[A1.9] Threshold
sidecar/config mismatch …")` at threshold load time.

---

## 4. Updated assumption status

| ID | Audit status before | Audit status now | Residual risk |
|---|---|---|---|
| A1.3 declustering OFF | Justified by Prophet whitening | Justified for 12/17 metrics; documented sensitivity (≤25 % T_atk shift) for 5 volatile metrics. | If the operator changes the sampling cadence or adds bursty metrics, re-run the lightweight ablation. |
| A1.5 single global EVT threshold per metric | Possible regime drift, not measured | Measured: weekday-day FPR 4.5× target, weekend / night 0×. Year-projection 2.4×. | Term-vs-vacation residual variance is not modelled. Adding a calendar-aware EVT (one threshold per calendar bucket) is the next logical step for a year-long deployment. |
| A1.9 surrogate-vs-deployed mismatch | Documented in sidecar caveat field, no enforcement | Hard-raise at runtime if any sensitive knob mismatches; `eval_threshold_sweep.csv` reports realised vs target FPR + status flag; 2026-05-07 adds mode-specific WBF/ABF sidecars and a strict recalibrated comparison harness. | The gap is reduced in auditability but not eliminated: the ageing-aware WBF/ABF recalibration reached 4.31-4.34 % realised FPR on evaluation for a 0.1 % holdout target. Keep the production WBF sidecar unless a fresh full-chain calibration validates target FPR. |
| A3.2 non-injected = normal | REAL_ATTACK_CATALOG only used by `evaluate_real_ddos.py`; outage windows counted as FP in main FAR | Catalog and outage windows now consolidated in both qualifier-side and detector-side FAR; tests prevent regression. | None within the codebase; remaining risk is operator-side (un-catalogued real attacks within the RedeRio test span). |
| A3.3 injection signatures too clean | Risk acknowledged but not measured | Linear QP degradation with σ documented; no collapse up to σ=0.20. | The perturbation model is multivariate Gaussian on normalised triplets; real-traffic noise may be heavier-tailed. Treat the curve as a lower-bound robustness check. |
| A6.4 evidence_scale=3.0 heuristic | "must recalibrate on real noisy data" comment | Heatmap shows a broad plateau; published value sits in the interior. | The plateau is established on synthetic catalog only; A3.3 noise ablation is the partial answer to "what if real data is noisier." |
| A6.5 u_raw>0.82 heuristic | Same | Heatmap shows the same broad plateau in the u_raw axis. | Same caveat as A6.4. |
| A7.3 Wilson CI ignores autocorrelation | Open | Wilson CIs now use Newey-West n_eff. | The lag truncation L=10 is a default; sensitivity to L is not yet swept. |
| A7.5 BCa is iid | Open (block bootstrap mentioned but not implemented) | Block bootstrap implemented and used in the headline F1/MCC CIs (block_length = median episode length, 36 windows on RedeRio). | Block length is heuristic; could be tuned per metric. The Politis-Romano subsampling alternative is not implemented. |

---

## 5. How to reproduce

```bash
# Tests (snapshot 2026-05-07 LOT 1: 312 passed, 5 deselected)
python -m pytest tests/ -q --ignore=tests/test_cli_run_pipeline.py

# Param sensitivity (≈ 2 minutes)
PYTHONPATH=src python -m sl_ads.ablation.ablation_sbn_param_sensitivity

# Signature noise (≈ 2 minutes)
PYTHONPATH=src python -m sl_ads.ablation.ablation_signature_noise

# Regime FPR + year projection (< 1 minute)
PYTHONPATH=src python -m sl_ads.ablation.evaluate_regime_fpr

# EVT declustering — lightweight mode (< 1 minute)
PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m sl_ads.ablation.ablation_evt_declustering

# EVT declustering — full retraining (multi-hour, only if lightweight flags large deltas)
PYTHONPATH=src python -m sl_ads.ablation.ablation_evt_declustering --mode full --execute

# Re-run main detection evaluation with the new columns
PYTHONPATH=src python -m sl_ads.evaluate.evaluate_injection
```

---

## 6. Open follow-ups (NOT in scope of 2026-05-06 pass)

1. **Multi-seed evaluation** (5 seeds, mean ± std) on F1 / MCC / VUS —
   code in `src/sl_ads/evaluate/run_multi_seed.py`, just needs to be
   run (8-10 h compute).
2. **Calendar-aware EVT thresholds** for A1.5 — _audit-grade opt-in
   shipped 2026-05-07 (Phase B PATCH H2), default OFF based on
   empirical evidence_. Per-regime calibration is implemented but the
   on-disk benchmark on the canonical RedeRio reference run shows
   per-regime ``t_atk`` values cluster within ≈ ±5 % of the global
   ``t_atk`` (median ACTIVE/GLOBAL = 1.045, QUIET/GLOBAL = 0.983).
   First-order tail estimate: this would reduce the FPR overshoot
   from 7.02× to ≈ 3.5× on `canonical_ACTIVE` — still well above the
   1.0× target. Conclusion: the regime-FPR overshoot is **not
   located at the per-metric EVT calibration step**; root-cause
   investigation under Option C is in
   `docs/review/regime_fpr_root_cause_analysis.md`.

   The H2 code is preserved as opt-in (`CALENDAR_EVT_ENABLED=False`
   by default) for (a) ablation studies, (b) datasets where
   per-regime heteroscedasticity may be larger than on RedeRio, and
   (c) audit traceability (we considered, tested, measured). 22 unit
   tests in `tests/test_calendar_aware_evt.py`; design and post-mortem
   in `docs/review/calendar_evt_design.md`.
3. **Modern SOTA baselines** — TranAD / AnomalyTransformer / TimesNet
   under the same RedeRio protocol. Plan emitted; needs GPU and the
   external public repos. (Phase B item — USAD as primary candidate
   given local CPU-only constraints; full Kitsune deferred to Phase D
   along with the raw-NetFlow data refactoring.)
4. ~~**Heavy-tailed signature noise** (A3.3 refinement)~~ — _CLOSED
   2026-05-07 (Phase A L3.3)_. Cauchy and Student-t (df=3) variants
   shipped via `--distributions` flag of
   `ablation_signature_noise.py`. QP degrades smoothly without
   collapse: 0.422 (Gaussian σ=0.20) / 0.369 (Student-t) / 0.253
   (Cauchy). One unit test in `tests/test_scientific_hardening_20260505.py`.

## 7. Items intentionally not fixed (with reasoning)

The following proposed fixes were considered and *declined* with explicit
scientific or engineering rationale; they are documented here so a
reviewer can audit the decision rather than assume oversight.

| Proposal | Decision | Rationale |
|---|---|---|
| Add `dst_port_entropy` to distinguish DNS_AMP from NTP_AMP | DECLINED | The metric is already in the dataset and in the `entropy` group of `QUALIFY_GROUP_SOURCES`. The aggregation level (30 s windows, entropy of the dst-port distribution) cannot distinguish "concentrated on port 53" from "concentrated on port 123" — the entropy is low in both cases. Distinguishing would require per-port flow data, which the standardised CSV does not contain. Documented in §11.3 of `ASSUMPTIONS.md`. |
| `R_init = a_edp · 2W` for NETWORK_OUTAGE_NOV17 cold-start | DECLINED | The bijection makes `proj_atk_init = a_atk` invariant in the magnitude scalar of `R_init` (algebra in `opinions_pipeline.py:531-543`). Increasing the prior weight only slows convergence to fresh evidence — the *opposite* of the intended fix. Mathematically counterproductive. |
| `FINAL_SYSTEM_CBF` column rename to `FINAL_SYSTEM_FUSED` | DECLINED | Forward-compatible alias `paths.get_detection_col_fused` already exists (PATCH TASK-44). 53 callsites for cosmetic gain. The actual fusion mode is recorded in `fusion_mode_at_compute_opinions.json`. |
| Per-metric block length in `paired_bootstrap_bca_ci` | DECLINED for headline | Empirical ACF on `FINAL_SYSTEM_CBF_proj_atk` crosses 1/e at lag 31; current block 36 is within 5 % of the empirically optimal value. Per-attack refinement matters only for individual PPV CIs, where the existing Newey-West n_eff (max_lag=10) already captures binary-indicator autocorrelation correctly. |
