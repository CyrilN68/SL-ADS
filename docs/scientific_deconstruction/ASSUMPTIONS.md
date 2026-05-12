# ASSUMPTIONS.md — Critical Inventory of Assumptions

> **Current status as of 2026-05-12.** This document used to list 50+
> assumptions of which 9 were *open* (no enforcement, no measurement).
> After the hardening passes, the residual open items are limited to those
> listed in §11.4 below; every other CRITICAL/HIGH assumption now has either a
> code-level guard rail OR a quantitative measurement artefact. The dated
> 2026-05-06 hardening report is archived; current status lives in
> `docs/AUDIT_CURRENT_STATUS.md` and `docs/audit/audit_verification_tracker.md`.
> Current paper-facing numbers are from complete run `2e12261d55a8f975`.
> Numeric examples from older hardening runs are retained as diagnostic history
> unless an active status document marks them as current paper-facing results.
>
> Each assumption below is tagged with:
> - **Type** ∈ {Statistical, SL-theoretical, Data, Numerical, Engineering}
> - **Visibility** ∈ {Explicit, Implicit}
> - **Sensitivity** ∈ {LOW, MEDIUM, HIGH, CRITICAL}
> - **Location** = file:line (or file:function for behaviour-level assumptions)
> - **Formal expression** when meaningful
> - **What breaks if violated** — concrete failure mode in this codebase
> - **2026-05-06 status** when the assumption was strengthened or measured.

Assumptions are grouped by file. The order within a file is roughly *upstream-first*.

---

## 1. `src/sl_ads/train/train_models.py`

### A1.1 — Training span contains no real attack
- **Type**: Data
- **Visibility**: Implicit (only enforced by the `TRAIN_EXCLUSIONS` list)
- **Sensitivity**: **CRITICAL**
- **Location**: `train_models.py:1184–1204`; `config.py:TRAIN_EXCLUSIONS` (50+ ranges)
- **Formal**: ∀ t ∈ `df_train`, `label(t) = normal`.
- **What breaks**: EVT thresholds, EDP and `DECISION_THRESHOLD` are all calibrated on this span. A real attack hidden inside `df_train_calib` shifts the surrogate `proj_atk` distribution → `δ` is biased high → operational FPR-budget mismatched. RedeRio has documented exclusions, but this assumption cannot be re-verified by the pipeline itself.
- **2026-05-06 mitigation**: a defensive *unsupervised* second-opinion audit script (`src/sl_ads/audit/audit_train_span.py`) now runs the existing `ConsensusLabeller` (STL+Hampel+CUSUM) on the train span only and writes `outputs/scientific_hardening/audit_train_span.csv` plus `_summary.json`. Reference run on RedeRio with the 3 volumetric metrics (bytes/packets/flows): 80,642 train windows, 3,120 already in `TRAIN_EXCLUSIONS`, 306 *new* high-severity (3 metrics consensus) candidates, 4,751 medium (2 metrics), 13,969 low (1 metric). The top-N suspect windows (weekend evenings 2025-10-18 → 2025-10-26) are listed in the summary for manual review and are NOT automatically added to `TRAIN_EXCLUSIONS`. The script is intentionally non-destructive.

### A1.2 — Heavy-tail / GPD validity above `t₀ = quantile(0.90)`
- **Type**: Statistical (EVT)
- **Visibility**: Explicit (Coles 2001 §4.2 condition `σ̃ = σ − ξ·t₀ > 0` is checked)
- **Sensitivity**: **HIGH**
- **Location**: `train_models.py:413–451` (`_evt_threshold_pair`)
- **Formal**: For excesses `Y = X − t₀ | X > t₀`, `Y ~ GPD(ξ, σ)` (Pickands 1975, Balkema–de Haan 1974) on the calibration span.
- **What breaks**: When `σ̃ ≤ 0`, GPD is unstable — fallback to empirical quantile (recorded in `_FALLBACK_LOG['evt_sigma_mod']`). Cited as "EVT instable 7/17" in `docs/honest_limitations.md`. Empirical fallback is non-extrapolating: if observed peaks under-represent the right tail, `T_atk` is biased low → false positives in deployment.
- **2026-05-06 strengthening**: the EVT fallback chain in `_evt_threshold` now reads `Grimshaw MLE → PWM (Hosking-Wallis 1987) → scipy.genpareto.fit → empirical quantile` — PWM is inserted as a closed-form "MLE-failed" fallback that is the standard EVT recommendation for `xi > 0.5` heavy tails (Coles 2001 §4.3.4) and for small samples. This narrows the empirical-quantile regime to the genuinely malformed cases. PWM correctness verified against synthetic GPD samples (tests `test_pwm_gpd_fit_recovers_known_parameters` and `test_pwm_gpd_fit_handles_heavy_tail_xi_above_half`).

### A1.3 — Independent residual exceedances (no declustering needed)
- **Type**: Statistical (EVT)
- **Visibility**: Explicit (declustering disabled, `EVT_DECLUSTER_RUN=−1`)
- **Sensitivity**: **MEDIUM**
- **Location**: `train_models.py:387–394`
- **Formal**: Excesses are i.i.d. given `t₀`.
- **Justification (in code)**: Prophet residuals are pre-whitened (Taylor & Letham 2018); reconstruction residuals on QR(0.5) at 30 s sampling are assumed to exhibit only microbursts.
- **What breaks**: If residuals exhibit run-length clusters (e.g. flash-crowd episodes), `n_peaks` over-counts non-independent exceedances → MLE biased, GPD quantile `T_q` shifted. The implementation has the Davison–Smith (1990) declustering code, but it is not used in production.
- **2026-05-06 measurement** (lightweight ablation harness `src/sl_ads/ablation/ablation_evt_declustering.py`, artefacts under `outputs/scientific_hardening/evt_declustering_*`):
  - Out of 17 RedeRio metrics, 12 are insensitive to `EVT_DECLUSTER_RUN ∈ {1, 3, 5}` (Δ T_susp = 0).
  - 5 volatile metrics (`prophet_packets`, `prophet_udp`, `reconst_udp_from_flows`, `reconst_fin_from_syn`, `prophet_avg_pkt_size`) shift by up to 24.7 % on T_susp and 25.3 % on T_atk.
  - Verdict: declustering OFF is justified for the bulk of the metric set but NOT a free lunch for highly bursty traffic features. The `--mode full --execute` harness is preserved for cases where the operator wants a downstream re-evaluation, but the lightweight mode answers the audit's question (does declustering meaningfully change the calibration?) with a quantitative yes-on-5/no-on-12 split.

### A1.4 — `EVT_MIN_PEAKS = 50` sufficient for stable MLE
- **Type**: Statistical
- **Visibility**: Explicit (`config.py:EVT_MIN_PEAKS = 50`; check at `train_models.py:301, 385, 414`)
- **Sensitivity**: MEDIUM
- **Formal**: Sample size for Grimshaw MLE; below threshold ⇒ empirical quantile.
- **What breaks**: For `ξ > 0.5` (variance-infinite tail), Grimshaw's MLE convergence is slow (Coles 2001 §4.3); `n=50` may admit non-trivial bias on the right tail. The code falls back to empirical quantile, which is conservative for `T_susp` (under-estimating the upper extreme).

### A1.5 — `Q_*` excess probabilities are correctly translated for the deployment regime
- **Type**: SL-theoretical / Statistical
- **Visibility**: Explicit (`EVT_Q_*` interpreted as `P(|residual| > T | normal)`)
- **Sensitivity**: HIGH
- **Location**: `train_models.py:530–537`
- **Formal**: After whitening by Prophet/QR, residuals are zero-mean stationary; the marginal exceedance probability is invariant in deployment.
- **What breaks**: Prophet residuals are *not* strictly stationary across regimes (e.g. holiday vs term-time on RedeRio). The code uses *one global threshold per metric*; if the residual variance differs between train and test seasons, the realised FPR diverges from `EVT_Q_*`.
- **2026-05-12 measurement** (`src/sl_ads/ablation/evaluate_regime_fpr.py`, artefacts under `outputs/scientific_hardening/regime_fpr.*`): the audited span covers 45 days (2025-11-10 -> 2025-12-25). Realised FPR by calendar/time regime, with the operator target of 0.001:

| Regime | n_normal | FP | FPR | Ratio to target |
|---|---:|---:|---:|---:|
| all_normal | 12,015 | 116 | 0.965 % | 9.65x |
| weekday_term_like | 7,676 | 114 | 1.485 % | 14.85x |
| weekend | 3,366 | 1 | 0.030 % | 0.30x |
| holiday_or_closure | 1,261 | 1 | 0.079 % | 0.79x |
| day_08_18 | 4,867 | 91 | 1.870 % | 18.70x |
| night_00_06 | 3,027 | 2 | 0.066 % | 0.66x |
| shoulder_06_08_18_24 | 4,121 | 23 | 0.558 % | 5.58x |
| canonical_ACTIVE | 3,135 | 91 | 2.903 % | 29.03x |
| canonical_QUIET | 8,880 | 25 | 0.282 % | 2.82x |

The **regime-weighted year projection** (assuming the observed regime mix is representative) is 1006.8 expected FP/year vs the 105.1 implied by the target, i.e. the realised FPR is 9.58x the target on a year-long deployment.

For a true year-long deployment, per-regime EVT thresholds are one principled
calibration primitive, but the complete-run root-cause analysis shows they are
not sufficient by themselves on RedeRio because the dominant mechanism is
correlated joint exceedance after fusion. The audit-grade implementation and
post-mortem are documented in `docs/review/calendar_evt_design.md`; it remains
off by default for the RedeRio reference configuration.

**2026-05-07 (PATCH H2 — calendar-aware EVT).** The per-regime EVT primitive
above is now implemented as an **audit-grade opt-in**. The new module
`src/sl_ads/calendar/regime.py` provides a canonical 2-bucket partition
``ACTIVE`` (weekday × not-holiday × hour ∈ [08, 18)) vs ``QUIET``
(everything else); ``train_models.calibrate_thresholds_per_regime_v2``
calibrates an EVT threshold pair per bucket and persists the result in
``models_pkg[metric]['thresholds_per_regime']``;
``train.compute_evidence`` reads the per-window regime via
``regime_of`` and dispatches to the bucket-specific thresholds when the
block is present (legacy scalar fields remain a defensive fallback for
old PKLs). The threshold sidecar gains a
``calendar_evt_signature`` field (versioned regime-fn signature)
enforced as a sensitive knob in
``paths.validate_threshold_sidecar_config`` so any drift between the
calibration-time and runtime regime function hard-raises (A1.9).

**Default ``CONFIG['CALENDAR_EVT_ENABLED'] = False``** — the published
reference uses the legacy single-threshold path because of the empirical
finding below. The flag is exposed for ablation studies and for datasets
with stronger heteroscedasticity than RedeRio.

**Empirical finding (post-mortem, 2026-05-07).**  An on-disk benchmark
of per-regime vs global EVT thresholds was run on the canonical
RedeRio reference run, using the persisted Prophet models and a chunked
``model.predict`` over the train span (no retraining required).  For
the 12 Prophet metrics the per-regime ``t_atk`` values cluster very
close to the global ``t_atk``:

| Statistic | ACTIVE / GLOBAL | QUIET / GLOBAL |
|---|---:|---:|
| Median ratio | 1.045 | 0.983 |
| Mean ratio   | 0.998 | 1.004 |

i.e. the per-regime thresholds differ from the global threshold by
≈ ±5 % in the central tendency.  Some metrics show meaningful per-bucket
divergence (`prophet_bytes` A/Q = 1.81; `prophet_entropy_src_port`
A/Q = 0.42, a counter-example where QUIET has the heavier tail), but
the median is essentially unchanged.

The complete-run follow-up confirms that **the regime-FPR mechanism is not
located only at the per-metric EVT calibration step**. The current
`regime_fpr_diagnosis` verdict is `H_correlation`: median per-metric
ACTIVE/QUIET exceedance = 0.571, fused p99.9 ratio = 1.522, and joint k=3
exceedance ratio = 4.524. The issue is correlated joint alarms in ACTIVE
windows, amplified by fusion. See
`docs/review/regime_fpr_root_cause_analysis.md`.

Design rationale and full sidecar API: see
``docs/review/calendar_evt_design.md``. Tests:
``tests/test_calendar_aware_evt.py`` (22 cases covering
``regime_of`` semantics, per-regime calibration, dispatcher
invariants, and sidecar validator backward compatibility).

### A1.6 — Prophet `growth='flat'` is appropriate
- **Type**: Engineering
- **Visibility**: Explicit (`train_models.py:1490`)
- **Sensitivity**: MEDIUM
- **Formal**: ∂g/∂t = 0 over the training horizon.
- **What breaks**: For datasets exhibiting multi-week trend (CESNET ISP capacity ramp), the `flat` choice forces residual mean drift → false positives at the trend-break boundary.

### A1.7 — QR(q=0.5) breakdown extends to physical-constraint-induced lack of leverage outliers
- **Type**: Engineering / Statistical
- **Visibility**: Explicit (lines 1244–1253)
- **Sensitivity**: MEDIUM
- **Formal**: For the rules with `fit_intercept=False` (e.g. `bytes ← packets`), the constraint `0 packets ⇒ 0 bytes` (Bridgman 1922) is satisfied physically, so leverage outliers are absent. LAD breakdown for response outliers is 50 % (Koenker & Bassett 1978).
- **What breaks**: Rules with `fit_intercept=True` (`udp ← flows`, `fin ← syn`, `tcp ← packets`) are *not* protected from leverage outliers; if a heavy-tailed feature contaminates training (despite exclusions), QR(0.5) inherits that bias and `R²_CV` may swing to negative — triggering the `DummyRegressor` mean fallback, which in turn produces *constant* residuals, killing `c3_online_rmse` weighting.

### A1.8 — `R²_CV ≥ 0` ⇒ R² is a meaningful "trust score"
- **Type**: Engineering
- **Visibility**: Explicit (PATCH-M1, lines 1303–1316)
- **Sensitivity**: HIGH (when `WBF_WEIGHT_MODE='trust_discount'`)
- **Formal**: 5-fold `TimeSeriesSplit` cross-validated `R²` is positive and proportional to predictive quality.
- **What breaks**: `trust_discount` mode is documented as pathological on RedeRio (5/12 Prophet metrics with `R²<0`). In the complete run `2e12261d55a8f975`, the harmonised ablation reports Full SL-ADS at `F1-cov=0.879` / `FPR=0.965%` while legacy R² trust-discount falls to `F1-cov=0.628` / `FPR=4.39%` and detects only `12/14` attacks at the calibrated operating point. Production mitigation: `WBF_WEIGHT_MODE='uniform'` (default) ignores R². The pathology is well-documented in `docs/audit/trust_discount_r2_analysis.md`.
- **2026-05-12 (PATCH D5 refresh) — MASE alternative tested and rejected for default**: a MASE-based trust map (Hyndman-Koehler 2006, `src/sl_ads/stats/mase.py`) is exposed as `WBF_WEIGHT_MODE='mase'` and as the `mase_legacy` ablation run. Empirical evaluation on the canonical RedeRio reference run shows that MASE fails *symmetrically* to R²: at 30 s sampling, Naive-1 persistence dominates most Prophet metrics, and under the canonical alpha=1 trust map the ablation silences the detector (0/14 attacks detected). R² and MASE disagree on which metrics are "trustworthy" and are uncorrelated on this dataset. Both proxies measure benign-regime predictive skill, which is **not** the property required for anomaly-detection trust. `'uniform'` therefore remains the published default; both `'trust_discount'` and `'mase'` are kept as audit-grade ablation modes. Detail: `docs/audit/trust_discount_r2_analysis.md` §4.1.

### A1.9 — DECISION_THRESHOLD calibration surrogate matches the deployed pipeline
- **Type**: SL-theoretical
- **Visibility**: Explicit (PATCH TASK-45, sidecar field `calibration_surrogate_caveat`)
- **Sensitivity**: **CRITICAL**
- **Location**: `train_models.py:1929–1944`, `_compute_training_proj_atk` (lines 976–1080); enforcement in `paths.validate_threshold_sidecar_config` (added 2026-05-06).
- **Formal**: The legacy generic threshold sidecar is calibrated on a training surrogate, while the deployed chain applies ageing, intra-method WBF, method grouping, inter-method fusion, and optional contextual discount. The two are equal only if these operators are identity transformations on the calibration span.
- **What breaks**: Any production change to `LAMBDA_DECAY`, `INTER_METHOD_FUSION`, `WBF_WEIGHT_MODE`, `BALANCE_RATIO`, `CD_ALPHA_ATTACK`, or method-group structure after training can invalidate the calibrated `δ`. The sidecar stores the calibration-time configuration and runtime code now checks sensitive mismatches.
- **2026-05-06 enforcement**: `paths.validate_threshold_sidecar_config` is now invoked automatically by `paths.get_decision_threshold` and **raises `RuntimeError("[A1.9] Threshold sidecar/config mismatch …")`** when any of the five sensitive knobs disagrees between the runtime CONFIG and the calibration sidecar. Tests `test_sensitive_sidecar_config_match_passes` and `test_sensitive_sidecar_config_mismatch_raises` (in `tests/test_config_and_sidecar.py`) lock this in.
- **2026-05-12 complete-run reporting**: `eval_threshold_sweep.csv` and `eval_summary_v3.json` expose `fpr_target`, `fpr_ratio_to_target`, and `fpr_target_status`. Status `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` is emitted whenever the realised FPR exceeds 2× the target. On the complete RedeRio run `2e12261d55a8f975`, the realised global FPR is `0.965%` (`9.65×` the 0.1% target), and canonical ACTIVE reaches `2.903%` (`29.03×`). This is now the paper-facing limitation.
- **2026-05-07 mode-specific calibration**: strict WBF/ABF sidecars can replay the deployed fusion stack from persisted opinions and calibrate each mode under its own threshold. The dated WBF/ABF strict run (`FPR=4.31-4.34%`) is retained as a historical fusion-operator diagnostic only; final paper-facing detection and FPR values come from the complete run `2e12261d55a8f975`.
- **Residual risk**: the code now prevents unnoticed configuration drift and supports mode-specific fusion sidecars, but realised FPR can still exceed the nominal training target under regime shift. This is a calibrated-performance limitation, not a hidden mismatch.

### A1.10 — `proj_atk` is a smooth surrogate for `b_atk`
- **Type**: SL-theoretical
- **Visibility**: Explicit (lines 953–959)
- **Sensitivity**: LOW
- **Formal**: `proj_atk = b_atk + a_atk·u`. With `a_atk` near `1/3`, `proj_atk ≥ b_atk` always; `proj_atk ≥ a_atk` (vacuous opinion). For `u → 0`, `proj_atk → b_atk`.
- **What breaks**: If `a_atk` is near 1 (very anomalous prior), `proj_atk` is dominated by `u·a_atk` and the threshold detects *uncertainty*, not *belief in attack*. EDP floor `0.005` keeps `a_atk` small; the assumption holds.

### A1.11 — EDP captures stationary prior on a per-metric basis
- **Type**: Statistical / Empirical Bayes
- **Visibility**: Explicit (lines 634–659)
- **Sensitivity**: MEDIUM
- **Formal**: `a_safe ≈ E[r_safe/n_window]`, `a_susp ≈ E[r_susp/n_window]`, `a_atk ≈ E[r_atk/n_window]`. Computed on training residuals.
- **What breaks**: If the test span has a different residual distribution (e.g. seasonal shift), the EDP biases all opinions; in particular, `a_atk` is never updated online (the legacy `adaptive_base_rate` module is archived and *not* loaded — see `compute_opinions_pipeline.py:24–31`). The `EMPIRICAL_PRIOR_FLOOR = 0.005` prevents `a_atk = 0` cold-start, but does not protect against drift.

---

## 2. `src/sl_ads/train/compute_evidence.py`

### A2.1 — Same NaN policy in train and inference
- **Type**: Engineering / Data
- **Visibility**: Explicit (audit constat #2)
- **Sensitivity**: HIGH
- **Location**: `compute_evidence.py:153–161`, `train_models.py:1149–1156`, both routed through `preprocessing_utils.preprocess_metrics`.
- **Formal**: limited forward-fill (`limit=NAN_FFILL_LIMIT=10` ≈ 5 min), no `fillna(0)` on network metrics.
- **What breaks**: A divergent imputation between training and evidence stages would shift residuals → calibrated thresholds invalid. The unique-policy is enforced by sharing `preprocess_metrics`, but `non_metric_cols` get `ffill().fillna(0)` (acceptable) while metric cols only get bounded `ffill`. The split is correct.

### A2.2 — `split_date` consistency between trained PKL and current `CONFIG`
- **Type**: Data
- **Visibility**: Explicit (anti-leak check, lines 184–194)
- **Sensitivity**: **CRITICAL**
- **Formal**: `models_pkg['_meta_split_date'] == CONFIG['split_date']`.
- **What breaks**: If a model trained with split `S1` is reused with a config pointing to `S2`, evaluation could (i) include training data in test, leaking labels; (ii) miss real attacks. The check `return` early on mismatch — strict.

### A2.3 — Window invariant `P + S + N = n_window`
- **Type**: SL-theoretical (bijection input shape)
- **Visibility**: Explicit (docstring lines 67–69)
- **Sensitivity**: HIGH
- **Formal**: For each window of size `n`, `Σ_j (p_j + s_j + n_j) = n` because the trapezoidal map outputs convex combinations summing to 1.
- **What breaks**: Partial windows (last batch, `n < WINDOW_SIZE`) are accepted but with `P+S+N = n < WINDOW_SIZE` (PATCH M-06/F09). The bijection then produces `u = W/(W+n) > W/(W+WINDOW_SIZE)`, which is the SL-correct response (less evidence ⇒ more uncertainty). Documented but worth flagging: per-window `proj_atk` is *not* directly comparable across full vs partial windows.

### A2.4 — Trapezoidal map is monotone in `|residual|` and continuous
- **Type**: Engineering
- **Visibility**: Implicit (piecewise-linear by construction)
- **Sensitivity**: LOW
- **Location**: `compute_evidence.py:38–96`
- **What breaks**: Continuity at the breakpoints `t_trap, t_susp, t_atk` is enforced by the linear ramps. Monotonicity ensures larger residual ⇒ no smaller `n` evidence.

### A2.5 — Direction tag on a metric is correct
- **Type**: Engineering / Domain
- **Visibility**: Explicit (`CONFIG['ASYMMETRIC_THRESHOLD_METRICS']`)
- **Sensitivity**: HIGH
- **Formal**: For each metric, `direction ∈ {pos, neg, both, sym}`. Directional filtering ensures residuals in the "wrong" sense produce `(p,s,n)=(1,0,0)` (line 71–79).
- **What breaks**: Misclassifying `direction='pos'` for a metric where deficits are also anomalous (e.g. SLOWLORIS lowers byte volume) → directional filter zeroes evidence → false negatives. The catalog encodes a `direction='both'` option that emits five-state evidence (`S_pos, N_pos, S_neg, N_neg`) preserving the coarsening identity (Jøsang §3.5.4): `S = S_pos + S_neg`, `N = N_pos + N_neg`.

---

## 3. `src/sl_ads/inject/evidence_level.py`

### A3.1 — Catalog windows are disjoint
- **Type**: Engineering
- **Visibility**: Explicit (overlap check, lines 876–893)
- **Sensitivity**: HIGH
- **Formal**: ∀ i ≠ j, `[t_start_i, t_end_i] ∩ [t_start_j, t_end_j] = ∅`.
- **What breaks**: Without disjointness, the second injection overwrites the first → multiple labels per window are silently impossible (single-label only). The check raises explicitly.

### A3.2 — Non-injected windows have ground-truth label `normal`
- **Type**: Data
- **Visibility**: Implicit (assumed by the evaluator)
- **Sensitivity**: **CRITICAL**
- **Formal**: ∀ t ∉ ⋃ catalog windows, `label(t) = normal`.
- **What breaks**: If RedeRio contains *unlabelled* real attacks outside the catalog, those windows count as FP for the SL detector → headline FPR is artificially inflated. Conversely, the IF baseline could be tuned on contaminated normal data → unfair comparison. The pipeline contains `REAL_ATTACK_CATALOG` (1 entry on RedeRio, Nov 12 UDP DDoS), but only used by `evaluate_real_ddos.py`, not subtracted from the global FAR base.
- **2026-05-06 fix** — both reporting sites now consolidate the catalogue:
  - `evaluate_qualify_sbn._compute_global_detection_stats` adds `CONFIG["EVAL"]["REAL_ATTACK_CATALOG"]` to the attack/outage tri-class partition (test `test_real_attack_catalog_is_subtracted_from_global_far`).
  - `evaluate_injection.windows_outside_attacks` strips REAL_ATTACKS (including NETWORK_OUTAGE) from the FPR denominator; the `(tn, fp)` count and the bootstrap CI are computed on the outage-excluded vectors (test `test_threshold_sweep_excludes_real_attacks_outage_from_fpr`).
  - Complete-run reporting now exposes both protocols. `catalog_outages_separate` gives F1 micro 0.8666, F1 macro 0.9292, FPR 0.965%; `operator_faithful_anomaly` gives F1 micro 0.8257, F1 macro 0.9056, FPR 0.965%. The paper must state whether outages are counted as positives or reported separately.

### A3.3 — Injection signature is representative of real attacks (no domain shift)
- **Type**: Engineering / Data
- **Visibility**: Implicit (signatures are literature-derived, not network-calibrated)
- **Sensitivity**: **HIGH**
- **Formal**: For each `(metric, attack)`, `(P_norm, S_norm, N_norm)` reflects the *attack-conditional distribution* on RedeRio.
- **What breaks**: The signatures inject *active safe* signals on discriminative metrics (e.g. UDP_FLOOD has `prophet_icmp = (0, 1, 9)` ⇒ strong "this is NOT ICMP" evidence). Real-world attacks may not produce such clean negative signals. Headline F1 is a *conservative-to-optimistic* upper bound depending on which way the bias flows; precision on attack-type qualification is the most exposed metric (over-fits to synthetic discriminators).
- **2026-05-06 measurement** (`src/sl_ads/ablation/ablation_signature_noise.py`, artefacts under `outputs/scientific_hardening/signature_noise_ablation.*`): Gaussian perturbation σ ∈ {0, 0.05, 0.10, 0.15, 0.20} on the normalised injected (P,S,N) triplets, 5 repetitions:

| σ | mean QP | DR (gate) | autre_rate |
|---:|---:|---:|---:|
| 0.00 | 0.607 | 0.828 | 0.0 |
| 0.05 | 0.580 | 0.828 | 0.0 |
| 0.10 | 0.533 | 0.828 | 0.0 |
| 0.15 | 0.481 | 0.828 | 0.0 |
| 0.20 | 0.426 | 0.828 | 0.0 |

QP degrades roughly linearly without collapsing — at σ=0.20 (a perturbation comparable to a noisy real-traffic attack) the qualifier still typifies 60 % of detected windows correctly. The DR is unchanged because the gate is open/closed based on `FINAL_SYSTEM_CBF_proj_atk`, which is *upstream* of the qualifier and therefore not affected by triplet noise. Heavier-tailed (Cauchy / Student-t) noise is documented as a residual follow-up.

### A3.4 — `α(t)` ramp profile preserves bijection invariant
- **Type**: SL-theoretical
- **Visibility**: Explicit (lines 1015–1020 of `evidence_level.py`)
- **Sensitivity**: LOW
- **Formal**: `P_t + S_t + N_t = (1−α_t)·n_window + α_t·n_window = n_window`.
- **What breaks**: Algebraically guaranteed; numerically safe.

### A3.5 — Injection is invisible to the threshold calibrator
- **Type**: Engineering
- **Visibility**: Explicit (PATCH 2026-05-06 — hard-raise enforcement)
- **Sensitivity**: **CRITICAL**
- **Formal**: `df_train_calib` (used in `_compute_training_proj_atk`) ⊆ `df[df['ds'] ≤ split_date]`; injection window starts after `split_date`.
- **What breaks**: A catalog event placed before `split_date` would contaminate `δ` calibration. RedeRio catalog ranges from `2025-11-16` to `2025-12-20`, after `split_date=2025-11-09`. Verified by inspection; no automatic check exists.
- **2026-05-06 enforcement**: `evidence_level._validate_catalog` now raises `ValueError("[A3.5] …")` if any catalog event has `start <= CONFIG['split_date']`. Tests `test_a35_catalog_validator_rejects_pre_split_event` and `test_a35_catalog_validator_accepts_post_split_event` lock this in. The check is invoked unconditionally during the `inject` step, so an operator cannot bypass it by skipping the validator.

---

## 4. `src/sl_ads/core/subjective_logic.py`

### A4.1 — Opinion validity `Σb + u = 1`
- **Type**: SL-theoretical (Jøsang Def. 3.1)
- **Visibility**: Explicit (`MultinomialOpinion.__init__`, lines 64–69)
- **Sensitivity**: **CRITICAL** (foundation)
- **Formal**: `b ∈ R³_{≥0}`, `u ∈ [0,1]`, `b_safe + b_susp + b_atk + u = 1`.
- **Enforcement**: Renormalisation if drift detected (lines 64–66); `ValueError` if violation persists beyond `1e-6`.
- **What breaks**: All downstream operators rely on this. A bug here would silently produce non-probabilistic outputs.

### A4.2 — Bijection is well-defined for `r ≥ 0`, `Σr ≥ 0`
- **Type**: SL-theoretical (Jøsang Def. 3.9)
- **Visibility**: Explicit (lines 128–199)
- **Sensitivity**: HIGH
- **Formal**: `(r → ω): b_i = r_i/(W+Σr); u = W/(W+Σr)`; `(ω → r): r_i = W·b_i/u`. Cap on dogmatic case (`u<1e-9`) at `r_max = W·1e4`.
- **What breaks**: Negative residuals (impossible by construction since `(p,s,n)≥0`) would break it. The dogmatic cap may distort downstream WBF when one metric is near-dogmatic — Reconstruction with `R²>0.99` can saturate.

### A4.3 — Confidence weighting in WBF is bounded and non-negative
- **Type**: SL-theoretical / Numerical
- **Visibility**: Explicit (lines 562–580)
- **Sensitivity**: LOW
- **Formal**: `w_i = max(ext_w_i, 0) · (1−u_i) ≥ 0`; degenerate case `Σw < 1e-12` returns vacuous opinion.
- **What breaks**: A negative `R²` propagated as `ext_w` would create negative weights — the `max(ext_w_i, 0)` clamp fixes this. Note: in `r2_static` mode this means R²<0 metrics contribute *nothing* to WBF, which is the documented behaviour but may be undesired.

### A4.4 — Asymmetric escalation conflict is the design choice
- **Type**: Engineering (deviation from canonical Jøsang Eq. 12.4)
- **Visibility**: Explicit (lines 247–306, with extensive docstring)
- **Sensitivity**: **CRITICAL** (controls ageing dynamics)
- **Formal**: `K_asym = b_prev[safe]·b_curr[atk] + b_prev[atk]·b_curr[safe] + b_prev[safe]·b_curr[susp] + b_prev[susp]·b_curr[atk]`; explicitly omits de-escalation pairs.
- **What breaks**: Symmetric BCF (canonical) treats every cross-product equally; using asymmetric escalation makes ageing aggressive on attack onsets but slow on recovery. *If* the goal were minimum-time return-to-normal-FPR, the canonical form would be preferable. The function `compute_conflict_degree_canonical` exists for this comparison; production runs use the asymmetric form.

### A4.5 — `α = 1/K_max` ⇒ exact hard-reset on max conflict
- **Type**: Engineering / Numerical
- **Visibility**: Explicit (config.py end-of-file recomputation; lines 332–343 of `opinions_pipeline.py`)
- **Sensitivity**: HIGH
- **Formal**: For `WINDOW_SIZE=10`, `W=3`, `b_curr_max = 10/13`, `b_prev_max = 20/23`, so `K_max = b_prev_max·b_curr_max = 0.6688`, `α = 1.495`.
- **What breaks**: The "hard reset" property guarantees `λ_dyn = 0` at the maximum conflict; a different α (e.g., 1.0) only reduces λ partially, leaving residual evidence that biases the next window. Since the threshold `δ` is calibrated on a no-ageing surrogate, the ageing dynamics directly impact realised performance.

### A4.6 — WBF N-ary is `Eq. 12.27`-faithful, not literal `Eq. 12.22-24`
- **Type**: SL-theoretical
- **Visibility**: Explicit (PATCH M-01 / F01 docstring, lines 510–532 of `subjective_logic.py`)
- **Sensitivity**: MEDIUM
- **Formal**: Production WBF averages *evidence vectors* with weights `w_i = ext_w_i · c_i`, then bijection-back. Algebraically equivalent on N=2 with `ext_w` uniform; small float-discrepancies elsewhere.
- **What breaks**: If a reviewer expects line-by-line correspondence with Eq. 12.22, the N-ary form is *not* that; the codebase ships `fusion_wbf_canonical_two` for that use case. Acknowledged in code comments.

### A4.7 — CBF independence assumption is *known* to be violated and *consciously* not used by default
- **Type**: SL-theoretical (Jøsang Theorem 12.2)
- **Visibility**: Explicit (PATCH M-11/CBF, lines 376–393 of `opinions_pipeline.py`)
- **Sensitivity**: **HIGH** (only matters in CBF ablation runs)
- **Formal**: CBF evidence accumulation (Theorem 12.2) requires source independence. Prophet and Reconstruction operate on the same raw windows, and the residual-correlation audit reported cross max |rho|=0.915.
- **What breaks**: In `INTER_METHOD_FUSION='cbf'` runs, evidence is double-counted; for headline numbers the default is `wbf` precisely to avoid this. ABF was implemented and strictly recalibrated because it is theoretically attractive for dependent sources, but it did not outperform WBF on RedeRio. The legacy `cbf` mode is preserved for ablation/comparability.
- **2026-05-06 enforcement**: a `RuntimeWarning` is now emitted at module import time whenever `CONFIG['INTER_METHOD_FUSION'] == 'cbf'`. The warning quotes the cross max |rho|=0.915 between Prophet and Reconstruction (residual_correlation audit 2026-05-04) and steers the operator back to `'wbf'` for headline numbers. Test `test_a47_cbf_emits_dependence_warning` locks this in.

### A4.8 — `BALANCE_RATIO` is an heuristic deviation from Theorem 12.2
- **Type**: Engineering (explicit deviation)
- **Visibility**: Explicit (PATCH-m4, lines 345–371 of `opinions_pipeline.py`)
- **Sensitivity**: MEDIUM
- **Formal**: Legacy CBF mode can pre-multiply the dominant group's evidence by `1/ratio` before CBF. This knob is not part of the WBF/ABF headline path.
- **What breaks**: Not theoretically justified; prefer explicit method groups plus `INTER_METHOD_FUSION='wbf'` or `'abf'` for current experiments. RedeRio default `BALANCE_RATIO=1.0` (inactive).

### A4.9 — Trust discount is probability-sensitive (Def. 14.6)
- **Type**: SL-theoretical
- **Visibility**: Explicit (lines 924–945)
- **Sensitivity**: LOW (only in deprecated `trust_discount` mode)
- **Formal**: `b' = t·b`, `u' = 1 − t·(1−u)`. Constraint `Σb' + u' = 1` algebraically preserved.
- **What breaks**: Algebraically safe.

### A4.10 — Contextual discount α-vector is in [0,1]³
- **Type**: SL-theoretical (Mercier-Quost-Denoeux 2008)
- **Visibility**: Explicit (lines 948–987, `np.clip(α, 0, 1)`)
- **Sensitivity**: MEDIUM
- **Formal**: `b'_i = α_i · b_i`; `u' = 1 − Σ α_i·b_i`.
- **What breaks**: Negative α would create negative beliefs; the `np.clip` defends against this. In production `α = [1, 1, CD_ALPHA_ATTACK]` reduces only `b_atk` of the Reconstruction group.

---

## 5. `src/sl_ads/core/opinions_pipeline.py`

### A5.1 — `_target_freq = freq_data × WINDOW_SIZE` matches `compute_evidence` aggregation
- **Type**: Data alignment
- **Visibility**: Explicit (lines 437–451)
- **Sensitivity**: HIGH
- **Formal**: `n_window` slices of `freq_data` form one opinion window.
- **What breaks**: A mismatch would resample evidence to a frequency where rows no longer satisfy `P+S+N = n_window` ⇒ the bijection becomes inconsistent. The `origin='epoch'` setting is timezone-deterministic.

### A5.2 — State memory `R_init = a_edp · W` ⇒ initial `proj_atk = a_atk_edp`
- **Type**: SL-theoretical
- **Visibility**: Explicit (lines 503–530)
- **Sensitivity**: LOW (warm-up only)
- **Formal**: With `Σa_edp = 1` and `R_init = a_edp · W`, `Σ R_init = W` ⇒ `D = 2W`, `u_init = ½`, `proj_atk_init = a_atk`.
- **What breaks**: Without this prior, `R_init = 0` would give `u=1` initially and `proj_atk = a_atk` only after warm-up — same end-value, but the convergence dynamics under ageing are different. Acceptable.

### A5.3 — Three-level fusion order is interchangeable enough that headline F1 is robust
- **Type**: SL-theoretical / Engineering
- **Visibility**: Implicit (no formal proof of associativity across mixed operators)
- **Sensitivity**: MEDIUM
- **Formal**: Production order is `Conflict-Aware Ageing -> intra-method WBF per CONFIG['FUSION_METHOD_GROUPS'] -> inter-method fusion policy`. Reordering, for example inter-method first and ageing second, is not equivalent. Ageing operates on per-metric evidence, so it must precede group fusion.
- **What breaks**: A reviewer asking "does the order matter?" needs an explicit ablation. The codebase has the `ablation_fusion_mode.py` and `ablation_temporal_sbn.py` modules that touch order-related dimensions.

### A5.4 — `proj_atk` of the inter-method fused opinion is the headline detection score
- **Type**: SL-theoretical
- **Visibility**: Explicit (`paths.get_decision_variable` returns `proj_atk` from sidecar)
- **Sensitivity**: HIGH
- **Formal**: `D(t) = 1 iff proj_atk_FINAL(t) >= δ`. The CSV column `FINAL_SYSTEM_CBF_proj_atk` is read by every evaluator (PATCH TASK-44 documents that the prefix is historical and may correspond to WBF, ABF, CBF, BCF, projected CCF, MinBF, MaxBF, or hierarchical fusion).
- **What breaks**: A bug in `INTER_METHOD_FUSION` (e.g. silently fall through to CBF instead of WBF) would change the column semantics without renaming. The sidecar `fusion_mode_at_compute_opinions.json` records the actual mode for downstream cross-check.

### A5.5 — `c3_online_rmse` gating on `prev_proj_atk_metric` is per-metric
- **Type**: Engineering
- **Visibility**: Explicit (lines 562–567, with Chandola et al. 2009 reference)
- **Sensitivity**: LOW (only in `online_rmse` mode)
- **Formal**: A per-metric reset prevents one attacking metric from contaminating the RMSE baseline of unrelated metrics.
- **What breaks**: A global gate would erase RMSE state on every metric whenever any metric attacks — destroying useful baselines. Per-metric is correct.

---

## 6. `src/sl_ads/qualify/sbn_qualifier.py`

### A6.1 — Naive Bayes-style evidence summation across groups
- **Type**: Statistical (independence)
- **Visibility**: Explicit (lines 1080–1088 of `sbn_qualifier.py`)
- **Sensitivity**: MEDIUM
- **Formal**: `P(g₁,g₂,…|k) ≈ ∏_g P(g|k)` ⇒ `e(k) = Σ_g e_g(k)`.
- **Known violations** (acknowledged):
  - `volume ↔ protocol_tcp/udp` correlated for volumetric floods (bias *favours* correct attack ⇒ acceptable).
  - `tcp_flags ↔ reconstruction` originally double-counted; corrected by removing `reconst_fin_from_syn` from the `reconstruction` group.
- **What breaks**: Domingos & Pazzani (1997) shows Naive Bayes is robust under *moderate* dependence, but the bound is empirical. The qualifier's precision (QP) is the most exposed metric.
- **2026-05-06 measurement** (`src/sl_ads/ablation/ablation_qualifier_group_independence.py`, artefacts `outputs/scientific_hardening/qualifier_group_correlations*`): on the 12 production groups (66 pairs) the empirical max |rho| between projected-evidence vectors is

| regime | max abs rho | n_pairs HIGH (>=0.6) | n_pairs MODERATE (0.3-0.6) |
|---|---:|---:|---:|
| attack windows | 0.957 | 32 / 66 | 15 / 66 |
| normal windows | 0.450 | 0 / 66 | 27 / 66 |

The strong correlations on attack windows are EXPECTED — a UDP_FLOOD activates `volume`, `protocol_udp` and `reconstruction` simultaneously by construction — and add evidence in the SAME direction. Domingos-Pazzani robustness still holds for argmax decision (NB stays optimal under positive correlations on the correct class). The measured violation does NOT bias the QP downward; it only means the qualifier is OVER-confident on volumetric attacks. The reported QP is therefore a lower bound on what a properly de-correlated qualifier would achieve. Reviewer-grade verdict: `NB_VIOLATED_RECONSIDER_GROUPS` for ROC area; `NB_OK_FOR_TOP1_DECISION` for argmax classification.

### A6.2 — Geometric-mean group projection minimises KL to pooled opinion
- **Type**: Statistical (logarithmic opinion pooling)
- **Visibility**: Explicit (lines 605–651)
- **Sensitivity**: LOW
- **Formal**: `P^g_s = (∏_{m∈g} P^m_s)^{1/|g|}`, renormalised. Genest & Zidek (1986); Aczél & Daróczy (1975).
- **What breaks**: With one zero in the group, `P^g_s = 0` (multiplicative annihilation) — the codebase handles the missing-metric case explicitly (lines 626–630). For ε-protected projections, that is a non-issue.

### A6.3 — Conditional opinions `c^{k|g}(s)` are correct expert priors
- **Type**: Engineering (expert elicitation)
- **Visibility**: Explicit (manually defined, lines 138–511)
- **Sensitivity**: **HIGH**
- **Formal**: Per attack `k`, per group `g`, per state `s`, `c^{k|g}(s) ∈ [0,1]`, `Σ_s c^{k|g}(s) = 1`.
- **Sources cited**: Sharafaldin 2018 (CIC-IDS2017), Mirsky 2018 (Kitsune), Moustafa 2015 (UNSW-NB15), Rossow 2014 (DNS amp), Hutchins 2011 (kill chain), MITRE ATT&CK, Van Rijswijk-Deij 2014 (NTP BAF).
- **What breaks**: Templates calibrated on 3rd-party datasets transfer imperfectly to RedeRio; built-in sensitivity analysis (`--sensitivity`, perturbation ±0.05) measures internal stability but not robustness to network-specific noise.
- **2026-05-06 measurement** (`src/sl_ads/ablation/ablation_qualifier_loo_templates.py`, artefacts `outputs/scientific_hardening/qualifier_loo_*`):
  - Baseline micro QP across 12 attacks = 0.676.
  - Removing an *unrelated* attack template from `SBN_COND_OPINIONS`: mean ΔQP = **+0.020** (essentially no degradation; templates are weakly entangled).
  - Removing the *matching* attack template (self-drop): QP collapses to 0 (tautological), but the autre_anomalie residual class only catches **0%** of the now-orphaned windows — they are forced onto the closest neighbour template (DNS_AMP→NTP_AMP, BOTNET_CC→PORT_SCAN, SLOWLORIS→PORT_SCAN, DATA_EXFIL→PORT_SCAN, etc.). This is a **novelty-detection warning**: the current `u_raw_threshold=0.82` is too high to catch genuinely novel attacks; they masquerade as known neighbours. Reviewer-grade fix would lower the threshold (heatmap shows a plateau down to 0.30 with no false-novelty cost on synthetic data) but at the cost of a higher autre_rate on real noisy traffic.

### A6.4 — `evidence_scale = 3.0` is empirically sufficient
- **Type**: Engineering (heuristic)
- **Visibility**: Explicit (line 1518)
- **Sensitivity**: MEDIUM
- **Formal**: Multiplier on `Σ_g max(0, score(k,g) − 1/3)` before bijection.
- **What breaks**: Smaller scale ⇒ more uncertainty (`u` larger) ⇒ more `Autre_Anomalie`. Larger scale ⇒ over-confident attribution. No theoretical derivation; calibrated empirically against the synthetic catalog.
- **2026-05-06 measurement** (`src/sl_ads/ablation/ablation_sbn_param_sensitivity.py`, artefacts under `outputs/scientific_hardening/sbn_param_sensitivity*`): the published value `evidence_scale=3.0` sits inside a broad plateau where QP=0.607, DR=0.828, autre_rate=0 for `evidence_scale ∈ [1.0, 10.0]` and `u_raw_threshold ∈ [0.30, 0.99]`. The plateau breaks only at extreme corners: `evidence_scale ≤ 0.5` forces every detected window into `autre_anomalie` (autre_rate → 1.0); `u_raw_threshold ≤ 0.10` reaches the same regime by lowering the bar.

### A6.5 — `u_raw > 0.82` ⇒ residual class `Autre_Anomalie`
- **Type**: Engineering (threshold)
- **Visibility**: Explicit (line 1537)
- **Sensitivity**: MEDIUM
- **Formal**: `qual_status = autre_anomalie ⇔ u_raw > 0.82`.
- **Calibration source**: synthetic perfect signatures; the comment at lines 901–904 explicitly states "must recalibrate on real noisy data".
- **What breaks**: If real attacks generate `u` pattern not captured by synthetic catalog, the residual class either over- or under-fires.
- **2026-05-06 measurement**: see A6.4 above — the same heatmap covers both knobs. The published value `u_raw_threshold=0.82` sits inside the plateau and the system is stable for any choice in `[0.30, 0.99]` paired with the production `evidence_scale=3.0`.

### A6.6 — Markov transition matrix encodes meaningful kill-chain priors
- **Type**: Engineering (expert prior)
- **Visibility**: Explicit (lines 522–588)
- **Sensitivity**: LOW (only in `--temporal` mode, default OFF for reproducibility)
- **Formal**: `T[i,j] = P(type_t = j | type_{t−1} = i)`; row-normalised.
- **What breaks**: With temporal mode disabled (production default), this assumption is dormant.

---

## 7. `src/sl_ads/evaluate/*` and statistical helpers

### A7.1 — Single fixed `δ` (no on-test threshold tuning)
- **Type**: Engineering (anti-leakage)
- **Visibility**: Explicit (PATCH M-03, comment line 14 of `evaluate_injection.py`)
- **Sensitivity**: **CRITICAL**
- **Formal**: `δ = 0.20` (or sidecar value) used unchanged for all reported F1.
- **What breaks**: Reporting `max_δ F1(δ)` overfits to the test set (Varma & Simon 2006). The `ablation_*_sweeps.csv` files exist for sensitivity, but headline numbers come from the calibrated `δ`.

### A7.2 — Range-aware AUC buffer `L_max` defaults to median anomaly run length
- **Type**: Engineering
- **Visibility**: Explicit (`vus_metrics.py:297–303`)
- **Sensitivity**: MEDIUM
- **Formal**: `L_max = median(anomaly_run_lengths)`.
- **What breaks**: VUS values depend on `L_max`; reporting requires the realised value. The metric is meant to be range-tolerant, not range-arbitrary.

### A7.3 — Wilson CI assumes binomial, ignores autocorrelation
- **Type**: Statistical
- **Visibility**: Implicit
- **Sensitivity**: MEDIUM
- **Formal**: For TPR estimate `p̂`, Wilson CI uses `n` independent trials.
- **What breaks**: Anomaly windows are autocorrelated; effective sample size `n_eff < n`. The CI is therefore *narrower* than truth ⇒ overstated precision. The codebase has `stats/residual_correlation.py` (Newey-West-style) but does not propagate `n_eff` into the headline CIs of `axelsson_ppv.py`.
- **2026-05-06 fix**: `axelsson_ppv._bca_ci_proportion` now accepts an `n_eff` argument; `per_attack_ppv_table` computes `n_eff` from a Newey-West-style autocorrelation correction (`stats/residual_correlation.newey_west_eff_n`, lag truncation L=10) and passes it to the Wilson formula. Outputs gain `base_rate_n_eff` and `tpr_n_eff` columns so a reviewer can audit the inflation. The realised `n_eff/n` ratio on RedeRio normal windows is ~1488/12015 ≈ 12 % (i.e. CIs are ~3× wider after the correction).

### A7.4 — McNemar test assumes paired observations
- **Type**: Statistical
- **Visibility**: Explicit (`stats/mcnemar.py:9–10`)
- **Sensitivity**: LOW (only when comparing two classifiers on the same test set)
- **Formal**: discordant pair counts `n01, n10`; `χ²_corrected = (|n01 − n10| − 1)² / (n01 + n10)`.
- **What breaks**: For `n_disc < 25`, the χ² approximation is unreliable; the code falls back to exact binomial (Pembury Smith & Ruxton 2020).

### A7.5 — BCa bootstrap requires iid sampling
- **Type**: Statistical
- **Visibility**: Implicit (resampling assumes exchangeability)
- **Sensitivity**: HIGH
- **Formal**: `n_boot=2000` resamples; jackknife for acceleration.
- **What breaks**: For autocorrelated time series, naive BCa under-estimates variance. Block bootstrap (Künsch 1989) would be safer; not implemented.
- **2026-05-06 fix**: `stats/bootstrap_ci.bootstrap_bca_ci` and `paired_bootstrap_bca_ci` now accept `block_length`; when set, they sample contiguous blocks of size `block_length` (moving block bootstrap) and use delete-one-block jackknife for the BCa acceleration. `evaluate_injection.global_threshold_sweep` calls them with `block_length = median attack-episode length` (36 windows on RedeRio). Bootstrap method, resampling mode and block length are persisted in `eval_threshold_sweep.csv` for traceability. Test `test_bca_bootstrap_supports_moving_blocks` locks the API. Effect on RedeRio: F1 95 % CI widens from [0.760, 0.807] (iid) to [0.665, 0.875] (BCa-block 36) — an ~5× wider interval, properly reflecting the auto-correlated nature of windowed evaluation.

### A7.6 — Baseline thresholds are calibrated off-test
- **Type**: Engineering / Fairness
- **Visibility**: Explicit (`compare_no_sl_fair.py`, `compare_raw_baselines_fair.py`, PATCH C-01/F02 in `compare_if_fair.py`)
- **Sensitivity**: MEDIUM
- **Formal**: same-evidence no-SL thresholds are selected from train-calib residuals; raw IF / robust-z / PCA thresholds are selected from pre-split normal windows; test labels are used only for reporting.
- **What breaks**: Test-FPR matching or best-test thresholds would leak labels and inflate baselines. The old `_find_if_threshold_matching_fpr` remains documented as a leakage pattern, not a paper-facing path.

### A7.7 — `INJECTED_ATTACK_CATALOG` is the *only* source of synthetic ground truth
- **Type**: Engineering (DRY)
- **Visibility**: Explicit (PATCH-C1, `evaluate_qualify_injected.py:64–84`)
- **Sensitivity**: HIGH
- **Formal**: A single canonical list in `config.py` is read by both injector and evaluator.
- **What breaks**: Local copies that drift would silently mis-label test windows ⇒ wrong F1.

---

## 8. `src/sl_ads/preprocessing_utils.py`

### A8.1 — `0` is not a valid metric value (zero ≠ absence)
- **Type**: Data semantics
- **Visibility**: Explicit (`compute_evidence.py:153–161` comment, PATCH m-07/F25)
- **Sensitivity**: HIGH
- **Formal**: For network metrics, `value = 0` is a valid measurement (no traffic), distinct from `NaN` (no measurement). `fillna(0)` is forbidden on metric columns.
- **What breaks**: Naive `fillna(0)` would teach Prophet that absence-of-measurement looks like normal-zero traffic, biasing residuals on non-instrumented periods.

### A8.2 — Forward-fill within `NAN_FFILL_LIMIT=10` (5 min) is acceptable
- **Type**: Engineering / Data
- **Visibility**: Explicit
- **Sensitivity**: MEDIUM
- **Formal**: Trous of length `< 5 min` propagated; longer trous remain NaN.
- **What breaks**: A `>5 min` trou aligned with an attack would produce NaNs in the residual, which downstream propagate to `(p,s,n) = (NaN,…)`; the trapezoidal map handles this by not counting that timestep. Real concern: brief attacks lasting `< 5 min` could be smoothed away by ffill — but `5 min = WINDOW_SIZE` exactly, so this is window-bounded.

---

## 9. `src/sl_ads/config.py`

### A9.1 — `CONFLICT_ALPHA` matches the deployed `WINDOW_SIZE` and `W`
- **Type**: Numerical / Engineering
- **Visibility**: Explicit (recomputed at end of `config.py`, lines 2418–2427)
- **Sensitivity**: HIGH
- **Formal**: `α = (WINDOW_SIZE+W)·(2·WINDOW_SIZE+W) / (WINDOW_SIZE·2·WINDOW_SIZE)`; for `W=10, K=3`, `α = 1.495`.
- **What breaks**: Manual override of `CONFLICT_ALPHA` without recomputing breaks the hard-reset property (cf. A4.5).

### A9.2 — `FPR_TARGET_DECISION` is the operator's chosen FPR budget
- **Type**: Engineering
- **Visibility**: Explicit (`config.py:618`)
- **Sensitivity**: HIGH
- **Formal**: For RedeRio, `FPR_TARGET_DECISION = 0.001` (1‰); the auto-calibrated `δ` realises this on the calibration set.
- **What breaks**: Realised FPR on the test set may differ due to (i) calibration-vs-deploy distribution shift, (ii) surrogate-vs-deployed pipeline mismatch (cf. A1.9). Empirical FPR in test should be reported next to the targeted FPR.

### A9.3 — `RECONST_RULES` capture genuine structural relations
- **Type**: Engineering (domain-specific)
- **Visibility**: Explicit (`config.py:247–268`)
- **Sensitivity**: MEDIUM
- **Formal**: Each `(target, feature)` pair must yield positive `R²_CV` on training (otherwise mean-fallback).
- **What breaks**: A spurious correlation in the training horizon will produce a confident but unstable QR(0.5) — feed false confidence into the WBF aggregation. The mean-fallback mechanism mitigates by collapsing the metric to `R²=0`.

### A9.4 — `ACTIVE_METRICS` exhausts the relevant predictive features
- **Type**: Engineering (feature selection)
- **Visibility**: Explicit (`config.py:219–239`, RedeRio uses 12 metrics)
- **Sensitivity**: HIGH
- **Formal**: The set of metrics on which Prophet is fitted determines coverage of attack types.
- **What breaks**: Adding a new metric with poor `R²` *can* hurt performance (more dogmatic vacuous opinions in WBF); removing a metric specific to one attack family destroys the qualifier's discriminative power.

---

## 10. Cross-cutting / system-level assumptions

### A10.1 — Reproducibility under deterministic seeds
- **Type**: Engineering
- **Visibility**: Explicit (`SL_RANDOM_SEED` env, default `0`)
- **Sensitivity**: LOW
- **What breaks**: Prophet uses Stan; its variational inference is seed-dependent. Reported numbers for a specific seed; ablations sweep `5` seeds (`evaluate/run_multi_seed.py`).

### A10.2 — Pipeline invocation order matches `run_pipeline.py` profile
- **Type**: Engineering
- **Visibility**: Explicit (steps definition lines 71–110 of `run_pipeline.py`)
- **Sensitivity**: HIGH
- **Formal**: `train → evidence → inject → opinions → eval_injection → qualify_sbn → eval_qualify → ablation → compare_if → audit`. RedeRio is the only profile with `inject` and `audit`; CESNET / METR-LA / GECCO skip injection.
- **What breaks**: Running `opinions` before `evidence` or after `qualify_sbn` would either crash or operate on stale CSVs. The launcher dispatches them sequentially.

### A10.3 — Evaluation labels are derived from `injection_label` × `injection_ramp_alpha` consistently
- **Type**: Engineering
- **Visibility**: Implicit
- **Sensitivity**: HIGH
- **Formal**: A window with `α > 0` is a positive ground-truth window for evaluation; `α = 0` (outside catalog) is normal.
- **What breaks**: Some evaluators may treat `α ∈ (0, 1)` (ramp segments) the same as `α = 1` (plateau), under-penalising "weak" attacks. Stratification by `α` is recommended but not enforced.

### A10.4 — `FINAL_SYSTEM_CBF_*` column prefix is historical
- **Type**: Engineering
- **Visibility**: Explicit (PATCH TASK-44, `paths.py:96–107`)
- **Sensitivity**: LOW
- **Formal**: The prefix `FINAL_SYSTEM_CBF` is kept for backward compatibility with downstream consumers, but the actual fusion may be WBF, ABF, CBF, BCF, projected CCF, MinBF, MaxBF, or hierarchical. The sidecar `fusion_mode_at_compute_opinions.json` records the actual mode.
- **What breaks**: A reviewer expecting CBF semantics from the column name would be misled; the sidecar must be consulted.

---

## 11. Sensitivity ranking (CRITICAL items)

### 11.1 — CRITICAL items, post 2026-05-06 hardening

| # | Assumption | Failure consequence | 2026-05-06 status |
|---|---|---|---|
| A1.1 | Training span is attack-free | All thresholds biased; F1 invalid | Defensive consensus audit (`audit_train_span.py`) flags new suspect windows for manual review; not auto-pruned. |
| A1.9 | DECISION_THRESHOLD surrogate matches deployed pipeline | δ uncalibrated, FPR untrustworthy | Hard-raise on knob mismatch (`paths.validate_threshold_sidecar_config`), mode-specific fusion sidecars for WBF/ABF, realised FPR ratio reported in `eval_threshold_sweep.csv`. |
| A3.2 | Non-injected windows are `normal` | Headline FPR conflates real attacks with errors | REAL_ATTACK_CATALOG + REAL_ATTACKS outages are explicitly handled instead of silently counted as ordinary normal traffic. Final F1 reporting uses both protocols: `catalog_outages_separate` and `operator_faithful_anomaly`. |
| A3.5 | Injection invisible to threshold calibrator | δ contaminated by injected events | Hard-raise in `_validate_catalog` if any catalog event has `start <= split_date`. |
| A4.1 | `Σb + u = 1` opinion validity | All SL operations meaningless | Already enforced (renormalisation + ValueError) since v1. |
| A4.4 | Asymmetric escalation conflict is the design choice | Defines the dynamic-ageing semantics; alternative ablations exist | Code-comment + ablation hooks; no change. |
| A4.7 | CBF independence assumption violated | Over-confident inter-method fusion in legacy mode | RuntimeWarning at module import time when `INTER_METHOD_FUSION='cbf'`; strict ABF comparison done, WBF remains default on RedeRio. |
| A7.1 | No on-test threshold tuning | F1 inflated by overfitting to test | Sidecar-driven `_select_best_row` (PATCH TASK-34) + escape-hatch warning. |
| A7.7 | Catalog is single source of truth | Drift = silent label corruption | PATCH-C1 enforced; no change. |

### 11.2 — HIGH items now measured (post 2026-05-06)

| # | Assumption | Measurement artefact | Verdict |
|---|---|---|---|
| A1.2 | GPD validity above `t₀=Q90` | `_pwm_gpd_fit` test suite | PWM fallback inserted between Grimshaw and scipy. |
| A1.3 | Independent residual exceedances (declustering off) | `outputs/scientific_hardening/evt_declustering_*` | 12/17 metrics insensitive; 5 volatile metrics shift up to 25 % when declustering enabled. Justified for the bulk; documented exception. |
| A1.5 | Single global EVT threshold per metric | `outputs/scientific_hardening/regime_fpr_*`; PATCH H2 (calendar-aware EVT) implemented as audit-grade opt-in 2026-05-07 | Calendar-aware EVT shipped (`sl_ads/calendar/regime.py`, `train_models.calibrate_thresholds_per_regime_v2`, `compute_evidence` regime dispatch, sidecar A1.9 ``calendar_evt_signature``). **Default `CALENDAR_EVT_ENABLED=False`**. The 2026-05-10 values are diagnostic only because that run was incomplete; the complete rerun must refresh realised regime-FPR. Current paper stance: report realised FPR and keep α-sweep/contextual discount as exploratory future work. |
| A3.3 | Injection signatures too clean | `outputs/scientific_hardening/signature_noise_ablation.*` | Linear QP degradation 0.607→0.426 for σ=0→0.20. No collapse. |
| A6.1 | Naive Bayes group independence | `outputs/scientific_hardening/qualifier_group_correlations*` | 32/66 attack-window pairs HIGH dependence; argmax decision robust, ROC area inflated. |
| A6.3 | Conditional templates correctness | `outputs/scientific_hardening/qualifier_loo_*` | Other-template robust (Δ +0.02 on average); novelty handling weak (autre_anomalie=0% on self-drop). |
| A6.4 | `evidence_scale=3.0` heuristic | `outputs/scientific_hardening/sbn_param_sensitivity.*` | Plateau for `[1.0, 10.0]` × `[0.30, 0.99]`; published value at the centre. |
| A6.5 | `u_raw=0.82` heuristic | (same heatmap) | Plateau as above; same caveat for novel-attack handling (cf. A6.3). |
| A7.3 | Wilson CI ignores autocorrelation | `axelsson_ppv` Newey-West n_eff | Realised n_eff/n ≈ 12 % on RedeRio; CIs ~3× wider. |
| A7.5 | BCa bootstrap iid sampling | `bootstrap_ci.bootstrap_bca_ci(block_length=…)` | Block bootstrap (Künsch 1989) wired in; F1 CI widens from 0.760-0.807 (iid) to 0.665-0.875 (BCa-block 36). |

### 11.3 — Items that are *intentionally* not fixed

| Item | Reason for declining the proposed fix |
|---|---|
| `dst_port_entropy` for DNS_AMP↔NTP_AMP confusion | The aggregated `entropy_dst_port` metric is already in the dataset and in the qualifier; distinguishing port-53 from port-123 traffic requires per-port flow data that the standardised CSV does not contain. Cannot be implemented without re-aggregating raw NetFlow upstream. Documented limitation. |
| `R_init = a_edp · 2W` for NETWORK_OUTAGE_NOV17 boundary case | The bijection makes proj_atk_init invariant in the magnitude of R_init (`proj_atk_init = a_atk` for any positive c). Increasing c only slows reaction to fresh evidence — the *opposite* of the intended fix. Mathematically counterproductive. |
| `FINAL_SYSTEM_CBF` column rename to `FINAL_SYSTEM_FUSED` | Forward-compatible alias `paths.get_detection_col_fused` already exists (PATCH TASK-44). 53 callsites would have to be migrated for a purely cosmetic gain. The actual fusion mode is recorded in `fusion_mode_at_compute_opinions.json`. |
| Per-metric bootstrap block length | Empirical ACF on the headline detection score crosses 1/e at lag 31; current block length 36 is within 5 % of the empirically optimal value. Per-attack refinement matters only for individual PPV CIs and the Newey-West n_eff (max_lag=10) already captures the binary-indicator autocorrelation. |

### 11.4 — Residual open items (to action before journal submission)

| ID | Description | Estimated effort | Owner |
|---|---|---:|---|
| **TASK-12** | Multi-seed evaluation (5 seeds, mean ± std) on the headline F1/MCC/VUS table — code exists in `src/sl_ads/evaluate/run_multi_seed.py`, just needs to be run (8-10 h compute). | 1 day compute | next compute slot |
| **A1.5-followup** | _CLOSED_ 2026-05-07 (PATCH H2) — per-regime EVT thresholds (one calibration per calendar bucket) shipped as `CALENDAR_EVT_ENABLED` opt-in. Awaiting next retrain to populate the per-regime block; activation will be evaluated against a re-run of `evaluate_regime_fpr.py` (canonical partition rows). | retrain + re-audit | engineering |
| **TKDE/VLDB SOTA baselines** | Run TranAD / Anomaly Transformer / TimesNet under the same RedeRio protocol. Plan emitted at `outputs/scientific_hardening/compare_sota_tsad_plan.json`; needs GPU and external repo cloning. | ~5 days compute + integration | engineering |
| **NETWORK_OUTAGE_NOV17 boundary case** | _CLOSED for current run._ The old "cold-start" explanation was retracted. Complete-run counts: NOV17 1/3, DEC1617 188/339, REAL_DDOS 184/190; both F1 protocols are reported in `eval_f1_protocol_comparison.csv`. | paper wording only | disclosure |
| **Heavy-tailed signature noise** (A3.3 refinement) | Current Gaussian perturbation is a lower bound; a Cauchy / Student-t perturbation would probe the qualifier under a more realistic noise model. | ~1 day | future work |

These are explicitly *out of scope* of the 2026-05-06 hardening pass and are
documented here for the next iteration; nothing in §11.4 is a known
correctness defect, only an opportunity for further evidence.
