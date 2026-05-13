# PIPELINE_LOGIC.md — End-to-End Reasoning Chain

> Reconstructs the **complete causal chain** from raw network telemetry to the final
> binary anomaly decision and the multi-class qualifier output. Each step lists:
>
> - **Input data** (shape, source)
> - **Transformation** (the formula or algorithm executed)
> - **Output data** (shape, persistence)
> - **Methods used** (cross-references to `METHODS.md`)
> - **Assumptions used** (cross-references to `ASSUMPTIONS.md`)
> - **Causal justification** ("This step exists because…")
>
> The reference profile is **RedeRio**, the most complete pipeline. Other profiles
> (METR-LA, CESNET, GECCO-IoT) are strict subsets — see §10 below.

---

## 0. Glossary

| Symbol                        | Meaning                                                                                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `X_t`                         | Per-metric scalar measurement at time `t` (every 30 s for RedeRio).                                                                                                 |
| `M_p`                         | Set of Prophet-modelled metrics (12 on RedeRio: bytes, packets, flows, syn, icmp, udp, tcp, fin, entropy_src_ip, entropy_src_port, entropy_dst_port, avg_pkt_size). |
| `M_r`                         | Set of Reconstruction-modelled metric pairs (5 on RedeRio: bytes←packets, bytes←entropy_src_port, udp←flows, fin←syn, tcp←packets).                                 |
| `e_t`                         | Signed point residual `y_t − ŷ_t`.                                                                                                                                  |
| `(p,s,n)_t`                   | Instantaneous trapezoid evidence proportions for one residual; `p+s+n=1` by construction, but this is not yet a Subjective-Logic opinion.                           |
| `n_window`                    | Effective number of observed points in the current evidence window. Full RedeRio windows have `WINDOW_SIZE = 10` (= 5 min at 30 s); final partial windows may have `n_window < 10`. |
| `(P,S,N)`                     | Per-window accumulated evidence counts: `P=sum_t p_t`, `S=sum_t s_t`, `N=sum_t n_t`, so `P+S+N = n_window`.                                                          |
| `r = (P,S,N)`                 | Evidence vector supplied to the Jøsang evidence-to-opinion bijection; it becomes an opinion only after `evidence_to_opinion(r, W, a)`.                               |
| `ω = (b, u, a)`               | A multinomial opinion after bijection (`sum(b) + u = 1`).                                                                                                            |
| `R_τ`                         | State-memory accumulated evidence for a metric at window τ.                                                                                                         |
| `K_τ`                         | Conflict degree between `R_{τ−1}` and `r_τ`.                                                                                                                        |
| `λ_dyn`                       | Dynamic ageing factor; `K_eff = clip(α·K, 0, 1)` and `λ_dyn = λ_base · (1 − K_eff)` in the current `γ=1` configuration.                                               |
| `proj_atk = b_atk + a_atk·u`  | Decision variable.                                                                                                                                                  |
| `δ`                           | Auto-calibrated decision threshold.                                                                                                                                 |

---

## 1. Step `train` — Train forecasters, calibrate thresholds, compute prior

### 1.1 Inputs
- Standardized CSV `data_standardized/RedeRio.csv` (30 s cadence; 12 core
  Prophet-modelled metrics in the RedeRio profile)
- `CONFIG['split_date'] = 2025-11-09 23:59:59` partitioning train ∪ test
- `CONFIG['CALIB_SPLIT_FRACTION'] = 0.25` further splitting train into model-fit ∪ threshold-calibration

### 1.2 Sub-step 1.2.A — NaN policy and exclusions
- **Transformation**: zero-out training metric values during `TRAIN_EXCLUSIONS` (~50 maintenance ranges); apply `preprocess_metrics(df, limit_ffill=10)` (forward-fill ≤ 5 min, never `fillna(0)` on metric cols).
- **Output**: `df_train_model` (75 % of train) and `df_train_calib` (25 % of train).
- **Methods**: §1.1 of METHODS.md.
- **Assumptions**: A1.1 (training span attack-free), A8.1, A8.2 (NaN semantics).
- **Justification**: Prophet ignores NaN on `y` (Taylor & Letham 2018); `fillna(0)` on bytes would teach Prophet that downtime resembles normal-zero.

### 1.3 Sub-step 1.2.B — Fit Prophet (12 metrics)
- **Input**: per-metric series `(ds, y)` from `df_train_model` + calendar regressors.
- **Transformation**: `Prophet(growth='flat', daily_seasonality=False, weekly=True, +daily_weekday/weekend conditional, +hourly Fourier).fit()`.
- **Output**: 12 fitted Prophet models, residual vectors `e_train_p[k]`, `R²_CV` per model via `cross_validation(initial='14d', period='3d', horizon='1d')`.
- **Methods**: §1.1 of METHODS.md (Prophet).
- **Assumptions**: A1.6 (`growth='flat'`), A1.5 (`Q_*` excess invariance).
- **Justification**: Capture seasonal structure so residuals carry only the "unmodelled" component.

### 1.4 Sub-step 1.2.C — Fit Quantile Regression (5 pairs)
- **Input**: `(X = feature, y = target)` from each rule in `RECONST_RULES`.
- **Transformation**: `QuantileRegressor(q=0.5, solver='highs', alpha=0.0, fit_intercept ← rule)`. 5-fold `TimeSeriesSplit` for `R²_CV`. If `R²_CV < 0` and `allow_mean_fallback=True`, swap to `DummyRegressor(strategy='mean')` and set `R² = 0`.
- **Output**: 5 fitted models + residuals `e_train_r[k]`.
- **Methods**: §1.2 of METHODS.md.
- **Assumptions**: A1.7 (LAD breakdown + Bridgman dimensional homogeneity), A1.8 (R²_CV ≥ 0 trust).
- **Justification**: Capture *structural* relations that Prophet (univariate) cannot — e.g. `bytes/packet` ratio, `fin/syn` ratio.

### 1.5 Sub-step 1.2.D — EVT/POT threshold calibration
- **Input**: signed residuals `e_calib` from `df_train_calib` (PATCH TASK-22, leak-free).
- **Transformation** (per metric, per direction `pos`/`neg`/`both`/`sym`):
  1. `t₀ = quantile(|e_calib|, 0.90)`
  2. `excesses = (|e_calib|[|e_calib| > t₀]) − t₀`
  3. If `len(excesses) < EVT_MIN_PEAKS=50` → empirical quantile fallback (logged in `_FALLBACK_LOG`)
  4. Else MLE via Grimshaw 1D root-finding (`_grimshaw_fit`); fallback to PWM, then `scipy.genpareto.fit`; final fallback to empirical
  5. Validity check Coles §4.2: `σ̃ = σ − ξ·t₀ > 0`; if not, empirical fallback (logged)
  6. `T_susp = t₀ + (σ/ξ)((q_susp · n_total/n_peaks)^{−ξ} − 1)`, similarly for `T_atk`
  7. Safety: `T_atk ← max(T_atk, 1.10·T_susp)`
  8. `T_trapeze_base = 0.1 · T_susp`
- **Output**: per-metric `(T_susp, T_atk, T_trapeze_base)` plus directional variants for `direction='both'`.
- **Methods**: §1.4 of METHODS.md.
- **Assumptions**: A1.2 (GPD validity), A1.3 (declustering off), A1.4 (sample sufficiency), A1.5 (excess invariance).
- **Justification**: Replace ad-hoc empirical quantiles by extrapolating the residual tail — produces sharper thresholds with formal probabilistic interpretation `P(|residual| > T_atk | normal) ≤ q_atk`.

### 1.6 Sub-step 1.2.E — EDP (Empirical Dirichlet Prior)
- **Input**: training residuals `e_train` per metric, calibrated thresholds.
- **Transformation**: simulate the trapezoidal map per training window; compute marginal class proportions:
  - `a_safe = mean_τ(P_τ) / n_window`
  - `a_susp = mean_τ(S_τ) / n_window`
  - `a_atk = max(mean_τ(N_τ) / n_window, 0.005)` (floor `EMPIRICAL_PRIOR_FLOOR`)
  - renormalise to `Σa = 1`
  - For `direction='both'`: also `a_susp_pos`, `a_atk_pos`, `a_susp_neg`, `a_atk_neg` with the coarsening identity `a_susp = a_susp_pos + a_susp_neg`.
- **Output**: `models_pkg['empirical_priors'][metric_key]`.
- **Methods**: §2.12 of METHODS.md.
- **Assumptions**: A1.11 (per-metric stationary prior).
- **Justification**: The default uniform prior `(1/3, 1/3, 1/3)` over-expresses ignorance and inflates `a_atk` ⇒ false positives. Empirical Bayes (Robbins 1955; Ferguson 1973) reduces FPR by anchoring the prior to observed-normal marginal frequencies.

### 1.7 Sub-step 1.2.F — Resolve `RECONST_ATTACK_RELIABILITY`
- **Input**: training residuals + EDP.
- **Transformation**: if `CONFIG['RECONST_ATTACK_RELIABILITY']='auto'`, simulate evidence per window; count "Prophet-suspect AND Reconst-confident-safe" → `blind_rate`; `α_attack = clip(1 − blind_rate, 0.05, 1.0)`. If the config is numeric, use that value directly.
- **Output**: `models_pkg['reconst_attack_reliability']`. In the complete RedeRio run `2e12261d55a8f975`, this value is fixed from config at `1.0`, so no contextual-discount alpha is claimed as calibrated or shipped.
- **Methods**: §3.3 of METHODS.md.
- **Justification**: Reconstruction can be structurally blind to some applicative attacks (e.g. SLOWLORIS does not necessarily break bytes/packets ratio). The optional contextual discount α is kept as an ablation/future-work mechanism; the current production reference leaves Reconstruction attack evidence undiscounted (`α=1.0`).

### 1.8 Sub-step 1.2.G — Auto-calibrate `DECISION_THRESHOLD`
- **Input**: residuals `_calib_signed_residuals` (or fall-back to training residuals).
- **Transformation** via `_compute_training_proj_atk`:
  1. For each window τ, evaluate `(P_τ, S_τ, N_τ)` per metric using the trapezoidal map and calibrated thresholds.
  2. Convert each metric window to `op_leaf = evidence_to_opinion((P,S,N), W=3, a_edp[metric])`.
  3. Optionally apply uncertainty maximisation and trust-discount mode, matching `CONFIG`.
  4. Group metrics with `CONFIG['FUSION_METHOD_GROUPS']`; apply intra-method WBF and the configured Reconstruction contextual discount.
  5. Fuse method-level opinions with the requested `INTER_METHOD_FUSION` mode (`wbf` in the current RedeRio run).
  6. `proj_atk_τ = op_fused.b[2] + op_fused.a[2] · op_fused.u`.
  7. `δ = quantile(proj_atk_τ, 1 − FPR_TARGET_DECISION)` with `FPR_TARGET_DECISION = 0.001`.
  8. Sparse-distribution guards (cf. METHODS §3.2) for cases where most `proj_atk ≈ 0`.
- **Output**: `models_pkg['_decision_threshold']`, `models_pkg['_decision_variable'] = 'proj_atk'`, mode-specific `models_pkg['_decision_threshold_by_fusion_mode']`, plus JSON sidecars `<model>_threshold*.json` with the deployment configuration alongside δ (PATCH TASK-45/TASK-55).
- **Methods**: §3.1 + §3.2 of METHODS.md.
- **Assumptions**: A1.9 (surrogate vs deployment match), A1.10 (`proj_atk` smooth), A7.1 (no on-test tuning).
- **Justification**: Pre-calibrate δ on a hold-out span so that headline F1 is reported at a *fixed* operating point — never tuned on the test set. The calibration now replays the deployed grouping and inter-method fusion, but remains an instantaneous training-span surrogate because it does not replay temporal ageing over the full test stream. The sidecar caveat documents that deployment-configuration drift invalidates δ.

### 1.9 Persisted artefacts
- `trained_models_<VERSION>.pkl` (joblib): contains every Prophet/QR model, all thresholds, EDP, trust scores, decision threshold, decision variable, `_meta_split_date`.
- `trained_models_<VERSION>_threshold.json` (sidecar): δ, decision variable, FPR target, calibration strategy, fallback log, deployed-configuration snapshot.
- `trained_models_<VERSION>_fallbacks.json`: full event log of EVT and reconstruction fallbacks per metric.

---

## 2. Step `evidence` — Per-window evidence triplets

### 2.1 Inputs
- `df = pd.read_csv(CONFIG['file_path'])` (full dataset)
- `models_pkg` (loaded from PKL)
- `df_test = df[df['ds'] > split_date]` (the inference span)

### 2.2 Pre-processing
- Same `preprocess_metrics(df, limit_ffill=10)` as training (A2.1).
- Anti-leak verification: `models_pkg['_meta_split_date'] == CONFIG['split_date']` else `return` (A2.2).

### 2.3 Per-window loop (size `WINDOW_SIZE = 10`)
For each batch `i:i+10` of `df_test`:

For each metric `k` (Prophet or Reconstruction):
1. **Predict**:
   - Prophet: `ŷ = model.predict(batch)['yhat']`; capture interval width `iw = mean(yhat_upper − yhat_lower)`.
   - Reconstruction: `ŷ = model.predict(batch[[feature]])`.
2. **Residual**: `e_t = y_t − ŷ_t` for each timestep.
3. **Per-step `rmse_win`** for use in `c3_online_rmse`: `rmse_win = sqrt(mean(e²))`.
4. **Per-step trapezoidal map**:
   ```
   for j in 1..n_window:
       (p_j, s_j, n_j) ← compute_instantaneous_evidence(
           e_{t_j},
           t_susp[k], t_atk[k], t_trapeze_base[k],
           direction[k]
       )
   ```
   For `direction='both'`, also accumulate `S_pos, N_pos, S_neg, N_neg`.
5. **Window aggregation**:
   ```
   P_k = Σ p_j ;  S_k = Σ s_j ;  N_k = Σ n_j   →   P_k + S_k + N_k = n_window
   ```

### 2.4 Outputs
- `evidence_<VERSION>.csv`: one row per window with columns `<clean_key>_P`, `_S`, `_N`, `_iw`, `_rmse`, plus `_S_pos/_N_pos/_S_neg/_N_neg` for `direction='both'`.
- `metadata_<VERSION>.csv`: per-metric metadata (`type`, `r2_weight`, `threshold_suspect`, `threshold_attack`, `direction`, `kurtosis`, `cv`).
- `raw_data_<VERSION>.csv`: per-step `(timestamp, metric_key, real, pred, abs_error)` for downstream plotting.

### 2.5 Methods / assumptions / justification
- **Methods**: §1.3 (residuals), §2.4 (trapezoidal map).
- **Assumptions**: A2.1, A2.2, A2.3 (window invariant), A2.4 (continuity), A2.5 (direction tag).
- **Justification**: This step *only* converts continuous residuals into per-window evidence triplets; it neither aggregates across metrics nor adds dynamics. This separation is required because the trapezoidal map is *per-metric* and *per-direction*, while the SL fusion stack (next step) is *cross-metric*. Persisting `(P,S,N)` to disk also enables reproducibility and ablations without re-running Prophet.

---

## 3. Step `inject` — Synthetic anomaly injection (RedeRio only)

### 3.1 Inputs
- `evidence_<VERSION>.csv`
- `CONFIG['INJECTED_ATTACK_CATALOG']` (13 attacks, deterministic, calendar-anchored)

### 3.2 Per-attack transformation
For each catalog entry `atk = {name, start, end, intensity, signature}`:
1. Pick the windows τ ∈ `[atk.start, atk.end]`.
2. `n_windows = len(windows)`; build a trapezoidal ramp profile `α(τ) ∈ [0, 1]` with `ramp_frac` rise/fall.
3. For each metric `m` listed in the signature:
   - Read the raw weights `(ev_normal, ev_suspect, ev_attack)`.
   - Normalise so they sum to `n_window` (preserves the bijection invariant exactly).
   - Apply per-window:
     ```
     P_τ ← (1 − α(τ)) · n_window + α(τ) · P_norm
     S_τ ← α(τ) · S_norm
     N_τ ← α(τ) · N_norm
     ```
   - Write `injection_label = atk.name`, `injection_ramp_alpha = α(τ)` to ground-truth columns.

### 3.3 Outputs
- `evidence_<VERSION>_attacks.csv` consumed by `sl_ads.core.opinions_pipeline`.

### 3.4 Methods / assumptions / justification
- **Methods**: §5 of METHODS.md.
- **Assumptions**: A3.1 (disjoint catalog), A3.2 (non-injected = normal), A3.3 (signature representativity), A3.4 (invariant preserved), A3.5 (invisible to threshold calibrator since events lie in test span only).
- **Justification**: The 13 controlled catalog episodes are synthetic and provide perfect timing/type labels. RedeRio also contains observed incidents (`REAL_DDOS` and network outages), which are handled in the evaluation protocols rather than injected here. Injecting the catalog at the *evidence* level (rather than raw or residual) bypasses the Prophet/QR pipeline so the same forecasters are tested on the same residuals — only the bottom-up evidence triplets change. This isolates the SL stack as the system under test, but it is not a raw-traffic realism claim.

---

## 4. Step `opinions` — Three-level SL fusion

### 4.1 Inputs
- `evidence_<VERSION>_attacks.csv` (or `evidence_<VERSION>.csv` if no injection)
- `metadata_<VERSION>.csv`
- `models_pkg` (for EDP, trust scores)
- `δ` (loaded from sidecar via `paths.get_decision_threshold`)

### 4.2 Resampling
Compute `_target_freq = freq_data × WINDOW_SIZE = 30 s × 10 = 5 min`. Resample evidence by `sum`, `origin='epoch'`, `closed='left'`. RedeRio: alignment only (CSV already at 5-min cadence); CESNET (10 min × 1 = 10 min): same.

### 4.3 State-memory initialisation
For each metric `k`:
- If EDP available: `R_τ=0[k] = a_edp · W` (so `proj_atk_init = a_atk_edp` exactly via `D = 2W`).
- Else: `R_τ=0[k] = (0, 0, 0)` (vacuous).

### 4.4 Per-window loop

**Inputs at window τ**: row of resampled evidence; per-metric current `r_τ[k] = (P, S, N)`.

#### 4.4.A — Level 1: Conflict-Aware Ageing (per metric)

For each metric `k`:
```
K_τ[k] = compute_asymmetric_escalation_conflict(R_{τ-1}[k], r_τ[k], W=3)
K_eff = clip(α · K, 0, 1)               # α = CONFLICT_ALPHA = 1.495
λ_dyn[k] = λ_base · (1 − K_eff)         # γ = 1 (linear)
R_τ[k] = λ_dyn[k] · R_{τ-1}[k] + r_τ[k]
```

Then bijection (Def. 3.9):
```
ω_temp[k] = evidence_to_opinion(R_τ[k], W=3, a=a_edp[k])
```

If `UNCERTAINTY_MAXIMIZATION=True`: `ω_temp[k] ← uncertainty_maximized(ω_temp[k])`.

Optional `c3_weight[k]`:
- `'uniform'` → `1.0`
- `'r2_static'` → `r²` of the metric (legacy)
- `'prophet_interval'` → `1 / iw_τ`
- `'online_rmse'` → `1 / safe_rmse_state[k]` (gated on per-metric `prev_proj_atk[k] < δ/2`)

If `WBF_WEIGHT_MODE='trust_discount'` (deprecated, ablation only): apply `apply_trust_discount(ω_temp[k], t = trust_score[k])`; set `weight = 1`.

**Methods**: §2.4 (trapezoid → bijection input), §2.5 (Conflict-Aware Ageing), §2.2 (bijection), §2.9 (trust discount), §2.3 (uncertainty maximisation).
**Assumptions**: A4.4, A4.5, A5.1, A5.2.

#### 4.4.B — Level 2: Intra-method WBF

```
ω_prophet = fusion_wbf_n_sources([ω_temp[k] : k ∈ M_p],
                                  external_weights=weights_prophet,
                                  W=3)
ω_reconst = fusion_wbf_n_sources([ω_temp[k] : k ∈ M_r],
                                  external_weights=weights_reconst,
                                  W=3)
```
This averages evidence vectors weighted by `ext_w · (1−u)` (Eq. 12.27 evidence-space), then bijection-back.

**Methods**: §2.6.
**Assumptions**: A4.3, A4.6.

#### 4.4.C — Level 3: Inter-method fusion

Optional contextual discount on Reconstruction (only on the `attack` hypothesis):
```
if CD_ALPHA_ATTACK < 1:
    ω_reconst' = apply_contextual_discount(ω_reconst, [1, 1, CD_ALPHA_ATTACK])
else:
    ω_reconst' = ω_reconst
```

Then dispatch on `INTER_METHOD_FUSION`:

| mode | formula |
|---|---|
| `wbf` (default) | `ω_final = fusion_wbf_n_sources([ω_prophet, ω_reconst'], external_weights=None, W=3)` (confidence-weighted) |
| `abf` | averaging fusion over method-level opinions; implemented for dependent-source ablation, not default on RedeRio after strict recalibration |
| `hierarchical` | equal method-level evidence averaging, retained for compatibility with older hierarchical experiments |
| `cbf` (legacy) | optional `boost_opinion_evidence` if `BALANCE_RATIO ≠ 1`, then `fusion_cbf(ω_prophet, ω_reconst')` |
| `bcf` | belief-constraint/Dempster-style fusion; conflict-pathology stress test |
| `ccf` | projected consensus/compromise-inspired experimental fusion |
| `minbf` | conservative lower-envelope fusion stress test |
| `maxbf` | upper-envelope fusion stress test |

**Methods**: §2.6 (WBF), §2.8 (CBF), §2.8a (operator dispatch), §2.10 (contextual discount), §2.11 (boost).
**Assumptions**: A4.7 (CBF independence violated by construction), A4.8 (BALANCE_RATIO heuristic), A4.10 (contextual discount α-vector).

#### 4.4.D — Decision

```
proj_atk_τ = ω_final.b[2] + ω_final.a[2] · ω_final.u
D_τ = (proj_atk_τ ≥ δ)
```

### 4.5 Outputs
- `detection_results_INJECTED.csv` (or `detection_results.csv` if no injection): one row per 5-min window with, for each metric and the system-level fused opinion:
  - Beliefs: `*_b_safe`, `*_b_susp`, `*_b_atk`
  - Uncertainty: `*_u`
  - Projected probabilities: `*_proj_safe`, `*_proj_susp`, `*_proj_atk`
  - Per-metric ageing diagnostics: `*_conflict_K`, `*_lambda_dyn`
  - EDP base rates: `*_a_safe`, `*_a_susp`, `*_a_atk`
  - 5-state directional projections (for `direction='both'`): `*_dir_pos_proj_*`, `*_dir_neg_proj_*`
- Per-metric PNG plots `graph_<metric>.png` (top: opinion components; bottom: real vs predicted).
- Sidecar `fusion_mode_at_compute_opinions.json` recording the actual inter-method fusion mode used (PATCH TASK-44).

### 4.6 Causal justification
Three levels because the system has **three sources of uncertainty** that must be combined **in this order**:

- **Level 1 (Ageing)** because anomaly evidence is *not memoryless*: a brief spike alone is not enough to declare an attack, but persistent evidence is. The conflict-aware modulation provides asymmetric reset on escalation (calm→alarm) so transitions are not smoothed out.
- **Level 2 (Intra-method WBF)** because Prophet has 12 metrics with heterogeneous reliability; rather than treat them as 12 independent CBF accumulators (which would double-count correlated bytes/packets), the confidence-weighted average produces a single Prophet-group opinion. Same for the 5 Reconstruction pairs.
- **Level 3 (Inter-method)** because Prophet and Reconstruction are *not independent* (same raw windows in input). CBF is the canonical SL fusion under independence (Theorem 12.2); when independence is violated, WBF (the default) avoids double-counting while retaining confidence weighting. ABF is implemented because it is theoretically plausible for dependent methods, but the strict 2026-05-07 per-mode recalibration did not justify replacing WBF on RedeRio. The contextual discount on `attack` reflects the *known structural blindness* of Reconstruction to applicative attacks (SLOWLORIS, low-volume DoS).

---

## 5. Step `eval_injection` — Detection metrics at fixed δ

### 5.1 Inputs
- `detection_results_INJECTED.csv`
- `INJECTED_ATTACK_CATALOG` (single source of truth)
- `δ` from sidecar

### 5.2 Transformation
1. Construct `y_true(τ)`: `1` if window τ ∈ ⋃ catalog windows, else `0`.
2. Construct `y_pred(τ) = (proj_atk_τ ≥ δ)`.
3. Compute confusion matrix → `precision, recall, F1, F_β=2, FAR, MCC` at fixed δ.
4. Compute threshold-sweep across `δ ∈ [0.01, 0.99]` (for ROC/PR curves only).
5. Compute range-aware metrics: `R-AUC-ROC(L)`, `R-AUC-PR(L)`, `VUS-ROC`, `VUS-PR` with `L_max = median(anomaly_run_lengths)`.
6. Compute Wilson CI on FAR (closed-form; binomial assumption — see A7.3).

### 5.3 Outputs
- `eval_injection_<base>_<ts>.csv`: per-attack and global metrics.
- ROC/PR curves PNG.

### 5.4 Methods / assumptions / justification
- **Methods**: §4.1 (point metrics), §4.2 (VUS), §4.4 (Wilson, BCa).
- **Assumptions**: A7.1 (fixed δ, no test tuning), A7.2 (`L_max` reporting), A7.3 (Wilson ignores autocorrelation), A7.7 (catalog single-source).
- **Justification**: Headline F1 must be reported at the calibrated δ to be trustworthy; threshold sweeps and VUS are reported as supplementary.

---

## 6. Step `qualify_sbn` — Cause attribution

### 6.1 Inputs
- `detection_results_INJECTED.csv` (with all per-metric `*_proj_*` columns)
- `CONFIG['QUALIFY_GROUP_SOURCES']`: groups of metrics → a meta-feature
- `CONFIG['SBN_COND_OPINIONS']`: 13 attack types × ~10 groups × 3 states = ~390 expert-elicited probabilities

### 6.2 Per-window L1 → L6 cascade

#### 6.2.1 — L1: Gate
```
gate_open(τ) = (proj_atk_τ ≥ δ)
```
If closed, emit `qual_status = 'normal'`, skip L2–L6 for that window.

#### 6.2.2 — L2: Group projection
For each group `g`:
```
P^g_s(τ) = geomean_{m ∈ g} P^m_s(τ),  s ∈ {Safe, Susp, Anom}
P^g_·(τ) ← P^g_·(τ) / Σ_s P^g_s(τ)   (renormalisation)
```

#### 6.2.3 — L3: Template score
For each attack `k`, group `g`:
```
score(k, g, τ) = Σ_s P^obs_g(τ)_s · c^{k|g}(s)        (absolute mode)
              [or Σ_s (P^obs_g(τ)_s − mean_g(τ)_s) · c^{k|g}(s)  (contrastive mode)]
```

#### 6.2.4 — L4: Evidence aggregation + bijection
```
e(k, τ) = Σ_g max(0, score(k,g,τ) − 1/3) · evidence_scale(=3.0)
```
Then bijection (Def. 3.9) on `(e(k_1), …, e(k_K))` with `W = K = |Θ_qual_named|`:
```
b(k, τ) = e(k, τ) / (Σ_k e(k, τ) + W)
u(τ)   = W / (Σ_k e(k, τ) + W)
```

Diagnostic scalar `novelty_lr(τ) = 1 / (max(L) / mean(L))` (concentration of the likelihood vector).

#### 6.2.5 — L5: Temporal smoothing (optional, OFF by default)
If `--temporal`:
```
b_prev(τ) ← T^T · b_prev(τ−1)       (Markov transition)
ω_prev_disc(τ) = apply_trust_discount(ω_prev(τ), λ^Δt)
ω(τ) = _wbf_two(ω_curr(τ), 1 − w_temp,
                 ω_prev_disc(τ), w_temp)        (w_temp = 0.30 default)
```

#### 6.2.6 — L6: Uncertainty maximisation (optional, ON by default)
If `apply_um`: `ω(τ) ← uncertainty_maximized(ω(τ))`, with `Autre_Anomalie` excluded from `K`.

### 6.3 Per-window outputs
```
top1(τ) = argmax_k b(τ)[k]    over named types only
qual_status(τ) =
    'normal'            if gate_closed
    'autre_anomalie'    if u_raw(τ) > 0.82
    'qualified'         else
top1_proj(τ) = b_top + u(τ) · 1/K   (uniform a inside Θ_qual_named)
```

### 6.4 Outputs
- `qualif_types_sbn.csv`: timestamps, gate_open, top1_type, top1_b, top1_proj, qual_status, u_sbn, novelty_lr, b_sbn_*[type], u_sbn_raw.

### 6.5 Methods / assumptions / justification
- **Methods**: §2.13 of METHODS.md (full SBN proxy stack).
- **Assumptions**: A6.1 (Naive Bayes), A6.2 (geomean pooling), A6.3 (expert priors), A6.4 (`evidence_scale=3` heuristic), A6.5 (residual-class threshold).
- **Justification**: A binary alarm (`gate_open`) is operationally insufficient — operators need a *reason* (UDP flood vs SYN flood vs SLOWLORIS). The qualifier converts each attack hypothesis into a sub-opinion via expert templates; the residual class `Autre_Anomalie` captures novelty.

---

## 7. Step `eval_qualify` — Qualifier metrics

### 7.1 Inputs
- `qualif_types_sbn.csv`
- `INJECTED_ATTACK_CATALOG` (with `expected` per attack)

### 7.2 Per-attack metrics
For each catalog event `atk`:
- `n_total = #{windows ∈ [atk.start, atk.end]}`
- `n_detected = #{gate_open ∧ τ ∈ [atk.start, atk.end]}` ⇒ DR
- `n_correct = #{top1 == atk.expected ∧ qual_status == 'qualified' ∧ τ ∈ ...}`
- `QP = n_correct / n_qualified` (guard: skip `qual_status='no_groups'`)
- `F1 = 2·DR·QP / (DR+QP)`, `F2 = (1+4)·DR·QP / (4·DR+QP)`
- `TTQ = first_correct_τ.start − atk.start` (in minutes)

### 7.3 Global metrics on classes
- Tri-class partition (PATCH m-04/F21): `in_attack`, `in_outage`, `in_normal`
- TP/FN bases use `in_attack`; FAR uses `in_normal` only (NETWORK_OUTAGE excluded)
- Novelty AUC-ROC: `roc_auc_score(known=0/novel=1, novelty_lr)` on `qual_status ≠ 'no_groups'` only (PATCH m-03/F20).

### 7.4 Outputs
- `eval_qualify_<base>_<ts>.csv`
- `eval_qualify_summary_<base>_<ts>.json` with macro/micro DR, QP, F1, F2 plus global detection block.

### 7.5 Methods / assumptions / justification
- **Methods**: §4.1, §4.2.
- **Assumptions**: A7.1 (fixed δ), A10.3 (label semantics), PATCH m-03 for novelty (no Youden tuning on test).

---

## 8. Step `ablation` — Component isolation

### 8.1 Sweeps (RedeRio)
Per-run config variations launched against the same trained models / evidence:

- `lambda ∈ {0, 0.50, 0.85, 0.99}` (ageing memory)
- `conflict_aware ∈ {True, False}` (Conflict-Aware ON/OFF)
- `use_prophet, use_reconst ∈ {True, False}` (component isolation)
- `wbf_weight_mode ∈ {uniform, r2_static, trust_discount, prophet_interval, online_rmse}`
- `inter_method_fusion ∈ {wbf, abf, cbf, bcf, ccf, minbf, maxbf, hierarchical}`
- `cd_alpha_attack ∈ {0.05, 0.10, 0.20, 0.50, 1.0}`
- `adaptive_base_rate ∈ {True, False}` (EDP ON/OFF)
- `sl_param_k ∈ {2, 3, 4}` (W sensitivity)
- `nan_ffill_limit ∈ {0, 5, 10, 20, 30}` (data hygiene sensitivity)

For each variant the entire opinions step is re-run; the headline threshold
remains the auto-calibrated δ from training.

### 8.2 Outputs
- `ablation_summary.csv` (one headline row per run at calibrated δ)
- `ablation_threshold_sensitivity.csv` (alternative thresholds, sensitivity only)
- `ablation_all_sweeps.csv` (full δ-sweep emitted by the active harness)
- `ablation_f1_curves.png`, `ablation_comparison.png`, `ablation_bar_comparison.png`

### 8.3 Methods / assumptions / justification
- Tests the marginal contribution of each design knob; permits attribution of headline F1 gain.

---

## 9. Step `compare_if` and add-on baseline comparisons

### 9.1 Procedure
1. The pipeline step `compare_if` runs the legacy raw IsolationForest agreement comparison. In the current RedeRio artifact it evaluates against the raw CSV `label` pseudo-labels, while the SL row comes from the injected detection CSV; treat it as a diagnostic only, not as a paper-facing attack-detection comparison.
2. After a complete run, `compare_no_sl_fair.py` compares the same evidence with and without the Subjective Logic layer. It calibrates non-SL thresholds on train-calib residuals and evaluates once on the test span.
3. In PowerShell, set `$env:SL_FORCE_NONINJECTED_OPINIONS = "1"` and `$env:SL_SKIP_OPINION_PLOTS = "1"`, then run `python -m sl_ads.core.opinions_pipeline` to write `opinions_non_injected/detection_results_RAW.csv`, a raw-only SL score that does not overwrite `detection_results_INJECTED.csv`.
4. `compare_raw_baselines_fair.py` trains IF, robust-z, and PCA baselines directly on raw metrics using pre-split normal rows. It requires the non-injected SL score and excludes synthetic injection windows because the catalog attacks are injected at evidence level, not raw-traffic level.
5. All paper-facing paired tests compare fixed, pre-calibrated operating points; test labels are not used to choose thresholds.

### 9.2 Outputs
- `evaluation_if_fair/fair_if_vs_sl_summary.csv` for the legacy IF agreement comparison.
- `evaluation_no_sl_fair/no_sl_fair_summary.csv` and `no_sl_fair_paired_vs_sl.csv` for the direct SL-vs-no-SL answer.
- `opinions_non_injected/detection_results_RAW.csv` for the raw-only SL score.
- `evaluation_raw_baselines/raw_baselines_summary.csv` and `raw_baselines_paired_vs_sl.csv` for raw-data baselines.

### 9.3 Justification
The comparison layer separates three questions that reviewers otherwise
conflate: same ADS with vs without SL, raw statistical anomaly agreement, and
real-event raw-baseline detection. Earlier versions tuned IF on the test set
(PATCH C-01/F02); paper-facing comparisons now calibrate thresholds only on
train/pre-split data.

---

## 10. Step `audit` — Episode-level integrity check (RedeRio only)

### 10.1 Procedure
1. Group contiguous `gate_open=True` windows with `gap_min=15 min` → episodes.
2. Match each episode to `INJECTED_ATTACK_CATALOG ∪ REAL_ATTACKS` via IoU.
3. Classify: `KNOWN` (matched), `UNKNOWN` (novel), `FALSE_POSITIVE`.
4. Compute Wilson CI on FAR.
5. Per-episode: duration, peak `proj_atk`, novelty metrics (`u_sbn`, novelty_entropy, novelty_lr), top-1 type, pre-DDoS activity flag (Benson 2010 IMC).

### 10.2 Outputs
- `audit_episodes.csv`, `audit_summary.json`.
- Optional LaTeX export (`--latex`).

### 10.3 Justification
Cross-validates the detector at the episode (not window) level: ensures the system reports coherent attacks rather than scattered alarms, and surfaces unknown anomalies for manual inspection.

---

## 11. Profile differences

| profile | injection? | qualifier? | eval_qualify? | audit? | ablation type | calibration source |
|---|---|---|---|---|---|---|
| **RedeRio** | yes (built-in catalog) | yes | yes (`evaluate_qualify_sbn`) | yes | `run_ablation` | hold-out calib split |
| **METR-LA** | no | yes | yes (`evaluate_qualify_injected`) | no | `run_ablation_labeled` | labels in dataset |
| **GECCO-IoT** | no | yes | no | no | `run_ablation_labeled` | labels in dataset |
| **CESNET-TimeSeries24** | no | yes | no | no | `run_ablation_labeled` | labels in dataset |

`run_pipeline.py` dispatches the appropriate profile from `_PIPELINE_BY_DATASET`. The non-RedeRio profiles skip the `inject` step because they either ship ground-truth labels (METR-LA) or use a labels-or-pseudo-labels evaluation path (GECCO-IoT, CESNET-TimeSeries24).

---

## 12. Conceptual data dependencies (raw → decision)

```
raw CSV ──ffill── y_t  ──Prophet/QR── ŷ_t
                          │
                          ▼
                       e_t = y_t − ŷ_t  ─┐
                                         ├── EVT/POT  ⇒ T_susp, T_atk
                                         ├── EDP        ⇒ a_edp
                                         └── trapezoid ⇒ (p,s,n)_t
                                                          │
                                                  Σ_window
                                                          ▼
                                              (P, S, N) per metric
                                                          │
                                              [Optional: catalog injection]
                                                          ▼
              ┌──────────── Conflict-Aware Ageing (Eq. 16.5 + α·K) ────────────┐
              │                                                                │
              ▼                                                                │
    R_τ[k] = λ_dyn · R_{τ−1}[k] + r_τ[k]   ─── bijection (Def. 3.9, W=3) ──→ ω_temp[k]
                                                                                │
              ┌── ω_temp[k ∈ M_p] ─── WBF N-ary ──→ ω_prophet ───┐              │
              │                                                  │              │
              └── ω_temp[k ∈ M_r] ─── WBF N-ary ──→ ω_reconst ──┘              │
                                                                                │
                                ω_reconst' = contextual_discount(ω_reconst,    │
                                              [1, 1, CD_ALPHA_ATTACK])         │
                                                                                │
                  WBF default / ABF / hierarchical / CBF / BCF / CCF / min/max  │
                  on (ω_prophet, ω_reconst')   ──→  ω_final                    │
                                                       │                        │
                                                       ▼                        │
                              proj_atk_τ = b_atk + a_atk · u  ──── (≥ δ?) ─→ D_τ
                                                       │
                                                       ▼
                                            qualify_sbn (L1..L6)
                                                       │
                                                       ▼
                                  (top1 attack type, novelty flag, qual_status)
```

The graph above is the **complete causal chain**. There are no hidden side-channels: every quantity reaching the final decision is one of the labelled arrows.

---

## 13. Audit checklist (load-bearing claims)

A reviewer auditing publication-grade F1 numbers must verify:

1. **A1.1**: training span `df[df['ds'] ≤ split_date]` is attack-free (manual inspection of `TRAIN_EXCLUSIONS`).
2. **A1.9**: deployment configuration matches `<model>_threshold.json`; otherwise δ is invalid.
3. **A3.2**: non-injected windows in test span do not contain real attacks (cross-check with `REAL_ATTACK_CATALOG`).
4. **A3.5**: `INJECTED_ATTACK_CATALOG` events all start after `split_date`.
5. **A4.1**: `Σb + u = 1` invariant holds at every persisted opinion (sample-test on output CSV).
6. **A7.1**: headline F1 is computed at fixed δ from sidecar — not at `argmax_δ`.
7. **A7.7**: catalog read by injector and by evaluator is the same Python list (no local copies).
8. **PATCH TASK-44**: the inter-method fusion mode recorded in `fusion_mode_at_compute_opinions.json` matches the configuration claimed in the paper.

If any of these fails, the reported F1 / FPR / DR / QP cannot be trusted, regardless of the SL-formal correctness of the operators themselves.
