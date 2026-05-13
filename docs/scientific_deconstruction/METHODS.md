# METHODS.md — Formal Inventory of Methods

> Subjective-Logic Anomaly Detection System (SL-ADS), branch `main`, 2026-05-04.
> Scope: minimal subgraph reachable from `run_pipeline.py` for the **RedeRio** profile.

The pipeline is a strict composition of five method families:

1. **Time-series forecasting / reconstruction** (Prophet + Quantile Regression)
2. **Extreme Value Theory** for threshold calibration (Peaks-Over-Threshold + Generalized Pareto)
3. **Subjective-Logic** machinery (bijection, ageing, fusion, discounting, projected probability)
4. **Heuristic / engineering** layers (trapezoidal evidence map, balance-ratio rebalancing, conflict-aware ageing, contextual discount)
5. **Evaluation / statistical inference** (Wilson CI, Wald CI, BCa bootstrap, McNemar χ², range-aware VUS)

The conventions used throughout this document:

- `b ∈ R³_{≥0}` — belief vector over `Θ = {Safe, Suspect, Attack}`,
- `u ∈ [0,1]` — uncertainty,
- `a ∈ R³_{≥0}` — base rate (Σ a = 1),
- `Σ b + u = 1` always (Jøsang 2016 Def. 3.1),
- `r ∈ R³_{≥0}` — evidence vector with `r_safe + r_susp + r_atk = n_window`,
- `W = K = |Θ| = 3` — non-informative prior weight (Jøsang 2016 §3.5.2),
- `proj_atk = b_atk + a_atk · u` — projected probability (Jøsang 2016 Eq. 3.23),
- `δ` — auto-calibrated decision threshold on `proj_atk`.

---

## 0. Pipeline Steps (RedeRio profile, `run_pipeline.py`)

| #   | Step name        | Module                                                                                                                                                | Purpose                                                                                                                                                                                 |
| --- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `train`          | `sl_ads.train.train_models`                                                                                                                           | Fit Prophet + QR(0.5), calibrate EVT thresholds, compute EDP, resolve `RECONST_ATTACK_RELIABILITY` (fixed at `1.0` in the current RedeRio run) and auto-calibrate `DECISION_THRESHOLD`. |
| 2   | `evidence`       | `sl_ads.train.compute_evidence`                                                                                                                       | Per-window forecast/reconstruction → signed residuals → trapezoidal map → evidence triplets `(P,S,N)`.                                                                                  |
| 3   | `inject`         | `sl_ads.inject.evidence_level`                                                                                                                        | Overwrite evidence triplets in the test span using the deterministic `INJECTED_ATTACK_CATALOG`.                                                                                         |
| 4   | `opinions`       | `sl_ads.core.opinions_pipeline`                                                                                                                       | 3-level SL fusion (ageing -> intra-method WBF by method group -> inter-method WBF/ABF/CBF/BCF/projected-CCF/MinBF/MaxBF/hierarchical) -> `detection_results_INJECTED.csv` on injected RedeRio runs, or `detection_results.csv` without injection. |
| 5   | `eval_injection` | `sl_ads.evaluate.evaluate_injection`                                                                                                                  | F1 / FPR / coverage at fixed `δ`, plus threshold sweeps.                                                                                                                                |
| 6   | `qualify_sbn`    | `sl_ads.qualify.sbn_qualifier`                                                                                                                        | Per-window cause attribution over 13 attack types via SL-template matching.                                                                                                             |
| 7   | `eval_qualify`   | RedeRio: `sl_ads.evaluate.evaluate_qualify_sbn`; METR-LA: `sl_ads.evaluate.evaluate_qualify_injected`; GECCO-IoT / CESNET-TimeSeries24: step skipped. | Per-attack DR / QP / F1 / F2 / TTQ; novelty AUC-ROC.                                                                                                                                    |
| 8   | `ablation`       | `sl_ads.ablation.run_ablation`                                                                                                                        | Component sweep (λ, weights, fusion, EDP).                                                                                                                                              |
| 9   | `compare_if`     | `sl_ads.compare.compare_if_fair`                                                                                                                      | Legacy raw IF pseudo-label agreement diagnostic; paper-facing add-ons are `compare_no_sl_fair.py` and `compare_raw_baselines_fair.py`.                                                   |
| 10  | `audit`          | `sl_ads.audit.audit_full_dataset`                                                                                                                     | Episode grouping, IoU matching, novelty metrics, Wilson CI on FAR.                                                                                                                      |

---

## 1. Time-Series Forecasting and Reconstruction

### 1.1 Prophet (per-metric univariate forecaster)

| field | value |
|---|---|
| **Name** | Prophet additive decomposition |
| **Category** | Time-series forecasting |
| **Location** | `src/sl_ads/train/train_models.py::train_models` (training); `src/sl_ads/train/compute_evidence.py::compute_evidence` (inference) |
| **Inputs** | per-metric scalar series `y_t`, time index `ds`, calendar regressors `on_weekday`, `on_weekend`, holidays from `CONFIG['HOLIDAYS_LIST']` |
| **Outputs** | point forecast `ŷ_t = yhat`, lower/upper bounds (`yhat_lower`, `yhat_upper`), interval width `iw_t = yhat_upper − yhat_lower` |
| **Mathematical form** | `y_t = g(t) + s(t) + h(t) + ε_t`, with `g(t)` constant (`growth='flat'`), `s(t)` weekly + conditional daily-weekday + conditional daily-weekend + hourly seasonalities, `h(t)` from `holidays_df`. Mode `additive` for metrics in `SEASONALITY_ADDITIVE`, else `multiplicative`. |
| **Hyperparameters** | `interval_width=0.95`, `changepoint_prior_scale=0.05`, `holidays_prior_scale=10`, `seasonality_prior_scale=10`, `weekly_seasonality=True`, daily Fourier order = 12, hourly Fourier order = 5. |
| **R² estimation** | Time-series rolling-origin cross-validation via `prophet.diagnostics.cross_validation(initial='14 days', period='3 days', horizon='1 day')`, then `R²_CV = 1 − MSE/var(y)`. Falls back to in-sample R² with `UserWarning` if CV fails. |
| **Residual** | `e_t = y_t − ŷ_t` (signed, used downstream by trapezoid map) |
| **Dependencies** | Stan back-end (cmdstanpy); **stationarity assumption is partial**: `growth='flat'` enforces no long-term trend, valid for 4-week training horizon on a fixed-capacity backbone. |
| **Downstream impact** | Residual feeds (i) trapezoidal evidence map, (ii) EVT threshold calibration, (iii) EDP estimation, and optional C3 weighting (`prophet_interval` uses `1/iw`; `online_rmse` uses per-window RMSE). Current RedeRio reference uses `C3_WEIGHT_MODE='uniform'`. |
| **Reference** | Taylor & Letham (2018) *Forecasting at scale*, *American Statistician* 72(1):37-45. |

### 1.2 Quantile Regression (per-pair structural reconstruction)

| field | value |
|---|---|
| **Name** | Quantile Regression at q=0.5 (LAD median regression) |
| **Category** | Robust regression / structural reconstruction |
| **Location** | `src/sl_ads/train/train_models.py::train_models` (Reconstruction branch) |
| **Inputs** | feature `x`, target `y` from `RECONST_RULES`. RedeRio uses 5 pairs: `bytes ← packets`, `bytes ← entropy_src_port`, `udp ← flows`, `fin ← syn`, `tcp ← packets`. |
| **Outputs** | `ŷ = α + βx` (or `ŷ = βx` if `fit_intercept=False`) |
| **Estimator** | `sklearn.linear_model.QuantileRegressor(quantile=0.5, alpha=0.0, solver='highs', fit_intercept=<from rule>)` |
| **Mathematical form** | minimisation of L1 loss `Σ ρ_{0.5}(y − α − βx)`, with `ρ_τ(u) = u(τ − 1{u<0})` (Koenker & Bassett 1978). Quantile q=0.5 → LAD median. |
| **Robustness** | LAD breakdown point 50 % for *response* outliers (Koenker & Bassett 1978); *not* robust to leverage outliers (Rousseeuw & Leroy 1987 §3.3). Leverage robustness is provided by physical constraint `fit_intercept=False` for extensive quantities (Bridgman 1922 dimensional analysis). |
| **R² estimation** | 5-fold `TimeSeriesSplit` CV with `r2_score`; `r2 = mean(scores)`. |
| **Mean-fallback** | If `r2_CV < 0` AND rule has `allow_mean_fallback=True` → `DummyRegressor(strategy='mean')`, `r2 ← 0`. Recorded in `_FALLBACK_LOG['reconstruction_dummy']`. Justification: Hyndman & Athanasopoulos (2021) §3.1 baseline. |
| **Why QR not RANSAC** | Documented in the Reconstruction branch of `train_models`: training set is by construction attack-free (split before injections), so RANSAC's outlier-rejection causes *seed variance* (22–75 % on `T_susp`); QR(0.5) uses all clean points, deterministic. |
| **Reference** | Koenker & Bassett (1978) *Econometrica* 46(1):33-50; Rousseeuw & Leroy (1987); Bridgman (1922). |

### 1.3 Forecasting / reconstruction residuals

| field | value |
|---|---|
| **Name** | Signed point-residual sequence |
| **Category** | Statistical residual modelling |
| **Location** | `src/sl_ads/train/compute_evidence.py::compute_evidence` |
| **Form** | `e_t = y_t − ŷ_t`, signed; sign carries directional information (positive = surplus, negative = deficit). |
| **Used by** | EVT threshold calibration (1.4), trapezoidal evidence map (3.1), EDP (3.4), DECISION_THRESHOLD calibration (5.2), per-window RMSE for `c3_weight=online_rmse`. |

### 1.4 Generalized Pareto Distribution — Peaks-Over-Threshold

| field | value |
|---|---|
| **Name** | EVT/POT calibration of `T_susp`, `T_atk` |
| **Category** | EVT / tail modelling |
| **Location** | `src/sl_ads/train/train_models.py::_grimshaw_fit`, `::_pwm_gpd_fit`, `::_evt_threshold`, `::_evt_threshold_pair`, `::calibrate_thresholds_v2`, `::calibrate_thresholds_per_regime_v2` |
| **Mathematical form** | (i) initial threshold `t₀ = quantile(|e|, EVT_INIT_QUANTILE=0.90)`; (ii) excesses `Y = e[|e|>t₀] − t₀`; (iii) GPD MLE via Grimshaw 1D reduction `g(θ) = θW(θ)(1+V(θ)) − V(θ) = 0` with `V(θ) = mean log(1+θxᵢ)`, `W(θ) = mean xᵢ/(1+θxᵢ)`, returning `(ξ̂, σ̂)`; (iv) GPD quantile (Siffer 2017 Eq. 4): `T_q = t₀ + (σ/ξ)(q_cond^{−ξ} − 1)` if `|ξ|≥1e-10`, else `T_q = t₀ + σ log(1/q_cond)`, with `q_cond = q · n_total / n_peaks`. |
| **Calibration targets** | `EVT_Q_SUSP_PROPHET=0.01`, `EVT_Q_ATK_PROPHET=0.001`, `EVT_Q_SUSP_RANSAC=0.01`, `EVT_Q_ATK_RANSAC=0.001` (legacy key name `RANSAC`; active reconstruction model is QR(0.5)). These are **excess** probabilities `P(|e|>T | normal)`. |
| **GPD validity check** | Coles 2001 §4.2: `σ̃ = σ − ξ·t₀ > 0`; if not, fall back to empirical quantile. Logged in `_FALLBACK_LOG['evt_sigma_mod']`. |
| **Fallback hierarchy** | (1) Grimshaw MLE → (2) PWM closed-form fallback → (3) `scipy.stats.genpareto.fit` (`floc=0`) → (4) empirical quantile `quantile(excesses, 1−q_cond)`. Each fallback recorded in `_FALLBACK_LOG`. |
| **Safety margin** | `T_atk ← max(T_atk, THRESHOLD_SAFETY_MARGIN · T_susp)` with `THRESHOLD_SAFETY_MARGIN=1.10`. |
| **Declustering** | Disabled (`EVT_DECLUSTER_RUN=−1`); justification: Prophet residuals are pre-whitened. Davison & Smith (1990) declustering implementation present but not active. |
| **Asymmetric branch** | For `direction='both'`, two independent GPD fits on positive and negative excesses produce `T_susp_pos`, `T_atk_pos`, `T_susp_neg`, `T_atk_neg`. |
| **Inputs** | signed residuals from a single metric (model surrogate calibration uses out-of-sample residuals of `df_train_calib`, see 5.1). |
| **Outputs** | `(T_susp, T_atk, T_trapeze_base)` per metric, plus directional variants. |
| **Reference** | Grimshaw (1993) *Technometrics* 35(2):185–191; Coles (2001) *Intro. to Statistical Modeling of Extreme Values* §4.2–4.3; Siffer et al. (2017) *KDD* (SPOT); Davison & Smith (1990) *JRSS-B*; Pickands (1975); Balkema & de Haan (1974). |

---

## 2. Subjective-Logic Operators

All operators are implemented in `src/sl_ads/core/subjective_logic.py` and consumed by `src/sl_ads/core/opinions_pipeline.py`.

### 2.1 Multinomial Opinion class

| field | value |
|---|---|
| **Name** | `MultinomialOpinion(b, u, a)` |
| **Location** | `subjective_logic.py::MultinomialOpinion` |
| **Form** | enforces `Σb + u = 1` (renormalisation if drift > 1e-6, ValueError beyond 1e-6 of consistency error). |
| **Methods** | `projected_prob() = b + a·u` (Jøsang 2016 Eq. 3.23); `confidence() = 1 − u` (Eq. 3.43); `uncertainty_maximized()` (Eq. 3.27). |

### 2.2 Bijection Evidence ↔ Opinion (Def. 3.9)

| field | value |
|---|---|
| **Name** | `evidence_to_opinion(r, W=3, a)` and `opinion_to_evidence(op, W=3)` |
| **Location** | `subjective_logic.py::evidence_to_opinion`, `::opinion_to_evidence` |
| **Form** | Forward: `b_i = r_i / (Σr + W)`, `u = W / (Σr + W)`. Inverse: `r_i = W b_i / u`, capped at `r_max = W · SL_EVIDENCE_MAX_FACTOR` (default `1e4`) when `u < 1e-9` to prevent float64 overflow. |
| **Cap warning** | Once-per-metric `RuntimeWarning` (`_SL_CAP_WARNED` set) when the cap is hit. |
| **Reference** | Jøsang (2016) Def. 3.9. |

### 2.3 Uncertainty maximisation (Eq. 3.27)

| field | value |
|---|---|
| **Name** | `MultinomialOpinion.uncertainty_maximized()` |
| **Location** | `subjective_logic.py::MultinomialOpinion.uncertainty_maximized` |
| **Form** | `ü = min_i { P(x_i) / a(x_i) : a(x_i) > 0 }`; `b̈_i = max(P_i − a_i·ü, 0)`. |
| **Activation** | `CONFIG['UNCERTAINTY_MAXIMIZATION']` (default `False` in production). |

### 2.4 Trapezoidal evidence map (`compute_instantaneous_evidence`)

| field | value |
|---|---|
| **Name** | Directional fuzzy-trapezoidal map of a signed residual to `(p,s,n)` |
| **Category** | Heuristic — fuzzy membership (Dubois & Prade 1988-ish); not derived from Jøsang. |
| **Location** | `compute_evidence.py::compute_instantaneous_evidence`; mirrored single-point version `train_models.py::_apply_trapezoid_single`. |
| **Form** | Given signed residual `e`, thresholds `(t_trap, t_susp, t_atk)`, and direction `d ∈ {pos, neg, both, sym}`: <br/> &nbsp;&nbsp;• if `d='pos' ∧ e ≤ 0` or `d='neg' ∧ e ≥ 0`: return `(1,0,0)`; <br/> &nbsp;&nbsp;• `e' = e` (pos), `\|e\|` (neg / sym); <br/> &nbsp;&nbsp;• piecewise linear: `(1,0,0)` if `e' < t_trap`; ramp `(1−α, α, 0)` if `e' ∈ [t_trap, t_susp)`; ramp `(0, 1−α, α)` if `e' ∈ [t_susp, t_atk)`; `(0,0,1)` if `e' ≥ t_atk`. |
| **Window aggregation** | `r = Σ_{j=1..n_window} (p_j, s_j, n_j)` ⇒ `r_safe + r_susp + r_atk = n_window` (invariant explicit in `compute_evidence.py::compute_evidence`). |
| **Critical caveat** | This is a **fuzzy mapping**, not an SL operator. The triplet `r` is a *count*-like aggregate fed into the bijection (Def. 3.9), but the trapezoidal weights `(p,s,n)` themselves are not opinions and have no probabilistic meaning. |
| **Hyperparameter** | `T_TRAPEZE_RATIO=0.1` ⇒ `t_trap = 0.1·t_susp` (configurable). Optional quantile-based variant via `USE_QUANTILE_TRAPEZE=True` and `Q_TRAPEZE_BASE=0.95`. |
| **Reference** | None canonical; engineering choice. The 5-state directional refinement (`_S_pos`, `_S_neg`, `_N_pos`, `_N_neg`) matches Jøsang 2016 §3.5.4 *coarsening* identity (`r_susp = r_susp_pos + r_susp_neg`, in `compute_evidence.py::compute_evidence`). |

### 2.5 Conflict-Aware Ageing (`temporal_adaptive_ageing`)

| field | value |
|---|---|
| **Name** | Conflict-Aware Ageing — adaptive temporal forgetting |
| **Category** | Engineering extension of Jøsang (2016) §16.2.2 Eq. 16.5. |
| **Location** | `subjective_logic.py::temporal_adaptive_ageing` |
| **Form** | `K = K(r_prev, r_curr)` (cf. 2.5.1 below); `K_eff = clip(α K, 0, 1)`; `λ_dyn = λ_base · (1 − K_eff)^γ`; `R_{τ+1} = λ_dyn · R_τ + r_{τ+1}`. With `γ=1` (linear), `α=CONFIG['CONFLICT_ALPHA']=1.495`, `λ_base=CONFIG['LAMBDA_DECAY']=0.85`. |
| **Construction of α** | Computed in `config.py` such that `λ_dyn = 0` exactly when conflict reaches its theoretical maximum: `α = 1/K_max` with `K_max = b_prev_max · b_curr_max`, `b_curr_max = N_win/(N_win+W_SL)`, `b_prev_max = 2N_win/(2N_win+W_SL)`. For RedeRio `N_win=10` and `W_SL=3`, so `α≈1.495`. |
| **Effect** | Asymmetrically erases accumulated evidence on escalating transitions (calm→alarm, suspect→attack). De-escalation is *not* penalised. |
| **Ablation hook** | `conflict_aware=False` reverts to fixed Eq. 16.5: `R_{τ+1} = λ_base · R_τ + r_{τ+1}`. |
| **Reference** | Jøsang (2016) §16.2.2 Eq. 16.5 (base form); the `(1-K)^γ` modulation is engineering. |

#### 2.5.1 Conflict degree variants

Three variants, selected via `CONFIG['CONFLICT_MODE']`:

| variant | `CONFLICT_MODE` | function | formula |
|---|---|---|---|
| asymmetric escalation (default) | `belief_mass` | `subjective_logic.py::compute_asymmetric_escalation_conflict` | `K = b_prev[safe]·b_curr[atk] + b_prev[atk]·b_curr[safe] + b_prev[safe]·b_curr[susp] + b_prev[susp]·b_curr[atk]` (omits de-escalation cross-products by design) |
| symmetric BCF (canonical) | (used only by `compute_conflict_degree_canonical`) | `subjective_logic.py::compute_conflict_degree_canonical` | `K = Σ_{i≠j} b_prev[i]·b_curr[j]` (Jøsang 2016 Eq. 12.4 multinomial extension) |
| projected probability | `projected_prob` | `subjective_logic.py::compute_conflict_degree_projected` | same cross-products on `P = b + a·u` |
| symmetric KL | `kl_symmetric` | `subjective_logic.py::compute_conflict_degree_kl` | `K = 1 − exp(−KL_sym(P‖Q) / τ)` with `τ=CONFIG['CONFLICT_KL_TAU']=1.0` |

### 2.6 Weighted Belief Fusion N-ary (Eq. 12.27 evidence space)

| field | value |
|---|---|
| **Name** | `fusion_wbf_n_sources(opinions, external_weights, W)` |
| **Location** | `subjective_logic.py::fusion_wbf_n_sources` |
| **Form** | `w_i = ext_w_i · c_i` with `c_i = 1 − u_i`; `w̄_i = w_i / Σw`; `r_fused = Σ w̄_i · opinion_to_evidence(op_i)`; `a_fused = Σ w̄_i · a_i`; return `evidence_to_opinion(r_fused, W, a_fused)`. |
| **Faithfulness statement** | Faithful to Eq. 12.27 (confidence-weighted *evidence* averaging via the bijection). **Not** a literal opinion-space reproduction of Eq. 12.22–12.24; that form is provided by the dedicated `fusion_wbf_canonical_two`. |
| **Degenerate handling** | If `Σw < 1e-12`: return vacuous opinion `([0,0,0], 1, a_ref)`. |
| **Use sites** | (i) Intra-method fusion inside each `CONFIG['FUSION_METHOD_GROUPS']` group, (ii) optional inter-method `wbf` mode, (iii) confidence-weighted evidence averaging whenever a future third method family is added. |
| **Reference** | Jøsang (2016) §12.5 Eq. 12.27 (evidence-space confidence-weighted average). |

### 2.7 Weighted Belief Fusion 2-source canonical (Eq. 12.22–12.24)

| field | value |
|---|---|
| **Name** | `fusion_wbf_canonical_two(op_A, op_B)` |
| **Location** | `subjective_logic.py::fusion_wbf_canonical_two` |
| **Form (Case I, ∃u_i > 0)** | `D = c_A u_B + c_B u_A`; `b^⋄ = (c_A u_B b_A + c_B u_A b_B)/D`; `u^⋄ = u_A u_B (c_A + c_B)/D`; `a^⋄ = (c_A a_A + c_B a_B)/(c_A + c_B)`. |
| **Form (Case II, u_A=u_B=0)** | symmetric: `b = ½(b_A+b_B)`, `u=0`, `a = ½(a_A+a_B)`. |
| **Used as** | reference / unit-test only; *not* on the production path (which uses `fusion_wbf_n_sources` for arity > 1). |
| **Reference** | Jøsang (2016) Def. 12.7, Eq. 12.22–12.24. |

### 2.8 Cumulative Belief Fusion (Eq. 12.14)

| field | value |
|---|---|
| **Name** | `fusion_cbf(op_A, op_B)` |
| **Location** | `subjective_logic.py::fusion_cbf` |
| **Form (Case I)** | `denom = u_A + u_B − u_A u_B`; `b^⋄ = (b_A u_B + b_B u_A)/denom`; `u^⋄ = (u_A u_B)/denom`; `a^⋄ = (a_A u_B + a_B u_A − ½(a_A+a_B) u_A u_B)/denom`, then projected to simplex. |
| **Form (Case II, dogmatic)** | `γ_A = c_A/(c_A+c_B)`; `b = γ_A b_A + (1−γ_A) b_B`, `u=0`. |
| **Independence assumption** | CBF is the SL fusion that requires *independent* sources (Jøsang Theorem 12.2, Eq. 12.17). The pipeline acknowledges this assumption is not satisfied by Prophet ⊥ Reconstruction (same raw windows in input) — `INTER_METHOD_FUSION='wbf'` is the default to avoid the dependency violation. |
| **Use sites** | (i) Inter-method fusion when `CONFIG['INTER_METHOD_FUSION']='cbf'` (dispatched in `opinions_pipeline.py::compute_opinions`), with optional `BALANCE_RATIO` rebalancing. |
| **Reference** | Jøsang (2016) Def. 12.5, Eq. 12.14, Theorem 12.2. |

### 2.8a Inter-method fusion operator dispatch

| mode | status | intended interpretation |
|---|---|---|
| `wbf` | Default headline mode on RedeRio | Confidence-weighted evidence averaging; preserves weights and performed slightly better than ABF in the strict 2026-05-07 recalibration. |
| `abf` | Implemented, tested, not default on RedeRio | Averaging fusion for dependent sources; theoretically attractive for Prophet/Reconstruction dependence, but did not improve F1/MCC/FPR after per-mode calibration. |
| `cbf` | Legacy ablation only | Cumulative evidence fusion; rejected for headline use because method independence is violated. |
| `bcf` | Implemented for ablation/sensitivity | Belief-constraint/Dempster-style fusion; useful to expose conflict pathologies, not recommended as default under strong disagreement. |
| `ccf` | Projected experimental implementation | Consensus/compromise-inspired projected operator; research-only until more thoroughly validated. |
| `minbf` | Stress-test lower-envelope fusion | Conservative AND-like behaviour; expected to hurt recall when one method is blind. |
| `maxbf` | Stress-test upper-envelope fusion | OR-like behaviour; expected to inflate FPR. |
| `hierarchical` | Compatibility mode | Equal method-level evidence averaging; retained for comparability with earlier hierarchical experiments. |

The dispatch layer consumes method-level opinions, so adding a third family
should be done by extending `CONFIG['FUSION_METHOD_GROUPS']` rather than by
hard-coding another Prophet/Reconstruction branch.

### 2.9 Trust discounting (Def. 14.6)

| field | value |
|---|---|
| **Name** | `apply_trust_discount(op, t)` |
| **Location** | `subjective_logic.py::apply_trust_discount` |
| **Form** | `b' = t·b`, `u' = 1 − t·(1−u) = 1 − t·Σb`. |
| **Use site** | optional in WBF mode `trust_discount` (deprecated, retained for ablation only — emits `RuntimeWarning` at module import time in `opinions_pipeline.py`). |
| **Reference** | Jøsang (2016) Def. 14.6, Eq. 14.6. |

### 2.10 Contextual discounting (Mercier–Quost–Denoeux)

| field | value |
|---|---|
| **Name** | `apply_contextual_discount(op, alpha)` |
| **Location** | `subjective_logic.py::apply_contextual_discount` |
| **Form** | `b'_i = α_i · b_i`; `u' = 1 − Σ b'_i`. |
| **Use site** | Reconstruction group on `attack` hypothesis only: `α = [1, 1, CD_ALPHA_ATTACK]`, with `CD_ALPHA_ATTACK ∈ [0,1]`, default `1.0` (no discount), auto-calibrated from "structural blindness rate" (see 5.3) when set to `'auto'`. |
| **Reference** | Mercier, Quost & Denoeux (2008) *Information Fusion* 9(2):246-258. |

### 2.11 Evidence boost (`boost_opinion_evidence`)

| field | value |
|---|---|
| **Name** | Multiplicative scaling of evidence prior to CBF |
| **Category** | Heuristic (explicitly flagged as deviation from Jøsang Theorem 12.2) |
| **Location** | `subjective_logic.py::boost_opinion_evidence` |
| **Form** | `r ← ratio · opinion_to_evidence(op, W)`; `op' = evidence_to_opinion(r, W, a)`. |
| **Use site** | When `INTER_METHOD_FUSION='cbf'` and `BALANCE_RATIO ≠ 1.0`, the dominant group is divided by `BALANCE_RATIO` to compensate for `N_prophet ≠ N_reconst`. |
| **Documented status** | The CBF-rebalancing block in `opinions_pipeline.py::compute_opinions` explicitly labels this an *engineering extension* of CBF and recommends the `hierarchical` (0.5/0.5 WBF) variant as the theoretically clean alternative. |

### 2.12 Empirical Dirichlet Prior (EDP)

| field | value |
|---|---|
| **Name** | `compute_edp_from_residuals` |
| **Category** | Empirical-Bayes prior estimation |
| **Location** | `train_models.py::compute_edp_from_residuals` |
| **Form** | for each metric, simulate the trapezoidal map on training residuals window-by-window, accumulate per-class evidence sums, divide by `n_window` to get marginal class proportions: `a_safe = mean(P_win)/n_window`, etc.; floor on `a_atk` (and on `a_atk_pos / a_atk_neg` for direction='both') at `EMPIRICAL_PRIOR_FLOOR=0.005`; renormalise to simplex. 5-state extension preserves coarsening identity `a_susp = a_susp_pos + a_susp_neg`. |
| **Storage** | `models_pkg['empirical_priors'][metric_key] = {'a_safe', 'a_susp', 'a_atk', (+ pos/neg variants)}`. |
| **Use site** | per-metric base rate `a_inj` injected into the bijection at every window inside `opinions_pipeline.py::compute_opinions`. Initial state memory `R_init = a_edp · W` so that `proj_atk_init = a_edp[atk]` exactly. |
| **Reference** | Ferguson (1973) *Annals of Statistics* 1:209–230 (Dirichlet process / Bayesian non-parametric prior); Robbins (1955, 1983) *Empirical Bayes*. (The code-comment explicitly cautions against citing Efron & Morris 1973 here — no shrinkage is used, only marginal frequencies.) |

### 2.13 SBN cause-attribution operator

| field | value |
|---|---|
| **Name** | `qualify_anomaly_sbn` (template-based Subjective-Bayesian-Network proxy) |
| **Category** | Hybrid: SL bijection + Naive-Bayes-like template scoring |
| **Location** | `src/sl_ads/qualify/sbn_qualifier.py` (single file; main entry `sbn_qualify_row`) |
| **Frame** | `Θ_qual = {UDP_FLOOD, SYN_FLOOD, ICMP_FLOOD, DNS_AMP, HTTP_FLOOD, SLOWLORIS, PORT_SCAN, DATA_EXFIL, NETWORK_OUTAGE, BOTNET_CC, NTP_AMP, BRUTE_FORCE_SSH, DNS_TUNNELING} ∪ {Autre_Anomalie}`; \|Θ_qual\| = 13 named + 1 residual. |
| **L1 — gate** | open ⇔ `proj_atk(FINAL_SYSTEM) > δ`. |
| **L2 — group projection** | per group `g` of metrics, `P^g_s = geomean_{m ∈ g} P^m_s` for `s ∈ {Safe, Susp, Anom}` (logarithmic opinion pooling, Genest & Zidek 1986), then renormalised to simplex. |
| **L3 — template score** | `score(k, g) = Σ_s P^obs_g(s) · c^{k\|g}(s)` where `c^{k\|g}` is an expert-elicited conditional opinion `P(state \| attack_k)` (manually specified at the top of `sbn_qualifier.py`, see the `SBN_PARAMS` block and the per-attack helpers). Two modes: `absolute` (default) and `contrastive` (subtracts per-group mean). |
| **L4 — evidence aggregation** | `e(k) = Σ_g max(0, score(k,g) − 1/3) · evidence_scale`; `1/3` is the neutral baseline over `{Safe,Susp,Anom}`, not the number of attack types. Then apply the bijection (Def. 3.9) with `W = \|Θ_qual_named\|`. |
| **L5 — temporal smoothing (optional)** | Markov transition `b_prev ← T^T b_prev` (kill-chain priors from Hutchins et al. 2011), discounted by `λ^Δt`, fused with current opinion via 2-source WBF (`_wbf_two`). Default `λ_temporal=0.80`, `temporal_weight=0.30`. |
| **L6 — uncertainty maximisation (optional)** | Eq. 3.27 with `Autre_Anomalie` excluded from the `K` count. |
| **Top-1 attribution** | `top1 = argmax_k b_final(k)` over named types only (`b_Autre_Anomalie ≡ 0`). |
| **Novelty / residual class** | `qual_status = 'autre_anomalie'` ⇔ `u_raw > SBN_NOVELTY_U_RAW_THRESHOLD = 0.82`. Auxiliary scalar `novelty_lr = 1 / (max L / mean L)` (concentration of likelihood vector). |
| **Reference** | Jøsang (2016) §3.5.2, §3.6, §12.5, Def. 14.6, §14.4 (residual class); Genest & Zidek (1986); Hutchins, Cloppert & Amin (2011); Domingos & Pazzani (1997) on Naive-Bayes robustness; Sharafaldin et al. (2018), Mirsky et al. (2018), Moustafa & Slay (2015), Rossow (2014) for attack signatures. |

---

## 3. Decision-threshold calibration (`DECISION_THRESHOLD`)

### 3.1 Calibration variable

| field | value |
|---|---|
| **Name** | `proj_atk = b_atk + a_atk · u` (Jøsang Eq. 3.23) |
| **Justification** | Smoother and unimodal compared to `b_atk` alone (which is bimodal under sparse evidence); see the docstring of `train_models.py::_compute_training_proj_atk`. |

### 3.2 Calibration procedure (`_compute_training_proj_atk`)

| field | value |
|---|---|
| **Name** | Surrogate `proj_atk` over hold-out calibration windows |
| **Location** | `train_models.py::_compute_training_proj_atk` |
| **Surrogate** | `_compute_training_proj_atk` replays the deployed leaf-to-group structure on hold-out calibration residuals: trapezoid → per-leaf opinion with EDP → intra-method WBF by `FUSION_METHOD_GROUPS` → contextual discount → requested inter-method fusion mode. It is still a surrogate because it is instantaneous and does not replay the temporal ageing state over the full deployment stream. |
| **Threshold rule** | `δ = quantile(proj_atk_train, 1 − FPR_TARGET_DECISION)` with `FPR_TARGET_DECISION = 0.001` (RedeRio) or `0.01` (METR-LA). |
| **Sparse-distribution guards** | (i) bijection-floor detection: if `δ ≈ 1/(W+W)` within `CALIB_BIJECTION_FLOOR_TOL=0.01`, replace by `floor·(1−λ_base)·CALIB_AGEING_WIN_FRACTION=0.5`; (ii) if all `proj_atk ≈ 0`, fallback `δ = b_atk_min · 0.5`; (iii) `MIN_DECISION_THRESHOLD` plancher. |
| **Persistence** | Stored both in `models_pkg['_decision_threshold']` and in a sidecar `<model_name>_threshold.json` so downstream code can read it without unpickling the full PKL (helper `paths.py::get_decision_threshold`). |
| **Documented caveat (PATCH TASK-45 + TASK-55)** | The sidecar stores the deployed configuration alongside the threshold; downstream evaluators verify sensitive knob matches, and strict fusion comparisons must use mode-specific thresholds. Realised test FPR can still drift because the calibration residual distribution and the temporally-aged deployment stream are not identical. |
| **Reference** | Ruff et al. (2021) *TPAMI* — anomaly-detection holdout calibration. |

### 3.3 Optional auto-calibration of `RECONST_ATTACK_RELIABILITY`

| field | value |
|---|---|
| **Name** | `_auto_calibrate_reconst_reliability` |
| **Location** | `train_models.py::_auto_calibrate_reconst_reliability` |
| **Definition** | `blind_rate = #{windows : Prophet shows attack-evidence AND Reconst belief in safe > 0.85} / #{Prophet-suspect windows}`; `α_attack = clip(1 − blind_rate, 0.05, 1.0)`. Activated only when `CONFIG['RECONST_ATTACK_RELIABILITY']='auto'`. The current complete RedeRio run uses the explicit config value `1.0`, so this mechanism is documented as an available option, not a shipped calibration result. |

---

## 4. Evaluation methods

### 4.1 Detection metrics (per-attack and global)

| metric | definition | location |
|---|---|---|
| `DR` (detection rate, recall) | `n_detected / n_attack_windows` | `evaluate_qualify_sbn.py` |
| `QP` (qualification precision) | `n_correct / n_qualified` (top-1 attack-type matches expected) | `evaluate_qualify_sbn.py` |
| `F1` | `2·DR·QP / (DR+QP)` | `evaluate_qualify_sbn.py` |
| `F_β=2` | `(1+β²)·DR·QP / (β²·DR + QP)` (Tavallaee 2009 IDS standard) | `evaluate_qualify_sbn.py` |
| `FAR` (FPR on normal) | `FP / (FP + TN)` excluding `NETWORK_OUTAGE` periods | `evaluate_qualify_sbn.py::_compute_global_detection_stats` |
| `MCC` (Matthews) | `(TP·TN − FP·FN)/√{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}` | `evaluate_qualify_sbn.py::_compute_global_detection_stats` |
| `TTQ` (time-to-qualify) | minutes from `t_start` to first correct top-1 | `evaluate_qualify_sbn.py` |

### 4.2 Range-aware AUC / VUS (Paparrizos et al. 2022)

| metric | definition | location |
|---|---|---|
| `R-AUC-ROC(L)` | AUC after expanding each anomaly run by buffer L | `evaluate/vus_metrics.py` |
| `R-AUC-PR(L)` | range-aware average precision | `evaluate/vus_metrics.py` |
| `VUS-ROC` | `(1/L_max) ∫₀^{L_max} R-AUC-ROC(L) dL` (trapezoidal) | `evaluate/vus_metrics.py` |
| `VUS-PR` | analogous on PR | `evaluate/vus_metrics.py` |
| Existence recall | fraction of true ranges with ≥ 1 positive prediction | `evaluate/vus_metrics.py` |

### 4.3 Axelsson PPV analysis (axelsson_ppv.py)

| metric | definition | location |
|---|---|---|
| Wilson score CI | closed-form for proportions; preferred over Wald near 0/1 | `evaluate/axelsson_ppv.py` |
| Per-attack base rate | `π = n_attack_windows / n_total` | `evaluate/axelsson_ppv.py` |
| PPV (Bayesian) | `PPV = TPR·π / (TPR·π + FPR·(1−π))` | `evaluate/axelsson_ppv.py` |
| `FPR_required(target_PPV)` | inversion: `FPR ≤ TPR·π·(1/PPV − 1)/(1−π)` | `evaluate/axelsson_ppv.py` |

### 4.4 Statistical inference helpers

| name | description | location |
|---|---|---|
| BCa bootstrap CI | bias-corrected accelerated CI (Efron 1987); jackknife-based acceleration; default `n_boot=2000` | `stats/bootstrap_ci.py::bootstrap_bca_ci` |
| McNemar test | `χ²` continuity-corrected for `n_disc ≥ 25`, exact binomial for `n_disc < 25` | `stats/mcnemar.py` |
| Effective sample size | Newey-West-style autocorrelation correction (lag ≤ 10, weights `1 − lag/(L+1)`) | `stats/residual_correlation.py::newey_west_eff_n` |
| VIF collinearity | warning if `VIF > 5` | `stats/residual_correlation.py` |

### 4.5 Baseline comparison scripts

| field | value |
|---|---|
| **Same-evidence no-SL** | `compare_no_sl_fair.py` removes the SL bijection, uncertainty, EDP, ageing, and fusion, then scores the same evidence CSV with direct scalar functions over attack evidence `N`. Thresholds are calibrated on persisted train-calib residuals only. |
| **Raw baselines** | `compare_raw_baselines_fair.py` trains `IsolationForest`, robust modified-z, and PCA reconstruction error directly on raw `ACTIVE_METRICS`, using pre-split normal rows only. It requires `opinions_non_injected/detection_results_RAW.csv` for the SL row and excludes synthetic injection windows because the 13 catalog attacks are injected at evidence level, not raw-traffic level. |
| **Legacy IF agreement** | `compare_if_fair.py` remains available for raw pseudo-label agreement. Its F1 must not be compared to catalog F1 because the label source differs and, on the current RedeRio artifact, injected SL alarms can be scored against non-injected pseudo-labels. |
| **Threshold discipline** | All paper-facing baseline thresholds are calibrated on train/pre-split normal data. Test labels are used only for reporting metrics and paired tests. |

### 4.6 Episode-level audit (`audit_full_dataset.py`)

| field | value |
|---|---|
| **Episode grouping** | contiguous `gate_open=True` windows merged with `gap_min=15 min`. |
| **Matching** | IoU between episode and known events (`INJECTED_ATTACK_CATALOG ∪ REAL_ATTACKS`). |
| **Wilson CI** | on FAR (`stats/proportions_wilson`). |
| **Per-episode novelty** | `u_sbn`, `novelty_entropy`, `novelty_lr`, top-1 attack type, pre-DDoS activity flag (Benson 2010 IMC). |

---

## 5. Synthetic injection mechanism (`inject/evidence_level.py`)

| field | value |
|---|---|
| **Name** | Evidence-space deterministic anomaly injector |
| **Category** | Synthetic ground-truth generation (engineering / experimental design) |
| **Location** | `src/sl_ads/inject/evidence_level.py` |
| **Where** | Operates on `evidence_<VERSION>.csv` columns (`<metric>_P/S/N`) — *not* on raw data, *not* on residuals. Produces `evidence_<VERSION>_attacks.csv` consumed by `sl_ads.core.opinions_pipeline`. |
| **Catalog** | 13 attacks defined in `CONFIG['INJECTED_ATTACK_CATALOG']` (PATCH-C1 unified source, in `config.py`): UDP_FLOOD_DDOS, SYN_FLOOD_DDOS, ICMP_FLOOD_BURST, NTP_AMPLIFICATION, DNS_AMPLIFICATION, HTTP_FLOOD_L7_DDOS, SLOWLORIS_DOS, PORT_SCAN, DATA_EXFILTRATION_SLOW, BRUTE_FORCE_SSH, BOTNET_CC_BEACONING, DNS_TUNNELING, UNKNOWN_ANOMALY_CONTROL. |
| **Per-attack signature** | for each `(metric, attack)` tuple, raw `(ev_normal, ev_suspect, ev_attack)` weights are normalised to sum `WINDOW_SIZE` (preserves bijection invariant `P+S+N = n_window`). |
| **Time profile** | trapezoidal ramp (`make_ramp(n, ramp_frac)`): linear rise/fall on `ramp_frac · n` windows each, plateau in between; `α(t) ∈ [0,1]`. |
| **Per-window injection** | `P_t = (1−α_t) · n_window + α_t · P_norm`; `S_t = α_t · S_norm`; `N_t = α_t · N_norm`. Invariant `P+S+N = n_window` preserved exactly. |
| **Determinism** | no random seed; same injection on every run. Calendar-anchored (e.g. `2025-12-15 22:00 → 2025-12-16 06:00` for SLOWLORIS_DOS). |
| **Disjointness** | overlap check in `evidence_level.py::_validate_catalog` prevents double-injection. |
| **Ground truth columns** | `injection_label` (attack name or `"normal"`) and `injection_ramp_alpha ∈ [0,1]` written into the output CSV. |

---

## 6. Notation summary

| symbol | meaning |
|---|---|
| `Θ` | frame of discernment for the anomaly opinion: `{Safe, Suspect, Attack}`; `K=3`. |
| `Θ_qual` | frame for qualifier: 13 named attack types + `Autre_Anomalie`. |
| `b, u, a` | belief vector, uncertainty, base rate of an opinion `ω = (b, u, a)`. |
| `P(x) = b(x) + a(x)·u` | projected probability (Jøsang Eq. 3.23). |
| `r = (r_safe, r_susp, r_atk)` | evidence vector; `r_i ≥ 0`; `Σr = n_window` for full windows. |
| `W = K = 3` | non-informative prior weight in the bijection. |
| `λ_base, λ_dyn` | base and dynamic ageing factors. |
| `K (in §2.5)` | conflict degree (`∈ [0,1]`); `K_eff = clip(α·K, 0, 1)`. |
| `δ` | decision threshold on `proj_atk`; `δ = quantile(proj_atk_calib, 1 − FPR_target)`. |
| `T_susp, T_atk` | EVT-calibrated residual thresholds. |
| `T_trapeze_base` | start of the trapezoidal ramp; `0.1 · T_susp` by default. |

---

## 7. Method-by-step coverage matrix

| step | residual-stat | EVT | SL bijection | SL ageing | SL fusion | SL discounting | heuristic |
|---|---|---|---|---|---|---|---|
| 1. train | ✓ | ✓ | ✓ (EDP, threshold surrogate) | — | — | (auto-calibrate α_attack) | mean fallback, balance ratio |
| 2. evidence | ✓ | (uses thresholds from step 1) | — | — | — | — | trapezoidal map |
| 3. inject | — | — | — | — | — | — | catalog overwrite |
| 4. opinions | — | — | ✓ | ✓ (Conflict-Aware) | ✓ (WBF/ABF/CBF/BCF/projected-CCF/MinBF/MaxBF/hierarchical) | ✓ (contextual on Reconst) | method groups, balance ratio, c3 weighting |
| 5. eval_injection | — | — | — | — | — | — | Wilson CI, F1 sweeps |
| 6. qualify_sbn | — | — | ✓ (per attack type) | — | ✓ (WBF temporal blend) | ✓ (trust on temporal prior) | template scoring, Markov prior |
| 7. eval_qualify | — | — | — | — | — | — | F1/F2, AUC novelty |
| 8. ablation | — | — | — | — | — | — | sweep over knobs |
| 9. compare_if | — | — | — | — | — | — | legacy raw IF agreement; add-on fair baseline scripts |
| 10. audit | — | — | — | — | — | — | episode IoU, Wilson CI |

The audit therefore reduces to verifying:

1. The bijection (§2.2) and the trapezoidal map (§2.4) — *only* SL-grounded transformation of residuals to opinions;
2. The conflict-aware ageing (§2.5) and fusion stack (§2.6, §2.7, §2.8, §2.8a) — the heart of the SL pipeline;
3. The threshold calibration chain (1.4 → 3.2) — the only quantity controlling the operational FPR;
4. The injection/evaluation symmetry (§5 / §4) — the only path through which F1 numbers reach the paper.

These four items are the load-bearing claims and are audited in detail in `ASSUMPTIONS.md` and `PIPELINE_LOGIC.md`.
