# REFERENCES.md — Canonical literature inventory

> Every concept that appears in `METHODS.md` / `ASSUMPTIONS.md` / `PIPELINE_LOGIC.md` /
> `THEORY_GRAPH.md` is mapped here to:
>
> - the **canonical reference** (the original or accepted authoritative source)
> - the **field**
> - a **short explanation** of what the codebase imports from that reference
> - the **call sites** in the codebase (file/function or symbol)
>
> When the codebase explicitly cites a reference in code-comments, the citation
> is reproduced verbatim. When the reference is implicit, the **inferred-from**
> column points to the operator that uses it.
>
> Section 9 lists internal cross-references (PATCH numbers, audit documents)
> for traceability across the audit corpus.

---

## 1. Subjective Logic

| Concept | Reference | Field | Used by |
|---|---|---|---|
| Multinomial opinion `(b, u, a)`, constraint `Σb + u = 1` | Jøsang, A. (2016). *Subjective Logic: A Formalism for Reasoning Under Uncertainty*. Springer. ISBN 978-3-319-42337-1. **Def. 3.1, §3.2.** | Uncertainty reasoning | `core/subjective_logic.py::MultinomialOpinion` |
| Bijection evidence ↔ opinion | Jøsang (2016) **Def. 3.9.** | Subjective Logic | `core/subjective_logic.py::evidence_to_opinion`, `::opinion_to_evidence` |
| Projected probability `P(x) = b(x) + a(x)·u` | Jøsang (2016) **Eq. 3.23.** | Subjective Logic | `core/subjective_logic.py::MultinomialOpinion.projected_prob`; decision rule, EDP, threshold calibration |
| Confidence `c = 1 − u` | Jøsang (2016) **Eq. 3.43.** | Subjective Logic | `core/subjective_logic.py::MultinomialOpinion.confidence`; WBF weighting |
| Uncertainty maximisation, base-rate-preserving | Jøsang (2016) **§3.6, Eq. 3.27.** | Subjective Logic | `core/subjective_logic.py::MultinomialOpinion.uncertainty_maximized`; SBN qualifier L6 |
| Coarsening / refining of frame of discernment | Jøsang (2016) **§3.5.4.** | Subjective Logic | 5-state ⇋ 3-state directional bookkeeping in `compute_evidence.py` and `core/opinions_pipeline.py::compute_opinions` |
| Cumulative Belief Fusion (CBF) | Jøsang (2016) **Def. 12.5, Eq. 12.14.** | Belief fusion | `core/subjective_logic.py::fusion_cbf` |
| CBF ≡ evidence sum (independence) | Jøsang (2016) **Theorem 12.2, Eq. 12.17.** | Belief fusion | rationale for `BALANCE_RATIO`; documented assumption violation in inter-method fusion default |
| Averaging Belief Fusion (ABF) | Jøsang & McAnally (2009), Jøsang (2016) **Eq. 12.20**, Wang & Jøsang (2017) multi-source form. | Belief fusion | implemented inter-method option for dependent-source ablation; strict WBF/ABF comparison recorded in TASK-55 |
| Belief Constraint Fusion (BCF) | Jøsang (2016) **Eq. 12.31**; Dempster-Shafer lineage. | Belief fusion | inter-method ablation/stress-test operator; not default due conflict pathologies |
| Belief constraint (BCF) conflict | Jøsang (2016) **Eq. 12.4** (multinomial extension). | Belief fusion | `core/subjective_logic.py::compute_conflict_degree_canonical`; reference for asymmetric variant `::compute_asymmetric_escalation_conflict` |
| Weighted Belief Fusion (2 sources, opinion-space, Case I/II) | Jøsang (2016) **Def. 12.7, Eq. 12.22-24.** | Belief fusion | `core/subjective_logic.py::fusion_wbf_canonical_two` |
| Weighted Belief Fusion (N sources, evidence-space) | Jøsang (2016) **§12.5, Eq. 12.27.** | Belief fusion | `core/subjective_logic.py::fusion_wbf_n_sources`; intra-method and inter-method WBF in `core/opinions_pipeline.py::compute_opinions` |
| Consensus and Compromise Fusion (CCF) | Wang, Jøsang & Zhang (2017); Jøsang (2018) categories of belief fusion. | Belief fusion | projected experimental `ccf` inter-method mode; research-only until further validation |
| Minimum/Maximum belief fusion | van der Heijden, Kopp & Kargl (2018) survey of multi-source SL fusion operators. | Belief fusion | `minbf`/`maxbf` ablation stress tests |
| Probability-sensitive trust discount | Jøsang (2016) **Def. 14.6, Eq. 14.6.** | Trust modelling | `core/subjective_logic.py::apply_trust_discount`; deprecated `WBF_WEIGHT_MODE='trust_discount'` |
| Residual class / reject-style ignorance | Jøsang (2016) **§14.4** + Chow (1970). | Decision theory | `Autre_Anomalie` handling in `qualify/sbn_qualifier.py` |
| Subjective Bayesian Networks (acknowledged proxy) | Jøsang (2016) **Chapter 14** discussion. | SL networks | `qualify/sbn_qualifier.py` (template-based; explicitly *not* a full SBN — see the module header docstring) |
| Temporal evidence ageing | Jøsang (2016) **§16.2.2, Eq. 16.5** `R_{τ+1} = λ R_τ + r_{τ+1}`. | Temporal SL | `core/subjective_logic.py::temporal_adaptive_ageing` |
| Contextual discount (α-vector per hypothesis) | Mercier, D., Quost, B., & Denoeux, T. (2006/2008). *Contextual discounting of belief functions*. ECSQARU 2006 / *Information Fusion* 9(2):246–258, 2008. | Belief functions | `core/subjective_logic.py::apply_contextual_discount`; used on Reconstruction's `attack` hypothesis |

---

## 2. Extreme Value Theory

| Concept | Reference | Field | Used by |
|---|---|---|---|
| Tail of i.i.d. sequence converges to GPD | Pickands, J. (1975). *Statistical inference using extreme order statistics*. *Annals of Statistics* 3(1):119–131. <br/>Balkema, A. A., & de Haan, L. (1974). *Residual life time at great age*. *Annals of Probability* 2(5):792–804. | EVT / tail | `_evt_threshold`, `_evt_threshold_pair` justification |
| GPD MLE via 1D root-finding | Grimshaw, S. D. (1993). *Computing maximum likelihood estimates for the generalized Pareto distribution*. *Technometrics* 35(2):185–191. | EVT | `train/train_models.py::_grimshaw_fit` |
| Stability plot, validity condition `σ̃ = σ − ξ·t₀ > 0` | Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer. **§4.2–4.3.** | EVT | validity check inside `train/train_models.py::_evt_threshold_pair` |
| Declustering (run-length) | Davison, A. C., & Smith, R. L. (1990). *Models for exceedances over high thresholds*. *JRSS-B* 52(3):393–442. | EVT | declustering branch inside `train/train_models.py::_evt_threshold` — disabled in production |
| SPOT online algorithm, GPD-quantile formula | Siffer, A., Fouque, P. A., Termier, A., & Largouet, C. (2017). *Anomaly detection in streams with extreme value theory*. *KDD 2017*. | EVT / streaming | `_evt_threshold` Eq. 4 use; `EVT_INIT_QUANTILE = 0.90` is the SPOT-recommended initial level |

---

## 3. Time-series forecasting and reconstruction

| Concept | Reference | Field | Used by |
|---|---|---|---|
| Prophet additive decomposition | Taylor, S. J., & Letham, B. (2018). *Forecasting at scale*. *American Statistician* 72(1):37–45. | Forecasting | Prophet model fit in `train/train_models.py::train_models`; `growth='flat'`, daily-conditional Fourier seasonality |
| Time-series rolling-origin CV | Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.), §3.4 evaluation. | Forecasting | `prophet.diagnostics.cross_validation(initial='14d', period='3d', horizon='1d')` inside `train/train_models.py::train_models` |
| Mean-baseline forecast (R²<0 fallback) | Hyndman & Athanasopoulos (2021) **§3.1** — Average method. | Forecasting | `DummyRegressor(strategy='mean')` fallback inside `train/train_models.py::train_models` |
| Quantile regression / LAD | Koenker, R., & Bassett, G. (1978). *Regression quantiles*. *Econometrica* 46(1):33–50. | Robust regression | `QuantileRegressor(quantile=0.5, alpha=0, solver='highs')` inside `train/train_models.py::train_models` |
| LAD breakdown bound (50 % response, 0 % leverage) | Rousseeuw, P. J., & Leroy, A. M. (1987). *Robust Regression and Outlier Detection*. Wiley. **§3.3.** | Robust statistics | Justifies QR(0.5) for response outliers + `fit_intercept=False` for leverage protection |
| Dimensional homogeneity (origin = 0 for extensive quantities) | Bridgman, P. W. (1922). *Dimensional Analysis*. Yale UP. | Physics | `RECONST_RULES` `fit_intercept=False` for `bytes ← packets`, `bytes ← entropy_src_port` |
| Cross-validation removes in-sample bias | Stone, M. (1974). *Cross-validatory choice and assessment of statistical predictions*. *JRSS-B* 36(2):111–147. | Statistics | `TimeSeriesSplit(5)` for QR R² inside `train/train_models.py::train_models`; justifies CV-based trust scores |
| In-sample R² is optimistic | Hastie, Tibshirani, Friedman (2009). *The Elements of Statistical Learning*, §7.10. | Statistics | PATCH-M1 cross-citation in train code |
| Mean Absolute Scaled Error (MASE) — scale-invariant accuracy metric | Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. *International Journal of Forecasting* 22(4):679–688. | Forecasting | `stats/mase.py::compute_mase`; `train/train_models.py` per-Prophet `mase_score` persistence; `WBF_WEIGHT_MODE='mase'` (PATCH D5) |
| Forecast skill score relative to a reference baseline | Murphy, A. H. (1988). *Skill scores based on the mean square error and their relationships to the correlation coefficient*. *Monthly Weather Review* 116(12):2417–2424. | Forecast verification | `stats/mase.py::mase_to_trust` skill-score interpretation `1 − MASE` (PATCH D5) |
| MASE pathology at high sampling frequency (Naive-1 dominant) | Hyndman, R. J., & Koehler, A. B. (2006) §3 — empirical observation that MASE's informativeness depends on the dominance of the Naive-1 baseline. | Forecasting / methodology | Documented in `docs/audit/trust_discount_r2_analysis.md` §4.1 and current ablation (`MASE-Trust legacy` detects 0/14 attacks) |

---

## 4. Empirical Bayes / Dirichlet prior

| Concept | Reference | Field | Used by |
|---|---|---|---|
| Dirichlet process / Bayesian non-parametric prior | Ferguson, T. S. (1973). *A Bayesian analysis of some nonparametric problems*. *Annals of Statistics* 1(2):209–230. | Bayesian non-parametrics | `train/train_models.py::compute_edp_from_residuals`; EDP storage in `models_pkg['empirical_priors']` |
| Empirical Bayes prior estimation | Robbins, H. (1955). *An empirical Bayes approach to statistics*. Berkeley Symp. Math. Stat. Prob. 1:157–163. <br/>Robbins, H. (1983). *Some thoughts on empirical Bayes estimation*. *Annals of Statistics* 11(3):713–723. | Statistics | EDP derivation (no shrinkage, marginal frequencies) |
| Note (deliberate omission) | In-code comment inside `train/train_models.py::compute_edp_from_residuals`: "ne pas citer Efron & Morris 1973 ici — cet article porte sur les estimateurs de James-Stein à rétrécissement (shrinkage), non implémenté dans notre EDP." | Statistics | EDP design choice |

---

## 5. Anomaly detection / IDS theory

| Concept | Reference | Field | Used by |
|---|---|---|---|
| Anomaly detection holdout calibration | Ruff, L., et al. (2021). *A unifying review of deep learning-based anomaly detection*. *TPAMI*. | Anomaly detection | `train/train_models.py::_compute_training_proj_atk` for δ calibration |
| Test-set tuning ⇒ leakage | Varma, S., & Simon, R. (2006). *Bias in error estimation when using cross-validation for model selection*. *BMC Bioinformatics* 7:91. | ML methodology | PATCH-C2 / PATCH M-03 anti-leakage rationale |
| Likelihood ratio = sufficient statistic | Neyman, J., & Pearson, E. S. (1933). *On the problem of the most efficient tests of statistical hypotheses*. *Phil. Trans. R. Soc. A* 231:289–337. | Statistics | `_lr_novelty` rationale in `qualify/sbn_qualifier.py` |
| Reject option / residual class | Chow, C. K. (1970). *On optimum recognition error and reject tradeoff*. *IEEE Trans. Information Theory* 16(1):41–46. | Pattern recognition | `Autre_Anomalie` residual class |
| Naive Bayes robustness under moderate dependence | Domingos, P., & Pazzani, M. (1997). *On the optimality of the simple Bayesian classifier under zero-one loss*. *Machine Learning* 29(2-3):103–130. | Classification theory | rationale for evidence summation across groups in `_evidence_sum_scores` |
| Weight of evidence | Good, I. J. (1952). *Rational decisions*. *JRSS-B* 14:107–114. | Decision theory | underlying score interpretation in qualifier |
| Logarithmic opinion pooling | Genest, C., & Zidek, J. V. (1986). *Combining probability distributions: A critique and an annotated bibliography*. *Statistical Science* 1(1):114–135. | Forecasting / fusion | `qualify/sbn_qualifier.py::_compute_group_projected` (geometric mean) |
| Aczél–Daróczy axiomatic justification of geometric mean | Aczél, J., & Daróczy, Z. (1975). *On Measures of Information and Their Characterizations*. Academic Press. | Information theory | same as Genest-Zidek 1986 |
| Anomaly detection survey (independence of detectors) | Chandola, V., Banerjee, A., & Kumar, V. (2009). *Anomaly detection: A survey*. *ACM Computing Surveys* 41(3):1–58. | Anomaly detection | per-metric `c3_online_rmse` gating inside `core/opinions_pipeline.py::compute_opinions` |

---

## 6. Network attacks and signatures (qualifier templates)

| Concept | Reference | Field | Used by |
|---|---|---|---|
| Cyber Kill Chain (attack progression) | Hutchins, E. M., Cloppert, M. J., & Amin, R. M. (2011). *Intelligence-driven computer network defense informed by analysis of adversary campaigns*. *Lockheed Martin Corp.* | Threat modelling | `qualify/sbn_qualifier.py::_build_transition_matrix` (Markov priors) |
| MITRE ATT&CK | The MITRE Corporation (continuously updated). *MITRE ATT&CK®*. https://attack.mitre.org/ | Threat modelling | qualifier vocabulary: PORT_SCAN (T1046), DATA_EXFIL (T1048) |
| CIC-IDS2017 dataset (UDP/SYN/HTTP signatures) | Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A. A. (2018). *Toward generating a new intrusion detection dataset and intrusion traffic characterization*. *ICISSP 2018*. | Datasets | reference for `c^{k\|g}` opinions of UDP_FLOOD_DDOS, SYN_FLOOD_DDOS, HTTP_FLOOD_L7_DDOS |
| Kitsune (Slowloris, Port Scan profiles) | Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018). *Kitsune: An ensemble of autoencoders for online network intrusion detection*. *NDSS 2018*. | IDS | reference for SLOWLORIS_DOS, AGGRESSIVE_PORT_SCAN profiles |
| UNSW-NB15 dataset | Moustafa, N., & Slay, J. (2015). *UNSW-NB15: A comprehensive data set for network intrusion detection systems*. *MilCIS 2015*. | Datasets | reference for ICMP_FLOOD_BURST profile |
| DNS reflection BAF | Rossow, C. (2014). *Amplification hell: Revisiting network protocols for DDoS abuse*. *NDSS 2014*. | Network security | DNS_AMPLIFICATION qualifier template |
| NTP amplification (BAF=556.9) | van Rijswijk-Deij, R., Sperotto, A., & Pras, A. (2014). *DNSSEC and its potential for DDoS attacks*. APNIC/IMC. | Network security | NTP_AMPLIFICATION qualifier template |
| SSH brute-force flow signatures | Hofstede, R., et al. (2014). *Flow monitoring explained*. *IEEE Communications Surveys & Tutorials* 16(4). | NetFlow analysis | BRUTE_FORCE_SSH profile |
| DNS tunneling | (overview) e.g. Aiello et al. (2023). MDPI *Sensors* surveys (cited in code as "MDPI 2023"). | Covert channels | DNS_TUNNELING profile |
| Slowloris HTTP DoS | Hansen, R. (2009). *Slowloris HTTP DoS*. ha.ckers.org / RSnake. | DoS | SLOWLORIS_DOS profile |
| BotNet C&C beaconing | Garcia, S., Grill, M., Stiborek, J., & Zunino, A. (2014). *An empirical comparison of botnet detection methods*. *Computers & Security* 45:100–123. | Network security | BOTNET_CC_BEACONING profile |
| Pre-DDoS baseline activity | Benson, T., Akella, A., & Maltz, D. A. (2010). *Network traffic characteristics of data centers in the wild*. *IMC 2010*. | Traffic analysis | pre-DDoS activity flag in `audit/audit_full_dataset.py` |

---

## 7. Statistical inference helpers

| Concept | Reference | Field | Used by |
|---|---|---|---|
| Wilson score interval | Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical inference*. *JASA* 22(158):209–212. | Statistics | `audit/audit_full_dataset.py`; `evaluate/axelsson_ppv.py` |
| Wilson > Wald near 0/1 | Brown, L. D., Cai, T. T., & DasGupta, A. (2002). *Confidence intervals for a binomial proportion and asymptotic expansions*. *Annals of Statistics* 30(1):160–201. | Statistics | rationale for Wilson in proportions CI |
| BCa bootstrap (2nd-order accurate) | Efron, B. (1987). *Better bootstrap confidence intervals*. *JASA* 82(397):171–185. <br/>Efron, B., & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. CRC. | Resampling | `stats/bootstrap_ci.py::bootstrap_bca_ci`; jackknife acceleration in the same module |
| Block bootstrap (preferred for time series; *not* implemented) | Künsch, H. R. (1989). *The jackknife and the bootstrap for general stationary observations*. *Annals of Statistics* 17(3):1217–1241. | Resampling | mentioned in the audit's open-issue list (assumption A7.5) |
| Newey–West autocorrelation correction | Newey, W. K., & West, K. D. (1987). *A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix*. *Econometrica* 55(3):703–708. | Time-series stats | `stats/residual_correlation.py::newey_west_eff_n` |
| McNemar's test | McNemar, Q. (1947). *Note on the sampling error of the difference between correlated proportions or percentages*. *Psychometrika* 12(2):153–157. | Categorical statistics | `stats/mcnemar.py` |
| Continuity correction (small `n_disc`) | Pembury Smith, M. Q. R., & Ruxton, G. D. (2020). *Effective use of the McNemar test*. *Behavioral Ecology and Sociobiology* 74:133. | Statistics | `stats/mcnemar.py` (header docstring + small-`n` branch) |
| Variance Inflation Factor (VIF) | Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). *Regression Diagnostics*. Wiley. | Statistics | `stats/residual_correlation.py` (collinearity check) |
| Matthews correlation coefficient | Matthews, B. W. (1975). *Comparison of the predicted and observed secondary structure of T4 phage lysozyme*. *Biochim. Biophys. Acta* 405(2):442–451. | Classification metric | `evaluate/evaluate_qualify_sbn.py::_compute_global_detection_stats` |
| F-β IDS standard | Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). *A detailed analysis of the KDD CUP 99 data set*. *IEEE CISDA*. | IDS evaluation | F2 with β=2 in `evaluate/evaluate_qualify_sbn.py` |
| Range-based PR / recall (anomaly time-series) | Tatbul, N., Lee, T. J., Zdonik, S., et al. (2018). *Precision and recall for time series*. *NeurIPS 2018*. | Time-series AD | `vus_metrics.py` |
| Volume-Under-Surface (VUS) | Paparrizos, J., Boniol, P., Palpanas, T., et al. (2022). *Volume Under the Surface: A new accuracy evaluation measure for time-series anomaly detection*. *VLDB 2022*. | Time-series AD | `evaluate/vus_metrics.py` |
| Base-rate fallacy / PPV in IDS | Axelsson, S. (2000). *The base-rate fallacy and the difficulty of intrusion detection*. *ACM TISSEC* 3(3):186–205. | IDS evaluation | `evaluate/axelsson_ppv.py` (full module) |
| Sun et al. on anomaly thresholding | Sun, S. et al. (cited as "Sun et al. ICML 2024" in the module docstring of `paths.py`). | Anomaly detection | rationale for `proj_atk` quantile threshold |
| Ali et al. on threshold calibration | Ali, S., et al. (cited as "Ali et al. TISSEC 2013" in the module docstring of `paths.py`). | IDS | rationale cross-reference |

---

## 8. Robustness / unsupervised baselines

| Concept | Reference | Field | Used by |
|---|---|---|---|
| Isolation Forest | Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest*. *ICDM 2008*. | Unsupervised AD | `compare/compare_if_fair.py` |
| Shewhart control chart (z-score SPC) | Shewhart, W. A. (1931). *Economic control of quality of manufactured product*. Van Nostrand. | Statistical Process Control | baseline in `ablation/run_ablation_labeled.py` |

---

## 9. Internal cross-references (PATCH numbers, audit docs)

The codebase uses PATCH/TASK identifiers extensively to record audit decisions.
The most relevant for this scientific deconstruction are:

| ID | Date | What it changed |
|---|---|---|
| **PATCH M-01 / F01** | 2026-04-24 | Added `fusion_wbf_canonical_two` for literal Eq. 12.22-24 reproduction; clarified that `fusion_wbf_n_sources` is faithful to Eq. 12.27 (evidence-space). |
| **PATCH M-03** | 2026-04-21 | Switched to static EDP loaded from PKL; deprecated `adaptive_base_rate.py`. |
| **PATCH M-06 / F09** | 2026-04-21 | Accept partial last window (size < `WINDOW_SIZE`) with higher uncertainty rather than dropping. |
| **PATCH M-07 / F25** | 2026-04-21 | Forbid `fillna(0)` on metric columns; bounded `ffill(limit=10)` only. |
| **PATCH M-08 / F11** | 2026-04-21 | Configurable cap on dogmatic-evidence overflow `SL_EVIDENCE_MAX_FACTOR=1e4`. |
| **PATCH M-08 / F28** | 2026-04-21 | Full fallback audit log (`_FALLBACK_LOG` → `<model>_fallbacks.json`). |
| **PATCH M-11 / CBF** | 2026-04 | Default `INTER_METHOD_FUSION='wbf'` since CBF independence not satisfied. |
| **PATCH M-14** | — | BCa bootstrap replaces percentile bootstrap for biased statistics. |
| **PATCH-C1** | 2026-04-19 | Single canonical `INJECTED_ATTACK_CATALOG`; eliminate local copies. |
| **PATCH-C2** | 2026-04-18 | Remove Youden-on-test threshold from novelty AUC; reporting-only. |
| **PATCH C-01 / F02** | 2026-04-21 | Isolation Forest threshold calibrated on pre-split normal hold-out (was test-set leak). |
| **PATCH-m4** | 2026-04-18/19 | Document `BALANCE_RATIO` as explicit deviation from Theorem 12.2; recommend `hierarchical` as principled alternative. |
| **PATCH-m3** | 2026-04-18 | Hard fail (no silent fallback) if `FPR_TARGET_DECISION` missing from CONFIG. |
| **PATCH-M1** | 2026-04-18 | TimeSeriesSplit CV `R²` replaces in-sample `R²` (Stone 1974, HTF 2009 §7.10). |
| **PATCH TASK-20** | 2026-04-26 | Hard `FileNotFoundError` if injection requested but `_attacks.csv` missing (was silent fallback to non-injected). |
| **PATCH TASK-22** | 2026-04-26 | Calibration leak-free: thresholds calibrated on `df_train_calib` (out-of-sample) not on training residuals. |
| **PATCH TASK-23** | 2026-04-26 | Externalise calibration "magic numbers" (`CALIB_BIJECTION_FLOOR_TOL`, etc.). |
| **PATCH TASK-25** | — | Hardcoded CSV fallback removed (was silently evaluating wrong run). |
| **PATCH TASK-26** | 2026-04-26 | Rename `compute_conflict_degree → compute_asymmetric_escalation_conflict` to expose the asymmetric design; add `compute_conflict_degree_canonical`. |
| **PATCH TASK-27** | 2026-04-26 | `fusion_cbf` degenerate-denom fallback symmetric (was asymmetrically privileged op_A). |
| **PATCH TASK-28** | — | Targeted `warnings.filterwarnings`; never global ignore. |
| **PATCH TASK-40** | 2026-04-27 | STL fail policy `'raise'` in production; `'abstain'` for ablation. |
| **PATCH TASK-43** | 2026-04-27 | Update docstrings to reference active training entrypoint. |
| **PATCH TASK-44** | 2026-04-27 | `fusion_mode_at_compute_opinions.json` sidecar records actual fusion mode (column prefix `FINAL_SYSTEM_CBF` is historical). |
| **PATCH TASK-45** | 2026-04-27 | Threshold sidecar stores deployment configuration alongside δ for surrogate-vs-deployed cross-check. |
| **PATCH TASK-55** | 2026-05-07 | Inter-method fusion dispatch extended to WBF/ABF/CBF/BCF/projected-CCF/MinBF/MaxBF/hierarchical; method groups added; strict per-mode WBF/ABF recalibration kept WBF as RedeRio default. |

Companion audit documents (in `docs/`):
- `docs/AUDIT_CURRENT_STATUS.md`
- `docs/audit/audit_verification_tracker.md`
- `docs/audit/wu_keogh_self_assessment.md`
- `docs/audit/trust_discount_r2_analysis.md` *(documents the R²-pathology that motivates `WBF_WEIGHT_MODE='uniform'`)*
- `docs/DOCS_GOVERNANCE.md`
- `docs/honest_limitations.md`
- `docs/review/M10_sbn_architecture_analysis.md`
- `docs/review/FUSION_OPERATOR_ABLATION_20260506.md`
- superseded 2026-04 audit drafts are preserved under `docs/archive/2026-05-07_audit_cleanup/`
- `docs/review/PUBLICATION_TABLES.md`
- historical hardening/reorganisation notes are preserved under `docs/archive/`

---

## 10. Citation discipline (rules used in this audit)

1. **One canonical reference per concept.** When the codebase uses a textbook
   (e.g. Jøsang 2016, Coles 2001), the textbook is cited; the underlying
   primary source is added when relevant (e.g. Pickands 1975 for the GPD
   limit), but only one is used at the call site.
2. **No drive-by citations.** A reference appears here only if the codebase
   cites it in comments OR if the implementation is *demonstrably* derived
   from it (line-by-line correspondence). Tangential references mentioned by
   the agents but not actually used in code are not listed.
3. **Explicit deviations are flagged.** Where the codebase explicitly departs
   from a canonical reference (e.g. asymmetric escalation conflict vs Eq. 12.4;
   `BALANCE_RATIO` vs Theorem 12.2; trapezoidal evidence map without a
   canonical SL source), the deviation is named in the corresponding section
   so a reviewer can challenge it.
4. **Empirical-Bayes "Efron & Morris 1973 NOT cited"** is preserved as written
   in the source (the EDP is non-shrinkage marginal frequencies, not
   James-Stein).
