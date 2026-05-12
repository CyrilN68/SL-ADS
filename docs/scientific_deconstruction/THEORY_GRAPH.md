# THEORY_GRAPH.md — Conceptual Dependency Graph

> Three node families:
> - **🟢 Method** — concrete operator / algorithm in the codebase
> - **🟡 Theorem / Identity** — formal mathematical statement that justifies a method
> - **🔴 Assumption** — premise that must hold for the upstream theorem to apply
>
> Three edge labels:
> - `→ depends on` — the source method/theorem cannot be invoked without the target
> - `→ implies` — the source theorem produces the target identity / property
> - `→ justifies` — the source theorem licenses the use of the target method

---

## 1. Top-level dependency graph (Mermaid)

```mermaid
graph TD
    %% ---- Theorems / Identities ----
    T_BalkemaPickands["🟡 Pickands–Balkema–de Haan (1974/75)<br/>tail of an i.i.d. sequence converges to GPD"]
    T_GrimshawMLE["🟡 Grimshaw 1993 1D MLE<br/>g(θ) = θW(θ)(1+V(θ)) − V(θ) = 0"]
    T_ColesValidity["🟡 Coles 2001 §4.2<br/>σ̃ = σ − ξ·t₀ > 0  ⇒ GPD valid"]
    T_DavisonSmithDecluster["🟡 Davison–Smith 1990<br/>declustering for non-i.i.d. exceedances"]
    T_BijectionDef39["🟡 Jøsang 2016 Def. 3.9<br/>(b_i = r_i/(W+Σr), u = W/(W+Σr))"]
    T_ProjectedProb323["🟡 Jøsang 2016 Eq. 3.23<br/>P(x) = b(x) + a(x)·u"]
    T_UMaxEq327["🟡 Jøsang 2016 Eq. 3.27<br/>uncertainty maximisation preserves P(x)"]
    T_AgeingEq165["🟡 Jøsang 2016 Eq. 16.5<br/>R_{τ+1} = λ R_τ + r_{τ+1}"]
    T_BCFEq124["🟡 Jøsang 2016 Eq. 12.4 BCF<br/>K = Σ_{i≠j} b_A[i]·b_B[j]"]
    T_CBFEq1214["🟡 Jøsang 2016 Eq. 12.14 CBF"]
    T_CBFTheorem122["🟡 Jøsang 2016 Theorem 12.2<br/>CBF ≡ evidence sum (independence)"]
    T_ABFEq1220["🟡 Jøsang 2016 Eq. 12.20 ABF<br/>(dependent-source averaging)"]
    T_BCFEq1231["🟡 Jøsang 2016 Eq. 12.31 BCF<br/>(constraint fusion)"]
    T_CCFWang2017["🟡 Wang & Jøsang 2017 CCF<br/>(consensus + compromise)"]
    T_WBFEq1227["🟡 Jøsang 2016 Eq. 12.27 WBF<br/>(evidence-space confidence avg.)"]
    T_WBFEq1222["🟡 Jøsang 2016 Eq. 12.22-24 WBF<br/>(opinion-space, 2 sources, Case I/II)"]
    T_TrustDef146["🟡 Jøsang 2016 Def. 14.6<br/>probability-sensitive trust discount"]
    T_ContextualMercier["🟡 Mercier–Quost–Denoeux 2008<br/>α-vector contextual discount"]
    T_CoarseningJosang["🟡 Jøsang 2016 §3.5.4<br/>coarsening identity (5→3 states)"]
    T_DirichletFerguson["🟡 Ferguson 1973<br/>Dirichlet process / Bayesian prior"]
    T_RobbinsEB["🟡 Robbins 1955/1983<br/>Empirical Bayes prior estimation"]
    T_KoenkerBassett["🟡 Koenker–Bassett 1978<br/>LAD = QR(0.5), 50% breakdown response outliers"]
    T_RousseeuwLeroy["🟡 Rousseeuw–Leroy 1987<br/>LAD breakdown 0% on leverage outliers"]
    T_BridgmanDimensional["🟡 Bridgman 1922<br/>dimensional analysis / homogeneity"]
    T_TaylorLetham["🟡 Taylor–Letham 2018<br/>Prophet additive decomposition"]
    T_HyndmanMean["🟡 Hyndman–Athanasopoulos 2021 §3.1<br/>mean baseline forecast"]
    T_StoneCV["🟡 Stone 1974<br/>cross-validation removes in-sample bias"]
    T_RuffHoldout["🟡 Ruff et al. 2021<br/>holdout calibration for AD"]
    T_GenestZidekLOP["🟡 Genest–Zidek 1986<br/>logarithmic opinion pooling"]
    T_DomingosNB["🟡 Domingos–Pazzani 1997<br/>NB robust under moderate dependence"]
    T_HutchinsKillchain["🟡 Hutchins et al. 2011<br/>Cyber Kill Chain (Lockheed Martin)"]
    T_NeymanPearson["🟡 Neyman–Pearson 1933<br/>likelihood ratio = sufficient statistic"]
    T_ChowReject["🟡 Chow 1970 IEEE TIT<br/>reject option / residual class"]
    T_PaparrizosVUS["🟡 Paparrizos et al. 2022<br/>range-aware VUS-ROC / VUS-PR"]
    T_TatbulRange["🟡 Tatbul et al. 2018<br/>range-based AD precision/recall"]
    T_AxelssonPPV["🟡 Axelsson 2000<br/>base-rate fallacy / PPV"]
    T_WilsonCI["🟡 Wilson 1927<br/>score CI for proportions"]
    T_BrownCaiCIs["🟡 Brown–Cai–DasGupta 2002<br/>Wilson > Wald near 0/1"]
    T_EfronBCa["🟡 Efron 1987<br/>BCa bootstrap (2nd-order accurate)"]
    T_KunschBlock["🟡 Künsch 1989<br/>block bootstrap for time series"]
    T_NeweyWest["🟡 Newey–West 1987<br/>autocorrelation-corrected variance"]
    T_VarmaSimon["🟡 Varma–Simon 2006<br/>test-set tuning ⇒ leakage"]
    T_PembBino["🟡 Pembury Smith–Ruxton 2020<br/>exact binomial when n_disc<25"]
    T_LiuIF["🟡 Liu–Ting–Zhou 2008<br/>Isolation Forest (path-length AD)"]

    %% ---- Methods (the 🟢 nodes) ----
    M_Prophet["🟢 Prophet forecaster<br/>(growth='flat', seasonal+holidays)"]
    M_QR05["🟢 QuantileRegressor q=0.5<br/>(LAD, deterministic)"]
    M_DummyMean["🟢 DummyRegressor strategy='mean'<br/>(R²<0 fallback)"]
    M_GrimshawCode["🟢 _grimshaw_fit"]
    M_EvtPair["🟢 _evt_threshold / _evt_threshold_pair"]
    M_TrapezeMap["🟢 compute_instantaneous_evidence<br/>(piecewise linear, directional)"]
    M_EDP["🟢 compute_edp_from_residuals"]
    M_AutoCDAlpha["🟢 _auto_calibrate_reconst_reliability"]
    M_DecisionThr["🟢 _compute_training_proj_atk → δ"]
    M_Bijection["🟢 evidence_to_opinion / opinion_to_evidence"]
    M_UMax["🟢 MultinomialOpinion.uncertainty_maximized"]
    M_ConflictAsym["🟢 compute_asymmetric_escalation_conflict"]
    M_ConflictCanonical["🟢 compute_conflict_degree_canonical"]
    M_ConflictProj["🟢 compute_conflict_degree_projected"]
    M_ConflictKL["🟢 compute_conflict_degree_kl"]
    M_AdaptiveAgeing["🟢 temporal_adaptive_ageing"]
    M_WBFNary["🟢 fusion_wbf_n_sources<br/>(evidence-space, Eq. 12.27)"]
    M_WBFCanonical["🟢 fusion_wbf_canonical_two<br/>(opinion-space, Eq. 12.22-24)"]
    M_CBF["🟢 fusion_cbf"]
    M_ABF["🟢 fusion_abf_n_sources"]
    M_BCFusion["🟢 fusion_bcf_n_sources"]
    M_CCFusion["🟢 fusion_ccf_n_sources<br/>(projected)"]
    M_MinMaxBF["🟢 fusion_minbf_n_sources / fusion_maxbf_n_sources"]
    M_TrustDiscount["🟢 apply_trust_discount"]
    M_ContextualDiscount["🟢 apply_contextual_discount"]
    M_BoostEvidence["🟢 boost_opinion_evidence<br/>(BALANCE_RATIO heuristic)"]
    M_FivestateCoarse["🟢 5-state directional bookkeeping<br/>(_S_pos / _N_pos / _S_neg / _N_neg)"]
    M_SBNQualifier["🟢 qualify_anomaly_sbn (L1..L6)"]
    M_SBNGeomean["🟢 _compute_group_projected (geomean pool)"]
    M_SBNScore["🟢 _sbn_group_score (template match)"]
    M_SBNNovelty["🟢 _lr_novelty (concentration metric)"]
    M_SBNTransition["🟢 _build_transition_matrix"]
    M_VUS["🟢 vus_metrics (R-AUC, VUS-ROC/PR)"]
    M_AxelssonPPVCode["🟢 axelsson_ppv module"]
    M_BCaBoot["🟢 stats/bootstrap_ci.bca"]
    M_McNemar["🟢 stats/mcnemar"]
    M_ResidCorr["🟢 stats/residual_correlation"]
    M_IFFair["🟢 compare_if_fair (FPR-matched)"]
    M_AuditEpisode["🟢 audit_full_dataset"]

    %% ---- Critical assumptions (🔴) ----
    A_CleanTrain["🔴 A1.1 training span attack-free"]
    A_StationaryResidual["🔴 A1.5 zero-mean stationary residuals"]
    A_GPDExceeds["🔴 A1.2 σ̃>0 (Coles validity)"]
    A_DeclusterOff["🔴 A1.3 i.i.d. exceedances (decluster off)"]
    A_FlatGrowth["🔴 A1.6 no long-term trend"]
    A_R2Positive["🔴 A1.8 R²_CV ≥ 0 trust"]
    A_LeverageProtected["🔴 A1.7 fit_intercept=False ⇒ no leverage outliers"]
    A_OpinionInvariant["🔴 A4.1 Σb+u=1 (foundation)"]
    A_NaiveBayesQual["🔴 A6.1 Naive Bayes: P(g_1,g_2,…|k) ≈ ∏ P(g|k)"]
    A_ExpertPriorsCorrect["🔴 A6.3 SBN_COND_OPINIONS reflect attack truth"]
    A_DRY_Catalog["🔴 A7.7 INJECTED_ATTACK_CATALOG single source"]
    A_DeltaSurrogate["🔴 A1.9 surrogate ≈ deployed pipeline"]
    A_FixedDelta["🔴 A7.1 fixed δ on test"]
    A_NonInjectedNormal["🔴 A3.2 non-injected = normal"]
    A_InjAfterSplit["🔴 A3.5 catalog after split_date"]
    A_BinomialIID["🔴 A7.3 windows i.i.d. for Wilson CI"]
    A_BootstrapIID["🔴 A7.5 i.i.d. resampling for BCa"]

    %% ---- Edges: residuals → evidence ----
    T_TaylorLetham --> M_Prophet
    A_FlatGrowth --> M_Prophet
    M_Prophet -->|produces| residual_p["e_t = y_t - ŷ_t (Prophet)"]
    T_KoenkerBassett --> M_QR05
    T_RousseeuwLeroy --> M_QR05
    T_BridgmanDimensional --> M_QR05
    A_LeverageProtected --> M_QR05
    M_QR05 --> residual_r["e_t (Reconstruction)"]
    T_HyndmanMean --> M_DummyMean
    M_DummyMean --> residual_r

    %% ---- residuals → trapezoid → (P,S,N) ----
    residual_p --> M_TrapezeMap
    residual_r --> M_TrapezeMap
    A_StationaryResidual --> M_TrapezeMap
    M_TrapezeMap --> evidence_PSN["(P, S, N)_metric per window"]

    %% ---- thresholds for the trapezoid come from EVT ----
    T_BalkemaPickands -->|justifies| T_GrimshawMLE
    T_GrimshawMLE --> M_GrimshawCode
    T_ColesValidity --> M_EvtPair
    A_GPDExceeds --> M_EvtPair
    A_DeclusterOff --> M_EvtPair
    T_DavisonSmithDecluster -.->|optional| M_EvtPair
    M_GrimshawCode --> M_EvtPair
    A_CleanTrain --> M_EvtPair
    M_EvtPair --> thresholds["T_susp, T_atk, T_trapeze_base"]
    thresholds --> M_TrapezeMap

    %% ---- EDP ----
    T_DirichletFerguson --> M_EDP
    T_RobbinsEB --> M_EDP
    A_CleanTrain --> M_EDP
    M_TrapezeMap --> M_EDP
    M_EDP --> a_edp["a_edp per metric"]
    T_CoarseningJosang --> M_EDP

    %% ---- bijection foundation ----
    T_BijectionDef39 --> M_Bijection
    T_BijectionDef39 --> T_ProjectedProb323
    A_OpinionInvariant --> M_Bijection
    evidence_PSN --> M_Bijection
    a_edp --> M_Bijection
    M_Bijection --> opinion_temp["ω_temp = (b, u, a)"]

    %% ---- ageing ----
    T_AgeingEq165 --> M_AdaptiveAgeing
    T_BCFEq124 -->|extended (asym.)| M_ConflictAsym
    T_BCFEq124 --> M_ConflictCanonical
    T_ProjectedProb323 --> M_ConflictProj
    M_ConflictAsym --> M_AdaptiveAgeing
    M_ConflictCanonical -.->|alt| M_AdaptiveAgeing
    M_ConflictProj -.->|alt| M_AdaptiveAgeing
    M_ConflictKL -.->|alt| M_AdaptiveAgeing
    M_AdaptiveAgeing --> R_state["R_τ accumulated evidence"]
    R_state --> M_Bijection

    %% ---- WBF intra-method ----
    T_WBFEq1227 --> M_WBFNary
    T_WBFEq1222 -.->|literal 2-src| M_WBFCanonical
    opinion_temp --> M_WBFNary
    M_WBFNary --> ω_prophet["ω_prophet"]
    M_WBFNary --> ω_reconst["ω_reconst"]

    %% ---- contextual discount on Reconst ----
    T_ContextualMercier --> M_ContextualDiscount
    ω_reconst --> M_ContextualDiscount
    a_edp -.-> M_AutoCDAlpha
    M_AutoCDAlpha -->|α_attack| M_ContextualDiscount
    M_ContextualDiscount --> ω_reconst_d["ω_reconst'"]

    %% ---- inter-method fusion ----
    T_CBFEq1214 --> M_CBF
    T_CBFTheorem122 --> M_CBF
    T_WBFEq1227 -->|default WBF inter| M_WBFNary
    T_ABFEq1220 -->|tested, not default| M_ABF
    T_BCFEq1231 -->|stress-test| M_BCFusion
    T_CCFWang2017 -->|experimental| M_CCFusion
    M_BoostEvidence -.->|BALANCE_RATIO| M_CBF
    ω_prophet --> M_WBFNary
    ω_reconst_d --> M_WBFNary
    M_WBFNary --> ω_final["ω_final"]
    ω_prophet -.-> M_ABF
    ω_reconst_d -.-> M_ABF
    M_ABF -.-> ω_final
    ω_prophet -.-> M_BCFusion
    ω_reconst_d -.-> M_BCFusion
    M_BCFusion -.-> ω_final
    ω_prophet -.-> M_CCFusion
    ω_reconst_d -.-> M_CCFusion
    M_CCFusion -.-> ω_final
    ω_prophet -.-> M_MinMaxBF
    ω_reconst_d -.-> M_MinMaxBF
    M_MinMaxBF -.-> ω_final
    ω_prophet -.-> M_CBF
    ω_reconst_d -.-> M_CBF
    M_CBF --> ω_final

    %% ---- decision ----
    T_ProjectedProb323 --> M_DecisionThr
    A_DeltaSurrogate --> M_DecisionThr
    A_FixedDelta --> M_DecisionThr
    T_RuffHoldout --> M_DecisionThr
    T_StoneCV --> M_DecisionThr
    M_DecisionThr -->|δ| decision["D_τ = (proj_atk_τ ≥ δ)"]
    ω_final --> decision
    A_NonInjectedNormal --> decision
    A_InjAfterSplit --> decision

    %% ---- qualifier ----
    T_GenestZidekLOP --> M_SBNGeomean
    M_SBNGeomean --> M_SBNScore
    A_NaiveBayesQual --> M_SBNScore
    A_ExpertPriorsCorrect --> M_SBNScore
    M_SBNScore --> M_SBNQualifier
    T_BijectionDef39 --> M_SBNQualifier
    T_UMaxEq327 --> M_UMax
    M_UMax --> M_SBNQualifier
    T_HutchinsKillchain --> M_SBNTransition
    M_SBNTransition -.->|Markov prior, optional| M_SBNQualifier
    T_TrustDef146 --> M_TrustDiscount
    M_TrustDiscount -.->|temporal blend| M_SBNQualifier
    T_NeymanPearson --> M_SBNNovelty
    T_ChowReject --> M_SBNQualifier
    T_DomingosNB --> M_SBNQualifier
    M_SBNQualifier --> qualifier_out["top1, novelty, qual_status"]

    %% ---- evaluation ----
    decision --> eval_metrics["F1, FPR, MCC, F2, TTQ"]
    qualifier_out --> eval_metrics
    A_DRY_Catalog --> eval_metrics
    T_PaparrizosVUS --> M_VUS
    T_TatbulRange --> M_VUS
    M_VUS --> eval_metrics
    T_AxelssonPPV --> M_AxelssonPPVCode
    T_WilsonCI --> M_AxelssonPPVCode
    T_BrownCaiCIs --> M_AxelssonPPVCode
    A_BinomialIID --> M_AxelssonPPVCode
    T_EfronBCa --> M_BCaBoot
    A_BootstrapIID --> M_BCaBoot
    T_KunschBlock -.->|preferred| M_BCaBoot
    T_PembBino --> M_McNemar
    T_NeweyWest --> M_ResidCorr
    T_VarmaSimon --> A_FixedDelta
    T_LiuIF --> M_IFFair
```

> The Mermaid syntax above renders directly in GitLab/GitHub previews. The graph
> is intentionally dense; the table-form decomposition below makes individual
> chains explicit.

---

## 2. Reasoning chains in tabular form

### Chain A — Tail probability ⇒ EVT thresholds

| Hop | Node | Rule used |
|---|---|---|
| 1 | 🔴 i.i.d. residuals + heavy tail | premise |
| 2 | 🟡 Pickands–Balkema–de Haan (1974/75) | tail ⇒ GPD limit |
| 3 | 🟡 Grimshaw 1993 1D MLE | parameter estimation |
| 4 | 🟡 Coles 2001 §4.2 (σ̃>0) | validity check |
| 5 | 🟢 `_grimshaw_fit` | implementation |
| 6 | 🟢 `_evt_threshold_pair` | apply Siffer 2017 quantile formula |
| 7 | T_susp, T_atk per metric | output, fed to trapezoidal map |

Failure of step 1 (residuals not i.i.d., e.g. heavy autocorrelation): step 3 over-estimates `n_peaks`, biasing the GPD MLE. The codebase mitigates by relying on Prophet whitening (assumption A1.3).

### Chain B — Empirical evidence ⇒ Opinion

| Hop | Node | Rule |
|---|---|---|
| 1 | 🟢 `compute_instantaneous_evidence` | trapezoidal heuristic on signed residual |
| 2 | 🔴 trapezoidal triplet sums to 1 (constructive) | per-step |
| 3 | Σ over `n_window` ⇒ `(P,S,N)` with `Σ = n_window` | window aggregation |
| 4 | 🟡 Bijection Def. 3.9 | `b = r/(W+Σr); u = W/(W+Σr)` |
| 5 | 🟡 Eq. 3.23 | `proj_atk = b_atk + a_atk · u` |
| 6 | 🟢 `MultinomialOpinion(b, u, a)` | enforces `Σb+u=1` |

The trapezoidal heuristic is the **only non-Jøsang transformation** in the lower stack. It is a fuzzy mapping; its outputs do not have probabilistic meaning until they are aggregated and bijected (steps 3–5).

### Chain C — Conflict-Aware Ageing

| Hop | Node | Rule |
|---|---|---|
| 1 | 🟡 Eq. 16.5 | base ageing `R_{τ+1} = λ R_τ + r_{τ+1}` |
| 2 | 🟡 Eq. 12.4 BCF (multinomial extension) | `K = Σ_{i≠j} b_A[i]·b_B[j]` |
| 3 | 🟢 `compute_asymmetric_escalation_conflict` | omits de-escalation cross-products by design |
| 4 | engineering identity | `K_max = b_prev_max · b_curr_max ⇒ α := 1/K_max` |
| 5 | 🟢 `temporal_adaptive_ageing` | `λ_dyn = λ_base · (1 − αK)^γ` with `γ=1` |
| 6 | 🔴 escalation = "real anomaly", not noise | premise |

Step 6 is the load-bearing claim: the asymmetric design treats escalation as a *signal*, not a transient artefact. The pipeline ships canonical-symmetric (Eq. 12.4) and projected-probability variants for ablation.

### Chain D — Inter-method fusion (default)

| Hop | Node | Rule |
|---|---|---|
| 1 | 🔴 Prophet ⊥ Reconst is *not* satisfied | acknowledged |
| 2 | 🟡 Theorem 12.2 | CBF ≡ evidence sum **under independence** |
| 3 | 🟡 Eq. 12.27 WBF | confidence-weighted average of evidence (no independence required) |
| 4 | 🟢 `fusion_wbf_n_sources` | implementation |
| 5 | 🟡 Eq. 12.20 ABF | dependent-source comparison implemented, not adopted on RedeRio |
| 6 | 🟡 Mercier-Quost-Denoeux 2008 | contextual discount |
| 7 | 🟢 `apply_contextual_discount(ω_reconst, [1, 1, α])` | knock down `b_atk` of Reconst when blind |

The pipeline switches from CBF (canonical) to WBF (default `INTER_METHOD_FUSION='wbf'`) precisely because step 1 fails. ABF was added and strictly recalibrated as the main dependent-source alternative, but the 2026-05-07 RedeRio result kept WBF. CBF, BCF, projected CCF, MinBF, and MaxBF remain ablation/sensitivity knobs.

### Chain E — Decision threshold

| Hop | Node | Rule |
|---|---|---|
| 1 | 🟡 Stone 1974 / Ruff 2021 | hold-out calibration prevents in-sample bias |
| 2 | 🟢 `_compute_training_proj_atk` | aggregate evidence across metrics, bijection-back |
| 3 | 🔴 surrogate ≈ deployed pipeline (A1.9) | premise (caveat documented) |
| 4 | 🟡 Eq. 3.23 | `proj_atk` smooth and unimodal |
| 5 | 🟢 quantile `δ = quantile(proj_atk, 1 − FPR_target)` | empirical inverse-CDF |
| 6 | sidecar `<model>_threshold.json` | δ persisted, decoupling training from inference |

The quantile rule is empirical; the *FPR claim* depends on the surrogate matching the deployed chain. PATCH TASK-45 stores the deployment configuration in the sidecar so a downstream consistency check is possible.

### Chain F — Qualifier (cause attribution)

| Hop | Node | Rule |
|---|---|---|
| 1 | 🟡 Genest–Zidek 1986 LOP | geometric mean pool minimises KL |
| 2 | 🟢 `_compute_group_projected` | implementation |
| 3 | 🟡 Naive Bayes (Domingos & Pazzani 1997) | acceptable bias under moderate dependence |
| 4 | 🔴 expert priors `c^{k\|g}(s)` correct | A6.3 |
| 5 | 🟢 `_sbn_group_score`, evidence aggregation | template scoring |
| 6 | 🟡 Bijection Def. 3.9 | per-attack opinion |
| 7 | 🟡 Hutchins Kill Chain | Markov transition prior (optional) |
| 8 | 🟡 Chow 1970 / Jøsang §14.4 | residual class `Autre_Anomalie` for novelty |

The qualifier inherits *all* assumptions of the detection stack plus the additional A6.* premises.

### Chain G — Statistical inference

| Hop | Node | Rule |
|---|---|---|
| 1 | 🔴 windows i.i.d. (Wilson) / exchangeable (BCa) | premise |
| 2 | 🟡 Wilson 1927 / Brown–Cai–DasGupta 2002 | closed-form CI for proportions |
| 3 | 🟡 Efron 1987 BCa / Künsch 1989 block bootstrap | resampling-based CI |
| 4 | 🟡 Newey–West 1987 | autocorrelation-corrected variance |
| 5 | 🟢 `axelsson_ppv`, `bca`, `mcnemar` | implementations |

The pipeline implements (3) but not block bootstrap (Künsch); residual correlation is checked separately by `stats/residual_correlation.py`. A reviewer should flag whether headline CIs use the full effective-sample-size correction.

---

## 3. Cross-cutting "the chain breaks if…" annotations

| If this assumption fails | …then the following theorem stops applying | …and the headline number that breaks |
|---|---|---|
| A1.1 (clean train) | Pickands–Balkema-de-Haan tail estimation | `T_susp`, `T_atk`, EDP, δ — the entire calibration |
| A1.2 (Coles σ̃>0) | Grimshaw MLE | `T_susp`, `T_atk` for the affected metric (fallback empirical) |
| A1.3 (i.i.d. exceedances) | independence in tail estimation | EVT quantile formula |
| A1.6 (`growth='flat'`) | Prophet decomposition | residual stationarity downstream |
| A1.8 (R²_CV ≥ 0) | trust-discount weighting | Complete-run ablation collapse documented: Full `F1-cov=0.879` vs legacy R² trust-discount `F1-cov=0.628`, `12/14` attacks |
| A1.9 (surrogate≈deployed) | Ruff hold-out validity | δ and operational FPR |
| A3.5 (catalog after split) | calibration anti-leak | δ trustworthiness |
| A4.1 (Σb+u=1) | Jøsang Def. 3.1 | every SL operator |
| A6.1 (Naive Bayes qualifier) | Domingos-Pazzani robustness | qualification precision QP |
| A7.1 (fixed δ on test) | Varma–Simon anti-leakage | F1 trustworthiness |
| A7.7 (catalog single source) | DRY ground truth | entire evaluation |

---

## 4. Reading guide

To verify a specific claim:

- **"Our system has FPR ≈ 0.001"** ⇒ do not claim on the current complete run; realised global FPR is `0.965%` (`9.65×` target). Trace Chain E + verify A1.1, A1.9, A3.2 if a future run claims target-level FPR.
- **"Our F1 is X at fixed threshold"** ⇒ trace Chain B → C → D + verify A7.1, A7.7.
- **"Our qualifier achieves QP = Y"** ⇒ trace Chain F + verify A6.1–A6.5.
- **"Our system outperforms IF at matched FPR"** ⇒ trace `compare_if_fair` + verify A7.6 (PATCH C-01/F02).
- **"Our system is range-tolerant"** ⇒ trace `vus_metrics` + verify A7.2 (`L_max` reporting).

The graph above renders these chains in one place; the verification list in `PIPELINE_LOGIC.md §13` is the operational checklist.

---

## 5. Ordering of theoretical commitments (most fundamental → most peripheral)

1. **Σb + u = 1** (Jøsang Def. 3.1) — without this, nothing is well-defined.
2. **Bijection Def. 3.9** — translates evidence ↔ opinion.
3. **Eq. 3.23** — defines `proj_atk`, the decision variable.
4. **Eq. 16.5** — base ageing form (extended by the asymmetric Conflict-Aware variant).
5. **Eq. 12.27 WBF** — primary fusion operator (intra-method and default inter-method); ABF/BCF/CCF/min/max are ablation operators.
6. **Pickands-Balkema-de-Haan tail theorem** — justifies EVT calibration.
7. **Stone 1974 / Ruff 2021** — justifies hold-out calibration of δ.
8. **Genest-Zidek 1986** — justifies geometric-mean group pooling in qualifier.
9. **Mercier-Quost-Denoeux 2008** — justifies contextual discount.
10. **Hutchins Kill Chain (2011)** — informs (optional) Markov priors in qualifier.

A reviewer can audit the system in this order; failure at level *n* invalidates levels ≥ *n* but leaves levels < *n* intact. The most fundamental commitments (1–5) are checked by hundreds of unit tests in `tests/`; the more peripheral ones rely on documented design choices and ablation evidence.
