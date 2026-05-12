# Honest Limitations - Draft for Paper Section 5.3

**Status:** 2026-05-12, complete RedeRio 17-leaf rerun integrated
**Purpose:** drop-in draft for the paper's "Limitations" section.
The text below is written in third person, neutral tone, suitable
for direct insertion into the paper TEX file (with light copy-edit).

**Current numeric reference.** Paper-facing values come from run
`2e12261d55a8f975` / `resultats_RedeRio_trained_v4s_v4_v3`, completed
2026-05-12. The 2026-05-10 reconstruction-only diagnostic values are retained
only as audit provenance and are superseded wherever they conflict with the
following values: catalog/outages-separate F1 micro = 0.8666, operator-faithful
anomaly F1 micro = 0.8257, MCC = 0.8587, realised global FPR = 0.965%,
canonical ACTIVE FPR = 2.903%, 14/14 attacks detected.

---

## 5.3 Limitations

We acknowledge five limitations that bound the generalisability of
our results and the strength of the claims we make. They are listed
in roughly decreasing order of practical impact.

### 5.3.1 Synthetic anomaly injection

All attack scenarios in this work are *synthetically injected* into
the otherwise benign RedeRio capture (Section 3.1). This design
choice gives us perfect labels and full control over attack timing
and intensity, but exposes us to two well-documented benchmark
flaws (Wu and Keogh, 2021):

  - **Triviality / structural upper bounds.** Some volumetric or
    reconstruction-violating scenarios are easy for simple structural
    evidence scores. We therefore report the full per-episode table,
    same-evidence no-SL comparators, and the strong reconstruction-only
    diagnostic baseline instead of claiming that Subjective Logic
    universally dominates every simpler scalar rule.

  - **Distributional fingerprint.** Our default injection operates at
    the *evidence* level, after Prophet residuals have been computed.
    A reviewer may legitimately argue that this bypasses the realism
    constraint imposed by raw-traffic generation. The current public
    comparison therefore separates tasks explicitly: `compare_no_sl_fair.py`
    compares SL vs no-SL on the same evidence-level attack task, while
    `compare_raw_baselines_fair.py` trains IF / LOF / OCSVM / SGD-OCSVM
    / robust-z / PCA on raw metrics but excludes synthetic attack windows
    because those attacks do not exist in the raw traffic. The SL row in
    that table is generated from `opinions_non_injected/detection_results_RAW.csv`,
    not from the injected detection CSV. Raw-baseline numbers must not be
    cited as if they were evaluated on the synthetic catalog.

### 5.3.2 Single-dataset evaluation

We evaluate on a single Brazilian ISP capture (RedeRio). While the
volume and topological diversity of this dataset are competitive
with public alternatives (Sharafaldin et al., 2018; Mirsky et al.,
2018), we have not yet replicated our pipeline on the UCR Time
Series 2024 splits or the multimodal benchmark of Baldan et al.
(2025). Generalisability claims are therefore stated conditionally
on the RedeRio distribution. The codebase is dataset-agnostic
(`run_pipeline.py --dataset ...` accepts METR-LA, GECCO-IoT,
CESNET-TimeSeries24) and we expect the qualitative conclusions to
transfer, but the quantitative numbers in Section 4 should be read
as in-distribution estimates.

### 5.3.3 Subjective Logic fusion assumptions

The Weighted Belief Fusion (WBF) operator we use as the default
inter-method aggregation step (Joesang, 2016, Sec. 12.5.4) is not a
claim that Prophet and Reconstruction are independent in the strong
Cumulative Belief Fusion (CBF) sense. CBF remains a legacy ablation
operator because it adds evidence and can double-count correlated
sources. Our diagnostics explicitly show dependence: the 12 x 12
Prophet, 5 x 5 Reconstruction and 17 x 17 cross-method residual
matrices include high-correlation pairs, with cross-method maxima up
to |rho| = 0.915 on RedeRio.

For this reason the code now supports mode-specific inter-method
operators: `wbf`, `abf`, `cbf`, `bcf`, `ccf`, `minbf`, `maxbf` and
`hierarchical`. Averaging Belief Fusion (ABF) is the most conservative
theoretical candidate when sources are known to be dependent. However,
the strict recalibrated comparison of 2026-05-07 did not empirically
confirm an ABF switch on RedeRio: with one threshold sidecar per mode
and no ablation bypass, WBF obtained `F1=0.7057`, `MCC=0.7087`,
`FPR=4.31 %`, while ABF obtained `F1=0.7046`, `MCC=0.7077`,
`FPR=4.34 %`. We therefore keep WBF as the default and expose ABF as a
configurable alternative for datasets or future method groups where the
trade-off reverses.

In addition, we ship `WBF_WEIGHT_MODE = "uniform"` as the *default*
and reference weighting scheme.  The trust-discount alternative
(Joesang, 2016, Def. 14.6) using per-source R^2 as a proxy for
trust is implemented but **not enabled by default** because we
identified a pathological inversion empirically confirmed on the
RedeRio benchmark and refreshed on the complete run `2e12261d55a8f975`
(Full `F1-cov=0.879` vs legacy R² trust-discount `F1-cov=0.628`,
`12/14` attacks detected). The evidence is documented in
`docs/audit/trust_discount_r2_analysis.md`.

The pathology stems from R^2 being an inappropriate proxy for
source reliability when forecasting models *underfit* the most
informative streams.  Our training data contains 5 of 12 Prophet
models with negative training-time R^2:

| Metric                    | R^2 (train)  |
|---------------------------|--------------|
| `prophet_syn`             | -2.851       |
| `prophet_tcp`             | -1.526       |
| `prophet_flows`           | -0.645       |
| `prophet_avg_pkt_size`    | -0.395       |
| `prophet_entropy_src_ip`  | -0.111       |

These metrics are precisely those carrying low-volume attack
signatures (SYN flood ratio anomaly, slow connection burstiness,
entropy drift on source IP).  When `WBF_WEIGHT_MODE = "trust_discount"`
floors negative R^2 to a small constant `TRUST_SCORE_FLOOR = 0.05`,
the high-R^2 *volumetric* streams (`prophet_bytes` R^2=0.79,
`prophet_packets` R^2=0.76) dominate the fusion and noise out the
discriminative low-volume signal.  Empirical impact on the canonical
binary detection metric (F1-coverage, evidence-level injection,
14 attacks):

| Configuration                                | F1-cov | Det.   | FPR % |
|----------------------------------------------|--------|--------|-------|
| Uniform WBF reference, threshold 0.103       | 0.879  | 14/14  | 0.97  |
| Trust-Discount legacy, R^2 pathology         | 0.628  | 12/14  | 4.39  |
| MASE-Trust legacy                            | 0.000  | 0/14   | 0.00  |
| Reconst Only, structural upper-bound         | 0.923  | 14/14  | 0.43  |

Trust-discount loses 0.251 F1-cov vs uniform, drops 2 attacks, and
inflates FPR by 3.4 percentage points.  This is the empirical
confirmation of the analytic argument given in Joesang (2016)
Sec. 14.3: trust-discounting is theoretically sound *iff* the trust
proxy genuinely orders sources by reliability for the task at hand.
On unsupervised forecasting residuals, training-time R^2 reflects
*forecastability of benign traffic* — which is *anti-correlated*
with attack discriminability for low-volume scenarios.

We retain "Reconst Only" as an upper bound (it sidesteps the
forecasting branch entirely), but our reference is the full
17-metric Uniform Weights configuration because it preserves
volumetric coverage (UDP/SYN flood) that Reconst-Only cannot detect
without a structural relation violation.

**MASE evaluation (PATCH D5, 2026-05-07).** We implemented and tested
MASE (Mean Absolute Scaled Error, Hyndman & Koehler, 2006) as an
alternative trust proxy (`WBF_WEIGHT_MODE = 'mase'`,
`src/sl_ads/stats/mase.py`).  At the 30 s sampling rate of RedeRio,
the Naive-1 persistence baseline is dominant: most Prophet metrics
yield MASE > 1, i.e. predict the next sample *worse* than the
trivial persistence baseline.  Under the canonical α=1 Joesang
Def. 14.6 trust map `trust = max(floor, 1 − α·MASE)`, every Prophet
source is silenced down to the floor (TRUST_SCORE_FLOOR=0.05).  This
is the *opposite* failure mode of R²: R² over-trusts the volumetric
metrics that under-fit the discriminative low-volume signals; MASE
over-discounts *all* sources because persistence at 30 s is hard to
beat.  Crucially, the two proxies disagree on which metrics are
"trustworthy" — for example `prophet_fin` has the highest R² (0.81)
but a very poor MASE (3.53), while `prophet_avg_pkt_size` has
R²=−0.39 but MASE ≈ 0.95.  The two scores are uncorrelated on this
dataset.  Both R² and MASE measure *predictive* skill on benign
data, while what matters for anomaly detection is *discriminative*
behaviour when an anomaly arrives — a property neither proxy
captures.  We therefore reaffirm `WBF_WEIGHT_MODE = 'uniform'` as the
published default and expose `'mase'` only as an audit-grade
alternative whose pathology is empirically documented in
`docs/audit/trust_discount_r2_analysis.md` §4.1.

A discriminative trust proxy that addresses both pathologies — for
example a directional information score (mutual information between
residuals and a held-out labelled anomaly indicator) under
Mercier-Quost-Denoeux (2008) contextual discounting — is documented
as future work.  Such a proxy would require a small labelled
calibration split and is therefore not a strict drop-in replacement
for `'uniform'`, which has zero data requirements beyond benign
training.

### 5.3.4 Single-seed evaluation and run-to-failure risk

Our reported metrics are computed on a single deterministic injection
seed (Wu and Keogh's flaw #4). The infrastructure for multi-seed
evaluation is in place (`stats_bootstrap_ci.py` provides BCa CIs over
arbitrary point estimates), but we have not yet executed k=5 or k=10
seeded runs and folded the variance into our headline figures. A
reviewer who treats our F1 numbers as point estimates with implicit
uncertainty should mentally inflate them by ~ 1-2 percentage points
of standard error, which is the range observed in our preliminary
multi-seed sensitivity probes.

### 5.3.5 Baseline comparison scope

We now report two baseline families. First, `compare_no_sl_fair.py`
answers the central ablation question by removing the Subjective Logic
layer while keeping the same evidence and train-calib threshold policy.
Against the all-leaf no-SL mean-evidence comparator, Full SL-ADS improves
F1 micro from 0.8268 to 0.8666 and reduces FPR from 1.665 % to 0.965 %
with a paired block-bootstrap ΔF1 of +0.040 [0.011, 0.085]. Second,
`compare_raw_baselines_fair.py` trains IsolationForest, LOF novelty, exact RBF
One-Class SVM, SGDOneClassSVM with RBF features, robust-z, and PCA directly on
raw metrics and evaluates only raw-valid protocols
(pseudo-label agreement and real incidents, excluding synthetic
injection windows) against a non-injected SL output.

The measured gain over the all-leaf no-SL scalar is therefore real but
moderate. We do not present the additional Subjective Logic machinery as
justified by a large F1 jump alone. Its scientific value is the explicit
uncertainty representation, auditable fusion semantics, outage-aware
multi-source behaviour, and compatibility with downstream cause
qualification. The strong `no_sl_reconst_mean_N` RedeRio result is reported
as a diagnostic baseline and should be rerun on other datasets before making
cross-distribution claims.

The raw baselines are intentionally not evaluated on the 13 synthetic catalog
episodes. Those attacks are injected after residual computation, at the
evidence layer, and do not exist in the raw traffic. Testing raw IF / LOF / PCA
/ OCSVM on the synthetic catalog would require a separate raw-traffic injection
generator, followed by feature extraction and a full rerun.

On the three real-event intervals, the raw baselines mostly detect the DDoS and
miss the outages: IF covers 96.8% of the DDoS window, LOF/PCA/exact RBF-OCSVM
cover 100%, but they do not detect the two outage episodes. SL-ADS covers 96.8%
of the DDoS and partially covers both outages. Therefore, raw-baseline `1/3`
event counts should be interpreted as "DDoS detected, outages missed", not as
failure to detect the main attack.

This is sufficient for a defensible "with SL vs without SL" paper, but
not for a broad SOTA claim. We do not yet compare against Kitsune
(Mirsky et al., 2018), USAD/TranAD/TimesNet, or other deep TSAD models
under the same evidence-level catalog. Those baselines remain future work.

### 5.3.6 Limited attack diversity

The injection catalog covers 13 synthetic scenarios across
four attack families (DDoS volumetric, slow-rate DoS, reconnaissance,
amplification). It does not cover application-layer evasion, IPv6
fragmentation attacks, or recent ML-targeted adversarial inputs
(Athalye et al., 2018). The pipeline architecture admits straightforward
extension to new scenarios via the same catalog format, but the
empirical evaluation in Section 4 is bounded by the catalog as shipped.

---

## What this section is *not* admitting

To pre-empt over-interpretation, we note three concerns that have
been raised informally but for which we have evidence supporting our
position:

  1. **Canonical Subjective Logic compliance.** Section 3.5 documents
     - and `tests/test_fusion_wbf_canonical.py` numerically verifies -
     that our WBF implementation agrees with the canonical 2-source
     formula (Joesang, 2016, Eq. 12.22) to machine epsilon. This is
     not a limitation; it is a tested invariant.

  2. **Bijection b + u = 1.** Property-tested across 8 scenarios in
     the same test file; the bijection is preserved by every fusion
     operator we use.

  3. **Holiday calendar for Prophet.** The active training module
     `src/sl_ads/train/train_models.py` consumes the populated Brazil
     holiday list from `src/sl_ads/config.py`. We do not consider this
     a remaining limitation because `tests/test_holidays_brazil.py`
     verifies the national holidays in the 2025-10-13 to 2026-01-01
     RedeRio window.

---

## References (limitations-specific)

- Athalye, A., Carlini, N., & Wagner, D. (2018). "Obfuscated gradients
  give a false sense of security: circumventing defenses to adversarial
  examples." ICML.
- Baldan, D., Maggio, M., et al. (2025). "MUDEM: A multimodal benchmark
  for unsupervised anomaly detection." Pattern Recognition (in press).
- Bartos, K., Sofka, M., & Franc, V. (2016). "Optimized invariant
  representation of network traffic for detecting unseen malware
  variants." USENIX Security.
- Hyndman, R. J. & Koehler, A. B. (2006). "Another look at measures of
  forecast accuracy." International Journal of Forecasting 22 (4):
  679-688.
- Joesang, A. (2016). *Subjective Logic: A Formalism for Reasoning
  Under Uncertainty.* Springer.
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). "Isolation forest."
  ICDM.
- Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018).
  "Kitsune: an ensemble of autoencoders for online network intrusion
  detection." NDSS.
- Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A. A. (2018).
  "Toward generating a new intrusion detection dataset and intrusion
  traffic characterization." ICISSP.
- Wu, R. & Keogh, E. (2021). "Current Time Series Anomaly Detection
  Benchmarks are Flawed and are Creating the Illusion of Progress."
  IEEE TKDE 35 (3): 2421-2429.

---

## audit_codex 2026-04-26 disclosures (Phase G)

### CRIT-02: calibration surrogate vs deployed score (mitigated, residual FPR drift)

The decision threshold persisted in ``trained_models_*_threshold.json``
is calibrated on a **simplified** version of the deployed score path:
``train_models._compute_training_proj_atk()`` aggregates per-window
evidence, applies the SL bijection (Def. 3.9, Jøsang 2016) and
projected probability (Eq. 3.23) but does **not** replay (i) the
ageing operator with λ_dyn = λ_base·(1 − K_eff)^γ, (ii) the
trust-discount/contextual-discount stages, nor (iii) the inter-method
fusion stage (WBF / ABF / CBF / BCF / projected CCF / MinBF / MaxBF /
hierarchical).  Hence the calibrated
threshold is exact only when the deployed configuration matches the
metadata persisted in the sidecar.

**What we now persist** (PATCH TASK-45, 2026-04-27):
``fusion_mode_at_calibration``, ``wbf_weight_mode``, ``lambda_decay``,
``cd_alpha_attack``, ``balance_ratio``, and a textual
``calibration_surrogate_caveat`` field inside the sidecar.

**What is still pending**: a fresh production training/evaluation run
that validates a mode-specific sidecar as the deployed default. The
strict harness can already replay the ageing-aware method-fusion path
from persisted opinions for WBF/ABF comparisons, but the generic
production sidecar remains the reference unless a validated
mode-specific sidecar is selected.

Operational impact: non-negligible and explicitly disclosed. The
paper-facing complete run `2e12261d55a8f975` reaches realised global
FPR `0.965%` despite the holdout target `FPR_TARGET_DECISION=0.001`
(`0.1%`). The canonical ACTIVE regime reaches `2.903%`. This is
treated as calibration drift under regime shift and correlated
multi-metric exceedances, not as a hidden implementation mismatch.

2026-05-07 update: a stricter recalibration harness now writes
mode-specific threshold sidecars (for example
``trained_models_*_threshold_abf.json``) and can replay the ageing-aware
method fusion on the held-out normal split. This improves auditability
but also exposes calibration drift: the WBF/ABF strict comparison
reached `FPR=4.31-4.34 %` on the evaluation run despite the holdout
target `FPR_TARGET_DECISION=0.001`. Those sidecars are therefore treated
as historical ablation artefacts. They must not be cited as the final
RedeRio FPR; use the complete-run values above and in
`docs/review/PUBLICATION_TABLES.md`.

### MAJ-05: CESNET-TimeSeries24 synthetic timestamps

CESNET-TimeSeries24 distributes ``id_time`` as an integer counter
rather than a wall-clock timestamp.  ``cesnet_adapter.py`` synthesizes
a 10-minute step calendar from a fixed anchor (default 2024-01-01).

**Consequences.** Calendar-aware analyses are biased: Prophet weekly
seasonality is fit against the synthetic axis, hour-of-day is
aligned to the anchor's UTC offset, and any ablation that assumes
"weekday vs weekend" patterns must be considered indicative only.
The non-calendar-aware components (STL with explicit period= 144,
residual-modelling, Hampel filter, CUSUM) are unaffected.

**Mitigations** (PATCH TASK-46, 2026-04-27): the
``CESNET_TIMESTAMP_MODE`` config key now controls the policy — the
default ``fabricated_warning`` synthesizes timestamps and emits a
UserWarning at every load; ``reject`` refuses to load the dataset
(use this when calendar-aware analyses are mandatory).

### MAJ-09: FINAL_SYSTEM_CBF column prefix is historical

Despite the column-name prefix ``FINAL_SYSTEM_CBF``, the underlying
fusion operator depends on ``CONFIG['INTER_METHOD_FUSION']`` and may
be WBF, ABF, CBF, BCF, projected CCF, MinBF, MaxBF or hierarchical
equal-weight fusion.  The 31 downstream consumers were not migrated
because the column-rename is invasive and out of scope for the
publication patch window.

**Mitigation** (PATCH TASK-44, 2026-04-27): every
``sl_ads.core.opinions_pipeline`` run now writes
``fusion_mode_at_compute_opinions.json`` next to its CSV, recording
``actual_fusion_mode``, ``wbf_weight_mode``, ``balance_ratio``,
``cd_alpha_attack`` and ``lambda_decay``.  The helper
``paths.get_fusion_mode_for_run(OUTPUT_DIR)`` reads this metadata.
Reviewers can therefore verify which operator produced any given
artifact without consulting ``config.py``.

---

## 5.3.7 EVT calibration limits (complete run `2e12261d55a8f975`)

**Status note (2026-05-12).** Paper-facing values are refreshed from the
complete 17-leaf run. The realised global FPR is **0.965%** against a nominal
`FPR_TARGET_DECISION = 0.001` (0.1%), i.e. **9.65x** the target. On the
canonical regime partition, ACTIVE reaches **2.903%** FPR and QUIET reaches
**0.282%** FPR. These realised rates must be reported, not only the nominal
target.

The Peaks-Over-Threshold / Generalised Pareto Distribution (POT/GPD)
calibration of per-metric thresholds (Pickands 1975, Davison & Smith 1990)
underpins our directional anomaly thresholds `t_susp` (FPR target ≈ 1%)
and `t_atk` (FPR target ≈ 0.1%).  The implementation chain in
``src/sl_ads/train/train_models.py`` follows applied-EVT best practice:

1. **Grimshaw (1993) MLE** — primary GPD parameter estimator.  Same
   choice as the SPOT/DSPOT framework (Siffer et al. 2017, KDD), which
   is the reference EVT-anomaly algorithm.
2. **scipy.stats.genpareto.fit** — generic optimiser fallback if
   Grimshaw fails to converge.
3. **GPD validity check** — σ̃ = σ − ξ·t₀ > 0 (Coles 2001 §4.2);
   when violated, the GPD MLE is mathematically inadmissible at this
   threshold (heavy-tail with bounded-domain endpoint).
4. **Empirical quantile fallback** — last resort when (1)–(3) all fail.

We log every fallback event to a per-run audit trail
(``trained_models_*_fallbacks.json``, PATCH m-08/F28).  On the
2026-05-12 complete run, the fallback audit recorded **8 events across
2 fallback kinds** and **4/17 metrics violated the empirical FPR target**:

| Metric                  | FPR_susp_emp | FPR_atk_emp | Cible_susp | Cible_atk | Status   |
|-------------------------|--------------|-------------|------------|-----------|----------|
| `prophet_flows`         | 0.0834       | 0.00139     | 0.0100     | 0.0010    | ⚠ HIGH   |
| `prophet_syn`           | 0.1158       | 0.02685     | 0.0100     | 0.0010    | ⚠ HIGH   |
| `prophet_tcp`           | 0.0383       | 0.00901     | 0.0100     | 0.0010    | ⚠ HIGH   |
| `prophet_udp`           | 0.1774       | 0.15608     | 0.0100     | 0.0010    | ⚠ HIGH (×156)|

The four metrics with FPR violations are among the metrics with
training R^2 < 0 (Section 5.3.3 above).  This is not a coincidence:
Prophet's residual variance is artificially compressed when the model
underfits, producing a left-truncated empirical distribution that the
GPD MLE cannot honestly extrapolate.  The σ̃ ≤ 0 fallback then
reduces the threshold to the empirical quantile — defensible, but
biased toward over-generation of `t_susp` exceedances.

**Why we keep the chain as-is.**  Three reasons:

  - The pathology is *intrinsic* to forecasting models with R^2 < 0,
    not to the EVT layer itself.  Replacing Grimshaw with PWM
    (Hosking & Wallis 1987) or with profile-likelihood
    (Smith 1985) does not fix a malformed residual distribution.
  - The fallback to empirical quantile is *conservative*: it accepts
    a slightly inflated empirical FPR rather than producing a
    GPD-extrapolated quantile that would have larger systematic error.
  - The downstream Subjective Logic fusion (proj_atk) integrates 17
    metrics; a single metric with degraded calibration moves the
    fused threshold by < 0.04 (Phase G calibration audit, see CRIT-02
    above), which is within the EVT confidence band.

**Future-work alternatives** (post-publication backlog):

  - **Probability-Weighted Moments** (Hosking & Wallis 1987,
    *Technometrics* 29(3)) as intermediate fallback before the
    empirical quantile.  More stable than MLE for n_peaks < 100.
  - **Mean Residual Life (MRL) plot** for adaptive `t₀`
    selection (Coles 2001 §4.3.4) instead of the fixed 90th
    percentile we currently use.
  - **Bayesian POT** with informative priors on ξ (Stephenson 2002,
    *Extremes* 5(2)) for credible intervals on the persisted
    quantiles.

**Calendar-aware EVT (PATCH H2 — Phase B, 2026-05-07).**  The single
global EVT threshold above produces the documented FPR overshoot on
weekday-daytime traffic. On the 2026-05-12 complete run, the canonical
2-bucket partition
``ACTIVE = weekday × not(holiday) × hour ∈ [08, 18)`` vs ``QUIET =
everything else`` gives `canonical_ACTIVE` FPR = **2.903 %** (29.03×
the 0.1 % target) and `canonical_QUIET` FPR = **0.282 %** (2.82×).
The annualised projection is roughly **1007 false-positive windows/year**
versus **105/year** under the nominal budget, assuming the audited regime mix
is representative.

We have implemented per-regime EVT calibration:
``sl_ads/calendar/regime.py`` (canonical partition + versioned
signature ``weekday_x_daytime_x_holiday/v1@2026-05-07``);
``train_models.calibrate_thresholds_per_regime_v2`` (per-bucket EVT
calibration using the same Grimshaw → PWM → empirical fallback chain
as the global path); regime dispatch in ``train.compute_evidence``;
sidecar A1.9 strict on a new ``calendar_evt_signature`` field that
hard-fails when the calibration-time and runtime regime functions
diverge (e.g. moved daytime hour boundaries, different holiday
calendar).  The 22 unit tests in ``tests/test_calendar_aware_evt.py``
cover boundary semantics, vectorisation parity, per-regime
calibration recovery on synthetic GPD samples, dispatcher
invariants, and sidecar backward compatibility (old PKLs still load
as long as no signature drift is claimed).

The default ``CONFIG['CALENDAR_EVT_ENABLED'] = False`` is preserved for the
current paper. The complete-run root-cause script
``sl_ads.audit.regime_fpr_diagnosis`` identifies `H_correlation`, not pure
per-metric threshold heteroscedasticity: joint k=3 exceedances are 4.524×
higher on ACTIVE than QUIET. Calendar-aware EVT is therefore an audit-grade
mechanism/future-work path, not the published reference. Design rationale and
full sidecar API: ``docs/review/calendar_evt_design.md``.

---

## 5.3.8 Qualifier confusion matrix — intrinsic catalog limits

The 13-class Subjective Bayesian Network qualifier (Section 4) reaches
DR_macro = 91.1% / QP_macro = 67.6% / F1_macro = 65.0% on the
2026-05-12 complete run.  The DR-QP gap is concentrated on **three attack
families that share evidence signatures with classes already in the
catalog**.  These are not implementation bugs; they are
*representational* limits of the closed 13-class taxonomy.

| Attack injected         | Top-1 incorrect  | Confused with (top hit) | Cause                                                                                           |
|-------------------------|------------------|--------------------------|-------------------------------------------------------------------------------------------------|
| `BOTNET_CC_BEACONING`   | `PORT_SCAN` (36) | `PORT_SCAN`              | C2 beaconing has the same low-bytes / high-flows / high-entropy signature as port scanning      |
| `DNS_TUNNELING`         | `PORT_SCAN` (54) | `PORT_SCAN`, `BOTNET_CC` | DNS tunneling is multi-class by construction — it borrows from scan, beaconing, and exfiltration |
| `DNS_AMPLIFICATION`     | `NTP_AMP` (35)   | `NTP_AMP`                | UDP-based amplification with high BPC ratio is signature-equivalent between DNS and NTP         |

**Reviewer-facing framing.**  Three observations are simultaneously
true and must be stated together to avoid both over- and
under-claiming:

  1. The catalog is a *closed-world* taxonomy of 13 attack types
     (Section 3.3, `config.py:946-989`).  Within that closed world,
     three pairs of attack types share evidence-level signatures.
     The qualifier therefore cannot disambiguate them without
     additional discriminating features (e.g., DNS-port-specific
     `dst_port_entropy` for DNS_AMP vs NTP_AMP).
  2. The qualifier *does* detect that an anomaly has occurred —
     these three attacks are consistently flagged at the detection
     stage with high `proj_atk`.  The failure is purely on the
     *type* assignment, not on the binary-detection performance.
  3. Adding discriminating features would be a content-bearing
     extension of the system rather than a fix.  In Section 5.3.6
     we list this as future work (DNS-specific entropy, IPv6
     fragmentation features, application-layer telemetry).

**Numerical signature.** On the 14-attack injection set:

  - 8/12 known synthetic attack types reach QP > 50% (assigned to
    the right class on at least half the detected windows).
  - 3/12 attacks (BOTNET_CC, DNS_TUNNELING, DNS_AMP) reach
    QP = 0% on the SBN qualifier; BRUTE_FORCE_SSH is detected but
    mostly confused with PORT_SCAN (QP = 11.8%).
  - The argmax naïve-Bayes baseline does *better* than the SBN on
    these three (Table 2 in `PUBLICATION_TABLES.md`), at the cost
    of losing the uncertainty channel `u_sbn`.

This trade-off is reported transparently as the H-F2 hypothesis
(now consolidated into `docs/scientific_deconstruction/ASSUMPTIONS.md`;
the original threat catalog is archived). The resolution
in v11 of the system will be a two-step qualifier: SBN for
uncertainty-aware confidence + targeted feature extension for the
three confused families.

---

## 5.3.9 Methodological gap between `compare_if` and `eval_injection`

We report two binary-detection F1 scores in different sections of
the paper that look numerically incompatible:

  - `eval_injection`: SL-ADS  F1 = 0.867 (canonical, reference
    label set — 14 curated attacks).
  - `compare_if`:    SL-ADS  F1 ≈ 0.181, IF-fpr-matched F1 ≈ 0.349
    (raw label column — pseudo-anomalies via STL+Hampel+CUSUM
    consensus, 30 709 / 211 417 positives).

**These two F1 values measure different things and are not directly
comparable.**  The clarification belongs in the paper (Section 4.0,
evaluation methodology) and in the `compare_if` script header.

The labels used in the two evaluations are deliberately different:

  - `eval_injection` measures detection on the **expert-curated
    attack catalog**: 14 well-defined attack episodes injected into
    the evidence stream (`config.INJECTED_ATTACK_CATALOG`) plus 1
    real DDOS episode.  Total positives ≈ 1 100 windows out of
    ~13 000 windows on the 5-minute decision unit.  SL-ADS is
    *calibrated to detect these episodes*, the F1 here measures
    in-distribution detection performance.
  - `compare_if` measures detection on the **unsupervised consensus
    pseudo-labels**: any window flagged by ≥2/3 of (STL residual
    z-score, Hampel filter, CUSUM change-point) is positive
    (`rederio_adapter.apply_pseudo_labels`).  Total positives ≈
    30 709 windows = 14.5% of the dataset.  Many are noise events
    (network bursts, end-of-day flushes) that SL-ADS deliberately
    does *not* flag because they don't resemble attack-shaped
    residual surges.

The disagreement between SL-ADS and the consensus pseudo-labels is
*expected*: the two label sources answer different questions
("Is this an attack?" vs "Is this an unusual time slice?").
Reporting both is informative because it bounds the operating
range of the system.  Reporting them on the same axis without the
caveat would be misleading.

**Operational clarification we will add to the paper:**

> §4.0 paragraph: *"We report two complementary detection scores:
> the canonical F1 against the curated attack catalog
> (Section 4.1, F1 = 0.867) measures the system's ability to flag
> known attack patterns; the comparison F1 against the unsupervised
> consensus pseudo-labels (Appendix D, F1 ≈ 0.16) measures how
> often the system agrees with statistical change-point detectors.
> The two scores are not directly comparable: the curated catalog
> contains 14 expert-defined episodes, whereas the consensus
> pseudo-labels include any window flagged by ≥ 2/3 of
> STL/Hampel/CUSUM, ≈ 14.5% of the dataset.  Isolation Forest
> agrees with the consensus pseudo-labels more often (F1 ≈ 0.17)
> because it inherits the same noise-flagging behaviour, not
> because it detects more attacks."*

**McNemar paired tests** (Dietterich 1998) on the 13 078 5-min
decision windows produce significant disagreement in all three
regimes (p < 0.0001 each):

  - SL vs IF-fair-window:    statistic = 1773  -> significant
  - SL vs IF-fpr-matched:    statistic = 688   -> significant
  - SL vs IF-k1-descriptive: statistic = 1953  -> significant

The McNemar test only counts disagreement, not which detector is
correct; on the *catalog labels* SL-ADS dominates IF
(F1 0.867, 14/14 episodes detected).  On the *pseudo-labels*,
IF-fpr-matched agrees with the pseudo-labels more often than SL-ADS
(F1 0.349 vs 0.181); this is consistent with the methodology gap above
and should not be framed as superior attack detection.

**Improvement we plan to add post-publication.**  An FPR-matched
SL-ADS evaluation against the same pseudo-labels (analogous to the
existing IF-fpr-matched protocol).  This would equalise operating
points and let us report a clean "best F1 at FPR=1.85%" for both
detectors against the same label source.  The infrastructure is
already in place via the calibration helper
``compare_if_fair._calibrate_if_threshold_from_normal``;
extending it to SL is listed in the open-work section of
``docs/AUDIT_CURRENT_STATUS.md``.

---

## 5.3.10 Slowloris persistent gap (65.6% recall, 90 min TTD)

Slowloris (RFC 7230 slow-HTTP-DoS) is the hardest attack family
in our catalog.  On the 8-hour injected episode the complete run reaches
65.6% time-window coverage (63/96 windows) with 18-window time-to-detect
(90 minutes from attack onset).  Detection is partial both in *coverage*
and in *latency*.  This is a **persistent gap**, documented to set
operational expectations and to justify the pipeline-level mitigations
discussed in Section 3.4.

**Why Slowloris is hard.**  By design, Slowloris keeps connection-level
volumes (bytes, packets, flows) within seasonal bands.  The attack
signature is a pattern of *unfinished* connections (FIN/SYN ratio
collapse, slow header trickle), not a volumetric surge.  Three of our
17 metrics carry the relevant signal — `prophet_fin`, `reconst_fin_from_syn`,
`prophet_entropy_src_ip` — and all three have weak forecastability
(R^2 < 0.25, see Section 5.3.3) because their benign baseline is
itself bursty.

**Mitigation status.**  The current RedeRio reference configuration ships
uniform WBF with `CD_ALPHA_ATTACK = 1.0`; no contextual-discount alpha is
claimed as a production fix. Contextual-discount variants remain ablation
and future-work evidence only: in the harmonised complete-run ablation they
do not improve the publication operating point enough to justify a default
change. ABF is available for dependent-source experiments but is not the
RedeRio default after strict recalibration.

**What is left as a known limitation.** Even with the mitigations,
we do not reach 80% recall on Slowloris in the configuration we
recommend for publication.  The remaining one-third of windows correspond
to the slow-onset phase (first 2 hours) where the conflict-aware
ageing operator is still resetting the long-memory state from
prior benign windows.  Reducing the persistence parameter `λ` from
0.85 to 0.50 would close this gap on Slowloris but degrade other
attacks (cf. ablation `λ=0.50`).  We elect to keep λ=0.85 as the
operational default.

This persistent gap is the principal threat-to-validity bound on
our headline F1.  We report it explicitly because under-reporting
would mislead operators about the *time-to-detect* performance on
slow-rate DoS, which is the family with the highest operational
relevance for academic-network defence.

---

## 5.3.11 Network outages as operator-relevant anomalies

**Status note (2026-05-12).** Complete-run outage metrics are now available.
They replace the 2026-05-10 reconstruction-only post-mortem values. Outages
are reported in two ways: separately for catalog comparability, and as positive
anomaly windows in the operator-faithful F1 protocol.

Complete-run outage recall:

| Event | Recall |
|---|---:|
| `NETWORK_OUTAGE_NOV17` | 1/3 = 33.3% |
| `NETWORK_OUTAGE_DEC1617` | 188/339 = 55.5% |
| `REAL_DDOS` / `DDOS_ATTACK` | 184/190 = 96.8% |

Protocol impact:

| Protocol | F1 micro | Positives | Meaning |
|---|---:|---:|---|
| `catalog_outages_separate` | 0.8666 | 721 | catalog/injection attacks; outages separate |
| `operator_faithful_anomaly` | 0.8257 | 1063 | catalog + real incidents, including outages |

The scientific conclusion is not that outages should be hidden from F1. The
conclusion is that both definitions answer different questions and both should
be reported.

The outage result is a genuine limitation: the long December outage is only
partially detected, and the short November outage has only one detected window.
This lowers the operator-faithful anomaly F1 and should be discussed as an
operational trade-off of the current threshold/fusion configuration.

Earlier versions of this section (pre-2026-05-07) attributed the
miss to a "cold-start of the conflict-aware ageing operator" within
the "first 24 hours after the train/test split" and described the
event as one of "inertial smoothing".  All three claims were
incorrect: the event is at +7.5 days (not 24 h), the system has long
converged at that point, and the proj_atk trace shows a local
sub-threshold response rather than inertial smoothing.  The corrected
wording above reflects the actual run trace.

---

## 5.3.12 AUC novelty_lr in-sample = 0.654 (PATCH-C2 reporting only)

The novelty-LR head (logistic regression on the qualifier residual,
see Section 4.5) reports an AUC-ROC of 0.654 on the current evaluation split.
This number must be read with two cautions:

  - It is *in-sample* (computed on the same windows used to fit
    the LR).  The asymptotic optimism of in-sample AUC is small
    for k-fold-validated linear models (Hanley & McNeil 1982,
    *Radiology* 143) but non-zero.  A held-out estimate would be
    nominally 0.02-0.05 lower (Hastie, Tibshirani & Friedman 2009,
    Ch. 7).
  - It is *reporting-only*.  Following PATCH-C2 (audit reconciliation
    2026-04-21), the novelty-LR threshold `LR_NOVELTY_THR` is
    persisted as `None` in the configuration.  No downstream
    decision uses the LR score to binarise; we expose the AUC for
    transparency about the *novelty channel* in the qualifier and
    nothing more.

This avoids the threshold-leakage concern of Varma & Simon (2006,
*BMC Bioinformatics* 7) and Japkowicz & Shah (2011) by construction:
no test-derived threshold is selected.  The 0.654 figure is reported
in Section 4.5 with the in-sample qualification and the explicit
non-decision use-case.

We also note Youden's J (Youden 1950, *Cancer* 3) on the same head:
the in-sample cut-point is 0.715, but it is not used operationally. This is consistent with a
"signal-present but weak" novelty channel, appropriate for
*monitoring* but not for *gating* decisions.
