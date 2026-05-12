# Publication Tables - Current Canonical Inputs

**Status date:** 2026-05-12  
**Scope:** paper-facing table inputs for the complete RedeRio 17-leaf run.

Canonical run:

| Item | Value |
|---|---|
| Run id | `2e12261d55a8f975` |
| Source results | `../results/resultats_RedeRio_trained_v4s_v4_v3/` |
| Archived mirror | `current_version/results/2e12261d55a8f975/` |
| Launcher | `run_pipeline.py` |
| Pipeline status | 10/10 steps OK, exit code 0 |
| Runtime | 582m28s |
| Evidence windows | 13,078 |
| Detection column | `FINAL_SYSTEM_CBF_proj_atk` |
| Fusion mode | `wbf` with uniform weights |
| Decision threshold | 0.102614 from sidecar |

The 2026-05-10 reconstruction-only diagnostic run is superseded for paper
numbers. Do not cite its F1/FPR values as current results.

## 1. Detection Headline

Artifact:
`evaluation/eval_threshold_sweep.csv`

| Metric | Value |
|---|---:|
| F1 micro, pure window | 0.8666 |
| F1 macro, pure window | 0.9292 |
| F1 micro 95% CI, moving-block BCa | [0.7600, 0.9232] |
| MCC | 0.8587 |
| MCC 95% CI, moving-block BCa | [0.7445, 0.9155] |
| Precision, window | 0.8466 |
| TPR, window | 0.8877 |
| FPR, window | 0.0097 |
| FPR, percent | 0.97% |
| FPR target | 0.10% |
| Realised / target FPR ratio | 9.65x |
| Attacks detected | 14/14 |
| Coverage recall, episode mean | 0.9150 |
| VUS-PR / VUS-ROC | 0.6038 / 0.8562 |

Paper wording: the system detects all catalogued episodes, but the realised
false-positive rate exceeds the nominal 0.1% target and must be disclosed.

## 2. F1 Protocol Comparison

Artifact:
`evaluation/eval_f1_protocol_comparison.csv`

| Protocol | Positives | TP | FP | FN | F1 micro | F1 macro | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `catalog_outages_separate` | 721 | 640 | 116 | 81 | 0.8666 | 0.9292 | 0.8877 | 0.965% |
| `operator_faithful_anomaly` | 1063 | 829 | 116 | 234 | 0.8257 | 0.9056 | 0.7799 | 0.965% |

Paper wording: report both. The first is best for comparability with the
synthetic/catalog attack protocol; the second is more operator-faithful because
network outages are treated as anomalies.

## 3. Per-Episode Detection Table

Artifact:
`evaluation/eval_detection_summary.csv`

| Episode | Coverage | TTD | Max P(atk) |
|---|---:|---:|---:|
| AGGRESSIVE_PORT_SCAN | 100.0% | 0 min | 0.326 |
| BOTNET_CC_BEACONING | 81.2% | 30 min | 0.201 |
| BRUTE_FORCE_SSH | 94.4% | 10 min | 0.344 |
| DATA_EXFILTRATION_SLOW | 76.4% | 50 min | 0.203 |
| DNS_AMPLIFICATION | 97.2% | 5 min | 0.409 |
| DNS_TUNNELING | 90.3% | 25 min | 0.338 |
| HTTP_FLOOD_L7_DDOS | 94.4% | 5 min | 0.378 |
| ICMP_FLOOD_BURST | 100.0% | 0 min | 0.359 |
| NTP_AMPLIFICATION | 97.2% | 5 min | 0.453 |
| REAL_DDOS | 96.8% | 14 min | 0.295 |
| SLOWLORIS_DOS | 65.6% | 90 min | 0.217 |
| SYN_FLOOD_DDOS | 100.0% | 0 min | 0.398 |
| UDP_FLOOD_DDOS | 95.8% | 10 min | 0.432 |
| UNKNOWN_ANOMALY_CONTROL | 91.7% | 10 min | 0.211 |

Paper wording: Slowloris remains the hardest catalogued episode and should be
kept in the limitations section.

## 4. Outage Bucket

Artifact:
`audit_episodes.csv` plus the console audit summary.

| Event | Recall |
|---|---:|
| REAL_DDOS / DDOS_ATTACK | 184/190 = 96.8% |
| NETWORK_OUTAGE_DEC1617 | 188/339 = 55.5% |
| NETWORK_OUTAGE_NOV17 | 1/3 = 33.3% |

These are not hidden false positives. They are known real incidents; report
them separately and include them in the operator-faithful F1 protocol.

## 5. Regime-FPR

Artifacts:
`outputs/scientific_hardening/regime_fpr.csv` and
`outputs/scientific_hardening/regime_fpr_diagnosis.{csv,json,md}`

| Regime | Normal windows | FP | FPR | Ratio to 0.1% target |
|---|---:|---:|---:|---:|
| all_normal | 12,015 | 116 | 0.965% | 9.65x |
| canonical_ACTIVE | 3,135 | 91 | 2.903% | 29.03x |
| canonical_QUIET | 8,880 | 25 | 0.282% | 2.82x |
| day_08_18 | 4,867 | 91 | 1.870% | 18.70x |
| weekend | 3,366 | 1 | 0.030% | 0.30x |
| holiday_or_closure | 1,261 | 1 | 0.079% | 0.79x |

Root-cause verdict from the complete run: `H_correlation`. Individual
per-metric exceedance rates are not the main mechanism; joint k=3 exceedances
are 4.524x higher on ACTIVE than QUIET, so correlated benign daytime bursts are
amplified by fusion.

## 6. Qualification

Artifact:
`eval_qualify_summary_qualif_types_sbn_20260512_003447.json`

| Metric | Macro | Micro |
|---|---:|---:|
| DR, detection rate | 91.1% | 85.6% |
| QP, qualification precision | 67.6% | 61.1% |
| F1, DR/QP harmonic | 65.0% | 71.3% |
| F2, DR-prioritised | 66.4% | 64.8% |

Known qualifier failures: `BOTNET_CC_BEACONING`, `DNS_TUNNELING`, and
`DNS_AMPLIFICATION` are detected but assigned to a neighboring family. This is
a closed-taxonomy feature limitation, not a binary detection failure.

Novelty channel: `UNKNOWN_ANOMALY_CONTROL` detection rate is 91.7%;
`novelty_lr` AUC is 0.654, in-sample and reporting-only. Do not use the Youden
threshold as an operating decision.

## 7. Ablation And Baselines

Artifact:
`ablation_uniform/ablation_summary.csv`

`ablation_summary.csv` contains one calibrated-threshold headline row per run.
Threshold-grid rows are kept separately in
`ablation_uniform/ablation_threshold_sensitivity.csv` and must be cited only as
sensitivity analysis.

| Variant | F1-cov | F1-bin | Precision | FPR | Detected |
|---|---:|---:|---:|---:|---:|
| Full SL-ADS, WBF uniform, threshold 0.103 | 0.879 | 0.917 | 0.847 | 0.965% | 14/14 |
| Reconst Only, threshold 0.103 | 0.923 | 0.962 | 0.926 | 0.433% | 14/14 |
| Prophet Only, threshold 0.103 | 0.676 | 0.713 | 0.554 | 3.479% | 14/14 |
| No C1 fixed lambda, threshold 0.103 | 0.753 | 0.780 | 0.639 | 3.113% | 14/14 |
| No C4/EDP, threshold 0.103 | 0.851 | 0.876 | 0.779 | 1.565% | 14/14 |
| MASE-Trust legacy | 0.000 | 0.000 | 0.000 | 0.000% | 0/14 |

The ablation harness now uses the same `catalog_outages_separate` normal-window
protocol as `eval_injection`, so the Full SL-ADS F1-cov and FPR align with the
main evaluation. Legacy outages-as-normal values are retained only in explicit
`*_legacy_outages_as_normal` CSV columns for traceability. The `Reconst Only`
result is an important upper-bound diagnostic, not a reason to claim the full
system is worse unless the paper explicitly reframes the contribution around
structural residuals.

## 8. Isolation Forest Comparison

Artifact:
`evaluation_if_fair/fair_if_vs_sl_summary.csv`

This comparison uses consensus pseudo-labels, not the curated attack catalog.
It measures agreement with statistical anomaly labels and is not directly
comparable to the catalog F1 above.

| System | F1 | FPR | Precision | Recall |
|---|---:|---:|---:|---:|
| SL-ADS | 0.181 | 5.824% | 0.381 | 0.119 |
| IF-fair-window | 0.387 | 44.922% | 0.286 | 0.596 |
| IF-fpr-matched | 0.349 | 7.537% | 0.514 | 0.264 |
| IF-k1-descriptive | 0.393 | 53.017% | 0.277 | 0.673 |

Paper wording: IF agrees more with pseudo-labels; this does not prove better
attack detection. Keep the methodology caveat next to the table.

## 8bis. Same Evidence, With vs Without Subjective Logic

Artifacts:
`evaluation_no_sl_fair/no_sl_fair_summary.csv`,
`evaluation_no_sl_fair/no_sl_fair_paired_vs_sl.csv`

This is the direct answer to the original "same ADS with SL vs without SL"
question. All rows use the same evidence CSV, the same train-calib-only
threshold policy, and the same catalog/outages-separate protocol. The non-SL
rows remove the SL bijection, uncertainty, EDP, ageing, and SL fusion; they use
direct scalar scores over the attack-evidence mass `N`.

| System | What replaces SL? | F1 micro | F1 macro | MCC | Precision | TPR | FPR | Detected | Coverage | F1-cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full SL-ADS | SL bijection + EDP + conflict-aware ageing + WBF | 0.8666 | 0.9292 | 0.8587 | 0.8466 | 0.8877 | 0.965% | 14/14 | 0.915 | 0.8795 |
| no_sl_mean_N | mean raw attack evidence over all leaves | 0.8268 | 0.9077 | 0.8185 | 0.7644 | 0.9001 | 1.665% | 14/14 | 0.924 | 0.8367 |
| no_sl_prophet_mean_N | mean attack evidence over Prophet leaves | 0.7206 | 0.8500 | 0.7138 | 0.6118 | 0.8766 | 3.337% | 14/14 | 0.926 | 0.7367 |
| no_sl_reconst_mean_N | mean attack evidence over reconstruction leaves | 0.8890 | 0.9412 | 0.8826 | 0.9004 | 0.8779 | 0.583% | 13/14 | 0.863 | 0.8811 |
| no_sl_hard_vote_90_r2 | R2-weighted hard vote on leaves with `N >= 0.90` | 0.1950 | 0.5778 | 0.1614 | 0.2473 | 0.1609 | 2.938% | 1/14 | 0.044 | 0.0741 |

Paired block-bootstrap on F1 micro, catalog/outages-separate protocol:

| Comparison | Delta F1 micro, Full SL - baseline | 95% BCa CI | Verdict |
|---|---:|---:|---|
| Full SL-ADS vs `no_sl_mean_N` | +0.0399 | [0.0108, 0.0852] | SL better under the all-leaf no-SL comparator |
| Full SL-ADS vs `no_sl_prophet_mean_N` | +0.1460 | [0.0899, 0.2511] | SL better |
| Full SL-ADS vs `no_sl_reconst_mean_N` | -0.0224 | [-0.1174, 0.0444] | no robust F1 difference; reconstruction-only is a strong diagnostic baseline |

Paper wording: against the leak-free all-leaf no-SL mean-evidence comparator,
the SL layer improves F1 micro and MCC while reducing FPR. Do **not** claim
universal dominance over every non-SL scalar: reconstruction-only evidence is a
strong RedeRio baseline with slightly higher pure window F1/MCC, but it detects
only 13/14 episodes and has lower episode coverage. Under the
operator-faithful protocol, `no_sl_reconst_mean_N` drops to F1 micro 0.7169
because outages are real anomalies and reconstruction-only misses much of that
bucket.

Effect-size caveat: the all-leaf SL gain is statistically positive but modest
(`+0.040` F1 micro). The paper should not justify the added Subjective Logic
complexity from F1 alone. The stronger argument is that the SL layer provides a
principled uncertainty-bearing fusion interface, an auditable operator choice,
outage-aware robustness, and a cause-qualification channel, while preserving
competitive detection performance.

## 8ter. Raw-Data Baselines

Artifacts:
`evaluation_raw_baselines/raw_baselines_summary.csv`,
`evaluation_raw_baselines/raw_baselines_paired_vs_sl.csv`

Raw baselines answer a different question: "how do classical detectors perform
directly on raw network metrics?" They are **not** the same-ADS-without-SL
comparison. The 13 catalog attacks are injected at evidence level, not into the
raw traffic CSV, so raw baselines cannot fairly be evaluated on those synthetic
episodes. The script therefore reports two raw-valid protocols and excludes
synthetic injection windows:

1. `real_events_excluding_synthetic`: real DDoS + network outages.
2. `pseudo_csv_excluding_synthetic`: the standardized CSV `label` column,
   useful as statistical-anomaly agreement, not as catalog attack detection.

Raw-data results on real events, train-calibrated thresholds. The SL row uses
`opinions_non_injected/detection_results_RAW.csv`, not the injected detection
CSV.

| System | F1 micro | F1 macro | MCC | Precision | TPR | FPR | ROC-AUC | PR-AUC | Real events detected | Real-event coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full SL-ADS | 0.7343 | 0.8615 | 0.7240 | 0.7723 | 0.6998 | 0.916% | 0.9736 | 0.7791 | 3/3 | 0.619 |
| raw_iforest | 0.5000 | 0.7424 | 0.5467 | 0.8937 | 0.3471 | 0.183% | 0.9958 | 0.8275 | 1/3 | 0.323 |
| raw_lof_novelty | 0.5013 | 0.7428 | 0.5351 | 0.8341 | 0.3583 | 0.316% | 0.9955 | 0.8404 | 1/3 | 0.333 |
| raw_pca_reconstruction | 0.4823 | 0.7327 | 0.5003 | 0.7375 | 0.3583 | 0.566% | 0.9598 | 0.5991 | 1/3 | 0.333 |
| raw_ocsvm_rbf | 0.4086 | 0.6928 | 0.3903 | 0.4751 | 0.3583 | 1.756% | 0.8420 | 0.4257 | 1/3 | 0.333 |
| raw_robust_z_max | 0.0990 | 0.5387 | 0.1630 | 0.5472 | 0.0544 | 0.200% | 0.9520 | 0.4310 | 1/3 | 0.049 |
| raw_sgd_ocsvm_rbf | 0.0781 | 0.5280 | 0.1223 | 0.4107 | 0.0432 | 0.275% | 0.8742 | 0.3329 | 1/3 | 0.040 |

Raw-data results on pseudo-labels, excluding synthetic intervals:

| System | F1 micro | MCC | Precision | TPR | FPR | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full SL-ADS | 0.1594 | 0.1561 | 0.5652 | 0.0928 | 2.187% | 0.6349 | 0.3621 |
| raw_iforest | 0.1289 | 0.2281 | 0.9807 | 0.0690 | 0.042% | 0.6286 | 0.4052 |
| raw_lof_novelty | 0.1425 | 0.2421 | 0.9869 | 0.0768 | 0.031% | 0.6497 | 0.4141 |
| raw_ocsvm_rbf | 0.1848 | 0.2293 | 0.7687 | 0.1050 | 0.968% | 0.6589 | 0.4629 |
| raw_sgd_ocsvm_rbf | 0.0360 | 0.1153 | 0.9643 | 0.0183 | 0.021% | 0.6626 | 0.4523 |
| raw_pca_reconstruction | 0.1543 | 0.2464 | 0.9537 | 0.0839 | 0.125% | 0.6273 | 0.4227 |
| raw_robust_z_max | 0.0320 | 0.1032 | 0.9057 | 0.0163 | 0.052% | 0.5873 | 0.3741 |

Interpretation: IF, LOF, and PCA rank raw real-event anomalies well, but their
deployable train-calibrated thresholds are conservative and miss most outage
windows. Exact RBF One-Class SVM and SGDOneClassSVM are included as informative
classical one-class baselines, but are not headline comparisons. These are
useful external baselines, not evidence that any raw detector is better on the
catalog task. The raw-baseline script hard-requires the non-injected SL CSV
unless the operator sets an explicit diagnostic override. LOF / exact OCSVM /
SGD-OCSVM use deterministic train-normal fit caps to avoid uncontrolled
quadratic runtime; thresholds are still calibrated on train-normal windows.

Per-real-event nuance: the `1/3` raw-baseline event count is not random. IF
detects the real DDoS with 96.8% window coverage, while LOF, PCA, and exact
RBF-OCSVM detect it with 100% coverage. They miss the two network-outage
episodes. SL-ADS detects the real DDoS with 96.8% coverage and also detects
both outages partially (`DEC1617` 55.5%, `NOV17` 33.3%). If the paper's
primary operational target is attack detection rather than outage monitoring,
the DDoS-specific comparison is more favourable to the raw baselines than the
aggregate `1/3` event count suggests.

To evaluate raw IF / PCA / LOF / OCSVM on the 13 synthetic catalog attacks, one
would need a separate raw-traffic injection generator and then rerun feature
extraction, training, and comparison. Reusing evidence-level synthetic labels
for raw baselines would be a protocol mismatch. The current evidence-level
comparison already covers the injected task through `compare_no_sl_fair.py` and
ablation runs.

## 9. Claims Allowed / Not Allowed

Allowed:

- "On the complete RedeRio run, SL-ADS detects 14/14 catalogued attack
  episodes with catalog/outages-separate F1 micro 0.867 and MCC 0.859."
- "When outages are treated as operator-relevant anomalies, F1 micro is 0.826."
- "The realised global FPR is 0.965%, exceeding the nominal 0.1% calibration
  target; ACTIVE daytime windows dominate the overshoot."
- "Regime-aware contextual discounting remains exploratory/future work."
- "Compared with the leak-free same-evidence `no_sl_mean_N` comparator,
  Full SL-ADS improves F1 micro from 0.827 to 0.867, MCC from 0.819 to 0.859,
  and FPR from 1.665% to 0.965%; paired block-bootstrap ΔF1 is
  +0.040 [0.011, 0.085]."
- "Raw-data IF/LOF/PCA/OCSVM baselines are reported separately on raw-valid protocols;
  they are not evaluated on synthetic catalog attacks because those attacks
  were injected at evidence level, not raw-traffic level."
- "The measured SL gain over the all-leaf no-SL scalar is modest; the paper's
  value proposition should include uncertainty-aware fusion, auditability,
  episode/outage robustness, and qualification, not just F1."

Not allowed:

- Do not claim the FPR target is met.
- Do not claim alpha=0.90 or any alpha-sweep value as a shipped production
  setting; it was exploratory and would require train-calib selection.
- Do not compare `compare_if` F1 directly against catalog F1 as if the labels
  were the same.
- Do not claim SL universally dominates every non-SL scalar. Reconstruction-only
  evidence is a strong diagnostic baseline on RedeRio and must be disclosed.
- Do not claim the reconstruction-only result generalises without rerunning the
  same comparison on other datasets or raw-level attack injections.
- Do not evaluate raw-data baselines on evidence-level synthetic attacks unless
  a raw-traffic injection generator is added and documented.
- Do not cite the 2026-05-10 reconstruction-only F1=0.915 as the current result.
