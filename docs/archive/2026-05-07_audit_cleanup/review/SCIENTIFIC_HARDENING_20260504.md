# Scientific Hardening - reviewer-safe claims for SL-ADS

**Date:** 2026-05-04  
**Scope:** RedeRio reference run `resultats_RedeRio_trained_v4s_v4_v2`,
publication tables, TSAD metrics, fusion assumptions, and the legacy
`sbn_qualifier` terminology.

This note is intentionally conservative.  It defines what can be claimed
without over-selling the system, what must appear in the final tables, and
what remains a scientific risk for a strict review committee.

---

## 1. Correct contribution wording

The main empirical evidence is RedeRio temporal network traffic with a
catalog of injected attacks plus one real DDoS interval.  The claim must
therefore be:

> SL-ADS is a controlled and interpretable detection/qualification pipeline
> for temporal network traffic, evaluated on RedeRio with timestamped
> injected attacks and a real DDoS interval.

Do **not** claim:

> state-of-the-art general IDS, universal TSAD benchmark winner, or causal
> attack attribution system.

The contribution can be defended as:

1. A reproducible temporal-network detection pipeline with explicit
   Prophet/Reconstruction evidence, Subjective Logic opinions, calibrated
   operational threshold, and archived run manifests.
2. A transparent SL fusion design where WBF is the reference because CBF
   would require a stronger independence assumption between branches.
3. An interpretable typological qualifier for known attack templates,
   including an uncertainty/novelty signal, but not a causal classifier.
4. A publication protocol that reports range-aware TSAD metrics, not only
   point F1.

---

## 2. Modern SOTA baseline protocol

A serious reviewer can reasonably ask why the paper compares mainly against
Isolation Forest and internal ablations while recent TSAD work includes
transformers, task-general time-series models, benchmark suites, and
foundation-model baselines.

Current benchmark context checked on 2026-05-04: the TSB-AD project page
last updated its leaderboard on 2026-04-01 and ranks methods by average
VUS-PR.  Its NeurIPS 2024 paper identifies flawed datasets, biased
evaluation measures, and inconsistent benchmarking as field-level risks,
so the SL-ADS paper must align its claims with that protocol culture.

Minimum reviewer-safe baseline set under the **same RedeRio protocol**:

| Family | Required examples | Why it matters |
|---|---|---|
| Classical/statistical | Isolation Forest, Matrix Profile/MERLIN or similar discord baseline | TSB-AD reports that simple/statistical methods remain strong; IF alone is too narrow. |
| Reconstruction/forecasting deep TSAD | TranAD, USAD/OmniAnomaly or MTAD-GAT/GDN | Multivariate reconstruction baselines match the reconstruction logic of SL-ADS. |
| Transformer TSAD | Anomaly Transformer | Directly targets unsupervised time-series anomaly detection with association discrepancy. |
| General time-series backbone | TimesNet | A modern ICLR task-general model with anomaly-detection results. |
| Benchmark/foundation-model line | TSB-AD reference algorithms plus MOMENT/Chronos/TimesFM/OFA/Lag-Llama where applicable | TSB-AD 2024 explicitly benchmarks statistical, NN, and foundation-model methods under unified setup. |

Protocol requirements:

1. Same train/test split and same timestamp attack catalog.
2. Same score-to-window alignment and no look-ahead.
3. Hyperparameters selected on train/calibration only.
4. No point-adjustment in headline metrics. If reported, label it as
   secondary and biased.
5. Primary metric: VUS-PR. Secondary: VUS-ROC, R-AUC-PR/ROC at max buffer,
   MCC, operational FAR, time-to-detect, and F1 with caveat.
6. A baseline that cannot be run under the same protocol must be listed as
   "not comparable", not omitted silently.

Until this table exists, the paper should say:

> We do not claim SOTA over modern TSAD.  We provide a controlled,
> interpretable IDS pipeline and report a reproducible protocol for adding
> modern baselines.

---

## 3. VUS and point-metric hardening

Point F1 remains useful operationally because false windows cost analyst
time, but it is not enough for TSAD.  Time-series anomalies are ranges,
detectors may be early or late by a few windows, and point adjustment can
inflate results.

The final detection table must include, at minimum:

| Metric | Current reference value | Artifact |
|---|---:|---|
| VUS-PR | `0.603828` | `../results/resultats_RedeRio_trained_v4s_v4_v2/evaluation/eval_vus_summary.csv` |
| VUS-ROC | `0.856201` | same |
| R-AUC-PR at max buffer | `0.490526` | same |
| R-AUC-ROC at max buffer | `0.759609` | same |
| Existence recall | `1.000000` | same |
| Number of anomaly ranges | `14` | same |
| Max buffer | `36` windows | same |

Use F1/MCC/FAR as operating-point metrics:

| Metric | Current reference value |
|---|---:|
| F1 micro, pure window-level | `0.784` |
| F1 macro, pure window-level | `0.885` |
| MCC | `0.772` |
| Operational FPR | `1.64%` in detection evaluation; qualification audit FAR remains a separate audit value |
| Detected attacks | `14/14` |

Paper wording:

> Because TSAD labels are interval-valued, VUS-PR is the primary
> threshold-free metric. Point F1, MCC and FAR are reported as operational
> metrics at the selected threshold, not as the sole evidence of TSAD
> quality.

---

## 4. Prophet/Reconstruction dependence

The mathematical WBF operator is valid as a belief-fusion operator, but the
empirical statement "Prophet and Reconstruction are independent" is not
supported by the current RedeRio residuals.

New diagnostic artifact:

`../results/resultats_RedeRio_trained_v4s_v4_v2/diagnostics/residual_correlation/residual_correlation_summary.json`

Actual RedeRio normal-window residual audit:

| Matrix | mean abs off-diagonal rho | max abs rho | worst pair | Verdict |
|---|---:|---:|---|---|
| Prophet | `0.304` | `0.889` | `prophet_syn` vs `prophet_tcp` | HIGH |
| Reconstruction | `0.149` | `0.366` | `reconst_bytes_from_entropy_src_port` vs `reconst_fin_from_syn` | MODERATE |
| Cross 17x17 | `0.265` | `0.915` | `prophet_udp` vs `reconst_udp_from_flows` | HIGH |

VIF warnings include `prophet_tcp = 18.09`, `reconst_udp_from_flows = 12.28`,
`prophet_udp = 11.30`, and several volume/protocol metrics above 5.

Secondary legacy Pearson check:

`investigations/compute_pearson_independence.py` now imports
`sl_ads.config` and uses the canonical attack catalogs.  On the same
RedeRio run, normalized-evidence Pearson correlations on normal windows
give mean cross `|rho| = 0.115`, max `|rho| = 0.805`
(`prophet_entropy_src_port_N` vs `reconst_tcp_from_packets_N`), with
`53/60` cross pairs below `0.3`.  This does not overturn the residual
audit above; it confirms that most pairs are weakly correlated while a
small set of shared traffic-derived features remains strongly dependent.

Conclusion:

1. Do not write that Prophet and Reconstruction are independent evidence
   sources.
2. Write that they are partially redundant views of the same traffic
   window.
3. Keep CBF as a legacy/sensitivity mode only; CBF would add evidence and
   can overstate certainty under dependence.
4. Report WBF uniform as the conservative reference and add the
   CBF/WBF/hierarchical ablation table when the full sweep is run.
5. The high correlation is not a fatal flaw for detection, but it limits
   causal/statistical interpretation of fused confidence.

Reviewer-safe wording:

> Residual correlations show that the two branches are not independent.
> We therefore avoid CBF as the reference inter-method fusion and report
> WBF uniform as a conservative pooling of partially redundant evidence.

---

## 5. "SBN" terminology and strict SBN feasibility

### What the code actually implements

The current `sl_ads.qualify.sbn_qualifier` is an
**expert-template-driven Subjective Logic qualifier** (SL-TQ).  It uses:

1. Group-level projected probabilities from upstream SL opinions.
2. Expert compatibility templates named `SBN_COND_OPINIONS` for legacy
   compatibility.
3. Dot-product compatibility scores, not Bayesian likelihood propagation.
4. Evidence accumulation plus the SL evidence-to-opinion bijection.
5. Optional temporal WBF and uncertainty maximisation.

It is a typological ranking and uncertainty mechanism, not a canonical
Subjective Bayesian Network.

### Could a strict SBN be used?

Yes, but it would be a different model.  A strict SBN would require:

1. A directed acyclic graph: attack type -> latent mechanism -> metric
   groups -> observed states.
2. Conditional subjective opinions for every edge/state, with explicit
   belief, disbelief/alternative mass, uncertainty, and base rates.
3. Deduction/abduction propagation through the graph, not only a
   discriminative dot product.
4. Enough labeled attack data to estimate or validate the conditional
   opinions, or a formal expert-elicitation protocol with sensitivity
   analysis.
5. External validation labels for attack families, not only injected labels.

Expected effect under the current evidence:

| Question | Conservative answer |
|---|---|
| Would strict SBN be possible? | Technically yes. |
| Would it automatically improve F1 or VUS? | No. |
| What would it likely improve first? | Calibration of uncertainty and honest "unknown" outputs. |
| Main risk | More parameters than the current labels can support, leading to fragile expert tables. |
| Best use now | Future extension after collecting real labeled attacks or running robust expert elicitation. |

Paper wording:

> The qualifier is not used as a causal SBN. It is a SL-template
> typological reasoning layer whose uncertainty reflects compatibility
> with known templates.

---

## 6. Remaining scientific risks after this hardening

| Priority | Risk | Current status | Required reviewer-safe treatment |
|---|---|---|---|
| P1 | Modern SOTA baselines not yet run under identical RedeRio protocol | Open | Do not claim SOTA; add baseline plan/table. |
| P1 | Main evidence is injected RedeRio | Open by design | State controlled/injected evaluation; avoid general IDS claims. |
| P1 | Residual dependence is high | Measured 2026-05-04 | Report correlations; use WBF uniform; keep CBF sensitivity. |
| P1 | Qualification labels are co-designed with injection templates | Known | Present as closed-world typology, not generalization. |
| P2 | VUS now generated but old tables were F1-first | Fixed in code, docs being updated | Make VUS-PR primary in final tables. |
| P2 | Strict SBN term was overstated | Fixed in code/doc terminology | Use SL-TQ / expert-template qualifier. |
| P2 | One real DDoS interval is not external validation | Open | Report as sanity check, not proof of deployment performance. |
| P2 | Full multi-seed complete reruns remain expensive | Harness exists | Use multi-seed if claiming numerical stability. |
| P3 | Some old docs still use legacy paths/names | Partly historical | Mark `PUBLICATION_TABLES.md` old sections as historical unless updated. |

---

## 7. Primary sources used for this hardening

1. Tatbul et al., "Precision and Recall for Time Series", NeurIPS 2018:  
   https://papers.nips.cc/paper/7462-precision-and-recall-for-time-series
2. Paparrizos et al., "Volume Under the Surface", PVLDB 2022:  
   https://www.vldb.org/pvldb/vol15/p2774-paparrizos.pdf
3. Liu and Paparrizos, "The Elephant in the Room: Towards A Reliable
   Time-Series Anomaly Detection Benchmark", NeurIPS Datasets and
   Benchmarks 2024 / TSB-AD:  
   https://nips.cc/virtual/2024/poster/97690  
   https://thedatumorg.github.io/TSB-AD/
4. Wu and Keogh, "Current Time Series Anomaly Detection Benchmarks are
   Flawed and are Creating the Illusion of Progress", IEEE TKDE:  
   https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/
5. Xu et al., "Anomaly Transformer", ICLR 2022:  
   https://openreview.net/forum?id=LzQQ89U1qm_
6. Wu et al., "TimesNet", ICLR 2023:  
   https://openreview.net/forum?id=ju_Uqw384Oq
7. Tuli et al., "TranAD", PVLDB 2022:  
   https://www.vldb.org/pvldb/vol15/p1201-tuli.pdf
8. Joesang, "Subjective Logic: A Formalism for Reasoning Under
   Uncertainty", Springer 2016:  
   https://link.springer.com/book/10.1007/978-3-319-42337-1
