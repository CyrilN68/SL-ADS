# Reviewer-Target Calibration

**Status:** 2026-04-25, Phase C
**Purpose:** map plausible reviewer profiles to the risks they will
prioritise and the evidence we must therefore foreground.

This document is for the *paper authors* (i.e. ourselves) - it is not
shipped with the paper. Its job is to make the trade-offs in venue
selection conscious and to align the paper structure with the most
likely reviewer's prior.

---

## 1. The calibration grid

| Venue tier | Likely reviewer | Top-3 risks they will probe | What we must foreground |
|------------|-----------------|------------------------------|--------------------------|
| **IEEE TISSEC / ACM TISSEC / NDSS** | Security researcher with formal-methods bias | (a) ground-truth integrity (Wu & Keogh #3); (b) base-rate fallacy on FPR claims (Axelsson 2000); (c) attack realism vs SOTA (Mirsky Kitsune 2018, Bartos LSTM 2016) | Section on label provenance, Axelsson per-attack PPV table, evidence-level vs raw-data ablation. |
| **IEEE TKDE / VLDB / ICDE** | Time-series / data-mining specialist | (a) Wu & Keogh 2021 four-flaw checklist; (b) Paparrizos VUS-PR / VUS-ROC 2022 alternative metric; (c) Baldan et al. 2025 multimodal benchmark comparison | Wu & Keogh self-assessment doc, VUS-PR table appendix, comparison vs UCR Time Series 2024 splits. |
| **IEEE Trans. Information Forensics & Security** | Practical IDS engineer | (a) operational FPR vs cost; (b) detection latency (TTD); (c) interpretability for SOC operators | Per-attack TTD with BCa CI, DET curve at operational FPR, attribution table per attack family. |
| **IEEE INFOCOM / CoNEXT / SIGCOMM** | Networking systems researcher | (a) raw traffic realism, RFC compliance of injected attacks; (b) computational cost in flows/sec; (c) deployment in real ISP topology | Raw-injection ablation, throughput table, RedeRio deployment provenance section. |
| **MDPI Sensors / MDPI Electronics** | Generalist applied | Mostly novelty + clarity, less depth on stats | Strong intro / clear figures suffice. **Risk: dismissed as "another anomaly detection paper"** if no clear novelty hook. |
| **Journal of Subjective Logic / similar Joesang community workshop** | Subjective-Logic specialist | (a) canonical bijection compliance; (b) WBF independence assumption; (c) use of trust-discount per Def. 14.6 | sl_formulas_v2 docstring, residual correlation analysis, trust_discount/R^2 doc. |

## 2. Risk-by-priority for our paper

After cross-checking the calibration grid with our actual contributions,
the **top-tier reviewer concerns** for which we have the *thinnest*
evidence are:

1. **Wu & Keogh 2021 #1 / triviality of volumetric attacks**
   - Mitigated by ablation_injection_level.py + paper qualifier.
   - **Status: PASS after Phase C work.**

2. **Axelsson 2000 base-rate fallacy on operational FPR**
   - Currently we report FPR on the test split, which combines benign
     and injected periods. The reviewer will demand a per-class
     conditional probability table P(alarm | benign), P(benign | alarm).
   - **Status: PARTIAL. Action: add Axelsson table to Section 4.4.**

3. **Paparrizos 2022 VUS-PR / VUS-ROC**
   - We report only F1/MCC/FPR at a fixed threshold. Modern reviewers
     prefer threshold-free metrics; Paparrizos (VLDB 2022) gives both
     volume-under-the-surface variants for streaming evaluation.
   - **Status: NOT YET DONE. Action: add VUS-PR computation in
     `evaluate_injection_v2.py` and a comparison table.**

4. **Subjective-Logic canonical compliance**
   - Resolved by the canonical_two function (sl_formulas_v2.py:438+,
     2.22e-16 agreement with evidence-space WBF) and the bijection
     property tests (tests/test_fusion_wbf_canonical.py - 8/8 PASS).
   - **Status: PASS.**

5. **Mirsky Kitsune 2018 / Bartos LSTM 2016 baseline comparisons**
   - We currently compare only against IsolationForest and the trivial
     z-score baseline. A network-security reviewer will demand at least
     Kitsune as a baseline.
   - **Status: NOT YET DONE. Action: integrate Kitsune in
     `compare_if_fair.py`. ETA: 1-2 days of engineering.**

6. **Baldan et al. 2025 multimodal benchmark**
   - Recent (2025) work uses standardized multivariate splits. We do
     not currently benchmark against it.
   - **Status: DEFERRED to v11 (out of scope for this submission).**

## 3. Decision: target venue

Given our actual evidence:

  * Strong: SL canonical compliance, residual correlation analysis,
    Wu & Keogh self-assessment, BCa CI + McNemar.
  * Moderate: trivial-baseline coverage, paired ablations.
  * Thin: Kitsune comparison, VUS-PR, Axelsson table.

**Recommendation:** target **IEEE TIFS** or **Computer & Security**
(Elsevier) as primary venue.
- IEEE TIFS reviewers will weigh the SL formalism and the
  reproducibility of our injection catalog highly; Kitsune comparison
  is *expected but not dealbreaker* since our contribution is the SL
  fusion, not a new feature extractor.
- Computer & Security has accepted similar SL-based IDS papers (e.g.
  Cerutti et al. 2020) and tolerates a slightly thinner baseline list
  if the formalism is rigorous.

**Avoid:** IEEE TKDE / VLDB - they will demand the full Wu & Keogh
remediation (which we have only partially done) AND VUS-PR (which we
have not done at all). The cost-benefit is poor at this stage.

**Avoid:** IEEE TISSEC / ACM TISSEC - they will demand a Kitsune
baseline AND a deployed validation; we have neither.

**Fallback:** MDPI Sensors / Electronics if the primary submission is
rejected without major-revision option.

## 4. Editor / cover-letter framing

When we submit, the cover letter should foreground:

1. The canonical Joesang 2016 compliance (machine-epsilon agreement
   with the evidence-space WBF, 8 property tests).
2. The transparent ablation suite (4 scripts, all with self-tests).
3. The reproducibility artefact: every CI in the paper is reproducible
   from `stats_bootstrap_ci.py` + the seeded injection catalog.

We **must not** oversell on:
  * "State of the art" - we do not have a Kitsune-level comparison.
  * "Operational deployment" - the RedeRio data is offline.

## 5. Live status (as of this session)

| Reviewer concern | Evidence available | Verdict |
|------------------|-------------------|---------|
| Wu & Keogh #1 (triviality) | `ablation_injection_level.py` flagging | PARTIAL (paper text needed) |
| Wu & Keogh #2 (realism) | dual-path injection + realism probe | PARTIAL (paper text needed) |
| Wu & Keogh #3 (mislabel) | synthetic ground truth by construction | PASS |
| Wu & Keogh #4 (overfit) | train/test split + ablation defence of defaults | PARTIAL (multi-seed pending) |
| Axelsson 2000 base-rate | per-attack PPV table | NEEDED |
| Paparrizos 2022 VUS-PR | none | NEEDED |
| Joesang 2016 canonical | sl_formulas_v2 + tests | PASS |
| Joesang Def. 14.6 trust-discount | trust_discount_r2_analysis.md | PASS (defendable opt-in) |
| Mirsky Kitsune 2018 baseline | none | NEEDED before submission |
| Bartos LSTM 2016 baseline | none | OPTIONAL |
| Baldan 2025 multimodal | none | DEFERRED v11 |
| Hutchins 2011 Kill Chain | `ablation_temporal_sbn.py` H1/H2/H3 | PASS |

## 6. Bottom line

We are **submission-ready for Computer & Security** modulo:
  * Add Axelsson PPV table.
  * Add Kitsune baseline.
  * Add VUS-PR appendix.

These are 1-2 weeks of engineering, not architectural changes.
