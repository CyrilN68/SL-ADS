# Wu & Keogh (2021) Self-Assessment

**Status:** 2026-04-25, Phase C
**Cite:** Wu, R. & Keogh, E. (2021). *Current Time Series Anomaly Detection
Benchmarks are Flawed and are Creating the Illusion of Progress.*
IEEE TKDE 35(3): 2421-2429.

This document audits our pipeline against the four flaws identified by
Wu & Keogh and explains, item by item, what we do, what is at risk, and
how the evidence supports our claims. Every "PASS / FAIL / PARTIAL"
verdict has a code-traceable justification.

---

## Flaw #1 - Triviality

> "Anomalies that can be detected by a one-line baseline (mean +
> 3 sigma, threshold-on-z) make the benchmark uninformative."

| Aspect | Verdict | Evidence |
|--------|---------|----------|
| **Volumetric attacks (UDP/SYN flood)** | **PARTIAL** | `ablation_injection_level.py` self-test on the synthetic volumetric scenario reaches F1=0.92 with z-score>3 and IsolationForest, both >= 90 % of any sophisticated detector -> we explicitly flag these as TRIVIAL in the framework output. |
| **Slowloris / low-rate attacks** | **PASS** | Same self-test, "subtle" + "slowloris" scenarios: best trivial F1 < 30 % of pipeline F1 -> NON_TRIVIAL flag fires. |
| **Volumetric realism** | **MITIGATION REQUIRED** | We do not claim our pipeline "beats trivial baselines" on volumetric injections; we report numbers but mark them as low-information. The peer-reviewable claim is restricted to the *combination* of: (a) attribution / signature recovery (b) low-rate attacks (c) detection latency, where trivial baselines fail. |

**Action item paper-side:** add a "Trivial baseline" row to every
detection table (Section 4.x). Numbers come from
`ablation_injection_level.py --out results/ablation_injection_level.json`.

---

## Flaw #2 - Unrealistic anomaly density / mode

> "Injected anomalies have distributional fingerprints (huge mean shift,
> step jumps) that no real attacker produces."

| Aspect | Verdict | Evidence |
|--------|---------|----------|
| **Evidence-level injection (default)** | **AT RISK** | `inject_at_evidence_level.py` operates *after* Prophet residuals, so the upstream signal is never inspected for plausibility. The KS distance between injected and base PSN is necessarily 1.0 (UNREALISTIC_OUTLIER) by construction. |
| **Raw-data-level injection (new path)** | **PASS** | `ablation_injection_level.py` provides a parallel raw-data path that perturbs (bytes, packets, syn, ...) before Prophet, then lets the pipeline derive PSN organically. The "realism probe" reports KS, mean-shift, and skewness drift. |
| **Cross-validation between paths** | **EVIDENCE PROVIDED** | Paired BCa CI on F1 gap + McNemar on discordant pairs, per scenario, in the comparison CSV. |

**Action item paper-side:** every reported metric must be qualified as
"evidence-level injection" or "raw-data injection". Whichever path we
present as primary, the *other* must be reported in an appendix table.

---

## Flaw #3 - Mislabeled ground truth

> "The labels in standard benchmark datasets (Yahoo S5, NAB, Numenta) are
> wrong often enough that any detector achieves spuriously high accuracy
> by overfitting to the labels rather than the anomalies."

| Aspect | Verdict | Evidence |
|--------|---------|----------|
| **Synthetic injection (our case)** | **PASS** | Labels are constructed by the injector itself (start_ts, end_ts of the attack window in `INJECTED_ATTACK_CATALOG`, config.py:946-989) - they are the ground truth by construction. No third-party label noise. |
| **Background-traffic period** | **PASS** | The non-injection segments are labelled benign by *exclusion*. We acknowledge the residual risk that the public RedeRio capture contains undocumented anomalies, but by removing all known maintenance windows (config.py:HOLIDAY_LIKE_WINDOWS) we reduce this to weeks of the year. |
| **Label-set breadth** | **PARTIAL** | Only attack-window granularity is labelled; we do *not* label individual flow records. This is intentional given Wu & Keogh's caution (over-labelling at sub-event granularity creates hidden errors), but it caps our reportable metric set: TTD, F1 over 1-min bins, and event-level recall, NOT flow-level precision. |

---

## Flaw #4 - Run-to-failure / over-tuned bias

> "Detectors are tuned on the same set as they evaluate; reported
> performance is a fitting curve."

| Aspect | Verdict | Evidence |
|--------|---------|----------|
| **Train / test temporal split** | **PASS** | `src/sl_ads/train/train_models.py` fits Prophet on the first ~70 % of timeline; injections from `src/sl_ads/config.py` are placed in the last ~30 %. EVT thresholds calibrated on the train split only. |
| **Hyperparameter selection** | **PARTIAL** | `WBF_WEIGHT_MODE`, `TRUST_SCORE_FLOOR`, `SBN_TEMPORAL_ENABLED` were chosen by *defending* the default values via ablation (this session). They were not tuned on the test scenarios. The risk that an unconscious bias drove the *inventory* of tested values remains - mitigation: every defended value is recorded in `docs/audit/audit_verification_tracker.md` with its rationale. |
| **Multi-seed evaluation** | **NEEDED** | Currently we rely on the deterministic injection seed (config.py:RANDOM_SEED). The reviewer-clean upgrade is to run k=5 seeds and report median + IQR via bootstrap CI; the framework is in place (`stats_bootstrap_ci.py`) but the pipeline glue is not yet wired. **This is the highest-priority remaining code task.** |

---

## Summary Table

| Flaw | Verdict | Action |
|------|---------|--------|
| #1 Triviality | PARTIAL | Surface trivial-baseline F1 in every table; mark volumetric as low-info. |
| #2 Unrealistic density | PARTIAL | Mandatory dual-path reporting (evidence-level + raw-data injection). |
| #3 Mislabel | PASS | Acknowledged in limitations; restrict claims to event granularity. |
| #4 Run-to-failure | PARTIAL | Multi-seed evaluation pending; one open item. |

**Bottom line:** none of the flaws apply in *strong* form, but flaws
#1 and #2 require explicit defensive disclosures in the paper, and
flaw #4 requires a multi-seed re-run before the reviewer-clean version.

---

## Reproduction commands

```bash
python ablation_injection_level.py --self-test
python analysis_residual_correlation.py --self-test
# Real data paths (when residuals.csv is available):
python ablation_injection_level.py --out results/ablation_injection_level.json
python analysis_residual_correlation.py --residuals-csv data/residuals.csv
```
