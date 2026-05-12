# Trust-Discount / R-squared Analysis: Slowloris Pathology

**Status:** 2026-05-12, active evidence note refreshed against complete run
`2e12261d55a8f975`
**Affects:** `WBF_WEIGHT_MODE = "trust_discount"` (config.py:243),
`TRUST_SCORE_FLOOR = 0.05`, `src/sl_ads/core/opinions_pipeline.py`,
`src/sl_ads/core/subjective_logic.py` (`apply_trust_discount`, Joesang
Def. 14.6).

## TL;DR

The **default** `WBF_WEIGHT_MODE = "uniform"` is reviewer-clean: it does
not depend on any secondary regression metric. The **alternative**
`"trust_discount"` mode uses per-source R-squared as a proxy for trust
(Joesang Def. 14.6). On Slowloris attacks, the metrics that *carry the
signal* are precisely the ones with the *worst* R-squared on training
benign data (because Slowloris exploits low-volume structural anomalies
that Prophet does not learn well even on benign traffic). This creates
a pathological inversion: the most informative source gets the smallest
trust weight.

This document documents the pathology, lays out the three alternatives
we considered, and explains why the chosen mitigation (default UNIFORM
+ optional trust_discount with explicit warning) is reviewer-defensible.

---

## 1. The pathology, stated precisely

Let s_i be a source detector with training-time R^2 = R_i and a
detection score m_i on a Slowloris event. Under
`WBF_WEIGHT_MODE = "trust_discount"`:

```
weight_i = max(R_i, TRUST_SCORE_FLOOR)
fused_score ~ sum_i (weight_i * m_i) / sum_i weight_i
```

Slowloris targets `prophet_fin`, `prophet_entropy_src_ip`, and
`reconst_fin_from_syn`. On RedeRio, the general problem is that
benign-regime forecastability is not the same property as attack
discriminativeness. Some attack-relevant streams have low or negative
R^2, while others have high R^2 but poor one-step MASE; neither proxy
reliably ranks sources by anomaly-detection usefulness.

  * `fin` count is bursty even in benign traffic (closing connections);
  * `entropy_src_ip` is dominated by short-tailed user diversity that
    Prophet's daily/weekly seasonality model under-fits;
  * `fin_from_syn` reconstruction is constrained by stochastic
    application-layer behaviour we do not model.

Under trust_discount, the fusion is driven by a benign forecasting
metric rather than by labelled anomaly utility. The complete-run
ablation confirms the operational consequence: legacy R^2 trust-discount
drops from Full SL-ADS `F1-cov=0.879` to `F1-cov=0.628`, raises FPR to
`4.39%`, and detects only `12/14` attacks at the calibrated operating
point.

## 2. Quantification (synthetic mockup + real residual audit)

Before the real data residuals run, we ran a synthetic mockup
(`analysis_residual_correlation.py --self-test`) that reproduces
the qualitative effect: when one stream is by construction noisy
(R^2 ~ 0.2) but carries the attack signal, trust_discount fusion
delivers F1 ~ 0.30 vs uniform fusion F1 ~ 0.65. The same gap shows
in the paired BCa CI: delta F1 = +0.34 [+0.27, +0.40] in favour of
uniform.

The real RedeRio residual audit was run on 2026-05-04 and confirms that
Prophet/Reconstruction residuals are not independent.  The cross 17x17
matrix has mean absolute off-diagonal rho = 0.265, max absolute rho =
0.915 (`prophet_udp` vs `reconst_udp_from_flows`), and verdict HIGH.
Artifacts are in
`../results/resultats_RedeRio_trained_v4s_v4_v2/diagnostics/residual_correlation/`.

## 3. Alternatives considered

### Option A - drop trust_discount entirely

**Pros**: simplest, no defence required.
**Cons**: removes a feature that *does* help on volumetric attacks
(where R^2 and signal are aligned). Loss of a SL-canonical mechanism
(Joesang Def. 14.6) that some reviewers will expect to see used.

### Option B - replace R^2 with MASE (Hyndman & Koehler 2006)

MASE = mean(|y_t - y_hat_t|) / mean(|y_t - y_{t-1}|)

It is *scale-invariant* and compares the model to a Naive-1 persistence
baseline: a series that is hard to predict but consistently close to
Naive-1 scores near MASE = 1. This was theoretically attractive, but the
complete RedeRio run shows that 30 s network traffic is dominated by
short-horizon persistence; MASE therefore becomes an overly harsh trust
proxy and discounts almost every source.

**Pros**: theoretically sound, well-known in time-series literature.
**Cons**: requires the Naive-1 baseline metric to be cached for every
source; mild engineering cost (factor of 2x training time).

### Option C - keep R^2, ship UNIFORM as default, document trust_discount as opt-in

**Pros**: zero code change in the default path. Ablation tables show
both modes; the reader sees the trade-off.
**Cons**: leaves trust_discount in the codebase; future maintainers may
turn it on without reading the warning.

## 4. Decision

**Adopted: Option C (uniform default) confirmed, Option B (MASE) implemented
and tested as opt-in alternative; finding below justifies the default.**

Justification:

1. The default `WBF_WEIGHT_MODE = "uniform"` is already shipped
   (config.py:243). The reviewer-relevant pathology is a *theoretical*
   risk only when someone overrides the default.
2. The audit document and the code now carry an explicit warning at
   the trust_discount activation site in `src/sl_ads/core/opinions_pipeline.py`:
   any future caller is told to read this analysis first.
3. **MASE-based weighting (Option B) is now implemented** in
   `src/sl_ads/stats/mase.py` and exposed as `WBF_WEIGHT_MODE='mase'`
   (PATCH D5, 2026-05-07). The empirical evaluation below shows that
   MASE produces a **theoretically clean but operationally pathological**
   trust score on RedeRio at the 30 s sampling rate, in a way that
   **mirrors but does not equal** the R² pathology. We therefore
   maintain `uniform` as the published default and document MASE as an
   audit-grade alternative whose behaviour at lower sampling rates may
   differ qualitatively.

## 4.1 Empirical MASE evaluation on RedeRio (PATCH D5, 2026-05-07)

The complete reference run `trained_models_RedeRio_trained_v4s_v4_v3`
persists raw per-Prophet MASE scores and the derived MASE-trust values.
It yields the following table:

| Metric                          | R² (CV)  | MASE     | trust (α=1, floor=0.05) | Naive-1 verdict |
|---------------------------------|---------:|---------:|-------------------------:|-----------------|
| `prophet_avg_pkt_size`          | -0.395   |   0.838  | 0.162                    | informative |
| `prophet_bytes`                 |  0.440   |   0.989  | 0.050                    | informative but floored by trust map |
| `prophet_packets`               |  0.634   |   1.062  | 0.050                    | worse than Naive-1 |
| `prophet_icmp`                  |  0.063   |   1.013  | 0.050                    | worse than Naive-1 |
| `prophet_fin`                   |  0.810   |   3.223  | 0.050                    | worse than Naive-1 |
| `prophet_udp`                   |  0.014   |   3.595  | 0.050                    | much worse |
| `prophet_flows`                 | -0.645   |   4.022  | 0.050                    | much worse |
| `prophet_tcp`                   | -1.526   |   4.548  | 0.050                    | much worse |
| `prophet_syn`                   | -2.851   |   5.295  | 0.050                    | much worse |
| `prophet_entropy_src_port`      |  0.698   |   7.768  | 0.050                    | much worse |
| `prophet_entropy_src_ip`        | -0.111   |   7.919  | 0.050                    | much worse |
| `prophet_entropy_dst_port`      |  0.489   |  10.220  | 0.050                    | much worse |

**Reading.** Ten out of twelve Prophet metrics have MASE > 1, i.e.
predict the next 30 s value *worse* than the trivial persistence
baseline (Hyndman & Koehler 2006 §3, "no-skill" point). `prophet_bytes`
barely beats Naive-1 (`MASE=0.989`) but maps to `1-MASE=0.011`, below the
trust floor. Only `prophet_avg_pkt_size` gets a non-floor trust value
(`0.162`). Under the canonical α=1 trust map, the detector is therefore
almost completely silenced.
RANSAC reconstructions inherit `trust = floor` because MASE is
undefined for non-temporal cross-feature regression
(`mase_score = NaN` → `mase_to_trust` falls back to floor).

**Why this is not a bug.** At 30 s sampling on a campus-network
benign trace, consecutive samples share most of their volumetric mass:
`y_t ≈ y_{t-1}` so the Naive-1 baseline absolute error
`|y_t − y_{t-1}|` is very small. Prophet, by design, smooths over a
1-day seasonal model and cannot match this short-horizon persistence
on absolute one-step error. This is consistent with Hyndman & Koehler
2006 §3 noting that MASE is *most informative* on lower-frequency
series (monthly, quarterly) where Naive-1 is not dominant.

**Why this is also not the whole story.** The metric with the highest
R² (`prophet_fin`, R²=0.810) has a poor MASE (3.22), and the
metric with the best MASE (`prophet_avg_pkt_size`, MASE=0.838) has
R²=−0.395. The two trust proxies are **uncorrelated** on this dataset
because they measure different aspects of forecasting quality:
  - R² rewards explained variance (Prophet captures seasonality →
    high R² but absolute errors are still > Naive-1).
  - MASE rewards short-horizon persistence (Naive-1 is competitive at
    30 s → MASE > 1 even when R² is high).

Neither proxy is appropriate for *anomaly-detection* trust because
both confuse "this model predicts well in benign regime" with
"this model is informative when an anomaly arrives" — the exact
proposition we want to use it for. A model with low predictive skill
on benign data may still produce **discriminative** residuals when an
attack pushes the metric outside the benign regime. R² and MASE both
miss this distinction.

**Recommendation reaffirmed.** `WBF_WEIGHT_MODE = 'uniform'` remains
the published default. Both `'trust_discount'` (R²) and `'mase'`
(Hyndman-Koehler) are exposed for ablation purposes only. A discriminative
trust proxy specific to anomaly detection (e.g. directional information
score on a labelled calibration split, Mercier-Quost-Denoeux 2008
contextual discounting per metric) is the principled replacement and
is documented as future work.

The MASE mode validates this reasoning by **failing in the opposite
direction** to R²: R² over-trusts the volumetric metrics (which
under-fit the bursty discriminative signals); MASE over-discounts
*everything* because Naive-1 dominates at 30 s. Only `'uniform'`
escapes both pathologies.

## 5. Code changes required

| File | Change | Status |
|------|--------|--------|
| `src/sl_ads/core/opinions_pipeline.py` | Runtime warning/comment that `trust_discount` is opt-in and points to this analysis | **DONE** |
| `src/sl_ads/config.py` | Default remains `WBF_WEIGHT_MODE='uniform'`; `trust_discount` documented as pathological on RedeRio | **DONE** |
| `src/sl_ads/core/subjective_logic.py` (`apply_trust_discount`) | Joesang Def. 14.6 implementation retained for ablation/trust experiments | **DONE** |
| `docs/audit/audit_verification_tracker.md` TASK-09 | Track real-data confirmation | **DONE** |
| `src/sl_ads/stats/mase.py` (NEW) | MASE + Joesang trust map (Hyndman-Koehler 2006). | **DONE** (PATCH D5) |
| `src/sl_ads/train/train_models.py` | Persist `mase_score` (raw) per Prophet metric and the top-level `mase_scores` dict. | **DONE** (PATCH D5) |
| `src/sl_ads/train/compute_mase_postfit.py` (NEW) | Patch existing PKLs without retraining; emits a `*.mase_postfit_audit.json` audit trail. | **DONE** (PATCH D5) |
| `src/sl_ads/core/opinions_pipeline.py` | New `WBF_WEIGHT_MODE='mase'` branch, plus warning if `mase_scores` missing. | **DONE** (PATCH D5) |
| `tests/test_mase_weighting.py` (NEW) | 19 unit tests covering MASE math, edge cases, and Joesang trust-map invariants. | **DONE** (PATCH D5) |
| Ablation: trust_discount vs uniform vs MASE | Documented in §4.1 above; `uniform` reaffirmed as published default. | **DONE** (PATCH D5) |

## 6. Paper-side implications

- Section 3.5 (Subjective Logic fusion): one paragraph stating "we
  default to uniform-weighted WBF; we evaluated trust-discount fusion
  per Joesang Def. 14.6 and found a pathology on low-R^2 / low-volume
  attacks; results reported in Appendix C".
- Appendix C: full ablation table with the BCa CI and McNemar p-values
  per scenario.
- Limitations (Section 5.3): explicit mention of MASE as a future
  upgrade.

## References

- Hyndman, R. J. & Koehler, A. B. (2006). "Another look at measures of
  forecast accuracy." *International Journal of Forecasting* 22 (4):
  679-688. - introduces MASE.
- Joesang, A. (2016). *Subjective Logic*. Springer. Definition 14.6
  (Trust transitivity / discounting).
