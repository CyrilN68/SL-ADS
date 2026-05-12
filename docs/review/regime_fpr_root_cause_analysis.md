# Regime-FPR Root-Cause Analysis

**Status:** 2026-05-12, refreshed on complete 17-leaf run  
**Run:** `2e12261d55a8f975` / `resultats_RedeRio_trained_v4s_v4_v3`  
**Scope:** A1.5 realised false-positive rate under calendar regimes.

The 2026-05-10 reconstruction-only diagnostic and the older alpha-sweep tables
are superseded for paper-facing numbers. They remain useful audit provenance,
but the current paper must use the complete-run values below.

## 1. Realised FPR

Artifacts:

- `outputs/scientific_hardening/regime_fpr.csv`
- `outputs/scientific_hardening/regime_fpr_summary.json`
- `outputs/scientific_hardening/regime_fpr_diagnosis.{csv,json,md}`

| Regime | Normal windows | FP | FPR | Ratio to 0.1% target |
|---|---:|---:|---:|---:|
| all_normal | 12,015 | 116 | 0.965% | 9.65x |
| weekday_term_like | 7,676 | 114 | 1.485% | 14.85x |
| weekend | 3,366 | 1 | 0.030% | 0.30x |
| holiday_or_closure | 1,261 | 1 | 0.079% | 0.79x |
| day_08_18 | 4,867 | 91 | 1.870% | 18.70x |
| night_00_06 | 3,027 | 2 | 0.066% | 0.66x |
| shoulder_06_08_18_24 | 4,121 | 23 | 0.558% | 5.58x |
| canonical_ACTIVE | 3,135 | 91 | 2.903% | 29.03x |
| canonical_QUIET | 8,880 | 25 | 0.282% | 2.82x |

Annualised projection from the audited 45.4-day span:

| Quantity | Value |
|---|---:|
| Expected realised FPR, regime-weighted | 0.958% |
| Expected FP windows / year | 1006.8 |
| FP windows / year under target | 105.1 |
| Realised / target annual ratio | 9.58x |

This means the nominal `FPR_TARGET_DECISION = 0.001` is not achieved on the
test span. This is a paper limitation, not a hidden success.

## 2. Root-Cause Verdict

The complete-run diagnosis returns:

| Diagnostic ratio, ACTIVE / QUIET | Value |
|---|---:|
| Median per-metric exceedance rate | 0.571 |
| Fused `proj_atk` p99.9 | 1.522 |
| Joint exceedance k=3 | 4.524 |

**Verdict: `H_correlation`.**

The per-metric exceedance rate is not higher in ACTIVE; it is lower than QUIET
on median. The problem is that benign ACTIVE traffic produces more simultaneous
multi-metric alarms. When three or more correlated metrics fire in the same
window, the fusion layer accumulates redundant evidence and the fused tail
crosses the operating threshold more often.

Paper wording:

> Realised false positives are concentrated in active weekday/daytime traffic.
> The root cause is not simply per-metric EVT miscalibration; it is correlated
> joint exceedance across physically coupled traffic metrics.

## 3. Fix Status

Calendar-aware EVT (TASK-57) is implemented but remains audit-grade/off by
default. It addresses per-regime per-metric thresholds, while the complete-run
diagnosis points primarily to correlation at the fusion layer.

Per-regime contextual discount (TASK-59) remains **exploratory/future work**.
No alpha value is shipped or claimed. Any future alpha must be selected on a
train-calib split, locked, and evaluated once on test to avoid leakage.

Current paper stance:

- Report the realised regime-FPR values above.
- Keep production/reference alpha at 1.0.
- Describe contextual discounting or correlation-aware fusion as future work.

## 4. Reproducibility

From `current_version/`:

```bash
python -m sl_ads.ablation.evaluate_regime_fpr
PYTHONIOENCODING=utf-8 python -m sl_ads.audit.regime_fpr_diagnosis
```

Expected diagnostic headline:

```text
[VERDICT] H_correlation
median per-metric A/Q = 0.571
fused p99.9 A/Q       = 1.522
joint k=3 A/Q         = 4.524
```
