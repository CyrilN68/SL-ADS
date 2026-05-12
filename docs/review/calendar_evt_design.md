# Calendar-aware EVT thresholds — Design and Post-Mortem (H2, Phase B)

**Status:** 2026-05-12, code shipped as audit-grade opt-in; default OFF for
RedeRio after the complete-run refresh.
**Author:** Phase B kickoff; updated after implementation/post-mortem.
**Affects:** A1.5 (regime-FPR overshoot: complete run `2e12261d55a8f975`
reports global FPR `0.965%` = `9.65×` target and canonical ACTIVE FPR
`2.903%` = `29.03×` target),
A1.9 (calibration sidecar enforcement), and the threshold-loading path in
`paths.py` / `compute_evidence.py`.
**Goal:** document the calendar-aware EVT sidecar API and why it remains
audit-grade opt-in. The complete-run root cause is correlation-level
(`H_correlation`) rather than pure per-metric calendar miscalibration, so this
feature is not claimed as the RedeRio fix; see
`docs/review/regime_fpr_root_cause_analysis.md`.

This document began as the design note for the **API of the sidecar**, the
**regime partition**, the **fallback strategy**, and the **A1.9
strict-validation update**. It is now also the post-mortem record for why the
feature remains audit-grade opt-in rather than the RedeRio default.

---

## 1. Why this is needed (and what is already true)

The earlier 2026-05-06 regime-FPR audit showed calendar concentration of
false positives. The paper-facing complete run `2e12261d55a8f975` supersedes
those preliminary ratios: global realised FPR is `0.965%` (`9.65×` target),
canonical ACTIVE is `2.903%` (`29.03×` target), and canonical QUIET is
`0.282%` (`2.82×` target). The current diagnosis is `H_correlation`: ACTIVE
traffic produces more simultaneous multi-metric exceedances, even though the
median per-metric exceedance rate is lower than QUIET. This is a documented
limitation, not a hidden bug.

What we already have:

  - A clean **regime audit** module (`evaluate_regime_fpr.py`) that
    classifies windows by `weekday × daytime × holiday`.
  - A per-metric **EVT pipeline** (`_evt_threshold_pair` in
    `train_models.py`) with PWM/Grimshaw/empirical fallback chain
    (PATCH 2026-05-06, A1.2).
  - A **sidecar enforcement** (`paths.validate_threshold_sidecar_config`)
    that already hard-raises on any calibration-vs-runtime mismatch on
    sensitive knobs (A1.9, PATCH TASK-45).
  - A **mode-specific sidecar** mechanism (TASK-55) that emits and
    validates per-mode threshold sidecars for WBF/ABF strict
    recalibration.

The H2 design re-uses each of these primitives and only adds a calendar
dimension on top.

---

## 2. Regime partition (disjoint, deployable, EVT-stable)

Two partition shapes were considered.

### 2.1 Option α — 2-bucket partition (recommended)

```
ACTIVE  =  weekday × not(holiday) × hour ∈ [08, 18)
QUIET   =  everything else
```

| Bucket | n_normal_windows (audit run) | n_peaks at q=0.10 | EVT-stable? |
|---|---:|---:|---|
| `ACTIVE`  | 4 867 | ≈ 487 | yes (≥ EVT_MIN_PEAKS=50) |
| `QUIET`   | 7 148 | ≈ 715 | yes (≥ 50) |

**Pros.** Largest possible buckets → tightest GPD fits → fewest empirical
fallbacks. Two buckets are easy to explain in the paper. Maps cleanly to
the dominant FPR overshoot mechanism (weekday-daytime is the only regime
above target).

**Cons.** Coarser than the regime audit. Day-of-week structure within
weekdays (Mon vs Fri patterns) is ignored.

### 2.2 Option β — 4-bucket partition

```
WEEKDAY_DAY    =  weekday × not(holiday) × hour ∈ [08, 18)
WEEKDAY_NIGHT  =  weekday × not(holiday) × hour ∉ [08, 18)
WEEKEND        =  weekend × not(holiday)
HOLIDAY        =  holiday or closure (calendar-driven)
```

| Bucket | n_normal_windows (audit run) | n_peaks at q=0.10 | EVT-stable? |
|---|---:|---:|---|
| `WEEKDAY_DAY`   | 4 867 | ≈ 487 | yes |
| `WEEKDAY_NIGHT` | 2 809 | ≈ 281 | yes |
| `WEEKEND`       | 3 366 | ≈ 337 | yes |
| `HOLIDAY`       | 1 261 | ≈ 126 | borderline (close to EVT_MIN_PEAKS=50, OK on most metrics, may fall back on rare ones) |

**Pros.** Matches the operator's natural calendar concept; weekend traffic
profile is visibly different from weekday-night profile in the audit
data.

**Cons.** `HOLIDAY` is the smallest bucket; if a metric has only 30 % of
its observations exceed the q=0.10 init threshold, fall-back to empirical
quantile might dominate (~38 peaks instead of ≈126).

### 2.3 Recommendation

**Ship Option α (2 buckets)** as the published reference and document
Option β as an opt-in extension. Justification:

1. The regime-FPR audit identifies **one** dominant regime
   (`canonical_ACTIVE` / weekday-daytime). The complete run confirms the
   concentration but attributes the mechanism mainly to correlated joint
   exceedances, not isolated per-metric EVT drift.
2. With Option β the smallest bucket (`HOLIDAY`, 1 261 windows) would
   require a per-metric peak-count check; Option α leaves at least 4 867
   windows in every bucket on RedeRio, which is comfortable.
3. Two buckets keep the Section 3.4 paper paragraph readable. A reader
   can audit "active vs quiet" in one sentence; "weekday-day vs
   weekday-night vs weekend vs holiday" needs a table.

The implementation will be parametric on the bucket function so swapping
to Option β is a 5-line change at the partition-builder site, not a
refactor.

---

## 3. Calibration flow

### 3.1 Train time

```
df_train_calib  ──►  per metric m:
                      excesses_per_bucket[m] = {
                         "ACTIVE": signed_residuals[bucket=="ACTIVE"],
                         "QUIET":  signed_residuals[bucket=="QUIET"],
                      }
                     ▼
                     for bucket in {"ACTIVE","QUIET"}:
                         (t_susp, t_atk) = _evt_threshold_pair(
                             excesses_per_bucket[m][bucket],
                             q_evt_susp, q_evt_atk,
                             safety_margin,
                             metric_key=f"{m}:{bucket}",
                             branch=branch,
                         )
                         models_pkg[m]["thresholds_per_regime"][bucket] = {
                             "t_susp": t_susp, "t_atk": t_atk,
                             "t_trapeze_base": ...
                         }
```

The legacy single-bucket fields (`models_pkg[m]['t_susp']`, `['t_atk']`,
`['t_trapeze_base']`) are **kept** as a defensive fallback (used only if
the runtime cannot determine the bucket of a window — a guardrail, not a
production path). Their values become the bucket-weighted average of
the per-regime values, so any consumer that ignored regimes still gets a
sensible global threshold.

### 3.2 Inference time (`compute_evidence.py`)

```
for window in test_span:
    bucket = regime_of(window["timestamp"])     # pure-timestamp dispatch
    t_susp = pkg["thresholds_per_regime"][bucket]["t_susp"]
    t_atk  = pkg["thresholds_per_regime"][bucket]["t_atk"]
    (P,S,N) = trapezoidal_map(residual, t_susp, t_atk, t_trapeze_base, ...)
```

`regime_of` is a pure function of the window timestamp and the holiday
calendar (already in `CONFIG['HOLIDAYS_LIST']`, populated by TASK-15).
**No state, no race**. The function lives in `sl_ads/calendar/regime.py`
(new module) and is the single source of truth used by the regime audit
and the threshold dispatch.

### 3.3 What per-regime EVT would guarantee if its assumptions held

By construction, EVT calibration on the bucket-conditional excesses
estimates `P(|residual| > T | bucket = b, normal) = q_evt`. The
regime-aware dispatch then aligns the per-bucket realised FPR with this
target, regardless of the relative population of buckets in deployment.
The current single-threshold design implicitly aggregates buckets in
proportion to their training-span weight, which fails when the
deployment regime mix differs.

Important post-mortem caveat: this guarantee is per metric and conditional on
stable, weakly dependent residual tails inside each bucket. The complete-run
RedeRio failure mode is mostly *joint* exceedance correlation after fusion.
Calendar-aware EVT is therefore a useful opt-in calibration primitive, but it
is not claimed to solve the current RedeRio FPR overshoot by itself.

---

## 4. Sidecar API change (A1.9 strict)

The current sidecar persists scalar fields:

```json
{
  "decision_threshold": 0.12915...,
  "fusion_mode_at_calibration": "wbf",
  "wbf_weight_mode": "uniform",
  "lambda_decay": 0.85,
  ...
}
```

The H2 sidecar adds a **per-regime block** while keeping the legacy
scalar field for backward compatibility:

```json
{
  "decision_threshold": 0.12915...,
  "fusion_mode_at_calibration": "wbf",
  "wbf_weight_mode": "uniform",
  "lambda_decay": 0.85,

  "calendar_evt": {
    "enabled": true,
    "partition": "alpha_2_bucket",
    "buckets": ["ACTIVE", "QUIET"],
    "regime_fn_signature":
      "weekday_x_daytime_x_holiday/v1@2026-05-07",
    "thresholds_per_regime": {
      "ACTIVE": { "decision_threshold": 0.135 },
      "QUIET":  { "decision_threshold": 0.110 }
    }
  },
  ...
}
```

`paths.validate_threshold_sidecar_config` is extended:

  - **Knob match.** `calendar_evt.partition` and
    `calendar_evt.regime_fn_signature` are added to
    `_SIDECAR_SENSITIVE_KNOBS`. Any change in the regime function
    signature (e.g. moving from a 2-bucket to a 4-bucket partition, or
    swapping the daytime hour boundaries) is treated as a calibration
    change and triggers a hard raise unless the operator explicitly
    sets `SL_ALLOW_THRESHOLD_FUSION_MISMATCH_FOR_ABLATION=1`.
  - **Backward compatibility.** When `calendar_evt.enabled = false` the
    runtime ignores `thresholds_per_regime` and falls back to the
    scalar `decision_threshold` (legacy behaviour, identical to the
    current shipped pipeline).

### 4.1 Why `regime_fn_signature` matters

Two operators with the same `partition = "alpha_2_bucket"` but different
holiday calendars (e.g. UFRJ vs another university) would produce
calibrations that disagree on which days fall into `QUIET`. Persisting
the regime-fn signature lets the validator hard-fail on calendar drift
between training and deployment, in the same spirit as the existing
A1.9 sidecar/runtime guard.

---

## 5. Test plan

Tests live in `tests/test_calendar_aware_evt.py` (new file) and exercise:

  - `regime_of` is deterministic over a 1-year synthetic timestamp grid.
  - `regime_of` partitions the audit-run windows identically to
    `evaluate_regime_fpr._regime_masks` for the buckets that overlap.
  - Per-regime EVT calibration on synthetic GPD samples recovers
    target quantiles within the documented precision band (PWM
    self-test extended to two buckets).
  - Per-regime sidecar round-trip: write → read → match; mismatch on
    `regime_fn_signature` triggers `RuntimeError("[A1.9] …")`.
  - End-to-end synthetic check: a deliberately regime-skewed test span
    (30 % weekday-day only) yields realised FPR within ±20 % of the operator
    target with calendar-aware thresholds, vs > 4× without. This is a
    synthetic sidecar sanity check, not the final RedeRio complete-run result.
  - Backward compatibility: a sidecar with `calendar_evt.enabled = false`
    behaves byte-identically to the pre-H2 pipeline.

The fifth test is an implementation sanity check for the sidecar. It should
not be cited as a paper result for RedeRio; the paper-facing regime-FPR table
is `docs/review/regime_fpr_root_cause_analysis.md`.

---

## 6. Effort estimate

| Stage | Effort |
|---|---:|
| `sl_ads/calendar/regime.py` (new module + tests) | ½ day |
| `train_models.py` per-bucket calibration loop | ½ day |
| `compute_evidence.py` runtime dispatch | ½ day |
| `paths.py` sidecar API extension + A1.9 enforcement | ½ day |
| `tests/test_calendar_aware_evt.py` (5 cases above) | 1 day |
| Retrain + regime-FPR re-audit | ½ day compute + ½ day verify |
| Docs update (`ASSUMPTIONS.md` A1.5, hardening report,
  `honest_limitations.md` §5.3.7) | ½ day |
| **Total** | **~4 days** (within Phase B budget) |

---

## 7. Historical decisions

1. **Partition.** Ship Option α (2-bucket) as default, Option β as
   opt-in via `CONFIG['CALENDAR_EVT_PARTITION']`?  → expected: yes.
2. **Backward compatibility.** Keep `calendar_evt.enabled = false` as
   the default for the next pkl regenerated by the Phase B retrain (so
   the H2 code lands inert) and flip to `true` only after the
   regime-FPR re-audit confirms ≤ 1.2× overshoot? → expected: yes —
   this is the cautious deployment path.
3. **Holiday source.** Reuse `CONFIG['HOLIDAYS_LIST']` (TASK-15
   Brazilian national holidays) or take a `--holidays` CLI override
   for adapters? → expected: reuse, with override.
4. **Sidecar fields.** Persist also the per-bucket realised FPR on the
   train-calibration set so a reviewer can audit "did the bucket sizes
   match?" without re-running the audit? → expected: yes.
5. **Failure mode.** When a bucket has fewer than `EVT_MIN_PEAKS`
   peaks, fall back to: (a) empirical quantile within the bucket;
   (b) the global EVT threshold; or (c) the closest sibling bucket
   (e.g. `HOLIDAY` borrows from `WEEKEND` if too small)?  → propose
   (a), with a logged warning.

These design decisions were implemented as audit-grade opt-in support. The
current paper keeps the feature OFF for RedeRio and reports the realised FPR
overshoot as a limitation/future-work driver rather than as solved.

---

## 8. References

- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme
  Values*. Springer. §4.2 (GPD validity), §4.3.4 (declustering).
- Davison, A. C. & Smith, R. L. (1990). *Models for exceedances over
  high thresholds*. JRSS B 52(3), 393–442.
- Hosking, J. R. M. & Wallis, J. R. (1987). *Parameter and quantile
  estimation for the generalised Pareto distribution*. Technometrics
  29(3), 339–349. PWM fallback already shipped in PATCH 2026-05-06.
- Pickands, J. (1975). *Statistical inference using extreme order
  statistics*. Annals of Statistics 3(1), 119–131.
- Siffer, A. et al. (2017). *Anomaly Detection in Streams with Extreme
  Value Theory*. KDD. Justification of regime-conditional EVT.
- Joesang, A. (2016). *Subjective Logic*. Springer. §3.5.4 (frame
  coarsening, used to keep per-regime evidence comparable across
  buckets at the trapezoidal-map output).
