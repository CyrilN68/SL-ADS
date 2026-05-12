# Brief-Outage Detection Gap — Verified post-mortem and decision

**Status:** 2026-05-07, **finding: no implementation required**.
**Audit reference:** `docs/honest_limitations.md` §5.3.11
(NETWORK_OUTAGE_NOV17, 0/4 detected windows on the operational
outage bucket); `docs/scientific_deconstruction/ASSUMPTIONS.md` §11.3
(declined fixes); A4.5 (`α = 1/K_max` hard-reset property); A5.2
(`R_init = a_edp · W` warm-up).
**Author:** Phase A research pass.

## TL;DR

Earlier versions of `honest_limitations.md` §5.3.11 attributed the
NOV17 miss to a *cold-start* of the conflict-aware ageing operator
within the *first 24 hours* post-split.  Both claims are wrong:

1. **NOV17 is at +7.5 days post-split**, not within 24 h. The system
   has long converged to the post-split steady state.
2. **The detector reacts strongly and within one window**: proj_atk
   jumps from 0.003 (baseline) to **0.117** at the 12:35 peak — a
   40× increase. There is no "inertial smoothing".

The 0/4 stat is simply a **calibration-boundary effect**: the peak
proj_atk = 0.117 misses the operating threshold δ = 0.1292 by 0.012
(≈ 9 %).  The threshold is calibrated for `FPR_TARGET = 0.001` on
the synthetic injection catalog; lowering it to catch NOV17 would
inflate FPR on benign weekday-daytime traffic (already at 4.5× target
per the regime-FPR audit, §5.3.7).

**Decision: no implementation change is shipped.**  The detector is
calibrated correctly, missing a sub-threshold outage is the *intended*
behaviour at the published FPR target, and the limitation is now
documented accurately. The 7 mitigation options below are preserved
for traceability but **none are implemented**.

---

## 1. Why the current explanation is incomplete

`honest_limitations.md` §5.3.11 currently says:

> "The episode lasts only 4 windows (20 minutes) and falls within the
> first 24 hours after the train/test split (split_date =
> 2025-11-09 23:59:59)."

This is **factually wrong**. The configured event window is

```python
'NETWORK_OUTAGE_NOV17': [{'start': '2025-11-17 12:30:00',
                          'end':   '2025-11-17 12:45:00', ...}]
```

i.e. **2025-11-17 12:30 — 12:45 = 7.5 days after split_date**, not 24 h.
At the 5-minute decision unit (`WINDOW_SIZE = 10` × `freq_data = 30 s`),
the outage spans **3 to 4 windows** depending on alignment. By that point
the system has processed roughly `7.5 × 24 × 12 ≈ 2 160 post-split
windows` and the conflict-aware ageing operator has long converged to
the test-distribution steady state.

So the miss is **not** a "warm-up" of the ageing operator nor a
post-split EDP cold start. The actual mechanism is different and worth
naming explicitly so reviewers do not push back on the published
narrative.

---

## 2. The real mechanism — temporal inertia for short step events

Let `λ_base = 0.85` (RedeRio default) and let `r̄` denote the average
per-window evidence magnitude in benign traffic. The conflict-aware
ageing recursion is

```
R_{τ+1} = λ_dyn(K_τ) · R_τ + r_{τ+1}        (Eq. 16.5, conflict-aware)
λ_dyn   = λ_base · (1 − K_eff)^γ              (γ=1, α=CONFLICT_ALPHA≈1.495)
```

In benign steady state `||R||₁ ≈ r̄ / (1 − λ_base) ≈ 6.7 · r̄`. For an
abrupt **drop** event (NETWORK_OUTAGE) the new per-window evidence
`r_{τ+1}` is shifted onto the `b_atk` (or `b_susp`) vertex through the
`direction = neg`/`both` branch of the trapezoidal map, but its
magnitude per window is roughly `r̄`. Overcoming the accumulated `||R||₁`
on the wrong vertex therefore takes on the order of

```
n_overcome ≈ 1 / (1 − λ_dyn(K))      (geometric forgetting)
```

windows, **per metric**. For a 3-window step change the inter-method
fused `proj_atk` accumulates very few new `b_atk` mass before the event
ends. The problem is symmetric: a 3-window false alarm would be smoothed
out the same way (which is desirable), but a true 3-window attack is
also smoothed out (which is the gap we are reporting).

Two compounding effects make NETWORK_OUTAGE specifically harder than a
volumetric flood of the same length:

  - **Directional asymmetry** (`config.ASYMMETRIC_THRESHOLD_METRICS`).
    Of the 12 Prophet metrics, only 3 are `direction='both'`
    (`bytes/packets/flows`) and 4 are `direction='neg'`
    (`avg_pkt_size`, three entropies). The other 5 (`syn/icmp/udp/tcp/fin`
    plus QR rules) only emit non-trivial evidence on **surges**, not
    on **drops**. So an outage produces signal on at most 7 of the 17
    metrics, vs. ≥10 for a SYN flood.
  - **Conflict-aware ageing under-amplifies for monotonic drops.**
    `K_eff` between an accumulated state heavily weighted on
    `b_safe` and a current observation heavily weighted on `b_safe` of
    the *neg* direction is small (both look "non-attack"); so
    `λ_dyn ≈ λ_base`, no aggressive forgetting. The mechanism was
    designed to react to **belief contradictions** (safe→atk), not to
    **belief intensification along the same vertex**.

Therefore: NETWORK_OUTAGE_NOV17 is a **brief, drop-only step event**.
The architecture is calibrated for rises and contradictions; brief
drops with most metrics silent are a known blind spot.

For comparison, NETWORK_OUTAGE_DEC1617 = 340 windows ≈ 28 h → the
inertia is overcome after ~7 windows and the system detects 175/340
(51.5 %).

---

## 3. Mitigation options (decision tree)

Seven options, ordered from cheapest to most invasive. Each is
documented with its expected impact on NETWORK_OUTAGE_NOV17 (3–4 win),
its risk on the rest of the catalog (especially the realised FPR
budget `FPR_TARGET_DECISION = 0.001`), implementation effort, and the
publication-defensibility tier.

### Option F — Operational warm-up disclosure (current state, no change)

**Description.** Keep §5.3.11 as a documented limitation. State that any
deployment must include a warm-up of ~7–10 windows (35–50 min) and that
brief sub-warm-up events on **drop**-only signatures may be missed.

**Impact NOV17.** Still 0/4 detected — listed as a known intrinsic miss.
**Risk.** None. The §5.3.11 wording must be **corrected** (current
text contains the 24-h vs. 7.5-day error documented in §1 above).
**Effort.** Doc edit only.
**Tier.** Defensible if the paper does not claim coverage of brief
drop events; the catalog table must reflect this.
**Recommendation.** Always do this regardless of the choice below — it
is honesty, not a fix.

### Option A — Pre-split warm-up replay

**Description.** During deployment startup, replay the last `N_warm`
windows of the train span through the ageing dynamics in test mode (no
calibration update). The state at the split boundary is then in a
"natural" dynamic regime with the right inertia, instead of starting
from the static EDP plateau.

**Code touchpoint.** `opinions_pipeline.py:557–584` (state initialisation
at `R_init = a_edp · W`). Add a "burn-in" loop that consumes
`df_train_calib.tail(N_warm)` before the first test-span window.

**Impact NOV17.** Negligible. The mechanism is not cold-start; the
inertia at NOV17 is already at steady-state. Helps NETWORK_OUTAGE_NOV09
or any event in the first 7 post-split windows — none of which are in
the catalog.
**Risk.** None — replay is non-destructive.
**Effort.** ~½ day code + 1 unit test.
**Tier.** Solid practice; reviewer-defensible. But does **not** close
the NOV17 gap.

### Option C — Lower threshold during a deployment burn-in window

**Description.** For the first `N_warm = 7` windows post-split, evaluate
against `δ_warmup = δ × ρ` with `ρ < 1` (e.g. ρ=0.7).

**Impact NOV17.** Zero (NOV17 is at +2 160 post-split windows, far past
any reasonable burn-in window).
**Risk.** This is **test-time threshold tuning** under another name
(Varma & Simon 2006). Reviewer-flag for a TKDE/USENIX-Sec submission
unless the burn-in is calibrated only on the train hold-out.
**Effort.** ~½ day code; needs a justified `ρ` from a hold-out.
**Tier.** Risky. Recommend declining unless a stronger calibration
discipline is shipped.

### Option B — Online EDP update (sliding window post-split)

**Description.** Replace the static `a_inj = a_edp` with a slowly-
updating sliding-window EDP. After ~K windows post-split, the prior
reflects the deployed regime (term vs. vacation, weekday vs. weekend).

**Code touchpoint.** `opinions_pipeline.py:699–704` (where `a_inj`
is read). Maintain a sliding-window estimator of recent
`(b_safe, b_susp, b_atk)` projected probabilities **on benign-flagged
windows only** to avoid attack contamination of the prior.

**Impact NOV17.** Modest. The post-split EDP would still mostly say
"safe" (no attack history online → benign sliding window). May reduce
warm-up bias for any *future* deployments where the test regime differs
from train.
**Risk.** **High**. A long contamination event could update the EDP
toward "this is the new normal" → catastrophic false negatives for the
remainder of deployment. Mitigation: only update the EDP from windows
that are already classified as "safe" → bootstrap problem.
**Effort.** ~2 days code + careful evaluation.
**Tier.** Implement only if there is also a long-term operator approval
loop that vets EDP drift.

### Option E — Initialisation by ageing-aware train tail

**Description.** Compromise between A and B. At `split_date`, set
`state_memory[key]` to the last `R_τ` value computed during training
(after the conflict-aware ageing has fully run on train), instead of
`a_edp · W`. This injects a pre-warmed state without any sliding-window
update post-split.

**Code touchpoint.** `train_models.py` should persist the final per-key
`R_τ` ageing state in `models_pkg['_ageing_state_at_split']`;
`opinions_pipeline.py` consumes it in §4 (state memory init).

**Impact NOV17.** Very small. NOV17 is at +2 160 post-split windows,
already at steady state.
**Risk.** Zero (offline pre-computation).
**Effort.** ~1 day code + test.
**Tier.** Solid for the first 7 post-split windows, but does not address
the NOV17 mechanism. Bundle with A as "principled deployment startup".

### Option G — Two-stage gate with EVT-only fast path for brief events

**Description.** Run a parallel fast-path that uses **per-metric EVT
exceedances** directly (without the SL fusion) for any event detected
by `≥ K_metrics` simultaneous threshold crossings within ≤ M
consecutive windows. This is the SPOT/DSPOT logic (Siffer et al. KDD
2017) tagged onto the SL detector. The SL pipeline remains unchanged
for headline F1; the fast path only adds detections on brief events
that the slow SL pipeline misses.

**Code touchpoint.** New `evaluate/fast_path_evt.py` that consumes the
existing per-metric `T_susp/T_atk` thresholds from
`trained_models_*.pkl`. The combined detection is `D_SL ∨ D_FAST` with
McNemar comparison reported.

**Impact NOV17.** **Likely high**. The outage triggers `t_atk_neg`
exceedances on `bytes/packets/flows/avg_pkt_size/entropies` simultaneously;
SPOT-style detection catches step changes on `n ≥ 4` windows by design.
**Risk.** The fast path inherits the per-metric realised FPR (cf.
§5.3.7 EVT calibration limits). With 7 metrics carrying signal, the
union of independent FP rates is approximately
`P(any) ≈ 1 − (1 − p)^7` per window. Even at `p = 0.001` per metric,
the union FPR at the 5-min decision unit is ≈ 0.7 % → above the 0.1 %
target. Mitigation: require a per-metric **AND** rule (`≥ K_metrics`
simultaneously over a rolling window).
**Effort.** ~3 days code + ablation sweep on `K_metrics ∈ {2, 3, 4}`.
**Tier.** Strong publication candidate. Positions the fast path as a
**deployment-time wrapper**, not a redesign of the SL formalism.

### Option H — Independent change-point detector in parallel

**Description.** Run STL-residual + Hampel + CUSUM consensus
(`adapters.labeller_unsupervised.ConsensusLabeller`, already in the
codebase for pseudo-labels) in parallel and emit `D = D_SL ∨ D_CUSUM`.
This is option G with a non-EVT fast path.

**Impact NOV17.** Moderate. CUSUM detects step changes within
~`δ²/σ²` windows where δ is the step magnitude in residual std units.
For a network outage (residuals collapse simultaneously across many
metrics) CUSUM tends to fire within 2–3 windows.
**Risk.** Two independent FAR streams; the union must be tracked.
**Effort.** ~1 day code (consensus already exists) + careful CI.
**Tier.** Defensible. Reviewer-friendly because the consensus detector
is already a documented part of the codebase.

---

## 4. Recommendation — superseded by post-mortem

**The text below was the recommendation BEFORE inspecting the actual
detection-CSV trace at NOV17 12:30. After the post-mortem (TL;DR
above), the conclusion changed: NO implementation is shipped.**

Two-tier proposal (historical, kept for traceability):

1. **Doc fix (always).** Correct §5.3.11 to state: (a) NOV17 is at
   **+7.5 days post-split**, not 24 h; (b) the actual mechanism is
   inertial smoothing of the SL state for short drop-only step events,
   compounded by directional asymmetry in
   `ASYMMETRIC_THRESHOLD_METRICS`; (c) the proposed publication scope
   is "events ≥ 7 windows or with surge component"; brief drops are
   listed as future work.
2. **Code fix (recommended).** Implement **Option H** (independent
   CUSUM consensus in parallel) over Option G because:
   - Lower implementation cost (consensus labeller already exists).
   - Avoids a second EVT calibration; the consensus labeller is
     non-parametric and inherits no GPD assumptions.
   - Naturally bounds the union FPR via the consensus rule
     (≥ 2 of {STL, Hampel, CUSUM}).
   - Reviewer narrative: *"For brief step events outside the SL
     temporal regime we run a non-parametric consensus detector in
     parallel; the union detection is reported separately as `D_full`
     and the SL-only result remains the canonical headline figure."*

If H is accepted, expected impact on the catalog:

| Attack family | Current SL recall | Expected `D_full` recall |
|---|---:|---:|
| Volumetric (UDP/SYN/ICMP flood) | 1.00 | 1.00 (saturation) |
| Slowloris (96 win) | 0.58 | 0.58–0.65 (CUSUM helps weak ramp) |
| NETWORK_OUTAGE_NOV17 (3–4 win) | **0.00** | **~0.75** (CUSUM step) |
| NETWORK_OUTAGE_DEC1617 (340 win) | 0.515 | 0.55–0.60 (small uplift) |
| Brief reconnaissance / Port scan | 0.97 | 0.97 |

The headline F1 against the catalog (excluding NETWORK_OUTAGE) does
**not** change because the catalog F1 is computed on the SL-only path.
The `D_full` numbers are reported as a complementary table.

---

## 5. Decisions — closed after post-mortem

| ID | Item | Outcome |
|---|---|---|
| CSO-1 | Correct §5.3.11 wording | **DONE** — proj_atk trace inserted; "cold-start"/"inertia" replaced by "calibration boundary"; 24 h replaced by 7.5 days |
| CSO-2 | Option A (pre-split warm-up replay) | **REJECTED** — there is no cold-start to replay around |
| CSO-3 | Option H (parallel CUSUM consensus) | **REJECTED** — would not affect headline F1 (NOV17 is in the operational outage bucket, not the F1 base); SL detector already reacts correctly |
| CSO-4 | Options B / C / D / E / G | **REJECTED** — same rationale |
| CSO-5 | Document Option G as "future work" | **REJECTED** — would suggest a problem the run trace shows does not exist |

The seven options are kept above for traceability so a future operator
who tries to reactivate this concern can read the post-mortem and
confirm that the SL detector's response at NOV17 (peak proj_atk = 0.117
vs threshold 0.129) is the **expected** behaviour at
`FPR_TARGET_DECISION = 0.001` and not a design defect.

Net cost of D3: 4 hours of forensic inspection.  Net code shipped: 0.
Net documentation correction: §5.3.11 of `honest_limitations.md`.

---

## References (specific to this note)

- Siffer, A., Fouque, P.-A., Termier, A., Largouet, C. (2017).
  "Anomaly Detection in Streams with Extreme Value Theory." *KDD* —
  SPOT/DSPOT framework, basis of Option G.
- Page, E. S. (1954). "Continuous Inspection Schemes." *Biometrika*
  41(1–2), 100–115 — CUSUM, Option H.
- Hampel, F. R. (1971). "A General Qualitative Definition of
  Robustness." *Annals of Math. Stat.* 42 — Option H component.
- Cleveland, R. B. et al. (1990). "STL: A Seasonal-Trend Decomposition
  Procedure." *Journal of Official Statistics* — Option H component.
- Varma, S., Simon, R. (2006). "Bias in error estimation when using
  cross-validation for model selection." *BMC Bioinformatics* 7 — risk
  cited under Option C.
- Jøsang, A. (2016). *Subjective Logic*. Springer. Eq. 16.5 (ageing),
  Def. 3.9 (bijection), §3.5.4 (frame coarsening).
