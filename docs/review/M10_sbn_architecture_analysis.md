# M-10 / F17 — SBN Architecture Analysis & Feasibility Study

**Status:** PATCH M-10 / F17 (2026-04-24) — code-side artefact.
**Audit item:** archived `CONSOLIDATED_AUDIT_REVIEW.md` §1.2, MAJOR.
**Scope:** `qualify_anomaly_sbn.py` and the paper sections that introduce
the term *Subjective Bayesian Network* (SBN).

**2026-05-04 terminology update:** the preferred manuscript term is now
**expert-template-driven Subjective Logic qualifier** or **SL-template
qualifier (SL-TQ)**.  The file/module/column names containing `sbn` are
legacy compatibility names only.

---

## 1. Executive summary

The code in `qualify_anomaly_sbn.py` is **not** a canonical Subjective
Bayesian Network in the sense of Jøsang 2016 Chapter 14 (propagation on a
directed acyclic graph of SL opinions via conditional SL deduction). It
is an **expert-template-driven Subjective Logic qualifier** whose
pipeline is a closed-form discriminative scorer followed by the SL
evidence–opinion bijection. This document:

1. Inventories, line by line, what the current code actually does
   (Section 2).
2. States the canonical SBN definition and identifies what is missing
   (Section 3).
3. Analyses the feasibility of upgrading to a genuine SBN (Section 4).
4. Argues why the current expert-template form is a principled and
   defensible choice for the paper's problem statement (Section 5).
5. Provides the exact terminology to use throughout the paper, with
   reviewer-safe citations (Section 6).
6. Lists testable claims the reviewer can verify against the code
   (Section 7).

**Bottom line.** A literal SBN implementation is technically possible
but would require conditional-opinion tables we do not have the
statistical power to estimate from public IDS datasets (see §4). The
current architecture — expert templates combined with the standard SL
bijection — is the correct engineering compromise for a reproducible,
template-driven IDS qualifier. The paper must name it accordingly.

---

## 2. Code inventory — what `qualify_anomaly_sbn.py` actually does

The function `sbn_qualify_row` (`qualify_anomaly_sbn.py` L950–L1240)
executes the following pipeline for each decision window:

### L1 — Anomaly gate (L1025–L1028)
```python
p_atk = row.get(_DET_COL, row.get('FINAL_SYSTEM_CBF_proj_atk', 0.0))
if pd.isna(p_atk) or p_atk < threshold:
    return _empty_result()
```
A window below the detection threshold is skipped — qualification only
runs on windows already flagged as anomalous by the upstream SL fusion
chain. **This is not part of an SBN: it is a deterministic gate.**

### L2 — Group projection (L594–L640 `_compute_group_projected`)
For each semantic group `g` (e.g. `volume`, `protocols_tcp`,
`reconstruction`) and each ternary state `s ∈ {Safe, Susp, Anom}`:
```python
P^g_s = geomean_{m ∈ group}( P^m_s )
```
The per-metric projected probabilities `P^m_s` come pre-computed from
the CSV produced by `src/sl_ads/core/opinions_pipeline.py` (i.e. they are already the
output of the SL bijection at the metric level). The group-level pool
is the **geometric mean** (logarithmic opinion pooling — Genest & Zidek
1986 §4, *Statistical Science*). This is an aggregation choice, **not**
a Bayesian inference operation.

### L3 — Per-group compatibility score (L643–L674 `_sbn_group_score`)
For each attack type `k` and each active group `g`:
```python
Score(k, g) = Σ_s P^{obs}_{g,s} · c^{k|g}_s
```
where `c^{k|g}_s = P(G=s | type_k)` comes from
`SBN_COND_OPINIONS` in `config.py` — **a hand-authored expert table**,
not a learned conditional probability distribution. The docstring
itself acknowledges (L662–L665):

> *"Note : ce n'est PAS P(obs|k) au sens d'un modèle génératif
> bayésien strict. C'est une approximation par proxy discrétisé"*

### L4 — Evidence summation (L685–L707 `_evidence_sum_scores`)
```python
e(k) = Σ_g max(0, Score(k,g) - 1/3) · evidence_scale
```
This is a **rectified expert-signal accumulator** — per group, we
discount the neutrality floor 1/3 and sum positive excess across
groups. This is neither a likelihood product nor a posterior
update: it is a bespoke evidence-engineering operator. It is clearly
documented as such (L697–L699):

> *"Interprétation Bayes Factor : l'écart à la neutralité 1/3 est
> équivalent au log-odd d'un test d'hypothèse contre H0 (neutralité
> uniforme)"*

### L5 — SL bijection (L709–L757 `_sl_bijection`)
```python
b(k) = e(k) / D ,  u = W / D ,  D = Σ_k e(k) + W  (W = K)
```
This is the **standard multinomial SL evidence-to-opinion bijection**
(Jøsang 2016 Def. 3.9, §3.5.2) with W = K (number of hypotheses) — i.e.
the canonical prior-weight choice that yields `u = 1` under zero
evidence.

### L6 — Temporal prior via kill-chain transition (L1146–L1205)
Optional Markovian prior: previous window's opinion is mapped through a
kill-chain transition matrix `T[k_prev][k_next]`, exponentially
discounted by `λ^Δt` (Jøsang 2016 Def. 14.6 trust discounting), then
fused with the current opinion via `_wbf_two` (a bounded two-source
weighted belief fusion with a **fixed** weight `temporal_weight`). This
is the one part of the pipeline that *does* operate on a probabilistic
graph (the transition matrix), but it is a first-order Markov chain on
types, not a full SBN.

### L7 — Uncertainty Maximisation (L760–L808 `_apply_um`)
Optional Jøsang 2016 Eq. 3.27 — re-allocates probability mass from the
attack base rate toward `u`, amplifying the novelty signal.

### Summary of pipeline arithmetic
```
observed metric opinions (from CSV, already SL bijected)
    ↓  geometric mean per group (L2)
group-level projections P^obs_{g,s}
    ↓  dot-product against expert tables c^{k|g}_s (L3)
compatibility scores Score(k, g)
    ↓  rectified neutrality-discounted sum (L4)
evidence counts e(k)
    ↓  SL bijection with W = K (L5)
raw opinion (b^raw, u^raw)
    ↓  optional Markov temporal prior via WBF (L6)
    ↓  optional Uncertainty Maximisation (L7)
final opinion (b_sbn, u_sbn)
```

**None of the steps L2, L3, or L4 is a canonical SBN operation.** Only
L5 and L6 use genuine Jøsang operators (bijection, WBF, trust
discounting).

---

## 3. Canonical Subjective Bayesian Network (Jøsang 2016 Ch. 14)

### 3.1 Definition
A *Subjective Bayesian Network* is a directed acyclic graph `G = (V, E)`
of random variables where:

1. Each node `X_i` carries an SL opinion `ω(X_i)` over its state space.
2. Each directed edge `X_j → X_i` carries a **conditional SL opinion**
   `ω(X_i | X_j = x_j)` for every value of `X_j`, satisfying the SL
   axioms (Σb + u = 1, base rate on simplex).
3. Marginal opinions on a node are obtained by **conditional SL
   deduction** (Jøsang Eq. 14.1–14.12) propagated through the DAG.

### 3.2 What distinguishes a true SBN from expert scoring
| Requirement | Canonical SBN | `qualify_anomaly_sbn.py` |
|---|---|---|
| DAG over random variables | Yes, explicit | No — flat: groups → types |
| Conditional opinions on each edge | Yes, full SL opinion | No — expert probability **vectors** `c^{k\|g}_s`, not opinions (no `u_cond`) |
| SL deduction propagation (Eq. 14.1–14.12) | Yes | No — replaced by rectified dot-product + bijection |
| Handles multi-parent conditionals | Yes (with independence or covariance) | No — each group scored independently |
| Learns conditional opinions from data | Yes (posterior SL update) | No — hand-authored from literature |

### 3.3 The specific Jøsang operator that is NOT used
Jøsang 2016 **Eq. 14.4 — conditional deduction on a single edge**:
```
ω(X ‖ Y) = ω(Y) ⊛ [ ω(X|y) for y ∈ dom(Y) ]
```
where `⊛` is the SL conditional deduction operator. The code replaces
this by:
```
Score(k, g) = Σ_s P^{obs}_{g,s} · c^{k|g}_s
```
which is the **projected-probability expectation of the conditional
probability vector** — algebraically a dot product, not an SL
deduction. The two coincide only when `u^{obs}_g = 0` AND `u^{k|g} = 0`
AND a specific choice of base rate — i.e. in a degenerate dogmatic
corner of the input space. **The current code does not lie in that
corner in general.**

---

## 4. Feasibility of upgrading to a canonical SBN

### 4.1 What a canonical upgrade would require

To turn `qualify_anomaly_sbn.py` into a genuine SBN, we would need:

1. **Full conditional SL opinions** `ω(X_i | X_j = x_j)` for every
   (parent, child, parent-value) triple — i.e. a distribution *and* an
   uncertainty mass `u_cond^{k|g}` on each edge. The current
   `SBN_COND_OPINIONS` table provides only the distribution.
2. **A data-driven estimation procedure** for these conditional
   opinions, with statistical justification for the uncertainty
   parameter (e.g. Jøsang 2016 §3.5.3 — Dirichlet evidence prior with
   W pseudo-counts).
3. **A full implementation of conditional SL deduction** (Jøsang 2016
   Eq. 14.1–14.12) including its covariance-aware multi-parent
   extension (Eq. 14.16–14.25).

### 4.2 Data-feasibility assessment

On the RedeRio dataset (and on every public IDS benchmark we surveyed
— CIC-IDS 2017, UNSW-NB15, Kitsune):

- Attack windows per attack type are sparse: in the RedeRio active split
  we see O(10¹)–O(10²) windows per type, heavily imbalanced. Estimating
  a 3-state conditional per group at this sample size yields Dirichlet
  credible intervals roughly `±0.10–0.20` on each component — i.e. the
  learned conditional would be **more uncertain than the expert
  template**.
- The alternative (elicit conditional opinions from an expert) IS what
  the code already does. Upgrading from an expert probability vector to
  an expert SL opinion would require additionally eliciting `u_cond`
  per edge, with no data to validate the elicitation.
- Conditional deduction propagation is O(|types| × |groups|) per
  window even in the simplest one-level DAG; a full multi-parent
  extension is O(|types|^|parents|). The dot-product approximation is
  O(|types| × |groups|) — same asymptotic cost, but orders of magnitude
  less code complexity.

**Conclusion.** A canonical SBN upgrade is blocked by data scarcity,
not by algorithmic capability. The paper's problem — qualifying a
*small number* of attack types from a *small number* of noisy
aggregated group signals — is not in the regime where a full SBN
provides a signal-to-noise advantage over expert templates.

### 4.3 What the current architecture gains by not being a canonical SBN

1. **Exact reproducibility** — the expert tables in `config.py` are
   fully documented and auditable; no learned parameters.
2. **Interpretability** — the dot-product `Σ_s P^{obs}_{g,s} c^{k|g}_s`
   is directly readable as "expected compatibility of the observed
   group with the type's signature".
3. **Stability under class imbalance** — expert templates do not
   overfit the dominant attack class.
4. **Principled uncertainty via the bijection** — `u = W/(W + Σe(k))`
   still captures absence-of-evidence correctly.

---

## 5. Paper terminology — the canonical correction

Throughout the paper, replace every occurrence of:

> *"Subjective Bayesian Network (SBN) qualifier"* / *"SBN"*

with one of the following (preferred listed first):

1. **"Expert-template-driven Subjective Logic qualifier"** — most
   faithful to the architecture, reader-safe for SL reviewers.
2. **"Naïve-Bayes-style Subjective Logic pooling"** — shorter, invokes
   the conditional-independence assumption (Rish 2001, IJCAI; Mitchell
   1997 Ch. 6) that genuinely is made in the code.
3. **"Rectified dot-product scoring with SL bijection"** — most
   literal, best for a methodology subsection heading.

The **first occurrence** (abstract / introduction) must define the term
exactly once, e.g.:

> *"We use an expert-template-driven Subjective Logic qualifier
> (sometimes colloquially called SBN in earlier drafts of this work)
> that computes per-group compatibility scores against hand-authored
> type signatures, aggregates them into evidence masses, and maps the
> evidence to an SL opinion via the standard bijection (Jøsang 2016
> Def. 3.9). Unlike a canonical Subjective Bayesian Network (Jøsang
> 2016 Ch. 14), the qualifier does not implement conditional SL
> deduction on a DAG — see `docs/review/M10_sbn_architecture_analysis.md`
> for the detailed comparison."*

---

## 6. Code-side rename option (optional but recommended)

To make the terminology alignment visible in the repository, consider
renaming symbols **in a future non-urgent PR** — this is NOT required
for the current paper revision but removes all residual confusion:

| Current | Proposed |
|---|---|
| `qualify_anomaly_sbn.py` | `qualify_anomaly_expert.py` or `qualify_anomaly_template.py` |
| `sbn_qualify_row` | `expert_template_qualify_row` |
| `SBN_COND_OPINIONS` | `EXPERT_CONDITIONAL_PROFILES` |
| `SBN_EVIDENCE_SCALE` | `TEMPLATE_EVIDENCE_SCALE` |
| `SBN_EVIDENCE_MODE` | `TEMPLATE_EVIDENCE_MODE` |
| `b_sbn_*`, `u_sbn`, `novelty_lr` | unchanged — already downstream public API |

Rationale for the partial scope: the column names `b_sbn_*` / `u_sbn`
are already written to hundreds of CSV outputs and are parsed by the
evaluation scripts — renaming them would break every downstream tool
and all saved runs. The *internal* symbols and the *module name* are
safe to rename.

**This rename is deliberately NOT executed in PATCH M-10 / F17** —
the paper-side terminology fix is sufficient to close the audit item,
and the code rename is a refactoring job that should be batched with
other cosmetic cleanups.

---

## 7. Testable / falsifiable claims

Each claim below has a concrete verification method grounded in the
current code. A reviewer can reproduce these in under 5 minutes.

### C1. The "SBN" qualifier does not implement SL conditional deduction.
**Verify:** grep for `josang` / `deduction` / `\b⊛\b` in
`qualify_anomaly_sbn.py` — there is no invocation of an SL conditional
deduction operator. `_sbn_group_score` (L643–L674) is algebraically a
dot product; the function name `_sbn_group_score` is a historical label.

### C2. The conditional "opinions" are probability vectors, not SL opinions.
**Verify:** inspect `SBN_COND_OPINIONS` in `config.py`. Each entry is a
dict `{Safe: float, Susp: float, Anom: float}` with `Σ = 1`. No `u`
component is defined. `_normalize_cond_opinion` (L677–L682) enforces
`Σ = 1`, confirming the no-uncertainty structure.

### C3. The evidence accumulator is a rectified expert signal, not a likelihood.
**Verify:** `_evidence_sum_scores` (L685–L707) computes
`Σ_g max(0, Score − 1/3) × evidence_scale`. A true likelihood
`P(observation | type_k)` would be a **product** over independent
groups (or a log-sum for numerical stability), not a rectified sum of
excesses. The docstring explicitly flags this (L697–L699).

### C4. Only L5 (SL bijection) and L6 (temporal WBF + trust discount) use canonical Jøsang operators.
**Verify:** `_sl_bijection` (L709–L757) cites Jøsang 2016 Def. 3.9;
`_discount_opinion` (L839–L870) cites Jøsang 2016 Def. 14.6;
`_wbf_two` (L810–L838) implements the 2-source WBF. Every other
operator in the file is bespoke.

### C5. The architecture is a naïve-Bayes-style pool in disguise.
**Verify:** in the absolute mode (`SBN_EVIDENCE_MODE = "absolute"`,
L1120–L1128), `e(k)` is a **sum** of independently-computed group
scores. In log-space this is a sum of log-compatibilities, which is
exactly the naïve-Bayes log-posterior up to the rectification and the
evidence scaling. The docstring acknowledges this directly (L1069–L1077).

### C6. Upgrading to a canonical SBN requires conditional uncertainty we cannot estimate from public data.
**Verify:** count the per-type active-window samples in
`detection_results_INJECTED.csv` (column `injection_attack_type`) —
for every public IDS benchmark, min-class counts are under 200, which
is below the Dirichlet sample-size floor for estimating 3-component
conditional opinions with `u_cond < 0.1` (Jøsang 2016 §3.5.3, rule of
thumb `W × 10` evidence points per component).

---

## 8. Closing of audit item M-10 / F17

- **Paper-side action:** apply the terminology rename of Section 5
  throughout the abstract, introduction, methodology, and any figure
  caption that mentions "SBN".
- **Code-side action:** none required for publication (see Section 6
  for the deferred optional rename).
- **This document** is the long-form justification the reviewer can
  cite; its path is the one referenced in the paper's Section 5
  footnote.

---

**References used in this memo:**
- Jøsang, A. (2016) *Subjective Logic: A Formalism for Reasoning Under
  Uncertainty*, Springer. Chapters 3 (bijection), 12 (CBF/WBF),
  13 (trust networks), 14 (SBNs and conditional deduction).
- Genest, C. & Zidek, J.V. (1986) "Combining probability
  distributions: a critique and an annotated bibliography",
  *Statistical Science* 1(1):114–135.
- Rish, I. (2001) "An empirical study of the naive Bayes classifier",
  *IJCAI 2001 Workshop on Empirical Methods in AI*.
- Mitchell, T. (1997) *Machine Learning*, McGraw-Hill, Ch. 6.
- Duda, R.O. & Hart, P.E. (1973) *Pattern Classification and Scene
  Analysis*, Wiley, §2.6 (nearest-mean classifier).
- Aczél, J. & Daróczy, Z. (1975) *On Measures of Information and Their
  Characterizations*, Academic Press.
- Good, I.J. (1952) "Rational decisions", *Journal of the Royal
  Statistical Society, Series B* 14(1):107–114.
- Hutchins, E.M. et al. (2011) *Intelligence-Driven Computer Network
  Defense Informed by Analysis of Adversary Campaigns and Intrusion
  Kill Chains*, Lockheed Martin.
