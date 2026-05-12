# Reproducibility checklist — SL-ADS RedeRio reference run

Following the NeurIPS 2024 Reproducibility Checklist
(`https://neurips.cc/public/guides/PaperChecklist`) and the ACM artifact
review template, this page documents every reproducibility-relevant
property of the SL-ADS pipeline. The complete 17-leaf RedeRio rerun finished on
2026-05-12 (`run_id=2e12261d55a8f975`); paper-facing numbers are bound to that
config and output archive.

For the current audit state, start with `docs/AUDIT_CURRENT_STATUS.md`.

## Code

- [x] **Code published** — `current_version/` of this repository (MIT
      License, `LICENSE` at repo root).
- [x] **Entrypoints** — `run_pipeline.py` for the full pipeline; per-step
      modules under `src/sl_ads/{train,inject,core,qualify,evaluate,
      ablation,audit,compare}`.
- [x] **Pinned environment** — `requirements.txt` and `pyproject.toml`
      lock all Python dependencies.  Stan/Prophet (used by Prophet) is
      pinned via the `prophet` package; cmdstanpy is bundled.
- [x] **OS / hardware footprint** — single-machine (Windows 11 / Linux /
      macOS); no GPU required. On current RedeRio data, the full pipeline can
      take several hours; `evidence` is usually the long step.
- [x] **Random seeds** — `SL_RANDOM_SEED` env-var controls the global
      seed (default 0).  Every stochastic component (Prophet/Stan,
      bootstrap CI, McNemar, signature-noise ablation) reads from this
      single source.

## Data

- [x] **Dataset description** — RedeRio is a campus-network flow-level capture
      from UFRJ Brazil, 30 s aggregation, 12 core Prophet-modelled metrics.
      Standardised CSV at `data_standardized/RedeRio.csv`.
- [x] **Train / test split** — `CONFIG['split_date'] = 2025-11-09 23:59:59`.
      Train = pre-split, test = post-split.  PATCH TASK-22 enforces
      `df_train_calib` (last 25% of train) is held out from threshold
      calibration.
- [x] **Anti-leak** — A1.9 sidecar config check raises at runtime if
      any calibration-sensitive knob disagrees with the calibrated
      sidecar.  A3.5 catalog validator rejects any injected event with
      `start <= split_date`.
- [x] **Synthetic injection** — `CONFIG['INJECTED_ATTACK_CATALOG']`
      lists 13 deterministic attacks (calendar-anchored, no random
      seed); `evidence_level.py` injects them at the SL-evidence level.
- [x] **Real anomalies** — `CONFIG['EVAL']['REAL_ATTACK_CATALOG']`
      contains 1 real DDoS event (Nov 12, RedeRio operator log).
      Operational outages are listed in `REAL_ATTACKS['NETWORK_OUTAGE_*']`.

## Methodology

- [x] **Hyperparameters** — every threshold / knob is documented in
      `src/sl_ads/config.py`; the published reference uses the
      file-default values plus `WBF_WEIGHT_MODE='uniform'` and
      `INTER_METHOD_FUSION='wbf'` (PATCH M-11). ABF and the other
      inter-method operators are available through the same config key, but
      the strict 2026-05-07 WBF/ABF recalibration kept WBF as the reference.
- [x] **Selection criteria** — `δ` is auto-calibrated on
      `df_train_calib` for the operator-chosen FPR target
      (`FPR_TARGET_DECISION = 0.001` on RedeRio).  Headline F1 is
      reported at this fixed `δ` (PATCH TASK-34).
- [x] **No test-set tuning** — the calibrator never sees the test
      span; the McNemar / bootstrap CIs are computed at the calibrated
      `δ` only.
- [x] **Evaluation metrics** — primary `F1_micro_pure` on a binary
      window-level partition, reported under both publication protocols:
      `catalog_outages_separate` and `operator_faithful_anomaly`
      (`evaluation/eval_f1_protocol_comparison.csv`). Secondary metrics:
      `F1_macro_pure`, `MCC`, `Accuracy`, `Precision`, `TPR/FPR(window)`,
      plus range-aware `VUS-PR / VUS-ROC / R-AUC-PR / R-AUC-ROC`
      (Paparrizos et al. 2022).
- [x] **Confidence intervals** — block bootstrap (Künsch 1989) with
      `block_length = median attack-episode length = 36 windows`,
      `n_boot = 1000`.  Bootstrap method, resampling mode and block
      length are persisted in `eval_threshold_sweep.csv` for traceability.
- [x] **Statistical tests** — McNemar paired test
      (`stats/mcnemar.py`), Wilson 95% CI on FAR (Newey-West n_eff
      correction for autocorrelation, A7.3 fix 2026-05-06).
- [x] **Computational budget** — see "OS / hardware" above; per-step
      timings in `run_pipeline.py` log.

## Statistical rigour

- [x] **Multi-seed stability** — `src/sl_ads/evaluate/run_multi_seed.py`
      sweeps 5 seeds.  Multi-seed numbers are not in the headline table
      (deferred TASK-12); they are reported in Appendix C if available.
- [x] **Honest limitations** — `docs/honest_limitations.md` documents
      every known caveat (Slowloris recall gap, NETWORK_OUTAGE cold
      start, novelty AUC reporting-only, etc.).
- [x] **Independent and no-SL baselines** — `compare_no_sl_fair.py`
      reports the same-evidence ADS-without-SL comparators with train-calib
      thresholds and paired tests. `compare_raw_baselines_fair.py` reports
      raw-data IF / LOF / OCSVM / SGD-OCSVM / robust-z / PCA baselines on raw-valid protocols only and
      requires `opinions_non_injected/detection_results_RAW.csv` so the SL row
      is not contaminated by evidence-level synthetic injections.
      The legacy `compare_if_fair.py` remains available for pseudo-label
      agreement; its F1 must not be compared directly with catalog F1.

## Artefacts and audit trail

- [x] **Manifest** — `outputs/_run_manifest.json` records git SHA,
      config hash, input hashes and per-step timings (TASK-32).
- [x] **Fallback log** — `<model>_fallbacks.json` records every EVT or
      reconstruction fallback so a reviewer can trace which metrics
      hit the empirical-quantile / DummyRegressor branches.
- [x] **Sidecars** — `<model>_threshold.json`,
      optional `<model>_threshold_<mode>.json` sidecars, and
      `fusion_mode_at_compute_opinions.json` capture the calibration
      and runtime configuration so a reviewer can verify there is no
      surrogate-vs-deployed mismatch. Mode-specific sidecars are used for
      strict ablations such as WBF vs ABF and must not be treated as a new
      production threshold unless the target FPR is validated.
- [x] **Calendar-aware EVT signature (PATCH H2, 2026-05-07)** — when
      `CONFIG['CALENDAR_EVT_ENABLED'] = True` the threshold sidecar
      gains a `calendar_evt_signature` field (the versioned regime-fn
      signature, e.g.
      `"weekday_x_daytime_x_holiday/v1@2026-05-07"`).
      `paths.validate_threshold_sidecar_config` adds this to the
      sensitive-knob list and hard-raises on any drift between the
      calibration-time and runtime regime function. PKLs produced before
      H2 (`calendar_evt_signature` absent) are accepted in legacy mode
      with a `missing` field in the validator report — no hard failure.
- [x] **Hardening reports** — current hardening status is consolidated in
      `docs/AUDIT_CURRENT_STATUS.md`, `docs/audit/audit_verification_tracker.md`,
      and `docs/scientific_deconstruction/ASSUMPTIONS.md`. Dated historical
      hardening reports are preserved under `docs/archive/`.

## Known limits and disclosures

- [x] **Single-dataset evaluation** — RedeRio only.  Cross-dataset
      adapters exist for METR-LA, GECCO-IoT and CESNET-TimeSeries24
      but headline numbers are not aggregated across datasets.
- [x] **Modern SOTA baselines** —  Isolation Forest is the only
      fair-FPR baseline in the headline.  TranAD / Anomaly Transformer
      / TimesNet skeletons exist (`src/sl_ads/compare/compare_*.py`)
      but full evaluation is deferred (TKDE/VLDB scope, see the archived
      2026-05-04 hardening note and current status file).
- [x] **Closed-world qualifier** — the SL-template qualifier
      (`sbn_qualifier.py`) is calibrated on 13 attack types from
      CIC-IDS2017 / UNSW-NB15 / Kitsune; novelty is handled by the
      `Autre_Anomalie` residual class but its calibration is reported
      as `reporting-only` (PATCH-C2).
- [x] **Naive Bayes assumption violations** — A6.1 audit (2026-05-06)
      reports 32/66 group pairs with HIGH dependence on attack
      windows; the qualifier QP is therefore an upper-bound estimate
      under correlated evidence (Domingos & Pazzani 1997 robustness
      regime).

## Author contact

For reproducibility questions or artifact-evaluation requests:
- Issues: GitHub issue tracker on this repository
- Citation: `CITATION.cff` at repository root
