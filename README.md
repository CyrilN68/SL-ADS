# SL-ADS: Subjective-Logic Anomaly Detection System

SL-ADS is a research prototype for evidence-aware anomaly detection on
time-series network telemetry. The default case study is the RedeRio/UFRJ
Brazilian network trace. The pipeline combines temporal forecasting residuals,
cross-feature reconstruction residuals, pre-SL evidence triplets, synthetic
attack injection, Subjective Logic opinion fusion, temporal ageing, and cause
attribution.
## What The Pipeline Does

The default full run executes:

```text
train -> evidence -> inject -> opinions -> eval_injection
      -> qualify_sbn -> eval_qualify -> ablation -> compare_if -> audit
```

At a high level:

- `train` fits Prophet-style temporal baselines and reconstruction baselines,
  calibrates residual thresholds, and writes the model artifact.
- `evidence` converts residuals for each time window into per-metric evidence
  triplets `(P, S, N)`. This is still pre-Subjective-Logic state.
- `inject` creates the controlled synthetic attack evaluation span by editing
  those evidence triplets.
- `opinions` converts the evidence triplets into Subjective Logic opinions,
  then applies ageing and fusion to obtain the system-level anomaly opinion.
- `eval_injection` computes detection metrics on the injected attack catalog
  and the configured real/anomaly events.
- `qualify_sbn` and `eval_qualify` attribute detections to attack/cause
  families. Historical names still contain `sbn`, but the current qualifier is
  an expert-template Subjective Logic qualifier, not a canonical Bayesian
  network.
- `ablation`, `compare_if`, and `audit` run robustness, baseline comparison,
  and consistency checks.

## Repository Layout

```text
.
|-- run_pipeline.py          # main launcher
|-- requirements.txt         # pinned runtime dependencies
|-- pyproject.toml           # package/test configuration
|-- src/sl_ads/              # implementation package
|   |-- train/               # model training, EVT thresholds, guardrails
|   |-- core/                # Subjective Logic operators and fusion
|   |-- inject/              # synthetic attack injection
|   |-- evaluate/            # detection and qualification metrics
|   |-- qualify/             # cause attribution
|   |-- ablation/            # controlled ablation studies
|   |-- audit/               # scientific audit utilities
|   |-- stats/               # statistical helpers
|   |-- calendar/            # optional calendar/regime logic
|   |-- adapters/            # per-dataset I/O adapters (RedeRio / METR-LA / GECCO / CESNET)
|   |-- compare/             # SL-vs-no-SL and raw-baseline comparison scripts
|   `-- notebooks/           # Marimo interactive notebooks
|-- tests/                   # pytest suite
|-- docs/                    # audit, limitations, methods, and paper notes
|-- outputs/                 # regenerated run output, git-ignored
`-- results/                 # regenerated archived runs, git-ignored
```

## Data Layout

Data files are not committed to git. By default, the configuration expects data
directories next to this repository:

```text
../data_standardized/RedeRio.csv
../data_standardized/METR_LA.csv
../data_standardized/GECCO.csv
../data_standardized/CESNET.csv
```

For the default RedeRio run, the standardized file is:

```text
../data_standardized/RedeRio.csv
```

The raw RedeRio path used by the adapter configuration is:

```text
../data/dataset_1310_2912_v30s.csv
```

## Setup

Python 3.13 is the tested public runtime. On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick Verification Without Private Data

A reviewer can verify the command-line entry point, pipeline configuration, and regular test suite without access to the private RedeRio/UFRJ trace.

If pytest is not already installed, first install the development extras:

```bash
python -m pip install -e ".[dev]"
```

Then run:

```bash
python run_pipeline.py --list-steps
python run_pipeline.py --dry-run
python -m pytest -q --tb=short
```

The full default pipeline run (`python run_pipeline.py`) and slow tests require the data files described in the Data Layout section.


## Basic Commands

Inspect the configured pipeline:

```bash
python run_pipeline.py --list-steps
python run_pipeline.py --dry-run
```

Run the full default RedeRio pipeline:

```bash
python run_pipeline.py
```

Run only part of the pipeline:

```bash
python run_pipeline.py --from-step train --to-step train
python run_pipeline.py --from-step compare_if --to-step audit
```

Run the paper-facing comparison add-ons after a complete pipeline run:

PowerShell:

```powershell
python -m sl_ads.compare.compare_no_sl_fair
$env:SL_FORCE_NONINJECTED_OPINIONS = "1"
$env:SL_SKIP_OPINION_PLOTS = "1"
python -m sl_ads.core.opinions_pipeline
$env:SL_FORCE_NONINJECTED_OPINIONS = $null
$env:SL_SKIP_OPINION_PLOTS = $null
python -m sl_ads.compare.compare_raw_baselines_fair
```

`compare_no_sl_fair` is the direct "same ADS with vs without Subjective Logic"
comparison. The forced non-injected opinions command writes
`opinions_non_injected/detection_results_RAW.csv` without overwriting the main
injected result. `compare_raw_baselines_fair` trains IF / LOF / One-Class SVM /
SGDOneClassSVM / robust-z / PCA on raw network metrics and hard-requires that
non-injected SL CSV; it deliberately excludes synthetic attack windows because
the catalog attacks are injected at evidence level, not raw-traffic level.

Switch dataset:

```bash
python run_pipeline.py --dataset RedeRio
python run_pipeline.py --dataset METR-LA
python run_pipeline.py --dataset GECCO-IoT
python run_pipeline.py --dataset CESNET-TimeSeries24
```

The full RedeRio run can take several hours. The latest complete run took about
9h42 on a local Windows workstation, with the `evidence` step taking about
8h51 because it computes per-window evidence triplets for every active model
leaf over the full trace.

## Tests

Run the regular test suite:

```bash
pytest
```

Run tests with short tracebacks:

```bash
pytest -q --tb=short
```

Some tests marked `slow` require completed pipeline outputs on disk and are not
part of the default pytest configuration.

## Current Scientific Guardrails

Several guardrails are intentionally strict:

- The training artifact must contain all expected Prophet and reconstruction
  leaves. If Prophet fails silently, downstream evidence computation fails
  instead of producing reconstruction-only metrics.
- The threshold sidecar records the fusion mode, weighting mode, calibration
  target, and other configuration values used at calibration time.
- The current reference fusion mode is uniform WBF unless explicitly changed in
  `src/sl_ads/config.py`.
- MASE is stored as an audit/optional trust diagnostic for Prophet temporal
  models. It is not used by the current uniform-WBF headline metrics.
- MASE is undefined for cross-feature reconstruction models, so reconstruction
  entries intentionally store `NaN` for MASE. Reconstruction reliability is
  handled separately.

## How To Understand The Project

The README is only the entry point. A new reader can understand the full
scientific and engineering logic by following these files:

1. `docs/scientific_deconstruction/PIPELINE_LOGIC.md` - end-to-end reasoning
   chain from raw telemetry to binary anomaly decisions and cause attribution.
2. `docs/scientific_deconstruction/METHODS.md` - formal inventory of the
   forecasting, reconstruction, EVT, Subjective Logic, fusion, ageing, and
   evaluation methods.
3. `docs/scientific_deconstruction/ASSUMPTIONS.md` - assumptions, failure
   modes, mitigations, and remaining threats to validity.
4. `docs/scientific_deconstruction/THEORY_GRAPH.md` - map from theoretical
   assumptions and operators to implementation files.
5. `docs/scientific_deconstruction/REFERENCES.md` - bibliography and
   theory-to-code traceability.

For paper work, also read `docs/AUDIT_CURRENT_STATUS.md` and
`docs/honest_limitations.md` before using any metric in a manuscript. Those
files state which claims are current, which are exploratory, and which protocol
choices are used for the paper.

Current paper-facing numbers are summarised in
`docs/review/PUBLICATION_TABLES.md`. The latest complete RedeRio run is
`current_version/results/2e12261d55a8f975/` and reports catalog/outages-separate
F1 micro = 0.8666, operator-faithful anomaly F1 micro = 0.8257, MCC = 0.8587,
14/14 attacks detected, and realised global FPR = 0.965%.

For the original "ADS with SL vs the same ADS without SL" question, see
`compare_no_sl_fair.py` and `docs/review/PUBLICATION_TABLES.md` §8bis. The
measured all-leaf no-SL gain is positive but modest; the reconstruction-only
no-SL baseline is strong on RedeRio and must be disclosed.

## Documentation

Recommended reading order:

1. `docs/README.md` - documentation index.
2. `docs/AUDIT_CURRENT_STATUS.md` - current audit state and remaining risks.
3. `docs/scientific_deconstruction/PIPELINE_LOGIC.md` - complete execution
   logic, step by step.
4. `docs/scientific_deconstruction/METHODS.md` - method inventory and
   formulas.
5. `docs/scientific_deconstruction/ASSUMPTIONS.md` - assumptions and threats
   to validity.
6. `docs/honest_limitations.md` - paper-facing limitations.
7. `docs/review/PUBLICATION_TABLES.md` - tables and metrics intended for the
   manuscript.
8. `docs/ARTIFACT_APPENDIX.md` and `docs/REPRODUCIBILITY_CHECKLIST.md` -
   artifact-review material.

## Generated Artifacts

The following files/directories are generated and git-ignored:

```text
outputs/
results/
*.pkl
*.parquet
pipeline_run_summary.json
```
