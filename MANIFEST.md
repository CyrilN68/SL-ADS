# IDS-SL Run Manifest
<!-- manifest:header:v1 -->

**Historical run log. For current paper-facing claims, use docs/AUDIT_CURRENT_STATUS.md.**

This file is appended automatically by `utils_manifest.py` at the end of every evaluation run (PATCH m-06 / F23).
Each section below is a single experimental run: timestamp, version descriptor, source CSV, key metrics, environment.

**Do not edit entries manually** — they encode the reproducibility trail of the paper. New runs append at the bottom.

---

## 2026-04-25 13:58:57 UTC — `trained_models_v9_v9_v4s_v3_v5`

- **Source CSV:** `../results/resultats_trained_models_v9_v9_v4s_v3_v5\detection_results_INJECTED.csv`
- **Git SHA:** `4109a26` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.154` |
| F1 — binary | `0.857` |
| F1 — micro (pure) | `0.781` |
| F1 — macro (pure) | `0.884` |
| F1 — coverage-weighted | `0.807` |
| F1 — TTD-penalized | `0.781` |
| Precision (window) | `0.750` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.873` |
| FPR (%) | `1.590` |
| MCC | `0.769` |
| TPR (window-level) | `0.816` |
| FPR (window-level) | `0.016` |
| Median TTD (min) | `12.500` |
| Detected attacks | `14` |
| Total attacks | `14` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `1.26.4` |
| pandas | `2.2.3` |
| scipy | `1.17.0` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.2.2` |
| matplotlib | `3.10.8` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_trained_models_v9_v9_v4s_v3_v5\evaluation` |

---

## 2026-04-26 09:38:40 UTC — `trained_models_v9_v9_v4s_v3_v5`

- **Run ID:** `948f3ac76e5d1af9`
- **Source CSV:** `../results/resultats_trained_models_v9_v9_v4s_v3_v5\detection_results_INJECTED.csv`
- **Git SHA:** `4109a26` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.154` |
| F1 — binary | `0.857` |
| F1 — micro (pure) | `0.781` |
| F1 — macro (pure) | `0.884` |
| F1 — coverage-weighted | `0.807` |
| F1 — TTD-penalized | `0.781` |
| Precision (window) | `0.750` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.873` |
| FPR (%) | `1.590` |
| MCC | `0.769` |
| TPR (window-level) | `0.816` |
| FPR (window-level) | `0.016` |
| Median TTD (min) | `12.500` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.975` |
| f1_binary_hybrid_episode_recall | `0.857` |
| f1_coverage_hybrid_episode_recall | `0.807` |
| f1_ttd_hybrid_episode_recall | `0.781` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `1.26.4` |
| pandas | `2.2.3` |
| scipy | `1.17.0` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.2.2` |
| matplotlib | `3.10.8` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_trained_models_v9_v9_v4s_v3_v5\evaluation` |

---

