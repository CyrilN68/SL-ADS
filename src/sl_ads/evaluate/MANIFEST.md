# IDS-SL Run Manifest
<!-- manifest:header:v1 -->

This file is appended automatically by `utils_manifest.py` at the end of every evaluation run (PATCH m-06 / F23).
Each section below is a single experimental run: timestamp, version descriptor, source CSV, key metrics, environment.

**Do not edit entries manually** — they encode the reproducibility trail of the paper. New runs append at the bottom.

---

## 2026-04-29 08:25:06 UTC — `trained_models_v9_v9_v4s_v3_v5`

- **Run ID:** `94d888565b15e751`
- **Source CSV:** `../results/resultats_trained_models_v9_v9_v4s_v3_v5\detection_results_INJECTED.csv`
- **Git SHA:** `772fa09` *(dirty working tree)*

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

## 2026-04-30 00:02:46 UTC — `trained_models_v9_v9_v4s_v4`

- **Run ID:** `a4f07c7b2555000b`
- **Source CSV:** `../results/resultats_trained_models_v9_v9_v4s_v4\detection_results_INJECTED.csv`
- **Git SHA:** `772fa09` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.129` |
| F1 — binary | `0.855` |
| F1 — micro (pure) | `0.784` |
| F1 — macro (pure) | `0.885` |
| F1 — coverage-weighted | `0.811` |
| F1 — TTD-penalized | `0.788` |
| Precision (window) | `0.746` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.888` |
| FPR (%) | `1.640` |
| MCC | `0.772` |
| TPR (window-level) | `0.827` |
| FPR (window-level) | `0.016` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.975` |
| f1_binary_hybrid_episode_recall | `0.855` |
| f1_coverage_hybrid_episode_recall | `0.811` |
| f1_ttd_hybrid_episode_recall | `0.788` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_trained_models_v9_v9_v4s_v4\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-02 03:31:22 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `60562616181b1b57`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `e35fce9` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.129` |
| F1 — binary | `0.855` |
| F1 — micro (pure) | `0.784` |
| F1 — macro (pure) | `0.885` |
| F1 — coverage-weighted | `0.811` |
| F1 — TTD-penalized | `0.788` |
| Precision (window) | `0.746` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.888` |
| FPR (%) | `1.640` |
| MCC | `0.772` |
| TPR (window-level) | `0.827` |
| FPR (window-level) | `0.016` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.975` |
| f1_binary_hybrid_episode_recall | `0.855` |
| f1_coverage_hybrid_episode_recall | `0.811` |
| f1_ttd_hybrid_episode_recall | `0.788` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-04 14:34:34 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `60562616181b1b57`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `e35fce9` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.129` |
| F1 — binary | `0.855` |
| F1 — micro (pure) | `0.784` |
| F1 — macro (pure) | `0.885` |
| F1 — coverage-weighted | `0.811` |
| F1 — TTD-penalized | `0.788` |
| Precision (window) | `0.746` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.888` |
| FPR (%) | `1.640` |
| MCC | `0.772` |
| TPR (window-level) | `0.827` |
| FPR (window-level) | `0.016` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.975` |
| f1_binary_hybrid_episode_recall | `0.855` |
| f1_coverage_hybrid_episode_recall | `0.811` |
| f1_ttd_hybrid_episode_recall | `0.788` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-04 14:50:56 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `60562616181b1b57`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `e35fce9` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.129` |
| F1 — binary | `0.855` |
| F1 — micro (pure) | `0.784` |
| F1 — macro (pure) | `0.885` |
| F1 — coverage-weighted | `0.811` |
| F1 — TTD-penalized | `0.788` |
| Precision (window) | `0.746` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.888` |
| FPR (%) | `1.640` |
| MCC | `0.772` |
| TPR (window-level) | `0.827` |
| FPR (window-level) | `0.016` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.975` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.855` |
| f1_coverage_hybrid_episode_recall | `0.811` |
| f1_ttd_hybrid_episode_recall | `0.788` |
| range_auc_pr_at_max | `0.491` |
| range_auc_roc_at_max | `0.760` |
| vus_pr | `0.604` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-05 16:55:39 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `60562616181b1b57`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `e35fce9` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.129` |
| F1 — binary | `0.855` |
| F1 — micro (pure) | `0.784` |
| F1 — macro (pure) | `0.885` |
| F1 — coverage-weighted | `0.811` |
| F1 — TTD-penalized | `0.788` |
| Precision (window) | `0.746` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.888` |
| FPR (%) | `1.640` |
| MCC | `0.772` |
| TPR (window-level) | `0.827` |
| FPR (window-level) | `0.016` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.975` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.855` |
| f1_coverage_hybrid_episode_recall | `0.811` |
| f1_ttd_hybrid_episode_recall | `0.788` |
| fpr_ratio_to_target | `16.430` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.491` |
| range_auc_roc_at_max | `0.760` |
| vus_pr | `0.604` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-05 21:53:07 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `60562616181b1b57`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `e35fce9` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.129` |
| F1 — binary | `0.855` |
| F1 — micro (pure) | `0.784` |
| F1 — macro (pure) | `0.885` |
| F1 — coverage-weighted | `0.811` |
| F1 — TTD-penalized | `0.788` |
| Precision (window) | `0.746` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.888` |
| FPR (%) | `1.640` |
| MCC | `0.772` |
| TPR (window-level) | `0.827` |
| FPR (window-level) | `0.016` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.975` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.855` |
| f1_coverage_hybrid_episode_recall | `0.811` |
| f1_ttd_hybrid_episode_recall | `0.788` |
| fpr_ratio_to_target | `16.430` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.491` |
| range_auc_roc_at_max | `0.760` |
| vus_pr | `0.604` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-05 21:56:56 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `60562616181b1b57`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `e35fce9` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.129` |
| F1 — binary | `0.976` |
| F1 — micro (pure) | `0.886` |
| F1 — macro (pure) | `0.940` |
| F1 — coverage-weighted | `0.920` |
| F1 — TTD-penalized | `0.890` |
| Precision (window) | `0.954` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.888` |
| FPR (%) | `0.240` |
| MCC | `0.882` |
| TPR (window-level) | `0.827` |
| FPR (window-level) | `0.002` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.988` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.976` |
| f1_coverage_hybrid_episode_recall | `0.920` |
| f1_ttd_hybrid_episode_recall | `0.890` |
| fpr_ratio_to_target | `2.330` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.491` |
| range_auc_roc_at_max | `0.760` |
| vus_pr | `0.604` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-06 15:33:53 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `72ee236330dc7fa5`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.129` |
| F1 — binary | `0.976` |
| F1 — micro (pure) | `0.886` |
| F1 — macro (pure) | `0.940` |
| F1 — coverage-weighted | `0.920` |
| F1 — TTD-penalized | `0.890` |
| Precision (window) | `0.954` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.888` |
| FPR (%) | `0.240` |
| MCC | `0.882` |
| TPR (window-level) | `0.827` |
| FPR (window-level) | `0.002` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.988` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.976` |
| f1_coverage_hybrid_episode_recall | `0.920` |
| f1_ttd_hybrid_episode_recall | `0.890` |
| fpr_ratio_to_target | `2.330` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.491` |
| range_auc_roc_at_max | `0.760` |
| vus_pr | `0.604` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-06 15:36:02 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `a794e03fdc30ce0e`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.129` |
| F1 — binary | `0.976` |
| F1 — micro (pure) | `0.886` |
| F1 — macro (pure) | `0.940` |
| F1 — coverage-weighted | `0.920` |
| F1 — TTD-penalized | `0.890` |
| Precision (window) | `0.954` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.888` |
| FPR (%) | `0.240` |
| MCC | `0.882` |
| TPR (window-level) | `0.827` |
| FPR (window-level) | `0.002` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.988` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.976` |
| f1_coverage_hybrid_episode_recall | `0.920` |
| f1_ttd_hybrid_episode_recall | `0.890` |
| fpr_ratio_to_target | `2.330` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.491` |
| range_auc_roc_at_max | `0.760` |
| vus_pr | `0.604` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-06 15:38:22 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `55f6257f4f11f983`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.129` |
| F1 — binary | `0.976` |
| F1 — micro (pure) | `0.886` |
| F1 — macro (pure) | `0.940` |
| F1 — coverage-weighted | `0.920` |
| F1 — TTD-penalized | `0.890` |
| Precision (window) | `0.954` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.888` |
| FPR (%) | `0.240` |
| MCC | `0.882` |
| TPR (window-level) | `0.827` |
| FPR (window-level) | `0.002` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.988` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.976` |
| f1_coverage_hybrid_episode_recall | `0.920` |
| f1_ttd_hybrid_episode_recall | `0.890` |
| fpr_ratio_to_target | `2.330` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.491` |
| range_auc_roc_at_max | `0.760` |
| vus_pr | `0.604` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-06 15:40:46 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `772e14c74bf23444`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.129` |
| F1 — binary | `0.973` |
| F1 — micro (pure) | `0.888` |
| F1 — macro (pure) | `0.941` |
| F1 — coverage-weighted | `0.919` |
| F1 — TTD-penalized | `0.890` |
| Precision (window) | `0.947` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.893` |
| FPR (%) | `0.280` |
| MCC | `0.884` |
| TPR (window-level) | `0.835` |
| FPR (window-level) | `0.003` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.988` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.973` |
| f1_coverage_hybrid_episode_recall | `0.919` |
| f1_ttd_hybrid_episode_recall | `0.890` |
| fpr_ratio_to_target | `2.750` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.492` |
| range_auc_roc_at_max | `0.760` |
| vus_pr | `0.606` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-06 15:43:03 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `3fe049cca97c9b5a`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.129` |
| F1 — binary | `1.000` |
| F1 — micro (pure) | `0.744` |
| F1 — macro (pure) | `0.866` |
| F1 — coverage-weighted | `0.872` |
| F1 — TTD-penalized | `0.812` |
| Precision (window) | `1.000` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.772` |
| FPR (%) | `0.000` |
| MCC | `0.760` |
| TPR (window-level) | `0.592` |
| FPR (window-level) | `0.000` |
| Median TTD (min) | `12.500` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.977` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `1.000` |
| f1_coverage_hybrid_episode_recall | `0.872` |
| f1_ttd_hybrid_episode_recall | `0.812` |
| fpr_ratio_to_target | `0.000` |
| fpr_target | `0.001` |
| fpr_target_status | `OK` |
| range_auc_pr_at_max | `0.536` |
| range_auc_roc_at_max | `0.764` |
| vus_pr | `0.672` |
| vus_roc | `0.863` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-06 15:44:51 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `796adf7e74c98d20`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.129` |
| F1 — binary | `0.999` |
| F1 — micro (pure) | `0.840` |
| F1 — macro (pure) | `0.916` |
| F1 — coverage-weighted | `0.922` |
| F1 — TTD-penalized | `0.888` |
| Precision (window) | `0.998` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.856` |
| FPR (%) | `0.010` |
| MCC | `0.844` |
| TPR (window-level) | `0.725` |
| FPR (window-level) | `0.000` |
| Median TTD (min) | `12.500` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.984` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.999` |
| f1_coverage_hybrid_episode_recall | `0.922` |
| f1_ttd_hybrid_episode_recall | `0.888` |
| fpr_ratio_to_target | `0.080` |
| fpr_target | `0.001` |
| fpr_target_status | `OK` |
| range_auc_pr_at_max | `0.568` |
| range_auc_roc_at_max | `0.779` |
| vus_pr | `0.716` |
| vus_roc | `0.873` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-06 15:46:04 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `ad3b575b51f7aefa`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.129` |
| F1 — binary | `0.999` |
| F1 — micro (pure) | `0.776` |
| F1 — macro (pure) | `0.883` |
| F1 — coverage-weighted | `0.895` |
| F1 — TTD-penalized | `0.854` |
| Precision (window) | `0.998` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.812` |
| FPR (%) | `0.010` |
| MCC | `0.787` |
| TPR (window-level) | `0.635` |
| FPR (window-level) | `0.000` |
| Median TTD (min) | `12.500` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.979` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.999` |
| f1_coverage_hybrid_episode_recall | `0.895` |
| f1_ttd_hybrid_episode_recall | `0.854` |
| fpr_ratio_to_target | `0.080` |
| fpr_target | `0.001` |
| fpr_target_status | `OK` |
| range_auc_pr_at_max | `0.581` |
| range_auc_roc_at_max | `0.787` |
| vus_pr | `0.730` |
| vus_roc | `0.879` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-06 15:47:17 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `00043ddd46cb49b2`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.129` |
| F1 — binary | `0.828` |
| F1 — micro (pure) | `0.789` |
| F1 — macro (pure) | `0.887` |
| F1 — coverage-weighted | `0.797` |
| F1 — TTD-penalized | `0.778` |
| Precision (window) | `0.707` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.915` |
| FPR (%) | `2.220` |
| MCC | `0.781` |
| TPR (window-level) | `0.892` |
| FPR (window-level) | `0.022` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.973` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.828` |
| f1_coverage_hybrid_episode_recall | `0.797` |
| f1_ttd_hybrid_episode_recall | `0.778` |
| fpr_ratio_to_target | `22.140` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.472` |
| range_auc_roc_at_max | `0.750` |
| vus_pr | `0.576` |
| vus_roc | `0.848` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-07 08:35:15 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `2d26c50ac464ded6`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.103` |
| F1 — binary | `0.917` |
| F1 — micro (pure) | `0.867` |
| F1 — macro (pure) | `0.929` |
| F1 — coverage-weighted | `0.879` |
| F1 — TTD-penalized | `0.856` |
| Precision (window) | `0.847` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.915` |
| FPR (%) | `0.970` |
| MCC | `0.859` |
| TPR (window-level) | `0.888` |
| FPR (window-level) | `0.010` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.985` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.917` |
| f1_coverage_hybrid_episode_recall | `0.879` |
| f1_ttd_hybrid_episode_recall | `0.856` |
| fpr_ratio_to_target | `9.570` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.491` |
| range_auc_roc_at_max | `0.760` |
| vus_pr | `0.604` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |

---

## 2026-05-07 08:36:06 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `5ab36388717053b1`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.103` |
| F1 — binary | `0.917` |
| F1 — micro (pure) | `0.867` |
| F1 — macro (pure) | `0.929` |
| F1 — coverage-weighted | `0.879` |
| F1 — TTD-penalized | `0.856` |
| Precision (window) | `0.847` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.915` |
| FPR (%) | `0.970` |
| MCC | `0.859` |
| TPR (window-level) | `0.888` |
| FPR (window-level) | `0.010` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.985` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.917` |
| f1_coverage_hybrid_episode_recall | `0.879` |
| f1_ttd_hybrid_episode_recall | `0.856` |
| fpr_ratio_to_target | `9.570` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.491` |
| range_auc_roc_at_max | `0.760` |
| vus_pr | `0.604` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |

---

## 2026-05-07 08:38:12 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `2d26c50ac464ded6`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.103` |
| F1 — binary | `0.917` |
| F1 — micro (pure) | `0.867` |
| F1 — macro (pure) | `0.929` |
| F1 — coverage-weighted | `0.879` |
| F1 — TTD-penalized | `0.856` |
| Precision (window) | `0.847` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.915` |
| FPR (%) | `0.970` |
| MCC | `0.859` |
| TPR (window-level) | `0.888` |
| FPR (window-level) | `0.010` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.985` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.917` |
| f1_coverage_hybrid_episode_recall | `0.879` |
| f1_ttd_hybrid_episode_recall | `0.856` |
| fpr_ratio_to_target | `9.570` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.491` |
| range_auc_roc_at_max | `0.760` |
| vus_pr | `0.604` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |

---

## 2026-05-07 08:39:26 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `5ab36388717053b1`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.103` |
| F1 — binary | `0.917` |
| F1 — micro (pure) | `0.867` |
| F1 — macro (pure) | `0.929` |
| F1 — coverage-weighted | `0.879` |
| F1 — TTD-penalized | `0.856` |
| Precision (window) | `0.847` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.915` |
| FPR (%) | `0.970` |
| MCC | `0.859` |
| TPR (window-level) | `0.888` |
| FPR (window-level) | `0.010` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.985` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.917` |
| f1_coverage_hybrid_episode_recall | `0.879` |
| f1_ttd_hybrid_episode_recall | `0.856` |
| fpr_ratio_to_target | `9.570` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.491` |
| range_auc_roc_at_max | `0.760` |
| vus_pr | `0.604` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |

---

## 2026-05-07 08:44:52 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `4292bbdc25c14815`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.060` |
| F1 — binary | `0.723` |
| F1 — micro (pure) | `0.706` |
| F1 — macro (pure) | `0.841` |
| F1 — coverage-weighted | `0.709` |
| F1 — TTD-penalized | `0.699` |
| Precision (window) | `0.566` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.949` |
| FPR (%) | `4.310` |
| MCC | `0.709` |
| TPR (window-level) | `0.936` |
| FPR (window-level) | `0.043` |
| Median TTD (min) | `5.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.956` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.723` |
| f1_coverage_hybrid_episode_recall | `0.709` |
| f1_ttd_hybrid_episode_recall | `0.699` |
| fpr_ratio_to_target | `43.040` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.491` |
| range_auc_roc_at_max | `0.760` |
| vus_pr | `0.604` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |

---

## 2026-05-07 08:46:11 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `cf8dd1f512f45fae`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.060` |
| F1 — binary | `0.722` |
| F1 — micro (pure) | `0.705` |
| F1 — macro (pure) | `0.840` |
| F1 — coverage-weighted | `0.708` |
| F1 — TTD-penalized | `0.698` |
| Precision (window) | `0.564` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.949` |
| FPR (%) | `4.340` |
| MCC | `0.708` |
| TPR (window-level) | `0.936` |
| FPR (window-level) | `0.043` |
| Median TTD (min) | `5.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.956` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.722` |
| f1_coverage_hybrid_episode_recall | `0.708` |
| f1_ttd_hybrid_episode_recall | `0.698` |
| fpr_ratio_to_target | `43.290` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.491` |
| range_auc_roc_at_max | `0.760` |
| vus_pr | `0.604` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |

---

## 2026-05-07 09:03:26 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `4a1f1b6dfb41e9c9`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.060` |
| F1 — binary | `0.723` |
| F1 — micro (pure) | `0.706` |
| F1 — macro (pure) | `0.841` |
| F1 — coverage-weighted | `0.709` |
| F1 — TTD-penalized | `0.699` |
| Precision (window) | `0.566` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.949` |
| FPR (%) | `4.310` |
| MCC | `0.709` |
| TPR (window-level) | `0.936` |
| FPR (window-level) | `0.043` |
| Median TTD (min) | `5.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.956` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.723` |
| f1_coverage_hybrid_episode_recall | `0.709` |
| f1_ttd_hybrid_episode_recall | `0.699` |
| fpr_ratio_to_target | `43.040` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.491` |
| range_auc_roc_at_max | `0.760` |
| vus_pr | `0.604` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-07 09:04:53 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `5757ffe035d3113d`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.060` |
| F1 — binary | `0.722` |
| F1 — micro (pure) | `0.705` |
| F1 — macro (pure) | `0.840` |
| F1 — coverage-weighted | `0.708` |
| F1 — TTD-penalized | `0.698` |
| Precision (window) | `0.564` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.949` |
| FPR (%) | `4.340` |
| MCC | `0.708` |
| TPR (window-level) | `0.936` |
| FPR (window-level) | `0.043` |
| Median TTD (min) | `5.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.956` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.722` |
| f1_coverage_hybrid_episode_recall | `0.708` |
| f1_ttd_hybrid_episode_recall | `0.698` |
| fpr_ratio_to_target | `43.290` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.491` |
| range_auc_roc_at_max | `0.760` |
| vus_pr | `0.604` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-10 18:22:58 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `6bbb739c2d7c63f9`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.125` |
| F1 — binary | `0.974` |
| F1 — micro (pure) | `0.915` |
| F1 — macro (pure) | `0.955` |
| F1 — coverage-weighted | `0.924` |
| F1 — TTD-penalized | `0.893` |
| Precision (window) | `0.949` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.899` |
| FPR (%) | `0.280` |
| MCC | `0.911` |
| TPR (window-level) | `0.883` |
| FPR (window-level) | `0.003` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.991` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.974` |
| f1_coverage_hybrid_episode_recall | `0.924` |
| f1_ttd_hybrid_episode_recall | `0.893` |
| fpr_ratio_to_target | `2.830` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.602` |
| range_auc_roc_at_max | `0.802` |
| vus_pr | `0.741` |
| vus_roc | `0.885` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `3` |
| fallbacks_counts | `{"reconstruction_dummy": 2, "evt_sigma_mod": 1}` |
| fallbacks_metrics | `{"reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"], "evt_sigma_mod": ["tcp_packets"]}` |

---

## 2026-05-10 18:35:12 UTC — `RedeRio_trained_v4s_v4_v2`

- **Run ID:** `6bbb739c2d7c63f9`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v2\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.125` |
| F1 — binary | `0.974` |
| F1 — micro (pure) | `0.915` |
| F1 — macro (pure) | `0.955` |
| F1 — coverage-weighted | `0.924` |
| F1 — TTD-penalized | `0.893` |
| Precision (window) | `0.949` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.899` |
| FPR (%) | `0.280` |
| MCC | `0.911` |
| TPR (window-level) | `0.883` |
| FPR (window-level) | `0.003` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.991` |
| existence_recall | `1.000` |
| f1_binary_hybrid_episode_recall | `0.974` |
| f1_coverage_hybrid_episode_recall | `0.924` |
| f1_ttd_hybrid_episode_recall | `0.893` |
| fpr_ratio_to_target | `2.830` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| range_auc_pr_at_max | `0.602` |
| range_auc_roc_at_max | `0.802` |
| vus_pr | `0.741` |
| vus_roc | `0.885` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v2\evaluation` |
| fallbacks_total | `3` |
| fallbacks_counts | `{"reconstruction_dummy": 2, "evt_sigma_mod": 1}` |
| fallbacks_metrics | `{"reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"], "evt_sigma_mod": ["tcp_packets"]}` |

---

## 2026-05-11 22:34:41 UTC — `RedeRio_trained_v4s_v4_v3`

- **Run ID:** `9a31486813a7a387`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v3\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.103` |
| F1 — binary | `0.917` |
| F1 — micro (pure) | `0.867` |
| F1 — macro (pure) | `0.929` |
| F1 — coverage-weighted | `0.879` |
| F1 — TTD-penalized | `0.856` |
| Precision (window) | `0.847` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.915` |
| FPR (%) | `0.970` |
| MCC | `0.859` |
| TPR (window-level) | `0.888` |
| FPR (window-level) | `0.010` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.985` |
| existence_recall | `1.000` |
| f1_anomaly_macro_pure | `0.906` |
| f1_anomaly_micro_pure | `0.826` |
| f1_binary_hybrid_episode_recall | `0.917` |
| f1_catalog_macro_pure | `0.929` |
| f1_catalog_micro_pure | `0.867` |
| f1_coverage_hybrid_episode_recall | `0.879` |
| f1_protocol_policy | `report_both` |
| f1_ttd_hybrid_episode_recall | `0.856` |
| fpr_anomaly_pct | `0.965` |
| fpr_anomaly_window | `0.010` |
| fpr_catalog_pct | `0.965` |
| fpr_catalog_window | `0.010` |
| fpr_ratio_to_target | `9.650` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| n_positive_anomaly_windows | `1063` |
| n_positive_catalog_windows | `721` |
| range_auc_pr_at_max | `0.491` |
| range_auc_roc_at_max | `0.760` |
| tpr_anomaly_window | `0.780` |
| tpr_catalog_window | `0.888` |
| vus_pr | `0.604` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v3\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

## 2026-05-11 23:10:53 UTC — `RedeRio_trained_v4s_v4_v3`

- **Run ID:** `9a31486813a7a387`
- **Source CSV:** `../results/resultats_RedeRio_trained_v4s_v4_v3\detection_results_INJECTED.csv`
- **Git SHA:** `bf8c270` *(dirty working tree)*

### Key metrics

| Metric | Value |
|---|---|
| Operational threshold | `0.103` |
| F1 — binary | `0.917` |
| F1 — micro (pure) | `0.867` |
| F1 — macro (pure) | `0.929` |
| F1 — coverage-weighted | `0.879` |
| F1 — TTD-penalized | `0.856` |
| Precision (window) | `0.847` |
| Recall — binary | `1.000` |
| Recall — coverage | `0.915` |
| FPR (%) | `0.970` |
| MCC | `0.859` |
| TPR (window-level) | `0.888` |
| FPR (window-level) | `0.010` |
| Median TTD (min) | `10.000` |
| Detected attacks | `14` |
| Total attacks | `14` |
| accuracy | `0.985` |
| existence_recall | `1.000` |
| f1_anomaly_macro_pure | `0.906` |
| f1_anomaly_micro_pure | `0.826` |
| f1_binary_hybrid_episode_recall | `0.917` |
| f1_catalog_macro_pure | `0.929` |
| f1_catalog_micro_pure | `0.867` |
| f1_coverage_hybrid_episode_recall | `0.879` |
| f1_protocol_policy | `report_both` |
| f1_ttd_hybrid_episode_recall | `0.856` |
| fpr_anomaly_pct | `0.965` |
| fpr_anomaly_window | `0.010` |
| fpr_catalog_pct | `0.965` |
| fpr_catalog_window | `0.010` |
| fpr_ratio_to_target | `9.650` |
| fpr_target | `0.001` |
| fpr_target_status | `EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY` |
| n_positive_anomaly_windows | `1063` |
| n_positive_catalog_windows | `721` |
| range_auc_pr_at_max | `0.491` |
| range_auc_roc_at_max | `0.760` |
| tpr_anomaly_window | `0.780` |
| tpr_catalog_window | `0.888` |
| vus_pr | `0.604` |
| vus_roc | `0.856` |

### Environment

| Component | Version |
|---|---|
| python | `3.13.3` |
| platform | `Windows-11-10.0.26200-SP0` |
| numpy | `2.4.4` |
| pandas | `3.0.2` |
| scipy | `1.17.1` |
| scikit-learn | `1.8.0` |
| statsmodels | `0.14.6` |
| joblib | `1.5.3` |
| prophet | `1.3.0` |
| matplotlib | `3.10.9` |

### Run-specific details

| Key | Value |
|---|---|
| catalog_mode | `injected` |
| lambda_decay | `0.85` |
| um_enabled | `False` |
| window_min | `5` |
| context_h | `2.0` |
| col_det | `FINAL_SYSTEM_CBF_proj_atk` |
| output_dir | `../results/resultats_RedeRio_trained_v4s_v4_v3\evaluation` |
| fallbacks_total | `8` |
| fallbacks_counts | `{"evt_sigma_mod": 6, "reconstruction_dummy": 2}` |
| fallbacks_metrics | `{"evt_sigma_mod": ["flows:pos", "syn", "tcp", "entropy_src_ip", "entropy_dst_port", "tcp_packets"], "reconstruction_dummy": ["reconst_fin_from_syn", "reconst_tcp_from_packets"]}` |

---

