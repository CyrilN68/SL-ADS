"""sl_ads.ablation — Controlled-variable ablation studies.

Modules:
    run_ablation             — Master ablation runner.  Migrated from
                               ``run_ablation_v2.py``.
    run_ablation_labeled     — Variant for datasets with real labels
                               (METR-LA, GECCO, CESNET).
    ablation_fusion_mode     — CBF vs WBF vs hierarchical.
    ablation_injection_level — Evidence-level vs raw-metric injection.
    ablation_nan_ffill       — NaN forward-fill limit sensitivity.
    ablation_sbn_novelty     — SBN novelty u_raw threshold sweep.
    ablation_temporal_sbn    — Temporal SBN window-size sensitivity.
"""
