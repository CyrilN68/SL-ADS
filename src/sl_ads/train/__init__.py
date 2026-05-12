"""sl_ads.train — Model training and evidence pre-computation.

Modules:
    train_models      — Prophet + reconstruction baselines + EVT/FPR
                        threshold calibration.  Migrated from legacy
                        ``train_v10.py``.
    compute_evidence  — Evidence vectors precomputed from raw metrics
                        and trained models.  Migrated from legacy
                        ``compute_evidence_v2.py``.
"""
