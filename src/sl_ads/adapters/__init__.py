"""sl_ads.adapters — Per-dataset adapters.

Each adapter normalises a raw dataset into a standardised CSV with
columns ``timestamp``, ``label`` (when available) and a fixed set of
network/IoT metrics.  Migrated from the legacy ``dataset_adapter/``
folder; the public class names are unchanged.

Modules:
    adapter_base           — Abstract base class.
    cesnet_adapter         — CESNET-TimeSeries24 (synthetic timestamps,
                              cf. audit_codex MAJ-05).
    gecco_adapter          — GECCO-IoT (water sensors).
    metr_la_adapter        — METR-LA traffic speed sensors.
    rederio_adapter        — UFRJ Rio de Janeiro packet aggregates.
    labeller_unsupervised  — STL+Hampel+CUSUM consensus pseudo-labeller.
    metric_selector        — Pearson/MI metric pre-selection.
    run_cross_dataset      — Multi-dataset adapter runner.
"""
