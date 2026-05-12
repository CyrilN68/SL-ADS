# preprocessing_utils.py — à importer dans train_v10.py ET compute_evidence_v2.py
"""
Politique NaN unifiée train/inference (PATCH m-07 / F25).

Un seul point d'entrée (`preprocess_metrics`) pour garantir que les
résidus Prophet calibrés sur les données d'entraînement et les
évidences calculées en inférence utilisent exactement la même
stratégie d'imputation.  Toute divergence ici casserait la
calibration EVT (Siffer 2017) et la bijection SL (Jøsang §3.5.2).

La politique tient en trois règles :

1. **Forward-fill limité** : pour chaque colonne métrique, on propage
   la dernière valeur observée pendant au plus ``limit_ffill`` pas de
   temps.  Au-delà, la valeur reste NaN et doit être traitée comme
   "donnée manquante" par les couches supérieures
   (``compute_evidence_v2`` retourne u=1 pour les fenêtres entièrement
   NaN, cf. PATCH M-06/F09 sur l'acceptation des fenêtres partielles).

2. **Jamais ``fillna(0)`` sur les métriques réseau.**  0 byte ≠
   absence de capture : assimiler les deux introduit un biais
   systématique vers la classe "safe" dans les opinions.

3. **Whitelist explicite de colonnes non-métriques** (PATCH TASK-37,
   audit_codex MAJ-02, 2026-04-27) : seules les colonnes listées dans
   ``NON_METRIC_COLUMNS`` (labels, flags, ds…) sont préservées
   intactes.  Toute autre colonne est traitée comme métrique et
   forward-fillée.  L'appelant peut passer ``metric_cols`` ou
   ``non_metric_cols`` pour outrepasser la heuristique.

Le paramètre ``limit_ffill`` est exposé dans ``config.py`` sous le nom
``NAN_FFILL_LIMIT`` (défaut = 10) et peut être sweepé en ablation via
``run_ablation_v2.py``.
"""

# PATCH TASK-37 (audit_codex MAJ-02, 2026-04-27).  Default non-metric
# columns: timestamps, labels, injection bookkeeping.  Anything not in
# this list AND not the ``ds`` timestamp is treated as a metric and
# forward-filled per the unified NaN policy.
NON_METRIC_COLUMNS = (
    'ds', 'timestamp', 'time',
    'label', 'y_true', 'is_anomaly', 'anomaly',
    'injected', 'is_injected', 'attack_id', 'attack_type', 'family',
    'split', 'fold', 'window_id',
)


def preprocess_metrics(df, limit_ffill=10, metric_cols=None,
                       non_metric_cols=None, strict=False):
    """
    Apply the unified NaN policy to metric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame; ``ds`` (or ``timestamp``) is always preserved.
    limit_ffill : int, optional
        Maximum number of consecutive time steps to forward-fill before
        leaving the remainder as NaN.  Must be ``>= 0``.  ``0`` disables
        forward-fill entirely (strict policy; useful for ablation).
    metric_cols : list[str] | None, optional
        Explicit whitelist of columns to forward-fill.  If supplied, all
        other columns are left untouched and ``non_metric_cols`` is
        ignored.  This is the preferred form for callers that already
        know their schema (PATCH TASK-37).
    non_metric_cols : list[str] | None, optional
        Override of the default ``NON_METRIC_COLUMNS`` whitelist.  Only
        used when ``metric_cols`` is None.
    strict : bool, optional
        If True, raise ``ValueError`` when a column is neither in the
        metric set nor in the non-metric whitelist (helps catch schema
        drift in CI).  Default False (permissive: unknown columns are
        treated as metrics with a one-line stderr warning).

    Returns
    -------
    pd.DataFrame
        Same frame, modified in place — metric columns are
        forward-filled with the configured limit; long gaps remain NaN.

    Notes
    -----
    * No ``fillna(0)`` is ever applied here — 0 byte is a valid metric
      value on an idle link and must not be confused with "no data".
    * The operation is idempotent: re-running with the same ``limit_ffill``
      produces the same output.
    """
    if metric_cols is None:
        nmc = set(non_metric_cols) if non_metric_cols is not None else set(NON_METRIC_COLUMNS)
        metric_cols = []
        unknown = []
        for c in df.columns:
            if c in nmc:
                continue
            # Heuristic: 'flag', 'mask', 'gt_' prefixes are non-metric by name
            cl = c.lower()
            if cl.startswith(('flag_', 'mask_', 'gt_')) or cl.endswith(('_flag', '_mask', '_gt')):
                continue
            if not _is_numeric_dtype(df[c]):
                unknown.append(c)
                continue
            metric_cols.append(c)
        if unknown:
            if strict:
                raise ValueError(
                    f"preprocess_metrics(strict=True): non-numeric columns "
                    f"that are not in non_metric_cols whitelist: {unknown}"
                )
            else:
                import warnings as _w
                _w.warn(
                    f"preprocess_metrics: skipping non-numeric columns "
                    f"not in NON_METRIC_COLUMNS whitelist: {unknown}",
                    category=UserWarning, stacklevel=2,
                )

    if limit_ffill and limit_ffill > 0 and metric_cols:
        df[metric_cols] = df[metric_cols].ffill(limit=limit_ffill)
    # limit_ffill == 0  →  strict policy: leave all NaNs untouched.
    return df


def _is_numeric_dtype(series):
    """Lightweight helper avoiding the pandas import-level dependency."""
    try:
        import pandas as _pd
        return _pd.api.types.is_numeric_dtype(series)
    except Exception:
        return False