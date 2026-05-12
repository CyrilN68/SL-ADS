import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from scipy.stats import median_abs_deviation

# Phase H — absolute import via sl_ads package.
from sl_ads import config


class ConsensusLabeller:
    def __init__(self, period):
        self.period = period
        self.vote_threshold = config.ADAPTER_VOTE_THRESHOLD  # Importé du config.py !

    def _algo_stl_resid(self, series, threshold=3.0):
        """
        Décomposition STL. Vote 1 si le résidu dépasse 3 écarts-types.

        PATCH TASK-40 (audit_codex MAJ-04, 2026-04-27).  Previously, STL
        failures returned a silent all-zero vote which caused the
        labeller to output ``NORMAL`` for the entire series — biasing
        the pseudo-labels and producing artificially high specificity in
        downstream ROC curves.  We now:

          * record the failure (exception type + message) on the
            instance;
          * if ``STL_FAIL_POLICY='raise'`` (CONFIG-driven, default for
            production), re-raise; if ``STL_FAIL_POLICY='abstain'``,
            return ``np.full(len, np.nan)`` so the consensus arithmetic
            propagates ``nan`` instead of fake zero votes.

        ``np.nan`` votes are treated as "no signal" by the consensus
        threshold and never count as NORMAL.
        """
        try:
            stl = STL(series, period=self.period, robust=True)
            res = stl.fit()
            resid = res.resid
            std_resid = np.nanstd(resid)
            # Anomalie si le résidu est très grand
            return (np.abs(resid) > threshold * std_resid).astype(int)
        except Exception as e:
            self._stl_last_error = (type(e).__name__, str(e))
            policy = getattr(config, 'STL_FAIL_POLICY', 'raise')
            print(f"[labeller][MAJ-04] STL failure ({type(e).__name__}: {e}) "
                  f"— policy={policy}")
            if policy == 'raise':
                raise RuntimeError(
                    f"STL decomposition failed (period={self.period}, "
                    f"len={len(series)}): {type(e).__name__}: {e}. "
                    "Set CONFIG.STL_FAIL_POLICY='abstain' to fall back to "
                    "an abstaining vote (np.nan), but do NOT use the "
                    "silent-zero behaviour for publication runs."
                ) from e
            if policy == 'abstain':
                return np.full(len(series), np.nan)
            # Unknown policy → safe default: raise.
            raise RuntimeError(
                f"Unknown STL_FAIL_POLICY={policy!r}. Use 'raise' or 'abstain'."
            ) from e

    def _algo_hampel(self, series, window_size=7, n_sigmas=3.0):
        """
        Filtre de Hampel utilisant la MAD (Median Absolute Deviation).
        Robuste aux valeurs extrêmes isolées.
        """
        rolling_median = series.rolling(window=window_size, center=True).median()
        # Calcul de la MAD sur la fenêtre
        rolling_mad = series.rolling(window=window_size, center=True).apply(
            lambda x: median_abs_deviation(x, scale='normal', nan_policy='omit')
        )

        lower_bound = rolling_median - (n_sigmas * rolling_mad)
        upper_bound = rolling_median + (n_sigmas * rolling_mad)

        is_anomaly = (series < lower_bound) | (series > upper_bound)
        return is_anomaly.fillna(False).astype(int)

    def _algo_cusum(self, series, drift=0.5, threshold=5.0):
        """
        Algorithme CUSUM (Cumulative Sum) pour détecter les anomalies continues/lentes.
        """
        pos_cusum = np.zeros(len(series))
        neg_cusum = np.zeros(len(series))
        mean_val = np.nanmean(series)
        std_val = np.nanstd(series)

        # Normalisation
        z_series = (series - mean_val) / (std_val + 1e-8)

        for i in range(1, len(z_series)):
            pos_cusum[i] = max(0, pos_cusum[i - 1] + z_series.iloc[i] - drift)
            neg_cusum[i] = max(0, neg_cusum[i - 1] - z_series.iloc[i] - drift)

        is_anomaly = (pos_cusum > threshold) | (neg_cusum > threshold)
        return is_anomaly.astype(int)

    def generate_labels(self, series):
        """
        Agrège les votes des 3 algorithmes.

        PATCH TASK-40 (audit_codex MAJ-04, 2026-04-27).  STL abstentions
        (NaN votes from ``_algo_stl_resid`` when ``STL_FAIL_POLICY=
        'abstain'``) are now propagated explicitly: the per-point sum is
        computed by ``np.nansum`` so an abstaining algorithm contributes
        zero votes (instead of poisoning the sum with NaN), and the
        abstention rate is logged so that downstream callers can decide
        whether to trust the resulting pseudo-labels.
        """
        print(f"Génération de pseudo-labels sur une série de {len(series)} points...")

        # Remplacer les NaN par interpolation pour les calculs
        series_clean = series.interpolate().bfill().ffill()

        vote_stl = np.asarray(self._algo_stl_resid(series_clean), dtype=float)
        vote_hampel = np.asarray(self._algo_hampel(series_clean), dtype=float)
        vote_cusum = np.asarray(self._algo_cusum(series_clean), dtype=float)

        # nansum: an abstaining algo (NaN) contributes 0 votes for that point.
        total_votes = np.nansum(np.stack([vote_stl, vote_hampel, vote_cusum]), axis=0)
        n_stl_abstain = int(np.isnan(vote_stl).sum())
        if n_stl_abstain:
            print(f"[labeller][MAJ-04] STL abstained on {n_stl_abstain}/{len(series)} "
                  f"points ({n_stl_abstain / len(series) * 100:.2f}%) — "
                  "consensus uses Hampel+CUSUM only at those points.")

        # Décision par consensus (>= ADAPTER_VOTE_THRESHOLD)
        final_labels = (total_votes >= self.vote_threshold).astype(int)

        nb_anomalies = int(np.sum(final_labels))
        print(
            f"Terminé. Consensus trouvé : {nb_anomalies} anomalies détectées ({(nb_anomalies / len(series)) * 100:.2f}%).")

        return final_labels