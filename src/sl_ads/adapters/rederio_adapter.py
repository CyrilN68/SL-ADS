import pandas as pd
import numpy as np
import sys
import os

# Phase H — absolute import via sl_ads package; legacy ``sys.path.insert``
# of ``..`` is no longer needed since the package is installed via
# ``pyproject.toml`` (or src/ is added to sys.path via conftest.py).
from sl_ads.adapters.adapter_base import AdapterBase


class RederioAdapter(AdapterBase):
    """
    Adaptateur pour le dataset RedeRio (UFRJ, réseau brésilien).
    - Fréquence : 30s → 2880 points/jour
    - 14 métriques réseau, sans labels
    - Pseudo-labelling multi-métrique : vote sur bytes, packets, syn,
      entropy_src_ip, entropy_dst_port (morphologies d'attaques complémentaires)

    NOTE performance : STL avec period=2880 sur ~210k points prend ~5-15 min.
    Utiliser DOWNSAMPLE_TO_MIN > 0 pour accélérer (ex: 5 → rééchantillonnage à 5min,
    period=288). Les labels sont ensuite reprojetés sur la résolution native 30s.
    """

    # Toutes les métriques réseau disponibles — couvrent des aspects complémentaires :
    #   Volume global    : flows, packets, bytes
    #   Protocoles       : tcp, udp, icmp
    #   Flags TCP        : syn (SYN flood), fin (teardowns), rst (resets)
    #   Entropies        : entropy_src_ip (scan), entropy_src_port, entropy_dst_port
    #   Taille paquet    : avg_pkt_size (fragmentation, tunneling)
    METRICS_FOR_LABELLING = [
        'flows', 'packets', 'bytes',
        'tcp', 'udp', 'icmp',
        'syn', 'fin', 'rst',
        'entropy_src_ip', 'entropy_src_port', 'entropy_dst_port',
        'avg_pkt_size'
    ]

    # Rééchantillonnage optionnel avant le labeller (en minutes). 0 = désactivé (natif 30s).
    # À 5min : ~42k points, STL beaucoup plus rapide.
    DOWNSAMPLE_TO_MIN = 5

    def load_raw_data(self):
        raw_path = self.config.get("path_raw", "")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"[RedeRio] Fichier introuvable : {raw_path}")

        print(f"  -> [RedeRio] Lecture de {raw_path}...")
        self.raw_data = pd.read_csv(raw_path)
        print(f"  -> [RedeRio] {self.raw_data.shape[0]} lignes chargées "
              f"({self.raw_data.shape[1]} colonnes).")

    def extract_metrics(self):
        df = self.raw_data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Pas de label dans le dataset brésilien
        if 'label' not in df.columns:
            df['label'] = 0

        # PATCH TASK-36 (audit_codex MAJ-01, 2026-04-27) — NaN policy.
        # ``fillna(0)`` on raw network metrics fabricates evidence: a missing
        # bytes/packets sample becomes a "perfectly idle interval" and biases
        # every downstream estimator (Prophet residual variance shrinks, the
        # reconstruction baseline learns a zero attractor, the labeller votes
        # NORMAL on missing intervals).  We instead preserve NaNs here and
        # delegate gap handling to ``preprocess_metrics()``, which applies a
        # bounded forward-fill (``limit=CONFIG.get('NAN_FFILL_LIMIT')``).  The
        # ``label`` column keeps zero-default semantics (unlabelled = NORMAL),
        # which is a deliberate pseudo-labelling policy, not a metric default.
        _LABEL_LIKE = {"label", "anomaly", "is_anomaly", "y", "y_true"}
        for col in df.columns:
            if col == 'timestamp':
                continue
            _coerced = pd.to_numeric(df[col], errors='coerce')
            if col in _LABEL_LIKE:
                df[col] = _coerced.fillna(0).astype(int)
            else:
                df[col] = _coerced  # NaNs preserved → handled by preprocess_metrics

        self.standardized_data = df.sort_values('timestamp').reset_index(drop=True)
        _nan_pct = (
            self.standardized_data.drop(columns=['timestamp', 'label'], errors='ignore')
            .isna().mean().mean() * 100.0
        )
        print(f"  -> [RedeRio] Format standardisé : {self.standardized_data.shape} "
              f"(NaN moyen sur métriques = {_nan_pct:.3f}%)")

    def apply_pseudo_labels(self):
        """
        Vote multi-métrique : pour chaque métrique pertinente, on lance le
        ConsensusLabeller (lui-même un vote 3-algo interne : STL + Hampel + CUSUM).
        Un point est étiqueté anomalie si AU MOINS UNE métrique le signale.
        Cela maximise le rappel — cohérent avec une phase de pseudo-labelling où
        les faux négatifs sont plus coûteux que les faux positifs.
        """
        from sl_ads.adapters.labeller_unsupervised import ConsensusLabeller  # Phase H

        period = self.config["seasonality_period"]
        available = [c for c in self.METRICS_FOR_LABELLING
                     if c in self.standardized_data.columns]

        if not available:
            raise ValueError("[RedeRio] Aucune métrique de labelling disponible dans le dataset.")

        # --- Rééchantillonnage optionnel (accélération STL) ---
        if self.DOWNSAMPLE_TO_MIN > 0:
            freq_str = f"{self.DOWNSAMPLE_TO_MIN}min"
            # Recalcul du period pour la résolution downsamplée
            native_seconds = 30
            downsample_seconds = self.DOWNSAMPLE_TO_MIN * 60
            period_ds = max(2, round(period * native_seconds / downsample_seconds))

            print(f"  -> [RedeRio] Rééchantillonnage à {freq_str} avant labelling "
                  f"(period {period} → {period_ds}, ~{len(self.standardized_data) * native_seconds // downsample_seconds} points).")

            df_ds = self.standardized_data.set_index('timestamp').resample(freq_str).mean()
            df_ds = df_ds.reset_index()
            labeller = ConsensusLabeller(period=period_ds)
            n_native = len(self.standardized_data)
        else:
            df_ds = self.standardized_data
            labeller = ConsensusLabeller(period=period)
            n_native = len(self.standardized_data)

        print(f"  -> [RedeRio] Labelling multi-métrique sur : {available}")

        # Matrice de votes : shape (n_ds_points, n_metrics)
        vote_matrix = np.zeros((len(df_ds), len(available)), dtype=int)

        for i, col in enumerate(available):
            labels_i = labeller.generate_labels(df_ds[col])
            vote_matrix[:, i] = np.asarray(labels_i, dtype=int)
            n = int(vote_matrix[:, i].sum())
            print(f"     {col}: {n} anomalies ({n / len(df_ds) * 100:.2f}%)")

        # PATCH TASK-40 (audit_codex MAJ-04, 2026-04-27): METRIC_VOTE_THRESHOLD
        # is now sourced from CONFIG (key REDERIO_METRIC_VOTE_THRESHOLD),
        # default 5 (cf. config.py).  Justification: each metric flags
        # ~4-8% of points independently; >=5/13 enforces multi-dimensional
        # agreement (DDoS volumetric, mass scans), targeting ~3-7%
        # anomaly rate.  Sweep this in ablation if the resulting rate
        # drifts.
        try:
            import sl_ads.config as _cfg_red  # Phase H
            METRIC_VOTE_THRESHOLD = int(getattr(_cfg_red, 'REDERIO_METRIC_VOTE_THRESHOLD', 5))
        except Exception:
            METRIC_VOTE_THRESHOLD = 5
        labels_ds = (vote_matrix.sum(axis=1) >= METRIC_VOTE_THRESHOLD).astype(int)
        n_ds_anomalies = int(labels_ds.sum())
        print(f"  -> [RedeRio] Vote métriques (≥{METRIC_VOTE_THRESHOLD}/13) : "
              f"{n_ds_anomalies} anomalies ({n_ds_anomalies / len(df_ds) * 100:.2f}% des points downsamplés)")

        # --- Reprojection sur la résolution native 30s ---
        if self.DOWNSAMPLE_TO_MIN > 0:
            df_ds_labels = df_ds[['timestamp']].copy()
            df_ds_labels['label_ds'] = labels_ds

            merged = pd.merge_asof(
                self.standardized_data.sort_values('timestamp'),
                df_ds_labels.sort_values('timestamp'),
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta(f"{self.DOWNSAMPLE_TO_MIN}min")
            )
            self.standardized_data['label'] = merged['label_ds'].fillna(0).astype(int)
        else:
            self.standardized_data['label'] = labels_ds

        n_total = int(self.standardized_data['label'].sum())
        print(f"  -> [RedeRio] Pseudo-labels natifs (30s) : {n_total} anomalies "
              f"({n_total / n_native * 100:.2f}%)")
