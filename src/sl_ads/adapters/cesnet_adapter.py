import pandas as pd
import glob
import os
import numpy as np
from sl_ads.adapters.adapter_base import AdapterBase  # Phase H


class CesnetAdapter(AdapterBase):
    """
    Adaptateur réel pour le dataset CESNET-TimeSeries24.
    Fusionne les 289 fichiers CSV de sous-réseaux en une vue ISP globale.
    """

    def load_raw_data(self):
        raw_dir = self.config.get("path_raw", "")

        # Scan récursif pour trouver vos 289 fichiers (même s'ils sont dans data/cesnet/raw/)
        all_files = glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)

        if not all_files:
            raise FileNotFoundError(f"[CESNET] ERREUR : Aucun fichier CSV trouvé dans {raw_dir}.")

        print(f"  -> [CESNET] Scan terminé : {len(all_files)} fichiers CSV trouvés.")
        print("  -> [CESNET] Lecture en cours... (Cela peut prendre un peu de temps)")

        df_list = []
        for file_path in all_files:
            try:
                df_temp = pd.read_csv(file_path)
                df_list.append(df_temp)
            except Exception as e:
                print(f"     [Avertissement] Fichier ignoré ({file_path}) : {e}")

        if not df_list:
            raise ValueError("[CESNET] ERREUR : Tous les fichiers étaient vides ou illisibles.")

        self.raw_data = pd.concat(df_list, ignore_index=True)
        print(f"  -> [CESNET] Données brutes fusionnées en mémoire : {self.raw_data.shape[0]} lignes.")

    def extract_metrics(self):
        df = self.raw_data.copy()

        # 1. Standardisation des colonnes obligatoires (Basé sur VOS données)
        col_mapping = {
            'id_time': 'timestamp',  # C'est votre vraie colonne de temps
            'Attack_Label': 'label',
            'is_anomaly': 'label'
        }
        df.rename(columns=col_mapping, inplace=True)

        if 'timestamp' not in df.columns:
            raise ValueError("[CESNET] Colonne temporelle (id_time) introuvable dans les CSV.")

        # PATCH TASK-46 (audit_codex MAJ-05, 2026-04-27) — synthetic
        # timestamps disclosure.  CESNET-TimeSeries24 ships ``id_time`` as
        # an integer counter, NOT a wall-clock timestamp.  We synthesize a
        # 10-minute-step timeline from a fixed anchor (2024-01-01) so that
        # downstream Prophet and STL components have a regular calendar
        # axis.  This is sufficient for *non-calendar-aware* anomaly
        # detection (statistical decomposition, residual modelling) but
        # MUST NOT be relied upon for analyses that depend on real
        # weekday/hour-of-day patterns.  The honest_limitations.md file
        # documents this; the ``CESNET_TIMESTAMP_MODE`` config key lets
        # callers opt into a different anchor or reject CESNET entirely.
        try:
            from sl_ads.config import CONFIG as _CFG_CESNET  # Phase H
            _ts_mode = _CFG_CESNET.get('CESNET_TIMESTAMP_MODE', 'fabricated_warning')
            _ts_anchor = _CFG_CESNET.get('CESNET_TIMESTAMP_ANCHOR', '2024-01-01')
        except Exception:
            _ts_mode = 'fabricated_warning'
            _ts_anchor = '2024-01-01'

        if _ts_mode == 'reject':
            raise RuntimeError(
                "[CESNET][MAJ-05] CESNET_TIMESTAMP_MODE='reject' — refusing "
                "to run on CESNET because real timestamps are not available."
            )
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['timestamp'] = pd.to_datetime(_ts_anchor) + pd.to_timedelta(df['timestamp'] * 10, unit='m')
        if _ts_mode == 'fabricated_warning':
            import warnings as _w
            _w.warn(
                f"[CESNET][MAJ-05] timestamps are SYNTHETIC (10-min steps "
                f"from {_ts_anchor}). Calendar-aware analyses (e.g. Prophet "
                f"weekly seasonality) will be biased. "
                "See docs/honest_limitations.md.",
                category=UserWarning, stacklevel=2,
            )

        # GESTION DU LABEL MANQUANT : Si le dataset téléchargé n'a pas les attaques annotées
        if 'label' not in df.columns:
            print("  -> [CESNET] Info : Aucune colonne de label d'attaque trouvée. Création d'un label = 0 par défaut.")
            df['label'] = 0

        # PATCH TASK-36 (audit_codex MAJ-01, 2026-04-27) — NaN policy.
        # Replace ``fillna(0)`` (silently fabricates idle traffic) with NaN
        # preservation; downstream ``preprocess_metrics()`` does the bounded
        # forward-fill.  Labels keep zero-default semantics (unlabelled =
        # NORMAL) — that's a deliberate pseudo-labelling policy, not a
        # metric default.
        for col in df.columns:
            if col in ('timestamp', 'label'):
                continue
            _coerced = pd.to_numeric(df[col], errors='coerce')
            df[col] = _coerced
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
        _nan_pct_metrics = (
            df.drop(columns=['timestamp', 'label'], errors='ignore')
            .isna().mean().mean() * 100.0
        )
        print(f"  -> [CESNET] NaN moyen sur métriques pré-aggregation = {_nan_pct_metrics:.3f}%")

        print("  -> [CESNET] Agrégation des 289 institutions par timestamp...")



        # CORRECTION — Paradoxe de Simpson (Blyth, 1972) :
        # Faire la moyenne de ratios entre sous-réseaux de tailles très différentes
        # donne un résultat faux. Ex : réseau A (10 paquets, 90% TCP) et réseau B
        # (10 000 paquets, 10% TCP) → moyenne des ratios = 50%, mais le vrai
        # ratio global est ~10.07%.
        # SOLUTION : exclure les colonnes non-additives du groupby, puis les
        # recalculer depuis les volumes absolus agrégés (ou moyenne pondérée).
        agg_dict = {'label': 'max'}
        deferred_cols = []  # colonnes non-additives traitées post-agrégation

        for col in df.columns:
            if col in ['timestamp', 'label']:
                continue
            col_lower = col.lower()
            if any(kw in col_lower for kw in ['ratio', 'average', 'avg', 'std']):
                deferred_cols.append(col)   # non-additive → traitement différé
            elif pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = 'sum'       # volumes et compteurs : additifs

        df_agg = df.groupby('timestamp').agg(agg_dict).reset_index()
        df_agg.sort_values('timestamp', inplace=True)

        # --- Recalcul post-agrégation des colonnes non-additives ---
        for col in deferred_cols:
            col_lower = col.lower()
            recomputed = False

            # tcp_udp_ratio : recalcul depuis les colonnes absolues TCP et UDP agrégées
            if 'tcp_udp' in col_lower:
                tcp_col = next((c for c in df_agg.columns
                                if 'tcp' in c.lower() and 'udp' not in c.lower()
                                and 'ratio' not in c.lower()), None)
                udp_col = next((c for c in df_agg.columns
                                if 'udp' in c.lower() and 'tcp' not in c.lower()
                                and 'ratio' not in c.lower()), None)
                if tcp_col and udp_col:
                    df_agg[col] = df_agg[tcp_col] / (df_agg[udp_col] + 1e-9)
                    print(f"  -> [CESNET] '{col}' recalculé depuis {tcp_col}/{udp_col} "
                          f"(correction Paradoxe de Simpson).")
                    recomputed = True

            if not recomputed:
                # Fallback : moyenne pondérée par n_packets (meilleure approximation)
                wt_col = next((c for c in ['n_packets', 'n_flows', 'n_bytes']
                               if c in df.columns), None)
                if wt_col is not None:
                    wm = df.groupby('timestamp').apply(
                        lambda g, c=col, w=wt_col: np.average(
                            g[c].fillna(0),
                            weights=np.abs(g[w].fillna(0)) + 1e-9
                        )
                    ).reset_index(name=col)
                    df_agg = df_agg.merge(wm, on='timestamp', how='left')
                    print(f"  -> [CESNET] '{col}' : moyenne pondérée par {wt_col} "
                          f"(fallback Simpson, colonnes absolues non identifiées).")
                else:
                    print(f"  -> [CESNET] '{col}' ignoré — impossible de corriger "
                          f"le biais de Simpson (aucun volume absolu disponible).")
        print(f"  -> [CESNET] Série temporelle globale ISP : {df_agg.shape[0]} pas de temps.")

        # 3. Réduction de dimension avec MetricSelector
        # 3. On garde TOUTES les métriques pour RANSAC et Prophet
        print("  -> [CESNET] Conservation de toutes les métriques pour modélisation RANSAC...")
        self.standardized_data = df_agg

        print(f"  -> [CESNET] Préparation terminée. Format final : {self.standardized_data.shape}")