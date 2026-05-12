import pandas as pd
import glob
import os
import sys
from sl_ads.adapters.adapter_base import AdapterBase  # Phase H


class MetrLaAdapter(AdapterBase):
    def load_raw_data(self):
        raw_dir = self.config.get("path_raw", "")
        # On cherche spécifiquement vos fichiers .parquet (train, test, val)
        all_files = glob.glob(os.path.join(raw_dir, "**", "*.parquet"), recursive=True)

        if not all_files:
            raise FileNotFoundError(f"[METR-LA] Aucun fichier Parquet trouvé dans {raw_dir}")

        print(f"  -> [METR-LA] {len(all_files)} fichiers Parquet trouvés. Fusion en cours...")

        # Pandas lit les Parquet de manière native et ultra-rapide
        df_list = [pd.read_parquet(file_path) for file_path in all_files]
        self.raw_data = pd.concat(df_list, ignore_index=True)

    def extract_metrics(self):
        df = self.raw_data.copy()

        # 1. On renomme la colonne de temps spécifique à ce fichier
        df.rename(columns={'t0_timestamp': 'timestamp'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # 2. On isole la vitesse actuelle du capteur (le 'd0' à l'instant t+0)
        df.rename(columns={'x_t+0_d0': 'speed'}, inplace=True)

        # ── Détection de la colonne sensor_id ────────────────────────────────
        sensor_id_col = next((c for c in df.columns
                              if c.lower() in ['sensor_id', 'node_id', 'detector_id',
                                               'sensor', 'loop_id', 'id']), None)
        has_sensor_ids = sensor_id_col is not None
        if has_sensor_ids:
            n_sensors = df[sensor_id_col].nunique()
            print(f"  -> [METR-LA] {n_sensors} capteurs détectés (colonne '{sensor_id_col}')")
        else:
            print(f"  -> [METR-LA] Colonne sensor_id non trouvée — agrégation globale.")

        print("  -> [METR-LA] Calcul des 4 métriques d'agrégation spatiale...")

        # ── 4 métriques spatiales par timestamp ──────────────────────────────
        # Chaque timestamp a N lignes (une par capteur).
        #   speed          : moyenne globale (tendance réseau)
        #   speed_std      : écart-type spatial (hétérogénéité → signal incident localisé)
        #   speed_pct_cong : fraction capteurs < 35 mph (zones de congestion réelle)
        #   speed_p10      : 10e percentile (capteurs les plus lents = goulots d'étranglement)
        # Ref : Chen et al. (2001) IEEE T-SMC — divergence spatiale = discriminateur primaire
        #        Coifman (2002) §4.2 — p10 robuste aux capteurs défaillants (Hampel 1974)
        #        HCM (2010) §10-3 — seuil congestion autoroute Los Angeles = 35 mph
        grp = df.groupby('timestamp')['speed']
        df_agg = pd.DataFrame({
            'speed':          grp.mean(),
            'speed_std':      grp.std().fillna(0),
            'speed_pct_cong': grp.apply(lambda x: float((x < 35).mean())),
            'speed_p10':      grp.quantile(0.10),
        }).reset_index()

        # ── Sélection de capteurs représentatifs par corridor (optionnel) ────
        # Si un mapping corridor→sensor_id est disponible, on ajoute 1 métrique
        # par corridor freeway (I-5, I-10, US-101, I-405, I-110).
        # Sinon : top-5 capteurs par variance (sélection non supervisée, pas de leakage).
        # Ref : Li et al. (2018) DCRNN ICLR — capteurs clés Los Angeles freeways
        corridor_metrics_added = 0
        if has_sensor_ids:
            # Essayer de charger un fichier de mapping corridor si disponible
            corridor_map_path = os.path.join(
                os.path.dirname(self.config.get("path_raw", "")), "sensor_corridors.csv"
            )
            if os.path.exists(corridor_map_path):
                print(f"  -> [METR-LA] Chargement mapping corridor : {corridor_map_path}")
                corridor_df = pd.read_csv(corridor_map_path)
                # Format attendu : colonnes sensor_id, corridor (ex: "I-405")
                for corridor_name, group in corridor_df.groupby('corridor'):
                    sensor_ids = group['sensor_id'].tolist()
                    mask = df[sensor_id_col].isin(sensor_ids)
                    if mask.sum() == 0:
                        continue
                    col_name = f"speed_{corridor_name.replace('-','').replace('/','')}"
                    corridor_speed = (df[mask].groupby('timestamp')['speed'].mean()
                                      .rename(col_name))
                    df_agg = df_agg.merge(corridor_speed, on='timestamp', how='left')
                    df_agg[col_name] = df_agg[col_name].fillna(df_agg['speed'])
                    corridor_metrics_added += 1
                    print(f"     → Corridor {corridor_name} : {len(sensor_ids)} capteurs → {col_name}")
            else:
                # PATCH TASK-41 (audit_codex MAJ-06, 2026-04-27).
                #
                # Previously the top-5 variance-ranked sensors were selected on
                # the FULL ``df`` (train ∪ test).  Even though the comment
                # argued "structural property of the road network, label-
                # independent", this is still test-informed feature selection:
                # any incident that perturbs sensor variance during the
                # evaluation period would change the picked sensors and hence
                # change the features the model sees.  By Arp et al. (2022)
                # §4.2 ("Sampling Bias / Tuning on Test"), this is leakage.
                #
                # Fix: read the split timestamp from CONFIG (same value the
                # downstream pipeline uses to slice train/test) and compute the
                # variance ranking on the train slice ONLY.  The selected
                # sensor IDs are then used to build features over the full
                # timeline — but the IDs themselves no longer depend on the
                # test slice.
                _split_date = None
                try:
                    sys.path.insert(0, os.path.abspath(os.path.join(
                        os.path.dirname(__file__), '..')))
                    from sl_ads.config import SELECTED_SPLIT as _SS  # Phase H
                    _split_date = pd.to_datetime(_SS)
                except Exception:
                    _split_date = self.config.get('split_date', None)
                    if _split_date is not None:
                        _split_date = pd.to_datetime(_split_date)

                if _split_date is not None:
                    _train_mask = df['timestamp'] < _split_date
                    n_train_rows = int(_train_mask.sum())
                    print(f"  -> [METR-LA] Sélection top-5 par variance "
                          f"(TRAIN-only, {n_train_rows} rows < {_split_date}).")
                    sensor_var = (df.loc[_train_mask]
                                  .groupby(sensor_id_col)['speed']
                                  .var()
                                  .sort_values(ascending=False))
                    if sensor_var.empty:
                        raise RuntimeError(
                            f"[METR-LA][MAJ-06] Train slice is empty for "
                            f"split_date={_split_date}; cannot compute "
                            "leakage-free sensor ranking."
                        )
                else:
                    import warnings as _w
                    _w.warn(
                        "[METR-LA][MAJ-06] split_date unknown — falling back "
                        "to FULL-data variance ranking. This re-introduces "
                        "test-set leakage on the sensor selection step. Set "
                        "config.SELECTED_SPLIT or DATASETS_CONFIG['METR-LA']"
                        "['split_date'] to fix.",
                        category=UserWarning, stacklevel=1,
                    )
                    sensor_var = df.groupby(sensor_id_col)['speed'].var().sort_values(ascending=False)

                top_sensors = sensor_var.head(5).index.tolist()
                for i, sid in enumerate(top_sensors):
                    col_name = f"speed_sensor_{i+1}"
                    sensor_speed = (df[df[sensor_id_col] == sid]
                                    .groupby('timestamp')['speed'].mean()
                                    .rename(col_name))
                    df_agg = df_agg.merge(sensor_speed, on='timestamp', how='left')
                    df_agg[col_name] = df_agg[col_name].fillna(df_agg['speed'])
                    corridor_metrics_added += 1
                    print(f"     → Capteur {sid} (var={sensor_var[sid]:.2f}) → {col_name}")

        df_agg = df_agg.sort_values('timestamp').reset_index(drop=True)

        # ── Labels ───────────────────────────────────────────────────────────
        label_col = next((c for c in df.columns
                          if c.lower() in ['label', 'is_anomaly', 'anomaly']), None)
        if label_col:
            df_agg['label'] = (df.groupby('timestamp')[label_col].max()
                               .reindex(df_agg['timestamp']).fillna(0).astype(int).values)
            n_pos = int(df_agg['label'].sum())
            print(f"  -> [METR-LA] {n_pos} fenêtres anomales trouvées dans les parquets.")
        else:
            df_agg['label'] = 0
            print("  -> [METR-LA] Aucun label dans les parquets — label=0 (ConsensusLabeller requis).")

        self.standardized_data = df_agg

        n_metrics = len([c for c in df_agg.columns
                         if c not in ['timestamp', 'label']])
        print(f"  -> [METR-LA] Format final : {df_agg.shape} | "
              f"{n_metrics} métriques | {corridor_metrics_added} corridor(s)")
        print(f"     Colonnes : {[c for c in df_agg.columns if c != 'timestamp']}")


if __name__ == "__main__":
    # ── Test standalone de l'adaptateur ─────────────────────────────────────
    # Usage : python metr_la_adapter.py
    # Permet de tester la génération du CSV sans lancer le pipeline complet.
    import sys
    print("=" * 60)
    print("  MetrLaAdapter — test standalone")
    print("=" * 60)

    # Lire le config pour récupérer les vrais chemins
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _script_dir)
    try:
        from sl_ads.config import DATASETS_CONFIG  # Phase H
        cfg = DATASETS_CONFIG.get("METR-LA", {})
    except ImportError:
        # Fallback chemins relatifs si config non disponible
        cfg = {
            "path_raw": "../data/METR-LA/raw/",
            "path_out": "../data_standardized/METR_LA.csv",
        }

    print(f"  Répertoire raw   : {cfg.get('path_raw')}")
    print(f"  Fichier sortie   : {cfg.get('path_out')}")

    adapter = MetrLaAdapter("METR-LA", cfg)
    try:
        # run_pipeline() gère automatiquement l'extraction ET le pseudo-labelling
        adapter.run_pipeline()
        df_out = adapter.standardized_data
        print(f"\n✅ CSV généré : {cfg.get('path_out')}")
        print(f"   {len(df_out)} lignes | {list(df_out.columns)}")
        print(f"\n   Aperçu :")
        print(df_out.head(3).to_string(index=False))
        print(f"\n   Labels : {int(df_out['label'].sum())} anomalies "
              f"/ {len(df_out)} fenêtres "
              f"({100*df_out['label'].mean():.2f}%)")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()