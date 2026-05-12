import pandas as pd
import glob
import os
from sl_ads.adapters.adapter_base import AdapterBase  # Phase H


class GeccoAdapter(AdapterBase):
    def load_raw_data(self):
        # PATCH TASK-42 (audit_codex MAJ-07, 2026-04-27).  Previously this
        # method silently read ``files[0]`` and ignored every other CSV in
        # the directory tree — meaning a future GECCO redistribution
        # split into multiple files would lose data without any warning.
        # Now we either:
        #   (a) load and concatenate ALL CSVs (default, GECCO_LOAD_MODE
        #       == "concat") — sorted by filename for determinism;
        #   (b) require exactly one CSV (GECCO_LOAD_MODE == "single") and
        #       raise if more than one is present.
        raw_dir = self.config.get("path_raw", "")
        files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True))
        if not files:
            raise FileNotFoundError(f"[GECCO] Aucun fichier trouvé dans {raw_dir}")

        try:
            import sl_ads.config as _cfg_gecco  # Phase H
            mode = getattr(_cfg_gecco, 'GECCO_LOAD_MODE', 'concat')
        except Exception:
            mode = 'concat'

        if mode == 'single':
            if len(files) != 1:
                raise RuntimeError(
                    f"[GECCO][MAJ-07] GECCO_LOAD_MODE='single' but found "
                    f"{len(files)} CSV files in {raw_dir}: {files}. "
                    "Either set GECCO_LOAD_MODE='concat' to merge them or "
                    "trim the directory to exactly one file."
                )
            print(f"  -> [GECCO] Lecture du fichier {files[0]}...")
            self.raw_data = pd.read_csv(files[0])
            return

        # Default mode: concat-all (deterministic, no silent data loss).
        print(f"  -> [GECCO] Lecture concaténée de {len(files)} fichier(s) "
              "(mode 'concat', tri lexicographique pour déterminisme).")
        df_list = []
        for fp in files:
            try:
                _df = pd.read_csv(fp)
                df_list.append(_df)
                print(f"     ✓ {os.path.basename(fp)} : {_df.shape}")
            except Exception as _e:
                print(f"     ✗ {os.path.basename(fp)} : SKIPPED ({_e})")

        if not df_list:
            raise ValueError(
                f"[GECCO][MAJ-07] All {len(files)} candidate CSV(s) failed "
                f"to parse in {raw_dir}."
            )
        self.raw_data = pd.concat(df_list, ignore_index=True)
        print(f"  -> [GECCO] Concaténation totale : {self.raw_data.shape}")

    def extract_metrics(self):
        df = self.raw_data.copy()

        # Renommage pour SL-ADS
        df.rename(columns={'Time': 'timestamp', 'EVENT': 'label'}, inplace=True)

        # Conversion du label TRUE/FALSE en 1/0
        if df['label'].dtype == object or df['label'].dtype == bool:
            # Gère les chaînes de caractères "TRUE"/"FALSE" ou les booléens
            df['label'] = df['label'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0, '1': 1, '0': 0})
            df['label'] = df['label'].fillna(0).astype(int)

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Nettoyage rapide (suppression des colonnes inutiles)
        cols_to_keep = ['timestamp', 'label', 'Tp', 'Cl', 'pH', 'Redox', 'Leit', 'Trueb']
        self.standardized_data = df[[c for c in cols_to_keep if c in df.columns]]

        print(f"  -> [GECCO] Format final prêt : {self.standardized_data.shape}")