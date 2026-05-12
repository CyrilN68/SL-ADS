import pandas as pd
from abc import ABC, abstractmethod
import os


class AdapterBase(ABC):
    """
    Classe abstraite forçant une structure commune pour tous les datasets.
    Assure l'interopérabilité cross-domain.
    """

    def __init__(self, dataset_name, config_dict):
        self.dataset_name = dataset_name
        self.config = config_dict
        self.raw_data = None
        self.standardized_data = None

    @abstractmethod
    def load_raw_data(self):
        """Doit charger les données brutes dans self.raw_data (DataFrame)"""
        pass

    @abstractmethod
    def extract_metrics(self):
        """Gère la sélection et le renommage des colonnes (timestamp, metric1..., label)"""
        pass

    def run_pipeline(self):
        """Orchestrateur interne pour l'adaptateur"""
        print(f"--- Processing Dataset: {self.dataset_name} ---")
        self.load_raw_data()
        self.extract_metrics()

        # Injection ou labellisation si demandé dans le config.py
        if self.config.get("needs_injection") and not self.config.get("has_labels"):
            print("Lancement du pseudo-labelling consensuel...")
            self.apply_pseudo_labels()

        self.save_standardized_data()

    def apply_pseudo_labels(self):
        """Fait appel au Labeller Unsupervised (défini plus bas)"""
        from sl_ads.adapters.labeller_unsupervised import ConsensusLabeller  # Phase H
        labeller = ConsensusLabeller(period=self.config["seasonality_period"])

        # On assume que les colonnes de métriques commencent après 'timestamp'
        metric_cols = [c for c in self.standardized_data.columns if c not in ['timestamp', 'label']]

        # Application sur la première métrique principale pour l'exemple
        self.standardized_data['label'] = labeller.generate_labels(self.standardized_data[metric_cols[0]])

    def save_standardized_data(self):
        """Sauvegarde universelle au format accepté par SL-ADS"""
        out_path = self.config["path_out"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.standardized_data.to_csv(out_path, index=False)
        print(f"Dataset standardisé sauvegardé sous : {out_path}\n")