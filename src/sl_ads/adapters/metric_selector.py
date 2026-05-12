import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

# Phase H — absolute import via sl_ads package.
from sl_ads import config


class MetricSelector:
    def __init__(self, method=None, max_metrics=None, corr_threshold=0.85):
        """
        :param corr_threshold: Seuil au-delà duquel deux métriques sont jugées trop
                               similaires (redondantes). 0.85 est un standard empirique.
        """
        self.method = method or config.METRIC_SELECTION_METHOD
        self.max_metrics = max_metrics or config.MAX_METRICS_TO_KEEP
        self.corr_threshold = corr_threshold

    def _remove_constant_features(self, df, metric_cols):
        """
        Élimine les colonnes dont les valeurs ne changent jamais (capteurs éteints/constants).
        """
        # VarianceThreshold de scikit-learn supprime les colonnes avec une variance de 0
        selector = VarianceThreshold(threshold=0.0)
        selector.fit(df[metric_cols])

        # Récupère les colonnes qui ont survécu
        kept_cols = [metric_cols[i] for i in selector.get_support(indices=True)]
        dropped_cols = set(metric_cols) - set(kept_cols)

        if dropped_cols:
            print(f"  -> Variance nulle : {len(dropped_cols)} métriques supprimées {dropped_cols}")

        return kept_cols

    def _filter_by_pearson(self, df, metric_cols):
        """
        Élimine les métriques hautement corrélées linéairement pour éviter
        le double-comptage d'évidence en Logique Subjective.
        """
        print(f"  -> Calcul de la matrice de corrélation de Pearson...")
        # Calcul de la matrice de corrélation absolue
        corr_matrix = df[metric_cols].corr().abs()

        # On ne regarde que le triangle supérieur de la matrice pour éviter les doublons
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Trouve les colonnes dont la corrélation dépasse le seuil
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.corr_threshold)]

        kept_cols = [col for col in metric_cols if col not in to_drop]
        print(f"  -> Redondance (Pearson > {self.corr_threshold}) : {len(to_drop)} métriques supprimées.")

        return kept_cols

    def select_metrics(self, df, ignore_cols=['timestamp', 'label']):
        """
        Fonction principale appelée par les adaptateurs.
        """
        print(f"\n--- Sélection des Métriques (Max: {self.max_metrics}) ---")

        # Identifier les colonnes qui sont des métriques (ni timestamp, ni label)
        metric_cols = [col for col in df.columns if col not in ignore_cols]
        print(f"  -> Métriques initiales : {len(metric_cols)}")

        # 1. Nettoyage de base : retirer les capteurs constants
        metric_cols = self._remove_constant_features(df, metric_cols)

        # 2. Nettoyage de redondance : Pearson
        if self.method.lower() == "pearson":
            metric_cols = self._filter_by_pearson(df, metric_cols)

        # (Optionnel) Si on voulait implémenter Mutual Information (MI), ce serait un elif ici.

        # 3. Limitation au MAX_METRICS (pour éviter d'exploser les temps de calcul de l'IDS)
        if len(metric_cols) > self.max_metrics:
            # Si on a encore trop de métriques, on garde celles avec la plus grande variance
            # (souvent celles qui contiennent le plus de "signal" d'anomalie potentiel)
            variances = df[metric_cols].var().sort_values(ascending=False)
            metric_cols = variances.head(self.max_metrics).index.tolist()
            print(f"  -> Limitation stricte : conservation des {self.max_metrics} métriques les plus variables.")

        print(f"  -> Métriques finales conservées ({len(metric_cols)}) : {metric_cols}\n")

        # On retourne le DataFrame avec uniquement les colonnes utiles
        final_cols = [c for c in ignore_cols if c in df.columns] + metric_cols
        return df[final_cols]