import sys
import os
import importlib

# Phase H — script root resolution.  This file lives at
# ``src/sl_ads/adapters/run_cross_dataset.py``; the project root is
# three levels up (src → project root).  Paths inside ``config.py``
# (e.g. ``../data/...``) are still relative to the project root, so we
# chdir there.  src/ is added to sys.path so ``sl_ads.config`` is
# importable.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..'))
_SRC_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
os.chdir(_PROJECT_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from sl_ads import config  # Phase H


def get_adapter_class(dataset_name):
    """
    Outil utilisé : Design Pattern 'Factory' avec Importation Dynamique (importlib).
    Pourquoi ? Plutôt que de charger tous les adaptateurs au démarrage (ce qui consomme
    de la RAM) ou de faire une longue liste de if/elif, ce pattern charge uniquement
    le code nécessaire pour le dataset actif.
    """
    # Mapping entre le nom du dataset du config.py et le fichier/classe Python associé
    adapter_mapping = {
        "RedeRio": ("rederio_adapter", "RederioAdapter"),
        "CESNET-TimeSeries24": ("cesnet_adapter", "CesnetAdapter"),
        "METR-LA": ("metr_la_adapter", "MetrLaAdapter"),
        "GECCO-IoT": ("gecco_adapter", "GeccoAdapter")
        # Ajoutez vos futurs datasets Tier 1 ou Tier 2 ici (ex: SMD, SWaT)
    }

    if dataset_name not in adapter_mapping:
        raise ValueError(f"Adaptateur non défini dans le mapping pour : {dataset_name}")

    module_name, class_name = adapter_mapping[dataset_name]

    # Importation dynamique du module (ex: importe cesnet_adapter.py)
    print(f"  -> Chargement dynamique du module {module_name}...")
    module = importlib.import_module(module_name)

    # Récupération de la classe à l'intérieur du module (ex: CesnetAdapter)
    return getattr(module, class_name)


def main():
    print("=====================================================")
    print("   Orchestrateur d'Adaptation Cross-Domain SL-ADS    ")
    print("=====================================================\n")

    # Lecture stricte depuis le fichier de configuration global
    datasets = config.DATASETS_CONFIG

    # Compteurs pour le résumé final
    processed = 0
    skipped = 0

    for dataset_name, dataset_params in datasets.items():
        if not dataset_params.get("active", False):
            print(f"[SKIP] Dataset '{dataset_name}' marqué inactif dans config.py.")
            skipped += 1
            continue

        print(f"\n[START] Pipeline pour le dataset : {dataset_name}")
        print(f"  -> Configuration chargée : {dataset_params}")

        try:
            # 1. On récupère la bonne classe d'adaptateur
            AdapterClass = get_adapter_class(dataset_name)

            # 2. On instancie la classe avec les paramètres du config.py
            adapter_instance = AdapterClass(dataset_name, dataset_params)

            # 3. Lancement du pipeline (défini dans AdapterBase)
            adapter_instance.run_pipeline()
            processed += 1

        except ImportError as ie:
            print(f"[ERREUR] Fichier adaptateur manquant pour {dataset_name} : {ie}")
            print(f"Avez-vous bien créé le fichier Python correspondant ?")
        except Exception as e:
            print(f"[ERREUR CRITIQUE] Échec du traitement pour {dataset_name} : {e}")

    print("\n=====================================================")
    print(f"Terminé. Datasets traités : {processed} | Ignorés : {skipped}")
    print("=====================================================")


if __name__ == "__main__":
    main()