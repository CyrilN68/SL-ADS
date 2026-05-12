"""
compare_labeller_vs_sl.py
=========================
Compare les pseudo-labels du ConsensusLabeller (RedeRio.csv, colonne 'label')
avec les détections du modèle SL-ADS (detection_results*.csv, FINAL_SYSTEM_CBF_*).

═══════════════════════════════════════════════════════════════════════════════
PATCH-m6 (2026-04-18 / 2026-04-19) — CADRE ÉPISTÉMOLOGIQUE : accord inter-annotateurs
═══════════════════════════════════════════════════════════════════════════════

ATTENTION — Ce script N'EST PAS une validation scientifique de SL-ADS.

Les pseudo-labels du ConsensusLabeller (RedeRio.csv) sont générés par un
dispositif statistique non-supervisé (fusion de détecteurs hétérogènes) ;
ils NE constituent PAS une ground truth vérifiée par expert humain.

Par conséquent, les métriques (κ Cohen, accord %) produites ici doivent être
interprétées comme un **accord inter-annotateurs** (inter-rater agreement,
Cohen 1960 Educ. Psychol. Meas.) entre DEUX détecteurs, et non comme une
mesure de performance de SL-ADS par rapport à une vérité terrain.

Interprétation admissible :
  • Forte corrélation → les deux détecteurs réagissent à des signaux communs
    (cohérence empirique). Ne prouve PAS que l'un ou l'autre est correct.
  • Faible corrélation → les deux méthodes explorent des sous-espaces
    différents du signal. N'invalide NI le labeller NI SL-ADS.

Interprétation interdite :
  • "SL-ADS a une précision de X% par rapport à RedeRio.label" — FAUX.
  • "Le labeller valide SL-ADS" — FAUX (circularité : c'est un détecteur
    comparé à un autre détecteur, pas à un gold standard annoté).

La validation scientifique proprement dite s'effectue via evaluate_injection_v2.py
(catalogue synthétique `INJECTED_ATTACK_CATALOG` avec ground truth parfaite).

Ref : Cohen (1960) ; Landis & Koch (1977) Biometrics 33(1):159-174 ; Artstein
      & Poesio (2008) Comput. Linguist. 34(4):555-596 (inter-annotator measures).
═══════════════════════════════════════════════════════════════════════════════

Utilisation :
    python compare_labeller_vs_sl.py                      # seuil auto (sidecar JSON)
    python compare_labeller_vs_sl.py --threshold 0.30     # seuil manuel
    python compare_labeller_vs_sl.py --no-plot            # sans graphiques

Sorties :
    compare_labeller_vs_sl.csv        — tableau aligné (timestamp, pseudo_label, sl_score, sl_pred)
    compare_labeller_vs_sl.png        — timeline des accords / désaccords
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# PATCH TASK-28 (audit_tmp MAJ-07, 2026-04-26): targeted filters only.
# L'ancien ``warnings.filterwarnings("ignore")`` global masquait des alertes
# scientifiquement significatives (divisions silencieuses, dépréciations).
# On ne supprime que les warnings non actionnables pour ce script.
warnings.filterwarnings("ignore", category=FutureWarning,      module=r"pandas")
warnings.filterwarnings("ignore", category=UserWarning,        module=r"matplotlib")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"matplotlib")

# === Config & paths === (Phase H — absolute imports via sl_ads package)
from sl_ads.config import CONFIG
from sl_ads.paths import get_results_dir, get_decision_threshold, get_detection_col

RESULTS_DIR       = get_results_dir(CONFIG, up_levels=1)
PSEUDOLABEL_FILE  = CONFIG["file_path"]           # data_standardized/RedeRio.csv
WINDOW_SIZE       = CONFIG.get("WINDOW_SIZE", 10)  # nb de points par fenêtre SL-ADS
FREQ              = CONFIG.get("freq_data", "30s")


# =============================================================================
def load_detection_results(results_dir: str) -> pd.DataFrame:
    """
    Charge detection_results.csv (SANS injection) en priorité pour la comparaison
    vs labeller — les deux doivent porter sur les mêmes données non-modifiées.
    Si seul le fichier INJECTED existe, on le charge mais on filtre les fenêtres
    injectées (injection_label != 0) pour éviter les faux FP artificiels.
    """
    clean_path = os.path.join(results_dir, "detection_results.csv")
    inj_path   = os.path.join(results_dir, "detection_results_INJECTED.csv")

    if os.path.exists(clean_path):
        print(f"[SL-ADS] Chargement (non-injecté) : {clean_path}")
        return pd.read_csv(clean_path, parse_dates=["timestamp"])

    if os.path.exists(inj_path):
        print(f"[SL-ADS] Chargement (injecté) : {inj_path}")
        df = pd.read_csv(inj_path, parse_dates=["timestamp"])
        if "injection_label" in df.columns:
            n_before = len(df)
            df = df[df["injection_label"] == 0].copy()
            print(f"[SL-ADS] Filtrage fenêtres injectées : {n_before - len(df)} retirées → {len(df)} restantes")
        return df

    raise FileNotFoundError(
        f"Aucun fichier detection_results*.csv trouvé dans {results_dir}.\n"
        "Lance d'abord : python run_full_sl_ads.py --dataset RedeRio --to-step opinions"
    )


def load_pseudo_labels(path: str) -> pd.DataFrame:
    """Charge le fichier standardisé RedeRio avec pseudo-labels."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier pseudo-labels introuvable : {path}\n"
            "Lance d'abord : cd dataset_adapter && python run_cross_dataset.py"
        )
    print(f"[Labeller] Chargement : {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "label" not in df.columns:
        raise ValueError("Colonne 'label' absente — pseudo-labels non générés ?")
    return df[["timestamp", "label"]].rename(columns={"label": "pseudo_label"})


def align_pseudo_labels(df_sl: pd.DataFrame, df_lbl: pd.DataFrame,
                        window_size: int, freq: str) -> pd.DataFrame:
    """
    Aligne les pseudo-labels (résolution native 30s) sur les fenêtres SL-ADS.
    Un point SL est marqué 1 si au moins un des `window_size` instants natifs
    qui le précèdent contenait un pseudo-label = 1.
    """
    # Durée d'une fenêtre en secondes
    freq_seconds = pd.tseries.frequencies.to_offset(freq).nanos / 1e9
    window_duration = pd.Timedelta(seconds=int(freq_seconds * window_size))

    # Pour chaque timestamp SL, chercher le max des labels dans la fenêtre [t-W, t]
    df_lbl_indexed = df_lbl.set_index("timestamp").sort_index()

    aligned = []
    for ts in df_sl["timestamp"]:
        window_labels = df_lbl_indexed.loc[ts - window_duration: ts, "pseudo_label"]
        aligned.append(int(window_labels.max()) if len(window_labels) > 0 else 0)

    return aligned


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Precision, Recall, F1, accord (accuracy simple)."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def print_report(metrics: dict, threshold: float, n_labeller: int,
                 n_sl: int, n_total: int):
    print("\n" + "="*60)
    print("  COMPARAISON  ConsensusLabeller  vs  SL-ADS")
    print("="*60)
    print(f"  Fenêtres comparées   : {n_total:>8,}")
    print(f"  Pseudo-labels = 1    : {n_labeller:>8,}  ({n_labeller/n_total*100:.1f}%)")
    print(f"  SL-ADS détecte = 1   : {n_sl:>8,}  ({n_sl/n_total*100:.1f}%)")
    print(f"  Seuil SL utilisé     : {threshold:.3f}")
    print("-"*60)
    print(f"  TP (accord anomalie) : {metrics['tp']:>8,}")
    print(f"  TN (accord normal)   : {metrics['tn']:>8,}")
    print(f"  FP (SL seul)         : {metrics['fp']:>8,}  (SL détecte, labeller non)")
    print(f"  FN (labeller seul)   : {metrics['fn']:>8,}  (labeller détecte, SL non)")
    print("-"*60)
    print(f"  Precision  : {metrics['precision']:.3f}  (parmi détections SL, % confirmées par labeller)")
    print(f"  Recall     : {metrics['recall']:.3f}  (parmi anomalies labeller, % vues par SL)")
    print(f"  F1         : {metrics['f1']:.3f}")
    print(f"  Accord     : {metrics['accuracy']:.3f}")
    print("="*60)


def plot_comparison(df: pd.DataFrame, output_path: str):
    """Timeline accord / désaccord sur toute la période."""
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.suptitle("Comparaison ConsensusLabeller vs SL-ADS — RedeRio", fontsize=13, fontweight='bold')

    ts = df["timestamp"]

    # Score SL brut
    ax0 = axes[0]
    ax0.plot(ts, df["sl_score"], color="#2196F3", linewidth=0.6, alpha=0.8)
    ax0.axhline(df["threshold"].iloc[0], color="red", linewidth=1, linestyle="--",
                label=f"Seuil = {df['threshold'].iloc[0]:.2f}")
    ax0.set_ylabel("proj_atk SL-ADS")
    ax0.legend(fontsize=8)
    ax0.set_ylim(0, 1)

    # Pseudo-labels
    ax1 = axes[1]
    lbl_vals = df["pseudo_label"].values.astype(float)
    ax1.fill_between(ts, 0, lbl_vals, step="mid", color="#FF9800", alpha=0.6, label="Pseudo-label (labeller)")
    ax1.set_ylabel("Pseudo-label")
    ax1.set_ylim(-0.1, 1.3)
    ax1.legend(fontsize=8)

    # Accords et désaccords
    ax2 = axes[2]
    tp_mask = (df["pseudo_label"] == 1) & (df["sl_pred"] == 1)
    fp_mask = (df["pseudo_label"] == 0) & (df["sl_pred"] == 1)
    fn_mask = (df["pseudo_label"] == 1) & (df["sl_pred"] == 0)

    ax2.fill_between(ts, 0, tp_mask.astype(float), step="mid",
                     color="#4CAF50", alpha=0.8, label=f"TP ({tp_mask.sum():,})")
    ax2.fill_between(ts, 0, fp_mask.astype(float) * 0.7, step="mid",
                     color="#F44336", alpha=0.6, label=f"FP SL seul ({fp_mask.sum():,})")
    ax2.fill_between(ts, 0, fn_mask.astype(float) * 0.4, step="mid",
                     color="#FF9800", alpha=0.6, label=f"FN labeller seul ({fn_mask.sum():,})")
    ax2.set_ylabel("Accord")
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend(fontsize=8)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Sauvegardé : {output_path}")


# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None,
                        help="Seuil de décision SL (défaut : sidecar JSON ou 0.20)")
    parser.add_argument("--no-plot", action="store_true", help="Désactiver le graphique")
    args = parser.parse_args()

    # --- Chargement ---
    df_sl  = load_detection_results(RESULTS_DIR)
    df_lbl = load_pseudo_labels(PSEUDOLABEL_FILE)

    # --- Seuil ---
    threshold = args.threshold if args.threshold is not None else get_decision_threshold(CONFIG)
    print(f"[Config] Seuil de décision : {threshold:.3f}")

    # --- Colonne de score SL ---
    det_col = get_detection_col(CONFIG)
    if det_col not in df_sl.columns:
        # Fallback si la variable change selon les artefacts
        fallbacks = ["FINAL_SYSTEM_CBF_proj_atk", "FINAL_SYSTEM_CBF_b_atk"]
        det_col = next((c for c in fallbacks if c in df_sl.columns), None)
        if det_col is None:
            print(f"[ERREUR] Colonne de détection introuvable. Colonnes disponibles :")
            print([c for c in df_sl.columns if "FINAL" in c or "proj_atk" in c])
            sys.exit(1)
    print(f"[SL-ADS] Colonne de détection : {det_col}")

    # --- Alignement temporel ---
    print(f"[Align] Alignement pseudo-labels sur fenêtres SL (WINDOW_SIZE={WINDOW_SIZE}, freq={FREQ})...")
    aligned_labels = align_pseudo_labels(df_sl, df_lbl, WINDOW_SIZE, FREQ)

    # --- Construction du tableau comparatif ---
    df_compare = pd.DataFrame({
        "timestamp":    df_sl["timestamp"],
        "sl_score":     df_sl[det_col],
        "sl_pred":      (df_sl[det_col] >= threshold).astype(int),
        "pseudo_label": aligned_labels,
        "threshold":    threshold,
    })

    # --- Métriques ---
    m = compute_metrics(df_compare["pseudo_label"].values, df_compare["sl_pred"].values)
    print_report(m,
                 threshold=threshold,
                 n_labeller=int(df_compare["pseudo_label"].sum()),
                 n_sl=int(df_compare["sl_pred"].sum()),
                 n_total=len(df_compare))

    # --- Export CSV ---
    out_csv = os.path.join(RESULTS_DIR, "compare_labeller_vs_sl.csv")
    df_compare.to_csv(out_csv, index=False)
    print(f"[CSV] Sauvegardé : {out_csv}")

    # --- Graphique ---
    if not args.no_plot:
        out_png = os.path.join(RESULTS_DIR, "compare_labeller_vs_sl.png")
        plot_comparison(df_compare, out_png)


if __name__ == "__main__":
    main()
