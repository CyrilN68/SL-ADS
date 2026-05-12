"""
extract_individual_f1.py — F1 individuels par métrique feuille
===============================================================
Calcule les métriques de détection de chaque métrique feuille
individuellement (avant toute fusion inter-métriques) sur l'attaque
DDoS réelle (12-13 novembre 2025).

Produit :
  - individual_f1_table4.csv   : Table 4 du rapport (F1 prelim + stabilité)
  - individual_f1_table12.csv  : Table 12 du rapport (F1 + TP/Total + notes)
  - individual_f1_table12.png  : Version graphique publication-ready de Table 12

Paramètres lus depuis config.py :
  CONFIG["EVAL"]["RESULTS_DIR"]         : dossier des résultats
  CONFIG["EVAL"]["RESULTS_CSV_NAME"]    : nom du CSV de détection
  CONFIG["EVAL"]["REAL_ATTACK_CATALOG"] : définition de l'attaque réelle
  CONFIG["EVAL_REAL"]["THRESHOLD"]      : seuil décisionnel (b_atk)
  CONFIG["VERSION_NAME"]                : pour localiser le metadata CSV

Usage :
  Placer dans le dossier du projet (avec config.py) et exécuter :
    python extract_individual_f1.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from config import CONFIG
except ImportError:
    print("❌ config.py introuvable.")
    sys.exit(1)

try:
    _parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from paths import get_decision_threshold as _get_decision_threshold
    from paths import get_detection_col as _get_detection_col
    _HAS_PATHS_EIF = True
except ImportError:
    _HAS_PATHS_EIF = False

# ==============================================================================
# PARAMÈTRES — tous depuis config.py
# ==============================================================================
_EVAL      = CONFIG.get("EVAL", {})
_EVAL_REAL = CONFIG.get("EVAL_REAL", {})
VERSION_NAME  = CONFIG.get("VERSION_NAME", "trained_models_v9_v6_v4s")

RESULTS_DIR   = _EVAL.get("RESULTS_DIR", f"../results/resultats_{VERSION_NAME}")
RESULTS_CSV   = os.path.join(RESULTS_DIR, _EVAL.get("RESULTS_CSV_NAME", "detection_results_INJECTED.csv"))
METADATA_CSV  = os.path.join(RESULTS_DIR, "../..", f"resultats_{VERSION_NAME}",
                              f"metadata_{VERSION_NAME}.csv")
OUTPUT_DIR    = os.path.join(RESULTS_DIR, "evaluation_real_ddos")

THRESHOLD     = _EVAL_REAL.get("THRESHOLD",
                _get_decision_threshold(CONFIG, up_levels=2) if _HAS_PATHS_EIF
                else _EVAL.get("DECISION_THRESHOLD", 0.20))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# CHARGEMENT
# ==============================================================================
print(f"\n{'='*60}")
print("  extract_individual_f1.py — F1 par métrique feuille")
print(f"{'='*60}\n")

# Résultats de détection
if not os.path.exists(RESULTS_CSV):
    print(f"❌ Fichier introuvable : {RESULTS_CSV}")
    sys.exit(1)
df = pd.read_csv(RESULTS_CSV, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"-> CSV détection : {len(df)} fenêtres | {df['timestamp'].min()} → {df['timestamp'].max()}")

# Métadonnées (r2, stability_class, clean_key, type)
if not os.path.exists(METADATA_CSV):
    # Chercher dans le même dossier que le CSV de résultats
    alt = os.path.join(RESULTS_DIR, f"metadata_{VERSION_NAME}.csv")
    if os.path.exists(alt):
        METADATA_CSV = alt
    else:
        print(f"❌ Metadata CSV introuvable : {METADATA_CSV}")
        print("   Vérifier VERSION_NAME dans config.py.")
        sys.exit(1)

df_meta = pd.read_csv(METADATA_CSV)
print(f"-> Metadata       : {len(df_meta)} métriques")

# Reconstruction du dictionnaire : deux index pour retrouver les métadonnées
# depuis les deux conventions de nommage :
#   - metric_key   : "prophet_bytes", "reconst_bytes_from_packets"  (metadata CSV)
#   - csv_prefix   : "P_bytes", "R_bytes_from_packets"              (colonnes du CSV détection)
# compute_opinions_v2.py applique : metric_key → csv_prefix via :
#   .replace("prophet_", "P_").replace("reconst_", "R_").replace("->", "_to_")
meta_by_metric_key  = {}  # index par metric_key  (ex: "prophet_bytes")
meta_by_csv_prefix  = {}  # index par csv_prefix  (ex: "P_bytes")

for _, row in df_meta.iterrows():
    metric_key = row["metric_key"]
    # Reconstituer le csv_prefix tel que compute_opinions_v2 le calcule
    csv_prefix = (metric_key
                  .replace("prophet_", "P_")
                  .replace("reconst_", "R_")
                  .replace("->", "_to_"))

    entry = {
        "metric_key":      metric_key,
        "csv_prefix":      csv_prefix,
        "type":            row["type"],
        "r2":              float(row.get("r2_weight",       row.get("r2", 0.0))),
        "stability_class": str(row.get("stability_class",  "?")),
    }
    meta_by_metric_key[metric_key] = entry
    meta_by_csv_prefix[csv_prefix] = entry

print(f"   metric_keys  : {list(meta_by_metric_key.keys())}")
print(f"   csv_prefixes : {list(meta_by_csv_prefix.keys())}")

# ==============================================================================
# DÉFINITION DE L'ATTAQUE RÉELLE
# ==============================================================================
real_attacks = _EVAL.get("REAL_ATTACK_CATALOG", [])
if not real_attacks:
    print("❌ REAL_ATTACK_CATALOG vide dans config.py.")
    sys.exit(1)

atk    = real_attacks[0]
t0     = pd.Timestamp(atk["start"])
t1     = t0 + pd.Timedelta(hours=atk["duration_h"])
mask_atk = (df["timestamp"] >= t0) & (df["timestamp"] < t1)
df_atk   = df[mask_atk].copy()
n_gt     = len(df_atk)

print(f"\n-> Attaque réelle : {atk['name']}")
print(f"   Période        : {t0} → {t1}")
print(f"   Fenêtres GT    : {n_gt}")
print(f"   Seuil          : b_atk ≥ {THRESHOLD}\n")

if n_gt == 0:
    print("❌ Aucune fenêtre dans la période d'attaque. Vérifier REAL_ATTACK_CATALOG.")
    sys.exit(1)

# Fenêtres normales pour FPR (hors toutes attaques du catalogue)
try:
    from inject_at_evidence_level import ATTACK_CATALOG as _synthetic
    full_catalog = list(_synthetic)
except ImportError:
    full_catalog = []
full_catalog = full_catalog + real_attacks

def is_in_attack(ts, catalog):
    for a in catalog:
        t_s = pd.Timestamp(a["start"])
        t_e = t_s + pd.Timedelta(hours=a["duration_h"])
        if t_s <= ts < t_e:
            return True
    return False

mask_normal = df["timestamp"].apply(lambda ts: not is_in_attack(ts, full_catalog))
df_normal   = df[mask_normal].copy()
n_normal    = len(df_normal)
print(f"-> Fenêtres normales (hors toutes attaques) : {n_normal}")

# ==============================================================================
# CALCUL F1 PAR MÉTRIQUE
# ==============================================================================

def compute_individual_f1(df_atk, df_normal, csv_prefix, threshold):
    """
    Calcule les métriques de détection pour une métrique feuille isolée.

    Calcule sur DEUX colonnes en parallèle :
      - {csv_prefix}_b_atk   : belief mass directe (C1 ageing output)
      - {csv_prefix}_proj_atk : probabilité projetée P(Atk) = b + a*u
                                (intègre le base rate adaptatif C4)

    Le rapport utilise la colonne proj_atk pour la règle de décision
    (cf. Section 6.2.1 bug #1 identifié en passe cohérence).
    On calcule les deux pour comparaison.

    Retourne un dict avec les métriques pour les deux colonnes.
    """
    col_b    = f"{csv_prefix}_b_atk"
    col_proj = f"{csv_prefix}_proj_atk"

    # Vérifier au moins une colonne présente
    if col_b not in df_atk.columns and col_proj not in df_atk.columns:
        return None

    results = {}
    for col_name, col in [("b_atk", col_b), ("proj_atk", col_proj)]:
        if col not in df_atk.columns:
            results[col_name] = None
            continue

        # NaN dans les données d'attaque : remplacé par 0 (fenêtre sans signal = non-détecté).
        vals_gt = df_atk[col].fillna(0.0).values

        # NaN dans les données normales : forward fill puis 0 résiduel.
        # Remplir par 0 directement créerait des FP artificiels si threshold > 0,
        # car b_atk=NaN signifie "absent" et non "normal confirmé".
        if col in df_normal.columns:
            _nor_series = df_normal[col].ffill().bfill().fillna(0.0)
            vals_nor = _nor_series.values
        else:
            vals_nor = np.zeros(len(df_normal))

        det_gt  = (vals_gt  >= threshold).astype(int)
        det_nor = (vals_nor >= threshold).astype(int)

        tp = int(det_gt.sum())
        fn = int(n_gt - tp)
        fp = int(det_nor.sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[col_name] = {
            "tp": tp, "fp": fp, "fn": fn,
            "f1":        round(f1,             3),
            "precision": round(precision,      3),
            "recall":    round(recall,         3),
            "coverage_pct": round(100.0 * tp / n_gt if n_gt > 0 else 0.0, 1),
            "fpr_pct":      round(100.0 * fp / n_normal if n_normal > 0 else 0.0, 3),
            "max_val":      round(float(vals_gt.max()), 4),
        }

    return results

# ==============================================================================
# TABLE 4 et TABLE 12 — structure
# ==============================================================================

# LABEL_MAP : csv_prefix → (label_rapport, branch, note)
# csv_prefix = ce que compute_opinions_v2 produit comme préfixe de colonne
LABEL_MAP = {
    "P_bytes":                        ("bytes",                "Prophet", "Primary flood signal"),
    "P_packets":                      ("packets",              "Prophet", "Correlated with bytes; independent temporal signal"),
    "P_flows":                        ("flows",                "Prophet", "Connection count; sensitive to SYN floods"),
    "P_syn":                          ("syn",                  "Prophet", "SYN-specific; UDP flood: unaffected (expected)"),
    "P_entropy_src_ip":               ("entropy_src_ip",       "Prophet", "Single-source flood: entropy unchanged"),
    "P_entropy_src_port":             ("entropy_src_port",     "Prophet", "Not violated by this attack type"),
    "P_entropy_dst_port":             ("entropy_dst_port",     "Prophet", "—"),
    "P_avg_pkt_size":                 ("avg_pkt_size",         "Prophet", "Partial: packet size altered during burst phases"),
    "R_bytes_from_packets":           ("bytes ← packets",      "RANSAC",  "Linear relation preserved during flood"),
    "R_bytes_from_entropy_src_port":  ("bytes ← entropy_src_port", "RANSAC", "Volume/diversity correlation intact"),
}

rows_t4  = []  # Table 4
rows_t12 = []  # Table 12

print(f"{'─'*90}")
print(f"  {'Métrique':<30} {'F1 b_atk':>9}  {'F1 proj':>9}  {'TP':>5}/{n_gt}  {'FPR%':>6}  {'Stab':>4}  {'R²':>5}")
print(f"{'─'*90}")

for csv_prefix, (label, branch, note) in LABEL_MAP.items():
    # Lookup metadata : csv_prefix est la clé correcte (P_bytes, R_bytes_from_packets…)
    m = meta_by_csv_prefix.get(csv_prefix)
    if m is None:
        # Fallback : essayer les variantes _from_ / _to_
        for variant in [csv_prefix.replace("_from_", "_to_"),
                        csv_prefix.replace("_to_", "_from_")]:
            m = meta_by_csv_prefix.get(variant)
            if m: break

    r2   = m["r2"]              if m else 0.0
    stab = m["stability_class"] if m else "?"

    # Calcul F1 sur b_atk et proj_atk
    res = compute_individual_f1(df_atk, df_normal, csv_prefix, THRESHOLD)

    if res is None:
        # Colonne absente — essayer variante de nommage
        for variant in [csv_prefix.replace("_from_", "_to_"),
                        csv_prefix.replace("_to_", "_from_")]:
            res = compute_individual_f1(df_atk, df_normal, variant, THRESHOLD)
            if res is not None:
                break

    if res is None:
        f1_b_str   = "[not logged]"
        f1_p_str   = "[not logged]"
        f1_report  = "[not logged]"   # valeur à mettre dans le rapport
        tp_str     = "—"
        max_b      = "—"
        print(f"  {'⚠️  ' + csv_prefix:<30} {'[col. absente]':>50}")
    else:
        r_b = res.get("b_atk")
        r_p = res.get("proj_atk")

        f1_b_str  = f"{r_b['f1']:.3f}"  if r_b else "—"
        f1_p_str  = f"{r_p['f1']:.3f}"  if r_p else "—"

        # Colonne de référence pour le rapport : proj_atk (cohérent avec
        # la règle de décision du système D(T)=1 si P(Atk)≥δ)
        r_ref     = r_p if r_p else r_b
        f1_report = f"{r_ref['f1']:.3f}" if r_ref else "—"
        tp_n      = r_ref["tp"] if r_ref else 0
        tp_str    = f"{tp_n}/{n_gt}"
        max_b     = r_b["max_val"] if r_b else "—"
        fpr_ref   = r_ref["fpr_pct"] if r_ref else 0.0

        print(f"  {label:<30} {f1_b_str:>9}  {f1_p_str:>9}  {tp_str:>9}  {fpr_ref:>5.3f}%  "
              f"{stab:>4}  {r2:>5.3f}")

    # Table 4 — caractérisation préliminaire
    rows_t4.append({
        "Metric":          label,
        "Method":          branch,
        "F1 (prelim.)":    f1_report,
        "Stability":       stab,
        "R2":              round(r2, 3),
        "Notes":           note,
    })

    # Table 12 — per-metric sur le real DDoS
    rows_t12.append({
        "Metric":       label,
        "Branch":       branch,
        "R²":           round(r2, 3),
        "Class":        stab,
        "Ind. F1":      f1_report,
        "F1 (b_atk)":   f1_b_str,
        "F1 (proj)":    f1_p_str,
        "TP/Total":     tp_str,
        "Max b_atk":    max_b,
        "Notes":        note,
    })

print(f"{'─'*90}\n")

# ==============================================================================
# SAUVEGARDE CSV
# ==============================================================================
df_t4  = pd.DataFrame(rows_t4)
df_t12 = pd.DataFrame(rows_t12)

path_t4  = os.path.join(OUTPUT_DIR, "individual_f1_table4.csv")
path_t12 = os.path.join(OUTPUT_DIR, "individual_f1_table12.csv")

df_t4.to_csv(path_t4,   index=False)
df_t12.to_csv(path_t12, index=False)

print(f"-> Sauvegardé : {path_t4}")
print(f"-> Sauvegardé : {path_t12}")

# ==============================================================================
# FIGURE TABLE 12 — publication-ready
# ==============================================================================
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.axis("off")

col_labels = ["Metric", "Branch", "R²", "Class", "Ind. F1 (proj)", "TP/Total", "Notes"]
cell_data  = [[r["Metric"], r["Branch"], r["R²"], r["Class"],
               r["Ind. F1"], r["TP/Total"], r["Notes"]]
              for r in rows_t12]

col_widths = [0.17, 0.08, 0.05, 0.06, 0.10, 0.08, 0.29]

tbl = ax.table(
    cellText=cell_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    colWidths=col_widths,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.6)

# Formatage header
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2E4057")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Couleurs lignes alternées + surlignage
for i, row in enumerate(rows_t12):
    f1_v = row["Ind. F1"]
    try:
        f1_num = float(f1_v)
        detected = f1_num > 0
    except ValueError:
        detected = False
        f1_num   = -1

    base_color = "#F0F4F8" if i % 2 == 0 else "#FFFFFF"
    for j in range(len(col_labels)):
        cell = tbl[i + 1, j]
        cell.set_facecolor(base_color)
        cell.set_edgecolor("#CCCCCC")
        if j == 4:  # Ind. F1
            if f1_v == "[not logged]":
                cell.set_facecolor("#E8E8E8")
                cell.set_text_props(color="#888888", fontstyle="italic")
            elif detected:
                cell.set_facecolor("#C8E6C9")
                cell.set_text_props(fontweight="bold")
            elif f1_num == 0.0:
                cell.set_facecolor("#FFCDD2")
        if j == 1:
            if row["Branch"] == "Prophet":
                cell.set_facecolor("#E3F2FD")
            else:
                cell.set_facecolor("#FFF3E0")

ax.set_title(
    f"Table 12 — Per-metric characterisation on the real DDoS attack\n"
    f"δ = {THRESHOLD:.2f} on P(Atk) = proj_atk (before inter-metric fusion) | "
    f"Attack: {atk['name']} | {n_gt} GT windows",
    fontsize=10, fontweight="bold", pad=12
)

path_fig = os.path.join(OUTPUT_DIR, "individual_f1_table12.png")
fig.savefig(path_fig, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"-> Figure       : {path_fig}")

# ==============================================================================
# RÉSUMÉ CONSOLE
# ==============================================================================
print(f"\n{'='*88}")
print(f"  RÉSUMÉ — F1 individuels sur attaque réelle (δ={THRESHOLD} sur proj_atk)")
print(f"{'='*88}")
print(f"\n  {'Métrique':<30} {'F1 (proj)':>10}  {'F1 (b_atk)':>10}  {'TP/Total':>9}  {'Class':<5}  {'R²':>5}")
print(f"  {'─'*80}")
for r in rows_t12:
    print(f"  {r['Metric']:<30} {r['Ind. F1']:>10}  {r['F1 (b_atk)']:>10}  "
          f"{r['TP/Total']:>9}  {r['Class']:<5}  {r['R²']:>5}")

print(f"\n  NOTE : 'Ind. F1 (proj)' = valeur à reporter dans le rapport (cohérent")
print(f"         avec D(T)=1 si P(Atk)≥δ, cf. Section 6.2.1 du rapport).")
print(f"         'F1 (b_atk)' = mesure sur la belief mass brute (informatif).\n")

# Cas [not logged] restants
pending = [r for r in rows_t12 if r["Ind. F1"] == "[not logged]"]
if pending:
    print(f"  ⚠️  Colonnes absentes du CSV ({len(pending)}) :")
    for r in pending:
        print(f"     - {r['Metric']} ({r['Branch']})")
else:
    print(f"  ✅ Toutes les métriques calculées — aucune colonne manquante.")

print(f"\n  Fichiers produits :")
print(f"    {path_t4}")
print(f"    {path_t12}")
print(f"    {path_fig}\n")