"""
evaluate_real_ddos.py — Évaluation détaillée de l'attaque DDoS réelle
======================================================================
Analyse la vraie attaque UDP DDoS (12-13 novembre 2025) directement dans
le CSV de détection produit par compute_opinions_v2.py, sans injection.

Correction FPR : le masque "outside" exclut TOUTES les périodes d'attaque
du catalogue (synthétiques + réelle). Les fenêtres synthétiques injectées
ne sont pas du trafic normal et ne doivent pas entrer dans le calcul du FPR.

Sorties :
  - real_ddos_summary.csv     : métriques globales (coverage, TTD, FPR corrigé)
  - real_ddos_per_metric.csv  : évolution de b_atk par métrique feuille
  - real_ddos_timeline.png    : b_atk système + métriques feuilles
  - real_ddos_baserate.png    : évolution des base rates adaptatifs

Paramètres dans config.py :
  CONFIG["EVAL"]["RESULTS_DIR"]          : dossier résultats compute_opinions
  CONFIG["EVAL"]["RESULTS_CSV_NAME"]     : nom du CSV de détection
  CONFIG["EVAL"]["REAL_ATTACK_CATALOG"]  : définition de l'attaque réelle
  CONFIG["EVAL_REAL"]["THRESHOLD"]       : seuil opérationnel (b_atk)
  CONFIG["EVAL_REAL"]["CONTEXT_H"]       : heures de contexte avant/après
  CONFIG["EVAL_REAL"]["LEAF_METRICS"]    : métriques feuilles à tracer
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

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
    _HAS_PATHS_ERD = True
except ImportError:
    _HAS_PATHS_ERD = False

_EVAL      = CONFIG.get("EVAL", {})
_EVAL_REAL = CONFIG.get("EVAL_REAL", {})
VERSION_NAME = CONFIG.get("VERSION_NAME", "trained_models_v9_v6_v4s")

RESULTS_DIR  = _EVAL.get("RESULTS_DIR", f"../results/resultats_{VERSION_NAME}")
RESULTS_CSV  = os.path.join(RESULTS_DIR,
               _EVAL.get("RESULTS_CSV_NAME", "detection_results_INJECTED.csv"))
OUTPUT_DIR   = os.path.join(RESULTS_DIR, "evaluation_real_ddos")
THRESHOLD    = _EVAL_REAL.get("THRESHOLD",
               _get_decision_threshold(CONFIG, up_levels=2) if _HAS_PATHS_ERD
               else _EVAL.get("DECISION_THRESHOLD", 0.20))
CONTEXT_H    = _EVAL_REAL.get("CONTEXT_H", 4.0)
# Priorite: EVAL_REAL > EVAL > default
_default_det_col = (_get_detection_col(CONFIG, up_levels=2) if _HAS_PATHS_ERD
                    else _EVAL.get("COL_PROJ_ATK", "FINAL_SYSTEM_CBF_proj_atk"))
COL_DET = _EVAL_REAL.get("COL_DET", _EVAL.get("COL_DET", _default_det_col))
COL_BSUSP    = _EVAL.get("COL_BSUSP",    "FINAL_SYSTEM_CBF_b_susp")
COL_BSAFE    = _EVAL.get("COL_BSAFE",    "FINAL_SYSTEM_CBF_b_safe")
COL_PROJ_ATK = _EVAL.get("COL_PROJ_ATK", "FINAL_SYSTEM_CBF_proj_atk")
LEAF_METRICS = _EVAL_REAL.get("LEAF_METRICS", [
    "P_bytes", "P_packets", "P_avg_pkt_size",
    "P_entropy_src_ip", "P_entropy_src_port",
    "R_bytes_to_packets", "R_bytes_to_entropy_src_port",
])
_PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728",
            "#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]


def load_data():
    if not os.path.exists(RESULTS_CSV):
        print(f"❌ Fichier introuvable : {RESULTS_CSV}")
        sys.exit(1)
    df = pd.read_csv(RESULTS_CSV, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"-> CSV chargé : {len(df)} fenêtres | "
          f"{df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def get_full_catalog():
    """Catalogue complet (synthétique + réel) pour calculer le FPR correctement."""
    catalog = []
    try:
        from inject_at_evidence_level import ATTACK_CATALOG as _C
        catalog = list(_C)
    except ImportError:
        pass
    if _EVAL.get("INCLUDE_REAL_ATTACK", True):
        catalog += _EVAL.get("REAL_ATTACK_CATALOG", [])
    return catalog


def get_real_attack():
    attacks = _EVAL.get("REAL_ATTACK_CATALOG", [])
    if not attacks:
        print("❌ REAL_ATTACK_CATALOG vide dans config.py.")
        sys.exit(1)
    return attacks[0]


def compute_detection_metrics(df, atk, full_catalog, threshold):
    """
    FPR calculé sur les fenêtres hors de TOUTES les attaques du catalogue.
    Les fenêtres synthétiques injectées ne constituent pas du trafic normal.
    """
    t0 = pd.Timestamp(atk["start"])
    t1 = t0 + pd.Timedelta(hours=atk["duration_h"])
    gt_mask = (df["timestamp"] >= t0) & (df["timestamp"] < t1)

    # Masque trafic strictement normal : hors de toute période d'attaque
    out_mask = pd.Series(True, index=df.index)
    for a in full_catalog:
        a0 = pd.Timestamp(a["start"])
        a1 = a0 + pd.Timedelta(hours=a["duration_h"])
        out_mask &= ~((df["timestamp"] >= a0) & (df["timestamp"] < a1))

    b_gt  = df.loc[gt_mask,  COL_DET].fillna(0.0).values
    b_out = df.loc[out_mask, COL_DET].fillna(0.0).values
    ts_gt = df.loc[gt_mask, "timestamp"]
    n_gt  = len(b_gt)
    n_out = len(b_out)

    det_mask = df.loc[gt_mask, COL_DET].fillna(0.0).values >= threshold
    n_detected = int(det_mask.sum())
    detected   = n_detected > 0
    coverage   = n_detected / n_gt if n_gt > 0 else 0.0

    ttd_min = float("nan")
    if detected:
        t_first = ts_gt.iloc[int(np.argmax(det_mask))]
        ttd_min = max(0.0, (t_first - t0).total_seconds() / 60.0)

    fp  = int((b_out >= threshold).sum())
    fpr = 100.0 * fp / n_out if n_out > 0 else 0.0

    return {
        "name":             atk["name"],
        "start":            str(t0)[:19],
        "end":              str(t1)[:19],
        "duration_h":       atk["duration_h"],
        "intensity":        atk["intensity"],
        "threshold":        threshold,
        "n_gt_windows":     n_gt,
        "n_detected":       n_detected,
        "coverage_pct":     round(100.0 * coverage, 1),
        "detected":         detected,
        "ttd_minutes":      round(ttd_min, 1) if not np.isnan(ttd_min) else None,
        "max_b_atk":        round(float(b_gt.max()), 4) if n_gt > 0 else 0.0,
        "mean_b_atk":       round(float(b_gt.mean()), 4) if n_gt > 0 else 0.0,
        "fp_windows":       fp,
        "n_normal_windows": n_out,
        "fpr_pct":          round(fpr, 3),
    }


def extract_per_metric(df, atk):
    t0  = pd.Timestamp(atk["start"])
    t1  = t0 + pd.Timedelta(hours=atk["duration_h"])
    t_s = t0 - pd.Timedelta(hours=CONTEXT_H)
    t_e = t1 + pd.Timedelta(hours=CONTEXT_H)
    sub = df.loc[(df["timestamp"] >= t_s) & (df["timestamp"] <= t_e)].copy()
    rows = []
    for mk in LEAF_METRICS:
        for _, r in sub.iterrows():
            rows.append({
                "timestamp": r["timestamp"], "metric": mk,
                "b_atk": r.get(f"{mk}_b_atk", float("nan")),
                "a_atk": r.get(f"{mk}_a_atk", float("nan")),
            })
    return pd.DataFrame(rows), sub, t0, t1


def plot_timeline(sub, t0, t1, atk, output_dir):
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    dates = sub["timestamp"]
    for i, mk in enumerate(LEAF_METRICS):
        col = f"{mk}_b_atk"
        if col in sub.columns:
            ax1.step(dates, sub[col].fillna(0), where="post",
                     color=_PALETTE[i % len(_PALETTE)], lw=1.2, alpha=0.75,
                     label=mk.replace("_", " "))
    ax1.step(dates, sub[COL_DET].fillna(0), where="post",
             color="black", lw=2.5, label="System CBF (b_atk)", zorder=10)
    ax1.axhline(THRESHOLD, color="#888888", lw=1.5, linestyle="--",
                label=f"Threshold {THRESHOLD:.2f}")
    ax1.axvspan(t0, t1, alpha=0.08, color="#d62728", label="Attack window")
    ax1.axvline(t0, color="#d62728", lw=1.5, linestyle=":", alpha=0.8)
    ax1.axvline(t1, color="#d62728", lw=1.0, linestyle=":", alpha=0.5)
    ax1.set_ylabel("Belief Mass (b_atk)", fontsize=11)
    ax1.set_ylim(-0.03, 1.05)
    ax1.set_title(
        f"Real DDoS Attack — Detection Timeline\n"
        f"{atk['name']} | Start: {str(t0)[:19]} | Duration: {atk['duration_h']} h",
        fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", ncol=3, fontsize=7.5, framealpha=0.85)
    ax1.grid(True, alpha=0.25, linestyle="--")
    plt.setp(ax1.get_xticklabels(), visible=False)
    gt_series = pd.Series(0.0, index=sub.index)
    gt_mask = (sub["timestamp"] >= t0) & (sub["timestamp"] < t1)
    gt_series[gt_mask.values] = 1.0
    ax2.fill_between(dates, gt_series, step="post", color="#d62728", alpha=0.5)
    ax2.set_ylim(-0.15, 1.35)
    ax2.set_ylabel("Ground\nTruth", fontsize=9, labelpad=8)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Normal", "Attack"], fontsize=8.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax2.tick_params(axis="x", labelsize=8.5)
    ax2.grid(True, alpha=0.25, linestyle="--")
    fname = os.path.join(output_dir, "real_ddos_timeline.png")
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"   figure: {fname}")


def plot_baserate(sub, t0, t1, atk, output_dir):
    fig, ax = plt.subplots(figsize=(14, 5))
    dates = sub["timestamp"]
    for i, mk in enumerate(LEAF_METRICS):
        col = f"{mk}_a_atk"
        if col in sub.columns:
            ax.step(dates, sub[col].fillna(float("nan")), where="post",
                    color=_PALETTE[i % len(_PALETTE)], lw=1.3, alpha=0.8,
                    label=mk.replace("_", " "))
    ax.axvspan(t0, t1, alpha=0.08, color="#d62728", label="Attack window")
    ax.axvline(t0, color="#d62728", lw=1.5, linestyle=":", alpha=0.8)
    ax.axvline(t1, color="#d62728", lw=1.0, linestyle=":", alpha=0.5)
    ax.axhline(1/3, color="#aaaaaa", lw=1.0, linestyle="--", alpha=0.6,
               label="Neutral prior (1/3)")
    ax.set_ylabel("Adaptive Base Rate — a(Atk)", fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Adaptive Base Rate Evolution — Real DDoS\n"
                 "TransitionMemory calibration during attack",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", ncol=3, fontsize=7.5, framealpha=0.85)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(True, alpha=0.25, linestyle="--")
    fname = os.path.join(output_dir, "real_ddos_baserate.png")
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"   figure: {fname}")


def main():
    print(f"\n{'='*60}")
    print("  evaluate_real_ddos.py — Real Attack Evaluation")
    print(f"{'='*60}\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df           = load_data()
    atk          = get_real_attack()
    full_catalog = get_full_catalog()

    print(f"-> Attaque       : {atk['name']}")
    print(f"   Début         : {atk['start']}")
    print(f"   Durée         : {atk['duration_h']} h")
    print(f"   Seuil         : b_atk >= {THRESHOLD}")
    print(f"   Catalogue FPR : {len(full_catalog)} périodes exclues du trafic normal\n")

    metrics = compute_detection_metrics(df, atk, full_catalog, THRESHOLD)
    pd.DataFrame([metrics]).to_csv(
        os.path.join(OUTPUT_DIR, "real_ddos_summary.csv"), index=False)
    print("-> Métriques :")
    for k, v in metrics.items():
        print(f"   {k:<22} : {v}")

    df_per_metric, sub, t0, t1 = extract_per_metric(df, atk)
    df_per_metric.to_csv(
        os.path.join(OUTPUT_DIR, "real_ddos_per_metric.csv"), index=False)

    print("\n-> Génération des graphiques...")
    plot_timeline(sub, t0, t1, atk, OUTPUT_DIR)
    plot_baserate(sub, t0, t1, atk, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print(f"  RÉSULTAT — {atk['name']}")
    print(f"{'='*60}")
    print(f"  Statut         : {'DÉTECTÉE' if metrics['detected'] else 'MANQUÉE'}")
    print(f"  Coverage       : {metrics['coverage_pct']}%")
    print(f"  TTD            : {metrics['ttd_minutes']} min")
    print(f"  Max b_atk      : {metrics['max_b_atk']}")
    print(f"  FPR (corrigé)  : {metrics['fpr_pct']}%  ({metrics['n_normal_windows']} fenêtres normales)")
    print(f"\n  Résultats : {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()