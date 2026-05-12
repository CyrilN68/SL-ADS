"""
verify_outage_episodes.py
=========================
Vérifie statistiquement les épisodes FP/INCONNU classifiés NETWORK_OUTAGE
en analysant TOUTES les métriques disponibles (17 métriques v10).

Pour chaque épisode suspect, calcule :
  - z-score par métrique vs baseline train
  - direction du signal (chute vs excès)
  - cohérence avec signature OUTAGE (chute globale) vs autre anomalie

Outputs :
  verify_outage_all_metrics.png   — heatmap z-scores par épisode × métrique
  verify_outage_stats.csv         — tableau complet des statistiques
  verify_outage_report.txt        — rapport texte avec conclusion par épisode

Usage :
  python verify_outage_episodes.py
  (depuis le répertoire du projet, à côté de config.py)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys

try:
    from config import CONFIG
except ImportError:
    print("❌ config.py introuvable.")
    sys.exit(1)

# ==============================================================================
# PARAMÈTRES
# ==============================================================================
RAW_DATA_PATH = os.path.join(
    "../../results",
    f"resultats_{CONFIG['VERSION_NAME']}",
    f"raw_data_{CONFIG['VERSION_NAME']}.csv"
)
SPLIT_DATE   = pd.Timestamp(CONFIG['split_date'])
RESAMPLE_WIN = "5min"

# Épisodes FP/INCONNU OUTAGE à vérifier (tirés de l'audit)
EPISODES = [
    {"name": "Nov17_transit",    "start": "2025-11-17 12:30:00", "end": "2025-11-17 12:45:00"},
    {"name": "Dec18_post_ICMP",  "start": "2025-12-18 13:30:00", "end": "2025-12-18 15:35:00"},
    {"name": "Dec19_morning",    "start": "2025-12-19 10:35:00", "end": "2025-12-19 13:30:00"},
    {"name": "Dec19_midday1",    "start": "2025-12-19 13:55:00", "end": "2025-12-19 14:35:00"},
    {"name": "Dec19_midday2",    "start": "2025-12-19 15:00:00", "end": "2025-12-19 15:40:00"},
    {"name": "Dec22_morning",    "start": "2025-12-22 09:50:00", "end": "2025-12-22 16:30:00"},
    {"name": "Dec23_morning",    "start": "2025-12-23 09:45:00", "end": "2025-12-23 16:15:00"},
    # Épisode de référence confirmé (coupure réelle Dec 16–17)
    {"name": "Dec16_OUTAGE_REF", "start": "2025-12-16 12:30:00", "end": "2025-12-16 18:00:00"},
]

# Toutes les métriques brutes à analyser
METRICS = [
    'bytes', 'packets', 'flows', 'syn',
    'icmp', 'udp', 'tcp', 'fin',
    'entropy_src_ip', 'entropy_src_port', 'entropy_dst_port', 'avg_pkt_size',
]

# Fenêtre de baseline : 3 semaines en train, représentative des patterns horaires
# On prend les 7 derniers jours du train pour avoir les stats les plus récentes
BASELINE_START = pd.Timestamp("2025-11-10 00:00:00")   # début du test
BASELINE_END   = pd.Timestamp("2025-11-30 00:00:00")   # avant toute injection Nov/Dec

OUTPUT_DIR = os.path.join(
    "../../results",
    f"resultats_{CONFIG['VERSION_NAME']}"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_PNG    = os.path.join(OUTPUT_DIR, "verify_outage_all_metrics.png")
OUT_CSV    = os.path.join(OUTPUT_DIR, "verify_outage_stats.csv")
OUT_REPORT = os.path.join(OUTPUT_DIR, "verify_outage_report.txt")

print(f"-> Chargement résidus Prophet : {RAW_DATA_PATH}")
df_raw = pd.read_csv(RAW_DATA_PATH, parse_dates=['timestamp'])
df_raw['residual'] = df_raw['real'] - df_raw['pred']

# Pivoter : une colonne par métrique, valeur = résidu signé
df_5m = df_raw.pivot_table(
    index='timestamp', columns='metric_key',
    values='residual', aggfunc='mean'
).reset_index().rename_axis(None, axis=1)

# Extraire les noms de métriques disponibles depuis les colonnes
# Les clés sont du type "prophet_bytes", "reconst_bytes_from_packets" etc.
# On garde uniquement les prophet_ pour les métriques de base
METRICS = [c for c in df_5m.columns if c.startswith('prophet_')]
print(f"   Métriques résidus disponibles : {METRICS}")

# ==============================================================================
# STATISTIQUES BASELINE (train, sans les derniers jours si contamination)
# ==============================================================================
print(f"-> Calcul baseline ({BASELINE_START.date()} → {BASELINE_END.date()})...")
base = df_5m[(df_5m['timestamp'] >= BASELINE_START) & (df_5m['timestamp'] <= BASELINE_END)]

stats = {}
for m in METRICS:
    vals = base[m].dropna()
    stats[m] = {
        'mean': float(vals.mean()),  # ≈ 0 pour résidus Prophet bien calibrés
        'std':   float(vals.std()) + 1e-9,
        'q99':   float(vals.quantile(0.99)),
        'q999':  float(vals.quantile(0.999)),
        'q01':   float(vals.quantile(0.01)),   # queue basse (chute)
        'q001':  float(vals.quantile(0.001)),
        'iqr':   float(vals.quantile(0.75) - vals.quantile(0.25)),
        'q25':   float(vals.quantile(0.25)),
        'q75':   float(vals.quantile(0.75)),
    }

# ==============================================================================
# ANALYSE PAR ÉPISODE × MÉTRIQUE
# ==============================================================================
print("-> Analyse des épisodes...")

def analyze_episode(ep, df_5m, stats):
    """Retourne un dict {métrique: {zscore, direction, anomaly_type, ...}}"""
    t_start = pd.Timestamp(ep['start'])
    t_end   = pd.Timestamp(ep['end'])

    ep_df = df_5m[(df_5m['timestamp'] >= t_start) & (df_5m['timestamp'] <= t_end)]
    n_win = len(ep_df)

    results = {'n_windows': n_win, 'duration_min': (t_end - t_start).total_seconds() / 60}
    metric_results = {}

    for m in METRICS:
        s = stats[m]
        vals = ep_df[m].dropna()
        if len(vals) == 0:
            metric_results[m] = {'zscore': 0.0, 'direction': 'unknown', 'ratio': 1.0}
            continue

        ep_mean = float(vals.mean())  # résidu moyen (peut être négatif)
        zscore = ep_mean / s['std']  # z-score du résidu (mean résidu train ≈ 0 par construction)
        ratio = ep_mean / s['mean'] if abs(s['mean']) > 1e-6 else float('nan')

        # Direction
        if zscore < -2.5:
            direction = 'CHUTE'
        elif zscore > 2.5:
            direction = 'HAUSSE'
        else:
            direction = 'normal'

        # % fenêtres sous Q1% (chute extrême) et au-dessus Q99% (excès)
        pct_low  = float((vals < s['q01']).mean() * 100)
        pct_high = float((vals > s['q99']).mean() * 100)

        metric_results[m] = {
            'ep_mean':   ep_mean,
            'train_mean': s['mean'],
            'zscore':    zscore,
            'ratio':     ratio,
            'direction': direction,
            'pct_low':   pct_low,
            'pct_high':  pct_high,
        }

    results['metrics'] = metric_results

    # Score global outage : combien de métriques volumétriques en chute ?
    vol_metrics = ['prophet_bytes', 'prophet_packets', 'prophet_flows', 'prophet_tcp']
    proto_metrics = ['prophet_icmp', 'prophet_udp', 'prophet_syn', 'prophet_fin']
    entropy_metrics = ['prophet_entropy_src_ip', 'prophet_entropy_src_port',
                       'prophet_entropy_dst_port', 'prophet_avg_pkt_size']

    n_vol_chute   = sum(1 for m in vol_metrics  if metric_results.get(m, {}).get('direction') == 'CHUTE')
    n_proto_chute = sum(1 for m in proto_metrics if metric_results.get(m, {}).get('direction') == 'CHUTE')
    n_vol_hausse  = sum(1 for m in vol_metrics  if metric_results.get(m, {}).get('direction') == 'HAUSSE')

    # Classification heuristique
    if n_vol_chute >= 3 and n_proto_chute >= 2:
        classification = "OUTAGE_PROBABLE"
    elif n_vol_chute >= 2:
        classification = "OUTAGE_POSSIBLE"
    elif n_vol_hausse >= 3:
        classification = "FLOOD_PROBABLE"
    else:
        classification = "AMBIGUE"

    results['n_vol_chute']    = n_vol_chute
    results['n_proto_chute']  = n_proto_chute
    results['n_vol_hausse']   = n_vol_hausse
    results['classification'] = classification

    return results

# Analyser tous les épisodes
all_results = {}
for ep in EPISODES:
    print(f"   [{ep['name']}] {ep['start']} → {ep['end']}")
    all_results[ep['name']] = analyze_episode(ep, df_5m, stats)

# ==============================================================================
# HEATMAP Z-SCORES
# ==============================================================================
print("-> Génération heatmap...")

ep_names   = [ep['name'] for ep in EPISODES]
zscore_mat = np.zeros((len(METRICS), len(ep_names)))
for j, ep in enumerate(EPISODES):
    for i, m in enumerate(METRICS):
        zscore_mat[i, j] = all_results[ep['name']]['metrics'].get(m, {}).get('zscore', 0.0)

fig, ax = plt.subplots(figsize=(max(12, len(ep_names) * 1.5), max(8, len(METRICS) * 0.6)))

# Colormap symétrique (bleu = chute, rouge = hausse)
vmax = max(5.0, float(np.abs(zscore_mat).max()))
im = ax.imshow(zscore_mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
plt.colorbar(im, ax=ax, label='Z-score vs baseline train')

ax.set_xticks(range(len(ep_names)))
ax.set_xticklabels(ep_names, rotation=30, ha='right', fontsize=9)
ax.set_yticks(range(len(METRICS)))
ax.set_yticklabels(METRICS, fontsize=9)

# Annotations
for i in range(len(METRICS)):
    for j in range(len(ep_names)):
        z = zscore_mat[i, j]
        color = 'white' if abs(z) > vmax * 0.6 else 'black'
        ax.text(j, i, f'{z:.1f}', ha='center', va='center',
                fontsize=7.5, color=color, fontweight='bold' if abs(z) > 3 else 'normal')

# Séparateur épisode de référence (dernier)
ax.axvline(len(ep_names) - 1.5, color='yellow', lw=2, ls='--')
ax.text(len(ep_names) - 0.5, -0.7, 'REF', color='yellow', fontsize=8, ha='center')

ax.set_title("Z-scores par épisode FP/INCONNU OUTAGE vs baseline train\n"
             "Bleu=chute | Rouge=excès | Seuil ±2.5σ | Col. REF = coupure Dec16 confirmée",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"   📊 Heatmap : {OUT_PNG}")

# ==============================================================================
# TABLEAU CSV
# ==============================================================================
rows = []
for ep in EPISODES:
    r = all_results[ep['name']]
    base_row = {
        'episode':        ep['name'],
        'start':          ep['start'],
        'end':            ep['end'],
        'duration_min':   r['duration_min'],
        'n_windows':      r['n_windows'],
        'classification': r['classification'],
        'n_vol_chute':    r['n_vol_chute'],
        'n_proto_chute':  r['n_proto_chute'],
        'n_vol_hausse':   r['n_vol_hausse'],
    }
    for m in METRICS:
        mr = r['metrics'].get(m, {})
        base_row[f'z_{m}']    = round(mr.get('zscore', 0.0), 2)
        base_row[f'dir_{m}']  = mr.get('direction', '')
        base_row[f'ratio_{m}'] = round(mr.get('ratio', 1.0), 2)
    rows.append(base_row)

df_stats = pd.DataFrame(rows)
df_stats.to_csv(OUT_CSV, index=False)
print(f"   📄 Stats CSV : {OUT_CSV}")

# ==============================================================================
# RAPPORT TEXTE
# ==============================================================================
# Colonnes pour l'affichage compacte
vol_m    = ['prophet_bytes', 'prophet_packets', 'prophet_flows', 'prophet_tcp']
proto_m  = ['prophet_icmp', 'prophet_udp', 'prophet_syn', 'prophet_fin']
entr_m   = ['prophet_entropy_src_ip', 'prophet_entropy_src_port', 'prophet_avg_pkt_size']

with open(OUT_REPORT, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("VÉRIFICATION ÉPISODES OUTAGE — Toutes métriques (17 v10)\n")
    f.write(f"Baseline : {BASELINE_START.date()} → {BASELINE_END.date()}\n")
    f.write("=" * 70 + "\n\n")
    f.write("Seuil anomalie : |z| > 2.5 (≈ top/bottom 1% de la distribution normale)\n")
    f.write("OUTAGE_PROBABLE : ≥3 métriques vol. en chute ET ≥2 proto en chute\n")
    f.write("Ref : Barford et al. (2002) IMW — classification outage vs flash crowd\n\n")

    for ep in EPISODES:
        r   = all_results[ep['name']]
        mtr = r['metrics']
        ref = " [RÉFÉRENCE CONFIRMÉE]" if ep['name'] == "Dec16_OUTAGE_REF" else ""

        f.write(f"{'─' * 70}\n")
        f.write(f"Épisode  : {ep['name']}{ref}\n")
        f.write(f"Période  : {ep['start']} → {ep['end']}  ({r['duration_min']:.0f} min)\n")
        f.write(f"Fenêtres : {r['n_windows']}\n")
        f.write(f"\n  MÉTRIQUES VOLUMÉTRIQUES :\n")
        for m in vol_m:
            mr = mtr.get(m, {})
            z  = mr.get('zscore', 0.0)
            d  = mr.get('direction', '?')
            ra = mr.get('ratio', 1.0)
            flag = " ◄" if abs(z) > 2.5 else ""
            f.write(f"    {m:<22} z={z:+.2f}  dir={d:<8} ratio=×{ra:.2f}{flag}\n")

        f.write(f"\n  MÉTRIQUES PROTOCOLAIRES :\n")
        for m in proto_m:
            mr = mtr.get(m, {})
            z  = mr.get('zscore', 0.0)
            d  = mr.get('direction', '?')
            ra = mr.get('ratio', 1.0)
            flag = " ◄" if abs(z) > 2.5 else ""
            f.write(f"    {m:<22} z={z:+.2f}  dir={d:<8} ratio=×{ra:.2f}{flag}\n")

        f.write(f"\n  ENTROPIES / STRUCTURE :\n")
        for m in entr_m:
            mr = mtr.get(m, {})
            z  = mr.get('zscore', 0.0)
            d  = mr.get('direction', '?')
            ra = mr.get('ratio', 1.0)
            flag = " ◄" if abs(z) > 2.5 else ""
            f.write(f"    {m:<22} z={z:+.2f}  dir={d:<8} ratio=×{ra:.2f}{flag}\n")

        f.write(f"\n  SCORE : vol_chute={r['n_vol_chute']}/4  proto_chute={r['n_proto_chute']}/4  "
                f"vol_hausse={r['n_vol_hausse']}/4\n")
        f.write(f"  ➤ CLASSIFICATION : {r['classification']}\n\n")

    f.write("=" * 70 + "\n")
    f.write("TABLEAU COMPARATIF RÉSUMÉ\n")
    f.write("=" * 70 + "\n")
    f.write(f"  {'Episode':<22} {'Class.':<22} {'vol↓':>5} {'proto↓':>7} {'vol↑':>5}\n")
    f.write(f"  {'-' * 62}\n")
    for ep in EPISODES:
        r = all_results[ep['name']]
        ref = " ★" if ep['name'] == "Dec16_OUTAGE_REF" else ""
        f.write(f"  {ep['name']:<22} {r['classification']:<22} "
                f"{r['n_vol_chute']:>5} {r['n_proto_chute']:>7} {r['n_vol_hausse']:>5}{ref}\n")

print(f"   📄 Rapport  : {OUT_REPORT}")

# ==============================================================================
# RÉSUMÉ CONSOLE
# ==============================================================================
print(f"\n{'=' * 70}")
print(f"  RÉSUMÉ — Classification des épisodes FP/INCONNU OUTAGE")
print(f"{'=' * 70}")
print(f"  {'Episode':<22} {'Classification':<22} {'vol↓':>5} {'proto↓':>7}")
print(f"  {'-' * 58}")
for ep in EPISODES:
    r   = all_results[ep['name']]
    ref = " ★" if ep['name'] == "Dec16_OUTAGE_REF" else ""
    print(f"  {ep['name']:<22} {r['classification']:<22} "
          f"{r['n_vol_chute']:>5} {r['n_proto_chute']:>7}{ref}")
print(f"\n  ★ = épisode de référence (coupure Dec 16 confirmée par analyse brute)")
print(f"  OUTAGE_PROBABLE : ≥3 métriques vol en chute + ≥2 proto en chute")
print(f"\n✅ Vérification terminée.")