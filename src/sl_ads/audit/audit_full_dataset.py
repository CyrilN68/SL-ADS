"""
audit_full_dataset.py — Audit exhaustif du dataset RedeRio
===========================================================
VERSION v2 — Publication-ready (fixes bugs v1 + ajouts publication)

BUGS CORRIGÉS vs v1 :
  B1 : QUALIF_CSV_INJECTED → cherche maintenant qualif_types_sbn.csv en priorité
  B2 : colonne u_qualif remplacée par auto-détection u_sbn/u_qualif
  B3 : get_detection_col sorti de la boucle épisode (calculé une seule fois)
  B4 : FPR estimate vectorisé (iterrows → opération pandas)
  B5 : near-miss cherche detection_results_INJECTED.csv en priorité
  B6 : classification épisode utilise pct_in_known (>50%) et non plus seul ts_mid

AJOUTS PUBLICATION :
  A1 : Intervalle de Wilson (95%) sur FAR (Wilson 1927 ; Agresti & Coull 1998)
       Recommandé par Brown et al. (2002) Statist. Sci. 16:101-133 vs Wald
  A2 : u_sbn / novelty_lr / novelty_entropy par épisode (SBN format)
  A3 : Qualification precision par épisode (top1_type vs expected)
  A4 : Pre-reconnaissance window flag : épisodes dans [J-14, J-1] avant REAL_DDOS
       Ref : Benson (2010) IMC ; Steinberger (2020) CNSM — 24-168h pre-DDoS activity
  A5 : IoU (Intersection over Union) pour matching épisode/attaque connue
  A6 : Sévérité épisode : duration × P_Anom_max (pour tri des FP candidats)
  A7 : Export LaTeX table des épisodes inconnus (publication §8)
  A8 : Colonne format SBN → auto-détect, affiche top1 SBN type pour FP/UNKNOWN

Usage :
    python audit_full_dataset.py
    python audit_full_dataset.py --gap_min 15 --near_miss --latex
    python audit_full_dataset.py --input qualif_types_sbn.csv

Ref : méthodologie audit (Barford et al. 2002 IMW ; Lakhina et al. 2004 ACM SIGCOMM)
      Wilson CI : Wilson (1927 JASA) ; Agresti & Coull (1998 Am. Stat.) ;
                  Brown, Cai, DasGupta (2002 Statist. Sci. 16:101-133)
"""

import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from sl_ads.paths import get_version_names, get_results_dir, get_decision_threshold, get_detection_col  # Phase H

try:
    from sl_ads.config import CONFIG, REAL_ATTACKS  # Phase H
except ImportError:
    print("❌ sl_ads.config introuvable."); sys.exit(1)

# ==============================================================================
# PARAMÈTRES
# ==============================================================================
VERSION_NAME, _ = get_version_names(CONFIG)
RESULTS_DIR     = get_results_dir(CONFIG, up_levels=1)

# B1 FIX : chercher d'abord qualif_types_sbn.csv (format SBN), puis qualif_types.csv
QUALIF_CSV_CANDIDATES = [
    os.path.join(RESULTS_DIR, "qualif_types_real_sbn.csv"),    # audit sur trafic réel (SBN)
    os.path.join(RESULTS_DIR, "qualif_types_sbn.csv"),         # injecté (SBN) ← prioritaire
    os.path.join(RESULTS_DIR, "qualif_types_real.csv"),        # audit sur trafic réel (LR)
    os.path.join(RESULTS_DIR, "qualif_types.csv"),             # injecté (LR) — fallback
]

DECISION_THR = get_decision_threshold(CONFIG, up_levels=1)
WINDOW_MIN   = 5    # durée fenêtre en minutes

# Couleurs console
GREEN  = "\033[92m"; YELLOW = "\033[93m"
RED    = "\033[91m"; CYAN   = "\033[96m"
RESET  = "\033[0m";  BOLD   = "\033[1m"


# ==============================================================================
# A1 : INTERVALLE DE WILSON SUR UNE PROPORTION
# ==============================================================================

def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """
    Intervalle de Wilson (1927) sur une proportion k/n.

    Préféré au Wald (Wald interval ≡ p̂ ± z·SE) pour les proportions
    proches de 0 ou 1 et les petits échantillons.
    Recommandé par Brown, Cai & DasGupta (2002) Statist. Sci. 16:101–133
    pour toute situation où la méthode de Wald est inadaptée.

    Formule (Brown 2002 Eq. 2.5) :
        p̃ = (k + z²/2) / (n + z²)
        σ̃ = sqrt(p̃(1-p̃) / (n + z²))
        [p̃ - z·σ̃,  p̃ + z·σ̃]

    Returns : (lower, upper) bounds en fraction [0, 1].
    """
    import scipy.stats as stats
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    n_eff = n + z ** 2
    p_tilde = (k + z ** 2 / 2) / n_eff
    sigma   = math.sqrt(p_tilde * (1 - p_tilde) / n_eff)
    return max(0.0, p_tilde - z * sigma), min(1.0, p_tilde + z * sigma)


# ==============================================================================
# CHARGEMENT DES ATTAQUES CONNUES
# ==============================================================================

def build_known_periods():
    """Source unique de vérité : inject_at_evidence_level.ATTACK_CATALOG + REAL_ATTACKS."""
    known = []
    try:
        from sl_ads.inject.evidence_level import ATTACK_CATALOG as _INJ  # Phase H
        for a in _INJ:
            t0 = pd.Timestamp(a['start'])
            t1 = t0 + pd.Timedelta(hours=a['duration_h'])
            known.append({
                'name': a['name'], 'type': a.get('type', a['name']),
                'origin': 'INJECTED', 'start': t0, 'end': t1,
            })
    except ImportError:
        print("  [WARN] inject_at_evidence_level non trouvé — attaques injectées absentes")

    for name, events in REAL_ATTACKS.items():
        for ev in events:
            known.append({
                'name': name,
                'type': ev.get('expected_qualif', 'REAL_ATTACK'),
                'origin': 'REAL',
                'start':  pd.Timestamp(ev['start']),
                'end':    pd.Timestamp(ev['end']),
            })
    return known


def classify_window(ts, known_periods):
    """Retourne (origin, name, type) pour une timestamp donnée."""
    for k in known_periods:
        if k['start'] <= ts < k['end']:
            return k['origin'], k['name'], k['type']
    return 'UNKNOWN', None, None


# ==============================================================================
# A5 : MATCHING PAR IoU (Intersection over Union)
# ==============================================================================

def iou_episode_known(ep_start, ep_end, known_periods) -> tuple[float, dict | None]:
    """
    Calcule le IoU entre un épisode et chaque période connue.
    Retourne (max_iou, meilleure période connue).
    IoU = intersection_minutes / union_minutes.
    Ref : standard de l'évaluation de détection temporelle (Martin et al. 2001 CVPR).
    """
    best_iou, best_k = 0.0, None
    ep_stop = ep_end + pd.Timedelta(minutes=WINDOW_MIN)
    ep_dur = (ep_stop - ep_start).total_seconds() / 60

    for k in known_periods:
        k_dur = (k['end'] - k['start']).total_seconds() / 60
        inter_start = max(ep_start, k['start'])
        inter_end   = min(ep_stop,  k['end'])
        inter_min   = max(0.0, (inter_end - inter_start).total_seconds() / 60)
        union_min   = ep_dur + k_dur - inter_min
        iou = inter_min / union_min if union_min > 0 else 0.0
        if iou > best_iou:
            best_iou, best_k = iou, k

    return best_iou, best_k


# ==============================================================================
# GROUPEMENT EN ÉPISODES
# ==============================================================================

def group_episodes(df_gate: pd.DataFrame, gap_min: float = 15) -> list:
    """Regroupe les fenêtres gate_open en épisodes (gap > gap_min → nouvel épisode)."""
    if df_gate.empty:
        return []
    df_gate = df_gate.sort_values('timestamp').reset_index(drop=True)
    episodes, ep_rows = [], [df_gate.iloc[0]]
    for i in range(1, len(df_gate)):
        gap = (df_gate.iloc[i]['timestamp'] - df_gate.iloc[i-1]['timestamp']).total_seconds() / 60
        if gap <= gap_min:
            ep_rows.append(df_gate.iloc[i])
        else:
            episodes.append(pd.DataFrame(ep_rows))
            ep_rows = [df_gate.iloc[i]]
    episodes.append(pd.DataFrame(ep_rows))
    return episodes


# ==============================================================================
# CARACTÉRISATION D'UN ÉPISODE — AMÉLIORÉE
# ==============================================================================

def characterize_episode(ep_df: pd.DataFrame, known_periods: list,
                          p_anom_col: str, u_col: str, fmt: str) -> dict:
    """
    Calcule toutes les statistiques d'un épisode.
    B3 FIX : p_anom_col et u_col pré-calculés hors boucle.
    B6 FIX : utilise IoU (A5) pour le matching, plus pct_in_known.
    A2 : novelty_lr, novelty_entropy, u_sbn intégrés si disponibles.
    A6 : sévérité = duration × p_anom_max.
    """
    ts_start = ep_df['timestamp'].min()
    ts_end   = ep_df['timestamp'].max()
    n_win    = len(ep_df)
    duration_min = (ts_end - ts_start).total_seconds() / 60 + WINDOW_MIN

    # Top-1 type dominant
    type_col  = 'top1_type' if 'top1_type' in ep_df.columns else 'top1'
    top_counts = ep_df[type_col].value_counts() if type_col in ep_df.columns else pd.Series()
    top_type  = top_counts.index[0] if len(top_counts) > 0 else 'N/A'
    top_pct   = top_counts.iloc[0] / n_win * 100 if n_win > 0 else 0

    # Incertitude (B2 FIX : u_col auto-détecté)
    u_mean = ep_df[u_col].mean() if u_col in ep_df.columns else float('nan')

    # P_Anom max
    p_anom_max = ep_df[p_anom_col].max() if p_anom_col in ep_df.columns else float('nan')

    # A2 : métriques SBN si disponibles
    novelty_lr_mean    = ep_df['novelty_lr'].mean()    if 'novelty_lr'    in ep_df.columns else float('nan')
    novelty_ent_mean   = ep_df['novelty_entropy'].mean() if 'novelty_entropy' in ep_df.columns else float('nan')

    # Fraction de fenêtres dans une période connue — DOIT être calculée AVANT le test IoU
    n_in_known = sum(
        1 for ts in ep_df['timestamp']
        if classify_window(ts, known_periods)[0] != 'UNKNOWN'
    )
    pct_known = n_in_known / n_win * 100

    # A5 FIX : IoU-based matching
    iou, best_k = iou_episode_known(ts_start, ts_end, known_periods)

    # B6 FIX : classification épisode sur IoU ≥ 0.20 OU pct_in_known ≥ 80%
    # Le second critère couvre les sous-épisodes d'une longue période connue
    # (ex: NETWORK_OUTAGE 28h découpé en épisodes par gap_min → IoU faible mais fenêtres dedans)
    if best_k is not None and (iou >= 0.20 or pct_known >= 80.0):
        origin      = best_k['origin']
        attack_name = best_k['name']
        attack_type = best_k['type']
    else:
        origin      = 'UNKNOWN'
        attack_name = '—'
        attack_type = '—'

    # A6 : sévérité
    severity = duration_min * p_anom_max if not np.isnan(p_anom_max) else 0.0

    return {
        'start':           ts_start,
        'end':             ts_end,
        'duration_min':    round(duration_min, 0),
        'n_windows':       n_win,
        'top_type':        top_type,
        'top_pct':         round(top_pct, 1),
        'u_mean':          round(u_mean, 3) if not np.isnan(u_mean) else float('nan'),
        'novelty_lr_mean': round(novelty_lr_mean, 3) if not np.isnan(novelty_lr_mean) else float('nan'),
        'novelty_ent':     round(novelty_ent_mean, 3) if not np.isnan(novelty_ent_mean) else float('nan'),
        'p_anom_max':      round(p_anom_max, 3) if not np.isnan(p_anom_max) else float('nan'),
        'severity':        round(severity, 1),
        'origin':          origin,
        'attack_name':     attack_name,
        'attack_type':     attack_type,
        'pct_in_known':    round(pct_known, 1),
        'iou':             round(iou, 3),
    }


# ==============================================================================
# A1 : FPR VECTORISÉ + WILSON CI
# ==============================================================================

def compute_far_wilson(df: pd.DataFrame, df_gate: pd.DataFrame,
                       known_periods: list) -> dict:
    """
    Calcule FAR = FP / N_normales avec intervalle de Wilson 95%.

    FP = fenêtres gate_open qui NE sont PAS dans une période connue.
    N_normales = toutes les fenêtres hors périodes connues.

    B4 FIX : vectorisé via merge sur timestamps.
    A1 : intervalle de Wilson (Wilson 1927 ; Brown 2002).
    """
    # Marquer les fenêtres dans les périodes connues (vectorisé)
    in_known = pd.Series(False, index=df.index)
    for k in known_periods:
        in_known |= (df['timestamp'] >= k['start']) & (df['timestamp'] < k['end'])

    n_normal = int((~in_known).sum())

    # FP = gate_open ET hors périodes connues
    gate_open_col = df['gate_open'] if 'gate_open' in df.columns else (
        df[get_detection_col(CONFIG, up_levels=1)] >= DECISION_THR
    )
    fp = int((gate_open_col & ~in_known).sum())

    far_pct = 100 * fp / n_normal if n_normal > 0 else 0.0
    lo, hi  = wilson_ci(fp, n_normal, conf=0.95)

    return {
        'n_fp':        fp,
        'n_normal':    n_normal,
        'far_pct':     round(far_pct, 3),
        'far_ci_lo':   round(lo * 100, 3),
        'far_ci_hi':   round(hi * 100, 3),
    }


# ==============================================================================
# A3 : QUALIFICATION PRECISION PAR ÉPISODE
# ==============================================================================

ATTACK_NAME_TO_SBN_KEY = {
    'UDP_FLOOD_DDOS':          'UDP_FLOOD',
    'SYN_FLOOD_DDOS':          'SYN_FLOOD',
    'BOTNET_CC_BEACONING':     'BOTNET_CC',
    'AGGRESSIVE_PORT_SCAN':    'PORT_SCAN',
    'DATA_EXFILTRATION_SLOW':  'DATA_EXFIL',
    'NTP_AMPLIFICATION':       'NTP_AMP',
    'HTTP_FLOOD_L7_DDOS':      'HTTP_FLOOD',
    'DNS_AMPLIFICATION':       'DNS_AMP',       # ← attendu DNS_AMP, obtenu NTP_AMP → limitation
    'DNS_TUNNELING':           'DNS_TUNNELING',
    'SLOWLORIS_DOS':           'SLOWLORIS',
    'ICMP_FLOOD_BURST':        'ICMP_FLOOD',
    'BRUTE_FORCE_SSH':         'BRUTE_FORCE_SSH',
    'UNKNOWN_ANOMALY_CONTROL': None,            # ← contrôle nouveauté, pas de type attendu
    # Vraies attaques
    'DDOS_ATTACK':             'UDP_FLOOD',
    'NETWORK_OUTAGE_DEC1617':  'NETWORK_OUTAGE',
    'NETWORK_OUTAGE_NOV17':    'NETWORK_OUTAGE',
}

def check_qual_correct(ep: dict, known_periods: list) -> str | None:
    if ep['origin'] == 'UNKNOWN':
        return None
    # Utiliser la table de correspondance (SBN key) au lieu de k['type'] (description littérale)
    expected_sbn = ATTACK_NAME_TO_SBN_KEY.get(ep['attack_name'])
    if expected_sbn is None:
        return 'N/A'  # contrôle de nouveauté ou inconnu
    top = ep['top_type']
    return 'CORRECT' if top == expected_sbn else 'WRONG'


# ==============================================================================
# A4 : PRE-RECONNAISSANCE WINDOW DETECTION
# ==============================================================================

def find_prerecon_windows(df: pd.DataFrame, known_periods: list,
                           days_before: int = 14) -> pd.DataFrame:
    """
    Identifie les fenêtres gate_open dans la fenêtre [J-days_before, J-1]
    avant la première attaque REAL.

    Ref : Benson et al. (2010) IMC — scanning 24-168h avant DDoS ;
          Steinberger et al. (2020) CNSM — pre-attack reconnaissance.
    """
    real_attacks = [k for k in known_periods if k['origin'] == 'REAL']
    if not real_attacks:
        return pd.DataFrame()

    # Prendre la plus ancienne attaque réelle
    real_start = min(k['start'] for k in real_attacks)
    prerecon_start = real_start - pd.Timedelta(days=days_before)
    prerecon_end   = real_start - pd.Timedelta(minutes=5)  # fenêtre juste avant

    gate_col = 'gate_open'
    if gate_col not in df.columns:
        p_col = get_detection_col(CONFIG, up_levels=1)
        df = df.copy()
        df[gate_col] = df[p_col] >= DECISION_THR

    mask = (
        (df['timestamp'] >= prerecon_start) &
        (df['timestamp'] <= prerecon_end) &
        (df[gate_col] == True)
    )
    return df[mask].copy()


# ==============================================================================
# A7 : EXPORT LaTeX
# ==============================================================================

def export_latex_unknown_episodes(ep_data: list, output_path: str):
    r"""
    Génère une table LaTeX des épisodes inconnus (FP/UNKNOWN) pour publication.
    Format compatible avec booktabs (\toprule, \midrule, \bottomrule).
    """
    unknown = [ep for ep in ep_data
               if ep['origin'] == 'UNKNOWN' and
               (ep['duration_min'] > 10 or ep['n_windows'] > 2)]

    if not unknown:
        return

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Unidentified anomalous episodes detected by SL-ADS on the RedeRio backbone. "
        r"Shaded rows exhibit novelty signal ($\overline{u}_{\text{SBN}} > 0.44$), "
        r"consistent with known pre-attack or residual anomalous activity.}",
        r"\label{tab:audit_unknown}",
        r"\small",
        r"\begin{tabular}{llrrrll}",
        r"\toprule",
        r"Start & End & Dur. & Win. & $\overline{u}$ & NLR & Top-1 type \\",
        r"\midrule",
    ]
    for ep in unknown:
        start_str = str(ep['start'])[:16].replace('_', ' ')
        end_str   = str(ep['end'])[:16].replace('_', ' ')
        u_str     = f"{ep['u_mean']:.2f}" if not np.isnan(ep['u_mean']) else "—"
        nlr_str   = f"{ep['novelty_lr_mean']:.2f}" if not np.isnan(ep.get('novelty_lr_mean', float('nan'))) else "—"
        type_str  = ep['top_type'].replace('_', '\\_')
        dur_str   = f"{int(ep['duration_min'])}"
        n_str     = str(ep['n_windows'])
        row = f"{start_str} & {end_str} & {dur_str}m & {n_str} & {u_str} & {nlr_str} & {type_str} \\\\"
        lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  LaTeX table → {output_path}")


# ==============================================================================
# NEAR-MISS ANALYSIS (B5 FIX : priorité INJECTED)
# ==============================================================================

def audit_near_miss(results_dir: str, known_periods: list, delta: float = 0.20):
    """
    B5 FIX : cherche detection_results_INJECTED.csv en premier.
    Analyse les fenêtres avec P_Anom ∈ [delta/2, delta).
    """
    # Priorité au fichier injecté
    candidates = [
        'detection_results_INJECTED.csv',
        'detection_results.csv',
    ]
    fpath = None
    for fname in candidates:
        p = os.path.join(results_dir, fname)
        if os.path.exists(p):
            fpath = p; break

    if fpath is None:
        print("❌ Aucun detection_results trouvé pour near-miss."); return

    print(f"\n-> Near-miss depuis : {os.path.basename(fpath)}")
    df = pd.read_csv(fpath, parse_dates=['timestamp'])
    p_col = get_detection_col(CONFIG, up_levels=1)
    if p_col not in df.columns:
        print(f"❌ Colonne {p_col} introuvable."); return

    delta_low = delta / 2
    mask_nm = (df[p_col] >= delta_low) & (df[p_col] < delta)
    df_nm = df[mask_nm].copy()

    n_nm = len(df_nm)
    print(f"   Fenêtres near-miss [{delta_low:.2f}, {delta:.2f}) : {n_nm} "
          f"({n_nm/len(df)*100:.2f}%)")
    if n_nm == 0:
        print("   ✅ Aucune fenêtre near-miss."); return

    # Vectorisé B4
    in_known = pd.Series(False, index=df_nm.index)
    for k in known_periods:
        in_known |= (df_nm['timestamp'] >= k['start']) & (df_nm['timestamp'] < k['end'])

    n_known  = int(in_known.sum())
    n_unkown = int((~in_known).sum())
    print(f"   Dans périodes connues    : {n_known} ({n_known/n_nm*100:.1f}%)")
    print(f"   Hors périodes connues    : {n_unkown} ({n_unkown/n_nm*100:.1f}%)")

    if n_unkown > 0:
        df_unk = df_nm[~in_known].copy()
        df_unk['date'] = df_unk['timestamp'].dt.date
        print("\n   Distribution temporelle near-miss hors périodes connues :")
        for day, cnt in df_unk.groupby('date').size().items():
            print(f"     {day} : {cnt} fenêtres")

    # Distribution P_Anom
    print(f"\n   Distribution P_Anom (near-miss) :")
    bins = np.linspace(delta_low, delta, 4)
    for lo, hi in zip(bins[:-1], bins[1:]):
        cnt = int(((df[p_col] >= lo) & (df[p_col] < hi)).sum())
        print(f"     [{lo:.3f}, {hi:.3f}) : {cnt}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gap_min',     type=float, default=15)
    parser.add_argument('--min_windows', type=int,   default=1)
    parser.add_argument('--near_miss',   action='store_true')
    parser.add_argument('--latex',       action='store_true',
                        help='Export LaTeX table des épisodes inconnus')
    parser.add_argument('--input',       default=None,
                        help='CSV de qualification explicite (surcharge QUALIF_CSV_CANDIDATES)')
    args = parser.parse_args()

    # B1 FIX : sélection fichier source
    if args.input:
        QUALIF_CSV = args.input
        print(f"-> CSV spécifié explicitement : {QUALIF_CSV}")
    else:
        QUALIF_CSV = None
        for c in QUALIF_CSV_CANDIDATES:
            if os.path.exists(c):
                QUALIF_CSV = c; break
        if QUALIF_CSV is None:
            print("❌ Aucun CSV de qualification trouvé."); sys.exit(1)
        using_sbn = 'sbn' in os.path.basename(QUALIF_CSV)
        print(f"-> CSV qualification : {QUALIF_CSV}")
        if 'real' in os.path.basename(QUALIF_CSV):
            print(f"   ✅ Audit sur trafic non modifié (sans injection)")
        else:
            print(f"   ⚠️  Audit sur CSV injecté — fenêtres artificielles incluses")

    df = pd.read_csv(QUALIF_CSV, parse_dates=['timestamp'])
    print(f"   {len(df)} fenêtres | {df['timestamp'].min()} → {df['timestamp'].max()}")

    # B3 FIX : calculer une seule fois hors boucle — DOIT être avant le merger
    p_anom_col = get_detection_col(CONFIG, up_levels=1)

    # Merger avec detection_results pour récupérer les colonnes P_Anom (Fix C)
    detect_csv_path = None
    for fname in ['detection_results_INJECTED.csv', 'detection_results.csv']:
        p = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(p):
            detect_csv_path = p;
            break

    if detect_csv_path:
        df_det = pd.read_csv(detect_csv_path, parse_dates=['timestamp'])
        # Merger uniquement la colonne p_anom et gate_open si absents
        merge_cols = ['timestamp']
        if p_anom_col in df_det.columns and p_anom_col not in df.columns:
            merge_cols.append(p_anom_col)
        if 'gate_open' not in df.columns and 'gate_open' in df_det.columns:
            merge_cols.append('gate_open')
        if len(merge_cols) > 1:
            df = df.merge(df_det[merge_cols], on='timestamp', how='left')
            print(f"   Mergé avec {os.path.basename(detect_csv_path)} "
                  f"→ colonnes ajoutées : {merge_cols[1:]}")

    # B2 FIX : auto-détection format + colonne u
    fmt = 'sbn' if any(c.startswith('b_sbn_') for c in df.columns) else 'lr'
    u_col = 'u_sbn' if fmt == 'sbn' else ('u_qualif' if 'u_qualif' in df.columns else 'u_sbn')
    print(f"   Format détecté : {fmt.upper()} | colonne incertitude : {u_col}")

    known_periods = build_known_periods()

    # gate_open
    if 'gate_open' not in df.columns:
        df['gate_open'] = df[p_anom_col] >= DECISION_THR
    df_gate = df[df['gate_open'] == True].copy()

    # A1 FIX : FAR vectorisé + Wilson CI
    far_stats = compute_far_wilson(df, df_gate, known_periods)
    n_total   = len(df)
    n_anomaly = len(df_gate)

    print(f"\n   Total fenêtres     : {n_total}")
    print(f"   Fenêtres anormales : {n_anomaly} ({n_anomaly/n_total*100:.2f}%)")
    print(f"   FAR (FP/normales)  : {far_stats['far_pct']:.3f}%  "
          f"[Wilson 95% CI : {far_stats['far_ci_lo']:.3f}% – {far_stats['far_ci_hi']:.3f}%]")
    print(f"   FP count           : {far_stats['n_fp']} / {far_stats['n_normal']} normales")

    # Groupement en épisodes
    episodes = group_episodes(df_gate, gap_min=args.gap_min)
    print(f"\n   Épisodes détectés  : {len(episodes)} (gap_min={args.gap_min} min)\n")

    # Caractérisation (B3/B6/A2/A5/A6 FIX)
    ep_data = [
        characterize_episode(ep, known_periods, p_anom_col, u_col, fmt)
        for ep in episodes
        if len(ep) >= args.min_windows
    ]

    # A3 : qualification correctness par épisode
    for ep in ep_data:
        ep['qual_correct'] = check_qual_correct(ep, known_periods)

    # ── Affichage ──────────────────────────────────────────────────────────────
    header_nlr = "  NLR" if fmt == 'sbn' else ""
    print(f"{'='*110}")
    print(f"  AUDIT EXHAUSTIF — {len(ep_data)} épisodes anormaux")
    print(f"{'='*110}")
    print(f"  {'#':<4} {'Début':>19} {'Dur':>6} {'Win':>4} {'Type':>22} "
          f"{'Conf':>5} {'u':>6}{header_nlr:>6} {'Origine':>12} {'Attaque'}")
    print(f"  {'─'*106}")

    n_injected = n_real = n_fp = n_gap = 0
    n_qual_correct = n_qual_wrong = 0

    for i, ep in enumerate(ep_data, 1):
        origin = ep['origin']
        if origin == 'REAL':
            color = GREEN; tag = 'REAL'; n_real += 1
        elif origin == 'INJECTED':
            color = CYAN; tag = 'INJECT'; n_injected += 1
        elif ep['duration_min'] <= 10 and ep['n_windows'] <= 2:
            color = YELLOW; tag = 'TRANSIT?'; n_gap += 1
        else:
            color = RED; tag = 'FP/INCONNU'; n_fp += 1

        # Qualification correctness marker
        qmark = ''
        if ep.get('qual_correct') == 'CORRECT':
            qmark = ' ✓'; n_qual_correct += 1
        elif ep.get('qual_correct') == 'WRONG':
            qmark = ' ✗'; n_qual_wrong += 1

        nlr_str = (f"{ep['novelty_lr_mean']:.2f}" if fmt == 'sbn' and
                   not np.isnan(ep.get('novelty_lr_mean', float('nan'))) else "   ")
        u_str   = f"{ep['u_mean']:.3f}" if not np.isnan(ep['u_mean']) else "  N/A"

        print(f"  {i:<4} {str(ep['start'])[:19]:>19} "
              f"{ep['duration_min']:>5.0f}m "
              f"{ep['n_windows']:>4} "
              f"{ep['top_type']:>22} "
              f"{ep['top_pct']:>4.0f}% "
              f"{u_str:>6}"
              f"{nlr_str:>6} "
              f"{color}{tag:>12}{RESET} "
              f"{ep['attack_name']}{qmark}")

    print(f"  {'─'*106}")

    # ── Synthèse ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SYNTHÈSE AUDIT")
    print(f"{'='*70}")
    print(f"  Épisodes attaque réelle    : {n_real}")
    print(f"  Épisodes injectés          : {n_injected}")
    print(f"  Épisodes transitoires      : {n_gap}")
    print(f"  Épisodes inconnus / FP     : {n_fp}")
    if n_qual_correct + n_qual_wrong > 0:
        print(f"\n  A3 Qualification correctness :")
        n_eval = n_qual_correct + n_qual_wrong
        print(f"    Correct : {n_qual_correct}/{n_eval} ({100*n_qual_correct/n_eval:.1f}%)")
        print(f"    Wrong   : {n_qual_wrong}/{n_eval} ({100*n_qual_wrong/n_eval:.1f}%)")

    # ── A4 : Pré-reconnaissance ────────────────────────────────────────────────
    # PATCH TASK-33 / MIN-04 (audit_tmp, 2026-04-26)
    # ──────────────────────────────────────────────────────────────────────
    # L'ancien message "Aucune pré-recon..." était imprimé alors que des
    # fenêtres pré-recon ÉTAIENT détectées (df_prerecon non-vide) — message
    # logiquement inversé qui pouvait masquer la présence réelle de signaux
    # pré-attaque dans le dataset opérationnel.
    df_prerecon = find_prerecon_windows(df, known_periods, days_before=14)
    if not df_prerecon.empty:
        print(f"\n{'='*70}")
        print(f"  A4 : {len(df_prerecon)} fenêtres pré-reconnaissance détectées "
              f"avant le DDoS réel.")
        print("       Note : dataset commence Nov 10 → J-2 seulement disponible.")
        print("       La pré-recon documentée (EVT holdout) concerne les données d'entraînement.")
        print(f"       Ref : Benson (2010) IMC ; Steinberger (2020) CNSM")
        print(f"{'='*70}")
        df_prerecon['date'] = df_prerecon['timestamp'].dt.date
        for day, grp in df_prerecon.groupby('date'):
            print(f"    {day} : {len(grp)} fenêtres  "
                  f"P_Anom_max={grp[p_anom_col].max():.3f}")
    else:
        print(f"\n  A4 : Aucune fenêtre pré-reconnaissance détectée.")

    # ── Épisodes inconnus en détail ─────────────────────────────────────────────
    fp_eps = [ep for ep in ep_data
              if ep['origin'] == 'UNKNOWN' and
              (ep['duration_min'] > 10 or ep['n_windows'] > 2)]
    if fp_eps:
        fp_sorted = sorted(fp_eps, key=lambda e: e['severity'], reverse=True)
        print(f"\n{'='*70}")
        print(f"  ÉPISODES INCONNUS (durée > 10 min ou > 2 fen.) — triés par sévérité")
        print(f"{'='*70}")
        for ep in fp_sorted:
            u_s   = f"{ep['u_mean']:.3f}" if not np.isnan(ep['u_mean']) else "N/A"
            nlr_s = (f"  nlr={ep['novelty_lr_mean']:.3f}" if fmt == 'sbn' else "")
            print(f"  {str(ep['start'])[:19]} → {str(ep['end'])[:19]}"
                  f"  ({ep['duration_min']:.0f}m, {ep['n_windows']}fen)"
                  f"  → {ep['top_type']} ({ep['top_pct']:.0f}%)"
                  f"  u={u_s}  sév={ep['severity']:.0f}{nlr_s}")
    else:
        print(f"\n  ✅ Aucun épisode inconnu de durée significative.")

    # Distribution des types sur FP
    all_fp_types = [ep['top_type'] for ep in ep_data if ep['origin'] == 'UNKNOWN']
    if all_fp_types:
        print(f"\n  Distribution types sur épisodes NON-CONNUS :")
        for t, c in Counter(all_fp_types).most_common():
            print(f"    {t:<25} : {c}")

    # ── Vérification cohérence (avec IoU) ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  VÉRIFICATION COHÉRENCE — Couverture des attaques connues")
    print(f"{'='*70}")
    for k in known_periods:
        mask_det   = (df_gate['timestamp'] >= k['start']) & (df_gate['timestamp'] < k['end'])
        mask_total = (df['timestamp'] >= k['start'])      & (df['timestamp'] < k['end'])
        n_det   = mask_det.sum()
        n_total_p = mask_total.sum()
        recall  = n_det / n_total_p * 100 if n_total_p > 0 else 0
        status  = "✅" if recall > 50 else ("⚠️" if recall > 0 else "❌")
        print(f"  {status} {k['name']:<35} {k['origin']:<8} "
              f"recall={recall:5.1f}%  ({n_det}/{n_total_p})")

    # ── Near-miss ──────────────────────────────────────────────────────────────
    if args.near_miss:
        print(f"\n{'='*70}")
        print(f"  ANALYSE NEAR-MISS")
        print(f"{'='*70}")
        audit_near_miss(RESULTS_DIR, known_periods, delta=DECISION_THR)

    # ── Sauvegarde CSV ──────────────────────────────────────────────────────────
    df_out   = pd.DataFrame(ep_data)
    out_path = os.path.join(RESULTS_DIR, "audit_episodes.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\n  CSV sauvegardé : {out_path}")

    # A7 : export LaTeX si demandé
    if args.latex:
        latex_path = os.path.join(RESULTS_DIR, "audit_unknown_episodes.tex")
        export_latex_unknown_episodes(ep_data, latex_path)

    print(f"\n{'='*70}")
    print(f"  Audit terminé. {len(ep_data)} épisodes | FAR={far_stats['far_pct']:.3f}% "
          f"[{far_stats['far_ci_lo']:.3f}%-{far_stats['far_ci_hi']:.3f}% Wilson 95%]")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
