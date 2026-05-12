"""
ablation_trapeze.py — Sensibilité aux seuils t_susp / t_atk (et confirmation
                      de l'invariance de T_TRAPEZE_RATIO)
=============================================================================
Recharge les résidus pré-calculés (raw_data_{VERSION}.csv) et re-simule le
pipeline P/S/N → opinion → WBF uniforme pour une grille de facteurs sur les
seuils EVT, sans relancer le pipeline complet (6h+).

Durée typique : < 1 minute.

Pourquoi les seuils et pas le ratio du trapèze ?
    La zone de transition [t_trapeze_base, t_susp] redistribue uniquement
    l'évidence entre P (Safe) et S (Suspect), jamais vers N (Attack).
    Or proj_atk = b_atk + a_atk·u dépend exclusivement de N et de la
    somme totale P+S+N (= WINDOW_SIZE, constante). Le ratio du trapèze est
    donc structurellement sans effet sur la variable de décision.

    Les paramètres qui comptent réellement sont t_susp et t_atk : ils
    contrôlent quand les résidus commencent à générer de l'évidence N.

Usage :
    python ablation_trapeze.py
    python ablation_trapeze.py --factors 0.5 0.7 0.9 1.0 1.1 1.3 1.5 2.0
    python ablation_trapeze.py --threshold 0.25

Sorties (dans RESULTS_DIR/ablation_trapeze/) :
    threshold_sensitivity.csv   — F1 / précision / rappel par facteur
    threshold_sensitivity.png   — courbes
    trapeze_invariance.txt      — confirmation textuelle de l'invariance
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))

try:
    from config import CONFIG
except ImportError:
    print("❌ Impossible de charger config.py.")
    sys.exit(1)

from paths import get_version_names, get_results_dir, get_decision_threshold
from sl_formulas_v2 import evidence_to_opinion

# ==============================================================================
# PARAMÈTRES
# ==============================================================================

VERSION_NAME, _ = get_version_names(CONFIG)
RESULTS_DIR     = get_results_dir(CONFIG, up_levels=1)
WINDOW_SIZE     = CONFIG.get('WINDOW_SIZE', 10)
W_BIJ           = float(CONFIG.get('SL_PARAM_K', 3.0))
T_TRAPEZE_RATIO = float(CONFIG.get('T_TRAPEZE_RATIO', 0.1))
DECISION_THR    = get_decision_threshold(CONFIG, up_levels=1)

RAW_DATA_PATH = os.path.join(RESULTS_DIR, f"raw_data_{VERSION_NAME}.csv")
META_PATH     = os.path.join(RESULTS_DIR, f"metadata_{VERSION_NAME}.csv")
OUT_DIR       = os.path.join(RESULTS_DIR, "ablation_trapeze")

# Grille de facteurs multiplicatifs appliqués à t_susp ET t_atk simultanément.
# factor=1.0 = seuils EVT d'origine.
DEFAULT_FACTORS = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]

# Ratios trapèze testés pour confirmation d'invariance (section secondaire)
TRAPEZE_RATIOS_CHECK = [0.0, 0.1, 0.5, 0.9]


# ==============================================================================
# NOYAU : compute_instantaneous_evidence (domain-specific, non SL pure)
# Dépend des artefacts directionnels de train_v10 — reste dans ce pipeline.
# ==============================================================================

def _instant_evidence(signed_r: float, t_susp: float, t_atk: float,
                      t_trapeze_base: float, direction) -> tuple:
    """Mapping trapézoïdal directionnel résidu → (p, s, n), p+s+n=1."""
    if direction == 'pos':
        if signed_r <= 0.0:
            return (1.0, 0.0, 0.0)
        e = signed_r
    elif direction == 'neg':
        if signed_r >= 0.0:
            return (1.0, 0.0, 0.0)
        e = abs(signed_r)
    else:
        e = abs(signed_r)

    if e < t_trapeze_base:
        return (1.0, 0.0, 0.0)
    elif e < t_susp:
        w = t_susp - t_trapeze_base
        a = (e - t_trapeze_base) / w if w > 0 else 1.0
        return (1.0 - a, a, 0.0)
    elif e < t_atk:
        w = t_atk - t_susp
        a = (e - t_susp) / w if w > 0 else 1.0
        return (0.0, 1.0 - a, a)
    else:
        return (0.0, 0.0, 1.0)


# ==============================================================================
# RECOMPUTE EVIDENCE
# ==============================================================================

def recompute_evidence(raw_df: pd.DataFrame, meta_df: pd.DataFrame,
                       susp_factor: float = 1.0, atk_factor: float = 1.0,
                       trapeze_ratio: float = None) -> pd.DataFrame:
    """
    Recalcule l'évidence fenêtrée depuis les résidus bruts.

    susp_factor, atk_factor : multiplicateurs appliqués aux seuils EVT.
    trapeze_ratio           : si None, utilise T_TRAPEZE_RATIO de config.
    """
    if trapeze_ratio is None:
        trapeze_ratio = T_TRAPEZE_RATIO

    meta = {
        row['metric_key']: {
            'clean_key': row['clean_key'],
            't_susp':    float(row['threshold_suspect']) * susp_factor,
            't_atk':     float(row['threshold_attack'])  * atk_factor,
            'direction': row['direction'] if row['direction'] != 'both' else None,
        }
        for _, row in meta_df.iterrows()
    }

    evidence_rows = []
    for metric_key, grp in raw_df.groupby('metric_key', sort=False):
        if metric_key not in meta:
            continue
        m  = meta[metric_key]
        ck = m['clean_key']
        t_susp = m['t_susp']
        t_atk  = m['t_atk']
        t_base = trapeze_ratio * t_susp
        direction = m['direction']

        grp = grp.sort_values('timestamp').reset_index(drop=True)
        for win_start in range(0, len(grp), WINDOW_SIZE):
            win = grp.iloc[win_start: win_start + WINDOW_SIZE]
            if win.empty:
                break
            ts = win['timestamp'].iloc[-1]
            P, S, N = 0.0, 0.0, 0.0
            for _, step in win.iterrows():
                p, s, n = _instant_evidence(
                    float(step['real']) - float(step['pred']),
                    t_susp, t_atk, t_base, direction
                )
                P += p; S += s; N += n
            evidence_rows.append({'timestamp': ts, f'{ck}_P': P,
                                  f'{ck}_S': S, f'{ck}_N': N})

    if not evidence_rows:
        return pd.DataFrame()

    ev_long = pd.DataFrame(evidence_rows)
    ev_wide = ev_long.groupby('timestamp').first().reset_index()
    for _, m in meta.items():
        ck = m['clean_key']
        for col in [f'{ck}_P', f'{ck}_S', f'{ck}_N']:
            if col not in ev_wide.columns:
                ev_wide[col] = 0.0
    return ev_wide.sort_values('timestamp').reset_index(drop=True)


# ==============================================================================
# FUSION WBF UNIFORME
# ==============================================================================

def fuse_opinions(ev_df: pd.DataFrame, meta_df: pd.DataFrame) -> np.ndarray:
    """WBF uniforme sans ageing → proj_atk par fenêtre."""
    clean_keys = meta_df['clean_key'].tolist()
    scores = []
    for _, row in ev_df.iterrows():
        opinions = []
        for ck in clean_keys:
            P = float(row.get(f'{ck}_P', 0.0) or 0.0)
            S = float(row.get(f'{ck}_S', 0.0) or 0.0)
            N = float(row.get(f'{ck}_N', 0.0) or 0.0)
            if np.isnan(P): P = 0.0
            if np.isnan(S): S = 0.0
            if np.isnan(N): N = 0.0
            opinions.append(evidence_to_opinion([P, S, N], W=W_BIJ))
        b = np.mean([op.b for op in opinions], axis=0)
        u = float(np.mean([op.u for op in opinions]))
        scores.append(float(b[2] + (1.0 / 3.0) * u))
    return np.array(scores)


# ==============================================================================
# LABELS depuis le catalogue d'attaques
# ==============================================================================

def load_labels(ev_df: pd.DataFrame) -> np.ndarray | None:
    """
    Génère les labels fenêtre depuis CONFIG['EVAL']['REAL_ATTACK_CATALOG'].
    Chaque attaque est définie par {"start": ..., "duration_h": ...}.
    Fallback sur colonne 'label'/'is_attack' dans le CSV source si catalogue vide.
    """
    catalog = list(CONFIG.get('EVAL', {}).get('REAL_ATTACK_CATALOG', []))

    if not catalog:
        try:
            src = pd.read_csv(CONFIG['file_path'])
            label_col = next((c for c in ['label', 'is_attack', 'anomaly', 'attack']
                              if c in src.columns), None)
            if label_col is None:
                return None
            src['ds'] = pd.to_datetime(src['timestamp'])
            test_src = src[src['ds'] > pd.to_datetime(CONFIG['split_date'])]
            test_src = test_src.sort_values('ds').reset_index(drop=True)
            test_src['wid'] = test_src.index // WINDOW_SIZE
            win_labels = test_src.groupby('wid')[label_col].max().reset_index(drop=True)
            n = min(len(win_labels), len(ev_df))
            arr = win_labels.values[:n].astype(int)
            return arr if arr.sum() > 0 else None
        except Exception:
            return None

    timestamps = pd.to_datetime(ev_df['timestamp'])
    y_true = np.zeros(len(ev_df), dtype=int)
    for atk in catalog:
        try:
            t0 = pd.Timestamp(atk['start'])
            t1 = t0 + pd.Timedelta(hours=float(atk['duration_h']))
            y_true[(timestamps >= t0) & (timestamps < t1)] = 1
        except (KeyError, ValueError):
            continue

    if y_true.sum() == 0:
        print("   [WARN] Catalogue chargé mais aucune fenêtre ne tombe dans "
              "un intervalle d'attaque. Vérifiez split_date vs dates du catalogue.")
        return None
    return y_true


# ==============================================================================
# MÉTRIQUES
# ==============================================================================

def compute_metrics(y_true: np.ndarray, y_score: np.ndarray,
                    threshold: float) -> dict:
    n = min(len(y_true), len(y_score))
    yt, yp = y_true[:n], (y_score[:n] >= threshold).astype(int)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {'f1': f1, 'precision': p, 'recall': r, 'tp': tp, 'fp': fp, 'fn': fn}


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    best_f1, best_thr = 0.0, DECISION_THR
    for thr in np.round(np.arange(0.01, 1.0, 0.02), 3):
        f1 = compute_metrics(y_true, y_score, thr)['f1']
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_f1, best_thr


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--factors', nargs='+', type=float, default=DEFAULT_FACTORS,
                        help="Facteurs multiplicatifs sur t_susp et t_atk")
    parser.add_argument('--threshold', type=float, default=DECISION_THR,
                        help="Seuil de décision opérationnel")
    args = parser.parse_args()

    print(f"--- ABLATION SEUILS ({VERSION_NAME}) ---")
    print(f"Seuil opérationnel : {args.threshold:.3f}")
    print(f"Facteurs testés    : {args.factors}")

    for path, name in [(RAW_DATA_PATH, 'raw_data'), (META_PATH, 'metadata')]:
        if not os.path.exists(path):
            print(f"❌ {path} introuvable. Lancez d'abord compute_evidence_v2.py.")
            sys.exit(1)

    print(f"-> Chargement raw_data...")
    raw_df  = pd.read_csv(RAW_DATA_PATH)
    meta_df = pd.read_csv(META_PATH)
    print(f"   {len(raw_df):,} lignes, {raw_df['metric_key'].nunique()} métriques")

    # Labels depuis le catalogue d'attaques
    ev_ref   = recompute_evidence(raw_df, meta_df)  # factor=1.0
    y_true   = load_labels(ev_ref)
    has_lbls = y_true is not None
    if has_lbls:
        print(f"   Labels : {int(y_true.sum())} fenêtres attaque / {len(y_true)} total")
    else:
        print("   [INFO] Pas de labels — affichage distribution uniquement")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ==========================================================================
    # SECTION 1 — Sensibilité aux seuils t_susp / t_atk
    # ==========================================================================
    print("\n=== Sensibilité aux seuils (t_susp × factor, t_atk × factor) ===")
    thr_results = []
    score_by_factor = {}

    for factor in args.factors:
        ev_df  = recompute_evidence(raw_df, meta_df, susp_factor=factor, atk_factor=factor)
        scores = fuse_opinions(ev_df, meta_df)
        score_by_factor[factor] = scores

        row = {'factor': factor,
               'mean_proj_atk': float(np.mean(scores)),
               'p95_proj_atk':  float(np.percentile(scores, 95)),
               'pct_above_thr': float(np.mean(scores >= args.threshold))}

        if has_lbls:
            n = min(len(y_true), len(scores))
            m  = compute_metrics(y_true[:n], scores[:n], args.threshold)
            bf1, bthr = best_threshold(y_true[:n], scores[:n])
            row.update({'f1_operational': m['f1'], 'precision': m['precision'],
                        'recall': m['recall'], 'f1_best': bf1, 'best_thr': bthr})
            print(f"  factor={factor:.2f}  F1_op={m['f1']:.3f}  "
                  f"prec={m['precision']:.3f}  rec={m['recall']:.3f}  "
                  f"F1*={bf1:.3f} (δ*={bthr:.2f})")
        else:
            print(f"  factor={factor:.2f}  mean={row['mean_proj_atk']:.4f}  "
                  f"p95={row['p95_proj_atk']:.4f}  "
                  f"%>δ={row['pct_above_thr']:.3f}")
        thr_results.append(row)

    thr_df = pd.DataFrame(thr_results)
    thr_csv = os.path.join(OUT_DIR, "threshold_sensitivity.csv")
    thr_df.to_csv(thr_csv, index=False, float_format='%.4f')
    print(f"\n📄 {thr_csv}")
    print(thr_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ==========================================================================
    # SECTION 2 — Confirmation invariance T_TRAPEZE_RATIO (facteur=1.0 fixe)
    # ==========================================================================
    print("\n=== Confirmation invariance T_TRAPEZE_RATIO (attendu : scores identiques) ===")
    trap_scores = {}
    for ratio in TRAPEZE_RATIOS_CHECK:
        ev_df = recompute_evidence(raw_df, meta_df, trapeze_ratio=ratio)
        s = fuse_opinions(ev_df, meta_df)
        trap_scores[ratio] = s
        print(f"  ratio={ratio:.1f}  mean={np.mean(s):.6f}  p95={np.percentile(s,95):.6f}")

    inv_confirmed = all(
        np.allclose(trap_scores[TRAPEZE_RATIOS_CHECK[0]], trap_scores[r], rtol=1e-5)
        for r in TRAPEZE_RATIOS_CHECK[1:]
    )
    inv_msg = ("INVARIANCE CONFIRMÉE : T_TRAPEZE_RATIO n'a aucun effet sur proj_atk.\n"
               "Explication : la zone [t_trapeze_base, t_susp] redistribue P↔S\n"
               "mais jamais N. Or proj_atk = b_atk + a_atk·u dépend uniquement\n"
               "de N et de P+S+N (= WINDOW_SIZE, constant). Paramètre librement\n"
               "fixable ; la valeur T_TRAPEZE_RATIO=0.1 dans config.py est valide.\n"
               if inv_confirmed else
               "ATTENTION : résultats non identiques — vérifier le calcul.")
    print(f"\n  → {inv_msg.splitlines()[0]}")

    inv_path = os.path.join(OUT_DIR, "trapeze_invariance.txt")
    with open(inv_path, 'w', encoding='utf-8') as f:
        f.write(inv_msg)

    # ==========================================================================
    # GRAPHIQUES
    # ==========================================================================
    n_panels = (3 if has_lbls else 2)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    factors_arr = np.array(args.factors)

    # Panneau 1 : boxplot proj_atk par facteur
    ax = axes[0]
    ax.boxplot([score_by_factor[f] for f in args.factors],
               tick_labels=[str(f) for f in args.factors])
    ax.axhline(args.threshold, color='red', linestyle='--', lw=1,
               label=f'δ={args.threshold:.2f}')
    ax.set_xlabel("Facteur seuil (× t_susp, × t_atk)")
    ax.set_ylabel("proj_atk (WBF uniforme, sans ageing)")
    ax.set_title("Distribution proj_atk vs facteur seuil")
    ax.legend(fontsize=8)

    # Panneau 2 : % détections
    ax = axes[1]
    ax.plot(factors_arr, thr_df['pct_above_thr'].values, 'o-', color='darkorange')
    ax.axvline(1.0, color='gray', linestyle=':', lw=1, label='seuils EVT originaux')
    ax.set_xlabel("Facteur seuil")
    ax.set_ylabel("Fraction fenêtres détectées")
    ax.set_title("Taux de détection brut vs facteur seuil")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panneau 3 : F1 (si labels)
    if has_lbls:
        ax = axes[2]
        ax.plot(factors_arr, thr_df['f1_operational'].values, 'o-',
                label=f'F1 (δ={args.threshold:.2f})', color='steelblue')
        ax.plot(factors_arr, thr_df['f1_best'].values, 's--',
                label='F1 best (δ*)', color='green')
        ax.axvline(1.0, color='gray', linestyle=':', lw=1, label='seuils EVT')
        ax.set_xlabel("Facteur seuil")
        ax.set_ylabel("F1 fenêtre")
        ax.set_title("F1 vs facteur de calibration des seuils EVT")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Sensibilité seuils t_susp / t_atk — {VERSION_NAME}\n"
                 f"(WBF uniforme, sans ageing)", fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "threshold_sensitivity.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 {plot_path}")
    print("✅ Terminé.")


if __name__ == "__main__":
    main()
