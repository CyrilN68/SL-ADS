"""
benchmark_compute_time.py — Comparaison temps de calcul SL vs Baselines
========================================================================
Mesure le temps de calcul par fenêtre (ms/win) pour :
    - Système SL complet (Ageing + WBF + CBF)
    - WBF seul (sans ageing temporel)
    - CBF seul (sans WBF intra-méthode)
    - Isolation Forest (sklearn)
    - Seuil statique (vote sur evidence N columns)

Rapporte également la RAM peak pour chaque configuration.

Usage :
    python benchmark_compute_time.py
    python benchmark_compute_time.py --n_windows 500   # limiter pour debug rapide

Ref : §8.3 du rapport (System Latency and Resource Usage)
"""
import argparse
import os
import sys
import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

try:
    from config import CONFIG
except ImportError:
    print("❌ config.py introuvable."); sys.exit(1)

try:
    _parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from paths import get_decision_threshold as _get_decision_threshold
    _HAS_PATHS_BCT = True
except ImportError:
    _HAS_PATHS_BCT = False

try:
    import sl_formulas_v2 as sl
except ImportError:
    print("❌ sl_formulas_v2.py introuvable."); sys.exit(1)

import joblib

# ==============================================================================
# PARAMÈTRES
# ==============================================================================
VERSION_NAME       = CONFIG.get("VERSION_NAME", "trained_models_v9_v6_v4s")
VERSION_NAME_MODIF = CONFIG.get("VERSION_NAME_MODIF", f"{VERSION_NAME}_attacks")

EVIDENCE_CSV = f"../results/resultats_{VERSION_NAME}/evidence_{VERSION_NAME_MODIF}.csv"
METADATA_CSV = f"../results/resultats_{VERSION_NAME}/metadata_{VERSION_NAME}.csv"
MODEL_PATH   = f"../../trained_models_{VERSION_NAME}.pkl"

LAMBDA_DECAY   = CONFIG.get('LAMBDA_DECAY', 0.95)
W_BIJ          = CONFIG.get('SL_PARAM_K', 3.0)
CONFLICT_ALPHA = CONFIG.get('CONFLICT_ALPHA', 1.495)
DECISION_THR   = (_get_decision_threshold(CONFIG, up_levels=2)
                  if _HAS_PATHS_BCT else CONFIG.get('EVAL', {}).get('DECISION_THRESHOLD', 0.20))

# PATCH TASK-33 / MIN-02 (audit_tmp, 2026-04-26)
# ──────────────────────────────────────────────────────────────────────────
# Constantes externalisées depuis CONFIG plutôt que hardcodées.
# WINDOW_MINUTES = taille de fenêtre (5 par défaut), BENCH_VOTE_THRESHOLD =
# nombre minimum de pas-de-temps "anormaux" dans la fenêtre pour qu'une
# source vote (= 50% par défaut, soit window_minutes/2).
WINDOW_MINUTES        = int(CONFIG.get('WINDOW_MINUTES', 5))
BENCH_VOTE_THRESHOLD  = int(CONFIG.get('BENCH_VOTE_THRESHOLD', WINDOW_MINUTES // 2))
# Temps total de référence pour `compute_evidence` (Prophet inference)
# — historiquement mesuré à 32305.9s sur ce dataset. Externalisé pour
# éviter un nombre magique non documenté au milieu d'un f-string. Si
# inconnu (autre dataset), passer None et le print sera adapté.
BENCH_PROPHET_TOTAL_S = float(CONFIG.get('BENCH_PROPHET_TOTAL_S', 32305.9))

# ==============================================================================
# CHARGEMENT DES DONNÉES
# ==============================================================================

def load_data(n_windows: int | None = None):
    print(f"-> Chargement evidence CSV : {EVIDENCE_CSV}")
    df_ev_raw = pd.read_csv(EVIDENCE_CSV, parse_dates=['timestamp'])
    evidence_cols = [c for c in df_ev_raw.columns if c.endswith(('_P', '_S', '_N'))]
    df_ev = (df_ev_raw.set_index('timestamp')[evidence_cols]
             .resample('5min', origin='start_day', closed='left', label='left')
             .sum().reset_index())

    df_meta = pd.read_csv(METADATA_CSV)
    meta_dict = {row['metric_key']: row for _, row in df_meta.iterrows()}
    prophet_keys = [k for k, v in meta_dict.items() if v['type'] == 'prophet']
    reconst_keys = [k for k, v in meta_dict.items() if v['type'] == 'reconstruction']

    _models_pkg = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else {}
    edp_dict = _models_pkg.get('empirical_priors') if CONFIG.get('USE_EMPIRICAL_PRIOR', True) else None

    if n_windows:
        df_ev = df_ev.head(n_windows)

    print(f"   {len(df_ev)} fenêtres | {len(prophet_keys)} Prophet + {len(reconst_keys)} RANSAC")
    return df_ev, meta_dict, prophet_keys, reconst_keys, edp_dict


# ==============================================================================
# BENCHMARK : pipeline SL complet
# ==============================================================================

def bench_full_sl(df_ev, meta_dict, prophet_keys, reconst_keys, edp_dict):
    """Pipeline complet : Ageing → WBF → CBF → décision."""
    prior_uniform = np.array(CONFIG.get('SL_PRIOR_A', [1/3, 1/3, 1/3]))
    state_memory = {k: np.zeros(3) for k in prophet_keys + reconst_keys}
    all_keys = prophet_keys + reconst_keys
    decisions = []

    t0 = time.perf_counter()
    for _, row in df_ev.iterrows():
        ops_prophet, ops_reconst = [], []

        for key in all_keys:
            clean_key = meta_dict[key]['clean_key']
            P = row.get(f"{clean_key}_P", 0.0)
            S = row.get(f"{clean_key}_S", 0.0)
            N = row.get(f"{clean_key}_N", 0.0)
            r_current = np.array([P, S, N])

            # Niveau 1 — Ageing adaptatif (H1)
            R_new, _, _ = sl.temporal_adaptive_ageing(
                r_accumulated=state_memory[key], r_current=r_current,
                lam_base=LAMBDA_DECAY, W=W_BIJ, alpha=CONFLICT_ALPHA)
            state_memory[key] = R_new

            # Base rate
            if edp_dict and key in edp_dict:
                e = edp_dict[key]
                a_inj = np.array([e['a_safe'], e['a_susp'], e['a_atk']])
            else:
                a_inj = prior_uniform

            op = sl.evidence_to_opinion(R_new, W=W_BIJ, a=a_inj)

            # Niveau 2 — WBF
            if meta_dict[key]['type'] == 'prophet':
                ops_prophet.append(op)
            else:
                ops_reconst.append(op)

        op_p = sl.fusion_wbf_n_sources(ops_prophet)
        op_r = sl.fusion_wbf_n_sources(ops_reconst)
        # Niveau 3 — CBF
        op_final = sl.fusion_cbf(op_p, op_r)
        decisions.append(float(op_final.projected_prob()[2]) >= DECISION_THR)

    elapsed = time.perf_counter() - t0
    return elapsed, decisions


# ==============================================================================
# BENCHMARK : WBF seul (sans ageing — memoryless)
# ==============================================================================

def bench_wbf_only(df_ev, meta_dict, prophet_keys, reconst_keys, edp_dict):
    """WBF instantané sur evidence courante (pas d'ageing)."""
    prior_uniform = np.array(CONFIG.get('SL_PRIOR_A', [1/3, 1/3, 1/3]))
    all_keys = prophet_keys + reconst_keys
    decisions = []

    t0 = time.perf_counter()
    for _, row in df_ev.iterrows():
        ops_prophet, ops_reconst = [], []
        for key in all_keys:
            clean_key = meta_dict[key]['clean_key']
            r = np.array([row.get(f"{clean_key}_P", 0.0),
                          row.get(f"{clean_key}_S", 0.0),
                          row.get(f"{clean_key}_N", 0.0)])
            a_inj = prior_uniform
            if edp_dict and key in edp_dict:
                e = edp_dict[key]
                a_inj = np.array([e['a_safe'], e['a_susp'], e['a_atk']])
            op = sl.evidence_to_opinion(r, W=W_BIJ, a=a_inj)
            if meta_dict[key]['type'] == 'prophet':
                ops_prophet.append(op)
            else:
                ops_reconst.append(op)
        op_p = sl.fusion_wbf_n_sources(ops_prophet)
        op_r = sl.fusion_wbf_n_sources(ops_reconst)
        op_final = sl.fusion_cbf(op_p, op_r)
        decisions.append(float(op_final.projected_prob()[2]) >= DECISION_THR)

    elapsed = time.perf_counter() - t0
    return elapsed, decisions


# ==============================================================================
# BENCHMARK : Seuil statique (vote sur colonnes N)
# ==============================================================================

def bench_static_threshold(df_ev, meta_dict):
    """Vote majoritaire sur colonnes _N (preuve anomalie brute)."""
    all_keys = list(meta_dict.keys())
    n_sources = len(all_keys)
    threshold_vote = n_sources // 2   # majorité simple
    decisions = []

    t0 = time.perf_counter()
    for _, row in df_ev.iterrows():
        votes = 0
        for key in all_keys:
            clean_key = meta_dict[key]['clean_key']
            N = row.get(f"{clean_key}_N", 0.0)
            # PATCH TASK-33 / MIN-02 : seuil ≥ BENCH_VOTE_THRESHOLD (= 50% du
            # window_size par défaut), externalisé depuis CONFIG.
            if N >= BENCH_VOTE_THRESHOLD:
                votes += 1
        decisions.append(votes >= threshold_vote)

    elapsed = time.perf_counter() - t0
    return elapsed, decisions


# ==============================================================================
# BENCHMARK : Isolation Forest
# ==============================================================================

def bench_isolation_forest(df_ev, meta_dict):
    """Isolation Forest sur feature vector des colonnes _N."""
    all_keys = list(meta_dict.keys())
    N_cols = [f"{meta_dict[k]['clean_key']}_N" for k in all_keys]
    N_cols = [c for c in N_cols if c in df_ev.columns]

    X = df_ev[N_cols].fillna(0).values

    # Train sur les 2 premiers tiers (simuler le split normal/test)
    n_train = int(len(X) * 0.6)
    X_train, X_test = X[:n_train], X[n_train:]

    t0 = time.perf_counter()
    clf = IsolationForest(contamination=0.01, random_state=42, n_jobs=1)
    clf.fit(X_train)
    # Temps d'inférence uniquement (split du chronométrage)
    t_fit = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds = clf.predict(X)  # -1 = anomalie
    t_infer = time.perf_counter() - t1

    decisions = [p == -1 for p in preds]
    return t_fit, t_infer, len(X), decisions


# ==============================================================================
# MESURE RAM
# ==============================================================================

def measure_ram(func, *args):
    tracemalloc.start()
    result = func(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1024 / 1024   # MB


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_windows', type=int, default=None,
                        help='Limiter le nombre de fenêtres (None = toutes)')
    args = parser.parse_args()

    df_ev, meta_dict, prophet_keys, reconst_keys, edp_dict = load_data(args.n_windows)
    N = len(df_ev)

    print(f"\n{'='*70}")
    print(f"  BENCHMARK TEMPS DE CALCUL — {N} fenêtres de 5 min")
    print(f"  Métriques : {len(prophet_keys)} Prophet + {len(reconst_keys)} RANSAC = "
          f"{len(prophet_keys)+len(reconst_keys)} sources")
    print(f"{'='*70}\n")

    results = {}

    # --- SL complet ---
    print("-> [1/4] Pipeline SL complet (Ageing + WBF + CBF)...")
    (t_sl, _), ram_sl = measure_ram(bench_full_sl, df_ev, meta_dict,
                                     prophet_keys, reconst_keys, edp_dict)
    results['Full SL (Ageing+WBF+CBF)'] = {'total_s': t_sl, 'ram_mb': ram_sl}

    # --- WBF seul ---
    print("-> [2/4] WBF + CBF sans ageing (memoryless)...")
    (t_wbf, _), ram_wbf = measure_ram(bench_wbf_only, df_ev, meta_dict,
                                       prophet_keys, reconst_keys, edp_dict)
    results['WBF+CBF memoryless'] = {'total_s': t_wbf, 'ram_mb': ram_wbf}

    # --- Seuil statique ---
    print("-> [3/4] Seuil statique (vote majoritaire)...")
    (t_thr, _), ram_thr = measure_ram(bench_static_threshold, df_ev, meta_dict)
    results['Static Threshold'] = {'total_s': t_thr, 'ram_mb': ram_thr}

    # --- Isolation Forest ---
    print("-> [4/4] Isolation Forest (c=0.01)...")
    tracemalloc.start()
    t_fit, t_infer, n_if, _ = bench_isolation_forest(df_ev, meta_dict)
    _, ram_if = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['Isolation Forest (fit)']   = {'total_s': t_fit,   'ram_mb': ram_if/1024/1024}
    results['Isolation Forest (infer)'] = {'total_s': t_infer, 'ram_mb': ram_if/1024/1024}

    # --- Rapport ---
    print(f"\n{'='*70}")
    print(f"  {'Configuration':<35} {'Total (s)':>10} {'ms/fenêtre':>12} {'RAM (MB)':>10}")
    print(f"  {'-'*67}")
    for name, r in results.items():
        n = N if 'Forest' not in name else n_if
        ms_per_win = (r['total_s'] / n * 1000) if n > 0 else 0
        print(f"  {name:<35} {r['total_s']:>10.2f} {ms_per_win:>12.3f} {r['ram_mb']:>10.1f}")

    print(f"{'='*70}")
    overhead = results['Full SL (Ageing+WBF+CBF)']['total_s'] / results['Static Threshold']['total_s']
    print(f"\n  Overhead SL vs seuil statique : ×{overhead:.1f}")
    print(f"  Overhead SL vs WBF memoryless : ×"
          f"{results['Full SL (Ageing+WBF+CBF)']['total_s']/results['WBF+CBF memoryless']['total_s']:.2f}"
          f"  (coût de H1 ageing)")

    # Sauvegarde CSV
    out_csv = f"../results/resultats_{VERSION_NAME}/benchmark_compute_time.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = []
    for name, r in results.items():
        n = N
        rows.append({'configuration': name,
                     'n_windows': n,
                     'total_s': round(r['total_s'], 3),
                     'ms_per_window': round(r['total_s']/n*1000, 3),
                     'ram_mb': round(r['ram_mb'], 1)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n  Résultats sauvegardés : {out_csv}")
    print(f"\n  NOTE §8.3 : ces temps concernent uniquement compute_opinions (fusion SL).")
    # PATCH TASK-33 / MIN-02 : valeurs externalisées depuis CONFIG.
    if BENCH_PROPHET_TOTAL_S and BENCH_PROPHET_TOTAL_S > 0:
        print(f"  compute_evidence (Prophet inference) : {BENCH_PROPHET_TOTAL_S:.0f}s total, "
              f"~{BENCH_PROPHET_TOTAL_S/N*1000:.0f} ms/fenêtre — dominant et indépendant du choix de fusion.")
    else:
        print(f"  compute_evidence (Prophet inference) : non mesuré pour ce dataset "
              f"(set BENCH_PROPHET_TOTAL_S in CONFIG to populate this line).")


if __name__ == "__main__":
    main()