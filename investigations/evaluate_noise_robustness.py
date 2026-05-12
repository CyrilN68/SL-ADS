"""
evaluate_noise_robustness.py — Robustesse au bruit de collecte (multi-λ)
=========================================================================
Mesure la dégradation du F1 en fonction du taux de discontinuités simulées,
pour plusieurs valeurs de λ. Permet de montrer que l'ageing temporel atténue
l'effet des inter-file gaps, et que λ élevé est plus robuste.

Principe :
  Pour chaque λ ∈ LAMBDA_VALUES et chaque taux p ∈ NOISE_RATES :
    1. Copier le CSV d'evidence
    2. Mettre à zéro les colonnes _P/_S/_N de p% des fenêtres hors-attaque
    3. Rejouer la fusion SL avec ce λ
    4. Évaluer F1-binary, F1-coverage, FPR

Sorties :
  - noise_robustness.csv          : résultats complets (λ × taux × seed)
  - noise_robustness_f1cov.png    : F1-cov vs taux de bruit par λ
  - noise_robustness_fpr.png      : FPR vs taux de bruit par λ

Paramètres dans config.py :
  CONFIG["NOISE_ROBUSTNESS"]["NOISE_RATES"]     : liste de taux en %
  CONFIG["NOISE_ROBUSTNESS"]["LAMBDA_VALUES"]   : liste de λ à tester
  CONFIG["NOISE_ROBUSTNESS"]["N_SEEDS"]         : répétitions par taux
  CONFIG["NOISE_ROBUSTNESS"]["THRESHOLD"]       : seuil opérationnel p_atk (prob projetée)

Pipeline utilisé : Full SL-ADS (λ paramétrable, EDP + trust_discount + C1 conflict_aware).
  Les composants figés (EDP, trust_scores) sont chargés depuis le pkl entraîné.
  CONFIG["NOISE_ROBUSTNESS"]["NOISE_OUTSIDE_ONLY"] : True = hors-attaque seulement
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from config import CONFIG
except ImportError:
    print("❌ config.py introuvable.")
    sys.exit(1)

try:
    import sl_formulas_v2 as sl
except ImportError:
    print("❌ sl_formulas_v2.py introuvable.")
    sys.exit(1)

try:
    from paths import get_model_path, get_decision_threshold, get_decision_variable
    import joblib as _jl
    _HAS_PATHS = True
except ImportError:
    _HAS_PATHS = False

# ==============================================================================
# PARAMÈTRES
# ==============================================================================
_EVAL  = CONFIG.get("EVAL", {})
_NR    = CONFIG.get("NOISE_ROBUSTNESS", {})

VERSION_NAME = CONFIG.get("VERSION_NAME", "trained_models_v9_v6_v4s")
VERSION_NAME_MODIF = CONFIG.get("VERSION_NAME_MODIF", f"{VERSION_NAME}_attacks")
RESULTS_DIR = CONFIG.get("EVAL", {}).get("RESULTS_DIR", f"../../results/resultats_{VERSION_NAME}")
EVIDENCE_CSV = os.path.join(RESULTS_DIR, f"evidence_{VERSION_NAME_MODIF}.csv")
METADATA_CSV = os.path.join(RESULTS_DIR, f"metadata_{VERSION_NAME}.csv")
OUTPUT_DIR   = os.path.join(RESULTS_DIR, "noise_robustness")

NOISE_RATES        = _NR.get("NOISE_RATES", [0, 5, 15, 30])
# λ à tester : λ=0 = sans mémoire (markovien pur), λ=0.95 = référence
LAMBDA_VALUES      = _NR.get("LAMBDA_VALUES", [0.0, 0.50, 0.85, 0.95, 0.99])
N_SEEDS            = _NR.get("N_SEEDS",       5)
THRESHOLD          = _NR.get("THRESHOLD",
                              get_decision_threshold(CONFIG, up_levels=1)
                              if _HAS_PATHS else _EVAL.get("DECISION_THRESHOLD", 0.20))
NOISE_OUTSIDE_ONLY = _NR.get("NOISE_OUTSIDE_ONLY", True)

W_BIJ          = CONFIG.get("SL_PARAM_K",      3.0)
PRIOR_A        = np.array(CONFIG.get("SL_PRIOR_A", [1/3, 1/3, 1/3]), dtype=float)
CONFLICT_ALPHA = CONFIG.get("CONFLICT_ALPHA",  1.495)

# Chargement du pkl entraîné (EDP + trust_scores) — même logique que run_ablation_v2
_edp_dict    = None
_trust_scores = None
_decision_variable = 'proj_atk'
if _HAS_PATHS:
    _model_path = get_model_path(CONFIG, up_levels=1)
    _decision_variable = get_decision_variable(CONFIG, up_levels=1)
    if os.path.exists(_model_path):
        _models_pkg = _jl.load(_model_path)
        if CONFIG.get("USE_EMPIRICAL_PRIOR", True) and _models_pkg.get("empirical_priors"):
            _edp_dict = _models_pkg["empirical_priors"]
        _trust_scores = _models_pkg.get("trust_scores", None)
        print(f"-> PKL chargé : EDP={'oui' if _edp_dict else 'non'}, "
              f"trust_scores={'oui' if _trust_scores else 'non'}, "
              f"decision_variable={_decision_variable}")
    else:
        print(f"⚠️  PKL introuvable ({_model_path}) — prior uniforme utilisé")

_PALETTE_LAMBDA = {
    0.0:  "#aaaaaa",
    0.50: "#ff7f0e",
    0.85: "#2ca02c",
    0.95: "#1f77b4",
    0.99: "#d62728",
}


# ==============================================================================
# CHARGEMENT
# ==============================================================================

def load_evidence_and_meta():
    for path in [EVIDENCE_CSV, METADATA_CSV]:
        if not os.path.exists(path):
            print(f"❌ Fichier introuvable : {path}")
            sys.exit(1)
    df_ev   = pd.read_csv(EVIDENCE_CSV, parse_dates=["timestamp"])
    df_ev   = df_ev.sort_values("timestamp").reset_index(drop=True)
    df_meta = pd.read_csv(METADATA_CSV)
    print(f"-> Evidence : {len(df_ev)} fenêtres")
    print(f"-> Metadata : {len(df_meta)} métriques")
    return df_ev, df_meta


def load_attack_catalog():
    catalog = []
    try:
        from inject_at_evidence_level import ATTACK_CATALOG as _C
        catalog = list(_C)
    except ImportError:
        pass
    if _EVAL.get("INCLUDE_REAL_ATTACK", True):
        catalog += _EVAL.get("REAL_ATTACK_CATALOG", [])
    return catalog


# ==============================================================================
# FUSION SL (replay avec λ paramétrable)
# ==============================================================================

def replay_sl_fusion(df_ev, df_meta, lam):
    """Rejoue la fusion SL avec un λ donné — pipeline Full SL-ADS :
    EDP (empirical priors), trust discounting C3, ageing adaptatif C1 (conflict_aware).
    Retourne une Series p_atk (prob projetée = b[2] + a[2]*u)."""
    meta = {}
    for _, row in df_meta.iterrows():
        mk = row["metric_key"]
        meta[mk] = {
            "clean_key": row["clean_key"],
            "type":      row["type"],
            "r2":        max(float(row["r2_weight"]), 0.01),
        }

    prophet_keys = [k for k, v in meta.items() if v["type"] == "prophet"]
    reconst_keys = [k for k, v in meta.items() if v["type"] == "reconstruction"]
    all_keys     = prophet_keys + reconst_keys

    state_memory = {k: np.zeros(3) for k in all_keys}
    p_atk_list   = []
    ts_list      = []

    for _, row in df_ev.iterrows():
        ops_prophet, w_prophet = [], []
        ops_reconst, w_reconst = [], []

        for key in all_keys:
            ck    = meta[key]["clean_key"]
            P     = float(row.get(f"{ck}_P", 0.0) or 0.0)
            S     = float(row.get(f"{ck}_S", 0.0) or 0.0)
            N     = float(row.get(f"{ck}_N", 0.0) or 0.0)
            r_curr = np.array([P, S, N])

            # C1 — ageing adaptatif (conflict_aware=True, comme Full SL-ADS)
            r_acc, _, _ = sl.temporal_adaptive_ageing(
                state_memory[key], r_curr, lam, W=W_BIJ,
                alpha=CONFLICT_ALPHA, conflict_aware=True)
            state_memory[key] = r_acc

            # Base rate : EDP entraîné si disponible, sinon prior uniforme (fallback)
            if _edp_dict is not None and key in _edp_dict:
                edp_k = _edp_dict[key]
                a = np.array([edp_k["a_safe"], edp_k["a_susp"], edp_k["a_atk"]])
            else:
                a = PRIOR_A.copy()

            op = sl.evidence_to_opinion(r_acc, W=W_BIJ, a=a)

            # C3 — trust discounting (Full SL-ADS) ; w=1.0 car discount absorbé dans op
            if _trust_scores is not None and key in _trust_scores:
                t = float(_trust_scores[key])
                op = sl.apply_trust_discount(op, t)
                w = 1.0
            else:
                w = meta[key]["r2"]  # fallback r2_static si pas de trust_scores

            if meta[key]["type"] == "prophet":
                ops_prophet.append(op); w_prophet.append(w)
            else:
                ops_reconst.append(op); w_reconst.append(w)

        op_p = sl.fusion_wbf_n_sources(ops_prophet, external_weights=w_prophet) if ops_prophet else None
        op_r = sl.fusion_wbf_n_sources(ops_reconst, external_weights=w_reconst) if ops_reconst else None

        if op_p is not None and op_r is not None:
            op_final = sl.fusion_cbf(op_p, op_r)
        elif op_p is not None:
            op_final = op_p
        else:
            op_final = op_r

        p_atk = (float(op_final.b[2]) if _decision_variable == 'b_atk'
                 else float(op_final.projected_prob()[2])) if op_final is not None else 0.0
        p_atk_list.append(p_atk)
        ts_list.append(row["timestamp"])

    return pd.Series(p_atk_list, index=range(len(ts_list)), name="p_atk"), ts_list


# ==============================================================================
# INJECTION BRUIT
# ==============================================================================

def inject_noise(df_ev, noise_rate_pct, catalog, rng):
    df_noisy  = df_ev.copy()
    ev_cols   = [c for c in df_ev.columns if c.endswith(("_P", "_S", "_N"))]

    if NOISE_OUTSIDE_ONLY:
        noise_mask = pd.Series(True, index=df_ev.index)
        for atk in catalog:
            t0 = pd.Timestamp(atk["start"])
            t1 = t0 + pd.Timedelta(hours=atk["duration_h"])
            noise_mask &= ~((df_ev["timestamp"] >= t0) & (df_ev["timestamp"] < t1))
    else:
        noise_mask = pd.Series(True, index=df_ev.index)

    eligible_idx = df_ev.index[noise_mask].tolist()
    n_to_zero    = int(len(eligible_idx) * noise_rate_pct / 100.0)
    if n_to_zero > 0:
        chosen = rng.choice(eligible_idx, size=n_to_zero, replace=False)
        df_noisy.loc[chosen, ev_cols] = 0.0
    return df_noisy, n_to_zero


# ==============================================================================
# ÉVALUATION
# ==============================================================================

def evaluate_p_atk(b_atk_vals, ts_list, catalog, threshold):
    ts = pd.Series(ts_list)
    n_total, n_tp, cov_sum, n_det_win = 0, 0, 0.0, 0

    for atk in catalog:
        t0 = pd.Timestamp(atk["start"])
        t1 = t0 + pd.Timedelta(hours=atk["duration_h"])
        gt_mask = ((ts >= t0) & (ts < t1)).values
        b_gt    = b_atk_vals[gt_mask]
        if len(b_gt) == 0:
            continue
        n_total += 1
        det = b_gt >= threshold
        if det.any():
            n_tp      += 1
            n_det_win += int(det.sum())
            cov_sum   += det.sum() / len(b_gt)

    out_mask = pd.Series(True, index=range(len(ts)))
    for atk in catalog:
        t0 = pd.Timestamp(atk["start"])
        t1 = t0 + pd.Timedelta(hours=atk["duration_h"])
        out_mask &= ~((ts >= t0) & (ts < t1))

    b_out = b_atk_vals[out_mask.values]
    fp    = int((b_out >= threshold).sum())
    n_out = len(b_out)
    fpr   = 100.0 * fp / n_out if n_out > 0 else 0.0
    prec  = n_det_win / (n_det_win + fp) if (n_det_win + fp) > 0 else 0.0
    r_bin = n_tp / n_total if n_total > 0 else 0.0
    r_cov = cov_sum / n_total if n_total > 0 else 0.0

    def f1(p, r): return 2*p*r/(p+r) if (p+r) > 0 else 0.0

    return {
        "f1_binary":   round(f1(prec, r_bin), 4),
        "f1_coverage": round(f1(prec, r_cov), 4),
        "precision":   round(prec,             4),
        "recall_bin":  round(r_bin,            4),
        "recall_cov":  round(r_cov,            4),
        "fpr_pct":     round(fpr,              3),
        "n_detected":  n_tp,
        "n_attacks":   n_total,
    }


# ==============================================================================
# GRAPHIQUES
# ==============================================================================

def plot_results(df_results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for lam in LAMBDA_VALUES:
        sub = df_results[df_results["lambda"] == lam]
        agg = sub.groupby("noise_rate_pct").agg(
            f1_cov_mean=("f1_coverage", "mean"),
            f1_cov_std= ("f1_coverage", "std"),
            fpr_mean=   ("fpr_pct",     "mean"),
            fpr_std=    ("fpr_pct",     "std"),
        ).reset_index()
        x     = agg["noise_rate_pct"].values
        color = _PALETTE_LAMBDA.get(lam, "#333333")
        label = f"λ={lam:.2f}"

        axes[0].plot(x, agg["f1_cov_mean"], "o-", color=color, lw=2, label=label)
        axes[0].fill_between(x,
                             agg["f1_cov_mean"] - agg["f1_cov_std"],
                             agg["f1_cov_mean"] + agg["f1_cov_std"],
                             color=color, alpha=0.10)
        axes[1].plot(x, agg["fpr_mean"], "^-", color=color, lw=2, label=label)
        axes[1].fill_between(x,
                             agg["fpr_mean"] - agg["fpr_std"],
                             agg["fpr_mean"] + agg["fpr_std"],
                             color=color, alpha=0.10)

    axes[0].set_xlabel("Taux de fenêtres nullifiées (% hors-attaque)", fontsize=11)
    axes[0].set_ylabel("F1-Coverage", fontsize=11)
    axes[0].set_title("F1-Coverage vs Taux de Bruit", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, linestyle="--")

    axes[1].set_xlabel("Taux de fenêtres nullifiées (% hors-attaque)", fontsize=11)
    axes[1].set_ylabel("False Positive Rate (%)", fontsize=11)
    axes[1].set_title("FPR vs Taux de Bruit", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, linestyle="--")

    fig.suptitle(
        f"Robustesse au Bruit de Collecte — Comparaison multi-λ\n"
        f"Threshold p_atk ≥ {THRESHOLD} | {N_SEEDS} seeds par taux",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    fname = os.path.join(output_dir, "noise_robustness.png")
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"   figure: {fname}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(f"\n{'='*65}")
    print("  evaluate_noise_robustness.py")
    print(f"  Taux testés  : {NOISE_RATES}%")
    print(f"  Lambda testés: {LAMBDA_VALUES}")
    print(f"  Seeds        : {N_SEEDS}")
    print(f"  Seuil        : p_atk >= {THRESHOLD}")
    print(f"{'='*65}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_ev, df_meta = load_evidence_and_meta()
    catalog        = load_attack_catalog()
    print(f"-> Catalogue : {len(catalog)} attaques\n")

    all_rows = []

    for lam in LAMBDA_VALUES:
        print(f"\n  === λ = {lam:.2f} ===")
        for noise_pct in NOISE_RATES:
            f1_covs, f1_bins, fprs = [], [], []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed)
                df_noisy, n_zeroed = inject_noise(df_ev, noise_pct, catalog, rng)
                b_atk_series, ts_list = replay_sl_fusion(df_noisy, df_meta, lam)
                res = evaluate_p_atk(b_atk_series.values, ts_list, catalog, THRESHOLD)
                f1_covs.append(res["f1_coverage"])
                f1_bins.append(res["f1_binary"])
                fprs.append(res["fpr_pct"])
                all_rows.append({
                    "lambda": lam, "noise_rate_pct": noise_pct,
                    "seed": seed, "n_zeroed": n_zeroed, **res,
                })

            print(f"  Bruit {noise_pct:>3}% | "
                  f"F1-cov = {np.mean(f1_covs):.3f} ± {np.std(f1_covs):.3f} | "
                  f"FPR = {np.mean(fprs):.3f}%")

    df_results = pd.DataFrame(all_rows)
    df_results.to_csv(os.path.join(OUTPUT_DIR, "noise_robustness.csv"), index=False)
    print(f"\n-> CSV sauvegardé : noise_robustness.csv")

    print("\n-> Génération du graphique...")
    plot_results(df_results, OUTPUT_DIR)
    print(f"\n✅ Résultats dans : {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()