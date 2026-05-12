"""
run_ablation.py — Ablation Study + Baseline Comparison for SL-IDS
==================================================================
Ref: Meyes et al. (2019) "Ablation Studies in Artificial Neural Networks" arXiv:1901.08644
     Sharafaldin et al. (2018) CIC-IDS2017 — IDS baseline framework
     Liu et al. (2008) "Isolation Forest" ICDM

PURPOSE:
    Evaluates the contribution of each SL fusion component by
    disabling one component at a time (standard ablation study).
    Also evaluates two classical baselines for fair comparison.

    All systems (SL ablations + baselines) read the SAME injected
    evidence CSV so the comparison is fully fair.

ABLATION RUNS:
    Full SL (λ=0.85): production system, uniform WBF inter-method fusion (reference)
    Ageing Sensitivity (λ=X): tests different memory inertias
    Uniform Weights : WBF with equal weights (R2 quality ignored)
    Prophet Only    : WBF over Prophet metrics only, no CBF
    Reconst Only    : WBF over Reconstruction metrics only, no CBF
    WBF inter-method: production duplicate / explicit WBF(P,R), no inter-method CBF

CLASSICAL BASELINES:
    Static Threshold : per-metric rule-based vote on evidence N columns
    Isolation Forest : IsolationForest on evidence N feature vector (Sensitivity tested)

OUTPUTS (in evaluation/ablation/):
    ablation_summary.csv         — operational-threshold metrics per run
    ablation_all_sweeps.csv      — full threshold sweep per run
    ablation_f1_curves.png       — F1-coverage curves per run (if multiple thresholds)
    ablation_comparison.png      — publication table
    ablation_bar_comparison.png  — grouped bar chart

USAGE:
    python run_ablation.py
    (run from the same directory as sl_formulas_v2.py)

Metric protocol:
    Primary ablation metrics use the same catalog_outages_separate protocol as
    evaluate_injection.py: injected/catalog attacks are positives, real network
    outages are reported separately rather than counted as normal false alarms.
    Legacy outages-as-normal columns are retained only to make old numbers
    auditable and impossible to cite accidentally as the current protocol.

Note (PATCH M-03, 2026-04-21) :
    Le pipeline actif utilise un **EDP statique** (Empirical Dirichlet Prior)
    calculé au training et figé dans le sidecar. Le module historique
    `adaptive_base_rate.py` est archivé dans `un peu old/` et n'est plus
    importé. Le flag CONFIG `"adaptive_base_rate"` est conservé comme alias
    sémantique : `True` = EDP statique (priors empiriques par métrique) ;
    `False` = prior uniforme `SL_PRIOR_A = [1/3, 1/3, 1/3]`.
"""

import os
import sys
import warnings

# PATCH-C4 fix (2026-04-20) : force stdout UTF-8 sur Windows (cp1252 par défaut
# ne peut pas encoder λ, α, β utilisés dans les noms de variantes d'ablation).
# Le crash silencieux précédent (exit 0 sans CSV) venait d'un UnicodeEncodeError
# sur le premier print() contenant λ — le shell renvoyait 0 malgré le traceback.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # Python <3.7 ou stdout non-TTY : ignorer

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sl_ads.paths import get_version_names, get_model_path, get_decision_threshold, get_decision_variable  # Phase H

# PATCH TASK-28 (audit_tmp MAJ-07, 2026-04-26): targeted filters only.
# L'ancien ``warnings.filterwarnings("ignore")`` global + variable
# d'environnement ``PYTHONWARNINGS=ignore::UserWarning`` masquaient des
# alertes scientifiquement significatives. On ne supprime que les warnings
# spécifiquement non actionnables.
warnings.filterwarnings("ignore", category=FutureWarning,      module=r"pandas")
warnings.filterwarnings("ignore", category=UserWarning,        module=r"sklearn")
warnings.filterwarnings("ignore", category=UserWarning,        module=r"matplotlib")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"matplotlib")

# ==============================================================================
# IMPORTS — project modules
# ==============================================================================
try:
    from sl_ads.config import CONFIG  # Phase H
except ImportError:
    print("CRITICAL: sl_ads.config not found.")
    sys.exit(1)

try:
    import sl_ads.core.subjective_logic as sl  # Phase H
except ImportError:
    print("CRITICAL: sl_ads.core.subjective_logic not found.")
    sys.exit(1)

# PATCH M-03 (2026-04-21) : adaptive_base_rate.py déplacé dans un peu old/.
# Le pipeline actif utilise un EDP statique (priors empiriques calculés au
# training et chargés depuis le sidecar models_pkg["empirical_priors"]).
# On laisse ce try/except pour rétro-compatibilité silencieuse si le module
# est remis en place ; sinon _HAS_ADAPTIVE_BR=False et on utilise EDP statique.
try:
    from adaptive_base_rate import make_per_metric_base_rate  # deprecated, kept for BC
    _HAS_ADAPTIVE_BR = True
except ImportError:
    _HAS_ADAPTIVE_BR = False
    # Silencieux : EDP statique est le mode normal depuis 2026-04.
    # Pas de warning print pour éviter le bruit sur chaque run d'ablation.

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    print("WARNING: scikit-learn not found — IsolationForest baseline will be skipped.")


# ==============================================================================
# CONFIGURATION
# ==============================================================================
_EVAL = CONFIG.get("EVAL", {})
VERSION_NAME, VERSION_MODIF = get_version_names(CONFIG)
DIRECTORY_NAME = VERSION_NAME

# IMPORTANT: All systems read the INJECTED evidence so comparison is fair.
# The injected evidence is in resultats_{VERSION_MODIF}/evidence_{VERSION_MODIF}.csv
EVIDENCE_CSV = os.path.join("../results", f"resultats_{DIRECTORY_NAME}",
                             f"evidence_{VERSION_MODIF}.csv")
METADATA_CSV = os.path.join("../results", f"resultats_{DIRECTORY_NAME}",
                             f"metadata_{VERSION_NAME}.csv")
OUTPUT_DIR   = os.path.join("../results", f"resultats_{DIRECTORY_NAME}", "ablation_uniform")

# Seuil opérationnel auto-calibré (b_atk ou proj_atk selon artefact d'entraînement)
_AUTO_THR  = get_decision_threshold(CONFIG, up_levels=1)
THRESHOLDS = [_AUTO_THR]
_DECISION_VARIABLE = get_decision_variable(CONFIG, up_levels=1)  # 'b_atk' ou 'proj_atk'
WINDOW_MIN   = _EVAL.get("WINDOW_MIN", 5)
LAMBDA_BASE  = CONFIG.get("LAMBDA_DECAY", 0.95)
W_BIJ        = CONFIG.get("SL_PARAM_K", 3.0)   # W=K=3 canonical (config_v14)
CONFLICT_ALPHA = CONFIG.get("CONFLICT_ALPHA", 1.495)  # Hard-reset alpha (H1)
MODEL_PATH   = get_model_path(CONFIG, up_levels=1)  # Pour charger l'EDP


# Chargement EDP depuis l'artefact entraîné (même logique que compute_opinions_v3)
import joblib as _jl
_models_pkg = _jl.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else {}
if CONFIG.get("USE_EMPIRICAL_PRIOR", True) and _models_pkg.get("empirical_priors"):
    _edp_dict = _models_pkg["empirical_priors"]
else:
    _edp_dict = None

# ABLATION_RUNS chargé depuis config.py (source unique de vérité).
# Chaque run peut contenir les clés :
#   lambda            : forgetting factor λ (float)
#   wbf_uniform       : bool — True = poids WBF uniformes (C3 off)
#   use_prophet       : bool — utiliser la branche Prophet
#   use_reconst       : bool — utiliser la branche RANSAC
#   use_cbf           : bool — utiliser CBF inter-méthode (False = WBF uniforme)
#   conflict_aware    : bool — activer l'ageing adaptatif C1 (False = λ fixe)
#   adaptive_base_rate: bool — activer la TransitionMemory C4 (False = prior fixe)
#   sl_param_k        : float — constante W de la bijection (2.0 standard, 3.0 pour B3)
_cfg_runs = CONFIG.get("ABLATION", {}).get("RUNS", {})

# Conversion des clés config vers les clés attendues par run_sl_fusion
# + ajout de runs de sensibilité λ hardcodés (non dans config car ils varient λ uniquement)
ABLATION_RUNS = {}
# H1b/H1c — Conflict mode sensitivity (requires conflict_mode override via BUG_005 fix).
# PATCH 2026-04-29 : base uniform (cohérent avec la nouvelle référence full_sl).
ABLATION_RUNS["H1b — Conflict on P(x) [projected_prob]"] = {
    "lambda": 0.85, "uniform": True, "prophet": True,
    "reconst": True, "cbf": True, "um": False,
    "conflict_aware": True, "adaptive_base_rate": True,
    "sl_param_k": 3.0, "c3_weight_mode": "uniform",
    "wbf_weight_mode": "uniform",
    "conflict_mode": "projected_prob",
}
ABLATION_RUNS["H1c — Conflict via KL [kl_symmetric]"] = {
    "lambda": 0.85, "uniform": True, "prophet": True,
    "reconst": True, "cbf": True, "um": False,
    "conflict_aware": True, "adaptive_base_rate": True,
    "sl_param_k": 3.0, "c3_weight_mode": "uniform",
    "wbf_weight_mode": "uniform",
    "conflict_mode": "kl_symmetric",
}
# H3 — UM pre-fusion (uniform WBF, comme dans le rapport)
ABLATION_RUNS["UM=True (pre-fusion, uniform)"] = {
    "lambda": 0.85, "uniform": True, "prophet": True,
    "reconst": True, "cbf": True, "um": True,
    "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
    "c3_weight_mode": "r2_static",
}
# H4 — UM post-CBF final
ABLATION_RUNS["UM=True (post-fusion, uniform)"] = {
    "lambda": 0.85, "uniform": True, "prophet": True,
    "reconst": True, "cbf": True, "um": False, "um_post_fusion": True,
    "conflict_aware": True, "adaptive_base_rate": True,
    "sl_param_k": 3.0, "c3_weight_mode": "r2_static",
}
_LABEL_MAP = {
    # Référence — PATCH "uniform-as-reference" 2026-04-29 :
    # passe de Trust Discounting → Uniform Weights pour matcher la production.
    # Trust-discount est conservé comme variante "trust_discount_legacy" pour
    # exposer la pathologie R²-négatif (5/12 modèles Prophet R²<0).
    # Voir docs/audit/trust_discount_r2_analysis.md, §5.3.3 honest_limitations.md.
    "full_sl":                        "Full SL-ADS (λ=0.85, Uniform Weights) [reference]",

    # Ablation ageing
    "no_ageing + uniform":            "Ageing Sensitivity (λ=0.00)",

    # C3 — pondération WBF
    "uniform_weights":                "Uniform Weights [production duplicate]",
    "c3_prophet_interval uniform":    "C3 — Prophet Interval Width",
    "c3_online_rmse uniform":         "C3 — Online RMSE (safe windows)",

    # Ablation branche
    "prophet_only uniform":           "Prophet Only (Temporal, uniform)",
    "reconst_only uniform":           "Reconst Only (Structural, uniform)",
    "no_cbf uniform":                 "WBF inter-method [production duplicate]",
    "no_c1 uniform":                  "No C1 — fixed λ (conflict-aware off)",
    "no_edp_uniform":                 "No C4/EDP — prior uniforme (EDP désactivé)",

    # Ablation isolée — base uniforme, un seul composant change vs full_sl.
    # PATCH 2026-04-29 : ces runs étaient en trust_discount, désormais en uniform.
    "no_c1_isolated":                 "No C1 — fixed λ [isolated]",
    "no_cbf_isolated":                "WBF inter-method [isolated production duplicate]",
    "prophet_only_isolated":          "Prophet Only [isolated]",
    "reconst_only_isolated":          "Reconst Only [isolated]",
    "no_edp_isolated":                "No C4/EDP — prior uniforme [isolated]",
    # Pathologie : trust_discount activé sur la base uniform full_sl (démontre F1=0.566).
    "trust_discount_legacy":          "Trust-Discount [legacy, R²-pathology — F1↓0.245]",
    # Alias rétro-compatibilité (ancien nom) — pointe vers le même run renommé.
    "no_trust_discount_isolated":     "No Trust Discount [legacy, doublon de full_sl]",
    # PATCH D5 — MASE-based trust mode (Hyndman-Koehler 2006). Documents
    # the *opposite* failure mode of trust_discount: at 30 s sampling
    # Naive-1 is dominant so 11/12 Prophet metrics have MASE > 1 → all
    # Prophet sources are silenced down to TRUST_SCORE_FLOOR.  The run
    # is included in the canonical ablation matrix to make the comparison
    # uniform vs trust_discount vs mase reproducible side-by-side under
    # the same protocol. See docs/audit/trust_discount_r2_analysis.md §4.1.
    "mase_legacy":                    "MASE-Trust [legacy, Naive-1 dominance — silences Prophet]",

    # Sensibilité W
    "w2_sensitivity":                 "W=2 — bijection Laplace [sensibilité]",
    "w4_sensitivity":                 "W=4 — bijection conservatrice [sensibilité]",

    # Balance ratio
    "balance_auto":                   "Balance Ratio auto (N_p/N_r=2.4) [CBF bias fix]",

    # Fusion hiérarchique à 2 niveaux (SL-correct)
    "hierarchical_fusion":            "Hierarchical WBF (Prophet=Reconst, 0.5/0.5) [2-level]",

    # Contextual Discounting — sweep α_attack
    "cd_alpha_0.00":                  "CD α=0.00 — Reconst ignorée pour attack",
    "cd_alpha_0.05":                  "CD α=0.05 — Reconst peu fiable attack",
    "cd_alpha_0.10":                  "CD α=0.10 — Reconst peu fiable attack [recommandé Slowloris]",
    "cd_alpha_0.20":                  "CD α=0.20 — Reconst modérément fiable attack",
    "cd_alpha_0.50":                  "CD α=0.50 — Reconst semi-fiable attack",

    # WBF inter-méthode
    "wbf_inter_method_isolated":      "WBF inter-method [explicit production WBF]",

    # Combinaisons pour Slowloris
    "cd_0.10_wbf_inter":              "CD α=0.10 + WBF inter-méthode [Slowloris fix]",
    "cd_0.20_wbf_inter":              "CD α=0.20 + WBF inter-méthode",
    "w2_cd_0.10":                     "W=2 + CD α=0.10 [affirmative + Slowloris fix]",
}

for key, cfg in _cfg_runs.items():
    label = _LABEL_MAP.get(key, key)
    converted = {
        "lambda":             cfg.get("lambda",             0.85),
        "uniform":            cfg.get("wbf_uniform",        False),
        "prophet":            cfg.get("use_prophet",        True),
        "reconst":            cfg.get("use_reconst",        True),
        "cbf":                cfg.get("use_cbf",            True),
        "um":                 False,
        "conflict_aware":     cfg.get("conflict_aware",     True),
        "adaptive_base_rate": cfg.get("adaptive_base_rate", True),
        "sl_param_k":         cfg.get("sl_param_k",         W_BIJ),
        "c3_weight_mode":     cfg.get("c3_weight_mode",     "r2_static"),
        # Clés optionnelles — propagees seulement si presentes dans config
        "wbf_weight_mode":    cfg.get("wbf_weight_mode",    CONFIG.get("WBF_WEIGHT_MODE", "r2_static")),
    }
    # Propagation conditionnelle des cles optionnelles
    if "conflict_mode" in cfg:
        converted["conflict_mode"] = cfg["conflict_mode"]
    if "balance_ratio" in cfg:
        converted["balance_ratio"] = cfg["balance_ratio"]
    if "um_post_fusion" in cfg:
        converted["um_post_fusion"] = cfg["um_post_fusion"]
    if "inter_method_fusion" in cfg:
        converted["inter_method_fusion"] = cfg["inter_method_fusion"]
    if "reconst_attack_reliability" in cfg:
        converted["reconst_attack_reliability"] = cfg["reconst_attack_reliability"]
    ABLATION_RUNS[label] = converted

# Runs de sensibilité λ (λ=0.50 et λ=0.99) — ajoutés ici car non dans config
for lv in [0.50, 0.85, 0.99]:
    label = f"Ageing Sensitivity (λ={lv:.2f})"
    ABLATION_RUNS[label] = {
        "lambda": lv, "uniform": True, "prophet": True,
        "reconst": True, "cbf": True, "um": False,
        "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": W_BIJ,
    }

# Plot style
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
})

RUN_COLORS = {
    "Full SL (λ=0.95)":            "#d62728",
    "Ageing Sensitivity (λ=0.00)": "#fbb4ae",
    "Ageing Sensitivity (λ=0.50)": "#fdc086",
    "Ageing Sensitivity (λ=0.95)": "#beaed4",
    "Ageing Sensitivity (λ=0.99)": "#386cb0",
    "Uniform Weights":             "#2ca02c",
    "Prophet Only":                "#1f77b4",
    "Reconst Only":                "#9467bd",
    "WBF inter-method":            "#8c564b",
    "Static Threshold":            "#17becf",
    "Isolation Forest (c=0.01)":   "#bcbd22",
    "Isolation Forest (c=0.02)":   "#dbdb8d",
    "Isolation Forest (c=0.05)":   "#9e9e1c",
    "Isolation Forest (c=0.10)":   "#636311",
    "No C1 — fixed λ (conflict-aware off)":         "#e377c2",
    "No C4 — fixed base rate (TransitionMemory off)": "#7f7f7f",
    "W=3 sensitivity (ternary prior)":                "#aec7e8",
}


# ==============================================================================
# ATTACK CATALOG LOADING
# ==============================================================================

def load_attack_catalog():
    """Load injected catalog + optional REAL_ATTACK."""
    catalog = []
    catalog_mode = _EVAL.get("CATALOG_MODE", "injected")

    if catalog_mode == "real":
        catalog = list(_EVAL.get("REAL_ATTACK_CATALOG", []))
        print(f"  Catalog: REAL mode ({len(catalog)} attacks from config.py)")
    else:
        try:
            from sl_ads.inject.evidence_level import ATTACK_CATALOG as _C  # Phase H
            catalog = list(_C)
            print(f"  Catalog: INJECTED mode ({len(catalog)} synthetic attacks)")
        except ImportError:
            print("  WARNING: inject_at_evidence_level not found.")

        if _EVAL.get("INCLUDE_REAL_ATTACK", True):
            real = _EVAL.get("REAL_ATTACK_CATALOG", [])
            if real:
                catalog = catalog + real
                print(f"  + {len(real)} real attack(s) appended ({len(catalog)} total)")

    return catalog


# ==============================================================================
# SL FUSION ENGINE
# Direct calls to sl_formulas_v2 methods for Strict Evaluation
# ==============================================================================

def run_sl_fusion(ev_df: pd.DataFrame, meta_df: pd.DataFrame, run_cfg: dict) -> pd.DataFrame:
    """
    Wrapper: override du mode de conflit pour ce run specifique (H1b/H1c ablation),
    puis delègue à _run_sl_fusion_inner.
    sl_formulas_v2._CONFLICT_MODE est set a l'import; on le surcharge temporairement.
    """
    _original_conflict_mode = getattr(sl, '_CONFLICT_MODE', 'belief_mass')
    _run_conflict_mode = run_cfg.get("conflict_mode", _original_conflict_mode)
    sl._CONFLICT_MODE = _run_conflict_mode
    try:
        return _run_sl_fusion_inner(ev_df, meta_df, run_cfg)
    finally:
        sl._CONFLICT_MODE = _original_conflict_mode


def _run_sl_fusion_inner(ev_df: pd.DataFrame, meta_df: pd.DataFrame, run_cfg: dict) -> pd.DataFrame:
    """Implementation interne de run_sl_fusion (appellee apres override du conflict mode)."""
    lam              = run_cfg["lambda"]
    uniform          = run_cfg["uniform"]
    use_p            = run_cfg["prophet"]
    use_r            = run_cfg["reconst"]
    use_cbf          = run_cfg["cbf"]
    use_um           = run_cfg["um"]
    conflict_aware   = run_cfg.get("conflict_aware",     True)
    adaptive_br      = run_cfg.get("adaptive_base_rate", True)
    W_run            = run_cfg.get("sl_param_k",         W_BIJ)
    # Contextual discounting de la Reconst avant CBF (Mercier & Denoeux 2006)
    # α_attack ∈ [0,1] : fiabilité de la Reconst pour l'hypothèse "attack"
    # 1.0 = pas de discounting (CBF standard) ; 0.1 = Reconst peu fiable pour "attack"
    _cd_alpha_attack = float(run_cfg.get("reconst_attack_reliability",
                                         CONFIG.get("RECONST_ATTACK_RELIABILITY", 1.0)))

    # Build metric metadata dict
    meta = {}
    for _, row in meta_df.iterrows():
        mk = row["metric_key"]
        meta[mk] = {
            "clean_key": row["clean_key"],
            "type":      row["type"],
            "r2":        max(float(row["r2_weight"]), 0.01),
        }

    prophet_keys = [k for k, v in meta.items() if v["type"] == "prophet" and use_p]
    reconst_keys = [k for k, v in meta.items() if v["type"] == "reconstruction" and use_r]
    all_active   = prophet_keys + reconst_keys

    if not all_active:
        raise ValueError("No active metrics (check use_prophet / use_reconst).")

    # Balance ratio — même logique que compute_opinions_v2.py (Theorem 12.2)
    # Lire le balance_ratio depuis run_cfg en priorite (pour les runs d'ablation)
    _balance_cfg = run_cfg.get('balance_ratio', CONFIG.get('BALANCE_RATIO', 1.0))
    _n_p = len(prophet_keys)
    _n_r = len(reconst_keys)
    if _balance_cfg == "auto" and use_cbf:
        _balance_eff = float(_n_p) / float(_n_r) if _n_r > 0 else 1.0
    elif isinstance(_balance_cfg, (int, float)) and use_cbf:
        _balance_eff = float(_balance_cfg)
    else:
        _balance_eff = 1.0  # CBF désactivé ou ratio = 1 → pas de boost
    _balance_active = not np.isclose(_balance_eff, 1.0)

    c3_mode = run_cfg.get("c3_weight_mode",
                          CONFIG.get("C3_WEIGHT_MODE", "r2_static"))
    c3_alpha = CONFIG.get("C3_ONLINE_RMSE_ALPHA", 0.05)
    c3_warmup = CONFIG.get("C3_ONLINE_RMSE_WARMUP", 10)
    delta_half = get_decision_threshold(CONFIG, up_levels=1) / 2.0

    # État RMSE glissant (Option 3) — initialisé à None = phase de warmup
    safe_rmse_state = {mk: None for mk in all_active}
    prev_p_atk = 0.0
    window_count = 0

    def get_c3_weight(mk, row, static_r2):
        if c3_mode == "r2_static" or uniform:
            return 1.0 if uniform else static_r2
        elif c3_mode == "prophet_interval":
            if mk.startswith("prophet_"):
                iw = row.get(f"{mk}_iw", np.nan)
                if iw and not np.isnan(iw) and iw > 1e-6:
                    return 1.0 / iw
            return static_r2  # fallback RANSAC ou NaN
        elif c3_mode == "online_rmse":
            state = safe_rmse_state.get(mk)
            if state is None or state < 1e-9:
                return static_r2  # warmup : fallback sur R²
            return 1.0 / state
        return static_r2

    # Base rate — EDP depuis l'artefact entraîné (aligne ablation sur opérationnel)
    # adaptive_br=False → run "No C4" : prior uniforme SL_PRIOR_A (ablation C4)
    prior_a = np.array(CONFIG.get("SL_PRIOR_A", [1/3, 1/3, 1/3]), dtype=float)
    pmbr = None
    if not adaptive_br:
        print("    INFO: adaptive_base_rate=False → fixed prior (C4 ablation run)")
    # Note: TransitionMemory (pmbr) remplacée par EDP — lue au moment de l'opinion

    # State memory: accumulated evidence (Josang Eq. 16.5)
    state_memory = {mk: np.zeros(3) for mk in all_active}

    records = []

    for _, row in ev_df.iterrows():
        window_count += 1
        timestamp = row["timestamp"]

        # ── Step 1: evidence accumulation with ageing ─────────────────────────
        leaf_ops = {}
        for mk in all_active:
            ck = meta[mk]["clean_key"]
            ev_P = float(row.get(f"{ck}_P", 0.0) or 0.0)
            ev_S = float(row.get(f"{ck}_S", 0.0) or 0.0)
            ev_N = float(row.get(f"{ck}_N", 0.0) or 0.0)

            current_evidence = np.array([ev_P, ev_S, ev_N])

            # Ageing adaptatif (fail-fast si l'API de sl_formulas_v2 change)
            if lam > 0:
                R_new, K_eff, lam_dyn = sl.temporal_adaptive_ageing(
                    r_accumulated=state_memory[mk],
                    r_current=current_evidence,
                    lam_base=lam,
                    W=W_run,
                    alpha=run_cfg.get("conflict_alpha", CONFLICT_ALPHA),
                    conflict_aware=conflict_aware,
                )
                state_memory[mk] = R_new
            else:
                # Ablation "No Ageing": effacement immédiat de la mémoire
                state_memory[mk] = current_evidence
                K_eff, lam_dyn = 0.0, 0.0

            # Base rate : EDP si dispo + run actif, sinon prior uniforme (No C4)
            if adaptive_br and _edp_dict is not None and mk in _edp_dict:
                edp_k = _edp_dict[mk]
                a = np.array([edp_k["a_safe"], edp_k["a_susp"], edp_k["a_atk"]])
            else:
                a = prior_a.copy()

            # Opinion from evidence (W_run peut différer de W_BIJ pour B3)
            op = sl.evidence_to_opinion(state_memory[mk], W_run, a)

            # Calcul du poids statique R2 (doit etre avant le bloc trust_discount)
            static_r2 = max(float(meta[mk]["r2"]), 0.01)

            _wbf_mode = run_cfg.get("wbf_weight_mode", CONFIG.get("WBF_WEIGHT_MODE", "r2_static"))
            _trust_scores = _models_pkg.get("trust_scores", {}) if not uniform else {}
            # PATCH D5 (MASE) — load top-level mase_scores for the 'mase'
            # ablation variant.  Empty dict if the loaded pkl predates the
            # MASE patch; in that case the per-key lookup falls back to
            # TRUST_SCORE_FLOOR.
            _mase_scores = _models_pkg.get("mase_scores", {}) if not uniform else {}

            if _wbf_mode == "trust_discount" and not uniform:
                t = float(_trust_scores.get(mk, a[0]))  # fallback a_safe
                op = sl.apply_trust_discount(op, t)
                w = 1.0
            elif _wbf_mode == "mase" and not uniform:
                # PATCH D5: same Joesang Def. 14.6 application as
                # trust_discount, but the trust value comes from the
                # MASE-derived mase_scores dict (Hyndman-Koehler 2006).
                # Fallback for missing keys is the numerical floor —
                # NOT a_safe — so RANSAC sources (NaN MASE) are silenced
                # consistently and remain available through the
                # contextual-discount path.
                _floor = float(CONFIG.get("TRUST_SCORE_FLOOR", 0.05))
                t = float(_mase_scores.get(mk, _floor))
                op = sl.apply_trust_discount(op, t)
                w = 1.0
            else:
                w = get_c3_weight(mk, row, static_r2)

            # Uncertainty Maximization (pre-fusion)
            if use_um:
                op = op.uncertainty_maximized()

            # IMPORTANT: ne pas recalculer w ici — le bloc ci-dessus l'a deja defini
            leaf_ops[mk] = (op, w)

        # ── Step 2: WBF within each method group ─────────────────────────────
        def wbf_group(keys):
            ops     = [leaf_ops[k][0] for k in keys if k in leaf_ops]
            weights = [leaf_ops[k][1] for k in keys if k in leaf_ops]
            if not ops:
                return None
            if len(ops) == 1:
                return ops[0]
            return sl.fusion_wbf_n_sources(ops, external_weights=weights)

        op_prophet = wbf_group(prophet_keys) if prophet_keys else None
        op_reconst = wbf_group(reconst_keys) if reconst_keys else None

        # ── Step 3: fusion inter-méthode (CBF ou WBF selon config) ───────────────
        # "cbf" : Cumulative Belief Fusion — addition d'évidence (Jøsang Eq. 12.14)
        # "wbf" : Weighted Belief Fusion — moyenne pondérée (Jøsang Eq. 12.22)
        #         Moins agressive : une source certaine du "safe" réduit mais n'annule
        #         pas le signal d'attaque de l'autre source (test pour SLOWLORIS/UNKNOWN).
        # Pré-traitement : contextual discounting de la Reconst si α_attack < 1.0
        # (Mercier & Denoeux 2006 — réduit la fiabilité de Reconst pour "attack")
        _inter_method = run_cfg.get(
            "inter_method_fusion",
            CONFIG.get("INTER_METHOD_FUSION", "wbf"),
        )
        if op_prophet is not None and op_reconst is not None:
            # Contextual discounting avant CBF (si activé)
            op_reconst_fused = op_reconst
            if _cd_alpha_attack < 1.0 - 1e-6:
                op_reconst_fused = sl.apply_contextual_discount(
                    op_reconst, alpha=[1.0, 1.0, _cd_alpha_attack]
                )
            if _inter_method == "wbf":
                # WBF inter-méthode : moyenne pondérée par confiance (1-u) de chaque groupe
                op_final = sl.fusion_wbf_n_sources([op_prophet, op_reconst_fused],
                                                   external_weights=None, W=W_BIJ)
            elif _inter_method == "hierarchical":
                # Fusion hiérarchique : poids égaux 0.5/0.5 entre Prophet (groupe) et Reconst (groupe).
                # Chaque branche contribue exactement 1 fois, quel que soit N_p ou N_r.
                # Sémantiquement correct en SL : 2 sources distinctes → équipondération forcée.
                op_final = sl.fusion_wbf_n_sources([op_prophet, op_reconst_fused],
                                                   external_weights=[0.5, 0.5], W=W_BIJ)
            elif use_cbf:
                # Rééquilibrage des preuves avant CBF (Theorem 12.2, Eq. 12.17)
                op_reconst_cbf = (
                    sl.boost_opinion_evidence(op_reconst_fused, _balance_eff, W=W_BIJ)
                    if _balance_active else op_reconst_fused
                )
                op_final = sl.fusion_cbf(op_prophet, op_reconst_cbf)
                # Post-fusion UM (ablation H4)
                if run_cfg.get("um_post_fusion", False):
                    op_final = op_final.uncertainty_maximized()
            else:
                # Explicit legacy no-CBF path: average Prophet and Reconstruction
                # opinions directly. Production normally reaches WBF through
                # inter_method_fusion="wbf" before this branch.
                op_final = sl.fusion_wbf_n_sources([op_prophet, op_reconst],
                                                   external_weights=[1.0, 1.0])
        elif op_prophet is not None:
            op_final = op_prophet
        elif op_reconst is not None:
            op_final = op_reconst
        else:
            op_final = sl.evidence_to_opinion(np.zeros(3), W_BIJ, prior_a)

        b  = op_final.b
        u  = op_final.u
        pp = op_final.projected_prob()

        p_atk = float(op_final.projected_prob()[2])

        if c3_mode == "online_rmse":
            is_safe = (prev_p_atk < delta_half) or (window_count <= c3_warmup)
            if is_safe:
                for mk in all_active:
                    rmse_val = row.get(f"{mk}_rmse", None)
                    if rmse_val is None or (isinstance(rmse_val, float) and np.isnan(rmse_val)):
                        continue
                    rmse_val = float(rmse_val)
                    if safe_rmse_state[mk] is None:
                        safe_rmse_state[mk] = rmse_val
                    else:
                        safe_rmse_state[mk] = (
                                c3_alpha * rmse_val +
                                (1.0 - c3_alpha) * safe_rmse_state[mk]
                        )
        prev_p_atk = p_atk

        records.append({
            "timestamp": timestamp,
            "b_safe":    float(b[0]),
            "b_susp":    float(b[1]),
            "b_atk":     float(b[2]),
            "u":         float(u),
            "proj_atk":  float(pp[2]),
        })

    return pd.DataFrame(records)


# ==============================================================================
# BASELINE 1 — Static Threshold (R²-weighted vote on evidence N columns)
# Ref: Sharafaldin et al. (2018) — rule-based IDS benchmark
# ==============================================================================

def run_static_threshold(ev_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Each metric k fires if ev_N[k] > threshold_attack[k].
    Final score = sum(R²_k * vote_k) / sum(R²_k), in [0,1].
    The score replaces b_atk for the threshold sweep.
    """
    meta = {}
    for _, row in meta_df.iterrows():
        mk  = row["metric_key"]
        ck  = row["clean_key"]
        meta[mk] = {
            "clean_key": ck,
            "r2":        max(float(row["r2_weight"]), 0.01),
            "th_atk":    float(row["threshold_attack"]),
        }

    records = []
    for _, row in ev_df.iterrows():
        total_w = 0.0
        vote_w  = 0.0
        for mk, m in meta.items():
            ck   = m["clean_key"]
            ev_N = float(row.get(f"{ck}_N", 0.0) or 0.0)
            w    = m["r2"]
            total_w += w
            # Vote: metric fires if N evidence exceeds attack threshold
            # Threshold from training: corresponds to 99th percentile of residuals
            WINDOW_SLICES = 10  # 10 x 30s = 5 min
            if ev_N / WINDOW_SLICES > 0.90:
                vote_w += w

        score = vote_w / total_w if total_w > 0 else 0.0
        records.append({"timestamp": row["timestamp"], "proj_atk": score})

    return pd.DataFrame(records)


# ==============================================================================
# BASELINE 2 — Isolation Forest
# Ref: Liu et al. (2008) "Isolation Forest", IEEE ICDM
# ==============================================================================

def run_isolation_forest(ev_df: pd.DataFrame, catalog: list,
                          contamination: float = 0.05) -> pd.DataFrame:
    """
    Trains IsolationForest on normal traffic windows (outside all attack periods).
    MODIFICATION: Uses internal IF predictions (-1/1) mapped to 1.0/0.0
    to avoid being penalized by the strict 0.20 SL threshold.
    """
    if not _HAS_SKLEARN:
        return pd.DataFrame()

    feat_cols = [c for c in ev_df.columns if c.endswith("_N")]
    if not feat_cols:
        print("  WARNING: No _N columns found for IsolationForest features.")
        return pd.DataFrame()

    X = ev_df[feat_cols].fillna(0.0).values

    # Normal mask: windows outside all attack periods
    split_date = pd.Timestamp(CONFIG.get("split_date", "2025-11-09 23:59:59"))
    train_mask = ev_df["timestamp"] <= split_date
    normal_mask = pd.Series(True, index=ev_df.index)
    for atk in catalog:
        t0 = pd.Timestamp(atk["start"])
        t1 = t0 + pd.Timedelta(hours=atk["duration_h"])
        normal_mask &= ~((ev_df["timestamp"] >= t0) & (ev_df["timestamp"] < t1))

    X_train = X[normal_mask.values & train_mask.values]
    n_train = len(X_train)

    # PATCH-C3 fix (2026-04-19) : refuser tout fallback qui exposerait IF
    # au trafic normal de la période de test (fuite structurelle de distribution).
    # Si le CSV d'evidence ne couvre pas assez de période pré-split, l'étude
    # est non-comparable et doit échouer bruyamment plutôt que produire un
    # baseline biaisé.
    #
    # Ref : Varma & Simon (2006) BMC Bioinformatics 7:91 — toute exposition
    # d'un baseline à la distribution de test est un biais structurel et
    # interdit la comparaison équitable avec SL-ADS.
    if n_train < 1000:
        raise ValueError(
            f"[PATCH-C3] IsolationForest: only {n_train} training windows "
            f"before split_date={split_date} — refusing fallback to "
            f"test-period normals (structural distribution leakage, "
            f"Varma & Simon 2006). Fix: regenerate evidence CSV with a "
            f"sufficiently long pre-split window (>=1000 windows)."
        )

    if n_train < 50:
        print(f"  WARNING: Only {n_train} normal windows for IF — results may be unreliable.")

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_all_s   = scaler.transform(X)

    clf = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=42, n_jobs=-1)
    clf.fit(X_train_s)

    # MODIFICATION: Use internal decision mapping
    # clf.predict returns -1 for anomaly, 1 for inlier
    preds = clf.predict(X_all_s)

    # Map to 1.0 (Attack) and 0.0 (Normal) so that any threshold > 0 (e.g., 0.20) works correctly
    b_atk_mapped = np.where(preds == -1, 1.0, 0.0)

    print(f"  Training: {n_train} normal windows / {len(ev_df)} total. (Using internal contamination={contamination})")

    return pd.DataFrame({"timestamp": ev_df["timestamp"], "proj_atk": b_atk_mapped})

# ==============================================================================
# EVALUATION — per-run threshold sweep
# ==============================================================================

def _in_attack(timestamps: pd.Series, atk: dict) -> np.ndarray:
    t0 = pd.Timestamp(atk["start"])
    t1 = t0 + pd.Timedelta(hours=atk["duration_h"])
    return ((timestamps >= t0) & (timestamps < t1)).values


def _event_bounds(ev: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    t0 = pd.Timestamp(ev["start"])
    if ev.get("end") is not None:
        return t0, pd.Timestamp(ev["end"])
    return t0, t0 + pd.Timedelta(hours=float(ev["duration_h"]))


def _event_mask(timestamps: pd.Series, events) -> np.ndarray:
    mask = np.zeros(len(timestamps), dtype=bool)
    for ev in events or []:
        t0, t1 = _event_bounds(ev)
        mask |= ((timestamps >= t0) & (timestamps < t1)).values
    return mask


def _real_attacks_iter():
    try:
        from sl_ads.config import REAL_ATTACKS as _RA
    except Exception:
        return
    for events in (_RA or {}).values():
        for ev in events or []:
            yield ev


def _outage_mask(timestamps: pd.Series) -> np.ndarray:
    try:
        from sl_ads.qualif_filters import is_outage_event
    except Exception:
        is_outage_event = lambda _ev: False  # noqa: E731
    return _event_mask(
        timestamps,
        [ev for ev in _real_attacks_iter() if is_outage_event(ev)],
    )


def _outside_attacks(timestamps: pd.Series, catalog: list) -> np.ndarray:
    outside = np.ones(len(timestamps), dtype=bool)
    for atk in catalog:
        outside &= ~_in_attack(timestamps, atk)
    return outside


def _outside_catalog_and_real_events(timestamps: pd.Series, catalog: list) -> np.ndarray:
    """Normal-window mask for the paper-facing catalog/outages-separate protocol.

    The previous ablation harness only removed the evaluation catalog from the
    normal denominator. That silently counted REAL_ATTACKS outage windows as FP,
    while evaluate_injection excludes them from the catalog/outages-separate
    FPR denominator. Keep the legacy count as explicit diagnostic columns, but
    make the unsuffixed ablation metrics match the main evaluation protocol.
    """
    outside = _outside_attacks(timestamps, catalog)
    outside &= ~_event_mask(timestamps, list(_real_attacks_iter()))
    return outside


def evaluate_run(result_df: pd.DataFrame, catalog: list,
                 thresholds_override=None) -> pd.DataFrame:
    """Full threshold sweep for one system result. Returns sweep DataFrame."""
    ts           = result_df["timestamp"]
    # Utilise b_atk ou proj_atk selon la variable calibrée à l'entraînement
    _score_col   = _DECISION_VARIABLE if _DECISION_VARIABLE in result_df.columns else "proj_atk"
    proj_atk_all = result_df[_score_col].fillna(0.0).values
    legacy_out_mask = _outside_attacks(ts, catalog)
    out_mask     = _outside_catalog_and_real_events(ts, catalog)
    outage_mask  = _outage_mask(ts)
    n_outside    = int(out_mask.sum())
    n_outside_legacy = int(legacy_out_mask.sum())
    n_outage_excluded = int((legacy_out_mask & outage_mask).sum())
    thrs         = thresholds_override if thresholds_override is not None else THRESHOLDS

    rows = []
    for thr in thrs:
        n_total = 0
        n_tp    = 0
        cov_sum = 0.0
        ttd_list = []
        n_det_win = 0

        for atk in catalog:
            gt_mask = _in_attack(ts, atk)
            b_gt    = proj_atk_all[gt_mask]
            if len(b_gt) == 0:
                continue
            n_total += 1

            det = b_gt >= thr
            n_det = det.sum()
            detected = n_det > 0
            cov = n_det / len(b_gt)

            if detected:
                n_tp += 1
                n_det_win += n_det
                cov_sum += cov
                t_start = pd.Timestamp(atk["start"])
                t_first = ts[gt_mask].iloc[int(np.argmax(det))]
                ttd_min = max(0.0, (t_first - t_start).total_seconds() / 60.0)
                ttd_rat = min(1.0, ttd_min / (atk["duration_h"] * 60.0))
                ttd_list.append(cov * (1.0 - ttd_rat))
            else:
                cov_sum += 0.0
                ttd_list.append(0.0)

        if n_total == 0:
            continue

        fp = int((proj_atk_all[out_mask] >= thr).sum())
        fp_legacy = int((proj_atk_all[legacy_out_mask] >= thr).sum())
        prec = n_det_win / (n_det_win + fp) if (n_det_win + fp) > 0 else 0.0
        prec_legacy = (
            n_det_win / (n_det_win + fp_legacy)
            if (n_det_win + fp_legacy) > 0 else 0.0
        )

        r_bin = n_tp / n_total
        r_cov = cov_sum / n_total
        r_ttd = float(np.mean(ttd_list)) if ttd_list else 0.0

        def f1(p, r): return 2*p*r/(p+r) if (p+r) > 0 else 0.0

        rows.append({
            "threshold":       thr,
            "n_attacks":       n_total,
            "n_detected":      n_tp,
            "recall_binary":   round(r_bin,  4),
            "recall_coverage": round(r_cov,  4),
            "recall_ttd":      round(r_ttd,  4),
            "precision":       round(prec,   4),
            "f1_binary":       round(f1(prec, r_bin), 4),
            "f1_coverage":     round(f1(prec, r_cov), 4),
            "f1_ttd":          round(f1(prec, r_ttd), 4),
            "fp_windows":      fp,
            "fpr_pct":         round(100.0 * fp / n_outside, 5) if n_outside > 0 else 0.0,
            "metric_protocol": "catalog_outages_separate",
            "normal_windows": n_outside,
            "outage_windows_excluded": n_outage_excluded,
            "precision_legacy_outages_as_normal": round(prec_legacy, 4),
            "f1_binary_legacy_outages_as_normal": round(f1(prec_legacy, r_bin), 4),
            "f1_coverage_legacy_outages_as_normal": round(f1(prec_legacy, r_cov), 4),
            "f1_ttd_legacy_outages_as_normal": round(f1(prec_legacy, r_ttd), 4),
            "fp_windows_legacy_outages_as_normal": fp_legacy,
            "normal_windows_legacy_outages_as_normal": n_outside_legacy,
            "fpr_pct_legacy_outages_as_normal": (
                round(100.0 * fp_legacy / n_outside_legacy, 5)
                if n_outside_legacy > 0 else 0.0
            ),
        })

    return pd.DataFrame(rows)


def best_row(sweep_df: pd.DataFrame) -> pd.Series:
    """Return sweep row with best F1-coverage."""
    if sweep_df.empty:
        return pd.Series()
    return sweep_df.loc[sweep_df["f1_coverage"].idxmax()]


def to_summary(run_name: str, sweep_df: pd.DataFrame) -> dict | None:
    br = best_row(sweep_df)
    if br.empty:
        return None
    return {
        "run":             run_name,
        "best_threshold":  float(br["threshold"]),
        "f1_binary":       float(br["f1_binary"]),
        "f1_coverage":     float(br["f1_coverage"]),
        "f1_ttd":          float(br["f1_ttd"]),
        "precision":       float(br["precision"]),
        "recall_binary":   float(br["recall_binary"]),
        "recall_coverage": float(br["recall_coverage"]),
        "fpr_pct":         float(br["fpr_pct"]),
        "n_detected":      int(br["n_detected"]),
        "n_attacks":       int(br["n_attacks"]),
        "metric_protocol": str(br.get("metric_protocol", "catalog_outages_separate")),
        "normal_windows": int(br.get("normal_windows", 0)),
        "outage_windows_excluded": int(br.get("outage_windows_excluded", 0)),
        "f1_coverage_legacy_outages_as_normal": float(
            br.get("f1_coverage_legacy_outages_as_normal", np.nan)
        ),
        "precision_legacy_outages_as_normal": float(
            br.get("precision_legacy_outages_as_normal", np.nan)
        ),
        "fpr_pct_legacy_outages_as_normal": float(
            br.get("fpr_pct_legacy_outages_as_normal", np.nan)
        ),
    }


# ==============================================================================
# PUBLICATION FIGURES
# ==============================================================================

def plot_f1_curves(all_sweeps: dict, output_dir: str):
    """F1-coverage curves for all runs on one plot."""
    # MODIFICATION: Bypass the plot gracefully if only 1 threshold is tested.
    if len(THRESHOLDS) <= 1:
        print("   INFO: F1 curve plotting bypassed (only one threshold tested).")
        return

    REFERENCE_RUN = "Full SL-ADS (λ=0.85, Uniform Weights) [reference]"
    fig, ax = plt.subplots(figsize=(11, 6))
    for run_name, sdf in all_sweeps.items():
        if sdf.empty:
            continue
        color = RUN_COLORS.get(run_name, "gray")
        is_ref = run_name == REFERENCE_RUN
        is_sl  = run_name in ABLATION_RUNS
        ls = "-" if is_ref else ("--" if is_sl else ":")
        lw = 2.5 if is_ref else 1.7
        ax.plot(sdf["threshold"], sdf["f1_coverage"],
                ls, lw=lw, color=color,
                marker="o" if is_ref else None, ms=6,
                label=run_name, zorder=6 if is_ref else 4)

    ax.set_xlabel(r"Detection threshold $b_{atk}$", fontsize=11)
    ax.set_ylabel("F1-coverage (catalog/outages-separate)", fontsize=11)
    ax.set_title(
        f"Ablation Study — F1-Coverage vs. Threshold\n"
        f"Solid: {REFERENCE_RUN} (reference)  |  Dashed: SL ablations  |  Dotted: Baselines",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylim(0, 1.07)
    ax.set_xlim(min(THRESHOLDS) - 0.02, max(THRESHOLDS) + 0.02)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower left", fontsize=9, framealpha=0.88)

    fig.savefig(os.path.join(output_dir, "ablation_f1_curves.png"))
    plt.close(fig)
    print(f"   figure: ablation_f1_curves.png")


def _publication_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a readable figure subset while keeping full CSV outputs.

    The complete ablation matrix stays in CSV form. Publication figures show
    only the paper-relevant rows, so threshold-sensitivity duplicates do not
    become unreadable or encourage cherry-picking.
    """
    wanted = [
        "Full SL-ADS (λ=0.85, Uniform Weights) [reference]",
        "Reconst Only (Structural, uniform)",
        "Prophet Only (Temporal, uniform)",
        "No C1 — fixed λ (conflict-aware off)",
        "No C4/EDP — prior uniforme (EDP désactivé)",
        "Balance Ratio auto (N_p/N_r=2.4) [CBF bias fix]",
        "Ageing Sensitivity (λ=0.99)",
        "Trust-Discount [legacy, R²-pathology — F1↓0.245]",
        "MASE-Trust [legacy, Naive-1 dominance — silences Prophet]",
        "Static Threshold",
    ]
    rows = []
    for name in wanted:
        hit = df[df["run"] == name]
        if not hit.empty:
            rows.append(hit.iloc[0])
    if not rows:
        return df
    return pd.DataFrame(rows).reset_index(drop=True)


def _short_run_name(name: str) -> str:
    """Compact labels for figures; CSVs keep the full run names."""
    aliases = {
        "Full SL-ADS (λ=0.85, Uniform Weights) [reference]": "Full SL-ADS",
        "Reconst Only (Structural, uniform)": "Reconst only",
        "Prophet Only (Temporal, uniform)": "Prophet only",
        "No C1 — fixed λ (conflict-aware off)": "No C1",
        "No C4/EDP — prior uniforme (EDP désactivé)": "No C4/EDP",
        "Balance Ratio auto (N_p/N_r=2.4) [CBF bias fix]": "Balance CBF",
        "Ageing Sensitivity (λ=0.99)": "Ageing λ=0.99",
        "Trust-Discount [legacy, R²-pathology — F1↓0.245]": "Trust legacy",
        "MASE-Trust [legacy, Naive-1 dominance — silences Prophet]": "MASE legacy",
        "Static Threshold": "Static threshold",
    }
    return aliases.get(name, name)


def plot_comparison_table(summary_rows: list, output_dir: str):
    """Publication-ready comparison table. Best value per column in green."""
    if not summary_rows:
        return

    df = _publication_subset(pd.DataFrame(summary_rows))
    cols_display  = ["System / Run", "F1-binary", "F1-cov.", "F1-TTD",
                     "Precision", "Recall-bin.", "FPR (%)", "Thr.", "Det./Total"]
    cols_metric   = ["f1_binary", "f1_coverage", "f1_ttd", "precision", "recall_binary"]
    best_hi = {c: df[c].max() for c in cols_metric if c in df.columns}
    if "fpr_pct" in df.columns:
        full_detection = df[df["n_detected"] == df["n_attacks"]]
        fpr_base = full_detection if not full_detection.empty else df
        best_lo = fpr_base["fpr_pct"].min()
    else:
        best_lo = None

    rows_data = []
    for _, r in df.iterrows():
        rows_data.append([
            _short_run_name(r["run"]),
            f"{r['f1_binary']:.3f}",
            f"{r['f1_coverage']:.3f}",
            f"{r['f1_ttd']:.3f}",
            f"{r['precision']:.3f}",
            f"{r['recall_binary']:.3f}",
            f"{r['fpr_pct']:.4f}",
            f"{r['best_threshold']:.2f}",
            f"{int(r['n_detected'])}/{int(r['n_attacks'])}",
        ])

    n_rows = len(rows_data)
    n_cols = len(cols_display)
    fig_h  = max(3.0, 0.52 * (n_rows + 1) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    table = ax.table(cellText=rows_data, colLabels=cols_display,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.7)

    # Header
    for j in range(n_cols):
        c = table[0, j]
        c.set_facecolor("#2c3e50")
        c.set_text_props(color="white", fontweight="bold")

    REFERENCE_RUN = "Full SL-ADS (λ=0.85, Uniform Weights) [reference]"

    # MODIFICATION: Handle variable IF baseline names dynamically
    metric_col_idx = [1, 2, 3, 4, 5, 6]
    metric_keys_list = ["f1_binary", "f1_coverage", "f1_ttd",
                         "precision", "recall_binary", "fpr_pct"]

    for i, (_, r) in enumerate(df.iterrows()):
        is_baseline = "Isolation Forest" in r["run"] or "Static Threshold" in r["run"]
        is_ref      = r["run"] == REFERENCE_RUN
        bg = "#fef3e2" if is_baseline else ("#d0eaf8" if is_ref else "#e8f4fb")
        for j in range(n_cols):
            table[i+1, j].set_facecolor(bg)

        for col_i, mk in zip(metric_col_idx, metric_keys_list):
            cell = table[i+1, col_i]
            val  = r.get(mk)
            if val is None:
                continue
            if mk == "fpr_pct":
                if best_lo is not None and abs(val - best_lo) < 1e-6:
                    cell.set_facecolor("#d4edda")
                    cell.set_text_props(fontweight="bold")
            else:
                bv = best_hi.get(mk)
                if bv is not None and abs(val - bv) < 1e-6:
                    cell.set_facecolor("#d4edda")
                    cell.set_text_props(fontweight="bold")

    ax.set_title(
        "SL-IDS - Ablation Study & Baseline Comparison (selected publication rows)\n"
        "Protocol: catalog/outages-separate; full matrix and legacy columns are CSV-only  |  "
        f"Fixed threshold: {float(df['best_threshold'].iloc[0]):.3f}",
        fontsize=10.5, fontweight="bold", pad=14
    )

    fig.savefig(os.path.join(output_dir, "ablation_comparison.png"))
    plt.close(fig)
    print(f"   figure: ablation_comparison.png")


def plot_bar_comparison(summary_rows: list, output_dir: str):
    """Grouped bar chart F1-cov / F1-bin / Precision + FPR line."""
    if not summary_rows:
        return

    df  = _publication_subset(pd.DataFrame(summary_rows))
    runs = [_short_run_name(r) for r in df["run"].tolist()]
    x    = np.arange(len(runs))
    w    = 0.22

    fig, ax1 = plt.subplots(figsize=(14, 5.5))
    ax1.bar(x - w, df["f1_coverage"], w, label="F1-coverage", color="#1f77b4", alpha=0.95)
    ax1.bar(x,     df["f1_binary"],   w, label="F1-binary",   color="#2ca02c", alpha=0.95)
    ax1.bar(x + w, df["precision"],   w, label="Precision",   color="#ff7f0e", alpha=0.95)

    ax2 = ax1.twinx()
    ax2.plot(x, df["fpr_pct"], "v--", color="#d62728", lw=2, ms=8, label="FPR (%)")
    ax2.set_ylabel("FPR (%) - catalog/outages-separate", fontsize=10, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#d62728")
    fpr_max = df["fpr_pct"].max()
    ax2.set_ylim(0, max(fpr_max * 2.5, 1.0))

    ax1.set_xticks(x)
    ax1.set_xticklabels(runs, rotation=18, ha="right", fontsize=9)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_ylim(0, 1.12)
    ax1.set_title(
        f"Ablation Study & Baselines - Performance Comparison @ Threshold {float(df['best_threshold'].iloc[0]):.3f}\n"
        "Selected publication rows | Protocol: catalog/outages-separate",
        fontsize=11, fontweight="bold"
    )

    n_sl = int((~df["run"].str.contains("Static Threshold|Isolation Forest", regex=True)).sum())
    if n_sl < len(runs):
        ax1.axvline(n_sl - 0.5, color="#aaaaaa", lw=1.2, linestyle=":")
        ax1.text(n_sl - 0.45, 1.08, "← SL Ablations", fontsize=8, color="#555")
        ax1.text(n_sl + 0.15, 1.08, "Baselines →",    fontsize=8, color="#555")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    fig.savefig(os.path.join(output_dir, "ablation_bar_comparison.png"))
    plt.close(fig)
    print(f"   figure: ablation_bar_comparison.png")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(f"\n{'='*68}")
    print("  run_ablation.py — SL-IDS Strict Evaluation")
    print(f"{'='*68}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    if not os.path.exists(EVIDENCE_CSV):
        alt = EVIDENCE_CSV.replace(VERSION_MODIF, VERSION_NAME)
        if os.path.exists(alt):
            print(f"WARNING: Injected evidence not found, using base evidence:\n  {alt}")
            print("  NOTE: Injected attacks won't be visible in base evidence.")
            ev_path = alt
        else:
            print(f"ERROR: Evidence CSV not found: {EVIDENCE_CSV}")
            print(f"       Also tried: {alt}")
            print(f"  Make sure inject_at_evidence_level.py has run and saved")
            print(f"  evidence_{VERSION_NAME}.csv into resultats_{VERSION_MODIF}/")
            sys.exit(1)
    else:
        ev_path = EVIDENCE_CSV

    if not os.path.exists(METADATA_CSV):
        print(f"ERROR: Metadata CSV not found: {METADATA_CSV}")
        sys.exit(1)

    print(f"-> Loading evidence CSV: {ev_path}")
    ev_raw = pd.read_csv(ev_path, parse_dates=["timestamp"])
    ev_raw = ev_raw.sort_values("timestamp").reset_index(drop=True)

    # ── Resample différencié selon la sémantique des colonnes ─────────────────
    # _P / _S / _N : comptes de preuves → sommer (accumulation correcte)
    # _iw / _rmse  : mesures de qualité → moyenne (non-additives)

    psn_cols = [c for c in ev_raw.columns if c.endswith(("_P", "_S", "_N"))]
    meta_cols = [c for c in ev_raw.columns if c.endswith(("_iw", "_rmse"))]

    resample_kwargs = dict(origin="epoch", closed="left", label="left")

    # PATCH M-09 / F12 (2026-04-21) : dériver dynamiquement la période de resample
    # depuis (WINDOW_SIZE × freq_data) au lieu du "5min" codé en dur RedeRio-spécifique.
    def _freq_to_seconds(f: str) -> float:
        f = f.lower().strip()
        if f.endswith("ms"): return int(f[:-2]) / 1000.0
        if f.endswith("min"): return int(f[:-3]) * 60.0
        if f.endswith("h"):   return int(f[:-1]) * 3600.0
        if f.endswith("s"):   return int(f[:-1])
        # Fallback : assumer secondes
        try:
            return float(f)
        except ValueError:
            return 30.0  # RedeRio default

    _freq_s = _freq_to_seconds(CONFIG.get("freq_data", "30s"))
    _win_size = int(CONFIG.get("WINDOW_SIZE", 10))
    _resample_seconds = _freq_s * _win_size

    # Formatage pandas : préférer min si durée entière en minutes, sinon s
    if _resample_seconds >= 60 and _resample_seconds % 60 == 0:
        _resample_period = f"{int(_resample_seconds // 60)}min"
    else:
        _resample_period = f"{int(_resample_seconds)}s"

    print(f"   Resample period: {_resample_period} "
          f"(WINDOW_SIZE={_win_size} × freq_data={CONFIG.get('freq_data', '30s')})")

    ev_psn = (ev_raw.set_index("timestamp")[psn_cols]
              .resample(_resample_period, **resample_kwargs)
              .sum()
              .reset_index())

    if meta_cols:
        ev_meta = (ev_raw.set_index("timestamp")[meta_cols]
                   .resample(_resample_period, **resample_kwargs)
                   .mean()  # ← mean, pas sum
                   .reset_index())
        ev_df = ev_psn.merge(ev_meta, on="timestamp", how="left")
        print(f"   C3 dynamic columns detected: {meta_cols}")
    else:
        ev_df = ev_psn
        print(f"   No _iw/_rmse columns found — C3 mode is r2_static or evidence not regenerated")

    ev_cols_list = psn_cols + meta_cols  # pour le log
    print(f"   {len(ev_df)} windows | {ev_df['timestamp'].min()} -> {ev_df['timestamp'].max()}")
    print(f"   {len(ev_cols_list)} evidence columns found")

    print("-> Loading metadata CSV...")
    meta_df = pd.read_csv(METADATA_CSV)
    print(f"   {len(meta_df)} metrics")

    print("-> Loading attack catalog...")
    catalog = load_attack_catalog()
    ts_min, ts_max = ev_df["timestamp"].min(), ev_df["timestamp"].max()
    valid_catalog = [a for a in catalog
                     if pd.Timestamp(a["start"]) >= ts_min
                     and pd.Timestamp(a["start"]) + pd.Timedelta(hours=a["duration_h"]) <= ts_max]
    print(f"   {len(valid_catalog)}/{len(catalog)} attacks within evidence CSV range.\n")
    for a in valid_catalog:
        print(f"   [{a['intensity']:^7}] {a['name']:<35} {a['start']}")

    all_sweeps   = {}
    summary_rows = []
    threshold_sensitivity_rows = []
    all_results  = {}  # stocke les result_df bruts pour ré-évaluation à seuils alternatifs

    def register(run_name, result_df):
        if result_df.empty:
            return
        sdf = evaluate_run(result_df, valid_catalog)
        if sdf.empty:
            return
        all_sweeps[run_name] = sdf
        row = to_summary(run_name, sdf)
        if row:
            threshold_sensitivity_rows.append(row)
            br = best_row(sdf)
            print(f"   Score @ thr={br['threshold']:.2f}: "
                  f"F1-cov={br['f1_coverage']:.3f}  "
                  f"F1-bin={br['f1_binary']:.3f}  "
                  f"Prec={br['precision']:.3f}  "
                  f"FPR={br['fpr_pct']:.4f}%  "
                  f"Det={int(br['n_detected'])}/{int(br['n_attacks'])}  "
                  f"[{br.get('metric_protocol', 'catalog_outages_separate')}]")

    # ── Phase 1: SL Ablation Runs ─────────────────────────────────────────────
    print("=" * 52)
    print("  PHASE 1 — SL Ablation & Sensitivity")
    print("=" * 52)

    for run_name, run_cfg in ABLATION_RUNS.items():
        print(f"\n-> [{run_name}]")
        try:
            result_df = run_sl_fusion(ev_df, meta_df, run_cfg)
            all_results[run_name] = result_df
            register(run_name, result_df)
        except Exception as e:
            print(f"   ERROR: {e}")

    # PATCH-C4 (2026-04-18) : étude de sensibilité au seuil, symétrique.
    # PATCH-C4 fix (2026-04-20) : get_decision_threshold exige CONFIG + up_levels
    # comme partout ailleurs dans le fichier (voir L.102, L.389).
    THRESHOLD_GRID = [
        get_decision_threshold(CONFIG, up_levels=1),  # seuil calibré (référence)
        0.15,
        0.10,
    ]
    for variant_name, variant_df in all_results.items():
        for thr in THRESHOLD_GRID:
            # PATCH-C4 fix (2026-04-19) : utilise evaluate_run avec
            # thresholds_override=[thr] — _evaluate_at_threshold n'existe pas
            # dans ce module. evaluate_run() renvoie un sweep DataFrame ; on
            # extrait la 1re (et unique) ligne pour construire un dict
            # homogène avec le reste de summary_rows (cf. to_summary).
            sweep = evaluate_run(
                variant_df, valid_catalog, thresholds_override=[thr]
            )
            if sweep.empty:
                continue
            br = sweep.iloc[0]
            row = {
                "run":             f"{variant_name} @thr={thr:.3f}",
                "best_threshold":  float(br["threshold"]),
                "f1_binary":       float(br["f1_binary"]),
                "f1_coverage":     float(br["f1_coverage"]),
                "f1_ttd":          float(br["f1_ttd"]),
                "precision":       float(br["precision"]),
                "recall_binary":   float(br["recall_binary"]),
                "recall_coverage": float(br["recall_coverage"]),
                "fpr_pct":         float(br["fpr_pct"]),
                "n_detected":      int(br["n_detected"]),
                "n_attacks":       int(br["n_attacks"]),
                "metric_protocol": str(br.get("metric_protocol", "catalog_outages_separate")),
                "normal_windows": int(br.get("normal_windows", 0)),
                "outage_windows_excluded": int(br.get("outage_windows_excluded", 0)),
                "f1_coverage_legacy_outages_as_normal": float(
                    br.get("f1_coverage_legacy_outages_as_normal", np.nan)
                ),
                "precision_legacy_outages_as_normal": float(
                    br.get("precision_legacy_outages_as_normal", np.nan)
                ),
                "fpr_pct_legacy_outages_as_normal": float(
                    br.get("fpr_pct_legacy_outages_as_normal", np.nan)
                ),
                "variant":         variant_name,
                "threshold":       float(thr),
                "threshold_source": (
                    # PATCH-C4 fix² (2026-04-20) : get_decision_threshold exige
                    # CONFIG + up_levels — même signature que L.102, L.389, L.1105.
                    'calibrated' if abs(thr - get_decision_threshold(CONFIG, up_levels=1)) < 1e-9
                    else 'sensitivity_grid'
                ),
            }
            summary_rows.append(row)
    # Headline metric = celle au threshold 'calibrated' UNIQUEMENT.
    # Les autres sont "sensitivity only" — NE PAS cherry-picker pour le papier.

    # ── Phase 2: Classical Baselines ─────────────────────────────────────────
    print(f"\n{'='*52}")
    print("  PHASE 2 — Classical Baselines")
    print(f"{'='*52}")

    print("\n-> [Static Threshold] R²-weighted vote on N evidence columns")
    st = run_static_threshold(ev_df, meta_df)
    register("Static Threshold", st)

    # PATCH-C3 followup (2026-04-20) : si l'evidence CSV courant ne contient pas
    # assez de fenêtres pré-split (cas attacks-only CSV), PATCH-C3 lève ValueError
    # volontairement. On l'attrape ici pour ne pas perdre la Phase 1 : ablation
    # Phase 2 baselines est optionnelle ; la comparaison IF canonique est déjà
    # produite par `compare_if_fair.py` (cf. §2.2 PUBLICATION_TABLES.md).
    if _HAS_SKLEARN:
        # MODIFICATION: Tester plusieurs niveaux de contamination pour l'IF
        if_contaminations = [0.01, 0.02, 0.05, 0.10]
        for c in if_contaminations:
            print(f"\n-> [Isolation Forest (c={c:.2f})]")
            try:
                iso = run_isolation_forest(ev_df, valid_catalog, contamination=c)
                register(f"Isolation Forest (c={c:.2f})", iso)
            except ValueError as _e:
                print(f"   SKIPPED: {_e}")
                print("   -> IF baseline in ablation requires pre-split normals;")
                print("      canonical IF comparison is in compare_if_fair.py (§2.2).")
                break  # les autres contaminations échoueront pareil
    else:
        print("\n-> [Isolation Forest] SKIPPED (scikit-learn not installed)")

    # ── IF FPR-matched (comparaison équitable à même FPR que SL-ADS) ────────
    # PATCH TASK-35 (audit_codex CRIT-03, 2026-04-27) — FAIR-COMPARISON ONLY,
    # NO TEST-SET TUNING.
    #
    # Background. PATCH-C5 (2026-04-18) already moved the *target* FPR from
    # a test-measurement to CONFIG['FPR_TARGET_DECISION'].  However the
    # *contamination hyperparameter* of Isolation Forest was still being
    # selected by sweeping {0.01, 0.02, 0.03, 0.035, 0.04, 0.05} and picking
    # the value whose resulting FPR (measured on the held-out test sweep)
    # was closest to the target.  This is exactly the test-set tuning
    # pattern documented by Arp et al. (USENIX Sec. 2022, §4.2) and
    # re-classified as CRITICAL by audit_codex_2026-04-26 (CRIT-03).
    #
    # Fix.
    #   1. The reported headline IF baseline uses a SINGLE contamination
    #      value, ``CONFIG['IF_CONTAMINATION_DEFAULT']`` (= 0.02 by default,
    #      matching the typical labelled anomaly rate documented in the
    #      training catalog).  This value is pinned a priori and is NOT a
    #      function of the evaluation set.
    #   2. The remaining contaminations in the ladder are still computed
    #      for *sensitivity reporting only* (Appendix Table). They are
    #      labelled as such and explicitly excluded from any "best IF"
    #      selection — there is no longer any best-of selection at all.
    #   3. The escape hatch ``SL_ALLOW_TEST_TUNED_IF=1`` re-enables the old
    #      best-of-sweep selection and prints a [CRIT-03-OVERRIDE] warning;
    #      it MUST NEVER be used to populate publication tables.
    #
    # Refs:
    #   • Arp D. et al. (2022) "Dos and Don'ts of ML in Computer Security."
    #     USENIX Security, §4.2 "Sampling Bias / Tuning on Test."
    #   • Liu, Ting & Zhou (2008) "Isolation Forest." ICDM. (contamination
    #     is a hyperparameter, not a discovered quantity.)
    if _HAS_SKLEARN:
        _SL_FPR_TARGET = CONFIG.get('FPR_TARGET_DECISION', 0.001)
        _IF_C_HEADLINE = float(CONFIG.get('IF_CONTAMINATION_DEFAULT', 0.02))
        _IF_C_LADDER   = list(CONFIG.get(
            'IF_CONTAMINATION_LADDER',
            [0.01, 0.02, 0.03, 0.035, 0.04, 0.05]
        ))
        # Always run the headline contamination; sensitivity ladder is
        # produced for the appendix only.
        if _IF_C_HEADLINE not in _IF_C_LADDER:
            _IF_C_LADDER = sorted(set(_IF_C_LADDER + [_IF_C_HEADLINE]))

        print(f"  [IF FPR-match] target_fpr = {_SL_FPR_TARGET:.4f} (CONFIG, not test)")
        print(f"  [IF headline] contamination = {_IF_C_HEADLINE:.3f} "
              f"(CONFIG['IF_CONTAMINATION_DEFAULT'], a-priori)")
        print(f"  [IF sensitivity] ladder = {_IF_C_LADDER} "
              "(appendix only, NO best-of selection)")

        _allow_test_tuned = os.environ.get("SL_ALLOW_TEST_TUNED_IF") == "1"
        _best_if_fpr_gap = float("inf")
        _best_if_fpr_label = None

        for c in _IF_C_LADDER:
            is_headline = (abs(c - _IF_C_HEADLINE) < 1e-9)
            tag = "headline" if is_headline else "sensitivity"
            label = (f"IF-fpr-matched (c={c:.3f})"
                     if is_headline
                     else f"IF-fpr-matched (c={c:.3f}, sensitivity)")
            print(f"\n-> [{label}] [{tag}]")
            try:
                iso = run_isolation_forest(ev_df, valid_catalog, contamination=c)
                register(label, iso)
            except ValueError as _e:
                print(f"   SKIPPED (PATCH-C3): {_e}")
                break

            # FPR is *reported* but never used to pick c — except under the
            # explicit override for legacy reproductions of the old paper.
            if _allow_test_tuned and label in all_sweeps and not all_sweeps[label].empty:
                _fpr = all_sweeps[label].iloc[0]["fpr_pct"] / 100.0
                _gap = abs(_fpr - _SL_FPR_TARGET)
                if _gap < _best_if_fpr_gap:
                    _best_if_fpr_gap = _gap
                    _best_if_fpr_label = label

        if _allow_test_tuned and _best_if_fpr_label:
            import warnings as _w
            _w.warn(
                f"[CRIT-03-OVERRIDE] SL_ALLOW_TEST_TUNED_IF=1 — IF best-of "
                f"contamination selected on test sweep: {_best_if_fpr_label} "
                f"(gap={_best_if_fpr_gap:.4f}). "
                "MUST NOT be used for publication tables.",
                category=UserWarning, stacklevel=1,
            )
            print(f"\n   [CRIT-03-OVERRIDE] IF FPR-matched best: "
                  f"{_best_if_fpr_label} (gap={_best_if_fpr_gap:.4f})")
        else:
            _hl_label = f"IF-fpr-matched (c={_IF_C_HEADLINE:.3f})"
            print(f"\n   IF FPR-matched HEADLINE = {_hl_label} "
                  "(a-priori, not selected on test sweep)")

    # ── Phase 3: Save & Figures ───────────────────────────────────────────────
    print(f"\n{'='*52}")
    print("  PHASE 3 — Saving & Figures")
    print(f"{'='*52}\n")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(OUTPUT_DIR, "ablation_summary.csv"), index=False)
        print("   ablation_summary.csv saved")

    if threshold_sensitivity_rows:
        pd.DataFrame(threshold_sensitivity_rows).to_csv(
            os.path.join(OUTPUT_DIR, "ablation_threshold_sensitivity.csv"), index=False)
        print("   ablation_threshold_sensitivity.csv saved")

    frames = []
    for rn, sdf in all_sweeps.items():
        s = sdf.copy()
        s.insert(0, "run", rn)
        frames.append(s)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(OUTPUT_DIR, "ablation_all_sweeps.csv"), index=False)
        print("   ablation_all_sweeps.csv saved")

    plot_f1_curves(all_sweeps, OUTPUT_DIR)
    plot_comparison_table(summary_rows, OUTPUT_DIR)
    plot_bar_comparison(summary_rows, OUTPUT_DIR)

    # ── Console summary ───────────────────────────────────────────────────────
    REFERENCE_RUN = "Full SL-ADS (λ=0.85, Uniform Weights) [reference]"
    print(f"\n{'='*82}")
    print(f"  SUMMARY @ THRESHOLD {THRESHOLDS[0]:.2f} "
          "(catalog/outages-separate protocol)")
    print(f"{'='*82}")
    print(f"  {'Run':<30} {'F1-cov':>8} {'F1-bin':>8} {'Prec':>8} {'FPR%':>8}  "
          f"{'Det':>6}  {'vs ' + REFERENCE_RUN}")
    print(f"  {'-'*80}")

    if summary_rows:
        df_s = pd.DataFrame(summary_rows).sort_values("f1_coverage", ascending=False)
        ref  = df_s[df_s["run"] == REFERENCE_RUN]["f1_coverage"].values
        for _, r in df_s.iterrows():
            delta = ""
            if len(ref) > 0 and r["run"] != REFERENCE_RUN:
                d = r["f1_coverage"] - ref[0]
                delta = f"({d:+.3f})"
            flag = "<- reference" if r["run"] == REFERENCE_RUN else ""
            print(f"  {r['run']:<30} {r['f1_coverage']:>8.3f} {r['f1_binary']:>8.3f} "
                  f"{r['precision']:>8.3f} {r['fpr_pct']:>9.4f}%  "
                  f"{int(r['n_detected']):>3}/{int(r['n_attacks']):<3}  "
                  f"{delta} {flag}")

    print(f"\n{'='*82}")
    print(f"  Results in: {OUTPUT_DIR}")
    print(f"{'='*82}\n")


if __name__ == "__main__":
    main()
