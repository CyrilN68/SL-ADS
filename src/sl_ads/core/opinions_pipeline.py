"""
compute_opinions_v3.py — Pipeline de Fusion SL à 3 Niveaux
============================================================
Ref: Jøsang, A. (2016). Subjective Logic. Springer.

Niveau 1 : Fusion Temporelle Adaptative Conflict-Aware (Section 16.2.2, Eq. 16.5)
           λ_dyn = λ_base × max(0, 1 − α×K), K = degré de conflit prev/curr
Niveau 2 : Fusion Intra-Méthode par WBF N-ary (Section 12.5, Eq. 12.22)
Niveau 3 : Fusion Inter-Méthode par CBF ou WBF (Section 12.3, Eq. 12.14 / 12.22)

Entrée  : CSV de preuves (compute_evidence_v2.py)
Sortie  : CSV d'opinions (detection_results*.csv) + graphiques PNG par métrique
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os
import warnings
import joblib
import sl_ads.core.subjective_logic as sl  # Phase H
from sl_ads.core import fusion_policy
from sl_ads.paths import get_version_names, get_model_path, get_decision_threshold  # Phase H
# Note (PATCH M-03, 2026-04-21) :
#   Le module historique `adaptive_base_rate.py` a été archivé dans `un peu old/`.
#   Le pipeline actif utilise un **EDP statique** (Empirical Dirichlet Prior) :
#   les priors par métrique sont calculés par `train_v10.py` sur la fenêtre de
#   calibration (post-split) et figés dans `models_pkg["empirical_priors"]`.
#   Ici on les charge depuis le sidecar — aucune mise à jour en ligne.
#   Le flag CONFIG "adaptive_base_rate": True/False contrôle l'activation de
#   l'EDP statique (True) ou le fallback sur prior uniforme SL_PRIOR_A (False).

# ==============================================================================
# UTILITAIRES
# ==============================================================================

def _freq_to_seconds(f: str) -> int:
    """Convertit une chaîne de fréquence Pandas ('30s', '10min', '1h') en secondes."""
    f = f.lower().strip()
    if f.endswith('min'):  return int(f[:-3]) * 60
    elif f.endswith('h'):  return int(f[:-1]) * 3600
    elif f.endswith('s'):  return int(f[:-1])
    return 300  # défaut : 5 min


# ==============================================================================
# CONFIGURATION
# ==============================================================================
try:
    from sl_ads.config import CONFIG  # Phase H
except ImportError:
    print("❌ CRITIQUE : Impossible de charger sl_ads.config.")
    sys.exit(1)

# --- PARAMÈTRES D'ENTRÉE/SORTIE ---
VERSION_NAME, VERSION_NAME_MODIF = get_version_names(CONFIG)
# Sélection de l'evidence : version injectée (_attacks) si le dataset utilise l'injection,
# version originale sinon.
# RedeRio : ATTACK_CATALOG=None (catalogue intégré) → injection active → use _attacks.
# CESNET / METR-LA / GECCO : ATTACK_CATALOG=[] (vide) → pas d'injection → use original.
_has_injection  = CONFIG.get("ATTACK_CATALOG") is None or bool(CONFIG.get("ATTACK_CATALOG"))
# None  → catalogue intégré (RedeRio)          → injection active
# [{...}] → catalogue custom (METR-LA, etc.)   → injection active
# []    → injection désactivée (CESNET, GECCO)  → evidence originale
# Note : l'injection utilise toujours VERSION_NAME (pas VERSION_NAME_MODIF)
# pour nommer le fichier _attacks. VERSION_NAME_MODIF peut différer selon paths.py.
_ev_attacks_name = f"evidence_{VERSION_NAME}_attacks.csv"
_ev_orig_name    = f"evidence_{VERSION_NAME}.csv"
_ev_attacks_path = os.path.join("../results", f"resultats_{VERSION_NAME}", _ev_attacks_name)
_attacks_exists  = os.path.exists(_ev_attacks_path)
_force_noninjected = os.environ.get("SL_FORCE_NONINJECTED_OPINIONS", "").strip().lower() in ("1", "true", "yes")

if _force_noninjected:
    EVIDENCE_CSV = _ev_orig_name
    _OPINION_OUTPUT_SUFFIX = "_RAW"
    _OPINION_OUTPUT_SUBDIR = "opinions_non_injected"
    print(f"-> Evidence sélectionnée : {_ev_orig_name}  [non-injected forced by SL_FORCE_NONINJECTED_OPINIONS=1]")
elif _has_injection and _attacks_exists:
    EVIDENCE_CSV = _ev_attacks_name
    _OPINION_OUTPUT_SUFFIX = "_INJECTED"
    _OPINION_OUTPUT_SUBDIR = ""
    print(f"-> Evidence sélectionnée : {_ev_attacks_name}  [injection active ✅]")
elif _has_injection and not _attacks_exists:
    # PATCH TASK-20 (audit_tmp CRIT-02, 2026-04-26)
    # ──────────────────────────────────────────────────────────────────────────
    # Si l'injection est demandée par la config (`ATTACK_CATALOG=None` ou non-vide)
    # mais que le fichier `evidence_*_attacks.csv` n'existe pas, on lève une
    # FileNotFoundError dure. Le silent-fallback historique vers le CSV
    # non-injecté (avec `print` warning seulement) pouvait laisser un opérateur
    # croire qu'il évalue un run injecté alors qu'il évalue le baseline ; le
    # CSV de sortie perdait alors le suffixe `_INJECTED` et contaminait les
    # tableaux de manière indétectable.
    # Échappatoire pour les power-users : exporter SL_ALLOW_NONINJECTED_FALLBACK=1
    # avant d'exécuter le script. Le run continuera avec un avertissement bruyant.
    if os.environ.get("SL_ALLOW_NONINJECTED_FALLBACK", "0").lower() in ("1", "true", "yes"):
        EVIDENCE_CSV = _ev_orig_name
        _OPINION_OUTPUT_SUFFIX = ""
        _OPINION_OUTPUT_SUBDIR = ""
        print(f"⚠️  [SL_ALLOW_NONINJECTED_FALLBACK=1] injection active mais "
              f"_attacks introuvable → FALLBACK explicite sur {_ev_orig_name}.")
        print(f"   ATTENTION : la sortie ne portera PAS le suffixe `_INJECTED`.")
    else:
        raise FileNotFoundError(
            f"[CRIT-02] Injection demandée par la config "
            f"(ATTACK_CATALOG={CONFIG.get('ATTACK_CATALOG')!r}) mais "
            f"{_ev_attacks_path} introuvable.\n"
            f"  → Lancer `python inject_at_evidence_level.py` au préalable.\n"
            f"  → Pour autoriser explicitement le fallback non-injecté, "
            f"exporter SL_ALLOW_NONINJECTED_FALLBACK=1."
        )
else:
    EVIDENCE_CSV = _ev_orig_name
    _OPINION_OUTPUT_SUFFIX = ""
    _OPINION_OUTPUT_SUBDIR = ""
    print(f"-> Evidence sélectionnée : {_ev_orig_name}  [pas d'injection]")
RAW_DATA_CSV = f"raw_data_{VERSION_NAME}.csv"
METADATA_CSV = f"metadata_{VERSION_NAME}.csv"
INPUT_DIR = os.path.join("../results", f"resultats_{VERSION_NAME}")
OUTPUT_DIR = os.path.join("../results", f"resultats_{VERSION_NAME}", _OPINION_OUTPUT_SUBDIR)

# --- CHARGEMENT DE L'EDP DEPUIS L'ARTEFACT ENTRAÎNÉ ---
# L'Empirical Dirichlet Prior est calculé dans train_v10.py et stocké dans le PKL.
# compute_opinions le charge ici pour l'injecter comme base rate à chaque métrique feuille.
MODEL_PATH = get_model_path(CONFIG, up_levels=1)
if os.path.exists(MODEL_PATH):
    _models_pkg = joblib.load(MODEL_PATH)
else:
    _models_pkg = {}
    print(f"⚠️  AVERTISSEMENT : artefact PKL introuvable ({MODEL_PATH}).")
    print(f"    Tous les paramètres calibrés (EDP, seuil de décision, trust scores)")
    print(f"    basculent sur les valeurs par défaut de config.py. Lancer train_v10.py.")
# edp_dict = {metric_key: {'a_safe': float, 'a_susp': float, 'a_atk': float}} ou None
if CONFIG.get("USE_EMPIRICAL_PRIOR", True) and _models_pkg.get('empirical_priors'):
    _edp_dict = _models_pkg['empirical_priors']
    print(f"-> EDP chargé pour {len(_edp_dict)} métriques depuis {MODEL_PATH}")
else:
    _edp_dict = None
    print(f"-> EDP désactivé — prior uniforme SL_PRIOR_A utilisé")


# --- PARAMÈTRE AGEING (Section 16.2.2) ---
LAMBDA_DECAY = CONFIG.get('LAMBDA_DECAY')
UNCERTAINTY_MAXIMIZATION = CONFIG.get('UNCERTAINTY_MAXIMIZATION', False)
WBF_WEIGHT_MODE = CONFIG.get("WBF_WEIGHT_MODE", "uniform") #r2_static
TRUST_SCORES = _models_pkg.get("trust_scores", {})
# PATCH D5 (MASE trust mode, Hyndman-Koehler 2006).  Loaded eagerly so a
# missing key surfaces immediately rather than at the first per-window
# trust lookup.  Empty dict means "no MASE persistence in the .pkl"; the
# downstream lookup falls back to ``TRUST_SCORE_FLOOR`` per metric.
MASE_TRUST_SCORES = _models_pkg.get("mase_scores", {})

# TASK-14 / Phase H+ — Disclosure runtime du mode trust_discount.
# Le mode "trust_discount" est conservé uniquement comme baseline d'ablation
# (run `trust_discount_legacy`, F1=0.566) afin d'exposer la pathologie R²
# négatif documentée dans `docs/audit/trust_discount_r2_analysis.md` et
# `docs/honest_limitations.md` §5.3.3. La référence de production est
# WBF_WEIGHT_MODE="uniform" (F1=0.811). Émettre un avertissement explicite
# évite qu'un futur opérateur active ce mode sans réaliser que les chiffres
# obtenus ne sont pas comparables aux tables headline du papier.
if WBF_WEIGHT_MODE == "trust_discount":
    warnings.warn(
        "WBF_WEIGHT_MODE='trust_discount' active la pathologie R² documentée "
        "(F1≈0.566 vs 0.811 en référence 'uniform'). Ce mode est réservé aux "
        "ablations 'trust_discount_legacy'. Voir "
        "docs/audit/trust_discount_r2_analysis.md et "
        "docs/honest_limitations.md §5.3.3.",
        category=RuntimeWarning,
        stacklevel=2,
    )

# PATCH D5 (MASE trust mode) — Disclosure runtime quand le mode 'mase' est
# actif.  Le but est d'éviter qu'un opérateur active ce mode sans réaliser
# (i) qu'il faut un PKL avec ``mase_scores`` populated (sinon trust=floor
# partout) et (ii) que le sidecar de calibration doit lui aussi être
# recalibré (A1.9 hard-raise prévient automatiquement ; cf. paths.py).
if WBF_WEIGHT_MODE == "mase":
    if not MASE_TRUST_SCORES:
        warnings.warn(
            "WBF_WEIGHT_MODE='mase' is active but the loaded models_pkg "
            "does not contain a 'mase_scores' dict — every source will "
            "fall back to TRUST_SCORE_FLOOR. Re-run training with the "
            "current train_models.py to persist MASE trust scores, or "
            "use the post-fit helper sl_ads.train.compute_mase_postfit "
            "to patch the existing PKL.",
            category=RuntimeWarning,
            stacklevel=2,
        )
    else:
        n_finite = sum(1 for v in MASE_TRUST_SCORES.values()
                       if isinstance(v, (int, float)) and v > CONFIG.get('TRUST_SCORE_FLOOR', 0.05))
        warnings.warn(
            f"WBF_WEIGHT_MODE='mase' active (Hyndman-Koehler 2006 "
            f"trust map). {n_finite}/{len(MASE_TRUST_SCORES)} sources "
            f"above TRUST_SCORE_FLOOR. Headline figures must be "
            f"re-calibrated under this mode (sidecar A1.9 hard-raise "
            f"will block evaluation if the calibration sidecar was "
            f"produced under a different WBF_WEIGHT_MODE).",
            category=UserWarning,
            stacklevel=2,
        )

# A4.7 / 2026-05-06 — RuntimeWarning when CBF is active. CBF is Joesang
# 2016 Theorem 12.2 only under source independence; the residual-correlation
# audit (`stats/residual_correlation`, run 2026-05-04) reports cross max
# |rho| = 0.915 between Prophet and Reconstruction on RedeRio. Using CBF
# as the inter-method operator therefore double-counts evidence and
# produces over-confident attack opinions. WBF (the default since
# PATCH M-11) is the conservative pooling under dependence and must be
# preferred unless the operator explicitly wants the legacy ablation.
_inter_method_fusion_runtime = CONFIG.get("INTER_METHOD_FUSION", "wbf")
if _inter_method_fusion_runtime == "cbf":
    warnings.warn(
        "INTER_METHOD_FUSION='cbf' fuses Prophet and Reconstruction under "
        "the independence assumption of Joesang 2016 Theorem 12.2. The "
        "RedeRio residual-correlation audit reports cross max |rho| = 0.915 "
        "between the two branches; CBF therefore double-counts evidence "
        "and over-states attack confidence. The published reference uses "
        "INTER_METHOD_FUSION='wbf' (PATCH M-11). Switch back to 'wbf' or "
        "'hierarchical' for headline numbers; keep 'cbf' for explicit "
        "ablation/sensitivity runs only. See "
        "docs/scientific_deconstruction/ASSUMPTIONS.md A4.7 and "
        "docs/AUDIT_CURRENT_STATUS.md for the current metric protocol and "
        "run-status caveats.",
        category=RuntimeWarning,
        stacklevel=2,
    )
elif _inter_method_fusion_runtime == "bcf":
    warnings.warn(
        "INTER_METHOD_FUSION='bcf' activates Dempster-style Belief "
        "Constraint Fusion. This operator is conflict-sensitive and can "
        "produce Zadeh-style counter-intuitive certainty under strong "
        "Prophet/Reconstruction disagreement. Keep it for explicit "
        "ablation only, not headline SL-ADS results.",
        category=RuntimeWarning,
        stacklevel=2,
    )

# --- DECISION_THRESHOLD auto-calibré (depuis sidecar mode-spécifique) ---
# Variable de décision : proj_atk = b_atk + a_atk × u  (Jøsang 2016, Eq. 3.23)
# Calibré dans train_v10.py sur quantile(proj_atk, 1 − FPR_target) sur trafic normal.
# Priorité : sidecar spécifique au mode INTER_METHOD_FUSION.
# Fallback  : sidecar legacy, puis CONFIG['EVAL']['DECISION_THRESHOLD'].
DECISION_THRESHOLD = get_decision_threshold(CONFIG, up_levels=1)
print(f"-> DECISION_THRESHOLD résolu = {DECISION_THRESHOLD:.4f} "
      f"(mode fusion={CONFIG.get('INTER_METHOD_FUSION', 'wbf')}, "
      f"FPR cible {CONFIG.get('FPR_TARGET_DECISION', 0.01)*100:.1f}%)")


# ==============================================================================
# VISUALISATION
# ==============================================================================
def plot_metric_complete(metric_name, dates_window, op_history,
                         dates_raw=None, raw_real=None, raw_pred=None,
                         base_rates_history=None):
    """
    Graphique : Opinion SL (Haut) + Données Brutes (Bas).

    Paramètres :
        base_rates_history : liste de np.ndarray [a_safe, a_susp, a_atk] par fenêtre.
                             Si fourni, les base rates adaptatifs sont tracés en courbes.
                             Si None (ex: métriques agrégées), pas de base rates tracés.
    """
    b_safe = [op.b[0] for op in op_history]
    b_susp = [op.b[1] for op in op_history]
    b_atk  = [op.b[2] for op in op_history]
    u_val  = [op.u for op in op_history]
    p_safe = [op.projected_prob()[0] for op in op_history]
    p_susp = [op.projected_prob()[1] for op in op_history]
    p_anom = [op.projected_prob()[2] for op in op_history]

    has_raw = (dates_raw is not None and len(dates_raw) > 0)

    if has_raw:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                       gridspec_kw={'height_ratios': [1, 1.5], 'hspace': 0.1})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
        ax2 = None

    # --- PLOT 1 : OPINIONS ---
    ax1.step(dates_window, b_safe, where='post', color='#2ca02c', label='Safe', alpha=0.9)
    ax1.step(dates_window, b_susp, where='post', color='#ff7f0e', label='Suspect', alpha=0.9)
    ax1.step(dates_window, b_atk,  where='post', color='#d62728', label='Attack', lw=1.5)
    ax1.step(dates_window, u_val,  where='post', color='blue', linestyle=':', label='Uncertainty', alpha=0.8, lw=2.0)

    ax1.step(dates_window, p_safe, where='post', color='#1a7a1a', linestyle='--',
             label='P(Safe)', alpha=0.7, lw=1.2)
    ax1.step(dates_window, p_susp, where='post', color='#cc5500', linestyle='--',
             label='P(Suspect)', alpha=0.7, lw=1.2)
    ax1.step(dates_window, p_anom, where='post', color='#8b0000', linestyle='--',
             label='P(Anomalie)', alpha=0.7, lw=1.2)

    # --- Base rates adaptatifs a(x) — COURBES si disponibles, sinon rien ---
    # [CORRECTION] : on ne trace plus de ligne horizontale statique (op_history[0].a)
    # car les base rates évoluent désormais à chaque fenêtre.
    if base_rates_history is not None and len(base_rates_history) == len(dates_window):
        a_safe_hist = [a[0] for a in base_rates_history]
        a_susp_hist = [a[1] for a in base_rates_history]
        a_atk_hist  = [a[2] for a in base_rates_history]
        ax1.step(dates_window, a_safe_hist, where='post', color='#2ca02c',
                 linestyle=':', lw=1.0, alpha=0.5, label='a(Safe) adapt.')
        ax1.step(dates_window, a_susp_hist, where='post', color='#ff7f0e',
                 linestyle=':', lw=1.0, alpha=0.5, label='a(Susp) adapt.')
        ax1.step(dates_window, a_atk_hist,  where='post', color='#d62728',
                 linestyle=':', lw=1.0, alpha=0.5, label='a(Atk) adapt.')

    ax1.set_title(f"Subjective Opinion: {metric_name}", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Belief Mass")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='upper left', ncol=5, fontsize='small')
    ax1.grid(True, alpha=0.3)

    # --- PLOT 2 : DONNÉES BRUTES ---
    if has_raw and ax2:
        ax2.plot(dates_raw, raw_real, color='dimgray', label='Real Value', lw=1, alpha=0.7)
        ax2.plot(dates_raw, raw_pred, color='#2ca02c', linestyle='--', label='Predicted (Model)', lw=1.5)
        ax2.fill_between(dates_raw, raw_real, raw_pred, color='red', alpha=0.1, label='Abs Error')
        ax2.set_title(f"Raw Data & Prediction : {metric_name}", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    clean = metric_name.replace("->", "_to_").replace(" ", "").replace("/", "_")
    plt.savefig(os.path.join(OUTPUT_DIR, f"graph_{clean}.png"), dpi=100)
    plt.close()


def save_results_to_csv(viz_db, output_dir):
    """Sauvegarde les opinions finales en CSV, incluant les base rates adaptatifs."""
    print("--- Saving Opinion CSV Results ---")
    ref_key = "FINAL_SYSTEM_CBF"
    if ref_key not in viz_db or not viz_db[ref_key]['dates_win']:
        return

    data_dict = {'timestamp': viz_db[ref_key]['dates_win']}

    for key, data in viz_db.items():
        ops = data['ops']
        if not ops:
            continue

        # Convention : même nommage que compute_evidence_v2 (noms complets, "→" → "_to_").
        clean_key = key.replace("->", "_to_")

        data_dict[f"{clean_key}_b_safe"] = [op.b[0] for op in ops]
        data_dict[f"{clean_key}_b_susp"] = [op.b[1] for op in ops]
        data_dict[f"{clean_key}_b_atk"]  = [op.b[2] for op in ops]
        data_dict[f"{clean_key}_u"]       = [op.u for op in ops]

        data_dict[f"{clean_key}_proj_safe"] = [op.projected_prob()[0] for op in ops]
        data_dict[f"{clean_key}_proj_susp"] = [op.projected_prob()[1] for op in ops]
        data_dict[f"{clean_key}_proj_atk"]  = [op.projected_prob()[2] for op in ops]

        if len(data['ev_P']) == len(ops):
            data_dict[f"{clean_key}_ev_P"] = data['ev_P']
            data_dict[f"{clean_key}_ev_S"] = data['ev_S']
            data_dict[f"{clean_key}_ev_N"] = data['ev_N']

        if len(data['conflict_K']) == len(ops):
            data_dict[f"{clean_key}_conflict_K"]  = data['conflict_K']
            data_dict[f"{clean_key}_lambda_dyn"]  = data['lambda_dyn']

        # [AJOUT] Base rates adaptatifs (métriques feuilles uniquement)
        if len(data['base_rate_hist']) == len(ops):
            data_dict[f"{clean_key}_a_safe"] = [a[0] for a in data['base_rate_hist']]
            data_dict[f"{clean_key}_a_susp"] = [a[1] for a in data['base_rate_hist']]
            data_dict[f"{clean_key}_a_atk"]  = [a[2] for a in data['base_rate_hist']]

        # ── Colonnes directionnelles 5-états pour qualify_anomaly_sbn ─────────
        # Convention : {clean_key}_dir_pos_proj_{safe/susp/atk} et _dir_neg_*
        # Utilisées par QUALIFY_GROUP_SOURCES : volume_surplus et volume_deficit.
        # Transparent pour _compute_group_projected (même convention {src}_{col_suffix}).
        if (len(data['dir_pos_atk']) == len(ops) and
                not all(np.isnan(x) for x in data['dir_pos_atk'])):
            _ck_pos = f"{clean_key}_dir_pos"
            _ck_neg = f"{clean_key}_dir_neg"
            data_dict[f"{_ck_pos}_proj_safe"] = data['dir_safe']
            data_dict[f"{_ck_pos}_proj_susp"] = data['dir_pos_susp']
            data_dict[f"{_ck_pos}_proj_atk"] = data['dir_pos_atk']
            data_dict[f"{_ck_neg}_proj_safe"] = data['dir_safe']
            data_dict[f"{_ck_neg}_proj_susp"] = data['dir_neg_susp']
            data_dict[f"{_ck_neg}_proj_atk"] = data['dir_neg_atk']

    df_out = pd.DataFrame(data_dict)
    # Nom dynamique selon la source d'evidence sélectionnée en tête de fichier.
    _suffix = _OPINION_OUTPUT_SUFFIX
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"detection_results{_suffix}.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"✅ CSV Saved: {csv_path}")


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================
def compute_opinions():
    print(f"--- COMPUTE OPINIONS V3 ({VERSION_NAME}) ---")
    _wbf_label = WBF_WEIGHT_MODE if WBF_WEIGHT_MODE != 'uniform' else 'WBF(uniform)'
    # PATCH M-11/CBF : défaut WBF (indépendance Prophet⊥Reconst non garantie).
    _inter_cfg = CONFIG.get('INTER_METHOD_FUSION', 'wbf')
    _inter_label = {
        'wbf': 'WBF(confidence)',
        'abf': 'ABF(dependent-source average)',
        'cbf': 'CBF',
        'bcf': 'BCF(Dempster)',
        'ccf': 'CCF(projected)',
        'minbf': 'MinBF',
        'maxbf': 'MaxBF',
        'hierarchical': 'EvidenceAvg(0.5/0.5 hierarchical)',
    }.get(_inter_cfg, _inter_cfg)
    print(f"    Pipeline: Adaptive Ageing(λ_base={LAMBDA_DECAY}) → {_wbf_label} → {_inter_label}")
    print(f"    Conflict-Aware: λ_dyn = λ_base × max(0, 1 − α×K), α={CONFIG.get('CONFLICT_ALPHA', 1.495)}, K=conflit prev/curr")
    print(f"    Base Rate: Empirical Dirichlet Prior (EDP, W={CONFIG.get('SL_PARAM_K', 3.0)})")

    # 1. PARAMÈTRES SL (depuis config)
    try:
        # W=K=3 : facteur de prior-weight canonique pour un domaine ternaire θ={Safe,Suspect,Attack}.
        # Jøsang (2016) §3.5.2 : W = |θ| = cardinal de la frame of discernment.
        W_BIJ = CONFIG['SL_PARAM_K']
        # CONFLICT_ALPHA — Facteur d'amplification du conflit garantissant le hard-reset.
        # Recalculé dynamiquement dans config.py (fin de fichier) selon :
        #   b_curr_max = WINDOW_SIZE / (WINDOW_SIZE + W)
        #   b_prev_max = (2×WINDOW_SIZE) / (2×WINDOW_SIZE + W)   [approx. 2 fenêtres cumulées]
        #   K_conflict_max = b_prev_max × b_curr_max
        #   CONFLICT_ALPHA = 1 / K_conflict_max
        # Propriété : λ_dyn = λ_base × max(0, 1 − α×K) → 0 exactement quand K = K_conflict_max
        # (contradiction maximale entre preuves accumulées et preuves courantes).
        # Valeur par défaut (1.495) : WINDOW_SIZE=10, W=3 (dataset brésilien).
        CONFLICT_ALPHA = CONFIG.get('CONFLICT_ALPHA', 1.495)

        # --- BALANCE RATIO — rééquilibrage inter-méthode legacy pour CBF ---
        # ═══════════════════════════════════════════════════════════════════════
        # PATCH-m4 (2026-04-18 / 2026-04-19) — DÉVIATION EXPLICITE DE JØSANG Th. 12.2
        # ═══════════════════════════════════════════════════════════════════════
        # Jøsang (2016) Subjective Logic, Theorem 12.2 (Eq. 12.17) définit CBF
        # comme addition exacte de preuves r(ω_A) + r(ω_B), sans pondération.
        #
        # Le mécanisme BALANCE_RATIO introduit ici réduit la source dominante
        # par un facteur 1/ratio avant fusion. C'est une EXTENSION HEURISTIQUE
        # du cadre CBF, pas une application littérale du Théorème 12.2.
        #
        # JUSTIFICATION OPÉRATIONNELLE :
        #   Si N_prophet >> N_reconst, la méthode Prophet accumule
        #   structurellement plus de preuves (simple fait quantitatif), ce qui
        #   ferait dominer Prophet indépendamment de la qualité discriminative
        #   relative des deux méthodes. Le rééquilibrage aligne les cardinalités
        #   effectives avant agrégation.
        #
        # STATUT ÉPISTÉMOLOGIQUE :
        #   - Heuristique motivée ✓ (pas de dérivation théorique formelle).
        #   - Applicable uniquement au mode "cbf" dans le code courant.
        #   - Les modes "abf" et "hierarchical" évitent plutôt l'accumulation
        #     inter-méthode : ABF pour sources dépendantes, hierarchical par
        #     moyenne d'évidence à poids fixes [0.5, 0.5].
        #
        # Pour le paper : déclarer explicitement BALANCE_RATIO comme choix de
        # design ; rapporter les résultats avec "hierarchical" en ablation.
        # ═══════════════════════════════════════════════════════════════════════
        _balance_cfg = CONFIG.get('BALANCE_RATIO', 1.0)
        # Le ratio effectif est calculé après chargement des métadonnées (voir ci-dessous).

        # INTER_METHOD_FUSION : choix de l'opérateur inter-méthode Prophet/Reconst
        #
        # PATCH M-11/CBF : le défaut est maintenant "wbf" car l'hypothèse
        # d'indépendance Prophet⊥Reconst nécessaire à la CBF (Jøsang Theorem 12.2)
        # n'est pas garantie par construction (mêmes fenêtres brutes en entrée).
        #
        # "wbf"          = Weighted Belief Fusion pondérée par confiance (1-u).
        # "abf"          = Averaging Belief Fusion, candidat principal quand les
        #                  deux branches sont dépendantes.
        # "hierarchical" = moyenne d'évidence à poids fixes [0.5, 0.5], sans
        #                  multiplication par la confiance inter-branche.
        # "cbf"          = Cumulative Belief Fusion, valide sous indépendance
        #                  stricte ; conservé pour reproductibilité/ablation.
        # "bcf"          = Dempster/Belief Constraint Fusion, sensible au conflit.
        # "ccf"          = projection singleton consensus/compromise expérimentale.
        # "minbf/maxbf"  = bornes heuristiques AND/OR pour tests de sensibilité.
        INTER_METHOD_FUSION = CONFIG.get('INTER_METHOD_FUSION', 'wbf')

        # CONTEXTUAL DISCOUNTING de la Reconst (Mercier & Denoeux 2006)
        # α_attack ∈ [0,1] : fiabilité de la Reconst pour l'hypothèse "attack"
        # "auto" → lire depuis models_pkg['reconst_attack_reliability'] (calibré par train_v10)
        # 1.0   → CBF standard (pas de discounting)
        # 0.1   → Reconst peu fiable pour "attack" (mode manuel)
        # Pour désactiver : RECONST_ATTACK_RELIABILITY = 1.0 dans config.py
        _raw_cd = CONFIG.get('RECONST_ATTACK_RELIABILITY', 1.0)
        if str(_raw_cd).strip().lower() == 'auto':
            CD_ALPHA_ATTACK = float(_models_pkg.get('reconst_attack_reliability', 1.0))
            print(f"-> CD_ALPHA_ATTACK = {CD_ALPHA_ATTACK:.3f} "
                  f"(mode 'auto', calibré par train_v10 depuis artefact)")
        else:
            CD_ALPHA_ATTACK = float(_raw_cd)
            print(f"-> CD_ALPHA_ATTACK = {CD_ALPHA_ATTACK:.3f} (mode manuel, config.py)")
    except KeyError as e:
        print(f"❌ Erreur Config: Clé manquante {e}")
        return

    # 2. CHARGEMENT DES CSV DE PREUVES & AGREGATION 5 MIN
    ev_path   = os.path.join(INPUT_DIR, EVIDENCE_CSV)
    meta_path = os.path.join(INPUT_DIR, METADATA_CSV)
    raw_path  = os.path.join(INPUT_DIR, RAW_DATA_CSV)

    for fpath in [ev_path, meta_path]:
        if not os.path.exists(fpath):
            print(f"❌ Fichier introuvable : {fpath}")
            return

    print("-> Loading evidence CSV...")
    df_ev_raw = pd.read_csv(ev_path, parse_dates=['timestamp'])

    # Extension aux composantes directionnelles du frame 5-états (direction='both').
    # Suffixes _S_pos/_N_pos/_S_neg/_N_neg : évidence directionnelle pour qualify_anomaly_sbn.
    # Invariant après resample sum() : _S = _S_pos + _S_neg, _N = _N_pos + _N_neg ✓
    evidence_cols = [c for c in df_ev_raw.columns
                     if c.endswith(('_P', '_S', '_N', '_S_pos', '_N_pos', '_S_neg', '_N_neg'))]
    if not evidence_cols:
        print("❌ CRITIQUE : Aucune colonne de preuve (_P, _S, _N) trouvée.")
        sys.exit(1)

    # Calcul de la fréquence cible = freq_data × WINDOW_SIZE (résolution native des fenêtres).
    # Évite la création de fenêtres fantômes sur les datasets à granularité > 30s.
    # Ex : CESNET (freq=10min, W=1) → '10min' ; RedeRio (freq=30s, W=10) → '5min'.
    _native_s    = _freq_to_seconds(CONFIG.get('freq_data', '30s')) * int(CONFIG.get('WINDOW_SIZE', 10))
    _target_freq = f"{_native_s // 60}min" if _native_s % 60 == 0 else f"{_native_s}s"
    _freq_label  = CONFIG.get('freq_data', '30s')

    print(f"-> 🔄 Processing: Resampling to {_target_freq} fixed windows (Summing Evidence)...")

    # origin='epoch' → bins ancrés sur l'epoch Unix (déterministe, timezone-agnostic).
    # 'start_day' est non-déterministe entre fuseaux horaires → résultats non-reproductibles
    # cross-dataset (RedeRio UTC-3 ≠ METR-LA UTC-8 ≠ CESNET UTC+1).
    df_ev = (df_ev_raw
             .set_index('timestamp')[evidence_cols]
             .resample(_target_freq, origin='epoch', closed='left', label='left')
             .sum()
             .reset_index())

    print(f"   Evidence CSV ({_freq_label}, agrégée par compute_evidence_v2) : {len(df_ev_raw)} fenêtres")
    if len(df_ev) == len(df_ev_raw):
        print(f"   Resample ({_target_freq}) : alignement index uniquement (pas de réduction — CSV déjà à la bonne résolution).")
    else:
        print(f"   Resample ({_target_freq}) : {len(df_ev_raw)} → {len(df_ev)} fenêtres (agrégation active).")

    print("-> Loading metadata CSV...")
    df_meta = pd.read_csv(meta_path)

    raw_data = {}
    if os.path.exists(raw_path):
        print("-> Loading raw data CSV...")
        df_raw = pd.read_csv(raw_path, parse_dates=['timestamp'])
        for key, grp in df_raw.groupby('metric_key'):
            raw_data[key] = grp.sort_values('timestamp')

    # 3. RECONSTRUCTION DES STRUCTURES MÉTADONNÉES
    meta_dict = {}
    for _, row in df_meta.iterrows():
        meta_dict[row['metric_key']] = {
            'clean_key':          row['clean_key'],
            'type':               row['type'],
            'r2_weight':          row['r2_weight'],
            'threshold_suspect':  row['threshold_suspect'],
            'threshold_attack':   row['threshold_attack']
        }

    method_groups = fusion_policy.get_method_groups(CONFIG)
    group_keys = fusion_policy.group_metric_keys(meta_dict, CONFIG)
    group_name_by_metric = {
        key: group["name"]
        for group in method_groups
        for key in group_keys.get(group["name"], [])
    }
    all_metric_keys = [
        key
        for group in method_groups
        for key in group_keys.get(group["name"], [])
    ]
    system_keys = [group["output_key"] for group in method_groups] + ["FINAL_SYSTEM_CBF"]

    # Legacy aliases retained for existing balance-ratio diagnostics.
    prophet_keys = group_keys.get("prophet", [])
    reconst_keys = group_keys.get("reconstruction", [])

    # METRIC_WEIGHTS supprimé (inutilisé) — r2_weight lu inline via meta_dict[key]['r2_weight'].

    viz_db = {}
    for k in all_metric_keys + system_keys:
        viz_db[k] = {
            'dates_win': [], 'ops': [],
            'dates_raw': [], 'real': [], 'pred': [],
            'ev_P': [], 'ev_S': [], 'ev_N': [],
            'conflict_K': [], 'lambda_dyn': [],
            'base_rate_hist': [],
            # Composantes directionnelles 5-états (direction='both' uniquement).
            # Stockées ici pour export CSV → qualify_anomaly_sbn via groupes
            # volume_surplus / volume_deficit (Jøsang §3.5.4 coarsening, W=W_BIJ).
            'dir_pos_susp': [], 'dir_pos_atk': [],
            'dir_neg_susp': [], 'dir_neg_atk': [],
            'dir_safe': [],
        }

    # =========================================================================
    # 4. STATE MEMORY — Vecteurs de preuves accumulées (Eq. 16.5)
    # =========================================================================
    # Initialisation avec l'EDP prior comme "évidence virtuelle" (R_init = a_edp × W).
    # Justification : l'EDP est la distribution empirique du trafic normal calculée à
    # l'entraînement (Ferguson 1973 ; Jaynes 1957). Partir de l'opinion vacuouse (U=1)
    # ignore ce prior et sur-exprime l'ignorance pendant la phase de warm-up.
    #
    # Propriété exacte (bijection Def. 3.9, Jøsang 2016) :
    #   sum(R_init) = W×sum(a_edp) = W×1 = W
    #   D = sum(R_init) + W = 2W
    #   u_init = W / 2W = 0.5   (exactement, pas approximativement)
    #   b_i    = (a_i × W) / 2W = a_i / 2
    #   proj_i = b_i + a_i × u_init = a_i/2 + a_i/2 = a_i
    # → La probabilité projetée initiale est exactement égale au prior EDP.
    # L'ageing λ converge vers le régime permanent dès les premières fenêtres.
    if _edp_dict is not None:
        state_memory = {}
        for key in all_metric_keys:
            if key in _edp_dict:
                edp_k = _edp_dict[key]
                a_init = np.array([edp_k['a_safe'], edp_k['a_susp'], edp_k['a_atk']])
                state_memory[key] = a_init * float(W_BIJ)
            else:
                state_memory[key] = np.zeros(3)
    else:
        state_memory = {key: np.zeros(3) for key in all_metric_keys}

    # =========================================================================
    # 5. BASE RATE — Empirical Dirichlet Prior (EDP, calculé dans train_v10.py)
    # =========================================================================
    # Prior fixe par métrique, chargé depuis l'artefact PKL.
    # Cold start : si EDP absent pour une métrique → fallback prior uniforme SL_PRIOR_A.

    # Calcul du ratio de balance effectif (constant sur toutes les fenêtres)
    _n_prophet = len(prophet_keys)
    _n_reconst = len(reconst_keys)

    if _balance_cfg == "auto":
        BALANCE_RATIO_EFF = float(_n_prophet) / float(_n_reconst) if _n_reconst > 0 else 1.0
    elif isinstance(_balance_cfg, (int, float)) and float(_balance_cfg) > 0:
        BALANCE_RATIO_EFF = float(_balance_cfg)
    else:
        BALANCE_RATIO_EFF = 1.0

    _balance_active = not np.isclose(BALANCE_RATIO_EFF, 1.0)
    print(f"    Balance Ratio : cfg='{_balance_cfg}' → effectif={BALANCE_RATIO_EFF:.3f} "
          f"(Prophet={_n_prophet}, Reconst={_n_reconst}, actif={_balance_active})")
    print("    Method groups : " + ", ".join(
        f"{group['name']}={len(group_keys.get(group['name'], []))}"
        for group in method_groups
    ))

    print(f"-> Computing opinions on {len(df_ev)} windows (λ={LAMBDA_DECAY})...")

    c3_mode = CONFIG.get("C3_WEIGHT_MODE", "uniform")
    c3_warmup = CONFIG.get("C3_ONLINE_RMSE_WARMUP", 10)
    c3_alpha = CONFIG.get("C3_ONLINE_RMSE_ALPHA", 0.05)

    # RMSE glissant par métrique — None = phase de warmup (avant c3_warmup fenêtres normales)
    safe_rmse_state = {key: None for key in all_metric_keys}
    # delta_half : demi-seuil de décision, constante sur tout le run
    delta_half = DECISION_THRESHOLD / 2.0
    # prev_proj_atk_metric : proj_atk[key] de la fenêtre précédente (gating per-metric).
    # Justification : le reset du RMSE de référence est déclenché par chaque métrique
    # indépendamment, évitant qu'un signal d'attaque sur une seule métrique efface les
    # états RMSE des métriques non-affectées (cf. Chandola et al. 2009, CSUR §3.1 :
    # indépendance des détecteurs dans un ensemble hybride).
    prev_proj_atk_metric = {key: 0.0 for key in all_metric_keys}
    window_count = 0

    def get_c3_weight(key, clean_key, row, c3_mode, safe_rmse_state, static_r2):
        """
        Retourne le poids C3 selon le mode configuré.

        key         : clé métrique brute ("prophet_bytes", etc.)
        clean_key   : clé nettoyée dans le CSV evidence (key.replace("->","_to_"))
                      Nécessaire pour lire les colonnes _iw/_rmse écrites par compute_evidence_v2.
        row         : ligne courante du DataFrame evidence
        c3_mode     : "uniform" | "r2_static" | "prophet_interval" | "online_rmse"
        safe_rmse_state : dict {key: float | None}
        static_r2   : R² statique stocké dans metadata (fallback)
        """
        if c3_mode == "uniform":
            return 1.0

        elif c3_mode == "r2_static":
            return static_r2

        elif c3_mode == "prophet_interval":
            if key.startswith("prophet_"):
                # Utilise clean_key (cohérent avec compute_evidence_v2 qui écrit f"{clean_key}_iw")
                iw = row.get(f"{clean_key}_iw", None)
                if iw is not None and iw > 1e-6:
                    return 1.0 / iw  # inversement proportionnel à l'incertitude de prédiction
            return static_r2  # fallback Reconst ou données manquantes

        elif c3_mode == "online_rmse":
            state = safe_rmse_state.get(key)
            if state is None:
                return static_r2  # phase de warmup : fallback sur R²
            return 1.0 / state  # inversement proportionnel au RMSE courant en régime normal

        return static_r2

    # =========================================================================
    # BOUCLE PAR FENÊTRE
    # =========================================================================
    for idx, row in df_ev.iterrows():
        current_date = row['timestamp']
        window_count += 1

        ops_by_group = {group["name"]: [] for group in method_groups}
        weights_by_group = {group["name"]: [] for group in method_groups}

        for key in all_metric_keys:
            clean_key = meta_dict[key]['clean_key']

            P = row.get(f"{clean_key}_P", 0.0)
            S = row.get(f"{clean_key}_S", 0.0)
            N = row.get(f"{clean_key}_N", 0.0)
            r_current = np.array([P, S, N])

            viz_db[key]['ev_P'].append(P)
            viz_db[key]['ev_S'].append(S)
            viz_db[key]['ev_N'].append(N)

            # =============================================================
            # NIVEAU 1 : Fusion Temporelle Adaptative (Conflict-Aware)
            # =============================================================
            R_new, K_conf, lam_dyn = sl.temporal_adaptive_ageing(
                r_accumulated=state_memory[key],
                r_current=r_current,
                lam_base=LAMBDA_DECAY,
                W=W_BIJ,
                alpha=CONFLICT_ALPHA,
            )
            state_memory[key] = R_new

            # Base rate (EDP) : prior empirique calculé à l'entraînement (train_v10).
            # Statique par métrique — Ferguson (1973), Efron & Morris (1973 JASA).
            # Fallback : prior uniforme SL_PRIOR_A si la métrique est absente de l'EDP.
            if _edp_dict is not None and key in _edp_dict:
                edp_k = _edp_dict[key]
                a_inj = np.array([edp_k['a_safe'], edp_k['a_susp'], edp_k['a_atk']])
            else:
                a_inj = np.array(CONFIG['SL_PRIOR_A'])   # fallback : prior uniforme
            op_temporal = sl.evidence_to_opinion(R_new, W=W_BIJ, a=a_inj)

            # Uncertainty-maximisation optionnelle (Def. 3.6, Eq. 3.27)
            if UNCERTAINTY_MAXIMIZATION:
                op_temporal = op_temporal.uncertainty_maximized()

            viz_db[key]['dates_win'].append(current_date)
            viz_db[key]['ops'].append(op_temporal)
            viz_db[key]['conflict_K'].append(K_conf)
            viz_db[key]['lambda_dyn'].append(lam_dyn)
            viz_db[key]['base_rate_hist'].append(a_inj.copy()) ##"empirical static prior" (EDP) ?

            # ── Probabilités projetées directionnelles (frame 5-états) ───────
            # Calculées uniquement pour direction='both'. Utilise W_BIJ (= W ternaire)
            # pour garantir l'invariant de coarsening (Jøsang §3.5.4) :
            #   proj_atk_ternaire = proj_atk_pos + proj_atk_neg  (exactement)
            # car D_5 = P + S_pos + S_neg + N_pos + N_neg + W_BIJ = D_ternaire.
            _S_pos = row.get(f"{clean_key}_S_pos", float('nan'))
            _N_pos = row.get(f"{clean_key}_N_pos", float('nan'))
            _S_neg = row.get(f"{clean_key}_S_neg", float('nan'))
            _N_neg = row.get(f"{clean_key}_N_neg", float('nan'))

            if not (np.isnan(_S_pos) or np.isnan(_N_pos)):
                # D_5 = même dénominateur que l'opinion ternaire (W_BIJ constant)
                _D5 = P + S + N + W_BIJ  # P+S+N = WINDOW_SIZE (somme des preuves)
                _u5 = W_BIJ / _D5 if _D5 > 1e-12 else 1.0
                # Base rates 5-états depuis EDP (avec fallback sur ternaire / 2)
                _edp5 = _edp_dict.get(key, {}) if _edp_dict else {}
                _a_sp = _edp5.get('a_susp_pos', a_inj[1] / 2.0)
                _a_np = _edp5.get('a_atk_pos', a_inj[2] / 2.0)
                _a_sn = _edp5.get('a_susp_neg', a_inj[1] / 2.0)
                _a_nn = _edp5.get('a_atk_neg', a_inj[2] / 2.0)
                # Probabilités projetées directionnelles (Jøsang 2016, Eq. 3.17)
                viz_db[key]['dir_pos_susp'].append(float(_S_pos / _D5 + _a_sp * _u5))
                viz_db[key]['dir_pos_atk'].append(float(_N_pos / _D5 + _a_np * _u5))
                viz_db[key]['dir_neg_susp'].append(float(_S_neg / _D5 + _a_sn * _u5))
                viz_db[key]['dir_neg_atk'].append(float(_N_neg / _D5 + _a_nn * _u5))
                viz_db[key]['dir_safe'].append(float(op_temporal.projected_prob()[0]))
            else:
                viz_db[key]['dir_pos_susp'].append(float('nan'))
                viz_db[key]['dir_pos_atk'].append(float('nan'))
                viz_db[key]['dir_neg_susp'].append(float('nan'))
                viz_db[key]['dir_neg_atk'].append(float('nan'))
                viz_db[key]['dir_safe'].append(float('nan'))

            # Calcul du poids de source (C3 ou trust-discount via a_safe)
            static_r2 = max(float(meta_dict[key]['r2_weight']), 0.01)
            w = get_c3_weight(key, clean_key, row, c3_mode, safe_rmse_state, static_r2)
            if WBF_WEIGHT_MODE == "trust_discount":
                t = float(TRUST_SCORES.get(key, a_inj[0]))  # fallback a_safe
                op_temporal = sl.apply_trust_discount(op_temporal, t)
                w = 1.0
            elif WBF_WEIGHT_MODE == "mase":
                # PATCH D5 (MASE trust mode, Hyndman-Koehler 2006).
                # Same Joesang Def. 14.6 application as ``trust_discount``
                # but the trust value is read from ``mase_scores`` instead
                # of ``trust_scores``.  Fallback for missing keys
                # (typically RANSAC reconstructions, where MASE is
                # undefined for non-temporal regressions) is the
                # numerical floor — not ``a_safe`` — so RANSAC sources
                # are silenced through the WBF weighting and remain
                # available through the contextual-discount path
                # ``RECONST_ATTACK_RELIABILITY``.
                _floor = float(CONFIG.get("TRUST_SCORE_FLOOR", 0.05))
                t = float(MASE_TRUST_SCORES.get(key, _floor))
                op_temporal = sl.apply_trust_discount(op_temporal, t)
                w = 1.0

            # =============================================================
            # MISE À JOUR DU RMSE DE RÉFÉRENCE — per-metric (C3 online_rmse)
            # =============================================================
            # Le poids w vient d'être calculé (causal : utilise l'état issu des
            # fenêtres 1..N-1). On met maintenant l'état à jour pour la fenêtre N+1.
            # Gating : on n'intègre ce RMSE dans la baseline que si la fenêtre
            # PRÉCÉDENTE de CETTE métrique était clairement normale (prev_proj_atk < δ/2)
            # OU si on est encore en warm-up.
            if c3_mode == "online_rmse":
                p_atk_leaf = float(op_temporal.projected_prob()[2])
                is_safe_metric = (prev_proj_atk_metric[key] < delta_half) or (window_count <= c3_warmup)
                rmse_win = row.get(f"{clean_key}_rmse", None)
                if rmse_win is not None and not np.isnan(float(rmse_win)):
                    if is_safe_metric:
                        if safe_rmse_state[key] is None:
                            safe_rmse_state[key] = float(rmse_win)  # init
                        else:
                            # Lissage exponentiel (EWMA) — α contrôle la réactivité
                            safe_rmse_state[key] = (
                                c3_alpha * float(rmse_win) +
                                (1.0 - c3_alpha) * safe_rmse_state[key]
                            )
                # Anti-contamination : reset si transition attaque→normal sur CETTE métrique.
                # Sans reset, un RMSE gonflé pendant une attaque contaminerait les poids C3
                # des fenêtres normales suivantes de cette métrique spécifiquement.
                if prev_proj_atk_metric[key] >= delta_half and p_atk_leaf < delta_half:
                    safe_rmse_state[key] = None
                prev_proj_atk_metric[key] = p_atk_leaf

            group_name = group_name_by_metric[key]
            ops_by_group[group_name].append(op_temporal)
            weights_by_group[group_name].append(w)

        # =================================================================
        # NIVEAU 2 : Fusion Intra-Méthode par WBF N-ary (Eq. 12.22)
        # a_fused = Σ w_i·a_i  — propagation automatique des base rates
        # =================================================================
        method_ops_for_fusion = []
        for group in method_groups:
            group_name = group["name"]
            group_ops = ops_by_group.get(group_name, [])
            if not group_ops:
                continue

            # PATCH D5: 'mase' uses the same fold-trust-into-the-opinion
            # path as 'trust_discount', so external WBF weights must be
            # disabled (uniform N-ary WBF) — the discount has already been
            # applied per-source via ``apply_trust_discount`` above.
            _trust_into_opinion = WBF_WEIGHT_MODE in ("trust_discount", "mase")
            ext_w = None if _trust_into_opinion else weights_by_group[group_name]
            # W_BIJ passed explicitly to avoid frozen module-level defaults.
            op_agg = sl.fusion_wbf_n_sources(group_ops, external_weights=ext_w, W=W_BIJ)

            viz_key = group["output_key"]
            viz_db[viz_key]['dates_win'].append(current_date)
            viz_db[viz_key]['ops'].append(op_agg)

            explicit_alpha = CD_ALPHA_ATTACK if group.get("attack_discount_config_key") else None
            method_ops_for_fusion.append(
                fusion_policy.apply_method_discount(
                    op_agg,
                    group,
                    CONFIG,
                    explicit_alpha=explicit_alpha,
                )
            )

        # =================================================================
        # NIVEAU 3 : Fusion Inter-Methode selon INTER_METHOD_FUSION.
        #
        # Les opinions deja agregees par methode sont fusionnees via la
        # meme politique que la calibration du seuil.  La liste peut contenir
        # 2 methodes (Prophet/Reconstruction aujourd'hui) ou davantage si
        # CONFIG['FUSION_METHOD_GROUPS'] est etendue.
        # =================================================================
        op_final = fusion_policy.fuse_method_opinions(
            method_ops_for_fusion,
            mode=INTER_METHOD_FUSION,
            W=W_BIJ,
            balance_ratio_eff=BALANCE_RATIO_EFF,
        )
        # Variable de décision : proj_atk = b_atk + a_atk × u (Jøsang 2016, Eq. 3.23)
        # Cohérent avec la calibration du seuil dans train_v10 (cf. _compute_training_proj_atk).
        p_atk = float(op_final.projected_prob()[2])

        viz_db["FINAL_SYSTEM_CBF"]['dates_win'].append(current_date)
        viz_db["FINAL_SYSTEM_CBF"]['ops'].append(op_final)


    # =========================================================================
    # INJECTION DES DONNÉES BRUTES POUR GRAPHIQUES
    # =========================================================================
    for key in all_metric_keys:
        if key in raw_data:
            grp = raw_data[key]
            viz_db[key]['dates_raw'] = grp['timestamp'].tolist()
            viz_db[key]['real']      = grp['real'].tolist()
            viz_db[key]['pred']      = grp['pred'].tolist()

    # =========================================================================
    # SAUVEGARDE & GRAPHIQUES
    # =========================================================================
    save_results_to_csv(viz_db, OUTPUT_DIR)

    # PATCH TASK-44 (audit_codex MAJ-09, 2026-04-27).  Persist the actual
    # inter-method fusion mode that was used to produce FINAL_SYSTEM_CBF*
    # columns.  Despite the column-name prefix ``FINAL_SYSTEM_CBF`` (kept
    # for back-compat with 31 downstream consumers), the underlying
    # operator depends on ``CONFIG['INTER_METHOD_FUSION']`` and may be
    # WBF (Eq. 12.22-24) or hierarchical equal-weight WBF instead of CBF
    # (Eq. 12.14).  The sidecar JSON below makes the mode transparent so
    # reviewers and the publication tables can cite it without reading
    # config.py.  Cf. ``paths.get_fusion_mode_for_run()``.
    try:
        import json as _json_fmd
        _fusion_meta = {
            "column_prefix": "FINAL_SYSTEM_CBF",
            "actual_fusion_mode": str(INTER_METHOD_FUSION),
            "wbf_weight_mode": CONFIG.get("WBF_WEIGHT_MODE", "uniform"),
            "balance_ratio": float(CONFIG.get("BALANCE_RATIO", 1.0)),
            "cd_alpha_attack": float(CD_ALPHA_ATTACK),
            "lambda_decay": float(CONFIG.get("LAMBDA_DECAY", 0.85)),
            "method_groups": CONFIG.get("FUSION_METHOD_GROUPS"),
            "comment": (
                "audit_codex MAJ-09: column prefix FINAL_SYSTEM_CBF is "
                "historical; actual_fusion_mode field carries the truth "
                "(wbf | abf | cbf | bcf | ccf | minbf | maxbf | hierarchical). "
                "Use paths.get_fusion_mode_"
                "for_run(OUTPUT_DIR) to read this from downstream code."
            ),
            "patch_id": "TASK-44 (audit_codex MAJ-09)",
            "iso_date": "2026-04-27",
        }
        _fmd_path = os.path.join(OUTPUT_DIR, "fusion_mode_at_compute_opinions.json")
        with open(_fmd_path, "w", encoding="utf-8") as _f_fmd:
            _json_fmd.dump(_fusion_meta, _f_fmd, indent=2)
        print(f"   fusion_mode_at_compute_opinions.json saved -> "
              f"actual_fusion_mode='{_fusion_meta['actual_fusion_mode']}'")
    except Exception as _e_fmd:
        print(f"[WARN][MAJ-09] Could not write fusion_mode sidecar: {_e_fmd}")

    if os.environ.get("SL_SKIP_OPINION_PLOTS", "").strip().lower() in ("1", "true", "yes"):
        print("-> Opinion plots skipped via SL_SKIP_OPINION_PLOTS=1")
    else:
        print("-> Generating plots...")
        for k, v in viz_db.items():
            if not v['ops']:
                continue
            # Passer base_rate_hist uniquement pour les métriques feuilles
            br_hist = v['base_rate_hist'] if v['base_rate_hist'] else None
            plot_metric_complete(k, v['dates_win'], v['ops'],
                                 v['dates_raw'], v['real'], v['pred'],
                                 base_rates_history=br_hist)

    # Résumé final
    final_ops = viz_db["FINAL_SYSTEM_CBF"]['ops']
    if final_ops:
        last_op = final_ops[-1]
        print(f"\n    Dernière opinion système : {last_op}")
        print(f"    Prob. projetée : Safe={last_op.projected_prob()[0]:.3f}, "
              f"Susp={last_op.projected_prob()[1]:.3f}, "
              f"Atk={last_op.projected_prob()[2]:.3f}")

    # Diagnostic base rates finaux
    print("\n--- Base Rates adaptatifs finaux ---")
    if _edp_dict:
        print(f"   EDP actif : {len(_edp_dict)} priors chargés depuis l'artefact")

    print(f"\n✅ Opinions V3 computed. Results in '{OUTPUT_DIR}'")


if __name__ == "__main__":
    compute_opinions()
