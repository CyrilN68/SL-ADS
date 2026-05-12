"""
run_ablation_labeled.py — Ablation Study pour datasets avec vérité terrain
===========================================================================
Version adaptée de run_ablation_v2.py pour les nouveaux datasets (GECCO, CESNET...).
Remplace l'évaluation par catalogue d'attaques par une évaluation window-level
sur labels binaires réels.

ABLATION RUNS (composants SL désactivés un par un) :
    Full SL-ADS                  : référence production WBF
    No Ageing (λ=0)              : supprime la mémoire temporelle → effet C1
    Ageing λ sensibilité         : λ=0.50 / λ=0.99 vs λ=0.85
    Uniform Weights              : poids WBF uniformes (R² ignoré) → effet C3
    Prophet Only                 : supprime la branche RANSAC → effet CBF
    Reconst Only                 : supprime la branche Prophet → effet CBF
    WBF inter-method             : production duplicate / explicit WBF(P,R)
    No Conflict-Aware (C1 off)   : λ fixe sans reset de conflit
    No EDP (C4 off)              : prior uniforme au lieu du prior empirique

BASELINES CLASSIQUES (pour répondre à "un seuil simple peut-il rivaliser ?") :
    Static Threshold             : vote R²-pondéré sur colonnes _N (calibré sur labels)
    Best Single Metric           : meilleure colonne _N individuelle (oracle par métrique)
    Calibrated Vote (oracle)     : seuil optimal sur le vote statique (meilleur δ possible)
    Isolation Forest             : IF entraîné sur les fenêtres normales (split_date)
    Z-Score SPC                  : carte de contrôle Shewhart sur chaque _N

QUESTION CLÉ : SL au seuil opérationnel calibré vs baselines à leur δ optimal
               → si SL gagne, la SL apporte une valeur réelle
               → si perte, un seuil bien calibré suffit

OUTPUTS (dans RESULTS_DIR/ablation_labeled/) :
    ablation_lbl_summary.csv          — métriques au seuil opérationnel + meilleur δ
    ablation_lbl_sweeps.csv           — sweep complet (0.01→0.99) par run
    ablation_lbl_episodes.csv         — détail épisodes par run
    ablation_lbl_comparison.png       — table de comparaison publication-ready
    ablation_lbl_bar.png              — barres groupées F1-cov / F1-win / Precision / FPR
    ablation_lbl_f1curves.png         — courbes F1-cov vs δ pour tous les runs
    ablation_lbl_sl_gain.png          — scatter SL vs baseline : gain apporté par la SL

USAGE :
    python run_ablation_labeled.py
    python run_ablation_labeled.py --threshold 0.25   # forcer un seuil opérationnel
    python run_ablation_labeled.py --episode_gap 1    # tolérer 1 fenêtre dans un épisode

Ref: Meyes et al. (2019) arXiv:1901.08644 — Ablation Studies
     Tatbul et al. (2018) NeurIPS — Precision and Recall for Time Series
     Liu et al. (2008) ICDM — Isolation Forest
"""

import argparse
import os
import sys
import warnings
from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# PATCH TASK-28 (audit_tmp MAJ-07, 2026-04-26): targeted filters only.
warnings.filterwarnings("ignore", category=FutureWarning,      module=r"pandas")
warnings.filterwarnings("ignore", category=UserWarning,        module=r"matplotlib")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"matplotlib")

from sl_ads.paths import get_version_names, get_results_dir, get_model_path, get_decision_threshold, get_decision_variable  # Phase H

try:
    from sl_ads.config import CONFIG  # Phase H
except ImportError:
    print("❌ sl_ads.config introuvable."); sys.exit(1)

try:
    import sl_ads.core.subjective_logic as sl  # Phase H
except ImportError:
    print("❌ sl_ads.core.subjective_logic introuvable."); sys.exit(1)

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    print("⚠  scikit-learn absent — IsolationForest ignoré.")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
_EVAL          = CONFIG.get("EVAL", {})
VERSION_NAME, VERSION_MODIF = get_version_names(CONFIG)
RESULTS_DIR    = get_results_dir(CONFIG, up_levels=1)
OUTPUT_DIR     = os.path.join(RESULTS_DIR, "ablation_labeled")
EVIDENCE_CSV   = os.path.join(RESULTS_DIR, f"evidence_{VERSION_MODIF}.csv")
METADATA_CSV   = os.path.join(RESULTS_DIR, f"metadata_{VERSION_NAME}.csv")
STANDARDIZED_CSV = CONFIG.get("file_path", "")

OP_THRESHOLD   = get_decision_threshold(CONFIG, up_levels=1)   # auto-calibré depuis sidecar JSON
WINDOW_MIN     = _EVAL.get("WINDOW_MIN", 5)
W_BIJ          = CONFIG.get("SL_PARAM_K", 3.0)
CONFLICT_ALPHA = CONFIG.get("CONFLICT_ALPHA", 1.495)
ACTIVE_DATASET = CONFIG.get("ACTIVE_DATASET", "")
WINDOW_SLICES  = CONFIG.get("WINDOW_SIZE", 10)   # slices par fenêtre d'évidence

MODEL_PATH         = get_model_path(CONFIG, up_levels=1)
_models_pkg        = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else {}
_edp_dict          = (_models_pkg.get("empirical_priors")
                      if CONFIG.get("USE_EMPIRICAL_PRIOR", True) else None)
_DECISION_VARIABLE = get_decision_variable(CONFIG, up_levels=1)  # 'b_atk' ou 'proj_atk'

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
})


# ==============================================================================
# DÉFINITION DES RUNS D'ABLATION
# Chaque run désactive un composant SL précis (standard ablation study).
# ==============================================================================
_prior_a = np.array(CONFIG.get("SL_PRIOR_A", [1/3, 1/3, 1/3]))

# Référence : Full SL-ADS aligné sur la production actuelle.
# RedeRio utilise run_ablation.py ; ce runner est réservé aux datasets à labels
# natifs. Il garde néanmoins la même définition de "Full SL-ADS" pour éviter
# deux références scientifiques incompatibles dans le dépôt.
_REF = {
    "lambda": 0.85, "uniform": True, "prophet": True, "reconst": True,
    "cbf": False, "um": False, "conflict_aware": True, "adaptive_base_rate": True,
    "sl_param_k": W_BIJ, "c3_weight_mode": "uniform",
    "wbf_weight_mode": "uniform",
}

ABLATION_RUNS = OrderedDict([
    # ── Référence ────────────────────────────────────────────────────────────
    ("Full SL-ADS [référence production WBF]", {**_REF}),

    # ── C1 : Conflict-Aware Ageing ────────────────────────────────────────────
    ("No Ageing (λ=0) — C1",       {**_REF, "lambda": 0.00}),
    ("No Conflict-Aware — C1",      {**_REF, "conflict_aware": False}),
    ("Ageing λ=0.50",               {**_REF, "lambda": 0.50, "uniform": True}),
    ("Ageing λ=0.99",               {**_REF, "lambda": 0.99, "uniform": True}),

    # ── C3 : Pondération R² (qualité source) ─────────────────────────────────
    ("Poids uniformes [production duplicate] — C3", {**_REF}),

    # ── CBF : Fusion inter-méthode ────────────────────────────────────────────
    ("Prophet Only (no Reconst)",   {**_REF, "reconst": False}),
    ("Reconst Only (no Prophet)",   {**_REF, "prophet": False}),
    ("WBF inter-method [production duplicate]", {**_REF, "cbf": False, "uniform": True}),
    ("CBF inter-method [legacy sensitivity]", {**_REF, "cbf": True, "uniform": True}),

    # ── C4 : Empirical Dirichlet Prior ────────────────────────────────────────
    ("No EDP — prior uniforme (C4)",{**_REF, "adaptive_base_rate": False}),
])


# ==============================================================================
# MOTEUR DE FUSION SL (repris de run_ablation_v2.py)
# ==============================================================================

def run_sl_fusion(ev_df: pd.DataFrame, meta_df: pd.DataFrame,
                  run_cfg: dict) -> pd.DataFrame:
    """Exécute le pipeline SL complet selon la configuration du run."""
    _orig = getattr(sl, "_CONFLICT_MODE", "belief_mass")
    sl._CONFLICT_MODE = run_cfg.get("conflict_mode", _orig)
    try:
        return _fusion_inner(ev_df, meta_df, run_cfg)
    finally:
        sl._CONFLICT_MODE = _orig


def _fusion_inner(ev_df: pd.DataFrame, meta_df: pd.DataFrame,
                  run_cfg: dict) -> pd.DataFrame:
    lam            = run_cfg["lambda"]
    uniform        = run_cfg["uniform"]
    use_p          = run_cfg["prophet"]
    use_r          = run_cfg["reconst"]
    use_cbf        = run_cfg["cbf"]
    use_um         = run_cfg.get("um", False)
    conflict_aware = run_cfg.get("conflict_aware", True)
    adaptive_br    = run_cfg.get("adaptive_base_rate", True)
    W_run          = run_cfg.get("sl_param_k", W_BIJ)
    wbf_mode       = run_cfg.get("wbf_weight_mode",
                                  CONFIG.get("WBF_WEIGHT_MODE", "r2_static"))

    meta = {row["metric_key"]: {
                "clean_key": row["clean_key"],
                "type":      row["type"],
                "r2":        max(float(row["r2_weight"]), 0.01),
            } for _, row in meta_df.iterrows()}

    prophet_keys = [k for k, v in meta.items() if v["type"] == "prophet"   and use_p]
    reconst_keys = [k for k, v in meta.items() if v["type"] == "reconstruction" and use_r]
    all_active   = prophet_keys + reconst_keys

    if not all_active:
        raise ValueError("Aucune métrique active (vérifier use_prophet/use_reconst).")

    _balance_cfg = run_cfg.get("balance_ratio", CONFIG.get("BALANCE_RATIO", 1.0))
    _n_p, _n_r   = len(prophet_keys), len(reconst_keys)
    if _balance_cfg == "auto" and use_cbf and _n_r > 0:
        _bal_eff = float(_n_p) / float(_n_r)
    elif isinstance(_balance_cfg, (int, float)) and use_cbf:
        _bal_eff = float(_balance_cfg)
    else:
        _bal_eff = 1.0
    _bal_active = not np.isclose(_bal_eff, 1.0)

    trust_scores = _models_pkg.get("trust_scores", {}) if not uniform else {}
    state_mem    = {mk: np.zeros(3) for mk in all_active}
    records      = []

    for _, row in ev_df.iterrows():
        leaf_ops = {}
        for mk in all_active:
            ck    = meta[mk]["clean_key"]
            ev    = np.array([
                float(row.get(f"{ck}_P", 0.0) or 0.0),
                float(row.get(f"{ck}_S", 0.0) or 0.0),
                float(row.get(f"{ck}_N", 0.0) or 0.0),
            ])

            if lam > 0:
                R_new, _, _ = sl.temporal_adaptive_ageing(
                    r_accumulated=state_mem[mk], r_current=ev,
                    lam_base=lam, W=W_run,
                    alpha=run_cfg.get("conflict_alpha", CONFLICT_ALPHA),
                    conflict_aware=conflict_aware,
                )
                state_mem[mk] = R_new
            else:
                state_mem[mk] = ev  # No Ageing : mémoire effacée à chaque pas

            if adaptive_br and _edp_dict and mk in _edp_dict:
                e = _edp_dict[mk]
                a = np.array([e["a_safe"], e["a_susp"], e["a_atk"]])
            else:
                a = _prior_a.copy()

            op     = sl.evidence_to_opinion(state_mem[mk], W_run, a)
            r2_w   = max(float(meta[mk]["r2"]), 0.01)

            if wbf_mode == "trust_discount" and not uniform:
                t  = float(trust_scores.get(mk, a[0]))
                op = sl.apply_trust_discount(op, t)
                w  = 1.0
            else:
                w = 1.0 if uniform else r2_w

            if use_um:
                op = op.uncertainty_maximized()

            leaf_ops[mk] = (op, w)

        def wbf(keys):
            ops = [leaf_ops[k][0] for k in keys if k in leaf_ops]
            ws  = [leaf_ops[k][1] for k in keys if k in leaf_ops]
            if not ops:   return None
            if len(ops)==1: return ops[0]
            return sl.fusion_wbf_n_sources(ops, external_weights=ws)

        op_p = wbf(prophet_keys) if prophet_keys else None
        op_r = wbf(reconst_keys) if reconst_keys else None

        if op_p and op_r:
            if use_cbf:
                op_r_b = sl.boost_opinion_evidence(op_r, _bal_eff, W=W_BIJ) \
                         if _bal_active else op_r
                op_f = sl.fusion_cbf(op_p, op_r_b)
            else:
                op_f = sl.fusion_wbf_n_sources([op_p, op_r], external_weights=[1.0, 1.0])
        elif op_p:
            op_f = op_p
        elif op_r:
            op_f = op_r
        else:
            op_f = sl.evidence_to_opinion(np.zeros(3), W_BIJ, _prior_a)

        records.append({
            "timestamp": row["timestamp"],
            "proj_atk":  float(op_f.b[2] if _DECISION_VARIABLE == 'b_atk'
                              else op_f.projected_prob()[2]),
        })

    return pd.DataFrame(records)


# ==============================================================================
# BASELINES CLASSIQUES
# ==============================================================================

def baseline_static_threshold(ev_df: pd.DataFrame,
                               meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vote R²-pondéré sur colonnes _N : chaque métrique vote si son évidence négative
    dépasse 90% des slices de la fenêtre.
    Même logique que run_ablation_v2.py mais avec WINDOW_SLICES depuis config.
    """
    meta = {row["metric_key"]: {
                "clean_key": row["clean_key"],
                "r2": max(float(row["r2_weight"]), 0.01),
            } for _, row in meta_df.iterrows()}

    records = []
    for _, row in ev_df.iterrows():
        total_w = vote_w = 0.0
        for mk, m in meta.items():
            ck   = m["clean_key"]
            ev_N = float(row.get(f"{ck}_N", 0.0) or 0.0)
            w    = m["r2"]
            total_w += w
            if ev_N / max(WINDOW_SLICES, 1) > 0.90:
                vote_w += w
        records.append({"timestamp": row["timestamp"],
                         "proj_atk": vote_w / total_w if total_w > 0 else 0.0})
    return pd.DataFrame(records)


def baseline_best_single_metric(ev_df: pd.DataFrame,
                                 meta_df: pd.DataFrame,
                                 y_true: np.ndarray) -> tuple[pd.DataFrame, str]:
    """
    Teste chaque colonne _N comme signal d'anomalie unique.
    Retourne le DataFrame du meilleur score et son nom de métrique.
    'Meilleur' = meilleur F1-window au seuil optimal (oracle).
    """
    best_score = -1.0
    best_df    = pd.DataFrame()
    best_name  = ""

    for _, row_m in meta_df.iterrows():
        ck = row_m["clean_key"]
        col = f"{ck}_N"
        if col not in ev_df.columns:
            continue

        raw = ev_df[col].fillna(0.0).values
        # Normalise la colonne en [0,1] pour la rendre comparable à proj_atk
        vmax = raw.max()
        score_arr = raw / vmax if vmax > 1e-9 else raw

        result_df = pd.DataFrame({"timestamp": ev_df["timestamp"], "proj_atk": score_arr})

        # Aligne sur y_true (même longueur ici car même ev_df)
        thrs  = np.round(np.arange(0.01, 1.0, 0.02), 3)
        best_f1 = 0.0
        for thr in thrs:
            y_pred = (score_arr >= thr).astype(int)
            tp = int(((y_pred==1)&(y_true==1)).sum())
            fp = int(((y_pred==1)&(y_true==0)).sum())
            fn = int(((y_pred==0)&(y_true==1)).sum())
            p  = tp/(tp+fp) if (tp+fp)>0 else 0.0
            r  = tp/(tp+fn) if (tp+fn)>0 else 0.0
            f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
            if f1 > best_f1:
                best_f1 = f1

        if best_f1 > best_score:
            best_score = best_f1
            best_df    = result_df
            best_name  = ck

    return best_df, best_name


def baseline_zscore_spc(ev_df: pd.DataFrame, meta_df: pd.DataFrame,
                         split_date: pd.Timestamp) -> pd.DataFrame:
    """
    Carte de contrôle de Shewhart (Z-score) sur chaque _N.
    Entraîné sur la période avant split_date (trafic normal).
    Score = max(0, Z-score) normalisé et pondéré par R².
    """
    meta = {row["metric_key"]: {
                "clean_key": row["clean_key"],
                "r2": max(float(row["r2_weight"]), 0.01),
            } for _, row in meta_df.iterrows()}

    train_mask = ev_df["timestamp"] <= split_date
    stats      = {}
    for mk, m in meta.items():
        col = f"{m['clean_key']}_N"
        if col not in ev_df.columns:
            continue
        train_vals = ev_df.loc[train_mask, col].fillna(0.0).values
        if len(train_vals) > 1:
            stats[mk] = {"mu": train_vals.mean(), "sigma": max(train_vals.std(), 1e-6)}

    if not stats:
        # Fallback : retourner colonne nulle
        return pd.DataFrame({"timestamp": ev_df["timestamp"], "proj_atk": 0.0})

    records = []
    for _, row in ev_df.iterrows():
        total_w = z_w = 0.0
        for mk, m in meta.items():
            if mk not in stats:
                continue
            col  = f"{m['clean_key']}_N"
            val  = float(row.get(col, 0.0) or 0.0)
            z    = (val - stats[mk]["mu"]) / stats[mk]["sigma"]
            w    = m["r2"]
            total_w += w
            z_w += w * max(0.0, z)
        # Normalise par 3σ → score [0..1+] saturé
        score = min(1.0, (z_w / total_w) / 3.0) if total_w > 0 else 0.0
        records.append({"timestamp": row["timestamp"], "proj_atk": score})
    return pd.DataFrame(records)


def baseline_isolation_forest(ev_df: pd.DataFrame,
                               split_date: pd.Timestamp,
                               contamination: float = 0.05) -> pd.DataFrame:
    if not _HAS_SKLEARN:
        return pd.DataFrame()
    feat_cols = [c for c in ev_df.columns if c.endswith("_N")]
    if not feat_cols:
        return pd.DataFrame()

    X         = ev_df[feat_cols].fillna(0.0).values
    train_mask = ev_df["timestamp"] <= split_date
    X_train   = X[train_mask.values]

    if len(X_train) < 20:
        print(f"   ⚠  IF: seulement {len(X_train)} fenêtres avant split_date — skipping.")
        return pd.DataFrame()

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_all_s   = scaler.transform(X)

    clf = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=42, n_jobs=-1)
    clf.fit(X_train_s)

    # Score de décision (plus négatif = plus anormal), normalisé en [0,1]
    raw_scores = -clf.decision_function(X_all_s)  # plus haut = plus anormal
    vmin, vmax = raw_scores.min(), raw_scores.max()
    scores = (raw_scores - vmin) / (vmax - vmin + 1e-9)

    print(f"   IF entraîné sur {len(X_train)} fenêtres normales / {len(ev_df)} total "
          f"(contamination={contamination})")
    return pd.DataFrame({"timestamp": ev_df["timestamp"], "proj_atk": scores})


# ==============================================================================
# ÉVALUATION LABEL-BASED
# ==============================================================================

def find_episodes(df: pd.DataFrame, gap: int = 0) -> list:
    """Regroupe les fenêtres label=1 consécutives en épisodes."""
    episodes, ep_idx, in_ep, gap_count = [], [], False, 0
    for i, row in df.iterrows():
        if row["label"] == 1:
            ep_idx.append(i); in_ep = True; gap_count = 0
        elif in_ep:
            gap_count += 1
            if gap_count <= gap:
                ep_idx.append(i)
            else:
                episodes.append(_make_ep(df, ep_idx)); ep_idx = []; in_ep = False; gap_count = 0
    if ep_idx:
        episodes.append(_make_ep(df, ep_idx))
    return episodes


def _make_ep(df, idx):
    ts = df.loc[idx, "timestamp"]
    return {"start":      ts.iloc[0],
            "end":        ts.iloc[-1],
            "timestamps": set(ts.tolist()),  # lookup O(1) robuste aux reindex post-merge
            "indices":    idx,               # conservé pour compatibilité
            "n_anom":     int(df.loc[idx, "label"].sum())}


_f1 = lambda p, r: 2*p*r/(p+r) if (p+r) > 0 else 0.0


def sweep_metrics(result_df: pd.DataFrame, df_merged: pd.DataFrame,
                  episodes: list, thresholds: np.ndarray) -> pd.DataFrame:
    """
    Sweep de seuils : calcule les métriques fenêtre-niveau et épisode-niveau
    pour chaque seuil. Retourne un DataFrame avec toutes les métriques.
    """
    # Aligner result_df sur df_merged (inner join sur timestamp)
    df = df_merged.merge(
        result_df.rename(columns={"proj_atk": "score"}),
        on="timestamp", how="inner"
    )

    y_true  = df["label"].values
    y_score = df["score"].values
    rows    = []

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tp = int(((y_pred==1)&(y_true==1)).sum())
        fp = int(((y_pred==1)&(y_true==0)).sum())
        fn = int(((y_pred==0)&(y_true==1)).sum())
        tn = int(((y_pred==0)&(y_true==0)).sum())

        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        fpr  = fp/(fp+tn) if (fp+tn)>0 else 0.0
        f1w  = _f1(prec, rec)

        # Épisode-niveau
        n_ep = len(episodes)
        if n_ep > 0:
            cov_sum, ttd_scores, n_det_ep, n_det_wins = 0.0, [], 0, 0
            for ep in episodes:
                # Sélection par timestamp (robuste aux reindex après inner merge)
                ep_mask   = df["timestamp"].isin(ep["timestamps"])
                ep_scores = df.loc[ep_mask, "score"].values if ep_mask.any() else np.array([])
                if len(ep_scores) == 0:
                    ttd_scores.append(0.0); continue

                det_mask   = ep_scores >= thr
                n_d        = int(det_mask.sum())
                cov        = n_d / max(ep["n_anom"], 1)
                cov_sum   += cov
                if n_d > 0:
                    n_det_ep  += 1
                    n_det_wins += n_d
                    ttd_ratio  = min(1.0, int(np.argmax(det_mask)) * WINDOW_MIN /
                                     max(ep["n_anom"] * WINDOW_MIN, 1))
                    ttd_scores.append(cov * (1.0 - ttd_ratio))
                else:
                    ttd_scores.append(0.0)

            r_bin = n_det_ep / n_ep
            r_cov = cov_sum / n_ep
            r_ttd = float(np.mean(ttd_scores))
            prec_ep = n_det_wins/(n_det_wins+fp) if (n_det_wins+fp)>0 else 0.0
        else:
            r_bin = r_cov = r_ttd = prec_ep = 0.0

        rows.append({
            "threshold":       round(float(thr), 3),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision":       round(prec,          4),
            "recall":          round(rec,            4),
            "f1_window":       round(f1w,            4),
            "fpr_pct":         round(fpr * 100,      4),
            "recall_binary":   round(r_bin,          4),
            "recall_coverage": round(r_cov,          4),
            "recall_ttd":      round(r_ttd,          4),
            "f1_binary":       round(_f1(prec_ep, r_bin), 4),
            "f1_coverage":     round(_f1(prec_ep, r_cov), 4),
            "f1_ttd":          round(_f1(prec_ep, r_ttd), 4),
        })

    return pd.DataFrame(rows)


def summarize_run(run_name: str, sweep_df: pd.DataFrame,
                  op_thr: float, run_type: str = "sl") -> dict | None:
    """Extrait les métriques au seuil opérationnel + au meilleur seuil."""
    if sweep_df.empty:
        return None

    # Au seuil opérationnel
    op_row = sweep_df.iloc[(sweep_df["threshold"] - op_thr).abs().argmin()]

    # Au meilleur seuil F1-coverage (oracle)
    best_row = sweep_df.loc[sweep_df["f1_coverage"].idxmax()]

    return {
        "run":                   run_name,
        "run_type":              run_type,
        # Opérationnel δ=0.20
        "op_threshold":          op_thr,
        "op_f1_window":          float(op_row["f1_window"]),
        "op_f1_coverage":        float(op_row["f1_coverage"]),
        "op_f1_ttd":             float(op_row["f1_ttd"]),
        "op_precision":          float(op_row["precision"]),
        "op_recall":             float(op_row["recall"]),
        "op_recall_coverage":    float(op_row["recall_coverage"]),
        "op_fpr_pct":            float(op_row["fpr_pct"]),
        # Meilleur seuil possible (oracle)
        "best_threshold":        float(best_row["threshold"]),
        "best_f1_window":        float(sweep_df.loc[sweep_df["f1_window"].idxmax(), "f1_window"]),
        "best_f1_coverage":      float(best_row["f1_coverage"]),
        "best_f1_ttd":           float(best_row["f1_ttd"]),
        "best_precision":        float(best_row["precision"]),
        "best_recall_coverage":  float(best_row["recall_coverage"]),
        "best_fpr_pct":          float(best_row["fpr_pct"]),
    }


# ==============================================================================
# GRAPHIQUES
# ==============================================================================

def _color(run_name: str) -> str:
    palette = {
        "Full SL-ADS":         "#d62728",
        "No Ageing":           "#fbb4ae",
        "No Conflict":         "#fdae6b",
        "Ageing λ=0.50":       "#fdc086",
        "Ageing λ=0.99":       "#386cb0",
        "Poids uniformes":     "#2ca02c",
        "Prophet Only":        "#1f77b4",
        "Reconst Only":        "#9467bd",
        "WBF inter-method":    "#8c564b",
        "CBF inter-method":    "#bcbd22",
        "No EDP":              "#7f7f7f",
        "Static Threshold":    "#17becf",
        "Z-Score SPC":         "#bcbd22",
        "Best Single Metric":  "#e377c2",
        "Isolation Forest":    "#ff7f0e",
    }
    for key, color in palette.items():
        if key.lower() in run_name.lower():
            return color
    return "#aaaaaa"


def plot_f1_curves(all_sweeps: dict, op_thr: float, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for rname, sdf in all_sweeps.items():
        if sdf.empty or "f1_coverage" not in sdf.columns:
            continue
        is_ref     = "référence" in rname.lower()
        is_baseline = any(x in rname for x in ["Static", "Z-Score", "Best Single", "Isolation"])
        ls = "-" if is_ref else ("--" if not is_baseline else ":")
        lw = 2.5 if is_ref else (1.7 if not is_baseline else 1.3)
        ax.plot(sdf["threshold"], sdf["f1_coverage"],
                ls, lw=lw, color=_color(rname), label=rname,
                zorder=6 if is_ref else (4 if not is_baseline else 2))
    ax.axvline(op_thr, color="black", linestyle=":", lw=1.2,
               label=f"δ opérationnel = {op_thr:.2f}")
    ax.set_xlabel("Seuil de décision δ"); ax.set_ylabel("F1-coverage")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title(f"Ablation Study — F1-coverage vs δ\n"
                 f"{ACTIVE_DATASET or 'Brésil'} ({VERSION_NAME})\n"
                 f"Plein=référence SL  |  Tiret=ablations SL  |  Pointillé=baselines",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9, ncol=2)
    path = os.path.join(out_dir, "ablation_lbl_f1curves.png")
    fig.savefig(path); plt.close(fig)
    print(f"   -> {os.path.basename(path)}")


def plot_bar_comparison(summary_rows: list, op_thr: float, out_dir: str) -> None:
    """Barres groupées : SL à δ opérationnel vs baselines à leur meilleur δ."""
    if not summary_rows:
        return
    df   = pd.DataFrame(summary_rows)
    runs = df["run"].tolist()
    x    = np.arange(len(runs))
    w    = 0.20

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, len(runs)*1.1), 5.5))

    # --- Gauche : métriques au seuil opérationnel ---
    ax1.bar(x - w,   df["op_f1_coverage"], w, label="F1-cov (op.)",  color="#1f77b4", alpha=0.9)
    ax1.bar(x,       df["op_f1_window"],   w, label="F1-win (op.)",  color="#2ca02c", alpha=0.9)
    ax1.bar(x + w,   df["op_precision"],   w, label="Precision (op.)",color="#ff7f0e", alpha=0.9)
    ax1_r = ax1.twinx()
    ax1_r.plot(x, df["op_fpr_pct"], "v--", color="#d62728", lw=2, ms=7, label="FPR%")
    ax1_r.set_ylabel("FPR (%)", color="#d62728"); ax1_r.tick_params(axis="y", labelcolor="#d62728")
    ax1_r.set_ylim(0, max(df["op_fpr_pct"].max() * 3, 2.0))
    ax1.set_title(f"Au seuil opérationnel δ={op_thr:.2f}", fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(runs, rotation=22, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.12); ax1.set_ylabel("Score")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=8, loc="upper right")

    # --- Droite : métriques au meilleur δ (oracle) ---
    ax2.bar(x - w,   df["best_f1_coverage"], w, label="F1-cov (best δ)", color="#1f77b4", alpha=0.9)
    ax2.bar(x,       df["best_f1_window"],   w, label="F1-win (best δ)", color="#2ca02c", alpha=0.9)
    ax2.bar(x + w,   df["best_precision"],   w, label="Precision (best δ)",color="#ff7f0e",alpha=0.9)
    ax2_r = ax2.twinx()
    ax2_r.plot(x, df["best_fpr_pct"], "v--", color="#d62728", lw=2, ms=7, label="FPR%")
    ax2_r.set_ylabel("FPR (%)", color="#d62728"); ax2_r.tick_params(axis="y", labelcolor="#d62728")
    ax2_r.set_ylim(0, max(df["best_fpr_pct"].max() * 3, 2.0))
    ax2.set_title("Au meilleur δ (oracle — borne supérieure)", fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(runs, rotation=22, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.12); ax2.set_ylabel("Score")
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1+h2, l1+l2, fontsize=8, loc="upper right")

    n_sl = sum(1 for r in summary_rows if r["run_type"] == "sl")
    for ax in [ax1, ax2]:
        if n_sl < len(runs):
            ax.axvline(n_sl - 0.5, color="#aaaaaa", lw=1.2, linestyle=":")
            ax.text(n_sl - 0.45, 1.09, "← SL", fontsize=7, color="#555")
            ax.text(n_sl + 0.1,  1.09, "Baselines →", fontsize=7, color="#555")

    fig.suptitle(f"Ablation Study + Baselines — {ACTIVE_DATASET or 'Brésil'} ({VERSION_NAME})\n"
                 f"Gauche : δ opérationnel fixe   |   Droite : meilleur δ oracle",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_lbl_bar.png")
    fig.savefig(path); plt.close(fig)
    print(f"   -> {os.path.basename(path)}")


def plot_comparison_table(summary_rows: list, op_thr: float, out_dir: str) -> None:
    if not summary_rows:
        return
    df = pd.DataFrame(summary_rows)

    cols_h = ["Système", "Type",
              f"F1-win\n(δ={op_thr:.2f})", f"F1-cov\n(δ={op_thr:.2f})",
              f"Prec\n(δ={op_thr:.2f})", f"FPR%\n(δ={op_thr:.2f})",
              "Best δ", "F1-win\n(best)", "F1-cov\n(best)", "Prec\n(best)"]

    rows_data = []
    for _, r in df.iterrows():
        rows_data.append([
            r["run"], r["run_type"].upper(),
            f"{r['op_f1_window']:.3f}", f"{r['op_f1_coverage']:.3f}",
            f"{r['op_precision']:.3f}",  f"{r['op_fpr_pct']:.2f}",
            f"{r['best_threshold']:.2f}",
            f"{r['best_f1_window']:.3f}", f"{r['best_f1_coverage']:.3f}",
            f"{r['best_precision']:.3f}",
        ])

    hi_cols_op   = ["op_f1_window", "op_f1_coverage", "op_precision"]
    hi_cols_best = ["best_f1_window", "best_f1_coverage", "best_precision"]
    best_hi_op   = {c: df[c].max() for c in hi_cols_op}
    best_hi_best = {c: df[c].max() for c in hi_cols_best}
    best_lo_op   = df["op_fpr_pct"].min()
    best_lo_best = df["best_fpr_pct"].min()

    n_rows = len(rows_data)
    fig_h  = max(3.5, 0.55 * (n_rows + 1) + 1.8)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=rows_data, colLabels=cols_h,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.8)
    for j in range(len(cols_h)):
        c = tbl[0, j]; c.set_facecolor("#2c3e50")
        c.set_text_props(color="white", fontweight="bold")

    col_map_op   = {2: "op_f1_window", 3: "op_f1_coverage", 4: "op_precision", 5: "op_fpr_pct"}
    col_map_best = {7: "best_f1_window", 8: "best_f1_coverage", 9: "best_precision"}

    for i, (_, r) in enumerate(df.iterrows()):
        is_ref = "référence" in r["run"].lower()
        is_bl  = r["run_type"] == "baseline"
        bg     = "#fef3e2" if is_bl else ("#d0eaf8" if is_ref else "#e8f4fb")
        for j in range(len(cols_h)):
            tbl[i+1, j].set_facecolor(bg)
        # Surligner best
        for j, mk in col_map_op.items():
            val = r.get(mk)
            if val is None: continue
            if mk == "op_fpr_pct":
                if abs(val - best_lo_op) < 1e-6:
                    tbl[i+1,j].set_facecolor("#d4edda"); tbl[i+1,j].set_text_props(fontweight="bold")
            else:
                if abs(val - best_hi_op.get(mk, -1)) < 1e-6:
                    tbl[i+1,j].set_facecolor("#d4edda"); tbl[i+1,j].set_text_props(fontweight="bold")
        for j, mk in col_map_best.items():
            val = r.get(mk)
            if val is None: continue
            if abs(val - best_hi_best.get(mk, -1)) < 1e-6:
                tbl[i+1,j].set_facecolor("#d4edda"); tbl[i+1,j].set_text_props(fontweight="bold")

    ax.set_title(
        f"SL-ADS — Ablation Study & Baseline Comparison\n"
        f"{ACTIVE_DATASET or 'Brésil'} ({VERSION_NAME})  |  "
        f"Gauche: δ={op_thr:.2f} opérationnel  |  Droite: meilleur δ oracle\n"
        f"Cellules vertes = meilleure valeur par colonne",
        fontsize=10, fontweight="bold", pad=12
    )
    path = os.path.join(out_dir, "ablation_lbl_comparison.png")
    fig.savefig(path); plt.close(fig)
    print(f"   -> {os.path.basename(path)}")


def plot_sl_vs_baseline_gain(summary_rows: list, op_thr: float, out_dir: str) -> None:
    """
    Scatter : SL reference vs meilleure baseline sur F1-cov.
    X = meilleure baseline (best δ oracle), Y = SL à δ opérationnel.
    Point au-dessus de la diagonale → SL gagne même avec son seuil contraint.
    Point sous la diagonale → baseline calibrée rivalise ou dépasse.
    """
    if not summary_rows:
        return
    df = pd.DataFrame(summary_rows)
    ref = df[df["run"].str.contains("référence", case=False)]
    if ref.empty:
        return
    ref_val_op   = float(ref["op_f1_coverage"].iloc[0])
    ref_val_best = float(ref["best_f1_coverage"].iloc[0])

    baselines = df[df["run_type"] == "baseline"]
    if baselines.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(ref_val_op,   color="#d62728", linestyle="--", lw=1.5,
               label=f"SL δ={op_thr:.2f} (opérationnel) = {ref_val_op:.3f}")
    ax.axhline(ref_val_best, color="#d62728", linestyle=":",  lw=1.5,
               label=f"SL best δ (oracle) = {ref_val_best:.3f}")

    for _, r in baselines.iterrows():
        ax.scatter(r["best_f1_coverage"], r["best_f1_coverage"],
                   s=90, color=_color(r["run"]), zorder=5)
        ax.annotate(r["run"].split("(")[0].strip(),
                    (r["best_f1_coverage"], r["best_f1_coverage"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    # Barres horizontales pour chaque baseline
    for _, r in baselines.iterrows():
        ax.barh(r["run"], r["best_f1_coverage"],
                color=_color(r["run"]), alpha=0.7, height=0.4)

    ax.set_xlabel("F1-coverage au meilleur δ oracle")
    ax.set_title(f"Gain de la SL vs baselines calibrées\n"
                 f"Ligne rouge pleine = SL opérationnel (δ={op_thr:.2f}) — "
                 f"baselines ont leur δ optimal",
                 fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_lbl_sl_gain.png")
    fig.savefig(path); plt.close(fig)
    print(f"   -> {os.path.basename(path)}")


# ==============================================================================
# CONSOLE REPORT
# ==============================================================================

def print_report(summary_rows: list, op_thr: float) -> None:
    df  = pd.DataFrame(summary_rows)
    sep = "─" * 90

    print(f"\n{'='*90}")
    print(f"  ABLATION STUDY — {ACTIVE_DATASET or 'Brésil'} ({VERSION_NAME})")
    print(f"  Seuil opérationnel : δ={op_thr:.2f}   |   Colonne droite = meilleur δ oracle")
    print(f"{'='*90}")
    print(f"  {'Système':<44} {'F1-cov':>8} {'F1-win':>8} {'FPR%':>7}  |  {'best δ':>6} {'F1-cov*':>8} {'F1-win*':>8}")
    print(sep)

    for rtype, label in [("sl", "── SL Ablations ──"), ("baseline", "── Baselines classiques ──")]:
        sub = df[df["run_type"] == rtype]
        if sub.empty:
            continue
        print(f"  {label}")
        for _, r in sub.iterrows():
            marker = "◄ REF" if "référence" in r["run"].lower() else ""
            print(f"  {r['run']:<44} "
                  f"{r['op_f1_coverage']:>8.4f} "
                  f"{r['op_f1_window']:>8.4f} "
                  f"{r['op_fpr_pct']:>7.2f}  |  "
                  f"{r['best_threshold']:>6.2f} "
                  f"{r['best_f1_coverage']:>8.4f} "
                  f"{r['best_f1_window']:>8.4f}  {marker}")

    print(sep)
    ref = df[df["run"].str.contains("référence", case=False)]
    bls = df[df["run_type"] == "baseline"]
    if not ref.empty and not bls.empty:
        sl_op   = float(ref["op_f1_coverage"].iloc[0])
        sl_best = float(ref["best_f1_coverage"].iloc[0])
        bl_best = float(bls["best_f1_coverage"].max())
        gain_op   = sl_op   - bl_best
        gain_best = sl_best - bl_best
        print(f"\n  ► Gain SL (δ opérationnel) vs meilleure baseline (oracle) : {gain_op:+.4f}")
        print(f"  ► Gain SL (meilleur δ)     vs meilleure baseline (oracle) : {gain_best:+.4f}")
        if gain_op > 0:
            print(f"  ✅ La SL bat la meilleure baseline même avec son seuil opérationnel contraint.")
        else:
            print(f"  ⚠  La meilleure baseline calibrée rivalise avec la SL à δ={op_thr:.2f}.")
            print(f"     → Recalibrer le seuil (δ={float(ref['best_threshold'].iloc[0]):.2f} optimal) "
                  f"ou vérifier la qualité des features.")
    print(f"{'='*90}\n")


# ==============================================================================
# CHARGEMENT DES DONNÉES
# ==============================================================================

def load_evidence(ev_path: str, split_date: pd.Timestamp = None) -> pd.DataFrame:
    ev_raw = pd.read_csv(ev_path, parse_dates=["timestamp"]).sort_values("timestamp")
    if split_date is not None:
        ev_raw = ev_raw[ev_raw["timestamp"] > split_date].copy()
    psn  = [c for c in ev_raw.columns if c.endswith(("_P","_S","_N"))]
    qual = [c for c in ev_raw.columns if c.endswith(("_iw","_rmse"))]
    kw   = dict(origin="start_day", closed="left", label="left")
    ev_p = ev_raw.set_index("timestamp")[psn].resample("5min", **kw).sum().reset_index()
    if qual:
        ev_q = ev_raw.set_index("timestamp")[qual].resample("5min", **kw).mean().reset_index()
        return ev_p.merge(ev_q, on="timestamp", how="left")
    return ev_p


def load_and_align_labels(standardized_csv, window_min, split_date=None):
    df = pd.read_csv(standardized_csv, parse_dates=["timestamp"])
    if split_date is not None:
        df = df[df["timestamp"] > split_date].copy()
    if "label" not in df.columns:
        print(f"❌ Colonne 'label' absente de {os.path.basename(standardized_csv)}.")
        sys.exit(1)
    return (df.set_index("timestamp")["label"]
              .resample(f"{window_min}min", origin="start_day", closed="left", label="left")
              .max().fillna(0).astype(int).reset_index())


# ==============================================================================
# MAIN
# ==============================================================================

def main(op_thr: float, episode_gap: int) -> None:
    print(f"\n{'='*70}")
    print(f"  run_ablation_labeled.py — {ACTIVE_DATASET or 'Brésil'} ({VERSION_NAME})")
    print(f"  δ opérationnel : {op_thr:.2f}  |  Episode gap : {episode_gap} fenêtres")
    print(f"{'='*70}\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Evidence + metadata
    ev_path = EVIDENCE_CSV
    if not os.path.exists(ev_path):
        alt = ev_path.replace(VERSION_MODIF, VERSION_NAME)
        if os.path.exists(alt):
            print(f"⚠  Evidence injectée absente → fallback sur {os.path.basename(alt)}")
            ev_path = alt
        else:
            print(f"❌ {EVIDENCE_CSV}\n   Lancez d'abord inject_at_evidence_level.py."); sys.exit(1)
    if not os.path.exists(METADATA_CSV):
        print(f"❌ {METADATA_CSV}"); sys.exit(1)

    print(f"-> Chargement evidence : {os.path.basename(ev_path)}")
    split_date = pd.Timestamp(CONFIG.get("split_date", "2025-01-01"))
    ev_df = load_evidence(ev_path, split_date=split_date)
    meta_df = pd.read_csv(METADATA_CSV)
    print(f"   {len(ev_df)} fenêtres | {len(meta_df)} métriques")

    # 2. Labels
    if not STANDARDIZED_CSV or not os.path.exists(STANDARDIZED_CSV):
        print(f"❌ CSV standardisé introuvable : '{STANDARDIZED_CSV}'"); sys.exit(1)
    print(f"-> Chargement labels : {os.path.basename(STANDARDIZED_CSV)}")
    df_lab  = load_and_align_labels(STANDARDIZED_CSV, WINDOW_MIN)
    df_merged = pd.merge(ev_df[["timestamp"]], df_lab, on="timestamp", how="inner").reset_index(drop=True)
    y_true  = df_merged["label"].values
    print(f"   {len(df_merged)} fenêtres alignées | {int(y_true.sum())} anormales "
          f"({100*y_true.mean():.2f}%)")

    # 3. Épisodes
    episodes = find_episodes(df_merged, gap=episode_gap)
    print(f"   {len(episodes)} épisodes anormaux détectés")

    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 3)
    split_date = pd.Timestamp(CONFIG.get("split_date", "2025-01-01"))

    ev_df_test = ev_df[ev_df["timestamp"] > split_date].copy()
    df_merged = pd.merge(
        ev_df_test[["timestamp"]],
        df_lab,
        on="timestamp",
        how="inner"
    ).reset_index(drop=True)
    y_true = df_merged["label"].values
    print(f"-> Hold-out strict appliqué : {len(ev_df_test)} fenêtres après {split_date.date()}")
    print(f"   ({int(y_true.sum())} anormales — {100 * y_true.mean():.2f}%)")

    all_sweeps, summary_rows, all_ep_rows = {}, [], []

    def register(run_name, result_df, run_type="sl"):
        if result_df is None or result_df.empty:
            return
        sdf = sweep_metrics(result_df, df_merged, episodes, thresholds)
        if sdf.empty:
            return
        all_sweeps[run_name] = sdf
        row = summarize_run(run_name, sdf, op_thr, run_type)
        if row:
            summary_rows.append(row)
            print(f"   δ={op_thr:.2f}: F1-cov={row['op_f1_coverage']:.3f}  "
                  f"F1-win={row['op_f1_window']:.3f}  FPR={row['op_fpr_pct']:.2f}%  |  "
                  f"best δ={row['best_threshold']:.2f}: F1-cov={row['best_f1_coverage']:.3f}")

    # ── Phase 1 : SL Ablations ────────────────────────────────────────────────
    print(f"\n{'='*52}\n  PHASE 1 — SL Ablations\n{'='*52}")
    for run_name, run_cfg in ABLATION_RUNS.items():
        print(f"\n-> [{run_name}]")
        try:
            result_df = run_sl_fusion(ev_df, meta_df, run_cfg)
            register(run_name, result_df, "sl")
        except Exception as e:
            print(f"   ⚠  Skipped : {e}")

    # ── Phase 2 : Baselines classiques ────────────────────────────────────────
    print(f"\n{'='*52}\n  PHASE 2 — Baselines classiques\n{'='*52}")

    print("\n-> [Static Threshold (vote R² sur _N)]")
    try:
        register("Static Threshold (R²-vote)",
                 baseline_static_threshold(ev_df_test, meta_df), "baseline")
    except Exception as e:
        print(f"   ⚠  {e}")

    print("\n-> [Z-Score SPC (Shewhart)]")
    try:
        register("Z-Score SPC",
                 baseline_zscore_spc(ev_df_test, meta_df, split_date), "baseline")
    except Exception as e:
        print(f"   ⚠  {e}")

    print("\n-> [Best Single Metric (oracle par métrique)]")
    try:
        bsm_df, bsm_name = baseline_best_single_metric(ev_df, meta_df, y_true)
        if not bsm_df.empty:
            print(f"   Meilleure métrique : {bsm_name}")
            register(f"Best Single Metric ({bsm_name})", bsm_df, "baseline")
    except Exception as e:
        print(f"   ⚠  {e}")

    for cont in [0.01, 0.05, 0.10]:
        print(f"\n-> [Isolation Forest (c={cont})]")
        try:
            if_df = baseline_isolation_forest(ev_df_test, split_date, contamination=cont)
            if not if_df.empty:
                register(f"Isolation Forest (c={cont})", if_df, "baseline")
        except Exception as e:
            print(f"   ⚠  {e}")

    # ── Rapport + Sauvegarde ─────────────────────────────────────────────────
    print_report(summary_rows, op_thr)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(OUTPUT_DIR, "ablation_lbl_summary.csv"), index=False)

    if all_sweeps:
        frames = []
        for rname, sdf in all_sweeps.items():
            sdf = sdf.copy(); sdf.insert(0, "run", rname)
            frames.append(sdf)
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(OUTPUT_DIR, "ablation_lbl_sweeps.csv"), index=False)

    print("-> Génération des graphiques...")
    plot_f1_curves(all_sweeps, op_thr, OUTPUT_DIR)
    plot_bar_comparison(summary_rows, op_thr, OUTPUT_DIR)
    plot_comparison_table(summary_rows, op_thr, OUTPUT_DIR)
    plot_sl_vs_baseline_gain(summary_rows, op_thr, OUTPUT_DIR)

    # ── Phase 3 : Analyse Rush / Off-Peak (METR-LA uniquement) ───────────────
    # Stratification temporelle pour tester si SL-ADS est plus robuste que IF
    # en heures de pointe (trafic structuré → Prophet plus précis).
    # Ref : Daganzo (1994) Transportation Research B — fundamental diagram shifts.
    #       HCM (2010) §10-3 — heures de pointe LA : 7-9h et 17-19h PST.
    # Note : il ne s'agit PAS de tuning mais d'évaluation stratifiée post-hoc.
    # Aucun hyperparamètre n'est modifié → pas de data leakage.
    if ACTIVE_DATASET == "METR-LA" and all_sweeps:
        print(f"\n{'='*52}\n  PHASE 3 — Analyse Rush / Off-Peak (METR-LA)\n{'='*52}")

        def _split_rush_offpeak(df_with_ts: pd.DataFrame):
            """Stratifie par heure locale (PST, UTC-8)."""
            hour = pd.to_datetime(df_with_ts['timestamp']).dt.hour
            rush_mask = ((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))
            return df_with_ts[rush_mask].copy(), df_with_ts[~rush_mask].copy()

        df_rush, df_offpeak = _split_rush_offpeak(df_merged)
        ep_rush    = find_episodes(df_rush,    gap=episode_gap)
        ep_offpeak = find_episodes(df_offpeak, gap=episode_gap)

        n_anom_rush    = int(df_rush["label"].sum())
        n_anom_offpeak = int(df_offpeak["label"].sum())
        print(f"   Rush    (7-9h, 17-19h) : {len(df_rush):5d} fenêtres | "
              f"{n_anom_rush} anomalies | {len(ep_rush)} épisodes")
        print(f"   Off-peak               : {len(df_offpeak):5d} fenêtres | "
              f"{n_anom_offpeak} anomalies | {len(ep_offpeak)} épisodes")

        if n_anom_rush == 0 or n_anom_offpeak == 0:
            print("   ⚠  Pas assez d'anomalies dans l'une des périodes → analyse ignorée.")
        else:
            rush_rows, offpeak_rows = [], []
            # On évalue uniquement Full SL (référence) et Best Single Metric
            runs_to_stratify = {k: v for k, v in all_sweeps.items()
                                if "référence" in k or "Best Single" in k}

            for run_name, _ in runs_to_stratify.items():
                # Récupérer le result_df depuis l'ablation déjà calculée
                if run_name not in all_sweeps:
                    continue
                # Recalculer sweep sur sous-ensemble rush / off-peak
                # On recharge le score depuis all_sweeps (colonne proj_atk dans ev_df)
                # En pratique on re-fusionne le score SL sur les timestamps filtrés
                # → utilisation de la colonne "score" dans les sweeps déjà calculés
                sweep_full = all_sweeps[run_name]
                # Fallback : si pas de données, skip
                if sweep_full.empty:
                    continue

                best_row_rush    = None
                best_row_offpeak = None

                # On recalcule depuis ev_df_test pour avoir les scores sur les sous-périodes
                try:
                    if "référence" in run_name:
                        result_df_run = run_sl_fusion(ev_df, meta_df, ABLATION_RUNS[run_name])
                    else:
                        # Best Single Metric — recalculer
                        result_df_run, _ = baseline_best_single_metric(ev_df, meta_df,
                                                                        df_merged["label"].values)
                    if result_df_run is None or result_df_run.empty:
                        continue

                    sdf_rush    = sweep_metrics(result_df_run, df_rush,    ep_rush,    thresholds)
                    sdf_offpeak = sweep_metrics(result_df_run, df_offpeak, ep_offpeak, thresholds)

                    def _best(sdf):
                        if sdf.empty:
                            return None
                        return sdf.loc[sdf["f1_coverage"].idxmax()]

                    br, bo = _best(sdf_rush), _best(sdf_offpeak)
                    if br is not None:
                        rush_rows.append({
                            "run": run_name, "period": "rush",
                            "best_f1_cov": round(float(br["f1_coverage"]), 4),
                            "best_threshold": round(float(br["threshold"]), 3),
                            "fpr_pct": round(float(br["fpr_pct"]), 3),
                            "n_anomalies": n_anom_rush,
                        })
                    if bo is not None:
                        offpeak_rows.append({
                            "run": run_name, "period": "off-peak",
                            "best_f1_cov": round(float(bo["f1_coverage"]), 4),
                            "best_threshold": round(float(bo["threshold"]), 3),
                            "fpr_pct": round(float(bo["fpr_pct"]), 3),
                            "n_anomalies": n_anom_offpeak,
                        })
                except Exception as e:
                    print(f"   ⚠  [{run_name}] skipped rush/offpeak : {e}")

            all_strat = rush_rows + offpeak_rows
            if all_strat:
                df_strat = pd.DataFrame(all_strat)
                out_strat = os.path.join(OUTPUT_DIR, "ablation_lbl_rush_offpeak.csv")
                df_strat.to_csv(out_strat, index=False)
                print(f"\n  Rush vs Off-Peak (meilleur δ oracle) :")
                print(f"  {'Run':<35} {'Period':<10} {'F1-cov':>7} {'δ':>6} {'FPR%':>7}")
                print(f"  {'-'*65}")
                for _, row in df_strat.iterrows():
                    print(f"  {row['run']:<35} {row['period']:<10} "
                          f"{row['best_f1_cov']:>7.4f} {row['best_threshold']:>6.3f} "
                          f"{row['fpr_pct']:>7.3f}%")
                print(f"\n   → {out_strat}")

    print(f"\n✅ Résultats dans : {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None,
                        help=f"Seuil opérationnel (défaut: DECISION_THRESHOLD={OP_THRESHOLD})")
    parser.add_argument("--episode_gap", type=int, default=0,
                        help="Fenêtres normales tolérées dans un épisode (défaut: 0)")
    args = parser.parse_args()
    main(
        op_thr      = args.threshold if args.threshold is not None else OP_THRESHOLD,
        episode_gap = args.episode_gap,
    )
