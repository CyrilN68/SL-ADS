"""
evaluate_injection.py — Complete IDS Evaluation Pipeline
=========================================================
Ref : Sharafaldin et al. (2018) ICISSP — CIC-IDS2017 (TP/FP/FN framework)
      Jøsang (2016) Subjective Logic — Chap. 12 (opinions, base rates)
      Tatbul et al. (2018) NeurIPS — Precision and Recall for Time Series
      Ferling et al. (2022) — Time-Aware Evaluation of Network IDS

All parameters are read from config.py → CONFIG["EVAL"].
Set CATALOG_MODE = "injected" to use synthetic ATTACK_CATALOG from
inject_at_evidence_level.py, or "real" to use REAL_ATTACK_CATALOG from config.py.

MODIFICATION: Évaluation stricte opérationnelle.
    - Seuil fixé en dur à 0.20 pour refléter les performances réelles sans optimisation a posteriori.

OUTPUTS:
    eval_detection_summary.csv     — per-attack x threshold metrics
    eval_threshold_sweep.csv       — precision/recall/F1 variants per threshold
    eval_baserate_audit.csv        — adaptive base rate evolution per attack
    eval_learning_comparison.csv   — R1 vs R2+ comparison per attack family
    graphs/attack_{NAME}.png       — publication-ready attack timeline
    graphs/threshold_sweep.png     — precision/recall/F1 sweep (pub-quality)
    graphs/summary_table.png       — summary table for article insertion
"""

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from sl_ads.paths import get_version_names, get_results_dir  # Phase H

# PATCH-M2 fix (2026-04-19) : imports sklearn au niveau module pour le
# bootstrap IC 95 % (Efron 1979) appliqué à F1 et MCC.
from sklearn.metrics import f1_score, matthews_corrcoef
from sl_ads.evaluate.vus_metrics import vus_summary
from sl_ads.stats.bootstrap_ci import bootstrap_bca_ci

# PATCH TASK-28 (audit_tmp MAJ-07, 2026-04-26)
# ──────────────────────────────────────────────────────────────────────────
# L'ancien ``warnings.filterwarnings("ignore")`` global masquait toutes les
# alertes (numpy RuntimeWarning, pandas SettingWithCopyWarning, sklearn
# deprecation, etc.) — y compris des problèmes scientifiquement
# significatifs (divisions par zéro silencieuses, dépréciations de signature).
# On ne supprime désormais que les warnings spécifiquement non actionnables
# pour ce script, en laissant remonter tout le reste.
warnings.filterwarnings("ignore", category=FutureWarning,    module=r"pandas")
warnings.filterwarnings("ignore", category=UserWarning,      module=r"matplotlib")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"matplotlib")

# ==============================================================================
# CONFIGURATION — all parameters loaded from config.py
# ==============================================================================
try:
    from sl_ads.config import CONFIG  # Phase H
except ImportError:
    print("CRITICAL: sl_ads.config not found.")
    sys.exit(1)

_EVAL = CONFIG.get("EVAL", {})
VERSION_NAME, _ = get_version_names(CONFIG)

RESULTS_DIR           = get_results_dir(CONFIG, up_levels=1)
# Auto-detect CSV name: compute_opinions_v3 writes "detection_results_INJECTED.csv"
# when injection is active, otherwise "detection_results.csv".
# We mirror the same priority: prefer _INJECTED if it exists.
_base_csv    = _EVAL.get("RESULTS_CSV_NAME", "detection_results.csv")
_inj_csv     = _base_csv.replace(".csv", "_INJECTED.csv")
_inj_path    = os.path.join(RESULTS_DIR, _inj_csv)
_base_path   = os.path.join(RESULTS_DIR, _base_csv)
RESULTS_CSV  = _inj_path if os.path.exists(_inj_path) else _base_path
OUTPUT_DIR            = os.path.join(RESULTS_DIR, "evaluation")
WINDOW_MIN            = _EVAL.get("WINDOW_MIN", 5)
CONTEXT_H             = _EVAL.get("CONTEXT_H", 2.0)

# Coherence check: WINDOW_MIN must match WINDOW_SIZE × FREQ
def _freq_to_sec(f: str) -> float:
    f = f.lower().strip()
    if f.endswith('min'): return int(f[:-3]) * 60
    elif f.endswith('h'):  return int(f[:-1]) * 3600
    elif f.endswith('s'):  return int(f[:-1])
    return 300
_computed_window_min = CONFIG.get('WINDOW_SIZE', 10) * _freq_to_sec(CONFIG.get("SELECTED_FREQ", "30s")) / 60.0
if abs(_computed_window_min - WINDOW_MIN) > 0.1:
    import warnings as _w
    _w.warn(
        f"WINDOW_MIN={WINDOW_MIN} min from config but WINDOW_SIZE×FREQ={_computed_window_min:.1f} min. "
        f"TTD-in-windows will be incorrect. Auto-correcting.",
        stacklevel=1
    )
    WINDOW_MIN = _computed_window_min

# Seuil auto-calibré à l'entraînement (EVT/FPR), fallback config.py
from sl_ads.paths import get_decision_threshold, get_detection_col  # Phase H
_decision_thr = get_decision_threshold(CONFIG, up_levels=1)
THRESHOLDS = [_decision_thr]

COL_DET = _EVAL.get("COL_DET", _EVAL.get("COL_PROJ_ATK", get_detection_col(CONFIG, up_levels=1)))
COL_BSUSP             = _EVAL.get("COL_BSUSP",    "FINAL_SYSTEM_CBF_b_susp")
COL_BSAFE             = _EVAL.get("COL_BSAFE",    "FINAL_SYSTEM_CBF_b_safe")
COL_PROJ_ATK          = _EVAL.get("COL_PROJ_ATK", "FINAL_SYSTEM_CBF_proj_atk")
LEAF_METRICS_TO_AUDIT = _EVAL.get("LEAF_METRICS_TO_AUDIT", [
    "P_bytes", "P_packets", "P_flows", "P_syn",
    "P_entropy_src_ip", "P_entropy_src_port", "P_entropy_dst_port", "P_avg_pkt_size",
    "R_bytes_to_packets", "R_bytes_to_entropy_src_port",
])
CATALOG_MODE = _EVAL.get("CATALOG_MODE", "injected")
LAMBDA_DECAY = CONFIG.get("LAMBDA_DECAY", 0.85)

# UM is always False in the injection evaluation pipeline (disabled upstream in run_ablation.py).
# Read from config for display only — does not affect any computation here.
UNCERTAINTY_MAXIMIZATION = CONFIG.get("UNCERTAINTY_MAXIMIZATION", False)

# ==============================================================================
# ATTACK CATALOG — "injected": from inject_at_evidence_level.py
#                  "real":     from CONFIG["EVAL"]["REAL_ATTACK_CATALOG"]
#                  + INCLUDE_REAL_ATTACK flag to append real attack(s)
# ==============================================================================
def load_attack_catalog():
    """Load attack catalog based on CATALOG_MODE + optional REAL_ATTACK append."""
    catalog = []

    if CATALOG_MODE == "real":
        catalog = list(_EVAL.get("REAL_ATTACK_CATALOG", []))
        print(f"CATALOG_MODE=real — {len(catalog)} attack(s) loaded from config.py")
    else:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from sl_ads.inject.evidence_level import ATTACK_CATALOG as _INJECTED  # Phase H
            catalog = list(_INJECTED)
            print(f"CATALOG_MODE=injected — {len(catalog)} synthetic attack(s) loaded")
        except ImportError:
            print("WARNING: Could not import inject_at_evidence_level. Set CATALOG_MODE='real' in config.py.")

        # Append real attack(s) if requested
        if _EVAL.get("INCLUDE_REAL_ATTACK", True):
            real_attacks = _EVAL.get("REAL_ATTACK_CATALOG", [])
            if real_attacks:
                catalog = catalog + real_attacks
                print(f"  + {len(real_attacks)} real attack(s) appended -> {len(catalog)} total")

    return catalog

ATTACK_CATALOG = load_attack_catalog()

# Import make_ramp for ramp-phase stratified coverage (S7)
try:
    from sl_ads.inject.evidence_level import make_ramp as _make_ramp  # Phase H
    _HAVE_MAKE_RAMP = True
except ImportError:
    _HAVE_MAKE_RAMP = False

# ==============================================================================
# PLOT STYLE — publication quality
# ==============================================================================
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
})

INTENSITY_COLORS = {
    "extreme": "#d62728",
    "high":    "#ff7f0e",
    "medium":  "#1f77b4",
    "low":     "#2ca02c",
}
INTENSITY_LABELS = {
    "extreme": "Extreme",
    "high":    "High",
    "medium":  "Medium",
    "low":     "Low",
}


# ==============================================================================
# UTILITIES
# ==============================================================================

def attack_family(name: str) -> str:
    return re.sub(r"_R\d+$", "", name)


def attack_occurrence(name: str) -> int:
    m = re.search(r"_R(\d+)$", name)
    return int(m.group(1)) if m else 1


def theoretical_ttd_windows(ev_safe: float, ev_attack: float,
                             lam: float, threshold_b: float,
                             W: float = None, max_win: int = 50) -> int:
    """
    Theoretical detection delay (windows) via analytical simulation.

    Single-metric upper-bound model (best-case for the most discriminant metric).
    Uses the Jøsang (2016) Eq. 16.5 exponential evidence accumulation:
        r_k = λ · r_{k-1} + e_k
    where e_k = (ev_safe, 0, ev_attack) at each window k.

    Parameters
    ----------
    ev_safe   : normalized safe evidence of the peak metric (→ r[0], col_P convention)
    ev_attack : normalized attack evidence of the peak metric (→ r[2], col_N convention)
    lam       : exponential decay factor λ (LAMBDA_DECAY from config)
    threshold_b : detection threshold on b_atk = r[2] / (r.sum() + W)
    W         : SL prior strength. MUST equal the number of multinomial states
                (Jøsang 2016, Def. 3.9 : W = |X|). Pour le modèle ternaire
                Safe/Susp/Attack on a |X| = 3. Si None, lu depuis
                CONFIG["SL_PARAM_K"] avec fallback 3.0 (patch M-02 / F05,
                2026-04-21).
    max_win   : maximum windows before returning max_win (not detected)

    Returns
    -------
    int : number of windows until first detection, or max_win if not detected.

    Note: This is a theoretical lower-bound on TTD (best-case single metric).
    The real system fuses N metrics via WBF+CBF, which may converge faster or
    slower depending on inter-metric agreement (see ttd_gap_windows in outputs).
    """
    # PATCH M-02 / F05 (2026-04-21) : W doit correspondre à SL_PARAM_K (nombre
    # d'états multinomiaux Safe/Susp/Attack). L'ancien défaut W=2.0 produisait
    # une borne théorique incohérente avec la calibration réelle K=3.
    if W is None:
        W = float(CONFIG.get("SL_PARAM_K", 3.0))
    r = np.zeros(3)
    for k in range(max_win):
        r = lam * r + np.array([ev_safe, 0.0, ev_attack])
        D = r.sum() + W
        b_atk = r[2] / D  # r[2] = attack evidence (N column convention)
        if b_atk >= threshold_b:
            return k + 1
    return max_win


def windows_in_attack(df: pd.DataFrame, atk: dict) -> pd.Index:
    t0   = pd.Timestamp(atk["start"])
    t1   = t0 + pd.Timedelta(hours=atk["duration_h"])
    mask = (df["timestamp"] >= t0) & (df["timestamp"] < t1)
    return df.index[mask]


def _event_bounds(ev: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    t0 = pd.Timestamp(ev["start"])
    if ev.get("end") is not None:
        return t0, pd.Timestamp(ev["end"])
    return t0, t0 + pd.Timedelta(hours=float(ev["duration_h"]))


def _real_attacks_iter():
    """Iterate REAL_ATTACKS entries (may be empty in standalone test setups)."""
    try:
        from sl_ads.config import REAL_ATTACKS as _RA  # noqa: WPS433 — lazy
    except Exception:  # pragma: no cover - defensive import
        return
    for events in (_RA or {}).values():
        for ev in events or []:
            yield ev


def windows_outside_attacks(df: pd.DataFrame, catalog: list) -> pd.Index:
    """Indices of windows that are *not* in any catalog event nor in any
    REAL_ATTACKS event (including planned outages).

    A3.2 fix: the threshold-sweep FPR denominator must exclude both
    explicitly-injected attacks (the catalog) and operational outage
    intervals from REAL_ATTACKS. Otherwise outage windows that the
    detector correctly reacts to inflate the headline FPR by 5×–10×
    compared to the regime-by-regime breakdown produced by
    `evaluate_regime_fpr.py`.
    """
    outside = pd.Series(True, index=df.index)
    for atk in catalog or []:
        t0, t1 = _event_bounds(atk)
        outside &= ~((df["timestamp"] >= t0) & (df["timestamp"] < t1))
    for ev in _real_attacks_iter():
        t0, t1 = _event_bounds(ev)
        outside &= ~((df["timestamp"] >= t0) & (df["timestamp"] < t1))
    return df.index[outside]


def _event_window_mask(df: pd.DataFrame, events) -> pd.Series:
    """Boolean mask for arbitrary timestamp events, de-duplicated by union."""
    mask = pd.Series(False, index=df.index)
    for ev in events or []:
        t0, t1 = _event_bounds(ev)
        mask |= (df["timestamp"] >= t0) & (df["timestamp"] < t1)
    return mask


def _catalog_window_mask(df: pd.DataFrame, catalog: list) -> pd.Series:
    return _event_window_mask(df, catalog or [])


def _real_attack_window_mask(df: pd.DataFrame) -> pd.Series:
    return _event_window_mask(df, list(_real_attacks_iter()))


def _outage_window_mask(df: pd.DataFrame) -> pd.Series:
    try:
        from sl_ads.qualif_filters import is_outage_event
    except Exception:
        is_outage_event = lambda _ev: False  # noqa: E731
    outage_events = [ev for ev in _real_attacks_iter() if is_outage_event(ev)]
    return _event_window_mask(df, outage_events)


def _binary_window_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Canonical binary window-level metrics from explicit TP/FP/TN/FN."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    p_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = 2 * p_pos * r_pos / (p_pos + r_pos) if (p_pos + r_pos) > 0 else 0.0

    p_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    r_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg = 2 * p_neg * r_neg / (p_neg + r_neg) if (p_neg + r_neg) > 0 else 0.0

    n = tp + tn + fp + fn
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "n_windows": n,
        "n_positive": tp + fn,
        "n_negative": tn + fp,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision_window": p_pos,
        "tpr_window": r_pos,
        "fpr_window": fpr,
        "fpr_pct": 100.0 * fpr,
        "f1_micro_pure": f1_pos,
        "f1_macro_pure": (f1_pos + f1_neg) / 2.0,
        "accuracy": (tp + tn) / n if n > 0 else 0.0,
        "mcc": (tp * tn - fp * fn) / denom if denom > 0 else 0.0,
    }


def f1_protocol_comparison(df: pd.DataFrame, catalog: list,
                           threshold: float) -> pd.DataFrame:
    """
    Compare the two publication F1 protocols selected for reporting.

    `catalog_outages_separate` matches the historical A3.2/evaluate_injection
    metric: catalog attacks are positives and outage windows are excluded from
    the F1 base, then reported separately.

    `operator_faithful_anomaly` treats all known real incidents from
    REAL_ATTACKS, including NETWORK_OUTAGE_* windows, as anomaly positives.
    This is the honest IDS/operator view: a network outage is not benign.
    """
    catalog_mask = _catalog_window_mask(df, catalog or [])
    outage_mask = _outage_window_mask(df)
    real_mask = _real_attack_window_mask(df)
    anomaly_mask = catalog_mask | real_mask
    y_pred_full = (df[COL_DET].fillna(0.0).values >= threshold).astype(np.int8)

    fpr_target = float(CONFIG.get("FPR_TARGET_DECISION", np.nan))
    rows = []
    specs = [
        (
            "catalog_outages_separate",
            "catalog attacks positive; outage windows excluded from F1 base",
            catalog_mask,
            ~outage_mask,
        ),
        (
            "operator_faithful_anomaly",
            "catalog plus REAL_ATTACKS positive; outages are anomaly positives",
            anomaly_mask,
            pd.Series(True, index=df.index),
        ),
    ]

    for protocol, description, true_mask, valid_mask in specs:
        y_true = true_mask.values.astype(np.int8)
        valid = valid_mask.values.astype(bool)
        metrics = _binary_window_metrics(y_true[valid], y_pred_full[valid])
        ratio = (
            metrics["fpr_window"] / fpr_target
            if np.isfinite(fpr_target) and fpr_target > 0
            else np.nan
        )
        rows.append({
            "protocol": protocol,
            "description": description,
            "threshold": float(threshold),
            "fpr_target": fpr_target if np.isfinite(fpr_target) else np.nan,
            "fpr_ratio_to_target": ratio if np.isfinite(ratio) else np.nan,
            **metrics,
        })

    out = pd.DataFrame(rows)
    numeric_cols = [
        "threshold", "fpr_target", "fpr_ratio_to_target",
        "precision_window", "tpr_window", "fpr_window", "fpr_pct",
        "f1_micro_pure", "f1_macro_pure", "accuracy", "mcc",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = out[col].astype(float).round(6)
    return out


def median_attack_block_length(df: pd.DataFrame, catalog: list) -> int:
    """Median episode length in sampled windows for moving-block bootstrap."""
    lengths = [len(windows_in_attack(df, atk)) for atk in catalog or []]
    lengths = [n for n in lengths if n > 1]
    if not lengths:
        return 1
    return int(max(2, min(int(np.median(lengths)), len(df))))


# ==============================================================================
# AXIS 1+2 — Detection and Time-To-Detect per attack x threshold
# ==============================================================================

def evaluate_detection(df: pd.DataFrame, catalog: list) -> pd.DataFrame:
    rows       = []
    outside_idx = windows_outside_attacks(df, catalog)
    n_outside   = len(outside_idx)

    for atk in catalog:
        gt_idx = windows_in_attack(df, atk)
        if len(gt_idx) == 0:
            continue

        gt_df    = df.loc[gt_idx]
        b_atk_gt = gt_df[COL_DET].fillna(0.0).values
        t_start  = pd.Timestamp(atk["start"])

        # Theoretical TTD: single-metric model using the most discriminant metric.
        # Convention from inject_at_evidence_level: (ev_attack, ev_suspect, ev_normal)
        #   ev_attack → col_N → r[2] (attack evidence in SL)
        #   ev_normal → col_P → r[0] (safe evidence in SL)
        # Values must be normalized (WINDOW_SIZE scale) as actually injected.
        # Real attacks (REAL_ATTACK_CATALOG) may not carry a "signature" — skip TTD theory.
        _inj_W = CONFIG.get('WINDOW_SIZE', 10)
        _best_ev_atk_norm, _best_ev_safe_norm = 0.0, 0.0
        for _ev_a, _ev_s, _ev_n in atk.get("signature", {}).values():
            _total = _ev_a + _ev_s + _ev_n
            if _total <= 0:
                continue
            _scale = _inj_W / _total
            _ev_a_n = _ev_a * _scale  # normalized attack evidence (→ r[2])
            _ev_n_n = _ev_n * _scale  # normalized safe evidence   (→ r[0])
            if _ev_a_n > _best_ev_atk_norm:
                _best_ev_atk_norm = _ev_a_n
                _best_ev_safe_norm = _ev_n_n

        for thr in THRESHOLDS:
            detected_mask = b_atk_gt >= thr
            n_detected    = detected_mask.sum()
            detected      = n_detected > 0
            coverage      = 100.0 * n_detected / len(gt_idx)

            # Ramp-phase stratified coverage: plateau windows only (alpha >= 0.8).
            # Distinguishes detection failures in the ramp phase from plateau failures.
            # A system that only detects during the plateau but misses the ramp should
            # have high coverage_plateau but lower coverage_pct overall.
            coverage_plateau = np.nan
            if _HAVE_MAKE_RAMP and len(gt_idx) > 0 and "ramp_frac" in atk:
                _profile      = _make_ramp(len(gt_idx), atk["ramp_frac"])
                _plateau_mask = np.array([a >= 0.8 for a in _profile])
                _plateau_b    = b_atk_gt[_plateau_mask]
                if len(_plateau_b) > 0:
                    coverage_plateau = 100.0 * (_plateau_b >= thr).sum() / len(_plateau_b)

            if detected:
                first_det_pos = np.argmax(detected_mask)
                t_first_det   = gt_df["timestamp"].iloc[first_det_pos]
                ttd_min       = (t_first_det - t_start).total_seconds() / 60.0
                ttd_win       = int(ttd_min / WINDOW_MIN)
            else:
                ttd_min = np.nan
                ttd_win = np.nan

            ttd_theo = theoretical_ttd_windows(_best_ev_safe_norm, _best_ev_atk_norm, LAMBDA_DECAY, thr)
            fp_count = (df.loc[outside_idx, COL_DET].fillna(0.0) >= thr).sum()

            # b_susp during attack (measures SL "uncertain-but-suspicious" signal)
            b_susp_gt = gt_df[COL_BSUSP].fillna(0.0).values if COL_BSUSP in gt_df.columns else np.array([0.0])

            rows.append({
                "name":          atk["name"],
                "family":        attack_family(atk["name"]),
                "occurrence":    attack_occurrence(atk["name"]),
                "type":          atk["type"],
                "intensity":     atk["intensity"],
                "duration_h":    atk["duration_h"],
                "n_gt_windows":  len(gt_idx),
                "threshold":     thr,
                "detected":      detected,
                "n_detected":    n_detected,
                "coverage_pct":       round(coverage, 1),
                "coverage_plateau_pct": round(coverage_plateau, 1) if not np.isnan(coverage_plateau) else np.nan,
                "ttd_windows":   ttd_win,
                "ttd_minutes":   round(ttd_min, 1) if not np.isnan(ttd_min) else np.nan,
                "ttd_theo_win":  ttd_theo,
                "max_b_atk":     round(float(b_atk_gt.max()), 4),
                "mean_b_atk":    round(float(b_atk_gt.mean()), 4),
                "ttd_gap_windows": (int(ttd_win - ttd_theo)
                                    if (not np.isnan(ttd_win) and ttd_win is not np.nan)
                                    else np.nan),
                "mean_b_susp_during": round(float(b_susp_gt.mean()), 4),
                "max_b_susp_during":  round(float(b_susp_gt.max()),  4),
                "fp_outside":    int(fp_count),
                "n_outside_win": n_outside,
                "fpr_pct":       round(100.0 * fp_count / n_outside, 2) if n_outside > 0 else 0.0,
            })

    return pd.DataFrame(rows)


# ==============================================================================
# GLOBAL SWEEP — Precision / Recall (3 variants) / F1 per threshold
# ==============================================================================

def global_threshold_sweep(df_eval: pd.DataFrame, df: pd.DataFrame,
                           catalog: list = None) -> pd.DataFrame:
    """
    Three recall/F1 variants:
      - Binary recall   : attack detected (0 or 1) — standard IDS literature
      - Coverage recall : weighted by coverage_pct (Ferling et al. 2022)
      - TTD recall      : coverage x (1 - TTD/duration) — penalizes late detection
    """
    rows = []

    # Use provided catalog (valid_catalog from main) to avoid including out-of-range attacks.
    _catalog = catalog if catalog is not None else ATTACK_CATALOG

    # Window-level binary ground truth (pure classification view):
    # y_true = 1 inside an attack window, 0 otherwise.
    gt_mask = pd.Series(False, index=df.index)
    for atk in _catalog:
        atk_idx = windows_in_attack(df, atk)
        gt_mask.loc[atk_idx] = True
    y_true = gt_mask.values.astype(int)

    # A3.2 — REAL_ATTACKS outage windows must be excluded from the FPR
    # denominator: they are operational network outages (not attacks, not
    # quiet normal traffic), and treating them as "normal" inflates FPR.
    # The tri-class partition (attack/outage/normal) used by
    # _compute_global_detection_stats in evaluate_qualify_sbn.py is the
    # canonical fix; we apply the same logic here at the threshold-sweep level.
    outage_mask = pd.Series(False, index=df.index)
    try:
        from sl_ads.qualif_filters import is_outage_event
    except Exception:
        is_outage_event = lambda _ev: False  # noqa: E731
    for ev in _real_attacks_iter():
        if not is_outage_event(ev):
            continue
        t0, t1 = _event_bounds(ev)
        outage_mask |= (df["timestamp"] >= t0) & (df["timestamp"] < t1)
    valid_mask = ~outage_mask.values

    for thr, g in df_eval.groupby("threshold"):
        n_attacks_total = len(g)

        # Binary recall
        n_tp          = g["detected"].sum()
        recall_binary = n_tp / n_attacks_total if n_attacks_total > 0 else 0.0

        # Coverage-weighted recall
        recall_coverage = (g["coverage_pct"].fillna(0.0).sum() / 100.0 / n_attacks_total
                           if n_attacks_total > 0 else 0.0)

        # TTD-penalized recall — formal derivation:
        #   score_i = cov_i × (1 − TTD_i / duration_i)
        #
        #   cov_i ∈ [0,1]: fraction of attack windows above threshold (coverage recall).
        #   (1 − TTD_i/duration_i) ∈ [0,1]: temporal bonus for early detection.
        #     = 1 if detected at t=0 (immediate), = 0 if detected at t=duration (too late).
        #
        #   Interpretation: score_i is the fraction of the attack duration that was
        #   effectively covered by early detection. Averaging over attacks gives the
        #   mean "temporally-discounted coverage" across the evaluation catalog.
        #
        #   Edge cases: cov=0 → score=0 (missed); TTD > duration → clamped at 0.
        #   Relation to Tatbul et al. (2018): analogous to their ExistenceReward ×
        #   OverlapReward, extended with a temporal penalty term. Not directly
        #   equivalent but consistent in spirit with time-series-aware evaluation.
        scores_ttd = []
        for _, row in g.iterrows():
            cov = row["coverage_pct"] / 100.0
            if row["detected"] and not np.isnan(row["ttd_minutes"]):
                ttd_ratio = min(1.0, row["ttd_minutes"] / (row["duration_h"] * 60.0))
            else:
                ttd_ratio = 1.0
            scores_ttd.append(cov * (1.0 - ttd_ratio))
        recall_ttd = float(np.mean(scores_ttd)) if scores_ttd else 0.0

        # ── Metric design rationale (hybrid precision×recall) ──────────────────
        # Precision at WINDOW level: each falsely flagged window is an operational
        #   false alarm (SOC alert cost), so window-granularity is the right denominator.
        # Recall at ATTACK level: operationally, what matters is whether an attack
        #   episode was detected at all (binary 0/1 per attack), not how many windows
        #   were flagged. This is consistent with Sharafaldin et al. (2018) IDS eval.
        # Pure window-level equivalents (f1_micro_pure, f1_macro_pure) are provided
        #   separately for direct comparability with ML binary classification literature.
        fp_out    = g["fp_outside"].iloc[0]
        n_out     = g["n_outside_win"].iloc[0]
        tp_win    = g["n_detected"].sum()
        precision = tp_win / (tp_win + fp_out) if (tp_win + fp_out) > 0 else 0.0

        # Pure F1 metrics (literature-comparable): computed on the
        # outage-excluded subset (A3.2). Outage windows are neither
        # attacks (so y_true=0) nor true normal traffic; counting them
        # toward TN/FP would mis-state both the FPR and the F1.
        y_pred_full = (df[COL_DET].fillna(0.0).values >= thr).astype(int)
        y_true_eval = y_true[valid_mask]
        y_pred = y_pred_full[valid_mask]
        tp = int(((y_true_eval == 1) & (y_pred == 1)).sum())
        tn = int(((y_true_eval == 0) & (y_pred == 0)).sum())
        fp = int(((y_true_eval == 0) & (y_pred == 1)).sum())
        fn = int(((y_true_eval == 1) & (y_pred == 0)).sum())

        # Positive-class F1 (binary attack-vs-normal).
        p_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_pos = 2 * p_pos * r_pos / (p_pos + r_pos) if (p_pos + r_pos) > 0 else 0.0

        # Negative-class F1 (normal windows) for macro-F1.
        p_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        r_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_neg = 2 * p_neg * r_neg / (p_neg + r_neg) if (p_neg + r_neg) > 0 else 0.0

        # Accuracy (TP+TN)/(TP+TN+FP+FN)
        # Note: misleading on imbalanced datasets — prefer MCC or F1.
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        # Macro-F1 = unweighted mean of per-class F1 scores
        f1_macro = (f1_pos + f1_neg) / 2.0
        # Micro-F1 in binary = F1 of positive class (by definition)
        f1_micro = f1_pos

        # ─── PATCH-M2 (2026-04-18 / fix 2026-04-19) : Bootstrap IC 95 % (Efron 1979) ──
        # Variables corrigées : y_true / y_pred (pas y_true_all / y_pred_all
        # qui n'existaient pas dans ce scope). Imports f1_score / matthews_corrcoef
        # déplacés au niveau module (cf. début du fichier).
        N_BOOTSTRAP = 1000
        bootstrap_block_len = median_attack_block_length(df, _catalog)

        def _bootstrap_ci(y_true_arr, y_pred_arr, metric_fn, n=N_BOOTSTRAP):
            """IC 95 % percentile bootstrap sur n samples."""
            scores = []
            idx_all = np.arange(len(y_true_arr))
            for _ in range(n):
                idx = RNG.choice(idx_all, size=len(idx_all), replace=True)
                try:
                    s = metric_fn(y_true_arr[idx], y_pred_arr[idx])
                    if not np.isnan(s):
                        scores.append(s)
                except Exception:
                    continue
            if len(scores) < n // 2:
                return (float('nan'), float('nan'), float('nan'))
            lo, hi = np.percentile(scores, [2.5, 97.5])
            return (float(np.mean(scores)), float(lo), float(hi))

        # Bootstrap CI on the outage-excluded vectors (A3.2 consistency).
        f1_ci = bootstrap_bca_ci(
            y_true_eval, y_pred, f1_score,
            n_boot=N_BOOTSTRAP, seed=42, block_length=bootstrap_block_len,
        )
        mcc_ci = bootstrap_bca_ci(
            y_true_eval, y_pred, matthews_corrcoef,
            n_boot=N_BOOTSTRAP, seed=43, block_length=bootstrap_block_len,
        )
        f1_mean, f1_lo, f1_hi = f1_ci["point"], f1_ci["ci_low"], f1_ci["ci_high"]
        mcc_mean, mcc_lo, mcc_hi = mcc_ci["point"], mcc_ci["ci_low"], mcc_ci["ci_high"]
        print(f"  thr={thr:.2f} | F1 = {f1_mean:.3f} [IC95% : {f1_lo:.3f} – {f1_hi:.3f}]"
              f" | MCC = {mcc_mean:.3f} [IC95% : {mcc_lo:.3f} – {mcc_hi:.3f}]")
        print(f"       CI resampling: {f1_ci['resampling']} "
              f"(block_length={bootstrap_block_len} windows)")
        # NB : les colonnes f1_mean_boot / f1_ci_lo / f1_ci_hi / mcc_mean_boot /
        # mcc_ci_lo / mcc_ci_hi sont ajoutées directement à la ligne du sweep
        # ci-dessous (cf. rows.append), de sorte que eval_threshold_sweep.csv
        # contient nativement les IC sans fichier supplémentaire.

        # Matthews Correlation Coefficient — recommended for imbalanced classes
        # (Chicco & Jurman 2020, BMC Genomics). MCC ∈ [-1, 1], MCC=1 = perfect.
        # Invariant to class imbalance unlike accuracy or F1.
        _mcc_denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = (tp * tn - fp * fn) / _mcc_denom if _mcc_denom > 0 else 0.0

        # ROC operating point at this threshold (window-level TPR and FPR).
        # Full ROC/AUC requires a threshold sweep; this gives the single operating point.
        tpr_win = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_win = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fpr_target = float(CONFIG.get("FPR_TARGET_DECISION", np.nan))
        fpr_ratio_to_target = (
            fpr_win / fpr_target
            if np.isfinite(fpr_target) and fpr_target > 0
            else np.nan
        )
        fpr_target_status = (
            "EXCEEDS_2X_TARGET_RECALIBRATE_OR_JUSTIFY"
            if np.isfinite(fpr_ratio_to_target) and fpr_ratio_to_target > 2.0
            else "OK"
        )
        if fpr_target_status != "OK":
            print(
                f"  [A1.9] empirical FPR={fpr_win:.6f} exceeds "
                f"target={fpr_target:.6f} by {fpr_ratio_to_target:.1f}x."
            )

        def f1(p, r): return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        # PATCH TASK-21 (audit_tmp CRIT-03, 2026-04-26)
        # ──────────────────────────────────────────────────────────────────────
        # Les colonnes `f1_binary_hybrid_episode_recall` et
        # `f1_coverage_hybrid_episode_recall` mélangent volontairement deux
        # unités d'analyse (precision_window × recall_episode) ; elles sont
        # historiquement utilisées en évaluation IDS (Sharafaldin et al. 2018)
        # mais ne sont **pas** des F1 canoniques. Pour la conformité avec la
        # littérature ML standard (Chicco & Jurman 2020), `f1_micro_pure` et
        # `f1_macro_pure` (window-level pur, dérivés de TP/FP/TN/FN) sont les
        # quantités à reporter dans les tableaux principaux.
        # Les anciens noms `f1_binary` / `f1_coverage` sont **conservés comme
        # alias rétro-compatibles** afin que les CSV historiques restent
        # lisibles, mais une note explicite figure dans le summary report.
        f1_binary_hybrid    = f1(precision, recall_binary)
        f1_coverage_hybrid  = f1(precision, recall_coverage)
        f1_ttd_hybrid       = f1(precision, recall_ttd)

        rows.append({
            "threshold":           thr,
            "n_attacks":           n_attacks_total,
            "n_detected_attacks":  int(n_tp),
            "n_missed_attacks":    int(n_attacks_total - n_tp),
            "recall_binary":       round(recall_binary,   4),
            "recall_coverage":     round(recall_coverage, 4),
            "recall_ttd":          round(recall_ttd,      4),
            "precision_window":    round(precision,       4),
            # ── Hybrid metrics (precision_window × recall_episode) ─────────
            #     Reporting-only — see CRIT-03 disclosure in §5.3 of paper.
            "f1_binary_hybrid_episode_recall":   round(f1_binary_hybrid,   4),
            "f1_coverage_hybrid_episode_recall": round(f1_coverage_hybrid, 4),
            "f1_ttd_hybrid_episode_recall":      round(f1_ttd_hybrid,      4),
            # ── Canonical pure metrics (window-level only) ────────────────
            #     Catalog/outages-separate protocol. For publication, cite
            #     these together with eval_f1_protocol_comparison.csv.
            "f1_micro_pure": round(f1_micro, 4),  # = f1_pos (binary, window-level)
            "f1_macro_pure": round(f1_macro, 4),
            "accuracy":      round(accuracy, 4),
            "mcc":           round(mcc,      4),   # Matthews Correlation Coefficient
            "tpr_window":    round(tpr_win,  4),   # ROC operating point — TPR
            "fpr_window":    round(fpr_win,  4),   # ROC operating point — FPR
            # ── DEPRECATED aliases (kept for downstream-script compatibility) ─
            #     Will be removed in v11. Use *_hybrid_episode_recall instead.
            "fpr_target":    round(fpr_target, 6) if np.isfinite(fpr_target) else np.nan,
            "fpr_ratio_to_target": (
                round(float(fpr_ratio_to_target), 2)
                if np.isfinite(fpr_ratio_to_target) else np.nan
            ),
            "fpr_target_status": fpr_target_status,
            "f1_binary":           round(f1_binary_hybrid,   4),
            "f1_coverage":         round(f1_coverage_hybrid, 4),
            "f1_ttd":              round(f1_ttd_hybrid,      4),
            "recall_attack":       round(recall_binary, 4),
            "f1_score":            round(f1_binary_hybrid, 4),
            # ── Operational counts ─────────────────────────────────────────
            "fp_windows":          int(fp_out),
            "fpr_pct":             round(100.0 * fp_out / n_out, 2) if n_out > 0 else 0.0,
            # ── PATCH-M2 : Bootstrap IC 95 % (Efron 1979) ──────────────────
            "f1_mean_boot":        round(f1_mean,  4),
            "f1_ci_lo":            round(f1_lo,    4),
            "f1_ci_hi":            round(f1_hi,    4),
            "mcc_mean_boot":       round(mcc_mean, 4),
            "mcc_ci_lo":           round(mcc_lo,   4),
            "mcc_ci_hi":           round(mcc_hi,   4),
            "n_bootstrap":         N_BOOTSTRAP,
            "bootstrap_seed":      42,
            "bootstrap_method":    f1_ci["method"],
            "bootstrap_resampling": f1_ci["resampling"],
            "bootstrap_block_length": bootstrap_block_len,
        })

    return pd.DataFrame(rows).sort_values("threshold")


# ==============================================================================
# AXIS 3 — Adaptive base rate audit
# ==============================================================================

def audit_base_rates(df: pd.DataFrame, catalog: list) -> pd.DataFrame:
    rows = []
    for atk in catalog:
        t0 = pd.Timestamp(atk["start"])
        t1 = t0 + pd.Timedelta(hours=atk["duration_h"])
        t_post = t1 + pd.Timedelta(hours=CONTEXT_H)

        gt_mask   = (df["timestamp"] >= t0) & (df["timestamp"] < t1)
        post_mask = (df["timestamp"] >= t1) & (df["timestamp"] < t_post)

        for ck in LEAF_METRICS_TO_AUDIT:
            col      = f"{ck}_a_atk"
            col_proj = f"{ck}_proj_atk"
            if col not in df.columns:
                continue

            a_during    = df.loc[gt_mask,   col].fillna(np.nan).values
            a_post      = df.loc[post_mask, col].fillna(np.nan).values
            a_proj_vals = (df.loc[gt_mask, col_proj].fillna(np.nan).values
                           if col_proj in df.columns else np.array([]))
            n_30min  = max(1, int(30 / WINDOW_MIN))
            a_post30 = a_post[:n_30min] if len(a_post) >= n_30min else a_post

            rows.append({
                "name":                 atk["name"],
                "metric":               ck,
                "a_atk_before":         round(float(df.loc[df["timestamp"] < t0, col].dropna().iloc[-1])
                                              if df.loc[df["timestamp"] < t0, col].dropna().shape[0] > 0
                                              else np.nan, 4),
                "a_atk_max_during":     round(float(np.nanmax(a_during)),  4) if len(a_during) > 0 else np.nan,
                "a_atk_mean_during":    round(float(np.nanmean(a_during)), 4) if len(a_during) > 0 else np.nan,
                "a_atk_30min_after":    round(float(np.nanmean(a_post30)), 4) if len(a_post30) > 0 else np.nan,
                "proj_atk_max_during":  round(float(np.nanmax(a_proj_vals)),  4) if len(a_proj_vals) > 0 else np.nan,
                "proj_atk_mean_during": round(float(np.nanmean(a_proj_vals)), 4) if len(a_proj_vals) > 0 else np.nan,
                "rises_during":         bool(np.nanmax(a_during) > (float(
                                             df.loc[df["timestamp"] < t0, col].dropna().iloc[-1]
                                         ) if df.loc[df["timestamp"] < t0, col].dropna().shape[0] > 0 else 0))
                                        if len(a_during) > 0 else False,
            })
    return pd.DataFrame(rows)


# ==============================================================================
# AXIS 4 — Repetition learning comparison
# ==============================================================================

def learning_comparison(df_eval: pd.DataFrame, df: pd.DataFrame,
                         catalog: list) -> pd.DataFrame:
    family_map: dict = {}
    for atk in catalog:
        family_map.setdefault(attack_family(atk["name"]), []).append(atk)

    rows = []
    ref_thr = THRESHOLDS[len(THRESHOLDS) // 2]

    for fam, members in family_map.items():
        if len(members) < 2:
            continue
        for atk in sorted(members, key=lambda a: attack_occurrence(a["name"])):
            occ    = attack_occurrence(atk["name"])
            gt_idx = windows_in_attack(df, atk)
            if len(gt_idx) == 0:
                continue
            b_vals        = df.loc[gt_idx, COL_DET].fillna(0.0).values
            t_start       = pd.Timestamp(atk["start"])
            detected_mask = b_vals >= ref_thr
            ttd_min = np.nan
            if detected_mask.any():
                t_first = df.loc[gt_idx, "timestamp"].iloc[np.argmax(detected_mask)]
                ttd_min = max(0.0, (t_first - t_start).total_seconds() / 60.0)
            a_atk_vals = []
            for ck in LEAF_METRICS_TO_AUDIT:
                col = f"{ck}_a_atk"
                if col in df.columns:
                    vals = df.loc[gt_idx, col].fillna(np.nan).values
                    a_atk_vals.extend(vals[~np.isnan(vals)].tolist())
            rows.append({
                "family":            fam,
                "occurrence":        occ,
                "name":              atk["name"],
                "threshold":         ref_thr,
                "detected":          bool(detected_mask.any()),
                "coverage_pct":      round(100.0 * detected_mask.sum() / len(gt_idx), 1),
                "ttd_minutes":       round(ttd_min, 1) if not np.isnan(ttd_min) else np.nan,
                "max_b_atk":         round(float(b_vals.max()), 4),
                "mean_a_atk_during": round(float(np.mean(a_atk_vals)), 4) if a_atk_vals else np.nan,
            })

    if not rows:
        return pd.DataFrame()

    df_lr    = pd.DataFrame(rows)
    deltas   = []
    for fam, g in df_lr.groupby("family"):
        g  = g.sort_values("occurrence")
        r1 = g[g["occurrence"] == 1]
        if r1.empty:
            continue
        r1 = r1.iloc[0]
        for _, rx in g[g["occurrence"] > 1].iterrows():
            deltas.append({
                "family":         fam,
                "occurrence":     rx["occurrence"],
                "delta_ttd_min":  round(r1["ttd_minutes"] - rx["ttd_minutes"], 1)
                                  if not (np.isnan(r1["ttd_minutes"]) or np.isnan(rx["ttd_minutes"]))
                                  else np.nan,
                "delta_coverage": round(rx["coverage_pct"] - r1["coverage_pct"], 1),
                "delta_max_b_atk":round(rx["max_b_atk"] - r1["max_b_atk"], 4),
                "delta_mean_a_atk": round(rx["mean_a_atk_during"] - r1["mean_a_atk_during"], 4)
                                    if not (np.isnan(r1["mean_a_atk_during"])
                                            or np.isnan(rx["mean_a_atk_during"])) else np.nan,
                "comment_ttd": "improved" if (not np.isnan(r1["ttd_minutes"] - rx["ttd_minutes"])
                                              and r1["ttd_minutes"] - rx["ttd_minutes"] > 0)
                               else "degraded / stable",
            })
    return pd.merge(df_lr, pd.DataFrame(deltas), on=["family", "occurrence"], how="left")


# ==============================================================================
# FIGURES — publication quality, English labels, 300 dpi
# ==============================================================================

def plot_attack_timeline(df: pd.DataFrame, atk: dict, output_dir: str):
    """Per-attack detection timeline. Top: belief masses. Bottom: ground truth."""
    t0  = pd.Timestamp(atk["start"])
    t1  = t0 + pd.Timedelta(hours=atk["duration_h"])
    t_s = t0 - pd.Timedelta(hours=CONTEXT_H)
    t_e = t1 + pd.Timedelta(hours=CONTEXT_H)

    sub = df.loc[(df["timestamp"] >= t_s) & (df["timestamp"] <= t_e)].copy()
    if sub.empty:
        return

    col_color = INTENSITY_COLORS.get(atk["intensity"], "gray")
    dates     = sub["timestamp"]

    fig = plt.figure(figsize=(14, 7))
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[4, 1], hspace=0.06)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.step(dates, sub[COL_DET].fillna(0),  where="post",
             color="#d62728", lw=2.2, label=r"$b_{atk}$ (CBF system)", zorder=5)
    ax1.step(dates, sub[COL_BSUSP].fillna(0), where="post",
             color="#ff7f0e", lw=1.4, alpha=0.75, label=r"$b_{susp}$")
    ax1.step(dates, sub[COL_BSAFE].fillna(0), where="post",
             color="#2ca02c", lw=1.2, alpha=0.55, label=r"$b_{safe}$")
    if COL_PROJ_ATK in sub.columns:
        ax1.step(dates, sub[COL_PROJ_ATK].fillna(0), where="post",
                 color="#8b0000", lw=1.2, linestyle="--", alpha=0.7,
                 label=r"$P(atk)$ projected")

    ax1.axvspan(t0, t1, alpha=0.10, color=col_color,
                label=f"Attack window ({INTENSITY_LABELS.get(atk['intensity'], atk['intensity'])})")
    ax1.axvline(t0, color=col_color, lw=1.5, linestyle=":", alpha=0.85)
    ax1.axvline(t1, color=col_color, lw=1.0, linestyle=":", alpha=0.55)

    thr_styles = [(_decision_thr, "#999999", "-")]
    gt_mask = (sub["timestamp"] >= t0) & (sub["timestamp"] < t1)
    gt_sub  = sub.loc[gt_mask]

    for thr_v, tcol, ls in thr_styles:
        ax1.axhline(thr_v, color=tcol, lw=1.5, linestyle=ls, alpha=0.85,
                    label=f"Operational Threshold {thr_v:.2f}")
        det_mask = gt_sub[COL_DET].fillna(0) >= thr_v
        if det_mask.any():
            t_first = gt_sub.loc[det_mask, "timestamp"].iloc[0]
            ttd_m   = (t_first - t0).total_seconds() / 60.0
            ax1.annotate(f"TTD = {ttd_m:.0f} min",
                         xy=(t_first, thr_v + 0.027), fontsize=8,
                         color=tcol, fontstyle="italic",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, lw=0))

    ax1.set_ylim(-0.03, 1.08)
    ax1.set_ylabel("Belief Mass / Projected Probability", fontsize=11)
    ax1.set_title(
        f"Attack Detection — {atk['name'].replace('_', ' ')}\n"
        f"Type: {atk['type']}   Intensity: {INTENSITY_LABELS.get(atk['intensity'], atk['intensity'])}"
        f"   Duration: {atk['duration_h']} h   Start: {str(t0)[:19]}",
        fontsize=11, fontweight="bold", pad=8
    )
    ax1.legend(loc="upper right", fontsize=8.5, ncol=4, framealpha=0.85,
               edgecolor="#cccccc")
    plt.setp(ax1.get_xticklabels(), visible=False)

    gt_series = pd.Series(0.0, index=sub.index)
    gt_series[gt_mask.values] = 1.0
    ax2.fill_between(dates, gt_series, step="post",
                     color=col_color, alpha=0.55, label="Attack period")
    ax2.set_ylim(-0.15, 1.35)
    ax2.set_ylabel("Ground\nTruth", fontsize=9, labelpad=8)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Normal", "Attack"], fontsize=8.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax2.tick_params(axis="x", labelsize=8.5)

    fname = os.path.join(output_dir, "graphs", f"attack_{atk['name']}.png")
    fig.savefig(fname)
    plt.close(fig)
    print(f"   figure: {fname}")


def plot_threshold_sweep(df_sweep: pd.DataFrame, output_dir: str):
    """Precision / Recall (3 variants) / F1 vs threshold + FPR bar."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df_sweep["threshold"], df_sweep["recall_binary"],
            "o-",  color="#2ca02c", lw=2.0, ms=7,
            label="Recall — binary (attack detected)")
    ax.plot(df_sweep["threshold"], df_sweep["recall_coverage"],
            "D--", color="#17becf", lw=1.8, ms=6,
            label="Recall — coverage-weighted")
    ax.plot(df_sweep["threshold"], df_sweep["recall_ttd"],
            "v:",  color="#9467bd", lw=1.6, ms=6,
            label="Recall — coverage x TTD-penalized")
    ax.plot(df_sweep["threshold"], df_sweep["precision_window"],
            "s--", color="#1f77b4", lw=2.0, ms=7,
            label="Precision (window-level)")
    ax.plot(df_sweep["threshold"], df_sweep["f1_binary"],
            "^-",  color="#d62728", lw=2.4, ms=8,
            label="F1 — binary", zorder=6)
    # PATCH TASK-21 (audit_tmp CRIT-03): explicit hybrid column for plot.
    _hyb_plot = ("f1_coverage_hybrid_episode_recall"
                 if "f1_coverage_hybrid_episode_recall" in df_sweep.columns
                 else "f1_coverage")
    ax.plot(df_sweep["threshold"], df_sweep[_hyb_plot],
            "^--", color="#8c564b", lw=2.0, ms=7,
            label="F1 — coverage-weighted (hybrid)", zorder=5)

    # PATCH TASK-21 (audit_tmp CRIT-03, 2026-04-26): explicit hybrid metric.
    _hyb_col = ("f1_coverage_hybrid_episode_recall"
                if "f1_coverage_hybrid_episode_recall" in df_sweep.columns
                else "f1_coverage")
    best_idx = df_sweep[_hyb_col].idxmax()
    best     = df_sweep.loc[best_idx]
    ax.annotate(
        f"Best F1-coverage (hybrid) = {best[_hyb_col]:.3f}\n@ threshold = {best['threshold']:.2f}",
        xy=(best["threshold"], best[_hyb_col]),
        xytext=(best["threshold"] + 0.045, best[_hyb_col] - 0.12),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
        fontsize=9, color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.88, lw=0.8)
    )

    ax2 = ax.twinx()
    ax2.bar(df_sweep["threshold"], df_sweep["fpr_pct"],
            width=0.022, alpha=0.22, color="gray", label="FPR (%)")
    ax2.set_ylabel("False Positive Rate (%)", fontsize=10, color="#555555")
    ax2.tick_params(axis="y", labelcolor="#555555", labelsize=9)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#888888")

    ax.set_xlabel(r"Detection threshold $P(\mathrm{Atk})$", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(min(THRESHOLDS) - 0.025, max(THRESHOLDS) + 0.025)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_title(
        "Detection Performance vs. Decision Threshold\n"
        "Subjective Logic IDS — Precision, Recall (3 variants), F1, FPR",
        fontsize=11, fontweight="bold"
    )
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc="lower left", fontsize=8.5, framealpha=0.88)

    fname = os.path.join(output_dir, "graphs", "threshold_sweep.png")
    fig.savefig(fname)
    plt.close(fig)
    print(f"   figure: {fname}")


def plot_summary_table(df_eval: pd.DataFrame, df_sweep: pd.DataFrame,
                       output_dir: str):
    """
    Article-ready summary table (PNG, 300 dpi).
    Rows = attacks, columns = key metrics. Color-coded by detection status.
    """
    best_sweep = _select_best_row(df_sweep)
    best_thr = float(best_sweep["threshold"])
    sub = df_eval[df_eval["threshold"] == best_thr].sort_values("name")

    def _compact_label(value: str) -> str:
        labels = {
            "UNKNOWN_ANOMALY_CONTROL": "UNKNOWN",
            "UDP_FLOOD_DDOS": "UDP_FLOOD",
            "SYN_FLOOD_DDOS": "SYN_FLOOD",
            "BOTNET_CC_BEACONING": "BOTNET_CC",
            "AGGRESSIVE_PORT_SCAN": "PORT_SCAN",
            "DATA_EXFILTRATION_SLOW": "DATA_EXFIL",
            "HTTP_FLOOD_L7_DDOS": "HTTP_L7",
            "DNS_AMPLIFICATION": "DNS_AMP",
            "SLOWLORIS_DOS": "SLOWLORIS",
            "ICMP_FLOOD_BURST": "ICMP_BURST",
            "NTP_AMPLIFICATION": "NTP_AMP",
            "BRUTE_FORCE_SSH": "SSH_BRUTE",
            "DNS_TUNNELING": "DNS_TUNNEL",
            "REAL_DDOS": "REAL_DDOS",
        }
        return labels.get(str(value), str(value).replace("_", " "))

    col_headers = ["Attack", "Family", "Intensity", "Det.",
                   "Coverage", "TTD", "Max P(atk)"]
    rows_data = []
    for _, r in sub.iterrows():
        ttd_str = f"{r['ttd_minutes']:.0f} min" if not np.isnan(r["ttd_minutes"]) else "--"
        rows_data.append([
            _compact_label(r["name"]),
            _compact_label(r.get("family", r["name"])),
            INTENSITY_LABELS.get(r["intensity"], r["intensity"]),
            "Yes" if r["detected"] else "No",
            f"{r['coverage_pct']:.1f}%",
            ttd_str,
            f"{r['max_b_atk']:.3f}",
        ])

    global_row = [
        "GLOBAL",
        "--", "--",
        f"{int(best_sweep['n_detected_attacks'])}/{int(best_sweep['n_attacks'])}",
        f"F1(win)={best_sweep['f1_micro_pure']:.3f}",
        f"FPR = {best_sweep['fpr_pct']:.2f}%",
        f"MCC={best_sweep['mcc']:.3f}",
    ]

    fig_h = max(4.5, 0.34 * (len(rows_data) + 2) + 0.95)
    fig, ax = plt.subplots(figsize=(11.6, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=rows_data + [global_row],
        colLabels=col_headers,
        cellLoc="center",
        loc="upper center",
        bbox=[0.02, 0.02, 0.96, 0.88],
        colWidths=[0.20, 0.17, 0.12, 0.08, 0.12, 0.11, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1.0, 1.20)

    for j in range(len(col_headers)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(rows_data):
        det = row[3]
        for j in range(len(col_headers)):
            cell = table[i + 1, j]
            if det == "Yes":
                cell.set_facecolor("#eaf4ea" if i % 2 == 0 else "#d4edd4")
            else:
                cell.set_facecolor("#fde8e8" if i % 2 == 0 else "#fbd4d4")
            if j == 3:
                cell.set_text_props(
                    color="#1a7a1a" if det == "Yes" else "#cc0000",
                    fontweight="bold"
                )

    for j in range(len(col_headers)):
        table[len(rows_data) + 1, j].set_facecolor("#dce6f1")
        table[len(rows_data) + 1, j].set_text_props(fontweight="bold")

    um_status = "enabled" if UNCERTAINTY_MAXIMIZATION else "disabled"
    ax.set_title(
        f"SL-ADS detection summary | threshold={best_thr:.3f} | "
        f"lambda={LAMBDA_DECAY} | UM {um_status}",
        fontsize=10.0, fontweight="bold", pad=8
    )

    fname = os.path.join(output_dir, "graphs", "summary_table.png")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02)
    fig.savefig(fname)
    plt.close(fig)
    print(f"   figure: {fname}")


def plot_baserate_audit(df: pd.DataFrame, atk: dict, metric_ck: str,
                        output_dir: str):
    """Adaptive base rate evolution around one attack (English labels)."""
    col_a = f"{metric_ck}_a_atk"
    if col_a not in df.columns:
        return

    t0  = pd.Timestamp(atk["start"])
    t1  = t0 + pd.Timedelta(hours=atk["duration_h"])
    t_s = t0 - pd.Timedelta(hours=CONTEXT_H)
    t_e = t1 + pd.Timedelta(hours=CONTEXT_H)

    sub = df.loc[(df["timestamp"] >= t_s) & (df["timestamp"] <= t_e)].copy()
    if sub.empty or sub[col_a].isna().all():
        return

    fig, ax = plt.subplots(figsize=(13, 4.5))
    dates = sub["timestamp"]

    ax.step(dates, sub[col_a].fillna(np.nan), where="post",
            color="#d62728", lw=2.0, label=r"$a_{atk}$ adaptive base rate")
    col_proj = f"{metric_ck}_proj_atk"
    if col_proj in sub.columns:
        ax.step(dates, sub[col_proj].fillna(0), where="post",
                color="#8b0000", lw=1.5, linestyle="--", alpha=0.8,
                label=r"$P(atk)$ projected")
    col_b = f"{metric_ck}_b_atk"
    if col_b in sub.columns:
        ax.step(dates, sub[col_b].fillna(0), where="post",
                color="#d62728", lw=1.0, alpha=0.30, linestyle=":",
                label=r"$b_{atk}$ belief mass")

    col_color = INTENSITY_COLORS.get(atk["intensity"], "gray")
    ax.axvspan(t0, t1, alpha=0.10, color=col_color, label="Attack window")
    ax.axvline(t0, color=col_color, lw=1.3, linestyle=":", alpha=0.75)
    ax.axvline(t1, color=col_color, lw=1.0, linestyle=":", alpha=0.50)

    ax.set_title(
        f"Adaptive Base Rate — {atk['name'].replace('_', ' ')} / {metric_ck}",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylabel(r"$a_{atk}$", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.tick_params(axis="x", labelsize=8.5)

    fname = os.path.join(output_dir, "graphs",
                         f"baserate_{atk['name']}_{metric_ck.replace('->','_to_')}.png")
    fig.savefig(fname)
    plt.close(fig)


def plot_learning_comparison(df_lr: pd.DataFrame, family: str, df: pd.DataFrame,
                              catalog: list, output_dir: str):
    """R1 vs R2+ normalized detection profiles (English, pub-quality)."""
    members = sorted([a for a in catalog if attack_family(a["name"]) == family],
                     key=lambda a: attack_occurrence(a["name"]))
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.cm.get_cmap("tab10", len(members))

    for idx_m, atk in enumerate(members):
        occ     = attack_occurrence(atk["name"])
        t0      = pd.Timestamp(atk["start"])
        t1      = t0 + pd.Timedelta(hours=atk["duration_h"])
        dur_sec = (t1 - t0).total_seconds()
        gt_mask = (df["timestamp"] >= t0) & (df["timestamp"] < t1)
        sub     = df.loc[gt_mask].copy()
        if sub.empty:
            continue
        t_norm = [(ts - t0).total_seconds() / dur_sec * 100.0
                  for ts in sub["timestamp"]]
        ax.step(t_norm, sub[COL_DET].fillna(0).values, where="post",
                lw=2.0, color=cmap(idx_m),
                label=f"Occurrence {occ} ({atk['name']})", zorder=5 - idx_m)

    for thr_v, tcol, ls in [(_decision_thr, "#aaa", "-")]:
        ax.axhline(thr_v, color=tcol, lw=1.5, linestyle=ls, alpha=0.7,
                   label=f"Operational Threshold {thr_v:.2f}")

    ax.set_xlabel("Normalized attack time (%)", fontsize=11)
    ax.set_ylabel(r"$b_{atk}$ (CBF system)", fontsize=11)
    ax.set_title(
        f"Repetition Learning — Family: {family.replace('_', ' ')}\n"
        "Detection profiles overlaid on normalized time axis",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.07)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)

    fname = os.path.join(output_dir, "graphs", f"learning_{family}.png")
    fig.savefig(fname)
    plt.close(fig)
    print(f"   figure: {fname}")


# ==============================================================================
# CONSOLE REPORT — English, clean table format
# ==============================================================================

def _select_best_row(df_sweep: pd.DataFrame) -> pd.Series:
    """
    PATCH TASK-34 (audit_codex CRIT-01, 2026-04-27)
    ────────────────────────────────────────────────────────────────────────
    Operational threshold row resolution — **NO TEST-SET TUNING**.

    The deployed decision threshold is calibrated *a priori* during training
    (EVT/FPR procedure in ``train_v10.py``) and persisted to a JSON sidecar
    next to the trained-models pickle (``*_threshold.json``).  This function
    therefore selects the row of ``df_sweep`` whose ``threshold`` column
    matches the sidecar value (closest match within ``1e-9``), ensuring that
    every metric reported in the publication is computed at the same
    operating point that would be used in production.

    The previous implementation took ``df_sweep[metric_col].idxmax()`` which
    is mathematically equivalent to picking the threshold by argmax over the
    held-out evaluation set — a textbook case of test-set leakage as
    documented by Varma & Simon (2006, BMC Bioinformatics) and Arp et al.
    (2022, USENIX Security §4.2 "Sampling Bias / Tuning on Test").  Even
    when ``THRESHOLDS`` has length 1 (current default), the argmax pattern
    is dormant and would silently re-introduce leakage if future runs
    extended the sweep — hence the defensive rewrite.

    Escape hatch (research / exploration only):
        Setting ``SL_ALLOW_TEST_TUNED_THRESHOLD=1`` reverts to the legacy
        argmax behaviour and prints a ``[CRIT-01-OVERRIDE]`` warning.  This
        mode MUST NEVER be used to populate publication tables.

    Refs:
      • Varma S., Simon R. (2006) "Bias in error estimation when using
        cross-validation for model selection." BMC Bioinformatics 7:91.
      • Arp D., Quiring E., Pendlebury F. et al. (2022) "Dos and Don'ts
        of Machine Learning in Computer Security." USENIX Security.
      • Stone M. (1974) "Cross-validatory choice and assessment of
        statistical predictions." J.R.S.S. B 36(2).
    """
    metric_col = (
        "f1_coverage_hybrid_episode_recall"
        if "f1_coverage_hybrid_episode_recall" in df_sweep.columns
        else "f1_coverage"
    )

    # ── Escape hatch (legacy / research only) ────────────────────────────
    if os.environ.get("SL_ALLOW_TEST_TUNED_THRESHOLD") == "1":
        import warnings as _w
        _w.warn(
            "[CRIT-01-OVERRIDE] SL_ALLOW_TEST_TUNED_THRESHOLD=1 — selecting "
            "threshold by argmax(metric) on the evaluation sweep. This is "
            "test-set tuning and MUST NOT be used for publication tables.",
            category=UserWarning, stacklevel=2,
        )
        return df_sweep.loc[df_sweep[metric_col].idxmax()]

    # ── Default path: threshold from sidecar (a-priori-calibrated) ───────
    if "threshold" not in df_sweep.columns:
        raise RuntimeError(
            "[CRIT-01] _select_best_row: df_sweep is missing a 'threshold' "
            "column — cannot enforce a-priori threshold selection."
        )

    # _decision_thr is loaded at import-time (line ~100) from the sidecar.
    # We rematch by closest threshold to support legacy float roundoff in
    # the CSV (e.g. 0.2 vs 0.19999999...).
    _diff = (df_sweep["threshold"].astype(float) - float(_decision_thr)).abs()
    _imin = _diff.idxmin()
    if float(_diff.loc[_imin]) > 1e-6:
        raise RuntimeError(
            f"[CRIT-01] _select_best_row: sidecar threshold "
            f"{float(_decision_thr):.6f} not present in df_sweep "
            f"(closest available = {float(df_sweep['threshold'].iloc[_imin]):.6f}, "
            f"|diff|={float(_diff.loc[_imin]):.2e}). "
            "Re-run the sweep at the calibrated threshold or restore the sidecar."
        )
    return df_sweep.loc[_imin]


def print_summary_report(df_eval: pd.DataFrame, df_sweep: pd.DataFrame,
                          df_lr: pd.DataFrame, catalog: list,
                          df_protocol: pd.DataFrame = None):
    best_row = _select_best_row(df_sweep)
    thr_best = best_row["threshold"]

    print("\n" + "=" * 84)
    print("  EVALUATION REPORT - Subjective Logic IDS (Strict Threshold)")
    print("=" * 84)
    um_status = "ENABLED" if UNCERTAINTY_MAXIMIZATION else "DISABLED"
    print(f"\n  Configuration: lambda={LAMBDA_DECAY}  |  UM={um_status}")
    print(f"  Operational threshold : b_atk >= {thr_best:.2f}")
    print("  Threshold selected by : calibrated sidecar "
          f"(fusion={CONFIG.get('INTER_METHOD_FUSION', 'wbf')})")
    print(f"\n  {'Metric':<38} {'Value':>8}   Description")
    print(f"  {'-' * 78}")
    # ── Canonical pure window-level metrics (catalog/outages-separate) ───
    print(f"  [CANONICAL — pure window-level, catalog/outages-separate]")
    print(f"  {'F1 - micro (pure window)':<38} {best_row['f1_micro_pure']:>8.3f}   TP/FP/TN/FN binary, window-level")
    print(f"  {'F1 - macro (pure window)':<38} {best_row['f1_macro_pure']:>8.3f}   Macro-averaged, two classes")
    print(f"  {'Precision (window)':<38} {best_row['precision_window']:>8.3f}   TP_wins / (TP_wins + FP_wins)")
    print(f"  {'TPR (window-level)':<38} {best_row['tpr_window']:>8.3f}   ROC op. point — true positive rate")
    print(f"  {'FPR (window-level)':<38} {best_row['fpr_window']:>8.3f}   ROC op. point — false positive rate")
    print(f"  {'Accuracy':<38} {best_row['accuracy']:>8.3f}   (TP+TN)/(TP+FP+TN+FN)")
    print(f"  {'MCC':<38} {best_row['mcc']:>8.3f}   Matthews Corr. Coeff. (imbalance-robust)")
    if df_protocol is not None and not df_protocol.empty:
        print(f"\n  [F1 PROTOCOL COMPARISON — report both in paper]")
        for _, prow in df_protocol.iterrows():
            print(
                f"  {prow['protocol']:<38} {prow['f1_micro_pure']:>8.3f}   "
                f"FPR={prow['fpr_pct']:.3f}% | positives={int(prow['n_positive'])}"
            )
        print("  catalog_outages_separate = injected/catalog attacks; outages reported separately")
        print("  operator_faithful_anomaly = catalog + REAL_ATTACKS, outages are positives")
    # ── Hybrid metrics (precision_window × recall_episode) ────────────────
    print(f"\n  [HYBRID — precision_window × recall_episode, IDS-literature only]")
    print(f"  {'F1 - binary (hybrid)':<38} {best_row['f1_binary_hybrid_episode_recall']:>8.3f}   Sharafaldin 2018 / Mirsky 2018 style")
    print(f"  {'F1 - coverage-weighted (hybrid)':<38} {best_row['f1_coverage_hybrid_episode_recall']:>8.3f}   Penalizes partial detection")
    print(f"  {'F1 - TTD-penalized (hybrid)':<38} {best_row['f1_ttd_hybrid_episode_recall']:>8.3f}   Penalizes late + partial detection")
    print(f"  {'Recall - binary (episode)':<38} {best_row['recall_binary']:>8.3f}   Detected attacks / total attacks")
    print(f"  {'Recall - coverage (episode)':<38} {best_row['recall_coverage']:>8.3f}   Mean coverage across attacks")
    # ── Operational ──────────────────────────────────────────────────────
    print(f"\n  [OPERATIONAL]")
    print(f"  {'FPR (% of normal windows)':<38} {best_row['fpr_pct']:>7.2f}%   Empirical false positive rate")
    print(f"  {'Attacks detected':<38} {int(best_row['n_detected_attacks']):>5}/{int(best_row['n_attacks'])}   (>=1 detection within episode)")
    if "fpr_target" in best_row and "fpr_ratio_to_target" in best_row:
        print(f"  {'FPR target / ratio':<38} {best_row['fpr_target']:>8.4f}   "
              f"ratio={best_row['fpr_ratio_to_target']:.2f}x  "
              f"{best_row.get('fpr_target_status', '')}")
    print(f"\n  Note: hybrid F1 mixes two units of analysis (precision @ window,")
    print(f"        recall @ episode); see CRIT-03 disclosure in §5.3 of paper.")

    print(f"\n  {'-' * 84}")
    print(f"  {'Attack':<32} {'Status':<11} {'Cov%':>6}  {'TTD':>8}  {'Max b_atk':>9}  Intensity")
    print(f"  {'-' * 84}")

    for _, r in df_eval[df_eval["threshold"] == thr_best].sort_values("name").iterrows():
        status  = "DETECTED" if r["detected"] else "MISSED  "
        ttd_str = f"{r['ttd_minutes']:.0f} min" if not np.isnan(r["ttd_minutes"]) else "-"
        marker  = "[+]" if r["detected"] else "[-]"
        print(f"  {r['name']:<32} {marker} {status:<8} "
              f"{r['coverage_pct']:>5.1f}%  {ttd_str:>8}  "
              f"{r['max_b_atk']:>9.3f}  {r['intensity']}")

    if not df_lr.empty and "delta_ttd_min" in df_lr.columns:
        print(f"\n  {'-' * 84}")
        print(f"  REPETITION LEARNING @ threshold = {thr_best:.2f}")
        for _, r in df_lr[~df_lr["delta_ttd_min"].isna()].iterrows():
            print(f"  {r['family']:<32} occ={r['occurrence']}  "
                  f"dTTD={r['delta_ttd_min']:+.0f}min  "
                  f"dCov={r['delta_coverage']:+.1f}%  "
                  f"{r.get('comment_ttd', '')}")

    print("\n" + "=" * 84 + "\n")


def _attack_end_timestamp(atk: dict) -> pd.Timestamp:
    """Return the exclusive end timestamp for a catalog attack entry."""
    if "end" in atk and pd.notna(atk["end"]):
        return pd.Timestamp(atk["end"])
    return pd.Timestamp(atk["start"]) + pd.Timedelta(hours=float(atk["duration_h"]))


def compute_vus_table(df: pd.DataFrame, catalog: list,
                      df_sweep: pd.DataFrame) -> pd.DataFrame:
    """Compute range-aware VUS metrics for the timestamp attack catalog.

    VUS-ROC/VUS-PR are threshold-free.  The binary prediction is only used
    for existence-based recall at the selected operational threshold.
    """
    best_row = _select_best_row(df_sweep)
    threshold = float(best_row["threshold"])

    y_true = np.zeros(len(df), dtype=np.int8)
    ts = df["timestamp"]
    n_catalog_with_overlap = 0
    for atk in catalog:
        t0 = pd.Timestamp(atk["start"])
        t1 = _attack_end_timestamp(atk)
        mask = ((ts >= t0) & (ts < t1)).to_numpy()
        if mask.any():
            n_catalog_with_overlap += 1
            y_true[mask] = 1

    y_score = df[COL_DET].astype(float).to_numpy()
    y_pred = (y_score >= threshold).astype(np.int8)
    out = vus_summary(y_true, y_score, y_pred=y_pred)
    out.update({
        "threshold": threshold,
        "score_column": COL_DET,
        "label_scope": "timestamp_catalog",
        "n_catalog_attacks_with_overlap": int(n_catalog_with_overlap),
    })
    return pd.DataFrame([out])


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(f"\n{'=' * 62}")
    print("  evaluate_injection.py — SL-IDS Strict Evaluation")
    print(f"{'=' * 62}\n")

    if not os.path.exists(RESULTS_CSV):
        print(f"ERROR: File not found: {RESULTS_CSV}")
        sys.exit(1)
    if not ATTACK_CATALOG:
        print("ERROR: ATTACK_CATALOG empty. Check CATALOG_MODE in config.py.")
        sys.exit(1)

    os.makedirs(os.path.join(OUTPUT_DIR, "graphs"), exist_ok=True)

    print(f"-> Loading: {RESULTS_CSV}")
    df = pd.read_csv(RESULTS_CSV, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"   {len(df)} windows | {df['timestamp'].min()} -> {df['timestamp'].max()}")

    if COL_DET not in df.columns:
        print(f"ERROR: Column '{COL_DET}' not found.")
        sys.exit(1)

    csv_start, csv_end = df["timestamp"].min(), df["timestamp"].max()
    valid_catalog = []
    for atk in ATTACK_CATALOG:
        t0 = pd.Timestamp(atk["start"])
        t1 = t0 + pd.Timedelta(hours=atk["duration_h"])
        if t0 >= csv_start and t1 <= csv_end:
            valid_catalog.append(atk)
        else:
            print(f"   WARNING: [{atk['name']}] outside CSV range — skipped.")
    print(f"   {len(valid_catalog)}/{len(ATTACK_CATALOG)} attacks within range.")

    # ── P2: Coherence check between injection_label (if present) and timestamp GT ──
    if "injection_label" in df.columns:
        gt_from_label = (df["injection_label"] != "normal")
        gt_from_ts    = pd.Series(False, index=df.index)
        for _atk in valid_catalog:
            _t0 = pd.Timestamp(_atk["start"])
            _t1 = _t0 + pd.Timedelta(hours=_atk["duration_h"])
            gt_from_ts |= ((df["timestamp"] >= _t0) & (df["timestamp"] < _t1))
        _mismatch = (gt_from_label != gt_from_ts).sum()
        if _mismatch > 0:
            print(f"   WARNING [P2]: {_mismatch} windows disagree between "
                  f"'injection_label' column and timestamp-based ground truth. "
                  f"Check pipeline alignment (timestamp drift or out-of-range attacks).")
        else:
            print("   [P2] injection_label ↔ timestamp GT: coherent ✓")
    else:
        print("   [P2] 'injection_label' not in detection CSV "
              "(compute_opinions_v3 does not forward it — expected).")

    # Axis 1+2
    print("\n-> [Axis 1+2] Detection metrics & TTD...")
    df_eval  = evaluate_detection(df, valid_catalog)
    df_sweep = global_threshold_sweep(df_eval, df, valid_catalog)
    best_row = _select_best_row(df_sweep)
    df_protocol = f1_protocol_comparison(
        df, valid_catalog, float(best_row["threshold"])
    )
    df_eval.to_csv(os.path.join(OUTPUT_DIR, "eval_detection_summary.csv"),  index=False)
    df_sweep.to_csv(os.path.join(OUTPUT_DIR, "eval_threshold_sweep.csv"),   index=False)
    df_protocol.to_csv(os.path.join(OUTPUT_DIR, "eval_f1_protocol_comparison.csv"), index=False)
    print("   eval_detection_summary.csv saved")
    print("   eval_threshold_sweep.csv saved")
    print("   eval_f1_protocol_comparison.csv saved")

    df_vus = compute_vus_table(df, valid_catalog, df_sweep)
    df_vus.to_csv(os.path.join(OUTPUT_DIR, "eval_vus_summary.csv"), index=False)
    vrow = df_vus.iloc[0]
    print("   eval_vus_summary.csv saved "
          f"(VUS-PR={vrow['vus_pr']:.3f}, VUS-ROC={vrow['vus_roc']:.3f})")

    # Axis 3
    print("\n-> [Axis 3] Adaptive base rate audit...")
    df_br = audit_base_rates(df, valid_catalog)
    if not df_br.empty:
        df_br.to_csv(os.path.join(OUTPUT_DIR, "eval_baserate_audit.csv"), index=False)
        print("   eval_baserate_audit.csv saved")
    else:
        print("   WARNING: No a_atk columns found.")

    # Axis 4
    print("\n-> [Axis 4] Repetition learning analysis...")
    df_lr = learning_comparison(df_eval, df, valid_catalog)
    if not df_lr.empty:
        df_lr.to_csv(os.path.join(OUTPUT_DIR, "eval_learning_comparison.csv"), index=False)
        print("   eval_learning_comparison.csv saved")
        for fam in df_lr[df_lr["occurrence"] > 1]["family"].unique():
            plot_learning_comparison(df_lr, fam, df, valid_catalog, OUTPUT_DIR)
    else:
        print(
            "   [Axis 4] No attack family has 2+ occurrences (no _R2 suffix found).\n"
            "   This axis measures adaptive base-rate learning: the SL system should\n"
            "   detect R2 occurrences faster (lower TTD) and with higher coverage than R1\n"
            "   because the adaptive Dirichlet prior has already shifted toward 'attack'\n"
            "   after the first occurrence (Jøsang 2016, §12 — EDP update).\n"
            "   To activate: duplicate an attack entry with a '_R2' suffix, e.g.:\n"
            "     {'name': 'UDP_FLOOD_DDOS_R2', ...same signature, different start...}"
        )

    # Figures
    if os.environ.get("SL_SKIP_EVAL_PLOTS", "").strip().lower() in ("1", "true", "yes"):
        print("\n-> Evaluation plots skipped via SL_SKIP_EVAL_PLOTS=1")
    else:
        print("\n-> Per-attack timeline figures...")
        for atk in valid_catalog:
            plot_attack_timeline(df, atk, OUTPUT_DIR)

        print("\n-> Threshold sweep figure...")
        # MODIFICATION: Bypasser le sweep s'il n'y a qu'un seul seuil
        if len(THRESHOLDS) > 1:
            plot_threshold_sweep(df_sweep, OUTPUT_DIR)
        else:
            print("   INFO: Threshold sweep plot bypassed (strict operational threshold).")

        print("\n-> Publication summary table...")
        plot_summary_table(df_eval, df_sweep, OUTPUT_DIR)

        print("\n-> Base rate audit figures...")
        top_metrics = [ck for ck in LEAF_METRICS_TO_AUDIT if f"{ck}_a_atk" in df.columns][:3]
        for atk in valid_catalog:
            for ck in top_metrics:
                plot_baserate_audit(df, atk, ck, OUTPUT_DIR)

    print_summary_report(df_eval, df_sweep,
                          df_lr if not df_lr.empty else pd.DataFrame(),
                          valid_catalog, df_protocol)
    print(f"Evaluation complete. Results in: {OUTPUT_DIR}\n")

    # ------------------------------------------------------------------
    # PATCH m-06 / F23 : auto-populated run manifest
    # ------------------------------------------------------------------
    # Append a chronological row to MANIFEST.md at the project root.
    # Failures are swallowed by utils_manifest so the pipeline is never
    # affected — but we still report success/failure to stdout.
    try:
        from sl_ads.utils_manifest import append_manifest_entry  # Phase H

        # PATCH TASK-21 (audit_tmp CRIT-03, 2026-04-26): explicit hybrid metric.
        _hyb_col = ("f1_coverage_hybrid_episode_recall"
                    if "f1_coverage_hybrid_episode_recall" in df_sweep.columns
                    else "f1_coverage")
        best_row = df_sweep.loc[df_sweep[_hyb_col].idxmax()]
        # Median TTD across *detected* attacks only (robust to missed
        # attacks which would otherwise pollute with NaNs).
        _ttd_series = (
            df_eval.loc[df_eval["threshold"] == best_row["threshold"],
                        "ttd_minutes"]
            .dropna()
        )
        _median_ttd = float(_ttd_series.median()) if not _ttd_series.empty else None
        _protocol_metrics = {}
        for _, _prow in df_protocol.iterrows():
            if _prow["protocol"] == "catalog_outages_separate":
                _prefix = "catalog"
            elif _prow["protocol"] == "operator_faithful_anomaly":
                _prefix = "anomaly"
            else:
                continue
            _protocol_metrics.update({
                f"f1_{_prefix}_micro_pure": float(_prow["f1_micro_pure"]),
                f"f1_{_prefix}_macro_pure": float(_prow["f1_macro_pure"]),
                f"fpr_{_prefix}_window": float(_prow["fpr_window"]),
                f"fpr_{_prefix}_pct": float(_prow["fpr_pct"]),
                f"tpr_{_prefix}_window": float(_prow["tpr_window"]),
                f"n_positive_{_prefix}_windows": int(_prow["n_positive"]),
            })

        # PATCH TASK-21 (audit_tmp CRIT-03, 2026-04-26):
        #   * `f1_*_pure`                       → canonical, citable in paper
        #   * `f1_*_hybrid_episode_recall`      → hybrid IDS-literature metric
        #   * legacy aliases (`f1_binary`, `f1_coverage`, `f1_ttd`) kept for
        #     backward compatibility but their values are identical to the
        #     hybrid columns.  Will be removed in v11.
        metrics_payload = {
            "threshold":                          float(best_row["threshold"]),
            # — canonical pure window-level metrics
            "f1_micro_pure":                      float(best_row.get("f1_micro_pure", float("nan"))),
            "f1_macro_pure":                      float(best_row.get("f1_macro_pure", float("nan"))),
            "accuracy":                           float(best_row.get("accuracy", float("nan"))),
            "mcc":                                float(best_row.get("mcc", float("nan"))) if "mcc" in best_row else None,
            "tpr_window":                         float(best_row.get("tpr_window", float("nan"))) if "tpr_window" in best_row else None,
            "fpr_window":                         float(best_row.get("fpr_window", float("nan"))) if "fpr_window" in best_row else None,
            # — range-aware TSAD metrics (primary for publication tables)
            "vus_pr":                             float(vrow.get("vus_pr", float("nan"))),
            "vus_roc":                            float(vrow.get("vus_roc", float("nan"))),
            "range_auc_pr_at_max":                float(vrow.get("range_auc_pr_at_max", float("nan"))),
            "range_auc_roc_at_max":               float(vrow.get("range_auc_roc_at_max", float("nan"))),
            "existence_recall":                   float(vrow.get("existence_recall", float("nan"))),
            # — hybrid (precision_window × recall_episode)
            "f1_binary_hybrid_episode_recall":    float(best_row.get("f1_binary_hybrid_episode_recall", float("nan"))),
            "f1_coverage_hybrid_episode_recall":  float(best_row.get("f1_coverage_hybrid_episode_recall", float("nan"))),
            "f1_ttd_hybrid_episode_recall":       float(best_row.get("f1_ttd_hybrid_episode_recall", float("nan"))),
            # — operational
            "precision_window":                   float(best_row.get("precision_window", float("nan"))),
            "recall_binary":                      float(best_row.get("recall_binary", float("nan"))),
            "recall_coverage":                    float(best_row.get("recall_coverage", float("nan"))),
            "fpr_target":                         float(best_row.get("fpr_target", float("nan"))) if "fpr_target" in best_row else None,
            "fpr_ratio_to_target":                float(best_row.get("fpr_ratio_to_target", float("nan"))) if "fpr_ratio_to_target" in best_row else None,
            "fpr_target_status":                  str(best_row.get("fpr_target_status", "")) if "fpr_target_status" in best_row else None,
            "fpr_pct":                            float(best_row.get("fpr_pct", float("nan"))),
            "median_ttd_min":                     _median_ttd,
            "n_detected_attacks":                 int(best_row.get("n_detected_attacks", 0)),
            "n_attacks":                          int(best_row.get("n_attacks", 0)),
            "f1_protocol_policy":                 "report_both",
            **_protocol_metrics,
            # — DEPRECATED aliases (back-compat, identical to hybrid columns)
            "f1_binary":                          float(best_row.get("f1_binary", float("nan"))),
            "f1_coverage":                        float(best_row.get("f1_coverage", float("nan"))),
            "f1_ttd":                             float(best_row.get("f1_ttd", float("nan"))),
        }
        extras_payload = {
            "catalog_mode":   CATALOG_MODE,
            "lambda_decay":   LAMBDA_DECAY,
            "um_enabled":     UNCERTAINTY_MAXIMIZATION,
            "window_min":     WINDOW_MIN,
            "context_h":      CONTEXT_H,
            "col_det":        COL_DET,
            "output_dir":     OUTPUT_DIR,
        }

        # PATCH m-08 / F28 : surface training-time fallback audit in the
        # manifest so the reviewer can see at a glance how many metrics
        # hit EVT/reconstruction fallbacks during the run that produced
        # this CSV.  Read from the threshold sidecar (already lightweight).
        try:
            import json as _json
            from sl_ads.paths import get_threshold_sidecar_path  # Phase H
            _sc = get_threshold_sidecar_path(CONFIG, up_levels=1)
            if os.path.exists(_sc):
                with open(_sc, encoding="utf-8") as _scf:
                    _sc_data = _json.load(_scf)
                _fb = _sc_data.get("fallbacks") or {}
                if _fb:
                    extras_payload["fallbacks_total"]   = _fb.get("total", 0)
                    extras_payload["fallbacks_counts"]  = _fb.get("counts", {})
                    extras_payload["fallbacks_metrics"] = _fb.get("metrics", {})
        except Exception:
            pass  # manifest must never fail on sidecar read
        # PATCH TASK-32 (audit_tmp MAJ-11, 2026-04-26): include CONFIG +
        # input file fingerprints to enable a deterministic run_id that
        # is reproducible across machines.
        _input_paths_for_id = [RESULTS_CSV]
        try:
            from sl_ads.paths import get_threshold_sidecar_path as _gtsp  # Phase H
            _sc_p = _gtsp(CONFIG, up_levels=1)
            if os.path.exists(_sc_p):
                _input_paths_for_id.append(_sc_p)
        except Exception:
            pass
        manifest_path = append_manifest_entry(
            metrics=metrics_payload,
            version_name=VERSION_NAME,
            csv_path=RESULTS_CSV,
            project_root=os.path.dirname(os.path.abspath(__file__)),
            extras=extras_payload,
            config=CONFIG,
            input_paths=_input_paths_for_id,
        )
        if manifest_path is not None:
            print(f"-> MANIFEST.md updated: {manifest_path}")
    except Exception as _mf_exc:
        # utils_manifest is already defensive, but catch any upstream
        # surprise (e.g. df_sweep schema drift) to stay non-fatal.
        print(f"   WARNING: manifest append failed ({_mf_exc!r}); "
              "run outputs are unaffected.")


if __name__ == "__main__":
    main()
