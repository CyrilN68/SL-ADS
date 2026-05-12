"""
compare_if_fair.py — Comparaison méthodologiquement équitable SL-ADS vs Isolation Forest
==========================================================================================

Objectif
--------
Éviter 3 asymétries méthodologiques dans la comparaison :
1) Régime d'entraînement : IF entraîné sur split temporel identique à SL (pré-split).
2) Espace de features : IF entraîné sur métriques brutes réseau (pas sur evidences SL _N).
3) Point de fonctionnement : IF évalué au même FPR que SL (matching FPR).

Usage minimal
-------------
    python compare_if_fair.py

Le script lit `config.py` et tente de localiser automatiquement :
- CSV brut réseau (`CONFIG["file_path"]`)
- CSV résultats SL (`CONFIG["EVAL"]["RESULTS_CSV_NAME"]` dans `RESULTS_DIR`)

Sorties
-------
Dans `results/resultats_<VERSION_NAME>/evaluation_if_fair/` :
- fair_if_vs_sl_summary.csv
- fair_if_vs_sl_points.csv
- fair_if_vs_sl_report.md
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
import re

# PATCH-C6 (2026-04-21) : force stdout UTF-8 sur Windows (cp1252 ne peut pas
# encoder les emojis ℹ️ ❌ utilisés dans les messages).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    from sl_ads.config import CONFIG  # Phase H
except ImportError as exc:  # pragma: no cover
    raise SystemExit("❌ sl_ads.config introuvable.") from exc

try:
    from sl_ads.paths import get_decision_threshold as _get_decision_threshold  # Phase H
    from sl_ads.paths import get_detection_col as _get_detection_col  # Phase H
    _HAS_PATHS_COMPARE = True
except ImportError:
    _HAS_PATHS_COMPARE = False


@dataclass
class BinaryMetrics:
    threshold: float
    fpr: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _compute_binary_metrics(scores: np.ndarray, y_true: np.ndarray, thr: float) -> BinaryMetrics:
    y_pred = (scores >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return BinaryMetrics(
        threshold=float(thr),
        fpr=float(fpr),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def _build_label_mask(ts: pd.Series, attacks: list[dict]) -> np.ndarray:
    mask = np.zeros(len(ts), dtype=bool)
    for atk in attacks:
        t0, t1 = _attack_bounds(atk)
        mask |= ((ts >= t0) & (ts < t1)).values
    return mask


def _attack_bounds(atk: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Normalise les formats d'attaque supportés.

    Formats supportés:
    - {"start": "...", "duration_h": ...}
    - {"start": "...", "end": "..."}
    """
    if "start" not in atk:
        raise KeyError("Attack entry missing 'start'.")

    t0 = pd.Timestamp(atk["start"])

    if "duration_h" in atk and atk["duration_h"] is not None:
        t1 = t0 + pd.Timedelta(hours=float(atk["duration_h"]))
    elif "end" in atk and atk["end"] is not None:
        t1 = pd.Timestamp(atk["end"])
    else:
        raise KeyError("Attack entry must include either 'duration_h' or 'end'.")

    if t1 <= t0:
        raise ValueError(f"Invalid attack interval: end ({t1}) <= start ({t0}).")

    return t0, t1


def _nearest_timestamp_join(raw_df: pd.DataFrame, sl_df: pd.DataFrame, tolerance: str = "5min") -> pd.DataFrame:
    # Données triées indispensables pour merge_asof
    raw_sorted = raw_df.sort_values("timestamp").reset_index(drop=True)
    sl_sorted = sl_df.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        raw_sorted,
        sl_sorted[["timestamp", "sl_score"]],
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    merged = merged.dropna(subset=["sl_score"]).copy()

    # Fallback robuste: si aucun match avec merge_asof (pas de recouvrement au pas choisi),
    # tenter un merge exact sur timestamp pour diagnostiquer si les horodatages sont identiques.
    if merged.empty:
        exact = raw_sorted.merge(sl_sorted[["timestamp", "sl_score"]], on="timestamp", how="inner")
        if not exact.empty:
            return exact

    return merged


def _find_if_threshold_matching_fpr(if_scores: np.ndarray, y_true: np.ndarray, target_fpr: float) -> BinaryMetrics:
    """[DEPRECATED — C-01/F02] Picks the IF decision threshold by minimising
    ``|fpr - target_fpr|`` *on the same labels used for evaluation*.
    This leaks the test labels into model selection and is flagged CRITICAL
    by the consolidated audit (C-01/F02).

    Prefer :func:`_calibrate_if_threshold_from_normal` below, which computes
    the threshold from a held-out (pre-split, normal-only) calibration set
    and is therefore free of test-label leakage.

    Kept in the module for backward compatibility and for diagnostic
    purposes only; must NOT be used to produce numbers reported in the
    main comparison tables.
    """
    # Seuils candidats sur quantiles fins pour un matching robuste
    qs = np.linspace(0.0, 1.0, 2001)
    candidates = np.quantile(if_scores, qs)

    best = None
    best_gap = float("inf")

    for thr in np.unique(candidates):
        m = _compute_binary_metrics(if_scores, y_true, thr)
        gap = abs(m.fpr - target_fpr)
        if gap < best_gap:
            best = m
            best_gap = gap

    if best is None:  # pragma: no cover
        raise RuntimeError("Impossible de trouver un seuil IF pour le matching FPR.")
    return best


# ──────────────────────────────────────────────────────────────────────
# PATCH C-01 / F02 — Threshold calibration on pre-split (leak-free).
# ──────────────────────────────────────────────────────────────────────
def _calibrate_if_threshold_from_normal(
        calib_scores: np.ndarray,
        target_fpr: float,
) -> tuple[float, float]:
    """Return ``(threshold, empirical_fpr_on_calib)`` picked on a held-out
    *normal-only* calibration set.

    Because the calibration set contains only normal windows, the fraction
    of calibration scores exceeding any threshold ``thr`` is exactly an
    estimate of ``FPR(thr)``.  We therefore pick
    ``thr = np.quantile(calib_scores, 1 - target_fpr)`` so that the
    empirical FPR on calibration equals ``target_fpr`` up to ties.

    This avoids the methodological flaw of
    :func:`_find_if_threshold_matching_fpr` (C-01/F02), which used the test
    labels themselves to tune the threshold.

    Parameters
    ----------
    calib_scores : array-like of shape (n_calib,)
        IF scores (higher = more anomalous) over the pre-split
        *normal-only* calibration windows.  In this codebase they are
        produced by aggregating ``-if_model.decision_function(X_train_s)``
        on ``df_train[is_attack==0]`` with the same decision-window
        floor used on the test set, then reading the same aggregation
        (``if_score_max`` or ``if_score_mean``) as requested via
        ``--if-window-score-agg``.
    target_fpr : float
        Desired operating FPR.  Must be in (0, 1).

    Returns
    -------
    threshold : float
        The calibrated decision threshold.  A test-set score ``s`` is
        declared anomalous iff ``s >= threshold``.
    empirical_fpr_on_calib : float
        The empirical fraction of calibration scores ``>= threshold``.
        Reported for transparency; should be close to ``target_fpr``
        (exact match only modulo quantile ties).

    Notes
    -----
    * When ``calib_scores`` is sparse, the empirical FPR can deviate
      noticeably from ``target_fpr`` because of quantile granularity.
      The caller should log both numbers.
    * The Isolation Forest having seen these same rows during ``.fit()``
      introduces a (small) optimism bias on calibration-set scores.
      This is acceptable here because (a) the model was fit on
      normal-only data without supervision, (b) no *test* labels are
      consumed, and (c) anomaly-detection literature routinely
      calibrates thresholds on normal training data (Liu et al. 2008
      §5.2; Emmott et al. 2015 §4).
    """
    if not (0.0 < target_fpr < 1.0):
        raise ValueError(
            f"target_fpr must be in (0, 1); got {target_fpr!r}")
    calib_scores = np.asarray(calib_scores, dtype=float).ravel()
    if calib_scores.size == 0:
        raise ValueError("Empty calibration score array — "
                         "cannot calibrate IF threshold.")
    threshold = float(np.quantile(calib_scores, 1.0 - target_fpr))
    empirical_fpr = float(np.mean(calib_scores >= threshold))
    return threshold, empirical_fpr


def parse_args() -> argparse.Namespace:
    eval_cfg = CONFIG.get("EVAL", {})
    version_name = CONFIG.get("VERSION_NAME", "trained_models_v9_v6_v4s")
    default_results_dir = eval_cfg.get("RESULTS_DIR", f"../../results/resultats_{version_name}")
    default_sl_csv = os.path.join(default_results_dir, eval_cfg.get("RESULTS_CSV_NAME", "detection_results_INJECTED.csv"))
    # Calcul dynamique de la fenêtre en minutes (FREQ * WINDOW_SIZE)
    # Ex: freq "1min" * 10 = "10min"
    freq_str = CONFIG.get("freq_data", "1min")
    win_size = CONFIG.get("WINDOW_SIZE", 5)
    val = int(re.search(r'\d+', freq_str).group())
    unit = ''.join(re.findall(r'[a-zA-Z]+', freq_str))
    # Normalisation simple pour convertir en minutes
    if "s" in unit.lower():
        total_minutes = max(1, int((val * win_size) / 60))
    elif "min" in unit.lower() or "m" in unit.lower():
        total_minutes = val * win_size
    else:
        total_minutes = 5  # fallback
    dynamic_window = f"{total_minutes}min"

    parser = argparse.ArgumentParser(description="Comparaison équitable SL-ADS vs IF (split + raw features + FPR match).")
    parser.add_argument("--raw-csv", default=CONFIG.get("file_path"), help="CSV brut métriques réseau (timestamp + ACTIVE_METRICS).")
    parser.add_argument("--sl-csv", default=default_sl_csv, help="CSV sortie SL contenant le score attaque système.")
    parser.add_argument("--split-date", default=CONFIG.get("split_date"), help="Date de split train/test (incluse côté train).")
    _default_sl_thr = (_get_decision_threshold(CONFIG, up_levels=1)
                       if _HAS_PATHS_COMPARE else eval_cfg.get("DECISION_THRESHOLD", 0.16))
    parser.add_argument("--sl-threshold", type=float, default=_default_sl_thr, help="Seuil opérationnel SL pour point de comparaison.")
    parser.add_argument("--timestamp-col", default="timestamp", help="Nom de la colonne temporelle.")
    _default_score_col = (_get_detection_col(CONFIG, up_levels=1)
                          if _HAS_PATHS_COMPARE
                          else eval_cfg.get("COL_PROJ_ATK", eval_cfg.get("COL_BATK", "FINAL_SYSTEM_CBF_proj_atk")))
    parser.add_argument("--sl-score-col", default=_default_score_col,
                        help="Colonne score SL (probabilité/masse attaque).")
    parser.add_argument("--merge-tolerance", default="5min",
                        help="Tolérance max pour l'alignement temporel raw↔SL (ex: 30s, 2min, 5min).")
    parser.add_argument("--output-dir", default=f"../results/resultats_{version_name}/evaluation_if_fair", help="Dossier de sortie.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--label-mode", choices=["ddos_only", "eval_config"], default="eval_config",
                        help="Source des labels attaque: DDOS_ATTACK seul ou catalogue EVAL (injecté/réel).")
    parser.add_argument("--decision-window", default=dynamic_window,
                        help="Unité de décision fixe pour la comparaison stricte (ex: 5min).")
    parser.add_argument("--if-min-anom-slices", type=int, default=2,
                        help="Nb minimal de slices anormales par fenêtre fixe pour déclarer anomalie IF.")
    parser.add_argument("--if-contamination", default="sklearn_auto",
                        help=(
                            "Contamination IF — aucune valeur par défaut ne doit "
                            "être calibrée sur les labels du test (fuite). Valeurs supportées : "
                            "`sklearn_auto` (défaut, offset interne de sklearn — SANS accès aux labels) ; "
                            "`train` (proportion des positifs observée dans le split TRAIN uniquement, "
                            "requiert `label` dans le CSV brut) ; "
                            "flottant explicite (ex. 0.02) — doit être justifié par "
                            "domaine/littérature, pas par les labels test."
                        ))
    parser.add_argument("--target-fpr-pct", type=float, default=1.85,
                        help="Méthode B: FPR cible (%) pour IF. Si absent, utilise FPR mesuré de SL.")
    parser.add_argument("--if-window-score-agg", choices=["max", "mean"], default="max",
                        help="Agrégation des scores IF slice->fenêtre fixe pour decision_function.")
    return parser.parse_args()


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Rendu Markdown sans dépendance optionnelle `tabulate`."""
    headers = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.values.tolist()]

    # Largeur de colonnes
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt_row(vals: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    lines = [_fmt_row(headers), sep]
    lines.extend(_fmt_row(r) for r in rows)
    return "\n".join(lines)


def _aggregate_to_fixed_windows(
    df_slices: pd.DataFrame,
    window: str,
    if_binary_col: str,
    sl_score_col: str,
    label_col: str,
    min_if_anom_slices: int,
) -> pd.DataFrame:
    """
    Agrège les slices 30s en fenêtres fixes (sans tolérance), pour comparer
    SL et IF à la même unité décisionnelle.
    """
    d = df_slices.copy()
    d["window_start"] = d["timestamp"].dt.floor(window)

    agg = (
        d.groupby("window_start", as_index=False)
        .agg(
            n_slices=("timestamp", "count"),
            if_anom_slices=(if_binary_col, "sum"),
            sl_score=(sl_score_col, "max"),
            y_true=(label_col, "max"),
        )
    )
    agg["if_pred"] = (agg["if_anom_slices"] >= int(min_if_anom_slices)).astype(int)
    return agg


def _load_attack_catalog_from_eval() -> list[dict]:
    eval_cfg = CONFIG.get("EVAL", {})
    catalog_mode = eval_cfg.get("CATALOG_MODE", "injected")
    catalog: list[dict] = []

    if catalog_mode == "real":
        catalog = list(eval_cfg.get("REAL_ATTACK_CATALOG", []))
    else:
        try:
            # Phase H: permet l'import depuis le project root quand ce
            # script est lancé depuis investigations/ (anciennement
            # "modèle évaluation/").
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from sl_ads.inject.evidence_level import ATTACK_CATALOG as _INJECTED  # Phase H
            catalog = list(_INJECTED)
        except ImportError:
            catalog = []
        if eval_cfg.get("INCLUDE_REAL_ATTACK", True):
            catalog += list(eval_cfg.get("REAL_ATTACK_CATALOG", []))

    return catalog


def _resolve_sl_csv_path(requested_path: str) -> str:
    """
    Résout le chemin du CSV de détection SL.

    PATCH-C6 (2026-04-21) : correction du typo "dadza" → "RESULTS_CSV_NAME".
    Comme `compare_if_fair.py` tourne sur données injectées, on essaie d'abord
    la variante `_INJECTED` puis retombe sur la version non-injectée (miroir
    de la logique d'`evaluate_injection_v2.py` L62–L66).
    """
    if requested_path and os.path.exists(requested_path):
        return requested_path

    eval_cfg = CONFIG.get("EVAL", {})
    version_name = CONFIG.get("VERSION_NAME", "trained_models_v9_v6_v4s")

    # Nom de base depuis config (ex: "detection_results.csv")
    base_csv = eval_cfg.get("RESULTS_CSV_NAME", "detection_results.csv")
    # Variante INJECTED correspondante (ex: "detection_results_INJECTED.csv")
    injected_csv = base_csv.replace(".csv", "_INJECTED.csv")

    # Prefer INJECTED variant (ce script compare explicitement sur données injectées)
    csv_names = [injected_csv, base_csv]

    candidates = []
    for csv_name in csv_names:
        candidates.extend([
            os.path.join(eval_cfg.get("RESULTS_DIR", ""), csv_name),
            os.path.join(f"../../results/resultats_{version_name}", csv_name),
            os.path.join(f"../results/resultats_{version_name}", csv_name),
            os.path.join(f"../results/resultats_{version_name}_attacks", csv_name),
        ])

    for p in candidates:
        if p and os.path.exists(p):
            print(f"ℹ️ SL CSV auto-résolu: {p}")
            return p

    raise SystemExit(
        "❌ CSV SL introuvable. Essayé: "
        + ", ".join([p for p in candidates if p])
    )


def _build_label_mask_from_csv(raw_df: pd.DataFrame,
                               label_col: str = "label") -> "np.ndarray | None":
    """
    Construit y_true depuis une colonne de labels binaires dans le CSV brut.

    Utilisé pour les datasets avec vérité terrain réelle (METR-LA, GECCO-IoT, CESNET).
    Retourne None si la colonne est absente ou entièrement nulle (fallback vers catalog).

    Ref : Fawcett (2006) Pattern Recognit. Lett. §2 — y_true construction methods.
    """
    if label_col not in raw_df.columns:
        return None
    y = (raw_df[label_col].fillna(0) > 0).astype(int).values
    if y.sum() == 0:
        return None
    return y


def main() -> None:
    args = parse_args()

    if not args.raw_csv or not os.path.exists(args.raw_csv):
        raise SystemExit(f"❌ CSV brut introuvable: {args.raw_csv}")
    args.sl_csv = _resolve_sl_csv_path(args.sl_csv)

    split_date = pd.Timestamp(args.split_date)
    metrics = list(CONFIG.get("ACTIVE_METRICS", []))

    print("\n=== Chargement des données ===")
    raw_df = pd.read_csv(args.raw_csv, parse_dates=[args.timestamp_col])
    sl_df = pd.read_csv(args.sl_csv, parse_dates=[args.timestamp_col])

    if args.sl_score_col not in sl_df.columns:
        raise SystemExit(f"❌ Colonne SL introuvable: {args.sl_score_col}")

    missing = [c for c in metrics if c not in raw_df.columns]
    if missing:
        raise SystemExit(f"❌ Métriques manquantes dans raw CSV: {missing}")

    raw_df = raw_df.rename(columns={args.timestamp_col: "timestamp"})
    sl_df = sl_df.rename(columns={args.timestamp_col: "timestamp", args.sl_score_col: "sl_score"})

    # =========================
    # 1) RAW timeline (référence principale)
    # =========================
    raw_df = raw_df.sort_values("timestamp").reset_index(drop=True)

    # ── Construction de y_true (auto-detect: labels CSV > catalogue d'attaques) ──
    # Mode 1 : colonne 'label' dans le CSV brut (METR-LA, GECCO-IoT, CESNET).
    #          Utilisé automatiquement si la colonne est présente et non-nulle.
    # Mode 2 : catalogue d'attaques horodaté (RedeRio et datasets injectés).
    # Ref : Fawcett (2006) Pattern Recognit. Lett. §2 – y_true construction equivalence.
    csv_labels = _build_label_mask_from_csv(raw_df, label_col="label")
    if csv_labels is not None:
        raw_df["is_attack"] = csv_labels
        n_pos = int(raw_df["is_attack"].sum())
        print(f"  Labels: CSV (colonne 'label') — {n_pos}/{len(raw_df)} positifs "
              f"({100 * n_pos / len(raw_df):.2f}%)")
    else:
        if args.label_mode == "eval_config":
            attacks = _load_attack_catalog_from_eval()
        else:
            attacks = list(CONFIG.get("DDOS_ATTACK", []))
        if not attacks:
            raise SystemExit(
                "❌ Aucun label disponible pour construire y_true.\n"
                "   Dataset labelisé : vérifier colonne 'label' dans CONFIG['file_path'].\n"
                "   Dataset injecté  : vérifier ATTACK_CATALOG / DDOS_ATTACK dans config.py."
            )
        raw_df["is_attack"] = _build_label_mask(raw_df["timestamp"], attacks).astype(int)
        n_pos = int(raw_df["is_attack"].sum())
        print(f"  Labels: catalogue — {len(attacks)} attaques, {n_pos}/{len(raw_df)} positifs "
              f"({100 * n_pos / len(raw_df):.2f}%)")

    tmin = raw_df["timestamp"].min()
    tmax = raw_df["timestamp"].max()
    if split_date <= tmin or split_date >= tmax:
        split_date = tmin + (tmax - tmin) / 2
        print(
            "⚠️ split_date hors plage des données alignées. "
            f"Fallback auto au milieu de série: {split_date}"
        )

    train_mask = raw_df["timestamp"] <= split_date
    test_mask = raw_df["timestamp"] > split_date

    df_train = raw_df.loc[train_mask].copy()
    df_test = raw_df.loc[test_mask].copy()

    if df_train.empty or df_test.empty:
        raise SystemExit(
            "❌ Split invalide: train ou test vide. "
            f"Range alignée=[{tmin} -> {tmax}], split={split_date}, "
            f"train={len(df_train)}, test={len(df_test)}."
        )

    print(f"Train windows: {len(df_train)} | Test windows: {len(df_test)}")
    print(f"Période train: {df_train['timestamp'].min()} -> {df_train['timestamp'].max()}")
    print(f"Période test : {df_test['timestamp'].min()} -> {df_test['timestamp'].max()}")

    # Garde-fou qualité: si train trop petit, on ne publie pas de comparaison.
    if len(df_train) < 1000:
        raise SystemExit(
            "❌ Train trop court (<1000 fenêtres). Résultats non fiables; "
            "ne pas intégrer ces résultats au rapport."
        )

    # --- IF FAIR: entraînement strictement pré-split sur fenêtres normales ---
    train_normal = df_train[df_train["is_attack"] == 0]
    if train_normal.empty:
        raise SystemExit("❌ Aucune fenêtre normale en train pour entraîner IF.")

    X_train = train_normal[metrics].fillna(0.0).values
    X_test = df_test[metrics].fillna(0.0).values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Résolution de la contamination d'Isolation Forest (sans fuite test) ──
    # Politique publiable :
    #   - "sklearn_auto"  → sklearn passe contamination='auto' (offset_ interne,
    #                       AUCUN accès aux labels — pas de fuite possible).
    #   - "train"         → proportion de labels positifs dans le SPLIT TRAIN
    #                       (nécessite labels CSV fiables côté train ; si absents
    #                       ou nuls, bascule en sklearn_auto avec WARN explicite).
    #   - float explicite → valeur fixée (constante de littérature/domaine) ;
    #                       à justifier dans le rapport, pas issue du test.
    # Ref : Liu et al. (2008) "Isolation Forest" ICDM ;
    #       Emmott et al. (2015) "A Meta-Analysis of the Anomaly Detection Problem".
    contamination_arg = str(args.if_contamination).strip()
    contamination_source = "user_constant"
    if contamination_arg in ("sklearn_auto", "auto_sklearn", "auto"):
        # NB : on mappe explicitement l'ancien alias "auto" vers le comportement
        # sklearn natif. L'ancienne calibration "auto" basée sur les labels test
        # est retirée (fuite méthodologique).
        contamination = "auto"
        contamination_source = "sklearn_auto"
        print("  IF contamination = sklearn 'auto' (offset interne, sans labels) "
              "— politique sans fuite.")
    elif contamination_arg in ("train", "train_labels"):
        # Estimation depuis le split TRAIN exclusivement — jamais depuis le test.
        n_atk_train = int(df_train["is_attack"].sum())
        if n_atk_train > 0 and len(df_train) > 0:
            contamination = float(n_atk_train) / float(len(df_train))
            contamination_source = "train_labels"
            print(
                f"  IF contamination calibrée sur TRAIN uniquement : "
                f"{n_atk_train}/{len(df_train)} = {contamination:.4f}"
            )
        else:
            # Train sans positifs (hypothèse nominale "train propre") → fallback sans fuite.
            contamination = "auto"
            contamination_source = "sklearn_auto_fallback_no_train_labels"
            print(
                "  [WARN] Aucun label positif dans le TRAIN (ou colonne 'label' absente). "
                "Fallback sklearn 'auto' (sans fuite). "
                "Si une constante de littérature est requise, passer --if-contamination <float>."
            )
    else:
        try:
            contamination = float(contamination_arg)
        except ValueError as exc:
            raise SystemExit(
                f"❌ --if-contamination invalide: {args.if_contamination}. "
                "Attendu: 'sklearn_auto', 'train', ou un flottant ∈ (0, 0.5)."
            ) from exc
        if not (0.0 < contamination <= 0.5):
            raise SystemExit(
                f"❌ --if-contamination hors borne sklearn: {contamination}. "
                "Attendu: flottant ∈ (0, 0.5]."
            )
        print(f"  IF contamination = constante utilisateur : {contamination}")

    if_model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=args.random_state,
        n_jobs=-1,
    )
    if_model.fit(X_train_s)

    # 0/1 et score continu par slice 30s, puis agrégation stricte en fenêtres fixes.
    if_pred_slice = (if_model.predict(X_test_s) == -1).astype(int)
    if_score_slice = -if_model.decision_function(X_test_s)

    # ── PATCH C-01/F02 : scores IF sur le split PRE-SPLIT (normal-only) ──
    # Ces scores servent UNIQUEMENT à calibrer le seuil "fpr-matched" sans
    # fuite depuis les labels test (cf. _calibrate_if_threshold_from_normal).
    # Important : on n'utilise ici que ``train_normal`` (is_attack == 0).
    if_score_slice_calib = -if_model.decision_function(X_train_s)

    # =========================
    # 2) Alignement strict sur fenêtres fixes (sans tolérance)
    # =========================
    sl_test = sl_df.loc[sl_df["timestamp"] > split_date, ["timestamp", "sl_score"]].copy()
    if sl_test.empty:
        raise SystemExit("❌ Aucun score SL en période test.")

    slice_df = df_test[["timestamp", "is_attack"]].copy()
    slice_df["if_pred_slice"] = if_pred_slice
    slice_df["if_score_slice"] = if_score_slice

    # Associe score SL à chaque slice via fenêtre fixe identique.
    # Si plusieurs scores SL dans la même fenêtre, on garde le max.
    sl_test["window_start"] = sl_test["timestamp"].dt.floor(args.decision_window)
    sl_win = (
        sl_test.groupby("window_start", as_index=False)
        .agg(sl_score=("sl_score", "max"))
    )

    slice_df["window_start"] = slice_df["timestamp"].dt.floor(args.decision_window)
    test_windows = (
        slice_df.groupby("window_start", as_index=False)
        .agg(
            n_slices=("timestamp", "count"),
            if_anom_slices=("if_pred_slice", "sum"),
            if_score_max=("if_score_slice", "max"),
            if_score_mean=("if_score_slice", "mean"),
            y_true=("is_attack", "max"),
        )
    )

    common = test_windows.merge(sl_win, on="window_start", how="inner")
    if common.empty:
        raise SystemExit("❌ Aucune fenêtre fixe commune entre SL et IF en test.")

    coverage = len(common) / len(test_windows)
    print(f"Coverage fenêtres fixes SL/IF: {coverage*100:.2f}%")
    if coverage < 0.90:
        raise SystemExit(
            "❌ Couverture SL insuffisante (<90% des fenêtres fixes test). "
            "Résultats potentiellement biaisés; ne pas intégrer au rapport."
        )

    common["sl_pred"] = (common["sl_score"] >= args.sl_threshold).astype(int)
    common["if_pred"] = (common["if_anom_slices"] >= int(args.if_min_anom_slices)).astype(int)

    y_test = common["y_true"].astype(int).values
    sl_operating = _compute_binary_metrics(common["sl_pred"].values, y_test, 0.5)
    if_rule = _compute_binary_metrics(common["if_pred"].values, y_test, 0.5)

    score_col = "if_score_max" if args.if_window_score_agg == "max" else "if_score_mean"
    if_scores_win = common[score_col].fillna(0.0).values
    target_fpr = (args.target_fpr_pct / 100.0) if args.target_fpr_pct is not None else sl_operating.fpr

    # ── PATCH C-01/F02 : seuil IF calibré sur PRE-SPLIT (train_normal) ──
    # L'ancien appel ``_find_if_threshold_matching_fpr(if_scores_win, y_test,
    # target_fpr=target_fpr)`` choisissait le seuil IF en minimisant
    # ``|fpr - target_fpr|`` sur les LABELS DE TEST — c'est du label leakage
    # (audit consolidé §1.1 C-01/F02).  La correction :
    #   1. Calculer les scores IF sur les fenêtres ``train_normal`` (pré-split,
    #      uniquement normales) avec la même agrégation que le test.
    #   2. Choisir le seuil tel que P(score >= thr | train_normal) ≈ target_fpr
    #      — pas d'accès aux labels test ; pas de fuite possible.
    #   3. Appliquer ce seuil sur ``if_scores_win`` (test) pour obtenir la
    #      ``BinaryMetrics`` reportée ensuite.
    #
    # Agrégation train_normal en fenêtres identiques au test (window_start +
    # max/mean), puis calibration par quantile.
    calib_slice_df = train_normal[["timestamp"]].copy()
    calib_slice_df["if_score_slice"] = if_score_slice_calib
    calib_slice_df["window_start"] = calib_slice_df["timestamp"].dt.floor(
        args.decision_window)
    calib_windows = (
        calib_slice_df.groupby("window_start", as_index=False)
        .agg(if_score_max=("if_score_slice", "max"),
              if_score_mean=("if_score_slice", "mean"))
    )
    if calib_windows.empty:
        raise SystemExit(
            "❌ [C-01/F02] Aucune fenêtre normale pré-split pour calibrer "
            "le seuil IF.  Impossible de publier un chiffre IF-fpr-matched "
            "sans fuite ; vérifier ``split_date`` et la disponibilité de "
            "``train_normal``."
        )
    calib_scores_win = calib_windows[score_col].fillna(0.0).values

    if_thr_calibrated, emp_fpr_calib = _calibrate_if_threshold_from_normal(
        calib_scores_win, target_fpr=target_fpr
    )
    if_fpr_matched = _compute_binary_metrics(
        if_scores_win, y_test, if_thr_calibrated
    )
    print(
        f"  [C-01/F02] IF threshold calibré sur pré-split (sans fuite) : "
        f"thr={if_thr_calibrated:.6f}  "
        f"target_fpr={target_fpr:.4f}  "
        f"emp_fpr_calib={emp_fpr_calib:.4f}  "
        f"(n_calib={len(calib_scores_win)})"
    )

    # Variante descriptive: seuil IF en nb de slices = 1 (ultra sensible)
    if_pred_k1 = (common["if_anom_slices"] >= 1).astype(int).values
    if_k1 = _compute_binary_metrics(if_pred_k1, y_test, 0.5)

    os.makedirs(args.output_dir, exist_ok=True)

    summary_df = pd.DataFrame([
        {
            "system": "SL-ADS",
            "regime": "pre-split train (pipeline natif)",
            "feature_space": f"fixed windows {args.decision_window}",
            "operating_point": f"threshold={args.sl_threshold}",
            "threshold": args.sl_threshold,
            "fpr_pct": round(100 * sl_operating.fpr, 4),
            "precision": round(sl_operating.precision, 6),
            "recall": round(sl_operating.recall, 6),
            "f1": round(sl_operating.f1, 6),
            "tp": sl_operating.tp,
            "fp": sl_operating.fp,
            "fn": sl_operating.fn,
            "tn": sl_operating.tn,
        },
        {
            "system": "IF-fair-window",
            "regime": "pre-split train only",
            "feature_space": f"raw metrics -> {args.decision_window}",
            "operating_point": f"IF rule: >= {args.if_min_anom_slices} slices anomalies/window",
            "threshold": float(args.if_min_anom_slices),
            "fpr_pct": round(100 * if_rule.fpr, 4),
            "precision": round(if_rule.precision, 6),
            "recall": round(if_rule.recall, 6),
            "f1": round(if_rule.f1, 6),
            "tp": if_rule.tp,
            "fp": if_rule.fp,
            "fn": if_rule.fn,
            "tn": if_rule.tn,
        },
        {
            # PATCH C-01/F02 : seuil calibré sur pre-split train_normal (pas
            # sur les labels test). Le champ ``regime`` et ``operating_point``
            # reflètent explicitement la nouvelle politique pour la table.
            "system": "IF-fpr-matched",
            "regime": "pre-split train only (threshold calibrated on "
                      "pre-split normal-only windows — leak-free, C-01/F02)",
            "feature_space": f"raw metrics -> {args.decision_window} ({score_col})",
            "operating_point": (f"decision_function threshold calibrated on "
                                f"pre-split normals @ target FPR "
                                f"{target_fpr*100:.3f}% "
                                f"(calib emp FPR {emp_fpr_calib*100:.3f}%, "
                                f"n_calib={len(calib_scores_win)})"),
            "threshold": if_thr_calibrated,
            "fpr_pct": round(100 * if_fpr_matched.fpr, 4),
            "precision": round(if_fpr_matched.precision, 6),
            "recall": round(if_fpr_matched.recall, 6),
            "f1": round(if_fpr_matched.f1, 6),
            "tp": if_fpr_matched.tp,
            "fp": if_fpr_matched.fp,
            "fn": if_fpr_matched.fn,
            "tn": if_fpr_matched.tn,
        },
        {
            "system": "IF-k1-descriptive",
            "regime": "pre-split train only",
            "feature_space": f"raw metrics -> {args.decision_window}",
            "operating_point": "IF rule: >= 1 slice anomalie/window",
            "threshold": 1.0,
            "fpr_pct": round(100 * if_k1.fpr, 4),
            "precision": round(if_k1.precision, 6),
            "recall": round(if_k1.recall, 6),
            "f1": round(if_k1.f1, 6),
            "tp": if_k1.tp,
            "fp": if_k1.fp,
            "fn": if_k1.fn,
            "tn": if_k1.tn,
        },
    ])

    # ─── PATCH-M3 (2026-04-18 / fix 2026-04-19) : McNemar paired test (Dietterich 1998) ─
    # Variables corrigées pour refléter les noms locaux :
    #   sl_preds       -> common["sl_pred"].values
    #   if_preds_fair  -> common["if_pred"].values
    #   if_preds_matched -> (if_scores_win >= if_fpr_matched.threshold).astype(int)
    #   if_preds_k1    -> if_pred_k1 (déjà défini L.572, singulier)
    #   out_dir        -> args.output_dir
    # Ref : Dietterich (1998) Neural Computation 10(7):1895-1923.
    from statsmodels.stats.contingency_tables import mcnemar

    def _mcnemar_sl_vs_if(y_true, y_pred_sl, y_pred_if, label=""):
        """Test de McNemar (1947 Psychometrika) pour classifiers appariés."""
        # Table de contingence : les prédictions correctes/incorrectes des deux systèmes
        both_ok = int(((y_pred_sl == y_true) & (y_pred_if == y_true)).sum())
        sl_ok_if_ko = int(((y_pred_sl == y_true) & (y_pred_if != y_true)).sum())
        sl_ko_if_ok = int(((y_pred_sl != y_true) & (y_pred_if == y_true)).sum())
        both_ko = int(((y_pred_sl != y_true) & (y_pred_if != y_true)).sum())
        table = [[both_ok, sl_ok_if_ko], [sl_ko_if_ok, both_ko]]
        result = mcnemar(table, exact=True)
        print(f"  McNemar SL vs {label}: statistic={result.statistic:.3f}, "
              f"p-value={result.pvalue:.4f}")
        return {'label': label, 'table': table,
                'statistic': float(result.statistic),
                'pvalue': float(result.pvalue)}

    sl_preds_arr       = common["sl_pred"].astype(int).values
    if_preds_fair_arr  = common["if_pred"].astype(int).values
    if_preds_matched_arr = (if_scores_win >= if_fpr_matched.threshold).astype(int)

    mcnemar_results = []
    for if_name, if_preds in [('IF-fair',        if_preds_fair_arr),
                              ('IF-fpr-matched', if_preds_matched_arr),
                              ('IF-k1',          if_pred_k1)]:
        mcnemar_results.append(
            _mcnemar_sl_vs_if(y_test, sl_preds_arr, if_preds, label=if_name)
        )

    pd.DataFrame(mcnemar_results).to_csv(
        os.path.join(args.output_dir, 'mcnemar_sl_vs_if.csv'), index=False
    )

    # ─── BCa 95% CI on F1 (TASK-10, PROC-03) ──────────────────────────────
    # Wires sl_ads.stats.bootstrap_ci into the comparison so every reported
    # F1 ships with its second-order-accurate (Efron 1987) interval and so
    # the SL-vs-IF gap comes with a paired-difference CI (whose exclusion
    # of zero is the rigorous analogue of McNemar p < 0.05).
    from sl_ads.stats.bootstrap_ci import (
        bootstrap_bca_ci,
        paired_bootstrap_bca_ci,
    )
    from sklearn.metrics import f1_score as _f1_score

    _bca_systems = [
        ("SL-ADS",            sl_preds_arr),
        ("IF-fair-window",    if_preds_fair_arr),
        ("IF-fpr-matched",    if_preds_matched_arr),
        ("IF-k1-descriptive", if_pred_k1),
    ]

    _f1_ci_rows = []
    _f1_ci_by_system: dict[str, dict] = {}
    for _sys_name, _preds in _bca_systems:
        _ci = bootstrap_bca_ci(y_test, _preds, _f1_score, n_boot=2000, seed=args.random_state)
        _f1_ci_by_system[_sys_name] = _ci
        _f1_ci_rows.append({
            "system": _sys_name,
            "metric": "f1",
            "point": round(_ci["point"], 6),
            "ci_low": round(_ci["ci_low"], 6),
            "ci_high": round(_ci["ci_high"], 6),
            "n_boot": _ci.get("n_boot", 2000),
            "alpha": _ci.get("alpha", 0.05),
        })

    pd.DataFrame(_f1_ci_rows).to_csv(
        os.path.join(args.output_dir, "bootstrap_ci_f1.csv"), index=False
    )

    _paired_rows = []
    for _if_name, _if_preds in [
        ("IF-fair-window",    if_preds_fair_arr),
        ("IF-fpr-matched",    if_preds_matched_arr),
        ("IF-k1-descriptive", if_pred_k1),
    ]:
        _pair = paired_bootstrap_bca_ci(
            y_test, sl_preds_arr, _if_preds, _f1_score,
            n_boot=2000, seed=args.random_state,
        )
        _paired_rows.append({
            "comparison": f"SL-ADS vs {_if_name}",
            "metric": "f1_diff (SL - IF)",
            "point": round(_pair["point"], 6),
            "ci_low": round(_pair["ci_low"], 6),
            "ci_high": round(_pair["ci_high"], 6),
            "ci_excludes_zero": bool(_pair["ci_low"] > 0 or _pair["ci_high"] < 0),
            "n_boot": _pair.get("n_boot", 2000),
            "alpha": _pair.get("alpha", 0.05),
        })

    pd.DataFrame(_paired_rows).to_csv(
        os.path.join(args.output_dir, "bootstrap_ci_paired_sl_vs_if.csv"),
        index=False,
    )

    # Augment per-system summary with f1 CI columns (single-sample BCa).
    summary_df["f1_ci_low"] = summary_df["system"].map(
        lambda s: round(_f1_ci_by_system[s]["ci_low"], 6) if s in _f1_ci_by_system else float("nan")
    )
    summary_df["f1_ci_high"] = summary_df["system"].map(
        lambda s: round(_f1_ci_by_system[s]["ci_high"], 6) if s in _f1_ci_by_system else float("nan")
    )

    print("\n  BCa 95% CI on F1 (single-sample):")
    for _row in _f1_ci_rows:
        print(f"    {_row['system']:<22} F1 = {_row['point']:.3f} "
              f"[{_row['ci_low']:.3f}, {_row['ci_high']:.3f}]")
    print("  BCa 95% CI on paired F1 difference (SL − IF):")
    for _row in _paired_rows:
        _star = " *" if _row["ci_excludes_zero"] else ""
        print(f"    {_row['comparison']:<32} ΔF1 = {_row['point']:+.3f} "
              f"[{_row['ci_low']:+.3f}, {_row['ci_high']:+.3f}]{_star}")

    points_df = pd.DataFrame([
        {
            "name": "decision_window",
            "value": args.decision_window,
        },
        {
            "name": "if_min_anom_slices",
            "value": args.if_min_anom_slices,
        },
        {
            "name": "n_common_windows",
            "value": len(common),
        },
        {
            "name": "if_contamination",
            "value": args.if_contamination,
        },
        {
            "name": "if_contamination_source",
            "value": contamination_source,
        },
        {
            "name": "if_contamination_effective",
            "value": contamination if isinstance(contamination, str) else round(float(contamination), 6),
        },
        {
            "name": "if_fpr_target_pct",
            "value": round(target_fpr * 100.0, 6),
        },
    ])

    out_summary = os.path.join(args.output_dir, "fair_if_vs_sl_summary.csv")
    out_points = os.path.join(args.output_dir, "fair_if_vs_sl_points.csv")
    out_report = os.path.join(args.output_dir, "fair_if_vs_sl_report.md")

    summary_df.to_csv(out_summary, index=False)
    points_df.to_csv(out_points, index=False)

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("# Fair comparison SL-ADS vs Isolation Forest\n\n")
        f.write("## Methodological constraints enforced\n")
        f.write("1. Training regime: IF trained only on pre-split normal windows.\n")
        f.write("2. Feature space: IF trained on raw network metrics (ACTIVE_METRICS).\n")
        f.write(f"3. Decision unit parity: SL and IF compared on fixed {args.decision_window} windows (strict, no tolerance).\n")
        f.write(f"4. IF operational rule: anomaly if >= {args.if_min_anom_slices} anomalous slices per fixed window.\n\n")
        f.write("5. IF score-based operating point uses decision_function on fixed windows.\n")
        f.write(f"   - Contamination CLI: {args.if_contamination}\n")
        f.write(f"   - Contamination effective: {contamination} (source: {contamination_source})\n")
        f.write(
            "   - Policy: contamination is never calibrated on TEST labels "
            "(leakage-free protocol).\n"
        )
        f.write(f"   - FPR target alignment (Method B): {target_fpr*100:.4f}%\n")
        # PATCH C-01/F02 : rendre visible dans le rapport la politique de
        # calibration du seuil IF-fpr-matched (leak-free).
        f.write(
            f"6. IF-fpr-matched threshold calibration: computed on the "
            f"pre-split *normal-only* calibration set "
            f"(n_calib={len(calib_scores_win)} windows).  "
            f"Target FPR = {target_fpr*100:.4f}% ; "
            f"empirical FPR on calibration = {emp_fpr_calib*100:.4f}% ; "
            f"calibrated threshold = {if_thr_calibrated:.6f}.  "
            f"No test label is used in threshold selection "
            f"(addresses C-01/F02 of the consolidated audit).\n\n"
        )
        f.write("## Dataset windows\n")
        f.write(f"- Train: {len(df_train)} windows\n")
        f.write(f"- Test: {len(df_test)} windows\n")
        f.write(f"- Split date: {split_date}\n\n")
        f.write("## Results\n\n")
        f.write(_to_markdown_table(summary_df))
        f.write("\n")

    print("\n=== Résultats ===")
    print(summary_df.to_string(index=False))
    print(f"\n✅ CSV résumé  : {out_summary}")
    print(f"✅ CSV points  : {out_points}")
    print(f"✅ Rapport MD  : {out_report}")
    print(f"✅ BCa CI F1   : {os.path.join(args.output_dir, 'bootstrap_ci_f1.csv')}")
    print(f"✅ BCa CI paired: {os.path.join(args.output_dir, 'bootstrap_ci_paired_sl_vs_if.csv')}")


if __name__ == "__main__":
    main()