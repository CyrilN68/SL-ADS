"""
evaluate_qualify_sbn.py — Évaluation de qualify_anomaly_sbn.py
==============================================================
Compatible avec les deux formats de sortie :
  - qualify_anomaly.py     : colonnes 'b_qualif_*', 'u_qualif'
  - qualify_anomaly_sbn.py : colonnes 'b_sbn_*', 'u_sbn'

Lit automatiquement le bon format selon les colonnes présentes.

Métriques calculées :
  DR  — Detection Rate         = n_detected / n_total
                                  fraction des fenêtres d'attaque où gate_open=True
                                  ≠ recall de détection global (cf. evaluate_injection_v2.py)
  QP  — Qualification Precision = n_correct / n_detected
                                  fraction des détections avec top-1 correct
                                  ≠ précision binaire (pas de comptage des vrais négatifs globaux)
  F1  — Moyenne harmonique de DR et QP
  TTQ — Time-to-Qualify        : temps (min) jusqu'à la 1ère fenêtre correctement qualifiée
  Macro-DR / Macro-QP / F1     : moyenne non pondérée sur K types (poids égaux)
  Micro-DR / Micro-QP / F1     : pondérée par n_total_k (représentative par volume)
  F-β (β=2)— F-measure pondéré  = (1+β²)*DR*QP / (β²*DR + QP)  β=2 → DR 2× plus important
                                  Standard IDS : β=2 (Tavallaee 2009 ; Moustafa 2015)
  FAR — False Alarm Rate        = FP / (FP+TN) sur les fenêtres hors-injection
                                  FP = gate_open pendant trafic normal
                                  Complémentaire du DR (trade-off détection/fausse alarme)
  MCC — Matthews Correlation    = (TP·TN - FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        Coefficient (détection)   Robuste aux classes déséquilibrées (Matthews 1975)
  AUC-ROC novelty_lr           : discriminabilité connu(0) vs inconnu(1) — INFORMATION ONLY
  novelty_entropy              : entropie de Shannon normalisée sur b_sbn_*
                                 ∈ [0,1], 0 = type dominant, 1 = distribution uniforme
                                 Invariante à K (normalisation log2(K))

Ground truth (H1) : toutes les fenêtres dans [t_start, t_end] sont considérées positives
(injection active et uniforme sur toute la durée déclarée).

Types dans le catalogue SBN mais SANS injection disponible (non évalués ici) :
  NETWORK_OUTAGE → évalué via --real (REAL_ATTACKS config.py)
  BGP_HIJACK     → pas d'injection disponible
  BOTNET_CC      → pas d'injection disponible

Usage :
    python evaluate_qualify_sbn.py                         # injected (défaut)
    python evaluate_qualify_sbn.py --injected              # attaques injectées
    python evaluate_qualify_sbn.py --real                  # attaques réelles
    python evaluate_qualify_sbn.py --both                  # injected + real
    python evaluate_qualify_sbn.py --csv qualif_types.csv  # chemin CSV explicite
    python evaluate_qualify_sbn.py --no_save               # pas de fichier de sortie
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score, roc_curve as _sklearn_roc_curve
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# ── PATCH m-03/F20 + m-04/F21 + m-09/F30 ───────────────────────────────────
# Shared qualification-filter helpers (see qualif_filters.py).
# `qualified_mask`               : keep only rows where gate_open AND the
#                                  qualifier actually produced a non-sentinel
#                                  inference (qual_status != 'no_groups').
#                                  Required for meaningful statistics on
#                                  u_sbn / novelty_lr / top1_type / entropy.
# `partition_catalog_by_outage` : split an event list into (attacks, outages)
#                                  so NETWORK_OUTAGE events don't pollute
#                                  attack-precision aggregates (m-04/F21).
# `is_outage_event`              : single-event predicate used when iterating
#                                  REAL_ATTACKS and routing each entry.
from sl_ads.qualif_filters import (  # Phase H
    qualified_mask,
    partition_catalog_by_outage,
    is_outage_event,
)

try:
    from sl_ads.config import CONFIG, REAL_ATTACKS  # Phase H
    from sl_ads.paths import get_results_dir  # Phase H
    RESULTS_DIR    = get_results_dir(CONFIG, up_levels=1)
    WINDOW_MIN     = CONFIG.get('WINDOW_MINUTES', 5)
    NOVELTY_THR    = CONFIG.get('SBN_NOVELTY_THRESHOLD',
                                CONFIG.get('QUALIFY_UM_NOVELTY_THRESHOLD', 0.50))
    # ⚠️  PATCH-C2 (2026-04-18) : le seuil binarisé est RETIRÉ des résultats publiables.
    # Motif : LR_NOVELTY_THR=0.71 était dérivé du Youden index mesuré sur le jeu TEST
    # (Youden=0.734), ce qui constitue une sélection de seuil test-on-test —
    # leakage méthodologique (Varma & Simon 2006, BMC Bioinformatics 7:91).
    #
    # Usage recommandé pour le papier :
    #   - rapporter AUC-ROC + intervalle de confiance bootstrap (Hanley & McNeil 1982)
    #   - rapporter la courbe ROC complète en annexe
    #   - NE PAS rapporter de métrique dépendant de ce seuil (precision/recall/F1)
    #
    # Usage opérationnel (déploiement) :
    #   - calibrer sur un jeu hold-out disjoint du test (ex : 10 % du CALIB réservé)
    #   - documenter la FPR cible opérationnelle AVANT mesure
    LR_NOVELTY_THR = CONFIG.get('SBN_LR_NOVELTY_THRESHOLD', None)  # None = reporting-only

    if LR_NOVELTY_THR is not None:
        import warnings

        warnings.warn(
            "LR_NOVELTY_THR défini — les métriques binarisées de nouveauté sont "
            "INFORMATIONNELLES et ne doivent pas être rapportées comme validation externe. "
            "Voir PATCH-C2 (SCIENTIFIC_AUDIT.md §9.2).",
            UserWarning, stacklevel=2
        )
except ImportError:
    # Phase H — fallback when sl_ads.config cannot be imported (e.g.
    # standalone smoke run from a clean checkout).  Was historically
    # 'modèle évaluation' (renamed in Phase H to 'investigations').
    RESULTS_DIR    = 'investigations'
    WINDOW_MIN     = 5
    NOVELTY_THR    = 0.65
    LR_NOVELTY_THR = 0.85
    REAL_ATTACKS   = {}

# Paramètre β pour le F-β score (IDS : β=2 → DR deux fois plus important que QP)
# Ref : van Rijsbergen (1979) Information Retrieval §7 ;
#       Tavallaee et al. (2009) IEEE NSS — évaluation des IDS sur KDD Cup'99
FBETA: float = 2.0

# ──────────────────────────────────────────────────────────────────────────────
# CALCUL GLOBAL DES STATISTIQUES DE DÉTECTION (FAR, MCC)
# ──────────────────────────────────────────────────────────────────────────────

def _event_start_end(ev: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return inclusive event bounds for {start,end} or {start,duration_h}."""
    t0 = pd.Timestamp(ev['start'])
    if ev.get('end') is not None:
        return t0, pd.Timestamp(ev['end'])
    return t0, t0 + pd.Timedelta(hours=float(ev['duration_h']))


def _compute_global_detection_stats(
        df: pd.DataFrame,
        attack_events: list,
        outage_events: list | None = None,
) -> dict:
    """
    Calcule FAR et MCC sur l'ensemble du dataset en distinguant TROIS classes
    de fenêtres (patch m-04/F21, taxonomie séparée des NETWORK_OUTAGE) :

      - Fenêtres d'attaque  : dans [t_start, t_end) de chaque entrée d'``attack_events``
                              + entrées non-outage de REAL_ATTACKS
                              + CONFIG["EVAL"]["REAL_ATTACK_CATALOG"].
      - Fenêtres d'outage    : dans [t_start, t_end) de chaque entrée d'``outage_events``
                              + entrées outage de REAL_ATTACKS (``is_outage_event``).
                              Ces fenêtres sont EXCLUES de la base FAR (pas du trafic
                              normal) ET de la base TP/FN (pas des attaques).
      - Fenêtres normales   : toutes les autres.

    Définitions :
      TP  = gate_open dans les périodes d'attaque
      FN  = gate_closed dans les périodes d'attaque
      FP  = gate_open dans les périodes normales   → fausses alarmes
      TN  = gate_closed dans les périodes normales
      FAR = FP / (FP + TN)   [False Alarm Rate pendant trafic normal]
      MCC = (TP·TN - FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Les statistiques sur les fenêtres d'outage sont rapportées séparément
    (``gate_open_during_outage``) : taux de détection gate pendant une panne
    data-plane n'a pas la même sémantique qu'une détection d'attaque
    (cf. qualif_filters.is_outage_event).

    Ref : Matthews (1975) BBA — coefficient de corrélation φ (MCC).
          Brodersen et al. (2010) ICPR — interprétabilité du MCC.
          Sokolova & Lapalme (2009) IPM — métriques multi-classes pour IDS.
    """
    # Marquer toutes les fenêtres appartenant à une période d'attaque
    in_attack = pd.Series(False, index=df.index)
    for ev in attack_events or []:
        t0, t1 = _event_start_end(ev)
        in_attack |= (df['timestamp'] >= t0) & (df['timestamp'] < t1)

    # Marquer toutes les fenêtres d'outage (séparées des attaques, m-04/F21)
    in_outage = pd.Series(False, index=df.index)
    for ev in outage_events or []:
        t0, t1 = _event_start_end(ev)
        in_outage |= (df['timestamp'] >= t0) & (df['timestamp'] < t1)

    # Parcourir REAL_ATTACKS et router chaque entrée vers la bonne bucket.
    # is_outage_event tolère _R2/_R3 markers et différents champs d'étiquette.
    for _events in REAL_ATTACKS.values():
        for ev in _events:
            t0, t1 = _event_start_end(ev)
            ev_mask = (df['timestamp'] >= t0) & (df['timestamp'] < t1)
            if is_outage_event(ev):
                in_outage |= ev_mask
            else:
                in_attack |= ev_mask

    # A3.2: the operational FAR denominator must also remove/credit the
    # explicit REAL_ATTACK_CATALOG used by evaluate_injection.py.  This keeps
    # the headline FAR from counting the RedeRio DDoS annotation as a FP.
    for ev in CONFIG.get("EVAL", {}).get("REAL_ATTACK_CATALOG", []) or []:
        t0, t1 = _event_start_end(ev)
        ev_mask = (df['timestamp'] >= t0) & (df['timestamp'] < t1)
        if is_outage_event(ev):
            in_outage |= ev_mask
        else:
            in_attack |= ev_mask

    # Base "normale" = ni attaque ni outage (base propre pour FAR/MCC)
    in_normal = ~(in_attack | in_outage)

    df_atk = df[in_attack]
    df_out = df[in_outage]
    df_nrm = df[in_normal]

    TP = int(df_atk['gate_open'].sum())
    FN = int((~df_atk['gate_open']).sum())
    FP = int(df_nrm['gate_open'].sum())
    TN = int((~df_nrm['gate_open']).sum())

    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    denom_mcc = np.sqrt(float((TP + FP)) * float((TP + FN))
                        * float((TN + FP)) * float((TN + FN)))
    MCC = (TP * TN - FP * FN) / denom_mcc if denom_mcc > 0 else 0.0

    # Statistiques outage (séparées, m-04/F21)
    n_outage_windows   = int(len(df_out))
    gate_during_outage = int(df_out['gate_open'].sum()) if n_outage_windows > 0 else 0

    return {
        'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN,
        'n_attack_windows': len(df_atk),
        'n_normal_windows': len(df_nrm),
        'n_outage_windows': n_outage_windows,
        'gate_open_during_outage': gate_during_outage,
        'outage_gate_rate': (
            round(gate_during_outage / n_outage_windows, 4)
            if n_outage_windows > 0 else None
        ),
        'FAR': round(FAR, 4),
        'MCC': round(float(MCC), 4),
        'real_attack_catalog_subtracted_from_far': True,
        'real_attack_catalog_events': len(CONFIG.get("EVAL", {}).get("REAL_ATTACK_CATALOG", []) or []),
    }

# ──────────────────────────────────────────────────────────────────────────────
# CATALOGUE DES ATTAQUES INJECTÉES — imported from config.INJECTED_ATTACK_CATALOG
# DRY-compliant depuis PATCH-C1 (audit 2026-04-18).
# ──────────────────────────────────────────────────────────────────────────────
try:
    from sl_ads.config import INJECTED_ATTACK_CATALOG as INJECTED_ATTACKS  # Phase H
except ImportError as _e:
    raise ImportError(
        "INJECTED_ATTACK_CATALOG absent de config.py — "
        "appliquer PATCH-C1 avant d'exécuter ce script"
    ) from _e


# ──────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ──────────────────────────────────────────────────────────────────────────────

def _detect_format(df: pd.DataFrame) -> str:
    """
    Détecte le format du CSV produit par la qualification :
      - 'sbn'    : SL-template qualifier, legacy SBN column names               → b_sbn_*,    u_sbn
      - 'argmax' : baseline no-SL argmax (qualify_argmax_baseline.py)           → b_argmax_*, u_argmax
      - 'lr'     : Likelihood-ratio legacy (ancien qualify_anomaly.py)          → b_qualif_*, u_qualif
    """
    if any(c.startswith('b_sbn_') for c in df.columns):
        return 'sbn'
    if any(c.startswith('b_argmax_') for c in df.columns):
        return 'argmax'
    if any(c.startswith('b_qualif_') for c in df.columns):
        return 'lr'
    raise ValueError("Format de CSV non reconnu : ni b_sbn_* ni b_argmax_* ni b_qualif_*")


def _get_col_prefix(fmt: str) -> tuple[str, str]:
    """Retourne (prefix_beliefs, col_uncertainty) selon le format."""
    if fmt == 'sbn':
        return 'b_sbn_', 'u_sbn'
    if fmt == 'argmax':
        return 'b_argmax_', 'u_argmax'
    return 'b_qualif_', 'u_qualif'


def _validate_columns(df: pd.DataFrame) -> None:
    """Vérifie l'existence des colonnes critiques. Lève ValueError si manquantes."""
    required = ['timestamp', 'gate_open', 'top1_type']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes critiques manquantes dans le CSV : {missing}.\n"
            f"Vérifier que qualify_anomaly_sbn.py (version actuelle) a bien produit ce fichier."
        )


def _add_novelty_entropy(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Calcule l'entropie de Shannon normalisée sur la distribution de croyances b_sbn_*.

    novelty_entropy = H(b) / log2(K)   ∈ [0, 1]
    où H(b) = -Σ_k b_k * log2(b_k)  (entropie de Shannon)
        K   = nombre de types actifs (colonnes b_sbn_* hors '_raw_')

    Propriétés :
      novelty_entropy → 0  : un type domine (attaque bien identifiée)
      novelty_entropy → 1  : distribution uniforme (anomalie inconnue)

    Avantage vs novelty_lr : invariante au nombre de types K.
    L'ajout de nouveaux types dans le catalogue ne dégrade pas la discriminabilité
    car on normalise par log2(K).

    Ref : Shannon (1948) BSTJ — "A Mathematical Theory of Communication".
          Jøsang (2016) §3.3 — mesure d'incertitude subjective.
    """
    b_cols = [c for c in df.columns if c.startswith(prefix) and '_raw_' not in c]
    if not b_cols:
        df = df.copy()
        df['novelty_entropy'] = float('nan')
        return df

    K      = len(b_cols)
    b_vals = df[b_cols].values.clip(0)
    b_sum  = b_vals.sum(axis=1, keepdims=True)
    # Normalisation vers distribution de probabilité
    with np.errstate(invalid='ignore', divide='ignore'):
        b_norm = np.where(b_sum > 1e-10, b_vals / b_sum, 1.0 / K)
    # Entropie de Shannon : -Σ p log2(p), avec 0*log2(0) = 0 par convention
    with np.errstate(invalid='ignore', divide='ignore'):
        log_b = np.where(b_norm > 1e-12, np.log2(b_norm), 0.0)
    H     = -np.sum(b_norm * log_b, axis=1)
    H_max = np.log2(K) if K > 1 else 1.0
    df = df.copy()
    df['novelty_entropy'] = H / H_max
    return df


def _compute_novelty_auc(
        df: pd.DataFrame,
        known_attacks: list,
        novelty_attacks: list,
        lr_col: str,
) -> dict | None:
    """
    Calcule l'AUC-ROC du signal `novelty_lr` pour la tâche de détection de nouveauté.
    Labels : attaques connues → 0,  anomalies inconnues → 1.

    ⚠️  INFORMATION ONLY — pas de data leakage au sens strict :
      L'AUC caractérise la discriminabilité du signal pour le RAPPORT TECHNIQUE,
      mais NE DOIT PAS être utilisée pour sélectionner LR_NOVELTY_THR sur ces données.
      Sélectionner un seuil sur le même jeu = data leakage.
      Le seuil de Youden est fourni comme estimation in-sample de référence ;
      la calibration correcte requiert un jeu de validation indépendant.

    Ref : Hanley & McNeil (1982) Radiology — AUC-ROC comme mesure de discrimination.
          Youden (1950) Cancer — J = TPR - FPR, indice de performance diagnostique.
    """
    # PATCH m-03/F20 : restreindre aux fenêtres RÉELLEMENT qualifiées.
    # Inclure les fenêtres qual_status='no_groups' (u_sbn=1, novelty_lr=1 sentinel)
    # biaiserait l'AUC — ces valeurs ne proviennent pas d'une inférence.
    qual = qualified_mask(df)
    scores, labels = [], []
    for ev in known_attacks:
        t0   = pd.Timestamp(ev['start'])
        t1   = pd.Timestamp(ev['end'])
        mask = (df['timestamp'] >= t0) & (df['timestamp'] < t1) & qual
        vals = df.loc[mask, lr_col].dropna().tolist()
        scores.extend(vals)
        labels.extend([0] * len(vals))
    for ev in novelty_attacks:
        t0   = pd.Timestamp(ev['start'])
        t1   = pd.Timestamp(ev['end'])
        mask = (df['timestamp'] >= t0) & (df['timestamp'] < t1) & qual
        vals = df.loc[mask, lr_col].dropna().tolist()
        scores.extend(vals)
        labels.extend([1] * len(vals))

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0 or len(scores) < 4:
        return None

    scores_arr = np.array(scores, dtype=float)
    labels_arr = np.array(labels, dtype=int)

    if _HAS_SKLEARN:
        try:
            auc = float(roc_auc_score(labels_arr, scores_arr))
            fpr_arr, tpr_arr, thr_arr = _sklearn_roc_curve(labels_arr, scores_arr)
            j_idx     = int(np.argmax(tpr_arr - fpr_arr))
            youden_thr = float(thr_arr[j_idx])
        except Exception:
            return None
    else:
        # Calcul AUC manuel par intégration trapézoïdale
        thresholds = sorted(set(scores_arr.tolist()), reverse=True)
        fpr_pts, tpr_pts = [0.0], [0.0]
        for thr in thresholds:
            tp = int(((scores_arr >= thr) & (labels_arr == 1)).sum())
            fp = int(((scores_arr >= thr) & (labels_arr == 0)).sum())
            fpr_pts.append(fp / n_neg)
            tpr_pts.append(tp / n_pos)
        fpr_pts.append(1.0)
        tpr_pts.append(1.0)
        auc    = float(np.trapz(tpr_pts, fpr_pts))
        j_vals = np.array(tpr_pts) - np.array(fpr_pts)
        j_idx  = int(np.argmax(j_vals))
        youden_thr = float(thresholds[j_idx - 1]) if 0 < j_idx <= len(thresholds) else float('nan')

    return {
        'auc': auc,
        'youden_thr': youden_thr,
        'n_known_windows': n_neg,
        'n_novel_windows': n_pos,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ÉVALUATION PRINCIPALE — ATTAQUES INJECTÉES
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_injected(df: pd.DataFrame, fmt: str) -> tuple[list, list, dict]:
    """
    Évalue la qualification sur les attaques injectées synthétiquement.
    Retourne (known_rows, novelty_rows, global_stats) pour sauvegarde CSV.

    PATCH m-03/F20 + m-04/F21 (2026-04-23)
    --------------------------------------
    * Les statistiques top-1 / u / novelty_lr / entropy sont calculées sur les
      fenêtres ``qualified_mask`` (gate_open AND qual_status != 'no_groups').
      Inclure 'no_groups' biaiserait u_mean vers 1 et pollurait la distribution
      top1 (sentinels). Voir ``qualif_filters.qualified_mask``.
    * Le DR (Detection Rate) reste calculé sur ``gate_open`` seul — gate ouverte
      sans groupes qualifiables est toujours un événement de détection.
    * Les événements NETWORK_OUTAGE éventuels dans le catalogue (si présents)
      sont retirés et traités séparément par ``partition_catalog_by_outage``.
    """
    prefix, u_col = _get_col_prefix(fmt)
    lr_col      = 'novelty_lr' if 'novelty_lr' in df.columns else u_col
    has_entropy = 'novelty_entropy' in df.columns

    # PATCH m-04/F21 : isoler les NETWORK_OUTAGE du catalogue (s'il y en a).
    # INJECTED_ATTACK_CATALOG ne contient historiquement pas d'outages, mais
    # cette séparation prépare l'ajout éventuel de NETWORK_OUTAGE_SYNTHETIC
    # (cf. review_evaluate_qualify_sbn.md §6.1) sans régression.
    attack_events, outage_events = partition_catalog_by_outage(INJECTED_ATTACKS)

    W = 134
    print(f"\n{'─'*W}")
    print(f"  {'Attaque':<28} {'Intens.':<8} {'Fenêtres':>9} "
          f"{'Détect.':>9} {'DR':>7} {'QP':>8} {'F1':>7} {'F2':>7}  {'Top-1 incorrect'}")
    print(f"{'─'*W}")

    known_rows   = []
    novelty_rows = []
    known_atk_roc   = []
    novelty_atk_roc = []

    for ev in attack_events:
        t_start  = pd.Timestamp(ev['start'])
        t_end    = pd.Timestamp(ev['end'])
        expected = ev['expected']

        mask    = (df['timestamp'] >= t_start) & (df['timestamp'] < t_end)
        df_ev   = df[mask].copy()
        # df_det : fenêtres gate_open — base pour DR (détection binaire).
        df_det  = df_ev[df_ev['gate_open']]
        # df_qual : fenêtres réellement qualifiées — base pour u/lr/entropy/top1 (m-03/F20).
        df_qual = df_ev[qualified_mask(df_ev)]

        n_total     = len(df_ev)
        n_detected  = int(df_ev['gate_open'].sum())
        n_qualified = int(qualified_mask(df_ev).sum())
        dr          = n_detected / n_total if n_total > 0 else 0.0

        # u/lr/entropy : calculés sur les qualifiées (m-03/F20) — les lignes
        # 'no_groups' portent des sentinels (u=1, novelty_lr=1) qui biaiseraient
        # ces moyennes si on utilisait df_det.
        u_mean   = df_qual[u_col].mean()  if n_qualified > 0 else float('nan')
        lr_mean  = df_qual[lr_col].mean() if n_qualified > 0 else float('nan')
        # PATCH-C2 fix (2026-04-20) : quand LR_NOVELTY_THR=None (défaut post-audit,
        # reporting-only), on NE CALCULE PAS la fraction binarisée ni le "signal
        # dépassé", car cela nécessiterait un seuil test-on-test (Varma & Simon 2006).
        # Les comparaisons `>` avec None lèvent TypeError.
        if LR_NOVELTY_THR is not None and n_qualified > 0:
            lr_frac = float((df_qual[lr_col] > LR_NOVELTY_THR).mean())
        else:
            lr_frac = float('nan')
        ent_mean = df_qual['novelty_entropy'].mean() if (has_entropy and n_qualified > 0) else float('nan')

        # ── Contrôle de nouveauté ──────────────────────────────────────────
        if ev['is_novelty_control']:
            # PATCH-C2 fix : _sig_ok est défini UNIQUEMENT si on a un seuil.
            if LR_NOVELTY_THR is None:
                _sig_ok  = None   # reporting-only, pas de binarisation
                sig_str  = "— reporting-only (PATCH-C2)"
            else:
                _sig_ok  = bool(not np.isnan(lr_mean) and lr_mean > LR_NOVELTY_THR)
                sig_str  = "✓ signal nouveauté" if _sig_ok else "✗ non signalé"
            # Top-1 : calculée sur les fenêtres qualifiées (m-03/F20).
            top1_dis = df_qual['top1_type'].value_counts().head(3) if n_qualified > 0 else pd.Series([], dtype=int)
            top1_str = ', '.join(f"{t}({c})" for t, c in top1_dis.items()) if len(top1_dis) else "—"

            print(f"  {ev['name']:<28} {ev['intensity']:<8} {n_total:>9} "
                  f"{n_detected:>9} {dr*100:>6.1f}%  {'N/A (contrôle)':<18}  {sig_str}")
            if LR_NOVELTY_THR is None:
                print(f"  {'  novelty_lr moyen':<34}: {lr_mean:.3f}  "
                      f"[AUC-ROC à rapporter, pas de seuil binarisé — PATCH-C2]")
            else:
                print(f"  {'  novelty_lr moyen':<34}: {lr_mean:.3f}  "
                      f"frac>{LR_NOVELTY_THR:.2f}: {lr_frac*100:.1f}%  "
                      f"[seuil {'dépassé ✓' if _sig_ok else 'non dépassé ✗'}]")
            if has_entropy:
                print(f"  {'  novelty_entropy moyen':<34}: {ent_mean:.3f}")
            print(f"  {'  Top-1 attribué (informatif)':<34}: {top1_str}")

            novelty_rows.append({
                'attack': ev['name'], 'n_total': n_total, 'n_detected': n_detected,
                'dr': round(dr, 4),
                'lr_mean': round(lr_mean, 4) if not np.isnan(lr_mean) else None,
                'lr_frac_above_thr': round(lr_frac, 4) if not np.isnan(lr_frac) else None,
                'entropy_mean': round(ent_mean, 4) if not np.isnan(ent_mean) else None,
                'novelty_detected': _sig_ok, 'top1_dist': top1_str,
            })
            novelty_atk_roc.append(ev)

        # ── Attaque connue ─────────────────────────────────────────────────
        else:
            # PATCH m-03/F20 : QP mesure la précision de qualification sur les
            # fenêtres qualifiées (non 'no_groups').  Utiliser df_det inclurait
            # les fenêtres no_groups dont le top1_type est un sentinel
            # (injecté par _empty_result()) qui compte presque toujours comme
            # "incorrect", biaisant QP à la baisse. La base légitime est df_qual.
            n_correct = int((df_qual['top1_type'] == expected).sum())
            qp   = n_correct / n_qualified if n_qualified > 0 else 0.0
            f1   = 2 * dr * qp / (dr + qp) if (dr + qp) > 0 else 0.0
            # F-β (β=FBETA=2) : pénalise moins les faux-négatifs (DR > QP en importance)
            # Ref : van Rijsbergen (1979) ; Tavallaee (2009) NSS pour IDS
            fb2  = (1 + FBETA**2) * dr * qp / (FBETA**2 * dr + qp) if (FBETA**2 * dr + qp) > 0 else 0.0

            wrong     = df_qual[df_qual['top1_type'] != expected]['top1_type']
            wrong_str = ', '.join(
                f"{t}({c})" for t, c in wrong.value_counts().head(3).items()
            ) if len(wrong) > 0 else "—"

            # TTQ — sort préalable par timestamp indispensable (cf. B2).
            # Base df_qual : TTQ mesure le délai avant la 1ère qualification
            # correcte (non-sentinel), pas seulement le 1er gate_open.
            df_corr = df_qual[df_qual['top1_type'] == expected].sort_values('timestamp')
            if len(df_corr) > 0:
                ttq_s   = (df_corr['timestamp'].iloc[0] - t_start).total_seconds()
                ttq     = ttq_s / 60.0
                # TTQ < WINDOW_MIN : 1ère fenêtre immédiatement correcte (attaque extrême)
                ttq_str = f"<{WINDOW_MIN} min" if ttq < WINDOW_MIN else f"{ttq:.0f} min"
            else:
                ttq, ttq_str = float('nan'), "non qualifié"

            print(f"  {ev['name']:<28} {ev['intensity']:<8} {n_total:>9} "
                  f"{n_detected:>9} {dr*100:>6.1f}%  {qp*100:>7.1f}%  {f1*100:>6.1f}% {fb2*100:>6.1f}%  {wrong_str}")
            ent_part = f"   entropy: {ent_mean:.3f}" if has_entropy and not np.isnan(ent_mean) else ""
            # PATCH-C2 fix (2026-04-20) : affichage conditionnel du seuil binarisé.
            if LR_NOVELTY_THR is None:
                lr_part = f"novelty_lr: {lr_mean:.3f}   [reporting-only, PATCH-C2]"
            else:
                lr_part = (f"novelty_lr: {lr_mean:.3f}   "
                           f"frac>{LR_NOVELTY_THR:.2f}: {lr_frac*100:.1f}%")
            print(f"  {'  ' + u_col + ' moyen':<34}: {u_mean:.3f}   "
                  f"{lr_part}{ent_part}   TTQ: {ttq_str}")

            known_rows.append({
                'attack': ev['name'], 'expected': expected,
                'intensity': ev['intensity'],
                'n_total': n_total, 'n_detected': n_detected, 'n_correct': n_correct,
                'dr':   round(dr, 4),
                'qp':   round(qp, 4),
                'f1':   round(f1, 4),
                'f2':   round(fb2, 4),
                'TTQ_min': round(ttq, 1) if not np.isnan(ttq) else None,
                'u_mean':  round(u_mean, 4)   if not np.isnan(u_mean)  else None,
                'lr_mean': round(lr_mean, 4)  if not np.isnan(lr_mean) else None,
                'lr_frac_above_thr': round(lr_frac, 4) if not np.isnan(lr_frac) else None,
                'entropy_mean': round(ent_mean, 4) if not np.isnan(ent_mean) else None,
            })
            known_atk_roc.append(ev)

    print(f"{'─'*W}")

    # ── [A] Résumé — Attaques connues ──────────────────────────────────────
    df_k = pd.DataFrame(known_rows)
    if len(df_k) > 0:
        macro_dr  = df_k['dr'].mean()
        macro_qp  = df_k['qp'].mean()
        macro_f1  = df_k['f1'].mean()
        macro_fb2 = df_k['f2'].mean()
        n_tot    = int(df_k['n_total'].sum())
        n_det    = int(df_k['n_detected'].sum())
        n_cor    = int(df_k['n_correct'].sum())
        micro_dr  = n_det / n_tot if n_tot > 0 else 0.0
        micro_qp  = n_cor / n_det if n_det > 0 else float('nan')
        micro_f1  = (2 * micro_dr * micro_qp / (micro_dr + micro_qp)
                     if (micro_dr + micro_qp) > 0 else 0.0)
        micro_fb2 = ((1 + FBETA**2) * micro_dr * micro_qp
                     / (FBETA**2 * micro_dr + micro_qp)
                     if (FBETA**2 * micro_dr + micro_qp) > 0 else 0.0) if not np.isnan(micro_qp) else 0.0

        print(f"\n  [A] ATTAQUES CONNUES — {len(known_rows)} types / {n_tot} fenêtres")
        print(f"\n  {'Métrique':<30}  {'Macro':>8}  {'Micro':>8}")
        print(f"  {'':─<50}")
        print(f"  {'DR  (Detection Rate)':<30}  {macro_dr*100:>7.1f}%  {micro_dr*100:>7.1f}%")
        print(f"  {'QP  (Qualif. Precision)':<30}  {macro_qp*100:>7.1f}%  {micro_qp*100:>7.1f}%")
        print(f"  {'F1  (harmon. DR/QP)':<30}  {macro_f1*100:>7.1f}%  {micro_f1*100:>7.1f}%")
        print(f"  {'F2  (β=2, DR prioritaire)':<30}  {macro_fb2*100:>7.1f}%  {micro_fb2*100:>7.1f}%")
        print(f"\n  Macro = poids égaux | Micro = pondéré par n_fenêtres")
        print(f"  F2 (β=2) : pénalise 4× plus les FN que F1 — standard IDS (Tavallaee 2009)")

        print(f"\n  TTQ (Time-to-Qualify) par attaque :")
        for r in known_rows:
            if r['TTQ_min'] is not None:
                # Formater TTQ < WINDOW_MIN identiquement au tableau ligne par ligne
                ttq_d = (f"<{WINDOW_MIN} min  (1ère fenêtre correcte)"
                         if r['TTQ_min'] < WINDOW_MIN
                         else f"{r['TTQ_min']:.0f} min")
            elif r['n_detected'] == 0:
                ttq_d = "non qualifié  (DR = 0 % — attaque non détectée)"
            else:
                ttq_d = "non qualifié  (QP = 0 % — type mal classifié)"
            print(f"    {r['attack']:<28}: {ttq_d}")

    # ── [B] Résumé — Contrôle nouveauté ───────────────────────────────────
    # PATCH-C2 fix (2026-04-20) : affichage sûr quand LR_NOVELTY_THR=None.
    if len(novelty_rows) > 0:
        print(f"\n  [B] CONTRÔLE NOUVEAUTÉ — {len(novelty_rows)} attaque(s) hors-catalogue")
        if LR_NOVELTY_THR is None:
            print(f"  ℹ️  PATCH-C2 : LR_NOVELTY_THR = None (reporting-only).")
            print(f"      Le seuil test-derived a été retiré ; rapport AUC-ROC uniquement (cf. [C]).")
        else:
            print(f"  ⚠️  Seuil LR_NOVELTY_THR = {LR_NOVELTY_THR:.2f} calibré sur signatures")
            print(f"      théoriques parfaites (non empirique). Voir [C] pour calibration.")
        for r in novelty_rows:
            lr_s   = f"{r['lr_mean']:.3f}" if r['lr_mean'] is not None else "nan"
            ent_s  = (f"  entropy={r['entropy_mean']:.3f}" if r['entropy_mean'] is not None else "")
            if LR_NOVELTY_THR is None:
                signal_str = "— (reporting-only)"
                frc_info   = ""
            else:
                frc_s      = (f"{r['lr_frac_above_thr']*100:.1f}%"
                              if r['lr_frac_above_thr'] is not None else "nan")
                frc_info   = f"frac>{LR_NOVELTY_THR:.2f}: {frc_s}  "
                signal_str = f"signal={'✓' if r['novelty_detected'] else '✗'}"
            print(f"      {r['attack']}: DR={r['dr']*100:.1f}%  "
                  f"lr_mean={lr_s}  {frc_info}{ent_s}  {signal_str}")
            print(f"      Top-1 attribué (informatif) : {r['top1_dist']}")

    # ── [C] AUC-ROC novelty_lr ─────────────────────────────────────────────
    print(f"\n  [C] AUC-ROC novelty_lr — connu(0) vs inconnu(1) — INFORMATION ONLY")
    print(f"      ⚠️  Calculé en-échantillon. Ne pas utiliser pour sélectionner le seuil.")
    print(f"          Ref : Hanley & McNeil (1982) ; Youden (1950)")
    roc = _compute_novelty_auc(df, known_atk_roc, novelty_atk_roc, lr_col)
    if roc:
        auc = roc['auc']
        yth = roc['youden_thr']
        print(f"      AUC = {auc:.3f}  |  {roc['n_known_windows']} fen. connues  |  "
              f"{roc['n_novel_windows']} fen. inconnues")
        if LR_NOVELTY_THR is None:
            print(f"      Seuil de Youden (in-sample) : {yth:.3f}  "
                  f"[PATCH-C2 : seuil courant = None, reporting-only]")
        else:
            print(f"      Seuil de Youden (in-sample) : {yth:.3f}  "
                  f"vs seuil actuel : {LR_NOVELTY_THR:.2f}")
            if not np.isnan(yth) and yth < LR_NOVELTY_THR - 0.05:
                print(f"      → Seuil actuel ({LR_NOVELTY_THR:.2f}) probablement trop élevé.")
                print(f"        Recalibration recommandée sur un jeu d'injection de validation.")
    else:
        print(f"      Calcul impossible (données insuffisantes ou classes absentes).")

    # ── [D] Métriques globales de détection (FAR + MCC) ───────────────────
    print(f"\n  [D] MÉTRIQUES GLOBALES DE DÉTECTION (sur {len(df)} fenêtres totales)")
    print(f"      Fenêtres d'attaque = périodes injection attaques + REAL_ATTACKS non-outage")
    print(f"      Fenêtres d'outage  = NETWORK_OUTAGE (REAL_ATTACKS/REAL_ATTACK_CATALOG)  — séparées m-04/F21")
    print(f"      Fenêtres normales  = tout le reste (base FAR/MCC)")
    # PATCH m-04/F21 : on passe les attaques et outages séparément. Le module
    # _compute_global_detection_stats consolide REAL_ATTACKS + REAL_ATTACK_CATALOG
    # via is_outage_event.
    gds = _compute_global_detection_stats(df, attack_events, outage_events)
    n_atk_w = gds['n_attack_windows']
    n_nrm_w = gds['n_normal_windows']
    n_out_w = gds.get('n_outage_windows', 0)
    TP, FP, FN, TN = gds['TP'], gds['FP'], gds['FN'], gds['TN']
    FAR = gds['FAR']
    MCC = gds['MCC']
    print(f"      {'Fenêtres attaque':<30}: {n_atk_w:>6}  (TP={TP}, FN={FN})")
    print(f"      {'Fenêtres outage':<30}: {n_out_w:>6}  "
          f"(gate_open={gds.get('gate_open_during_outage', 0)})")
    print(f"      {'Fenêtres normales':<30}: {n_nrm_w:>6}  (FP={FP}, TN={TN})")
    print(f"      {'FAR (False Alarm Rate)':<30}: {FAR*100:.3f}%  ({FP} fausses alarmes)")
    print(f"      {'MCC (Matthews Corr.)':<30}: {MCC:+.3f}  [-1..+1], +1 = parfait")
    if n_out_w > 0 and gds.get('outage_gate_rate') is not None:
        print(f"      {'Gate rate pendant outage':<30}: {gds['outage_gate_rate']*100:.1f}%  "
              f"(bucket séparé — m-04/F21)")
    if FAR > 0.01:
        print(f"      ⚠️  FAR > 1% — vérifier le seuil de gate ou les seuils de détection")
    if MCC < 0.8:
        print(f"      ⚠️  MCC < 0.8 — performance globale à améliorer")

    return known_rows, novelty_rows, gds


# ──────────────────────────────────────────────────────────────────────────────
# ÉVALUATION — ATTAQUES RÉELLES
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_real(df: pd.DataFrame, fmt: str) -> list:
    """
    Évalue la qualification sur les attaques réelles annotées du dataset.
    Mêmes métriques que evaluate_injected() pour permettre la comparaison directe.

    PATCH m-03/F20 + m-04/F21 (2026-04-23)
    --------------------------------------
    * Statistiques top-1 / lr / entropy restreintes à ``qualified_mask``.
    * Les événements NETWORK_OUTAGE sont traités séparément : leur ligne dans
      ``real_rows`` porte un marqueur ``is_outage=True`` et le résumé macro est
      calculé sur les seules attaques (les outages ont leur propre sous-bloc).
    """
    _, u_col    = _get_col_prefix(fmt)
    lr_col      = 'novelty_lr' if 'novelty_lr' in df.columns else u_col
    has_entropy = 'novelty_entropy' in df.columns
    real_rows   = []

    for attack_name, events in REAL_ATTACKS.items():
        for ev in events:
            t_start    = pd.Timestamp(ev['start'])
            t_end      = pd.Timestamp(ev['end'])
            expected   = ev.get('expected_qualif', '')
            duration_h = (t_end - t_start).total_seconds() / 3600

            # PATCH m-04/F21 : décider si cet événement est un outage pour le
            # marquer dans real_rows (utilisé pour le résumé macro séparé).
            ev_is_outage = is_outage_event({
                'type': attack_name,
                'expected_qualif': expected,
                'name': ev.get('name', attack_name),
            })

            mask  = (df['timestamp'] >= t_start) & (df['timestamp'] < t_end)
            df_ev = df[mask].copy()

            n_total     = len(df_ev)
            n_detected  = int(df_ev['gate_open'].sum())
            n_qualified = int(qualified_mask(df_ev).sum())
            # PATCH m-03/F20 : n_correct doit se restreindre à df_qual pour ne
            # pas compter les lignes no_groups (top1_type=sentinel).
            df_qual     = df_ev[qualified_mask(df_ev)]
            n_correct   = int((df_qual['top1_type'] == expected).sum())

            print(f"\n{'='*64}")
            bucket_tag = " [OUTAGE bucket séparé — m-04/F21]" if ev_is_outage else ""
            print(f"  Attaque réelle : {attack_name}{bucket_tag}  ({ev.get('reason', '')})")
            print(f"  Période : {t_start} → {t_end}  ({duration_h:.1f} h)")
            print(f"  Type attendu   : {expected}")
            print(f"{'='*64}")
            print(f"  Fenêtres dans la période : {n_total}")

            if n_total == 0:
                print(f"  ⚠️  Aucune fenêtre — période hors dataset actif.")
                continue

            dr = n_detected / n_total
            print(f"  DR (Detection Rate)    : {n_detected}/{n_total}  ({dr*100:.1f}%)")

            if n_detected > 0:
                # QP base : qualifiées (m-03/F20). n_correct/n_qualified.
                qp = n_correct / n_qualified if n_qualified > 0 else float('nan')
                f1 = 2 * dr * qp / (dr + qp) if (n_qualified > 0 and (dr + qp) > 0) else 0.0
                print(f"  QP (Qualif. Precision) : {n_correct}/{n_qualified}  "
                      f"({qp*100:.1f}%)" if n_qualified > 0
                      else f"  QP (Qualif. Precision) : n/a (0 fenêtre qualifiée ; no_groups)")
                if n_qualified > 0:
                    print(f"  F1                     : {f1*100:.1f}%")
                print(f"  Fenêtres qualifiées    : {n_qualified}/{n_detected}  "
                      f"(diff = no_groups, m-03/F20)")

                print(f"\n  Distribution top-1 (sur fenêtres qualifiées) :")
                for t, c in df_qual['top1_type'].value_counts().items():
                    marker = " ✓" if t == expected else ""
                    total_base = max(n_qualified, 1)
                    print(f"    {t:28s}: {c:4d}  ({100*c/total_base:.1f}%){marker}")

                # TTQ — base qualifiée (m-03/F20), sort préalable indispensable.
                df_corr = (df_qual[df_qual['top1_type'] == expected]
                           .sort_values('timestamp'))
                if len(df_corr) > 0:
                    ttq_s   = (df_corr['timestamp'].iloc[0] - t_start).total_seconds()
                    ttq     = ttq_s / 60.0
                    ttq_str = f"<{WINDOW_MIN} min" if ttq < WINDOW_MIN else f"{ttq:.1f} min"
                    print(f"\n  TTQ (Time-to-Qualify)  : {ttq_str}")
                else:
                    ttq = float('nan')
                    print(f"\n  TTQ : jamais qualifié correctement")

                # lr/entropy : base qualifiée (m-03/F20).
                lr_mean  = df_qual[lr_col].mean() if n_qualified > 0 else float('nan')
                # PATCH-C2 fix (2026-04-20) : None-safe.
                if LR_NOVELTY_THR is not None and n_qualified > 0:
                    lr_frac  = float((df_qual[lr_col] > LR_NOVELTY_THR).mean())
                else:
                    lr_frac  = float('nan')
                ent_mean = df_qual['novelty_entropy'].mean() if (has_entropy and n_qualified > 0) else float('nan')
                if LR_NOVELTY_THR is None:
                    print(f"\n  novelty_lr moyen  : {lr_mean:.3f}  "
                          f"[PATCH-C2 : seuil binarisé retiré, reporting-only]")
                else:
                    print(f"\n  novelty_lr moyen  : {lr_mean:.3f}  "
                          f"frac>{LR_NOVELTY_THR:.2f}: {lr_frac*100:.1f}%  "
                          f"({'signal nouveauté' if not np.isnan(lr_mean) and lr_mean > LR_NOVELTY_THR else 'pas de nouveauté'})")
                if has_entropy and not np.isnan(ent_mean):
                    print(f"  novelty_entropy   : {ent_mean:.3f}")

                real_rows.append({
                    'attack': attack_name, 'expected': expected,
                    'is_outage': ev_is_outage,
                    'duration_h': round(duration_h, 1),
                    'n_total': n_total, 'n_detected': n_detected,
                    'n_qualified': n_qualified, 'n_correct': n_correct,
                    'dr': round(dr, 4),
                    'qp': round(qp, 4) if not np.isnan(qp) else None,
                    'f1': round(f1, 4),
                    'TTQ_min': round(ttq, 1) if not np.isnan(ttq) else None,
                    'lr_mean': round(lr_mean, 4) if not np.isnan(lr_mean) else None,
                    'lr_frac_above_thr': round(lr_frac, 4) if not np.isnan(lr_frac) else None,
                    'entropy_mean': round(ent_mean, 4) if not np.isnan(ent_mean) else None,
                })
            else:
                real_rows.append({
                    'attack': attack_name, 'expected': expected,
                    'is_outage': ev_is_outage,
                    'duration_h': round(duration_h, 1),
                    'n_total': n_total, 'n_detected': 0,
                    'n_qualified': 0, 'n_correct': 0,
                    'dr': 0.0, 'qp': None, 'f1': 0.0,
                    'TTQ_min': None, 'lr_mean': None,
                    'lr_frac_above_thr': None, 'entropy_mean': None,
                })

    # Résumé global — m-04/F21 : séparer attaques et outages
    df_r   = pd.DataFrame(real_rows)
    df_r_v = df_r[df_r['n_total'] > 0] if len(df_r) else df_r
    if len(df_r_v) > 0:
        df_atk_v = df_r_v[~df_r_v.get('is_outage', False)]
        df_out_v = df_r_v[df_r_v.get('is_outage', False)]

        if len(df_atk_v) > 0:
            print(f"\n{'='*64}")
            print(f"  RÉSUMÉ ATTAQUES RÉELLES — {len(df_atk_v)} événements (hors outages)")
            print(f"  Macro-DR : {df_atk_v['dr'].mean()*100:.1f}%  |  "
                  f"Macro-QP : {df_atk_v['qp'].dropna().mean()*100:.1f}%  |  "
                  f"Macro-F1 : {df_atk_v['f1'].mean()*100:.1f}%")
        if len(df_out_v) > 0:
            print(f"\n{'='*64}")
            print(f"  RÉSUMÉ OUTAGES RÉELS — {len(df_out_v)} événements (bucket séparé m-04/F21)")
            print(f"  Macro-DR : {df_out_v['dr'].mean()*100:.1f}%  |  "
                  f"Macro-QP : {df_out_v['qp'].dropna().mean()*100:.1f}%  |  "
                  f"Macro-F1 : {df_out_v['f1'].mean()*100:.1f}%")
            print(f"  Note : QP/F1 sur outages mesurent la reconnaissance du type "
                  f"NETWORK_OUTAGE — ≠ détection d'attaque adversariale.")

    return real_rows


# ──────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE DES RÉSULTATS
# ──────────────────────────────────────────────────────────────────────────────

def _save_results(
        known_rows:   list,
        novelty_rows: list,
        real_rows:    list,
        output_dir:   str,
        csv_source:   str,
        global_stats: dict | None = None,
) -> None:
    """
    Sauvegarde CSV + JSON datestampés.

    Fichiers produits :
      eval_qualify_{base}_{ts}.csv          — métriques par attaque (known + real)
      eval_qualify_summary_{base}_{ts}.json — résumé global + seuils utilisés
    """
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = os.path.splitext(os.path.basename(csv_source))[0]

    all_rows = known_rows + real_rows
    if all_rows:
        out_csv = os.path.join(output_dir, f"eval_qualify_{base}_{ts}.csv")
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        print(f"\n  → Résultats métriques : {out_csv}")

    summary: dict = {
        'timestamp': ts,
        'source_csv': csv_source,
        'LR_NOVELTY_THR': LR_NOVELTY_THR,
    }
    if known_rows:
        df_k  = pd.DataFrame(known_rows)
        n_det = int(df_k['n_detected'].sum())
        n_cor = int(df_k['n_correct'].sum())
        micro_dr  = float(n_det / max(int(df_k['n_total'].sum()), 1))
        micro_qp  = float(n_cor / max(n_det, 1))
        micro_fb2 = (float((1 + FBETA**2) * micro_dr * micro_qp
                           / (FBETA**2 * micro_dr + micro_qp))
                     if (FBETA**2 * micro_dr + micro_qp) > 0 else 0.0)
        summary.update({
            'n_attack_types_injected': len(known_rows),
            'macro_DR':  round(float(df_k['dr'].mean()), 4),
            'macro_QP':  round(float(df_k['qp'].mean()), 4),
            'macro_F1':  round(float(df_k['f1'].mean()), 4),
            'macro_F2':  round(float(df_k['f2'].mean()), 4),
            'micro_DR':  round(micro_dr, 4),
            'micro_QP':  round(micro_qp, 4),
            'micro_F2':  round(micro_fb2, 4),
            'attacks_not_qualified': [r['attack'] for r in known_rows if r['TTQ_min'] is None],
            'FBETA': FBETA,
        })
    if global_stats:
        summary['global_detection'] = global_stats
    if novelty_rows:
        summary['novelty_controls'] = novelty_rows
    if real_rows:
        summary['real_events'] = real_rows

    out_json = os.path.join(output_dir, f"eval_qualify_summary_{base}_{ts}.json")
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    print(f"  → Résumé JSON          : {out_json}")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Évaluation qualification SBN / LR heuristique',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--csv', default=None,
        help='Chemin explicite vers le CSV de qualify_anomaly_sbn.py')
    parser.add_argument(
        '--balance_ratio', default=None,
        help='Suffixe variante balance_ratio dans le nom de fichier (ex: "auto")')
    parser.add_argument(
        '--injected', action='store_true', default=False,
        help='Évalue les attaques injectées synthétiquement (défaut si aucun flag)')
    parser.add_argument(
        '--real', action='store_true', default=False,
        help='Évalue les attaques réelles (depuis REAL_ATTACKS de config.py)')
    parser.add_argument(
        '--both', action='store_true', default=False,
        help='Évalue injected + real en un seul passage')
    parser.add_argument(
        '--no_save', action='store_true', default=False,
        help='Désactive la sauvegarde des résultats sur disque')
    args = parser.parse_args()

    # ── Résolution du CSV ──────────────────────────────────────────────────
    # PATCH TASK-25 (audit_tmp MAJ-04, 2026-04-26)
    # ──────────────────────────────────────────────────────────────────────
    # Le hardcoded fallback vers `resultats_trained_models_v9_v9_v4s_v3_v3`
    # (avec décalage de version) a été supprimé : il pouvait silencieusement
    # évaluer un *autre* run que celui de la config courante, faussant les
    # comparaisons inter-versions et masquant des erreurs de configuration.
    # Désormais, si le CSV n'est pas dans le RESULTS_DIR canonique
    # (résolu via paths.get_results_dir(CONFIG)), on lève FileNotFoundError
    # avec un message explicite. Pour autoriser explicitement un fallback
    # sur un autre run, l'utilisateur doit passer `--csv <chemin>`.
    if args.csv:
        csv_path = args.csv
    else:
        _bal       = f'_balance_{args.balance_ratio}' if args.balance_ratio else ''
        candidates = [
            os.path.join(RESULTS_DIR, f'qualif_types_sbn{_bal}.csv'),
            os.path.join(RESULTS_DIR, f'qualif_types{_bal}.csv'),
        ]
        csv_path = None
        for c in candidates:
            if os.path.exists(c):
                csv_path = c
                break
        if csv_path is None:
            raise FileNotFoundError(
                f"[MAJ-04] Aucun CSV de qualification trouvé dans le "
                f"RESULTS_DIR canonique :\n"
                f"   {RESULTS_DIR}\n"
                f"  Candidats inspectés :\n"
                + "\n".join(f"   - {c}" for c in candidates) + "\n"
                f"  Solutions possibles :\n"
                f"   1) Lancer `python qualify_anomaly_sbn.py` pour produire "
                f"le CSV manquant.\n"
                f"   2) Passer explicitement `--csv <chemin>` pour pointer "
                f"vers un autre run (le hardcoded fallback vers "
                f"`resultats_trained_models_v9_v9_v4s_v3_v3` a été supprimé "
                f"car il masquait des erreurs de configuration silencieuses)."
            )

    print(f"→ Lecture : {csv_path}")
    df = pd.read_csv(csv_path)

    # Bug 1 : gate_open peut être lu comme string "True"/"False" par pandas
    # → cast explicite en bool pour garantir le comportement attendu
    if 'gate_open' in df.columns:
        df['gate_open'] = df['gate_open'].map(
            lambda x: x if isinstance(x, bool) else str(x).strip().lower() == 'true'
        ).astype(bool)

    # Bug 2 : timestamps — normaliser en tz-naive pour comparaison avec INJECTED_ATTACKS
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=False)
    if hasattr(df['timestamp'].dt, 'tz') and df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    _validate_columns(df)
    print(f"  [DEBUG] RESULTS_DIR = {RESULTS_DIR}")
    print(f"  [DEBUG] gate_open dtype = {df['gate_open'].dtype}  "
          f"  True={df['gate_open'].sum()}  False={(~df['gate_open']).sum()}")

    fmt    = _detect_format(df)
    prefix = _get_col_prefix(fmt)[0]
    df     = _add_novelty_entropy(df, prefix)

    print(f"  Format détecté : {fmt.upper()}  ({len(df)} fenêtres)")

    known_rows   = []
    novelty_rows = []
    real_rows    = []
    global_stats = None

    run_injected = args.injected or args.both or (not args.real and not args.both)
    run_real     = args.real or args.both

    if run_injected:
        known_rows, novelty_rows, global_stats = evaluate_injected(df, fmt)
    if run_real:
        real_rows = evaluate_real(df, fmt)

    if not args.no_save and (known_rows or real_rows):
        _save_results(known_rows, novelty_rows, real_rows,
                      RESULTS_DIR, csv_path, global_stats)
