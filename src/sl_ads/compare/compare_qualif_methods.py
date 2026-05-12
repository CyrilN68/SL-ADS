"""
compare_qualif_methods.py — Comparaison exhaustive LR vs SBN
=============================================================
Compare les deux méthodes de qualification du type d'anomalie :

  (A) LR heuristique  : qualify_anomaly.py    → colonnes b_qualif_*, u_qualif
  (B) SBN rigoreux    : qualify_anomaly_sbn.py → colonnes b_sbn_*,    u_sbn

Axes de comparaison (8 sections) :
──────────────────────────────────
  §1  Gate-level agreement          — Cohen's κ, matrice de confusion gate
  §2  Métriques opérationnelles     — Recall, Précision, TTQ par attaque × méthode
  §3  Accord de classification       — % top-1 identique, κ (quand les deux gates)
  §4  Analyse des désaccords         — tableau croisé des divergences type→type
  §5  Incertitude                    — u_qualif vs u_sbn : mean, IQR, distribut.
  §6  Signal de nouveauté            — UNKNOWN_ANOMALY_CONTROL : u vs novelty_lr
  §7  Confiance top-1               — top1_b : mean, std, fraction confident
  §8  Stabilité temporelle           — flipping rate par attaque (fenêtres où top1 change)
  §§  Contrainte SL (sanity check)  — Σb + u ≈ 1.0 pour les deux méthodes

Usage :
    python compare_qualif_methods.py                          # auto-détection des CSV
    python compare_qualif_methods.py --lr_csv path/lr.csv --sbn_csv path/sbn.csv
    python compare_qualif_methods.py --real                   # attaques réelles
    python compare_qualif_methods.py --out_csv comparison.csv # exporte le merge

Tous les seuils proviennent de config.py — ne les modifiez pas ici.

Ref :
  Cohen (1960)         — κ comme accord au-delà du hasard
  Jøsang (2016) §3.6   — Uncertainty Maximisation
  Neyman-Pearson (1933) — LR comme statistique suffisante
  Sharafaldin et al. (2018) CIC-IDS2017 — séparation détection/classification
"""

import argparse
import os
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
from sl_ads.config import INJECTED_ATTACK_CATALOG as INJECTED_ATTACKS  # Phase H

# ── PATCH m-09/F30 + m-04/F21 ──────────────────────────────────────────────
# Shared qualification-filter helpers.  See qualif_filters.py for the rationale
# (no_groups sentinel rows must be excluded from statistics over u / novelty_lr
# / top1_type / entropy, and NETWORK_OUTAGE events must be partitioned from
# attack events in per-attack aggregates).
from sl_ads.qualif_filters import (  # Phase H
    qualified_mask,
    partition_catalog_by_outage,
    is_outage_event,
    QUAL_STATUS_NO_GROUPS,
)


def _qualified_side(merged: pd.DataFrame, side: str) -> pd.Series:
    """
    Return a boolean mask selecting rows where the ``side`` ('lr' or 'sbn')
    actually qualified — i.e. gate_open_{side}=True AND
    qual_status_{side} != 'no_groups'.

    If the qual_status_{side} column is missing (e.g. legacy LR CSV produced
    by an old qualify_anomaly.py that did not expose the column), this falls
    back to gate_open_{side} alone (mirrors ``qualified_mask`` semantics).
    """
    gate_col = f"gate_open_{side}"
    qs_col   = f"qual_status_{side}"
    if gate_col not in merged.columns:
        return pd.Series(True, index=merged.index)
    gate = merged[gate_col].fillna(False).astype(bool)
    if qs_col not in merged.columns:
        return gate
    return gate & (merged[qs_col].astype(str) != QUAL_STATUS_NO_GROUPS)


warnings.filterwarnings('ignore', category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — tous les seuils viennent de config.py
# ──────────────────────────────────────────────────────────────────────────────
try:
    from sl_ads.config import CONFIG, REAL_ATTACKS  # Phase H
    from sl_ads.paths import (  # Phase H
        get_results_dir,
        get_decision_threshold,
        get_detection_col,
    )
    RESULTS_DIR     = get_results_dir(CONFIG, up_levels=1)
    WINDOW_MIN      = CONFIG.get('WINDOW_MINUTES', 5)
    # PATCH TASK-24 (audit_tmp MAJ-03, 2026-04-26)
    # ──────────────────────────────────────────────────────────────────────
    # Le seuil opérationnel doit provenir du sidecar JSON écrit par
    # train_v10.py après auto-calibration EVT/FPR (voir paths.py:44-64).
    # L'ancien hard-coded fallback ``CONFIG.get('DECISION_THRESHOLD', 0.20)``
    # bypassait cette résolution canonique et pouvait diverger d'au moins
    # 2 ordres de grandeur sur certaines configurations (cf. audit
    # SCIENTIFIC_AUDIT_REPORT.md MAJ-03 ligne 90).
    GATE_THRESHOLD  = get_decision_threshold(CONFIG, up_levels=1)
    DETECTION_COL   = get_detection_col(CONFIG, up_levels=1)
    # Seuils de nouveauté (peuvent différer entre les deux méthodes)
    LR_NOVELTY_THR  = CONFIG.get('QUALIFY_UM_NOVELTY_THRESHOLD',
                                  CONFIG.get('QUALIFY_GATE_THRESHOLD', 0.40))
    SBN_NOVELTY_THR = CONFIG.get('SBN_NOVELTY_THRESHOLD',
                                  CONFIG.get('SBN_LR_NOVELTY_THRESHOLD', 0.65))
    # Seuil de confiance top-1 pour "classification confiante"
    CONF_THRESHOLD  = CONFIG.get('COMPARE_CONF_THRESHOLD', 0.30)
except ImportError:
    print("[WARN] config.py non trouvé — valeurs de fallback utilisées.")
    RESULTS_DIR     = '.'
    WINDOW_MIN      = 5
    GATE_THRESHOLD  = 0.20
    DETECTION_COL   = "FINAL_SYSTEM_CBF_b_atk"
    LR_NOVELTY_THR  = 0.65
    SBN_NOVELTY_THR = 0.65
    CONF_THRESHOLD  = 0.30
    REAL_ATTACKS    = {}

# ──────────────────────────────────────────────────────────────────────────────
# CATALOGUE ATTAQUES (source unique : config.INJECTED_ATTACK_CATALOG)
# ──────────────────────────────────────────────────────────────────────────────
# PATCH-C1 fix (2026-04-19) : le hard-code qui shadow-ait l'import ligne 44 a
# été supprimé. On ne définit plus INJECTED_ATTACKS localement — on réutilise
# la liste importée (identique à celle qu'utilisent evaluate_qualify_injected.py,
# evaluate_qualify_sbn.py et inject_at_evidence_level.py).
# Le catalogue source (config.INJECTED_ATTACK_CATALOG) utilise le nom
# 'UNKNOWN_ANOMALY_CONTROL' (aligné sur inject_at_evidence_level.py et sur
# les évaluateurs downstream). L'ancien hard-code local avait seulement
# 9 attaques et dupliquait les timestamps — source de divergences silencieuses.

KNOWN_ATTACKS   = [ev for ev in INJECTED_ATTACKS if not ev.get('is_novelty_control', False)]
NOVELTY_ATTACKS = [ev for ev in INJECTED_ATTACKS if ev.get('is_novelty_control', False)]
# PATCH m-04/F21 : partition NETWORK_OUTAGE out of attack tables so they don't
# pollute the macro QP / TTQ / flipping-rate aggregates (they have a distinct
# taxonomy and a separate bucket in evaluate_qualify_sbn.py/evaluate_real).
_ATTACK_EVENTS_NO_OUTAGE, _OUTAGE_EVENTS = partition_catalog_by_outage(INJECTED_ATTACKS)


# ──────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ──────────────────────────────────────────────────────────────────────────────

def _hdr(title: str, width: int = 110):
    """Affiche un en-tête de section."""
    print(f"\n{'═'*width}")
    print(f"  {title}")
    print(f"{'═'*width}")


def _subhdr(title: str, width: int = 80):
    print(f"\n  ── {title} {'─'*(width - len(title) - 6)}")


def _cohen_kappa(y1: pd.Series, y2: pd.Series) -> float:
    """
    Cohen's κ (Cohen 1960) : accord au-delà du hasard entre deux classificateurs.

    κ = (p_o - p_e) / (1 - p_e)
      p_o = fréquence d'accord observé
      p_e = fréquence d'accord attendue par hasard

    Interprétation : < 0.20 mauvais, 0.21-0.40 passable, 0.41-0.60 modéré,
                     0.61-0.80 bon, > 0.80 quasi-parfait (Landis & Koch 1977).
    """
    # Normalisation : convertir en str pour uniformiser les NaN
    s1 = y1.fillna('').astype(str)
    s2 = y2.fillna('').astype(str)
    labels = sorted(set(s1) | set(s2))

    total = len(s1)
    if total == 0:
        return float('nan')

    # Accord observé
    p_o = (s1 == s2).sum() / total

    # Accord attendu par hasard
    p_e = sum(
        (s1 == lbl).sum() / total * (s2 == lbl).sum() / total
        for lbl in labels
    )

    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def _iqr(s: pd.Series) -> tuple[float, float, float]:
    """Retourne (Q1, médiane, Q3) d'une série."""
    q = s.dropna().quantile([0.25, 0.50, 0.75]).values
    return float(q[0]), float(q[1]), float(q[2])


def _flipping_rate(df_ev: pd.DataFrame, col: str) -> float:
    """
    Fréquence à laquelle le top-1 change entre deux fenêtres consécutives.

    Mesure la stabilité temporelle du classificateur (instabilité ↔ label flipping).
    Ref : stabilité des prédictions, Breiman (1996) Bagging Predictors.
    """
    seq = df_ev[col].dropna()
    if len(seq) < 2:
        return float('nan')
    changes = (seq.values[1:] != seq.values[:-1]).sum()
    return changes / (len(seq) - 1)


def _ttq(df_ev: pd.DataFrame, t_start: pd.Timestamp, gate_col: str,
         top1_col: str, expected: str) -> float | None:
    """Time-to-qualify : délai avant la première fenêtre correctement qualifiée."""
    df_c = df_ev[df_ev[gate_col] & (df_ev[top1_col] == expected)]
    if len(df_c) == 0:
        return None
    return (df_c['timestamp'].iloc[0] - t_start).total_seconds() / 60


# ──────────────────────────────────────────────────────────────────────────────
# §1 — GATE-LEVEL AGREEMENT
# ──────────────────────────────────────────────────────────────────────────────

def section_gate_agreement(merged: pd.DataFrame):
    """
    Compare l'accord gate open/close entre les deux méthodes.

    Une gate ouverte par une méthode et fermée par l'autre signifie que les
    deux méthodes ne s'accordent pas sur le fait qu'une anomalie est détectable.
    Le κ de Cohen (1960) quantifie l'accord au-delà du hasard.
    """
    _hdr("§1  GATE-LEVEL AGREEMENT")

    g_lr  = merged['gate_open_lr'].fillna(False)
    g_sbn = merged['gate_open_sbn'].fillna(False)

    both_open   = (g_lr & g_sbn).sum()
    lr_only     = (g_lr & ~g_sbn).sum()
    sbn_only    = (~g_lr & g_sbn).sum()
    both_closed = (~g_lr & ~g_sbn).sum()
    total       = len(merged)

    kappa = _cohen_kappa(g_lr.astype(str), g_sbn.astype(str))

    print(f"\n  Fenêtres totales : {total:,}")
    print(f"\n  Matrice de confusion gate :")
    print(f"  {'':>20}  {'SBN=Open':>10}  {'SBN=Closed':>10}")
    print(f"  {'LR=Open':>20}  {both_open:>10,}  {lr_only:>10,}")
    print(f"  {'LR=Closed':>20}  {sbn_only:>10,}  {both_closed:>10,}")

    p_agree = (both_open + both_closed) / total
    print(f"\n  Accord global     : {p_agree*100:.1f}%")
    print(f"  Cohen's κ (gate)  : {kappa:.3f}  "
          f"({'quasi-parfait' if kappa > 0.80 else 'bon' if kappa > 0.60 else 'modéré' if kappa > 0.40 else 'passable' if kappa > 0.20 else 'mauvais'})")

    print(f"\n  Fenêtres gate_open :")
    print(f"    LR seule   (SBN fermée) : {lr_only:5,}  ({100*lr_only/total:.1f}%)")
    print(f"    SBN seule  (LR fermée)  : {sbn_only:5,}  ({100*sbn_only/total:.1f}%)")
    print(f"    Les deux ouvertes        : {both_open:5,}  ({100*both_open/total:.1f}%)")
    print(f"    Les deux fermées         : {both_closed:5,}  ({100*both_closed/total:.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# §2 — MÉTRIQUES OPÉRATIONNELLES PAR ATTAQUE
# ──────────────────────────────────────────────────────────────────────────────

def section_operational_metrics(merged: pd.DataFrame, attacks: list) -> dict:
    """
    Recall, Précision, TTQ par attaque × méthode.

    Recall    = n_détectées / n_fenêtres dans la période
    Précision = n_correctement_qualifiées / n_détectées
    TTQ       = délai avant la première qualification correcte (min)

    Ref : séparation détection/classification — Sharafaldin et al. (2018).
    """
    _hdr("§2  MÉTRIQUES OPÉRATIONNELLES PAR ATTAQUE × MÉTHODE")

    W = 28   # largeur colonne attaque
    print(f"\n  {'Attaque':<{W}} {'Intens':<7} "
          f"{'LR:Rec':>7} {'LR:Prec':>8} {'LR:TTQ':>7} "
          f"{'SBN:Rec':>8} {'SBN:Prec':>9} {'SBN:TTQ':>8}  {'Δ Rec':>6} {'Δ Prec':>7}")
    print(f"  {'─'*105}")

    results = {}
    known_lr_recalls, known_lr_precs = [], []
    known_sbn_recalls, known_sbn_precs = [], []

    for ev in attacks:
        t_start = pd.Timestamp(ev['start'])
        t_end   = pd.Timestamp(ev['end'])
        mask    = (merged['timestamp'] >= t_start) & (merged['timestamp'] <= t_end)
        df_ev   = merged[mask]
        n_total = len(df_ev)

        if n_total == 0:
            print(f"  {ev['name']:<{W}}  ⚠️  hors dataset")
            continue

        if ev['is_novelty_control']:
            # Pour le contrôle, on affiche juste le recall (gate)
            n_lr  = df_ev['gate_open_lr'].sum()
            n_sbn = df_ev['gate_open_sbn'].sum()
            print(f"  {ev['name']:<{W}} {'(' + ev['intensity'] + ')':<7} "
                  f"  [contrôle nouveauté — recall gate LR:{100*n_lr/n_total:.0f}%"
                  f"  SBN:{100*n_sbn/n_total:.0f}%]")
            results[ev['name']] = {'n_total': n_total, 'novelty': True,
                                   'recall_lr': n_lr/n_total, 'recall_sbn': n_sbn/n_total}
            continue

        expected = ev['expected']

        # PATCH m-09/F30 : Precision basée sur les fenêtres qualifiées par
        # méthode (gate_open_{side} AND qual_status_{side} != 'no_groups').
        # Inclure les lignes no_groups biaiserait prec vers 0 (top1_type
        # sentinel ≠ expected).  Recall reste sur gate_open seul (détection).
        q_lr_ev  = _qualified_side(df_ev, 'lr')
        q_sbn_ev = _qualified_side(df_ev, 'sbn')

        # LR
        n_lr      = int(df_ev['gate_open_lr'].sum())
        n_qual_lr = int(q_lr_ev.sum())
        n_ok_lr   = int((q_lr_ev & (df_ev['top1_lr'] == expected)).sum())
        rec_lr    = n_lr / n_total
        prec_lr   = n_ok_lr / n_qual_lr if n_qual_lr > 0 else float('nan')
        # TTQ : 1ère fenêtre QUALIFIÉE correcte (pas seulement 1er gate_open)
        _df_lr_corr = df_ev[q_lr_ev & (df_ev['top1_lr'] == expected)].sort_values('timestamp')
        ttq_lr = ((_df_lr_corr['timestamp'].iloc[0] - t_start).total_seconds() / 60
                  if len(_df_lr_corr) > 0 else None)

        # SBN
        n_sbn      = int(df_ev['gate_open_sbn'].sum())
        n_qual_sbn = int(q_sbn_ev.sum())
        n_ok_sbn   = int((q_sbn_ev & (df_ev['top1_sbn'] == expected)).sum())
        rec_sbn    = n_sbn / n_total
        prec_sbn   = n_ok_sbn / n_qual_sbn if n_qual_sbn > 0 else float('nan')
        _df_sbn_corr = df_ev[q_sbn_ev & (df_ev['top1_sbn'] == expected)].sort_values('timestamp')
        ttq_sbn = ((_df_sbn_corr['timestamp'].iloc[0] - t_start).total_seconds() / 60
                   if len(_df_sbn_corr) > 0 else None)

        d_rec  = rec_sbn - rec_lr
        d_prec = (prec_sbn - prec_lr) if not (np.isnan(prec_lr) or np.isnan(prec_sbn)) else float('nan')

        def _fmt_ttq(v): return f"{v:.0f}m" if v is not None else "—"
        def _fmt_p(v): return f"{v*100:.0f}%" if not np.isnan(v) else "—"

        sign_r = "+" if d_rec > 0 else ""
        sign_p = "+" if not np.isnan(d_prec) and d_prec > 0 else ""

        print(f"  {ev['name']:<{W}} ({ev['intensity']:<6}) "
              f"{rec_lr*100:>6.0f}% {_fmt_p(prec_lr):>8} {_fmt_ttq(ttq_lr):>7} "
              f"{rec_sbn*100:>7.0f}% {_fmt_p(prec_sbn):>8} {_fmt_ttq(ttq_sbn):>8}  "
              f"{sign_r}{d_rec*100:>5.1f}% "
              f"{sign_p if not np.isnan(d_prec) else ''}{f'{d_prec*100:.1f}%' if not np.isnan(d_prec) else '—':>7}")

        results[ev['name']] = {
            'n_total': n_total, 'expected': expected, 'novelty': False,
            'recall_lr': rec_lr, 'prec_lr': prec_lr, 'ttq_lr': ttq_lr,
            'recall_sbn': rec_sbn, 'prec_sbn': prec_sbn, 'ttq_sbn': ttq_sbn,
        }
        known_lr_recalls.append(rec_lr)
        known_lr_precs.append(prec_lr)
        known_sbn_recalls.append(rec_sbn)
        known_sbn_precs.append(prec_sbn)

    print(f"  {'─'*105}")
    if known_lr_recalls:
        mr_lr  = np.nanmean(known_lr_recalls)
        mp_lr  = np.nanmean(known_lr_precs)
        mr_sbn = np.nanmean(known_sbn_recalls)
        mp_sbn = np.nanmean(known_sbn_precs)
        print(f"  {'MACRO MOYENNE (attaques connues)':<{W+8}} "
              f"{mr_lr*100:>6.1f}% {mp_lr*100:>8.1f}%{'':>8}"
              f"{mr_sbn*100:>8.1f}% {mp_sbn*100:>9.1f}%{'':>9}"
              f"  {(mr_sbn-mr_lr)*100:>+6.1f}% {(mp_sbn-mp_lr)*100:>+7.1f}%")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# §3 — ACCORD DE CLASSIFICATION (quand les deux gates sont ouvertes)
# ──────────────────────────────────────────────────────────────────────────────

def section_classification_agreement(merged: pd.DataFrame):
    """
    Sur les fenêtres où les DEUX méthodes ont gate_open, mesure l'accord top-1.

    κ ici mesure si les deux méthodes "voient la même chose" au-delà du hasard.
    Un κ faible suggère des architectures de scoring fondamentalement différentes.
    """
    _hdr("§3  ACCORD DE CLASSIFICATION (qualifié dans les deux — m-09/F30)")

    # PATCH m-09/F30 : restreindre aux fenêtres réellement qualifiées par les
    # deux méthodes. Inclure les no_groups rendrait l'accord artificiellement
    # bas (top1 sentinel ≠ top1 inféré côté opposé).
    q_lr  = _qualified_side(merged, 'lr')
    q_sbn = _qualified_side(merged, 'sbn')
    both  = merged[q_lr & q_sbn].copy()
    n = len(both)
    print(f"\n  Fenêtres qualifiées des deux côtés : {n:,}")
    if n == 0:
        print("  ⚠️  Aucune fenêtre commune — impossible de comparer la classification.")
        return

    n_agree = (both['top1_lr'] == both['top1_sbn']).sum()
    kappa   = _cohen_kappa(both['top1_lr'], both['top1_sbn'])

    print(f"  Accord top-1     : {n_agree}/{n} ({100*n_agree/n:.1f}%)")
    print(f"  Cohen's κ (type) : {kappa:.3f}  "
          f"({'quasi-parfait' if kappa > 0.80 else 'bon' if kappa > 0.60 else 'modéré' if kappa > 0.40 else 'passable' if kappa > 0.20 else 'mauvais'})")

    # Par période d'attaque (NETWORK_OUTAGE exclu — m-04/F21)
    _subhdr("Accord par attaque (outages exclus)")
    print(f"\n  {'Attaque':<30} {'N commun':>9} {'Accord':>8} {'κ':>7}")
    print(f"  {'─'*58}")
    for ev in _ATTACK_EVENTS_NO_OUTAGE:
        if ev['is_novelty_control']:
            continue
        t_s = pd.Timestamp(ev['start'])
        t_e = pd.Timestamp(ev['end'])
        sub = both[(both['timestamp'] >= t_s) & (both['timestamp'] <= t_e)]
        if len(sub) == 0:
            continue
        n_ag = (sub['top1_lr'] == sub['top1_sbn']).sum()
        k_ev = _cohen_kappa(sub['top1_lr'], sub['top1_sbn'])
        print(f"  {ev['name']:<30} {len(sub):>9,} {100*n_ag/len(sub):>7.1f}%  {k_ev:>6.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# §4 — ANALYSE DES DÉSACCORDS
# ──────────────────────────────────────────────────────────────────────────────

def section_disagreement_analysis(merged: pd.DataFrame):
    """
    Sur les fenêtres où les deux gates sont ouvertes mais les top-1 diffèrent,
    affiche le tableau croisé LR→SBN et les 10 premiers exemples.

    Les désaccords récurrents révèlent des biais architecturaux systémiques.
    """
    _hdr("§4  ANALYSE DES DÉSACCORDS DE CLASSIFICATION (qualifié 2 côtés)")

    # PATCH m-09/F30 : fenêtres qualifiées des deux côtés — sinon les
    # désaccords dominants seraient "SBN top1_sentinel vs LR type_réel",
    # ce qui n'est pas un désaccord de modèle mais une absence de qualif.
    q_lr  = _qualified_side(merged, 'lr')
    q_sbn = _qualified_side(merged, 'sbn')
    both  = merged[q_lr & q_sbn].copy()
    disagree = both[both['top1_lr'] != both['top1_sbn']]
    n_d = len(disagree)
    n_b = len(both)

    print(f"\n  Fenêtres communes avec désaccord : {n_d}/{n_b} "
          f"({100*n_d/n_b:.1f}%)" if n_b > 0 else "  Aucune fenêtre commune.")
    if n_d == 0:
        return

    # Tableau croisé LR→SBN (top 5 paires les plus fréquentes)
    _subhdr("Paires de désaccord les plus fréquentes  (LR→SBN)")
    pairs = disagree.groupby(['top1_lr', 'top1_sbn']).size().sort_values(ascending=False)
    print(f"\n  {'LR classifie comme':<28} → {'SBN classifie comme':<28}  {'N':>5}  {'%':>6}")
    print(f"  {'─'*72}")
    for (lr_t, sbn_t), cnt in pairs.head(10).items():
        print(f"  {lr_t:<28} → {sbn_t:<28}  {cnt:>5}  {100*cnt/n_d:>5.1f}%")

    # Exemples temporels
    _subhdr("10 premiers désaccords (chronologique)")
    print(f"\n  {'Timestamp':<22} {'LR top-1':<22} {'SBN top-1':<22} {'Δu (SBN-LR)':>12}")
    print(f"  {'─'*80}")
    for _, r in disagree.head(10).iterrows():
        du = r.get('u_sbn', float('nan')) - r.get('u_qualif', float('nan'))
        print(f"  {str(r['timestamp']):<22} {r['top1_lr']:<22} {r['top1_sbn']:<22} {du:>+12.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# §5 — COMPARAISON DES INCERTITUDES
# ──────────────────────────────────────────────────────────────────────────────

def section_uncertainty(merged: pd.DataFrame, attacks: list):
    """
    Compare u_qualif (LR) et u_sbn (SBN) sur les fenêtres gate_open.

    Une incertitude systématiquement plus élevée pour une méthode peut indiquer :
      - Meilleure honnêteté épistémique (signal de nouveauté amplifié)
      - Ou scoring moins discriminant → croyances plus plates

    Ref : Jøsang (2016) §3.6, Eq. 3.27 — Uncertainty Maximisation.
    L'invariant UM : les probabilités projetées P(xi) sont préservées, donc
    une u plus haute n'implique pas une classification moins précise.
    """
    _hdr("§5  INCERTITUDE : u_qualif (LR) vs u_sbn (SBN)  [qualifié 2 côtés]")

    # PATCH m-09/F30 : u_qualif/u_sbn valent 1.0 (sentinels) pour les lignes
    # no_groups.  Les inclure tirerait les moyennes vers 1 et gonflerait les
    # écarts inter-méthodes artificiellement.  On ne garde que qualifié × 2.
    q_lr  = _qualified_side(merged, 'lr')
    q_sbn = _qualified_side(merged, 'sbn')
    both = merged[q_lr & q_sbn].copy()
    if len(both) == 0:
        print("  Aucune fenêtre commune.")
        return

    def _stats(s):
        q1, med, q3 = _iqr(s)
        return s.mean(), s.std(), q1, med, q3

    u_lr  = both['u_qualif']
    u_sbn = both['u_sbn']

    print(f"\n  {'Statistique':<18} {'LR':>10} {'SBN':>10} {'Δ (SBN−LR)':>12}")
    print(f"  {'─'*52}")
    for label, v_lr, v_sbn in [
        ('Moyenne',  u_lr.mean(),                  u_sbn.mean()),
        ('Std',      u_lr.std(),                   u_sbn.std()),
        ('Q1',       u_lr.quantile(0.25),          u_sbn.quantile(0.25)),
        ('Médiane',  u_lr.quantile(0.50),          u_sbn.quantile(0.50)),
        ('Q3',       u_lr.quantile(0.75),          u_sbn.quantile(0.75)),
        ('Max',      u_lr.max(),                   u_sbn.max()),
    ]:
        delta = v_sbn - v_lr
        print(f"  {label:<18} {v_lr:>10.4f} {v_sbn:>10.4f} {delta:>+12.4f}")

    # Par attaque
    _subhdr("Incertitude moyenne par attaque (gate_open dans les deux)")
    print(f"\n  {'Attaque':<30} {'u_LR':>8} {'u_SBN':>8} {'Δ':>8}")
    print(f"  {'─'*58}")
    for ev in attacks:
        if ev['is_novelty_control']:
            continue
        t_s = pd.Timestamp(ev['start'])
        t_e = pd.Timestamp(ev['end'])
        sub = both[(both['timestamp'] >= t_s) & (both['timestamp'] <= t_e)]
        if len(sub) == 0:
            continue
        mu_lr  = sub['u_qualif'].mean()
        mu_sbn = sub['u_sbn'].mean()
        print(f"  {ev['name']:<30} {mu_lr:>8.4f} {mu_sbn:>8.4f} {mu_sbn-mu_lr:>+8.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# §6 — SIGNAL DE NOUVEAUTÉ (UNKNOWN_ANOMALY_CONTROL)
# ──────────────────────────────────────────────────────────────────────────────

def section_novelty(merged: pd.DataFrame):
    """
    Évalue la détection de nouveauté sur UNKNOWN_ANOMALY_CONTROL.

    Métriques :
      LR  : u_qualif > LR_NOVELTY_THR (post Uncertainty Maximisation)
      SBN : novelty_lr > SBN_NOVELTY_THR  OU  u_sbn > SBN_NOVELTY_THR

    La détection de nouveauté est un sous-problème du reject option
    (Chow 1970, IEEE TIT) : le système doit signaler "je ne sais pas"
    pour les anomalies hors-catalogue.
    """
    _hdr("§6  SIGNAL DE NOUVEAUTÉ — UNKNOWN_ANOMALY_CONTROL")

    ev = next(e for e in INJECTED_ATTACKS if e['is_novelty_control'])
    t_s = pd.Timestamp(ev['start'])
    t_e = pd.Timestamp(ev['end'])
    mask = (merged['timestamp'] >= t_s) & (merged['timestamp'] <= t_e)
    df_ev = merged[mask]
    n = len(df_ev)

    if n == 0:
        print("  ⚠️  Période hors dataset.")
        return

    print(f"\n  Période : {t_s} → {t_e}  ({n} fenêtres)")
    print(f"  Intensité injectée : {ev['intensity']}")

    # Gate
    n_gate_lr  = df_ev['gate_open_lr'].sum()
    n_gate_sbn = df_ev['gate_open_sbn'].sum()
    # PATCH m-09/F30 : les signaux de nouveauté doivent exclure les fenêtres
    # no_groups (u=1 sentinel) — sinon lr_signal/sbn_signal sont automatiquement
    # positifs pour les no_groups et on sur-estime le taux de détection novelty.
    q_lr_ev  = _qualified_side(df_ev, 'lr')
    q_sbn_ev = _qualified_side(df_ev, 'sbn')
    n_qual_lr  = int(q_lr_ev.sum())
    n_qual_sbn = int(q_sbn_ev.sum())
    print(f"\n  Gate ouverte (détection primaire) :")
    print(f"    LR  : {n_gate_lr}/{n} ({100*n_gate_lr/n:.1f}%)  "
          f"[qualifiées : {n_qual_lr}]")
    print(f"    SBN : {n_gate_sbn}/{n} ({100*n_gate_sbn/n:.1f}%)  "
          f"[qualifiées : {n_qual_sbn}]")

    # Signal de nouveauté LR : u_qualif sur qualifiées (m-09/F30)
    u_lr_vals  = df_ev.loc[q_lr_ev,  'u_qualif']
    u_sbn_vals = df_ev.loc[q_sbn_ev, 'u_sbn']
    lr_signal  = (u_lr_vals > LR_NOVELTY_THR).sum()  if len(u_lr_vals) else 0
    sbn_signal = (u_sbn_vals > SBN_NOVELTY_THR).sum() if len(u_sbn_vals) else 0

    print(f"\n  Signal de nouveauté (base qualifié — m-09/F30) :")
    print(f"    LR  — u_qualif  > {LR_NOVELTY_THR:.2f} : "
          f"{lr_signal}/{n_qual_lr if n_qual_lr else 1} fenêtres  "
          f"(u moy = {u_lr_vals.mean():.4f})")
    print(f"    SBN — u_sbn     > {SBN_NOVELTY_THR:.2f} : "
          f"{sbn_signal}/{n_qual_sbn if n_qual_sbn else 1} fenêtres  "
          f"(u moy = {u_sbn_vals.mean():.4f})")

    if 'novelty_lr' in merged.columns:
        nov_lr_vals = df_ev.loc[q_sbn_ev, 'novelty_lr']
        nov_sig     = (nov_lr_vals > SBN_NOVELTY_THR).sum()
        print(f"    SBN — novelty_lr > {SBN_NOVELTY_THR:.2f} : "
              f"{nov_sig}/{n_qual_sbn if n_qual_sbn else 1} fenêtres  "
              f"(nov_lr moy = {nov_lr_vals.mean():.4f})")

    # Verdict
    print(f"\n  Verdict :")
    print(f"    LR  : {'✓ signal nouveauté' if lr_signal > 0 else '✗ non signalé'}")
    print(f"    SBN : {'✓ signal nouveauté' if sbn_signal > 0 else '✗ non signalé'}")


# ──────────────────────────────────────────────────────────────────────────────
# §7 — CONFIANCE TOP-1
# ──────────────────────────────────────────────────────────────────────────────

def section_confidence(merged: pd.DataFrame, attacks: list):
    """
    Compare la confiance top-1 (croyance b du type classifié en tête).

    Une confiance élevée signifie que la masse de croyance est concentrée
    sur un type — signe d'un scoring discriminant.

    CONF_THRESHOLD (depuis config.py) définit le seuil de "classification confiante".
    """
    _hdr(f"§7  CONFIANCE TOP-1  (top1_b)  — seuil confiance = {CONF_THRESHOLD:.2f}  [qualifié 2 côtés]")

    # PATCH m-09/F30 : top1_b est un sentinel (souvent 0) pour les lignes
    # no_groups — les inclure ferait chuter la moyenne artificiellement.
    q_lr  = _qualified_side(merged, 'lr')
    q_sbn = _qualified_side(merged, 'sbn')
    both = merged[q_lr & q_sbn].copy()
    if len(both) == 0:
        print("  Aucune fenêtre commune.")
        return

    b_lr  = both['top1_b_lr'].dropna()
    b_sbn = both['top1_b_sbn'].dropna()

    print(f"\n  {'Statistique':<18} {'LR':>10} {'SBN':>10} {'Δ (SBN−LR)':>12}")
    print(f"  {'─'*52}")
    for label, v_lr, v_sbn in [
        ('Moyenne',   b_lr.mean(),          b_sbn.mean()),
        ('Std',       b_lr.std(),           b_sbn.std()),
        ('Q1',        b_lr.quantile(0.25),  b_sbn.quantile(0.25)),
        ('Médiane',   b_lr.quantile(0.50),  b_sbn.quantile(0.50)),
        ('Q3',        b_lr.quantile(0.75),  b_sbn.quantile(0.75)),
    ]:
        print(f"  {label:<18} {v_lr:>10.4f} {v_sbn:>10.4f} {v_sbn-v_lr:>+12.4f}")

    n_conf_lr  = (b_lr  >= CONF_THRESHOLD).sum()
    n_conf_sbn = (b_sbn >= CONF_THRESHOLD).sum()
    print(f"\n  Fenêtres « confiantes » (top1_b ≥ {CONF_THRESHOLD:.2f}) :")
    print(f"    LR  : {n_conf_lr}/{len(b_lr)} ({100*n_conf_lr/len(b_lr):.1f}%)")
    print(f"    SBN : {n_conf_sbn}/{len(b_sbn)} ({100*n_conf_sbn/len(b_sbn):.1f}%)")

    _subhdr("Confiance top-1 par attaque (gate_open dans les deux)")
    print(f"\n  {'Attaque':<30} {'top1_b LR':>10} {'top1_b SBN':>11} {'Δ':>8}")
    print(f"  {'─'*62}")
    for ev in attacks:
        if ev['is_novelty_control']:
            continue
        t_s = pd.Timestamp(ev['start'])
        t_e = pd.Timestamp(ev['end'])
        sub = both[(both['timestamp'] >= t_s) & (both['timestamp'] <= t_e)]
        if len(sub) == 0:
            continue
        mb_lr  = sub['top1_b_lr'].mean()
        mb_sbn = sub['top1_b_sbn'].mean()
        print(f"  {ev['name']:<30} {mb_lr:>10.4f} {mb_sbn:>11.4f} {mb_sbn-mb_lr:>+8.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# §8 — STABILITÉ TEMPORELLE (flipping rate)
# ──────────────────────────────────────────────────────────────────────────────

def section_temporal_stability(merged: pd.DataFrame, attacks: list):
    """
    Mesure la fréquence à laquelle le top-1 change entre fenêtres consécutives.

    Un flipping rate élevé indique une instabilité de classification —
    le classificateur oscille entre types selon du bruit de fenêtre.
    Ref : Breiman (1996) Bagging Predictors — stabilité des prédictions.

    Note : la méthode SBN avec prior temporel (--temporal) devrait avoir
    un flipping rate inférieur à la méthode LR avec lissage Brown.
    """
    _hdr("§8  STABILITÉ TEMPORELLE — Flipping rate du top-1")

    print(f"\n  {'Attaque':<30} {'Flip LR':>9} {'Flip SBN':>10} {'Δ':>8}  Interprétation")
    print(f"  {'─'*75}")

    for ev in attacks:
        if ev['is_novelty_control']:
            continue
        t_s = pd.Timestamp(ev['start'])
        t_e = pd.Timestamp(ev['end'])
        mask = (merged['timestamp'] >= t_s) & (merged['timestamp'] <= t_e)
        df_ev = merged[mask]

        # PATCH m-09/F30 : flipping rate calculé sur fenêtres qualifiées par
        # chaque méthode (transitions de classification réelles, pas des
        # transitions vers/depuis le sentinel no_groups).
        q_lr_ev  = _qualified_side(df_ev, 'lr')
        q_sbn_ev = _qualified_side(df_ev, 'sbn')
        lr_open  = df_ev[q_lr_ev].sort_values('timestamp')
        sbn_open = df_ev[q_sbn_ev].sort_values('timestamp')

        flip_lr  = _flipping_rate(lr_open,  'top1_lr')
        flip_sbn = _flipping_rate(sbn_open, 'top1_sbn')

        if np.isnan(flip_lr) and np.isnan(flip_sbn):
            continue

        delta = (flip_sbn - flip_lr) if not (np.isnan(flip_lr) or np.isnan(flip_sbn)) else float('nan')
        interp = ("SBN plus stable" if not np.isnan(delta) and delta < -0.05
                  else "LR plus stable" if not np.isnan(delta) and delta > 0.05
                  else "équivalent")
        flip_lr_s  = f"{flip_lr*100:.1f}%" if not np.isnan(flip_lr) else "—"
        flip_sbn_s = f"{flip_sbn*100:.1f}%" if not np.isnan(flip_sbn) else "—"
        delta_s    = f"{delta*100:+.1f}%" if not np.isnan(delta) else "—"
        print(f"  {ev['name']:<30} {flip_lr_s:>9} {flip_sbn_s:>10} {delta_s:>8}  {interp}")


# ──────────────────────────────────────────────────────────────────────────────
# §§ SANITY CHECKS (contrainte SL)
# ──────────────────────────────────────────────────────────────────────────────

def section_sl_sanity(merged: pd.DataFrame):
    """
    Vérifie la contrainte SL fondamentale : Σb + u = 1 pour les deux méthodes.

    Une violation indique un bug dans la bijection ou une renormalisation manquante.
    Ref : Jøsang (2016) Def. 3.1 — opinion quintuple (b, d, u, a).
    """
    _hdr("§§  SANITY CHECK — Contrainte SL  Σb + u = 1")

    both = merged[merged['gate_open_lr'] & merged['gate_open_sbn']].copy()
    if len(both) == 0:
        return

    # LR : colonnes b_qualif_*
    b_lr_cols  = [c for c in both.columns if c.startswith('b_qualif_')]
    # SBN : colonnes b_sbn_* (sans _raw_)
    b_sbn_cols = [c for c in both.columns if c.startswith('b_sbn_') and '_raw_' not in c]

    print(f"\n  (sur {len(both):,} fenêtres avec gate_open dans les deux)")
    if b_lr_cols and 'u_qualif' in both.columns:
        sum_lr = both[b_lr_cols].sum(axis=1) + both['u_qualif']
        print(f"\n  LR  — Σb_qualif + u_qualif :")
        print(f"    Moyenne : {sum_lr.mean():.6f}  (doit ≈ 1.0)")
        print(f"    Max écart |1 - sum| : {(sum_lr - 1).abs().max():.2e}")
        ok_lr = ((sum_lr - 1).abs() < 1e-4).mean()
        print(f"    Fraction dans ε=1e-4 : {ok_lr*100:.1f}%  "
              f"{'✓' if ok_lr > 0.99 else '⚠️  violations numériques'}")
    else:
        print("  LR  : colonnes b_qualif_* ou u_qualif manquantes.")

    if b_sbn_cols and 'u_sbn' in both.columns:
        sum_sbn = both[b_sbn_cols].sum(axis=1) + both['u_sbn']
        print(f"\n  SBN — Σb_sbn + u_sbn :")
        print(f"    Moyenne : {sum_sbn.mean():.6f}  (doit ≈ 1.0)")
        print(f"    Max écart |1 - sum| : {(sum_sbn - 1).abs().max():.2e}")
        ok_sbn = ((sum_sbn - 1).abs() < 1e-4).mean()
        print(f"    Fraction dans ε=1e-4 : {ok_sbn*100:.1f}%  "
              f"{'✓' if ok_sbn > 0.99 else '⚠️  violations numériques'}")
    else:
        print("  SBN : colonnes b_sbn_* ou u_sbn manquantes.")


# ──────────────────────────────────────────────────────────────────────────────
# SCORECARD RÉCAPITULATIF
# ──────────────────────────────────────────────────────────────────────────────

def section_scorecard(op_results: dict):
    """Tableau récapitulatif côte-à-côte pour le rapport."""
    _hdr("SCORECARD — Résumé de la comparaison LR vs SBN")

    known = {k: v for k, v in op_results.items() if not v.get('novelty', False)}
    if not known:
        return

    macro_rec_lr  = np.nanmean([v['recall_lr']  for v in known.values()])
    macro_rec_sbn = np.nanmean([v['recall_sbn'] for v in known.values()])
    macro_pre_lr  = np.nanmean([v['prec_lr']    for v in known.values()])
    macro_pre_sbn = np.nanmean([v['prec_sbn']   for v in known.values()])

    winner_rec  = "SBN" if macro_rec_sbn > macro_rec_lr + 0.01 else ("LR" if macro_rec_lr > macro_rec_sbn + 0.01 else "Égalité")
    winner_prec = "SBN" if macro_pre_sbn > macro_pre_lr + 0.01 else ("LR" if macro_pre_lr > macro_pre_sbn + 0.01 else "Égalité")

    print(f"\n  {'Métrique':<35} {'LR':>10} {'SBN':>10} {'Vainqueur':>12}")
    print(f"  {'─'*70}")
    print(f"  {'Macro-Recall (attaques connues)':<35} {macro_rec_lr*100:>9.1f}% {macro_rec_sbn*100:>9.1f}%  {winner_rec:>10}")
    print(f"  {'Macro-Précision (attaques connues)':<35} {macro_pre_lr*100:>9.1f}% {macro_pre_sbn*100:>9.1f}%  {winner_prec:>10}")

    # TTQ moyen
    ttqs_lr  = [v['ttq_lr']  for v in known.values() if v.get('ttq_lr')  is not None]
    ttqs_sbn = [v['ttq_sbn'] for v in known.values() if v.get('ttq_sbn') is not None]
    if ttqs_lr or ttqs_sbn:
        mean_ttq_lr  = np.mean(ttqs_lr)  if ttqs_lr  else float('nan')
        mean_ttq_sbn = np.mean(ttqs_sbn) if ttqs_sbn else float('nan')
        winner_ttq = ("SBN" if not np.isnan(mean_ttq_sbn) and
                                (np.isnan(mean_ttq_lr) or mean_ttq_sbn < mean_ttq_lr - 1)
                      else "LR" if not np.isnan(mean_ttq_lr) and
                                   (np.isnan(mean_ttq_sbn) or mean_ttq_lr < mean_ttq_sbn - 1)
                      else "Égalité")
        print(f"  {'TTQ moyen (min, ↓ mieux)':<35} "
              f"{f'{mean_ttq_lr:.1f}':>9} {f'{mean_ttq_sbn:.1f}':>10}  {winner_ttq:>10}")

    print(f"\n  Note : les chiffres §2 par attaque sont la source de vérité.")
    print(f"  Le scorecard ne résume pas §1 (gate), §5 (u), §7 (conf), §8 (flip).")
    print(f"  Consulter ces sections pour un jugement architectural complet.")


# ──────────────────────────────────────────────────────────────────────────────
# MERGE DES DEUX CSV
# ──────────────────────────────────────────────────────────────────────────────

def _load_and_merge(lr_csv: str, sbn_csv: str) -> pd.DataFrame:
    """
    Charge les deux CSV et les fusionne sur le timestamp.

    Renomme les colonnes pour éviter les ambiguïtés :
      gate_open    → gate_open_lr / gate_open_sbn
      top1_type    → top1_lr / top1_sbn
      top1_b       → top1_b_lr / top1_b_sbn
      u_qualif     → u_qualif (LR)
      u_sbn        → u_sbn (SBN)
    """
    print(f"→ Lecture LR  : {lr_csv}")
    df_lr = pd.read_csv(lr_csv,  parse_dates=['timestamp'])
    print(f"→ Lecture SBN : {sbn_csv}")
    df_sbn = pd.read_csv(sbn_csv, parse_dates=['timestamp'])

    # Renommage LR
    # PATCH m-09/F30 : qual_status est aussi renommée pour éviter que le
    # suffixe ('', '_sbn_dup') lors du merge ne supprime silencieusement la
    # version SBN. Sans ce renommage, _qualified_side('sbn') retomberait sur
    # gate_open seul (perte de la sémantique no_groups).
    lr_rename = {
        'gate_open':   'gate_open_lr',
        'top1_type':   'top1_lr',
        'top1_b':      'top1_b_lr',
        'qual_status': 'qual_status_lr',
    }
    df_lr = df_lr.rename(columns=lr_rename)

    # Renommage SBN
    sbn_rename = {
        'gate_open':   'gate_open_sbn',
        'top1_type':   'top1_sbn',
        'top1_b':      'top1_b_sbn',
        'qual_status': 'qual_status_sbn',
    }
    df_sbn = df_sbn.rename(columns=sbn_rename)

    # Merge sur timestamp (inner : on ne garde que les fenêtres communes)
    merged = df_lr.merge(df_sbn, on='timestamp', how='inner', suffixes=('', '_sbn_dup'))
    # Supprimer les éventuelles colonnes dupliquées
    dup_cols = [c for c in merged.columns if c.endswith('_sbn_dup')]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    print(f"\n  LR  : {len(df_lr):,} fenêtres")
    print(f"  SBN : {len(df_sbn):,} fenêtres")
    print(f"  Merge inner : {len(merged):,} fenêtres communes")

    # Détection du signal de nouveauté SBN si présent
    for col in ['novelty_lr', 'novelty_score']:
        if col in df_sbn.columns and col not in merged.columns:
            # Réintégrer depuis df_sbn
            merged = merged.merge(df_sbn[['timestamp', col]], on='timestamp', how='left')

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def _find_csv(name: str) -> str | None:
    candidates = [
        os.path.join(RESULTS_DIR, name),
        os.path.join('.', name),
        os.path.join('..', 'results', name),
    ]
    # Chercher dans les sous-dossiers results/
    for root, dirs, files in os.walk(RESULTS_DIR if os.path.isdir(RESULTS_DIR) else '.'):
        if name in files:
            candidates.insert(0, os.path.join(root, name))
        break  # un seul niveau
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Comparaison exhaustive LR vs SBN — qualification du type d\'anomalie',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--lr_csv',  default=None,
                        help='CSV de qualify_anomaly.py (défaut : auto-détection qualif_types.csv)')
    parser.add_argument('--sbn_csv', default=None,
                        help='CSV de qualify_anomaly_sbn.py (défaut : auto-détection qualif_types_sbn.csv)')
    parser.add_argument('--balance_ratio', default=None,
                        help='Suffixe balance_ratio (ex: "auto") — modifie les noms de fichiers')
    parser.add_argument('--real', action='store_true', default=False,
                        help='Mode attaques réelles (utilise REAL_ATTACKS de config.py)')
    parser.add_argument('--out_csv', default=None,
                        help='Exporte le DataFrame mergé dans ce fichier CSV')
    parser.add_argument('--skip_sections', default='',
                        help='Sections à sauter (ex: "3,4,8")')
    args = parser.parse_args()

    skip = {int(s) for s in args.skip_sections.split(',') if s.strip().isdigit()}

    bal_sfx = f'_balance_{args.balance_ratio}' if args.balance_ratio else ''

    # Résolution automatique des CSV
    if args.lr_csv is None:
        args.lr_csv = _find_csv(f'qualif_types{bal_sfx}.csv')
        if args.lr_csv is None:
            print(f"[ERREUR] CSV LR introuvable. "
                  f"Passez --lr_csv ou placez qualif_types{bal_sfx}.csv dans {RESULTS_DIR}")
            sys.exit(1)

    if args.sbn_csv is None:
        args.sbn_csv = _find_csv(f'qualif_types_sbn{bal_sfx}.csv')
        if args.sbn_csv is None:
            print(f"[ERREUR] CSV SBN introuvable. "
                  f"Passez --sbn_csv ou placez qualif_types_sbn{bal_sfx}.csv dans {RESULTS_DIR}")
            sys.exit(1)

    # Chargement et merge
    print("\n" + "═"*110)
    print("  COMPARE_QUALIF_METHODS — Comparaison LR vs SBN")
    print("═"*110)

    merged = _load_and_merge(args.lr_csv, args.sbn_csv)

    # Sélection du catalogue
    # PATCH m-04/F21 : Les NETWORK_OUTAGE sont retirés des listes principales
    # pour ne pas être agrégés comme des attaques dans les macros.  Ils sont
    # loggés dans une section dédiée plus bas.
    if not args.real:
        attacks = _ATTACK_EVENTS_NO_OUTAGE
        outages = _OUTAGE_EVENTS
    else:
        _real_all = [
            {'name': name, 'expected': evs[0].get('expected_qualif', ''),
             'start': evs[0]['start'], 'end': evs[0]['end'],
             'intensity': '—', 'is_novelty_control': False}
            for name, evs in REAL_ATTACKS.items()
        ]
        attacks, outages = partition_catalog_by_outage(_real_all)

    if outages:
        print(f"\n[m-04/F21] {len(outages)} événement(s) NETWORK_OUTAGE retirés "
              f"des tableaux d'attaques :")
        for o in outages:
            print(f"    - {o.get('name', '?')}  "
                  f"({o.get('start', '?')} → {o.get('end', '?')})")
        print(f"  (Ces événements ne sont PAS des attaques adversariales — "
              f"taxonomie séparée. Cf. qualif_filters.is_outage_event.)")

    # Exécution des sections
    if 1 not in skip:
        section_gate_agreement(merged)
    if 2 not in skip:
        op_results = section_operational_metrics(merged, attacks)
    else:
        op_results = {}
    if 3 not in skip:
        section_classification_agreement(merged)
    if 4 not in skip:
        section_disagreement_analysis(merged)
    if 5 not in skip:
        section_uncertainty(merged, attacks)
    if 6 not in skip:
        section_novelty(merged)
    if 7 not in skip:
        section_confidence(merged, attacks)
    if 8 not in skip:
        section_temporal_stability(merged, attacks)

    section_sl_sanity(merged)

    if op_results:
        section_scorecard(op_results)

    # Export optionnel
    if args.out_csv:
        merged.to_csv(args.out_csv, index=False)
        print(f"\n✅ DataFrame mergé exporté : {args.out_csv}")

    print(f"\n{'═'*110}")
    print(f"  Fin de la comparaison.")
    print(f"{'═'*110}\n")