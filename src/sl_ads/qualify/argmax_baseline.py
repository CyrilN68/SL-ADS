"""
qualify_argmax_baseline.py — Baseline « no-SL » pour la qualification d'anomalie
================================================================================

OBJECTIF SCIENTIFIQUE
---------------------
Servir de baseline équitable pour isoler la valeur ajoutée du raisonnement
Subjective Logic (SL) dans l'étape de qualification du type d'anomalie.

Ce script est le pendant no-SL de ``qualify_anomaly_sbn.py`` et produit un CSV
structurellement comparable, consommable par ``evaluate_qualify_sbn.py``
(une fois celui-ci étendu pour reconnaître le format ``b_argmax_*``).

PROTOCOLE DE COMPARAISON — CE QUI EST IDENTIQUE AU SBN (contrôle)
-----------------------------------------------------------------
Afin d'isoler strictement la contribution du raisonnement SL à la QUALIFICATION
(et non à d'autres étapes du pipeline), le baseline partage avec le SBN :

  1. Même CSV d'entrée : ``detection_results_INJECTED.csv`` (sortie compute_opinions_v3).
  2. Mêmes projections observées par source : ``{source}_proj_safe|proj_susp|proj_atk``
     (issues de la chaîne de détection SL — colonnes L1→L3 de l'architecture SL).
  3. Même pooling logarithmique intra-groupe (geomean normalisée) :
     P^g_s = geomean_{m∈g} P^m_s, puis renormalisation sur {Safe, Susp, Anom}.
     Le pooling géométrique est une opération de statistique pure — pas de SL :
     Genest & Zidek (1986) *Statist. Sci.* §4 : caractérisation axiomatique
     du pooling logarithmique (logarithmic opinion pool).
  4. Même connaissance experte : ``SBN_COND_OPINIONS[type_k][group_g] = {S, M, A}``
     (distributions conditionnelles issues de la littérature IDS).
  5. Même ``gate_open`` : HÉRITÉ DE LA DÉTECTION SL (CSV d'entrée).
     → Isole strictement la qualification ; les comparaisons DR sont à égalité
       par construction, la comparaison porte sur QP / F1 / F-β / top-1.

CE QUI DIFFÈRE — LA CONTRIBUTION SL TESTÉE
------------------------------------------
Le baseline NE CONTIENT AUCUNE des briques SL suivantes :

  (a) Bijection évidence → masse de croyance (Jøsang 2016 §3.5.2).
  (b) Uncertainty Maximisation (Jøsang 2016 Eq. 3.27).
  (c) Prior temporel Markovien (Hutchins 2011 kill chain + WBF Jøsang §14.6).
  (d) Classe résiduelle ``Autre_Anomalie`` via seuil u_raw
      (mécanisme de signal de nouveauté du SBN).
  (e) Probabilité projetée P_proj(k) = b(k) + u·a(k) avec prior a(k).

À la place, le baseline effectue :

    Score(k, g) = Σ_{s∈{S,M,A}} P^{obs}_{g,s} · c^{k|g}_s      (produit scalaire)
    logL(k)     = Σ_{g ∈ G_actifs} log( max(Score(k,g), ε) )    (pooling log inter-groupe)
    top1_type   = argmax_k logL(k)                              (classification déterministe)
    b_argmax[k] = softmax(logL(k))                              (distribution normalisée, POUR COMPARAISON DES DISTRIBUTIONS UNIQUEMENT)
    u_argmax    = 0.0                                           (baseline sans incertitude)
    novelty_lr  = 1 / (max(L) / mean(L))                        (indice de concentration — même formule que SBN)

Le pooling inter-groupe par somme de log-scores (≡ produit des scores) correspond
à l'hypothèse d'indépendance conditionnelle des groupes (Naive Bayes ; Rish 2001,
IJCAI). Cette hypothèse est la MÊME que celle du SBN — c'est exclusivement la
couche d'agrégation SL (belief/u/prior/temporel) qui diffère.

La distribution ``b_argmax`` produite par softmax n'est PAS une masse de
croyance Subjective Logic ; c'est une distribution de probabilité dérivée
uniquement pour peupler le format CSV attendu par l'évaluateur (entropie de
nouveauté). Elle ne doit JAMAIS être interprétée comme un u / belief SL.

OUTPUT CSV
----------
Fichier : ``<results_dir>/qualif_types_argmax.csv``
Colonnes :
  - ``timestamp``, ``gate_open``, ``top1_type``, ``top1_b``, ``top1_proj``, ``qual_status``
  - ``u_argmax`` (= 0.0 par construction — baseline sans incertitude)
  - ``u_argmax_raw`` (= 0.0 — idem)
  - ``novelty_lr`` (indice de concentration, même définition que SBN)
  - ``b_argmax_{TYPE}`` pour chaque type dans SBN_COND_OPINIONS (K colonnes)
  - ``b_argmax_raw_{TYPE}`` (copie de ``b_argmax_{TYPE}`` — pas de pipeline temporel)
  - ``b_argmax_Autre_Anomalie`` ≡ 0.0 (classe résiduelle non-modélisée par le baseline)

COMPATIBILITÉ avec ``evaluate_qualify_sbn.py``
----------------------------------------------
L'évaluateur ``evaluate_qualify_sbn.py`` doit être étendu pour reconnaître le
préfixe ``b_argmax_`` et la colonne ``u_argmax`` (format ``argmax``). Sans cette
extension, il échoue sur le CSV produit par ce script.

USAGE
-----
    python qualify_argmax_baseline.py
    python qualify_argmax_baseline.py --input detection_results_INJECTED.csv
    python qualify_argmax_baseline.py --output qualif_types_argmax_debug.csv
    python qualify_argmax_baseline.py --threshold 0.20

RÉFÉRENCES MÉTHODOLOGIQUES
--------------------------
- Duda & Hart (1973) *Pattern Classification*, §2.6 — nearest-mean classifier.
- Genest & Zidek (1986) *Statist. Sci.* §4 — logarithmic opinion pool.
- Rish (2001) IJCAI Workshop — Naive Bayes sous dépendances résiduelles.
- Domingos & Pazzani (1997) *Machine Learning* 29(2-3) — robustesse Naive Bayes.
- Good (1950) *Probability and the Weighing of Evidence* §6 — weight of evidence ratio.
- Sharafaldin et al. (2018) ICISSP — framework d'évaluation IDS.
- Tatbul et al. (2018) NeurIPS — precision/recall sur séries temporelles.

Ce script écrit UNIQUEMENT dans un CSV ``qualif_types_argmax.csv``.
Il ne modifie aucun autre fichier du pipeline.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Imports projet — suivent strictement la même convention que qualify_anomaly_sbn.py
# ──────────────────────────────────────────────────────────────────────────────
try:
    from sl_ads.config import CONFIG, QUALIFY_GROUP_SOURCES as _GS_MODULE  # Phase H
    from sl_ads.paths import get_results_dir, get_decision_threshold, get_detection_col  # Phase H
    _THRESHOLD    = get_decision_threshold(CONFIG, up_levels=1)
    _DET_COL      = get_detection_col(CONFIG, up_levels=1)
    _OUTPUT_DIR   = get_results_dir(CONFIG, up_levels=1)
    GROUP_SOURCES = CONFIG.get('QUALIFY_GROUP_SOURCES', _GS_MODULE)
    SBN_COND      = CONFIG.get('SBN_COND_OPINIONS', {})
except ImportError as exc:  # pragma: no cover — garde-fou
    print(f"[WARN] config.py/paths.py non trouvés — fallback ({exc}).")
    _THRESHOLD    = 0.20
    _DET_COL      = 'FINAL_SYSTEM_CBF_proj_atk'
    _OUTPUT_DIR   = '.'
    GROUP_SOURCES = {}
    SBN_COND      = {}

# ──────────────────────────────────────────────────────────────────────────────
# Constantes partagées (identiques à qualify_anomaly_sbn.py — pour garantir
# la stricte équivalence d'entrée et d'interprétation)
# ──────────────────────────────────────────────────────────────────────────────
STATES      = ('Safe', 'Susp', 'Anom')
STATE_COLS  = {'Safe': 'proj_safe', 'Susp': 'proj_susp', 'Anom': 'proj_atk'}
EPS         = 1e-12
NEUTRAL_COND = {'Safe': 1.0 / 3.0, 'Susp': 1.0 / 3.0, 'Anom': 1.0 / 3.0}


# ──────────────────────────────────────────────────────────────────────────────
# POOLING INTRA-GROUPE — copie conforme de _compute_group_projected (SBN)
# Nécessaire pour que les deux méthodes partagent EXACTEMENT la même vision
# des observations {P^g_s}. Toute divergence ici biaiserait la comparaison.
# ──────────────────────────────────────────────────────────────────────────────
def _compute_group_projected(row: pd.Series, group_name: str) -> dict[str, float] | None:
    """
    P^{obs}_{g,s} = geomean_{m∈g} P^m_s (puis renormalisation sur {Safe, Susp, Anom}).

    Strictement identique à qualify_anomaly_sbn._compute_group_projected.
    Retourne None si aucune source du groupe n'a de colonne renseignée.

    Ref : Genest & Zidek (1986) *Statist. Sci.* §4 ; Aczél & Daróczy (1975).
    """
    sources = GROUP_SOURCES.get(group_name, [])
    result: dict[str, float | None] = {}

    for state, col_suffix in STATE_COLS.items():
        vals: list[float] = []
        for src in sources:
            col = f"{src}_{col_suffix}"
            if col in row.index and pd.notna(row[col]):
                vals.append(float(row[col]))
        if not vals:
            result[state] = None
        else:
            log_mean = np.mean([np.log(max(v, EPS)) for v in vals])
            result[state] = float(np.exp(log_mean))

    if all(v is None for v in result.values()):
        return None

    total = sum(v for v in result.values() if v is not None)
    if total < EPS:
        return None
    for state in STATES:
        if result[state] is None:
            result[state] = 0.0
    total = sum(result.values())
    if total > EPS:
        result = {s: v / total for s, v in result.items()}   # type: ignore[misc]
    return result                                             # type: ignore[return-value]


def _normalize_cond_opinion(cond: dict) -> dict[str, float]:
    """
    Normalise une distribution conditionnelle sur {Safe, Susp, Anom} (somme = 1).
    Strictement identique à qualify_anomaly_sbn._normalize_cond_opinion.
    """
    total = sum(cond.get(s, 0.0) for s in STATES)
    if total < EPS:
        return dict(NEUTRAL_COND)
    return {s: cond.get(s, 0.0) / total for s in STATES}


def _group_compat_score(group_pp: dict[str, float], cond_opinion: dict[str, float]) -> float:
    """
    Score de compatibilité par groupe : produit scalaire entre observation et signature.

        Score(k, g) = Σ_s P^{obs}_{g,s} · c^{k|g}_s

    Propriétés :
      - ∈ (0, 1] après normalisation des deux distributions ;
      - neutre = 1/3 si P^obs uniforme ou c^{k|g} uniforme ;
      - maximum si P^obs concentré sur l'état attendu par c^{k|g}.

    Ref : Duda & Hart (1973) §2.6 — nearest-mean classifier.
    """
    score = 0.0
    for state in STATES:
        p_obs  = group_pp.get(state, 0.0)
        p_cond = cond_opinion.get(state, 1.0 / 3.0)
        score += p_obs * p_cond
    return max(score, EPS)


# ──────────────────────────────────────────────────────────────────────────────
# NOYAU : CLASSIFICATION ARGMAX D'UNE FENÊTRE (no-SL)
# ──────────────────────────────────────────────────────────────────────────────
def argmax_qualify_row(
    row:        pd.Series,
    sbn_cond:   dict,
    threshold:  float,
    det_col:    str,
) -> dict[str, Any]:
    """
    Qualification no-SL d'une fenêtre — argmax du log-likelihood dot-product.

    Input
    -----
    row        : ligne de detection_results_INJECTED.csv
    sbn_cond   : SBN_COND_OPINIONS (matrice experte K×G)
    threshold  : seuil gate P(Anom) identique au SBN
    det_col    : colonne de décision (ex. 'FINAL_SYSTEM_CBF_proj_atk')

    Output
    ------
    dict avec : timestamp, gate_open, top1_type, top1_b, top1_proj, qual_status,
                u_argmax, u_argmax_raw, novelty_lr, b_argmax_{TYPE}, b_argmax_raw_{TYPE}.
    """
    type_names = list(sbn_cond.keys())
    K = len(type_names)

    def _empty_row() -> dict[str, Any]:
        # Valeurs-contrat identiques à qualify_anomaly_sbn._empty_result :
        # u_* = 1.0 et novelty_lr = 1.0 pour les fenêtres NORMALES (gate_open=False).
        # NE PAS interpréter comme signal de nouveauté (cf. docstring SBN).
        r: dict[str, Any] = {
            'gate_open':     False,
            'top1_type':     '',
            'top1_b':        0.0,
            'top1_proj':     0.0,
            'qual_status':   'normal',
            'u_argmax':      1.0,
            'u_argmax_raw':  1.0,
            'novelty_lr':    1.0,
        }
        for k in type_names:
            r[f'b_argmax_{k}']     = 0.0
            r[f'b_argmax_raw_{k}'] = 0.0
        r['b_argmax_Autre_Anomalie']     = 0.0
        r['b_argmax_raw_Autre_Anomalie'] = 0.0
        return r

    # ── L1 : Gate — strictement la même que le SBN (isole la qualification) ──
    p_atk_raw = row.get(det_col, row.get('FINAL_SYSTEM_CBF_proj_atk', 0.0))
    try:
        p_atk = float(p_atk_raw)
    except (TypeError, ValueError):
        p_atk = float('nan')
    if pd.isna(p_atk) or p_atk < threshold:
        return _empty_row()

    result = _empty_row()
    result['gate_open'] = True

    # ── L2 : P^{obs}_{g,s} par geomean (strict parallèle au SBN) ──────────────
    group_pp: dict[str, dict[str, float]] = {}
    for grp in GROUP_SOURCES:
        pp = _compute_group_projected(row, grp)
        if pp is not None:
            group_pp[grp] = pp

    if not group_pp:
        # Aucune donnée de groupe — qualification impossible (baseline et SBN
        # sont logés à la même enseigne, cas 'no_groups').
        result['qual_status']  = 'no_groups'
        result['novelty_lr']   = 1.0
        # u_argmax reste à 1.0 (cohérent avec SBN pour filtrage aval).
        return result

    # ── L3 : Score par (type, groupe) — produit scalaire ─────────────────────
    score_matrix: dict[str, dict[str, float]] = {}
    for type_k, cond_by_group in sbn_cond.items():
        if not isinstance(cond_by_group, dict):
            continue
        score_matrix[type_k] = {}
        for grp, cond_raw in cond_by_group.items():
            if grp not in group_pp:
                continue
            cond_norm = _normalize_cond_opinion(cond_raw)
            score_matrix[type_k][grp] = _group_compat_score(group_pp[grp], cond_norm)

    if not score_matrix or all(len(v) == 0 for v in score_matrix.values()):
        # Aucun groupe SBN_COND ∩ groupes observés → cas dégénéré.
        result['qual_status'] = 'no_groups'
        result['novelty_lr']  = 1.0
        return result

    # ── L4 : Pooling inter-groupe — somme des log-scores (≡ produit) ─────────
    # log L(k) = Σ_{g ∈ actifs(k)} log max(ε, Score(k, g))
    # Intuition : hypothèse de Naive Bayes (indépendance conditionnelle des
    # groupes sachant le type). C'est LA MÊME hypothèse que le SBN — seule
    # l'étape suivante (bijection SL vs argmax direct) diffère.
    #
    # Pour assurer la comparabilité quand des groupes sont manquants pour
    # certains types, on normalise par |actifs(k)| (moyenne log → geomean).
    log_likelihoods: dict[str, float] = {}
    for type_k, grp_sc in score_matrix.items():
        if not grp_sc:
            log_likelihoods[type_k] = math.log(1.0 / 3.0)  # neutre
            continue
        log_scores = [math.log(max(sc, EPS)) for sc in grp_sc.values()]
        log_likelihoods[type_k] = sum(log_scores) / len(log_scores)

    # ── Likelihoods linéaires L(k) ∈ (0, 1] — utilisées pour novelty_lr ──────
    likelihoods_lin = {k: math.exp(v) for k, v in log_likelihoods.items()}

    # ── novelty_lr — indice de concentration, MÊME DÉFINITION que le SBN ─────
    # Ref : Good (1950) §6 ; indice de Herfindahl inversé.
    L_vals = list(likelihoods_lin.values())
    mean_L = sum(L_vals) / max(len(L_vals), 1)
    max_L  = max(L_vals) if L_vals else 0.0
    lr     = (max_L / mean_L) if mean_L > EPS else 1.0
    novelty_lr = float(1.0 / lr) if lr > EPS else 1.0

    # ── Distribution b_argmax via softmax des log-likelihoods ─────────────────
    # AVERTISSEMENT : ceci N'EST PAS une masse de croyance SL. C'est une
    # distribution de probabilité produite UNIQUEMENT pour peupler le format
    # attendu par evaluate_qualify_sbn (calcul de l'entropie de nouveauté).
    #
    # Propriétés :
    #   - softmax : stable numériquement (subtract max) ;
    #   - Σ_k b_argmax[k] = 1 → compatible avec le calcul d'entropie ;
    #   - NE contient AUCUNE information d'incertitude (u_argmax ≡ 0).
    log_max = max(log_likelihoods.values())
    exp_lk  = {k: math.exp(v - log_max) for k, v in log_likelihoods.items()}
    Z = sum(exp_lk.values())
    if Z < EPS:
        # Dégénère en uniforme (tous les scores à 0 après subtract max impossible,
        # garde-fou de numérique)
        b_argmax = {k: 1.0 / max(K, 1) for k in type_names}
    else:
        b_argmax = {k: exp_lk.get(k, 0.0) / Z for k in type_names}

    # ── Décision top-1 ────────────────────────────────────────────────────────
    top1 = max(type_names, key=lambda k: log_likelihoods.get(k, -math.inf))
    top1_b = b_argmax.get(top1, 0.0)

    # ── Remplissage résultat ─────────────────────────────────────────────────
    for k in type_names:
        bk = b_argmax.get(k, 0.0)
        result[f'b_argmax_{k}']     = bk
        result[f'b_argmax_raw_{k}'] = bk  # pas de pipeline temporel → raw = final
    # Classe résiduelle non-modélisée par le baseline (contribution SBN)
    result['b_argmax_Autre_Anomalie']     = 0.0
    result['b_argmax_raw_Autre_Anomalie'] = 0.0

    result['top1_type']  = top1
    result['top1_b']     = top1_b
    # top1_proj = top1_b car baseline sans prior ni incertitude
    result['top1_proj']  = top1_b
    # Baseline sans incertitude : u_argmax ≡ 0. NE PAS confondre avec u_sbn.
    result['u_argmax']     = 0.0
    result['u_argmax_raw'] = 0.0
    result['novelty_lr']   = novelty_lr
    result['qual_status']  = 'qualified'

    return result


# ──────────────────────────────────────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ──────────────────────────────────────────────────────────────────────────────
def run(
    input_csv:  str,
    output_csv: str,
    threshold:  float,
    det_col:    str,
    sbn_cond:   dict,
) -> pd.DataFrame:
    """Charge le CSV de détection, applique argmax_qualify_row à chaque ligne, écrit le CSV."""
    if not sbn_cond:
        raise RuntimeError(
            "[ARGMAX] SBN_COND_OPINIONS vide — vérifier config.py. "
            "Le baseline argmax a besoin des mêmes signatures expertes que le SBN "
            "pour être un contrôle équitable."
        )
    if not GROUP_SOURCES:
        raise RuntimeError(
            "[ARGMAX] QUALIFY_GROUP_SOURCES vide — vérifier config.py. "
            "Sans groupes définis, aucune qualification possible."
        )

    type_names = list(sbn_cond.keys())
    K = len(type_names)

    print(f"[ARGMAX] Baseline no-SL — K={K} types : {type_names}")
    print(f"  Groupes ({len(GROUP_SOURCES)}) : {list(GROUP_SOURCES.keys())}")
    print(f"  Gate   = {det_col} >= {threshold:.3f}  (hérité SL, isole la qualification)")
    print(f"  Sortie = {output_csv}")
    print(f"[ARGMAX] Lecture : {input_csv}")

    df = pd.read_csv(input_csv, parse_dates=['timestamp'])
    print(f"  {len(df)} fenêtres, {len(df.columns)} colonnes")

    required = {'timestamp', det_col}
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {input_csv} : {missing}")

    # Warnings colonnes manquantes par groupe (strictement comme le SBN)
    for grp, srcs in GROUP_SOURCES.items():
        for src in srcs:
            for suf in STATE_COLS.values():
                col = f"{src}_{suf}"
                if col not in df.columns:
                    print(f"  [WARN] Colonne manquante : {col} (groupe {grp})")

    # ── Classification ligne par ligne ───────────────────────────────────────
    print(f"\n[ARGMAX] Classification en cours ...")
    results_list: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        r = argmax_qualify_row(
            row       = row,
            sbn_cond  = sbn_cond,
            threshold = threshold,
            det_col   = det_col,
        )
        r['timestamp'] = row['timestamp']
        results_list.append(r)

    df_out = pd.DataFrame(results_list)
    # Réordonne : timestamp d'abord, puis colonnes structurées
    col_order = [
        'timestamp', 'gate_open', 'top1_type', 'top1_b', 'top1_proj', 'qual_status',
        'u_argmax', 'u_argmax_raw', 'novelty_lr',
    ]
    b_cols     = [f'b_argmax_{k}' for k in type_names] + ['b_argmax_Autre_Anomalie']
    b_raw_cols = [f'b_argmax_raw_{k}' for k in type_names] + ['b_argmax_raw_Autre_Anomalie']
    col_order += b_cols + b_raw_cols
    # Ajoute toute colonne absente de col_order à la fin (robustesse)
    extra_cols = [c for c in df_out.columns if c not in col_order]
    df_out = df_out[col_order + extra_cols]

    # ── Diagnostics d'intégrité ──────────────────────────────────────────────
    n_windows    = len(df_out)
    n_triggered  = int(df_out['gate_open'].sum())
    n_no_groups  = int((df_out['qual_status'] == 'no_groups').sum())
    print(f"\n[ARGMAX] gate_open={n_triggered}/{n_windows} "
          f"({100 * n_triggered / max(n_windows, 1):.2f}%)  "
          f"no_groups={n_no_groups}")

    # Vérification : Σ b_argmax_* (hors _raw_) ∈ {0 (normal), 1 (qualifié)}
    b_cols_live = [c for c in df_out.columns if c.startswith('b_argmax_') and '_raw_' not in c]
    if b_cols_live:
        sums = df_out[b_cols_live].sum(axis=1).values
        mask_open = df_out['gate_open'].values.astype(bool)
        if mask_open.any():
            err_open = float(np.max(np.abs(sums[mask_open] - 1.0)))
            print(f"  Intégrité : max|Σb_argmax − 1| (fenêtres gate_open) = {err_open:.2e} "
                  f"(objectif < 1e-9)")

    df_out.to_csv(output_csv, index=False)
    print(f"[OK] CSV ARGMAX sauvegardé : {output_csv}")
    return df_out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_input(explicit: str | None) -> str:
    if explicit is not None:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"--input fourni mais introuvable : {explicit}")
        return explicit

    candidates = [
        os.path.join(_OUTPUT_DIR, 'detection_results_INJECTED.csv'),
        os.path.join(_OUTPUT_DIR, 'detection_results.csv'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Aucun detection_results*.csv trouvé dans {_OUTPUT_DIR}. "
        "Passer --input <chemin>."
    )


def _resolve_output(explicit: str | None, input_csv: str) -> str:
    if explicit is not None:
        return explicit
    base_dir = os.path.dirname(input_csv) or _OUTPUT_DIR
    return os.path.join(base_dir, 'qualif_types_argmax.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Baseline no-SL (argmax log-likelihood) pour la qualification "
            "d'anomalie — contrôle équitable vs qualify_anomaly_sbn.py"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input',     default=None,
                        help="CSV de compute_opinions_v3.py (detection_results*.csv)")
    parser.add_argument('--output',    default=None,
                        help="CSV de sortie (défaut : qualif_types_argmax.csv)")
    parser.add_argument('--threshold', type=float, default=_THRESHOLD,
                        help="Seuil gate P(Anom) — doit être IDENTIQUE à celui du SBN "
                             "pour garantir l'équité de la comparaison")
    parser.add_argument('--det-col',   default=_DET_COL,
                        help="Colonne de décision du gate (cohérente avec train_v10 sidecar)")
    args = parser.parse_args()

    input_csv  = _resolve_input(args.input)
    output_csv = _resolve_output(args.output, input_csv)

    run(
        input_csv  = input_csv,
        output_csv = output_csv,
        threshold  = args.threshold,
        det_col    = args.det_col,
        sbn_cond   = SBN_COND,
    )
