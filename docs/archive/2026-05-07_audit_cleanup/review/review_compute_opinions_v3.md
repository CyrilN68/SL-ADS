# Recensement technique — `compute_opinions_v3.py`

## 1. Rôle dans la pipeline

Ce module est le **cœur décisionnel** du système IDS. Il prend en entrée les preuves
élémentaires (triplets P/S/N par métrique et par fenêtre temporelle) produites par
`compute_evidence_v2.py` et produit des **opinions subjectives fusionnées** selon la
Logique Subjective (Jøsang 2016).

Il est exécuté **après** `compute_evidence_v2.py` (et optionnellement
`inject_at_evidence_level.py`) et **avant** les scripts d'évaluation
(`qualify_anomaly_sbn.py`, `evaluate_with_labels.py`, etc.).

---

## 2. Architecture à 3 niveaux de fusion

| Niveau | Nom | Opérateur | Référence | Entrée → Sortie |
|--------|-----|-----------|-----------|-----------------|
| N1 | Ageing Conflict-Aware | λ_dyn = λ_base × max(0, 1 − α×K) | Jøsang 2016, Eq. 16.5 | Preuves brutes → Preuves accumulées + opinion feuille |
| N2 | WBF intra-méthode | Weighted Belief Fusion N-ary | Jøsang 2016, Eq. 12.22 | Opinions feuilles → op_prophet / op_reconst |
| N3 | CBF ou WBF inter-méthode | CBF (défaut) ou WBF | Jøsang 2016, Eq. 12.14 / 12.22 | op_prophet + op_reconst → op_final |

---

## 3. Flux de données détaillé

```
evidence_{VERSION}.csv          metadata_{VERSION}.csv      raw_data_{VERSION}.csv
        |                               |                            |
        v                               v                            v
  Resample (.sum)           meta_dict reconstruction          raw_data dict
  → df_ev (fenêtres)        → prophet_keys, reconst_keys      (pour graphiques)
        |
        v
  BOUCLE PAR FENÊTRE (N fenêtres)
    |
    ├─ POUR CHAQUE MÉTRIQUE FEUILLE :
    |     ├─ N1 : temporal_adaptive_ageing(R_acc, R_curr, λ, α)
    |     |        → R_new, K_conf, λ_dyn
    |     ├─ evidence_to_opinion(R_new, W, a_EDP)
    |     ├─ [opt] uncertainty_maximized()
    |     ├─ get_c3_weight(clean_key, row, mode)
    |     ├─ [opt] apply_trust_discount()
    |     └─ update safe_rmse_state[key]  (per-metric)
    |
    ├─ N2 : fusion_wbf_n_sources(ops_prophet, w)  → op_prophet
    |        fusion_wbf_n_sources(ops_reconst, w)  → op_reconst
    |
    ├─ [opt] apply_contextual_discount(op_reconst, CD_ALPHA_ATTACK)
    |
    └─ N3 : fusion_cbf(op_prophet, op_reconst)    → op_final  [défaut]
             ou fusion_wbf_n_sources([...])         → op_final  [wbf]
             ou CBF avec boost_opinion_evidence()   → op_final  [balance]
        |
        └─ p_atk = op_final.projected_prob()[2]   [variable de décision]

        → viz_db (toutes métriques + niveaux agrégés)

  → save_results_to_csv → detection_results[_INJECTED].csv
  → plot_metric_complete → graph_{metric}.png
```

---

## 4. Paramètres configurables (config.py)

| Paramètre | Type | Défaut | Rôle |
|-----------|------|--------|------|
| `LAMBDA_DECAY` | float | — | Taux d'oubli temporel de base (N1) |
| `CONFLICT_ALPHA` | float | calculé dynamiquement | Amplification conflit (hard-reset garanti) |
| `SL_PARAM_K` | float | 3.0 | Prior weight W (bijection Def. 3.9) |
| `SL_PRIOR_A` | list[float] | [1/3,1/3,1/3] | Prior uniforme (fallback si EDP absent) |
| `USE_EMPIRICAL_PRIOR` | bool | True | Active/désactive l'EDP |
| `WBF_WEIGHT_MODE` | str | "uniform" | Pondération N2 : uniform / r2_static / trust_discount |
| `C3_WEIGHT_MODE` | str | "uniform" | Pondération C3 : uniform / r2_static / prophet_interval / online_rmse |
| `C3_ONLINE_RMSE_WARMUP` | int | 10 | Fenêtres de warmup avant activation online_rmse |
| `C3_ONLINE_RMSE_ALPHA` | float | 0.05 | Coefficient EWMA du RMSE glissant |
| `INTER_METHOD_FUSION` | str | "wbf" | Operateur N3 : "wbf", "abf", "cbf", "bcf", "ccf", "minbf", "maxbf" ou "hierarchical" |
| `BALANCE_RATIO` | float/"auto" | 1.0 | Correction biais N_prophet vs N_reconst |
| `RECONST_ATTACK_RELIABILITY` | float/"auto" | 1.0 | Contextual discounting Reconst |
| `UNCERTAINTY_MAXIMIZATION` | bool | False | Efface b, maximise u (Def. 3.6) |

---

## 5. Paramètres calibrés (chargés depuis le PKL `train_v10`)

| Clé PKL | Rôle | Fallback si absent |
|---------|------|--------------------|
| `empirical_priors` | EDP par métrique {a_safe, a_susp, a_atk} | Prior uniforme SL_PRIOR_A |
| `_decision_threshold` | Seuil δ calibré sur FPR cible | CONFIG['EVAL']['DECISION_THRESHOLD'] |
| `_decision_variable` | Variable de décision ('proj_atk') | 'proj_atk' |
| `trust_scores` | Scores de fiabilité par métrique | a_safe (EDP) |
| `reconst_attack_reliability` | Fiabilité Reconst pour "attack" | 1.0 (pas de discounting) |

---

## 6. Sorties produites

| Fichier | Colonnes clés | Notes |
|---------|---------------|-------|
| `detection_results.csv` | `{metric}_b_safe/b_susp/b_atk/u`, `{metric}_proj_safe/susp/atk`, `{metric}_ev_P/S/N`, `{metric}_conflict_K`, `{metric}_lambda_dyn`, `{metric}_a_safe/susp/atk` | Injection inactive |
| `detection_results_INJECTED.csv` | idem | Injection active (ATTACK_CATALOG=None) |
| `graph_{metric}.png` | — | Opinion SL (haut) + données brutes/prédiction (bas) |

**Convention de nommage CSV :** identique à `compute_evidence_v2.py` — noms de métriques
complets (`prophet_bytes`, `reconst_bytes`), `"->"` remplacé par `"_to_"`.

---

## 7. Hypothèses posées

| # | Hypothèse | Justification / Référence |
|---|-----------|--------------------------|
| H1 | Les preuves P/S/N sont **additives** sur une fenêtre | Resample `.sum()` — cohérent avec la définition multinomiale de l'évidence SL (Jøsang 2016, Def. 3.9) |
| H2 | Le prior EDP est **stationnaire** sur la durée d'inférence | EDP calculé à l'entraînement et figé dans le PKL |
| H3 | W = K = 3 est le **prior-weight canonique** pour θ = {Safe, Suspect, Attack} | Jøsang (2016) §3.5.2 : W = |θ| |
| H4 | Le RMSE de la fenêtre précédente est un proxy valide du régime courant (lag = 1 fenêtre) | Compromis contamination / réactivité — acceptable si fenêtre ≤ 5 min |
| H5 | `proj_atk` = b_atk + a_atk × u est la **variable de décision optimale** | Jøsang 2016, Eq. 3.23 ; calibration FPR dans train_v10 |
| H6 | Prophet et Reconst sont des **sources indépendantes** | Requis pour que CBF ≡ addition d'évidence (Jøsang 2016, Th. 12.2) — **non vérifié empiriquement** : les deux méthodes observent les mêmes flux réseau |

> **H6 — Point critique pour le rapport :** l'indépendance conditionnelle des deux
> méthodes n'est pas démontrée. Une analyse de corrélation de leurs sorties sur données
> normales (coefficient de Pearson ou information mutuelle sur b_atk) renforcerait la
> validité de la CBF inter-méthode.

---

## 8. Propriétés mathématiques vérifiées

### 8.1 Initialisation EDP (warm-up)

Avec `R_init = a_edp × W` (bijection Def. 3.9) :

```
sum(R_init) = W × sum(a_edp) = W × 1 = W
D = W + W = 2W
u_init = W / 2W = 0.5           (exact)
b_i    = (a_i × W) / 2W = a_i/2 (exact)
proj_i = b_i + a_i × u = a_i    (exact)
```

→ La probabilité projetée initiale est **exactement** égale au prior EDP.
Partant de l'opinion vacuouse (U=1) ignorerait ce prior et sur-exprimerait l'ignorance
pendant le warm-up.

### 8.2 CONFLICT_ALPHA — garantie de hard-reset

```
b_curr_max = WINDOW_SIZE / (WINDOW_SIZE + W)
b_prev_max = (2×WINDOW_SIZE) / (2×WINDOW_SIZE + W)
K_conflict_max = b_prev_max × b_curr_max
CONFLICT_ALPHA = 1 / K_conflict_max
```

Propriété : `λ_dyn = λ_base × max(0, 1 − α×K) = 0` exactement quand K = K_conflict_max.
Recalculé dynamiquement dans `config.py` selon WINDOW_SIZE et SL_PARAM_K actifs.

---

## 9. Points pour la rédaction du rapport

1. **Indépendance des sources (H6)** : justifier ou quantifier empiriquement la
   corrélation Prophet/Reconst sur trafic normal.

2. **Choix W=3** : W = |θ| = 3 est le choix canonique (Jøsang 2016, §3.5.2). Tout
   autre valeur modifie la vitesse de convergence vers l'évidence et doit être justifiée.

3. **`proj_atk` comme variable de décision** : préféré à `b_atk` car c'est une distribution
   continue et lisse (pas bimodale concentrée en 0). Le seuil δ est calibré sur
   `quantile(proj_atk, 1 − FPR_target)` sur trafic strictement normal (Ruff et al. 2021).

4. **Gating per-metric du RMSE** : le reset RMSE est déclenché par chaque métrique
   indépendamment (basé sur `proj_atk_leaf` de cette métrique, pas sur la décision système
   finale). Justification : Chandola et al. (2009), CSUR §3.1 — indépendance des détecteurs
   dans un ensemble hybride.

5. **Contextual discounting** : la fiabilité de la Reconst pour "attack" peut être calibrée
   automatiquement depuis l'artefact train_v10 (mode "auto"). Réf. : Mercier, Quost &
   Denoeux (2006/2008), ECSQARU / Information Fusion.

6. **Balance Ratio** : en mode "auto", le ratio effectif = N_prophet / N_reconst corrige
   le biais d'accumulation de preuves de la source dominante avant CBF. La source dominante
   est **réduite** (multipliée par 1/ratio), pas la source minoritaire amplifiée.

---

## 10. Dépendances

| Module | Usage |
|--------|-------|
| `sl_formulas_v2.py` | Opérateurs SL : `evidence_to_opinion`, `temporal_adaptive_ageing`, `fusion_wbf_n_sources`, `fusion_cbf`, `boost_opinion_evidence`, `apply_trust_discount`, `apply_contextual_discount` |
| `compute_evidence_v2.py` | Producteur des CSV d'entrée (evidence + metadata + raw_data) |
| `train_v10.py` | Producteur du PKL (EDP, seuil calibré, trust scores) |
| `config.py` | Tous les hyperparamètres + calcul dynamique CONFLICT_ALPHA |
| `paths.py` | `get_version_names`, `get_model_path` |

---

## 11. Corrections appliquées lors de la revue (2026-04-11)

| ID | Nature | Description |
|----|--------|-------------|
| B1 | Bug latent | `get_c3_weight` + bloc `online_rmse` utilisaient `key` brut pour lire `_iw`/`_rmse` au lieu de `clean_key` (mismatch avec convention `compute_evidence_v2`) |
| B2 | Commentaire trompeur | Description BALANCE_RATIO inversée — corrigée (source dominante réduite, pas minoritaire amplifiée) |
| B3 | Warning manquant | Absence de message si PKL introuvable — tous les paramètres calibrés tombaient silencieusement sur défauts |
| S1 | Formule incomplète | Print `λ_dyn = λ_base × (1-K)` ne mentionnait pas le facteur α |
| S2 | Problème méthodologique | Reset RMSE global sur toutes les métriques lors d'une décision système → refactorisé en **per-metric** (basé sur `proj_atk` de chaque métrique individuellement) |
| S3 | Justification insuffisante | CONFLICT_ALPHA commenté avec ancienne formule erronée — remplacé par la dérivation exacte depuis config.py |
| S5 | Code mort | `METRIC_WEIGHTS` construit mais jamais utilisé — supprimé (trace en commentaire) |
| F1 | Docstring | Nom de fichier `v2` → `v3`, description des niveaux complétée |
| F2 | Références obsolètes | `train_v9` → `train_v10` (3 occurrences) |
| F3 | Prints | "V2" → "V3" (2 occurrences) |
| F4 | CSV hardcodé | `detection_results_INJECTED.csv` toujours → nom dynamique selon `_has_injection` |
| F5 | Commentaire mort | `#proj_atk` vestige de toggle supprimé |
| F6 | Structure | `_freq_to_seconds` défini localement → déplacé au niveau module |
| F7 | Nommage CSV | `prophet_` → `P_`, `reconst_` → `R_` dans `save_results_to_csv` supprimés — convention unifiée avec `compute_evidence_v2` |
| EDP | Commentaire inexact | `≈` → `=` (propriétés exactes démontrées) + dérivation complète ajoutée |
| proj_atk | Simplification | Logique de décision `b_atk / proj_atk` selon `_decision_variable` → toujours `proj_atk` (cohérent avec calibration train_v10) |
