# Revue scientifique et technique — `evaluate_injection_v2.py`

> Révision du : 2026-04-11  
> Contexte : préparation à l'évaluation par un jury scientifique  
> Fichier principal : `evaluate_injection_v2.py`  
> Dépendances directes : `inject_at_evidence_level.py`, `config.py`, `paths.py`, `compute_opinions_v3.py` (via le CSV résultat)

---

## 1. Rôle du module dans la chaîne SL-ADS

`evaluate_injection_v2.py` est le **module d'évaluation quantitative** de la chaîne
SL-ADS, exécuté après `compute_opinions_v3.py` sur le CSV de détection produit
depuis l'evidence injectée. Il mesure les performances de détection du système
sur des attaques **synthétiques contrôlées** (catalogue défini dans
`inject_at_evidence_level.py`) à travers quatre axes d'analyse :

| Axe | Mesure |
|---|---|
| Axe 1+2 | Détection (TP/FP/FN) + Time-To-Detect par attaque × seuil |
| Axe 3 | Audit du taux de base adaptatif `a_atk` avant / pendant / après attaque |
| Axe 4 | Comparaison R1 vs R2+ (apprentissage par répétition, SL adaptatif) |
| Global | Balayage précision / rappel (3 variantes) / F1 en fonction du seuil |

La colonne de décision évaluée est `COL_DET` (par défaut `FINAL_SYSTEM_CBF_proj_atk`
ou `FINAL_SYSTEM_CBF_b_atk`), lue dynamiquement depuis `paths.get_detection_col()`.

---

## 2. Fonctions et outils employés

| Fonctionnalité | Implémentation | Référence |
|---|---|---|
| Vérité terrain (GT) | Horodatages du catalogue → masque booléen | — |
| Détection par seuil | `COL_DET >= threshold` | Sharafaldin et al. 2018 |
| TTD observé | Premier franchissement du seuil dans la fenêtre GT | — |
| TTD théorique | Simulation analytique de l'accumulation exponentielle SL | Jøsang 2016, Eq. 16.5 |
| TTD gap | `ttd_windows − ttd_theo_win` — mesure de l'effet WBF/CBF | dérivation locale |
| Rappel binaire | Attaque détectée / total attaques | IDS standard |
| Rappel couverture | Moyenne des `coverage_pct` par attaque / 100 | Ferling et al. 2022 |
| Rappel TTD-pénalisé | `cov × (1 − TTD/durée)` — dérivé formellement dans le code | dérivation locale |
| Couverture plateau | Détection restreinte aux fenêtres alpha ≥ 0.8 (phase plateau) | inject_at_evidence_level |
| Précision fenêtre | `TP_fenêtres / (TP_fenêtres + FP_fenêtres)` | — |
| F1 (5 variantes) | binaire, couverture, TTD-pénalisé, micro-pur, macro-pur | — |
| MCC | Matthews Correlation Coefficient | Chicco & Jurman 2020 |
| Point opérationnel ROC | (TPR, FPR) à fenêtre-niveau au seuil opérationnel | — |
| Audit base rate | Évolution de `{metric}_a_atk` sur ±CONTEXT_H | SL Jøsang §12 |
| Analyse b_susp | `mean_b_susp_during`, `max_b_susp_during` par attaque | SL Jøsang §3 |
| Cohérence GT | Vérification `injection_label` ↔ timestamps | dérivation locale |
| Cohérence WINDOW_MIN | Vérification `WINDOW_SIZE × FREQ = WINDOW_MIN × 60s` | — |
| Apprentissage répété | Comparaison R1→R2+ TTD et couverture | SL adaptatif |
| Figures | matplotlib 300 dpi, style publication | — |

---

## 3. Hypothèses posées

### 3.1 Hypothèses sur la vérité terrain

- **H1 – Vérité terrain définie par les horodatages du catalogue** : toute fenêtre
  temporelle dont le timestamp tombe dans `[start, start + duration_h[` est
  considérée "attaque". Cela inclut les phases de ramp (alpha < 1) où l'évidence
  injectée est encore partielle — la GT est donc légèrement généreuse vis-à-vis
  du signal réel. La métrique `coverage_plateau_pct` permet de mesurer séparément
  la détection en phase plateau (alpha ≥ 0.8) et de dissocier les deux effets.

- **H2 – Indépendance entre injection et évaluation** : la vérité terrain est
  recalculée depuis les timestamps du catalogue, indépendamment de la colonne
  `injection_label` écrite par `inject_at_evidence_level.py`. Une vérification de
  cohérence (P2) signale toute divergence. Note : `compute_opinions_v3.py` ne
  préserve pas `injection_label` dans le CSV de détection (il reconstruit un
  DataFrame from scratch), donc la vérification est faite uniquement si la colonne
  est présente.

- **H3 – Non-chevauchement des attaques** : le validateur `_check_no_overlap()`
  de `inject_at_evidence_level.py` garantit l'absence de recouvrement. L'évaluateur
  ne le revalide pas — il fait confiance à la garantie amont.

### 3.2 Hypothèses sur les métriques de performance

- **H4 – Précision fenêtre × rappel attaque (hybride intentionnel)** : la précision
  est calculée au niveau de la **fenêtre** (`TP_win / (TP_win + FP_win)`) car chaque
  fenêtre falsement détectée est une fausse alarme opérationnelle (coût SOC). Le rappel
  est calculé au niveau de l'**attaque** car opérationnellement, ce qui compte est
  de savoir si un épisode d'attaque a été détecté (pas combien de fenêtres). Cette
  asymétrie est documentée dans le code par un commentaire formel et est cohérente
  avec Sharafaldin et al. (2018). Les métriques pures fenêtre-niveau (`f1_micro_pure`,
  `f1_macro_pure`) sont fournies en parallèle pour comparaison directe avec la
  littérature ML.

- **H5 – TTD-rappel : formule originale dérivée formellement** :
  `recall_ttd_i = cov_i × (1 − TTD_i/duration_i)`  
  `recall_ttd = mean(recall_ttd_i)` sur toutes les attaques.  
  Propriétés : score_i ∈ [0,1], = 1 si détection immédiate à couverture totale,
  = 0 si manquée ou détectée exactement à la fin. TTD > durée est plafonné à 0.
  Analogie avec Tatbul et al. (2018) : ExistenceReward × OverlapReward, étendu
  d'une pénalité temporelle. La dérivation est documentée en commentaire dans le code.

- **H6 – TTD théorique = modèle à une seule métrique (borne inférieure)** :
  `theoretical_ttd_windows` simule l'accumulation SL sur la métrique la plus
  discriminante (max ev_attack normalisé). Le système réel fusionne N métriques
  via WBF+CBF. Le gap `ttd_gap_windows = ttd_windows − ttd_theo_win` mesure
  l'effet net de la fusion multi-métriques : positif = fusion ralentit (métriques
  contradictoires), négatif = fusion accélère (métriques concordantes).

- **H7 – WINDOW_MIN cohérence vérifiée automatiquement** : `WINDOW_MIN` est
  comparé à `WINDOW_SIZE × FREQ / 60` au démarrage. En cas d'écart, un warning
  est émis et la valeur est corrigée automatiquement.

### 3.3 Hypothèses sur le système SL évalué

- **H8 – UNCERTAINTY_MAXIMIZATION** : lu depuis `CONFIG` (affichage uniquement).
  N'affecte aucun calcul dans ce module — le mode UM est contrôlé en amont dans
  `run_ablation.py`.

- **H9 – Seuil opérationnel unique** : `THRESHOLDS = [_decision_thr]` (issu du
  calibrage EVT/FPR à l'entraînement). Le balayage de seuils est prévu mais la
  figure correspondante est bypassée si `len(THRESHOLDS) == 1`.

---

## 4. Sorties produites

| Fichier / Figure | Contenu | Usage |
|---|---|---|
| `eval_detection_summary.csv` | Par attaque × seuil : détection, coverage, coverage_plateau, TTD, TTD-gap, b_susp, FPR | Table principale article |
| `eval_threshold_sweep.csv` | Précision, rappel (3), F1 (5), MCC, TPR/FPR, point ROC par seuil | Figures comparaison |
| `eval_baserate_audit.csv` | `a_atk` avant / pendant / après par attaque × métrique | Audit adaptatif |
| `eval_learning_comparison.csv` | Comparaison R1 vs R2+ par famille (si _R2 présents) | Axe apprentissage |
| `graphs/attack_{NAME}.png` | Timeline : `b_atk`, `b_susp`, `b_safe`, seuil, GT | Figure par attaque |
| `graphs/threshold_sweep.png` | Précision/Rappel/F1/FPR vs seuil (si > 1 seuil) | Courbe globale |
| `graphs/summary_table.png` | Tableau récapitulatif article-ready (PNG 300 dpi) | Insert article |
| `graphs/baserate_{atk}_{metric}.png` | Évolution `a_atk` autour d'une attaque | Axe 3 audit |
| `graphs/learning_{family}.png` | Profils R1/R2+ sur axe temps normalisé (si _R2 présents) | Axe 4 répétition |

---

## 5. Paramètres de configuration impliqués

| Paramètre | Source | Rôle | Valeur typique |
|---|---|---|---|
| `COL_DET` | `paths.get_detection_col()` | Colonne de décision évaluée | `FINAL_SYSTEM_CBF_proj_atk` |
| `_decision_thr` | `paths.get_decision_threshold()` | Seuil calibré à l'entraînement | ~0.20 |
| `WINDOW_MIN` | `CONFIG["EVAL"]["WINDOW_MIN"]` → vérifié vs `WINDOW_SIZE×FREQ` | Durée d'une fenêtre en minutes | 5 |
| `CONTEXT_H` | `CONFIG["EVAL"]["CONTEXT_H"]` | Contexte avant/après attaque pour figures | 2.0 |
| `LAMBDA_DECAY` | `CONFIG["LAMBDA_DECAY"]` | Décroissance exponentielle SL | 0.85 |
| `LEAF_METRICS_TO_AUDIT` | `CONFIG["EVAL"]` ou défaut | Métriques auditées sur `a_atk` | liste de 10 |
| `CATALOG_MODE` | `CONFIG["EVAL"]["CATALOG_MODE"]` | Source du catalogue : "injected" ou "real" | "injected" |
| `INCLUDE_REAL_ATTACK` | `CONFIG["EVAL"]` | Ajouter des attaques réelles au catalogue injecté | True |
| `RESULTS_CSV_NAME` | `CONFIG["EVAL"]["RESULTS_CSV_NAME"]` | Nom du CSV de détection à évaluer | `detection_results_INJECTED.csv` |

---

## 6. Problèmes résolus (corrections appliquées)

| ID | Sévérité initiale | Description | Statut |
|---|---|---|---|
| B1 | Majeur | `_EVAL.get("dazdaz", ...)` → `_EVAL.get("RESULTS_CSV_NAME", ...)` | ✅ Corrigé |
| B2 | Majeur | Seuil `0.20` hardcodé → `_decision_thr` dans `plot_learning_comparison` | ✅ Corrigé |
| B3 | Moyen | `global_threshold_sweep` utilise `valid_catalog` filtré au lieu du global | ✅ Corrigé |
| B4 | Majeur | TTD théorique normalisé (WINDOW_SIZE scale) — cohérent avec l'injection réelle | ✅ Corrigé |
| F2 | Mineur | `UNCERTAINTY_MAXIMIZATION` lu depuis config au lieu d'être hardcodé | ✅ Corrigé |
| P2 | Moyen | Vérification cohérence `injection_label` ↔ GT timestamps au démarrage | ✅ Corrigé |
| P3 | Moyen | Vérification `WINDOW_SIZE × FREQ = WINDOW_MIN × 60s` avec auto-correction | ✅ Corrigé |
| S1 | Majeur | Justification formelle de la métrique hybride précision-fenêtre × rappel-attaque dans le code | ✅ Documenté |
| S2 | Majeur | Dérivation formelle de `recall_ttd` avec cas limites et relation Tatbul 2018 | ✅ Documenté |
| S3 | Moyen | Colonne `ttd_gap_windows` ajoutée pour analyser l'effet de la fusion WBF/CBF | ✅ Ajouté |
| S4 | Majeur | MCC ajouté dans `global_threshold_sweep` et rapport console | ✅ Ajouté |
| S5 | Moyen | Axe 4 : message explicatif sur la mécanique SL et comment l'activer | ✅ Documenté |
| S6 | Moyen | `mean_b_susp_during` et `max_b_susp_during` ajoutés dans `eval_detection_summary` | ✅ Ajouté |
| S7 | Moyen | `coverage_plateau_pct` (alpha ≥ 0.8) ajouté — analyse stratifiée ramp vs plateau | ✅ Ajouté |
| S8 | Moyen | `WINDOW_MIN` auto-corrigé depuis `WINDOW_SIZE × FREQ` (= P3) | ✅ Corrigé |

---

## 7. Points ouverts (non bloquants pour le jury, à mentionner)

### 7.1 Axe 4 inactif sur le catalogue courant

Le catalogue de 9 attaques ne contient aucune occurrence `_R2`. L'axe est
prévu pour des extensions futures et le message console l'explique maintenant
explicitement avec la mécanique SL sous-jacente (mise à jour EDP après R1).

**Recommandation pour le rapport** : indiquer que l'axe 4 est implémenté et
fonctionnel, mais nécessite l'injection de doublons temporels pour être activé.
Citer Jøsang (2016) §12 sur la mise à jour du prior Dirichlet.

### 7.2 Absence de courbe ROC/AUC complète

Avec un seuil unique (`len(THRESHOLDS) == 1`), il est impossible de tracer
une courbe ROC ou Précision-Rappel complète. Le **point opérationnel ROC**
(TPR, FPR) est maintenant reporté dans `eval_threshold_sweep.csv`.

**Pour un article** : soit activer le balayage de seuils pour obtenir l'AUC-ROC
et AUC-PR, soit comparer le point opérationnel à des baselines publiées.

### 7.3 `b_susp` non exploité dans les figures

`b_susp` est tracé dans les timelines mais `mean_b_susp_during` n'est pas
encore utilisé dans la figure de synthèse `summary_table.png`. Pour les attaques
de faible intensité, un `b_susp` élevé est un signal d'incertitude SL pertinent
à commenter.

### 7.4 `injection_label` non transmis par `compute_opinions_v3.py`

Le CSV de détection ne contient pas `injection_label` (compute_opinions reconstruit
son DataFrame from scratch). La vérification P2 le détecte et signale que c'est
le comportement attendu. Si l'on souhaitait propager `injection_label`, il faudrait
modifier `save_results_to_csv` dans `compute_opinions_v3.py` pour joindre les
colonnes de métadonnées d'injection depuis le CSV evidence.

---

## 8. Métriques produites dans `eval_detection_summary.csv`

| Colonne | Description |
|---|---|
| `name`, `family`, `occurrence` | Identité et structure de l'attaque |
| `type`, `intensity`, `duration_h` | Caractéristiques catalogue |
| `n_gt_windows` | Nombre de fenêtres dans la fenêtre de vérité terrain |
| `threshold` | Seuil de décision évalué |
| `detected` | L'attaque a-t-elle déclenché au moins une alarme ? |
| `n_detected`, `coverage_pct` | Couverture globale (toutes phases) |
| `coverage_plateau_pct` | Couverture restreinte aux fenêtres plateau (alpha ≥ 0.8) |
| `ttd_windows`, `ttd_minutes` | Temps à la première détection |
| `ttd_theo_win` | TTD théorique (modèle mono-métrique normalisé) |
| `ttd_gap_windows` | `ttd_windows − ttd_theo_win` : effet de la fusion |
| `max_b_atk`, `mean_b_atk` | Amplitude de la masse de croyance attack |
| `mean_b_susp_during`, `max_b_susp_during` | Masse suspecte pendant l'attaque |
| `fp_outside`, `n_outside_win`, `fpr_pct` | Faux positifs hors fenêtres d'attaque |

## 9. Métriques produites dans `eval_threshold_sweep.csv`

| Colonne | Description |
|---|---|
| `recall_binary` | Proportion d'attaques détectées (0/1 par attaque) |
| `recall_coverage` | Moyenne des taux de couverture sur les attaques |
| `recall_ttd` | Rappel couverture × pénalité temporelle TTD |
| `precision_window` | Précision au niveau fenêtre |
| `f1_binary`, `f1_coverage`, `f1_ttd` | F1 pour chacun des 3 rappels |
| `f1_micro_pure` | F1 micro (= F1 classe positive) au niveau fenêtre |
| `f1_macro_pure` | F1 macro (moyenne des deux classes) au niveau fenêtre |
| `mcc` | Matthews Correlation Coefficient |
| `tpr_window`, `fpr_window` | Point opérationnel ROC à fenêtre-niveau |
| `accuracy` | Exactitude (trompeuse si déséquilibre — fournie par complétude) |

---

## 10. Points forts à valoriser dans le rapport

- **Trois variantes de rappel** (binaire, couverture, TTD-pénalisé) offrent une
  vision complète de la qualité de détection au-delà du simple 0/1.
- **MCC** : métrique scalaire recommandée pour les classes déséquilibrées
  (Chicco & Jurman 2020), absent de la plupart des évaluations IDS publiées.
- **TTD gap** : analyse de l'effet de la fusion multi-métriques WBF+CBF sur le
  délai de détection — contribution originale.
- **Couverture plateau vs totale** : distingue les défaillances en phase de montée
  des défaillances en régime permanent — pertinent pour les attaques lentes.
- **Audit du taux de base adaptatif** (Axe 3) : vérifie que l'apprentissage SL ne
  s'empoisonne pas durablement après une injection (résilience post-attaque).
- **Catalogue de 9 attaques typées** couvrant les principales familles réseaux
  (DDoS volumétriques, DoS lents, scan, exfiltration, amplification), chacune
  documentée avec références primaires (CIC-IDS2017, UNSW-NB15, Kitsune, Cloudflare).
- **Indépendance injection/évaluation** : la vérité terrain est recalculée depuis
  les timestamps — pas de circularité.
- **Seuil opérationnel calibré a priori** (EVT/FPR à l'entraînement) : évaluation
  strictement hors-échantillon.
- **Figures publication-ready** : 300 dpi, axes LaTeX, grille légère, légendes
  complètes, couleurs accessibles (palette matplotlib standard).

---

## 11. Références citées dans le fichier

| Référence | Contribution à ce module |
|---|---|
| Sharafaldin et al. (2018) ICISSP | Framework TP/FP/FN IDS, dataset CIC-IDS2017, justification précision fenêtre |
| Jøsang (2016) Subjective Logic, Chap. 12 | Accumulation exponentielle Eq. 16.5 — TTD théorique ; mise à jour EDP Axe 4 |
| Tatbul et al. (2018) NeurIPS | Precision & Recall for Time Series — analogie avec recall_ttd |
| Ferling et al. (2022) | Time-Aware Evaluation — coverage-weighted recall |
| Chicco & Jurman (2020) BMC Genomics | MCC — justification pour classes déséquilibrées |
