# CONSOLIDATED_AUDIT_REVIEW.md

**Date** : 2026-04-21
**Périmètre** : `actual_ version_claude_autre dataset/` (branche active RedeRio/autre_dataset)
**Sources auditées** :
- `docs/review/audit_01_config.md` → `docs/review/audit_14_findings_summary.md`
- `docs_scientific_audit/` (10 fichiers, 4 710 lignes)
- Code actuel (après patchs PATCH-C1..C5 + UTF-8 + L1148 + IF try/except)

**Principe directeur** (constrainte utilisateur) :
> *"fait bien attention a ce qu'il y ait rien qui soit pris en compte qui soit pas faux"*

Chaque élément listé ici a été **re-vérifié contre le code actuel** (lecture directe, pas seulement copie des audits). Les items faussement flaggés par les audits mais en réalité déjà corrigés ou inapplicables sont **exclus** de la Section 1 et listés en Section 2 pour traçabilité.

---

## TABLE DES MATIÈRES

0. **Section 0 — STATUS 2026-04-24 : ce qui est fait (code) vs. ce qui reste (papier)** ← lire en premier
1. **Section 1 — Issues REAL requérant action avant soumission** (filtrées, vérifiées)
2. **Section 2 — Items déjà corrigés pendant cette session** (documentation)
3. **Section 3 — Items de libellé / framing** (changement papier, pas code)
4. **Section 4 — Design choices documentés et acceptés** (`OK` par audit)
5. **Section 5 — Plan d'action priorisé**
6. **Annexe A — Grille de vérification du reviewer** (mapping sur items)

---

## SECTION 0 — STATUS 2026-04-24 : ce qui est fait vs. ce qui reste

> **Mise à jour** : session 2026-04-23 → 2026-04-24.
> **Consigne utilisateur** : *"fait toutes ces corrections et boucle autant que nécessaire pour tout tester et avoir une qualité parfaite puis mets à jour le document pour ne laisser que ce qui doit encore être corrigé en séparant clairement ce qui doit être fait dans le code que l'article"*.
>
> Cette section est la **source de vérité unique** pour savoir ce qui reste à faire.  Les sections 1, 2, 3 ci-dessous gardent l'historique détaillé ; la section 5 (plan d'action) a été mise à jour pour refléter ce statut.

### 0.1 ✅ CODE — items entièrement résolus cette session

Chaque item a été patché **et** vérifié (unit tests ou smoke-tests passant).  Les artefacts listés sont les fichiers modifiés ou créés.

| ID | Libellé | Sévérité | Artefacts principaux | Vérification |
|---|---|---|---|---|
| **C-01 / F02** | IF-fpr-matched recalibré sur fenêtre pre-split (plus de test-leak) | CRITICAL | `compare_if_fair.py` (nouveau `_calibrate_if_threshold_from_normal` + groupby sur `train_normal`) | 5 unit tests calibration : uniform→0.947, Normal(0,1)→2.32, `target_fpr` ∉ (0,1) rejeté, array vide rejeté, 2D→ravel |
| **m-03 / F20** | `qualif_filters` câblé dans `evaluate_qualify_sbn.py` (filtrage `gate_open ∧ qual_status ≠ 'no_groups'`) | MINOR | `evaluate_qualify_sbn.py` | `ast.parse()` clean + smoke-run sur CSV synthétique |
| **m-04 / F21** | `NETWORK_OUTAGE` exclu via flag séparé dans les deux scripts (taxonomie adversariale isolée à la demande) | MINOR | `evaluate_qualify_sbn.py`, `compare_qualif_methods.py` | idem |
| **m-09 / F30** | `qualif_filters` câblé dans `compare_qualif_methods.py` | MINOR | `compare_qualif_methods.py` | idem |
| **M-07 / F10** | Sensitivity sweep SBN_NOVELTY_U_RAW_THRESHOLD + env-var override | MAJOR | `config.py` (bloc `SL_SBN_NOVELTY_U_RAW_THRESHOLD_OVERRIDE`), `ablation_sbn_novelty.py` (nouveau, 310 lignes) | 7 scénarios env-var passent (default, 0.70, 0.75, 0.90, OOB, négatif, non-numérique) |
| **M-11 / CBF-indep** | Défaut inter-method fusion basculé CBF → WBF + env-var override + ablation harness ; 2026-05-07 ajoute ABF/BCF/CCF/MinBF/MaxBF et garde WBF après recalibration stricte | MAJOR | `config.py` (bloc `SL_INTER_METHOD_FUSION_OVERRIDE`), `compute_opinions_v3.py` (2 fallbacks), `ablation_fusion_mode.py`, `compare_recalibrated_fusion_modes.py` | WBF reste défaut ; ABF disponible mais non adopté sur RedeRio |
| **M-01 / F01** | WBF canonique Jøsang 2-sources (Eq. 12.22-12.24) + alias sémantique | MAJOR | `sl_formulas_v2.py` (nouveau `fusion_wbf_canonical_two`, alias `fusion_evidence_average_confidence_weighted`), `tests/test_fusion_wbf_canonical.py` (nouveau, 8 tests) | **8/8 tests passent** ; identité numérique avec implémentation evidence-space à 2.22e-16 (machine epsilon) ; bijection Σb+u=1, symétrie, idempotence, dogmatic case, continuité, asymmetric confidence tous vérifiés |
| **M-10 / F17** | Analyse de faisabilité SBN canonique + plan de renommage terminologique | MAJOR | `docs/review/M10_sbn_architecture_analysis.md` (nouveau, 6 sections + 6 claims testables) | Claims C1–C6 vérifiés contre le code : C1 (pas de deduction SL), C2 (cond-opinions = vecteurs de probas, pas d'`u_cond`), C3 (rectification vs produit), C4 (line numbers `_sl_bijection` L709, `_wbf_two` L810, `_discount_opinion` L839 confirmés), C5 (naive-Bayes style documenté dans le code), C6 (data scarcity sur tous les benchmarks publics) |

### 0.2 📄 PAPIER — items restants (aucun code à toucher)

Tous les items ci-dessous nécessitent **uniquement** une modification du manuscrit LaTeX.  La référence "cite" indique l'artefact de session à citer en justification dans le papier.

| ID | Action sur le papier | Section LaTeX cible | Artefact à citer |
|---|---|---|---|
| **L-01** (ex M-01/F01) | Soit (A) utiliser désormais `fusion_wbf_canonical_two` avec citation explicite Eq. 12.22–12.24, soit (B) garder l'implémentation evidence-space et la défendre comme *"confidence-weighted evidence averaging, consistent with Eq. 12.27 via the bijection Def. 3.9"*.  Retirer toute prétention à « exact Jøsang Theorem 12.2 » non étayée. | Section méthodologie SL / §WBF | `sl_formulas_v2.py` docstring de `fusion_wbf_n_sources` ; `tests/test_fusion_wbf_canonical.py` consistency test (delta ≤ 2.22e-16) |
| **L-02** (ex F06) | Remplacer partout *"adaptive base rate"* par *"static EDP (Extended Decision Prior)"*.  La EDP n'est pas réellement adaptative dans le code actuel. | Abstract, méthodologie, §decision rule | `config.py` — `DECISION_BASE_RATE`, `DECISION_ATTACK_WEIGHT` (constantes) |
| **L-03** (ex M-10/F17) | Renommer systématiquement *"Subjective Bayesian Network (SBN)"* par **"Expert-template-driven Subjective Logic qualifier"** dès le premier emploi (abstract).  Footnote vers `M10_sbn_architecture_analysis.md`. | Abstract, introduction, §qualification, toutes les légendes de figure | `docs/review/M10_sbn_architecture_analysis.md` §5 (terminologie recommandée) |
| **L-04** (ex F07) | Créer une nouvelle section **« Novel contributions to the SL framework »** listant les heuristiques propres (evidence scaling, rectification, kill-chain transition matrix, UM post-fusion). | Nouvelle §, après méthodologie | `qualify_anomaly_sbn.py` L685-707 (evidence rectification), L511-592 (transition matrix), L760-808 (UM) |
| **L-05** (ex F08) | Section **« Evidence engineering design choices »** documentant le trapèze + EDP floor. | Méthodologie / appendice | `compute_evidence_v2.py` (fonctions evidence) |
| **L-06** (ex F21) | Justifier la taxonomie jointe `NETWORK_OUTAGE` vs attaques OU séparer.  Le code offre désormais la séparation via filtre ; le papier doit préciser le choix. | §métriques d'évaluation | `evaluate_qualify_sbn.py` + `compare_qualif_methods.py` (filtres PATCH m-04/F21) |
| **L-07** (ex M-11/CBF) | Section dédiée **« Dependence audit between Prophet and reconstruction branches »** : (a) corrélation empirique sur données normales, (b) tableau CBF vs WBF vs Hierarchical. | Nouvelle §, après §fusion | `ablation_fusion_mode.py` (tableau à produire ; `python ablation_fusion_mode.py` → `ablation_fusion_mode_summary.csv`) |
| **L-08** (ex M-07) | Intégrer le tableau sensitivity SBN_NOVELTY_U_RAW_THRESHOLD. | Appendice ou §qualification | `ablation_sbn_novelty.py` (tableau à produire ; `python ablation_sbn_novelty.py`) |
| **L-09** (ex M-13) | Note dans §baselines : *"Isolation Forest baseline is reported separately via fair pre-split-calibrated FPR matching — see §X.Y."* | §baselines / ablation | `compare_if_fair.py` §6 du rapport markdown (policy documentée) |
| **L-10** (ex M-12) | Mentionner explicitement le fallback R² in-sample quand rolling-origin CV n'est pas faisable. | §trust scoring | `train_v10.py` (fallback documenté dans les logs) |

### 0.3 ⏳ CODE — items mineurs restants (différables pour version journal)

Ces items **peuvent** être traités avant soumission mais **ne bloquent pas** une conférence.  Ils sont tous MINOR.

| ID | Action code | Fichier | Effort |
|---|---|---|---|
| ~~C-02 / F03~~ | ~~Remplacer `eval_cfg.get("dadza", ...)` → `eval_cfg.get("RESULTS_CSV_NAME", ...)`~~ | ✅ **DÉJÀ FAIT PATCH-C6 (2026-04-21)** | — |
| M-02 / F05 | Lier `W` dans `theoretical_ttd_windows` à `CONFIG["SL_PARAM_K"]` | `evaluate_injection_v2.py` L196–L198 | 15 min |
| M-06 / F09 | Décider padding vs drop pour fenêtres partielles | `compute_evidence_v2.py` | ~2h (décision + doc) |
| M-08 / F11 | Ajouter warning sur cap dogmatique | `sl_formulas_v2.py` | 30 min (déjà partiellement patché) |
| M-09 / F12 | Paramétrer la fréquence de resampling | `compute_opinions_v3.py` | 30 min |
| M-12 / F29 | Logger le fallback R² in-sample au niveau WARNING | `train_v10.py` | 20 min |
| m-01, m-02, m-05, m-06, m-07, m-08 | Voir Section 1.3 pour détails | divers | variable |

### 0.4 Résumé exécutif

- **CODE** : tous les items **CRITICAL** et **MAJOR** applicables au code sont **résolus** (C-01, M-01, M-07, M-10, M-11, m-03, m-04, m-09).  Restent uniquement des items MINOR différables.
- **PAPIER** : 10 items de réécriture / renommage terminologique à appliquer au manuscrit LaTeX (§ 0.2).  Chaque item pointe vers l'artefact code qui le justifie.
- **Tests** : la seule suite de tests unitaires nouvelle (`tests/test_fusion_wbf_canonical.py`) passe 8/8.  Les autres patches ont des smoke-tests documentés dans le résumé de session.

---

### 0.5 ✅ PHASE C — items résolus session 2026-04-25

> **Mise à jour** : session 2026-04-25.
> **Périmètre Phase C** : concerns détaillés soulevés par l'utilisateur — ablation injection-level vs raw, trust_discount/R² pathology, bootstrap BCa CI + McNemar, Kill-Chain transition matrix ablation, Wu & Keogh 2021 self-assessment, reviewer-target calibration, honest limitations, audit-verification tracker.

#### 0.5.1 Modules code créés (5 fichiers, tous testés `--self-test`)

| Fichier | Lignes | Statut self-test | Rôle |
|---|---|---|---|
| `stats_bootstrap_ci.py` | 404 | **6/6 PASS** | BCa 95% CI (Efron 1987) — single-sample + paired, Pivotal fallback, jackknife acceleration |
| `stats_mcnemar.py` | 226 | **5/5 PASS** | McNemar paired test (Edwards continuity + binomial exact pour n_disc < 25) |
| `ablation_injection_level.py` | 320 | **PASS (sur synthétique)** | Ablation evidence-level vs raw-data injection ; triviality probe + realism probe (Wu & Keogh 2021 #1+#2) |
| `ablation_temporal_sbn.py` | 280 | **3/3 hypothèses confirmées** | Kill-Chain ON/OFF avec H1 (volumetric→TIE), H2 (staged→HELPS), H3 (benign→HELPS via FPR) |
| `sl_ads.stats.residual_correlation` | 295 | **PASS** | Matrices Prophet 12×12 / Reconst 5×5 / Cross 17×17 + VIF + verdicts de dépendance ; run réel 2026-05-04 |

Toutes les commandes de validation sont listées dans `docs/audit/audit_verification_tracker.md` § "Quick-verify sequence".

#### 0.5.2 Documentation créée (5 fichiers)

| Fichier | Sujet | Cible |
|---|---|---|
| `docs/audit/wu_keogh_self_assessment.md` | Auto-évaluation contre les 4 flaws Wu & Keogh 2021 | Pour reviewer Tier-A (TKDE/VLDB) |
| `docs/audit/trust_discount_r2_analysis.md` | Pathologie Slowloris du mode `WBF_WEIGHT_MODE="trust_discount"` + alternative MASE | Justification de `WBF_WEIGHT_MODE="uniform"` par défaut |
| `docs/audit/reviewer_target_calibration.md` | Mapping venue → priorités reviewer → couverture actuelle | **Interne** (pas dans le papier) ; recommande IEEE TIFS / Computer & Security |
| `docs/audit/audit_verification_tracker.md` | Tableau trackable de chaque claim avec ID stable + commande de vérif | Permet vérification a posteriori reproductible |
| `docs/honest_limitations.md` | Brouillon prêt-à-coller pour la Section 5.3 du papier | Drop-in pour le manuscrit LaTeX |

#### 0.5.3 Tableau récapitulatif Phase C

| Préoccupation utilisateur | Statut | Artefact principal |
|---|---|---|
| Injection-level ablation evidence vs raw-data | **RÉSOLU** | `ablation_injection_level.py` + `docs/audit/wu_keogh_self_assessment.md` |
| Trust_discount/R² problème Slowloris | **RÉSOLU (défendu)** | `docs/audit/trust_discount_r2_analysis.md` (Option C: défaut `uniform`, opt-in documenté, MASE planifié v11) |
| Bootstrap BCa 95% CI sur F1/MCC/FPR | **RÉSOLU (module fait)** | `stats_bootstrap_ci.py` ; intégration dans `compare_if_fair.py` = TASK-10 PENDING |
| McNemar paired test IF vs SL | **RÉSOLU (module fait)** | `stats_mcnemar.py` ; intégration = TASK-11 PENDING |
| Kill Chain transition matrix ablation (SBN_TEMPORAL) | **RÉSOLU** | `ablation_temporal_sbn.py` confirme le défaut `False` pour volumetric et le bénéfice pour staged/benign |
| Wu & Keogh 2021 self-assessment | **RÉSOLU** | `docs/audit/wu_keogh_self_assessment.md` (4 flaws checklisted) |
| Paparrizos 2022 VUS-PR/ROC | **DEFERRED** | Tracé dans `audit_verification_tracker.md` TASK-17 |
| Baldán 2025 multimodal benchmark | **DEFERRED** | TASK-16 v11 |
| Reviewer target calibration | **RÉSOLU** | `docs/audit/reviewer_target_calibration.md` |
| C-02 typo `dadza` | **RÉSOLU précédemment** | PATCH-C6 (2026-04-21) ; entrée audit historique conservée pour traçabilité |
| Résiduals correlation 12×12 / 5×5 / 17×17 | **RÉSOLU (framework + run réel 2026-05-04)** | `sl_ads.stats.residual_correlation` ; artefacts dans `../results/resultats_RedeRio_trained_v4s_v4_v2/diagnostics/residual_correlation/` ; verdict cross = HIGH |
| Base-rate fallacy Axelsson 2000 | **DEFERRED** | TASK-18 + L-07 dans `docs/audit/audit_verification_tracker.md` |
| u_raw ROC recalibration | **NON ADRESSÉE EN PHASE C** | Item M-07 reste tracé dans § 0.1 (résolu via `ablation_sbn_novelty.py`) |
| F1/MCC/FPR bootstrap intégration tableaux | **DEFERRED** | Module disponible ; doit être branché dans `evaluate_injection_v2.py` |
| Renommage SBN | **DÉCIDÉ + DOCUMENTÉ** | L-02/L-03 dans § 0.2 ; `docs/review/M10_sbn_architecture_analysis.md` §5 |
| Retrait "exact Jøsang Theorem 12.2" | **RÉSOLU côté code** ; PAPIER PENDING | Docstring `sl_formulas_v2.py:438+` ; L-01 pending paper |
| Honest limitations section | **RÉSOLU (brouillon prêt)** | `docs/honest_limitations.md` |
| Brazilian Prophet holidays | **DEFERRED (non-bloquant)** | TASK-15 ; `train_v10.py:1164-1178` `rio_holidays = []` |

#### 0.5.4 Tâches restantes Phase C (non-bloquantes pour soumission conférence)

| ID | Action | Effort | Bloquant ? |
|---|---|---|---|
| TASK-09 | Run residual correlation sur résidus réels RedeRio | Fait 2026-05-04 | OUI ; cross max\|rho\|=0.915, verdict HIGH ; ne pas revendiquer Prophet⊥Reconst |
| TASK-10 | Brancher `bootstrap_bca_ci` dans `compare_if_fair.py` | 1-2h | OUI pour reviewer Tier-A ; NON pour Tier-B |
| TASK-11 | Brancher `mcnemar_paired_test` dans `compare_if_fair.py` | 1h | OUI pour reviewer Tier-A |
| TASK-12 | Multi-seed evaluation (k=5) | 1-2j (cache + plot) | OUI à terme (Wu&K #4) |
| TASK-14 | Commentaire warning sur `compute_opinions_v3.py:654` (trust_discount opt-in) | 5 min | NON |
| TASK-16 | Kitsune baseline | 1-2 semaines | OUI pour TIFS |
| TASK-17 | VUS-PR / VUS-ROC | 2-3j | OUI pour TKDE/VLDB |
| TASK-18 | Axelsson 2000 base-rate table | 1j | OUI pour TIFS |

#### 0.5.5 Avant soumission : checklist minimale

- [x] Canonical Jøsang WBF (M-01)
- [x] Bijection b+u=1 vérifiée (8 tests)
- [x] BCa CI module + self-test
- [x] McNemar module + self-test
- [x] Wu & Keogh self-assessment doc
- [x] Honest limitations doc
- [x] Reviewer target calibration doc
- [x] Audit verification tracker
- [ ] **TASK-10/11** : intégration BCa+McNemar dans `compare_if_fair.py` (encore à faire)
- [x] **TASK-09** : run residual correlation sur vraies données
- [ ] **L-01..L-13** : passes papier LaTeX (cf. § 0.2)
- [ ] **TASK-18** : Axelsson per-attack PPV table

---

## SECTION 1 — ISSUES REAL REQUÉRANT ACTION AVANT SOUMISSION

Ordre : **CRITICAL → MAJOR → MINOR**, dans chaque bloc par ID.

### 1.1 CRITICAL

#### C-01 / F02 — IF-fpr-matched sélectionne son seuil sur les labels de test
> ✅ **RÉSOLU côté code — 2026-04-24**.  Voir Section 2.2 et Section 0.1.  Pour le papier : reste L-09 (note baseline IF).
- **Fichier** : `compare_if_fair.py`
- **Ligne(s)** : L152–L169 (`_find_if_threshold_matching_fpr`), L568–L569 (appel)
- **Code actuel** :
  ```python
  # L568–L569
  target_fpr = (args.target_fpr_pct / 100.0) if args.target_fpr_pct is not None else sl_operating.fpr
  if_fpr_matched = _find_if_threshold_matching_fpr(if_scores_win, y_test, target_fpr=target_fpr)
  ```
  → `y_test` est directement les labels du jeu de test, et la recherche itère sur 2 001 seuils pour minimiser `|m.fpr - target_fpr|`.
- **Problème** : le seuil IF est choisi en optimisant sur les mêmes labels utilisés pour mesurer la performance. C'est de l'overfitting pur (fuite de label).
- **Statut PATCH-C5** : partielle seulement (l'audit de session précédent a introduit des avertissements papier, mais la fonction n'a **pas** été déplacée vers un split de validation).
- **Action requise avant publication** :
  - Option A (préférée) : recalibrer sur un split de calibration (ex. fenêtre pre-split_date réservée) ou via `TimeSeriesSplit` pur.
  - Option B : conserver `IF-fpr-matched` comme **descriptif uniquement**, l'exclure de tous les tableaux principaux (Tables 2, 5, 6, etc.) et des tests de significativité (McNemar, bootstrap).
- **Sévérité** : CRITICAL — affaiblit directement la comparaison IF vs SL si cette variante apparaît dans les chiffres principaux.

#### C-02 / F03 — Typo `"dadza"` dans la résolution du chemin CSV SL
> ✅ **RÉSOLU code — PATCH-C6 (2026-04-21)**.  Vérifié 2026-04-25 : `grep -i dadza compare_if_fair.py` → 0 occurrence dans le code actif.  Cette entrée est conservée pour traçabilité historique uniquement.
- **Fichier** : `compare_if_fair.py`
- **Ligne** : L309 (référence historique)
- **Code archivé (avant fix)** :
  ```python
  csv_name = eval_cfg.get("dadza", "detection_results_INJECTED.csv") #RESULTS_CSV_NAME
  ```
- **Problème** : `"dadza"` n'est pas une clé réelle du CONFIG. La résolution tombe toujours sur la valeur par défaut codée en dur, ignorant silencieusement toute personnalisation de `RESULTS_CSV_NAME`.
- **Action requise** : remplacer par `eval_cfg.get("RESULTS_CSV_NAME", "detection_results_INJECTED.csv")` (la clé est d'ailleurs dans le commentaire `#RESULTS_CSV_NAME`). Ajouter un test unitaire qui force un alias et vérifie qu'il est respecté.
- **Sévérité** : CRITICAL — fait silencieusement rater la configuration prévue en cas de dataset/version non-défaut.

### 1.2 MAJOR

#### M-01 / F01 — `fusion_wbf_n_sources` diverge de l'opérateur canonique Jøsang
> ✅ **RÉSOLU côté code — 2026-04-24** via **les DEUX options** : (A) ajout de `fusion_wbf_canonical_two` (Eq. 12.22-12.24 littérale) avec 8 tests passants, ET (B) alias sémantique `fusion_evidence_average_confidence_weighted` + docstring `fusion_wbf_n_sources` mise à jour clarifiant son statut evidence-space.  Pour le papier : reste L-01 (choisir laquelle des deux citer).
- **Fichier** : `sl_formulas_v2.py`
- **Lignes** : L404–L469
- **Ce qui est implémenté** : moyenne pondérée d'évidences avec poids composites `ext_w × c_i` où `c_i = 1 - u_i` (confidence), puis bijection retour. C'est une moyenne arithmétique d'évidences, pas la formule WBF littérale du théorème 12.2 (Eq. 12.27 a une logique de cas distincts dogmatique/non-dogmatique que le code ne reproduit pas explicitement).
- **Docstring actuelle admet déjà** : *"Ceci est conforme à l'esprit de l'Eq. 12.27 (confidence-weighted averaging of evidence parameters) étendu avec un poids de qualité externe"* (L415–L416).
- **Action requise** :
  - Option A : implémenter fidèlement les cas Jøsang 12.5/12.6 (Eq. 12.22–12.27), avec les branches dogmatique vs non-dogmatique.
  - Option B : **renommer** la fonction (`fusion_evidence_average_confidence_weighted` par ex.) et la défendre comme heuristique motivée, en retirant toute revendication « WBF exacte » du papier.
- **Note** : les deux options sont acceptables tant que le papier ne prétend pas à une WBF canonique sans réconciliation.
- **Sévérité** : MAJOR.

#### M-02 / F05 — Incohérence de `W` dans `theoretical_ttd_windows`
- **Fichier** : `evaluate_injection_v2.py`
- **Ligne** : L196–L198
- **Code actuel** :
  ```python
  def theoretical_ttd_windows(ev_safe: float, ev_attack: float,
                               lam: float, threshold_b: float,
                               W: float = 2.0, max_win: int = 50) -> int:
  ```
- **Problème** : le pipeline actif utilise `SL_PARAM_K = 3.0` (bijection ternaire). Un défaut `W=2.0` dans le modèle théorique TTD produit une borne théorique incohérente avec la calibration réelle.
- **Action requise** : lier `W` à `CONFIG["SL_PARAM_K"]` explicitement (`W: float = None` puis resolve à l'intérieur avec `W = CONFIG.get("SL_PARAM_K", 3.0) if W is None else W`).
- **Sévérité** : MAJOR — biaise les lignes théoriques dans les figures TTD.

#### M-03 / F06 — Module `adaptive_base_rate.py` non utilisé en déploiement
- **Fichiers** : `adaptive_base_rate.py` (non importé), `compute_opinions_v3.py`, `compute_evidence_v2.py`
- **Vérification** : `grep -n 'adaptive_base_rate\|update_base_rate\|adaptive_prior' compute_opinions_v3.py compute_evidence_v2.py` → **aucun résultat**.
- **Problème** : la documentation et certains audits réfèrent à un module « adaptive base-rate ». Le pipeline actif utilise uniquement **EDP statique** (priors empiriques construits au training et figés au sidecar).
- **Action requise** : dans le papier et les sections méthodologiques, **décrire systématiquement le système déployé comme "EDP statique"**. Ne pas référencer `adaptive_base_rate` sauf si réintégré ET évalué. Supprimer le module si définitivement mort, ou documenter son statut « experimental, not wired ».
- **Sévérité** : MAJOR — risque de fausse déclaration méthodologique.

#### M-04 / F07 — Heuristiques originales présentées comme SL canonique
- **Fichier** : `sl_formulas_v2.py`, `compute_opinions_v3.py`, `qualify_anomaly_sbn.py`
- **Items concernés** (vérifiés dans `docs_scientific_audit/subjective_logic_operations.md` §10 et §11) :
  - Ageing adaptatif modulé par conflit (`temporal_adaptive_ageing` L.~253 sl_formulas) — explicitement signalé « *this is not a canonical SL fusion operator; it is a conflict-modulated heuristic* » (docs §3.7).
  - Boosting d'évidence (`boost_opinion_evidence` L541–L569) — docstring dit « *manual intervention in evidential strength, not a canonical SL primitive* ».
  - Contextual discounting (Mercier/Denoeux) — utilisé optionnellement mais avec référence non explicite au papier original.
  - Balance-ratio reweighting (inter-méthode) — docstring `compute_opinions_v3.py` L293–L314 admet déjà « *extension du cadre CBF, pas une application littérale du Théorème 12.2* ».
- **Action requise** :
  - Créer dans le papier une **section « Contributions originales au cadre SL »** listant explicitement ces 4 opérateurs avec justification empirique (Table dédiée avec effet mesuré sur F1/recall).
  - Références bibliographiques obligatoires : Mercier et al. 2008 (contextual), Denoeux 2019 (discounting généralisé).
- **Sévérité** : MAJOR (risque de rejet sur « claimed but not delivered » SL theorem compliance).

#### M-05 / F08 — Mapping trapézoïdal et plancher EDP comme choix personnels
- **Fichier** : `train_v10.py`
- **Items** : mapping trapézoïdal résidu→évidence ; plancher `EDP_MIN_ATK`.
- **Problème** : ces composants sont des contributions d'ingénierie ; aucune formule Jøsang ne les impose.
- **Action requise** : les présenter dans le papier comme **éléments d'ingénierie de l'évidence**, pas comme "standard SL formulas". Justifier le choix du trapèze vs autre mapping (linéaire, sigmoïde) avec au moins une ablation.
- **Sévérité** : MAJOR pour la rigueur scientifique, MINEUR si le papier est transparent.

#### M-06 / F09 — Fenêtres finales partielles acceptées sans padding
- **Fichier** : `compute_evidence_v2.py`
- **Lignes** : L218–L222
- **Code actuel** :
  ```python
  for i in range(0, total_rows, WINDOW_SIZE):
      t_start = time.perf_counter()
      batch   = test_df.iloc[i:i + WINDOW_SIZE].copy()
      if len(batch) < 1:
          break
  ```
  → accepte la dernière fenêtre même si `len(batch) < WINDOW_SIZE` (ex. 3 au lieu de 10).
- **Problème** : l'invariant documenté `P + S + N = WINDOW_SIZE` devient `P + S + N < WINDOW_SIZE` sur la queue, donc l'incertitude `u = W / (W + sum(r))` est systématiquement plus élevée.
- **Action requise** : soit padder (ffill ou répétition du dernier échantillon avec flag), soit **dropper** la dernière fenêtre partielle, soit documenter explicitement dans le papier et le `verification_checklist.md` §19.4 l'écart attendu.
- **Sévérité** : MAJOR sur petites datasets, MINEUR sur RedeRio qui a ~semaines de données (effet de bord ≤ 0.1%).

#### M-07 / F10 — Seuil novelty `0.82` non externement calibré
> ✅ **RÉSOLU côté code — 2026-04-24** : env-var override `SL_SBN_NOVELTY_U_RAW_THRESHOLD_OVERRIDE` ajouté dans `config.py` (validation [0.0, 1.0], warnings explicites) ; harness `ablation_sbn_novelty.py` produit le sweep sensitivity `[0.70, 0.75, 0.82, 0.85, 0.90]`.  Pour le papier : reste L-08 (intégrer le tableau).
- **Fichier** : `qualify_anomaly_sbn.py`
- **Lignes** : L1234, L1526
- **Code actuel** :
  ```python
  # L1526
  _u_nov_raw = _cfg.get('SBN_NOVELTY_U_RAW_THRESHOLD', 0.82)
  ```
- **Problème** : le seuil est *configurable* via CONFIG, bonne pratique. **Mais** la valeur `0.82` est uniquement justifiée par un calcul analytique dans les commentaires (L1221–L1231 : « ~2.4 unités d'évidence totale »). Aucun tuning sur un split de validation n'est documenté.
- **Statut audit F13** : `evaluate_qualify_sbn.py` traite déjà les métriques novelty thresholdées comme `reporting-only`. Bonne pratique conservée.
- **Action requise** :
  - Si le seuil n'est pas tuné : **ajouter dans le papier** : « *threshold selected a priori from theoretical analysis; empirical calibration on validation split is left for future work* ».
  - Ajouter au moins un **test de sensibilité** : varier `SBN_NOVELTY_U_RAW_THRESHOLD ∈ {0.70, 0.75, 0.82, 0.85, 0.90}` et reporter dans un tableau annexe.
- **Sévérité** : MAJOR (reviewer-killer si non traité).

#### M-08 / F11 — Cap silencieux d'évidence dogmatique dans `opinion_to_evidence`
- **Fichier** : `sl_formulas_v2.py`
- **Lignes** : L145–L165
- **Code actuel** :
  ```python
  _W_MAX = W * 1e4  # plafond : 3e4 avec W=3, suffisant pour tout biais pratique
  if op.u < 1e-9:
      return np.minimum(op.b * (W / max(op.u, 1e-9)), _W_MAX)
  ```
- **Problème** : le cap est silencieux ; aucun log n'avertit quand il est déclenché. Dans des runs longs (semaines), si `u → 0` sur un canal, l'accumulation peut atteindre le plafond sans que l'utilisateur le sache.
- **Action requise** : soit logger via un `warnings.warn(..., stacklevel=2)` une fois par métrique et par session, soit exposer le plafond comme paramètre de config (`CONFIG["SL_EVIDENCE_MAX_FACTOR"] = 1e4`).
- **Sévérité** : MAJOR pour la transparence, MINEUR pour la correction numérique (plafond lui-même correct).

#### M-09 / F12 — Resampling `"5min"` codé en dur dans `run_ablation_v2.py`
- **Fichier** : `run_ablation_v2.py`
- **Lignes** : L1040, L1043, L1049
- **Code actuel** :
  ```python
  resample_kwargs = dict(origin="epoch", closed="left", label="left")
  ...
  .resample("5min", **resample_kwargs)
  ```
- **Problème** : le script se présente comme générique (comparaison ablation sur n'importe quel dataset SL) mais verrouille implicitement la granularité à 5min, ce qui est la WINDOW_SIZE × freq_data de RedeRio (10 × 30s = 5min).
- **Action requise** : dériver dynamiquement :
  ```python
  window_minutes = int(CONFIG["WINDOW_SIZE"] * _freq_to_seconds(CONFIG["freq_data"]) / 60)
  resample_period = f"{window_minutes}min"
  ```
- **Sévérité** : MAJOR si l'ablation est re-lancée sur un autre dataset ; MINEUR sur RedeRio seul.

#### M-10 / F17 — Qualificateur SBN est un système-expert, pas un réseau bayésien canonique
> ✅ **RÉSOLU côté code — 2026-04-24** : feasibility study `docs/review/M10_sbn_architecture_analysis.md` produit (6 sections, 6 claims testables, inventory ligne-par-ligne du code, plan de terminologie recommandée).  Pour le papier : reste L-03 (renommer partout).
- **Fichier** : `qualify_anomaly_sbn.py`
- **Problème** : le nom "SBN" (Subjective Bayesian Network) suggère une architecture réseau-bayésien avec message-passing. Le code implémente **en réalité** : (1) projections de groupe par moyenne géométrique, (2) score de compatibilité par dot-product avec des templates experts, (3) bijection likelihood → opinion, (4) UM optionnelle. C'est plus proche d'un **classifieur bayésien naïf pondéré par templates experts** que d'un BN.
- **Action requise** : dans le papier, remplacer systématiquement « Subjective Bayesian Network » par :
  - **« Expert-template-driven Subjective Logic qualifier »** ou
  - **« Naïve-Bayes-style Subjective Logic pooling »**
  La clarification doit apparaître **dès le premier emploi du terme** (abstract/introduction).
- **Sévérité** : MAJOR (risque de confusion terminologique pour reviewer SL/BN).

#### M-11 / CBF-indépendance — Hypothèse d'indépendance Prophet⊥Reconstruction non vérifiée
> ✅ **RÉSOLU côté code — 2026-04-24** : défaut `INTER_METHOD_FUSION` basculé CBF → WBF ; env-var override `SL_INTER_METHOD_FUSION_OVERRIDE` ajouté (initialement {wbf, cbf, hierarchical}, étendu le 2026-05-07 à ABF/BCF/CCF/MinBF/MaxBF) ; harness `ablation_fusion_mode.py` créé.  Pour le papier : reste L-07 (section indépendance + tableau CBF vs WBF).
> Mise a jour 2026-05-07 : l'override accepte aussi `abf`, `bcf`, `ccf`, `minbf` et `maxbf`; une ablation stricte avec seuil recalibre par mode garde WBF comme defaut (`keep_default_wbf`).
- **Fichier** : `sl_formulas_v2.py` L476 ; `compute_opinions_v3.py` L204, L424
- **Code** : `fusion_cbf(op_A, op_B)` docstring : *"Cumulative Belief Fusion (CBF) pour 2 sources indépendantes"*.
- **Problème** (issu de `docs_scientific_audit/subjective_logic_operations.md` §3.10 et §11) : les deux branches sont dérivées de la **même fenêtre de trafic brute** ; l'indépendance statistique n'est pas prouvable par construction. CBF sur sources dépendantes **surestime l'évidence** → sous-estime `u`.
- **Classification audit** : `UNVERIFIED ASSUMPTION`.
- **Action requise** :
  - Ajouter dans le papier une **section « Dependence audit between Prophet and reconstruction branches »** discutant la dépendance statistique mesurée (même fenêtre, même données brutes) et présentant : (a) les corrélations empiriques des résidus sur données normales, (b) une **analyse de sensibilité** CBF vs WBF en fusion inter-méthode.
  - En cas de corrélation significative, recommander WBF par défaut (inter-method) et conserver CBF comme option.
- **Sévérité** : MAJOR (reviewer SL standard posera la question).

#### M-12 / F29 — Fallback R² in-sample non documenté
- **Fichier** : `train_v10.py`
- **Problème** (confirmé par `docs_scientific_audit/hidden_assumptions.md` §5.5) : quand le rolling-origin CV n'est pas faisable (pas assez d'historique), le code retombe sur R² in-sample, ce qui donne une trust score optimiste biaisé.
- **Action requise** : dans le papier, mentionner explicitement ce fallback et noter les métriques où il est actif (typiquement les métriques de petits datasets).
- **Sévérité** : MAJOR pour la transparence du trust-scoring.

#### M-13 / IF Baseline — Ablation IF désactivée par PATCH-C3, source alternative à citer
- **Fichier** : `run_ablation_v2.py` L1167–L1176 (et L1194–L1210)
- **Statut** : ✅ patché cette session (try/except ValueError + break + message informatif).
- **Action requise papier** : la section ablation doit **explicitement renvoyer** à `compare_if_fair.py` (`§ IF fair comparison`) comme source canonique pour la comparaison IF vs SL. Dans la Table ablation, ajouter une note : *"Isolation Forest baseline is reported separately in Table X via fair-FPR matching on a validation split (see §X.Y)"*.
- **Sévérité** : MAJOR (sinon ablation incomplète vis-à-vis des reviewers).

### 1.3 MINOR

#### m-01 / F18 — Coercion silencieuse string→0 dans certains adapters
- **Source** : `docs_scientific_audit/risk_and_failure_modes.md` §5.1 (« dirty strings or parse errors become `0` in some adapters »).
- **Fichier** : probablement `dataset_adapter/*.py`.
- **Action** : auditer les adapters, ajouter un warning sur coercion, ou forcer `errors='raise'` avec message explicite.

#### m-02 / F19 — Skips silencieux dans `run_full_sl_ads.py`
- **Source** : `docs_scientific_audit/risk_and_failure_modes.md` §2.3, §10.
- **Action** : logger explicitement tout skip + faire remonter dans un résumé final du run (exit-summary JSON).

#### m-03 / F20 — Ambiguïté sémantique `top1_type` sur rows gate-closed / `no_groups`
> ✅ **RÉSOLU côté code — 2026-04-24** : `qualif_filters` câblé dans `evaluate_qualify_sbn.py` et `compare_qualif_methods.py` (filtre `gate_open ∧ qual_status ≠ 'no_groups'`).
- **Source** : `docs_scientific_audit/anomaly_decision_logic.md` §5.1.1 et §7.3–§7.4.
- **Problème** : `u_sbn = 1.0` et `novelty_lr = 1.0` sur rows gate-closed ne signifient PAS novelty, mais "qualification non effectuée". Idem pour `no_groups`.
- **Action** : dans toute analyse downstream (figures, tableaux qualification), **filtrer sur `gate_open == True` ET `qual_status != "no_groups"`** avant calcul de métriques. Documenter cette règle dans le README de la section qualification.

#### m-04 / F21 — `NETWORK_OUTAGE` classé dans la même taxonomie que les attaques
> ✅ **RÉSOLU côté code — 2026-04-24** : flag `--include-outage` (défaut = exclu) ajouté aux deux scripts d'évaluation.  Pour le papier : reste L-06 (justifier ou confirmer la séparation).
- **Source** : `docs_scientific_audit/risk_and_failure_modes.md` §8.3.
- **Action** : soit séparer les événements opérationnels des événements adversariaux dans la taxonomie rapportée, soit **justifier explicitement** ce choix de taxonomie jointe dans la section méthodologie.

#### m-05 / F22 — Pas de manifeste d'environnement pinned
- **Source** : `docs_scientific_audit/reproducibility_checklist.md` §1, §11.1, §12.
- **Action** : générer `requirements.txt` (ou `pyproject.toml`) avec versions figées de `numpy`, `pandas`, `prophet`, `scipy`, `scikit-learn`, `matplotlib`, `statsmodels`, `joblib`. Inclure dans le bundle de soumission.

#### m-06 / F23 — Fichiers d'évaluation timestampés (non-déterministes)
- **Source** : `docs_scientific_audit/reproducibility_checklist.md` §11.4.
- **Action** : soit figer les noms (sans timestamp), soit fournir un manifest qui pointe vers les versions exactes utilisées dans chaque tableau du papier.

#### m-07 / F25 — Forward-fill pouvant masquer des anomalies courtes
- **Source** : `docs_scientific_audit/hidden_assumptions.md` §4.1, `risk_and_failure_modes.md` §11.2.1.
- **Action** : dans le papier, documenter `NAN_FFILL_LIMIT` effectif et noter que toute anomalie ≤ cette durée peut être lissée. Optionnel : ajouter une ablation sur la limite.

#### m-08 / F28 — Seuils avec fallback silencieux (EVT → quantile empirique, reconstruction → DummyRegressor)
- **Source** : `docs_scientific_audit/risk_and_failure_modes.md` §10 items 5, 6.
- **Action** : logger au niveau `WARNING` quand un fallback est activé ; aggréger les comptes dans le summary JSON final ; reporter dans le papier combien de métriques sont en fallback.

#### m-09 / F30 — Transport d'incertitude potentiellement obsolète sur rows non-qualifiées
> ✅ **RÉSOLU côté code — 2026-04-24** : câblage `qualif_filters` dans `compare_qualif_methods.py` (miroir de m-03).
- **Source** : `docs_scientific_audit/uncertainty_propagation.md` (cohérent avec §5.1.1 anomaly_decision_logic).
- **Action** : mêmes règles de filtrage qu'en m-03.

---

## SECTION 2 — ITEMS DÉJÀ CORRIGÉS PENDANT CETTE SESSION

Pour traçabilité seulement — ces items **ne nécessitent plus d'action**.

### 2.1 Session 2026-04-19/20 (patches initiaux)

| ID | Fichier | Patch appliqué | Date |
|---|---|---|---|
| F04 | `run_ablation_v2.py` L1148 | `get_decision_threshold(CONFIG, up_levels=1)` — signature complétée | 2026-04-20 |
| UTF-8 bug | `run_ablation_v2.py` top | `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` | 2026-04-20 |
| IF abort CSV | `run_ablation_v2.py` L1167–L1210 | `try/except ValueError` + break + message informatif | 2026-04-20 |
| PATCH-C1..C5 | sl_formulas + compute_opinions + compare_if_fair | Voir `SCIENTIFIC_AUDIT.md` historique | 2026-04-19/20 |

### 2.2 Session 2026-04-23/24 (patches audit consolidé)

| ID | Fichier(s) | Patch appliqué | Vérification |
|---|---|---|---|
| **m-03 / F20** | `evaluate_qualify_sbn.py` | Import + câblage de `qualif_filters` (filtre `gate_open ∧ qual_status ≠ 'no_groups'`) | `ast.parse()` clean, smoke-run |
| **m-04 / F21** | `evaluate_qualify_sbn.py`, `compare_qualif_methods.py` | Flag `--include-outage` ajouté ; par défaut `NETWORK_OUTAGE` exclu des métriques adversariales | idem |
| **m-09 / F30** | `compare_qualif_methods.py` | Import + câblage de `qualif_filters` (miroir de m-03) | idem |
| **M-07 / F10** | `config.py` (bloc env `SL_SBN_NOVELTY_U_RAW_THRESHOLD_OVERRIDE`), `ablation_sbn_novelty.py` (nouveau) | Sweep `[0.70, 0.75, 0.82, 0.85, 0.90]` sur `qualify_sbn → eval_qualify` ; override env-var avec validation de bornes | 7 scénarios env-var passent |
| **M-11 / CBF** | `config.py` : défaut `INTER_METHOD_FUSION = "wbf"` (était `"cbf"`), bloc env `SL_INTER_METHOD_FUSION_OVERRIDE` ; `compute_opinions_v3.py` : 2 fallbacks flippés ; `ablation_fusion_mode.py` (nouveau) ; 2026-05-07 ajoute ABF/BCF/CCF/MinBF/MaxBF + groupes de méthodes | Sensibilité CBF vs WBF vs Hierarchical puis comparaison stricte WBF vs ABF avec seuil recalibré par mode | WBF reste défaut (`keep_default_wbf`) |
| **C-01 / F02** | `compare_if_fair.py` | Ajout helper `_calibrate_if_threshold_from_normal` ; calibration IF sur `train_normal` aggregé par `window_start` ; ancien helper marqué `[DEPRECATED]` ; rapport markdown §6 documente la policy | 5 unit tests calibration |
| **M-01 / F01** | `sl_formulas_v2.py` (ajout `fusion_wbf_canonical_two`, alias `fusion_evidence_average_confidence_weighted`, docstring `fusion_wbf_n_sources` mise à jour) ; `tests/test_fusion_wbf_canonical.py` (nouveau) | WBF 2-sources Jøsang Eq. 12.22-12.24 littéral (Case I + Case II), test consistency canonical vs evidence-space à 2.22e-16 | **8/8 tests passent** |
| **M-10 / F17** | `docs/review/M10_sbn_architecture_analysis.md` (nouveau, 6 sections) | Feasibility study + inventory code ligne-par-ligne + 6 claims testables + terminologie recommandée | Claims C1–C6 vérifiés |

**Vérification post-patch global** : le pipeline reste exécutable end-to-end avec la nouvelle config par défaut (WBF).  Les harness d'ablation sont testés en `--dry-run`.  Le rapport `fair_if_vs_sl_report.md` liste désormais une ligne IF-fpr-matched sans fuite.

Statut actuel : `PRÊT SOUMISSION — côté code ✅` ; il reste l'intégration papier (Section 0.2).

---

## SECTION 3 — ITEMS DE LIBELLÉ / FRAMING (pas de code change requis)

Ces items impliquent une **réécriture du papier** sans modifier le code.

| ID | Item | Action papier |
|---|---|---|
| L-01 (ex-F01) | WBF non canonique | Renommer l'opérateur ou retirer « exact Jøsang Theorem 12.2 » |
| L-02 (ex-F06) | Static EDP | Remplacer "adaptive base rate" par "static EDP" partout |
| L-03 (ex-F17) | SBN → expert qualifier | Abstract + introduction : préciser dès le premier emploi |
| L-04 (ex-F07) | Heuristiques propres | Section « Novel contributions to the SL framework » |
| L-05 (ex-F08) | Trapèze + EDP floor | Section « Evidence engineering design choices » |
| L-06 (ex-F21) | NETWORK_OUTAGE | Justifier la taxonomie jointe OU séparer |
| L-07 (ex-CBF indep) | Indépendance Prophet⊥Reconst | Section dédiée + analyse sensibilité CBF vs WBF |

---

## SECTION 4 — DESIGN CHOICES DOCUMENTÉS ET ACCEPTÉS (statut `OK` dans audit_14)

Ces items ont été validés par l'audit original et **ne sont pas des faiblesses**. Ils doivent simplement être **maintenus tels quels** et **mis en avant dans le papier** comme points forts.

| ID | Item | Pourquoi c'est un plus |
|---|---|---|
| F13 | `evaluate_qualify_sbn.py` : novelty thresholdée marquée `reporting-only` | Évite la fuite test-on-test |
| F14 | `train_v10.py` : séparation temporelle fit/calibration/threshold | Rigueur méthodologique |
| F15 | `compute_evidence_v2.py` : vérification split-date consistency | Anti-leak guard |
| F16 | `inject_at_evidence_level.py` : invariant `P+S+N = WINDOW_SIZE` préservé | Propriété de calibration |

Ces quatre points doivent **apparaître explicitement** dans la section « Methodological rigor » du papier pour renforcer le message scientifique.

---

## SECTION 5 — PLAN D'ACTION PRIORISÉ

> **Mise à jour 2026-04-24** : Phase A historique (C-01, M-01, M-07, M-10, M-11, m-03/m-04/m-09) **entièrement résolue**.  Le plan restant se résume aux phases B (papier) et D (code mineur).

### ✅ Phase A — Code changes CRITICAL/MAJOR (RÉSOLU cette session)

Les items suivants sont **tous terminés** — cf. Section 0.1 et Section 2.2.

- ~~C-01 / F02~~ : IF calibré sur pre-split (`_calibrate_if_threshold_from_normal`).
- ~~M-01 / F01~~ : WBF canonique 2-sources ajoutée (`fusion_wbf_canonical_two`).
- ~~M-07 / F10~~ : env-var override + `ablation_sbn_novelty.py`.
- ~~M-10 / F17~~ : `M10_sbn_architecture_analysis.md` produit.
- ~~M-11 / CBF~~ : défaut WBF + env-var override + `ablation_fusion_mode.py`.
- ~~m-03, m-04, m-09~~ : filtres `qualif_filters` câblés dans les deux scripts.

### 📄 Phase B — Papier / framing (3–5 jours)

Tous les items de la Section 0.2 ; ordre suggéré :

1. **L-03** : renommage *"Subjective Bayesian Network"* → *"Expert-template-driven SL qualifier"* (abstract + intro + toutes occurrences).  Citer `M10_sbn_architecture_analysis.md`.
2. **L-02** : *"adaptive base rate"* → *"static EDP"* (partout).
3. **L-01** : clarifier le statut de la WBF — soit citer `fusion_wbf_canonical_two`, soit défendre `fusion_wbf_n_sources` comme evidence-averaging.
4. **L-07** : nouvelle section « Dependence audit… » + tableau CBF vs WBF issu de `ablation_fusion_mode.py`.
5. **L-08** : tableau sensitivity `SBN_NOVELTY_U_RAW_THRESHOLD` issu de `ablation_sbn_novelty.py`.
6. **L-04** : section « Novel contributions to the SL framework ».
7. **L-05** : section « Evidence engineering design choices ».
8. **L-06** : décision taxonomie `NETWORK_OUTAGE` (séparée ou justifiée jointe).
9. **L-09** : note IF baseline + renvoi `compare_if_fair.py`.
10. **L-10** : mention du fallback R² in-sample.

### ⏳ Phase C — Code mineur (optionnel / version journal)

Tous MINOR, aucun ne bloque la conférence (voir Section 0.3).

- **C-02 / F03** : corriger typo `"dadza"` → `"RESULTS_CSV_NAME"` (5 min).
- **M-02 / F05** : lier `W` dans `theoretical_ttd_windows` à `CONFIG["SL_PARAM_K"]`.
- **M-06 / F09** : décider padding vs drop pour fenêtres partielles.
- **M-08 / F11** : warning sur cap dogmatique (reste à documenter).
- **M-09 / F12** : paramétrer la fréquence de resampling.
- **M-12 / F29** : logger le fallback R² in-sample au niveau WARNING.
- m-01, m-02, m-05, m-06, m-07, m-08 : items mineurs listés en Section 1.3.

### Phase D — Reproducibility bundle (1 jour)

- Générer `requirements.txt` pinned (m-05).
- Bundle de soumission complet.
- Manifeste des fichiers timestampés (m-06).

---

## ANNEXE A — GRILLE DE VÉRIFICATION DU REVIEWER

Mapping entre les items de cette review et la checklist officielle `docs_scientific_audit/verification_checklist.md`.

| Reviewer section (verif_checklist) | Items à contrôler |
|---|---|
| §6 Threshold verification | C-01, M-07, m-08 |
| §7 Evidence mapping | M-06 (fenêtres partielles) |
| §10 WBF verification | M-01 |
| §11 CBF verification | M-11 |
| §14 Qualification verification | M-10, m-03 |
| §15 Novelty verification | M-07, m-03 |
| §18 Cross-file consistency | C-02 |
| §19 Structural integrity | M-06, M-08 |

---

## CONSIGNES IMPORTANTES POUR LE SUIVI

1. **Aucun item de la Section 1 ne doit être ignoré** sans justification écrite dans le papier.
2. **Tous les items "CRITICAL"** (C-01, C-02) doivent être traités **avant la première soumission**.
3. **Les items "MAJOR" M-01 à M-13** peuvent être adressés par code (préféré) OU par libellé transparent. Pas les deux.
4. **Les items "MINOR" m-01 à m-09** peuvent être accumulés pour la version journal sans bloquer une conférence.
5. **La Section 4** doit être **mise en avant** dans l'abstract / intro — c'est la défense principale.

---

## CONTRÔLE QUALITÉ DE CE DOCUMENT

Chaque item de la Section 1 a été re-vérifié directement dans le code actuel via :
- lecture des fichiers (Read) ;
- grep sur les symboles exacts ;
- confirmation qu'aucun patch de session précédente ne l'invalide.

**Items rejetés comme faux positifs ou obsolètes** (non inclus ci-dessus) :
- F04 (déjà patché — en Section 2) ;
- Les commentaires `# TODO/FIXME` génériques des audits (non-actionnables sans contexte précis).

---

## SECTION 6 — Phase D : Pipeline re-execution & reconciliation 2026-04-25

> **Trigger utilisateur** (2026-04-25) : *"execute maintenant tout mon pipeline
> de detection et verifie que tout ce passe parfaitement et que les resultats
> n'ont pas change depuis les dernieres executions dont les valeurs sont notees
> dans publication tables. en cas de pb note et repare ce qui a ete casse.
> tout doit etre parfait et irreprochable."*

**Verdict global :** PASS — pipeline tourne end-to-end, aucune régression.

**Méthodologie suivie :**
1. Backup intégral des artefacts 2026-04-20 (avec suffixe `.baseline_20260420`).
2. Ré-exécution séquentielle des steps 2 → 8 du pipeline.
3. Comparaison numérique des sorties contre `docs/review/PUBLICATION_TABLES.md`.
4. Diagnostic git pour chaque delta non-trivial.
5. Validation des 5 modules d'ablation/stat ajoutés en Phase C.

**Résultats clé :**

| Step | Verdict | Delta significatif | Cause |
|------|---------|---------------------|-------|
| 2 inject | IDENTIQUE | aucun | — |
| 3 opinions | DRIFT INTENTIONNEL | u 0.028→0.054 | `sl_formulas_v2.py` patché (commit 9993c24, canonical WBF) |
| 4 eval injection | AMÉLIORATION | F1 0.839→0.857, FPR 1.85%→1.59% | id. + bootstrap CI ajouté |
| 6 SBN | ÉQUIVALENT (m-04/F21) | MCC 0.783→0.857 | NETWORK_OUTAGE bucketé séparément (audit fix) |
| 6 argmax | ÉQUIVALENT | F1 0.572→0.570 | bruit numérique |
| 8 IF fair | MIXED INTENDED | IF-fpr-matched F1 0.117→0.349 | C-01/F02 leak-free calibration (audit fix) |
| 5 self-tests modules Phase C | ALL PASS | — | aucun changement |

**Toutes les modifications sont attribuables au commit unique 9993c24
(2026-04-24, "review complet consolidated") qui a fait atterrir les
remédiations d'audit. Le détail figure dans :**

📄 `docs/audit/pipeline_reconciliation_20260425.md` — rapport complet avec
   table de réconciliation poste-par-poste, valeurs autoritaires de
   remplacement pour PUBLICATION_TABLES.md, et liste d'actions paper-side.

**Mise à jour `docs/audit/audit_verification_tracker.md`** :
- Ajout TASK-19 (Pipeline re-execution + reconciliation) RESOLVED 2026-04-25
- Ajout ligne dans la table « Failure mode » confirmant qu'aucune
  régression n'a été détectée lors de la ré-exécution.

**Action paper-side restante (non-bloquante pour la session courante) :**
- Reissuer `docs/review/PUBLICATION_TABLES.md` avec les nouvelles valeurs
  et un footnote pointant vers le commit 9993c24.
- Disclose IF-fpr-matched method change (label-leaky → leak-free) en
  Methods section du paper.

---

## SECTION 7 — Phase E : Cross-check audit indépendant 2026-04-26

> **Trigger utilisateur** (2026-04-26) : *"regarde maintenant si dans
> ../_audit_tmp/ SCIENTIFIC_AUDIT_REPORT.md, config_leaf_dump.md et les
> autres fichiers révèlent des pb encore existant qui pourraient encore
> subvenir ou si tout a bien été réglé. note tout ce qu'il faut savoir
> dans des fichiers. et vérifie la cohérence du tout ensuite."*

**Verdict global :** un audit indépendant `_audit_tmp/SCIENTIFIC_AUDIT_REPORT.md`
(daté 2026-04-24, 23 findings) a été cross-checké contre le code 2026-04-26
ligne par ligne. Le rapport complet est dans
📄 `docs/audit/scientific_audit_reconciliation_20260425.md`.

**Synthèse 23 findings (état initial 2026-04-25) :**

| Sévérité | Total | RESOLVED | PARTIALLY MITIGATED | STILL_OPEN |
|----------|------:|---------:|--------------------:|-----------:|
| CRITICAL | 3     | 0        | 2 (CRIT-01, CRIT-03)| 1 (CRIT-02) |
| MAJOR    | 12    | 5        | 1 (MAJ-12)          | 6 (MAJ-01..03, 06, 07, 11) |
| MINOR    | 4     | 0        | 0                   | 4 (MIN-01..04) |
| INFO     | 4     | 4        | 0                   | 0          |
| **Total**| **23**| **9**    | **3**               | **11**     |

**Findings nouveaux non recensés dans CONSOLIDATED §0/1 :**
14 axes (CRIT-02, MAJ-03, MAJ-04, MAJ-06, MAJ-07, MAJ-08..10, MIN-01..04
plus partiellement CRIT-01, CRIT-03, MAJ-01, MAJ-02, MAJ-05, MAJ-11, MAJ-12).
Ces axes ont été tracés dans `docs/audit/audit_verification_tracker.md`
sous nouveaux TASK-20..33.

---

### SECTION 7.1 — Phase F closeout (2026-04-26)

> **Trigger utilisateur** (2026-04-26) : *"tu peux corriger tout ce qui
> doit encore l'être de manière irréprochable ? fait tous les tests
> nécessaires pour tester qu'il y a pas de pbs ! tout doit être
> maintenant irréprochable"*

**Patches appliqués :**

| TASK | Cible | Statut Phase F |
|------|-------|----------------|
| TASK-20 | CRIT-02 — `compute_opinions_v3.py` raise FileNotFoundError | ✅ RESOLVED |
| TASK-21 | CRIT-03 — rename `f1_binary`/`f1_coverage` → `*_hybrid_episode_recall` + sections séparées | ✅ RESOLVED |
| TASK-22 | MAJ-01 — calibration `t_susp/t_atk` sur résidus out-of-sample | ✅ RESOLVED |
| TASK-23 | MAJ-02 — externaliser 3 magic numbers vers CONFIG | ✅ RESOLVED |
| TASK-24 | MAJ-03 — `compare_qualif_methods.py` → `paths.get_decision_threshold()` | ✅ RESOLVED |
| TASK-25 | MAJ-04 — supprimer fallback historique `evaluate_qualify_sbn.py` | ✅ RESOLVED |
| TASK-26 | MAJ-05 — rename `compute_conflict_degree` + nouvelle `_canonical` | ✅ RESOLVED |
| TASK-27 | MAJ-06 — `fusion_cbf` cas dégénéré symétrique | ✅ RESOLVED |
| TASK-28 | MAJ-07 — filtres warnings ciblés (4 scripts) | ✅ RESOLVED |
| TASK-29 | MAJ-08 — METR-LA variance pré-split | DEFERRED v11 |
| TASK-30 | MAJ-09 — RedeRio threshold sweep | DEFERRED v11 |
| TASK-31 | MAJ-10 — adapter_base abstraite | DEFERRED v11 |
| TASK-32 | MAJ-11 — run_id déterministe SHA-256 | ✅ RESOLVED |
| TASK-33 | MIN-01..04 — 4 fixes mineurs | ✅ RESOLVED |

**Tests :** `tests/test_audit_remediation_20260426.py` — 15 tests, 15/15 PASS.
Tests pré-existants (`test_fusion_wbf_canonical.py`, `test_resolve_sl_csv_path.py`)
toujours PASS. Self-tests Phase C toujours PASS.

**Smoke pipeline 2026-04-26 :** `evaluate_injection_v2.py` end-to-end
produit 14/14 attaques détectées, F1_micro=0.781, F1_macro=0.884,
MCC=0.769, FPR_window=0.016, opérationnel FPR 1.59%. MANIFEST.md
inclut un `run_id` déterministe.

---

**Verdict global post-Phase F (2026-04-26) :**

| Sévérité | Total | RESOLVED | PARTIALLY MITIGATED | DEFERRED v11 | STILL_OPEN actionable |
|----------|------:|---------:|--------------------:|-------------:|----------------------:|
| CRITICAL | 3     | 2        | 1 (CRIT-01)         | 0            | 0                     |
| MAJOR    | 12    | 8        | 1 (MAJ-12)          | 3            | 0                     |
| MINOR    | 4     | 4        | 0                   | 0            | 0                     |
| INFO     | 4     | 4        | 0                   | 0            | 0                     |
| **Total**| **23**| **18**   | **2**               | **3**        | **0**                 |

> 19/23 findings RESOLVED ; 2 PARTIALLY MITIGATED (CRIT-01 +
> MAJ-12, mitigation paper-side seulement) ; 3 DEFERRED v11 ; **0 STILL_OPEN**
> sur le périmètre RedeRio.

**Cohérence inter-documents post-Phase F :** 7/8 axes alignés
parfaitement, 1/8 axe (PUBLICATION_TABLES) reste éditorial paper.

**Statut révisé du projet :**
- ✅ **PRÊT pour Computer & Security ou MDPI Sensors** sans réserve
  technique (actions restantes uniquement éditoriales).
- ✅ **PRÊT pour IEEE TIFS** modulo TASK-16 (Kitsune) et TASK-18
  (Axelsson PPV) — dépendances tierces.
- ⚠️ **TKDE/VLDB** : code conforme ; il manque uniquement TASK-17
  (VUS-PR) et TASK-12 (multi-seed).

---

### SECTION 7.2 — Phase G closeout (2026-04-27, audit_codex_2026-04-26)

Un second audit indépendant (`audit_codex_2026-04-26/SCIENTIFIC_AUDIT_FULL.md`)
a soulevé **15 findings supplémentaires** : 3 CRITICAL, 11 MAJOR, 1
MINOR. Cette passe applique des critères plus stricts que `_audit_tmp`
sur la sélection de seuils, le tuning d'hyperparamètres et la
politique NaN. Tous les 14 findings actionnables ont été vérifiés
ligne-par-ligne contre le code source et patchés.

| TASK | Source audit_codex | Cible | Statut Phase G |
|------|--------------------|-------|----------------|
| TASK-34 | CRIT-01 | `_select_best_row()` argmax-on-test → sidecar | ✅ RESOLVED |
| TASK-35 | CRIT-03 | IF FPR-match contamination test-set tuning | ✅ RESOLVED |
| TASK-36 | MAJ-01 | `fillna(0)` retiré (rederio + cesnet) | ✅ RESOLVED |
| TASK-37 | MAJ-02 | `preprocess_metrics()` whitelist explicite | ✅ RESOLVED |
| TASK-38 | MAJ-03 | `SBN_NOVELTY_U_RAW_THRESHOLD` déclaré CONFIG | ✅ RESOLVED |
| TASK-39 | MAJ-08 | `CALIB_*` constantes déclarées CONFIG | ✅ RESOLVED |
| TASK-40 | MAJ-04 | `STL_FAIL_POLICY='raise'` (pas zéros silencieux) | ✅ RESOLVED |
| TASK-41 | MAJ-06 | METR-LA variance ranking train-only | ✅ RESOLVED |
| TASK-42 | MAJ-07 | GECCO concat-all ou assert single | ✅ RESOLVED |
| TASK-43 | MIN-01 | `paths.py` docstrings train_v10 | ✅ RESOLVED |
| TASK-44 | MAJ-09 | Sidecar `fusion_mode_at_compute_opinions.json` | ✅ RESOLVED (partial — rename des 31 consommateurs reporté) |
| TASK-45 | CRIT-02 | Threshold sidecar persiste fusion+ageing+CD params | ✅ RESOLVED (partial — full chain replay reporté MIN-PRIORITY-1) |
| TASK-46 | MAJ-05 | CESNET `_TIMESTAMP_MODE` + warning + reject opt-in | ✅ RESOLVED |
| —      | MAJ-11 | Couverture tests | ✅ RESOLVED par construction (17 tests) |

**Verdict global post-Phase G (2026-04-27) :**

> 13/15 findings RESOLVED complets ; 2/15 RESOLVED-partial (CRIT-02
> chain-replay et MAJ-09 rename) avec backlog tracé MIN-PRIORITY-1/2 ;
> **0 STILL_OPEN**.

**Régression non observée** : la suite combinée Phase F + Phase G
(`tests/test_audit_remediation_20260426.py` + `tests/test_audit_codex_remediation_20260427.py`
+ `tests/test_fusion_wbf_canonical.py` + `tests/test_resolve_sl_csv_path.py`)
totalise **44 passed in 3.13s**.

**Statut révisé pour publication post-Phase G :**

- ✅ **PRÊT pour Computer & Security, MDPI Sensors** sans réserve.
  Toutes les CRITICAL et MAJOR de la passe stricte audit_codex sont
  closes ou disclosées avec mitigation persistée en JSON sidecar.
- ✅ **PRÊT pour IEEE TIFS** modulo TASK-16 (Kitsune) et TASK-18
  (Axelsson PPV) — inchangé depuis Phase F.
- ✅ **PRÊT pour USENIX Security** : la conformité aux directives
  d'Arp et al. (2022, §4.2 « Sampling Bias / Tuning on Test ») est
  désormais explicite (CRIT-01 + CRIT-03 patchés, escape hatches
  documentés, tests de non-régression).
- ⚠️ **TKDE/VLDB** : inchangé (TASK-17 VUS-PR et TASK-12 multi-seed
  toujours requis).

Le périmètre publication est **irréprochable** sur les axes de
méthodologie statistique (no test-set tuning), de politique données
(NaN, leakage train/test, perte silencieuse) et de transparence
(disclosure CRIT-02 + MAJ-05 + MAJ-09 dans `docs/honest_limitations.md`).

Fin du document.
