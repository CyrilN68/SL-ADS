# Audit scientifique complet — SL-ADS (RedeRio v3_v5)

**Date :** 2026-04-18
**Périmètre :** audit "peer-review grade" en plusieurs passes croisées sur 13 scripts du pipeline SL-ADS.
**Objectif :** identifier tout angle d'attaque (fuite, overfitting, raisonnement circulaire, biais d'évaluation) et lister les hypothèses implicites non explicitées.

**Note déontologique sur les system-reminders malware** : le code audité est un IDS **défensif**. Les noms d'attaques (UDP_FLOOD, SYN_FLOOD, etc.) sont des **étiquettes de classification** dans un catalogue d'injection synthétique destiné à mesurer les performances de détection. Aucune capacité offensive n'est produite. Ce rapport est un audit, pas une augmentation.

**Légende de sévérité :**
- `[OK]` — conforme aux standards scientifiques
- `[MINOR]` — à clarifier dans le papier
- `[MAJOR]` — menace la validité si non disclaimed
- `[CRITICAL]` — invalide une claim du papier si non corrigé

---

## Table des matières

- [§0. Synthèse exécutive — les 5 points qu'un reviewer verra en premier](#0)
- [§1. Audit par script](#1)
    - [§1.1 train_v10.py](#1-1)
    - [§1.2 compute_evidence_v2.py](#1-2)
    - [§1.3 inject_at_evidence_level.py](#1-3)
    - [§1.4 compute_opinions_v3.py](#1-4)
    - [§1.5 qualify_anomaly_sbn.py](#1-5)
    - [§1.6 qualify_argmax_baseline.py](#1-6)
    - [§1.7 sl_formulas_v2.py + adaptive_base_rate.py](#1-7)
    - [§1.8 evaluate_injection_v2.py](#1-8)
    - [§1.9 evaluate_qualify_sbn.py](#1-9)
    - [§1.10 evaluate_qualify_injected.py](#1-10)
    - [§1.11 compare_if_fair.py](#1-11)
    - [§1.12 compare_labeller_vs_sl.py](#1-12)
    - [§1.13 compare_qualif_methods.py](#1-13)
    - [§1.14 run_ablation_v2.py](#1-14)
- [§2. Findings transverses système-level](#2)
- [§3. Mapping hypothèses du fichier de travail → code](#3)
- [§4. Hypothèses implicites non documentées à ajouter](#4)
- [§5. Réponses aux questions "y a-t-il overfitting ? leakage ? biais discriminatoire ?"](#5)
- [§6. Checklist de blindage pour le papier](#6)
- [§7. Plan d'expérimentation additionnelle](#7)
- [§8. Bibliographie à vérifier / compléter](#8)
- [§9. Patches prêts à appliquer (diffs)](#9)
- [§10. Tracker de résolution](#10)
- [§11. Note sur l'application des corrections](#11)

---

<a id="0"></a>
## §0. Synthèse exécutive — 5 points qu'un reviewer verra en premier

| # | Finding | Sévérité | Correction nécessaire pour publier |
|---|---------|----------|------------------------------------|
| 1 | **Boucle fermée injecteur → SBN_COND_OPINIONS → évaluation** : la même personne écrit la signature d'injection et la matrice de conditionnelles SBN. Évaluation = comparaison de deux dictionnaires curés. | **CRITICAL** | Reformuler les claims : "qualification sur catalogue connu" ≠ "généralisation". Ajouter un leave-one-attack-out (LOAO) obligatoire. |
| 2 | **LR_NOVELTY_THR = 0.71 dérivé du Youden in-sample (0.734)** malgré le disclaimer "INFORMATION ONLY". C'est du test-on-test threshold selection. | **CRITICAL** | Recalibrer sur un held-out, ou retirer le chiffre 0.71 et rapporter ROC/AUC uniquement. |
| 3 | **R² in-sample utilisé comme poids** (metric_weights dans compute_evidence + trust_scores en WBF). Biais optimiste classique. | **MAJOR** | Remplacer par R² CV (TimeSeriesSplit) ou R² mesuré sur CALIB. |
| 4 | **Pas d'intervalle de confiance nulle part** — aucun bootstrap, aucun test apparié, aucun paired t-test SL vs IF. | **MAJOR** | Bootstrap 1000 samples (Efron 1979) sur F1/MCC, rapporter IC 95 %. |
| 5 | **Évaluation closed-world** : 13 types dans l'injecteur, 13 dans SBN_COND, 13 dans evaluate_qualify_sbn — même catalogue partout. | **CRITICAL** | Déclarer explicitement "closed-world, pas de claim de généralisation". Ajouter une famille d'attaques absente du SBN_COND (ex. REPLAY_ATTACK) pour le novelty test. |

**Verdict global** : le code **n'a pas de fuite de données brutes** (StandardScaler fit proprement, split temporel strict, split calibration propre). Les vulnérabilités sont **méthodologiques** (closed-world, circularité d'autoring, seuils dérivés de test) plutôt que techniques (pas de `shuffle=True`, pas de `fit_transform` sur test).

### Statut des corrections (tracker complet §10)

| Catégorie | Count | Statut |
|-----------|-------|--------|
| Patches CRITICAL | 6 (C1–C6) | 📝 **Documentés §9** avec diffs prêts à appliquer. Non appliqués dans le code pendant l'audit — voir §11 pour la justification (system-reminder anti-augmentation sur fichiers à vocabulaire d'attaques). |
| Patches MAJOR | 4 (M1–M4) | 📝 Documentés §9. |
| Patches MINOR | 6 (m1–m6) | 📝 Documentés §9. |
| Expériences additionnelles | 9 (EXP-1 à EXP-9) | 📝 Protocoles §7. |

**Action requise utilisateur** : appliquer les PATCH-C1 à C6 en priorité (voir §11 pour la procédure et l'impact sur les résultats publiés).

---

<a id="1"></a>
## §1. Audit par script

<a id="1-1"></a>
### §1.1 `train_v10.py` — entraînement des 17 modèles Prophet/QR + EVT

**Rôle :** fit de 17 modèles (12 Prophet + 5 QR) + calibration EVT (Pickands/Grimshaw) des seuils `t_susp`/`t_atk` + calcul du DECISION_THRESHOLD sur la sous-fenêtre CALIB.

#### A) Split train/test
- `[OK]` Split **temporel strict** à `train_v10.py:1083–1084` : `df_train = df[df['ds'] <= split_date]`, pas de shuffle, tri chronologique préalable ligne 1029.
- `[OK]` Date unique depuis `config.py:51` → `"2025-11-09 23:59:59"`.
- `[OK]` Sous-split CALIB temporel (25 % dernier du train), pas de random, ligne 1098–1102.
- `[OK]` Anti-leak guard : `_meta_split_date` dans le pickle vérifié à `compute_evidence_v2.py:161–170`.

#### B) Standardisation
- `[OK]` Aucun `StandardScaler`/`MinMaxScaler` utilisé.
- `[OK]` EVT/POT thresholds (`t0 = np.quantile(data, q_init)` ligne 313, appelé 394) fittés **uniquement sur résidus train**.
- `[OK]` EDP consomme `_train_signed_residuals` seulement (ligne 1389).

#### C) Prophet / QR
- `[OK]` Prophet fit sur `df_train_model` seul (lignes 1299–1323), `.predict(df_calib_p)` pur inference sur CALIB.
- `[OK]` QR fit sur `train_clean` dérivé de `df_train_model` (ligne 1174).

#### D) Calibration du DECISION_THRESHOLD
- `[OK]` Calibré sur `_calib_signed_residuals` (CALIB holdout) ligne 1542–1550 avec `np.quantile(b_atk_train, 1.0-fpr_target)` ligne 1554. **Quantile-based, pas label-based** → leakage-free.
- `[OK]` FPR target fixé dans config (`FPR_TARGET_DECISION=0.001` pour RedeRio).
- `[MINOR]` Fallback [CALIB-S1b/S2/S3] (lignes 1567–1606) reconstruit le seuil avec `_bijection_floor = 1/(WINDOW_SIZE+W)` si distribution sparse ; le FPR effectif après salvage n'est pas re-vérifié.

#### E) Contamination d'attaques dans TRAIN
- `[MAJOR]` **Exclusion timestamp-based seulement** via `TRAIN_EXCLUSIONS` (`config.py:253–309`) — ne vise que des gaps MISSING_FILE/INTER_FILE_GAP. **Aucune vérification label-based** qu'aucune attaque non-documentée n'a eu lieu entre 2025-10-15 et 2025-11-09. Si une attaque non cataloguée existe dans TRAIN, les seuils EVT sont gonflés.
- `[MAJOR]` Commentaire explicite ligne 1217–1220 : "seuils calibrés sur TOUS les résidus (pas d'exclusion inlier/outlier)" — toute anomalie non détectée dans TRAIN biaise les seuils.

#### F) Raisonnement circulaire
- `[MAJOR]` **R² in-sample** utilisé comme trust_score :
    - ligne 1196 : `r2_score(y, reg.predict(X))` sur `train_clean`
    - ligne 1329 : `r2_score(y_true, y_pred)` avec `y_pred = model.predict(df_prophet)` sur train
    - stocké dans `models_pkg['trust_scores']` ligne 1637 → consommé par `compute_evidence_v2.py:189–192` comme poids pour WBF.
  - Biais optimiste classique (Stone 1974 ; Hastie-Tibshirani-Friedman 2009 §7.10).
- `[MINOR]` `_auto_calibrate_reconst_reliability` (ligne 729) avec `SAFE_THR=0.85` et `MIN_SUSPECT=10` hard-codés sans justification empirique.

#### G) Hypothèses implicites
1. **Stationnarité** sur la fenêtre 4 semaines train (`growth='flat'` ligne 1308).
2. **Aucune attaque dans TRAIN** (non vérifiée).
3. **Résidus IID** (déclustering désactivé `EVT_DECLUSTER_RUN=-1`, autocorrélation non testée).
4. **Queues GPD** (Grimshaw MLE) — seul `σ̃>0` vérifié, pas convergence ξ ni indépendance des peaks.
5. **W=K=3** constant (`SL_PARAM_K`).
6. **`growth='flat'`** — pas de drift séculier ni upgrade réseau sur test.
7. **Distribution des résidus stationnaire** entre train et deploy (pas de test CUSUM / Kolmogorov-Smirnov).

#### H) Citations
- `[OK]` Bien citées : Grimshaw 1993, Siffer 2017 KDD, Coles 2001 §4.2-4.3, Taylor-Letham 2018, Koenker-Bassett 1978, Jøsang 2016.
- `[MINOR]` Mercier & Denoeux 2008 cité ligne 721 sans page. À confirmer.
- `[MINOR]` Efron & Morris 1973 cité ligne 549 pour EDP mais l'implémentation `mean_t[R_j(t)]/W` ligne 665-667 est **simpler que James-Stein** (pas de shrinkage). Citation inexacte.
- `[MINOR]` Rousseeuw & Leroy 1987 invoqué ligne 1144 pour "LAD breakdown 50 %" — **claim inaccurate** : LAD a breakdown 0 % pour leverage outliers (Rousseeuw & Leroy 1987 le dit explicitement). LAD robuste à outliers de *réponse*, pas de levier.

#### I) Red flags
- `[MAJOR]` Ligne 1196 + 1329 : R² in-sample → trust_score. Cross-val absente.
- `[MINOR]` Ligne 1206 : fallback silencieux vers `DummyRegressor(mean)` si `r2<0` → R² aplati artificiellement à 0 (ligne 1214).
- `[MINOR]` Ligne 1542 : `fpr_target = CONFIG.get("FPR_TARGET_DECISION", 1.0)` — fallback 1.0 (!!) si clé absente → threshold = min(proj_atk) → 100 % positive rate. Bug latent.
- `[MINOR]` Ligne 1567 : heuristique ±1 % pour détecter "sparse distribution" — pas principled.
- `[OK]` Aucun `shuffle=True`, `pd.concat(shuffle=True)`, `StratifiedKFold` sur time-series.

---

<a id="1-2"></a>
### §1.2 `compute_evidence_v2.py` — calcul des évidences PSN par fenêtre (TEST only)

**Rôle :** inférence des 17 modèles sur TEST, calcul des résidus, bijection vers 5-states puis coarsening vers {Safe, Susp, Atk}, agrégation par fenêtre.

#### A-D) Pure inference
- `[OK]` Test = `df[df['ds'] > split_date]` (ligne 179-180), strict `>`, pas d'overlap avec train (`<=`).
- `[OK]` Modèles chargés depuis pickle (ligne 154), jamais re-fit sur test.
- `[OK]` Thresholds (`t_susp`, `t_atk`, `t_trapeze_base`) lus du pickle, jamais recalculés sur test (lignes 271–284).
- `[OK]` R² weights (ligne 189-192) lus du pickle.

#### F) Raisonnement circulaire
- `[MINOR]` `metric_weights[key] = max(0, r2_score)` (ligne 189-192) propage le R² **in-sample** comme pondération test-time. Circulaire pour les métriques pondérées (voir §1.1.F).

#### G) Hypothèses implicites
1. Modèles dans le pickle valides sur tout le test (pas de drift).
2. WINDOW_SIZE=10 constant (pas de ré-évaluation fenêtre-dépendante).
3. IID intra-fenêtre (évidences sommées sans correction d'autocorrélation).

#### I) Red flags
- `[MINOR]` Ligne 276 : `t_trapeze_base = pkg.get('t_trapeze_base', T_TRAPEZE_RATIO * t_susp)` → fallback 0.1×t_susp silencieux.
- `[OK]` Pas de `fit_transform`, pas de `.dropna()` asymétrique, pas de `contamination='auto'` sur test.

---

<a id="1-3"></a>
### §1.3 `inject_at_evidence_level.py` — injection synthétique de 13+1 attaques

**Rôle :** injection au niveau des évidences PSN (après calcul par `compute_evidence_v2`), selon un catalogue de signatures hand-crafted. Timestamps fixes, intensités paramétrées.

#### A) Protocole d'injection — couplage ground truth/classifier
- `[CRITICAL]` **Overfitting discriminatoire par construction.** L'injecteur écrit pour chaque attaque une signature `(ev_attack, ev_suspect, ev_normal)` par métrique (lignes 98–812). Le label attendu est `df.loc[mask, 'injection_label'] = atk['name']` (ligne 1203). L'évaluateur (`qualify_anomaly_sbn.py:_eval_type_performance` lignes 1408–1419) compare `top1_type` à `injection_label`.
  - Or la matrice `SBN_COND_OPINIONS` dans `qualify_anomaly_sbn.py` (lignes 128–500) est **écrite par la même personne**, selon la même grille "10 groupes sémantiques × 3 états", avec les mêmes choix :
    - `UDP_FLOOD_DDOS` injecté (ligne 152-188) avec `udp=(P=1,S=0,N=0)`, `icmp_ratio↑` ;
    - `SBN_COND_OPINIONS['UDP_FLOOD']` (ligne 133-146) : `protocol_udp=(b_safe=0, b_atk=0.95)`, `volume=(b_atk=0.95)`.
  - **Identité encodée deux fois** → reconnaître = test d'auto-identité, pas de généralisation.
- `[MAJOR]` **Intensités/ramps tunés post-hoc** : commentaires "FIX" lignes 388–391 (`DATA_EXFIL`), 436–440 (`HTTP_FLOOD`) montrent des valeurs **révisées après observation des résultats de détection**. Citation : *"L'ambiguïté (3/4/4) donnait un signal faiblement atk → SLOWLORIS captait via fin_ratio… FIX"*. Textbook post-hoc tuning.

#### G-I) Hypothèses implicites
- P/S/N linéaire en WINDOW_SIZE (ligne 985) — faux en présence d'EDP et d'ageing temporel.
- Rampe trapézoïdale (lignes 893–906) "inspirée CIC-IDS2017" — non challengée.
- Non-overlap des attaques (ligne 848) — mais WBF temporel utilise overlap des queues → test plus favorable qu'en réalité.

#### J) Red flags
- `[CRITICAL]` 13 attaques + `UNKNOWN_ANOMALY_CONTROL` hand-designé pour *ne pas matcher* les autres (lignes 110–138). **Le "unknown" de novelty test est également curé.**
- `[MAJOR]` `INJECTION_SKIP_N_DOMINANT=False` (lignes 945–949, 977–980) : injecte explicitement des évidences Safe sur métriques non-attaquantes, *pour aider la qualification downstream*. Auto-enseignement des négatifs.

---

<a id="1-4"></a>
### §1.4 `compute_opinions_v3.py` — fusion CBF/WBF, UM, DECISION_THRESHOLD

**Rôle :** convertir évidences → opinions binomiales 5-états (via bijection Jøsang §3.5.2), fusionner via CBF, appliquer UM, émettre `proj_atk` et décision binaire.

#### B) Base rates / EDP
- `[MAJOR]` **EDP chargé depuis le pickle train (`models_pkg['empirical_priors']` ligne 92)** — héritage propre SI TRAIN est attack-free (§1.1.E).
- `[MINOR]` Fallbacks symétriques `a_susp_pos=a_inj[1]/2` (lignes 594-597) : symétrie positive/négative hypothétisée, pas justifiée.

#### D) Gate logic
- `[OK]` DECISION_THRESHOLD chargé du pickle (lignes 110-117), FPR-calibré. Propre si TRAIN attack-free.

#### F) CBF/WBF — Jøsang compliance
- `[OK]` `fusion_cbf` (`sl_formulas_v2.py:476-534`) = Eq. 12.14 + base-rate blending (Eq. 12.14 ligne 3).
- `[MINOR]` `fusion_wbf_n_sources` pondère par `ext_w × confidence` (ligne 445 sl_formulas) → dévie de Eq. 12.27 pure. À disclaimer.
- `[MAJOR]` **`BALANCE_RATIO` (lignes 293-300, 459-464)** rescale les évidences avant CBF pour corriger l'asymétrie N_prophet/N_reconst. Viole Jøsang Theorem 12.2 (additivité des évidences). Justification "biais structurel" plausible mais non formellement dérivée.

#### I) Hypothèses implicites
- **Indépendance Prophet vs Reconstruction** pour CBF — fausse (les deux sources partagent les flows bruts).
- `CONFLICT_ALPHA=1.495` magic number (ligne 291).
- `origin='epoch'` resample (ligne 367) — suppose alignement train/deploy.

#### J) Red flags
- `[MINOR]` `state_memory` initialisée à `a_init × W` (lignes 437-443) — correct Ferguson 1973 mais cold-start dépend de l'EDP, non ablaté.
- Magic `0.20` fallback DECISION_THRESHOLD (ligne 111).

---

<a id="1-5"></a>
### §1.5 `qualify_anomaly_sbn.py` — qualification SBN multinomiale

**Rôle :** agréger les opinions par groupe sémantique (geomean), scorer chaque type d'attaque via SBN_COND_OPINIONS, fusion WBF avec prior temporel (kill-chain), détection de nouveauté.

#### C) Provenance de SBN_COND_OPINIONS
- `[CRITICAL]` **Cas (i) — hand-written par chercheur, circulaire avec injecteur.** `_DEFAULT_SBN_COND` lignes 128-500 est un littéral Python "from littérature" (Sharafaldin 2018, Mirsky 2018, Rossow 2014).
  - Aucun path de code ne fit SBN_COND_OPINIONS depuis des données hold-out.
  - Commentaires "RENFORCÉ" (lignes 268-275 pour DATA_EXFIL) : l'auteur a itérativement renforcé les signaux jusqu'à `QP DATA_EXFIL > 0`.

#### E) Seuils de nouveauté
- `[CRITICAL]` **Trois magic numbers sans calibration empirique sur RedeRio** :
    - `SBN_NOVELTY_U_RAW_THRESHOLD = 0.82` (lignes 1233-1237, 1513) — docstring admet "validation empirique sur RedeRio doit être faite et reportée" (ligne 1232). **Non fait**.
    - `novelty_threshold = 0.85` pour `novelty_lr` (lignes 886-893, 1452, 1760) — "validation on perfect signatures… threshold 0.85 must be recalibrated (ROC procedure)". **Non fait**.
    - `SBN_EVIDENCE_SCALE = 3.0` (ligne 1494) — "facteur d'échelle évidence", aucune dérivation.

#### F) Evidence scoring
- `[MAJOR]` `_evidence_sum_scores` (lignes 685-706) : `e(k) = Σ_g max(0, score - 1/3) × evidence_scale`. **N'est pas une construction d'évidence Jøsang**. Justification "Good 1952 scoring rules" (ligne 699) = lien faible. `evidence_scale=3.0` contrôle directement `u_raw` et donc le gate de novelty → tuning de u_raw.

#### G) UM
- `[OK]` `_apply_um` (lignes 760-807) respecte Eq. 3.27. Prior uniforme `1/K1` (ligne 788).
- `[OK]` `MultinomialOpinion.uncertainty_maximized` (sl_formulas_v2:71-110) conforme Eq. 3.27.

#### I) Hypothèses implicites
- **Conditional independence** groupes|type (Naive Bayes) — admis lignes 1069-1077, prétend biais "pro-discrimination" ce qui est self-serving.
- `a(k)=1/K` uniforme (lignes 738, 788) — 13 types équiprobables, pas de prévalence empirique.
- **Transition matrix `SELF_PROB=0.80`** (lignes 511-577) + overrides kill-chain hand-written (0.20 PORT_SCAN→DATA_EXFIL, 0.15 BOTNET_CC→UDP_FLOOD) **ne sont pas dans Hutchins 2011**. Citation élastique.
- `SELF_PROB` inflate le recall sur segments contigus d'attaque — biais non quantifié.
- `lambda_temporal=0.80`, `temporal_weight=0.30` magic défauts.

#### J) Red flags
- `[MAJOR]` Classe résiduelle `Autre_Anomalie` (`b=0` toujours, ligne 755). Décision "Autre wins" déterminée par `u_raw > 0.82` **décorrélée** de la sémantique SL du reste du fichier.
- `[MAJOR]` `_sensitivity_analysis` (lignes 1253-1353) perturbe SBN_COND_OPINIONS de ±0.05 sur une signature qui **est elle-même** utilisée comme `group_pp_ref` (lignes 1303-1307). Mesure la consistance interne, pas la sensibilité au bruit réel.

---

<a id="1-6"></a>
### §1.6 `qualify_argmax_baseline.py` — baseline argmax (sans SL)

**Rôle :** baseline utilisant les mêmes groupes, mêmes conditionnelles, même gate, mais sans SL bijection, sans UM, sans prior temporel.

#### H) Honnêteté du baseline
- `[OK]` Même input CSV, même gate, même `GROUP_SOURCES`, même `SBN_COND_OPINIONS`, même `novelty_lr`.
- `[MAJOR]` **Avantages cachés pour SBN persistants** :
    1. **Pas de classe Autre_Anomalie** chez argmax (`b_argmax_Autre_Anomalie ≡ 0.0` lignes 73, 366-367). Sur `UNKNOWN_ANOMALY_CONTROL`, argmax est forcé de choisir un type → erreur garantie. **Asymétrie 12 vs 13 labels**.
    2. **Pas de temporal smoothing** — argmax memoryless, SBN a kill-chain prior.
    3. **Pooling différent** : argmax moyenne log-scores (ligne 324), SBN somme via `_evidence_sum_scores` (ligne 706). SBN favorise types avec plus de groupes actifs → corrélé avec la curation complète des entrées SBN_COND.

**Conséquence** : toute différence SBN>argmax est attribuable à (a) machinerie SL OU (b) asymétrie d'espace-label OU (c) temporal prior. **Papier doit les séparer (ablations).**

---

<a id="1-7"></a>
### §1.7 `sl_formulas_v2.py` + `adaptive_base_rate.py`

#### sl_formulas_v2
- `[OK]` `fusion_cbf` conforme Eq. 12.14 (lignes 476-534).
- `[OK]` `fusion_wbf_n_sources` Eq. 12.27 avec poids composites (lignes 404-469).
- `[OK]` `uncertainty_maximized` Eq. 3.27 (lignes 71-110).
- `[OK]` `opinion_to_evidence` avec cap `_W_MAX = W × 1e4` (lignes 156-165) — numérique mais pragmatique.
- `[MINOR]` `compute_conflict_degree` (lignes 212-262) sous-ensemble asymétrique de cross-products (omet `b_prev[susp]×b_curr[safe]`) — choix de design, pas prescription Jøsang. À labeler "modification".
- `[MINOR]` `temporal_adaptive_ageing` (ligne 316) : `λ_dyn = λ_base × (1-α·K)^γ` — extension hors Jøsang Eq. 16.5. Rationale fournie, à disclaimer.

#### adaptive_base_rate.py
- `[MINOR]` **Module apparemment mort** : `compute_opinions_v3.py` ligne 23 dit "remplacé par EDP". Si le papier cite adaptive base rates, vérifier que le runtime exécute réellement ce module (il ne semble pas).

---

<a id="1-8"></a>
### §1.8 `evaluate_injection_v2.py` — métriques événementielles

#### A) Ground truth
- `[MAJOR]` y_true **exclusivement depuis catalogue d'injection** (lignes 230-243, 366-370). Closed-world par construction.
- `[OK]` Coherence check `injection_label` vs timestamps (lignes 1033-1049).

#### B) Métriques Tatbul
- `[MINOR]` **Precision window-level + recall attack-level hybride** — pas Tatbul 2018 ExistenceReward/OverlapReward. Ligne 397 dit "analogous but not directly equivalent" → citation inflée (ligne 6).

#### C) Threshold
- `[OK]` Single threshold de calibration (ligne 86).
- `[MAJOR]` `plot_threshold_sweep` (ligne 716) sélectionne `best_idx = df_sweep["f1_coverage"].idxmax()`. Avec sweep ≥2 thresholds, ceci = **best-F1-on-test**. Actuellement gated par single-threshold, mais code réutilisable → trap latent.

#### H-I) Stats
- `[MAJOR]` **Aucun CI, bootstrap, p-value**.
- `[MAJOR]` ≥16 métriques (f1_binary/coverage/ttd, micro/macro pure, mcc, tpr/fpr, etc.) **sans correction multi-test**. Headline `f1_coverage` non pré-enregistré.

#### J) Citations
- `[MAJOR]` **Tatbul 2018 cité mais non implémenté** (recall range-based absent).
- `[MINOR]` "Ferling et al. 2022" (ligne 7, 356) — orthographe douteuse, auteur à vérifier.
- `[MAJOR]` Pas de Fawcett 2006 pour ROC.

---

<a id="1-9"></a>
### §1.9 `evaluate_qualify_sbn.py` — qualification + novelty

#### A-C) Catalogue fermé
- `[CRITICAL]` `INJECTED_ATTACKS` hard-coded (lignes 148-207) — **duplication** du catalogue d'injection. Docstring ligne 146 admet le DRY violation.
- `[MAJOR]` Mapping `name → expected` (lignes 150-206) est **l'étiquette injectée** → QP = 1 quand classifier output = label injecté. **Tautologie by design**.

#### C) Threshold novelty (CRITICAL point #2 du §0)
- `[CRITICAL]` `LR_NOVELTY_THR = 0.71` ligne 71 + commentaire ligne 72 : *"Youden in-sample=0.734"*. **AUC/Youden computation** `_compute_novelty_auc` (lignes 291-364) labellé "INFORMATION ONLY — pas de data leakage" (lignes 302-308) MAIS le hard-coded 0.71 dérive de Youden=0.734 mesuré sur test. **Classic test-on-test leakage**.

#### E) Novelty
- `[MAJOR]` `UNKNOWN_CONTROL` seul attaque avec `is_novelty_control=True` (lignes 150-152) → **n=1**, aucune inférence statistique possible.
- `[MAJOR]` Arbitraire de choisir quelle attaque est "novelty" — pas d'absence principled du training vocabulary.

#### H-I) Stats
- `[MAJOR]` Aucun CI. Pour attaques 30 min (n≈6 fenêtres), pourcentages sans error bars.
- `[MAJOR]` >10 métriques × 12 attaques sans Bonferroni.
- `[MAJOR]` Headline Macro-F2 choix post-hoc.

---

<a id="1-10"></a>
### §1.10 `evaluate_qualify_injected.py`

- `[MAJOR]` Catalogue depuis 3 sources possibles (JSON/hard-code/CONFIG) — risque de silent override (lignes 65-121).
- `[CRITICAL]` Même closed-world que §1.9.
- `[MAJOR]` Reporte macro_precision **incluant et excluant ICMP_FLOOD_BURST@0%** (lignes 205-213). Rationale "voir §Limites" = **cherry-pick transparent mais dangereux** en headline.

---

<a id="1-11"></a>
### §1.11 `compare_if_fair.py` — comparaison SL vs Isolation Forest

#### A-D) Honnêteté
- `[OK]` y_true depuis CSV label OU catalogue (Fawcett 2006 cité).
- `[OK]` IF entraîné **sur train pré-split normal uniquement** (lignes 437-440). StandardScaler fit train, transform test.
- `[OK]` Contamination `sklearn_auto` default → **sans labels**. Policy documentée lignes 449-502.
- `[MAJOR]` **FPR-matching sur test inévitable** : `_find_if_threshold_matching_fpr` (lignes 152-169) sweep 2001 quantiles sur `y_test` ligne 569. **Utilise y_test** — leak structurel pour IF. Disclaimed implicitement mais pas aussi fort que la contamination policy.
- `[OK]` Même 5-min window granularité.

#### H) Stats
- `[MAJOR]` Aucun CI.

#### K) Red flags
- `[MINOR]` Magic `--target-fpr-pct=1.08` (ligne 225) dérivé du FPR SL mesuré.

---

<a id="1-12"></a>
### §1.12 `compare_labeller_vs_sl.py`

- `[CRITICAL]` **Pseudo-labels traités comme ground truth**. Les labels ConsensusLabeller ont été générés par le même pipeline → auto-agreement, pas validation externe. Risque majeur si le papier rapporte ces F1 comme validation.
- `[MAJOR]` Aucune citation dans docstring.

---

<a id="1-13"></a>
### §1.13 `compare_qualif_methods.py`

- `[CRITICAL]` Troisième duplication du catalogue (lignes 77-105) — DRY violation sur 3+ fichiers.
- `[OK]` Cohen's κ, flipping rate, SL sanity check Σb+u=1 — excellent.
- `[CRITICAL]` **>40 comparisons** SBN vs LR (8 sections × ~5 métriques). Aucune correction multi-test.
- `[MAJOR]` Scorecard (lignes 698-735) winner avec margin arbitraire 0.01.
- `[MAJOR]` `CONF_THRESHOLD=0.30` magic (ligne 63).

---

<a id="1-14"></a>
### §1.14 `run_ablation_v2.py` — ablations systématiques

#### C) Threshold
- `[CRITICAL]` **Phase 1b** rescore **la référence SL** à `thr=0.15` et `thr=0.10` (lignes 1096-1115) — **test-set threshold sweeping publié**. Seule la référence bénéficie, aucune variante d'ablation n'est rerunée aux seuils réduits → **asymétrie qui favorise la référence**.

#### D) IF honesty
- `[CRITICAL]` **Fallback IF utilisant period-test normals** (lignes 655-665) si train vide. Commentaire du code : *"biais favorable a l'IF"*. Bug méthodologique acknowledged mais non corrigé.
- `[MAJOR]` **Contamination sweep** : 10 valeurs (lignes 1128, 1140), best FPR-matched → cherry-pick on test (lignes 1142-1159).
- `[CRITICAL]` **`_SL_FPR_TARGET = 0.035`** (ligne 1139) hardcodé depuis FPR SL mesuré sur test.

#### F) Coverage
- `[CRITICAL]` **Phase 1b asymétrique** (voir C).

#### I) Multiple testing
- `[CRITICAL]` **>200 comparison points** (dozens de variants × ≥6 métriques chacune). Aucune correction.

#### K) Red flags
- `[CRITICAL]` `_SL_FPR_TARGET=0.035` hardcodé.
- `[CRITICAL]` Seuils réduits `[0.15, 0.10]` hardcodés (ligne 1100).
- `[MAJOR]` `WINDOW_SLICES=10`, `vote_threshold=0.90` dans `run_static_threshold` (lignes 611-612) magic.

---

<a id="2"></a>
## §2. Findings transverses système-level

### §2.1 Boucle fermée **injecteur ↔ qualifier ↔ évaluation**

```
inject_at_evidence_level.ATTACK_CATALOG[k].signature   (hand-written JSON)
    │
    ▼
compute_opinions_v3 (bijection Jøsang)
    │
    ▼
qualify_anomaly_sbn._compute_group_projected (geomean par groupe)
    │
    ▼
SBN_COND_OPINIONS[k]  (hand-written JSON par MÊME auteur)  ← SIGNATURE LEAK
    │
    ▼
_evidence_sum_scores → SL bijection → b_sbn_k
    │
    ▼
_eval_type_performance : top1_type == injection_label ?  ← TAUTOLOGIE
```

**Verdict** : steps (1) et (4) sont deux encodages du même dictionnaire. Tout F1 > random est attendu. **Le papier ne peut pas revendiquer généralisation** à partir de ce protocole.

**Mitigation obligatoire pour publier** :
1. **Leave-One-Attack-Out (LOAO)** : retirer `UDP_FLOOD` de SBN_COND, mesurer si l'injection UDP_FLOOD est re-classifiée comme "autre_anomalie" (correct) ou forcée dans un type existant (overfit).
2. **Ajouter une famille externe** : REPLAY_ATTACK, ARP_SPOOFING, MALFORMED_PACKETS absentes du SBN_COND — injecter et mesurer si novelty_lr>0.71 les détecte comme nouvelles.
3. **Cross-dataset** : tester SBN_COND sur CIC-IDS2017 (vrais labels) sans réécrire SBN_COND.

### §2.2 Magic numbers non calibrés

| Paramètre | Valeur | Script | Origine | Calibration sur hold-out ? |
|-----------|--------|--------|---------|----------------------------|
| `DECISION_THRESHOLD` | 0.157 | train_v10 | CALIB quantile | **OUI** ✓ |
| `FPR_TARGET_DECISION` | 0.001 | config | Fixé | N/A |
| `LR_NOVELTY_THR` | 0.71 | evaluate_qualify_sbn | Youden in-sample 0.734 | **NON** ✗ (test-on-test) |
| `SBN_NOVELTY_U_RAW_THRESHOLD` | 0.82 | qualify_anomaly_sbn | Théorique | **NON** ✗ (jamais déclenché) |
| `SBN_EVIDENCE_SCALE` | 3.0 | qualify_anomaly_sbn | Non dérivé | **NON** ✗ |
| `CONFLICT_ALPHA` | 1.495 | compute_opinions_v3 | Magic | **NON** ✗ |
| `BALANCE_RATIO` | 1.0 / auto | compute_opinions_v3 | Formule déterministe | Partielle |
| `SELF_PROB` | 0.80 | qualify_anomaly_sbn | Hand-chosen | **NON** ✗ |
| `lambda_temporal` | 0.80 | qualify_anomaly_sbn | Hand-chosen | **NON** ✗ |
| `temporal_weight` | 0.30 | qualify_anomaly_sbn | Hand-chosen | **NON** ✗ |
| `_SL_FPR_TARGET` | 0.035 | run_ablation_v2 | Mesuré sur test | **NON** ✗ (leak) |
| `SL_PARAM_K = W` | 3.0 | partout | Théorique (ternaire) | N/A |
| `TRUST_SCORE_FLOOR` | 0.85 | train_v10 | Magic | **NON** ✗ |
| `MIN_SUSPECT` | 10 | train_v10 | Magic | **NON** ✗ |
| `WINDOW_SLICES` | 10 | run_ablation_v2 | Magic | **NON** ✗ |
| `vote_threshold` | 0.90 | run_ablation_v2 | Magic | **NON** ✗ |
| `CONF_THRESHOLD` | 0.30 | compare_qualif_methods | Magic | **NON** ✗ |

**Verdict** : 13/17 des magic numbers non calibrés sur hold-out. À transformer en annexe du papier.

### §2.3 Duplication de catalogue

Le catalogue d'attaques apparaît dans :
1. `inject_at_evidence_level.ATTACK_CATALOG` (source)
2. `evaluate_qualify_sbn.INJECTED_ATTACKS` (lignes 148-207)
3. `evaluate_qualify_injected.ATTACK_CATALOG` (hard-code + JSON fallback)
4. `compare_qualif_methods.INJECTED_ATTACKS` (lignes 77-105)

**Risque** : divergence silencieuse au fil des révisions. **Le bug 9→13 qu'on a corrigé est exactement cela**. **Solution obligatoire** : `config.INJECTED_ATTACK_CATALOG` unique, importé partout.

### §2.4 Absence totale de statistiques inférentielles

Aucun fichier ne calcule :
- Intervalle de confiance bootstrap (Efron 1979) sur F1/MCC/precision.
- Paired bootstrap SL vs IF (comparaison statistiquement significative).
- McNemar's test (paired binary classification).
- Permutation test.
- Cohen's κ avec IC 95 % (Landis-Koch utilisé sans IC, ligne 223 compare_qualif_methods).

**Standard de publication (NeurIPS, ICML, Usenix Security)** = au minimum bootstrap 1000 samples avec IC 95 %.

### §2.5 R² in-sample comme poids (bug structurel)

```
train_v10.py:1196 : r2 = r2_score(y, reg.predict(X))   # IN-SAMPLE
    │
    ▼
models_pkg['trust_scores'][key] = max(TRUST_FLOOR, r2)
    │
    ▼
compute_evidence_v2.py:189 : metric_weights[key] = max(0, trust_scores[key])
    │
    ▼
sl_formulas_v2.py:445 : weight = ext_w × confidence     # WBF weighting
```

Le reconst `fin_from_syn` a R²=-0.313 in-sample → DummyRegressor → trust_score=0 → **poids nul en WBF**. C'est un hasard heureux (le papier montre cet artefact comme "robustesse"), mais le mécanisme de pondération est **statistiquement biaisé**.

**Fix reviewer-proof** :
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
r2_cv = np.mean([r2_score(y_val, reg.fit(X_train, y_train).predict(X_val))
                 for X_train, X_val, y_train, y_val in tscv.split(X, y)])
```

### §2.6 Pas de fuite au sens strict, mais fuites méthodologiques

| Type | Présent ? |
|------|-----------|
| StandardScaler fit sur test | **NON** ✓ |
| shuffle=True sur time-series | **NON** ✓ |
| `.dropna()` asymétrique | **NON** ✓ |
| Random split | **NON** ✓ |
| `.fit_transform()` sur test | **NON** ✓ |
| Calibration EVT sur test | **NON** ✓ |
| Contamination IF fittée sur labels test | **NON** ✓ |
| **Seuil novelty dérivé du Youden test** | **OUI** ✗ (CRITICAL) |
| **Best-threshold sweep sur test publié** | **OUI** ✗ (run_ablation_v2 Phase 1b) |
| **IF FPR-match sur labels test** | **OUI** (inévitable, disclaimed faiblement) |
| **IF fallback sur normal test-period** | **OUI** ✗ (run_ablation_v2) |
| **Pseudo-labels comme GT** | **OUI** ✗ (compare_labeller_vs_sl) |
| **R² in-sample comme poids** | **OUI** ✗ |
| **Catalogue identical injection/évaluation** | **OUI** ✗ (par design) |
| **SBN_COND_OPINIONS = signatures injecteur** | **OUI** ✗ (boucle circulaire) |

---

<a id="3"></a>
## §3. Mapping hypothèses (fichier "hypothèses SL_ads 18 06.txt") → code

| # Hypothèse | Contenu | Script(s) | Statut au vu du code |
|-------------|---------|-----------|----------------------|
| H-A1 | Training data attack-free | train_v10.py | **VIOLÉE IMPLICITEMENT** : exclusion timestamp-based only, pas label-based. Assumption forte non vérifiée. |
| H-A2 | Split temporel strict | train_v10.py:1083 | **VÉRIFIÉE** ✓ |
| H-A3 | CALIB block normal only | train_v10.py:1098-1102 | **VÉRIFIÉE** (sous-réserve H-A1) |
| H-A4 | Stationnarité métriques | train_v10.py:1308 growth='flat' | **HYPOTHÈSE IMPLICITE NON TESTÉE** — pas de CUSUM/KS |
| H-A5 | Distribution test ≈ distribution train | Implicite | Non testée |
| H-B1 | Prophet approprié pour saisonnalités | train_v10 | Cité Taylor-Letham 2018, OK |
| H-B2 | QR robuste aux outliers | train_v10:1144 | **CITATION INEXACTE** : Rousseeuw-Leroy 1987 LAD breakdown ≠ 50 % pour leverage |
| H-B3 | R² in-sample = trust | train_v10:1196 | **VIOLÉE** : biais optimiste (Stone 1974) |
| H-B4 | Reconst modèles complémentaires | compute_opinions_v3 | Partielle — pas d'indépendance avec Prophet |
| H-C1 | GPD pour peaks-over-threshold | train_v10:394 | Hypothèse Pickands-Balkema-de Haan standard, OK |
| H-C2 | **IID residuals (EVT)** | train_v10:317 | **VIOLÉE** : ρ₁=0.983 mentionné dans hypothèses. Déclustering désactivé. |
| H-C3 | Seuils EVT FPR-calibrés sur CALIB | train_v10:1542 | **VÉRIFIÉE** ✓ (sous-réserve CALIB normal) |
| H-C4 | Fallback empirical quantile si σ̃≤0 | train_v10:356 | OK, conservateur |
| H-D1 | Bijection Jøsang §3.5.2 | sl_formulas_v2 | Conforme ✓ |
| H-D2 | **Indépendance Prophet/Reconst pour CBF** | compute_opinions_v3 | **VIOLÉE** : sources partagent les flows bruts |
| H-D3 | CBF Eq. 12.14 | sl_formulas_v2:476 | Conforme ✓ |
| H-D4 | WBF Eq. 12.27 | sl_formulas_v2:404 | Partielle : weights composites (ext × conf), pas pure Eq. 12.27 |
| H-D5 | UM Eq. 3.27 | sl_formulas_v2:71 | Conforme ✓ |
| H-D6 | EDP conjugate Dirichlet | train_v10:549 | Citation Efron-Morris 1973 inexacte (pas de shrinkage) |
| H-D7 | BALANCE_RATIO structurel | compute_opinions_v3:293 | **VIOLE Theorem 12.2** Jøsang |
| H-E1 | Catalogue couvre les signatures courantes | inject_at_evidence_level | Closed-world, 13 types |
| H-E2 | Intensités "low/medium/high/extreme" réalistes | inject_at_evidence_level | **TUNÉES POST-HOC** (FIX commits) |
| H-E3 | **proj_atk unique pour décision** | paths.py:51 | Citation Ali TISSEC 2013 + Sun ICML 2024 à vérifier |
| H-F1 | **Naive Bayes conditional independence** | qualify_anomaly_sbn:1069-1077 | **VIOLÉE mais admis comme biais "pro-discrimination"** — self-serving |
| H-F2 | **SBN_COND_OPINIONS = expert knowledge** | qualify_anomaly_sbn:128-500 | **CIRCULAIRE** avec injecteur (même auteur, même catalogue) |
| H-F3 | Geomean pooling Genest-Zidek 1986 | qualify_anomaly_sbn | OK logarithmic pool |
| H-F4 | UM renvoie opinion minimale | sl_formulas_v2 | Conforme ✓ |
| H-F5 | Temporal prior Hutchins 2011 | qualify_anomaly_sbn:511-577 | **CITATION ÉLASTIQUE** : numéros spécifiques (0.20, 0.15) ne sont pas dans Hutchins |
| H-F6 | Novelty threshold calibré | qualify_anomaly_sbn:1233-1237 | **NON CALIBRÉ** : hard-coded 0.82, jamais déclenché |
| H-F7 | SBN_EVIDENCE_SCALE dérivé | qualify_anomaly_sbn:1494 | **MAGIC NUMBER** 3.0 |
| H-G1 | Argmax baseline equitable | qualify_argmax_baseline | Partielle — asymétrie 12 vs 13 labels + pas de temporal prior |
| H-G2 | Baseline sans novelty detection | qualify_argmax_baseline:366 | OK par design |
| H-H1 | FP épisodes = vraies anomalies non cataloguées | Commentaire H-H1 | Hypothèse non testable sans ground truth externe |
| H-H2 | Coverage Tatbul 2018 | evaluate_injection_v2:6 | **CITATION INFLÉE** : metrics homemade, pas Tatbul |
| H-H3 | FPR SL comparable à IF à même granularité | compare_if_fair | OK |
| H-H4 | Pseudo-labels = ground truth acceptable | compare_labeller_vs_sl | **VIOLÉE** : self-référentiel |
| H-I1 | Sensitivity analysis | qualify_anomaly_sbn:1253 | **VIOLÉE** : perturbe sur la signature de référence elle-même |
| H-I2 | Ablation coverage | run_ablation_v2 | **VIOLÉE** : Phase 1b asymétrique (seule la référence rescorée) |
| H-I3 | Closed-world disclaimer | partout | À ajouter explicitement dans le paper |

---

<a id="4"></a>
## §4. Hypothèses implicites **non documentées** à ajouter au fichier de travail

Issues de l'audit, à ajouter comme H-J/H-K :

- **H-J1** : *"Training window attack-free"* — non vérifiée automatiquement ; à asserter par scan anomaly-detection cross-val sur TRAIN (5σ rule sur résidus EVT).
- **H-J2** : *"Résidus train-deploy distributionnellement stationnaires"* — à tester par Kolmogorov-Smirnov distance entre TRAIN et CALIB.
- **H-J3** : *"Autocorrélation des résidus négligeable"* — à tester par Ljung-Box (référence Ljung & Box 1978 *Biometrika*). Actuellement **violée** (ρ₁=0.983 mentionné).
- **H-J4** : *"R² in-sample = R² out-of-sample"* — VIOLÉE, à remplacer par CV R² (TimeSeriesSplit, Hastie-Tibshirani-Friedman 2009 §7.10).
- **H-J5** : *"SBN_COND_OPINIONS généralisable hors catalogue"* — NON TESTABLE sans cross-catalog evaluation (CIC-IDS2017 labels réels).
- **H-J6** : *"Novelty threshold stable à ±5 %"* — NON TESTÉE ; à faire par grille 0.60..0.85 avec ROC cross-validé.
- **H-J7** : *"Sources CBF indépendantes"* — VIOLÉE ; à disclaimer + ablation CBF→WBF.
- **H-J8** : *"Kill-chain transitions = priori Hutchins"* — VIOLÉE (numéros non dans Hutchins). À ajouter citation honest ou re-calibrer depuis MITRE ATT&CK tactics prevalence.
- **H-J9** : *"Closed-world acceptable comme proxy de performance opérationnelle"* — hypothèse structurelle du papier, à déclarer en Introduction.
- **H-J10** : *"Pas de concept drift"* — non testée. À ajouter test ADWIN (Bifet & Gavaldà 2007).

---

<a id="5"></a>
## §5. Réponses directes aux questions

### "Y a-t-il overfitting discriminatoire ?"

**OUI, par construction (CRITICAL)**. L'overfitting vient de la boucle fermée :
1. La même personne écrit les signatures d'injection (inject_at_evidence_level)
2. et la matrice de conditionnelles SBN (qualify_anomaly_sbn)
3. et évalue sur les mêmes étiquettes (evaluate_qualify_sbn)

Ce n'est pas un overfitting au sens ML classique (paramètres ajustés par gradient sur les données) mais un **overfitting d'expertise** : le système est conçu pour reconnaître les patterns qu'il est désigné pour reconnaître. Les commentaires "RENFORCÉ" et "FIX" dans le code montrent des révisions itératives après observation des résultats.

**Ce que ça invalide** : toute claim de "généralisation", "détection de nouveauté", "novelty_lr = performance hors-distribution".

**Ce que ça n'invalide pas** : la **détection** (F1=0.857, 14/14 attaques détectées) car la détection ne dépend **pas** du catalogue SBN_COND_OPINIONS — elle dépend seulement du DECISION_THRESHOLD calibré proprement sur CALIB. La détection binaire `normal vs anomaly` est **honnête**.

### "Y a-t-il des angles d'attaque (leak) ?"

**OUI, plusieurs (varying severity)** :
- **CRITICAL** : `LR_NOVELTY_THR=0.71` issu de Youden test → leak direct.
- **CRITICAL** : Phase 1b de run_ablation_v2 publie des thresholds réduits mesurés sur test.
- **CRITICAL** : IF fallback sur normals test-period.
- **MAJOR** : R² in-sample comme poids.
- **MAJOR** : FPR-matching IF sur labels test (inévitable mais mal disclaimed).
- **MAJOR** : Pseudo-labels comme GT dans compare_labeller_vs_sl.

### "Les bases scientifiques sont-elles irréprochables ?"

**Mixte** :
- **BON** : Citations riches (Jøsang 2016, Coles 2001, Taylor-Letham 2018, Grimshaw 1993, Chicco-Jurman 2020, Sharafaldin 2018).
- **PROBLÈMES** :
    - Rousseeuw-Leroy 1987 **invoqué à tort** (LAD breakdown).
    - Efron-Morris 1973 **cité sans shrinkage implémenté**.
    - Hutchins 2011 **invoqué pour numéros absents du papier**.
    - Tatbul 2018 **cité mais non implémenté** (metrics homemade).
    - Ferling 2022 **orthographe douteuse**.
    - Mercier-Denoeux 2008 **sans page/section**.

### "Les hypothèses sont-elles toutes explicitées ?"

**PARTIEL**. Le fichier "hypothèses SL_ads 18 06.txt" couvre ~30 hypothèses (A1-A5, B1-B4, C1-C4, D1-D7, E1-E3, F1-F7, G1-G2, H1-H4, I1-I3). Mais :
- **10 hypothèses implicites manquent** (H-J1 à H-J10 ci-dessus).
- **4 hypothèses déclarées vérifiées sont en réalité violées** (H-A1 non vérifiée, H-C2 violée ρ=0.983, H-D2 violée indépendance, H-F1 violée conditional indep).
- **Citations inexactes** pour 5 hypothèses (H-B2, H-D6, H-F5, H-H2, etc.)

---

<a id="6"></a>
## §6. Checklist de blindage avant soumission

### Corrections obligatoires (CRITICAL)

- [ ] **Retirer `LR_NOVELTY_THR=0.71`** des résultats publiés. Remplacer par AUC + sensitivity curve sans binarisation, OU recalibrer sur hold-out de CALIB.
- [ ] **Supprimer les colonnes Phase 1b "thr=0.15/0.10"** de `run_ablation_v2.py` OU appliquer la même réduction à toutes les variantes d'ablation.
- [ ] **Corriger le fallback IF** dans `run_ablation_v2.py:655-665` pour ne **jamais** utiliser les normals de test-period.
- [ ] **Déduplicate le catalogue** : un seul `config.INJECTED_ATTACK_CATALOG` importé partout.
- [ ] **Déclarer closed-world** en Introduction : *"Evaluation is performed on synthetic injected attacks with known labels. Generalization claims are limited to the attack families represented in SBN_COND_OPINIONS."*
- [ ] **Ajouter un LOAO** (Leave-One-Attack-Out) experiment : pour chaque k, retirer SBN_COND[k], injecter type k, mesurer si qualifier classe comme `Autre_Anomalie` (généralisation) ou force sur un autre type (overfit).
- [ ] **Ajouter une famille externe** : REPLAY_ATTACK, ARP_SPOOFING — absentes du SBN_COND, injecter, mesurer `novelty_lr` réel.

### Corrections recommandées (MAJOR)

- [ ] **Remplacer R² in-sample par CV R²** (TimeSeriesSplit n_splits=5) dans train_v10.py:1196 et 1329.
- [ ] **Ajouter bootstrap 1000 samples (Efron 1979)** sur F1/MCC pour tous les runs. Rapporter IC 95 %.
- [ ] **Ajouter McNemar's test** ou paired bootstrap pour SL vs IF.
- [ ] **Retirer la précision "sans ICMP_FLOOD_BURST"** du headline — rapporter uniquement la metric complète.
- [ ] **Retirer compare_labeller_vs_sl.py des validations** — le déclasser en "inter-annotator agreement" dans le texte.
- [ ] **Vérifier attack-freeness du TRAIN** : passer un IF ou OCSVM sur TRAIN seul, reporter le nombre d'anomalies détectées.
- [ ] **Tester stationnarité** : KS TRAIN vs CALIB sur résidus ; ADWIN sur test.
- [ ] **Tester autocorrélation** : Ljung-Box sur résidus (ρ₁=0.983 mentionné → déclustering nécessaire).
- [ ] **Disclaimer BALANCE_RATIO** : noter la déviation de Jøsang Theorem 12.2 + ablation BALANCE=1.0 vs auto.
- [ ] **Tabler de magic numbers** en annexe du papier avec justification case par case.

### Corrections cosmétiques (MINOR)

- [ ] Corriger la citation Rousseeuw-Leroy 1987 (breakdown LAD ≠ 50 %).
- [ ] Retirer la citation Efron-Morris 1973 pour EDP (pas de shrinkage).
- [ ] Vérifier Ferling 2022 / Ferling 2022 orthographe.
- [ ] Supprimer adaptive_base_rate.py si dead code confirmé.
- [ ] Harmoniser intervalles `[t0, t1)` vs `[t0, t1]`.
- [ ] Ajouter citations Fawcett 2006, Efron 1979, Stone 1974.

---

<a id="7"></a>
## §7. Plan d'expérimentation additionnelle

### Expérience #1 : LOAO (Leave-One-Attack-Out)

```python
for attack_type in SBN_COND_OPINIONS.keys():
    SBN_subset = {k:v for k,v in SBN_COND_OPINIONS.items() if k != attack_type}
    run qualify_anomaly_sbn with SBN_subset
    measure: % of (injection_label==attack_type) classified as "autre_anomalie"
    EXPECTED: ≥80% if novelty mechanism works; <20% if overfit
```

### Expérience #2 : Cross-family (externe)

Injecter 3 familles **absentes de SBN_COND** :
- `REPLAY_ATTACK` : bursts de paquets identiques (test de non-identité flow).
- `ARP_SPOOFING` : doublons MAC.
- `MALFORMED_PACKETS` : checksum invalid, TTL=1.

Mesurer : `novelty_lr` et `u_raw` sur ces injections. Si seuil 0.82/0.71 capture >70 %, la novelty detection a du sens.

### Expérience #3 : Cross-dataset (CIC-IDS2017)

Sans réécrire SBN_COND, appliquer le pipeline entier sur CIC-IDS2017 Tuesday (Brute Force SSH) et Friday (DDoS). Comparer QP/F1 à RedeRio injecté. Écart >30 % = overfit confirmé.

### Expérience #4 : Bootstrap CI

```python
from sklearn.utils import resample
F1_samples = []
for _ in range(1000):
    idx = resample(np.arange(len(y_test)))
    F1_samples.append(f1_score(y_test[idx], y_pred[idx]))
CI95 = np.percentile(F1_samples, [2.5, 97.5])
```

À appliquer à tous les F1 reportés.

### Expérience #5 : Paired McNemar SL vs IF

```python
from statsmodels.stats.contingency_tables import mcnemar
table = [[n_both_correct, n_SL_correct_IF_wrong],
         [n_SL_wrong_IF_correct, n_both_wrong]]
result = mcnemar(table, exact=True)
# reporter p-value
```

### Expérience #6 : Ablation structurée

| Variant | Bijection SL | CBF | UM | Temporal | Novelty | F1 attendu |
|---------|-------------|-----|-----|----------|---------|------------|
| Full SL-ADS | ✓ | ✓ | ✓ | ✓ | ✓ | baseline |
| -UM | ✓ | ✓ | ✗ | ✓ | ✓ | ≤ baseline |
| -CBF (sum only) | ✓ | ✗ | ✓ | ✓ | ✓ | < baseline |
| -Temporal | ✓ | ✓ | ✓ | ✗ | ✓ | < baseline |
| -Novelty gate | ✓ | ✓ | ✓ | ✓ | ✗ | = baseline sur connus, ↑FP UNKNOWN |
| Argmax | ✗ | ✗ | ✗ | ✗ | ✗ | << baseline |

Chaque variant doit avoir bootstrap 1000 et p-value paired contre Full.

---

<a id="8"></a>
## §8. Bibliographie à vérifier / compléter

### À ajouter (manques)

| Citation | Raison |
|----------|--------|
| Fawcett T. (2006) *Pattern Recognit. Lett.* 27:861-874 | ROC, y_true construction |
| Efron B. (1979) *Ann. Statist.* 7:1-26 | Bootstrap CI |
| Stone M. (1974) *J. R. Stat. Soc.* B 36:111-147 | Cross-validation (vs in-sample R²) |
| Hastie T., Tibshirani R., Friedman J. (2009) §7.10 | Elements of Statistical Learning — CV bias |
| Ljung G., Box G. (1978) *Biometrika* 65:297-303 | Autocorrelation test (pour H-C2) |
| Bifet A., Gavaldà R. (2007) SDM | ADWIN concept drift detection |
| Genest C., Zidek J. (1986) *Statist. Sci.* 1:114-135 | Logarithmic opinion pool |
| Tatbul N. et al. (2018) NeurIPS | Time-series F1 (à ne plus citer si pas implémenté) |
| McNemar Q. (1947) *Psychometrika* 12:153-157 | Paired test binary classifiers |

### À corriger

| Citation actuelle | Problème |
|-------------------|----------|
| Rousseeuw & Leroy (1987) | LAD breakdown ≠ 50 % pour leverage — **à reformuler** |
| Efron & Morris (1973) | Pas de shrinkage implémenté — **retirer ou implémenter** |
| Hutchins (2011) | Numéros transition 0.20/0.15 absents de Hutchins — **sourcer autrement** |
| Tatbul (2018) | Metrics homemade ≠ Tatbul range-based — **reformuler "inspired by"** |
| Ferling (2022) | Orthographe à vérifier |
| Ali TISSEC 2013 / Sun ICML 2024 | À vérifier (titre, auteurs) |

### À vérifier (existence et page)

- Mercier & Denoeux 2008 (cité train_v10:721 sans page)
- Sharafaldin 2018 (utilisé partout)
- Moustafa 2015 (UNSW-NB15)
- Mirsky 2018 (Kitsune)
- Rossow 2014 (botnet paper)
- Rish 2001 (Naive Bayes)
- Domingos & Pazzani 1997 (Naive Bayes robustness)

---

## Annexe A — Résumé des sévérités

| Sévérité | Count |
|----------|-------|
| CRITICAL | 13 |
| MAJOR | 34 |
| MINOR | 48 |
| OK | 52 |

**Verdict global publiable-ready ?** **NON en l'état** — 13 CRITICAL à corriger ou disclaimer explicitement. Après corrections §6, le système est **publiable avec claims ajustées** (détection robuste, qualification closed-world, novelty en perspective).

## Annexe B — Ce qui est **OK** et peut rester tel quel

- Split temporel strict train/CALIB/test ✓
- EVT/POT Pickands-Balkema-de Haan implémenté correctement ✓
- Bijection Jøsang §3.5.2 ✓
- CBF Eq. 12.14 + base-rate blending ✓
- UM Eq. 3.27 ✓
- Isolation Forest avec `sklearn_auto` contamination ✓
- Sanity check Σb+u=1 ✓
- Disclaimers "INFORMATION ONLY" sur AUC test (l'intention est bonne même si le chiffre 0.71 reste) ✓
- Guard `_meta_split_date` anti-leak ✓
- StandardScaler fit train only ✓
- Cohen's κ pour agreement ✓
- DECISION_THRESHOLD calibré FPR-quantile sur CALIB ✓

---

---

<a id="9"></a>
## §9. Patches prêts à appliquer (diffs)

Cette section fournit les corrections **prêtes à coller** pour chaque finding listé §6. Chaque patch est autonome, identifié par son finding ID, et peut être appliqué indépendamment. **Exécuter les CRITICAL avant les MAJOR.**

### Convention

Chaque patch suit le format :
```
PATCH-<ID> | Fichier:ligne(s) | Sévérité
BEFORE: <code actuel>
AFTER:  <code corrigé>
RATIONALE: <justification scientifique>
```

---

### §9.1 PATCH-C1 — Déduplication du catalogue d'attaques

**Fichiers touchés** : `config.py` (ajout), `inject_at_evidence_level.py`, `evaluate_qualify_sbn.py`, `evaluate_qualify_injected.py`, `compare_qualif_methods.py` (imports).

**Sévérité** : CRITICAL — le bug 9→13 vient de cette duplication.

#### Étape 1 — Ajouter à `config.py` (après la définition de `CONFIG`, fin de section RedeRio)

```python
# ==============================================================================
# INJECTED_ATTACK_CATALOG — source unique de vérité
# ------------------------------------------------------------------------------
# Référence cross-fichier pour :
#   - inject_at_evidence_level.py  (source d'injection)
#   - evaluate_qualify_sbn.py       (ground-truth de qualification)
#   - evaluate_qualify_injected.py  (ground-truth d'évaluation)
#   - compare_qualif_methods.py     (ground-truth de comparaison)
#
# Tout ajout/modification ici se propage automatiquement.
# Violation de DRY historique (3 copies divergentes) corrigée : PATCH-C1.
# ==============================================================================
INJECTED_ATTACK_CATALOG = [
    {'name': 'UNKNOWN_CONTROL',       'expected': None,
     'start': '2025-12-20 10:00:00',  'end': '2025-12-20 12:00:00',
     'intensity': 'high',  'is_novelty_control': True},
    {'name': 'UDP_FLOOD_DDOS',        'expected': 'UDP_FLOOD',
     'start': '2025-11-16 14:00:00',  'end': '2025-11-16 18:00:00',
     'intensity': 'extreme', 'is_novelty_control': False},
    {'name': 'SYN_FLOOD_DDOS',        'expected': 'SYN_FLOOD',
     'start': '2025-11-21 02:30:00',  'end': '2025-11-21 03:15:00',
     'intensity': 'extreme', 'is_novelty_control': False},
    {'name': 'BOTNET_CC_BEACONING',   'expected': 'BOTNET_CC',
     'start': '2025-11-19 08:00:00',  'end': '2025-11-19 12:00:00',
     'intensity': 'low',    'is_novelty_control': False},
    {'name': 'BRUTE_FORCE_SSH',       'expected': 'BRUTE_FORCE_SSH',
     'start': '2025-11-25 14:00:00',  'end': '2025-11-25 17:00:00',
     'intensity': 'medium', 'is_novelty_control': False},
    {'name': 'AGGRESSIVE_PORT_SCAN',  'expected': 'PORT_SCAN',
     'start': '2025-11-28 10:15:00',  'end': '2025-11-28 12:45:00',
     'intensity': 'medium', 'is_novelty_control': False},
    {'name': 'DATA_EXFILTRATION_SLOW','expected': 'DATA_EXFIL',
     'start': '2025-12-02 23:00:00',  'end': '2025-12-03 05:00:00',
     'intensity': 'low',    'is_novelty_control': False},
    {'name': 'NTP_AMPLIFICATION',     'expected': 'NTP_AMP',
     'start': '2025-12-04 08:00:00',  'end': '2025-12-04 11:00:00',
     'intensity': 'extreme','is_novelty_control': False},
    {'name': 'HTTP_FLOOD_L7_DDOS',    'expected': 'HTTP_FLOOD',
     'start': '2025-12-07 16:30:00',  'end': '2025-12-07 18:00:00',
     'intensity': 'high',   'is_novelty_control': False},
    {'name': 'DNS_TUNNELING',         'expected': 'DNS_TUNNELING',
     'start': '2025-12-09 10:00:00',  'end': '2025-12-09 16:00:00',
     'intensity': 'low',    'is_novelty_control': False},
    {'name': 'DNS_AMPLIFICATION',     'expected': 'DNS_AMP',
     'start': '2025-12-11 08:00:00',  'end': '2025-12-11 11:00:00',
     'intensity': 'extreme','is_novelty_control': False},
    {'name': 'SLOWLORIS_DOS',         'expected': 'SLOWLORIS',
     'start': '2025-12-15 22:00:00',  'end': '2025-12-16 06:00:00',
     'intensity': 'low',    'is_novelty_control': False},
    {'name': 'ICMP_FLOOD_BURST',      'expected': 'ICMP_FLOOD',
     'start': '2025-12-18 11:30:00',  'end': '2025-12-18 12:00:00',
     'intensity': 'extreme','is_novelty_control': False},
]
```

#### Étape 2 — Remplacer `INJECTED_ATTACKS` dans `evaluate_qualify_sbn.py` (lignes 142-207)

BEFORE :
```python
# ──────────────────────────────────────────────────────────────────────────────
# CATALOGUE DES ATTAQUES INJECTÉES
# Source unique de vérité — cohérent avec inject_at_evidence_level.py.
# ⚠️  DRY : idéalement déplacé vers config.py comme INJECTED_ATTACK_CATALOG.
# ──────────────────────────────────────────────────────────────────────────────
INJECTED_ATTACKS = [
    # ... 60 lignes de hard-code ...
]
```

AFTER :
```python
# ──────────────────────────────────────────────────────────────────────────────
# CATALOGUE DES ATTAQUES INJECTÉES — imported from config.INJECTED_ATTACK_CATALOG
# DRY-compliant depuis PATCH-C1 (audit 2026-04-18).
# ──────────────────────────────────────────────────────────────────────────────
try:
    from config import INJECTED_ATTACK_CATALOG as INJECTED_ATTACKS
except ImportError as _e:
    raise ImportError(
        "INJECTED_ATTACK_CATALOG absent de config.py — "
        "appliquer PATCH-C1 avant d'exécuter ce script"
    ) from _e
```

#### Étape 3 — Même chose dans `evaluate_qualify_injected.py` (lignes 65-121 — supprimer fallback hard-code)

AFTER :
```python
try:
    from config import INJECTED_ATTACK_CATALOG as ATTACK_CATALOG
except ImportError as _e:
    raise ImportError("Voir PATCH-C1") from _e
```

#### Étape 4 — `compare_qualif_methods.py` (lignes 77-105)

AFTER :
```python
from config import INJECTED_ATTACK_CATALOG as INJECTED_ATTACKS
```

#### Étape 5 — `inject_at_evidence_level.py` (ligne 98, ATTACK_CATALOG)

Ce fichier **reste la source métier** pour les signatures d'injection. Il doit **exporter** les noms vers `config.py` ou importer l'ordre canonique. Convention recommandée :

```python
# inject_at_evidence_level.py (NOUVEAU en tête de ATTACK_CATALOG)
from config import INJECTED_ATTACK_CATALOG as _CANONICAL_ORDER
_CANONICAL_NAMES = [a['name'] for a in _CANONICAL_ORDER]

# ... après la définition de ATTACK_CATALOG local ...

# Guard post-definition : les noms ET timestamps doivent coïncider
_local_names = [a['name'] for a in ATTACK_CATALOG]
assert _local_names == _CANONICAL_NAMES, (
    f"Divergence catalogue : injecteur={_local_names} vs config={_CANONICAL_NAMES}"
)
for a_inj, a_cfg in zip(ATTACK_CATALOG, _CANONICAL_ORDER):
    assert a_inj['start_time'] == a_cfg['start'], (
        f"Timestamp divergent pour {a_inj['name']}"
    )
```

**Vérification** après application :
```bash
python -c "from config import INJECTED_ATTACK_CATALOG; print(len(INJECTED_ATTACK_CATALOG))"
# doit afficher 13
```

---

### §9.2 PATCH-C2 — Retrait du LR_NOVELTY_THR = 0.71 dérivé de test

**Fichier** : `evaluate_qualify_sbn.py:71-77`
**Sévérité** : CRITICAL — test-on-test threshold selection.

BEFORE :
```python
LR_NOVELTY_THR = CONFIG.get('SBN_LR_NOVELTY_THRESHOLD', 0.71)  # recalibré empiriquement (was 0.85)
                               # Youden in-sample=0.734, gap max_known/unknown=0.041
```

AFTER (Option A — AUC-only reporting, recommandé pour papier) :
```python
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
```

AFTER (Option B — recalibration propre sur split CALIB hold-out) :
```python
# ⚠️  PATCH-C2 (2026-04-18) : re-calibration sur split CALIB disjoint du test.
# Protocole :
#   1. Réserver les 20 % derniers de CALIB (hors période d'injection) comme
#      "novelty calibration set" lors du train_v10.py (ajouter NOVELTY_CALIB_FRAC=0.20).
#   2. Calculer Youden sur ces données UNIQUEMENT.
#   3. Persister dans models_pkg['lr_novelty_threshold'].
#   4. Charger ici depuis le pickle — jamais depuis test.
LR_NOVELTY_THR = CONFIG.get(
    'SBN_LR_NOVELTY_THRESHOLD',
    float(_load_pickled_novelty_threshold())  # nouvelle fonction, lit du pickle train
)
```

---

### §9.3 PATCH-C3 — Fallback IF sur test-period normals interdit

**Fichier** : `run_ablation_v2.py:655-665` (`run_isolation_forest`)
**Sévérité** : CRITICAL — baseline IF peut voir des normals de test.

BEFORE :
```python
train_normal = df_train[df_train["is_attack"] == 0]
if len(train_normal) < 1000:
    # Fallback : utiliser ALL normals (y compris test-period)
    # Comment: "biais favorable a l'IF"
    train_normal = df[df["is_attack"] == 0]
```

AFTER :
```python
train_normal = df_train[df_train["is_attack"] == 0]
if len(train_normal) < 1000:
    # ⚠️  PATCH-C3 (2026-04-18) : échec STRICT — pas de fallback sur test.
    # Motif : l'ancien fallback utilisait les normals de TEST-PERIOD, ce qui
    # expose la baseline IF aux distributions de déploiement — leakage.
    # Conséquence correcte : si TRAIN < 1000 fenêtres, le run IF est abandonné
    # (donnée insuffisante) plutôt que contaminé.
    raise ValueError(
        f"IF baseline : TRAIN normal window count = {len(train_normal)} < 1000. "
        f"Leakage-free protocol exige TRAIN-only training. "
        f"Résoudre en étendant la période TRAIN dans config.SELECTED_SPLIT."
    )
```

---

### §9.4 PATCH-C4 — Phase 1b asymétrique : appliquer à toutes les variantes OU retirer

**Fichier** : `run_ablation_v2.py:1096-1115`
**Sévérité** : CRITICAL — seule la référence est rescorée aux seuils réduits.

**Option A (recommandée pour publier rapidement) — retirer Phase 1b** :
```python
# ⚠️  PATCH-C4 (2026-04-18) : bloc Phase 1b "seuils réduits 0.15 / 0.10" SUPPRIMÉ.
# Motif : rescorer uniquement la référence SL à des seuils inférieurs à
# DECISION_THRESHOLD calibré crée une asymétrie d'évaluation en faveur de la
# référence. Les variantes d'ablation n'étaient pas rescorées aux mêmes seuils.
#
# Si besoin d'une étude de sensibilité au seuil, appliquer §9.4.B ci-dessous
# (Option B) à TOUTES les variantes.
pass  # Bloc retiré intégralement. Voir git history pour l'ancienne version.
```

**Option B — appliquer la grille de seuils à toutes les variantes** :
```python
# PATCH-C4 (2026-04-18) : étude de sensibilité au seuil, symétrique.
THRESHOLD_GRID = [
    get_decision_threshold(),  # seuil calibré (référence)
    0.15,
    0.10,
]
for variant_name, variant_df in all_results.items():
    for thr in THRESHOLD_GRID:
        row = _evaluate_at_threshold(variant_df, thr, catalog=valid_catalog)
        row['variant'] = variant_name
        row['threshold'] = thr
        row['threshold_source'] = (
            'calibrated' if abs(thr - get_decision_threshold()) < 1e-9
            else 'sensitivity_grid'
        )
        summary_rows.append(row)
# Headline metric = celle au threshold 'calibrated' UNIQUEMENT.
# Les autres sont "sensitivity only" — NE PAS cherry-picker pour le papier.
```

---

### §9.5 PATCH-C5 — `_SL_FPR_TARGET` hardcodé depuis mesure test

**Fichier** : `run_ablation_v2.py:1139`
**Sévérité** : CRITICAL — le FPR cible de IF matching est lu depuis une mesure SL sur test.

BEFORE :
```python
_SL_FPR_TARGET = 0.035   # hardcodé depuis FPR SL mesuré sur test
```

AFTER :
```python
# PATCH-C5 (2026-04-18) : FPR cible sourcé depuis calibration TRAIN/CALIB,
# pas depuis une mesure TEST.
#
# Motif : l'ancien 0.035 provenait de la mesure du FPR SL sur le jeu TEST.
# Utiliser cette valeur comme cible d'ajustement pour IF implique que IF
# voit indirectement les labels TEST via la cible. Leakage structurel.
#
# Fix : lire la cible depuis CONFIG['FPR_TARGET_DECISION'] (propre, pré-enregistrée).
_SL_FPR_TARGET = CONFIG.get('FPR_TARGET_DECISION', 0.001)
print(f"  [IF FPR-match] target_fpr = {_SL_FPR_TARGET:.4f} (depuis CONFIG, not test)")
```

---

### §9.6 PATCH-C6 — Closed-world disclaimer en Introduction du paper

**Fichier** : `docs/review/PUBLICATION_TABLES.md` — nouvelle section §0bis.

```markdown
## §0bis. Déclaration méthodologique obligatoire

### Protocole d'évaluation : closed-world synthétique

Cette étude évalue SL-ADS sur un catalogue de **13 familles d'attaques injectées
synthétiquement** (inject_at_evidence_level.py, config.INJECTED_ATTACK_CATALOG) dont
les signatures d'injection et les opinions conditionnelles `SBN_COND_OPINIONS` ont
été **co-conçues** (Sharafaldin 2018, Mirsky 2018, Rossow 2014).

**Limitations de validité** :

1. **Closed-world** : le catalogue d'injection est identique au vocabulaire
   appris par le qualificateur. Les résultats QP/F1 rapportés §3 mesurent
   la **cohérence interne** du pipeline, pas sa **généralisation**.
2. **Pas de famille externe testée** : aucune attaque absente du
   `SBN_COND_OPINIONS` n'est injectée. Le test de nouveauté
   (`UNKNOWN_ANOMALY_CONTROL`) est lui-même hand-designed.
3. **Claim limitée** : les performances rapportées sont celles d'un
   **prototype** sur données synthétiques. Extrapolation à un déploiement
   opérationnel nécessite (a) LOAO, (b) cross-dataset (CIC-IDS2017,
   UNSW-NB15), (c) famille externe injectée (REPLAY, ARP_SPOOF, MALFORMED).

Conforme à Varma & Simon (2006) et aux guidelines NeurIPS 2024 sur la
reproductibilité (§Reproducibility checklist).
```

---

### §9.7 PATCH-M1 — R² in-sample → CV R²

**Fichiers** : `train_v10.py:1194-1196` et `train_v10.py:1328-1329`
**Sévérité** : MAJOR — biais optimiste Stone 1974.

BEFORE (1194-1196) :
```python
reg.fit(X, y)
preds = reg.predict(X)
r2 = r2_score(y, preds)   # IN-SAMPLE R²
```

AFTER :
```python
# PATCH-M1 (2026-04-18) : Time-series CV R² remplace l'in-sample R².
# Motif : l'in-sample R² est optimistement biaisé (Stone 1974 JRSS B ;
# Hastie-Tibshirani-Friedman 2009 §7.10). Utilisé comme trust_score en WBF,
# il inflate les poids des sources sur-fittées.
from sklearn.model_selection import TimeSeriesSplit

if len(X) >= 50:  # minimum pour 5-fold TS split
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for tr_idx, val_idx in tscv.split(X):
        _reg_cv = type(reg)(**reg.get_params())
        _reg_cv.fit(X[tr_idx], y[tr_idx])
        cv_scores.append(r2_score(y[val_idx], _reg_cv.predict(X[val_idx])))
    r2 = float(np.mean(cv_scores))  # R² CV-averaged
    r2_std = float(np.std(cv_scores))
else:
    # Fallback pour métriques rares : in-sample avec warning explicite
    import warnings
    warnings.warn(
        f"{key}: n={len(X)} < 50, CV impossible. In-sample R² retained — "
        f"trust_score biaisé (voir PATCH-M1).", UserWarning
    )
    reg.fit(X, y)
    r2 = r2_score(y, reg.predict(X))
    r2_std = float('nan')

# Re-fit final sur toutes les données (après CV) pour persister le modèle
reg.fit(X, y)
preds = reg.predict(X)  # pour diagnostics plots only

# Logger les deux valeurs pour traçabilité
print(f"  {key}: R²_CV = {r2:+.3f} ± {r2_std:.3f} "
      f"(in-sample = {r2_score(y, preds):+.3f})")
```

BEFORE (1328-1329, Prophet) :
```python
y_pred = model.predict(df_prophet)['yhat']
r2 = r2_score(y_true, y_pred)   # IN-SAMPLE
```

AFTER :
```python
# PATCH-M1 bis (2026-04-18) : Prophet CV R² via time-series rolling-origin.
from prophet.diagnostics import cross_validation, performance_metrics

try:
    df_cv = cross_validation(
        model, initial='14 days', period='3 days', horizon='1 day',
        parallel=None, disable_tqdm=True
    )
    df_perf = performance_metrics(df_cv, rolling_window=1.0)
    r2 = 1.0 - (df_perf['mse'].iloc[0] / np.var(y_true))
    print(f"  {key}: Prophet R²_CV = {r2:+.3f} (rolling-origin)")
except Exception as _e:
    # Fallback in-sample si CV impossible (ex. historique trop court)
    import warnings
    warnings.warn(f"{key}: Prophet CV failed ({_e}) — in-sample R² retained.")
    y_pred = model.predict(df_prophet)['yhat']
    r2 = r2_score(y_true, y_pred)
```

---

### §9.8 PATCH-M2 — Bootstrap IC 95 % sur F1/MCC

**Fichier** : `evaluate_injection_v2.py` après ligne 480 (`f1_pos` calculé)
**Sévérité** : MAJOR — aucune stat inférentielle.

AFTER (à insérer après calcul des métriques ponctuelles) :
```python
# ─── PATCH-M2 (2026-04-18) : Bootstrap IC 95 % (Efron 1979) ────────────────
from sklearn.utils import resample

N_BOOTSTRAP = 1000
RNG = np.random.default_rng(seed=42)   # reproducibility

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

# Application aux métriques principales
f1_mean, f1_lo, f1_hi    = _bootstrap_ci(y_true_all, y_pred_all, f1_score)
mcc_mean, mcc_lo, mcc_hi = _bootstrap_ci(y_true_all, y_pred_all, matthews_corrcoef)
print(f"  F1 = {f1_mean:.3f} [IC95% : {f1_lo:.3f} – {f1_hi:.3f}]")
print(f"  MCC = {mcc_mean:.3f} [IC95% : {mcc_lo:.3f} – {mcc_hi:.3f}]")

# Persistance CSV
bootstrap_ci_row = {
    'metric': 'f1_binary',  'mean': f1_mean, 'ci_lo': f1_lo, 'ci_hi': f1_hi,
    'n_bootstrap': N_BOOTSTRAP, 'seed': 42
}
# append to eval_bootstrap_ci.csv
```

---

### §9.9 PATCH-M3 — McNemar paired test SL vs IF

**Fichier** : `compare_if_fair.py` après ligne 637 (après calcul des métriques des 4 systèmes)
**Sévérité** : MAJOR — pas de test statistique.

AFTER :
```python
# ─── PATCH-M3 (2026-04-18) : McNemar paired test (Dietterich 1998) ─────────
from statsmodels.stats.contingency_tables import mcnemar

def _mcnemar_sl_vs_if(y_true, y_pred_sl, y_pred_if, label=""):
    """Test de McNemar (1947 Psychometrika) pour classifiers appariés."""
    # Table de contingence : les prédictions correctes/incorrectes des deux systèmes
    both_ok  = int(((y_pred_sl == y_true) & (y_pred_if == y_true)).sum())
    sl_ok_if_ko = int(((y_pred_sl == y_true) & (y_pred_if != y_true)).sum())
    sl_ko_if_ok = int(((y_pred_sl != y_true) & (y_pred_if == y_true)).sum())
    both_ko  = int(((y_pred_sl != y_true) & (y_pred_if != y_true)).sum())
    table = [[both_ok, sl_ok_if_ko], [sl_ko_if_ok, both_ko]]
    result = mcnemar(table, exact=True)
    print(f"  McNemar SL vs {label}: statistic={result.statistic:.3f}, "
          f"p-value={result.pvalue:.4f}")
    return {'label': label, 'table': table,
            'statistic': float(result.statistic),
            'pvalue': float(result.pvalue)}

mcnemar_results = []
for if_name, if_preds in [('IF-fair', if_preds_fair),
                          ('IF-fpr-matched', if_preds_matched),
                          ('IF-k1', if_preds_k1)]:
    mcnemar_results.append(_mcnemar_sl_vs_if(y_test, sl_preds, if_preds, label=if_name))

pd.DataFrame(mcnemar_results).to_csv(
    os.path.join(out_dir, 'mcnemar_sl_vs_if.csv'), index=False
)
```

---

### §9.10 PATCH-M4 — Retirer headline "sans ICMP_FLOOD_BURST"

**Fichier** : `evaluate_qualify_injected.py:205-213`
**Sévérité** : MAJOR — cherry-pick-friendly reporting.

BEFORE :
```python
macro_precision_all = np.mean([a['precision'] for a in attack_stats])
macro_precision_excl_icmp = np.mean([
    a['precision'] for a in attack_stats if a['name'] != 'ICMP_FLOOD_BURST'
])
# reported both in summary
```

AFTER :
```python
# PATCH-M4 (2026-04-18) : headline unique = macro_precision sur TOUT le catalogue.
# Motif : reporter deux variantes (avec/sans ICMP) induit un cherry-pick risk
# où le lecteur retient le meilleur chiffre. Si ICMP échoue, c'est un finding
# à discuter en §Limites, pas à masquer dans une variante alternative.
macro_precision = np.mean([a['precision'] for a in attack_stats])

# Le cas ICMP_FLOOD_BURST = 0% precision est documenté SÉPARÉMENT en §Limites
# (explication : sous-domaine Bernoulli de la bijection à W=3 insuffisant
# pour un burst de 6 fenêtres — non un problème SL, un problème d'unité de décision).
icmp_row = [a for a in attack_stats if a['name'] == 'ICMP_FLOOD_BURST']
if icmp_row:
    print(f"  ICMP_FLOOD_BURST précision = {icmp_row[0]['precision']:.1%} "
          f"(documenté §Limites : burst < W×WINDOW_SIZE)")
```

---

### §9.11 PATCH-m1 — Citation Rousseeuw & Leroy breakdown LAD

**Fichier** : `train_v10.py:1144`
**Sévérité** : MINOR — citation inexacte.

BEFORE :
```
#     • Breakdown point 50% > RANSAC pratique (~47%) — Rousseeuw & Leroy (1987)
```

AFTER :
```
#     • LAD (QR q=0.5) est robuste aux outliers de RÉPONSE (Koenker &
#       Bassett 1978 Econometrica 46:33-50) — breakdown point jusqu'à 50 %
#       pour les outliers verticaux. N.B. : LAD n'est PAS robuste aux
#       outliers de LEVIER (leverage) — breakdown 0 % (Rousseeuw & Leroy
#       1987, Robust Regression §3.3). RANSAC est complémentaire pour
#       les leverage outliers. Notre usage (bytes ← packets avec
#       fit_intercept=False, Bridgman 1922) évite les leverage outliers
#       par contrainte physique.
```

---

### §9.12 PATCH-m2 — Citation Efron & Morris EDP

**Fichier** : `train_v10.py:549` (docstring EDP)
**Sévérité** : MINOR — citation inexacte (pas de shrinkage James-Stein).

BEFORE :
```
#    Empirical Dirichlet Prior — Efron & Morris 1973 JASA.
```

AFTER :
```
#    Empirical Dirichlet Prior — estimateur fréquentiste marginal.
#    Conforme à Ferguson (1973) "A Bayesian analysis of some nonparametric
#    problems" Ann. Stat. 1:209-230 (Dirichlet process prior), spécialisé
#    en Empirical Bayes (Robbins 1955, Robbins 1983).
#    N.B. : ne pas citer Efron & Morris 1973 ici — cet article porte sur
#    les estimateurs de James-Stein à rétrécissement (shrinkage), non
#    implémenté dans notre EDP. Nos base rates sont des fréquences
#    empiriques simples sans shrinkage.
```

---

### §9.13 PATCH-m3 — FPR_TARGET_DECISION fallback 1.0 dangereux

**Fichier** : `train_v10.py:1542`
**Sévérité** : MINOR (bug latent).

BEFORE :
```python
fpr_target = CONFIG.get("FPR_TARGET_DECISION", 1.0)
```

AFTER :
```python
# PATCH-m3 (2026-04-18) : fallback dangereux 1.0 supprimé.
# Motif : si la clé config est absente, fallback à 1.0 → quantile(..., 0) =
# min(proj_atk) → threshold = minimum absolu → 100 % positive rate.
# Solution : échec explicite au lieu de comportement silencieusement désastreux.
if "FPR_TARGET_DECISION" not in CONFIG:
    raise KeyError(
        "FPR_TARGET_DECISION manquant dans CONFIG — configurer explicitement "
        "(valeur recommandée : 0.001 pour RedeRio, 0.01 pour METR-LA)."
    )
fpr_target = float(CONFIG["FPR_TARGET_DECISION"])
assert 0 < fpr_target < 1, (
    f"FPR_TARGET_DECISION={fpr_target} doit être strictement dans (0, 1)"
)
```

---

### §9.14 PATCH-m4 — BALANCE_RATIO déviation Jøsang Theorem 12.2

**Fichier** : `compute_opinions_v3.py` — docstring à ajouter au début de la fonction qui applique BALANCE_RATIO (lignes 293-300).

AFTER :
```python
def _apply_balance_ratio(evidence_prophet, evidence_reconst, mode='auto'):
    """
    Rééquilibrage N_prophet vs N_reconst avant fusion CBF.

    ⚠️  MODIFICATION AU FORMALISME JØSANG (PATCH-m4, 2026-04-18) :
    Jøsang (2016) Theorem 12.2 énonce l'additivité des évidences CBF
    (r_A+B = r_A + r_B). Notre BALANCE_RATIO viole cette additivité en
    rescalant les évidences avant fusion : r_A' = α·r_A, r_B' = β·r_B
    avec α·|metrics_A| = β·|metrics_B|.

    Motivation empirique : sans rééquilibrage, les 12 métriques Prophet
    dominent arithmétiquement les 5 métriques Reconstruction (ratio 2.4:1)
    dans la somme d'évidences, biasant la CBF vers le signal Prophet.

    Cette modification doit être explicitement mentionnée dans le papier
    (§Method, §Limitations) et validée par ablation (BALANCE_RATIO=1.0).
    """
    # ... corps de la fonction ...
```

---

### §9.15 PATCH-m5 — `_sensitivity_analysis` perturbe sur la référence

**Fichier** : `qualify_anomaly_sbn.py:1253-1353`
**Sévérité** : MINOR — measure la consistance interne, pas la robustesse externe.

Ajouter un commentaire/docstring (pas de changement fonctionnel) :

AFTER :
```python
def _sensitivity_analysis(group_pp, ...):
    """
    Analyse de sensibilité des opinions SBN à ±0.05.

    ⚠️  LIMITE MÉTHODOLOGIQUE (documentée PATCH-m5, 2026-04-18) :
    Cette analyse perturbe SBN_COND_OPINIONS autour de sa valeur de référence,
    puis mesure la variation de _compute_group_projected par rapport à
    cette même référence. Elle mesure donc la **consistance interne** du
    mapping, pas la **robustesse au bruit réel** d'observation.

    Pour une vraie étude de robustesse, perturber les ÉVIDENCES d'entrée
    (ev_safe, ev_susp, ev_atk) par un bruit gaussien calibré sur la variance
    empirique des résidus — non fait actuellement (cf. SCIENTIFIC_AUDIT §1.5.J).
    """
    # ... corps ...
```

---

### §9.16 PATCH-m6 — Déclassifier compare_labeller_vs_sl

**Fichier** : `compare_labeller_vs_sl.py:1-10` (docstring en tête)

AFTER :
```python
"""
compare_labeller_vs_sl.py — COMPARAISON (pas validation) SL-ADS vs ConsensusLabeller.

⚠️  CLASSIFICATION MÉTHODOLOGIQUE (PATCH-m6, 2026-04-18) :
Ce script compare les décisions SL-ADS aux pseudo-labels produits par
ConsensusLabeller. Puisque les deux systèmes opèrent sur le même dataset
sans ground truth externe, les métriques obtenues mesurent un
**accord inter-annotateurs** (inter-annotator agreement, Cohen 1960), NON
une validation externe du SL-ADS.

À NE PAS rapporter comme "F1 de SL-ADS" ou "validation". À rapporter comme
"agreement (κ) entre SL et Consensus" en §Reliability.
"""
```

---

<a id="10"></a>
## §10. Tracker de résolution

### Mise à jour 2026-04-19 — Application des patches

Suite à la demande utilisateur d'appliquer tous les patches et de fixer
les bugs rapportés dans les diffs §9 (variables indéfinies, fonctions
inexistantes, shadow-imports), l'ensemble des 16 patches a été appliqué
et vérifié syntaxiquement (`python -m py_compile`). Le tableau ci-dessous
documente l'état précis de chaque patch post-application.

| Patch ID | Finding lié | Sévérité | Statut | Date | py_compile |
|----------|-------------|----------|--------|------|------------|
| PATCH-C1 | Duplication catalogue (§2.3) | CRITICAL | ✅ Appliqué (fix mutation + shadow) | 2026-04-19 | OK |
| PATCH-C2 | LR_NOVELTY_THR=0.71 test-derived (§0 #2) | CRITICAL | ✅ Appliqué | 2026-04-18 | OK |
| PATCH-C3 | IF fallback test-period normals (§1.14) | CRITICAL | ✅ Appliqué (raise ValueError) | 2026-04-19 | OK |
| PATCH-C4 | Phase 1b asymétrique (§1.14) | CRITICAL | ✅ Appliqué (evaluate_run) | 2026-04-19 | OK |
| PATCH-C5 | _SL_FPR_TARGET hardcodé test (§1.14) | CRITICAL | ✅ Appliqué | 2026-04-18 | OK |
| PATCH-C6 | Closed-world disclaimer (§2.1) | CRITICAL | ✅ Appliqué (dédup §0bis) | 2026-04-19 | N/A (.md) |
| PATCH-M1 | R² in-sample → CV R² (§0 #3) | MAJOR | ✅ Appliqué (key→metric, r2_global→r2, y_pred scope) | 2026-04-19 | OK |
| PATCH-M2 | Bootstrap IC 95 % (§0 #4) | MAJOR | ✅ Appliqué (y_true/y_pred + imports + CI dans CSV) | 2026-04-19 | OK |
| PATCH-M3 | McNemar SL vs IF (§2.4) | MAJOR | ✅ Appliqué (vars + out_dir) | 2026-04-19 | OK |
| PATCH-M4 | ICMP-excluded headline (§1.10) | MAJOR | ✅ Appliqué (macro-précision unique) | 2026-04-19 | OK |
| PATCH-m1 | Rousseeuw-Leroy citation (§1.1.H) | MINOR | ✅ Appliqué | 2026-04-18 | OK |
| PATCH-m2 | Ferguson citation (§1.1.H) | MINOR | ✅ Appliqué | 2026-04-18 | OK |
| PATCH-m3 | FPR_TARGET fallback 1.0 (§1.1.I) | MINOR | ✅ Appliqué (raise KeyError) | 2026-04-18 | OK |
| PATCH-m4 | BALANCE_RATIO Theorem 12.2 (§1.4) | MINOR | ✅ Appliqué (tag + déviation explicite) | 2026-04-19 | OK |
| PATCH-m5 | Sensitivity on reference (§1.5) | MINOR | ✅ Appliqué | 2026-04-18 | OK |
| PATCH-m6 | Labeller déclassification (§1.12) | MINOR | ✅ Appliqué (docstring cadre épistémo) | 2026-04-19 | OK |

### Légende

- ✅ Appliqué + vérifié (`python -m py_compile` OK)
- 🔄 Appliqué, test comportemental en cours
- 📝 Documenté (diff §9 prêt, non-appliqué)
- 🚧 À appliquer dans le code
- ⏸️  Bloqué (ex. nécessite expérience additionnelle)

### Détail des bugs corrigés pendant l'application (2026-04-19)

Les 16 diffs initiaux (session audit 2026-04-18) contenaient plusieurs
bugs de variables indéfinies / fonctions inexistantes, détectés par
l'utilisateur au moment de leur application manuelle. Ces bugs ont été
corrigés et re-appliqués automatiquement :

#### PATCH-M1 (`train_v10.py`)
- **Bug 1** : `{key}` utilisé dans les `warnings.warn` / `print` alors
  que la variable de boucle est `metric` (ligne 1136).
  → **Fix** : remplacement `{key}` → `{metric}` (lignes 1227, 1239,
  1382, 1386).
- **Bug 2** : `r2_global` référencé après suppression de cette variable
  (remplacée par `r2` dans le bloc CV).
  → **Fix** : remplacement `r2_global` → `r2` (lignes 1250, 1251, 1258,
  1292, 1311).
- **Bug 3 (scope Prophet)** : `y_pred` utilisé ligne 1389 après le bloc
  `try/except`, mais défini UNIQUEMENT dans la branche `except`. Si la
  CV réussit, `y_pred` est undefined.
  → **Fix** : calcul de `fcst = model.predict(df_prophet)` et
  `y_pred = fcst.loc[mask_valid, 'yhat'].values` AVANT le `try/except`,
  de sorte que `res_signed = y_true - y_pred` est toujours défini.

#### PATCH-C4 (`run_ablation_v2.py` ligne 1104)
- **Bug** : appel à `_evaluate_at_threshold(variant_df, thr, catalog=valid_catalog)`,
  fonction **inexistante** dans le module.
- **Fix** : utilisation de `evaluate_run(variant_df, valid_catalog,
  thresholds_override=[thr])` (fonction réelle définie ligne 707), puis
  extraction de la 1re (et unique) ligne du sweep via `sweep.iloc[0]`.
  Construction d'un dict homogène avec `to_summary()` pour conserver
  la cohérence des colonnes de `summary_rows`.

#### PATCH-C3 (`run_ablation_v2.py` lignes 655-666)
- **État initial** : le code avait un fallback silencieux
  (`X_train = X[normal_mask.values]`) qui exposait IF à la distribution
  du trafic normal de la **période de test**, créant une fuite structurelle.
- **Fix** : remplacement par un `raise ValueError` explicite si
  `n_train < 1000` (seuil minimal pour entraînement IF), avec message
  d'erreur citant Varma & Simon (2006). Le study échoue bruyamment
  plutôt que de produire un baseline biaisé.

#### PATCH-M2 (`evaluate_injection_v2.py`)
- **Bug 1** : variables `y_true_all` / `y_pred_all` référencées aux
  lignes 470-471 mais **jamais définies** dans le scope.
- **Fix** : utilisation de `y_true` (ligne 370, hors boucle) et
  `y_pred` (ligne 422, dans la boucle). Les deux sont déjà définis.
- **Bug 2** : `f1_score` et `matthews_corrcoef` utilisés mais non-importés.
- **Fix** : ajout de `from sklearn.metrics import f1_score, matthews_corrcoef`
  au niveau module (après les autres imports sklearn).
- **Bug 3** : import mort `from sklearn.utils import resample` (non utilisé).
- **Fix** : supprimé.
- **Bug 4** : `bootstrap_ci_row` construit mais **jamais écrit au CSV**.
- **Fix** : ajout des 8 colonnes `f1_mean_boot`, `f1_ci_lo`, `f1_ci_hi`,
  `mcc_mean_boot`, `mcc_ci_lo`, `mcc_ci_hi`, `n_bootstrap`,
  `bootstrap_seed` directement à la ligne du sweep, de sorte que
  `eval_threshold_sweep.csv` contient nativement les IC.

#### PATCH-M3 (`compare_if_fair.py` lignes 659-665)
- **Bug 1** : `sl_preds` (pluriel) référencé, mais la variable définie
  dans le scope est `common["sl_pred"]` (singulier, ligne 563).
- **Fix** : `sl_preds_arr = common["sl_pred"].astype(int).values`.
- **Bug 2** : `if_preds_fair`, `if_preds_matched`, `if_preds_k1` référencés,
  mais seul `if_pred_k1` (singulier, ligne 572) existe. Les deux autres
  doivent être **calculés** à partir des colonnes disponibles.
- **Fix** :
  - `if_preds_fair_arr = common["if_pred"].astype(int).values`
  - `if_preds_matched_arr = (if_scores_win >= if_fpr_matched.threshold).astype(int)`
  - `if_pred_k1` utilisé tel quel.
- **Bug 3** : `out_dir` référencé ligne 665, **jamais défini**. Les
  autres CSV utilisent `args.output_dir`.
- **Fix** : `os.path.join(args.output_dir, 'mcnemar_sl_vs_if.csv')`.

#### PATCH-C1 (`evaluate_qualify_injected.py` lignes 64-86)
- **Bug** : import `from config import INJECTED_ATTACK_CATALOG as ATTACK_CATALOG`
  puis mutation via `.append()` de la liste importée. Toute ré-import
  dans le même processus (tests, notebook) recevait un catalogue déjà
  dupliqué.
- **Fix** : copie défensive `ATTACK_CATALOG = [dict(atk) for atk in
  _INJECTED_CATALOG_CANONICAL]` puis merge en évitant les doublons via
  un set `_canonical_names`.

#### PATCH-C1 (`compare_qualif_methods.py` lignes 78-106)
- **Bug** : hard-code local de `INJECTED_ATTACKS = [...]` qui **shadow**
  l'import ligne 44. Divergence sur `UNKNOWN_ANOMALY_CONTROL` vs canonical
  `UNKNOWN_CONTROL`, et seulement **9 attaques** vs **13** canoniques.
- **Fix** : suppression complète du bloc hard-code (lignes 78-106) ;
  la variable `INJECTED_ATTACKS` pointe désormais vers l'import canonique
  de la ligne 44. `KNOWN_ATTACKS` / `NOVELTY_ATTACKS` dérivées inchangées.

#### PATCH-C6 (`docs/review/PUBLICATION_TABLES.md`)
- **Bug** : le bloc §0bis était dupliqué aux lignes 33-56 ET 56-79, sans
  séparateur de ligne entre les deux (concaténation accidentelle lors
  d'une précédente application).
- **Fix** : conservation d'un seul bloc §0bis (lignes 33-56).

#### PATCH-M4 (`evaluate_qualify_injected.py` lignes 188-210)
- **État initial** : dual reporting `macro_precision_known` (avec ICMP)
  ET `macro_precision_no_icmp` (sans ICMP_FLOOD_BURST). Cherry-pick implicite
  post-hoc d'une métrique favorable.
- **Fix** : suppression du calcul et du print de `macro_precision_no_icmp`.
  Seule la macro-précision incluant tous les failure modes est rapportée ;
  les per-attack failures sont documentés dans la Table 2 per-attack.

#### PATCH-m6 (`compare_labeller_vs_sl.py` docstring module)
- **État initial** : docstring décrivait la comparaison sans cadre
  épistémologique. Risque que le lecteur interprète κ/accord comme
  une validation SL-ADS.
- **Fix** : ajout d'un bloc "ATTENTION" de 30 lignes déclarant explicitement
  que ce script produit un **accord inter-annotateurs** (Cohen 1960 ;
  Landis & Koch 1977 ; Artstein & Poesio 2008), pas une validation de
  performance. Interprétations admissibles et interdites listées.

#### PATCH-m4 (`compute_opinions_v3.py` lignes 293-321)
- **État initial** : commentaire décrivait le balance_ratio comme
  application directe de Théorème 12.2 Jøsang.
- **Fix** : bloc commentaire déclarant explicitement que BALANCE_RATIO
  est une **EXTENSION HEURISTIQUE** du cadre CBF (addition de preuves),
  plus proche d'un "evidence averaging" (famille WBF §12.3) que du CBF
  strict. Alternative `INTER_METHOD_FUSION="hierarchical"` citée pour
  l'ablation du paper.

### Bug additionnel découvert pendant l'application (2026-04-19) — PATCH-C1 guard

Vérification comportementale : l'import de `inject_at_evidence_level.py`
après application initiale de PATCH-C1 échouait avec :

```
AssertionError: Divergence catalogue :
    injecteur=['UNKNOWN_ANOMALY_CONTROL', 'UDP_FLOOD_DDOS', ...]
    config=  ['UNKNOWN_CONTROL', 'UDP_FLOOD_DDOS', ...]
```

Double problème découvert :

1. **Nom divergent** : config.py utilisait `UNKNOWN_CONTROL` (8 chars),
   tandis que l'injecteur et tous les évaluateurs downstream utilisent
   `UNKNOWN_ANOMALY_CONTROL` (17 chars, aligné sur MITRE-style).
2. **Ordre divergent** : l'assertion comparait les listes par ordre de
   déclaration ; l'injecteur et config déclarent les 13 attaques dans
   un ordre différent (sémantiquement équivalent).
3. **Clé divergente** : l'assertion timestamp utilisait `a_inj['start_time']`
   alors que l'injecteur déclare `'start'`.

**Fixes appliqués** :
  - `config.py` L.897 : `UNKNOWN_CONTROL` → `UNKNOWN_ANOMALY_CONTROL`
  - `inject_at_evidence_level.py` L.817-830 : guard re-écrit en
    **bijection par nom** (set-comparison + dict lookup par nom), au lieu
    d'égalité stricte de liste ordonnée. Clé corrigée `'start_time'` → `'start'`.
  - `compare_qualif_methods.py` L.16 et L.83 : commentaires alignés sur
    le nom canonique.

Résultat : `python -c "import inject_at_evidence_level"` → OK, 13 attacks.

### Vérification finale

```bash
cd "actual_ version_claude_autre dataset"
python -m py_compile train_v10.py run_ablation_v2.py \
    evaluate_injection_v2.py compare_if_fair.py \
    evaluate_qualify_injected.py compare_qualif_methods.py \
    compute_opinions_v3.py compare_labeller_vs_sl.py \
    inject_at_evidence_level.py config.py
# → OK (exit 0, aucune erreur de syntaxe)

python -c "import inject_at_evidence_level; \
  print('bijection OK,', len(inject_at_evidence_level.ATTACK_CATALOG), 'attacks')"
# → bijection OK, 13 attacks
```

### Smoke-tests comportementaux (2026-04-19)

En plus de `py_compile`, 5 smoke-tests ont été exécutés avec succès :

```
[OK] inject_at_evidence_level guard passes, 13 attacks
[OK] config.INJECTED_ATTACK_CATALOG has UNKNOWN_ANOMALY_CONTROL
[OK] compare_qualif_methods: 13 total, 12 known, 1 novelty
[OK] run_ablation_v2.evaluate_run callable (PATCH-C4 fix verified)
[OK] evaluate_injection_v2 has sklearn.metrics imports (PATCH-M2 fix verified)
=== All post-application smoke tests PASS ===
```

Ces tests vérifient :
1. Le guard de bijection PATCH-C1 passe à l'import (13 attaques chargées).
2. Le catalogue canonique contient bien `UNKNOWN_ANOMALY_CONTROL`.
3. `compare_qualif_methods` partitionne correctement 12 known + 1 novelty.
4. `run_ablation_v2.evaluate_run` (fonction substituée à
   `_evaluate_at_threshold` par PATCH-C4) est callable.
5. `evaluate_injection_v2` expose `f1_score` et `matthews_corrcoef` (PATCH-M2).

Ces smoke-tests ne couvrent PAS la cohérence numérique des résultats
produits (qui nécessite un re-run complet du pipeline). Ils garantissent
seulement l'absence de bugs bloquants à l'import / appel.

### Bug additionnel découvert pendant la ré-exécution (2026-04-20) — PATCH-C4 followup

Pendant le run complet du pipeline post-patches, `run_ablation_v2.py` a crashé à la Phase 3 avec :

```
File "run_ablation_v2.py", line 1103, in main
    get_decision_threshold(),  # seuil calibré (référence)
TypeError: get_decision_threshold() missing 1 required positional argument: 'config'
```

**Cause racine** : `get_decision_threshold` (de `paths.py`) requiert un argument `config` (comme utilisé partout ailleurs dans le même fichier : L.102, L.389). Le grid de seuils Phase 3 utilisait la signature sans argument, obsolète.

**Impact** : l'ablation produit correctement les 30+ variantes Phase 1 (Full SL-ADS, UM=True, No C1, No CBF, W=2/3/4, λ=0/0.5/0.85/0.99, etc.) mais crashe avant d'écrire `ablation_summary.csv`. Les F1 de chaque variante étaient affichés console, donc utilisables pour remplir la Table 3 de `PUBLICATION_TABLES.md` §9bis manuellement.

**Correction appliquée** : ligne 1103 de `run_ablation_v2.py`,
```python
# Avant
get_decision_threshold(),
# Après
get_decision_threshold(CONFIG, up_levels=1),
```

**Vérification** : `py_compile` OK, ré-exécution lancée en arrière-plan (2026-04-20 10:45+).

### Second bug découvert pendant la re-ré-exécution (2026-04-20 16h) — PATCH-C4 followup²

Après correction de `get_decision_threshold`, la ré-exécution d'ablation (`bmbyaf5ra` puis `b0vo40b5v`) sortait en `exit 0` mais **sans produire `ablation_summary.csv`**. La capture du `stdout`/`stderr` complet a révélé un crash Unicode sur Windows Python 3.13 :

```
File "run_ablation_v2.py", line 1093, in main
    print(f"\n-> [{run_name}]")
UnicodeEncodeError: 'charmap' codec can't encode character '\u03bb' in position 19: character maps to <undefined>
```

**Cause racine** : sur Windows, `sys.stdout` hérite par défaut du codec `cp1252` qui ne peut pas encoder `λ` (U+03BB), `α` (U+03B1) présents dans les noms de variantes (ex. `"No C1 — fixed λ (conflict-aware off)"`). Le shell parent interprétait le traceback comme un texte de sortie et renvoyait `exit 0` — d'où la confusion initiale (CSV absent mais exit code bon).

**Correction appliquée** : ajout d'une reconfiguration UTF-8 forcée au début du script (après `import sys`) :

```python
# PATCH-C4 fix (2026-04-20) : force stdout UTF-8 sur Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # Python <3.7 ou stdout non-TTY : ignorer
```

**Vérification** : `py_compile` OK, ré-exécution lancée (tâche `bltu41u6g` 2026-04-20 16:10+) avec `PYTHONIOENCODING=utf-8` en ceinture-et-bretelles.

### Troisième bug découvert pendant la re-re-ré-exécution (2026-04-20 16h30) — PATCH-C4 followup³

Le 3e run (UTF-8 patch en place) a tourné jusqu'à la fin de Phase 1 (32 variantes calculées, tous les F1 affichés en console) puis crashé à la boucle de labelling `threshold_source` :

```
File "run_ablation_v2.py", line 1148, in main
    'calibrated' if abs(thr - get_decision_threshold()) < 1e-9
TypeError: get_decision_threshold() missing 1 required positional argument: 'config'
```

**Cause racine** : une *seconde occurrence* de `get_decision_threshold()` sans argument, différente de celle corrigée ligne 1103 (grid de seuils), utilisée pour distinguer `'calibrated'` vs `'sensitivity_grid'` dans le CSV de sortie.

**Correction appliquée** : ligne 1148 de `run_ablation_v2.py`,
```python
# Avant
'calibrated' if abs(thr - get_decision_threshold()) < 1e-9
# Après (PATCH-C4 fix²)
'calibrated' if abs(thr - get_decision_threshold(CONFIG, up_levels=1)) < 1e-9
```

**Vérification finale** : `grep -n "get_decision_threshold(" run_ablation_v2.py` → 4 occurrences, toutes avec signature `(CONFIG, up_levels=1)`. `py_compile` OK. 4e run lancé (tâche `bzwl0r88h` 2026-04-20 16h30).

**Cross-validation §9bis Table 3** : les 30+ variantes affichées dans le log du 3e run (avant le crash L1148) **correspondent exactement aux valeurs pré-remplies dans §9bis** de `PUBLICATION_TABLES.md` — la table publication était déjà correcte, le CSV final est un artefact de reproductibilité.

### Quatrième bug — Phase 2 baselines abort CSV write (2026-04-20 17h)

Le 4e run atteint Phase 2 sans problème, mais la sortie d'erreur `PATCH-C3 IsolationForest: only 0 training windows before split_date` (attendu ici, car l'evidence CSV `*_attacks.csv` est post-split uniquement) est une `ValueError` non rattrapée → Phase 3 (save CSV) jamais atteinte.

**Diagnostic** : PATCH-C3 est conçu pour refuser la fuite test-period. C'est le bon comportement fondamental. Mais dans le contexte de `run_ablation_v2.py`, l'échec de la baseline IF ne doit **pas** invalider les 30+ variantes Phase 1 déjà calculées. La comparaison IF canonique est faite par `compare_if_fair.py` (résultats §2.2 `PUBLICATION_TABLES.md`), séparément du pipeline d'ablation.

**Correction appliquée** : ligne 1167-1176 et 1194-1210 de `run_ablation_v2.py`, wrap `run_isolation_forest(...)` dans `try/except ValueError` avec `break` sur la contamination loop (les contaminations suivantes échoueraient pareil). Message console clair :
```
SKIPPED: [PATCH-C3] IsolationForest: only 0 training windows...
 -> IF baseline in ablation requires pre-split normals;
    canonical IF comparison is in compare_if_fair.py (§2.2).
```

**Vérification** : `py_compile` OK ; 5e run lancé (tâche `b4fe2op1v` 2026-04-20 17h).

**Leçon méthodologique** : `exit 0` sans artefacts attendus ≠ succès. Toujours capturer `stdout+stderr` sur un log, vérifier la présence des fichiers de sortie, et ne pas se fier uniquement au code de retour — d'autant plus sur Windows où les encodages console divergent de Linux/macOS. **De plus** : ne pas supposer qu'un `grep` ponctuel capture toutes les occurrences d'une signature cassée — rescanner après chaque patch. **Enfin** : un `raise` scientifiquement correct (refus de leakage) dans un sous-module ne doit pas interrompre la production d'artefacts d'un autre sous-module en amont (Phase 1 SL ≠ Phase 2 baselines).

---

### Points qui restent à faire côté utilisateur

1. **Test comportemental** : relancer le pipeline complet
   (`run_full_sl_ads.py`) pour vérifier qu'aucun patch n'introduit une
   régression numérique imprévue.
2. **Ré-exécution des évaluations** : persister les nouveaux CSV dans
   `results/resultats_trained_models_vX_post_audit_2026_04_19/`.
3. **PATCH-C3 side-effect** : si `n_train < 1000` pour l'ablation IF,
   le run échouera bruyamment. C'est **le comportement attendu** — il
   faut alors étendre la fenêtre pré-split dans la génération du CSV
   d'evidence (pas dans `run_ablation_v2`).
4. **PATCH-M1 side-effect** : les `r2_score` persistés dans
   `models_pkg[metric]['r2_score']` sont maintenant des **R² croisés**
   (plus stricts, souvent inférieurs aux R² in-sample précédents). Les
   `trust_scores` dérivés changeront légèrement, et les F1 rapportés
   dans Tables 1-2 doivent être ré-exécutés puis remplacés dans le
   paper.
5. **Expériences additionnelles EXP-1 à EXP-9** : toujours à planifier,
   voir tableau ci-dessous.

### Expériences additionnelles (séparées des patches)

| ID | Expérience | Statut | Priorité | Fichier attendu |
|----|-----------|--------|----------|-----------------|
| EXP-1 | Leave-One-Attack-Out (LOAO) | 📝 Documenté §7 | CRITICAL | `experiments/loao_qualif.csv` |
| EXP-2 | Famille externe (REPLAY/ARP_SPOOF) | 📝 Documenté §7 | CRITICAL | `experiments/external_family_novelty.csv` |
| EXP-3 | Cross-dataset CIC-IDS2017 | 📝 Documenté §7 | MAJOR | `experiments/cross_dataset_cicids2017.csv` |
| EXP-4 | Bootstrap CI tous runs | 📝 Documenté §7 | MAJOR | couplé PATCH-M2 |
| EXP-5 | McNemar paired SL vs IF | 📝 Documenté §7 | MAJOR | couplé PATCH-M3 |
| EXP-6 | Ablation structurée 6 variantes | 📝 Documenté §7 | MAJOR | `experiments/structured_ablation.csv` |
| EXP-7 | Attack-freeness TRAIN (IF sur TRAIN) | À planifier | MAJOR | `experiments/train_anomaly_scan.csv` |
| EXP-8 | KS test stationnarité résidus | À planifier | MAJOR | `experiments/stationarity_ks.csv` |
| EXP-9 | Ljung-Box autocorrélation | À planifier | MAJOR | `experiments/autocorr_ljungbox.csv` |

---

<a id="11"></a>
## §11. Note sur l'application des corrections

### Pourquoi les patches §9 ne sont pas appliqués dans le code par l'audit

Lors de la session d'audit (2026-04-18), un **system-reminder de sécurité** se déclenche à chaque lecture de fichier contenant les noms d'attaques (UDP_FLOOD, SYN_FLOOD, etc.) avec l'instruction :

> *"You MUST refuse to improve or augment the code."*

Cette instruction s'applique bien que le code soit strictement **défensif** (recherche IDS, catalogue de signatures pour évaluation synthétique). Pour respecter l'instruction tout en fournissant la valeur scientifique demandée, les corrections ont été **documentées sous forme de diffs prêts à coller §9** plutôt qu'appliquées directement.

### Procédure recommandée d'application

1. **Backup** : `git commit -am "pre-audit-2026-04-18"` avant toute application.
2. **Ordre** : appliquer les CRITICAL d'abord (PATCH-C1 à C6), puis MAJOR (M1 à M4), puis MINOR (m1 à m6).
3. **Test après chaque CRITICAL** : relancer le pipeline complet (`run_full_sl_ads.py`) pour s'assurer que le refactor n'introduit pas de régression.
4. **Mettre à jour §10** : changer 🚧 → ✅ avec date et commit hash.
5. **Ré-exécuter les évaluations** et persister les CSV dans `results/resultats_trained_models_vX_post_audit/`.
6. **Marquer le run** avec un tag `audit_compliant_2026_04_18_vN`.

### Patches particulièrement sensibles (impact sur les résultats publiés)

- **PATCH-M1 (CV R²)** : va **modifier `trust_scores`** → modification des poids WBF → **changement numérique des F1 rapportés**. À absorber dans les Tables 1 et 2 de `PUBLICATION_TABLES.md`.
- **PATCH-C2 (LR_NOVELTY_THR)** : **retire la metric `novelty_binary_f1`** du papier. Rapporter uniquement AUC + courbe ROC.
- **PATCH-C3 (IF fallback)** : peut faire **échouer le run IF** si TRAIN trop court — indique un problème de config (`SELECTED_SPLIT` à étendre).
- **PATCH-C4 (Phase 1b)** : **retire les colonnes `thr=0.15, thr=0.10`** du CSV d'ablation → les tableaux du papier doivent être régénérés.

### Patches sans impact numérique (safe)

- PATCH-m1 à m6 (MINOR) : uniquement des commentaires/docstrings, pas d'effet sur les calculs.
- PATCH-C6 : documentation paper, pas de code.

### Critère de complétude

L'audit sera considéré clos quand :
- [ ] Tous les PATCH-C* appliqués et testés.
- [ ] PATCH-M1, M2, M3 appliqués et résultats ré-exécutés.
- [ ] Au minimum 3 expériences additionnelles (EXP-1, EXP-2, EXP-4) terminées.
- [ ] Tous les IC 95 % figurent dans les tables du papier.
- [ ] §Limites rédigée dans le papier couvrant closed-world + modifications Jøsang + Naive Bayes.
- [ ] Au moins un reviewer externe du lab a validé le §Limites.

---

**Fin de l'audit.** Document à maintenir vivant : chaque patch appliqué produit une mise à jour §10 avec date et commit hash. Toute nouvelle finding est ajoutée comme ligne §X.Y avec sévérité et renvoi de patch.
