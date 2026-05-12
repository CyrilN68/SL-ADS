# Revue scientifique et technique — `qualify_anomaly_sbn.py`

> Révision du : 2026-04-12  
> Contexte : préparation à l'évaluation par un jury scientifique  
> Fichier principal : `qualify_anomaly_sbn.py`  
> Dépendances directes : `sl_formulas_v2.py`, `config.py` (`QUALIFY_GROUP_SOURCES`, `SBN_COND_OPINIONS`), `paths.py`

---

## 1. Rôle du module dans la chaîne SL-ADS

`qualify_anomaly_sbn.py` est le **cinquième et dernier maillon** de la chaîne de détection, en aval de `evaluate_injection_v2.py` (qui fournit les colonnes `FINAL_SYSTEM_CBF_*`). Son rôle est de **qualifier le type d'anomalie** parmi K classes (UDP_FLOOD, SYN_FLOOD, ICMP_FLOOD, DNS_AMP, HTTP_FLOOD, SLOWLORIS, PORT_SCAN, DATA_EXFIL, NETWORK_OUTAGE, BGP_HIJACK, BOTNET_CC) pour chaque fenêtre temporelle jugée anormale (gate P(Anom) ≥ δ).

L'architecture est décomposée en 6 couches théoriques :

| Couche | Rôle | Outil formel |
|--------|------|--------------|
| L1 | Gate binaire P(Anom) ≥ δ | Seuil sur `FINAL_SYSTEM_CBF_proj_atk` |
| L2 | Agrégation intra-groupe | Moyenne géométrique des P^proj par métrique |
| L3 | Opinions conditionnelles expertes P(G=s\|type_k) | `SBN_COND_OPINIONS` (config.py) |
| L4 | Score SBN inter-groupes + bijection SL | Dot-product + Jøsang Déf. 3.9 |
| L5 | Prior temporel Markovien | WBF + matrice de transition Kill Chain |
| L6 | Uncertainty Maximisation | Jøsang Eq. 3.27 |

**Entrée** : CSV de `compute_opinions_v3.py` / `evaluate_injection_v2.py` (colonnes `{src}_proj_{safe,susp,atk}`, `FINAL_SYSTEM_CBF_proj_atk`).  
**Sortie** : `qualif_types_sbn.csv` — colonnes `gate_open`, `top1_type`, `top1_b`, `b_sbn_{type}`, `u_sbn`, `novelty_lr`, `b_sbn_raw_{type}`, `u_sbn_raw`.

---

## 2. Problèmes identifiés — classifiés par sévérité

### ═══ CRITIQUES (invalidation ou biais potentiel du pipeline) ═══

---

#### BUG-01 — Double-comptage de `reconst_fin_from_syn` dans deux groupes

**Localisation** : `config.py` — `QUALIFY_GROUP_SOURCES`  
**Description** : La métrique `reconst_fin_from_syn` apparaît à la fois dans le groupe `tcp_flags` ET dans le groupe `reconstruction` :

```python
'tcp_flags':     ['prophet_syn', 'prophet_fin', 'reconst_fin_from_syn'],
'reconstruction': ['reconst_bytes_from_packets', ..., 'reconst_fin_from_syn', ...],
```

Pour tout type d'attaque sensible à ces deux groupes (SLOWLORIS, SYN_FLOOD, DATA_EXFIL), la contribution de `reconst_fin_from_syn` est comptée **deux fois** : une fois via le score du groupe `tcp_flags` et une fois via `reconstruction`. Ceci gonfle artificiellement les scores pour ces types et invalide l'hypothèse d'indépendance conditionnelle Naive Bayes (L3).

**Correction appliquée** : Création d'un groupe `fin_ratio` **dédié** contenant uniquement `reconst_fin_from_syn`. Le groupe `tcp_flags` est réduit à `['prophet_syn', 'prophet_fin']` et `reconstruction` ne contient plus la métrique ratio. Cette architecture à 10 groupes élimine simultanément le double-comptage ET la dilution geomean qui annihilait le signal SLOWLORIS : avec `tcp_flags = [syn_ATK, fin_SAFE]`, `geomean(0.8, 0.05) ≈ 0.20`, soit un score ≈ 1/3 équivalent à une absence totale d'évidence. Le groupe `fin_ratio` seul porte désormais le signal `reconst_fin_from_syn` sans dilution.

---

#### BUG-02 — Renormalisation incorrecte des masses après la matrice de transition Markovienne

**Localisation** : `sbn_qualify_row()`, lignes 789–793

```python
b_vec = np.array([b_prev.get(t, 0.0) for t in type_names_tm])
b_trans_vec = transition_matrix.T @ b_vec
b_prev = {t: float(b_trans_vec[i]) for i, t in enumerate(type_names_tm)}
b_prev['Autre_Anomalie'] = 0.0
sum_b = sum(b_prev.values())
if sum_b > EPS:
    b_prev = {k: v / sum_b for k, v in b_prev.items()}  # ← BUG
```

**Problème** : L'opinion précédente `prev_opinion` est une opinion SL valide avec `sum(b) + u = 1`. Si `u_prev > 0` alors `sum(b_prev) = 1 - u_prev < 1`. Après l'application de T (qui préserve la somme des masses : `sum(T^T @ b) = sum(b)`), `sum(b_trans_vec) = 1 - u_prev`.

La renormalisation `/ sum_b` force `sum(b_new) = 1`, ce qui revient à poser **u = 0** (opinion dogmatique) avant le discounting. L'incertitude de l'opinion précédente est ainsi supprimée, puis partiellement réintroduite par `_discount_opinion` — ce double traitement est incorrect.

**Correction** : Ne pas renormaliser après T. Conserver `b_trans_vec` tel quel (la somme ≤ 1 est cohérente avec le u_prev) et passer `u_prev` inchangé à `_discount_opinion`. Supprimer le bloc de renormalisation.

---

#### PROB-03 — Le groupe `volume` de `NETWORK_OUTAGE` ne distingue pas la direction de l'anomalie

**Localisation** : `_DEFAULT_SBN_COND['NETWORK_OUTAGE']` et `config.py SBN_COND_OPINIONS`

```python
'volume': _strong_anom(),    # bytes+packets chutent → résidu négatif
```

**Problème** : La classification ternaire `{Safe, Susp, Anom}` est définie dans compute_opinions_v3.py pour des résidus élevés (Prophet_score > threshold). Elle ne distingue pas **direction** (surplus vs. déficit). Un résidu positif (flood) et un résidu négatif (outage) reçoivent tous deux `proj_atk` élevé. Ainsi, une panne réseau et un UDP Flood ont tous deux `volume = strong_anom`. La discrimination repose alors sur les groupes protocole (`protocol_tcp`, `protocol_udp`, `protocol_icmp`) et `connections` — ce qui fonctionne empiriquement mais doit être **explicitement justifié** comme mécanisme de discrimination plutôt que de volume.

La clé discriminante est que `NETWORK_OUTAGE` a `connections = strong_anom` AND tous les protocoles à `strong_safe`, ce qui n'est possible que si la chute de volume est accompagnée d'une chute de tous les protocoles. Cette chaîne de raisonnement doit être documentée.

**Recommandation** : Ajouter une métrique directionnelle (signe du résidu Prophet) comme colonne supplémentaire dans compute_opinions_v3.py, ou documenter explicitement que la discrimination OUTAGE vs FLOOD ne dépend pas du groupe `volume` mais des groupes protocole.

---

### ═══ MAJEURS (biais ou manque de rigueur scientifique) ═══

---

#### FORM-04 — `globals().get('CONFIG', {})` dans une fonction noyau

**Localisation** : `sbn_qualify_row()`, ligne 736

```python
_ev_scale = globals().get('CONFIG', {}).get('SBN_EVIDENCE_SCALE', 3.0)
```

**Problème** : L'usage de `globals()` dans une fonction appelée en boucle (une fois par fenêtre) est une anti-pattern qui rend le comportement non reproductible si `CONFIG` est modifié entre les appels. De plus, `sbn_qualify_row()` est une fonction pure (pas de side effects attendus) mais lit silencieusement l'état global. `SBN_EVIDENCE_SCALE` devrait être un paramètre explicite de la fonction.

**Correction** :
```python
def sbn_qualify_row(..., evidence_scale: float = 3.0) -> dict:
```
Et dans `run()`, passer `evidence_scale = CONFIG.get('SBN_EVIDENCE_SCALE', 3.0)`.

---

#### FORM-05 — `SBN_TEMPORAL_ENABLED` dans config.py non lu par `run()`

**Localisation** : `config.py` ligne 1367, `run()` dans `qualify_anomaly_sbn.py`

```python
# config.py
SBN_TEMPORAL_ENABLED = False
CONFIG["SBN_TEMPORAL_ENABLED"] = SBN_TEMPORAL_ENABLED
```

Mais dans `run()`, `apply_temporal` vient uniquement de l'argument CLI `--temporal` (défaut `False`). `CONFIG["SBN_TEMPORAL_ENABLED"]` n'est jamais lu. Il y a donc **deux sources de vérité distinctes** pour le même paramètre. Si un utilisateur configure `SBN_TEMPORAL_ENABLED = True` dans config.py, la qualification reste sans temporal sauf ajout explicite de `--temporal`.

**Correction** : Dans `run()`, lire la config en fallback :
```python
apply_temporal = apply_temporal or CONFIG.get('SBN_TEMPORAL_ENABLED', False)
```

---

#### FORM-06 — Commentaire erroné dans `_sl_bijection`

**Localisation** : `_sl_bijection()`, lignes 493–494

```python
# W = K : valeur canonique pour un cadre à K hypothèses (Jøsang §3.5.2).
# min(W, 2) garantit que le domaine binaire minimum est respecté.
K = len(likelihoods)
W = float(max(K, 2))
```

**Problème** : Le commentaire dit `min(W, 2)` (plafonne à 2) mais le code fait `max(K, 2)` (plancher à 2). Ces deux opérations sont opposées. `max(K, 2)` est correct (garantit W ≥ 2 pour éviter la division par zéro dans le cas K=1), mais le commentaire est faux et prête à confusion pour un lecteur.

**Correction** : Remplacer le commentaire par :
```python
# W = K : valeur canonique (Jøsang §3.5.2). max(K, 2) garantit W ≥ 2 (domaine minimal binaire).
```

---

#### FORM-07 — Matrice de transition : initialisation inutile + sémantique des overrides non explicitée

**Localisation** : `_build_transition_matrix()`, ligne 313

```python
T = np.ones((K, K))  # base : faible transition uniforme
for i in range(K):
    T[i, i] = SELF_PROB
    ...
```

**Problème 1** : `np.ones((K, K))` est immédiatement écrasé par la double boucle. L'initialisation à 1 est inutile et source de confusion. Utiliser `np.zeros`.

**Problème 2** : Les overrides `(PORT_SCAN → DATA_EXFIL, 0.20)` et `(PORT_SCAN → HTTP_FLOOD, 0.05)` s'ajoutent à la diagonale de 0.80 et aux off-diag de `0.20/(K-1) ≈ 0.025`. Après normalisation ligne, la diagonale `T[PORT_SCAN, PORT_SCAN]` n'est **plus 0.80** mais approximativement `0.80 / (0.80 + 0.20 + 0.05 + 7×0.025) ≈ 0.80/1.225 ≈ 0.653`. La persistance effective de PORT_SCAN est significativement plus faible que la valeur annoncée de 0.80. Le tableau des valeurs effectives post-normalisation doit être documenté pour chaque type.

**Correction** : Utiliser `np.zeros`, et ajouter en commentaire les valeurs effectives post-normalisation des types clés.

---

#### STAT-08 — `_sbn_group_score` n'est pas un rapport de vraisemblance bayésien au sens strict

**Localisation** : `_sbn_group_score()`, docstring

```
Ref : marginalisation bayésienne P(obs|k) = Σ_s P(obs=s) * P(obs=s|k)
(approx. : on utilise P^proj comme proxy de P(obs=s))
```

**Problème** : Le calcul `Score = Σ_s P^obs_s · c^{k|g}_s` est l'**espérance** de `c^{k|g}_s` sous la distribution observée, pas `P(obs|k)` au sens d'un modèle génératif. La formule bayésienne exacte nécessiterait un modèle de vraisemblance `P(P^obs = p | type_k)` sur l'espace continu des probabilités projetées, pas juste un produit scalaire discret.

L'approche est une **approximation par proxy discrétisé**, valide comme heuristique mais pas exactement une marginalisation bayésienne. Cela n'invalide pas le calcul mais la terminologie doit être corrigée.

**Correction** dans le docstring :
```
Score(k,g) = E_{s ~ P^obs_g}[c^{k|g}_s]  — espérance du signal d'appartenance à k
sous la distribution observée du groupe g.
Interprétation : mesure de compatibilité entre l'observation et la signature experte.
Note : ce n'est pas P(obs|k) au sens d'un modèle génératif — c'est une approximation
par proxy discret (Duda & Hart 1973, nearest-mean classifier).
```

---

#### STAT-09 — Hypothèse d'indépendance conditionnelle Naive Bayes non discutée pour les corrélations résiduelles

**Localisation** : `sbn_qualify_row()`, commentaire L3-L4, ligne 735

```python
# Hypothèse : indépendance conditionnelle des groupes étant donné le type
# (Naive Bayes — Rish 2001 IJCAI ; Mitchell 1997 Ch.6).
```

**Problème** : Cette hypothèse est reconnue mais non discutée. Or, des groupes comme `volume` et `protocol_udp` sont **structurellement corrélés** pour un UDP_FLOOD (explosion UDP implique explosion volume). La violation partielle de Naive Bayes est connue pour surestimer les scores des types avec de fortes co-occurrences (Domingos & Pazzani 1997).

L'architecture à 9 groupes sémantiques a précisément été conçue pour réduire ces corrélations. Cela doit être **argumenté explicitement** — les corrélations résiduelles attendues (ex. `volume` ↔ `connections` pour les floods volumétriques) doivent être listées et leur impact estimé ou borné.

---

#### STAT-10 — `_evidence_sum_scores` : le facteur `evidence_scale=3.0` est heuristique ; commentaire config.py obsolète

**Localisation** : `_evidence_sum_scores()` + `config.py`, commentaire SBN_EVIDENCE_SCALE

```python
e(k) = Σ_g max(0, score - 1/3) * evidence_scale
```

Le commentaire dans config.py dit :
```
# Avec W=3 et 9 groupes parfaitement alignés : u_min ≈ W/(9*2/3*SCALE + W)
```

**Problème** : Ce commentaire est **obsolète** — il était écrit pour l'ancien W=3 fixe. Avec W=K=11, la formule correcte est :
$$u_\text{min} \approx \frac{11}{9 \times \frac{2}{3} \times 3 + 11} = \frac{11}{18 + 11} \approx 0.38$$

Le commentaire dans config.py doit être mis à jour avec la formule W=K et la valeur effective `u_min ≈ 0.38`.

---

#### STAT-11 — `novelty_lr` : calibration sur signatures théoriques uniquement + incohérence de seuil

**Localisation** : `_lr_novelty()`, docstring + `SBN_LR_NOVELTY_THRESHOLD = 0.85` (config) vs `--novelty` défaut `0.65` (CLI)

**Problème 1** : La validation est faite sur des **signatures théoriques parfaites** (group_pp = SBN_COND[k]), pas sur des données réelles. Sur des données avec bruit, les attaques connues peuvent avoir une dominance relative plus faible (novelty_lr plus proche de 1). Il est possible que le seuil 0.85 génère des faux positifs de nouveauté sur des attaques connues bruitées.

**Recommandation** : Valider le seuil empiriquement sur le dataset RedeRio injecté et reporter la courbe ROC de novelty_lr.

**Problème 2** : La CLI `--novelty` a un défaut de `0.65` (ligne 1254) mais `SBN_LR_NOVELTY_THRESHOLD = 0.85` dans config.py. Dans `run()`, c'est la config qui prend la priorité sur la valeur passée en argument. Ces deux valeurs divergent silencieusement — le comportement effectif dépend de la source de lecture, ce qui n'est pas transparent.

**Correction** : Unifier à une seule valeur (0.85 recommandé, validé sur signatures) et documenter le seuil dans les deux endroits.

---

#### STAT-12 — Vérification contrainte SL par moyenne plutôt que par maximum d'erreur

**Localisation** : `run()`, lignes 1179–1181

```python
sl_check = (b_sum + df_anom['u_sbn']).mean()
print(f"  Contrainte SL : sum(b) + u = {sl_check:.5f}  (doit ~= 1.0)")
```

**Problème** : La vérification sur la **moyenne** peut masquer des fenêtres individuelles avec des violations. Si quelques fenêtres ont `sum(b) + u = 0.95` et d'autres `sum(b) + u = 1.05`, la moyenne peut afficher 1.0 malgré des violations locales réelles.

**Correction** :
```python
sl_errors = (b_sum + df_anom['u_sbn'] - 1.0).abs()
print(f"  Contrainte SL : max_err={sl_errors.max():.2e}  mean_err={sl_errors.mean():.2e}")
```

---

#### METH-13 — Base rates implicites uniformes dans `_sl_bijection` et `_apply_um` — hypothèse non justifiée

**Problème** : Dans `_sl_bijection`, la formule W=K implique un prior uniforme `a(k) = 1/K` sur les K types. Dans `_apply_um`, le prior uniforme `a_i = 1/K1` est explicitement posé. Mais l'hypothèse que chaque type d'attaque est a priori **équiprobable** est une hypothèse forte d'**ignorance a priori** qui doit être justifiée explicitement.

**Recommandation** : Ajouter dans le docstring de `_sl_bijection` et `_apply_um` la justification explicite : "Prior uniforme a(k)=1/K posé faute de distribution empirique des fréquences d'attaque sur le réseau RedeRio — correspond au prior non-informatif de Jøsang (§3.5.2)."

---

### ═══ MINEURS (forme, clarté, robustesse) ═══

---

#### TECH-14 — Type hints Python 3.10+ incompatibles avec Python 3.9

**Localisation** : Signatures de fonctions multiples

```python
def sbn_qualify_row(..., prev_opinion: dict | None = None) -> dict:
def _sl_bijection(...) -> tuple[dict, float]:
```

Ces annotations `X | Y` et `tuple[...]` minuscules nécessitent Python ≥ 3.10. Sur Python 3.9 (utilisé dans certains environnements HPC), cela lève une `TypeError` au chargement.

**Correction** : Ajouter en tête de fichier `from __future__ import annotations` (solution minimale) ou remplacer par `Optional[dict]` / `Tuple[dict, float]` de `typing`.

---

#### TECH-15 — Validation d'entrée manquante pour λ et temporal_weight

Aucune validation que `λ ∈ (0, 1]` et `temporal_weight ∈ [0, 1]`. Avec `λ > 1`, le signal temporel s'amplifie au lieu de décroître. Avec `temporal_weight > 1`, le WBF est hors domaine et produit des masses négatives.

**Correction** : Ajouter en début de `run()` :
```python
assert 0 < lambda_temporal <= 1.0, f"lambda_temporal hors domaine : {lambda_temporal}"
assert 0 <= temporal_weight <= 1.0, f"temporal_weight hors domaine : {temporal_weight}"
```

---

#### TECH-16 — `_compare_outputs` : `gate_open` potentiellement entier vs booléen

**Localisation** : `_compare_outputs()`, ligne 1209

Si le fichier CSV legacy a `gate_open = 0/1` (entier), le filtrage fonctionne mais peut lever un FutureWarning pandas. Ajouter `.astype(bool)` pour la robustesse.

---

#### FORM-17 — Référence Shannon 1948 inappropriée pour la moyenne géométrique

**Localisation** : `_compute_group_projected()`, docstring

```
(Théorie de l'information — Shannon 1948 ; Aczél & Daróczy 1975)
```

Shannon 1948 est la théorie de l'information/entropie, qui n'est **pas** la justification de la moyenne géométrique. La référence correcte est Aczél & Daróczy 1975 (caractérisation axiomatique des moyennes) ou, pour le contexte d'agrégation de distributions de probabilités, Genest & Zidek (1986, Statist. Sci.) sur le "logarithmic opinion pooling" (dont la moyenne géométrique normalisée est la forme canonique).

**Correction** : Remplacer Shannon 1948 par : `(Aczél & Daróczy 1975 ; Genest & Zidek 1986 Statist. Sci. — logarithmic opinion pooling)`

---

#### FORM-18 — `novelty_lr` décrit comme "proche de 0" pour les attaques connues mais vaut 0.47–0.62

**Localisation** : Commentaires dans `_lr_novelty()` et `_empty_result()`

```
# Attaque connue (un type domine clairement) : LR >> 1 → novelty_lr proche de 0
```

Mais la calibration dit `0.47–0.62`, qui n'est pas "proche de 0". La description est trompeuse.

**Correction** : Remplacer "proche de 0" par "modéré (0.4–0.6)" dans tous les commentaires associés.

---

#### FORM-19 — Elicitation experte non documentée formellement

**Localisation** : `_DEFAULT_SBN_COND` et `SBN_COND_OPINIONS` dans config.py

Les `SBN_COND_OPINIONS` sont des distributions expertes sur {Safe, Susp, Anom}. La méthode d'élicitation n'est pas documentée formellement : sur quelle base a-t-on choisi `_strong_anom() = {S:0.03, M:0.07, A:0.90}` ? La référence à Cooke (1991) dans `_sensitivity_analysis()` est appropriée mais doit être mentionnée au niveau de la **définition** des opinions, pas seulement dans l'analyse de sensibilité.

**Recommandation** : Ajouter un tableau formel dans le rapport avec pour chaque entrée (type, groupe) : valeur utilisée, source littéraire, niveau de confiance d'élicitation.

---

#### FORM-20 — L'analyse de sensibilité ne teste pas les perturbations croisées inter-types

**Localisation** : `_sensitivity_analysis()`, ligne 900

```python
for p_grp in list(sbn_cond[ref_type].keys()):  # perturbe uniquement SBN_COND[ref_type]
```

L'analyse perturbe les opinions du type `ref_type` et vérifie si ce type reste top-1. Mais elle ne vérifie pas si une perturbation des opinions **d'un autre type** peut "voler" la place de `ref_type`. Une analyse complète (Cooke 1991) testerait toutes les paires (type_perturbed, type_ref).

---

## 3. Hypothèses explicites du pipeline

| # | Hypothèse | Localisation | Justification fournie | Status |
|---|-----------|--------------|----------------------|--------|
| H1 | Agrégation intra-groupe par moyenne géométrique des P^proj | `_compute_group_projected()` | Aczél 1975 (incomplète) | Corriger référence (FORM-17) |
| H2 | Indépendance conditionnelle des 10 groupes (Naive Bayes) | `sbn_qualify_row()` L3-L4 | Rish 2001, Mitchell 1997 ; Domingos & Pazzani 1997 | Corrélations résiduelles documentées (STAT-09 ✅) |
| H3 | Base rates uniformes a(k) = 1/K pour tous les types | `_sl_bijection()`, `_apply_um()` | Jøsang §3.5.2 | Justifier prior non-informatif (METH-13) |
| H4 | Persistance des attaques T[k,k] = 0.80 | `_build_transition_matrix()` | Hutchins 2011 Kill Chain | Documenter valeurs effectives post-normalisation (FORM-07) |
| H5 | Classification ternaire sans direction de l'anomalie | `NETWORK_OUTAGE` SBN_COND | Observation RedeRio | Documenter mécanisme discrimination (PROB-03) |
| H6 | SBN_COND_OPINIONS stable à ±5% de perturbation | `_sensitivity_analysis()` | Analyse interne | Valider empiriquement (FORM-20) |
| H7 | Seuil novelty_lr = 0.85 discrimine connu vs inconnu | `SBN_LR_NOVELTY_THRESHOLD` | Calibration théorique uniquement | Valider sur RedeRio empiriquement (STAT-11) |

---

## 4. Outils formels utilisés et références

| Outil | Équation/Définition | Référence |
|-------|---------------------|-----------|
| Opinion multinomiale ω = (b, u, a) avec Σb + u = 1 | Déf. 2.1 | Jøsang (2016) §2 |
| Probabilité projetée P(x) = b(x) + a(x)·u | Eq. 3.23 | Jøsang (2016) §3.5 |
| Bijection SL évidence→opinion b(k) = e(k)/D, u = W/D | Déf. 3.9 | Jøsang (2016) §3.5.2 |
| Uncertainty Maximisation ü = min_i(P(xi)/a(xi)) | Eq. 3.27 | Jøsang (2016) §3.6 |
| WBF (Weighted Belief Fusion) | Eq. 12.22 | Jøsang (2016) §12.6 |
| Trust discounting b_d = λ·b, u_d = 1 − λ·(1−u) | Déf. 14.6 | Jøsang (2016) §14.3 |
| Déduction SBN multinomiale | §14.3–14.4 | Jøsang (2016) §14 |
| Rapport de vraisemblance comme statistique suffisante | — | Neyman & Pearson (1933) |
| Classe résiduelle / reject option | — | Chow (1970) IEEE TIT |
| Naive Bayes classifier | — | Rish (2001) IJCAI |
| Kill chain Markovien | — | Hutchins et al. (2011) Lockheed Martin |
| Signatures UDP/SYN/HTTP flood | Table III | Sharafaldin et al. (2018) CIC-IDS2017 |
| Signatures Slowloris, Port Scan | Feature taxonomy | Mirsky et al. (2018) Kitsune |
| ICMP discriminateur | Table V | Moustafa & Slay (2015) UNSW-NB15 |
| DNS amplification (×28–556) | Table 2 | Rossow (2014) NDSS |
| MITRE ATT&CK T1046/T1048/T1498/T1499 | — | MITRE 2023 |
| Élicitation experte bayésienne | — | Cooke (1991) Experts in Uncertainty |
| Logarithmic opinion pooling (moyenne géom.) | — | Genest & Zidek (1986) Statist. Sci. |

---

## 5. Description fonctionnelle complète — ce que fait le module

### 5.1 Initialisation

1. **Lecture configuration** : `SBN_COND_OPINIONS` (config.py ou fallback `_DEFAULT_SBN_COND`), `QUALIFY_GROUP_SOURCES`, `DECISION_THRESHOLD`.
2. **Cohérence GROUP_SOURCES ↔ SBN_COND** : Warning si un groupe dans SBN_COND n'est pas dans GROUP_SOURCES.
3. **Construction matrice de transition** T[K×K] Markovienne si `apply_temporal=True` (Hutchins 2011).
4. **Analyse de sensibilité optionnelle** : `--sensitivity` — perturbe chaque masse SBN_COND de ±5% et vérifie la stabilité du top-1 (Cooke 1991).

### 5.2 Qualification par fenêtre (L1–L6)

**L1 — Gate** : Si `P(Anom) = FINAL_SYSTEM_CBF_proj_atk < δ` → résultat vide (`gate_open=False`, `u=1`).

**L2 — Agrégation intra-groupe** : Pour chaque groupe g et chaque état s ∈ {Safe, Susp, Anom} :
$$P^g_s = \exp\left(\frac{1}{|g|}\sum_{m \in g} \log \max(P^m_s, \varepsilon)\right)$$
Normalisation ∑_s P^g_s = 1. Résultat : `group_pp[g] = {Safe:..., Susp:..., Anom:...}`.

**L3 — Score dot-product** : Pour chaque type k et groupe g :
$$\text{Score}(k, g) = \sum_{s \in \{S,M,A\}} P^g_s \cdot c^{k|g}_s$$

**L4 — Bijection SL** :
$$e(k) = \sum_g \max\!\left(0,\; \text{Score}(k,g) - \tfrac{1}{3}\right) \times \text{SCALE}$$
$$b(k) = \frac{e(k)}{D},\quad u = \frac{W}{D},\quad D = \sum_k e(k) + W,\quad W = K$$

**L5 — Prior temporel** (si `apply_temporal=True`) :
- Propagation Markovienne : $b_\text{trans}[j] = \sum_i T[i,j] \cdot b_\text{prev}[i]$ (Chapman-Kolmogorov ord. 1)
- Décroissance : $\text{decay} = \lambda^{\Delta t}$, puis $b_d = \text{decay} \cdot b$, $u_d = 1 - \text{decay}(1-u)$
- WBF : $b_f(k) = (1-w_T) \cdot b_\text{curr}(k) + w_T \cdot b_d(k)$

**L6 — Uncertainty Maximisation** (si `apply_um=True`) :
$$\ddot{u} = K \cdot \min_k P(x_k), \quad \ddot{b}(k) = P(x_k) - \tfrac{\ddot{u}}{K}$$

**Métriques de sortie** :
- `top1_type` : argmax b_sbn
- `top1_b` : croyance maximale
- `u_sbn` : incertitude post-UM (proxy de familiarité avec le type)
- `novelty_lr = 1 / (\max_k e(k) / \overline{e})` : signal de nouveauté basé sur la concentration de l'évidence

### 5.3 Post-traitement et évaluation

- **Stats résumées** : distribution des top-1, u_sbn moyen, signal de nouveauté, vérification contrainte SL Σb + u ≈ 1.
- **Évaluation quantitative** (si colonne `injected_type` disponible) : Précision/Rappel/F1 par type + matrice de confusion top-5 types.
- **Comparaison SBN vs heuristique LR** (si `--compare`) : accord top-1 par fusion sur timestamp.

---

## 6. Paramètres clés et leurs valeurs par défaut

| Paramètre | Défaut | Source | Impact |
|-----------|--------|--------|--------|
| `δ` (gate threshold) | `DECISION_THRESHOLD` (config) | `evaluate_injection_v2.py` | Contrôle le taux de qualification |
| `SCALE` (`SBN_EVIDENCE_SCALE`) | 3.0 | config.py | Amplitude évidence → u_min ≈ 0.35 pour K=11, 10 groupes |
| `W = K` | dynamique (11) | Jøsang §3.5.2 | u=1 à évidence nulle |
| `λ` (temporal decay) | 0.80 | config.py | Demi-vie ≈ 4 fenêtres (0.80^4 ≈ 0.41) |
| `w_T` (temporal weight) | 0.30 | config.py | 30% du prior précédent dans WBF |
| novelty_threshold | 0.85 | config.py + CLI | Unifié (STAT-11 corrigé) |

---

## 7. Points clés pour la rédaction d'un rapport technique

1. **Décrire la qualification comme un problème de classification bayésienne à K classes** avec prior non-informatif (uniforme), vraisemblance approchée par dot-product expert-élicité, et classe résiduelle portée par u (Jøsang §14.4 — classe résiduelle sans évidence directe).

2. **Justifier l'architecture à 10 groupes sémantiques** comme réduction des corrélations inter-métriques (Naive Bayes approximativement valide) : volume, protocole TCP/UDP/ICMP (3 groupes séparés), tcp_flags (prophet_syn + prophet_fin uniquement), **fin_ratio** (reconst_fin_from_syn — groupe dédié), connexions, entropie, taille paquet, reconstruction — chaque groupe capture une dimension orthogonale du trafic. La création du groupe `fin_ratio` dédié élimine simultanément le double-comptage (BUG-01) et la dilution geomean qui annihilait la discrimination SLOWLORIS (geomean[syn_ATK=0.8, fin_SAFE=0.05] ≈ 0.20 → score ≈ 1/3 → évidence nulle).

3. **Documenter formellement la méthode d'élicitation** des SBN_COND_OPINIONS : sources littéraires (CIC-IDS2017, Kitsune, UNSW-NB15, MITRE ATT&CK), conventions d'élicitation (_strong_anom, _mod_anom, etc.), et résultats de l'analyse de sensibilité Cooke (1991).

4. **Signaler la distinction OUTAGE vs FLOOD** comme un choix de modélisation délibéré : pas de gate séparée (contrairement à qualify_anomaly.py), discrimination native SBN par signature protocole multi-groupe.

5. **Documenter les limitations** :
   - Naive Bayes avec corrélations résiduelles volume/protocol_udp, tcp_flags/fin_ratio (atténuées par la séparation en 10 groupes orthogonaux)
   - BUG-01 double-comptage reconst_fin_from_syn : ✅ corrigé par groupe `fin_ratio` dédié
   - BUG-02 renormalisation incorrecte après matrice de transition : ✅ corrigé
   - Calibration novelty_lr sur données théoriques seulement (à valider empiriquement)
   - Base rates uniformes (ignorance a priori)

6. **Présenter la L6 (Uncertainty Maximisation)** comme un signal de nouveauté (anomalie inconnue) plutôt qu'une amélioration de la classification des types connus — l'UM est neutre sur P(xi), elle redistribue u sans changer les probabilités projetées.

7. **Présenter les métriques de comparaison** SBN vs heuristique LR : avantages — confiance variable par groupe (u encodé dans SBN_COND), stabilité numérique (dot-product vs LR brut), discrimination native OUTAGE, prior temporel Markovien.

---

## 8. Résumé des corrections prioritaires avant jury

| ID | Priorité | Action | Fichier | Status |
|----|----------|--------|---------|--------|
| BUG-01 | 🔴 P1 | Groupe `fin_ratio` dédié : `reconst_fin_from_syn` retiré de `tcp_flags` ET `reconstruction` → groupe propre ; SLOWLORIS `tcp_flags=neutral` | `config.py:875-880` + `sbn.py:_DEFAULT_SBN_COND` | ✅ Corrigé |
| BUG-02 | 🔴 P1 | Suppression renormalisation post-T (bug amplification confiance ×3) | `sbn.py:828-833` | ✅ Corrigé |
| FORM-04 | 🟠 P2 | `evidence_scale` comme paramètre explicite de `sbn_qualify_row` | `sbn.py:662` | ✅ Corrigé |
| FORM-05 | 🟠 P2 | Lecture `SBN_TEMPORAL_ENABLED` depuis CONFIG dans `run()` | `sbn.py:1107-1108` | ✅ Corrigé |
| FORM-06 | 🟠 P2 | Commentaire `min(W,2)` → `max(K,2)` dans `_sl_bijection` | `sbn.py:~497` | ✅ Corrigé |
| FORM-07 | 🟠 P2 | `np.ones` → `np.zeros` + note sur valeurs T effectives post-normalisation | `sbn.py:313` | ✅ Corrigé |
| STAT-08 | 🟡 P3 | Docstring `_sbn_group_score` : espérance, pas vraisemblance bayésienne | `sbn.py:422` | ✅ Corrigé |
| STAT-09 | 🟡 P3 | Discussion corrélations résiduelles Naive Bayes avec sources | `sbn.py:763-771` | ✅ Corrigé |
| STAT-10 | 🟠 P2 | Commentaire `u_min` avec formule générale W=K | `config.py:1377-1384` | ✅ Corrigé |
| STAT-11 | 🟠 P2 | Unification novelty_threshold → 0.85 partout (CLI + config) | `sbn.py:~1295` | ✅ Corrigé |
| STAT-12 | 🟡 P3 | Vérification SL sur max_error fenêtre par fenêtre | `sbn.py:~1220` | ✅ Corrigé |
| METH-13 | 🟡 P3 | Justification prior uniforme a(k)=1/K dans docstrings | `sbn.py:497,515` | ✅ Corrigé |
| FORM-17 | 🟡 P3 | Shannon 1948 → Genest & Zidek 1986 (logarithmic opinion pooling) | `sbn.py:370` | ✅ Corrigé |
| FORM-18 | 🟡 P3 | "proche de 0" → "modéré (0.47-0.62)" pour novelty_lr | `sbn.py:626,694,859` | ✅ Corrigé |
| TECH-14 | 🟡 P3 | `from __future__ import annotations` (Python 3.9+) | `sbn.py:48` | ✅ Corrigé |
| TECH-15 | 🟡 P3 | Validation λ ∈ (0,1] et w_T ∈ [0,1] en début de `run()` | `sbn.py:1098-1102` | ✅ Corrigé |
| PROB-03 | 🔴 P1 | Documentation limitation directionnelle + mécanisme discrimination OUTAGE | `sbn.py+config.py` | ✅ Documenté |

---

## 9. Points ouverts post-corrections (recherche future)

| Point | Description | Impact |
|-------|-------------|--------|
| Directionnalité du résidu | Ajouter `{src}_residual_sign ∈ {+1,-1}` dans compute_opinions_v3.py pour distinguer surplus (flood) vs déficit (outage) dans TOUS les groupes | Amélioration discrimination OUTAGE vs FLOOD pour tous types |
| Calibration empirique novelty_lr | Valider le seuil 0.85 sur données RedeRio injectées (courbe ROC) | Le seuil actuel est calibré sur signatures théoriques uniquement |
| Sensibilité croisée | Étendre `_sensitivity_analysis` aux perturbations d'un type vers un autre | Couverture Cooke (1991) complète |
| Prior non-uniforme | Si des stats de fréquence d'attaque deviennent disponibles, remplacer a(k)=1/K | Amélioration des base rates |
