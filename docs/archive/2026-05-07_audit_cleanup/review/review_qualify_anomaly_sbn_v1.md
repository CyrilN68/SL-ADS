# Revue scientifique et technique — `qualify_anomaly_sbn.py`

> Révision du : 2026-04-11  
> Contexte : préparation à l'évaluation par un jury scientifique  
> Fichier principal : `qualify_anomaly_sbn.py`  
> Dépendances directes : `config.py`, `paths.py`, `compute_opinions_v3.py` (fournit le CSV d'entrée)

---

## 1. Rôle du module dans la chaîne SL-ADS

`qualify_anomaly_sbn.py` est le **cinquième et dernier maillon** de la chaîne de
détection. Il prend le CSV de scores d'opinion produit par `compute_opinions_v3.py`
(colonnes `FINAL_SYSTEM_CBF_proj_atk`, `P_{metric}_{col}`, …) et produit, pour
chaque fenêtre temporelle franchissant le seuil de décision `δ`, une **qualification
de type d'attaque** dans un cadre Subjectif Bayésien (SBN).

Le pipeline interne au module suit une architecture à 6 couches explicites :

| Couche | Nom | Opération |
|---|---|---|
| L1 | Gate d'activation | `P(Anom) ≥ δ` sur la colonne `_DET_COL` |
| L2 | Opinions de groupes | Geomean des probabilités projetées par groupe sémantique |
| L3 | Score SBN par type | Dot-product ou log-LR entre obs. et opinions conditionnelles expertes |
| L4 | Bijection SL | Evidence → opinion (Jøsang Déf. 3.9) avec W=K dynamique |
| L5 | Prior temporel (opt.) | WBF avec opinion t-1 escomptée (décroissance λ^Δt) |
| L6 | Uncertainty Maximisation (opt.) | Amplifie u pour les anomalies inconnues (Jøsang Eq. 3.27) |

Sorties CSV : `qualif_types_sbn.csv` — une ligne par fenêtre, avec pour chaque
type d'attaque : belief mass `b_sbn_{type}`, uncertainty `u_sbn`, métriques de
nouveauté `novelty_lr`, top-1 type et indicateur `gate_open`.

---

## 2. Fonctions et outils employés

| Fonctionnalité | Implémentation | Référence |
|---|---|---|
| Opinions conditionnelles expertes | `_DEFAULT_SBN_COND` / `CONFIG['SBN_COND_OPINIONS']` | Sharafaldin 2018 ; Mirsky 2018 ; Moustafa 2015 ; MITRE ATT&CK |
| Geomean par groupe | `_compute_group_projected()` | Aczél & Daróczy 1975 |
| Score dot-product | `_sbn_group_score()` | Jøsang 2016 §14.3 |
| Log-LR (mode opt.) | `_evidence_log_lr_scores()` | Good 1950 ; Rish 2001 |
| Bijection SL evidence→opinion | `_sl_bijection()` | Jøsang 2016 Déf. 3.9 |
| Uncertainty Maximisation | `_apply_um()` | Jøsang 2016 Eq. 3.27 |
| Prior temporel (WBF + décroissance) | `_wbf_two()` + `_discount_opinion()` | Jøsang 2016 Eq. 12.22 ; Déf. 14.6 |
| Matrice de transition Markovienne | `_build_transition_matrix()` | Hutchins et al. 2011 (Kill Chain) |
| Métrique de nouveauté | `_lr_novelty()` | Shafer 1976 (DempsterShafer) |
| Gate panne réseau (OUTAGE) | `_outage_gate_check()` | qualify_anomaly.py original |
| Lecture / écriture CSV | `pandas.read_csv / to_csv` | — |

---

## 3. Hypothèses posées (explicites et implicites)

### 3.1 Hypothèses sur les données d'entrée

- **H1 – Colonne de détection disponible** : le CSV d'entrée doit contenir la
  colonne `_DET_COL` (ex. `FINAL_SYSTEM_CBF_proj_atk`) produite par
  `compute_opinions_v3.py`. Son absence provoque un `ValueError` (vérification
  en ligne 1034).

- **H2 – Colonnes de probabilités projetées par métrique disponibles** : les
  colonnes `{metric}_proj_safe`, `{metric}_proj_susp`, `{metric}_proj_atk` sont
  supposées présentes pour chaque source listée dans `GROUP_SOURCES`. Leur absence
  est signalée par un `[WARN]` mais n'arrête pas le traitement — la fenêtre est
  qualifiée avec les groupes disponibles uniquement.

- **H3 – Stationnarité inter-fenêtres** : la qualification suppose que le type
  d'attaque est constant sur une fenêtre de `WINDOW_SIZE` pas. Une transition
  en cours de fenêtre (attaque changeant de nature) produit un score mixte.

### 3.2 Hypothèses du réseau bayésien subjectif

- **H4 – Indépendance conditionnelle des groupes (Naive Bayes)** : le score
  de type est calculé comme une somme (log-espace ou dot-product) sur les groupes
  sémantiques, ce qui revient à supposer que les groupes sont conditionnellement
  indépendants étant donné le type d'attaque. Or certains groupes sont corrélés
  (ex. `volume` et `connections` augmentent conjointement lors d'un DDoS). Cette
  hypothèse est documentée pour le mode `log_lr` (Rish 2001) mais non pour le
  mode `dot_product`.

- **H5 – Opinions conditionnelles expertisées** : l'intégralité de la matrice
  `SBN_COND_OPINIONS` est déterminée a priori par expertise humaine (avec
  référencement littérature). Aucun apprentissage sur des données étiquetées n'est
  effectué. Les valeurs numériques des niveaux `_strong_anom`, `_mod_safe`, etc.
  sont des conventions partagées entre tous les types (cohérence interne) mais
  non calibrées sur le dataset RedeRio.

- **H6 – Prior uniforme sur les types** : la bijection SL utilise W=K
  (Jøsang §3.5.2), ce qui implique un prior uniforme P(H_k) = 1/K sur les K
  types. Ce prior est raisonnable en l'absence d'information a priori sur la
  fréquence relative des attaques, mais devrait être discuté (ex. PORT_SCAN plus
  fréquent que BGP_HIJACK dans RedeRio).

- **H7 – Geomean comme agrégation intra-groupe** : la moyenne géométrique des
  probabilités projetées par état et par groupe (L2) est justifiée par son
  invariance d'ordre de magnitude (Aczél & Daróczy 1975). Mais la geomean
  d'une collection de vecteurs simplex (un par métrique) puis la renormalisation
  ne produisent pas un vecteur qui soit l'espérance géométrique sur le simplex —
  c'est une approximation dont les propriétés asymptotiques ne sont pas discutées.

- **H8 – Séparation temporelle : événements = fenêtres indépendantes + Markov-1**
  : hors prior temporel, chaque fenêtre est qualifiée indépendamment des
  précédentes. Avec prior temporel (L5), la dépendance est limitée à t-1
  (Markov-1). Les attaques longues sont supposées stationner dans le même type.

### 3.3 Hypothèses sur le prior temporel

- **H9 – Décroissance exponentielle homogène** : le poids du prior temporel
  décroît comme `λ^Δt` avec `λ=0.80` par défaut (soit demi-vie ≈ 3 fenêtres =
  15 minutes pour RedeRio). Ce paramètre n'a pas été calibré sur données ; il
  est fixé empiriquement.

- **H10 – WBF approprié pour sources dépendantes** : le WBF (Eq. 12.22) est
  défini dans Jøsang pour deux sources indépendantes. L'utilisation ici pour
  fusionner l'opinion courante et l'opinion t-1 (même phénomène, sources
  dépendantes temporellement) est une extension non standard, justifiée dans le
  code par la non-indépendance mais sans référence formelle alternative.

- **H11 – Poids temporel effectif = `w_temporal × λ^Δt`** : la pondération
  du prior diminue doublement (via decay dans `_discount_opinion` ET via
  `w_temp_eff = temporal_weight * decay`). Ce double escompte n'est pas
  documenté comme un choix délibéré.

### 3.4 Hypothèses sur la gate OUTAGE

- **H12 – Signes physiques discriminants pour la panne réseau** : la gate OUTAGE
  suppose que lors d'une panne, bytes ET packets sont anormaux (P_atk > 0.50)
  MAIS que les protocoles individuels (ICMP, UDP, SYN, TCP) restent normaux
  (P_safe > 0.85). Ces seuils sont repris de `qualify_anomaly.py` sans
  recalibration sur le dataset RedeRio avec l'architecture SBN.

---

## 4. Sorties produites

| Colonne CSV | Signification |
|---|---|
| `timestamp` | Horodatage de la fenêtre (depuis le CSV d'entrée) |
| `gate_open` | Booléen : fenêtre classifiée comme anormale (P_Anom ≥ δ) |
| `top1_type` | Type d'attaque dominant (argmax des belief masses) |
| `top1_b` | Belief mass du type dominant ∈ [0, 1] |
| `b_sbn_{type}` | Belief mass pour chaque type après L5+L6 |
| `u_sbn` | Incertitude SBN finale ∈ [0, 1] |
| `novelty_score` | Identique à `novelty_lr` (alias redondant — cf. B5) |
| `novelty_lr` | 1 / (max_L / mean_L) : mesure de nouveauté par dominance LR |
| `b_sbn_raw_{type}` | Belief mass avant prior temporel et UM (diagnostics) |
| `u_sbn_raw` | Incertitude avant L5+L6 |
| `b_sbn_Autre_Anomalie` | Toujours 0.0 (classe résiduelle portée par u) |

---

## 5. Paramètres de configuration impliqués

| Paramètre CONFIG | Rôle | Valeur typique |
|---|---|---|
| `SBN_COND_OPINIONS` | Matrice d'opinions conditionnelles P(G=s\|type_k) | `_DEFAULT_SBN_COND` |
| `QUALIFY_GROUP_SOURCES` | Groupes sémantiques : {nom → [métriques]} | `volume`, `protocol_tcp`, … |
| `SBN_EVIDENCE_SCALE` | Facteur multiplicatif de l'évidence (3.0 = K=3) | 3.0 |
| `SBN_SCORING_MODE` | Mode de scoring : `"dot_product"` ou `"log_lr"` | `"dot_product"` |
| `SBN_OUTAGE_GATE_ENABLED` | Active/désactive la gate OUTAGE séparée | `True` |
| `SBN_OUTAGE_PARAMS` | Seuils `atk_thr`, `safe_thr` pour la gate OUTAGE | `{0.50, 0.85}` |
| `SBN_LR_NOVELTY_THRESHOLD` | Seuil `novelty_lr` pour signal de nouveauté | 0.85 |
| `QUALIFY_VERBOSE` | Activation du mode verbeux | `False` |

CLI uniquement :

| Argument CLI | Paramètre | Défaut |
|---|---|---|
| `--threshold` | Seuil gate δ | `_THRESHOLD` depuis config |
| `--W` | Paramètre bijection SL (non utilisé — cf. B2) | 3.0 |
| `--temporal` | Active L5 prior temporel | `False` |
| `--no_um` | Désactive L6 UM | `False` |
| `--lambda_t` | Facteur de décroissance λ | 0.80 |
| `--w_temp` | Poids maximal du prior temporel | 0.30 |
| `--novelty` | Seuil u_sbn pour nouveauté (non appliqué en sortie — cf. F2) | 0.65 |
| `--compare` | CSV qualify_anomaly.py pour comparaison | None |

---

## 6. Problèmes identifiés

### 6.1 Bugs / erreurs silencieuses

#### B1 — `'CONFIG' in dir()` toujours False dans les fonctions (lignes 1007, 1012, 1100)

**Problème fondamental :** À l'intérieur des fonctions `run()` (lignes 1007, 1012)
et dans la boucle de stats (ligne 1100), le code utilise :
```python
sbn_cond = CONFIG.get('SBN_COND_OPINIONS', _DEFAULT_SBN_COND) \
           if 'CONFIG' in dir() else _DEFAULT_SBN_COND
```
`dir()` sans argument, appelé depuis l'intérieur d'une fonction, retourne les noms
du scope **local** uniquement — pas les globaux. Or `CONFIG` est une variable de
module. Le résultat : `'CONFIG' in dir()` est **toujours `False`** à l'intérieur
de `run()`, donc `sbn_cond` prend toujours `_DEFAULT_SBN_COND` même quand
`config.py` est chargé et contient `SBN_COND_OPINIONS`. La configuration utilisateur
est silencieusement ignorée.

Le code l'a **reconnu lui-même** à la ligne 873–874 pour `sbn_qualify_row()` (qui
utilise correctement `globals().get('CONFIG', {})`), mais la correction n'a pas été
appliquée à `run()`.

**Correction :**
```python
# Au lieu de :
sbn_cond = CONFIG.get('SBN_COND_OPINIONS', _DEFAULT_SBN_COND) \
           if 'CONFIG' in dir() else _DEFAULT_SBN_COND
# Utiliser :
try:
    sbn_cond = CONFIG.get('SBN_COND_OPINIONS', _DEFAULT_SBN_COND) or _DEFAULT_SBN_COND
except NameError:
    sbn_cond = _DEFAULT_SBN_COND
# Ou plus simplement (CONFIG est garanti accessible si l'import a réussi) :
_g = globals()
sbn_cond = _g['CONFIG'].get('SBN_COND_OPINIONS', _DEFAULT_SBN_COND) \
           if 'CONFIG' in _g else _DEFAULT_SBN_COND
```

---

#### B2 — Paramètre `W` passé mais jamais utilisé dans `_sl_bijection`

**Problème :** L'argument `--W` du CLI et le paramètre `W` de `sbn_qualify_row()`
(ligne 785) donnent l'impression que l'utilisateur peut contrôler la constante de
bijection SL. En réalité, `_sl_bijection()` (ligne 586) calcule W dynamiquement :
```python
W = float(K) if K > 0 else 1.0  # W = len(likelihoods) — toujours
```
Le paramètre `W` de `sbn_qualify_row()` et l'argument `--W` du CLI sont donc
entièrement **dead code**. L'aide CLI (`--W Constante bijection SL W (domaine
ternaire = 3.0)`) est trompeuse.

**Correction :** Supprimer le paramètre `W` de `sbn_qualify_row()` et `run()`, et
supprimer `--W` du CLI, ou bien utiliser W comme valeur de base dans `_sl_bijection`
pour permettre l'ablation (ex. `W = w_param if w_param > 0 else float(K)`).

---

#### B3 — Matrice de transition Markovienne définie mais jamais utilisée

**Problème :** `_build_transition_matrix()` (lignes 297–343) est documentée avec
motivation kill chain et construite proprement, mais elle n'est appelée **nulle part**
dans le code. La validation que les lignes somment à 1.0 (normalisées) et l'intégration
dans L5 (prior temporel structurant les transitions) ne sont pas implémentées.

En l'état actuel, L5 utilise uniquement l'opinion brute t-1 (persistance non
structurée), pas les probabilités de transition PORT_SCAN → DATA_EXFIL etc.

**Correction :** Soit supprimer `_build_transition_matrix()` et retirer la mention
de la matrice de transition dans la docstring du module (ligne 24), soit l'intégrer
en multipliant `b_prev[k]` par `T[top1_prev][k]` avant l'escompte WBF.

---

#### B4 — Double escompte du prior temporel non documenté

**Problème :** La décroissance temporelle est appliquée **deux fois** successivement :

1. `_discount_opinion(b_prev, u_prev, decay)` avec `decay = λ^Δt` → produit `b_prev_d`
2. `w_temp_eff = temporal_weight * decay` → le poids WBF est lui aussi multiplié par `decay`

Le prior temporel est donc escompté en pratique comme `decay²` (en termes d'influence
nette). Cela n'est jamais mentionné et crée une ambiguïté sur la sémantique de
`temporal_weight` : est-il le poids à t-1, à t-2, ou à distance infinie ?

**Correction :** Choisir un seul mécanisme d'escompte et documenter l'effet net.

---

#### B5 — `novelty_score` et `novelty_lr` identiques en sortie (ligne 952–953)

**Problème :**
```python
result['novelty_score'] = nov_lr  # = 1/LR_dominance
result['novelty_lr']    = nov_lr  # 1/LR_dominance (plus sensible)
```
Ces deux colonnes portent **exactement la même valeur**. La docstring de `run()`
dit que `novelty_threshold` est un "seuil u_sbn", ce qui implique que
`novelty_score` était censé être `u_sbn` à l'origine. La refactorisation vers
`nov_lr` n'a pas mis à jour le deuxième champ ni la documentation.

**Correction :** Soit supprimer `novelty_score` (alias redondant), soit lui assigner
`u_sbn` final (sa signification originale). Mettre à jour la docstring.

---

#### B6 — Gate OUTAGE contredit l'architecture annoncée

**Problème :** La docstring du module (ligne 25) affirme explicitement :
> "NETWORK_OUTAGE via signature SBN native **(sans gate OUTAGE séparé)**"

Mais l'implémentation (lignes 854–870) contient exactement une gate séparée qui
**bypasse** la classification SBN et affecte directement `b_sbn_NETWORK_OUTAGE`.
C'est une contradiction directe entre la spécification annoncée et l'implémentation.

De plus, la belief mass assignée est `min(P_bytes_proj_atk, P_packets_proj_atk)`,
ce qui n'est pas une belief mass SBN au sens de Jøsang — c'est une probabilité
scalaire directement copiée. L'opinion résultante ne satisfait pas la décomposition
SL complète (pas de calcul des b pour les autres types, pas d'a_k).

**Correction (option A — cohérence architecturale) :** Supprimer la gate OUTAGE
séparée. Encoder NETWORK_OUTAGE uniquement via ses `SBN_COND_OPINIONS` (chute de
volume → contribution naturelle dans le score SBN).

**Correction (option B — pragmatique) :** Supprimer la mention "sans gate OUTAGE
séparé" de la docstring et documenter clairement le bypass comme un choix délibéré
de robustesse.

---

#### B7 — Incohérence docstring overrides matrice de transition (ligne 303 vs code 330)

**Problème :** Le commentaire dans `_build_transition_matrix` dit :
```
# NETWORK_OUTAGE → NETWORK_OUTAGE : 0.90
```
Mais la liste des overrides (ligne 330) contient :
```python
('NETWORK_OUTAGE', 'NETWORK_OUTAGE', 0.92),
```
Valeur effective : **0.92** ≠ **0.90**. Incohérence commentaire/code.

---

#### B8 — Gate L1 : fallback colonne silencieux (ligne 834)

**Problème :**
```python
p_atk = row.get(_DET_COL, row.get('FINAL_SYSTEM_CBF_proj_atk', 0.0))
```
Si `_DET_COL` n'est pas dans la ligne, on tente `FINAL_SYSTEM_CBF_proj_atk`.
Si cette colonne est elle-même absente, `p_atk = 0.0` → la gate ne s'ouvre jamais
→ le module retourne des résultats vides sans avertissement. Ce cas est masqué
car la vérification en ligne 1034 ne vérifie que `'FINAL_SYSTEM_CBF_proj_atk'`
et pas `_DET_COL`.

**Correction :** Ajouter `_DET_COL` à la vérification de colonnes obligatoires en
ligne 1034, ou lever une exception explicite si ni `_DET_COL` ni le fallback ne sont
présents dans la ligne.

---

### 6.2 Problèmes scientifiques et manques de justification

#### S1 — Score dot-product : interprétation probabiliste formelle absente

**Problème :** La fonction `_sbn_group_score()` calcule :
```
Score(k,g) = Σ_s P^obs_{g,s} · c^{k|g}_s
```
La docstring dit "C'est le produit scalaire". Mais ce produit scalaire de deux
distributions de probabilité est une **espérance** de `c^{k|g}_s` sous la
distribution observée `P^obs_{g,s}`, soit :
```
Score(k,g) = E_{s ~ P^obs_g}[c^{k|g}_s]
```
Ce n'est pas `P(obs_g | H_k)` mais plutôt une approximation par proxy. La
référence citée (marginalisation bayésienne) est incorrecte — ce n'est pas
`P(obs|k) = Σ_s P(obs=s) * P(obs=s|k)` car `P^obs_{g,s}` et `c^{k|g}_s`
sont sur le même domaine (états s), pas sur des espaces différents.

**Pour le rapport :** Reformuler comme "espérance du score d'appartenance à type k
sous la distribution observée du groupe g" et clarifier que c'est une approximation
non-paramétrique de la vraisemblance.

---

#### S2 — Hypothèse d'indépendance conditionnelle non discutée pour `dot_product`

**Problème :** Le mode `log_lr` documente explicitement l'hypothèse Naive Bayes
(Rish 2001). Le mode `dot_product` (mode par défaut) effectue aussi une agrégation
additive sur les groupes mais sans documenter cette hypothèse. Or les groupes
`volume` et `connections` sont fortement corrélés (Pearson r > 0.7 attendu en DDoS),
ce qui viole l'indépendance conditionnelle.

**Pour le rapport :** Quantifier la corrélation inter-groupes sur le dataset RedeRio
(ou citer une borne connue du biais Naive Bayes sous corrélation — Zhang 2004
AAAI). Soit accepter le biais comme compromis de robustesse, soit passer en mode
`log_lr` par défaut.

---

#### S3 — Valeurs numériques des niveaux d'opinion non calibrées

**Problème :** Les fonctions `_strong_anom()` → `{S:0.03, M:0.07, A:0.90}`,
`_mod_anom()` → `{S:0.08, M:0.22, A:0.70}`, etc. sont des **conventions fixes**
non calibrées sur données. Leur seule justification est l'écart à l'uniformité
(ud ≈ 0.10 pour fort, ≈ 0.20 pour modéré). Or ces valeurs déterminent directement
la séparabilité inter-types et le niveau absolu d'évidence en sortie de `_sl_bijection`.

Questions sans réponse dans le code :
- Pourquoi `ud = 0.10` pour "STRONG" et non 0.05 ?
- Les 7 niveaux ont-ils été testés par ablation ?
- Une erreur de calibration de ±0.05 sur les masses modifie-t-elle le top-1 ?

**Pour le rapport :** Inclure une table de sensibilité montrant comment
`top1_type` change si les masses sont perturbées de ±10%.

---

#### S4 — `_evidence_scale` fixé à 3.0 avec justification partielle

**Problème :** `SBN_EVIDENCE_SCALE = 3.0` dans `_sl_bijection` joue le rôle de
facteur multiplicatif de l'évidence. La justification dans le code ("même rôle
que dans _evidence_sum_scores") est circulaire. Pour K=11 types et 9 groupes,
l'évidence maximale possible (9 groupes à score=1.0, soit contribution max = 2/3
par groupe en mode dot_product) est :
```
e_max = 9 × (1.0 - 1/3) × 3.0 = 9 × 0.667 × 3.0 = 18.0
u_min = K / (18 + K) = 11 / 29 ≈ 0.38
```
Cette valeur plancher de u ≈ 0.38 pour une attaque "parfaitement connue" est un
paramètre de conception non discuté. Si `evidence_scale = 9.0`, u_min ≈ 0.20.

**Pour le rapport :** Documenter le trade-off et justifier `evidence_scale = 3.0`
(ou le nommer `SBN_EVIDENCE_SCALE` dans CONFIG avec explication).

---

#### S5 — Absence totale d'évaluation quantitative de la qualification

**Problème majeur :** L'ensemble du module est une qualification de **type**
d'attaque, mais aucune métrique de performance de cette qualification n'est calculée.
Aucune comparaison avec des ground-truth de type n'est effectuée (même dans le
mode `--compare`). Le mode `--compare` compare uniquement les top-1 entre SBN et
l'ancien module heuristique, pas contre une vérité terrain.

**Pour le rapport :** Le jury scientifique va demander : "Quelle est la précision de
la qualification de type ?". Si le dataset d'injection (`evaluate_injection_v2.py`)
contient des étiquettes de type d'attaque, un pipeline d'évaluation
`precision@type` doit être implémenté.

---

#### S6 — `_lr_novelty` : interprétation Dempster-Shafer discutable

**Problème :** La référence citée pour `_lr_novelty` est "Shafer 1976,
Dempster-Shafer Theory §4". Mais la métrique calculée est un rapport
`max(L) / mean(L)`, qui est la statistique de Cochran pour la dispersion de
moyennes — pas un concept DST. Le rapport LR maximum/moyen est plus proche
des travaux de Neyman-Pearson ou de Good's Weight of Evidence. La référence
DST est inappropriée.

**Correction :** Citer plutôt Good (1950) §6 (the maximum evidence score) ou
formuler comme "indice de concentration de Herfindahl" sur les scores.

---

#### S7 — Seuil `novelty_lr > 0.85` non calibré

**Problème :** Le seuil `SBN_LR_NOVELTY_THRESHOLD = 0.85` est présenté avec une
"validation" (lignes 724–726) : attaques connues 0.47–0.62, anomalie inconnue ~0.996.
Mais cette validation est faite sur les scores des opinions conditionnelles
elles-mêmes (signatures théoriques) et non sur des données réelles. Sur des
données réelles, le signal peut être bruité. De plus, le seuil n'est pas calibré
sur le dataset RedeRio.

---

#### S8 — Gate L1 appliquée avant L2 : les fenêtres `gate_open=False` ont u_sbn=1.0 par construction

**Problème :** Les fenêtres sous le seuil δ retournent `_empty_result()` avec
`u_sbn = 1.0`. Cela signifie que dans les stats globales (toutes fenêtres), u_sbn
moyen est fortement tiré vers 1.0 par les fenêtres non-anormales. Pour un jury,
il faut distinguer clairement :
- `u_sbn` pour les fenêtres avec `gate_open = True` (seul signal de nouveauté pertinent)
- `u_sbn` pour toutes les fenêtres (artificiellement proche de 1.0 si FPR bas)

Le code affiche déjà la moyenne sur `df_anom` uniquement — c'est correct, mais
le CSV final contient des `u_sbn = 1.0` pour les fenêtres normales, ce qui peut
tromper une analyse avale.

---

### 6.3 Problèmes de forme / lisibilité

#### F1 — Paramètre `W` : aide CLI incorrecte

**Problème :** `--W ... help='Constante bijection SL W (domaine ternaire = 3.0)'`
suggère que ce paramètre contrôle la bijection SL. Voir B2 — il ne fait rien.

---

#### F2 — `novelty_threshold` CLI accepté mais non appliqué dans le CSV de sortie

**Problème :** L'argument `--novelty` est lu par argparse et passé à `run()` comme
`novelty_threshold`, mais cette valeur n'est utilisée nulle part — ni pour créer
une colonne dans le CSV de sortie, ni comme filtre. La stat dans `run()` utilise
`_lr_thr = CONFIG.get('SBN_LR_NOVELTY_THRESHOLD', 0.85)`, pas `novelty_threshold`.
Le paramètre est donc silencieusement ignoré.

---

#### F3 — `prev_gate` variable redondante

**Problème :** `prev_gate` (ligne 1051) sert uniquement à distinguer "on était en
attaque et on est sorti" de "on n'a jamais été en attaque". Mais `prev_opinion`
est `None` dans les deux cas après un reset (ligne 1082). Le comportement pourrait
être simplifié par un check `if prev_opinion is not None: delta_windows += 1`.

---

#### F4 — `--temporal` désactivé par défaut sans explication

**Problème :** L5 (prior temporel) est l'innovation principale de l'architecture
SBN vs qualify_anomaly.py. Sa désactivation par défaut n'est pas justifiée dans
la CLI ni dans la docstring. Un jury va demander pourquoi l'option la plus
sophistiquée est off par défaut.

**Suggestion :** Ajouter dans le help : "Désactivé par défaut pour reproductibilité;
activer pour améliorer la cohérence des séquences d'attaque."

---

#### F5 — Langues mélangées dans les commentaires de `_DEFAULT_SBN_COND`

Les commentaires alternatifs anglais/français à l'intérieur de `_DEFAULT_SBN_COND`
(ex: `# bytes+packets ↑↑↑ — signal primaire` vs `# DISCRIMINATEUR vs ICMP_FLOOD`).
Harmoniser en une seule langue pour la lisibilité d'un rapport international.

---

#### F6 — `type_names = [k for k in sbn_cond.keys()]` redondant (ligne 816 vs 1019)

La liste `type_names` est reconstituée à l'intérieur de `sbn_qualify_row` à
chaque appel (pour `_empty_result()`). Elle est aussi calculée dans `run()` (ligne
1019). La passer en paramètre éviterait la recalculation à chaque ligne.

---

### 6.4 Problèmes structurels

#### ST1 — Double mécanisme d'escompte sans interface unifiée

**Problème :** `_discount_opinion()` implémente l'escompte Jøsang Déf. 14.6.
`_wbf_two()` avec poids différents fait une pondération. L5 utilise les deux
simultanément, ce qui résulte en un comportement difficile à analyser analytiquement.
Pas d'interface permettant de tester chaque mécanisme séparément.

**Recommandation :** Exposer `SBN_TEMPORAL_MODE: "discount_only" | "wbf_only" | "both"`
dans CONFIG pour permettre des ablations propres.

---

#### ST2 — `_compute_group_projected` normalise après geomean mais pas avant

**Problème :** Chaque métrique apporte des probabilités projetées qui somment à 1.0
(par construction dans `compute_opinions_v3.py`). La geomean est calculée état par
état (pas sur le simplex conjoint). Puis le résultat est renormalisé. Mais si
certaines métriques ont des probabilités très proches de 0 pour certains états, le
`log(max(v, EPS))` sature — le signal de ces états est artificiellement amplifié
par le log. Ce cas (attaque forte avec une métrique à `proj_atk ≈ 0`) n'est pas
testé.

---

#### ST3 — `run()` fait à la fois traitement et I/O : couplage fort

**Problème :** `run()` lit le CSV, effectue la qualification, écrit le CSV, et
affiche les stats dans une seule fonction. Cela rend les tests unitaires très
difficiles (nécessite de créer des fichiers CSV). La logique de qualification pure
(`sbn_qualify_row`) est bien séparée, mais l'orchestration dans `run()` mélange
les responsabilités.

**Recommandation :** Séparer `_run_qualification(df, sbn_cond, …) → pd.DataFrame`
de l'I/O.

---

#### ST4 — Absence de vérification de cohérence SBN_COND_OPINIONS vs GROUP_SOURCES

**Problème :** `SBN_COND_OPINIONS` est indexé par `{type → {groupe → distribution}}`.
`GROUP_SOURCES` est indexé par `{groupe → [métriques]}`. Rien ne vérifie que les
groupes dans `SBN_COND_OPINIONS` correspondent exactement aux groupes dans
`GROUP_SOURCES`. Si un groupe manque dans l'un ou l'autre, il est silencieusement
ignoré (les scores pour ce groupe sont omis). Un type pourrait être scoré sur
3 groupes au lieu de 9 sans avertissement.

**Correction :**
```python
missing_groups = set(sbn_cond[type_k].keys()) - set(GROUP_SOURCES.keys())
if missing_groups:
    print(f"[WARN] Type {type_k} : groupes dans SBN_COND non définis dans GROUP_SOURCES : {missing_groups}")
```

---

## 7. Récapitulatif priorisation

| Priorité | ID | Nature | Effort |
|---|---|---|---|
| 🔴 Critique | B1 | `'CONFIG' in dir()` toujours False → SBN_COND de config jamais utilisé | faible |
| 🔴 Critique | B6 | Gate OUTAGE contredit l'architecture annoncée et viole la cohérence SBN | moyen |
| 🔴 Critique | B2 | Paramètre `W` dead code — CLI trompeur | faible |
| 🔴 Critique | B3 | Matrice de transition dead code — L5 non structuré | moyen |
| 🟠 Important | B4 | Double escompte temporel non documenté | doc |
| 🟠 Important | B8 | Fallback colonne gate L1 silencieux → gate jamais ouverte | faible |
| 🟠 Important | S1 | Interprétation probabiliste du dot-product formellement incorrecte | doc |
| 🟠 Important | S5 | Aucune métrique quantitative de qualification de type | moyen |
| 🟠 Important | S7 | Seuil novelty non calibré sur données réelles | doc/exp |
| 🟡 Mineur | B5 | `novelty_score` = `novelty_lr` redondant | faible |
| 🟡 Mineur | B7 | Commentaire matrice transition 0.90 vs code 0.92 | faible |
| 🟡 Mineur | F1, F2 | CLI : `--W` et `--novelty` sans effet | faible |
| 🟡 Mineur | F3, F4 | Variables redondantes ; `--temporal` off par défaut non justifié | faible |
| 🔵 Structurel | ST4 | Absence validation GROUP_SOURCES vs SBN_COND_OPINIONS | faible |
| 🔵 Structurel | ST1 | Double escompte sans interface d'ablation | moyen |
| 🔵 Structurel | ST3 | `run()` mélange traitement et I/O | moyen |
| 📝 Rapport | S2, S3, S4 | Hypothèses non quantifiées : indépendance groupes, calibration masses, evidence_scale | doc |
| 📝 Rapport | S6, S8 | Référence DST incorrecte ; interprétation u_sbn all-windows trompeuse | doc |

---

## 8. Éléments à rédiger dans le rapport technique

### 8.1 Section « Architecture SBN à 6 couches »

- Schéma des 6 couches avec entrées/sorties explicites
- Justifier le passage de 3 (ternaire) à K (multinomial) pour W
- Distinguer les deux modes de scoring (dot_product vs log_lr) et leur base théorique
- Préciser que L5 et L6 sont des extensions optionnelles dont les effets peuvent être ablés

### 8.2 Section « Opinions conditionnelles expertes (`SBN_COND_OPINIONS`) »

- Expliquer que ces matrices sont de l'élicitation d'expert (Cooke 1991 — méthode
  de référence pour l'élicitation Bayésienne) et non de l'apprentissage supervisé
- Présenter un tableau récapitulatif des 11 types × 9 groupes avec les niveaux
  assignés (safe/susp/anom) et les sources littérature pour chaque décision
- Discuter la sensibilité du top-1 à une perturbation de ±0.05 sur les masses
- Mentionner explicitement que pour un jury : des données d'entraînement étiquetées
  permettraient d'apprendre ces distributions au lieu de les éliciter

### 8.3 Section « Score SBN et bijection SL »

- Définir formellement le dot-product comme `E_{s~P^obs}[c^k_s]`
- Montrer que ce score est compris dans [min(c^k), max(c^k)] ⊂ [0,1]
- Formaliser la bijection `e(k) → b(k)` (Jøsang Déf. 3.9) avec W=K
- Analyser les valeurs limite : u=1.0 (nouveauté absolue), u_min≈0.38 (attaque certaine)
- Comparer dot-product vs log-LR sur un exemple RedeRio

### 8.4 Section « Prior temporel Markovien »

- Justifier WBF vs CBF pour sources dépendantes (ou reconnaître la limitation)
- Présenter la courbe de décroissance λ^Δt pour λ=0.80 (demi-vie ≈ 3 fenêtres)
- Distinguer clairement l'escompte Jøsang Déf. 14.6 de la pondération WBF
- Exposer la matrice de transition (même si non implémentée) comme travail futur

### 8.5 Section « Gate OUTAGE »

- Documenter la décision pragmatique de garder une gate séparée pour OUTAGE
  (retirer la mention "sans gate séparée" de la docstring)
- Justifier les seuils `atk_thr=0.50, safe_thr=0.85` sur les données RedeRio
- Proposer une alternative SBN native (déjà présente dans `_DEFAULT_SBN_COND['NETWORK_OUTAGE']`)
  comme validation croisée de la gate heuristique

### 8.6 Section « Évaluation de la qualification de type »

- **À implémenter** : pipeline d'évaluation type vs ground-truth depuis les
  injections de `evaluate_injection_v2.py`
- Métriques à reporter : precision@type, recall@type, confusion matrix (K×K)
- Comparer avec l'heuristique LR de qualify_anomaly.py sur les mêmes injections

### 8.7 Section « Hypothèses et limites »

- H4 Naive Bayes : discuter le biais sous corrélation inter-groupes (Zhang 2004)
- H5 Élicitation expert : sensibilité et comparaison avec apprentissage supervisé
- H9 Décroissance λ^0.80 : impact sur la persistance d'attaque (ablation λ ∈ {0.5, 0.7, 0.9})
- H10 WBF non-indépendant : quelle alternative formelle ?
- Limiter the claim sur BGP_HIJACK et BOTNET_CC : ces types n'ont aucun équivalent
  dans le dataset RedeRio actuel — la qualification est non évaluable

---

## 9. Corrections appliquées (2026-04-12)

| ID | Action | Statut |
|---|---|---|
| B1 | `'CONFIG' in dir()` → `'CONFIG' in globals()` via `_g = globals()` dans `run()` et stats | ✅ |
| B2 | `W` supprimé de `sbn_qualify_row()`, `run()`, `--W` du CLI. `_sl_bijection` : `W = max(K, 2)` | ✅ |
| B3 | `_build_transition_matrix` intégrée en L5 : `b_trans = T.T @ b_prev` (Chapman-Kolmogorov) | ✅ |
| B4 | Double escompte supprimé : `w_temp_eff = temporal_weight` fixe ; `_discount_opinion` seul responsable de la décroissance | ✅ |
| B5 | `novelty_score` supprimé (alias de `novelty_lr`). Seul `novelty_lr` subsiste | ✅ |
| B6 | Gate OUTAGE bypass supprimée. NETWORK_OUTAGE classifié nativement via `SBN_COND_OPINIONS`. Docstring corrigée | ✅ |
| B7 | Commentaire `0.90` corrigé en `0.92` pour cohérence avec le code | ✅ |
| B8 | `_DET_COL` ajouté dans `required_cols` (vérification colonnes obligatoires) | ✅ |
| log_lr | Mode `log_lr` et `_evidence_log_lr_scores` supprimés (dégradation constatée expérimentalement) | ✅ |
| ST4 | Vérification cohérence `GROUP_SOURCES` ↔ `SBN_COND_OPINIONS` ajoutée dans `run()` | ✅ |
| F2 | `novelty_threshold` branché sur `_lr_thr` dans les stats (défaut 0.85 au lieu de 0.65) | ✅ |
| F4 | `--temporal` : help enrichi expliquant la désactivation par défaut | ✅ |
| S5 | `_eval_type_performance()` : precision/recall/F1/N + matrice de confusion par type vs ground-truth | ✅ |
| Sensibilité | `_sensitivity_analysis()` : perturbation ±0.05 sur chaque masse SBN_COND, rapport stabilité par type | ✅ |
| CLI | `--sensitivity` ajouté pour déclencher l'analyse de sensibilité depuis la ligne de commande | ✅ |

---

## 10. Notes croisées vers les autres modules

| Dépendance | Impact |
|---|---|
| `compute_opinions_v3.py` | Fournit `FINAL_SYSTEM_CBF_proj_atk` et `P_{metric}_{col}`. Toute modification de nommage des colonnes de sortie casse `qualify_anomaly_sbn.py` silencieusement (cf. B8). |
| `evaluate_injection_v2.py` | Produit `detection_results_INJECTED.csv` — entrée principale. Si la colonne `_DET_COL` change (ex. `CBF` → `WBF`), B8 devient critique. |
| `config.py` | `QUALIFY_GROUP_SOURCES` doit être cohérent avec les clés de `SBN_COND_OPINIONS`. Vérification manquante (ST4). |
| `sl_formulas_v2.py` | Non importé ici. La bijection SL est réimplémentée localement dans `_sl_bijection()`. Si `sl_formulas_v2.evidence_to_opinion()` évolue, la divergence est silencieuse. |

---

*Document généré lors de la revue de code du 2026-04-11.*
