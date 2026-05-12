# Revue scientifique et technique — `compute_evidence_v2.py`

> Révision du : 2026-04-11  
> Contexte : préparation à l'évaluation par un jury scientifique  
> Fichier principal : `compute_evidence_v2.py`  
> Dépendances directes : `sl_formulas_v2.py`, `config.py`, `paths.py`

---

## 1. Rôle du module dans la chaîne SL-ADS

`compute_evidence_v2.py` est le **deuxième maillon** de la chaîne de détection
après `train_v10.py`. Il prend les modèles prédictifs entraînés (Prophet +
régresseurs de reconstruction) et produit, pour chaque fenêtre temporelle de
`WINDOW_SIZE` pas (défaut : 10 × 30 s = **5 minutes**), un **triplet de preuves
brutes** `(P, S, N)` ∈ ℝ³ au sens de Jøsang (Def. 3.9) :

| Colonne CSV | Signification |
|---|---|
| `{key}_P` | somme des preuves instantanées "Safe" sur la fenêtre |
| `{key}_S` | somme des preuves instantanées "Suspect" sur la fenêtre |
| `{key}_N` | somme des preuves instantanées "Attack" sur la fenêtre |
| `{key}_rmse` | RMSE du modèle sur la fenêtre (passé à C3 comme `online_rmse`) |
| `{key}_iw` | largeur moyenne de l'intervalle de confiance Prophet (ou NaN pour reconst.) |

Ces vecteurs sont ensuite consommés par `compute_opinions_v3.py` pour la
bijection évidence→opinion (Def. 3.9) et les fusions WBF/C3.

---

## 2. Fonctions et outils employés

| Fonctionnalité | Implémentation | Référence |
|---|---|---|
| Prédiction temporelle | `prophet.Prophet.predict(batch)` | Taylor & Letham 2018 |
| Reconstruction linéaire | `sklearn` regressor `.predict()` | — |
| Sérialisation modèles | `joblib.load()` | — |
| Mapping résidu→évidence | `compute_instantaneous_evidence()` (local) | Jøsang 2016, §3 |
| Bijection évidence→opinion | `sl_formulas_v2.evidence_to_opinion()` | Jøsang 2016, Def. 3.9 |
| Classification de stabilité | `_classify_stability()` (local) | seuils ad hoc |
| Profiling mémoire | `tracemalloc` | stdlib Python |
| Profiling temps | `time.perf_counter()` | stdlib Python |
| Priorité CPU | `psutil.HIGH_PRIORITY_CLASS` | Windows uniquement |

---

## 3. Hypothèses posées (explicites et implicites)

### 3.1 Hypothèses sur les données

- **H1 – Stationnarité intra-fenêtre** : les `WINDOW_SIZE` pas d'une fenêtre
  sont traités de façon homogène (même état du système). Si une attaque démarre
  au milieu d'une fenêtre, son signal est dilué par les pas sains.

- **H2 – Imputation forward-fill puis zéro** (`df.ffill().fillna(0)`) : deux
  stratégies enchaînées impliquent : (a) les valeurs manquantes courtes sont
  imputées par la dernière valeur connue ; (b) les manquantes restantes
  (début de série ou NaN structurel) sont imputées à zéro.

- **H3 – Séparation train/test par `split_date`** : toutes les données après
  `split_date` constituent l'ensemble de test ; les modèles ne les ont jamais vus.

### 3.2 Hypothèses sur la détection

- **H4 – Résidu signé comme signal d'anomalie** : `e_t = y_t − ŷ_t` porte
  l'information d'écart. La direction (positive / négative / symétrique) est
  codée dans `pkg['direction']`.

- **H5 – Mapping trapézoïdal de l'évidence** : la fonction de membership choisie
  est linéaire par morceaux (trapézoïde). Elle suppose une transition progressive
  entre les zones Safe → Suspect → Attack. Ce choix est fréquent en logique floue
  mais n'est pas prouvé optimal.

- **H6 – Accumulation additive sur la fenêtre** : `P = Σ p_t`, `S = Σ s_t`,
  `N = Σ n_t` sur `t ∈ [i, i+WINDOW_SIZE[`. Par construction,
  `P + S + N = WINDOW_SIZE` (sauf fenêtre tronquée en fin de série). Ce vecteur
  est interprété comme un vecteur de preuves `r` pour `evidence_to_opinion(r, W)`.

- **H7 – Poids proportionnel au R²** : le coefficient de détermination du modèle
  sur les données d'entraînement (`r2_score`) est utilisé comme poids de
  confiance dans les fusions WBF/C3 downstream. Cela suppose une corrélation
  entre R² en entraînement et fiabilité en test.

### 3.3 Hypothèses sur le modèle SL

- **H8 – Base rate uniforme** : `a = [1/3, 1/3, 1/3]` (défaut `MultinomialOpinion`).
  Suppose une probabilité a priori égale pour Safe, Suspect, Attack.

- **H9 – W = 3 (SL_PARAM_K)** : paramètre de la bijection Jøsang. Valeur
  standard pour le cas ternaire (W = nombre de singletons). Toute modification
  de W change l'échelle des preuves et donc les seuils downstream.

---

## 4. Sorties produites

| Fichier | Contenu | Usage |
|---|---|---|
| `evidence_{VERSION}.csv` | Triplets P/S/N + RMSE + IW par fenêtre | Entrée `compute_opinions_v3.py` |
| `raw_data_{VERSION}.csv` | Résidus instantanés par pas (debug) | Visualisation, diagnostic |
| `metadata_{VERSION}.csv` | Profil par indicateur : type, R², seuils, stabilité | Documentation, auditabilité |

---

## 5. Paramètres de configuration impliqués

| Paramètre CONFIG | Rôle | Valeur typique |
|---|---|---|
| `WINDOW_SIZE` | Taille de fenêtre temporelle (en pas) | 10 |
| `split_date` | Date de séparation train/test | `"2025-11-09 23:59:59"` |
| `file_path` | Chemin du CSV de données brutes | `../data/dataset_*.csv` |
| `HOLIDAYS_LIST` | Jours fériés pour la feature `on_weekend` | liste ou vide |
| `TUNING_MODE` + `TUNING_START/END` | Restreindre l'analyse à une plage | facultatif |
| `SL_PARAM_K` (W) | Constante de bijection SL | 3.0 |
| `VERSION_NAME` | Versionnage des artefacts | ex. `trained_models_v9_v9_v4s_v3_v2` |
| `TIME_BINARY_TO_CSV` | Temps de conversion binaire→CSV | chargé mais inutilisé ici |

---

## 6. Problèmes identifiés

### 6.1 Bugs / erreurs silencieuses

#### B1 — `is_holiday = False` scalaire (ligne 125)
**Code :**
```python
else:
    is_holiday = False
df['on_weekend'] = (is_cal_wknd | is_holiday).astype(int)
```
**Problème :** `is_holiday` est un booléen Python, pas une `pd.Series`. Le
`|` avec une Series pandas fonctionne par diffusion implicite mais est
sémantiquement incorrect et produit un `FutureWarning` sur certaines versions.

**Correction :**
```python
else:
    is_holiday = pd.Series(False, index=df.index)
```

---

#### B2 — Clé CSV incohérente pour `_iw` vs `_P/_S/_N` (lignes 191 vs 241)
**Code :**
```python
ev_row[f"{key}_iw"] = ...          # key brut, peut contenir "->"
ev_row[f"{clean_key}_P"] = P       # clean_key, "->" remplacé par "_to_"
```
**Problème :** Pour un indicateur dont le nom contient `->`, la colonne `_iw`
aura un nom différent de `_P/_S/_N` dans le CSV. Le parsing downstream sera
silencieusement incorrect.

**Correction :** Utiliser `clean_key` partout dès que la clé est écrite dans `ev_row`.

---

#### B3 — `EXT_TIME` chargé mais jamais utilisé (lignes 108-110)
**Code :**
```python
EXT_TIME = CONFIG['TIME_BINARY_TO_CSV']
```
**Problème :** Cette variable est extraite (et peut lever une `KeyError`) mais
n'est utilisée nulle part dans le fichier. Dead code risqué.

**Correction :** Supprimer ce bloc ou l'utiliser dans le résumé de performance.

---

#### B4 — `total_flows_count` calculé mais jamais sauvegardé (lignes 157, 171-174)
**Code :**
```python
total_flows_count = 0
...
total_flows_count += batch['flows'].sum()
```
**Problème :** La variable est incrémentée mais n'est ni affichée, ni sauvegardée.
Dead code.

**Correction :** L'inclure dans le résumé final (`print`) ou le supprimer.

---

#### B5 — `r2_score` non clampé peut être négatif (ligne 151)
**Code :**
```python
metric_weights = {key: models_pkg[key].get('r2_score', 0.5) for key in all_metric_keys}
```
**Problème :** R² peut être négatif si le modèle prédit moins bien que la
moyenne. Un poids négatif passé à WBF/C3 downstream peut corrompre la fusion.

**Correction :**
```python
metric_weights = {
    key: max(0.0, models_pkg[key].get('r2_score', 0.5))
    for key in all_metric_keys
}
```

---

#### B6 — Seuils de fallback `1e-6` / `1e-5` dangereux (lignes 218-219)
**Code :**
```python
t_susp = pkg.get('t_susp', pkg.get('thresholds', {}).get('suspect', 1e-6))
t_atk  = pkg.get('t_atk',  pkg.get('thresholds', {}).get('attack',  1e-5))
```
**Problème :** Pour des métriques réseau en bytes (ordres de grandeur 10³–10⁶),
si le package n'a pas de seuil, tout résidu > 1e-6 sera classifié Attack.
Cela masquerait silencieusement un artefact d'entraînement.

**Correction :** Lever une exception ou un warning explicite si les seuils sont absents.
```python
if 't_susp' not in pkg and 'thresholds' not in pkg:
    raise KeyError(f"[{key}] Aucun seuil trouvé dans le package modèle.")
```

---

### 6.2 Problèmes scientifiques et manques de justification

#### S1 — `_classify_stability` : seuils arbitraires non référencés (lignes 89-95)
```python
def _classify_stability(r2: float, kurtosis: float, cv: float) -> str:
    if r2 >= 0.65 and kurtosis <= 5 and cv <= 0.5:
        return "A"
    elif r2 >= 0.40 or (kurtosis <= 10 and cv <= 1.5):
        return "B"
    else:
        return "C"
```
**Problème :** Les seuils r2=0.65/0.40, kurtosis=5/10, cv=0.5/1.5 ne sont
référencés dans aucun article. Leur origine (empirique sur le dataset RedeRio ?
tradition du domaine ?) doit être documentée.

**Pour le rapport :** Indiquer explicitement comment ces seuils ont été fixés
(grid-search, validation sur jeu de validation, règle empirique issue de la
littérature), et vérifier si la classification impacte les résultats finaux.

---

#### S2 — Imputation des données manquantes non justifiée (ligne 116)
```python
df = df.sort_values('ds').ffill().fillna(0)
```
**Problème :** L'enchaînement forward-fill + zéro-remplissage est une décision
méthodologique importante pour les séries temporelles réseau. Il faudrait :
- Quantifier le taux de valeurs manquantes dans le dataset
- Justifier la stratégie (durée typique des lacunes, comportement attendu du
  trafic lors d'une indisponibilité)
- Documenter l'impact potentiel sur les résidus (un zéro imputable crée un
  faux grand résidu si le modèle prédit la valeur attendue)

---

#### S3 — Fenêtre temporelle comme unité d'évidence non discutée
**Problème :** La granularité de l'évidence est `WINDOW_SIZE × freq_données` =
5 minutes (pour RedeRio). Ce choix implique qu'une attaque de moins de 5 minutes
peut être diluée à moins d'une preuve Attack dans la fenêtre. L'impact sur les
métriques de détection (latence de détection, sensibilité) n'est pas discuté.

**Pour le rapport :** Analyser la sensibilité du F1/recall aux valeurs de
`WINDOW_SIZE` (1, 5, 10, 20).

---

#### S4 — Sémantique de l'accumulation P+S+N non documentée dans le code
**Problème :** Le lecteur du code (ou du CSV) ne sait pas que `P+S+N = WINDOW_SIZE`
par construction. Sans ce commentaire, il est difficile de comprendre pourquoi
les preuves sont des entiers et non des probabilités dans `[0,1]`.

**Correction :** Ajouter un commentaire expliquant que `(P,S,N)` est un
vecteur `r` au sens de Jøsang Def. 3.9, utilisé avec `evidence_to_opinion(r, W=3)`.

---

#### S5 — `t_trapeze_base` par défaut à `0.9 × t_susp` non justifié (ligne 220)
**Problème :** La zone de pré-suspicion occupe 10% de `t_susp`. Ce paramètre
n'est pas exposé dans CONFIG, ce qui empêche toute ablation. Une valeur de
transition plus large (ex. 50%) donnerait une montée plus progressive de l'évidence.

**Pour le rapport :** Soit justifier empiriquement le 10%, soit exposer le
paramètre dans CONFIG sous le nom `T_TRAPEZE_RATIO` et l'inclure dans l'analyse de sensibilité.

---

#### S6 — Absence de vérification de fuite train/test
**Problème :** Rien ne garantit que le `models_pkg` chargé a été entraîné sur
des données antérieures à `split_date`. Une expérimentation mal ordonnée
(rechargement d'un ancien artefact) pourrait produire des métriques de
détection artificiellement gonflées.

**Correction :** Stocker et vérifier la `train_end_date` dans le package modèle :
```python
train_end = models_pkg.get('train_end_date')
if train_end and pd.to_datetime(train_end) >= split_date:
    raise RuntimeError("Fuite train/test : modèle entraîné au-delà de split_date.")
```

---

#### S7 — La largeur d'intervalle Prophet (`_iw`) est moyennée sur la fenêtre
**Problème :** `iw = mean(yhat_upper - yhat_lower)` agrège l'incertitude
ponctuelle en une valeur scalaire par fenêtre. Si l'intervalle est très large
sur un seul pas (zone à forte incertitude), cela est dilué. Cette valeur
est probablement utilisée pour pondérer la confiance downstream — l'agrégation
doit être documentée et justifiée.

---

### 6.3 Problèmes de forme / lisibilité

#### F1 — Double instruction sur une ligne (ligne 222, PEP8)
```python
t_susp_pos = pkg.get(...); t_atk_pos = pkg.get(...)  # violation PEP8 E702
```

#### F2 — Référence à `train_v9.py` obsolète (ligne 132)
```python
print(f"❌ Modèle '{MODEL_PATH}' introuvable. Lancez train_v9.py d'abord.")
```
Le nom du fichier d'entraînement a évolué. Utiliser un message générique ou
référencer le bon fichier.

#### F3 — Nommage trompeur `avg_inf` (ligne 289)
```python
avg_inf = np.mean(proc_times)
```
"inf" suggère "inference" mais la variable mesure le temps complet d'une fenêtre
(inférence + calcul d'évidence + accumulation des raw_rows). Renommer en
`avg_window_ms` ou `avg_proc_time`.

#### F4 — `compute_evidence()` sans docstring
La fonction principale n'a aucune documentation. Pour un jury scientifique,
c'est le premier point qu'il lira.

#### F5 — Commentaires mélangés français/anglais
Exemple : `# RMSE fenêtre (pour C3 online_rmse)` vs `# Split test set`.
Pour un rapport international, harmoniser en anglais. Pour un rapport
francophone, harmoniser en français.

---

### 6.4 Problèmes structurels

#### ST1 — `compute_instantaneous_evidence` duplique `sl.fuzzy_trapezoid_relative`
La fonction locale est une réécriture étendue (avec support directionnel) de
`sl_formulas_v2.fuzzy_trapezoid_relative`. Les deux coexistent, créant un risque
de divergence silencieuse si l'une est modifiée sans l'autre.

**Recommandation :** Faire évoluer `sl_formulas_v2.fuzzy_trapezoid_relative`
pour accepter un argument `direction`, et supprimer la fonction locale.

---

#### ST2 — État global calculé à l'import
`VERSION_NAME`, `OUTPUT_DIR`, `MODEL_PATH`, `WINDOW_SIZE` sont instanciés au
niveau module. Cela rend le module non testable sans `config.py` valide et
empêche l'appel avec différentes configs dans le même process.

**Recommandation :** Passer `config` en paramètre de `compute_evidence()` et
calculer ces variables à l'intérieur.

---

#### ST3 — `raw_rows` entièrement en mémoire avant dump
Pour un dataset long : `N_fenêtres × N_métriques × WINDOW_SIZE` lignes avant
écriture. Exemple : 10 000 × 10 × 10 = 1M lignes (~100 MB). Pour les gros
datasets ou les expériences longues, utiliser `pd.DataFrame.to_csv` en mode
append par batch ou un streaming writer.

---

#### ST4 — Pas de schéma de validation pour `models_pkg`
La structure du dictionnaire `.pkl` est implicite et documentée nulle part.
Une validation explicite à l'entrée (ex. via `pydantic` ou un check manuel)
renforcerait la reproductibilité et faciliterait le débogage.

---

## 7. Récapitulatif priorisation

| Priorité | ID | Nature | Effort |
|---|---|---|---|
| 🔴 Critique | B2 | Incohérence clé `_iw` vs `_P/_S/_N` dans CSV | faible |
| 🔴 Critique | B6 | Seuils fallback 1e-6 dangereux | faible |
| 🔴 Critique | B5 | R² non clampé → poids négatif | faible |
| 🟠 Important | S6 | Fuite train/test non vérifiée | moyen |
| 🟠 Important | S2 | Imputation non justifiée | doc |
| 🟠 Important | S4 | Sémantique P+S+N non documentée | doc |
| 🟡 Mineur | B1 | `is_holiday` scalaire | faible |
| 🟡 Mineur | B3, B4 | Dead code (`EXT_TIME`, `total_flows_count`) | faible |
| 🟡 Mineur | F1–F5 | Forme/lisibilité | faible |
| 🔵 Structurel | ST1 | Duplication `compute_instantaneous_evidence` | moyen |
| 🔵 Structurel | ST2 | État global à l'import | fort |
| 📝 Rapport | S1, S3, S5, S7 | Justifications à rédiger | doc |

---

## 8. Éléments à rédiger dans le rapport technique

### 8.1 Section « Pipeline de calcul de l'évidence »
- Décrire le rôle de chaque fenêtre temporelle (5 min pour RedeRio)
- Justifier le choix `WINDOW_SIZE = 10` (trade-off latence / robustesse)
- Expliquer la bijection `r = [P,S,N]` → opinion (Def. 3.9, W=3)
- Citer Jøsang (2016) Sections 3.4–3.5

### 8.2 Section « Mapping résidu → évidence »
- Présenter la fonction trapézoïdale avec schéma (zones safe/suspect/attack)
- Justifier les seuils `t_susp`, `t_atk` (issus de `train_v10` par percentile)
- Justifier le traitement directionnel (asymétrie physique des anomalies réseau)
- Discuter le paramètre `t_trapeze_base` (zone de transition)

### 8.3 Section « Stabilité des modèles prédictifs »
- Définir les classes A/B/C avec les seuils (Tableau §6.2 S1)
- Documenter d'où viennent les seuils R²/kurtosis/CV
- Indiquer comment la classe de stabilité est exploitée downstream (C3 ?)

### 8.4 Section « Hypothèses et limites »
- H1 Stationnarité intra-fenêtre → latence minimale de détection
- H2 Imputation → risque de faux résidus sur les lacunes
- H4–H5 Résidu signé + trapézoïde → justification de la forme choisie
- H7 R² comme poids → discussion biais/variance

### 8.5 Section « Reproductibilité »
- Décrire le versionnage (`VERSION_NAME`, `VERSION_SUFFIX`)
- Indiquer que le seuil de décision est auto-calibré et stocké dans le sidecar JSON
- Mentionner l'override par variable d'environnement (`SL_ACTIVE_DATASET`)

---

## 9. Modifications appliquées (2026-04-11)

| ID | Action |
|---|---|
| B1 | `is_holiday = pd.Series(False, index=df.index)` |
| B2 | `ev_row[f"{clean_key}_iw"]` et `ev_row[f"{clean_key}_rmse"]` uniformisés |
| B3 | Bloc `EXT_TIME` supprimé |
| B4 | `total_flows_count` supprimé |
| B5 | `r2_score` clampé à `max(0.0, ...)` |
| B6 | Seuils fallback remplacés par un exit propre avec message d'erreur explicite |
| S1 | `_classify_stability` et colonne `stability_class` supprimées du metadata |
| S2 | `ffill().fillna(0)` → `fillna(0)` avec commentaire justificatif |
| S4 | Docstring de `compute_evidence()` explique la sémantique `P+S+N = WINDOW_SIZE` |
| S5 | `T_TRAPEZE_RATIO` ajouté dans `config.py` et utilisé pour `t_trapeze_base` |
| S6 | Vérification anti-fuite via `_meta_split_date` (stocké dans le pkg par `train_v10.py`) |
| ST1 | Commentaire expliquant pourquoi `compute_instantaneous_evidence` est distinct de `sl.fuzzy_trapezoid_relative` |
| F1 | `P += p; S += s; N += n` → 3 lignes séparées |
| F2 | Référence `train_v9.py` → `train_v10.py` |
| F3 | `avg_inf` → `avg_window_time` |
| F4 | Docstring ajoutée sur `compute_evidence()` |
| — | Import `sl_formulas_v2 as sl` supprimé (non utilisé) |
| — | `clean_key` déplacé en tête de boucle pour usage uniforme |

### Problème résiduel cross-fichier (non corrigé ici)

`compute_opinions_v3.py` lignes **436** et **604** lisent `f"{key}_iw"` et `f"{key}_rmse"`
(clé brute), alors que `compute_evidence_v2.py` écrit désormais `f"{clean_key}_iw"`
et `f"{clean_key}_rmse"` (clé nettoyée). **Actuellement sans impact** : aucune clé
active ne contient `"->"` donc `key == clean_key`. À corriger dans `compute_opinions_v3.py`
lors de sa prochaine revue.

---

*Document généré lors de la revue de code du 2026-04-11.*
