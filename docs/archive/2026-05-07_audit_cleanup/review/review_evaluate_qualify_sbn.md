# Revue scientifique et technique — `evaluate_qualify_sbn.py`

> Révision du : 2026-04-13  
> Contexte : préparation à l'évaluation par un jury scientifique  
> Fichier principal : `evaluate_qualify_sbn.py`  
> Dépendances directes : `qualify_anomaly_sbn.py`, `config.py` (`REAL_ATTACKS`), `paths.py`

---

## 1. Rôle du module dans la chaîne SL-ADS

`evaluate_qualify_sbn.py` est le **module d'évaluation terminal** de la chaîne SL-ADS.
Il prend en entrée le CSV produit par `qualify_anomaly_sbn.py` (ou `qualify_anomaly.py`)
et mesure quantitativement les performances de qualification :

| Mode | Déclenchement | Ce qu'il évalue |
|---|---|---|
| `--injected` (défaut) | `evaluate_injected()` | 8 types d'attaque synthétiques + 1 contrôle de nouveauté |
| `--real` | `evaluate_real()` | 3 événements réels annotés du dataset RedeRio |

Le module répond à quatre questions fondamentales :
1. **Détection** : quelle fraction des fenêtres injectées sont-elles détectées (gate ouverte) ?
2. **Qualification** : parmi les fenêtres détectées, le type attribué est-il correct (top-1) ?
3. **Latence** : combien de fenêtres s'écoulent entre début d'attaque et première qualification correcte (TTQ) ?
4. **Nouveauté** : le signal `novelty_lr` dépasse-t-il le seuil `LR_NOVELTY_THR` pour les anomalies hors-catalogue ?

---

## 2. Fonctions et outils employés

| Fonctionnalité | Implémentation | Référence |
|---|---|---|
| Détection format CSV | `_detect_format(df)` | heuristique sur noms de colonnes |
| Résolution colonnes SBN vs LR | `_get_col_prefix(fmt)` | — |
| Évaluation injections synthétiques | `evaluate_injected(df, fmt)` | itération sur `INJECTED_ATTACKS` |
| Évaluation attaques réelles | `evaluate_real(df, fmt)` | itération sur `REAL_ATTACKS` (config.py) |
| Calcul recall/precision par attaque | inline dans `evaluate_injected` | définitions classiques TP/FN/FP |
| TTQ (Time-to-Qualify) | `iloc[0]` sur fenêtres correctement classifiées | — |
| Signal de nouveauté `novelty_lr` | lecture colonne CSV (produit par `qualify_anomaly_sbn.py`) | Good (1950) § rapport de vraisemblance |
| Agrégation macro recall/precision | `df_known['recall'].mean()` | macro-moyenne non pondérée |

---

## 3. Hypothèses posées (explicites et implicites)

### 3.1 Hypothèses sur le ground truth

- **H1 — Toutes les fenêtres dans [t_start, t_end] sont positives** : `n_total = len(df_ev)` sert
  de dénominateur au recall. Cela suppose que l'injection est **active et uniforme** sur toute la durée
  déclarée, sans rampe de montée, sans latence de propagation. Si l'injection a un délai d'effet
  (ex. : le trafic injecté n'affecte les résidus qu'après quelques pas), les premières fenêtres
  seront comptées comme faux négatifs de façon incorrecte.

- **H2 — `gate_open = True` ⟺ anomalie détectée (TP ou FP)** : le recall vaut `n_detected / n_total`.
  Cela suppose que la gate est un détecteur binaire parfaitement synchronisé avec l'injection.

- **H3 — `top1_type` est la prédiction de qualification** : le type SBN le plus probable est
  utilisé comme décision unique, sans prise en compte de la distribution complète des croyances.
  Les cas `top1_type == ''` (fenêtres avec `gate_open=False`) sont exclus.

- **H4 — Pas de fenêtres hors-dataset** : si la période d'attaque est hors de la plage active
  du CSV, `n_total = 0` est affiché avec un avertissement. La vérification est présente pour
  les attaques réelles mais pas pour les attaques injectées.

### 3.2 Hypothèses sur les métriques

- **H5 — Précision définie comme précision de qualification, pas de détection** :
  `precision = n_correct / n_detected` mesure, parmi les fenêtres où la gate est ouverte,
  la fraction correctement typées. Il ne s'agit **pas** de la précision de détection
  (TP / (TP + FP) globale). Cette distinction est critique pour l'interprétation.

- **H6 — Recall = taux de détection (DR), pas recall de classification** :
  `recall = n_detected / n_total` est le taux de détection pendant la période d'attaque.
  Ce n'est pas le recall au sens de la classification du type (TP_type / (TP_type + FN_type)).

- **H7 — TTQ = temps jusqu'à la première qualification correcte** : `iloc[0]` sur les fenêtres
  correctement qualifiées suppose que `df` est **trié par timestamp**. Si ce n'est pas le cas,
  TTQ sera erroné (voir B4).

- **H8 — `novelty_lr` mesuré uniquement sur fenêtres gate_open** : la moyenne de `novelty_lr`
  est calculée sur `df_ev[df_ev['gate_open']]`. Les fenêtres non-détectées (gate_close) ont
  `novelty_lr = 1.0` par convention dans `qualify_anomaly_sbn.py` mais ne doivent pas être
  interprétées comme un signal de nouveauté — cette exclusion est correcte.

### 3.3 Hypothèses sur le seuil de nouveauté

- **H9 — Seuil LR_NOVELTY_THR = 0.85 validé sur signatures théoriques** : la docstring de
  `qualify_anomaly_sbn.py` indique explicitement que ce seuil est validé sur des signatures
  **parfaites** (non bruitées) et doit être recalibré empiriquement. Aucune calibration ROC
  n'est effectuée dans ce fichier d'évaluation.

---

## 4. Sorties produites

`evaluate_qualify_sbn.py` produit uniquement des **sorties console** (stdout). Il n'y a pas
d'écriture de fichier CSV ni JSON.

| Section console | Contenu | Condition |
|---|---|---|
| Tableau par attaque | Recall, Précision, Top-1 incorrect, u_sbn moyen, novelty_lr moyen | `--injected` |
| [A] Attaques connues | Macro recall / précision + TTQ par attaque | `--injected` |
| [B] Contrôle nouveauté | Recall + novelty_lr + verdict | `--injected` |
| Bloc par événement réel | Recall, distribution top-1, TTQ, novelty_lr | `--real` |

**Absence critique** : aucun fichier de résultats n'est sauvegardé, ce qui rend la
comparaison entre versions et la traçabilité expérimentale impossibles sans redirection stdout.

---

## 5. Paramètres de configuration impliqués

| Paramètre | Source | Valeur typique | Rôle |
|---|---|---|---|
| `SBN_NOVELTY_THRESHOLD` | `config.py` / fallback 0.65 | 0.65 | Seuil u_sbn pour nouveauté (non utilisé dans ce fichier) |
| `SBN_LR_NOVELTY_THRESHOLD` | `config.py` / fallback 0.85 | 0.85 | Seuil novelty_lr pour signal de nouveauté |
| `WINDOW_MIN` | `CONFIG['WINDOW_MINUTES']` | 5 | Durée d'une fenêtre (pour contexte, non utilisé en calcul) |
| `RESULTS_DIR` | `paths.get_results_dir()` | `../results/resultats_…/` | Répertoire de recherche du CSV |
| `REAL_ATTACKS` | `config.REAL_ATTACKS` | dict 3 événements | Catalogue des attaques réelles annotées |

---

## 6. Problèmes identifiés

---

### 6.1 QUESTION CENTRALE : Pourquoi NETWORK_OUTAGE n'apparaît-il pas dans le tableau ?

#### Réponse directe

Le tableau `Attaque | Intens. | Fenêtres | Détect. | Recall | Qualif. | Précision | Top-1 incorrect`
est généré par `evaluate_injected()`, qui itère exclusivement sur la liste `INJECTED_ATTACKS`.
**`NETWORK_OUTAGE` n'est pas dans `INJECTED_ATTACKS`.**

Les coupures réseau réelles (`NETWORK_OUTAGE_NOV17`, `NETWORK_OUTAGE_DEC1617`) sont
présentes dans `REAL_ATTACKS` et évaluées **uniquement** via le flag `--real`.

#### Conséquences scientifiques

Cela crée deux lacunes :

1. **Aucun test contrôlé de NETWORK_OUTAGE** : contrairement aux 8 autres types (UDP_FLOOD,
   SYN_FLOOD, etc.) qui sont testés par injection synthétique avec intensité et durée maîtrisées,
   NETWORK_OUTAGE n'est testé que sur des événements naturels dont la "vérité terrain" est
   incertaine (confiance via analyse résiduelle, pas de label automatique).

2. **BGP_HIJACK et BOTNET_CC** sont dans le catalogue SBN (`_DEFAULT_SBN_COND`) et dans la
   matrice de transition, mais **ne sont ni injectés ni évalués** par aucun mode.

#### Correction recommandée

Ajouter dans `INJECTED_ATTACKS` (en choisissant une période sans autre injection) :
```python
{'name': 'NETWORK_OUTAGE_SYNTHETIC', 'expected': 'NETWORK_OUTAGE',
 'start': '2025-12-22 02:00:00', 'end': '2025-12-22 04:00:00',
 'intensity': 'high', 'is_novelty_control': False},
{'name': 'BGP_HIJACK_SYNTHETIC', 'expected': 'BGP_HIJACK',
 'start': '2025-12-23 10:00:00', 'end': '2025-12-23 13:00:00',
 'intensity': 'medium', 'is_novelty_control': False},
{'name': 'BOTNET_CC_SYNTHETIC', 'expected': 'BOTNET_CC',
 'start': '2025-12-24 00:00:00', 'end': '2025-12-24 06:00:00',
 'intensity': 'low', 'is_novelty_control': False},
```

---

### 6.2 Bugs / erreurs silencieuses

#### B1 — Logique de résolution du CSV cassée : `if csv_path is None` n'est jamais atteint (ligne 264–270)

**Code actuel :**

```python
csv_path = r"../../results/resultats_trained_models_v9_v9_v4s_v3_v2/qualif_types_sbn.csv"  # ligne 264
for c in candidates:
    if os.path.exists(c):
        csv_path = c
        break
if csv_path is None:  # ← JAMAIS vrai : csv_path est déjà une string non-nulle
    print(...)
    sys.exit(1)
```

**Problème :** La guard `if csv_path is None` ne peut jamais se déclencher car `csv_path` a
déjà été assigné à un chemin hardcodé (Windows, version-dépendant, séparateurs `\`). Si le
chemin hardcodé n'existe pas et qu'aucun candidat n'est trouvé, le programme continue et
crashe sur `pd.read_csv()` avec une erreur opaque.

**Correction :**
```python
csv_path = None
for c in candidates:
    if os.path.exists(c):
        csv_path = c
        break
if csv_path is None:
    # Fallback explicite avec vérification
    fallback = os.path.join("..", "results",
                            "resultats_trained_models_v9_v9_v4s_v3_v2",
                            "qualif_types_sbn.csv")
    if os.path.exists(fallback):
        csv_path = fallback
        print(f"  [!] Fallback hardcodé utilisé : {fallback}")
    else:
        print(f"Aucun CSV trouvé dans {RESULTS_DIR}. Passe --csv <chemin>.")
        sys.exit(1)
```

---

#### B2 — TTQ calculé sans tri préalable par timestamp (ligne 144–145)

**Code :**
```python
df_correct = df_ev[df_ev['gate_open'] & (df_ev['top1_type'] == expected)]
ttq = ((df_correct['timestamp'].iloc[0] - t_start).total_seconds() / 60
       if len(df_correct) else float('nan'))
```

**Problème :** `df_ev` est un sous-ensemble de `df` filtré par masque temporel, mais si `df`
n'est pas trié par timestamp (ce qui peut arriver si le CSV a été produit avec des concatenations
pandas non triées), `iloc[0]` ne correspond pas à la première fenêtre chronologiquement correcte.
`TTQ` peut être négatif ou fantaisiste.

**Correction :**
```python
df_correct = df_ev[df_ev['gate_open'] & (df_ev['top1_type'] == expected)]\
    .sort_values('timestamp')
```

Identique ligne 223–225 dans `evaluate_real()`.

---

#### B3 — `novelty_lr` non protégé contre division par zéro dans `evaluate_real()` (ligne 231–233)

**Code :**
```python
lr_col  = 'novelty_lr' if 'novelty_lr' in df_ev.columns else u_col
lr_mean = df_ev[df_ev['gate_open']][lr_col].mean()
print(f"  novelty_lr moyen : {lr_mean:.3f}  ...")
```

**Problème :** Si `n_detected == 0`, la moyenne sur un DataFrame vide retourne `NaN`, et
`f"{lr_mean:.3f}"` affiche `nan` — ce qui n'est pas une erreur Python mais est trompeur
(le message "pas de nouveauté" ne devrait pas s'afficher si aucune fenêtre n'est détectée).
Le bloc est à l'intérieur du `if n_detected > 0:` dans `evaluate_injected()` mais **pas**
dans `evaluate_real()`.

**Correction :** Dans `evaluate_real()`, placer le calcul de `lr_mean` à l'intérieur du
bloc `if n_detected > 0:`.

---

#### B4 — `NOVELTY_CONTROLS` défini mais jamais utilisé (ligne 40)

**Code :**
```python
NOVELTY_CONTROLS = {'UNKNOWN_ANOMALY_CONTROL'}
```

**Problème :** Cette constante est définie au niveau module mais n'est jamais référencée.
La logique de contrôle de nouveauté utilise `ev['is_novelty_control']` dans chaque entrée
du dictionnaire. Dead code.

**Correction :** Supprimer ou utiliser cette constante comme référence canonique (remplacer
`ev['is_novelty_control']` par `ev['name'] in NOVELTY_CONTROLS` pour unifier les deux
mécanismes).

---

#### B5 — Chemin hardcodé Windows incompatible Linux/Mac (ligne 264)

**Code :**
```python
csv_path = r"../results/resultats_trained_models_v9_v9_v4s_v3_v2\qualif_types_sbn.csv"
```

Séparateur `\` dans une raw string : non portable. Utiliser `os.path.join()` ou `pathlib.Path`.

---

#### B6 — `n_detected` non casté en `int` dans `evaluate_real()` vs `evaluate_injected()`

Dans `evaluate_injected()` : `n_detected = int(df_ev['gate_open'].sum())` — cast explicite.
Dans `evaluate_real()` : `n_detected = df_ev['gate_open'].sum()` — numpy int64.
Inconsistance mineure mais peut causer des problèmes de sérialisation/formatage selon la version pandas.

---

### 6.3 Problèmes scientifiques et manques de justification

#### S1 — Absence totale d'évaluation pour NETWORK_OUTAGE, BGP_HIJACK, BOTNET_CC

Le catalogue SBN contient 11 types + Autre_Anomalie. Le module d'évaluation ne teste que 8 types
via injection synthétique. Les 3 types manquants :

| Type | Dans `INJECTED_ATTACKS` | Dans `REAL_ATTACKS` | Dans SBN_COND |
|---|---|---|---|
| `NETWORK_OUTAGE` | ✗ | ✓ (2 événements, mode `--real`) | ✓ |
| `BGP_HIJACK` | ✗ | ✗ | ✓ |
| `BOTNET_CC` | ✗ | ✗ | ✓ |

Pour un jury scientifique, **un type dans le catalogue mais non évalué est une prétention non
vérifiée**. Il faut soit évaluer ces types (injection synthétique), soit explicitement
reconnaître dans le rapport qu'ils sont dans le catalogue "par design" mais non validés faute
d'injection disponible.

---

#### S2 — Recall et précision : définitions implicites non documentées

Le code utilise :
- **Recall** = `n_detected / n_total` = fraction des fenêtres d'attaque détectées (= Detection Rate / DR)
- **Précision** = `n_correct / n_detected` = fraction des détections correctement typées

Ces deux métriques sont **correctes et utiles** mais ne correspondent pas aux définitions standard
du recall et de la précision en classification binaire. Il faut les nommer explicitement :
- `recall` → `detection_rate` (DR) ou `window_recall`
- `precision` → `qualification_accuracy` ou `type_precision`

**Ou bien**, définir explicitement que le ground truth est binaire (dans la période = positif)
et que recall = DR et precision = précision de qualification conditionnelle à la détection.

---

#### S3 — Macro-moyenne non pondérée : biais pour les attaques de durées très différentes

```python
macro_recall    = df_known['recall'].mean()
macro_precision = df_known['precision'].mean()
```

Les 8 attaques ont des durées très différentes (4h pour UDP_FLOOD vs 8h pour SLOWLORIS vs
30 min pour ICMP_FLOOD). Une macro-moyenne non pondérée donne le même poids à une attaque
de 30 min (6 fenêtres) et à une attaque de 8h (96 fenêtres). Une **micro-moyenne pondérée**
par `n_total` serait plus représentative :

```python
micro_recall = df_known['n_detected'].sum() / df_known['n_total'].sum()
micro_precision = df_known['n_correct'].sum() / df_known['n_detected'].sum()
```

Pour un jury, **rapporter les deux** (macro et micro) est recommandé.

---

#### S4 — TTQ sans distribution ni comparaison avec TTD

Le TTQ (Time-to-Qualify, première qualification correcte) est rapporté uniquement comme
**première occurrence**. Manques :

1. **Distribution TTQ** : si la première fenêtre correcte est un outlier (noise spike), le TTQ
   réel est sous-estimé. Minimum, médiane, P95 devraient être rapportés.
2. **Comparaison TTQ / TTD** : TTD (Time-to-Detect, première gate ouverte) est calculable
   depuis `evaluate_injection_v2.py` ou directement ici. Le ratio TTQ/TTD mesure le surcoût
   de la qualification vs la détection brute.
3. **TTQ pour attaque jamais qualifiée** : `TTQ_min = None` est affiché comme "non qualifié",
   mais la cause (non détecté ? détecté mais mauvais type ?) n'est pas distinguée.

---

#### S5 — Absence de courbe ROC pour `novelty_lr`

Le seuil `LR_NOVELTY_THR = 0.85` est validé sur signatures **théoriques parfaites**, pas sur
des données bruitées réelles. La procédure recommandée dans `qualify_anomaly_sbn.py`
("recalibrer empiriquement — courbe ROC sur novelty_lr") n'est **jamais effectuée** dans
ce module d'évaluation.

Pour un jury, il faut :
1. Calculer `novelty_lr` pour les 8 attaques connues (labels = 0) et pour `UNKNOWN_ANOMALY_CONTROL` (label = 1)
2. Tracer la courbe ROC et rapporter l'AUC
3. Justifier le seuil 0.85 sur cette base empirique (ou le remplacer par le seuil maximisant Youden's J)

---

#### S6 — Validation du contrôle de nouveauté insuffisante

Le test actuel pour `UNKNOWN_ANOMALY_CONTROL` :
```python
_sig_ok = not np.isnan(lr_mean) and lr_mean > LR_NOVELTY_THR
```

Cela vérifie seulement que la **moyenne** de `novelty_lr` dépasse le seuil. Issues :
1. Une moyenne peut dépasser 0.85 si quelques fenêtres sont très hautes et d'autres basses.
   La **fraction** de fenêtres avec `novelty_lr > 0.85` est plus robuste.
2. Aucune comparaison statistique avec les attaques connues (séparation des distributions).
   Un test t de Welch ou une AUC ROC entre `novelty_lr(known attacks)` vs `novelty_lr(UNKNOWN)`
   renforcerait la claim.
3. Le contrôle ne teste que **un seul** type d'anomalie inconnue. Le résultat est donc
   anecdotique sans un ensemble de N contrôles variés.

---

#### S7 — Absence de test sur la co-occurrence et chevauchement des périodes

Plusieurs injections se succèdent à quelques jours d'intervalle. Si une injection laisse des
artefacts résiduels dans les modèles (via le prior temporel SBN), les résultats d'une
injection suivante peuvent être biaisés. `evaluate_qualify_sbn.py` ne vérifie pas que les
périodes d'évaluation sont bien séparées temporellement.

Concernant les attaques réelles et injectées : l'injection SLOWLORIS_DOS (15-16 déc.) et
`NETWORK_OUTAGE_DEC1617` (16-17 déc.) sont séparées de seulement 6h30, ce qui peut contaminer
l'évaluation `--real` du NETWORK_OUTAGE par les résidus du prior temporel SLOWLORIS.

---

#### S8 — Absence d'intervalles de confiance sur les métriques

Macro recall 87% sur 8 attaques : quelle est la fiabilité statistique de cette estimation ?
Avec seulement 8 observations, un bootstrap (ex. : 1000 rééchantillonnages) produirait un
intervalle de confiance [65%, 98%] typique, qui change radicalement l'interprétation.
Pour un jury scientifique, rapporter des métriques sans CI sur des échantillons < 30 est
insuffisant.

---

#### S9 — Absence de matrice de confusion globale (cross-attaque)

`evaluate_injected()` affiche, pour chaque attaque, les types top-1 incorrects sous forme
de string (`wrong_str`). Il n'y a pas de **matrice de confusion agrégée** cross-attaques.
Pourtant, `qualify_anomaly_sbn.py` inclut une fonction `_evaluate_with_types()` qui produit
cette matrice. Il faudrait l'appeler depuis `evaluate_qualify_sbn.py` sur le sous-ensemble
des fenêtres injectées.

---

#### S10 — Recall injecté ≠ recall de détection global

`evaluate_injection_v2.py` mesure le recall global de **détection** (gate ouverte sur toute
la période de test). `evaluate_qualify_sbn.py` mesure le recall de détection **uniquement
pendant les périodes d'injection**. Ces deux métriques se complètent mais ne sont jamais
rapportées conjointement. La différence (fenêtres injectées vs toutes fenêtres de test)
détermine le taux de faux positifs — absent de ce fichier.

---

### 6.4 Anomalies manquantes dans le catalogue

Le tableau ci-dessous répertorie les types d'anomalies pertinents pour un **réseau académique
de type université fédérale** (UFRJ/RedeRio) et leur couverture dans le pipeline.

| Type d'anomalie | Dans SBN | Injecté | Réel | Détectable ? | Commentaire |
|---|---|---|---|---|---|
| UDP_FLOOD | ✓ | ✓ | ✓ | oui | volume+protocol_udp |
| SYN_FLOOD | ✓ | ✓ | — | oui | fin_ratio discriminateur |
| ICMP_FLOOD | ✓ | ✓ | — | oui | protocol_icmp+packet_size |
| DNS_AMP | ✓ | ✓ | — | oui | packet_size+udp |
| HTTP_FLOOD | ✓ | ✓ | — | oui | connections+tcp |
| SLOWLORIS | ✓ | ✓ | — | oui | fin_ratio clé |
| PORT_SCAN | ✓ | ✓ | — | oui | entropy+connections |
| DATA_EXFIL | ✓ | ✓ | — | partiel | signal faible (low intensity) |
| NETWORK_OUTAGE | ✓ | **✗** | ✓ | oui | **injection manquante** |
| BGP_HIJACK | ✓ | **✗** | **✗** | partiel | **non évalué du tout** |
| BOTNET_CC | ✓ | **✗** | **✗** | partiel | **non évalué du tout** |
| **NTP_AMP / SSDP_AMP** | ✗ | ✗ | ✗ | oui | même mécanisme que DNS_AMP, facteur ×500 |
| **BRUTE_FORCE (SSH/RDP)** | ✗ | ✗ | ✗ | oui | connections↑↑ + entropy_dst_port faible |
| **DNS_TUNNELING** | ✗ | ✗ | ✗ | partiel | UDP + packet_size anormal |
| **Memcached AMP** | ✗ | ✗ | ✗ | oui | UDP + packet_size extrême (×51000) |
| **Ransomware lateral** | ✗ | ✗ | ✗ | partiel | SMB/NetBIOS invisible en flow-level classique |
| **Crypto-jacking** | ✗ | ✗ | ✗ | difficile | signal faible, graduel, TCP persistant |
| **Légitime anormal (backup)** | ✗ | ✗ | ✗ | difficile | hors scope IDS mais important pour FP |

#### Recommandations prioritaires

**À ajouter au catalogue SBN et à l'évaluation :**

1. **NTP_AMP** — Amplification NTP (réponse monlist : ×556). Signature presque identique à
   DNS_AMP (UDP ↑, gros paquets, peu de sources). Réf. : Van Rijswijk-Deij et al. 2014 IMC.
   Ajouter en variante de DNS_AMP avec `packet_size=strong_anom` et `protocol_udp=strong_anom`.

2. **BRUTE_FORCE** — Particulièrement pertinent pour un campus universitaire (SSH exposé).
   Signature : `connections=strong_anom` (flood de connexions TCP courtes),
   `entropy=mod_safe` (destination unique : port 22/3389),
   `fin_ratio=mod_susp` (connexions reset rapidement).
   Réf. : Mirsky 2018 Kitsune ; MITRE ATT&CK T1110.

3. **DNS_TUNNELING** — Exfiltration via DNS (canal covert). Signature :
   `protocol_udp=mod_anom` (nombreuses requêtes DNS),
   `packet_size=mod_anom` (requêtes TXT de grande taille),
   `entropy=strong_anom` (noms de domaines encodés → entropie élevée).
   Réf. : Born & Gustafson 2010 ; MITRE ATT&CK T1071.004.
   **Distinct de DNS_AMP** : direction différente (requêtes sortantes, pas réponses entrantes).

---

### 6.5 Problèmes de forme / lisibilité

#### F1 — Pas de sortie fichier : reproductibilité zéro

Tous les résultats vont sur stdout. Pour un jury, la reproductibilité exige que les métriques
soient sauvegardées dans un fichier datestampé. Ajouter :
```python
out_csv = os.path.join(RESULTS_DIR, f"eval_qualify_{datetime.now():%Y%m%d_%H%M}.csv")
pd.DataFrame(known_rows).to_csv(out_csv, index=False)
```

#### F2 — Absence de récapitulatif dans `evaluate_real()`

`evaluate_injected()` affiche un bloc `[A]` et `[B]` de synthèse. `evaluate_real()` n'a aucun
récapitulatif global. Pour les 3 événements réels, un tableau synthétique analogue serait utile.

#### F3 — `wrong_str` peut dépasser la largeur de ligne (ligne 141)

```python
wrong_str = ', '.join(f"{t}({c})" for t, c in wrong.value_counts().items())
```
Pour des attaques avec beaucoup de types top-1 incorrects, cette chaîne peut être très longue
et casser l'alignement tabulaire. Limiter à `head(3)`.

#### F4 — `--injected` par défaut non explicite dans le code (lignes 249–250)

```python
parser.add_argument('--injected', action='store_true', default=False, ...)
parser.add_argument('--real',     action='store_true', default=False, ...)
```
Si ni `--injected` ni `--real` n'est spécifié, le code exécute `evaluate_injected()` (else branch).
Ce comportement est documenté dans la docstring mais pas dans le `--help` argparse.

#### F5 — `u_col` inutilisé dans `evaluate_real()` (ligne 187)

```python
def evaluate_real(df: pd.DataFrame, fmt: str):
    _, u_col = _get_col_prefix(fmt)
```
`u_col` est utilisé comme fallback pour `lr_col` si `novelty_lr` absent. Mais il n'est pas
utilisé pour afficher l'incertitude SBN (u_sbn) dans le bilan par événement. Cohérence à améliorer.

#### F6 — Docstring module ne mentionne pas les 3 modes effectifs

La docstring (lignes 1–15) mentionne `--injected` et `--real` mais pas :
- Que `BGP_HIJACK` et `BOTNET_CC` sont dans le catalogue mais jamais évalués
- Que le mode `--real` nécessite `REAL_ATTACKS` dans `config.py`
- Que `NETWORK_OUTAGE` est seulement évalué en mode `--real`

---

### 6.6 Problèmes structurels

#### ST1 — Duplication de `INJECTED_ATTACKS` entre ce fichier et `evaluate_qualify_injected.py`

La docstring ligne 38 indique "même catalogue que `evaluate_qualify_injected.py`". C'est une
violation du principe DRY (Don't Repeat Yourself). Si une attaque est ajoutée ou une date
modifiée dans un fichier, il faut penser à l'autre.

**Recommandation :** Déplacer `INJECTED_ATTACKS` dans `config.py` ou dans un module
`attack_catalog.py` et l'importer depuis les deux fichiers d'évaluation.

#### ST2 — `evaluate_injected()` et `evaluate_real()` ont des niveaux de rigueur asymétriques

`evaluate_injected()` :
- Calcule recall, précision, TTQ, novelty_lr, wrong_str, u_mean
- Agrège en macro recall/précision

`evaluate_real()` :
- Calcule recall, distribution top-1, TTQ, novelty_lr
- Pas d'agrégation globale
- Pas de `u_mean`

Pour un jury, la comparaison injection/réel exige la même rigueur métrique dans les deux cas.

#### ST3 — Pas de vérification d'existence de colonnes critiques

```python
df = pd.read_csv(csv_path, parse_dates=['timestamp'])
```
Le code ne vérifie pas que `gate_open`, `top1_type`, `novelty_lr` existent dans le CSV.
Si le CSV est produit par une ancienne version de `qualify_anomaly_sbn.py` sans ces colonnes,
le code crashe avec une `KeyError` opaque au lieu d'un message d'erreur explicite.

#### ST4 — Aucun mode `--both` pour exécuter injection + réel

Si on veut un rapport complet, il faut lancer le script deux fois et concaténer les sorties.
Un flag `--both` (ou absence de flag → tout) simplifierait l'usage.

---

## 7. Récapitulatif priorisation

| Priorité | ID | Nature | Effort |
|---|---|---|---|
| 🔴 Critique | §6.1 | NETWORK_OUTAGE absent de l'évaluation injectée | moyen |
| 🔴 Critique | B1 | Logique résolution CSV cassée (csv_path jamais None) | faible |
| 🔴 Critique | B2 | TTQ sans tri timestamp → valeur possiblement incorrecte | faible |
| 🟠 Important | S1 | BGP_HIJACK et BOTNET_CC non évalués | moyen |
| 🟠 Important | S2 | Recall/Précision mal nommés (DR vs precision de qualification) | doc |
| 🟠 Important | S3 | Macro-moyenne non pondérée — ajouter micro-moyenne | faible |
| 🟠 Important | S5 | Absence de courbe ROC pour novelty_lr | moyen |
| 🟠 Important | S9 | Matrice de confusion globale manquante | moyen |
| 🟠 Important | F1 | Aucune sortie fichier (reproductibilité) | faible |
| 🟡 Mineur | B3 | novelty_lr NaN non protégé dans evaluate_real | faible |
| 🟡 Mineur | B4 | `NOVELTY_CONTROLS` dead code | trivial |
| 🟡 Mineur | S4 | TTQ sans distribution ni comparaison TTD | moyen |
| 🟡 Mineur | S6 | Validation nouveauté : fraction > moyenne, test stat | moyen |
| 🟡 Mineur | S8 | Pas d'intervalles de confiance | moyen |
| 🟡 Mineur | F2–F6 | Forme/lisibilité | faible |
| 🔵 Structurel | ST1 | Duplication INJECTED_ATTACKS | moyen |
| 🔵 Structurel | ST3 | Pas de validation des colonnes CSV | faible |
| 📝 Catalogue | §6.4 | NTP_AMP, BRUTE_FORCE, DNS_TUNNELING à ajouter | fort |

---

## 8. Éléments à rédiger dans le rapport technique

### 8.1 Section « Protocole d'évaluation de la qualification »

- Définir formellement les métriques : DR (Detection Rate = recall fenêtre), PA (Precision
  de qualification = n_correct/n_detected), TTQ (Time-to-Qualify = fenêtres jusqu'à
  première classification correcte)
- Distinguer DR de `evaluate_injection_v2.py` (recall de détection global) et PA de ce module
- Justifier le choix du critère `top-1` (classification stricte) vs `top-k`
- Mentionner l'absence de micro-moyenne pondérée comme limite

### 8.2 Section « Catalogue des attaques testées »

Tableau complet pour le rapport :

| Type | Intensité | Durée | Fenêtres | Source bibliographique |
|---|---|---|---|---|
| UDP_FLOOD | extreme | 4h | 48 | Sharafaldin 2018 CIC-IDS2017 |
| SYN_FLOOD | extreme | 45 min | 9 | MITRE T1498.001 |
| PORT_SCAN | medium | 2h30 | 30 | MITRE T1046 |
| DATA_EXFIL | low | 6h | 72 | MITRE T1048 |
| HTTP_FLOOD | high | 1h30 | 18 | Sharafaldin 2018 |
| DNS_AMP | extreme | 3h | 36 | Rossow 2014 NDSS |
| SLOWLORIS | low | 8h | 96 | Mirsky 2018 Kitsune |
| ICMP_FLOOD | extreme | 30 min | 6 | Moustafa 2015 UNSW-NB15 |
| UNKNOWN_ANOMALY_CONTROL | high | 2h | 24 | (contrôle interne) |
| NETWORK_OUTAGE | — | 15 min | 3 | données réelles RedeRio Nov 2025 |
| NETWORK_OUTAGE | — | ~28h | ~336 | données réelles RedeRio Déc 2025 |

- Justifier les intensités (ex. : `extreme` = seuil au-delà de la plage de percentiles d'entraînement)
- Justifier les durées (ex. : `SLOWLORIS` long car attaque basse intensité — signal faible nécessite accumulation)

### 8.3 Section « NETWORK_OUTAGE : anomalie structurelle vs anomalie malveillante »

Point critique pour un jury : le pipeline doit clairement distinguer :
- **Anomalie malveillante** (attaque) : trafic excessif sur un ou plusieurs protocoles
- **Anomalie structurelle** (panne) : chute globale de tous les protocoles simultanément

La discrimination par SBN repose sur la signature `protocol_tcp/udp/icmp = strong_safe +
volume = strong_anom` pour NETWORK_OUTAGE. **Limites à documenter :**
1. La ternaire {Safe, Susp, Anom} ne capture pas la direction du résidu (surplus vs déficit).
   Un flood ICMP très intense pourrait, dans un cas extrême, saturer le réseau et présenter
   le même profil volumétrique qu'un outage.
2. La discrimination repose donc sur la **conjonction** volume_anom + protocoles_safe,
   qui est un signal robuste mais pas infaillible.
3. Extension recommandée : ajouter une colonne `residual_sign ∈ {+1, -1, 0}` par groupe
   (surplus vs déficit) comme mentionné dans le commentaire ligne 793 de `qualify_anomaly_sbn.py`.

### 8.4 Section « Signal de nouveauté : calibration et limites »

- Définir `novelty_lr = 1 / (max(L) / mean(L))` avec référence Good (1950)
- Rapporter la distribution empirique de `novelty_lr` sur les 8 attaques connues et
  sur `UNKNOWN_ANOMALY_CONTROL`
- Justifier ou recalibrer le seuil 0.85 via courbe ROC (actuellement sur signatures théoriques)
- Mettre en garde : sur données bruitées, les attaques connues peuvent avoir `novelty_lr`
  proche de 0.85 (noté dans `qualify_anomaly_sbn.py` ligne 689)

### 8.5 Section « Limites de l'évaluation »

- BGP_HIJACK et BOTNET_CC : dans le catalogue SBN mais non évalués (absence d'injection)
- NETWORK_OUTAGE : évalué uniquement sur 2 événements réels (pas d'injection synthétique contrôlée)
- Recall calculé uniquement pendant les périodes d'injection (pas de taux de faux positifs global)
- TTQ = première fenêtre correcte, pas la médiane ou le P95
- Taille des échantillons faible (8 types × N fenêtres/type) : pas d'intervalles de confiance
- Prior temporel désactivé par défaut (`SBN_TEMPORAL_ENABLED = False`) :
  les métriques de TTQ ne bénéficient pas du lissage temporel Markovien

---

## 9. Corrections à appliquer (priorité critique + importante)

| ID | Action | Fichier |
|---|---|---|
| B1 | `csv_path = None` avant la boucle candidates ; guard `if csv_path is None` réactivée | `evaluate_qualify_sbn.py` |
| B2 | `.sort_values('timestamp')` avant `iloc[0]` pour TTQ (injected + real) | `evaluate_qualify_sbn.py` |
| B3 | Déplacer calcul `lr_mean` dans bloc `if n_detected > 0:` dans `evaluate_real()` | `evaluate_qualify_sbn.py` |
| B4 | Supprimer `NOVELTY_CONTROLS` ou utiliser comme source de vérité unique | `evaluate_qualify_sbn.py` |
| §6.1 | Ajouter `NETWORK_OUTAGE_SYNTHETIC` dans `INJECTED_ATTACKS` | `evaluate_qualify_sbn.py` / `config.py` |
| S1 | Ajouter `BGP_HIJACK_SYNTHETIC` et `BOTNET_CC_SYNTHETIC` (ou documenter absence) | idem |
| S3 | Ajouter micro-moyenne pondérée en complément de la macro-moyenne | `evaluate_qualify_sbn.py` |
| F1 | Sauvegarder `known_rows` et `novelty_rows` en CSV datestampé | `evaluate_qualify_sbn.py` |
| ST1 | Déplacer `INJECTED_ATTACKS` dans `config.py` comme `INJECTED_ATTACK_CATALOG` | `config.py` |
| ST3 | Vérification d'existence des colonnes critiques après lecture CSV | `evaluate_qualify_sbn.py` |

---

*Document généré lors de la revue de code du 2026-04-13.*
