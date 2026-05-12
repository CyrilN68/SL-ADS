# Revue scientifique et technique — `inject_at_evidence_level.py`

> Révision du : 2026-04-11
> Contexte : préparation à l'évaluation par un jury scientifique
> Fichier principal : `inject_at_evidence_level.py`
> Dépendances directes : `config.py`, `paths.py`
> Dépendances amont : `compute_evidence_v2.py` (produit le CSV d'entrée)
> Dépendances aval : `compute_opinions_v3.py` (consomme le CSV de sortie)

---

## 1. Rôle du module dans la chaîne SL-ADS

`inject_at_evidence_level.py` est un module **optionnel** intercalé entre
`compute_evidence_v2.py` et `compute_opinions_v3.py`. Son rôle est de **simuler
des attaques synthétiques directement au niveau des preuves brutes** (CSV
`evidence_*.csv`) afin d'évaluer la détectabilité et la qualifiabilité du
système SL-ADS sur un dataset (RedeRio) qui ne contient pas d'attaques réelles
étiquetées.

**Positionnement dans la chaîne :**

```
train_v10.py → compute_evidence_v2.py
                         ↓
             evidence_{VERSION}.csv
                         ↓  [inject_at_evidence_level.py — optionnel]
             evidence_{VERSION}_attacks.csv
                         ↓
             compute_opinions_v3.py
                         ↓
             detection_results_INJECTED.csv
```

**Résultat :** le module substitue les valeurs P/S/N des fenêtres temporelles
correspondant à une attaque par des valeurs de signature précalibrées, puis
sauvegarde le fichier evidence modifié sous un nom distinct. `compute_opinions_v3`
choisit automatiquement la version injectée ou non selon la disponibilité du
fichier.

---

## 2. Fonctions et outils employés

| Fonctionnalité | Implémentation | Référence |
|---|---|---|
| Catalogue d'attaques | Dictionnaire Python inline + override `CONFIG['ATTACK_CATALOG']` | Signatures issues de CIC-IDS2017, UNSW-NB15, etc. |
| Profil de montée (ramp) | `make_ramp(n_windows, ramp_frac)` — trapézoïdal | CIC-IDS2017 (profil d'injection synthétique) |
| Injection fenêtre par fenêtre | `inject_attack_into_evidence()` — écrasement ciblé des colonnes _P/_S/_N | — |
| Vérification de plage | Comparaison `start/end` vs `csv_start/csv_end` | — |
| Génération du calendrier | `generate_attack_schedule_txt()` — fichier TXT lisible | — |
| Centralisations des chemins | `paths.get_version_names()`, `paths.get_results_dir()` | — |
| Copie sans injection | `shutil.copy2()` si `ATTACK_CATALOG == []` | — |

---

## 3. Hypothèses posées (explicites et implicites)

### 3.1 Hypothèses sur la représentativité des signatures

- **H1 – Signatures calibrées à dire d'expert** : les triplets `(ev_attack,
  ev_suspect, ev_normal)` de chaque métrique sont fixés manuellement en s'appuyant
  sur la littérature (CIC-IDS2017, UNSW-NB15, Kitsune, etc.) et sur la connaissance
  du comportement de chaque métrique dans le dataset RedeRio.  
  *Limite :* aucune validation quantitative (ex. calibration sur données réelles
  d'attaque) ne confirme que ces valeurs sont représentatives.

- **H2 – Les signatures sont discrétisées en 4 niveaux d'intensité** : `low`,
  `medium`, `high`, `extreme`. Ces niveaux ne sont documentés que dans le résumé
  textuel (`ATTACK_SCHEDULE.txt`). Ils n'influencent pas la logique de l'injection
  (seul `ramp_frac` est utilisé dans le code).

- **H3 – Les métriques absentes du CSV sont ignorées silencieusement** : si une
  métrique de la signature n'existe pas dans le CSV d'evidence (car non entraînée
  dans ce run), l'injection continue sans erreur. Les compteurs `missing` et
  `skipped` sont affichés mais non traçables après coup.

### 3.2 Hypothèses sur la validité des valeurs injectées

- **H4 – Les valeurs injectées ne respectent pas l'invariant P+S+N = WINDOW_SIZE** :
  `compute_evidence_v2` garantit `P+S+N = WINDOW_SIZE = 10` pour toute fenêtre
  naturelle. Les signatures peuvent avoir une somme bien supérieure (ex :
  `(40.0, 8.0, 0.5)` → P+S+N = 48.5 pour `prophet_bytes` en UDP_FLOOD). La
  bijection SL reste formellement valide (elle mappe tout r ≥ 0), mais produit
  des opinions avec une certitude (`1-u`) structurellement plus élevée que celle
  qui pourrait être obtenue naturellement. **Ce choix doit être explicitement
  documenté et justifié.**

- **H5 – Le profil de ramp multiplie P, S et N par le même facteur** : au pic,
  P+S+N = signature_sum. Au démarrage, P+S+N → 0 (toute l'évidence tend vers 0).
  Cela produit une progression de l'incertitude vers la certitude, plutôt qu'une
  véritable montée de normalité vers attaque. La fenêtre de démarrage exprime
  "presque aucune preuve" et non "trafic normal avant attaque".

- **H6 – Chaque attaque s'applique à toutes les fenêtres de la plage horaire**,
  y compris les fenêtres qui tomberaient sur un creux naturel du trafic (nuit,
  weekend). Aucun ajustement n'est fait pour le contexte temporel.

### 3.3 Hypothèses sur la compatibilité aval

- **H7 – Le nom du fichier injecté suit la convention `{VERSION_NAME}_attacks`** :
  `compute_opinions_v3.py` cherche exactement `evidence_{VERSION_NAME}_attacks.csv`.
  `inject` produit `evidence_{VERSION_NAME_MODIF}.csv`. La compatibilité est garantie
  **uniquement si** `VERSION_NAME_MODIF == f"{VERSION_NAME}_attacks"`, ce qui est
  le cas dans `config.py` (hardcodé) mais n'est pas validé programmatiquement.

- **H8 – Le CSV raw_data n'est pas modifié par l'injection** : les graphiques de
  `compute_opinions_v3` affichent les résidus réels (non injectés) pour les fenêtres
  d'attaque. La courbe d'opinion (basée sur les preuves injectées) sera cohérente
  avec une attaque, mais la courbe des données brutes montrera du trafic normal.
  C'est délibéré mais non documenté.

---

## 4. Catalogue d'attaques

| # | Nom | Type | Durée | Intensité | Début | Métriques signalées |
|---|-----|------|-------|-----------|-------|---------------------|
| 0 | UNKNOWN_ANOMALY_CONTROL | Contrôle hors-catalogue | 2h | high | 2025-12-20 10:00 | Aucun groupe dominant |
| 1 | UDP_FLOOD_DDOS | DDoS volumétrique UDP | 4h | extreme | 2025-11-16 14:00 | bytes, packets, udp ↑↑↑ |
| 2 | SYN_FLOOD_DDOS | DDoS TCP SYN | 0.75h | extreme | 2025-11-21 02:30 | syn ↑↑↑, fin/syn → 0 |
| 3 | AGGRESSIVE_PORT_SCAN | Reconnaissance | 2.5h | medium | 2025-11-28 10:15 | flows, entropy_dst_port ↑↑↑ |
| 4 | DATA_EXFILTRATION_SLOW | Exfiltration furtive | 6h | low | 2025-12-02 23:00 | bytes↑, flows↓ |
| 5 | HTTP_FLOOD_L7_DDOS | DDoS applicatif | 1.5h | high | 2025-12-07 16:30 | flows, syn, fin ↑↑↑ |
| 6 | DNS_AMPLIFICATION | DDoS réflexion DNS | 3h | extreme | 2025-12-11 08:00 | bytes, avg_pkt_size, udp ↑↑↑ |
| 7 | SLOWLORIS_DOS | DoS lent | 8h | low | 2025-12-15 22:00 | flows↑, bytes↓, fin/syn→0 |
| 8 | ICMP_FLOOD_BURST | DoS volumétrique ICMP | 0.5h | extreme | 2025-12-18 11:30 | icmp ↑↑↑, packets↑↑↑ |

### 4.1 Références des signatures

| Attaque | Source principale | Paramètre discriminant |
|---|---|---|
| UDP Flood | Sharafaldin et al. 2018, CIC-IDS2017 | `prophet_udp` ↑↑↑, `prophet_icmp` Safe |
| SYN Flood | Roesch 1999 (Snort) ; MITRE T1498.001 | `reconst_fin_from_syn` (fin/syn→0) |
| Port Scan | Mirsky et al. 2018 (Kitsune) ; MITRE T1046 | `prophet_entropy_dst_port` ↑↑↑ |
| Exfiltration | MITRE T1048 ; Jiang et al. 2018 | `prophet_flows` ↓, `prophet_avg_pkt_size` ↑ |
| HTTP Flood | Sharafaldin 2018 (Hulk/GoldenEye) | fin/syn ≈ 1 (vs SYN Flood fin/syn→0) |
| DNS Amplification | RFC 7534 ; Cloudflare 2024 | `prophet_avg_pkt_size` ↑↑↑ |
| Slowloris | Hansen 2009 ; CIC-IDS2017 | `prophet_bytes` ↓↓, `prophet_flows` ↑↑ |
| ICMP Flood | Moustafa & Slay 2015 (UNSW-NB15) | `prophet_icmp` ↑↑↑, `prophet_udp` Safe |

---

## 5. Paramètres de configuration impliqués

| Paramètre CONFIG | Rôle | Valeur typique |
|---|---|---|
| `ATTACK_CATALOG` | `None` = catalogue RedeRio intégré ; `[]` = désactivé | `None` (non clé dans config = RedeRio) |
| `INJECTION_SKIP_N_DOMINANT` | Si True : ignore métriques où le signal normal domine | `False` |
| `VERSION_NAME` | Nom du CSV d'entrée | `"trained_models_v9_v9_v4s"` |
| `VERSION_NAME_MODIF` | Suffixe du CSV de sortie | `"trained_models_v9_v9_v4s_attacks"` |
| `RESULTS_DIR` | Répertoire de résultats | `"../results/resultats_{VERSION_NAME}"` |

---

## 6. Sorties produites

| Fichier | Contenu | Usage |
|---|---|---|
| `evidence_{VERSION_NAME_MODIF}.csv` | CSV d'evidence avec fenêtres d'attaque modifiées | Entrée `compute_opinions_v3.py` |
| `ATTACK_SCHEDULE.txt` | Calendrier humainement lisible des attaques + signatures | Documentation, rapport |

**Absent (lacune) :** aucun fichier de vérité terrain (`ground_truth_*.csv`)
n'est produit. L'évaluation quantitative (F1, précision, rappel) doit être
reconstruite depuis `ATTACK_SCHEDULE.txt` par les scripts aval
(`evaluate_injection_v2.py`, etc.).

---

## 7. Problèmes identifiés

### 7.1 Problèmes scientifiques — critiques

#### SC1 — Invariant P+S+N = WINDOW_SIZE systématiquement violé [CRITIQUE]

**Description :**
`compute_evidence_v2.py` garantit que pour toute fenêtre naturelle :
```
P + S + N = WINDOW_SIZE  (= 10 pour RedeRio)
```
C'est l'invariant fondamental du vecteur de preuves `r` (Jøsang 2016, Def. 3.9).

Les signatures injectées peuvent avoir des sommes bien supérieures.
Exemple : `prophet_bytes` dans `UDP_FLOOD_DDOS` → `(40.0, 8.0, 0.5)` → P+S+N = 48.5.

La bijection `evidence_to_opinion(r, W=3)` reste formellement correcte pour
tout `r ≥ 0`, mais elle produit une opinion avec `u = W / (sum(r) + W) = 3/51.5 = 0.058`,
soit une certitude bien supérieure à celle d'une fenêtre naturelle d'attaque
(u = 3/13 = 0.23 si P=0, S=0, N=10). Les opinions injectées sont donc
**structurellement plus certaines que toute opinion naturellement atteignable**
par le pipeline.

**Impact :** si les seuils de décision sont calibrés sur du trafic naturel,
les attaques injectées franchissent ces seuils avec une marge artificielle. Les
métriques de détection (F1, recall) sont susceptibles d'être gonflées.

**Propositions :**

Option A — Normaliser chaque signature au WINDOW_SIZE :
```python
def _normalize_signature(ev_atk, ev_sus, ev_nor, window_size):
    total = ev_atk + ev_sus + ev_nor
    if total <= 0:
        return (0.0, 0.0, float(window_size))
    scale = window_size / total
    return (ev_atk * scale, ev_sus * scale, ev_nor * scale)
```
Appliqué dans `inject_attack_into_evidence` avant le ramp.
*Avantage :* l'opinion injectée a la même plage de certitude qu'une fenêtre naturelle.
*Inconvénient :* les signatures doivent être revalidées après normalisation.

Option B — Documenter explicitement le choix de surpasser WINDOW_SIZE :
Ajouter dans la docstring de `inject_attack_into_evidence` et dans l'introduction
du module un paragraphe expliquant que les valeurs injectées représentent une
« intensité d'attaque » relative, pas un vecteur de preuves calibré, et justifier
pourquoi cela reste acceptable pour l'évaluation qualitative du système.

**Pour le rapport :** cette décision doit figurer explicitement dans la section
Méthode. Un jury scientifique qui voit P+S+N = 48.5 avec W = 3 et WINDOW_SIZE = 10
sera confus si ce choix n'est pas justifié.

---

#### SC2 — Profil de ramp : sémantique d'interpolation incorrecte [IMPORTANT]

**Description :**
Le profil `make_ramp` multiplie P, S et N par un facteur `scale ∈ [0, 1]`. À
`scale = 0.5`, l'opinion résultante n'est pas « mi-chemin entre trafic normal et
attaque », mais « trafic avec moitié de l'évidence d'attaque ». La différence
est non négligeable :

- `scale=0.5` → P=0.25, S=4.0, N=20 → P+S+N=24.25 → u=0.110 (certitude partielle)
- Interpolation correcte entre (P_norm=10,S=0,N=0) et (P=0.5,S=8,N=40) à 50% →
  P=5.25, S=4, N=20 → u=3/(29.25+3)=0.093 + signal encore dominé par normale

**Proposition :**
Pour une interpolation sémantiquement correcte entre trafic normal et attaque :
```python
def _interpolate_signature(alpha, normal_sig, attack_sig, window_size):
    """alpha=0 → trafic normal ; alpha=1 → attaque à pleine intensité."""
    P_n, S_n, N_n = window_size, 0.0, 0.0  # état normal
    P_a, S_a, N_a = attack_sig
    return (
        (1 - alpha) * P_n + alpha * P_a,
        (1 - alpha) * S_n + alpha * S_a,
        (1 - alpha) * N_n + alpha * N_a,
    )
```
Cela produit un démarrage depuis un vrai état normal et une montée progressive
vers l'état d'attaque.

---

#### SC3 — Aucune vérification d'absence de recouvrement temporel entre attaques [IMPORTANT]

**Description :**
Le catalogue actuel ne présente pas de recouvrement, mais rien dans le code
n'empêche d'en introduire un. Si deux attaques se recouvrent, la seconde injection
écrase silencieusement les valeurs de la première. Le résultat est une signature
hybride non documentée.

**Correction :**
```python
def _check_no_overlap(attacks):
    intervals = []
    for a in attacks:
        s = pd.Timestamp(a['start'])
        e = s + pd.Timedelta(hours=a['duration_h'])
        for (s2, e2, name2) in intervals:
            if s < e2 and e > s2:
                raise ValueError(
                    f"Recouvrement temporel entre '{a['name']}' et '{name2}'"
                )
        intervals.append((s, e, a['name']))
```
À appeler dans `main()` avant l'injection.

---

#### SC4 — Absence de fichier de vérité terrain (ground truth) [IMPORTANT]

**Description :**
Le CSV de sortie (`evidence_{VERSION}_attacks.csv`) ne contient aucune colonne
indiquant les périodes d'attaque. L'évaluation (F1, recall) doit reconstruire
la vérité terrain depuis `ATTACK_SCHEDULE.txt`. Cette reconstruction :
- est sujette à des erreurs de correspondance timestamps ;
- n'est pas reproductible automatiquement ;
- ne couvre pas les cas partiels (fenêtre en cours de ramp).

**Correction :**
Ajouter une colonne `injection_label` dans le CSV de sortie :
```python
df['injection_label'] = 'normal'
for atk in valid_attacks:
    s = pd.Timestamp(atk['start'])
    e = s + pd.Timedelta(hours=atk['duration_h'])
    mask = (df['timestamp'] >= s) & (df['timestamp'] < e)
    df.loc[mask, 'injection_label'] = atk['name']
```
Et optionnellement une colonne `injection_ramp_alpha` (0 = normal, 1 = attaque
à pleine intensité) pour évaluer la détection en fonction de l'intensité.

---

#### SC5 — Signatures à "signal Safe actif" : hypothèse non vérifiée [MINEUR]

**Description :**
Plusieurs signatures injectent intentionnellement des valeurs de signal Safe élevé
sur certaines métriques pour « enseigner » au classifieur que ce n'est pas un autre
type d'attaque (ex : `prophet_icmp` → (0.0, 1.0, 9.0) pour `UDP_FLOOD_DDOS`
indique « ICMP normal, pas ICMP flood »). Cette stratégie est valide en principe,
mais suppose que la qualification aval (`qualify_anomaly_sbn.py`) sait exploiter
ces signaux discriminants. Ce n'est pas vérifié formellement.

**Pour le rapport :** décrire ce mécanisme de « signal discriminant actif » comme
une décision de conception, et indiquer s'il améliore effectivement la précision
de la qualification lors des ablations.

---

### 7.2 Bugs et erreurs silencieuses

#### B1 — Fallback hardcodé dans main() [MINEUR]

```python
version_modif = CONFIG.get('VERSION_NAME_MODIF', 'v15_full_v3_attacks')
```
**Problème :** si `VERSION_NAME_MODIF` est absent de CONFIG, la chaîne de secours
`'v15_full_v3_attacks'` ne correspond à aucun fichier réel et est trompeuse.

**Correction :** utiliser directement `VERSION_NAME_MODIF` déjà extrait en tête
de fichier :
```python
print(f"   Dans compute_opinions_v3.py, vérifier VERSION_NAME_MODIF = '{VERSION_NAME_MODIF}'")
```

---

#### B2 — `import shutil` dans le corps de la fonction [MINEUR]

```python
import shutil  # ligne 713, à l'intérieur de main()
os.makedirs(os.path.dirname(OUTPUT_EVIDENCE_CSV), exist_ok=True)
shutil.copy2(INPUT_EVIDENCE_CSV, OUTPUT_EVIDENCE_CSV)
```
**Correction :** déplacer `import shutil` en tête de fichier.

---

#### B3 — Chaînes hardcodées dans generate_attack_schedule_txt [MINEUR]

- Ligne 638 : `"Dataset    : RedeRio — Ilha do Fundão, Oct-Dec 2025"` → ne s'adapte
  pas au changement de dataset.
- Ligne 641 : `"Métriques  : 17 (12 Prophet + 5 RANSAC)"` → ne reflète pas le
  nombre réel de métriques actives.
- Ligne 734 : `"(attendu : 15 métriques × 3 = 45 + timestamp + ...)"` → idem.

**Correction :**
```python
_n_prophet_sig = len(CONFIG.get('ACTIVE_METRICS', []))
_n_reconst_sig = len(CONFIG.get('RECONST_RULES', []))
lines.append(f"Dataset    : {CONFIG.get('ACTIVE_DATASET', 'RedeRio')}")
lines.append(f"Métriques  : {_n_prophet_sig + _n_reconst_sig} "
             f"({_n_prophet_sig} Prophet + {_n_reconst_sig} RANSAC)")
```

---

#### B4 — Aucune validation que les valeurs de signature sont ≥ 0 [MINEUR]

Une signature contenant une valeur négative (erreur de saisie) produirait une
évidence négative et une opinion invalide dans la bijection SL.

**Correction :** ajouter dans `inject_attack_into_evidence` :
```python
for k, (a, s, n) in attack['signature'].items():
    if a < 0 or s < 0 or n < 0:
        raise ValueError(f"[{attack['name']}][{k}] valeurs de signature négatives : "
                         f"({a}, {s}, {n})")
```

---

#### B5 — Double ligne vide superflue (ligne 68-69) [TRIVIAL]

```python
OUTPUT_EVIDENCE_CSV = os.path.join(RESULTS_DIR, f"evidence_{VERSION_NAME_MODIF}.csv")


# ====...
```
Une seule ligne vide suffit (PEP8 E303).

---

### 7.3 Problèmes de compatibilité avec compute_opinions_v3.py

#### CP1 — Dépendance implicite entre VERSION_NAME_MODIF et le fichier attendu [IMPORTANT]

**Mécanisme actuel :**
`inject` produit : `evidence_{VERSION_NAME_MODIF}.csv`
`compute_opinions` cherche : `evidence_{VERSION_NAME}_attacks.csv`

Ces deux noms coïncident **uniquement si** :
```
VERSION_NAME_MODIF == f"{VERSION_NAME}_attacks"
```
Cette égalité est respectée dans `config.py` mais n'est pas vérifiée par assertion
dans aucun des deux scripts.

**Correction :** Ajouter dans `inject_at_evidence_level.py` :
```python
_expected_by_opinions = f"{VERSION_NAME}_attacks"
if VERSION_NAME_MODIF != _expected_by_opinions:
    print(f"⚠️  VERSION_NAME_MODIF='{VERSION_NAME_MODIF}' ≠ '{_expected_by_opinions}' "
          f"attendu par compute_opinions_v3. Vérifier config.py.")
```

---

#### CP2 — Les colonnes du CSV injecté n'incluent pas clean_key pour les métriques avec "->" [MINEUR]

`inject_attack_into_evidence` utilise `metric_key` brut (depuis le catalogue) pour
construire les noms de colonnes `col_P`, `col_S`, `col_N`. Si une métrique avait
un nom contenant `"->"`, le nom de colonne écrit différerait de ce que `compute_opinions`
attend (qui lit `clean_key = key.replace("->", "_to_")`).

**Situation actuelle :** aucune métrique active ne contient `"->"`, donc pas d'impact.
**Recommandation :** utiliser `metric_key.replace("->", "_to_")` dans `inject_attack_into_evidence`
pour se mettre en conformité défensive avec `compute_evidence_v2`.

---

#### CP3 — Le CSV raw_data n'est pas synchronisé avec l'injection [DOCUMENTAIRE]

Après injection, les fenêtres d'attaque du CSV evidence contiennent les preuves
injectées, mais le CSV `raw_data_{VERSION}.csv` contient toujours les résidus
originaux (trafic normal). Dans `compute_opinions_v3`, la section "Données brutes"
des graphiques montrera donc du trafic normal pour les fenêtres d'attaque, alors
que la section "Opinion" montrera un signal d'attaque.

**Pas un bug** (le raw_data est un artefact de debug du modèle, pas du trafic
injecté), mais ce comportement doit être documenté pour éviter toute confusion
lors de la visualisation des résultats.

---

#### CP4 — compute_opinions fonctionne correctement sans injection [VALIDÉ ✅]

La logique de sélection dans `compute_opinions_v3.py` (lignes 53–70) est correcte :
```python
_has_injection = CONFIG.get("ATTACK_CATALOG") is None  # True pour RedeRio
_attacks_exists = os.path.exists(_ev_attacks_path)

if _has_injection and _attacks_exists:   → fichier injecté ✅
elif _has_injection and not _attacks_exists: → fallback original + warning ✅
else:                                       → pas d'injection ✅
```
Ce mécanisme garantit que `compute_opinions` fonctionne dans tous les cas :
- Avec injection (fichier `_attacks` présent)
- Sans injection mais dataset nécessitant une injection (fallback avec avertissement)
- Dataset sans injection (CESNET, METR-LA)

---

### 7.4 Problèmes de forme et lisibilité

#### F1 — Module docstring obsolète [MINEUR]
Le docstring mentionne des chemins `v9_v5_v4s` et est annoté `VERSION v2 (patch
nouvelles métriques)`. Pour un audit scientifique, la version et les noms de
chemin doivent être cohérents avec la version courante du pipeline.

#### F2 — PIPELINE dans le docstring non dynamique [MINEUR]
```
evidence_v9_v5_v4s.csv → ... → detection_results_INJECTED.csv
```
Ces noms doivent être dynamiques ou supprimés du docstring.

#### F3 — Convention (ev_attack, ev_suspect, ev_normal) vs (P, S, N) non unifiée
Le code utilise alternativement `(ev_attack_raw, ev_suspect_raw, ev_normal_raw)` et
`(P, S, N)` pour désigner la même chose. Choisir une convention et l'appliquer partout.

#### F4 — Note "ASYMMETRIC_THRESHOLD_METRICS" dans le docstring est devenue stale
La note concernant les nouvelles métriques à ajouter dans `ASYMMETRIC_THRESHOLD_METRICS`
devrait indiquer si cela a été fait ou si c'est encore une TODO active.

---

### 7.5 Problèmes structurels

#### ST1 — Le catalogue est un bloc de données Python, pas un fichier de configuration [STRUCTUREL]
Pour un jury scientifique, les paramètres expérimentaux (dates, durées, intensités,
signatures) devraient idéalement résider dans un fichier externe (JSON, YAML) versionné
indépendamment du code. Cela permettrait de modifier les attaques sans toucher au code
Python et de tracer les changements d'expériences.

**Recommandation :** exporter le catalogue vers `attack_catalog.json` et charger
avec `json.load()`. Le code d'injection reste inchangé.

#### ST2 — Pas de validation de schéma des entrées du catalogue [STRUCTUREL]
Chaque entrée du catalogue doit avoir : `name`, `type`, `start`, `duration_h`,
`intensity`, `ramp_frac`, `signature`. Une clé manquante provoquerait une
`KeyError` sans message d'erreur clair.

**Correction (validation minimale) :**
```python
REQUIRED_KEYS = {'name', 'type', 'start', 'duration_h', 'ramp_frac', 'signature'}
for atk in ATTACK_CATALOG:
    missing = REQUIRED_KEYS - set(atk.keys())
    if missing:
        raise ValueError(f"Entrée catalogue incomplète : {missing} manquants dans '{atk.get('name','?')}'")
```

#### ST3 — `make_ramp` ne garantit pas `len(profile) == n_windows` pour n_windows=0
Si `n_windows=0` (aucune fenêtre dans la plage), `np.ones(0)` est valide mais
la boucle suivante est silencieuse. Le `if not idxs: return 0` en amont protège,
mais `make_ramp(0, ...)` reste une entrée inattendue.

---

## 8. Récapitulatif priorisation

| Statut | ID | Nature | Résolution |
|---|---|---|---|
| ✅ Corrigé | SC1 | Invariant P+S+N ≠ WINDOW_SIZE | `_normalize_signature()` — P+S+N = WINDOW_SIZE maintenu |
| ✅ Corrigé | SC2 | Sémantique du profil de ramp incorrecte | Interpolation linéaire normal↔attaque |
| ✅ Corrigé | SC3 | Absence de vérification de recouvrement temporel | `_check_no_overlap()` ajoutée |
| ✅ Corrigé | SC4 | Absence de fichier ground truth en sortie | Colonnes `injection_label` + `injection_ramp_alpha` |
| ✅ Corrigé | CP1 | Dépendance implicite VERSION_NAME_MODIF ↔ _attacks | Assertion de cohérence dans `main()` |
| ✅ Corrigé | CP2 | Nommage colonnes sans clean_key | `clean_metric_key = metric_key.replace("->","_to_")` |
| ✅ Corrigé | B1 | Fallback hardcodé `'v15_full_v3_attacks'` | `VERSION_NAME_MODIF` utilisé directement |
| ✅ Corrigé | B2 | `import shutil` dans le corps de fonction | Déplacé en tête de fichier |
| ✅ Corrigé | B3 | Chaînes hardcodées dans generate_attack_schedule_txt | Dynamique depuis CONFIG |
| ✅ Corrigé | B4 | Valeurs de signature négatives non validées | `_validate_catalog()` + erreur explicite |
| ✅ Corrigé | ST2 | Schéma des entrées catalogue non validé | `_validate_catalog()` : clés requises + noms dupliqués |
| 📝 Rapport | SC5 | Signal Safe actif : hypothèse non vérifiée | À documenter dans le rapport (§ 9.3) |
| 🔵 Non traité | ST1 | Catalogue en dur dans le code (vs JSON externe) | Choix de conception conservé |
| 🔵 Non traité | CP3 | raw_data non synchronisé après injection | Comportement documenté, délibéré |

---

## 9. Éléments à rédiger dans le rapport technique

### 9.1 Section « Génération de vérité terrain synthétique »

- Justifier le choix de l'injection au niveau des **preuves** plutôt qu'au niveau
  des données brutes (évite la rétro-propagation à travers Prophet/RANSAC, permet
  de contrôler précisément le signal SL injecté, indépendamment des résidus réels).
- Documenter la **normalisation des signatures** : les poids bruts du catalogue sont
  des valeurs relatives guidées par la littérature ; `_normalize_signature()` les
  ramène à P+S+N = WINDOW_SIZE avant injection, garantissant que les opinions injectées
  ont la même plage de certitude `u = W/(W + WINDOW_SIZE)` que toute fenêtre naturelle
  (Jøsang 2016, Def. 3.9).
- Documenter le **profil de ramp par interpolation linéaire** : chaque triplet injecté
  est `(P, S, N) = ((1−α)·W + α·P_n, α·S_n, α·N_n)`, avec α ∈ [0,1] donné par le
  profil trapézoïdal (inspiration CIC-IDS2017). Propriété : P+S+N = WINDOW_SIZE ∀α.
  Sémantique : α=0 = trafic entièrement normal (certitude Safe) ; α=1 = attaque à
  pleine intensité.

### 9.2 Section « Catalogue d'attaques »

- Tableau synthétique (Tab. X) : nom, MITRE ATT&CK, source de la signature,
  durée, intensité.
- Justifier la diversité des 9 types : couvrir les dimensions volumétrique,
  protocolaire, comportementale et temporelle.
- Indiquer comment les signatures ont été fixées (choix expert guidé par la
  littérature + validation visuelle des opinions produites).
- Préciser que le catalogue est extensible et paramétrable depuis `config.py`
  (`ATTACK_CATALOG = [...]` pour un dataset custom).

### 9.3 Section « Mécanisme de signal discriminant actif »

- Expliquer que certaines métriques reçoivent un signal Safe actif pour renforcer
  la discrimination entre types d'attaque similaires (ex : ICMP=Safe pendant
  UDP_FLOOD discrimine UDP_FLOOD de ICMP_FLOOD).
- Lier à la qualification (`qualify_anomaly_sbn.py`) : le système aval peut
  exploiter ces sous-signaux pour identifier le vecteur d'attaque.

### 9.4 Section « Validité de l'évaluation par injection »

- Discuter les biais potentiels : les seuils de décision ont été calibrés sur
  du trafic naturel. Grâce à la normalisation (P+S+N = WINDOW_SIZE), les opinions
  injectées ont la même plage de certitude que les fenêtres naturelles : les
  métriques de détection ne sont pas artificiellement gonflées par une sur-certitude.
- Comparer avec les méthodes alternatives : injection au niveau des données brutes
  (Mirsky 2018 Kitsune), datasets synthétiques complets (CIC-IDS2017), labels
  réels partiels (METR-LA).
- Mentionner que les métriques de détection obtenues sur le dataset injecté
  constituent une **borne supérieure** de performance : les attaques réelles seront
  moins nettes que les signatures synthétiques.

### 9.5 Section « Reproductibilité »

- Indiquer que le catalogue est versionné avec le code (fichier Python ou JSON).
- Mentionner que `ATTACK_SCHEDULE.txt` est généré automatiquement et contient
  les informations nécessaires pour reproduire l'évaluation.
- Documenter les paramètres qui changent le résultat : `INJECTION_SKIP_N_DOMINANT`,
  `ramp_frac`, les valeurs du catalogue.

---

## 10. Vérification de compatibilité avec compute_opinions_v3.py

| Scénario | Résultat | Commentaire |
|---|---|---|
| RedeRio + inject lancé | ✅ OK | `_attacks_exists=True` → fichier injecté chargé |
| RedeRio + inject NON lancé | ⚠️ Warning + fallback | Evidence originale utilisée, warning affiché |
| CESNET/METR-LA/GECCO | ✅ OK | `ATTACK_CATALOG=[]` → `_has_injection=False` → evidence originale |
| VERSION_NAME_MODIF ≠ VERSION_NAME+"_attacks" | ❌ Fichier non trouvé | Dépendance implicite non vérifiée |
| Métriques injectées absentes du modèle | ✅ Ignorées silencieusement | Log des métriques manquantes affiché |
| raw_data après injection | ⚠️ Non synchronisé | Graphiques raw_data montrent trafic normal sur fenêtres d'attaque |

**Conclusion compatibilité :** compute_opinions fonctionne correctement avec et sans
injection dans tous les cas standards. La seule fragilité est la dépendance implicite
de nommage (CP1), qui est actuellement respectée par la configuration.

---

---

## 11. Corrections appliquées (2026-04-11)

| ID | Action |
|---|---|
| SC1 | Ajout de `_normalize_signature()` — chaque signature normalisée à P+S+N = WINDOW_SIZE avant injection |
| SC2 | Ramp par interpolation linéaire : `P=(1−α)·W + α·ev_n`, `S=α·ev_s`, `N=α·ev_n` — invariant maintenu à tout α |
| SC3 | Ajout de `_check_no_overlap()` — appelée avant injection dans `main()` |
| SC4 | Colonnes `injection_label` et `injection_ramp_alpha` ajoutées au CSV de sortie |
| B1 | `version_modif` fallback supprimé → `VERSION_NAME_MODIF` utilisé directement |
| B2 | `import shutil` déplacé en tête de fichier |
| B3 | `generate_attack_schedule_txt()` : dataset, nb métriques calculés dynamiquement depuis CONFIG |
| B4 | `_validate_catalog()` : validation des valeurs négatives + clés requises + noms dupliqués |
| B5 | Double ligne vide supprimée |
| CP1 | Assertion `VERSION_NAME_MODIF == VERSION_NAME + "_attacks"` ajoutée dans `main()` |
| CP2 | `clean_metric_key = metric_key.replace("->", "_to_")` utilisé pour les noms de colonnes |
| F1/F2 | Module docstring mis à jour (v3, pipeline dynamique, principe de normalisation) |
| — | `WINDOW_SIZE = CONFIG.get('WINDOW_SIZE', 10)` ajouté au niveau module |
| — | `_validate_catalog()` appelée dans `main()` avant toute injection |

### Comportement des signatures après normalisation (exemples)

| Attaque | Métrique | Poids bruts (a,s,n) | Normalisés sur W=10 | u naturel | u max naturel |
|---|---|---|---|---|---|
| UDP_FLOOD | prophet_bytes | (40, 8, 0.5) | (8.25, 1.65, 0.103) | 3/13 = 0.231 | 3/13 = 0.231 |
| SYN_FLOOD | prophet_syn | (45, 5, 0) | (9, 1, 0) | 3/13 = 0.231 | 3/13 = 0.231 |
| SLOWLORIS | prophet_bytes | (0.3, 2, 8) | (0.29, 1.96, 7.75) | 3/13 = 0.231 | 3/13 = 0.231 |

L'incertitude `u` des opinions injectées (plateau, α=1) est désormais **identique** à
celle de n'importe quelle fenêtre naturelle avec WINDOW_SIZE=10, W=3.

*Document généré lors de la revue de code du 2026-04-11.*
