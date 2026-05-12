# Revue scientifique et technique — `qualify_anomaly_sbn.py` (v2)

> Révision du : 2026-04-12 (v2 — après application des corrections de la revue v1)
> Contexte : préparation à l'évaluation par un jury scientifique
> Fichier principal : `qualify_anomaly_sbn.py`
> Dépendances directes : `config.py`, `paths.py`, `compute_opinions_v3.py` (fournit le CSV d'entrée)
> Revue précédente : `review_qualify_anomaly_sbn_v1.md` (2026-04-11)

---

## 0. Statut des corrections de la revue v1

| ID v1 | Statut | Note de vérification |
|---|---|---|
| B1 `CONFIG in dir()` toujours False | ✅ Corrigé | `_g = globals()` utilisé dans `run()` ligne 1124 |
| B2 `W` dead code | ✅ Corrigé | Supprimé de `sbn_qualify_row()`, CLI et `run()` |
| B3 Matrice transition jamais appelée | ✅ Corrigé | Intégrée en L5 : `T.T @ b_vec` (lignes 848–852) |
| B4 Double escompte temporel | ✅ Corrigé | `w_temp_eff = temporal_weight` fixe, `_discount_opinion` seul responsable |
| B5 `novelty_score` alias redondant | ✅ Corrigé | Supprimé — seul `novelty_lr` subsiste |
| B6 Gate OUTAGE bypass contredisant la docstring | ✅ Corrigé | Gate séparée supprimée ; OUTAGE via SBN natif |
| B7 Commentaire 0.90 vs code 0.92 | ✅ Partiel | 0.92 corrigé ; nouveau problème 0.25 vs 0.20 (NB4) |
| B8 Fallback gate L1 silencieux | ✅ Corrigé | `_DET_COL` ajouté dans `required_cols` ligne 1162 |
| Mode `log_lr` dégradant | ✅ Supprimé | `_evidence_log_lr_scores` absent du code actuel |
| ST4 Cohérence GROUP_SOURCES ↔ SBN_COND | ✅ Corrigé | Vérification lignes 1142–1146 |
| F2 `novelty_threshold` non branché | ✅ Corrigé | Branché via `_lr_thr` ligne 1229 |
| F4 `--temporal` help insuffisant | ✅ Corrigé | Explication désactivation par défaut ajoutée |
| S5 Aucune évaluation quantitative de type | ✅ Ajouté | `_eval_type_performance()` lignes 1007–1084 |
| Sensibilité SBN_COND | ✅ Ajouté | `_sensitivity_analysis()` + CLI `--sensitivity` |
| F3 `prev_gate` redondant | ❌ Non corrigé | Toujours présent dans `run()` |
| S2 Indépendance conditionnelle dot-product | ⚠️ Partiel | Documenté dans le code mais non quantifié empiriquement |
| S6 Référence DST incorrecte pour `_lr_novelty` | ❌ Non corrigé | Cite toujours Shafer 1976 DST |
| S7 Seuil novelty non calibré sur données réelles | ❌ Non corrigé | Validation uniquement sur signatures théoriques |
| S8 `u_sbn=1.0` pour fenêtres normales trompeur | ❌ Non corrigé | CSV contient u_sbn=1.0 pour toutes les fenêtres normales |

---

## 1. Rôle du module dans la chaîne SL-ADS

`qualify_anomaly_sbn.py` est le **cinquième et dernier maillon** de la chaîne de
détection. Il prend le CSV de scores d'opinion produit par `compute_opinions_v3.py`
(colonnes `FINAL_SYSTEM_CBF_proj_atk`, `{metric}_proj_safe/susp/atk`, …) et produit,
pour chaque fenêtre temporelle franchissant le seuil de décision `δ`, une
**qualification de type d'attaque** dans un cadre Subjectif Bayésien (SBN).

Le pipeline interne suit une architecture à 6 couches explicites :

| Couche | Nom | Opération | Référence |
|---|---|---|---|
| L1 | Gate d'activation | `P(Anom) ≥ δ` sur `_DET_COL` | — |
| L2 | Opinions de groupes | Geomean des prob. projetées par groupe | Aczél & Daróczy 1975 |
| L3 | Score SBN par type | Dot-product `E_{s~P^obs_g}[c^{k\|g}_s]` | Jøsang 2016 §14.3 |
| L4 | Bijection SL | Evidence → opinion, W=K dynamique | Jøsang 2016 Déf. 3.9 |
| L5 | Prior temporel (opt.) | WBF + matrice T (Kill Chain) + escompte λ^Δt | §12.22, Déf. 14.6, Hutchins 2011 |
| L6 | Uncertainty Maximisation (opt.) | Amplifie u pour anomalies inconnues | Jøsang 2016 Eq. 3.27 |

**Entrée** : CSV de `compute_opinions_v3.py` / `evaluate_injection_v2.py`
(colonnes `{src}_proj_{safe,susp,atk}`, `FINAL_SYSTEM_CBF_proj_atk`).

**Sortie** : `qualif_types_sbn.csv` — colonnes `gate_open`, `top1_type`, `top1_b`,
`b_sbn_{type}`, `u_sbn`, `novelty_lr`, `b_sbn_raw_{type}`, `u_sbn_raw`.

---

## 2. Fonctions et outils employés

| Fonctionnalité | Implémentation | Référence théorique |
|---|---|---|
| Opinions conditionnelles expertes | `_DEFAULT_SBN_COND` / `CONFIG['SBN_COND_OPINIONS']` | Sharafaldin 2018 ; Mirsky 2018 ; Moustafa 2015 ; MITRE ATT&CK ; Rossow 2014 |
| Geomean par groupe (L2) | `_compute_group_projected()` | Aczél & Daróczy 1975 ; Genest & Zidek 1986 |
| Score dot-product (L3) | `_sbn_group_score()` | Jøsang 2016 §14.3 |
| Evidence accumulation (L3→L4) | `_evidence_sum_scores()` | Good 1952 Ann. Math. Stat. §4 |
| Bijection SL evidence→opinion (L4) | `_sl_bijection()` | Jøsang 2016 Déf. 3.9, §3.5.2 |
| Uncertainty Maximisation (L6) | `_apply_um()` | Jøsang 2016 Eq. 3.27 |
| WBF fusion (L5) | `_wbf_two()` | Jøsang 2016 Eq. 12.22 |
| Escompte temporel (L5) | `_discount_opinion()` | Jøsang 2016 Déf. 14.6 |
| Matrice de transition Markovienne (L5) | `_build_transition_matrix()` | Hutchins et al. 2011 (Kill Chain) |
| Propagation Markovienne | `T.T @ b_vec` | Chapman-Kolmogorov, ordre 1 |
| Métrique de nouveauté | `_lr_novelty()` | (voir NS2 ci-dessous) |
| Analyse de sensibilité | `_sensitivity_analysis()` | Cooke 1991 (élicitation d'expert) |
| Évaluation quantitative de type | `_eval_type_performance()` | Métriques standard P/R/F1 |
| Lecture / écriture CSV | `pandas.read_csv / to_csv` | — |

---

## 3. Hypothèses posées (explicites et implicites)

### 3.1 Hypothèses sur les données d'entrée

- **H1 – Colonne de détection disponible** : le CSV d'entrée doit contenir la
  colonne `_DET_COL` (défaut : `FINAL_SYSTEM_CBF_proj_atk`). Son absence
  déclenche un `ValueError` explicite (ligne 1164). ✓

- **H2 – Colonnes de probabilités projetées disponibles** : les colonnes
  `{metric}_proj_safe/susp/atk` pour chaque source de `GROUP_SOURCES` sont
  supposées présentes. Leur absence partielle est signalée par `[WARN]` mais
  ne bloque pas : la fenêtre est qualifiée sur les groupes disponibles seulement.
  **Risque** : si GROUP_SOURCES est vide ou tous les groupes sont absents, la
  qualification retourne `u_sbn = 1.0` pour toutes les fenêtres sans avertissement
  bloquant (voir NB5).

- **H3 – Stationnarité intra-fenêtre** : le type d'attaque est supposé constant
  sur toute la durée d'une fenêtre. Une transition en cours de fenêtre produit
  un score mixte non modélisé.

- **H4 – Alignement positionnel pour `_eval_type_performance`** : la fonction
  aligne `df_out` et `df_input` par position de ligne (`.values`), pas par
  timestamp. Si l'ordre des lignes diffère entre les deux DataFrames, les
  métriques sont silencieusement erronées.

### 3.2 Hypothèses du réseau bayésien subjectif

- **H5 – Indépendance conditionnelle des groupes (Naive Bayes)** : l'accumulation
  additive des scores de groupe `e(k) = Σ_g contrib_g` revient à supposer que
  les groupes sont conditionnellement indépendants étant donné le type. Or `volume`
  et `connections` sont structurellement corrélés (r > 0.7 attendu en DDoS).
  La justification par Domingos & Pazzani (1997) est citée dans le code (ligne 793)
  mais aucune mesure empirique de corrélation inter-groupes sur RedeRio n'est fournie.

- **H6 – Opinions conditionnelles expertisées** : la matrice `SBN_COND_OPINIONS`
  est construite par élicitation d'expert avec référencement littérature. Aucun
  apprentissage sur données étiquetées n'est effectué. Les 7 niveaux
  (`_strong_anom` = `{0.03, 0.07, 0.90}`, etc.) sont des conventions non calibrées
  sur le dataset RedeRio.

- **H7 – Prior uniforme sur les types** : W=K dans `_sl_bijection` implique
  P(H_k) = 1/K pour tout k. Ce prior non-informatif (Jøsang §3.5.2) est raisonnable
  en l'absence de statistiques d'attaque sur le réseau cible, mais dans un
  contexte réel PORT_SCAN est bien plus fréquent que BGP_HIJACK. Ce choix
  doit être justifié explicitement dans le rapport.

- **H8 – Geomean comme agrégateur optimal intra-groupe** : justifiée par le
  logarithmic opinion pooling (Genest & Zidek 1986). Mais la geomean est calculée
  état par état (pas sur le simplex conjoint), puis les résultats sont renormalisés.
  Cette procédure n'est pas équivalente à la moyenne géométrique sur le simplex
  (log-ratio transform). Ses propriétés asymptotiques ne sont pas discutées.

- **H9 – Asymétrie `max(0, score - 1/3)` : contre-indications non pénalisées** :
  si `score(k,g) < 1/3` (groupe contre-indiqué pour le type k), la contribution est
  nulle (pas négative). Cela signifie qu'un type avec plusieurs groupes
  contre-indiqués mais un seul groupe fortement concordant peut dominer un type
  avec tous les groupes neutres. Cette asymétrie est une propriété de conception
  qui doit être documentée et justifiée (voir NS1).

### 3.3 Hypothèses sur le prior temporel

- **H10 – Markov d'ordre 1** : la dépendance temporelle est limitée à t-1.

- **H11 – Décroissance exponentielle homogène** : λ=0.80 par défaut (demi-vie ≈
  3 fenêtres = 15 minutes pour RedeRio). Non calibré sur données.

- **H12 – WBF pour sources dépendantes** : WBF défini dans Jøsang pour sources
  indépendantes. L'opinion courante et l'opinion t-1 portent sur le même phénomène.
  L'usage est justifié comme meilleur compromis vs CBF (docstring `_wbf_two`),
  mais sans référence formelle alternative.

- **H13 – Précédence `SBN_TEMPORAL_ENABLED`** : si `SBN_TEMPORAL_ENABLED = True`
  dans config.py, le prior temporel est forcé activé même sans `--temporal` au CLI.
  Le CLI ne peut pas désactiver ce qui est forcé par config (voir NB3).

### 3.4 Hypothèses sur NETWORK_OUTAGE

- **H14 – Discrimination OUTAGE vs FLOOD via protocoles** : lors d'une panne,
  tous les protocoles (tcp, udp, icmp) ont des résidus Safe (chute globale).
  Pour un flood, au moins un protocole est anormal. Cette logique est correcte
  mais suppose une panne totale — une panne partielle (un seul protocole coupé)
  peut produire un signal similaire à un flood.

- **H15 – Ternaire indifférente à la direction** : le triplet `(proj_safe, proj_susp, proj_atk)`
  ne distingue pas un surplus (flood) d'un déficit (panne) en volume. La
  discrimination OUTAGE vs FLOOD repose exclusivement sur les groupes protocoles,
  pas sur le signe du résidu. Limitation documentée dans le code (ligne 779) mais
  non résolue.

---

## 4. Sorties produites

| Colonne CSV | Signification | Remarque |
|---|---|---|
| `timestamp` | Horodatage de la fenêtre | Copié depuis le CSV d'entrée |
| `gate_open` | Fenêtre classifiée comme anormale (P_Anom ≥ δ) | Booléen |
| `top1_type` | Type d'attaque dominant (argmax belief masses) | Chaîne vide si `gate_open=False` |
| `top1_b` | Belief mass du type dominant ∈ [0, 1] | 0.0 si `gate_open=False` |
| `b_sbn_{type}` | Belief mass après L5+L6, pour chaque type | 0.0 pour fenêtres normales |
| `b_sbn_Autre_Anomalie` | Toujours 0.0 (classe résiduelle via u·a) | Conforme Jøsang §14.4 |
| `u_sbn` | Incertitude SBN finale ∈ [0, 1] | **1.0 par construction pour fenêtres normales** |
| `novelty_lr` | 1/(max_L/mean_L) — concentration du signal de type | ~1.0 = inconnu ; ~0.5 = connu |
| `b_sbn_raw_{type}` | Belief mass avant L5+L6 (diagnostics) | |
| `u_sbn_raw` | Incertitude avant L5+L6 (diagnostics) | |

**Avertissement** : `u_sbn = 1.0` pour TOUTES les fenêtres `gate_open=False`.
Dans une analyse avale (corrélation u_sbn avec d'autres signaux), il faut
impérativement filtrer sur `gate_open=True`, sans quoi la moyenne de `u_sbn`
est artificiellement tirée vers 1.0.

---

## 5. Paramètres de configuration impliqués

| Paramètre CONFIG | Rôle | Valeur typique |
|---|---|---|
| `SBN_COND_OPINIONS` | Matrice d'opinions conditionnelles P(G=s\|type_k) | `_DEFAULT_SBN_COND` si absent |
| `QUALIFY_GROUP_SOURCES` | Groupes sémantiques : {nom → [métriques]} | 10 groupes définis dans config.py |
| `SBN_EVIDENCE_SCALE` | Facteur multiplicatif de l'évidence | 3.0 |
| `SBN_TEMPORAL_ENABLED` | Force le prior temporel (override CLI) | `False` |
| `SBN_LR_NOVELTY_THRESHOLD` | Seuil `novelty_lr` pour signal de nouveauté | 0.85 |

CLI uniquement :

| Argument CLI | Paramètre | Défaut |
|---|---|---|
| `--threshold` | Seuil gate δ | `_THRESHOLD` depuis config |
| `--temporal` | Active L5 prior temporel | `False` |
| `--no_um` | Désactive L6 UM | `False` |
| `--lambda_t` | Facteur de décroissance λ | 0.80 |
| `--w_temp` | Poids fixe du prior temporel | 0.30 |
| `--novelty` | Seuil `novelty_lr` pour stats | 0.85 |
| `--compare` | CSV `qualify_anomaly.py` pour comparaison | None |
| `--sensitivity` | Lance l'analyse de sensibilité ±0.05 | False |

---

## 6. Problèmes identifiés (v2 — analyse du code post-corrections v1)

### 6.1 Bugs / erreurs silencieuses

#### NB1 — `_extract_opinion_from_row` : dead code (ligne 890)

**Problème :** La fonction `_extract_opinion_from_row()` (lignes 890–898) est définie
mais n'est appelée **nulle part** dans le code. La logique d'extraction de l'opinion
depuis la ligne résultat est dupliquée inline dans `run()` (lignes 1202–1204) :
```python
b_dict = {k: r.get(f'b_sbn_{k}', 0.0) for k in type_names}
b_dict['Autre_Anomalie'] = r.get('b_sbn_Autre_Anomalie', 0.0)
prev_opinion = {'b': b_dict, 'u': r['u_sbn']}
```
La coexistence des deux implémentations crée un risque de divergence silencieuse si
l'une est modifiée sans l'autre.

**Correction :** Supprimer `_extract_opinion_from_row()` ou l'utiliser dans `run()`.

---

#### NB2 — `_sensitivity_analysis` ignore `SBN_EVIDENCE_SCALE` de la config (ligne 933)

**Problème :** Dans `_top1_with_cond()` (interne à `_sensitivity_analysis`), les
likelihoods sont calculés avec la valeur par défaut de `evidence_scale` :
```python
likes = {t: _evidence_sum_scores(sc) for t, sc in scores.items()}
# ↑ evidence_scale=3.0 par défaut — TOUJOURS, même si SBN_EVIDENCE_SCALE ≠ 3.0 en config
```
Mais `run()` utilise `evidence_scale = float(_cfg.get('SBN_EVIDENCE_SCALE', 3.0))`.
Si `SBN_EVIDENCE_SCALE = 9.0` dans config.py, l'analyse de sensibilité opère sur
une échelle différente de la qualification réelle. Les résultats `STABLE/INSTABLE`
ne correspondent pas au comportement du système en production.

**Correction :**
```python
def _sensitivity_analysis(sbn_cond: dict, perturb: float = 0.05,
                          evidence_scale: float = 3.0) -> None:
    # Dans _top1_with_cond :
    likes = {t: _evidence_sum_scores(sc, evidence_scale=evidence_scale)
             for t, sc in scores.items()}
```
Et dans `run()` :
```python
if run_sensitivity:
    _sensitivity_analysis(sbn_cond, evidence_scale=evidence_scale)
```

---

#### NB3 — Précédence CLI < config pour `apply_temporal` (ligne 1128)

**Problème :**
```python
apply_temporal = apply_temporal or bool(_cfg.get('SBN_TEMPORAL_ENABLED', False))
```
Si `SBN_TEMPORAL_ENABLED = True` dans config.py, le prior temporel est TOUJOURS
activé même si l'utilisateur n'a pas passé `--temporal`. Or l'utilisateur qui
veut reproduire des résultats **sans** prior temporel ne peut pas désactiver ce
comportement via CLI — il doit modifier config.py. C'est contre-intuitif.

**Correction :** Documenter ce comportement explicitement dans le help CLI, ou
ajouter un flag `--no_temporal` qui force `apply_temporal = False`.

---

#### NB4 — Incohérence docstring `_build_transition_matrix` : PORT_SCAN→DATA_EXFIL 0.25 vs code 0.20

**Problème (B7-bis, non corrigé par la revue v1) :** La docstring (ligne 329) indique :
```
- PORT_SCAN → DATA_EXFIL : 0.25 (recon → exfil, kill chain Hutchins 2011)
```
Mais la liste `overrides` (ligne 357) contient :
```python
('PORT_SCAN', 'DATA_EXFIL', 0.20),
```
La valeur effective est **0.20**, pas **0.25**.

**Correction :** Aligner la docstring sur le code (choisir une valeur et l'appliquer aux deux endroits).

---

#### NB5 — `GROUP_SOURCES = {}` silencieux si `QUALIFY_GROUP_SOURCES` absent de config.py

**Problème :** Les lignes 61–75 font :
```python
from config import CONFIG, QUALIFY_GROUP_SOURCES as _GS_MODULE
GROUP_SOURCES = CONFIG.get('QUALIFY_GROUP_SOURCES', _GS_MODULE)
```
Si `QUALIFY_GROUP_SOURCES` n'est pas défini dans config.py, l'import lève un
`ImportError` et `GROUP_SOURCES = {}` (fallback). Dans ce cas, `group_pp = {}`
pour toutes les fenêtres → early return avec `u_sbn = 1.0` → aucune classification.
Ce cas d'échec total est silencieux.

**Correction :** Ajouter dans `run()` :
```python
if not GROUP_SOURCES:
    raise RuntimeError("[SBN] GROUP_SOURCES vide — vérifier QUALIFY_GROUP_SOURCES dans config.py")
```

---

#### NB6 — `_compare_outputs` fragile aux différences de format timestamp (ligne 1283)

**Problème :** La comparaison SBN vs heuristique effectue un merge sur `timestamp`.
Si les deux CSV ont des précisions différentes (`2025-11-10 08:00:00` vs
`2025-11-10 08:00:00.000`), le merge inner retourne 0 lignes et affiche
`"Aucune fenêtre commune gate_open"` sans diagnostic de la cause réelle.

**Correction :**
```python
df_sbn['timestamp'] = pd.to_datetime(df_sbn['timestamp']).dt.floor('s')
df_old['timestamp']  = pd.to_datetime(df_old['timestamp']).dt.floor('s')
```

---

#### NB7 — `prev_gate` redondant (non corrigé depuis F3 de la revue v1)

**Problème :** `prev_gate` (lignes 1179, 1206–1213) est redondant avec l'état de
`prev_opinion`. `prev_opinion is not None` remplirait exactement le même rôle.
La variable alourdit la lecture sans valeur fonctionnelle ajoutée.

**Impact :** Pas de bug fonctionnel — mineur.

---

### 6.2 Problèmes scientifiques et manques de justification

#### NS1 — Asymétrie de `_evidence_sum_scores` : contre-indications non pénalisées

**Problème :** La formule `e(k) = Σ_g max(0, score(k,g) - 1/3) × scale` tronque
les contributions négatives à zéro. Une contre-indication forte (ex. pour UDP_FLOOD,
observer des paquets TCP) contribue autant qu'une observation neutre — aucune
pénalité n'est appliquée. Cela signifie qu'un type avec 5 groupes concordants et
4 groupes contre-indiqués peut dominer un type avec 5 groupes neutres et 4 groupes
légèrement concordants.

Ce comportement est une **propriété de conception délibérée** (non un bug) — elle
favorise les types avec au moins un groupe très discriminant. Mais ce choix n'est
nulle part documenté comme tel. Un jury scientifique peut demander pourquoi la
symétrie `|score - 1/3|` (pénalisation symétrique) n'a pas été choisie.

**Pour le rapport :** Expliquer le choix de `max(0, ·)` : il produit des masses
d'évidence non-négatives (condition nécessaire pour Jøsang Déf. 3.9), et la
robustesse aux contre-indications isolées est considérée comme un avantage dans
un contexte de données réseau bruitées.

---

#### NS2 — Référence incorrecte pour `_lr_novelty` (non corrigé depuis S6 de la revue v1)

**Problème :** La docstring cite "Shafer 1976, Dempster-Shafer Theory §4" pour
justifier `max(L)/mean(L)`. Cette statistique n'est pas un concept de DST.
C'est plus précisément l'inverse de l'**indice de Herfindahl normalisé** (mesure
de concentration sur un ensemble de K valeurs), ou alternativement interprétable
comme un rapport de vraisemblance maximum.

**Correction :** Citer :
- Good (1950), *Probability and the Weighing of Evidence*, §6 — pour l'interprétation
  en termes de poids d'évidence maximum.
- Supprimer la référence Shafer 1976 DST qui est hors contexte.

---

#### NS3 — Seuil `novelty_lr > 0.85` validé uniquement sur signatures théoriques

**Problème :** Le seuil 0.85 est présenté avec une "validation" (lignes 678–680) :
- Attaques connues : 0.47–0.62 (< 0.85 ✓)
- Anomalie inconnue : 0.996 (> 0.85 ✓)

Mais cette validation est effectuée en injectant comme `group_pp` les opinions
conditionnelles elles-mêmes (cas théoriquement parfait). Sur des données réelles
bruitées, les `group_pp` ne correspondent pas exactement aux signatures et les
valeurs de `novelty_lr` pour des attaques connues peuvent être plus proches de 1.
Le seuil 0.85 n'a jamais été calibré sur des données RedeRio avec injections.

**Pour le rapport :** Indiquer explicitement que ce seuil est "calibré sur signatures
théoriques, non sur données". Proposer une procédure de calibration : injecter des
fenêtres d'attaques connues et des fenêtres de trafic réel non étiqueté, puis
sélectionner le seuil au ROC optimal.

---

#### NS4 — L'analyse de sensibilité ne teste pas la discriminabilité inter-types

**Problème :** `_sensitivity_analysis` construit la signature "parfaite" de chaque
type k en posant `group_pp = SBN_COND[k]` (la signature expert elle-même), puis
vérifie que top-1 = k. Ceci teste uniquement la **cohérence interne** des opinions
(si on observe exactement ce qu'on attend pour k, on classifie bien k).

Mais cela ne teste pas si deux types se discriminent l'un l'autre quand l'observation
est intermédiaire. Exemple : SLOWLORIS et SYN_FLOOD ont tous deux `fin_ratio = strong_anom`.
La seule différence est `volume = strong_safe` pour SLOWLORIS vs `weak_susp` pour
SYN_FLOOD. Est-ce suffisant pour discriminer sur données réelles bruitées ?

**Correction :** Ajouter un test de discrimination pairwise :
```python
# Pour chaque paire (k1, k2), créer group_pp = moyenne(SBN_COND[k1], SBN_COND[k2])
# et vérifier que le top-1 est le type attendu (le plus "fort" des deux) ou
# que u_sbn est élevé (signal ambigu correctement détecté comme tel).
```

---

#### NS5 — La matrice de transition n'est pas validée stochastiquement à l'exécution

**Problème :** `_build_transition_matrix` normalise les lignes à 1.0 et est documentée
comme produisant une matrice stochastique. Mais il n'y a aucune assertion vérifiant que
`T[i, :].sum() ≈ 1.0` et `T[i, j] ≥ 0` à la sortie. Un futur développeur ajoutant
un override négatif (erreur typographique) passerait silencieusement.

**Correction :**
```python
assert np.allclose(T.sum(axis=1), 1.0, atol=1e-9), "Matrice T non stochastique"
assert (T >= 0).all(), "Matrice T contient des entrées négatives"
```

---

#### NS6 — Chapman-Kolmogorov : `Autre_Anomalie` exclue de la propagation — invariant implicite

**Problème :** Dans L5 (lignes 848–852), la multiplication `T.T @ b_vec` est faite
sur les types de `type_names_tm` (clés de `sbn_cond`), excluant `Autre_Anomalie`.
La masse résiduelle `b_prev['Autre_Anomalie']` est systématiquement mise à 0.0
après la multiplication (ligne 852).

En pratique, `b_sbn_Autre_Anomalie = 0.0` est un invariant garanti par `_sl_bijection`
et `_apply_um`. Mais ce n'est nulle part documenté comme invariant dans le code.
Si un futur développeur brise cet invariant (ex. en assignant une masse à
`Autre_Anomalie`), la propagation Markovienne perdra silencieusement de la masse.

**Correction :** Ajouter un commentaire explicite :
```python
# Invariant : b_prev['Autre_Anomalie'] = 0.0 toujours (garanti par _sl_bijection
# et _apply_um). La classe résiduelle est portée par u*a_autre, pas par b.
```

---

#### NS7 — `u_sbn = 1.0` pour fenêtres normales dans le CSV — biais avale

**Problème (non corrigé depuis S8 de la revue v1) :** Toutes les fenêtres avec
`gate_open = False` ont `u_sbn = 1.0` par construction (`_empty_result()`).
Dans le CSV final, il est impossible de distinguer une fenêtre réellement normale
d'une fenêtre quasi-anormale (P_atk = δ - ε).

Si un analyste calcule `novelty_lr.mean()` sur tout le CSV sans filtrer sur
`gate_open`, il obtient une moyenne artificiellement proche de 1.0 (toutes les
fenêtres normales ont `novelty_lr = 1.0`).

**Correction recommandée :** Ajouter dans le CSV output une colonne
`qual_status` ∈ `{"normal", "qualified", "no_groups"}` pour distinguer les cas,
ou documenter dans le header du CSV que l'analyse doit se faire sur `gate_open=True`.

---

#### NS8 — BGP_HIJACK et BOTNET_CC non évaluables sur RedeRio

**Problème :** Le dataset RedeRio ne contient aucun exemple injecté de BGP_HIJACK
ni de BOTNET_CC. Ces deux types sont inclus dans `_DEFAULT_SBN_COND` et dans la
matrice de transition, mais il est impossible de valider leurs opinions conditionnelles
sur des données réelles.

**Pour le rapport :** Mentionner explicitement que `BGP_HIJACK` et `BOTNET_CC`
appartiennent à la taxonomie mais n'ont pas d'équivalent injecté dans RedeRio —
leur précision de classification est **non évaluable** dans ce contexte et
représente un travail futur.

---

### 6.3 Problèmes de forme / lisibilité

#### NF1 — Commentaire `_sl_bijection` : `min(W,2)` vs `max(K,2)` (ligne 533)

```python
# max(K, 2) garantit W ≥ 2 (domaine binaire minimal — évite D=0 pour K=1).
```
Le commentaire est correct dans le code actuel mais dans une version précédente il
disait `min(W, 2)` (sens inverse). Vérifier que la formulation actuelle est bien
`max(K, 2)` (plancher à 2) et non `min(K, 2)` (plafond à 2). ✓

---

#### NF2 — `_eval_type_performance` : alignement positionnel non documenté (ligne 1027)

```python
eval_df = pd.DataFrame({
    'gate_open': df_out['gate_open'].values,
    'gt':        df_input[gt_col].values,  # ← alignement par position
})
```
L'alignement par `.values` n'est pas documenté. Si les deux DataFrames ont le même
nombre de lignes mais des ordres différents (ex. après un filtre ou un tri), les
métriques seraient silencieusement erronées.

**Correction :** Soit aligner sur timestamp, soit ajouter un assert de longueur
avec un message d'erreur clair.

---

#### NF3 — Langues mélangées dans les commentaires

Les commentaires alternent anglais/français dans les mêmes sections de
`_DEFAULT_SBN_COND` (ex. `# bytes+packets ↑↑↑ — signal primaire` vs `# DISCRIMINATEUR CLÉ`).
Pour un jury international, harmoniser en une seule langue.

---

#### NF4 — `type_names` recalculé à chaque appel de `sbn_qualify_row` (ligne 733)

```python
type_names = [k for k in sbn_cond.keys()]
```
Recalculé à chaque appel (N_fenêtres fois). Passer `type_names` en paramètre
depuis `run()` réduirait la recalculation et rendrait l'API plus explicite.

---

### 6.4 Problèmes structurels

#### NST1 — `_sl_bijection` réimplémentée localement — risque de divergence avec `sl_formulas_v2`

**Problème :** `sl_formulas_v2.py` contient `evidence_to_opinion()`. La bijection
SL est réimplémentée localement dans `_sl_bijection()` (W=K dynamique). Si
`sl_formulas_v2` évolue (ex. changement de sémantique de W), la divergence sera
silencieuse car `qualify_anomaly_sbn.py` n'importe pas `sl_formulas_v2`.

**Recommandation :** Documenter explicitement que `_sl_bijection` est une
réimplémentation locale volontaire (pour W=K dynamique vs W=3 fixe dans sl_formulas_v2)
et qu'elle doit être maintenue cohérente manuellement.

---

#### NST2 — `run()` mélange traitement, I/O, stats, évaluation et comparaison

**Problème :** `run()` combine en une seule fonction :
(1) lecture CSV, (2) qualification ligne par ligne, (3) écriture CSV,
(4) stats, (5) évaluation vs ground-truth, (6) sensibilité, (7) comparaison.

Cela rend les tests unitaires impossibles sans créer des fichiers CSV physiques.

**Recommandation :** Extraire `_run_qualification(df, sbn_cond, …) → pd.DataFrame`
pour la logique pure, et garder `run()` comme orchestrateur I/O.

---

## 7. Vérification de cohérence du pipeline end-to-end

### 7.1 Invariant SL : Σb + u = 1

Vérifié ligne par ligne dans les stats (lignes 1252–1257) pour les fenêtres
`gate_open = True` avec rapport de l'erreur max et moyenne. ✓

### 7.2 Cohérence W=K

`_sl_bijection` utilise `W = max(K, 2)` avec `K = len(likelihoods)`.
`likelihoods` est construit depuis `scores_by_type` qui itère sur `sbn_cond.keys()`.
K est donc déterministe et cohérent avec `type_names`. ✓

### 7.3 Normalisation des opinions conditionnelles

`_normalize_cond_opinion()` est appelée avant chaque `_sbn_group_score()`,
garantissant que les masses `cond_raw` somment à 1.0 même si elles ont été
spécifiées avec des erreurs d'arrondi. ✓

### 7.4 Conservation de la masse Markovienne dans L5

La multiplication `T.T @ b_vec` conserve `sum(b_trans_vec) = sum(b_vec)` car T
est stochastique (chaque ligne somme à 1). L'invariant SL
`sum(b_prev) + u_prev = 1` est préservé. ✓

### 7.5 Monotonie de `_apply_um` : u_max ≥ u

Démonstration : `u_max = K1 × min(P)` avec `P[k] = b[k] + u/K1`.
`min(P) ≥ u/K1` car `b[k] ≥ 0`. Donc `u_max = K1 × min(P) ≥ u`. ✓
L'UM ne peut qu'augmenter l'incertitude — propriété conforme à Jøsang Eq. 3.27.

### 7.6 Validation des paramètres CLI

`run()` valide `lambda_temporal ∈ (0,1]` et `temporal_weight ∈ [0,1]` avec des
`ValueError` explicites (lignes 1119–1122). ✓

### 7.7 Point non vérifié : config partielle de `SBN_COND_OPINIONS`

Si `CONFIG['SBN_COND_OPINIONS']` existe mais ne contient que 5 des 11 types,
le K effectif sera 5. La matrice de transition sera reconstruite sur 5 types,
les autres ignorés silencieusement. Ce scénario de config partielle n'est ni
détecté ni documenté.

---

## 8. Récapitulatif priorisation

| Priorité | ID | Nature | Effort |
|---|---|---|---|
| 🔴 Critique | NB2 | `_sensitivity_analysis` avec scale incorrecte si `SBN_EVIDENCE_SCALE ≠ 3.0` | faible |
| 🔴 Critique | NB5 | `GROUP_SOURCES = {}` silencieux → qualification nulle sans erreur | faible |
| 🟠 Important | NB3 | CLI override par config non documenté pour `apply_temporal` | faible |
| 🟠 Important | NB4 | Docstring 0.25 vs code 0.20 pour PORT_SCAN→DATA_EXFIL | trivial |
| 🟠 Important | NS1 | Asymétrie `max(0,·)` non documentée comme choix délibéré | doc |
| 🟠 Important | NS4 | Analyse de sensibilité ne teste pas la discrimination pairwise | moyen |
| 🟠 Important | NS5 | Matrice T non vérifiée stochastiquement à l'exécution | faible |
| 🟡 Mineur | NB1 | `_extract_opinion_from_row` dead code | faible |
| 🟡 Mineur | NB6 | Merge timestamp fragile dans `_compare_outputs` | faible |
| 🟡 Mineur | NB7 | `prev_gate` redondant | faible |
| 🟡 Mineur | NF1–NF4 | Forme et lisibilité | trivial |
| 🔵 Structurel | NST1 | `_sl_bijection` locale — divergence silencieuse vs `sl_formulas_v2` | moyen |
| 🔵 Structurel | NST2 | `run()` mélange traitement et I/O | fort |
| 📝 Rapport | NS2 | Référence DST incorrecte pour `_lr_novelty` | doc |
| 📝 Rapport | NS3 | Seuil novelty non calibré sur données réelles | doc/exp |
| 📝 Rapport | NS6 | Invariant `Autre_Anomalie = 0.0` implicite — non documenté | doc |
| 📝 Rapport | NS7 | `u_sbn = 1.0` pour fenêtres normales — biais avale | doc |
| 📝 Rapport | NS8 | BGP_HIJACK / BOTNET_CC non évaluables sur RedeRio | doc |

---

## 9. Éléments à rédiger dans le rapport technique

### 9.1 Section « Architecture SBN à 6 couches »

- Schéma des 6 couches avec types et dimensions des entrées/sorties
- Justifier le passage de W=3 (ternaire) à W=K (multinomial) :
  propriété `u=1.0` à évidence nulle (Jøsang §3.5.2)
- Distinguer les deux régimes : détection (L1, binaire) vs qualification (L2–L6, multinomiale)
- Préciser que L5 et L6 sont ablables pour tests de reproductibilité
- Confirmer que L5 intègre la matrice de transition Markovienne (kill chain)

### 9.2 Section « Opinions conditionnelles expertes (`SBN_COND_OPINIONS`) »

- Tableau des 11 types × 10 groupes (depuis la version 10-groupes) avec le niveau
  assigné et la source littérature pour chaque case discriminante
- Décrire les 7 niveaux standardisés (strong_anom → strong_safe) et la convention
  `ud` (distance à l'uniforme) comme mesure de certitude de l'opinion
- Résultats de `_sensitivity_analysis` : tableau STABLE/INSTABLE par type pour ±0.05
- **Importante limitation** : valeurs non calibrées sur RedeRio (H6) —
  discussion du biais d'élicitation (Cooke 1991 §4)
- Discussion sur les types non évaluables : BGP_HIJACK, BOTNET_CC (NS8)

### 9.3 Section « Score SBN et bijection SL »

- Formaliser le dot-product comme `E_{s~P^obs_g}[c^k_s]` (espérance, pas vraisemblance)
- Démontrer que `Score(k,g) ∈ [min(c^k), max(c^k)] ⊂ [0,1]`
- Montrer la formule complète : `e(k) = Σ_g max(0, Score(k,g) - 1/3) × scale`
- Justifier `max(0, ·)` : masses d'évidence non-négatives (Jøsang Déf. 3.9) +
  robustesse aux contre-indications isolées (NS1)
- Analyser les cas limites : u=1.0 (tous groupes neutres) et
  `u_min ≈ 11/(18+11) ≈ 0.38` (9 groupes parfaits, K=11, scale=3.0)

### 9.4 Section « Prior temporel Markovien »

- Présenter la chaîne de Markov : T (kill chain) → `b_trans = T^T b_prev`
  (Chapman-Kolmogorov ordre 1, Hutchins 2011)
- Présenter l'escompte : `_discount_opinion` (Jøsang Déf. 14.6)
- Présenter la fusion WBF (Jøsang Eq. 12.22) avec poids fixe `w_temp`
- Montrer la courbe `λ^Δt` pour λ=0.80 : demi-vie ≈ 3 fenêtres = 15 min (RedeRio)
- Justifier WBF vs CBF pour sources dépendantes (H12)
- Inclure une ablation `λ ∈ {0.5, 0.7, 0.9}` sur un exemple de séquence d'attaque
- Reporter les valeurs effectives post-normalisation de la matrice T pour les types clés
  (ex. PORT_SCAN self-prob ≈ 0.65, pas 0.80 — NB4)

### 9.5 Section « Uncertainty Maximisation »

- Démontrer que `_apply_um` préserve les probabilités projetées P(xi) = b(xi) + a_i·u
- Montrer la propriété `u_max ≥ u` (monotonie — §7.5)
- Illustrer l'effet différentiel : attaque connue (u_max modéré) vs anomalie inconnue (u_max → 1.0)
- Justifier l'exclusion de `Autre_Anomalie` du calcul de K (sinon u_max = u, UM sans effet)

### 9.6 Section « Qualification de NETWORK_OUTAGE »

- Expliquer la suppression de la gate séparée (cohérence architecturale)
- Décrire le mécanisme de discrimination via les groupes protocoles (H14)
- Documenter les limitations H14 (panne partielle) et H15 (ternaire sans direction)
- Proposer l'extension future : colonne `{src}_residual_sign` dans `compute_opinions_v3.py`

### 9.7 Section « Évaluation quantitative de la qualification »

- Présenter les résultats de `_eval_type_performance` (P/R/F1/N par type, matrice de confusion)
- Si ground-truth disponible dans `detection_results_INJECTED.csv` : reporter les métriques
- Comparer avec l'heuristique LR (`--compare`)
- Signaler explicitement : BGP_HIJACK et BOTNET_CC non évaluables sur RedeRio (NS8)

### 9.8 Section « Hypothèses et limites »

- H5 Naive Bayes : discuter le biais sous corrélation inter-groupes (Domingos & Pazzani 1997)
- H6 Élicitation expert : sensibilité ±0.05 (voir §9.2)
- H11 Décroissance λ : calibration et ablation
- H12 WBF non-indépendant : quelle alternative formelle ?
- H15 Ternaire sans direction : impact sur discrimination OUTAGE vs FLOOD
- Limite globale : tous les résultats de qualification dépendent de `gate_open = True`,
  qui dépend du pipeline `compute_opinions_v3.py` → erreurs en cascade si la
  détection est faussée

---

## 10. Analyses approfondies — Questions du 2026-04-12

### 10.1 Escompte temporel L5 : même valeur que dans `compute_opinions_v3.py` ?

**Réponse : ce ne sont pas les mêmes paramètres, mais ils devraient être alignés.**

`compute_opinions_v3.py` utilise `LAMBDA_DECAY` (lu depuis `config.py`) dans
`sl.temporal_adaptive_ageing()` — cet ageing s'applique **au niveau de chaque
métrique Prophet** pour pondérer les observations récentes vs anciennes dans
la fusion temporelle intra-série.

`qualify_anomaly_sbn.py` utilise `lambda_temporal` (paramètre CLI `--lambda_t`, défaut 0.80)
dans `_discount_opinion()` — cet escompte s'applique **au niveau de la qualification
de type**, entre les fenêtres consécutives, pour déprécier le prior d'une fenêtre t-1.

**Deux niveaux de mémoire distincts :**

| Niveau | Paramètre | Fichier | Portée |
|---|---|---|---|
| Détection intra-série | `LAMBDA_DECAY` | `compute_opinions_v3.py` | Lissage des opinions Prophet par métrique |
| Qualification inter-fenêtres | `lambda_temporal` | `qualify_anomaly_sbn.py` | Persistance du type d'attaque entre fenêtres |

**Pour le rapport :** Ces deux paramètres peuvent être égaux (mémoire système uniforme)
ou différents (détection plus réactive, qualification plus persistante). Actuellement,
`LAMBDA_DECAY` est dans `config.py` et `lambda_temporal` est passé via CLI avec défaut 0.80.
Pour la cohérence et la reproductibilité, **aligner les deux sur le même paramètre CONFIG**
(ex. `TEMPORAL_DECAY`) ou documenter explicitement qu'ils sont indépendants et calibrés
séparément.

**Correction recommandée dans `run()`** (déjà documentée dans le code) :
```
lambda_temporal : utiliser la même valeur que LAMBDA_DECAY dans config.py
```

---

### 10.2 Détection d'anomalie inconnue ("Autre_Anomalie") — Refonte et correction

#### Pourquoi l'approche P_proj avec a_autre=2/K a échoué (régression DATA_EXFIL)

L'implémentation initiale utilisait `u_final` (après UM) pour décider si Autre gagne :
```
Autre gagne si : b_top < u_final / K
```
Pour DATA_EXFIL (attaque "low" intensity) : u_sbn ≈ 0.689, u/K ≈ 0.063, b_top ≈ 0.055 < 0.063
→ **Autre_Anomalie gagne systématiquement pour DATA_EXFIL** → 0% précision (régression).

**Erreur fondamentale** : utiliser `u_final` (après UM) est incorrect pour la détection
de nouveauté. L'UM augmente artificiellement u en préservant P_proj — c'est un mécanisme
de communication de l'incertitude, pas un indicateur de l'intensité de l'évidence.

#### Solution corrigée : u_raw comme indicateur de nouveauté

**Décision top1 :** revenir à `argmax(b_final)` parmi les types nommés.
- Propriété mathématique : `argmax_k b(k) = argmax_k P_proj(k)` quand a(k) est uniforme
  (car u/K est une constante additive identique → n'affecte pas l'argmax)
- **Équivalence garantie avec Jøsang §14.4** pour K types à prior égal

**Détection de nouveauté :** utiliser `u_raw` (avant UM, avant temporal) avec seuil :
```
qual_status = 'autre_anomalie'  si  u_raw > SBN_NOVELTY_U_RAW_THRESHOLD (défaut 0.82)
```

**Pourquoi u_raw et non u_sbn ?**

| Méthode | Valeur pour DATA_EXFIL (low) | Valeur pour UNKNOWN |
|---|---|---|
| u_sbn (post-UM) | ≈ 0.689 | ≈ 0.85-0.95 |
| u_raw (pre-UM) | ≈ 0.650-0.680 | ≈ 0.85-0.95 |
| novelty_lr | ≈ 0.695 | ≈ 0.736 (observé) |

→ u_raw < 0.82 pour DATA_EXFIL → **pas d'Autre_Anomalie** ✓  
→ u_raw → 1.0 pour signal totalement inconnu → **Autre_Anomalie** ✓

**Calibration analytique du seuil 0.82 :**
```
u_raw = K / (e_total + K)
e_total = K × (1/u_raw - 1) = 11 × (1/0.82 - 1) ≈ 2.4 unités d'évidence
```
Soit < 2.4 "pas" d'évidence au-dessus de la neutralité 1/3 pour déclencher Autre.

| u_raw | e_total | Interprétation |
|---|---|---|
| 0.458 | ≈ 13 | 9 groupes `_strong_anom` — attaque parfaitement discriminée |
| 0.65 | ≈ 5.9 | Signal modéré — DATA_EXFIL type |
| 0.82 | ≈ 2.4 | **Seuil** — très peu d'évidence |
| 0.90 | ≈ 1.2 | Signal quasi-nul |
| 1.00 | 0 | Aucune évidence — anomalie inconnue totale |

**Limitation honnête :** ce seuil de 0.82 est calibré analytiquement, pas empiriquement.
Sa validation requiert des injections d'attaques réellement inconnues sur RedeRio
et le calcul du ROC `u_raw` vs `ground_truth_novel`. C'est un travail futur documenté.

#### Rôle résiduel de `novelty_lr`

`novelty_lr` reste utile comme **signal complémentaire** dans `evaluate_qualify_sbn.py` :
- Critère 1 (principal) : `qual_status == 'autre_anomalie'` (>30% des fenêtres détectées)
- Critère 2 (fallback) : `novelty_lr > 0.85`
- Signal considéré actif si au moins un des deux est positif

Pour UNKNOWN_ANOMALY_CONTROL avec `novelty_lr = 0.736 < 0.85` : le critère 2 échoue.
La détection dépendra alors du critère 1 (u_raw > 0.82). Si l'UNKNOWN_ANOMALY_CONTROL
a u_raw ≈ 0.75-0.80, le seuil 0.82 pourrait encore être manqué — ce qui signifie que
cette attaque de contrôle **ressemble trop** à des types connus pour être détectée comme
vraiment nouvelle. C'est une limitation du système qui doit être reportée.

---

### 10.3 Matrice de transition Markovienne : rôle et fonctionnement détaillé

**Contexte :** la matrice T est utilisée dans L5 (prior temporel) pour propager la
classification de la fenêtre précédente vers la fenêtre courante en tenant compte
des séquences d'attaque plausibles.

#### Définition formelle

`T[i, j] = P(type_actuel = j | type_précédent = i)` — probabilité de transition du
type i au type j entre deux fenêtres consécutives.

La matrice est **stochastique** : chaque ligne somme à 1.0 (propriété vérifiée
par assertion dans `_build_transition_matrix`).

#### Construction

1. **Base :** diagonale = 0.80 (persistance nominale — "une attaque en cours a 80% de
   chance de continuer"). Off-diagonal = 0.20 / (K-1) ≈ 0.02 uniformément.

2. **Overrides kill chain (Hutchins 2011) :** après normalisation, les valeurs
   effectives diffèrent des valeurs nominales :

| Transition | Valeur nominale | Valeur effective (K=11) |
|---|---|---|
| PORT_SCAN → PORT_SCAN | 0.80 | ≈ 0.65 |
| PORT_SCAN → DATA_EXFIL | 0.20 | ≈ 0.16 |
| PORT_SCAN → HTTP_FLOOD | 0.05 | ≈ 0.04 |
| BOTNET_CC → BOTNET_CC | 0.80 | ≈ 0.68 |
| BOTNET_CC → UDP_FLOOD | 0.15 | ≈ 0.13 |
| BOTNET_CC → SYN_FLOOD | 0.10 | ≈ 0.08 |
| NETWORK_OUTAGE → NETWORK_OUTAGE | 0.92 | ≈ 0.87 |
| BGP_HIJACK → NETWORK_OUTAGE | 0.20 | ≈ 0.16 |

3. **Normalisation ligne par ligne** → matrice stochastique.

#### Utilisation dans L5

```
b_trans[j] = Σ_i T[i, j] × b_prev[i]    (Chapman-Kolmogorov, ordre 1)
```

en notation matricielle : `b_trans = T^T × b_prev`.

**Interprétation :** `b_trans[j]` est la probabilité d'être dans le type j à l'instant t,
sachant la distribution de types à t-1 et les probabilités de transition.

**Ensuite :** `b_trans` est escompté via `_discount_opinion(b_trans, u_prev, λ^Δt)`,
puis fusionné avec l'opinion courante via WBF.

#### Impact concret

Sans matrice (T=I, identité) : le prior temporel dit "on était en PORT_SCAN → on est
probablement encore en PORT_SCAN". Avec la matrice : "on était en PORT_SCAN → il y a
16% de chances qu'on soit passé à DATA_EXFIL". Cela permet d'anticiper les séquences
d'attaque et d'augmenter `b_sbn_DATA_EXFIL` même avant que le signal DATA_EXFIL
devienne dominant dans les observations.

---

### 10.4 Analyse des cas limites : u=1.0 et u_min

#### Cas 1 : u_sbn = 1.0 (vacuité totale)

Se produit dans 3 scénarios distincts :

| Scénario | `gate_open` | `qual_status` |
|---|---|---|
| P_atk < δ (fenêtre normale) | False | `normal` |
| P_atk ≥ δ mais GROUP_SOURCES vide | True | `no_groups` |
| P_atk ≥ δ, groupes disponibles, tous neutres | True | `autre_anomalie` |

**Mécanisme pour le scénario 3 :** si `group_pp[g] = {Safe:1/3, Susp:1/3, Anom:1/3}`
pour tout g, alors `Score(k, g) = Σ_s (1/3) × c^k_s = 1/3` (car les c^k_s somment à 1).
Donc `max(0, 1/3 - 1/3) = 0` pour tout g, k. → `e(k) = 0` pour tout k.
→ D = 0 + K = K → `u = K/K = 1.0` ✓, `b(k) = 0` pour tout k ✓.

**Preuve analytique :** La vacuité est garantie par la construction de `_evidence_sum_scores`
avec le terme `max(0, score - 1/3)` : si aucun groupe n'est discriminant (tous scores = 1/3),
aucune évidence n'est accumulée → bijection SL retourne l'opinion vacuouse `(b=0, u=1)`.

#### Cas 2 : u_min (attaque parfaitement identifiée)

**Bornes théoriques** pour K=11, scale=3.0, 9 groupes :

Avec observations `group_pp[g] = SBN_COND[k]` (signature parfaite) :

| Niveau d'opinion | Score(k,g) | Contribution | e(k) par groupe | u_min (9 groupes) |
|---|---|---|---|---|
| `_strong_anom` `{0.03, 0.07, 0.90}` | 0.816 | 1.449 | 1.449 | ≈ 0.458 |
| `_strong_safe`  `{0.85, 0.12, 0.03}` | 0.738 | 1.213 | 1.213 | ≈ 0.499 |
| Degenerate `{0, 0, 1}` (impossible) | 1.000 | 2.000 | 2.000 | ≈ 0.379 |

**Calcul pour _strong_anom :**
```
Score = 0.03² + 0.07² + 0.90² = 0.0009 + 0.0049 + 0.81 = 0.8158
e_max_per_group = (0.8158 - 0.333) × 3 = 1.449
e_total = 9 × 1.449 = 13.04
u_min = K / (e_total + K) = 11 / (13.04 + 11) = 11 / 24.04 ≈ 0.458
```

**Remarque importante :** Le commentaire dans le code citait `u_min ≈ 0.38` basé sur
des observations dégénérées `{0, 0, 1}` (impossibles en pratique). La valeur réaliste
pour `_strong_anom` est ≈ **0.458**.

**Implication pour `a_autre = 2/K` :** avec u=0.458, `b_top ≈ 0.48 >> u/K = 0.042`.
→ Une attaque parfaitement identifiée ne déclenchera jamais Autre_Anomalie ✓.

Après `_apply_um`, u_max ≥ u_sbn, donc l'incertitude croît encore — mais b_top
décroît proportionnellement, le ratio `b_top / u` reste constant, et la décision
Autre_Anomalie est préservée.

---

### 10.5 NETWORK_OUTAGE : mécanisme détaillé et améliorations possibles

#### Mécanisme actuel

**Signature SBN_COND pour NETWORK_OUTAGE :**
```
volume        = strong_anom   ← résidu absolu élevé (chute globale)
protocol_tcp  = strong_safe   ← pas d'explosion TCP
protocol_udp  = strong_safe   ← pas d'explosion UDP
protocol_icmp = strong_safe   ← pas de flood ICMP
tcp_flags     = strong_safe   ← pas de SYN/FIN surge
fin_ratio     = neutral       ← trafic minimal → ratio indéterminé
connections   = strong_anom   ← flows chutent aussi
entropy       = neutral       ← peu de paquets → entropie indéterminée
packet_size   = neutral       ← indéterminé
reconstruction = strong_anom  ← relations structurelles cassées
```

**Discrimination OUTAGE vs FLOOD :**

Pour un OUTAGE avec observation : `group_pp[protocol_udp] = {Safe:0.9, Susp:0.08, Anom:0.02}` :
```
OUTAGE : Score = 0.9×0.85 + 0.08×0.12 + 0.02×0.03 ≈ 0.775 → e += 1.33 (fort positif)
UDP_FLOOD: Score = 0.9×0.03 + 0.08×0.07 + 0.02×0.90 ≈ 0.051 → e += 0    (tronqué)
```
Le groupe `protocol_udp` contribue fortement à OUTAGE mais zéro à UDP_FLOOD ✓.

**Limitation fondamentale (H15) :** la ternaire `{proj_safe, proj_susp, proj_atk}` est
produite à partir du résidu absolu `|e_t| = |y_t − ŷ_t|` — indifférente au signe.
Or :
- Flood : `y_t >> ŷ_t` → résidu positif → `proj_atk` élevé
- Outage : `y_t << ŷ_t` → résidu négatif → `proj_atk` élevé **aussi**

Le volume est anormal dans les deux cas, avec la même ternaire. La discrimination
repose exclusivement sur le fait que pour un flood, au moins un groupe **protocole**
est anormal, alors que pour un outage, tous les protocoles sont safe.

#### Ce mécanisme fonctionne seulement si :
1. La panne est **totale** (tous les protocoles chutent ensemble)
2. Les métriques protocole discriminent correctement (pas de faux positifs)

#### Améliorations possibles

**Option A — Métriques directionnelles (recommandée, mais pipeline change)**

Ajouter dans `compute_evidence_v2.py` une accumulation séparée pour résidus positifs et négatifs :
```
{key}_P_pos, {key}_S_pos, {key}_N_pos   ← résidus positifs (surplus)
{key}_P_neg, {key}_S_neg, {key}_N_neg   ← résidus négatifs (déficit)
```

Dans `compute_opinions_v3.py`, produire :
```
{src}_proj_pos_atk   ← probabilité d'anomalie dans le sens positif
{src}_proj_neg_atk   ← probabilité d'anomalie dans le sens négatif
```

Dans `qualify_anomaly_sbn.py`, créer deux groupes de volume :
```
volume_surplus  → discrimine les floods (résidu positif)
volume_deficit  → discrimine l'outage (résidu négatif)
```
Ceci résout complètement H15 et permet une discrimination robuste même en cas de
panne partielle. **C'est un changement de pipeline de ~3 fichiers.**

**Option B — Score de cohérence inter-protocoles (sans pipeline change)**

Ajouter un groupe synthétique `coherence` qui mesure si **tous** les protocoles sont
simultanément anormaux (outage) vs un seul (flood). Cette cohérence peut être calculée
directement dans `qualify_anomaly_sbn.py` à partir des `group_pp` existants :
```python
# Dans _compute_group_projected, ajout d'un groupe 'coherence' :
# coherence_anomalie = geomean(protocol_tcp_atk, protocol_udp_atk, protocol_icmp_atk)
# Si tous les protocoles sont safe → coherence_safe élevé → signal OUTAGE
```
Cette option est moins précise mais évite les changements pipeline.

**Option C — Modèle de détection bi-directionnel au niveau de compute_evidence_v2.py**

Modifier la fonction trapézoïdale dans `compute_evidence_v2.py` pour distinguer les
anomalies directionnelles :
- `direction = 'up'` : résidu positif → flood (actuellement utilisé pour certains indicateurs)
- `direction = 'down'` : résidu négatif → outage

Le paramètre `direction` est déjà partiellement implémenté dans `compute_evidence_v2.py`.
L'extension consisterait à produire des colonnes séparées dans le CSV de sortie.

**Recommandation pour le rapport :** Implémenter Option A pour la version finale (travail
futur documenté), mentionner Option B comme amélioration incrémentale immédiate, et
expliciter H15 comme limitation connue de l'architecture actuelle.

---

### 10.6 H15 — Résolution complète : impact sur le pipeline

**Oui, résoudre H15 proprement nécessite de modifier 3 fichiers :**

| Fichier | Modification |
|---|---|
| `compute_evidence_v2.py` | Séparer l'accumulation P/S/N en deux colonnes (+/−) par indicateur |
| `compute_opinions_v3.py` | Produire `{src}_proj_pos_{safe/susp/atk}` et `{src}_proj_neg_{safe/susp/atk}` |
| `qualify_anomaly_sbn.py` | Créer des groupes `volume_surplus` et `volume_deficit` dans GROUP_SOURCES, et mettre à jour `_DEFAULT_SBN_COND` |

**Scope du changement :** il ne s'agit pas d'un changement d'architecture — c'est
une **extension des colonnes du CSV intermédiaire** (doublement des colonnes de
preuves). Les formules SL restent identiques. Le travail principal est dans
`compute_evidence_v2.py` (distinguer les résidus par signe dans la boucle de calcul).

**Justification scientifique :** Une attaque de type "vol silencieux" (exfiltration
lente avec réduction du trafic légitime) pourrait être confondue avec un outage dans
l'architecture actuelle. La direction du résidu est une information physique importante
qui ne devrait pas être perdue.

---

## 12. Modifications appliquées (2026-04-12)

| ID | Action | Statut |
|---|---|---|
| NB1 | `_extract_opinion_from_row()` supprimée (dead code) | ✅ |
| NB2 | `evidence_scale` passé à `_sensitivity_analysis()` et utilisé dans `_top1_with_cond` | ✅ |
| NB4 | Docstring corrigée : `0.25` → `0.20` pour PORT_SCAN→DATA_EXFIL | ✅ |
| NB5 | `if not GROUP_SOURCES: raise RuntimeError(...)` ajouté dans `run()` | ✅ |
| NB6 | `.dt.floor('s')` avant merge dans `_compare_outputs` | ✅ |
| NS2 | Référence Shafer 1976 DST → Good (1950) dans docstring `_lr_novelty` | ✅ |
| NS3 | Note "calibration théorique uniquement" ajoutée dans docstring `_lr_novelty` | ✅ |
| NS5 | Assertions stochastiques ajoutées dans `_build_transition_matrix` | ✅ |
| NS7 | Commentaire `u_sbn=1.0` pour fenêtres normales dans `_empty_result()` | ✅ |
| **REDESIGN** | Décision top1 via probabilités projetées P_proj (Jøsang §14.4) | ✅ |
| **REDESIGN** | `autre_anomalie_prior = 2/K` par défaut — configurable via `SBN_AUTRE_ANOMALIE_PRIOR` | ✅ |
| **REDESIGN** | Colonne `qual_status` ∈ `{normal, qualified, autre_anomalie, no_groups}` | ✅ |
| **REDESIGN** | Colonne `top1_proj` (probabilité projetée du top-1) | ✅ |
| **REDESIGN** | Validation config partielle `SBN_COND_OPINIONS` avec impact sur u_min | ✅ |
| **REDESIGN** | Stats `run()` distinguent fenêtres qualifiées vs no_groups | ✅ |
| **REDESIGN** | Docstring `lambda_temporal` : lien avec `LAMBDA_DECAY` de compute_opinions_v3.py | ✅ |

### Modifications restantes (non appliquées — rapport ou effort moyen)

| ID | Action | Priorité |
|---|---|---|
| NB3 | Documenter précédence config > CLI pour `apply_temporal` ; envisager `--no_temporal` | 🟠 |
| NB7 | Supprimer `prev_gate` redondant | 🟡 |
| NS4 | Ajouter test discrimination pairwise dans `_sensitivity_analysis` | 🟠 |
| NST1 | Documenter explicitement la divergence potentielle avec `sl_formulas_v2` | 🔵 |
| NST2 | Séparer `run()` en logique pure + I/O | 🔵 |
| **H15** | Résidus directionnels (+/−) dans compute_evidence_v2.py + compute_opinions_v3.py | 🔵 travail futur |

---

## 11. Notes croisées vers les autres modules

| Dépendance | Impact |
|---|---|
| `compute_opinions_v3.py` | Fournit `FINAL_SYSTEM_CBF_proj_atk` et `{metric}_proj_{safe/susp/atk}`. Toute modification du nommage casse la gate L1 ou les groupes L2 silencieusement (NB5). |
| `evaluate_injection_v2.py` | Produit `detection_results_INJECTED.csv` — entrée principale. Si la colonne `injected_type` est absente, `_eval_type_performance` est silencieusement ignorée (ligne 1024). |
| `config.py` | `QUALIFY_GROUP_SOURCES` doit être cohérent avec les clés de `SBN_COND_OPINIONS`. Vérification partielle (lignes 1142–1146) : groupes dans SBN_COND non définis dans GROUP_SOURCES sont signalés, mais groupes dans GROUP_SOURCES non utilisés dans SBN_COND ne sont pas détectés. |
| `sl_formulas_v2.py` | Non importé. `_sl_bijection()` est une réimplémentation locale volontaire (W=K dynamique). Maintenir cohérence avec `evidence_to_opinion()` (NST1). |
| `paths.py` | `get_decision_threshold()` fournit `_THRESHOLD`. Si recalibré, le seuil de la gate L1 change sans notification. |

---

*Document généré lors de la revue de code du 2026-04-12 (v2).*
