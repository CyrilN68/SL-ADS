# Audit Scientifique du Pipeline SL-ADS

## Objet

Ce document synthétise un audit méthodologique et logique du pipeline contenu dans `actual_ version_claude_autre dataset/`, avec un objectif précis : estimer ce qui, dans l'état actuel, empêcherait une présentation scientifiquement solide dans un article ou un rapport technique exigeant.

Le périmètre couvert ici est le pipeline complet :

- `run_full_sl_ads.py`
- `train_v10.py`
- `compute_evidence_v2.py`
- `compute_opinions_v3.py`
- `inject_at_evidence_level.py`
- `qualify_anomaly_sbn.py`
- `evaluate_with_labels.py`
- `evaluate_injection_v2.py`
- `evaluate_qualify_injected.py`
- `evaluate_qualify_real.py`
- `run_ablation_labeled.py`
- `compare_if_fair.py`
- `dataset_adapter/*`

## Verdict Global

En l'état, le pipeline n'est pas encore au niveau "publication robuste" pour des revendications fortes de performance, surtout sur les datasets labelisés et sur la comparaison de baselines.

Il y a plusieurs points très prometteurs dans l'architecture :

- une vraie logique de pipeline de bout en bout ;
- une séparation explicite entre détection, fusion SL et qualification ;
- des efforts visibles de calibration, d'ablation et de comparaison ;
- une volonté de justification théorique dans les commentaires.

Mais il existe aussi plusieurs menaces sérieuses à la validité interne :

- fuite méthodologique dans l'évaluation labelisée ;
- désalignement entre calibration et inférence réelle ;
- hypothèses non satisfaites ou non vérifiées ;
- confusion entre test synthétique aval et évaluation end-to-end ;
- incohérences de labels et de fenêtres selon dataset.

## Résumé Exécutif

### Bloquants avant soumission

| Sévérité | Point | Pourquoi c'est bloquant |
|---|---|---|
| Critique | Les scripts `evaluate_with_labels.py` et `run_ablation_labeled.py` évaluent toute la série et pas seulement le test hold-out | Toute métrique principale sur GECCO/CESNET devient méthodologiquement optimiste et non publiable comme performance de généralisation |
| Critique | Le prétraitement train/inférence n'est pas cohérent sur les valeurs manquantes | Le système déployé n'évalue pas les mêmes données que celles sur lesquelles il a été calibré |
| Critique | Le seuil de décision est calibré avec une fusion simplifiée différente de la fusion réellement déployée | Le seuil "opérationnel" n'est pas scientifiquement attaché au score final utilisé en production |
| Critique | Le fichier `data_standardized/METR_LA.csv` actuellement utilisé contient 0 labels positifs | Toute évaluation "labelisée" sur METR-LA est invalide ou triviale dans l'état actuel du workspace |
| Élevée | Les datasets GECCO et CESNET ont des anomalies déjà présentes avant `split_date`, alors que `train_v10.py` raisonne comme si le train était normal | Les hypothèses d'apprentissage et certaines justifications théoriques ne tiennent pas telles quelles |

### Points majeurs à cadrer si le pipeline est présenté

| Sévérité | Point | Impact sur le discours scientifique |
|---|---|---|
| Élevée | L'injection synthétique modifie directement les colonnes d'évidence | On ne peut pas présenter ces résultats comme une évaluation end-to-end du détecteur |
| Élevée | La CBF suppose des sources indépendantes alors que Prophet et Reconst partagent les mêmes variables brutes | Il faut présenter la CBF comme une heuristique utile, pas comme une fusion probabiliste rigoureusement justifiée par indépendance |
| Élevée | `compare_if_fair.py` calibre `contamination` d'Isolation Forest à partir des labels du test | La comparaison "fair" n'est pas publiable comme benchmark strict sans reformulation ni correction |
| Élevée | Les fenêtres d'évaluation sont incohérentes selon les scripts et parfois figées à 5 minutes | Les métriques peuvent être décalées temporellement ou non comparables inter-datasets |
| Élevée | Les signatures injectées et les opinions SBN viennent de la même logique experte/littérature | La qualification injectée risque d'être circulaire et optimiste |

## Constat 1 - Fuite méthodologique dans l'évaluation labelisée

### Ce qui se passe

Les scripts d'évaluation sur datasets labelisés n'isolent pas la période test.

- `evaluate_with_labels.py:606-638` aligne les labels avec toutes les opinions puis calcule les métriques sur l'ensemble des fenêtres communes.
- `run_ablation_labeled.py:935-953` aligne les labels avec toute l'évidence puis lance les sweeps et résumés sur l'ensemble du jeu.

Le `split_date` n'est pas utilisé pour restreindre l'évaluation au test dans ces deux scripts.

### Pourquoi c'est grave

Si les métriques principales sont calculées sur des fenêtres ayant déjà servi à l'apprentissage ou à la calibration, on ne mesure plus la généralisation hors échantillon. On mesure une performance mêlant :

- des segments vus en entraînement ;
- des segments réellement de test ;
- un état mémoire SL qui a déjà intégré l'historique amont.

Pour un article scientifique, cela invalide les revendications de performance hold-out si ce point n'est pas explicitement limité.

### Vérification sur les fichiers réellement pointés par `config.py`

Audit local du workspace au moment de l'analyse :

| Dataset | Fichier utilisé | Labels positifs train | Labels positifs test | Commentaire |
|---|---|---:|---:|---|
| RedeRio | `data/dataset_1310_2912_v30s.csv` | non disponible | non disponible | pas de colonne `label` |
| METR-LA | `data_standardized/METR_LA.csv` | 0 / 8918 | 0 / 25331 | le fichier actuellement utilisé est entièrement à 0 |
| GECCO-IoT | `data_standardized/GECCO.csv` | 1475 / 84252 | 251 / 55314 | train contaminé |
| CESNET-TimeSeries24 | `data_standardized/CESNET.csv` | 55 / 4033 | 1036 / 36265 | train contaminé |

### Conséquence scientifique

En l'état, les résultats labelisés ne doivent pas être revendiqués comme :

- performance sur test hold-out ;
- performance de généralisation temporelle ;
- comparaison équitable avec d'autres méthodes.

### Ce qu'il faudra dire si rien n'est corrigé

- "L'évaluation labelisée actuelle est descriptive sur la série complète."
- "Ces résultats ne constituent pas une estimation hors échantillon stricte."

## Constat 2 - Incohérence train / inférence sur les valeurs manquantes

### Évidence

- `train_v10.py:1049-1055` :
  - les métriques sont `ffill(limit=10)` ;
  - `fillna(0)` est explicitement évité pour ne pas assimiler un manque de mesure à un vrai trafic nul.
- `compute_evidence_v2.py:127` :
  - l'inférence fait `df = df.sort_values('ds').fillna(0)`.

### Pourquoi c'est grave

Le modèle est appris sur un régime de données où les trous courts sont propagés, mais il est déployé sur un régime où tout manque devient un zéro physique. Cela peut créer :

- des résidus artificiels ;
- des faux signaux d'anomalie ;
- une mauvaise estimation des seuils EVT et de la stabilité des baselines.

### Conséquence scientifique

Le pipeline n'a pas une définition unique de l'observation réseau. En article, cela fragilise :

- la reproductibilité ;
- l'interprétation causale des anomalies ;
- la validité des seuils appris.

## Constat 3 - Le seuil de décision n'est pas calibré sur le score réellement déployé

### Évidence

- `train_v10.py:873-977` calcule `proj_atk` d'entraînement en :
  - sommant directement les évidences P/S/N de toutes les métriques ;
  - moyennant les priors EDP ;
  - appliquant une seule bijection evidence -> opinion.
- `compute_opinions_v3.py:540-701` déploie au contraire :
  - ageing temporel adaptatif par métrique ;
  - prior EDP par métrique ;
  - discounting de confiance éventuel ;
  - WBF intra-branche ;
  - discounting contextuel éventuel sur Reconst ;
  - CBF ou WBF final entre branches.

### Pourquoi c'est grave

Le seuil opérationnel est calibré sur un objet différent du score final réellement utilisé. Ce n'est pas un simple détail d'implémentation ; c'est un changement de statistique de décision.

### Conséquence scientifique

On ne peut pas écrire honnêtement :

- "le seuil opérationnel du système final a été calibré automatiquement" ;

mais seulement quelque chose comme :

- "un seuil approché a été calibré sur une proxy simplifiée de la fusion finale".

Pour un papier sérieux, il faut soit réaligner la calibration sur la vraie chaîne de fusion, soit présenter explicitement cette approximation.

## Constat 4 - L'hypothèse "train normal" ne tient pas pour tous les datasets

### Évidence dans le code

- `train_v10.py:1138-1145` justifie certains choix en supposant que le train ne contient aucune attaque.
- `train_v10.py:1084-1105` découpe seulement par `split_date`.
- `train_v10.py` ignore les labels pendant l'entraînement ; il n'y a pas de filtrage explicite des fenêtres anormales via `label`.

### Évidence sur les données configurées

Audit local :

- GECCO-IoT : 1475 fenêtres positives dans le train ;
- CESNET-TimeSeries24 : 55 fenêtres positives dans le train ;
- METR-LA : le fichier utilisé a 0 labels positifs partout, donc l'affirmation "1202 anomalies" du launcher n'est pas matérialisée par le CSV réellement chargé.

### Pourquoi c'est important

Plusieurs justifications méthodologiques supposent implicitement un train sain :

- robustesse versus RANSAC ;
- calibration des résidus ;
- seuil EVT ;
- calibration du seuil de décision.

Si le train contient des anomalies, ces arguments doivent être reformulés ou démontrés empiriquement.

## Constat 5 - L'injection synthétique ne teste pas le pipeline end-to-end

### Évidence

- `inject_at_evidence_level.py:9-26` et `:1142-1223` injecte directement dans les colonnes d'évidence.
- Le pipeline injecté court-circuite donc la partie :
  - données brutes ;
  - prédiction Prophet / régression ;
  - calcul des résidus ;
  - conversion résidu -> évidence.

### Pourquoi c'est grave

L'expérience synthétique ne mesure pas :

- la capacité du pipeline à transformer un motif brut en résidu discriminant ;
- la robustesse des seuils d'évidence ;
- la dérive potentielle du prétraitement.

Elle mesure surtout :

- la qualité de la fusion SL aval ;
- la qualité de la qualification ;
- la sensibilité du score final à des signatures imposées.

### Formulation correcte dans un papier

Il faut parler de :

- "stress test aval au niveau des évidences"
- ou "controlled downstream perturbation study"

et éviter :

- "synthetic end-to-end intrusion detection evaluation"

## Constat 6 - Hypothèse d'indépendance de la CBF non satisfaite ou non démontrée

### Évidence

- `sl_formulas_v2.py:478` définit la CBF pour "2 sources indépendantes".
- `compute_opinions_v3.py:693-701` fusionne Prophet et Reconst via `fusion_cbf`.
- `config.py:175-196` montre que Reconst reconstruit à partir de variables déjà présentes dans le même espace d'observation :
  - `bytes <- packets`
  - `udp <- flows`
  - `fin <- syn`
  - `tcp <- packets`

### Pourquoi c'est important

Prophet et Reconst ne sont pas des capteurs indépendants au sens probabiliste fort. Ils exploitent des vues différentes, mais sur les mêmes signaux bruts, avec dépendances structurelles explicites.

### Conséquence scientifique

La CBF peut rester utile empiriquement, mais elle doit être présentée comme :

- une stratégie de fusion heuristique inspirée de la Subjective Logic ;

et non comme :

- une fusion strictement justifiée par indépendance des sources.

## Constat 7 - La comparaison "fair" avec Isolation Forest n'est pas encore strictement fair

### Évidence

- `compare_if_fair.py:432-436` calibre `contamination` avec la proportion d'attaques du test.
- `compare_if_fair.py:414-417` entraîne IF sur `train_normal`, donc avec accès aux labels pour nettoyer le train.
- `compare_if_fair.py:196` fixe par défaut `decision_window="5min"`.

### Pourquoi c'est problématique

Deux biais existent :

1. calibrer `contamination` avec le test est une fuite d'information ;
2. utiliser les labels pour nettoyer le train d'IF peut avantager IF par rapport à SL si SL n'a pas bénéficié du même nettoyage explicite.

### Conséquence scientifique

Cette comparaison peut être utile en analyse interne, mais ne doit pas être présentée comme benchmark final publiable sans correction du protocole.

## Constat 8 - Incohérences de fenêtre temporelle et d'ancrage selon les scripts

### Évidence

- `config.py:29-52` :
  - CESNET : `10min x 1`
  - GECCO : `1min x 5`
  - METR-LA : `5min x 1`
  - RedeRio : `30s x 10`
- `config.py:598` impose `EVAL["WINDOW_MIN"] = 5`.
- `compute_opinions_v3.py:347-359` resample selon la fréquence issue du dataset avec `origin='epoch'`.
- `evaluate_with_labels.py:132-137` rééchantillonne les labels avec `origin="start_day"`.
- `run_ablation_labeled.py:886-899` resample l'évidence en `5min` codé en dur, puis les labels à `window_min`.
- `evaluate_injection_v2.py:74-82` tente de corriger `WINDOW_MIN` via `CONFIG.get("SELECTED_FREQ", "30s")`, alors que la config stocke la fréquence dans `freq_data`.

### Pourquoi c'est grave

La définition de la "fenêtre d'évaluation" n'est pas uniforme entre :

- production des opinions ;
- alignement des labels ;
- ablation ;
- évaluation injection ;
- benchmark baseline.

Un décalage d'origine ou de largeur de fenêtre suffit à changer :

- la couverture ;
- le TTD ;
- le FPR ;
- le nombre d'épisodes.

## Constat 9 - Circularité potentielle entre signatures injectées et SBN de qualification

### Évidence

- `inject_at_evidence_level.py:98-815` encode des signatures d'attaques guidées par la littérature et par expertise.
- `qualify_anomaly_sbn.py:78-118` et le reste du fichier encode des opinions conditionnelles SBN également guidées par littérature et expertise.

### Pourquoi c'est important

Si les attaques injectées sont construites avec les mêmes signatures conceptuelles que celles utilisées pour définir les règles de qualification, le système peut paraître très bon en qualification simplement parce que le générateur de test et le classifieur partagent la même ontologie.

### Conséquence scientifique

La qualification injectée doit être présentée comme :

- validation de cohérence interne du raisonnement expert ;

et non comme :

- preuve forte de généralisation à des signatures réelles non vues.

## Constat 10 - L'évaluation de qualification injectée duplique la vérité terrain à la main

### Évidence

- `evaluate_qualify_injected.py:63-92` redéfinit une liste `INJECTED_ATTACKS` avec des dates hardcodées.
- `inject_at_evidence_level.py` possède déjà le catalogue et produit aussi une planification d'attaque.

### Pourquoi c'est risqué

La vérité terrain d'évaluation n'est pas lue depuis l'artefact réellement injecté, mais réécrite à la main dans un deuxième script. Toute divergence future entre :

- le catalogue d'injection ;
- les dates effectivement injectées ;
- le script d'évaluation ;

créera une erreur silencieuse.

### Problème supplémentaire de terminologie

- `evaluate_qualify_injected.py:138` définit `precision = n_correct / n_detected`.

Cette quantité n'est pas une précision globale au sens classification temporelle standard. C'est une exactitude conditionnelle parmi les fenêtres détectées pendant la période d'attaque. Si elle est rapportée comme "precision" sans définition stricte, cela gonfle artificiellement la lisibilité des résultats.

## Constat 11 - Auto-calibration de la fiabilité Reconst scientifiquement fragile

### Évidence

- `train_v10.py:728-845` estime `RECONST_ATTACK_RELIABILITY` à partir d'un taux de "cécité structurelle" où Prophet est suspect et Reconst reste très safe.

### Pourquoi c'est discutable

Cette calibration n'est pas apprise sur de vraies attaques annotées, mais sur des désaccords en régime normal ou quasi-normal. Elle mesure surtout :

- un désaccord inter-modèles ;

pas directement :

- une "blindness to attacks".

### Conséquence scientifique

Il faut présenter ce paramètre comme une heuristique de pondération contextuelle, pas comme une estimation fidèle de la fiabilité en attaque.

## Constat 12 - Risques secondaires mais réels

### Conflit non défaut dans la SL

- `sl_formulas_v2.py:316-383` permet `CONFLICT_MODE = projected_prob` ou `kl_symmetric`.
- `compute_opinions_v3.py:540-546` appelle `temporal_adaptive_ageing` sans fournir `a_prev` ni `a_curr`.

Conséquence : si l'on change de mode de conflit, le calcul peut retomber sur des priors uniformes implicites et ignorer les EDP dans le conflit. C'est un bug latent pour analyses de sensibilité.

### Terminologie "adaptive base rate"

- `compute_opinions_v3.py:567` remplit `base_rate_hist`, mais les priors utilisés sont statiques par métrique et viennent du train.

Si le rapport parle encore de "base rate adaptatif", il y a un risque de confusion conceptuelle. Ici, on est plutôt sur un prior empirique statique injecté dans une chaîne dynamique.

### Calendrier CESNET synthétique

- `dataset_adapter/cesnet_adapter.py:57` crée les timestamps à partir de `2024-01-01 + id_time * 10 min`.

Si Prophet exploite des patterns calendaires jour/semaine, cette date artificielle doit être explicitement justifiée, sinon on introduit une saisonnalité potentiellement fictive.

## Ce que je considérerais publiable aujourd'hui

### Revendications raisonnables

- Une architecture de fusion SL multi-métriques, multi-branches, capable de produire un score temporel cohérent.
- Une étude interne de sensibilité de la chaîne de fusion.
- Un stress test aval au niveau des évidences synthétiques.
- Une démonstration de faisabilité de qualification orientée expertise.

### Revendications non solides en l'état

- Performance hold-out robuste sur datasets labelisés.
- Supériorité comparative "fair" face à Isolation Forest.
- Validation end-to-end par injection synthétique.
- Généralisation forte de la qualification à des familles réelles non vues.

## Priorités de correction avant soumission

1. Restreindre toutes les évaluations labelisées à la période test stricte.
2. Harmoniser le prétraitement train / inférence.
3. Recalibrer le seuil sur la vraie chaîne de fusion finale.
4. Auditer les labels réellement présents par dataset et corriger le cas METR-LA.
5. Fixer une définition unique de la fenêtre d'évaluation et de son ancrage.
6. Refaire le benchmark IF sans utiliser les labels du test pour calibrer `contamination`.
7. Séparer clairement "stress test sur évidences" et "évaluation end-to-end".
8. Débrancher la duplication manuelle des catalogues d'attaque pour l'évaluation de qualification.

## Conclusion

Le pipeline a une base méthodologique riche et ambitieuse, mais il y a encore des écarts importants entre :

- ce que le code fait réellement ;
- ce que les commentaires affirment ;
- ce qu'un reviewer scientifique exigera.

La bonne nouvelle est que la majorité des problèmes sont des problèmes de protocole, d'alignement et de reporting, pas une absence totale de logique. Autrement dit : il y a une base exploitable, mais il faut verrouiller la validité expérimentale avant de transformer le pipeline en contribution scientifique solide.
