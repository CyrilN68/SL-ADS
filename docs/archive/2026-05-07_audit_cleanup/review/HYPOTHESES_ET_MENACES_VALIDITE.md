# Hypothèses du Pipeline et Menaces à la Validité

## But du document

Ce document liste les hypothèses implicites et explicites posées par le pipeline, puis les relie aux menaces classiques de validité d'un travail scientifique :

- validité interne ;
- validité externe ;
- validité de construit ;
- validité statistique ;
- reproductibilité.

Le principe est simple : toute hypothèse qui reste non testée doit être déclarée dans le rapport, et idéalement accompagnée d'une vérification empirique.

## 1. Hypothèses sur les données

## H1 - Le train représente un trafic majoritairement normal

Le pipeline d'entraînement suppose que la période avant `split_date` peut servir de référence pour apprendre :

- les modèles Prophet ;
- les régressions de reconstruction ;
- les seuils EVT ;
- les priors empiriques ;
- le seuil de décision.

### Risque

Cette hypothèse est fausse ou au moins incomplète si des anomalies sont présentes avant `split_date`.

### Audit local

- GECCO-IoT : train contaminé.
- CESNET-TimeSeries24 : train contaminé.
- RedeRio : absence de labels, hypothèse non vérifiable localement.
- METR-LA : fichier configuré avec labels tous nuls, donc hypothèse non testable dans l'état actuel.

### À mettre dans le rapport

- comment `split_date` a été choisi ;
- si le train a été vérifié comme sain ;
- si des exclusions ou nettoyages explicites ont été utilisés ;
- sinon, reconnaître que le pipeline est entraîné sur trafic possiblement contaminé.

## H2 - Les valeurs manquantes n'ont pas la même signification qu'un trafic nul

Cette hypothèse est explicitement reconnue dans `train_v10.py`, où `fillna(0)` est évité pour les métriques.

### Risque

Elle n'est pas respectée à l'inférence, où les manques deviennent des zéros.

### À mettre dans le rapport

- politique de gestion des manques ;
- cohérence train/test ;
- justification métier de la différence si elle est maintenue.

## H3 - Les labels fournis par les datasets standardisés sont fiables

Les scripts d'évaluation labelisée supposent que la colonne `label` est exploitable telle quelle.

### Risque

Cette hypothèse doit être auditée dataset par dataset. Dans le workspace analysé :

- METR-LA ne matérialise pas les anomalies attendues dans le CSV réellement chargé ;
- GECCO et CESNET ont des labels, mais ils apparaissent aussi dans le train.

### À mettre dans le rapport

- provenance exacte des labels ;
- transformation appliquée avant standardisation ;
- cohérence entre le texte du pipeline et les fichiers réellement chargés.

## 2. Hypothèses sur les modèles de base

## H4 - Prophet capture la normalité temporelle utile

Le pipeline suppose que Prophet est pertinent comme modèle de baseline pour la structure temporelle des métriques.

### Hypothèses implicites

- saisonnalité suffisamment stable ;
- pas de rupture massive de régime ;
- features calendaires sémantiquement valides.

### Risque

Cette hypothèse est plus fragile sur :

- des datasets où la temporalité réelle est reconstruite artificiellement ;
- des contextes où le train est contaminé ;
- des signaux très peu saisonniers.

### À mettre dans le rapport

- quelles composantes Prophet sont activées ;
- pourquoi elles sont plausibles pour chaque dataset ;
- quels contrôles ont été faits en cas de timestamps synthétiques.

## H5 - Les relations de reconstruction sont stables et interprétables

Les règles de reconstruction supposent des relations structurelles de type :

- `bytes <- packets`
- `udp <- flows`
- `fin <- syn`
- `tcp <- packets`

### Risque

Ces relations ne sont pas toutes d'égale solidité. Certaines sont très plausibles physiquement, d'autres sont seulement des proxys comportementaux.

### À mettre dans le rapport

- justification par paire ;
- qualité de fit observée sur train ;
- cas de fallback vers la moyenne ;
- limites d'interprétation quand `R² < 0`.

## H6 - Les queues de résidus sont suffisamment stables pour EVT

La calibration de seuils suppose que la modélisation des extrêmes est stable et comparable entre train et exploitation.

### Risque

Cette hypothèse peut être cassée par :

- contamination du train ;
- prétraitement non cohérent ;
- dérive de fréquence ou de fenêtre.

### À mettre dans le rapport

- sur quelles données les seuils EVT sont calibrés ;
- quelles vérifications de stabilité ont été faites ;
- quelles métriques utilisent un fallback et pourquoi.

## 3. Hypothèses sur la Subjective Logic et la fusion

## H7 - La conversion évidence -> opinion est une représentation pertinente

Le pipeline suppose qu'un triplet P/S/N agrégé capture correctement :

- le normal ;
- le doute ;
- l'attaque.

### Risque

C'est une hypothèse de construit. Elle doit être défendue empiriquement, pas seulement théoriquement.

### À mettre dans le rapport

- sémantique exacte de P, S, N ;
- convention de mapping résidu -> évidence ;
- invariants conservés.

## H8 - La mémoire temporelle adaptative est adaptée au phénomène

Le pipeline suppose que l'âge de l'information doit dépendre du conflit entre présent et passé.

### Risque

Cette hypothèse est plausible, mais elle change le comportement du système par rapport à une détection i.i.d. classique. Elle doit être décrite comme choix de modélisation, pas comme vérité neutre.

### À mettre dans le rapport

- pourquoi une mémoire temporelle est justifiée ;
- comment elle interagit avec la détection en ligne ;
- si l'état mémoire traverse la frontière train/test.

## H9 - Les priors empiriques sont stables et transférables

Le pipeline utilise des EDP par métrique.

### Risque

Ils ne sont pas adaptatifs dans le temps au sens strict ; ils sont appris sur train puis réinjectés. Leur stabilité hors distribution doit être démontrée ou reconnue comme hypothèse.

### À mettre dans le rapport

- méthode exacte d'estimation des EDP ;
- différence entre prior empirique statique et base rate adaptatif ;
- cas où le prior uniforme est utilisé.

## H10 - Les branches Prophet et Reconst peuvent être fusionnées via CBF

La CBF suppose des sources indépendantes.

### Risque

Dans ce pipeline, l'indépendance n'est pas garantie car les deux branches partagent des variables sources.

### À mettre dans le rapport

- reconnaître explicitement que l'indépendance forte n'est pas démontrée ;
- présenter la CBF comme un mécanisme pragmatique de combinaison ;
- si possible, ajouter une étude empirique de corrélation entre branches.

## 4. Hypothèses sur la qualification

## H11 - Les signatures expertes SBN correspondent aux classes d'attaque visées

Le qualifier SBN suppose qu'un jeu de relations conditionnelles codées à la main représente suffisamment bien les familles d'attaque.

### Risque

Cette hypothèse peut tenir en système expert, mais elle doit être assumée comme telle.

### À mettre dans le rapport

- ce qui est appris vs ce qui est expertisé ;
- sources bibliographiques ;
- nature heuristique ou normative des règles.

## H12 - L'incertitude de qualification reflète la nouveauté

Le pipeline suppose qu'un `u_qualif` élevé peut servir de signal de nouveauté.

### Risque

Un haut niveau d'incertitude peut venir de plusieurs causes :

- conflit entre groupes ;
- bruit ;
- faiblesse du signal ;
- vraie nouveauté.

### À mettre dans le rapport

- que mesure exactement `u_qualif` ;
- pourquoi on l'interprète comme nouveauté ;
- limites de cette interprétation.

## 5. Hypothèses sur l'évaluation

## H13 - Les expériences synthétiques reflètent la réalité du détecteur

Cette hypothèse ne tient pas complètement, car l'injection est faite au niveau des évidences.

### À mettre dans le rapport

- séparer clairement "évaluation synthétique aval" et "évaluation réelle labelisée" ;
- ne pas les fusionner dans une revendication unique.

## H14 - Les fenêtres d'évaluation sont cohérentes entre tous les scripts

Aujourd'hui, cette hypothèse est fragilisée par les différences entre :

- fréquence dataset ;
- `WINDOW_MIN` global ;
- resampling en `5min` codé en dur ;
- `origin="epoch"` vs `origin="start_day"`.

### À mettre dans le rapport

- définition unique de l'unité de décision ;
- ancrage temporel choisi ;
- justification si différentes unités coexistent selon expérience.

## H15 - Les meilleurs seuils trouvés sur les sweeps décrivent une performance exploitable

Les sweeps de seuil peuvent être très utiles, mais uniquement comme analyse descriptive ou oracle.

### Risque

Si un "best threshold" est choisi sur le même jeu que celui sur lequel la performance est rapportée, on fait de l'optimisation sur test.

### À mettre dans le rapport

- distinguer seuil opérationnel fixé à l'avance ;
- seuil oracle descriptif ;
- seuil sélectionné sur validation.

## 6. Menaces à la validité à déclarer explicitement

## Validité interne

Menaces principales :

- évaluation sur toute la série au lieu du seul test ;
- contamination du train ;
- calibration d'un benchmark via labels du test ;
- divergence entre pipeline calibré et pipeline déployé.

## Validité de construit

Menaces principales :

- CBF utilisée sans indépendance démontrée ;
- interprétation de `u_qualif` comme nouveauté ;
- usage du terme "precision" pour des métriques conditionnelles non standard.

## Validité externe

Menaces principales :

- signatures injectées proches des règles du qualifier ;
- datasets aux sémantiques très différentes ;
- timestamps reconstruits artificiellement sur certains jeux.

## Validité statistique

Menaces principales :

- absence éventuelle d'intervalle de confiance ;
- choix de seuils sur le même jeu que le reporting ;
- déséquilibre de classes fort selon dataset.

## Reproductibilité

Menaces principales :

- duplication manuelle de catalogues d'attaque ;
- fichiers standardisés dont l'état réel n'est pas toujours cohérent avec le discours du launcher ;
- hétérogénéité des paramètres de fenêtre selon scripts.

## 7. Tests minimaux à prévoir pour sécuriser ces hypothèses

1. Audit systématique des labels avant et après split pour chaque dataset.
2. Vérification train/test du prétraitement des NaN.
3. Calibration du seuil sur le vrai score final.
4. Évaluation labelisée strictement restreinte au test.
5. Étude de dépendance empirique entre branche Prophet et branche Reconst.
6. Lecture automatique de la vérité terrain synthétique depuis l'artefact d'injection.
7. Tableau de cohérence des fenêtres et des ancrages temporels.

## Conclusion

La plupart des hypothèses du pipeline sont défendables si elles sont :

- testées ;
- limitées ;
- formulées honnêtement.

Le risque aujourd'hui n'est pas d'avoir un pipeline sans logique, mais d'avoir un pipeline dont certaines hypothèses restent implicites ou contredites par l'état réel des données et des scripts d'évaluation. Pour un article sérieux, ces hypothèses doivent devenir une section explicite du rapport, pas un savoir tacite encapsulé dans le code.
