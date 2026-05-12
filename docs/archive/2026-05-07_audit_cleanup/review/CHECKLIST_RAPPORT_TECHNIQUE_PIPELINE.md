# Checklist pour le Rapport Technique et le Papier

## Usage de ce document

Cette checklist sert à préparer un rapport technique scientifiquement défendable. Elle est orientée "ce qu'un reviewer ou un lecteur exigeant demandera".

Le principe recommandé :

1. décrire exactement ce que fait le code ;
2. séparer les résultats robustes des résultats exploratoires ;
3. déclarer explicitement les limites et hypothèses ;
4. éviter toute formulation plus forte que le protocole réellement exécuté.

## 1. Ce qui doit impérativement apparaître dans le rapport

## 1.1. Provenance des données

Inclure un tableau par dataset avec :

- source originale ;
- type de labels disponibles ;
- granularité temporelle native ;
- transformations de standardisation ;
- fichier effectivement utilisé par le pipeline ;
- nombre de fenêtres total ;
- nombre de labels positifs total ;
- nombre de labels positifs avant et après `split_date`.

### Note spécifique issue de l'audit local

Dans le workspace analysé :

- `data_standardized/METR_LA.csv` contient actuellement 0 labels positifs ;
- GECCO et CESNET ont des positifs déjà présents avant `split_date`.

Il faut impérativement que le rapport reflète l'état réel des fichiers utilisés, pas uniquement l'intention du design.

## 1.2. Définition exacte de l'unité de décision

Le rapport doit fixer sans ambiguïté :

- la fréquence d'observation ;
- la taille de fenêtre ;
- l'ancrage du resampling ;
- la fréquence finale du score évalué ;
- si cette fréquence dépend du dataset ou non.

Ajouter un tableau simple :

| Dataset | Fréquence native | Fenêtre SL | Fréquence score final | Fréquence évaluation |
|---|---|---|---|---|

## 1.3. Politique de gestion des valeurs manquantes

Décrire explicitement :

- traitement train ;
- traitement inférence ;
- justification métier.

Si le comportement diffère entre les deux, il faut le signaler comme limitation.

## 1.4. Chaîne de décision complète

Décrire clairement les étapes :

1. prétraitement ;
2. modèles de base ;
3. résidus ;
4. mapping résidu -> évidence P/S/N ;
5. ageing temporel ;
6. injection des priors EDP ;
7. discounting éventuel ;
8. fusion intra-branche ;
9. fusion inter-branche ;
10. score final et seuil.

Le lecteur doit pouvoir comprendre sur quel objet exact le seuil est appliqué.

## 1.5. Distinction stricte entre expériences

Séparer dans le rapport trois familles d'expériences :

- évaluation réelle labelisée ;
- stress test synthétique au niveau des évidences ;
- qualification experte SBN.

Ne pas mélanger ces trois axes dans un seul chiffre global.

## 2. Formulations recommandées et formulations à éviter

## À utiliser

- "évaluation hold-out sur la période test" si et seulement si le code filtre réellement le test ;
- "stress test aval au niveau des évidences" pour l'injection actuelle ;
- "oracle threshold analysis" pour les meilleurs seuils trouvés a posteriori ;
- "conditional classification accuracy among detected attack windows" pour les métriques de qualification injectée si elles restent définies ainsi ;
- "expert-driven qualification layer" pour le SBN.

## À éviter

- "end-to-end synthetic evaluation" pour l'injection actuelle ;
- "fair baseline comparison" si le benchmark utilise les labels du test pour calibrer la baseline ;
- "adaptive base rate" si le prior est en pratique statique par métrique ;
- "precision" sans définition, lorsque la quantité mesurée n'est pas la précision standard sur l'ensemble du temps.

## 3. Sections minimales à ajouter dans le rapport

## 3.1. Menaces à la validité

Section obligatoire avec au moins :

- contamination possible du train ;
- dépendance non démontrée entre branches fusionnées en CBF ;
- injection réalisée au niveau des évidences ;
- caractère expert et non appris du qualifier SBN ;
- éventuelle incohérence de labels ou de fenêtres selon dataset.

## 3.2. Protocole d'évaluation

Préciser explicitement :

- quelle période sert à entraîner ;
- quelle période sert à calibrer ;
- quelle période sert à évaluer ;
- comment le seuil opérationnel est déterminé ;
- si les sweeps servent à sélectionner ou seulement à illustrer.

## 3.3. Reproductibilité

Fournir :

- version exacte du code ;
- hash ou archive des CSV standardisés ;
- fichier de config exact ;
- `split_date` par dataset ;
- colonnes actives par dataset ;
- seeds et dépendances ;
- convention des sorties.

## 4. Tableaux et figures à prévoir

## Tableaux recommandés

1. Tableau des datasets et des labels réels.
2. Tableau des métriques actives par dataset.
3. Tableau des règles de reconstruction avec justification.
4. Tableau des hypothèses de fusion SL.
5. Tableau des menaces à la validité et mitigations.
6. Tableau séparant résultats opérationnels et résultats oracle.

## Figures recommandées

1. Schéma du pipeline complet.
2. Timeline train / calibration / test par dataset.
3. Figure du score final sur un cas réel.
4. Figure distincte pour l'injection synthétique, clairement étiquetée comme expérience aval.
5. Courbes seuil -> F1/FPR uniquement comme analyse exploratoire, pas comme métrique principale.

## 5. Ce qu'il faut verrouiller avant d'afficher des chiffres "principaux"

Avant de transformer un résultat en chiffre principal de résumé, vérifier :

- l'évaluation est strictement faite sur le test ;
- le seuil rapporté correspond bien au score final réellement déployé ;
- la définition de la fenêtre est cohérente entre score et labels ;
- le dataset utilisé contient bien les labels attendus ;
- la baseline de comparaison n'utilise pas le test pour se calibrer ;
- le chiffre n'est pas un oracle présenté comme opérationnel.

## 6. Checklist de formulation pour chaque type de résultat

## Détection labelisée

Toujours préciser :

- dataset ;
- période évaluée ;
- nombre d'anomalies ;
- seuil utilisé ;
- métrique fenêtre-niveau ;
- métrique épisode-niveau ;
- si le train est exclu de l'évaluation.

## Injection synthétique

Toujours préciser :

- niveau d'injection ;
- catalogue utilisé ;
- métriques perturbées ;
- fait que l'expérience ne couvre pas la production des évidences à partir des données brutes.

## Qualification SBN

Toujours préciser :

- si la qualification repose sur règles expertes ;
- comment le ground truth de type est défini ;
- si l'expérience est réelle ou injectée ;
- si la taxonomie testée est proche de celle utilisée pour construire les règles.

## 7. Revendications possibles après correction des points critiques

Si les points critiques sont corrigés, le pipeline pourrait soutenir des revendications du type :

- détection séquentielle multi-méthodes avec fusion Subjective Logic ;
- étude d'ablation des briques de fusion ;
- comparaison avec baselines sur protocole temporel strict ;
- qualification experte interprétable.

## 8. Revendications à ne pas faire tant que les points critiques restent ouverts

- "Le système généralise bien sur datasets labelisés."
- "Le seuil est calibré automatiquement sur le système final."
- "Les expériences synthétiques prouvent l'efficacité end-to-end."
- "La comparaison IF est équitable et directement publiable."
- "METR-LA démontre la détection supervisée réelle" tant que le CSV utilisé contient 0 positifs.

## 9. Ordre conseillé pour la suite

1. Corriger ou au minimum cadrer l'évaluation hold-out.
2. Auditer et figer les datasets réellement utilisés.
3. Aligner calibration et déploiement du score final.
4. Unifier la définition des fenêtres et du resampling.
5. Refaire les résultats principaux après ce nettoyage protocolaire.
6. N'écrire le texte du papier qu'après cette stabilisation.

## Conclusion

Le rapport technique doit être écrit comme une preuve de maîtrise du protocole, pas seulement comme une description d'algorithmes. Si tu suis cette checklist, tu éviteras les deux critiques les plus dures des reviewers :

- "le protocole n'est pas propre" ;
- "les revendications dépassent ce que les expériences démontrent".
