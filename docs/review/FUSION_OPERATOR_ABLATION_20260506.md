# Fusion Operator Ablation - 2026-05-06

**Status note (2026-05-12).** This is an active historical diagnostic for the
fusion-operator decision, not the source of paper-facing headline metrics. The
final complete-run values are in `docs/review/PUBLICATION_TABLES.md` and
`docs/AUDIT_CURRENT_STATUS.md` (`run 2e12261d55a8f975`). In particular, the
dated strict WBF/ABF FPR values `4.31-4.34%` below are retained to explain why
ABF was not shipped, but they must not be cited as the final RedeRio FPR.

## Position

Je suis d'accord avec la direction generale de l'analyse recue, mais pas avec
un accord complet sans nuance.

- ABF est bien le candidat le plus defendable au niveau inter-methode quand
  Prophet et Reconstruction sont traites comme deux sources dependantes.
- WBF reste defendable et production-safe: il pondere par confiance et les
  resultats empiriques restent solides.
- CBF doit rester legacy/ablation: l'addition d'evidence suppose une
  independance que le diagnostic RedeRio ne soutient pas.
- BCF est correct comme generalisation de Dempster, mais sensible au conflit;
  il ne doit pas devenir un choix production sans garde-fous.
- CCF est interessant, mais l'implementation actuelle est une projection sur
  frame singleton. Ce n'est pas une implementation complete hyper-opinion.
- MinBF et MaxBF sont des bornes heuristiques utiles pour l'ablation, pas des
  operateurs de production.

Point important corrige pendant l'audit: l'ancien mode `hierarchical` appelait
la WBF avec `external_weights=[0.5, 0.5]`; comme la WBF multiplie ensuite par
la confiance `(1-u)`, ce mode n'etait pas une vraie contribution egale des deux
branches. Il utilise maintenant une moyenne d'evidence a poids fixes.

## Changements

- Ajout des operateurs inter-methode dans
  `src/sl_ads/core/subjective_logic.py`:
  `abf`, `cbf`, `bcf`, `ccf`, `minbf`, `maxbf`, `hierarchical` via
  dispatcher `fusion_by_mode`.
- Extension de `INTER_METHOD_FUSION` dans `config.py`, `opinions_pipeline.py`
  et `paths.py`.
- Extension du harness `src/sl_ads/ablation/ablation_fusion_mode.py` aux 8
  modes.
- Ajout d'un bypass explicite et borne pour l'ablation:
  `SL_ALLOW_THRESHOLD_FUSION_MISMATCH_FOR_ABLATION=1`. La production reste
  stricte sur le sidecar de calibration.
- Ajout de tests unitaires pour les nouveaux operateurs et le bypass
  d'ablation.

## Verification

Commandes executees:

```powershell
python -m py_compile src\sl_ads\core\subjective_logic.py src\sl_ads\core\opinions_pipeline.py src\sl_ads\ablation\ablation_fusion_mode.py src\sl_ads\config.py
python -m pytest tests\test_sl_operators_extended.py tests\test_fusion_wbf_canonical.py tests\test_config_and_sidecar.py -p no:cacheprovider
python src\sl_ads\ablation\ablation_fusion_mode.py --modes wbf abf hierarchical cbf bcf ccf minbf maxbf --out-csv results\ablation_fusion_mode_20260506_rerun.csv
```

Resultats de verification:

- Compilation Python: OK.
- Tests unitaires cibles: `88 passed in 0.61s`.
- Ablation complete: 8/8 modes en `returncode=0`.

Le CSV valide est `results/ablation_fusion_mode_20260506_rerun.csv`.
Un fichier `results/ablation_fusion_mode_20260506.csv`, s'il est present, vient
d'une premiere execution interrompue par mismatch sidecar/encoding et ne doit
pas etre utilise.

Verification stricte ajoutee le 2026-05-07:

```powershell
python -m sl_ads.ablation.compare_recalibrated_fusion_modes --modes wbf,abf --from-step opinions
```

Cette execution recalibre un sidecar par mode, puis relance `opinions` et
`eval_injection` sans bypass `SL_ALLOW_THRESHOLD_FUSION_MISMATCH_FOR_ABLATION`.
Elle a ete relancee avec plots desactives (`SL_SKIP_OPINION_PLOTS=1`,
`SL_SKIP_EVAL_PLOTS=1`) pour eviter la commande complete de plus de 6 h.
L'artefact valide est
`current_version/results/fusion_mode_recalibrated/20260507_110115/`.

## Resultats

Ces chiffres utilisent le seuil calibre WBF comme reference fixe. Ils servent a
comparer les modes sous le meme seuil, pas a publier une garantie FPR finale
pour chaque operateur sans recalibration.

| Mode | F1 binary | F1 coverage | Precision window | Recall binary | Recall coverage | FPR (%) | MCC | Median TTD (min) | Attacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wbf | 0.976 | 0.920 | 0.954 | 1.000 | 0.888 | 0.240 | 0.882 | 10.0 | 14/14 |
| abf | 0.976 | 0.920 | 0.954 | 1.000 | 0.888 | 0.240 | 0.882 | 10.0 | 14/14 |
| hierarchical | 0.976 | 0.920 | 0.954 | 1.000 | 0.888 | 0.240 | 0.882 | 10.0 | 14/14 |
| cbf | 0.973 | 0.919 | 0.947 | 1.000 | 0.893 | 0.280 | 0.884 | 10.0 | 14/14 |
| bcf | 1.000 | 0.872 | 1.000 | 1.000 | 0.772 | 0.000 | 0.760 | 12.5 | 14/14 |
| ccf | 0.999 | 0.922 | 0.998 | 1.000 | 0.856 | 0.010 | 0.844 | 12.5 | 14/14 |
| minbf | 0.999 | 0.895 | 0.998 | 1.000 | 0.812 | 0.010 | 0.787 | 12.5 | 14/14 |
| maxbf | 0.828 | 0.797 | 0.707 | 1.000 | 0.915 | 2.220 | 0.781 | 10.0 | 14/14 |

## Interpretation

WBF, ABF et hierarchical sont empiriquement identiques sur cette premiere
campagne a seuil fixe. Comme ABF est plus coherent avec la dependance
Prophet/Reconst, il devient le candidat theorique principal a tester sous
recalibration stricte. Les chiffres ci-dessus ne suffisent pas a changer le
defaut production, car ils reemploient le seuil WBF.

BCF, CCF et MinBF donnent des FPR tres faibles, mais au prix d'une couverture
temporelle plus faible et d'un TTD median plus lent. MaxBF confirme le scenario
attendu: recall maximal mais FPR trop eleve.

## Recalibration stricte WBF vs ABF - 2026-05-07

Deux controles ont ete executes pour eviter le biais "ABF au seuil WBF":

1. Recalibration instantanee par mode:
   WBF et ABF obtiennent le meme seuil `0.102614` et les memes metriques
   (`F1=0.8672`, `MCC=0.8593`, `FPR=0.97 %`). Ce mode est utile comme smoke
   test, mais il ne rejoue pas l'ageing complet.
2. Recalibration avec ageing par mode:
   WBF obtient `threshold=0.059663`, ABF obtient `threshold=0.059518`, puis
   chaque mode est evalue avec son propre sidecar.

Resultat strict avec ageing:

| Mode | Threshold | Precision | Recall | Coverage | F1 | MCC | FPR (%) | VUS-PR | VUS-ROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wbf | 0.059663 | 0.5658 | 1.0000 | 0.9491 | 0.7057 | 0.7087 | 4.31 | 0.603828 | 0.856201 |
| abf | 0.059518 | 0.5644 | 1.0000 | 0.9491 | 0.7046 | 0.7077 | 4.34 | 0.603942 | 0.856201 |

Par famille d'attaque, les deux modes detectent `14/14` attaques. Slowloris
reste le cas difficile (`coverage=78.1 %`, `TTD=60 min`) et les inconnues sont
detectees (`coverage=95.8 %`, `TTD=5 min`) dans les deux modes. Le seul ecart
mesurable est le FPR normal, legerement meilleur pour WBF.

Decision: **ne pas basculer le defaut vers ABF** sur RedeRio. ABF reste
implemente, configurable et scientifiquement defensible pour d'autres datasets
ou pour des groupes de methodes plus clairement dependants. Sur cette evaluation
stricte, WBF est au moins aussi bon et garde l'avantage pratique des poids.

Limite importante: les seuils recalibres sur le holdout normal ne tiennent pas
la cible `FPR_TARGET_DECISION=0.001` sur l'evaluation stricte (`4.31-4.34 %`
avec ageing). Ces sidecars sont donc des artefacts d'ablation, pas un nouveau
seuil production. Le mode WBF par defaut retombe sur le sidecar generique
production du run complet
`trained_models_RedeRio_trained_v4s_v4_v3_threshold.json`
(`decision_threshold=0.10261397856924838`).

Guardrail applique apres relecture: `recalibrate_fusion_thresholds.py` n'ecrit
plus le sidecar generique sauf option explicite `--write-generic`, et
`compare_recalibrated_fusion_modes.py` restaure les sidecars preexistants apres
l'ablation. Cela evite qu'un test WBF/ABF remplace silencieusement le seuil de
production.

## Sources

- A. Josang, [*Subjective Logic: A Formalism for Reasoning Under Uncertainty*](https://link.springer.com/book/10.1007/978-3-319-42337-1),
  Springer, 2016, Chapitre 12.
- A. Josang, J. Diaz, M. Rifqi, ["Cumulative and averaging fusion of beliefs"](https://doi.org/10.1016/j.inffus.2009.05.005),
  *Information Fusion*, 2010.
- A. Josang, D. Wang, J. Zhang, ["Multi-source fusion in subjective logic"](https://doi.org/10.23919/ICIF.2017.8009820),
  FUSION 2017.
- R. W. van der Heijden, H. Kopp, F. Kargl, ["Multi-Source Fusion Operations in Subjective Logic"](https://arxiv.org/abs/1805.01388),
  FUSION 2018 / arXiv `1805.01388`.
