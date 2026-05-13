# ==========================================
# CROSS-DOMAIN DATASETS CONFIGURATION (Ajout)
# ==========================================
# Changer ACTIVE_DATASET ici pour sélectionner le dataset.
# En production, passer --dataset à run_full_sl_ads.py → variable d'env SL_ACTIVE_DATASET.
ACTIVE_DATASET = "RedeRio"           # Réseau UFRJ (brésilien, 15 métriques)
#ACTIVE_DATASET = "CESNET-TimeSeries24"
#ACTIVE_DATASET = "METR-LA"           # Trafic routier Los Angeles
#ACTIVE_DATASET = "GECCO-IoT"        # Capteurs eau potable (pH, chlorine, turbidity)

# VERSION_SUFFIX : suffixe ajouté à tous les noms de version (ex: "_v2", "_exp1").
# Laisser vide ("") pour le comportement par défaut.
# Permet de lancer plusieurs expériences en parallèle sans écraser les résultats.
VERSION_SUFFIX = "_v4_v3"    # ex: "_v2" → trained_models_v9_v9_v4s_v2, METR_LA_v1_v2

import os as _os_cfg
_env_dataset = _os_cfg.environ.get('SL_ACTIVE_DATASET', '').strip()
_env_version_suffix = _os_cfg.environ.get('SL_VERSION_SUFFIX', None)
if _env_version_suffix is not None:
    VERSION_SUFFIX = _env_version_suffix
if _env_dataset:
    ACTIVE_DATASET = _env_dataset   # override par variable d'environnement (parallélisme)

# Seuil de décision pour le consensus de pseudo-labellisation (ex: 2/3 des algos)
ADAPTER_VOTE_THRESHOLD = 2

# PATCH TASK-40 (audit_codex MAJ-04, 2026-04-27).  Pseudo-label policy.
#
# STL_FAIL_POLICY :
#   - 'raise'   : raise on any STL decomposition failure (default).  This
#                 is the production policy — silent zero votes biased the
#                 consensus toward NORMAL and inflated specificity.
#   - 'abstain' : on STL failure, return NaN votes for the affected slice
#                 so the consensus arithmetic skips them via np.nansum.
#                 Use when running ablations on synthetic series where
#                 occasional decomposition failures are expected.
STL_FAIL_POLICY = 'raise'

# Per-dataset metric-vote threshold for the multi-metric consensus loop
# in ``rederio_adapter.apply_pseudo_labels``.  Previously hardcoded as
# the local constant ``METRIC_VOTE_THRESHOLD = 5`` (audit_codex MAJ-04);
# now declared centrally so ablations can sweep it.
REDERIO_METRIC_VOTE_THRESHOLD = 5

# Paramètres de sélection des métriques
METRIC_SELECTION_METHOD = "pearson" # Options: "pearson", "mi" (Mutual Information)
MAX_METRICS_TO_KEEP = 10

if ACTIVE_DATASET == "CESNET-TimeSeries24":
    SELECTED_FILE = "../data_standardized/CESNET.csv"
    SELECTED_SPLIT = "2024-01-29 00:00:00" # 4 semaines
    #SELECTED_SPLIT = "2024-03-25 00:00:00"  # 12 semaines de train (3 mois)
    SELECTED_FREQ = "10min"
    SELECTED_WINDOW = 1 # CESNET est déjà agrégé par pas de 10min
elif ACTIVE_DATASET == "GECCO-IoT":
    SELECTED_FILE = "../data_standardized/GECCO.csv"
    # Le dataset va d'Août à fin Septembre. On garde Août pour l'entraînement.
    #SELECTED_SPLIT = "2016-09-01 00:00:00"
    SELECTED_SPLIT = "2016-10-01 00:00:00"  # 2 mois de train
    SELECTED_FREQ = "1min"
    SELECTED_WINDOW = 5
elif ACTIVE_DATASET == "METR-LA":
    SELECTED_FILE = "../data_standardized/METR_LA.csv"
    # METR-LA va de Mars à fin Juin 2012. 4 semaines d'entraînement = début Avril.
    SELECTED_SPLIT = "2012-04-01 00:00:00"
    SELECTED_FREQ = "5min"
    SELECTED_WINDOW = 1
else:   # "" (vide) ou "RedeRio" → dataset brésilien UFRJ
    # Fichier standardisé (avec pseudo-labels ConsensusLabeller) si disponible
    SELECTED_FILE = "../data_standardized/RedeRio.csv"
    SELECTED_SPLIT = "2025-11-09 23:59:59"
    SELECTED_FREQ = "30s"
    SELECTED_WINDOW = 10

# Dictionnaire central pour orchestrer les datasets
DATASETS_CONFIG = {
    "RedeRio": {
        "active": True,
        "path_raw": "../data/dataset_1310_2912_v30s.csv",   # fichier unique (pas de dossier)
        "path_out": "../data_standardized/RedeRio.csv",
        "has_labels": False,
        "seasonality_period": 2880,  # 30s × 2880 = 86400s = 1 jour
        "metrics": "auto",
        "needs_injection": True      # pseudo-labelling multi-métrique
    },
    "CESNET-TimeSeries24": {
        "active": False,
        "path_raw": "../data/cesnet/raw/",
        "path_out": "../data_standardized/CESNET.csv",
        "has_labels": False,        # <--- METTEZ FALSE ICI
        "seasonality_period": 144,  # CESNET fréq=10min → 1440min/10min = 144 périodes/jour
        "metrics": "auto",
        "needs_injection": True     # <--- METTEZ TRUE ICI
    },
    "METR-LA": {
        "active": True,
        "path_raw": "../data/METR-LA/raw/",
        "path_out": "../data_standardized/METR_LA.csv",
        "has_labels": True,    # 1843 anomalies réelles (col 'label') — pic la nuit, likely sensor faults
        "eval_mode": "labeled",  # évaluation sur vraies étiquettes (compare_if, ablation_labeled)
        "seasonality_period": 288,
        "metrics": "auto",
        "needs_injection": False,  # injection désactivée — évaluation sur labels réels uniquement
    },
    "GECCO-IoT": {
        "active": False,
        "path_raw": "../data/gecco/raw/",
        "path_out": "../data_standardized/GECCO.csv",
        "has_labels": True,
        "seasonality_period": 1440, # 1 min intervals = 1440/day
        "metrics": ["pH", "chlorine", "turbidity"],
        "needs_injection": False
    }
}

CONFIG = {
    # ==============================================================================
    # 1. CONFIGURATION DU DATASET ET DU TEMPS
    # ==============================================================================
    # "file_path": "data/toutesMtric_entropyGood_fusionTrieeNaNTrain_1310_1412.csv",

    "file_path": SELECTED_FILE,
    "split_date": SELECTED_SPLIT,
    "freq_data": SELECTED_FREQ,
    "WINDOW_SIZE": SELECTED_WINDOW,
    # --------------------------------

    "QUALIFY_VERBOSE": False,  # True pour debug uniquement

    # Full-run integrity guard. A production/paper pipeline run must not
    # silently continue with a partial model artifact, for example 5
    # reconstruction leaves only after all Prophet fits failed. Leave this
    # enabled for all reportable runs; set False only for explicit debugging
    # ablations whose outputs are labelled as partial.
    "REQUIRE_ALL_MODELS": True,

    # ─────────────────────────────────────────────────────────────────
    # RANDOM_SEED — graine maîtresse pour la reproductibilité multi-run
    # ─────────────────────────────────────────────────────────────────
    # Référence : Wu & Keogh (2021) flaw #4 — single-seed evaluation.
    # Convention :
    #   - None        ⇒ seed déterministe lue depuis l'env var SL_RANDOM_SEED
    #                   (par défaut 0 si absente).  Mode "single-seed reportable".
    #   - int ∈ [0,9] ⇒ override direct (utilisé par le runner multi-seed).
    #
    # Le runner ``src.sl_ads.evaluate.multi_seed`` boucle sur k=5 graines
    # ({0,1,2,3,4}) en propageant cette valeur via SL_RANDOM_SEED ; la
    # variance inter-seed est ensuite agrégée pour fournir un écart-type
    # sur les métriques publiées (NeurIPS / USENIX-Sec convention).
    "RANDOM_SEED": None,

    # ------------------------------------------------------------------
    # PATCH m-07 / F25 — politique NaN unifiée train/inference
    # ------------------------------------------------------------------
    # `NAN_FFILL_LIMIT` = nombre maximal de tranches consécutives (au pas
    # `freq_data`) qu'on autorise à combler par forward-fill.  Au-delà,
    # la valeur reste NaN et sera ignorée en aval.
    #
    # Interprétation pratique (exemple RedeRio, freq_data=30s) :
    #     NAN_FFILL_LIMIT=10  ⇒  trou ≤ 10 × 30 s = 5 min imputé par la
    #     dernière mesure valide ;  trou > 5 min → la valeur manquante
    #     est préservée et compute_evidence retourne u=1 pour la fenêtre.
    #
    # Effet de bord à documenter (Threats to Validity) :
    # toute anomalie strictement plus courte que NAN_FFILL_LIMIT × freq_data
    # peut être lissée si elle coïncide avec un trou de capture.  Le choix
    # par défaut (10) est un compromis entre robustesse aux trous courts
    # (drops réseau, restart capteur) et préservation des anomalies
    # transitoires observables à l'échelle de la fenêtre SL (WINDOW_SIZE).
    #
    # Appliqué UNIQUEMENT aux colonnes métriques (non timestamp / labels)
    # via preprocessing_utils.preprocess_metrics().  La politique stricte
    # "jamais fillna(0)" reste en vigueur : 0 byte ≠ absence de donnée.
    #
    # Ablation : ablation_nan_ffill.py lance un sweep sur ce paramètre
    # (valeurs recommandées : {0, 5, 10, 20, 30}).  Il surcharge la valeur
    # ci-dessous via la variable d'environnement SL_NAN_FFILL_LIMIT_OVERRIDE
    # (lue en bas de ce fichier, voir section "Overrides env var").
    "NAN_FFILL_LIMIT": 10,

    #"WINDOW_SIZE": 10,

    # --- MODE TUNING RAPIDE ---
    # Si True, ne lance l'analyse que sur la période critique pour aller vite.
    "TUNING_MODE": False,
    "TUNING_START": "2025-11-10 00:00:00",
    "TUNING_END": "2025-11-18 00:00:00",

    # Noms de version centralisés
    "VERSION_NAME": "RedeRio_trained_v4s",
    "VERSION_NAME_MODIF": "RedeRio_trained_v4s_attacks",
    "RESULTS_DIR": "../results/resultats_RedeRio_trained_v4s",
    "EVIDENCE_CSV_NAME": "evidence_RedeRio_trained_v4s_attacks.csv",
    "METADATA_CSV_NAME": "metadata_RedeRio_trained_v4s.csv",

    # Uncertainty-Maximisation (Section 3.6, Eq. 3.27 — Jøsang 2016)
    # Si True, chaque opinion feuille est transformée en son équivalent uncertainty-
    # maximisé AVANT la fusion. La probabilité projetée est préservée (invariant),
    # mais les belief masses sont redistribuées vers l'incertitude (au moins un b_i = 0).
    # Pertinent si on considère les métriques comme épistémiques (événement unique),
    # plutôt qu'aléatoires (processus fréquentiste).
    # NOTE : Impact nul sur P(x) → impact attendu sur fusion WBF/CBF (via u et b).
    "UNCERTAINTY_MAXIMIZATION": False,

    "F1_COVERAGE_WEIGHTED": True,  # Pondère TP par coverage_pct (0-1)
    # False = F1 binaire classique

    # Conflict degree computation mode (H1 sensitivity test)
    # "belief_mass"    : original K on b (Jøsang §11 inspiration) — default
    # "projected_prob" : K on projected probabilities P(x) = b + a·u (includes uncertainty)
    # "kl_symmetric"   : symmetric KL divergence on P (formally maximally rigorous)
    "CONFLICT_MODE": "belief_mass",  # change to test H1 variants

    # KL normalisation temperature (only used when CONFLICT_MODE = "kl_symmetric")
    "CONFLICT_KL_TAU": 1.0,

    "RECONST_QUANTILE": 0.5,  # LAD/médiane — Koenker & Bassett (1978, Econometrica)
                           # Robuste aux outliers résiduels (<50% contamination).
                           # Ajuster vers 0.3 si le dataset a beaucoup d'anomalies
                           # dans la période d'entraînement (RECONST_QUANTILE=0.3).

    # ==============================================================================
    # 2. DÉFINITION DU SYSTÈME DE DÉTECTION
    # ==============================================================================

    # --- MÉTHODE 1 : PRÉDICTION TEMPORELLE (PROPHET) ---

    "ACTIVE_METRICS": [
        # ── Métriques volumétriques (déjà dans pipeline) ──────────────────────
        'bytes',  # R²=0.791 — Stable; signal volumétrique primaire
        'packets',  # R²=0.761 — Stable; fréquence volumétrique
        'flows',  # R²=0.468 — Volatile; taux de connexion
        # ── Métriques TCP/flags (déjà dans pipeline) ──────────────────────────
        'syn',  # R²=0.097 — Chaotic; handshakes TCP (SYN Flood)
        # ── NOUVELLES métriques protocole ─────────────────────────────────────
        'icmp',  # NOUVEAU — discriminateur ICMP Flood vs UDP Flood
        # Ref: UNSW-NB15 (Moustafa & Slay 2015) — feature top-5
        'udp',  # NOUVEAU — confirme UDP Flood (volume UDP anormal)
        # Ref: CIC-IDS2017 (Sharafaldin 2018) — feature discriminante
        'tcp',  # NOUVEAU — base pour ratio fin/syn ; volume TCP
        'fin',  # NOUVEAU — ratio complétion connexion (fin/syn ≈ 1 normal)
        # Ref: Roesch (1999) Snort — fin/syn signature SYN Flood
        # ── Métriques entropie (déjà dans pipeline) ───────────────────────────
        'entropy_src_ip',  # R²=0.064 — Chaotic; diversité source
        'entropy_src_port',  # R²=0.608 — Stable; diversité ports source
        'entropy_dst_port',  # R²=0.254 — Chaotic; diversité ports destination
        'avg_pkt_size',  # R²=0.243 — Chaotic; structure paquet
    ],

    # --- MÉTHODE 2 : RECONSTITUTION STRUCTURELLE (RÉGRESSION) ---
    # Structure : {"target": ..., "feature": ..., "fit_intercept": bool}
    # fit_intercept=False pour les grandeurs extensives liées par une contrainte physique
    # d'origine (0 packets = 0 bytes). Élimine un degré de liberté non physique.
    # Ref : contrainte d'homogénéité dimensionnelle (Bridgman 1922).

    "RECONST_RULES": [
        # ── Paires existantes (validées) ──────────────────────────────────────
        {"target": "bytes", "feature": "packets",
         "comment": "Cohérence volumétrique — VOLUME←VOLUME (Bridgman 1922 ✓)"},
        {"target": "bytes", "feature": "entropy_src_port",
         "comment": "Cohérence comportementale — VOLUME←ENTROPY ✓"},
        {"target": "udp", "feature": "flows",
         "allow_mean_fallback": True,  # R²=0.333 : queue lourde → EVT instable
         "comment": "Fraction UDP des flux. Mean fallback si R²<0 (EVT queue lourde sur résidus sym)."},

        # Paires à R² négatif — modèle dégradé vers la moyenne si R²<0 dans train_v9
        # (voir flag "allow_mean_fallback": True)
        {"target": "fin", "feature": "syn",
         "allow_mean_fallback": True,
         "comment": "Complétion connexion (Roesch 1999 Snort). "
                    "Normal: fin≈syn. SYN Flood: fin<<syn. "
                    "Fallback mean si R²<0 (pas de relation linéaire stable en train normal)."},
        {"target": "tcp", "feature": "packets",
         "allow_mean_fallback": True,
         "comment": "Fraction TCP/paquets. Perturbé par ICMP Flood (tcp/packets↓). "
                    "Fallback mean si R²<0."},
    ],

    # C3 — Source reliability weighting mode (Jøsang §14.3 / §12.5)
    # "uniform"        : WBF native pure (pas de poids externes, pas de discounting)
    #                    ← DÉFAUT PRODUCTION + RÉFÉRENCE PAPER (run 2026-04-29 :
    #                      F1_micro=0.784, 14/14 attaques détectées, FPR=1.64%).
    #                      Couverture complète des 17 métriques (12 Prophet + 5 Reconst).
    # "trust_discount" : probability-sensitive trust discounting (Def. 14.6) [OPT-IN, LEGACY].
    #     t_i = fraction fenêtres Safe en train → appliqué avant WBF, externe_weights=None.
    #     Les scores sont calculés par train_v10.py et stockés dans le .pkl (trust_scores).
    #     PATHOLOGY DOCUMENTÉE : sur RedeRio, 5/12 modèles Prophet ont R²<0
    #     (prophet_syn=-2.851, prophet_tcp=-1.526, etc.).  Le trust-discount utilise R²
    #     comme proxy de confiance → assigne du poids au bruit pur, dégrade F1 à 0.566
    #     vs 0.811 en uniform.  Voir docs/audit/trust_discount_r2_analysis.md et
    #     §5.3.3 honest_limitations.md.  À NE PAS utiliser pour les chiffres rapportés.
    # "r2_static"      : poids R² × confiance (legacy, external_weights=r2)
    # "mase"           : trust dérivé de MASE (Hyndman-Koehler 2006).
    #     trust_k = max(TRUST_SCORE_FLOOR, 1 - MASE_TRUST_ALPHA × MASE_k)
    #     Borné dans [floor, 1] : Joesang Def. 14.6 compatible, jamais
    #     d'amplification.  Pour les sources Prophet (temporelles) MASE
    #     est calculé à l'entraînement et persisté dans
    #     ``models_pkg['mase_scores']``.  Pour les sources RANSAC
    #     (non-temporelles) MASE est NaN par construction → fallback
    #     trust=floor (la fiabilité RANSAC reste gouvernée par
    #     ``RECONST_ATTACK_RELIABILITY``, contextual discounting).
    #     Motivation : R²<0 est ambigu (modèle pire que la moyenne)
    #     alors que MASE>1 a une sémantique claire (modèle pire que
    #     persistence triviale Naive-1 = doit être discounté).
    #     Voir ``sl_ads/stats/mase.py`` et
    #     ``docs/audit/trust_discount_r2_analysis.md`` §3 Option B.
    "WBF_WEIGHT_MODE": "uniform",
    # NOTE : WBF_TRUST_SCORES N'EST PLUS DANS config.py.
    # Les scores sont lus depuis models_pkg['trust_scores'] (R² floored)
    # ou models_pkg['mase_scores'] (MASE-derived) selon WBF_WEIGHT_MODE.

    # MASE_TRUST_ALPHA — pente du trust map dérivé de MASE (Hyndman-Koehler 2006).
    # ``trust = max(floor, 1 - MASE_TRUST_ALPHA × MASE)``.  α=1 (default) =
    # interprétation "skill score" : MASE=1 (no-skill point) ⇒ trust = 0
    # (floored).  α<1 = pénalité plus douce ; α>1 = silenciation plus
    # agressive.  Cette constante n'a pas d'effet quand
    # WBF_WEIGHT_MODE != 'mase'.
    "MASE_TRUST_ALPHA": 1.0,

    # PATCH H2 (calendar-aware EVT thresholds) — when enabled, the
    # training step computes a per-regime EVT calibration (ACTIVE vs
    # QUIET buckets defined by sl_ads.calendar.regime) in addition to
    # the legacy global threshold.  The per-regime block is persisted
    # in ``models_pkg[metric]['thresholds_per_regime']`` and consumed
    # by compute_evidence at inference time *if present*.  Default
    # False so the next clean retrain reproduces the legacy behaviour
    # byte-for-byte unless the operator explicitly opts in.
    # Justification : addresses A1.5 (FPR overshoot 4.5× on weekday-day
    # vs 0× elsewhere → annual projection 2.4× the 0.1 % budget).  See
    # ``docs/review/calendar_evt_design.md`` and §5.3.7 of
    # ``honest_limitations.md``.
    "CALENDAR_EVT_ENABLED": False,

    "C3_WEIGHT_MODE": "uniform",   # Niveau 2 : poids WBF intra-méthode

    # RECONST_ATTACK_RELIABILITY — Discounting contextuel de la Reconstruction (Mercier & Denoeux 2006)
    # Contrôle la fiabilité de la branche Reconstruction pour l'hypothèse "attack".
    # Justification : RANSAC surveille les relations structurelles (bytes~packets, etc.).
    # Ces relations restent NORMALES pendant les attaques applicatives (SLOWLORIS, anomalies
    # de comportement) → la Reconst génère de l'évidence "safe" avec certitude élevée,
    # ce qui dilue le signal d'attaque de Prophet via CBF.
    # Le discounting contextuel résout cela : α_safe=1.0 (on fait confiance à son "normal"),
    # α_attack=RECONST_ATTACK_RELIABILITY (on ne lui fait pas confiance pour "pas d'attaque").
    #
    # Calibrage :
    #   1.0 → comportement original CBF (pas de discounting) ← défaut historique
    #   0.1 → 10% de fiabilité pour l'hypothèse attack (recommandé pour SLOWLORIS/UNKNOWN)
    #   0.0 → Reconst ignorée pour tout ce qui concerne "attack" (trop agressif)
    #   → Tester via ablations : runs "cd_alpha_0.05", "cd_alpha_0.10", "cd_alpha_0.20"
    #   → Choisir la valeur qui maximise F1-cov sans dégrader le FPR
    #   → Valeur typique pour attaques mixtes (volumétrique + applicatif) : 0.10-0.20
    "RECONST_ATTACK_RELIABILITY": 1.0,  # "auto"=calibration auto | 0.10=manuel | 1.0=desactive

    # INTER_METHOD_FUSION — mode de fusion entre Prophet et Reconstruction (Niveau 3)
    #
    # PATCH M-11/CBF (revue consolidée § 1.2) : le défaut passe de "cbf" à "wbf".
    # Motivation : les branches Prophet et Reconstruction sont dérivées de la MÊME
    # fenêtre de trafic brute.  L'hypothèse d'indépendance statistique requise par
    # la CBF (Jøsang 2016, Theorem 12.2) n'est pas garantie par construction ; une
    # corrélation non nulle entre les deux branches ferait surestimer l'évidence
    # et donc sous-estimer `u` (voir CONSOLIDATED_AUDIT_REVIEW §1.2 M-11).  La WBF
    # (moyenne pondérée par confiance) reste valide même en présence de
    # corrélation ; elle est donc l'option par défaut sûre.
    #
    # Modes disponibles :
    # "wbf"          : Weighted Belief Fusion (Jøsang Eq. 12.22) — DÉFAUT M-11.
    #                  Moyenne pondérée par confiance (1-u) de chaque branche.
    #                  Robuste aux dépendances entre sources.
    # "cbf"          : Cumulative Belief Fusion (Jøsang Eq. 12.14) — addition
    #                  d'évidence.  Valide uniquement sous hypothèse
    #                  d'indépendance ; conservé en option pour reproductibilité
    #                  des anciens résultats et ablation.
    # "abf"          : Averaging Belief Fusion — opérateur dédié aux sources
    #                  dépendantes ; candidat théorique principal pour le niveau 3.
    # "bcf"          : Belief Constraint Fusion / Dempster — sensible au conflit,
    #                  conservé uniquement pour ablation.
    # "ccf"          : Consensus & Compromise Fusion projeté sur le frame ternaire
    #                  singleton de SL-ADS ; exploratoire.
    # "minbf"        : minimum des croyances par classe ; ablation extrême AND.
    # "maxbf"        : maximum des croyances par classe ; ablation extrême OR.
    # "hierarchical" : moyenne d'évidence avec poids égaux [0.5, 0.5].
    #                  Sémantiquement propre pour 2 groupes distincts.
    #
    # Override runtime :
    # SL_INTER_METHOD_FUSION_OVERRIDE={wbf,abf,cbf,bcf,ccf,minbf,maxbf,hierarchical}
    # (bloc en fin de config.py).
    "INTER_METHOD_FUSION": "wbf",

    # Method-level grouping for the final SL fusion layer.
    #
    # Adding a third method should normally only require:
    #   1. emitting metrics with a new metadata `type`;
    #   2. adding one group below with that type;
    #   3. choosing whether the final INTER_METHOD_FUSION remains ABF/WBF/etc.
    #
    # Current rationale:
    #   - WBF remains the intra-method pooling operator for metric families.
    #   - INTER_METHOD_FUSION controls the method-level operator only.
    #   - ABF is valid for N sources, but it is conservative; WBF may remain
    #     preferable when a future method is independent and should keep a
    #     confidence/quality-weight advantage.
    "FUSION_METHOD_GROUPS": [
        {
            "name": "prophet",
            "metric_types": ["prophet"],
            "output_key": "METHODE_1_PROPHET",
        },
        {
            "name": "reconstruction",
            "metric_types": ["reconstruction"],
            "output_key": "METHODE_2_RECONST",
            "attack_discount_config_key": "RECONST_ATTACK_RELIABILITY",
        },
    ],
    "FUSION_UNKNOWN_METHOD_POLICY": "raise",

    # Threshold sidecars are now mode-specific.  Training/calibration writes
    # one sidecar per mode listed here, so changing only INTER_METHOD_FUSION
    # can safely switch between ABF and WBF without reusing the wrong threshold.
    "THRESHOLD_CALIBRATION_FUSION_MODES": ["wbf", "abf"],

    # --- ZONES D'EXCLUSION (PANNES / MAINTENANCE) ---
    # Périodes du TRAIN où le réseau était HS ou le capteur défaillant.
    # Ces données seront ignorées par l'entraînement pour ne pas polluer le modèle "Normal".
    "TRAIN_EXCLUSIONS": [
        {'start': '2025-10-15 14:19:58', 'end': '2025-10-15 14:23:18', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-21 15:55:00', 'end': '2025-10-22 10:35:00', 'reason': 'MISSING_FILE'},
        {'start': '2025-10-21 20:52:45', 'end': '2025-10-22 15:31:04', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-26 02:00:00', 'end': '2025-10-26 03:00:00', 'reason': 'MISSING_FILE'},
        {'start': '2025-10-28 22:31:50', 'end': '2025-10-28 22:35:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-28 22:40:00', 'end': '2025-10-28 22:41:35', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-28 23:20:54', 'end': '2025-10-28 23:25:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-28 23:48:00', 'end': '2025-10-28 23:50:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 00:05:00', 'end': '2025-10-29 00:07:35', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 00:23:37', 'end': '2025-10-29 00:25:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 00:30:00', 'end': '2025-10-29 00:31:01', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 00:38:05', 'end': '2025-10-29 00:40:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 01:25:00', 'end': '2025-10-29 01:25:07', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 01:33:25', 'end': '2025-10-29 01:35:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 02:04:56', 'end': '2025-10-29 02:05:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 02:55:00', 'end': '2025-10-29 02:58:23', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 03:48:34', 'end': '2025-10-29 03:50:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 03:55:00', 'end': '2025-10-29 03:58:47', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 04:24:10', 'end': '2025-10-29 04:25:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 04:35:00', 'end': '2025-10-29 04:37:39', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 05:18:38', 'end': '2025-10-29 05:20:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 05:25:00', 'end': '2025-10-29 05:28:20', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 05:30:39', 'end': '2025-10-29 05:35:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 06:40:00', 'end': '2025-10-29 06:41:11', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 07:06:34', 'end': '2025-10-29 07:10:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 08:35:00', 'end': '2025-10-29 08:38:28', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 08:40:51', 'end': '2025-10-29 08:47:20', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 09:00:00', 'end': '2025-10-29 09:05:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 09:22:26', 'end': '2025-10-29 09:25:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 11:30:36', 'end': '2025-10-29 11:36:57', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 13:18:12', 'end': '2025-10-29 13:20:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 14:14:18', 'end': '2025-10-29 14:15:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-10-29 14:20:00', 'end': '2025-10-29 14:20:44', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-11-16 05:05:00', 'end': '2025-11-16 05:15:00', 'reason': 'MISSING_FILE'},
        {'start': '2025-11-16 09:04:56', 'end': '2025-11-16 09:13:59', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-11-17 12:35:00', 'end': '2025-11-17 12:50:00', 'reason': 'MISSING_FILE'},
        {'start': '2025-11-17 16:34:10', 'end': '2025-11-17 16:42:09', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-11-22 15:39:05', 'end': '2025-11-22 15:39:11', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-11-22 16:09:05', 'end': '2025-11-22 16:09:12', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-11-22 16:48:47', 'end': '2025-11-22 16:50:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-11-22 16:55:00', 'end': '2025-11-22 16:57:13', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-11-22 17:02:44', 'end': '2025-11-22 17:05:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-11-22 17:10:00', 'end': '2025-11-22 17:11:55', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-11-22 22:09:03', 'end': '2025-11-22 22:10:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-11-22 22:15:00', 'end': '2025-11-22 22:17:39', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-12-16 16:03:16', 'end': '2025-12-16 16:05:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-12-18 00:05:00', 'end': '2025-12-18 00:06:54', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-12-18 03:58:56', 'end': '2025-12-18 03:59:07', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-12-18 05:58:56', 'end': '2025-12-18 05:58:59', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-12-18 06:04:14', 'end': '2025-12-18 06:04:20', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-12-18 07:59:51', 'end': '2025-12-18 07:59:53', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-12-18 09:48:59', 'end': '2025-12-18 09:49:04', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-12-22 18:51:08', 'end': '2025-12-22 18:55:00', 'reason': 'INTER_FILE_GAP'},
        {'start': '2025-12-25 10:15:00', 'end': '2025-12-25 10:30:00', 'reason': 'MISSING_FILE'},

        {'start': '2025-12-16 12:02:00', 'end': '2025-12-17 20:06:30', 'reason': 'MISSING_FILE_12'},
        {'start': '2025-12-22 14:50:00', 'end': '2025-12-22 14:59:30', 'reason': 'MISSING_FILE_12'},
        {'start': '2025-12-17 23:57:00', 'end': '2025-12-17 23:58:00', 'reason': 'MISSING_FILE_12'},
        {'start': '2025-12-18 04:11:00', 'end': '2025-12-18 04:12:00', 'reason': 'MISSING_FILE_12'},
        {'start': '2025-12-18 03:54:30', 'end': '2025-12-18 03:55:30', 'reason': 'MISSING_FILE_12'},
        {'start': '2025-12-18 03:55:30', 'end': '2025-12-18 03:56:30', 'reason': 'MISSING_FILE_12'},
        {'start': '2025-12-18 03:56:30', 'end': '2025-12-18 03:57:30', 'reason': 'MISSING_FILE_12'},
        {'start': '2025-12-18 03:57:30', 'end': '2025-12-18 03:58:30', 'reason': 'MISSING_FILE_12'},
        {'start': '2025-12-18 03:58:30', 'end': '2025-12-18 03:59:30', 'reason': 'MISSING_FILE_12'},
        {'start': '2025-12-18 04:01:00', 'end': '2025-12-18 04:02:00', 'reason': 'MISSING_FILE_12'},
    ],

    "DDOS_ATTACK": [
        {'start': '2025-11-12 18:21:13', 'end': '2025-11-13 10:14:36', 'reason': 'UDP_DDOS_ATTACK'},
    ],

    # --- CALENDRIER SPÉCIFIQUE (Knowledge Injection) ---
    # Prophet nécessite un DataFrame avec colonnes 'ds' (date) et 'holiday'
    # (nom de l'événement).  Cela permet d'apprendre un coefficient
    # spécifique par type de fête si récurrent, ou un coefficient global
    # si on les nomme tous pareil.
    #
    # TASK-15 (audit_verification_tracker.md) — Choix de modélisation :
    # tous les jours sont étiquetés ``University_Closed`` afin que Prophet
    # apprenne **un seul** biais commun aux fermetures.  Le rationale est
    # double :
    # (1) parcimonie statistique — une trace de 78 jours seulement
    #     (Oct-Déc 2025 + 1er janv. 2026) est trop courte pour identifier
    #     un coefficient distinct par jour férié, surtout pour ceux
    #     n'apparaissant qu'une fois ;
    # (2) le dataset provient d'un campus UFRJ : la sémantique
    #     opérationnelle pertinente est "campus fermé" (chute de trafic),
    #     pas la nature légale du jour férié.
    #
    # Couverture vérifiée par ``tests/test_holidays_brazil.py`` contre la
    # référence ``holidays.Brazil()`` : tous les fériés *nationaux*
    # tombant dans la plage du dataset (RedeRio 2025-10-13 → 2025-12-29)
    # sont présents.  Les dates UFRJ-spécifiques (27 oct., 21 nov.,
    # 26-31 déc.) sont ajoutées en complément.
    "HOLIDAYS_LIST": [
        {'ds': '2025-10-12', 'holiday': 'University_Closed'},  # 'Nossa_Senhora'},
        # {'ds': '2025-10-15', 'holiday': 'Dia_do_Professor'},
        {'ds': '2025-10-27', 'holiday': 'University_Closed'},  # 'Servidor_Publico_UFRJ'}, #A REGARDER !
        {'ds': '2025-11-02', 'holiday': 'University_Closed'},  # 'Finados'},
        {'ds': '2025-11-15', 'holiday': 'University_Closed'},  # 'Proclamacao_Republica'},
        {'ds': '2025-11-20', 'holiday': 'University_Closed'},  # 'Zumbi_Palmares'},
        {'ds': '2025-11-21', 'holiday': 'University_Closed'},  # 'Pont_Rio'},
        {'ds': '2025-12-24', 'holiday': 'University_Closed'},  # 'Christmas_Eve'},
        {'ds': '2025-12-25', 'holiday': 'University_Closed'},  # 'Christmas'},
        {'ds': '2025-12-31', 'holiday': 'University_Closed'},  # 'New_Year_Eve'},
        {'ds': '2026-01-01', 'holiday': 'University_Closed'},  # 'New_Year'},
        # Ajoutez les périodes de fermeture de l'université comme un événement unique "University_Closed"
        # pour que Prophet apprenne un biais commun à ces jours.
        {'ds': '2025-12-26', 'holiday': 'University_Closed'},
        {'ds': '2025-12-27', 'holiday': 'University_Closed'},
        {'ds': '2025-12-28', 'holiday': 'University_Closed'},
        {'ds': '2025-12-29', 'holiday': 'University_Closed'},
        {'ds': '2025-12-30', 'holiday': 'University_Closed'},
    ],

    # Paramètres avancés de Prophet pour correspondre au rapport (Section 2.4.1.1)
    "PROPHET_PARAMS": {
        "interval_width": 0.95,  # Intervalle de confiance à 95% (vs 0.80 par défaut) [cite: 1999]
        "changepoint_prior_scale": 0.05,  # Flexibilité face aux changements de tendance
        "holidays_prior_scale": 10.0,  # Flexibilité pour l'impact des jours fériés (10.0 = fort impact autorisé)
        "seasonality_prior_scale": 10.0,
    },
    # Reproducibility guard: Prophet 1.3.0 can mask backend auto-detection
    # failures as "'Prophet' object has no attribute 'stan_backend'" on this
    # Windows/Python 3.13 environment.  The packaged CMDSTANPY backend is the
    # only supported backend for reportable runs, so request it explicitly.
    "PROPHET_STAN_BACKEND": "CMDSTANPY",

    # ==============================================================================
    # 3. PARAMÈTRES DE LA LOGIQUE SUBJECTIVE (SL)
    # ==============================================================================

    "LAMBDA_DECAY": 0.85,  #85,  # ageing des preuves (λ_base)

    # H1 — Facteur d'amplification du conflit (Conflict-Aware Ageing)
    # Formule : λ_dyn = λ_base × max(0, 1 − α × K)
    # α = 1.0 (défaut original) → λ_dyn_min = λ_base × (1 − K_max) > 0 (inertie résiduelle)
    # α = 1/K_max (correction) → λ_dyn = 0 lors d'une contradiction maximale (hard reset)
    #
    # K_max avec W=3 :
    #   b_prev_max = 20/(20+3) = 20/23
    #   b_curr_max = 10/(10+3) = 10/13
    #   K_max = (20/23)×(10/13) = 200/299 ≈ 0.669
    #   α_corrected = 299/200 = 1.495
    #
    # Propriété : λ_dyn → 0 exactement à contradiction maximale, sans inertie résiduelle.
    # Le système effectue un vrai hard reset (mémoire effacée) sur contradiction forte.
    # Ref : dérivation en §5.2.3 du rapport.
    # CONFLICT_ALPHA : recalculé dynamiquement après CONFIG selon WINDOW_SIZE et SL_PARAM_K.
    # Valeur provisoire ci-dessous (RedeRio WINDOW_SIZE=10, SL_PARAM_K=3) — sera écrasée en bas du fichier.
    "CONFLICT_ALPHA": 1.495,
    "LAMBDA_TRANSITION": 0.95,  # mémoire des transitions (nouveau)

    "BALANCE_RATIO": 1.0,  # "auto" = N_prophet/N_reconst ; float = fixe ; 1.0 = désactivé

    "EVT_DECLUSTER_RUN": -1,   # -1 = désactivé (résidus Prophet déjà blanchis)
                                # Davison & Smith (1990) — r=1 : séparation minimale
    "EVT_MIN_PEAKS":     50,   # Relevé de 30 → 50 (Coles 2001 §4.3 recommande ≥50)

    # Métriques dont Prophet doit utiliser le mode 'additive' (signal peut être nul ou négatif).
    # Pour toutes les autres : mode 'multiplicative' (volumes strictement positifs).
    # À redéfinir dans le moteur de substitution cross-domain ci-dessous pour chaque dataset.
    "SEASONALITY_ADDITIVE": [
        'bytes', 'packets', 'flows',   # ← ajouter ces trois
        'syn', 'icmp', 'udp', 'tcp', 'fin',
        'entropy_src_ip', 'entropy_src_port', 'entropy_dst_port', 'avg_pkt_size',
    ],

    # Catalogue d'injections synthétiques.
    # None  → utilise le catalogue brésilien codé dans inject_at_evidence_level.py
    # []    → injection désactivée (copie simple du CSV d'évidence)
    # [...]  → catalogue custom défini ici pour ce dataset
    "ATTACK_CATALOG": None,

    # Constante de la bijection (W ou k) dans la formule u = k / (P+S+N+k).
    # k=2.0 est la valeur standard (Laplace) qui donne une inertie modérée au démarrage.
    # k=1.0 rendrait le système plus "nerveux" (l'incertitude baisse plus vite).
    "SL_PARAM_K": 3.0,  # W=K=3 : valeur canonique pour domaine à 3 états (Jøsang §3.5.2)
    # Aligné avec W_T=K=3 (TransitionMemory), cohérence théorique complète

    # Fraction de t_susp délimitant le début de la zone de pré-suspicion dans le
    # mapping trapézoïdal (compute_evidence_v2.py).
    # t_trapeze_base = T_TRAPEZE_RATIO × t_susp.
    # 0.1 → transition Safe→Suspect sur 10 % de t_susp (montée rapide).
    # Augmenter vers 0.5 pour une montée plus progressive (moins sensible au bruit).
    "T_TRAPEZE_RATIO": 0.1,

    # ── [LEGACY - PATCH M-03, 2026-04-21] ──────────────────────────────────────
    # W dans la bijection de la matrice de transition. Historiquement utilisé par
    # `adaptive_base_rate.py` (archivé dans `un peu old/`). Le pipeline actif
    # utilise un EDP **statique** calculé au training : ce paramètre n'est plus
    # lu en production mais est conservé pour rétro-compatibilité des configs.
    # Si un jour on réintroduit une mise à jour en ligne du base-rate, ce W
    # redeviendra pertinent.
    #
    # Valeur canonique Jøsang Def. 3.9 : W = K = |X| = 3 (états Safe/Susp/Attack).
    "SL_TRANSITION_W": 3.0,

    # ── PATCH M-08 / F11 (2026-04-21) ─────────────────────────────────────────
    # Plafond du facteur d'évidence dogmatique dans opinion_to_evidence().
    # Quand u → 0, le ratio W/u diverge : on plafonne à W * SL_EVIDENCE_MAX_FACTOR
    # pour éviter l'overflow float64 (~1.8e308).
    # Défaut : 1e4 → cap à W * 1e4 = 3e4 pour W=3, suffisant pour tout biais pratique.
    # Augmenter pour traces longues avec accumulation extrême (runs de plusieurs
    # semaines) ; diminuer pour renforcer la calibration numérique.
    "SL_EVIDENCE_MAX_FACTOR": 1e4,

    # Taux de base (Prior Knowledge) [Safe, Suspect, Attack]
    # Utilisé pour projeter la probabilité quand l'incertitude est totale (P = b + a*u).
    # [0.90, 0.09, 0.01] = On suppose a priori que le réseau est sain à 90%, bruité à 9%, attaqué à 1% (5% seuil haut ->changes).
    # C'est un "Domain Knowledge Prior" pour éviter les faux positifs en cas de doute.
    # "SL_PRIOR_A": [0.867,0.10,0.033],
    "SL_PRIOR_A": [1 / 3, 1 / 3, 1 / 3],
    # ==============================================================================
    # 4. CALIBRATION SCIENTIFIQUE (TRAIN ONLY)
    # ==============================================================================

    # --- B. PARAMÈTRES DE SEUILLAGE AUTOMATIQUE (HYBRIDE) ---
    # REMPLACE "SIGMA_THRESHOLDS".
    # Ces paramètres dictent comment train_v4.py calcule les seuils absolus.

    "C3_ONLINE_RMSE_WARMUP": 10,
    # Option 3 : nb de fenêtres de démarrage où toutes les fenêtres sont traitées
    # comme "sûres" pour initialiser le RMSE (évite division par zéro)

    "C3_ONLINE_RMSE_ALPHA": 0.05,
    # Option 3 : facteur d'oubli exponentiel pour le RMSE glissant
    # rmse_t = alpha * rmse_t + (1-alpha) * rmse_{t-1}
    # alpha=0.05 → demi-vie ≈ 14 fenêtres de 5 min ≈ 70 min (cohérent avec C1)

    # ==============================================================================
    # 5. MATRICE DE DÉCISION AUTOMATIQUE (AUTO-CALIBRATION)
    # ==============================================================================
    # Définit les quantiles à appliquer selon la classe de stabilité diagnostiquée.
    # A (Stable)   : R² élevé, Kurtosis faible, CV faible.
    # B (Volatile) : R² moyen ou Kurtosis moyen.
    # C (Chaotic)  : R² faible (<0.4) OU Kurtosis extrême (>10) OU CV élevé.

    # ============================================================
    # THRESHOLD CALIBRATION — deux branches séparées (Prophet / RANSAC)
    # ============================================================
    # Deux paires de quantiles indépendantes, une par branche de détection.
    # RANSAC est calibré sur les inliers (R² structurellement plus élevé,
    # queue des résidus plus serrée) ; Prophet opère sur tous les points.
    # Un réglage par branche permet d'ajuster la sensibilité indépendamment.
    #
    # Toutes les métriques actives sont en régime Classe C sur ce dataset
    # (R²<0.40 ou kurtosis>10 pour la majorité) → quantiles élevés.
    # À recalibrer lors du déploiement sur un nouveau réseau.
    "Q_SUSP_PROPHET": 0.99,  # T_susp Prophet  (top 1% des résidus normaux)
    "Q_ATK_PROPHET": 0.999,  # T_atk  Prophet  (top 0.05%)
    "Q_SUSP_RANSAC": 0.995,  # T_susp RANSAC
    "Q_ATK_RANSAC": 0.9995,  # T_atk  RANSAC   (ajuster si inlier-R² différent)

    # Marge de sécurité : T_atk >= THRESHOLD_SAFETY_MARGIN * T_susp
    "THRESHOLD_SAFETY_MARGIN": 1.10,

    # ============================================================
    # CALIBRATION EVT — seuils par la méthode des excès (POT/GPD)
    # ============================================================
    # Si True : T_susp / T_atk calculés par MLE Grimshaw (1993) sur les excès
    #   au-dessus du seuil POT t0 = quantile(EVT_INIT_QUANTILE).
    #   Théorème P-B-H (1974) : toute queue ≥ t0 converge vers une GPD —
    #   aucune hypothèse distributionnelle sur les résidus n'est requise.
    # Si False : quantiles empiriques np.quantile (comportement original).
    # Ref : Grimshaw (1993) Technometrics 35(2) ; Siffer et al. KDD 2017 (SPOT).
    "USE_EVT_THRESHOLDS": True,

    # Probabilité d'excès inconditionnel cible (dataset-indépendante).
    # Contrairement aux quantiles CDF (0.99, 0.9995…) qui indexent un tableau,
    # ces valeurs expriment une garantie probabiliste portable sur tout dataset.
    # EVT par branche (remplace les anciens Q_SUSP_PROPHET etc. en mode EVT)
    # Probabilités d'excès cibles — interprétation : P(|résidu| > T | normal)
    # Correspondent exactement aux anciens quantiles empiriques : EVT_Q = 1 - Q_old
    "EVT_Q_SUSP_PROPHET": 0.01,
    "EVT_Q_ATK_PROPHET":  0.001,
    "EVT_Q_SUSP_RANSAC":  0.01,
    "EVT_Q_ATK_RANSAC":   0.001,
    "EVT_INIT_QUANTILE": 0.90,   # Quantile t0 pour isoler la queue POT (90e pc)

    # Auto-calibration de DECISION_THRESHOLD
    # Calculé en fin d'entraînement via proj_atk sur fenêtres normales du train.
    # Variable cible : proj_atk = b_atk + a_atk·u (Jøsang Eq. 3.23).
    # Stocké dans models_pkg['_decision_threshold'] et lu par compute_opinions.
    # Ref : voir docs/scientific_deconstruction/REFERENCES.md
    # (hold-out calibration: Ruff et al. 2021; leakage: Varma & Simon 2006).
    "FPR_TARGET_DECISION": 0.001,   # Cible : ≤ 0.1% FP — compense le mismatch EVT in-sample vs calib set hors-train

    #Exemple de différents seuils sur redeio v9v9v4s avec 1/4 pour threshold time
    #FPR 1.0% (10−2) ➔ Seuil = 0.1318
    #FPR 0.1% (10−3) ➔ Seuil = 0.1498 (+0.0180)
    #FPR 0.01% (10−4) ➔ Seuil = 0.1644 (+0.0146)
    #FPR 0.001% (10−5) ➔ Seuil = 0.1672 (+0.0028)

    # Fraction du training réservée à la calibration de DECISION_THRESHOLD.
    # Ces données ne participent PAS au fit Prophet/RANSAC ni à la calibration
    # T_susp/T_atk — elles servent uniquement à estimer la distribution de b_atk
    # en conditions proches de la production (distribution shift naturel).
    # Avec 4 semaines train : 0.25 = 1 sem. calibration, 3 sem. modèles.
    "CALIB_SPLIT_FRACTION": 0.25,  # 25% du train pour calibration DECISION_THRESHOLD (hors-train)

    # ============================================================
    # EMPIRICAL DIRICHLET PRIOR (EDP) — base rate adaptatif
    # ============================================================
    # Remplace le prior uniforme a=[1/3,1/3,1/3] par l'estimateur des
    # moments de la concentration Dirichlet sur les données d'entraînement.
    #
    # Formule (Ferguson 1973 ; Robbins 1955/1983 Empirical Bayes) :
    #   a_j(k) = mean_t[ R_j(k,t) ] / W   pour j ∈ {safe, susp, anom}
    #   W = taille fenêtre = 10 (slices de 30s par fenêtre de 5 min)
    #
    # Propriétés clés :
    #   1. Réduit la FPR cold-start : P(Anom)|u=1 = a_atk ≈ 0.003 << δ=0.20
    #      (vs 0.333 avec le prior uniforme → risque documenté §8)
    #   2. Hétérogénéité inter-métriques encodée dans le prior :
    #      a_atk(syn) > a_atk(bytes) car syn a une queue haute plus lourde
    #   3. Réduction au prior uniforme si évidences parfaitement équilibrées
    #      (principe d'entropie maximale de Jaynes 1957 — propriété d'équivariance)
    #   4. Calculé une fois à l'entraînement depuis evidence_train.csv,
    #      stocké dans l'artefact du modèle, appliqué à tout le test
    #
    # USE_EMPIRICAL_PRIOR = False → prior SL_PRIOR_A (uniforme par défaut)
    # USE_EMPIRICAL_PRIOR = True  → EDP calculé sur les données d'entraînement
    "USE_EMPIRICAL_PRIOR": True,
    "EMPIRICAL_PRIOR_FLOOR": 0.005,  # Plancher sur a_atk (évite prior nul)

    # ============================================================
    # FUZZY TRAPEZE — Option B data-driven (à tester, désactivé par défaut)
    # ============================================================
    # True : début de la zone pré-suspicion = Q(Q_TRAPEZE_BASE) au lieu de
    # 0.9 * T_susp → pente déterminée par la CDF empirique des résidus.
    # Référence : Dubois & Prade (1988), fonctions d'appartenance trapézoïdales.
    "USE_QUANTILE_TRAPEZE": False,
    "Q_TRAPEZE_BASE": 0.95,

    # ============================================================
    # ASYMMETRIC THRESHOLDS (seuils directionnels par métrique)
    # ============================================================
    # 'pos' : anomalie = excès positif (y > yhat) — métriques volumétriques
    # 'neg' : anomalie = déficit      (y < yhat) — métriques structurelles
    # None  : symétrique |résidu|      — comportement historique
    "ASYMMETRIC_THRESHOLD_METRICS": {
        # Prophet — volumétriques bidir (excès = attaque, chute = panne)
        # direction='both' : deux paires de seuils pos/neg calibrées séparément.
        # La chute Slowloris reste sous T_susp_neg (Safe), sans gêner flows.
        "bytes": "both",
        "packets": "both",
        "flows": "both",
        "syn": "pos",  # SYN flood uniquement excès
        # Prophet — structurels (attaque = réduction de diversité/taille)
        "avg_pkt_size": "neg",
        "entropy_src_ip": "neg",
        "entropy_src_port": "neg",
        "entropy_dst_port": "neg",
        # RANSAC — target = bytes (volumétrique)
        "bytes_packets": "pos",
        "bytes_entropy_src_port": "pos",
        # ASYMMETRIC_THRESHOLD_METRICS : ajouter pour le prochain retrain
        "icmp": "pos",  # flood toujours en excès positif
        "udp": "pos",
        "tcp": "pos",
        "fin": "sym",  # peut monter (HTTP) ou descendre (Slowloris)
        "fin_syn": "sym",
        "tcp_packets": "sym",
        "udp_flows": "sym",
    },

    # --- PERFORMANCE (VOTRE MESURE RÉELLE) ---
    # Temps mesuré : 1.136s pour convertir le binaire en CSV
    "TIME_BINARY_TO_CSV": 1.136,

    "INJECTION_SKIP_N_DOMINANT": False,  # Ne pas injecter les métriques N-dominantes : pas de raison de passer a True !

    # ==============================================================================
    # 7. ÉVALUATION (evaluate_injection.py)
    # ==============================================================================
    "EVAL": {
        # Mode : "injected" = catalogue synthétique (inject_at_evidence_level.py)
        #        "real"     = vraie attaque dans REAL_ATTACK_CATALOG ci-dessous
        "CATALOG_MODE": "injected",

        # Dossier résultats à évaluer (relatif au script)
        "RESULTS_DIR": "__AUTO_FROM_TOP_LEVEL__",
        "RESULTS_CSV_NAME": "detection_results.csv",

        # Fenêtre temporelle (minutes) et contexte graphique (heures)
        "WINDOW_MIN": 5,
        "CONTEXT_H": 2.0,

        # Seuils b_atk testés dans le sweep
        "THRESHOLDS": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],

        # [AJOUT] Seuil opérationnel strict (sans optimisation a posteriori)
        # C'est ce seuil qui est rapporté dans l'article comme performance réelle.
        "DECISION_THRESHOLD": 0.15,#0.20,  # P(Atk) seuil opérationnel strict

        # Colonnes du CSV d'opinions
        "COL_BATK": "FINAL_SYSTEM_CBF_b_atk",
        "COL_BSUSP": "FINAL_SYSTEM_CBF_b_susp",
        "COL_BSAFE": "FINAL_SYSTEM_CBF_b_safe",
        "COL_PROJ_ATK": "FINAL_SYSTEM_CBF_proj_atk",

        # Métriques feuilles à auditer (base rates)
        "LEAF_METRICS_TO_AUDIT": [
            "P_bytes", "P_packets", "P_flows", "P_syn",
            "P_entropy_src_ip", "P_entropy_src_port",
            "P_entropy_dst_port", "P_avg_pkt_size",
            "R_bytes_to_packets", "R_bytes_to_entropy_src_port",
        ],

        # Catalogue de la vraie attaque DDoS du dataset de base
        # Format identique à ATTACK_CATALOG dans inject_at_evidence_level.py
        "INCLUDE_REAL_ATTACK": True,
        "REAL_ATTACK_CATALOG": [
            {
                "name": "REAL_DDOS",
                "type": "DDoS",
                "intensity": "extreme",
                "start": "2025-11-12 18:21:13",  # ← à remplir avec la vraie date
                "duration_h": 15.89,  # ← à remplir
                "signature": {
                    "prophet_avg_pkt_size": (5.54, 3.23, 0.24),
                    "prophet_bytes": (0.17, 0.1, 9.0),
                    "prophet_entropy_dst_port": (10.0, 0.0, 0.0),
                    "prophet_entropy_src_ip": (10.0, 0.0, 0.0),
                    "prophet_entropy_src_port": (10.0, 0.0, 0.0),
                    "prophet_flows": (9.38, 0.45, 0.0),
                    "prophet_packets": (1.0, 0.28, 8.13),
                    "prophet_syn": (10.0, 0.0, 0.0),
                    "reconst_bytes_from_entropy_src_port": (1.0, 0.58, 8.0),
                    "reconst_bytes_from_packets": (5.0, 0.74, 4.26),
                }
            }
        ],
        "UNKNOWN_EPISODES_TO_VERIFY": [
            {
                "start": "2025-11-17 12:30:00",
                "end": "2025-11-17 12:45:00",
                "note": "NETWORK_OUTAGE confirmé — interruption réseau abrupte 15 min. "
                        "Tous résidus Prophet en chute (z=-2.9 à -3.8 selon métrique). "
                        "UDP moins affecté (z=-0.83), profil créneau rectangulaire. "
                        "Probable redémarrage équipement. Ajouté dans REAL_ATTACKS.",
            },
            {
                "start": "2025-12-16 12:30:00",
                "end": "2025-12-17 16:45:00",
                "note": "NETWORK_OUTAGE confirmé — dégradation progressive sur 5h (diurne). "
                        "Partie nocturne non détectée par design (Prophet prédit trafic bas la nuit). "
                        "Résidus progressifs z=-3.7 à -4.1, UDP moins affecté (z=-1.3). "
                        "Profil maintenance réseau ou défaillance de lien. Recall=43.5%.",
            },
        ],
    },
    "ABLATION": {
        # Active les runs d'ablation dans compute_opinions_v2.py
        # Chaque clé = un run, chaque run produit son propre CSV de détection
        # PATCH "uniform-as-reference" 2026-04-29 : full_sl + tous les "_isolated"
        # passent en uniform pour matcher la config production (config.py L261).
        # La pathologie trust_discount × R²<0 (cf. trust_discount_r2_analysis.md)
        # est désormais représentée par la variante dédiée "trust_discount_legacy".
        "ENABLED": True,
        "RUNS": {
            # Référence : matche la config production (WBF_WEIGHT_MODE="uniform").
            "full_sl": {"lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
            "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
            "c3_weight_mode": "uniform",    # ← cohérent avec opérationnel
            "wbf_weight_mode": "uniform",
            "inter_method_fusion": "wbf"},   # ← matche INTER_METHOD_FUSION production
            "no_ageing + uniform": {"lambda": 0.00, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
                          "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0},
            "uniform_weights": {"lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
                                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0},
            "prophet_only uniform": {"lambda": 0.85, "wbf_uniform": True, "use_reconst": False, "use_prophet": True},
            "reconst_only uniform": {"lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": False},
            "no_cbf uniform": {"lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
                       "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0, "use_cbf": False},
            "no_c1 uniform": {"lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
                      "conflict_aware": False, "adaptive_base_rate": True, "sl_param_k": 3.0},
            # no_edp_uniform: teste l'effet de l'EDP (Empirical Dirichlet Prior)
            # adaptive_base_rate=False -> utilise SL_PRIOR_A=[1/3,1/3,1/3] (prior uniforme)
            # Ce run mesure la contribution de l'EDP (C4), pas de la TransitionMemory
            # (TransitionMemory a ete remplacee par l'EDP dans train_v9.py)
            "no_edp_uniform": {"lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
                      "conflict_aware": True, "adaptive_base_rate": False, "sl_param_k": 3.0},
            # w3_sensitivity supprime (doublon de uniform_weights) - remplace par w2/w4 sensitivity
            # Voir les runs w2_sensitivity et w4_sensitivity ci-dessous (BUG_016)

            # ── NOUVEAUX — doivent être ICI, dans RUNS ──────────────────
            "c3_prophet_interval uniform": {
                "lambda": 0.85, "wbf_uniform": True,
                "use_reconst": True, "use_prophet": True,
                "use_cbf": True, "conflict_aware": True,
                "adaptive_base_rate": True, "sl_param_k": 3.0,
                "c3_weight_mode": "prophet_interval",
            },
            "c3_online_rmse uniform": {
                "lambda": 0.85, "wbf_uniform": True,
                "use_reconst": True, "use_prophet": True,
                "use_cbf": True, "conflict_aware": True,
                "adaptive_base_rate": True, "sl_param_k": 3.0,
                "c3_weight_mode": "online_rmse",
            },
            # ── RUNS ISOLES (ablation stricte: un seul composant change) ───────────────
            # Chaque run = full_sl MOINS exactement un composant.
            # PATCH 2026-04-29 : la base est uniform (matche full_sl + production).
            # La pathologie trust_discount est représentée par "trust_discount_legacy".
            "no_c1_isolated": {
                "lambda": 0.85, "wbf_uniform": True,
                "use_reconst": True, "use_prophet": True,
                "conflict_aware": False,  # <- seul changement vs full_sl
                "adaptive_base_rate": True, "sl_param_k": 3.0,
                "wbf_weight_mode": "uniform"
            },
            "no_cbf_isolated": {
                "lambda": 0.85, "wbf_uniform": True,
                "use_reconst": True, "use_prophet": True,
                "use_cbf": False,  # <- seul changement
                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
                "wbf_weight_mode": "uniform"
            },
            "prophet_only_isolated": {
                "lambda": 0.85, "wbf_uniform": True,
                "use_reconst": False,  # <- seul changement
                "use_prophet": True,
                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
                "wbf_weight_mode": "uniform"
            },
            "reconst_only_isolated": {
                "lambda": 0.85, "wbf_uniform": True,
                "use_reconst": True,
                "use_prophet": False,  # <- seul changement
                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
                "wbf_weight_mode": "uniform"
            },
            "no_edp_isolated": {
                "lambda": 0.85, "wbf_uniform": True,
                "use_reconst": True, "use_prophet": True,
                "conflict_aware": True,
                "adaptive_base_rate": False,  # <- desactive EDP -> prior uniforme SL_PRIOR_A
                "sl_param_k": 3.0, "wbf_weight_mode": "uniform"
            },

            # ── SENSIBILITE W (bijection SL) ────────────────────────────────────────
            # W=2: valeur Laplace standard (beaucoup de papiers SL)
            # W=3: valeur canonique domaine ternaire (Josang §3.5.2) <- production
            # W=4: valeur conservative (plus de preuve requise avant convergence)
            "w2_sensitivity": {
                "lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 2.0,
                "wbf_weight_mode": "uniform"
            },
            "w4_sensitivity": {
                "lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 4.0,
                "wbf_weight_mode": "uniform"
            },

            # ── SENSIBILITE BALANCE_RATIO ────────────────────────────────────────────
            # Teste l'effet du reeequilibrage des preuves avant CBF (Theorem 12.2)
            # Ref: Josang 2016, CBF = addition de preuves -> biais si N_A != N_B
            "balance_auto": {
                "lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
                "wbf_weight_mode": "uniform", "balance_ratio": "auto",  # N_p/N_r
                "inter_method_fusion": "cbf"  # CBF-only sensitivity; production remains WBF
            },

            # ── CONTEXTUAL DISCOUNTING (sensibilité α_attack pour Reconst) ────────────
            # Tests l'effet de α_attack sur la détection des attaques applicatives.
            # α=1.0 = CBF standard ; α=0.0 = Reconst ignorée pour "attack".
            # Ref : Mercier, Quost & Denoeux (2006) — apply_contextual_discount() dans sl_formulas_v2.
            "cd_alpha_0.00": {
                "lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
                "wbf_weight_mode": "uniform", "reconst_attack_reliability": 0.00
            },
            "cd_alpha_0.05": {
                "lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
                "wbf_weight_mode": "uniform", "reconst_attack_reliability": 0.05
            },
            "cd_alpha_0.10": {
                "lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
                "wbf_weight_mode": "uniform", "reconst_attack_reliability": 0.10
            },
            "cd_alpha_0.20": {
                "lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
                "wbf_weight_mode": "uniform", "reconst_attack_reliability": 0.20
            },
            "cd_alpha_0.50": {
                "lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
                "wbf_weight_mode": "uniform", "reconst_attack_reliability": 0.50
            },

           "wbf_inter_method_isolated": {
                "lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
                "wbf_weight_mode": "uniform", "inter_method_fusion": "wbf"
            },

            # ── COMBINAISONS SLOWLORIS ─────────────────────────────────────────
            "cd_0.10_wbf_inter": {
                "lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
                "wbf_weight_mode": "uniform",
                "reconst_attack_reliability": 0.10, "inter_method_fusion": "wbf"
            },
            "cd_0.20_wbf_inter": {
                "lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
                "wbf_weight_mode": "uniform",
                "reconst_attack_reliability": 0.20, "inter_method_fusion": "wbf"
            },
            # PATHOLOGIE TRUST_DISCOUNT — démontre la régression F1 (0.811→0.566).
            # full_sl est désormais uniform ; cette variante = full_sl AVEC trust_discount
            # activé pour exposer la pathology R²-négatif (5/12 modèles Prophet R²<0).
            # Ref : docs/audit/trust_discount_r2_analysis.md
            "trust_discount_legacy": {
                "lambda": 0.85, "wbf_uniform": False,    # ← seul changement vs full_sl
                "use_reconst": True, "use_prophet": True,
                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
                "c3_weight_mode": "uniform",
                "wbf_weight_mode": "trust_discount",       # PATHOLOGIE — ne pas utiliser
            },

            # PATCH D5 — MASE-Trust legacy variant (Hyndman-Koehler 2006).
            # Documents the *opposite* failure mode of trust_discount: at
            # 30 s sampling Naive-1 is dominant so 11/12 Prophet metrics
            # have MASE > 1 → all Prophet sources are silenced down to
            # TRUST_SCORE_FLOOR.  Included so the canonical ablation
            # matrix carries the comparison uniform vs trust_discount vs
            # mase reproducibly under the same protocol.
            # Ref : docs/audit/trust_discount_r2_analysis.md §4.1.
            "mase_legacy": {
                "lambda": 0.85, "wbf_uniform": False,    # ← seul changement vs full_sl
                "use_reconst": True, "use_prophet": True,
                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
                "c3_weight_mode": "uniform",
                "wbf_weight_mode": "mase",                 # PATCH D5 — opt-in
            },

            # ── FUSION HIÉRARCHIQUE À 2 NIVEAUX (sémantiquement correct en SL) ─────────
            # Prophet (groupe) et Reconst (groupe) = 2 sources distinctes → poids égaux 0.5/0.5.
            # Niveau 1 : WBF intra-groupe (métriques → 1 opinion par groupe, poids uniformes).
            # Niveau 2 : WBF inter-groupes avec external_weights=[0.5, 0.5] (équipondération forcée).
            # Vs balance_ratio : ne surestime pas la certitude (pas de preuves fictives).
            # Vs wbf confidence : ne laisse pas la confiance relative biaiser l'inter-fusion.
            # Ref : Jøsang (2016) §12.3 — fusion de sources hétérogènes à poids fixe.
            "hierarchical_fusion": {
                "lambda": 0.85, "wbf_uniform": True, "use_reconst": True, "use_prophet": True,
                "use_cbf": True,               # intercepté avant CBF par inter_method_fusion
                "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
                "c3_weight_mode": "uniform", "wbf_weight_mode": "uniform",
                "inter_method_fusion": "hierarchical",  # poids égaux 0.5/0.5 entre groupes
            },
        },  # ← RUNS se ferme ici
    },

    # ==============================================================================
    # À AJOUTER dans config.py, dans le dictionnaire CONFIG
    # ==============================================================================

    # --- Évaluation attaque réelle (evaluate_real_ddos.py) ---
    "EVAL_REAL": {
        "THRESHOLD": 0.15,
        "CONTEXT_H": 4.0,
        "LEAF_METRICS": [
            "P_bytes", "P_packets", "P_flows", "P_syn",
            "P_entropy_src_ip", "P_entropy_src_port", "P_entropy_dst_port",
            "P_avg_pkt_size",
            "R_bytes_to_packets", "R_bytes_to_entropy_src_port",
        ],
        # Unifier : utiliser proj_atk partout (cohérent avec la règle D(T)=1 si P(Atk)≥δ)
        "COL_BATK": "FINAL_SYSTEM_CBF_proj_atk",   # ← renommer en COL_DET
        "COL_DET": "FINAL_SYSTEM_CBF_proj_atk",
    },

    # --- Robustesse au bruit de collecte (evaluate_noise_robustness.py) ---
    "NOISE_ROBUSTNESS": {
        "EVIDENCE_CSV_NAME": "__AUTO_FROM_TOP_LEVEL__",
        "METADATA_CSV_NAME": "__AUTO_FROM_TOP_LEVEL__",
        "NOISE_RATES": [0, 1, 2, 5, 10, 15, 20, 30],
        "LAMBDA_VALUES": [0.0, 0.50, 0.85, 0.95, 0.99],
        "N_SEEDS": 5,
        "THRESHOLD": 0.20,
        "NOISE_OUTSIDE_ONLY": True,
    },
}

# ==============================================================================
# INJECTED_ATTACK_CATALOG — source unique de vérité
# ------------------------------------------------------------------------------
# Référence cross-fichier pour :
#   - inject_at_evidence_level.py  (source d'injection)
#   - evaluate_qualify_sbn.py       (ground-truth de qualification)
#   - evaluate_qualify_injected.py  (ground-truth d'évaluation)
#   - compare_qualif_methods.py     (ground-truth de comparaison)
#
# Tout ajout/modification ici se propage automatiquement.
# Violation de DRY historique (3 copies divergentes) corrigée : PATCH-C1.
# ==============================================================================
INJECTED_ATTACK_CATALOG = [
    # PATCH-C1 fix (2026-04-19) : nom aligné sur l'injecteur canonique
    # `inject_at_evidence_level.py` (UNKNOWN_ANOMALY_CONTROL) et sur les
    # évaluateurs downstream (evaluate_qualify_injected.py L.61, audit_full_dataset.py).
    {'name': 'UNKNOWN_ANOMALY_CONTROL', 'expected': None,
     'start': '2025-12-20 10:00:00',  'end': '2025-12-20 12:00:00',
     'intensity': 'high',  'is_novelty_control': True},
    {'name': 'UDP_FLOOD_DDOS',        'expected': 'UDP_FLOOD',
     'start': '2025-11-16 14:00:00',  'end': '2025-11-16 18:00:00',
     'intensity': 'extreme', 'is_novelty_control': False},
    {'name': 'SYN_FLOOD_DDOS',        'expected': 'SYN_FLOOD',
     'start': '2025-11-21 02:30:00',  'end': '2025-11-21 03:15:00',
     'intensity': 'extreme', 'is_novelty_control': False},
    {'name': 'BOTNET_CC_BEACONING',   'expected': 'BOTNET_CC',
     'start': '2025-11-19 08:00:00',  'end': '2025-11-19 12:00:00',
     'intensity': 'low',    'is_novelty_control': False},
    {'name': 'BRUTE_FORCE_SSH',       'expected': 'BRUTE_FORCE_SSH',
     'start': '2025-11-25 14:00:00',  'end': '2025-11-25 17:00:00',
     'intensity': 'medium', 'is_novelty_control': False},
    {'name': 'AGGRESSIVE_PORT_SCAN',  'expected': 'PORT_SCAN',
     'start': '2025-11-28 10:15:00',  'end': '2025-11-28 12:45:00',
     'intensity': 'medium', 'is_novelty_control': False},
    {'name': 'DATA_EXFILTRATION_SLOW','expected': 'DATA_EXFIL',
     'start': '2025-12-02 23:00:00',  'end': '2025-12-03 05:00:00',
     'intensity': 'low',    'is_novelty_control': False},
    {'name': 'NTP_AMPLIFICATION',     'expected': 'NTP_AMP',
     'start': '2025-12-04 08:00:00',  'end': '2025-12-04 11:00:00',
     'intensity': 'extreme','is_novelty_control': False},
    {'name': 'HTTP_FLOOD_L7_DDOS',    'expected': 'HTTP_FLOOD',
     'start': '2025-12-07 16:30:00',  'end': '2025-12-07 18:00:00',
     'intensity': 'high',   'is_novelty_control': False},
    {'name': 'DNS_TUNNELING',         'expected': 'DNS_TUNNELING',
     'start': '2025-12-09 10:00:00',  'end': '2025-12-09 16:00:00',
     'intensity': 'low',    'is_novelty_control': False},
    {'name': 'DNS_AMPLIFICATION',     'expected': 'DNS_AMP',
     'start': '2025-12-11 08:00:00',  'end': '2025-12-11 11:00:00',
     'intensity': 'extreme','is_novelty_control': False},
    {'name': 'SLOWLORIS_DOS',         'expected': 'SLOWLORIS',
     'start': '2025-12-15 22:00:00',  'end': '2025-12-16 06:00:00',
     'intensity': 'low',    'is_novelty_control': False},
    {'name': 'ICMP_FLOOD_BURST',      'expected': 'ICMP_FLOOD',
     'start': '2025-12-18 11:30:00',  'end': '2025-12-18 12:00:00',
     'intensity': 'extreme','is_novelty_control': False},
]


QUALIFY_GROUP_SOURCES = {
    # Convention de nommage : préfixe "prophet_" pour les métriques Prophet,
    # "reconst_" pour les métriques de reconstruction (RANSAC/quantile).
    # Ces préfixes correspondent aux colonnes produites par compute_opinions_v3.py.

    # Groupe volumétrique — signal principal flood
    'volume': ['prophet_bytes', 'prophet_packets'],

    # Groupe protocole TCP/UDP/ICMP — discriminateurs inter-protocoles
    # Correction de la confusion ICMP/UDP via canaux séparés
    'protocol_tcp':  ['prophet_tcp'],
    'protocol_udp':  ['prophet_udp'],
    'protocol_icmp': ['prophet_icmp'],

    # Groupe flags TCP bruts — comptages SYN et FIN de Prophet
    # Note : prophet_syn et prophet_fin sont deux mesures INDÉPENDANTES (l'une ne dérive
    # pas de l'autre). reconst_fin_from_syn (ratio) est séparé dans 'fin_ratio' pour éviter
    # toute dilution par le geomean et pour respecter l'indépendance conditionnelle Naive Bayes.
    # Ref : Roesch (1999) Snort — SYN surge signature
    'tcp_flags': ['prophet_syn', 'prophet_fin'],

    # Groupe ratio fin/syn RANSAC — discriminateur clé SLOWLORIS vs DATA_EXFIL vs SYN_FLOOD
    # Un groupe dédié à une seule métrique élimine la dilution par geomean et isole
    # le signal discriminant. Ref : Roesch (1999) ; Mirsky (2018) Kitsune — Slowloris
    'fin_ratio': ['reconst_fin_from_syn'],

    # Groupe connexions — Slowloris, Port Scan
    'connections': ['prophet_flows'],

    # Groupe entropies — source ip, ports
    'entropy': ['prophet_entropy_src_ip', 'prophet_entropy_src_port', 'prophet_entropy_dst_port'],

    # Groupe structure paquet
    'packet_size': ['prophet_avg_pkt_size'],

    # Groupe reconstruction RANSAC (relations structurelles inter-métriques)
    # Capture les anomalies dans les relations entre volumes, entropies et protocoles.
    # reconst_fin_from_syn est dans 'fin_ratio' (groupe dédié) — absent ici pour
    # garantir l'indépendance conditionnelle des groupes (Naive Bayes, Rish 2001).
    'reconstruction': ['reconst_bytes_from_packets', 'reconst_bytes_from_entropy_src_port',
                       'reconst_tcp_from_packets', 'reconst_udp_from_flows'],

    # Groupes directionnels 5-états — surplus volumétrique (flood) vs déficit (outage).
    # Colonnes {metric}_dir_pos_proj_* / {metric}_dir_neg_proj_* produites par
    # compute_opinions_v3. Seules les métriques direction='both' : bytes, packets, flows.
    # Invariant de coarsening : proj_atk_pos + proj_atk_neg = proj_atk ternaire (Jøsang §3.5.4).
    # Réf : OUTAGE (Matta 2012) vs FLOOD (Sharafaldin 2018).
    'volume_surplus': ['prophet_bytes_dir_pos', 'prophet_packets_dir_pos', 'prophet_flows_dir_pos'],
    'volume_deficit': ['prophet_bytes_dir_neg', 'prophet_packets_dir_neg', 'prophet_flows_dir_neg'],
}

ATTACK_PRIORS_LITERATURE_BASELINE = {
    # UDP Flood : volume bytes+paquets élevés, Str cassé, TCP absent.
    # Pkt:Safe est le discriminant clé vs ICMP_FLOOD (paquets UDP ~64-128B
    # vs ICMP paddé ~1400B). Ent_sport retiré : la vraie attaque Nov 12
    # montre entropy_src_port=Safe (botnet sans spoofing aléatoire).
    # Ref : Mirsky et al. (2018) Kitsune ; observation RedeRio Nov 2025
    'UDP_FLOOD': {'Vol': 'Anom', 'Pkts': 'Anom', 'Str': 'Anom',
                  'TCP': 'Safe', 'Pkt': 'Safe'},

    # SYN Flood : explosion TCP, IP spoofées, paquets SYN uniformes, Str anormal.
    # Ref : MITRE ATT&CK T1498.001
    'SYN_FLOOD': {'TCP': 'Anom', 'Ent_ip': 'Anom',
                  'Pkt': 'Anom', 'Str': 'Anom'},

    # Port Scan : flows vers ports variés, bytes faibles, dst_port diversifié.
    # Ref : MITRE ATT&CK T1046
    'PORT_SCAN': {'TCP': 'Anom', 'Ent_dport': 'Anom', 'Vol': 'Safe'},

    # Exfiltration lente : bytes élevés, gros paquets, faible débit paquet,
    # canal inhabituel (Str2 anormal), ports sources fixes (Ent_sport:Safe).
    # Ref : MITRE ATT&CK T1048
    'DATA_EXFIL': {'Vol': 'Anom', 'TCP': 'Safe', 'Pkts': 'Safe',
                   'Pkt': 'Anom', 'Str2': 'Anom', 'Ent_sport': 'Safe'},

    # HTTP Flood L7 : explosion flows+bytes+paquets simultanée.
    # Ref : Sharafaldin et al. (2018) CIC-IDS2017
    'HTTP_FLOOD': {'TCP': 'Anom', 'Vol': 'Anom', 'Pkts': 'Anom'},

    # DNS Amplification : gros bytes (réponses jusqu'à 4KB), gros paquets,
    # Ent_ip élevé (réflecteurs variés), port source 53 fixe (Ent_sport:Safe).
    # Ref : Cloudflare DDoS Threat Report Q3 2024
    'DNS_AMP': {'Vol': 'Anom', 'Pkts': 'Anom', 'Str': 'Anom',
                'Ent_ip': 'Anom', 'Pkt': 'Anom', 'Ent_sport': 'Safe'},

    # Slowloris : connexions TCP persistantes, quasi-zéro payload et paquets.
    # Str:Safe + Pkts:Safe discriminent vs SYN_FLOOD.
    # Ref : Mirsky et al. (2018) Kitsune — slow-rate DoS
    'SLOWLORIS': {'TCP': 'Anom', 'Vol': 'Safe',
                  'Str': 'Safe', 'Pkts': 'Safe'},

    # ICMP Flood : gros volume, gros paquets (padding ~1400B), Str cassé,
    # TCP absent, pas de port source (ICMP n'en a pas → Ent_sport:Safe).
    # Pkt:Anom est le discriminant primaire vs UDP_FLOOD (Pkt:Safe).
    # Ref : Moustafa & Slay (2015) UNSW-NB15
    'ICMP_FLOOD': {'Vol': 'Anom', 'Pkts': 'Anom', 'Str': 'Anom',
                   'TCP': 'Safe', 'Pkt': 'Anom', 'Ent_sport': 'Safe'},

    'NETWORK_OUTAGE': {
        'volume':           'Anom',  # bytes/packets chutent → résidu négatif
        'connections':      'Anom',  # flows chutent aussi
        'protocol_icmp':    'Safe',  # pas d'ICMP flood
        'protocol_udp':     'Safe',  # pas d'UDP flood
        'protocol_tcp':     'Safe',  # pas de TCP surge
        'tcp_flags':        'Safe',  # pas de SYN/FIN surge
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# SBN_COND_OPINIONS : Opinions conditionnelles P(G_g = s | type_k)
# ──────────────────────────────────────────────────────────────────────────────
# Format : {type_k: {groupe_g: {Safe: p, Susp: p, Anom: p}}}
# Contrainte : sum(Safe + Susp + Anom) = 1.0 par entrée
#
# Interprétation des niveaux de certitude :
#   STRONG predictor (ud≈0.10) : {'S':0.03, 'M':0.07, 'A':0.90}
#   MOD    predictor (ud≈0.20) : {'S':0.08, 'M':0.22, 'A':0.70}
#   WEAK   predictor (ud≈0.35) : {'S':0.15, 'M':0.35, 'A':0.50}
#   NON-DISCRIMINANT           : {'S':0.33, 'M':0.34, 'A':0.33}
#   STRONG absent              : {'S':0.85, 'M':0.12, 'A':0.03}
#
# Sources : Sharafaldin 2018, Mirsky 2018, Moustafa 2015, Rossow 2014,
#           MITRE ATT&CK T1046/T1048/T1498/T1499, Cloudflare DDoS Report 2024
# ──────────────────────────────────────────────────────────────────────────────

SBN_COND_OPINIONS = {

    # ── UDP FLOOD ─────────────────────────────────────────────────────────────
    # Volume+UDP explosent, TCP/ICMP absents, paquets petits (~64-128B)
    # Ref : Sharafaldin 2018 CIC-IDS2017 Table III ; Mirsky 2018 Kitsune
    'UDP_FLOOD': {
        'volume':        {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # FORT
        'protocol_udp':  {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # DISCRIMINATEUR PRIMAIRE
        'protocol_tcp':  {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # TCP ABSENT
        'protocol_icmp': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # ICMP ABSENT
        'tcp_flags':     {'Safe': 0.82, 'Susp': 0.15, 'Anom': 0.03},  # syn≈0, fin≈0 (pas TCP)
        'fin_ratio':     {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # ratio fin/syn : pas de TCP → SAFE
        'connections':   {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # flows modérément anormaux
        'entropy':       {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # src_port ↑ (spoofing)
        'packet_size':   {'Safe': 0.70, 'Susp': 0.20, 'Anom': 0.10},  # petits paquets ~64B
        'reconstruction':{'Safe': 0.08, 'Susp': 0.22, 'Anom': 0.70},  # bytes~packets cassé
        'volume_surplus': {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # bytes/pkt/flows ↑↑↑ — FORT
        'volume_deficit': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # pas de chute
    },

    # ── SYN FLOOD ─────────────────────────────────────────────────────────────
    # syn ↑↑↑, fin≈0 → R_fin_from_syn très bas (demi-connexions jamais fermées)
    # Ref : MITRE ATT&CK T1498.001 ; Roesch 1999 Snort
    'SYN_FLOOD': {
        'volume':        {'Safe': 0.30, 'Susp': 0.45, 'Anom': 0.25},  # volume modéré (SYN sans payload)
        'protocol_tcp':  {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # TCP ↑↑↑ DISCRIMINATEUR
        'protocol_udp':  {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # UDP absent
        'protocol_icmp': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # ICMP absent
        'tcp_flags':     {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # syn ↑↑↑ (fort signal brut)
        'fin_ratio':     {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # ratio fin/syn → 0 (demi-conn.) ATK
        'connections':   {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # flows élevés (demi-conn.)
        'entropy':       {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # src_ip aléatoire (spoofing)
        'packet_size':   {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # paquets SYN petits ~60B UNIFORME
        'reconstruction':{'Safe': 0.30, 'Susp': 0.45, 'Anom': 0.25},
        'volume_surplus': {'Safe': 0.30, 'Susp': 0.45, 'Anom': 0.25},  # volume modéré (SYN sans payload)
        'volume_deficit': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # pas de chute
    },

    # ── ICMP FLOOD ────────────────────────────────────────────────────────────
    # icmp ↑↑↑, gros paquets (padding ~1400B), syn=0, fin=0
    # DISCRIMINATEURS PRIMAIRES vs UDP_FLOOD : protocol_icmp + packet_size
    # Ref : Moustafa & Slay 2015 UNSW-NB15 feature top-5
    'ICMP_FLOOD': {
        'volume':        {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # volume ↑↑↑
        'protocol_udp':  {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # UDP ABSENT — vs UDP_FLOOD
        'protocol_tcp':  {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # TCP absent
        'protocol_icmp': {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # ICMP ↑↑↑ DISCRIMINATEUR #1
        'tcp_flags':     {'Safe': 0.82, 'Susp': 0.15, 'Anom': 0.03},  # syn=0, fin=0 (pas TCP)
        'fin_ratio':     {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # pas de TCP → ratio SAFE
        'connections':   {'Safe': 0.15, 'Susp': 0.35, 'Anom': 0.50},  # flows≈0 (ICMP sans état)
        'entropy':       {'Safe': 0.82, 'Susp': 0.15, 'Anom': 0.03},  # entropie faible (paquets identiques)
        'packet_size':   {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # MTU ~1400B DISCRIMINATEUR #2
        'reconstruction':{'Safe': 0.08, 'Susp': 0.22, 'Anom': 0.70},
        'volume_surplus': {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # bytes ↑↑↑
        'volume_deficit': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # pas de chute
    },

    # ── DNS AMPLIFICATION ─────────────────────────────────────────────────────
    # Réponses UDP gonflées (×28-556), gros paquets, peu de sources (réflecteurs)
    # Ref : Rossow 2014 NDSS Table 2 (facteurs ampli.) ; Cloudflare DDoS Q3 2024
    'DNS_AMP': {
        'volume':        {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # bytes ↑↑↑ (réponses 4KB)
        'protocol_udp':  {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # udp ↑ (réponses port 53)
        'protocol_tcp':  {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # DNS principalement UDP
        'protocol_icmp': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # ICMP absent
        'tcp_flags':     {'Safe': 0.82, 'Susp': 0.15, 'Anom': 0.03},  # pas de SYN/FIN (UDP)
        'fin_ratio':     {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # UDP → pas de TCP → ratio SAFE
        'connections':   {'Safe': 0.70, 'Susp': 0.20, 'Anom': 0.10},  # flows normaux (peu de requêtes)
        'entropy':       {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # src_ip ↑ (réflecteurs) / src_port fixe
        'packet_size':   {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # très grands paquets DISCRIMINATEUR
        'reconstruction':{'Safe': 0.08, 'Susp': 0.22, 'Anom': 0.70},
        'volume_surplus': {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # bytes réponses ↑↑↑
        'volume_deficit': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # pas de chute
    },

    # ── HTTP FLOOD (L7) ───────────────────────────────────────────────────────
    # Connexions TCP COMPLÈTES (fin≈syn → R_fin_from_syn ≈ 1), volume+flows ↑↑↑
    # Ref : Sharafaldin 2018 CIC-IDS2017 ; Loránd Lékó 2020 USENIX
    'HTTP_FLOOD': {
        'volume':        {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},
        'protocol_tcp':  {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # TCP ↑↑ (conn. complètes)
        'protocol_udp':  {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},
        'protocol_icmp': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},
        'tcp_flags':     {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # syn↑ ET fin↑ (conn. complètes)
        'fin_ratio':     {'Safe': 0.70, 'Susp': 0.20, 'Anom': 0.10},  # fin≈syn → ratio ≈ 1 → SAFE
        'connections':   {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # flows ↑↑↑ FORT
        'entropy':       {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # src_ip variable
        'packet_size':   {'Safe': 0.30, 'Susp': 0.45, 'Anom': 0.25},  # paquets HTTP variables
        'reconstruction':{'Safe': 0.08, 'Susp': 0.22, 'Anom': 0.70},
        'volume_surplus': {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # volume ↑↑↑
        'volume_deficit': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # pas de chute
    },

    # ── SLOWLORIS ─────────────────────────────────────────────────────────────
    # Connexions TCP ouvertes jamais fermées → fin≈0, R_fin_from_syn → 0 (ATK FORT)
    # Volume et paquets FAIBLES (headers partiels seulement)
    # Ref : Mirsky 2018 Kitsune ; Hansen 2009 (Slowloris original paper)
    'SLOWLORIS': {
        'volume':        {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # bytes≈0 FORT
        'protocol_tcp':  {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # tcp ↑ (conn. maintenues)
        'protocol_udp':  {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},
        'protocol_icmp': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},
        'tcp_flags':     {'Safe': 0.33, 'Susp': 0.34, 'Anom': 0.33},  # syn↑ MAIS fin≈0 → signal MIXTE
        # geomean([syn_ATK, fin_SAFE]) → proj_atk modéré → condition neutre
        # Le discriminant est dans fin_ratio (groupe dédié ci-dessous).
        'fin_ratio':     {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # R_fin_from_syn → 0 = ATK FORT
        # DISCRIMINATEUR CLÉ vs DATA_EXFIL (fin_ratio=SAFE pour DATA_EXFIL)
        'connections':   {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # flows ↑↑ (conn. persistantes)
        'entropy':       {'Safe': 0.82, 'Susp': 0.15, 'Anom': 0.03},  # source unique → entropie faible
        'packet_size':   {'Safe': 0.33, 'Susp': 0.34, 'Anom': 0.33},  # non-discriminant (variable)
        'reconstruction':{'Safe': 0.30, 'Susp': 0.50, 'Anom': 0.20},
        # FIX : SLOWLORIS (Hansen 2009) — feature primaire = épuisement connexions TCP
        # (capturé par fin_ratio + connections). La reconstruction bytes~entropy est
        # un effet secondaire, non listé comme feature discriminante dans Mirsky 2018
        # Kitsune ni Sharafaldin 2018 CIC-IDS2017. La valeur strong_anom précédente
        # capturait toute attaque avec reconstruction anomale (DATA_EXFIL, HTTP_FLOOD).
        # Justification : 'weak_anom' {0.30/0.50/0.20} est défendable car le résidu
        # bytes~entropy pour Slowloris est secondaire et non constant selon l'implémentation.        'volume_surplus': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # bytes surplus≈0 (FORT)
        'volume_deficit': {'Safe': 0.82, 'Susp': 0.15, 'Anom': 0.03},  # pas de chute structurelle
    },

    # ── PORT SCAN ─────────────────────────────────────────────────────────────
    # flows ↑↑↑, entropy_dst_port ↑↑↑, bytes faibles (SYN seuls)
    # Ref : MITRE ATT&CK T1046 ; Nmap signatures (Gordon 2006)
    'PORT_SCAN': {
        'volume':        {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # bytes faibles
        'protocol_tcp':  {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # tcp ↑ (SYN+RST)
        'protocol_udp':  {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},
        'protocol_icmp': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},
        'tcp_flags':     {'Safe': 0.30, 'Susp': 0.45, 'Anom': 0.25},  # syn ↑, fin modéré (RST)
        'fin_ratio':     {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # RST ≠ FIN → ratio perturbé
        'connections':   {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # flows ↑↑↑ FORT DISCRIMINATEUR
        'entropy':       {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # entropy_dst_port ↑↑↑ FORT
        'packet_size':   {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # paquets très petits ~60B
        'reconstruction':{'Safe': 0.30, 'Susp': 0.45, 'Anom': 0.25},
        'volume_surplus': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # bytes faibles
        'volume_deficit': {'Safe': 0.82, 'Susp': 0.15, 'Anom': 0.03},  # pas de chute
    },

    # ── DATA EXFILTRATION LENTE ───────────────────────────────────────────────
    # Connexions TCP complètes fermées (fin≈syn → R_fin_from_syn≈1=NOR), gros paquets
    # Ref : MITRE ATT&CK T1048 ; Srivastava 2016 IEEE Big Data
    'DATA_EXFIL': {
        'volume':        {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # bytes ↑ mais modéré
        'protocol_tcp':  {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # tcp ↑ (exfil sur TCP)
        'protocol_udp':  {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},
        'protocol_icmp': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},
        'tcp_flags':     {'Safe': 0.82, 'Susp': 0.15, 'Anom': 0.03},  # syn=NOR, fin=NOR (conn. normales)
        'fin_ratio':     {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # fin≈syn → ratio≈1 = NOR FORT
        # DISCRIMINATEUR CLÉ vs SLOWLORIS (fin_ratio=ATK pour SLOWLORIS)
        'connections':   {'Safe': 0.08, 'Susp': 0.22, 'Anom': 0.70},  # flows ↑ (multiples conn.)
        'entropy':       {'Safe': 0.70, 'Susp': 0.20, 'Anom': 0.10},  # entropie faible (exfil ciblée)
        'packet_size':   {'Safe': 0.08, 'Susp': 0.22, 'Anom': 0.70},  # gros paquets (transfert)
        'reconstruction':{'Safe': 0.30, 'Susp': 0.45, 'Anom': 0.25},
        'volume_surplus': {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # bytes ↑ modéré
        'volume_deficit': {'Safe': 0.82, 'Susp': 0.15, 'Anom': 0.03},  # pas de chute
    },

    # ── NETWORK OUTAGE / PANNE ────────────────────────────────────────────────
    # Chute GLOBALE du trafic → résidus absolus élevés sur bytes, packets, flows.
    # NOTE DIRECTION : la ternaire {Safe,Susp,Anom} ne distingue pas surplus vs déficit.
    # volume=strong_anom est partagé avec les floods — discrimination via protocoles :
    # pour un OUTAGE, TOUS les protocoles (tcp,udp,icmp) chutent simultanément → safe,
    # alors que pour un FLOOD, au moins un protocole explose → anom.
    # Ref : observation RedeRio Déc 2025 ; Matta et al. (2012) ISP network events
    'NETWORK_OUTAGE': {
        'volume':        {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # résidu absolu élevé (chute)
        'protocol_tcp':  {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # pas d'explosion TCP
        'protocol_udp':  {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # pas d'explosion UDP
        'protocol_icmp': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # pas de flood ICMP
        'tcp_flags':     {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # pas de SYN/FIN surge
        'fin_ratio':     {'Safe': 0.33, 'Susp': 0.34, 'Anom': 0.33},  # ratio indéterminé (trafic minimal)
        'connections':   {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # flows CHUTENT
        'entropy':       {'Safe': 0.33, 'Susp': 0.34, 'Anom': 0.33},  # indéterminé (peu de paquets)
        'packet_size':   {'Safe': 0.33, 'Susp': 0.34, 'Anom': 0.33},  # indéterminé
        'reconstruction':{'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # relations structurelles cassées
        # ← DISCRIMINATEUR CLÉ : seul type où volume_deficit=FORT et volume_surplus=SAFE
        'volume_surplus': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # pas de flood positif
        'volume_deficit': {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # bytes/pkt/flows CHUTENT ← FORT
    },

    # ── BOTNET C&C (beaconing) ────────────────────────────────────────────────
    # Pattern temporel régulier, faible volume, connexions courtes répétées
    # Ref : MITRE ATT&CK T1071 ; Binsalleeh 2010 IEEE ComSec
    'BOTNET_CC': {
        'volume':        {'Safe': 0.70, 'Susp': 0.20, 'Anom': 0.10},  # trafic faible (heartbeat)
        'protocol_tcp':  {'Safe': 0.30, 'Susp': 0.45, 'Anom': 0.25},  # TCP (HTTP/HTTPS)
        'protocol_udp':  {'Safe': 0.60, 'Susp': 0.30, 'Anom': 0.10},
        'protocol_icmp': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},
        'tcp_flags':     {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # connexions courtes répétées
        'fin_ratio':     {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # beaconing → ratio perturbé (susp)
        'connections': {'Safe': 0.08, 'Susp': 0.22, 'Anom': 0.70},
        # FIX → mod_anom : beaconing = connexions TCP répétées = flows anomaux.
        # Ref : Garcia et al. (2014) DIMVA CTU-13 — nb_flows/min élevé = feature bot.
        'entropy':       {'Safe': 0.70, 'Susp': 0.20, 'Anom': 0.10},  # port C&C fixe → entropie faible
        'packet_size': {'Safe': 0.08, 'Susp': 0.22, 'Anom': 0.70},
        # FIX → mod_anom : petits paquets heartbeat (~100-500B) sont anomaux sur backbone.
        # Ref : Hofstede (2014) ACM SIGCOMM CCR — BPF "flat and low" = signature C&C.
        'reconstruction':{'Safe': 0.50, 'Susp': 0.30, 'Anom': 0.20},
        'volume_surplus': {'Safe': 0.70, 'Susp': 0.20, 'Anom': 0.10},  # faible trafic
        'volume_deficit': {'Safe': 0.82, 'Susp': 0.15, 'Anom': 0.03},  # pas de chute
    },

    # ──────────────────────────────────────────────────────────────────────────────
    # NTP AMPLIFICATION ─ BAF×556, identique DNS_AMP mais bytes et avg_pkt_size encore plus forts
    # Ref : van Rijswijk-Deij et al. (2014) APNIC/IMC — BAF NTP=556.9 vs DNS=28-54
    # Ref : Cisco CVE-2013-5211 — monlist: réponse 5500× la requête (8 bytes → ~43KB)
    # NOTE LIMITE : profil SBN très proche de DNS_AMP (même mécanisme UDP).
    # Discrimination par packet_size et reconstruction (plus extrêmes pour NTP).
    # ──────────────────────────────────────────────────────────────────────────────
    'NTP_AMP': {
        'volume': {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # bytes ↑↑↑↑ (BAF×556) FORT
        'protocol_udp': {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # UDP ↑↑↑ (NTP/123 pur) FORT
        'protocol_tcp': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # TCP absent (UDP pur)
        'protocol_icmp': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # ICMP absent
        'tcp_flags': {'Safe': 0.82, 'Susp': 0.15, 'Anom': 0.03},  # syn=0, fin=0 (pas de TCP)
        'fin_ratio': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # UDP → ratio indéfini → SAFE
        'connections': {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # flows modérés (réflecteurs UDP)
        'entropy': {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # src_ip ↑ (réflecteurs) / src_port=123 fixe
        # DISCRIMINATEUR vs DNS_AMP : packet_size encore plus extrême (monlist = liste 600 IPs)
        # Cisco CVE-2013-5211 : réponse jusqu'à 5500× la requête initiale de 8 bytes
        'packet_size': {'Safe': 0.01, 'Susp': 0.04, 'Anom': 0.95},  # paquets monlist très lourds
        'reconstruction': {'Safe': 0.08, 'Susp': 0.22, 'Anom': 0.70},
        # FIX : même profil que DNS_AMP. Le discriminateur NTP_AMP vs DNS_AMP est
        # le volume et packet_size (paquets monlist plus lourds), pas reconstruction.
        # Ref : Cisco CVE-2013-5211 — différence de BAF traduite dans le groupe volume.        'volume_surplus': {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # surplus massif (amplification)
        'volume_deficit': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},
    },

    # ──────────────────────────────────────────────────────────────────────────────
    # BRUTE FORCE SSH ─ dictionnaire sur SSH port 22
    # Ref : Najafabadi et al. (2015) ICMLA — flows=feature primaire (NetFlow 5min)
    # Ref : Hynek et al. (2020) IFIP SEC 35 — ML IP flows, campus network
    # Ref : Hofstede et al. (2014) ACM SIGCOMM CCR — SSHCure: PPF/BPF "flat traffic"
    # DISCRIMINATEURS vs PORT_SCAN :
    #   volume=weak_susp (bytes non-nuls, ≠ scan bytes≈0)
    #   tcp_flags=mod_anom (syn↑↑ constant, ≠ scan syn modéré)
    #   entropy=strong_anom (entropy_dst_port↑ : port 22 fixe = concentration anormale)
    # ──────────────────────────────────────────────────────────────────────────────
    'BRUTE_FORCE_SSH': {
        'volume': {'Safe': 0.30, 'Susp': 0.45, 'Anom': 0.25},  # bytes modéré (≠ DDoS volumétrique)
        'protocol_tcp': {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # 100% TCP — FORT (SSH=TCP)
        'protocol_udp': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # UDP absent (SSH=TCP)
        'protocol_icmp': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # ICMP absent
        'tcp_flags': {'Safe': 0.08, 'Susp': 0.22, 'Anom': 0.70},  # syn↑↑ (tentatives) + fin modéré
        # fin/syn < 1 : tentatives échouées → RST (pas FIN)
        # Ref : Hofstede 2014 SSHCure — ratio complétion TCP < 1 pendant phase BF
        'fin_ratio': {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # RST > FIN → ratio perturbé → SUSP
        # flows ↑↑↑ : SIGNAL PRIMAIRE (Najafabadi 2015 — feature discriminante #1)
        'connections': {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # flows ↑↑↑ FORT
        # entropy_dst_port ↑ (port 22 fixe=concentration) + entropy_src_port ↑ (ports bots variés)
        # entropy_src_ip modéré (botnet distribué, plusieurs IPs)
        'entropy': {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},  # entropie anormale FORT
        # avg_pkt_size ↓ : SSH handshake ~200B (Hofstede 2014 : BPF "flat and low")
        'packet_size': {'Safe': 0.08, 'Susp': 0.22, 'Anom': 0.70},  # petits paquets SSH — FORT
        'reconstruction': {'Safe': 0.30, 'Susp': 0.45, 'Anom': 0.25},  # légèrement perturbé (petits paquets)
        'volume_surplus': {'Safe': 0.30, 'Susp': 0.45, 'Anom': 0.25},  # pas de surplus volumétrique (≠ DDoS)
        'volume_deficit': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},
    },

    # ──────────────────────────────────────────────────────────────────────────────
    # DNS TUNNELING ─ exfiltration covert via DNS (MITRE ATT&CK T1048.001)
    # Ref : Sharma et al. (2018) Procedia CS 132 — entropy_src_port ↑↑ feature clé
    # Ref : Habibi et al. (2019) IEEE IM — single host compromised, backbone 10Gbps
    # Ref : MDPI Electronics (2023) 12(6):1467 — DNS normal 50-550B ; tunnel > 550B
    # DISCRIMINATEURS CLÉ vs DNS_AMP :
    #   entropy : src_ip=Safe (source unique ≠ nombreux réflecteurs DNS_AMP) — FORT
    # DISCRIMINATEURS CLÉ vs DATA_EXFIL :
    #   protocol_udp=mod_susp (DNS/UDP présent ≠ DATA_EXFIL=strong_safe)
    # LIMITE : profil "soft" → u_sbn élevé probable (attaque difficile à qualifier)
    # ──────────────────────────────────────────────────────────────────────────────
    'DNS_TUNNELING': {
        'volume': {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # bytes ↑ modéré (exfiltration)
        'protocol_udp': {'Safe': 0.08, 'Susp': 0.22, 'Anom': 0.70},
        # FIX → mod_anom : DNS tunneling utilise UDP/53 de façon soutenue.
        # Ref : Sharma et al. (2018) Procedia CS 132 — UDP flux anomal en DNS tunneling.
        # Génère de l'evidence positive quand prophet_udp_proj_atk est élevé.
        'protocol_tcp': {'Safe': 0.08, 'Susp': 0.22, 'Anom': 0.70},
        # FIX → mod_anom : DNS tunneling utilise aussi TCP/53 pour les grandes requêtes.
        # Ref : MDPI Electronics (2023) — DNS queries > 512B utilisent TCP.
        'protocol_icmp': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},  # ICMP absent (DNS=UDP/TCP)
        'tcp_flags': {'Safe': 0.30, 'Susp': 0.45, 'Anom': 0.25},  # syn modéré + fin modéré (TCP/53)
        # fin/syn ≈ 1 : TCP/53 se ferme proprement (≠ Slowloris fin≈0, ≠ SYN_Flood)
        'fin_ratio': {'Safe': 0.33, 'Susp': 0.34, 'Anom': 0.33},  # neutre (TCP/53 normal)
        # flows ↑ modéré : sessions DNS persistantes (hôte compromis → serveur attaquant)
        'connections': {'Safe': 0.08, 'Susp': 0.22, 'Anom': 0.70},
        # FIX → mod_anom : sessions DNS persistantes = flows anomaux.
        # Ref : Habibi et al. (2019) IEEE IM — flux DNS persistant vers serveur C2.
        # SIGNAL CLÉ vs DNS_AMP : entropy_src_ip = SAFE (source unique vs multi-réflecteurs)
        # Habibi 2019 : single compromised host = entropy_src_ip very low
        # entropy_src_port ↑↑ : client tunneling ports variés (Sharma 2018)
        # Geomean([src_ip=SAFE, src_port=ATK, dst_port=SAFE]) → safe dominant → mod_susp
        'entropy': {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # net modérément suspect
        # avg_pkt_size ↑ : noms encodés > DNS normal 50-550B (MDPI 2023 ; Habibi 2019)
        'packet_size': {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # paquets DNS > normale → SUSP
        # reconst_bytes_from_entropy_src_port perturbé (bytes↑ + entropy_src_port↑↑)
        'reconstruction': {'Safe': 0.20, 'Susp': 0.55, 'Anom': 0.25},  # légèrement anormal
        'volume_surplus': {'Safe': 0.30, 'Susp': 0.45, 'Anom': 0.25},  # surplus faible (exfiltration)
        'volume_deficit': {'Safe': 0.85, 'Susp': 0.12, 'Anom': 0.03},
    },
}

# ==============================================================================
# METR-LA — Los Angeles Traffic Speed Dataset
# ==============================================================================
# Une seule métrique (speed) → une seule source dans le groupe "traffic".
# Conséquence attendue : u_qualif élevé (≈ 0.60–0.80) car tous les types montrent
# traffic:Anom. Scientifiquement honnête : avec 1 métrique scalaire on ne peut pas
# discriminer le TYPE d'anomalie, seulement la CONFIRMER.
# La qualification vaut surtout pour le TTD et la confirmation de détection.
# Ref : Coifman (2002) – congestion vs incident speed profiles sur freeway loops
#       Li et al. (2018) DCRNN – profil de vitesse lors d'anomalies de trafic LA
#       Zhang et al. (2019) T-ITS – frozen sensor vs genuine speed drop
# ==============================================================================

SBN_COND_OPINIONS_METR_LA = {
    # ── CONGESTION ──────────────────────────────────────────────────────────
    # Ralentissement progressif par accumulation de véhicules.
    # Speed ↓ graduel sur plusieurs fenêtres (10-30 min).
    # Discriminant temporel (et non sémantico-métrique) vs INCIDENT.
    # Ref : Coifman 2002 §3 — congestion wave propagation speed ~18 km/h
    'CONGESTION': {
        'traffic': {'Safe': 0.05, 'Susp': 0.20, 'Anom': 0.75},
        # speed ↓ modéré-fort sur plusieurs fenêtres
    },

    # ── INCIDENT ────────────────────────────────────────────────────────────
    # Accident ou fermeture de voie → chute brutale en 1–2 fenêtres.
    # Speed ↓↓ soudain, profil "marche descendante".
    # Ref : Chen et al. (2001) §4.2 – incident signature: speed < 40 mph in <2 min
    'INCIDENT': {
        'traffic': {'Safe': 0.03, 'Susp': 0.12, 'Anom': 0.85},
        # speed ↓↓ fort et soudain
    },

    # ── SENSOR_MALFUNCTION ─────────────────────────────────────────────────
    # Capteur bloqué (frozen), hors gamme (spike), ou déconnecté.
    # Déviation arbitraire (↑ ou ↓) sans cohérence spatiale.
    # Plus incertain : peut ressembler à un trafic normal ou à une vitesse nulle.
    # Ref : Zhang et al. (2019) IEEE T-ITS – frozen value ≡ constant ± bruit résiduel
    'SENSOR_MALFUNCTION': {
        'traffic': {'Safe': 0.15, 'Susp': 0.35, 'Anom': 0.50},
        # déviation plus incertaine — pas de direction déterministe
    },
}

# Correction QUALIFY_GROUP_SOURCES METR-LA (cohérence avec colonnes réelles) :
# prophet_speed → colonnes CSV : prophet_speed_proj_safe / proj_susp / proj_atk
QUALIFY_GROUP_SOURCES_METR_LA = {
    'traffic': ['prophet_speed'],
}

ATTACK_PRIORS_METR_LA = {
    'CONGESTION': {'traffic': 'Anom'},
    'INCIDENT': {'traffic': 'Anom'},
    'SENSOR_MALFUNCTION': {'traffic': 'Anom'},
}

# ==============================================================================
# GECCO-IoT — Water Quality Sensor Dataset
# ==============================================================================
# 6 métriques : pH, Cl, Tp, Redox, Leit, Trueb
# 4 groupes : chemistry (pH+Redox), organic (Cl+Trueb), physical (Tp+Leit),
#             reconstruction (reconst_pH_from_Redox, reconst_Cl_from_Redox, reconst_Leit_from_Tp)
#
# DISCRIMINATEUR CLÉ : le groupe 'reconstruction' est la signature principale de
# SENSOR_FAULT : un capteur défaillant brise la relation physico-chimique prédite
# par RANSAC/QuantileReg, ce qui donne un résidu reconst élevé même si les autres
# capteurs sont normaux. C'est exactement l'exploitation de la Nernst equation
# (pH ~ f(Redox)) et de la corrélation conductivité-température.
# Ref : Storey et al. (2011) Water Research – PCA sur pH/Cl/Redox pour détection
#       Talagala et al. (2019) JCGS – isolated vs multivariate sensor fault patterns
#
# NOTE sur RECONST_RULES pour GECCO :
# Les paires sont définies dans config.py GECCO bloc :
#   {"target":"pH",   "feature":"Redox"} → Nernst + Pourbaix diagram (pH-Eh diagram)
#   {"target":"Cl",   "feature":"Redox"} → chlorine residual ~ oxidation potential
#   {"target":"Leit", "feature":"Tp"}    → Kohlrausch law : Λ(T) ≈ Λ₀(1 + α·ΔT)
# ==============================================================================

SBN_COND_OPINIONS_GECCO = {
    # ── CHEMICAL_CONTAMINATION ───────────────────────────────────────────────
    # Introduction d'un contaminant chimique (acide, base, solvant, pesticide).
    # Signature : pH ↓ ou ↑ (dépend du contaminant), Cl consommé → ↓,
    #             Redox shift, Turbidity ↑ (particules), Tp et Leit peu perturbés.
    # Les corrélations pH-Redox et Cl-Redox se brisent → reconst Anom.
    # Ref : Beck & Jawale (2018) GECCO §3 – chemical spike events;
    #       WHO (2017) §7.2 – pH hors [6.5–8.5] = contamination indicator
    'CHEMICAL_CONTAMINATION': {
        'chemistry': {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},
        # pH ET Redox dévient simultanément → FORT signal
        'organic': {'Safe': 0.05, 'Susp': 0.15, 'Anom': 0.80},
        # Cl consommé (↓), Trueb ↑ → fort signal
        'physical': {'Safe': 0.55, 'Susp': 0.35, 'Anom': 0.10},
        # Tp et Leit peu perturbés dans la phase initiale d'une contamination chimique
        'reconstruction': {'Safe': 0.03, 'Susp': 0.12, 'Anom': 0.85},
        # Corrélations pH-Redox et Cl-Redox brisées
    },

    # ── BIOLOGICAL_CONTAMINATION ────────────────────────────────────────────
    # Prolifération microbienne (bactéries, algues, biofilm).
    # Processus lent : Cl ↓↓ (consommé par désinfection), Redox ↓ (O2 consommé),
    # Turbidity ↑ progressive, pH légèrement perturbé.
    # Leit et Tp relativement stables (processus biologique, pas chimique).
    # Ref : WHO (2017) §11.3 – chlorine demand increase = microbiological indicator;
    #       LeChevallier & Au (2004) WHO – biofilm O2 consumption → Redox ↓
    'BIOLOGICAL_CONTAMINATION': {
        'chemistry': {'Safe': 0.08, 'Susp': 0.27, 'Anom': 0.65},
        # Redox ↓ (O2 consommé), pH légèrement perturbé
        'organic': {'Safe': 0.03, 'Susp': 0.12, 'Anom': 0.85},
        # Cl ↓↓ DOMINANT (consommé) + Trueb ↑ → signal fort
        'physical': {'Safe': 0.65, 'Susp': 0.25, 'Anom': 0.10},
        # Tp et Leit stables (biologique ≠ thermique)
        'reconstruction': {'Safe': 0.10, 'Susp': 0.28, 'Anom': 0.62},
        # Corrélations partiellement brisées (Cl-Redox surtout)
    },

    # ── SENSOR_FAULT ────────────────────────────────────────────────────────
    # Panne matérielle d'un capteur isolé (dérive, blocage, coupure).
    # Signature clé : UN capteur dévie, les autres restent normaux →
    # le groupe chemistry ou organic est Susp (pas Anom plein car 1/2 capteursOK),
    # MAIS la reconstruction BRISE (le capteur cassé ≠ son prédicteur → résidu fort).
    # C'est le discriminateur primaire par rapport à CONTAMINATION (qui affecte TOUT).
    # Ref : Storey et al. (2011) Water Res. – isolated fault vs correlated multivariate;
    #       Talagala et al. (2019) JCGS §4 – reconstruction error as fault indicator;
    #       Kroll et al. (2016) J. Environ. Eng. – frozen Cl sensor pattern
    'SENSOR_FAULT': {
        'chemistry': {'Safe': 0.40, 'Susp': 0.45, 'Anom': 0.15},
        # 1 des 2 capteurs (pH OU Redox) peut être défaillant → signal partiel
        'organic': {'Safe': 0.40, 'Susp': 0.45, 'Anom': 0.15},
        # Cl OU Trueb isolément défaillant → signal partiel
        'physical': {'Safe': 0.30, 'Susp': 0.50, 'Anom': 0.20},
        # Tp ou Leit peut dériver thermiquement → légèrement plus Susp
        'reconstruction': {'Safe': 0.05, 'Susp': 0.15, 'Anom': 0.80},
        # DISCRIMINATEUR PRIMAIRE : capteur faulty ≠ son prédicteur → résidu fort
    },

    # ── PROCESS_ANOMALY ─────────────────────────────────────────────────────
    # Anomalie systémique affectant tous les paramètres (choc hydraulique,
    # dégazage, forte pluie, vidange/remplissage).
    # Tous les capteurs dévient simultanément, de façon cohérente.
    # Ref : Beck & Jawale (2018) §2 – hydraulic disturbances in distribution network;
    #       WHO (2017) §3.1 – pressure transient contamination
    'PROCESS_ANOMALY': {
        'chemistry': {'Safe': 0.05, 'Susp': 0.20, 'Anom': 0.75},
        'organic': {'Safe': 0.05, 'Susp': 0.20, 'Anom': 0.75},
        'physical': {'Safe': 0.05, 'Susp': 0.20, 'Anom': 0.75},
        # Tout dévie simultanément — aucun groupe épargné
        'reconstruction': {'Safe': 0.15, 'Susp': 0.30, 'Anom': 0.55},
        # Corrélations partiellement préservées (perturbation simultanée cohérente)
    },
}

# QUALIFY_GROUP_SOURCES GECCO : noms exacts des colonnes CSV produites par
# compute_opinions_v3.py (préfixe prophet_ et reconst_{target}_from_{feature})
QUALIFY_GROUP_SOURCES_GECCO = {
    'chemistry':      ['prophet_pH',   'prophet_Redox'],
    'organic':        ['prophet_Cl',   'prophet_Trueb'],
    'physical':       ['prophet_Tp',   'prophet_Leit'],
    'reconstruction': ['reconst_pH_from_Redox', 'reconst_Cl_from_Redox',
                       'reconst_Leit_from_Tp'],
}

ATTACK_PRIORS_GECCO = {
    'CHEMICAL_CONTAMINATION':  {
        'chemistry': 'Anom', 'organic': 'Anom', 'physical': 'Safe', 'reconstruction': 'Anom',
    },
    'BIOLOGICAL_CONTAMINATION': {
        'chemistry': 'Susp', 'organic': 'Anom', 'physical': 'Safe', 'reconstruction': 'Susp',
    },
    'SENSOR_FAULT': {
        'chemistry': 'Susp', 'organic': 'Susp', 'physical': 'Susp', 'reconstruction': 'Anom',
    },
    'PROCESS_ANOMALY': {
        'chemistry': 'Anom', 'organic': 'Anom', 'physical': 'Anom', 'reconstruction': 'Susp',
    },
}

# ==============================================================================
# CESNET-TimeSeries24 — ISP Flow Data (Czech Republic)
# ==============================================================================
# Métriques : n_bytes, n_packets, n_flows, tcp_udp_ratio_packets, sum_n_dest_ports
# Source : agrégation ISP nationale → métriques de trafic réseau sans entropie fine
# Labels : DoS/DDoS annotations (voir Cejka et al. 2023 Nature SD §2)
#
# CHOIX DE QUALIFY_GROUP_SOURCES :
# On utilise uniquement les colonnes *_proj_safe/susp/atk standard (pas _dir_pos/_dir_neg)
# car CESNET a ASYMMETRIC_THRESHOLD_METRICS = 'both' pour bytes/packets/flows,
# et les colonnes directionnelles ne sont générées que si ce paramètre est actif.
# Pour la robustesse on s'appuie sur les colonnes toujours présentes.
#
# GROUPES SÉMANTIQUES :
#   volume     : n_bytes + n_packets (quantité brute de données)
#   connections: n_flows (nombre de flux = couche connexions)
#   ratio      : tcp_udp_ratio_packets (protocole dominant)
#   diversity  : sum_n_dest_ports (diversité des ports destination)
#
# Note : on retire volume_surplus / volume_deficit du QUALIFY_GROUP_SOURCES
# et on remplace par 'volume' (bidirectionnel) car qualify_anomaly_sbn.py
# utilise get_group_p() qui regarde proj_safe/susp/atk, pas les colonnes dir.
# La direction (hausse vs chute) est capturée par les ASYMMETRIC_THRESHOLD_METRICS
# en amont (dans compute_evidence), pas dans qualify.
#
# TYPES D'ATTAQUES :
# CESNET annote des attaques DoS/DDoS sans sous-catégorisation fine → 4 types
# représentent les familles de DDoS les plus documentées dans les données ISP.
# Ref : Cejka et al. (2023) §3 — DoS annotations; Sharafaldin (2018) — signatures
# ==============================================================================

SBN_COND_OPINIONS_CESNET = {
    # ── VOLUMETRIC_FLOOD ────────────────────────────────────────────────────
    # UDP, ICMP, ou TCP SYN flood : volume massif, beaucoup de flux, ratio UDP shift.
    # n_bytes↑↑, n_packets↑↑, n_flows↑, tcp_udp_ratio ↓ (UDP dominant).
    # Ports destination variés (pas discriminant pour botnet distribué).
    # Ref : Sharafaldin (2018) Table III – UDP/ICMP flood feature profile;
    #       Mirsky (2018) Kitsune – volume explosion as primary DDoS indicator
    'VOLUMETRIC_FLOOD': {
        'volume': {'Safe': 0.03, 'Susp': 0.07, 'Anom': 0.90},
        # n_bytes + n_packets ↑↑↑ → FORT discriminateur
        'connections': {'Safe': 0.05, 'Susp': 0.15, 'Anom': 0.80},
        # n_flows ↑ (flood génère beaucoup de flux courts)
        'ratio': {'Safe': 0.10, 'Susp': 0.30, 'Anom': 0.60},
        # tcp_udp_ratio shift (UDP flood → ratio ↓)
        'diversity': {'Safe': 0.55, 'Susp': 0.35, 'Anom': 0.10},
        # dest_ports peu discriminant pour flood distribué
        'flow_behavior': {'Safe':0.20, 'Susp':0.45, 'Anom':0.35},
        # duration normal (UDP flood = beaucoup de petits flux courts normaux)
    },

    # ── APPLICATION_LAYER_ATTACK ────────────────────────────────────────────
    # HTTP flood, Slowloris, DNS query flood : beaucoup de connexions courtes,
    # volume par flux modéré, TCP dominant (HTTP), ports destination variés.
    # n_flows↑↑, n_bytes ~normal, tcp_udp_ratio ↑ (TCP dominant pour HTTP).
    # Ref : Mirsky (2018) Kitsune – Slowloris: many connections, small packets;
    #       Roesch (1999) Snort – HTTP flood: high flow count, normal bandwidth
    'APPLICATION_LAYER_ATTACK': {
        'volume': {'Safe': 0.35, 'Susp': 0.45, 'Anom': 0.20},
        # bytes ~ normaux (pas de bandwidth explosion)
        'connections': {'Safe': 0.03, 'Susp': 0.12, 'Anom': 0.85},
        # n_flows↑↑ DOMINANT (nombreuses connexions TCP courtes ou semi-ouvertes)
        'ratio': {'Safe': 0.15, 'Susp': 0.35, 'Anom': 0.50},
        # TCP dominant (HTTP/HTTPS = couche 7)
        'diversity': {'Safe': 0.20, 'Susp': 0.40, 'Anom': 0.40},
        # ports variés (attaquants scannent plusieurs services)
        'flow_behavior': {'Safe': 0.03, 'Susp': 0.12, 'Anom': 0.85},
        # FORT DISCRIMINATEUR : Slowloris = duration >> normal ; HTTP flood = duration variable
    },

    # ── REFLECTION_AMPLIFICATION ────────────────────────────────────────────
    # DNS/NTP/SSDP/Memcached amplification : peu de sources → beaucoup de bytes.
    # Amplification factors : DNS×54, NTP×556, Memcached×51000 (Rossow 2014 NDSS).
    # n_bytes↑↑ (amplifiés), n_packets↑ (réponses amplifiées), n_flows modéré,
    # tcp_udp_ratio ↓↓ (UDP dominant — réflecteurs UDP),
    # sum_n_dest_ports STABLE (port de destination fixe : DNS=53, NTP=123).
    # C'est le DISCRIMINATEUR FORT vs VOLUMETRIC_FLOOD : dest_ports fixe ≠ flood distribué.
    # Ref : Rossow (2014) NDSS Table I – amplification factors;
    #       Kührer et al. (2014) USENIX Sec – NTP monlist reflection profile
    'REFLECTION_AMPLIFICATION': {
        'volume': {'Safe': 0.03, 'Susp': 0.12, 'Anom': 0.85},
        # bytes↑↑ (paquet amplifié >> paquet trigger)
        'connections': {'Safe': 0.20, 'Susp': 0.45, 'Anom': 0.35},
        # n_flows modéré (réflecteurs = serveurs légitimes, flux TCP limités)
        'ratio': {'Safe': 0.05, 'Susp': 0.15, 'Anom': 0.80},
        # UDP dominant FORT (DNS/NTP/SSDP → tout UDP)
        'diversity': {'Safe': 0.80, 'Susp': 0.15, 'Anom': 0.05},
        # DISCRIMINATEUR PRIMAIRE : port destination fixe (53, 123, 1900...)
        'flow_behavior': {'Safe': 0.45, 'Susp': 0.40, 'Anom': 0.15},
        # TTL des réflecteurs similaire au trafic légitime (vraies machines)
    },

    # ── NETWORK_OUTAGE ──────────────────────────────────────────────────────
    # Panne ou attaque de disponibilité : tous les métriques CHUTENT.
    # Volume ↓, connexions ↓, ratio change peu (disparition homogène).
    # Ref : Barford et al. (2002) IMW – outage vs flash crowd signatures;
    #       Matta et al. (2012) – blackholing in ISP routing causes abrupt drops
    'NETWORK_OUTAGE': {
        'volume': {'Safe': 0.03, 'Susp': 0.12, 'Anom': 0.85},
        # n_bytes + n_packets ↓↓ → chute quasi-totale
        'connections': {'Safe': 0.05, 'Susp': 0.20, 'Anom': 0.75},
        # n_flows ↓ (plus de trafic entrant/sortant)
        'ratio': {'Safe': 0.55, 'Susp': 0.35, 'Anom': 0.10},
        # Ratio TCP/UDP peu modifié (disparition homogène des deux protocoles)
        'diversity': {'Safe': 0.60, 'Susp': 0.30, 'Anom': 0.10},
        # Diversité ports ↓ aussi mais peu discriminant
        'flow_behavior': {'Safe': 0.30, 'Susp': 0.50, 'Anom': 0.20},
        # Duration et TTL peu modifiés lors d'une panne (trafic résiduel normal)
    },
}

# QUALIFY_GROUP_SOURCES CESNET :
# Utilise les préfixes prophet_{metric_name} SANS _dir_pos/_dir_neg
# (robustesse : ces colonnes existent toujours vs dir_pos/neg conditionnels)
# Noms de métriques = ACTIVE_METRICS définis dans le bloc CESNET de config.py :
# ['n_bytes', 'n_packets', 'n_flows', 'tcp_udp_ratio_packets', 'sum_n_dest_ports']
QUALIFY_GROUP_SOURCES_CESNET = {
    'volume': ['prophet_n_bytes', 'prophet_n_packets'],
    'connections': ['prophet_n_flows'],
    'ratio': ['prophet_tcp_udp_ratio_packets'],
    'diversity': ['prophet_sum_n_dest_ports'],
    'flow_behavior': ['prophet_avg_duration', 'prophet_avg_ttl'],
}

ATTACK_PRIORS_CESNET = {
    'VOLUMETRIC_FLOOD': {
        'volume': 'Anom', 'connections': 'Anom', 'ratio': 'Susp', 'diversity': 'Safe',
    },
    'APPLICATION_LAYER_ATTACK': {
        'volume': 'Safe', 'connections': 'Anom', 'ratio': 'Susp', 'diversity': 'Susp',
    },
    'REFLECTION_AMPLIFICATION': {
        'volume': 'Anom', 'connections': 'Susp', 'ratio': 'Anom', 'diversity': 'Safe',
    },
    'NETWORK_OUTAGE': {
        'volume': 'Anom', 'connections': 'Anom', 'ratio': 'Safe', 'diversity': 'Safe',
    },
}

# ATTACK_PRIORS_LITERATURE_BASELINE
# Dérivés uniquement de :
#   - Sharafaldin et al. (2018) CIC-IDS2017, Table III
#   - Mirsky et al. (2018) Kitsune, feature taxonomy
#   - Moustafa & Slay (2015) UNSW-NB15
#   - MITRE ATT&CK T1046/T1048/T1498/T1499
# ATTACK_PRIORS v15 — signatures mises à jour avec les groupes v15
# (protocol_tcp/udp/icmp, tcp_flags avec R_fin_from_syn, reconstruction 5 paires)
# Ref :
#   Roesch (1999) Snort — ratio fin/syn (tcp_flags)
#   Mirsky et al. (2018) Kitsune — Slowloris : connexions jamais fermées
#   Moustafa & Slay (2015) UNSW-NB15 — ICMP discriminateur
#   Sharafaldin et al. (2018) CIC-IDS2017 — HTTP/UDP/SYN flood features
#   MITRE ATT&CK T1046/T1048/T1498/T1499
ATTACK_PRIORS = {

    # ── UDP FLOOD ─────────────────────────────────────────────────────────
    'UDP_FLOOD': {
        'volume': 'Anom',        # bytes+packets très élevés
        'protocol_tcp': 'Safe',  # TCP non affecté
        'protocol_udp': 'Anom',  # UDP ↑↑↑ — discriminateur primaire vs ICMP
        'protocol_icmp': 'Safe', # ICMP normal — discriminateur vs ICMP_FLOOD
        'tcp_flags': 'Safe',     # syn=NOR, fin=NOR, R_fin_from_syn=NOR
        'connections': 'Susp',   # flows légèrement anormaux
        'entropy': 'Susp',       # src_port ↑ (spoofing aléatoire)
        'packet_size': 'Safe',   # paquets UDP normaux (~64-128B)
        'reconstruction': 'Anom',
    },

    # ── SYN FLOOD ─────────────────────────────────────────────────────────
    # syn ↑↑↑, fin ≈ 0 → R_fin_from_syn très élevé (demi-connexions)
    'SYN_FLOOD': {
        'volume': 'Susp',        # volume modéré (SYN sans payload)
        'protocol_tcp': 'Anom',  # tcp ↑↑ (tous SYN)
        'protocol_udp': 'Safe',
        'protocol_icmp': 'Safe',
        'tcp_flags': 'Anom',     # syn=ATK, fin=NOR, R_fin_from_syn=ATK (demi-conn.)
        'connections': 'Susp',   # flows élevés (demi-connexions)
        'entropy': 'Susp',       # src_ip aléatoire (spoofing)
        'packet_size': 'Anom',   # paquets SYN petits (~60B)
        'reconstruction': 'Susp',
    },

    # ── ICMP FLOOD ────────────────────────────────────────────────────────
    # icmp ↑↑↑ (discriminateur primaire), syn=0, fin=0
    'ICMP_FLOOD': {
        'volume': 'Anom',         # bytes+packets élevés
        'protocol_tcp': 'Safe',   # TCP absent
        'protocol_udp': 'Safe',   # UDP normal — discriminateur vs UDP_FLOOD
        'protocol_icmp': 'Anom',  # ICMP ↑↑↑ — discriminateur primaire
        'tcp_flags': 'Safe',      # syn=0, fin=0, R_fin_from_syn=NOR (pas de TCP)
        'connections': 'Anom',    # flows ≈ 0 (ICMP sans état)
        'entropy': 'Safe',        # entropie normale (paquets identiques)
        'packet_size': 'Anom',    # paquets MTU (~1400B)
        'reconstruction': 'Anom',
    },

    # ── PORT SCAN ─────────────────────────────────────────────────────────
    'PORT_SCAN': {
        'volume': 'Safe',
        'protocol_tcp': 'Susp',
        'protocol_udp': 'Safe',
        'protocol_icmp': 'Safe',
        'tcp_flags': 'Susp',      # syn ↑, fin modéré (RST des ports fermés)
        'connections': 'Anom',    # flows ↑↑↑ (scan massif)
        'entropy': 'Anom',        # entropy_dst_port ↑↑↑
        'packet_size': 'Anom',    # paquets très petits (SYN seuls)
        'reconstruction': 'Susp',
    },

    # ── HTTP FLOOD ────────────────────────────────────────────────────────
    # Connexions TCP complètes → fin ≈ syn → R_fin_from_syn modéré
    'HTTP_FLOOD': {
        'volume': 'Anom',
        'protocol_tcp': 'Anom',
        'protocol_udp': 'Safe',
        'protocol_icmp': 'Safe',
        'tcp_flags': 'Susp',      # syn=ATK, fin=ATK, R_fin_from_syn=modéré (conn. complètes)
        'connections': 'Anom',
        'entropy': 'Susp',
        'packet_size': 'Susp',
        'reconstruction': 'Anom',
    },

    # ── DNS AMPLIFICATION ─────────────────────────────────────────────────
    'DNS_AMP': {
        'volume': 'Anom',
        'protocol_tcp': 'Safe',   # DNS principalement UDP
        'protocol_udp': 'Susp',   # udp ↑ (réponses DNS)
        'protocol_icmp': 'Safe',
        'tcp_flags': 'Safe',      # pas de SYN/FIN (UDP), R_fin_from_syn=NOR
        'connections': 'Safe',    # flows normaux
        'entropy': 'Susp',        # src_ip ↑ (réflecteurs variés)
        'packet_size': 'Anom',    # très grands paquets (3000+ bytes)
        'reconstruction': 'Anom',
    },

    # ── SLOWLORIS ─────────────────────────────────────────────────────────
    # Connexions TCP ouvertes jamais fermées → fin ≈ 0, R_fin_from_syn très élevé
    # Ref : Mirsky et al. (2018) Kitsune — Slowloris signature primaire
    'SLOWLORIS': {
        'volume': 'Safe',         # bytes ≈ 0 (headers partiels seulement)
        'protocol_tcp': 'Susp',   # tcp ↑ (connexions maintenues ouvertes)
        'protocol_udp': 'Safe',
        'protocol_icmp': 'Safe',
        'tcp_flags': 'Anom',      # syn=ATK, fin=NOR, R_fin_from_syn=ATK fort
                                  # ← discriminateur clé vs DATA_EXFIL (R_fin_from_syn=NOR)
        'connections': 'Anom',    # flows ↑↑ (connexions maintenues)
        'entropy': 'Safe',        # source unique → entropie faible (Hansen 2009)
        # packet_size : non inclus — la taille des headers HTTP partiels (~200B)
        # est variable selon l'implémentation de l'attaque (Hansen 2009 ne la contraint pas).
        # Inclure packet_size:Anom constituerait un ajustement a posteriori sur les données
        # d'injection (obs. att=8.0) sans ancrage littérature indépendant — exclu.
        'reconstruction': 'Anom',
    },

    # ── DATA EXFILTRATION ─────────────────────────────────────────────────
    # Connexions complètes (fin ≈ syn) → R_fin_from_syn ≈ NOR
    'DATA_EXFIL': {
        'volume': 'Susp',
        'protocol_tcp': 'Susp',
        'protocol_udp': 'Safe',
        'protocol_icmp': 'Safe',
        'tcp_flags': 'Safe',      # syn=NOR, fin=NOR, R_fin_from_syn=NOR (fermeture normale)
                                  # ← distingue de SLOWLORIS (R_fin_from_syn=ATK)
        'connections': 'Anom',
        'entropy': 'Safe',
        'packet_size': 'Anom',
        'reconstruction': 'Susp',
    },

    # Note: QUALIFY_GATE_THRESHOLD est defini separement dans le bloc de synchronisation ci-dessous.

}

# ==============================================================================
# SYNCHRONISATION GLOBALE DES CHEMINS (source unique = haut de config)
# ==============================================================================
CONFIG["RESULTS_DIR"] = CONFIG.get("RESULTS_DIR", f"../results/resultats_{CONFIG['VERSION_NAME']}")
CONFIG["EVIDENCE_CSV_NAME"] = CONFIG.get(
    "EVIDENCE_CSV_NAME", f"evidence_{CONFIG['VERSION_NAME_MODIF']}.csv"
)
CONFIG["METADATA_CSV_NAME"] = CONFIG.get(
    "METADATA_CSV_NAME", f"metadata_{CONFIG['VERSION_NAME']}.csv"
)

# EVAL (scripts d'évaluation principaux)
CONFIG["EVAL"]["RESULTS_DIR"] = CONFIG["RESULTS_DIR"]

# NOISE_ROBUSTNESS (scripts d'évaluation robustesse)
CONFIG["NOISE_ROBUSTNESS"]["EVIDENCE_CSV_NAME"] = CONFIG["EVIDENCE_CSV_NAME"]
CONFIG["NOISE_ROBUSTNESS"]["METADATA_CSV_NAME"] = CONFIG["METADATA_CSV_NAME"]

# Synchronisation QUALIFY_GATE_THRESHOLD dans CONFIG (coherence avec qualify_anomaly.py)
_evt_decl_override = _os_cfg.environ.get("SL_EVT_DECLUSTER_RUN_OVERRIDE")
if _evt_decl_override not in (None, ""):
    CONFIG["EVT_DECLUSTER_RUN"] = int(_evt_decl_override)

CONFIG["QUALIFY_GATE_THRESHOLD"] = CONFIG["EVAL"]["DECISION_THRESHOLD"]

REAL_ATTACKS = {
    'DDOS_ATTACK': [
        {
            'start': '2025-11-12 18:21:13',
            'end': '2025-11-13 10:14:36',
            'reason': 'UDP_DDOS_ATTACK',
            # Type attendu pour qualify_anomaly.py
            'expected_qualif': 'UDP_FLOOD',
        },
    ],
    'NETWORK_OUTAGE_DEC1617': [
        {
            'start': '2025-12-16 12:30:00',
            'end':   '2025-12-17 16:45:00',
            'reason': 'Coupure réseau consécutive ou maintenance étendue',
            'expected_qualif': 'NETWORK_OUTAGE',
        },
    ],
    'NETWORK_OUTAGE_NOV17': [
        {
            'start': '2025-11-17 12:30:00',
            'end':   '2025-11-17 12:45:00',
            'reason': 'Interruption réseau brève 15 min — redémarrage équipement probable. '
                      'Tous résidus Prophet en chute simultanée (z < -2.9), UDP moins affecté. '
                      'Confirmé par analyse résidus qualify_real_anomalies.py.',
            'expected_qualif': 'NETWORK_OUTAGE',
        },
    ],
}

QUALIFY_GATE_THRESHOLD = CONFIG['EVAL']['DECISION_THRESHOLD']
QUALIFY_MIN_LIKELIHOOD = 0.0
QUALIFY_AUTRE_SENSITIVITY = 1.0
# Constante bijection pour la qualification LR heuristique (qualify_anomaly.py).
# NOTE : qualify_anomaly_SBN.py n'utilise plus ce W — il calcule W=K dynamiquement
# (Jøsang §3.5.2 : W=K=nb types d'attaque) dans _sl_bijection.
QUALIFY_W = 3.0
# Lissage temporel de la qualification.
# λ_q proche de 1 → forte inertie (classification stable mais lente à changer).
# λ_q proche de 0 → chaque fenêtre est indépendante (comportement actuel).
# Ref : filtre exponentiel sur séries temporelles (Brown 1956, Gardner 1985).
QUALIFY_TEMPORAL_ENABLED = False
QUALIFY_TEMPORAL_LAMBDA = 0.6  # à calibrer ; tester 0.4-0.8

CONFIG['QUALIFY_OUTAGE_ENABLED'] = True
CONFIG['QUALIFY_OUTAGE_ATK_THRESHOLD'] = 0.50   # 0.30 était trop permissif
CONFIG['QUALIFY_OUTAGE_SAFE_THRESHOLD'] = 0.85

# Uncertainty Maximisation post-qualification (Jøsang 2016, Def. 3.6, Eq. 3.27)
# Amplifie u_qualif pour les anomalies à croyances distribuées (signal nouveauté)
# sans modifier les P(xi) — neutre sur la classification connue.
# Prior implicite : uniforme a(xi)=1/(K+1), seule position défendable sans labels.
# True = activé (recommandé) ; False = comportement précédent (u_qualif brut).
CONFIG['QUALIFY_UM_POST_FUSION'] = True
# Seuil u_qualif pour le signal de nouveauté APRÈS UM (à calibrer sur données réelles).
# UM amplifie fortement u pour les croyances distribuées (inconnu) et modérément
# pour les attaques connues bien classifiées (b_top dominant).
# Simulation théorique (9 catégories, prior uniforme 1/9) :
#   b_top=0.45 (bien classifié) → u_UM ≈ 0.58  ← en dessous du seuil
#   b_top=0.25 (mal classifié)  → u_UM ≈ 0.81  ← au-dessus : incertitude justifiée
#   Inconnu (uniforme)          → u_UM ≈ 1.00  ← signal fort
# Seuil 0.65 = point de départ conservateur. Recalibrer après le premier run
# en vérifiant que toutes les attaques 100%-précision restent sous le seuil.
CONFIG['QUALIFY_UM_NOVELTY_THRESHOLD'] = 0.50


NOISE_SEED = 42    # graine pour inject_at_evidence_level (bruit résiduel)

# ── Paramètres globaux SBN ────────────────────────────────────────────────────
# W_SBN : non utilisé par qualify_anomaly_sbn.py (W=K calculé dynamiquement).
# Conservé pour référence (ex: scripts d'évaluation externes).
W_SBN = 3.0  # obsolète pour le pipeline SBN — voir _sl_bijection

# Paramètres temporels
SBN_TEMPORAL_ENABLED = False  # False = comportement sans mémoire (fenêtre indépendante)
SBN_TEMPORAL_LAMBDA = 0.80  # décroissance par fenêtre (5 min) → après 5 fenêtres : 0.80^5 ≈ 0.33
SBN_TEMPORAL_WEIGHT = 0.30  # poids maximal du prior temporel dans WBF (30% = modéré)

# Seuil signal de nouveauté post-UM (à calibrer)
SBN_NOVELTY_THRESHOLD = 0.65

# ── SBN — Qualification par Subjective Bayesian Network ──────────────────
# Facteur de mise à l'échelle de l'évidence (cf. _evidence_sum_scores)
# max évidence/groupe = (2/3) * SBN_EVIDENCE_SCALE   (score parfait = 1.0, neutralité = 1/3)
# Formule générale (W = K dynamique, Jøsang §3.5.2) :
#   u_min = W / (G * (2/3) * SCALE + W)  où G = nb groupes alignés, W = K = nb types
# Avec G=10 groupes parfaits, K=11 types, SCALE=3.0 :
#   u_min = 11 / (10 * 2/3 * 3.0 + 11) = 11 / (20 + 11) ≈ 0.35  (attaque bien connue)
# Avec G=0 groupes alignés : u_min = W/W = 1.0  (anomalie inconnue)
SBN_EVIDENCE_SCALE = 3.0

# ─── SBN qualification — leviers discriminabilité ────────────────────────────
# Levier 1 : Prior weight W dans la bijection SL multinomiale (Jøsang §3.5.2).
# W=K (défaut Jøsang) rend l'évidence insensible pour K grand (K=11 → u=0.56 typique).
# W=2 = valeur canonique binaire : u = 2/(Σe + 2) → ~0.19 pour la même évidence.
# Ref : Jøsang 2013 §3.2.2 (avertissement W=K large frames) ; R-EDL OpenReview 2023.
# Valeurs : float (ex. 2.0) ou "K" pour le comportement W=K original.
CONFIG["SBN_PRIOR_WEIGHT"] = "K"   # Changer à 2.0 pour tester -> moins bon ; K

# Levier 2 : Mode de calcul de l'évidence par groupe.
# "absolute"    : e(k,g) = max(0, score(k,g) - 1/3) * scale   [original]
# "contrastive" : e(k,g) = max(0, score(k,g) - mean_j score(j,g)) * scale
#                 → L'évidence est nulle pour les groupes où tous les types ont
#                   un score élevé (ex: volume fort = flood → UDP/SYN/ICMP partagent).
#                   Seuls les groupes qui DISCRIMINENT entre types contribuent.
# Ref : Good (1952) Ann. Math. Stat. §4 — Bayes Factor = rapport de vraisemblance relatif.
CONFIG["SBN_EVIDENCE_MODE"] = "absolute"   # Changer à "contrastive" pour tester, fait -> moins bon ; absolute

# Levier 3 : Pondération des groupes par divergence Jensen-Shannon inter-types.
# True  → weight(g) = JSD(P(G=s|k1),...,P(G=s|kK)) normalisé (mean=1.0)
#          Groupes très discriminants (fin_ratio, packet_size) → poids > 1.
#          Groupes partagés (volume pour tous les floods) → poids << 1.
# False → weight(g) = 1.0 pour tous les groupes [original]
# Ref : Yang & Pedersen (1997) ICML — information gain comme poids de feature.
CONFIG["SBN_USE_JSD_WEIGHTS"] = False   # Changer à True pour tester-> moins bon  : False

# Mode de scoring L4 pour la bijection SL :
#   "dot_product" (défaut) : e(k) = Σ_g max(0, Score(k,g) - 1/3) * scale
#                            Simple, interprétable, Good 1952 §4.
#   "log_lr"               : e(k) = Σ_g max(0, log(Score(k,g) / mean_k'[Score])) * scale
#                            Good (1950) Weight-of-Evidence + hypothèse Naive Bayes (Rish 2001).
#                            Plus discriminant car normalisé par la moyenne des types :
#                            les groupes non-discriminants contribuent exactement 0.
SBN_SCORING_MODE = "dot_product"   # "log_lr" disponible mais requiert SBN_COND_OPINIONS calibrées empiriquement

# Seuil de nouveauté sur novelty_lr = 1/LR_dominance (Shafer 1976)
# 0.85 validé sur signatures : connu → 0.47-0.62 / inconnu → 0.996
SBN_LR_NOVELTY_THRESHOLD = 0.71  # recalibré empiriquement (was 0.85)
                                   # Youden in-sample=0.734, gap max_known/unknown=0.041

# ── Gate OUTAGE pré-SBN ──────────────────────────────────────────────────────
# Active le bypass NETWORK_OUTAGE avant la classification SBN.
# Condition : volume_anom > vol_thr ET connections_anom > conn_thr
#             ET volume_anom - max(udp_anom, icmp_anom, tcp_anom) > margin
SBN_OUTAGE_GATE_ENABLED = True
SBN_OUTAGE_PARAMS = {
    'atk_thr':  0.50,   # P_bytes/packets_proj_atk minimum
    'safe_thr': 0.85,   # P_icmp/udp/syn/tcp_proj_safe minimum
}

CONFIG['COMPARE_CONF_THRESHOLD'] = 0.30   # seuil "classification confiante" §7

# ==============================================================================
# MOTEUR DE SUBSTITUTION CROSS-DOMAIN
# Principe : changer ACTIVE_DATASET en haut du fichier suffit pour tout configurer.
# Chaque bloc écrase les valeurs par défaut (dataset brésilien) de CONFIG.
# ==============================================================================

if not ACTIVE_DATASET or ACTIVE_DATASET == "RedeRio":
    # ── Dataset brésilien (RedeRio — UFRJ) ────────────────────────────────────
    # Toutes les valeurs sont déjà dans CONFIG (voir plus haut).
    # On expose uniquement QUALIFY_GROUP_SOURCES et ATTACK_PRIORS (déjà définis module-level).
    CONFIG["ACTIVE_DATASET"]        = ACTIVE_DATASET   # "" ou "RedeRio"
    CONFIG["ATTACK_CATALOG"]        = None   # utilise le catalogue brésilien intégré
    CONFIG["INCLUDE_REAL_ATTACK"]   = True
    CONFIG["QUALIFY_GROUP_SOURCES"] = QUALIFY_GROUP_SOURCES
    CONFIG["ATTACK_PRIORS"]         = ATTACK_PRIORS
    CONFIG["EVAL"]["INCLUDE_REAL_ATTACK"]   = True
    CONFIG["EVAL"]["REAL_ATTACK_CATALOG"]   = CONFIG["EVAL"].get("REAL_ATTACK_CATALOG", [])
    CONFIG["EVAL"]["LEAF_METRICS_TO_AUDIT"] = CONFIG["EVAL"].get("LEAF_METRICS_TO_AUDIT", [])

    CONFIG["SBN_COND_OPINIONS"] = SBN_COND_OPINIONS
    CONFIG["W_SBN"]                   = W_SBN
    CONFIG["SBN_TEMPORAL_ENABLED"]    = SBN_TEMPORAL_ENABLED
    CONFIG["SBN_TEMPORAL_LAMBDA"]     = SBN_TEMPORAL_LAMBDA
    CONFIG["SBN_TEMPORAL_WEIGHT"]     = SBN_TEMPORAL_WEIGHT
    CONFIG["SBN_NOVELTY_THRESHOLD"]   = SBN_NOVELTY_THRESHOLD
    CONFIG["SBN_SCORING_MODE"]        = SBN_SCORING_MODE
    CONFIG["SBN_EVIDENCE_SCALE"]      = SBN_EVIDENCE_SCALE

elif ACTIVE_DATASET == "CESNET-TimeSeries24":
    # ── CESNET-TimeSeries24 — ISP flow data ───────────────────────────────────
    CONFIG["ACTIVE_DATASET"] = "CESNET-TimeSeries24"
    CONFIG["LAMBDA_DECAY"] = 0.50   # vs 0.85 sur RedeRio

    # Versioning
    CONFIG["VERSION_NAME"]       = "CESNET_v1"
    CONFIG["VERSION_NAME_MODIF"] = "CESNET_v1_attacks"
    CONFIG["RESULTS_DIR"]        = f"../results/resultats_{CONFIG['VERSION_NAME']}"
    CONFIG["EVIDENCE_CSV_NAME"]  = f"evidence_{CONFIG['VERSION_NAME_MODIF']}.csv"
    CONFIG["METADATA_CSV_NAME"]  = f"metadata_{CONFIG['VERSION_NAME']}.csv"

    # Chemin et split
    CONFIG["split_date"] = SELECTED_SPLIT

    CONFIG["FPR_TARGET_DECISION"] = 0.01  # cohérent avec violations holdout

    # Purge exclusions brésiliennes
    CONFIG["HOLIDAYS_LIST"]   = []
    CONFIG["TRAIN_EXCLUSIONS"] = []
    CONFIG["DDOS_ATTACK"]      = []

    # Métriques (ou "auto" pour laisser train_v9 découvrir depuis le CSV)
    CONFIG["ACTIVE_METRICS"] = [
        'n_flows', 'n_packets', 'n_bytes',
        'tcp_udp_ratio_packets', 'tcp_udp_ratio_bytes',
        'avg_duration', 'avg_ttl',
        'sum_n_dest_ports', 'sum_n_dest_ip',
    ]

    CONFIG["RECONST_RULES"] = [
        {"target": "n_bytes",              "feature": "n_packets",             "fit_intercept": False},
        {"target": "n_packets",            "feature": "n_flows",               "fit_intercept": False},
        {"target": "sum_n_dest_ports",     "feature": "n_flows",               "fit_intercept": True},
        {"target": "sum_n_dest_ip",        "feature": "n_flows",               "fit_intercept": True},
        {"target": "tcp_udp_ratio_bytes",  "feature": "tcp_udp_ratio_packets", "fit_intercept": True},
    ]

    # Métriques additive pour Prophet (ratios et moyennes peuvent être proches de 0)
    CONFIG["SEASONALITY_ADDITIVE"] = [
        'tcp_udp_ratio_packets', 'tcp_udp_ratio_bytes', 'avg_duration', 'avg_ttl',
    ]

    # Seuils asymétriques : tous "both" par défaut (anomalies dans les deux sens)
    CONFIG["ASYMMETRIC_THRESHOLD_METRICS"] = {
        'n_flows': 'both', 'n_packets': 'both', 'n_bytes': 'both',
        'tcp_udp_ratio_packets': 'both', 'tcp_udp_ratio_bytes': 'both',
        'avg_duration': 'both', 'avg_ttl': 'both',
        'sum_n_dest_ports': 'both', 'sum_n_dest_ip': 'both',
        # RANSAC pairs
        'n_bytes_n_packets': 'pos', 'n_packets_n_flows': 'pos',
        'sum_n_dest_ports_n_flows': 'both', 'sum_n_dest_ip_n_flows': 'pos',
        'tcp_udp_ratio_bytes_tcp_udp_ratio_packets': 'both',
    }

    # Qualification sémantique (CESNET — préfixes prophet_/reconst_)
    CONFIG["QUALIFY_GROUP_SOURCES"] = QUALIFY_GROUP_SOURCES_CESNET

    CONFIG["ATTACK_PRIORS"] = ATTACK_PRIORS_CESNET

    CONFIG["SBN_COND_OPINIONS"] = SBN_COND_OPINIONS_CESNET

    # Injection désactivée (on utilise les vrais labels CESNET)
    CONFIG["ATTACK_CATALOG"]      = []
    CONFIG["INCLUDE_REAL_ATTACK"] = False
    CONFIG["EVAL"]["INCLUDE_REAL_ATTACK"]   = False
    CONFIG["EVAL"]["REAL_ATTACK_CATALOG"]   = []
    CONFIG["EVAL"]["LEAF_METRICS_TO_AUDIT"] = [
        'P_n_bytes', 'P_n_packets', 'P_n_flows',
        'P_tcp_udp_ratio_packets', 'P_sum_n_dest_ports',
    ]

elif ACTIVE_DATASET == "METR-LA":
    # ── METR-LA — Los Angeles traffic speed sensors ────────────────────────────
    CONFIG["ACTIVE_DATASET"] = "METR-LA"

    # Versioning
    CONFIG["VERSION_NAME"]       = "METR_LA_v1"
    CONFIG["VERSION_NAME_MODIF"] = "METR_LA_v1_attacks"
    CONFIG["RESULTS_DIR"]        = f"../results/resultats_{CONFIG['VERSION_NAME']}"
    CONFIG["EVIDENCE_CSV_NAME"]  = f"evidence_{CONFIG['VERSION_NAME_MODIF']}.csv"
    CONFIG["METADATA_CSV_NAME"]  = f"metadata_{CONFIG['VERSION_NAME']}.csv"

    CONFIG["LAMBDA_DECAY"] = 0.99  # vs 0.85 global (incidents persistants)
    CONFIG["FPR_TARGET_DECISION"] = 0.001
    # Justification : avec 1 métrique R²=0.24, un FPR cible de 0.1% est irréaliste
    # (la distribution proj_atk est plate, p99=p99.9 → seuil arbitraire).

    # Fusion hiérarchique : Prophet (groupe) et Reconst (groupe) contribuent avec poids égaux 0.5/0.5.
    # Sémantiquement correct pour 2 sources distinctes. BALANCE_RATIO non nécessaire ici
    # (il serait ignoré car la branche hierarchical intercepte avant _balance_active).
    CONFIG["INTER_METHOD_FUSION"] = "hierarchical"

    # Alternative : MIN_DECISION_THRESHOLD=0.05 pour contourner la calibration EVT instable

    # Chemin et split
    CONFIG["split_date"] = SELECTED_SPLIT

    # Purge exclusions brésiliennes
    CONFIG["HOLIDAYS_LIST"]    = []
    CONFIG["TRAIN_EXCLUSIONS"] = []
    CONFIG["DDOS_ATTACK"]      = []

    # Métriques : une seule (vitesse moyenne agrégée par timestamp)
    CONFIG["ACTIVE_METRICS"] = ['speed', 'speed_std', 'speed_pct_cong']

    # Pas de RANSAC : une seule métrique → pas de paire structurelle possible
    CONFIG["RECONST_RULES"] = [
        {"target": "speed_std", "feature": "speed", "fit_intercept": True,
         "comment": "Relation std~mean sur autoroute (Daganzo 1994 : fundamental diagram)"},
    ]


    # Speed peut être 0 (embouteillage total) → additive obligatoire
    CONFIG["SEASONALITY_ADDITIVE"] = ['speed', 'speed_std', 'speed_pct_cong']

    # Seuils asymétriques :
    # speed          : 'both' (chute = congestion, pic = anomalie capteur)
    # speed_std      : 'pos'  (std ↑ = hétérogénéité anormale ; ↓ jamais anormal)
    # speed_pct_cong : 'pos'  (fraction congestion ↑ = problème ; ↓ impossible anomalie)
    # speed_p10      : 'neg'  (p10 ↓ = capteurs goulot en chute)
    CONFIG["ASYMMETRIC_THRESHOLD_METRICS"] = {
        'speed':           'both',
        'speed_std':       'pos',
        'speed_pct_cong':  'pos',
        'speed_p10':       'neg',
        'speed_std_speed': 'pos',   # reconst pair
    }

    # Qualification : détection d'anomalie de trafic routier (METR-LA — préfixe prophet_)
    CONFIG["QUALIFY_GROUP_SOURCES"] = {
        'speed_level': ['prophet_speed'],  # vitesse globale
        'spatial_spread': ['prophet_speed_std'],  # hétérogénéité spatiale
        'congestion_rate': ['prophet_speed_pct_cong'],  # fraction capteurs en congestion
        'reconstruction': ['reconst_speed_std_from_speed'],
    }
    CONFIG["ATTACK_PRIORS"] = ATTACK_PRIORS_METR_LA

    CONFIG["SBN_COND_OPINIONS"] = {
        'CONGESTION': {
            'speed_level': {'Safe': 0.03, 'Susp': 0.12, 'Anom': 0.85},  # vitesse ↓ fort
            'spatial_spread': {'Safe': 0.15, 'Susp': 0.35, 'Anom': 0.50},  # std modéré (uniforme)
            'congestion_rate': {'Safe': 0.03, 'Susp': 0.12, 'Anom': 0.85},  # pct_cong ↑↑
            'reconstruction': {'Safe': 0.30, 'Susp': 0.40, 'Anom': 0.30},
        },
        'INCIDENT': {
            'speed_level': {'Safe': 0.05, 'Susp': 0.20, 'Anom': 0.75},  # vitesse ↓ fort
            'spatial_spread': {'Safe': 0.03, 'Susp': 0.12, 'Anom': 0.85},  # DISCRIMINATEUR :
            # std ↑↑ = quelques capteurs chutent (localisé) → Chen 2001
            'congestion_rate': {'Safe': 0.10, 'Susp': 0.30, 'Anom': 0.60},  # pct_cong modéré
            'reconstruction': {'Safe': 0.05, 'Susp': 0.20, 'Anom': 0.75},  # std ≠ prédit
        },
        'SENSOR_MALFUNCTION': {
            'speed_level': {'Safe': 0.20, 'Susp': 0.45, 'Anom': 0.35},
            'spatial_spread': {'Safe': 0.03, 'Susp': 0.12, 'Anom': 0.85},  # std ↑ (1 capteur défaillant)
            'congestion_rate': {'Safe': 0.60, 'Susp': 0.30, 'Anom': 0.10},  # peu de capteurs en cong.
            'reconstruction': {'Safe': 0.03, 'Susp': 0.12, 'Anom': 0.85},  # std brise relation avec mean
            # DISCRIMINATEUR vs INCIDENT : reconstruction Anom mais pct_cong Safe
        },
    }

    # Injection desactivee : evaluation sur labels reels (1843 anomalies METR_LA.csv col 'label').
    # SL-ADS F1=0.011 vs IF-fpr-matched F1=0.697 sur ces labels — a rapporter honnetement.
    CONFIG["ATTACK_CATALOG"] = []
    CONFIG["EVAL"]["CATALOG_MODE"] = "real"
    # INJECTION_ARCHIVE (desactivee) :
    # CONFIG["ATTACK_CATALOG"] = [
    #     {"name": "SUDDEN_INCIDENT", "type": "CONGESTION", "start": "2012-05-10 08:15:00",
    #      "duration_h": 1.5, "intensity": "high", "ramp_frac": 0.05,
    #      "signature": {"prophet_speed": (8.5,1.2,0.3), "prophet_speed_std": (8.5,1.2,0.3),
    #                    "prophet_speed_pct_cong": (6.0,3.0,1.0), "reconst_speed_std_from_speed": (7.5,2.0,0.5)}},
    #     {"name": "SLOW_CONGESTION", "type": "CONGESTION", "start": "2012-05-22 16:00:00",
    #      "duration_h": 3.0, "intensity": "medium", "ramp_frac": 0.40,
    #      "signature": {"prophet_speed": (7.5,2.0,0.5), "prophet_speed_std": (5.0,3.5,1.5),
    #                    "prophet_speed_pct_cong": (8.5,1.0,0.5), "reconst_speed_std_from_speed": (3.0,4.0,3.0)}},
    #     {"name": "SENSOR_FAULT_SPIKE", "type": "SENSOR_MALFUNCTION", "start": "2012-06-05 11:00:00",
    #      "duration_h": 0.5, "intensity": "extreme", "ramp_frac": 0.01,
    #      "signature": {"prophet_speed": (3.5,4.5,2.0), "prophet_speed_std": (8.5,1.2,0.3),
    #                    "prophet_speed_pct_cong": (1.0,3.0,6.0), "reconst_speed_std_from_speed": (8.5,1.2,0.3)}},
    # ]
    # CONFIG["EVAL"]["CATALOG_MODE"] = "injected"

    CONFIG["INCLUDE_REAL_ATTACK"] = False
    CONFIG["EVAL"]["INCLUDE_REAL_ATTACK"]   = False
    CONFIG["EVAL"]["REAL_ATTACK_CATALOG"]   = []
    CONFIG["EVAL"]["LEAF_METRICS_TO_AUDIT"] = [
        'P_speed', 'P_speed_std', 'P_speed_pct_cong', 'P_speed_p10',
    ]

elif ACTIVE_DATASET == "GECCO-IoT":
    # ── GECCO-IoT — Water quality monitoring ──────────────────────────────────
    CONFIG["ACTIVE_DATASET"] = "GECCO-IoT"

    CONFIG["WINDOW_SIZE"] = 10  # 10 × 1min = 10min

    # Versioning
    CONFIG["VERSION_NAME"]       = "GECCO_v1"
    CONFIG["VERSION_NAME_MODIF"] = "GECCO_v1_attacks"
    CONFIG["RESULTS_DIR"]        = f"../results/resultats_{CONFIG['VERSION_NAME']}"
    CONFIG["EVIDENCE_CSV_NAME"]  = f"evidence_{CONFIG['VERSION_NAME_MODIF']}.csv"
    CONFIG["METADATA_CSV_NAME"]  = f"metadata_{CONFIG['VERSION_NAME']}.csv"

    # Chemin et split
    CONFIG["split_date"] = SELECTED_SPLIT

    # Purge exclusions brésiliennes
    CONFIG["HOLIDAYS_LIST"]    = []
    CONFIG["TRAIN_EXCLUSIONS"] = []
    CONFIG["DDOS_ATTACK"]      = []

    # Métriques eau
    CONFIG["ACTIVE_METRICS"] = ['Cl', 'pH', 'Redox', 'Trueb']  # Tp et Leit retirés

    # Paires RANSAC : relations physico-chimiques
    CONFIG["RECONST_RULES"] = [
        {"target": "pH", "feature": "Redox", "fit_intercept": True},
        {"target": "Cl", "feature": "Redox", "fit_intercept": True},
        # Leit from Tp retiré (les deux features bruitées)
    ]

    # Toutes les métriques eau peuvent chuter ou monter → mode additive
    CONFIG["SEASONALITY_ADDITIVE"] = ['Cl', 'pH', 'Redox', 'Trueb']

    # Seuils asymétriques :
    # Cl : direction 'neg' seulement — contamination consume le chlore → Cl ↓
    #      (WHO 2017 §11.3 : chlorine demand increase = contamination indicator)
    # Autres : 'both' car excès ou déficit peuvent être anormaux
    CONFIG["ASYMMETRIC_THRESHOLD_METRICS"] = {
        'Cl': 'neg', 'pH': 'both',
        'Redox': 'both', 'Trueb': 'both',
        'pH_Redox': 'both', 'Cl_Redox': 'both',
    }

    CONFIG["LAMBDA_DECAY"] = 0.50  # dans bloc GECCO-IoT

    # Qualification : types de contamination eau (GECCO-IoT — préfixes prophet_/reconst_)
    CONFIG["QUALIFY_GROUP_SOURCES"] = {
        'chemistry': ['prophet_pH', 'prophet_Redox'],
        'organic': ['prophet_Cl', 'prophet_Trueb'],
        'reconstruction': ['reconst_pH_from_Redox', 'reconst_Cl_from_Redox'],
        # physical group supprimé
    }
    CONFIG["ATTACK_PRIORS"] = ATTACK_PRIORS_GECCO

    CONFIG["SBN_COND_OPINIONS"] = SBN_COND_OPINIONS_GECCO

    # GECCO a des labels réels → pas d'injection synthétique
    CONFIG["ATTACK_CATALOG"]      = []
    CONFIG["INCLUDE_REAL_ATTACK"] = False
    CONFIG["EVAL"]["INCLUDE_REAL_ATTACK"]   = False
    CONFIG["EVAL"]["REAL_ATTACK_CATALOG"]   = []
    CONFIG["EVAL"]["LEAF_METRICS_TO_AUDIT"] = [
        'P_pH', 'P_Cl', 'P_Tp', 'P_Redox', 'P_Leit', 'P_Trueb',
    ]

    # FPR cible adapté à GECCO : 0.5% au lieu de 0.001% (défaut RedeRio).
    # Justification : avec seulement 2052 fenêtres de calibration, FPR=0.001% force le
    # seuil au-dessus du maximum observé (δ=0.35) → zéro détection à δ opérationnel.
    # Les données capteurs eau ont des R² faibles (< 0.21) : un seuil plus permissif
    # est nécessaire pour obtenir une détection non-nulle (δ_opt ≈ 0.20, F1≈0.21).
    CONFIG["FPR_TARGET_DECISION"] = 0.01   # 0.5% FPR → seuil ≈ p99.5 de la calibration

else:
    # ── Dataset inconnu : avertissement et fallback sécurisé ──────────────────
    print(f"[CONFIG] ⚠️  ACTIVE_DATASET='{ACTIVE_DATASET}' non reconnu.")
    print(f"[CONFIG]    Ajoutez un bloc 'elif ACTIVE_DATASET == ...' dans le moteur de substitution.")
    CONFIG["ACTIVE_DATASET"]        = ACTIVE_DATASET
    CONFIG["ACTIVE_METRICS"]        = "auto"
    CONFIG["RECONST_RULES"]         = []
    CONFIG["HOLIDAYS_LIST"]         = []
    CONFIG["TRAIN_EXCLUSIONS"]      = []
    CONFIG["DDOS_ATTACK"]           = []
    CONFIG["SEASONALITY_ADDITIVE"]  = []
    CONFIG["ASYMMETRIC_THRESHOLD_METRICS"] = {}
    CONFIG["QUALIFY_GROUP_SOURCES"] = {}
    CONFIG["ATTACK_PRIORS"]         = {}
    CONFIG["ATTACK_CATALOG"]        = []
    CONFIG["INCLUDE_REAL_ATTACK"]   = False
    CONFIG["EVAL"]["INCLUDE_REAL_ATTACK"]   = False
    CONFIG["EVAL"]["REAL_ATTACK_CATALOG"]   = []
    CONFIG["EVAL"]["LEAF_METRICS_TO_AUDIT"] = []

# ==============================================================================
# POST-TRAITEMENT COMMUN (exécuté pour TOUS les datasets)
# ==============================================================================

# 1. CONFLICT_ALPHA dynamique — recalculé ici selon le WINDOW_SIZE actif
#    Formule : α = 1 / K_max, avec K_max = b_prev_max × b_curr_max
#    b_curr_max = WINDOW_SIZE / (WINDOW_SIZE + K)
#    b_prev_max ≈ (2*WINDOW_SIZE) / (2*WINDOW_SIZE + K)  [approximation 2 fenêtres cumulées]
_W_dyn = CONFIG["WINDOW_SIZE"]
_K_dyn = CONFIG["SL_PARAM_K"]
_b_curr = _W_dyn / (_W_dyn + _K_dyn)
_b_prev = (2 * _W_dyn) / (2 * _W_dyn + _K_dyn)
_K_conflict_max = _b_prev * _b_curr
CONFIG["CONFLICT_ALPHA"] = float(1.0 / _K_conflict_max) if _K_conflict_max > 1e-9 else 1.495

# 2. Application de VERSION_SUFFIX (si non vide) — permet de gérer plusieurs expériences
if VERSION_SUFFIX:
    _base_v = CONFIG["VERSION_NAME"]
    _base_m = CONFIG["VERSION_NAME_MODIF"]
    # On applique le suffixe uniquement si pas déjà présent (idempotence)
    if not _base_v.endswith(VERSION_SUFFIX):
        CONFIG["VERSION_NAME"]       = _base_v + VERSION_SUFFIX
        # Reconstruit VERSION_NAME_MODIF proprement (suffix avant "_attacks" n'existe plus)
        CONFIG["VERSION_NAME_MODIF"] = CONFIG["VERSION_NAME"] + "_attacks"
    CONFIG["RESULTS_DIR"]        = f"../results/resultats_{CONFIG['VERSION_NAME']}"
    CONFIG["EVIDENCE_CSV_NAME"]  = f"evidence_{CONFIG['VERSION_NAME_MODIF']}.csv"
    CONFIG["METADATA_CSV_NAME"]  = f"metadata_{CONFIG['VERSION_NAME']}.csv"

# 3. Resynchronisation des chemins (peut avoir changé dans le bloc dataset)
CONFIG["RESULTS_DIR"] = CONFIG.get("RESULTS_DIR",
                                    f"../results/resultats_{CONFIG['VERSION_NAME']}")
CONFIG["EVIDENCE_CSV_NAME"] = CONFIG.get(
    "EVIDENCE_CSV_NAME", f"evidence_{CONFIG['VERSION_NAME_MODIF']}.csv")
CONFIG["METADATA_CSV_NAME"] = CONFIG.get(
    "METADATA_CSV_NAME", f"metadata_{CONFIG['VERSION_NAME']}.csv")
CONFIG["EVAL"]["RESULTS_DIR"] = CONFIG["RESULTS_DIR"]
CONFIG["NOISE_ROBUSTNESS"]["EVIDENCE_CSV_NAME"] = CONFIG["EVIDENCE_CSV_NAME"]
CONFIG["NOISE_ROBUSTNESS"]["METADATA_CSV_NAME"] = CONFIG["METADATA_CSV_NAME"]

# 3. Cohérence seuil gate
CONFIG["QUALIFY_GATE_THRESHOLD"] = CONFIG["EVAL"]["DECISION_THRESHOLD"]

# 4. Exposition module-level de QUALIFY_GROUP_SOURCES et ATTACK_PRIORS
#    (utilisés par qualify_anomaly.py via 'from config import ...')
# Fallback sur le dict module-level (défini ci-dessus, ligne 856) si CONFIG n'a pas la clé.
# Python évalue le côté droit AVANT l'assignation — la valeur courante de QUALIFY_GROUP_SOURCES
# (ligne 856) est donc correctement utilisée comme fallback pour RedeRio.
QUALIFY_GROUP_SOURCES = CONFIG.get("QUALIFY_GROUP_SOURCES", QUALIFY_GROUP_SOURCES)
ATTACK_PRIORS         = CONFIG.get("ATTACK_PRIORS", {})

"""
PATCH - Remplacer le bloc "ABLATION" dans CONFIG par ce dictionnaire.
Toutes les clés utilisent la nomenclature reconnue par _LABEL_MAP dans run_ablation.py.
Les nouvelles clés sont documentées avec leur justification scientifique.

État du système (référence — PATCH "uniform-as-reference" 2026-04-29) :
  WBF_WEIGHT_MODE = "uniform"             ← matche la config production (config.py L261)
  C3_WEIGHT_MODE  = "uniform"             (uniform surpasse r2_static sur 17 métriques)
  BALANCE_RATIO   = 1.0
  RECONST_ATTACK_RELIABILITY = 1.0
  INTER_METHOD_FUSION = "wbf"             (PATCH M-11)
  δ auto-calibré ≈ 0.129  (FPR_target=1%, proj_atk)
  Résultat reference (run da8ab988fddaf681, 10 steps, 14/14 attaques) :
    F1_micro = 0.784  [IC95% 0.760-0.807]
    F1_macro = 0.885
    MCC      = 0.772
    Accuracy = 0.975
    FPR_op   = 1.64%
  Variantes "isolated" historiques (préfixe "_isolated") utilisaient
  trust_discount comme base ; conservées pour traçabilité mais dépréciées.
  La pathologie trust_discount × R²<0 est documentée dans
  docs/audit/trust_discount_r2_analysis.md et §5.3.3 honest_limitations.md.
"""

_ABLATION_RUNS = {

    # ==========================================================================
    # RÉFÉRENCE — Full SL-ADS opérationnel
    # Configuration EXACTE du système de publication (matche config.py L261).
    # PATCH "uniform-as-reference" 2026-04-29 : la référence passe de
    # trust_discount à uniform suite à la confirmation empirique de la
    # pathologie R²<0 → poids inversés (cf. trust_discount_r2_analysis.md).
    # ==========================================================================
    "full_sl": {
        "lambda": 0.85,
        "wbf_uniform": True,           # poids WBF uniformes — matche production
        "use_reconst": True,
        "use_prophet": True,
        "use_cbf": True,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",   # uniforme = meilleur sur 17 métriques
        "wbf_weight_mode": "uniform",  # ← changé de "trust_discount" (legacy)
        # Toutes les valeurs par défaut → comportement opérationnel exact
    },

    # ==========================================================================
    # GROUPE 1 — Ablation des composants (isolés, un seul changement / run)
    # Principe : Full SL MOINS exactement un composant.
    # PATCH "uniform-as-reference" 2026-04-29 : la base est désormais uniform
    # (matche full_sl ci-dessus).  Les anciennes variantes "_isolated" qui
    # utilisaient trust_discount sont conservées dans le groupe LEGACY ci-dessous
    # pour traçabilité historique uniquement.
    # Ref : Meyes et al. (2019) arXiv:1901.08644 — standard ablation methodology
    # ==========================================================================

    # C1 — Conflict-Aware Ageing (λ dynamique vs fixe)
    "no_c1_isolated": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "conflict_aware": False,       # ← seul changement
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
        "wbf_weight_mode": "uniform",
    },

    # CBF inter-méthode
    "no_cbf_isolated": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "use_cbf": False,              # ← seul changement
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
        "wbf_weight_mode": "uniform",
    },

    # Branche Prophet seule
    "prophet_only_isolated": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": False,          # ← seul changement
        "use_prophet": True,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
        "wbf_weight_mode": "uniform",
    },

    # Branche Reconst seule
    "reconst_only_isolated": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": False,          # ← seul changement
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
        "wbf_weight_mode": "uniform",
    },

    # C4/EDP — Empirical Dirichlet Prior
    "no_edp_isolated": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "conflict_aware": True,
        "adaptive_base_rate": False,   # ← seul changement : prior uniforme [1/3,1/3,1/3]
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
        "wbf_weight_mode": "uniform",
    },

    # PATHOLOGIE TRUST_DISCOUNT — démontre la régression F1 (0.811→0.566)
    # Cette variante est gardée pour montrer la pathologie dans la table d'ablation.
    # full_sl est désormais uniform ; cette variante = full_sl AVEC trust_discount activé.
    # Ref : docs/audit/trust_discount_r2_analysis.md (R²-négatif sur 5/12 modèles Prophet)
    # Renommée de "no_trust_discount_isolated" (sémantique inversée par PATCH 2026-04-29).
    "trust_discount_legacy": {
        "lambda": 0.85,
        "wbf_uniform": False,          # ← seul changement vs full_sl : trust_discount actif
        "use_reconst": True,
        "use_prophet": True,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
        "wbf_weight_mode": "trust_discount",  # PATHOLOGIE — ne pas utiliser en production
    },

    # ==========================================================================
    # GROUPE 2 — Ablation ageing (λ sensitivity)
    # Justification : λ contrôle la demi-vie mémorielle (Jøsang §16.2.2, Eq. 16.5)
    # λ=0.00 : pas de mémoire (chaque fenêtre indépendante)
    # λ=0.85 : demi-vie ≈ 31 fenêtres ≈ 155 min (opérationnel)
    # λ=0.99 : demi-vie ≈ 693 fenêtres ≈ 57h (mémoire quasi-permanente)
    # ==========================================================================
    "no_ageing + uniform": {
        "lambda": 0.00,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
    },
    # Sensibilités λ sont ajoutées hardcodées dans run_ablation_v2.py (λ=0.50/0.85/0.99)

    # ==========================================================================
    # GROUPE 3 — Sensibilité W (bijection SL, Jøsang Def. 3.9)
    # W=2 : constante de Laplace (littérature SL standard)
    # W=3 : canonique pour espace ternaire (Jøsang §3.5.2) ← production
    # W=4 : conservateur (plus de preuves requises avant convergence)
    # Impact attendu : W plus faible → u plus faible → croyances plus affirmatives
    # ==========================================================================
    "w2_sensitivity": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 2.0,             # ← seul changement
        "c3_weight_mode": "uniform",
        "wbf_weight_mode": "uniform",
    },
    "w4_sensitivity": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 4.0,             # ← seul changement
        "c3_weight_mode": "uniform",
        "wbf_weight_mode": "uniform",
    },

    # ==========================================================================
    # GROUPE 4 — Contextual Discounting de la Reconst (CD-alpha)
    # OBJECTIF PRINCIPAL : récupérer SLOWLORIS
    # Mécanisme : applique α=[1.0, 1.0, α_attack] sur l'opinion Reconst agrégée
    #   AVANT la fusion CBF. α_safe=1.0 (on fait confiance à Reconst pour "Safe")
    #   α_attack < 1.0 (on ne lui fait pas confiance pour "Attack").
    # Ref : Mercier, Quost & Denoeux (2006) — apply_contextual_discount()
    # Justification physique : pendant Slowloris, les relations bytes~packets
    #   restent normales (pas de volume) → Reconst génère Safe certain → dilue CBF.
    #   Le CD dit : "ta certitude Safe est valide mais ne m'empêche pas de voir Attack".
    # ==========================================================================
    "cd_alpha_0.00": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
        "wbf_weight_mode": "uniform",
        "reconst_attack_reliability": 0.00,  # Reconst ignorée pour "attack"
    },
    "cd_alpha_0.05": {
        "lambda": 0.85, "wbf_uniform": True,
        "use_reconst": True, "use_prophet": True,
        "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
        "c3_weight_mode": "uniform", "wbf_weight_mode": "uniform",
        "reconst_attack_reliability": 0.05,
    },
    "cd_alpha_0.10": {
        "lambda": 0.85, "wbf_uniform": True,
        "use_reconst": True, "use_prophet": True,
        "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
        "c3_weight_mode": "uniform", "wbf_weight_mode": "uniform",
        "reconst_attack_reliability": 0.10,  # Valeur recommandée par design
    },
    "cd_alpha_0.20": {
        "lambda": 0.85, "wbf_uniform": True,
        "use_reconst": True, "use_prophet": True,
        "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
        "c3_weight_mode": "uniform", "wbf_weight_mode": "uniform",
        "reconst_attack_reliability": 0.20,
    },
    "cd_alpha_0.50": {
        "lambda": 0.85, "wbf_uniform": True,
        "use_reconst": True, "use_prophet": True,
        "conflict_aware": True, "adaptive_base_rate": True, "sl_param_k": 3.0,
        "c3_weight_mode": "uniform", "wbf_weight_mode": "uniform",
        "reconst_attack_reliability": 0.50,
    },

    # ==========================================================================
    # GROUPE 5 — WBF inter-méthode (production / ex-"No CBF")
    # OBJECTIF : expliciter la fusion WBF(P,R) utilisée par la production
    # et conserver une sensibilité comparable aux anciens runs "No CBF".
    # CBF (Eq. 12.14) = addition de preuves → une source Safe certaine absorbe
    #   les preuves Attack de l'autre source (problème Slowloris documenté).
    # WBF (Eq. 12.22) = moyenne pondérée par confiance → Safe réduit Attack
    #   mais ne l'annule pas. Propriété utile quand les sources sont hétérogènes.
    # Ref : Jøsang (2016) §12.3 vs §12.5 — CBF vs WBF properties
    # ==========================================================================
    "wbf_inter_method_isolated": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
        "wbf_weight_mode": "uniform",
        "inter_method_fusion": "wbf",  # ← seul changement vs full_sl
    },

    # ==========================================================================
    # GROUPE 6 — Combinaisons pour Slowloris (runs nouveaux)
    # OBJECTIF : tester les synergies entre mécanismes pour SLOWLORIS
    # Justification : CD-alpha et WBF-inter agissent sur des points différents
    #   du pipeline. Leur combinaison peut récupérer Slowloris sans dégrader
    #   les autres attaques.
    # ==========================================================================

    # CD-alpha + WBF inter-méthode (synergie maximale pour Slowloris)
    "cd_0.10_wbf_inter": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
        "wbf_weight_mode": "uniform",
        "reconst_attack_reliability": 0.10,
        "inter_method_fusion": "wbf",
    },

    # CD-alpha_0.20 + WBF (second point de la courbe)
    "cd_0.20_wbf_inter": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
        "wbf_weight_mode": "uniform",
        "reconst_attack_reliability": 0.20,
        "inter_method_fusion": "wbf",
    },

    # W=2 + CD-alpha_0.10 (croyances plus affirmatives ET moins dilution Reconst)
    "w2_cd_0.10": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 2.0,             # bijection plus affirmative
        "c3_weight_mode": "uniform",
        "wbf_weight_mode": "uniform",
        "reconst_attack_reliability": 0.10,
    },

    # ==========================================================================
    # GROUPE 7 — Balance Ratio (rééquilibrage Prophet vs Reconst avant CBF)
    # Justification : CBF ≡ addition de preuves (Theorem 12.2).
    #   Avec 12 métriques Prophet et 5 Reconst, Prophet accumule ~2.4× plus
    #   de preuves → biais CBF en faveur de Prophet.
    #   balance_ratio="auto" = N_p/N_r = 12/5 = 2.4 → multiplie preuves Reconst
    # Ref : Jøsang (2016) Theorem 12.2, Eq. 12.17
    # ==========================================================================
    "balance_auto": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
        "wbf_weight_mode": "uniform",
        "balance_ratio": "auto",       # N_prophet/N_reconst = 12/5 = 2.4
    },

    # ==========================================================================
    # GROUPE 7b — Fusion Hiérarchique à 2 niveaux (sémantiquement correct en SL)
    # Chaque groupe (Prophet, Reconst) = 1 source d'information distincte.
    # → WBF inter-méthode avec poids ÉGAUX [0.5, 0.5], indépendamment de N_p et N_r.
    # Vs balance_ratio : ne surestime pas la certitude, n'ajoute pas de preuves fictives.
    # Vs wbf confidence : ne laisse pas la confiance relative biaiser la fusion.
    # Ref : Jøsang (2016) §12.3 — fusion de sources hétérogènes à poids fixe.
    # ==========================================================================
    "hierarchical_fusion": {
        "lambda": 0.85,
        "wbf_uniform": True,           # poids uniformes au niveau intra-groupe (C3 off)
        "use_reconst": True,
        "use_prophet": True,
        "use_cbf": True,               # intercepté avant CBF par inter_method_fusion
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
        "wbf_weight_mode": "uniform",
        "inter_method_fusion": "hierarchical",  # poids égaux 0.5/0.5 entre groupes
    },

    # ==========================================================================
    # GROUPE 8 — C3 : mode de pondération intra-WBF (sensibilité)
    # Ces runs utilisent des poids uniformes (wbf_uniform=True) pour isoler
    # l'effet du mode C3 sans interaction avec trust_discount.
    # Résultat attendu : "uniform" reste optimal (cf. mémoire v14 +0.071 F1_cov).
    # ==========================================================================
    "uniform_weights": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
    },
    "c3_prophet_interval uniform": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "use_cbf": True,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "prophet_interval",
    },
    "c3_online_rmse uniform": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "use_cbf": True,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "online_rmse",
    },

    # ==========================================================================
    # GROUPE 9 — Ablation EDP avec poids uniformes (variante "no_edp")
    # Teste l'effet de l'EDP (C4) dans le contexte uniforme.
    # Différent de no_edp_isolated qui est en mode trust_discount.
    # ==========================================================================
    "no_edp_uniform": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "conflict_aware": True,
        "adaptive_base_rate": False,   # EDP désactivé → prior [1/3,1/3,1/3]
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
    },

    # ==========================================================================
    # GROUPE 10 — Ablation Prophet/Reconst Only avec poids uniformes
    # Permet d'isoler la contribution de chaque branche dans le contexte
    # de publication (poids uniformes = configuration rapportée).
    # ==========================================================================
    "prophet_only uniform": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": False,
        "use_prophet": True,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
    },
    "reconst_only uniform": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": False,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
    },
    "no_cbf uniform": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "use_cbf": False,
        "conflict_aware": True,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
    },
    "no_c1 uniform": {
        "lambda": 0.85,
        "wbf_uniform": True,
        "use_reconst": True,
        "use_prophet": True,
        "conflict_aware": False,
        "adaptive_base_rate": True,
        "sl_param_k": 3.0,
        "c3_weight_mode": "uniform",
    },
}

# NOTE POUR run_ablation.py :
# Ajouter les labels des nouveaux runs dans _LABEL_MAP.
# PATCH 2026-04-29 : "isolated" runs ont basculé en uniform (matche nouvelle
# référence full_sl).  La pathologie trust_discount est exposée par la nouvelle
# variante "trust_discount_legacy".  L'ancien "no_trust_discount_isolated" est
# conservé comme alias rétro-compat (sa sémantique correspond désormais à full_sl).
_NEW_LABEL_MAP_ENTRIES = {
    "trust_discount_legacy":       "Trust-Discount [legacy, R²-pathology — F1↓0.245]",
    "no_trust_discount_isolated":  "No Trust Discount [legacy, doublon de full_sl]",
    "cd_0.10_wbf_inter":           "CD α=0.10 + WBF inter-méthode [Slowloris fix]",
    "cd_0.20_wbf_inter":           "CD α=0.20 + WBF inter-méthode",
    "w2_cd_0.10":                  "W=2 + CD α=0.10 [affirmative + Slowloris fix]",
    "no_edp_isolated":             "No C4/EDP — prior uniforme [isolated]",
    "no_c1_isolated":              "No C1 — fixed λ [isolated]",
    "no_cbf_isolated":             "WBF inter-method [isolated production duplicate]",
    "prophet_only_isolated":       "Prophet Only [isolated]",
    "reconst_only_isolated":       "Reconst Only [isolated]",
    "wbf_inter_method_isolated":   "WBF inter-method [explicit production WBF]",
    "balance_auto":                "Balance Ratio auto (N_p/N_r=2.4) [CBF bias fix]",
    "w2_sensitivity":              "W=2 — bijection Laplace [sensibilité]",
    "w4_sensitivity":              "W=4 — bijection conservatrice [sensibilité]",
    "cd_alpha_0.00":               "CD α=0.00 — Reconst ignorée pour attack",
    "cd_alpha_0.05":               "CD α=0.05 — Reconst très peu fiable attack",
    "cd_alpha_0.10":               "CD α=0.10 — Reconst peu fiable attack [recommandé]",
    "cd_alpha_0.20":               "CD α=0.20 — Reconst modérément fiable attack",
    "cd_alpha_0.50":               "CD α=0.50 — Reconst semi-fiable attack",
}

# Note: Marimo notebooks resolve their project root from ``__file__`` at
# import time (see ``src/sl_ads/notebooks/*.py``) and do not consume a
# config-level base-directory constant.

# ==============================================================================
# PATCH m-07 / F25 — Overrides env var (ablation harness)
# ==============================================================================
# Permet à `ablation_nan_ffill.py` de surcharger `NAN_FFILL_LIMIT` pour
# chaque run du sweep sans modifier ce fichier. Placé en toute fin de
# config.py pour que l'override l'emporte quoi qu'il arrive.
#
# Exemple : ``SL_NAN_FFILL_LIMIT_OVERRIDE=5 python run_full_sl_ads.py``
# impose NAN_FFILL_LIMIT=5 pour la session en cours.
#
# Règle : valeurs non entières ou négatives → ignorées (warning stdout).
import os as _os_ab
_nan_ff_env = _os_ab.environ.get("SL_NAN_FFILL_LIMIT_OVERRIDE")
if _nan_ff_env is not None and _nan_ff_env.strip() != "":
    try:
        _v = int(_nan_ff_env)
        if _v < 0:
            raise ValueError(f"must be >= 0, got {_v}")
        CONFIG["NAN_FFILL_LIMIT"] = _v
        print(f"[CONFIG] NAN_FFILL_LIMIT overridden via env var → {_v}")
    except (TypeError, ValueError) as _exc:
        print(f"[CONFIG][WARN] Ignoring invalid "
              f"SL_NAN_FFILL_LIMIT_OVERRIDE='{_nan_ff_env}' ({_exc})")


# ==============================================================================
# PATCH TASK-38 (audit_codex MAJ-03, 2026-04-27) — base default declaration
# ==============================================================================
# SBN novelty threshold on the raw uncertainty mass u_raw of the residual
# opinion.  When u_raw exceeds this value the qualifier marks the window
# as ``qual_status='autre_anomalie'`` (novelty / outside the modelled
# regime) instead of attributing it to one of the known cause families.
#
# Value 0.82 is the operating point retained by the M-07 sensitivity
# study (cf. ``ablation_sbn_novelty.py`` results in
# ``CONSOLIDATED_AUDIT_REVIEW.md`` §3.2).  audit_codex MAJ-03 flagged
# that the previous code only declared this constant indirectly inside
# the env-var override block, so a missing override silently fell back
# to the consumer-side default 0.82 in ``qualify_anomaly_sbn.py``.  We
# now bind the value to ``CONFIG`` explicitly so that:
#   - any consumer can read it without needing a default in its own get();
#   - the value is logged at startup with the rest of CONFIG;
#   - the unit test ``test_audit_codex_remediation_20260427`` can assert
#     CONFIG["SBN_NOVELTY_U_RAW_THRESHOLD"] is present.
CONFIG["SBN_NOVELTY_U_RAW_THRESHOLD"] = 0.82


# ==============================================================================
# PATCH M-07 / F10 — Override env var pour SBN_NOVELTY_U_RAW_THRESHOLD
# ==============================================================================
# Permet à `ablation_sbn_novelty.py` de surcharger le seuil u_raw utilisé
# par `qualify_anomaly_sbn.py` pour déclencher ``qual_status=autre_anomalie``
# (détection de nouveauté).
#
# Motivation (revue M-07 / F10) : le seuil par défaut 0.82 est heuristique
# ; l'article doit reporter un tableau de sensibilité sur
# ``SBN_NOVELTY_U_RAW_THRESHOLD ∈ {0.70, 0.75, 0.82, 0.85, 0.90}``.
#
# Exemple :
#   SL_SBN_NOVELTY_U_RAW_THRESHOLD_OVERRIDE=0.75 python run_full_sl_ads.py
# impose ``CONFIG["SBN_NOVELTY_U_RAW_THRESHOLD"] = 0.75`` pour la session.
#
# Règle : valeurs non numériques ou hors [0, 1] → ignorées (warning stdout).
_u_nov_env = _os_ab.environ.get("SL_SBN_NOVELTY_U_RAW_THRESHOLD_OVERRIDE")
if _u_nov_env is not None and _u_nov_env.strip() != "":
    try:
        _v = float(_u_nov_env)
    except (TypeError, ValueError) as _exc:
        print(f"[CONFIG][WARN] Ignoring non-numeric "
              f"SL_SBN_NOVELTY_U_RAW_THRESHOLD_OVERRIDE='{_u_nov_env}' ({_exc})")
    else:
        if 0.0 <= _v <= 1.0:
            CONFIG["SBN_NOVELTY_U_RAW_THRESHOLD"] = _v
            # Use ASCII arrow to stay robust against cp1252 / non-utf8 stdouts.
            print(f"[CONFIG] SBN_NOVELTY_U_RAW_THRESHOLD overridden via env var -> {_v:.3f}")
        else:
            print(f"[CONFIG][WARN] Ignoring out-of-range "
                  f"SL_SBN_NOVELTY_U_RAW_THRESHOLD_OVERRIDE={_v} (must be in [0.0, 1.0])")


# ==============================================================================
# PATCH M-11 / fusion-dependence — Override env var pour INTER_METHOD_FUSION
# ==============================================================================
# Permet aux ablations d'opérateurs SL de basculer dynamiquement entre les
# modes inter-méthode sans édition de ce fichier.  Le cas historique M-11
# compare WBF à CBF sous dépendance Prophet/Reconst ; l'audit 2026-05-06
# étend le sweep à ABF, BCF, CCF, MinBF et MaxBF.
#
# Exemple :
#   SL_INTER_METHOD_FUSION_OVERRIDE=abf python run_full_sl_ads.py --from-step opinions --to-step eval_injection
# compare ABF à la WBF de référence sur une même base.
#
# Règle : valeurs non reconnues -> ignorées (warning stdout).
_fusion_env = _os_ab.environ.get("SL_INTER_METHOD_FUSION_OVERRIDE")
if _fusion_env is not None and _fusion_env.strip() != "":
    _v = _fusion_env.strip().lower()
    _valid_fusion_modes = ("wbf", "abf", "cbf", "bcf", "ccf", "minbf", "maxbf", "hierarchical")
    if _v in _valid_fusion_modes:
        CONFIG["INTER_METHOD_FUSION"] = _v
        print(f"[CONFIG] INTER_METHOD_FUSION overridden via env var -> {_v}")
    else:
        print(f"[CONFIG][WARN] Ignoring unknown "
              f"SL_INTER_METHOD_FUSION_OVERRIDE='{_fusion_env}' "
              f"(expected one of: {', '.join(_valid_fusion_modes)})")

_thr_fusion_modes_env = _os_ab.environ.get("SL_THRESHOLD_CALIBRATION_FUSION_MODES")
if _thr_fusion_modes_env is not None and _thr_fusion_modes_env.strip() != "":
    _requested_modes = [
        _m.strip().lower()
        for _m in _thr_fusion_modes_env.split(",")
        if _m.strip()
    ]
    _valid_fusion_modes = ("wbf", "abf", "cbf", "bcf", "ccf", "minbf", "maxbf", "hierarchical")
    _accepted_modes = [_m for _m in _requested_modes if _m in _valid_fusion_modes]
    _rejected_modes = [_m for _m in _requested_modes if _m not in _valid_fusion_modes]
    if _accepted_modes:
        CONFIG["THRESHOLD_CALIBRATION_FUSION_MODES"] = _accepted_modes
        print("[CONFIG] THRESHOLD_CALIBRATION_FUSION_MODES overridden "
              f"via env var -> {_accepted_modes}")
    if _rejected_modes:
        print("[CONFIG][WARN] Ignoring unknown threshold calibration "
              f"fusion modes: {_rejected_modes}")

CONFIG["IF_CONTAMINATION_DEFAULT"] = 0.02  # fallback si train sans labels
CONFIG["IF_CONTAMINATION_MODE"] = "train"  # "train" | "fixed" | "auto_test" (déprécié)

# Sensitivity ladder used by ``run_ablation_v2.py`` Phase 2.  Headline IF
# contamination is always ``IF_CONTAMINATION_DEFAULT``; the other values
# are reported in the appendix sensitivity table only — no test-set
# tuning (audit_codex CRIT-03, PATCH TASK-35).
CONFIG["IF_CONTAMINATION_LADDER"] = [0.01, 0.02, 0.03, 0.035, 0.04, 0.05]


# ==============================================================================
# PATCH TASK-39 (audit_codex MAJ-08, 2026-04-27) — Calibration constants
# ==============================================================================
# These three constants control the EVT/FPR threshold-calibration path
# in ``train_v10.py`` (cf. ``_compute_training_proj_atk()`` and
# ``_calibrate_decision_threshold()``).  They were previously read via
# ``CONFIG.get(<key>, <default>)`` with no base declaration, which meant
# changing them required editing ``train_v10.py`` directly.  audit_codex
# MAJ-08 flagged this as a reproducibility hazard: the published
# threshold depends on these three numbers and the paper must be able
# to cite them centrally.  We now declare each in CONFIG so that:
#   * ``run_ablation_v2.py`` can sweep them;
#   * the unit test asserts each is present;
#   * any tuning is logged centrally rather than buried in train_v10.py.
#
# CALIB_BIJECTION_FLOOR_TOL : tolerance around the SL bijection floor
#   used to flag "essentially zero" projected_prob values during
#   calibration.  Keeping 0.01 reproduces the v10 baseline.
# CALIB_AGEING_WIN_FRACTION : fraction of an effective ageing window
#   (1/(1-λ)) over which the calibration survey aggregates evidence.
#   0.5 = half-life equivalent — Jøsang §3.5.4 ageing operator.
# CALIB_SPARSITY_CUTOFF : projected_prob values below this threshold
#   are treated as numerically zero (excluded from the EVT tail fit).
#   1e-9 matches the float64 effective precision floor used elsewhere
#   in the SL bijection (Def. 3.9, Jøsang 2016).
CONFIG["CALIB_BIJECTION_FLOOR_TOL"] = 0.01
CONFIG["CALIB_AGEING_WIN_FRACTION"] = 0.5
CONFIG["CALIB_SPARSITY_CUTOFF"]     = 1e-9


# ==============================================================================
# PATCH TASK-46 (audit_codex MAJ-05, 2026-04-27) — CESNET timestamp policy
# ==============================================================================
# CESNET-TimeSeries24 ships ``id_time`` as an integer counter rather
# than a wall-clock timestamp.  ``cesnet_adapter.py`` synthesizes a
# 10-minute step calendar from a fixed anchor.  The two keys below
# control how strict that policy is.
#   CESNET_TIMESTAMP_MODE :
#     - "fabricated_warning" : synthesize + emit UserWarning (default).
#     - "fabricated_silent"  : synthesize without warning (CI-friendly).
#     - "reject"             : refuse to load CESNET (use when calendar-
#                              aware analyses are mandatory).
#   CESNET_TIMESTAMP_ANCHOR : anchor wall-clock for the synthetic axis.
CONFIG["CESNET_TIMESTAMP_MODE"]   = "fabricated_warning"
CONFIG["CESNET_TIMESTAMP_ANCHOR"] = "2024-01-01"


# ==============================================================================
# PATCH TASK-42 (audit_codex MAJ-07, 2026-04-27) — GECCO file loading
# ==============================================================================
# GECCO ships as one CSV today; future redistributions may split per-
# month or per-year.  The previous adapter silently consumed only
# ``files[0]``.  ``GECCO_LOAD_MODE`` controls the behaviour:
#   - "concat" : load and concat all CSVs (sorted lexicographically) —
#                deterministic, no silent data loss (default).
#   - "single" : require exactly one CSV; raise otherwise.
# ``GECCO_LOAD_MODE`` is exposed as a top-level module attribute so the
# adapter can read it via ``import config``.
GECCO_LOAD_MODE = "concat"


# ==============================================================================
# PATCH TASK-53 (uniform-as-reference 2026-04-29) — RANDOM_SEED env override
# ==============================================================================
# Resolves CONFIG["RANDOM_SEED"] from the environment variable
# ``SL_RANDOM_SEED`` when CONFIG declares ``None`` (the default).  This
# lets the multi-seed runner (sl_ads.evaluate.multi_seed) launch parallel
# subprocesses with different seeds without rewriting config.py for each
# value.
#
# Resolution rules :
#   - CONFIG["RANDOM_SEED"] is an int             ⇒ keep as-is (explicit).
#   - CONFIG["RANDOM_SEED"] is None and SL_RANDOM_SEED is set
#     and parses as int                           ⇒ override.
#   - CONFIG["RANDOM_SEED"] is None and SL_RANDOM_SEED is unset / invalid
#                                                 ⇒ default to 0 (the
#                                                   reproducible "single-seed"
#                                                   value used by the
#                                                   published runs).
import os as _os_seed
if CONFIG.get("RANDOM_SEED") is None:
    _env_seed = _os_seed.environ.get("SL_RANDOM_SEED", "").strip()
    if _env_seed:
        try:
            CONFIG["RANDOM_SEED"] = int(_env_seed)
        except ValueError:
            CONFIG["RANDOM_SEED"] = 0
    else:
        CONFIG["RANDOM_SEED"] = 0
RANDOM_SEED = CONFIG["RANDOM_SEED"]
