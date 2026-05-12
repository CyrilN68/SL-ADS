"""
sl_formulas.py — Subjective Logic Operators
============================================
Ref: Jøsang, A. (2016). Subjective Logic: A Formalism for Reasoning Under
     Uncertainty. Springer. ISBN 978-3-319-42337-1.

All equation references (Eq. X.XX) refer to this book.
"""
import numpy as np
import warnings

# CONFLICT MODE — configurable via config.py CONFLICT_MODE
# "belief_mass" (default, backward-compatible)
# "projected_prob" : K on projected probabilities P(x) = b + a·u
# "kl_symmetric"   : symmetric KL divergence on P
try:
    from sl_ads.config import CONFIG as _CONFLICT_CONFIG
    _CONFLICT_MODE = _CONFLICT_CONFIG.get("CONFLICT_MODE", "belief_mass")
    _CONFLICT_KL_TAU = _CONFLICT_CONFIG.get("CONFLICT_KL_TAU", 1.0)
except ImportError:
    _CONFLICT_MODE = "belief_mass"
    _CONFLICT_KL_TAU = 1.0

# Charger W depuis config une seule fois au niveau module (evite l'import dans la boucle)
try:
    from sl_ads.config import CONFIG as _SL_CONFIG
    _WBF_W = float(_SL_CONFIG.get('SL_PARAM_K', 3.0))
    # PATCH M-08 / F11 (2026-04-21) : plafond d'évidence dogmatique configurable
    _SL_EVIDENCE_MAX_FACTOR = float(_SL_CONFIG.get('SL_EVIDENCE_MAX_FACTOR', 1e4))
except ImportError:
    _WBF_W = 3.0
    _SL_EVIDENCE_MAX_FACTOR = 1e4

# PATCH M-08 / F11 : état de log pour éviter le spam — on warn une fois par métrique
# (identifiée par son id mémoire, fallback session si introspection non dispo).
_SL_CAP_WARNED = set()


# ==============================================================================
# CLASSE OPINION MULTINOMIALE
# ==============================================================================

class MultinomialOpinion:
    """
    Opinion multinomiale ternaire : ω = (b, u, a)
    - b : vecteur de croyances [b_safe, b_suspect, b_attack]
    - u : incertitude scalaire
    - a : base rate [a_safe, a_suspect, a_attack]

    Contrainte : sum(b) + u = 1
    """

    def __init__(self, beliefs, u, a=None):
        self.b = np.array(beliefs, dtype=float)
        self.u = float(u)

        if a is not None:
            self.a = np.array(a, dtype=float)
        else:
            self.a = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        # Normalisation de sécurité
        total = np.sum(self.b) + self.u
        if not np.isclose(total, 1.0) and total > 0:
            self.b /= total
            self.u /= total

        if not np.isclose(np.sum(self.b) + self.u, 1.0, atol=1e-6):
            raise ValueError(f"Opinion invalide: Σb + u = {np.sum(self.b) + self.u}")

    def projected_prob(self):
        """Probabilité projetée P(x) = b(x) + a(x)*u (Eq. 3.23)"""
        return self.b + self.a * self.u

    def confidence(self):
        """Confiance c = 1 - u (Eq. 3.43)"""
        return 1.0 - self.u

    def uncertainty_maximized(self):
        """
        Retourne l'opinion uncertainty-maximisée ω̈ (Jøsang 2016, Section 3.6, Eq. 3.27).

        Principe : on cherche le u_max tel que les b_i restent ≥ 0 et au moins un = 0.
        La probabilité projetée P(x) = b(x) + a(x)*u est préservée (invariant).

        Eq. 3.27 :  ü = min_i( P(x_i) / a(x_i) )   pour a(x_i) > 0
        Eq. 3.12 :  b̈(x_i) = P(x_i) - a(x_i) * ü   (recalculé depuis P invariant)

        Cas limites :
            - a(x_i) = 0 → on ignore cette dimension pour le calcul du min
            - opinion déjà dogmatique (u=0) → retourne une copie inchangée
            - opinion vacuuse (u=1) → retourne une copie inchangée (déjà max)
        """
        P = self.projected_prob()  # P(x) = b + a*u  — invariant

        # Candidats : seulement les dimensions où a(x_i) > 0
        candidates = []
        for i in range(len(self.b)):
            if self.a[i] > 1e-12:
                candidates.append(P[i] / self.a[i])

        if not candidates:
            return MultinomialOpinion(self.b.copy(), self.u, self.a.copy())

        u_max = float(np.min(candidates))
        u_max = float(np.clip(u_max, 0.0, 1.0))

        # Recalcul des belief masses depuis P (Eq. 3.12 inversée)
        b_new = P - self.a * u_max
        b_new = np.clip(b_new, 0.0, None)

        # Renormalisation de sécurité (erreurs numériques)
        total = np.sum(b_new) + u_max
        if total > 1e-12 and not np.isclose(total, 1.0):
            b_new /= total
            u_max /= total

        return MultinomialOpinion(b_new, u_max, self.a.copy())

    def __repr__(self):
        return f"Op(Safe={self.b[0]:.3f}, Susp={self.b[1]:.3f}, Atk={self.b[2]:.3f}, U={self.u:.3f})"


# ==============================================================================
# 1. BIJECTION EVIDENCE <-> OPINION (Definition 3.9)
# ==============================================================================

def evidence_to_opinion(r, W=3.0, a=None):
    """
    Bijection Evidence → Opinion (Def. 3.9, cas multinomial).

    Formule :
        b(x_i) = r(x_i) / (W + sum(r))
        u       = W      / (W + sum(r))

    Paramètres :
        r  : vecteur de preuves [r_safe, r_suspect, r_attack]
        W  : constante de la bijection (non-informative prior weight, W=k dans ta config)
        a  : base rate [a_safe, a_suspect, a_attack]
    """
    r = np.array(r, dtype=float)
    total_r = np.sum(r)
    D = total_r + W

    if D == 0:
        return MultinomialOpinion([0, 0, 0], 1.0, a)

    b = r / D
    u = W / D
    return MultinomialOpinion(b, u, a)


def opinion_to_evidence(op, W=3.0, _metric_key=None):
    """
    Bijection Opinion → Evidence (Def. 3.9, inverse).

    Formule :
        r(x_i) = W * b(x_i) / u

    Retourne : vecteur r de preuves

    Cas dogmatique (u → 0) : évidence théoriquement infinie.
    On la plafonne à `W * CONFIG["SL_EVIDENCE_MAX_FACTOR"]` (défaut 1e4) pour
    éviter l'overflow float64 lors de l'accumulation R dans des runs longs.
    Le plafond préserve le comportement qualitatif (score très élevé) sans
    risquer de dépasser la plage float64 (~1.8e308).

    PATCH M-08 / F11 (2026-04-21) :
        - Facteur `1e4` exposé via CONFIG["SL_EVIDENCE_MAX_FACTOR"].
        - Émission d'un `warnings.warn` la 1ʳᵉ fois que le cap est atteint
          (identifié par `_metric_key` si fourni, sinon global session).

    Parameters
    ----------
    op : MultinomialOpinion
    W  : SL prior strength (= nombre d'états multinomiaux, Jøsang Def. 3.9)
    _metric_key : str optionnel — identifiant de métrique pour dé-dupliquer
                  les warnings (un warn par métrique par session).
    """
    _W_MAX = W * _SL_EVIDENCE_MAX_FACTOR
    if op.u < 1e-9:
        # Cas dogmatique : plafonner plutôt que W/1e-12 ≈ 3e11 (risque overflow)
        raw = op.b * (W / max(op.u, 1e-9))
        capped = np.minimum(raw, _W_MAX)
        # Émettre un warning une seule fois par métrique (ou par session si inconnu)
        if (raw > _W_MAX).any():
            key = _metric_key if _metric_key is not None else "__unknown_metric__"
            if key not in _SL_CAP_WARNED:
                _SL_CAP_WARNED.add(key)
                warnings.warn(
                    f"[sl_formulas_v2] opinion_to_evidence: dogmatic opinion "
                    f"(u={op.u:.2e}) capped at W*{_SL_EVIDENCE_MAX_FACTOR:.0e} "
                    f"for metric={key!r}. Further occurrences silenced for this key.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return capped

    return W * op.b / op.u


# ==============================================================================
# 2. MAPPING FUZZY TRAPÉZOÏDAL (Evidence Instantanée)
# ==============================================================================

def fuzzy_trapezoid_relative(z, k_susp, k_atk, transition_pct=0.1):
    """
    Mapping d'erreur absolue z -> vecteur de preuves instantanées [P, S, N].
    P = preuve Safe, S = preuve Suspect, N = preuve Attaque.

    Zone trapézoïdale avec transition douce.

    NOTE : Cette fonction travaille sur |résidu| et ne gère pas les seuils
    asymétriques. Utiliser compute_instantaneous_evidence() pour le cas
    directionnel (pos / neg / both) issu des artefacts train_v10.
    """
    start_doubt = k_susp * (1.0 - transition_pct)
    if z < start_doubt:
        return np.array([1.0, 0.0, 0.0])
    elif z < k_susp:
        width = k_susp - start_doubt
        ratio = (z - start_doubt) / (width + 1e-9)
        return np.array([1.0 - ratio, ratio, 0.0])
    elif z < k_atk:
        width = k_atk - k_susp
        ratio = (z - k_susp) / (width + 1e-9)
        return np.array([0.0, 1.0 - ratio, ratio])
    else:
        return np.array([0.0, 0.0, 1.0])


# ==============================================================================
# 3. FUSION TEMPORELLE AVEC AGEING (Section 16.2.2, Eq. 16.5)
# ==============================================================================

def temporal_ageing_fusion(r_accumulated, r_current, lam):
    """
    Ageing temporel FIXE des preuves (Eq. 16.5 de Jøsang).
    Conservé pour compatibilité. Voir temporal_adaptive_ageing() pour la v2.
    """
    r_accumulated = np.array(r_accumulated, dtype=float)
    r_current = np.array(r_current, dtype=float)
    return lam * r_accumulated + r_current


def compute_asymmetric_escalation_conflict(r_prev, r_curr, W=3.0):
    """
    PATCH TASK-26 (audit_tmp MAJ-05, 2026-04-26) — RENAMED from
    ``compute_conflict_degree`` to make the asymmetric design explicit.
    ────────────────────────────────────────────────────────────────────────

    "Escalation-only" pseudo-conflict between accumulated state and current
    observation, used as the trigger for *Conflict-Aware Ageing*
    (`temporal_adaptive_ageing`).

    This is **NOT** a canonical Subjective-Logic conflict measure
    (Jøsang 2016 §11 / Eq. 12.4 BCF).  It is a deliberately asymmetric
    heuristic that flags only **escalating** transitions (a state where
    the accumulator is calm but the current window is alarming, or vice
    versa) — de-escalation transitions (e.g. ``b_prev[atk] × b_curr[susp]``)
    are *intentionally omitted* because they correspond to normal
    cool-downs, not contradictions.

    For a canonical, fully-symmetric SL conflict on the same belief
    vectors, see :func:`compute_conflict_degree_canonical`.

    Method:
        1. Project the two evidence vectors into opinions via the bijection
           (Def. 3.9 of Jøsang 2016).
        2. Compute K = sum of cross products of beliefs on mutually
           exclusive singletons — but only for *escalating* directions:

            K = b_prev[safe]    × b_curr[attack]    (calm → attack)
              + b_prev[attack]  × b_curr[safe]      (attack → calm — fast cool-down)
              + b_prev[safe]    × b_curr[suspect]   (calm → suspect)
              + b_prev[suspect] × b_curr[attack]    (suspect → attack)

    Omitted on purpose (de-escalation, treated as normal):
        - b_prev[suspect] × b_curr[safe]
        - b_prev[attack]  × b_curr[suspect]

    Parameters:
        r_prev : evidence vector [r_safe, r_susp, r_atk] (accumulated)
        r_curr : evidence vector [r_safe, r_susp, r_atk] (current)
        W      : bijection constant (Def. 3.9)

    Returns: K ∈ [0, 1] (conflict-like score for ageing modulation)
    """
    r_prev = np.array(r_prev, dtype=float)
    r_curr = np.array(r_curr, dtype=float)

    # Projection en croyances via bijection (Def. 3.9)
    D_prev = np.sum(r_prev) + W
    D_curr = np.sum(r_curr) + W

    b_prev = r_prev / D_prev if D_prev > 0 else np.zeros(3)
    b_curr = r_curr / D_curr if D_curr > 0 else np.zeros(3)

    # Conflit : produits croisés sur singletons disjoints
    # Transitions "contradictoires" (montée brutale de sévérité)
    K = (b_prev[0] * b_curr[2]      # safe_prev × attack_curr
       + b_prev[2] * b_curr[0]      # attack_prev × safe_curr
       + b_prev[0] * b_curr[1]      # safe_prev × suspect_curr
       + b_prev[1] * b_curr[2])     # suspect_prev × attack_curr

    return float(np.clip(K, 0.0, 1.0))


def compute_conflict_degree_canonical(r_prev, r_curr, W=3.0):
    """
    PATCH TASK-26 (audit_tmp MAJ-05, 2026-04-26) — NEW canonical version.
    ────────────────────────────────────────────────────────────────────────

    Symmetric belief-mass conflict (Jøsang 2016, Eq. 12.4 BCF, ternary
    multinomial extension). All cross products on mutually exclusive
    singletons are summed, in both directions:

        K = Σ_{i≠j}  b_prev[i] · b_curr[j]
          = (b_prev[safe]   · (b_curr[susp] + b_curr[atk])
           + b_prev[susp]   · (b_curr[safe] + b_curr[atk])
           + b_prev[atk]    · (b_curr[safe] + b_curr[susp]))

    This is the unbiased, fully-symmetric variant. Use this for theoretical
    analysis and unit tests; use :func:`compute_asymmetric_escalation_conflict`
    for production ageing-modulation (intended escalation-only behaviour).

    Parameters: identical to :func:`compute_asymmetric_escalation_conflict`.
    Returns: K ∈ [0, 1].
    """
    r_prev = np.array(r_prev, dtype=float)
    r_curr = np.array(r_curr, dtype=float)

    D_prev = np.sum(r_prev) + W
    D_curr = np.sum(r_curr) + W

    b_prev = r_prev / D_prev if D_prev > 0 else np.zeros(3)
    b_curr = r_curr / D_curr if D_curr > 0 else np.zeros(3)

    # Sum of *all* off-diagonal cross-products — symmetric BCF conflict.
    K = (b_prev[0] * (b_curr[1] + b_curr[2])
       + b_prev[1] * (b_curr[0] + b_curr[2])
       + b_prev[2] * (b_curr[0] + b_curr[1]))

    return float(np.clip(K, 0.0, 1.0))


def compute_conflict_degree(r_prev, r_curr, W=3.0):
    """
    PATCH TASK-26 (audit_tmp MAJ-05, 2026-04-26) — back-compat alias.
    ────────────────────────────────────────────────────────────────────────

    Deprecated public name.  Forwards to
    :func:`compute_asymmetric_escalation_conflict` (the asymmetric
    escalation-only heuristic used by Conflict-Aware Ageing).

    Use :func:`compute_asymmetric_escalation_conflict` for clarity, or
    :func:`compute_conflict_degree_canonical` for the symmetric
    Jøsang-2016 BCF conflict.

    This alias is kept silent (no DeprecationWarning) to avoid log noise
    in the high-throughput temporal-adaptive-ageing inner loop.
    """
    return compute_asymmetric_escalation_conflict(r_prev, r_curr, W=W)



def compute_conflict_degree_projected(r_prev, r_curr, a_prev, a_curr, W=3.0):
    """
    Conflict degree K computed from projected probabilities P(x) = b(x) + a(x)·u
    (Jøsang Eq. 3.23). Incorporates base rate and uncertainty into conflict.

    At u→0: K_proj → K_belief (converges monotonically).
    At u_sys=0.006 (SL-ADS regime): |K_proj − K_belief| < 0.04,
    ranking of conflict events preserved, λ_dyn ordering unchanged.

    Asymmetric exclusion of de-escalation transitions preserved from original K.
    """
    r_prev = np.array(r_prev, dtype=float)
    r_curr = np.array(r_curr, dtype=float)
    a_prev = np.array(a_prev, dtype=float)
    a_curr = np.array(a_curr, dtype=float)
    D_prev = np.sum(r_prev) + W
    D_curr = np.sum(r_curr) + W
    b_prev = r_prev / D_prev if D_prev > 0 else np.zeros(3)
    u_prev = W / D_prev if D_prev > 0 else 1.0
    b_curr = r_curr / D_curr if D_curr > 0 else np.zeros(3)
    u_curr = W / D_curr if D_curr > 0 else 1.0
    P_prev = b_prev + a_prev * u_prev
    P_curr = b_curr + a_curr * u_curr
    K = (P_prev[0]*P_curr[2] + P_prev[2]*P_curr[0]
       + P_prev[0]*P_curr[1] + P_prev[1]*P_curr[2])
    return float(np.clip(K, 0.0, 1.0))


def compute_conflict_degree_kl(r_prev, r_curr, a_prev, a_curr, W=3.0, eps=1e-12):
    """
    Conflict degree K via symmetric KL divergence on projected probabilities.
    KL_sym(P_prev ‖ P_curr) = KL(P‖Q) + KL(Q‖P), normalised to [0,1] via
    K = 1 − exp(−KL_sym / tau)  where tau = CONFIG["CONFLICT_KL_TAU"] (default 1.0).
    """
    r_prev = np.array(r_prev, dtype=float)
    r_curr = np.array(r_curr, dtype=float)
    a_prev = np.array(a_prev, dtype=float)
    a_curr = np.array(a_curr, dtype=float)
    D_prev = np.sum(r_prev) + W
    D_curr = np.sum(r_curr) + W
    b_prev = r_prev / D_prev if D_prev > 0 else np.zeros(3)
    u_prev = W / D_prev if D_prev > 0 else 1.0
    b_curr = r_curr / D_curr if D_curr > 0 else np.zeros(3)
    u_curr = W / D_curr if D_curr > 0 else 1.0
    P = np.clip(b_prev + a_prev * u_prev, eps, None); P /= P.sum()
    Q = np.clip(b_curr + a_curr * u_curr, eps, None); Q /= Q.sum()
    kl_sym = float(np.sum(P * np.log(P/Q)) + np.sum(Q * np.log(Q/P)))
    return float(np.clip(1.0 - np.exp(-kl_sym / _CONFLICT_KL_TAU), 0.0, 1.0))


def temporal_adaptive_ageing(r_accumulated, r_current, lam_base, W=3.0,
                             alpha=1.0, gamma=1.0, conflict_aware=True,
                             a_prev=None, a_curr=None):
    """
    Ageing temporel ADAPTATIF — "Conflict-Aware Ageing".

    Extension de l'Eq. 16.5 (Jøsang Sect. 16.2.2) avec un λ dynamique
    modulé par le degré de conflit entre l'état accumulé et l'observation.

    Principe :
        1. Calculer K = conflit entre r_accumulated et r_current
        2. Amplifier : K_eff = min(1, α × K)
        3. Appliquer la courbe : λ_dyn = λ_base × (1 - K_eff)^γ
        4. R_{τ+1} = λ_dyn · R_τ + r_{τ+1}

    Paramètres de tuning :
        alpha (α) : Amplificateur de conflit.
                    α = 1.0 → comportement original (inertie résiduelle ≈ 23-31%)
                    α = 1/K_max → hard reset exact à contradiction maximale (λ_dyn = 0).
                    Valeur canonique pour W=3 : α = (R_acc+W)(R_win+W)/(R_acc×R_win)
                      = 23×13/(20×10) = 299/200 = 1.495  (config.py CONFLICT_ALPHA).
                    Ce paramètre est passé via compute_opinions sous le nom alpha=CONFLICT_ALPHA.
        gamma (γ) : Exposant de la courbe d'oubli, défaut=1.0
                    γ = 1 → linéaire (original)
                    γ = 2 → quadratique (recommandé : fort reset en conflit,
                            quasi-neutre en stabilité)
                    γ = 3 → cubique (très agressif)

    Effet sur l'incertitude :
        Quand λ_dyn → 0, les preuves accumulées sont effacées.
        Le total (r_safe + r_susp + r_atk) redevient petit,
        donc u = W / (Σr + W) remonte automatiquement via la bijection.

    Paramètres :
        r_accumulated : vecteur de preuves accumulées R_τ
        r_current     : vecteur de preuves instantanées r_{τ+1}
        lam_base      : facteur de longévité de base λ ∈ [0, 1]
        W             : constante de bijection (Def. 3.9)
        alpha         : amplificateur de conflit (≥ 1.0)
        gamma         : exposant courbe d'oubli (≥ 1.0)

    Retourne : (R_{τ+1}, K, λ_dyn)
    """
    r_accumulated = np.array(r_accumulated, dtype=float)
    r_current = np.array(r_current, dtype=float)

    # Store base rates for conflict routing (used if CONFLICT_MODE != belief_mass)
    temporal_adaptive_ageing._a_prev = a_prev if a_prev is not None else np.full(3, 1/3)
    temporal_adaptive_ageing._a_curr = a_curr if a_curr is not None else np.full(3, 1/3)

    if not conflict_aware:
        # Ablation "No C1" : ageing fixe standard (Eq. 16.5 sans adaptation).
        # λ_dyn = λ_base constant, K = 0 (aucun conflit détecté).
        R_new = lam_base * r_accumulated + r_current
        return R_new, 0.0, lam_base

    # Étape 1 : Calcul du conflit (mode configurable via CONFLICT_MODE)
    if _CONFLICT_MODE == 'projected_prob':
        K_raw = compute_conflict_degree_projected(
            r_accumulated, r_current,
            a_prev=getattr(temporal_adaptive_ageing, '_a_prev', np.full(3, 1/3)),
            a_curr=getattr(temporal_adaptive_ageing, '_a_curr', np.full(3, 1/3)),
            W=W)
    elif _CONFLICT_MODE == 'kl_symmetric':
        K_raw = compute_conflict_degree_kl(
            r_accumulated, r_current,
            a_prev=getattr(temporal_adaptive_ageing, '_a_prev', np.full(3, 1/3)),
            a_curr=getattr(temporal_adaptive_ageing, '_a_curr', np.full(3, 1/3)),
            W=W)
    else:
        K_raw = compute_conflict_degree(r_accumulated, r_current, W)

    # Étape 2 : Amplification
    K_eff = float(np.clip(alpha * K_raw, 0.0, 1.0))

    # Étape 3 : λ dynamique avec courbe de puissance
    lam_dyn = lam_base * ((1.0 - K_eff) ** gamma)

    # Étape 4 : Ageing adaptatif (Eq. 16.5 avec λ_dyn)
    R_new = lam_dyn * r_accumulated + r_current

    return R_new, K_eff, lam_dyn


# ==============================================================================
# 4. WEIGHTED BELIEF FUSION — N SOURCES (Section 12.5, Eq. 12.22-12.27)
# ==============================================================================

def fusion_wbf_n_sources(opinions, external_weights=None, W=None):
    """
    Weighted Belief Fusion — evidence-space confidence-weighted averaging
    (Jøsang §12.5, Eq. 12.27 via the opinion-evidence bijection Def. 3.9).

    PATCH M-01 / F01 (2026-04-24) — Nature of this operator
    --------------------------------------------------------
    This implementation realises the WBF *in evidence space*: it maps each
    opinion to its evidence vector r_i (Jøsang Def. 3.9, bijection with
    prior weight W), takes a weighted mean with weights
    ``w_i = external_weight_i × c_i`` (c_i = 1 - u_i, Eq. 3.43), then maps
    the averaged evidence back to an opinion.

    This is *faithful to Eq. 12.27* (confidence-weighted averaging of
    evidence parameters) and supports arbitrary N sources and optional
    external quality weights (e.g. R²).  It is **NOT** a literal
    reproduction of the 2-source opinion-space formulas of Eq. 12.22-12.24
    — those have explicit Case I / Case II branches on dogmatism.

    For the literal canonical 2-source opinion-space form
    (Jøsang 2016 Def. 12.7, Eq. 12.22-12.24), use
    :func:`fusion_wbf_canonical_two` (same function implements the
    dogmatic Case II branch).  The two implementations are *algebraically
    consistent* on 2 sources with no external weights (both yield
    c_A, c_B-weighted averaging consistent with the bijection), but
    floating-point discrepancies in edge cases (near-dogmatic opinions)
    may differ at the 1e-9 level — see the unit tests in
    ``test_sl_formulas_v2.py`` for a numerical comparison.

    En mode trust_discount (Jøsang §14.3), les opinions sont discountées
    en amont (apply_trust_discount) et external_weights doit être None —
    WBF native pure.

    Alias
    -----
    :func:`fusion_evidence_average_confidence_weighted` is an alias that
    makes the evidence-averaging nature explicit at the call site.

    Paramètres :
        opinions         : liste de MultinomialOpinion
        external_weights : liste de poids externes (ex: R² scores), optionnel
        W                : constante de bijection (défaut : _WBF_W lu depuis config)

    Retourne : MultinomialOpinion fusionnée
    """
    if not opinions:
        return MultinomialOpinion([0, 0, 0], 1.0)

    N = len(opinions)
    if N == 1:
        return opinions[0]

    a_ref = opinions[0].a
    # W paramétrable pour éviter la variable gelée lors d'ablations dynamiques.
    # Si non fourni, fallback sur _WBF_W (importé au niveau module depuis config).
    W = W if W is not None else _WBF_W

    # Calcul des poids composites
    weights = np.zeros(N)
    for i, op in enumerate(opinions):
        c_i = op.confidence()  # c = 1 - u (Eq. 3.43)
        ext_w = external_weights[i] if external_weights is not None else 1.0

        # Sécurité : R² <= 0 → poids nul
        ext_w = max(ext_w, 0.0)

        weights[i] = ext_w * c_i

    total_weight = np.sum(weights)

    # Cas dégénéré : toutes les sources sont vacueuses ou de poids nul
    if total_weight < 1e-12:
        return MultinomialOpinion([0, 0, 0], 1.0, a_ref)

    # Normalisation des poids
    norm_weights = weights / total_weight

    # Eq. 12.27 : r_fused = Σ(r_i × w_i) / Σ(w_i)
    # En pratique, on fait la moyenne pondérée des vecteurs evidence
    r_fused = np.zeros(3)
    for i, op in enumerate(opinions):
        r_i = opinion_to_evidence(op, W)
        r_fused += r_i * norm_weights[i]

    # Moyenne pondérée du base rate (Eq. 12.23 cas II / Eq. 12.22)
    a_fused = np.zeros(3)
    for i, op in enumerate(opinions):
        a_fused += op.a * norm_weights[i]

    # Conversion retour en opinion via bijection
    return evidence_to_opinion(r_fused, W, a_fused)


# Alias sémantique (PATCH M-01 / F01) — clarifie la nature de l'opérateur au point
# d'appel.  La WBF implémentée ici est fidèle à Eq. 12.27 (moyenne pondérée par la
# confiance dans l'espace des évidences) ; le nom "evidence_average_confidence_
# weighted" rend cette nature explicite lorsque la traçabilité du papier vers le code
# est importante.  Pointe sur la même fonction (aucune duplication).
fusion_evidence_average_confidence_weighted = fusion_wbf_n_sources


FUSION_OPERATOR_MODES = (
    "wbf",
    "wbf_canonical",
    "cbf",
    "abf",
    "bcf",
    "ccf",
    "minbf",
    "maxbf",
    "hierarchical",
)


def _normalise_simplex(vec, fallback=None):
    """Return a non-negative vector normalised on the simplex."""
    arr = np.array(vec, dtype=float)
    arr = np.clip(arr, 0.0, None)
    s = float(arr.sum())
    if s > 1e-12:
        return arr / s
    if fallback is not None:
        fb = np.array(fallback, dtype=float)
        fb = np.clip(fb, 0.0, None)
        fb_s = float(fb.sum())
        if fb_s > 1e-12:
            return fb / fb_s
    return np.full_like(arr, 1.0 / len(arr), dtype=float)


def _mean_base_rate(opinions, weights=None):
    """Confidence/weight aware base-rate average, projected to the simplex."""
    opinions = list(opinions)
    if not opinions:
        return np.full(3, 1.0 / 3.0)

    if weights is None:
        weights_arr = np.ones(len(opinions), dtype=float)
    else:
        weights_arr = np.array(weights, dtype=float)
        if len(weights_arr) != len(opinions):
            raise ValueError("weights length must match opinions length")
        weights_arr = np.clip(weights_arr, 0.0, None)

    if float(weights_arr.sum()) < 1e-12:
        weights_arr = np.ones(len(opinions), dtype=float)

    a = np.zeros_like(opinions[0].a, dtype=float)
    for w_i, op in zip(weights_arr, opinions):
        a += float(w_i) * op.a
    return _normalise_simplex(a, fallback=opinions[0].a)


def _copy_opinion(op):
    """Return a defensive copy of an opinion."""
    return MultinomialOpinion(op.b.copy(), op.u, op.a.copy())


def fusion_evidence_average_n_sources(opinions, external_weights=None, W=None):
    """
    Equal/externally weighted evidence average without confidence reweighting.

    This is useful for the SL-ADS ``hierarchical`` ablation where Prophet and
    Reconstruction are treated as two already-aggregated method-level sources
    that should each contribute once. It deliberately differs from WBF:
    ``fusion_wbf_n_sources`` multiplies any external weight by confidence
    c_i = 1 - u_i, while this helper uses only the supplied method weights.
    """
    opinions = list(opinions)
    if not opinions:
        return MultinomialOpinion([0, 0, 0], 1.0)
    if len(opinions) == 1:
        return _copy_opinion(opinions[0])

    W = W if W is not None else _WBF_W
    if external_weights is None:
        weights = np.ones(len(opinions), dtype=float)
    else:
        weights = np.array(external_weights, dtype=float)
        if len(weights) != len(opinions):
            raise ValueError("external_weights length must match opinions length")
        weights = np.clip(weights, 0.0, None)
    if float(weights.sum()) < 1e-12:
        return MultinomialOpinion([0, 0, 0], 1.0, _mean_base_rate(opinions))

    norm_weights = weights / float(weights.sum())
    r_fused = np.zeros_like(opinions[0].b, dtype=float)
    for w_i, op in zip(norm_weights, opinions):
        r_fused += float(w_i) * opinion_to_evidence(op, W)
    a_fused = _mean_base_rate(opinions, weights=norm_weights)
    return evidence_to_opinion(r_fused, W, a_fused)


def fusion_abf(op_A, op_B):
    """
    Averaging Belief Fusion (ABF), 2-source form.

    ABF is the Subjective Logic operator intended for dependent evidence
    sources. For non-dogmatic opinions this implements the usual
    uncertainty-weighted average:

        b(x) = (b_A(x) u_B + b_B(x) u_A) / (u_A + u_B)
        u    = 2 u_A u_B / (u_A + u_B)
        a(x) = (a_A(x) + a_B(x)) / 2

    If one or more dogmatic opinions are present, the dogmatic opinions are
    averaged and the result remains dogmatic.
    """
    return fusion_abf_n_sources([op_A, op_B])


def fusion_abf_n_sources(opinions):
    """
    Averaging Belief Fusion (ABF), N-source form.

    For all non-dogmatic opinions:

        D = sum_i prod_{j != i} u_j
        b(x) = sum_i b_i(x) prod_{j != i} u_j / D
        u    = N prod_i u_i / D

    Dogmatic inputs dominate as the dogmatic average, because dependent
    duplicated dogmatic evidence must not be counted cumulatively.
    """
    opinions = list(opinions)
    if not opinions:
        return MultinomialOpinion([0, 0, 0], 1.0)
    if len(opinions) == 1:
        return _copy_opinion(opinions[0])

    dogmatic = [op for op in opinions if op.u < 1e-9]
    if dogmatic:
        b = np.mean([op.b for op in dogmatic], axis=0)
        a = _mean_base_rate(dogmatic)
        return MultinomialOpinion(b, 0.0, a)

    u = np.array([op.u for op in opinions], dtype=float)
    prod_u = float(np.prod(u))
    weights = prod_u / np.clip(u, 1e-12, None)
    D = float(weights.sum())
    if D < 1e-12:
        return MultinomialOpinion([0, 0, 0], 1.0, _mean_base_rate(opinions))

    b = np.zeros_like(opinions[0].b, dtype=float)
    for w_i, op in zip(weights, opinions):
        b += float(w_i) * op.b
    b = b / D
    u_fused = float(len(opinions) * prod_u / D)
    a = _mean_base_rate(opinions)
    return MultinomialOpinion(b, u_fused, a)


def fusion_cbf_n_sources(opinions):
    """Cumulative Belief Fusion (CBF) folded over N sources."""
    opinions = list(opinions)
    if not opinions:
        return MultinomialOpinion([0, 0, 0], 1.0)
    out = _copy_opinion(opinions[0])
    for op in opinions[1:]:
        out = fusion_cbf(out, op)
    return out


def fusion_bcf(op_A, op_B):
    """
    Belief Constraint Fusion (BCF), i.e. Dempster-style constraint fusion.

    This singleton multinomial implementation uses the off-diagonal conflict
    mass K = sum_{i != j} b_A[i] b_B[j]. The operation is undefined at total
    conflict (K == 1); in that numerical corner case we return a vacuous
    opinion with the confidence-weighted base rate so ablation runs remain
    inspectable instead of crashing mid-sweep.
    """
    bA, uA = op_A.b, float(op_A.u)
    bB, uB = op_B.b, float(op_B.u)
    K = float(np.sum(bA) * np.sum(bB) - np.dot(bA, bB))
    denom = 1.0 - K
    a = _mean_base_rate([op_A, op_B], weights=[op_A.confidence(), op_B.confidence()])

    if denom <= 1e-12:
        warnings.warn(
            "fusion_bcf: total conflict (K ~= 1); returning vacuous "
            "fallback because Dempster/BCF is undefined.",
            RuntimeWarning,
            stacklevel=2,
        )
        return MultinomialOpinion([0, 0, 0], 1.0, a)

    b = (bA * bB + bA * uB + uA * bB) / denom
    u = (uA * uB) / denom
    return MultinomialOpinion(b, u, a)


def fusion_bcf_n_sources(opinions):
    """BCF folded over N sources; intended for ablation, not production."""
    opinions = list(opinions)
    if not opinions:
        return MultinomialOpinion([0, 0, 0], 1.0)
    out = _copy_opinion(opinions[0])
    for op in opinions[1:]:
        out = fusion_bcf(out, op)
    return out


def fusion_ccf(op_A, op_B):
    """
    Consensus & Compromise Fusion (CCF), singleton-frame projection.

    Full CCF is naturally expressed for richer belief masses. SL-ADS stores
    only singleton beliefs plus uncertainty, so this function implements the
    consensus/compromise idea in that reduced frame:

    * consensus = min(b_A, b_B), kept on the same state;
    * residual belief against uncertainty is kept on its original state;
    * residual conflict between different states is redistributed between
      the two states according to the fused base rates;
    * the result is normalised back to a valid multinomial opinion.

    The function is idempotent by construction and is suitable for controlled
    ablation. Treat it as a projected CCF approximation, not as a replacement
    for a full hyper-opinion CCF implementation.
    """
    if (np.allclose(op_A.b, op_B.b, atol=1e-12)
            and np.isclose(op_A.u, op_B.u, atol=1e-12)
            and np.allclose(op_A.a, op_B.a, atol=1e-12)):
        return _copy_opinion(op_A)

    a = _mean_base_rate([op_A, op_B])
    b_cons = np.minimum(op_A.b, op_B.b)
    rA = np.clip(op_A.b - b_cons, 0.0, None)
    rB = np.clip(op_B.b - b_cons, 0.0, None)

    b = b_cons.copy()
    b += rA * op_B.u + rB * op_A.u

    n = len(b)
    for i in range(n):
        for j in range(n):
            mass = float(rA[i] * rB[j])
            if mass <= 0.0:
                continue
            if i == j:
                b[i] += mass
                continue
            denom = float(a[i] + a[j])
            if denom > 1e-12:
                b[i] += mass * float(a[i]) / denom
                b[j] += mass * float(a[j]) / denom
            else:
                b[i] += 0.5 * mass
                b[j] += 0.5 * mass

    u = float(op_A.u * op_B.u)
    total = float(np.sum(b) + u)
    if total <= 1e-12:
        return MultinomialOpinion([0, 0, 0], 1.0, a)
    return MultinomialOpinion(b / total, u / total, a)


def fusion_ccf_n_sources(opinions):
    """Projected CCF folded over N sources; exact CCF ablation is 2-source."""
    opinions = list(opinions)
    if not opinions:
        return MultinomialOpinion([0, 0, 0], 1.0)
    out = _copy_opinion(opinions[0])
    for op in opinions[1:]:
        out = fusion_ccf(out, op)
    return out


def fusion_minbf(op_A, op_B):
    """Minimum Belief Fusion heuristic: keep the per-class minimum beliefs."""
    b = np.minimum(op_A.b, op_B.b)
    u = max(0.0, 1.0 - float(np.sum(b)))
    return MultinomialOpinion(b, u, _mean_base_rate([op_A, op_B]))


def fusion_minbf_n_sources(opinions):
    opinions = list(opinions)
    if not opinions:
        return MultinomialOpinion([0, 0, 0], 1.0)
    b = np.minimum.reduce([op.b for op in opinions])
    u = max(0.0, 1.0 - float(np.sum(b)))
    return MultinomialOpinion(b, u, _mean_base_rate(opinions))


def fusion_maxbf(op_A, op_B):
    """Maximum Belief Fusion heuristic: keep the per-class maximum beliefs."""
    b = np.maximum(op_A.b, op_B.b)
    s = float(np.sum(b))
    if s > 1.0:
        b = b / s
        u = 0.0
    else:
        u = 1.0 - s
    return MultinomialOpinion(b, u, _mean_base_rate([op_A, op_B]))


def fusion_maxbf_n_sources(opinions):
    opinions = list(opinions)
    if not opinions:
        return MultinomialOpinion([0, 0, 0], 1.0)
    b = np.maximum.reduce([op.b for op in opinions])
    s = float(np.sum(b))
    if s > 1.0:
        b = b / s
        u = 0.0
    else:
        u = 1.0 - s
    return MultinomialOpinion(b, u, _mean_base_rate(opinions))


def fusion_by_mode(opinions, mode="wbf", external_weights=None, W=None):
    """
    Dispatch helper used by ablation code.

    ``hierarchical`` is implemented as an equal evidence average without
    confidence reweighting. For ``wbf_canonical`` exactly two sources are required; it exists mainly for
    formula-level validation against Joesang Eq. 12.22-12.24.
    """
    opinions = list(opinions)
    mode = str(mode).strip().lower()
    W = W if W is not None else _WBF_W

    if mode == "wbf":
        return fusion_wbf_n_sources(opinions, external_weights=external_weights, W=W)
    if mode == "hierarchical":
        return fusion_evidence_average_n_sources(opinions, external_weights=external_weights, W=W)
    if mode == "wbf_canonical":
        if len(opinions) != 2:
            raise ValueError("wbf_canonical requires exactly two opinions")
        return fusion_wbf_canonical_two(opinions[0], opinions[1])
    if mode == "cbf":
        return fusion_cbf_n_sources(opinions)
    if mode == "abf":
        return fusion_abf_n_sources(opinions)
    if mode == "bcf":
        return fusion_bcf_n_sources(opinions)
    if mode == "ccf":
        return fusion_ccf_n_sources(opinions)
    if mode == "minbf":
        return fusion_minbf_n_sources(opinions)
    if mode == "maxbf":
        return fusion_maxbf_n_sources(opinions)
    raise ValueError(f"Unknown fusion mode {mode!r}; expected one of {FUSION_OPERATOR_MODES}")


# ==============================================================================
# 4-bis. WEIGHTED BELIEF FUSION — 2 SOURCES CANONIQUE (Eq. 12.22-12.24)
# PATCH M-01 / F01 (2026-04-24)
# ==============================================================================

def fusion_wbf_canonical_two(op_A, op_B):
    """
    Weighted Belief Fusion 2 sources — forme canonique opinion-space.
    Source : Jøsang 2016, Definition 12.7, Eq. (12.22)-(12.24).

    PATCH M-01 / F01 — Why this function exists
    -------------------------------------------
    The primary WBF operator in this module, :func:`fusion_wbf_n_sources`,
    is an *evidence-space* implementation (faithful to Eq. 12.27 via the
    bijection Def. 3.9) that also supports arbitrary N and external
    quality weights.  The consolidated audit (item M-01 / F01) required a
    literal opinion-space reproduction of the 2-source form of
    Eq. 12.22-12.24, with the two explicit branches on dogmatism, so that
    the paper's equations map verbatim onto one code function.  This
    function is that faithful reproduction.

    Formulas implemented
    --------------------
    Let c_A = 1 - u_A, c_B = 1 - u_B be the confidences (Eq. 3.43).

    **Case I (general — at least one source non-dogmatic, u_A > 0 ∨ u_B > 0)**
    Let D = c_A · u_B + c_B · u_A  (Eq. 12.22 denominator)::

        b^⋄(x) = [ c_A · u_B · b_A(x) + c_B · u_A · b_B(x) ] / D      (Eq. 12.22)
        u^⋄    = u_A · u_B · (c_A + c_B) / D                          (Eq. 12.23)
        a^⋄(x) = [ c_A · a_A(x) + c_B · a_B(x) ] / (c_A + c_B)        (Eq. 12.24)

    **Case II (both dogmatic — u_A = u_B = 0)**
    The Case I denominator vanishes.  Jøsang Eq. 12.24 defines the
    dogmatic limit as the symmetric weighted average of the dogmatic
    beliefs using γ_A = γ_B = 1/2 (the confidences both tend to 1
    simultaneously)::

        b^⋄(x) = 1/2 · b_A(x) + 1/2 · b_B(x)
        u^⋄    = 0
        a^⋄(x) = 1/2 · a_A(x) + 1/2 · a_B(x)

    Both branches satisfy the opinion bijection constraint Σb + u = 1 by
    construction (algebraic verification in the unit tests).

    Paramètres :
        op_A, op_B : MultinomialOpinion — les deux opinions à fusionner

    Retourne : MultinomialOpinion fusionnée

    Propriétés vérifiées par les tests unitaires :
        - Bijection : Σb^⋄ + u^⋄ = 1  pour toutes entrées aléatoires
        - Symétrie  : f(A, B) = f(B, A)
        - Cas idempotent : f(A, A) = A  (toute source avec elle-même)
        - Limite dogmatique continue : lim_{u→0} Case I = Case II

    See Also
    --------
    fusion_wbf_n_sources : version N-sources, espace des évidences,
        supporte les poids externes.  Algébriquement cohérente avec cette
        fonction dans la limite N=2, poids externes uniformes, mais via
        le moyennage des évidences plutôt que la forme opinion-space
        canonique.
    fusion_cbf : Cumulative Belief Fusion (Jøsang §12.3).
    """
    # Lire les champs une fois pour éviter les accès répétés.
    bA, uA, aA = op_A.b, float(op_A.u), op_A.a
    bB, uB, aB = op_B.b, float(op_B.u), op_B.a
    cA = 1.0 - uA
    cB = 1.0 - uB

    # --- Case II : les deux sources sont dogmatiques (u_A = u_B = 0) --------
    # Tolérance numérique 1e-9 alignée sur fusion_cbf (cohérence inter-opérateur).
    if uA < 1e-9 and uB < 1e-9:
        # γ_A = γ_B = 1/2 (limite symétrique de la confiance).  Pour des u
        # légèrement différents (e.g. 1e-10 vs 1e-12), on tombe tout de même
        # ici car la formule Case I est numériquement instable ; on reste sur
        # le cas dogmatique pur.
        b_fused = 0.5 * bA + 0.5 * bB
        a_fused = 0.5 * aA + 0.5 * aB
        return MultinomialOpinion(b_fused, 0.0, a_fused)

    # --- Case I : cas général (Eq. 12.22-12.24) ----------------------------
    D = cA * uB + cB * uA
    if D < 1e-12:
        # Garde-fou : si D dégénère (ne devrait pas arriver hors Case II).
        # On retombe sur la moyenne arithmétique comme fallback neutre.
        warnings.warn(
            "fusion_wbf_canonical_two: denominator D ~ 0 in Case I "
            "(u_A, u_B unexpectedly both near 0 but not caught by Case II); "
            "falling back to symmetric mean.",
            RuntimeWarning, stacklevel=2,
        )
        b_fused = 0.5 * bA + 0.5 * bB
        a_fused = 0.5 * aA + 0.5 * aB
        return MultinomialOpinion(b_fused, 0.5 * (uA + uB), a_fused)

    # Eq. 12.22 — croyance fusionnée
    b_fused = (cA * uB * bA + cB * uA * bB) / D

    # Eq. 12.23 — incertitude fusionnée.  Identité algébrique vérifiée :
    #   Σb_fused + u_fused = [c_A² u_B + c_B² u_A + u_A u_B (c_A + c_B)] / D
    #                      = [c_A u_B (c_A + u_A) + c_B u_A (c_B + u_B)] / D
    #                      = [c_A u_B + c_B u_A] / D  = D/D = 1  ✓
    u_fused = (cA + cB) * uA * uB / D

    # Eq. 12.24 — base rate fusionné (moyenne confidence-weighted)
    total_c = cA + cB
    if total_c < 1e-12:
        # Limite inattendue (c_A + c_B ≈ 0 ⇒ u_A ≈ u_B ≈ 1 ⇒ sources
        # vacueuses) : on retombe sur la moyenne arithmétique du base rate.
        a_fused = 0.5 * aA + 0.5 * aB
    else:
        a_fused = (cA * aA + cB * aB) / total_c

    # Projection dans le simplex par sécurité numérique.
    a_fused = np.clip(a_fused, 0.0, 1.0)
    s = a_fused.sum()
    if s > 1e-12:
        a_fused = a_fused / s
    else:
        a_fused = 0.5 * aA + 0.5 * aB

    return MultinomialOpinion(b_fused, u_fused, a_fused)


# ==============================================================================
# 5. CUMULATIVE BELIEF FUSION — 2 SOURCES (Section 12.3, Eq. 12.14)
# ==============================================================================

def fusion_cbf(op_A, op_B):
    """
    Cumulative Belief Fusion (CBF) pour 2 sources indépendantes.
    Source : Jøsang, Definition 12.5, Eq. (12.14) et (12.15).

    Équivalent à l'addition des vecteurs de preuves (Theorem 12.2, Eq. 12.17).

    Utilisé pour la fusion inter-méthode (Prophet ⊕ Reconstruction).
    """

    # --- Case II : Les deux sources sont dogmatiques (Eq. 12.15) ---
    if op_A.u < 1e-9 and op_B.u < 1e-9:
        # Poids γ_i proportionnels à la confiance c_i = 1 - u_i (Eq. 12.15).
        # Pour u_A ≈ u_B ≈ 0, c_A = c_B = 1 → γ = 0.5 (cas symétrique standard).
        # Pour des u légèrement différents (ex: 1e-10 vs 1e-12), le poids correct
        # est c_i / (c_A + c_B) plutôt que 0.5 fixe.
        c_A = 1.0 - op_A.u
        c_B = 1.0 - op_B.u
        total_c = c_A + c_B
        gamma_A = c_A / total_c if total_c > 1e-12 else 0.5
        new_b = gamma_A * op_A.b + (1.0 - gamma_A) * op_B.b
        new_a = gamma_A * op_A.a + (1.0 - gamma_A) * op_B.a
        return MultinomialOpinion(new_b, 0.0, new_a)

    # --- Case I : Cas général (Eq. 12.14) ---
    denom = op_A.u + op_B.u - (op_A.u * op_B.u)

    if denom < 1e-12:
        # PATCH TASK-27 (audit_tmp MAJ-06, 2026-04-26)
        # ──────────────────────────────────────────────────────────────────
        # L'ancien fallback ``return op_A`` était asymétrique : il privilégiait
        # silencieusement la source A sur la source B sans justification.
        # En pratique cette branche n'est atteinte que si ``u_A + u_B``
        # est négligeable mais qu'AU MOINS un ``u`` est >= 1e-9 (sinon le
        # Case II dogmatique ci-dessus l'aurait capturé) — un cas-limite
        # quasi-impossible numériquement, mais on garantit malgré tout
        # un comportement symétrique : moyenne arithmétique de op_A et op_B
        # (équivalent à γ_A = 0.5 dans le Case II), avec u réduit à 0.
        c_A = 1.0 - op_A.u
        c_B = 1.0 - op_B.u
        total_c = c_A + c_B
        gamma_A = c_A / total_c if total_c > 1e-12 else 0.5
        b_fb = gamma_A * op_A.b + (1.0 - gamma_A) * op_B.b
        a_fb = gamma_A * op_A.a + (1.0 - gamma_A) * op_B.a
        # Petit warning explicite — ce cas-limite est suspect en production.
        import warnings as _warnings
        _warnings.warn(
            "fusion_cbf: degenerate denom (<1e-12) with non-dogmatic "
            "u_A,u_B — falling back to weighted average (gamma_A="
            f"{gamma_A:.3f}) instead of returning op_A asymmetrically.",
            RuntimeWarning, stacklevel=2,
        )
        return MultinomialOpinion(b_fb, 0.0, a_fb)

    # Croyance fusionnée
    b_fused = (op_A.b * op_B.u + op_B.b * op_A.u) / denom

    # Incertitude fusionnée
    u_fused = (op_A.u * op_B.u) / denom

    # Base rate fusionné (Eq. 12.14, troisième ligne)
    # Base rate fusionné (Eq. 12.14 / N=2)
    # Forme N=2 de la ligne a(x):
    #   a = [a_A*u_B + a_B*u_A - (N-1)*a_avg*u_A*u_B] / [u_A + u_B - u_A*u_B]
    # avec N=2, a_avg=(a_A+a_B)/2.
    a_avg = (op_A.a + op_B.a) / 2.0
    num_a = op_A.a * op_B.u + op_B.a * op_A.u - a_avg * op_A.u * op_B.u
    denom_a = denom

    if abs(denom_a) < 1e-12:
        a_fused = a_avg
    else:
        a_fused = num_a / denom_a

    # Robustesse numérique: projetter dans le simplex.
    a_fused = np.clip(a_fused, 0.0, 1.0)
    s = a_fused.sum()
    if s < 1e-12:
        a_fused = a_avg
    else:
        a_fused = a_fused / s

    return MultinomialOpinion(b_fused, u_fused, a_fused)


# ==============================================================================
# 6. EVIDENCE BOOSTING (Rééquilibrage inter-méthode)
# ==============================================================================

def boost_opinion_evidence(op, ratio, W=3.0):
    """
    Multiplie les preuves sous-jacentes d'une opinion par un ratio,
    puis reconvertit en opinion via bijection (Def. 3.9).

    Justification (Theorem 12.2, Eq. 12.17) :
        La CBF est équivalente à l'addition de preuves. Si une méthode A
        a N_A métriques et une méthode B a N_B métriques, A accumule
        ~(N_A/N_B)× plus de preuves. On compense en multipliant les preuves
        de B par ratio = N_A / N_B avant la CBF.

    Paramètres :
        op    : MultinomialOpinion à booster
        ratio : multiplicateur de preuves (ex: 4.0 si 8 métriques vs 2)
        W     : constante de bijection

    Retourne : MultinomialOpinion avec preuves amplifiées
    """
    if ratio <= 0 or np.isclose(ratio, 1.0):
        return op

    # Opinion → Evidence (Def. 3.9 inverse)
    r = opinion_to_evidence(op, W)

    # Boost
    r_boosted = r * ratio

    # Evidence → Opinion (Def. 3.9)
    return evidence_to_opinion(r_boosted, W, op.a)
# ==============================================================================

def evidence_to_opinion_custom(P, S, N, k=2.0, a=None):
    """Wrapper legacy → evidence_to_opinion"""
    return evidence_to_opinion(np.array([P, S, N]), W=k, a=a)


def fusion_wbf_2sources(op_A, op_B):
    """
    WBF pour 2 sources (legacy wrapper).
    Eq. 12.22 de Jøsang.
    """
    return fusion_wbf_n_sources([op_A, op_B])


def fusion_cumulative_josang(op_A, op_B):
    """Wrapper legacy → fusion_cbf"""
    return fusion_cbf(op_A, op_B)


def fusion_weighted_average(opinions_list, weights_list):
    """
    Wrapper legacy → fusion_wbf_n_sources avec poids externes.
    NOTE : Utilisé uniquement en mode r2_static (legacy).
    En mode trust_discount, appeler fusion_wbf_n_sources(opinions, external_weights=None)
    après apply_trust_discount() sur chaque opinion en amont.
    """
    if not opinions_list:
        return MultinomialOpinion([0, 0, 0], 1.0)
    return fusion_wbf_n_sources(opinions_list, external_weights=weights_list)


def fusion_CBF(op_A, op_B):
    """Wrapper legacy → fusion_cbf"""
    return fusion_cbf(op_A, op_B)


def fusion_ccf_josang(op_A, op_B):
    """
    Wrapper legacy -> projected singleton CCF.
    """
    return fusion_ccf(op_A, op_B)

def apply_trust_discount(op: MultinomialOpinion, t: float) -> MultinomialOpinion:
    """
    Probability-sensitive trust discounting (Jøsang 2016, Def. 14.6, Eq. 14.6).

    Paramètres :
        op : MultinomialOpinion source
        t  : projected trust probability ∈ [0,1]
               t=1 → opinion inchangée (confiance totale)
               t=0 → opinion vacueuse (source ignorée)

    Retourne : MultinomialOpinion discountée
    """
    t = float(np.clip(t, 0.0, 1.0))
    if np.isclose(t, 1.0):
        return op  # Pas de discounting

    b_disc = op.b * t
    # u_disc = 1 - t × Σb(x) = 1 - t × (1 - u)
    u_disc = 1.0 - t * (1.0 - op.u)
    # Vérification : Σb_disc + u_disc = t(1-u) + 1 - t(1-u) = 1 ✓

    return MultinomialOpinion(b_disc, u_disc, op.a.copy())


def apply_contextual_discount(op: MultinomialOpinion,
                               alpha: list) -> MultinomialOpinion:
    """
    Discounting contextuel par hypothèse (Mercier, Quost & Denoeux 2006/2008,
    "Contextual Discounting of Belief Functions", ECSQARU / Information Fusion).

    Principe : chaque hypothèse x reçoit un coefficient de fiabilité α(x) ∈ [0,1]
    distinct, contrairement au discounting classique de Jøsang (scalar t uniforme).

    Formule (ternaire : safe, suspect, attack) :
        b_disc(x) = α(x) × b(x)          pour x ∈ {safe, suspect, attack}
        u_disc    = 1 − Σ_x α(x) × b(x)  (incertitude résiduelle)

    Sémantique dans notre contexte :
        α[0] = alpha_safe    : fiabilité de la source pour l'hypothèse "safe"
        α[1] = alpha_suspect : fiabilité pour l'hypothèse "suspect"
        α[2] = alpha_attack  : fiabilité pour l'hypothèse "attack"

    Usage typique pour la Reconstruction (RANSAC) :
        α = [1.0, 1.0, RECONST_ATTACK_RELIABILITY]
        → on fait totalement confiance à son évidence "safe" (structure normale = safe réel)
        → on ne fait que partiellement confiance à son "silence sur attack" car elle est
          aveugle aux attaques applicatives (SLOWLORIS, anomalies de comportement).

    Paramètres :
        op    : MultinomialOpinion à discounter
        alpha : liste [α_safe, α_suspect, α_attack], chaque valeur ∈ [0, 1]

    Retourne : MultinomialOpinion discountée (u plus élevé, b_attack réduit)

    Propriétés :
        - α = [1, 1, 1] → opinion inchangée (= trust_discount avec t=1)
        - α = [1, 1, 0] → b_attack mis à 0, toute l'incertitude restante vers u
        - α = [0, 0, 0] → opinion vacueuse (u=1), source complètement ignorée
        - Σb_disc + u_disc = 1 ✓ (préservé par construction)
    """
    alpha = np.clip(np.array(alpha, dtype=float), 0.0, 1.0)
    b_disc = alpha * op.b
    u_disc = float(np.clip(1.0 - np.sum(b_disc), 0.0, 1.0))
    return MultinomialOpinion(b_disc, u_disc, op.a.copy())
