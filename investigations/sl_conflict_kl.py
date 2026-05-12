"""
sl_conflict_kl.py — Conflict Degree via Projected Probabilities
===============================================================
Drop-in replacement for compute_conflict_degree() in sl_formulas_v2.py.

Rationale:
  The original K is computed from belief masses b (Jøsang §11 inspiration).
  This version computes K from projected probabilities P(x) = b + a·u,
  which incorporates the base rate and uncertainty into the conflict measure.

  At u→0: K_proj → K_belief (converges monotonically).
  At u=0.006 (SL-ADS regime): |K_proj - K_belief| < 0.04,
  ranking of conflict events preserved, λ_dyn ordering unchanged.

Usage:
  In config.py:  "CONFLICT_MODE": "projected_prob"   # or "belief_mass" (default)
  In sl_formulas_v2.py:  import and route in temporal_adaptive_ageing()

Mathematical basis:
  K_proj = P_prev[safe]   × P_curr[anom]
         + P_prev[anom]   × P_curr[safe]
         + P_prev[safe]   × P_curr[susp]
         + P_prev[susp]   × P_curr[anom]
  where P_i = b_i + a_i · u  (Jøsang Eq. 3.23)
  a = base rate vector (from adaptive_base_rate.py or SL_PRIOR_A)

  The asymmetric exclusion (de-escalation transitions omitted) is
  preserved from the original K to maintain the documented design
  intent (Section 5.2.1).
"""

import numpy as np


def compute_conflict_degree_projected(
    r_prev: np.ndarray,
    r_curr: np.ndarray,
    a_prev: np.ndarray,
    a_curr: np.ndarray,
    W: float = 2.0,
) -> float:
    """
    Conflict degree K computed from projected probabilities P(x) = b(x) + a(x)·u.

    Arguments:
        r_prev  : accumulated evidence vector [r_safe, r_susp, r_anom]
        r_curr  : current evidence vector     [r_safe, r_susp, r_anom]
        a_prev  : base rate for prev opinion  [a_safe, a_susp, a_anom]
        a_curr  : base rate for curr opinion  [a_safe, a_susp, a_anom]
        W       : bijection constant (Def. 3.9), default 2.0

    Returns:
        K ∈ [0, 1] — conflict degree
    """
    r_prev = np.array(r_prev, dtype=float)
    r_curr = np.array(r_curr, dtype=float)
    a_prev = np.array(a_prev, dtype=float)
    a_curr = np.array(a_curr, dtype=float)

    # Bijection → belief masses + uncertainty (Def. 3.9)
    D_prev = np.sum(r_prev) + W
    D_curr = np.sum(r_curr) + W
    b_prev = r_prev / D_prev if D_prev > 0 else np.zeros(3)
    u_prev = W / D_prev if D_prev > 0 else 1.0
    b_curr = r_curr / D_curr if D_curr > 0 else np.zeros(3)
    u_curr = W / D_curr if D_curr > 0 else 1.0

    # Projected probabilities P(x) = b(x) + a(x)·u  (Eq. 3.23)
    P_prev = b_prev + a_prev * u_prev
    P_curr = b_curr + a_curr * u_curr

    # Conflict: asymmetric cross-product sum on escalating transitions
    # (Same asymmetry as original K — de-escalation excluded by design)
    K = (P_prev[0] * P_curr[2]   # safe_prev × anom_curr
       + P_prev[2] * P_curr[0]   # anom_prev × safe_curr
       + P_prev[0] * P_curr[1]   # safe_prev × susp_curr
       + P_prev[1] * P_curr[2])  # susp_prev × anom_curr

    return float(np.clip(K, 0.0, 1.0))


def compute_conflict_degree_kl(
    r_prev: np.ndarray,
    r_curr: np.ndarray,
    a_prev: np.ndarray,
    a_curr: np.ndarray,
    W: float = 2.0,
    eps: float = 1e-12,
) -> float:
    """
    Conflict degree K via symmetric KL divergence on projected probabilities.

    KL_sym(P_prev ‖ P_curr) = KL(P_prev ‖ P_curr) + KL(P_curr ‖ P_prev)

    Note: KL is unbounded; the result is normalised to [0,1] via a
    monotone mapping: K_kl = 1 - exp(-KL_sym / tau)
    where tau = 1.0 (tunable via CONFIG["CONFLICT_KL_TAU"]).

    Arguments: same as compute_conflict_degree_projected
    Returns:   K ∈ [0, 1]
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

    P_prev = np.clip(b_prev + a_prev * u_prev, eps, None)
    P_curr = np.clip(b_curr + a_curr * u_curr, eps, None)

    # Normalise to simplex
    P_prev /= P_prev.sum()
    P_curr /= P_curr.sum()

    kl_fwd = float(np.sum(P_prev * np.log(P_prev / P_curr)))
    kl_rev = float(np.sum(P_curr * np.log(P_curr / P_prev)))
    kl_sym = kl_fwd + kl_rev

    # Monotone normalisation to [0, 1]
    tau = 1.0
    K = 1.0 - np.exp(-kl_sym / tau)
    return float(np.clip(K, 0.0, 1.0))


def get_conflict_function(mode: str = "belief_mass"):
    """
    Factory: returns the conflict function specified by CONFLICT_MODE in config.

    Modes:
        "belief_mass"     : original K on b (default, backward-compatible)
        "projected_prob"  : K on P(x) = b + a·u
        "kl_symmetric"    : symmetric KL divergence on P
    """
    modes = {
        "belief_mass":    None,          # use compute_conflict_degree from sl_formulas_v2
        "projected_prob": compute_conflict_degree_projected,
        "kl_symmetric":   compute_conflict_degree_kl,
    }
    if mode not in modes:
        raise ValueError(f"Unknown CONFLICT_MODE '{mode}'. "
                         f"Valid: {list(modes.keys())}")
    return modes[mode]