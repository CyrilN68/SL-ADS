"""
train_v10.py — SL-ADS Training Pipeline
========================================
Subjective Logic Anomaly Detection System (SL-ADS)
Applied to the RedeRio academic backbone (UFRJ, Ilha do Fundão).

Pipeline overview
-----------------
For each of the 17 network metrics (12 Prophet + 5 QR reconstruction):
  1. Fit a forecasting or reconstruction model on clean training data.
  2. Calibrate directional anomaly thresholds (T_susp, T_atk) via EVT/POT
     (Peaks-Over-Threshold, Grimshaw 1993) with automatic GPD validity check.
  3. Compute the Empirical Dirichlet Prior (EDP) from training residuals.

After model fitting:
  4. Auto-calibrate RECONST_ATTACK_RELIABILITY (if mode='auto').
  5. Auto-calibrate DECISION_THRESHOLD as quantile(proj_atk, 1-FPR_target)
     on held-out data (Ruff et al. 2021).
  6. Save all artifacts to a single .pkl file + lightweight sidecar JSON.

Key design choices (all justified below at point of use)
---------------------------------------------------------
  Prophet       : seasonal forecasting (Taylor & Letham 2018)
  QR(q=0.5)     : LAD regression for structural reconstruction (Koenker &
                  Bassett 1978) — deterministic, 50% breakdown point
  EVT/POT       : Grimshaw (1993) MLE on residual tails, with σ̃ validity
                  check (Coles 2001 §4.2) and empirical quantile fallback
  EDP           : Ferguson (1973) Dirichlet moment estimator
  proj_atk      : b_atk + a_atk·u (Jøsang 2016, Eq. 3.23) as decision var

References
----------
  Grimshaw (1993)        Technometrics 35(2) — GPD MLE
  Siffer et al. (2017)   KDD — SPOT algorithm, GPD quantile formula
  Coles (2001)           Intro. to Stat. Modelling of Extreme Values §4.2–4.3
  Davison & Smith (1990) JRSS-B — POT / declustering
  Taylor & Letham (2018) Am. Statistician — Prophet
  Koenker & Bassett (1978) Econometrica — quantile regression / LAD
  Rousseeuw & Leroy (1987) Robust Regression and Outlier Detection
  Ferguson (1973)         Ann. Statist. — Dirichlet process
  Jøsang (2016)          Subjective Logic, Springer §3, §12, §14
  Ruff et al. (2021)     TPAMI — anomaly detection review
  Bridgman (1922)        Dimensional Analysis — physical homogeneity
  Hyndman & Athanasopoulos (2021) Forecasting §3.1 — mean benchmark
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import math
import os
import sys
import json
import logging

import numpy as np
import pandas as pd
import joblib

from prophet import Prophet
from scipy import stats
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import genpareto
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import r2_score
from sl_ads.preprocessing_utils import preprocess_metrics  # Phase H
from sl_ads.stats.mase import compute_mase, mase_to_trust  # PATCH D5 (MASE trust mode)
from sl_ads.calendar.regime import (  # PATCH H2 (calendar-aware EVT)
    REGIME_BUCKETS,
    REGIME_FN_SIGNATURE,
    regime_of_series,
)

# Priorité haute pour éviter la préemption sur E-cores (Intel hybrid)
try:
    import psutil as _psutil
    _psutil.Process(os.getpid()).nice(_psutil.HIGH_PRIORITY_CLASS)
except Exception:
    pass  # Non-bloquant : Windows uniquement, peut nécessiter des droits admin

from sl_ads.paths import get_model_path, get_threshold_sidecar_path  # Phase H
from sl_ads.train.model_guardrails import validate_required_model_coverage

try:
    from sl_ads.config import CONFIG  # Phase H
except ImportError:
    print("❌ CRITIQUE : Impossible de charger sl_ads.config.")
    sys.exit(1)

from sl_ads.core import fusion_policy

# Silence les logs verbeux des librairies externes
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

# Taille de fenêtre (nombre de tranches de 30 s agrégées en une opinion)
# Centralisée dans config.py pour que toute modification soit propagée automatiquement.
WINDOW_SIZE = CONFIG.get('WINDOW_SIZE', 10)


def _ensure_prophet_tbb_on_path() -> None:
    """Expose Prophet's bundled TBB runtime before CmdStanPy probes PATH.

    On Windows, cmdstanpy checks ``where.exe tbb.dll`` before it appends the
    bundled TBB directory.  In UTF-8 mode the localized "not found" message can
    raise ``UnicodeDecodeError`` before cmdstanpy reaches its fallback.  Adding
    the known Prophet-bundled TBB directory up front makes the probe succeed and
    keeps Prophet initialisation deterministic.
    """
    if os.name != "nt":
        return
    try:
        import importlib_resources

        backend = str(CONFIG.get("PROPHET_STAN_BACKEND", "CMDSTANPY")).upper()
        if backend != "CMDSTANPY":
            return
        from prophet.models import CmdStanPyBackend

        cmdstan_root = (
            importlib_resources.files("prophet")
            / "stan_model"
            / f"cmdstan-{CmdStanPyBackend.CMDSTAN_VERSION}"
        )
        tbb_dir = cmdstan_root / "stan" / "lib" / "stan_math" / "lib" / "tbb"
        if not tbb_dir.exists():
            return
        tbb_str = str(tbb_dir)
        os.environ["STAN_TBB"] = tbb_str
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if tbb_str not in path_parts:
            os.environ["PATH"] = os.pathsep.join([tbb_str, *path_parts])
    except Exception as exc:
        print(f"[WARN] Could not preconfigure Prophet TBB path: {exc!r}")


# ==============================================================================
# PATCH m-08 / F28 — FALLBACK AUDIT LOG
# ==============================================================================
# Accumulator for all fallback events that occur during training.  Each entry
# is a dict describing the fallback kind, the metric it was applied to, and
# the reason. At the end of train_models() the whole log is serialized next
# to the model PKL (trained_models_{version}_fallbacks.json) and embedded
# in the threshold sidecar under the "fallbacks" key.
#
# Tracked kinds:
#   * evt_scipy_fallback    — Grimshaw MLE failed, scipy.genpareto.fit used
#   * evt_empirical_final   — all GPD fits failed → empirical quantile
#   * evt_short_data        — fewer peaks than min_peaks → empirical quantile
#   * evt_sigma_mod         — σ̃ = σ − ξ·t₀ ≤ 0 (Coles 2001 §4.2) → empirical
#   * reconstruction_dummy  — R²<0 and allow_mean_fallback=True → DummyRegressor
#
# The same metric may appear multiple times (e.g. once for q_susp and once
# for q_atk in _evt_threshold).  That's intentional — it keeps the audit
# trail faithful to what happened at runtime.
# ==============================================================================
_FALLBACK_LOG: list = []


def _record_fallback(kind: str,
                     metric_key: str = None,
                     branch: str = None,
                     **details) -> None:
    """Append a fallback event to ``_FALLBACK_LOG``.

    Parameters
    ----------
    kind : str
        One of the tracked kinds documented above.
    metric_key : str | None
        Short metric identifier (e.g. ``"bytes"``, ``"reconst_udp_from_flows"``).
        ``None`` when the caller does not know the metric (e.g. helper called
        from a non-metric context).
    branch : str | None
        ``"prophet"`` or ``"ransac"`` (reconstruction) — which calibration
        branch the fallback occurred in.
    **details : dict
        Arbitrary key/value pairs captured for forensics (ξ, σ̃, R², …).
    """
    entry = {"kind": kind, "metric": metric_key, "branch": branch}
    entry.update(details)
    _FALLBACK_LOG.append(entry)


# ==============================================================================
# SECTION 1 — EVT/POT : CALIBRATION DES SEUILS PAR LA THÉORIE DES VALEURS EXTRÊMES
# ==============================================================================
#
# Méthode : Peaks-Over-Threshold (POT).
# On fixe un seuil initial t₀ = Q(EVT_INIT_QUANTILE) des résidus absolus,
# puis on ajuste une GPD(ξ, σ) sur les excès (résidus > t₀ − t₀).
# Les seuils T_susp et T_atk sont extrapolés depuis cette GPD.
#
# Justification de t₀ = Q(90%) : stability plots (Coles 2001 §4.3) montrent
# que ξ et σ̃ = σ−ξ·t₀ sont stables pour 94% des métriques sur la plage
# Q(80%)–Q(97%), validant ce choix.
#
# Déclustering : désactivé (EVT_DECLUSTER_RUN = −1).
# Justification : les résidus Prophet sont déjà blanchis de la saisonnalité
# et de la tendance (Taylor & Letham 2018). L'autocorrélation résiduelle est
# structurellement faible ; le déclustering sur résidus blanchis retirerait
# principalement du signal réel, pas de la dépendance artificielle.
# ==============================================================================

def _grimshaw_fit(excesses: np.ndarray) -> tuple:
    """
    MLE pour la GPD(ξ, σ) via la réduction 1D de Grimshaw (1993).

    La condition MLE se réduit à trouver la racine de :
        g(θ) = θ · W(θ) · (1 + V(θ)) − V(θ) = 0
    avec  θ = ξ/σ,  V(θ) = mean(log(1+θ·xᵢ)),  W(θ) = mean(xᵢ/(1+θ·xᵢ)).
    Après θ̂ :  ξ̂ = V(θ̂),  σ̂ = V(θ̂)/θ̂.

    Références
    ----------
    Grimshaw (1993) Technometrics 35(2), 185–191.
    Siffer et al. (2017) KDD — algorithme SPOT.

    Returns
    -------
    (xi, sigma) : tuple[float, float]

    Raises
    ------
    ValueError si aucun candidat MLE valide n'est trouvé.
    """
    x    = np.asarray(excesses, dtype=float)
    n    = len(x)
    xbar = float(np.mean(x))
    xmax = float(np.max(x))
    eps  = 1e-8

    def _V(theta: float) -> float:
        z = 1.0 + theta * x
        return np.nan if np.any(z <= 0) else float(np.mean(np.log(z)))

    def _W(theta: float) -> float:
        z = 1.0 + theta * x
        return np.nan if np.any(z <= 0) else float(np.mean(x / z))

    def _g(theta: float) -> float:
        """Fonction dont la racine donne le MLE (Grimshaw 1993, Thm 1)."""
        v, w = _V(theta), _W(theta)
        return np.nan if (np.isnan(v) or np.isnan(w)) else theta * w * (1.0 + v) - v

    def _loglik(xi: float, sigma: float) -> float:
        if sigma <= 0:
            return -np.inf
        if abs(xi) < 1e-10:
            return float(-n * np.log(sigma) - np.sum(x) / sigma)
        z = 1.0 + xi * x / sigma
        return -np.inf if np.any(z <= 0) else float(
            -n * np.log(sigma) - (1.0 + 1.0 / xi) * np.sum(np.log(z))
        )

    candidates = []

    # ── Branche 1 : θ < 0 → ξ < 0 (queue bornée) ────────────────────────────
    # Faisabilité : 1 + θ·xmax > 0 → θ > −1/xmax
    theta_lo = -1.0 / xmax + eps
    theta_hi = -eps
    if theta_lo < theta_hi:
        try:
            g_lo, g_hi = _g(theta_lo), _g(theta_hi)
            if (not np.isnan(g_lo)) and (not np.isnan(g_hi)) and g_lo * g_hi < 0:
                th    = brentq(_g, theta_lo, theta_hi, xtol=eps, rtol=eps, maxiter=200)
                xi_c  = _V(th)
                if abs(th) > eps:
                    sigma_c = xi_c / th
                    if sigma_c > 0 and xi_c > -1.0:
                        candidates.append((xi_c, sigma_c, _loglik(xi_c, sigma_c)))
        except Exception:
            pass

    # ── Branche 2 : θ > 0 → ξ > 0 (queue lourde, Pareto) ───────────────────
    # Pour ξ ≥ 0.5 (variance infinie), la borne de Grimshaw 2(x̄−xmin)/(x̄²+s²)
    # devient instable. On maximise directement la log-vraisemblance profilée
    # ℓ_p(θ) = ℓ(ξ=V(θ), σ=V(θ)/θ), mathématiquement équivalente au
    # root-finding de Grimshaw mais robuste pour toutes valeurs de ξ.
    theta_max = 10.0 / max(xbar, eps)

    def _neg_profile_loglik(theta):
        if theta <= 0:
            return np.inf
        v = _V(theta)
        if np.isnan(v) or v <= 1e-12:
            return np.inf
        sigma_c = v / theta
        if sigma_c <= 0:
            return np.inf
        ll = _loglik(v, sigma_c)
        return -ll if np.isfinite(ll) else np.inf

    try:
        res = minimize_scalar(
            _neg_profile_loglik,
            bounds=(eps, theta_max),
            method='bounded',
            options={'xatol': eps * 10}
        )
        if res.success and np.isfinite(res.fun):
            th = float(res.x)
            xi_c = _V(th)
            if xi_c > 0 and abs(th) > eps:
                sigma_c = xi_c / th
                if sigma_c > 0:
                    candidates.append((xi_c, sigma_c, _loglik(xi_c, sigma_c)))
    except Exception:
        pass

    # ── Branche exponentielle : θ → 0, ξ → 0, σ → x̄ (toujours candidat) ───
    candidates.append((0.0, xbar, _loglik(0.0, xbar)))

    valid = [(xi, sigma, ll) for xi, sigma, ll in candidates if np.isfinite(ll)]
    if not valid:
        raise ValueError("Grimshaw: aucun candidat MLE valide.")
    xi_hat, sigma_hat, _ = max(valid, key=lambda c: c[2])
    return float(xi_hat), float(sigma_hat)


def _pwm_gpd_fit(excesses: np.ndarray) -> tuple:
    """Probability-Weighted-Moments estimator for the GPD (Hosking & Wallis 1987).

    Hosking & Wallis (1987) Technometrics 29(3), 339-349, parameterise
    the GPD as F(x) = 1 - (1 - k*x/alpha)^(1/k), and derive a closed-form
    estimator from the first two PWMs:

        a_0 = mean(x)
        a_1 = mean(x_(i) * (1 - F_i))      (sorted ascending, F_i = (i-0.35)/n)

    yielding

        k     = a_0 / (a_0 - 2*a_1) - 2
        alpha = 2 * a_0 * a_1 / (a_0 - 2*a_1)

    Beware of the sign convention: Hosking-Wallis use ``k`` and Coles
    (2001) / Pickands (1975) use ``xi = -k``.  We return the Coles-Pickands
    convention so the rest of the codebase is consistent (Grimshaw MLE
    in this module also returns Coles-convention xi).

    PWM is the standard "MLE-failed" fallback in EVT practice: it is
    closed-form, has no convergence issues for heavy-tailed distributions
    (xi > 0.5 where Grimshaw MLE becomes unstable), and is documented as
    the recommended estimator for n_peaks < 100 in Coles (2001) §4.3.4.
    The estimator is consistent for xi < 0.5 (Hosking-Wallis 1987 §3) and
    has finite asymptotic variance (positive bias for very heavy tails).

    References
    ----------
    Hosking & Wallis (1987) Technometrics 29(3): 339-349.
    Coles (2001) Intro. to Stat. Modeling of Extreme Values, §4.3.4.
    Embrechts, Kluppelberg, Mikosch (1997) Modelling Extremal Events §6.5.
    """
    x = np.asarray(excesses, dtype=float)
    n = len(x)
    if n < 4:
        raise ValueError("PWM: need >=4 excesses for a stable estimate.")
    x_sorted = np.sort(x)
    # Plotting position F_i = (i - 0.35) / n (Hosking & Wallis 1987 eq. 19).
    indices = np.arange(1, n + 1, dtype=float)
    F = (indices - 0.35) / n
    a0 = float(np.mean(x_sorted))
    a1 = float(np.mean(x_sorted * (1.0 - F)))
    denom = a0 - 2.0 * a1
    if denom <= 1e-12:
        raise ValueError(
            f"PWM: degenerate moments a0={a0:.4g}, a1={a1:.4g} (a0-2a1<=0)."
        )
    alpha = 2.0 * a0 * a1 / denom        # = sigma in Coles parameterisation
    k = a0 / denom - 2.0                 # Hosking-Wallis shape
    xi = -k                              # convert to Coles/Pickands convention
    sigma = alpha
    if sigma <= 0 or xi <= -1.0:
        raise ValueError(
            f"PWM: out-of-domain estimates xi={xi:.4g}, sigma={sigma:.4g}."
        )
    return float(xi), float(sigma)


def _evt_threshold(excesses: np.ndarray, q: float, n_total: int,
                    metric_key: str = None, branch: str = None) -> float:
    """
    Calcule z tel que P(résidu > t₀ + z | trafic normal) ≈ q.

    Méthode : quantile GPD (Siffer et al. KDD 2017, Eq. 4).
        z_q = (σ/ξ) · (q_cond^{−ξ} − 1)   si ξ ≠ 0
        z_q = σ · log(1/q_cond)             si ξ = 0
    avec q_cond = q · n_total / n_peaks  (renormalisation POT).

    ``metric_key`` / ``branch`` are used by PATCH m-08/F28 to attribute any
    fallback to the exact metric in ``_FALLBACK_LOG``. Both are optional
    (callers outside the training pipeline can omit them).

    Parameters
    ----------
    excesses : excès au-dessus de t₀ (valeurs positives)
    q        : probabilité d'excès inconditionnelle cible (ex : 1e-4)
    n_total  : nombre total de résidus d'entraînement

    Returns
    -------
    float ≥ 0
    """
    n_peaks   = len(excesses)
    min_peaks = CONFIG.get("EVT_MIN_PEAKS", 30)
    q_cond    = q * n_total / max(n_peaks, 1)

    if q_cond >= 1.0:
        return 0.0

    if n_peaks < min_peaks:
        emp_p = min(1.0 - q_cond, 1.0 - 1e-9)
        _record_fallback("evt_short_data", metric_key=metric_key, branch=branch,
                         q=q, n_peaks=int(n_peaks),
                         min_peaks=int(min_peaks), emp_p=float(emp_p))
        return float(np.quantile(excesses, emp_p)) if n_peaks > 0 else 0.0

    xi, sigma = None, None

    # 1. Grimshaw MLE (méthode de référence — Siffer 2017)
    try:
        xi, sigma = _grimshaw_fit(excesses)
    except Exception as e:
        print(f"  [EVT-WARN] Grimshaw MLE échoué ({e}) — tentative PWM fallback.")

    # 2. Fallback PWM (Hosking & Wallis 1987) — closed-form, robust for
    #    heavy tails (xi > 0.5) where MLE becomes unstable. Standard
    #    "MLE-failed" fallback in applied EVT (Coles 2001 §4.3.4).
    if xi is None or sigma is None or sigma <= 0 or xi < -1.0:
        try:
            xi, sigma = _pwm_gpd_fit(excesses)
            print(f"  [EVT-WARN] Fallback PWM (Hosking-Wallis 1987) utilisé "
                  f"(ξ={xi:.4f}, σ={sigma:.4f}).")
            _record_fallback("evt_pwm_fallback", metric_key=metric_key,
                             branch=branch, q=q,
                             xi=float(xi), sigma=float(sigma))
        except Exception as e:
            print(f"  [EVT-WARN] PWM fallback échoué ({e}) — tentative scipy.")

    # 3. Fallback scipy MLE (optimiseur générique, moins stable en queue lourde)
    if xi is None or sigma is None or sigma <= 0 or xi < -1.0:
        try:
            xi, _, sigma = genpareto.fit(excesses, floc=0)
            print(f"  [EVT-WARN] Fallback scipy.genpareto.fit utilisé "
                  f"(ξ={xi:.4f}, σ={sigma:.4f}). Vérifier la qualité des seuils.")
            _record_fallback("evt_scipy_fallback", metric_key=metric_key,
                             branch=branch, q=q,
                             xi=float(xi), sigma=float(sigma))
        except Exception as e:
            print(f"  [EVT-WARN] scipy.genpareto.fit échoué ({e}) — quantile empirique.")

    # 4. Dernier recours : quantile empirique
    if xi is None or sigma is None or sigma <= 0 or xi < -1.0:
        emp_p = min(1.0 - q_cond, 1.0 - 1e-9)
        print(f"  [EVT-WARN] Dernier recours : quantile empirique p={emp_p:.4f}.")
        _record_fallback("evt_empirical_final", metric_key=metric_key,
                         branch=branch, q=q, emp_p=float(emp_p),
                         xi=(float(xi) if xi is not None else None),
                         sigma=(float(sigma) if sigma is not None else None))
        return float(np.quantile(excesses, emp_p))

    # Quantile GPD (Siffer 2017, Eq. 4)
    q_c = max(q_cond, 1e-300)
    z   = sigma * np.log(1.0 / q_c) if abs(xi) < 1e-10 else (sigma / xi) * (q_c ** (-xi) - 1.0)
    return max(float(z), 0.0)


def _evt_threshold_pair(data: np.ndarray,
                         q_susp: float,
                         q_atk: float,
                         safety_margin: float,
                         metric_key: str = None,
                         branch: str = None) -> tuple:
    """
    Calcule (T_susp, T_atk) par POT/EVT sur un vecteur de résidus non-négatifs.

    Inclut une validation GPD : si σ̃ = σ − ξ·t₀ ≤ 0 (condition de validité
    de Coles 2001 §4.2), la GPD est instable pour ce t₀ et on revient au
    quantile empirique. Cela concerne les métriques à queue très lourde (ξ > 0.4).

    ``metric_key`` / ``branch`` (PATCH m-08/F28) are forwarded to
    :func:`_evt_threshold` so fallback events are attributed to the
    correct metric in ``_FALLBACK_LOG``.

    Parameters
    ----------
    data          : résidus directionnels (≥ 0)
    q_susp, q_atk : probabilités d'excès cibles (ex : 1e-2, 1e-3)
    safety_margin : T_atk ≥ safety_margin × T_susp (Siffer 2017)
    metric_key    : str | None — short metric id for fallback attribution.
    branch        : str | None — calibration branch ('prophet'|'ransac').

    Returns
    -------
    (t_susp, t_atk) : float, float
    """
    if len(data) < 10:
        return 1e-9, max(1e-9, safety_margin * 1e-9)

    n_total  = len(data)
    q_init   = CONFIG.get("EVT_INIT_QUANTILE", 0.90)
    t0       = float(np.quantile(data, q_init))
    min_peaks = CONFIG.get("EVT_MIN_PEAKS", 50)

    # ── Déclustering ─────────────────────────────────────────────────────────
    _run_len = CONFIG.get("EVT_DECLUSTER_RUN", -1)
    if _run_len < 0:
        # Déclustering désactivé : résidus Prophet déjà blanchis.
        # Voir justification dans le docstring du module (Section 1).
        excesses = data[data > t0] - t0
    else:
        # Déclustering run-length (Davison & Smith 1990) :
        # un cluster se ferme dès qu'un gap de ≥ r observations sous t₀ est observé.
        mask = data > t0
        clusters, in_run, run_start = [], False, 0
        for idx in range(len(mask)):
            if mask[idx] and not in_run:
                in_run, run_start = True, idx
            elif not mask[idx] and in_run:
                gap_end = min(idx + _run_len, len(mask))
                if not any(mask[idx:gap_end]):
                    clusters.append(float(np.max(data[run_start:idx]) - t0))
                    in_run = False
        if in_run:
            clusters.append(float(np.max(data[run_start:]) - t0))
        excesses = np.array(clusters) if clusters else (data[data > t0] - t0)
        if len(excesses) < len(data[data > t0]):
            print(f"  [EVT-DECLUSTER] {len(data[data > t0])} excès bruts → "
                  f"{len(excesses)} clusters indépendants (r={_run_len})")

    # ── Fit GPD avec validation σ̃ > 0 ────────────────────────────────────────
    if len(excesses) < min_peaks:
        # Pas assez de pics → quantile empirique direct
        t_susp = float(np.quantile(data, min(1.0 - q_susp, 1.0 - 1e-9)))
        t_atk  = float(np.quantile(data, min(1.0 - q_atk,  1.0 - 1e-9)))
        # PATCH m-08/F28 — audit trail for empirical fallback at pair level
        _record_fallback("evt_short_data_pair", metric_key=metric_key,
                         branch=branch, n_excesses=int(len(excesses)),
                         min_peaks=int(min_peaks),
                         t_susp=float(t_susp), t_atk=float(t_atk))
    else:
        # Validation GPD avant application : σ̃ = σ − ξ·t₀ doit être > 0.
        # σ̃ < 0 indique une queue trop lourde (ξ >> 0) pour ce t₀ :
        # le MLE est instable et les seuils extrapolés seraient aberrants.
        # Ref : Coles (2001) §4.2 — condition nécessaire de validité GPD.
        _valid_gpd = True
        _xi_check = _sigma_mod = None
        try:
            _xi_check, _, _sigma_check = genpareto.fit(excesses, floc=0)
            _sigma_mod = _sigma_check - _xi_check * t0
            if _sigma_mod <= 0:
                _valid_gpd = False
                print(f"  [EVT-WARN] σ̃ = {_sigma_mod:.4f} ≤ 0 "
                      f"(ξ={_xi_check:.3f}, t₀={t0:.4f}) "
                      f"→ GPD instable → fallback quantile empirique")
        except Exception:
            pass  # Si le check échoue, on tente quand même le fit GPD

        if not _valid_gpd:
            t_susp = float(np.quantile(data, min(1.0 - q_susp, 1.0 - 1e-9)))
            t_atk  = float(np.quantile(data, min(1.0 - q_atk,  1.0 - 1e-9)))
            # PATCH m-08/F28 — σ̃≤0 fallback is the paper's documented
            # "EVT instable 7/17" condition; log with full diagnostics.
            _record_fallback("evt_sigma_mod", metric_key=metric_key,
                             branch=branch,
                             xi=(float(_xi_check) if _xi_check is not None else None),
                             sigma_mod=(float(_sigma_mod) if _sigma_mod is not None else None),
                             t0=float(t0),
                             t_susp=float(t_susp), t_atk=float(t_atk))
        else:
            t_susp = t0 + _evt_threshold(excesses, q_susp, n_total,
                                          metric_key=metric_key, branch=branch)
            t_atk  = t0 + _evt_threshold(excesses, q_atk,  n_total,
                                          metric_key=metric_key, branch=branch)

    t_atk  = max(t_atk,  safety_margin * t_susp)
    t_susp = max(t_susp, 1e-9)
    t_atk  = max(t_atk,  1e-9)
    return t_susp, t_atk


# ==============================================================================
# SECTION 2 — CALIBRATION DES SEUILS DIRECTIONNELS
# ==============================================================================
#
# Les résidus de trafic réseau ne sont pas symétriques :
#   - bytes/packets : une surcharge est suspecte (dir='both' ou 'pos')
#   - entropies : une chute est suspecte, une hausse normale (dir='neg')
#   - syn : seul l'excès positif signale un SYN Flood (dir='pos')
#
# La direction est lue dans CONFIG["ASYMMETRIC_THRESHOLD_METRICS"] et stockée
# dans l'artefact pour que compute_evidence_v2.py applique le bon filtrage.
#
# Note sur r2 interne : la variable `_dispersion_ratio` calculée ici n'est PAS
# le R² du modèle (calculé séparément). C'est un ratio signal/dispersion
# utilisé uniquement pour la classification de stabilité A/B/C interne.
# ==============================================================================

def calibrate_thresholds_v2(signed_residuals: np.ndarray,
                              metric_key: str,
                              branch: str) -> dict:
    """
    Calcule T_susp, T_atk et T_trapeze_base pour une métrique et une branche.

    Parameters
    ----------
    signed_residuals : np.ndarray
        Résidus SIGNÉS (y − ŷ). La direction est lue dans CONFIG.
    metric_key : str
        Clé courte de la métrique (ex : 'bytes', 'bytes_packets').
    branch : str
        'prophet' ou 'ransac' (utilisé pour sélectionner les paramètres EVT).

    Returns
    -------
    dict avec clés : t_susp, t_atk, t_trapeze_base, t_susp_pos, t_atk_pos,
                     t_trapeze_base_pos, t_susp_neg, t_atk_neg,
                     t_trapeze_base_neg, direction, branch,
                     r2, kurtosis, cv, q_susp_used, q_atk_used, evt_mode
    """
    signed_residuals = signed_residuals[~np.isnan(signed_residuals)]
    if len(signed_residuals) < 10:
        return {'t_susp': 1e-6, 't_atk': 1e-5, 't_trapeze_base': 9e-7,
                'direction': 'sym', 'branch': branch,
                'r2': 0.0, 'kurtosis': 0.0, 'cv': 0.0,
                'q_susp_used': 0.99, 'q_atk_used': 0.9995}

    if branch == 'prophet':
        q_susp = CONFIG["Q_SUSP_PROPHET"]
        q_atk  = CONFIG["Q_ATK_PROPHET"]
    elif branch == 'ransac':
        # Note : la branche reconstruction utilise toujours 'ransac' comme
        # identifiant des paramètres EVT, indépendamment de l'estimateur
        # sous-jacent (ici QuantileRegressor). Le nom est conservé par
        # rétrocompatibilité avec config.py.
        q_susp = CONFIG["Q_SUSP_RANSAC"]
        q_atk  = CONFIG["Q_ATK_RANSAC"]
    else:
        raise ValueError(f"branch doit être 'prophet' ou 'ransac', reçu '{branch}'")

    safety_margin    = CONFIG.get("THRESHOLD_SAFETY_MARGIN", 1.10)
    use_qtrapeze     = CONFIG.get("USE_QUANTILE_TRAPEZE", False)
    q_trapeze_base_p = CONFIG.get("Q_TRAPEZE_BASE", 0.95)
    asym_cfg         = CONFIG.get("ASYMMETRIC_THRESHOLD_METRICS", {})
    direction        = asym_cfg.get(metric_key, None)
    use_evt          = CONFIG.get("USE_EVT_THRESHOLDS", True)

    # Paramètres EVT par branche (probabilités d'excès cibles P(|r| > T | normal))
    # Correspondance avec anciens quantiles : EVT_Q = 1 − Q_old
    if branch == 'prophet':
        q_evt_susp = CONFIG.get("EVT_Q_SUSP_PROPHET", CONFIG.get("EVT_Q_SUSP", 0.02))
        q_evt_atk  = CONFIG.get("EVT_Q_ATK_PROPHET",  CONFIG.get("EVT_Q_ATK",  0.01))
    else:
        q_evt_susp = CONFIG.get("EVT_Q_SUSP_RANSAC", CONFIG.get("EVT_Q_SUSP", 0.01))
        q_evt_atk  = CONFIG.get("EVT_Q_ATK_RANSAC",  CONFIG.get("EVT_Q_ATK",  0.001))

    if direction == 'both':
        # Deux paires de seuils : queue positive (surcharge) et négative (coupure)
        pos_r = signed_residuals[signed_residuals >= 0]
        neg_r = np.abs(signed_residuals[signed_residuals < 0])
        if len(pos_r) < 10:
            pos_r = np.abs(signed_residuals)
        if len(neg_r) < 10:
            neg_r = np.abs(signed_residuals)

        if use_evt:
            t_susp_pos, t_atk_pos = _evt_threshold_pair(pos_r, q_evt_susp, q_evt_atk,
                                                         safety_margin,
                                                         metric_key=f"{metric_key}:pos",
                                                         branch=branch)
            t_susp_neg, t_atk_neg = _evt_threshold_pair(neg_r, q_evt_susp, q_evt_atk,
                                                         safety_margin,
                                                         metric_key=f"{metric_key}:neg",
                                                         branch=branch)
        else:
            t_susp_pos = max(float(np.quantile(pos_r, q_susp)), 1e-9)
            t_atk_pos  = max(float(np.quantile(pos_r, q_atk)), safety_margin * t_susp_pos)
            t_susp_neg = max(float(np.quantile(neg_r, q_susp)), 1e-9)
            t_atk_neg  = max(float(np.quantile(neg_r, q_atk)), safety_margin * t_susp_neg)

        t_trapeze_base_pos = (float(np.quantile(pos_r, q_trapeze_base_p))
                              if use_qtrapeze else 0.9 * t_susp_pos)
        t_trapeze_base_neg = (float(np.quantile(neg_r, q_trapeze_base_p))
                              if use_qtrapeze else 0.9 * t_susp_neg)
        t_susp         = t_susp_pos
        t_atk          = t_atk_pos
        t_trapeze_base = t_trapeze_base_pos
    else:
        if direction == 'pos':
            directional = signed_residuals[signed_residuals >= 0]
            if len(directional) < 10:
                directional = np.abs(signed_residuals)
        elif direction == 'neg':
            directional = np.abs(signed_residuals[signed_residuals <= 0])
            if len(directional) < 10:
                directional = np.abs(signed_residuals)
        else:
            directional = np.abs(signed_residuals)
            direction   = 'sym'

        if use_evt:
            t_susp, t_atk = _evt_threshold_pair(directional, q_evt_susp, q_evt_atk,
                                                 safety_margin,
                                                 metric_key=metric_key, branch=branch)
        else:
            t_susp = float(np.quantile(directional, q_susp))
            t_atk  = float(np.quantile(directional, q_atk))
            t_atk  = max(t_atk, safety_margin * t_susp)
            t_susp = max(t_susp, 1e-9)
            t_atk  = max(t_atk,  1e-9)

        t_trapeze_base = (float(np.quantile(directional, q_trapeze_base_p))
                          if use_qtrapeze else 0.9 * t_susp)
        if use_qtrapeze:
            t_trapeze_base = min(t_trapeze_base, t_susp * 0.999)
        t_susp_pos = t_atk_pos = t_susp_neg = t_atk_neg = None
        t_trapeze_base_pos = t_trapeze_base_neg = None

    # Statistiques descriptives (traçabilité + classification de stabilité)
    abs_r       = np.abs(signed_residuals)
    _mean_abs_r = np.mean(abs_r)
    _var_abs_r  = np.sum((abs_r - _mean_abs_r) ** 2)
    # _dispersion_ratio : ratio signal/dispersion interne (NON le R² du modèle).
    # Utilisé uniquement pour la classification de stabilité A/B/C dans
    # compute_evidence_v2. Négatif quand mean > std (cas fréquent → classe C).
    _dispersion_ratio = (float(1.0 - np.sum(abs_r ** 2) / _var_abs_r)
                         if _var_abs_r > 0 else 0.0)
    kurtosis_val = float(stats.kurtosis(abs_r, fisher=True))
    cv           = float(np.std(abs_r) / _mean_abs_r) if _mean_abs_r > 0 else float('inf')

    return {
        't_susp':              t_susp,
        't_atk':               t_atk,
        't_trapeze_base':      t_trapeze_base,
        't_susp_pos':          t_susp_pos if direction == 'both' else t_susp,
        't_atk_pos':           t_atk_pos  if direction == 'both' else t_atk,
        't_trapeze_base_pos':  t_trapeze_base_pos if direction == 'both' else t_trapeze_base,
        't_susp_neg':          t_susp_neg if direction == 'both' else t_susp,
        't_atk_neg':           t_atk_neg  if direction == 'both' else t_atk,
        't_trapeze_base_neg':  t_trapeze_base_neg if direction == 'both' else t_trapeze_base,
        'direction':           direction,
        'branch':              branch,
        'r2':                  _dispersion_ratio,  # ratio interne, pas le R² modèle
        'kurtosis':            kurtosis_val,
        'cv':                  cv,
        'q_susp_used':         q_evt_susp if use_evt else q_susp,
        'q_atk_used':          q_evt_atk  if use_evt else q_atk,
        'evt_mode':            use_evt,
    }


def calibrate_thresholds_per_regime_v2(
    signed_residuals: np.ndarray,
    timestamps,
    metric_key: str,
    branch: str,
    holidays=None,
) -> dict:
    """Calibrate per-regime EVT thresholds (PATCH H2 — calendar-aware EVT).

    Calls :func:`calibrate_thresholds_v2` once per bucket of
    :data:`sl_ads.calendar.regime.REGIME_BUCKETS`, partitioning the
    residuals by the timestamp-driven regime function.

    Parameters
    ----------
    signed_residuals : np.ndarray
        Signed residuals (same shape as the global calibration residuals
        consumed by ``calibrate_thresholds_v2``).
    timestamps : array-like of pandas-coercible timestamps
        Per-residual timestamps, *aligned* with ``signed_residuals``.
        Must have the same length.
    metric_key, branch : str
        Forwarded to ``calibrate_thresholds_v2`` (used for fallback
        attribution in ``_FALLBACK_LOG``).  The bucket name is
        appended to ``metric_key`` (``f"{metric_key}:{bucket}"``) so
        per-regime fallback events stay attributable.
    holidays : optional iterable
        Forwarded to ``regime_of_series``.  ``CONFIG['HOLIDAYS_LIST']``
        is the canonical source on RedeRio (TASK-15 + PATCH H2).

    Returns
    -------
    dict
        ``{
            "regime_fn_signature": str,
            "buckets": [...],
            "thresholds_per_regime": {bucket: thresholds_v2_dict, ...},
            "n_residuals_per_regime": {bucket: int, ...},
        }``

        Buckets with fewer than ``EVT_MIN_PEAKS`` *windows* (not peaks)
        fall back to the empirical-quantile branch inside
        ``calibrate_thresholds_v2`` with the same ``_record_fallback``
        attribution as the global path; the returned dict still
        contains a usable ``thresholds_v2_dict`` for them.

    Notes
    -----
    Backward-compatibility contract: this helper does NOT replace
    ``calibrate_thresholds_v2``.  Callers are expected to keep the
    legacy scalar fields (``t_susp``, ``t_atk``, ``t_trapeze_base``,
    etc.) computed on the *global* residuals and store the per-regime
    block alongside them under ``models_pkg[metric]['thresholds_per_regime']``.
    Inference time chooses one or the other based on the sidecar
    structure (see ``sl_ads.train.compute_evidence``).
    """
    if len(signed_residuals) != len(timestamps):
        raise ValueError(
            f"calibrate_thresholds_per_regime_v2: residual/timestamp "
            f"length mismatch ({len(signed_residuals)} vs {len(timestamps)})"
        )
    ts_series = pd.Series(pd.to_datetime(list(timestamps)))
    regime_labels = regime_of_series(ts_series, holidays=holidays).to_numpy()

    out_thresholds = {}
    out_counts = {b: 0 for b in REGIME_BUCKETS}
    for bucket in REGIME_BUCKETS:
        mask = regime_labels == bucket
        bucket_residuals = signed_residuals[mask]
        out_counts[bucket] = int(mask.sum())
        if len(bucket_residuals) < 10:
            # Bucket has too few windows for any meaningful EVT or even
            # empirical fallback.  Use the global residuals as a
            # last-resort calibration so the dict has a usable entry
            # for every bucket; ``_record_fallback`` attributes the
            # event to ``{metric_key}:{bucket}:underflow``.
            _record_fallback(
                "calendar_evt_underflow_global_fallback",
                metric_key=f"{metric_key}:{bucket}",
                branch=branch,
                n_bucket_windows=int(mask.sum()),
            )
            bucket_residuals = signed_residuals
        out_thresholds[bucket] = calibrate_thresholds_v2(
            bucket_residuals,
            metric_key=f"{metric_key}:{bucket}",
            branch=branch,
        )
    return {
        "regime_fn_signature": REGIME_FN_SIGNATURE,
        "buckets": list(REGIME_BUCKETS),
        "thresholds_per_regime": out_thresholds,
        "n_residuals_per_regime": out_counts,
    }


# ==============================================================================
# SECTION 3 — EMPIRICAL DIRICHLET PRIOR (EDP)
# ==============================================================================
#
# L'opinion SL de chaque métrique est initialisée avec un prior a = (a_safe,
# a_susp, a_atk). Par défaut ce prior est uniforme (1/3, 1/3, 1/3), ce qui
# est non-informatif. L'EDP le remplace par une estimation empirique depuis
# les résidus d'entraînement (trafic strictement normal).
#
#    Empirical Dirichlet Prior — estimateur fréquentiste marginal.
#    Conforme à Ferguson (1973) "A Bayesian analysis of some nonparametric
#    problems" Ann. Stat. 1:209-230 (Dirichlet process prior), spécialisé
#    en Empirical Bayes (Robbins 1955, Robbins 1983).
#    N.B. : ne pas citer Efron & Morris 1973 ici — cet article porte sur
#    les estimateurs de James-Stein à rétrécissement (shrinkage), non
#    implémenté dans notre EDP. Nos base rates sont des fréquences
#    empiriques simples sans shrinkage.
#   a_j = mean_t[ R_j(t) ] / W   pour j ∈ {safe, susp, anom}
# où R_j(t) est la proportion de l'évidence de type j dans la fenêtre t,
# calculée par le mapping trapézoïdal sur les résidus.
#
# Interprétation : sur trafic normal, la grande majorité des résidus tombent
# sous T_susp → a_safe ≈ 0.98, a_susp ≈ 0.01, a_atk ≈ 0.005 (plancher).
# Ce prior pénalise légèrement les métriques bruyantes qui génèrent souvent
# des évidence suspectes même en l'absence d'attaque.
# ==============================================================================

def _apply_trapezoid_single(e: float,
                             t_susp: float,
                             t_atk: float,
                             t_trapeze_base: float) -> tuple:
    """
    Mapping trapézoïdal sur un résidu directionnel scalaire (≥ 0).
    Retourne (p, s, n) avec p + s + n = 1.
    """
    if e < t_trapeze_base:
        return (1.0, 0.0, 0.0)
    elif e < t_susp:
        width = t_susp - t_trapeze_base
        alpha = (e - t_trapeze_base) / width if width > 0 else 1.0
        return (1.0 - alpha, alpha, 0.0)
    elif e < t_atk:
        width = t_atk - t_susp
        alpha = (e - t_susp) / width if width > 0 else 1.0
        return (0.0, 1.0 - alpha, alpha)
    else:
        return (0.0, 0.0, 1.0)


def compute_edp_from_residuals(signed_residuals: np.ndarray,
                                thresh: dict,
                                window_size: int = WINDOW_SIZE,
                                prior_floor: float = 0.005) -> dict:
    """
    Calcule le prior Dirichlet empirique (EDP) d'une métrique.

    Pour les métriques direction='both', retourne également les composantes
    directionnelles (a_susp_pos, a_atk_pos, a_susp_neg, a_atk_neg) pour
    alimenter le frame 5-états de qualify_anomaly_sbn.

    Propriété de cohérence (Jøsang 2016 §3.5.4 — coarsening) :
      a_susp = a_susp_pos + a_susp_neg
      a_atk  = a_atk_pos  + a_atk_neg
    Les valeurs ternaires sont strictement identiques à l'ancienne version
    (aucune régression sur le pipeline de détection).

    Parameters
    ----------
    signed_residuals : résidus signés (y − ŷ) d'entraînement
    thresh           : artefact calibrate_thresholds_v2 de cette métrique
                       (doit contenir t_susp_pos/neg, t_atk_pos/neg,
                       t_trapeze_base_pos/neg pour direction='both')
    window_size      : nombre de tranches par fenêtre (= WINDOW_SIZE)
    prior_floor      : plancher sur a_atk (évite un prior identiquement nul)
                       Appliqué séparément à a_atk_pos et a_atk_neg pour
                       direction='both'.

    Returns
    -------
    dict : {'a_safe', 'a_susp', 'a_atk'}  (somme = 1, toutes directions)
           + {'a_susp_pos', 'a_atk_pos', 'a_susp_neg', 'a_atk_neg'}
             si direction='both' (somme des 5 = 1)
    """
    direction      = thresh['direction']
    t_susp         = thresh['t_susp']
    t_atk          = thresh['t_atk']
    t_trapeze_base = thresh['t_trapeze_base']
    t_susp_pos = thresh.get('t_susp_pos', t_susp)
    t_atk_pos  = thresh.get('t_atk_pos',  t_atk)
    t_tb_pos   = thresh.get('t_trapeze_base_pos', t_trapeze_base)
    t_susp_neg = thresh.get('t_susp_neg', t_susp)
    t_atk_neg  = thresh.get('t_atk_neg',  t_atk)
    t_tb_neg   = thresh.get('t_trapeze_base_neg', t_trapeze_base)

    P_list, S_list, N_list = [], [], []
    # Composantes directionnelles (direction='both' uniquement)
    # Invariant : S_list[i] = S_pos_list[i] + S_neg_list[i]  (idem N)
    S_pos_list, N_pos_list = [], []
    S_neg_list, N_neg_list = [], []

    for start in range(0, len(signed_residuals), window_size):
        window = signed_residuals[start:start + window_size]
        P_win = S_win = N_win = 0.0
        S_pos_win = N_pos_win = 0.0
        S_neg_win = N_neg_win = 0.0

        for r in window:
            if direction == 'pos':
                if r <= 0.0:
                    P_win += 1.0; continue
                p, s, n = _apply_trapezoid_single(r, t_susp, t_atk, t_trapeze_base)
            elif direction == 'neg':
                if r >= 0.0:
                    P_win += 1.0; continue
                p, s, n = _apply_trapezoid_single(abs(r), t_susp, t_atk, t_trapeze_base)
            elif direction == 'both':
                if r >= 0.0:
                    p, s, n = _apply_trapezoid_single(r, t_susp_pos, t_atk_pos, t_tb_pos)
                    S_pos_win += s; N_pos_win += n
                else:
                    p, s, n = _apply_trapezoid_single(abs(r), t_susp_neg, t_atk_neg, t_tb_neg)
                    S_neg_win += s; N_neg_win += n
            else:  # sym
                p, s, n = _apply_trapezoid_single(abs(r), t_susp, t_atk, t_trapeze_base)
            P_win += p; S_win += s; N_win += n

        P_list.append(P_win)
        S_list.append(S_win)
        N_list.append(N_win)
        if direction == 'both':
            S_pos_list.append(S_pos_win); N_pos_list.append(N_pos_win)
            S_neg_list.append(S_neg_win); N_neg_list.append(N_neg_win)

    mean_P = np.mean(P_list) / window_size
    mean_S = np.mean(S_list) / window_size
    mean_N = max(np.mean(N_list) / window_size, prior_floor)  # plancher a_atk

    total = mean_P + mean_S + mean_N
    if total <= 0:
        base = {'a_safe': 1/3, 'a_susp': 1/3, 'a_atk': 1/3}
        if direction == 'both':
            base.update({'a_susp_pos': 1/6, 'a_atk_pos': 1/6,
                         'a_susp_neg': 1/6, 'a_atk_neg': 1/6})
        return base

    result = {
        'a_safe': float(mean_P / total),
        'a_susp': float(mean_S / total),
        'a_atk':  float(mean_N / total),
    }

    # Composantes directionnelles pour direction='both'
    # Plancher appliqué séparément sur chaque côté (flood et déficit peuvent
    # avoir des queues très asymétriques sur trafic d'entraînement normal).
    if direction == 'both' and S_pos_list:
        mean_S_pos = np.mean(S_pos_list) / window_size
        mean_N_pos = max(np.mean(N_pos_list) / window_size, prior_floor)
        mean_S_neg = np.mean(S_neg_list) / window_size
        mean_N_neg = max(np.mean(N_neg_list) / window_size, prior_floor)

        # Renormalisation 5-états : remplace mean_N (plancher global) par la
        # somme des planchers directionnels (N_pos + N_neg ≥ 2 × prior_floor).
        # On recalcule total5 pour garantir Σa_i = 1 sur les 5 états.
        total5 = mean_P + mean_S_pos + mean_S_neg + mean_N_pos + mean_N_neg
        if total5 > 0:
            result['a_susp_pos'] = float(mean_S_pos / total5)
            result['a_atk_pos']  = float(mean_N_pos / total5)
            result['a_susp_neg'] = float(mean_S_neg / total5)
            result['a_atk_neg']  = float(mean_N_neg / total5)
            # Mise à jour ternaire cohérente avec les planchers directionnels
            # (Jøsang §3.5.4 : a_susp = a_susp_pos + a_susp_neg, idem atk)
            result['a_safe'] = float(mean_P / total5)
            result['a_susp'] = result['a_susp_pos'] + result['a_susp_neg']
            result['a_atk']  = result['a_atk_pos']  + result['a_atk_neg']

    return result


# ==============================================================================
# SECTION 4 — AUTO-CALIBRATION DE RECONST_ATTACK_RELIABILITY
# ==============================================================================
#
# La branche Reconstruction (QR) capture les anomalies structurelles
# (ex : bytes anormaux vs packets normaux). En revanche, pour les attaques
# applicatives (Slowloris) qui ne perturbent pas les relations structurelles,
# la Reconstruction génère de l'évidence "safe" avec haute confiance, ce qui
# dilue le signal d'attaque de Prophet via CBF.
#
# RECONST_ATTACK_RELIABILITY = α ∈ [0, 1] contrôle le discounting contextuel
# (Mercier & Denoeux 2008) de la Reconstruction pour l'hypothèse "attack" :
#   1.0 → pas de discounting (comportement CBF standard)
#   0.1 → Reconstruction quasi-ignorée pour "attack" (recommandé si Slowloris)
#
# En mode "auto", α est estimé depuis le taux de "cécité structurelle" :
# proportion de fenêtres où Prophet est suspect ET Reconst dit "safe".
# ==============================================================================

def _auto_calibrate_reconst_reliability(models_pkg: dict,
                                         calib_residuals: dict,
                                         window_size: int) -> float:
    """
    Calibre automatiquement RECONST_ATTACK_RELIABILITY via le taux de
    cécité structurelle de la Reconstruction par rapport à Prophet.

    Returns
    -------
    float : alpha_attack ∈ [0.05, 1.0]
    """
    try:
        import sl_ads.core.subjective_logic as sl  # Phase H
    except ImportError:
        print("  [AUTO-CD] sl_formulas_v2 non importable — alpha_attack=1.0 (désactivé)")
        return 1.0

    SAFE_THR    = 0.85   # seuil b_safe pour qu'on considère Reconst "dominée safe"
    MIN_SUSPECT = 10     # fenêtres Prophet-suspect minimum pour estimer

    W_BIJ = float(CONFIG.get('SL_PARAM_K', 3.0))
    edp   = models_pkg.get('empirical_priors') or {}

    prophet_keys = [k for k in models_pkg
                    if k.startswith('prophet_') and isinstance(models_pkg[k], dict)
                    and k in calib_residuals]
    reconst_keys = [k for k in models_pkg
                    if k.startswith('reconst_') and isinstance(models_pkg[k], dict)
                    and k in calib_residuals]

    if not prophet_keys or not reconst_keys:
        print("  [AUTO-CD] Métriques insuffisantes — alpha_attack=1.0 (désactivé)")
        return 1.0

    n_windows = None
    for key in prophet_keys + reconst_keys:
        n = len(calib_residuals[key]) // window_size
        if n > 0:
            n_windows = n if n_windows is None else min(n_windows, n)
    if not n_windows:
        print("  [AUTO-CD] Pas de fenêtres disponibles — alpha_attack=1.0 (désactivé)")
        return 1.0

    n_prophet_suspect = 0
    n_reconst_blind   = 0

    def _window_evidence(key, win_idx):
        """Calcule (P, S, N) pour une métrique sur une fenêtre."""
        res = calib_residuals[key]
        pkg = models_pkg[key]
        window = res[win_idx * window_size:(win_idx + 1) * window_size]
        if len(window) == 0:
            return 0.0, 0.0, 0.0
        direction = pkg['direction']
        t_s = pkg['t_susp']; t_a = pkg['t_atk']; t_tb = pkg['t_trapeze_base']
        t_sp  = pkg.get('t_susp_pos', t_s);  t_ap  = pkg.get('t_atk_pos', t_a)
        t_tbp = pkg.get('t_trapeze_base_pos', t_tb)
        t_sn  = pkg.get('t_susp_neg', t_s);  t_an  = pkg.get('t_atk_neg', t_a)
        t_tbn = pkg.get('t_trapeze_base_neg', t_tb)
        P_w = S_w = N_w = 0.0
        for r in window:
            if direction == 'pos':
                if r <= 0.0: P_w += 1.0; continue
                p, s, n = _apply_trapezoid_single(r, t_s, t_a, t_tb)
            elif direction == 'neg':
                if r >= 0.0: P_w += 1.0; continue
                p, s, n = _apply_trapezoid_single(abs(r), t_s, t_a, t_tb)
            elif direction == 'both':
                if r >= 0.0:
                    p, s, n = _apply_trapezoid_single(r, t_sp, t_ap, t_tbp)
                else:
                    p, s, n = _apply_trapezoid_single(abs(r), t_sn, t_an, t_tbn)
            else:
                p, s, n = _apply_trapezoid_single(abs(r), t_s, t_a, t_tb)
            P_w += p; S_w += s; N_w += n
        return P_w, S_w, N_w

    for win_idx in range(n_windows):
        # Evidence Prophet agrégée sur la fenêtre
        total_N_p = sum(_window_evidence(k, win_idx)[2] for k in prophet_keys)
        if total_N_p == 0:
            continue

        # Opinion Reconst sur la même fenêtre
        total_P_r = total_S_r = total_N_r = 0.0
        a_r_sum = np.zeros(3)
        n_r = 0
        for key in reconst_keys:
            P_w, S_w, N_w = _window_evidence(key, win_idx)
            total_P_r += P_w; total_S_r += S_w; total_N_r += N_w
            prior = edp.get(key)
            a_r_sum += (np.array([prior['a_safe'], prior['a_susp'], prior['a_atk']])
                        if prior else np.array([1/3, 1/3, 1/3]))
            n_r += 1

        if n_r == 0:
            continue

        a_r = a_r_sum / n_r
        s   = a_r.sum()
        if s > 0:
            a_r /= s
        op_reconst = sl.evidence_to_opinion(
            [total_P_r, total_S_r, total_N_r], W=W_BIJ, a=a_r
        )
        n_prophet_suspect += 1
        if op_reconst.b[0] > SAFE_THR:
            n_reconst_blind += 1

    print(f"  [AUTO-CD] {n_prophet_suspect} fenêtres Prophet-suspect / {n_windows} total")
    if n_prophet_suspect < MIN_SUSPECT:
        alpha_attack = 0.30
        print(f"  [AUTO-CD] Trop peu de fenêtres suspect ({n_prophet_suspect} < {MIN_SUSPECT})"
              f" → prior empirique alpha_attack={alpha_attack:.2f}")
    else:
        blind_rate   = n_reconst_blind / n_prophet_suspect
        alpha_attack = float(np.clip(1.0 - blind_rate, 0.05, 1.0))
        print(f"  [AUTO-CD] blind_rate = {n_reconst_blind}/{n_prophet_suspect} = {blind_rate:.3f}")
        print(f"  [AUTO-CD] alpha_attack = max(0.05, 1−{blind_rate:.3f}) = {alpha_attack:.3f}")

    return alpha_attack


# ==============================================================================
# SECTION 5 — AUTO-CALIBRATION DU DECISION_THRESHOLD
# ==============================================================================
#
# Variable de décision : proj_atk = b_atk + a_atk · u  (Jøsang 2016, Eq. 3.23)
# Contrairement à b_atk (bimodal, masse de probabilité concentrée en 0),
# proj_atk est une distribution continue et lisse → threshold stable.
#
# Méthode : DECISION_THRESHOLD = quantile(proj_atk, 1 − FPR_TARGET) sur le
# set de calibration hors-entraînement (derniers CALIB_SPLIT_FRACTION × T_train).
# Ref : validation set temporel pour l'anomaly detection (Ruff et al. 2021).
#
# Fusion inter-métriques : CBF additive (Jøsang 2016, Théorème 12.2) —
# toutes les preuves (P_i, S_i, N_i) sont sommées avant la bijection.
# Équivalent à une CBF itérée sur toutes les métriques.
#
# Note sur l'ageing : cette fonction calcule des opinions INSTANTANÉES (sans
# l'ageing λ de compute_opinions_v3). L'ageing lisse la distribution et réduit
# sa variance par rapport à l'instantané. Le seuil δ calibré ici est donc
# conservateur pour la détection en temps réel avec ageing.
# ==============================================================================

def _compute_training_proj_atk(models_pkg: dict,
                                 train_residuals: dict,
                                 window_size: int,
                                 fusion_mode: str | None = None) -> np.ndarray | None:
    """
    Calcule la distribution de proj_atk sur les fenêtres de calibration
    (trafic strictement normal) pour estimer DECISION_THRESHOLD.

    La calibration suit la meme structure method-level que l'inference :
    feuilles -> WBF intra-methode -> fusion inter-methode configuree.
    Elle reste volontairement instantanee (sans ageing temporel) pour garder
    une calibration conservatrice et rapide.

    Returns
    -------
    np.ndarray de proj_atk sur toutes les fenêtres, ou None si échec.
    """
    try:
        import sl_ads.core.subjective_logic as sl  # Phase H
    except ImportError:
        print("  [WARN] sl_formulas_v2 non importable — DECISION_THRESHOLD non auto-calibré.")
        return None

    W_BIJ = float(CONFIG.get('SL_PARAM_K', 3.0))
    edp   = models_pkg.get('empirical_priors') or {}
    mode = str(fusion_mode or CONFIG.get("INTER_METHOD_FUSION", "wbf")).strip().lower()

    metric_keys = [
        k for k in models_pkg
        if isinstance(models_pkg[k], dict)
        and not str(k).startswith("_")
        and k not in ('empirical_priors', 'trust_scores', 'mase_scores')
        and k in train_residuals
    ]
    if not metric_keys:
        return None

    metric_meta = {k: models_pkg[k] for k in metric_keys}
    try:
        method_groups = fusion_policy.get_method_groups(CONFIG)
        group_keys = fusion_policy.group_metric_keys(metric_meta, CONFIG)
    except ValueError as exc:
        print(f"  [WARN] Groupes de fusion invalides pour DECISION_THRESHOLD: {exc}")
        return None

    n_windows = None
    for key in metric_keys:
        n = len(train_residuals[key]) // window_size
        if n > 0:
            n_windows = n if n_windows is None else min(n_windows, n)
    if not n_windows:
        return None

    proj_atk_list = []
    c3_mode = str(CONFIG.get("C3_WEIGHT_MODE", "uniform")).lower()
    wbf_weight_mode = str(CONFIG.get("WBF_WEIGHT_MODE", "uniform")).lower()
    use_uncertainty_max = bool(CONFIG.get("UNCERTAINTY_MAXIMIZATION", False))
    fallback_prior = np.asarray(CONFIG.get("SL_PRIOR_A", [1.0 / 3.0] * 3), dtype=float)
    if fallback_prior.shape != (3,) or fallback_prior.sum() <= 0:
        fallback_prior = np.full(3, 1.0 / 3.0)
    fallback_prior = fallback_prior / fallback_prior.sum()

    def _prior_for_metric(key: str) -> np.ndarray:
        prior = edp.get(key) if isinstance(edp, dict) else None
        if prior:
            a = np.array([prior['a_safe'], prior['a_susp'], prior['a_atk']], dtype=float)
        else:
            a = fallback_prior.copy()
        s = float(a.sum())
        return a / s if s > 0 else np.full(3, 1.0 / 3.0)

    def _calib_weight_for_metric(key: str) -> float:
        if c3_mode == "uniform":
            return 1.0
        # Interval/online weights are not available on raw training residuals;
        # use the same static fallback as compute_opinions_v3.
        return max(float(models_pkg[key].get('r2_score', 0.01)), 0.01)

    for win_idx in range(n_windows):
        ops_by_group = {group["name"]: [] for group in method_groups}
        weights_by_group = {group["name"]: [] for group in method_groups}

        for key in metric_keys:
            res    = train_residuals[key]
            pkg    = models_pkg[key]
            window = res[win_idx * window_size:(win_idx + 1) * window_size]
            if len(window) == 0:
                continue

            direction = pkg['direction']
            t_susp = pkg['t_susp']; t_atk = pkg['t_atk']
            t_tb   = pkg['t_trapeze_base']
            t_sp   = pkg.get('t_susp_pos', t_susp)
            t_ap   = pkg.get('t_atk_pos',  t_atk)
            t_tbp  = pkg.get('t_trapeze_base_pos', t_tb)
            t_sn   = pkg.get('t_susp_neg', t_susp)
            t_an   = pkg.get('t_atk_neg',  t_atk)
            t_tbn  = pkg.get('t_trapeze_base_neg', t_tb)

            P_w = S_w = N_w = 0.0
            for r in window:
                if direction == 'pos':
                    if r <= 0.0: P_w += 1.0; continue
                    p, s, n = _apply_trapezoid_single(r, t_susp, t_atk, t_tb)
                elif direction == 'neg':
                    if r >= 0.0: P_w += 1.0; continue
                    p, s, n = _apply_trapezoid_single(abs(r), t_susp, t_atk, t_tb)
                elif direction == 'both':
                    if r >= 0.0:
                        p, s, n = _apply_trapezoid_single(r, t_sp, t_ap, t_tbp)
                    else:
                        p, s, n = _apply_trapezoid_single(abs(r), t_sn, t_an, t_tbn)
                else:
                    p, s, n = _apply_trapezoid_single(abs(r), t_susp, t_atk, t_tb)
                P_w += p; S_w += s; N_w += n

            op_leaf = sl.evidence_to_opinion([P_w, S_w, N_w], W=W_BIJ, a=_prior_for_metric(key))
            if use_uncertainty_max:
                op_leaf = op_leaf.uncertainty_maximized()
            if wbf_weight_mode == "trust_discount":
                trust = float(models_pkg.get("trust_scores", {}).get(key, _prior_for_metric(key)[0]))
                op_leaf = sl.apply_trust_discount(op_leaf, trust)
                weight = 1.0
            else:
                weight = _calib_weight_for_metric(key)

            for group_name, keys in group_keys.items():
                if key in keys:
                    ops_by_group[group_name].append(op_leaf)
                    weights_by_group[group_name].append(weight)
                    break

        method_ops_for_fusion = []
        for group in method_groups:
            group_name = group["name"]
            group_ops = ops_by_group.get(group_name, [])
            if not group_ops:
                continue
            ext_w = None if wbf_weight_mode == "trust_discount" else weights_by_group[group_name]
            op_group = sl.fusion_wbf_n_sources(group_ops, external_weights=ext_w, W=W_BIJ)
            explicit_alpha = None
            if group.get("attack_discount_config_key") == "RECONST_ATTACK_RELIABILITY":
                explicit_alpha = models_pkg.get(
                    "reconst_attack_reliability",
                    CONFIG.get("RECONST_ATTACK_RELIABILITY", 1.0),
                )
            method_ops_for_fusion.append(
                fusion_policy.apply_method_discount(
                    op_group,
                    group,
                    CONFIG,
                    explicit_alpha=explicit_alpha,
                )
            )

        if not method_ops_for_fusion:
            continue

        op = fusion_policy.fuse_method_opinions(
            method_ops_for_fusion,
            mode=mode,
            W=W_BIJ,
            balance_ratio_eff=_effective_balance_ratio(),
        )

        # proj_atk = b_atk + a_atk × u  (Jøsang 2016, Eq. 3.23)
        proj_atk_list.append(float(op.projected_prob()[2]))

    return np.array(proj_atk_list) if proj_atk_list else None


def _effective_balance_ratio() -> float:
    """Return the numeric CBF balance ratio active for calibration."""
    raw = CONFIG.get("BALANCE_RATIO_EFF", CONFIG.get("BALANCE_RATIO", 1.0))
    if raw is None or str(raw).strip().lower() == "auto":
        return 1.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.0


def _normalise_fusion_mode_list() -> list[str]:
    """Modes for which train_models writes mode-specific threshold sidecars."""
    valid = {
        "wbf", "abf", "cbf", "bcf", "ccf", "minbf", "maxbf", "hierarchical"
    }
    requested = [str(CONFIG.get("INTER_METHOD_FUSION", "wbf")).strip().lower()]
    requested.extend(CONFIG.get("THRESHOLD_CALIBRATION_FUSION_MODES", []) or [])

    modes = []
    for mode in requested:
        mode = str(mode).strip().lower()
        if not mode or mode in modes:
            continue
        if mode not in valid:
            print(f"  [WARN] Mode de calibration seuil ignore: {mode!r}")
            continue
        modes.append(mode)
    return modes or ["wbf"]


def _calibrate_decision_threshold_from_scores(scores: np.ndarray,
                                              fpr_target: float,
                                              source_label: str,
                                              mode_label: str) -> dict:
    """Apply the historical SL-ADS threshold calibration rules to one score vector."""
    b_atk_train = np.asarray(scores, dtype=float)
    p50 = float(np.quantile(b_atk_train, 0.50))
    p99 = float(np.quantile(b_atk_train, 0.99))
    decision_thr = float(np.quantile(b_atk_train, 1.0 - fpr_target))
    calib_strategy = "quantile standard"

    W_BIJ = float(CONFIG.get('SL_PARAM_K', 3.0))
    bijection_floor = 1.0 / (WINDOW_SIZE + W_BIJ)
    floor_tol = float(CONFIG.get('CALIB_BIJECTION_FLOOR_TOL', 0.01))
    ageing_frac = float(CONFIG.get('CALIB_AGEING_WIN_FRACTION', 0.5))
    spars_cutoff = float(CONFIG.get('CALIB_SPARSITY_CUTOFF', 1e-9))
    floor_lo = bijection_floor * (1.0 - floor_tol)
    floor_hi = bijection_floor * (1.0 + floor_tol)

    print(f"  [{mode_label.upper()}] Distribution proj_atk ({source_label}) :")

    if floor_lo <= decision_thr <= floor_hi:
        lam = float(CONFIG.get('LAMBDA_DECAY', 0.85))
        decision_thr = bijection_floor * (1.0 - lam) * ageing_frac
        calib_strategy = (
            f"plancher bijection -> accumulation ageing "
            f"(floor={bijection_floor:.4f}, lambda={lam:.2f}, "
            f"frac={ageing_frac:.2f}, delta={decision_thr:.6f})"
        )
        print(f"    [CALIB-S1b] quantile FPR ~= plancher bijection "
              f"({bijection_floor:.4f}, tol=+/-{floor_tol*100:.1f}%).")

    if decision_thr < spars_cutoff:
        nonzero_vals = b_atk_train[b_atk_train > spars_cutoff]
        if len(nonzero_vals) > 0:
            if float(np.min(nonzero_vals)) >= floor_lo:
                lam = float(CONFIG.get('LAMBDA_DECAY', 0.85))
                decision_thr = bijection_floor * (1.0 - lam) * ageing_frac
                calib_strategy = (
                    f"plancher bijection via S2b -> delta={decision_thr:.6f}"
                )
                print(f"    [CALIB-S2b] min non-nul = plancher bijection "
                      f"-> delta={decision_thr:.6f}")
            else:
                decision_thr = float(np.min(nonzero_vals))
                calib_strategy = f"min non-nul (n={len(nonzero_vals)})"
                print(f"    [CALIB-S2] p{(1-fpr_target)*100:.0f}~=0 mais "
                      f"{len(nonzero_vals)} fenetres non-nulles -> delta={decision_thr:.6f}")
        else:
            b_atk_min = 1.0 / (WINDOW_SIZE + W_BIJ)
            decision_thr = b_atk_min * ageing_frac
            calib_strategy = (
                f"seuil theorique (b_atk_min={b_atk_min:.6f} "
                f"x{ageing_frac:.2f})"
            )
            print(f"    [CALIB-S3] Toutes fenetres proj_atk~=0 -> delta={decision_thr:.6f}")

    min_thr = float(CONFIG.get('MIN_DECISION_THRESHOLD', 0.0))
    if min_thr > 0 and decision_thr < min_thr:
        print(f"    [CALIB] Seuil releve au plancher MIN_DECISION_THRESHOLD={min_thr:.4f}")
        decision_thr = min_thr
        calib_strategy += f" + plancher {min_thr}"

    print(f"    p50={p50:.4f}  p99={p99:.4f}  "
          f"DECISION_THRESHOLD={decision_thr:.4f}  "
          f"(sur {len(b_atk_train)} fenetres, variable=proj_atk)")
    print(f"    Strategie : {calib_strategy}")

    return {
        "decision_threshold": decision_thr,
        "decision_variable": "proj_atk",
        "calib_strategy": calib_strategy,
        "p50": p50,
        "p99": p99,
        "n_windows": int(len(b_atk_train)),
        "nonzero": int(np.sum(b_atk_train > spars_cutoff)),
        "source": source_label,
        "fusion_mode_at_calibration": mode_label,
    }


# ==============================================================================
# SECTION 6 — UTILITAIRES DE DONNÉES
# ==============================================================================

def apply_exclusions(df: pd.DataFrame, metric_col: str = None) -> pd.DataFrame:
    """
    Marque NaN les valeurs de la colonne `metric_col` pendant les périodes
    d'exclusion définies dans CONFIG['TRAIN_EXCLUSIONS'].

    Prophet ignore nativement les NaN via dropna() sur 'y' (Taylor 2018).
    QuantileRegressor reçoit les données après dropna() sur les colonnes
    feature et target — les NaN sont donc également ignorés.
    """
    exclusions = CONFIG.get('TRAIN_EXCLUSIONS', [])
    if not exclusions:
        return df
    df_clean = df.copy()
    count_excluded = 0
    for exc in exclusions:
        try:
            start = pd.to_datetime(exc['start'])
            end   = pd.to_datetime(exc['end'])
            mask  = (df_clean['ds'] >= start) & (df_clean['ds'] <= end)
            nb_pts = mask.sum()
            if nb_pts > 0 and metric_col and metric_col in df_clean.columns:
                df_clean.loc[mask, metric_col] = np.nan
                count_excluded += nb_pts
        except Exception as e:
            print(f"⚠️  Erreur règle exclusion : {e}")
    if count_excluded > 0:
        print(f"   ℹ️  Exclusion appliquée : {count_excluded} points marqués NaN.")
    return df_clean


# ==============================================================================
# SECTION 7 — ENTRAÎNEMENT PRINCIPAL
# ==============================================================================

def train_models():
    print("--- SL-ADS Training (train_v10 — EDP + EVT thresholds + QR reconstruction) ---")

    if not os.path.exists(CONFIG['file_path']):
        print(f"❌ Fichier introuvable : {CONFIG['file_path']}")
        return

    # ── Chargement et nettoyage des données ──────────────────────────────────
    df = pd.read_csv(CONFIG['file_path'])
    df['ds'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('ds')

    # Pré-exclusion des périodes de maintenance AVANT toute imputation.
    # Objectif : éviter que ffill() propage des valeurs fictives dans les gaps,
    # qui seraient ensuite vues par Prophet comme du trafic normal légitime.
    # Les métriques reçoivent NaN pour ces périodes ; Prophet les ignore via
    # dropna() sur 'y' (Taylor & Letham 2018, comportement documenté).
    metric_cols = CONFIG.get('ACTIVE_METRICS', [])
    for exc in CONFIG.get('TRAIN_EXCLUSIONS', []):
        try:
            s = pd.to_datetime(exc['start'])
            e = pd.to_datetime(exc['end'])
            mask = (df['ds'] >= s) & (df['ds'] <= e)
            if mask.any() and metric_cols:
                df.loc[mask, metric_cols] = np.nan
        except Exception:
            pass

    # ffill() uniquement sur les colonnes non-métriques (on_weekend, on_weekday).
    # Les métriques reçoivent un ffill limité (limit=10 ≈ 5 min) uniquement
    # pour les micro-coupures non listées dans TRAIN_EXCLUSIONS.
    # fillna(0) est volontairement ABSENT pour les métriques : 0 bytes signifie
    # "panne réseau" et polluerait la baseline Prophet.
    non_metric_cols = [c for c in df.columns if c not in metric_cols and c != 'ds']
    df[non_metric_cols] = df[non_metric_cols].ffill().fillna(0)
    df = preprocess_metrics(df, limit_ffill=CONFIG.get("NAN_FFILL_LIMIT", 10))

    if df.duplicated(subset=['ds']).any():
        nb_dupes = df.duplicated(subset=['ds']).sum()
        print(f"⚠️  {nb_dupes} timestamps dupliqués — suppression (keep='first').")
        df = df.drop_duplicates(subset=['ds'], keep='first')

    # ── Jours fériés ─────────────────────────────────────────────────────────
    rio_holidays = CONFIG.get('HOLIDAYS_LIST', CONFIG.get('RIO_HOLIDAYS', []))
    holidays_df  = None
    if rio_holidays:
        print(f" -> Chargement de {len(rio_holidays)} jours fériés.")
        holidays_df = pd.DataFrame(rio_holidays)
        holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
        holidays_df = holidays_df.drop_duplicates(subset=['ds']).reset_index(drop=True)
        if 'lower_window' not in holidays_df.columns:
            holidays_df['lower_window'] = 0
        if 'upper_window' not in holidays_df.columns:
            holidays_df['upper_window'] = 0

    is_cal_wknd = df['ds'].dt.dayofweek >= 5
    if holidays_df is not None:
        is_holiday_mask = df['ds'].dt.normalize().isin(holidays_df['ds'].dt.normalize())
        df['on_weekend'] = (is_cal_wknd | is_holiday_mask).astype(int)
    else:
        df['on_weekend'] = is_cal_wknd.astype(int)
    df['on_weekday'] = (1 - df['on_weekend']).astype(int)

    # ── Split train / calibration ────────────────────────────────────────────
    split_date = pd.to_datetime(CONFIG['split_date'])
    df_train   = df[df['ds'] <= split_date].copy()

    # Résolution ACTIVE_METRICS = "auto" (datasets inconnus, ex : CESNET)
    if CONFIG.get('ACTIVE_METRICS') == 'auto':
        ignore_cols  = {'timestamp', 'ds', 'label', 'on_weekend', 'on_weekday'}
        auto_metrics = [c for c in df_train.columns
                        if c not in ignore_cols and not c.startswith('Unnamed')]
        CONFIG['ACTIVE_METRICS'] = auto_metrics
        print(f"  -> ACTIVE_METRICS='auto' résolu : {auto_metrics}")

    # Sous-split temporel : les modèles sont entraînés sur les (1−f) premières
    # fractions, le DECISION_THRESHOLD est calibré sur les f dernières.
    # Cela évite un biais optimiste (résidus trop petits car le modèle s'est
    # ajusté sur ces mêmes données) — ref : Ruff et al. (2021).
    calib_fraction  = float(CONFIG.get("CALIB_SPLIT_FRACTION",0.0))
    df_train_sorted = df_train.sort_values('ds').reset_index(drop=True)
    n_model         = max(10, int(len(df_train_sorted) * (1.0 - calib_fraction)))
    df_train_model  = df_train_sorted.iloc[:n_model].copy()
    df_train_calib  = df_train_sorted.iloc[n_model:].copy()

    print(f"Total : {len(df)} | Train total : {len(df_train)} | Cutoff : {split_date}")
    print(f"  [model] Prophet/QR(q=0.5) + EVT thresholds : {len(df_train_model)} pts "
          f"({df_train_model['ds'].iloc[0].date()} -> {df_train_model['ds'].iloc[-1].date()})")
    if len(df_train_calib) > 0:
        print(f"  [calib] DECISION_THRESHOLD : {len(df_train_calib)} pts "
              f"({df_train_calib['ds'].iloc[0].date()} -> {df_train_calib['ds'].iloc[-1].date()})")
    else:
        print("  [calib] DECISION_THRESHOLD : vide — fallback sur données modèles.")

    if len(df_train_model) == 0:
        print("❌ Dataset d'entraînement (modèles) vide !"); return

    pp         = CONFIG.get('PROPHET_PARAMS', {})
    _ensure_prophet_tbb_on_path()
    models_pkg = {}
    _train_signed_residuals = {}  # résidus sur df_train_model → EDP + seuils
    _calib_signed_residuals = {}  # résidus sur df_train_calib → DECISION_THRESHOLD
    _training_failures = []

    metrics_prophet = CONFIG['ACTIVE_METRICS']
    metrics_reconst = [
        f"reconst_{r['target']}_from_{r['feature']}"
        for r in CONFIG['RECONST_RULES'] if r.get('target')
    ]
    metrics_list = metrics_prophet + metrics_reconst
    expected_model_keys = [f"prophet_{m}" for m in metrics_prophet] + metrics_reconst
    print(f"\nEntraînement de {len(metrics_list)} modèles...")

    for metric in metrics_list:

        # ── CAS 1 : RECONSTRUCTION par QuantileRegressor(q=0.5) ──────────────
        #
        # Choix de l'estimateur : QuantileRegressor(q=0.5) = régression médiane
        # (LAD, Koenker & Bassett 1978, Econometrica).
        #
        # Justification vs RANSAC :
        #   Le set d'entraînement ne contient aucune attaque (split_date précède
        #   toutes les injections). RANSAC est conçu pour rejeter des observations
        #   anormales contaminant l'estimation — sur données purement normales, il
        #   rejette arbitrairement des pics de trafic légitime selon son initialisation
        #   aléatoire, ce qui cause une variance de seed de 22–75% sur T_susp.
        #   QR(q=0.5) utilise TOUS les points propres avec une loss L1 robuste :
        #     • Déterministe (pas de seed) → reproducibilité parfaite
        #     • LAD (QR q=0.5) est robuste aux outliers de RÉPONSE (Koenker &
        #       Bassett 1978 Econometrica 46:33-50) — breakdown point jusqu'à 50 %
        #       pour les outliers verticaux. N.B. : LAD n'est PAS robuste aux
        #       outliers de LEVIER (leverage) — breakdown 0 % (Rousseeuw & Leroy
        #       1987, Robust Regression §3.3). RANSAC est complémentaire pour
        #       les leverage outliers. Notre usage (bytes ← packets avec
        #       fit_intercept=False, Bridgman 1922) évite les leverage outliers
        #       par contrainte physique.
        #     • Généralise sans intervention à tout nouveau dataset
        #
        # Contrainte physique (Bridgman 1922) :
        #   Pour bytes←packets, 0 paquet implique 0 octet → fit_intercept=False.
        #   Configuré dans RECONST_RULES['fit_intercept'] par paire.
        #
        # Mean fallback (Hyndman & Athanasopoulos 2021, §3.1) :
        #   Si R² < 0 (modèle linéaire moins bon que la moyenne) ET
        #   allow_mean_fallback=True → DummyRegressor(mean).
        #   Capture les décrochages d'attaque (ex : fin→0 lors d'un SYN Flood
        #   alors que ȳ_fin >> 0) sans FPR structurel d'un mauvais fit linéaire.
        # ─────────────────────────────────────────────────────────────────────
        if 'reconst' in metric:
            try:
                parts   = metric.replace('reconst_', '').split('_from_')
                target  = parts[0]
                feature = parts[1]
            except Exception:
                print(f"Erreur format règle : {metric}")
                _training_failures.append({
                    "branch": "reconstruction",
                    "metric": metric,
                    "expected_key": metric,
                    "reason": "invalid reconstruction rule format",
                })
                continue

            if feature not in df_train_model.columns or target not in df_train_model.columns:
                print(f"⚠️  Colonnes manquantes : {metric}")
                _training_failures.append({
                    "branch": "reconstruction",
                    "metric": metric,
                    "expected_key": metric,
                    "reason": f"missing columns target={target!r} feature={feature!r}",
                })
                continue

            _q_level = float(CONFIG.get("RECONST_QUANTILE", 0.5))
            sys.stdout.write(f" -> QR(q={_q_level}): {target} from {feature} ... ")
            sys.stdout.flush()

            temp_df     = apply_exclusions(df_train_model.copy(), metric_col=target)
            temp_df     = apply_exclusions(temp_df,               metric_col=feature)
            train_clean = temp_df.dropna(subset=[feature, target])
            X = train_clean[[feature]].values
            y = train_clean[target].values

            if len(X) < 50:
                print("⚠️  Pas assez de données. Skip.")
                _training_failures.append({
                    "branch": "reconstruction",
                    "metric": metric,
                    "expected_key": metric,
                    "reason": f"not enough data after exclusions (n={len(X)})",
                })
                continue

            fit_intercept_flag = next(
                (r.get('fit_intercept', True)
                 for r in CONFIG['RECONST_RULES']
                 if r.get('target') == target and r.get('feature') == feature),
                True
            )

            reg = QuantileRegressor(
                quantile=_q_level,
                fit_intercept=fit_intercept_flag,
                alpha=0.0,       # alpha=0 : pas de régularisation L1 (LAD pur)
                solver='highs'   # solveur LP le plus rapide disponible
            )
            # PATCH-M1 (2026-04-18) : Time-series CV R² remplace l'in-sample R².
            # Motif : l'in-sample R² est optimistement biaisé (Stone 1974 JRSS B ;
            # Hastie-Tibshirani-Friedman 2009 §7.10). Utilisé comme trust_score en WBF,
            # il inflate les poids des sources sur-fittées.
            from sklearn.model_selection import TimeSeriesSplit

            if len(X) >= 50:  # minimum pour 5-fold TS split
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = []
                for tr_idx, val_idx in tscv.split(X):
                    _reg_cv = type(reg)(**reg.get_params())
                    _reg_cv.fit(X[tr_idx], y[tr_idx])
                    cv_scores.append(r2_score(y[val_idx], _reg_cv.predict(X[val_idx])))
                r2 = float(np.mean(cv_scores))  # R² CV-averaged
                r2_std = float(np.std(cv_scores))
            else:
                # Fallback pour métriques rares : in-sample avec warning explicite
                import warnings
                warnings.warn(
                    f"{metric}: n={len(X)} < 50, CV impossible. In-sample R² retained — "
                    f"trust_score biaisé (voir PATCH-M1).", UserWarning
                )
                reg.fit(X, y)
                r2 = r2_score(y, reg.predict(X))
                r2_std = float('nan')

            # Re-fit final sur toutes les données (après CV) pour persister le modèle
            reg.fit(X, y)
            preds = reg.predict(X)  # pour diagnostics plots only

            # Logger les deux valeurs pour traçabilité
            print(f"  {metric}: R²_CV = {r2:+.3f} ± {r2_std:.3f} "
                  f"(in-sample = {r2_score(y, preds):+.3f})")
            res_signed = y - preds

            # Mean fallback si R² < 0 et autorisé dans RECONST_RULES
            _allow_mean_fb = next(
                (r.get('allow_mean_fallback', False)
                 for r in CONFIG['RECONST_RULES']
                 if r.get('target') == target and r.get('feature') == feature),
                False
            )
            if r2 < 0.0 and _allow_mean_fb:
                print(f"\n  [MEAN FALLBACK] {metric}: R²={r2:.3f} < 0 "
                      f"→ remplacement par DummyRegressor(strategy='mean')")
                # PATCH m-08/F28 — Log the reconstruction fallback to the
                # audit trail.  r2 (negative) is captured BEFORE it is
                # overwritten by 0.0 below so we retain the original value.
                _record_fallback("reconstruction_dummy",
                                 metric_key=metric, branch="ransac",
                                 feature=feature, target=target,
                                 r2_in_sample=float(r2),
                                 strategy="mean")
                _mean_model = DummyRegressor(strategy="mean")
                _mean_model.fit(X, y)
                preds      = _mean_model.predict(X)
                res_signed = y - preds
                reg        = _mean_model
                r2         = 0.0   # R²_mean = 0 par définition

            short_key = f"{target}_{feature}"

            # PATCH TASK-22 (audit_tmp MAJ-01, 2026-04-26)
            # ──────────────────────────────────────────────────────────────
            # Calibration leak-free : si un sous-split temporel de calibration
            # ``df_train_calib`` est disponible, on calibre t_susp/t_atk sur
            # les résidus *out-of-sample* de ce split (le modèle n'a jamais
            # vu ces données — résidus non biaisés).  Sinon (mode legacy
            # ``CALIB_SPLIT_FRACTION=0``), on retombe sur les résidus
            # in-sample avec un warning explicite.
            #
            # Réf : Ruff et al. 2021 (Comprehensive Review of Anomaly
            # Detection §4.4) ; Stone 1974 JRSS B ; HTF 2009 §7.10.
            _res_for_calib = None
            _calib_ts      = None       # PATCH H2 — aligned timestamps for per-regime calibration
            _calib_source  = "in-sample (legacy)"
            if len(df_train_calib) > 0:
                _calib_tmp_pre   = apply_exclusions(df_train_calib.copy(), metric_col=target)
                _calib_tmp_pre   = apply_exclusions(_calib_tmp_pre,        metric_col=feature)
                _calib_clean_pre = _calib_tmp_pre.dropna(subset=[feature, target])
                if len(_calib_clean_pre) >= 30:    # n minimum pour quantile EVT/MAD
                    X_cp = _calib_clean_pre[[feature]].values
                    y_cp = _calib_clean_pre[target].values
                    _res_for_calib = y_cp - reg.predict(X_cp)
                    if "ds" in _calib_clean_pre.columns:
                        _calib_ts = _calib_clean_pre["ds"].values
                    _calib_source  = f"calib hors-train (n={len(_calib_clean_pre)})"
            if _res_for_calib is None or len(_res_for_calib) < 30:
                if len(df_train_calib) > 0:
                    import warnings as _w22
                    _w22.warn(
                        f"{metric}: calibration set too small "
                        f"({len(df_train_calib)} pts) — falling back to "
                        f"in-sample residuals for t_susp/t_atk (biased low).",
                        UserWarning, stacklevel=2,
                    )
                _res_for_calib = res_signed
                _calib_ts      = None    # In-sample fallback: regime info unavailable

            # Les seuils sont calibrés sur les résidus de calibration
            # (out-of-sample si possible) — pas d'exclusion inlier/outlier
            # car compute_evidence_v2 évalue toutes les fenêtres de production.
            thresh = calibrate_thresholds_v2(_res_for_calib, metric_key=short_key, branch='ransac')

            # PATCH H2 — Per-regime calibration is OPT-IN via
            # CONFIG['CALENDAR_EVT_ENABLED'] (default False for backward
            # compatibility).  When enabled and timestamps are aligned with
            # the residuals, we partition by regime (ACTIVE / QUIET) and
            # re-run the EVT calibration per bucket.  The legacy scalar
            # fields ``t_susp``, ``t_atk``, ``t_trapeze_base`` are kept as
            # a fallback for old consumers that do not understand the
            # per-regime block.
            _per_regime_block = None
            if CONFIG.get("CALENDAR_EVT_ENABLED", False) and _calib_ts is not None:
                _per_regime_block = calibrate_thresholds_per_regime_v2(
                    np.asarray(_res_for_calib, dtype=float),
                    _calib_ts,
                    metric_key=short_key,
                    branch="ransac",
                    holidays=CONFIG.get("HOLIDAYS_LIST"),
                )
            print(f"    [calib-source] {metric}: {_calib_source}")

            # MAD asymétrique (traçabilité — non utilisé dans la détection)
            mad_global = np.median(np.abs(res_signed - np.median(res_signed)))
            if mad_global == 0:
                mad_global = 1e-9
            mask_pos = res_signed >= 0
            mask_neg = res_signed < 0
            mad_sup  = (np.median(np.abs(res_signed[mask_pos] - np.median(res_signed)))
                        if mask_pos.any() else mad_global)
            mad_inf  = (np.median(np.abs(res_signed[mask_neg] - np.median(res_signed)))
                        if mask_neg.any() else mad_global)

            models_pkg[metric] = {
                'model':             reg,
                'type':              'reconstruction',
                't_susp':            thresh['t_susp'],
                't_atk':             thresh['t_atk'],
                't_trapeze_base':    thresh['t_trapeze_base'],
                't_susp_pos':        thresh.get('t_susp_pos', thresh['t_susp']),
                't_atk_pos':         thresh.get('t_atk_pos',  thresh['t_atk']),
                't_trapeze_base_pos': thresh.get('t_trapeze_base_pos', thresh['t_trapeze_base']),
                't_susp_neg':        thresh.get('t_susp_neg', thresh['t_susp']),
                't_atk_neg':         thresh.get('t_atk_neg',  thresh['t_atk']),
                't_trapeze_base_neg': thresh.get('t_trapeze_base_neg', thresh['t_trapeze_base']),
                'direction':         thresh['direction'],
                # Format rétrocompatible avec les scripts d'évaluation existants
                'thresholds':        {'suspect': thresh['t_susp'], 'attack': thresh['t_atk']},
                'r2_score':          r2,
                # PATCH D5 (MASE trust mode) : MASE n'est pas défini pour des
                # régressions non-temporelles bytes ~ packets ; on persiste NaN
                # afin que ``mase_to_trust`` retombe sur ``TRUST_SCORE_FLOOR``
                # pour les métriques RANSAC.  Le contextual discounting
                # ``RECONST_ATTACK_RELIABILITY`` (Mercier-Quost-Denoeux 2008)
                # gère séparément la fiabilité de la branche reconstruction.
                'mase_score':        float('nan'),
                'mad':               mad_global,
                'mad_sup':           mad_sup,
                'mad_inf':           mad_inf,
                'kurtosis':          thresh['kurtosis'],
                'cv':                thresh['cv'],
                # PATCH H2 — per-regime block (None when calendar-aware EVT
                # is disabled).  See ``calibrate_thresholds_per_regime_v2``.
                'thresholds_per_regime': _per_regime_block,
            }
            _train_signed_residuals[metric] = res_signed

            # Résidus sur le set de calibration (→ DECISION_THRESHOLD)
            if len(df_train_calib) > 0:
                calib_tmp   = apply_exclusions(df_train_calib.copy(), metric_col=target)
                calib_tmp   = apply_exclusions(calib_tmp,             metric_col=feature)
                calib_clean = calib_tmp.dropna(subset=[feature, target])
                if len(calib_clean) >= 1:
                    X_calib = calib_clean[[feature]].values
                    y_calib = calib_clean[target].values
                    _calib_signed_residuals[metric] = y_calib - reg.predict(X_calib)

            print(f"OK. (R²={r2:.3f} | dir={thresh['direction']} | "
                  f"T_susp={thresh['t_susp']:.2f} | T_atk={thresh['t_atk']:.2f})")

        # ── CAS 2 : PRÉVISION PAR PROPHET ────────────────────────────────────
        #
        # Prophet décompose la série en tendance + saisonnalités + résidus
        # (Taylor & Letham 2018). Trois niveaux de saisonnalité :
        #   - weekly_seasonality : cycle hebdomadaire standard
        #   - daily_weekday / daily_weekend : saisonnalité journalière
        #     conditionnelle au type de jour (profil différent en semaine vs
        #     week-end / jours fériés)
        #   - hourly : variations infra-horaires (period=1/24 jour)
        #
        # Mode saisonnalité :
        #   Métriques dans SEASONALITY_ADDITIVE → mode additif.
        #   Requis pour les séries avec valeurs proches de zéro (bytes, flows,
        #   icmp, etc.) car le mode multiplicatif implique y_t = trend × season
        #   et génère des instabilités numériques quand y_t ≈ 0.
        #   Ref : documentation officielle Prophet (Taylor & Letham 2018).
        #
        # growth='flat' : pas de tendance longue. Le réseau académique RedeRio
        # a une capacité fixe sur l'horizon d'entraînement de 4 semaines.
        # ─────────────────────────────────────────────────────────────────────
        else:
            target_col = metric
            if target_col not in df_train_model.columns:
                print(f"⚠️  Colonne manquante : {target_col}")
                _training_failures.append({
                    "branch": "prophet",
                    "metric": metric,
                    "expected_key": f"prophet_{metric}",
                    "reason": f"missing column {target_col!r}",
                })
                continue

            sys.stdout.write(f" -> Prophet: {target_col} ... ")
            sys.stdout.flush()

            try:
                df_metric  = df_train_model[['ds', target_col, 'on_weekend', 'on_weekday']].copy()
                df_metric  = apply_exclusions(df_metric, metric_col=target_col)
                df_prophet = df_metric.rename(columns={target_col: 'y'}).reset_index(drop=True)

                seasonality_mode = ('additive'
                                    if metric in CONFIG.get('SEASONALITY_ADDITIVE', [])
                                    else 'multiplicative')

                model = Prophet(
                    growth='flat',
                    interval_width=pp.get('interval_width', 0.95),
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    seasonality_mode=seasonality_mode,
                    changepoint_prior_scale=pp.get('changepoint_prior_scale', 0.05),
                    holidays_prior_scale=pp.get('holidays_prior_scale', 10.0),
                    seasonality_prior_scale=pp.get('seasonality_prior_scale', 10.0),
                    holidays=holidays_df,
                    stan_backend=CONFIG.get("PROPHET_STAN_BACKEND", "CMDSTANPY"),
                )
                model.add_seasonality('daily_weekday', period=1, fourier_order=12,
                                      condition_name='on_weekday')
                model.add_seasonality('daily_weekend', period=1, fourier_order=12,
                                      condition_name='on_weekend')
                model.add_seasonality('hourly', period=1/24, fourier_order=5)
                model.fit(df_prophet)

                fcst       = model.predict(df_prophet)
                mask_valid = ~np.isnan(df_prophet['y'])
                y_true     = df_prophet.loc[mask_valid, 'y'].values
                # PATCH-M1 bis (2026-04-18) : Prophet CV R² via time-series rolling-origin.
                # Les résidus in-sample sont calculés de toute façon (pour
                # la calibration des seuils), la CV ne concerne que la
                # métrique R² rapportée.
                from prophet.diagnostics import cross_validation, performance_metrics

                # Prédictions in-sample (servent aux résidus dans tous les cas)
                fcst   = model.predict(df_prophet)
                y_pred = fcst.loc[mask_valid, 'yhat'].values

                try:
                    df_cv = cross_validation(
                        model, initial='14 days', period='3 days', horizon='1 day',
                        parallel=None, disable_tqdm=True
                    )
                    df_perf = performance_metrics(df_cv, rolling_window=1.0)
                    r2 = 1.0 - (df_perf['mse'].iloc[0] / np.var(y_true))
                    print(f"  {metric}: Prophet R²_CV = {r2:+.3f} (rolling-origin)")
                except Exception as _e:
                    # Fallback in-sample si CV impossible (ex. historique trop court)
                    import warnings
                    warnings.warn(f"{metric}: Prophet CV failed ({_e}) — in-sample R² retained.")
                    r2 = r2_score(y_true, y_pred)
                res_signed = y_true - y_pred

                # PATCH D5 (MASE trust mode) — Hyndman-Koehler 2006 Eq. 4.
                # MASE est défini pour des séries temporelles (one-step Naive-1
                # baseline).  La série y_true et y_pred sont alignées sur le
                # train span ; le dénominateur de Naive-1 est calculé sur y_true
                # uniquement.  Voir ``sl_ads.stats.mase`` pour la définition
                # détaillée et la justification du choix par rapport à R².
                _mase_value = compute_mase(y_true, y_pred)
                if math.isfinite(_mase_value):
                    print(f"  {metric}: Prophet MASE = {_mase_value:.3f} "
                          f"(< 1 informative; > 1 worse than Naive-1)")
                else:
                    print(f"  {metric}: Prophet MASE = NaN (constant series or "
                          f"insufficient data — trust will fall back to floor)")

                thresh          = calibrate_thresholds_v2(res_signed,
                                                           metric_key=target_col,
                                                           branch='prophet')
                metric_pkg_key  = f"prophet_{target_col}"

                # PATCH H2 — per-regime calibration when CALENDAR_EVT_ENABLED.
                # Prophet residuals are aligned with ``df_prophet``; we use
                # ``df_prophet.loc[mask_valid, 'ds']`` as the timestamp index
                # to keep residual/timestamp alignment identical to the
                # global residual computation above.
                _per_regime_block_p = None
                if CONFIG.get("CALENDAR_EVT_ENABLED", False):
                    _ts_for_calib = df_prophet.loc[mask_valid, "ds"].values
                    _per_regime_block_p = calibrate_thresholds_per_regime_v2(
                        np.asarray(res_signed, dtype=float),
                        _ts_for_calib,
                        metric_key=target_col,
                        branch="prophet",
                        holidays=CONFIG.get("HOLIDAYS_LIST"),
                    )

                models_pkg[metric_pkg_key] = {
                    'model':             model,
                    'type':              'prophet',
                    't_susp':            thresh['t_susp'],
                    't_atk':             thresh['t_atk'],
                    't_trapeze_base':    thresh['t_trapeze_base'],
                    't_susp_pos':        thresh.get('t_susp_pos', thresh['t_susp']),
                    't_atk_pos':         thresh.get('t_atk_pos',  thresh['t_atk']),
                    't_trapeze_base_pos': thresh.get('t_trapeze_base_pos', thresh['t_trapeze_base']),
                    't_susp_neg':        thresh.get('t_susp_neg', thresh['t_susp']),
                    't_atk_neg':         thresh.get('t_atk_neg',  thresh['t_atk']),
                    't_trapeze_base_neg': thresh.get('t_trapeze_base_neg', thresh['t_trapeze_base']),
                    'direction':         thresh['direction'],
                    'thresholds':        {'suspect': thresh['t_susp'], 'attack': thresh['t_atk']},
                    'r2_score':          r2,
                    # PATCH D5 (MASE trust mode) : raw MASE persisté pour audit.
                    # Le top-level ``mase_scores`` dict (calculé plus bas) dérive
                    # le trust effectif via ``mase_to_trust`` (Joesang Def. 14.6).
                    'mase_score':        float(_mase_value),
                    'kurtosis':          thresh['kurtosis'],
                    'cv':                thresh['cv'],
                    # PATCH H2 — per-regime block (None when calendar-aware
                    # EVT is disabled).  See
                    # ``calibrate_thresholds_per_regime_v2``.
                    'thresholds_per_regime': _per_regime_block_p,
                }
                _train_signed_residuals[metric_pkg_key] = res_signed

                # Résidus sur le set de calibration (→ DECISION_THRESHOLD)
                if len(df_train_calib) > 0:
                    df_calib_m = df_train_calib[
                        ['ds', target_col, 'on_weekend', 'on_weekday']
                    ].copy()
                    df_calib_m = apply_exclusions(df_calib_m, metric_col=target_col)
                    df_calib_p = df_calib_m.rename(
                        columns={target_col: 'y'}
                    ).reset_index(drop=True)
                    fcst_calib = model.predict(df_calib_p)
                    mask_calib = ~np.isnan(df_calib_p['y'])
                    if mask_calib.any():
                        y_c_true = df_calib_p.loc[mask_calib, 'y'].values
                        y_c_pred = fcst_calib.loc[mask_calib, 'yhat'].values
                        _calib_signed_residuals[metric_pkg_key] = y_c_true - y_c_pred

                print(f"OK. (R²={r2:.3f} | dir={thresh['direction']} | "
                      f"T_susp={thresh['t_susp']:.2f} | T_atk={thresh['t_atk']:.2f})")

            except Exception as e:
                print(f" ❌ Erreur Prophet sur {metric}: {e}")
                import traceback as _traceback
                _traceback.print_exc()
                _training_failures.append({
                    "branch": "prophet",
                    "metric": metric,
                    "expected_key": f"prophet_{metric}",
                    "error": repr(e),
                })
                if CONFIG.get("REQUIRE_ALL_MODELS", True):
                    raise RuntimeError(
                        f"Required Prophet model failed for metric {metric!r}; "
                        "aborting full training run before producing a partial "
                        "artifact."
                    ) from e

    # ── Empirical Dirichlet Prior (EDP) ──────────────────────────────────────
    models_pkg['_training_failures'] = _training_failures
    validate_required_model_coverage(
        models_pkg,
        CONFIG,
        expected_keys=expected_model_keys,
        failures=_training_failures,
    )

    empirical_priors = {}
    if CONFIG.get("USE_EMPIRICAL_PRIOR", True):
        print("\n--- Calcul de l'Empirical Dirichlet Prior (EDP) ---")
        prior_floor = CONFIG.get("EMPIRICAL_PRIOR_FLOOR", 0.005)
        for key in models_pkg:
            if not isinstance(models_pkg[key], dict):
                continue
            if key not in _train_signed_residuals:
                continue
            res = _train_signed_residuals[key]
            thresh_pkg = {
                'direction': models_pkg[key]['direction'],
                't_susp': models_pkg[key]['t_susp'],
                't_atk': models_pkg[key]['t_atk'],
                't_trapeze_base': models_pkg[key]['t_trapeze_base'],
                # Seuils directionnels pos/neg — nécessaires pour le frame 5-états
                # (Jøsang §3.5.4 coarsening). Fallback sur ternaire si absent.
                't_susp_pos': models_pkg[key].get('t_susp_pos',
                                                  models_pkg[key]['t_susp']),
                't_atk_pos': models_pkg[key].get('t_atk_pos',
                                                 models_pkg[key]['t_atk']),
                't_trapeze_base_pos': models_pkg[key].get('t_trapeze_base_pos',
                                                          models_pkg[key]['t_trapeze_base']),
                't_susp_neg': models_pkg[key].get('t_susp_neg',
                                                  models_pkg[key]['t_susp']),
                't_atk_neg': models_pkg[key].get('t_atk_neg',
                                                 models_pkg[key]['t_atk']),
                't_trapeze_base_neg': models_pkg[key].get('t_trapeze_base_neg',
                                                          models_pkg[key]['t_trapeze_base']),
            }
            edp = compute_edp_from_residuals(res, thresh_pkg,
                                             window_size=WINDOW_SIZE,
                                             prior_floor=prior_floor)
            empirical_priors[key] = edp
            _dir = models_pkg[key]['direction']
            if _dir == 'both' and 'a_atk_pos' in edp:
                print(f"  {key:<45} a_safe={edp['a_safe']:.4f}  "
                      f"a_susp={edp['a_susp']:.4f}  a_atk={edp['a_atk']:.4f}  "
                      f"[pos: s={edp['a_susp_pos']:.4f} a={edp['a_atk_pos']:.4f} | "
                      f"neg: s={edp['a_susp_neg']:.4f} a={edp['a_atk_neg']:.4f}]")
            else:
                print(f"  {key:<45} a_safe={edp['a_safe']:.4f}  "
                      f"a_susp={edp['a_susp']:.4f}  a_atk={edp['a_atk']:.4f}")
        models_pkg['empirical_priors'] = empirical_priors
        print(f"  -> {len(empirical_priors)} priors calculés et stockés dans l'artefact.")
    else:
        models_pkg['empirical_priors'] = None
        print("  -> USE_EMPIRICAL_PRIOR=False : prior uniforme (1/3, 1/3, 1/3) conservé.")

    # ── Auto-calibration de RECONST_ATTACK_RELIABILITY ──────────────────────
    _raw_cd_cfg = CONFIG.get('RECONST_ATTACK_RELIABILITY', 1.0)
    if str(_raw_cd_cfg).strip().lower() == 'auto':
        print("\n--- Auto-calibration de RECONST_ATTACK_RELIABILITY (mode 'auto') ---")
        _thr_res_cd = (_calib_signed_residuals if _calib_signed_residuals
                       else _train_signed_residuals)
        _src_cd = "calib hors-train" if _calib_signed_residuals else "train (fallback)"
        print(f"  Source résidus : {_src_cd}")
        auto_alpha = _auto_calibrate_reconst_reliability(
            models_pkg, _thr_res_cd, WINDOW_SIZE
        )
        models_pkg['reconst_attack_reliability'] = auto_alpha
        print(f"  -> reconst_attack_reliability = {auto_alpha:.3f}")
    else:
        try:
            _fixed_alpha = float(_raw_cd_cfg)
        except (ValueError, TypeError):
            _fixed_alpha = 1.0
        models_pkg['reconst_attack_reliability'] = _fixed_alpha
        print(f"\n  RECONST_ATTACK_RELIABILITY fixe = {_fixed_alpha:.3f} (depuis config.py)")

    # ── Récapitulatif R² et seuils ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'MODÈLE':<45} | {'R²':<8} | {'T_susp':<10} | {'T_atk':<10}")
    print("=" * 60)
    for key, pkg in sorted(models_pkg.items()):
        if not isinstance(pkg, dict):
            continue
        if key in ('empirical_priors', 'trust_scores', 'mase_scores'):
            continue
        r2_val = pkg.get('r2_score', 'N/A')
        t_s    = pkg.get('t_susp', 0.0)
        t_a    = pkg.get('t_atk',  0.0)
        if isinstance(r2_val, float):
            print(f"{key:<45} | {r2_val:<8.3f} | {t_s:<10.3f} | {t_a:<10.3f}")
        else:
            print(f"{key:<45} | {r2_val:<8} | {t_s:<10.3f} | {t_a:<10.3f}")
    print("=" * 60)

    # ── Validation EVT sur set de calibration ────────────────────────────────
    # Vérifie que les seuils EVT calibrés sur 75% du train respectent les
    # probabilités d'excès cibles sur les 25% holdout (Siffer et al. 2017 §4.2).
    if len(df_train_calib) > 0 and _calib_signed_residuals:
        print("\n--- Validation EVT sur set de calibration (holdout) ---")
        print(f"  {'Métrique':<45} | {'FPR_susp_emp':>12} | {'FPR_atk_emp':>11} | "
              f"{'FPR_susp_cible':>14} | {'FPR_atk_cible':>13} | {'Statut':>8}")
        print("  " + "-" * 110)

        n_violations_susp = n_violations_atk = n_checked = 0

        for key, pkg in sorted(models_pkg.items()):
            if not isinstance(pkg, dict) or key in ('empirical_priors', 'trust_scores', 'mase_scores'):
                continue
            if key not in _calib_signed_residuals:
                continue
            calib_res = _calib_signed_residuals[key]
            if len(calib_res) < 10:
                continue

            t_susp    = pkg.get('t_susp', 1e9)
            t_atk     = pkg.get('t_atk',  1e9)
            direction = pkg.get('direction', 'sym')

            if direction == 'pos':
                excess = calib_res[calib_res > 0]
            elif direction == 'neg':
                excess = np.abs(calib_res[calib_res < 0])
            elif direction == 'both':
                excess = np.concatenate([
                    calib_res[calib_res >= 0],
                    np.abs(calib_res[calib_res < 0])
                ])
            else:
                excess = np.abs(calib_res)

            n_total = len(excess)
            if n_total == 0:
                continue

            fpr_susp_emp = float(np.sum(excess > t_susp)) / n_total
            fpr_atk_emp  = float(np.sum(excess > t_atk))  / n_total

            branch = pkg.get('type', 'prophet')
            if branch == 'prophet':
                q_susp_tgt = CONFIG.get("EVT_Q_SUSP_PROPHET", 0.02)
                q_atk_tgt  = CONFIG.get("EVT_Q_ATK_PROPHET",  0.01)
            else:
                q_susp_tgt = CONFIG.get("EVT_Q_SUSP_RANSAC", 0.01)
                q_atk_tgt  = CONFIG.get("EVT_Q_ATK_RANSAC",  0.001)

            # Tolérance : 2× la cible (extrapolation EVT naturellement imprécise)
            ok_susp = fpr_susp_emp <= q_susp_tgt * 2.0
            ok_atk  = fpr_atk_emp  <= q_atk_tgt  * 2.0
            statut  = "✅ OK" if (ok_susp and ok_atk) else "⚠️  HAUT"

            if not ok_susp: n_violations_susp += 1
            if not ok_atk:  n_violations_atk  += 1
            n_checked += 1

            print(f"  {key:<45} | {fpr_susp_emp:>12.4f} | {fpr_atk_emp:>11.5f} | "
                  f"{q_susp_tgt:>14.4f} | {q_atk_tgt:>13.4f} | {statut:>8}")

        print("  " + "-" * 110)
        print(f"  Résumé : {n_checked} métriques | "
              f"Violations T_susp: {n_violations_susp} | Violations T_atk: {n_violations_atk}")
        if n_violations_atk > n_checked * 0.3:
            print("  ⚠️  >30% des métriques dépassent 2×FPR_cible sur T_atk.")
        else:
            print("  ✅  Seuils EVT validés sur holdout — calibration cohérente.")
    else:
        print("\n  [EVT VALIDATION] Set de calibration vide — validation ignorée.")

    # ── Auto-calibration de DECISION_THRESHOLD ────────────────────────────────
    # PATCH-m3 (2026-04-18) : fallback dangereux 1.0 supprimé.
    # Motif : si la clé config est absente, fallback à 1.0 → quantile(..., 0) =
    # min(proj_atk) → threshold = minimum absolu → 100 % positive rate.
    # Solution : échec explicite au lieu de comportement silencieusement désastreux.
    if "FPR_TARGET_DECISION" not in CONFIG:
        raise KeyError(
            "FPR_TARGET_DECISION manquant dans CONFIG — configurer explicitement "
            "(valeur recommandée : 0.001 pour RedeRio, 0.01 pour METR-LA)."
        )
    fpr_target = float(CONFIG["FPR_TARGET_DECISION"])
    assert 0 < fpr_target < 1, (
        f"FPR_TARGET_DECISION={fpr_target} doit être strictement dans (0, 1)"
    )
    _thr_residuals = (_calib_signed_residuals if _calib_signed_residuals
                      else _train_signed_residuals)
    _thr_source    = "calib hors-train" if _calib_signed_residuals else "train (fallback)"
    print(f"\n--- Auto-calibration de DECISION_THRESHOLD "
          f"(FPR cible = {fpr_target*100:.1f}%, source = {_thr_source}) ---")

    _threshold_calibration_scores = {}
    _threshold_calibration_results = {}
    _calib_modes = _normalise_fusion_mode_list()
    print(f"  Modes de calibration seuil : {_calib_modes}")

    for _mode in _calib_modes:
        _scores = _compute_training_proj_atk(
            models_pkg,
            _thr_residuals,
            WINDOW_SIZE,
            fusion_mode=_mode,
        )
        if _scores is None or len(_scores) == 0:
            print(f"  [WARN] Impossible de calculer proj_atk pour mode={_mode!r}.")
            continue
        _threshold_calibration_scores[_mode] = _scores
        _threshold_calibration_results[_mode] = _calibrate_decision_threshold_from_scores(
            _scores,
            fpr_target=fpr_target,
            source_label=_thr_source,
            mode_label=_mode,
        )

    _active_fusion_mode = str(CONFIG.get("INTER_METHOD_FUSION", "wbf")).strip().lower()
    _active_calib_result = _threshold_calibration_results.get(_active_fusion_mode)
    if _active_calib_result is None and _threshold_calibration_results:
        _active_fusion_mode = next(iter(_threshold_calibration_results))
        _active_calib_result = _threshold_calibration_results[_active_fusion_mode]
        print(f"  [WARN] Aucun seuil pour le mode actif; fallback sur {_active_fusion_mode!r}.")

    if _active_calib_result is not None:
        models_pkg['_decision_threshold'] = _active_calib_result["decision_threshold"]
        models_pkg['_decision_variable']  = 'proj_atk'
        models_pkg['_decision_threshold_by_fusion_mode'] = {
            _mode: _res["decision_threshold"]
            for _mode, _res in _threshold_calibration_results.items()
        }
        models_pkg['_decision_variable_by_fusion_mode'] = {
            _mode: "proj_atk"
            for _mode in _threshold_calibration_results
        }
        models_pkg['_threshold_calibration_results'] = _threshold_calibration_results
        _calib_strategy = _active_calib_result["calib_strategy"]
        b_atk_train = _threshold_calibration_scores.get(_active_fusion_mode)
        print(f"  -> Seuil actif: mode={_active_fusion_mode}, "
              f"DECISION_THRESHOLD={models_pkg['_decision_threshold']:.6f}")
    else:
        b_atk_train = None

    if False and b_atk_train is not None and len(b_atk_train) > 0:
        p50          = float(np.quantile(b_atk_train, 0.50))
        p99          = float(np.quantile(b_atk_train, 0.99))
        decision_thr = float(np.quantile(b_atk_train, 1.0 - fpr_target))
        _calib_strategy = "quantile standard"

        # ── Détection du plancher de bijection (cas sparse) ──────────────────
        # PATCH TASK-23 (audit_tmp MAJ-02, 2026-04-26)
        # ──────────────────────────────────────────────────────────────────
        # Externalisation des nombres "magiques" historiquement hardcodés
        # ici (0.5, 1e-9, ±1 %) en clés CONFIG explicitement nommées :
        #   CALIB_BIJECTION_FLOOR_TOL  : tolérance d'égalité avec le
        #       plancher 1/(WINDOW_SIZE+W). Default 0.01 (±1 %).
        #   CALIB_AGEING_WIN_FRACTION  : fraction de fenêtre-équivalente
        #       à laquelle déclencher la détection après ageing. Default 0.5.
        #   CALIB_SPARSITY_CUTOFF      : seuil sous lequel proj_atk est
        #       considéré "structurellement nul". Default 1e-9.
        # Si non définis dans config.py, les valeurs default conservent le
        # comportement historique exact.
        _W_BIJ_s1        = float(CONFIG.get('SL_PARAM_K', 3.0))
        _bijection_floor = 1.0 / (WINDOW_SIZE + _W_BIJ_s1)
        _floor_tol       = float(CONFIG.get('CALIB_BIJECTION_FLOOR_TOL', 0.01))
        _ageing_frac     = float(CONFIG.get('CALIB_AGEING_WIN_FRACTION', 0.5))
        _spars_cutoff    = float(CONFIG.get('CALIB_SPARSITY_CUTOFF', 1e-9))
        _floor_lo        = _bijection_floor * (1.0 - _floor_tol)
        _floor_hi        = _bijection_floor * (1.0 + _floor_tol)

        if _floor_lo <= decision_thr <= _floor_hi:
            _lambda          = float(CONFIG.get('LAMBDA_DECAY', 0.85))
            decision_thr     = _bijection_floor * (1.0 - _lambda) * _ageing_frac
            _calib_strategy  = (
                f"plancher bijection → accumulation ageing "
                f"(floor={_bijection_floor:.4f}, λ={_lambda:.2f}, "
                f"frac={_ageing_frac:.2f}, "
                f"δ=floor×(1−λ)×frac={decision_thr:.6f})"
            )
            print(f"  [CALIB-S1b] Quantile FPR ≈ plancher bijection ({_bijection_floor:.4f}, "
                  f"tol=±{_floor_tol*100:.1f}%).")
            print(f"              δ = {_bijection_floor:.4f} × (1−{_lambda:.2f}) × {_ageing_frac} "
                  f"= {decision_thr:.6f}")

        # ── Distribution sparse : p(FPR_target) ≈ 0 ──────────────────────────
        if decision_thr < _spars_cutoff:
            nonzero_vals = b_atk_train[b_atk_train > _spars_cutoff]
            if len(nonzero_vals) > 0:
                if float(np.min(nonzero_vals)) >= _floor_lo:
                    # Le minimum non-nul est le plancher de bijection
                    _lambda      = float(CONFIG.get('LAMBDA_DECAY', 0.85))
                    decision_thr = _bijection_floor * (1.0 - _lambda) * _ageing_frac
                    _calib_strategy = (
                        f"plancher bijection via S2b → δ=floor×(1−λ)×frac={decision_thr:.6f}"
                    )
                    print(f"  [CALIB-S2b] min non-nul = plancher bijection "
                          f"→ δ = {decision_thr:.6f}")
                else:
                    decision_thr    = float(np.min(nonzero_vals))
                    _calib_strategy = f"min non-nul (n={len(nonzero_vals)})"
                    print(f"  [CALIB-S2] p{(1-fpr_target)*100:.0f}≈0 mais "
                          f"{len(nonzero_vals)} fenêtres non-nulles → δ={decision_thr:.6f}")
            else:
                # Toutes les fenêtres d'entraînement ont proj_atk ≈ 0.
                # Cause : T_atk (quantile EVT 99.99%) trop élevé pour les résidus normaux
                # → aucune fenêtre ne génère d'évidence N > 0. C'est le comportement
                # correct. On fixe le seuil à `_ageing_frac × b_atk_min`.
                _W_BIJ       = float(CONFIG.get('SL_PARAM_K', 3.0))
                b_atk_min    = 1.0 / (WINDOW_SIZE + _W_BIJ)
                decision_thr = b_atk_min * _ageing_frac
                _calib_strategy = (
                    f"seuil théorique (b_atk_min={b_atk_min:.6f} → "
                    f"×{_ageing_frac:.2f})"
                )
                print(f"  [CALIB-S3] Toutes fenêtres proj_atk≈0 → δ = {decision_thr:.6f}")

        # ── Plancher configurable ─────────────────────────────────────────────
        _min_thr = float(CONFIG.get('MIN_DECISION_THRESHOLD', 0.0))
        if _min_thr > 0 and decision_thr < _min_thr:
            print(f"  [CALIB] Seuil relevé au plancher MIN_DECISION_THRESHOLD={_min_thr:.4f}")
            decision_thr    = _min_thr
            _calib_strategy += f" + plancher {_min_thr}"

        models_pkg['_decision_threshold'] = decision_thr
        models_pkg['_decision_variable']  = 'proj_atk'
        print(f"  Distribution proj_atk ({_thr_source}) :")
        print(f"    p50={p50:.4f}  p99={p99:.4f}  "
              f"DECISION_THRESHOLD={decision_thr:.4f}  "
              f"(sur {len(b_atk_train)} fenêtres, variable=proj_atk)")
        print(f"    Stratégie : {_calib_strategy}")
    elif _active_calib_result is None:
        print("  [WARN] Impossible de calculer proj_atk sur les données d'entraînement.")
        print("         DECISION_THRESHOLD NON auto-calibré — valeur config.py conservée.")
        _calib_strategy = "unknown"

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    output_file = get_model_path(CONFIG, up_levels=1)
    print(f"\nSauvegarde des modèles dans '{output_file}'...")

    # Trust scores = R² du modèle avec plancher (Jøsang 2016, §14.3, Def. 14.6).
    # t_k ∈ [floor, 1] : reflète la qualité prédictive de chaque source.
    # Note : ces scores sont une heuristique externe non dérivée de la théorie SL.
    # Ils sont utilisés uniquement en mode WBF_WEIGHT_MODE='trust_discount'.
    # En mode opérationnel (WBF_WEIGHT_MODE='uniform'), ils n'affectent pas la détection.
    _trust_floor = CONFIG.get('TRUST_SCORE_FLOOR', 0.05)
    models_pkg['trust_scores'] = {
        k: max(float(models_pkg[k].get('r2_score', _trust_floor)), _trust_floor)
        for k in empirical_priors
        if isinstance(models_pkg.get(k), dict)
    }

    # PATCH D5 (MASE trust mode, Hyndman-Koehler 2006) — Joesang Def. 14.6
    # compatible trust derived from MASE.
    #
    # MASE replaces R² as the trust proxy for **temporal** sources (Prophet).
    # The mapping ``trust = max(floor, 1 − α·MASE)`` is monotonically
    # non-increasing in MASE so a model that is worse than the Naive-1
    # baseline (MASE > 1) is silenced down to ``floor`` while a perfect
    # model (MASE = 0) gets full trust.  Critically, ``trust ∈ [floor, 1]``
    # — no source is *amplified* by trust, only discounted.
    #
    # For non-temporal RANSAC reconstructions ``mase_score`` is NaN by
    # design (MASE is undefined for cross-feature regression); the helper
    # ``mase_to_trust`` then falls back to ``TRUST_SCORE_FLOOR`` for those
    # keys.  RANSAC fiability is governed by ``RECONST_ATTACK_RELIABILITY``
    # (Mercier-Quost-Denoeux 2008 contextual discounting) which is
    # orthogonal to the WBF weighting mode.
    #
    # Used only when ``WBF_WEIGHT_MODE='mase'`` (opt-in).  Default
    # ``'uniform'`` ignores this dictionary entirely.
    _mase_alpha = float(CONFIG.get('MASE_TRUST_ALPHA', 1.0))
    models_pkg['mase_scores'] = {
        k: mase_to_trust(
            float(models_pkg[k].get('mase_score', float('nan'))),
            alpha=_mase_alpha,
            floor=_trust_floor,
        )
        for k in empirical_priors
        if isinstance(models_pkg.get(k), dict)
    }

    # Métadonnée d'auditabilité : split_date enregistrée dans le pkg pour
    # détection de fuite train/test lors du chargement dans compute_evidence_v2.py.
    models_pkg['_meta_split_date'] = str(CONFIG['split_date'])

    # 2026-05-06 — Persist the per-metric calibration residuals so
    # downstream auditing scripts (notably the EVT declustering ablation
    # under `src/sl_ads/ablation/ablation_evt_declustering.py`) can
    # rerun the EVT step on the *actual* calibration distribution
    # instead of an inference-span proxy. The residuals are dense
    # numpy arrays; total footprint ≈ 17 metrics × ~80k floats ≈ 11 MB.
    # No backwards-compatibility break: legacy consumers ignore the key.
    models_pkg['_calib_signed_residuals'] = {
        k: np.asarray(v, dtype=float) for k, v in _calib_signed_residuals.items()
    }

    joblib.dump(models_pkg, output_file)

    # Sidecar JSON léger : seuils auto-calibrés lisibles sans charger le pkl.
    # Lu par get_decision_threshold() dans paths.py et propagé à tous les scripts
    # d'évaluation.
    sidecar_path = get_threshold_sidecar_path(CONFIG, up_levels=1)

    # ── PATCH m-08 / F28 : Fallback audit summary ────────────────────────────
    # Break the log into (a) counts per kind, (b) list of affected metrics
    # per kind, (c) a compact text digest.  The full event stream is written
    # to a dedicated JSON next to the sidecar.
    _fb_counts: dict = {}
    _fb_metrics: dict = {}
    for _evt in _FALLBACK_LOG:
        _k = _evt.get("kind", "unknown")
        _fb_counts[_k] = _fb_counts.get(_k, 0) + 1
        _m = _evt.get("metric")
        if _m is not None:
            _fb_metrics.setdefault(_k, [])
            if _m not in _fb_metrics[_k]:
                _fb_metrics[_k].append(_m)

    sidecar_data = {
        "decision_threshold":    ((_active_calib_result or {}).get("decision_threshold")
                                  if '_active_calib_result' in locals()
                                  else models_pkg.get("_decision_threshold")),
        "decision_variable":     models_pkg.get("_decision_variable", "proj_atk"),
        "fpr_target":            CONFIG.get("FPR_TARGET_DECISION", 0.01),
        "evt_enabled":           CONFIG.get("USE_EVT_THRESHOLDS", True),
        "evt_q_susp":            CONFIG.get("EVT_Q_SUSP", 1e-2),
        "evt_q_atk":             CONFIG.get("EVT_Q_ATK",  1e-4),
        "calib_strategy":        ((_active_calib_result or {}).get("calib_strategy")
                                  if '_active_calib_result' in locals()
                                  else _calib_strategy),
        "b_atk_train_n_windows": ((_active_calib_result or {}).get("n_windows")
                                  if '_active_calib_result' in locals()
                                  else (int(len(b_atk_train)) if b_atk_train is not None else None)),
        "b_atk_train_nonzero":   ((_active_calib_result or {}).get("nonzero")
                                  if '_active_calib_result' in locals()
                                  else (int(np.sum(b_atk_train > 1e-9))
                                        if b_atk_train is not None else None)),
        # m-08/F28 : compact audit info is embedded here so any downstream
        # loader (paths.py, utils_manifest) can surface it cheaply.
        "fallbacks": {
            "total":     len(_FALLBACK_LOG),
            "counts":    _fb_counts,
            "metrics":   _fb_metrics,
        },
        # PATCH TASK-45 (audit_codex CRIT-02 + MAJ-09, 2026-04-27).
        # Persist the inter-method fusion mode and ageing/discount
        # parameters that *would* be active if the deployed pipeline ran
        # ``compute_opinions_v3`` end-to-end.  The current threshold is
        # calibrated on ``_compute_training_proj_atk`` which is a
        # SIMPLIFIED surrogate of the deployed chain (no ageing, no
        # contextual discount, no inter-method fusion stage); hence the
        # reported threshold is only fully consistent if the surrogate
        # matches the deployment.  Recording the deployment parameters
        # here lets evaluation scripts cross-check and warn when they
        # differ — see ``docs/honest_limitations.md`` §CRIT-02.
        "fusion_mode_at_calibration": str(
            _active_fusion_mode if '_active_fusion_mode' in locals()
            else CONFIG.get("INTER_METHOD_FUSION", "wbf")
        ),
        "wbf_weight_mode":            str(CONFIG.get("WBF_WEIGHT_MODE", "uniform")),
        "lambda_decay":               float(CONFIG.get("LAMBDA_DECAY", 0.85)),
        "cd_alpha_attack":             float(CONFIG.get("CD_ALPHA_ATTACK", 1.0)),
        "balance_ratio":              _effective_balance_ratio(),
        # PATCH H2 — calendar-aware EVT signature.  Persisted so that
        # ``paths.validate_threshold_sidecar_config`` can hard-fail on
        # any drift between the calibration-time regime function and
        # the runtime regime function (different daytime hours,
        # different holiday calendar, partition revision, etc.).  When
        # CALENDAR_EVT is disabled the field is None — older PKLs
        # without this field stay readable thanks to the "missing"
        # bucket in the validator (no hard failure).
        "calendar_evt_signature": (
            REGIME_FN_SIGNATURE
            if CONFIG.get("CALENDAR_EVT_ENABLED", False)
            else None
        ),
        "calendar_evt_enabled":   bool(CONFIG.get("CALENDAR_EVT_ENABLED", False)),
        "threshold_sidecar_scope":    "legacy_active_mode",
        "threshold_calibration_modes": list(_threshold_calibration_results.keys()),
        "method_groups":              CONFIG.get("FUSION_METHOD_GROUPS"),
        "calibration_surrogate":      "_compute_training_proj_atk (mode-aware method fusion, no ageing)",
        "calibration_surrogate_caveat": (
            "audit_codex CRIT-02: this threshold is calibrated on a simplified "
            "surrogate of the deployed chain. It now uses the configured "
            "method groups and inter-method fusion mode, but still omits "
            "temporal ageing. It is consistent with the deployed score IF "
            "the deployed configuration matches the values "
            "stored in this sidecar (fusion_mode_at_calibration, "
            "wbf_weight_mode, lambda_decay, cd_alpha_attack, balance_ratio, "
            "calendar_evt_signature)."
        ),
        "patch_id":                   "TASK-45 + TASK-50 + H2 (calendar-aware EVT)",
        "iso_date":                   "2026-05-07",
    }
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f, indent=2)
    print(f"[OK] Sidecar seuils : '{sidecar_path}'")

    for _mode, _res in _threshold_calibration_results.items():
        _mode_sidecar = dict(sidecar_data)
        _mode_sidecar.update({
            "decision_threshold": _res["decision_threshold"],
            "decision_variable": _res["decision_variable"],
            "calib_strategy": _res["calib_strategy"],
            "b_atk_train_n_windows": _res["n_windows"],
            "b_atk_train_nonzero": _res["nonzero"],
            "fusion_mode_at_calibration": _mode,
            "threshold_sidecar_scope": "mode_specific",
        })
        _mode_path = get_threshold_sidecar_path(
            CONFIG,
            up_levels=1,
            fusion_mode=_mode,
        )
        with open(_mode_path, "w", encoding="utf-8") as f:
            json.dump(_mode_sidecar, f, indent=2)
        print(f"[OK] Sidecar seuils ({_mode}) : '{_mode_path}'")

    # ── Full fallback log → dedicated JSON artifact ──────────────────────────
    # Same folder as the model PKL; name mirrors it so it's trivially
    # associated (trained_models_{version}_fallbacks.json).
    _model_path = get_model_path(CONFIG, up_levels=1)
    _fb_path = os.path.splitext(_model_path)[0] + "_fallbacks.json"
    try:
        with open(_fb_path, "w", encoding="utf-8") as _fbf:
            json.dump({
                "version":  CONFIG.get("VERSION_NAME", "unknown"),
                "total":    len(_FALLBACK_LOG),
                "counts":   _fb_counts,
                "metrics":  _fb_metrics,
                "events":   _FALLBACK_LOG,
            }, _fbf, indent=2)
        print(f"[OK] Fallback audit : '{_fb_path}' "
              f"({len(_FALLBACK_LOG)} events across {len(_fb_counts)} kinds)")
    except OSError as _exc:
        print(f"[WARN] Could not write fallback audit JSON: {_exc!r}")

    print("[OK] Entraînement terminé avec succès.")


if __name__ == "__main__":
    train_models()
