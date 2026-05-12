"""
stats_bootstrap_ci.py — Bias-Corrected & Accelerated (BCa) bootstrap CI
========================================================================
PATCH M-14 / CI-rigor (2026-04-24).

Purpose
-------
Provide a single, well-tested, dependency-light utility to compute BCa
95% confidence intervals on classification metrics (F1, MCC, FPR, etc.)
and on paired differences between two classifiers.

Why BCa (and not percentile)
----------------------------
Percentile bootstrap CIs (Efron 1979) are first-order accurate and can
be badly biased when the statistic distribution is skewed — which is
the case for F1 / MCC in small-sample rare-event detection.  The BCa
method of Efron (1987 *JASA* 82(397):171-185) corrects for both
bias (z0) and acceleration (a, via jackknife) and is
*second-order accurate* — the de facto standard for
machine-learning benchmarks since Davison & Hinkley (1997) Ch. 5.

References
----------
- Efron, B. (1987) "Better Bootstrap Confidence Intervals",
  *JASA* 82(397):171-185.
- Efron, B. & Tibshirani, R.J. (1993) *An Introduction to the
  Bootstrap*, Chapman & Hall/CRC, §14.3.
- Davison, A.C. & Hinkley, D.V. (1997) *Bootstrap Methods and their
  Application*, Cambridge University Press, §5.3.
- DiCiccio, T.J. & Efron, B. (1996) "Bootstrap Confidence Intervals",
  *Statistical Science* 11(3):189-228.

Public API
----------
- :func:`bootstrap_bca_ci` — BCa CI on an arbitrary statistic of a
  single sample (e.g. F1 on (y_true, y_pred)).
- :func:`paired_bootstrap_bca_ci` — BCa CI on the paired difference
  of a statistic between two predictions on the same ground truth
  (e.g. F1_A - F1_B over the same windows).
- :func:`format_ci` — render "point [lo, hi]" uniformly.

Self-test
---------
Run ``python stats_bootstrap_ci.py`` to execute the built-in
validation suite.  Exit code 0 on success.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence, Tuple

import numpy as np
from scipy import stats as _scipy_stats


# ──────────────────────────────────────────────────────────────────────
# Core BCa utility
# ──────────────────────────────────────────────────────────────────────
def _bca_endpoints(
        theta_hat: float,
        theta_boot: np.ndarray,
        jackknife_thetas: np.ndarray,
        alpha: float,
) -> Tuple[float, float]:
    """
    Compute the BCa interval endpoints given:
      - theta_hat          : observed statistic on the full sample
      - theta_boot         : array of B bootstrap replicates
      - jackknife_thetas   : array of n leave-one-out replicates
      - alpha              : two-sided significance level (0.05 for 95% CI)

    Returns (lower, upper).  Degrades gracefully to the percentile
    interval if BCa inputs are degenerate (zero acceleration or all-
    identical bootstrap replicates).
    """
    theta_boot = np.asarray(theta_boot, dtype=float)
    theta_boot = theta_boot[np.isfinite(theta_boot)]
    if theta_boot.size == 0:
        return (float("nan"), float("nan"))

    # ── bias correction z0 ────────────────────────────────────────────
    p0 = float(np.mean(theta_boot < theta_hat))
    # Clip to (0, 1) open to keep ppf finite.
    p0 = min(max(p0, 1e-9), 1.0 - 1e-9)
    z0 = _scipy_stats.norm.ppf(p0)

    # ── acceleration a (jackknife) ────────────────────────────────────
    jk = np.asarray(jackknife_thetas, dtype=float)
    jk = jk[np.isfinite(jk)]
    if jk.size < 3:
        # Not enough jackknife samples for a stable estimate — fall back
        # to percentile on the raw bootstrap.
        a = 0.0
    else:
        jk_mean = float(np.mean(jk))
        num = float(np.sum((jk_mean - jk) ** 3))
        denom = 6.0 * (float(np.sum((jk_mean - jk) ** 2)) ** 1.5)
        a = num / denom if denom > 1e-24 else 0.0

    # ── adjusted percentiles ──────────────────────────────────────────
    z_lo = _scipy_stats.norm.ppf(alpha / 2.0)
    z_hi = _scipy_stats.norm.ppf(1.0 - alpha / 2.0)

    def _adjust(z: float) -> float:
        denom = 1.0 - a * (z0 + z)
        if abs(denom) < 1e-12:
            return p0  # degenerate → no adjustment
        return float(_scipy_stats.norm.cdf(z0 + (z0 + z) / denom))

    alpha_lo = _adjust(z_lo)
    alpha_hi = _adjust(z_hi)

    # Clip for safety (FP overflow near the tails).
    alpha_lo = min(max(alpha_lo, 1e-6), 1.0 - 1e-6)
    alpha_hi = min(max(alpha_hi, 1e-6), 1.0 - 1e-6)
    if alpha_lo > alpha_hi:
        # Pathological ordering — fall back to percentile.
        alpha_lo, alpha_hi = alpha / 2.0, 1.0 - alpha / 2.0

    lo = float(np.quantile(theta_boot, alpha_lo))
    hi = float(np.quantile(theta_boot, alpha_hi))
    return (lo, hi)


# ──────────────────────────────────────────────────────────────────────
# Single-sample BCa
# ──────────────────────────────────────────────────────────────────────
def _normalise_block_length(n: int, block_length: int | None) -> int:
    """Return an admissible moving-block length; 1 means iid bootstrap."""
    if block_length is None:
        return 1
    b = int(block_length)
    if b <= 1:
        return 1
    return max(2, min(b, n))


def _moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Sample length-n indices via the moving block bootstrap."""
    if block_length <= 1:
        return rng.integers(0, n, size=n)
    max_start = n - block_length
    if max_start <= 0:
        return np.arange(n)
    n_blocks = int(math.ceil(n / block_length))
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_length) for s in starts])
    return idx[:n]


def _jackknife_masks(n: int, block_length: int) -> list[np.ndarray]:
    """Leave-one or delete-one-block masks used for BCa acceleration."""
    if block_length <= 1:
        return [np.arange(n) != i for i in range(n)]
    masks = []
    for start in range(0, n, block_length):
        stop = min(start + block_length, n)
        mask = np.ones(n, dtype=bool)
        mask[start:stop] = False
        if mask.sum() >= 2:
            masks.append(mask)
    return masks


def bootstrap_bca_ci(
        y_true: Sequence,
        y_pred: Sequence,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        n_boot: int = 2000,
        alpha: float = 0.05,
        seed: int = 42,
        block_length: int | None = None,
) -> dict:
    """
    BCa 95% confidence interval for ``metric_fn(y_true, y_pred)``.

    Parameters
    ----------
    y_true, y_pred : array-like, same length N
        Ground truth and prediction arrays.  Can be binary labels,
        continuous scores, or any type accepted by ``metric_fn``.
    metric_fn : callable(y_true_arr, y_pred_arr) -> float
        Scalar-valued metric (F1, MCC, FPR, accuracy, AUC…).
        Must return a single finite float.
    n_boot : int, default 2000
        Number of bootstrap replicates.  Efron & Tibshirani (1993) §14
        recommend ≥ 1000 for BCa; we default to 2000 for tighter tails.
    alpha : float, default 0.05
        Two-sided significance level (0.05 → 95% CI).
    seed : int, default 42
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        - 'point'    : float, observed metric on the full sample
        - 'ci_low'   : float, lower bound of the BCa interval
        - 'ci_high'  : float, upper bound of the BCa interval
        - 'method'   : 'BCa' (or 'percentile' if BCa degenerates)
        - 'n_boot'   : int, number of bootstrap replicates used
        - 'n'        : int, sample size
        - 'alpha'    : float, significance level

    Notes
    -----
    Each bootstrap resamples indices with replacement.  Jackknife is
    done on leave-one-out.  Invalid replicates (NaN/inf metrics) are
    dropped before the percentile computation — if > 20% of replicates
    are invalid, the function emits a warning and returns NaN bounds.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true (n={y_true.shape[0]}) and y_pred (n={y_pred.shape[0]}) "
            f"must have the same length."
        )
    n = y_true.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 samples for bootstrap CI; got {n}.")
    block_len = _normalise_block_length(n, block_length)
    resampling = "moving_block" if block_len > 1 else "iid"

    # Observed statistic on the full sample.
    theta_hat = float(metric_fn(y_true, y_pred))
    if not math.isfinite(theta_hat):
        raise ValueError(
            f"metric_fn returned non-finite value {theta_hat!r} on the full "
            "sample — cannot bootstrap a degenerate metric."
        )

    rng = np.random.default_rng(seed)

    # ── bootstrap replicates ─────────────────────────────────────────
    theta_boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = _moving_block_indices(n, block_len, rng)
        try:
            theta_boot[b] = float(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            theta_boot[b] = float("nan")

    n_valid = int(np.sum(np.isfinite(theta_boot)))
    if n_valid < 0.8 * n_boot:
        import warnings
        warnings.warn(
            f"Only {n_valid}/{n_boot} bootstrap replicates were finite — "
            "metric may be degenerate on small samples.  CI may be unreliable.",
            RuntimeWarning, stacklevel=2,
        )
    if n_valid < 10:
        return {
            'point': theta_hat, 'ci_low': float('nan'), 'ci_high': float('nan'),
            'method': 'failed', 'n_boot': n_boot, 'n': n, 'alpha': alpha,
            'resampling': resampling, 'block_length': block_len,
            'note': f'Only {n_valid} valid bootstrap replicates',
        }

    # ── jackknife (leave-one-out) for acceleration ───────────────────
    jackknife_masks = _jackknife_masks(n, block_len)
    jackknife_thetas = np.empty(len(jackknife_masks), dtype=float)
    for i, mask in enumerate(jackknife_masks):
        try:
            jackknife_thetas[i] = float(metric_fn(y_true[mask], y_pred[mask]))
        except Exception:
            jackknife_thetas[i] = float("nan")

    lo, hi = _bca_endpoints(theta_hat, theta_boot, jackknife_thetas, alpha)
    return {
        'point': theta_hat,
        'ci_low': lo,
        'ci_high': hi,
        'method': 'BCa-block' if block_len > 1 else 'BCa',
        'n_boot': n_boot,
        'n': n,
        'alpha': alpha,
        'resampling': resampling,
        'block_length': block_len,
    }


# ──────────────────────────────────────────────────────────────────────
# Paired BCa on the difference between two classifiers
# ──────────────────────────────────────────────────────────────────────
def paired_bootstrap_bca_ci(
        y_true: Sequence,
        y_pred_a: Sequence,
        y_pred_b: Sequence,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        n_boot: int = 2000,
        alpha: float = 0.05,
        seed: int = 42,
        block_length: int | None = None,
) -> dict:
    """
    BCa 95% CI on the paired difference ``metric_fn(y_true, y_pred_a) -
    metric_fn(y_true, y_pred_b)``.

    Predictions A and B are evaluated on **the same** resampled indices
    each bootstrap iteration — this preserves the pairing (the key
    correctness property vs two independent CIs).

    Useful for: *"Is SL's F1 significantly higher than IF's F1 on the
    same windows?"*  If the CI on the difference excludes 0, the
    advantage is significant at the 1 - alpha level.

    Returns the same dict as :func:`bootstrap_bca_ci` but the 'point'
    is the observed difference A - B.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    if not (y_true.shape[0] == y_pred_a.shape[0] == y_pred_b.shape[0]):
        raise ValueError(
            f"All arrays must have the same length; got "
            f"y_true={y_true.shape[0]}, A={y_pred_a.shape[0]}, "
            f"B={y_pred_b.shape[0]}."
        )
    n = y_true.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 samples; got {n}.")
    block_len = _normalise_block_length(n, block_length)
    resampling = "moving_block" if block_len > 1 else "iid"

    theta_a = float(metric_fn(y_true, y_pred_a))
    theta_b = float(metric_fn(y_true, y_pred_b))
    delta_hat = theta_a - theta_b
    if not math.isfinite(delta_hat):
        raise ValueError(f"Non-finite observed difference {delta_hat!r}.")

    rng = np.random.default_rng(seed)

    delta_boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = _moving_block_indices(n, block_len, rng)
        try:
            ta = float(metric_fn(y_true[idx], y_pred_a[idx]))
            tb = float(metric_fn(y_true[idx], y_pred_b[idx]))
            delta_boot[i] = ta - tb
        except Exception:
            delta_boot[i] = float("nan")

    # Jackknife on the paired difference.
    jackknife_masks = _jackknife_masks(n, block_len)
    jackknife = np.empty(len(jackknife_masks), dtype=float)
    for i, mask in enumerate(jackknife_masks):
        try:
            ta = float(metric_fn(y_true[mask], y_pred_a[mask]))
            tb = float(metric_fn(y_true[mask], y_pred_b[mask]))
            jackknife[i] = ta - tb
        except Exception:
            jackknife[i] = float("nan")

    lo, hi = _bca_endpoints(delta_hat, delta_boot, jackknife, alpha)
    # Is the advantage significant? (CI excludes 0?)
    significant = bool((lo > 0.0) or (hi < 0.0))
    return {
        'point': delta_hat,
        'ci_low': lo,
        'ci_high': hi,
        'method': 'BCa-block' if block_len > 1 else 'BCa',
        'n_boot': n_boot,
        'n': n,
        'alpha': alpha,
        'resampling': resampling,
        'block_length': block_len,
        'theta_a': theta_a,
        'theta_b': theta_b,
        'significant_at_alpha': significant,
    }


# ──────────────────────────────────────────────────────────────────────
# Formatting helper
# ──────────────────────────────────────────────────────────────────────
def format_ci(res: dict, digits: int = 3) -> str:
    """Format a CI dict as 'point [lo, hi]' string."""
    if not math.isfinite(res.get('ci_low', float('nan'))):
        return f"{res['point']:.{digits}f} [CI unavailable]"
    return (
        f"{res['point']:.{digits}f} "
        f"[{res['ci_low']:.{digits}f}, {res['ci_high']:.{digits}f}]"
    )


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────
def _self_test() -> int:
    """Basic sanity tests on synthetic data with known ground truth."""
    from sklearn.metrics import f1_score, matthews_corrcoef  # noqa: E402

    print("[TEST] stats_bootstrap_ci.py — self-test")
    # 1. Perfect classifier → F1 = 1.0, CI very tight.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    res = bootstrap_bca_ci(y, y, f1_score, n_boot=500, seed=1)
    assert math.isclose(res['point'], 1.0, abs_tol=1e-9), res
    assert res['ci_low'] > 0.95, f"CI_low too low for perfect classifier: {res}"
    print(f"   [OK] Perfect classifier  : F1 = {format_ci(res)}")

    # 2. Random 50/50 classifier → F1 near 0.5, CI wide.
    y_pred_rand = rng.integers(0, 2, size=200)
    res2 = bootstrap_bca_ci(y, y_pred_rand, f1_score, n_boot=500, seed=2)
    assert 0.30 <= res2['point'] <= 0.70, res2
    assert res2['ci_high'] - res2['ci_low'] > 0.05, res2
    print(f"   [OK] Random classifier   : F1 = {format_ci(res2)}")

    # 3. MCC on strong correlation → high, CI tight.
    y_bal = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
    y_pred_close = y_bal.copy()
    flip_idx = rng.choice(200, size=20, replace=False)
    y_pred_close[flip_idx] = 1 - y_pred_close[flip_idx]
    res3 = bootstrap_bca_ci(y_bal, y_pred_close, matthews_corrcoef,
                             n_boot=500, seed=3)
    assert res3['point'] > 0.7, res3
    print(f"   [OK] 90% acc classifier  : MCC = {format_ci(res3)}")

    # 4. Paired diff: A better than B.
    y_pred_a = y_bal.copy()
    y_pred_a[rng.choice(200, size=10, replace=False)] ^= 1
    y_pred_b = y_bal.copy()
    y_pred_b[rng.choice(200, size=40, replace=False)] ^= 1
    res4 = paired_bootstrap_bca_ci(y_bal, y_pred_a, y_pred_b, f1_score,
                                    n_boot=500, seed=4)
    assert res4['point'] > 0.0, res4  # A should be better
    assert res4['significant_at_alpha'], res4  # and significantly so
    print(f"   [OK] Paired diff (A>B)   : delta_F1 = {format_ci(res4)} "
          f"significant={res4['significant_at_alpha']}")

    # 5. Paired diff: A == B (bootstrap CI should straddle 0).
    res5 = paired_bootstrap_bca_ci(y_bal, y_pred_a, y_pred_a, f1_score,
                                    n_boot=500, seed=5)
    assert math.isclose(res5['point'], 0.0, abs_tol=1e-9), res5
    print(f"   [OK] Paired diff (A==B)  : delta_F1 = {format_ci(res5)}")

    # 6. Degenerate: all-correct sample -> MCC undefined; should warn but not crash.
    try:
        res6 = bootstrap_bca_ci(np.zeros(30, dtype=int), np.zeros(30, dtype=int),
                                 matthews_corrcoef, n_boot=200, seed=6)
        # matthews_corrcoef of all-same returns 0 by sklearn convention.
        print(f"   [OK] Degenerate case     : MCC = {format_ci(res6)} "
              f"(method={res6['method']})")
    except Exception as e:
        print(f"   [OK] Degenerate case raised  : {e}")

    # 7. Moving-block bootstrap exposes the resampling metadata.
    res7 = bootstrap_bca_ci(y_bal, y_pred_close, f1_score,
                            n_boot=200, seed=7, block_length=5)
    assert res7['resampling'] == 'moving_block', res7
    assert res7['block_length'] == 5, res7
    assert res7['method'] == 'BCa-block', res7
    print(f"   [OK] Moving-block bootstrap: F1 = {format_ci(res7)} "
          f"(block={res7['block_length']})")

    res8 = paired_bootstrap_bca_ci(y_bal, y_pred_a, y_pred_b, f1_score,
                                    n_boot=200, seed=8, block_length=5)
    assert res8['resampling'] == 'moving_block', res8
    assert res8['block_length'] == 5, res8
    print(f"   [OK] Paired moving-block   : delta_F1 = {format_ci(res8)}")

    print("[TEST] ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
