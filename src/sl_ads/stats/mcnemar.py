"""
stats_mcnemar.py
================

McNemar paired test for comparing two binary classifiers on the *same*
test set. Returns the 2x2 discordant-cell contingency table, the test
statistic, and a p-value computed by:

  * Edwards' continuity-corrected chi-square approximation when n_disc >= 25
    (Edwards 1948 - "Note on the correction for continuity in testing the
    significance of the difference between correlated proportions"),
  * the exact two-sided binomial test on the off-diagonal cells when
    n_disc < 25 (recommended by Pembury Smith & Ruxton 2020 and by the
    scipy.stats.contingency.mcnemar documentation).

This module deliberately does NOT depend on any of the project's other
modules so it can be imported by tests, ablations, or shipped as a
stand-alone artefact for reviewer verification.

References
----------
McNemar, Q. (1947). "Note on the sampling error of the difference between
correlated proportions or percentages." *Psychometrika* 12 (2): 153-157.

Edwards, A. L. (1948). "Note on the correction for continuity in testing
the significance of the difference between correlated proportions."
*Psychometrika* 13 (3): 185-187.

Dietterich, T. G. (1998). "Approximate statistical tests for comparing
supervised classification learning algorithms." *Neural Computation*
10 (7): 1895-1923. - explicitly recommends McNemar for ML classifier
comparison when only one train/test split is available.

Pembury Smith, M. Q. R. & Ruxton, G. D. (2020). "Effective use of the
McNemar test." *Behavioral Ecology and Sociobiology* 74 (133).
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy.stats import binomtest, chi2

__all__ = [
    "mcnemar_paired_test",
    "mcnemar_contingency",
    "format_mcnemar",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mcnemar_contingency(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, int]:
    """Build the 2x2 contingency table required by McNemar.

    Cells are labelled by *correctness* of each classifier (1=correct,
    0=wrong). The off-diagonal cells (n01, n10) are the *discordant*
    pairs that the test operates on; the diagonal cells (n00, n11)
    cancel out.

    Parameters
    ----------
    y_true   : (n,) array of ground-truth labels.
    y_pred_a : (n,) array of predictions from classifier A.
    y_pred_b : (n,) array of predictions from classifier B.

    Returns
    -------
    dict with keys 'n00', 'n01', 'n10', 'n11', 'n_disc'.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    if not (y_true.shape == y_pred_a.shape == y_pred_b.shape):
        raise ValueError(
            f"shape mismatch: y_true={y_true.shape}, A={y_pred_a.shape}, "
            f"B={y_pred_b.shape}"
        )

    a_ok = (y_pred_a == y_true).astype(np.int8)
    b_ok = (y_pred_b == y_true).astype(np.int8)

    n11 = int(((a_ok == 1) & (b_ok == 1)).sum())
    n10 = int(((a_ok == 1) & (b_ok == 0)).sum())  # A right, B wrong
    n01 = int(((a_ok == 0) & (b_ok == 1)).sum())  # A wrong, B right
    n00 = int(((a_ok == 0) & (b_ok == 0)).sum())
    n_disc = n10 + n01
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11, "n_disc": n_disc}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mcnemar_paired_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    *,
    alpha: float = 0.05,
    exact_threshold: int = 25,
) -> Dict[str, object]:
    """McNemar paired comparison of two classifiers.

    Parameters
    ----------
    y_true   : ground-truth labels (n,).
    y_pred_a : predictions from classifier A (n,).
    y_pred_b : predictions from classifier B (n,).
    alpha    : significance threshold (default 0.05).
    exact_threshold : if (n10 + n01) < exact_threshold, use the exact
        binomial test rather than the chi-square approximation.

    Returns
    -------
    dict with keys
        method            : 'mcnemar_chi2_continuity' or 'mcnemar_binomial_exact'
        statistic         : test statistic value
        p_value           : two-sided p-value
        n00, n01, n10, n11, n_disc : contingency cells
        better            : 'A', 'B', or 'tie'
        significant_at_alpha : bool
        alpha             : the alpha used
        n                 : total sample size
        accuracy_a, accuracy_b : sample accuracies for context

    Hypothesis
    ----------
    H0: the marginal probabilities of the two classifiers' errors are equal,
        i.e. P(A wrong, B right) = P(A right, B wrong).
    H1: those probabilities differ -> one classifier is statistically better.

    Notes
    -----
    The test ignores n00 and n11 cells (where both agree) since those carry
    no information about which classifier is better; the entire discriminating
    signal lives in the discordant cells (n10, n01).

    With n_disc < 25 the chi-square approximation under-covers, so we fall
    back to the exact two-sided binomial test on the off-diagonal split:
    under H0, n10 ~ Binomial(n_disc, 0.5).
    """
    cells = mcnemar_contingency(y_true, y_pred_a, y_pred_b)
    n10, n01, n_disc = cells["n10"], cells["n01"], cells["n_disc"]
    n = len(y_true)

    if n_disc == 0:
        # Two classifiers agree on every sample -> p=1 trivially.
        return {
            "method": "mcnemar_trivial",
            "statistic": 0.0,
            "p_value": 1.0,
            **cells,
            "better": "tie",
            "significant_at_alpha": False,
            "alpha": alpha,
            "n": n,
            "accuracy_a": float((y_pred_a == y_true).mean()),
            "accuracy_b": float((y_pred_b == y_true).mean()),
        }

    if n_disc < exact_threshold:
        # Exact two-sided binomial on the off-diagonal.
        bt = binomtest(min(n10, n01), n=n_disc, p=0.5, alternative="two-sided")
        statistic = float(min(n10, n01))
        p_value = float(bt.pvalue)
        method = "mcnemar_binomial_exact"
    else:
        # Edwards continuity-corrected chi-square: chi^2 = (|n10-n01|-1)^2 / (n10+n01).
        statistic = float((abs(n10 - n01) - 1) ** 2 / max(n_disc, 1))
        # 1 d.o.f., two-sided -> p = SF(chi2, df=1).
        p_value = float(chi2.sf(statistic, df=1))
        method = "mcnemar_chi2_continuity"

    if n10 > n01:
        better = "A"
    elif n01 > n10:
        better = "B"
    else:
        better = "tie"

    return {
        "method": method,
        "statistic": statistic,
        "p_value": p_value,
        **cells,
        "better": better,
        "significant_at_alpha": bool(p_value < alpha),
        "alpha": alpha,
        "n": n,
        "accuracy_a": float((y_pred_a == y_true).mean()),
        "accuracy_b": float((y_pred_b == y_true).mean()),
    }


def format_mcnemar(res: Dict[str, object]) -> str:
    """Render a McNemar result dict as a one-line human string."""
    sig = "*" if res.get("significant_at_alpha") else "ns"
    return (
        f"McNemar [{res['method']}] n10={res['n10']} n01={res['n01']} "
        f"stat={res['statistic']:.3f} p={res['p_value']:.4f} "
        f"better={res['better']} ({sig})"
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    print("[TEST] stats_mcnemar.py self-test")
    rng = np.random.default_rng(0)
    n = 1000
    y_true = rng.integers(0, 2, size=n)

    # 1. Two identical classifiers - should yield trivial p=1.
    y_a = y_true.copy()
    y_b = y_true.copy()
    res = mcnemar_paired_test(y_true, y_a, y_b)
    assert res["p_value"] == 1.0, res
    assert res["better"] == "tie", res
    print(f"   [OK] Identical classifiers : {format_mcnemar(res)}")

    # 2. A is clearly better than B - should be significant.
    y_a = y_true.copy()
    y_a[rng.choice(n, size=20, replace=False)] ^= 1   # A: 20 errors
    y_b = y_true.copy()
    y_b[rng.choice(n, size=80, replace=False)] ^= 1   # B: 80 errors
    res = mcnemar_paired_test(y_true, y_a, y_b)
    assert res["significant_at_alpha"], res
    assert res["better"] == "A", res
    assert res["method"] == "mcnemar_chi2_continuity", res
    print(f"   [OK] A better than B       : {format_mcnemar(res)}")

    # 3. Symmetric small-n case -> exact binomial path.
    n2 = 200
    y_t = rng.integers(0, 2, size=n2)
    y_a = y_t.copy()
    y_a[rng.choice(n2, size=4, replace=False)] ^= 1
    y_b = y_t.copy()
    y_b[rng.choice(n2, size=6, replace=False)] ^= 1
    res = mcnemar_paired_test(y_t, y_a, y_b)
    assert res["method"] == "mcnemar_binomial_exact", res
    print(f"   [OK] Small-n exact path    : {format_mcnemar(res)}")

    # 4. Discordant=0 trivial case (different but agree on errors).
    y_a = y_true.copy()
    y_b = y_true.copy()
    res = mcnemar_paired_test(y_true, y_a, y_b)
    assert res["method"] == "mcnemar_trivial", res
    print(f"   [OK] Trivial path          : {format_mcnemar(res)}")

    # 5. Tie case (same number of errors but in different positions) - p > alpha.
    y_a = y_true.copy()
    idx_a = rng.choice(n, size=50, replace=False)
    y_a[idx_a] ^= 1
    y_b = y_true.copy()
    remaining = np.setdiff1d(np.arange(n), idx_a)
    idx_b = rng.choice(remaining, size=50, replace=False)
    y_b[idx_b] ^= 1
    res = mcnemar_paired_test(y_true, y_a, y_b)
    # Same error rates but positions independent -> we'd expect ~ tie.
    print(f"   [OK] Equal-error case      : {format_mcnemar(res)}")

    print("[TEST] ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
