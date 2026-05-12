"""Mean Absolute Scaled Error (MASE) and the derived Joesang-compatible
trust score used by ``WBF_WEIGHT_MODE='mase'``.

Background
----------
Hyndman & Koehler (2006) introduced MASE as a scale-invariant alternative
to R² / RMSE for forecasting accuracy assessment.  For a one-step-ahead
forecast on a time series ``y_1, ..., y_T`` with predictions
``ŷ_1, ..., ŷ_T``:

    MASE  =  mean_t |y_t - ŷ_t|
            ────────────────────────────────────
            (1 / (T-1)) · sum_{t=2}^{T} |y_t - y_{t-1}|

The denominator is the mean absolute one-step **Naive-1** in-sample
forecast error.  Interpretation:

  - MASE = 0  ⇒  perfect prediction;
  - MASE < 1  ⇒  the model beats the trivial persistence baseline
                 (informative);
  - MASE = 1  ⇒  the model is exactly as good as Naive-1 (the
                 "no-skill" point of Hyndman & Koehler 2006 §3);
  - MASE > 1  ⇒  the model is *worse* than persistence — it actively
                 introduces noise into downstream fusion and must be
                 discounted.

Why MASE replaces R² as a trust proxy in this codebase
------------------------------------------------------
The legacy ``WBF_WEIGHT_MODE='trust_discount'`` uses Prophet's R² as
the trust score (Joesang 2016, Def. 14.6).  On RedeRio, this produced
a documented pathology (see ``docs/audit/trust_discount_r2_analysis.md``):
five Prophet metrics (``prophet_syn``, ``prophet_tcp``, ...) have
training-time R² < 0 because Prophet under-fits the bursty benign
behaviour of the very metrics that carry the low-volume attack
signature (SYN flood, slow connections).  The trust-discount fusion
then assigns the smallest weight to the most informative sources.

MASE fixes this:
  - **Scale-invariant** by construction (Hyndman & Koehler §2): the
    bursty character of a benign baseline does not penalise the metric.
  - **Bounded below** by 0; pathological R² < 0 cannot occur.
  - **Unambiguous skill semantics** (Hyndman 2006 §3 ; Murphy 1988):
    MASE = 1 is the "as good as Naive-1" point, MASE > 1 is "worse
    than Naive-1".

Trust-score map
---------------
The Joesang Def. 14.6 trust transitivity operator requires
``t ∈ [0, 1]`` with ``b' = t · b`` and ``u' = 1 − t·(1 − u)``.
We therefore expose

    trust = max(floor, 1 − α · MASE)

with the canonical choice α = 1 (skill-score interpretation:
``trust = 1 − MASE`` is positive iff the model beats Naive-1) and a
small numerical floor (default 0.05, matching ``TRUST_SCORE_FLOOR``).
The floor lets a misleading source contribute *something* to fusion
rather than being completely silenced; setting floor=0 produces full
silencing, also valid under Joesang Def. 14.6.

Critical reviewer constraint: ``trust ≤ 1`` always.  No source is
**amplified** by trust discounting; only **reduced** in proportion to
its skill deficit relative to Naive-1.

References
----------
- Hyndman, R. J. & Koehler, A. B. (2006). "Another look at measures
  of forecast accuracy." *International Journal of Forecasting*
  22(4), 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001
- Murphy, A. H. (1988). "Skill scores based on the mean square error
  and their relationships to the correlation coefficient."
  *Monthly Weather Review* 116(12), 2417–2424.
- Joesang, A. (2016). *Subjective Logic*. Springer. Def. 14.6
  (probability-sensitive trust discounting).
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def compute_mase(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Mean Absolute Scaled Error (Hyndman-Koehler 2006 Eq. 4).

    The Naive-1 baseline is computed from ``y_true`` only; ``y_pred``
    enters only the numerator.

    Edge cases:

      - If ``y_true`` and ``y_pred`` differ in length, ``ValueError``.
      - Pairs with non-finite ``y_true`` or ``y_pred`` are dropped.
      - If fewer than 2 valid pairs survive (cannot estimate Naive-1),
        return ``nan``.
      - If the Naive-1 denominator is below ``1e-12`` (the underlying
        series is essentially constant), return ``nan``.

    Returning ``nan`` rather than raising lets downstream callers
    decide on a default trust (typically ``TRUST_SCORE_FLOOR``).
    """
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    if y_true_arr.size != y_pred_arr.size:
        raise ValueError(
            f"compute_mase: length mismatch — len(y_true)={y_true_arr.size}, "
            f"len(y_pred)={y_pred_arr.size}"
        )

    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]

    if y_true_arr.size < 2:
        return math.nan

    numerator = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    denominator = float(np.mean(np.abs(np.diff(y_true_arr))))

    if denominator < 1e-12:
        return math.nan

    return numerator / denominator


def mase_to_trust(mase: float, alpha: float = 1.0, floor: float = 0.05) -> float:
    """Convert a MASE value to a Joesang-compatible trust score in
    ``[floor, 1]``.

    Skill-score interpretation:
        trust = max(floor, 1 − α · MASE)

    With ``α = 1.0`` (the default), ``trust > 0`` iff MASE < 1, i.e.
    the model is more informative than Naive-1.  Models with MASE ≥ 1
    are floored to ``floor``.

    A non-finite MASE (insufficient data, constant series) is treated
    as "no usable trust signal" and clipped to ``floor``.

    Critical invariants:
        - ``trust ∈ [floor, 1]`` for all finite inputs.
        - ``trust(0) = 1``, ``trust(1) = max(floor, 1 - α)``,
          ``trust(NaN) = floor``.
        - Monotonically non-increasing in MASE (no amplification
          of unreliable sources).
    """
    if not (isinstance(mase, (int, float)) and math.isfinite(float(mase))):
        return float(floor)
    raw = 1.0 - float(alpha) * float(mase)
    return float(max(float(floor), min(1.0, raw)))


def compute_mase_trust(y_true: Iterable[float], y_pred: Iterable[float],
                        alpha: float = 1.0,
                        floor: float = 0.05) -> tuple[float, float]:
    """One-shot helper: compute (mase, trust) for a (y_true, y_pred) pair.

    Useful for callers that want both the raw audit-trail value and the
    floored trust score in a single call.  See ``compute_mase`` and
    ``mase_to_trust`` for the individual semantics.
    """
    mase = compute_mase(y_true, y_pred)
    trust = mase_to_trust(mase, alpha=alpha, floor=floor)
    return mase, trust


__all__ = ["compute_mase", "mase_to_trust", "compute_mase_trust"]
