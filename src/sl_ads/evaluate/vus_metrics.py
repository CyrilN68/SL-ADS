"""
vus_metrics.py — Range-aware AUC and Volume Under the Surface (VUS) metrics
for time-series anomaly detection.

Purpose
-------
Standard point-wise AUC-ROC / AUC-PR are known to be misleading for time
series anomaly detection (Tatbul *et al.* 2018, NeurIPS) because they
penalise a detector that raises an alarm one step before/after the
ground-truth window even though, operationally, such a detection is
correct.  The community has therefore moved to **range-aware** metrics:

- **R-AUC-ROC / R-AUC-PR** (Paparrizos 2022, §3) — point-wise AUC computed
  on labels whose anomaly runs have been extended by a buffer ``L`` on
  each side, so a detection inside the buffer counts as a true positive.
- **VUS-ROC / VUS-PR** (Paparrizos 2022, §4) — *Volume Under the Surface*
  obtained by integrating R-AUC over buffer sizes ``L ∈ [0, L_max]``.
  This eliminates the arbitrary choice of a single buffer and reports
  the area under the (threshold, buffer) → metric surface.
- **Existence-based recall** (Tatbul 2018, Eq. 5) — fraction of true
  anomaly ranges that overlap at least one positive prediction; provided
  as a complement, not a substitute, since it ignores false alarms.

This module is dependency-light (numpy + sklearn) and provides a
``--self-test`` driver that can be run from a clean checkout.

References
----------
- Paparrizos, J., Boniol, P., Palpanas, T., Tsay, R., Elmore, A.,
  Franklin, M. (2022). "Volume Under the Surface: A New Accuracy
  Evaluation Measure for Time-Series Anomaly Detection."
  *Proc. VLDB Endow.* 15(11): 2774-2787.
- Tatbul, N., Lee, T. J., Zdonik, S., Alam, M., Gottschlich, J. (2018).
  "Precision and Recall for Time Series." *NeurIPS 2018*.
- Wu, R., Keogh, E. (2021). "Current Time Series Anomaly Detection
  Benchmarks are Flawed and are Creating the Illusion of Progress."
  *IEEE TKDE* (preprint).

Public API
----------
- :func:`find_anomaly_ranges` — list of inclusive ``(start, end)``
  intervals corresponding to maximal runs of 1s in a binary label vector.
- :func:`extend_anomaly_ranges` — widen each anomaly run by a buffer.
- :func:`range_auc_roc` / :func:`range_auc_pr` — R-AUC at a single
  buffer ``L``.
- :func:`vus_roc` / :func:`vus_pr` — Volume Under Surface (trapezoidal
  integration of R-AUC over ``L``).
- :func:`existence_recall` — Tatbul-style range-existence recall.
- :func:`vus_summary` — convenience wrapper returning every metric at
  once for a single ``(y_true, y_score)`` pair.

Self-test
---------
    python -m sl_ads.evaluate.vus_metrics --self-test

Tracks TASK-54 of ``docs/audit/audit_verification_tracker.md``.
"""
from __future__ import annotations

import math
import sys
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


# ──────────────────────────────────────────────────────────────────────
# Range utilities
# ──────────────────────────────────────────────────────────────────────
def find_anomaly_ranges(y_true: Sequence[int]) -> List[Tuple[int, int]]:
    """Return inclusive ``(start, end)`` pairs for each maximal run of 1s.

    Examples
    --------
    >>> find_anomaly_ranges([0, 1, 1, 0, 0, 1, 0])
    [(1, 2), (5, 5)]

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Binary anomaly labels.

    Returns
    -------
    list of (int, int)
        Each entry is ``(start_inclusive, end_inclusive)``.  Empty list
        if ``y_true`` contains no 1s.
    """
    y = np.asarray(y_true).astype(np.int8).ravel()
    if y.size == 0:
        return []
    diff = np.diff(np.concatenate(([0], y, [0])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1  # inclusive end
    return list(zip(starts.tolist(), ends.tolist()))


def extend_anomaly_ranges(y_true: Sequence[int], buffer: int) -> np.ndarray:
    """Return a copy of ``y_true`` whose every anomaly run is widened
    by ``buffer`` indices on each side (capped at sequence boundaries).

    This realises the buffered ground truth ``Y_ε`` of Paparrizos
    (2022) §3.1.  When ``buffer == 0`` the input is returned unchanged
    (modulo dtype cast).

    Parameters
    ----------
    y_true : array-like of {0, 1}
    buffer : int, ``>= 0``
        Number of indices added before each run start and after each
        run end.

    Returns
    -------
    numpy.ndarray of dtype uint8
    """
    if buffer < 0:
        raise ValueError(f"buffer must be >= 0, got {buffer}")
    y = np.asarray(y_true).astype(np.uint8).ravel().copy()
    if buffer == 0 or y.sum() == 0:
        return y
    n = y.size
    diff = np.diff(np.concatenate(([0], y, [0])).astype(int))
    starts = np.where(diff == 1)[0]
    ends_excl = np.where(diff == -1)[0]  # exclusive
    out = y.copy()
    for a, b in zip(starts, ends_excl):
        out[max(0, a - buffer): min(n, b + buffer)] = 1
    return out


# ──────────────────────────────────────────────────────────────────────
# Range-aware AUC at a single buffer
# ──────────────────────────────────────────────────────────────────────
def range_auc_roc(
    y_true: Sequence[int],
    y_score: Sequence[float],
    buffer: int,
) -> float:
    """Range-AUC-ROC at buffer ``L = buffer`` (Paparrizos 2022, §3).

    Computes the standard AUC-ROC on labels whose anomaly runs have
    been extended by ``buffer`` indices on each side.  Returns
    ``float('nan')`` if the buffered labels are degenerate (all 0 or
    all 1).
    """
    y_buf = extend_anomaly_ranges(y_true, buffer)
    n_pos = int(y_buf.sum())
    if n_pos == 0 or n_pos == y_buf.size:
        return float("nan")
    return float(roc_auc_score(y_buf, np.asarray(y_score, dtype=float)))


def range_auc_pr(
    y_true: Sequence[int],
    y_score: Sequence[float],
    buffer: int,
) -> float:
    """Range-AUC-PR at buffer ``L = buffer`` (Paparrizos 2022, §3).

    Average precision (Davis & Goadrich 2006) on extended labels.
    Returns ``float('nan')`` if the buffered label vector contains no
    positives.
    """
    y_buf = extend_anomaly_ranges(y_true, buffer)
    if y_buf.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_buf, np.asarray(y_score, dtype=float)))


# ──────────────────────────────────────────────────────────────────────
# Volume Under Surface
# ──────────────────────────────────────────────────────────────────────
def _vus(
    y_true: Sequence[int],
    y_score: Sequence[float],
    max_buffer: int,
    n_steps: int,
    auc_fn,
) -> float:
    """Trapezoidal integration of ``auc_fn(buffer)`` over
    ``buffer ∈ [0, max_buffer]`` evaluated on ``n_steps`` equispaced
    integer buffer values, normalised by the buffer range so the
    return value is on the same [0, 1] scale as a plain AUC.
    """
    if max_buffer < 0:
        raise ValueError(f"max_buffer must be >= 0, got {max_buffer}")
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")
    buffers = np.unique(
        np.linspace(0, max_buffer, n_steps).round().astype(int)
    )
    aucs = np.array([auc_fn(y_true, y_score, int(L)) for L in buffers])
    valid = np.isfinite(aucs)
    if valid.sum() < 2:
        # Cannot integrate meaningfully — fall back to a single-point
        # average if at least one finite value exists.
        return float(np.nanmean(aucs)) if valid.any() else float("nan")
    bv = buffers[valid].astype(float)
    av = aucs[valid].astype(float)
    span = bv[-1] - bv[0]
    if span <= 0:
        return float(av.mean())
    # NumPy 2.0 renamed ``trapz`` to ``trapezoid``; keep a graceful fallback
    # so the module works on both 1.x and 2.x.
    _trap = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    return float(_trap(av, bv) / span)


def vus_roc(
    y_true: Sequence[int],
    y_score: Sequence[float],
    max_buffer: int,
    n_steps: int = 11,
) -> float:
    """Volume Under Surface for ROC (Paparrizos 2022, §4).

    Equivalent to ``∫₀^{L_max} R-AUC-ROC(L) dL / L_max`` evaluated by
    the trapezoidal rule on ``n_steps`` integer buffer values.

    Parameters
    ----------
    y_true, y_score : array-like, length N
    max_buffer : int, ``>= 0``
        Upper bound of the buffer range.  In Paparrizos 2022 the
        recommended default is the median anomaly-run length.
    n_steps : int, default 11
        Number of buffer values sampled in ``[0, max_buffer]`` (after
        ``round`` + ``unique`` may collapse to fewer for small max_buffer).

    Returns
    -------
    float in ``[0, 1]`` or ``nan`` if every buffer yields a degenerate
    label vector.
    """
    return _vus(y_true, y_score, max_buffer, n_steps, range_auc_roc)


def vus_pr(
    y_true: Sequence[int],
    y_score: Sequence[float],
    max_buffer: int,
    n_steps: int = 11,
) -> float:
    """Volume Under Surface for PR (Paparrizos 2022, §4).  See
    :func:`vus_roc` for parameter semantics.
    """
    return _vus(y_true, y_score, max_buffer, n_steps, range_auc_pr)


# ──────────────────────────────────────────────────────────────────────
# Existence-based recall (Tatbul 2018)
# ──────────────────────────────────────────────────────────────────────
def existence_recall(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    """Tatbul *et al.* 2018, Eq. 5 — the *existence-based* recall is
    the fraction of true anomaly ranges that overlap at least one
    positive prediction.

    This complements range-AUC: it ignores false alarms entirely and
    therefore must be reported alongside a precision-aware metric.

    Returns ``float('nan')`` if ``y_true`` contains no positive range.
    """
    ranges = find_anomaly_ranges(y_true)
    if not ranges:
        return float("nan")
    yp = np.asarray(y_pred).astype(bool).ravel()
    detected = sum(1 for a, b in ranges if yp[a: b + 1].any())
    return detected / len(ranges)


# ──────────────────────────────────────────────────────────────────────
# One-shot convenience wrapper
# ──────────────────────────────────────────────────────────────────────
def vus_summary(
    y_true: Sequence[int],
    y_score: Sequence[float],
    y_pred: Sequence[int] | None = None,
    max_buffer: int | None = None,
    n_steps: int = 11,
) -> dict:
    """Return every range-aware metric in a single dict.

    If ``max_buffer`` is ``None``, it is set to the median anomaly run
    length in ``y_true`` (rounded up), per Paparrizos 2022's empirical
    recommendation.  If ``y_pred`` is provided, the existence-based
    recall is also computed.
    """
    y_true = np.asarray(y_true).astype(np.int8).ravel()
    if max_buffer is None:
        runs = find_anomaly_ranges(y_true)
        if runs:
            lens = np.array([b - a + 1 for a, b in runs])
            max_buffer = int(math.ceil(float(np.median(lens))))
        else:
            max_buffer = 0

    out = {
        "n": int(y_true.size),
        "n_anomalies": int(y_true.sum()),
        "n_ranges": len(find_anomaly_ranges(y_true)),
        "max_buffer": int(max_buffer),
        "n_buffer_steps": int(n_steps),
        "range_auc_roc_at_max": range_auc_roc(y_true, y_score, max_buffer),
        "range_auc_pr_at_max": range_auc_pr(y_true, y_score, max_buffer),
        "vus_roc": vus_roc(y_true, y_score, max_buffer, n_steps=n_steps),
        "vus_pr": vus_pr(y_true, y_score, max_buffer, n_steps=n_steps),
    }
    if y_pred is not None:
        out["existence_recall"] = existence_recall(y_true, y_pred)
    return out


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────
def _self_test() -> int:
    print("[TEST] vus_metrics.py — self-test")

    # 1. find_anomaly_ranges
    rs = find_anomaly_ranges([0, 1, 1, 0, 0, 1, 0])
    assert rs == [(1, 2), (5, 5)], rs
    print("   [OK] find_anomaly_ranges")

    rs_empty = find_anomaly_ranges([0, 0, 0])
    assert rs_empty == [], rs_empty
    print("   [OK] find_anomaly_ranges (empty case)")

    # 2. extend_anomaly_ranges
    y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
    yb1 = extend_anomaly_ranges(y, 1)
    assert yb1.tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 0], yb1.tolist()
    yb0 = extend_anomaly_ranges(y, 0)
    assert (yb0 == y).all()
    yb_clip = extend_anomaly_ranges([1, 0, 0, 0, 1], 100)
    assert (yb_clip == 1).all()
    print("   [OK] extend_anomaly_ranges")

    # 3. Perfect detector — R-AUC-ROC and R-AUC-PR finite at any buffer
    # that does not saturate the labels.  We use a longer sequence so
    # the buffer cannot cover every index.
    y_true = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                       0, 0, 0, 0])
    y_score_perfect = y_true.astype(float)
    for L in (0, 1, 3):
        y_buf = extend_anomaly_ranges(y_true, L)
        if y_buf.sum() in (0, y_buf.size):
            continue  # buffer saturates labels → degenerate (test inv. is upstream)
        a_roc = range_auc_roc(y_true, y_score_perfect, L)
        a_pr = range_auc_pr(y_true, y_score_perfect, L)
        # Buffer L>0 introduces new positives that the perfect detector
        # did not flag with score 1; the score there is 0.  AUC therefore
        # degrades from 1.0 (L=0) but should remain >= 0.5 (better than
        # chance) since core positives are still ranked above negatives.
        assert math.isfinite(a_roc) and a_roc >= 0.5, (L, a_roc)
        assert math.isfinite(a_pr) and a_pr > 0.0, (L, a_pr)
    a_roc0 = range_auc_roc(y_true, y_score_perfect, 0)
    assert math.isclose(a_roc0, 1.0, abs_tol=1e-9), a_roc0
    print(f"   [OK] perfect detector: R-AUC-ROC(L=0) = {a_roc0:.3f}")

    # 4. Random scores: AUC ≈ 0.5 ± 0.1 on a long sequence.
    rng = np.random.default_rng(0)
    y_rand_true = rng.integers(0, 2, 2000)
    y_rand_score = rng.random(2000)
    auc_rand = range_auc_roc(y_rand_true, y_rand_score, 0)
    assert 0.4 < auc_rand < 0.6, auc_rand
    print(f"   [OK] random detector: AUC = {auc_rand:.3f}")

    # 5. Total miss: detector outputs anti-correlated scores → AUC ≈ 0.
    y_score_bad = 1.0 - y_true.astype(float)
    auc_bad = range_auc_roc(y_true, y_score_bad, 0)
    assert auc_bad == 0.0, auc_bad
    print(f"   [OK] worst-case detector: AUC = {auc_bad:.3f}")

    # 6. Range-existence recall.
    # y_true has 2 ranges: (2,4) and (12,13).  y_pred_partial fires
    # inside BOTH (recall = 1.0); y_pred_one fires inside only the second
    # range (recall = 0.5).
    y_pred_partial = np.zeros_like(y_true)
    y_pred_partial[2] = 1   # inside range (2,4)
    y_pred_partial[13] = 1  # inside range (12,13)
    er = existence_recall(y_true, y_pred_partial)
    assert math.isclose(er, 1.0, abs_tol=1e-9), er
    y_pred_one = np.zeros_like(y_true)
    y_pred_one[13] = 1
    er2 = existence_recall(y_true, y_pred_one)
    assert math.isclose(er2, 0.5, abs_tol=1e-9), er2
    print(f"   [OK] existence_recall: 1.0 / 0.5")

    # 7. VUS — for the perfect detector at L_max=2 it should remain
    # close to 1 (score still ranks core positives above any negative).
    vusr = vus_roc(y_true, y_score_perfect, max_buffer=2, n_steps=3)
    vusp = vus_pr(y_true, y_score_perfect, max_buffer=2, n_steps=3)
    assert vusr >= 0.7, vusr
    assert vusp > 0.5, vusp
    print(f"   [OK] VUS perfect detector: VUS-ROC={vusr:.3f}, VUS-PR={vusp:.3f}")

    # 8. VUS on a sparse-anomaly random detector.  We construct labels
    # with ~5 % anomaly density in well-separated ranges so the buffered
    # labels do not saturate; under random scores VUS-ROC ≈ 0.5 and
    # VUS-PR ≈ buffered base rate.  The buffered base rate is at most
    # ~10 % in this configuration, far from the all-positive limit.
    rng2 = np.random.default_rng(1)
    N = 2000
    y_sparse = np.zeros(N, dtype=np.int8)
    for start in rng2.integers(50, N - 50, size=20):
        y_sparse[start: start + 5] = 1  # 20 ranges of length 5 → 5 % density
    score_sparse = rng2.random(N)
    vusr_sparse = vus_roc(y_sparse, score_sparse, max_buffer=5, n_steps=6)
    vusp_sparse = vus_pr(y_sparse, score_sparse, max_buffer=5, n_steps=6)
    # The buffered base rate at L_max=5 is ≤ (5 + 2·5)/period; we report
    # it for transparency rather than encoding a tight bound.
    buf_max = extend_anomaly_ranges(y_sparse, 5)
    buf_rate_max = float(buf_max.mean())
    assert 0.40 < vusr_sparse < 0.60, vusr_sparse
    # AP under random scores must lie between the unbuffered base rate
    # (lower bound, L=0) and the buffered base rate (upper bound, L=L_max).
    base_rate0 = float(y_sparse.mean())
    assert base_rate0 - 0.05 < vusp_sparse < buf_rate_max + 0.05, (
        vusp_sparse, base_rate0, buf_rate_max
    )
    print(f"   [OK] VUS random sparse detector: VUS-ROC={vusr_sparse:.3f} "
          f"(target 0.5), VUS-PR={vusp_sparse:.3f} "
          f"(in [{base_rate0:.3f}, {buf_rate_max:.3f}])")

    # 9. vus_summary.
    s = vus_summary(y_true, y_score_perfect, y_pred=y_pred_partial)
    assert s["n_ranges"] == 2
    assert s["n_anomalies"] == 5
    assert math.isfinite(s["vus_roc"])
    assert math.isfinite(s["vus_pr"])
    assert math.isclose(s["existence_recall"], 1.0, abs_tol=1e-9)
    print(f"   [OK] vus_summary: {s}")

    # 10. Buffer extension widens TPR coverage.  A detector that fires
    # *one step before* each true range scores 0 (or worse) on raw
    # range_auc_roc(L=0) but should jump above chance for L >= 1, since
    # the high-score points become true positives in the buffered labels.
    # We use a long, sparse sequence so the buffer cannot saturate.
    y_true_strict = np.zeros(40, dtype=np.int8)
    for s in (5, 20, 32):
        y_true_strict[s: s + 2] = 1
    y_score_off = np.zeros(40, dtype=float)
    for a, _b in find_anomaly_ranges(y_true_strict):
        if a > 0:
            y_score_off[a - 1] = 1.0
    auc_l0 = range_auc_roc(y_true_strict, y_score_off, 0)
    auc_l1 = range_auc_roc(y_true_strict, y_score_off, 1)
    # L=0: every "1" prediction is at a NEGATIVE → AUC <= 0.5.
    # L=1: the off-by-one points fall inside the buffer → AUC > 0.5.
    assert auc_l0 <= 0.5 + 1e-9, auc_l0
    assert auc_l1 > 0.5, (auc_l0, auc_l1)
    assert auc_l1 > auc_l0, (auc_l0, auc_l1)
    print(f"   [OK] off-by-one detector: AUC L=0:{auc_l0:.3f} → "
          f"L=1:{auc_l1:.3f} (rises above chance under buffering)")

    print("[OK] vus_metrics.py — ALL PASS")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return _self_test()
    print(__doc__)
    print("Use --self-test to run the validation suite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
