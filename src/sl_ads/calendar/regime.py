"""regime.py — Calendar-aware regime classification (PATCH H2 / TASK-57).

Purpose
-------
Single source of truth for "which regime is this timestamp in?", consumed
by:

  * `train_models.py` to partition residuals before per-regime EVT
    calibration;
  * `compute_evidence.py` to dispatch per-window thresholds at inference
    time;
  * `evaluate_regime_fpr.py` (audit) to verify the realised FPR per
    bucket;
  * `paths.py` (sidecar A1.9) to persist and validate the regime-fn
    signature so a calibration-vs-runtime calendar drift hard-raises.

Partition (Option α, 2-bucket)
------------------------------

    ACTIVE  =  weekday × not(holiday) × hour ∈ [08, 18)
    QUIET   =  everything else (weekend, holiday, night, shoulder)

Justification.  The 2026-05-06 regime-FPR audit (`outputs/scientific_
hardening/regime_fpr_summary.json`) reports realised-FPR-to-target
ratios of 4.52× on weekday-daytime, 1.70× on the shoulder hours and
0× on every other regime.  The dominant overshoot is concentrated on
weekday-daytime; aggregating weekend / night / holiday into a single
`QUIET` bucket is faithful to the empirical distribution because all
three regimes share a near-zero FPR profile.  A finer 4-bucket
partition was considered and rejected (see
`docs/review/calendar_evt_design.md` §2.3): the smallest bucket
(`HOLIDAY`, ~1.3k windows) would borderline-trigger the EVT
peak-count fallback for some metrics, while the gain over the
2-bucket partition is empirically marginal because the three quiet
buckets have indistinguishable FPR profiles.

Signature
---------
``REGIME_FN_SIGNATURE`` is the immutable label of this implementation
(partition shape, hour boundaries, day-of-week interpretation).  The
sidecar persists it; ``paths.validate_threshold_sidecar_config``
hard-raises if the runtime signature differs from the calibration
signature.  Bumping the signature is therefore the canonical
mechanism for shipping a new partition (e.g. moving to Option β):
the version part of the string changes, calibration drift is
detected, and the operator must retrain.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Public constants
# ─────────────────────────────────────────────────────────────────────
REGIME_FN_SIGNATURE = "weekday_x_daytime_x_holiday/v1@2026-05-07"
"""Versioned signature of the regime function.  Persisted in the
threshold sidecar; mismatches between calibration and runtime trigger
``RuntimeError("[A1.9] Threshold sidecar/config mismatch ...")``.
Update this string whenever the regime semantics change (e.g. moving
the daytime hour boundaries or switching to a finer partition)."""

REGIME_BUCKETS: tuple[str, ...] = ("ACTIVE", "QUIET")
"""Ordered tuple of bucket labels emitted by ``regime_of``.  The order
is canonical: persisted artefacts (CSV columns, sidecar dicts) preserve
this ordering for stable diffs."""

_DAYTIME_START_HOUR = 8
_DAYTIME_END_HOUR = 18  # exclusive: window with hour=18 is QUIET (shoulder)


# ─────────────────────────────────────────────────────────────────────
# Holiday lookup
# ─────────────────────────────────────────────────────────────────────
def _normalise_holiday_dates(holidays: Optional[Iterable]) -> set:
    """Convert any of (None, list of dates, list of {ds: ...} dicts,
    pandas-readable container) into a frozen set of ``datetime.date``.
    Empty ⇒ no holidays.
    """
    if holidays is None:
        return set()
    out = set()
    for entry in holidays:
        if isinstance(entry, dict):
            ds = entry.get("ds")
            if ds is None:
                continue
            out.add(pd.Timestamp(ds).date())
        else:
            out.add(pd.Timestamp(entry).date())
    return out


# ─────────────────────────────────────────────────────────────────────
# Public API — single timestamp
# ─────────────────────────────────────────────────────────────────────
def regime_of(timestamp, holidays: Optional[Iterable] = None) -> str:
    """Return the regime label (``"ACTIVE"`` or ``"QUIET"``) for one
    timestamp.

    Parameters
    ----------
    timestamp : pandas-coercible (str, datetime, pd.Timestamp, ...)
    holidays : optional iterable of holiday dates.  Each entry can be
        a date-like, a pandas-coercible string, or a ``{"ds": "YYYY-MM-DD"}``
        dict (matching the shape of ``CONFIG['HOLIDAYS_LIST']``).  Pass
        ``None`` to skip holiday awareness; the function still
        partitions on weekend/daytime alone.

    Returns
    -------
    str
        Either ``"ACTIVE"`` (weekday × not(holiday) × hour ∈ [8, 18))
        or ``"QUIET"`` (everything else).
    """
    ts = pd.Timestamp(timestamp)
    holiday_dates = _normalise_holiday_dates(holidays)

    is_weekend = ts.dayofweek >= 5
    is_holiday = ts.date() in holiday_dates
    is_daytime = _DAYTIME_START_HOUR <= ts.hour < _DAYTIME_END_HOUR

    if is_weekend or is_holiday or not is_daytime:
        return "QUIET"
    return "ACTIVE"


# ─────────────────────────────────────────────────────────────────────
# Public API — vectorised over a pandas Series of timestamps
# ─────────────────────────────────────────────────────────────────────
def regime_of_series(timestamps: pd.Series,
                      holidays: Optional[Iterable] = None) -> pd.Series:
    """Vectorised version of :func:`regime_of` for a pandas Series.

    Returns a Series of bucket labels with the same index as
    ``timestamps``.  Significantly faster than ``Series.apply(regime_of)``
    on RedeRio-sized inputs (~200k rows) because the date / hour /
    dayofweek extractions are vectorised once and the holiday lookup is
    done as an isin-against-a-set.
    """
    ts = pd.to_datetime(pd.Series(timestamps))
    holiday_dates = _normalise_holiday_dates(holidays)

    dow = ts.dt.dayofweek
    hour = ts.dt.hour
    date = ts.dt.date

    is_weekend = dow >= 5
    is_holiday = pd.Series(date).isin(holiday_dates).reindex(ts.index, fill_value=False)
    # Reindex defensively above; pd.Series(date) sometimes resets the index.
    if not is_holiday.index.equals(ts.index):
        is_holiday = pd.Series(
            [d in holiday_dates for d in date], index=ts.index
        )
    is_daytime = (hour >= _DAYTIME_START_HOUR) & (hour < _DAYTIME_END_HOUR)

    is_quiet = is_weekend | is_holiday | (~is_daytime)
    return is_quiet.map({True: "QUIET", False: "ACTIVE"})


# ─────────────────────────────────────────────────────────────────────
# Diagnostic helper
# ─────────────────────────────────────────────────────────────────────
def regime_counts(timestamps: pd.Series,
                   holidays: Optional[Iterable] = None) -> dict:
    """Return ``{bucket: count}`` for a Series of timestamps.  Useful
    in tests and in the train-time diagnostic printout.
    """
    s = regime_of_series(timestamps, holidays=holidays)
    out = {b: 0 for b in REGIME_BUCKETS}
    counts = s.value_counts().to_dict()
    out.update({k: int(v) for k, v in counts.items() if k in out})
    return out


__all__ = [
    "REGIME_FN_SIGNATURE",
    "REGIME_BUCKETS",
    "regime_of",
    "regime_of_series",
    "regime_counts",
]
