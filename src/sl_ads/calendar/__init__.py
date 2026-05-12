"""Calendar-aware utilities (regime classification, holiday lookup).

Public API:
    regime_of(timestamp, holidays=None) -> str
    REGIME_FN_SIGNATURE              -> versioned signature string
    REGIME_BUCKETS                   -> ordered tuple of bucket labels
"""
from sl_ads.calendar.regime import (
    REGIME_BUCKETS,
    REGIME_FN_SIGNATURE,
    regime_of,
    regime_of_series,
)

__all__ = [
    "REGIME_BUCKETS",
    "REGIME_FN_SIGNATURE",
    "regime_of",
    "regime_of_series",
]
