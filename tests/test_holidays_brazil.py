"""
test_holidays_brazil.py — Regression guard on the Brazilian holiday list
used by Prophet on the RedeRio dataset.

The trace covers 2025-10-13 to 2025-12-29 (plus 2026-01-01 carry-over).
We assert that every Brazilian *national* holiday in that window appears
in ``CONFIG["HOLIDAYS_LIST"]``.  University-specific closures
(27-Oct = Servidor Público UFRJ, 21-Nov = ponte rio, 26-31 Dec) are
allowed as additional entries.

Tracks TASK-15 in docs/audit/audit_verification_tracker.md.

Reference
---------
- ``holidays`` package (PyPI: ``holidays``) used as ground truth for the
  Brazilian national calendar.  Cited authoritative sources:
  Federal Law 662/1949 (Finados / Proclamação da República / Natal),
  Federal Law 6802/1980 (Nossa Senhora Aparecida),
  Federal Law 14759/2023 (Consciência Negra — national since 2024).
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# The 'holidays' package is part of the project requirements via Prophet's
# transitive dependencies; if a stripped-down environment is used we
# skip rather than fail.
holidays = pytest.importorskip("holidays")


from sl_ads.config import CONFIG  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Trace coverage window: any Brazilian national holiday whose date
# satisfies (DATA_START <= ds <= DATA_END) must appear in the config.
# ──────────────────────────────────────────────────────────────────────
DATA_START = date(2025, 10, 13)
# RedeRio runs through 2025-12-29; the first day of 2026 is also
# inside the test slice and must be covered.
DATA_END = date(2026, 1, 1)


def _config_holiday_dates() -> set[date]:
    """Return the set of dates declared in ``CONFIG['HOLIDAYS_LIST']``."""
    out: set[date] = set()
    for entry in CONFIG.get("HOLIDAYS_LIST", []):
        ds = entry.get("ds")
        if ds is None:
            continue
        # Accept both ``date`` objects and ISO strings ("YYYY-MM-DD").
        if isinstance(ds, date):
            out.add(ds)
        else:
            out.add(date.fromisoformat(str(ds)))
    return out


def _brazilian_national_holidays_in_window() -> dict[date, str]:
    cal = holidays.Brazil(years=[DATA_START.year, DATA_END.year])
    return {
        d: cal[d]
        for d in cal
        if DATA_START <= d <= DATA_END
    }


def test_at_least_one_holiday_declared():
    """Sanity check — RedeRio is *not* a clean-room dataset; at least one
    Brazilian holiday is expected in the window."""
    assert len(CONFIG.get("HOLIDAYS_LIST", [])) >= 1


def test_brazilian_national_holidays_are_covered():
    """Every national holiday in the data window must be declared."""
    declared = _config_holiday_dates()
    expected = _brazilian_national_holidays_in_window()
    missing = {d: name for d, name in expected.items() if d not in declared}
    assert not missing, (
        "The following Brazilian national holidays fall in the RedeRio "
        f"trace ({DATA_START} → {DATA_END}) but are absent from "
        f"CONFIG['HOLIDAYS_LIST']: {missing}.  Either add them or "
        "document a deliberate exclusion."
    )


def test_holiday_entries_have_required_keys():
    """Each entry must carry both ``ds`` and ``holiday`` keys; Prophet
    raises an opaque error otherwise."""
    for entry in CONFIG.get("HOLIDAYS_LIST", []):
        assert "ds" in entry, entry
        assert "holiday" in entry, entry


def test_no_duplicate_dates():
    """Duplicate dates would silently inflate Prophet's gradient — guard
    against accidental copy-paste regressions in the literal."""
    dates = [entry["ds"] for entry in CONFIG.get("HOLIDAYS_LIST", [])]
    assert len(dates) == len(set(dates)), (
        f"Duplicate ds values in HOLIDAYS_LIST: "
        f"{[d for d in dates if dates.count(d) > 1]}"
    )


def test_unified_label_strategy_is_used():
    """The design choice (TASK-15) is to use a single 'University_Closed'
    label across all closure days for parsimony.  Departure from this
    convention requires a config-level comment update; this test is the
    regression guard."""
    labels = {entry["holiday"] for entry in CONFIG.get("HOLIDAYS_LIST", [])}
    # We tolerate a small set in case future work splits the label into a
    # short controlled vocabulary; flag wide proliferation early.
    assert len(labels) <= 3, (
        f"HOLIDAYS_LIST uses {len(labels)} distinct labels {labels}; the "
        "documented design (TASK-15) keeps the label-set parsimonious to "
        "avoid Prophet over-fitting one coefficient per holiday on a "
        "78-day trace.  Update config.py docstring if this is intended."
    )
