"""
qualif_filters.py — Shared filters for downstream qualification analyses
========================================================================

Centralised helpers for PATCH m-03/F20 + m-09/F30 + m-04/F21.

Two concerns are addressed:

m-03/F20  + m-09/F30
--------------------
Distinguish between two kinds of ``gate_open`` rows in the
qualification CSV:

1. **``gate_open=True AND qual_status ∈ {'qualified','autre_anomalie'}``** —
   the SBN actually ran the type qualifier and produced a top-1 /
   uncertainty / novelty signal.  These rows are the only ones whose
   ``u_sbn``, ``novelty_lr``, ``novelty_entropy``, and ``top1_type``
   values are *meaningful*.

2. **``gate_open=True AND qual_status='no_groups'``** — the detector
   fired but the qualifier could not run because no protocol/directional
   group had enough data in that window.  These rows exist in the CSV
   but their ``u_sbn=1.0`` and ``novelty_lr=1.0`` are sentinel values
   injected by ``qualify_anomaly_sbn._empty_result()``, not real
   inferences.

Any downstream statistics over ``u_sbn`` / ``novelty_lr`` /
``novelty_entropy`` / ``top1_type`` MUST exclude the second class to
avoid biasing the mean uncertainty upwards and polluting the novelty
AUC with constant values.

Detection-level metrics (FAR, MCC, TP/FP/TN/FN based on ``gate_open``
only) are *unaffected* — gate firing on a window the qualifier cannot
classify is still a bona fide detection event (or false alarm when it
happens during normal traffic).

m-04/F21
--------
``NETWORK_OUTAGE`` events are data-plane failures rather than
adversarial attacks.  Downstream evaluation should keep them in a
separate bucket so that (a) FAR is not computed on outages as if they
were "normal traffic" and (b) the attack-precision tables are not
conflated with outage detection.  The helper :func:`is_outage_event`
returns True for catalog entries whose ``expected`` or
``expected_qualif`` label is ``"NETWORK_OUTAGE"``.

Usage
-----
    import pandas as pd
    from sl_ads.qualif_filters import qualified_mask, is_outage_event

    # Filter qualification-only rows
    mask = qualified_mask(df)
    df_qual = df[mask]
    u_mean  = df_qual["u_sbn"].mean()

    # Partition catalog
    attack_events = [e for e in ALL_EVENTS if not is_outage_event(e)]
    outage_events = [e for e in ALL_EVENTS if is_outage_event(e)]
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import pandas as pd

__all__ = [
    "qualified_mask",
    "split_qualification_rows",
    "is_outage_event",
    "partition_catalog_by_outage",
    "QUAL_STATUS_NORMAL",
    "QUAL_STATUS_QUALIFIED",
    "QUAL_STATUS_AUTRE",
    "QUAL_STATUS_NO_GROUPS",
    "OUTAGE_TYPE",
]

# Canonical qual_status values used by qualify_anomaly_sbn.py.
QUAL_STATUS_NORMAL     = "normal"
QUAL_STATUS_QUALIFIED  = "qualified"
QUAL_STATUS_AUTRE      = "autre_anomalie"
QUAL_STATUS_NO_GROUPS  = "no_groups"

# Canonical label for network-outage events in catalogs.
OUTAGE_TYPE = "NETWORK_OUTAGE"


def qualified_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean Series selecting only qualification-ready rows.

    A row is "qualification-ready" if ``gate_open=True`` AND the
    qualifier actually ran — i.e. ``qual_status != "no_groups"``.  If
    the ``qual_status`` column is missing (legacy CSV), falls back to
    ``gate_open`` alone with a warning semantic (no ``no_groups``
    marker means the CSV was produced by an older ``qualify_*`` that
    did not distinguish the two cases).

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``qualify_anomaly_sbn.py`` (or a compatible
        qualifier) containing at minimum a ``gate_open`` column.

    Returns
    -------
    pd.Series (dtype=bool)
        True for rows that should participate in downstream statistics
        over ``u_sbn`` / ``novelty_lr`` / ``top1_type`` / entropy.
    """
    if "gate_open" not in df.columns:
        # No detector gate — all rows count (legacy compat).
        return pd.Series(True, index=df.index)
    gate = df["gate_open"].astype(bool)
    if "qual_status" not in df.columns:
        return gate
    return gate & (df["qual_status"].astype(str) != QUAL_STATUS_NO_GROUPS)


def split_qualification_rows(df: pd.DataFrame) -> tuple[pd.DataFrame,
                                                           pd.DataFrame,
                                                           pd.DataFrame]:
    """Split the qualification CSV into three disjoint subsets.

    Returns
    -------
    qualified : rows with gate_open=True AND qual_status ∈ {qualified, autre_anomalie}
    no_groups : rows with gate_open=True AND qual_status == 'no_groups'
    closed    : rows with gate_open=False (detector did not fire)

    The three DataFrames are views into ``df`` and their row counts
    sum to ``len(df)``.
    """
    qual = qualified_mask(df)
    if "gate_open" in df.columns:
        gate = df["gate_open"].astype(bool)
    else:
        gate = pd.Series(True, index=df.index)

    closed    = df[~gate]
    no_groups = df[gate & ~qual]
    qualified = df[qual]
    return qualified, no_groups, closed


# ----------------------------------------------------------------------
# m-04 / F21 : NETWORK_OUTAGE taxonomy separation
# ----------------------------------------------------------------------
def is_outage_event(event: Mapping) -> bool:
    """Return True iff this catalog entry is a NETWORK_OUTAGE.

    Recognised fields (checked in order):
        ``type``, ``expected``, ``expected_qualif``, ``name``.

    The lookup is case-insensitive and tolerates several name shapes:

    * Exact match (after upper-case strip):  ``NETWORK_OUTAGE``.
    * Trailing occurrence markers: ``NETWORK_OUTAGE_R2`` — the
      ``_R<digit>`` suffix is stripped before comparison.
    * Compound identifiers starting with the canonical prefix followed
      by an underscore: ``NETWORK_OUTAGE_DEC1617``, ``NETWORK_OUTAGE_NOV17``
      (the bucket names used in ``REAL_ATTACKS``), with or without a
      trailing ``_R<digit>``.

    The motivation: config.py uses the compound form as a *bucket key*
    (``REAL_ATTACKS['NETWORK_OUTAGE_DEC1617']``) while individual entries
    carry the canonical form via ``expected_qualif``. Downstream code may
    pass either shape as ``event['type']`` or ``event['name']`` depending
    on the call site (``evaluate_real`` synthesises a dict from the bucket
    key).  We want both routes to land in the outage bucket.
    """
    for key in ("type", "expected", "expected_qualif", "name"):
        v = event.get(key) if isinstance(event, Mapping) else None
        if v is None:
            continue
        # Normalise: upper-case, strip, drop trailing occurrence tag.
        s = str(v).strip().upper()
        # Remove "_R2", "_R3", ... trailing occurrence markers (loop for
        # defensive handling of nested markers like "_R2_R3").
        while len(s) >= 3 and s[-1].isdigit() and s[-3] == "_" and s[-2] == "R":
            s = s[:-3]
        # Canonical match OR compound-name match (``NETWORK_OUTAGE_*``).
        if s == OUTAGE_TYPE or s.startswith(OUTAGE_TYPE + "_"):
            return True
    return False


def partition_catalog_by_outage(
        events: Iterable[Mapping],
) -> tuple[list, list]:
    """Partition a catalog iterable into (attack_events, outage_events).

    Preserves the original ordering within each partition so that
    downstream per-event tables remain deterministic.
    """
    attacks: list = []
    outages: list = []
    for ev in events:
        if is_outage_event(ev):
            outages.append(ev)
        else:
            attacks.append(ev)
    return attacks, outages
