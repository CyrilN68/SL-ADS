"""
axelsson_ppv.py — Per-attack base-rate fallacy analysis.

Axelsson (2000, TISSEC) showed that even an intrusion detector with a
nearly perfect true-positive rate (TPR) and a tiny false-alarm rate
(FPR) can have a *useless* positive predictive value (PPV) when the
prior probability of an attack is very small.  Concretely, by Bayes'
theorem,

                           TPR · π
        PPV = ─────────────────────────────────
              TPR · π + FPR · (1 − π)

where ``π`` is the prior probability of an attack on the *evaluation
unit* (here: a network-traffic window).  If ``π = 1e-5`` and the
detector reports TPR=0.99 and FPR=0.01, then ``PPV ≈ 1.0e-3`` — most
alarms are false.  Reporting only F1 / accuracy hides this collapse
because they all weight the very large negative-class population
identically.

This module provides the *operational* analysis a reviewer of an IDS
publication will look for:

1. The Bayesian PPV formula (and its FPR-target inverse),
2. A per-attack-type table that, for every entry in
   ``CONFIG["ATTACK_CATALOG"]``, reports the attack's empirical base
   rate, TPR at the operating threshold, the global FPR, the resulting
   PPV, and the FPR that would be required to obtain a clinical-grade
   PPV (default target = 0.5),
3. A markdown report ready to drop into a paper appendix.

References
----------
- Axelsson, S. (2000). "The base-rate fallacy and the difficulty of
  intrusion detection."  *ACM Transactions on Information and System
  Security* (TISSEC), 3(3), 186-205.
- Lippmann, R., Haines, J., Fried, D., Korba, J., Das, K. (2000).
  "The 1999 DARPA off-line intrusion detection evaluation."
  *Computer Networks*, 34(4), 579-595. — methodological baseline
  for per-attack reporting.

Public API
----------
- :func:`bayesian_ppv` — PPV from TPR, FPR, base rate.
- :func:`min_fpr_for_ppv` — the maximum FPR that still yields a target
  PPV given TPR and base rate.
- :func:`min_tpr_for_ppv` — the minimum TPR for a target PPV at a fixed
  FPR (useful when the operating threshold is set by the FPR budget).
- :func:`per_attack_ppv_table` — produces the per-attack DataFrame.
- :func:`format_axelsson_md` — renders a markdown report from the table.

Self-test::

    python -m sl_ads.evaluate.axelsson_ppv --self-test

Tracks TASK-18 in docs/audit/audit_verification_tracker.md.
"""
from __future__ import annotations

import argparse
import math
import sys
from typing import Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from sl_ads.stats.residual_correlation import newey_west_eff_n
except ImportError:  # pragma: no cover - standalone fallback
    newey_west_eff_n = None


# ──────────────────────────────────────────────────────────────────────
# Bayesian core
# ──────────────────────────────────────────────────────────────────────
def bayesian_ppv(tpr: float, fpr: float, base_rate: float) -> float:
    """Positive predictive value (precision) from Bayes' theorem.

    .. math::
        \\mathrm{PPV} = \\frac{TPR \\cdot \\pi}
                              {TPR \\cdot \\pi + FPR \\cdot (1 - \\pi)}

    Parameters
    ----------
    tpr : float in [0, 1]
        True positive rate (sensitivity / recall).
    fpr : float in [0, 1]
        False positive rate (false alarm rate).
    base_rate : float in [0, 1]
        Prior probability ``π`` of an attack on a randomly selected
        evaluation unit (a window in this codebase).

    Returns
    -------
    float in [0, 1]
        ``nan`` when both terms of the denominator are zero
        (degenerate detector with no positive output at all).
    """
    if not (0.0 <= tpr <= 1.0):
        raise ValueError(f"tpr must be in [0, 1], got {tpr}")
    if not (0.0 <= fpr <= 1.0):
        raise ValueError(f"fpr must be in [0, 1], got {fpr}")
    if not (0.0 <= base_rate <= 1.0):
        raise ValueError(f"base_rate must be in [0, 1], got {base_rate}")
    num = tpr * base_rate
    den = num + fpr * (1.0 - base_rate)
    if den <= 0.0:
        return float("nan")
    return float(num / den)


def min_fpr_for_ppv(
    target_ppv: float,
    tpr: float,
    base_rate: float,
) -> float:
    """Largest FPR consistent with a desired PPV at given TPR and base rate.

    Solving the Bayes formula for ``FPR``:

    .. math::
        FPR \\le \\frac{TPR \\cdot \\pi \\cdot (1 - \\mathrm{PPV})}
                       {(1 - \\pi) \\cdot \\mathrm{PPV}}

    Returns
    -------
    float in [0, 1] or ``inf`` when the target is trivially satisfiable
    (e.g. when ``target_ppv = 0``).  Returns ``0`` when no FPR can meet
    the target (e.g. ``target_ppv = 1``); in that case the only
    admissible operating point is *zero* false alarms.
    """
    if not (0.0 < target_ppv <= 1.0):
        raise ValueError(f"target_ppv must be in (0, 1], got {target_ppv}")
    if not (0.0 < tpr <= 1.0):
        raise ValueError(f"tpr must be in (0, 1], got {tpr}")
    if not (0.0 < base_rate < 1.0):
        raise ValueError(f"base_rate must be in (0, 1), got {base_rate}")
    if target_ppv >= 1.0:
        return 0.0
    return float(
        tpr * base_rate * (1.0 - target_ppv)
        / ((1.0 - base_rate) * target_ppv)
    )


def min_tpr_for_ppv(
    target_ppv: float,
    fpr: float,
    base_rate: float,
) -> float:
    """Minimum TPR required to reach ``target_ppv`` given FPR and base rate.

    .. math::
        TPR \\ge \\frac{(1 - \\pi) \\cdot \\mathrm{PPV} \\cdot FPR}
                       {\\pi \\cdot (1 - \\mathrm{PPV})}

    Returns ``inf`` when the constraint is infeasible (target PPV too
    high for the chosen FPR / base rate).
    """
    if not (0.0 < target_ppv < 1.0):
        raise ValueError(f"target_ppv must be in (0, 1), got {target_ppv}")
    if not (0.0 < fpr <= 1.0):
        raise ValueError(f"fpr must be in (0, 1], got {fpr}")
    if not (0.0 < base_rate < 1.0):
        raise ValueError(f"base_rate must be in (0, 1), got {base_rate}")
    needed = (
        (1.0 - base_rate) * target_ppv * fpr
        / (base_rate * (1.0 - target_ppv))
    )
    return float(needed) if needed <= 1.0 else float("inf")


# ──────────────────────────────────────────────────────────────────────
# Per-attack table from labelled data
# ──────────────────────────────────────────────────────────────────────
def _attack_window_count(
    attack: Mapping,
    timestamps: pd.Series,
) -> int:
    """Count how many sampled windows fall inside an attack interval.

    Accepts both ``{start, duration_h}`` and ``{start, end}`` schemas
    used elsewhere in the project (e.g. by ``compare_if_fair.py``).
    """
    if "start" not in attack:
        raise KeyError("attack entry missing 'start'.")
    t0 = pd.Timestamp(attack["start"])
    if attack.get("duration_h") is not None:
        t1 = t0 + pd.Timedelta(hours=float(attack["duration_h"]))
    elif attack.get("end") is not None:
        t1 = pd.Timestamp(attack["end"])
    else:
        raise KeyError("attack entry needs 'duration_h' or 'end'.")
    if t1 <= t0:
        raise ValueError(
            f"attack interval must satisfy end > start, got [{t0}, {t1})"
        )
    return int(((timestamps >= t0) & (timestamps < t1)).sum())


def _bca_ci_proportion(
    successes: int,
    n: int,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 42,
    n_eff: float | None = None,
) -> tuple[float, float]:
    """Wilson interval for a binomial proportion (closed-form, no bootstrap).

    Wilson 1927 is the standard textbook recommendation for proportions
    when ``n`` is small (< ~50) — avoiding the disastrous behaviour of
    the normal-approximation interval near 0 or 1.  For large ``n`` it
    coincides with the asymptotic normal CI.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    n_ci = float(n if n_eff is None else max(1.0, min(float(n_eff), float(n))))
    z = 1.959963984540054  # two-sided 95 %
    if not math.isclose(alpha, 0.05):
        from scipy.stats import norm  # local import — optional dep
        z = float(norm.isf(alpha / 2))
    denom = 1.0 + z * z / n_ci
    centre = (p + z * z / (2 * n_ci)) / denom
    half = z * math.sqrt(p * (1 - p) / n_ci + z * z / (4 * n_ci * n_ci)) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return (lo, hi)


def _effective_n_indicator(values: np.ndarray, max_lag: int = 10) -> float:
    """Newey-West-style effective n for an ordered binary indicator."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < 2 or newey_west_eff_n is None:
        return float(arr.size)
    try:
        return float(newey_west_eff_n(arr, max_lag=max_lag))
    except Exception:
        return float(arr.size)


def per_attack_ppv_table(
    attacks: Sequence[Mapping],
    timestamps: pd.Series,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    operating_fpr: Optional[float] = None,
    target_ppv: float = 0.5,
    alpha: float = 0.05,
    effective_n_lag: int = 10,
) -> pd.DataFrame:
    """Per-attack base-rate / PPV / detectability table.

    Parameters
    ----------
    attacks : sequence of dicts
        ``CONFIG["ATTACK_CATALOG"]``-shaped entries.  Each must have
        ``name`` (or ``id``) and ``start`` plus one of ``duration_h``
        or ``end``.
    timestamps : pd.Series of datetimes, length N
        One per evaluation unit (window).  Must be sorted.
    y_pred : array of {0, 1}, length N
        Detector decisions at the operating threshold.
    y_true : array of {0, 1}, length N
        Ground-truth labels (1 inside any attack interval, else 0).
    operating_fpr : float, optional
        Global false-alarm rate to use in the Bayes calculation.  If
        omitted, computed from ``y_pred`` and ``y_true`` directly.
    target_ppv : float in (0, 1), default 0.5
        Used to compute the column ``fpr_required_for_target_ppv``.

    Returns
    -------
    DataFrame with columns:
        attack, n_windows, base_rate, base_rate_ci_low, base_rate_ci_high,
        tpr, tpr_ci_low, tpr_ci_high, fpr_global, ppv,
        fpr_required_for_target_ppv
    """
    timestamps = pd.Series(timestamps).reset_index(drop=True)
    y_pred = np.asarray(y_pred).astype(int).ravel()
    y_true = np.asarray(y_true).astype(int).ravel()
    if not (len(timestamps) == y_pred.size == y_true.size):
        raise ValueError(
            f"timestamps / y_pred / y_true length mismatch: "
            f"{len(timestamps)}, {y_pred.size}, {y_true.size}"
        )
    n_total = int(y_true.size)

    if operating_fpr is None:
        # Global FPR computed on negative-only windows.
        n_neg = int((y_true == 0).sum())
        operating_fpr = float(
            ((y_pred == 1) & (y_true == 0)).sum() / max(1, n_neg)
        )

    rows = []
    for atk in attacks:
        name = str(atk.get("name") or atk.get("id") or "<unnamed>")
        n_atk = _attack_window_count(atk, timestamps)
        if n_atk == 0:
            rows.append({
                "attack": name,
                "n_windows": 0,
                "base_rate": 0.0,
                "base_rate_ci_low": 0.0,
                "base_rate_ci_high": 0.0,
                "base_rate_n_eff": float("nan"),
                "tpr": float("nan"),
                "tpr_ci_low": float("nan"),
                "tpr_ci_high": float("nan"),
                "tpr_n_eff": float("nan"),
                "fpr_global": operating_fpr,
                "ppv": float("nan"),
                "fpr_required_for_target_ppv": float("nan"),
            })
            continue

        # Mask of windows inside this attack only.
        t0 = pd.Timestamp(atk["start"])
        if atk.get("duration_h") is not None:
            t1 = t0 + pd.Timedelta(hours=float(atk["duration_h"]))
        else:
            t1 = pd.Timestamp(atk["end"])
        atk_mask = ((timestamps >= t0) & (timestamps < t1)).to_numpy()

        # TPR specific to this attack.
        n_tp_in_atk = int((y_pred[atk_mask] == 1).sum())
        tpr_atk = n_tp_in_atk / n_atk
        tpr_n_eff = _effective_n_indicator(
            (y_pred[atk_mask] == 1).astype(int), max_lag=effective_n_lag
        )
        tpr_lo, tpr_hi = _bca_ci_proportion(
            n_tp_in_atk, n_atk, alpha, n_eff=tpr_n_eff
        )

        # Empirical base rate for this attack (probability that a
        # randomly drawn window over the full evaluation period falls
        # inside this attack).
        pi = n_atk / n_total
        pi_n_eff = _effective_n_indicator(atk_mask.astype(int), max_lag=effective_n_lag)
        pi_lo, pi_hi = _bca_ci_proportion(n_atk, n_total, alpha, n_eff=pi_n_eff)

        ppv = bayesian_ppv(tpr_atk, operating_fpr, pi)

        try:
            fpr_req = min_fpr_for_ppv(target_ppv, tpr_atk, pi)
        except ValueError:
            fpr_req = float("nan")

        rows.append({
            "attack": name,
            "n_windows": n_atk,
            "base_rate": pi,
            "base_rate_ci_low": pi_lo,
            "base_rate_ci_high": pi_hi,
            "base_rate_n_eff": pi_n_eff,
            "tpr": tpr_atk,
            "tpr_ci_low": tpr_lo,
            "tpr_ci_high": tpr_hi,
            "tpr_n_eff": tpr_n_eff,
            "fpr_global": operating_fpr,
            "ppv": ppv,
            "fpr_required_for_target_ppv": fpr_req,
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Markdown rendering
# ──────────────────────────────────────────────────────────────────────
def format_axelsson_md(
    table: pd.DataFrame,
    target_ppv: float,
    operating_fpr: float,
) -> str:
    """Markdown report ready to be appended to the paper appendix."""
    lines = [
        "# Per-attack base-rate / PPV table (Axelsson 2000)",
        "",
        "Reference: Axelsson, S. (2000). *The base-rate fallacy and the "
        "difficulty of intrusion detection.* TISSEC 3(3): 186-205.",
        "",
        f"- Operating FPR (global)                  : {operating_fpr:.6f}",
        f"- Target PPV (column `fpr_required_…`)    : {target_ppv:.3f}",
        f"- Number of attack types covered          : {len(table)}",
        "",
        "## Interpretation",
        "",
        "* `base_rate`     — π = #windows inside the attack / #windows total.",
        "* `tpr`           — within-attack recall at the operating threshold.",
        "* `fpr_global`    — false alarm rate measured outside *all* attacks.",
        "* `ppv`           — PPV at the operating point: "
        "`tpr·π / (tpr·π + fpr·(1-π))`.",
        f"* `fpr_required_for_target_ppv` — largest FPR that still yields "
        f"PPV ≥ {target_ppv:.2f} for this attack.  Smaller than "
        f"`fpr_global` ⇒ the operating point is too lax.",
        "",
        "## Table",
        "",
    ]
    if table.empty:
        lines.append("_(empty)_")
        return "\n".join(lines)

    cols = list(table.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in table.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if not math.isfinite(v):
                    cells.append("NaN")
                elif abs(v) < 1e-3:
                    cells.append(f"{v:.2e}")
                else:
                    cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="axelsson_ppv",
        description="Per-attack base-rate / PPV table (TASK-18).",
    )
    p.add_argument("--detection-csv", default=None,
                   help="CSV containing columns: timestamp, y_true, y_pred. "
                        "If omitted, run --self-test instead.")
    p.add_argument("--target-ppv", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Two-sided significance level for proportion CIs.")
    p.add_argument("--output-csv", default=None)
    p.add_argument("--output-md", default=None)
    p.add_argument("--self-test", action="store_true")
    return p


def _self_test() -> int:
    print("[TEST] axelsson_ppv.py — self-test")

    # 1. Axelsson 2000, Section 6 textbook example: π=1e-5, TPR=0.7, FPR=1e-5
    # PPV ≈ 0.412 (his stated number).
    ppv = bayesian_ppv(0.7, 1e-5, 1e-5)
    assert math.isclose(ppv, 0.4118, abs_tol=2e-4), ppv
    print(f"   [OK] Axelsson 2000 §6 textbook case: PPV={ppv:.4f}  (expect 0.4118)")

    # 2. Sanity: bayesian_ppv invariants.
    assert bayesian_ppv(1.0, 0.0, 0.5) == 1.0
    assert bayesian_ppv(0.0, 1.0, 0.5) == 0.0
    assert math.isnan(bayesian_ppv(0.0, 0.0, 0.5))  # detector silent
    print("   [OK] PPV boundary cases")

    # 3. Inverse: min_fpr_for_ppv.  Round-trip: feed PPV(target_ppv).
    tpr, pi, target = 0.9, 1e-3, 0.5
    fpr_req = min_fpr_for_ppv(target, tpr, pi)
    assert math.isclose(bayesian_ppv(tpr, fpr_req, pi), target, abs_tol=1e-9), (
        fpr_req, bayesian_ppv(tpr, fpr_req, pi)
    )
    print(f"   [OK] min_fpr_for_ppv round-trip: required FPR={fpr_req:.6f}")

    # 4. min_tpr_for_ppv: same round-trip.
    tpr_req = min_tpr_for_ppv(target, fpr=1e-3, base_rate=1e-3)
    assert math.isclose(bayesian_ppv(tpr_req, 1e-3, 1e-3), target, abs_tol=1e-9), tpr_req
    print(f"   [OK] min_tpr_for_ppv round-trip: required TPR={tpr_req:.6f}")

    # 5. min_tpr returns inf when infeasible.
    inf = min_tpr_for_ppv(target_ppv=0.99, fpr=0.5, base_rate=0.001)
    assert math.isinf(inf), inf
    print("   [OK] min_tpr_for_ppv signals infeasible target")

    # 6. _bca_ci_proportion: covers the textbook (k=10, n=100) Wilson CI
    # whose published value is approximately (0.0552, 0.1740).
    lo, hi = _bca_ci_proportion(10, 100)
    assert abs(lo - 0.0552) < 1e-3, lo
    assert abs(hi - 0.1740) < 1e-3, hi
    print(f"   [OK] Wilson CI on (10, 100): [{lo:.4f}, {hi:.4f}]")

    # 7. per_attack_ppv_table: tiny synthetic dataset with one attack.
    ts = pd.date_range("2026-01-01", periods=100, freq="h")
    y_true = np.zeros(100, dtype=int)
    y_pred = np.zeros(100, dtype=int)
    # Attack on hours 10-19; detector flags hours 11-15 (5/10 TPR).
    y_true[10:20] = 1
    y_pred[11:16] = 1
    # And a single false alarm at hour 30.
    y_pred[30] = 1
    attacks = [{"name": "TEST_ATTACK", "start": ts[10], "end": ts[20]}]

    table = per_attack_ppv_table(attacks, ts, y_pred, y_true)
    row = table.iloc[0]
    assert row["n_windows"] == 10, row
    assert math.isclose(row["base_rate"], 0.10), row
    assert math.isclose(row["tpr"], 0.5), row
    # FPR = 1 / 90 = 0.01111…
    assert math.isclose(row["fpr_global"], 1 / 90, abs_tol=1e-9), row
    expected_ppv = bayesian_ppv(0.5, 1 / 90, 0.1)
    assert math.isclose(row["ppv"], expected_ppv, abs_tol=1e-9), row
    print(f"   [OK] per_attack_ppv_table: TPR={row['tpr']:.2f}, "
          f"PPV={row['ppv']:.4f}")

    # 8. Empty attack list ⇒ empty table.
    empty = per_attack_ppv_table([], ts, y_pred, y_true)
    assert empty.empty
    print("   [OK] empty catalogue ⇒ empty table")

    # 9. Markdown formatter.
    md = format_axelsson_md(table, target_ppv=0.5, operating_fpr=row["fpr_global"])
    assert "Axelsson" in md
    assert "TEST_ATTACK" in md
    assert "fpr_required_for_target_ppv" in md
    print("   [OK] format_axelsson_md")

    # 10. Length-mismatch raises.
    try:
        per_attack_ppv_table(attacks, ts, y_pred[:-1], y_true)
    except ValueError:
        pass
    else:
        raise AssertionError("length mismatch should raise.")
    print("   [OK] length-mismatch raises")

    print("[OK] axelsson_ppv.py — ALL PASS")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    if args.self_test or args.detection_csv is None:
        return _self_test()

    df = pd.read_csv(args.detection_csv)
    required = {"timestamp", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERR] missing columns in {args.detection_csv}: {missing}",
              file=sys.stderr)
        return 2
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    try:
        from sl_ads.config import CONFIG  # type: ignore
        attacks = CONFIG.get("ATTACK_CATALOG") or []
    except ImportError:
        attacks = []
    if not attacks:
        print("[ERR] CONFIG['ATTACK_CATALOG'] is empty; cannot build table.",
              file=sys.stderr)
        return 2

    table = per_attack_ppv_table(
        attacks=attacks,
        timestamps=df["timestamp"],
        y_pred=df["y_pred"].to_numpy(),
        y_true=df["y_true"].to_numpy(),
        target_ppv=args.target_ppv,
        alpha=args.alpha,
    )
    if args.output_csv:
        table.to_csv(args.output_csv, index=False)
        print(f"[OK] CSV written to {args.output_csv}")
    if args.output_md:
        md = format_axelsson_md(table, args.target_ppv,
                                float(table["fpr_global"].iloc[0])
                                if not table.empty else float("nan"))
        with open(args.output_md, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"[OK] Markdown written to {args.output_md}")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
