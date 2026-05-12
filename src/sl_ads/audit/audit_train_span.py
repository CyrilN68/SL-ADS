"""audit_train_span.py — Unsupervised audit of the training span (A1.1)
===================================================================

The pipeline relies on assumption A1.1 — *the training span is
attack-free* — to calibrate Prophet, EDP, EVT thresholds and the
decision threshold delta. Today this is enforced only through the manual
``CONFIG['TRAIN_EXCLUSIONS']`` list. Any unlabelled real attack hidden
inside ``df_train`` would silently bias every threshold and invalidate
the headline FPR claim.

This script runs an *unsupervised* second-opinion audit:

  - Per metric, it applies the existing ``ConsensusLabeller``
    (STL residual z-score + Hampel filter + CUSUM change-point) on the
    train span only.
  - It aggregates per-window suspicion across metrics (>= 3 metrics
    flagged ⇒ "high"; >= 2 ⇒ "medium"; 1 ⇒ "low").
  - It strips windows that already lie inside ``CONFIG['TRAIN_EXCLUSIONS']``
    so the operator only sees *new* suspicious windows.
  - It writes ``audit_train_span.csv`` (per-window flag table) and
    ``audit_train_span_summary.json`` (top-N suspicious windows).

This is a *defensive* tool: it never modifies the dataset and never
silently filters anything. The operator is expected to read the
top-N list and decide whether to add new exclusions.

Methodology rationale.  STL+Hampel+CUSUM are deliberately classical
non-parametric detectors operating on the raw metric series — no
Prophet, no SL, no shared assumption with the production pipeline.
A 2/3 consensus is robust against any single algorithm's biases
(STL fails on heavy seasonality, Hampel fails on long anomalies,
CUSUM fails on multi-modal regimes).  This is the same labeller the
``rederio_adapter`` uses for unsupervised pseudo-labels, which is
already accepted in the codebase as an independent second opinion.

References:
  - Cleveland et al. (1990) STL: J. Off. Stat. 6(1).
  - Hampel (1971), JASA 66(335), 1179-1186.
  - Page (1954), Biometrika 41(1-2).
  - Domingos & Pazzani (1997), ML 29(2-3), on consensus robustness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sl_ads.config import CONFIG
from sl_ads.adapters.labeller_unsupervised import ConsensusLabeller


def _load_train_span(file_path: Path) -> pd.DataFrame:
    """Load the dataset and trim to the train span.

    The CSV column for the timestamp is one of {ds, timestamp}; we
    standardise to ``ds`` internally so downstream code is uniform.
    """
    df = pd.read_csv(file_path)
    if "ds" not in df.columns and "timestamp" in df.columns:
        df["ds"] = pd.to_datetime(df["timestamp"])
    elif "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])
    else:
        raise ValueError(
            f"{file_path} has neither 'ds' nor 'timestamp' column."
        )
    split_date = pd.Timestamp(CONFIG["split_date"])
    df_train = df[df["ds"] <= split_date].copy().reset_index(drop=True)
    return df_train


def _load_already_excluded_mask(df_train: pd.DataFrame) -> pd.Series:
    """Boolean mask indicating windows already in CONFIG['TRAIN_EXCLUSIONS'].

    Each entry of TRAIN_EXCLUSIONS is a (start, end) timestamp pair.
    Windows inside any of those ranges are already known-anomalous and
    must be excluded from the audit's "new suspect" list.
    """
    excluded = pd.Series(False, index=df_train.index)
    excl = CONFIG.get("TRAIN_EXCLUSIONS", []) or []
    for entry in excl:
        # Tolerate (start, end) tuples or {'start','end'} dicts.
        if isinstance(entry, dict):
            t0 = pd.Timestamp(entry["start"])
            t1 = pd.Timestamp(entry["end"])
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            t0 = pd.Timestamp(entry[0])
            t1 = pd.Timestamp(entry[1])
        else:
            continue
        excluded |= (df_train["ds"] >= t0) & (df_train["ds"] <= t1)
    return excluded


def _audit_one_metric(series: pd.Series, period: int) -> np.ndarray:
    """Return per-point binary anomaly flag for one metric."""
    labeller = ConsensusLabeller(period=period)
    cleaned = series.ffill().fillna(0.0)
    return labeller.generate_labels(cleaned)


def audit_metrics(df_train: pd.DataFrame,
                  metrics: list[str],
                  period: int) -> pd.DataFrame:
    """Run the consensus labeller on every metric, return a flag matrix."""
    records = []
    for metric in metrics:
        if metric not in df_train.columns:
            continue
        try:
            flags = _audit_one_metric(df_train[metric], period)
        except Exception as exc:
            print(f"[audit_train_span] metric={metric} skipped: {exc}")
            continue
        records.append((metric, flags))
    if not records:
        return pd.DataFrame()
    out = pd.DataFrame({"ds": df_train["ds"]})
    for metric, flags in records:
        out[f"flag_{metric}"] = flags
    out["n_metrics_flagging"] = out[
        [c for c in out.columns if c.startswith("flag_")]
    ].sum(axis=1)
    return out


def summarise_audit(audit: pd.DataFrame, already_excluded: pd.Series,
                    top_n: int = 50) -> dict:
    """Aggregate per-window flags into a reviewer-facing report."""
    audit = audit.copy()
    audit["already_excluded"] = already_excluded.values
    new_flags = audit[~audit["already_excluded"]].copy()
    summary = {
        "n_train_windows": int(len(audit)),
        "n_already_excluded": int(audit["already_excluded"].sum()),
        "n_train_windows_audited": int((~audit["already_excluded"]).sum()),
        "n_high_severity": int((new_flags["n_metrics_flagging"] >= 3).sum()),
        "n_medium_severity": int(
            ((new_flags["n_metrics_flagging"] >= 2)
             & (new_flags["n_metrics_flagging"] < 3)).sum()
        ),
        "n_low_severity": int((new_flags["n_metrics_flagging"] == 1).sum()),
    }
    top = new_flags.sort_values("n_metrics_flagging", ascending=False).head(top_n)
    summary["top_suspect_windows"] = [
        {
            "ds": str(row["ds"]),
            "n_metrics_flagging": int(row["n_metrics_flagging"]),
            "metrics": [
                c.replace("flag_", "")
                for c in audit.columns
                if c.startswith("flag_") and bool(row.get(c, 0))
            ],
        }
        for _, row in top.iterrows()
    ]
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=None,
                   help="Path to the standardized dataset CSV.")
    p.add_argument(
        "--metrics", default="bytes,packets,flows,udp,tcp,syn,fin,icmp,"
        "entropy_src_ip,entropy_src_port,entropy_dst_port,avg_pkt_size",
        help="Comma-separated list of metric column names to audit.",
    )
    p.add_argument("--period", type=int, default=2880,
                   help="STL period (in samples). RedeRio @ 30s ⇒ 2880 = 1 day.")
    p.add_argument("--out-dir", default="outputs/scientific_hardening")
    p.add_argument("--top-n", type=int, default=50)
    args = p.parse_args()

    csv_path = Path(args.csv) if args.csv else Path(CONFIG["file_path"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[audit_train_span] reading {csv_path}")
    df_train = _load_train_span(csv_path)
    print(f"[audit_train_span] {len(df_train)} windows in train span "
          f"(<= split_date={CONFIG['split_date']})")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    audit = audit_metrics(df_train, metrics, period=args.period)
    if audit.empty:
        print("[audit_train_span] No metric audited; check column names.")
        return 1

    excluded = _load_already_excluded_mask(df_train)
    audit["already_excluded"] = excluded.values
    out_csv = out_dir / "audit_train_span.csv"
    audit.to_csv(out_csv, index=False)
    summary = summarise_audit(audit, excluded, top_n=args.top_n)
    out_json = out_dir / "audit_train_span_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_json}")
    print(f"[audit_train_span] high-severity new suspects: "
          f"{summary['n_high_severity']}")
    print(f"[audit_train_span] medium-severity new suspects: "
          f"{summary['n_medium_severity']}")
    print(f"[audit_train_span] top {min(args.top_n, 5)} suspect windows:")
    for row in summary["top_suspect_windows"][:5]:
        print(f"   {row['ds']}  flags={row['n_metrics_flagging']} "
              f"metrics={row['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
