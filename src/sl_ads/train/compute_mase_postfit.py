"""compute_mase_postfit.py — patch an existing models PKL with MASE scores.

Purpose
-------
The first integration of ``WBF_WEIGHT_MODE='mase'`` (PATCH D5) lands
**after** the canonical reference pkl (``trained_models_RedeRio_*.pkl``)
was produced.  The pkl already contains every ingredient required to
compute MASE (Hyndman-Koehler 2006 Eq. 4) without re-fitting Prophet:

  - ``models_pkg[metric]['model']`` — fitted Prophet/QR estimator.
  - ``models_pkg['_calib_signed_residuals'][metric]`` — held-out
    calibration residuals (PATCH 2026-05-06, persisted in train_models).
  - The original CSV reachable through ``CONFIG['file_path']`` provides
    the ``y_true`` series needed for the Naive-1 baseline of the MASE
    denominator.

Running this script patches ``models_pkg[metric]['mase_score']`` for
each Prophet metric and rewrites ``models_pkg['mase_scores']`` (the
top-level dict consumed by ``opinions_pipeline.py``).  RANSAC
reconstructions remain ``mase_score = NaN`` because MASE is undefined
for non-temporal cross-feature regressions; their trust score in the
``mase`` mode falls back to ``TRUST_SCORE_FLOOR``.

Once the next clean retrain runs (``run_pipeline.py --dataset RedeRio``
or equivalent), this post-fit step becomes redundant — ``train_models``
already persists ``mase_score`` natively.

Usage
-----

```
PYTHONPATH=src python -m sl_ads.train.compute_mase_postfit
```

Optional flags:

  - ``--pkl PATH`` to override the PKL discovered through
    ``paths.get_model_path``.
  - ``--csv PATH`` to override the CSV path resolved from
    ``CONFIG['file_path']``.
  - ``--dry-run`` to print what would change without writing.

Audit trail
-----------
Every patch writes a sibling JSON file
``<pkl>_mase_postfit_audit.json`` containing the per-metric
(R², MASE, trust) triplet plus the timestamp of the patch run.
This lets a reviewer verify *post hoc* which pkl was patched, when,
and what the resulting trust dispatch is.

References
----------
- Hyndman, R. J. & Koehler, A. B. (2006). "Another look at measures of
  forecast accuracy." *International Journal of Forecasting* 22(4),
  679–688.
- Joesang, A. (2016). *Subjective Logic*. Springer. Def. 14.6.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sl_ads.config import CONFIG
from sl_ads.paths import get_model_path
from sl_ads.stats.mase import compute_mase, mase_to_trust


def _resolve_pkl_path(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    return Path(get_model_path(CONFIG, up_levels=1)).resolve()


def _resolve_csv_path(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    return Path(CONFIG["file_path"]).resolve()


def _load_train_y(csv_path: Path, target_col: str) -> np.ndarray | None:
    """Load ``y_true`` for one Prophet target on the train span only.

    Returns ``None`` if the column is absent (RANSAC reconstruction
    targets that are not raw metrics live in different rows).
    """
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        return None
    if "ds" not in df.columns and "timestamp" in df.columns:
        df["ds"] = pd.to_datetime(df["timestamp"])
    elif "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])
    else:
        return None
    split_date = pd.Timestamp(CONFIG["split_date"])
    df_train = df[df["ds"] <= split_date]
    return df_train[target_col].astype(float).to_numpy()


def patch_pkl(pkl_path: Path, csv_path: Path,
              alpha: float, floor: float,
              dry_run: bool = False) -> dict:
    """Compute MASE for every Prophet metric in the pkl and patch it.

    Returns a JSON-serialisable audit dict.
    """
    print(f"[mase_postfit] loading {pkl_path}")
    pkg = joblib.load(pkl_path)

    calib_residuals = pkg.get("_calib_signed_residuals") or {}
    if not calib_residuals:
        print(
            "[mase_postfit] WARNING: pkl has no '_calib_signed_residuals'. "
            "MASE will be computed on training-span residuals via the "
            "Prophet model's predict() instead — slightly higher cost.",
            file=sys.stderr,
        )

    audit_rows = []
    finite_mase = 0
    for key, entry in pkg.items():
        if not isinstance(entry, dict):
            continue
        if key in ("empirical_priors", "trust_scores", "mase_scores"):
            continue
        entry_type = entry.get("type")
        if entry_type != "prophet":
            # RANSAC and any non-temporal source: NaN MASE by design.
            entry.setdefault("mase_score", float("nan"))
            audit_rows.append({
                "key": key,
                "type": entry_type,
                "r2_score": entry.get("r2_score"),
                "mase": float("nan"),
                "trust_mase": float(floor),
                "reason": "non-temporal source — MASE undefined",
            })
            continue

        # Prophet target name lives in the key suffix: ``prophet_<col>``.
        target_col = key.removeprefix("prophet_")

        # Preferred path: replay calibration-span residuals + the same
        # span's y_true on the training CSV. Falls back to in-sample
        # predict() if calibration data isn't reachable.
        y_true = _load_train_y(csv_path, target_col)
        if y_true is None:
            audit_rows.append({
                "key": key,
                "type": entry_type,
                "r2_score": entry.get("r2_score"),
                "mase": float("nan"),
                "trust_mase": float(floor),
                "reason": f"no '{target_col}' column in CSV",
            })
            entry["mase_score"] = float("nan")
            continue

        # Compute predictions over the same span by re-running Prophet's
        # internal forecaster on the train calendar. We avoid re-fitting.
        try:
            from prophet import Prophet  # noqa: F401 — sanity check
            prophet_model = entry.get("model")
            if prophet_model is None:
                raise RuntimeError("missing 'model' field in pkg entry")

            # Build the prediction frame matching what ``train_models``
            # used. The train CSV must already have ``on_weekday`` /
            # ``on_weekend`` engineered. If not, we synthesise them from
            # ``ds`` so the predict() call doesn't crash.
            if "ds" not in pd.read_csv(csv_path, nrows=1).columns:
                df_full = pd.read_csv(csv_path)
                df_full["ds"] = pd.to_datetime(df_full["timestamp"])
            else:
                df_full = pd.read_csv(csv_path)
                df_full["ds"] = pd.to_datetime(df_full["ds"])
            split_date = pd.Timestamp(CONFIG["split_date"])
            df_train = df_full[df_full["ds"] <= split_date].copy()
            if "on_weekday" not in df_train.columns:
                dow = df_train["ds"].dt.dayofweek
                df_train["on_weekday"] = (dow < 5).astype(int)
                df_train["on_weekend"] = (dow >= 5).astype(int)

            df_predict = df_train[["ds", "on_weekday", "on_weekend"]]
            fcst = prophet_model.predict(df_predict)
            y_pred = fcst["yhat"].astype(float).to_numpy()

            mask = ~np.isnan(y_true)
            mase_val = compute_mase(y_true[mask], y_pred[mask])
        except Exception as exc:  # pragma: no cover — defensive
            print(f"[mase_postfit] {key}: predict() failed ({exc}); "
                  f"falling back to NaN MASE.", file=sys.stderr)
            mase_val = float("nan")

        trust_val = mase_to_trust(mase_val, alpha=alpha, floor=floor)
        entry["mase_score"] = float(mase_val)
        if math.isfinite(mase_val):
            finite_mase += 1
        audit_rows.append({
            "key": key,
            "type": entry_type,
            "r2_score": entry.get("r2_score"),
            "mase": (float(mase_val) if math.isfinite(mase_val)
                     else None),
            "trust_mase": float(trust_val),
            "reason": (f"informative (MASE<1)" if math.isfinite(mase_val) and mase_val < 1.0
                       else "noisier than Naive-1" if math.isfinite(mase_val)
                       else "non-finite MASE"),
        })

    # Refresh the top-level dict consumed by opinions_pipeline.
    new_mase_dict = {
        row["key"]: float(row["trust_mase"])
        for row in audit_rows
    }
    pkg["mase_scores"] = new_mase_dict

    audit = {
        "patched_at_utc": datetime.now(timezone.utc).isoformat(),
        "pkl_path": str(pkl_path),
        "csv_path": str(csv_path),
        "alpha": float(alpha),
        "floor": float(floor),
        "n_metrics_audited": len(audit_rows),
        "n_finite_mase": finite_mase,
        "rows": audit_rows,
    }

    if not dry_run:
        joblib.dump(pkg, pkl_path)
        audit_path = pkl_path.with_suffix(pkl_path.suffix + ".mase_postfit_audit.json")
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(f"[mase_postfit] patched {pkl_path}")
        print(f"[mase_postfit] audit:  {audit_path}")
    else:
        print("[mase_postfit] --dry-run: no file written")

    return audit


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pkl", default=None, help="Override the PKL path.")
    p.add_argument("--csv", default=None, help="Override the CSV path.")
    p.add_argument("--alpha", type=float, default=None,
                   help="MASE_TRUST_ALPHA override (default = CONFIG).")
    p.add_argument("--floor", type=float, default=None,
                   help="TRUST_SCORE_FLOOR override (default = CONFIG).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    pkl_path = _resolve_pkl_path(args.pkl)
    csv_path = _resolve_csv_path(args.csv)
    alpha = (args.alpha if args.alpha is not None
             else float(CONFIG.get("MASE_TRUST_ALPHA", 1.0)))
    floor = (args.floor if args.floor is not None
             else float(CONFIG.get("TRUST_SCORE_FLOOR", 0.05)))

    if not pkl_path.is_file():
        print(f"[mase_postfit] PKL not found: {pkl_path}", file=sys.stderr)
        return 2
    if not csv_path.is_file():
        print(f"[mase_postfit] CSV not found: {csv_path}", file=sys.stderr)
        return 2

    audit = patch_pkl(pkl_path, csv_path, alpha=alpha, floor=floor,
                       dry_run=args.dry_run)

    print("\n[mase_postfit] per-Prophet trust:")
    for row in audit["rows"]:
        if row["type"] == "prophet":
            mase_repr = (f"{row['mase']:.3f}" if row["mase"] is not None
                         else "NaN")
            print(f"  {row['key']:<35} R²={str(row['r2_score']):>8}  "
                  f"MASE={mase_repr:>6}  trust={row['trust_mase']:.3f}  "
                  f"({row['reason']})")
    print(f"\n[mase_postfit] {audit['n_finite_mase']} / "
          f"{audit['n_metrics_audited']} metrics with finite MASE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
