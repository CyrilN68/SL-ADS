"""
INV-2 — Stationnarité (ADF) + autocorrélation des résidus Prophet (rho1)
=======================================================================

Objectif:
- Tester ADF sur bytes, packets, flows, entropy_src_ip (train brut 30s)
- Calculer rho1 sur résidus Prophet 30s + n_eff/n (Bayley-Hammersley)

Usage:
  cd "actual_ version"
  python3 investigate_inv2_stationarity.py

Sortie:
  ../results/resultats_<VERSION_NAME>/inv2_stationarity_residuals.csv
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import CONFIG
from statsmodels.tsa.stattools import adfuller, acf

METRICS = ["bytes", "packets", "flows", "entropy_src_ip"]
MAXLAG = 168  # demandé: 168


def _find_data_csv() -> str:
    candidates = [
        CONFIG.get("file_path"),
        "../data/dataset_1310_2912_v30s.csv",
        "../data/dataset_1310_2912_v30s.csv",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("Dataset introuvable. Vérifier CONFIG['file_path'].")


def _fit_prophet_and_residuals(df_train: pd.DataFrame, metric: str):
    try:
        from prophet import Prophet
    except Exception:
        return None

    if metric not in df_train.columns:
        return None

    seasonality_mode = "additive" if "entropy" in metric else "multiplicative"

    d = df_train[["ds", metric]].rename(columns={metric: "y"}).dropna().copy()
    if d.empty:
        return None

    d["on_weekend"] = (d["ds"].dt.dayofweek >= 5).astype(int)
    d["on_weekday"] = 1 - d["on_weekend"]

    m = Prophet(
        growth="flat",
        daily_seasonality=False,
        weekly_seasonality=True,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=0.05,
    )
    m.add_seasonality("daily_weekday", period=1, fourier_order=12, condition_name="on_weekday")
    m.add_seasonality("daily_weekend", period=1, fourier_order=12, condition_name="on_weekend")
    m.add_seasonality("hourly", period=1 / 24, fourier_order=5)

    m.fit(d)
    pred = m.predict(d[["ds", "on_weekend", "on_weekday"]])
    residuals = (d["y"].values - pred["yhat"].values).astype(float)
    return residuals


def main():
    data_csv = _find_data_csv()
    df = pd.read_csv(data_csv)

    ts_col = "timestamp" if "timestamp" in df.columns else "ds"
    if ts_col not in df.columns:
        raise ValueError("Aucune colonne timestamp/ds dans le dataset.")

    df["ds"] = pd.to_datetime(df[ts_col])
    df = df.sort_values("ds").reset_index(drop=True)

    split_date = pd.to_datetime(CONFIG.get("split_date", "2025-11-09 23:59:59"))
    df_train = df[df["ds"] <= split_date].copy()
    if df_train.empty:
        raise ValueError("Split train vide. Vérifier CONFIG['split_date'].")

    rows = []
    print("=" * 90)
    print("INV-2 | ADF + rho1 résidus Prophet (30s)")
    print(f"Data: {data_csv}")
    print(f"Train range: {df_train['ds'].iloc[0]} -> {df_train['ds'].iloc[-1]} | n={len(df_train)}")
    print("=" * 90)

    for metric in METRICS:
        if metric not in df_train.columns:
            print(f"[SKIP] {metric}: colonne absente")
            continue

        s = pd.to_numeric(df_train[metric], errors="coerce").dropna()
        if len(s) < 300:
            print(f"[SKIP] {metric}: série trop courte pour ADF")
            continue

        adf_stat, adf_p, usedlag, nobs, *_ = adfuller(s.values, maxlag=MAXLAG, autolag="AIC")

        residuals = _fit_prophet_and_residuals(df_train[["ds", metric]].copy(), metric)
        if residuals is None or len(residuals) < 5:
            rho1 = np.nan
            neff_ratio = np.nan
        else:
            rho1 = float(acf(residuals, nlags=1, fft=True)[1])
            rho1 = min(0.999999, max(-0.999999, rho1))
            neff = len(residuals) * (1.0 - rho1) / (1.0 + rho1)
            neff_ratio = neff / len(residuals)

        stationarity_note = "ADF rejects H0 => stationnaire (flat justifié)" if adf_p < 0.05 else "ADF non concluant => documenter limitation"
        autocorr_note = "rho1>0.5 => documenter n_eff" if (not np.isnan(rho1) and rho1 > 0.5) else "rho1<=0.5"

        print(
            f"{metric:15s} | ADF stat={adf_stat:>9.4f} p={adf_p:>10.6f} lag={usedlag:>3d} "
            f"| rho1={rho1:>7.3f} | n_eff/n={neff_ratio:>6.3f}"
        )

        rows.append({
            "metric": metric,
            "adf_stat": adf_stat,
            "adf_p_value": adf_p,
            "adf_usedlag": usedlag,
            "adf_nobs": nobs,
            "rho1_residuals": rho1,
            "n_eff_ratio": neff_ratio,
            "stationarity_note": stationarity_note,
            "autocorr_note": autocorr_note,
        })

    version_name = CONFIG.get("VERSION_NAME", "trained_models_v9_v6_v4s")
    out_dir = CONFIG.get("EVAL", {}).get("RESULTS_DIR", f"../../results/resultats_{version_name}")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "inv2_stationarity_residuals.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
