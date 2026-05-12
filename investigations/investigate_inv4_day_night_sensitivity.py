"""
INV-4 — Sensibilité jour/nuit sur preuve d'anomalie vs T_atk
=============================================================

Objectif:
- Quantifier l'écart jour/nuit demandé en §8.4.9 avec la bonne métrique:
  poids relatif du seuil T_atk par rapport au trafic normal.
- Ajouter des indicateurs de bruit de fond: centile 99 de N et médiane yhat.

Usage:
  cd "actual_ version"
  python3 investigate_inv4_day_night_sensitivity.py

Sortie:
  ../results/resultats_<VERSION_NAME>/inv4_day_night_sensitivity.csv
"""

import os
import pickle
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from config import CONFIG
except Exception:
    CONFIG = {}


VERSION_NAME = CONFIG.get("VERSION_NAME", "trained_models_v9_v6_v4s")
RESULTS_DIR = CONFIG.get("EVAL", {}).get("RESULTS_DIR", f"../results/resultats_{VERSION_NAME}")
VERSION_NAME_MODIF = CONFIG.get("VERSION_NAME_MODIF", f"{VERSION_NAME}_attacks")

EVIDENCE_CANDIDATES = [
    os.path.join(RESULTS_DIR, f"evidence_{VERSION_NAME}.csv"),
    os.path.join(RESULTS_DIR, f"evidence_{VERSION_NAME_MODIF}.csv"),
    os.path.join(RESULTS_DIR, "evidence_results.csv"),
]
PKL_CANDIDATES = [
    f"../trained_models_{VERSION_NAME}.pkl",
    os.path.join(RESULTS_DIR, "trained_models.pkl"),
]
OUT_CSV = os.path.join(RESULTS_DIR, "inv4_day_night_sensitivity.csv")
RAW_CANDIDATES = [
    CONFIG.get("file_path"),
    "../data/dataset_1310_2912_v30s.csv",
]

METRICS = ["prophet_bytes", "prophet_packets"]


def _base_metric(metric_key: str) -> str:
    return metric_key.replace("prophet_", "", 1)


def _find_existing(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _load_models(path: str):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def _extract_t_atk(models: dict, metric_key: str):
    data = models.get(metric_key)
    if isinstance(data, dict):
        v = data.get("t_atk")
        if v is not None:
            return float(v)
        th = data.get("thresholds", {})
        if isinstance(th, dict):
            v = th.get("attack", th.get("T_atk", th.get("t_atk")))
            if v is not None:
                return float(v)
    return float("nan")


def _pick_existing_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _raw_baselines_by_metric():
    raw_path = _find_existing(RAW_CANDIDATES)
    if raw_path is None:
        return {}, None

    rdf = pd.read_csv(raw_path)
    ts_col = "timestamp" if "timestamp" in rdf.columns else ("ds" if "ds" in rdf.columns else None)
    if ts_col is None:
        return {}, raw_path

    rdf["timestamp"] = pd.to_datetime(rdf[ts_col], errors="coerce")
    rdf = rdf.dropna(subset=["timestamp"]).copy()
    rdf["hour"] = rdf["timestamp"].dt.hour

    day_mask = (rdf["hour"] >= 8) & (rdf["hour"] <= 20)
    night_mask = (rdf["hour"] < 6) | (rdf["hour"] >= 22)

    baselines = {}
    for m in METRICS:
        base = _base_metric(m)
        if base not in rdf.columns:
            continue
        s = pd.to_numeric(rdf[base], errors="coerce")
        d = s[day_mask].dropna()
        n = s[night_mask].dropna()
        baselines[m] = {
            "raw_day_median": float(d.median()) if len(d) else float("nan"),
            "raw_night_median": float(n.median()) if len(n) else float("nan"),
            "raw_n_day": int(len(d)),
            "raw_n_night": int(len(n)),
        }
    return baselines, raw_path


def main():
    evidence_csv = _find_existing(EVIDENCE_CANDIDATES)
    if evidence_csv is None:
        raise FileNotFoundError(
            f"Aucun CSV evidence trouvé. Candidats: {EVIDENCE_CANDIDATES}"
        )

    pkl_path = _find_existing(PKL_CANDIDATES)
    if pkl_path is None:
        raise FileNotFoundError(f"Aucun PKL trouvé. Candidats: {PKL_CANDIDATES}")

    models = _load_models(pkl_path)
    raw_baselines, raw_path = _raw_baselines_by_metric()

    df = pd.read_csv(evidence_csv)
    if "timestamp" not in df.columns:
        raise ValueError("Le CSV evidence doit contenir une colonne 'timestamp'.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    # Spécification utilisateur
    day_mask = (df["hour"] >= 8) & (df["hour"] <= 20)
    night_mask = (df["hour"] < 6) | (df["hour"] >= 22)

    rows = []

    print("=" * 90)
    print("INV-4 | Sensibilité jour/nuit (T_atk relatif au trafic normal)")
    print(f"Evidence CSV: {evidence_csv}")
    print(f"Models PKL  : {pkl_path}")
    print(f"Raw CSV     : {raw_path if raw_path else 'not found (fallback unavailable)'}")
    print("=" * 90)

    for m in METRICS:
        base = _base_metric(m)
        col_n = f"{m}_N"
        if col_n not in df.columns:
            print(f"[SKIP] {m}: colonne absente ({col_n})")
            continue

        col_obs = _pick_existing_col(df, [
            f"{m}_real", f"{base}_real", base, f"{m}_y", "real"
        ])
        col_yhat = _pick_existing_col(df, [
            f"{m}_yhat", f"{m}_pred", f"{base}_yhat", f"{base}_pred", "pred", "yhat"
        ])

        t_atk = _extract_t_atk(models, m)

        day = pd.to_numeric(df.loc[day_mask, col_n], errors="coerce").dropna()
        night = pd.to_numeric(df.loc[night_mask, col_n], errors="coerce").dropna()

        day_med = float(day.median()) if len(day) else float("nan")
        night_med = float(night.median()) if len(night) else float("nan")

        day_q99_n = float(day.quantile(0.99)) if len(day) else float("nan")
        night_q99_n = float(night.quantile(0.99)) if len(night) else float("nan")

        # Baseline observé (trafic normal brut)
        if col_obs is not None:
            obs_day = pd.to_numeric(df.loc[day_mask, col_obs], errors="coerce").dropna()
            obs_night = pd.to_numeric(df.loc[night_mask, col_obs], errors="coerce").dropna()
            obs_day_med = float(obs_day.median()) if len(obs_day) else float("nan")
            obs_night_med = float(obs_night.median()) if len(obs_night) else float("nan")
        else:
            obs_day_med = float("nan")
            obs_night_med = float("nan")

        # Fallback principal demandé: médianes issues du dataset brut.
        if (pd.isna(obs_day_med) or pd.isna(obs_night_med)) and m in raw_baselines:
            obs_day_med = raw_baselines[m]["raw_day_median"]
            obs_night_med = raw_baselines[m]["raw_night_median"]

        # Baseline prédit (yhat)
        if col_yhat is not None:
            yhat_day = pd.to_numeric(df.loc[day_mask, col_yhat], errors="coerce").dropna()
            yhat_night = pd.to_numeric(df.loc[night_mask, col_yhat], errors="coerce").dropna()
            yhat_day_med = float(yhat_day.median()) if len(yhat_day) else float("nan")
            yhat_night_med = float(yhat_night.median()) if len(yhat_night) else float("nan")
        else:
            yhat_day_med = float("nan")
            yhat_night_med = float("nan")

        # Vraie métrique demandée: effort relatif de dépassement du seuil
        # (+% requis vs trafic normal médian).
        atk_effort_obs_day_pct = (100.0 * t_atk / obs_day_med) if pd.notna(t_atk) and pd.notna(obs_day_med) and obs_day_med != 0 else float("nan")
        atk_effort_obs_night_pct = (100.0 * t_atk / obs_night_med) if pd.notna(t_atk) and pd.notna(obs_night_med) and obs_night_med != 0 else float("nan")
        atk_effort_yhat_day_pct = (100.0 * t_atk / yhat_day_med) if pd.notna(t_atk) and pd.notna(yhat_day_med) and yhat_day_med != 0 else float("nan")
        atk_effort_yhat_night_pct = (100.0 * t_atk / yhat_night_med) if pd.notna(t_atk) and pd.notna(yhat_night_med) and yhat_night_med != 0 else float("nan")

        print(
            f"{m:16s} | T_atk={t_atk:>8.4f} | "
            f"effort_day={atk_effort_obs_day_pct:>7.1f}% effort_night={atk_effort_obs_night_pct:>7.1f}% | "
            f"N_q99_day={day_q99_n:>8.4f} N_q99_night={night_q99_n:>8.4f}"
        )

        rows.append(
            {
                "metric": m,
                "evidence_col_N": col_n,
                "obs_col": col_obs,
                "yhat_col": col_yhat,
                "T_atk": t_atk,
                "day_median_N": day_med,
                "night_median_N": night_med,
                "day_q99_N": day_q99_n,
                "night_q99_N": night_q99_n,
                "day_median_obs": obs_day_med,
                "night_median_obs": obs_night_med,
                "raw_day_median": raw_baselines.get(m, {}).get("raw_day_median", float("nan")),
                "raw_night_median": raw_baselines.get(m, {}).get("raw_night_median", float("nan")),
                "day_median_yhat": yhat_day_med,
                "night_median_yhat": yhat_night_med,
                "atk_effort_obs_day_pct": atk_effort_obs_day_pct,
                "atk_effort_obs_night_pct": atk_effort_obs_night_pct,
                "atk_effort_yhat_day_pct": atk_effort_yhat_day_pct,
                "atk_effort_yhat_night_pct": atk_effort_yhat_night_pct,
                "n_day": int(len(day)),
                "n_night": int(len(night)),
            }
        )

    if not rows:
        raise ValueError("Aucune métrique exploitable trouvée pour INV-4.")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
