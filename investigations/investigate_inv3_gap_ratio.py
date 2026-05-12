"""
INV-3 — Gap ratio (T_atk - T_susp) / T_susp sur les métriques
==============================================================

Objectif:
- Vérifier la robustesse de la zone de transition (>= 10%).
- Détecter les métriques avec overlap potentiel (gap_ratio < 0.10).

Usage:
  cd "actual_ version"
  python3 investigate_inv3_gap_ratio.py

Sorties:
  - ../results/resultats_<VERSION_NAME>/inv3_gap_ratio.csv
    - résumé console avec min_gap_ratio et liste WARNING
"""

import os
import pickle
import warnings
import pandas as pd
from config import CONFIG

warnings.filterwarnings("ignore")

VERSION_NAME = CONFIG.get("VERSION_NAME", "trained_models_v9_v6_v4s")
RESULTS_DIR = CONFIG.get("EVAL", {}).get("RESULTS_DIR", f"../results/resultats_{VERSION_NAME}")
PKL_CANDIDATES = [
    os.path.join(RESULTS_DIR, "trained_models.pkl"),
    f"../trained_models_{VERSION_NAME}.pkl",
]
OUT_CSV = os.path.join(RESULTS_DIR, "inv3_gap_ratio.csv")

def _load_models(path: str):
    # Priorité joblib (comme demandé), fallback pickle.
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def _find_pkl() -> str:
    for p in PKL_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "PKL introuvable. Vérifier PKL_CANDIDATES dans investigate_inv3_gap_ratio.py"
    )


def main():
    pkl_path = _find_pkl()
    models = _load_models(pkl_path)
    rows = []

    # Format A (souhaité initialement) :
    # models['thresholds'][metric] = {'T_susp':..., 'T_atk':...}
    if isinstance(models, dict) and "thresholds" in models and isinstance(models["thresholds"], dict):
        for metric, data in models["thresholds"].items():
            if not isinstance(data, dict):
                continue
            t_susp = data.get("T_susp", data.get("t_susp"))
            t_atk = data.get("T_atk", data.get("t_atk"))
            if t_susp is None or t_atk is None:
                continue
            gap_ratio = (t_atk - t_susp) / t_susp if t_susp > 0 else float("inf")
            rows.append(
                {
                    "metric": metric,
                    "T_susp": float(t_susp),
                    "T_atk": float(t_atk),
                    "gap_ratio": float(gap_ratio),
                    "warning_gap_lt_10pct": bool(gap_ratio < 0.10),
                }
            )

    # Format B (train_v9.py actuel) :
    # models[metric] = {..., 't_susp':..., 't_atk':..., 'thresholds': {'suspect', 'attack'}}
    if not rows and isinstance(models, dict):
        for metric, data in models.items():
            if not isinstance(data, dict):
                continue
            if metric == "empirical_priors":
                continue

            t_susp = data.get("t_susp")
            t_atk = data.get("t_atk")

            if t_susp is None or t_atk is None:
                th = data.get("thresholds", {})
                if isinstance(th, dict):
                    t_susp = th.get("suspect", th.get("T_susp", th.get("t_susp")))
                    t_atk = th.get("attack", th.get("T_atk", th.get("t_atk")))

            if t_susp is None or t_atk is None:
                continue

            gap_ratio = (t_atk - t_susp) / t_susp if t_susp > 0 else float("inf")
            rows.append(
                {
                    "metric": metric,
                    "T_susp": float(t_susp),
                    "T_atk": float(t_atk),
                    "gap_ratio": float(gap_ratio),
                    "warning_gap_lt_10pct": bool(gap_ratio < 0.10),
                }
            )

    if not rows:
        raise ValueError(
            "Aucune métrique valide trouvée. Formats supportés: "
            "models['thresholds'][metric] ou models[metric]['t_susp'/'t_atk']."
        )

    df = pd.DataFrame(rows).sort_values("gap_ratio")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    min_row = df.iloc[0]
    n_warn = int(df["warning_gap_lt_10pct"].sum())

    print("=" * 90)
    print("INV-3 | Gap ratio thresholds")
    print(f"PKL: {pkl_path}")
    print(f"Metrics analysées: {len(df)}")
    print(
        f"Minimum gap ratio = {min_row['gap_ratio']:.4f} "
        f"(metric={min_row['metric']}, T_susp={min_row['T_susp']:.4f}, T_atk={min_row['T_atk']:.4f})"
    )
    print(f"Nombre de métriques avec gap<10%: {n_warn}")

    if n_warn > 0:
        print("\n[WARNING] Métriques avec gap<10%:")
        warn_df = df[df["warning_gap_lt_10pct"]]
        for _, r in warn_df.iterrows():
            print(
                f"  - {r['metric']}: gap={r['gap_ratio']:.4f} "
                f"(T_susp={r['T_susp']:.4f}, T_atk={r['T_atk']:.4f})"
            )
    else:
        print("\nOK: aucune métrique avec gap<10%.")

    print(f"\nSaved: {OUT_CSV}")
    print("=" * 90)


if __name__ == "__main__":
    main()
