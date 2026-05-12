"""
extract_real_signature.py
=========================
Lit le CSV de preuves et calcule les valeurs médianes de (P, S, N)
pendant la vraie attaque DDoS pour construire REAL_ATTACK_CATALOG.

Usage : python extract_real_signature.py
Sortie : bloc Python prêt à coller dans config.py
"""

import pandas as pd
import numpy as np
import os
import sys
from config import CONFIG

# =============================================================================
# PARAMÈTRES — adapter si nécessaire
# =============================================================================
_version_name = CONFIG.get("VERSION_NAME", "trained_models_v9_v6_v4s")
EVIDENCE_CSV = f"../results/resultats_{_version_name}/evidence_{_version_name}.csv"

ATTACK_START = "2025-11-12 18:21:13"
ATTACK_END   = "2025-11-13 10:14:36"

# =============================================================================

def main():
    if not os.path.exists(EVIDENCE_CSV):
        print(f"ERROR: Evidence CSV not found: {EVIDENCE_CSV}")
        print("Check EVIDENCE_CSV path above.")
        sys.exit(1)

    print(f"Loading evidence CSV: {EVIDENCE_CSV}")
    df = pd.read_csv(EVIDENCE_CSV, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    t0 = pd.Timestamp(ATTACK_START)
    t1 = pd.Timestamp(ATTACK_END)
    duration_h = round((t1 - t0).total_seconds() / 3600.0, 2)

    mask = (df["timestamp"] >= t0) & (df["timestamp"] < t1)
    atk_df = df.loc[mask]

    if atk_df.empty:
        print(f"ERROR: No data found between {ATTACK_START} and {ATTACK_END}.")
        print(f"  CSV range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
        sys.exit(1)

    n_windows = len(atk_df)
    print(f"\nAttack window: {ATTACK_START} -> {ATTACK_END}")
    print(f"Duration: {duration_h} h | Windows found: {n_windows} (5-min)")

    # Find all metric keys (columns ending with _P, _S, _N)
    p_cols = [c for c in df.columns if c.endswith("_P") and c != "timestamp"]
    metric_keys = [c[:-2] for c in p_cols]  # strip _P

    print(f"\n{'─'*70}")
    print(f"  {'Metric':<35} {'P_med':>7} {'S_med':>7} {'N_med':>7}  {'Signal'}")
    print(f"{'─'*70}")

    sig_dict = {}
    for mk in sorted(metric_keys):
        col_p = f"{mk}_P"
        col_s = f"{mk}_S"
        col_n = f"{mk}_N"
        if not all(c in atk_df.columns for c in [col_p, col_s, col_n]):
            continue

        p_med = round(float(atk_df[col_p].median()), 2)
        s_med = round(float(atk_df[col_s].median()), 2)
        n_med = round(float(atk_df[col_n].median()), 2)

        total = p_med + s_med + n_med
        # Dominant signal: N (attack evidence) fraction
        dominant = "N-dominant" if n_med >= p_med else "P-dominant"

        print(f"  {mk:<35} {p_med:>7.2f} {s_med:>7.2f} {n_med:>7.2f}  {dominant}")
        sig_dict[mk] = (p_med, s_med, n_med)

    # -------------------------------------------------------------------------
    # Print ready-to-paste config block
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  Copy this into config.py → CONFIG['EVAL']['REAL_ATTACK_CATALOG']:")
    print(f"{'='*70}\n")

    print('"REAL_ATTACK_CATALOG": [')
    print('    {')
    print('        "name":       "REAL_DDOS",')
    print('        "type":       "DDoS",')
    print('        "intensity":  "extreme",')
    print(f'        "start":      "{ATTACK_START}",')
    print(f'        "duration_h": {duration_h},')
    print('        "signature": {')
    for mk, (p, s, n) in sig_dict.items():
        print(f'            "{mk}": ({p}, {s}, {n}),')
    print('        }')
    print('    }')
    print('],')

    # -------------------------------------------------------------------------
    # Also print summary stats for the paper
    # -------------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("  SUMMARY STATS (for paper Table - Attack characterization):")
    print(f"{'─'*70}")
    for mk, (p, s, n) in sig_dict.items():
        # Attack intensity = fraction of anomalous evidence
        total = p + s + n
        if total > 0:
            atk_fraction = (s + n) / total
            print(f"  {mk:<35}  anomaly_fraction = {atk_fraction:.1%}")


if __name__ == "__main__":
    main()