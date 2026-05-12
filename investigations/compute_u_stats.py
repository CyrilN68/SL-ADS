"""
compute_u_stats.py
Calcule les statistiques de u_sys sur le trafic normal.

Usage: python compute_u_stats.py <path_to_detection_results_INJECTED.csv>

Sortie : proportions de fenêtres avec u_sys > seuil sur trafic normal uniquement.
"""
import sys
import pandas as pd
import numpy as np
from config import CONFIG

# ── Paramètres ────────────────────────────────────────────────────────────────
_version_name = CONFIG.get("VERSION_NAME", "trained_models_v9_v6_v4s")
_default_results_dir = CONFIG.get("EVAL", {}).get("RESULTS_DIR", f"../results/resultats_{_version_name}")
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else f"{_default_results_dir}/detection_results_INJECTED.csv"
from inject_at_evidence_level import ATTACK_CATALOG as _SYNTH

def _build_attack_ranges():
    ranges = []
    for atk in _SYNTH:
        t0 = pd.Timestamp(atk["start"])
        t1 = t0 + pd.Timedelta(hours=atk["duration_h"])
        ranges.append((str(t0), str(t1)))
    for atk in CONFIG.get("EVAL", {}).get("REAL_ATTACK_CATALOG", []):
        t0 = pd.Timestamp(atk["start"])
        t1 = t0 + pd.Timedelta(hours=atk["duration_h"])
        ranges.append((str(t0), str(t1)))
    return ranges

ATTACK_RANGES = _build_attack_ranges()
COL_BATK  = "FINAL_SYSTEM_CBF_b_atk"
COL_BSUSP = "FINAL_SYSTEM_CBF_b_susp"
COL_BSAFE = "FINAL_SYSTEM_CBF_b_safe"
COL_TS    = "timestamp"

# ── Chargement ────────────────────────────────────────────────────────────────
print(f"Loading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH, parse_dates=[COL_TS])
print(f"  {len(df)} windows loaded")

# ── Calcul de u_sys ───────────────────────────────────────────────────────────
df["u_sys"] = 1.0 - df[COL_BATK] - df[COL_BSUSP] - df[COL_BSAFE]
df["u_sys"] = df["u_sys"].clip(0, 1)  # numerical safety

# ── Masque trafic normal ──────────────────────────────────────────────────────
mask_normal = pd.Series(True, index=df.index)
for start, end in ATTACK_RANGES:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    mask_normal &= ~((df[COL_TS] >= s) & (df[COL_TS] <= e))

df_normal = df[mask_normal].copy()
print(f"  {len(df_normal)} normal windows (attacks excluded)")

# ── Statistiques u_sys sur trafic normal ──────────────────────────────────────
u = df_normal["u_sys"]
print("\n═══ u_sys statistics on NORMAL traffic ═══")
print(f"  Mean u_sys          : {u.mean():.4f}")
print(f"  Median u_sys        : {u.median():.4f}")
print(f"  Std  u_sys          : {u.std():.4f}")
print(f"  Min                 : {u.min():.4f}")
print(f"  Max                 : {u.max():.4f}")
print()
for threshold in [0.3, 0.5, 0.7, 0.9]:
    pct = (u > threshold).mean() * 100
    n   = (u > threshold).sum()
    print(f"  u_sys > {threshold} : {pct:.2f}%  ({n} windows)")

# ── Analyse temporelle : combien de fenêtres au démarrage (cold-start) ───────
print("\n═══ Cold-start analysis (first 50 windows of test period) ═══")
df_sorted = df.sort_values(COL_TS)
first_50 = df_sorted.head(50)["u_sys"]
for threshold in [0.5, 0.7, 0.9]:
    n_above = (first_50 > threshold).sum()
    print(f"  Windows 1-50 with u_sys > {threshold} : {n_above}/50")

# ── Steady-state (after window 100) ──────────────────────────────────────────
print("\n═══ Steady-state (windows 100+, normal traffic only) ═══")
df_normal_sorted = df_normal.sort_values(COL_TS)
steady = df_normal_sorted.iloc[100:]["u_sys"]
print(f"  Mean u_sys (steady) : {steady.mean():.4f}")
pct_high = (steady > 0.5).mean() * 100
print(f"  u_sys > 0.5         : {pct_high:.2f}%")
pct_very_high = (steady > 0.7).mean() * 100
print(f"  u_sys > 0.7         : {pct_very_high:.2f}%")

print("\nDone.")