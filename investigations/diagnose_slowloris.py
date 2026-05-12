"""
diagnose_slowloris.py
=====================
Identifie POURQUOI Slowloris est maintenant détecté (v13 vs v12).
Lance APRÈS compute_opinions_v3 sur le dataset avec attaques injectées.

Sortie :
  - Print de l'évolution fenêtre par fenêtre des métriques clés pendant Slowloris
  - Identification de la contribution de chaque métrique / mécanisme

Usage : python diagnose_slowloris.py
"""
import pandas as pd
import numpy as np
import os, sys

try:
    from config import CONFIG
except ImportError:
    print("config.py introuvable"); sys.exit(1)

# ── Charger le CSV de détection ──
RESULTS_DIR = CONFIG["EVAL"]["RESULTS_DIR"]
CSV_PATH    = os.path.join(RESULTS_DIR, "detection_results_INJECTED.csv")
if not os.path.exists(CSV_PATH):
    print(f"Fichier introuvable : {CSV_PATH}"); sys.exit(1)

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])

# ── Fenêtre Slowloris ──
SLOWLORIS_START = pd.Timestamp("2025-12-15 22:00:00")
SLOWLORIS_DURATION_H = 1.0  # depuis le catalogue
SLOWLORIS_END = SLOWLORIS_START + pd.Timedelta(hours=SLOWLORIS_DURATION_H)

# ± contexte
CTX = pd.Timedelta(minutes=30)
mask = (df["timestamp"] >= SLOWLORIS_START - CTX) & (df["timestamp"] <= SLOWLORIS_END + CTX)
df_slow = df[mask].copy()
df_slow["in_attack"] = (df_slow["timestamp"] >= SLOWLORIS_START) & (df_slow["timestamp"] < SLOWLORIS_END)

print(f"Fenêtres analysées : {len(df_slow)} ({df_slow['timestamp'].min()} → {df_slow['timestamp'].max()})")
print(f"Fenêtres Slowloris : {df_slow['in_attack'].sum()}")
print()

# ── Métriques d'intérêt ──
# P(Anom) final
proj_col = "FINAL_SYSTEM_CBF_proj_atk"
if proj_col not in df.columns:
    # Try variations
    for c in df.columns:
        if "proj_atk" in c and "CBF" in c:
            proj_col = c; break

delta = 0.20
df_slow["detected"] = df_slow[proj_col] >= delta

print(f"=== Détection Slowloris (seuil δ={delta}) ===")
print(f"  Fenêtres détectées : {df_slow[df_slow['in_attack']]['detected'].sum()} / {df_slow['in_attack'].sum()}")
print(f"  Couverture        : {df_slow[df_slow['in_attack']]['detected'].mean():.1%}")
print()

# ── TTD ──
attack_windows = df_slow[df_slow["in_attack"]]
detected_windows = attack_windows[attack_windows["detected"]]
if len(detected_windows) > 0:
    first_det = detected_windows["timestamp"].iloc[0]
    ttd = (first_det - SLOWLORIS_START).total_seconds() / 60
    print(f"  TTD : {ttd:.1f} min (première détection : {first_det})")
print()

# ── Évolution des leaf metrics ──
print("=== Évolution des croyances feuilles (P(Anom) par métrique) ===")
leaf_proj_cols = [c for c in df.columns if c.endswith("_proj_atk") and "FINAL" not in c and "METHODE" not in c]
print(f"  {len(leaf_proj_cols)} métriques feuilles trouvées")

# Pendant l'attaque
df_atk_only = df_slow[df_slow["in_attack"]]
if len(df_atk_only) > 0 and leaf_proj_cols:
    print(f"\n  {'Métrique':<45} {'mean P(Anom)':>12} {'max P(Anom)':>12} {'% > δ':>8}")
    print("  " + "-"*82)
    leaf_means = {}
    for c in leaf_proj_cols:
        if c in df_atk_only.columns:
            vals = df_atk_only[c].fillna(0)
            leaf_means[c] = vals.mean()
    for c, mean_val in sorted(leaf_means.items(), key=lambda x: -x[1]):
        vals = df_atk_only[c].fillna(0)
        pct = (vals >= delta).mean()
        print(f"  {c:<45} {mean_val:>12.4f} {vals.max():>12.4f} {pct:>7.1%}")

# ── Conflit H1 pendant Slowloris ──
print("\n=== Conflit H1 (λ_dyn) pendant l'attaque ===")
conflict_cols = [c for c in df.columns if c.endswith("_conflict_K")]
lambda_cols   = [c for c in df.columns if c.endswith("_lambda_dyn")]
if conflict_cols and len(df_atk_only) > 0:
    print(f"  {'Métrique':<45} {'mean K':>10} {'max K':>10} {'mean λ_dyn':>12}")
    print("  " + "-"*80)
    for c in sorted(conflict_cols):
        k_vals = df_atk_only[c].fillna(0) if c in df_atk_only.columns else pd.Series([])
        l_col = c.replace("_conflict_K", "_lambda_dyn")
        l_vals = df_atk_only[l_col].fillna(0.95) if l_col in df_atk_only.columns else pd.Series([0.95]*len(df_atk_only))
        if len(k_vals) > 0:
            print(f"  {c.replace('_conflict_K',''):<45} {k_vals.mean():>10.4f} {k_vals.max():>10.4f} {l_vals.mean():>12.4f}")

# ── Comparaison pré/pendant/post ──
print("\n=== P(Anom) final : avant / pendant / après ===")
pre  = df_slow[(df_slow["timestamp"] < SLOWLORIS_START)]
dur  = df_slow[df_slow["in_attack"]]
post = df_slow[(df_slow["timestamp"] >= SLOWLORIS_END)]

for label, subset in [("Avant  ", pre), ("Pendant", dur), ("Après  ", post)]:
    if len(subset) > 0 and proj_col in subset.columns:
        vals = subset[proj_col].fillna(0)
        print(f"  {label} : mean={vals.mean():.4f}  max={vals.max():.4f}  % > δ = {(vals >= delta).mean():.1%}")

print("\n✓ Diagnostic complet.")