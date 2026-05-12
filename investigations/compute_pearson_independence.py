"""
compute_pearson_independence.py
Calcule les corrélations de Pearson entre les résidus Prophet et RANSAC
sur le trafic normal uniquement (fenêtres hors attaques).

Usage: python compute_pearson_independence.py

Sortie: matrice de corrélation + interprétation pour §5.4.1
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# PATCH TASK-33 / MIN-03 (audit_tmp, 2026-04-26)
# ──────────────────────────────────────────────────────────────────────────
# L'ancienne liste hardcodée ATTACK_PERIODS dupliquait silencieusement les
# timestamps du catalogue canonique (config.INJECTED_ATTACK_CATALOG +
# REAL_ATTACKS) et pouvait diverger en cas de mise à jour des dates
# d'attaque. On reconstruit désormais la liste depuis la config.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from sl_ads.config import CONFIG, INJECTED_ATTACK_CATALOG, REAL_ATTACKS
from sl_ads.paths import get_results_dir, get_version_names

_version_name, _version_modif = get_version_names(CONFIG)
_results_dir = Path(get_results_dir(CONFIG, up_levels=1))
if not _results_dir.is_absolute():
    _results_dir = (_PROJECT_ROOT / _results_dir).resolve()

_evidence_candidates = [
    _results_dir / CONFIG.get("EVIDENCE_CSV_NAME", f"evidence_{_version_modif}.csv"),
    _results_dir / f"evidence_{_version_name}.csv",
    _results_dir / f"evidence_{_version_modif}.csv",
]
EVIDENCE_CSV = next((p for p in _evidence_candidates if p.exists()), _evidence_candidates[0])


def _build_attack_periods() -> list[tuple[str, str]]:
    """
    Reconstruit ATTACK_PERIODS depuis CONFIG.INJECTED_ATTACK_CATALOG +
    REAL_ATTACKS. Source unique de vérité — évite la duplication silencieuse
    des timestamps (cf. audit MIN-03).
    """
    periods: list[tuple[str, str]] = []
    # 1) Attaques injectées (catalogue principal).
    for ev in INJECTED_ATTACK_CATALOG:
        start = ev.get("start")
        end = ev.get("end")
        if start and end:
            periods.append((str(start), str(end)))
    # 2) Attaques/incidents réels (REAL_ATTACKS).
    if isinstance(REAL_ATTACKS, dict):
        for events in REAL_ATTACKS.values():
            if isinstance(events, dict):
                iterable = [events]
            elif isinstance(events, (list, tuple)):
                iterable = events
            else:
                continue
            for ev in iterable:
                if not isinstance(ev, dict):
                    continue
                start = ev.get("start")
                end = ev.get("end")
                if start and end:
                    periods.append((str(start), str(end)))
    return periods


ATTACK_PERIODS = _build_attack_periods()

print(f"Loading {EVIDENCE_CSV} ...")
df = pd.read_csv(EVIDENCE_CSV, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"  {len(df)} total windows")

mask = pd.Series(True, index=df.index)
for start, end in ATTACK_PERIODS:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    mask &= ~((df["timestamp"] >= s) & (df["timestamp"] <= e))
df_n = df[mask].copy()
print(f"  {len(df_n)} normal windows")

prophet_N_cols = [c for c in df.columns if c.endswith("_N") and c.startswith("prophet_")]
reconst_N_cols = [c for c in df.columns if c.endswith("_N") and c.startswith("reconst_")]
print(f"  Prophet _N: {prophet_N_cols}")
print(f"  RANSAC  _N: {reconst_N_cols}")

print("\n" + "="*70)
print("PEARSON CORRELATIONS — Prophet vs RANSAC (normal traffic only)")
print("="*70)
print(f"{'Prophet':<35} {'RANSAC':<35} {'rho':>7} {'p':>10} {'|rho|<0.3':>10}")
print("-"*70)

results = []
for p in prophet_N_cols:
    for r in reconst_N_cols:
        x = df_n[p].fillna(0).values
        y = df_n[r].fillna(0).values
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            continue
        rho, pval = stats.pearsonr(x, y)
        results.append((p, r, rho, pval))
        flag = "OK" if abs(rho) < 0.3 else "CHECK"
        print(f"{p:<35} {r:<35} {rho:>7.4f} {pval:>10.2e} {flag:>10}")

rhos = [abs(r[2]) for r in results]
print(f"\nSummary: mean|rho|={np.mean(rhos):.4f}, max|rho|={np.max(rhos):.4f}")
print(f"Pairs |rho|<0.3: {sum(1 for r in rhos if r<0.3)}/{len(rhos)}")
max_pair = max(results, key=lambda x: abs(x[2]))
print(f"Most correlated: {max_pair[0]} <-> {max_pair[1]}, rho={max_pair[2]:.4f}")
