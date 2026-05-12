"""
evaluate_qualify_injected.py — v2 (patched)
============================================================
CHANGEMENT PAR RAPPORT À v1 :
    La macro-précision est maintenant calculée sur DEUX populations distinctes,
    conformément à la sémantique du module §8.5 :

    (A) KNOWN_ATTACKS : attaques avec ground-truth de classification
        → précision = fraction de fenêtres détectées correctement classifiées
        → UNKNOWN_ANOMALY_CONTROL est EXCLU de ce calcul

    (B) NOVELTY CONTROL : attaque de contrôle sans ground-truth de classification
        → métrique = u_sbn moyen + recall de détection primaire
        → L'objectif est de vérifier que le système signale "je ne sais pas"

    Ref : Distinção entre détection et classification dans la littérature IDS :
    Sharafaldin et al. (2018, CIC-IDS2017) font explicitement la différence entre
    la tâche de détection (binaire) et la tâche de classification (multiclasse),
    et ne pénalisent pas les métriques de classification sur des classes hors-catalogue.
"""
import argparse
import pandas as pd
import numpy as np
import os
from sl_ads.config import CONFIG  # Phase H
from sl_ads.paths import get_version_names, get_results_dir  # Phase H

parser = argparse.ArgumentParser(description='Évaluation qualification — attaques injectées')
parser.add_argument('--balance_ratio', default=None,
                    help='Variante balance_ratio à évaluer (ex: "auto"). '
                         'Lit qualif_types_balance_<val>.csv. '
                         'Ignoré si --qualif_csv est fourni.')
parser.add_argument('--qualif_csv', default=None,
                    help='Chemin explicite vers le CSV de qualify_anomaly.py.')
args = parser.parse_args()

VERSION_NAME, _ = get_version_names(CONFIG)
RESULTS_DIR = get_results_dir(CONFIG, up_levels=1)

if args.qualif_csv:
    QUALIF_CSV = args.qualif_csv
elif args.balance_ratio:
    QUALIF_CSV = os.path.join(RESULTS_DIR, f'qualif_types_balance_{args.balance_ratio}.csv')
else:
    QUALIF_CSV = os.path.join(RESULTS_DIR, "qualif_types_sbn.csv")

print(f"→ Lecture : {QUALIF_CSV}")

# Seuil de nouveauté : lu depuis config pour cohérence avec qualify_anomaly.py
# Après UM post-qualification, le seuil effectif est plus élevé que l'ancien 0.40.
NOVELTY_THRESHOLD = CONFIG.get('QUALIFY_UM_NOVELTY_THRESHOLD',
                               CONFIG.get('QUALIFY_GATE_THRESHOLD', 0.40))
WINDOW_MIN = CONFIG.get('WINDOW_MINUTES', 5)

df = pd.read_csv(QUALIF_CSV, parse_dates=['timestamp'])

# ===========================================================================
# CATALOGUE : séparer attaques connues vs attaque de contrôle
# ===========================================================================

NOVELTY_CONTROLS = {'UNKNOWN_ANOMALY_CONTROL'}   # ← pas de ground-truth classification


import json
try:
    # PATCH-C1 fix (2026-04-19) : on NE mute PAS la liste importée.
    # INJECTED_ATTACK_CATALOG (config.py) est la source unique de vérité ; elle
    # sert aussi à inject_at_evidence_level.py et compare_qualif_methods.py.
    # Si on faisait .append() dessus, un 2e import dans le même processus
    # (ex. pytest) recevrait un catalogue déjà dupliqué.
    from sl_ads.config import INJECTED_ATTACK_CATALOG as _INJECTED_CATALOG_CANONICAL  # Phase H
except ImportError as _e:
    raise ImportError(
        "config.INJECTED_ATTACK_CATALOG introuvable. Voir PATCH-C1 de "
        "docs/review/SCIENTIFIC_AUDIT.md pour la définition canonique."
    ) from _e

# Construction d'un catalogue LOCAL (jamais la liste canonique importée).
# Logique : on part du catalogue canonique et on le fusionne avec les
# overrides de CONFIG["ATTACK_CATALOG"] (qui peut ajouter des attaques
# déclarées dynamiquement via inject_at_evidence_level mais absentes du
# catalogue statique — en pratique, les deux doivent être en bijection,
# cf. guard in inject_at_evidence_level.py).
ATTACK_CATALOG = [dict(atk) for atk in _INJECTED_CATALOG_CANONICAL]
_canonical_names = {atk["name"] for atk in ATTACK_CATALOG}

# Phase H defensive fix: ``CONFIG["ATTACK_CATALOG"] = None`` is a
# valid state on datasets with no synthetic injection (METR-LA, GECCO,
# CESNET).  ``dict.get(k, [])`` returns None when the key exists with
# a None value, so we use ``or []`` to coerce.
raw_catalog = CONFIG.get("ATTACK_CATALOG") or []
for atk in raw_catalog:
    # Détection automatique du mode "contrôle de nouveauté"
    is_novelty = atk.get("is_novelty_control", False)
    if "UNKNOWN" in atk["name"]:
        is_novelty = True

    if atk["name"] in _canonical_names:
        # Déjà présent — on ne duplique pas. Source canonique = vérité.
        continue

    ATTACK_CATALOG.append({
        "name": atk["name"],
        "expected": None if is_novelty else atk.get("type", "Autre_Anomalie"),
        "start": atk["start"],
        "duration_h": atk.get("duration_h"),
        "end": atk.get("end"),
        "intensity": atk.get("intensity", "unknown"),
        "is_novelty_control": is_novelty
    })

# ===========================================================================
# ÉVALUATION
# ===========================================================================

print(f"\n{'─'*110}")
print(f"  {'Attaque':<28} {'Intens.':<8} {'Fenêtres':<10} "
      f"{'Détect.':<10} {'Recall':<8} {'Qualif.':<10} {'Précision':<10} {'Top-1 incorrect'}")
print(f"{'─'*110}")

known_rows    = []   # attaques avec GT classification
novelty_rows  = []   # attaques de contrôle

for ev in ATTACK_CATALOG:
    t_start = pd.Timestamp(ev['start'])
    if 'duration_h' in ev:
        t_end = t_start + pd.Timedelta(hours=ev['duration_h'])
    else:
        t_end = pd.Timestamp(ev['end'])
    expected = ev['expected']

    mask    = (df['timestamp'] >= t_start) & (df['timestamp'] <= t_end)
    df_ev   = df[mask]

    n_total    = len(df_ev)
    n_detected = int(df_ev['gate_open'].sum())
    recall     = n_detected / n_total if n_total > 0 else 0.0

    u_mean = df_ev[df_ev['gate_open']]['u_sbn'].mean() if n_detected > 0 else float('nan')

    if ev['is_novelty_control']:
        # Pour les contrôles : la "précision" n'a pas de sens → on mesure u_sbn
        # Un u_sbn élevé signifie que le système reconnaît ne pas reconnaître
        novelty_signal = "✓ signal nouveauté" if (not np.isnan(u_mean) and u_mean > NOVELTY_THRESHOLD) else "✗ non signalé"
        print(f"  {ev['name']:<28} {ev['intensity']:<8} {n_total:<10} "
              f"{n_detected:<10} {recall*100:>5.1f}%   {'N/A (contrôle)':<10} {'—':>10}    {novelty_signal}")
        print(f"  {'u_sbn moyen (signal nouveauté)':<34}: {u_mean:.3f}  "
              f"[seuil {NOVELTY_THRESHOLD:.2f}: {'dépassé ✓' if u_mean > NOVELTY_THRESHOLD else 'non dépassé ✗'}]")

        novelty_rows.append({
            'attack': ev['name'], 'n_detected': n_detected,
            'recall': recall, 'u_sbn': u_mean,
            'novelty_detected': u_mean > NOVELTY_THRESHOLD if not np.isnan(u_mean) else False,
        })
    else:
        n_correct = int((df_ev['gate_open'] & (df_ev['top1_type'] == expected)).sum())
        conditional_accuracy = n_correct / n_detected if n_detected > 0 else 0.0 # exactitude conditionnelle ≠ Precision(TP/TP+FP)

        # Erreurs de classification
        wrong = df_ev[df_ev['gate_open'] & (df_ev['top1_type'] != expected)]['top1_type']
        wrong_str = ', '.join(f"{t}({c})" for t, c in wrong.value_counts().items()) if len(wrong) else "—"

        # TTQ : première fenêtre correctement qualifiée
        df_correct = df_ev[df_ev['gate_open'] & (df_ev['top1_type'] == expected)]
        ttq = (df_correct['timestamp'].iloc[0] - t_start).total_seconds() / 60 if len(df_correct) else float('nan')

        print(f"  {ev['name']:<28} {ev['intensity']:<8} {n_total:<10} "
              f"{n_detected:<10} {recall*100:>5.1f}%   {n_correct:<10} {conditional_accuracy*100:>6.1f}%    {wrong_str}")
        print(f"  {'u_sbn moyen':<34}: {u_mean:.3f}")

        known_rows.append({
            'attack': ev['name'], 'expected': expected,
            'intensity': ev['intensity'],
            'n_total': n_total, 'n_detected': n_detected,
            'n_correct': n_correct, 'recall': recall, 'precision': conditional_accuracy,
            'TTQ_min': round(ttq, 1) if not np.isnan(ttq) else None,
            'u_sbn': u_mean,
        })

print(f"{'─'*110}")

# ===========================================================================
# MACRO MOYENNES — clairement séparées
# ===========================================================================

df_known   = pd.DataFrame(known_rows)
df_novelty = pd.DataFrame(novelty_rows)

macro_recall_known    = df_known['recall'].mean()
macro_precision_known = df_known['precision'].mean()

# PATCH-M4 (2026-04-18 / 2026-04-19) : macro-précision UNIQUE.
# L'ancien double reporting [avec / sans ICMP_FLOOD_BURST] constitue une
# cherry-pick implicite (on sélectionnait post-hoc la métrique favorable sans
# critère pré-enregistré). Conformément aux guidelines NeurIPS 2024 sur la
# reproductibilité, seule la macro-précision incluant ICMP_FLOOD_BURST est
# rapportée. Les failure modes par-attaque (dont ICMP_FLOOD_BURST @ 0 %) sont
# documentés individuellement dans la Table 2 per-attack (§Limites).

print(f"\n  [A] ATTAQUES CONNUES ({len(known_rows)} attaques avec ground-truth de classification)")
print(f"      Macro recall      : {macro_recall_known*100:.1f}%")
print(f"      Macro précision   : {macro_precision_known*100:.1f}%   "
      f"(inclut TOUS les failure modes observés — voir Table 2 per-attack)")

print(f"\n  [B] CONTRÔLE NOUVEAUTÉ ({len(novelty_rows)} attaques hors-catalogue)")
for r in novelty_rows:
    print(f"      {r['attack']}: recall={r['recall']*100:.1f}%  "
          f"u_sbn={r['u_sbn']:.3f}  "
          f"signal={'✓' if r['novelty_detected'] else '✗'}")

print(f"\n  NOTE MÉTHODOLOGIQUE :")
print(f"  La macro-précision rapportée ({macro_precision_known*100:.1f}%) inclut toutes")
print(f"  les attaques connues sans exclusion a posteriori. Les failure modes")
print(f"  (ex. ICMP_FLOOD_BURST @ 0 %) sont listés explicitement dans la Table 2")
print(f"  per-attack du rapport. Conforme NeurIPS 2024 §Reproducibility.")

print(f"\n  TTQ par attaque correctement qualifiée :")
for r in known_rows:
    ttq_display = f"{r['TTQ_min']} min" if r['TTQ_min'] is not None else "non qualifié"
    print(f"    {r['attack']:<28}: {ttq_display}")