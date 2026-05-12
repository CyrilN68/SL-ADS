"""ablation_qualifier_loo_templates.py — A6.3 Leave-One-Out template ablation
============================================================================

Assumption A6.3 says the conditional opinions ``SBN_COND_OPINIONS[k]``
are correct expert priors — calibrated from CIC-IDS2017, UNSW-NB15,
Kitsune, etc. The qualifier's --sensitivity flag perturbs each cell by
±0.05 to test internal stability, but it does not test what happens
when an entire attack TYPE is missing from the expert table.

This script runs leave-one-attack-out (LOAO):

  - For each attack type k_drop in the catalog:
    - Remove SBN_COND_OPINIONS[k_drop] from the expert table.
    - Re-run the qualifier on every injected attack k_inj.
    - Measure: (a) where do windows of attack k_drop go? (Autre_Anomalie?
      A neighbour attack type? Misclassified?), (b) how does the
      qualification of OTHER attack types degrade?

This isolates the per-template robustness: if removing template k_drop
collapses qualification of other types, the remaining templates are
"borrowing" identifiability from k_drop and the system is fragile.

Outputs:
  - outputs/scientific_hardening/qualifier_loo_results.csv
  - outputs/scientific_hardening/qualifier_loo_summary.json

References
----------
A6.3 in docs/scientific_deconstruction/ASSUMPTIONS.md.
Sharafaldin et al. 2018; Mirsky et al. 2018; Moustafa & Slay 2015 — source datasets.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from sl_ads.config import CONFIG, INJECTED_ATTACK_CATALOG
from sl_ads.paths import get_decision_threshold
from sl_ads.qualify.sbn_qualifier import sbn_qualify_row


def _default_detection_csv() -> Path:
    candidates = [
        Path("outputs/detection_results_INJECTED.csv"),
        Path(CONFIG.get("RESULTS_DIR", "")) / "detection_results_INJECTED.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No detection_results_INJECTED.csv found.")


def _attack_bounds(atk: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    t0 = pd.Timestamp(atk["start"])
    if atk.get("end") is not None:
        return t0, pd.Timestamp(atk["end"])
    return t0, t0 + pd.Timedelta(hours=float(atk["duration_h"]))


def _named_attacks() -> list[dict]:
    return [
        a for a in INJECTED_ATTACK_CATALOG
        if a.get("expected") and not a.get("is_novelty_control", False)
    ]


def _qualify_window(row: pd.Series, sbn_cond: dict, threshold: float,
                    evidence_scale: float, u_raw_thr: float) -> dict:
    return sbn_qualify_row(
        row,
        sbn_cond=sbn_cond,
        threshold=threshold,
        apply_temporal=False,
        apply_um=True,
        evidence_scale=evidence_scale,
        autre_anomalie_prior=u_raw_thr,
    )


def _evaluate_with_table(df: pd.DataFrame, sbn_cond: dict, threshold: float,
                         evidence_scale: float, u_raw_thr: float) -> dict:
    """Per-attack DR / QP / autre_rate using the supplied SBN table."""
    attacks = _named_attacks()
    results = {}
    for atk in attacks:
        t0, t1 = _attack_bounds(atk)
        sub = df[(df["timestamp"] >= t0) & (df["timestamp"] < t1)]
        if sub.empty:
            continue
        n_total = len(sub)
        n_detected = n_qualified = n_correct = n_autre = 0
        top1_counts: dict[str, int] = {}
        for _, row in sub.iterrows():
            r = _qualify_window(row, sbn_cond, threshold,
                                evidence_scale, u_raw_thr)
            if not r.get("gate_open"):
                continue
            n_detected += 1
            qstatus = r.get("qual_status")
            if qstatus == "no_groups":
                continue
            if qstatus == "autre_anomalie":
                n_autre += 1
                continue
            n_qualified += 1
            t1_hit = r.get("top1_type") or ""
            top1_counts[t1_hit] = top1_counts.get(t1_hit, 0) + 1
            if t1_hit == atk["expected"]:
                n_correct += 1
        results[atk["name"]] = {
            "n_total": n_total,
            "n_detected": n_detected,
            "n_qualified": n_qualified,
            "n_correct": n_correct,
            "n_autre": n_autre,
            "dr": n_detected / max(n_total, 1),
            "qp": n_correct / max(n_qualified, 1),
            "autre_rate": n_autre / max(n_detected, 1),
            "top1_distribution": top1_counts,
            "expected": atk["expected"],
        }
    return results


def run_loao(df: pd.DataFrame, base_sbn: dict, threshold: float,
             evidence_scale: float, u_raw_thr: float) -> tuple[pd.DataFrame, dict]:
    base = _evaluate_with_table(df, base_sbn, threshold,
                                 evidence_scale, u_raw_thr)
    rows = []
    summary = {
        "n_attack_types_in_table": len(base_sbn),
        "n_evaluated_attacks": len(base),
        "baseline_micro_qp": np.average(
            [v["qp"] for v in base.values() if v["n_qualified"] > 0]
        ) if base else None,
        "per_attack_baseline_qp": {
            k: v["qp"] for k, v in base.items()
        },
    }

    for k_drop in list(base_sbn.keys()):
        sbn_loo = deepcopy(base_sbn)
        del sbn_loo[k_drop]
        loo = _evaluate_with_table(df, sbn_loo, threshold,
                                    evidence_scale, u_raw_thr)
        # For each attack: did dropping the template change anything?
        for atk_name, b in base.items():
            l = loo.get(atk_name, {})
            row = {
                "k_dropped": k_drop,
                "attack": atk_name,
                "expected": b["expected"],
                "is_dropped_self": k_drop == b["expected"],
                "n_total": b["n_total"],
                "dr_baseline": b["dr"],
                "qp_baseline": b["qp"],
                "autre_baseline": b["autre_rate"],
                "dr_loo": l.get("dr", float("nan")),
                "qp_loo": l.get("qp", float("nan")),
                "autre_loo": l.get("autre_rate", float("nan")),
                "dr_delta": l.get("dr", float("nan")) - b["dr"],
                "qp_delta": l.get("qp", float("nan")) - b["qp"],
                "autre_delta": l.get("autre_rate", float("nan")) - b["autre_rate"],
                "loo_top1_distribution": l.get("top1_distribution", {}),
            }
            rows.append(row)

    df_out = pd.DataFrame(rows)
    # When we drop the expected attack's own template, we expect QP -> 0
    # (no template can match the right answer) and the qualifier should
    # ideally fall back to autre_anomalie or the closest neighbour.
    # The "robustness" claim is about OTHER attacks: their QP should
    # not drop substantially when an unrelated template is removed.
    self_drops = df_out[df_out["is_dropped_self"]]
    other_drops = df_out[~df_out["is_dropped_self"]]
    summary.update({
        "self_drop_qp_loo_mean": float(self_drops["qp_loo"].mean())
            if not self_drops.empty else None,
        "self_drop_autre_loo_mean": float(self_drops["autre_loo"].mean())
            if not self_drops.empty else None,
        "other_drop_qp_delta_mean": float(other_drops["qp_delta"].mean())
            if not other_drops.empty else None,
        "other_drop_qp_delta_max_abs": float(other_drops["qp_delta"].abs().max())
            if not other_drops.empty else None,
        "neighbour_attribution_when_self_dropped": [
            {"attack": row["attack"], "k_dropped_self": row["k_dropped"],
             "top1_when_dropped": row["loo_top1_distribution"]}
            for _, row in self_drops.iterrows()
        ],
    })
    return df_out, summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=None)
    p.add_argument("--out-dir", default="outputs/scientific_hardening")
    p.add_argument("--evidence-scale", type=float,
                   default=float(CONFIG.get("SBN_EVIDENCE_SCALE", 3.0)))
    p.add_argument("--u-raw-threshold", type=float,
                   default=float(CONFIG.get("SBN_NOVELTY_U_RAW_THRESHOLD", 0.82)))
    args = p.parse_args()

    csv_path = Path(args.csv) if args.csv else _default_detection_csv()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    threshold = get_decision_threshold(CONFIG, up_levels=1)
    base_sbn = CONFIG.get("SBN_COND_OPINIONS", {})

    res, summary = run_loao(df, base_sbn, threshold,
                             args.evidence_scale, args.u_raw_threshold)
    out_csv = out_dir / "qualifier_loo_results.csv"
    out_json = out_dir / "qualifier_loo_summary.json"
    # CSV cannot store dicts; convert top1 distribution to JSON string.
    res_csv = res.copy()
    res_csv["loo_top1_distribution"] = res_csv["loo_top1_distribution"].apply(
        lambda d: json.dumps(d, ensure_ascii=False)
    )
    res_csv.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_json}")
    print(json.dumps(summary, indent=2)[:2200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
