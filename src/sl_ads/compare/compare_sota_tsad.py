"""compare_sota_tsad.py — Modern TSAD baselines under the SL-ADS protocol
=========================================================================

Reviewer-grade comparison harness for modern time-series anomaly
detection methods.  Following the TSB-AD 2024 leaderboard, the
candidate baselines are:

  - **TranAD** (Tuli et al. PVLDB 2022) — transformer with adversarial
    training on multivariate residuals.
  - **Anomaly Transformer** (Xu et al. ICLR 2022) — association
    discrepancy between prior and series associations.
  - **TimesNet** (Wu et al. ICLR 2023) — task-general time-series
    backbone, anomaly-detection head.
  - **MOMENT / Chronos / TimesFM** — foundation-model baselines from
    TSB-AD; out of scope unless run on a separate shared GPU.

Protocol (mirroring `evaluate/evaluate_injection.py`):

  1. Train on the pre-split RedeRio data (no labels).
  2. Score every test-window timestamp.
  3. Calibrate the operating threshold to match the SL-ADS FPR target
     (`FPR_TARGET_DECISION = 0.001`) on a held-out *normal-only* window.
  4. Compute headline metrics: F1_micro, MCC, FPR, Precision, TPR,
     plus VUS-PR, VUS-ROC, R-AUC-PR, R-AUC-ROC.
  5. McNemar paired test SL-vs-baseline on the test span.

This module *implements* the protocol but only ships **placeholder
classes** for TranAD / AnomalyTransformer / TimesNet — actual
training requires an external GPU and the public PyTorch implementations
of those papers (https://github.com/imperial-qore/TranAD, etc.).  The
``--mode plan`` flag emits the run plan as JSON; `--mode execute`
expects the external repo to be installed.

Until we run this, the headline paper claim must be:

> "We compare SL-ADS to Isolation Forest at FPR-matched threshold;
>  we do not claim SOTA over modern TSAD methods such as TranAD,
>  Anomaly Transformer or TimesNet.  A reviewer-grade baseline
>  comparison is left to follow-up work."

This note is repeated in `docs/honest_limitations.md` §5.3.5.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sl_ads.config import CONFIG, INJECTED_ATTACK_CATALOG
from sl_ads.evaluate.vus_metrics import vus_summary


@dataclass
class BaselineSpec:
    name: str
    family: str
    paper: str
    public_repo: str
    expected_runtime_h: float
    requires_gpu: bool
    notes: str = ""
    extra: dict = field(default_factory=dict)


BASELINES: list[BaselineSpec] = [
    BaselineSpec(
        name="TranAD",
        family="reconstruction-transformer",
        paper="Tuli et al. PVLDB 2022",
        public_repo="https://github.com/imperial-qore/TranAD",
        expected_runtime_h=3.0,
        requires_gpu=True,
        notes="Multivariate; uses both Prophet residuals and reconstructions as input.",
    ),
    BaselineSpec(
        name="AnomalyTransformer",
        family="transformer-association",
        paper="Xu et al. ICLR 2022",
        public_repo="https://github.com/thuml/Anomaly-Transformer",
        expected_runtime_h=2.0,
        requires_gpu=True,
        notes="Association discrepancy; needs window length tuning for 5-min cadence.",
    ),
    BaselineSpec(
        name="TimesNet",
        family="task-general-backbone",
        paper="Wu et al. ICLR 2023",
        public_repo="https://github.com/thuml/TimesNet",
        expected_runtime_h=4.0,
        requires_gpu=True,
        notes="Task-general; the AD head is a simple MSE reconstruction.",
    ),
    BaselineSpec(
        name="USAD",
        family="autoencoder-adversarial",
        paper="Audibert et al. KDD 2020",
        public_repo="https://github.com/manigalati/usad",
        expected_runtime_h=1.5,
        requires_gpu=False,
        notes="Lightweight; runs on CPU. Useful as 'no-transformer' modern baseline.",
    ),
]


def emit_plan(out_dir: Path) -> Path:
    """Write the canonical baseline-run plan as JSON for reviewers."""
    payload = {
        "dataset": "RedeRio",
        "split_date": str(CONFIG["split_date"]),
        "fpr_target": float(CONFIG.get("FPR_TARGET_DECISION", 0.001)),
        "evaluation_metrics": [
            "F1_micro_pure", "F1_macro_pure", "MCC", "FPR",
            "Precision", "TPR",
            "VUS_PR", "VUS_ROC",
            "R_AUC_PR_at_max", "R_AUC_ROC_at_max",
            "existence_recall",
        ],
        "comparison_test": "McNemar paired (Dietterich 1998)",
        "ci_method": "Block bootstrap, block_length=median(attack_episode_length)=36",
        "baselines": [
            {
                "name": b.name,
                "family": b.family,
                "paper": b.paper,
                "public_repo": b.public_repo,
                "expected_runtime_h": b.expected_runtime_h,
                "requires_gpu": b.requires_gpu,
                "notes": b.notes,
            }
            for b in BASELINES
        ],
        "protocol_steps": [
            "1. Load RedeRio.csv; split at CONFIG['split_date'].",
            "2. For each baseline: train on pre-split data only, no labels.",
            "3. Score every test-span window timestamp.",
            "4. Calibrate threshold on held-out normal pre-split window "
            "to match FPR_TARGET_DECISION.",
            "5. Compute confusion matrix on test span at the calibrated threshold.",
            "6. Run vus_summary on (y_true, y_score) and append to a single CSV.",
            "7. Run McNemar paired test (SL-ADS vs baseline) on the test span.",
        ],
        "output_csv": "../results/sota_tsad_comparison.csv",
        "expected_total_compute": (
            f"{sum(b.expected_runtime_h for b in BASELINES):.1f} hours, "
            f"primarily GPU. Skip MOMENT/Chronos/TimesFM unless dedicated."
        ),
        "open_status": "PENDING — see docs/scientific_deconstruction/ASSUMPTIONS.md A7-bis.",
    }
    out_path = out_dir / "compare_sota_tsad_plan.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_path}")
    return out_path


def _attempt_baseline_import(name: str) -> bool:
    """Best-effort detection of whether a baseline implementation is available."""
    if name == "USAD":
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False
    # Other baselines need external repos that are not installable from PyPI.
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["plan", "probe", "execute"],
                   default="plan",
                   help="plan = emit run plan JSON; probe = check if baselines"
                        " are installed; execute = run available baselines (TODO).")
    p.add_argument("--out-dir", default="outputs/scientific_hardening")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = emit_plan(out_dir)

    if args.mode == "plan":
        return 0

    if args.mode in ("probe", "execute"):
        statuses = {b.name: _attempt_baseline_import(b.name) for b in BASELINES}
        print(json.dumps(statuses, indent=2))
        if args.mode == "probe":
            return 0

    if args.mode == "execute":
        print("[compare_sota_tsad] execute mode: not yet implemented.")
        print("[compare_sota_tsad] Plan emitted at:", plan_path)
        print("[compare_sota_tsad] To run TranAD / AnomalyTransformer / TimesNet,")
        print("[compare_sota_tsad] clone the public repos linked in the plan,")
        print("[compare_sota_tsad] adapt their data loaders to RedeRio.csv,")
        print("[compare_sota_tsad] and append the per-baseline rows to")
        print("[compare_sota_tsad] outputs/sota_tsad_comparison.csv.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
