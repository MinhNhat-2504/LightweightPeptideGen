#!/usr/bin/env python
"""
C10 — Controllability evaluation CLI.

Measures whether the conditional generator obeys requested physicochemical
targets. Produces, per feature, the Spearman/Pearson correlation between target
and achieved value + MAE, and a per-target table. Use the results to add a
"Controllability" subsection/table to the paper (substantiates the conditioning
contribution that is currently asserted but unmeasured).

Usage
-----
    python scripts/controllability.py \
        --checkpoint checkpoints/scst_model.pt \
        --train-csv dataset/train.csv \
        --features charge_at_pH7 instability_index gravy aromaticity \
        --n-per 200 --out results/controllability.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from peptidegen.inference import PeptideSampler
from peptidegen.evaluation.controllability import ControllabilityEvaluator, MEASURE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-csv", default="dataset/train.csv")
    ap.add_argument("--features", nargs="*", default=None,
                    help=f"subset of {list(MEASURE)} (default: all measurable)")
    ap.add_argument("--n-per", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--out", default="results/controllability.json")
    args = ap.parse_args()

    sampler = PeptideSampler.from_checkpoint(args.checkpoint)
    ce = ControllabilityEvaluator.from_train_csv(sampler, args.train_csv)

    report = ce.sweep_all(features=args.features, n_per=args.n_per,
                          temperature=args.temperature, top_p=args.top_p)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote controllability report -> {args.out}")

    print("\n=== Controllability (target -> achieved) ===")
    print(f"{'feature':22s} {'Spearman':>9s} {'Pearson':>9s} {'MAE':>10s}")
    for feat, r in report.items():
        sp = r.get("spearman", float("nan"))
        pe = r.get("pearson", float("nan"))
        mae = r.get("mae", float("nan"))
        print(f"{feat:22s} {sp:9.3f} {pe:9.3f} {mae:10.3f}")


if __name__ == "__main__":
    main()
