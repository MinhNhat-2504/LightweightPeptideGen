#!/usr/bin/env python
"""
B6/B7 — Train the ESM-2 oracle(s) and report independent-test metrics.

AMP oracle (uses the project's own labelled corpus):
    python scripts/train_oracle.py amp \
        --train dataset/train.csv --val dataset/val.csv --test dataset/test.csv \
        --model esm2_t12_35M_UR50D --out results/oracle_amp.pkl

Hemolysis oracle (uses an external labelled set you download, HemoPI/DBAASP):
    python scripts/train_oracle.py hemo \
        --train data/hemopi_train.csv --test data/hemopi_test.csv \
        --seq-col sequence --label-col label \
        --model esm2_t12_35M_UR50D --out results/oracle_hemo.pkl

Both write a ``*_report.json`` next to the pickle with ACC/MCC/AUC/Sn/Sp so the
paper can cite the real test-set AUC of the oracle (replacing the heuristic).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from peptidegen.evaluation.oracle import ESM2Oracle


def _load(path, seq_col, label_col):
    df = pd.read_csv(path)
    df = df[[seq_col, label_col]].dropna()
    df[seq_col] = df[seq_col].astype(str).str.upper().str.strip()
    df = df[df[seq_col].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{5,}")]
    return df[seq_col].tolist(), df[label_col].astype(int).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["amp", "hemo"], help="which oracle to train")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--cache-dir", default="results/esm_cache")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    out = args.out or f"results/oracle_{args.task}.pkl"

    tr_s, tr_y = _load(args.train, args.seq_col, args.label_col)
    va_s = va_y = te_s = te_y = None
    if args.val:
        va_s, va_y = _load(args.val, args.seq_col, args.label_col)
    if args.test:
        te_s, te_y = _load(args.test, args.seq_col, args.label_col)

    logger.info(f"[{args.task}] train={len(tr_s)} "
                f"val={len(va_s) if va_s else 0} test={len(te_s) if te_s else 0}")

    oracle = ESM2Oracle(model_name=args.model, cache_dir=args.cache_dir)
    report = oracle.train_and_eval(
        tr_s, tr_y, va_s, va_y, te_s, te_y, batch_size=args.batch_size
    )
    oracle.save(out)

    report_path = Path(out).with_name(Path(out).stem + "_report.json")
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Oracle saved -> {out}")
    logger.info(f"Report saved -> {report_path}")
    for split in ("val", "test"):
        if split in report:
            m = report[split]
            logger.info(f"  [{split}] ACC={m['ACC']:.3f} MCC={m['MCC']:.3f} "
                        f"AUC={m['AUC']:.3f} Sn={m['Sn']:.3f} Sp={m['Sp']:.3f}")


if __name__ == "__main__":
    main()
