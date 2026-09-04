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
import hashlib
import json
import logging
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import sklearn
import transformers

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from peptidegen.evaluation.oracle import ESM2Oracle
from peptidegen.data import validate_dataset_build


def _load(path, seq_col, label_col):
    df = pd.read_csv(path)
    df = df[[seq_col, label_col]].dropna()
    df[seq_col] = df[seq_col].astype(str).str.upper().str.strip()
    df = df[df[seq_col].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{5,}")]
    return df[seq_col].tolist(), df[label_col].astype(int).tolist()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file_record(path):
    item = Path(path)
    return {"path": str(item.resolve()), "sha256": _sha256(item)}


def _git_state():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip())
        return {"commit": commit, "dirty_worktree": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty_worktree": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["amp", "hemo"], help="which oracle to train")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--model-revision", default=None,
                    help="immutable Hugging Face commit; required for a reportable oracle")
    ap.add_argument("--cache-dir", default="results/esm_cache")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset-report", default=None)
    ap.add_argument("--allow-unverified-data", action="store_true",
                    help="smoke-test only; mark the oracle report non-reportable")
    args = ap.parse_args()

    out = args.out or f"results/oracle_{args.task}.pkl"

    if not args.model_revision and not args.allow_unverified_data:
        ap.error("reportable oracle training requires --model-revision")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_audit = None
    if not args.val or not args.test:
        if not args.allow_unverified_data:
            ap.error("reportable oracle training requires explicit --val and --test splits")
    else:
        try:
            dataset_audit = validate_dataset_build(
                args.train, args.val, args.test, report_path=args.dataset_report
            )
        except (ValueError, FileNotFoundError) as exc:
            if not args.allow_unverified_data:
                ap.error(str(exc))
            logger.warning("UNVERIFIED DATA smoke-test mode: %s", exc)

    tr_s, tr_y = _load(args.train, args.seq_col, args.label_col)
    va_s = va_y = te_s = te_y = None
    if args.val:
        va_s, va_y = _load(args.val, args.seq_col, args.label_col)
    if args.test:
        te_s, te_y = _load(args.test, args.seq_col, args.label_col)
    for split, labels in (("train", tr_y), ("validation", va_y), ("test", te_y)):
        if labels is not None and set(labels) != {0, 1}:
            ap.error(f"{split} oracle split must contain both binary labels")

    logger.info(f"[{args.task}] train={len(tr_s)} "
                f"val={len(va_s) if va_s else 0} test={len(te_s) if te_s else 0}")

    oracle = ESM2Oracle(
        model_name=args.model, model_revision=args.model_revision,
        cache_dir=args.cache_dir, random_state=args.seed,
    )
    report = oracle.train_and_eval(
        tr_s, tr_y, va_s, va_y, te_s, te_y, batch_size=args.batch_size
    )
    git = _git_state()
    report["reportable"] = bool(
        dataset_audit is not None
        and args.model_revision
        and git.get("commit")
        and git.get("dirty_worktree") is False
    )
    report["dataset_build_audit"] = dataset_audit
    report["task"] = args.task
    report["model"] = args.model
    oracle.save(out)

    report.update({
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": _file_record(out),
        "splits": {
            "train": {**_file_record(args.train), "n": len(tr_s)},
            "validation": ({**_file_record(args.val), "n": len(va_s)} if args.val else None),
            "test": ({**_file_record(args.test), "n": len(te_s)} if args.test else None),
        },
        "columns": {"sequence": args.seq_col, "label": args.label_col},
        "run_metadata": {
            "command": shlex.join(sys.argv),
            "seed": args.seed,
            "git": git,
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "transformers": transformers.__version__,
        },
    })

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
