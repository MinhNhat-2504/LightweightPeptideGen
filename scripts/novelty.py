#!/usr/bin/env python
"""
Novelty analysis of generated peptides vs the training set (CPU-only).

Fills the novelty table: corpus size, uniqueness, exact novelty vs train,
approximate novelty at a normalized-Levenshtein threshold delta, and the mean
minimum normalized Levenshtein distance to the training set.

Uses rapidfuzz for fast C++ nearest-neighbour search. Install once:
    pip install rapidfuzz

Usage:
    python scripts/novelty.py --gen-dir results/gen --train-csv dataset/train.csv \
        --out results/novelty.json
"""

import argparse
import glob
import json
import os
import random
import statistics


def read_fasta(path):
    seqs, buf = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if buf:
                    seqs.append("".join(buf)); buf = []
            elif line:
                buf.append(line)
    if buf:
        seqs.append("".join(buf))
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default="results/gen")
    ap.add_argument("--train-csv", default="dataset/train.csv")
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--delta", type=float, default=0.5,
                    help="approx-novelty threshold on normalized Levenshtein distance")
    ap.add_argument("--max-train", type=int, default=0,
                    help="0 = use all training sequences; else sample N (faster, approximate)")
    ap.add_argument("--out", default="results/novelty.json")
    args = ap.parse_args()

    # ---- generated corpus (all seeds in gen-dir) ----
    gen = []
    for fa in sorted(glob.glob(os.path.join(args.gen_dir, "*.fasta"))):
        gen += read_fasta(fa)
    gen = [s.upper().strip() for s in gen if s and s.strip()]
    n = len(gen)
    uniq = len(set(gen))
    if n == 0:
        raise SystemExit(f"No generated sequences found in {args.gen_dir}")

    # ---- training set ----
    import pandas as pd
    tr_full = sorted(set(
        pd.read_csv(args.train_csv)[args.seq_col].astype(str).str.upper().str.strip().tolist()))
    n_train_full = len(tr_full)
    full_set = set(tr_full)                    # for EXACT novelty (always full set)

    # subsample only the list used for the (expensive) Levenshtein NN search
    tr = tr_full
    if args.max_train and len(tr_full) > args.max_train:
        random.seed(42)
        tr = random.sample(tr_full, args.max_train)

    # ---- exact novelty: membership in the FULL training set ----
    exact_novel = sum(1 for s in gen if s not in full_set)

    # ---- min normalized Levenshtein distance to the training set ----
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    mind = []
    for i, s in enumerate(gen):
        best = process.extractOne(s, tr, scorer=Levenshtein.normalized_similarity)
        sim = best[1] if best else 0.0
        if sim > 1.0:                          # some versions return 0-100
            sim /= 100.0
        mind.append(1.0 - sim)                 # normalized distance in [0,1]
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n} sequences compared")

    approx_novel = sum(1 for d in mind if d > args.delta)
    rep = {
        "corpus_size": n,
        "unique_in_corpus": uniq,
        "uniqueness_pct": round(uniq / n * 100, 2),
        "exact_novelty_pct": round(exact_novel / n * 100, 2),
        "delta": args.delta,
        "approx_novelty_pct": round(approx_novel / n * 100, 2),
        "mean_min_levenshtein": round(statistics.mean(mind), 4),
        "n_train_compared": len(tr),
        "n_train_full": n_train_full,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print("\n=== Novelty ===")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
