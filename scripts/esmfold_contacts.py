#!/usr/bin/env python
"""
A2 (rigorous) — ESMFold-derived KNN <8 Å contact graphs (Colab / >=16 GB GPU).

The local pipeline (``ESM2HF.contact_graph``) approximates spatial proximity with
ESM-2 attention contacts. For the strongest, paper-faithful "<8 Å contact radius"
graph, this script folds each sequence with ESMFold, takes Cα coordinates, and
emits a binary adjacency (Cα–Cα distance < threshold Å). The result is cached to
a ``.npz`` keyed by sequence so the warm-up / training can load it offline.

Does NOT fit on an 8 GB laptop GPU — run on Colab.

Usage
-----
    python scripts/esmfold_contacts.py --input dataset/train.fasta \
        --output results/esmfold_contacts.npz --radius 8.0 --max-seqs 5000

Load later:
    import numpy as np
    d = np.load("results/esmfold_contacts.npz", allow_pickle=True)
    adj = d["SEQUENCE_STRING"]   # (L, L) uint8 contact matrix
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output", "-o", default="results/esmfold_contacts.npz")
    ap.add_argument("--radius", type=float, default=8.0, help="Cα–Cα contact radius (Å)")
    ap.add_argument("--max-seqs", type=int, default=5000)
    ap.add_argument("--model", default="facebook/esmfold_v1")
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = EsmForProteinFolding.from_pretrained(args.model).to(device)
    model.esm = model.esm.half()
    model.eval()

    seqs = [s.upper() for s in read_fasta(args.input)][: args.max_seqs]
    seqs = ["".join(c for c in s if c in "ACDEFGHIKLMNPQRSTVWY") for s in seqs]
    seqs = [s for s in seqs if len(s) >= 5]
    logger.info(f"Folding {len(seqs)} sequences for <{args.radius} Å contacts")

    store = {}
    with torch.no_grad():
        for k, seq in enumerate(seqs):
            ids = tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            out = model(ids)
            ca = out["positions"][-1, 0, :, 1, :].float().cpu().numpy()   # (L,3) Cα
            d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)   # (L,L)
            adj = (d < args.radius).astype("uint8")
            np.fill_diagonal(adj, 1)
            store[seq] = adj
            if k % 50 == 0:
                logger.info(f"{k}/{len(seqs)}  L={len(seq)} mean_deg={adj.sum(1).mean():.1f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **store)
    logger.info(f"Saved {len(store)} contact maps -> {args.output}")


if __name__ == "__main__":
    main()
