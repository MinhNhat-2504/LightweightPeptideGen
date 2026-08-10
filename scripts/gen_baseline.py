#!/usr/bin/env python
"""
Generate sequences from a trained baseline into the SAME FASTA layout the
proposed model uses (``<Name>_seed<N>.fasta``), so that scripts/evaluate_generated.py
scores every model under one identical protocol (same oracle, same stability /
foldability metrics) for a fair Table-5 comparison.

Usage:
    python scripts/gen_baseline.py --model hydramp --name HydrAMP \
        --checkpoint baselines/checkpoints/hydramp/best.pt --num 1000 --out-dir results/gen
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from peptidegen.data.vocabulary import VOCAB


def build_model(name):
    if name == "hydramp":
        from baselines.hydramp.model import HydrAMPModel
        return HydrAMPModel(vocab_size=24, embedding_dim=128, hidden_dim=256,
                            latent_dim=128, condition_dim=8, num_layers=2)
    if name == "m3cad":
        from baselines.m3cad.model import M3CADModel
        return M3CADModel(vocab_size=24, embedding_dim=128, hidden_dim=256,
                          latent_dim=128, condition_dim=8, cond_enc_dim=32, num_layers=2)
    if name == "esm2gen":
        from baselines.esm2gen.model import ESM2DecoderModel
        return ESM2DecoderModel(vocab_size=24, embedding_dim=128, hidden_dim=256,
                                latent_dim=128, esm_projection_dim=128, condition_dim=8)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["hydramp", "m3cad", "esm2gen"])
    ap.add_argument("--name", required=True, help="display name used in FASTA filenames")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1337])
    ap.add_argument("--out-dir", default="results/gen")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    os.makedirs(args.out_dir, exist_ok=True)

    for seed in args.seeds:
        torch.manual_seed(seed)
        seqs = []
        with torch.no_grad():
            for i in range(0, args.num, args.batch_size):
                b = min(args.batch_size, args.num - i)
                cond = torch.randn(b, 8, device=device)        # normalized feature space
                tok = model.generate(num_samples=b, condition=cond,
                                     sos_idx=VOCAB.sos_idx, eos_idx=VOCAB.eos_idx,
                                     max_len=52, temperature=1.0, top_p=0.9, device=device)
                seqs += VOCAB.batch_decode(tok, remove_special_tokens=True)
        path = os.path.join(args.out_dir, f"{args.name}_seed{seed}.fasta")
        with open(path, "w") as f:
            for j, s in enumerate(seqs):
                if s:
                    f.write(f">{args.name}_{seed}_{j}\n{s}\n")
        print(f"[{args.name}] seed {seed}: wrote {sum(1 for s in seqs if s)} seqs -> {path}")


if __name__ == "__main__":
    main()
