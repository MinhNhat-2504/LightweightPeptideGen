#!/usr/bin/env python
"""
Generate sequences from a trained architecture-inspired control into the same FASTA layout the
proposed model uses (``<Name>_seed<N>.fasta``), so that scripts/evaluate_generated.py
scores every model under one identical protocol (same oracle, same stability /
foldability metrics) for a fair Table-5 comparison.

Usage:
    python scripts/gen_baseline.py --model hydramp \
        --checkpoint baselines/checkpoints/hydramp/best.pt --num 1000 --out-dir results/gen
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from peptidegen.data.vocabulary import VOCAB


def resolve_condition_dim(checkpoint_path, explicit):
    """Chieu dieu kien phai khop mo hinh de xuat, khong duoc doan.

    Uu tien --condition-dim; neu khong co thi doc run_record.json ma
    train_baseline.py ghi canh checkpoint. Khong suy tu shape state_dict: duong di
    cua condition_dim khac nhau giua cac model (hydramp qua fc_h0, m3cad qua
    FeatureEncoder), nen quy tac suy se hong am tham khi doi model.
    """
    if explicit is not None:
        return int(explicit)
    record = Path(checkpoint_path).parent / "run_record.json"
    if record.is_file():
        value = json.loads(record.read_text(encoding="utf-8")).get("condition_dim")
        if value:
            return int(value)
    raise SystemExit(
        f"khong xac dinh duoc condition_dim: khong co {record} va khong truyen "
        "--condition-dim. Doi chung phai dung dung chieu dieu kien cua mo hinh de "
        "xuat (schema da duyet: 6); gia tri 8 go cung truoc day khong khop dataset "
        "da dung lai."
    )


def build_model(name, condition_dim):
    if name == "hydramp":
        from baselines.hydramp.model import HydrAMPModel
        return HydrAMPModel(vocab_size=24, embedding_dim=128, hidden_dim=256,
                            latent_dim=128, condition_dim=condition_dim, num_layers=2)
    if name == "m3cad":
        from baselines.m3cad.model import M3CADModel
        return M3CADModel(vocab_size=24, embedding_dim=128, hidden_dim=256,
                          latent_dim=128, condition_dim=condition_dim, cond_enc_dim=32, num_layers=2)
    if name == "esm2gen":
        from baselines.esm2gen.model import ESM2DecoderModel
        return ESM2DecoderModel(vocab_size=24, embedding_dim=128, hidden_dim=256,
                                latent_dim=128, esm_projection_dim=128, condition_dim=condition_dim)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["hydramp", "m3cad", "esm2gen"])
    ap.add_argument("--name", default=None, help="optional non-misleading control label")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1337])
    ap.add_argument("--out-dir", default="results/gen")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--condition-dim", type=int, default=None,
                    help="chieu dieu kien; mac dinh doc tu run_record.json canh checkpoint")
    args = ap.parse_args()

    default_names = {
        "hydramp": "HydrAMPInspiredCVAE",
        "m3cad": "M3CADInspiredMultimodalCVAE",
        "esm2gen": "ESM2DecoderControl",
    }
    args.name = args.name or default_names[args.model]
    if args.name.lower().replace("-", "") in {"hydramp", "m3cad"}:
        ap.error(
            "local baseline code is an architecture-inspired reimplementation; "
            "use a label such as HydrAMPInspiredCVAE or M3CADInspiredMultimodalCVAE"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cond_dim = resolve_condition_dim(args.checkpoint, args.condition_dim)
    print(f"[{args.name}] condition_dim = {cond_dim}")
    model = build_model(args.model, cond_dim)
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
                cond = torch.randn(b, cond_dim, device=device)  # normalized feature space
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
