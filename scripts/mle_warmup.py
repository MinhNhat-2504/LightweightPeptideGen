#!/usr/bin/env python
"""
MLE warm-up for the MultimodalFusionGenerator — makes the ESM-2 fusion real.

This is the missing piece that turns the paper's "ESM-2 + GATv2 cross-attention
fusion" from a structural claim into a trained reality. It runs teacher-forced
reconstruction of the real corpus while feeding **frozen ESM-2 token embeddings**
of each sequence into the generator's refinement path. The cross-attention of
Eq.(3) therefore genuinely consumes ESM-2 features, and the shared
fusion/decoder/(z,C)-prior backbone is shaped by them. The resulting checkpoint
is then used to warm-start adversarial training (``train.py --resume``).

After warm-up, set ``model.esm_dim`` in config.yaml to the ESM-2 embed dim used
here (480 for esm2_t12_35M, 1280 for esm2_t33_650M) so the GAN phase rebuilds
the identical architecture and loads these weights in full.

Usage
-----
    python scripts/mle_warmup.py --config config/config.yaml --conditional \
        --esm-model esm2_t12_35M_UR50D --epochs 5 --batch-size 256 \
        --out checkpoints/warmup.pt
    # then:
    # (set model.esm_dim: 480 in config.yaml)
    python train.py --config config/config.yaml --conditional --resume checkpoints/warmup.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from peptidegen.utils import load_config, set_seed, get_device
from peptidegen.data import ConditionalPeptideDataset, get_dataloader, VOCAB
from peptidegen.models import MultimodalFusionGenerator
from peptidegen.models.esm2_hf import ESM2HF, ESM2_EMBED_DIM


def save_warmup_checkpoint(G, path, config):
    """Save in the format train.py --resume / GANTrainer.load expect."""
    model_config = {}
    for attr in ("vocab_size", "embedding_dim", "hidden_dim", "latent_dim",
                 "max_length", "num_layers", "num_heads", "dropout", "condition_dim",
                 "mem_tokens", "esm_dim", "pad_idx", "sos_idx", "eos_idx",
                 "bidirectional", "use_attention"):
        if hasattr(G, attr):
            model_config[attr] = getattr(G, attr)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": 0,
        "global_step": 0,
        "generator_class": type(G).__name__,
        "generator": G.state_dict(),
        "model_config": model_config,
        "config": config,
        "warmup": True,
    }, path)
    logger.info(f"Saved warm-up checkpoint -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--conditional", action="store_true")
    ap.add_argument("--esm-model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--esm-max-length", type=int, default=64)
    ap.add_argument("--contact-graph", action="store_true",
                    help="A2: build KNN <8A residue contact graph from ESM-2 and "
                         "run GATv2 over it (else windowed adjacency)")
    ap.add_argument("--contact-thresh", type=float, default=0.5)
    ap.add_argument("--max-samples", type=int, default=None, help="cap train rows (debug)")
    ap.add_argument("--out", default="checkpoints/warmup.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    device = get_device()
    logger.info(f"Device: {device} | ESM: {args.esm_model}")

    # ---- data ----
    ds = ConditionalPeptideDataset.from_csv(
        data_cfg.get("train_csv", "dataset/train.csv"),
        max_length=data_cfg.get("max_seq_length", 50),
        min_length=data_cfg.get("min_seq_length", 5),
    )
    if args.max_samples:
        ds.sequences = ds.sequences[: args.max_samples]
        ds.features = ds.features[: args.max_samples]
        if ds.has_labels:
            ds.labels = ds.labels[: args.max_samples]
    cond_dim = ds.get_condition_dim() if args.conditional else None
    loader = get_dataloader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=False, drop_last=True)

    # ---- frozen ESM-2 (semantic stream) ----
    esm = ESM2HF(model_name=args.esm_model, device=device, freeze=True)
    esm_dim = esm.embed_dim

    # ---- generator with the ESM refinement pathway enabled ----
    G = MultimodalFusionGenerator(
        vocab_size=VOCAB.vocab_size,
        embedding_dim=model_cfg.get("embedding_dim", 128),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        latent_dim=model_cfg.get("latent_dim", 128),
        max_length=data_cfg.get("max_seq_length", 50),
        num_layers=model_cfg.get("num_layers", 3),
        num_heads=config.get("generator", {}).get("num_heads", 4),
        dropout=model_cfg.get("dropout", 0.2),
        condition_dim=cond_dim,
        mem_tokens=model_cfg.get("mem_tokens", 16),
        esm_dim=esm_dim,
    ).to(device)
    logger.info(f"Generator params: {sum(p.numel() for p in G.parameters()):,} "
                f"(esm_dim={esm_dim})")

    opt = AdamW(G.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pad_idx = VOCAB.pad_idx

    G.train()
    for epoch in range(args.epochs):
        tot, n = 0.0, 0
        for bi, batch in enumerate(loader):
            seqs = batch["sequence"]                       # list[str]
            tgt_in = batch["input_ids"].to(device)         # SOS + seq
            tgt_out = batch["target_ids"].to(device)       # seq + EOS
            cond = batch["condition"].to(device) if args.conditional else None

            with torch.no_grad():
                if args.contact_graph:
                    g = esm.contact_graph(seqs, thresh=args.contact_thresh,
                                          local_window=2, max_length=args.esm_max_length)
                    esm_tok, esm_mask, contact_adj = g["tokens"], g["mask"], g["adj"]
                else:
                    e = esm.embed(seqs, return_tokens=True, max_length=args.esm_max_length)
                    esm_tok, esm_mask, contact_adj = e["tokens"], e["mask"], None

            z = torch.randn(tgt_in.size(0), G.latent_dim, device=device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = G(z, target=tgt_in, condition=cond,
                           esm_tokens=esm_tok, esm_mask=esm_mask,
                           contact_adj=contact_adj)["logits"]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1),
                    ignore_index=pad_idx,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            tot += loss.item(); n += 1
            if (bi + 1) % 50 == 0:
                logger.info(f"epoch {epoch+1} batch {bi+1}/{len(loader)} "
                            f"recon_CE={tot/n:.4f}")
        logger.info(f"== epoch {epoch+1}/{args.epochs} mean recon_CE={tot/max(n,1):.4f} ==")

    save_warmup_checkpoint(G, args.out, config)
    logger.info("Done. Now set `model.esm_dim: %d` in config.yaml and run "
                "train.py --resume %s", esm_dim, args.out)


if __name__ == "__main__":
    main()
