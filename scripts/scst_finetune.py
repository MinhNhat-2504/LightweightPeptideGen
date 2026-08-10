#!/usr/bin/env python
"""
A4 — SCST reinforcement-learning fine-tuning with a balanced multi-objective
reward (stability + AMP + low-hemolysis).

Runs *after* GAN training. Loads a fusion-generator checkpoint, optimises the
self-critical policy gradient (paper Eq.10), and saves the fine-tuned generator.
The AMP / hemolysis oracles trained in B6/B7 are used as the reward models;
without them it falls back to physicochemical heuristics (faster, weaker).

Usage
-----
    python scripts/scst_finetune.py \
        --checkpoint checkpoints/best_model.pt \
        --amp-oracle results/oracle_amp.pkl \
        --hemo-oracle results/oracle_hemo.pkl \
        --steps 2000 --batch-size 64 --lr 1e-5 \
        --out checkpoints/scst_model.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from peptidegen.data import VOCAB
from peptidegen.inference import PeptideSampler
from peptidegen.training import SCSTTrainer, MultiObjectiveReward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="GAN generator checkpoint")
    ap.add_argument("--amp-oracle", default=None)
    ap.add_argument("--hemo-oracle", default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--w-stability", type=float, default=0.34)
    ap.add_argument("--w-amp", type=float, default=0.33)
    ap.add_argument("--w-hemolysis", type=float, default=0.33)
    ap.add_argument("--target-ii", type=float, default=40.0)
    ap.add_argument("--entropy-coef", type=float, default=0.02,
                    help="entropy bonus weight; >0 fights mode-collapse/low diversity")
    ap.add_argument("--condition-dim", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out", default="checkpoints/scst_model.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reuse the sampler loader to rebuild the exact generator from the ckpt
    sampler = PeptideSampler.from_checkpoint(args.checkpoint, device=device)
    G = sampler.G

    amp_oracle = hemo_oracle = None
    if args.amp_oracle:
        from peptidegen.evaluation.oracle import ESM2Oracle
        amp_oracle = ESM2Oracle.load(args.amp_oracle, device=device)
    if args.hemo_oracle:
        from peptidegen.evaluation.oracle import ESM2Oracle
        hemo_oracle = ESM2Oracle.load(args.hemo_oracle, device=device)
    if amp_oracle is None and hemo_oracle is None:
        logger.warning("No oracle given -> reward uses physicochemical heuristics.")

    reward = MultiObjectiveReward(
        amp_oracle=amp_oracle, hemo_oracle=hemo_oracle,
        w_stability=args.w_stability, w_amp=args.w_amp, w_hemolysis=args.w_hemolysis,
        target_ii=args.target_ii,
    )
    trainer = SCSTTrainer(G, reward, VOCAB, device=device, lr=args.lr,
                          entropy_bonus=args.entropy_coef)

    cond_dim = getattr(G, "condition_dim", None) or args.condition_dim
    logger.info(f"SCST: {args.steps} steps, batch={args.batch_size}, lr={args.lr}, "
                f"entropy_coef={args.entropy_coef}, "
                f"weights stab/amp/hemo={args.w_stability}/{args.w_amp}/{args.w_hemolysis}")

    run = {"scst_loss": 0.0, "reward_sample": 0.0, "advantage": 0.0, "entropy": 0.0}
    for step in range(1, args.steps + 1):
        conditions = torch.randn(args.batch_size, cond_dim) if cond_dim else None
        m = trainer.train_step(args.batch_size, conditions)
        for k in run:
            run[k] += m[k]
        if step % args.log_every == 0:
            n = args.log_every
            logger.info(f"step {step}/{args.steps} | loss={run['scst_loss']/n:.4f} "
                        f"| R_sample={run['reward_sample']/n:.4f} "
                        f"| adv={run['advantage']/n:.4f} "
                        f"| entropy={run['entropy']/n:.3f}")
            run = {k: 0.0 for k in run}

    # save (sampler-loadable format)
    model_config = {}
    for attr in ("vocab_size", "embedding_dim", "hidden_dim", "latent_dim",
                 "max_length", "num_layers", "num_heads", "dropout", "condition_dim",
                 "mem_tokens", "esm_dim", "pad_idx", "sos_idx", "eos_idx",
                 "bidirectional", "use_attention"):
        if hasattr(G, attr):
            model_config[attr] = getattr(G, attr)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 0, "generator_class": type(G).__name__,
                "generator": G.state_dict(), "model_config": model_config,
                "scst": True}, args.out)
    logger.info(f"Saved SCST-fine-tuned generator -> {args.out}")


if __name__ == "__main__":
    main()
