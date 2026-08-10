#!/usr/bin/env python
"""
C11 — ESM-2 backbone-size ablation + active-parameter accounting.

Two pieces of evidence for the paper's central "lightweight" claim:

  (a) ``oracle``  — AMP-oracle test AUC as the frozen ESM-2 backbone shrinks
      (8M -> 35M -> 150M -> 650M). If a small backbone already discriminates
      well, the 650M model is not essential, supporting the lightweight thesis.
      (Run the same idea for warm-up/generation once training budget allows.)

  (b) ``params``  — exact trainable ("active") parameter count of the fusion
      generator at the paper config, vs the frozen ESM-2 backbones. Use this to
      report Table 1's "~8M active params" honestly with the real number.

Usage
-----
    # (a) oracle AUC vs backbone size
    python scripts/backbone_ablation.py oracle \
        --train dataset/train.csv --test dataset/test.csv \
        --models esm2_t6_8M_UR50D esm2_t12_35M_UR50D esm2_t30_150M_UR50D esm2_t33_650M_UR50D \
        --out results/backbone_oracle.json

    # (b) generator active-param report (no training, no GPU)
    python scripts/backbone_ablation.py params --config config/config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_oracle(args):
    import pandas as pd
    from peptidegen.evaluation.oracle import ESM2Oracle

    def load(p):
        df = pd.read_csv(p)[["sequence", "label"]].dropna()
        df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
        df = df[df["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{5,}")]
        return df["sequence"].tolist(), df["label"].astype(int).tolist()

    tr_s, tr_y = load(args.train)
    te_s, te_y = load(args.test)

    rows = []
    for model in args.models:
        logger.info(f"=== backbone {model} ===")
        oracle = ESM2Oracle(model_name=model, cache_dir=f"results/esm_cache_{model}")
        rep = oracle.train_and_eval(tr_s, tr_y, test_seqs=te_s, test_labels=te_y,
                                    batch_size=args.batch_size)
        esm_params = sum(p.numel() for p in oracle._embedder.model.parameters())
        t = rep.get("test", {})
        rows.append({
            "backbone": model,
            "esm_params": esm_params,
            "embed_dim": oracle._embedder.embed_dim,
            "test_AUC": t.get("AUC"), "test_ACC": t.get("ACC"), "test_MCC": t.get("MCC"),
        })
        logger.info(f"  AUC={t.get('AUC'):.3f} ACC={t.get('ACC'):.3f} "
                    f"(ESM params={esm_params/1e6:.0f}M)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print("\n=== AMP-oracle AUC vs ESM-2 backbone size ===")
    print(f"{'backbone':24s} {'ESM params':>11s} {'AUC':>7s} {'ACC':>7s} {'MCC':>7s}")
    for r in rows:
        print(f"{r['backbone']:24s} {r['esm_params']/1e6:9.0f}M "
              f"{(r['test_AUC'] or 0):7.3f} {(r['test_ACC'] or 0):7.3f} {(r['test_MCC'] or 0):7.3f}")
    logger.info(f"Wrote {args.out}")


def run_params(args):
    from peptidegen.models import MultimodalFusionGenerator, CNNDiscriminator
    from peptidegen.models.esm2_hf import ESM2_EMBED_DIM
    from peptidegen.data import VOCAB
    from peptidegen.utils import load_config

    cfg = load_config(args.config)
    m = cfg.get("model", {})
    d = cfg.get("data", {})
    disc = cfg.get("discriminator", {})

    G = MultimodalFusionGenerator(
        vocab_size=VOCAB.vocab_size,
        embedding_dim=m.get("embedding_dim", 128), hidden_dim=m.get("hidden_dim", 512),
        latent_dim=m.get("latent_dim", 128), max_length=d.get("max_seq_length", 50),
        num_layers=m.get("num_layers", 3), num_heads=cfg.get("generator", {}).get("num_heads", 4),
        dropout=m.get("dropout", 0.2), condition_dim=8, mem_tokens=m.get("mem_tokens", 16),
        esm_dim=m.get("esm_dim"),
    )
    D = CNNDiscriminator(
        vocab_size=VOCAB.vocab_size, embedding_dim=m.get("embedding_dim", 128),
        hidden_dim=m.get("hidden_dim", 512),
        num_filters=disc.get("num_filters", [64, 128, 256]),
        kernel_sizes=disc.get("kernel_sizes", [3, 5, 7]),
    )
    g = sum(p.numel() for p in G.parameters() if p.requires_grad)
    dd = sum(p.numel() for p in D.parameters() if p.requires_grad)

    print("\n=== Active (trainable) parameters — supports Table 1 ===")
    print(f"  Fusion generator (G) : {g:,}  ({g/1e6:.2f} M)")
    print(f"  Discriminator (D)    : {dd:,}  ({dd/1e6:.2f} M)")
    print(f"  TOTAL active (G+D)   : {g+dd:,}  ({(g+dd)/1e6:.2f} M)")
    print("\n=== Frozen ESM-2 backbone (not trained, external) ===")
    for name, dim in ESM2_EMBED_DIM.items():
        print(f"  {name:22s} embed_dim={dim}")
    print("\nReport the G+D total as the paper's 'active parameters'; the ESM-2 "
          "backbone is frozen and shared, so it is not counted as trainable.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    po = sub.add_parser("oracle", help="AMP-oracle AUC vs ESM-2 backbone size")
    po.add_argument("--train", required=True)
    po.add_argument("--test", required=True)
    po.add_argument("--models", nargs="+", default=[
        "esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"])
    po.add_argument("--batch-size", type=int, default=32)
    po.add_argument("--out", default="results/backbone_oracle.json")

    pp = sub.add_parser("params", help="fusion-generator active-parameter report")
    pp.add_argument("--config", default="config/config.yaml")

    args = ap.parse_args()
    if args.cmd == "oracle":
        run_oracle(args)
    else:
        run_params(args)


if __name__ == "__main__":
    main()
