"""
Unified training entry point for all baseline models.

Usage:
    cd /export/users/1173693/iDragonCloud/1/LightweightPeptideGen
    
    # Train HydrAMP
    python baselines/train_baseline.py --model hydramp --epochs 100

    # Train M3-CAD
    python baselines/train_baseline.py --model m3cad --epochs 100

    # Train ESM2-Decoder
    python baselines/train_baseline.py --model esm2gen --epochs 100

    # Train PepGraphormer
    python baselines/train_baseline.py --model pepgraphormer --epochs 100

    python baselines/train_baseline.py --model hydramp --epochs 2 --batch-size 64

    # Train all sequentially
    python baselines/train_baseline.py --model all --epochs 100
"""

import json
import os
import random
import shlex
import sys
import argparse
from datetime import datetime, timezone

import numpy as np
import torch
from pathlib import Path

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from baselines.common.data_utils import get_dataloaders


def write_run_record(args, model_name, train_loader, val_loader):
    """Ghi ban ghi provenance canh checkpoint cua lan chay nay.

    Baseline KHONG di qua duong ong provenance cua du an (khong --dataset-report,
    khong artifact_reportable), nen no khong the mang reportable=true. Ban ghi nay
    khai ro dieu do, va luu chieu dieu kien de gen_baseline.py doc lai -- neu khong
    thi luc sinh chuoi khong con cach nao biet chieu dung.
    """
    dataset = train_loader.dataset
    record = {
        "schema_version": 1,
        "reportable": False,
        "not_reportable_because": (
            "Architecture-inspired control. train_baseline.py has no provenance gate: "
            "no dataset-build report is verified, no artifact_reportable flag is set, "
            "and the code commit is not recorded against the run. These numbers are "
            "controls under a shared protocol, never reportable artifacts of the "
            "proposed model, and never a reproduction of the named published method."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "model": model_name,
        "seed": args.seed,
        "condition_dim": dataset.get_condition_dim(),
        "condition_feature_names": list(getattr(dataset, "feature_names", [])),
        "label_filter": None if args.all_labels else 1,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "n_train": len(dataset),
        "n_val": len(val_loader.dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }
    out = Path(f"baselines/checkpoints/{model_name}/seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[train_baseline] Run record -> {out / 'run_record.json'}")


def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"[train_baseline] GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("[train_baseline] No GPU found, using CPU (training will be slow)")
    return device


# ── HydrAMP ─────────────────────────────────────────────────────────────────

def train_hydramp(args, train_loader, val_loader, device):
    from baselines.hydramp.model import HydrAMPModel
    from baselines.hydramp.trainer import HydrAMPTrainer

    # Chieu dieu kien LAY TU DU LIEU, khong go cung. Schema da duyet la 6;
    # gia tri 8 go cung truoc day khong khop dataset da dung lai.
    cond_dim = train_loader.dataset.get_condition_dim()
    model = HydrAMPModel(
        vocab_size=24,
        embedding_dim=128,
        hidden_dim=256,
        latent_dim=128,
        condition_dim=cond_dim,
        num_layers=2,
        dropout=0.2,
        pad_idx=0,
    )
    print(f"[HydrAMP] Parameters: {model.count_parameters():,}")

    trainer = HydrAMPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=3e-5,
        beta1=0.5,
        beta2=0.999,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_amp=(device == 'cuda'),
        kl_weight=1.0,
        amp_cls_weight=0.5,
        mic_cls_weight=0.5,
        checkpoint_dir=f'baselines/checkpoints/hydramp/seed{args.seed}',
        log_path=f'baselines/logs/hydramp_seed{args.seed}.log',
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(
        num_epochs=args.epochs,
        save_every=args.save_every,
        val_frequency=args.val_frequency,
    )


# ── M3-CAD ──────────────────────────────────────────────────────────────────

def train_m3cad(args, train_loader, val_loader, device):
    from baselines.m3cad.model import M3CADModel
    from baselines.m3cad.trainer import M3CADTrainer

    # Chieu dieu kien LAY TU DU LIEU, khong go cung. Schema da duyet la 6;
    # gia tri 8 go cung truoc day khong khop dataset da dung lai.
    cond_dim = train_loader.dataset.get_condition_dim()
    model = M3CADModel(
        vocab_size=24,
        embedding_dim=128,
        hidden_dim=256,
        latent_dim=128,
        condition_dim=cond_dim,
        cond_enc_dim=32,
        num_layers=2,
        dropout=0.2,
        pad_idx=0,
    )
    print(f"[M3-CAD] Parameters: {model.count_parameters():,}")

    trainer = M3CADTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=3e-5,
        beta1=0.5,
        beta2=0.999,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_amp=(device == 'cuda'),
        kl_weight=1.0,
        reg_weight=0.3,
        cls_weight=0.3,
        checkpoint_dir=f'baselines/checkpoints/m3cad/seed{args.seed}',
        log_path=f'baselines/logs/m3cad_seed{args.seed}.log',
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(
        num_epochs=args.epochs,
        save_every=args.save_every,
        val_frequency=args.val_frequency,
    )


# ── ESM2-Decoder ─────────────────────────────────────────────────────────────

def train_esm2gen(args, train_loader, val_loader, device):
    from baselines.esm2gen.model import ESM2DecoderModel
    from baselines.esm2gen.trainer import ESM2DecoderTrainer

    # Chieu dieu kien LAY TU DU LIEU, khong go cung. Schema da duyet la 6;
    # gia tri 8 go cung truoc day khong khop dataset da dung lai.
    cond_dim = train_loader.dataset.get_condition_dim()
    model = ESM2DecoderModel(
        vocab_size=24,
        embedding_dim=128,
        hidden_dim=256,
        latent_dim=128,
        esm_projection_dim=128,
        condition_dim=cond_dim,
        num_layers=2,
        dropout=0.2,
        pad_idx=0,
    )
    trainable = model.count_parameters()
    total = model.count_all_parameters()
    print(f"[ESM2-Decoder] Trainable: {trainable:,} | Total: {total:,}")

    trainer = ESM2DecoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=3e-5,
        beta1=0.5,
        beta2=0.999,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_amp=(device == 'cuda'),
        checkpoint_dir=f'baselines/checkpoints/esm2gen/seed{args.seed}',
        log_path=f'baselines/logs/esm2gen_seed{args.seed}.log',
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(
        num_epochs=args.epochs,
        save_every=args.save_every,
        val_frequency=args.val_frequency,
    )


# ── PepGraphormer ────────────────────────────────────────────────────────────

def train_pepgraphormer(args, train_loader, val_loader, device):
    from baselines.pepgraphormer.model import PepGraphormerDecoderModel
    from baselines.pepgraphormer.trainer import PepGraphormerTrainer
    
    # Chieu dieu kien LAY TU DU LIEU, khong go cung. Schema da duyet la 6;
    # gia tri 8 go cung truoc day khong khop dataset da dung lai.
    cond_dim = train_loader.dataset.get_condition_dim()
    model = PepGraphormerDecoderModel(
        vocab_size=24,
        embedding_dim=128,
        hidden_dim=256,
        latent_dim=128,
        enc_projection_dim=128,
        condition_dim=cond_dim,
        num_layers=2,
        dropout=0.2,
        pad_idx=0,
    )
    print(f"[PepGraphormer] Parameters: {model.count_parameters():,}")
    
    trainer = PepGraphormerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=3e-5,
        beta1=0.5,
        beta2=0.999,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_amp=(device == 'cuda'),
        checkpoint_dir=f'baselines/checkpoints/pepgraphormer/seed{args.seed}',
        log_path=f'baselines/logs/pepgraphormer_seed{args.seed}.log',
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    trainer.train(
        num_epochs=args.epochs,
        save_every=args.save_every,
        val_frequency=args.val_frequency,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train baseline peptide generation models')
    parser.add_argument('--model', type=str, default='hydramp',
                        choices=['hydramp', 'm3cad', 'esm2gen', 'pepgraphormer', 'all'],
                        help='Which baseline model to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--val-frequency', type=int, default=5,
                        help='Run validation every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    # Duong dan legacy (dataset/train.csv, dataset/val.csv) da bi loai bo: README
    # cam dung chung cho ket qua bao cao, va file khong con ton tai. Luu y ten khac
    # nhau -- validation.csv, khong phai val.csv.
    parser.add_argument('--train-csv', type=str,
                        default='dataset/rebuilt_2026-09-10/train.csv')
    parser.add_argument('--val-csv', type=str,
                        default='dataset/rebuilt_2026-09-10/validation.csv')
    parser.add_argument('--seed', type=int, default=42,
                        help='Training seed. R1.5 takes the independent TRAINING seed '
                             'as the unit of analysis, so each control needs one run '
                             'per seed; sampling seeds in gen_baseline.py are not a '
                             'substitute and would be pseudoreplicates of one run.')
    parser.add_argument('--all-labels', action='store_true',
                        help='Train on every label instead of antimicrobial rows only. '
                             'The proposed model uses label_value=1, so leave this off '
                             'for a matched comparison.')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Cap train rows for a fair subset comparison (0 = all)')
    args = parser.parse_args()

    # Dat seed truoc khi tao dataloader: thu tu shuffle phu thuoc no.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[train_baseline] Seed: {args.seed}")

    device = get_device()

    print(f"\n[train_baseline] Model: {args.model.upper()} | "
          f"Epochs: {args.epochs} | Batch: {args.batch_size}\n")

    # Create dataloaders
    train_loader, val_loader = get_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
        max_samples=args.max_samples,
        label_value=None if args.all_labels else 1,
    )

    # Train selected model(s)
    models_to_train = (
        ['hydramp', 'm3cad', 'esm2gen', 'pepgraphormer']
        if args.model == 'all' else [args.model]
    )

    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"  Training: {model_name.upper()}")
        print(f"{'='*60}\n")

        if model_name == 'hydramp':
            train_hydramp(args, train_loader, val_loader, device)
        elif model_name == 'm3cad':
            train_m3cad(args, train_loader, val_loader, device)
        elif model_name == 'esm2gen':
            train_esm2gen(args, train_loader, val_loader, device)
        elif model_name == 'pepgraphormer':
            train_pepgraphormer(args, train_loader, val_loader, device)

        write_run_record(args, model_name, train_loader, val_loader)

    print("\n[train_baseline] Done!")


if __name__ == '__main__':
    main()
