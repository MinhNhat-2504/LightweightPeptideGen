#!/usr/bin/env python
"""
Train LightweightPeptideGen.

Usage:
    python train.py --config config/config.yaml
    python train.py --config config/config.yaml --conditional
    python train.py --resume checkpoints/best_model.pt --epochs 200
"""

import argparse
import logging
import sys
from pathlib import Path
import torch

# Force UTF-8 stdout so the trainer's ✓/⭐/⚠️ log glyphs don't crash the
# Windows cp1252 console (UnicodeEncodeError).
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Configure logging BEFORE any peptidegen imports to avoid basicConfig no-op
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

from peptidegen.utils import load_config, set_seed, get_device
from peptidegen.data import ConditionalPeptideDataset, get_dataloader, VOCAB
from peptidegen.models import GRUGenerator, CNNDiscriminator, MultimodalFusionGenerator
from peptidegen.training import GANTrainer, ConditionalGANTrainer


def detect_checkpoint_architecture(ckpt: dict) -> dict:
    """Detect model architecture from checkpoint state_dict."""
    gen_key = 'generator' if 'generator' in ckpt else 'generator_state_dict'
    state_dict = ckpt.get(gen_key, {})

    arch = {}

    # Detect bidirectional from presence of reverse weights
    arch['bidirectional'] = any('_reverse' in k for k in state_dict.keys())

    # Detect hidden_dim from GRU weight shape: weight_ih_l0 is (3*hidden_dim, input_dim)
    if 'gru.weight_ih_l0' in state_dict:
        arch['hidden_dim'] = state_dict['gru.weight_ih_l0'].shape[0] // 3

    # Detect embedding_dim from embedding weight: (vocab_size, embedding_dim)
    if 'embedding.weight' in state_dict:
        arch['vocab_size'] = state_dict['embedding.weight'].shape[0]
        arch['embedding_dim'] = state_dict['embedding.weight'].shape[1]

    # Detect num_layers by counting unique layer indices
    layer_indices = set()
    for k in state_dict:
        if k.startswith('gru.weight_ih_l') and '_reverse' not in k:
            idx = k.replace('gru.weight_ih_l', '').split('_')[0]
            layer_indices.add(int(idx))
    if layer_indices:
        arch['num_layers'] = len(layer_indices)

    # Detect total_input_dim from init_hidden or latent_to_hidden
    if 'init_hidden.weight' in state_dict:
        arch['total_input_dim'] = state_dict['init_hidden.weight'].shape[1]
    elif 'latent_to_hidden.0.weight' in state_dict:
        arch['total_input_dim'] = state_dict['latent_to_hidden.0.weight'].shape[1]

    # Detect use_attention
    arch['use_attention'] = any('attention' in k for k in state_dict.keys())

    return arch


def main():
    parser = argparse.ArgumentParser(description='Train LightweightPeptideGen')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--conditional', action='store_true', help='Conditional training')
    parser.add_argument('--resume', type=str, default=None, help='Resume checkpoint')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Cap train/val rows (for quick smoke runs)')
    parser.add_argument('--g-steps', type=int, default=None,
                        help='Override generator steps/iter (lower = faster; '
                             'autoregressive generation is the bottleneck)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Where to save checkpoints (use a separate dir per '
                             'ablation so the main model is not overwritten)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fresh-optimizer', action='store_true',
                        help='Load model weights only; skip optimizer/scaler state '
                             '(use this to recover from corrupted optimizer state after NaN runs)')
    args = parser.parse_args()

    # Config
    config = load_config(args.config)
    training_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})

    # CLI overrides
    if args.g_steps is not None:
        training_cfg['g_steps'] = args.g_steps
        logger.info(f"Override g_steps -> {args.g_steps}")
    device_cfg = config.get('device', {})

    set_seed(args.seed)
    device = get_device()
    logger.info(f"Device: {device}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # =========================================================================
    # DATA
    # =========================================================================
    train_csv = data_cfg.get('train_csv', 'dataset/train.csv')
    val_csv = data_cfg.get('val_csv', 'dataset/val.csv')

    train_dataset = ConditionalPeptideDataset.from_csv(
        train_csv,
        max_length=data_cfg.get('max_seq_length', 50),
        min_length=data_cfg.get('min_seq_length', 5)
    )
    val_dataset = ConditionalPeptideDataset.from_csv(
        val_csv,
        max_length=data_cfg.get('max_seq_length', 50),
        min_length=data_cfg.get('min_seq_length', 5)
    )

    # Optional subset for quick smoke runs
    if args.max_samples:
        def _cap(ds, n):
            ds.sequences = ds.sequences[:n]
            ds.features = ds.features[:n]
            if getattr(ds, 'has_labels', False):
                ds.labels = ds.labels[:n]
        _cap(train_dataset, args.max_samples)
        _cap(val_dataset, max(1, args.max_samples // 5))
        logger.info(f"Subset for quick run: train={len(train_dataset)}, val={len(val_dataset)}")

    condition_dim = train_dataset.get_condition_dim() if args.conditional else 0

    # Batch size: CLI > config
    batch_size = args.batch_size or training_cfg.get('batch_size', 1024)
    num_workers = device_cfg.get('num_workers', 4)
    pin_memory = device_cfg.get('pin_memory', True) and torch.cuda.is_available()

    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = get_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    logger.info(f"Batch size: {batch_size}, Num workers: {num_workers}")
    if args.conditional:
        logger.info(f"Condition dim: {condition_dim}")

    # =========================================================================
    # MODELS - detect architecture from checkpoint when resuming
    # =========================================================================
    resume_condition_dim = condition_dim
    use_bidirectional = model_cfg.get('bidirectional', False)

    if args.resume:
        try:
            ckpt = torch.load(args.resume, map_location='cpu')

            # Prefer saved model_config (new checkpoints), fallback to weight detection (old)
            saved_model_cfg = ckpt.get('model_config', None)

            if saved_model_cfg:
                logger.info(f"Using saved model_config from checkpoint: {saved_model_cfg}")
                use_bidirectional = saved_model_cfg.get('bidirectional', use_bidirectional)
                for key in ('hidden_dim', 'embedding_dim', 'num_layers', 'use_attention',
                            'latent_dim', 'max_length', 'dropout',
                            # carry fusion-generator architecture so a MLE warm-up
                            # checkpoint (with ESM refinement) loads in full
                            'esm_dim', 'mem_tokens', 'num_heads'):
                    if key in saved_model_cfg:
                        model_cfg[key] = saved_model_cfg[key]
                if saved_model_cfg.get('condition_dim') is not None:
                    resume_condition_dim = saved_model_cfg['condition_dim']
            else:
                # Old checkpoint: detect architecture from weight shapes
                arch = detect_checkpoint_architecture(ckpt)
                logger.info(f"Detected checkpoint architecture from weights: {arch}")

                if arch.get('bidirectional') is not None:
                    use_bidirectional = arch['bidirectional']
                if arch.get('hidden_dim') is not None:
                    model_cfg['hidden_dim'] = arch['hidden_dim']
                if arch.get('embedding_dim') is not None:
                    model_cfg['embedding_dim'] = arch['embedding_dim']
                if arch.get('num_layers') is not None:
                    model_cfg['num_layers'] = arch['num_layers']
                if arch.get('use_attention') is not None:
                    model_cfg['use_attention'] = arch['use_attention']

                # Detect latent_dim and condition_dim from total_input_dim
                if arch.get('total_input_dim'):
                    total_input = arch['total_input_dim']
                    if args.conditional and condition_dim > 0:
                        # total_input_dim = latent_dim + condition_dim
                        detected_latent = total_input - condition_dim
                        if detected_latent > 0:
                            model_cfg['latent_dim'] = detected_latent
                            resume_condition_dim = condition_dim
                        else:
                            # condition_dim changed, treat full dim as latent
                            model_cfg['latent_dim'] = total_input
                            resume_condition_dim = 0
                    else:
                        model_cfg['latent_dim'] = total_input

            logger.info(f"Resume model: bidirectional={use_bidirectional}, "
                        f"hidden_dim={model_cfg.get('hidden_dim')}, "
                        f"latent_dim={model_cfg.get('latent_dim')}, "
                        f"num_layers={model_cfg.get('num_layers')}, "
                        f"condition_dim={resume_condition_dim}")
        except Exception as e:
            logger.warning(f"Could not read config from checkpoint: {e}. Using current config.")

    effective_condition_dim = resume_condition_dim if resume_condition_dim > 0 else None

    # Generator architecture selection.
    #   model.architecture: "fusion" (default) -> MultimodalFusionGenerator
    #                       "gru"               -> legacy GRUGenerator (ablation)
    # Also honoured: generator.type == "Transformer"/"GRU".
    gen_cfg = config.get('generator', {})
    arch = model_cfg.get('architecture',
                         'fusion' if gen_cfg.get('type', 'GRU') == 'Transformer' else 'gru')

    if arch == 'fusion':
        generator = MultimodalFusionGenerator(
            vocab_size=VOCAB.vocab_size,
            embedding_dim=model_cfg.get('embedding_dim', 128),
            hidden_dim=model_cfg.get('hidden_dim', 512),
            latent_dim=model_cfg.get('latent_dim', 128),
            max_length=data_cfg.get('max_seq_length', 50),
            num_layers=model_cfg.get('num_layers', 3),
            num_heads=gen_cfg.get('num_heads', 4),
            dropout=model_cfg.get('dropout', 0.2),
            condition_dim=effective_condition_dim or (8 if args.conditional else None),
            mem_tokens=model_cfg.get('mem_tokens', 16),
            gat_heads=config.get('structure_evaluator', {}).get('gat_heads', 4),
            # esm_dim: keep the ESM refinement pathway in the architecture so a
            # checkpoint warm-started by scripts/mle_warmup.py loads in full.
            # Set model.esm_dim in config to the ESM-2 embed dim used for warm-up
            # (e.g. 480 for esm2_t12_35M); leave null for a pure (z,C) prior.
            esm_dim=model_cfg.get('esm_dim'),
        )
        logger.info("Generator: MultimodalFusionGenerator (Transformer + ESM/GATv2 cross-attention fusion)")
    else:
        generator = GRUGenerator(
            vocab_size=VOCAB.vocab_size,
            embedding_dim=model_cfg.get('embedding_dim', 64),
            hidden_dim=model_cfg.get('hidden_dim', 256),
            latent_dim=model_cfg.get('latent_dim', 128),
            num_layers=model_cfg.get('num_layers', 2),
            dropout=model_cfg.get('dropout', 0.2),
            condition_dim=effective_condition_dim,
            bidirectional=use_bidirectional,
            use_attention=model_cfg.get('use_attention', True),
            use_gradient_checkpointing=config.get('device', {}).get('gradient_checkpointing', True),
        )
        logger.info("Generator: GRUGenerator (legacy ablation architecture)")

    disc_cfg = config.get('discriminator', {})
    discriminator = CNNDiscriminator(
        vocab_size=VOCAB.vocab_size,
        embedding_dim=model_cfg.get('embedding_dim', 64),
        hidden_dim=model_cfg.get('hidden_dim', 256),
        num_filters=disc_cfg.get('num_filters', [128, 256, 512]),
        kernel_sizes=disc_cfg.get('kernel_sizes', [3, 5, 7]),
    )

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    logger.info(f"Generator: {g_params:,} params")
    logger.info(f"Discriminator: {d_params:,} params")
    logger.info(f"Total: {g_params + d_params:,} params")

    # =========================================================================
    # TRAINER
    # =========================================================================
    TrainerClass = ConditionalGANTrainer if args.conditional else GANTrainer
    trainer_kwargs = dict(
        generator=generator,
        discriminator=discriminator,
        config=training_cfg,
        device=device,
    )
    if args.conditional:
        trainer_kwargs['condition_dim'] = resume_condition_dim if resume_condition_dim > 0 else 8

    trainer = TrainerClass(**trainer_kwargs)

    start_epoch = 0
    if args.resume:
        load_opt = not getattr(args, 'fresh_optimizer', False)
        start_epoch = trainer.load(args.resume, load_optimizer=load_opt) + 1
        if not load_opt:
            logger.info("Fresh optimizer mode: optimizer and AMP scaler re-initialized")
        trainer.epoch = start_epoch  # Ensure fit() starts from next epoch
        logger.info(f"Resumed from epoch {start_epoch}")

    # =========================================================================
    # TRAIN
    # =========================================================================
    epochs = args.epochs or training_cfg.get('num_epochs', training_cfg.get('epochs', 100))

    # When resuming, ensure we actually train: interpret --epochs as total target
    if start_epoch >= epochs:
        logger.warning(f"start_epoch ({start_epoch}) >= epochs ({epochs}). "
                       f"Will train {epochs} additional epochs from epoch {start_epoch}.")
        epochs = start_epoch + epochs

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    accum_steps = training_cfg.get('gradient_accumulation_steps', 1)
    logger.info("=" * 60)
    logger.info(f"Training for {epochs} epochs (starting from {start_epoch})")
    logger.info(f"Batch size: {batch_size}, Accumulation steps: {accum_steps}")
    logger.info(f"Effective batch: {batch_size * accum_steps}")
    logger.info(f"Config: g_steps={training_cfg.get('g_steps', 5)}, d_steps={training_cfg.get('d_steps', 1)}")
    logger.info(f"Weights: diversity={training_cfg.get('diversity_weight', 0.8)}, "
                f"adv={training_cfg.get('adversarial_weight', 0.4)}")
    logger.info("=" * 60)

    # Early stopping: nếu có val_loader thì dùng val_g_loss, nếu không thì g_loss
    early_stopping_metric = 'val_g_loss' if val_loader is not None else 'g_loss'
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=training_cfg.get('patience', training_cfg.get('early_stopping_patience', 30)),
        early_stopping_metric=early_stopping_metric,
        minimize_metric=True,
    )

    if torch.cuda.is_available():
        logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    logger.info("Training complete!")


if __name__ == '__main__':
    main()
