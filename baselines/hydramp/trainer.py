"""
HydrAMP Trainer: ELBO + AMP/MIC classifier losses.
"""

import os
import sys
import time
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from baselines.hydramp.model import HydrAMPModel
from baselines.common.data_utils import get_dataloaders, decode_tokens, get_vocab_info
from baselines.common.metrics import evaluate_generated_sequences

logger = logging.getLogger(__name__)


class HydrAMPTrainer:
    """
    Training loop for HydrAMP cVAE baseline.

    Loss = ELBO + λ_amp * BCE(amp_logit, amp_label) + λ_mic * BCE(mic_logit, mic_label)
    ELBO = CrossEntropy(reconstruction) - β * KL(q || p)
    """

    def __init__(
        self,
        model: HydrAMPModel,
        train_loader,
        val_loader,
        device: str = 'cuda',
        lr: float = 3e-5,
        beta1: float = 0.5,
        beta2: float = 0.999,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        use_amp: bool = True,
        kl_weight: float = 1.0,
        amp_cls_weight: float = 0.5,
        mic_cls_weight: float = 0.5,
        checkpoint_dir: str = 'baselines/checkpoints/hydramp',
        log_path: str = 'baselines/logs/hydramp.log',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.kl_weight = kl_weight
        self.amp_cls_weight = amp_cls_weight
        self.mic_cls_weight = mic_cls_weight
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Vocab info
        vocab_info = get_vocab_info()
        self.sos_idx = vocab_info['sos_idx']
        self.eos_idx = vocab_info['eos_idx']
        self.pad_idx = vocab_info['pad_idx']

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
        self.scaler = GradScaler(enabled=use_amp)
        self.best_val_loss = float('inf')

        # Logger setup
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(),
            ],
            format='%(asctime)s %(levelname)s %(message)s',
        )

    def _compute_loss(self, batch: dict) -> Dict[str, torch.Tensor]:
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        condition = batch['condition'].to(self.device)

        # AMP label from batch (if exists), otherwise use therapeutic_score > 0
        if 'label' in batch:
            amp_label = batch['label'].float().to(self.device)
        elif 'features_raw' in batch:
            # proxy: therapeutic_score (index 1 in feature order)
            amp_label = (batch['features_raw'][:, 1] > 0).float().to(self.device)
        else:
            amp_label = torch.zeros(input_ids.size(0), device=self.device)

        # MIC proxy: instability_index < 30 (index 0)
        if 'features_raw' in batch:
            mic_label = (batch['features_raw'][:, 0] < 30).float().to(self.device)
        else:
            mic_label = torch.ones(input_ids.size(0), device=self.device)

        out = self.model(input_ids, condition, target_ids)
        logits = out['logits']      # (B, L, vocab)
        mu = out['mu']
        logvar = out['logvar']
        amp_logit = out['amp_logit']
        mic_logit = out['mic_logit']

        B, L, V = logits.shape

        # Reconstruction loss (cross-entropy, ignore PAD)
        rec_loss = F.cross_entropy(
            logits.reshape(B * L, V),
            target_ids.reshape(B * L),
            ignore_index=self.pad_idx,
        )

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Classifier losses
        amp_loss = F.binary_cross_entropy_with_logits(amp_logit, amp_label)
        mic_loss = F.binary_cross_entropy_with_logits(mic_logit, mic_label)

        total_loss = (
            rec_loss
            + self.kl_weight * kl_loss
            + self.amp_cls_weight * amp_loss
            + self.mic_cls_weight * mic_loss
        )

        return {
            'total': total_loss,
            'rec': rec_loss,
            'kl': kl_loss,
            'amp_cls': amp_loss,
            'mic_cls': mic_loss,
        }

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        totals = {'total': 0.0, 'rec': 0.0, 'kl': 0.0, 'amp_cls': 0.0, 'mic_cls': 0.0}
        n = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                losses = self._compute_loss(batch)

            self.scaler.scale(losses['total']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k, v in losses.items():
                totals[k] += v.item()
            n += 1

        return {k: v / max(n, 1) for k, v in totals.items()}

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        totals = {'total': 0.0, 'rec': 0.0, 'kl': 0.0}
        n = 0

        with torch.no_grad():
            for batch in self.val_loader:
                with autocast(enabled=self.use_amp):
                    losses = self._compute_loss(batch)
                for k in ['total', 'rec', 'kl']:
                    totals[k] += losses[k].item()
                n += 1

        return {k: v / max(n, 1) for k, v in totals.items()}

    def _sample_condition(self, num_samples: int) -> torch.Tensor:
        """Sample condition vectors from validation set."""
        conditions = []
        for batch in self.val_loader:
            conditions.append(batch['condition'])
            if sum(c.size(0) for c in conditions) >= num_samples:
                break
        cond = torch.cat(conditions, dim=0)[:num_samples]
        return cond.to(self.device)

    def generate_sequences(self, num_samples: int = 500) -> list:
        """Generate sequences for evaluation."""
        self.model.eval()
        with torch.no_grad():
            condition = self._sample_condition(num_samples)
            token_ids = self.model.generate(
                num_samples=num_samples,
                condition=condition,
                sos_idx=self.sos_idx,
                eos_idx=self.eos_idx,
                max_len=52,
                temperature=1.0,
                top_p=0.9,
                device=self.device,
            )
            return decode_tokens(token_ids)

    def train(self, num_epochs: int = 100, save_every: int = 10, val_frequency: int = 5):
        logger.info(f"[HydrAMP] Starting training. Model params: {self.model.count_parameters():,}")
        history = []

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_losses = self.train_epoch()
            elapsed = time.time() - t0

            row = {'epoch': epoch, 'elapsed': elapsed}
            row.update({f'train_{k}': v for k, v in train_losses.items()})

            if epoch % val_frequency == 0 or epoch == num_epochs:
                val_losses = self.validate()
                row.update({f'val_{k}': v for k, v in val_losses.items()})

                # Generate a small sample for metrics
                seqs = self.generate_sequences(num_samples=200)
                metrics = evaluate_generated_sequences(seqs)
                row['entropy'] = metrics.get('entropy', 0.0)
                row['ngram_div'] = metrics.get('ngram_diversity_2', 0.0)
                row['uniqueness'] = metrics.get('uniqueness_ratio', 0.0)

                logger.info(
                    f"[HydrAMP] Ep {epoch:3d}/{num_epochs} | "
                    f"train_loss={train_losses['total']:.4f} "
                    f"val_loss={val_losses['total']:.4f} "
                    f"rec={val_losses['rec']:.4f} "
                    f"kl={val_losses['kl']:.4f} "
                    f"ent={row['entropy']:.3f} "
                    f"ngram={row['ngram_div']:.4f} "
                    f"({elapsed:.1f}s)"
                )

                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint('best.pt')
            else:
                logger.info(
                    f"[HydrAMP] Ep {epoch:3d}/{num_epochs} | "
                    f"train_loss={train_losses['total']:.4f} ({elapsed:.1f}s)"
                )

            if epoch % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt')

            history.append(row)

        # Save training history
        with open(os.path.join(self.checkpoint_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        logger.info("[HydrAMP] Training complete.")
        return history

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, path)
        logger.info(f"[HydrAMP] Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        logger.info(f"[HydrAMP] Loaded checkpoint from {path}")
