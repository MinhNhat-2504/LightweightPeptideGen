"""
M3-CAD Trainer: ELBO + Regression (MSE) + Multilabel Classification (BCE) losses.
"""

import os
import sys
import time
import json
import logging
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from baselines.m3cad.model import M3CADModel
from baselines.common.data_utils import get_dataloaders, decode_tokens, get_vocab_info
from baselines.common.metrics import evaluate_generated_sequences

logger = logging.getLogger(__name__)


class M3CADTrainer:
    """
    Training loop for M3-CAD simplified baseline.

    Loss = ELBO + λ_reg*MSE(stability) + λ_cls*BCE(multilabel)
    """

    def __init__(
        self,
        model: M3CADModel,
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
        reg_weight: float = 0.3,
        cls_weight: float = 0.3,
        checkpoint_dir: str = 'baselines/checkpoints/m3cad',
        log_path: str = 'baselines/logs/m3cad.log',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.kl_weight = kl_weight
        self.reg_weight = reg_weight
        self.cls_weight = cls_weight
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        vocab_info = get_vocab_info()
        self.sos_idx = vocab_info['sos_idx']
        self.eos_idx = vocab_info['eos_idx']
        self.pad_idx = vocab_info['pad_idx']

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
        )
        self.scaler = GradScaler(enabled=use_amp)
        self.best_val_loss = float('inf')

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
            format='%(asctime)s %(levelname)s %(message)s',
        )

    def _compute_loss(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        condition = batch['condition'].to(self.device)

        # Raw features for regression/classification targets
        features_raw = batch.get('features_raw', None)
        if features_raw is not None:
            features_raw = features_raw.to(self.device)
            # instability_index is index 0 in CONDITION_FEATURES
            instability_target = features_raw[:, 0]   # raw instability index
            # Multilabel: AMP (label), low_hemolytic (hemolytic_score index 2 < 3), stable (instab < 40)
            amp_label = batch['label'].float().to(self.device) if 'label' in batch \
                else (features_raw[:, 1] > 0).float()
            hemolytic_label = (features_raw[:, 2] < 3).float()     # hemolytic_score < 3
            stable_label = (features_raw[:, 0] < 40).float()       # instability < 40
            multilabels = torch.stack([amp_label, hemolytic_label, stable_label], dim=1)
        else:
            instability_target = torch.zeros(input_ids.size(0), device=self.device)
            multilabels = torch.ones(input_ids.size(0), 3, device=self.device)

        out = self.model(input_ids, condition, target_ids)
        logits = out['logits']          # (B, L, vocab)
        mu, logvar = out['mu'], out['logvar']
        stability_pred = out['stability_pred']
        cls_logits = out['cls_logits']

        B, L, V = logits.shape

        # ELBO components
        rec_loss = F.cross_entropy(
            logits.reshape(B * L, V),
            target_ids.reshape(B * L),
            ignore_index=self.pad_idx,
        )
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Regression loss — normalize instability to ~[0,1]
        reg_loss = F.mse_loss(stability_pred, instability_target / 100.0)

        # Multilabel classification loss
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, multilabels)

        total = (
            rec_loss
            + self.kl_weight * kl_loss
            + self.reg_weight * reg_loss
            + self.cls_weight * cls_loss
        )

        return {'total': total, 'rec': rec_loss, 'kl': kl_loss, 'reg': reg_loss, 'cls': cls_loss}

    def train_epoch(self):
        self.model.train()
        totals = {k: 0.0 for k in ['total', 'rec', 'kl', 'reg', 'cls']}
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

    def validate(self):
        self.model.eval()
        totals = {k: 0.0 for k in ['total', 'rec', 'kl']}
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                with autocast(enabled=self.use_amp):
                    losses = self._compute_loss(batch)
                for k in ['total', 'rec', 'kl']:
                    totals[k] += losses[k].item()
                n += 1
        return {k: v / max(n, 1) for k, v in totals.items()}

    def _sample_condition(self, num_samples):
        conditions = []
        for batch in self.val_loader:
            conditions.append(batch['condition'])
            if sum(c.size(0) for c in conditions) >= num_samples:
                break
        return torch.cat(conditions, dim=0)[:num_samples].to(self.device)

    def generate_sequences(self, num_samples=200):
        self.model.eval()
        with torch.no_grad():
            condition = self._sample_condition(num_samples)
            token_ids = self.model.generate(
                num_samples=num_samples,
                condition=condition,
                sos_idx=self.sos_idx,
                eos_idx=self.eos_idx,
                max_len=52,
                device=self.device,
            )
        return decode_tokens(token_ids)

    def train(self, num_epochs=100, save_every=10, val_frequency=5):
        logger.info(f"[M3-CAD] Training. Params: {self.model.count_parameters():,}")
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
                seqs = self.generate_sequences(200)
                metrics = evaluate_generated_sequences(seqs)
                row['entropy'] = metrics.get('entropy', 0.0)
                row['ngram_div'] = metrics.get('ngram_diversity_2', 0.0)
                row['uniqueness'] = metrics.get('uniqueness_ratio', 0.0)
                logger.info(
                    f"[M3-CAD] Ep {epoch:3d}/{num_epochs} | "
                    f"train={train_losses['total']:.4f} val={val_losses['total']:.4f} "
                    f"rec={val_losses['rec']:.4f} kl={val_losses['kl']:.4f} "
                    f"ent={row['entropy']:.3f} ngram={row['ngram_div']:.4f} ({elapsed:.1f}s)"
                )
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint('best.pt')
            else:
                logger.info(
                    f"[M3-CAD] Ep {epoch:3d}/{num_epochs} | train={train_losses['total']:.4f} ({elapsed:.1f}s)"
                )

            if epoch % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt')
            history.append(row)

        with open(os.path.join(self.checkpoint_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        logger.info("[M3-CAD] Training complete.")
        return history

    def save_checkpoint(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({'model_state_dict': self.model.state_dict(),
                    'best_val_loss': self.best_val_loss}, path)
        logger.info(f"[M3-CAD] Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
