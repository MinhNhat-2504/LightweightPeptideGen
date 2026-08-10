"""
PepGraphormer Trainer: Reconstruction (CrossEntropy) loss with graph representation conditioning.
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

from baselines.pepgraphormer.model import PepGraphormerDecoderModel
from baselines.common.data_utils import get_dataloaders, decode_tokens, get_vocab_info
from baselines.common.metrics import evaluate_generated_sequences

logger = logging.getLogger(__name__)


class PepGraphormerTrainer:
    """
    Training loop for PepGraphormer generative adaptation baseline.
    Loss: CrossEntropy reconstruction (teacher-forced).
    """

    def __init__(
        self,
        model: PepGraphormerDecoderModel,
        train_loader,
        val_loader,
        device: str = 'cuda',
        lr: float = 3e-5,
        beta1: float = 0.5,
        beta2: float = 0.999,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        use_amp: bool = True,
        checkpoint_dir: str = 'baselines/checkpoints/pepgraphormer',
        log_path: str = 'baselines/logs/pepgraphormer.log',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.grad_clip = grad_clip
        self.use_amp = use_amp
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

        out = self.model(input_ids, condition, target_ids)
        logits = out['logits']   # (B, L, vocab)
        B, L, V = logits.shape

        loss = F.cross_entropy(
            logits.reshape(B * L, V),
            target_ids.reshape(B * L),
            ignore_index=self.pad_idx,
        )
        return loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                loss = self._compute_loss(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                with autocast(enabled=self.use_amp):
                    loss = self._compute_loss(batch)
                total_loss += loss.item()
                n += 1
        return total_loss / max(n, 1)

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
        logger.info(
            f"[PepGraphormer] Training. "
            f"Trainable params: {self.model.count_parameters():,}"
        )
        history = []

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            elapsed = time.time() - t0
            row = {'epoch': epoch, 'train_loss': train_loss, 'elapsed': elapsed}

            if epoch % val_frequency == 0 or epoch == num_epochs:
                val_loss = self.validate()
                row['val_loss'] = val_loss
                seqs = self.generate_sequences(200)
                metrics = evaluate_generated_sequences(seqs)
                row['entropy'] = metrics.get('entropy', 0.0)
                row['ngram_div'] = metrics.get('ngram_diversity_2', 0.0)
                row['uniqueness'] = metrics.get('uniqueness_ratio', 0.0)
                logger.info(
                    f"[PepGraphormer] Ep {epoch:3d}/{num_epochs} | "
                    f"train={train_loss:.4f} val={val_loss:.4f} "
                    f"ent={row['entropy']:.3f} ngram={row['ngram_div']:.4f} ({elapsed:.1f}s)"
                )
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best.pt')
            else:
                logger.info(
                    f"[PepGraphormer] Ep {epoch:3d}/{num_epochs} | "
                    f"train={train_loss:.4f} ({elapsed:.1f}s)"
                )

            if epoch % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt')
            history.append(row)

        with open(os.path.join(self.checkpoint_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        logger.info("[PepGraphormer] Training complete.")
        return history

    def save_checkpoint(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, path)
        logger.info(f"[PepGraphormer] Checkpoint: {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
