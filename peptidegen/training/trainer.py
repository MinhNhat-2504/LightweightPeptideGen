"""
GANTrainer - Clean, modular GAN trainer for peptide generation.

Features:
    - Adaptive D training (skip when D too strong)
    - Strong diversity loss to prevent mode collapse
    - Label smoothing and noise injection
    - Automatic mixed precision (AMP)
    - Checkpoint management
    - Early stopping with validation
    - Comprehensive metrics tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from pathlib import Path
from typing import Dict, Optional, Any, List
import logging
import time
import json

from .losses import (
    DiversityLoss,
    FeatureMatchingLoss,
    ReconstructionLoss,
    NgramDiversityLoss,
    LengthPenaltyLoss,
    StabilityBiasLoss,
)

logger = logging.getLogger(__name__)


class GANTrainer:
    """
    GAN Trainer for peptide generation with anti-mode-collapse mechanisms.

    Args:
        generator: Generator model
        discriminator: Discriminator model
        config: Training config dict with keys:
            - learning_rate: G learning rate (default: 0.0003)
            - lr_discriminator: D learning rate (default: 0.00002)
            - g_steps: Generator steps per iteration (default: 5)
            - d_steps: Discriminator steps per iteration (default: 1)
            - label_smoothing: Label smoothing factor (default: 0.3)
            - noise_std: Noise injection std (default: 0.1)
            - diversity_weight: Diversity loss weight (default: 0.8)
            - adversarial_weight: Adversarial loss weight (default: 0.4)
            - use_amp: Use automatic mixed precision (default: True)
        device: PyTorch device
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        # Models
        self.G = generator.to(self.device)
        self.D = discriminator.to(self.device)
        self.accept_conditions = False

        # Training config (extract before _init_hyperparams)
        training_cfg = config.get('training', config)  # Use full config as fallback for compatibility

        # Hyperparameters
        self._init_hyperparams(training_cfg)

        # Optimizers
        self._init_optimizers()

        # Losses
        self.diversity_loss = DiversityLoss(
            entropy_weight=0.4,
            batch_sim_weight=0.3,
            pairwise_weight=0.3,
        )
        self.ngram_loss = NgramDiversityLoss(bigram_weight=0.5, trigram_weight=0.5)

        # Scaler
        self.use_amp = training_cfg.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # Length penalty loss (needs target_len_min/max from _init_hyperparams)
        self.length_penalty_loss = LengthPenaltyLoss(
            eos_idx=2,
            target_min=self.target_len_min,
            target_max=self.target_len_max,
        )

        # Stability bias loss (differentiable instability index penalty)
        # Vocab is injected lazily in train_step on first use
        self.stability_loss = StabilityBiasLoss(vocab=None)

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.best_val_loss = float('inf')
        self.history: List[Dict] = []
        self.d_skip_count = 0

    def _init_hyperparams(self, cfg: Dict):
        """Initialize hyperparameters from config."""
        # Learning rates
        self.lr_g = cfg.get('learning_rate', 0.0003)
        self.lr_d = cfg.get('lr_discriminator', 0.00002)

        # Training steps
        self.g_steps = cfg.get('g_steps', 5)
        self.d_steps = cfg.get('d_steps', 1)

        # Gradient accumulation
        self.accum_steps = cfg.get('gradient_accumulation_steps', 1)

        # Regularization
        self.label_smooth = cfg.get('label_smoothing', 0.3)
        self.noise_std = cfg.get('noise_std', 0.1)

        # Loss weights
        self.w_adv = cfg.get('adversarial_weight', 0.4)
        self.w_rec = cfg.get('reconstruction_weight', 0.0)
        self.w_div = cfg.get('diversity_weight', 0.8)
        self.w_fm = cfg.get('feature_matching_weight', 0.3)
        self.w_ngram = cfg.get('ngram_weight', 0.0)
        self.w_length = cfg.get('length_penalty_weight', 0.0)
        self.w_stability = cfg.get('stability_weight', 0.0)
        self.target_stability_ii = cfg.get('target_stability_ii', 30.0)
        # Target length range for LengthPenaltyLoss
        self.target_len_min = cfg.get('target_length_min', 10)
        self.target_len_max = cfg.get('target_length_max', 30)

        # Adaptive D thresholds
        self.d_gap_threshold = cfg.get('d_threshold', cfg.get('d_gap_threshold', 5.0))
        self.g_steps_boost = cfg.get('g_steps_boost', 3.0)

        # --- A3: discrete relaxation + GAN objective (config-selectable) ---
        # discrete_relaxation: 'softmax' (legacy) | 'gumbel' (straight-through)
        self.discrete_relax = cfg.get('discrete_relaxation', 'softmax')
        self.gumbel_tau_start = cfg.get('gumbel_tau_start', cfg.get('gumbel_tau', 1.0))
        self.gumbel_tau_end = cfg.get('gumbel_tau_end', self.gumbel_tau_start)
        self.gumbel_anneal_epochs = max(1, int(cfg.get('gumbel_anneal_epochs', 1)))
        self.gumbel_tau = self.gumbel_tau_start
        # gan_loss: 'bce' (non-saturating, legacy) | 'wgan_gp' (paper Eq.8-9)
        self.gan_loss = cfg.get('gan_loss', 'bce')
        self.lambda_gp = cfg.get('lambda_gp', 10.0)

    def _init_optimizers(self):
        """Initialize optimizers."""
        self.opt_G = AdamW(
            self.G.parameters(),
            lr=self.lr_g,
            betas=(0.5, 0.999),
            weight_decay=1e-4
        )
        self.opt_D = AdamW(
            self.D.parameters(),
            lr=self.lr_d,
            betas=(0.5, 0.999),
            weight_decay=1e-4
        )
        # LR Scheduler to avoid plateauing
        self.scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_G, mode='min', factor=0.5, patience=15, min_lr=1e-6
        )

    # =========================================================================
    # CORE TRAINING
    # =========================================================================

    def _relax(self, logits: torch.Tensor) -> torch.Tensor:
        """Map generator logits to a (B,L,V) D-input via the chosen relaxation.

        'gumbel' -> straight-through Gumbel-Softmax (discrete forward, soft
        gradient); 'softmax' -> plain softmax (legacy).
        """
        if self.discrete_relax == 'gumbel':
            return F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True, dim=-1)
        return F.softmax(logits, dim=-1)

    def _set_gumbel_temperature(self, epoch: int) -> None:
        """Apply a pre-specified exponential temperature schedule."""
        if self.gumbel_anneal_epochs <= 1 or self.gumbel_tau_start == self.gumbel_tau_end:
            self.gumbel_tau = float(self.gumbel_tau_start)
            return
        fraction = min(max(epoch, 0) / (self.gumbel_anneal_epochs - 1), 1.0)
        self.gumbel_tau = float(
            self.gumbel_tau_start
            * (self.gumbel_tau_end / self.gumbel_tau_start) ** fraction
        )

    def _reconstruction_loss(self, real_inputs: torch.Tensor,
                             targets: Optional[torch.Tensor],
                             conditions: Optional[torch.Tensor],
                             deterministic_latent: bool = False) -> torch.Tensor:
        """Teacher-forced next-token CE used during training and selection."""
        if targets is None:
            targets = torch.roll(real_inputs, shifts=-1, dims=1)
            targets[:, -1] = getattr(self.G, 'pad_idx', 0)
        z = torch.zeros(real_inputs.size(0), self.G.latent_dim, device=self.device)
        if not deterministic_latent:
            z.normal_()
        output = self.G(z, target=real_inputs, condition=conditions)
        logits = output['logits'] if isinstance(output, dict) else output
        length = min(logits.size(1), targets.size(1))
        return F.cross_entropy(
            logits[:, :length].reshape(-1, logits.size(-1)),
            targets[:, :length].reshape(-1),
            ignore_index=getattr(self.G, 'pad_idx', 0),
        )

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """WGAN-GP gradient penalty on real/fake (B,L,V) interpolates."""
        if real.shape != fake.shape:
            raise ValueError(
                "WGAN-GP requires identically shaped real and fake tensors; "
                f"received real={tuple(real.shape)} and fake={tuple(fake.shape)}"
            )
        B = real.size(0)
        alpha = torch.rand(B, 1, 1, device=real.device)
        inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_inter = self.D(inter)
        grads = torch.autograd.grad(
            outputs=d_inter, inputs=inter,
            grad_outputs=torch.ones_like(d_inter),
            create_graph=True, retain_graph=True,
        )[0].reshape(B, -1)
        return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()

    def _real_tokens_for_discriminator(
        self,
        real_inputs: torch.Tensor,
        reconstruction_targets: Optional[torch.Tensor],
        generated_length: int,
    ) -> torch.Tensor:
        """Return real next-token sequences with the same length as generated data.

        Dataset inputs normally contain ``SOS + residues`` whereas targets contain
        ``residues + EOS``.  The discriminator should compare generated next-token
        sequences with the latter.  Explicit alignment is also required by the
        interpolation used for WGAN-GP.
        """
        source = reconstruction_targets if reconstruction_targets is not None else real_inputs
        if source.size(1) > generated_length:
            return source[:, :generated_length]
        if source.size(1) < generated_length:
            return F.pad(
                source,
                (0, generated_length - source.size(1)),
                value=getattr(self.G, 'pad_idx', 0),
            )
        return source

    def train_step(
        self,
        real_seqs: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        reconstruction_targets: Optional[torch.Tensor] = None,
        loss_scale: float = 1.0,
    ) -> Dict[str, float]:
        """
        Single training step: forward + backward only.
        Optimizer stepping is managed by _train_epoch for gradient accumulation.

        Args:
            real_seqs: Real sequences (batch, seq_len)
            conditions: Optional conditioning features (batch, cond_dim)
            loss_scale: Scale factor for loss (1/accum_steps for gradient accumulation)

        Returns:
            Dict with training metrics (unscaled values for logging)
        """
        batch_size = real_seqs.size(0)
        real_seqs = real_seqs.to(self.device)
        if conditions is not None:
            conditions = conditions.to(self.device)
        if reconstruction_targets is not None:
            reconstruction_targets = reconstruction_targets.to(self.device)

        metrics = {}

        # =================================================================
        # DISCRIMINATOR TRAINING
        # =================================================================
        d_real_mean, d_fake_mean = 0, 0
        d_loss_val = 0
        d_gap = 0

        # WGAN-GP needs double-backward through the critic, which is unstable
        # under fp16 AMP — run D/G in fp32 when using it.
        step_amp = self.use_amp and self.gan_loss != 'wgan_gp'

        for _ in range(self.d_steps):
            with torch.amp.autocast('cuda', enabled=step_amp):
                # Fake samples (no grad to G during D step)
                with torch.no_grad():
                    _, fake_logits_d = self._generate(batch_size, conditions)
                fake_in_d = self._relax(fake_logits_d)

                # Compare like with like: generated logits represent next tokens
                # (residues + EOS), not decoder inputs (SOS + residues).  Aligning
                # lengths here also prevents invalid WGAN-GP interpolation.
                real_tokens_d = self._real_tokens_for_discriminator(
                    real_seqs, reconstruction_targets, fake_in_d.size(1)
                )
                real_onehot = F.one_hot(real_tokens_d, num_classes=self.D.vocab_size).float()
                if self.noise_std > 0:
                    real_onehot = real_onehot + torch.randn_like(real_onehot) * self.noise_std

                d_real = self.D(real_onehot)
                d_fake = self.D(fake_in_d)

                if self.gan_loss == 'wgan_gp':
                    # Eq.(8-9): critic loss = E[D(fake)] - E[D(real)] + λ·GP
                    gp = self._gradient_penalty(real_onehot.detach(), fake_in_d.detach())
                    d_loss = (d_fake.mean() - d_real.mean() + self.lambda_gp * gp) * loss_scale
                else:
                    real_labels = torch.ones_like(d_real) * (1.0 - self.label_smooth)
                    fake_labels = torch.zeros_like(d_fake) + (self.label_smooth * 0.5)
                    loss_real = F.binary_cross_entropy_with_logits(d_real, real_labels)
                    loss_fake = F.binary_cross_entropy_with_logits(d_fake, fake_labels)
                    d_loss = (loss_real + loss_fake) * loss_scale

            d_real_mean = d_real.mean().item()
            d_fake_mean = d_fake.mean().item()
            d_gap = d_real_mean - d_fake_mean

            # Adaptive D-skip only for BCE (gap is not bounded for WGAN critic).
            if self.gan_loss != 'wgan_gp' and d_gap > self.d_gap_threshold:
                self.d_skip_count += 1
                metrics['d_skip'] = True
                break

            # Backward only (no optimizer step)
            self.scaler.scale(d_loss).backward()

        d_loss_val = (d_loss.item() / loss_scale) if isinstance(d_loss, torch.Tensor) else 0
        d_skipped = (self.gan_loss != 'wgan_gp') and (d_gap > self.d_gap_threshold)
        metrics.update({
            'd_loss': d_loss_val,
            'd_real': d_real_mean,
            'd_fake': d_fake_mean,
            'd_gap': d_gap,
            'd_skipped': d_skipped,  # flag for _train_epoch to skip opt_D step
        })

        # =================================================================
        # GENERATOR TRAINING
        # =================================================================
        # More G steps when D is strong
        actual_g_steps = self.g_steps
        if d_gap > self.g_steps_boost:
            actual_g_steps = self.g_steps * 2

        for _ in range(actual_g_steps):
            with torch.amp.autocast('cuda', enabled=step_amp):
                # Generate
                fake_seqs, fake_logits = self._generate(batch_size, conditions)

                # Relax logits to a D-input (gumbel straight-through or softmax)
                fake_probs = self._relax(fake_logits)

                d_fake = self.D(fake_probs)
                if self.gan_loss == 'wgan_gp':
                    loss_adv = -d_fake.mean()                 # maximise critic score
                else:
                    loss_adv = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))

                # Diversity loss
                div_results = self.diversity_loss(fake_logits)
                loss_div = div_results['total']

                # N-gram diversity loss
                loss_ngram = self.ngram_loss(fake_logits) if self.w_ngram > 0 else fake_logits.new_tensor(0.0)

                # Length penalty loss
                loss_length = self.length_penalty_loss(fake_logits) if self.w_length > 0 else fake_logits.new_tensor(0.0)

                # Stability bias loss — lazy vocab injection on first call
                if self.w_stability > 0:
                    if self.stability_loss.weight_matrix is None:
                        from ..data.vocabulary import VOCAB
                        self.stability_loss.target_ii = self.target_stability_ii
                        self.stability_loss._build_matrix(VOCAB)
                    loss_stability = self.stability_loss(fake_logits)
                else:
                    loss_stability = fake_logits.new_tensor(0.0)

                loss_reconstruction = (
                    self._reconstruction_loss(real_seqs, reconstruction_targets, conditions)
                    if self.w_rec > 0 else fake_logits.new_tensor(0.0)
                )

                # Total G loss (scaled for accumulation)
                g_loss = (
                    self.w_adv * loss_adv
                    + self.w_rec * loss_reconstruction
                    + self.w_div * loss_div
                    + self.w_ngram * loss_ngram
                    + self.w_length * loss_length
                    + self.w_stability * loss_stability
                ) * loss_scale

            # Backward only (no optimizer step)
            self.scaler.scale(g_loss).backward()

        metrics.update({
            'g_loss': g_loss.item() / loss_scale,
            'loss_adv': loss_adv.item(),
            'loss_div': loss_div.item(),
            'loss_ngram': loss_ngram.item(),
            'loss_length': loss_length.item(),
            'loss_stability': loss_stability.item(),
            'loss_reconstruction': loss_reconstruction.item(),
            'entropy': div_results['token_entropy_value'],
        })

        self.global_step += 1
        return metrics

    def _generate(
        self,
        batch_size: int,
        conditions: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Generate fake sequences with logits."""
        z = torch.randn(batch_size, self.G.latent_dim, device=self.device)

        if hasattr(self.G, 'generate_with_logits'):
            return self.G.generate_with_logits(z, conditions)
        else:
            # Pass condition as keyword to match forward(z, target=None, condition=None)
            output = self.G(z, condition=conditions)

            # Handle dict output from autoregressive forward()
            if isinstance(output, dict):
                logits = output.get('logits', output.get('tokens'))
                tokens = logits.argmax(dim=-1)
            else:
                logits = output
                tokens = logits.argmax(dim=-1)
            return tokens, logits

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 50,
        early_stopping_patience: int = 15,
        early_stopping_metric: str = 'val_g_loss',
        minimize_metric: bool = True,
    ):
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation loader
            epochs: Number of epochs
            checkpoint_dir: Directory for checkpoints
            log_interval: Log every N batches
            early_stopping_patience: Epochs without improvement before stopping
            early_stopping_metric: Metric to monitor for early stopping
            minimize_metric: If True, minimize metric; else maximize
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training on {self.device} for {epochs} epochs")
        logger.info(f"G steps: {self.g_steps}, D steps: {self.d_steps}")
        logger.info(f"Gradient accumulation steps: {self.accum_steps}")
        logger.info(f"Diversity weight: {self.w_div}, Adversarial weight: {self.w_adv}")
        logger.info(f"Early stopping: patience={early_stopping_patience}, metric={early_stopping_metric}")

        best_metric_value = float('inf') if minimize_metric else -float('inf')
        patience_counter = 0
        # ``history`` is initialized for a new trainer and restored by load().
        # Do not discard earlier epochs when genuinely resuming a GAN run.

        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            self._set_gumbel_temperature(epoch)
            epoch_start = time.time()

            # Train epoch
            metrics = self._train_epoch(train_loader, log_interval)

            # Validate
            val_metrics = None
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader)
                # Merge val_metrics into metrics dict
                metrics.update({f'val_{k}': v for k, v in val_metrics.items()})

            # Log
            elapsed = time.time() - epoch_start
            self._log_epoch(epoch, epochs, metrics, elapsed)

            # Save metrics to history
            self.history.append({'epoch': epoch + 1, 'gumbel_tau': self.gumbel_tau, **metrics})

            # LR Scheduler
            g_loss_val = metrics.get(
                'val_loss_reconstruction', metrics.get('val_g_loss', metrics.get('g_loss', 0))
            )
            if hasattr(self, 'scheduler_G'):
                self.scheduler_G.step(g_loss_val)

            # Early stopping
            current_metric = metrics.get(early_stopping_metric, None)
            if current_metric is not None:
                improved = (current_metric < best_metric_value) if minimize_metric else (current_metric > best_metric_value)

                if improved:
                    best_metric_value = current_metric
                    patience_counter = 0
                    logger.info(f"⭐ Improved {early_stopping_metric}: {current_metric:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter < early_stopping_patience:
                        logger.info(
                            f"No improvement for {patience_counter}/{early_stopping_patience} epochs. "
                            f"Best: {best_metric_value:.4f}"
                        )
                    else:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        self.save(checkpoint_dir / 'final_model.pt')
                        self._save_history(checkpoint_dir)
                        logger.info("Training complete!")
                        return

            # Save checkpoints
            self._save_checkpoints(epoch, epochs, metrics, checkpoint_dir)

        # Final save
        self.save(checkpoint_dir / 'final_model.pt')
        self._save_history(checkpoint_dir)
        logger.info("Training complete!")

    def _train_epoch(self, loader, log_interval: int) -> Dict:
        """Train single epoch with gradient accumulation support."""
        self.G.train()
        self.D.train()

        epoch_metrics = {
            'd_loss': 0, 'g_loss': 0,
            'd_real': 0, 'd_fake': 0,
            'entropy': 0,
            'loss_adv': 0, 'loss_div': 0,
            'loss_ngram': 0, 'loss_length': 0, 'loss_stability': 0,
            'loss_reconstruction': 0,
        }
        n_batches = 0
        accum_steps = self.accum_steps
        loss_scale = 1.0 / accum_steps

        logger.info(f"Epoch {self.epoch+1} Training with {len(loader)} batches (accum_steps={accum_steps})...")

        # Zero gradients at start of epoch
        self.opt_D.zero_grad()
        self.opt_G.zero_grad()

        # FIX: track d_skipped across the *entire* accumulation window, not per-batch.
        # If D was skipped for any batch in the window, we skip the D optimizer step.
        d_skipped_in_window = False

        for batch_idx, batch in enumerate(loader):
            real_seqs, conditions, reconstruction_targets = self._parse_batch(batch)
            metrics = self.train_step(
                real_seqs, conditions, reconstruction_targets, loss_scale=loss_scale
            )

            if metrics.get('d_skipped', False):
                d_skipped_in_window = True

            is_accum_boundary = (batch_idx + 1) % accum_steps == 0
            is_last_batch = (batch_idx + 1) == len(loader)

            if is_accum_boundary or is_last_batch:
                # Clip and step G
                self.scaler.unscale_(self.opt_G)
                torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)

                # Only step D if it was NOT skipped in this entire window
                if not d_skipped_in_window:
                    self.scaler.unscale_(self.opt_D)
                    torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
                    self.scaler.step(self.opt_D)

                self.scaler.step(self.opt_G)
                self.scaler.update()

                self.opt_D.zero_grad()
                self.opt_G.zero_grad()
                d_skipped_in_window = False   # reset for next window

            for k in epoch_metrics:
                if k in metrics:
                    epoch_metrics[k] += metrics[k]
            n_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                d_gap = metrics.get('d_real', 0) - metrics.get('d_fake', 0)
                logger.info(
                    f"  Batch {batch_idx+1}/{len(loader)} | "
                    f"D: {metrics['d_loss']:.4f} | "
                    f"G: {metrics['g_loss']:.4f} | "
                    f"d_real: {metrics.get('d_real', 0):.3f} | "
                    f"d_fake: {metrics.get('d_fake', 0):.3f} | "
                    f"gap: {d_gap:.3f} | "
                    f"ent: {metrics.get('entropy', 0):.3f}"
                )

        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_batches, 1)

        return epoch_metrics

    def _parse_batch(self, batch) -> tuple:
        """Parse batch into inputs, conditions, and next-token targets."""
        if isinstance(batch, dict):
            # DataLoader returns dict with 'input_ids' (tokens)
            seqs = batch.get('input_ids', batch.get('tokens', batch.get('sequence')))
            conds = batch.get('condition', batch.get('features', batch.get('conditions')))
            if not self.accept_conditions:
                conds = None
            targets = batch.get('target_ids')
        elif isinstance(batch, (list, tuple)):
            seqs = torch.tensor(batch[0]) if not isinstance(batch[0], torch.Tensor) else batch[0]
            conds = torch.tensor(batch[1]) if len(batch) > 1 and not isinstance(batch[1], torch.Tensor) else (batch[1] if len(batch) > 1 else None)
            targets = None
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")
        return seqs, conds, targets

    def _validate_epoch(self, loader) -> Dict:
        """Evaluate on validation set without gradient computation."""
        self.G.eval()
        self.D.eval()

        epoch_metrics = {
            'd_loss': 0, 'g_loss': 0,
            'd_real': 0, 'd_fake': 0,
            'entropy': 0,
            'loss_adv': 0, 'loss_div': 0,
            'loss_ngram': 0, 'loss_length': 0, 'loss_reconstruction': 0,
        }
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                real_seqs, conditions, reconstruction_targets = self._parse_batch(batch)
                real_seqs = real_seqs.to(self.device)
                if conditions is not None:
                    conditions = conditions.to(self.device)
                if reconstruction_targets is not None:
                    reconstruction_targets = reconstruction_targets.to(self.device)

                batch_size = real_seqs.size(0)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    # FIX: no instance noise during validation — noise biases D
                    # scores and makes val metrics not reflect real-world performance.
                    _, fake_logits = self._generate(batch_size, conditions)
                    fake_probs = F.softmax(fake_logits, dim=-1)

                    real_tokens_d = self._real_tokens_for_discriminator(
                        real_seqs, reconstruction_targets, fake_probs.size(1)
                    )
                    real_onehot = F.one_hot(
                        real_tokens_d, num_classes=self.D.vocab_size
                    ).float()
                    d_real = self.D(real_onehot)

                    d_fake = self.D(fake_probs)
                    if self.gan_loss == 'wgan_gp':
                        # Gradient penalty is a training regularizer; validation
                        # reports the held-out Wasserstein critic gap.
                        d_loss = d_fake.mean() - d_real.mean()
                        loss_adv = -d_fake.mean()
                    else:
                        real_labels = torch.ones_like(d_real) * (1.0 - self.label_smooth)
                        fake_labels = torch.zeros_like(d_fake) + (self.label_smooth * 0.5)
                        loss_real = F.binary_cross_entropy_with_logits(d_real, real_labels)
                        loss_fake = F.binary_cross_entropy_with_logits(d_fake, fake_labels)
                        d_loss = loss_real + loss_fake
                        loss_adv = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
                    div_results = self.diversity_loss(fake_logits)
                    loss_div = div_results['total']
                    loss_reconstruction = (
                        self._reconstruction_loss(
                            real_seqs, reconstruction_targets, conditions,
                            deterministic_latent=True,
                        ) if self.w_rec > 0 else fake_logits.new_tensor(0.0)
                    )
                    g_loss = (
                        self.w_adv * loss_adv + self.w_div * loss_div
                        + self.w_rec * loss_reconstruction
                    )

                epoch_metrics['d_loss'] += d_loss.item()
                epoch_metrics['g_loss'] += g_loss.item()
                epoch_metrics['d_real'] += d_real.mean().item()
                epoch_metrics['d_fake'] += d_fake.mean().item()
                epoch_metrics['entropy'] += div_results['token_entropy_value']
                epoch_metrics['loss_adv'] += loss_adv.item()
                epoch_metrics['loss_div'] += loss_div.item()
                epoch_metrics['loss_reconstruction'] += loss_reconstruction.item()
                n_batches += 1

        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_batches, 1)

        self.G.train()
        self.D.train()

        return epoch_metrics

    def _log_epoch(self, epoch: int, total: int, metrics: Dict, elapsed: float):
        """Log epoch results with comprehensive metrics."""
        d_gap = metrics['d_real'] - metrics['d_fake']

        log_msg = (
            f"Epoch {epoch+1:2d}/{total} [{elapsed:6.1f}s] | "
            f"D: {metrics['d_loss']:.4f} | "
            f"G: {metrics['g_loss']:.4f} | "
            f"d_real: {metrics['d_real']:.3f} | "
            f"d_fake: {metrics['d_fake']:.3f} | "
            f"gap: {d_gap:+.3f} | "
            f"ent: {metrics['entropy']:.3f} | "
            f"ngram: {metrics.get('loss_ngram', 0):.4f} | "
            f"len_pen: {metrics.get('loss_length', 0):.4f} | "
            f"stab: {metrics.get('loss_stability', 0):.4f} | "
            f"recon: {metrics.get('loss_reconstruction', 0):.4f} | "
            f"tau: {self.gumbel_tau:.3f}"
        )

        # Add validation metrics if available
        if 'val_g_loss' in metrics:
            val_d_gap = metrics['val_d_real'] - metrics['val_d_fake']
            log_msg += (
                f" | Val | "
                f"D: {metrics['val_d_loss']:.4f} | "
                f"G: {metrics['val_g_loss']:.4f} | "
                f"gap: {val_d_gap:+.3f}"
            )

        logger.info(log_msg)

        # Warnings
        if d_gap > 8.0:
            logger.warning(f"⚠️ Potential mode collapse! train d_gap={d_gap:.3f}")
        if metrics['entropy'] < 1.0:
            logger.warning(f"⚠️ Low train entropy={metrics['entropy']:.3f}, diversity may be low")
        if 'val_entropy' in metrics and metrics['val_entropy'] < 1.0:
            logger.warning(f"⚠️ Low val entropy={metrics['val_entropy']:.3f}")

    def _save_checkpoints(
        self,
        epoch: int,
        total: int,
        metrics: Dict,
        checkpoint_dir: Path,
    ):
        """Save checkpoints."""
        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            self.save(checkpoint_dir / f'epoch_{epoch+1}.pt')

        # Use deterministic teacher-forced validation CE when available.
        g_loss_key = (
            'val_loss_reconstruction' if 'val_loss_reconstruction' in metrics
            else ('val_g_loss' if 'val_g_loss' in metrics else 'g_loss')
        )
        g_loss_val = metrics[g_loss_key]

        if g_loss_val < self.best_loss:
            prev_best = self.best_loss
            self.best_loss = g_loss_val
            self.save(checkpoint_dir / 'best_model.pt')
            if prev_best != float('inf'):
                logger.info(
                    f"✓ Saved best model: {g_loss_key}={prev_best:.4f} → {g_loss_val:.4f}"
                )
            else:
                logger.info(f"✓ Saved best model: {g_loss_key}={g_loss_val:.4f}")

    def _save_history(self, checkpoint_dir: Path):
        """Save training history to JSON."""
        if self.history:
            history_file = checkpoint_dir / 'history.json'
            with open(history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
            logger.info(f"Saved history to {history_file}")

    # =========================================================================
    # CHECKPOINT MANAGEMENT
    # =========================================================================

    def save(self, path: str):
        """Save checkpoint with full model architecture info."""
        # Capture model architecture for reliable resume
        model_config = {}
        G = self.G
        for attr in ('vocab_size', 'embedding_dim', 'hidden_dim', 'latent_dim',
                      'max_length', 'num_layers', 'num_heads', 'dropout', 'condition_dim',
                      'mem_tokens', 'esm_dim', 'gat_heads', 'gat_window', 'use_gat',
                      'fusion_type',
                      'bidirectional', 'use_attention', 'pad_idx', 'sos_idx', 'eos_idx'):
            if hasattr(G, attr):
                model_config[attr] = getattr(G, attr)

        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'generator_class': type(self.G).__name__,
            'generator': self.G.state_dict(),
            'discriminator': self.D.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D': self.opt_D.state_dict(),
            'scaler': self.scaler.state_dict(),
            'scheduler_G': self.scheduler_G.state_dict() if hasattr(self, 'scheduler_G') else None,
            'config': self.config,
            'model_config': model_config,
            'data_metadata': getattr(self, 'data_metadata', None),
            'run_metadata': getattr(self, 'run_metadata', None),
            'artifact_reportable': getattr(self, 'artifact_reportable', False),
            'history': self.history,
            'best_loss': self.best_loss,
            'best_val_loss': self.best_val_loss,
        }, path)
        logger.info(f"Saved: {path}")

    def load(self, path: str, load_optimizer: bool = True) -> int:
        """Load checkpoint. Returns epoch number.
        
        Args:
            path: Path to checkpoint file
            load_optimizer: If False, load only model weights and skip
                            optimizer/scaler state (useful when optimizer
                            state is corrupted by NaN training runs).
        """
        ckpt = torch.load(path, map_location=self.device)

        # Handle both old and new checkpoint formats
        gen_key = 'generator' if 'generator' in ckpt else 'generator_state_dict'
        dis_key = 'discriminator' if 'discriminator' in ckpt else 'discriminator_state_dict'

        # A partially loaded generator invalidates a reportable experiment.
        # Architecture-specific ablations construct the matching model before
        # reaching this point, so strict loading is the safe default.
        try:
            self.G.load_state_dict(ckpt[gen_key], strict=True)
        except Exception as exc:
            raise RuntimeError(f"Generator checkpoint is incompatible with the configured model: {path}") from exc

        if dis_key not in ckpt:
            # Expected when Phase 2 starts from an MLE-only warm-up checkpoint.
            logger.info("Parent checkpoint has no discriminator; using fresh D weights")
            d_loaded_ok = False
        else:
            try:
                self.D.load_state_dict(ckpt[dis_key], strict=True)
                d_loaded_ok = True
            except Exception as exc:
                raise RuntimeError(
                    f"Discriminator checkpoint is incompatible with the configured model: {path}"
                ) from exc

        # Load optimizers separately: skip opt_D if D architecture changed
        # (old optimizer momentum buffers would have wrong shapes → RuntimeError at step)
        opt_g_key = 'opt_G' if 'opt_G' in ckpt else 'optimizer_G_state_dict'
        opt_d_key = 'opt_D' if 'opt_D' in ckpt else 'optimizer_D_state_dict'

        def _has_nan_in_optimizer(opt_state_dict: dict) -> bool:
            """Check if any Adam moment buffer contains NaN."""
            for param_state in opt_state_dict.get('state', {}).values():
                for v in param_state.values():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        return True
            return False

        if load_optimizer:
            # --- G optimizer ---
            g_opt_loaded = False
            try:
                if opt_g_key in ckpt:
                    if _has_nan_in_optimizer(ckpt[opt_g_key]):
                        logger.warning("G optimizer state contains NaN — resetting to fresh optimizer")
                    else:
                        self.opt_G.load_state_dict(ckpt[opt_g_key])
                        g_opt_loaded = True
                        logger.info("Loaded G optimizer state")
            except (KeyError, RuntimeError, ValueError) as e:
                logger.warning(f"Could not load G optimizer state: {e}. Using fresh G optimizer.")
            if not g_opt_loaded:
                self._init_optimizers()

            # --- D optimizer ---
            if d_loaded_ok:
                d_opt_loaded = False
                try:
                    if opt_d_key in ckpt:
                        if _has_nan_in_optimizer(ckpt[opt_d_key]):
                            logger.warning("D optimizer state contains NaN — resetting to fresh optimizer")
                        else:
                            self.opt_D.load_state_dict(ckpt[opt_d_key])
                            d_opt_loaded = True
                            logger.info("Loaded D optimizer state")
                except (KeyError, RuntimeError, ValueError) as e:
                    logger.warning(f"Could not load D optimizer state: {e}. Using fresh D optimizer.")
            else:
                logger.info("D architecture changed → using fresh D optimizer")

            # --- AMP scaler ---
            scaler_loaded = False
            if 'scaler' in ckpt and self.use_amp:
                try:
                    scaler_sd = ckpt['scaler']
                    scale_val = scaler_sd.get('scale', None)
                    # Reject if scale is NaN/Inf/zero/tiny — all indicate corrupted state
                    scale_ok = (
                        scale_val is not None
                        and not (isinstance(scale_val, float) and (
                            scale_val != scale_val  # NaN check
                            or scale_val == float('inf')
                            or scale_val < 1.0       # scale < 1 = too small, corrupted
                        ))
                    )
                    if scale_ok:
                        self.scaler.load_state_dict(scaler_sd)
                        scaler_loaded = True
                        logger.info(f"Loaded AMP scaler state (scale={scale_val})")
                    else:
                        logger.warning(f"AMP scaler state corrupted (scale={scale_val}) — resetting to fresh scaler")
                except Exception as e:
                    logger.warning(f"Could not load scaler state: {e}. Using fresh scaler.")
            if not scaler_loaded and self.use_amp:
                # Re-create a fresh scaler
                self.scaler = torch.amp.GradScaler('cuda', enabled=True)
                logger.info("Using fresh AMP scaler (scale=65536.0)")
            if ckpt.get('scheduler_G') and hasattr(self, 'scheduler_G'):
                try:
                    self.scheduler_G.load_state_dict(ckpt['scheduler_G'])
                    logger.info("Loaded G scheduler state")
                except (KeyError, RuntimeError, ValueError) as exc:
                    raise RuntimeError(f"Invalid scheduler state in {path}") from exc
        else:
            logger.info("Skipping optimizer/scaler state (--fresh-optimizer): using fresh optimizers and AMP scaler")
            self._init_optimizers()
            if self.use_amp:
                self.scaler = torch.amp.GradScaler('cuda', enabled=True)

        self.epoch = ckpt.get('epoch', 0)
        self.global_step = ckpt.get('global_step', 0)
        self.best_loss = ckpt.get('best_loss', float('inf'))
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.history = list(ckpt.get('history') or [])
        logger.info(f"Loaded model from epoch {self.epoch}")
        return self.epoch


class ConditionalGANTrainer(GANTrainer):
    """
    GAN Trainer with feature conditioning support.

    Additional config options:
        - condition_dim: Conditioning vector dimension
        - feature_loss_weight: Weight for feature prediction loss
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        condition_dim: int = 8,
    ):
        super().__init__(generator, discriminator, config, device)
        self.accept_conditions = True
        self.condition_dim = condition_dim
        self.w_feature = config.get('feature_loss_weight', 0.1)

    def _generate(
        self,
        batch_size: int,
        conditions: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Generate with conditions."""
        z = torch.randn(batch_size, self.G.latent_dim, device=self.device)

        # Sample random conditions if not provided
        if conditions is None and self.condition_dim > 0:
            conditions = torch.randn(batch_size, self.condition_dim, device=self.device)

        if hasattr(self.G, 'generate_with_logits'):
            return self.G.generate_with_logits(z, conditions)
        else:
            # Pass condition as keyword argument to match forward(z, target=None, condition=None)
            output = self.G(z, condition=conditions)

            # Handle dict output from forward()
            if isinstance(output, dict):
                logits = output.get('logits', output.get('tokens'))
            else:
                logits = output

            tokens = logits.argmax(dim=-1)
            return tokens, logits
