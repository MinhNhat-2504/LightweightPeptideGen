"""
Loss functions for GAN training.

Classes:
    DiversityLoss        – entropy + pairwise distance to prevent mode collapse
    NgramDiversityLoss   – bigram/trigram concentration penalty
    LengthPenaltyLoss    – cumulative-EOS supervision for length control
    FeatureMatchingLoss  – L2 feature-mean matching for training stability
    ReconstructionLoss   – cross-entropy reconstruction
    StabilityBiasLoss    – differentiable instability-index penalty

Removed: WassersteinLoss, GradientPenalty (dead code; WGAN-GP logic lives
in GANTrainer._gradient_penalty / train_step).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DiversityLoss(nn.Module):
    """
    Diversity loss to prevent mode collapse.

    Components:
        - Token entropy: encourage diverse token choices per position
        - Pairwise distance: maximise cosine distance between samples
    """

    def __init__(
        self,
        entropy_weight: float = 0.3,
        batch_sim_weight: float = 0.3,   # kept for API compat; not used in total
        pairwise_weight: float = 0.4,
    ):
        super().__init__()
        self.w_entropy = entropy_weight
        self.w_batch = batch_sim_weight
        self.w_pairwise = pairwise_weight

    def forward(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: (batch, seq_len, vocab_size) generator output

        Returns:
            dict with 'total', 'entropy', 'batch_sim', 'pairwise',
                      'token_entropy_value'
        """
        probs = F.softmax(logits, dim=-1)
        batch_size, seq_len, vocab_size = probs.shape
        device = probs.device

        # 1. Token entropy — encourage high entropy (diverse token choices)
        token_entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float, device=device))
        entropy_loss = 1.0 - (token_entropy / max_entropy)   # lower = more diverse

        # 2 & 3. Compute similarity matrix once and reuse
        flat = probs.view(batch_size, -1)
        flat_norm = F.normalize(flat, dim=-1)
        similarity = torch.mm(flat_norm, flat_norm.t())       # (B, B)
        mask = 1.0 - torch.eye(batch_size, device=device)

        batch_sim = (similarity * mask).sum() / (mask.sum() + 1e-8)  # for logging

        dist_matrix = 1.0 - similarity
        pairwise_dist = (dist_matrix * mask).sum() / (mask.sum() + 1e-8)
        pairwise_loss = 1.0 / (pairwise_dist + 1.0)          # smaller = farther apart

        # batch_sim intentionally excluded from total (conflicts with pairwise grad)
        total = self.w_entropy * entropy_loss + self.w_pairwise * pairwise_loss

        return {
            'total': total,
            'entropy': entropy_loss,
            'batch_sim': batch_sim,
            'pairwise': pairwise_loss,
            'token_entropy_value': token_entropy.item(),
        }


class NgramDiversityLoss(nn.Module):
    """
    N-gram diversity loss — penalises repetitive motifs.

    Uses soft probabilities from logits to form bigram and trigram joint
    distributions, then penalises concentration via entropy.
    Differentiable — no token sampling required.

    Args:
        bigram_weight:  weight for bigram penalty (default 0.5)
        trigram_weight: weight for trigram penalty (default 0.5)
    """

    def __init__(self, bigram_weight: float = 0.5, trigram_weight: float = 0.5):
        super().__init__()
        self.w_bi = bigram_weight
        self.w_tri = trigram_weight

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, vocab_size)

        Returns:
            Scalar loss in [0, 1] — lower means more diverse n-grams.
        """
        # Run in fp32 to avoid NaN in einsum + entropy with fp16 AMP.
        with torch.amp.autocast('cuda', enabled=False):
            logits_f = torch.nan_to_num(
                logits.float(), nan=0.0, posinf=80.0, neginf=-80.0
            )
            probs_grad = F.softmax(logits_f, dim=-1)

            B, L, V = probs_grad.shape
            eps = 1e-10
            total = torch.zeros((), device=logits.device, dtype=torch.float32)
            max_log_bi = math.log(float(V * V))
            max_log_tri = math.log(float(V ** 3))

            def _concentration(joint: torch.Tensor, max_log: float) -> torch.Tensor:
                """1 - H(joint)/H_max  — high value means more concentrated."""
                joint = joint / (joint.sum() + eps)
                entropy = -(torch.xlogy(joint, joint.clamp(min=eps))).sum()
                return (1.0 - (entropy / (max_log + eps))).clamp(0.0, 1.0)

            # Bigrams
            if L >= 2 and self.w_bi > 0:
                p1 = probs_grad[:, :-1, :]           # (B, L-1, V)
                p2 = probs_grad[:, 1:, :]            # (B, L-1, V)
                bigram_joint = torch.einsum('nla,nlb->ab', p1, p2)   # (V, V)
                total = total + self.w_bi * _concentration(bigram_joint, max_log_bi)

            # Trigrams
            if L >= 3 and self.w_tri > 0:
                p1 = probs_grad[:, :-2, :]
                p2 = probs_grad[:, 1:-1, :]
                p3 = probs_grad[:, 2:, :]
                bi_part = torch.einsum('nla,nlb->ab', p1, p2)          # (V, V)
                tri_joint = torch.einsum('ab,nlc->abc', bi_part, p3)   # (V, V, V)
                total = total + self.w_tri * _concentration(tri_joint, max_log_tri)

        return torch.nan_to_num(total, nan=0.0, posinf=1.0).clamp(0.0, 1.0)


class LengthPenaltyLoss(nn.Module):
    """
    EOS supervision via *cumulative EOS probability*.

    The cumulative probability P(EOS has fired by position t) is approximated
    correctly as the survival-complement product:

        cum_eos[t] = 1 - ∏_{i=0}^{t} (1 - eos_prob[i])

    Two penalty terms:
        1. early_penalty: cum_eos at target_min should be ≈ 0
           (generator must NOT stop before target_min)
        2. late_penalty:  cum_eos at target_max should be ≈ 1
           (generator MUST stop before target_max)

    FIX vs. previous version: cumsum was used instead of the correct survival
    product, which could exceed 1.0 and be clamped, destroying the gradient.

    Args:
        eos_idx:    Token index of <EOS>
        target_min: Minimum desired sequence length (default 10)
        target_max: Maximum desired sequence length (default 30)
        early_weight: weight for early-stop penalty (default 1.0)
        late_weight:  weight for late-stop penalty (default 1.0)
    """

    def __init__(
        self,
        eos_idx: int = 2,
        target_min: int = 10,
        target_max: int = 30,
        early_weight: float = 1.0,
        late_weight: float = 1.0,
    ):
        super().__init__()
        self.eos_idx = eos_idx
        self.target_min = target_min
        self.target_max = target_max
        self.early_weight = early_weight
        self.late_weight = late_weight

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, vocab_size)

        Returns:
            Scalar length penalty loss >= 0.
        """
        logits_f = torch.nan_to_num(logits.float(), nan=0.0, posinf=80.0, neginf=-80.0)
        probs = F.softmax(logits_f, dim=-1)         # (B, L, V)
        B, L, V = probs.shape

        # EOS probability at each position: (B, L)
        eos_probs = probs[:, :, self.eos_idx]

        # FIX: correct cumulative EOS probability via survival product
        # cum_eos[t] = 1 - prod_{i<=t}(1 - eos_prob[i])
        # which is equivalent to: survival[:, t] = prod_{i<=t}(1 - eos_prob[i])
        survival = torch.cumprod(1.0 - eos_probs.clamp(0.0, 1.0 - 1e-7), dim=1)
        cum_eos = 1.0 - survival       # (B, L) — strictly in [0, 1]

        loss = logits.new_tensor(0.0).float()

        # 1. Early penalty: cum_eos at target_min should be low (~0)
        if self.early_weight > 0 and self.target_min > 0:
            idx = min(self.target_min - 1, L - 1)
            loss = loss + self.early_weight * cum_eos[:, idx].mean()

        # 2. Late penalty: cum_eos at target_max should be high (~1)
        if self.late_weight > 0 and self.target_max <= L:
            idx = min(self.target_max - 1, L - 1)
            loss = loss + self.late_weight * (1.0 - cum_eos[:, idx]).mean()

        return loss


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss — match intermediate discriminator features
    between real and fake samples. Helps stabilise GAN training.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        real_features: torch.Tensor,
        fake_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            real_features: features from discriminator on real data
            fake_features: features from discriminator on generated data

        Returns:
            L2 distance between feature means
        """
        return F.mse_loss(
            fake_features.mean(dim=0),
            real_features.mean(dim=0).detach(),
        )


class ReconstructionLoss(nn.Module):
    """Cross-entropy reconstruction loss for autoencoder-style training."""

    def __init__(self, ignore_index: int = 0, label_smoothing: float = 0.0):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (batch, seq_len, vocab_size)
            targets: (batch, seq_len)
        """
        batch_size, seq_len, vocab_size = logits.shape
        return self.loss_fn(logits.view(-1, vocab_size), targets.view(-1))


class StabilityBiasLoss(nn.Module):
    """
    Differentiable stability bias loss.

    Penalises dipeptide combinations with high expected instability index
    using soft token probabilities — no sampling, gradients flow cleanly.

    Formula:
        expected_II ≈ (10 / L) * Σ_t Σ_{a,b} p_t(a) * p_{t+1}(b) * W[a,b]
        loss = mean(ReLU(expected_II - target_ii)) / max_weight

    ReLU ensures loss is 0 when sequences are already stable, preventing
    mode collapse from unbounded minimisation.

    Args:
        vocab:      Vocabulary object (must have idx_to_aa and vocab_size)
        target_ii:  Target instability index to stay below (default: 30.0)
        max_weight: Normalisation constant (default: 58.28 = max weight in table)
    """

    def __init__(self, vocab=None, target_ii: float = 30.0, max_weight: float = 58.28):
        super().__init__()
        self.target_ii = target_ii
        self.max_weight = max_weight
        self.register_buffer('weight_matrix', None)
        if vocab is not None:
            self._build_matrix(vocab)

    def _build_matrix(self, vocab) -> None:
        """Build (V, V) instability weight matrix and store as buffer."""
        from ..constants import INSTABILITY_WEIGHTS
        V = vocab.vocab_size if hasattr(vocab, 'vocab_size') else 24
        W = torch.ones(V, V)
        for i, aa_i in vocab.idx_to_aa.items():
            for j, aa_j in vocab.idx_to_aa.items():
                dipeptide = aa_i + aa_j
                if dipeptide in INSTABILITY_WEIGHTS:
                    W[i, j] = INSTABILITY_WEIGHTS[dipeptide]
        self.weight_matrix = W

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, vocab_size)

        Returns:
            Scalar loss >= 0. Zero when expected II <= target_ii for all samples.
        """
        if self.weight_matrix is None:
            return logits.new_tensor(0.0)

        logits_f = torch.nan_to_num(logits.float(), nan=0.0, posinf=80.0, neginf=-80.0)
        probs = F.softmax(logits_f, dim=-1)         # (B, L, V)
        B, L, V = probs.shape
        if L < 2:
            return logits.new_tensor(0.0)

        W = self.weight_matrix.to(probs.device)     # (V, V)

        # Expected dipeptide weight at adjacent positions
        left = probs[:, :-1, :]     # (B, L-1, V)
        right = probs[:, 1:, :]     # (B, L-1, V)
        expected_weights = (left @ W * right).sum(dim=-1)   # (B, L-1)

        ii_approx = (10.0 / L) * expected_weights.sum(dim=-1)  # (B,)

        penalty = F.relu(ii_approx - self.target_ii)            # (B,) >= 0
        return (penalty / self.max_weight).mean()
