"""
Loss functions for GAN training.

New losses (2025-02 fix):
    NgramDiversityLoss  - penalizes bigram/trigram repetition (motif collapse)
    LengthPenaltyLoss   - EOS supervision to fix length collapse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DiversityLoss(nn.Module):
    """
    Diversity loss to prevent mode collapse.
    
    Components:
        - Token entropy: Encourage diverse token choices per position
        - Batch diversity: Different samples should be different
        - Pairwise distance: Maximize distance between generated samples
    """
    
    def __init__(
        self,
        entropy_weight: float = 0.3,
        batch_sim_weight: float = 0.3,
        pairwise_weight: float = 0.4,
    ):
        super().__init__()
        self.w_entropy = entropy_weight
        self.w_batch = batch_sim_weight
        self.w_pairwise = pairwise_weight
        
    def forward(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute diversity losses.
        
        Args:
            logits: (batch, seq_len, vocab_size) - Generator output logits
            
        Returns:
            dict with 'total', 'entropy', 'batch_sim', 'pairwise'
        """
        probs = F.softmax(logits, dim=-1)
        batch_size, seq_len, vocab_size = probs.shape
        device = probs.device
        
        # 1. Token entropy - encourage high entropy (diverse token choices)
        token_entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float, device=device))
        entropy_loss = 1.0 - (token_entropy / max_entropy)  # Lower = more diverse
        
        # 2 & 3. Compute similarity matrix once and reuse
        flat = probs.view(batch_size, -1)  # (batch, seq*vocab)
        flat_norm = F.normalize(flat, dim=-1)
        similarity = torch.mm(flat_norm, flat_norm.t())  # (batch, batch)
        mask = 1.0 - torch.eye(batch_size, device=device)
        
        # Batch similarity (for logging only — not used in total to avoid conflict)
        batch_sim = (similarity * mask).sum() / (mask.sum() + 1e-8)
        
        # Pairwise distance loss: reuse similarity matrix (1 - cosine sim = cosine dist)
        dist_matrix = 1.0 - similarity  # Reuse, don't recompute torch.mm
        pairwise_dist = (dist_matrix * mask).sum() / (mask.sum() + 1e-8)
        pairwise_loss = 1.0 / (pairwise_dist + 1.0)  # Invert: smaller loss = larger distance
        
        # Total: entropy + pairwise only (removed batch_sim — conflicts with pairwise gradient)
        total = (
            self.w_entropy * entropy_loss +
            self.w_pairwise * pairwise_loss
        )
        
        return {
            'total': total,
            'entropy': entropy_loss,
            'batch_sim': batch_sim,   # kept for logging/monitoring
            'pairwise': pairwise_loss,
            'token_entropy_value': token_entropy.item(),
        }



class NgramDiversityLoss(nn.Module):
    """
    N-gram diversity loss — penalizes repetitive motifs.

    Uses soft probabilities from logits to form bigram and trigram
    joint distributions, then penalizes concentration via entropy.
    This is differentiable and does NOT require token sampling.

    Args:
        bigram_weight:  Weight for bigram penalty (default 0.5)
        trigram_weight: Weight for trigram penalty (default 0.5)
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
        # Disable autocast for this entire block — fp16 AMP causes NaN in
        # einsum + entropy computations with concentrated distributions.
        # We cast to fp32 explicitly and compute everything in full precision.
        with torch.amp.autocast('cuda', enabled=False):
            logits_f = torch.nan_to_num(
                logits.detach().float(), nan=0.0, posinf=80.0, neginf=-80.0
            )
            probs = F.softmax(logits_f, dim=-1)      # (B, L, V) fp32

            # Re-attach gradient path through original logits so the loss
            # is still differentiable w.r.t. generator parameters.
            # We do this by computing probs separately with grad:
            probs_grad = F.softmax(
                torch.nan_to_num(logits.float(), nan=0.0, posinf=80.0, neginf=-80.0),
                dim=-1,
            )

            B, L, V = probs_grad.shape
            eps = 1e-10
            # Explicit fp32 accumulator — never inherit logits dtype (may be fp16)
            total = torch.zeros((), device=logits.device, dtype=torch.float32)

            def _safe_normalized_entropy(joint: torch.Tensor, max_log: float) -> torch.Tensor:
                """Compute 1 - H(joint)/H_max (concentration penalty), fp32-safe."""
                joint = joint / (joint.sum() + eps)
                entropy = -(torch.xlogy(joint, joint.clamp(min=eps))).sum()
                return (1.0 - (entropy / (max_log + eps))).clamp(0.0, 1.0)

            # Pre-compute max_log in Python to avoid fp16 tensor creation
            import math
            max_log_bi  = math.log(float(V * V))
            max_log_tri = math.log(float(V ** 3))

            # -- Bigrams --
            if L >= 2 and self.w_bi > 0:
                p1 = probs_grad[:, :-1, :]      # (B, L-1, V)
                p2 = probs_grad[:, 1:, :]        # (B, L-1, V)
                bigram_joint = torch.einsum('nla,nlb->ab', p1, p2)  # (V, V)
                total = total + self.w_bi * _safe_normalized_entropy(bigram_joint, max_log_bi)

            # -- Trigrams --
            if L >= 3 and self.w_tri > 0:
                p1 = probs_grad[:, :-2, :]       # (B, L-2, V)
                p2 = probs_grad[:, 1:-1, :]      # (B, L-2, V)
                p3 = probs_grad[:, 2:, :]        # (B, L-2, V)
                bi_part   = torch.einsum('nla,nlb->ab', p1, p2)       # (V, V)
                tri_joint = torch.einsum('ab,nlc->abc', bi_part, p3)  # (V, V, V)
                total = total + self.w_tri * _safe_normalized_entropy(tri_joint, max_log_tri)

        # Final guard — clamp and replace any residual NaN with 0 (no penalty)
        return torch.nan_to_num(total, nan=0.0, posinf=1.0).clamp(0.0, 1.0)



class LengthPenaltyLoss(nn.Module):
    """
    EOS supervision loss using cumulative probability.

    Instead of penalizing expected EOS position (which is always near L/2
    for uniform EOS probs and thus never triggers), this loss directly
    supervises the *cumulative* EOS probability at two checkpoints:

        1. early_penalty: P(EOS by target_min) should be ~0
           (generator must NOT stop before target_min)
        2. late_penalty: P(EOS by target_max) should be ~1
           (generator MUST stop before target_max)

    This gives an always-active gradient signal that shapes the EOS
    placement distribution, not just its mean.

    Args:
        eos_idx:    Token index of <EOS>
        target_min: Minimum desired sequence length (default 10)
        target_max: Maximum desired sequence length (default 30)
        early_weight: Weight for early-stop penalty (default 1.0)
        late_weight:  Weight for late-stop penalty (default 1.0)
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
        probs = F.softmax(logits_f, dim=-1)  # (B, L, V)
        B, L, V = probs.shape

        # EOS probability at each position: (B, L)
        eos_probs = probs[:, :, self.eos_idx]

        # Cumulative EOS probability up to each position: (B, L)
        # cum_eos[:, t] = probability that EOS has been generated by position t
        cum_eos = eos_probs.cumsum(dim=1).clamp(0.0, 1.0)

        loss = logits.new_tensor(0.0).float()

        # 1. Early penalty: cumulative EOS at target_min should be low (~0)
        #    → penalize premature stopping before target_min
        if self.early_weight > 0 and self.target_min > 0:
            idx = min(self.target_min - 1, L - 1)
            early_penalty = cum_eos[:, idx].mean()  # want near 0
            loss = loss + self.early_weight * early_penalty

        # 2. Late penalty: cumulative EOS at target_max should be high (~1)
        #    → penalize runaway sequences that don't stop by target_max
        if self.late_weight > 0 and self.target_max <= L:
            idx = min(self.target_max - 1, L - 1)
            late_penalty = (1.0 - cum_eos[:, idx]).mean()  # want near 0
            loss = loss + self.late_weight * late_penalty

        return loss


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss - match intermediate features between real and fake.
    Helps stabilize GAN training.
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
            real_features: Features from discriminator on real data
            fake_features: Features from discriminator on generated data
            
        Returns:
            L2 distance between feature means
        """
        return F.mse_loss(
            fake_features.mean(dim=0),
            real_features.mean(dim=0).detach()
        )


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for autoencoder-style training.
    """
    
    def __init__(self, ignore_index: int = 0, label_smoothing: float = 0.0):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, vocab_size)
            targets: (batch, seq_len)
            
        Returns:
            Cross-entropy loss
        """
        batch_size, seq_len, vocab_size = logits.shape
        return self.loss_fn(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )


class StabilityBiasLoss(nn.Module):
    """
    Differentiable stability bias loss.

    Penalizes dipeptide combinations with high expected instability index
    using soft token probabilities — no sampling, gradients flow cleanly.

    Formula:
        expected_II ≈ (10 / L) * Σ_{t} Σ_{a,b} p_t(a) * p_{t+1}(b) * W[a,b]
        loss = mean(ReLU(expected_II - target_ii)) / max_weight

    The ReLU ensures the loss is 0 when sequences are already stable
    (expected II ≤ target_ii), preventing mode collapse from unbounded
    minimization of a loss that has no natural lower bound.

    Args:
        vocab:      Vocabulary object (must have idx_to_aa and vocab_size)
        target_ii:  Target instability index to stay below (default: 30.0)
        max_weight: Normalization constant (default: 58.28 = max weight in table)
    """

    def __init__(self, vocab=None, target_ii: float = 30.0, max_weight: float = 58.28):
        super().__init__()
        self.target_ii = target_ii
        self.max_weight = max_weight
        # Pre-register as buffer (None) so .to(device) works before _build_matrix
        self.register_buffer('weight_matrix', None)
        if vocab is not None:
            self._build_matrix(vocab)

    def _build_matrix(self, vocab):
        """Build (V, V) instability weight matrix and store as buffer."""
        from ..constants import INSTABILITY_WEIGHTS
        V = vocab.vocab_size if hasattr(vocab, 'vocab_size') else 24
        W = torch.ones(V, V)  # default weight 1.0
        idx_to_aa = vocab.idx_to_aa  # {int: str}
        for i, aa_i in idx_to_aa.items():
            for j, aa_j in idx_to_aa.items():
                dipeptide = aa_i + aa_j
                if dipeptide in INSTABILITY_WEIGHTS:
                    W[i, j] = INSTABILITY_WEIGHTS[dipeptide]
        # Direct assignment works because weight_matrix is already registered as a buffer
        self.weight_matrix = W

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, vocab_size)

        Returns:
            Scalar loss >= 0. Zero when expected II <= target_ii for all samples.
            Positive only when generator produces sequences with high instability.
        """
        if self.weight_matrix is None:
            return logits.new_tensor(0.0)

        logits_f = torch.nan_to_num(logits.float(), nan=0.0, posinf=80.0, neginf=-80.0)
        probs = F.softmax(logits_f, dim=-1)  # (B, L, V)
        B, L, V = probs.shape
        if L < 2:
            return logits.new_tensor(0.0)

        W = self.weight_matrix.to(probs.device)  # (V, V)

        # Expected dipeptide weight at adjacent positions:
        # E[W(t, t+1)] = Σ_{a,b} p_t(a) * p_{t+1}(b) * W[a,b]
        left = probs[:, :-1, :]   # (B, L-1, V)
        right = probs[:, 1:, :]   # (B, L-1, V)
        # (B, L-1, V) @ (V, V) -> (B, L-1, V), then * right -> sum over V => (B, L-1)
        expected_weights = (left @ W * right).sum(dim=-1)  # (B, L-1)

        # Approximate instability index per sample
        ii_approx = (10.0 / L) * expected_weights.sum(dim=-1)  # (B,)

        # ReLU: only penalize when expected II exceeds the target threshold.
        # This bounds the loss at 0 from below, preventing generator from
        # collapsing to a few 'super-stable' dipeptide patterns.
        penalty = F.relu(ii_approx - self.target_ii)  # (B,) >= 0

        return (penalty / self.max_weight).mean()


class WassersteinLoss(nn.Module):

    """
    Wasserstein GAN loss - more stable than standard GAN loss.
    """
    
    def __init__(self, clip_value: float = 0.01):
        super().__init__()
        self.clip_value = clip_value
        
    def discriminator_loss(
        self,
        d_real: torch.Tensor,
        d_fake: torch.Tensor,
    ) -> torch.Tensor:
        """D loss: maximize D(real) - D(fake)"""
        return d_fake.mean() - d_real.mean()
        
    def generator_loss(self, d_fake: torch.Tensor) -> torch.Tensor:
        """G loss: maximize D(fake)"""
        return -d_fake.mean()
        
    def clip_weights(self, discriminator: nn.Module):
        """Clip discriminator weights for Lipschitz constraint."""
        for p in discriminator.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)


class GradientPenalty(nn.Module):
    """
    Gradient penalty for WGAN-GP.
    """
    
    def __init__(self, lambda_gp: float = 10.0):
        super().__init__()
        self.lambda_gp = lambda_gp
        
    def forward(
        self,
        discriminator: nn.Module,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gradient penalty.
        
        Args:
            discriminator: D model
            real_data: Real samples
            fake_data: Generated samples
            conditions: Optional conditions
            
        Returns:
            Gradient penalty term
        """
        batch_size = real_data.size(0)
        device = real_data.device
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, device=device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data.detach()
        interpolated.requires_grad_(True)
        
        # Get discriminator output
        d_interpolated = discriminator(interpolated, conditions)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Compute penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = self.lambda_gp * ((gradient_norm - 1) ** 2).mean()
        
        return penalty
