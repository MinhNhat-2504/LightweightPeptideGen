"""
Generator models for peptide sequence generation.

Classes:
    PeptideGenerator  – abstract base class
    GRUGenerator      – production GRU-based generator (used by GANTrainer)

Note: LSTMGenerator and TransformerGenerator have been removed (dead code).
      The production architecture is MultimodalFusionGenerator (fusion_generator.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
from torch.utils import checkpoint

from .components import SelfAttention


class PeptideGenerator(nn.Module):
    """
    Base class for peptide generators.
    """

    def __init__(
        self,
        vocab_size: int = 24,  # 20 AA + 4 special tokens
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        max_length: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        condition_dim: Optional[int] = None,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.num_layers = num_layers
        self.dropout = dropout
        self.condition_dim = condition_dim
        self.pad_idx = pad_idx

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Latent to hidden projection
        total_input_dim = latent_dim + (condition_dim or 0)
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        z: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate peptide sequences.

        Args:
            z: Latent vector (batch_size, latent_dim)
            target: Target sequence for teacher forcing (batch_size, seq_len)
            condition: Optional condition vector (batch_size, condition_dim)
            temperature: Sampling temperature

        Returns:
            Dictionary with 'logits', 'sequences', etc.
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def generate(
        self,
        batch_size: int = 1,
        z: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        min_length: int = 5,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate sequences autoregressively.

        Args:
            batch_size: Number of sequences to generate
            z: Optional latent vector
            condition: Optional condition vector
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling (0 to disable)
            top_p: Nucleus sampling threshold
            device: Device to use

        Returns:
            Generated sequences (batch_size, seq_len)
        """
        raise NotImplementedError("Subclasses must implement generate()")


class GRUGenerator(PeptideGenerator):
    """
    GRU-based generator for lightweight peptide generation.
    Used as the legacy/ablation architecture; the default production
    architecture is MultimodalFusionGenerator (fusion_generator.py).
    """

    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        max_length: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        condition_dim: Optional[int] = None,
        bidirectional: bool = False,
        use_attention: bool = True,
        pad_idx: int = 0,
        sos_idx: int = 1,
        eos_idx: int = 2,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_length=max_length,
            num_layers=num_layers,
            dropout=dropout,
            condition_dim=condition_dim,
            pad_idx=pad_idx,
        )

        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # GRU layers
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Attention layer
        if use_attention:
            self.attention = SelfAttention(gru_output_dim)

        # Output projection (overrides base class — correct dim after attention)
        self.output_projection = nn.Linear(gru_output_dim, vocab_size)

        # Initial hidden state projection
        total_input_dim = latent_dim + (condition_dim or 0)
        num_directions = 2 if bidirectional else 1
        self.init_hidden = nn.Linear(
            total_input_dim,
            num_layers * num_directions * hidden_dim
        )

    def _get_initial_hidden(
        self,
        z: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get initial hidden state from latent vector."""
        batch_size = z.size(0)
        if condition is not None:
            z = torch.cat([z, condition], dim=-1)
        h = self.init_hidden(z)
        num_directions = 2 if self.bidirectional else 1
        h = h.view(batch_size, self.num_layers * num_directions, self.hidden_dim)
        return h.permute(1, 0, 2).contiguous()

    def _apply_attention(self, output: torch.Tensor) -> torch.Tensor:
        """Apply attention if enabled — single helper used by all paths."""
        if self.use_attention:
            output = self.attention(output)
        return output

    def forward(
        self,
        z: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with teacher forcing (if target provided)."""
        batch_size = z.size(0)
        device = z.device

        hidden = self._get_initial_hidden(z, condition)

        if target is not None:
            # Teacher forcing mode
            embedded = self.embedding(target)

            if self.use_gradient_checkpointing and self.training:
                output, hidden = checkpoint.checkpoint(
                    self._gru_forward, embedded, hidden, use_reentrant=False
                )
            else:
                output, hidden = self.gru(embedded, hidden)

            output = self._apply_attention(output)
            logits = self.output_projection(output)
            return {'logits': logits, 'hidden': hidden}
        else:
            return self._generate_autoregressive(
                batch_size, z, condition, hidden, temperature, device
            )

    def _gru_forward(self, embedded: torch.Tensor, hidden: torch.Tensor) -> Tuple:
        """Helper for gradient checkpointing."""
        return self.gru(embedded, hidden)

    def _generate_autoregressive(
        self,
        batch_size: int,
        z: torch.Tensor,
        condition: Optional[torch.Tensor],
        hidden: torch.Tensor,
        temperature: float,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Generate sequences autoregressively (training GAN path, no top-k/p)."""
        current_token = torch.full(
            (batch_size, 1), self.sos_idx, dtype=torch.long, device=device
        )

        sequences = [current_token]
        all_logits = []

        for _ in range(self.max_length):
            embedded = self.embedding(current_token)
            output, hidden = self.gru(embedded, hidden)
            output = self._apply_attention(output)          # FIX: was missing before

            logits = self.output_projection(output[:, -1:, :])
            all_logits.append(logits)

            probs = F.softmax(logits.squeeze(1) / temperature, dim=-1)
            current_token = torch.multinomial(probs, num_samples=1)
            sequences.append(current_token)

        return {
            'sequences': torch.cat(sequences, dim=1),
            'logits': torch.cat(all_logits, dim=1),
            'hidden': hidden,
        }

    def generate(
        self,
        batch_size: int = 1,
        z: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        min_length: int = 5,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate sequences with top-k / nucleus sampling (inference path).
        FIX: attention is now correctly applied here, matching the training path.
        """
        if device is None:
            device = next(self.parameters()).device
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, device=device)
        if max_length is None:
            max_length = self.max_length

        hidden = self._get_initial_hidden(z, condition)

        current_token = torch.full(
            (batch_size, 1), self.sos_idx, dtype=torch.long, device=device
        )
        sequences = [current_token]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        for step in range(max_length):
            if finished.all():
                break

            embedded = self.embedding(current_token)
            output, hidden = self.gru(embedded, hidden)
            output = self._apply_attention(output)          # FIX: was missing

            logits = self.output_projection(output[:, -1, :]).clone()
            # Only canonical residues and EOS are legal next tokens.  EOS is
            # disabled until the requested minimum peptide length is reached.
            for idx in {self.pad_idx, self.sos_idx, 3}:  # 3 = <UNK>
                if 0 <= idx < logits.size(-1):
                    logits[:, idx] = -float('Inf')
            if step < min_length:
                logits[:, self.eos_idx] = -float('Inf')
            if finished.any():
                logits[finished] = -float('Inf')
                logits[finished, self.pad_idx] = 0.0

            if temperature == 0:
                current_token = logits.argmax(dim=-1, keepdim=True)
                finished = finished | (current_token.squeeze(-1) == self.eos_idx)
                sequences.append(current_token)
                continue
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                kth_val = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                logits[logits < kth_val] = -float('Inf')

            # Nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative prob above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right so the first token above threshold is kept
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            current_token = torch.multinomial(probs, num_samples=1)

            finished = finished | (current_token.squeeze(-1) == self.eos_idx)
            sequences.append(current_token)

        return torch.cat(sequences, dim=1)
