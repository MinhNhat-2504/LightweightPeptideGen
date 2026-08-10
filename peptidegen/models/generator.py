"""
Generator models for peptide sequence generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
from torch.utils import checkpoint

from .components import PositionalEncoding, MultiHeadAttention, SelfAttention


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
        total_input_dim = latent_dim
        if condition_dim is not None:
            total_input_dim += condition_dim
            
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
        
        # Adjust for bidirectional
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention layer
        if use_attention:
            self.attention = SelfAttention(gru_output_dim)
            
        # Output projection
        self.output_projection = nn.Linear(gru_output_dim, vocab_size)
        
        # Initialize hidden state projection
        total_input_dim = latent_dim
        if condition_dim is not None:
            total_input_dim += condition_dim
            
        num_directions = 2 if bidirectional else 1
        self.init_hidden = nn.Linear(
            total_input_dim, 
            num_layers * num_directions * hidden_dim
        )
        
    def _get_initial_hidden(
        self, 
        z: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get initial hidden state from latent vector."""
        batch_size = z.size(0)
        
        # Concatenate condition if provided
        if condition is not None:
            z = torch.cat([z, condition], dim=-1)
            
        # Project to hidden dimensions
        h = self.init_hidden(z)
        
        # Reshape to (num_layers * num_directions, batch_size, hidden_dim)
        num_directions = 2 if self.bidirectional else 1
        h = h.view(batch_size, self.num_layers * num_directions, self.hidden_dim)
        h = h.permute(1, 0, 2).contiguous()
        
        return h
    
    def forward(
        self,
        z: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with teacher forcing (if target provided).
        """
        batch_size = z.size(0)
        device = z.device
        
        # Get initial hidden state
        hidden = self._get_initial_hidden(z, condition)
        
        if target is not None:
            # Teacher forcing mode
            embedded = self.embedding(target)
            
            # Use gradient checkpointing to save memory
            if self.use_gradient_checkpointing and self.training:
                output, hidden = checkpoint.checkpoint(
                    self._gru_forward,
                    embedded,
                    hidden,
                    use_reentrant=False
                )
            else:
                output, hidden = self.gru(embedded, hidden)
            
            if self.use_attention:
                output = self.attention(output)
                
            logits = self.output_projection(output)
            
            return {
                'logits': logits,
                'hidden': hidden,
            }
        else:
            # Autoregressive generation
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
        """Generate sequences autoregressively."""
        # Start with SOS token
        current_token = torch.full(
            (batch_size, 1), self.sos_idx, dtype=torch.long, device=device
        )
        
        sequences = [current_token]
        all_logits = []
        
        for _ in range(self.max_length):
            embedded = self.embedding(current_token)
            output, hidden = self.gru(embedded, hidden)
            
            # Apply attention if enabled (was missing before — mismatch with training)
            if self.use_attention:
                output = self.attention(output)
            
            logits = self.output_projection(output[:, -1:, :])
            all_logits.append(logits)
            
            # Sample next token
            probs = F.softmax(logits.squeeze(1) / temperature, dim=-1)
            current_token = torch.multinomial(probs, num_samples=1)
            sequences.append(current_token)
            
        sequences = torch.cat(sequences, dim=1)
        all_logits = torch.cat(all_logits, dim=1)
        
        return {
            'sequences': sequences,
            'logits': all_logits,
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
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate sequences with various sampling strategies."""
        if device is None:
            device = next(self.parameters()).device
            
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, device=device)
            
        if max_length is None:
            max_length = self.max_length
            
        # Get initial hidden state
        hidden = self._get_initial_hidden(z, condition)
        
        # Start with SOS token
        current_token = torch.full(
            (batch_size, 1), self.sos_idx, dtype=torch.long, device=device
        )
        
        sequences = [current_token]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_length):
            if finished.all():
                break
                
            embedded = self.embedding(current_token)
            output, hidden = self.gru(embedded, hidden)
            
            logits = self.output_projection(output[:, -1, :]) / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
                
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')
                
            # Sample
            probs = F.softmax(logits, dim=-1)
            current_token = torch.multinomial(probs, num_samples=1)
            
            # Check for EOS
            finished = finished | (current_token.squeeze(-1) == self.eos_idx)
            
            sequences.append(current_token)
            
        return torch.cat(sequences, dim=1)


class LSTMGenerator(PeptideGenerator):
    """
    LSTM-based generator.
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
        pad_idx: int = 0,
        sos_idx: int = 1,
        eos_idx: int = 2,
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
        
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Initialize hidden state projection
        total_input_dim = latent_dim
        if condition_dim is not None:
            total_input_dim += condition_dim
            
        self.init_hidden = nn.Linear(total_input_dim, num_layers * hidden_dim)
        self.init_cell = nn.Linear(total_input_dim, num_layers * hidden_dim)
        
    def _get_initial_state(
        self, 
        z: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial hidden and cell states."""
        batch_size = z.size(0)
        
        if condition is not None:
            z = torch.cat([z, condition], dim=-1)
            
        h = self.init_hidden(z)
        c = self.init_cell(z)
        
        h = h.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c = c.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        
        return h, c
    
    def forward(
        self,
        z: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        batch_size = z.size(0)
        device = z.device
        
        h, c = self._get_initial_state(z, condition)
        
        if target is not None:
            embedded = self.embedding(target)
            output, (h, c) = self.lstm(embedded, (h, c))
            logits = self.output_projection(output)
            
            return {'logits': logits, 'hidden': (h, c)}
        else:
            return self._generate_autoregressive(batch_size, h, c, temperature, device)
    
    def _generate_autoregressive(
        self,
        batch_size: int,
        h: torch.Tensor,
        c: torch.Tensor,
        temperature: float,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        current_token = torch.full(
            (batch_size, 1), self.sos_idx, dtype=torch.long, device=device
        )
        
        sequences = [current_token]
        all_logits = []
        
        for _ in range(self.max_length):
            embedded = self.embedding(current_token)
            output, (h, c) = self.lstm(embedded, (h, c))
            
            logits = self.output_projection(output[:, -1:, :])
            all_logits.append(logits)
            
            probs = F.softmax(logits.squeeze(1) / temperature, dim=-1)
            current_token = torch.multinomial(probs, num_samples=1)
            sequences.append(current_token)
            
        sequences = torch.cat(sequences, dim=1)
        all_logits = torch.cat(all_logits, dim=1)
        
        return {'sequences': sequences, 'logits': all_logits, 'hidden': (h, c)}
    
    def generate(
        self,
        batch_size: int = 1,
        z: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
            
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, device=device)
            
        if max_length is None:
            max_length = self.max_length
            
        h, c = self._get_initial_state(z, condition)
        
        current_token = torch.full(
            (batch_size, 1), self.sos_idx, dtype=torch.long, device=device
        )
        
        sequences = [current_token]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length):
            if finished.all():
                break
                
            embedded = self.embedding(current_token)
            output, (h, c) = self.lstm(embedded, (h, c))
            
            logits = self.output_projection(output[:, -1, :]) / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')

            # Top-p (nucleus) filtering - consistent with GRUGenerator
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            current_token = torch.multinomial(probs, num_samples=1)
            
            finished = finished | (current_token.squeeze(-1) == self.eos_idx)
            sequences.append(current_token)
            
        return torch.cat(sequences, dim=1)


class TransformerGenerator(PeptideGenerator):
    """
    Lightweight Transformer-based generator.
    """
    
    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        max_length: int = 50,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        condition_dim: Optional[int] = None,
        pad_idx: int = 0,
        sos_idx: int = 1,
        eos_idx: int = 2,
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
        
        self.num_heads = num_heads
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_length + 2, dropout)
        
        # Latent projection
        total_input_dim = latent_dim
        if condition_dim is not None:
            total_input_dim += condition_dim
            
        self.latent_projection = nn.Linear(total_input_dim, embedding_dim)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        z: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        batch_size = z.size(0)
        device = z.device
        
        # Combine latent with condition
        if condition is not None:
            z = torch.cat([z, condition], dim=-1)
            
        # Project latent to memory
        memory = self.latent_projection(z).unsqueeze(1)  # (batch, 1, embedding_dim)
        
        if target is not None:
            # Teacher forcing
            seq_len = target.size(1)
            tgt_mask = self._generate_square_subsequent_mask(seq_len, device)
            
            embedded = self.embedding(target)
            embedded = self.pos_encoding(embedded)
            
            output = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask)
            logits = self.output_projection(output)
            
            return {'logits': logits}
        else:
            return self._generate_autoregressive(batch_size, memory, temperature, device)
    
    def _generate_autoregressive(
        self,
        batch_size: int,
        memory: torch.Tensor,
        temperature: float,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        current_sequence = torch.full(
            (batch_size, 1), self.sos_idx, dtype=torch.long, device=device
        )
        
        all_logits = []
        
        for step in range(self.max_length):
            tgt_mask = self._generate_square_subsequent_mask(
                current_sequence.size(1), device
            )
            
            embedded = self.embedding(current_sequence)
            embedded = self.pos_encoding(embedded)
            
            output = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask)
            logits = self.output_projection(output[:, -1:, :])
            all_logits.append(logits)
            
            probs = F.softmax(logits.squeeze(1) / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            current_sequence = torch.cat([current_sequence, next_token], dim=1)
            
        all_logits = torch.cat(all_logits, dim=1)
        
        return {'sequences': current_sequence, 'logits': all_logits}
    
    def generate(
        self,
        batch_size: int = 1,
        z: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
            
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, device=device)
            
        if max_length is None:
            max_length = self.max_length
            
        if condition is not None:
            z = torch.cat([z, condition], dim=-1)
            
        memory = self.latent_projection(z).unsqueeze(1)
        
        current_sequence = torch.full(
            (batch_size, 1), self.sos_idx, dtype=torch.long, device=device
        )
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length):
            if finished.all():
                break
                
            tgt_mask = self._generate_square_subsequent_mask(
                current_sequence.size(1), device
            )
            
            embedded = self.embedding(current_sequence)
            embedded = self.pos_encoding(embedded)
            
            output = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask)
            logits = self.output_projection(output[:, -1, :]) / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            finished = finished | (next_token.squeeze(-1) == self.eos_idx)
            current_sequence = torch.cat([current_sequence, next_token], dim=1)
            
        return current_sequence
