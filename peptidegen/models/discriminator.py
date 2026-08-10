"""
Discriminator models for GAN-based peptide generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import math

from .components import SelfAttention, ResidualBlock


class PeptideDiscriminator(nn.Module):
    """
    Base class for peptide discriminators.
    """
    
    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        max_length: int = 50,
        dropout: float = 0.2,
        use_spectral_norm: bool = True,
        pad_idx: int = 0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.dropout = dropout
        self.use_spectral_norm = use_spectral_norm
        self.pad_idx = pad_idx
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input sequence (batch_size, seq_len) or (batch_size, seq_len, vocab_size) for soft
            mask: Optional attention mask
            
        Returns:
            Discriminator scores (batch_size, 1)
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def get_feature(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract features from discriminator (for feature matching loss)."""
        raise NotImplementedError("Subclasses must implement get_feature()")


class CNNDiscriminator(PeptideDiscriminator):
    """
    CNN-based discriminator with multiple kernel sizes.
    Inspired by TextCNN architecture.
    """
    
    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        max_length: int = 50,
        num_filters: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.2,
        use_spectral_norm: bool = True,
        use_minibatch_std: bool = True,
        pad_idx: int = 0,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            max_length=max_length,
            dropout=dropout,
            use_spectral_norm=use_spectral_norm,
            pad_idx=pad_idx,
        )
        
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.use_minibatch_std = use_minibatch_std
        
        # Apply spectral normalization if requested
        norm_fn = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        # Convolutional layers for each kernel size
        self.convs = nn.ModuleList()
        for k_size, n_filter in zip(kernel_sizes, num_filters):
            conv = norm_fn(nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=n_filter,
                kernel_size=k_size,
                padding=k_size // 2
            ))
            self.convs.append(conv)
            
        # Total features from all conv layers
        total_filters = sum(num_filters)
        
        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            norm_fn(nn.Linear(total_filters, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
        
        # Output layer (if minibatch_std is true, features get +1 dimension)
        out_features = hidden_dim + 1 if self.use_minibatch_std else hidden_dim
        self.output_layer = norm_fn(nn.Linear(out_features, 1))
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        is_onehot: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input (batch_size, seq_len) indices or (batch_size, seq_len, vocab_size) soft
            mask: Optional attention mask
            is_onehot: If True, treat 3D input as one-hot/noisy one-hot
        """
        # Handle soft inputs (from generator)
        if x.dim() == 3:
            # Soft input: (batch, seq_len, vocab_size) -> weighted embedding
            embedded = torch.matmul(x, self.embedding.weight)
        else:
            # Hard input: (batch, seq_len) -> embedding lookup
            embedded = self.embedding(x)
            
        # (batch, seq_len, embedding_dim) -> (batch, embedding_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.leaky_relu(conv(embedded), 0.2)
            pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)
            conv_outputs.append(pooled)
            
        # Concatenate all conv outputs
        features = torch.cat(conv_outputs, dim=-1)
        
        # Feature extraction
        features = self.feature_layers(features)
        
        # Minibatch Standard Deviation trick (helps prevent mode collapse)
        if self.use_minibatch_std:
            # Add small epsilon to avoid NaN in std
            batch_std = torch.sqrt(features.var(dim=0, keepdim=True, unbiased=False) + 1e-8)
            mean_std = batch_std.mean().expand(features.size(0), 1)
            features = torch.cat([features, mean_std], dim=-1)
            
        # Output
        output = self.output_layer(features)
        
        return output
    
    def get_feature(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract intermediate features."""
        if x.dim() == 3:
            embedded = torch.matmul(x, self.embedding.weight)
        else:
            embedded = self.embedding(x)
            
        embedded = embedded.permute(0, 2, 1)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.leaky_relu(conv(embedded), 0.2)
            pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)
            conv_outputs.append(pooled)
            
        features = torch.cat(conv_outputs, dim=-1)
        return self.feature_layers(features)


class RNNDiscriminator(PeptideDiscriminator):
    """
    RNN-based discriminator using bidirectional GRU.
    """
    
    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        max_length: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_spectral_norm: bool = True,
        use_attention: bool = True,
        pad_idx: int = 0,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            max_length=max_length,
            dropout=dropout,
            use_spectral_norm=use_spectral_norm,
            pad_idx=pad_idx,
        )
        
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Bidirectional GRU
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        
        rnn_output_dim = hidden_dim * 2  # Bidirectional
        
        # Attention layer
        if use_attention:
            self.attention = SelfAttention(rnn_output_dim)
            
        # Apply spectral normalization
        norm_fn = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        # Feature layers
        self.feature_layers = nn.Sequential(
            norm_fn(nn.Linear(rnn_output_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
        
        # Output layer
        self.output_layer = norm_fn(nn.Linear(hidden_dim, 1))
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Handle soft inputs
        if x.dim() == 3:
            embedded = torch.matmul(x, self.embedding.weight)
        else:
            embedded = self.embedding(x)
            
        # RNN forward
        output, hidden = self.rnn(embedded)
        
        # Apply attention or use last hidden state
        if self.use_attention:
            output = self.attention(output, mask)
            # Global average pooling
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                output = (output * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
            else:
                output = output.mean(dim=1)
        else:
            # Use concatenated final hidden states
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
            output = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=-1)
            
        # Feature extraction
        features = self.feature_layers(output)
        
        # Output
        return self.output_layer(features)
    
    def get_feature(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.dim() == 3:
            embedded = torch.matmul(x, self.embedding.weight)
        else:
            embedded = self.embedding(x)
            
        output, _ = self.rnn(embedded)
        
        if self.use_attention:
            output = self.attention(output, mask)
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                output = (output * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
            else:
                output = output.mean(dim=1)
        else:
            output = output[:, -1, :]
            
        return self.feature_layers(output)


class HybridDiscriminator(PeptideDiscriminator):
    """
    Hybrid discriminator combining CNN and RNN features.
    """
    
    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        max_length: int = 50,
        num_filters: List[int] = [64, 128],
        kernel_sizes: List[int] = [3, 5],
        num_rnn_layers: int = 1,
        dropout: float = 0.2,
        use_spectral_norm: bool = True,
        pad_idx: int = 0,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            max_length=max_length,
            dropout=dropout,
            use_spectral_norm=use_spectral_norm,
            pad_idx=pad_idx,
        )
        
        norm_fn = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        # CNN branch
        self.convs = nn.ModuleList()
        for k_size, n_filter in zip(kernel_sizes, num_filters):
            conv = norm_fn(nn.Conv1d(
                embedding_dim, n_filter, k_size, padding=k_size // 2
            ))
            self.convs.append(conv)
            
        cnn_output_dim = sum(num_filters)
        
        # RNN branch
        self.rnn = nn.GRU(
            embedding_dim, hidden_dim // 2,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True,
        )
        rnn_output_dim = hidden_dim
        
        # Combine features
        total_dim = cnn_output_dim + rnn_output_dim
        
        self.feature_layers = nn.Sequential(
            norm_fn(nn.Linear(total_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            norm_fn(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
        )
        
        self.output_layer = norm_fn(nn.Linear(hidden_dim // 2, 1))
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.dim() == 3:
            embedded = torch.matmul(x, self.embedding.weight)
        else:
            embedded = self.embedding(x)
            
        # CNN branch
        embedded_cnn = embedded.permute(0, 2, 1)
        cnn_outputs = []
        for conv in self.convs:
            conv_out = F.leaky_relu(conv(embedded_cnn), 0.2)
            pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)
            cnn_outputs.append(pooled)
        cnn_features = torch.cat(cnn_outputs, dim=-1)
        
        # RNN branch
        rnn_output, hidden = self.rnn(embedded)
        rnn_features = rnn_output.mean(dim=1)
        
        # Combine
        combined = torch.cat([cnn_features, rnn_features], dim=-1)
        features = self.feature_layers(combined)
        
        return self.output_layer(features)
    
    def get_feature(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.dim() == 3:
            embedded = torch.matmul(x, self.embedding.weight)
        else:
            embedded = self.embedding(x)
            
        embedded_cnn = embedded.permute(0, 2, 1)
        cnn_outputs = []
        for conv in self.convs:
            conv_out = F.leaky_relu(conv(embedded_cnn), 0.2)
            pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)
            cnn_outputs.append(pooled)
        cnn_features = torch.cat(cnn_outputs, dim=-1)
        
        rnn_output, _ = self.rnn(embedded)
        rnn_features = rnn_output.mean(dim=1)
        
        combined = torch.cat([cnn_features, rnn_features], dim=-1)
        return self.feature_layers(combined)
