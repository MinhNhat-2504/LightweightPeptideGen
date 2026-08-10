"""
Common model components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(context)


class ConditionalBatchNorm(nn.Module):
    """
    Conditional Batch Normalization.
    Modulates batch norm parameters based on condition.
    """
    
    def __init__(self, num_features: int, condition_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.gamma = nn.Linear(condition_dim, num_features)
        self.beta = nn.Linear(condition_dim, num_features)
        
        # Initialize to identity transformation
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_features, ...)
            condition: Condition tensor of shape (batch_size, condition_dim)
        """
        # Batch norm
        out = self.bn(x)
        
        # Get modulation parameters
        gamma = self.gamma(condition).unsqueeze(-1)  # (batch, num_features, 1)
        beta = self.beta(condition).unsqueeze(-1)
        
        return gamma * out + beta


class ResidualBlock(nn.Module):
    """
    Residual block with optional conditional normalization.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        condition_dim: Optional[int] = None,
        use_spectral_norm: bool = False
    ):
        super().__init__()
        
        conv_fn = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        self.conv1 = conv_fn(nn.Conv1d(in_channels, out_channels, 3, padding=1))
        self.conv2 = conv_fn(nn.Conv1d(out_channels, out_channels, 3, padding=1))
        
        if condition_dim is not None:
            self.bn1 = ConditionalBatchNorm(out_channels, condition_dim)
            self.bn2 = ConditionalBatchNorm(out_channels, condition_dim)
        else:
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)
            
        self.condition_dim = condition_dim
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = conv_fn(nn.Conv1d(in_channels, out_channels, 1))
        else:
            self.shortcut = nn.Identity()
            
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        if self.condition_dim is not None and condition is not None:
            out = self.bn1(out, condition)
        else:
            out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        if self.condition_dim is not None and condition is not None:
            out = self.bn2(out, condition)
        else:
            out = self.bn2(out)
            
        return self.activation(out + residual)


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit for improved gradient flow.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class Highway(nn.Module):
    """
    Highway network layer.
    """
    
    def __init__(self, input_dim: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        
        self.transforms = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for transform, gate in zip(self.transforms, self.gates):
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x
        return x


class SelfAttention(nn.Module):
    """
    Simple self-attention layer.
    """
    
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim // 2
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.scale = math.sqrt(hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            mask: (batch_size, seq_len) - 1 for valid, 0 for padding
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V)
