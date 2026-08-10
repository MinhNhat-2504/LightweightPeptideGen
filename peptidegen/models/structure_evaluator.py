"""
Structure Evaluator for assessing peptide structural stability
Uses lightweight Graph Attention Network (GAT) for structure prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
import math


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention Layer.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Learnable parameters
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)
        
    def forward(
        self, 
        h: torch.Tensor, 
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            h: Node features (batch, num_nodes, in_features)
            adj: Adjacency matrix (batch, num_nodes, num_nodes)
            
        Returns:
            Updated node features (batch, num_nodes, out_features)
        """
        batch_size, num_nodes, _ = h.size()
        
        # Linear transformation
        Wh = self.W(h)  # (batch, num_nodes, out_features)
        
        # Compute attention coefficients
        # Concatenate features of node pairs
        Wh_repeat_i = Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1)
        Wh_repeat_j = Wh.unsqueeze(1).repeat(1, num_nodes, 1, 1)
        concat = torch.cat([Wh_repeat_i, Wh_repeat_j], dim=-1)
        
        e = self.leakyrelu(self.a(concat).squeeze(-1))
        
        # Mask non-adjacent pairs
        zero_vec = torch.full_like(e, -1e9)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)
        
        # Apply attention to features
        h_prime = torch.bmm(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class LightweightGAT(nn.Module):
    """
    Lightweight Graph Attention Network for peptide structure evaluation.
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList()
        
        # First layer
        self.attention_layers.append(nn.ModuleList([
            GraphAttentionLayer(input_dim, hidden_dim, dropout, concat=True)
            for _ in range(num_heads)
        ]))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.attention_layers.append(nn.ModuleList([
                GraphAttentionLayer(hidden_dim * num_heads, hidden_dim, dropout, concat=True)
                for _ in range(num_heads)
            ]))
            
        # Output layer (single head)
        if num_layers > 1:
            self.attention_layers.append(nn.ModuleList([
                GraphAttentionLayer(hidden_dim * num_heads, output_dim, dropout, concat=False)
            ]))
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (batch, num_nodes, input_dim)
            adj: Adjacency matrix (batch, num_nodes, num_nodes)
            
        Returns:
            Node embeddings (batch, num_nodes, output_dim)
        """
        h = x
        
        for i, layer_heads in enumerate(self.attention_layers[:-1]):
            # Multi-head attention
            h_cat = torch.cat([
                head(h, adj) for head in layer_heads
            ], dim=-1)
            h = self.dropout(h_cat)
            
        # Output layer (single head or average of heads)
        output_heads = self.attention_layers[-1]
        if len(output_heads) == 1:
            h = output_heads[0](h, adj)
        else:
            h = torch.mean(torch.stack([
                head(h, adj) for head in output_heads
            ]), dim=0)
            
        return h


class StructureEvaluator(nn.Module):
    """
    Evaluates structural stability of peptide sequences.
    Combines sequence features with graph-based structural analysis.
    """
    
    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        gat_hidden: int = 64,
        gat_heads: int = 4,
        num_gat_layers: int = 2,
        dropout: float = 0.2,
        max_length: int = 50,
        pad_idx: int = 0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.pad_idx = pad_idx
        
        # Amino acid embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Position embedding
        self.position_embedding = nn.Embedding(max_length + 2, embedding_dim)
        
        # GAT for structural analysis
        self.gat = LightweightGAT(
            input_dim=embedding_dim,
            hidden_dim=gat_hidden,
            output_dim=gat_hidden,
            num_heads=gat_heads,
            num_layers=num_gat_layers,
            dropout=dropout,
        )
        
        # Sequence encoder (lightweight BiGRU)
        self.seq_encoder = nn.GRU(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        # Combine GAT and sequence features
        combined_dim = gat_hidden + hidden_dim
        
        # Stability prediction head
        self.stability_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Secondary structure prediction (optional)
        self.ss_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # Helix, Sheet, Coil
        )
        
    def _build_adjacency_matrix(
        self, 
        seq_length: int, 
        batch_size: int, 
        device: torch.device,
        window_size: int = 3,
    ) -> torch.Tensor:
        """
        Build adjacency matrix for sequence graph.
        Connects residues within a window (local structure).
        
        Args:
            seq_length: Sequence length
            batch_size: Batch size
            device: Device
            window_size: Window size for local connections
            
        Returns:
            Adjacency matrix (batch, seq_length, seq_length)
        """
        adj = torch.zeros(seq_length, seq_length, device=device)
        
        for i in range(seq_length):
            for j in range(max(0, i - window_size), min(seq_length, i + window_size + 1)):
                adj[i, j] = 1.0
                
        # Add self-loops
        adj = adj + torch.eye(seq_length, device=device)
        
        # Normalize
        degree = adj.sum(dim=-1, keepdim=True)
        adj = adj / degree.clamp(min=1)
        
        # Expand for batch
        adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        
        return adj
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input sequence (batch, seq_len) or soft (batch, seq_len, vocab_size)
            mask: Attention mask
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with 'stability_score', 'ss_pred', etc.
        """
        # Handle soft inputs
        if x.dim() == 3:
            embedded = torch.matmul(x, self.embedding.weight)
        else:
            embedded = self.embedding(x)
            
        batch_size, seq_len, _ = embedded.size()
        device = embedded.device
        
        # Add position embedding
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embedded = self.position_embedding(positions)
        embedded = embedded + pos_embedded
        
        # Build adjacency matrix
        adj = self._build_adjacency_matrix(seq_len, batch_size, device)
        
        # GAT forward
        gat_output = self.gat(embedded, adj)
        
        # Global pooling for GAT output
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            gat_pooled = (gat_output * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
        else:
            gat_pooled = gat_output.mean(dim=1)
            
        # Sequence encoding
        seq_output, hidden = self.seq_encoder(embedded)
        seq_pooled = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        
        # Combine features
        combined = torch.cat([gat_pooled, seq_pooled], dim=-1)
        
        # Predictions
        stability_score = self.stability_head(combined)
        ss_pred = self.ss_head(combined)
        
        result = {
            'stability_score': stability_score.squeeze(-1),
            'ss_pred': ss_pred,
        }
        
        if return_features:
            result['gat_features'] = gat_output
            result['seq_features'] = seq_output
            result['combined_features'] = combined
            
        return result
    
    def compute_stability_loss(
        self,
        x: torch.Tensor,
        target_stability: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute stability loss for training.
        
        If target_stability is not provided, uses heuristic-based pseudo labels.
        """
        outputs = self.forward(x, mask)
        pred_stability = outputs['stability_score']
        
        if target_stability is not None:
            loss = F.binary_cross_entropy(pred_stability, target_stability)
        else:
            # Encourage high stability scores
            loss = -pred_stability.mean()
            
        return loss


class StabilityLoss(nn.Module):
    """
    Combined loss function that includes structural stability penalties.
    """
    
    def __init__(
        self,
        structure_evaluator: StructureEvaluator,
        stability_weight: float = 0.3,
        diversity_weight: float = 0.1,
    ):
        super().__init__()
        
        self.structure_evaluator = structure_evaluator
        self.stability_weight = stability_weight
        self.diversity_weight = diversity_weight
        
    def forward(
        self,
        generated_logits: torch.Tensor,
        generated_sequences: torch.Tensor,
        real_sequences: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined stability loss.
        
        Args:
            generated_logits: Logits from generator (batch, seq_len, vocab_size)
            generated_sequences: Generated sequences
            real_sequences: Optional real sequences for comparison
            
        Returns:
            Dictionary with loss components
        """
        # Get stability scores
        stability_output = self.structure_evaluator(
            F.softmax(generated_logits, dim=-1)
        )
        stability_score = stability_output['stability_score']
        
        # Stability loss (maximize stability)
        stability_loss = -stability_score.mean()
        
        # Diversity loss (encourage diverse generations)
        if generated_logits.size(0) > 1:
            # Pairwise distance between generated sequences
            logits_flat = generated_logits.view(generated_logits.size(0), -1)
            distances = torch.cdist(logits_flat, logits_flat)
            diversity_loss = -distances.mean()
        else:
            diversity_loss = torch.tensor(0.0, device=generated_logits.device)
            
        # Total loss
        total_loss = (
            self.stability_weight * stability_loss +
            self.diversity_weight * diversity_loss
        )
        
        return {
            'total_loss': total_loss,
            'stability_loss': stability_loss,
            'diversity_loss': diversity_loss,
            'stability_score': stability_score.mean(),
        }
