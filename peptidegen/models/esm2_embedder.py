"""
ESM2 Embedder for peptide sequence representation
Integrates Facebook's ESM2 model for extracting rich protein embeddings
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ESM2Embedder(nn.Module):
    """
    ESM2 Embedder wrapper for extracting protein sequence embeddings.
    
    Supports multiple ESM2 model variants:
    - esm2_t6_8M_UR50D: 6 layers, 8M params (fastest, lightweight)
    - esm2_t12_35M_UR50D: 12 layers, 35M params
    - esm2_t30_150M_UR50D: 30 layers, 150M params
    - esm2_t33_650M_UR50D: 33 layers, 650M params (good balance)
    - esm2_t36_3B_UR50D: 36 layers, 3B params (most accurate)
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        'esm2_t6_8M_UR50D': {'layers': 6, 'embed_dim': 320},
        'esm2_t12_35M_UR50D': {'layers': 12, 'embed_dim': 480},
        'esm2_t30_150M_UR50D': {'layers': 30, 'embed_dim': 640},
        'esm2_t33_650M_UR50D': {'layers': 33, 'embed_dim': 1280},
        'esm2_t36_3B_UR50D': {'layers': 36, 'embed_dim': 2560},
    }
    
    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: Optional[torch.device] = None,
        repr_layer: int = -1,  # -1 for last layer
        freeze: bool = True,
        pooling: str = "mean",  # mean, cls, max
    ):
        """
        Initialize ESM2 Embedder.
        
        Args:
            model_name: Name of ESM2 model variant
            device: Device to use
            repr_layer: Which layer to extract representations from (-1 for last)
            freeze: Whether to freeze ESM2 weights
            pooling: Pooling strategy for sequence representation
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.repr_layer = repr_layer
        self.pooling = pooling
        
        # Get model config
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODEL_CONFIGS.keys())}")
        
        self.config = self.MODEL_CONFIGS[model_name]
        self.embed_dim = self.config['embed_dim']
        self.num_layers = self.config['layers']
        
        # Determine actual repr layer
        if repr_layer == -1:
            self.actual_repr_layer = self.num_layers
        else:
            self.actual_repr_layer = repr_layer
            
        # Load model
        self.model, self.alphabet = self._load_model()
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Freeze if requested
        if freeze:
            self._freeze_model()
            
        self.model.eval()
        
        logger.info(f"Loaded ESM2 model: {model_name} (embed_dim={self.embed_dim})")
        
    def _load_model(self):
        """Load ESM2 model and alphabet."""
        try:
            import esm
            model, alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
            return model, alphabet
        except Exception as e:
            logger.error(f"Failed to load ESM2 model: {e}")
            raise
            
    def _freeze_model(self):
        """Freeze all ESM2 parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
            
    def _unfreeze_model(self):
        """Unfreeze all ESM2 parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
            
    def _pool_representations(
        self, 
        representations: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool sequence representations.
        
        Args:
            representations: (batch, seq_len, embed_dim)
            tokens: (batch, seq_len) - for identifying special tokens
            
        Returns:
            Pooled representations (batch, embed_dim)
        """
        if self.pooling == "cls":
            # Use CLS token representation (first token)
            return representations[:, 0, :]
            
        elif self.pooling == "mean":
            # Mean pooling over non-padding tokens
            # Exclude BOS and EOS tokens
            mask = (tokens != self.alphabet.padding_idx)
            mask = mask & (tokens != self.alphabet.cls_idx)
            mask = mask & (tokens != self.alphabet.eos_idx)
            mask = mask.unsqueeze(-1).float()
            
            pooled = (representations * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return pooled
            
        elif self.pooling == "max":
            # Max pooling
            mask = (tokens != self.alphabet.padding_idx).unsqueeze(-1)
            representations = representations.masked_fill(~mask, float('-inf'))
            pooled, _ = representations.max(dim=1)
            return pooled
            
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    @torch.no_grad()
    def forward(
        self,
        sequences: Union[List[str], List[Tuple[str, str]]],
        return_all_hiddens: bool = False,
        return_contacts: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract ESM2 embeddings from sequences.
        
        Args:
            sequences: List of sequences or list of (label, sequence) tuples
            return_all_hiddens: Whether to return all hidden layers
            return_contacts: Whether to return contact predictions
            
        Returns:
            Dictionary with:
                - 'embeddings': Sequence embeddings (batch, embed_dim)
                - 'token_embeddings': Per-token embeddings (batch, seq_len, embed_dim)
                - 'representations': Dict of hidden representations (if return_all_hiddens)
                - 'contacts': Contact predictions (if return_contacts)
        """
        # Prepare data
        if isinstance(sequences[0], str):
            # Convert to (label, sequence) format
            data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        else:
            data = sequences
            
        # Batch convert
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        # Determine which layers to extract
        if return_all_hiddens:
            repr_layers = list(range(self.num_layers + 1))
        else:
            repr_layers = [self.actual_repr_layer]
            
        # Forward pass
        results = self.model(
            batch_tokens,
            repr_layers=repr_layers,
            need_head_weights=return_contacts,
            return_contacts=return_contacts,
        )
        
        # Get token embeddings from target layer
        token_embeddings = results["representations"][self.actual_repr_layer]
        
        # Pool embeddings
        sequence_embeddings = self._pool_representations(token_embeddings, batch_tokens)
        
        output = {
            'embeddings': sequence_embeddings,
            'token_embeddings': token_embeddings,
            'batch_tokens': batch_tokens,
        }
        
        if return_all_hiddens:
            output['representations'] = results['representations']
            
        if return_contacts:
            output['contacts'] = results['contacts']
            
        return output
    
    def embed_sequences(self, sequences: List[str]) -> torch.Tensor:
        """
        Simple interface to get sequence embeddings.
        
        Args:
            sequences: List of amino acid sequences
            
        Returns:
            Embeddings tensor (batch, embed_dim)
        """
        return self.forward(sequences)['embeddings']
    
    def embed_tokens(self, sequences: List[str]) -> torch.Tensor:
        """
        Get per-token embeddings.
        
        Args:
            sequences: List of amino acid sequences
            
        Returns:
            Token embeddings (batch, max_seq_len, embed_dim)
        """
        return self.forward(sequences)['token_embeddings']


class LightweightESMProjector(nn.Module):
    """
    Lightweight projection layer to reduce ESM2 embedding dimension.
    Useful for making the model more efficient.
    """
    
    def __init__(
        self,
        esm_dim: int = 1280,  # ESM2-650M dimension
        output_dim: int = 128,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or (esm_dim + output_dim) // 2
        
        self.projector = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class ESM2StructureEvaluator(nn.Module):
    """
    Structure Evaluator using ESM2 embeddings combined with GAT.
    """
    
    def __init__(
        self,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        projection_dim: int = 128,
        gat_hidden: int = 64,
        gat_heads: int = 4,
        num_gat_layers: int = 2,
        dropout: float = 0.2,
        freeze_esm: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ESM2 Embedder
        self.esm_embedder = ESM2Embedder(
            model_name=esm_model_name,
            device=self.device,
            freeze=freeze_esm,
            pooling="mean",
        )
        
        esm_dim = self.esm_embedder.embed_dim
        output_dim = projection_dim
        
        # Projection layer
        self.projector = LightweightESMProjector(
            esm_dim=esm_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
        
        # Import GAT from structure evaluator
        from .structure_evaluator import LightweightGAT
        
        # GAT for structural analysis
        self.gat = LightweightGAT(
            input_dim=output_dim,
            hidden_dim=gat_hidden,
            output_dim=gat_hidden,
            num_heads=gat_heads,
            num_layers=num_gat_layers,
            dropout=dropout,
        )
        
        # Combine GAT and projected ESM features
        combined_dim = gat_hidden + output_dim
        
        # Stability prediction head
        self.stability_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def _build_adjacency_matrix(
        self, 
        seq_length: int, 
        batch_size: int, 
        device: torch.device,
        window_size: int = 3,
    ) -> torch.Tensor:
        """Build adjacency matrix for sequence graph."""
        adj = torch.zeros(seq_length, seq_length, device=device)
        
        for i in range(seq_length):
            for j in range(max(0, i - window_size), min(seq_length, i + window_size + 1)):
                adj[i, j] = 1.0
                
        adj = adj + torch.eye(seq_length, device=device)
        degree = adj.sum(dim=-1, keepdim=True)
        adj = adj / degree.clamp(min=1)
        
        return adj.unsqueeze(0).expand(batch_size, -1, -1)
    
    def forward(
        self,
        sequences: List[str],
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate structural stability of peptide sequences.
        
        Args:
            sequences: List of peptide sequences
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with stability scores and optional features
        """
        # Get ESM2 embeddings
        esm_output = self.esm_embedder(sequences)
        token_embeddings = esm_output['token_embeddings']  # (batch, seq_len, esm_dim)
        sequence_embeddings = esm_output['embeddings']  # (batch, esm_dim)
        
        batch_size, seq_len, _ = token_embeddings.shape
        
        # Project to lower dimension
        projected_tokens = self.projector(token_embeddings)  # (batch, seq_len, output_dim)
        projected_seq = self.projector(sequence_embeddings)  # (batch, output_dim)
        
        # Build adjacency matrix
        adj = self._build_adjacency_matrix(seq_len, batch_size, self.device)
        
        # GAT forward
        gat_output = self.gat(projected_tokens, adj)
        gat_pooled = gat_output.mean(dim=1)  # (batch, gat_hidden)
        
        # Combine features
        combined = torch.cat([gat_pooled, projected_seq], dim=-1)
        
        # Predict stability
        stability_score = self.stability_head(combined).squeeze(-1)
        
        result = {
            'stability_score': stability_score,
        }
        
        if return_features:
            result['esm_embeddings'] = sequence_embeddings
            result['projected_embeddings'] = projected_seq
            result['gat_features'] = gat_pooled
            result['combined_features'] = combined
            
        return result
    
    def compute_stability_loss(
        self,
        sequences: List[str],
        target_stability: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute stability loss."""
        outputs = self.forward(sequences)
        pred_stability = outputs['stability_score']
        
        if target_stability is not None:
            loss = nn.functional.binary_cross_entropy(pred_stability, target_stability)
        else:
            # Encourage high stability
            loss = -pred_stability.mean()
            
        return loss


# Utility function for easy loading
def load_esm2_embedder(
    model_name: str = "esm2_t33_650M_UR50D",
    device: Optional[torch.device] = None,
    **kwargs
) -> ESM2Embedder:
    """
    Load ESM2 embedder with specified model.
    
    Args:
        model_name: ESM2 model variant
        device: Device to use
        **kwargs: Additional arguments for ESM2Embedder
        
    Returns:
        ESM2Embedder instance
    """
    return ESM2Embedder(model_name=model_name, device=device, **kwargs)
