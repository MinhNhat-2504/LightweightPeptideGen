"""
ESM2-enhanced Generator for peptide sequence generation
Uses ESM2 embeddings as conditioning signal for generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
import logging

from .generator import PeptideGenerator, GRUGenerator
from .esm2_embedder import ESM2Embedder, LightweightESMProjector

logger = logging.getLogger(__name__)


class ESM2ConditionedGenerator(nn.Module):
    """
    Generator conditioned on ESM2 embeddings.
    Uses ESM2 to provide rich protein context for generation.
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
        esm_model_name: str = "esm2_t33_650M_UR50D",
        esm_projection_dim: int = 128,
        freeze_esm: bool = True,
        generator_type: str = "GRU",
        pad_idx: int = 0,
        sos_idx: int = 1,
        eos_idx: int = 2,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.esm_projection_dim = esm_projection_dim
        
        # ESM2 Embedder
        self.esm_embedder = ESM2Embedder(
            model_name=esm_model_name,
            device=self.device,
            freeze=freeze_esm,
            pooling="mean",
        )
        
        # Project ESM embeddings
        self.esm_projector = LightweightESMProjector(
            esm_dim=self.esm_embedder.embed_dim,
            output_dim=esm_projection_dim,
            dropout=dropout,
        )
        
        # Total condition dimension = latent + ESM projection
        condition_dim = latent_dim + esm_projection_dim
        
        # Base generator
        if generator_type == "GRU":
            self.generator = GRUGenerator(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                latent_dim=condition_dim,  # Use combined dimension
                max_length=max_length,
                num_layers=num_layers,
                dropout=dropout,
                condition_dim=None,  # Already included in latent
                pad_idx=pad_idx,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
            )
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
            
        logger.info(f"Created ESM2-conditioned generator with {esm_model_name}")
        
    def get_esm_condition(
        self,
        reference_sequences: Optional[List[str]] = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Get ESM2-based conditioning vector.
        
        Args:
            reference_sequences: Optional reference sequences for conditioning
            batch_size: Batch size (used if no reference sequences)
            
        Returns:
            ESM2 condition tensor
        """
        if reference_sequences is not None:
            # Get ESM2 embeddings from reference sequences
            esm_output = self.esm_embedder(reference_sequences)
            esm_embeddings = esm_output['embeddings']
            esm_condition = self.esm_projector(esm_embeddings)
        else:
            # Use zero condition (unconditional generation)
            esm_condition = torch.zeros(
                batch_size, self.esm_projection_dim, device=self.device
            )
            
        return esm_condition
    
    def forward(
        self,
        z: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        reference_sequences: Optional[List[str]] = None,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate peptide sequences with ESM2 conditioning.
        
        Args:
            z: Latent vector (batch_size, latent_dim)
            target: Target sequence for teacher forcing
            reference_sequences: Reference sequences for ESM2 conditioning
            temperature: Sampling temperature
        """
        batch_size = z.size(0)
        
        # Get ESM2 condition
        esm_condition = self.get_esm_condition(reference_sequences, batch_size)
        
        # Combine latent with ESM condition
        combined_z = torch.cat([z, esm_condition], dim=-1)
        
        # Generate
        return self.generator(combined_z, target, temperature=temperature)
    
    def generate(
        self,
        batch_size: int = 1,
        z: Optional[torch.Tensor] = None,
        reference_sequences: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Generate sequences with optional ESM2 conditioning."""
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            
        esm_condition = self.get_esm_condition(reference_sequences, batch_size)
        combined_z = torch.cat([z, esm_condition], dim=-1)
        
        return self.generator.generate(
            batch_size=batch_size,
            z=combined_z,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=self.device,
        )


class ESM2GuidedDiscriminator(nn.Module):
    """
    Discriminator enhanced with ESM2 features.
    Uses ESM2 embeddings to better distinguish real vs generated sequences.
    """
    
    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        max_length: int = 50,
        dropout: float = 0.2,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        esm_projection_dim: int = 128,
        freeze_esm: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Local embedding (for generated sequences)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # ESM2 Embedder
        self.esm_embedder = ESM2Embedder(
            model_name=esm_model_name,
            device=self.device,
            freeze=freeze_esm,
            pooling="mean",
        )
        
        # Project ESM embeddings
        self.esm_projector = LightweightESMProjector(
            esm_dim=self.esm_embedder.embed_dim,
            output_dim=esm_projection_dim,
            dropout=dropout,
        )
        
        # Local sequence encoder
        self.local_encoder = nn.GRU(
            embedding_dim,
            hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        
        # Combined classifier
        combined_dim = hidden_dim + esm_projection_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(
        self,
        sequences: List[str],
        x: Optional[torch.Tensor] = None,  # For compatibility with tensor input
    ) -> torch.Tensor:
        """
        Discriminate sequences.
        
        Args:
            sequences: List of peptide sequences (strings)
            x: Optional tensor input (indices)
            
        Returns:
            Discrimination scores
        """
        # Get ESM2 features
        esm_output = self.esm_embedder(sequences)
        esm_features = self.esm_projector(esm_output['embeddings'])
        
        # Get local features from token embeddings
        token_embeddings = esm_output['token_embeddings']
        batch_size = token_embeddings.size(0)
        
        # Use simple projection if dimensions don't match
        if token_embeddings.size(-1) != self.embedding.embedding_dim:
            # Project ESM token embeddings to local embedding dimension
            local_embed = self.esm_projector(token_embeddings)
        else:
            local_embed = token_embeddings
            
        # Encode with local encoder
        _, hidden = self.local_encoder(local_embed)
        local_features = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        
        # Combine features
        combined = torch.cat([local_features, esm_features], dim=-1)
        
        # Classify
        return self.classifier(combined)
    
    def get_feature(self, sequences: List[str]) -> torch.Tensor:
        """Extract features for feature matching loss."""
        esm_output = self.esm_embedder(sequences)
        esm_features = self.esm_projector(esm_output['embeddings'])
        
        token_embeddings = esm_output['token_embeddings']
        local_embed = self.esm_projector(token_embeddings)
        
        _, hidden = self.local_encoder(local_embed)
        local_features = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        
        return torch.cat([local_features, esm_features], dim=-1)
