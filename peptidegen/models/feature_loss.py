"""
Feature-based Loss Functions for Controlled Peptide Generation.

These losses guide the generator to produce peptides with desired properties:
- Stability Loss: Penalize high instability_index (want < 40)
- Therapeutic Loss: Reward high therapeutic_score
- Toxicity Loss: Penalize high hemolytic_score
- Quality Loss: Combined multi-objective loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

# Import constants from centralized module
from ..constants import AA_GROUPS, QUALITY_THRESHOLDS

# Type checking imports (not executed at runtime)
if TYPE_CHECKING:
    from ..data.vocabulary import PeptideVocabulary


class PeptideFeaturePredictor(nn.Module):
    """
    Neural network to predict peptide features from sequence representation.
    Used to compute feature-based losses during training.
    """
    
    # Feature names and their target ranges
    FEATURE_CONFIG = {
        'instability_index': {'target': 'low', 'threshold': 40.0, 'weight': 1.0},
        'therapeutic_score': {'target': 'high', 'threshold': 0.5, 'weight': 1.5},
        'hemolytic_score': {'target': 'low', 'threshold': 0.3, 'weight': 2.0},
        'aliphatic_index': {'target': 'high', 'threshold': 60.0, 'weight': 0.5},
        'gravy': {'target': 'range', 'min': -1.0, 'max': 0.5, 'weight': 0.3},
    }
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_features: int = 5,
        dropout: float = 0.2,
    ):
        """
        Initialize feature predictor.
        
        Args:
            input_dim: Input dimension (from generator hidden state)
            hidden_dim: Hidden layer dimension
            num_features: Number of features to predict
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_features = num_features
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_features),
        )
        
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Predict features from hidden state.
        
        Args:
            hidden_state: Generator hidden state (batch_size, hidden_dim)
            
        Returns:
            Predicted features (batch_size, num_features)
        """
        return self.predictor(hidden_state)


class StabilityLoss(nn.Module):
    """
    Loss to encourage stable peptides (instability_index < 40).
    """
    
    def __init__(self, threshold: float = 40.0, margin: float = 10.0):
        """
        Args:
            threshold: Instability index threshold for stable peptides
            margin: Soft margin for the loss
        """
        super().__init__()
        self.threshold = threshold
        self.margin = margin
        
    def forward(
        self, 
        predicted_ii: torch.Tensor, 
        target_ii: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute stability loss.
        
        Args:
            predicted_ii: Predicted instability index (batch_size,)
            target_ii: Optional target instability index
            
        Returns:
            Scalar loss value
        """
        # Penalize values above threshold with soft margin
        # Loss = max(0, predicted - threshold + margin)^2
        excess = F.relu(predicted_ii - self.threshold + self.margin)
        loss = torch.mean(excess ** 2)
        
        # If target is provided, also add MSE loss
        if target_ii is not None:
            loss = loss + F.mse_loss(predicted_ii, target_ii)
            
        return loss


class TherapeuticLoss(nn.Module):
    """
    Loss to encourage high therapeutic score.
    """
    
    def __init__(self, target_score: float = 1.0, weight: float = 1.0):
        """
        Args:
            target_score: Target therapeutic score to achieve
            weight: Loss weight
        """
        super().__init__()
        self.target_score = target_score
        self.weight = weight
        
    def forward(
        self, 
        predicted_score: torch.Tensor,
        target_score: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute therapeutic loss (want to maximize score).
        
        Args:
            predicted_score: Predicted therapeutic score (batch_size,)
            target_score: Optional target score
            
        Returns:
            Scalar loss value
        """
        # Negative because we want to maximize therapeutic score
        # Loss = -log(predicted_score + eps) to encourage high values
        loss = -torch.mean(torch.log(predicted_score + 1e-8))
        
        # Add MSE to target if provided
        if target_score is not None:
            loss = loss + F.mse_loss(predicted_score, target_score)
        else:
            # Pull towards target_score
            loss = loss + F.mse_loss(predicted_score, 
                                      torch.full_like(predicted_score, self.target_score))
            
        return self.weight * loss


class ToxicityLoss(nn.Module):
    """
    Loss to penalize high hemolytic (toxicity) score.
    """
    
    def __init__(self, threshold: float = 0.3, weight: float = 2.0):
        """
        Args:
            threshold: Maximum acceptable hemolytic score
            weight: Loss weight (higher = more penalty for toxic peptides)
        """
        super().__init__()
        self.threshold = threshold
        self.weight = weight
        
    def forward(
        self,
        predicted_toxicity: torch.Tensor,
        target_toxicity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute toxicity loss (want to minimize).
        
        Args:
            predicted_toxicity: Predicted hemolytic score (batch_size,)
            target_toxicity: Optional target score
            
        Returns:
            Scalar loss value
        """
        # Penalize values above threshold exponentially
        excess = F.relu(predicted_toxicity - self.threshold)
        loss = torch.mean(torch.exp(excess) - 1)
        
        # Add direct minimization term
        loss = loss + torch.mean(predicted_toxicity)
        
        if target_toxicity is not None:
            loss = loss + F.mse_loss(predicted_toxicity, target_toxicity)
            
        return self.weight * loss


class FeatureConditioningLoss(nn.Module):
    """
    Loss to ensure generated peptides match the input condition features.
    Used for conditional generation to control output properties.
    """
    
    def __init__(
        self,
        feature_names: List[str],
        feature_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            feature_names: List of feature names
            feature_weights: Optional weights for each feature
        """
        super().__init__()
        
        self.feature_names = feature_names
        self.num_features = len(feature_names)
        
        # Default weights
        default_weights = {
            'instability_index': 1.0,
            'therapeutic_score': 1.5,
            'hemolytic_score': 2.0,
            'aliphatic_index': 0.5,
            'hydrophobic_moment': 0.5,
            'gravy': 0.3,
            'charge_at_pH7': 0.3,
            'aromaticity': 0.3,
        }
        
        self.weights = []
        for name in feature_names:
            w = feature_weights.get(name, default_weights.get(name, 1.0)) if feature_weights else default_weights.get(name, 1.0)
            self.weights.append(w)
        self.weights = torch.tensor(self.weights)
        
    def forward(
        self,
        predicted_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute conditioning loss between predicted and target features.
        
        Args:
            predicted_features: Predicted feature values (batch_size, num_features)
            target_features: Target/condition feature values (batch_size, num_features)
            
        Returns:
            Tuple of (total_loss, per_feature_losses)
        """
        # Move weights to same device
        weights = self.weights.to(predicted_features.device)
        
        # Compute per-feature MSE
        per_feature_mse = (predicted_features - target_features) ** 2
        
        # Weighted sum
        weighted_mse = per_feature_mse * weights.unsqueeze(0)
        total_loss = torch.mean(weighted_mse)
        
        # Per-feature losses for logging
        per_feature_losses = {}
        for i, name in enumerate(self.feature_names):
            per_feature_losses[name] = torch.mean(per_feature_mse[:, i]).item()
            
        return total_loss, per_feature_losses


class MultiObjectiveFeatureLoss(nn.Module):
    """
    Combined multi-objective loss for peptide quality.
    
    Objectives:
        1. Stability: Low instability_index (< 40)
        2. Therapeutic: High therapeutic_score
        3. Safety: Low hemolytic_score
        4. Thermostability: High aliphatic_index
    """
    
    def __init__(
        self,
        stability_weight: float = 1.0,
        therapeutic_weight: float = 1.0,
        toxicity_weight: float = 1.5,
        conditioning_weight: float = 0.5,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Args:
            stability_weight: Weight for stability loss
            therapeutic_weight: Weight for therapeutic loss
            toxicity_weight: Weight for toxicity loss
            conditioning_weight: Weight for feature conditioning loss
            feature_names: Feature names for conditioning loss
        """
        super().__init__()
        
        self.stability_loss = StabilityLoss(threshold=40.0)
        self.therapeutic_loss = TherapeuticLoss(target_score=1.0)
        self.toxicity_loss = ToxicityLoss(threshold=0.3)
        
        self.stability_weight = stability_weight
        self.therapeutic_weight = therapeutic_weight
        self.toxicity_weight = toxicity_weight
        self.conditioning_weight = conditioning_weight
        
        if feature_names:
            self.conditioning_loss = FeatureConditioningLoss(feature_names)
        else:
            self.conditioning_loss = None
            
    def forward(
        self,
        predicted_features: Dict[str, torch.Tensor],
        target_features: Optional[Dict[str, torch.Tensor]] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined multi-objective loss.
        
        Args:
            predicted_features: Dict with predicted feature tensors
            target_features: Optional dict with target feature tensors
            condition: Optional condition tensor for conditioning loss
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(predicted_features.values())).device)
        
        # Stability loss
        if 'instability_index' in predicted_features:
            target_ii = target_features.get('instability_index') if target_features else None
            loss_stability = self.stability_loss(predicted_features['instability_index'], target_ii)
            total_loss = total_loss + self.stability_weight * loss_stability
            losses['stability'] = loss_stability.item()
            
        # Therapeutic loss  
        if 'therapeutic_score' in predicted_features:
            target_ts = target_features.get('therapeutic_score') if target_features else None
            loss_therapeutic = self.therapeutic_loss(predicted_features['therapeutic_score'], target_ts)
            total_loss = total_loss + self.therapeutic_weight * loss_therapeutic
            losses['therapeutic'] = loss_therapeutic.item()
            
        # Toxicity loss
        if 'hemolytic_score' in predicted_features:
            target_hs = target_features.get('hemolytic_score') if target_features else None
            loss_toxicity = self.toxicity_loss(predicted_features['hemolytic_score'], target_hs)
            total_loss = total_loss + self.toxicity_weight * loss_toxicity
            losses['toxicity'] = loss_toxicity.item()
            
        # Conditioning loss
        if self.conditioning_loss is not None and condition is not None:
            # Stack predicted features
            pred_stack = torch.stack([
                predicted_features.get(name, torch.zeros_like(condition[:, 0]))
                for name in self.conditioning_loss.feature_names
            ], dim=-1)
            
            loss_cond, per_feat_losses = self.conditioning_loss(pred_stack, condition)
            total_loss = total_loss + self.conditioning_weight * loss_cond
            losses['conditioning'] = loss_cond.item()
            losses.update({f'cond_{k}': v for k, v in per_feat_losses.items()})
            
        losses['total'] = total_loss.item()
        
        return total_loss, losses


class AminoAcidFeatureLoss(nn.Module):
    """
    Loss based on amino acid composition to encourage diversity
    and proper feature distribution.
    """
    
    # Amino acid groups and their properties
    AA_GROUPS = {
        'aliphatic': list('AVILM'),  # For aliphatic_index
        'aromatic': list('FWY'),      # For aromaticity
        'positive': list('KRH'),      # For positive charge
        'negative': list('DE'),       # For negative charge
        'hydrophobic': list('AVILMFWP'),  # For hydrophobicity
        'polar': list('STNQ'),        # For polarity
    }
    
    def __init__(
        self,
        vocab_size: int = 24,
        target_aliphatic_ratio: float = 0.3,
        target_diversity: float = 0.8,
    ):
        """
        Args:
            vocab_size: Vocabulary size
            target_aliphatic_ratio: Target ratio of aliphatic AAs
            target_diversity: Target sequence diversity (unique AAs / length)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.target_aliphatic_ratio = target_aliphatic_ratio
        self.target_diversity = target_diversity
        
    def forward(
        self,
        logits: torch.Tensor,
        vocab: 'PeptideVocabulary',
    ) -> torch.Tensor:
        """
        Compute AA composition loss from logits.
        
        Args:
            logits: Generator output logits (batch_size, seq_len, vocab_size)
            vocab: Vocabulary for AA mapping
            
        Returns:
            Composition loss
        """
        # Get probabilities
        probs = F.softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)
        
        # Average over sequence
        avg_probs = probs.mean(dim=1)  # (batch, vocab_size)
        
        # Compute aliphatic ratio using centralized constants
        aliphatic_indices = [vocab.aa_to_idx.get(aa, 0) for aa in AA_GROUPS['aliphatic']]
        aliphatic_prob = avg_probs[:, aliphatic_indices].sum(dim=-1)
        
        # Loss: encourage aliphatic ratio near target
        aliphatic_loss = (aliphatic_prob - self.target_aliphatic_ratio) ** 2
        
        # Diversity loss: encourage uniform distribution (entropy)
        # Exclude special tokens
        aa_probs = avg_probs[:, 4:]  # Assuming first 4 are special tokens
        entropy = -torch.sum(aa_probs * torch.log(aa_probs + 1e-10), dim=-1)
        max_entropy = np.log(20)  # 20 standard amino acids
        diversity_loss = (max_entropy - entropy) / max_entropy
        
        return torch.mean(aliphatic_loss + 0.5 * diversity_loss)
