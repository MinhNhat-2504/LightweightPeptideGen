"""
Training module for LightweightPeptideGen.

Classes:
    - GANTrainer: Base GAN trainer with anti-mode-collapse
    - ConditionalGANTrainer: GAN trainer with feature conditioning

Loss Functions:
    - DiversityLoss: Prevents mode collapse via entropy & batch diversity
    - NgramDiversityLoss: Bigram/trigram concentration penalty
    - LengthPenaltyLoss: Cumulative-EOS supervision for length control
    - FeatureMatchingLoss: Stabilises training via feature matching
    - ReconstructionLoss: Cross-entropy reconstruction loss
    - StabilityBiasLoss: Differentiable instability-index penalty

Removed: WassersteinLoss, GradientPenalty (dead code; WGAN-GP logic lives
in GANTrainer._gradient_penalty / train_step).
"""

from .trainer import GANTrainer, ConditionalGANTrainer
from .rl import SCSTTrainer, MultiObjectiveReward
from .losses import (
    DiversityLoss,
    NgramDiversityLoss,
    LengthPenaltyLoss,
    FeatureMatchingLoss,
    ReconstructionLoss,
    StabilityBiasLoss,
)

__all__ = [
    'GANTrainer',
    'ConditionalGANTrainer',
    'SCSTTrainer',
    'MultiObjectiveReward',
    'DiversityLoss',
    'NgramDiversityLoss',
    'LengthPenaltyLoss',
    'FeatureMatchingLoss',
    'ReconstructionLoss',
    'StabilityBiasLoss',
]
