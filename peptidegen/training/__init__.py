"""
Training module for LightweightPeptideGen.

Classes:
    - GANTrainer: Base GAN trainer with anti-mode-collapse
    - ConditionalGANTrainer: GAN trainer with feature conditioning

Loss Functions:
    - DiversityLoss: Prevents mode collapse via entropy & batch diversity
    - FeatureMatchingLoss: Stabilizes training via feature matching
    - ReconstructionLoss: Cross-entropy reconstruction loss
    - WassersteinLoss: WGAN loss functions
    - GradientPenalty: WGAN-GP gradient penalty
"""

from .trainer import GANTrainer, ConditionalGANTrainer
from .rl import SCSTTrainer, MultiObjectiveReward
from .losses import (
    DiversityLoss,
    FeatureMatchingLoss,
    ReconstructionLoss,
    WassersteinLoss,
    GradientPenalty,
)

__all__ = [
    'GANTrainer',
    'ConditionalGANTrainer',
    'SCSTTrainer',
    'MultiObjectiveReward',
    'DiversityLoss',
    'FeatureMatchingLoss',
    'ReconstructionLoss',
    'WassersteinLoss',
    'GradientPenalty',
]
