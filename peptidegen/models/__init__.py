"""
Models for Lightweight Peptide Generation
"""

from .generator import PeptideGenerator, GRUGenerator
from .fusion_generator import (
    MultimodalFusionGenerator,
    CrossAttentionFusion,
    DenseGATv2Layer,
)
from .discriminator import (
    PeptideDiscriminator,
    CNNDiscriminator,
    RNNDiscriminator,
    HybridDiscriminator,
)
from .structure_evaluator import StructureEvaluator, LightweightGAT, StabilityLoss
from .components import (
    PositionalEncoding,
    MultiHeadAttention,
    ConditionalBatchNorm,
    ResidualBlock,
)

# Feature-based loss (always available)
from .feature_loss import (
    PeptideFeaturePredictor,
    MultiObjectiveFeatureLoss,
    FeatureConditioningLoss,
    AminoAcidFeatureLoss,
)

# ESM2 imports (optional - may not be installed)
try:
    from .esm2_embedder import (
        ESM2Embedder,
        ESM2StructureEvaluator,
        LightweightESMProjector,
        load_esm2_embedder,
    )
    from .esm2_generator import ESM2ConditionedGenerator, ESM2GuidedDiscriminator
    HAS_ESM2 = True
except ImportError:
    ESM2Embedder = None
    ESM2StructureEvaluator = None
    LightweightESMProjector = None
    load_esm2_embedder = None
    ESM2ConditionedGenerator = None
    ESM2GuidedDiscriminator = None
    HAS_ESM2 = False

__all__ = [
    # Generators (LSTMGenerator and TransformerGenerator removed — dead code)
    'PeptideGenerator',
    'GRUGenerator',
    'MultimodalFusionGenerator',
    'CrossAttentionFusion',
    'DenseGATv2Layer',
    # Discriminators
    'PeptideDiscriminator',
    'CNNDiscriminator',
    'RNNDiscriminator',
    'HybridDiscriminator',
    # Structure Evaluators
    'StructureEvaluator',
    'LightweightGAT',
    'StabilityLoss',
    # Feature Loss
    'PeptideFeaturePredictor',
    'MultiObjectiveFeatureLoss',
    'FeatureConditioningLoss',
    'AminoAcidFeatureLoss',
    # Components
    'PositionalEncoding',
    'MultiHeadAttention',
    'ConditionalBatchNorm',
    'ResidualBlock',
    # ESM2 (optional)
    'ESM2Embedder',
    'ESM2StructureEvaluator',
    'LightweightESMProjector',
    'load_esm2_embedder',
    'ESM2ConditionedGenerator',
    'ESM2GuidedDiscriminator',
    'HAS_ESM2',
]
