"""
LightweightPeptideGen - Antimicrobial Peptide Generation with GANs

A lightweight GAN framework for generating stable antimicrobial peptides.

Modules:
    - data: Dataset, DataLoader, Vocabulary, Features
    - models: Generator, Discriminator, ESM2 embedder
    - training: GANTrainer, Loss functions
    - inference: PeptideSampler
    - evaluation: Metrics, Stability analysis

Example:
    >>> from peptidegen import GANTrainer, GRUGenerator, CNNDiscriminator
    >>> from peptidegen import PeptideSampler, load_config
    >>> 
    >>> # Training
    >>> trainer = GANTrainer(generator, discriminator, config)
    >>> trainer.fit(train_loader, epochs=100)
    >>> 
    >>> # Generation
    >>> sampler = PeptideSampler.from_checkpoint('checkpoints/best_model.pt')
    >>> sequences = sampler.sample(n=100, temperature=0.8)
"""

__version__ = "2.0.0"
__author__ = "Thesis Project"

# Force UTF-8 console so log glyphs (✓ ⭐ ⚠️) don't crash the Windows cp1252
# stdout with UnicodeEncodeError. Safe no-op on platforms already UTF-8.
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Data
from .data import (
    VOCAB,
    PeptideDataset,
    ConditionalPeptideDataset,
    get_dataloader,
    PeptideFeatureExtractor,
)

# Models
from .models import (
    GRUGenerator,
    LSTMGenerator,
    TransformerGenerator,
    CNNDiscriminator,
    RNNDiscriminator,
    StructureEvaluator,
)

# Training
from .training import (
    GANTrainer,
    ConditionalGANTrainer,
    DiversityLoss,
)

# Inference
from .inference import (
    PeptideSampler,
    load_generator,
)

# Utils
from .utils import load_config, set_seed, setup_logging, get_device

__all__ = [
    # Data
    'VOCAB',
    'PeptideDataset',
    'ConditionalPeptideDataset', 
    'get_dataloader',
    'PeptideFeatureExtractor',
    # Models
    'GRUGenerator',
    'LSTMGenerator',
    'TransformerGenerator',
    'CNNDiscriminator',
    'RNNDiscriminator',
    'StructureEvaluator',
    # Training
    'GANTrainer',
    'ConditionalGANTrainer',
    'DiversityLoss',
    # Inference
    'PeptideSampler',
    'load_generator',
    # Utils
    'load_config',
    'set_seed',
    'setup_logging',
    'get_device',
]
