"""
Inference module for peptide generation.

Classes:
    - PeptideSampler: Generate sequences with various sampling strategies

Functions:
    - load_generator: Quick load generator from checkpoint
"""

from .sampler import PeptideSampler, load_generator

__all__ = [
    'PeptideSampler',
    'load_generator',
]
