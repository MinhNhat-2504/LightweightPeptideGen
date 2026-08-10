# Evaluation module for LightweightPeptideGen
from .stability import (
    PeptideStabilityAnalyzer,
    calculate_instability_index,
    calculate_gravy,
    calculate_aliphatic_index,
    calculate_isoelectric_point,
    calculate_charge_at_pH,
    calculate_molecular_weight,
    calculate_aromaticity,
)

from .metrics import (
    calculate_diversity_metrics,
    calculate_length_statistics,
    calculate_amino_acid_distribution,
    detect_mode_collapse,
    compare_distributions,
    # AMP metrics (merged into metrics.py from amp_metrics.py)
    calculate_hemolytic_score,
    calculate_therapeutic_score,
    estimate_amp_probability,
    calculate_hydrophobicity,
    calculate_hydrophobic_moment,
    calculate_net_charge,
    analyze_amp_properties,
)

from .quality_filter import (
    PeptideQualityFilter,
    QualityCriteria,
    QualityScore,
    filter_generated_fasta,
)

__all__ = [
    # Stability
    'PeptideStabilityAnalyzer',
    'calculate_instability_index',
    'calculate_gravy',
    'calculate_aliphatic_index',
    'calculate_isoelectric_point',
    'calculate_charge_at_pH',
    'calculate_molecular_weight',
    'calculate_aromaticity',
    # Metrics
    'calculate_diversity_metrics',
    'calculate_length_statistics',
    'calculate_amino_acid_distribution',
    'detect_mode_collapse',
    'compare_distributions',
    # AMP Metrics (now in metrics.py)
    'calculate_hemolytic_score',
    'calculate_therapeutic_score',
    'estimate_amp_probability',
    'calculate_hydrophobicity',
    'calculate_hydrophobic_moment',
    'calculate_net_charge',
    'analyze_amp_properties',
    # Quality Filter
    'PeptideQualityFilter',
    'QualityCriteria',
    'QualityScore',
    'filter_generated_fasta',
]
