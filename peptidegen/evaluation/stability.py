"""
Peptide Stability Metrics Module.

Provides functions to calculate various stability metrics for peptides:
- Instability Index (Guruprasad et al., 1990)
- GRAVY (Grand Average of Hydropathicity)
- Aliphatic Index (Ikai, 1980)
- Isoelectric Point (pI)
- Charge at pH
- Molecular Weight
- Aromaticity
- Secondary Structure Propensity
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter

from ..constants import (
    HYDROPATHY_SCALE,
    MOLECULAR_WEIGHTS,
    INSTABILITY_WEIGHTS,
    AA_GROUPS,
)


# pKa values for charge calculations
PKA_VALUES = {
    'COOH': 2.34,  # C-terminal
    'NH2': 9.69,   # N-terminal
    'side': {
        'D': 3.86, 'E': 4.25, 'H': 6.00, 'C': 8.33,
        'Y': 10.07, 'K': 10.53, 'R': 12.48
    },
}

# Secondary structure propensity (Chou-Fasman)
HELIX_PROPENSITY = {
    'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
    'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
    'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
    'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06
}

SHEET_PROPENSITY = {
    'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
    'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
    'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
    'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70
}


def calculate_instability_index(sequence: str) -> float:
    """
    Calculate Instability Index (Guruprasad et al., 1990).
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Instability index value. < 40 indicates stable protein.
    """
    if len(sequence) < 2:
        return 0.0
    
    score = 0.0
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i:i+2]
        score += INSTABILITY_WEIGHTS.get(dipeptide, 1.0)
    
    return (10.0 / len(sequence)) * score


def calculate_gravy(sequence: str) -> float:
    """
    Calculate GRAVY (Grand Average of Hydropathicity).
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        GRAVY score. Negative = hydrophilic, positive = hydrophobic.
    """
    valid_aas = [aa for aa in sequence if aa in HYDROPATHY_SCALE]
    
    if not valid_aas:
        return 0.0
    
    return sum(HYDROPATHY_SCALE[aa] for aa in valid_aas) / len(valid_aas)


def calculate_aliphatic_index(sequence: str) -> float:
    """
    Calculate Aliphatic Index (Ikai, 1980).
    
    Higher values indicate better thermostability.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Aliphatic index value.
    """
    if len(sequence) == 0:
        return 0.0
    
    counts = Counter(sequence)
    n = len(sequence)
    
    ala = counts.get('A', 0) / n * 100
    val = counts.get('V', 0) / n * 100
    ile = counts.get('I', 0) / n * 100
    leu = counts.get('L', 0) / n * 100
    
    return ala + 2.9 * val + 3.9 * (ile + leu)


def calculate_molecular_weight(sequence: str) -> float:
    """
    Calculate molecular weight in Daltons.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Molecular weight in Da.
    """
    water_weight = 18.015  # Weight of water molecule
    
    valid_aas = [aa for aa in sequence if aa in MOLECULAR_WEIGHTS]
    if not valid_aas:
        return 0.0
    
    # Sum of AA weights minus water for each peptide bond
    return sum(MOLECULAR_WEIGHTS[aa] for aa in valid_aas) - (len(valid_aas) - 1) * water_weight


def calculate_charge_at_pH(sequence: str, pH: float = 7.0) -> float:
    """
    Calculate net charge at given pH.
    
    Args:
        sequence: Amino acid sequence
        pH: pH value (default 7.0)
        
    Returns:
        Net charge value.
    """
    pKa_side = PKA_VALUES['side']
    pKa_COOH = PKA_VALUES['COOH']
    pKa_NH2 = PKA_VALUES['NH2']
    
    # N-terminal positive charge
    charge = 1.0 / (1.0 + 10 ** (pH - pKa_NH2))
    
    # C-terminal negative charge
    charge -= 1.0 / (1.0 + 10 ** (pKa_COOH - pH))
    
    # Side chain charges
    counts = Counter(sequence)
    
    # Positive side chains (K, R, H)
    for aa in ['K', 'R', 'H']:
        if aa in counts and aa in pKa_side:
            charge += counts[aa] / (1.0 + 10 ** (pH - pKa_side[aa]))
    
    # Negative side chains (D, E, C, Y)
    for aa in ['D', 'E', 'C', 'Y']:
        if aa in counts and aa in pKa_side:
            charge -= counts[aa] / (1.0 + 10 ** (pKa_side[aa] - pH))
    
    return charge


def calculate_isoelectric_point(sequence: str) -> float:
    """
    Calculate isoelectric point (pI) using bisection method.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Isoelectric point value.
    """
    pH_low, pH_high = 0.0, 14.0
    
    while pH_high - pH_low > 0.01:
        pH_mid = (pH_low + pH_high) / 2
        charge = calculate_charge_at_pH(sequence, pH_mid)
        
        if charge > 0:
            pH_low = pH_mid
        else:
            pH_high = pH_mid
    
    return (pH_low + pH_high) / 2


def calculate_aromaticity(sequence: str) -> float:
    """
    Calculate fraction of aromatic amino acids (F, W, Y).
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Fraction of aromatic residues.
    """
    if len(sequence) == 0:
        return 0.0
    
    aromatic_aas = AA_GROUPS.get('aromatic', set('FWY'))
    aromatic = sum(1 for aa in sequence if aa in aromatic_aas)
    return aromatic / len(sequence)


def calculate_secondary_structure_propensity(sequence: str) -> Dict[str, float]:
    """
    Calculate secondary structure propensities.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary with helix and sheet propensities.
    """
    valid_aas = [aa for aa in sequence if aa in HELIX_PROPENSITY]
    
    if not valid_aas:
        return {'helix': 0.0, 'sheet': 0.0, 'helix_fraction': 0.5, 'sheet_fraction': 0.5}
    
    helix_score = sum(HELIX_PROPENSITY[aa] for aa in valid_aas) / len(valid_aas)
    sheet_score = sum(SHEET_PROPENSITY[aa] for aa in valid_aas) / len(valid_aas)
    
    # Normalize
    total = helix_score + sheet_score
    if total > 0:
        helix_frac = helix_score / total
        sheet_frac = sheet_score / total
    else:
        helix_frac = sheet_frac = 0.5
    
    return {
        'helix': helix_score,
        'sheet': sheet_score,
        'helix_fraction': helix_frac,
        'sheet_fraction': sheet_frac
    }


def calculate_amino_acid_composition(sequence: str) -> Dict[str, float]:
    """
    Calculate amino acid composition as percentages.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary mapping AA to percentage.
    """
    counts = Counter(sequence)
    n = len(sequence) if len(sequence) > 0 else 1
    
    composition = {}
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        composition[aa] = counts.get(aa, 0) / n * 100
    
    return composition


class PeptideStabilityAnalyzer:
    """
    Comprehensive peptide stability analyzer.
    """
    
    STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
    
    def __init__(self, stability_threshold: float = 40.0):
        """
        Initialize analyzer.
        
        Args:
            stability_threshold: Instability index threshold (default 40.0)
        """
        self.stability_threshold = stability_threshold
    
    def clean_sequence(self, sequence: str) -> str:
        """Remove non-standard amino acids from sequence."""
        return ''.join(aa for aa in sequence.upper() if aa in self.STANDARD_AA)
    
    def analyze(self, sequence: str) -> Optional[Dict]:
        """
        Analyze a single peptide sequence.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary of metrics or None if invalid.
        """
        sequence = self.clean_sequence(sequence)
        
        if len(sequence) < 5:
            return None
        
        metrics = {
            'length': len(sequence),
            'instability_index': calculate_instability_index(sequence),
            'gravy': calculate_gravy(sequence),
            'aliphatic_index': calculate_aliphatic_index(sequence),
            'molecular_weight': calculate_molecular_weight(sequence),
            'charge_ph7': calculate_charge_at_pH(sequence, 7.0),
            'isoelectric_point': calculate_isoelectric_point(sequence),
            'aromaticity': calculate_aromaticity(sequence),
        }
        
        # Secondary structure
        ss_props = calculate_secondary_structure_propensity(sequence)
        metrics['helix_propensity'] = ss_props['helix']
        metrics['sheet_propensity'] = ss_props['sheet']
        
        # Stability classification
        metrics['is_stable'] = metrics['instability_index'] < self.stability_threshold
        
        return metrics
    
    def analyze_batch(self, sequences: List[str]) -> Dict:
        """
        Analyze multiple sequences and compute statistics.
        
        Args:
            sequences: List of sequences
            
        Returns:
            Dictionary with metrics and summary statistics.
        """
        results = {
            'total_sequences': len(sequences),
            'valid_sequences': 0,
            'stable_sequences': 0,
            'metrics': [],
            'summary': {}
        }
        
        all_metrics = []
        
        for i, seq in enumerate(sequences):
            metrics = self.analyze(seq)
            if metrics:
                metrics['sequence'] = self.clean_sequence(seq)
                metrics['id'] = f"seq_{i}"
                all_metrics.append(metrics)
                results['valid_sequences'] += 1
                if metrics['is_stable']:
                    results['stable_sequences'] += 1
        
        results['metrics'] = all_metrics
        
        # Calculate summary statistics
        if all_metrics:
            numeric_keys = [
                'length', 'instability_index', 'gravy', 'aliphatic_index',
                'molecular_weight', 'charge_ph7', 'isoelectric_point',
                'aromaticity', 'helix_propensity', 'sheet_propensity'
            ]
            
            summary = {}
            for key in numeric_keys:
                values = [m[key] for m in all_metrics]
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
            
            summary['stability_rate'] = (
                results['stable_sequences'] / results['valid_sequences'] * 100
                if results['valid_sequences'] > 0 else 0.0
            )
            results['summary'] = summary
        
        return results
    
    def get_stable_sequences(self, sequences: List[str]) -> List[Tuple[str, Dict]]:
        """
        Filter and return only stable sequences.
        
        Args:
            sequences: List of sequences
            
        Returns:
            List of (sequence, metrics) tuples for stable sequences.
        """
        stable = []
        for seq in sequences:
            metrics = self.analyze(seq)
            if metrics and metrics['is_stable']:
                stable.append((self.clean_sequence(seq), metrics))
        return stable
