"""
Feature extraction for peptide sequences
"""

import numpy as np
from typing import List, Dict, Optional, Union
import math


class PeptideFeatureExtractor:
    """
    Extract physicochemical and structural features from peptide sequences.
    Lightweight implementation without heavy dependencies.
    """
    
    # Amino acid properties
    AA_PROPERTIES = {
        # Hydrophobicity (Kyte-Doolittle scale)
        'hydrophobicity': {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        },
        # Molecular weight
        'molecular_weight': {
            'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
            'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
            'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
            'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2
        },
        # Charge at pH 7
        'charge': {
            'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
            'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
            'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
            'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
        },
        # pKa of side chain
        'pKa': {
            'A': None, 'C': 8.3, 'D': 3.9, 'E': 4.1, 'F': None,
            'G': None, 'H': 6.0, 'I': None, 'K': 10.5, 'L': None,
            'M': None, 'N': None, 'P': None, 'Q': None, 'R': 12.5,
            'S': None, 'T': None, 'V': None, 'W': None, 'Y': 10.1
        },
        # Aromaticity
        'aromatic': {
            'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 1,
            'G': 0, 'H': 1, 'I': 0, 'K': 0, 'L': 0,
            'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 0,
            'S': 0, 'T': 0, 'V': 0, 'W': 1, 'Y': 1
        },
        # Aliphatic
        'aliphatic': {
            'A': 1, 'C': 0, 'D': 0, 'E': 0, 'F': 0,
            'G': 0, 'H': 0, 'I': 1, 'K': 0, 'L': 1,
            'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 0,
            'S': 0, 'T': 0, 'V': 1, 'W': 0, 'Y': 0
        },
        # Polar
        'polar': {
            'A': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 0,
            'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0,
            'M': 0, 'N': 1, 'P': 0, 'Q': 1, 'R': 1,
            'S': 1, 'T': 1, 'V': 0, 'W': 0, 'Y': 1
        },
        # Volume (Å³)
        'volume': {
            'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
            'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
            'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
            'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
        },
    }
    
    # Instability index weights (DIWV) - use complete table from constants
    INSTABILITY_WEIGHTS = None  # Will be loaded from constants

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize feature extractor.

        Args:
            feature_names: List of feature names to extract. If None, extract all.
        """
        # Load complete INSTABILITY_WEIGHTS from centralized constants
        from ..constants import INSTABILITY_WEIGHTS as COMPLETE_WEIGHTS
        self.INSTABILITY_WEIGHTS = COMPLETE_WEIGHTS
        self.all_features = [
            'length', 'molecular_weight', 'aromaticity', 'instability_index',
            'isoelectric_point', 'gravy', 'charge_at_pH7', 'hydrophobic_ratio',
            'positive_ratio', 'negative_ratio', 'aromatic_ratio', 'aliphatic_index',
            'boman_index', 'hydrophobic_moment', 'amphipathicity'
        ]
        
        if feature_names is None:
            self.feature_names = self.all_features
        else:
            self.feature_names = feature_names
            
    def extract(self, sequence: str) -> List[float]:
        """
        Extract features from a peptide sequence.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            List of feature values
        """
        sequence = sequence.upper()
        features = []
        
        for name in self.feature_names:
            if hasattr(self, f'_calc_{name}'):
                value = getattr(self, f'_calc_{name}')(sequence)
            else:
                value = 0.0
            features.append(value)
            
        return features
    
    def extract_dict(self, sequence: str) -> Dict[str, float]:
        """Extract features as dictionary."""
        values = self.extract(sequence)
        return dict(zip(self.feature_names, values))
    
    def extract_batch(self, sequences: List[str]) -> np.ndarray:
        """Extract features for a batch of sequences."""
        return np.array([self.extract(seq) for seq in sequences])
    
    # Feature calculation methods
    def _calc_length(self, sequence: str) -> float:
        return float(len(sequence))
    
    def _calc_molecular_weight(self, sequence: str) -> float:
        """Calculate molecular weight."""
        mw = sum(self.AA_PROPERTIES['molecular_weight'].get(aa, 0) for aa in sequence)
        # Subtract water for peptide bonds
        mw -= 18.015 * (len(sequence) - 1)
        return mw
    
    def _calc_aromaticity(self, sequence: str) -> float:
        """Calculate aromaticity (fraction of aromatic residues)."""
        aromatic_count = sum(self.AA_PROPERTIES['aromatic'].get(aa, 0) for aa in sequence)
        return aromatic_count / len(sequence) if sequence else 0.0
    
    def _calc_instability_index(self, sequence: str) -> float:
        """
        Calculate instability index.
        Proteins with II < 40 are considered stable.
        """
        if len(sequence) < 2:
            return 0.0
            
        # Simplified calculation
        dipeptide_score = 0.0
        for i in range(len(sequence) - 1):
            dipeptide = sequence[i:i+2]
            dipeptide_score += self.INSTABILITY_WEIGHTS.get(dipeptide, 1.0)
            
        return (10.0 / len(sequence)) * dipeptide_score
    
    def _calc_isoelectric_point(self, sequence: str) -> float:
        """Estimate isoelectric point (pI)."""
        # Count charged residues
        pos_count = sum(1 for aa in sequence if aa in 'KRH')
        neg_count = sum(1 for aa in sequence if aa in 'DE')
        
        # Simplified estimation
        if pos_count + neg_count == 0:
            return 7.0
            
        # Approximate pI based on charge balance
        charge_ratio = (pos_count - neg_count) / (pos_count + neg_count + 1)
        pI = 7.0 + charge_ratio * 3.0
        
        return max(1.0, min(14.0, pI))
    
    def _calc_gravy(self, sequence: str) -> float:
        """Calculate GRAVY (Grand Average of Hydropathy)."""
        if not sequence:
            return 0.0
        hydro_sum = sum(self.AA_PROPERTIES['hydrophobicity'].get(aa, 0) for aa in sequence)
        return hydro_sum / len(sequence)
    
    def _calc_charge_at_pH7(self, sequence: str) -> float:
        """Calculate net charge at pH 7."""
        return sum(self.AA_PROPERTIES['charge'].get(aa, 0) for aa in sequence)
    
    def _calc_hydrophobic_ratio(self, sequence: str) -> float:
        """Calculate ratio of hydrophobic residues."""
        hydrophobic = 'AILMFVWY'
        count = sum(1 for aa in sequence if aa in hydrophobic)
        return count / len(sequence) if sequence else 0.0
    
    def _calc_positive_ratio(self, sequence: str) -> float:
        """Calculate ratio of positively charged residues."""
        positive = 'KRH'
        count = sum(1 for aa in sequence if aa in positive)
        return count / len(sequence) if sequence else 0.0
    
    def _calc_negative_ratio(self, sequence: str) -> float:
        """Calculate ratio of negatively charged residues."""
        negative = 'DE'
        count = sum(1 for aa in sequence if aa in negative)
        return count / len(sequence) if sequence else 0.0
    
    def _calc_aromatic_ratio(self, sequence: str) -> float:
        """Same as aromaticity."""
        return self._calc_aromaticity(sequence)
    
    def _calc_aliphatic_index(self, sequence: str) -> float:
        """
        Calculate aliphatic index.
        Higher values indicate more thermostable proteins.
        """
        if not sequence:
            return 0.0
            
        n = len(sequence)
        ala = sequence.count('A') / n * 100
        val = sequence.count('V') / n * 100
        ile = sequence.count('I') / n * 100
        leu = sequence.count('L') / n * 100
        
        return ala + 2.9 * val + 3.9 * (ile + leu)
    
    def _calc_boman_index(self, sequence: str) -> float:
        """
        Calculate Boman index (protein-protein interaction potential).
        """
        # Solubility values
        solubility = {
            'A': 0.17, 'C': -0.24, 'D': 1.23, 'E': 2.02, 'F': -1.13,
            'G': 0.01, 'H': 0.96, 'I': -0.31, 'K': 0.99, 'L': -0.56,
            'M': -0.23, 'N': 0.42, 'P': 0.45, 'Q': 0.58, 'R': 0.81,
            'S': 0.13, 'T': 0.14, 'V': 0.07, 'W': -1.85, 'Y': -0.94
        }
        
        if not sequence:
            return 0.0
            
        return sum(solubility.get(aa, 0) for aa in sequence) / len(sequence)
    
    def _calc_hydrophobic_moment(self, sequence: str, angle: float = 100.0) -> float:
        """
        Calculate hydrophobic moment for alpha-helix (100°) or beta-sheet (160°).
        """
        if len(sequence) < 3:
            return 0.0
            
        angle_rad = math.radians(angle)
        
        sin_sum = 0.0
        cos_sum = 0.0
        
        for i, aa in enumerate(sequence):
            h = self.AA_PROPERTIES['hydrophobicity'].get(aa, 0)
            sin_sum += h * math.sin(i * angle_rad)
            cos_sum += h * math.cos(i * angle_rad)
            
        moment = math.sqrt(sin_sum**2 + cos_sum**2) / len(sequence)
        return moment
    
    def _calc_amphipathicity(self, sequence: str) -> float:
        """
        Calculate amphipathicity (related to hydrophobic moment).
        """
        return self._calc_hydrophobic_moment(sequence, angle=100.0)
    
    @property
    def feature_dim(self) -> int:
        """Return number of features."""
        return len(self.feature_names)


def compute_stability_score(sequence: str) -> float:
    """
    Compute a stability score for a peptide sequence.
    Higher scores indicate more stable peptides.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Stability score between 0 and 1
    """
    extractor = PeptideFeatureExtractor()
    features = extractor.extract_dict(sequence)
    
    # Combine multiple stability indicators
    scores = []
    
    # Instability index (< 40 is stable)
    ii = features.get('instability_index', 50)
    ii_score = max(0, 1 - ii / 80)  # Normalize
    scores.append(ii_score)
    
    # Aliphatic index (higher is more stable)
    ai = features.get('aliphatic_index', 0)
    ai_score = min(1, ai / 150)  # Normalize
    scores.append(ai_score)
    
    # GRAVY (moderate values are often better)
    gravy = features.get('gravy', 0)
    gravy_score = 1 - abs(gravy) / 4.5
    scores.append(max(0, gravy_score))
    
    # Charge balance
    pos = features.get('positive_ratio', 0)
    neg = features.get('negative_ratio', 0)
    charge_balance = 1 - abs(pos - neg)
    scores.append(charge_balance)
    
    # Average score
    return sum(scores) / len(scores)
