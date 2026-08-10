"""
Quality Filtering for Generated Peptides.

Filters peptides based on:
1. Stability: instability_index < 40
2. Therapeutic potential: therapeutic_score > threshold
3. Safety: hemolytic_score < threshold
4. Structural properties: aliphatic_index, hydrophobic_moment, etc.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import json

from ..constants import (
    HYDROPATHY_SCALE, INSTABILITY_WEIGHTS, AA_GROUPS,
    QUALITY_THRESHOLDS, PKA_VALUES
)


@dataclass
class QualityCriteria:
    """Quality criteria for peptide filtering."""
    # Stability
    max_instability_index: float = 40.0
    min_aliphatic_index: float = 50.0
    
    # Therapeutic potential
    min_therapeutic_score: float = 0.3
    
    # Safety
    max_hemolytic_score: float = 0.5
    
    # Sequence properties
    min_length: int = 10
    max_length: int = 50
    min_unique_aa: int = 5  # At least 5 different amino acids
    
    # Hydrophobicity
    min_gravy: float = -2.0
    max_gravy: float = 1.0
    
    # Charge
    min_charge: float = -5.0
    max_charge: float = 10.0
    
    # Optional: AMP-like properties
    require_amp_like: bool = False
    min_positive_ratio: float = 0.1  # At least 10% positive AAs for AMP
    

@dataclass
class QualityScore:
    """Quality assessment result for a single peptide."""
    sequence: str
    passes_filter: bool
    overall_score: float  # 0-100
    
    # Individual scores
    stability_score: float = 0.0
    therapeutic_score: float = 0.0
    safety_score: float = 0.0
    diversity_score: float = 0.0
    
    # Feature values
    features: Dict[str, float] = field(default_factory=dict)
    
    # Failure reasons
    failure_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'sequence': self.sequence,
            'passes_filter': self.passes_filter,
            'overall_score': self.overall_score,
            'stability_score': self.stability_score,
            'therapeutic_score': self.therapeutic_score,
            'safety_score': self.safety_score,
            'diversity_score': self.diversity_score,
            'features': self.features,
            'failure_reasons': self.failure_reasons,
        }


class PeptideQualityFilter:
    """
    Filter and score generated peptides based on quality criteria.
    Uses centralized constants from peptidegen.constants module.
    """
    
    def __init__(
        self,
        criteria: Optional[QualityCriteria] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize quality filter.
        
        Args:
            criteria: Quality criteria for filtering
            weights: Weights for overall score calculation
        """
        self.criteria = criteria or QualityCriteria()
        self.weights = weights or {
            'stability': 0.3,
            'therapeutic': 0.3,
            'safety': 0.25,
            'diversity': 0.15,
        }
        
    def compute_features(self, sequence: str) -> Dict[str, float]:
        """
        Compute all features for a peptide sequence.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary of feature values
        """
        sequence = sequence.upper()
        n = len(sequence)
        
        if n == 0:
            return {}
        
        features = {}
        
        # Length
        features['length'] = n
        
        # Amino acid counts
        aa_counts = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        # Unique amino acids
        features['unique_aa'] = len(set(sequence))
        
        # GRAVY (hydropathy) - using centralized constants
        gravy_sum = sum(HYDROPATHY_SCALE.get(aa, 0) for aa in sequence)
        features['gravy'] = gravy_sum / n
        
        # Charge ratios - using centralized AA_GROUPS
        positive_count = sum(1 for aa in sequence if aa in AA_GROUPS['positive'])
        negative_count = sum(1 for aa in sequence if aa in AA_GROUPS['negative'])
        features['positive_ratio'] = positive_count / n
        features['negative_ratio'] = negative_count / n
        # Charge at pH 7 - using Henderson-Hasselbalch approximation
        pH = 7.0
        charge = 0.0
        # N-terminal
        charge += 1.0 / (1.0 + 10 ** (pH - PKA_VALUES['N_TERM']))
        # C-terminal
        charge += -1.0 / (1.0 + 10 ** (PKA_VALUES['C_TERM'] - pH))
        # Side chains
        for aa in sequence:
            if aa in PKA_VALUES['SIDE_CHAINS']:
                pka = PKA_VALUES['SIDE_CHAINS'][aa]
                if aa in ('D', 'E', 'C', 'Y'):  # Acidic side chains
                    charge += -1.0 / (1.0 + 10 ** (pka - pH))
                else:  # Basic side chains (K, R, H)
                    charge += 1.0 / (1.0 + 10 ** (pH - pka))
        features['charge_at_pH7'] = charge
        
        # Aliphatic index
        ala = aa_counts.get('A', 0) / n * 100
        val = aa_counts.get('V', 0) / n * 100
        ile = aa_counts.get('I', 0) / n * 100
        leu = aa_counts.get('L', 0) / n * 100
        features['aliphatic_index'] = ala + 2.9 * val + 3.9 * (ile + leu)
        
        # Aromaticity - using centralized AA_GROUPS
        aromatic_count = sum(1 for aa in sequence if aa in AA_GROUPS['aromatic'])
        features['aromaticity'] = aromatic_count / n
        
        # Instability index (simplified)
        features['instability_index'] = self._calc_instability_index(sequence)
        
        # Hydrophobic ratio
        hydrophobic = AA_GROUPS['hydrophobic']
        features['hydrophobic_ratio'] = sum(1 for aa in sequence if aa in hydrophobic) / n
        
        # Estimated scores (simplified - real calculation would be more complex)
        # These are placeholder calculations
        features['therapeutic_score'] = self._estimate_therapeutic_score(features)
        features['hemolytic_score'] = self._estimate_hemolytic_score(features)
        
        return features
    
    def _calc_instability_index(self, sequence: str) -> float:
        """Calculate instability index using centralized INSTABILITY_WEIGHTS."""
        if len(sequence) < 2:
            return 0.0
        
        score = 0.0
        for i in range(len(sequence) - 1):
            dipeptide = sequence[i:i+2]
            # Use a default weight of 1.0 for unknown dipeptides
            score += INSTABILITY_WEIGHTS.get(dipeptide, 1.0)
        
        return (10.0 / len(sequence)) * score
    
    def _estimate_therapeutic_score(self, features: Dict[str, float]) -> float:
        """
        Estimate therapeutic score based on AMP-like properties.
        Higher for peptides with AMP-like characteristics.
        """
        score = 0.0
        
        # Positive charge is good for AMPs
        if features.get('positive_ratio', 0) > 0.15:
            score += 0.3
        
        # Moderate hydrophobicity
        gravy = features.get('gravy', 0)
        if -0.5 < gravy < 0.5:
            score += 0.3
        
        # Some aromaticity
        if features.get('aromaticity', 0) > 0.05:
            score += 0.2
        
        # Good stability
        if features.get('instability_index', 100) < 40:
            score += 0.2
        
        return min(score, 1.0)
    
    def _estimate_hemolytic_score(self, features: Dict[str, float]) -> float:
        """
        Estimate hemolytic (toxicity) score.
        Lower is better (less toxic).
        """
        score = 0.0
        
        # High hydrophobicity increases hemolysis
        gravy = features.get('gravy', 0)
        if gravy > 0.5:
            score += 0.3
        elif gravy > 0:
            score += 0.1
        
        # Very high positive charge can increase hemolysis
        if features.get('positive_ratio', 0) > 0.3:
            score += 0.2
        
        # High aromaticity can increase membrane interaction
        if features.get('aromaticity', 0) > 0.2:
            score += 0.2
        
        # Length factor
        length = features.get('length', 0)
        if length > 30:
            score += 0.1
        
        return min(score, 1.0)
    
    def evaluate_peptide(self, sequence: str) -> QualityScore:
        """
        Evaluate a single peptide and return quality score.
        
        Args:
            sequence: Peptide sequence
            
        Returns:
            QualityScore object
        """
        sequence = ''.join(aa for aa in sequence.upper() if aa in 'ACDEFGHIKLMNPQRSTVWY')
        
        if len(sequence) < self.criteria.min_length:
            return QualityScore(
                sequence=sequence,
                passes_filter=False,
                overall_score=0.0,
                failure_reasons=[f"Too short: {len(sequence)} < {self.criteria.min_length}"]
            )
        
        # Compute features
        features = self.compute_features(sequence)
        
        failure_reasons = []
        scores = {}
        
        # Check stability
        ii = features.get('instability_index', 100)
        ai = features.get('aliphatic_index', 0)
        
        if ii <= self.criteria.max_instability_index:
            scores['stability'] = 1.0 - (ii / 100)  # Normalize
        else:
            scores['stability'] = max(0, 1.0 - (ii / 200))
            failure_reasons.append(f"Unstable: II={ii:.1f} > {self.criteria.max_instability_index}")
        
        if ai < self.criteria.min_aliphatic_index:
            scores['stability'] *= 0.8
            failure_reasons.append(f"Low aliphatic index: {ai:.1f} < {self.criteria.min_aliphatic_index}")
        
        # Check therapeutic score
        ts = features.get('therapeutic_score', 0)
        if ts >= self.criteria.min_therapeutic_score:
            scores['therapeutic'] = ts
        else:
            scores['therapeutic'] = ts * 0.5
            failure_reasons.append(f"Low therapeutic: {ts:.2f} < {self.criteria.min_therapeutic_score}")
        
        # Check safety (hemolytic score)
        hs = features.get('hemolytic_score', 1)
        if hs <= self.criteria.max_hemolytic_score:
            scores['safety'] = 1.0 - hs
        else:
            scores['safety'] = max(0, 0.5 - hs)
            failure_reasons.append(f"Toxic: HS={hs:.2f} > {self.criteria.max_hemolytic_score}")
        
        # Check diversity
        unique_aa = features.get('unique_aa', 0)
        if unique_aa >= self.criteria.min_unique_aa:
            scores['diversity'] = min(unique_aa / 15, 1.0)  # Normalize to max 15 unique
        else:
            scores['diversity'] = unique_aa / self.criteria.min_unique_aa * 0.5
            failure_reasons.append(f"Low diversity: {unique_aa} unique AAs < {self.criteria.min_unique_aa}")
        
        # Check GRAVY range
        gravy = features.get('gravy', 0)
        if not (self.criteria.min_gravy <= gravy <= self.criteria.max_gravy):
            failure_reasons.append(f"GRAVY out of range: {gravy:.2f}")
        
        # Check charge range
        charge = features.get('charge_at_pH7', 0)
        if not (self.criteria.min_charge <= charge <= self.criteria.max_charge):
            failure_reasons.append(f"Charge out of range: {charge:.1f}")
        
        # Check AMP-like properties if required
        if self.criteria.require_amp_like:
            if features.get('positive_ratio', 0) < self.criteria.min_positive_ratio:
                failure_reasons.append(f"Not AMP-like: low positive charge")
        
        # Calculate overall score
        overall_score = sum(
            scores.get(k, 0) * self.weights.get(k, 0) 
            for k in ['stability', 'therapeutic', 'safety', 'diversity']
        ) * 100
        
        passes_filter = len(failure_reasons) == 0
        
        return QualityScore(
            sequence=sequence,
            passes_filter=passes_filter,
            overall_score=overall_score,
            stability_score=scores.get('stability', 0) * 100,
            therapeutic_score=scores.get('therapeutic', 0) * 100,
            safety_score=scores.get('safety', 0) * 100,
            diversity_score=scores.get('diversity', 0) * 100,
            features=features,
            failure_reasons=failure_reasons,
        )
    
    def filter_peptides(
        self,
        sequences: List[str],
        return_all: bool = False,
    ) -> Tuple[List[QualityScore], Dict[str, any]]:
        """
        Filter and score multiple peptides.
        
        Args:
            sequences: List of peptide sequences
            return_all: If True, return all peptides with scores, not just passing ones
            
        Returns:
            Tuple of (filtered_scores, statistics)
        """
        all_scores = [self.evaluate_peptide(seq) for seq in sequences]
        
        passing = [s for s in all_scores if s.passes_filter]
        failing = [s for s in all_scores if not s.passes_filter]
        
        # Compute statistics
        stats = {
            'total': len(sequences),
            'passing': len(passing),
            'failing': len(failing),
            'pass_rate': len(passing) / len(sequences) * 100 if sequences else 0,
        }
        
        if passing:
            stats['avg_score_passing'] = np.mean([s.overall_score for s in passing])
            stats['avg_stability'] = np.mean([s.stability_score for s in passing])
            stats['avg_therapeutic'] = np.mean([s.therapeutic_score for s in passing])
            stats['avg_safety'] = np.mean([s.safety_score for s in passing])
        
        if failing:
            # Count failure reasons
            reason_counts = {}
            for s in failing:
                for reason in s.failure_reasons:
                    key = reason.split(':')[0]
                    reason_counts[key] = reason_counts.get(key, 0) + 1
            stats['failure_reasons'] = reason_counts
        
        if return_all:
            return all_scores, stats
        else:
            return passing, stats
    
    def rank_peptides(
        self,
        sequences: List[str],
        top_k: int = 10,
    ) -> List[QualityScore]:
        """
        Rank peptides by quality score and return top-k.
        
        Args:
            sequences: List of peptide sequences
            top_k: Number of top peptides to return
            
        Returns:
            List of top-k QualityScore objects sorted by score
        """
        all_scores, _ = self.filter_peptides(sequences, return_all=True)
        sorted_scores = sorted(all_scores, key=lambda x: x.overall_score, reverse=True)
        return sorted_scores[:top_k]


def filter_generated_fasta(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    criteria: Optional[QualityCriteria] = None,
    report_path: Optional[Union[str, Path]] = None,
) -> Dict[str, any]:
    """
    Filter a FASTA file of generated peptides and save passing ones.
    
    Args:
        input_path: Input FASTA file path
        output_path: Output FASTA file path for passing peptides
        criteria: Quality criteria
        report_path: Optional path to save detailed report
        
    Returns:
        Statistics dictionary
    """
    # Parse input FASTA
    sequences = []
    ids = []
    
    with open(input_path, 'r') as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences.append(''.join(current_seq))
                    ids.append(current_id)
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            sequences.append(''.join(current_seq))
            ids.append(current_id)
    
    # Filter
    quality_filter = PeptideQualityFilter(criteria)
    all_scores, stats = quality_filter.filter_peptides(sequences, return_all=True)
    
    # Write passing peptides
    with open(output_path, 'w') as f:
        for i, score in enumerate(all_scores):
            if score.passes_filter:
                f.write(f">{ids[i]}_score{score.overall_score:.1f}\n")
                f.write(f"{score.sequence}\n")
    
    # Save detailed report
    if report_path:
        report = {
            'statistics': stats,
            'criteria': {
                'max_instability_index': criteria.max_instability_index if criteria else 40.0,
                'min_therapeutic_score': criteria.min_therapeutic_score if criteria else 0.3,
                'max_hemolytic_score': criteria.max_hemolytic_score if criteria else 0.5,
            },
            'peptides': [s.to_dict() for s in all_scores],
        }
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("QUALITY FILTERING REPORT")
    print(f"{'='*60}")
    print(f"Total peptides: {stats['total']}")
    print(f"Passing: {stats['passing']} ({stats['pass_rate']:.1f}%)")
    print(f"Failing: {stats['failing']}")
    
    if 'avg_score_passing' in stats:
        print(f"\nPassing peptides average scores:")
        print(f"  Overall: {stats['avg_score_passing']:.1f}")
        print(f"  Stability: {stats['avg_stability']:.1f}")
        print(f"  Therapeutic: {stats['avg_therapeutic']:.1f}")
        print(f"  Safety: {stats['avg_safety']:.1f}")
    
    if 'failure_reasons' in stats:
        print(f"\nFailure reasons:")
        for reason, count in sorted(stats['failure_reasons'].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    
    print(f"\nFiltered peptides saved to: {output_path}")
    
    return stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter generated peptides by quality')
    parser.add_argument('--input', '-i', required=True, help='Input FASTA file')
    parser.add_argument('--output', '-o', required=True, help='Output FASTA file')
    parser.add_argument('--report', '-r', default=None, help='Report JSON file')
    parser.add_argument('--max-ii', type=float, default=40.0, help='Max instability index')
    parser.add_argument('--min-ts', type=float, default=0.3, help='Min therapeutic score')
    parser.add_argument('--max-hs', type=float, default=0.5, help='Max hemolytic score')
    
    args = parser.parse_args()
    
    criteria = QualityCriteria(
        max_instability_index=args.max_ii,
        min_therapeutic_score=args.min_ts,
        max_hemolytic_score=args.max_hs,
    )
    
    filter_generated_fasta(
        input_path=args.input,
        output_path=args.output,
        criteria=criteria,
        report_path=args.report,
    )
