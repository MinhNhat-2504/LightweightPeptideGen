"""
Peptide Diversity and Quality Metrics Module.

Provides functions to calculate:
- Diversity metrics (uniqueness, n-gram diversity)
- Length statistics
- Amino acid distribution
- Mode collapse detection
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter


# Standard amino acids
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')


def calculate_amino_acid_distribution(sequences: List[str]) -> Dict[str, Dict]:
    """
    Calculate amino acid distribution across all sequences.
    
    Args:
        sequences: List of sequences
        
    Returns:
        Dictionary mapping AA to count and frequency.
    """
    all_aa = ''.join(sequences)
    aa_counts = Counter(all_aa)
    total = len(all_aa)
    
    distribution = {}
    for aa in STANDARD_AA:
        count = aa_counts.get(aa, 0)
        distribution[aa] = {
            'count': count,
            'frequency': count / total if total > 0 else 0
        }
    
    # Count non-standard
    non_standard = sum(count for aa, count in aa_counts.items() if aa not in STANDARD_AA)
    distribution['non_standard'] = {
        'count': non_standard,
        'frequency': non_standard / total if total > 0 else 0
    }
    
    return distribution


def calculate_diversity_metrics(sequences: List[str]) -> Dict[str, float]:
    """
    Calculate diversity metrics for generated sequences.
    
    Args:
        sequences: List of sequences
        
    Returns:
        Dictionary of diversity metrics.
    """
    if not sequences:
        return {}
    
    # Unique sequences
    unique_seqs = set(sequences)
    uniqueness = len(unique_seqs) / len(sequences)
    
    # Average pairwise similarity (Jaccard)
    if len(sequences) > 1:
        similarities = []
        sample_size = min(100, len(sequences))
        sampled = list(np.random.choice(sequences, sample_size, replace=False))
        
        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                set1, set2 = set(sampled[i]), set(sampled[j])
                if set1 or set2:
                    jaccard = len(set1 & set2) / len(set1 | set2)
                    similarities.append(jaccard)
        
        avg_similarity = float(np.mean(similarities)) if similarities else 0.0
    else:
        avg_similarity = 1.0
    
    # N-gram diversity
    bigrams = []
    trigrams = []
    for seq in sequences:
        for i in range(len(seq) - 1):
            bigrams.append(seq[i:i+2])
        for i in range(len(seq) - 2):
            trigrams.append(seq[i:i+3])
    
    unique_bigrams = len(set(bigrams)) / len(bigrams) if bigrams else 0
    unique_trigrams = len(set(trigrams)) / len(trigrams) if trigrams else 0
    
    return {
        'num_sequences': len(sequences),
        'unique_sequences': len(unique_seqs),
        'uniqueness_ratio': uniqueness,
        'avg_pairwise_similarity': avg_similarity,
        'diversity_score': 1 - avg_similarity,
        'bigram_diversity': unique_bigrams,
        'trigram_diversity': unique_trigrams,
    }


def calculate_length_statistics(sequences: List[str]) -> Dict[str, float]:
    """
    Calculate length statistics for sequences.
    
    Args:
        sequences: List of sequences
        
    Returns:
        Dictionary of length statistics.
    """
    if not sequences:
        return {}
    
    lengths = [len(seq) for seq in sequences]
    
    return {
        'min_length': int(min(lengths)),
        'max_length': int(max(lengths)),
        'mean_length': float(np.mean(lengths)),
        'std_length': float(np.std(lengths)),
        'median_length': float(np.median(lengths)),
        'length_distribution': dict(Counter(lengths)),
    }


def detect_mode_collapse(
    sequences: List[str], 
    entropy_threshold: float = 0.3,
    aa_usage_threshold: float = 0.5
) -> Dict[str, any]:
    """
    Detect potential mode collapse in generated sequences.
    
    Mode collapse indicators:
    - Low amino acid diversity (using only a few AAs)
    - Low sequence entropy
    - High repetition of patterns
    
    Args:
        sequences: List of sequences
        entropy_threshold: Minimum normalized entropy threshold
        aa_usage_threshold: Minimum fraction of AA types used
        
    Returns:
        Dictionary with mode collapse analysis.
    """
    if not sequences:
        return {'mode_collapse': False, 'reason': 'No sequences'}
    
    # Check amino acid usage
    all_aa = ''.join(sequences)
    aa_counts = Counter(all_aa)
    total = len(all_aa)
    
    # Calculate entropy
    frequencies = [count / total for count in aa_counts.values()]
    entropy = -sum(f * np.log2(f + 1e-10) for f in frequencies)
    max_entropy = np.log2(20)  # Maximum entropy with 20 amino acids
    normalized_entropy = entropy / max_entropy
    
    # Count unique AAs used
    unique_aas = len([aa for aa in aa_counts.keys() if aa in STANDARD_AA])
    aa_usage_ratio = unique_aas / 20
    
    # Detect patterns
    mode_collapse = False
    reasons = []
    
    if normalized_entropy < entropy_threshold:
        mode_collapse = True
        reasons.append(f"Low entropy ({normalized_entropy:.2f} < {entropy_threshold})")
    
    if aa_usage_ratio < aa_usage_threshold:
        mode_collapse = True
        reasons.append(f"Low AA usage ({unique_aas}/20 types, {aa_usage_ratio:.1%})")
    
    # Check for dominant amino acids
    sorted_aa = sorted(aa_counts.items(), key=lambda x: x[1], reverse=True)
    top_3_freq = sum(count for _, count in sorted_aa[:3]) / total
    if top_3_freq > 0.7:
        mode_collapse = True
        top_3_aa = ', '.join([aa for aa, _ in sorted_aa[:3]])
        reasons.append(f"Top 3 AAs ({top_3_aa}) dominate with {top_3_freq:.1%}")
    
    return {
        'mode_collapse': mode_collapse,
        'normalized_entropy': float(normalized_entropy),
        'unique_aa_count': unique_aas,
        'aa_usage_ratio': float(aa_usage_ratio),
        'top_3_aa_frequency': float(top_3_freq),
        'top_amino_acids': dict(sorted_aa[:5]),
        'reasons': reasons if mode_collapse else ['No mode collapse detected'],
    }


def compare_distributions(
    generated: List[str], 
    reference: List[str]
) -> Dict[str, float]:
    """
    Compare amino acid distributions between generated and reference sequences.
    
    Args:
        generated: List of generated sequences
        reference: List of reference sequences
        
    Returns:
        Dictionary with comparison metrics.
    """
    gen_dist = calculate_amino_acid_distribution(generated)
    ref_dist = calculate_amino_acid_distribution(reference)
    
    # Calculate KL divergence (simplified)
    kl_div = 0.0
    for aa in STANDARD_AA:
        p = ref_dist[aa]['frequency'] + 1e-10
        q = gen_dist[aa]['frequency'] + 1e-10
        kl_div += p * np.log(p / q)
    
    # Calculate JS divergence
    js_div = 0.0
    for aa in STANDARD_AA:
        p = ref_dist[aa]['frequency'] + 1e-10
        q = gen_dist[aa]['frequency'] + 1e-10
        m = (p + q) / 2
        js_div += 0.5 * p * np.log(p / m) + 0.5 * q * np.log(q / m)
    
    # Calculate absolute differences
    diffs = {}
    total_diff = 0.0
    for aa in STANDARD_AA:
        diff = abs(gen_dist[aa]['frequency'] - ref_dist[aa]['frequency'])
        diffs[aa] = diff
        total_diff += diff
    
    return {
        'kl_divergence': float(kl_div),
        'js_divergence': float(js_div),
        'total_frequency_diff': float(total_diff),
        'mean_frequency_diff': float(total_diff / 20),
        'per_aa_diff': diffs,
        'generated_distribution': {aa: gen_dist[aa]['frequency'] for aa in STANDARD_AA},
        'reference_distribution': {aa: ref_dist[aa]['frequency'] for aa in STANDARD_AA},
    }


def calculate_novelty(
    generated: List[str], 
    training: List[str]
) -> Dict[str, float]:
    """
    Calculate novelty of generated sequences vs training set.
    
    Args:
        generated: List of generated sequences
        training: List of training sequences
        
    Returns:
        Dictionary with novelty metrics.
    """
    training_set = set(training)
    
    novel = [seq for seq in generated if seq not in training_set]
    
    return {
        'num_generated': len(generated),
        'num_novel': len(novel),
        'novelty_ratio': len(novel) / len(generated) if generated else 0,
        'num_duplicates': len(generated) - len(novel),
    }


# =============================================================
# AMP-specific Metrics (merged from amp_metrics.py)
# =============================================================

# Eisenberg hydrophobicity scale (from HydrAMP)
EISENBERG_SCALE = {
    'A': 0.620, 'R': -2.530, 'N': -0.780, 'D': -0.900, 'C': 0.290,
    'Q': -0.850, 'E': -0.740, 'G': 0.480, 'H': -0.400, 'I': 1.380,
    'L': 1.060, 'K': -1.500, 'M': 0.640, 'F': 1.190, 'P': 0.120,
    'S': -0.180, 'T': -0.050, 'W': 0.810, 'Y': 0.260, 'V': 1.080,
}

# Hydrophobic moment angle (alpha-helix = 100°)
HELIX_ANGLE = 100 * math.pi / 180

# Amino acid properties for AMP scoring
POSITIVE_CHARGED = {'K', 'R', 'H'}
NEGATIVE_CHARGED = {'D', 'E'}
HYDROPHOBIC = {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'Y'}
AROMATIC = {'F', 'W', 'Y', 'H'}

# Hemolytic propensity scores (empirical, based on ESM2-GAT)
# Higher values indicate higher hemolytic activity
HEMOLYTIC_PROPENSITY = {
    'A': 0.2, 'R': 0.6, 'N': 0.1, 'D': -0.2, 'C': 0.3,
    'Q': 0.1, 'E': -0.2, 'G': 0.0, 'H': 0.3, 'I': 0.5,
    'L': 0.7, 'K': 0.5, 'M': 0.4, 'F': 0.8, 'P': 0.1,
    'S': 0.0, 'T': 0.1, 'W': 0.9, 'Y': 0.4, 'V': 0.4,
}


def calculate_hydrophobicity(sequence: str) -> float:
    """
    Calculate mean hydrophobicity using Eisenberg scale.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Mean hydrophobicity value
    """
    if not sequence:
        return 0.0
    
    values = [EISENBERG_SCALE.get(aa, 0) for aa in sequence.upper()]
    return sum(values) / len(values) if values else 0.0


def calculate_hydrophobic_moment(sequence: str, angle: float = HELIX_ANGLE, 
                                  window: int = 11) -> float:
    """
    Calculate hydrophobic moment (amphipathicity measure).
    
    Higher values indicate more amphipathic character,
    which is often associated with AMP activity.
    
    Args:
        sequence: Amino acid sequence
        angle: Angle between residues (100° for alpha-helix)
        window: Window size for calculation
        
    Returns:
        Maximum hydrophobic moment value
    """
    if len(sequence) < window:
        window = len(sequence)
    
    if window < 3:
        return 0.0
    
    max_moment = 0.0
    sequence = sequence.upper()
    
    for i in range(len(sequence) - window + 1):
        window_seq = sequence[i:i+window]
        
        sum_sin = 0.0
        sum_cos = 0.0
        
        for j, aa in enumerate(window_seq):
            h = EISENBERG_SCALE.get(aa, 0)
            sum_sin += h * math.sin(j * angle)
            sum_cos += h * math.cos(j * angle)
        
        moment = math.sqrt(sum_sin**2 + sum_cos**2) / window
        max_moment = max(max_moment, moment)
    
    return max_moment


def calculate_net_charge(sequence: str, pH: float = 7.0) -> float:
    """
    Calculate net charge at given pH.
    
    Args:
        sequence: Amino acid sequence
        pH: pH value (default 7.0)
        
    Returns:
        Net charge value
    """
    # pKa values
    pKa = {
        'K': 10.5, 'R': 12.5, 'H': 6.0,  # Basic
        'D': 3.9, 'E': 4.1,               # Acidic
        'C': 8.3, 'Y': 10.1,              # Slightly acidic
    }
    
    charge = 0.0
    sequence = sequence.upper()
    
    # N-terminus (pKa ~9.7)
    charge += 1 / (1 + 10**(pH - 9.7))
    
    # C-terminus (pKa ~2.3)
    charge -= 1 / (1 + 10**(2.3 - pH))
    
    for aa in sequence:
        if aa in ['K', 'R', 'H']:
            charge += 1 / (1 + 10**(pH - pKa[aa]))
        elif aa in ['D', 'E']:
            charge -= 1 / (1 + 10**(pKa[aa] - pH))
        elif aa in ['C', 'Y']:
            charge -= 1 / (1 + 10**(pKa[aa] - pH))
    
    return charge


def calculate_hemolytic_score(sequence: str) -> Tuple[float, str]:
    """
    Estimate hemolytic activity score.
    
    Based on empirical propensity scales and structural features
    associated with hemolytic activity.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Tuple of (score, risk_category)
        - score: 0-10 scale (lower is safer)
        - risk_category: 'low', 'medium', 'high'
    """
    if not sequence or len(sequence) < 5:
        return 0.0, 'unknown'
    
    sequence = sequence.upper()
    n = len(sequence)
    
    # Component scores
    scores = []
    
    # 1. Hemolytic propensity (0-10)
    propensity_sum = sum(HEMOLYTIC_PROPENSITY.get(aa, 0) for aa in sequence)
    propensity_score = (propensity_sum / n + 0.2) * 5  # Normalize to ~0-10
    propensity_score = max(0, min(10, propensity_score))
    scores.append(propensity_score * 0.3)
    
    # 2. Hydrophobicity contribution (high hydrophobicity = higher hemolytic)
    hydro = calculate_hydrophobicity(sequence)
    hydro_score = (hydro + 1) * 3  # Scale to 0-10
    hydro_score = max(0, min(10, hydro_score))
    scores.append(hydro_score * 0.25)
    
    # 3. Hydrophobic moment (high amphipathicity can increase hemolysis)
    moment = calculate_hydrophobic_moment(sequence)
    moment_score = moment * 10
    moment_score = max(0, min(10, moment_score))
    scores.append(moment_score * 0.2)
    
    # 4. Positive charge (moderately positive = better, very high = hemolytic)
    charge = calculate_net_charge(sequence)
    if charge < 0:
        charge_score = 3.0  # Negative charge less hemolytic
    elif charge < 5:
        charge_score = 2.0 + charge * 0.5  # Moderate positive
    else:
        charge_score = 4.5 + (charge - 5) * 0.5  # High positive more hemolytic
    charge_score = max(0, min(10, charge_score))
    scores.append(charge_score * 0.15)
    
    # 5. Aromatic content (high aromatic = higher hemolytic)
    aromatic_ratio = sum(1 for aa in sequence if aa in AROMATIC) / n
    aromatic_score = aromatic_ratio * 15
    aromatic_score = max(0, min(10, aromatic_score))
    scores.append(aromatic_score * 0.1)
    
    # Total score
    total_score = sum(scores)
    total_score = max(0, min(10, total_score))
    
    # Categorize
    if total_score < 2.5:
        category = 'low'
    elif total_score < 5.0:
        category = 'medium'
    else:
        category = 'high'
    
    return round(total_score, 2), category


def calculate_therapeutic_score(sequence: str) -> Tuple[float, str]:
    """
    Estimate therapeutic potential score.
    
    Higher score indicates better therapeutic potential
    (good AMP activity with low toxicity).
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Tuple of (score, category)
        - score: 0-10 scale (higher is better)
        - category: 'low', 'medium', 'high'
    """
    if not sequence or len(sequence) < 5:
        return 0.0, 'unknown'
    
    sequence = sequence.upper()
    n = len(sequence)
    
    scores = []
    
    # 1. Optimal length (10-30 AA is typically best)
    if 10 <= n <= 30:
        length_score = 10.0
    elif 5 <= n < 10:
        length_score = 5.0 + (n - 5)
    elif 30 < n <= 50:
        length_score = 10.0 - (n - 30) * 0.3
    else:
        length_score = 3.0
    scores.append(length_score * 0.15)
    
    # 2. Positive charge (2-7 is optimal for AMPs)
    charge = calculate_net_charge(sequence)
    if 2 <= charge <= 7:
        charge_score = 10.0
    elif 0 <= charge < 2:
        charge_score = 5.0 + charge * 2.5
    elif 7 < charge <= 10:
        charge_score = 10.0 - (charge - 7) * 1.5
    else:
        charge_score = 3.0
    scores.append(charge_score * 0.2)
    
    # 3. Amphipathicity (moderate is good)
    moment = calculate_hydrophobic_moment(sequence)
    if 0.3 <= moment <= 0.6:
        moment_score = 10.0
    elif moment < 0.3:
        moment_score = moment / 0.3 * 10
    else:
        moment_score = max(5, 10 - (moment - 0.6) * 10)
    scores.append(moment_score * 0.2)
    
    # 4. Hydrophobicity (slightly hydrophobic is good, too much is bad)
    hydro = calculate_hydrophobicity(sequence)
    if -0.2 <= hydro <= 0.3:
        hydro_score = 10.0
    elif hydro < -0.2:
        hydro_score = max(3, 10 + (hydro + 0.2) * 10)
    else:
        hydro_score = max(3, 10 - (hydro - 0.3) * 5)
    scores.append(hydro_score * 0.2)
    
    # 5. Positive to negative charge ratio
    pos_count = sum(1 for aa in sequence if aa in POSITIVE_CHARGED)
    neg_count = sum(1 for aa in sequence if aa in NEGATIVE_CHARGED)
    ratio = pos_count / (neg_count + 1)
    if 2 <= ratio <= 5:
        ratio_score = 10.0
    elif ratio < 2:
        ratio_score = ratio * 5
    else:
        ratio_score = max(5, 10 - (ratio - 5) * 0.5)
    scores.append(ratio_score * 0.15)
    
    # 6. Low hemolytic potential (inverse of hemolytic score)
    hemo_score, _ = calculate_hemolytic_score(sequence)
    safety_score = 10 - hemo_score
    scores.append(safety_score * 0.1)
    
    # Total score
    total_score = sum(scores)
    total_score = max(0, min(10, total_score))
    
    # Categorize
    if total_score >= 7.0:
        category = 'high'
    elif total_score >= 4.5:
        category = 'medium'
    else:
        category = 'low'
    
    return round(total_score, 2), category


def estimate_amp_probability(sequence: str) -> float:
    """
    Estimate probability of being an antimicrobial peptide.
    
    Based on physicochemical properties commonly found in AMPs.
    This is a simplified heuristic - for better accuracy,
    use a trained classifier.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Probability score (0-1)
    """
    if not sequence or len(sequence) < 5:
        return 0.0
    
    sequence = sequence.upper()
    n = len(sequence)
    
    scores = []
    
    # 1. Length (AMPs typically 10-50 AA)
    if 10 <= n <= 50:
        scores.append(1.0)
    elif 5 <= n < 10 or 50 < n <= 100:
        scores.append(0.5)
    else:
        scores.append(0.2)
    
    # 2. Positive charge (AMPs are typically cationic)
    charge = calculate_net_charge(sequence)
    if charge >= 2:
        scores.append(min(1.0, 0.5 + charge * 0.1))
    else:
        scores.append(max(0.2, 0.5 + charge * 0.1))
    
    # 3. Hydrophobic content (30-60% is typical)
    hydro_ratio = sum(1 for aa in sequence if aa in HYDROPHOBIC) / n
    if 0.3 <= hydro_ratio <= 0.6:
        scores.append(1.0)
    else:
        scores.append(0.5)
    
    # 4. Amphipathicity
    moment = calculate_hydrophobic_moment(sequence)
    if moment >= 0.3:
        scores.append(min(1.0, 0.5 + moment))
    else:
        scores.append(0.5)
    
    # 5. Contains key AMP residues (K, R, L, W)
    key_residues = sum(1 for aa in sequence if aa in {'K', 'R', 'L', 'W'}) / n
    if key_residues >= 0.2:
        scores.append(1.0)
    else:
        scores.append(0.5 + key_residues)
    
    # Weighted average
    weights = [0.15, 0.25, 0.2, 0.2, 0.2]
    probability = sum(s * w for s, w in zip(scores, weights))
    
    return round(min(1.0, max(0.0, probability)), 3)


def analyze_amp_properties(sequences: List[str]) -> Dict:
    """
    Comprehensive AMP property analysis for a batch of sequences.
    
    Args:
        sequences: List of amino acid sequences
        
    Returns:
        Dictionary with AMP-specific metrics
    """
    if not sequences:
        return {}
    
    results = {
        'hemolytic': {'scores': [], 'categories': Counter()},
        'therapeutic': {'scores': [], 'categories': Counter()},
        'amp_probability': [],
        'hydrophobicity': [],
        'hydrophobic_moment': [],
        'net_charge': [],
        'valid_count': 0,
    }
    
    for seq in sequences:
        if not seq or len(seq) < 5:
            continue
        
        results['valid_count'] += 1
        
        # Hemolytic
        hemo_score, hemo_cat = calculate_hemolytic_score(seq)
        results['hemolytic']['scores'].append(hemo_score)
        results['hemolytic']['categories'][hemo_cat] += 1
        
        # Therapeutic
        ther_score, ther_cat = calculate_therapeutic_score(seq)
        results['therapeutic']['scores'].append(ther_score)
        results['therapeutic']['categories'][ther_cat] += 1
        
        # AMP probability
        amp_prob = estimate_amp_probability(seq)
        results['amp_probability'].append(amp_prob)
        
        # Basic properties
        results['hydrophobicity'].append(calculate_hydrophobicity(seq))
        results['hydrophobic_moment'].append(calculate_hydrophobic_moment(seq))
        results['net_charge'].append(calculate_net_charge(seq))
    
    # Calculate summaries
    def summarize(values):
        if not values:
            return {}
        arr = np.array(values)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'median': float(np.median(arr)),
        }
    
    results['hemolytic']['summary'] = summarize(results['hemolytic']['scores'])
    results['therapeutic']['summary'] = summarize(results['therapeutic']['scores'])
    results['amp_probability_summary'] = summarize(results['amp_probability'])
    results['hydrophobicity_summary'] = summarize(results['hydrophobicity'])
    results['hydrophobic_moment_summary'] = summarize(results['hydrophobic_moment'])
    results['net_charge_summary'] = summarize(results['net_charge'])
    
    # Calculate rates
    n = results['valid_count']
    if n > 0:
        results['low_hemolytic_rate'] = results['hemolytic']['categories'].get('low', 0) / n * 100
        results['high_therapeutic_rate'] = results['therapeutic']['categories'].get('high', 0) / n * 100
        results['probable_amp_rate'] = sum(1 for p in results['amp_probability'] if p >= 0.7) / n * 100
    
    return results
