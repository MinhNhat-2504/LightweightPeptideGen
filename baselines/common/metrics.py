"""
Unified evaluation metrics for all baseline models.
Reports the same metrics as LightweightPeptideGen for fair comparison.
"""

import math
import numpy as np
import torch
from typing import List, Dict, Callable, Optional
from collections import Counter

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from peptidegen.evaluation.metrics import (
    calculate_diversity_metrics,
    estimate_amp_probability,
)
from peptidegen.evaluation.stability import (
    calculate_instability_index,
    calculate_gravy,
    calculate_aliphatic_index,
)

STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')


def compute_token_entropy(sequences: List[str]) -> float:
    """
    Compute Shannon entropy of amino acid distribution (same formula as LightweightPeptideGen).
    Target: > 2.80
    """
    if not sequences:
        return 0.0
    all_aa = ''.join(sequences)
    counts = Counter(all_aa)
    total = len(all_aa)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p + 1e-10)
    return entropy


def compute_ngram_diversity(sequences: List[str], n: int = 2) -> float:
    """
    Compute n-gram diversity as unique_ngrams / total_ngrams.
    Target: > 0.08
    """
    ngrams = []
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            ngrams.append(seq[i:i + n])
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def compute_validity(sequences: List[str]) -> float:
    """Fraction of sequences that only contain standard amino acids."""
    if not sequences:
        return 0.0
    valid = sum(1 for seq in sequences if seq and all(aa in STANDARD_AA for aa in seq))
    return valid / len(sequences)


def compute_uniqueness(sequences: List[str]) -> float:
    """Unique sequences / total sequences."""
    if not sequences:
        return 0.0
    return len(set(sequences)) / len(sequences)


def evaluate_generated_sequences(sequences: List[str]) -> Dict[str, float]:
    """
    Full evaluation suite identical to what LightweightPeptideGen tracks.

    Args:
        sequences: List of generated amino acid strings

    Returns:
        Dict with all metrics
    """
    # Filter to valid sequences
    valid_seqs = [s for s in sequences if s and len(s) >= 5 and all(aa in STANDARD_AA for aa in s)]

    metrics = {}
    metrics['num_generated'] = len(sequences)
    metrics['num_valid'] = len(valid_seqs)
    metrics['validity_ratio'] = len(valid_seqs) / max(len(sequences), 1)

    if not valid_seqs:
        return metrics

    # ── Diversity ────────────────────────────────────────
    metrics['entropy'] = compute_token_entropy(valid_seqs)
    metrics['ngram_diversity_2'] = compute_ngram_diversity(valid_seqs, n=2)
    metrics['ngram_diversity_3'] = compute_ngram_diversity(valid_seqs, n=3)
    metrics['uniqueness_ratio'] = compute_uniqueness(valid_seqs)

    # ── Length ───────────────────────────────────────────
    lengths = [len(s) for s in valid_seqs]
    metrics['mean_length'] = float(np.mean(lengths))
    metrics['std_length'] = float(np.std(lengths))
    metrics['min_length'] = int(min(lengths))
    metrics['max_length'] = int(max(lengths))
    in_range = sum(1 for l in lengths if 10 <= l <= 30)
    metrics['length_in_range_ratio'] = in_range / len(valid_seqs)

    # ── Stability ────────────────────────────────────────
    instability_scores = []
    gravy_scores = []
    
    # Sample for speed if dataset is huge
    stability_sample_size = 10000
    stability_sample = valid_seqs
    if len(valid_seqs) > stability_sample_size:
        indices = np.random.choice(len(valid_seqs), stability_sample_size, replace=False)
        stability_sample = [valid_seqs[i] for i in indices]

    for seq in stability_sample:
        try:
            instability_scores.append(calculate_instability_index(seq))
            gravy_scores.append(calculate_gravy(seq))
        except Exception:
            pass

    if instability_scores:
        metrics['mean_instability_index'] = float(np.mean(instability_scores))
        metrics['stable_ratio'] = sum(1 for s in instability_scores if s < 40) / len(instability_scores)
    if gravy_scores:
        metrics['mean_gravy'] = float(np.mean(gravy_scores))

    # ── AMP metrics ──────────────────────────────────────
    amp_sample_size = 10000
    amp_sample = valid_seqs
    if len(valid_seqs) > amp_sample_size:
        indices = np.random.choice(len(valid_seqs), amp_sample_size, replace=False)
        amp_sample = [valid_seqs[i] for i in indices]

    amp_probs = [estimate_amp_probability(seq) for seq in amp_sample]
    if amp_probs:
        metrics['mean_amp_probability'] = float(np.mean(amp_probs))
        metrics['probable_amp_ratio'] = sum(1 for p in amp_probs if p >= 0.7) / len(amp_probs)

    return metrics


def print_metrics_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Print a comparison table of metrics for multiple models.

    Args:
        results: {model_name: metrics_dict}

    Returns:
        Formatted markdown table string
    """
    # Key metrics to display
    key_metrics = [
        ('entropy', 'Token Entropy', '>2.80'),
        ('ngram_diversity_2', 'Bigram Diversity', '>0.08'),
        ('uniqueness_ratio', 'Uniqueness', '—'),
        ('validity_ratio', 'Validity', '—'),
        ('mean_length', 'Mean Length', '10–30'),
        ('mean_instability_index', 'Mean Instability', '<40'),
        ('stable_ratio', 'Stable Ratio', '—'),
        ('mean_amp_probability', 'AMP Probability', '—'),
        ('probable_amp_ratio', 'AMP Ratio (≥0.7)', '—'),
    ]

    model_names = list(results.keys())

    # Header
    header = '| Metric | Target | ' + ' | '.join(model_names) + ' |'
    sep = '|---|---|' + '|'.join(['---'] * len(model_names)) + '|'
    rows = [header, sep]

    for key, label, target in key_metrics:
        row = f'| **{label}** | {target} |'
        for model in model_names:
            val = results[model].get(key, None)
            if val is None:
                row += ' — |'
            elif isinstance(val, float):
                row += f' {val:.4f} |'
            else:
                row += f' {val} |'
        rows.append(row)

    return '\n'.join(rows)
