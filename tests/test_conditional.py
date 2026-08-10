#!/usr/bin/env python
"""Test the conditional dataset and quality filter."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from peptidegen.data.dataset import ConditionalPeptideDataset
from peptidegen.data.vocabulary import VOCAB
from peptidegen.evaluation.quality_filter import PeptideQualityFilter, QualityCriteria


def test_conditional_dataset():
    """Test ConditionalPeptideDataset."""
    print("=" * 60)
    print("TESTING CONDITIONAL DATASET")
    print("=" * 60)

    # Test ConditionalPeptideDataset
    print("\n1. Loading ConditionalPeptideDataset from CSV...")
    dataset = ConditionalPeptideDataset.from_csv(
        'dataset/train.csv',
        vocab=VOCAB,
        max_length=50,
        min_length=5,
    )

    print(f"   Dataset size: {len(dataset)}")
    print(f"   Condition dim: {dataset.get_condition_dim()}")

    # Test a sample
    sample = dataset[0]
    print(f"\n2. Sample data:")
    print(f"   Keys: {list(sample.keys())}")
    print(f"   Sequence: {sample['sequence'][:30]}...")
    print(f"   Condition shape: {sample['condition'].shape}")
    print(f"   Condition values: {sample['condition']}")
    print(f"   Features raw: {sample['features_raw']}")

    # Feature names
    print(f"\n3. Feature names used:")
    for i, name in enumerate(dataset.feature_names):
        print(f"   {i+1}. {name}")

    # Feature stats
    print(f"\n4. Feature statistics (from data):")
    stats = dataset.get_feature_stats()
    for name, stat in stats.items():
        if stat:
            print(f"   {name}: mean={stat['mean']:.2f}, std={stat['std']:.2f}")

    print("\n✅ ConditionalPeptideDataset test PASSED")
    return True


def test_quality_filter():
    """Test PeptideQualityFilter."""
    print("\n" + "=" * 60)
    print("TESTING QUALITY FILTER")
    print("=" * 60)

    # Load dataset for test sequences
    dataset = ConditionalPeptideDataset.from_csv(
        'dataset/train.csv',
        vocab=VOCAB,
        max_length=50,
        min_length=5,
    )

    # Test Quality Filter
    print("\n5. Testing PeptideQualityFilter...")
    criteria = QualityCriteria(
        max_instability_index=40.0,
        min_therapeutic_score=0.3,
        max_hemolytic_score=0.5,
    )

    quality_filter = PeptideQualityFilter(criteria)

    # Test with some sequences from dataset
    test_seqs = [dataset[i]['sequence'] for i in range(min(10, len(dataset)))]
    print(f"   Testing {len(test_seqs)} sequences...")

    all_scores, stats = quality_filter.filter_peptides(test_seqs, return_all=True)

    print(f"\n6. Filter results:")
    print(f"   Total: {stats['total']}")
    print(f"   Passing: {stats['passing']} ({stats['pass_rate']:.1f}%)")
    print(f"   Failing: {stats['failing']}")

    if stats['passing'] > 0:
        print(f"\n7. Average scores for passing peptides:")
        print(f"   Overall: {stats['avg_score_passing']:.1f}")
        print(f"   Stability: {stats['avg_stability']:.1f}")
        print(f"   Therapeutic: {stats['avg_therapeutic']:.1f}")
        print(f"   Safety: {stats['avg_safety']:.1f}")

    print("\n8. Sample peptide evaluation:")
    score = all_scores[0]
    print(f"   Sequence: {score.sequence[:30]}...")
    print(f"   Passes filter: {score.passes_filter}")
    print(f"   Overall score: {score.overall_score:.1f}")
    print(f"   Features:")
    for k, v in list(score.features.items())[:5]:
        print(f"     {k}: {v:.3f}")
    if score.failure_reasons:
        print(f"   Failure reasons: {score.failure_reasons}")

    print("\n✅ Quality Filter test PASSED")
    return True


if __name__ == '__main__':
    test_conditional_dataset()
    test_quality_filter()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
