#!/usr/bin/env python
"""Quick tests for all main components"""

import torch
import sys
sys.path.insert(0, '.')

def test_imports():
    """Test all imports work"""
    print("Testing imports...")
    from peptidegen.models.generator import GRUGenerator, LSTMGenerator, TransformerGenerator
    from peptidegen.models.discriminator import CNNDiscriminator
    from peptidegen.data.dataset import ConditionalPeptideDataset
    from peptidegen.data.vocabulary import VOCAB
    from peptidegen.training.trainer import GANTrainer
    from peptidegen.inference.sampler import PeptideSampler
    from peptidegen.evaluation.metrics import calculate_diversity_metrics, analyze_amp_properties
    print("✓ All imports OK")

def test_vocabulary():
    """Test vocabulary"""
    print("\nTesting vocabulary...")
    from peptidegen.data.vocabulary import VOCAB
    
    assert VOCAB.pad_idx == 0
    assert VOCAB.sos_idx == 1
    assert VOCAB.eos_idx == 2
    assert VOCAB.vocab_size == 24  # 4 special + 20 AA
    
    seq = "ACDEFG"
    encoded = VOCAB.encode(seq, add_special_tokens=True)
    decoded = VOCAB.decode(encoded)
    assert seq in decoded
    print(f"✓ Vocabulary: {seq} -> {encoded} -> {decoded}")

def test_dataset():
    """Test dataset loading"""
    print("\nTesting dataset...")
    from peptidegen.data.dataset import ConditionalPeptideDataset
    from peptidegen.data.vocabulary import VOCAB
    
    dataset = ConditionalPeptideDataset.from_csv(
        'dataset/train.csv', 
        vocab=VOCAB, 
        max_length=50, 
        min_length=5
    )
    
    batch = dataset[0]
    assert 'input_ids' in batch
    assert 'condition' in batch
    assert batch['condition'].shape[0] == 8  # 8 features
    print(f"✓ Dataset: {len(dataset)} samples, condition_dim={batch['condition'].shape[0]}")

def test_generator():
    """Test generator forward pass"""
    print("\nTesting generator...")
    from peptidegen.models.generator import GRUGenerator
    from peptidegen.data.vocabulary import VOCAB
    
    generator = GRUGenerator(
        vocab_size=VOCAB.vocab_size,
        embedding_dim=64,
        hidden_dim=256,
        latent_dim=128,
        max_length=50,
        num_layers=2,
        condition_dim=8,
        pad_idx=VOCAB.pad_idx,
        sos_idx=VOCAB.sos_idx,
        eos_idx=VOCAB.eos_idx,
    )
    
    batch_size = 4
    z = torch.randn(batch_size, 128)
    condition = torch.randn(batch_size, 8)
    
    # Test teacher forcing
    target = torch.randint(0, VOCAB.vocab_size, (batch_size, 20))
    output = generator(z, target=target, condition=condition)
    assert 'logits' in output
    assert output['logits'].shape == (batch_size, 20, VOCAB.vocab_size)
    
    # Test autoregressive
    output_free = generator(z, condition=condition)
    assert 'sequences' in output_free
    print(f"✓ Generator: logits shape={output['logits'].shape}")

def test_discriminator():
    """Test discriminator forward pass"""
    print("\nTesting discriminator...")
    from peptidegen.models.discriminator import CNNDiscriminator
    from peptidegen.data.vocabulary import VOCAB
    
    discriminator = CNNDiscriminator(
        vocab_size=VOCAB.vocab_size,
        embedding_dim=64,
        hidden_dim=256,
        max_length=52,
        pad_idx=VOCAB.pad_idx,
    )
    
    batch_size = 4
    seq_len = 30
    x = torch.randint(0, VOCAB.vocab_size, (batch_size, seq_len))
    
    output = discriminator(x)
    assert output.shape == (batch_size, 1)
    print(f"✓ Discriminator: output shape={output.shape}")

def test_evaluation():
    """Test evaluation metrics"""
    print("\nTesting evaluation...")
    from peptidegen.evaluation.metrics import calculate_diversity_metrics, analyze_amp_properties
    
    sequences = ["KLAKLAKKLAKLAK", "GIGKFLHSAKKFGKAFVGEIMNS", "RWKIFKKIEKMGRNIRDGIVKAG"]
    
    diversity = calculate_diversity_metrics(sequences)
    assert 'uniqueness_ratio' in diversity
    
    amp_props = analyze_amp_properties(sequences)
    assert 'hemolytic' in amp_props
    assert 'therapeutic' in amp_props
    
    print(f"✓ Evaluation: diversity={diversity['diversity_score']:.3f}, "
          f"hemolytic={amp_props['hemolytic']['summary']['mean']:.2f}")

def test_training_step():
    """Test one training step"""
    print("\nTesting training step...")
    from peptidegen.models.generator import GRUGenerator
    from peptidegen.models.discriminator import CNNDiscriminator
    from peptidegen.training.trainer import GANTrainer
    from peptidegen.data.vocabulary import VOCAB
    
    # Generator WITHOUT condition for standard trainer test
    generator = GRUGenerator(
        vocab_size=VOCAB.vocab_size,
        embedding_dim=64,
        hidden_dim=256,
        latent_dim=128,
        max_length=50,
        num_layers=2,
        condition_dim=None,  # No condition for standard trainer
        pad_idx=VOCAB.pad_idx,
        sos_idx=VOCAB.sos_idx,
        eos_idx=VOCAB.eos_idx,
    )
    
    discriminator = CNNDiscriminator(
        vocab_size=VOCAB.vocab_size,
        embedding_dim=64,
        hidden_dim=256,
        max_length=52,
        pad_idx=VOCAB.pad_idx,
    )
    
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        config={'learning_rate': 0.0001},
        device=torch.device('cpu'),
    )
    
    # Create fake batch
    batch = {
        'input_ids': torch.randint(0, VOCAB.vocab_size, (4, 52)),
    }
    
    losses = trainer.train_step(batch)
    assert 'g_loss' in losses
    assert 'd_loss' in losses
    print(f"✓ Training step: g_loss={losses['g_loss']:.4f}, d_loss={losses['d_loss']:.4f}")

def main():
    print("=" * 60)
    print("QUICK TESTS FOR LIGHTWEIGHTPEPTIDEGEN")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_vocabulary,
        test_dataset,
        test_generator,
        test_discriminator,
        test_evaluation,
        test_training_step,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
