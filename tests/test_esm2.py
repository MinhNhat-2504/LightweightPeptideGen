#!/usr/bin/env python
"""Test script for ESM2 integration."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_esm2_embedder():
    """Test ESM2 embedder."""
    print("=" * 60)
    print("Testing ESM2 Embedder")
    print("=" * 60)
    
    try:
        from peptidegen.models.esm2_embedder import ESM2Embedder, load_esm2_embedder
        
        # Test sequences
        sequences = [
            "MKLLVVAALVFAAGHA",
            "ACDEFGHIKLMNPQRS",
            "GIGKFLHSAKKFGKAFVGEIMNS",
        ]
        
        print(f"Test sequences: {len(sequences)}")
        for i, seq in enumerate(sequences):
            print(f"  {i+1}. {seq} (len={len(seq)})")
        
        # Load embedder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nDevice: {device}")
        
        # Use smaller model for testing
        model_name = "esm2_t6_8M_UR50D"  # Fastest model
        print(f"Loading ESM2 model: {model_name}")
        
        embedder = load_esm2_embedder(
            model_name=model_name,
            device=device,
            freeze=True,
            pooling="mean",
        )
        
        print(f"ESM2 embed_dim: {embedder.embed_dim}")
        
        # Get embeddings
        output = embedder(sequences)
        
        print(f"\nOutput keys: {list(output.keys())}")
        print(f"Sequence embeddings shape: {output['embeddings'].shape}")
        print(f"Token embeddings shape: {output['token_embeddings'].shape}")
        
        # Test simple interface
        seq_embeds = embedder.embed_sequences(sequences)
        print(f"\nSimple embed_sequences shape: {seq_embeds.shape}")
        assert output['embeddings'].shape[0] == len(sequences)
        assert output['token_embeddings'].shape[0] == len(sequences)
        assert seq_embeds.shape[0] == len(sequences)
        
        print("\n✅ ESM2 Embedder test PASSED")
        
    except Exception as e:
        print(f"\n❌ ESM2 Embedder test FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_esm2_structure_evaluator():
    """Test ESM2 Structure Evaluator."""
    print("\n" + "=" * 60)
    print("Testing ESM2 Structure Evaluator")
    print("=" * 60)
    
    try:
        from peptidegen.models.esm2_embedder import ESM2StructureEvaluator
        
        sequences = [
            "MKLLVVAALVFAAGHA",
            "ACDEFGHIKLMNPQRS",
        ]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use smaller model for testing
        evaluator = ESM2StructureEvaluator(
            esm_model_name="esm2_t6_8M_UR50D",  # Fastest
            projection_dim=64,
            gat_hidden=32,
            gat_heads=2,
            num_gat_layers=2,
            dropout=0.1,
            freeze_esm=True,
            device=device,
        )
        
        evaluator = evaluator.to(device)
        
        # Forward pass
        output = evaluator(sequences, return_features=True)
        
        print(f"\nOutput keys: {list(output.keys())}")
        print(f"Legacy proxy scores shape: {output['stability_score'].shape}")
        print(f"GAT features shape: {output['gat_features'].shape}")

        assert output['stability_score'].shape[0] == len(sequences)
        assert output['gat_features'].shape[0] == len(sequences)
        
        print("\n✅ ESM2 Structure Evaluator test PASSED")
        
    except Exception as e:
        print(f"\n❌ ESM2 Structure Evaluator test FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_esm2_generator():
    """Test ESM2 Conditioned Generator."""
    print("\n" + "=" * 60)
    print("Testing ESM2 Conditioned Generator")
    print("=" * 60)
    
    try:
        from peptidegen.models.esm2_generator import ESM2ConditionedGenerator
        from peptidegen.data.vocabulary import VOCAB
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        generator = ESM2ConditionedGenerator(
            vocab_size=VOCAB.vocab_size,
            embedding_dim=64,
            hidden_dim=128,
            latent_dim=64,
            max_length=50,
            num_layers=2,
            dropout=0.1,
            esm_model_name="esm2_t6_8M_UR50D",
            esm_projection_dim=64,
            freeze_esm=True,
            generator_type='GRU',
            pad_idx=VOCAB.pad_idx,
            sos_idx=VOCAB.sos_idx,
            eos_idx=VOCAB.eos_idx,
            device=device,
        )
        
        generator = generator.to(device)
        
        # Test forward pass
        batch_size = 2
        z = torch.randn(batch_size, 64, device=device)
        
        output = generator(z)
        
        print(f"\nOutput keys: {list(output.keys())}")
        print(f"Generated tokens shape: {output['sequences'].shape}")
        print(f"Logits shape: {output['logits'].shape}")
        
        # Decode sequences
        tokens = output['sequences']
        seqs = [VOCAB.decode(t.tolist()) for t in tokens]
        print(f"\nGenerated sequences:")
        for i, seq in enumerate(seqs):
            print(f"  {i+1}. {seq[:30]}...")
        assert output['sequences'].shape[0] == batch_size
        assert output['logits'].shape[0] == batch_size
        
        print("\n✅ ESM2 Conditioned Generator test PASSED")
        
    except Exception as e:
        print(f"\n❌ ESM2 Conditioned Generator test FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    test_esm2_embedder()
    test_esm2_structure_evaluator()
    test_esm2_generator()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("ESM2 Embedder: ✅ PASSED")
    print("ESM2 Structure Evaluator: ✅ PASSED")
    print("ESM2 Generator: ✅ PASSED")
    print("\n🎉 ALL TESTS PASSED!")
