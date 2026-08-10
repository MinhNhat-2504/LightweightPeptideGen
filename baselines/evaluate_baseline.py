"""
Generate sequences and evaluate a baseline model.

Usage:
    python baselines/evaluate_baseline.py --model hydramp --checkpoint checkpoints/hydramp/best.pt
    python baselines/evaluate_baseline.py --all
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from baselines.common.data_utils import get_dataloaders, decode_tokens, get_vocab_info
from baselines.common.metrics import evaluate_generated_sequences


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_model(model_name: str, checkpoint_path: str, num_samples: int, device: str):
    """Load model, generate sequences, and compute metrics."""
    print(f"\n[Evaluate] Testing {model_name} from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"  -> Checkpoint not found: {checkpoint_path}")
        return None

    # Get data specifically to sample conditions
    num_workers = min(os.cpu_count() or 4, 4)
    _, val_loader = get_dataloaders(
        train_csv='dataset/train.csv',
        val_csv='dataset/val.csv',
        batch_size=256,
        num_workers=num_workers,
    )

    # Sample conditions from validation set
    all_conditions = []
    for batch in val_loader:
        all_conditions.append(batch['condition'])
    val_conditions = torch.cat(all_conditions, dim=0).to(device)
    
    # If we need more than we have, sample with replacement
    if num_samples > val_conditions.size(0):
        print(f"  -> num_samples ({num_samples}) > val_set ({val_conditions.size(0)}). Sampling with replacement.")
        indices = torch.randint(0, val_conditions.size(0), (num_samples,))
        condition_tensor = val_conditions[indices]
    else:
        condition_tensor = val_conditions[:num_samples]

    # Load vocab info
    vocab_info = get_vocab_info()
    sos_idx = vocab_info['sos_idx']
    eos_idx = vocab_info['eos_idx']

    # Initialize model
    model = None
    if model_name == 'hydramp':
        from baselines.hydramp.model import HydrAMPModel
        model = HydrAMPModel(vocab_size=24, embedding_dim=128, hidden_dim=256,
                             latent_dim=128, condition_dim=8, num_layers=2)
    elif model_name == 'm3cad':
        from baselines.m3cad.model import M3CADModel
        model = M3CADModel(vocab_size=24, embedding_dim=128, hidden_dim=256,
                           latent_dim=128, condition_dim=8, cond_enc_dim=32, num_layers=2)
    elif model_name == 'esm2gen':
        from baselines.esm2gen.model import ESM2DecoderModel
        model = ESM2DecoderModel(vocab_size=24, embedding_dim=128, hidden_dim=256,
                                 latent_dim=128, esm_projection_dim=128, condition_dim=8)
    elif model_name == 'pepgraphormer':
        from baselines.pepgraphormer.model import PepGraphormerDecoderModel
        model = PepGraphormerDecoderModel(vocab_size=24, embedding_dim=128, hidden_dim=256,
                                          latent_dim=128, enc_projection_dim=128, condition_dim=8, num_layers=2)

    if model is None:
        raise ValueError(f"Unknown model: {model_name}")

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Generate
    print(f"  -> Generating {num_samples} sequences...")
    # Batched generation to avoid OOM
    batch_size = 1024
    all_seqs = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            cond_batch = condition_tensor[i:i+batch_size]
            
            if model_name == 'm3cad':
                # M3-CAD needs cond_feat passed natively through model.generate
                pass
            
            # All 4 models implement a standard .generate() signature
            token_ids = model.generate(
                num_samples=cond_batch.size(0),
                condition=cond_batch,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                max_len=52,
                temperature=1.0,  # default standard temperature
                top_p=0.9,        # default nucleus sampling threshold
                device=device,
            )
                
            batch_seqs = decode_tokens(token_ids)
            all_seqs.extend(batch_seqs)

    # Save sequences to FASTA
    results_dir = 'baselines/results'
    os.makedirs(results_dir, exist_ok=True)
    fasta_path = os.path.join(results_dir, f'{model_name}_1M.fasta')
    with open(fasta_path, 'w') as f:
        for j, seq in enumerate(all_seqs):
            f.write(f">seq_{j}\n{seq}\n")
    print(f"  -> Saved sequences to {fasta_path}")

    # Evaluate
    print("  -> Computing metrics...")
    metrics = evaluate_generated_sequences(all_seqs)

    # Save results
    out_dir = 'baselines/results'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{model_name}_eval.json')
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  -> Saved results to {out_path}")

    # Print summary
    print(f"     Entropy: {metrics.get('entropy', 0):.4f}")
    print(f"     N-gram(2): {metrics.get('ngram_diversity_2', 0):.4f}")
    print(f"     Instability: {metrics.get('mean_instability_index', 0):.2f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='hydramp',
                        choices=['hydramp', 'm3cad', 'esm2gen', 'pepgraphormer', 'all'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to specific checkpoint. If None, checks Default paths.')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of sequences to generate and evaluate')
    args = parser.parse_args()

    device = get_device()
    models = ['hydramp', 'm3cad', 'esm2gen', 'pepgraphormer'] if args.model == 'all' else [args.model]

    for m in models:
        ckpt = args.checkpoint if args.checkpoint else f'baselines/checkpoints/{m}/best.pt'
        evaluate_model(m, ckpt, args.num_samples, device)


if __name__ == '__main__':
    main()
