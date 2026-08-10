#!/usr/bin/env python
"""
Generate peptide sequences.

Usage:
    python generate.py --checkpoint checkpoints/best_model.pt --num 1000
    python generate.py --checkpoint checkpoints/best_model.pt --output results/generated.fasta
    python generate.py --checkpoint checkpoints/best_model.pt --temperature 0.8 --top-p 0.9
"""

import argparse
import logging
import sys
from pathlib import Path

# Configure logging BEFORE peptidegen imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

from peptidegen.utils import set_seed
from peptidegen.inference import PeptideSampler


def main():
    parser = argparse.ArgumentParser(description='Generate peptide sequences')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--num', '-n', type=int, default=100, help='Number of sequences')
    parser.add_argument('--output', '-o', type=str, default='generated.fasta', help='Output file')
    parser.add_argument('--format', type=str, default='fasta', choices=['fasta', 'csv'])
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=0, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9, help='Nucleus sampling')
    parser.add_argument('--min-length', type=int, default=5)
    parser.add_argument('--max-length', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    # Stability filtering
    parser.add_argument('--stable-only', action='store_true',
                        help='Only output sequences with Instability Index < --stability-threshold')
    parser.add_argument('--stability-threshold', type=float, default=40.0,
                        help='Max instability index for --stable-only (default: 40.0)')
    parser.add_argument('--oversample', type=int, default=3,
                        help='Oversample multiplier for --stable-only (default: 3x)')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Load sampler
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    sampler = PeptideSampler.from_checkpoint(args.checkpoint)
    
    # Generate
    logger.info(f"Generating {args.num} sequences...")
    if args.stable_only:
        logger.info(
            f"Stability filter ON: II < {args.stability_threshold}, "
            f"oversample={args.oversample}x"
        )
        sequences = sampler.sample_stable(
            n=args.num,
            stability_threshold=args.stability_threshold,
            oversample=args.oversample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_length=args.min_length,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
    else:
        sequences = sampler.sample(
            n=args.num,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_length=args.min_length,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == 'fasta':
        sampler.save_fasta(sequences, args.output)
    else:
        sampler.save_csv(sequences, args.output, include_features=True)
    
    logger.info(f"Generated {len(sequences)} sequences -> {args.output}")
    
    # Print sample
    if sequences:
        logger.info("Sample sequences:")
        for seq in sequences[:5]:
            logger.info(f"  {seq} (len={len(seq)})")


if __name__ == '__main__':
    main()
