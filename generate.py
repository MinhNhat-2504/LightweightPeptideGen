#!/usr/bin/env python
"""
Generate peptide sequences.

Usage:
    python generate.py --checkpoint checkpoints/best_model.pt --num 1000
    python generate.py --checkpoint checkpoints/best_model.pt --output results/generated.fasta
    python generate.py --checkpoint checkpoints/best_model.pt --temperature 0.8 --top-p 0.9
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path, base: Path) -> str:
    """Record paths relative to the sidecar directory when possible."""
    try:
        return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()
    except ValueError:
        return str(path.resolve())


def git_state() -> dict:
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ['git', 'status', '--porcelain'], check=True, capture_output=True, text=True
        ).stdout.strip())
        return {'commit': commit, 'dirty_worktree': dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {'commit': None, 'dirty_worktree': None}


def main():
    parser = argparse.ArgumentParser(description='Generate peptide sequences')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument(
        '--model-id', type=str, default=None,
        help='Stable model/ablation identifier; required for a reportable artifact',
    )
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
    # Optional empirical II filtering (for exploration only, never reportable).
    parser.add_argument('--ii-screen-only', '--stable-only', dest='ii_screen_only',
                        action='store_true',
                        help='Only output sequences with II below --ii-threshold; '
                             '--stable-only is a deprecated alias')
    parser.add_argument('--ii-threshold', '--stability-threshold', dest='ii_threshold',
                        type=float, default=40.0,
                        help='Maximum Instability Index for --ii-screen-only (default: 40.0)')
    parser.add_argument('--oversample', type=int, default=3,
                        help='Oversample multiplier for --ii-screen-only (default: 3x)')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Load sampler
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    sampler = PeptideSampler.from_checkpoint(args.checkpoint)
    
    # Generate
    logger.info(f"Generating {args.num} sequences...")
    if args.ii_screen_only:
        logger.info(
            f"Empirical II screen ON: II < {args.ii_threshold}, "
            f"oversample={args.oversample}x"
        )
        sequences = sampler.sample_ii_screen(
            n=args.num,
            ii_threshold=args.ii_threshold,
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

    checkpoint_path = Path(args.checkpoint)
    data_metadata = sampler.checkpoint_metadata.get('data_metadata') or {}
    condition_dim = getattr(sampler.G, 'condition_dim', None)
    feature_names = data_metadata.get('condition_feature_names') or []
    feature_stats = data_metadata.get('condition_feature_stats') or {}
    condition_metadata_complete = (
        not condition_dim
        or (len(feature_names) == condition_dim and all(name in feature_stats for name in feature_names))
    )
    git = git_state()
    checkpoint_commit = (
        ((sampler.checkpoint_metadata.get('run_metadata') or {}).get('git') or {}).get('commit')
    )
    sidecar_base = output_path.resolve().parent
    dataset_audit = data_metadata.get('dataset_build_audit') or {}
    dataset_audit_path = dataset_audit.get('path')
    dataset_audit_record = {
        'path': portable_path(Path(dataset_audit_path), sidecar_base)
        if dataset_audit_path else None,
        'sha256': dataset_audit.get('sha256'),
    }
    output_record = {
        'path': portable_path(output_path, sidecar_base),
        'sha256': sha256(output_path),
        'format': args.format,
        'records': len(sequences),
    }
    metadata = {
        'schema_version': 1,
        'path_base': 'metadata_file_parent',
        'model_id': args.model_id,
        'reportable': bool(
            sampler.checkpoint_metadata.get('artifact_reportable') is True
            and
            data_metadata.get('reportable_data') is True
            and len(sequences) == args.num
            and not args.ii_screen_only
            and condition_metadata_complete
            and git.get('commit')
            and git.get('dirty_worktree') is False
            and checkpoint_commit == git.get('commit')
            and args.model_id
        ),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'command': shlex.join(sys.argv),
        'seed': args.seed,
        'checkpoint': {
            'path': portable_path(checkpoint_path, sidecar_base),
            'sha256': sha256(checkpoint_path),
        },
        'output': output_record,
        'sampling': {
            'requested_sequences': args.num,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'top_p': args.top_p,
            'min_length': args.min_length,
            'max_length': args.max_length,
            'batch_size': args.batch_size,
            'ii_screen_only_post_filter': args.ii_screen_only,
            'stable_only_post_filter': args.ii_screen_only,
            'condition_mode': 'normalized_training_mean' if condition_dim else 'unconditional',
        },
        'condition_feature_names': feature_names,
        'dataset_build_audit': dataset_audit_record,
        'software': {
            'python': platform.python_version(),
            'torch': __import__('torch').__version__,
            'git': git,
        },
    }
    metadata_path = output_path.with_suffix(output_path.suffix + '.metadata.json')
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    logger.info(f"Generation metadata -> {metadata_path} (reportable={metadata['reportable']})")
    
    logger.info(f"Generated {len(sequences)} sequences -> {args.output}")
    
    # Print sample
    if sequences:
        logger.info("Sample sequences:")
        for seq in sequences[:5]:
            logger.info(f"  {seq} (len={len(seq)})")


if __name__ == '__main__':
    main()
