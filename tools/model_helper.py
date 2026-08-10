#!/usr/bin/env python
"""
🔬 PEPTIDE MODEL HELPER
=======================
Unified interface for:
- 3D Structure Folding (ESMFold/OmegaFold)
- Model Exporting (Checkpoint extraction)
- Training Log Analysis (Loss/Metric trends)
"""
import argparse
import sys
import json
import logging
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from peptidegen.utils import read_fasta, setup_logging

def run_export(args):
    setup_logging()
    logger = logging.getLogger(__name__)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'generator' in ckpt:
        torch.save({'state_dict': ckpt['generator'], 'config': ckpt.get('config', {})}, output_dir / 'generator.pt')
        logger.info("Exported generator.pt")

def main():
    parser = argparse.ArgumentParser(description="Peptide Model Helper")
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommands")
    exp_p = subparsers.add_parser("export", help="Export weights")
    exp_p.add_argument("--checkpoint", "-c", required=True)
    exp_p.add_argument("--output", "-o", default="exported_models")
    
    args = parser.parse_args()
    if args.subcommand == "export": run_export(args)
    else: parser.print_help()

if __name__ == "__main__":
    main()
