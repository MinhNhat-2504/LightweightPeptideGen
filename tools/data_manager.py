#!/usr/bin/env python
"""
🔬 PEPTIDE DATA MANAGER
=======================
Unified interface for:
- Preprocessing (Cleaning, Dedup, Balancing, Splitting)
- Validation (Integrity checks)
- Analysis (Biological properties)
- Visualization (Distribution plots)
"""
import argparse
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from peptidegen.utils import (
    read_fasta,
    save_json,
    setup_logging,
    setup_plot_style
)

class ProcessingConfig:
    MIN_LENGTH = 5
    MAX_LENGTH = 50
    VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')
    REMOVE_EXACT_DUPLICATES = True
    TRAIN_RATIO = 0.8
    TEST_RATIO = 0.2
    RANDOM_SEED = 42

class AdvancedFeatureExtractor:
    def __init__(self):
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        self.ProteinAnalysis = ProteinAnalysis

    def extract(self, seq: str) -> dict:
        analysis = self.ProteinAnalysis(seq)
        return {
            'length': len(seq),
            'instability_index': analysis.instability_index(),
            'gravy': analysis.gravy(),
            'isoelectric_point': analysis.isoelectric_point()
        }

def run_analyze(args):
    setup_logging()
    extractor = AdvancedFeatureExtractor()
    input_path = Path(args.input)
    sequences = read_fasta(input_path) if input_path.suffix != '.csv' else pd.read_csv(input_path)['sequence'].tolist()
    
    results = [extractor.extract(s) for s in tqdm(sequences)]
    save_json(pd.DataFrame(results).describe().to_dict(), args.output)

def run_plot(args):
    setup_plot_style()
    df = pd.read_csv(args.input) if Path(args.input).suffix == '.csv' else pd.DataFrame({'sequence': read_fasta(args.input)})
    if 'instability_index' not in df.columns:
        extractor = AdvancedFeatureExtractor()
        props = [extractor.extract(s) for s in df['sequence']]
        df = pd.concat([df, pd.DataFrame(props)], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df['instability_index'], ax=axes[0], color='skyblue', kde=True)
    sns.histplot(df['gravy'], ax=axes[1], color='salmon', kde=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)

def main():
    parser = argparse.ArgumentParser(description="Peptide Data Manager")
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommands")
    
    ana_p = subparsers.add_parser("analyze", help="Compute biological properties")
    ana_p.add_argument("--input", "-i", required=True)
    ana_p.add_argument("--output", "-o", default="results/data_stats.json")

    plt_p = subparsers.add_parser("plot", help="Visualize property distributions")
    plt_p.add_argument("--input", "-i", required=True)
    plt_p.add_argument("--output", "-o", default="results/plots/dataset_dist.png")

    args = parser.parse_args()
    if args.subcommand == "analyze": run_analyze(args)
    elif args.subcommand == "plot": run_plot(args)
    else: parser.print_help()

if __name__ == "__main__":
    main()
