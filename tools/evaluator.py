#!/usr/bin/env python
"""
🔬 PEPTIDE EVALUATOR & BENCHMARK SUITE
=====================================
Unified interface for:
- Diagnostic Reporting (Stability, Diversity, AMP)
- Benchmark Comparison (LPG vs Baselines)
- Novelty Analysis (Levenshtein uniqueness)
"""
import argparse
import sys
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from peptidegen.evaluation import (
    PeptideStabilityAnalyzer,
    calculate_diversity_metrics,
    calculate_amino_acid_distribution,
    detect_mode_collapse,
    analyze_amp_properties,
)
from peptidegen.utils import (
    read_fasta,
    save_json,
    setup_logging,
    setup_plot_style,
    plot_histogram,
    plot_bar_chart
)

# Initialize logging
logger = logging.getLogger(__name__)

# =============================================================================
# SUBCOMMAND: REPORT
# =============================================================================

def save_combined_diagnostic_report(results: dict, out_path: Path, threshold: float = 40.0):
    """Generate a single composite figure with all key diagnostic plots."""
    setup_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()

    stability = results.get('stability', {})
    metrics = stability.get('metrics', [])
    
    # 1. Instability
    ax = axes[0]
    if metrics:
        vals = [m.get('instability_index', 0.0) for m in metrics]
        ax.hist(vals, bins=30, color='#4e79a7', alpha=0.8)
        ax.axvline(x=threshold, color='r', linestyle='--', alpha=0.5)
        ax.set_title('Instability Index Distribution', fontweight='bold')
        ax.set_xlabel('Index')
    
    # 2. GRAVY
    ax = axes[1]
    if metrics:
        vals = [m.get('gravy', 0.0) for m in metrics]
        ax.hist(vals, bins=30, color='#f28e2c', alpha=0.8)
        ax.set_title('GRAVY Distribution', fontweight='bold')
        ax.set_xlabel('GRAVY')

    # 3. Length
    ax = axes[2]
    if metrics:
        vals = [m.get('length', 0) for m in metrics]
        ax.hist(vals, bins=30, color='#e15759', alpha=0.8)
        ax.set_title('Sequence Length Distribution', fontweight='bold')
        ax.set_xlabel('Length')

    # 4. AA Distribution
    ax = axes[3]
    aa_dist = results.get('aa_distribution', {})
    if aa_dist:
        aas = sorted([k for k in aa_dist.keys() if k != 'non_standard'])
        freqs = [aa_dist[a]['frequency'] for a in aas]
        ax.bar(aas, freqs, color='#76b7b2')
        ax.set_title('Amino Acid Distribution', fontweight='bold')
        ax.set_ylabel('Frequency')

    # 5. Diversity
    ax = axes[4]
    diversity = results.get('diversity', {})
    if diversity:
        keys = ['uniqueness_ratio', 'bigram_diversity', 'trigram_diversity', 'diversity_score']
        labels = ['Unique', 'Bigram', 'Trigram', 'Score']
        vals = [diversity.get(k, 0.0) for k in keys]
        ax.bar(labels, vals, color='#59a14f')
        ax.set_title('Diversity Metrics', fontweight='bold')

    # 6. AMP Probability
    ax = axes[5]
    amp_props = results.get('amp_properties', {})
    amp_probs = amp_props.get('amp_probability', [])
    if amp_probs:
        ax.hist(amp_probs, bins=20, color='#edc949', alpha=0.8)
        ax.axvline(x=0.8, color='r', linestyle='--', alpha=0.5)
        ax.set_title('AMP Probability Distribution', fontweight='bold')
        ax.set_xlabel('Probability')

    plt.tight_layout(pad=4.0)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_report(args):
    """Subcommand to generate full diagnostic report."""
    setup_logging()
    fasta_path = Path(args.input)
    if not fasta_path.exists():
        logger.error(f"Input file not found: {fasta_path}")
        return

    logger.info(f"Loading sequences from {fasta_path}")
    sequences = read_fasta(fasta_path)
    logger.info(f"Loaded {len(sequences)} sequences")

    # Core Analysis
    logger.info("Running core analysis...")
    
    # Sample for slow metrics if dataset is huge
    MAX_STABILITY_SAMPLE = 10000
    if len(sequences) > MAX_STABILITY_SAMPLE:
        logger.info(f"Sampling {MAX_STABILITY_SAMPLE} sequences for stability/AMP analysis due to large input size.")
        indices = np.random.choice(len(sequences), MAX_STABILITY_SAMPLE, replace=False)
        sample_seqs = [sequences[i] for i in indices]
    else:
        sample_seqs = sequences

    analyzer = PeptideStabilityAnalyzer(stability_threshold=args.threshold)
    stability = analyzer.analyze_batch(sample_seqs)
    
    # Diversity and AA distribution are fast, run on full set
    diversity = calculate_diversity_metrics(sequences)
    aa_dist = calculate_amino_acid_distribution(sequences)
    mode_collapse = detect_mode_collapse(sequences)
    
    try:
        amp_props = analyze_amp_properties(sample_seqs)
    except Exception as e:
        logger.warning(f"AMP property analysis failed: {e}")
        amp_props = {}

    results = {
        'stability': stability,
        'diversity': diversity,
        'aa_distribution': aa_dist,
        'mode_collapse': mode_collapse,
        'amp_properties': amp_props,
        'length_stats': {
            'mean_length': np.mean([len(s) for s in sequences]),
            'std_length': np.std([len(s) for s in sequences])
        }
    }

    # Save JSON
    save_json(results, args.output)
    logger.info(f"Report saved to {args.output}")

    # Generate Plots
    if args.plot:
        out_dir = Path(args.outdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_combined_diagnostic_report(results, out_dir / 'diagnostic_report.png', args.threshold)
        logger.info(f"Composite diagnostic report saved to {out_dir}/diagnostic_report.png")

# =============================================================================
# SUBCOMMAND: COMPARE
# =============================================================================

def load_results_for_comparison(results_dir: Path) -> dict:
    """Load and map all evaluation results from JSON files."""
    results = {}
    if not results_dir.exists():
        return results
    
    for f in results_dir.glob('*_eval.json'):
        model_name = f.stem.replace('_eval', '')
        display_name = {
            'lightweight_peptide_gen': 'Proposed (LPG)',
            'esm2gen': 'ESM2Gen',
            'hydramp': 'HydrAMP',
            'm3cad': 'M3-CAD',
            'pepgraphormer': 'PepGraphormer'
        }.get(model_name, model_name.capitalize())
        
        with open(f, 'r') as fp:
            raw_data = json.load(fp)
            # Map LPG vs Standard format
            if 'stability' in raw_data and isinstance(raw_data['stability'], dict):
                metrics = {
                    'entropy': raw_data.get('mode_collapse', {}).get('entropy', 0.0),
                    'ngram_diversity_2': raw_data.get('diversity', {}).get('bigram_diversity', 0.0),
                    'uniqueness_ratio': raw_data.get('diversity', {}).get('uniqueness_ratio', 0.0),
                    'stable_ratio': raw_data.get('stability', {}).get('stability_rate', 0.0) / 100.0,
                    'mean_length': raw_data.get('length_stats', {}).get('mean_length', 0.0),
                }
            else:
                metrics = {
                    'entropy': raw_data.get('entropy', 0.0),
                    'ngram_diversity_2': raw_data.get('ngram_diversity_2', 0.0),
                    'uniqueness_ratio': raw_data.get('uniqueness_ratio', 0.0),
                    'stable_ratio': raw_data.get('stable_ratio', 0.0),
                    'mean_length': raw_data.get('mean_length', 0.0),
                }
            results[display_name] = metrics
    return results

def run_compare(args):
    """Subcommand to run benchmark comparison."""
    setup_logging()
    results_dir = Path(args.results_dir)
    data = load_results_for_comparison(results_dir)

    if not data:
        logger.error(f"No *_eval.json results found in {results_dir}")
        return

    models = sorted(list(data.keys()))
    metrics_to_plot = {
        'Entropy': 'entropy',
        'Bigram Diversity': 'ngram_diversity_2',
        'Uniqueness Ratio': 'uniqueness_ratio',
        'Stable Ratio': 'stable_ratio',
        'Mean Length': 'mean_length'
    }

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_plot_style()
    colors = ['#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f', '#edc949']
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    axes = axes.flatten()

    for i, (metric_name, metric_key) in enumerate(metrics_to_plot.items()):
        values = [data[m].get(metric_key, 0) for m in models]
        ax = axes[i]
        ax.bar(models, values, color=colors[:len(models)], alpha=0.85, edgecolor='black', linewidth=0.3)
        ax.set_title(metric_name, fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=10)
        for j, val in enumerate(values):
            ax.text(j, val, f'{val:.3f}' if val < 10 else f'{val:.1f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    if len(metrics_to_plot) < len(axes):
        for j in range(len(metrics_to_plot), len(axes)):
            fig.delaxes(axes[j])

    fig.suptitle('LightweightPeptideGen: Comprehensive Benchmark Comparison', 
                 fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / 'model_comparison_composite.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Composite comparison plot saved to {out_dir}/model_comparison_composite.png")

# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Peptide Evaluator & Benchmark Suite")
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Command: report
    res_p = subparsers.add_parser("report", help="Generate full diagnostic report")
    res_p.add_argument("--input", "-i", type=str, required=True, help="Input FASTA file")
    res_p.add_argument("--output", "-o", type=str, default="results/eval_report.json", help="Output JSON path")
    res_p.add_argument("--plot", action="store_true", help="Generate diagnostic plots")
    res_p.add_argument("--outdir", type=str, default="results/plots", help="Plots output directory")
    res_p.add_argument("--threshold", type=float, default=40.0, help="Instability threshold")

    # Command: compare
    cmp_p = subparsers.add_parser("compare", help="Benchmark comparison across models")
    cmp_p.add_argument("--results-dir", type=str, default="baselines/results", help="Directory with eval JSONs")
    cmp_p.add_argument("--outdir", type=str, default="results/plots", help="Plots output directory")

    args = parser.parse_args()

    if args.subcommand == "report":
        run_report(args)
    elif args.subcommand == "compare":
        run_compare(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
