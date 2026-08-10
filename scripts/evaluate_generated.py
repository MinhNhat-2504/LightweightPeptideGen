#!/usr/bin/env python
"""
B5+B6+B7 evaluation harness with multi-seed mean±std and significance tests.

Scores generated peptides on:
    * stability rate (II<40)            — training *surrogate*, labelled as such
    * ESM-2 foldability (B5)            — pseudo-perplexity, contact-order, helix
    * AMP probability (B6)             — trained ESM-2 oracle
    * hemolysis (B7, optional)         — trained ESM-2 oracle
    * diversity / uniqueness

Aggregates across random seeds (mean ± std) and runs significance tests of the
proposed model vs each baseline (Mann-Whitney U on pooled per-sequence scores;
paired t-test on per-seed stability rates).

Input convention
----------------
A directory of FASTA files named ``<model>_seed<N>.fasta``, e.g.
    results/gen/LightweightPeptideGen_seed42.fasta
    results/gen/HydrAMP_seed42.fasta  ...

Usage
-----
    python scripts/evaluate_generated.py \
        --gen-dir results/gen --reference LightweightPeptideGen \
        --amp-oracle results/oracle_amp.pkl \
        --hemo-oracle results/oracle_hemo.pkl \
        --foldability --esm-model esm2_t12_35M_UR50D \
        --out results/benchmark.json
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from peptidegen.evaluation import PeptideStabilityAnalyzer, calculate_diversity_metrics

FNAME_RE = re.compile(r"(?P<model>.+?)_seed(?P<seed>\d+)", re.IGNORECASE)


def read_fasta(path):
    seqs, buf = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if buf:
                    seqs.append("".join(buf)); buf = []
            elif line:
                buf.append(line)
    if buf:
        seqs.append("".join(buf))
    return [s.upper() for s in seqs]


def per_run_metrics(seqs, analyzer, amp_oracle, hemo_oracle, fold_eval):
    """Scalar + per-sequence metrics for one FASTA (one model, one seed)."""
    stab = analyzer.analyze_batch(seqs)
    div = calculate_diversity_metrics(seqs)
    m = {
        "n": len(seqs),
        "stable_rate": stab.get("summary", {}).get("stability_rate", 0.0),  # surrogate
        "mean_ii": stab.get("summary", {}).get("instability_index", {}).get("mean", float("nan")),
        "uniqueness": div.get("uniqueness_ratio", 0.0),
        "bigram_div": div.get("bigram_diversity", 0.0),
        "mean_length": float(np.mean([len(s) for s in seqs])) if seqs else 0.0,
        "_per_seq": {},
    }
    if amp_oracle is not None:
        s = amp_oracle.score_generated(seqs)
        m["amp_mean_prob"] = s["mean_prob"]
        m["amp_positive_rate"] = s["positive_rate"]
        m["_per_seq"]["amp_prob"] = s["_per_sequence"]
    if hemo_oracle is not None:
        s = hemo_oracle.score_generated(seqs)
        m["hemo_mean_prob"] = s["mean_prob"]
        m["hemo_positive_rate"] = s["positive_rate"]   # fraction predicted hemolytic
        m["low_hemolytic_rate"] = 1.0 - s["positive_rate"]
        m["_per_seq"]["hemo_prob"] = s["_per_sequence"]
    if fold_eval is not None:
        f = fold_eval.evaluate(seqs, sample=min(500, len(seqs)))
        m["esm_pseudo_ppl"] = f["esm_pseudo_perplexity"]["mean"]
        m["helix_frac"] = f["helix_fraction"]["mean"]
        m["_per_seq"]["pseudo_ppl"] = f["_per_sequence"]["pseudo_perplexity"]
        if f.get("esm_contact_order"):
            m["esm_contact_order"] = f["esm_contact_order"]["mean"]
    return m


def aggregate(runs):
    """mean ± std across seeds for every scalar metric."""
    scalars = defaultdict(list)
    for r in runs:
        for k, v in r.items():
            if k != "_per_seq" and isinstance(v, (int, float)) and v == v:
                scalars[k].append(v)
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_seeds": len(v)}
            for k, v in scalars.items()}


def pool_per_seq(runs, key):
    out = []
    for r in runs:
        out += r.get("_per_seq", {}).get(key, [])
    return np.asarray(out, dtype=float)


def significance(ref_runs, base_runs):
    """Proposed vs baseline: Mann-Whitney on pooled per-seq + t-test on per-seed."""
    from scipy import stats

    res = {}
    for key in ("amp_prob", "pseudo_ppl", "hemo_prob"):
        a, b = pool_per_seq(ref_runs, key), pool_per_seq(base_runs, key)
        if a.size and b.size:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            res[f"{key}_mannwhitney_p"] = float(p)
            res[f"{key}_ref_mean"] = float(a.mean())
            res[f"{key}_base_mean"] = float(b.mean())
    # per-seed stability rate t-test
    ra = [r["stable_rate"] for r in ref_runs if "stable_rate" in r]
    rb = [r["stable_rate"] for r in base_runs if "stable_rate" in r]
    if len(ra) > 1 and len(rb) > 1:
        t, p = stats.ttest_ind(ra, rb, equal_var=False)
        res["stable_rate_ttest_p"] = float(p)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True, help="dir of <model>_seed<N>.fasta")
    ap.add_argument("--reference", default="LightweightPeptideGen", help="proposed model name")
    ap.add_argument("--amp-oracle", default=None)
    ap.add_argument("--hemo-oracle", default=None)
    ap.add_argument("--foldability", action="store_true")
    ap.add_argument("--esm-model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--threshold", type=float, default=40.0)
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args()

    analyzer = PeptideStabilityAnalyzer(stability_threshold=args.threshold)
    amp_oracle = hemo_oracle = fold_eval = None
    if args.amp_oracle:
        from peptidegen.evaluation.oracle import ESM2Oracle
        amp_oracle = ESM2Oracle.load(args.amp_oracle)
    if args.hemo_oracle:
        from peptidegen.evaluation.oracle import ESM2Oracle
        hemo_oracle = ESM2Oracle.load(args.hemo_oracle)
    if args.foldability:
        from peptidegen.evaluation.foldability import FoldabilityEvaluator
        fold_eval = FoldabilityEvaluator(model_name=args.esm_model)

    # group FASTA files by model
    by_model = defaultdict(list)
    for f in sorted(Path(args.gen_dir).glob("*.fasta")):
        mo = FNAME_RE.match(f.stem)
        model = mo.group("model") if mo else f.stem
        by_model[model].append(f)

    if not by_model:
        logger.error(f"No FASTA files in {args.gen_dir}")
        sys.exit(1)

    results = {"per_model": {}, "significance_vs_reference": {}}
    runs_by_model = {}
    for model, files in by_model.items():
        runs = []
        for fp in files:
            seqs = [s for s in read_fasta(fp) if len(s) >= 5]
            logger.info(f"[{model}] {fp.name}: {len(seqs)} seqs")
            runs.append(per_run_metrics(seqs, analyzer, amp_oracle, hemo_oracle, fold_eval))
        runs_by_model[model] = runs
        results["per_model"][model] = {"n_seeds": len(runs), "aggregate": aggregate(runs)}

    # significance: reference vs each baseline
    ref = args.reference
    if ref in runs_by_model:
        for model, runs in runs_by_model.items():
            if model == ref:
                continue
            results["significance_vs_reference"][model] = significance(runs_by_model[ref], runs)
    else:
        logger.warning(f"Reference model '{ref}' not found among {list(runs_by_model)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote benchmark -> {args.out}")

    # console summary table
    print("\n=== mean ± std across seeds ===")
    metrics = ["stable_rate", "amp_mean_prob", "amp_positive_rate",
               "low_hemolytic_rate", "esm_pseudo_ppl", "helix_frac", "uniqueness"]
    for model, info in results["per_model"].items():
        agg = info["aggregate"]
        cells = [f"{k}={agg[k]['mean']:.3f}±{agg[k]['std']:.3f}" for k in metrics if k in agg]
        print(f"  {model:28s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
