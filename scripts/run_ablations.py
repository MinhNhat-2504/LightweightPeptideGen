#!/usr/bin/env python
"""Aggregate completed ablation artifacts; never invent experiment results.

Every variant must point to a reportable ``evaluate_generated.py`` JSON file.
The script validates seed alignment, records artifact hashes, computes paired
seed-level contrasts against the full model, applies Holm correction, and emits
JSON/LaTeX summaries. See ``config/ablation_manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


DEFAULT_METRICS = [
    "stable_rate_ii_lt_40_percent",
    "mean_instability_index",
    "esm2_pseudo_perplexity",
    "uniqueness_ratio",
]
SAMPLING_KEYS = (
    "requested_sequences", "temperature", "top_k", "top_p",
    "min_length", "max_length", "ii_screen_only_post_filter",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def holm_adjust(values: Sequence[float]) -> List[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def load_variant(root: Path, spec: Dict[str, Any], expected_seeds: Sequence[int],
                 expected_n: int) -> Dict[str, Any]:
    report_path = (root / spec["benchmark_report"]).resolve()
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("reportable"):
        raise ValueError(f"{report_path}: benchmark is marked reportable=false")
    model = spec["model_key"]
    if model not in report.get("per_model", {}):
        raise ValueError(f"{report_path}: model_key '{model}' not found")
    item = report["per_model"][model]
    runs = item.get("runs", [])
    seeds = [int(run["seed"]) for run in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{report_path}: duplicate seeds")
    if set(seeds) != set(expected_seeds):
        raise ValueError(
            f"{report_path}: expected seeds {sorted(expected_seeds)}, found {sorted(seeds)}"
        )
    counts = {int(run.get("n", -1)) for run in runs}
    if counts != {expected_n}:
        raise ValueError(f"{report_path}: every seed must contain exactly {expected_n} sequences")
    sampling_signatures = {
        tuple((key, (run.get("generation_sampling") or {}).get(key)) for key in SAMPLING_KEYS)
        for run in runs
    }
    if len(sampling_signatures) != 1:
        raise ValueError(f"{report_path}: sampling settings differ across seeds")
    dataset_hashes = {run.get("dataset_build_report_sha256") for run in runs}
    code_commits = {run.get("code_commit") for run in runs}
    if None in dataset_hashes or len(dataset_hashes) != 1:
        raise ValueError(f"{report_path}: dataset provenance differs across seeds")
    if None in code_commits or len(code_commits) != 1:
        raise ValueError(f"{report_path}: code commit differs across seeds")
    return {
        "id": spec["id"],
        "label": spec["label"],
        "change": spec["change"],
        "model_key": model,
        "benchmark_report": str(report_path),
        "benchmark_sha256": sha256(report_path),
        "runs": runs,
        "aggregate": item["aggregate"],
        "sampling_signature": dict(next(iter(sampling_signatures))),
        "dataset_build_report_sha256": next(iter(dataset_hashes)),
        "code_commit": next(iter(code_commits)),
    }


def paired_contrast(full: Dict[str, Any], variant: Dict[str, Any], metric: str) -> Dict[str, Any]:
    full_by_seed = {int(row["seed"]): row for row in full["runs"]}
    variant_by_seed = {int(row["seed"]): row for row in variant["runs"]}
    if set(full_by_seed) != set(variant_by_seed):
        raise ValueError(
            f"seed mismatch full={sorted(full_by_seed)} vs {variant['id']}={sorted(variant_by_seed)}"
        )
    seeds = sorted(full_by_seed)
    if any(metric not in full_by_seed[s] or metric not in variant_by_seed[s] for s in seeds):
        return {}
    x = np.asarray([full_by_seed[s][metric] for s in seeds], dtype=float)
    y = np.asarray([variant_by_seed[s][metric] for s in seeds], dtype=float)
    diff = x - y
    if len(diff) < 2:
        return {}
    t_result = stats.ttest_rel(x, y)
    try:
        w_result = stats.wilcoxon(diff, alternative="two-sided", method="auto")
        wilcoxon_stat, wilcoxon_p = float(w_result.statistic), float(w_result.pvalue)
    except ValueError:
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0
    sem = stats.sem(diff)
    half = float(stats.t.ppf(0.975, len(diff) - 1) * sem) if sem > 0 else 0.0
    sd = diff.std(ddof=1)
    return {
        "unit_of_analysis": "matched_random_seed",
        "seeds": seeds,
        "full_mean": float(x.mean()),
        "variant_mean": float(y.mean()),
        "paired_difference_full_minus_variant": float(diff.mean()),
        "paired_difference_ci95": [float(diff.mean() - half), float(diff.mean() + half)],
        "paired_t": float(t_result.statistic) if math.isfinite(float(t_result.statistic)) else 0.0,
        "paired_t_p_raw": float(t_result.pvalue) if math.isfinite(float(t_result.pvalue)) else 1.0,
        "cohen_dz": float(diff.mean() / sd) if sd > 0 else 0.0,
        "wilcoxon_w": wilcoxon_stat,
        "wilcoxon_p_raw": wilcoxon_p,
    }


def tex_value(aggregate: Dict[str, Any], metric: str) -> str:
    if metric not in aggregate:
        return "--"
    item = aggregate[metric]
    mean = item["mean"]
    std = item.get("sample_std")
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def latex_table(variants: Sequence[Dict[str, Any]], contrasts: Dict[str, Any]) -> str:
    lines = [
        r"\begin{table*}[ht]",
        r"\caption{\RevisionMarker Pre-specified ablation study. Values are mean $\pm$ sample standard deviation across matched random seeds. II$<40$ is an empirical sequence-derived surrogate, not a thermodynamic stability measurement.}",
        r"\label{tab:ablation_audited}",
        r"\centering",
        r"\small",
        r"\RevisionTableRows",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Variant & II$<40$ (\%) & Mean II & ESM-2 pseudo-PPL & Uniqueness \\",
        r"\midrule",
    ]
    for variant in variants:
        agg = variant["aggregate"]
        lines.append(
            f"{variant['label']} & "
            f"{tex_value(agg, 'stable_rate_ii_lt_40_percent')} & "
            f"{tex_value(agg, 'mean_instability_index')} & "
            f"{tex_value(agg, 'esm2_pseudo_perplexity')} & "
            f"{tex_value(agg, 'uniqueness_ratio')} \\\\"
        )
    lines.extend([
        r"\bottomrule", r"\end{tabular}", r"\vspace{0.5em}",
        r"\RevisionTableRows", r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Paired contrast vs. full & $\Delta$ II$<40$ (95\% CI) & Cohen's $d_z$ & paired $p_{\mathrm{Holm}}$ & Wilcoxon $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ])
    metric = "stable_rate_ii_lt_40_percent"
    for variant in variants:
        if variant["id"] not in contrasts:
            continue
        item = contrasts[variant["id"]].get(metric)
        if not item:
            continue
        ci = item["paired_difference_ci95"]
        lines.append(
            f"{variant['label']} & {item['paired_difference_full_minus_variant']:.2f} "
            f"[{ci[0]:.2f}, {ci[1]:.2f}] & {item['cohen_dz']:.3f} & "
            f"{item['paired_t_p_holm']:.4g} & {item['wilcoxon_p_holm']:.4g} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="config/ablation_manifest.json")
    parser.add_argument("--out", default="results/ablation_study_summary.json")
    parser.add_argument("--tex-out", default="results/ablation_study_table.tex")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        parser.error("ablation manifest schema_version must be 1")
    specs = manifest.get("variants", [])
    ids = [item.get("id") for item in specs]
    if not specs or len(ids) != len(set(ids)):
        parser.error("manifest needs unique, non-empty variants")
    full_id = manifest.get("full_variant", "full")
    if full_id not in ids:
        parser.error(f"full_variant '{full_id}' is absent")

    expected_seeds = manifest.get("expected_seeds", [42, 123, 456, 789, 1337])
    expected_n = int(manifest.get("expected_sequences_per_seed", 1000))
    variants = [
        load_variant(manifest_path.parent, spec, expected_seeds, expected_n)
        for spec in specs
    ]
    sampling_signatures = {
        json.dumps(item["sampling_signature"], sort_keys=True) for item in variants
    }
    if len(sampling_signatures) != 1:
        parser.error("ablation variants do not share identical reportable sampling settings")
    dataset_hashes = {item["dataset_build_report_sha256"] for item in variants}
    code_commits = {item["code_commit"] for item in variants}
    if len(dataset_hashes) != 1:
        parser.error("ablation variants were evaluated against different dataset builds")
    if len(code_commits) != 1:
        parser.error("ablation variants were generated from different code commits")
    full = next(item for item in variants if item["id"] == full_id)
    metrics = manifest.get("metrics", DEFAULT_METRICS)
    contrasts: Dict[str, Any] = {}
    for variant in variants:
        if variant["id"] == full_id:
            continue
        contrasts[variant["id"]] = {}
        for metric in metrics:
            item = paired_contrast(full, variant, metric)
            if not item:
                continue
            contrasts[variant["id"]][metric] = item
    for key in ("paired_t_p_raw", "wilcoxon_p_raw"):
        p_refs: List[Dict[str, Any]] = []
        p_values: List[float] = []
        for variant_metrics in contrasts.values():
            for item in variant_metrics.values():
                p_refs.append(item)
                p_values.append(item[key])
        for item, adjusted in zip(p_refs, holm_adjust(p_values)):
            item[key.replace("_raw", "_holm")] = adjusted

    output = {
        "schema_version": 1,
        "reportable": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "full_variant": full_id,
        "variants": variants,
        "paired_contrasts": contrasts,
        "expected_seeds": expected_seeds,
        "expected_sequences_per_seed": expected_n,
        "multiple_testing_correction": (
            "Holm family-wise error rate within each pre-specified paired-test family"
        ),
    }
    out = Path(args.out)
    tex_out = Path(args.tex_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    tex_out.write_text(latex_table(variants, contrasts), encoding="utf-8")
    print(f"Wrote verified ablation artifacts: {out}, {tex_out}")


if __name__ == "__main__":
    main()
