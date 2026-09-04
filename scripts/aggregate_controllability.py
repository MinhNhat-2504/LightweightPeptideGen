#!/usr/bin/env python
"""Aggregate reportable controllability sweeps across independent training seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: List[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=float)
    sem = stats.sem(array)
    half = float(stats.t.ppf(0.975, len(array) - 1) * sem) if len(array) > 1 else None
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else None,
        "ci95": [float(array.mean() - half), float(array.mean() + half)] if half is not None else None,
        "n_seeds": int(len(array)),
    }


def tex_value(item: Dict[str, Any]) -> str:
    return f"{item['mean']:.3f} $\\pm$ {item['sample_std']:.3f}"


def latex(report: Dict[str, Any]) -> str:
    lines = [
        r"\begin{table*}[ht]",
        r"\caption{\RevisionMarker Conditional controllability across five independent training seeds. Values are mean $\pm$ sample standard deviation; the requested targets remain within the AMP training range.}",
        r"\label{tab:controllability_audited}",
        r"\centering\small",
        r"\RevisionTableRows",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        "Condition & Spearman $\\rho$ & Pearson $r$ & MAE (raw units) \\\\",
        r"\midrule",
    ]
    for feature, values in report["features"].items():
        label = feature.replace("_", r"\_")
        lines.append(
            f"{label} & {tex_value(values['spearman'])} & "
            f"{tex_value(values['pearson'])} & {tex_value(values['mae'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="results/controllability")
    parser.add_argument("--model", default="full")
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1337])
    parser.add_argument("--out", default="results/controllability_summary.json")
    parser.add_argument("--tex-out", default="results/controllability.tex")
    args = parser.parse_args()

    paths = [Path(args.input_dir) / f"{args.model}_seed{seed}.json" for seed in args.expected_seeds]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        parser.error(f"missing controllability runs: {missing}")

    reports = []
    for expected_seed, path in zip(args.expected_seeds, paths):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("reportable") is not True or int(report.get("seed", -1)) != expected_seed:
            parser.error(f"{path}: not reportable or seed mismatch")
        reports.append((path, report))

    feature_sets = [set(report["results"]) for _, report in reports]
    if any(features != feature_sets[0] for features in feature_sets[1:]):
        parser.error("condition-feature sets differ across seeds")
    train_hashes = {
        report.get("provenance", {}).get("train_csv_sha256") for _, report in reports
    }
    if None in train_hashes or len(train_hashes) != 1:
        parser.error("training CSV provenance differs across controllability runs")

    values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for _, report in reports:
        for feature, result in report["results"].items():
            for metric in ("spearman", "pearson", "mae"):
                value = float(result[metric])
                if not np.isfinite(value):
                    parser.error(f"non-finite {feature}.{metric}")
                values[feature][metric].append(value)
    output = {
        "schema_version": 1,
        "reportable": True,
        "model": args.model,
        "seeds": args.expected_seeds,
        "statistical_unit": "independent training seed",
        "train_csv_sha256": next(iter(train_hashes)),
        "source_reports": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path, _ in reports
        ],
        "features": {
            feature: {metric: summarize(metric_values) for metric, metric_values in feature_values.items()}
            for feature, feature_values in sorted(values.items())
        },
    }
    out, tex_out = Path(args.out), Path(args.tex_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    tex_out.write_text(latex(output), encoding="utf-8")
    print(f"Wrote verified controllability artifacts: {out}, {tex_out}")


if __name__ == "__main__":
    main()
