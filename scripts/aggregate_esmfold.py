#!/usr/bin/env python
"""Aggregate ESMFold confidence at the random-seed level.

Input CSVs and sidecars must be produced by ``esmfold_plddt.py`` and named
``<model>_seed<N>.csv``.  Exact sequence multisets are checked against the
corresponding generated FASTA before any result is accepted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

RUN_RE = re.compile(r"(?P<model>.+)_seed(?P<seed>\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fasta_sequences(path: Path) -> List[str]:
    result, buffer = [], []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith(">"):
                if buffer:
                    result.append("".join(buffer).upper())
                    buffer = []
            elif line:
                buffer.append(line)
    if buffer:
        result.append("".join(buffer).upper())
    return result


def summarize(values: List[float]) -> Dict:
    array = np.asarray(values, dtype=float)
    if len(array) > 1:
        sem = stats.sem(array)
        half = float(stats.t.ppf(0.975, len(array) - 1) * sem)
        std = float(array.std(ddof=1))
        ci = [float(array.mean() - half), float(array.mean() + half)]
    else:
        std, ci = None, None
    return {"mean": float(array.mean()), "sample_std": std, "ci95": ci, "n_seeds": len(array)}


def latex(report: Dict) -> str:
    lines = [
        r"\begin{table}[ht]", r"\caption{\RevisionMarker ESMFold prediction-confidence analysis. pLDDT is not a thermodynamic-stability measurement.}",
        r"\label{tab:esmfold_confidence}", r"\centering", r"\small",
        r"\RevisionTableRows", r"\begin{tabular}{lrr}", r"\toprule",
        r"Model & Mean pLDDT & Fraction pLDDT$\geq70$ (\%) \\", r"\midrule",
    ]
    for model, item in report["models"].items():
        plddt = item["aggregate"]["mean_plddt"]
        fraction = item["aggregate"]["fraction_plddt_ge_70_percent"]
        lines.append(
            f"{model} & {plddt['mean']:.2f} $\\pm$ {plddt['sample_std']:.2f} & "
            f"{fraction['mean']:.2f} $\\pm$ {fraction['sample_std']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--fasta-dir", required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1337])
    parser.add_argument("--expected-n", type=int, default=1000)
    parser.add_argument("--expected-models", nargs="+", default=["full"])
    parser.add_argument("--out", default="results/esmfold_summary.json")
    parser.add_argument("--tex-out", default="results/esmfold_summary.tex")
    args = parser.parse_args()

    csv_dir, fasta_dir = Path(args.csv_dir), Path(args.fasta_dir)
    runs = defaultdict(list)
    for csv_path in sorted(csv_dir.glob("*_seed*.csv")):
        match = RUN_RE.fullmatch(csv_path.stem)
        if not match:
            continue
        model, seed = match.group("model"), int(match.group("seed"))
        fasta_path = fasta_dir / f"{csv_path.stem}.fasta"
        sidecar_path = csv_path.with_suffix(".metadata.json")
        if not fasta_path.exists() or not sidecar_path.exists():
            parser.error(f"missing FASTA or metadata for {csv_path}")
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not metadata.get("reportable"):
            parser.error(f"{sidecar_path} is marked reportable=false")
        if metadata.get("seed") != seed:
            parser.error(f"seed mismatch in {sidecar_path}")
        if metadata.get("input_fasta_sha256") != sha256(fasta_path):
            parser.error(f"FASTA hash mismatch for {csv_path}")
        if metadata.get("output_csv_sha256") != sha256(csv_path):
            parser.error(f"CSV hash mismatch for {csv_path}")
        if not metadata.get("model_revision"):
            parser.error(f"{sidecar_path}: missing immutable model revision")
        if not metadata.get("command"):
            parser.error(f"{sidecar_path}: missing execution command")
        git = metadata.get("git") or {}
        if not git.get("commit") or git.get("dirty_worktree") is not False:
            parser.error(f"{sidecar_path}: ESMFold was not run from a clean immutable commit")
        generation_metadata = Path(str(metadata.get("generation_metadata_path", "")))
        if (
            not generation_metadata.is_file()
            or metadata.get("generation_metadata_sha256") != sha256(generation_metadata)
        ):
            parser.error(f"{sidecar_path}: generation metadata is absent or hash-mismatched")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        sequences = [row["sequence"].upper() for row in rows]
        source_sequences = fasta_sequences(fasta_path)
        if Counter(sequences) != Counter(source_sequences):
            parser.error(f"sequence multiset mismatch for {csv_path}")
        if len(rows) != args.expected_n:
            parser.error(f"{csv_path}: expected {args.expected_n} rows, found {len(rows)}")
        values = np.asarray([float(row["mean_plddt"]) for row in rows])
        runs[model].append({
            "seed": seed, "n": len(rows), "mean_plddt": float(values.mean()),
            "fraction_plddt_ge_70_percent": float((values >= 70).mean() * 100),
            "csv_path": str(csv_path.resolve()), "csv_sha256": sha256(csv_path),
            "metadata_path": str(sidecar_path.resolve()), "metadata_sha256": sha256(sidecar_path),
        })
    if not runs:
        parser.error("no ESMFold CSVs found")
    if set(runs) != set(args.expected_models):
        parser.error(f"expected models {sorted(args.expected_models)}, found {sorted(runs)}")
    expected = set(args.expected_seeds)
    revisions = set()
    software_versions = set()
    report = {
        "schema_version": 1, "reportable": True,
        "interpretation": "pLDDT is ESMFold prediction confidence, not thermodynamic stability.",
        "expected_seeds": args.expected_seeds, "expected_sequences_per_seed": args.expected_n,
        "models": {},
    }
    for model, model_runs in sorted(runs.items()):
        if {row["seed"] for row in model_runs} != expected:
            parser.error(f"{model}: incomplete seed set")
        for row in model_runs:
            sidecar = json.loads(Path(row["metadata_path"]).read_text(encoding="utf-8"))
            revisions.add((sidecar.get("model"), sidecar.get("model_revision")))
            software_versions.add((sidecar.get("torch_version"), sidecar.get("transformers_version")))
        report["models"][model] = {
            "runs": sorted(model_runs, key=lambda row: row["seed"]),
            "aggregate": {
                "mean_plddt": summarize([row["mean_plddt"] for row in model_runs]),
                "fraction_plddt_ge_70_percent": summarize([
                    row["fraction_plddt_ge_70_percent"] for row in model_runs
                ]),
            },
        }
    if len(revisions) != 1 or len(software_versions) != 1:
        parser.error("ESMFold model revision/software versions differ across seed runs")
    report["model_revision"] = list(revisions)[0]
    report["software_versions"] = list(software_versions)[0]
    out, tex_out = Path(args.out), Path(args.tex_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tex_out.write_text(latex(report), encoding="utf-8")
    print(f"Wrote {out} and {tex_out}")


if __name__ == "__main__":
    main()
