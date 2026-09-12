#!/usr/bin/env python
"""Export manuscript tables only after every revision artifact passes audit.

This is the sole supported route for replacing the red AUDIT HOLD boxes in the
LaTeX manuscript. It refuses incomplete/non-reportable JSON, an unresolved
release manifest, or a failed repository-wide artifact audit.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def load_report(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("reportable") is not True:
        raise ValueError(f"{label} is not marked reportable: {path}")
    return value


def esc(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def mean_sd(item: Dict[str, Any], scale: float = 1.0, digits: int = 2) -> str:
    mean = float(item["mean"]) * scale
    sd = item.get("sample_std")
    if sd is None:
        raise ValueError("five-seed manuscript values require sample_std")
    return f"{mean:.{digits}f} $\\pm$ {float(sd) * scale:.{digits}f}"


def dataset_table(report: Dict[str, Any]) -> str:
    lines = [
        r"\begin{table*}[ht]",
        r"\caption{\RevisionMarker Audited, sequential dataset-curation funnel. Counts at exact-deduplication and homology-representative stages are credited to the deterministic representative source.}",
        r"\label{tab:dataset-provenance}", r"\centering\small",
        r"\RevisionTableRows",
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Source & Label & Raw & Length & Canonical & Evidence & Exact unique & Homology reps. \\",
        r"\midrule",
    ]
    for source, item in report["preprocessing_funnel_by_source"].items():
        lines.append(
            f"{esc(source)} & {item['label']} & {item['raw_records']} & "
            f"{item['length_5_50']} & {item['canonical_after_length']} & "
            f"{item['evidence_accepted']} & {item['exact_dedup_representatives']} & "
            f"{item['homology_cluster_representatives']} \\\\"
        )
    item = report["preprocessing_funnel_global"]
    lines.extend([
        r"\midrule",
        f"Total & -- & {item['raw_records']} & {item['length_5_50']} & "
        f"{item['canonical_after_length']} & {item['evidence_accepted']} & "
        f"{item['exact_dedup_representatives']} & {item['homology_cluster_representatives']} \\\\ ",
        r"\bottomrule", r"\end{tabular}", r"\end{table*}",
    ])
    return "\n".join(lines)


def benchmark_table(report: Dict[str, Any]) -> str:
    lines = [
        r"\begin{table*}[ht]",
        r"\caption{\RevisionMarker De novo generation benchmark across five independent training and generation seeds. Values are mean $\pm$ sample standard deviation. II$<40$ is an empirical sequence screen, not a thermodynamic-stability measurement.}",
        r"\label{tab:main-benchmark}", r"\centering\small",
        r"\RevisionTableRows",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Model & Valid (\%) & Novel (\%) & Unique (\%) & II$<40$ (\%) & Mean II & ESM-2 pseudo-PPL \\",
        r"\midrule",
    ]
    for model, value in report["per_model"].items():
        agg = value["aggregate"]
        lines.append(
            f"{esc(model)} & {mean_sd(agg['validity_ratio'], 100)} & "
            f"{mean_sd(agg['exact_novelty_ratio_vs_training'], 100)} & "
            f"{mean_sd(agg['uniqueness_ratio'], 100)} & "
            f"{mean_sd(agg['stable_rate_ii_lt_40_percent'])} & "
            f"{mean_sd(agg['mean_instability_index'])} & "
            f"{mean_sd(agg['esm2_pseudo_perplexity'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def external_table(report: Dict[str, Any]) -> str:
    lines = [
        r"\begin{table*}[ht]",
        r"\caption{\RevisionMarker Independent in silico screens unused in training or model selection. Screen-positive rates are predictor outputs and are not experimental activity, toxicity, hemolysis, or safety evidence.}",
        r"\label{tab:external-validation}", r"\centering\small",
        r"\RevisionTableRows",
        r"\begin{tabular}{llllrr}", r"\toprule",
        r"Validator & Task & Version & Model & Mean score & Screen-positive (\%) \\",
        r"\midrule",
    ]
    for name, validator in report["validators"].items():
        for model, agg in validator["aggregate"].items():
            lines.append(
                f"{esc(name)} & {esc(validator['task'])} & {esc(validator['version'])} & "
                f"{esc(model)} & {mean_sd(agg['mean_score'], digits=3)} & "
                f"{mean_sd(agg['positive_rate'], 100)} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def resolve_dataset_dir(root: Path, explicit: str | None) -> Path:
    """Which dataset build to export numbers from.

    Hard-coding dataset/rebuilt/ silently exports the interim build - the one whose
    manifest declares itself NOT FOR SUBMISSION - into the manuscript, and reports
    success while doing it.
    """
    if explicit:
        return Path(explicit) if Path(explicit).is_absolute() else root / explicit
    config = root / "config/revision.yaml"
    if config.exists():
        for line in config.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("train_csv:"):
                value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                if value:
                    return root / Path(value).parent
    return root / "dataset/rebuilt"


def resolve_benchmark(root: Path, dataset_dir: Path, explicit: str | None) -> Path:
    """Main benchmark for the run that used `dataset_dir`.

    The drivers tag their output directory with the dataset directory name so that
    a rerun on new data cannot overwrite or be confused with an earlier run.
    """
    if explicit:
        return Path(explicit) if Path(explicit).is_absolute() else root / explicit
    tagged = root / f"results/ablations/full__{dataset_dir.name}/benchmark.json"
    if tagged.is_file():
        return tagged
    return root / "results/ablations/full/benchmark.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--paper-dir", required=True)
    parser.add_argument("--dataset-dir", default=None,
                        help="dataset build to export from; defaults to "
                             "config/revision.yaml data.train_csv")
    parser.add_argument("--benchmark", default=None,
                        help="path to the main benchmark.json; defaults to the run "
                             "directory matching the dataset directory")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    paper_dir = Path(args.paper_dir).resolve()
    table_dir = paper_dir / "generated_tables"

    audit_path = root / "results/artifact_audit.json"
    subprocess.run([
        sys.executable, str(root / "scripts/audit_artifacts.py"),
        "--root", str(root), "--out", str(audit_path), "--require-complete",
        "--dataset-dir", str(resolve_dataset_dir(root, args.dataset_dir)),
    ], check=True)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("submission_ready") is not True:
        raise ValueError("artifact audit did not mark the repository submission-ready")

    dataset_dir = resolve_dataset_dir(root, args.dataset_dir)
    benchmark_path = resolve_benchmark(root, dataset_dir, args.benchmark)
    print(f"dataset build : {dataset_dir}")
    print(f"main benchmark: {benchmark_path}")
    dataset = load_report(dataset_dir / "dataset_build_report.json", "dataset report")
    benchmark = load_report(benchmark_path, "main benchmark")
    external = load_report(root / "results/external_validation_report.json", "external validation")
    esmfold = load_report(root / "results/esmfold_summary.json", "ESMFold summary")
    load_report(root / "results/controllability_summary.json", "controllability summary")
    load_report(root / "results/ablation_study_summary.json", "ablation summary")

    release_path = root / "config/release_manifest.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if release.get("commit") != current_commit:
        raise ValueError("release-manifest commit does not equal the checked-out commit")

    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "dataset_provenance.tex").write_text(dataset_table(dataset), encoding="utf-8")
    (table_dir / "main_benchmark.tex").write_text(benchmark_table(benchmark), encoding="utf-8")
    (table_dir / "external_validation.tex").write_text(external_table(external), encoding="utf-8")
    for source, destination in (
        (root / "results/esmfold_summary.tex", table_dir / "esmfold_summary.tex"),
        (root / "results/controllability.tex", table_dir / "controllability.tex"),
        (root / "results/ablation_study_table.tex", table_dir / "ablation_study_table.tex"),
    ):
        if not source.is_file():
            raise ValueError(f"missing generated LaTeX source: {source}")
        shutil.copyfile(source, destination)

    full = benchmark["per_model"].get("full")
    if full is None:
        raise ValueError("main benchmark lacks model key 'full'")
    agg = full["aggregate"]
    plddt = esmfold["models"]["full"]["aggregate"]["mean_plddt"]
    abstract = (
        "Across five independent runs, the model achieved an II-screen-pass rate of "
        f"{mean_sd(agg['stable_rate_ii_lt_40_percent'])}\\%, uniqueness of "
        f"{mean_sd(agg['uniqueness_ratio'], 100)}\\%, and mean ESMFold pLDDT of "
        f"{mean_sd(plddt)}."
    )
    verified = "\n".join([
        r"\resultsverifiedtrue",
        r"\renewcommand{\VerifiedAbstractResults}{" + abstract + "}",
        r"\renewcommand{\VerifiedReleaseStatement}{An immutable code and artifact release is available at \url{" +
        esc(release["repository_url"]) + r"} (commit \texttt{" + esc(release["commit"]) +
        r"}; archive DOI: \url{https://doi.org/" + esc(release["archive_doi"]) + r"}).}",
    ])
    (table_dir / "verified_results.tex").write_text(verified, encoding="utf-8")
    print(f"Exported verified manuscript tables to {table_dir}")


if __name__ == "__main__":
    main()
