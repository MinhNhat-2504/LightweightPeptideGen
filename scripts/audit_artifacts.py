#!/usr/bin/env python
"""Fail-closed inventory for every artifact needed by the major revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

EXPECTED_SEEDS = {42, 123, 456, 789, 1337}
RUN_RE = re.compile(r"(?P<model>.+)_seed(?P<seed>\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sidecar_path(value: str, metadata_path: Path) -> Path:
    """Resolve portable sidecar-relative paths and legacy absolute paths."""
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (metadata_path.resolve().parent / path).resolve()


def fasta_count(path: Path) -> int:
    with path.open(encoding="utf-8", errors="strict") as handle:
        return sum(line.startswith(">") for line in handle)


def checkpoint_record(path: Path) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("generator", checkpoint.get("generator_state_dict", {}))
        parameter_count = sum(int(tensor.numel()) for tensor in state.values() if torch.is_tensor(tensor))
        data = checkpoint.get("data_metadata") or {}
        scst = checkpoint.get("scst_config") or {}
        run = checkpoint.get("run_metadata") or {}
        warmup = checkpoint.get("warmup_config") or {}
        seed = scst.get("seed", run.get("seed", warmup.get("seed")))
        record = {
            "path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
            "readable": True, "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "generator_class": checkpoint.get("generator_class"),
            "generator_state_tensor_elements": parameter_count,
            "warmup": bool(checkpoint.get("warmup")), "scst": bool(checkpoint.get("scst")),
            "seed": seed,
            "reportable_data": data.get("reportable_data"),
            "scst_reportable_data": scst.get("reportable_data") if checkpoint.get("scst") else None,
            "has_training_history": bool(checkpoint.get("history")),
            "has_model_config": bool(checkpoint.get("model_config")),
            "has_run_command": bool(run.get("command") or warmup),
            "artifact_reportable": checkpoint.get("artifact_reportable") is True,
            "code_commit": (run.get("git") or {}).get("commit"),
            "clean_worktree_at_run": (run.get("git") or {}).get("dirty_worktree") is False,
        }
        problems = []
        if record["reportable_data"] is not True:
            problems.append("unverified dataset provenance")
        if record["scst"] and record["scst_reportable_data"] is not True:
            problems.append("SCST parent data not reportable")
        if seed not in EXPECTED_SEEDS:
            problems.append("missing/unexpected training seed")
        if not record["has_training_history"]:
            problems.append("missing training history")
        if not record["has_model_config"]:
            problems.append("missing model config")
        if not record["has_run_command"]:
            problems.append("missing run command/config metadata")
        if not record["artifact_reportable"]:
            problems.append("checkpoint not marked artifact_reportable")
        if not record["code_commit"] or not record["clean_worktree_at_run"]:
            problems.append("checkpoint lacks clean immutable code provenance")
        record["problems"] = problems
        return record
    except Exception as exc:
        return {
            "path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
            "readable": False, "error": str(exc), "problems": ["unreadable checkpoint"],
        }


def load_report(path: Path, issues: List[str], label: str) -> Dict[str, Any] | None:
    if not path.exists():
        issues.append(f"Missing {label}: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(f"Unreadable {label}: {path}: {exc}")
        return None
    if value.get("reportable") is not True:
        issues.append(f"{label} is not reportable: {path}")
    return value


def checkpoint_paths(root: Path) -> Iterable[Path]:
    seen = set()
    for base in (root / "checkpoints", root / "results/ablations"):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.pt")):
            if "unverified" in path.parts or path.resolve() in seen:
                continue
            seen.add(path.resolve())
            yield path


def resolve_dataset_dir(root: Path, explicit: str | None) -> Path:
    """Which dataset build the audit should check.

    Hard-coding dataset/rebuilt/ silently audits the wrong build once the runs move
    to a different directory, which is exactly the failure this auditor exists to
    prevent.  Precedence: --dataset-dir, then config/revision.yaml data.train_csv,
    then the historical default.
    """
    if explicit:
        return (root / explicit) if not Path(explicit).is_absolute() else Path(explicit)
    config = root / "config/revision.yaml"
    if config.exists():
        for line in config.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("train_csv:"):
                value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                if value:
                    return root / Path(value).parent
    return root / "dataset/rebuilt"


def validate_dataset(root: Path, issues: List[str],
                     dataset_dir: Path | None = None) -> Dict[str, Any] | None:
    data_dir = dataset_dir if dataset_dir is not None else root / "dataset/rebuilt"
    report_path = data_dir / "dataset_build_report.json"
    report = load_report(report_path, issues, "dataset build report")
    if report is None:
        return None
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.csv"
        expected = (((report.get("split_artifacts") or {}).get(split) or {}).get("csv") or {}).get("sha256")
        if not path.exists() or not expected or sha256(path) != expected:
            issues.append(f"{split} CSV is absent or does not match the dataset report")
    try:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from peptidegen.data.integrity import validate_dataset_build
        validate_dataset_build(
            str(data_dir / "train.csv"),
            str(data_dir / "validation.csv"),
            str(data_dir / "test.csv"),
            report_path=str(report_path),
        )
    except (ValueError, FileNotFoundError) as exc:
        issues.append(f"Dataset integrity/provenance validation failed: {exc}")
    return {"path": str(report_path.resolve()), "sha256": sha256(report_path), "contents": report}


def validate_generation(path: Path, issues: List[str]) -> Dict[str, Any]:
    match = RUN_RE.fullmatch(path.stem)
    record: Dict[str, Any] = {
        "path": str(path.resolve()), "sha256": sha256(path), "records": fasta_count(path),
        "model": match.group("model") if match else None,
        "seed": int(match.group("seed")) if match else None,
    }
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    if not match:
        issues.append(f"Generated FASTA name does not encode model/seed: {path}")
    if not sidecar.exists():
        issues.append(f"Missing generation sidecar: {sidecar}")
        return record
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        record["metadata_path"] = str(sidecar.resolve())
        record["metadata_sha256"] = sha256(sidecar)
        record["metadata_reportable"] = metadata.get("reportable")
        output = metadata.get("output") or {}
        if metadata.get("reportable") is not True:
            issues.append(f"Generation sidecar is non-reportable: {sidecar}")
        if output.get("sha256") != record["sha256"] or int(output.get("records", -1)) != record["records"]:
            issues.append(f"Generation sidecar output mismatch: {sidecar}")
        if match and int(metadata.get("seed", -1)) != record["seed"]:
            issues.append(f"Generation sidecar seed mismatch: {sidecar}")
        if match and metadata.get("model_id") != record["model"]:
            issues.append(f"Generation sidecar model_id mismatch: {sidecar}")
        checkpoint = metadata.get("checkpoint") or {}
        checkpoint_path = sidecar_path(str(checkpoint.get("path", "")), sidecar)
        if not checkpoint.get("sha256") or not checkpoint_path.exists() or sha256(checkpoint_path) != checkpoint["sha256"]:
            issues.append(f"Generation checkpoint is absent or hash-mismatched: {sidecar}")
        dataset = metadata.get("dataset_build_audit") or {}
        dataset_path = sidecar_path(str(dataset.get("path", "")), sidecar)
        if not dataset.get("sha256") or not dataset_path.exists() or sha256(dataset_path) != dataset["sha256"]:
            issues.append(f"Generation dataset report is absent or hash-mismatched: {sidecar}")
    except Exception as exc:
        issues.append(f"Unreadable generation sidecar: {sidecar}: {exc}")
    return record


def validate_ablation_matrix(root: Path, issues: List[str]) -> Dict[str, Any]:
    manifest_path = root / "config/ablation_manifest.json"
    if not manifest_path.exists():
        issues.append(f"Missing ablation manifest: {manifest_path}")
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = {"manifest": str(manifest_path.resolve()), "manifest_sha256": sha256(manifest_path), "variants": {}}
    for spec in manifest.get("variants", []):
        report_path = (manifest_path.parent / spec["benchmark_report"]).resolve()
        report = load_report(report_path, issues, f"ablation benchmark {spec['id']}")
        item = {"path": str(report_path), "exists": report is not None}
        if report:
            model = (report.get("per_model") or {}).get(spec["model_key"], {})
            runs = model.get("runs") or []
            seeds = {int(run.get("seed", -1)) for run in runs}
            counts = {int(run.get("n", -1)) for run in runs}
            item.update({"seeds": sorted(seeds), "counts": sorted(counts)})
            if seeds != EXPECTED_SEEDS or counts != {1000}:
                issues.append(
                    f"{spec['id']} benchmark needs seeds {sorted(EXPECTED_SEEDS)} and 1000 sequences/seed"
                )
        status["variants"][spec["id"]] = item
    return status


def validate_release(root: Path, issues: List[str]) -> Dict[str, Any] | None:
    path = root / "config/release_manifest.json"
    if not path.exists():
        issues.append("Missing final immutable release manifest/DOI (config/release_manifest.json).")
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "repository_url", "commit", "archive_doi", "code_license",
        "data_availability", "baseline_policy", "baseline_evidence",
        "environment_lock", "environment_lock_sha256",
    )
    for field in required:
        value = str(manifest.get(field, "")).strip()
        if not value or "REPLACE_WITH" in value:
            issues.append(f"Release manifest has unresolved field: {field}")
    allowed_baseline_policies = {
        "official_retraining", "inspired_controls_only", "literature_context_only",
    }
    if manifest.get("baseline_policy") not in allowed_baseline_policies:
        issues.append(
            "Release baseline_policy must be official_retraining, "
            "inspired_controls_only, or literature_context_only."
        )
    lock = root / str(manifest.get("environment_lock", ""))
    if not lock.exists():
        issues.append(f"Release environment lock is missing: {lock}")
    elif sha256(lock) != manifest.get("environment_lock_sha256"):
        issues.append(f"Release environment lock hash mismatch: {lock}")
    try:
        current_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        if manifest.get("commit") != current_commit:
            issues.append("Release manifest commit does not match checked-out commit.")
        if dirty:
            issues.append("Working tree is dirty; immutable release cannot be verified.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        issues.append("Could not resolve git commit/worktree state for release.")
    return {"path": str(path.resolve()), "sha256": sha256(path), "contents": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="results/artifact_audit.json")
    parser.add_argument("--dataset-dir", default=None,
                        help="dataset build to audit; defaults to config/revision.yaml "
                             "data.train_csv, then dataset/rebuilt")
    parser.add_argument("--require-complete", action="store_true",
                        help="exit 2 unless every submission artifact passes")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    issues: List[str] = []

    dataset_dir = resolve_dataset_dir(root, args.dataset_dir)
    dataset_status = validate_dataset(root, issues, dataset_dir)
    checkpoints = [checkpoint_record(path) for path in checkpoint_paths(root)]
    if not checkpoints:
        issues.append("No checkpoints found for audit.")
    for item in checkpoints:
        for problem in item.get("problems", []):
            issues.append(f"Checkpoint {problem}: {item['path']}")

    generated = []
    results_root = root / "results"
    if results_root.exists():
        for path in sorted(results_root.rglob("*.fasta")):
            if "unverified" not in path.parts:
                generated.append(validate_generation(path, issues))
    if not generated:
        issues.append("No reportable generated FASTA artifacts found.")

    ablations = validate_ablation_matrix(root, issues)
    # The AMP oracle lives beside the dataset build it was trained on; hard-coding
    # results/oracles/amp/ audits the oracle from the interim data instead.
    amp_report = next(
        (c for c in (
            root / f"results/oracles/amp_{dataset_dir.name.replace('rebuilt_', '')}/oracle_amp_report.json",
            root / f"results/oracles/amp__{dataset_dir.name}/oracle_amp_report.json",
            root / "results/oracles/amp/oracle_amp_report.json",
        ) if c.is_file()),
        root / "results/oracles/amp/oracle_amp_report.json",
    )
    # Hemolysis is NOT a reward oracle: the SCST hemolysis weight is fixed at zero and
    # hemolysis is assessed only at evaluation time by an independent held-out
    # predictor, which must never have taken part in optimisation or model selection.
    hemo_report = next(
        (c for c in (
            root / "results/hemolysis/hemolysis_oracle_report.json",
            root / "results/oracles/hemolysis/oracle_hemo_report.json",
        ) if c.is_file()),
        root / "results/hemolysis/hemolysis_oracle_report.json",
    )
    required_reports = {
        "AMP reward oracle": amp_report,
        "independent hemolysis evaluation predictor": hemo_report,
        "ablation summary": root / "results/ablation_study_summary.json",
        "controllability summary": root / "results/controllability_summary.json",
        "external validation": root / "results/external_validation_report.json",
        "ESMFold summary": root / "results/esmfold_summary.json",
    }
    result_reports = {
        label: load_report(path, issues, label) for label, path in required_reports.items()
    }
    release = validate_release(root, issues)
    release_commit = ((release or {}).get("contents") or {}).get("commit")
    if release_commit:
        for item in checkpoints:
            if item.get("code_commit") != release_commit:
                issues.append(
                    f"Checkpoint code commit differs from release commit: {item['path']}"
                )

    report = {
        "schema_version": 2,
        "root": str(root),
        "submission_ready": not issues,
        "expected_seeds": sorted(EXPECTED_SEEDS),
        "dataset_build": dataset_status,
        "checkpoints": checkpoints,
        "generated_fastas": generated,
        "ablation_matrix": ablations,
        "required_result_reports": {
            label: {
                "path": str(required_reports[label].resolve()),
                "present": value is not None,
                "sha256": sha256(required_reports[label]) if required_reports[label].exists() else None,
            }
            for label, value in result_reports.items()
        },
        "release": release,
        "issues": issues,
        "baseline_warning": (
            "The local baselines/hydramp and baselines/m3cad modules are architecture-inspired "
            "controls, not verified executions of the authors' official repositories."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}; submission_ready={report['submission_ready']}; issues={len(issues)}")
    if args.require_complete and issues:
        sys.exit(2)


if __name__ == "__main__":
    main()
