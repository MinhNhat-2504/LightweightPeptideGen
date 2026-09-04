"""Fail-closed validation for dataset artifacts used by reportable runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recorded_path(value: str, report_file: Path) -> Path:
    """Resolve portable report-relative paths and legacy absolute paths."""
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (report_file.resolve().parent / path).resolve()


def validate_dataset_build(train_csv: str, validation_csv: str, test_csv: str,
                           report_path: str | None = None) -> Dict:
    paths = {
        "train": Path(train_csv), "validation": Path(validation_csv), "test": Path(test_csv),
    }
    report_file = Path(report_path) if report_path else paths["train"].parent / "dataset_build_report.json"
    if not report_file.exists():
        raise ValueError(
            f"missing auditable dataset build report: {report_file}. Rebuild with "
            "python -m peptidegen.data --manifest ... --cluster-tsv ..."
        )
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if not report.get("reportable"):
        raise ValueError(f"dataset build report is not reportable: {report_file}")
    metadata = report.get("manifest_metadata") or {}
    clustering = metadata.get("homology_clustering") or {}
    required_clustering = {
        "tool", "version", "command", "minimum_sequence_identity",
        "minimum_coverage", "coverage_mode", "representative_policy",
    }
    if required_clustering - set(clustering):
        raise ValueError(f"dataset report lacks complete homology-clustering provenance: {report_file}")

    for label, record in (
        ("manifest", report.get("manifest") or {}),
        ("cluster map", report.get("cluster_map") or {}),
        ("provenance ledger", report.get("ledger") or {}),
    ):
        artifact_path = recorded_path(str(record.get("path", "")), report_file)
        expected_hash = str(record.get("sha256", ""))
        if not artifact_path.is_file() or not expected_hash or sha256(artifact_path) != expected_hash:
            raise ValueError(f"{label} is absent or hash-mismatched in dataset report")

    for source, item in (report.get("preprocessing_funnel_by_source") or {}).items():
        source_path = recorded_path(str(item.get("source_file", "")), report_file)
        expected_hash = str(item.get("source_file_sha256", ""))
        if not source_path.is_file() or not expected_hash or sha256(source_path) != expected_hash:
            raise ValueError(f"source artifact '{source}' is absent or hash-mismatched")
        if not item.get("label_definition"):
            raise ValueError(f"source artifact '{source}' lacks an explicit label definition")
    artifacts = report.get("split_artifacts", {})
    frames = {}
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        expected = (artifacts.get(split) or {}).get("csv", {}).get("sha256")
        actual = sha256(path)
        if not expected or expected != actual:
            raise ValueError(f"{split} CSV does not match dataset build report: {path}")
        frame = pd.read_csv(path, usecols=lambda name: name in {"sequence", "label", "cluster_id"})
        required = {"sequence", "label", "cluster_id"}
        if set(frame.columns) != required:
            raise ValueError(f"{path}: reportable split requires columns {sorted(required)}")
        frames[split] = frame
        reported_count = ((report.get("counts") or {}).get(split) or {}).get("rows")
        if reported_count != len(frame):
            raise ValueError(f"{split} row count does not match dataset build report")

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        shared_sequences = set(frames[left].sequence) & set(frames[right].sequence)
        shared_clusters = set(frames[left].cluster_id) & set(frames[right].cluster_id)
        if shared_sequences or shared_clusters:
            raise ValueError(
                f"data leakage between {left}/{right}: {len(shared_sequences)} exact sequences, "
                f"{len(shared_clusters)} homology clusters"
            )
    combined = pd.concat(frames.values(), ignore_index=True)
    conflicts = combined.groupby("sequence")["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("dataset contains exact sequences with conflicting labels")
    return {
        "path": str(report_file.resolve()),
        "sha256": sha256(report_file),
        "report": report,
    }
