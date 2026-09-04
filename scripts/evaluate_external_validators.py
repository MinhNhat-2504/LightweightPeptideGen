#!/usr/bin/env python
"""Aggregate predictions exported by genuine third-party validators.

The script is intentionally *import-only*: it never substitutes a heuristic for
amPEPpy, ToxinPred3, or a hemolysis predictor, and it never fabricates baseline
sequences.  Each prediction CSV is matched exactly to its source FASTA before a
result is accepted.  See ``config/external_validation_manifest.example.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")
FNAME_RE = re.compile(r"(?P<model>.+?)_seed(?P<seed>\d+)$", re.IGNORECASE)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_fasta(path: Path) -> List[str]:
    sequences: List[str] = []
    buf: List[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith(">"):
                if buf:
                    sequences.append("".join(buf).upper())
                    buf = []
            elif line:
                buf.append(line)
    if buf:
        sequences.append("".join(buf).upper())
    bad = [s for s in sequences if not s or not set(s) <= CANONICAL]
    if bad:
        raise ValueError(f"{path}: {len(bad)} empty/non-canonical sequences")
    return sequences


def mean_std(values: Iterable[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) > 1:
        half = float(stats.t.ppf(0.975, len(arr) - 1) * stats.sem(arr))
        ci95 = [float(arr.mean() - half), float(arr.mean() + half)]
    else:
        ci95 = None
    return {
        "mean": float(arr.mean()),
        "sample_std": float(arr.std(ddof=1)) if len(arr) > 1 else None,
        "ci95": ci95,
        "n_seeds": int(len(arr)),
    }


def parse_run_id(stem: str) -> Tuple[str, int]:
    match = FNAME_RE.fullmatch(stem)
    if not match:
        raise ValueError(f"run id '{stem}' must match <model>_seed<N>")
    return match.group("model"), int(match.group("seed"))


def validate_manifest(manifest: Dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    validators = manifest.get("validators")
    if not isinstance(validators, list) or not validators:
        raise ValueError("manifest must define a non-empty validators list")
    for validator in validators:
        required = {
            "name", "version", "task", "used_for_training", "threshold",
            "positive_direction", "sequence_column", "score_column", "outputs",
            "citation", "command_or_service", "run_date",
            "training_data_or_independence_statement",
        }
        missing = required - set(validator)
        if missing:
            raise ValueError(f"validator entry missing {sorted(missing)}")
        if validator["used_for_training"]:
            raise ValueError(
                f"{validator['name']} is marked used_for_training=true and therefore is not an independent validator"
            )
        if validator["positive_direction"] not in {"higher", "lower"}:
            raise ValueError("positive_direction must be 'higher' or 'lower'")
        for field in required - {"used_for_training", "threshold", "outputs"}:
            value = str(validator.get(field, "")).strip()
            if not value or "REPLACE_WITH" in value:
                raise ValueError(f"{validator.get('name', 'validator')}: unresolved field '{field}'")


def evaluate(manifest: Dict[str, Any], manifest_dir: Path) -> Dict[str, Any]:
    validate_manifest(manifest)
    expected_seeds = {int(x) for x in manifest.get("expected_seeds", [42, 123, 456, 789, 1337])}
    expected_n = int(manifest.get("expected_sequences_per_seed", 1000))
    expected_models = set(manifest.get("expected_models", ["full"]))
    fasta_dir = (manifest_dir / manifest["generated_fasta_dir"]).resolve()
    fastas: Dict[str, Dict[str, Any]] = {}
    for path in sorted(fasta_dir.glob("*.fasta")):
        try:
            model, seed = parse_run_id(path.stem)
        except ValueError:
            continue
        seqs = read_fasta(path)
        if len(seqs) != expected_n:
            raise ValueError(f"{path}: expected {expected_n} sequences, found {len(seqs)}")
        sidecar_path = path.with_suffix(path.suffix + ".metadata.json")
        if not sidecar_path.is_file():
            raise ValueError(f"{path}: missing reportable generation sidecar")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        output = sidecar.get("output") or {}
        if (
            sidecar.get("reportable") is not True
            or output.get("sha256") != sha256(path)
            or sidecar.get("model_id") != model
            or int(sidecar.get("seed", -1)) != seed
        ):
            raise ValueError(f"{sidecar_path}: generation provenance is not reportable or hash-matched")
        fastas[path.stem] = {
            "model": model,
            "seed": seed,
            "path": str(path),
            "sha256": sha256(path),
            "metadata_path": str(sidecar_path.resolve()),
            "metadata_sha256": sha256(sidecar_path),
            "sequences": seqs,
        }
    if not fastas:
        raise ValueError(f"no <model>_seed<N>.fasta files found in {fasta_dir}")

    models = sorted({item["model"] for item in fastas.values()})
    if set(models) != expected_models:
        raise ValueError(f"expected models {sorted(expected_models)}, found {models}")
    for model in models:
        seeds = {item["seed"] for item in fastas.values() if item["model"] == model}
        if seeds != expected_seeds:
            raise ValueError(
                f"{model}: expected seeds {sorted(expected_seeds)}, found {sorted(seeds)}"
            )

    report: Dict[str, Any] = {
        "schema_version": 1,
        "reportable": True,
        "manifest_sha256": None,
        "expected_seeds": sorted(expected_seeds),
        "expected_sequences_per_seed": expected_n,
        "statistical_unit": "independent training seed",
        "interpretation": (
            "These are in silico screens from predictors unused in training; they are not "
            "experimental antimicrobial activity, hemolysis, cytotoxicity, or safety evidence."
        ),
        "validators": {},
    }
    for validator in manifest["validators"]:
        name = validator["name"]
        outputs = validator["outputs"]
        missing_runs = sorted(set(fastas) - set(outputs))
        extra_runs = sorted(set(outputs) - set(fastas))
        if missing_runs or extra_runs:
            raise ValueError(
                f"{name}: output mapping mismatch; missing={missing_runs}, extra={extra_runs}"
            )

        run_rows: List[Dict[str, Any]] = []
        threshold = float(validator["threshold"])
        higher = validator["positive_direction"] == "higher"
        for run_id, source in fastas.items():
            csv_path = (manifest_dir / outputs[run_id]).resolve()
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)
            frame = pd.read_csv(csv_path)
            seq_col = validator["sequence_column"]
            score_col = validator["score_column"]
            if seq_col not in frame or score_col not in frame:
                raise ValueError(f"{csv_path}: requires columns '{seq_col}' and '{score_col}'")
            predicted_sequences = frame[seq_col].astype(str).str.strip().str.upper().tolist()
            if Counter(predicted_sequences) != Counter(source["sequences"]):
                raise ValueError(
                    f"{csv_path}: prediction sequences do not exactly match {source['path']}"
                )
            scores = pd.to_numeric(frame[score_col], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(scores).all():
                raise ValueError(f"{csv_path}: non-finite prediction scores")
            positive = scores >= threshold if higher else scores <= threshold
            run_rows.append({
                "run_id": run_id,
                "model": source["model"],
                "seed": source["seed"],
                "n": int(len(scores)),
                "mean_score": float(scores.mean()),
                "positive_rate": float(positive.mean()),
                "prediction_path": str(csv_path),
                "prediction_sha256": sha256(csv_path),
                "fasta_path": source["path"],
                "fasta_sha256": source["sha256"],
            })

        aggregate: Dict[str, Any] = {}
        by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in run_rows:
            by_model[row["model"]].append(row)
        for model, rows in sorted(by_model.items()):
            aggregate[model] = {
                "mean_score": mean_std(r["mean_score"] for r in rows),
                "positive_rate": mean_std(r["positive_rate"] for r in rows),
            }
        report["validators"][name] = {
            "version": validator["version"],
            "task": validator["task"],
            "citation": validator.get("citation"),
            "command_or_service": validator.get("command_or_service"),
            "run_date": validator["run_date"],
            "training_data_or_independence_statement": validator[
                "training_data_or_independence_statement"
            ],
            "used_for_training": False,
            "threshold": threshold,
            "positive_direction": validator["positive_direction"],
            "runs": run_rows,
            "aggregate": aggregate,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="results/external_validation_report.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = evaluate(manifest, manifest_path.parent)
    report["manifest_path"] = str(manifest_path)
    report["manifest_sha256"] = sha256(manifest_path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote verified external-prediction report: {out}")


if __name__ == "__main__":
    main()
