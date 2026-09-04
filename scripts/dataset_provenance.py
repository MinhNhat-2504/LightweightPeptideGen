#!/usr/bin/env python
"""Audit dataset lineage from the files that actually exist.

This command deliberately does not contain manuscript counts.  It computes
split statistics, checks leakage/conflicting labels, records SHA-256 hashes,
and summarizes the source information present in the ``id`` column.  A source
funnel is emitted only when a row-level processing ledger is supplied.

The optional ledger must contain these columns:

``source, sequence, passes_length, passes_canonical, dedup_representative,
evidence_pass, homology_representative``.

Boolean stage columns are cumulative decisions recorded by the curation
pipeline. They are required because evidence filtering and homology membership
cannot be reconstructed from final split files.

Examples
--------
python scripts/dataset_provenance.py
python scripts/dataset_provenance.py --ledger dataset/provenance_ledger.csv
python scripts/dataset_provenance.py --allow-issues
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")
REQUIRED_SPLIT_COLUMNS = {"sequence", "label"}
LEDGER_COLUMNS = {
    "source",
    "sequence",
    "passes_length",
    "passes_canonical",
    "dedup_representative",
    "evidence_pass",
    "homology_representative",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_bool(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    values = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"ledger column '{column}' has non-boolean values: {unknown[:5]}")
    return values.isin({"true", "1", "yes"})


def source_from_id(value: Any, label: int) -> str:
    """Infer only what the stored ID explicitly reveals; never guess silently."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "missing-id"
    text = str(value).strip()
    method = re.search(r"sampling_method=([^_]+)", text)
    if method:
        return f"negative-sampling:{method.group(1)}"
    upper = text.upper()
    if upper.startswith("DRAMP"):
        return "DRAMP"
    if upper.startswith("DBAMP"):
        return "dbAMP"
    if upper.startswith("DBAASP"):
        return "DBAASP"
    if re.fullmatch(r"AP\d+", upper):
        return "APD-like-id"
    if text.startswith("sp|"):
        return "UniProtKB/Swiss-Prot"
    if text.startswith("tr|"):
        return "UniProtKB/TrEMBL"
    return "unresolved-positive-id" if int(label) == 1 else "unresolved-negative-id"


def audit_splits(paths: Dict[str, Path]) -> Dict[str, Any]:
    frames: List[pd.DataFrame] = []
    split_report: Dict[str, Any] = {}
    issues: List[Dict[str, Any]] = []

    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        missing = REQUIRED_SPLIT_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        frame = frame.copy()
        frame["split"] = split
        frame["sequence"] = frame["sequence"].astype(str).str.strip().str.upper()
        frame["label"] = frame["label"].astype(int)
        frames.append(frame)

        valid_alphabet = frame["sequence"].map(lambda s: bool(s) and set(s) <= CANONICAL)
        valid_length = frame["sequence"].str.len().between(5, 50)
        duplicate_mask = frame.duplicated("sequence", keep=False)
        split_report[split] = {
            "path": str(path),
            "sha256": sha256(path),
            "rows": int(len(frame)),
            "unique_sequences": int(frame["sequence"].nunique()),
            "labels": {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()},
            "invalid_alphabet_rows": int((~valid_alphabet).sum()),
            "outside_length_5_50_rows": int((~valid_length).sum()),
            "within_split_duplicate_rows": int(duplicate_mask.sum()),
        }

    combined = pd.concat(frames, ignore_index=True)
    overlap: Dict[str, Any] = {}
    names = list(paths)
    for i, left in enumerate(names):
        left_set = set(combined.loc[combined["split"] == left, "sequence"])
        for right in names[i + 1 :]:
            right_set = set(combined.loc[combined["split"] == right, "sequence"])
            shared = sorted(left_set & right_set)
            overlap[f"{left}__{right}"] = {"count": len(shared), "sequences": shared}
            if shared:
                issues.append({"type": "cross_split_exact_overlap", "splits": [left, right], "sequences": shared})

    label_counts = combined.groupby("sequence")["label"].nunique()
    conflicts = sorted(label_counts[label_counts > 1].index.tolist())
    if conflicts:
        rows = combined.loc[
            combined["sequence"].isin(conflicts), ["split", "sequence", "id", "label"]
        ].where(pd.notnull(combined), None)
        issues.append({"type": "conflicting_labels", "rows": rows.to_dict("records")})

    duplicate_rows = int(combined.duplicated(["sequence", "label"], keep=False).sum())
    invalid_labels = sorted(set(combined["label"]) - {0, 1})
    if invalid_labels:
        issues.append({"type": "invalid_labels", "values": invalid_labels})

    source_counts: Dict[str, Dict[str, int]] = {}
    if "id" in combined.columns:
        combined["source_evidence"] = [
            source_from_id(value, label) for value, label in zip(combined["id"], combined["label"])
        ]
        for label, group in combined.groupby("label"):
            source_counts[str(int(label))] = {
                str(k): int(v) for k, v in group["source_evidence"].value_counts().items()
            }
        embedded_labels = combined["id"].astype(str).str.extract(r"(?:^|_)AMP=([01])(?:_|$)")[0]
        has_embedded_label = embedded_labels.notna()
        embedded_numeric = pd.to_numeric(embedded_labels, errors="coerce")
        embedded_conflict = has_embedded_label & (embedded_numeric != combined["label"])
        if embedded_conflict.any():
            issues.append({
                "type": "stored_label_conflicts_with_id_embedded_amp_label",
                "rows": int(embedded_conflict.sum()),
                "explanation": (
                    "AMPBenchmark-style IDs encode AMP=0/1, but the CSV label disagrees. "
                    "These rows require reconstruction from the source dataset before training."
                ),
            })
        missing_ids = int(combined["id"].isna().sum())
        if missing_ids:
            issues.append({"type": "missing_source_ids", "rows": missing_ids})
    else:
        issues.append({"type": "missing_id_column", "rows": int(len(combined))})

    return {
        "splits": split_report,
        "combined": {
            "rows": int(len(combined)),
            "unique_sequences": int(combined["sequence"].nunique()),
            "duplicate_rows_same_label": duplicate_rows,
            "conflicting_label_sequences": len(conflicts),
        },
        "cross_split_overlap": overlap,
        "source_evidence_from_stored_ids": source_counts,
        "issues": issues,
    }


def ledger_funnel(path: Path) -> Dict[str, Any]:
    ledger = pd.read_csv(path)
    if "homology_representative" not in ledger.columns and "cdhit_representative" in ledger.columns:
        ledger = ledger.rename(columns={"cdhit_representative": "homology_representative"})
    missing = LEDGER_COLUMNS - set(ledger.columns)
    if missing:
        raise ValueError(f"{path}: missing ledger columns {sorted(missing)}")
    ledger = ledger.copy()
    ledger["sequence"] = ledger["sequence"].astype(str).str.strip().str.upper()
    for column in LEDGER_COLUMNS - {"source", "sequence"}:
        ledger[column] = as_bool(ledger[column], column)

    # Enforce a cumulative funnel. A later-stage pass without every earlier
    # stage is a provenance error rather than a count to be silently accepted.
    stages = [
        ("raw", pd.Series(True, index=ledger.index)),
        ("length_5_50", ledger["passes_length"]),
        ("canonical_20aa", ledger["passes_canonical"]),
        ("evidence_filtered", ledger["evidence_pass"]),
        ("exact_deduplicated", ledger["dedup_representative"]),
        ("homology_representative", ledger["homology_representative"]),
    ]
    previous = pd.Series(True, index=ledger.index)
    counts: Dict[str, Dict[str, int]] = {}
    violations: List[str] = []
    for stage, flag in stages:
        flag = flag & previous
        if stage != "raw":
            raw_flag = dict(stages)[stage]
            if (raw_flag & ~previous).any():
                violations.append(stage)
        counts[stage] = {
            **{str(k): int(v) for k, v in ledger.loc[flag, "source"].value_counts().items()},
            "TOTAL": int(flag.sum()),
        }
        previous = flag
    if violations:
        raise ValueError(f"non-cumulative ledger stage decisions: {violations}")
    return {"path": str(path), "sha256": sha256(path), "counts": counts}


def latex_summary(report: Dict[str, Any]) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\caption{Machine-audited integrity of the processed dataset splits.}",
        r"\label{tab:dataset_integrity_audit}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Split & Rows & Unique sequences & AMP & non-AMP \\",
        r"\midrule",
    ]
    for split, item in report["processed_dataset"]["splits"].items():
        labels = item["labels"]
        lines.append(
            f"{split.title()} & {item['rows']:,} & {item['unique_sequences']:,} & "
            f"{labels.get('1', 0):,} & {labels.get('0', 0):,} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="dataset/train.csv")
    parser.add_argument("--val", default="dataset/val.csv")
    parser.add_argument("--test", default="dataset/test.csv")
    parser.add_argument("--ledger", default=None, help="row-level source and preprocessing ledger")
    parser.add_argument("--out", default="results/data_provenance_audit.json")
    parser.add_argument("--tex-out", default="results/data_provenance_audit.tex")
    parser.add_argument(
        "--allow-issues",
        action="store_true",
        help="write the audit but return success even if leakage/provenance issues are found",
    )
    args = parser.parse_args()

    split_paths = {
        "train": Path(args.train),
        "validation": Path(args.val),
        "test": Path(args.test),
    }
    report: Dict[str, Any] = {
        "schema_version": 1,
        "generated_by": "scripts/dataset_provenance.py",
        "processed_dataset": audit_splits(split_paths),
        "source_funnel": None,
    }
    if args.ledger:
        report["source_funnel"] = ledger_funnel(Path(args.ledger))
    else:
        report["processed_dataset"]["issues"].append({
            "type": "missing_row_level_provenance_ledger",
            "explanation": (
                "Database-specific attrition, evidence filtering, and homology counts cannot "
                "be verified from the final split CSV files."
            ),
        })

    out = Path(args.out)
    tex_out = Path(args.tex_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    tex_out.write_text(latex_summary(report), encoding="utf-8")

    issues = report["processed_dataset"]["issues"]
    print(f"Wrote {out} and {tex_out}")
    print(f"Audit issues: {len(issues)}")
    for issue in issues:
        print(f"- {issue['type']}")
    if issues and not args.allow_issues:
        sys.exit(2)


if __name__ == "__main__":
    main()
