#!/usr/bin/env python
"""Build auditable peptide splits from an explicit source manifest.

The previous command inferred AMP labels from filenames and silently removed
duplicates.  That cannot support a provenance statement.  This implementation
requires the source, label policy, version/retrieval metadata, and evidence
policy to be declared for every input.  It writes a row-level ledger and keeps
homology clusters within a single split.

See ``config/dataset_manifest.example.json`` for the manifest schema.  Run the
command once with ``--prepare-clustering-fasta`` and cluster that FASTA with the
declared MMseqs2 command. A two-column cluster TSV is then mandatory for
a reportable build.  Native MMseqs2 ``representative<TAB>member`` output is
accepted with ``--cluster-format mmseqs_rep_member``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

from .features import PeptideFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")
REQUIRED_SOURCE_FIELDS = {
    "name", "path", "format", "label", "version", "retrieval_date",
    "url", "license", "evidence_policy", "label_definition", "sha256",
}
REQUIRED_CLUSTER_FIELDS = {
    "tool", "version", "command", "minimum_sequence_identity",
    "minimum_coverage", "coverage_mode", "representative_policy",
}
REQUIRED_MANIFEST_FIELDS = {"dataset_name", "curation_protocol", "homology_clustering", "sources"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path, base: Path) -> str:
    """Return an archive-portable path relative to the report directory."""
    try:
        return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()
    except ValueError:
        # Paths on different Windows drives cannot be made relative.
        return str(path.resolve())


def fasta_rows(path: Path) -> Iterator[Tuple[str, str, Dict[str, str]]]:
    record_id = None
    sequence: List[str] = []
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if record_id is not None:
                    yield record_id, "".join(sequence), {}
                record_id = line[1:].strip()
                sequence = []
            elif line:
                if record_id is None:
                    raise ValueError(f"{path}: FASTA sequence before first header")
                sequence.append(line)
    if record_id is not None:
        yield record_id, "".join(sequence), {}


def csv_rows(path: Path, sequence_column: str, id_column: str) -> Iterator[Tuple[str, str, Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig", errors="strict") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or sequence_column not in reader.fieldnames:
            raise ValueError(f"{path}: missing sequence column '{sequence_column}'")
        for index, row in enumerate(reader, start=2):
            record_id = row.get(id_column) or f"row-{index}"
            yield str(record_id), str(row[sequence_column]), row


def evidence_passes(source: Dict, raw: Dict[str, str]) -> bool:
    """Apply a declared row-level evidence rule when one is configured."""
    column = source.get("evidence_column")
    accepted = source.get("evidence_accept")
    if column is None:
        # A pre-filtered release is acceptable only when declared explicitly;
        # the declaration and file hash remain in the ledger/build report.
        return source["evidence_policy"] in {
            "experimentally_validated_only", "curated_positive_release",
            "not_applicable_negative",
        }
    if accepted is None:
        raise ValueError(f"source {source['name']}: evidence_column requires evidence_accept")
    return str(raw.get(column, "")).strip().lower() in {
        str(value).strip().lower() for value in accepted
    }


def load_manifest(path: Path) -> Tuple[Dict, List[Dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("dataset manifest schema_version must be 1")
    missing_manifest = REQUIRED_MANIFEST_FIELDS - set(manifest)
    if missing_manifest:
        raise ValueError(f"dataset manifest missing fields: {sorted(missing_manifest)}")
    for key in ("dataset_name", "curation_protocol"):
        value = str(manifest[key]).strip()
        if not value or "REPLACE_WITH" in value:
            raise ValueError(f"dataset manifest has unresolved field '{key}'")
    sources = manifest.get("sources") or []
    if not sources:
        raise ValueError("dataset manifest contains no sources")
    clustering = manifest.get("homology_clustering") or {}
    missing_clustering = REQUIRED_CLUSTER_FIELDS - set(clustering)
    if missing_clustering:
        raise ValueError(
            "manifest homology_clustering missing fields: "
            f"{sorted(missing_clustering)}"
        )
    if clustering["representative_policy"] != "one_deterministic_representative_per_cluster":
        raise ValueError(
            "homology_clustering.representative_policy must be "
            "one_deterministic_representative_per_cluster"
        )
    for key in ("tool", "version", "command"):
        value = str(clustering[key]).strip()
        if not value or "REPLACE_WITH" in value:
            raise ValueError(f"homology_clustering has unresolved field '{key}'")
    for key in ("minimum_sequence_identity", "minimum_coverage"):
        value = float(clustering[key])
        if not 0 < value <= 1:
            raise ValueError(f"homology_clustering.{key} must be in (0, 1]")
    if str(clustering["tool"]).strip().lower() not in {"mmseqs", "mmseqs2"}:
        raise ValueError("the frozen revision protocol requires MMseqs2 homology clustering")
    if clustering.get("output_format", "mmseqs_rep_member") != "mmseqs_rep_member":
        raise ValueError("homology_clustering.output_format must be mmseqs_rep_member")

    command = shlex.split(str(clustering["command"]))

    def command_value(flag: str) -> str:
        if flag not in command or command.index(flag) + 1 >= len(command):
            raise ValueError(f"homology_clustering.command must declare {flag}")
        return command[command.index(flag) + 1]

    command_identity = float(command_value("--min-seq-id"))
    command_coverage = float(command_value("-c"))
    command_cov_mode = int(command_value("--cov-mode"))
    if not math.isclose(
        command_identity, float(clustering["minimum_sequence_identity"]),
        rel_tol=0, abs_tol=1e-12,
    ):
        raise ValueError("MMseqs2 --min-seq-id disagrees with minimum_sequence_identity")
    if not math.isclose(
        command_coverage, float(clustering["minimum_coverage"]),
        rel_tol=0, abs_tol=1e-12,
    ):
        raise ValueError("MMseqs2 -c disagrees with minimum_coverage")
    if command_cov_mode != int(clustering["coverage_mode"]):
        raise ValueError("MMseqs2 --cov-mode disagrees with coverage_mode")
    for source in sources:
        missing = REQUIRED_SOURCE_FIELDS - set(source)
        if missing:
            raise ValueError(f"source entry missing fields: {sorted(missing)}")
        if source["format"] not in {"fasta", "csv"}:
            raise ValueError(f"source {source['name']}: format must be fasta or csv")
        if int(source["label"]) not in {0, 1}:
            raise ValueError(f"source {source['name']}: label must be 0 or 1")
        if int(source["label"]) == 1 and source["evidence_policy"] == "not_applicable_negative":
            raise ValueError(f"source {source['name']}: positive rows require an evidence policy")
        if int(source["label"]) == 1:
            if source.get("evidence_column") is not None:
                if not source.get("evidence_accept"):
                    raise ValueError(
                        f"source {source['name']}: evidence_column requires evidence_accept"
                    )
            else:
                provenance = str(source.get("prefilter_provenance", "")).strip()
                if source.get("prefiltered") is not True or not provenance or "REPLACE_WITH" in provenance:
                    raise ValueError(
                        f"source {source['name']}: a positive source without a row-level "
                        "evidence_column must declare prefiltered=true and prefilter_provenance"
                    )
        for key in ("version", "retrieval_date", "license", "label_definition", "sha256"):
            value = str(source[key]).strip()
            if not value or "REPLACE_WITH" in value or value == "YYYY-MM-DD":
                raise ValueError(f"source {source['name']}: unresolved provenance field '{key}'")
        declared_hash = str(source["sha256"]).lower()
        if len(declared_hash) != 64 or any(char not in "0123456789abcdef" for char in declared_hash):
            raise ValueError(f"source {source['name']}: sha256 must be 64 hexadecimal characters")
    return manifest, sources


def collect_rows(manifest_path: Path, minimum: int, maximum: int) -> Tuple[List[Dict], List[Dict]]:
    _, sources = load_manifest(manifest_path)
    accepted: List[Dict] = []
    ledger: List[Dict] = []
    for source in sources:
        source_path = (manifest_path.parent / source["path"]).resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        source_hash = sha256(source_path)
        if source_hash != str(source["sha256"]).lower():
            raise ValueError(
                f"source {source['name']}: SHA-256 mismatch for {source_path}; "
                f"manifest={source['sha256']} actual={source_hash}"
            )
        iterator: Iterable[Tuple[str, str, Dict[str, str]]]
        if source["format"] == "fasta":
            iterator = fasta_rows(source_path)
        else:
            iterator = csv_rows(
                source_path,
                source.get("sequence_column", "sequence"),
                source.get("id_column", "id"),
            )
        for record_id, raw_sequence, raw in iterator:
            sequence = "".join(str(raw_sequence).split()).upper()
            length_ok = minimum <= len(sequence) <= maximum
            alphabet_ok = bool(sequence) and set(sequence) <= CANONICAL_AAS
            canonical_stage = length_ok and alphabet_ok
            evidence_ok = canonical_stage and evidence_passes(source, raw)
            row = {
                "source": source["name"],
                "source_record_id": record_id,
                "source_file": Path(str(source["path"])).as_posix(),
                "source_path_base": "manifest_parent",
                "source_file_sha256": source_hash,
                "source_version": source["version"],
                "retrieval_date": source["retrieval_date"],
                "source_url": source["url"],
                "license": source["license"],
                "evidence_policy": source["evidence_policy"],
                "evidence_column": source.get("evidence_column", ""),
                "evidence_accept": ";".join(map(str, source.get("evidence_accept", []))),
                "prefiltered": bool(source.get("prefiltered", False)),
                "prefilter_provenance": source.get("prefilter_provenance", ""),
                "label_definition": source["label_definition"],
                "sequence": sequence,
                "label": int(source["label"]),
                "passes_length": length_ok,
                "passes_canonical": canonical_stage,
                "evidence_pass": evidence_ok,
                "dedup_representative": False,
                "homology_representative": False,
                "cluster_id": "",
                "split": "",
            }
            ledger.append(row)
            if evidence_ok:
                accepted.append(row)
    return accepted, ledger


def deduplicate(rows: List[Dict], ledger: List[Dict]) -> List[Dict]:
    by_sequence: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_sequence[row["sequence"]].append(row)
    conflicts = {
        sequence: group for sequence, group in by_sequence.items()
        if len({int(item["label"]) for item in group}) > 1
    }
    if conflicts:
        examples = {seq: [(r["source"], r["label"]) for r in group] for seq, group in list(conflicts.items())[:10]}
        raise ValueError(
            f"{len(conflicts)} exact sequences have conflicting labels; resolve them in source data: {examples}"
        )
    representatives = []
    for sequence in sorted(by_sequence):
        group = by_sequence[sequence]
        representative = group[0]
        representative["dedup_representative"] = True
        representative["all_source_records"] = ";".join(
            f"{row['source']}:{row['source_record_id']}" for row in group
        )
        representatives.append(representative)
    return representatives


def load_clusters(path: Path, cluster_format: str = "sequence_cluster") -> Dict[str, str]:
    clusters: Dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"{path}:{number}: expected sequence<TAB>cluster_id")
            if cluster_format == "sequence_cluster":
                sequence, cluster_id = parts[0], parts[1]
            elif cluster_format == "mmseqs_rep_member":
                cluster_id, sequence = parts[0], parts[1]
            else:
                raise ValueError(f"unsupported cluster format: {cluster_format}")
            sequence = sequence.strip().upper()
            cluster_id = cluster_id.strip()
            if not sequence or set(sequence) - CANONICAL_AAS:
                raise ValueError(f"{path}:{number}: cluster member is not a canonical sequence ID")
            if sequence in clusters and clusters[sequence] != cluster_id:
                raise ValueError(f"{path}:{number}: sequence assigned to multiple clusters")
            clusters[sequence] = cluster_id
    return clusters


def write_clustering_input(rows: List[Dict], output: Path, manifest_path: Path,
                           minimum: int, maximum: int) -> None:
    """Write exact-deduplicated sequences with sequence-valued FASTA IDs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" is required: on Windows the default translates "\n" to CRLF, and
    # MMseqs2 fails on CRLF FASTA input in ways that look like a crash rather than a
    # parse error, so the clustering step must never emit platform line endings.
    with output.open("w", newline="\n", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: item["sequence"]):
            sequence = row["sequence"]
            handle.write(f">{sequence}\n{sequence}\n")
    report_path = output.with_suffix(output.suffix + ".metadata.json")
    report_base = report_path.parent
    report = {
        "schema_version": 1,
        "reportable": False,
        "path_base": "metadata_file_parent",
        "purpose": "intermediate input for the manifest-declared homology clustering command",
        "manifest": {
            "path": portable_path(manifest_path, report_base),
            "sha256": sha256(manifest_path),
        },
        "filters": {"minimum_length": minimum, "maximum_length": maximum,
                    "alphabet": "ACDEFGHIKLMNPQRSTVWY"},
        "records": len(rows),
        "output": {"path": portable_path(output, report_base), "sha256": sha256(output)},
        "next_step": (
            "Run homology_clustering.command from the manifest, then rerun the dataset "
            "builder with --cluster-tsv and the matching --cluster-format."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def assign_splits(rows: List[Dict], clusters: Dict[str, str], seed: int,
                  fractions: Tuple[float, float, float]) -> Dict[str, List[Dict]]:
    missing = [row["sequence"] for row in rows if row["sequence"] not in clusters]
    if missing:
        raise ValueError(f"cluster map is missing {len(missing)} curated sequences; examples: {missing[:5]}")
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        row["cluster_id"] = clusters[row["sequence"]]
        groups[row["cluster_id"]].append(row)
    mixed = {cluster: group for cluster, group in groups.items() if len({r['label'] for r in group}) > 1}
    if mixed:
        raise ValueError(f"{len(mixed)} homology clusters mix AMP/non-AMP labels; curate before splitting")

    rng = random.Random(seed)
    # MMseqs2 redundancy reduction: retain one deterministic sequence
    # per cluster.  This both enforces the declared identity threshold and
    # guarantees that homologous members cannot leak across data splits.
    cluster_representatives = []
    for cluster in sorted(groups):
        representative = min(
            groups[cluster], key=lambda row: (row["source"], row["source_record_id"], row["sequence"])
        )
        representative["homology_representative"] = True
        cluster_representatives.append(representative)

    result = {"train": [], "validation": [], "test": []}
    for label in (0, 1):
        label_rows = [row for row in cluster_representatives if int(row["label"]) == label]
        rng.shuffle(label_rows)
        total = len(label_rows)
        raw_counts = [fraction * total for fraction in fractions]
        allocated = [math.floor(value) for value in raw_counts]
        for index in sorted(range(3), key=lambda i: raw_counts[i] - allocated[i], reverse=True)[: total - sum(allocated)]:
            allocated[index] += 1
        boundaries = (allocated[0], allocated[0] + allocated[1])
        for index, row in enumerate(label_rows):
            if index < boundaries[0]:
                split = "train"
            elif index < boundaries[1]:
                split = "validation"
            else:
                split = "test"
            row["split"] = split
            result[split].append(row)
    return result


def write_outputs(splits: Dict[str, List[Dict]], ledger: List[Dict], output: Path,
                  manifest_path: Path, cluster_path: Path, seed: int,
                  fractions: Tuple[float, float, float], cluster_format: str) -> None:
    for split, rows in splits.items():
        labels = {int(row["label"]) for row in rows}
        if labels != {0, 1}:
            raise ValueError(
                f"{split} split must contain both labels after cluster-level allocation; found {sorted(labels)}"
            )
    output.mkdir(parents=True, exist_ok=True)
    report_base = output.resolve()
    extractor = PeptideFeatureExtractor()
    split_artifacts = {}
    for split, rows in splits.items():
        rows = sorted(rows, key=lambda row: (row["label"], row["sequence"]))
        csv_path = output / f"{split}.csv"
        fasta_path = output / f"{split}.fasta"
        with csv_path.open("w", newline="", encoding="utf-8") as csv_handle, fasta_path.open("w", newline="\n", encoding="utf-8") as fasta_handle:
            writer = None
            for index, row in enumerate(rows):
                item = {
                    "id": f"{split}_{index}", "source": row["source"],
                    "source_record_id": row["source_record_id"],
                    "all_source_records": row.get("all_source_records", ""),
                    "cluster_id": row["cluster_id"], "sequence": row["sequence"],
                    "label": row["label"], **extractor.extract_dict(row["sequence"]),
                }
                if writer is None:
                    writer = csv.DictWriter(csv_handle, fieldnames=list(item))
                    writer.writeheader()
                writer.writerow(item)
                fasta_handle.write(f">{item['id']} source={row['source']} label={row['label']} cluster={row['cluster_id']}\n{row['sequence']}\n")
        split_artifacts[split] = {
            "csv": {"path": portable_path(csv_path, report_base), "sha256": sha256(csv_path)},
            "fasta": {"path": portable_path(fasta_path, report_base), "sha256": sha256(fasta_path)},
        }

    ledger_path = output / "provenance_ledger.csv"
    keys = list(ledger[0]) if ledger else []
    if any("all_source_records" in row for row in ledger):
        keys.append("all_source_records")
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ledger)

    manifest, sources = load_manifest(manifest_path)

    def stage_counts(rows: List[Dict]) -> Dict[str, int]:
        """Sequential preprocessing funnel computed from the row-level ledger."""
        return {
            "raw_records": len(rows),
            "length_5_50": sum(bool(row["passes_length"]) for row in rows),
            "canonical_after_length": sum(bool(row["passes_canonical"]) for row in rows),
            "evidence_accepted": sum(bool(row["evidence_pass"]) for row in rows),
            "exact_dedup_representatives": sum(bool(row["dedup_representative"]) for row in rows),
            "homology_cluster_representatives": sum(bool(row["homology_representative"]) for row in rows),
        }

    by_source = {}
    for source in sources:
        source_rows = [row for row in ledger if row["source"] == source["name"]]
        source_file = (manifest_path.parent / source["path"]).resolve()
        by_source[source["name"]] = {
            **stage_counts(source_rows),
            "label": int(source["label"]),
            "version": source["version"],
            "retrieval_date": source["retrieval_date"],
            "url": source["url"],
            "license": source["license"],
            "evidence_policy": source["evidence_policy"],
            "evidence_column": source.get("evidence_column"),
            "evidence_accept": source.get("evidence_accept"),
            "prefiltered": bool(source.get("prefiltered", False)),
            "prefilter_provenance": source.get("prefilter_provenance"),
            "label_definition": source["label_definition"],
            "source_file": portable_path(source_file, report_base),
            "source_file_sha256": sha256(source_file),
        }

    report = {
        "schema_version": 1,
        "reportable": True,
        "path_base": "dataset_build_report_parent",
        "manifest": {
            "path": portable_path(manifest_path, report_base),
            "sha256": sha256(manifest_path),
        },
        "manifest_metadata": {
            key: value for key, value in manifest.items() if key != "sources"
        },
        "cluster_map": {
            "path": portable_path(cluster_path, report_base), "sha256": sha256(cluster_path),
            "format": cluster_format,
        },
        "split_seed": seed,
        "split_fractions": dict(zip(("train", "validation", "test"), fractions)),
        "preprocessing_funnel_global": stage_counts(ledger),
        "preprocessing_funnel_by_source": by_source,
        "funnel_note": (
            "After cross-source exact deduplication and homology clustering, a retained "
            "sequence is credited to its deterministic representative source; per-source "
            "counts at those two stages therefore sum to the corresponding global count."
        ),
        "counts": {
            split: {
                "rows": len(rows),
                "AMP": sum(int(row["label"]) == 1 for row in rows),
                "non_AMP": sum(int(row["label"]) == 0 for row in rows),
                "clusters": len({row["cluster_id"] for row in rows}),
            }
            for split, rows in splits.items()
        },
        "split_artifacts": split_artifacts,
        "ledger": {"path": portable_path(ledger_path, report_base), "sha256": sha256(ledger_path)},
    }
    (output / "dataset_build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    cluster_group = parser.add_mutually_exclusive_group(required=True)
    cluster_group.add_argument("--cluster-tsv")
    cluster_group.add_argument(
        "--prepare-clustering-fasta",
        help="write filtered/exact-deduplicated FASTA and stop before homology clustering",
    )
    parser.add_argument(
        "--cluster-format", choices=["sequence_cluster", "mmseqs_rep_member"],
        default=None, help="defaults to homology_clustering.output_format in the manifest",
    )
    parser.add_argument("--output-dir", default="dataset/rebuilt")
    parser.add_argument("--min-length", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", nargs=3, type=float, default=(0.70, 0.15, 0.15),
                        metavar=("TRAIN", "VALIDATION", "TEST"))
    args = parser.parse_args()
    if not 0 < args.min_length <= args.max_length:
        parser.error("invalid sequence length limits")
    if any(value <= 0 for value in args.fractions) or abs(sum(args.fractions) - 1.0) > 1e-8:
        parser.error("split fractions must be positive and sum to 1")

    manifest_path = Path(args.manifest)
    manifest, _ = load_manifest(manifest_path)
    accepted, ledger = collect_rows(manifest_path, args.min_length, args.max_length)
    representatives = deduplicate(accepted, ledger)
    if args.prepare_clustering_fasta:
        output = Path(args.prepare_clustering_fasta)
        write_clustering_input(
            representatives, output, manifest_path, args.min_length, args.max_length
        )
        logger.info("Wrote %d exact-deduplicated sequences for clustering to %s", len(representatives), output)
        return
    cluster_path = Path(args.cluster_tsv)
    declared_cluster_format = manifest["homology_clustering"].get(
        "output_format", "mmseqs_rep_member"
    )
    if args.cluster_format and args.cluster_format != declared_cluster_format:
        parser.error("--cluster-format disagrees with homology_clustering.output_format")
    cluster_format = args.cluster_format or declared_cluster_format
    splits = assign_splits(
        representatives, load_clusters(cluster_path, cluster_format),
        args.seed, tuple(args.fractions)
    )
    write_outputs(splits, ledger, Path(args.output_dir), manifest_path, cluster_path,
                  args.seed, tuple(args.fractions), cluster_format)
    logger.info("Wrote auditable dataset build to %s", args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Dataset build refused: %s", exc)
        sys.exit(2)
