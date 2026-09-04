#!/usr/bin/env python
"""Evaluate generated ensembles with seed-level uncertainty and provenance.

Instability Index (II) is reported only as an empirical sequence-derived
surrogate. ESM-2 pseudo-perplexity is reported only as sequence plausibility.
An oracle used by SCST is retained as a diagnostic reward-oracle score and is
never labelled independent AMP validation. Genuine external AMP, toxicity, and
hemolysis outputs are handled by ``evaluate_external_validators.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from peptidegen.evaluation import PeptideStabilityAnalyzer, calculate_diversity_metrics
from peptidegen.data.features import PeptideFeatureExtractor


CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")
FNAME_RE = re.compile(r"(?P<model>.+?)_seed(?P<seed>\d+)$", re.IGNORECASE)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_path(value: str, metadata_path: Path) -> Path:
    """Resolve portable sidecar-relative paths and legacy absolute paths."""
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (metadata_path.resolve().parent / path).resolve()


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
    invalid = [s for s in sequences if not s or not set(s) <= CANONICAL]
    if invalid:
        raise ValueError(f"{path}: found {len(invalid)} empty/non-canonical sequences")
    return sequences


def parse_run(path: Path) -> Tuple[str, int]:
    match = FNAME_RE.fullmatch(path.stem)
    if not match:
        raise ValueError(f"{path.name} must match <model>_seed<N>.fasta")
    return match.group("model"), int(match.group("seed"))


def validate_generation_sidecar(path: Path, model: str, seed: int) -> Dict[str, Any]:
    """Validate the provenance sidecar written by ``generate.py``."""
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.exists():
        raise ValueError(f"{path}: missing generation metadata sidecar {metadata_path.name}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("reportable") is not True:
        raise ValueError(f"{metadata_path}: generation is not marked reportable")
    if metadata.get("model_id") != model:
        raise ValueError(
            f"{metadata_path}: model_id={metadata.get('model_id')!r} does not match file model {model!r}"
        )
    if int(metadata.get("seed", -1)) != seed:
        raise ValueError(f"{metadata_path}: seed does not match file name")
    output = metadata.get("output") or {}
    if output.get("sha256") != sha256(path):
        raise ValueError(f"{metadata_path}: output FASTA hash mismatch")
    if int(output.get("records", -1)) != fasta_count(path):
        raise ValueError(f"{metadata_path}: output record count mismatch")
    checkpoint = metadata.get("checkpoint") or {}
    checkpoint_path = sidecar_path(str(checkpoint.get("path", "")), metadata_path)
    if (
        not checkpoint.get("sha256") or not checkpoint_path.is_file()
        or sha256(checkpoint_path) != checkpoint.get("sha256")
    ):
        raise ValueError(f"{metadata_path}: checkpoint is absent or hash-mismatched")
    dataset_audit = metadata.get("dataset_build_audit") or {}
    dataset_report_path = sidecar_path(str(dataset_audit.get("path", "")), metadata_path)
    if (
        not dataset_audit.get("sha256") or not dataset_report_path.is_file()
        or sha256(dataset_report_path) != dataset_audit.get("sha256")
    ):
        raise ValueError(f"{metadata_path}: dataset-build report is absent or hash-mismatched")
    software = metadata.get("software") or {}
    git = software.get("git") or {}
    if not git.get("commit") or git.get("dirty_worktree") is not False:
        raise ValueError(f"{metadata_path}: generation was not made from a clean immutable git commit")
    command = shlex.split(metadata.get("command", ""))
    for flag, expected in (("--seed", str(seed)), ("--model-id", model)):
        if flag not in command or command.index(flag) + 1 >= len(command):
            raise ValueError(f"{metadata_path}: command does not record {flag}")
        if command[command.index(flag) + 1] != expected:
            raise ValueError(f"{metadata_path}: command {flag} does not match the artifact")
    sampling = metadata.get("sampling") or {}
    if int(sampling.get("requested_sequences", -1)) != int(output.get("records", -2)):
        raise ValueError(f"{metadata_path}: requested/output sequence counts disagree")
    return {
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256(metadata_path),
        "checkpoint_sha256": checkpoint["sha256"],
        "dataset_build_report_sha256": dataset_audit["sha256"],
        "code_commit": git["commit"],
        "generation_sampling": sampling,
        "model_id": metadata["model_id"],
    }


def fasta_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(line.startswith(">") for line in handle)


def per_run_metrics(
    sequences: Sequence[str],
    analyzer: PeptideStabilityAnalyzer,
    training_sequences: Sequence[str] | None = None,
    amp_oracle=None,
    hemo_oracle=None,
    plausibility_evaluator=None,
) -> Dict[str, Any]:
    stability = analyzer.analyze_batch(list(sequences))
    rows = stability.get("metrics", [])
    diversity = calculate_diversity_metrics(list(sequences))
    ii = np.asarray([row["instability_index"] for row in rows], dtype=float)
    extractor = PeptideFeatureExtractor(
        feature_names=["charge_at_pH7", "hydrophobic_moment"]
    )
    derived = [extractor.extract_dict(row["sequence"]) for row in rows]
    charge = np.asarray([row["charge_at_pH7"] for row in derived], dtype=float)
    hydro_moment = np.asarray([row["hydrophobic_moment"] for row in derived], dtype=float)
    training_set = set(training_sequences or [])
    result: Dict[str, Any] = {
        "n": int(len(sequences)),
        "validity_ratio": 1.0,
        "exact_novelty_ratio_vs_training": (
            float(np.mean([sequence not in training_set for sequence in sequences]))
            if training_set else None
        ),
        "stable_rate_ii_lt_40_percent": float((ii < analyzer.stability_threshold).mean() * 100),
        "mean_instability_index": float(ii.mean()),
        "mean_charge_ph7": float(charge.mean()),
        "mean_hydrophobic_moment": float(hydro_moment.mean()),
        "uniqueness_ratio": float(diversity.get("uniqueness_ratio", 0.0)),
        "bigram_diversity": float(diversity.get("bigram_diversity", 0.0)),
        "mean_length": float(np.mean([len(s) for s in sequences])),
        "_per_sequence": {
            "instability_index": ii.tolist(),
            "charge_ph7": charge.tolist(),
            "hydrophobic_moment": hydro_moment.tolist(),
        },
    }

    if amp_oracle is not None:
        scored = amp_oracle.score_generated(list(sequences))
        result["training_amp_reward_oracle_mean_probability"] = float(scored["mean_prob"])
        result["training_amp_reward_oracle_positive_rate"] = float(scored["positive_rate"])
        result["_per_sequence"]["training_amp_reward_oracle_probability"] = list(
            map(float, scored["_per_sequence"])
        )

    if hemo_oracle is not None:
        scored = hemo_oracle.score_generated(list(sequences))
        result["hemolysis_oracle_mean_probability"] = float(scored["mean_prob"])
        result["hemolysis_oracle_positive_rate"] = float(scored["positive_rate"])
        result["_per_sequence"]["hemolysis_oracle_probability"] = list(
            map(float, scored["_per_sequence"])
        )

    if plausibility_evaluator is not None:
        evaluated = plausibility_evaluator.evaluate(
            list(sequences), sample=min(500, len(sequences)), compute_contacts=False
        )
        result["esm2_pseudo_perplexity"] = float(evaluated["esm_pseudo_perplexity"]["mean"])
        result["chou_fasman_helix_propensity"] = float(evaluated["helix_fraction"]["mean"])
        result["_per_sequence"]["esm2_pseudo_perplexity"] = list(
            map(float, evaluated["_per_sequence"]["pseudo_perplexity"])
        )
        if evaluated.get("esm_contact_order"):
            result["esm2_attention_contact_order"] = float(
                evaluated["esm_contact_order"]["mean"]
            )
    return result


def scalar_metrics(run: Dict[str, Any]) -> Iterable[Tuple[str, float]]:
    for key, value in run.items():
        if key.startswith("_") or key in {"n", "seed"}:
            continue
        if isinstance(value, (int, float)) and value is not None and math.isfinite(float(value)):
            yield key, float(value)


def aggregate(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from scipy import stats

    values: Dict[str, List[float]] = defaultdict(list)
    for run in runs:
        for key, value in scalar_metrics(run):
            values[key].append(value)
    output: Dict[str, Any] = {}
    for key, items in values.items():
        arr = np.asarray(items, dtype=float)
        if len(arr) > 1:
            sem = stats.sem(arr)
            half = float(stats.t.ppf(0.975, len(arr) - 1) * sem)
            ci = [float(arr.mean() - half), float(arr.mean() + half)]
            std = float(arr.std(ddof=1))
        else:
            ci, std = None, None
        output[key] = {
            "mean": float(arr.mean()),
            "sample_std": std,
            "ci95": ci,
            "n_seeds": int(len(arr)),
        }
    return output


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta without allocating an O(n*m) dominance matrix."""
    xa = np.asarray(x, dtype=float)
    ys = np.sort(np.asarray(y, dtype=float))
    if len(xa) == 0 or len(ys) == 0:
        return float("nan")
    more = sum(int(np.searchsorted(ys, value, side="left")) for value in xa)
    less = sum(int(len(ys) - np.searchsorted(ys, value, side="right")) for value in xa)
    return float((more - less) / (len(xa) * len(ys)))


def hedges_g(x: Sequence[float], y: Sequence[float]) -> float:
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(xa) < 2 or len(ya) < 2:
        return float("nan")
    df = len(xa) + len(ya) - 2
    pooled_var = ((len(xa) - 1) * xa.var(ddof=1) + (len(ya) - 1) * ya.var(ddof=1)) / df
    if pooled_var <= 0:
        return 0.0
    correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(correction * (xa.mean() - ya.mean()) / math.sqrt(pooled_var))


def welch_ci(x: Sequence[float], y: Sequence[float]) -> Any:
    from scipy import stats

    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(xa) < 2 or len(ya) < 2:
        return None
    vx, vy = xa.var(ddof=1) / len(xa), ya.var(ddof=1) / len(ya)
    se = math.sqrt(vx + vy)
    if se == 0:
        diff = float(xa.mean() - ya.mean())
        return [diff, diff]
    df = (vx + vy) ** 2 / (
        vx**2 / (len(xa) - 1) + vy**2 / (len(ya) - 1)
    )
    half = float(stats.t.ppf(0.975, df) * se)
    diff = float(xa.mean() - ya.mean())
    return [diff - half, diff + half]


def significance(reference: Sequence[Dict[str, Any]], baseline: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Tests independent seed-level summaries; generated sequences are not pseudoreplicates."""
    from scipy import stats

    ref_values: Dict[str, List[float]] = defaultdict(list)
    base_values: Dict[str, List[float]] = defaultdict(list)
    for run in reference:
        for key, value in scalar_metrics(run):
            ref_values[key].append(value)
    for run in baseline:
        for key, value in scalar_metrics(run):
            base_values[key].append(value)

    output: Dict[str, Any] = {}
    for metric in sorted(set(ref_values) & set(base_values)):
        x, y = ref_values[metric], base_values[metric]
        if len(x) < 2 or len(y) < 2:
            continue
        t_stat, t_p = stats.ttest_ind(x, y, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
        output[metric] = {
            "unit_of_analysis": "random_seed",
            "n_reference_seeds": len(x),
            "n_baseline_seeds": len(y),
            "reference_mean": float(np.mean(x)),
            "baseline_mean": float(np.mean(y)),
            "mean_difference": float(np.mean(x) - np.mean(y)),
            "mean_difference_ci95_welch": welch_ci(x, y),
            "welch_t": float(t_stat),
            "welch_p_raw": float(t_p),
            "hedges_g": hedges_g(x, y),
            "mann_whitney_u": float(u_stat),
            "mann_whitney_p_raw": float(u_p),
            "cliffs_delta": cliffs_delta(x, y),
        }
    return output


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    """Holm family-wise error correction, preserving input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def add_multiple_testing_correction(comparisons: Dict[str, Any]) -> None:
    # Parametric and rank-based tests answer related but distinct questions;
    # adjust each pre-specified test family separately rather than doubling the
    # family size by mixing the two types of p-value.
    for key in ("welch_p_raw", "mann_whitney_p_raw"):
        refs: List[Dict[str, Any]] = []
        raw: List[float] = []
        for model_metrics in comparisons.values():
            for item in model_metrics.values():
                refs.append(item)
                raw.append(float(item[key]))
        for item, adjusted in zip(refs, holm_adjust(raw)):
            item[key.replace("_raw", "_holm")] = adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-dir", required=True)
    parser.add_argument("--reference", default="LightweightPeptideGen")
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1337])
    parser.add_argument("--expected-n", type=int, default=1000)
    parser.add_argument("--expected-models", nargs="*", default=None)
    parser.add_argument("--train-fasta", default=None,
                        help="training FASTA used for exact-sequence novelty; required for reportable runs")
    parser.add_argument("--allow-incomplete", action="store_true", help="smoke-test mode; not reportable")
    parser.add_argument("--amp-oracle", default=None, help="SCST reward oracle; diagnostic only")
    parser.add_argument("--training-oracle-id", default=None)
    parser.add_argument("--hemo-oracle", default=None)
    parser.add_argument("--hemo-oracle-id", default=None)
    parser.add_argument("--hemo-oracle-used-in-training", action="store_true")
    parser.add_argument("--sequence-plausibility", action="store_true")
    parser.add_argument("--foldability", action="store_true", help="deprecated alias for --sequence-plausibility")
    parser.add_argument("--esm-model", default="esm2_t12_35M_UR50D")
    parser.add_argument("--esm-model-revision", default=None,
                        help="immutable Hugging Face commit; required for reportable plausibility")
    parser.add_argument("--threshold", type=float, default=40.0)
    parser.add_argument("--out", default="results/benchmark.json")
    args = parser.parse_args()

    if args.amp_oracle and not args.training_oracle_id:
        parser.error("--training-oracle-id is required with --amp-oracle")
    if args.hemo_oracle and not args.hemo_oracle_id:
        parser.error("--hemo-oracle-id is required with --hemo-oracle")
    if not args.train_fasta and not args.allow_incomplete:
        parser.error("--train-fasta is required for a reportable benchmark")
    if (args.sequence_plausibility or args.foldability) and not args.esm_model_revision and not args.allow_incomplete:
        parser.error("reportable ESM-2 plausibility requires --esm-model-revision")

    training_sequences: List[str] = []
    training_fasta_record = None
    if args.train_fasta:
        training_path = Path(args.train_fasta)
        if not training_path.exists():
            parser.error(f"training FASTA not found: {training_path}")
        training_sequences = read_fasta(training_path)
        training_fasta_record = {
            "path": str(training_path.resolve()),
            "sha256": sha256(training_path),
            "n": len(training_sequences),
        }

    files = sorted(Path(args.gen_dir).glob("*.fasta"))
    if not files:
        parser.error(f"no FASTA files in {args.gen_dir}")
    by_model: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for path in files:
        try:
            model, seed = parse_run(path)
        except ValueError:
            continue
        by_model[model].append((seed, path))
    if not by_model:
        parser.error("no FASTA file names matched <model>_seed<N>.fasta")

    expected_seeds = set(args.expected_seeds)
    errors: List[str] = []
    if args.expected_models and set(args.expected_models) != set(by_model):
        errors.append(
            f"expected models {sorted(args.expected_models)}, found {sorted(by_model)}"
        )
    for model, runs in by_model.items():
        seeds = [seed for seed, _ in runs]
        if len(seeds) != len(set(seeds)):
            errors.append(f"{model}: duplicate seed files")
        if set(seeds) != expected_seeds:
            errors.append(f"{model}: expected seeds {sorted(expected_seeds)}, found {sorted(seeds)}")
        if not args.allow_incomplete:
            for seed, path in runs:
                try:
                    validate_generation_sidecar(path, model, seed)
                except (ValueError, OSError, json.JSONDecodeError) as exc:
                    errors.append(str(exc))
    if errors and not args.allow_incomplete:
        parser.error("; ".join(errors))

    analyzer = PeptideStabilityAnalyzer(stability_threshold=args.threshold)
    amp_oracle = hemo_oracle = plausibility = None
    if args.amp_oracle:
        from peptidegen.evaluation.oracle import ESM2Oracle
        amp_oracle = ESM2Oracle.load(args.amp_oracle)
    if args.hemo_oracle:
        from peptidegen.evaluation.oracle import ESM2Oracle
        hemo_oracle = ESM2Oracle.load(args.hemo_oracle)
    if args.sequence_plausibility or args.foldability:
        from peptidegen.evaluation.foldability import FoldabilityEvaluator
        plausibility = FoldabilityEvaluator(
            model_name=args.esm_model, model_revision=args.esm_model_revision
        )

    results: Dict[str, Any] = {
        "schema_version": 2,
        "reportable": not args.allow_incomplete and not errors,
        "integrity_warnings": errors,
        "metric_interpretation": {
            "stable_rate_ii_lt_40_percent": (
                "Empirical sequence-derived Instability Index threshold; not thermodynamic, "
                "folding, membrane, or experimental stability."
            ),
            "esm2_pseudo_perplexity": (
                "Protein-language-model sequence plausibility; not a folding-confidence metric."
            ),
            "training_amp_reward_oracle_positive_rate": (
                "Diagnostic only because the same oracle optimized SCST; not independent validation."
            ),
        },
        "evaluation_metadata": {
            "expected_seeds": args.expected_seeds,
            "expected_sequences_per_seed": args.expected_n,
            "ii_threshold": args.threshold,
            "amp_reward_oracle_id": args.training_oracle_id,
            "hemolysis_oracle_id": args.hemo_oracle_id,
            "hemolysis_oracle_used_in_training": args.hemo_oracle_used_in_training,
            "multiple_testing_correction": (
                "Holm family-wise error rate within each pre-specified test family "
                "(Welch and Mann-Whitney), across model/metric comparisons"
            ),
            "statistical_unit": "random seed",
            "training_fasta": training_fasta_record,
            "sequence_plausibility_model": (
                {"model": args.esm_model, "revision": args.esm_model_revision}
                if (args.sequence_plausibility or args.foldability) else None
            ),
        },
        "per_model": {},
        "significance_vs_reference": {},
    }
    runs_by_model: Dict[str, List[Dict[str, Any]]] = {}
    for model, entries in sorted(by_model.items()):
        runs: List[Dict[str, Any]] = []
        for seed, path in sorted(entries):
            sequences = read_fasta(path)
            if len(sequences) != args.expected_n:
                message = f"{path.name}: expected {args.expected_n} sequences, found {len(sequences)}"
                if not args.allow_incomplete:
                    parser.error(message)
                results["integrity_warnings"].append(message)
                results["reportable"] = False
            metrics = per_run_metrics(
                sequences, analyzer, training_sequences,
                amp_oracle, hemo_oracle, plausibility,
            )
            metrics["seed"] = seed
            metrics["fasta_path"] = str(path.resolve())
            metrics["fasta_sha256"] = sha256(path)
            try:
                metrics.update(validate_generation_sidecar(path, model, seed))
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                if not args.allow_incomplete:
                    parser.error(str(exc))
                results["integrity_warnings"].append(str(exc))
                results["reportable"] = False
            runs.append(metrics)
        runs_by_model[model] = runs
        results["per_model"][model] = {
            "n_seeds": len(runs),
            "runs": runs,
            "aggregate": aggregate(runs),
        }

    if args.reference not in runs_by_model:
        if not args.allow_incomplete:
            parser.error(f"reference '{args.reference}' not found")
        results["integrity_warnings"].append(f"reference '{args.reference}' not found")
        results["reportable"] = False
    else:
        for model, runs in runs_by_model.items():
            if model != args.reference:
                results["significance_vs_reference"][model] = significance(
                    runs_by_model[args.reference], runs
                )
        add_multiple_testing_correction(results["significance_vs_reference"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}; reportable={results['reportable']}")
    for model, item in results["per_model"].items():
        agg = item["aggregate"]
        stable = agg["stable_rate_ii_lt_40_percent"]
        print(
            f"{model}: II<40={stable['mean']:.2f} +/- "
            f"{stable['sample_std'] if stable['sample_std'] is not None else float('nan'):.2f}%"
        )


if __name__ == "__main__":
    main()
