"""Regression tests for the major-revision audit fixes."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from peptidegen.data import VOCAB
from peptidegen.data.dataset import ConditionalPeptideDataset
from peptidegen.inference import PeptideSampler
from peptidegen.models import CNNDiscriminator, GRUGenerator, MultimodalFusionGenerator
from peptidegen.training import GANTrainer
from peptidegen.models.esm2_hf import ESM2HF


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clustering_manifest() -> dict:
    return {
        "tool": "MMseqs2", "version": "15-6f452",
        "command": "mmseqs easy-cluster input.fasta out tmp --min-seq-id 0.4 -c 0.8 --cov-mode 0",
        "minimum_sequence_identity": 0.4, "minimum_coverage": 0.8,
        "coverage_mode": 0, "output_format": "mmseqs_rep_member",
        "representative_policy": "one_deterministic_representative_per_cluster",
    }


@pytest.mark.parametrize("generator", [
    GRUGenerator(
        vocab_size=VOCAB.vocab_size, embedding_dim=16, hidden_dim=32,
        latent_dim=8, max_length=12, num_layers=1, condition_dim=2,
        pad_idx=VOCAB.pad_idx, sos_idx=VOCAB.sos_idx, eos_idx=VOCAB.eos_idx,
    ),
    MultimodalFusionGenerator(
        vocab_size=VOCAB.vocab_size, embedding_dim=16, hidden_dim=32,
        latent_dim=8, max_length=12, num_layers=1, num_heads=2,
        condition_dim=2, mem_tokens=4, gat_heads=2, use_gat=False,
    ),
])
def test_sampling_is_autoregressive_and_masks_special_tokens(generator):
    torch.manual_seed(9)
    sampler = PeptideSampler(generator, device=torch.device("cpu"))
    sequences = sampler.sample(
        n=7, conditions=torch.zeros(7, 2), min_length=5, max_length=12,
        batch_size=4, temperature=0.9, top_k=5, top_p=0.95,
    )
    assert len(sequences) == 7
    assert all(5 <= len(sequence) <= 12 for sequence in sequences)
    assert all(set(sequence) <= set(VOCAB.STANDARD_AAS) for sequence in sequences)


def test_wgan_gp_aligns_real_targets_with_generated_length():
    """Regression test for the real=(B, L+2), fake=(B, L) GP crash."""
    generator = GRUGenerator(
        vocab_size=VOCAB.vocab_size, embedding_dim=12, hidden_dim=24,
        latent_dim=8, max_length=10, num_layers=1, condition_dim=None,
        pad_idx=VOCAB.pad_idx, sos_idx=VOCAB.sos_idx, eos_idx=VOCAB.eos_idx,
    )
    discriminator = CNNDiscriminator(
        vocab_size=VOCAB.vocab_size, embedding_dim=12, hidden_dim=16,
        max_length=12, num_filters=[8, 8], kernel_sizes=[3, 5],
        use_spectral_norm=False, use_minibatch_std=False, pad_idx=VOCAB.pad_idx,
    )
    trainer = GANTrainer(generator, discriminator, {
        "gan_loss": "wgan_gp", "lambda_gp": 1.0,
        "learning_rate": 1e-4, "lr_discriminator": 1e-4,
        "g_steps": 1, "d_steps": 1, "use_amp": False,
        "adversarial_weight": 1.0, "diversity_weight": 0.0,
        "reconstruction_weight": 0.1, "noise_std": 0.0,
    }, device=torch.device("cpu"))

    real_inputs = torch.randint(4, VOCAB.vocab_size, (3, 12))
    real_inputs[:, 0] = VOCAB.sos_idx
    targets = torch.randint(4, VOCAB.vocab_size, (3, 12))
    targets[:, -1] = VOCAB.eos_idx
    losses = trainer.train_step(real_inputs, reconstruction_targets=targets)
    assert np.isfinite(losses["d_loss"])
    assert np.isfinite(losses["g_loss"])


def test_validation_can_reuse_training_normalization():
    features_train = [{"instability_index": 10.0}, {"instability_index": 30.0}]
    train = ConditionalPeptideDataset(
        ["AAAAA", "CCCCC"], features_train,
        feature_names=["instability_index"], max_length=10,
    )
    validation = ConditionalPeptideDataset(
        ["DDDDD"], [{"instability_index": 20.0}],
        feature_names=["instability_index"], max_length=10,
        feature_stats=train.get_feature_stats(),
    )
    assert np.isclose(validation[0]["condition"].item(), 0.0, atol=1e-6)


def test_manifest_build_keeps_homology_clusters_out_of_multiple_splits(tmp_path):
    from peptidegen.data.__main__ import (
        assign_splits, collect_rows, deduplicate, load_clusters, write_outputs,
    )

    positive = tmp_path / "positive.fasta"
    negative = tmp_path / "negative.csv"
    positive.write_text(">p1\nKKLLKK\n>p2\nKKLLKA\n>p3\nRRWWRR\n>p4\nRRLLRR\n", encoding="utf-8")
    negative.write_text("id,sequence\nn1,AAAAAA\nn2,AAAAAT\nn3,CCCCCC\nn4,DDDDDD\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "dataset_name": "test dataset",
        "curation_protocol": "test protocol v1",
        "homology_clustering": _clustering_manifest(),
        "sources": [
            {
                "name": "curated-positive", "path": "positive.fasta", "format": "fasta",
                "label": 1, "version": "test", "retrieval_date": "2026-01-01",
                "url": "https://example.org/positive", "license": "test-only",
                "evidence_policy": "experimentally_validated_only",
                "prefiltered": True, "prefilter_provenance": "test fixture",
                "label_definition": "experimentally validated positive",
                "sha256": _sha256(positive),
            },
            {
                "name": "curated-negative", "path": "negative.csv", "format": "csv",
                "label": 0, "version": "test", "retrieval_date": "2026-01-01",
                "url": "https://example.org/negative", "license": "test-only",
                "evidence_policy": "not_applicable_negative",
                "label_definition": "curated negative",
                "sha256": _sha256(negative),
            },
        ],
    }), encoding="utf-8")
    clusters_path = tmp_path / "clusters.tsv"
    clusters_path.write_text(
        "KKLLKK\tp1\nKKLLKA\tp1\nRRWWRR\tp2\nRRLLRR\tp3\n"
        "AAAAAA\tn1\nAAAAAT\tn1\nCCCCCC\tn2\nDDDDDD\tn3\n", encoding="utf-8",
    )

    accepted, ledger = collect_rows(manifest_path, 5, 50)
    unique = deduplicate(accepted, ledger)
    splits = assign_splits(unique, load_clusters(clusters_path), 42, (0.5, 0.25, 0.25))
    retained_clusters = [row["cluster_id"] for rows in splits.values() for row in rows]
    assert len(retained_clusters) == len(set(retained_clusters)) == 6
    output = tmp_path / "built"
    write_outputs(
        splits, ledger, output, manifest_path, clusters_path, 42,
        (0.5, 0.25, 0.25), "sequence_cluster",
    )
    report = json.loads((output / "dataset_build_report.json").read_text())
    assert report["reportable"] is True
    assert report["path_base"] == "dataset_build_report_parent"
    assert not Path(report["manifest"]["path"]).is_absolute()
    assert not Path(report["cluster_map"]["path"]).is_absolute()
    assert not Path(report["ledger"]["path"]).is_absolute()
    assert sum(item["rows"] for item in report["counts"].values()) == 6
    assert report["preprocessing_funnel_global"] == {
        "raw_records": 8,
        "length_5_50": 8,
        "canonical_after_length": 8,
        "evidence_accepted": 8,
        "exact_dedup_representatives": 8,
        "homology_cluster_representatives": 6,
    }
    assert set(report["preprocessing_funnel_by_source"]) == {
        "curated-positive", "curated-negative",
    }
    from peptidegen.data.integrity import validate_dataset_build
    validate_dataset_build(
        str(output / "train.csv"), str(output / "validation.csv"),
        str(output / "test.csv"), str(output / "dataset_build_report.json"),
    )


def test_manifest_build_rejects_label_conflicts(tmp_path):
    from peptidegen.data.__main__ import collect_rows, deduplicate

    (tmp_path / "p.fasta").write_text(">p\nKKLLKK\n", encoding="utf-8")
    (tmp_path / "n.fasta").write_text(">n\nKKLLKK\n", encoding="utf-8")
    base = {
        "format": "fasta", "version": "test", "retrieval_date": "2026-01-01",
        "url": "https://example.org", "license": "test-only",
    }
    manifest = {
        "schema_version": 1,
        "dataset_name": "test conflict dataset",
        "curation_protocol": "test protocol v1",
        "homology_clustering": _clustering_manifest(),
        "sources": [
            {**base, "name": "p", "path": "p.fasta", "label": 1,
             "evidence_policy": "experimentally_validated_only",
             "prefiltered": True, "prefilter_provenance": "test fixture",
             "label_definition": "experimentally validated positive",
             "sha256": _sha256(tmp_path / "p.fasta")},
            {**base, "name": "n", "path": "n.fasta", "label": 0,
             "evidence_policy": "not_applicable_negative",
             "label_definition": "curated negative",
             "sha256": _sha256(tmp_path / "n.fasta")},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    accepted, ledger = collect_rows(manifest_path, 5, 50)
    with pytest.raises(ValueError, match="conflicting labels"):
        deduplicate(accepted, ledger)


def test_manifest_rejects_source_hash_mismatch(tmp_path):
    from peptidegen.data.__main__ import collect_rows

    source = tmp_path / "p.fasta"
    source.write_text(">p\nKKLLKK\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "dataset_name": "hash test",
        "curation_protocol": "test protocol v1",
        "homology_clustering": _clustering_manifest(),
        "sources": [{
            "name": "p", "path": "p.fasta", "format": "fasta", "label": 1,
            "version": "test", "retrieval_date": "2026-01-01",
            "url": "https://example.org", "license": "test-only",
            "evidence_policy": "experimentally_validated_only",
            "prefiltered": True, "prefilter_provenance": "test fixture",
            "label_definition": "experimentally validated positive",
            "sha256": "0" * 64,
        }],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        collect_rows(manifest_path, 5, 50)


def test_manifest_rejects_untraceable_positive_evidence(tmp_path):
    from peptidegen.data.__main__ import load_manifest

    source = tmp_path / "p.fasta"
    source.write_text(">p\nKKLLKK\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "dataset_name": "evidence test",
        "curation_protocol": "test protocol v1",
        "homology_clustering": _clustering_manifest(),
        "sources": [{
            "name": "p", "path": "p.fasta", "format": "fasta", "label": 1,
            "version": "test", "retrieval_date": "2026-01-01",
            "url": "https://example.org", "license": "test-only",
            "evidence_policy": "experimentally_validated_only",
            "label_definition": "experimentally validated positive",
            "sha256": _sha256(source),
        }],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="prefiltered=true"):
        load_manifest(manifest_path)


def test_mmseqs_native_cluster_map_uses_member_sequence(tmp_path):
    from peptidegen.data.__main__ import load_clusters

    path = tmp_path / "mmseqs_cluster.tsv"
    path.write_text("KKLLKK\tKKLLKK\nKKLLKK\tKKLLKA\n", encoding="utf-8")
    assert load_clusters(path, "mmseqs_rep_member") == {
        "KKLLKK": "KKLLKK", "KKLLKA": "KKLLKK",
    }


def test_residue_contacts_are_aligned_after_cls_token():
    class FakeESM(ESM2HF):
        def __init__(self):
            torch.nn.Module.__init__(self)

        def embed(self, sequences, **kwargs):
            tokens = torch.zeros(1, 6, 4)
            mask = torch.tensor([[0, 1, 1, 1, 0, 0]], dtype=torch.float32)
            contacts = torch.zeros(1, 3, 3)
            contacts[0, 0, 2] = contacts[0, 2, 0] = 0.9
            return {"tokens": tokens, "mask": mask, "contacts": contacts}

    graph = FakeESM().contact_graph(
        ["AAA"], thresh=0.5, local_window=0, max_length=6
    )["adj"][0]
    assert graph[1, 3] == 1 and graph[3, 1] == 1
    assert graph[0, 2] == 0 and graph[2, 0] == 0


def test_generation_sidecar_binds_model_checkpoint_and_dataset(tmp_path):
    from scripts.evaluate_generated import validate_generation_sidecar

    run_dir = tmp_path / "release" / "generated"
    checkpoint_dir = tmp_path / "release" / "checkpoints"
    dataset_dir = tmp_path / "release" / "dataset"
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)
    fasta = run_dir / "full_seed42.fasta"
    checkpoint = checkpoint_dir / "full.pt"
    dataset_report = dataset_dir / "dataset_build_report.json"
    fasta.write_text(">sequence_1\nKKLLKK\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    dataset_report.write_text('{"reportable": true}', encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "reportable": True,
        "model_id": "full",
        "seed": 42,
        "command": "generate.py --checkpoint ../checkpoints/full.pt --model-id full --seed 42",
        "output": {"sha256": _sha256(fasta), "records": 1},
        "checkpoint": {"path": "../checkpoints/full.pt", "sha256": _sha256(checkpoint)},
        "dataset_build_audit": {
            "path": "../dataset/dataset_build_report.json",
            "sha256": _sha256(dataset_report),
        },
        "sampling": {"requested_sequences": 1},
        "software": {"git": {"commit": "abc123", "dirty_worktree": False}},
    }
    sidecar = fasta.with_suffix(fasta.suffix + ".metadata.json")
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    record = validate_generation_sidecar(fasta, "full", 42)
    assert record["model_id"] == "full"
    with pytest.raises(ValueError, match="model_id"):
        validate_generation_sidecar(fasta, "no_gatv2", 42)
