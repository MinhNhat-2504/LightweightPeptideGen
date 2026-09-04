#!/usr/bin/env python
"""Teacher-forced MLE warm-up for the multimodal generator.

With ESM enabled, frozen ESM-2 residue embeddings refine the latent memory.
``--esm-attention-contacts`` additionally thresholds ESM-2 attention-derived
contact probabilities; it does *not* claim a C-alpha distance graph.  Genuine
distance contacts require a precomputed structural cache and are outside this
command's current implementation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from peptidegen.data import ConditionalPeptideDataset, VOCAB, get_dataloader, validate_dataset_build
from peptidegen.models import MultimodalFusionGenerator
from peptidegen.models.esm2_hf import ESM2HF
from peptidegen.utils import get_device, load_config, set_seed


def file_record(path: str) -> Dict[str, str]:
    item = Path(path)
    return {
        "path": str(item.resolve()),
        "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
    }


def git_state() -> Dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip())
        return {"commit": commit, "dirty_worktree": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty_worktree": None}


def model_config(generator: torch.nn.Module) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for attr in (
        "vocab_size", "embedding_dim", "hidden_dim", "latent_dim", "max_length",
        "num_layers", "num_heads", "dropout", "condition_dim", "mem_tokens",
        "esm_dim", "gat_heads", "gat_window", "use_gat", "fusion_type",
        "pad_idx", "sos_idx", "eos_idx", "bidirectional", "use_attention",
    ):
        if hasattr(generator, attr):
            output[attr] = getattr(generator, attr)
    return output


def stream_inputs(
    esm: Optional[ESM2HF],
    sequences,
    attention_contacts: bool,
    threshold: float,
    max_length: int,
) -> Tuple[Any, Any, Any]:
    if esm is None:
        return None, None, None
    with torch.no_grad():
        if attention_contacts:
            output = esm.contact_graph(
                sequences, thresh=threshold, local_window=2, max_length=max_length
            )
            return output["tokens"], output["mask"], output["adj"]
        output = esm.embed(sequences, return_tokens=True, max_length=max_length)
        return output["tokens"], output["mask"], None


def validation_loss(
    generator,
    loader,
    esm,
    device,
    conditional,
    attention_contacts,
    contact_threshold,
    esm_max_length,
    pad_idx,
) -> float:
    generator.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            sequences = batch["sequence"]
            target_in = batch["input_ids"].to(device)
            target_out = batch["target_ids"].to(device)
            condition = batch["condition"].to(device) if conditional else None
            esm_tokens, esm_mask, contact_adj = stream_inputs(
                esm, sequences, attention_contacts, contact_threshold, esm_max_length
            )
            # Fixed validation latent removes one source of model-selection noise.
            z = torch.zeros(target_in.size(0), generator.latent_dim, device=device)
            logits = generator(
                z,
                target=target_in,
                condition=condition,
                esm_tokens=esm_tokens,
                esm_mask=esm_mask,
                contact_adj=contact_adj,
            )["logits"]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_out.reshape(-1),
                ignore_index=pad_idx,
            )
            total += float(loss.item()) * target_in.size(0)
            count += target_in.size(0)
    generator.train()
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--all-labels", action="store_true",
                        help="diagnostic only; default reconstructs AMP label=1")
    parser.add_argument("--dataset-report", default=None,
                        help="auditable dataset_build_report.json (default: beside train CSV)")
    parser.add_argument("--allow-unverified-data", action="store_true",
                        help="smoke-test only: continue without a valid dataset build report")
    parser.add_argument("--no-esm", action="store_true", help="ESM-2 ablation")
    parser.add_argument("--no-gat", action="store_true", help="GATv2 ablation")
    parser.add_argument(
        "--fusion-type", choices=["cross_attention", "concat", "none"],
        default="cross_attention",
    )
    parser.add_argument("--esm-model", default="esm2_t12_35M_UR50D")
    parser.add_argument("--esm-model-revision", default=None,
                        help="immutable Hugging Face commit (or esm2.model_revision from config)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--esm-max-length", type=int, default=64)
    parser.add_argument(
        "--esm-attention-contacts", action="store_true",
        help="threshold ESM-2 attention contact probabilities for GATv2",
    )
    parser.add_argument(
        "--contact-graph", action="store_true",
        help="deprecated alias for --esm-attention-contacts",
    )
    parser.add_argument("--contact-thresh", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out", default="checkpoints/warmup.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    attention_contacts = args.esm_attention_contacts or args.contact_graph
    if attention_contacts and args.no_esm:
        parser.error("attention-derived contacts require ESM-2")
    if attention_contacts and args.no_gat:
        parser.error("contact graphs are unused when --no-gat is set")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")

    set_seed(args.seed)
    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    esm_revision = args.esm_model_revision or config.get("esm2", {}).get("model_revision")
    if not args.no_esm and not esm_revision and not args.allow_unverified_data:
        parser.error(
            "reportable ESM-2 warm-up requires --esm-model-revision or "
            "esm2.model_revision in the config"
        )
    train_csv = data_cfg.get("train_csv", "dataset/train.csv")
    val_csv = data_cfg.get("val_csv", "dataset/val.csv")
    test_csv = data_cfg.get("test_csv", "dataset/test.csv")
    dataset_audit = None
    try:
        dataset_audit = validate_dataset_build(
            train_csv, val_csv, test_csv, report_path=args.dataset_report
        )
    except (ValueError, FileNotFoundError) as exc:
        if not args.allow_unverified_data:
            parser.error(str(exc))
        logger.warning("UNVERIFIED DATA smoke-test mode: %s", exc)
    label_value = None if args.all_labels else 1
    device = get_device()

    train_dataset = ConditionalPeptideDataset.from_csv(
        train_csv,
        label_value=label_value,
        feature_names=data_cfg.get("condition_features"),
        max_length=data_cfg.get("max_seq_length", 50),
        min_length=data_cfg.get("min_seq_length", 5),
    )
    validation_dataset = ConditionalPeptideDataset.from_csv(
        val_csv,
        label_value=label_value,
        feature_names=data_cfg.get("condition_features"),
        max_length=data_cfg.get("max_seq_length", 50),
        min_length=data_cfg.get("min_seq_length", 5),
        feature_stats=train_dataset.get_feature_stats(),
    )
    if args.max_samples:
        for dataset, limit in (
            (train_dataset, args.max_samples),
            (validation_dataset, max(1, args.max_samples // 5)),
        ):
            dataset.sequences = dataset.sequences[:limit]
            dataset.features = dataset.features[:limit]
            if dataset.has_labels:
                dataset.labels = dataset.labels[:limit]

    condition_dim = train_dataset.get_condition_dim() if args.conditional else None
    configured_condition_dim = config.get("generator", {}).get("condition_dim")
    if args.conditional and configured_condition_dim is not None and configured_condition_dim != condition_dim:
        parser.error(
            f"generator.condition_dim={configured_condition_dim} conflicts with the "
            f"{condition_dim} configured condition_features"
        )
    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    validation_loader = get_dataloader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False,
    )

    esm = None if args.no_esm else ESM2HF(
        model_name=args.esm_model, model_revision=esm_revision,
        device=device, freeze=True,
    )
    esm_dim = None if esm is None else esm.embed_dim
    logger.info(
        "Device=%s | ESM=%s | GAT=%s | fusion=%s | attention_contacts=%s",
        device, "disabled" if esm is None else args.esm_model,
        not args.no_gat, args.fusion_type, attention_contacts,
    )

    generator = MultimodalFusionGenerator(
        vocab_size=VOCAB.vocab_size,
        embedding_dim=model_cfg.get("embedding_dim", 128),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        latent_dim=model_cfg.get("latent_dim", 128),
        max_length=data_cfg.get("max_seq_length", 50),
        num_layers=model_cfg.get("num_layers", 3),
        num_heads=config.get("generator", {}).get("num_heads", 4),
        dropout=model_cfg.get("dropout", 0.2),
        condition_dim=condition_dim,
        mem_tokens=model_cfg.get("mem_tokens", 16),
        gat_heads=config.get("structure_evaluator", {}).get("gat_heads", 4),
        gat_window=model_cfg.get("gat_window", 3),
        use_gat=not args.no_gat,
        fusion_type=args.fusion_type,
        esm_dim=esm_dim,
    ).to(device)
    logger.info("Generator trainable parameters: %s", f"{sum(p.numel() for p in generator.parameters()):,}")

    optimizer = AdamW(generator.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    generator.train()

    for epoch in range(1, args.epochs + 1):
        total = 0.0
        batches = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            sequences = batch["sequence"]
            target_in = batch["input_ids"].to(device)
            target_out = batch["target_ids"].to(device)
            condition = batch["condition"].to(device) if args.conditional else None
            esm_tokens, esm_mask, contact_adj = stream_inputs(
                esm, sequences, attention_contacts, args.contact_thresh, args.esm_max_length
            )
            z = torch.randn(target_in.size(0), generator.latent_dim, device=device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = generator(
                    z,
                    target=target_in,
                    condition=condition,
                    esm_tokens=esm_tokens,
                    esm_mask=esm_mask,
                    contact_adj=contact_adj,
                )["logits"]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_out.reshape(-1),
                    ignore_index=VOCAB.pad_idx,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
            batches += 1
            if batch_index % 50 == 0:
                logger.info("epoch %d batch %d/%d train_CE=%.4f", epoch, batch_index, len(train_loader), total / batches)

        train_ce = total / max(batches, 1)
        val_ce = validation_loss(
            generator, validation_loader, esm, device, args.conditional,
            attention_contacts, args.contact_thresh, args.esm_max_length, VOCAB.pad_idx,
        )
        history.append({"epoch": epoch, "train_cross_entropy": train_ce, "validation_cross_entropy": val_ce})
        logger.info("epoch %d/%d train_CE=%.4f validation_CE=%.4f", epoch, args.epochs, train_ce, val_ce)
        if val_ce < best_loss:
            best_loss = val_ce
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in generator.state_dict().items()})

    if best_state is None:
        raise RuntimeError("warm-up produced no checkpoint candidate")
    generator.load_state_dict(best_state)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    git = git_state()
    artifact_reportable = bool(
        dataset_audit is not None
        and (args.no_esm or esm_revision)
        and args.max_samples is None
        and git.get("commit")
        and git.get("dirty_worktree") is False
    )
    payload = {
        "epoch": 0,
        "global_step": 0,
        "generator_class": type(generator).__name__,
        "generator": generator.state_dict(),
        "model_config": model_config(generator),
        "config": config,
        "warmup": True,
        "artifact_reportable": artifact_reportable,
        "warmup_config": {
            "epochs_completed": args.epochs,
            "best_epoch": best_epoch,
            "best_validation_cross_entropy": best_loss,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
            "esm_model": None if args.no_esm else args.esm_model,
            "esm_model_revision": None if args.no_esm else esm_revision,
            "contact_graph_type": "esm2_attention_probability" if attention_contacts else None,
            "contact_probability_threshold": args.contact_thresh if attention_contacts else None,
            "generator_label_filter": label_value,
        },
        "data_metadata": {
            "train": file_record(train_csv),
            "validation": file_record(val_csv),
            "condition_feature_names": list(train_dataset.feature_names),
            "condition_feature_stats": train_dataset.get_feature_stats(),
            "dataset_build_audit": dataset_audit,
            "reportable_data": dataset_audit is not None,
            "esm_feature_extractor": (
                None if args.no_esm else {
                    "model": args.esm_model,
                    "revision": esm_revision,
                    "frozen": True,
                }
            ),
        },
        "run_metadata": {
            "command": shlex.join(sys.argv),
            "seed": args.seed,
            "git": git,
            "artifact_reportable": artifact_reportable,
        },
        "history": history,
    }
    torch.save(payload, output)
    logger.info("Saved best warm-up checkpoint (epoch=%d, validation_CE=%.4f) -> %s", best_epoch, best_loss, output)


if __name__ == "__main__":
    main()
