#!/usr/bin/env python
"""
A4 — SCST reinforcement-learning fine-tuning with a multi-objective reward
(empirical II screen + AMP-oracle score + low hemolysis-oracle score).

Runs *after* GAN training. Loads a fusion-generator checkpoint, optimises the
self-critical policy gradient (paper Eq.10), and saves the fine-tuned generator.
The AMP / hemolysis oracles trained in B6/B7 are used as the reward models;
without them it falls back to physicochemical heuristics (faster, weaker).

Usage
-----
    python scripts/scst_finetune.py \
        --checkpoint checkpoints/best_model.pt \
        --amp-oracle results/oracle_amp.pkl \
        --hemo-oracle results/oracle_hemo.pkl \
        --steps 2000 --batch-size 64 --lr 1e-5 \
        --out checkpoints/scst_model.pt
"""

import argparse
import hashlib
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from peptidegen.data import VOCAB
from peptidegen.inference import PeptideSampler
from peptidegen.training import SCSTTrainer, MultiObjectiveReward


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_oracle(path: str, expected_task: str) -> dict:
    checkpoint = Path(path)
    report_path = checkpoint.with_name(checkpoint.stem + "_report.json")
    if not checkpoint.is_file() or not report_path.is_file():
        raise ValueError(f"{expected_task} oracle checkpoint/report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("reportable") is not True:
        raise ValueError(f"{report_path}: oracle is not marked reportable")
    if report.get("task") != expected_task:
        raise ValueError(f"{report_path}: expected task={expected_task}")
    if not report.get("model_revision"):
        raise ValueError(f"{report_path}: missing immutable ESM-2 model revision")
    if ((report.get("checkpoint") or {}).get("sha256")) != _sha256(checkpoint):
        raise ValueError(f"{report_path}: oracle checkpoint hash mismatch")
    audit = report.get("dataset_build_audit") or {}
    if not audit.get("sha256"):
        raise ValueError(f"{report_path}: missing dataset-build provenance")
    git = ((report.get("run_metadata") or {}).get("git") or {})
    if not git.get("commit") or git.get("dirty_worktree") is not False:
        raise ValueError(f"{report_path}: oracle was not trained from a clean immutable commit")
    return {
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": _sha256(checkpoint)},
        "report": {"path": str(report_path.resolve()), "sha256": _sha256(report_path)},
        "task": expected_task,
        "model": report.get("model"),
        "model_revision": report.get("model_revision"),
        "dataset_build_report_sha256": audit.get("sha256"),
        "code_commit": git.get("commit"),
    }


def git_state() -> dict:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="GAN generator checkpoint")
    ap.add_argument("--amp-oracle", default=None)
    ap.add_argument("--hemo-oracle", default=None)
    ap.add_argument("--oracle-id", default="ESM2Oracle_amp",
                    help="exact identity of the AMP reward oracle, stored in the checkpoint")
    ap.add_argument("--hemo-oracle-id", default="ESM2Oracle_hemolysis",
                    help="exact identity of the hemolysis reward oracle")
    ap.add_argument("--allow-heuristic-reward", action="store_true",
                    help="allow missing reward oracles for smoke tests only")
    ap.add_argument("--allow-unverified-data", action="store_true",
                    help="smoke-test only: allow a parent checkpoint without verified dataset provenance")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument(
        "--w-ii-screen", "--w-stability", dest="w_ii_screen",
        type=float, default=0.34,
        help="weight of the empirical Instability-Index screen reward; "
             "--w-stability is a deprecated compatibility alias",
    )
    ap.add_argument("--w-amp", type=float, default=0.33)
    ap.add_argument("--w-hemolysis", type=float, default=0.33)
    ap.add_argument("--target-ii", type=float, default=40.0)
    ap.add_argument("--entropy-coef", type=float, default=0.02,
                    help="entropy bonus weight; >0 fights mode-collapse/low diversity")
    ap.add_argument("--condition-csv", default="dataset/train.csv",
                    help="sample normalized conditions from this training split")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out", default="checkpoints/scst_model.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.steps <= 0 or args.batch_size <= 0:
        ap.error("--steps and --batch-size must be positive")
    weights = (args.w_ii_screen, args.w_amp, args.w_hemolysis)
    if min(weights) < 0 or not np.isclose(sum(weights), 1.0, atol=1e-6):
        ap.error("SCST weights must be non-negative and sum to 1.0")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reuse the sampler loader to rebuild the exact generator from the ckpt
    sampler = PeptideSampler.from_checkpoint(args.checkpoint, device=device)
    G = sampler.G

    amp_oracle = hemo_oracle = None
    amp_oracle_audit = hemo_oracle_audit = None
    if args.amp_oracle:
        from peptidegen.evaluation.oracle import ESM2Oracle
        try:
            amp_oracle_audit = validate_oracle(args.amp_oracle, "amp")
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            if not args.allow_unverified_data:
                ap.error(str(exc))
            logger.warning("UNVERIFIED AMP oracle: %s", exc)
        amp_oracle = ESM2Oracle.load(args.amp_oracle, device=device)
    if args.hemo_oracle:
        from peptidegen.evaluation.oracle import ESM2Oracle
        try:
            hemo_oracle_audit = validate_oracle(args.hemo_oracle, "hemo")
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            if not args.allow_unverified_data:
                ap.error(str(exc))
            logger.warning("UNVERIFIED hemolysis oracle: %s", exc)
        hemo_oracle = ESM2Oracle.load(args.hemo_oracle, device=device)
    missing = []
    if args.w_amp > 0 and amp_oracle is None:
        missing.append("AMP")
    if args.w_hemolysis > 0 and hemo_oracle is None:
        missing.append("hemolysis")
    if missing and not args.allow_heuristic_reward:
        ap.error(
            "missing reward oracle(s): " + ", ".join(missing) +
            ". Supply the checkpoints, set the corresponding weight to zero, or use "
            "--allow-heuristic-reward for a non-reportable smoke test."
        )
    if missing:
        logger.warning("Heuristic reward fallback enabled for: %s", ", ".join(missing))

    reward = MultiObjectiveReward(
        amp_oracle=amp_oracle, hemo_oracle=hemo_oracle,
        oracle_id=args.oracle_id,
        w_stability=args.w_ii_screen, w_amp=args.w_amp, w_hemolysis=args.w_hemolysis,
        target_ii=args.target_ii,
        use_heuristic_fallback=args.allow_heuristic_reward,
    )
    trainer = SCSTTrainer(G, reward, VOCAB, device=device, lr=args.lr,
                          entropy_bonus=args.entropy_coef)

    cond_dim = getattr(G, "condition_dim", None)
    condition_pool = None
    condition_metadata = None
    parent_data = (getattr(sampler, "checkpoint_metadata", {}) or {}).get("data_metadata") or {}
    parent_reportable = (
        (getattr(sampler, "checkpoint_metadata", {}) or {}).get("artifact_reportable") is True
    )
    if not parent_reportable and not args.allow_unverified_data:
        ap.error(
            "parent checkpoint is not marked artifact_reportable; use "
            "--allow-unverified-data only for a non-reportable smoke test"
        )
    if parent_data.get("reportable_data") is not True and not args.allow_unverified_data:
        ap.error(
            "parent checkpoint does not prove a verified dataset build; use "
            "--allow-unverified-data only for a non-reportable smoke test"
        )
    if cond_dim:
        import pandas as pd
        from peptidegen.data.dataset import ConditionalPeptideDataset

        frame = pd.read_csv(args.condition_csv)
        if "label" in frame.columns:
            frame = frame.loc[frame["label"] == 1].copy()
        if frame.empty:
            ap.error(f"{args.condition_csv} contains no AMP-labelled rows for SCST conditioning")
        feature_names = parent_data.get("condition_feature_names")
        if not feature_names:
            if not args.allow_unverified_data:
                ap.error("parent checkpoint does not record condition_feature_names")
            feature_names = [
                name for name in ConditionalPeptideDataset.CONDITION_FEATURES
                if name in frame.columns
            ]
        feature_names = list(feature_names)
        if len(feature_names) != cond_dim:
            ap.error(
                f"checkpoint expects condition_dim={cond_dim}, but {args.condition_csv} "
                f"provides {len(feature_names)} configured features: {feature_names}"
            )
        missing_features = [name for name in feature_names if name not in frame.columns]
        if missing_features:
            ap.error(f"{args.condition_csv} is missing checkpoint condition features: {missing_features}")
        if frame[feature_names].isna().any().any():
            ap.error(f"{args.condition_csv} contains missing condition features")
        raw = frame[feature_names].to_numpy(dtype=np.float32)
        saved_stats = parent_data.get("condition_feature_stats")
        if not saved_stats:
            if not args.allow_unverified_data:
                ap.error("parent checkpoint does not record condition_feature_stats")
            means = raw.mean(axis=0)
            stds = raw.std(axis=0) + 1e-8
        else:
            missing_stats = [name for name in feature_names if name not in saved_stats]
            if missing_stats:
                ap.error(f"parent checkpoint is missing normalization statistics: {missing_stats}")
            means = np.asarray([saved_stats[name]["mean"] for name in feature_names], dtype=np.float32)
            stds = np.asarray([saved_stats[name]["std"] for name in feature_names], dtype=np.float32)
            if not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds <= 0).any():
                ap.error("parent checkpoint contains invalid condition normalization statistics")
        condition_pool = torch.from_numpy((raw - means) / stds)
        condition_path = Path(args.condition_csv)
        expected_train = (parent_data.get("train") or {}).get("sha256")
        actual_train = hashlib.sha256(condition_path.read_bytes()).hexdigest()
        if expected_train and expected_train != actual_train:
            ap.error(f"condition CSV hash does not match parent checkpoint: {args.condition_csv}")
        condition_metadata = {
            "path": str(condition_path.resolve()),
            "sha256": actual_train,
            "feature_names": feature_names,
            "means": means.tolist(),
            "stds": stds.tolist(),
            "n_rows": int(len(raw)),
            "label_filter": 1 if "label" in frame.columns else None,
        }
    logger.info(f"SCST: {args.steps} steps, batch={args.batch_size}, lr={args.lr}, "
                f"entropy_coef={args.entropy_coef}, "
                f"weights II-screen/AMP/hemolysis="
                f"{args.w_ii_screen}/{args.w_amp}/{args.w_hemolysis}")

    run = {"scst_loss": 0.0, "reward_sample": 0.0, "advantage": 0.0, "entropy": 0.0}
    history = []
    for step in range(1, args.steps + 1):
        if condition_pool is not None:
            indices = torch.randint(0, len(condition_pool), (args.batch_size,))
            conditions = condition_pool[indices]
        else:
            conditions = None
        m = trainer.train_step(args.batch_size, conditions)
        history.append({"step": step, **m})
        for k in run:
            run[k] += m[k]
        if step % args.log_every == 0:
            n = args.log_every
            logger.info(f"step {step}/{args.steps} | loss={run['scst_loss']/n:.4f} "
                        f"| R_sample={run['reward_sample']/n:.4f} "
                        f"| adv={run['advantage']/n:.4f} "
                        f"| entropy={run['entropy']/n:.3f}")
            run = {k: 0.0 for k in run}

    # save (sampler-loadable format)
    model_config = {}
    for attr in ("vocab_size", "embedding_dim", "hidden_dim", "latent_dim",
                 "max_length", "num_layers", "num_heads", "dropout", "condition_dim",
                 "mem_tokens", "esm_dim", "gat_heads", "gat_window", "use_gat",
                 "fusion_type", "pad_idx", "sos_idx", "eos_idx",
                 "bidirectional", "use_attention"):
        if hasattr(G, attr):
            model_config[attr] = getattr(G, attr)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    def _file_record(path):
        if not path:
            return None
        item = Path(path)
        return {
            "path": str(item.resolve()),
            "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
        }

    git = git_state()
    parent_commit = (
        (((getattr(sampler, "checkpoint_metadata", {}) or {}).get("run_metadata") or {}).get("git") or {}).get("commit")
    )
    oracle_commits_match = all(
        item is None or item.get("code_commit") == git.get("commit")
        for item in (amp_oracle_audit, hemo_oracle_audit)
    )
    artifact_reportable = bool(
        parent_reportable
        and parent_data.get("reportable_data") is True
        and (args.w_amp == 0 or amp_oracle_audit is not None)
        and (args.w_hemolysis == 0 or hemo_oracle_audit is not None)
        and not args.allow_heuristic_reward
        and parent_commit == git.get("commit")
        and oracle_commits_match
        and git.get("commit")
        and git.get("dirty_worktree") is False
    )
    torch.save({"epoch": 0, "global_step": args.steps,
                "generator_class": type(G).__name__,
                "generator": G.state_dict(), "model_config": model_config,
                "scst": True,
                "artifact_reportable": artifact_reportable,
                "scst_config": {
                    "steps": args.steps,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "entropy_coefficient": args.entropy_coef,
                    "seed": args.seed,
                    "reward": reward.describe(),
                    "amp_oracle_id": args.oracle_id if amp_oracle is not None else None,
                    "hemolysis_oracle_id": args.hemo_oracle_id if hemo_oracle is not None else None,
                    "heuristic_fallback": args.allow_heuristic_reward,
                    "reportable_data": parent_data.get("reportable_data") is True,
                    "reportable_oracles": (
                        (args.w_amp == 0 or amp_oracle_audit is not None)
                        and (args.w_hemolysis == 0 or hemo_oracle_audit is not None)
                    ),
                },
                "condition_metadata": condition_metadata,
                "data_metadata": parent_data,
                "parent_checkpoint": _file_record(args.checkpoint),
                "amp_oracle": _file_record(args.amp_oracle),
                "hemolysis_oracle": _file_record(args.hemo_oracle),
                "amp_oracle_audit": amp_oracle_audit,
                "hemolysis_oracle_audit": hemo_oracle_audit,
                "run_metadata": {
                    "command": shlex.join(sys.argv),
                    "seed": args.seed,
                    "git": git,
                    "artifact_reportable": artifact_reportable,
                    "parent_code_commit": parent_commit,
                },
                "history": history,
                "optimizer": trainer.opt.state_dict()}, args.out)
    logger.info(f"Saved SCST-fine-tuned generator -> {args.out}")


if __name__ == "__main__":
    main()
