#!/usr/bin/env python
"""Run ESMFold and export prediction-confidence metrics with provenance.

pLDDT is the model's per-residue confidence in its predicted coordinates.  It
is independent of the Instability Index training surrogate, but it is not a
measurement of thermodynamic stability, folding free energy, activity, or
experimental structure quality for these generated peptides.

ESMFold (~2.8 B params) does NOT fit on an 8 GB laptop GPU — run this on Colab
or any GPU with >=16 GB. The light-weight ESM-2 foldability metrics in
``peptidegen/evaluation/foldability.py`` run locally instead.

Colab setup
-----------
    pip install "transformers>=4.30" accelerate
    # optional secondary structure: pip install biotite

Usage
-----
    python scripts/esmfold_plddt.py \
        --input results/gen/LightweightPeptideGen_seed42.fasta \
        --output results/esmfold/LightweightPeptideGen_seed42.csv --seed 42

Output CSV columns: id, sequence, length, mean_plddt, ptm
(+ helix_frac, sheet_frac, coil_frac if biotite is installed).
"""

import argparse
import csv
import hashlib
import json
import logging
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")
RUN_RE = re.compile(r"(?P<model>.+)_seed(?P<seed>\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fasta(path: str):
    seqs = []
    name, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    seqs.append((name, "".join(buf)))
                name, buf = line[1:].strip(), []
            elif line:
                buf.append(line)
    if name is not None:
        seqs.append((name, "".join(buf)))
    return seqs


def git_state():
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


def secondary_structure_fractions(positions, sequence):
    """Optional SS via biotite annotate_sse (P-SEA). Returns (helix, sheet, coil) or None."""
    try:
        import numpy as np
        import biotite.structure as struc
    except Exception:
        return None
    try:
        # positions: (L, 3) CA coordinates
        ca = struc.AtomArray(len(sequence))
        ca.coord = np.asarray(positions, dtype=float)
        ca.chain_id[:] = "A"
        ca.res_id[:] = np.arange(1, len(sequence) + 1)
        ca.res_name[:] = "GLY"
        ca.atom_name[:] = "CA"
        sse = struc.annotate_sse(ca)  # 'a' helix, 'b' sheet, 'c' coil
        n = max(len(sse), 1)
        return ((sse == "a").sum() / n, (sse == "b").sum() / n, (sse == "c").sum() / n)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="FASTA of generated sequences")
    ap.add_argument("--output", "-o", default="results/esmfold_plddt.csv")
    ap.add_argument("--max-seqs", type=int, default=None,
                    help="optional smoke-test cap; omit for reportable evaluation")
    ap.add_argument("--model", default="facebook/esmfold_v1")
    ap.add_argument("--model-revision", default=None,
                    help="Hugging Face commit/revision; required for a reportable frozen run")
    ap.add_argument("--seed", type=int, required=True,
                    help="generation seed represented by the input FASTA")
    ap.add_argument("--allow-unverified-input", action="store_true",
                    help="smoke-test only: permit FASTA without a reportable generation sidecar")
    args = ap.parse_args()
    if not args.model_revision and not args.allow_unverified_input:
        ap.error("--model-revision is required for a reportable ESMFold run")

    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("ESMFold on CPU is extremely slow; a >=16 GB GPU is recommended.")

    logger.info(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    model = EsmForProteinFolding.from_pretrained(args.model, revision=args.model_revision)
    model = model.to(device)
    model.esm = model.esm.half()  # halve the language-model trunk to save VRAM
    model.eval()

    input_path = Path(args.input)
    run_match = RUN_RE.fullmatch(input_path.stem)
    if not run_match and not args.allow_unverified_input:
        ap.error("reportable input name must match <model>_seed<N>.fasta")
    generation_sidecar_path = input_path.with_suffix(input_path.suffix + ".metadata.json")
    generation_sidecar = None
    try:
        generation_sidecar = json.loads(generation_sidecar_path.read_text(encoding="utf-8"))
        generated_output = generation_sidecar.get("output") or {}
        if (
            generation_sidecar.get("reportable") is not True
            or generated_output.get("sha256") != sha256(input_path)
            or int(generation_sidecar.get("seed", -1)) != args.seed
            or (run_match and generation_sidecar.get("model_id") != run_match.group("model"))
            or (run_match and int(run_match.group("seed")) != args.seed)
        ):
            raise ValueError("generation sidecar is non-reportable or does not match input/seed")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if not args.allow_unverified_input:
            ap.error(f"invalid generation provenance for {input_path}: {exc}")
        logger.warning("UNVERIFIED ESMFold input: %s", exc)
        generation_sidecar = None
    seqs = read_fasta(args.input)
    invalid = [
        (name, seq) for name, seq in seqs
        if not (5 <= len(seq) <= 50 and set(seq.upper()) <= CANONICAL)
    ]
    if invalid:
        ap.error(f"input contains {len(invalid)} non-canonical/out-of-range sequences")
    if args.max_seqs is not None:
        seqs = seqs[: args.max_seqs]
    logger.info(f"Folding {len(seqs)} sequences")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for k, (name, seq) in enumerate(seqs):
            seq = seq.upper()
            ids = tokenizer([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            out = model(ids)
            plddt = out["plddt"]                      # (1, L, 37) or (1, L)
            if plddt.dim() == 3:
                # confidence on CA atom (index 1)
                plddt = plddt[..., 1]
            mean_plddt = float(plddt.mean().item())
            ptm = float(out["ptm"].item()) if "ptm" in out else float("nan")

            row = {"id": name, "sequence": seq, "length": len(seq),
                   "mean_plddt": round(mean_plddt, 2), "ptm": round(ptm, 4)}

            try:
                ca = out["positions"][-1, 0, :, 1, :].cpu().numpy()  # (L,3) CA
                ss = secondary_structure_fractions(ca, seq)
                if ss:
                    row.update(helix_frac=round(ss[0], 3),
                               sheet_frac=round(ss[1], 3),
                               coil_frac=round(ss[2], 3))
            except Exception:
                pass

            rows.append(row)
            if k % 25 == 0:
                logger.info(f"{k}/{len(seqs)}  mean_pLDDT={mean_plddt:.1f}")

    if rows:
        import numpy as np
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        arr = np.array([r["mean_plddt"] for r in rows])
        logger.info(f"Wrote {len(rows)} rows -> {args.output}")
        git = git_state()
        generated_commit = (((generation_sidecar or {}).get("software") or {}).get("git") or {}).get("commit")
        metadata = {
            "schema_version": 1,
            "reportable": bool(
                args.max_seqs is None
                and args.model_revision is not None
                and generation_sidecar is not None
                and git.get("commit")
                and git.get("dirty_worktree") is False
                and generated_commit == git.get("commit")
            ),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "interpretation": (
                "ESMFold pLDDT is prediction confidence, not thermodynamic or "
                "experimental structural stability."
            ),
            "input_fasta": str(input_path.resolve()),
            "input_fasta_sha256": sha256(input_path),
            "generation_metadata_path": str(generation_sidecar_path.resolve()),
            "generation_metadata_sha256": (
                sha256(generation_sidecar_path) if generation_sidecar is not None else None
            ),
            "output_csv": str(Path(args.output).resolve()),
            "output_csv_sha256": sha256(Path(args.output)),
            "seed": args.seed,
            "model": args.model,
            "model_revision": args.model_revision,
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "device": str(device),
            "python_version": platform.python_version(),
            "git": git,
            "input_rows": len(read_fasta(args.input)),
            "evaluated_rows": len(rows),
            "mean_plddt": float(arr.mean()),
            "fraction_plddt_ge_70": float((arr >= 70).mean()),
        }
        sidecar = Path(args.output).with_suffix(".metadata.json")
        sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Mean pLDDT prediction confidence = %.2f; fraction >=70 = %.1f%%",
                    arr.mean(), (arr >= 70).mean() * 100)
        logger.info("Metadata -> %s (reportable=%s)", sidecar, metadata["reportable"])
    else:
        logger.warning("No sequences folded.")
        sys.exit(1)


if __name__ == "__main__":
    main()
