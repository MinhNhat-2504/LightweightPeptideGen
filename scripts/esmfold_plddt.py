#!/usr/bin/env python
"""
B5 (rigorous) — ESMFold pLDDT + secondary-structure on generated peptides.

This is the *strongest* independent structural metric for the paper: it folds
each generated sequence with ESMFold and reports the mean pLDDT (per-residue
confidence) and, optionally, secondary-structure fractions. pLDDT is completely
independent of the Instability Index the generator is trained on.

ESMFold (~2.8 B params) does NOT fit on an 8 GB laptop GPU — run this on Colab
or any GPU with >=16 GB. The light-weight ESM-2 foldability metrics in
``peptidegen/evaluation/foldability.py`` run locally instead.

Colab setup
-----------
    pip install "transformers>=4.30" accelerate
    # optional secondary structure: pip install biotite

Usage
-----
    python scripts/esmfold_plddt.py --input results/generated.fasta \
        --output results/esmfold_plddt.csv --max-seqs 1000

Output CSV columns: id, sequence, length, mean_plddt, ptm
(+ helix_frac, sheet_frac, coil_frac if biotite is installed).
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
    ap.add_argument("--max-seqs", type=int, default=1000)
    ap.add_argument("--model", default="facebook/esmfold_v1")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("ESMFold on CPU is extremely slow; a >=16 GB GPU is recommended.")

    logger.info(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmForProteinFolding.from_pretrained(args.model)
    model = model.to(device)
    model.esm = model.esm.half()  # halve the language-model trunk to save VRAM
    model.eval()

    seqs = read_fasta(args.input)[: args.max_seqs]
    logger.info(f"Folding {len(seqs)} sequences")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for k, (name, seq) in enumerate(seqs):
            seq = "".join(c for c in seq.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
            if len(seq) < 5:
                continue
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
        logger.info(f"Mean pLDDT = {arr.mean():.2f} ± {arr.std():.2f}  "
                    f"(>=70 'confident': {(arr>=70).mean()*100:.1f}%)")
    else:
        logger.warning("No sequences folded.")
        sys.exit(1)


if __name__ == "__main__":
    main()
