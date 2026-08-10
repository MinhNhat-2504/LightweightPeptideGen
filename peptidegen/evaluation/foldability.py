"""
B5 — Independent structural / foldability metrics.

The generator is trained to minimise the dipeptide-based Instability Index
(II), so reporting "II < 40 stability rate" as the headline evidence of
structure is *circular*. This module provides metrics that are **independent
of II** and run on commodity hardware (RTX 4060, 8 GB):

    * ``esm_pseudo_perplexity`` — ESM-2 pseudo-perplexity. Lower = more
      evolutionarily plausible. Independent of II.
    * ``esm_contact_order``     — mean predicted long-range contact probability
      from ESM-2 attention maps. Higher = more predicted tertiary structure.
    * ``helix_fraction``        — Chou–Fasman helix propensity fraction
      (amphipathic helix is the canonical AMP motif).

For the most rigorous (but heavier) evidence, ``scripts/esmfold_plddt.py``
produces ESMFold pLDDT + secondary structure on a larger GPU / Colab.

Usage
-----
    from peptidegen.evaluation.foldability import FoldabilityEvaluator
    fe = FoldabilityEvaluator(model_name="esm2_t12_35M_UR50D")
    report = fe.evaluate(sequences)          # dict of mean/std + per-seq arrays
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _chou_fasman_helix_fraction(seq: str) -> float:
    from .stability import calculate_secondary_structure_propensity

    return calculate_secondary_structure_propensity(seq)["helix_fraction"]


class FoldabilityEvaluator:
    """Independent structural metrics backed by a frozen ESM-2 (HF)."""

    def __init__(
        self,
        model_name: str = "esm2_t12_35M_UR50D",
        device=None,
        max_length: int = 64,
    ):
        from ..models.esm2_hf import ESM2HF

        self.embedder = ESM2HF(model_name=model_name, device=device, freeze=True)
        self.max_length = max_length

    # ------------------------------------------------------------------ #
    def esm_contact_order(self, sequence: str, sep: int = 3) -> float:
        """Mean of medium/long-range (|i-j| >= sep) ESM-2 contact probabilities."""
        try:
            out = self.embedder.embed(
                [sequence], return_tokens=False, return_contacts=True,
                max_length=self.max_length,
            )
        except Exception:
            return float("nan")
        if "contacts" not in out:
            return float("nan")
        cmap = out["contacts"][0].cpu().numpy()      # (L, L)
        L = cmap.shape[0]
        if L <= sep:
            return 0.0
        ii, jj = np.triu_indices(L, k=sep)
        return float(cmap[ii, jj].mean())

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        sequences: List[str],
        sample: Optional[int] = 1000,
        compute_contacts: bool = True,
    ) -> Dict:
        """Compute per-sequence structural metrics + summary statistics."""
        seqs = [s for s in sequences if isinstance(s, str) and len(s) >= 5]
        if sample and len(seqs) > sample:
            rng = np.random.default_rng(42)
            seqs = [seqs[i] for i in rng.choice(len(seqs), sample, replace=False)]

        ppl, contact, helix = [], [], []
        for i, s in enumerate(seqs):
            ppl.append(self.embedder.pseudo_perplexity(s, max_length=self.max_length))
            if compute_contacts:
                contact.append(self.esm_contact_order(s))
            helix.append(_chou_fasman_helix_fraction(s))
            if i % 100 == 0:
                logger.info(f"foldability {i}/{len(seqs)}")

        def summ(a):
            a = np.asarray([x for x in a if x == x], dtype=float)  # drop NaN
            if a.size == 0:
                return {"mean": float("nan"), "std": float("nan"), "n": 0}
            return {"mean": float(a.mean()), "std": float(a.std()),
                    "median": float(np.median(a)), "n": int(a.size)}

        return {
            "n_sequences": len(seqs),
            "esm_model": self.embedder.model_name,
            "esm_pseudo_perplexity": summ(ppl),
            "esm_contact_order": summ(contact) if compute_contacts else None,
            "helix_fraction": summ(helix),
            "_per_sequence": {
                "pseudo_perplexity": ppl,
                "contact_order": contact if compute_contacts else [],
                "helix_fraction": helix,
            },
        }
