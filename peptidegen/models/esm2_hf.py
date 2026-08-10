"""
HuggingFace-backed ESM-2 wrapper.

The original ``esm2_embedder.py`` relies on the ``fair-esm`` package
(``import esm``). On many machines only ``transformers`` is installed, so this
module provides an equivalent, dependency-light wrapper built on
``transformers`` (``facebook/esm2_*``). It is shared by:

    * A1 — the multimodal-fusion generator (semantic stream H_seq),
    * B5 — ESM-2 foldability metrics (pseudo-perplexity, contact confidence),
    * B6 — the ESM-2 classifier oracle (mean-pooled embeddings).

Key features
------------
* Frozen by default (eval mode, ``requires_grad=False``) — the paper keeps the
  backbone frozen and trains only the projection.
* Token-level embeddings, mean-pooled embeddings, masked-LM logits and
  (optionally) attention-map contacts in a single forward.
* On-disk embedding cache keyed by ``(model_name, sequence)`` so the 129k-row
  corpus only has to be embedded once.

Model name accepts either the short fair-esm style id
(``esm2_t12_35M_UR50D``) or a full HF id (``facebook/esm2_t12_35M_UR50D``).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# embed_dim per variant — handy for sizing projection layers without loading.
ESM2_EMBED_DIM = {
    "esm2_t6_8M_UR50D": 320,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t36_3B_UR50D": 2560,
}


def _to_hf_id(model_name: str) -> str:
    return model_name if "/" in model_name else f"facebook/{model_name}"


class ESM2HF(nn.Module):
    """Frozen ESM-2 feature extractor backed by ``transformers``."""

    def __init__(
        self,
        model_name: str = "esm2_t12_35M_UR50D",
        device: Optional[torch.device] = None,
        freeze: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        from transformers import AutoTokenizer, EsmModel

        self.model_name = model_name
        self.hf_id = _to_hf_id(model_name)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        # fp16 on GPU keeps the 650M model inside 8 GB; fp32 on CPU.
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        # eager attention so output_attentions / predict_contacts work (peptides
        # are short, so the speed cost is negligible).
        self.model = EsmModel.from_pretrained(
            self.hf_id, add_pooling_layer=False, attn_implementation="eager"
        )
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.embed_dim = self.model.config.hidden_size

        self.frozen = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        logger.info(f"Loaded ESM-2 (HF) {self.hf_id} embed_dim={self.embed_dim} dtype={self.dtype}")

    # ------------------------------------------------------------------ #
    # Core forward
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def embed(
        self,
        sequences: List[str],
        return_tokens: bool = True,
        return_logits: bool = False,
        return_contacts: bool = False,
        max_length: int = 64,
    ) -> Dict[str, torch.Tensor]:
        """Encode a list of peptide strings.

        Returns a dict with (subset of):
            ``pooled``   (B, embed_dim)        mean over real residues
            ``tokens``   (B, L, embed_dim)     per-residue (incl. CLS/EOS)
            ``mask``     (B, L)                1 for real residue tokens
            ``logits``   (B, L, vocab)         masked-LM logits (return_logits)
            ``contacts`` list[(Li, Li)]        per-seq contact probs (return_contacts)
            ``input_ids``(B, L)
        """
        enc = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attn,
            output_attentions=return_contacts,
        )
        tok = out.last_hidden_state.float()  # (B, L, D)

        # Residue mask: drop CLS (first), EOS/pad. ESM special ids: cls/pad/eos.
        sp = self.tokenizer
        special = {sp.cls_token_id, sp.eos_token_id, sp.pad_token_id}
        res_mask = attn.clone().bool()
        for sid in special:
            if sid is not None:
                res_mask &= input_ids != sid
        res_mask_f = res_mask.unsqueeze(-1).float()
        pooled = (tok * res_mask_f).sum(1) / res_mask_f.sum(1).clamp(min=1)

        result: Dict[str, torch.Tensor] = {
            "pooled": pooled,
            "input_ids": input_ids,
            "mask": res_mask.float(),
        }
        if return_tokens:
            result["tokens"] = tok

        if return_logits:
            # EsmModel has no LM head; load the masked-LM head lazily.
            result["logits"] = self._lm_logits(input_ids, attn)

        if return_contacts:
            # transformers exposes predict_contacts on EsmModel via attentions.
            try:
                contacts = self.model.predict_contacts(input_ids, attn)  # (B, L', L')
                result["contacts"] = contacts.float()
            except Exception as e:  # pragma: no cover - model-version dependent
                logger.warning(f"predict_contacts unavailable: {e}")
        return result

    @torch.no_grad()
    def contact_graph(
        self,
        sequences: List[str],
        thresh: float = 0.5,
        knn: int = 0,
        local_window: int = 2,
        max_length: int = 64,
    ) -> Dict[str, torch.Tensor]:
        """Build a residue KNN/contact adjacency from ESM-2 attention contacts.

        This is the local (8 GB) surrogate for the paper's "KNN interaction
        graph with a contact radius < 8 Å": ESM-2 predicted contact
        probabilities stand in for spatial proximity (no MD, no folding).

        Args:
            thresh: connect (i,j) if contact prob >= thresh.
            knn: if >0, instead connect each residue to its top-``knn`` contacts.
            local_window: always connect |i-j| <= window (backbone chain), so the
                graph is never empty (avoids degenerate attention).

        Returns dict: ``tokens`` (B,L,D), ``mask`` (B,L), ``adj`` (B,L,L) float
        (symmetric, with self-loops; padded/special positions keep only a
        self-loop so GATv2 softmax never sees an all-masked row).
        """
        out = self.embed(sequences, return_tokens=True, return_contacts=True,
                         max_length=max_length)
        tokens, mask = out["tokens"], out["mask"]
        B, L, _ = tokens.shape
        device = tokens.device

        if "contacts" in out:
            cmap = out["contacts"]
            # align contact map to token length L if needed
            if cmap.size(-1) != L:
                c = torch.zeros(B, L, L, device=device)
                m = min(L, cmap.size(-1))
                c[:, :m, :m] = cmap[:, :m, :m]
                cmap = c
        else:
            cmap = torch.zeros(B, L, L, device=device)

        if knn and knn > 0:
            adj = torch.zeros(B, L, L, device=device)
            topk = torch.topk(cmap, min(knn, L), dim=-1).indices
            adj.scatter_(-1, topk, 1.0)
        else:
            adj = (cmap >= thresh).float()

        # backbone chain window + symmetry + self-loops
        idx = torch.arange(L, device=device)
        chain = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= local_window
        adj = ((adj + chain.float().unsqueeze(0)) > 0).float()
        adj = ((adj + adj.transpose(1, 2)) > 0).float()

        # keep edges only between real residues, but always self-loop
        resm = mask.unsqueeze(1) * mask.unsqueeze(2)        # (B,L,L)
        adj = adj * resm
        eye = torch.eye(L, device=device).unsqueeze(0)
        adj = ((adj + eye) > 0).float()
        return {"tokens": tokens, "mask": mask, "adj": adj}

    def _lm_logits(self, input_ids, attn) -> torch.Tensor:
        if not hasattr(self, "_lm"):
            from transformers import EsmForMaskedLM

            self._lm = EsmForMaskedLM.from_pretrained(self.hf_id).to(
                self.device, dtype=self.dtype
            )
            self._lm.eval()
        return self._lm(input_ids=input_ids, attention_mask=attn).logits.float()

    # ------------------------------------------------------------------ #
    # Pseudo-perplexity (used by B5 foldability metric)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def pseudo_perplexity(self, sequence: str, max_length: int = 64) -> float:
        """ESM-2 pseudo-perplexity of one sequence.

        Lower = more "natural"/evolutionarily plausible. Computed by masking
        each residue in turn and reading the true-token log-prob. This is an
        established proxy for sequence plausibility and is **independent of the
        Instability Index** the generator optimises.
        """
        enc = self.tokenizer(sequence, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        mask_id = self.tokenizer.mask_token_id

        # Residue positions only (skip CLS/EOS).
        sp = self.tokenizer
        positions = [
            i for i, t in enumerate(ids[0].tolist())
            if t not in {sp.cls_token_id, sp.eos_token_id, sp.pad_token_id}
        ]
        if not positions:
            return float("nan")

        # Batch all masked variants in a single forward (one row per position).
        P = len(positions)
        masked = ids.repeat(P, 1)
        attn_b = attn.repeat(P, 1)
        rows = torch.arange(P, device=self.device)
        pos_t = torch.tensor(positions, device=self.device)
        true_toks = ids[0, pos_t]
        masked[rows, pos_t] = mask_id

        logits = self._lm_logits(masked, attn_b)               # (P, L, vocab)
        logp = F.log_softmax(logits[rows, pos_t], dim=-1)       # (P, vocab)
        nll = -logp[rows, true_toks].mean().item()
        return float(torch.exp(torch.tensor(nll)))


# ---------------------------------------------------------------------- #
# Disk cache for pooled embeddings (used by the classifier oracle, B6)
# ---------------------------------------------------------------------- #
class ESM2EmbeddingCache:
    """Caches mean-pooled ESM-2 embeddings in a SINGLE consolidated file.

    Keyed by sequence hash, held in memory, loaded once and saved periodically
    (one big write instead of thousands of tiny .npy files). The per-file scheme
    was catastrophically slow on a Google Drive FUSE mount — this keeps Drive
    persistence with a single file.
    """

    def __init__(self, embedder: ESM2HF, cache_dir: Union[str, Path]):
        import os
        # ESM_CACHE_DIR overrides every caller (e.g. point to fast local /content
        # on Colab instead of a slow Google Drive mount).
        cache_dir = os.environ.get("ESM_CACHE_DIR", str(cache_dir))
        self.embedder = embedder
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"emb_{embedder.model_name}.pkl"
        self._mem = {}
        if self.cache_file.exists():
            try:
                import pickle
                with open(self.cache_file, "rb") as fh:
                    self._mem = pickle.load(fh)
                logger.info(f"Loaded {len(self._mem)} cached embeddings from {self.cache_file}")
            except Exception as e:
                logger.warning(f"Could not read embedding cache ({e}); starting fresh")
                self._mem = {}

    def _key(self, seq: str) -> str:
        return hashlib.md5(f"{self.embedder.model_name}:{seq}".encode()).hexdigest()

    def _save(self):
        import pickle, os
        tmp = str(self.cache_file) + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(self._mem, fh, protocol=4)
        os.replace(tmp, self.cache_file)  # atomic; one write

    def embed_many(self, sequences: List[str], batch_size: int = 32) -> "np.ndarray":
        import numpy as np

        out = [None] * len(sequences)
        todo, todo_idx = [], []
        for i, s in enumerate(sequences):
            k = self._key(s)
            if k in self._mem:
                out[i] = self._mem[k]
            else:
                todo.append(s)
                todo_idx.append(i)

        dirty = 0
        for b in range(0, len(todo), batch_size):
            chunk = todo[b : b + batch_size]
            pooled = self.embedder.embed(chunk, return_tokens=False)["pooled"].cpu().numpy()
            for j, s in enumerate(chunk):
                gi = todo_idx[b + j]
                out[gi] = pooled[j]
                self._mem[self._key(s)] = pooled[j]
            dirty += len(chunk)
            if (b // batch_size) % 20 == 0:
                logger.info(f"ESM embed {b + len(chunk)}/{len(todo)} new sequences")
            if dirty >= 5000:                # periodic single-file checkpoint
                self._save()
                dirty = 0
        if todo:
            self._save()

        if not out:
            return np.zeros((0, self.embedder.embed_dim), dtype=np.float32)
        return np.stack(out, axis=0)
