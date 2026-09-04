"""
Multimodal-fusion Transformer generator (paper §3.4–§3.6).

This is the architecture the paper *describes* but the original code did not
implement: a compact Transformer decoder whose cross-attention memory is the
**fused** representation of a semantic stream (ESM-2, H_seq) and a structural
stream (GATv2, H_graph), combined by a Multi-Head Cross-Attention layer

        H_fusion = softmax( (H_seq W_Q)(H_graph W_K)^T / sqrt(d) )(H_graph W_V)

exactly as in Eq. (3). The configured physicochemical condition C and the noise
latent z are injected through the same memory builder so that de-novo
generation (no input sequence) and teacher-forced MLE share one backbone.

Design notes
------------
* **De-novo path (default in the GAN loop):** the memory is built from (z, C)
  only — ESM-2 is *not* run per step, so g_steps=10 stays fast.
* **Refinement path (MLE warm-up):** pass ``esm_tokens`` (cached frozen ESM-2
  token embeddings of the real sequence). They additively refine H_seq via a
  Perceiver-style resampler, so the fusion genuinely consumes ESM-2 features
  during teacher forcing without coupling the heavy backbone into autograd.
* **Graph stream:** a dense GATv2 layer over a windowed adjacency at de novo
  inference. During warm-up, callers may provide a residue adjacency (the
  current local pipeline uses thresholded ESM-2 attention contacts). The model
  itself makes no claim about coordinate-derived distances.

Interface is drop-in compatible with ``GANTrainer`` / ``PeptideSampler``:
``forward(z, target=None, condition=None, temperature=1.0, esm_tokens=None)``
returns ``{'logits': (B, L, V)}`` and ``generate_with_logits(z, condition)``
returns ``(tokens, logits)``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import PositionalEncoding


class DenseGATv2Layer(nn.Module):
    """GATv2 (Brody et al., 2022) over a dense adjacency mask.

    Dynamic attention: the scoring linear ``a`` is applied *after* the
    LeakyReLU of ``W[x_i || x_j]`` (this is the GATv2 fix over the static GAT).
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(heads, out_dim))
        self.leaky = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_dim)   adj: (B, N, N) with >0 where connected
        B, N, _ = x.shape
        Wh = self.W(x).view(B, N, self.heads, self.out_dim)            # (B,N,H,d)
        # pairwise sum h_i + h_j (GATv2 uses concat→linear, equivalent to
        # summing two linear maps of a shared W applied to i and j).
        hi = Wh.unsqueeze(2)                                            # (B,N,1,H,d)
        hj = Wh.unsqueeze(1)                                            # (B,1,N,H,d)
        e = self.leaky(hi + hj)                                         # (B,N,N,H,d)
        e = (e * self.a).sum(-1)                                        # (B,N,N,H)
        e = e.permute(0, 3, 1, 2)                                       # (B,H,N,N)

        mask = (adj.unsqueeze(1) > 0)                                   # (B,1,N,N)
        e = e.masked_fill(~mask, float("-inf"))
        att = torch.softmax(e, dim=-1)
        att = torch.nan_to_num(att, nan=0.0)
        att = self.dropout(att)

        Wh_h = Wh.permute(0, 2, 1, 3)                                   # (B,H,N,d)
        out = torch.matmul(att, Wh_h)                                   # (B,H,N,d)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.heads * self.out_dim)
        return F.elu(out)


class ConcatFusion(nn.Module):
    """Ablation baseline: simple concatenation [H_seq || H_graph] projected to d_model."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_seq, h_graph, key_padding_mask=None):
        # h_seq: (B, M, d), h_graph: (B, M, d) or (B, L, d)
        if h_graph.size(1) != h_seq.size(1):
            h_graph = h_graph[:, :h_seq.size(1), :]
            if h_graph.size(1) < h_seq.size(1):
                pad = torch.zeros(h_seq.size(0), h_seq.size(1) - h_graph.size(1), h_seq.size(2), device=h_seq.device)
                h_graph = torch.cat([h_graph, pad], dim=1)
        cat = torch.cat([h_seq, h_graph], dim=-1)
        return self.norm(h_seq + self.dropout(self.proj(cat)))


class CrossAttentionFusion(nn.Module):
    """Eq. (3): H_seq is Query, H_graph is Key/Value."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_seq, h_graph, key_padding_mask=None):
        fused, _ = self.attn(query=h_seq, key=h_graph, value=h_graph,
                             key_padding_mask=key_padding_mask, need_weights=False)
        fused = torch.nan_to_num(fused, nan=0.0)
        return self.norm(h_seq + fused)


class MultimodalFusionGenerator(nn.Module):
    """Compact Transformer generator with ESM-2 / GATv2 cross-attention fusion."""

    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 128,      # d_model (paper: 128)
        hidden_dim: int = 512,         # FFN hidden (paper: 512)
        latent_dim: int = 128,
        max_length: int = 50,
        num_layers: int = 3,           # paper: 3 decoder layers
        num_heads: int = 4,            # paper: 4 heads
        dropout: float = 0.2,
        condition_dim: Optional[int] = 6,
        mem_tokens: int = 16,          # number of fused memory tokens
        gat_heads: int = 4,
        gat_window: int = 3,
        use_gat: bool = True,          # set to False for w/o GATv2 ablation
        fusion_type: str = "cross_attention",  # 'cross_attention' | 'concat' | 'none'
        esm_dim: Optional[int] = None,  # set to enable the ESM refinement path
        pad_idx: int = 0,
        sos_idx: int = 1,
        eos_idx: int = 2,
    ):
        super().__init__()
        d = embedding_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.condition_dim = condition_dim
        self.mem_tokens = mem_tokens
        self.use_gat = use_gat
        self.fusion_type = fusion_type
        self.gat_heads = gat_heads
        self.gat_window = gat_window
        self.esm_dim = esm_dim
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        # kept for checkpoint-save compatibility with GANTrainer.save()
        self.bidirectional = False
        self.use_attention = True

        cond_dim = condition_dim or 0

        # ---- (z, C) -> M semantic memory tokens (H_seq prior) ----
        self.mem_queries = nn.Parameter(torch.randn(mem_tokens, d) * 0.02)
        self.zc_proj = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )
        # FiLM modulation of the query bank by the (z,C) summary.
        self.film = nn.Linear(d, 2 * d)

        # ---- optional ESM refinement (Perceiver resampler) ----
        if esm_dim is not None:
            self.esm_in = nn.Linear(esm_dim, d)
            self.esm_resampler = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
            self.esm_norm = nn.LayerNorm(d)

        # ---- structural stream (GATv2) ----
        if use_gat:
            self.gat = DenseGATv2Layer(d, d // gat_heads, heads=gat_heads, dropout=dropout)
        else:
            self.gat = None
        # ---- fusion layer (Eq. 3 or Ablation) ----
        if fusion_type == "concat":
            self.fusion = ConcatFusion(d, dropout)
        elif fusion_type == "none":
            self.fusion = None
        else:
            self.fusion = CrossAttentionFusion(d, num_heads, dropout)

        # ---- compact Transformer decoder ----
        self.embedding = nn.Embedding(vocab_size, d, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d, max_length + 2, dropout)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.output_projection = nn.Linear(d, vocab_size)

    # ------------------------------------------------------------------ #
    def _build_adjacency(self, n: int, device: torch.device) -> torch.Tensor:
        """Windowed adjacency over the M memory nodes (de novo phases).

        Warm-up callers may replace this with an explicitly documented residue
        adjacency without changing the generator.
        """
        idx = torch.arange(n, device=device)
        adj = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= self.gat_window
        return adj.float().unsqueeze(0)  # (1, N, N) — broadcasts over batch

    def _build_memory(
        self,
        z: torch.Tensor,
        condition: Optional[torch.Tensor],
        esm_tokens: Optional[torch.Tensor] = None,
        esm_mask: Optional[torch.Tensor] = None,
        contact_adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = z.size(0)
        device = z.device
        if self.condition_dim and condition is None:
            condition = torch.zeros(B, self.condition_dim, device=device)
        zc = z if condition is None else torch.cat([z, condition], dim=-1)
        summary = self.zc_proj(zc)                                   # (B, d)

        # H_seq prior: query bank FiLM-modulated by (z,C)
        gamma, beta = self.film(summary).chunk(2, dim=-1)            # (B, d) each
        h_seq = self.mem_queries.unsqueeze(0).expand(B, -1, -1)      # (B, M, d)
        h_seq = h_seq * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        # optional ESM-2 refinement of the semantic stream (MLE / refinement)
        res = None
        if esm_tokens is not None and self.esm_dim is not None:
            res = self.esm_in(esm_tokens.to(h_seq.dtype))            # (B, L, d)
            key_padding = ~esm_mask.bool() if esm_mask is not None else None
            refined, _ = self.esm_resampler(
                query=h_seq, key=res, value=res,
                key_padding_mask=key_padding, need_weights=False,
            )
            h_seq = self.esm_norm(h_seq + torch.nan_to_num(refined, nan=0.0))

        # structural stream H_graph
        if self.gat is None:
            h_graph = h_seq
        elif res is not None and contact_adj is not None:
            # Warm-up: GATv2 over the caller-supplied residue adjacency.
            h_graph = self.gat(res, contact_adj)                     # (B, L, d) residues
        else:
            # de-novo / no-contacts fallback: GATv2 over the memory tokens.
            adj = self._build_adjacency(self.mem_tokens, device)
            h_graph = self.gat(h_seq, adj)                           # (B, M, d)

        # multimodal fusion
        if self.fusion is not None:
            kv_pad = ~esm_mask.bool() if (esm_mask is not None and res is not None and contact_adj is not None) else None
            memory = self.fusion(h_seq, h_graph, key_padding_mask=kv_pad)
        else:
            memory = h_seq
        return memory

    @staticmethod
    def _causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        z: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        esm_tokens: Optional[torch.Tensor] = None,
        esm_mask: Optional[torch.Tensor] = None,
        contact_adj: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        memory = self._build_memory(z, condition, esm_tokens, esm_mask, contact_adj)

        if target is not None:
            # teacher forcing
            seq_len = target.size(1)
            tgt = self.pos_encoding(self.embedding(target))
            mask = self._causal_mask(seq_len, z.device)
            out = self.decoder(tgt, memory, tgt_mask=mask)
            return {"logits": self.output_projection(out)}

        tokens, logits = self._autoregressive(memory, temperature)
        return {"sequences": tokens, "logits": logits}

    def _mask_generation_logits(
        self,
        logits: torch.Tensor,
        step: int,
        min_length: int = 5,
        finished: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mask non-residue tokens and force padding after the first EOS."""
        logits = logits.clone()
        invalid = {self.pad_idx, self.sos_idx, 3}  # vocabulary index 3 is <UNK>
        for idx in invalid:
            if 0 <= idx < logits.size(-1):
                logits[..., idx] = float("-inf")
        if step < min_length:
            logits[..., self.eos_idx] = float("-inf")
        if finished is not None and finished.any():
            logits[finished] = float("-inf")
            logits[finished, self.pad_idx] = 0.0
        return logits

    def _autoregressive(
        self,
        memory: torch.Tensor,
        temperature: float,
        min_length: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = memory.size(0)
        device = memory.device
        cur = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        all_logits = []
        for step in range(self.max_length):
            tgt = self.pos_encoding(self.embedding(cur))
            mask = self._causal_mask(cur.size(1), device)
            out = self.decoder(tgt, memory, tgt_mask=mask)
            logits = self.output_projection(out[:, -1:, :])          # (B,1,V)
            logits = self._mask_generation_logits(
                logits.squeeze(1), step, min_length, finished
            ).unsqueeze(1)
            all_logits.append(logits)
            probs = F.softmax(logits.squeeze(1) / max(temperature, 1e-6), dim=-1)
            nxt = torch.multinomial(probs, 1)
            # Avoid in-place mutation: ``finished`` was used as an indexing
            # mask in the graph for prior steps and autograd tracks its version.
            finished = finished | (nxt.squeeze(1) == self.eos_idx)
            cur = torch.cat([cur, nxt], dim=1)
        return cur, torch.cat(all_logits, dim=1)

    def generate_with_logits(
        self,
        z: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        esm_tokens: Optional[torch.Tensor] = None,
        esm_mask: Optional[torch.Tensor] = None,
        contact_adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast path used by GANTrainer: returns (tokens, logits)."""
        memory = self._build_memory(z, condition, esm_tokens, esm_mask, contact_adj)
        return self._autoregressive(memory, temperature=1.0, min_length=5)

    def rl_rollout(
        self,
        z: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        greedy: bool = False,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """SCST rollout. Returns (tokens (B,T), summed log-prob of chosen tokens).

        Log-probs after the first <EOS> are masked to zero so the policy
        gradient only credits the emitted prefix. Used by ``SCSTTrainer``:
        call with ``greedy=False`` for the sampled sequence (keep grad) and
        ``greedy=True`` (under no_grad) for the self-critical baseline.
        """
        memory = self._build_memory(z, condition)
        device = memory.device
        B = memory.size(0)
        ml = max_length or self.max_length
        cur = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        toks, logps, ents = [], [], []
        alive_steps = []
        for step in range(ml):
            tgt = self.pos_encoding(self.embedding(cur))
            mask = self._causal_mask(cur.size(1), device)
            out = self.decoder(tgt, memory, tgt_mask=mask)
            logits = self.output_projection(out[:, -1, :])      # (B, V)
            logits = self._mask_generation_logits(logits, step, 5, finished)
            logp_all = F.log_softmax(logits, dim=-1)
            if greedy:
                nxt = logits.argmax(dim=-1)
            else:
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
            lp = logp_all.gather(1, nxt.unsqueeze(1)).squeeze(1)  # (B,)
            alive = (~finished).float()
            lp = lp * alive
            # per-step policy entropy (only count positions before <EOS>)
            ent = -(logp_all.exp() * logp_all).sum(dim=-1) * alive
            toks.append(nxt)
            logps.append(lp)
            ents.append(ent)
            alive_steps.append(alive)
            finished = finished | (nxt == self.eos_idx)
            cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
            if finished.all():
                break
        tokens = torch.stack(toks, dim=1)                        # (B, T)
        logp_sum = torch.stack(logps, dim=1).sum(dim=1)          # (B,)
        # mean entropy per sequence over emitted positions
        ent_stack = torch.stack(ents, dim=1)                     # (B, T)
        lengths = torch.stack(alive_steps, dim=1).sum(dim=1).clamp(min=1)
        entropy = ent_stack.sum(dim=1) / lengths                 # (B,)
        return tokens, logp_sum, entropy

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        z: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        min_length: int = 5,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        device = device or next(self.parameters()).device
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, device=device)
        memory = self._build_memory(z, condition)
        ml = max_length or self.max_length
        cur = torch.full((z.size(0), 1), self.sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(z.size(0), dtype=torch.bool, device=device)
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        for step in range(ml):
            if finished.all():
                break
            tgt = self.pos_encoding(self.embedding(cur))
            mask = self._causal_mask(cur.size(1), device)
            out = self.decoder(tgt, memory, tgt_mask=mask)
            logits = self.output_projection(out[:, -1, :])
            logits = self._mask_generation_logits(logits, step, min_length, finished)
            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
                finished = finished | (nxt.squeeze(-1) == self.eos_idx)
                cur = torch.cat([cur, nxt], dim=1)
                continue
            logits = logits / temperature
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                logits[logits < kth] = float("-inf")
            if top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                rm = cum > top_p
                rm[..., 1:] = rm[..., :-1].clone()
                rm[..., 0] = 0
                logits[rm.scatter(1, si, rm)] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            finished = finished | (nxt.squeeze(-1) == self.eos_idx)
            cur = torch.cat([cur, nxt], dim=1)
        return cur
