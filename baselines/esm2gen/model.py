"""
ESM2-Decoder Baseline: ESM2-8M (frozen encoder) + GRU Decoder.

Approach:
    1. ESM2-8M encodes input sequences → per-token embeddings (320-dim)
    2. Mean-pool → sequence embedding (320-dim) → project to 128-dim
    3. Sample z ~ N(0,I)^128, combine with sequence embedding → h0
    4. GRU Decoder generates new sequences autoregressively

This is an encoder-conditioned generative model:
    - The encoder captures real sequence "style" via ESM2 representations
    - The decoder learns to generate valid peptides conditioned on that style + noise
    - At inference: condition from z only (unconditional), or from test sequences

Training objective: Reconstruction (CrossEntropy, teacher-forced)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False


class ESM2Encoder(nn.Module):
    """
    Frozen ESM2-8M encoder that produces sequence embeddings.
    Falls back to a lightweight BiGRU encoder if ESM2 is unavailable.
    """

    def __init__(
        self,
        projection_dim: int = 128,
        freeze: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_esm = ESM_AVAILABLE
        self.projection_dim = projection_dim

        if self.use_esm:
            print("[ESM2-Decoder] Loading ESM2-8M (esm2_t6_8M_UR50D)...")
            self.esm_model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.batch_converter = self.alphabet.get_batch_converter()
            esm_embed_dim = 320  # ESM2-8M output dim

            if freeze:
                for p in self.esm_model.parameters():
                    p.requires_grad = False
                print("[ESM2-Decoder] ESM2-8M frozen")

            self.projection = nn.Sequential(
                nn.Linear(esm_embed_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            # Fallback: lightweight BiGRU encoder
            print("[ESM2-Decoder] ESM2 not available — using BiGRU encoder fallback.")
            self.fallback_embedding = nn.Embedding(24, 128)
            self.fallback_gru = nn.GRU(128, 192, num_layers=2, batch_first=True,
                                        bidirectional=True, dropout=0.1)
            self.projection = nn.Sequential(
                nn.Linear(384, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
            )

    def forward_esm(self, sequences: list, device) -> torch.Tensor:
        """Run ESM2 on a list of string sequences. Returns (B, projection_dim)."""
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = self.esm_model(tokens, repr_layers=[6], return_contacts=False)
        token_repr = results['representations'][6]          # (B, L+2, 320)
        # Mean pool (excluding BOS/EOS)
        seq_embedding = token_repr[:, 1:-1, :].mean(dim=1)  # (B, 320)
        return self.projection(seq_embedding)               # (B, proj_dim)

    def forward_fallback(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Fallback BiGRU encoder. Returns (B, proj_dim)."""
        emb = self.fallback_embedding(input_ids)
        _, h = self.fallback_gru(emb)
        h = torch.cat([h[-2], h[-1]], dim=-1)               # (B, 384)
        return self.projection(h)

    def forward(self, input_ids: torch.Tensor, sequences: list = None) -> torch.Tensor:
        """
        Returns (B, projection_dim) sequence embeddings.

        Args:
            input_ids: (B, L) token tensor (used for fallback or when sequences is None)
            sequences: List of raw string sequences (for ESM2 path)
        """
        if self.use_esm and sequences is not None:
            return self.forward_esm(sequences, input_ids.device)
        else:
            return self.forward_fallback(input_ids)


class GRUDecoder(nn.Module):
    """Standard autoregressive GRU decoder, identical to PeptideDecoder in HydrAMP."""

    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        enc_emb_dim: int = 128,   # ESM2 projection dim
        condition_dim: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # z + esm_embedding + condition → hidden state
        self.fc_h0 = nn.Linear(latent_dim + enc_emb_dim + condition_dim, hidden_dim * num_layers)
        self.gru = nn.GRU(
            embedding_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def _make_h0(self, z, esm_emb, condition):
        B = z.size(0)
        h0_input = torch.cat([z, esm_emb, condition], dim=-1)
        h0 = torch.tanh(self.fc_h0(h0_input))
        h0 = h0.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        return h0

    def forward(self, z, esm_emb, condition, input_ids):
        h = self._make_h0(z, esm_emb, condition)
        emb = self.dropout(self.embedding(input_ids))
        out, _ = self.gru(emb, h)
        return self.output_proj(self.dropout(out))    # (B, L, vocab)

    def generate(self, z, esm_emb, condition, sos_idx, eos_idx, max_len=52,
                 temperature=1.0, top_p=0.9):
        B = z.size(0)
        device = z.device
        h = self._make_h0(z, esm_emb, condition)
        token = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)
        generated = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            emb = self.embedding(token)
            out, h = self.gru(emb, h)
            logits = self.output_proj(out.squeeze(1)) / max(temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(1e-8)
            sampled = torch.multinomial(sorted_probs, 1)
            token = sorted_idx.gather(-1, sampled)
            generated.append(token)
            finished |= (token.squeeze(-1) == eos_idx)
            if finished.all():
                break

        return torch.cat(generated, dim=1)


class ESM2DecoderModel(nn.Module):
    """
    Full ESM2-Decoder model.

    Training: Encode real sequence with ESM2 → condition decoder → reconstruct.
    Inference: Sample z ~ N(0,I), use zero/mean ESM2 embedding or random sample from val set.
    """

    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        esm_projection_dim: int = 128,
        condition_dim: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.esm_projection_dim = esm_projection_dim
        self.condition_dim = condition_dim

        self.esm_encoder = ESM2Encoder(
            projection_dim=esm_projection_dim,
            freeze=True,
            dropout=dropout,
        )
        self.decoder = GRUDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            enc_emb_dim=esm_projection_dim,
            condition_dim=condition_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, input_ids, condition, target_ids, sequences=None):
        """
        Args:
            input_ids: (B, L) SOS + seq
            condition: (B, 8)
            target_ids: (B, L) seq + EOS
            sequences: List[str] raw sequences (optional, for ESM2 path)
        """
        # Encode the real sequence (using ESM2 or fallback)
        esm_emb = self.esm_encoder(target_ids, sequences)    # (B, proj_dim)
        # Sample z
        z = torch.randn(input_ids.size(0), self.latent_dim, device=input_ids.device)
        logits = self.decoder(z, esm_emb, condition, input_ids)
        return {'logits': logits, 'z': z, 'esm_emb': esm_emb}

    def generate(self, num_samples, condition, sos_idx, eos_idx,
                 max_len=52, temperature=1.0, top_p=0.9, device='cuda',
                 ref_esm_emb=None):
        """
        Generate sequences. If ref_esm_emb is provided, use it; otherwise use zeros.
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            if ref_esm_emb is not None:
                esm_emb = ref_esm_emb
            else:
                # Use zero embedding at inference (unconditional)
                esm_emb = torch.zeros(num_samples, self.esm_projection_dim, device=device)
            return self.decoder.generate(
                z, esm_emb, condition, sos_idx, eos_idx, max_len, temperature, top_p
            )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self):
        return sum(p.numel() for p in self.parameters())
