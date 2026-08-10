"""
HydrAMP Baseline: Conditional Variational Autoencoder with AMP/MIC classifiers.

Architecture (faithful to Miszta et al. 2022):
    Encoder: Embedding → GRU → (μ, logσ²)  [latent_dim=128]
    Decoder: z + condition → GRU → logits
    Classifier 1: z → MLP → P(AMP)         [binary, BCE loss]
    Classifier 2: z → MLP → P(low_MIC)     [binary, proxied by therapeutic_score > 0]

Loss:
    L = ELBO + λ_amp * BCE(P_AMP) + λ_mic * BCE(P_MIC)
    ELBO = CE_reconstruction - β * KL(q(z|x) || p(z))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class PeptideEncoder(nn.Module):
    """
    Bidirectional GRU encoder → produces (μ, logσ²) for latent space.
    """
    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # Project bidirectional hidden state to latent parameters
        gru_out_dim = hidden_dim * 2  # bidirectional
        self.fc_mu = nn.Linear(gru_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(gru_out_dim, latent_dim)

    def forward(self, input_ids: torch.Tensor) -> tuple:
        """
        Args:
            input_ids: (B, L)
        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        emb = self.dropout(self.embedding(input_ids))   # (B, L, emb)
        _, hidden = self.gru(emb)                        # (num_layers*2, B, H)
        # Take last layer forward + backward hidden states
        h_fwd = hidden[-2]  # forward last layer
        h_bwd = hidden[-1]  # backward last layer
        h = torch.cat([h_fwd, h_bwd], dim=-1)           # (B, 2H)
        h = self.dropout(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class PeptideDecoder(nn.Module):
    """
    Autoregressive GRU decoder conditioned on z and condition vector.
    """
    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        condition_dim: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Project z + condition to initial hidden state
        self.fc_h0 = nn.Linear(latent_dim + condition_dim, hidden_dim * num_layers)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher-forced decoding.

        Args:
            z: (B, latent_dim)
            condition: (B, condition_dim)
            input_ids: (B, L) — shifted right input (SOS + seq)

        Returns:
            logits: (B, L, vocab_size)
        """
        B = z.size(0)
        # Build initial hidden state from z + condition
        h0_input = torch.cat([z, condition], dim=-1)             # (B, latent+cond)
        h0 = torch.tanh(self.fc_h0(h0_input))                    # (B, H*layers)
        h0 = h0.view(B, self.num_layers, self.hidden_dim)        # (B, layers, H)
        h0 = h0.permute(1, 0, 2).contiguous()                    # (layers, B, H)

        emb = self.dropout(self.embedding(input_ids))             # (B, L, emb)
        out, _ = self.gru(emb, h0)                               # (B, L, H)
        out = self.dropout(out)
        logits = self.output_proj(out)                            # (B, L, vocab)
        return logits

    def generate(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 52,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Returns:
            token_ids: (B, max_len)
        """
        B = z.size(0)
        device = z.device

        h0_input = torch.cat([z, condition], dim=-1)
        h = torch.tanh(self.fc_h0(h0_input))
        h = h.view(B, self.num_layers, self.hidden_dim)
        h = h.permute(1, 0, 2).contiguous()

        token = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)
        generated = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            emb = self.embedding(token)                           # (B, 1, emb)
            out, h = self.gru(emb, h)                            # (B, 1, H)
            logits = self.output_proj(out.squeeze(1))             # (B, vocab)

            # Temperature + nucleus sampling
            logits = logits / max(temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)

            # Nucleus (top-p) sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(sorted_probs, 1)
            token = sorted_idx.gather(-1, sampled)                # (B, 1)

            generated.append(token)
            finished |= (token.squeeze(-1) == eos_idx)
            if finished.all():
                break

        return torch.cat(generated, dim=1)                        # (B, T)


class AMPClassifier(nn.Module):
    """
    Small MLP classifier for AMP probability or MIC (low MIC proxy).
    Input: latent z. Output: binary logit.
    """
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns logit (B,)"""
        return self.net(z).squeeze(-1)


class HydrAMPModel(nn.Module):
    """
    Full HydrAMP model:
        Encoder → (μ, logσ²) → z (reparameterized)
        Decoder → reconstructed sequence logits
        AMP Classifier → P(AMP)
        MIC Classifier → P(low_MIC)  [proxied by therapeutic_score threshold]
    """
    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        condition_dim: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        self.encoder = PeptideEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=pad_idx,
        )
        self.decoder = PeptideDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.amp_classifier = AMPClassifier(latent_dim, hidden_dim=64, dropout=dropout)
        self.mic_classifier = AMPClassifier(latent_dim, hidden_dim=64, dropout=dropout)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self,
        input_ids: torch.Tensor,
        condition: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> dict:
        """
        Forward pass for training.

        Args:
            input_ids: (B, L) — SOS + sequence
            condition: (B, 8) — normalized physicochemical features
            target_ids: (B, L) — sequence + EOS (teacher-forcing targets)

        Returns:
            dict with logits, mu, logvar, amp_logit, mic_logit
        """
        mu, logvar = self.encoder(target_ids)  # encode full sequence incl. EOS
        z = self.reparameterize(mu, logvar)

        logits = self.decoder(z, condition, input_ids)          # (B, L, vocab)
        amp_logit = self.amp_classifier(z)                       # (B,)
        mic_logit = self.mic_classifier(z)                       # (B,)

        return {
            'logits': logits,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'amp_logit': amp_logit,
            'mic_logit': mic_logit,
        }

    def generate(
        self,
        num_samples: int,
        condition: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 52,
        temperature: float = 1.0,
        top_p: float = 0.9,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        Generate sequences by sampling z ~ N(0, I).

        Args:
            num_samples: Number of sequences to generate
            condition: (num_samples, condition_dim) condition vectors
            ...
        Returns:
            token_ids: (num_samples, max_len)
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            return self.decoder.generate(
                z, condition, sos_idx, eos_idx, max_len, temperature, top_p
            )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
