"""
M3-CAD Simplified Baseline: Multimodal Conditional VAE with Regression + Multilabel Classifier.

Adapted from: Qian et al. 2023, "M3-CAD: Multimodal, Multitask, Multilabel,
Conditionally-Controlled Antimicrobial Peptide Discovery."

Simplification: No 3D voxel branch (dataset lacks structural data).
Instead: dual-encoder using sequence features + 8 physicochemical features.

Architecture:
    Sequence Encoder: Embedding → GRU → seq_feat (256-dim)
    Feature Encoder: MLP(8) → cond_feat (32-dim)
    Combined: concat → Linear → (μ, logσ²)

    Decoder: z + cond_feat → GRU → logits
    Regression: z → MLP → stability proxy (instability_index)
    Classifier: z + cond_feat → MLP → multilabel (AMP, low_hemolytic, stable)

Loss:
    L = ELBO + λ_reg * MSE(predicted_stability, instability_index)
             + λ_cls * BCE(classifier, labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceEncoder(nn.Module):
    """Sequence branch encoder."""
    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embedding_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * 2  # bidirectional

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns (B, hidden_dim*2)"""
        emb = self.dropout(self.embedding(input_ids))
        _, h = self.gru(emb)
        h = torch.cat([h[-2], h[-1]], dim=-1)  # last layer fwd + bwd
        return self.dropout(h)


class FeatureEncoder(nn.Module):
    """Physicochemical feature encoder (the 'functional attributes' branch in M3-CAD)."""
    def __init__(self, feature_dim: int = 8, out_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class M3CADDecoder(nn.Module):
    """GRU decoder conditioned on z + feature encoding."""
    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        cond_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc_h0 = nn.Linear(latent_dim + cond_dim, hidden_dim * num_layers)
        self.gru = nn.GRU(
            embedding_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, cond_feat, input_ids):
        """Teacher-forced decoding. Returns logits (B, L, vocab)."""
        B = z.size(0)
        h0 = torch.tanh(self.fc_h0(torch.cat([z, cond_feat], dim=-1)))
        h0 = h0.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        emb = self.dropout(self.embedding(input_ids))
        out, _ = self.gru(emb, h0)
        return self.output_proj(self.dropout(out))

    def generate(self, z, cond_feat, sos_idx, eos_idx, max_len=52, temperature=1.0, top_p=0.9):
        """Autoregressive generation. Returns token ids (B, T)."""
        B = z.size(0)
        device = z.device
        h0 = torch.tanh(self.fc_h0(torch.cat([z, cond_feat], dim=-1)))
        h = h0.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

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
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            sampled = torch.multinomial(sorted_probs, 1)
            token = sorted_idx.gather(-1, sampled)
            generated.append(token)
            finished |= (token.squeeze(-1) == eos_idx)
            if finished.all():
                break

        return torch.cat(generated, dim=1)


class RegressionModule(nn.Module):
    """Predicts instability index proxy from z."""
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)   # (B,)


class MultilabelClassifier(nn.Module):
    """
    Multilabel classification: AMP | low_hemolytic | stable.
    3 binary outputs.
    """
    def __init__(self, latent_dim: int = 128, cond_dim: int = 32, num_labels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels),
        )

    def forward(self, z: torch.Tensor, cond_feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, cond_feat], dim=-1))   # (B, num_labels)


class M3CADModel(nn.Module):
    """
    M3-CAD Simplified Model (no 3D branch).

    Two encoders:
        - sequence encoder (GRU)
        - feature encoder (MLP on 8 physicochemical features)
    Combined → latent z.

    Three outputs:
        - Decoder: sequence generation
        - Regression: stability score prediction
        - Classifier: AMP / hemolytic / stable multilabel
    """

    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        condition_dim: int = 8,
        cond_enc_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        self.seq_encoder = SequenceEncoder(
            vocab_size, embedding_dim, hidden_dim, num_layers, dropout, pad_idx
        )
        self.feat_encoder = FeatureEncoder(condition_dim, cond_enc_dim, dropout)

        # Combined → μ, logσ²
        combined_dim = self.seq_encoder.out_dim + cond_enc_dim
        self.fc_mu = nn.Linear(combined_dim, latent_dim)
        self.fc_logvar = nn.Linear(combined_dim, latent_dim)

        self.decoder = M3CADDecoder(
            vocab_size, embedding_dim, hidden_dim, latent_dim, cond_enc_dim, num_layers, dropout
        )
        self.regression = RegressionModule(latent_dim, hidden_dim=64)
        self.classifier = MultilabelClassifier(latent_dim, cond_enc_dim, num_labels=3)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, input_ids, condition, target_ids):
        """
        Args:
            input_ids: (B, L) SOS+seq
            condition: (B, 8) normalized physicochemical features
            target_ids: (B, L) seq+EOS
        """
        seq_feat = self.seq_encoder(target_ids)         # (B, 512)
        cond_feat = self.feat_encoder(condition)        # (B, 32)
        combined = torch.cat([seq_feat, cond_feat], dim=-1)

        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        z = self.reparameterize(mu, logvar)

        logits = self.decoder(z, cond_feat, input_ids)
        stability_pred = self.regression(z)
        cls_logits = self.classifier(z, cond_feat)

        return {
            'logits': logits,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'stability_pred': stability_pred,
            'cls_logits': cls_logits,
        }

    def generate(self, num_samples, condition, sos_idx, eos_idx, max_len=52,
                 temperature=1.0, top_p=0.9, device='cuda'):
        self.eval()
        with torch.no_grad():
            cond_feat = self.feat_encoder(condition)
            z = torch.randn(num_samples, self.latent_dim, device=device)
            return self.decoder.generate(z, cond_feat, sos_idx, eos_idx, max_len, temperature, top_p)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
