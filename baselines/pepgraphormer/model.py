"""
PepGraphormer Baseline (Generative Adaptation).

Adapted from lincubator/PepGraphormer logic.
The original model is a classifier (Predicts AMP probability).
To make it comparable in a generative context, we use a "PepGraphormer-Decoder" architecture:
    1. Sequences are represented as 1D sequence embeddings (BiGRU proxy for ESM2 for speed/simplicity here,
       or direct BiGRU if ESM is too heavy for rapid baseline).
    2. We apply a GNN (GAT/GraphSAGE) over the sequence elements.
    3. The pooled GNN representation is used as conditioning for a GRU Decoder.
    
Architecture:
    Encoder: Embedding → BiGRU → GATConv → Global Mean Pool → z_cond
    Decoder: sample z ~ N(0,1) + z_cond → GRU → generate sequence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class PepGraphormerEncoder(nn.Module):
    """
    A sequence graph encoder.
    Treats the sequence as a linear chain graph.
    Passes sequence through BiGRU, then a GAT layer, then global pooling.
    """
    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        projection_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embedding_dim, hidden_dim // 2, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        # GAT expects node features and edge indices.
        # We will build linear chain edges dynamically in forward.
        self.gat = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU()
        )
        self.pad_idx = pad_idx

    def _build_linear_chain_edges(self, lengths, device):
        """Build edge_index for a batch of linear chains."""
        B = len(lengths)
        edge_indices = []
        batch_idx = []
        
        offset = 0
        for i, l in enumerate(lengths):
            if l <= 1:
                batch_idx.extend([i] * max(1, l))
                offset += max(1, l)
                continue
                
            # Linear chain: 0->1, 1->2, ...
            # Bidirectional: also 1->0, 2->1, ...
            src = torch.arange(l - 1, device=device) + offset
            dst = torch.arange(1, l, device=device) + offset
            
            edges_fwd = torch.stack([src, dst], dim=0)
            edges_bwd = torch.stack([dst, src], dim=0)
            
            edge_indices.append(torch.cat([edges_fwd, edges_bwd], dim=1))
            batch_idx.extend([i] * l)
            offset += l
            
        if not edge_indices:
            return torch.empty((2, 0), dtype=torch.long, device=device), torch.tensor(batch_idx, dtype=torch.long, device=device)
            
        return torch.cat(edge_indices, dim=1), torch.tensor(batch_idx, dtype=torch.long, device=device)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, proj_dim).
        input_ids: (B, L)
        lengths: (B,) lengths of valid tokens
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # 1. Sequence Embedding (analogous to ESM2 features in original paper)
        emb = self.embedding(input_ids)
        out, _ = self.gru(emb)   # (B, L, hidden_dim)
        
        # 2. Flatten for PyTorch Geometric
        # We only want valid tokens
        node_features = []
        for i in range(B):
            l = lengths[i].item()
            node_features.append(out[i, :l, :])
            
        x = torch.cat(node_features, dim=0)  # (total_nodes, hidden_dim)
        
        # 3. Graph Construction (Linear chain)
        edge_index, batch = self._build_linear_chain_edges(lengths.cpu().tolist(), device)
        
        # 4. GAT Convolution
        x = self.gat(x, edge_index)
        x = F.elu(x)
        
        # 5. Readout (Global Pooling)
        graph_repr = global_mean_pool(x, batch)  # (B, hidden_dim)
        
        # Ensure graph_repr has size B (handles empty graphs edge cases)
        if graph_repr.size(0) < B:
            pad = torch.zeros(B - graph_repr.size(0), graph_repr.size(1), device=device)
            graph_repr = torch.cat([graph_repr, pad], dim=0)
            
        return self.projection(graph_repr)


class GRUDecoder(nn.Module):
    """Same decoder as used in ESM2-Decoder."""
    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        enc_emb_dim: int = 128,
        condition_dim: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc_h0 = nn.Linear(latent_dim + enc_emb_dim + condition_dim, hidden_dim * num_layers)
        self.gru = nn.GRU(
            embedding_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def _make_h0(self, z, enc_emb, condition):
        B = z.size(0)
        h0_input = torch.cat([z, enc_emb, condition], dim=-1)
        h0 = torch.tanh(self.fc_h0(h0_input))
        return h0.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

    def forward(self, z, enc_emb, condition, input_ids):
        h = self._make_h0(z, enc_emb, condition)
        emb = self.dropout(self.embedding(input_ids))
        out, _ = self.gru(emb, h)
        return self.output_proj(self.dropout(out))

    def generate(self, z, enc_emb, condition, sos_idx, eos_idx, max_len=52, temperature=1.0, top_p=0.9):
        B = z.size(0)
        device = z.device
        h = self._make_h0(z, enc_emb, condition)
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


class PepGraphormerDecoderModel(nn.Module):
    """
    Full generative adaptation of PepGraphormer.
    Encoder builds a graph from sequence, applies GAT, global pools.
    Decoder uses pooled graph representation + noise + physiochemical condition to generate.
    """
    def __init__(
        self,
        vocab_size: int = 24,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        enc_projection_dim: int = 128,
        condition_dim: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_projection_dim = enc_projection_dim
        self.pad_idx = pad_idx

        self.encoder = PepGraphormerEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            projection_dim=enc_projection_dim,
            num_heads=4,
            dropout=dropout,
            pad_idx=pad_idx,
        )
        
        self.decoder = GRUDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            enc_emb_dim=enc_projection_dim,
            condition_dim=condition_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, input_ids, condition, target_ids):
        # Calculate lengths of target_ids for graph construction
        mask = (target_ids != self.pad_idx).long()
        lengths = mask.sum(dim=1)
        
        # Encode target graph
        graph_emb = self.encoder(target_ids, lengths)
        
        # Decode
        z = torch.randn(input_ids.size(0), self.latent_dim, device=input_ids.device)
        logits = self.decoder(z, graph_emb, condition, input_ids)
        return {'logits': logits, 'graph_emb': graph_emb}

    def generate(self, num_samples, condition, sos_idx, eos_idx, max_len=52, temperature=1.0, top_p=0.9, device='cuda'):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            graph_emb = torch.zeros(num_samples, self.enc_projection_dim, device=device)
            return self.decoder.generate(z, graph_emb, condition, sos_idx, eos_idx, max_len, temperature, top_p)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
