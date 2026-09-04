"""
Peptide sequence sampling and generation.

Supports multiple sampling strategies:
    - Greedy: argmax at each position
    - Temperature: softmax with temperature scaling
    - Top-k: Sample from top k tokens
    - Nucleus (top-p): Sample from smallest set with cumulative prob >= p
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
import logging

from ..data.vocabulary import VOCAB

logger = logging.getLogger(__name__)


class PeptideSampler:
    """
    High-level interface for peptide sequence generation.
    
    Args:
        generator: Trained generator model
        device: PyTorch device
        vocab: Vocabulary (default: VOCAB)
        
    Example:
        >>> sampler = PeptideSampler.from_checkpoint('checkpoints/best_model.pt')
        >>> sequences = sampler.sample(n=100, temperature=0.8)
        >>> sampler.save_fasta(sequences, 'generated.fasta')
    """
    
    def __init__(
        self,
        generator: torch.nn.Module,
        device: Optional[torch.device] = None,
        vocab=None,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.G = generator.to(self.device)
        self.G.eval()
        self.vocab = vocab or VOCAB
        self.checkpoint_path = None
        self.checkpoint_metadata = {}
        
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        generator_class=None,
        generator_kwargs: Dict = None,
        device=None,
    ) -> 'PeptideSampler':
        """
        Create sampler from checkpoint.
        
        Args:
            checkpoint_path: Path to .pt checkpoint
            generator_class: Generator class (auto-detect if None)
            generator_kwargs: Generator constructor args
            device: Device
            
        Returns:
            PeptideSampler instance
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        model_cfg = ckpt.get('model_config') or ckpt.get('config', {}).get('model', ckpt.get('config', {}))

        # Auto-detect generator class: prefer the saved class name, then fall
        # back to state_dict key sniffing, then GRUGenerator (legacy default).
        if generator_class is None:
            cls_name = ckpt.get('generator_class')
            keys = ckpt.get('generator', {}).keys()
            if cls_name == 'MultimodalFusionGenerator' or any(
                k.startswith(('mem_queries', 'fusion.', 'decoder.')) for k in keys
            ):
                from ..models import MultimodalFusionGenerator
                generator_class = MultimodalFusionGenerator
            else:
                from ..models import GRUGenerator
                generator_class = GRUGenerator

        # Build kwargs from checkpoint config
        if generator_kwargs is None:
            if generator_class.__name__ == 'MultimodalFusionGenerator':
                generator_kwargs = {
                    'vocab_size': model_cfg.get('vocab_size', VOCAB.vocab_size),
                    'embedding_dim': model_cfg.get('embedding_dim', 128),
                    'hidden_dim': model_cfg.get('hidden_dim', 512),
                    'latent_dim': model_cfg.get('latent_dim', 128),
                    'max_length': model_cfg.get('max_length', 50),
                    'num_layers': model_cfg.get('num_layers', 3),
                    'num_heads': model_cfg.get('num_heads', 4),
                    'dropout': model_cfg.get('dropout', 0.2),
                    'condition_dim': model_cfg.get('condition_dim'),
                    'mem_tokens': model_cfg.get('mem_tokens', 16),
                    'esm_dim': model_cfg.get('esm_dim'),
                    'gat_heads': model_cfg.get('gat_heads', 4),
                    'gat_window': model_cfg.get('gat_window', 3),
                    'use_gat': model_cfg.get('use_gat', True),
                    'fusion_type': model_cfg.get('fusion_type', 'cross_attention'),
                }
            else:
                generator_kwargs = {
                    'vocab_size': model_cfg.get('vocab_size', VOCAB.vocab_size),
                    'embedding_dim': model_cfg.get('embedding_dim', 64),
                    'hidden_dim': model_cfg.get('hidden_dim', 256),
                    'latent_dim': model_cfg.get('latent_dim', 128),
                    'num_layers': model_cfg.get('num_layers', 2),
                    'dropout': model_cfg.get('dropout', 0.2),
                    'condition_dim': model_cfg.get('condition_dim'),
                    'bidirectional': model_cfg.get('bidirectional', False),
                    'use_attention': model_cfg.get('use_attention', False),
                }

        # Create and load
        generator = generator_class(**generator_kwargs)
        generator.load_state_dict(ckpt['generator'])
        
        sampler = cls(generator, device)
        sampler.checkpoint_path = str(Path(checkpoint_path).resolve())
        # Keep non-tensor provenance/configuration fields available to
        # downstream evaluators without exposing or duplicating model weights.
        sampler.checkpoint_metadata = {
            key: ckpt.get(key)
            for key in (
                'epoch', 'global_step', 'generator_class', 'model_config',
                'data_metadata', 'condition_metadata', 'warmup_config',
                'scst_config', 'parent_checkpoint', 'artifact_reportable',
                'run_metadata', 'amp_oracle_audit', 'hemolysis_oracle_audit',
            )
            if key in ckpt
        }
        return sampler
        
    # =========================================================================
    # SAMPLING METHODS
    # =========================================================================
    
    @torch.no_grad()
    def sample(
        self,
        n: int = 100,
        conditions: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        min_length: int = 5,
        max_length: int = 50,
        batch_size: int = 64,
    ) -> List[str]:
        """
        Generate peptide sequences.
        
        Args:
            n: Number of sequences to generate
            conditions: Optional conditioning tensor (n, cond_dim)
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling threshold
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            batch_size: Batch size for generation
            
        Returns:
            List of peptide sequences
        """
        sequences = []
        max_total_attempts = n * 5  # Prevent infinite loop if generation is very poor
        total_generated = 0
        
        while len(sequences) < n and total_generated < max_total_attempts:
            curr_batch = min(batch_size, n - len(sequences))
            # If we are close, still generate at least a small batch for efficiency
            curr_batch = max(curr_batch, min(batch_size, 32))
            
            # Get conditions for this batch
            conds = None
            if conditions is not None:
                # Sample conditions with replacement if we run out
                idx = torch.randint(0, len(conditions), (curr_batch,))
                conds = conditions[idx].to(self.device)
                
            # Generate
            batch_seqs = self._generate_batch(
                curr_batch, conds, temperature, top_k, top_p, min_length, max_length
            )
            total_generated += curr_batch
            
            # Filter by length
            for seq in batch_seqs:
                if min_length <= len(seq) <= max_length:
                    sequences.append(seq)
                    if len(sequences) >= n:
                        break
                        
        logger.info(f"Generated {len(sequences)}/{n} valid sequences (total attempted: {total_generated})")
        return sequences[:n]
        
    def _generate_batch(
        self,
        batch_size: int,
        conditions: Optional[torch.Tensor],
        temperature: float,
        top_k: int,
        top_p: float,
        min_length: int,
        max_length: int,
    ) -> List[str]:
        """Generate a batch autoregressively with the requested sampling policy.

        Re-sampling a completed tensor of per-position logits is invalid for an
        autoregressive model: later logits were conditioned on the model's first
        sampled prefix, not on the replacement tokens.  Delegate sampling to the
        generator so top-k/top-p/temperature are applied before each next-token
        decision.
        """
        z = torch.randn(batch_size, self.G.latent_dim, device=self.device)

        # If the model was trained with conditions but none are provided,
        # supply zero conditions so init_hidden gets the right input dim.
        if conditions is None and getattr(self.G, 'condition_dim', None):
            conditions = torch.zeros(
                batch_size, self.G.condition_dim, device=self.device
            )

        tokens = self.G.generate(
            batch_size=batch_size,
            z=z,
            condition=conditions,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_length=min_length,
            max_length=max_length,
            device=self.device,
        )

        # Both production generators include the leading SOS token.
        if tokens.size(1) > 0 and (
            tokens[:, 0] == getattr(self.G, 'sos_idx', self.vocab.sos_idx)
        ).all():
            tokens = tokens[:, 1:]

        return self._tokens_to_sequences(tokens)

        
    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Sample tokens from logits using specified strategy."""
        batch_size, seq_len, vocab_size = logits.shape
        
        # Apply temperature
        logits = logits / max(temperature, 1e-8)
        
        tokens = []
        for pos in range(seq_len):
            pos_logits = logits[:, pos, :]  # (batch, vocab)
            
            # Top-k filtering
            if top_k > 0:
                top_k_val = min(top_k, vocab_size)
                indices_to_remove = pos_logits < torch.topk(pos_logits, top_k_val)[0][..., -1, None]
                pos_logits[indices_to_remove] = float('-inf')
                
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(pos_logits, descending=True)
                cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumsum > p
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                pos_logits[indices_to_remove] = float('-inf')
                
            # Sample
            probs = F.softmax(pos_logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            tokens.append(token)
            
        return torch.stack(tokens, dim=1)
        
    def _tokens_to_sequences(self, tokens: torch.Tensor) -> List[str]:
        """Convert token indices to peptide sequences."""
        sequences = []
        
        for i in range(tokens.size(0)):
            seq_tokens = tokens[i].cpu().tolist()
            
            # Convert to amino acids
            seq = []
            for t in seq_tokens:
                aa = self.vocab.idx_to_aa.get(t, '')
                if aa in ['<PAD>', '<EOS>', '']:
                    break
                if aa not in ['<SOS>', '<UNK>']:
                    seq.append(aa)
                    
            sequences.append(''.join(seq))
            
        return sequences
        
    @torch.no_grad()
    def sample_ii_screen(
        self,
        n: int = 100,
        ii_threshold: float = 40.0,
        oversample: int = 3,
        max_attempts: int = 10,
        conditions: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        min_length: int = 5,
        max_length: int = 50,
        batch_size: int = 64,
    ) -> List[str]:
        """
        Generate sequences passing an empirical Instability Index screen.

        Uses rejection sampling: generates oversample*n sequences per attempt
        and filters by II. Stops when n passing sequences are collected or
        max_attempts is exhausted.

        Args:
            n: Target number of II-screen-passing sequences
            ii_threshold: Maximum Instability Index (default: 40.0)
            oversample: Multiplier — generate this many × n per attempt
            max_attempts: Max generation rounds before giving up
            conditions, temperature, top_k, top_p, min_length, max_length, batch_size:
                Passed through to sample()

        Returns:
            Passing sequences (may be fewer than n if max_attempts is exceeded).

        This is a post-generation empirical screen, not evidence of physical
        stability. Reportable benchmark generation forbids this filter.
        """
        from ..evaluation.stability import calculate_instability_index

        passing_seqs: List[str] = []
        attempts = 0

        while len(passing_seqs) < n and attempts < max_attempts:
            need = n - len(passing_seqs)
            to_generate = need * oversample

            batch_seqs = self.sample(
                n=to_generate,
                conditions=conditions,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_length=min_length,
                max_length=max_length,
                batch_size=batch_size,
            )

            for seq in batch_seqs:
                if len(passing_seqs) >= n:
                    break
                try:
                    ii = calculate_instability_index(seq)
                    if ii < ii_threshold:
                        passing_seqs.append(seq)
                except Exception:
                    continue

            attempts += 1
            logger.info(
                f"sample_ii_screen attempt {attempts}: "
                f"{len(passing_seqs)}/{n} passing sequences collected "
                f"({len(batch_seqs)} generated, "
                f"rate={len(passing_seqs)/max(1, len(passing_seqs) + (to_generate - len(batch_seqs)))*100:.0f}%)"
            )

        if len(passing_seqs) < n:
            logger.warning(
                f"sample_ii_screen: only {len(passing_seqs)}/{n} passing sequences found "
                f"after {attempts} attempts. "
                f"Try increasing oversample or relaxing ii_threshold."
            )

        return passing_seqs

    @torch.no_grad()
    def sample_stable(self, *args, stability_threshold: float = 40.0, **kwargs) -> List[str]:
        """Deprecated compatibility alias for sample_ii_screen."""
        return self.sample_ii_screen(
            *args, ii_threshold=stability_threshold, **kwargs
        )

    # =========================================================================
    # CONDITIONAL GENERATION
    # =========================================================================

    @torch.no_grad()
    def sample_with_properties(
        self,
        n: int,
        target_properties: Dict[str, float],
        tolerance: float = 0.2,
        max_attempts: int = 10,
        **kwargs,
    ) -> List[str]:
        """
        Generate sequences with target properties.
        
        Args:
            n: Number of sequences
            target_properties: Dict with target values, e.g.:
                {'instability_index': 30, 'charge': 5}
            tolerance: Acceptable deviation from target
            max_attempts: Max generation attempts
            **kwargs: Passed to sample()
            
        Returns:
            List of sequences matching criteria
        """
        from ..data.features import PeptideFeatureExtractor
        extractor = PeptideFeatureExtractor()
        
        valid_seqs = []
        attempts = 0
        
        while len(valid_seqs) < n and attempts < max_attempts:
            # Generate batch
            batch_seqs = self.sample(n=n*2, **kwargs)
            
            # Filter by properties
            for seq in batch_seqs:
                if len(valid_seqs) >= n:
                    break
                    
                try:
                    features = extractor.extract_dict(seq)
                    match = True

                    for prop, target in target_properties.items():
                        if prop in features:
                            actual = features[prop]
                            if abs(actual - target) > tolerance * abs(target):
                                match = False
                                break
                                
                    if match:
                        valid_seqs.append(seq)
                except:
                    continue
                    
            attempts += 1
            
        return valid_seqs
        
    # =========================================================================
    # OUTPUT
    # =========================================================================
    
    def save_fasta(
        self,
        sequences: List[str],
        output_path: str,
        prefix: str = 'gen',
    ):
        """Save sequences to FASTA file."""
        with open(output_path, 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f">{prefix}_{i+1}\n{seq}\n")
        logger.info(f"Saved {len(sequences)} sequences to {output_path}")
        
    def save_csv(
        self,
        sequences: List[str],
        output_path: str,
        include_features: bool = False,
    ):
        """Save sequences to CSV file."""
        import csv
        
        rows = []
        if include_features:
            from ..data.features import PeptideFeatureExtractor
            extractor = PeptideFeatureExtractor()
            
        for i, seq in enumerate(sequences):
            row = {'id': i+1, 'sequence': seq, 'length': len(seq)}
            
            if include_features:
                try:
                    features = extractor.extract_dict(seq)
                    row.update(features)
                except:
                    pass
                    
            rows.append(row)
            
        with open(output_path, 'w', newline='') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
                
        logger.info(f"Saved {len(sequences)} sequences to {output_path}")


def load_generator(checkpoint_path: str, device=None) -> PeptideSampler:
    """Convenience function to load a generator."""
    return PeptideSampler.from_checkpoint(checkpoint_path, device=device)
