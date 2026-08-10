"""
Peptide Dataset classes for loading and processing peptide data
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import random

from .vocabulary import VOCAB, PeptideVocabulary
from .features import PeptideFeatureExtractor


class PeptideDataset(Dataset):
    """
    General peptide dataset that handles sequences and optional labels/features.
    """
    
    def __init__(
        self,
        sequences: List[str],
        labels: Optional[List[int]] = None,
        features: Optional[List[Dict]] = None,
        vocab: PeptideVocabulary = VOCAB,
        max_length: int = 50,
        min_length: int = 5,
        add_special_tokens: bool = True,
        feature_extractor: Optional[PeptideFeatureExtractor] = None,
    ):
        """
        Initialize peptide dataset.
        
        Args:
            sequences: List of peptide sequences
            labels: Optional list of labels (e.g., AMP/non-AMP)
            features: Optional list of feature dictionaries
            vocab: Vocabulary for encoding sequences
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            add_special_tokens: Whether to add SOS/EOS tokens
            feature_extractor: Optional feature extractor for computing features on-the-fly
        """
        self.vocab = vocab
        self.max_length = max_length
        self.min_length = min_length
        self.add_special_tokens = add_special_tokens
        self.feature_extractor = feature_extractor
        
        # Filter sequences by length
        self.sequences = []
        self.labels = []
        self.features = []
        
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            if self.min_length <= seq_len <= self.max_length:
                self.sequences.append(seq.upper())
                if labels is not None:
                    self.labels.append(labels[i])
                if features is not None:
                    self.features.append(features[i])
                    
        self.has_labels = len(self.labels) > 0
        self.has_features = len(self.features) > 0
        
        print(f"Loaded {len(self.sequences)} sequences "
              f"(filtered from {len(sequences)} by length [{min_length}, {max_length}])")
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
                - 'input_ids': Encoded sequence tensor
                - 'attention_mask': Attention mask (1 for real tokens, 0 for padding)
                - 'length': Original sequence length
                - 'label': Label (if available)
                - 'features': Feature tensor (if available)
        """
        sequence = self.sequences[idx]
        
        # Encode sequence
        input_ids = self.vocab.encode(
            sequence, 
            add_special_tokens=self.add_special_tokens,
            max_length=self.max_length + 2 if self.add_special_tokens else self.max_length
        )
        
        # Create attention mask (use 'tok' to avoid shadowing outer 'idx' parameter)
        attention_mask = [1 if tok != self.vocab.pad_idx else 0 for tok in input_ids]
        
        # Calculate actual length
        length = len(sequence)
        if self.add_special_tokens:
            length += 2  # SOS and EOS
            
        result = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'length': torch.tensor(length, dtype=torch.long),
            'sequence': sequence,
        }
        
        # Add label if available
        if self.has_labels:
            result['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        # Add features if available or compute on-the-fly
        if self.has_features:
            result['features'] = torch.tensor(
                list(self.features[idx].values()), dtype=torch.float
            )
        elif self.feature_extractor is not None:
            features = self.feature_extractor.extract(sequence)
            result['features'] = torch.tensor(features, dtype=torch.float)
            
        return result


class PeptideFastaDataset(PeptideDataset):
    """
    Dataset that loads peptides from a FASTA file.
    """
    
    def __init__(
        self,
        fasta_path: Union[str, Path],
        vocab: PeptideVocabulary = VOCAB,
        max_length: int = 50,
        min_length: int = 5,
        add_special_tokens: bool = True,
        feature_extractor: Optional[PeptideFeatureExtractor] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize dataset from FASTA file.
        
        Args:
            fasta_path: Path to FASTA file
            vocab: Vocabulary for encoding
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            add_special_tokens: Whether to add SOS/EOS tokens
            feature_extractor: Optional feature extractor
            max_samples: Maximum number of samples to load (None for all)
        """
        sequences, labels = self._load_fasta(fasta_path, max_samples)
        
        super().__init__(
            sequences=sequences,
            labels=labels if any(l is not None for l in labels) else None,
            vocab=vocab,
            max_length=max_length,
            min_length=min_length,
            add_special_tokens=add_special_tokens,
            feature_extractor=feature_extractor,
        )
        
    @staticmethod
    def _load_fasta(fasta_path: Union[str, Path], max_samples: Optional[int] = None) -> Tuple[List[str], List[Optional[int]]]:
        """
        Load sequences from FASTA file.
        
        Args:
            fasta_path: Path to FASTA file
            max_samples: Maximum number of samples
            
        Returns:
            Tuple of (sequences, labels)
        """
        sequences = []
        labels = []
        
        current_header = None
        current_sequence = []
        
        with open(fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('>'):
                    # Save previous sequence
                    if current_header is not None and current_sequence:
                        seq = ''.join(current_sequence)
                        sequences.append(seq)
                        
                        # Parse label from header (e.g., >AMP_0 or >nonAMP_1)
                        if 'nonAMP' in current_header or 'non_AMP' in current_header:
                            labels.append(0)
                        elif 'AMP' in current_header:
                            labels.append(1)
                        else:
                            labels.append(None)
                            
                        if max_samples and len(sequences) >= max_samples:
                            break
                            
                    current_header = line[1:]
                    current_sequence = []
                else:
                    current_sequence.append(line)
                    
            # Don't forget the last sequence
            if current_header is not None and current_sequence:
                if not max_samples or len(sequences) < max_samples:
                    seq = ''.join(current_sequence)
                    sequences.append(seq)
                    
                    if 'nonAMP' in current_header or 'non_AMP' in current_header:
                        labels.append(0)
                    elif 'AMP' in current_header:
                        labels.append(1)
                    else:
                        labels.append(None)
                        
        return sequences, labels


class PeptideGenerationDataset(Dataset):
    """
    Dataset specifically for training generative models.
    Provides teacher forcing targets and other generation-specific data.
    """
    
    def __init__(
        self,
        sequences: List[str],
        vocab: PeptideVocabulary = VOCAB,
        max_length: int = 50,
        min_length: int = 5,
    ):
        """
        Initialize generation dataset.
        
        Args:
            sequences: List of peptide sequences
            vocab: Vocabulary
            max_length: Maximum sequence length
            min_length: Minimum sequence length
        """
        self.vocab = vocab
        self.max_length = max_length
        self.min_length = min_length
        
        # Filter sequences
        self.sequences = [
            seq.upper() for seq in sequences 
            if min_length <= len(seq) <= max_length
        ]
        
        # Actual max length for padding
        self.padded_length = max_length + 2  # +2 for SOS and EOS
        
        print(f"Generation dataset: {len(self.sequences)} sequences")
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample for generation training.
        
        Returns:
            Dictionary containing:
                - 'input_ids': Input sequence (SOS + sequence)
                - 'target_ids': Target sequence (sequence + EOS)
                - 'length': Sequence length
        """
        sequence = self.sequences[idx]
        seq_len = len(sequence)
        
        # Encode without special tokens first
        encoded = [self.vocab.aa_to_idx[aa] for aa in sequence if aa in self.vocab.aa_to_idx]
        
        # Input: SOS + sequence (for teacher forcing)
        input_ids = [self.vocab.sos_idx] + encoded
        
        # Target: sequence + EOS
        target_ids = encoded + [self.vocab.eos_idx]
        
        # Pad to fixed length
        input_padded = input_ids + [self.vocab.pad_idx] * (self.padded_length - len(input_ids))
        target_padded = target_ids + [self.vocab.pad_idx] * (self.padded_length - len(target_ids))
        
        # Truncate if necessary
        input_padded = input_padded[:self.padded_length]
        target_padded = target_padded[:self.padded_length]
        
        return {
            'input_ids': torch.tensor(input_padded, dtype=torch.long),
            'target_ids': torch.tensor(target_padded, dtype=torch.long),
            'length': torch.tensor(seq_len + 1, dtype=torch.long),  # +1 for EOS
            'sequence': sequence,
        }
    
    @classmethod
    def from_fasta(cls, fasta_path: Union[str, Path], **kwargs) -> 'PeptideGenerationDataset':
        """Create dataset from FASTA file."""
        sequences, _ = PeptideFastaDataset._load_fasta(fasta_path)
        return cls(sequences=sequences, **kwargs)


class ConditionalPeptideDataset(Dataset):
    """
    Dataset for conditional peptide generation with features from CSV.
    Uses peptide properties as conditions for controlled generation.
    
    Features used:
        - instability_index: Protein stability metric (< 40 = stable)
        - therapeutic_score: Therapeutic potential
        - hemolytic_score: Toxicity indicator (lower = safer)
        - aliphatic_index: Thermostability
        - hydrophobic_moment: Amphipathicity
        - gravy: Hydropathicity
        - charge_at_pH7: Net charge
        - aromaticity: Aromatic content
    """
    
    # Feature columns to use as conditions
    CONDITION_FEATURES = [
        'instability_index',
        'therapeutic_score', 
        'hemolytic_score',
        'aliphatic_index',
        'hydrophobic_moment',
        'gravy',
        'charge_at_pH7',
        'aromaticity',
    ]
    
    # Normalization statistics (will be computed from data)
    FEATURE_STATS = {
        'instability_index': {'mean': 40.0, 'std': 30.0},
        'therapeutic_score': {'mean': 0.5, 'std': 1.0},
        'hemolytic_score': {'mean': 0.3, 'std': 0.5},
        'aliphatic_index': {'mean': 80.0, 'std': 40.0},
        'hydrophobic_moment': {'mean': 0.4, 'std': 0.3},
        'gravy': {'mean': -0.3, 'std': 1.0},
        'charge_at_pH7': {'mean': 2.0, 'std': 5.0},
        'aromaticity': {'mean': 0.1, 'std': 0.1},
    }
    
    def __init__(
        self,
        sequences: List[str],
        features: List[Dict[str, float]],
        labels: Optional[List[int]] = None,
        vocab: PeptideVocabulary = VOCAB,
        max_length: int = 50,
        min_length: int = 5,
        normalize_features: bool = True,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize conditional dataset.
        
        Args:
            sequences: List of peptide sequences
            features: List of feature dictionaries for each sequence
            labels: Optional list of labels (AMP=1, nonAMP=0)
            vocab: Vocabulary for encoding
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            normalize_features: Whether to normalize features
            feature_names: Which features to use (default: CONDITION_FEATURES)
        """
        self.vocab = vocab
        self.max_length = max_length
        self.min_length = min_length
        self.normalize_features = normalize_features
        self.feature_names = feature_names or self.CONDITION_FEATURES
        self.padded_length = max_length + 2  # +2 for SOS and EOS

        # Instance-level copy of feature stats to avoid mutating class variable
        self.feature_stats = dict(self.FEATURE_STATS)
        
        # Filter by sequence length and store data
        self.sequences = []
        self.features = []
        self.labels = []
        
        for i, seq in enumerate(sequences):
            seq = seq.upper()
            if min_length <= len(seq) <= max_length:
                self.sequences.append(seq)
                self.features.append(features[i])
                if labels is not None:
                    self.labels.append(labels[i])
        
        self.has_labels = len(self.labels) > 0
        
        # Compute feature statistics from data
        self._compute_feature_stats()
        
        print(f"Conditional dataset: {len(self.sequences)} sequences")
        print(f"  Features: {self.feature_names}")
        print(f"  Feature dim: {len(self.feature_names)}")
        
    def _compute_feature_stats(self):
        """Compute mean and std for each feature from data."""
        import numpy as np

        for feat_name in self.feature_names:
            values = [f.get(feat_name, 0.0) for f in self.features]
            if values:
                self.feature_stats[feat_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)) + 1e-8  # Avoid division by zero
                }
        
    def _normalize_feature(self, value: float, feat_name: str) -> float:
        """Normalize a single feature value."""
        if not self.normalize_features:
            return value
        stats = self.feature_stats.get(feat_name, {'mean': 0, 'std': 1})
        return (value - stats['mean']) / stats['std']

    def _denormalize_feature(self, value: float, feat_name: str) -> float:
        """Denormalize a feature value back to original scale."""
        stats = self.feature_stats.get(feat_name, {'mean': 0, 'std': 1})
        return value * stats['std'] + stats['mean']
    
    def get_condition_dim(self) -> int:
        """Get dimension of condition vector."""
        return len(self.feature_names)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample with sequence and condition features.
        
        Returns:
            Dictionary containing:
                - 'input_ids': Input sequence (SOS + sequence)
                - 'target_ids': Target sequence (sequence + EOS)
                - 'length': Sequence length
                - 'condition': Normalized feature vector
                - 'features_raw': Raw feature values
                - 'label': AMP label (if available)
        """
        sequence = self.sequences[idx]
        features = self.features[idx]
        seq_len = len(sequence)
        
        # Encode sequence
        encoded = [self.vocab.aa_to_idx[aa] for aa in sequence if aa in self.vocab.aa_to_idx]
        
        # Input: SOS + sequence (for teacher forcing)
        input_ids = [self.vocab.sos_idx] + encoded
        
        # Target: sequence + EOS
        target_ids = encoded + [self.vocab.eos_idx]
        
        # Pad to fixed length
        input_padded = input_ids + [self.vocab.pad_idx] * (self.padded_length - len(input_ids))
        target_padded = target_ids + [self.vocab.pad_idx] * (self.padded_length - len(target_ids))
        
        # Truncate if necessary
        input_padded = input_padded[:self.padded_length]
        target_padded = target_padded[:self.padded_length]
        
        # Extract and normalize condition features
        condition = []
        features_raw = []
        for feat_name in self.feature_names:
            raw_value = features.get(feat_name, 0.0)
            features_raw.append(raw_value)
            condition.append(self._normalize_feature(raw_value, feat_name))
        
        result = {
            'input_ids': torch.tensor(input_padded, dtype=torch.long),
            'target_ids': torch.tensor(target_padded, dtype=torch.long),
            'length': torch.tensor(seq_len + 1, dtype=torch.long),
            'sequence': sequence,
            'condition': torch.tensor(condition, dtype=torch.float),
            'features_raw': torch.tensor(features_raw, dtype=torch.float),
        }
        
        if self.has_labels:
            result['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return result
    
    @classmethod
    def from_csv(
        cls, 
        csv_path: Union[str, Path],
        sequence_col: str = 'sequence',
        label_col: str = 'label',
        **kwargs
    ) -> 'ConditionalPeptideDataset':
        """
        Create dataset from CSV file with features.
        
        Args:
            csv_path: Path to CSV file
            sequence_col: Column name for sequences
            label_col: Column name for labels
            **kwargs: Additional arguments for __init__
            
        Returns:
            ConditionalPeptideDataset instance
        """
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        sequences = df[sequence_col].tolist()
        labels = df[label_col].tolist() if label_col in df.columns else None
        
        # Extract features
        feature_cols = [col for col in cls.CONDITION_FEATURES if col in df.columns]
        features = []
        for _, row in df.iterrows():
            feat_dict = {col: float(row[col]) for col in feature_cols}
            features.append(feat_dict)
        
        print(f"Loaded {len(sequences)} sequences from {csv_path}")
        print(f"Available features: {feature_cols}")
        
        return cls(
            sequences=sequences,
            features=features,
            labels=labels,
            feature_names=feature_cols,
            **kwargs
        )
    
    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """Return feature statistics for reference."""
        return {name: self.FEATURE_STATS.get(name, {}) for name in self.feature_names}
