"""
Peptide Vocabulary - Mapping amino acids to indices
"""

import torch
from typing import Dict, List, Optional


class PeptideVocabulary:
    """
    Vocabulary class for peptide sequences.
    Maps amino acids to indices and vice versa.
    """
    
    # Standard 20 amino acids
    STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self, include_special_tokens: bool = True):
        """
        Initialize vocabulary.
        
        Args:
            include_special_tokens: Whether to include special tokens (PAD, SOS, EOS, UNK)
        """
        self.include_special_tokens = include_special_tokens
        self._build_vocab()
        
    def _build_vocab(self):
        """Build vocabulary mappings."""
        self.aa_to_idx: Dict[str, int] = {}
        self.idx_to_aa: Dict[int, str] = {}
        
        idx = 0
        
        # Add special tokens first
        if self.include_special_tokens:
            for token in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
                self.aa_to_idx[token] = idx
                self.idx_to_aa[idx] = token
                idx += 1
        
        # Add standard amino acids
        for aa in self.STANDARD_AAS:
            self.aa_to_idx[aa] = idx
            self.idx_to_aa[idx] = aa
            idx += 1
            
        # Store special indices
        if self.include_special_tokens:
            self.pad_idx = self.aa_to_idx[self.PAD_TOKEN]
            self.sos_idx = self.aa_to_idx[self.SOS_TOKEN]
            self.eos_idx = self.aa_to_idx[self.EOS_TOKEN]
            self.unk_idx = self.aa_to_idx[self.UNK_TOKEN]
        else:
            self.pad_idx = 0
            self.sos_idx = None
            self.eos_idx = None
            self.unk_idx = None
            
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.aa_to_idx)
    
    def encode(self, sequence: str, add_special_tokens: bool = True, 
               max_length: Optional[int] = None) -> List[int]:
        """
        Encode amino acid sequence to indices.
        
        Args:
            sequence: Amino acid sequence string
            add_special_tokens: Whether to add SOS and EOS tokens
            max_length: Maximum sequence length (will pad if shorter)
            
        Returns:
            List of indices
        """
        indices = []
        
        # Add SOS token
        if add_special_tokens and self.include_special_tokens:
            indices.append(self.sos_idx)
        
        # Encode amino acids
        for aa in sequence.upper():
            if aa in self.aa_to_idx:
                indices.append(self.aa_to_idx[aa])
            elif self.include_special_tokens:
                indices.append(self.unk_idx)
                
        # Add EOS token
        if add_special_tokens and self.include_special_tokens:
            indices.append(self.eos_idx)
            
        # Pad or truncate to max_length
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length]
            elif len(indices) < max_length:
                indices.extend([self.pad_idx] * (max_length - len(indices)))
                
        return indices
    
    def decode(self, indices: List[int], remove_special_tokens: bool = True) -> str:
        """
        Decode indices back to amino acid sequence.
        
        Args:
            indices: List of indices
            remove_special_tokens: Whether to remove special tokens from output
            
        Returns:
            Amino acid sequence string
        """
        sequence = []
        
        for idx in indices:
            if idx in self.idx_to_aa:
                token = self.idx_to_aa[idx]
                
                if remove_special_tokens:
                    if token in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
                        if token == self.EOS_TOKEN:
                            break  # Stop at EOS
                        continue
                        
                sequence.append(token)
                
        return "".join(sequence)
    
    def batch_encode(self, sequences: List[str], add_special_tokens: bool = True,
                     max_length: Optional[int] = None, 
                     return_tensors: bool = True) -> torch.Tensor:
        """
        Encode a batch of sequences.
        
        Args:
            sequences: List of amino acid sequences
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            return_tensors: Whether to return PyTorch tensors
            
        Returns:
            Tensor of shape (batch_size, max_length)
        """
        # Find max length if not specified
        if max_length is None:
            max_length = max(len(s) for s in sequences)
            if add_special_tokens and self.include_special_tokens:
                max_length += 2  # For SOS and EOS
                
        encoded = [
            self.encode(seq, add_special_tokens, max_length) 
            for seq in sequences
        ]
        
        if return_tensors:
            return torch.tensor(encoded, dtype=torch.long)
        return encoded
    
    def batch_decode(self, indices_batch: torch.Tensor, 
                     remove_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of indices.
        
        Args:
            indices_batch: Tensor of shape (batch_size, seq_length)
            remove_special_tokens: Whether to remove special tokens
            
        Returns:
            List of amino acid sequences
        """
        if isinstance(indices_batch, torch.Tensor):
            indices_batch = indices_batch.tolist()
            
        return [
            self.decode(indices, remove_special_tokens) 
            for indices in indices_batch
        ]
    
    def get_aa_frequencies(self, sequences: List[str]) -> Dict[str, float]:
        """
        Calculate amino acid frequencies in a set of sequences.
        
        Args:
            sequences: List of amino acid sequences
            
        Returns:
            Dictionary mapping amino acids to their frequencies
        """
        counts = {aa: 0 for aa in self.STANDARD_AAS}
        total = 0
        
        for seq in sequences:
            for aa in seq.upper():
                if aa in counts:
                    counts[aa] += 1
                    total += 1
                    
        if total > 0:
            frequencies = {aa: count / total for aa, count in counts.items()}
        else:
            frequencies = {aa: 0.0 for aa in self.STANDARD_AAS}
            
        return frequencies


# Global vocabulary instance
VOCAB = PeptideVocabulary(include_special_tokens=True)
