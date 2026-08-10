"""
DataLoader utilities for peptide datasets
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Optional, Callable
import numpy as np


def collate_peptides(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for peptide batches.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary with tensors
    """
    result = {}
    
    # Get all keys from first sample
    keys = batch[0].keys()
    
    for key in keys:
        if key == 'sequence':
            # Keep sequences as list of strings
            result[key] = [sample[key] for sample in batch]
        else:
            # Stack tensors
            result[key] = torch.stack([sample[key] for sample in batch])
            
    return result


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """
    Create a DataLoader for peptide dataset.
    
    Args:
        dataset: Peptide dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory (for GPU)
        drop_last: Whether to drop last incomplete batch
        collate_fn: Custom collate function
        
    Returns:
        DataLoader instance
    """
    if collate_fn is None:
        collate_fn = collate_peptides
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


class InfiniteDataLoader:
    """
    DataLoader that cycles infinitely through the dataset.
    Useful for GAN training where we need continuous batches.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self._create_dataloader()
        
    def _create_dataloader(self):
        """Create internal dataloader."""
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=collate_peptides,
        )
        self.iterator = iter(self.dataloader)
        
    def __iter__(self):
        return self
    
    def __next__(self) -> Dict[str, torch.Tensor]:
        try:
            batch = next(self.iterator)
        except StopIteration:
            # Reset iterator only, reuse existing DataLoader
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get next batch."""
        return next(self)


class BalancedBatchSampler:
    """
    Sampler that ensures balanced batches (equal positive and negative samples).
    Useful for training discriminators.
    """
    
    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        drop_last: bool = True,
    ):
        """
        Initialize balanced sampler.
        
        Args:
            labels: List of labels (0 or 1)
            batch_size: Batch size (must be even)
            drop_last: Whether to drop last incomplete batch
        """
        assert batch_size % 2 == 0, "Batch size must be even for balanced sampling"
        
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Separate indices by label
        self.pos_indices = [i for i, l in enumerate(labels) if l == 1]
        self.neg_indices = [i for i, l in enumerate(labels) if l == 0]
        
        self.n_batches = min(
            len(self.pos_indices) // (batch_size // 2),
            len(self.neg_indices) // (batch_size // 2)
        )
        
    def __iter__(self):
        # Shuffle indices
        pos_shuffled = np.random.permutation(self.pos_indices)
        neg_shuffled = np.random.permutation(self.neg_indices)
        
        half_batch = self.batch_size // 2
        
        for i in range(self.n_batches):
            pos_batch = pos_shuffled[i * half_batch:(i + 1) * half_batch]
            neg_batch = neg_shuffled[i * half_batch:(i + 1) * half_batch]
            
            batch = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch)
            
            yield batch.tolist()
            
    def __len__(self):
        return self.n_batches
