"""
Data utilities shared by all baseline models.
Wraps peptidegen.data pipeline for reuse.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any, Optional

from peptidegen.data.dataset import ConditionalPeptideDataset
from peptidegen.data.vocabulary import VOCAB


def get_dataloaders(
    train_csv: str,
    val_csv: str,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_seq_length: int = 50,
    min_seq_length: int = 5,
    max_samples: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders using the project's ConditionalPeptideDataset.

    Args:
        max_samples: if >0, cap the number of train rows (and val at max_samples//5).
            Use to train baselines on the same subset as the proposed model for a
            fair comparison / faster runs.

    Returns:
        (train_loader, val_loader)
    """
    print(f"[data_utils] Loading train data from {train_csv}...")
    train_dataset = ConditionalPeptideDataset.from_csv(
        csv_path=train_csv,
        sequence_col='sequence',
        label_col='label',
        max_length=max_seq_length,
        min_length=min_seq_length,
        normalize_features=True,
    )

    print(f"[data_utils] Loading val data from {val_csv}...")
    val_dataset = ConditionalPeptideDataset.from_csv(
        csv_path=val_csv,
        sequence_col='sequence',
        label_col='label',
        max_length=max_seq_length,
        min_length=min_seq_length,
        normalize_features=True,
        feature_stats=train_dataset.get_feature_stats(),
    )

    if max_samples and max_samples > 0:
        def _cap(ds, k):
            ds.sequences = ds.sequences[:k]
            ds.features = ds.features[:k]
            if getattr(ds, 'has_labels', False):
                ds.labels = ds.labels[:k]
        _cap(train_dataset, max_samples)
        _cap(val_dataset, max(1, max_samples // 5))
        print(f"[data_utils] Subset: train={len(train_dataset)}, val={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print(f"[data_utils] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader


def get_vocab_info() -> Dict[str, Any]:
    """Return vocabulary information."""
    return {
        'vocab_size': VOCAB.vocab_size,
        'pad_idx': VOCAB.pad_idx,
        'sos_idx': VOCAB.sos_idx,
        'eos_idx': VOCAB.eos_idx,
        'unk_idx': VOCAB.unk_idx,
        'vocab': VOCAB,
    }


def decode_tokens(token_ids: torch.Tensor) -> list:
    """
    Decode a batch of token id tensors to sequence strings.

    Args:
        token_ids: (B, L) tensor

    Returns:
        List of strings
    """
    seqs = []
    for ids in token_ids.tolist():
        seq = VOCAB.decode(ids, remove_special_tokens=True)
        seqs.append(seq)
    return seqs
