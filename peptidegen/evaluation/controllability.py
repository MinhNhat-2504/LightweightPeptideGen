"""
C10 — Controllability evaluation for conditional generation.

The paper claims "explicit conditioning on eight physicochemical properties via
Cross-Attention" but never measures whether the generator actually *obeys* the
condition. This module fills that gap: it sweeps a target value for one
physicochemical feature, generates sequences conditioned on it, measures the
achieved property, and reports how tightly generation tracks the request
(Spearman/Pearson correlation + mean absolute error in raw units).

This is a new, reviewer-friendly result that directly substantiates the
conditioning contribution — independent of stability/AMP metrics.

Usage
-----
    from peptidegen.evaluation.controllability import ControllabilityEvaluator
    ce = ControllabilityEvaluator.from_train_csv(sampler, "dataset/train.csv")
    report = ce.sweep("charge_at_pH7", n_per=200)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

# condition feature -> function measuring it on a generated sequence string
from .stability import (
    calculate_instability_index,
    calculate_gravy,
    calculate_aliphatic_index,
    calculate_charge_at_pH,
    calculate_aromaticity,
)
from .metrics import calculate_hydrophobic_moment

MEASURE = {
    "instability_index": calculate_instability_index,
    "gravy": calculate_gravy,
    "aliphatic_index": calculate_aliphatic_index,
    "charge_at_pH7": lambda s: calculate_charge_at_pH(s, 7.0),
    "aromaticity": calculate_aromaticity,
    "hydrophobic_moment": calculate_hydrophobic_moment,
    "length": len,
}

# the 8-dim condition order used during training (ConditionalPeptideDataset)
CONDITION_FEATURES = [
    "instability_index", "therapeutic_score", "hemolytic_score", "aliphatic_index",
    "hydrophobic_moment", "gravy", "charge_at_pH7", "aromaticity",
]


class ControllabilityEvaluator:
    def __init__(self, sampler, feature_names: List[str], feature_stats: Dict[str, Dict[str, float]]):
        self.sampler = sampler
        self.feature_names = feature_names            # condition order
        self.stats = feature_stats                    # {feat: {mean,std}}

    # ------------------------------------------------------------------ #
    @classmethod
    def from_train_csv(cls, sampler, train_csv: str):
        """Recover the exact condition order + normalisation stats from train.csv."""
        import pandas as pd

        df = pd.read_csv(train_csv)
        names = [c for c in CONDITION_FEATURES if c in df.columns]
        stats = {c: {"mean": float(df[c].mean()), "std": float(df[c].std()) + 1e-8}
                 for c in names}
        logger.info(f"Controllability condition order: {names}")
        return cls(sampler, names, stats)

    # ------------------------------------------------------------------ #
    def _condition_vector(self, feature: str, raw_value: float, n: int) -> torch.Tensor:
        """All features at their mean (z=0) except `feature` set to the target."""
        vec = torch.zeros(len(self.feature_names), dtype=torch.float)
        if feature in self.feature_names:
            i = self.feature_names.index(feature)
            st = self.stats[feature]
            vec[i] = (raw_value - st["mean"]) / st["std"]
        return vec.unsqueeze(0).repeat(n, 1)

    def sweep(
        self,
        feature: str,
        targets: Optional[Sequence[float]] = None,
        n_per: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        min_length: int = 5,
        max_length: int = 50,
    ) -> Dict:
        """Sweep `feature` across target values; measure achieved property."""
        if feature not in MEASURE:
            raise ValueError(f"Cannot measure '{feature}'. Options: {list(MEASURE)}")
        measure = MEASURE[feature]

        # default targets = quantiles of the training distribution of this feature
        if targets is None:
            st = self.stats.get(feature, {"mean": 0.0, "std": 1.0})
            targets = [st["mean"] + q * st["std"] for q in (-1.5, -0.75, 0.0, 0.75, 1.5)]

        rows = []
        for tgt in targets:
            cond = self._condition_vector(feature, float(tgt), n_per)
            seqs = self.sampler.sample(
                n=n_per, conditions=cond, temperature=temperature, top_p=top_p,
                min_length=min_length, max_length=max_length, batch_size=min(128, n_per),
            )
            achieved = [measure(s) for s in seqs if len(s) >= 1]
            arr = np.asarray(achieved, dtype=float)
            rows.append({
                "target": float(tgt),
                "achieved_mean": float(arr.mean()) if arr.size else float("nan"),
                "achieved_std": float(arr.std()) if arr.size else float("nan"),
                "n": int(arr.size),
            })
            logger.info(f"[{feature}] target={tgt:.3f} -> achieved={rows[-1]['achieved_mean']:.3f} (n={arr.size})")

        tg = np.array([r["target"] for r in rows])
        ac = np.array([r["achieved_mean"] for r in rows])
        valid = ~np.isnan(ac)
        report = {"feature": feature, "rows": rows}
        if valid.sum() >= 2:
            from scipy.stats import spearmanr, pearsonr

            report["spearman"] = float(spearmanr(tg[valid], ac[valid]).correlation)
            report["pearson"] = float(pearsonr(tg[valid], ac[valid])[0])
            report["mae"] = float(np.mean(np.abs(tg[valid] - ac[valid])))
        return report

    def sweep_all(self, features: Optional[List[str]] = None, **kw) -> Dict[str, Dict]:
        feats = features or [f for f in self.feature_names if f in MEASURE]
        return {f: self.sweep(f, **kw) for f in feats}
