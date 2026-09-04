"""
C10 — Controllability evaluation for conditional generation.

The model uses explicit physicochemical conditioning, but the submitted paper
did not measure whether the generator actually *obeys* the
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
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

# condition feature -> function measuring it on a generated sequence string
from ..data.features import PeptideFeatureExtractor

_FEATURE_EXTRACTOR = PeptideFeatureExtractor()


def _measure(feature: str):
    """Use the exact feature implementation used to create training CSVs."""
    return lambda sequence: float(_FEATURE_EXTRACTOR.extract_dict(sequence)[feature])

MEASURE = {name: _measure(name) for name in (
    "instability_index", "gravy", "aliphatic_index", "charge_at_pH7",
    "aromaticity", "hydrophobic_moment", "length",
)}

# Legacy condition candidates. Reportable evaluation uses the exact ordered
# feature list and training statistics stored in the checkpoint.
CONDITION_FEATURES = [
    "instability_index", "therapeutic_score", "hemolytic_score", "aliphatic_index",
    "hydrophobic_moment", "gravy", "charge_at_pH7", "aromaticity",
]


class ControllabilityEvaluator:
    def __init__(self, sampler, feature_names: List[str], feature_stats: Dict[str, Dict[str, float]],
                 provenance: Optional[Dict] = None):
        self.sampler = sampler
        self.feature_names = feature_names            # condition order
        self.stats = feature_stats                    # {feat: {mean,std}}
        self.provenance = provenance or {}

    # ------------------------------------------------------------------ #
    @classmethod
    def from_train_csv(cls, sampler, train_csv: str, label_value: int = 1,
                       audit_rows: int = 512, rtol: float = 1e-4,
                       atol: float = 1e-5):
        """Recover condition metadata and verify feature-definition identity.

        Controllability is meaningless if a column called ``aliphatic_index``
        was produced as a residue fraction but is re-measured as the Ikai
        index.  This constructor therefore checks stored values against the
        exact current feature extractor and refuses to continue on mismatch.
        """
        import pandas as pd

        csv_path = Path(train_csv)
        df = pd.read_csv(csv_path)
        if "label" in df.columns:
            df = df.loc[df["label"] == label_value].copy()
        if df.empty:
            raise ValueError(f"{train_csv} has no rows after label={label_value} filtering")

        checkpoint_data = (getattr(sampler, "checkpoint_metadata", {}) or {}).get("data_metadata") or {}
        if (getattr(sampler, "checkpoint_metadata", {}) or {}).get("artifact_reportable") is not True:
            raise ValueError("checkpoint is not marked artifact_reportable")
        if checkpoint_data.get("reportable_data") is not True:
            raise ValueError("checkpoint does not prove a provenance-audited dataset build")
        names = checkpoint_data.get("condition_feature_names")
        if not names:
            raise ValueError("checkpoint does not record the ordered condition feature list")
        expected_dim = getattr(sampler.G, "condition_dim", None)
        if expected_dim and len(names) != expected_dim:
            raise ValueError(
                f"checkpoint expects {expected_dim} condition features, metadata/CSV provides {len(names)}: {names}"
            )
        missing = [name for name in names if name not in df.columns]
        if missing:
            raise ValueError(f"{train_csv} is missing checkpoint condition columns: {missing}")
        if df[names].isna().any().any():
            raise ValueError(f"{train_csv} contains missing condition values")

        measurable = [name for name in names if name in MEASURE]
        sample = df.iloc[np.linspace(0, len(df) - 1, min(audit_rows, len(df)), dtype=int)]
        mismatches = {}
        for name in measurable:
            stored = sample[name].to_numpy(dtype=float)
            recalculated = np.asarray([MEASURE[name](str(seq)) for seq in sample["sequence"]])
            close = np.isclose(stored, recalculated, rtol=rtol, atol=atol, equal_nan=False)
            if not bool(close.all()):
                mismatches[name] = {
                    "rows_checked": int(len(sample)),
                    "rows_mismatched": int((~close).sum()),
                    "example_stored": float(stored[np.flatnonzero(~close)[0]]),
                    "example_recalculated": float(recalculated[np.flatnonzero(~close)[0]]),
                }
        if mismatches:
            raise ValueError(
                "condition-feature definitions do not match the current evaluator; "
                f"rebuild the dataset before reporting controllability: {mismatches}"
            )

        computed_stats = {
            c: {
                "mean": float(df[c].mean()),
                "std": float(df[c].to_numpy(dtype=float).std(ddof=0)) + 1e-8,
                "min": float(df[c].min()),
                "max": float(df[c].max()),
            }
            for c in names
        }
        saved_stats = checkpoint_data.get("condition_feature_stats")
        stats = saved_stats or computed_stats
        if saved_stats:
            for name in names:
                for key in ("mean", "std"):
                    if not np.isclose(float(saved_stats[name][key]), computed_stats[name][key], rtol=rtol, atol=atol):
                        raise ValueError(
                            f"checkpoint {name}.{key}={saved_stats[name][key]} does not match "
                            f"{train_csv} value {computed_stats[name][key]}"
                        )
        train_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        expected_train_sha256 = (checkpoint_data.get("train") or {}).get("sha256")
        if not expected_train_sha256 or expected_train_sha256 != train_sha256:
            raise ValueError("training CSV hash does not match checkpoint data provenance")
        checkpoint_path = getattr(sampler, "checkpoint_path", None)
        checkpoint_sha256 = None
        if checkpoint_path:
            checkpoint_sha256 = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
        provenance = {
            "train_csv": str(csv_path.resolve()),
            "train_csv_sha256": train_sha256,
            "label_filter": label_value if "label" in pd.read_csv(csv_path, nrows=1).columns else None,
            "rows": int(len(df)),
            "feature_definition_audit_rows": int(len(sample)),
            "feature_definitions_match": True,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_build_audit": checkpoint_data.get("dataset_build_audit"),
        }
        logger.info(f"Controllability condition order: {names}")
        return cls(sampler, names, stats, provenance)

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
            candidates = [st["mean"] + q * st["std"] for q in (-1.5, -0.75, 0.0, 0.75, 1.5)]
            lower, upper = st.get("min"), st.get("max")
            targets = [
                float(np.clip(value, lower, upper)) if lower is not None and upper is not None else float(value)
                for value in candidates
            ]

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
                "achieved_values": arr.tolist(),
            })
            logger.info(f"[{feature}] target={tgt:.3f} -> achieved={rows[-1]['achieved_mean']:.3f} (n={arr.size})")

        tg = np.array([r["target"] for r in rows])
        ac = np.array([r["achieved_mean"] for r in rows])
        valid = ~np.isnan(ac)
        report = {"feature": feature, "rows": rows, "n_per_target_requested": n_per}
        if valid.sum() >= 2:
            from scipy.stats import spearmanr, pearsonr

            report["spearman"] = float(spearmanr(tg[valid], ac[valid]).correlation)
            report["pearson"] = float(pearsonr(tg[valid], ac[valid])[0])
            report["mae"] = float(np.mean(np.abs(tg[valid] - ac[valid])))
            rng = np.random.default_rng(20260904)
            boot = {"spearman": [], "pearson": [], "mae": []}
            valid_rows = [rows[index] for index in np.flatnonzero(valid)]
            for _ in range(2000):
                boot_means = []
                for row in valid_rows:
                    values = np.asarray(row["achieved_values"], dtype=float)
                    boot_means.append(float(rng.choice(values, len(values), replace=True).mean()))
                target_values = np.asarray([row["target"] for row in valid_rows], dtype=float)
                achieved_values = np.asarray(boot_means, dtype=float)
                boot["spearman"].append(float(spearmanr(target_values, achieved_values).correlation))
                boot["pearson"].append(float(pearsonr(target_values, achieved_values)[0]))
                boot["mae"].append(float(np.mean(np.abs(target_values - achieved_values))))
            report["bootstrap_ci95"] = {
                key: [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]
                for key, values in boot.items()
            }
        return report

    def sweep_all(self, features: Optional[List[str]] = None, **kw) -> Dict[str, Dict]:
        feats = features or [f for f in self.feature_names if f in MEASURE]
        results = {f: self.sweep(f, **kw) for f in feats}
        expected = int(kw.get("n_per", 200))
        complete = all(
            row["n"] == expected
            for result in results.values()
            for row in result["rows"]
        )
        return {
            "schema_version": 2,
            "reportable": complete,
            "provenance": self.provenance,
            "results": results,
        }
