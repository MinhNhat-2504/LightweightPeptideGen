"""
B6/B7 — Real, reproducible AMP & hemolysis oracles.

The original evaluation scored "AMP probability" with a hand-written
physicochemical heuristic (``metrics.estimate_amp_probability``) while the
paper claimed PepGraphormer (AUC 0.815) as the oracle. This module replaces the
heuristic with a **trained ESM-2 classifier** whose performance is measured on
an independent held-out test set — so the number reported in the paper is real
and reproducible offline.

    ESM2Oracle
        frozen ESM-2 (mean-pooled embedding, cached)  ->  LogisticRegression head

* AMP oracle: trained on the project's own labelled corpus (``label`` column in
  train.csv / val.csv / test.csv).
* Hemolysis oracle: trained on an external labelled set (HemoPI / DBAASP) the
  user downloads; same loader, ``sequence,label`` columns.

Use ``ESM2Oracle.train_and_eval(...)`` to fit + report test metrics, then
``oracle.predict_proba(seqs)`` to score generated peptides.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """ACC / MCC / AUC / sensitivity / specificity (mirrors the paper's Table 4)."""
    from sklearn.metrics import roc_auc_score, matthews_corrcoef

    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(len(y_true), 1)
    sn = tp / max(tp + fn, 1)
    sp = tn / max(tn + fp, 1)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        mcc = float("nan")
    return {"ACC": acc, "MCC": mcc, "AUC": auc, "Sn": sn, "Sp": sp,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn}


class ESM2Oracle:
    """Frozen-ESM-2 + LogisticRegression sequence classifier."""

    def __init__(
        self,
        model_name: str = "esm2_t12_35M_UR50D",
        model_revision: Optional[str] = None,
        cache_dir: str = "results/esm_cache",
        device=None,
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.cache_dir = cache_dir
        self.device = device
        self.C = C
        self.random_state = random_state
        self._embedder = None
        self._cache = None
        self.clf = None
        self.threshold = 0.5

    # ------------------------------------------------------------------ #
    def _ensure_embedder(self):
        if self._embedder is None:
            from ..models.esm2_hf import ESM2HF, ESM2EmbeddingCache

            self._embedder = ESM2HF(
                model_name=self.model_name, model_revision=self.model_revision,
                device=self.device, freeze=True,
            )
            self._cache = ESM2EmbeddingCache(self._embedder, self.cache_dir)

    def embed(self, sequences: Sequence[str], batch_size: int = 32) -> np.ndarray:
        self._ensure_embedder()
        return self._cache.embed_many(list(sequences), batch_size=batch_size)

    # ------------------------------------------------------------------ #
    def train_and_eval(
        self,
        train_seqs: Sequence[str], train_labels: Sequence[int],
        val_seqs: Optional[Sequence[str]] = None, val_labels: Optional[Sequence[int]] = None,
        test_seqs: Optional[Sequence[str]] = None, test_labels: Optional[Sequence[int]] = None,
        batch_size: int = 32,
    ) -> Dict:
        """Fit the LR head on ESM-2 embeddings and report split metrics."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        logger.info(f"Embedding {len(train_seqs)} train sequences with {self.model_name} ...")
        Xtr = self.embed(train_seqs, batch_size)
        ytr = np.asarray(train_labels).astype(int)

        self.clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=self.C,
                max_iter=2000,
                class_weight="balanced",
                random_state=self.random_state,
            ),
        )
        self.clf.fit(Xtr, ytr)

        report = {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "classifier": {
                "type": "LogisticRegression",
                "C": self.C,
                "max_iter": 2000,
                "class_weight": "balanced",
                "random_state": self.random_state,
                "decision_threshold": self.threshold,
            },
            "n_train": len(train_seqs),
        }
        report["train"] = classification_metrics(ytr, self.clf.predict_proba(Xtr)[:, 1])

        if val_seqs is not None and val_labels is not None:
            Xv = self.embed(val_seqs, batch_size)
            report["val"] = classification_metrics(
                np.asarray(val_labels).astype(int), self.clf.predict_proba(Xv)[:, 1]
            )
        if test_seqs is not None and test_labels is not None:
            Xte = self.embed(test_seqs, batch_size)
            report["test"] = classification_metrics(
                np.asarray(test_labels).astype(int), self.clf.predict_proba(Xte)[:, 1]
            )
        return report

    # ------------------------------------------------------------------ #
    def predict_proba(self, sequences: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Oracle not trained/loaded.")
        seqs = [s for s in sequences if isinstance(s, str) and len(s) >= 5]
        X = self.embed(seqs, batch_size)
        return self.clf.predict_proba(X)[:, 1]

    def score_generated(self, sequences: Sequence[str], threshold: float = 0.5, batch_size: int = 32) -> Dict:
        """Summary used in the generated-corpus evaluation tables."""
        p = self.predict_proba(sequences, batch_size)
        return {
            "n": int(p.size),
            "mean_prob": float(p.mean()),
            "std_prob": float(p.std()),
            "positive_rate": float((p >= threshold).mean()),
            "_per_sequence": p.tolist(),
        }

    # ------------------------------------------------------------------ #
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"clf": self.clf, "model_name": self.model_name,
                         "model_revision": self.model_revision,
                         "threshold": self.threshold, "C": self.C,
                         "random_state": self.random_state}, f)
        logger.info(f"Saved oracle -> {path}")

    @classmethod
    def load(cls, path: str, cache_dir: str = "results/esm_cache", device=None) -> "ESM2Oracle":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        o = cls(
            model_name=blob["model_name"], model_revision=blob.get("model_revision"),
            cache_dir=cache_dir, device=device, C=blob.get("C", 1.0),
            random_state=blob.get("random_state", 42),
        )
        o.clf = blob["clf"]
        o.threshold = blob.get("threshold", 0.5)
        return o
