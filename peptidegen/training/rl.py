"""
Self-Critical Sequence Training (SCST) with multi-objective reward.

Reward (per generated sequence, all terms in [0, 1], higher = better):

    R = w_stab * ii_screen + w_amp * amp_proxy + w_hemo * (1 - hemolysis)

    ii_screen  = sigmoid-band reward on Instability Index (II); it is not a
                 physical-stability measurement
    amp_proxy  = TRAINING oracle only — NOT used for evaluation reporting
    hemolysis  = TRAINING oracle only — NOT used for evaluation reporting

SCST loss:  L = -(R(sample) - R(greedy)) * Σ_t log P(y^s_t)

EVALUATION INTEGRITY WARNING
    The oracle used here as a training reward signal MUST NOT be reused to
    report "AMP rate" in results — that constitutes circular evaluation
    (Goodhart's Law). Physicochemical rules may be reported as a descriptive
    pre-screen, but they are not an independent AMP activity predictor.

    Literature basis:
        Brown et al. 2019, J. Chem. Inf. Model. (GuacaMol)
        Szymczak et al. 2023, Nature Commun. (HydrAMP)
        Krakovna et al. 2020, DeepMind (Specification Gaming)

References
----------
Rennie et al. (2017) CVPR — SCST method
Hancock & Sahl (2006) Nature Biotechnol. — AMP criteria
Haney et al.  (2019) Front. Chem.       — validated thresholds
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

logger = logging.getLogger(__name__)


class MultiObjectiveReward:
    """
    Training reward: empirical II screen + AMP proxy + low-hemolysis proxy.

    Args:
        amp_oracle:           External ML classifier with .predict_proba(seqs).
                              Labelled as `oracle_id` in logs. May be None (uses
                              physicochemical fallback from metrics.py).
        hemo_oracle:          External ML hemolysis classifier (optional).
        oracle_id:            Human-readable identifier for the AMP oracle being
                              used (e.g. "ESM2Oracle", "iAMPpred", "heuristic").
                              Stored so downstream code can check independence.
        w_stability/amp/hemo: Reward weights (must sum to ~1).
        target_ii:            Upper bound on instability index (II) for band reward.
        ii_floor:             Lower bound on II (prevents gamed negative II).
        ii_scale:             Sigmoid scale for the band gates.
        use_heuristic_fallback: Permit physicochemical proxies when an oracle
                              is absent. Intended only for smoke tests because
                              these proxies do not constitute activity or
                              hemolysis validation.
    """

    def __init__(
        self,
        amp_oracle=None,
        hemo_oracle=None,
        oracle_id: str = 'heuristic',
        w_stability: float = 0.34,
        w_amp: float = 0.33,
        w_hemolysis: float = 0.33,
        target_ii: float = 40.0,
        ii_floor: float = 0.0,
        ii_scale: float = 12.0,
        use_heuristic_fallback: bool = False,
    ):
        self.amp_oracle = amp_oracle
        self.hemo_oracle = hemo_oracle
        self.oracle_id = oracle_id

        self.w_stab = w_stability
        self.w_amp = w_amp
        self.w_hemo = w_hemolysis
        self.target_ii = target_ii
        self.ii_floor = ii_floor
        self.ii_scale = ii_scale
        self.use_heuristic = use_heuristic_fallback

        # Warn early if an ML oracle is used (so caller knows to use independent eval)
        if amp_oracle is not None:
            logger.warning(
                "SCST training reward uses ML oracle '%s' for AMP scoring. "
                "Do NOT report 'AMP rate' using this same oracle; use a genuinely "
                "external predictor or experimental assay instead.",
                oracle_id,
            )
        else:
            logger.info(
                "SCST training reward: no external AMP oracle provided. "
                "Using physicochemical heuristic fallback (oracle_id='%s'). "
                "This mode is suitable for smoke tests only and must not be "
                "presented as independent activity validation.",
                oracle_id,
            )

    def _stability(self, seqs: List[str]) -> List[float]:
        """
        Band reward for instability index (II).

        Rewards sequences where  ii_floor <= II <= target_ii.
        The product of two sigmoid gates prevents the policy from gaming
        the reward by driving II to unphysically negative values.
        """
        from ..evaluation.stability import calculate_instability_index

        def _sig(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x / self.ii_scale))

        out = []
        for s in seqs:
            if len(s) < 2:
                out.append(0.0)
                continue
            ii = calculate_instability_index(s)
            # gate_low:  reward when II > ii_floor  (physically plausible)
            # gate_high: reward when II < the empirical screen threshold
            r = _sig(ii - self.ii_floor) * _sig(self.target_ii - ii)
            out.append(r)
        return out

    def _amp_proxy(self, seqs: List[str]) -> List[float]:
        """
        AMP likelihood reward — TRAINING SIGNAL ONLY, not for reporting.

        Priority:
          1. External ML oracle (amp_oracle.predict_proba)
          2. Physicochemical heuristic (estimate_amp_probability from metrics.py)
          3. Zero (if use_heuristic=False and no oracle)
        """
        if self.amp_oracle is not None:
            p = [0.0] * len(seqs)
            valid = [(i, s) for i, s in enumerate(seqs) if len(s) >= 5]
            if valid:
                try:
                    probs = self.amp_oracle.predict_proba([s for _, s in valid])
                    for (i, _), pr in zip(valid, probs):
                        p[i] = float(pr)
                except Exception as exc:
                    logger.warning("AMP oracle prediction failed: %s.", exc)
                    if self.use_heuristic:
                        logger.warning("Using explicitly enabled AMP heuristic fallback.")
                        from ..evaluation.metrics import estimate_amp_probability
                        p = [estimate_amp_probability(s) for s in seqs]
            return p

        if self.use_heuristic:
            from ..evaluation.metrics import estimate_amp_probability
            return [estimate_amp_probability(s) for s in seqs]

        return [0.0] * len(seqs)

    def _hemolysis_proxy(self, seqs: List[str]) -> List[float]:
        """
        Hemolysis likelihood — TRAINING SIGNAL ONLY, not for reporting.
        Returns fraction in [0, 1]; 1 = predicted hemolytic.
        """
        if self.hemo_oracle is not None:
            h = [0.0] * len(seqs)
            valid = [(i, s) for i, s in enumerate(seqs) if len(s) >= 5]
            if valid:
                try:
                    probs = self.hemo_oracle.predict_proba([s for _, s in valid])
                    for (i, _), pr in zip(valid, probs):
                        h[i] = float(pr)
                except Exception as exc:
                    logger.warning("Hemolysis oracle prediction failed: %s.", exc)
                    if self.use_heuristic:
                        logger.warning("Using explicitly enabled hemolysis heuristic fallback.")
                        from ..evaluation.metrics import calculate_hemolytic_score
                        h = [calculate_hemolytic_score(s)[0] / 10.0 for s in seqs]
            return h

        if self.use_heuristic:
            from ..evaluation.metrics import calculate_hemolytic_score
            return [calculate_hemolytic_score(s)[0] / 10.0 for s in seqs]

        return [0.0] * len(seqs)

    def __call__(self, seqs: List[str]) -> torch.Tensor:
        """
        Compute per-sequence reward tensor.

        R[i] = w_stab * stability[i]
              + w_amp  * amp_proxy[i]
              + w_hemo * (1 - hemolysis_proxy[i])
        """
        stab = self._stability(seqs)
        amp = self._amp_proxy(seqs)
        hemo = self._hemolysis_proxy(seqs)

        r = [
            self.w_stab * s + self.w_amp * a + self.w_hemo * (1.0 - h)
            for s, a, h in zip(stab, amp, hemo)
        ]
        return torch.tensor(r, dtype=torch.float)

    def describe(self) -> Dict[str, object]:
        """Return a serialisable description for logging / checkpointing."""
        return {
            'oracle_id': self.oracle_id,
            'has_external_amp_oracle': self.amp_oracle is not None,
            'has_external_hemo_oracle': self.hemo_oracle is not None,
            'w_stability': self.w_stab,
            'w_amp': self.w_amp,
            'w_hemolysis': self.w_hemo,
            'target_ii': self.target_ii,
            'ii_floor': self.ii_floor,
        }


class SCSTTrainer:
    """
    Self-Critical Sequence Training fine-tuner.

    Logs `training_oracle_id` at startup so downstream evaluation code can
    verify it is using an independent evaluator.

    Args:
        generator:    Generator model with `.rl_rollout()` method.
        reward_fn:    MultiObjectiveReward instance.
        vocab:        Vocabulary with `.batch_decode()`.
        device:       Torch device (auto-detected if None).
        lr:           Learning rate (default 1e-5).
        entropy_bonus: Coefficient for entropy regularisation to prevent
                      mode collapse during RL fine-tuning (default 0).
    """

    def __init__(
        self,
        generator: nn.Module,
        reward_fn: MultiObjectiveReward,
        vocab,
        device: Optional[torch.device] = None,
        lr: float = 1e-5,
        entropy_bonus: float = 0.0,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.G = generator.to(self.device)
        self.reward_fn = reward_fn
        self.vocab = vocab
        self.opt = AdamW(self.G.parameters(), lr=lr, weight_decay=1e-4)
        self.entropy_bonus = entropy_bonus

        logger.info(
            "SCSTTrainer initialised | oracle_id='%s' | "
            "Do not reuse this reward oracle as an independent reporting metric.",
            reward_fn.oracle_id,
        )

    def _decode(self, tokens: torch.Tensor) -> List[str]:
        return self.vocab.batch_decode(tokens, remove_special_tokens=True)

    def train_step(
        self,
        batch_size: int,
        conditions: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        One SCST gradient update.

        Returns metrics dict including:
            scst_loss, reward_sample, reward_greedy, advantage, entropy
        """
        self.G.train()
        z = torch.randn(batch_size, self.G.latent_dim, device=self.device)
        if conditions is not None:
            conditions = conditions.to(self.device)

        # Sampled rollout — retains log-probs for gradient
        sample_tok, logp, entropy = self.G.rl_rollout(z, conditions, greedy=False)

        # Greedy baseline — no gradient
        with torch.no_grad():
            greedy_tok, _, _ = self.G.rl_rollout(z, conditions, greedy=True)

        r_sample = self.reward_fn(self._decode(sample_tok)).to(self.device)
        r_greedy = self.reward_fn(self._decode(greedy_tok)).to(self.device)
        advantage = r_sample - r_greedy     # (B,)

        # SCST policy-gradient loss
        # Entropy bonus counteracts mode collapse from reward saturation.
        loss = -(advantage.detach() * logp).mean()
        if self.entropy_bonus > 0:
            loss = loss - self.entropy_bonus * entropy.mean()

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
        self.opt.step()

        return {
            'scst_loss':     float(loss.item()),
            'reward_sample': float(r_sample.mean().item()),
            'reward_greedy': float(r_greedy.mean().item()),
            'advantage':     float(advantage.mean().item()),
            'entropy':       float(entropy.mean().item()),
        }
