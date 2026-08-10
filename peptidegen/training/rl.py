"""
A4 — Self-Critical Sequence Training (SCST) with a balanced multi-objective
reward (paper Eq.10, extended).

Closes two gaps the original code/paper left open:
  * SCST reward shaping was claimed (Eq.10 + an ablation row) but never coded.
  * The reward jointly targets stability **and** antimicrobial activity **and**
    low hemolysis, directly addressing the two weak spots in the results
    (AMP-prob below baselines, ~86% predicted hemolytic).

Reward (per generated sequence, all terms in [0,1], higher is better):

    R = w_stab * stability + w_amp * amp_prob + w_hemo * (1 - hemolysis)

    stability = sigmoid((II_target - II) / scale)   # rewards II below target
    amp_prob  = ESM-2 AMP oracle  (or heuristic fallback)
    hemolysis = ESM-2 hemolysis oracle (or heuristic fallback, scaled to [0,1])

SCST loss:  L = - (R(sample) - R(greedy)) * sum_t log P(y^s_t)
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
    """Balanced stability + AMP + low-hemolysis reward over decoded peptides."""

    def __init__(
        self,
        amp_oracle=None,
        hemo_oracle=None,
        w_stability: float = 0.34,
        w_amp: float = 0.33,
        w_hemolysis: float = 0.33,
        target_ii: float = 40.0,
        ii_floor: float = 0.0,
        ii_scale: float = 12.0,
        use_heuristic_fallback: bool = True,
    ):
        self.amp_oracle = amp_oracle
        self.hemo_oracle = hemo_oracle
        self.w_stab = w_stability
        self.w_amp = w_amp
        self.w_hemo = w_hemolysis
        self.target_ii = target_ii
        self.ii_floor = ii_floor
        self.ii_scale = ii_scale
        self.use_heuristic = use_heuristic_fallback

    def _stability(self, seqs: List[str]) -> List[float]:
        from ..evaluation.stability import calculate_instability_index

        def _sig(x):
            return 1.0 / (1.0 + math.exp(-x / self.ii_scale))

        out = []
        for s in seqs:
            if len(s) < 2:
                out.append(0.0); continue
            ii = calculate_instability_index(s)
            # BAND reward: high only when ii_floor <= II <= target_ii.
            # The previous monotonic reward (sigmoid(target - II)) had no lower
            # bound, so SCST gamed it by pushing II to absurd negative values
            # (mean II ~= -5). Multiplying a low-end gate (II above floor) with a
            # high-end gate (II below target) removes that exploit: rewards a
            # physically plausible stable band and penalises unphysically low II.
            r = _sig(ii - self.ii_floor) * _sig(self.target_ii - ii)
            out.append(r)
        return out

    def _amp(self, seqs: List[str]) -> List[float]:
        if self.amp_oracle is not None:
            p = [0.0] * len(seqs)
            valid = [(i, s) for i, s in enumerate(seqs) if len(s) >= 5]
            if valid:
                probs = self.amp_oracle.predict_proba([s for _, s in valid])
                for (i, _), pr in zip(valid, probs):
                    p[i] = float(pr)
            return p
        if self.use_heuristic:
            from ..evaluation.metrics import estimate_amp_probability
            return [estimate_amp_probability(s) for s in seqs]
        return [0.0] * len(seqs)

    def _hemolysis(self, seqs: List[str]) -> List[float]:
        if self.hemo_oracle is not None:
            h = [0.0] * len(seqs)
            valid = [(i, s) for i, s in enumerate(seqs) if len(s) >= 5]
            if valid:
                probs = self.hemo_oracle.predict_proba([s for _, s in valid])
                for (i, _), pr in zip(valid, probs):
                    h[i] = float(pr)
            return h
        if self.use_heuristic:
            from ..evaluation.metrics import calculate_hemolytic_score
            # heuristic returns 0-10; scale to [0,1]
            return [calculate_hemolytic_score(s)[0] / 10.0 for s in seqs]
        return [0.0] * len(seqs)

    def __call__(self, seqs: List[str]) -> torch.Tensor:
        stab = self._stability(seqs)
        amp = self._amp(seqs)
        hemo = self._hemolysis(seqs)
        r = [self.w_stab * s + self.w_amp * a + self.w_hemo * (1.0 - h)
             for s, a, h in zip(stab, amp, hemo)]
        return torch.tensor(r, dtype=torch.float)


class SCSTTrainer:
    """Self-critical sequence training fine-tuner for the fusion generator."""

    def __init__(
        self,
        generator: nn.Module,
        reward_fn: MultiObjectiveReward,
        vocab,
        device: Optional[torch.device] = None,
        lr: float = 1e-5,
        entropy_bonus: float = 0.0,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = generator.to(self.device)
        self.reward_fn = reward_fn
        self.vocab = vocab
        self.opt = AdamW(self.G.parameters(), lr=lr, weight_decay=1e-4)
        self.entropy_bonus = entropy_bonus

    def _decode(self, tokens: torch.Tensor) -> List[str]:
        return self.vocab.batch_decode(tokens, remove_special_tokens=True)

    def train_step(self, batch_size: int, conditions: Optional[torch.Tensor] = None) -> Dict[str, float]:
        self.G.train()
        z = torch.randn(batch_size, self.G.latent_dim, device=self.device)
        if conditions is not None:
            conditions = conditions.to(self.device)

        # sampled rollout (keeps grad through log-probs)
        sample_tok, logp, entropy = self.G.rl_rollout(z, conditions, greedy=False)
        # greedy baseline (no grad)
        with torch.no_grad():
            greedy_tok, _, _ = self.G.rl_rollout(z, conditions, greedy=True)

        r_sample = self.reward_fn(self._decode(sample_tok)).to(self.device)
        r_greedy = self.reward_fn(self._decode(greedy_tok)).to(self.device)
        advantage = r_sample - r_greedy                           # (B,)

        # SCST loss; subtracting the entropy term *maximizes* policy entropy,
        # which counteracts the mode-collapse / low-diversity that pure reward
        # maximization causes (the reward saturates and the policy peaks).
        loss = -(advantage.detach() * logp).mean()
        if self.entropy_bonus > 0:
            loss = loss - self.entropy_bonus * entropy.mean()

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
        self.opt.step()

        return {
            "scst_loss": float(loss.item()),
            "reward_sample": float(r_sample.mean().item()),
            "reward_greedy": float(r_greedy.mean().item()),
            "advantage": float(advantage.mean().item()),
            "entropy": float(entropy.mean().item()),
        }
