"""
Independent AMP Evaluation Suite.

Provides evaluation metrics that are INDEPENDENT of the training oracle.
This is critical for reporting unbiased results: if the SCST reward uses
an ML-based AMP predictor, the same predictor MUST NOT be used to report
"AMP rate" in the paper — that would be circular (Goodhart's Law).

Design principle (see references below):
    Training oracle   → optimises the generator toward some proxy
    Independent eval  → measures whether the generator actually produced AMPs

References
----------
Hancock & Sahl (2006). Nature Biotechnol.  — physicochemical AMP criteria
Haney et al.  (2019). Front. Chem.         — validated thresholds
Fjell et al.  (2012). Nature Rev. Drug Discov. — amphipathicity / charge rules
Brown et al.  (2019). J. Chem. Inf. Model. — GuacaMol: independent evaluation
Szymczak et al. (2023). Nature Commun.     — HydrAMP uses AMP Scanner v2 (independent)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physicochemical constants (Hancock/Haney criteria)
# ---------------------------------------------------------------------------

# Eisenberg (1984) consensus hydrophobicity scale
_HYDRO = {
    'A': 0.620, 'R': -2.530, 'N': -0.780, 'D': -0.900, 'C': 0.290,
    'Q': -0.850, 'E': -0.740, 'G': 0.480,  'H': -0.400, 'I': 1.380,
    'L': 1.060,  'K': -1.500, 'M': 0.640,  'F': 1.190,  'P': 0.120,
    'S': -0.180, 'T': -0.050, 'W': 0.810,  'Y': 0.260,  'V': 1.080,
}

# pKa values for charge calculation
_PKA = {
    'K': 10.5, 'R': 12.5, 'H': 6.0,   # basic
    'D': 3.9,  'E': 4.1,               # acidic
    'C': 8.3,  'Y': 10.1,              # weakly acidic
}

_HELIX_ANGLE = 100.0 * math.pi / 180.0   # alpha-helix angular periodicity

_HYDROPHOBIC_AA = frozenset('AVILMFWY')


# ---------------------------------------------------------------------------
# Low-level helpers (no ML, no imports beyond stdlib + math)
# ---------------------------------------------------------------------------

def _net_charge(seq: str, pH: float = 7.4) -> float:
    """Net charge at physiological pH (Henderson-Hasselbalch)."""
    seq = seq.upper()
    charge = (
        1.0 / (1.0 + 10 ** (pH - 9.7))    # N-terminus pKa ~9.7
        - 1.0 / (1.0 + 10 ** (2.3 - pH))  # C-terminus pKa ~2.3
    )
    for aa in seq:
        if aa in ('K', 'R', 'H'):
            charge += 1.0 / (1.0 + 10 ** (pH - _PKA[aa]))
        elif aa in ('D', 'E', 'C', 'Y'):
            charge -= 1.0 / (1.0 + 10 ** (_PKA[aa] - pH))
    return charge


def _hydrophobic_ratio(seq: str) -> float:
    """Fraction of hydrophobic residues (AVILMFWY)."""
    if not seq:
        return 0.0
    return sum(1 for aa in seq.upper() if aa in _HYDROPHOBIC_AA) / len(seq)


def _hydrophobic_moment(seq: str, angle: float = _HELIX_ANGLE, window: int = 11) -> float:
    """
    Maximum hydrophobic moment (Eisenberg et al. 1982) over sliding window.
    Higher values indicate greater amphipathicity — a key AMP property.
    """
    seq = seq.upper()
    n = len(seq)
    if n < 3:
        return 0.0
    win = min(window, n)
    best = 0.0
    for i in range(n - win + 1):
        s = seq[i: i + win]
        sx = sum(_HYDRO.get(aa, 0) * math.sin((j + 1) * angle) for j, aa in enumerate(s))
        cx = sum(_HYDRO.get(aa, 0) * math.cos((j + 1) * angle) for j, aa in enumerate(s))
        best = max(best, math.sqrt(sx ** 2 + cx ** 2) / win)
    return best


def _gravy(seq: str) -> float:
    """Grand Average of Hydropathicity (Kyte & Doolittle 1982)."""
    seq = seq.upper()
    if not seq:
        return 0.0
    return sum(_HYDRO.get(aa, 0.0) for aa in seq) / len(seq)


# ---------------------------------------------------------------------------
# PhysicochemAMPScorer — fully independent, rule-based
# ---------------------------------------------------------------------------

class PhysicochemAMPScorer:
    """
    Rule-based AMP likelihood scorer derived from established physicochemical
    criteria (Hancock & Sahl 2006; Haney et al. 2019; Fjell et al. 2012).

    **No machine learning is used** — this scorer is entirely derived from
    physicochemical theory and literature-validated thresholds, making it
    a truly independent evaluation tool regardless of which ML oracle is
    used during SCST training.

    Scoring criteria (each contributes 0–1 to a 5-component score):
        1. Net charge at pH 7.4: optimal +2 to +9 (cationic AMPs)
        2. Hydrophobic ratio: optimal 30–60 %
        3. Hydrophobic moment: ≥ 0.25 (amphipathic)
        4. GRAVY (hydropathicity): optimal -1.5 to +0.5
        5. Length: optimal 10–50 AA

    Args:
        charge_range:     (min, max) net charge for full score (default: 2, 9)
        hydro_range:      (min, max) hydrophobic ratio (default: 0.3, 0.6)
        moment_threshold: minimum hydrophobic moment (default: 0.25)
        gravy_range:      (min, max) GRAVY (default: -1.5, 0.5)
        length_range:     (min, max) sequence length (default: 10, 50)
        weights:          (charge, hydro, moment, gravy, length) — must sum to 1
        pH:               pH for charge calculation (default: 7.4)
    """

    def __init__(
        self,
        charge_range: Tuple[float, float] = (2.0, 9.0),
        hydro_range: Tuple[float, float] = (0.30, 0.60),
        moment_threshold: float = 0.25,
        gravy_range: Tuple[float, float] = (-1.5, 0.5),
        length_range: Tuple[int, int] = (10, 50),
        weights: Tuple[float, ...] = (0.30, 0.25, 0.20, 0.15, 0.10),
        pH: float = 7.4,
        amp_threshold: float = 0.5,
    ):
        assert abs(sum(weights) - 1.0) < 1e-6, "weights must sum to 1"
        self.charge_range = charge_range
        self.hydro_range = hydro_range
        self.moment_threshold = moment_threshold
        self.gravy_range = gravy_range
        self.length_range = length_range
        self.w = weights
        self.pH = pH
        self.amp_threshold = amp_threshold

    def _score_one(self, seq: str) -> Dict[str, float]:
        """Return per-criterion scores (0–1) and the composite AMP score."""
        seq = seq.upper().strip()
        n = len(seq)

        # --- 1. Net charge -----------------------------------------------
        charge = _net_charge(seq, self.pH)
        cmin, cmax = self.charge_range
        if cmin <= charge <= cmax:
            s_charge = 1.0
        elif charge < cmin:
            s_charge = max(0.0, 1.0 - (cmin - charge) / cmin)
        else:
            s_charge = max(0.0, 1.0 - (charge - cmax) / cmax)

        # --- 2. Hydrophobic ratio -----------------------------------------
        hr = _hydrophobic_ratio(seq)
        rmin, rmax = self.hydro_range
        if rmin <= hr <= rmax:
            s_hydro = 1.0
        elif hr < rmin:
            s_hydro = hr / rmin if rmin > 0 else 0.0
        else:
            s_hydro = max(0.0, 1.0 - (hr - rmax) / (1.0 - rmax + 1e-6))

        # --- 3. Amphipathicity (hydrophobic moment) -----------------------
        hm = _hydrophobic_moment(seq)
        if hm >= self.moment_threshold:
            s_moment = min(1.0, hm / (self.moment_threshold * 2))
        else:
            s_moment = hm / self.moment_threshold

        # --- 4. GRAVY (hydropathicity) ------------------------------------
        gv = _gravy(seq)
        gmin, gmax = self.gravy_range
        if gmin <= gv <= gmax:
            s_gravy = 1.0
        elif gv < gmin:
            s_gravy = max(0.0, 1.0 + (gv - gmin))
        else:
            s_gravy = max(0.0, 1.0 - (gv - gmax))

        # --- 5. Length ----------------------------------------------------
        lmin, lmax = self.length_range
        if lmin <= n <= lmax:
            s_len = 1.0
        elif n < lmin:
            s_len = n / lmin
        else:
            s_len = max(0.0, 1.0 - (n - lmax) / lmax)

        composite = (
            self.w[0] * s_charge
            + self.w[1] * s_hydro
            + self.w[2] * s_moment
            + self.w[3] * s_gravy
            + self.w[4] * s_len
        )

        return {
            'physico_amp_score': composite,
            'is_amp_physico': composite >= self.amp_threshold,
            'net_charge': charge,
            'hydrophobic_ratio': hr,
            'hydrophobic_moment': hm,
            'gravy': gv,
            'length': n,
            # Per-criterion breakdown (useful for debugging / ablation)
            '_s_charge': s_charge,
            '_s_hydro': s_hydro,
            '_s_moment': s_moment,
            '_s_gravy': s_gravy,
            '_s_len': s_len,
        }

    def score(self, seq: str) -> Dict[str, float]:
        """Score a single sequence."""
        if not seq or len(seq) < 5:
            return {
                'physico_amp_score': 0.0, 'is_amp_physico': False,
                'net_charge': 0.0, 'hydrophobic_ratio': 0.0,
                'hydrophobic_moment': 0.0, 'gravy': 0.0, 'length': len(seq),
            }
        return self._score_one(seq)

    def score_batch(self, sequences: List[str]) -> Dict[str, object]:
        """
        Score a batch of sequences and return aggregate statistics.

        Returns a dict with:
            per_sequence    : list of per-seq score dicts
            amp_rate_physico: fraction with physico_amp_score >= threshold
            mean_score      : mean composite score
            mean_charge     : mean net charge at pH 7.4
            mean_hydro_ratio: mean hydrophobic ratio
            mean_moment     : mean hydrophobic moment
        """
        if not sequences:
            return {'amp_rate_physico': 0.0, 'mean_score': 0.0, 'per_sequence': []}

        per_seq = [self.score(s) for s in sequences]
        n = len(per_seq)

        def _mean(key):
            vals = [d[key] for d in per_seq if isinstance(d.get(key), (int, float))]
            return sum(vals) / len(vals) if vals else 0.0

        return {
            'amp_rate_physico': sum(d['is_amp_physico'] for d in per_seq) / n,
            'mean_score':       _mean('physico_amp_score'),
            'mean_charge':      _mean('net_charge'),
            'mean_hydro_ratio': _mean('hydrophobic_ratio'),
            'mean_moment':      _mean('hydrophobic_moment'),
            'mean_gravy':       _mean('gravy'),
            'per_sequence':     per_seq,
        }


# ---------------------------------------------------------------------------
# IndependentEvalSuite — wraps PhysicochemAMPScorer + optional external tools
# ---------------------------------------------------------------------------

class IndependentEvalSuite:
    """
    Composite independent evaluator for generated peptides.

    Always runs:
        - PhysicochemAMPScorer   (fully rule-based, no ML)

    Optionally runs (if installed / credentials provided):
        - AMPlify (Li et al. 2022) via amplify package or REST API
          Install: pip install amplify-amp   (if available)

    Args:
        training_oracle_id: identifier string of the oracle used during SCST
                            training — used to warn if it matches the eval oracle.
                            Pass None if no oracle was used during training.
        use_amplify:        whether to try importing the AMPlify package
        amp_threshold:      physicochemical AMP probability threshold (default 0.5)
    """

    KNOWN_TRAINING_ORACLES = {'esm2oracle', 'esm2_oracle', 'esm2', 'esm-2'}

    def __init__(
        self,
        training_oracle_id: Optional[str] = None,
        use_amplify: bool = False,
        amp_threshold: float = 0.5,
    ):
        self.training_oracle_id = (training_oracle_id or '').lower()
        self.physico = PhysicochemAMPScorer(amp_threshold=amp_threshold)
        self.amplify = None

        # Warn about circular evaluation
        if self.training_oracle_id in self.KNOWN_TRAINING_ORACLES:
            logger.warning(
                "Training oracle is '%s' (ML-based). "
                "Reporting AMP rate using the same oracle would be circular "
                "(Goodhart's Law). Use IndependentEvalSuite for reporting. "
                "See: Brown et al. 2019 (GuacaMol), Szymczak et al. 2023 (HydrAMP).",
                training_oracle_id,
            )

        if use_amplify:
            try:
                import amplify as _amp_pkg  # type: ignore
                self.amplify = _amp_pkg
                logger.info("AMPlify loaded for independent evaluation.")
            except ImportError:
                logger.warning(
                    "AMPlify not installed (pip install amplify-amp). "
                    "Falling back to physicochemical scoring only."
                )

    def evaluate(self, sequences: List[str]) -> Dict[str, object]:
        """
        Run independent evaluation on a list of sequences.

        Returns dict with:
            physico: PhysicochemAMPScorer aggregate output
            amplify: AMPlify output (if available)
            summary: combined high-level metrics for reporting
        """
        valid = [s for s in sequences if s and len(s) >= 5]

        physico_result = self.physico.score_batch(valid)
        result: Dict[str, object] = {'physico': physico_result}

        if self.amplify is not None:
            try:
                amp_scores = self.amplify.predict(valid)   # assumed interface
                amp_rate = sum(1 for s in amp_scores if s >= 0.5) / len(valid) if valid else 0.0
                result['amplify'] = {
                    'amp_rate_amplify': amp_rate,
                    'mean_score': sum(amp_scores) / len(amp_scores) if amp_scores else 0.0,
                    'per_sequence': list(amp_scores),
                }
            except Exception as e:
                logger.warning(f"AMPlify prediction failed: {e}")

        # Compact summary for reporting
        result['summary'] = {
            'n_evaluated':       len(valid),
            # Primary independent metric: physicochemical AMP rate
            'amp_rate_physico':  physico_result['amp_rate_physico'],
            'mean_charge':       physico_result['mean_charge'],
            'mean_hydro_moment': physico_result['mean_moment'],
            # Secondary (if available)
            'amp_rate_amplify':  result.get('amplify', {}).get('amp_rate_amplify'),
        }
        return result


# ---------------------------------------------------------------------------
# Convenience: oracle independence check
# ---------------------------------------------------------------------------

def check_oracle_independence(training_oracle_id: str, eval_oracle_id: str) -> bool:
    """
    Return True if the two oracles are independent (different).
    Logs a warning if they appear to be the same model.

    Args:
        training_oracle_id: e.g. 'ESM2Oracle', 'heuristic'
        eval_oracle_id:     e.g. 'PhysicochemAMPScorer', 'AMPlify'
    """
    t = training_oracle_id.lower().replace('-', '').replace('_', '')
    e = eval_oracle_id.lower().replace('-', '').replace('_', '')
    if t == e or (t in e) or (e in t):
        logger.error(
            "Circular evaluation detected: training oracle '%s' appears to be "
            "the same as evaluation oracle '%s'. "
            "This inflates reported AMP rates (Goodhart's Law). "
            "Use IndependentEvalSuite with PhysicochemAMPScorer instead.",
            training_oracle_id, eval_oracle_id,
        )
        return False
    return True
