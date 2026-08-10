"""
Centralized constants for amino acid properties.

This module contains all amino acid related constants used across the project
to avoid code duplication and ensure consistency.
"""

from typing import Dict, Set, List

# Standard 20 amino acids
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Special tokens
SPECIAL_TOKENS = {
    'PAD': '<PAD>',
    'SOS': '<SOS>',
    'EOS': '<EOS>',
    'UNK': '<UNK>',
}

# Amino acid groups
AA_GROUPS: Dict[str, Set[str]] = {
    'aliphatic': set('AVILM'),      # Aliphatic (contribute to aliphatic index)
    'aromatic': set('FWY'),          # Aromatic amino acids
    'positive': set('KRH'),          # Positively charged
    'negative': set('DE'),           # Negatively charged
    'polar': set('STNQ'),            # Polar uncharged
    'hydrophobic': set('AVILMFWP'),  # Hydrophobic
    'small': set('AGST'),            # Small amino acids
    'tiny': set('AGS'),              # Tiny amino acids
    'proline': set('P'),             # Proline (helix breaker)
    'cysteine': set('C'),            # Cysteine (disulfide bonds)
}

# Kyte-Doolittle hydropathy scale
HYDROPATHY_SCALE: Dict[str, float] = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Molecular weights (Daltons)
MOLECULAR_WEIGHTS: Dict[str, float] = {
    'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
    'Q': 146.2, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
    'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
    'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
}

# pKa values
PKA_VALUES = {
    'C_TERM': 2.34,   # C-terminal carboxyl
    'N_TERM': 9.69,   # N-terminal amino
    'SIDE_CHAINS': {
        'D': 3.86, 'E': 4.25, 'H': 6.00, 'C': 8.33,
        'Y': 10.07, 'K': 10.53, 'R': 12.48
    }
}

# Chou-Fasman secondary structure propensities
HELIX_PROPENSITY: Dict[str, float] = {
    'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
    'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
    'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
    'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06
}

SHEET_PROPENSITY: Dict[str, float] = {
    'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
    'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
    'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
    'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70
}

# Instability Index dipeptide weights (Guruprasad et al., 1990)
# Complete table for all 400 dipeptides
INSTABILITY_WEIGHTS: Dict[str, float] = {
    'WW': 1.0, 'WC': 1.0, 'WM': 24.68, 'WH': 24.68, 'WY': 1.0, 
    'WF': 1.0, 'WQ': 1.0, 'WN': 13.34, 'WI': 1.0, 'WR': 1.0, 
    'WD': 1.0, 'WP': 1.0, 'WT': -14.03, 'WK': 1.0, 'WE': 1.0, 
    'WV': -7.49, 'WS': 1.0, 'WG': -9.37, 'WA': -14.03, 'WL': 13.34,
    'CW': 24.68, 'CC': 1.0, 'CM': 33.6, 'CH': 33.6, 'CY': 1.0, 
    'CF': 1.0, 'CQ': -6.54, 'CN': 1.0, 'CI': 1.0, 'CR': 1.0, 
    'CD': 20.26, 'CP': 20.26, 'CT': 33.6, 'CK': 1.0, 'CE': 1.0, 
    'CV': -6.54, 'CS': 1.0, 'CG': 1.0, 'CA': 1.0, 'CL': 20.26,
    'MW': 1.0, 'MC': 1.0, 'MM': -1.88, 'MH': 58.28, 'MY': 24.68, 
    'MF': 1.0, 'MQ': -6.54, 'MN': 1.0, 'MI': 1.0, 'MR': -6.54, 
    'MD': 1.0, 'MP': 44.94, 'MT': -1.88, 'MK': 1.0, 'ME': 1.0, 
    'MV': 1.0, 'MS': 44.94, 'MG': 1.0, 'MA': 13.34, 'ML': 1.0,
    'HW': -1.88, 'HC': 1.0, 'HM': 1.0, 'HH': 1.0, 'HY': 44.94, 
    'HF': -9.37, 'HQ': 1.0, 'HN': 24.68, 'HI': 44.94, 'HR': 1.0, 
    'HD': 1.0, 'HP': -1.88, 'HT': -6.54, 'HK': 24.68, 'HE': 1.0, 
    'HV': 1.0, 'HS': 1.0, 'HG': -9.37, 'HA': 1.0, 'HL': 1.0,
    'YW': -9.37, 'YC': 1.0, 'YM': 44.94, 'YH': 13.34, 'YY': 13.34, 
    'YF': 1.0, 'YQ': 1.0, 'YN': 1.0, 'YI': 1.0, 'YR': -15.91, 
    'YD': 24.68, 'YP': 13.34, 'YT': -7.49, 'YK': 1.0, 'YE': -6.54, 
    'YV': 1.0, 'YS': 1.0, 'YG': -7.49, 'YA': 24.68, 'YL': 1.0,
    'FW': 1.0, 'FC': 1.0, 'FM': 1.0, 'FH': 1.0, 'FY': 33.6, 
    'FF': 1.0, 'FQ': 1.0, 'FN': 1.0, 'FI': 1.0, 'FR': 1.0, 
    'FD': 13.34, 'FP': 20.26, 'FT': 1.0, 'FK': -14.03, 'FE': 1.0, 
    'FV': 1.0, 'FS': 1.0, 'FG': 1.0, 'FA': 1.0, 'FL': 1.0,
    'QW': 1.0, 'QC': -6.54, 'QM': 1.0, 'QH': 1.0, 'QY': -6.54, 
    'QF': -6.54, 'QQ': 20.26, 'QN': 1.0, 'QI': 1.0, 'QR': 1.0, 
    'QD': 20.26, 'QP': 20.26, 'QT': 1.0, 'QK': 1.0, 'QE': 20.26, 
    'QV': -6.54, 'QS': 44.94, 'QG': 1.0, 'QA': 1.0, 'QL': 1.0,
    'NW': -9.37, 'NC': -1.88, 'NM': 1.0, 'NH': 1.0, 'NY': 1.0, 
    'NF': -14.03, 'NQ': -6.54, 'NN': 1.0, 'NI': 44.94, 'NR': 1.0, 
    'ND': 1.0, 'NP': -1.88, 'NT': -7.49, 'NK': 24.68, 'NE': 1.0, 
    'NV': 1.0, 'NS': 1.0, 'NG': -14.03, 'NA': 1.0, 'NL': 1.0,
    'IW': 1.0, 'IC': 1.0, 'IM': 1.0, 'IH': 13.34, 'IY': 1.0, 
    'IF': 1.0, 'IQ': 1.0, 'IN': 1.0, 'II': 1.0, 'IR': 1.0, 
    'ID': 1.0, 'IP': -1.88, 'IT': 1.0, 'IK': -7.49, 'IE': 44.94, 
    'IV': -7.49, 'IS': 1.0, 'IG': 1.0, 'IA': 1.0, 'IL': 20.26,
    'RW': 58.28, 'RC': 1.0, 'RM': 1.0, 'RH': 20.26, 'RY': -6.54, 
    'RF': 1.0, 'RQ': 20.26, 'RN': 13.34, 'RI': 1.0, 'RR': 58.28, 
    'RD': 1.0, 'RP': 20.26, 'RT': 1.0, 'RK': 1.0, 'RE': 1.0, 
    'RV': 1.0, 'RS': 44.94, 'RG': -7.49, 'RA': 1.0, 'RL': 1.0,
    'DW': 1.0, 'DC': 1.0, 'DM': 1.0, 'DH': 1.0, 'DY': 1.0, 
    'DF': -6.54, 'DQ': 1.0, 'DN': 1.0, 'DI': 1.0, 'DR': -6.54, 
    'DD': 1.0, 'DP': 1.0, 'DT': -14.03, 'DK': -7.49, 'DE': 1.0, 
    'DV': 1.0, 'DS': 20.26, 'DG': 1.0, 'DA': 1.0, 'DL': 1.0,
    'PW': -1.88, 'PC': -6.54, 'PM': -6.54, 'PH': 1.0, 'PY': 1.0, 
    'PF': 20.26, 'PQ': 20.26, 'PN': 1.0, 'PI': 1.0, 'PR': -6.54, 
    'PD': -6.54, 'PP': 20.26, 'PT': 1.0, 'PK': 1.0, 'PE': 18.38, 
    'PV': 20.26, 'PS': 20.26, 'PG': 1.0, 'PA': 20.26, 'PL': 1.0,
    'TW': -14.03, 'TC': 1.0, 'TM': 1.0, 'TH': 1.0, 'TY': 1.0, 
    'TF': 13.34, 'TQ': -6.54, 'TN': -14.03, 'TI': 1.0, 'TR': 1.0, 
    'TD': 1.0, 'TP': 1.0, 'TT': 1.0, 'TK': 1.0, 'TE': 20.26, 
    'TV': 1.0, 'TS': 1.0, 'TG': -7.49, 'TA': 1.0, 'TL': 1.0,
    'KW': 1.0, 'KC': 1.0, 'KM': 33.6, 'KH': 1.0, 'KY': 1.0, 
    'KF': 1.0, 'KQ': 24.68, 'KN': 1.0, 'KI': -7.49, 'KR': 33.6, 
    'KD': 1.0, 'KP': -6.54, 'KT': 1.0, 'KK': 1.0, 'KE': 1.0, 
    'KV': -7.49, 'KS': 1.0, 'KG': -7.49, 'KA': 1.0, 'KL': -7.49,
    'EW': -14.03, 'EC': 44.94, 'EM': 1.0, 'EH': -6.54, 'EY': 1.0, 
    'EF': 1.0, 'EQ': 20.26, 'EN': 1.0, 'EI': 20.26, 'ER': 1.0, 
    'ED': 20.26, 'EP': 20.26, 'ET': 1.0, 'EK': 1.0, 'EE': 33.6, 
    'EV': 1.0, 'ES': 20.26, 'EG': 1.0, 'EA': 1.0, 'EL': 1.0,
    'VW': 1.0, 'VC': 1.0, 'VM': 1.0, 'VH': 1.0, 'VY': -6.54, 
    'VF': 1.0, 'VQ': 1.0, 'VN': 1.0, 'VI': 1.0, 'VR': 1.0, 
    'VD': -14.03, 'VP': 20.26, 'VT': -7.49, 'VK': -1.88, 'VE': 1.0, 
    'VV': 1.0, 'VS': 1.0, 'VG': -7.49, 'VA': 1.0, 'VL': 1.0,
    'SW': 1.0, 'SC': 33.6, 'SM': 1.0, 'SH': 1.0, 'SY': 1.0, 
    'SF': 1.0, 'SQ': 20.26, 'SN': 1.0, 'SI': 1.0, 'SR': 20.26, 
    'SD': 1.0, 'SP': 44.94, 'ST': 1.0, 'SK': 1.0, 'SE': 20.26, 
    'SV': 1.0, 'SS': 20.26, 'SG': 1.0, 'SA': 1.0, 'SL': 1.0,
    'GW': 13.34, 'GC': 1.0, 'GM': 1.0, 'GH': 1.0, 'GY': -7.49, 
    'GF': 1.0, 'GQ': 1.0, 'GN': -7.49, 'GI': -7.49, 'GR': 1.0, 
    'GD': 1.0, 'GP': 1.0, 'GT': -7.49, 'GK': -7.49, 'GE': -6.54, 
    'GV': 1.0, 'GS': 1.0, 'GG': 13.34, 'GA': -7.49, 'GL': 1.0,
    'AW': 1.0, 'AC': 44.94, 'AM': 1.0, 'AH': -7.49, 'AY': 1.0, 
    'AF': 1.0, 'AQ': 1.0, 'AN': 1.0, 'AI': 1.0, 'AR': 1.0, 
    'AD': -7.49, 'AP': 20.26, 'AT': 1.0, 'AK': 1.0, 'AE': 1.0, 
    'AV': 1.0, 'AS': 1.0, 'AG': 1.0, 'AA': 1.0, 'AL': 1.0,
    'LW': 24.68, 'LC': 1.0, 'LM': 1.0, 'LH': 1.0, 'LY': 1.0, 
    'LF': 1.0, 'LQ': 33.6, 'LN': 1.0, 'LI': 1.0, 'LR': 20.26, 
    'LD': 1.0, 'LP': 20.26, 'LT': 1.0, 'LK': -7.49, 'LE': 1.0, 
    'LV': 1.0, 'LS': 1.0, 'LG': 1.0, 'LA': 1.0, 'LL': 1.0
}

# Quality thresholds for peptide evaluation
QUALITY_THRESHOLDS = {
    'instability_index': {
        'stable': 40.0,      # Below this = stable
        'moderately_stable': 50.0,
    },
    'aliphatic_index': {
        'good': 60.0,        # Above this = good thermostability
    },
    'therapeutic_score': {
        'promising': 0.5,    # Above this = promising
    },
    'hemolytic_score': {
        'safe': 0.3,         # Below this = safe
        'acceptable': 0.5,
    },
    'gravy': {
        'min': -2.0,
        'max': 1.0,
    },
}

# Feature names used for conditional generation
CONDITION_FEATURE_NAMES: List[str] = [
    'instability_index',
    'therapeutic_score',
    'hemolytic_score',
    'aliphatic_index',
    'hydrophobic_moment',
    'gravy',
    'charge_at_pH7',
    'aromaticity',
]


# Utility functions
def get_aa_property(aa: str, property_name: str, default: float = 0.0) -> float:
    """Get a property value for an amino acid."""
    property_map = {
        'hydropathy': HYDROPATHY_SCALE,
        'molecular_weight': MOLECULAR_WEIGHTS,
        'helix_propensity': HELIX_PROPENSITY,
        'sheet_propensity': SHEET_PROPENSITY,
    }
    if property_name in property_map:
        return property_map[property_name].get(aa.upper(), default)
    return default


def is_in_group(aa: str, group_name: str) -> bool:
    """Check if amino acid belongs to a group."""
    return aa.upper() in AA_GROUPS.get(group_name, set())


def calculate_instability_index(sequence: str) -> float:
    """Calculate instability index for a sequence."""
    if len(sequence) < 2:
        return 0.0
    
    score = 0.0
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i:i+2].upper()
        score += INSTABILITY_WEIGHTS.get(dipeptide, 1.0)
    
    return (10.0 / len(sequence)) * score


def calculate_aliphatic_index(sequence: str) -> float:
    """Calculate aliphatic index for thermostability."""
    if len(sequence) == 0:
        return 0.0
    
    from collections import Counter
    counts = Counter(sequence.upper())
    n = len(sequence)
    
    ala = counts.get('A', 0) / n * 100
    val = counts.get('V', 0) / n * 100
    ile = counts.get('I', 0) / n * 100
    leu = counts.get('L', 0) / n * 100
    
    return ala + 2.9 * val + 3.9 * (ile + leu)


def calculate_gravy(sequence: str) -> float:
    """Calculate GRAVY (Grand Average of Hydropathicity)."""
    valid_aas = [aa for aa in sequence.upper() if aa in HYDROPATHY_SCALE]
    if not valid_aas:
        return 0.0
    return sum(HYDROPATHY_SCALE[aa] for aa in valid_aas) / len(valid_aas)
