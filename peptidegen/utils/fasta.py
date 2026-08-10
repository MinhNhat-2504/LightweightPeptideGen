import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def read_fasta(file_path: str, return_ids: bool = False) -> List:
    """Read FASTA and return sequences (optionally with IDs as (id, seq) tuples)."""
    sequences = []
    current_id = None
    current_seq = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if current_id and current_seq:
                    seq = "".join(current_seq)
                    sequences.append((current_id, seq) if return_ids else seq)
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id and current_seq:
            seq = "".join(current_seq)
            sequences.append((current_id, seq) if return_ids else seq)
            
    return sequences
