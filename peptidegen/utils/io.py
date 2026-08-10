import json
import yaml
from pathlib import Path
import logging

def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_config(path):
    # utf-8: config.yaml contains Vietnamese comments; the Windows default
    # cp1252 codec would raise UnicodeDecodeError on them.
    with open(path, 'r', encoding='utf-8') as f:
        if Path(path).suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        return json.load(f)

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def set_seed(seed: int = 42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
