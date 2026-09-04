#!/usr/bin/env python
"""
Preprocess raw datasets from dataset/raw using peptidegen.data package.

Usage:
    python scripts/preprocess_data.py --raw-dir dataset/raw --output-dir dataset
"""

import sys
from pathlib import Path

# Add root directory to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from peptidegen.data.__main__ import main

if __name__ == "__main__":
    main()
