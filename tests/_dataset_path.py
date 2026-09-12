# -*- coding: utf-8 -*-
"""Tim file train.csv dang duoc dung, thay vi go cung mot duong dan.

Cac test truoc day tro thang vao 'dataset/train.csv'. Duong dan do khong con ton
tai sau khi bo du lieu duoc dung lai; dataset that duoc khai bao trong
config/revision.yaml. Tim theo thu tu: config -> cac thu muc rebuilt -> duong dan cu.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _from_config():
    config = ROOT / "config/revision.yaml"
    if not config.exists():
        return None
    for line in config.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("train_csv:"):
            value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            if value:
                return ROOT / value
    return None


def train_csv() -> Path:
    """Path to the training CSV, or skip the test if no dataset is present."""
    candidates = [_from_config()]
    candidates += sorted(ROOT.glob("dataset/rebuilt*/train.csv"), reverse=True)
    candidates.append(ROOT / "dataset/train.csv")
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    pytest.skip(
        "no training CSV found; build one with "
        "`python -m peptidegen.data --manifest config/dataset_manifest.json ...`"
    )
