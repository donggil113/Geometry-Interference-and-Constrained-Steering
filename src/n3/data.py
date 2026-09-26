"""Pinned data sources and deterministic splits."""
from __future__ import annotations

import json
import os  # noqa: F401
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PINNED = ROOT / "configs" / "pinned.json"

# Dataset files (repo, revision, path). Revisions are the HF commit shas observed on 2026-09-26.
DATA_FILES = {
    "owt": ("Skylion007/openwebtext", "79d93d786212f7344586290adb811d4ae6a1762c", "plain_text/train-00000-of-00080.parquet"),
    "pile10k": ("NeelNanda/pile-10k", "127bfedcd5047750df5ccf3a12979a47bfa0bafa", "data/train-00000-of-00001-4746b8785c874cc7.parquet"),
    "emotion_train": ("dair-ai/emotion", "cab853a1dbdf4c42c2b3ef2173804746df8825fe", "split/train-00000-of-00001.parquet"),
    "yahoo_train0": ("community-datasets/yahoo_answers_topics", "6652a1e7c94f7260a0bfd0c9092dd48e2d536ea1", "yahoo_answers_topics/train-00000-of-00002.parquet"),
}


def fetch(key: str) -> Path:
    from huggingface_hub import hf_hub_download

    repo, rev, path = DATA_FILES[key]
    return Path(hf_hub_download(repo_id=repo, filename=path, revision=rev, repo_type="dataset"))


def load_df(key: str) -> pd.DataFrame:
    return pd.read_parquet(fetch(key))


# Disjoint document ranges of the OWT shard used for different purposes (never overlap).
OWT_RANGES = {
    "validation": (0, 400),        # checkpoint / hook validation (FVU, CE recovered)
    "stats": (400, 2400),          # activation statistics (norm scale, feature frequencies)
    "prompts": (2400, 4400),       # candidate generation prompts (dev/test split below)
    "utility": (4400, 4800),       # held-out LM-loss utility text
}


def owt_texts(purpose: str) -> list[str]:
    df = load_df("owt")
    a, b = OWT_RANGES[purpose]
    return df["text"].iloc[a:b].tolist()


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_default)


def _default(o):
    try:
        import numpy as np

        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def load_json(path):
    with open(path) as f:
        return json.load(f)



