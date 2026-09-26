"""Named (model, SAE) settings pinned in configs/pinned.json."""
from __future__ import annotations

from functools import lru_cache

from .data import PINNED, load_json
from .models import HFModel, TLModel
from .saes import load_sae


@lru_cache(maxsize=None)
def _model(key):
    if key == "gpt2":
        return TLModel("gpt2", hook="blocks.6.hook_resid_pre")
    if key == "gemma3_270m":
        pin = load_json(PINNED)["settings"]["gemma3_270m_jumprelu_L12"]
        return HFModel("unsloth/gemma-3-270m", revision=pin["model_revision"], layer=12)
    raise KeyError(key)


@lru_cache(maxsize=None)
def _sae(name):
    pin = load_json(PINNED)["settings"][name]
    return load_sae(pin["sae_release"], pin["sae_id"], name)[0]


def load_setting(name: str):
    key = "gemma3_270m" if name.startswith("gemma3") else "gpt2"
    return _model(key), _sae(name)
