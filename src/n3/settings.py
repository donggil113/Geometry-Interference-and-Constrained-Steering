"""Named (model, SAE) settings pinned in configs/pinned.json."""
from __future__ import annotations

from functools import lru_cache

from .data import PINNED, load_json
from .models import HFModel, TLModel
from .saes import load_sae


@lru_cache(maxsize=None)
def _model(key):
    if key == "gpt2":
        # HF GPT-2, output of block 5 (== TL blocks.6.hook_resid_pre), mean-centred (== TL center_writing_weights)
        pin = load_json(PINNED)["settings"]["gpt2_relu_jb_L6"]
        return HFModel("openai-community/gpt2", revision=pin["model_revision"], layer=5, center=True, manual_bos=True,
                       hook_name="transformer.h.5.output (centred) == blocks.6.hook_resid_pre")
    if key == "gpt2_tl":
        return TLModel("gpt2", hook="blocks.6.hook_resid_pre")
    if key == "gemma3_270m":
        pin = load_json(PINNED)["settings"]["gemma3_270m_jumprelu_L12"]
        return HFModel("unsloth/gemma-3-270m", revision=pin["model_revision"], layer=12)
    raise KeyError(key)


MEAN_INVARIANT = {"gpt2_relu_jb_L6", "gpt2_topk_oai_L6"}  # GPT-2 reads the residual only through LayerNorms


@lru_cache(maxsize=None)
def _sae(name):
    pin = load_json(PINNED)["settings"][name]
    w = load_sae(pin["sae_release"], pin["sae_id"], name)[0]
    w.center_input = name in MEAN_INVARIANT
    return w


def load_setting(name: str):
    key = "gemma3_270m" if name.startswith("gemma3") else "gpt2"
    return _model(key), _sae(name)
