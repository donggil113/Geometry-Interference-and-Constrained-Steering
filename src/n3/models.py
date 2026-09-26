"""Model adapters: residual-stream read / edit hooks and seeded generation.

Two back ends:
  * TransformerLens HookedTransformer (GPT-2 small, default weight processing, which is what the
    jbloom residual SAEs were trained on: center_writing_weights=True, fold_ln=True).
  * Hugging Face causal LM (Gemma-3-270m) with a forward hook on `model.model.layers[L]` output,
    matching Gemma Scope 2's `hf_hook_point_in = model.layers.L.output`.

An *editor* is a callable  editor(h: (n, d) float32) -> delta (n, d)  applied to every position
except the BOS position of the first (prompt) forward pass.
"""
from __future__ import annotations

import contextlib
from typing import Callable

import torch

Editor = Callable[[torch.Tensor], torch.Tensor]


class _EditState:
    def __init__(self, editor: Editor | None, skip_bos: bool = True):
        self.editor = editor
        self.skip_bos = skip_bos
        self.calls = 0

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.editor is None:
            return x
        b, s, d = x.shape
        start = 1 if (self.skip_bos and self.calls == 0) else 0
        self.calls += 1
        if s - start <= 0:
            return x
        h = x[:, start:, :].reshape(-1, d).float()
        delta = self.editor(h).to(x.dtype).reshape(b, s - start, d)
        x = x.clone()
        x[:, start:, :] = x[:, start:, :] + delta
        return x


class TLModel:
    backend = "transformer_lens"

    def __init__(self, name: str = "gpt2", hook: str = "blocks.6.hook_resid_pre"):
        from transformer_lens import HookedTransformer

        self.name = name
        self.model = HookedTransformer.from_pretrained(name, device="cpu")
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.hook = hook
        self.bos_id = self.tokenizer.bos_token_id

    def to_tokens(self, texts, prepend_bos=True, max_len=None):
        toks = self.model.to_tokens(texts, prepend_bos=prepend_bos)
        return toks if max_len is None else toks[:, :max_len]

    @torch.no_grad()
    def resid(self, tokens: torch.Tensor, hook: str | None = None) -> torch.Tensor:
        hook = hook or self.hook
        _, cache = self.model.run_with_cache(tokens, names_filter=[hook], stop_at_layer=_layer_of(hook) + 1)
        return cache[hook]

    @contextlib.contextmanager
    def editing(self, editor: Editor | None, skip_bos: bool = True):
        st = _EditState(editor, skip_bos)

        def fn(x, hook):
            return st.apply(x)

        with self.model.hooks(fwd_hooks=[(self.hook, fn)] if editor is not None else []):
            yield st

    @torch.no_grad()
    def logits(self, tokens, editor: Editor | None = None):
        with self.editing(editor):
            return self.model(tokens)

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, editor: Editor | None, max_new_tokens: int, seed: int,
                 temperature: float = 1.0, top_p: float = 0.9) -> torch.Tensor:
        torch.manual_seed(seed)
        with self.editing(editor):
            out = self.model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=True,
                                      temperature=temperature, top_p=top_p, stop_at_eos=False,
                                      verbose=False, use_past_kv_cache=True, prepend_bos=False)
        return out


class HFModel:
    backend = "hf"

    def __init__(self, repo: str, revision: str, layer: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = repo
        self.tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, dtype=torch.float32)
        self.model.eval()
        self.layer = layer
        self.hook = f"model.layers.{layer}.output"
        self.bos_id = self.tokenizer.bos_token_id

    def to_tokens(self, texts, prepend_bos=True, max_len=None):
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=prepend_bos)
        toks = enc["input_ids"]
        return toks if max_len is None else toks[:, :max_len]

    def _layer_module(self):
        return self.model.model.layers[self.layer]

    @torch.no_grad()
    def resid(self, tokens: torch.Tensor, hook: str | None = None) -> torch.Tensor:
        store = {}

        def fn(mod, inp, out):
            store["h"] = (out[0] if isinstance(out, tuple) else out).detach()

        hnd = self._layer_module().register_forward_hook(fn)
        try:
            self.model(tokens)
        finally:
            hnd.remove()
        return store["h"]

    @contextlib.contextmanager
    def editing(self, editor: Editor | None, skip_bos: bool = True):
        st = _EditState(editor, skip_bos)

        def fn(mod, inp, out):
            if isinstance(out, tuple):
                return (st.apply(out[0]),) + tuple(out[1:])
            return st.apply(out)

        hnd = self._layer_module().register_forward_hook(fn) if editor is not None else None
        try:
            yield st
        finally:
            if hnd is not None:
                hnd.remove()

    @torch.no_grad()
    def logits(self, tokens, editor: Editor | None = None):
        with self.editing(editor):
            return self.model(tokens).logits

    @torch.no_grad()
    def generate(self, tokens, editor, max_new_tokens, seed, temperature=1.0, top_p=0.9):
        torch.manual_seed(seed)
        with self.editing(editor):
            out = self.model.generate(tokens, attention_mask=torch.ones_like(tokens), max_new_tokens=max_new_tokens,
                                      min_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                      top_p=top_p, top_k=0, pad_token_id=self.tokenizer.pad_token_id)
        return out


def _layer_of(hook: str) -> int:
    return int(hook.split(".")[1])
