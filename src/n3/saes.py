"""Exact, inspectable SAE encoder/decoder wrappers.

We re-implement the encoder of each pretrained SAE (ReLU / JumpReLU / TopK, with or without
run-time LayerNorm input normalization) so that pre-activations, active sets, effective
thresholds and input Jacobians are available.  `check_against_saelens` verifies bit-level
agreement with the SAELens implementation the checkpoint was published for.

Conventions
-----------
h        : residual-stream activation(s) at the hook, shape (..., d)
pre(h)   : encoder pre-activation, shape (..., m)
f(h)     : feature activations, shape (..., m)
W_enc    : (d, m), W_dec : (m, d)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

LN_EPS = 1e-5  # SAELens run_time_activation_ln_in eps


@dataclass
class SAEWrap:
    name: str
    kind: str  # "relu" | "jumprelu" | "topk"
    norm: str  # "none" | "layer_norm"
    W_enc: torch.Tensor
    b_enc: torch.Tensor
    W_dec: torch.Tensor
    b_dec: torch.Tensor
    apply_b_dec_to_input: bool
    threshold: torch.Tensor | None = None
    k: int | None = None
    hook: str = ""
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    @property
    def d(self) -> int:
        return self.W_enc.shape[0]

    @property
    def m(self) -> int:
        return self.W_enc.shape[1]

    @classmethod
    def from_saelens(cls, sae, name: str, hook: str) -> "SAEWrap":
        arch = sae.cfg.architecture() if callable(getattr(sae.cfg, "architecture", None)) else sae.cfg.architecture
        kind = {"standard": "relu", "jumprelu": "jumprelu", "topk": "topk"}[arch]
        thr = sae.threshold.detach().clone().float() if kind == "jumprelu" else None
        k = int(sae.cfg.k) if kind == "topk" else None
        if kind == "topk" and getattr(sae.cfg, "rescale_acts_by_decoder_norm", False):
            raise NotImplementedError("rescale_acts_by_decoder_norm TopK not supported")
        return cls(
            name=name,
            kind=kind,
            norm=sae.cfg.normalize_activations,
            W_enc=sae.W_enc.detach().clone().float(),
            b_enc=sae.b_enc.detach().clone().float(),
            W_dec=sae.W_dec.detach().clone().float(),
            b_dec=sae.b_dec.detach().clone().float(),
            apply_b_dec_to_input=bool(sae.cfg.apply_b_dec_to_input),
            threshold=thr,
            k=k,
            hook=hook,
        )

    # ------------------------------------------------------------ normalization
    def _normalize(self, h: torch.Tensor):
        if self.norm == "none":
            return h, None
        if self.norm == "layer_norm":
            mu = h.mean(dim=-1, keepdim=True)
            c = h - mu
            std = c.std(dim=-1, keepdim=True)  # unbiased (d-1), as in SAELens
            return c / (std + LN_EPS), (mu, std)
        raise NotImplementedError(self.norm)

    def _denormalize(self, x: torch.Tensor, stats):
        if stats is None:
            return x
        mu, std = stats
        return x * std + mu

    # ------------------------------------------------------------------ encode
    def pre(self, h: torch.Tensor) -> torch.Tensor:
        x, _ = self._normalize(h)
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        return x @ self.W_enc + self.b_enc

    def act_from_pre(self, pre: torch.Tensor) -> torch.Tensor:
        if self.kind == "relu":
            return pre.clamp_min(0.0)
        if self.kind == "jumprelu":
            return pre * (pre > self.threshold).to(pre.dtype)
        if self.kind == "topk":
            vals, idx = torch.topk(pre, k=self.k, dim=-1, sorted=False)
            out = torch.zeros_like(pre)
            out.scatter_(-1, idx, vals.clamp_min(0.0))
            return out
        raise NotImplementedError(self.kind)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return self.act_from_pre(self.pre(h))

    def encode_full(self, h: torch.Tensor):
        p = self.pre(h)
        return p, self.act_from_pre(p)

    def decode(self, f: torch.Tensor, h_for_stats: torch.Tensor | None = None) -> torch.Tensor:
        out = f @ self.W_dec + self.b_dec
        if self.norm == "layer_norm":
            assert h_for_stats is not None, "LN SAE decode needs the input to recover (mu, std)"
            _, stats = self._normalize(h_for_stats)
            out = self._denormalize(out, stats)
        return out

    def effective_threshold(self, pre: torch.Tensor) -> torch.Tensor:
        """Per-feature activation threshold at this input (broadcast to pre's shape).

        relu: 0; jumprelu: theta; topk: max(0, k-th largest pre-activation) (input dependent).
        A feature is active iff pre > threshold (topk: pre >= kth and > 0).
        """
        if self.kind == "relu":
            return torch.zeros_like(pre)
        if self.kind == "jumprelu":
            return self.threshold.expand_as(pre)
        kth = torch.topk(pre, k=self.k, dim=-1).values[..., -1:]
        return kth.clamp_min(0.0).expand_as(pre)

    # ---------------------------------------------------------------- Jacobian
    def pre_jacobian(self, h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """d pre_idx / d h.

        h: (n, d); idx: (n, k) long (use -1 for padding -> zero row).
        Returns J: (n, k, d).  For norm == "none" rows are W_enc[:, j]^T (input independent).
        For LayerNorm input normalization, J = W_enc[:, idx]^T @ J_N(h) where
        J_N = (1/(s+eps)) [C - c c^T / ((d-1) s (s+eps))], C = I - 11^T/d, c = h - mean(h), s = std(h).
        """
        n, k = idx.shape
        valid = idx >= 0
        safe = idx.clamp_min(0)
        W = self.W_enc.T[safe]  # (n, k, d)
        W = W * valid.unsqueeze(-1).to(W.dtype)
        if self.norm == "none":
            return W
        d = h.shape[-1]
        mu = h.mean(dim=-1, keepdim=True)
        c = h - mu  # (n, d)
        s = c.std(dim=-1, keepdim=True)  # (n, 1)
        a = 1.0 / (s + LN_EPS)  # (n,1)
        # J_N w = a * (C w - c (c.w) / ((d-1) s (s+eps)))  (J_N symmetric)
        Wc = W - W.mean(dim=-1, keepdim=True)  # C w
        cw = torch.einsum("nkd,nd->nk", W, c)  # c . w  (c already centered so c.Cw == c.w)
        coef = (cw / ((d - 1) * s * (s + LN_EPS)))  # (n,k)
        J = a.unsqueeze(-1) * (Wc - coef.unsqueeze(-1) * c.unsqueeze(1))
        return J

    def decoder_dirs(self, idx) -> torch.Tensor:
        return self.W_dec[idx]


def load_sae(release: str, sae_id: str, name: str | None = None) -> tuple[SAEWrap, object]:
    from sae_lens import SAE

    out = SAE.from_pretrained(release, sae_id, device="cpu")
    sae = out[0] if isinstance(out, tuple) else out
    hook = sae.cfg.metadata.hook_name
    return SAEWrap.from_saelens(sae, name or f"{release}/{sae_id}", hook), sae


@torch.no_grad()
def check_against_saelens(w: SAEWrap, sae, h: torch.Tensor) -> dict:
    """Compare our encode/decode with SAELens on activations h (n, d)."""
    f_ref = sae.encode(h)
    rec_ref = sae.decode(f_ref) if w.norm == "none" else sae(h)  # LN decode needs stored stats
    f = w.encode(h)
    rec = w.decode(f, h)
    return {
        "max_abs_diff_f": float((f - f_ref).abs().max()),
        "max_abs_diff_rec": float((rec - rec_ref).abs().max()),
        "active_set_agree": float(((f > 0) == (f_ref > 0)).float().mean()),
    }
