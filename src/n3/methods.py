"""Norm-matched activation edits for SAE feature steering.

Every method maps residual activations h (n, d) to an edit delta (n, d) with ||delta_i|| = alpha
(per position; alpha is absolute, callers express it as a multiple of the median residual norm).

Methods
-------
none       : delta = 0
dec        : alpha * normalized sum_t w_t W_dec[t]                 (standard SAE decoder steering)
enc        : alpha * normalized sum_t w_t d pre_t / dh             (encoder-gradient steering)
pinv       : alpha * normalized J^+ [w_T; 0]  with J = [J_T; J_P]  (min-norm edit that moves pre_T in
             proportion w_T while keeping the pre-activations of protected features fixed, to first order)
ridge      : alpha * normalized J^T (J J^T + lam I)^-1 [w_T; 0]    (interpolates enc (lam->inf) and pinv)
dec_proj   : alpha * normalized (I - Pi_P) d_T                     (decoder direction projected onto the null
             space of the protected features' pre-activation Jacobian: min ||delta - c d_T|| s.t. J_P delta = 0)
*_fs       : finite-step correction for pinv / dec_proj: after the linear solve, the true encoder is
             evaluated at h + delta; non-target features that become active are added to the constraint set
             (J_j delta = 0) and the solve is repeated (constraint generation, `fs_rounds` rounds).
opt        : direct activation optimization on the sphere ||delta|| = alpha of
             sum_t w_t pre_t(h+delta)/s_t - mu_P * mean_P ((pre_j(h+delta)-pre_j(h))/s_j)^2
                                           - mu_new * sum_{inactive j} relu(pre_j(h+delta) - thr_j(h) + margin)/s_j
             over a candidate feature subset, initialised at the decoder direction.
diffmean   : alpha * normalized (mean h | concept - mean h | other concepts)   (generic baseline)
random     : alpha * fixed random unit vector                                 (generic baseline)
dec_rand_feat      : alpha * decoder row of a random SAE feature (privileged-direction control)
dec_proj_randP     : like dec_proj but the protected rows are replaced by an equal number (per position) of
dec_proj_fs_randP    uniformly random features' rows (specificity control for the protected set / FS additions)

P (protected set) at each position = features active at h, excluding T (optionally the top-m by activation).
All Jacobians are of *pre-activations*, which are affine in h for norm="none" SAEs (ReLU / JumpReLU) and
first-order only for LayerNorm-normalized SAEs (TopK OAI).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .saes import SAEWrap

METHODS = ["none", "dec", "enc", "pinv", "ridge", "dec_proj", "pinv_fs", "dec_proj_fs", "opt", "diffmean", "random",
           "dec_rand_feat", "dec_proj_randP", "dec_proj_fs_randP"]
GENERIC_DIR = {"diffmean", "random", "dec_rand_feat"}  # fixed direction supplied via EditSpec.generic_dir
SAE_FREE = {"none", "diffmean", "random"}


@dataclass
class EditSpec:
    method: str
    T: torch.Tensor  # (K,) long
    wT: torch.Tensor  # (K,) float, positive, sums to 1
    alpha: float
    protect_topm: int | None = None  # None = all active non-target features
    ridge_rel: float = 1.0
    fs_rounds: int = 3
    fs_max_add: int = 64
    opt_steps: int = 25
    opt_lr: float = 0.08
    opt_mu_p: float = 1.0
    opt_mu_new: float = 1.0
    opt_margin: float = 0.0
    opt_candidates: int = 512
    generic_dir: torch.Tensor | None = None  # (d,) for diffmean / random
    feat_scale: torch.Tensor | None = None  # (m,) typical activation scale for opt
    chunk: int = 256


def _normalize_rows(v: torch.Tensor, alpha: float) -> torch.Tensor:
    nrm = v.norm(dim=-1, keepdim=True)
    return torch.where(nrm > 1e-12, alpha * v / nrm.clamp_min(1e-12), torch.zeros_like(v))


def _protected_idx(sae: SAEWrap, pre: torch.Tensor, f: torch.Tensor, T: torch.Tensor, topm: int | None) -> torch.Tensor:
    """Padded (n, k) index tensor of protected features (active at h, not in T); -1 = padding."""
    act = f > 0
    act[:, T] = False
    counts = act.sum(-1)
    kmax = int(counts.max().item()) if act.numel() else 0
    if topm is not None:
        kmax = min(kmax, topm)
    if kmax == 0:
        return torch.full((f.shape[0], 1), -1, dtype=torch.long)
    score = torch.where(act, f, torch.full_like(f, -1.0))
    vals, idx = torch.topk(score, k=kmax, dim=-1)
    idx = torch.where(vals > 0, idx, torch.full_like(idx, -1))
    return idx


def _batched_min_norm(J: torch.Tensor, r: torch.Tensor, ridge_abs: torch.Tensor | None = None) -> torch.Tensor:
    """delta = J^T (J J^T + reg)^-1 r  for J (n, k, d), r (n, k).  Zero (padding) rows are handled by
    putting 1 on their diagonal (their r is 0 so they contribute nothing)."""
    G = J @ J.transpose(1, 2)  # (n,k,k)
    diag = torch.diagonal(G, dim1=1, dim2=2)
    pad = diag <= 1e-20
    eye = torch.eye(G.shape[-1], dtype=G.dtype).expand_as(G)
    scale = diag.clamp_min(0).mean(-1, keepdim=True).clamp_min(1e-12)  # (n,1)
    if ridge_abs is None:
        reg = 1e-6 * scale
    else:
        reg = ridge_abs
    G = G + eye * (reg.unsqueeze(-1)) + torch.diag_embed(pad.to(G.dtype))
    x = torch.linalg.solve(G, r.unsqueeze(-1)).squeeze(-1)  # (n,k)
    return torch.einsum("nk,nkd->nd", x, J)


def _project_out(J: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(I - Pi_rowspace(J)) v for J (n,k,d), v (n,d)."""
    Jv = torch.einsum("nkd,nd->nk", J, v)
    return v - _batched_min_norm(J, Jv)


class Editor:
    """Stateful editor: computes delta for a batch of positions and accumulates internal metrics."""

    def __init__(self, sae: SAEWrap | None, spec: EditSpec, collect: bool = True):
        self.sae = sae
        self.spec = spec
        self.collect = collect
        self.stats = {k: 0.0 for k in ["n", "tgt_gain", "tgt_pre_gain", "p_rel_change", "p_lost", "new_active", "delta_norm",
                                       "fs_violations", "secs", "err_norm_ratio", "err_delta_frac", "n_constraints"]}
        self.T = spec.T
        self.wT = spec.wT
        if sae is not None:
            self.d_T = (spec.wT.unsqueeze(-1) * sae.W_dec[spec.T]).sum(0)
            # candidate features for `opt`: T + those whose pre-activation is most moved by dec / enc directions
            if spec.method == "opt":
                inter = (self.d_T @ sae.W_enc).abs() + ((sae.W_enc[:, spec.T] @ spec.wT) @ sae.W_enc).abs() / sae.W_enc.norm(dim=0).clamp_min(1e-6)
                inter[spec.T] = float("inf")
                self.cand = torch.topk(inter, k=min(spec.opt_candidates, sae.m)).indices

    # ------------------------------------------------------------------ core
    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        out = []
        for s in range(0, h.shape[0], self.spec.chunk):
            out.append(self._edit_chunk(h[s:s + self.spec.chunk]))
        return torch.cat(out, 0)

    def _edit_chunk(self, h: torch.Tensor) -> torch.Tensor:
        sp = self.spec
        t0 = time.perf_counter()
        m = sp.method
        if m == "none" or sp.alpha == 0:
            delta = torch.zeros_like(h)
            pre = f = None
        elif m in GENERIC_DIR:
            delta = _normalize_rows(sp.generic_dir.expand_as(h).clone(), sp.alpha)
            pre = f = None
        else:
            sae = self.sae
            pre, f = sae.encode_full(h)
            delta = self._sae_method(h, pre, f)
        secs = time.perf_counter() - t0
        if self.collect:
            self._collect(h, delta, pre, f, secs)
        return delta

    def _target_grad(self, h):
        J_T = self.sae.pre_jacobian(h, self.T.expand(h.shape[0], -1))  # (n,K,d)
        return J_T, torch.einsum("k,nkd->nd", self.wT, J_T)

    def _sae_method(self, h, pre, f):
        sp, sae = self.spec, self.sae
        n = h.shape[0]
        m = sp.method
        if m == "dec":
            return _normalize_rows(self.d_T.expand(n, -1).clone(), sp.alpha)
        J_T, g_T = self._target_grad(h)
        if m == "enc":
            return _normalize_rows(g_T, sp.alpha)
        Pidx = _protected_idx(sae, pre, f, self.T, sp.protect_topm)
        if m in ("pinv", "ridge", "pinv_fs"):
            rounds = sp.fs_rounds if m == "pinv_fs" else 0
            return self._solve_iter(h, pre, f, Pidx, rounds, lambda J_P: self._pinv_dir(J_T, J_P, ridge=(m == "ridge")))[0]
        if m in ("dec_proj", "dec_proj_fs"):
            rounds = sp.fs_rounds if m == "dec_proj_fs" else 0
            dT = self.d_T.expand(n, -1)
            return self._solve_iter(h, pre, f, Pidx, rounds, lambda J_P: _project_out(J_P, dT))[0]
        if m in ("dec_proj_randP", "dec_proj_fs_randP"):
            rounds = sp.fs_rounds if m == "dec_proj_fs_randP" else 0
            dT = self.d_T.expand(n, -1)
            _, idx = self._solve_iter(h, pre, f, Pidx, rounds, lambda J_P: _project_out(J_P, dT))
            ridx = self._random_like(idx)
            return _normalize_rows(_project_out(sae.pre_jacobian(h, ridx), dT), sp.alpha)
        if m == "opt":
            # generation runs under torch.inference_mode(); autograd needs ordinary tensors
            with torch.inference_mode(False), torch.enable_grad():
                return self._opt(h.clone(), pre.clone(), f.clone(), Pidx.clone())
        raise ValueError(m)

    def _pinv_dir(self, J_T, J_P, ridge=False):
        n, K, d = J_T.shape
        J = torch.cat([J_T, J_P], 1)
        r = torch.cat([self.wT.expand(n, -1), torch.zeros(n, J_P.shape[1])], 1)
        if ridge:
            diag = torch.diagonal(J @ J.transpose(1, 2), dim1=1, dim2=2)
            valid = diag > 1e-20
            scale = (diag * valid).sum(-1) / valid.sum(-1).clamp_min(1)
            return _batched_min_norm(J, r, ridge_abs=(self.spec.ridge_rel * scale).unsqueeze(-1))
        return _batched_min_norm(J, r)

    def _solve_iter(self, h, pre, f, Pidx, rounds, dir_fn):
        sp, sae = self.spec, self.sae
        idx = Pidx
        J_P = sae.pre_jacobian(h, idx)
        delta = _normalize_rows(dir_fn(J_P), sp.alpha)
        self._last_fs_viol = 0
        for _ in range(rounds):
            f_new = sae.encode(h + delta)
            newly = (f_new > 0) & (f <= 0)
            newly[:, self.T] = False
            if not bool(newly.any()):
                break
            score = torch.where(newly, f_new, torch.full_like(f_new, -1.0))
            kadd = min(sp.fs_max_add, int(newly.sum(-1).max()))
            vals, add = torch.topk(score, k=kadd, dim=-1)
            add = torch.where(vals > 0, add, torch.full_like(add, -1))
            idx = torch.cat([idx, add], 1)
            J_P = sae.pre_jacobian(h, idx)
            delta = _normalize_rows(dir_fn(J_P), sp.alpha)
        if rounds:
            f_new = sae.encode(h + delta)
            v = (f_new > 0) & (f <= 0)
            v[:, self.T] = False
            self._last_fs_viol = int(v.sum())
        self._last_n_constraints = float((idx >= 0).sum(-1).float().mean())
        return delta, idx

    def _random_like(self, idx):
        """Same number of constraint rows per position, but uniformly random non-target features."""
        if not hasattr(self, "_rng"):
            self._rng = torch.Generator().manual_seed(4242 + int(self.T.sum()))
        n, k = idx.shape
        counts = (idx >= 0).sum(-1)
        r = torch.randint(0, self.sae.m, (n, k), generator=self._rng)
        bad = torch.isin(r, self.T)
        while bool(bad.any()):
            r[bad] = torch.randint(0, self.sae.m, (int(bad.sum()),), generator=self._rng)
            bad = torch.isin(r, self.T)
        keep = torch.arange(k)[None, :] < counts[:, None]
        return torch.where(keep, r, torch.full_like(r, -1))

    def _opt(self, h, pre, f, Pidx):
        sp, sae = self.spec, self.sae
        n = h.shape[0]
        cand = self.cand
        scale = sp.feat_scale[cand] if sp.feat_scale is not None else torch.ones(len(cand))
        scale = scale.clamp_min(1e-3)
        is_T = torch.zeros(len(cand), dtype=torch.bool)
        pos_T = torch.stack([(cand == t).nonzero()[0, 0] for t in self.T])
        is_T[pos_T] = True
        # protected = active at h (full encoder) restricted to candidates, or all active if norm-free
        pre_c0 = pre[:, cand]
        f_c0 = f[:, cand]
        thr0 = sae.effective_threshold(pre)[:, cand]
        activeP = (f_c0 > 0) & ~is_T
        inactive = (f_c0 <= 0) & ~is_T
        # protected features outside the candidate set are handled by an exact linear constraint for norm="none"
        J_P = sae.pre_jacobian(h, Pidx)
        u = _project_out(J_P, self.d_T.expand(n, -1).clone()) if sae.norm == "none" else self.d_T.expand(n, -1).clone()
        u = u.clone().requires_grad_(True)
        opt = torch.optim.Adam([u], lr=sp.opt_lr * float(u.detach().norm(dim=-1).mean()))
        Wc = sae.W_enc[:, cand]
        wT_c = torch.zeros(len(cand))
        wT_c[pos_T] = self.wT
        with torch.enable_grad():
            for _ in range(sp.opt_steps):
                delta = sp.alpha * u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                if sae.norm == "none":
                    delta = delta - _batched_min_norm(J_P, torch.einsum("nkd,nd->nk", J_P, delta))
                    delta = sp.alpha * delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                x, _ = sae._normalize(h + delta)
                if sae.apply_b_dec_to_input:
                    x = x - sae.b_dec
                pre_c = x @ Wc + sae.b_enc[cand]
                gain = ((pre_c - pre_c0) / scale * wT_c).sum(-1)
                dp = ((pre_c - pre_c0) / scale) ** 2
                pen_p = (dp * activeP).sum(-1) / activeP.sum(-1).clamp_min(1)
                pen_new = (torch.relu(pre_c - thr0 + sp.opt_margin * scale) / scale * inactive).sum(-1)
                loss = -(gain - sp.opt_mu_p * pen_p - sp.opt_mu_new * pen_new).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            delta = sp.alpha * u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            if sae.norm == "none":
                delta = delta - _batched_min_norm(J_P, torch.einsum("nkd,nd->nk", J_P, delta))
                delta = _normalize_rows(delta, sp.alpha)
        return delta.detach()

    # --------------------------------------------------------------- metrics
    @torch.no_grad()
    def _collect(self, h, delta, pre, f, secs):
        st = self.stats
        n = h.shape[0]
        st["n"] += n
        st["secs"] += secs
        st["delta_norm"] += float(delta.norm(dim=-1).sum())
        if self.sae is None:
            return
        sae = self.sae
        if pre is None:
            pre, f = sae.encode_full(h)
        pre2, f2 = sae.encode_full(h + delta)
        st["tgt_gain"] += float(((f2[:, self.T] - f[:, self.T]) * self.wT).sum())
        st["tgt_pre_gain"] += float(((pre2[:, self.T] - pre[:, self.T]) * self.wT).sum())
        P = f > 0
        P[:, self.T] = False
        fP, fP2 = f * P, f2 * P
        rel = (fP2 - fP).norm(dim=-1) / fP.norm(dim=-1).clamp_min(1e-6)
        st["p_rel_change"] += float(rel.sum())
        st["p_lost"] += float((P & (f2 <= 0)).sum())
        newly = (f2 > 0) & (f <= 0)
        newly[:, self.T] = False
        st["new_active"] += float(newly.sum())
        st["fs_violations"] += float(getattr(self, "_last_fs_viol", 0))
        st["n_constraints"] += float(getattr(self, "_last_n_constraints", 0.0)) * n
        # off-manifold / error-channel diagnostics: SAE error e(x) = x - dec(enc(x))
        e0 = h - sae.decode(f, h)
        e1 = (h + delta) - sae.decode(f2, h + delta)
        st["err_norm_ratio"] += float((e1.norm(dim=-1) / e0.norm(dim=-1).clamp_min(1e-6)).sum())
        dn = delta.norm(dim=-1)
        st["err_delta_frac"] += float(torch.where(dn > 0, (e1 - e0).norm(dim=-1) / dn.clamp_min(1e-9), torch.zeros_like(dn)).sum())

    def summary(self) -> dict:
        st = self.stats
        n = max(st["n"], 1)
        return {
            "positions": st["n"],
            "tgt_gain": st["tgt_gain"] / n,
            "tgt_pre_gain": st["tgt_pre_gain"] / n,
            "p_rel_change": st["p_rel_change"] / n,
            "p_lost_per_pos": st["p_lost"] / n,
            "new_active_per_pos": st["new_active"] / n,
            "delta_norm": st["delta_norm"] / n,
            "ms_per_pos": 1000 * st["secs"] / n,
            "err_norm_ratio": st["err_norm_ratio"] / n,
            "err_delta_frac": st["err_delta_frac"] / n,
            "n_constraints": st["n_constraints"] / n,
            "fs_violations_total": st["fs_violations"],
        }
