"""Feature statistics, concept-feature selection and DiffMean directions for one (model, SAE) setting.

Everything here uses only *training* texts of the concept datasets and the OWT 'stats' range.
Selection rule (hyper-parameters frozen on dev concepts):
    score_j(c) = (mu_c,j - mu_notc,j) / sqrt((var_c,j + var_notc,j)/2 + eps)
where mu/var are over per-text mean activations (BOS excluded), 'not c' = other concepts of the same
family.  Features whose OWT firing density exceeds `max_density` are excluded.  T(c) = top-K by score,
w_T proportional to score.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .concepts import all_concepts, concept_texts
from .data import ROOT, owt_texts

CACHE = ROOT / "results" / "cache"


@torch.no_grad()
def padded_resid(model, texts, seq_len, pad_id=0):
    """Right-padded batch -> (H (b, s, d), lengths).  Causal attention makes padding after the real
    tokens irrelevant for the real positions."""
    toks = [model.to_tokens([t], prepend_bos=True)[0][:seq_len] for t in texts]
    lens = torch.tensor([len(t) for t in toks])
    L = int(lens.max())
    batch = torch.full((len(toks), L), pad_id, dtype=torch.long)
    for i, t in enumerate(toks):
        batch[i, :len(t)] = t
    return model.resid(batch).float(), lens


@torch.no_grad()
def per_text_means(model, sae, texts, seq_len=64, bs=16):
    """Per-text mean feature activations (N, m) and residuals (N, d); BOS excluded."""
    fm, hm = [], []
    for i in range(0, len(texts), bs):
        H, lens = padded_resid(model, texts[i:i + bs], seq_len)
        for b in range(H.shape[0]):
            Hb = H[b, 1:lens[b]]
            if Hb.shape[0] == 0:
                continue
            fm.append(sae.encode(Hb).mean(0))
            hm.append(Hb.mean(0))
    return torch.stack(fm), torch.stack(hm)


@torch.no_grad()
def compute_setting_stats(model, sae, setting: str, n_per_concept=300, n_owt=400, seq_len=64):
    out_path = CACHE / setting / "featstats.pt"
    if out_path.exists():
        return torch.load(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ct = concept_texts(n_per_concept)
    Fm, Hm = {}, {}
    for cid, texts in ct.items():
        Fm[cid], Hm[cid] = per_text_means(model, sae, texts, seq_len)
        print("stats", setting, cid, Fm[cid].shape, flush=True)
    # OWT firing density and typical nonzero activation
    dens = torch.zeros(sae.m)
    ssum = torch.zeros(sae.m)
    npos = 0
    norms = []
    for t in owt_texts("stats")[:n_owt]:
        toks = model.to_tokens([t], prepend_bos=True)[0][:128]
        H = model.resid(toks.unsqueeze(0))[0, 1:].float()
        f = sae.encode(H)
        dens += (f > 0).float().sum(0)
        ssum += f.sum(0)
        npos += H.shape[0]
        norms.append(H.norm(dim=-1))
    density = dens / npos
    scale = ssum / dens.clamp_min(1)
    norms = torch.cat(norms)
    stats = dict(F=Fm, H=Hm, density=density, scale=scale, resid_norm_median=float(norms.median()), npos=npos)
    torch.save(stats, out_path)
    return stats


def select_features(stats, cid: str, K: int, max_density: float = 0.05, eps: float = 1e-4):
    fam = cid.split(":")[0]
    others = [c["cid"] for c in all_concepts() if c["family"] == fam and c["cid"] != cid]
    Fc = stats["F"][cid]
    Fo = torch.cat([stats["F"][o] for o in others])
    mu_c, mu_o = Fc.mean(0), Fo.mean(0)
    var_c, var_o = Fc.var(0), Fo.var(0)
    score = (mu_c - mu_o) / torch.sqrt((var_c + var_o) / 2 + eps)
    score = torch.where((stats["density"] <= max_density) & (mu_c > 0), score, torch.full_like(score, -float("inf")))
    vals, idx = torch.topk(score, K)
    w = vals.clamp_min(1e-6)
    return idx, w / w.sum(), score


def diffmean_dir(stats, cid: str):
    fam = cid.split(":")[0]
    others = [c["cid"] for c in all_concepts() if c["family"] == fam and c["cid"] != cid]
    v = stats["H"][cid].mean(0) - torch.cat([stats["H"][o] for o in others]).mean(0)
    return v / v.norm()


def random_dir(d: int, cid: str, seed: int = 1234):
    g = torch.Generator().manual_seed(seed + sum(map(ord, cid)))
    v = torch.randn(d, generator=g)
    return v / v.norm()
