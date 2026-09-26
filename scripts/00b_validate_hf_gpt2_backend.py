"""Check that the fast HF GPT-2 backend (block-5 output, mean-centred) reproduces TransformerLens
blocks.6.hook_resid_pre (center_writing_weights=True) activations, SAE codes, and edited logits."""
import torch

import n3  # noqa: F401
from n3.data import ROOT, owt_texts, save_json
from n3.features import padded_resid
from n3.methods import Editor, EditSpec
from n3.settings import _model, _sae

torch.set_grad_enabled(False)
hf, tl = _model("gpt2"), _model("gpt2_tl")
texts = owt_texts("validation")[:8]
Hh, lh = padded_resid(hf, texts, 64)
Ht, lt = padded_resid(tl, texts, 64)
assert torch.equal(lh, lt)
mask = torch.arange(Hh.shape[1])[None, :] < lh[:, None]
mask[:, 0] = False
rep = {"resid_max_abs_diff": float((Hh - Ht)[mask].abs().max()), "resid_rel_diff": float((Hh - Ht)[mask].norm() / Ht[mask].norm())}
for name in ["gpt2_relu_jb_L6", "gpt2_topk_oai_L6"]:
    sae = _sae(name)
    fh, ft = sae.encode(Hh[mask]), sae.encode(Ht[mask])
    rep[f"{name}_active_set_agree"] = float(((fh > 0) == (ft > 0)).float().mean())
    rep[f"{name}_f_max_abs_diff"] = float((fh - ft).abs().max())
    spec = EditSpec("dec", T=torch.tensor([16960]), wT=torch.ones(1), alpha=40.0)
    toks = torch.stack([hf.to_tokens([t])[0][:48] for t in texts])
    lh_ = torch.log_softmax(hf.logits(toks, Editor(sae, spec, collect=False)).float(), -1)
    lt_ = torch.log_softmax(tl.logits(toks, Editor(sae, spec, collect=False)).float(), -1)
    rep[f"{name}_edited_logprob_max_abs_diff"] = float((lh_ - lt_)[:, 1:].abs().max())
    rep[f"{name}_edited_kl"] = float((lt_.exp() * (lt_ - lh_)).sum(-1)[:, 1:].mean())
print(rep)
save_json(rep, ROOT / "results" / "validation" / "hf_vs_tl_gpt2_backend.json")
