"""Step 2: pin model / SAE checkpoints, hooks and normalization; validate them empirically.

Writes configs/pinned.json and results/validation/checkpoint_validation.json.

Checks per SAE setting
  * our re-implemented encoder == SAELens encoder (max abs diff, active-set agreement)
  * FVU and L0 on held-out OpenWebText / Pile text (BOS excluded) vs. the published L0
  * CE loss: clean, SAE-reconstruction spliced in, zero-ablation -> fraction of CE recovered
  * GPT-2: blocks.5.hook_resid_post == blocks.6.hook_resid_pre (so ReLU and TopK SAEs share one site)
"""
from __future__ import annotations

import time

import torch
from huggingface_hub import HfApi

import n3  # noqa: F401  (sets HF_HOME)
from n3.data import DATA_FILES, ROOT, owt_texts, load_df, save_json
from n3.models import HFModel, TLModel
from n3.saes import check_against_saelens, load_sae

torch.set_grad_enabled(False)
api = HfApi()

SETTINGS = {
    "gpt2_relu_jb_L6": dict(model="gpt2", backend="tl", release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre",
                            sae_repo="jbloom/GPT2-Small-SAEs-Reformatted", hook="blocks.6.hook_resid_pre", published_l0=None),
    "gpt2_topk_oai_L6": dict(model="gpt2", backend="tl", release="gpt2-small-resid-post-v5-32k", sae_id="blocks.5.hook_resid_post",
                             sae_repo="jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs", hook="blocks.6.hook_resid_pre", published_l0=32),
    "gemma3_270m_jumprelu_L12": dict(model="unsloth/gemma-3-270m", backend="hf", release="gemma-scope-2-270m-pt-res",
                                     sae_id="layer_12_width_16k_l0_medium", sae_repo="google/gemma-scope-2-270m-pt", layer=12,
                                     published_l0=60),
}


def repo_sha(repo, repo_type="model"):
    info = api.repo_info(repo, repo_type=repo_type)
    return info.sha


def tokens_from_texts(model, texts, seq_len, n):
    rows = []
    for t in texts:
        toks = model.to_tokens([t], prepend_bos=True)[0]
        if toks.shape[0] >= seq_len:
            rows.append(toks[:seq_len])
        if len(rows) == n:
            break
    return torch.stack(rows)


def ce_from_logits(logits, tokens):
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = tokens[:, 1:]
    return -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()


def validate(setting_name, cfg, model, texts):
    w, sae = load_sae(cfg["release"], cfg["sae_id"], setting_name)
    toks = tokens_from_texts(model, texts, seq_len=128, n=24)
    H = model.resid(toks)  # (b, s, d)
    h = H[:, 1:].reshape(-1, H.shape[-1]).float()
    agree = check_against_saelens(w, sae, h[:2048])
    f = w.encode(h)
    rec = w.decode(f, h)
    fvu = ((h - rec) ** 2).sum() / ((h - h.mean(0)) ** 2).sum()
    l0 = (f > 0).float().sum(-1)
    norms = h.norm(dim=-1)

    clean = ce_from_logits(model.logits(toks), toks)
    splice = ce_from_logits(model.logits(toks, editor=lambda x: w.decode(w.encode(x), x) - x), toks)
    zero = ce_from_logits(model.logits(toks, editor=lambda x: -x), toks)
    rec_frac = (zero - splice) / (zero - clean)
    out = dict(
        sae_kind=w.kind, normalization=w.norm, apply_b_dec_to_input=w.apply_b_dec_to_input, d=w.d, m=w.m, k=w.k,
        saelens_hook=w.hook, edit_hook=model.hook, **agree,
        fvu=float(fvu), l0_mean=float(l0.mean()), l0_median=float(l0.median()), published_l0=cfg.get("published_l0"),
        resid_norm_median=float(norms.median()), resid_norm_p10=float(norms.quantile(0.1)), resid_norm_p90=float(norms.quantile(0.9)),
        ce_clean=clean, ce_spliced=splice, ce_zero_ablation=zero, ce_recovered=float(rec_frac),
        dec_row_norm_min=float(w.W_dec.norm(dim=1).min()), dec_row_norm_max=float(w.W_dec.norm(dim=1).max()),
        n_tokens=int(h.shape[0]),
    )
    return out


def main():
    t0 = time.time()
    pinned = {"date": "2026-09-26", "settings": {}, "data_files": {k: dict(repo=v[0], revision=v[1], path=v[2]) for k, v in DATA_FILES.items()}}
    report = {}

    gpt2 = TLModel("gpt2", hook="blocks.6.hook_resid_pre")
    gpt2_sha = repo_sha("openai-community/gpt2")
    texts = owt_texts("validation")
    toks = tokens_from_texts(gpt2, texts, 64, 8)
    a = gpt2.resid(toks, "blocks.5.hook_resid_post")
    b = gpt2.resid(toks, "blocks.6.hook_resid_pre")
    report["gpt2_resid_post5_equals_resid_pre6_maxdiff"] = float((a - b).abs().max())

    for name in ["gpt2_relu_jb_L6", "gpt2_topk_oai_L6"]:
        cfg = SETTINGS[name]
        report[name] = validate(name, cfg, gpt2, texts)
        pinned["settings"][name] = dict(
            model="gpt2 (HF openai-community/gpt2)", model_revision=gpt2_sha, backend="transformer_lens HookedTransformer.from_pretrained('gpt2') default processing (fold_ln, center_writing_weights, center_unembed)",
            sae_release=cfg["release"], sae_id=cfg["sae_id"], sae_repo=cfg["sae_repo"], sae_repo_revision=repo_sha(cfg["sae_repo"]),
            edit_hook=cfg["hook"], sae_kind=report[name]["sae_kind"], normalization=report[name]["normalization"],
            prepend_bos=True, bos_position_edited=False,
        )
        print(name, report[name], flush=True)
    del gpt2

    cfg = SETTINGS["gemma3_270m_jumprelu_L12"]
    g_sha = repo_sha(cfg["model"])
    gem = HFModel(cfg["model"], revision=g_sha, layer=cfg["layer"])
    ptexts = load_df("pile10k")["text"].iloc[:400].tolist()
    name = "gemma3_270m_jumprelu_L12"
    report[name] = validate(name, cfg, gem, ptexts)
    pinned["settings"][name] = dict(
        model=cfg["model"] + " (ungated mirror of google/gemma-3-270m; validated via SAE reconstruction below)", model_revision=g_sha,
        backend="transformers AutoModelForCausalLM float32; forward hook on model.model.layers[12] output",
        sae_release=cfg["release"], sae_id=cfg["sae_id"], sae_repo=cfg["sae_repo"], sae_repo_revision=repo_sha(cfg["sae_repo"]),
        edit_hook="model.layers.12.output", sae_kind=report[name]["sae_kind"], normalization=report[name]["normalization"],
        prepend_bos=True, bos_position_edited=False,
    )
    print(name, report[name], flush=True)

    report["elapsed_s"] = time.time() - t0
    save_json(pinned, ROOT / "configs" / "pinned.json")
    save_json(report, ROOT / "results" / "validation" / "checkpoint_validation.json")


if __name__ == "__main__":
    main()
