"""Tier-3 cost: edit-computation time per position and end-to-end generation overhead, on an otherwise idle CPU.

usage: python scripts/14_runtime.py [setting]
Writes results/runtime/<setting>.json.  Every method uses its dev-frozen (config, alpha) from configs/protocol.json.
"""
import sys
import time

import torch

import n3  # noqa: F401
from n3.concepts import load_splits
from n3.data import ROOT, load_json, owt_texts, save_json
from n3.evaluate import Condition, build_editor, equal_length_prompts
from n3.features import compute_setting_stats, padded_resid
from n3.settings import load_setting

torch.set_grad_enabled(False)
setting = sys.argv[1] if len(sys.argv) > 1 else "gpt2_relu_jb_L6"
proto = load_json(ROOT / "configs" / "protocol.json")
model, sae = load_setting(setting)
stats = compute_setting_stats(model, sae, setting)
H, lens = padded_resid(model, owt_texts("validation")[300:340], 64)
pos = torch.cat([H[b, 1:lens[b]] for b in range(H.shape[0])])[:1024]
prompts = load_splits()["prompts"]["test"][:24]
ptoks, _ = equal_length_prompts(model, prompts)
cid = load_splits()["test_concepts"][0]
out = {"setting": setting, "threads": torch.get_num_threads(), "methods": {}}


def cond_for(m):
    cfg = proto["cfg_by_method"][m].split("|")
    kw = dict(K=proto["K_by_method"][m])
    for tok in cfg[1:]:
        if tok.startswith("fs"):
            kw["fs_rounds"] = int(tok[2:])
        elif tok.startswith("r"):
            kw["ridge_rel"] = float(tok[1:])
    return Condition(setting, cid, m, proto["alpha_by_method"][m] or 0.3, **kw)


for m in ["dec", "enc", "pinv", "ridge", "dec_proj", "pinv_fs", "dec_proj_fs", "opt", "diffmean", "random"]:
    c = cond_for(m)
    res = {}
    for n in (24, 1024):
        ed = build_editor(sae, stats, c, collect=False)
        ed(pos[:n])  # warm-up
        reps = 5 if n == 24 else 2
        t0 = time.perf_counter()
        for _ in range(reps):
            ed(pos[:n])
        res[f"ms_per_pos_n{n}"] = 1000 * (time.perf_counter() - t0) / reps / n
    t0 = time.perf_counter()
    model.generate(ptoks, build_editor(sae, stats, c, collect=False), max_new_tokens=40, seed=0)
    res["gen_secs_24x40"] = time.perf_counter() - t0
    out["methods"][m] = res
    print(m, {k: round(v, 3) for k, v in res.items()}, flush=True)
t0 = time.perf_counter()
model.generate(ptoks, None, max_new_tokens=40, seed=0)
out["gen_secs_24x40_unsteered"] = time.perf_counter() - t0
save_json(out, ROOT / "results" / "runtime" / f"{setting}.json")
print("unsteered", out["gen_secs_24x40_unsteered"])
