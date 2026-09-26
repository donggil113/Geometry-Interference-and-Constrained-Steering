"""Run a grid of steering conditions and store generations + internal + utility metrics.

usage: python scripts/04_sweep.py <run_name> <setting> <split: dev|test> <grid json>
grid json keys: methods, alphas, Ks, concepts (optional; default = split concepts), extra (dict of Condition
overrides, applied to all SAE methods), generic_methods (methods run once per alpha independent of K).
Resumable: conditions already present in the output file are skipped.
"""
import json
import sys
import time

import torch

import n3  # noqa: F401
from n3.concepts import load_splits
from n3.data import ROOT, owt_texts
from n3.evaluate import Condition, append_row, code_fingerprint, done_keys, equal_length_prompts, run_condition
from n3.features import compute_setting_stats
from n3.settings import load_setting

torch.set_grad_enabled(False)
run, setting, split, grid = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
model, sae = load_setting(setting)
stats = compute_setting_stats(model, sae, setting)
sp = load_splits()
prompts = sp["prompts"][split]
concepts = grid.get("concepts") or sp[f"{split}_concepts"]
ptoks, prompts = equal_length_prompts(model, prompts)
assert ptoks.shape[1] == sp["prompt_tokens"] + 1 or setting.startswith("gemma"), ptoks.shape
assert (ptoks[:, 0] == model.bos_id).all()
util = []
for t in owt_texts("utility"):
    tk = model.to_tokens([t])[0]
    if len(tk) >= 64:
        util.append(tk[:64])
    if len(util) == 16:
        break
util_tokens = torch.stack(util)
part = grid.get("part", "")
down = None
if grid.get("downstream"):  # independent internal readout: (capture layer, SAE release, SAE id)
    from n3.saes import load_sae
    lay, rel, sid = grid["downstream"]
    d_sae = load_sae(rel, sid)[0]
    d_sae.center_input = setting.startswith("gpt2")
    down = (lay, d_sae)
out = ROOT / "results" / "runs" / run / f"{setting}__{split}{('__' + part) if part else ''}.jsonl"
code = code_fingerprint()
have = done_keys(out, code)

conds = [Condition(setting, concepts[0], "none", 0.0)]
extra = grid.get("extra", {})
for cid in concepts:
    for a in grid["alphas"]:
        for m in grid.get("generic_methods", []):
            conds.append(Condition(setting, cid, m, a, K=1))
        for K in grid["Ks"]:
            for m in grid["methods"]:
                conds.append(Condition(setting, cid, m, a, K=K, **extra))
for spec in grid.get("explicit", []):  # explicit per-method operating points, e.g. frozen (method, K, alpha, extras)
    for cid in concepts:
        conds.append(Condition(setting, cid, **spec))
seen = set()
conds = [c for c in conds if not (c.key() in seen or seen.add(c.key()))]
todo = [c for c in conds if c.key() not in have]
print(f"{run} {setting} {split}: {len(conds)} conditions, {len(todo)} to run", flush=True)
t0 = time.time()
for i, c in enumerate(todo):
    r = run_condition(model, sae, stats, c, ptoks, util_tokens, down=down)
    r["split"] = split
    r["prompts"] = prompts
    r["code"] = code
    append_row(out, r)
    if i % 10 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(todo)} {c.cid} {c.method} a={c.alpha_mult} K={c.K} gen={r['gen_secs']:.1f}s "
              f"elapsed={el/60:.1f}m eta={(el/(i+1))*(len(todo)-i-1)/60:.1f}m", flush=True)
print("done", flush=True)
