"""Step 4-5: finite-step realizability analysis of target/protected feature edits on neutral positions.

For every concept (dev + held-out; this analysis never looks at behavior) and K in {1, 3}:
sample positions from OWT 'validation' documents, compute per position
  c_free, c_lin, kappa = c_lin / c_free, finite-step violations of the linear solution, exact QP cost
  (ReLU / JumpReLU) or Gauss-Newton + true-forward checks (TopK + LN), decoder-steering cost and collateral.
Target change Delta_t = typical nonzero activation of feature t on OWT ('scale').
Budget for 'within budget' flags = 1.0 x median residual norm.
Writes results/realizability/<setting>.jsonl and a summary json.
"""
import json
import sys
import time

import numpy as np
import torch

import n3  # noqa: F401
from n3.concepts import all_concepts
from n3.data import ROOT, load_df, owt_texts, save_json
from n3.features import compute_setting_stats, padded_resid, select_features
from n3.realizability import analyze_position_affine, analyze_position_topk_ln
from n3.settings import load_setting

torch.set_grad_enabled(False)
setting = sys.argv[1]
n_pos = int(sys.argv[2]) if len(sys.argv) > 2 else 96
Ks = [1, 3]
model, sae = load_setting(setting)
stats = compute_setting_stats(model, sae, setting)
budget = 1.0 * stats["resid_norm_median"]

texts = owt_texts("validation")[200:260] if not setting.startswith("gemma") else load_df("pile10k")["text"].iloc[400:460].tolist()
H, lens = padded_resid(model, texts, 64)
rng = np.random.RandomState(0)
pos = []
for b in range(H.shape[0]):
    L = int(lens[b])
    if L < 16:
        continue
    for p in rng.choice(np.arange(4, L), size=3, replace=False):
        pos.append(H[b, p])
pos = torch.stack(pos)[:n_pos]
print(setting, "positions", pos.shape, "budget", budget, flush=True)

out_path = ROOT / "results" / "realizability" / f"{setting}.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
done = set()
if out_path.exists():
    done = {(json.loads(l)["cid"], json.loads(l)["K"], json.loads(l)["pos"]) for l in open(out_path)}
t0 = time.time()
with open(out_path, "a") as fout:
    for c in all_concepts():
        for K in Ks:
            T, wT, _ = select_features(stats, c["cid"], K)
            dT = (wT.unsqueeze(-1) * sae.W_dec[T]).sum(0)
            Delta = stats["scale"][T].clamp_min(1e-3)
            for i in range(pos.shape[0]):
                if (c["cid"], K, i) in done:
                    continue
                if sae.kind == "topk":
                    r = analyze_position_topk_ln(sae, pos[i], T, dT, Delta, budget=budget)
                else:
                    r = analyze_position_affine(sae, pos[i], T, dT, Delta, budget=budget)
                r.update(cid=c["cid"], K=K, pos=i, T=T.tolist(), resid_norm=float(pos[i].norm()))
                fout.write(json.dumps(r) + "\n")
            fout.flush()
            print(c["cid"], K, f"{time.time()-t0:.0f}s", flush=True)
