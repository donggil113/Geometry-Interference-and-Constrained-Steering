"""Quick sanity pilot on two dev concepts (not used for any selection decision beyond the alpha grid range)."""
import sys

import numpy as np
import torch

import n3  # noqa: F401
from n3.concepts import load_splits
from n3.data import owt_texts
from n3.evaluate import Condition, load_judges, run_condition
from n3.features import compute_setting_stats, select_features
from n3.settings import load_setting

torch.set_grad_enabled(False)
setting = sys.argv[1] if len(sys.argv) > 1 else "gpt2_relu_jb_L6"
model, sae = load_setting(setting)
stats = compute_setting_stats(model, sae, setting)
sp = load_splits()
prompts = sp["prompts"]["dev"][:12]
ptoks = model.to_tokens(prompts)
util_tokens = torch.stack([model.to_tokens([t])[0][:64] for t in owt_texts("utility")[:8] if len(model.to_tokens([t])[0]) >= 64][:8])
judges = load_judges()
for cid in ["topic:5", "emotion:1"]:
    T, w, score = select_features(stats, cid, 3)
    print(cid, "T", T.tolist(), "w", [round(x, 3) for x in w.tolist()], "score", [round(float(score[t]), 2) for t in T], "density", [round(float(stats['density'][t]), 4) for t in T])
    fam, lab = cid.split(":")
    for method, a in [("none", 0.0), ("dec", 0.5), ("dec", 1.0), ("diffmean", 0.5), ("diffmean", 1.0), ("enc", 1.0), ("pinv", 1.0), ("dec_proj", 1.0), ("dec_proj_fs", 1.0), ("opt", 1.0)]:
        cond = Condition(setting, cid, method, a, K=3)
        r = run_condition(model, sae, stats, cond, ptoks, util_tokens)
        probs = judges[fam].probs(r["conts"])[:, int(lab)]
        nll = judges["fluency"].cond_nll(prompts, r["conts"])
        it = r["internal"]
        print(f"  {method:12s} a={a:.2f} C={probs.mean():.3f} nll={nll.mean():.2f} dCE={r['utility']['dce']:.3f} "
              f"tgt={it['tgt_gain']:.2f} pRel={it['p_rel_change']:.3f} new={it['new_active_per_pos']:.1f} ms/pos={it['ms_per_pos']:.2f} gen={r['gen_secs']:.1f}s")
        print("     >", r["conts"][0][:150].replace("\n", " "))
