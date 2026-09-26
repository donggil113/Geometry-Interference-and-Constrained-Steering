"""H2: do realizability diagnostics (computed on neutral positions, never looking at behavior) predict per-concept
behavioral steering success -- better than geometry-only / frequency baselines?

usage: python scripts/10_h2_diagnosis.py <setting> <run_with_all_16_concepts> [<more runs> ...]
Protocol values (K, alpha per method, primary diagnostic, outcome method) are read from configs/protocol.json.
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import n3  # noqa: F401
from n3.data import ROOT, load_json, save_json
from n3.features import compute_setting_stats, select_features
from n3.settings import load_setting

setting = sys.argv[1]
runs = sys.argv[2:]
proto = load_json(ROOT / "configs" / "protocol.json")
K = proto["K_by_method"][proto["h2"]["outcome_method"]]
diag_df = pd.read_csv(ROOT / "results" / "realizability" / f"per_concept_{setting}_K{K}.csv").set_index("cid")

# behavioral outcomes (per concept, averaged over prompts) from analysis frames
frames = [pd.read_parquet(ROOT / "results" / "analysis" / r / "frame.parquet") for r in runs]
df = pd.concat(frames)
df = df[df.setting == setting]
outc = {}
for m in [proto["h2"]["outcome_method"], "dec", "diffmean"]:
    a, k = proto["alpha_by_method"][m], proto["K_by_method"][m]
    sub = df[(df.method == m) & np.isclose(df.alpha, a) & (df.K == k)]
    outc[m] = sub.groupby("cid")["dC"].mean()

# geometry-only baselines from the SAE weights (Khan et al.-style decoder crowding) and feature frequency
model, sae = load_setting(setting)
stats = compute_setting_stats(model, sae, setting)
Wd = sae.W_dec / sae.W_dec.norm(dim=1, keepdim=True)
geo = {}
for cid in diag_df.index:
    T, wT, _ = select_features(stats, cid, K)
    cos = Wd[T] @ Wd.T  # (K, m)
    cos[torch.arange(len(T)), T] = 0
    geo[cid] = dict(max_dec_cos=float((cos.abs().max(1).values * wT).sum()),
                    neighbor_density_03=float(((cos.abs() > 0.3).sum(1).float() * wT).sum()),
                    log_density=float((torch.log10(stats["density"][T].clamp_min(1e-7)) * wT).sum()))
geo = pd.DataFrame(geo).T

tab = diag_df.join(geo).join(pd.DataFrame({f"dC_{m}": v for m, v in outc.items()}))
tab = tab.dropna(subset=[f"dC_{proto['h2']['outcome_method']}"])
rng = np.random.RandomState(0)


def rho_ci(x, y, n=10000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r = spearmanr(x, y).correlation
    bs = []
    for _ in range(n):
        i = rng.randint(0, len(x), len(x))
        if len(set(i)) < 4:
            continue
        bs.append(spearmanr(x[i], y[i]).correlation)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(rho=float(r), lo=float(lo), hi=float(hi), n=int(len(x)))


res = {"setting": setting, "K": K, "n_concepts": int(len(tab)), "outcome_method": proto["h2"]["outcome_method"], "predictors": {}}
y = tab[f"dC_{proto['h2']['outcome_method']}"]
preds = proto["h2"]["diagnostics"] + ["max_dec_cos", "neighbor_density_03", "log_density", "enc_dec_cos", "dC_diffmean"]
for p in preds:
    if p in tab:
        res["predictors"][p] = rho_ci(tab[p], y)
prim = proto["h2"]["primary_diagnostic"]
base_best = max(abs(res["predictors"][p]["rho"]) for p in ["max_dec_cos", "neighbor_density_03", "log_density", "enc_dec_cos", "dC_diffmean"] if p in res["predictors"])
pr = res["predictors"][prim]
res["h2_supported"] = bool(abs(pr["rho"]) >= 0.5 and (pr["lo"] > 0 or pr["hi"] < 0) and abs(pr["rho"]) > base_best)
res["best_baseline_abs_rho"] = base_best
tab.to_csv(ROOT / "results" / "analysis" / f"h2_table_{setting}.csv")
save_json(res, ROOT / "results" / "analysis" / f"h2_{setting}.json")
print(json.dumps(res, indent=1))
