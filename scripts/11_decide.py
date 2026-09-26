"""Apply the pre-registered H1 decision rule to a held-out run (reads configs/protocol.json).

usage: python scripts/11_decide.py <test_run_name> <setting>
Primary endpoint: dC of PRIMARY_CORRECTED vs each comparator, each at its dev-frozen (K, alpha); paired two-way
cluster bootstrap over (concept, prompt); Holm across comparators.  GO iff all comparisons reject with est > 0,
fluency guard (dNLL(primary) - dNLL(comp) <= 0.1) holds, and behavioral collateral is not worse than dec.
Robustness (reported): matched-fluency interpolation, NLI judge, downstream-layer drift.
"""
import json
import sys

import numpy as np
import pandas as pd

import n3  # noqa: F401
from n3.analysis import cluster_bootstrap_diff, condition_table, holm, interp_at_budget
from n3.data import ROOT, load_json, save_json

run, setting = sys.argv[1], sys.argv[2]
proto = load_json(ROOT / "configs" / "protocol.json")
df = pd.read_parquet(ROOT / "results" / "analysis" / run / "frame.parquet")
df = df[df.setting == setting]
prim = proto["primary_corrected"]
comps = proto["h1_comparators"]


def sel(m):
    return dict(method=m, K=proto["K_by_method"][m], alpha=proto["alpha_by_method"][m])


def subset(m):
    s = sel(m)
    return df[(df.method == m) & (df.K == s["K"]) & np.isclose(df.alpha, s["alpha"])]


res = {"run": run, "setting": setting, "primary": prim, "comparisons": {}, "levels": {}}
for m in [prim] + comps + proto.get("reference_methods", []):
    sub = subset(m)
    if len(sub) == 0:
        continue
    cols = [c for c in ["dC", "C", "dNLL", "dRel", "protJS", "dDist2", "dce", "kl", "down_p_rel_drift", "nli",
                        "int_tgt_gain", "int_p_rel_change", "int_new_active_per_pos", "int_err_delta_frac", "int_ms_per_pos"] if c in sub]
    res["levels"][m] = {c: float(sub[c].mean()) for c in cols} | {"n": int(len(sub)), **sel(m)}

pv = {}
for c in comps:
    a, b = sel(prim), sel(c)
    fa = df.copy()
    # align selections on the frame: build a column 'arm'
    fa = fa[((fa.method == prim) & (fa.K == a["K"]) & np.isclose(fa.alpha, a["alpha"])) |
            ((fa.method == c) & (fa.K == b["K"]) & np.isclose(fa.alpha, b["alpha"]))]
    r = cluster_bootstrap_diff(fa, {"method": prim}, {"method": c}, metric="dC")
    r_nll = cluster_bootstrap_diff(fa, {"method": prim}, {"method": c}, metric="dNLL")
    res["comparisons"][c] = dict(dC=r, dNLL=r_nll)
    if "nli" in fa:
        res["comparisons"][c]["nli"] = cluster_bootstrap_diff(fa, {"method": prim}, {"method": c}, metric="nli")
    pv[c] = r["p_two_sided"]
rej = holm(pv)
fluency_ok = {c: res["comparisons"][c]["dNLL"]["est"] <= 0.1 for c in comps}
win = {c: rej[c] and res["comparisons"][c]["dC"]["est"] > 0 for c in comps}
L = res["levels"]
collateral_ok = all(L[prim].get(k, 0) <= L["dec"].get(k, 0) + tol for k, tol in [("protJS", 0.01), ("dce", 0.05)]) and \
    L[prim].get("dRel", 0) >= L["dec"].get("dRel", 0) - 0.02
res["holm_reject"] = rej
res["fluency_guard_ok"] = fluency_ok
res["collateral_not_worse_than_dec"] = bool(collateral_ok)
res["H1_GO"] = bool(all(win.values()) and all(fluency_ok.values()) and collateral_ok)

# robustness: matched-fluency interpolation along full alpha curves (where available)
tab = condition_table(df, by=("method", "alpha", "K"))
rob = {}
for m in [prim] + comps:
    k = proto["K_by_method"][m]
    t = tab[(tab.method == m) & (tab.K == k)]
    if t.alpha.nunique() >= 3:
        rob[m] = {str(b): interp_at_budget(t, b) for b in [0.5, 1.0, 1.5]}
res["matched_fluency_dC"] = rob
save_json(res, ROOT / "results" / "analysis" / run / f"decision_{setting}.json")
print(json.dumps({k: v for k, v in res.items() if k != "comparisons"}, indent=1))
for c in comps:
    r = res["comparisons"][c]
    print(f"{prim} - {c}: dC {r['dC']['est']:+.3f} [{r['dC']['lo']:+.3f}, {r['dC']['hi']:+.3f}] p={r['dC']['p_two_sided']:.4f}  "
          f"dNLL {r['dNLL']['est']:+.3f} [{r['dNLL']['lo']:+.3f}, {r['dNLL']['hi']:+.3f}]")
