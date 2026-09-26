"""Apply the pre-registered H1 decision rule (all thresholds read from configs/protocol.json).

usage: python scripts/11_decide.py <test_run_name> <setting>
Primary endpoint: dC of PRIMARY_CORRECTED vs each comparator, each at its dev-frozen config and alpha, paired on
(concept, prompt).  Primary inference: exact concept-level sign-flip test, Holm across comparators; the two-way
cluster-bootstrap CI must also exclude 0.  Guards: fluency (dNLL), degeneration (dDist2), NLI-judge concordance,
behavioral collateral vs dec.  Everything else (matched-fluency curves, downstream drift, controls) is reported.
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
rule = proto["h1_rule"]
df = pd.read_parquet(ROOT / "results" / "analysis" / run / "frame.parquet")
df = df[df.setting == setting]
prim = proto["primary_corrected"]
comps = proto["h1_comparators"]


def sel(m):
    return dict(cfg=proto["cfg_by_method"][m], alpha=float(proto["alpha_by_method"][m]))


def subset(m):
    s = sel(m)
    return df[(df.cfg == s["cfg"]) & np.isclose(df.alpha, s["alpha"])]


res = {"run": run, "setting": setting, "primary": prim, "rule": rule, "comparisons": {}, "levels": {}}
for m in [prim] + comps + proto.get("controls", []) + proto.get("reference_methods", []):
    if m not in proto["cfg_by_method"]:
        continue
    sub = subset(m)
    if len(sub) == 0:
        res["levels"][m] = {"missing": True}
        continue
    cols = [c for c in ["dC", "C", "dNLL", "dRel", "protJS", "dDist2", "dce", "kl", "down_p_rel_drift", "down_new_active_per_pos",
                        "nli", "int_tgt_gain", "int_p_rel_change", "int_new_active_per_pos", "int_err_delta_frac",
                        "int_ms_per_pos"] if c in sub]
    g = sub.groupby("cid")[cols].mean()  # concept-balanced
    res["levels"][m] = {c: float(g[c].mean()) for c in cols} | {"n_rows": int(len(sub)), "n_concepts": int(g.shape[0]), **sel(m)}

missing = [m for m in [prim] + comps if res["levels"].get(m, {}).get("missing", m not in res["levels"])]
if missing:
    raise SystemExit(f"missing arms {missing}: the decision cannot be made (intent-to-treat: fix the run, do not drop arms)")

pv = {}
for c in comps:
    a, b = sel(prim), sel(c)
    fa = df[((df.cfg == a["cfg"]) & np.isclose(df.alpha, a["alpha"])) | ((df.cfg == b["cfg"]) & np.isclose(df.alpha, b["alpha"]))]
    comp = {}
    for metric in ["dC", "dNLL", "dDist2", "protJS", "dRel"] + (["nli"] if "nli" in fa and fa["nli"].notna().any() else []):
        comp[metric] = cluster_bootstrap_diff(fa, {"cfg": a["cfg"]}, {"cfg": b["cfg"]}, metric=metric)
    res["comparisons"][c] = comp
    pv[c] = comp["dC"]["p_signflip"]
rej = holm(pv, alpha=rule["alpha"])
checks = {}
for c in comps:
    r = res["comparisons"][c]
    checks[c] = dict(
        holm_reject=rej[c],
        direction_positive=r["dC"]["est"] > 0,
        bootstrap_lo_gt0=(r["dC"]["lo"] > 0) if rule.get("require_bootstrap_lo_gt0", True) else True,
        fluency_guard=r["dNLL"]["est"] <= rule["fluency_guard_max_dNLL_excess"],
        degeneration_guard=r["dDist2"]["est"] >= rule["deg_guard_min_dDist2_diff"],
        nli_concordance=(r["nli"]["est"] > 0) if (rule.get("nli_concordance") and "nli" in r) else (not rule.get("nli_concordance")),
    )
    checks[c]["pass"] = all(checks[c].values())
L = res["levels"]
col = rule["collateral"]
ref = L[col["vs"]]
collateral_ok = (L[prim]["protJS"] <= ref["protJS"] + col["protJS_max_excess"] and
                 L[prim]["dce"] <= ref["dce"] + col["dce_max_excess"] and
                 L[prim]["dRel"] >= ref["dRel"] - col["dRel_max_deficit"])
res["checks"] = checks
res["collateral_not_worse_than_" + col["vs"]] = bool(collateral_ok)
res["H1_GO"] = bool(all(v["pass"] for v in checks.values()) and collateral_ok)

# robustness (reported, not part of the rule): matched-fluency interpolation along alpha curves
tab = condition_table(df)
rob = {}
for m in [prim] + comps:
    t = tab[tab.cfg == proto["cfg_by_method"][m]]
    if t.alpha.nunique() >= 3:
        rob[m] = {str(b): interp_at_budget(t, b) for b in [0.5, 1.0, 1.5]}
res["matched_fluency_dC"] = rob
save_json(res, ROOT / "results" / "analysis" / run / f"decision_{setting}.json")
print(json.dumps({"H1_GO": res["H1_GO"], "checks": checks, "collateral_ok": bool(collateral_ok), "matched_fluency_dC": rob}, indent=1))
for c in comps:
    r = res["comparisons"][c]["dC"]
    print(f"{prim} - {c}: dC {r['est']:+.3f} boot95 [{r['lo']:+.3f}, {r['hi']:+.3f}] signflip p={r['p_signflip']:.4f} "
          f"({r['n_concepts_positive']}/{r['n_concepts']} concepts > 0)")
