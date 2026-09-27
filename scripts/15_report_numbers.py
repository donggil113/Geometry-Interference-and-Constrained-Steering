"""Recompute every number quoted in docs/04_report.md from the analysis frames -> results/analysis/report_numbers.json.

usage: python scripts/15_report_numbers.py
"""
import json

import numpy as np
import pandas as pd

import n3  # noqa: F401
from n3.analysis import cluster_bootstrap_diff, condition_table, interp_at_budget
from n3.data import ROOT, load_json, save_json

proto = load_json(ROOT / "configs" / "protocol.json")
prim = proto["primary_corrected"]
out = {}


def arm_rows(df, m, alpha=None):
    a = proto["alpha_by_method"][m] if alpha is None else alpha
    return df[(df.cfg == proto["cfg_by_method"][m]) & np.isclose(df.alpha, a)]


def levels(df, methods, cols):
    res = {}
    for m in methods:
        sub = arm_rows(df, m)
        if len(sub):
            g = sub.groupby("cid")[[c for c in cols if c in sub]].mean()
            res[m] = {c: float(g[c].mean()) for c in g.columns} | {"n_concepts": int(g.shape[0])}
    return res


def contrast(df, m1, m2, metric, a1=None, a2=None):
    if metric not in df:
        return None
    s1, s2 = arm_rows(df, m1, a1), arm_rows(df, m2, a2)
    if not len(s1) or not len(s2):
        return None
    fa = pd.concat([s1, s2])
    try:
        r = cluster_bootstrap_diff(fa, {"cfg": proto["cfg_by_method"][m1]}, {"cfg": proto["cfg_by_method"][m2]}, metric=metric)
    except ValueError:  # e.g. NLI scored only for the decision arms
        return None
    return {k: r[k] for k in ["est", "lo", "hi", "p_signflip", "n_concepts_positive", "n_concepts"]}


COLS = ["dC", "C", "dNLL", "dRel", "protJS", "dDist2", "dce", "kl", "down_p_rel_drift", "down_new_active_per_pos", "nli",
        "int_tgt_gain", "int_p_rel_change", "int_new_active_per_pos", "int_err_delta_frac", "int_ms_per_pos"]
ALL = ["dec_proj", "dec", "dec_proj_randP", "diffmean", "enc", "dec_rand_feat", "random", "prompt", "pinv", "ridge",
       "dec_proj_fs", "pinv_fs", "opt"]

for run, setting in [("test1", "gpt2_relu_jb_L6"), ("topk1", "gpt2_topk_oai_L6")]:
    p = ROOT / "results" / "analysis" / run / "frame.parquet"
    if not p.exists():
        continue
    df = pd.read_parquet(p)
    df = df[df.setting == setting]
    o = {"levels": levels(df, ALL, COLS), "contrasts": {}, "by_family_vs_dec": {}}
    for m2 in ["dec", "diffmean", "enc", "random", "dec_proj_randP"]:
        for metric in ["dC", "dNLL", "protJS", "dRel", "dce", "kl", "nli", "down_p_rel_drift", "down_new_active_per_pos", "dDist2"]:
            if metric in df:
                r = contrast(df, prim, m2, metric)
                if r:
                    o["contrasts"][f"{prim}-{m2}:{metric}"] = r
    for m1, m2 in [("dec", "dec_rand_feat"), ("prompt", prim), ("prompt", "diffmean"), ("dec_rand_feat", "random")]:
        r = contrast(df, m1, m2, "dC")
        if r:
            o["contrasts"][f"{m1}-{m2}:dC"] = r
    # other corrected arms vs dec at their frozen points (exploratory)
    for m1 in ["pinv", "ridge", "dec_proj_fs", "pinv_fs", "opt", "enc"]:
        for metric in ["dC", "dNLL", "dce"]:
            r = contrast(df, m1, "dec", metric)
            if r:
                o["contrasts"][f"{m1}-dec:{metric}"] = r
    # control vs primary, extra readouts
    for metric in [m for m in ["down_new_active_per_pos", "kl", "protJS", "dce"] if m in df]:
        r = contrast(df, prim, "dec_proj_randP", metric)
        if r:
            o["contrasts"][f"{prim}-dec_proj_randP:{metric}"] = r
    # post hoc penalized score dC - lam * dNLL vs diffmean
    s1, s2 = arm_rows(df, prim), arm_rows(df, "diffmean")
    if len(s1) and len(s2):
        for lam in [0.12, 0.25, 0.35, 0.46]:
            fa = pd.concat([s1, s2]).copy()
            fa["pen"] = fa["dC"] - lam * fa["dNLL"]
            r = cluster_bootstrap_diff(fa, {"cfg": proto["cfg_by_method"][prim]}, {"cfg": proto["cfg_by_method"]["diffmean"]}, metric="pen")
            o["contrasts"][f"posthoc_penalized_lambda{lam}:{prim}-diffmean"] = {k: r[k] for k in ["est", "lo", "hi", "p_signflip", "n_concepts_positive", "n_concepts"]}
    pa = proto["alpha_by_method"][prim]
    for m2 in ["dec", "diffmean", "enc", "random"]:
        for metric in ["dC", "dNLL", "dce", "protJS"]:
            r = contrast(df, prim, m2, metric, a1=pa, a2=pa)
            if r:
                o["contrasts"][f"normmatched:{prim}-{m2}:{metric}"] = r
    for fam in ["topic", "emotion", "entity"]:
        sub = df[df.family == fam]
        for m2 in ["dec", "diffmean"]:
            r = contrast(sub, prim, m2, "dC")
            if r:
                o["by_family_vs_dec"][f"{fam}:{prim}-{m2}"] = r
    tab = condition_table(df)
    mf = {}
    for m in [prim, "dec", "diffmean"]:
        t = tab[tab.cfg == proto["cfg_by_method"][m]]
        if t.alpha.nunique() >= 3:
            mf[m] = {"curve": t.sort_values("alpha")[["alpha", "dC", "dNLL", "dce", "n_concepts"]].to_dict("records"),
                     "dC_at_dNLL": {str(b): interp_at_budget(t, b) for b in [0.25, 0.44, 0.5, 0.875, 1.0]}}
    o["matched_fluency"] = mf
    out[run] = o

for s in ["gpt2_relu_jb_L6", "gpt2_topk_oai_L6", "gemma3_270m_jumprelu_L12"]:
    p = ROOT / "results" / "realizability" / f"summary_{s}.json"
    if p.exists():
        out[f"realizability_{s}"] = load_json(p)
for s in ["gpt2_relu_jb_L6"]:
    p = ROOT / "results" / "analysis" / f"h2_{s}.json"
    if p.exists():
        out[f"h2_{s}"] = load_json(p)
    # sensitivity (amendment A6): the section-6 outcome `dec` instead of the frozen `dec_proj`
    t = ROOT / "results" / "analysis" / f"h2_table_{s}.csv"
    if t.exists():
        from scipy.stats import spearmanr
        tab = pd.read_csv(t, index_col=0)
        d = proto["h2"]["primary_diagnostic"]
        ok = tab[[d, "dC_dec"]].replace([np.inf, -np.inf], np.nan).dropna()
        out[f"h2_{s}_dec_outcome"] = {"diagnostic": d, "rho": float(spearmanr(ok[d], ok["dC_dec"]).correlation), "n": int(len(ok))}
p = ROOT / "results" / "runtime" / "gpt2_relu_jb_L6.json"
if p.exists():
    out["runtime"] = load_json(p)
save_json(out, ROOT / "results" / "analysis" / "report_numbers.json")
print(json.dumps({k: list(v.keys()) if isinstance(v, dict) else v for k, v in out.items()}, indent=1)[:2000])
