"""Aggregate a scored run: per-condition tables, budgeted (fluency-constrained) selection, internal vs behavior.

usage: python scripts/07_analyze.py <run_name> [budget_nll=1.0]
Writes results/analysis/<run>/{frame.parquet, table.csv, budget_*.csv, summary.json}.
Each run file (part) carries its own unsteered baseline row (same process, same seed).
"""
import json
import sys

import numpy as np
import pandas as pd

import n3  # noqa: F401
from n3.analysis import budget_select, condition_table, per_prompt_frame
from n3.data import ROOT, save_json
from n3.evaluate import load_rows

run = sys.argv[1]
B = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
rdir = ROOT / "results" / "runs" / run
frames = []
for path in sorted(rdir.glob("*.jsonl")):
    if path.name.endswith(".scores.jsonl"):
        continue
    rows = load_rows(path)
    sp = path.with_suffix(".scores.jsonl")
    if not sp.exists():
        print("unscored", path.name)
        continue
    scores = {}
    for l in open(sp):
        r = json.loads(l)
        scores[r["key"]] = r["scores"]
    base = [r for r in rows if r["cond"]["method"] == "none"]
    assert len(base) == 1, path
    df = per_prompt_frame([r for r in rows if r["cond"]["method"] != "none"], scores, base[0]["key"])
    df["part"] = path.stem
    frames.append(df)
df = pd.concat(frames, ignore_index=True)
out = ROOT / "results" / "analysis" / run
out.mkdir(parents=True, exist_ok=True)
df.to_parquet(out / "frame.parquet")

tab = condition_table(df, by=("method", "alpha", "K"))
tab.to_csv(out / "table.csv", index=False)
summary = {"run": run, "n_rows": int(len(df)), "concepts": sorted(df.cid.unique().tolist()), "budgets": {}}
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
for b in [0.5, 1.0, 1.5]:
    sel = budget_select(tab, b, by=("method", "K"))
    sel = sel.sort_values("dC", ascending=False)
    sel.to_csv(out / f"budget_{b}.csv", index=False)
    best_per_method = sel.sort_values("dC", ascending=False).groupby("method").head(1)
    summary["budgets"][str(b)] = best_per_method[["method", "K", "alpha", "dC", "dNLL", "dRel", "protJS", "dce", "int_tgt_gain",
                                                    "int_p_rel_change", "int_new_active_per_pos", "int_ms_per_pos"]].to_dict("records")
    print(f"\n=== budget dNLL <= {b}: best (K, alpha) per method ===")
    print(best_per_method[["method", "K", "alpha", "dC", "dNLL", "dRel", "protJS", "dce", "int_tgt_gain", "int_p_rel_change",
                           "int_new_active_per_pos", "int_ms_per_pos"]].round(3).to_string(index=False))

# per-concept view at the primary budget
sel = budget_select(tab, B, by=("method", "K"))
best = sel.sort_values("dC", ascending=False).groupby("method").head(1)
pc = []
for _, r in best.iterrows():
    sub = df[(df.method == r["method"]) & (df.K == r["K"]) & (np.isclose(df.alpha, r["alpha"]))]
    g = sub.groupby("cid")[["dC", "dNLL"]].mean()
    for cid, v in g.iterrows():
        pc.append(dict(method=r["method"], cid=cid, dC=v.dC, dNLL=v.dNLL))
pcd = pd.DataFrame(pc).pivot_table(index="cid", columns="method", values="dC")
print(f"\n=== per-concept dC at budget {B} ===")
print(pcd.round(3).to_string())
summary["per_concept_dC_at_budget"] = pcd.round(4).to_dict()

# internal (encoder self-consistency) vs behavior: rank correlation across all (method, K, alpha) conditions
from scipy.stats import spearmanr

cc = tab[tab.method.isin(["dec", "enc", "pinv", "ridge", "dec_proj", "pinv_fs", "dec_proj_fs", "opt"])]
rho_tg = spearmanr(cc["int_tgt_gain"], cc["dC"]).correlation
rho_p = spearmanr(cc["int_p_rel_change"], cc["dNLL"]).correlation
rho_new = spearmanr(cc["int_new_active_per_pos"], cc["dNLL"]).correlation
summary["internal_vs_behavior_spearman"] = dict(tgt_gain_vs_dC=rho_tg, p_rel_change_vs_dNLL=rho_p, new_active_vs_dNLL=rho_new)
print("\ninternal vs behavior (Spearman over SAE conditions):", summary["internal_vs_behavior_spearman"])
save_json(summary, out / "summary.json")
