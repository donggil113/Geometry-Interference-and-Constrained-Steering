"""Freeze the held-out protocol from dev results only -> configs/protocol.json.

usage: python scripts/08_freeze_protocol.py <dev_run>
Selection (dev concepts x dev prompts, GPT-2 / jb ReLU SAE):
  * B_NLL = 1.0 nats/token fluency budget, degeneration floor dDist2 >= -0.10 (fixed a priori).
  * For every method: the (config, K, alpha) with the largest concept-balanced dev dC under the budget.
  * PRIMARY_CORRECTED = argmax dev dC among the realizability-corrected methods.
Nothing here reads held-out data.
"""
import subprocess
import sys

import pandas as pd

import n3  # noqa: F401
from n3.analysis import budget_select, condition_table
from n3.concepts import load_splits
from n3.data import ROOT, save_json
from n3.evaluate import code_fingerprint

dev_run = sys.argv[1]
B, DEG = 1.0, -0.10
df = pd.read_parquet(ROOT / "results" / "analysis" / dev_run / "frame.parquet")
assert set(df.cid.unique()) <= set(load_splits()["dev_concepts"]), "dev frame contains non-dev concepts"
tab = condition_table(df)
sel = budget_select(tab, B, degeneration_floor=DEG)
best = sel.sort_values("dC", ascending=False).groupby("method").head(1).set_index("method")

CORRECTED = ["pinv", "ridge", "dec_proj", "pinv_fs", "dec_proj_fs", "opt"]
prim = best.loc[[m for m in CORRECTED if m in best.index], "dC"].idxmax()
cfg_by, K_by, a_by = {}, {}, {}
for m, r in best.iterrows():
    cfg_by[m], K_by[m], a_by[m] = r["cfg"], int(r["K"]), float(r["alpha"])
# controls / references at the primary's operating point
pk, pa = K_by[prim], a_by[prim]
controls = ["dec_rand_feat"]
cfg_by["dec_rand_feat"], K_by["dec_rand_feat"], a_by["dec_rand_feat"] = cfg_by["random"].replace("random|", "dec_rand_feat|"), K_by["random"], pa
if prim in ("dec_proj", "dec_proj_fs"):
    c = prim + "_randP"
    controls.append(c)
    cfg_by[c], K_by[c], a_by[c] = cfg_by[prim].replace(prim + "|", c + "|"), pk, pa
cfg_by["prompt"], K_by["prompt"], a_by["prompt"] = cfg_by["random"].replace("random|", "prompt|"), 1, 0.0


def explicit(m):
    cfg = cfg_by[m].split("|")
    kw = dict(method=m, alpha_mult=a_by[m], K=K_by[m])
    for tok in cfg[1:]:
        if tok.startswith("P") and tok[1:] != "None":
            kw["protect_topm"] = int(tok[1:])
        elif tok.startswith("r"):
            kw["ridge_rel"] = float(tok[1:])
        elif tok.startswith("fs"):
            kw["fs_rounds"] = int(tok[2:])
        elif tok.startswith("mp"):
            kw["opt_mu_p"] = float(tok[2:])
        elif tok.startswith("mn"):
            kw["opt_mu_new"] = float(tok[2:])
        elif tok.startswith("d"):
            kw["max_density"] = float(tok[1:])
    return kw


all_methods = ["dec", "enc", "pinv", "ridge", "dec_proj", "pinv_fs", "dec_proj_fs", "opt", "diffmean", "random"] + controls + ["prompt"]
alphas = sorted(df.alpha.unique().tolist())
git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
proto = dict(
    frozen_from_dev_run=dev_run, frozen_at_commit=git, code_fingerprint=code_fingerprint(),
    B_NLL=B, degeneration_floor=DEG, primary_setting="gpt2_relu_jb_L6",
    primary_corrected=prim, corrected_methods=CORRECTED,
    cfg_by_method=cfg_by, K_by_method=K_by, alpha_by_method=a_by,
    dev_selected=best[["cfg", "K", "alpha", "dC", "dNLL", "dDist2", "dRel", "protJS", "dce"]].reset_index().to_dict("records"),
    h1_comparators=["dec", "enc", "diffmean", "random"], controls=controls, reference_methods=["prompt"],
    h1_rule=dict(test="exact concept-level sign-flip on per-concept mean dC differences", alpha=0.05, multiplicity="holm",
                 require_bootstrap_lo_gt0=True, fluency_guard_max_dNLL_excess=0.10, deg_guard_min_dDist2_diff=-0.05,
                 nli_concordance=True,
                 collateral=dict(vs="dec", protJS_max_excess=0.01, dce_max_excess=0.05, dRel_max_deficit=0.02)),
    h2=dict(outcome_method=prim, primary_diagnostic="c_qp_rel",
            diagnostics=["c_qp_rel", "kappa", "c_lin_rel", "c_dec_rel", "dec_P_rel_change", "dec_new_active", "lin_new_active"],
            rule="supported iff |rho| >= 0.5, bootstrap CI excludes 0, and |rho| > max |rho| of baselines "
                 "(max_dec_cos, neighbor_density_03, log_density, enc_dec_cos, dC_diffmean); all 24 concepts on held-out prompts"),
    test_grids=dict(
        primary=dict(setting="gpt2_relu_jb_L6", split="test", explicit=[explicit(m) for m in all_methods],
                     pareto_methods=["dec", "diffmean", prim], pareto_alphas=alphas,
                     downstream=[7, "gpt2-small-res-jb", "blocks.8.hook_resid_pre"]),
        h2_dev_concepts_on_test_prompts=dict(setting="gpt2_relu_jb_L6", split="test",
                                             concepts=load_splits()["dev_concepts"],
                                             explicit=[explicit(m) for m in [prim, "dec", "diffmean"]]),
        topk_transfer=dict(setting="gpt2_topk_oai_L6", split="test",
                           explicit=[explicit(m) for m in all_methods if m != "prompt"],
                           pareto_methods=["dec", prim], pareto_alphas=alphas),
    ),
    notes=["alpha is a multiple of the setting's median residual norm (GPT-2 L6: 78.9)",
           "every edit is projected onto 1-perp before norm matching (GPT-2 is mean-invariant)",
           "Amendment A1: held-out entity family (8 DBpedia classes) added before any held-out run"],
)
save_json(proto, ROOT / "configs" / "protocol.json")
print(pd.DataFrame(proto["dev_selected"]).round(3).to_string(index=False))
print("PRIMARY_CORRECTED =", prim)
