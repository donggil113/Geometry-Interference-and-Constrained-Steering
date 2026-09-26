"""Summarize finite-step realizability results per setting and per concept (steps 4-5).

usage: python scripts/09_realizability_summary.py <setting> [<setting> ...]
Writes results/realizability/summary_<setting>.json and per-concept CSVs.
"""
import json
import sys

import numpy as np
import pandas as pd

import n3  # noqa: F401
from n3.data import ROOT, save_json

for setting in sys.argv[1:]:
    rows = [json.loads(l) for l in open(ROOT / "results" / "realizability" / f"{setting}.jsonl")]
    df = pd.DataFrame(rows)
    df["c_lin_rel"] = df["c_lin"] / df["resid_norm"]
    df["c_free_rel"] = df["c_free"] / df["resid_norm"]
    df["c_dec_rel"] = df["c_dec"] / df["resid_norm"]
    if "c_qp" in df:
        df["c_qp_rel"] = df["c_qp"] / df["resid_norm"]
    out = {"setting": setting, "n_rows": len(df), "by_K": {}}
    num = [c for c in ["kappa", "c_free_rel", "c_lin_rel", "c_qp_rel", "qp_over_lin", "c_dec_rel", "lin_new_active",
                       "lin_crossings", "lin_first_crossing_t", "dec_P_rel_change", "dec_new_active", "dec_P_lost",
                       "enc_dec_cos", "n_active", "near_kink_protected", "qp_n_binding", "qp_rounds",
                       "lin_T_err_rel", "lin_P_rel_change", "lin_P_lost", "gn_c", "gn_pre_residual_rel", "gn_P_lost", "gn_new_active"] if c in df]
    for K, sub in df.groupby("K"):
        s = {}
        for c in num:
            v = sub[c].replace([np.inf, -np.inf], np.nan)
            s[c] = dict(median=float(v.median()), p10=float(v.quantile(0.1)), p90=float(v.quantile(0.9)),
                        frac_inf=float(np.isinf(sub[c]).mean()))
        for flag in ["eq_rank_full", "qp_feasible", "qp_converged", "qp_within_budget", "lin_within_budget", "dec_within_budget",
                     "topk_structural_conflict", "gn_T_reached"]:
            if flag in sub:
                s["frac_" + flag] = float(sub[flag].astype(float).mean())
        if "qp_check_P_err" in sub:
            s["max_qp_check_T_err"] = float(sub["qp_check_T_err"].max())
            s["max_qp_check_P_err"] = float(sub["qp_check_P_err"].max())
            s["frac_qp_check_new_active_gt0"] = float((sub["qp_check_new_active"] > 0).mean())
        s["frac_T_active_at_h"] = float((sub["T_active"] > 0).mean())
        out["by_K"][str(K)] = s
        # per-concept medians (diagnostics used for H2)
        pc = sub.replace([np.inf, -np.inf], np.nan).groupby("cid")[[c for c in num if c in sub]].median()
        pc.to_csv(ROOT / "results" / "realizability" / f"per_concept_{setting}_K{K}.csv")
        # degeneracy check: spread across concepts of the key diagnostics
        s["between_concept_cv"] = {c: float(pc[c].std() / abs(pc[c].mean())) for c in ["kappa", "c_lin_rel", "c_qp_rel", "c_dec_rel"] if c in pc and pc[c].mean() != 0}
    save_json(out, ROOT / "results" / "realizability" / f"summary_{setting}.json")
    print(json.dumps(out, indent=1)[:4000])
