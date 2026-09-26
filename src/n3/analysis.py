"""Aggregation of judged generations into per-condition metrics, budgeted endpoints and bootstrap CIs.

Behavioral metrics (independent judges, relative to the unsteered continuation of the *same* prompt with
the *same* sampling noise):
  dC      : judge P(target concept) - baseline P(target concept)
  dNLL    : Qwen2.5-0.5B conditional NLL(continuation | prompt) - baseline      (fluency cost, nats/token)
  dRel    : MiniLM cos(prompt, continuation) - baseline                         (prompt relevance)
  protJS  : Jensen-Shannon divergence of the *other* family's judge distribution vs baseline
            (topic steering -> emotion distribution should stay; emotion steering -> topic should stay)
  dDist2  : distinct-2 - baseline                                               (degeneration)
Utility: dCE / KL on held-out OWT text.  Internal (encoder self-consistency): tgt_gain, p_rel_change,
new_active_per_pos -- reported separately, never mixed into behavioral endpoints.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _js(p, q, eps=1e-9):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    m = 0.5 * (p + q)
    return 0.5 * (p * np.log(p / m)).sum(-1) + 0.5 * (q * np.log(q / m)).sum(-1)


def per_prompt_frame(rows, scores, baseline_key) -> pd.DataFrame:
    """Long frame: one line per (condition, prompt) with behavioral deltas vs baseline."""
    b = scores[baseline_key]
    bt, be = np.array(b["topic_probs"]), np.array(b["emotion_probs"])
    recs = []
    for r in rows:
        if r["key"] not in scores:
            continue
        s = scores[r["key"]]
        c = r["cond"]
        fam, lab = c["cid"].split(":")
        lab = int(lab)
        tp, ep = np.array(s["topic_probs"]), np.array(s["emotion_probs"])
        if fam == "topic":
            C, C0 = tp[:, lab], bt[:, lab]
            prot = _js(ep, be)
        else:
            C, C0 = ep[:, lab], be[:, lab]
            prot = _js(tp, bt)
        n = len(C)
        base = dict(key=r["key"], setting=c["setting"], cid=c["cid"], family=fam, method=c["method"], alpha=c["alpha_mult"],
                    K=c["K"], protect_topm=c["protect_topm"], ridge_rel=c["ridge_rel"], fs_rounds=c["fs_rounds"],
                    opt_mu_p=c["opt_mu_p"], opt_mu_new=c["opt_mu_new"],
                    **{k: v for k, v in r["utility"].items() if k != "ce_clean"}, **{f"int_{k}": v for k, v in r["internal"].items()},
                    gen_secs=r["gen_secs"])
        for i in range(n):
            rec = dict(base, prompt=i, C=C[i], C0=C0[i], dC=C[i] - C0[i], dNLL=s["nll"][i] - b["nll"][i],
                       NLL=s["nll"][i], dRel=s["rel"][i] - b["rel"][i], protJS=prot[i],
                       dDist2=s["distinct2"][i] - b["distinct2"][i])
            if "nli" in s:
                rec["nli"] = s["nli"][i]
            recs.append(rec)
    return pd.DataFrame(recs)


METRICS = ["C", "dC", "dNLL", "dRel", "protJS", "dDist2", "dce", "kl", "down_p_rel_drift", "down_new_active_per_pos", "nli",
           "int_tgt_gain", "int_p_rel_change", "int_new_active_per_pos", "int_err_delta_frac", "int_err_norm_ratio",
           "int_n_constraints", "int_ms_per_pos"]


def condition_table(df: pd.DataFrame, by=("method", "alpha", "K")) -> pd.DataFrame:
    by = list(by)
    g = df.groupby(by + ["cid"])[[m for m in METRICS if m in df]].mean().reset_index()
    return g.groupby(by)[[m for m in METRICS if m in g]].mean().reset_index()


def budget_select(tab: pd.DataFrame, budget: float, cost="dNLL", gain="dC", by=("method", "K")) -> pd.DataFrame:
    """For each method config, pick the alpha with max `gain` subject to `cost` <= budget (alpha=0 allowed)."""
    out = []
    for key, sub in tab.groupby(list(by)):
        ok = sub[sub[cost] <= budget]
        if len(ok) == 0:
            best = dict(zip(by, key if isinstance(key, tuple) else (key,)), alpha=0.0, **{gain: 0.0, cost: 0.0})
        else:
            best = ok.loc[ok[gain].idxmax()].to_dict()
        out.append(best)
    return pd.DataFrame(out)


def cluster_bootstrap_diff(df: pd.DataFrame, sel_a: dict, sel_b: dict, metric="dC", n_boot=5000, seed=0):
    """Paired difference mean(metric | a) - mean(metric | b) over (cid, prompt), bootstrap resampling
    concepts and prompts (two-way cluster bootstrap)."""
    rng = np.random.RandomState(seed)

    def pick(sel):
        m = np.ones(len(df), bool)
        for k, v in sel.items():
            m &= (df[k] == v).values
        return df[m].pivot_table(index="cid", columns="prompt", values=metric)

    A, B = pick(sel_a), pick(sel_b)
    cids = sorted(set(A.index) & set(B.index))
    A, B = A.loc[cids], B.loc[cids]
    D = (A - B).values  # (concepts, prompts)
    est = float(np.nanmean(D))
    nc, npm = D.shape
    boots = np.empty(n_boot)
    for b in range(n_boot):
        ci = rng.randint(0, nc, nc)
        pi = rng.randint(0, npm, npm)
        boots[b] = np.nanmean(D[np.ix_(ci, pi)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_two = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))
    return dict(est=est, lo=float(lo), hi=float(hi), p_two_sided=p_two, n_concepts=nc, n_prompts=npm,
                per_concept=dict(zip(cids, np.nanmean(D, 1).tolist())))


def holm(pvals: dict, alpha=0.05) -> dict:
    """Holm-Bonferroni step-down; returns {name: reject?}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, still = {}, True
    for i, (k, p) in enumerate(items):
        still = still and (p <= alpha / (m - i))
        out[k] = bool(still)
    return out


def interp_at_budget(tab_m: pd.DataFrame, budget: float, cost="dNLL", gain="dC") -> float:
    """Linear interpolation of gain at cost == budget along a method's alpha curve (alpha=0 -> (0,0)).
    Returns NaN if the curve never reaches the budget (reported, not imputed)."""
    t = tab_m.sort_values("alpha")
    xs = np.concatenate([[0.0], t[cost].values])
    ys = np.concatenate([[0.0], t[gain].values])
    for i in range(1, len(xs)):
        if xs[i] >= budget:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            return float(y0 + (y1 - y0) * (budget - x0) / (x1 - x0)) if x1 > x0 else float(y1)
    return float("nan")
