"""Aggregation of judged generations into per-condition metrics, budgeted endpoints and bootstrap CIs.

Behavioral metrics (independent judges, relative to the unsteered continuation of the *same* prompt with
the *same* sampling noise):
  dC      : judge P(target concept) - baseline P(target concept)
  dNLL    : Qwen2.5-0.5B conditional NLL(continuation | prompt) - baseline      (fluency cost, nats/token)
  dRel    : MiniLM cos(prompt, continuation) - baseline                         (prompt relevance)
  protJS  : Jensen-Shannon divergence of a *protected* family's judge distribution vs baseline
            (topic / entity steering -> emotion distribution should stay; emotion steering -> topic should stay)
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
        elif fam == "entity":
            xp, bx = np.array(s["entity_probs"]), np.array(b["entity_probs"])
            C, C0 = xp[:, lab], bx[:, lab]
            prot = _js(ep, be)  # entity steering: the emotion distribution should stay
        else:
            C, C0 = ep[:, lab], be[:, lab]
            prot = _js(tp, bt)
        n = len(C)
        cfg = (f'{c["method"]}|K{c["K"]}|P{c["protect_topm"]}|r{c["ridge_rel"]}|fs{c["fs_rounds"]}|'
               f'mp{c["opt_mu_p"]}|mn{c["opt_mu_new"]}|d{c["max_density"]}')
        base = dict(key=r["key"], cfg=cfg, split=r.get("split"), setting=c["setting"], cid=c["cid"], family=fam, method=c["method"], alpha=c["alpha_mult"],
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


def condition_table(df: pd.DataFrame, by=("cfg", "method", "K", "alpha")) -> pd.DataFrame:
    """Concept-balanced means: average over prompts within concept, then over concepts.
    Always keyed by the full config id ('cfg') so different hyper-parameter variants are never pooled."""
    by = list(by)
    if "cfg" not in by:
        by = ["cfg"] + by
    cols = [m for m in METRICS if m in df]
    g = df.groupby(by + ["cid"], dropna=False)[cols].mean().reset_index()
    out = g.groupby(by, dropna=False)[cols].mean().reset_index()
    out["n_concepts"] = g.groupby(by, dropna=False).size().values
    return out


def budget_select(tab: pd.DataFrame, budget: float, cost="dNLL", gain="dC", by=("cfg", "method", "K"),
                  degeneration_floor: float | None = None) -> pd.DataFrame:
    """For each method config pick the alpha with max `gain` subject to `cost` <= budget (and, if given,
    dDist2 >= degeneration_floor).  alpha = 0 (gain 0, cost 0) is always an admissible candidate."""
    out = []
    for key, sub in tab.groupby(list(by), dropna=False):
        ok = sub[sub[cost] <= budget]
        if degeneration_floor is not None and "dDist2" in ok:
            ok = ok[ok["dDist2"] >= degeneration_floor]
        keyd = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        if len(ok) == 0 or ok[gain].max() <= 0:
            best = dict(keyd, alpha=0.0, **{gain: 0.0, cost: 0.0}, selected_zero=True)
        else:
            best = dict(ok.loc[ok[gain].idxmax()].to_dict(), selected_zero=False)
        out.append(best)
    return pd.DataFrame(out)


def _paired_matrix(df, sel, metric):
    m = np.ones(len(df), bool)
    for k, v in sel.items():
        m &= np.isclose(df[k].values, v) if isinstance(v, float) else (df[k] == v).values
    sub = df[m]
    if len(sub) == 0:
        raise ValueError(f"empty arm {sel}")
    dup = sub.duplicated(subset=["cid", "prompt"]).any()
    if dup:
        raise ValueError(f"arm {sel} has duplicate (cid, prompt) cells -- configs would be pooled")
    return sub.pivot(index="cid", columns="prompt", values=metric)


def signflip_test(per_concept_diffs: np.ndarray) -> float:
    """Exact two-sided sign-flip randomization test on concept-level mean differences (2^n flips)."""
    d = np.asarray(per_concept_diffs, float)
    n = len(d)
    obs = abs(d.mean())
    signs = np.array(np.meshgrid(*[[-1, 1]] * n)).reshape(n, -1).T  # (2^n, n)
    null = np.abs((signs * d).mean(1))
    return float((null >= obs - 1e-12).mean())


def cluster_bootstrap_diff(df: pd.DataFrame, sel_a: dict, sel_b: dict, metric="dC", n_boot=5000, seed=0):
    """Paired difference mean(metric | a) - mean(metric | b) over (cid, prompt).
    Primary inference: exact concept-level sign-flip test (p_signflip).  Sensitivity: two-way cluster
    bootstrap over concepts and prompts (percentile CI)."""
    rng = np.random.RandomState(seed)
    A, B = _paired_matrix(df, sel_a, metric), _paired_matrix(df, sel_b, metric)
    cids = sorted(set(A.index) & set(B.index))
    if len(cids) < len(set(A.index) | set(B.index)):
        raise ValueError("arms cover different concepts")
    A, B = A.loc[cids], B.loc[cids]
    D = (A - B).values  # (concepts, prompts)
    if np.isnan(D).all():
        raise ValueError("all-NaN differences")
    est = float(np.nanmean(D))
    per_c = np.nanmean(D, 1)
    nc, npm = D.shape
    boots = np.empty(n_boot)
    for b in range(n_boot):
        ci = rng.randint(0, nc, nc)
        pi = rng.randint(0, npm, npm)
        boots[b] = np.nanmean(D[np.ix_(ci, pi)])
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_boot = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))
    return dict(est=est, lo=float(lo), hi=float(hi), p_signflip=signflip_test(per_c), p_boot_two_sided=p_boot,
                n_concepts=nc, n_prompts=npm, n_concepts_positive=int((per_c > 0).sum()),
                per_concept=dict(zip(cids, per_c.tolist())))


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
