"""Figures for the report (static PNG, light theme; palette = validated reference slots 1-3 + neutral gray).

usage: python scripts/13_figures.py <test_run> [<topk_run>]
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import n3  # noqa: F401,E402
from n3.analysis import condition_table  # noqa: E402
from n3.data import ROOT, load_json  # noqa: E402

SURF, INK, INK2, GRID, GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9a9993"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # blue, orange, aqua (validated all-pairs for 3 series)
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
proto = load_json(ROOT / "configs" / "protocol.json")
prim = proto["primary_corrected"]
LABEL = {"dec": "decoder (dec)", "diffmean": "DiffMean", prim: f"corrected ({prim})"}
COLOR = {"dec": S1, "diffmean": S2, prim: S3}


def pareto(run, setting, fname, title):
    df = pd.read_parquet(ROOT / "results" / "analysis" / run / "frame.parquet")
    df = df[df.setting == setting]
    tab = condition_table(df)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m in ["dec", "diffmean", prim]:
        t = tab[tab.cfg == proto["cfg_by_method"].get(m, "")].sort_values("alpha")
        if len(t) < 2:
            continue
        xs = np.r_[0, t.dNLL.values]
        ys = np.r_[0, t.dC.values]
        ax.plot(xs, ys, color=COLOR[m], lw=2, marker="o", ms=5, label=LABEL[m], zorder=3)
        ax.annotate(LABEL[m], (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, 0), color=INK2, fontsize=9, va="center")
    for m, cfg in proto["cfg_by_method"].items():
        if m in ("dec", "diffmean", prim, "prompt"):
            continue
        t = tab[(tab.cfg == cfg) & np.isclose(tab.alpha, proto["alpha_by_method"][m])]
        if len(t):
            ax.scatter(t.dNLL, t.dC, s=36, color=GRAY, zorder=2, edgecolor=SURF, linewidth=1.5)
            ax.annotate(m, (float(t.dNLL.iloc[0]), float(t.dC.iloc[0])), textcoords="offset points", xytext=(5, -9), color=INK2, fontsize=8)
    ax.axvline(proto["B_NLL"], color=INK2, lw=1, ls="--")
    ax.text(proto["B_NLL"], ax.get_ylim()[1] * 0.97, " fluency budget", color=INK2, fontsize=8, va="top")
    ax.set_xlabel("fluency cost  ΔNLL (nats/token, Qwen2.5-0.5B judge)")
    ax.set_ylabel("concept lift  ΔC (independent classifier)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=160)
    plt.close(fig)


def internal_vs_behavior(run, setting, fname):
    df = pd.read_parquet(ROOT / "results" / "analysis" / run / "frame.parquet")
    df = df[df.setting == setting]
    rows = []
    for m in ["dec", "enc", "pinv", "ridge", "dec_proj", "pinv_fs", "dec_proj_fs", "opt", "diffmean", "random"]:
        cfg = proto["cfg_by_method"].get(m)
        sub = df[(df.cfg == cfg) & np.isclose(df.alpha, proto["alpha_by_method"][m])]
        if len(sub):
            g = sub.groupby("cid")[["int_tgt_gain", "int_new_active_per_pos", "dC", "dNLL"]].mean().mean()
            rows.append(dict(method=m, alpha=proto["alpha_by_method"][m], **g.to_dict()))
    t = pd.DataFrame(rows).set_index("method")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    order = t.sort_values("dC").index
    for ax, col, lab in zip(axes, ["int_tgt_gain", "int_new_active_per_pos", "dC"],
                            ["encoder: target activation gain", "encoder: newly active features / position", "behavior: concept lift ΔC"]):
        colr = [S3 if m == prim else (S2 if col == "dC" else S1) for m in order]
        ax.barh(range(len(order)), t.loc[order, col], color=colr, height=0.6)
        ax.set_title(lab, loc="left", fontsize=10)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"{m} (α={t.loc[m, 'alpha']:.2f})" for m in order])
        ax.grid(axis="y", visible=False)
    fig.suptitle("Tier 0 (same-SAE readback) vs Tier 2 (independent judges), each method at its dev-frozen strength", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=160)
    plt.close(fig)


def realizability_costs(fname):
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    labels, data = [], []
    for setting, short in [("gpt2_relu_jb_L6", "GPT-2 ReLU"), ("gpt2_topk_oai_L6", "GPT-2 TopK+LN"), ("gemma3_270m_jumprelu_L12", "Gemma-3 JumpReLU")]:
        p = ROOT / "results" / "realizability" / f"{setting}.jsonl"
        if not p.exists():
            continue
        d = pd.DataFrame([json.loads(l) for l in open(p)])
        d = d[d.K == proto["K_by_method"][prim]]
        for col, nm in [("c_free", "free"), ("c_lin", "lin (P=)"), ("c_qp", "QP (P=, P0)"), ("gn_c", "GN (P=)"), ("c_dec", "decoder")]:
            if col in d:
                v = (d[col] / d["resid_norm"]).replace([np.inf, -np.inf], np.nan).dropna()
                if len(v):
                    labels.append(f"{short}\n{nm}")
                    data.append(v.values)
    ax.boxplot(data, vert=True, showfliers=False, medianprops=dict(color=S2, lw=2), boxprops=dict(color=INK2),
               whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("edit norm / ‖h‖ to reach the target")
    ax.set_yscale("log")
    ax.set_title("Minimum edit norm to realize a typical target change (neutral positions)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=160)
    plt.close(fig)


def h2_scatter(setting, fname):
    p = ROOT / "results" / "analysis" / f"h2_table_{setting}.csv"
    if not p.exists():
        return
    t = pd.read_csv(p, index_col=0)
    d = proto["h2"]["primary_diagnostic"]
    y = f"dC_{proto['h2']['outcome_method']}"
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for fam, col in [("topic", S1), ("emotion", S2), ("entity", S3)]:
        s = t[t.index.str.startswith(fam)]
        ax.scatter(s[d], s[y], s=40, color=col, label=fam, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.set_xlabel(f"{d} (realizability diagnostic, neutral positions)")
    ax.set_ylabel(f"held-out ΔC of {proto['h2']['outcome_method']}")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("H2: does the diagnostic predict behavioral steerability?", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    test_run = sys.argv[1]
    pareto(test_run, "gpt2_relu_jb_L6", "pareto_relu_heldout.png", "Held-out concepts × prompts — GPT-2 L6, ReLU SAE (primary)")
    internal_vs_behavior(test_run, "gpt2_relu_jb_L6", "internal_vs_behavior_relu.png")
    realizability_costs("realizability_costs.png")
    h2_scatter("gpt2_relu_jb_L6", "h2_relu.png")
    if len(sys.argv) > 2:
        pareto(sys.argv[2], "gpt2_topk_oai_L6", "pareto_topk_heldout.png", "Held-out — GPT-2 L6, TopK+LN SAE (frozen transfer)")
    print("figures in", FIG)
