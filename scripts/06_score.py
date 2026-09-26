"""Score all generations of a run with the independent judges.

usage: python scripts/06_score.py <run_name> [--nli | --nli-decision-arms] [--only=<substring of file name>]
--nli-decision-arms: NLI judge only for the rows the pre-registered H1 decision needs (primary + comparators at
their frozen (config, alpha)); all other judges for every row.
"""
import sys
from pathlib import Path

import torch

import n3  # noqa: F401
from n3.data import ROOT
from n3.evaluate import load_judges, load_rows, score_rows

torch.set_grad_enabled(False)
run = sys.argv[1]
nli = "--nli" in sys.argv
nli_arms = "--nli-decision-arms" in sys.argv
only = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")), None)
judges = load_judges(nli=nli or nli_arms)
if nli_arms:
    from n3.data import load_json
    proto = load_json(ROOT / "configs" / "protocol.json")
    arms = {(proto["cfg_by_method"][m], float(proto["alpha_by_method"][m])) for m in [proto["primary_corrected"]] + proto["h1_comparators"]}


def needs_nli(r):
    c = r["cond"]
    cfg = (f'{c["method"]}|K{c["K"]}|P{c["protect_topm"]}|r{c["ridge_rel"]}|fs{c["fs_rounds"]}|'
           f'mp{c["opt_mu_p"]}|mn{c["opt_mu_new"]}|d{c["max_density"]}')
    return (cfg, float(c["alpha_mult"])) in arms
files = []
for path in sorted((ROOT / "results" / "runs" / run).glob("*.jsonl")):
    if path.name.endswith(".scores.jsonl") or (only and only not in path.name):
        continue
    rows, seen = [], set()
    for r in load_rows(path):
        if r["key"] not in seen:
            seen.add(r["key"])
            rows.append(r)
    files.append((path, rows))

if nli_arms:
    # pass A: rows the pre-registered H1 decision needs (+ each file's unsteered baseline), all judges incl. NLI
    for path, rows in files:
        need = [r for r in rows if needs_nli(r) or r["cond"]["method"] == "none"]
        score_rows(need, rows[0]["prompts"], {k: v for k, v in judges.items() if k != "nli"}, path.with_suffix(".scores.jsonl"))
        score_rows([r for r in need if r["cond"]["method"] != "none"], rows[0]["prompts"], judges, path.with_suffix(".scores.jsonl"), use_nli=True)
        print("decision arms scored:", path.name, len(need), flush=True)
    print("DECISION_ARMS_DONE", flush=True)
for path, rows in files:
    s = score_rows(rows, rows[0]["prompts"], {k: v for k, v in judges.items() if k != "nli"} if nli_arms else judges,
                   path.with_suffix(".scores.jsonl"), use_nli=nli)
    print(path.name, len(rows), "rows,", len(s), "scored", flush=True)
