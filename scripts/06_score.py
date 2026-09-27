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
def priority(r):
    """Scoring order after the decision arms: controls / reference, norm-matched points, other frozen points, rest."""
    if not nli_arms:
        return 0
    c = r["cond"]
    m, a = c["method"], float(c["alpha_mult"])
    fa = float(proto["alpha_by_method"].get(m, -1))
    if m in proto.get("controls", []) + proto.get("reference_methods", []):
        return 0
    if abs(a - float(proto["alpha_by_method"][proto["primary_corrected"]])) < 1e-9 and m in proto["h1_comparators"]:
        return 1
    if abs(a - fa) < 1e-9:
        return 2
    return 3


nonli = {k: v for k, v in judges.items() if k != "nli"} if nli_arms else judges
for level in sorted({priority(r) for _, rows in files for r in rows}):
    for path, rows in files:
        todo = [r for r in rows if priority(r) == level]
        s = score_rows(todo, rows[0]["prompts"], nonli, path.with_suffix(".scores.jsonl"), use_nli=nli)
        print(f"priority {level}:", path.name, len(todo), "rows,", len(s), "scored in file", flush=True)
print("ALL_DONE", flush=True)
