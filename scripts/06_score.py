"""Score all generations of a run with the independent judges.

usage: python scripts/06_score.py <run_name> [--nli]
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
judges = load_judges(nli=nli)
for path in sorted((ROOT / "results" / "runs" / run).glob("*.jsonl")):
    if path.name.endswith(".scores.jsonl"):
        continue
    rows = load_rows(path)
    prompts = rows[0]["prompts"]
    scored = path.with_suffix(".scores.jsonl")
    s = score_rows(rows, prompts, judges, scored, use_nli=nli)
    print(path.name, len(rows), "rows,", len(s), "scored", flush=True)
