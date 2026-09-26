"""Launch a frozen held-out grid from configs/protocol.json (two parallel parts over concepts).

usage: python scripts/12_run_test.py <grid_name> <run_name>
"""
import json
import os
import subprocess
import sys

import n3  # noqa: F401
from n3.concepts import load_splits
from n3.data import ROOT, load_json

grid_name, run = sys.argv[1], sys.argv[2]
proto = load_json(ROOT / "configs" / "protocol.json")
g = proto["test_grids"][grid_name]
explicit = list(g["explicit"])
for m in g.get("pareto_methods", []):
    base = next(e for e in g["explicit"] if e["method"] == m)
    for a in g["pareto_alphas"]:
        explicit.append({**base, "alpha_mult": a})
concepts = g.get("concepts") or load_splits()["test_concepts"]
parts = [concepts[0::2], concepts[1::2]]
procs = []
for i, cs in enumerate(parts):
    grid = dict(methods=[], alphas=[], Ks=[], explicit=explicit, concepts=cs, part="ab"[i])
    if g.get("downstream"):
        grid["downstream"] = g["downstream"]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    log = open(ROOT / "results" / f"logs_{run}_{'ab'[i]}.txt", "w")
    procs.append(subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts" / "04_sweep.py"), run, g["setting"], g["split"], json.dumps(grid)],
                                  stdout=log, stderr=subprocess.STDOUT, env=env))
print("launched", run, [p.pid for p in procs], "explicit conditions per concept:", len(explicit))
for p in procs:
    p.wait()
print("finished", [p.returncode for p in procs])
