"""Compute feature statistics (concept contrasts, OWT density, activation scale, DiffMean) per setting."""
import sys
import time

import torch

import n3  # noqa: F401
from n3.features import compute_setting_stats
from n3.settings import load_setting

torch.set_grad_enabled(False)
for name in sys.argv[1:]:
    t = time.time()
    model, sae = load_setting(name)
    st = compute_setting_stats(model, sae, name)
    print(name, "done", f"{time.time()-t:.0f}s", "median norm", st["resid_norm_median"], flush=True)
