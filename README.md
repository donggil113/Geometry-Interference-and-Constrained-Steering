# Which Sparse Autoencoder Feature Edits Are Realizable? Geometry, Interference, and Constrained Steering (N3)

**Core question.** Can a desired change in target SAE features **T** be realized while the
non-target (protected) features **P** are preserved? When that realizability is diagnosed and
corrected, does it help **independent downstream behavior**, beyond the SAE encoder's own scores and
beyond generic (SAE-free) steering?

## Verdict: **NO_GO**

The result is pre-registered (freeze commit `4282fb5`) and was independently re-verified. Full report:
[`docs/04_report.md`](docs/04_report.md).

- **Inside the SAE, (T, P) edits are realizable and cheap** (`docs/03_realizability.md`).
  - For ReLU and JumpReLU, the exact finite-step QP raises the targets while keeping *every* active
    feature fixed and *every* inactive feature off, at 4–15% of ‖h‖. It is feasible in 100% of cases.
  - Local Jacobian solutions flip 9–74 gates and break 3–11% of the way along the step. The exact
    correction adds only 1–8% norm.
  - Under TopK+LayerNorm, full preservation is structurally impossible (slot eviction) in >99% of
    requests.
- **The correction does not help behavior** (GPT-2 small L6, jb ReLU SAE, 16 held-out concepts × 48
  held-out prompts; independent judges).
  - The corrected edit `dec_proj` removes all protected-feature change inside the SAE. It is **not shown
    better than plain decoder steering**: ΔC +0.013, p = 0.37. The same holds against DiffMean (p = 0.32).
  - At matched norm and at matched fluency, DiffMean is ahead (secondary / post hoc).
  - A random protected set behaves identically.
  - Encoder-score-maximizing edits steer *worse*.
  - The null replicates on a held-out TopK SAE: −0.010, p = 0.34.
- **The realizability diagnostic does not predict steerability.** ρ = +0.05 over 24 concepts. Generic
  DiffMean steerability predicts it better (ρ = +0.51).
- **No SAE retraining is warranted.**
Pre-registered protocol: [`docs/02_protocol.md`](docs/02_protocol.md) and `configs/protocol.json`.
Literature audit (step 1): [`docs/01_literature_audit.md`](docs/01_literature_audit.md).

## Layout

| path | content |
|---|---|
| `configs/pinned.json` | model / SAE / dataset checkpoints (HF commit shas), hooks, normalization |
| `configs/splits.json` | frozen dev / held-out concepts and prompts (seed 20260926) |
| `configs/protocol.json` | hyper-parameters frozen on dev before any held-out run |
| `src/n3/saes.py` | exact re-implementation of ReLU / JumpReLU / TopK(+LayerNorm) encoders, pre-activations, Jacobians |
| `src/n3/models.py` | residual read/edit hooks (HF GPT-2 == TL `blocks.6.hook_resid_pre`, Gemma-3-270m layer 12) |
| `src/n3/methods.py` | norm-matched edits: dec, enc, pinv, ridge, dec_proj, *_fs (finite-step), opt, diffmean, random |
| `src/n3/realizability.py` | exact finite-step QP (ReLU/JumpReLU), first-order + true-forward analysis (TopK+LN) |
| `src/n3/judges.py` | independent behavioral judges (topic / emotion classifiers, NLI, Qwen fluency, MiniLM relevance) |
| `src/n3/evaluate.py`, `src/n3/analysis.py` | generation / utility / judging harness; aggregation and cluster bootstrap |
| `scripts/00_*` … `scripts/0x_*` | pipeline steps (see below) |
| `results/validation/` | checkpoint, backend, judge validation |
| `results/runs/`, `results/analysis/`, `results/realizability/` | raw generations, scores, aggregates |

## Pipeline

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sae-lens transformer_lens sentence-transformers cvxpy scipy pandas pyarrow pytest && pip install -e .
python scripts/00_pin_and_validate.py            # step 2: pins + FVU / L0 / CE-recovered
python scripts/00b_validate_hf_gpt2_backend.py   # fast HF backend == TransformerLens hook
python scripts/01_splits_and_judges.py           # frozen splits; judge validation
python scripts/02_feature_stats.py gpt2_relu_jb_L6 gpt2_topk_oai_L6
python scripts/04_sweep.py dev1 gpt2_relu_jb_L6 dev '<grid json>'   # dev grid
python scripts/06_score.py dev1 && python scripts/07_analyze.py dev1
python scripts/08_freeze_protocol.py dev2                           # freeze -> configs/protocol.json
python scripts/12_run_test.py primary test1 && python scripts/06_score.py test1 --nli-decision-arms
python scripts/07_analyze.py test1 && python scripts/11_decide.py test1 gpt2_relu_jb_L6   # pre-registered H1
python scripts/05_realizability.py gpt2_relu_jb_L6 48 && python scripts/09_realizability_summary.py gpt2_relu_jb_L6  # steps 4-5
python scripts/10_h2_diagnosis.py gpt2_relu_jb_L6 test1 h2test      # H2
python scripts/14_runtime.py && python scripts/15_report_numbers.py && python scripts/13_figures.py test1 topk1
pytest -q tests/
```

Hardware used: 4 CPU cores, 15 GB RAM, no GPU (all numbers reproducible on CPU).
