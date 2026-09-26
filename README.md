# Which Sparse Autoencoder Feature Edits Are Realizable? Geometry, Interference, and Constrained Steering (N3)

**Core question.** Can a desired change in target SAE features **T** be realized while the
non-target (protected) features **P** are preserved? When that realizability is diagnosed and
corrected, does it help **independent downstream behavior**, beyond the SAE encoder's own scores and
beyond generic (SAE-free) steering?

Status and verdict: see [`docs/04_report.md`](docs/04_report.md) (written after the held-out run).
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
python scripts/05_realizability.py gpt2_relu_jb_L6                 # steps 4-5
pytest -q tests/
```

Hardware used: 4 CPU cores, 15 GB RAM, no GPU (all numbers reproducible on CPU).
