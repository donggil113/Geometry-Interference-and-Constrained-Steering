# N3 report: Which SAE feature edits are realizable? Geometry, interference, and constrained steering

*Status: <!--VERDICT-->*

## 0. Verdict (pre-registered decision)

<!--VERDICT_BLOCK-->

## 1. What was done (mapping to the brief)

| brief step | where |
|---|---|
| 1. AxBench, 2026 SAE-steering follow-ups, geometric-recovery / causal-validation audits | `docs/01_literature_audit.md`: 79 papers, each verified on a primary page, plus a completeness critic |
| 2. Pin model / SAE hooks, normalization, checkpoints | `configs/pinned.json`, `results/validation/*.json` (§2) |
| 3. Decoder, encoder-gradient, Jacobian pinv/ridge, direct activation optimization; norm-matched | `src/n3/methods.py` (§3) |
| 4. Finite-step realizability of T with P preserved, without claiming rank limits as a theorem | `docs/03_realizability.md` |
| 5. ReLU / JumpReLU / TopK active-set changes and non-differentiability; no global claims from Taylor | `docs/03_realizability.md`, `src/n3/realizability.py` |
| 6. Features and thresholds fixed on dev; evaluated on held-out prompts, concepts and another SAE setting | `docs/02_protocol.md`, `configs/protocol.json` (pre-registration commit `4282fb5`) |
| 7. Encoder self-consistency kept separate from independent behavior; target, collateral, utility and runtime measured | §4–§6 |
| 8. Minimal decisive single-model experiment first | GPT-2 small L6 + jb ReLU SAE (§5) |

## 2. Fixed infrastructure

- **Model.** GPT-2 small (HF `openai-community/gpt2` @ `607a30d`).
  - The edit site is the output of block 5, which equals TransformerLens `blocks.6.hook_resid_pre`.
  - The stream is mean-centred, which equals TL `center_writing_weights`.
  - Checked against TransformerLens:
    - residual rel. diff 1e-6;
    - identical active sets;
    - edited-logit KL 1e-8 for h-independent *and* h-dependent edits;
    - identical greedy generations through the KV cache (`results/validation/hf_vs_tl_gpt2_backend.json`).
- **SAEs (all validated by re-implementing the encoder and comparing with SAELens bit for bit).**

  | setting | SAE | FVU | L0 | CE recovered |
  |---|---|---|---|---|
  | primary | jbloom `gpt2-small-res-jb` L6 (ReLU, 24,576) | 0.126 | 51 | 98.4% |
  | held-out SAE, same site | OpenAI v5 32k TopK (k = 32, LayerNorm input) | 0.133 | 32 | 98.8% |
  | realizability only | Gemma Scope 2 L12 16k JumpReLU on Gemma-3-270m | 0.124 | 67 (published 60) | 99.1% |

  - For Gemma-3-270m, the ungated mirror `unsloth/gemma-3-270m` was used. The SAE reconstruction
    confirms its weights.
- **Mean invariance (found by the adversarial code review).**
  - Every W_enc column of the jb SAE has a large component (median 73% of its norm) along the
    all-ones direction. GPT-2 ignores that direction: every read of the residual goes through a
    LayerNorm.
  - Uncorrected, the encoder-side methods satisfied their constraints "through" that invisible
    direction.
  - All SAEs on GPT-2 therefore read `h − mean(h)`, their Jacobians are centred, and every edit is
    projected onto 1⊥ *before* norm matching.
  - The pre-fix runs are archived as invalid: `results/runs_invalid/`.
- **Judges.** All are independent of GPT-2 and of every SAE (`results/validation/judge_validation.json`).
  - Concept judges: Yahoo-topic BERT (acc 0.73), emotion DistilBERT (0.92), DBpedia BERT (0.995).
  - Zero-shot NLI judge: DeBERTa-v3.
  - Fluency judge: Qwen2.5-0.5B.
  - Relevance judge: MiniLM.

## 3. Methods (all norm-matched per position, ‖δ‖ = α·median‖h‖, BOS never edited)

| family | method | what it does |
|---|---|---|
| SAE, uncorrected | `dec` | concept-feature decoder direction |
| | `enc` | encoder gradient of the target pre-activations |
| SAE, realizability-corrected | `pinv` | min-norm edit that moves pre_T while holding every active protected pre-activation |
| | `ridge` | Tikhonov between `enc` and `pinv` |
| | `dec_proj` | decoder direction projected onto the null space of the active protected rows |
| | `*_fs` | finite-step correction: features the edit would switch on are added to the constraints and the solve is repeated (constraint generation) |
| | `opt` | projected-gradient activation optimization with the full encoder |
| generic | `diffmean` | mean difference of concept vs. other-concept residuals |
| | `random` | fixed random direction |
| controls | `dec_rand_feat` | decoder row of a random feature |
| | `dec_proj_randP` | same projection, onto a count-matched random protected set |
| reference | `prompt` | concept cue in the prompt; not an activation edit |

## 4. Development findings (dev concepts × dev prompts only)

<!--DEV_BLOCK-->

## 5. Held-out results: primary setting (the minimal decisive experiment)

<!--HELDOUT_BLOCK-->

## 6. Generalization: another SAE on the same site (TopK + LayerNorm)

<!--TOPK_BLOCK-->

## 7. H2: does the realizability diagnosis predict behavior?

<!--H2_BLOCK-->

## 8. Cost

<!--RUNTIME_BLOCK-->

## 9. Deviations, limitations, and what is *not* claimed

<!--LIMITS_BLOCK-->
