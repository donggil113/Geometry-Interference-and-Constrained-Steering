# N3 protocol (pre-registered before any held-out run)

Status: **frozen** once `configs/protocol.json` is committed. Everything under "decided on dev" is
filled in from dev-split results only, and the commit hash that freezes it comes *before* the first
held-out generation (`results/runs/test_*`).

## 1. Question

Can a desired change of target SAE features **T** be realized while non-target (protected) features
**P** are preserved? When realizability is diagnosed and corrected (constrained edits), does that
help **independent downstream behavior**, beyond (a) the SAE encoder's own self-consistency scores
and (b) generic, SAE-free steering baselines?

## 2. Primary testbed (the minimal decisive experiment, step 8)

* Model: GPT-2 small (TransformerLens default processing), edit site `blocks.6.hook_resid_pre`.
* SAE: jbloom `gpt2-small-res-jb` layer 6 (ReLU, no input normalization, d_sae = 24576).
* Other SAE settings (generalization, frozen hyper-parameters transferred without re-tuning):
  OpenAI v5 32k TopK (k = 32, LayerNorm-normalized input) at the *same* site; Gemma Scope 2 JumpReLU
  on Gemma-3-270m layer 12 (realizability analysis; behavior only if the primary result is GO).
* All pins: `configs/pinned.json`; validation: `results/validation/checkpoint_validation.json`.

## 3. Concepts, prompts, splits (`configs/splits.json`, seed 20260926)

* 16 concepts from two families with independent supervised judges:
  10 Yahoo! Answers topics and 6 dair-ai/emotion labels.
* Dev concepts (8): Health, Education & Reference, Sports, Business & Finance, Entertainment & Music,
  joy, fear, surprise. Held-out concepts (8): Society & Culture, Science & Mathematics,
  Computers & Internet, Family & Relationships, Politics & Government, sadness, love, anger.
* Prompts: 12-token OpenWebText prefixes (disjoint document ranges); 24 dev prompts and 48 held-out
  prompts.
* Concept texts for feature selection and DiffMean come only from the training splits of the
  concept datasets. The judges are validated on the test splits
  (`results/validation/judge_validation.json`).

## 4. Methods (all norm-matched: ||delta|| = alpha x median residual norm, at every non-BOS position)

`dec`, `enc`, `pinv`, `ridge`, `dec_proj`, `pinv_fs`, `dec_proj_fs`, `opt` (SAE-based, see
`src/n3/methods.py`), plus the generic `diffmean` and `random` baselines and the unsteered `none`.
The protected set P at a position = the features active there, excluding T.

## 5. Measurements (kept separate)

* **Encoder self-consistency (internal):** target activation gain, relative change of the protected
  activations, number of newly activated features, all measured with the same SAE encoder. These are
  never used as evidence of behavioral benefit.
* **Behavior (independent judges, relative to the same-prompt, same-noise unsteered continuation):**
  * dC: target-concept probability from the family judge.
  * dNLL: fluency cost, from Qwen2.5-0.5B conditional NLL.
  * dRel: relevance to the prompt, from MiniLM cosine.
  * protJS: Jensen–Shannon shift of the *other* family's judge, as a behavioral collateral measure.
  * dDist2: degeneration (change in distinct-2).
* **Utility:** dCE and KL on held-out OWT text with the edit applied.
* **Runtime:** ms per position for the edit computation, plus generation wall-clock.

## 6. Hypotheses and decision rules

* **H1 (correction helps behavior).** On held-out concepts × held-out prompts (primary setting), the
  corrected method selected on dev (one method, `PRIMARY_CORRECTED`) achieves a higher dC than both
  `dec` and `diffmean`. Each method is run at its own dev-selected alpha, where selection means the
  largest dev dC subject to dev dNLL <= `B_NLL`.
  * Test: paired two-way cluster bootstrap over concepts and prompts (5000 resamples).
  * **GO** requires lower 95% bound > 0 against **both** comparators.
  * It also requires that the corrected method's held-out dNLL does not exceed the comparator's by
    more than 0.1 nats. This prevents buying concept with fluency.
  * Robustness (reported, not required): the matched-fluency Pareto comparison, the NLI judge, and
    the TopK setting.
* **H2 (diagnosis predicts behavior).** The per-concept realizability diagnostics are computed on
  neutral positions without looking at behavior: median kappa, exact-QP cost, and decoder
  collateral.
  * They are correlated (Spearman) with per-concept `dec` behavioral dC at the dev-selected alpha,
    over all 16 concepts.
  * H2 is supported only if |rho| >= 0.5 with a bootstrap CI excluding 0, **and** it beats the simple
    baselines: feature density, encoder–decoder cosine, and DiffMean dC.
* **NO_GO** for H1 if any of the following holds:
  * Improvements appear only in encoder self-consistency metrics.
  * The behavioral CI includes 0 or is negative against `dec` or `diffmean`.
  * The corrected method is statistically indistinguishable from `diffmean`/`random`.
* Large SAE retraining is out of scope unless H1 is GO.

## 7. Decided on dev (filled from `results/runs/dev_*` only; see `configs/protocol.json`)

* alpha grid, `B_NLL`, K (number of target features), P rule, ridge lambda, and `PRIMARY_CORRECTED`:
  see `configs/protocol.json`.
