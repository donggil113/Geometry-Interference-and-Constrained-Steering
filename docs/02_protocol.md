# N3 protocol (pre-registered before any held-out run)

> **Reading note, added after the held-out run.** §7 (frozen decisions and amendments A1–A6) supersedes
> the original plan in §3 and §6. Specifically, §7 adds the entity family (16 held-out concepts, not 8),
> the concept-level sign-flip test with Holm correction, the absolute fluency guard, and H2 over 24
> concepts. The H1 decision was NO_GO under **both** the §6 and the §7 rules.
>
> **Disclosures.**
> - (i) `configs/splits.json` with amendment A1 was accidentally left out of the pre-registration
>   commit `4282fb5` and was committed afterwards. The seeded function that creates it,
>   `concepts.amend_splits_entity_family`, *was* committed before the freeze, and it regenerates
>   exactly the 8 classes used.
> - (ii) The switch from a relative to an absolute fluency guard (commit `fb23c60`) was made after the
>   dev analysis, 27 s before the freeze. On dev it favoured the primary method. It does not affect the
>   NO_GO outcome.
> - (iii) `protocol.json` records `frozen_at_commit = fb23c60`, the parent of the freeze commit. The
>   code fingerprint is identical.
> - (iv) `dec_rand_feat` is frozen at K = 1: a single random feature, at the primary's α.

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

## 7. Decided on dev: frozen in `configs/protocol.json`

All values below come from the `dev2` run: 8 dev concepts × 24 dev prompts, GPT-2 with the jb ReLU
SAE, after the centring fix.

- **Fixed a priori:**
  - `B_NLL` = 1.0 nats/token.
  - Degeneration floor: dDist2 ≥ −0.10.
  - α grid {0.1, 0.2, 0.3, 0.45, 0.65} × median ‖h‖ (78.9). α = 0.9 was dropped mid-dev because
    every method exceeds B_NLL there.
  - P = all active non-target features (`protect_topm` = None), `ridge_rel` = 1, `fs_rounds` = 3,
    `opt` μ_P = μ_new = 1.
- **Selected on dev:** each method's (K, α) = the largest concept-balanced dev dC under the budget.

| method | K | α | dev dC | dev dNLL |
|---|---|---|---|---|
| dec_proj | 3 | 0.30 | 0.365 | 0.65 |
| dec | 3 | 0.30 | 0.337 | 0.63 |
| diffmean | – | 0.20 | 0.307 | 0.49 |
| dec_proj_fs | 3 | 0.30 | 0.260 | 0.53 |
| pinv | 3 | 0.30 | 0.219 | 0.74 |
| ridge | 3 | 0.30 | 0.216 | 0.82 |
| enc | 3 | 0.30 | 0.195 | 0.95 |
| pinv_fs | 3 | 0.45 | 0.168 | 0.91 |
| opt | 1 | 0.20 | 0.082 | 0.13 |
| random | – | 0.10 | 0.017 | 0.06 |

- **PRIMARY_CORRECTED = `dec_proj`.** This is the decoder direction projected onto the null space of
  the active protected features' (centred) encoder rows. It is the argmax of dev dC over the corrected
  methods.
- **H1 comparators:** `dec`, `enc`, `diffmean`, `random`, each at its own frozen point.
- **Controls, run at the primary's (K, α):**
  - `dec_rand_feat`: the decoder row of a random feature.
  - `dec_proj_randP`: the same projection, but onto a count-matched *random* protected set.
- **Reference:** `prompt`, a concept cue prepended to the prompt. It is not an activation edit.

### Amendments made before any held-out run, each prompted by the code review

- **A1.** Held-out *family* `entity`: 8 seeded DBpedia-14 classes. Held-out concepts: 8 → 16.
- **A2.** Inference.
  - Primary test: an exact concept-level sign-flip test on per-concept mean dC differences, with
    Holm correction across the 4 comparators (α = 0.05).
  - The two-way (concept × prompt) cluster-bootstrap 95% CI must also exclude 0.
  - The earlier two-way bootstrap alone is anti-conservative with this few concepts.
- **A3.** Guards.
  - Degeneration: dDist2 difference ≥ −0.05.
  - NLI-judge concordance: the direction of the dC difference must agree under the zero-shot NLI
    judge. This matters because the concept texts come from the classifier judges' training sources.
  - Absolute fluency guard: the primary's held-out dNLL ≤ B_NLL + 0.1.
  - Collateral vs `dec`: protJS +0.01, dce +0.05, dRel −0.02.
  - The relative fluency guard was replaced before any held-out run, because comparators sit at their
    own dev-selected α. For example, `random` selects α = 0.1 at a cost of ≈ 0 nats.
- **A4.** Secondary norm-matched comparison: every comparator at the primary's α = 0.30. Reported,
  not part of GO.
- **A5.** H2 uses all 24 concepts (8 dev + 16 held-out), evaluated on the held-out prompts at the
  frozen α.
  - Primary diagnostic: `c_qp_rel`.
  - κ was found degenerate (between-concept CV ≈ 0.02–0.04; `docs/03_realizability.md`), so it cannot
    be the primary diagnostic.

### Amendment recorded after the held-out run (the change itself was made at freeze)

- **A6.** The H2 outcome is the held-out ΔC of `dec_proj` (the primary corrected method), not of
  `dec` as §6 states. The change was made when `configs/protocol.json` was frozen
  (`h2.outcome_method = dec_proj`) but was not written up here at the time.
  - Under the §6 outcome (`dec`), the primary diagnostic gives ρ = +0.07 (n = 24;
    `results/analysis/report_numbers.json`, key `h2_gpt2_relu_jb_L6_dec_outcome`). Under the frozen
    outcome it gives ρ = +0.05.
  - H2 is not supported either way.

The run tag and fingerprint are in `configs/protocol.json` (`frozen_at_commit`, `code_fingerprint`).
