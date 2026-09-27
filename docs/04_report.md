# N3 report: Which SAE feature edits are realizable? Geometry, interference, and constrained steering

*Status: **NO_GO**. H1 was decided on held-out data under the pre-registered rule (freeze commit `4282fb5`); H2 is not supported. Both results were independently re-verified.*

## 0. Verdict (pre-registered decision)

**NO_GO.** Realizability-aware (constrained) SAE edits do not improve independent behavior over plain
decoder steering, and they do not beat the generic DiffMean baseline. The pattern is exactly the one the
brief defines as NO_GO: "같은 encoder 점수만 좋아지거나 generic baseline과 같으면 NO_GO".

1. **Q1: can T be changed while P is preserved? At the SAE's own readout, yes, cheaply, with one
   structural exception.** (`docs/03_realizability.md`)
   - For ReLU and JumpReLU SAEs, the exact finite-step QP realizes a typical target change with *every*
     active feature preserved and *every* inactive feature kept off. It needs 4–15% of ‖h‖, and 100% of
     requests are feasible.
   - Interference inflates the cost by only 7–29%.
   - Local Jacobian/pseudoinverse solutions are not finite-step valid: they flip 9–74 gates, and the
     first flip occurs 3–11% of the way along the step. The exact correction adds only 1–8% norm.
   - Under TopK+LN, full P preservation is structurally impossible in >99% of requests, because of
     slot eviction.
2. **Q2: does diagnosing or correcting realizability help downstream behavior? No.** This is the
   minimal decisive experiment: GPT-2 small, jb ReLU SAE, 16 held-out concepts × 48 held-out prompts.
   - The pre-registered primary corrected edit (`dec_proj`) removes all protected-feature change and
     45% of spurious activations *inside the SAE*. Behaviorally it is not distinguishable from plain
     decoder steering: ΔC +0.013, CI [−0.033, +0.058], p = 0.37.
   - It is not distinguishable from DiffMean at the frozen operating points: +0.052, p = 0.32.
   - At *matched norm*, DiffMean is significantly better: −0.115, p = 0.032.
   - A count-matched *random* protected set gives the same behavior.
   - Methods that maximize the encoder's own target score (enc, pinv, ridge, opt) steer behavior
     *worse*. Within each strength, Tier-0 target gain vs Tier-2 ΔC has ρ ≈ −0.1.
   - The same null replicates on a held-out TopK SAE at the same site: `dec_proj` − `dec` = −0.010,
     p = 0.34.
3. **H2: the realizability diagnostic does not predict steerability.** Exact QP cost vs held-out ΔC over
   24 concepts: ρ = +0.05, CI [−0.32, +0.41]. The best predictor is the concept's generic DiffMean
   steerability (ρ = +0.51).

**What remains positive.**
- The selected SAE concept features *are* causally meaningful: `dec` beats a random decoder row by
  +0.30 to +0.39 ΔC.
- The exact finite-step analysis cleanly separates "realizable inside the SAE" from "moves behavior".
  The binding constraint on behavior lies outside the SAE's same-layer encoder geometry.
- Per the brief ("거대 SAE 재학습은 핵심 가설을 확인하기 전에 시작하지 마라"), no SAE retraining is warranted.


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

Data: 8 dev concepts × 24 dev prompts, 8 SAE methods × K ∈ {1, 3} and 2 generic methods, over
α ∈ {0.1, 0.2, 0.3, 0.45, 0.65}. That is 722 conditions and 17,328 judged continuations
(`results/analysis/dev2/`).

**(a) Matched norm, α = 0.30, K = 3.** Tier 0 is the same-SAE readback; Tier 2 is the independent
judges.

| method | ΔC (Tier 2) | ΔNLL | ΔCE (OWT) | target gain (Tier 0) | ‖Δf_P‖/‖f_P‖ (Tier 0) | new features/pos (Tier 0) | Δerror/‖δ‖ | ms/pos |
|---|---|---|---|---|---|---|---|---|
| dec_proj | **0.365** | 0.65 | 0.35 | 4.9 | **0** | 38 | 0.72 | 1.9 |
| dec | 0.337 | 0.63 | 0.37 | 5.5 | 0.26 | 55 | 1.25 | 0.08 |
| dec_proj_fs | 0.260 | 0.53 | 0.27 | 2.7 | 0 | **0.006** | 0.81 | 9.8 |
| pinv | 0.219 | 0.74 | 0.33 | 9.9 | 0 | 192 | 2.12 | 1.9 |
| ridge | 0.216 | 0.82 | 0.36 | 11.9 | 0.21 | 332 | 3.57 | 2.1 |
| enc | 0.195 | 0.95 | 0.41 | **12.8** | 0.78 | 1157 | **11.9** | 0.11 |
| (K = 1) diffmean | 0.441 | 1.04 | 0.41 | 1.0 | 0.28 | 57 | 1.26 | 0.01 |
| (K = 1) random | 0.005 | 0.49 | 0.16 | 0.0 | 0.23 | 40 | 1.10 | 0.01 |

**(b) Encoder self-consistency does not transfer to behavior.**
- Across all SAE methods, K and concepts, the Spearman correlation between Tier-0 target gain and
  Tier-2 ΔC is computed *within* each α, because pooling over α would let strength drive the
  correlation. It is −0.10 at every α ≥ 0.2 (n = 128 cells each).
- The methods that raise the SAE's own target score most per unit norm steer behavior worst:
  enc, ridge and pinv.
- They also push the residual furthest off the SAE's manifold. Their error-channel change is 2–12×
  the edit norm, against 0.7× for `dec_proj`.
- The number of *spuriously activated* features does track fluency damage: ρ(new_active, ΔNLL) =
  +0.34, +0.60 and +0.66 at α = 0.3, 0.45 and 0.65.

**(c) Budget-selected operating points** (ΔNLL ≤ 1.0, distinct-2 floor −0.10). These are frozen in
`configs/protocol.json`.
- dec_proj 0.365 > dec 0.337 > diffmean 0.307 > dec_proj_fs 0.260 > pinv 0.219 ≈ ridge 0.216 >
  enc 0.195 > pinv_fs 0.168 > opt 0.082 > random 0.017.
- `dec_proj` was therefore pre-registered as the primary corrected method.
- Its dev lead over `dec` is small (+0.028) and not consistent across concepts: it is ahead on 6 of 8.


## 5. Held-out results: primary setting (the minimal decisive experiment)

Setting: 16 held-out concepts in three families. Five topics and three emotions were never used in
dev, and the entity family was entirely unseen (amendment A1). There are 48 held-out prompts. Every
arm runs at its dev-frozen operating point, with the same sampling noise per prompt.
Files: `results/analysis/test1/decision_gpt2_relu_jb_L6.json`, `results/logs_decide_test1.txt`.

**5.1 Pre-registered H1 decision: NO_GO.** Primary: `dec_proj`, K = 3, α = 0.30.

| dec_proj minus … | ΔC diff | two-way bootstrap 95% | exact sign-flip p (two-sided) | concepts > 0 | Holm |
|---|---|---|---|---|---|
| dec (K3, 0.30) | +0.013 | [−0.033, +0.058] | 0.37 | 11/16 | not rejected |
| diffmean (0.20) | +0.052 | [−0.056, +0.159] | 0.32 | 10/16 | not rejected |
| enc (K3, 0.30) | +0.182 | [+0.102, +0.269] | 1e-4 | 15/16 | rejected |
| random (0.10) | +0.339 | [+0.223, +0.461] | <1e-4 | 16/16 | rejected |

- **Guards:** all passed. The primary's ΔNLL is 0.875 ≤ 1.1. Degeneration and NLI concordance pass,
  and collateral is not worse than `dec`.
- **Why NO_GO:** the corrected edit is **not shown to be better** than plain decoder steering or than
  the generic DiffMean baseline. This is "not shown better", not "shown equivalent": the upper bound
  against `dec` is +0.058, about 18% of dec's lift.
- **Rule robustness:** NO_GO also holds under the original §6 rule (bootstrap only, relative fluency
  guard).
- **Independent reproduction:** an adversarial verification workflow recomputed everything from the
  raw generation and score files with independent code. Numbers matched to 4 decimals, and
  provenance, pairing and fingerprints were confirmed (`results/review/`).

**5.2 Arm levels.** Concept-balanced means, each arm at its frozen point.

| arm | ΔC | ΔNLL | ΔRel | protJS | ΔCE | layer-8 SAE drift (Tier 1) | ‖Δf_P‖/‖f_P‖ (Tier 0) | new feats/pos (Tier 0) | Δerror/‖δ‖ |
|---|---|---|---|---|---|---|---|---|---|
| dec_proj | 0.341 | 0.875 | −0.108 | 0.260 | 0.302 | 0.312 | **0.000** | 44 | 0.74 |
| dec | 0.328 | 0.851 | −0.110 | 0.271 | 0.315 | 0.343 | 0.306 | 80 | 1.97 |
| dec_proj_randP (control) | 0.322 | 0.819 | −0.109 | 0.268 | 0.307 | 0.337 | 0.304 | 83 | 1.94 |
| diffmean (α 0.2) | 0.289 | **0.438** | −0.075 | 0.247 | 0.169 | 0.244 | 0.180 | 28 | 1.06 |
| enc | 0.159 | 0.980 | −0.071 | 0.275 | 0.336 | 0.390 | 0.801 | 1186 | 13.9 |
| dec_rand_feat (control) | 0.028 | 0.554 | −0.046 | 0.251 | 0.207 | 0.321 | 0.283 | 64 | 1.93 |
| random (α 0.1) | 0.002 | 0.007 | −0.019 | 0.174 | 0.013 | 0.097 | 0.079 | 7 | 1.04 |
| prompt cue (reference) | 0.078 | 0.114 | −0.022 | 0.254 | 0 | 0 | – | – | – |

**5.3 What the correction changes, tier by tier.** `dec_proj` minus `dec`, paired, 16 concepts.

- **Tier 0, the edited layer's SAE:**
  - protected-feature change: 0.31 → 0.000;
  - spurious new activations: 80 → 44;
  - error-channel disturbance: 1.97 → 0.74 of ‖δ‖.
- **Tier 1, layer-8 SAE on held-out OWT text:**
  - protected drift: 0.343 → 0.312 (16/16 concepts, p < 1e-4);
  - new activations: 52 → 36.
  - About 90% of the layer-6 preservation has already been lost by layer 8.
- **Tier 2, behavior:**

  | metric | dec_proj − dec | p |
  |---|---|---|
  | ΔC | +0.013 | 0.37 |
  | NLI | +0.022 | 0.18 |
  | protected-attribute JS | −0.012 | 0.20 |
  | ΔRel | +0.002 | 0.72 |
  | ΔNLL | +0.024 | 0.48 |
  | ΔCE | −0.013 | 0.074 |

  None is significant.
- **Specificity control:** replacing the active protected set with a count-matched *random* set of
  features gives the same behavior. `dec_proj` − `dec_proj_randP`: ΔC +0.019 (p = 0.16); ΔNLL,
  protJS and ΔCE all n.s. Only layer-8 drift differs (−0.026, 16/16).

The correction does exactly what it is designed to do inside the SAE, and nothing that the
independent judges can detect. This is the brief's NO_GO pattern: "only the encoder score improves".

**5.4 Against the generic baseline (secondary and exploratory, not part of the rule).**

- **Norm-matched (pre-registered secondary A4; every comparator at α = 0.30, not multiplicity-corrected).**

  | dec_proj minus … | ΔC diff | 95% CI | sign-flip p | ΔNLL diff |
  |---|---|---|---|---|
  | diffmean | **−0.115** | [−0.223, −0.013] | 0.032 | −0.03 |
  | dec | +0.013 | | 0.37 | |
  | enc | +0.182 | | 1e-4 | |
  | random | +0.339 | | <1e-4 | |

  At the same edit norm, the generic DiffMean direction gives a *larger* concept lift than the
  realizability-corrected SAE edit, at essentially the same fluency cost.

At their frozen points:

- `dec_proj` costs significantly more fluency than DiffMean: ΔNLL +0.44 (p = 0.0002, 15/16). It also
  costs more ΔCE (+0.13) and more downstream drift (+0.07, 16/16).
- Its lift is not significantly higher.
- Its small ΔC edge over DiffMean comes almost entirely from the entity family (+0.096). Without that
  family it is +0.007.
- Penalized scores (ΔC − λ·ΔNLL) favour DiffMean for any λ ≳ 0.12 per nat. At λ = 0.46, the dev local
  slope of `dec_proj`, the difference is −0.149 (p = 0.011, uncorrected, λ chosen post hoc).
- **Matched fluency** (held-out curves over α ∈ {0.1, 0.2, 0.3, 0.45, 0.65}; linear interpolation at a
  fixed ΔNLL; per-concept paired sign-flip tests are post hoc; figure `docs/figures/pareto_relu_heldout.png`).

  | ΔC at ΔNLL = | 0.25 | 0.5 | 1.0 |
  |---|---|---|---|
  | DiffMean | 0.18 | 0.31 | 0.47 |
  | dec | 0.15 | 0.24 | 0.36 |
  | dec_proj | 0.13 | 0.22 | 0.36 |

  - Per concept, `dec_proj` − `dec` is −0.014 (p = 0.21) at 0.5 nats and −0.000 (p = 0.97) at 1.0 nats.
  - `dec_proj` − DiffMean is −0.093 (p = 0.048) and −0.106 (p = 0.08).
  - DiffMean lies on or above both SAE curves at every fluency cost. The correction moves the SAE curve
    nowhere.

![Held-out Pareto](figures/pareto_relu_heldout.png)
![Tier 0 vs Tier 2](figures/internal_vs_behavior_relu.png)

**5.5 Controls and references.**

- The concept features' decoder directions carry real concept information: `dec` − `dec_rand_feat` =
  +0.30 (p = 0.0005, 14/16). Feature selection works; the correction does not add to it.
- A prompt cue barely steers base GPT-2 small (ΔC +0.08). This differs from AxBench, whose models are
  instruction-tuned.


## 6. Generalization: another SAE on the same site (TopK + LayerNorm)

- **Setup.** The held-out SAE is OpenAI v5 32k TopK (k = 32) with LayerNorm input, at the *same*
  residual site (`blocks.5.hook_resid_post` equals `blocks.6.hook_resid_pre`).
  - All dev-frozen hyper-parameters (K, α, methods) are transferred without re-tuning.
  - Feature selection re-applies the frozen rule to this SAE's statistics.
  - Same 16 held-out concepts × 48 held-out prompts.
  - After H1 was decided, the grid was reduced to the decision arms plus controls
    (`results/runs/topk1_NOTES.txt`).
  - Files: `results/analysis/topk1/`.

| arm (frozen point) | ΔC | ΔNLL | protJS | ΔCE | Tier 0: ‖Δf_P‖/‖f_P‖ | Tier 0: new feats/pos |
|---|---|---|---|---|---|---|
| dec_proj | 0.400 | 0.652 | 0.256 | 0.356 | 0.172 | 7.3 |
| dec | 0.410 | 0.625 | 0.280 | 0.369 | 0.200 | 6.6 |
| dec_proj_randP (control) | 0.413 | 0.630 | 0.272 | 0.362 | 0.198 | 6.7 |
| enc | 0.330 | 0.705 | 0.276 | 0.331 | 0.250 | 10.1 |
| diffmean (α 0.2) | 0.289 | 0.438 | 0.247 | 0.169 | 0.161 | 6.4 |
| dec_rand_feat (control) | 0.023 | 0.530 | 0.245 | 0.213 | 0.195 | 8.4 |
| random (α 0.1) | 0.002 | 0.007 | 0.174 | 0.013 | 0.085 | 2.7 |

| dec_proj minus … | ΔC diff | 95% CI | sign-flip p | concepts > 0 |
|---|---|---|---|---|
| dec | **−0.010** | [−0.047, +0.028] | 0.34 | 7/16 |
| dec_proj_randP (count-matched random P) | −0.013 | [−0.050, +0.023] | 0.051 | 4/16 |
| enc | +0.070 | [+0.001, +0.139] | 0.031 | 11/16 |
| diffmean | +0.111 | [+0.023, +0.198] | 0.015 | 10/16 |
| random | +0.398 | [+0.282, +0.511] | <1e-4 | 16/16 |

- **The central null replicates on a different SAE architecture.** The realizability correction does
  not improve on plain decoder steering; the point estimate is slightly negative. It is behaviorally
  identical to projecting out a random protected set.
- **Even the Tier-0 benefit mostly disappears under TopK+LN.** Protected change drops only from 0.20 to
  0.17, because LayerNorm couples all features and T entering the top-k evicts protected ones. This is
  the structural conflict predicted in `docs/03_realizability.md`.
- **Against DiffMean.** With this SAE, the decoder-based arms beat DiffMean at the frozen points
  (+0.11, p = 0.015). They do so at a higher fluency and utility cost (ΔNLL 0.65 vs 0.44; ΔCE 0.36 vs
  0.17), so the comparison is not fluency-matched. The same holds for `dec`: the gain is not from the
  correction.
- The pre-registered rule applied to this setting also gives NO_GO (`dec` not rejected).


## 7. H2: does the realizability diagnosis predict behavior?

- **Pre-registered test.** Primary diagnostic `c_qp_rel`: the exact finite-step QP cost of raising
  T by its typical activation while preserving P, over ‖h‖, median over 48 neutral positions. Outcome:
  held-out-prompt ΔC of `dec_proj` at its frozen point. n = 24 concepts: 8 dev + 16 held-out, all
  evaluated on the held-out prompts.
- **Result: H2 is not supported.**

  | predictor | Spearman ρ with ΔC(dec_proj) | 95% bootstrap CI |
  |---|---|---|
  | **c_qp_rel (primary)** | **+0.05** | [−0.32, +0.41] |
  | c_lin_rel | +0.05 | [−0.31, +0.40] |
  | κ (interference inflation; degenerate, CV ≈ 0.02) | +0.19 | [−0.29, +0.60] |
  | c_dec_rel (decoder cost) | +0.15 | [−0.25, +0.50] |
  | decoder collateral ‖Δf_P‖/‖f_P‖ | +0.25 | [−0.15, +0.58] |
  | decoder spurious activations | +0.42 | [−0.02, +0.72] |
  | *baseline:* max decoder cosine (Khan et al.-style crowding) | −0.17 | [−0.51, +0.21] |
  | *baseline:* decoder neighbor density (cos > 0.3) | +0.13 | [−0.30, +0.55] |
  | *baseline:* log feature density | +0.34 | [−0.13, +0.71] |
  | *baseline:* encoder–decoder cosine | +0.18 | [−0.21, +0.50] |
  | *baseline:* **DiffMean ΔC of the same concept** | **+0.51** | **[+0.11, +0.80]** |

- **Reading.**
  - How costly a concept edit is to realize inside the SAE, including its interference with the active
    set, carries no information about how well the concept can be steered behaviorally.
  - The best predictor of SAE-based steerability is a *generic*, SAE-free quantity: how well DiffMean
    steers the same concept.
- Files: `results/analysis/h2_gpt2_relu_jb_L6.json`, `results/analysis/h2_table_gpt2_relu_jb_L6.csv`.


## 8. Cost

Measured on an idle CPU (4 threads) with `scripts/14_runtime.py`, each method at its frozen point
(`results/runtime/gpt2_relu_jb_L6.json`).

| method | edit cost, ms / position (n = 24) | edit cost, ms / position (n = 1024) | 24 prompts × 40 tokens generation, s (unsteered 4.39) |
|---|---|---|---|
| dec / diffmean / random / enc | 0.005–0.02 | 0.001–0.006 | 4.5–4.7 (+3–7%) |
| dec_proj / pinv / ridge | 0.53–0.57 | 0.65–0.77 | 5.43–5.48 (+24%) |
| dec_proj_fs / pinv_fs | 3.1–3.6 | 4.3–5.8 | 8.8–10.0 (+100–128%) |
| opt (20 projected-gradient steps, full encoder) | 14.4 | 10.8 | 20.2 (+360%) |

- The corrected methods cost 65–1800× more per edit than decoder or DiffMean steering.
- They buy only Tier-0 (SAE-internal) improvements.

**Realizability figure.**
![Realizability costs](figures/realizability_costs.png)
![H2](figures/h2_relu.png)

## 9. Deviations, limitations, and what is *not* claimed

**Not claimed.**
- No new theorem. The rank and feasibility of the linear system are trivial here: full row rank in 100%
  of cases. The exact QP is textbook convex optimization, and exact reasoning over ReLU activation
  patterns is standard in verification (Reluplex, MIPVerify).
- No global guarantee from local (Jacobian) solutions. Where the encoder is piecewise affine
  (ReLU/JumpReLU), the finite-step results are exact up to a 1e-3 margin and solver tolerance. The
  TopK+LN statements are first-order results checked with true forward passes.
- No claim that realizability-aware editing *hurts*. It is not shown to help, and at matched norm it is
  beaten by DiffMean.

**Deviations from the brief and the literature protocol** (see also the addendum to `docs/01_literature_audit.md`).
1. *Model scale and judge.* CPU only and no API keys, so we could not reproduce AxBench itself
   (Gemma-2-2B/9B-IT, GemmaScope, gpt-4o-mini judge). The decisive experiment uses GPT-2 small, a base
   model, with supervised classifier judges plus an NLI judge. Absolute concept scores are therefore not
   comparable to AxBench numbers. The *design* follows AxBench (factor selection on dev, held-out
   evaluation) and the audit's P-requirements where feasible.
2. *Extended arms not run:* SAE-TS, FGAA, COAST, S&P, RePS/HyperSteer. The minimal decisive arm set
   includes decoder, encoder, pinv/ridge, exact-finite-step, activation optimization, DiffMean, random
   and prompting.
3. *Single sample per prompt* (48 prompts × 16 concepts per arm), with common random numbers across
   arms.
4. *Concept texts vs judges.* Concept texts for feature selection and DiffMean come from the training
   splits of the same datasets the classifier judges were trained on. This is why NLI concordance was
   required. The NLI judge agrees in direction and significance.
5. *Held-out SAE setting.* The TopK+LN SAE sits at the same site in the same model. The Gemma-3-270m
   JumpReLU setting was used for realizability only. Its behavioral run was pre-registered as
   conditional on GO and was therefore not run.
6. *Mid-course fixes, all before any held-out run.*
   - The all-ones-direction centring fix (dev1 invalidated).
   - α = 0.9 dropped from dev.
   - Amendments A1–A5.
   - The fluency-guard change.
   - All are listed in `docs/02_protocol.md`. After H1 was decided, the TopK transfer grid was reduced
     to the decision arms plus controls (`results/runs/topk1_NOTES.txt`).
7. *Scoring order.* The NLI judge was run only for the rows needed by the decision rule, to save
   compute. The scoring order was prioritized: decision arms, then controls, then norm-matched points,
   then the others.
8. *Statistics.* H1 uses 16 held-out concepts. With an exact sign-flip test and Holm correction over 4
   comparators, a real but small advantage (for example +0.02 ΔC over `dec`) could be missed. The
   bootstrap CI against `dec` ([−0.033, +0.058]) bounds how large such an advantage could plausibly be.

