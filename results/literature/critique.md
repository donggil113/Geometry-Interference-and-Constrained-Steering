## Missing work
All entries below were confirmed on arXiv abs pages or official pages during this pass. Unless a section is named, each claim comes from the abstract.

- **The six papers in §1.4 exist and can be verified, so the reason given for excluding them no longer holds.** Their arXiv IDs match these titles:
  - 2608.28806, *Enhancing SAE-based Steering via Neighbor Integrated Feature Selection* (Liu, Wang, Zou). Useful steering features are spread across neighbour groups created by feature splitting. https://arxiv.org/abs/2608.28806
  - 2603.18353, *Interpretability without actionability* (Basu et al.). SAE feature steering "produced zero effect despite 3,695 significant features". Concept-bottleneck steering was "indistinguishable from random perturbation (p=0.84)". https://arxiv.org/abs/2603.18353
  - 2607.20596, *Are Single-Token SAE Features Causally Necessary?* (Cho et al.). On the same base model, "GemmaScope and BatchTopK features are causally anchored, LlamaScope features locally redundant". Cross-family claims are "sensitive to training methodology". This bears directly on the held-out-SAE requirement (N8). https://arxiv.org/abs/2607.20596
  - 2609.30218, *Minimally Invasive Steering* (MISVO; Entesari et al.). Pre-logit steering is penalised with a local-KL Fisher quadratic. This is prior art next to FishBack [61]. https://arxiv.org/abs/2609.30218
  - 2605.23040, *Steered Generation via Gradient-Based Optimization on Sparse Query Features* (Bhattacharyya, Rooshenas). Gradient optimisation of SAE codes, applied to attention queries. This is prior art for the direct-optimisation arm. https://arxiv.org/abs/2605.23040
  - 2602.04903, *Mind the Performance Gap* (Sprejer et al.). With Goodfire Auto Steer, MMLU falls from 66% to 46% on Llama-8B, and "simple prompting achieves the best overall balance". https://arxiv.org/abs/2602.04903
  - 2401.12631, the Wu et al. reply to Makelov et al., also exists. https://arxiv.org/abs/2401.12631
- **Templeton et al. 2024, *Scaling Monosemanticity*.** This is where Err(h)-preserving clamping comes from: the SAE(x) term is replaced "and leave[s] the error term unchanged", and it is applied "at every token position". Clamping was at 5–10× the maximum activation. P3's "clamping with Err(h)" arm should cite it. https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
- **Ameisen et al. 2025, *Circuit Tracing* (methods).** It defines "constrained patching" as opposed to "iterative patching". It also warns that adding a decoder vector at each layer "risk[s] double counting a feature's effect". This matters for P10 persistence and any multi-layer arm. https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- **Wollschläger et al., arXiv:2502.17420.** "orthogonality alone does not imply independence under intervention". They define *representational independence*. This is a direct precedent for T/P interference not being readable from geometry. https://arxiv.org/abs/2502.17420
- **da Silva & Heimersheim, arXiv:2606.24964 (Jun 2026).** Activations sit on plateaus that are robust to small perturbations. The response exponent is p>2 for SAE-decoder directions and p≈2 for random directions. So an isotropic, norm-matched random control is weak by construction, and small finite steps may be absorbed. https://arxiv.org/abs/2606.24964
  - Related: Giglemiani et al., arXiv:2409.15019. Synthetic activations composed of SAE latents show "less pronounced activation plateaus". https://arxiv.org/abs/2409.15019
- **Gur-Arieh et al., arXiv:2501.08319 (ACL 2025)** and **Paulo et al., arXiv:2410.13928 (intervention scoring).** Input-centric labels "fail to capture the causal effect of the feature on outputs". This bears on the construct validity of judging T by its Neuronpedia label. https://arxiv.org/abs/2501.08319 ; https://arxiv.org/abs/2410.13928
- **SteeringSafety (Siu et al., arXiv:2509.13450, ICML 2026).** All methods show "substantial entanglement", with social behaviours degrading by up to 76%, and "DIM is consistently effective". This is a ready-made template for behavioural collateral. https://arxiv.org/abs/2509.13450
- **Exact piecewise-linear verification and recourse.** This literature already covers exact reasoning across ReLU gates, minimum-distortion search with certificates, and integer-programming feasibility certificates. It bounds the novelty claimed in §8.2(a).
  - Tjeng et al., MIPVerify, arXiv:1711.07356: https://arxiv.org/abs/1711.07356
  - Katz et al., Reluplex, arXiv:1702.01135: https://arxiv.org/abs/1702.01135
  - Ustun et al., arXiv:1809.06514: https://arxiv.org/abs/1809.06514
- **AlphaEdit, arXiv:2410.02355.** It projects edits onto "the null space of the preserved knowledge". This is the source of AlphaSteer [20], and N3's null-space P-preservation needs to cite it. https://arxiv.org/abs/2410.02355
- **Held-out SAE settings and infrastructure.**
  - Matryoshka SAEs, arXiv:2503.17547, report reduced absorption. https://arxiv.org/abs/2503.17547
  - Gemma Scope 2 (DeepMind, 19 Dec 2025) covers Gemma 3 with SAEs and transcoders, trained with Matryoshka, which "resolves certain flaws discovered in Gemma Scope". https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/
  - Kissane et al., AF 2024: base-model SAEs "don't transfer on Gemma v1 2B" and fail on high-norm BOS tokens. This is relevant to P1, which applies base SAEs to the IT model. https://www.alignmentforum.org/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models
- **Steerability predictors N7 must beat.**
  - Braun et al., arXiv:2505.22637: cosine agreement of training differences predicts steering success. https://arxiv.org/abs/2505.22637
  - Billa, arXiv:2604.15557: the logit-lens LAP predicts effectiveness at ρ≈0.86–0.91. https://arxiv.org/abs/2604.15557
  - Tan et al., arXiv:2407.12404: steerability is "highly variable across different inputs". https://arxiv.org/abs/2407.12404
- **Other 2025–2026 SAE-steering work absent from §4.**
  - SAS (Bayat et al.), arXiv:2503.00177: https://arxiv.org/abs/2503.00177
  - Feature Flow (Laptev et al.), arXiv:2502.03032, cross-layer feature maps for steering: https://arxiv.org/abs/2502.03032
  - CircuitSteer, arXiv:2608.05732, multi-layer SAE circuits: https://arxiv.org/abs/2608.05732
  - REINS (EMNLP 2026), arXiv:2608.28233, suppresses and enhances features jointly and notes "apparent safety through collapse": https://arxiv.org/abs/2608.28233
  - ContrastiveSteer, arXiv:2506.12576, with its "contamination" off-target metric: https://arxiv.org/abs/2506.12576
  - SAE-StatSteer, arXiv:2607.19364, where "raw success systematically overstates usable control": https://arxiv.org/abs/2607.19364
  - MAxBench, arXiv:2609.13072, where "no method consistently outperforms prompting": https://arxiv.org/abs/2609.13072
  - MAT-Steer, arXiv:2502.12446, orthogonality and sparsity to reduce inter-attribute conflict: https://arxiv.org/abs/2502.12446
  - CAST, arXiv:2409.05907, conditional steering as a state-dependent baseline: https://arxiv.org/abs/2409.05907

## Overstated claims
- **§7.6(1), "Infeasibility at a norm budget can be certified by an LP (Farkas duality)".** This is wrong for an L2 budget.
  - Farkas or an LP certifies only the linear system with no budget, or with an L1/L∞ budget.
  - For L2, the certificate is the QP's optimal value exceeding B, via QP or SOCP duality.
  - Strict inequalities for inactive P need a margin. So the result is "exact up to margin and solver/bf16 tolerance".
- **§7.6(1), "exact / no Taylor approximation".** This holds only at the same-SAE, same-token, same-layer readout, which is Tier 0, the tier the audit itself flags as biased.
  - Exactness there says nothing about Q2.
  - The certificate may also be degenerate. d_model=2304 is far larger than the tens to hundreds of active-P equality constraints, so same-layer feasibility is generically guaranteed. Only the min-norm value and the roughly 16k inactive-set inequalities carry information. The audit never checks this.
- **§8.2(a), "exact finite-step feasibility across gate flips" presented as open.** Exact MILP/LP analysis over ReLU activation patterns with certificates is standard (Reluplex, MIPVerify, recourse IP). The open part is narrower: applying it to SAE (T, P) requests and showing that it *predicts behaviour*.
- **§5.5(5), encoder-gradient "optimal by construction".** This is true only for a single T, with no P, under ReLU/JumpReLU.
  - With P constraints, the minimum-norm solution is w_T projected off the span of the active-P rows.
  - Under TopK, raising a_T does not guarantee entry into the active set.
  - Also state that the P-constrained encoder arm and the QP arm can coincide, since that collapses two of the planned arms.
- **P1, "apply edits as h+δ with the original e kept, as SpARE and Jørgensen & Hansen do".** For residual-space arms, e(h+δ) ≠ e(h). SpARE and J&H edit in code space and add a decoded delta, which is a different convention. The audit conflates the two, so it misstates what is actually held fixed.
- **§10 GO rule.** GO requires N1–N3 and N9 but no pass on behavioural P-preservation or collateral (N6, N12), even though Q1 is about preservation.
  - N1 compares only against DiffMean.
  - The user's NO_GO ("parity with a generic baseline") also needs "beats uncorrected decoder and encoder steering on identical T at matched norm" and "beats random" as GO conditions.
- **§1.4 exclusions are now stale.** All six excluded IDs resolve to real papers (see Missing work). The completeness claim in §1.5 understates the gap.
- **Tier 1 "TPP-style probes" [57].** This sits uneasily with [35] ("should not be used"). The audit should say why a steering adaptation escapes that critique.

## Missing protocol requirements
- **Split the arms into two tiers, as brief item 8 implies.**
  - *Minimal decisive set*: decoder, encoder-row, ridge/QP, direct optimisation, DiffMean, random, and Err-clamping. All norm-matched, one model, one hook.
  - *Extended set*: SAE-TS, FGAA, COAST, S&P, RePS. P3 currently makes all of them mandatory before the decisive test.
- **Pre-register the construction of T and P.** This covers:
  - the rule for building P (all active latents, decoder-cosine neighbours, or an attribute with ground truth), plus a sweep over |P|
  - the edit direction (raise vs suppress, since ReLU puts a floor at 0)
  - the Δz_T target in natural units (activation quantile, not a multiple beyond the maximum)
  - the budget B, defined relative to ‖h‖ or the DiffMean norm used
- **Degeneracy check.** Report the distribution of certificate and min-norm values on dev. If feasibility is near-universal or constant, N7 is untestable and counts as NO_GO for the diagnosis claim.
- **Adaptive-magnitude control.** Per-token QP edits are state-dependent. Include state-dependent generic baselines:
  - Err-clamping to the same z_T target
  - per-token target-matched decoder and DiffMean
  - CAST-style conditional application

  Otherwise "realizability" is confounded with adaptive magnitude.
- **Stronger random controls.** Add norm-matched random *SAE decoder rows* or feature directions alongside isotropic random, because feature directions are privileged (2606.24964).
- **Intent-to-treat.** Infeasible requests and solver failures count as failures, or fall back to a pre-registered default. They must not be dropped.
- **Construct validity.** The concept is currently defined by the SAE's input-centric label. Use output-centric or intervention-scored labels, or concepts defined independently of the SAE.
  - Behavioural P-readouts need ground truth (RAVEL-style) or an output-centric logit footprint of d_P, not a judge scoring auto-interp labels.
- **Off-manifold diagnostics for h+δ.**
  - SAE FVU at h+δ and in downstream SAEs
  - norm ratio
  - Δe, reported per arm

  These flag solutions that game the encoder.
- **Generation-time mechanics.**
  - Re-solve per token as h drifts, and handle the KV cache.
  - Exclude BOS and other high-norm tokens.
  - Run encoder and QP in fp32, report the fraction of latents within ε of θ, and report solver time per generated token.
- **Held-out SAE mapping rule.** Fix on dev how a concept maps to features in another SAE (decoder cosine or activation correlation).
  - The cheapest held-out setting is another GemmaScope L0 or width at the same hook.
  - Gemma Scope 2 (Matryoshka) changes the model and is a second step.
- **Mediation and reversal test for N15.** After the edit, restore T to its original value downstream and check that the behaviour reverts. This addresses the dormant-pathway concern of [56].
- **Evaluation hygiene.**
  - Equal dev tuning budget per arm.
  - Judge blinded to arm and order.
  - A second judge family, or a fallback if gpt-4o-mini-2024-07-18 is unavailable.
  - Note that the same model generates the concept data and judges it.
  - Several samples per prompt at T=1.0.
  - Multiple-comparison correction across arms and cells.
  - A timestamped pre-registration (commit hash).
- **N7 predictor baselines.** Add Braun directional agreement and Billa's LAP to the geometry predictors [13] and [29].