# Steps 4–5: finite-step realizability of (T, P) feature edits

Code: `src/n3/realizability.py`, `scripts/05_realizability.py`, `scripts/09_realizability_summary.py`.
Raw data: `results/realizability/<setting>.jsonl`. Summaries: `summary_<setting>.json` and
`per_concept_<setting>_K*.csv`.

## Definition

**Realizable (as used in N3).** An edit request (h, T, Δ_T, P) is *realizable within budget B at the
same-layer SAE readout* when some δ with ‖δ‖ ≤ B satisfies all three conditions:

- **(T)** the target activations reach f_T(h+δ) = τ_T, where τ_T = max(pre_T, θ_T) + Δ_T;
- **(P=)** every active non-target feature keeps its value;
- **(P0)** every inactive non-target feature stays off.

This differs from "prompt-realizability", the question of whether a steered state has a prompt
preimage (reference [75] and §8.1(15) of `docs/01_literature_audit.md`).

- Δ_t is the typical nonzero activation of feature t on OpenWebText.
- B = 1.0 × the median residual norm.
- T is the frozen top-K concept features. The same selection rule is used in the behavioral experiments.

## Exactness

**ReLU and JumpReLU without input normalization.** Pre-activations are affine in h. For GPT-2 the
SAE reads the mean-centred residual and the edits are restricted to 1⊥, because GPT-2 is invariant
to the residual mean. Under these conditions:

- (T) and (P=) are linear equalities.
- (P0) is a set of linear inequalities pre_j(h+δ) ≤ θ_j − margin, with margin 1e-3.
- The encoder is affine on the whole feasible set for every constrained feature.
- So min ‖δ‖ is an exact convex QP. We solve it by constraint generation over (P0) with Clarabel,
  adding at most 48 of the most-violated constraints per round.
- Every accepted solution is re-checked against **all** constraints with the float32 SAE encoder.

This is textbook convex optimization, also standard in network verification (Reluplex, MIPVerify).
It is **not** claimed as a new result. The informative quantities are the **cost** (minimum norm)
and the **finite-step gap** between the linear (Taylor) solution and the exact one. The rank
condition is trivially satisfied (|T| + |P| ≪ d; full row rank in 100% of cases), and it carries no
information.

**TopK + LayerNorm (OpenAI v5).** This encoder is not piecewise affine in h. We report:

- first-order costs;
- the true-forward error of the linear solution;
- a Gauss–Newton correction on the true encoder;
- the exact decoder cost along the decoder direction, by bracketing plus bisection on the true
  activations;
- the structural TopK conflict: if T is inactive and all k active features are protected, then
  (T) and (P=) are jointly infeasible, because T entering the top-k must evict one of them.

No global guarantee is claimed for this setting.

## Results: GPT-2 L6, jb ReLU SAE (primary), 30 concepts × 48 neutral OWT positions

| | K = 1 | K = 3 |
|---|---|---|
| active protected features \|A\| (median) | 47 | 47 |
| target already active at h | 1.3% | 3.1% |
| free cost c_free / ‖h‖ | 0.074 | 0.129 |
| linear, P= only, c_lin / ‖h‖ | 0.081 | 0.141 |
| **exact QP (P=, P0), c_qp / ‖h‖** | **0.082** | **0.145** |
| interference inflation κ = c_lin / c_free | 1.07 (p10–p90 1.04–1.18) | 1.09 (1.05–1.14) |
| finite-step inflation c_qp / c_lin | 1.008 | 1.021 |
| gate flips of the linear solution (features wrongly switched on) | 9 (p90 23) | 27 (p90 57) |
| first threshold crossing along the linear step, t ∈ (0, 1] | 0.11 | 0.05 |
| binding (P0) constraints at the QP optimum | 8 | 20 |
| decoder steering to reach the same target: c_dec / ‖h‖ | 0.11 | 0.28 |
| decoder collateral: ‖Δf_P‖ / ‖f_P‖, new features | 0.09, 8 | 0.24, 43 |
| feasible within B; QP converged; max verified T / P error (float32) | 100%; 100%; ≤2e-5 / ≤6e-5 | 100%; 100%; ≤3e-5 / ≤6e-5 |

### Reading

1. **At the SAE's own readout, practically every requested edit is realizable, and cheaply.**
   - Raising a concept feature to its typical activation while preserving all ~47 active features
     and keeping all ~24.5k inactive features off costs 8% (K = 1) to 14% (K = 3) of the residual
     norm.
   - Preserving the active features (P=) costs only 7–9% extra norm, because the geometric
     interference between T and the active set is small. The full (P=, P0) protection costs 8–12%.
2. **The linear (Jacobian/pseudoinverse) solution is not finite-step valid.**
   - It switches on a median of 9 (K = 1) or 27 (K = 3) unprotected features.
   - The first gate flip occurs about 5–11% of the way along the step. The local Taylor model is
     therefore valid only for a small fraction of the edit.
   - The exact QP repairs this for only +1–2% norm. Finite-step correction is **necessary, and
     cheap in norm** (+1–2%). In wall-clock, constraint generation roughly doubles generation time
     (`docs/04_report.md` §8).
3. **Decoder steering is a poor way to realize an SAE edit.** It needs 1.4–1.9× the QP norm and
   changes 9–24% of the protected activations.
4. **Degeneracy check.** Required by the protocol before any H2 test.
   - Feasibility is universal, so a feasible/infeasible certificate carries no information.
   - κ is nearly constant across concepts (between-concept CV 0.04 / 0.02), so it is **degenerate as
     a predictor**.
   - Only the cost magnitudes (c_qp: CV 0.35 / 0.20; c_dec: CV 0.50 / 0.42) vary enough to be
     tested in H2.

## Results: GPT-2 L6, OpenAI v5 TopK (k = 32) with LayerNorm input (same site), 30 concepts × 48 positions

| | K = 1 | K = 3 |
|---|---|---|
| first-order free / linear-protected cost (/‖h‖) | 0.116 / 0.119 | 0.203 / 0.207 |
| κ | 1.02 (CV 0.006) | 1.02 (CV 0.003) |
| true-forward error of the linear solution on T (relative) | 0.8% | 3.5% |
| **structural TopK conflict** (T inactive and all k slots protected) | **99.3%** | **100%** |
| linear solution: protected lost / new active | 4 / 3 | 8 / 5 |
| Gauss–Newton on the true encoder: residual on T and A pre-activations | 0 | 0 |
| … yet protected features lost (evicted) / new active | 3 / 2 | 8 / 5 |
| decoder steering: cost / ‖Δf_P‖/‖f_P‖ / lost / new | 0.15 / 0.12 / 4 / 3 | 0.42 / 0.28 / 12 / 9 (3.4% not reachable within the bracket; 7.1% not within budget B) |

### Reading

- Under TopK, "keep every active feature" is **structurally unrealizable**, whatever the budget.
  - The target starts inactive in 98–99% of positions, and the k slots are full.
  - Gauss–Newton hits every target and protected *pre-activation* exactly, yet 3–8 protected
    features are still evicted.
  - LayerNorm couples all features, so other features also move and enter the top-k.
- A realizable P for TopK must leave at least |T| slots free. The frozen protocol does not tune
  this, and the `*_fs` constraint generation cannot repair evictions, as expected.
- These TopK statements come from a local (first-order) model checked with true forward passes.
  No global guarantee is claimed.

## Results: Gemma-3-270m L12, Gemma Scope 2 JumpReLU 16k (medium L0), 30 concepts × 48 Pile positions

| | K = 1 | K = 3 |
|---|---|---|
| active protected \|A\| | 57 | 57 |
| c_free / c_lin / **c_qp** (/‖h‖) | 0.034 / 0.042 / **0.044** | 0.053 / 0.068 / **0.074** |
| κ | 1.24 (p10–p90 1.13–1.37; CV 0.05) | 1.29 (1.17–1.43; CV 0.04) |
| c_qp / c_lin | 1.03 | 1.08 |
| gate flips of the linear solution; first crossing t | 22; 0.056 | 74; 0.028 |
| decoder: c_dec / ‖Δf_P‖/‖f_P‖ / new | 0.10 / 0.18 / 9 | 0.21 / 0.28 / 27 |
| encoder–decoder cosine of the target | 0.34 | 0.30 |
| QP verified exact / undetermined (float32 rounding at ‖h‖ ≈ 1.3e4) | 97.5% / 2.5% | 95.5% / 4.5% |

### Reading

- Same picture as the ReLU SAE, with more interference: κ ≈ 1.25.
- The finite-step gap is larger: the linear edit flips 22–74 gates, and the first flip occurs 3–6%
  of the way along the step.
- Decoder steering is 2.3–2.8× more expensive than the exact edit, because encoder and decoder
  rows of the same latent are far apart (cosine ≈ 0.3).
- The 2.5% (K = 1) / 4.5% (K = 3) "undetermined" cases each have 1–2 latents that end within float32 rounding of their
  JumpReLU threshold. The absolute margin is 1e-3, against a float32 ulp of ~1e-3 at this activation
  scale. They are **not** counted as feasible (intent-to-treat;
  `scripts/09_realizability_summary.py` reports them as `frac_qp_feasible_undetermined`).

## Cross-setting summary (steps 4–5)

1. **Rank or feasibility is never the binding question.** Every request is linearly feasible, and
   exactly feasible within budget for every ReLU request and for 95.5–97.5% of JumpReLU requests
   (the rest undetermined at float32 tolerance). The exception is TopK, where a combinatorial
   constraint (slot eviction) makes full preservation impossible. By construction of P, that
   conflict equals the share of positions where T starts inactive. None of this is claimed as a
   theorem.
2. **The informative quantities are finite-step properties.**
   - Local (Jacobian or pseudoinverse) solutions break after 3–11% of the step and flip 9–74 gates
     (ReLU/JumpReLU). Under TopK the linear solution switches on 3–5 new features.
   - The exact correction is cheap for piecewise-affine encoders (+1–8% norm).
   - Under TopK the correction cannot restore the evicted slots.
3. **Realizing an SAE edit and steering behavior are different operations.** The same-layer
   encoder needs only 4–14% of ‖h‖ to move a target by its typical activation. Behavioral concept
   steering needs pushes of roughly 0.2–0.5 ‖h‖ (see `docs/04_report.md`). Tier-0 realizability
   therefore constrains behavior at most weakly; that is what H1 and H2 test.
