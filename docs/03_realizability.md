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
| feasible within B; QP converged; verified T/P error | 100%; 100%; 0 / 0 | 100%; 100%; 0 / 0 |

### Reading

1. **At the SAE's own readout, practically every requested edit is realizable, and cheaply.**
   - Raising a concept feature to its typical activation while preserving all ~47 active features
     and keeping all ~24.5k inactive features off costs 8% (K = 1) to 15% (K = 3) of the residual
     norm.
   - Protecting P costs only 7–9% extra norm, because the geometric interference between T and the
     active set is small.
2. **The linear (Jacobian/pseudoinverse) solution is not finite-step valid.**
   - It switches on a median of 9 (K = 1) or 27 (K = 3) unprotected features.
   - The first gate flip occurs about 5–11% of the way along the step. The local Taylor model is
     therefore valid only for a small fraction of the edit.
   - The exact QP repairs this for only +1–2% norm. Finite-step correction is **necessary but
     cheap**.
3. **Decoder steering is a poor way to realize an SAE edit.** It needs 1.4–1.9× the QP norm and
   changes 9–24% of the protected activations.
4. **Degeneracy check.** Required by the protocol before any H2 test.
   - Feasibility is universal, so a feasible/infeasible certificate carries no information.
   - κ is nearly constant across concepts (between-concept CV 0.04 / 0.02), so it is **degenerate as
     a predictor**.
   - Only the cost magnitudes (c_qp: CV 0.35 / 0.20; c_dec: CV 0.50 / 0.42) vary enough to be
     tested in H2.

*(TopK and JumpReLU settings: see below, filled when those runs complete.)*
