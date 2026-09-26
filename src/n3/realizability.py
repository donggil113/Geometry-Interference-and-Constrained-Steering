"""Finite-step realizability of SAE feature edits  (target set T up, protected set P preserved).

Problem at one activation h (all quantities in the SAE's pre-activation coordinates):
    find delta minimizing ||delta||  s.t.
      (T)   pre_T(h+delta) = tau_T                     (tau_T = max(pre_T, thr_T) + Delta_T)
      (P=)  f_j(h+delta)   = f_j(h)       j in A       (active, non-target features keep their value)
      (P0)  f_j(h+delta)   = 0            j in I       (inactive, non-target features stay off)
For ReLU / JumpReLU SAEs without input normalization, pre(h) = W_enc^T(h - b) + b_enc is affine and the
constraint set is a polyhedron on which the encoder is affine for every constrained feature: (T), (P=)
are linear equalities and (P0) are linear inequalities pre_j(h+delta) <= thr_j.  The minimum-norm edit is
therefore an exact convex QP (no Taylor approximation); we solve it by constraint generation over (P0)
(exact at termination: the final point is feasible for all constraints and optimal for a relaxation).

The *linearized* cost that ignores (P0) is  c_lin = ||[J_T; J_A]^+ [tau_T - pre_T; 0]||; with
|T| + |A| <= d and generic rows this system is always solvable: the rank condition is elementary linear
algebra (not a result of this project).  What is informative is the *cost* (norm) and how far the
linear solution is from finite-step feasibility.

For TopK + LayerNorm SAEs (OpenAI v5) the encoder is not piecewise affine in h; we report first-order
quantities plus true-forward checks and a Gauss-Newton correction, and make no global claims.
Structural fact for TopK: if T is inactive and all k active features are protected, (T) and (P=) are
jointly infeasible (T entering the top-k must evict one) -- counted explicitly.
"""
from __future__ import annotations

import numpy as np
import torch

from .saes import SAEWrap


def _pinv_solve(J: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Minimum-norm solution of J x = r (J full row rank generically)."""
    x, *_ = np.linalg.lstsq(J, r, rcond=None)
    return x


def _crossings(pre0, slope, thr, exclude):
    """Number of threshold crossings of pre_j(h + t delta) for t in (0, 1]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (thr - pre0) / slope
    ok = (t > 0) & (t <= 1) & np.isfinite(t)
    ok[exclude] = False
    return int(ok.sum()), (float(t[ok].min()) if ok.any() else 1.0)


def exact_qp(Jeq, req, Jin, bin_, solver="CLARABEL"):
    """min ||x||^2 s.t. Jeq x = req, Jin x <= bin.  Returns x or None."""
    import cvxpy as cp

    d = Jeq.shape[1]
    x = cp.Variable(d)
    cons = [Jeq @ x == req]
    if Jin is not None and Jin.shape[0]:
        cons.append(Jin @ x <= bin_)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), cons)
    try:
        prob.solve(solver=solver)
    except Exception:
        prob.solve(solver="OSQP", eps_abs=1e-9, eps_rel=1e-9, max_iter=200000)
    if x.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return None, prob.status
    return np.asarray(x.value), prob.status


@torch.no_grad()
def analyze_position_affine(sae: SAEWrap, h: torch.Tensor, T: torch.Tensor, dT: torch.Tensor, Delta: torch.Tensor,
                            margin: float = 1e-3, max_rounds: int = 60, budget: float | None = None,
                            add_per_round: int = 48) -> dict:
    """Exact finite-step analysis for norm='none' ReLU / JumpReLU SAEs at one position h (d,)."""
    assert sae.norm == "none" and sae.kind in ("relu", "jumprelu")
    W = sae.W_enc.double().numpy()  # (d, m)
    pre = sae.pre(h.unsqueeze(0))[0].double().numpy()
    f = sae.encode(h.unsqueeze(0))[0].double().numpy()
    thr = np.zeros_like(pre) if sae.kind == "relu" else sae.threshold.double().numpy()
    Tn = T.numpy()
    isT = np.zeros(sae.m, bool)
    isT[Tn] = True
    A = np.where((f > 0) & ~isT)[0]
    I = np.where((f <= 0) & ~isT)[0]
    tau = np.maximum(pre[Tn], thr[Tn]) + Delta.double().numpy()
    rT = tau - pre[Tn]
    JT, JA = W[:, Tn].T, W[:, A].T
    out = dict(n_active=len(A), T_active=int((f[Tn] > 0).sum()), rT=rT.tolist())

    # unconstrained and linear-protected costs
    x_free = _pinv_solve(JT, rT)
    Jeq = np.concatenate([JT, JA], 0)
    req = np.concatenate([rT, np.zeros(len(A))])
    rank = np.linalg.matrix_rank(Jeq)
    out["eq_rank_full"] = bool(rank == Jeq.shape[0])
    x_lin = _pinv_solve(Jeq, req)
    out["c_free"] = float(np.linalg.norm(x_free))
    out["c_lin"] = float(np.linalg.norm(x_lin))
    out["kappa"] = out["c_lin"] / max(out["c_free"], 1e-12)
    out["lin_residual"] = float(np.abs(Jeq @ x_lin - req).max())

    # finite-step check of the linear solution: which inactive features switch on
    pre_lin = pre + W.T @ x_lin
    viol = (pre_lin > thr) & (f <= 0) & ~isT
    out["lin_new_active"] = int(viol.sum())
    out["lin_new_active_mass"] = float(np.maximum(pre_lin[viol], 0).sum())
    ncross, tmin = _crossings(pre, W.T @ x_lin, thr, isT)
    out["lin_crossings"], out["lin_first_crossing_t"] = ncross, tmin
    near = np.abs(pre - thr) < 1e-2 * np.maximum(np.abs(thr), 1.0)
    out["near_kink_protected"] = int(near[A].sum() + near[I].sum())

    # exact QP by constraint generation over the inactive set
    work = np.zeros(0, dtype=int)
    x = x_lin
    status = "lin"
    converged = False
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        pre_x = pre + W.T @ x
        v = np.where((pre_x > thr - margin) & (f <= 0) & ~isT)[0]
        v = np.setdiff1d(v, work)
        if len(v) == 0:
            converged = True
            break
        # add only the most violated constraints (classic cutting-plane / constraint generation)
        v = v[np.argsort(-(pre_x[v] - thr[v]))[:add_per_round]]
        work = np.concatenate([work, v])
        Jin = W[:, work].T
        bin_ = thr[work] - margin - pre[work]
        x_new, status = exact_qp(Jeq, req, Jin, bin_)
        if x_new is None:
            x = None
            break
        x = x_new
    out["qp_status"] = status if (x is None or converged) else "max_rounds_not_converged"
    out["qp_converged"] = bool(converged)
    out["qp_rounds"] = rounds
    if x is None or not converged:
        out["c_qp"] = float("inf") if x is None else float(np.linalg.norm(x))
        out["qp_feasible"] = False if x is None else None  # None = undetermined (not converged)
    else:
        pre_x = pre + W.T @ x
        f_x = sae.act_from_pre(torch.tensor(pre_x, dtype=torch.float32).unsqueeze(0))[0].double().numpy()
        out["c_qp"] = float(np.linalg.norm(x))
        out["qp_feasible"] = True
        out["qp_n_ineq_working"] = int(len(work))
        out["qp_n_binding"] = int((np.abs(pre_x[work] - (thr[work] - margin)) < 1e-4 * np.maximum(1, np.abs(thr[work]))).sum()) if len(work) else 0
        # verify with the true (float32) encoder
        out["qp_check_T_err"] = float(np.abs(f_x[Tn] - tau).max())
        out["qp_check_P_err"] = float(np.abs(f_x[A] - f[A]).max()) if len(A) else 0.0
        out["qp_check_new_active"] = int(((f_x > 0) & (f <= 0) & ~isT).sum())
    out["qp_over_lin"] = out["c_qp"] / max(out["c_lin"], 1e-12)

    # decoder steering: scale along dT until the (weighted) target is reached; collateral at that scale
    dhat = dT.double().numpy() / np.linalg.norm(dT.numpy())
    gain = JT @ dhat  # pre-activation increase per unit norm, per target feature
    out["dec_gain_per_norm"] = gain.tolist()
    if (gain > 0).all():
        c_dec = float(np.max(rT / gain))
        pre_d = pre + W.T @ (c_dec * dhat)
        f_d = sae.act_from_pre(torch.tensor(pre_d, dtype=torch.float32).unsqueeze(0))[0].double().numpy()
        out["c_dec"] = c_dec
        out["dec_P_rel_change"] = float(np.linalg.norm(f_d[A] - f[A]) / max(np.linalg.norm(f[A]), 1e-9))
        out["dec_P_lost"] = int(((f_d[A] <= 0)).sum())
        out["dec_new_active"] = int(((f_d > 0) & (f <= 0) & ~isT).sum())
    else:
        out["c_dec"] = float("inf")
    out["enc_dec_cos"] = float((JT.sum(0) / np.linalg.norm(JT.sum(0))) @ dhat)
    if budget is not None:
        out["qp_within_budget"] = bool(out["c_qp"] <= budget)
        out["lin_within_budget"] = bool(out["c_lin"] <= budget)
        out["dec_within_budget"] = bool(out["c_dec"] <= budget)
    return out


@torch.no_grad()
def analyze_position_topk_ln(sae: SAEWrap, h: torch.Tensor, T: torch.Tensor, dT: torch.Tensor, Delta: torch.Tensor,
                             gn_iters: int = 10, budget: float | None = None) -> dict:
    """First-order + true-forward analysis for TopK SAEs with LayerNorm input normalization."""
    assert sae.kind == "topk"
    hh = h.unsqueeze(0).double()
    W = sae.W_enc.double()
    sd = SAEWrap(**{**sae.__dict__, "W_enc": W, "b_enc": sae.b_enc.double(), "W_dec": sae.W_dec.double(), "b_dec": sae.b_dec.double()})
    pre = sd.pre(hh)[0]
    f = sd.act_from_pre(pre.unsqueeze(0))[0]
    kth = torch.topk(pre, sae.k).values[-1]
    isT = torch.zeros(sae.m, dtype=torch.bool)
    isT[T] = True
    A = torch.where((f > 0) & ~isT)[0]
    tau = torch.maximum(pre[T], kth) + Delta.double()
    rT = tau - pre[T]
    out = dict(n_active=int(len(A)), T_active=int((f[T] > 0).sum()), k=sae.k)
    out["topk_structural_conflict"] = bool((f[T] <= 0).any() and (len(A) + len(T) > sae.k))
    idxT = T.unsqueeze(0)
    JT = sd.pre_jacobian(hh, idxT)[0].numpy()
    JA = sd.pre_jacobian(hh, A.unsqueeze(0))[0].numpy() if len(A) else np.zeros((0, sae.d))
    Jeq = np.concatenate([JT, JA], 0)
    req = np.concatenate([rT.numpy(), np.zeros(len(A))])
    x_free = _pinv_solve(JT, rT.numpy())
    x_lin = _pinv_solve(Jeq, req)
    out["c_free"], out["c_lin"] = float(np.linalg.norm(x_free)), float(np.linalg.norm(x_lin))
    out["kappa"] = out["c_lin"] / max(out["c_free"], 1e-12)

    def true_eval(x):
        p2 = sd.pre(hh + torch.tensor(x).unsqueeze(0))[0]
        f2 = sd.act_from_pre(p2.unsqueeze(0))[0]
        return p2, f2

    p2, f2 = true_eval(x_lin)
    out["lin_T_err_rel"] = float(((p2[T] - tau).abs() / rT.abs().clamp_min(1e-9)).max())
    out["lin_P_rel_change"] = float((f2[A] - f[A]).norm() / f[A].norm().clamp_min(1e-9)) if len(A) else 0.0
    out["lin_P_lost"] = int((f2[A] <= 0).sum())
    out["lin_new_active"] = int(((f2 > 0) & (f <= 0) & ~isT).sum())
    # Gauss-Newton correction on the true encoder (pre-activation targets for T and A)
    x = x_lin.copy()
    for _ in range(gn_iters):
        p2, _ = true_eval(x)
        res = np.concatenate([(p2[T] - tau).numpy(), (p2[A] - pre[A]).numpy()])
        if np.abs(res).max() < 1e-6 * max(1.0, float(rT.abs().max())):
            break
        J2 = np.concatenate([sd.pre_jacobian(hh + torch.tensor(x).unsqueeze(0), idxT)[0].numpy(),
                             sd.pre_jacobian(hh + torch.tensor(x).unsqueeze(0), A.unsqueeze(0))[0].numpy() if len(A) else np.zeros((0, sae.d))], 0)
        x = x - _pinv_solve(J2, res)
    p2, f2 = true_eval(x)
    out["gn_c"] = float(np.linalg.norm(x))
    out["gn_pre_residual_rel"] = float(np.abs(np.concatenate([(p2[T] - tau).numpy(), (p2[A] - pre[A]).numpy()])).max() / max(float(rT.abs().max()), 1e-9))
    out["gn_T_reached"] = bool((f2[T] >= tau - 1e-3 * rT.abs()).all())
    out["gn_P_lost"] = int((f2[A] <= 0).sum())
    out["gn_new_active"] = int(((f2 > 0) & (f <= 0) & ~isT).sum())
    dhat = (dT / dT.norm()).double()
    gain = JT @ dhat.numpy()
    out["enc_dec_cos"] = float((JT.sum(0) / np.linalg.norm(JT.sum(0))) @ dhat.numpy())
    if (gain > 0).all():
        # first-order scale, then true-forward line search to reach target
        c = float(np.max(rT.numpy() / gain))
        for _ in range(30):
            p3, f3 = true_eval(c * dhat.numpy())
            if (p3[T] >= tau - 1e-6).all():
                break
            c *= 1.25
        out["c_dec"] = c if (p3[T] >= tau - 1e-6).all() else float("inf")
        out["dec_P_rel_change"] = float((f3[A] - f[A]).norm() / f[A].norm().clamp_min(1e-9)) if len(A) else 0.0
        out["dec_P_lost"] = int((f3[A] <= 0).sum())
        out["dec_new_active"] = int(((f3 > 0) & (f <= 0) & ~isT).sum())
    else:
        out["c_dec"] = float("inf")
    if budget is not None:
        out["lin_within_budget"] = bool(out["c_lin"] <= budget)
        out["dec_within_budget"] = bool(out["c_dec"] <= budget)
    return out
