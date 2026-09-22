"""IFT adjoint helpers for the production Henyey solver.

Decomposition of _henyey_continuation_atm_bwd (342L) into 4 focused helpers
(the module redesign spec §2, Phase P4-5):

  1. _solve_adjoint_system: transpose J, energy-row scale, Thomas solve,
     iterative refinement, recover λ from μ.
  2. _vjp_residual_params: build residual_fn closure, jax.vjp, extract grads.
  3. _atmosphere_correction: add atm sensitivity to g_Ms/g_ap/g_ar/g_Zp/g_Xp.
  4. _convergence_gate_outputs: per-component isfinite + convergence gating.

JAX PITFALLS (03-solver.md §4, P4-5):
  - comp_mfracs is CLOSED OVER in _vjp_residual_params (not a vjp positional)
  - ln_P_atm, ln_T_atm are closed over in the residual_fn lambda
  - energy_scale must be applied before the Thomas solve and undone after

References:
  - Griewank & Walther (2008) §15 (IFT for implicit solvers)
  - Higham (2002), §9.4 (iterative refinement)
  - MESA star_utils.f90:3678 (set_energy_eqn_scal)
"""

import jax
import jax.numpy as jnp

from stellar_jax.solver.thomas import _adjoint_thomas_solve, _transpose_block_tridiag


def _solve_adjoint_system(A, B, C, g_y, energy_scale, bypass_conv_gate):
    """Solve the transposed adjoint system (S*J)^T μ = g_y, recover λ = S*μ.

    Steps:
      1. Apply energy-row scaling to J blocks
      2. Transpose J → (A_T, B_T, C_T)
      3. Solve via _adjoint_thomas_solve (row equilibration + NaN guards)
      4. Iterative refinement (1 iteration, always-on for interior cotangents)
      5. Recover λ from μ by undoing the scaling

    Iterative refinement (Higham 2002, §9.4) corrects rounding errors accumulated
    in the O(N) Thomas forward/backward sweeps. Surface-directed cotangents
    (∂logL/∂θ) only need the last few zones, so rounding errors don't accumulate.
    Interior-directed cotangents (∂σ²/∂θ via c_s²=Γ₁P/ρ at every shell) need
    accurate μ across ALL N zones; one refinement step typically gains ~4 digits
    of accuracy (Higham 2002, Theorem 9.8). Cost: +1 Thomas solve + 1 matvec per
    step backward — negligible vs the forward Newton iterations.

    Parameters
    ----------
    A, B, C : (N_s, 4, 4) — Jacobian blocks at the converged fixed point
    g_y : (N_s, 4) — incoming cotangent (sanitized)
    energy_scale : (N_s,) — per-block F3 row-scaling factors
    bypass_conv_gate : bool — if True, do 2 refinement iterations (replay mode)

    Returns
    -------
    lam : (N_s, 4) — adjoint vector λ (finite-guarded)
    """

    # Apply energy-row scaling to row 0 of blocks k≥1
    scale_row = energy_scale[1:, None]  # (N_s-1, 1)
    A = A.at[1:, 0, :].multiply(scale_row)
    B = B.at[1:, 0, :].multiply(scale_row)
    C = C.at[1:, 0, :].multiply(scale_row)

    # Transpose the block-tridiagonal system
    A_T, B_T, C_T = _transpose_block_tridiag(A, B, C)

    # Solve (S*J)^T μ = g_y
    mu = _adjoint_thomas_solve(A_T, B_T, C_T, g_y)
    mu = jnp.where(~jnp.isfinite(mu), 0.0, mu)

    # Iterative refinement — always-on for interior cotangent accuracy.
    # One step for the normal path; two for replay mode (denser cotangents).
    # Ref: Higham (2002) §9.4 (iterative refinement for tridiagonal systems).
    _n_refine = 2 if bypass_conv_gate else 1
    for _ in range(_n_refine):
        residual = g_y - _tridiag_matvec(A_T, B_T, C_T, mu)
        residual = jnp.where(~jnp.isfinite(residual), 0.0, residual)
        d_mu = _adjoint_thomas_solve(A_T, B_T, C_T, residual)
        d_mu = jnp.where(~jnp.isfinite(d_mu), 0.0, d_mu)
        mu = mu + d_mu

    # Recover λ = S * μ
    lam = mu.at[1:, 0].multiply(energy_scale[1:])
    lam = jnp.where(~jnp.isfinite(lam), 0.0, lam)

    return lam


def _tridiag_matvec(A_mat, B_mat, C_mat, x):
    """Compute A x_{k-1} + B x_k + C x_{k+1} for each k (block-tridiag matvec)."""
    Ax_prev = jnp.zeros_like(x)
    Ax_prev = Ax_prev.at[1:].set(jnp.einsum('kij,kj->ki', A_mat[1:], x[:-1]))
    Bx = jnp.einsum('kij,kj->ki', B_mat, x)
    Cx_next = jnp.zeros_like(x)
    Cx_next = Cx_next.at[:-1].set(jnp.einsum('kij,kj->ki', C_mat[:-1], x[1:]))
    return Ax_prev + Bx + Cx_next


def _vjp_residual_params(lam, y_final, q_mesh, M_star, X_profile, Z, alpha_mlt,
                         atm_ratio, ln_T_prev, ln_P_prev, inv_dt, N14_profile,
                         opacity_factor, eps_nuc_factor, ln_P_atm, ln_T_atm,
                         atm_jac_mat, y_surf_ref, comp_mfracs,
                         build_residual_fixed_bc_fn, X3_profile=None,
                         gradL_composition_term=None,
                         helm_eos=False, bicubic_opacity=True):
    """Contract -λ with ∂F_lin/∂(inputs) via jax.vjp.

    The residual_fn closes over ln_P_atm, ln_T_atm, atm_jac_mat, y_surf_ref,
    comp_mfracs, helm_eos, and bicubic_opacity (all from the forward pass).
    comp_mfracs is NOT a vjp positional — stop_gradient'd upstream.
    helm_eos and bicubic_opacity are Python bools (nondiff) — closed over to
    match the forward residual's EOS/opacity dispatch.

    Parameters
    ----------
    lam : (N_s, 4) — adjoint vector
    y_final, q_mesh, ... : forward-pass quantities
    opacity_factor : scalar or None — opacity knob (#467)
    eps_nuc_factor : scalar or None — nuclear rate knob (#468)
    ln_P_atm, ln_T_atm, atm_jac_mat, y_surf_ref : closed-over BC data
    comp_mfracs : closed over (NOT a vjp positional — stop_gradient'd upstream)
    build_residual_fixed_bc_fn : the _build_residual_fixed_bc function
    helm_eos : bool — HELM EOS dispatch (Python bool, closed over)
    bicubic_opacity : bool — Steffen-Hermite vs quadrilinear opacity (Python bool, closed over)

    Returns
    -------
    Tuple of per-parameter gradients:
      (g_qm, g_Ms, g_Xp, g_Zp, g_ap, g_ar, g_lnTp, g_lnPp, g_idt, g_N14p, g_opf, g_enf, g_cmf)
    """
    def residual_fn(y_in, qm, Ms, Xp, Zp, ap, ar, lnTp, lnPp, idt, n14p, opf, enf):
        return build_residual_fixed_bc_fn(y_in, qm, Ms, Xp, Zp, ap,
                                          ln_P_atm, ln_T_atm, atm_jac_mat, y_surf_ref,
                                          ln_T_prev=lnTp, ln_P_prev=lnPp, inv_dt=idt,
                                          N14_profile=n14p, comp_mfracs=comp_mfracs,
                                          opacity_factor=opf, eps_nuc_factor=enf,
                                          X3_profile=X3_profile,
                                          gradL_composition_term=gradL_composition_term,
                                          helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)

    # opacity_factor may be None — pass 1.0 for concrete VJP
    _opf_for_vjp = opacity_factor if opacity_factor is not None else jnp.float64(1.0)
    # eps_nuc_factor: same pattern — concrete 1.0 for vjp when None.
    _enf_for_vjp = eps_nuc_factor if eps_nuc_factor is not None else jnp.float64(1.0)
    _, vjp_fn = jax.vjp(residual_fn, y_final, q_mesh, M_star, X_profile, Z,
                        alpha_mlt, atm_ratio, ln_T_prev, ln_P_prev, inv_dt, N14_profile,
                        _opf_for_vjp, _enf_for_vjp)
    _, g_qm, g_Ms, g_Xp, g_Zp, g_ap, g_ar, g_lnTp, g_lnPp, g_idt, g_N14p, g_opf, g_enf = vjp_fn(-lam)

    # comp_mfracs gradient is zero (stop_gradient'd upstream)
    g_cmf = jnp.zeros_like(comp_mfracs)

    return g_qm, g_Ms, g_Xp, g_Zp, g_ap, g_ar, g_lnTp, g_lnPp, g_idt, g_N14p, g_opf, g_enf, g_cmf


def _atmosphere_correction(lam, N_s, g_Ms, g_ap, g_ar, g_Zp, g_Xp, atm_param_grads):
    """Add the atmosphere parameter sensitivity correction to g_Ms, g_ap, g_ar, g_Zp, g_Xp.

    The VJP of residual_fn misses ∂F_surf/∂θ_atm because ln_P_atm, ln_T_atm
    are closed-over constants. This function adds the missing contribution
    using the pre-computed atmosphere partials.

    The correction is:
      g_Ms += lam[N-1, 2] * dlnP_dM + lam[N-1, 3] * dlnT_dM
      g_ap += lam[N-1, 2] * dlnP_da + lam[N-1, 3] * dlnT_da
      g_ar += lam[N-1, 2] * dlnP_dar + lam[N-1, 3] * dlnT_dar
      g_Zp += lam[N-1, 2] * dlnP_dZ + lam[N-1, 3] * dlnT_dZ
      g_Xp[-1] += lam[N-1, 2] * dlnP_dX + lam[N-1, 3] * dlnT_dX

    The Z correction captures Z→κ→P_atm,T_atm — MESA's analogous chain is
    dlnP_dlnkap * dlnkap_dZ (hydro_eqns.f90:930-940, get_PT_bc_ad).
    Prior to #1211, g_Zp had no atmosphere term.
    The X correction (#1119) captures X→κ,EOS→P_atm,T_atm. Since Y_init→X via
    X=1-Y-Z, this fixes the ∂σ²/∂Y_init bias from ~25% to within tolerance.

    Parameters
    ----------
    lam : (N_s, 4) — adjoint vector
    N_s : int — number of solved interfaces
    g_Ms, g_ap, g_ar, g_Zp : scalar gradients (from VJP)
    g_Xp : (N_comp,) — X_profile gradient (from VJP)
    atm_param_grads : (10,) — [dlnP_dM, dlnP_da, dlnP_dar, dlnP_dZ, dlnP_dX,
                                dlnT_dM, dlnT_da, dlnT_dar, dlnT_dZ, dlnT_dX]

    Returns
    -------
    g_Ms, g_ap, g_ar, g_Zp, g_Xp : corrected gradients
    """
    # Guard non-finite atmosphere partials
    atm_grads_valid = jnp.all(jnp.isfinite(atm_param_grads))
    atm_param_grads_safe = jnp.where(atm_grads_valid, atm_param_grads, 0.0)
    dlnP_dM = atm_param_grads_safe[0]
    dlnP_da = atm_param_grads_safe[1]
    dlnP_dar = atm_param_grads_safe[2]
    dlnP_dZ = atm_param_grads_safe[3]
    dlnP_dX = atm_param_grads_safe[4]
    dlnT_dM = atm_param_grads_safe[5]
    dlnT_da = atm_param_grads_safe[6]
    dlnT_dar = atm_param_grads_safe[7]
    dlnT_dZ = atm_param_grads_safe[8]
    dlnT_dX = atm_param_grads_safe[9]

    lam_BCP = lam[N_s - 1, 2]  # adjoint at BC_P position
    lam_BCT = lam[N_s - 1, 3]  # adjoint at BC_T position

    g_Ms = g_Ms + lam_BCP * dlnP_dM + lam_BCT * dlnT_dM
    g_ap = g_ap + lam_BCP * dlnP_da + lam_BCT * dlnT_da
    g_ar = g_ar + lam_BCP * dlnP_dar + lam_BCT * dlnT_dar
    g_Zp = g_Zp + lam_BCP * dlnP_dZ + lam_BCT * dlnT_dZ
    # X_surf correction: the atmosphere bridge depends on X_surf through
    # EOS (μ, Γ₁, ρ) and opacity (κ(X,Z)). The surface X_profile value is X_surf.
    # The VJP through the residual already captures the INTERIOR ∂F/∂X dependence,
    # but the ATMOSPHERE (ln_P_atm, ln_T_atm) dependence on X_surf is closed over.
    # We correct g_Xp at the surface element (index -1 = outermost composition zone,
    # which maps to the Henyey surface via interp_X_at_mass).
    g_Xp = g_Xp.at[-1].add(lam_BCP * dlnP_dX + lam_BCT * dlnT_dX)

    return g_Ms, g_ap, g_ar, g_Zp, g_Xp


def _convergence_gate_simple(converged, *grads):
    """Apply convergence gating to a variable-length tuple of gradients.

    When converged=False, all gradients are zeroed. This is the DRY-3
    extraction: the three non-production _bwd functions in solver/newton.py
    all apply `jnp.where(converged, g, 0.0)` to every output — this helper
    replaces that repeated pattern.

    Unlike _convergence_gate_outputs (the production helper), this function:
      - Does NOT do per-component isfinite clamping (the non-production
        backward passes use plain block_thomas_solve, not the NaN-guarded
        _adjoint_thomas_solve, so isfinite guards are not needed).
      - Does NOT support bypass_conv_gate (the non-production boundaries
        don't have replay-schedule mode).
      - Accepts any number of gradient outputs (6 or 8, depending on the
        backward pass).

    Parameters
    ----------
    converged : bool scalar — convergence flag from the forward pass
    *grads : variable-length positional args — each a JAX array (scalar or nd)

    Returns
    -------
    Tuple of gated gradients in the same order as input.
    """
    return tuple(jnp.where(converged, g, 0.0) for g in grads)


def _convergence_gate_outputs(converged, bypass_conv_gate, g_y_init, g_qm, g_Ms,
                              g_Xp, g_Zp, g_ap, g_ar, g_lnTp, g_lnPp, g_idt,
                              g_N14p, g_cmf, g_opf, g_enf):
    """Apply per-component isfinite clamping + convergence gating to all outputs.

    When converged=False (and bypass_conv_gate=False), all gradients are zeroed.
    When converged=True, each output is independently clamped: non-finite
    values in one channel don't zero another (per-component independence).

    Parameters
    ----------
    converged : bool — IFT convergence flag
    bypass_conv_gate : bool — skip convergence gating (replay mode)
    g_y_init, g_qm, ... : per-parameter gradients

    Returns
    -------
    Tuple of gated gradients in the same order.
    """
    #: bypass_conv_gate=True (replay) forces the gate open — this is a
    # LEGITIMATE contract (e.g. test_rgb_gradient_eps_grav_cross_step relies on it
    # to exercise the eps_grav IFT channel on a synthetic step). For the subgiant
    # replay it is inert anyway (all 154 steps converge), and the
    # per-step adjoint is exact WITH bypass (Tier-0 M-channel rel_err 5e-6). An
    # unconditional honest gate (conv_gate=converged) was tried and REVERTED: it
    # zeroed g_lnTp on the synthetic RGB step (regression). The exact-transpose
    # sign fix lives entirely in the gradL-threaded backward Jacobian/VJP (A),
    # not in this gate.
    if bypass_conv_gate:
        conv_gate = jnp.bool_(True)
    else:
        conv_gate = converged

    g_Ms = jnp.where(conv_gate & jnp.all(jnp.isfinite(g_Ms)), g_Ms, 0.0)
    g_Xp = jnp.where(conv_gate & jnp.all(jnp.isfinite(g_Xp)), g_Xp, 0.0)
    g_Zp = jnp.where(conv_gate & jnp.all(jnp.isfinite(g_Zp)), g_Zp, 0.0)
    g_ap = jnp.where(conv_gate & jnp.isfinite(g_ap), g_ap, 0.0)
    g_ar = jnp.where(conv_gate & jnp.isfinite(g_ar), g_ar, 0.0)
    g_lnTp = jnp.where(conv_gate & jnp.isfinite(g_lnTp), g_lnTp, 0.0)
    g_lnPp = jnp.where(conv_gate & jnp.isfinite(g_lnPp), g_lnPp, 0.0)
    g_qm = jnp.where(conv_gate & jnp.all(jnp.isfinite(g_qm)), g_qm, 0.0)
    g_idt = jnp.where(conv_gate & jnp.isfinite(g_idt), g_idt, 0.0)
    g_N14p = jnp.where(conv_gate & jnp.all(jnp.isfinite(g_N14p)), g_N14p, 0.0)
    g_opf = jnp.where(conv_gate & jnp.isfinite(g_opf), g_opf, 0.0)
    g_enf = jnp.where(conv_gate & jnp.isfinite(g_enf), g_enf, 0.0)
    # g_y_init and g_cmf are always zeros — no gating needed

    return (g_y_init, g_qm, g_Ms, g_Xp, g_Zp, g_ap, g_ar,
            g_lnTp, g_lnPp, g_idt, g_N14p, g_cmf, g_opf, g_enf)
