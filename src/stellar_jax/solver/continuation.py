"""Production @custom_vjp boundary: _henyey_continuation_atm + entry points.

This is the STRUCTURE DIFFERENTIABILITY BOUNDARY for per-step evolution.
Contains the highest-risk code in the entire codebase:
  - @custom_vjp with nondiff_argnums=(7, 8, 9, 21, 22) — positional fragility
  - The IFT backward pass (decomposed into 4 helpers in solver/adjoint.py)
  - The production entry points called by evolution.py

The defvjp registration MUST stay in the SAME FILE as the function definition
(JAX pitfall §5.7). The signature MUST NOT be reordered (§5.1).

Phase P4-4 of the solver/ refactoring (per the module redesign spec).

References:
  - Griewank & Walther (2008) §15 (IFT for implicit solvers)
  - Blondel et al. (2022), arxiv:2105.15183
  - MESA hydro_eqns.f90:817 (get_PT_bc_ad — atmosphere BC partials)
  - MESA star_solver.f90:739-953 (Armijo line-search)
"""
import functools

import jax
import jax.numpy as jnp
from jax import lax

from stellar_jax.config.mesh_defaults import COMP_MFRACS
from stellar_jax.solver.residual import _build_residual_fixed_bc, _build_residual_atm
from stellar_jax.solver.jacobian import _jacobian_blocks_fixed_bc
from stellar_jax.solver.surface_bc import _surface_bc_atm_values, _surface_bc_atm_values_with_ratio
from stellar_jax.solver.thomas import block_thomas_solve
from stellar_jax.solver.damping import _apply_per_zone_damping
from stellar_jax.solver.eps_grav import _energy_row_scale_factors
from stellar_jax.solver.conditioning import conditioned_solve, armijo_line_search, _TOL_MAX_CORRECTION
from stellar_jax.solver.adjoint import (
    _solve_adjoint_system, _vjp_residual_params,
    _atmosphere_correction, _convergence_gate_outputs,
)
from stellar_jax.solver.contracts import assert_continuation_atm_signature

# Convergence tolerance for the IFT validity check.
_HENYEY_CONV_TOL = 1e-4

# Adjoint IFT-trust gate threshold — DECOUPLED from the Newton solve `tol`.
# The adjoint backward pass zeros a step's gradient only when the (scaled) residual
# at the converged solution exceeds this bound (IFT error ∝ ‖F(y*)‖). This is the
# pre- EFFECTIVE gate (3 × the old solve tol 0.01 = 0.03), kept FIXED so that
# tightening the Newton solve tol to 1e-4 (for solve accuracy, incl. the subgiant
# replay) does NOT over-tighten the adjoint gate and spuriously zero the gradient of
# stiff / hard-to-converge steps that are already converged-enough for the IFT
# linearization. 0.03 is main's long-validated adjoint-trust level (worst-case IFT
# error ~3%, well within the <10% gradient-integrity gates). Newton convergence
# (continuation.py `R_norm < tol`) and the forward `newton_converged` flag are
# unaffected — this constant governs the ADJOINT gate only.
_ADJOINT_GATE_TOL = 0.03


# ═══════════════════════════════════════════════════════════════
# Production continuation solver — @custom_vjp boundary
# ═══════════════════════════════════════════════════════════════

@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 21, 22))
def _henyey_continuation_atm(y_prev, q_mesh, M_star, X_profile, Z, alpha_mlt,
                             atm_ratio, n_iter, tol, bypass_conv_gate, ln_T_prev, ln_P_prev, inv_dt,
                             N14_profile=None, comp_mfracs=None, opacity_factor=None, eps_nuc_factor=None,
                             X_prev_profile=None, gradL_composition_term=None, X3_profile=None,
                             ln_rho_guess=None, helm_eos=False, bicubic_opacity=True):
    """Newton continuation with linearized atmosphere BC + IFT @custom_vjp (#289).

    Architecture (MESA-style, issue #306):
      1. Compute the atmosphere bridge ONCE before Newton (3000 RK2 steps)
         to get (ln_P_atm, ln_T_atm) at the mesh surface + Jacobian (atm_jac).
      2. Newton iterations use linearized BCs: the surface targets respond to
         changes in y_surf via atm_jac, without re-calling the bridge.
      3. The IFT backward pass evaluates the TRUE residual at y_final.

    R_phot is solved as part of the Newton system: the atmosphere Jacobian
    (atm_jac) is computed via jacfwd through _surface_bc_atm_values_with_ratio,
    which derives R_phot = atm_ratio * exp(y_surf[0]) internally. This means
    the Jacobian includes ∂(BC)/∂(ln_r_surf) through the full path
    y_surf[0] → R_phot → bridge → (P,T). This mirrors MESA's explicit
    dlnPsurf_dlnR and dlnTsurf_dlnR terms in the surface Jacobian block
    (hydro_eqns.f90:928, 966; get_PT_bc_ad at line 817).

    Because R_phot is a function of the solved state y*, the IFT backward
    naturally bounds ∂R_phot/∂θ — no circular carry feedback, no
    stop_gradient, no EMA.

    When ln_T_prev, ln_P_prev, and inv_dt are provided, eps_grav (Form C) is
    included in the energy equation F3.
    """
    N_s = y_prev.shape[0]
    X_surf = X_profile[-1]

    # === ONCE before Newton: bridge + Jacobian ===
    y_surf_init = y_prev[N_s - 1]
    R_phot = atm_ratio * jnp.exp(y_surf_init[0])
    ln_P_atm, ln_T_atm = _surface_bc_atm_values(y_surf_init, M_star, X_surf, Z, alpha_mlt, R_phot,
                                                 helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
    atm_jac = jax.jacfwd(_surface_bc_atm_values_with_ratio, argnums=0)(
        y_surf_init, M_star, X_surf, Z, alpha_mlt, atm_ratio,
        helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
    atm_jac_mat = jnp.stack([atm_jac[0], atm_jac[1]])  # (2, 4)

    # === Energy-row scaling — MESA set_energy_eqn_scal (star_utils.f90:3678) ===
    # Computed ONCE per step from y_prev (= MESA energy_start = start-of-step state).
    # Applied to the F3 (energy equation) row of the residual AND Jacobian EVERY
    # Newton iteration. Convergence is judged on the SCALED residuals — matching
    # MESA's conditioning of the energy equation. Always-on (no bypass gate).
    energy_scale = _energy_row_scale_factors(y_prev, q_mesh, X_profile, Z, inv_dt,
                                            comp_mfracs=comp_mfracs,
                                            helm_eos=helm_eos)

    # === Newton with linearized BCs + conditioning (SOLV chain -345) ===
    # Convergence-gated while_loop: exits on convergence instead of
    # running all n_iter iterations. Bit-identical to the prior scan (the scan
    # froze dy=0 after convergence; while_loop just exits early).
    # Matches MESA star_solver.f90:325 `iter_loop: do while (.not. passed_tol_tests)`.
    def _scale_residual(R_raw):
        """Apply energy-row scaling to F3 row (row index 0 of each block k≥1)."""
        return R_raw.at[1:, 0].multiply(energy_scale[1:])

    def residual_fn_for_armijo(y_trial):
        """Residual function for Armijo line-search (fixed BCs, closed over).

        Returns the SCALED residual — Armijo merit f=½‖F_scaled‖² must be
        consistent with the convergence criterion on R_norm of the scaled system.
        """
        R_raw = _build_residual_fixed_bc(y_trial, q_mesh, M_star, X_profile, Z, alpha_mlt,
                                        ln_P_atm, ln_T_atm, atm_jac_mat, y_surf_init,
                                        ln_T_prev=ln_T_prev, ln_P_prev=ln_P_prev,
                                        inv_dt=inv_dt, N14_profile=N14_profile,
                                        comp_mfracs=comp_mfracs,
                                        opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                                        X_prev_profile=X_prev_profile,
                                        gradL_composition_term=gradL_composition_term, X3_profile=X3_profile,
                                        ln_rho_guess=ln_rho_guess,
                                        helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
        return _scale_residual(R_raw)

    def newton_cond_fn(state):
        """Continue while not converged and iteration budget remains."""
        _, converged, i = state
        return (~converged) & (i < n_iter)

    def newton_body_fn(state):
        """One Newton iteration: residual → Jacobian → solve → damp → Armijo → update."""
        y, _, i = state
        R_raw = _build_residual_fixed_bc(y, q_mesh, M_star, X_profile, Z, alpha_mlt,
                                     ln_P_atm, ln_T_atm, atm_jac_mat, y_surf_init,
                                     ln_T_prev=ln_T_prev, ln_P_prev=ln_P_prev,
                                     inv_dt=inv_dt, N14_profile=N14_profile,
                                     comp_mfracs=comp_mfracs,
                                     opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                                     X_prev_profile=X_prev_profile,
                                     gradL_composition_term=gradL_composition_term, X3_profile=X3_profile,
                                     ln_rho_guess=ln_rho_guess,
                                     helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
        # Apply energy-row scaling to residual (MESA: resid_ad = scal * resid_ad)
        R = _scale_residual(R_raw)
        R_norm = jnp.max(jnp.abs(R))

        # Apply energy-row scaling to Jacobian rows (MESA: partials scaled by scal)
        A_raw, B_raw, C_raw = _jacobian_blocks_fixed_bc(y, q_mesh, M_star, X_profile, Z, alpha_mlt,
                                            ln_P_atm, ln_T_atm, atm_jac_mat, y_surf_init,
                                            ln_T_prev=ln_T_prev, ln_P_prev=ln_P_prev,
                                            inv_dt=inv_dt, comp_mfracs=comp_mfracs,
                                            opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                                            X3_profile=X3_profile, ln_rho_guess=ln_rho_guess,
                                            helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
        scale_col = energy_scale[1:, None]  # (N_c, 1) for broadcasting over 4 columns
        A = A_raw.at[1:, 0, :].multiply(scale_col)
        B = B_raw.at[1:, 0, :].multiply(scale_col)
        C = C_raw.at[1:, 0, :].multiply(scale_col)

        dy = conditioned_solve(A, B, C, -R, R_norm, levenberg=True)
        dy = _apply_per_zone_damping(dy, R_norm)
        alpha = armijo_line_search(y, dy, R, residual_fn_for_armijo, R_norm)
        dy_final = alpha * dy
        xscale = jnp.maximum(1.0, jnp.abs(y))
        max_corr = jnp.max(jnp.abs(dy_final) / xscale)
        converged_new = (R_norm < tol) & (max_corr < _TOL_MAX_CORRECTION)
        y_new = y + dy_final
        return (y_new, converged_new, i + 1)

    init_state = (y_prev, jnp.bool_(False), jnp.int32(0))

    y_final, newton_converged, _ = lax.while_loop(
        newton_cond_fn, newton_body_fn, init_state)
    return y_final, newton_converged


def _henyey_continuation_atm_fwd(y_prev, q_mesh, M_star, X_profile, Z, alpha_mlt,
                                 atm_ratio, n_iter, tol, bypass_conv_gate, ln_T_prev, ln_P_prev, inv_dt,
                                 N14_profile=None, comp_mfracs=None, opacity_factor=None, eps_nuc_factor=None,
                                 X_prev_profile=None, gradL_composition_term=None, X3_profile=None,
                                 ln_rho_guess=None, helm_eos=False, bicubic_opacity=True):
    y_final, newton_converged = _henyey_continuation_atm(
        y_prev, q_mesh, M_star, X_profile, Z,
        alpha_mlt, atm_ratio, n_iter, tol, bypass_conv_gate,
        ln_T_prev, ln_P_prev, inv_dt, N14_profile, comp_mfracs, opacity_factor, eps_nuc_factor,
        X_prev_profile=X_prev_profile, gradL_composition_term=gradL_composition_term, X3_profile=X3_profile,
        ln_rho_guess=ln_rho_guess,
        helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)

    # Save atmosphere-BC data for the backward pass.
    # FIX (/): Evaluate the bridge at the CONVERGED surface state
    # y_final[N-1] instead of the frozen y_prev[N-1].
    #
    # Rationale: the IFT backward differentiates the TRUE residual F(y*, θ)
    # where the atmosphere BCs are evaluated at y_surf = y_final, not at a
    # frozen y_prev.  The Newton linearizes BCs at y_prev, but at convergence
    # F_lin(y*) ≈ F_true(y*) ≈ 0.  The Jacobian and parameter sensitivities
    # of F_true at y* are what the IFT needs.
    #
    # Effect: the ∂ν²/∂M chain through R★ = atm_ratio·exp(y_surf[0]) gets
    # correct parameter sensitivities (∂bridge/∂M at the solved state), and
    # the adjoint system uses the true Jacobian at y*.
    #
    # Measured (research): before fix, atm_ratio tangent JVP −0.87 vs
    # FD +0.10 (WRONG SIGN); after fix, both negative (CORRECT SIGN).
    N_s = y_prev.shape[0]
    X_surf = X_profile[-1]
    y_surf_ref = y_final[N_s - 1]  # ← FIXED: was y_prev[N_s - 1]
    R_phot_ref = atm_ratio * jnp.exp(y_surf_ref[0])
    ln_P_atm, ln_T_atm = _surface_bc_atm_values(y_surf_ref, M_star, X_surf, Z, alpha_mlt, R_phot_ref,
                                                 helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
    atm_jac = jax.jacfwd(_surface_bc_atm_values_with_ratio, argnums=0)(
        y_surf_ref, M_star, X_surf, Z, alpha_mlt, atm_ratio,
        helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
    atm_jac_mat = jnp.stack([atm_jac[0], atm_jac[1]])  # (2, 4)

    # Atmosphere parameter partials (MESA: dlnPsurf_dlnM, dlnTsurf_dlnM, etc.)
    # Z is included because the atmosphere BC depends on Z through opacity
    # (κ(T,ρ,X,Z)→P_atm,T_atm). MESA propagates this via dlnP_dlnkap * dlnkap_dZ
    # (hydro_eqns.f90:930-940); our jacfwd captures it naturally through the
    # opacity evaluations inside _atm_bridge_endpoint. Prior to, Z was
    # closed over → the ∂(ln_P_atm,ln_T_atm)/∂Z term was missing from g_Zp.
    # X_surf is included because the atmosphere bridge depends on
    # X_surf through EOS (μ, Γ₁) and opacity (κ(X,Z)). Since Y_init→X via
    # X=1-Y-Z, the missing ∂(atm)/∂X term biased ∂σ²/∂Y_init by ~25%.
    # MESA's analogous chain: dlnP_dlnkap * dlnkap_dX + EOS(X) partials.
    #
    # Evaluated at y_final (converged state) — same as the bridge and Jacobian.
    def _atm_of_params(params_vec):
        """Bridge as a function of [M_star, alpha_mlt, atm_ratio, Z, X_surf]."""
        Ms_p, ap_p, ar_p, Z_p, Xs_p = (params_vec[0], params_vec[1],
                                         params_vec[2], params_vec[3], params_vec[4])
        R_p = ar_p * jnp.exp(y_surf_ref[0])
        lnP, lnT = _surface_bc_atm_values(y_surf_ref, Ms_p, Xs_p, Z_p, ap_p, R_p,
                                           helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
        return jnp.array([lnP, lnT])
    params_ref = jnp.array([M_star, alpha_mlt, atm_ratio, Z, X_surf])
    atm_param_jac = jax.jacfwd(_atm_of_params)(params_ref)
    # (10,): [dlnP_dM, dlnP_da, dlnP_dar, dlnP_dZ, dlnP_dX,
    #         dlnT_dM, dlnT_da, dlnT_dar, dlnT_dZ, dlnT_dX]
    atm_param_grads = jnp.array([
        atm_param_jac[0, 0], atm_param_jac[0, 1], atm_param_jac[0, 2], atm_param_jac[0, 3],
        atm_param_jac[0, 4],
        atm_param_jac[1, 0], atm_param_jac[1, 1], atm_param_jac[1, 2], atm_param_jac[1, 3],
        atm_param_jac[1, 4],
    ])

    # IFT convergence gate: check whether the solve converged well enough for
    # the IFT linearization (dy/dθ = -J⁻¹ · ∂F/∂θ) to be valid.
    #
    # Uses the SCALED residual norm — the same system the Newton converges on.
    # The energy-row scaling (MESA set_energy_eqn_scal) is a diagonal
    # preconditioner S that does not change the fixed point: S·F(y*)=0 iff
    # F(y*)=0, and the IFT adjoint dy/dθ = -(S·J)⁻¹·∂(S·F)/∂θ = -J⁻¹·∂F/∂θ
    # (S cancels algebraically). So convergence quality for the IFT is
    # correctly measured in the scaled norm — the same criterion the Newton
    # uses to declare convergence. Using the raw (unscaled) norm would make
    # the gate inconsistent with the Newton: the Newton declares "converged"
    # but the gate rejects because the raw energy-equation residual (O(cp·T/dt)
    # times the scaled residual) exceeds the threshold. This incorrectly zeros
    # gradient contributions from well-converged steps.
    R_final_raw = _build_residual_fixed_bc(
        jax.lax.stop_gradient(y_final), q_mesh,  # GP-9: CONVERGENCE_FLAGS
        jax.lax.stop_gradient(M_star), X_profile, Z, alpha_mlt,  # GP-9: CONVERGENCE_FLAGS
        ln_P_atm, ln_T_atm, atm_jac_mat, y_surf_ref,
        ln_T_prev=ln_T_prev, ln_P_prev=ln_P_prev, inv_dt=inv_dt,
        N14_profile=N14_profile, comp_mfracs=comp_mfracs,
        opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
        X_prev_profile=X_prev_profile, X3_profile=X3_profile,
        helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
    # Apply the same energy-row scaling used in the Newton convergence criterion
    energy_scale_fwd = _energy_row_scale_factors(y_prev, q_mesh, X_profile, Z, inv_dt,
                                                 comp_mfracs=comp_mfracs,
                                                 helm_eos=helm_eos)
    R_final_scaled = R_final_raw.at[1:, 0].multiply(energy_scale_fwd[1:])
    final_R_norm = jnp.max(jnp.abs(R_final_scaled))
    # IFT convergence gate: gradients are zeroed for steps where the Newton
    # residual is too large for the IFT linearization to be trustworthy.
    #: the adjoint IFT-trust gate is DECOUPLED from the Newton solve `tol`.
    # It was `final_R_norm < 3.0 * tol`, which chained the gate to the solve tol;
    # when tightened tol 0.01→1e-4 for solve accuracy, that silently tightened
    # the gate 0.03→3e-4, zeroing the gradient of converged-enough stiff steps
    # (regressed test_gradient_energy_row_scaling_stiff_regime: AD 24% low). The gate
    # now uses the fixed _ADJOINT_GATE_TOL (= main's proven 0.03); IFT error ∝ ‖F(y*)‖
    # (Griewank & Walther 2008, §15), so 0.03 → ≤~3% adjoint error, well within the gates.
    converged = final_R_norm < _ADJOINT_GATE_TOL

    # AD-safe sanitization of y_final for the backward pass.
    y_final_safe = jnp.where(jnp.isfinite(y_final), y_final, y_prev)


    return (y_final, newton_converged), (y_final_safe, q_mesh, M_star, X_profile, Z, alpha_mlt,
                     atm_ratio, converged, ln_T_prev, ln_P_prev, inv_dt,
                     ln_P_atm, ln_T_atm, atm_jac_mat, y_surf_ref, atm_param_grads,
                     N14_profile, comp_mfracs, opacity_factor, eps_nuc_factor,
                     X_prev_profile, X3_profile, gradL_composition_term)


def _henyey_continuation_atm_bwd(n_iter, tol, bypass_conv_gate, helm_eos, bicubic_opacity, res, g_out):
    """IFT backward pass — decomposed into 4 helpers (issue #503).

    Architecture (MESA hydro_eqns.f90:817, get_PT_bc_ad):
      1. Solve the transposed adjoint system (S*J)^T μ = g → recover λ
      2. Contract -λ with ∂F_lin/∂θ via VJP
      3. Add atmosphere parameter sensitivity correction
      4. Apply convergence gating + per-component isfinite clamping

    Decomposition per the module redesign spec §2, Phase P4-5.

    nondiff args (n_iter, tol, bypass_conv_gate, helm_eos, bicubic_opacity) are
    Python values passed as leading positional args by JAX's custom_vjp dispatch.
    """
    g_y, _ = g_out
    y_final, q_mesh, M_star, X_profile, Z, alpha_mlt, atm_ratio, converged, \
        ln_T_prev, ln_P_prev, inv_dt, \
        ln_P_atm, ln_T_atm, atm_jac_mat, y_surf_ref, atm_param_grads, \
        N14_profile, comp_mfracs, opacity_factor, eps_nuc_factor, \
        X_prev_profile, X3_profile, gradL_composition_term = res
    N_s = y_final.shape[0]

    # Input sanitization: replace NaN/Inf cotangent with zeros.
    g_y = jnp.where(~jnp.isfinite(g_y), 0.0, g_y)

    # Recompute the raw Jacobian at y_final for the IFT adjoint solve.
    # J1 backward reuse (carrying blocks from the forward) is deferred to a
    # follow-up (/): the forward body applies energy-row scaling to the
    # blocks before the Thomas solve, but _solve_adjoint_system expects raw
    # blocks and applies scaling itself. Adapting that interface is a clean
    # follow-up; the ~7-20× forward speedup is the primary win of.
    A, B, C = _jacobian_blocks_fixed_bc(y_final, q_mesh, M_star, X_profile, Z, alpha_mlt,
                                        ln_P_atm, ln_T_atm, atm_jac_mat, y_surf_ref,
                                        ln_T_prev=ln_T_prev, ln_P_prev=ln_P_prev,
                                        inv_dt=inv_dt, comp_mfracs=comp_mfracs,
                                        opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                                        X3_profile=X3_profile,
                                        gradL_composition_term=gradL_composition_term,
                                        helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)

    energy_scale = _energy_row_scale_factors(y_final, q_mesh, X_profile, Z, inv_dt,
                                            comp_mfracs=comp_mfracs,
                                            helm_eos=helm_eos)

    lam = _solve_adjoint_system(A, B, C, g_y, energy_scale, bypass_conv_gate)

    # Step 2: VJP contraction.
    (g_qm, g_Ms, g_Xp, g_Zp, g_ap, g_ar, g_lnTp, g_lnPp, g_idt,
     g_N14p, g_opf, g_enf, g_cmf) = _vjp_residual_params(
        lam, y_final, q_mesh, M_star, X_profile, Z, alpha_mlt,
        atm_ratio, ln_T_prev, ln_P_prev, inv_dt, N14_profile,
        opacity_factor, eps_nuc_factor,
        ln_P_atm, ln_T_atm, atm_jac_mat, y_surf_ref, comp_mfracs,
        _build_residual_fixed_bc, X3_profile=X3_profile,
        gradL_composition_term=gradL_composition_term,
        helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)

    # Step 3: Atmosphere sensitivity correction.
    g_Ms, g_ap, g_ar, g_Zp, g_Xp = _atmosphere_correction(
        lam, N_s, g_Ms, g_ap, g_ar, g_Zp, g_Xp, atm_param_grads)

    # Step 4: Convergence gating.
    # g_y_prev = 0 is correct: the IFT says the converged solution depends
    # on parameters, not on the initial guess. The atmosphere bridge is now
    # evaluated at y_final (not y_prev) in the _fwd residuals, so the
    # backward uses the TRUE residual Jacobian and parameter sensitivities.
    g_y_init = jnp.zeros_like(y_final)

    return _convergence_gate_outputs(
        converged, bypass_conv_gate, g_y_init, g_qm, g_Ms, g_Xp, g_Zp,
        g_ap, g_ar, g_lnTp, g_lnPp, g_idt, g_N14p, g_cmf, g_opf, g_enf) + (None, None, None, None)


_henyey_continuation_atm.defvjp(_henyey_continuation_atm_fwd,
                                _henyey_continuation_atm_bwd)

# Compile-time assertion: verify nondiff_argnums=(7,8,9,21,22) matches the signature.
assert_continuation_atm_signature(_henyey_continuation_atm)


# ═══════════════════════════════════════════════════════════════
# Production entry points
# ═══════════════════════════════════════════════════════════════

def henyey_solve_from_state_atm(y_prev, q_mesh, M_star, X_profile, Z, alpha_mlt,
                                R_phot, n_iter=30, tol=1e-4,
                                ln_T_prev=None, ln_P_prev=None, inv_dt=None,
                                atm_ratio=None, N14_profile=None, comp_mfracs=None,
                                bypass_conv_gate=False, opacity_factor=None, eps_nuc_factor=None,
                                X_prev_profile=None, gradL_composition_term=None, X3_profile=None,
                                ln_rho_guess=None, helm_eos=False, bicubic_opacity=True):
    """Continuation solver with atmosphere BC for production evolution (#289).

    See the module redesign spec §3 for the full contract.

    gradL_composition_term: Optional (N_c,) array of per-cell Ledoux composition
        gradient terms. When None (default, lax.scan MS path), the Ledoux term
        is inactive (XLA trace identity preserved). When an array (adaptive_forward
        RGB path, precomputed externally), it stabilizes the H-shell convective
        boundary. MESA turb_support.f90:273.
    """
    _ln_T_prev = ln_T_prev if ln_T_prev is not None else jnp.zeros(y_prev.shape[0])
    _ln_P_prev = ln_P_prev if ln_P_prev is not None else jnp.zeros(y_prev.shape[0])
    _inv_dt = inv_dt if inv_dt is not None else jnp.float64(0.0)
    _N14_profile = N14_profile if N14_profile is not None else jnp.full(X_profile.shape[0], 0.251 * Z)
    _comp_mfracs = comp_mfracs if comp_mfracs is not None else jnp.array(COMP_MFRACS)
    _atm_ratio = atm_ratio if atm_ratio is not None else R_phot / jnp.exp(y_prev[-1, 0])

    y_final, newton_converged = _henyey_continuation_atm(
        y_prev, q_mesh, M_star, X_profile, Z,
        alpha_mlt, _atm_ratio, n_iter, tol, bypass_conv_gate,
        _ln_T_prev, _ln_P_prev, _inv_dt, _N14_profile, _comp_mfracs, opacity_factor, eps_nuc_factor,
        X_prev_profile=X_prev_profile, gradL_composition_term=gradL_composition_term, X3_profile=X3_profile,
        ln_rho_guess=ln_rho_guess,
        helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)

    R_phot_final = _atm_ratio * jnp.exp(jax.lax.stop_gradient(y_final[-1, 0]))  # GP-9: CONVERGENCE_FLAGS
    R_final_raw = _build_residual_atm(
        jax.lax.stop_gradient(y_final), q_mesh,  # GP-9: CONVERGENCE_FLAGS
        jax.lax.stop_gradient(M_star), X_profile, Z, alpha_mlt,  # GP-9: CONVERGENCE_FLAGS
        R_phot=R_phot_final, ln_T_prev=_ln_T_prev,
        ln_P_prev=_ln_P_prev, inv_dt=_inv_dt,
        comp_mfracs=_comp_mfracs, opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
        N14_profile=_N14_profile, X3_profile=X3_profile,
        helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
    # Apply energy-row scaling for consistency with the Newton convergence criterion
    _energy_scale = _energy_row_scale_factors(y_prev, q_mesh, X_profile, Z, _inv_dt,
                                             comp_mfracs=_comp_mfracs,
                                             helm_eos=helm_eos)
    R_final = R_final_raw.at[1:, 0].multiply(_energy_scale[1:])
    res_norm = jax.lax.stop_gradient(jnp.max(jnp.abs(R_final)))  # GP-9: CONVERGENCE_FLAGS
    return {
        'y': y_final,
        'converged': newton_converged,
        'residual_norm': res_norm,
    }


def _henyey_init_atm(y_prev, q_mesh, M_star, X_profile, Z, alpha_mlt,
                     R_phot, n_iter, tol):
    """Cold-start Newton with FULL bridge inside Newton (for ZAMS init only).

    NOT used in the per-step evolution loop. Use _henyey_continuation_atm.
    Convergence-gated while_loop (#805) — same pattern as the production solver.
    """
    from stellar_jax.solver.residual import _build_residual_atm
    from stellar_jax.solver.jacobian import _jacobian_blocks_atm

    def newton_cond_fn(state):
        _, converged, i = state
        return (~converged) & (i < n_iter)

    def newton_body_fn(state):
        y, _, i = state
        R = _build_residual_atm(y, q_mesh, M_star, X_profile, Z, alpha_mlt,
                                R_phot=R_phot)
        R_norm = jnp.max(jnp.abs(R))
        A, B, C = _jacobian_blocks_atm(y, q_mesh, M_star, X_profile, Z, alpha_mlt,
                                       R_phot=R_phot)
        dy = block_thomas_solve(A, B, C, -R)
        t_damp = jnp.clip((1.0 - R_norm) / 0.9, 0.0, 1.0)
        max_corr = 0.5 + 1.5 * t_damp
        max_struct = jnp.max(jnp.abs(dy[:, :3]))
        alpha_damp = jnp.minimum(1.0, max_corr / (max_struct + 1e-30))
        dy = dy * alpha_damp

        def _backtrack_step(alpha, _):
            y_trial = y + alpha * dy
            R_trial = _build_residual_atm(y_trial, q_mesh, M_star, X_profile, Z,
                                          alpha_mlt, R_phot=R_phot)
            R_trial_norm = jnp.max(jnp.abs(R_trial))
            should_reduce = (R_trial_norm > R_norm) & (alpha > 1.0 / 16.0)
            return jnp.where(should_reduce, alpha * 0.5, alpha), None

        alpha_final, _ = lax.scan(_backtrack_step, jnp.float64(1.0), None, length=4)
        dy_final = lax.stop_gradient(alpha_final) * dy  # GP-9: CONVERGENCE_FLAGS
        y_new = y + dy_final
        converged_new = R_norm < tol
        return (y_new, converged_new, i + 1)

    init_state = (y_prev, jnp.bool_(False), jnp.int32(0))
    (y_final, _, _) = lax.while_loop(newton_cond_fn, newton_body_fn, init_state)
    return y_final


def henyey_init_from_state_atm(y_prev, q_mesh, M_star, X_profile, Z, alpha_mlt,
                               R_phot, n_iter=60, tol=1e-4, X3_profile=None):
    """Cold-start solver with full bridge (ZAMS pre-convergence only).

    Uses _henyey_init_atm then passes through _henyey_continuation_atm for
    platform-independent IFT backward.
    """
    y_init_converged = _henyey_init_atm(y_prev, q_mesh, M_star, X_profile, Z,
                                        alpha_mlt, R_phot, n_iter, tol)
    y_detached = jax.lax.stop_gradient(y_init_converged)  # GP-9: CONVERGENCE_FLAGS

    _atm_ratio = R_phot / jnp.exp(y_detached[-1, 0])
    _zeros = jnp.zeros(y_detached.shape[0])
    _n14_init = jnp.full(X_profile.shape[0], 0.251 * Z)
    _comp_mfracs_init = jnp.array(COMP_MFRACS)
    y_final, newton_converged = _henyey_continuation_atm(
        y_detached, q_mesh, M_star, X_profile, Z,
        alpha_mlt, _atm_ratio, 5, tol, False,
        _zeros, _zeros, jnp.float64(0.0), _n14_init, _comp_mfracs_init,
        X3_profile=X3_profile)

    R_final = _build_residual_atm(
        jax.lax.stop_gradient(y_final), q_mesh,  # GP-9: CONVERGENCE_FLAGS
        jax.lax.stop_gradient(M_star), X_profile, Z, alpha_mlt,  # GP-9: CONVERGENCE_FLAGS
        R_phot=R_phot, N14_profile=_n14_init, X3_profile=X3_profile)
    res_norm = jax.lax.stop_gradient(jnp.max(jnp.abs(R_final)))  # GP-9: CONVERGENCE_FLAGS
    return {
        'y': y_final,
        'converged': newton_converged,
        'residual_norm': res_norm,
    }
