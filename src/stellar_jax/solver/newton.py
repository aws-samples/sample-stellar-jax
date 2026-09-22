"""Newton iteration loops + IFT @custom_vjp boundaries (non-production).

.. note:: **LEGACY** — the production Newton solver lives in
   ``solver/continuation.py`` (continuation-based, atmosphere BC). This
   module is retained for diagnostic and ZAMS cold-start use only.
  - henyey_solve_differentiable: ZAMS differentiable solver
  - _henyey_continuation_ift + fwd/bwd: static-equilibrium continuation with IFT
  - _henyey_eps_grav_ift_impl + fwd/bwd: eps_grav-coupled Newton with IFT
  - henyey_solve_from_state: legacy entry point (dispatches to the above)

The PRODUCTION continuation solver is in solver/continuation.py.

Phase P4-4 of the solver/ refactoring (per the module redesign spec).
"""
import functools

import jax
import jax.numpy as jnp
from jax import lax

from stellar_jax.config.constants import Msun, Lsun, sigma_sb, Y_BBN, DY_DZ
from stellar_jax.config.mesh_defaults import N_COMP
from stellar_jax.solver.thomas import block_thomas_solve, _transpose_block_tridiag
from stellar_jax.solver.damping import _apply_per_zone_damping
from stellar_jax.solver.residual import _build_residual
from stellar_jax.solver.jacobian import _jacobian_blocks
from stellar_jax.solver.eps_grav import _energy_row_scale_factors
from stellar_jax.solver.adjoint import _convergence_gate_simple
# Convergence tolerance for dynamic Newton iteration.
_HENYEY_CONV_TOL = 1e-4


# ═══════════════════════════════════════════════════════════════
# Basic Newton (no IFT, diagnostic only)
# ═══════════════════════════════════════════════════════════════

def henyey_solve(M_solar, Z, alpha_mlt, logP_guess, logT_guess, r_guess, L_guess,
                 q_mesh, n_iter=50, tol=1e-4):
    """Henyey Newton-Raphson solver for stellar structure.

    Solves on the first N_s = len(q_mesh)-1 interfaces (excluding the outermost
    cell which spans the unresolved envelope). The outermost solved interface is
    pinned (Dirichlet BC) to the values from the shooting/atmosphere model.

    Returns dict with 'log_L', 'log_Teff', 'converged', 'residual_norm', 'y'.
    """
    M_star = jnp.float64(M_solar) * Msun
    Z_j = jnp.float64(Z)
    alpha_j = jnp.float64(alpha_mlt)

    Y = Y_BBN + DY_DZ * Z
    X_init = max(1.0 - Y - Z, 0.5)
    X_profile = jnp.full(N_COMP, X_init)

    ln_r = jnp.log(jnp.maximum(r_guess, 1.0))
    ln_P = jnp.log(10.0) * logP_guess
    ln_T = jnp.log(10.0) * logT_guess
    ell = L_guess / Lsun
    y_full = jnp.stack([ln_r, ln_P, ln_T, ell], axis=-1)

    N_s = q_mesh.shape[0] - 1
    y = y_full[:N_s]

    def newton_step(y, _):
        R = _build_residual(y, q_mesh, M_star, X_profile, Z_j, alpha_j)
        R_norm = jnp.max(jnp.abs(R))
        A, B, C = _jacobian_blocks(y, q_mesh, M_star, X_profile, Z_j, alpha_j)
        dy = block_thomas_solve(A, B, C, -R)
        dy = _apply_per_zone_damping(dy, R_norm)
        y_new = y + dy
        return y_new, R_norm

    y_final, residuals = lax.scan(newton_step, y, None, length=n_iter)

    R_final = _build_residual(y_final, q_mesh, M_star, X_profile, Z_j, alpha_j)
    res_norm = jnp.max(jnp.abs(R_final))

    r_surf = jnp.exp(y_full[-1, 0])
    ell_surf = y_full[-1, 3]
    L_surf = ell_surf * Lsun
    T_eff = (L_surf / (4.0 * jnp.pi * sigma_sb * r_surf**2))**0.25
    log_Teff = jnp.log10(T_eff)
    log_L = jnp.log10(jnp.maximum(L_surf, 1e-30) / Lsun)

    return {
        'log_L': float(log_L),
        'log_Teff': float(log_Teff),
        'converged': bool(res_norm < tol),
        'residual_norm': float(res_norm),
        'y': y_final,
    }


# ═══════════════════════════════════════════════════════════════
# ZAMS pinned-BC Newton + IFT @custom_vjp
# ═══════════════════════════════════════════════════════════════

@functools.partial(jax.custom_vjp, nondiff_argnums=(6,))
def _henyey_newton(y_init, q_mesh, M_star, X_profile, Z, alpha_mlt, n_iter):
    """Newton iteration with IFT @custom_vjp on the converged state.

    nondiff_argnums=(6,) → n_iter is static.
    """
    def newton_step(carry, _):
        y_st, converged = carry
        R = _build_residual(y_st, q_mesh, M_star, X_profile, Z, alpha_mlt)
        R_norm = jnp.max(jnp.abs(R))
        A, B, C = _jacobian_blocks(y_st, q_mesh, M_star, X_profile, Z, alpha_mlt)
        dy = block_thomas_solve(A, B, C, -R)
        dy = _apply_per_zone_damping(dy, R_norm)
        dy = jnp.where(converged, 0.0, dy)
        y_new = y_st + dy
        converged_new = converged | (R_norm < _HENYEY_CONV_TOL)
        return (y_new, converged_new), None

    (y_final, _), _ = lax.scan(newton_step, (y_init, jnp.bool_(False)), None, length=n_iter)
    return y_final


def _henyey_newton_fwd(y_init, q_mesh, M_star, X_profile, Z, alpha_mlt, n_iter):
    """Forward pass: run Newton, save converged state + inputs for backward."""
    y_final = _henyey_newton(y_init, q_mesh, M_star, X_profile, Z, alpha_mlt, n_iter)
    R_final = _build_residual(y_final, q_mesh, M_star, X_profile, Z, alpha_mlt)
    res_norm = jnp.max(jnp.abs(R_final))
    converged = res_norm < _HENYEY_CONV_TOL
    return y_final, (y_final, q_mesh, M_star, X_profile, Z, alpha_mlt, converged)


def _henyey_newton_bwd(n_iter, res, g_y):
    """IFT backward pass through converged Henyey Newton solution.

    At convergence, ∂y*/∂θ = -J_y^{-1} @ ∂F/∂θ, so the VJP is
    g^T @ ∂y*/∂θ = -(J_y^{-T} @ g)^T @ ∂F/∂θ.
    """
    y_final, q_mesh, M_star, X_profile, Z, alpha_mlt, converged = res
    N_s = y_final.shape[0]

    if n_iter == 0:
        g_qm = jnp.zeros_like(q_mesh)
        g_Ms = jnp.zeros_like(M_star)
        g_Xp = jnp.zeros_like(X_profile)
        g_Zp = jnp.zeros_like(Z)
        g_ap = jnp.zeros_like(alpha_mlt)
        return g_y, g_qm, g_Ms, g_Xp, g_Zp, g_ap

    A, B, C = _jacobian_blocks(y_final, q_mesh, M_star, X_profile, Z, alpha_mlt)

    A_T, B_T, C_T = _transpose_block_tridiag(A, B, C)
    lam = block_thomas_solve(A_T, B_T, C_T, g_y)

    def residual_fn_split(y_interior, y_surface_23, qm, Ms, Xp, Zp, ap):
        y_full = y_interior.at[N_s - 1, 2].set(y_surface_23[0])
        y_full = y_full.at[N_s - 1, 3].set(y_surface_23[1])
        return _build_residual(y_full, qm, Ms, Xp, Zp, ap)

    y_surface_23 = y_final[N_s - 1, 2:4]
    _, vjp_fn = jax.vjp(residual_fn_split, y_final, y_surface_23,
                        q_mesh, M_star, X_profile, Z, alpha_mlt)
    _g_y_in, g_surf_23, g_qm, g_Ms, g_Xp, g_Zp, g_ap = vjp_fn(-lam)

    g_y_init = jnp.zeros_like(y_final)
    g_y_init = g_y_init.at[N_s - 1, 2].set(g_y[N_s - 1, 2] + g_surf_23[0])
    g_y_init = g_y_init.at[N_s - 1, 3].set(g_y[N_s - 1, 3] + g_surf_23[1])

    return _convergence_gate_simple(converged, g_y_init, g_qm, g_Ms, g_Xp, g_Zp, g_ap)


_henyey_newton.defvjp(_henyey_newton_fwd, _henyey_newton_bwd)


def henyey_solve_differentiable(M_solar, Z, alpha_mlt, n_mesh=1000, n_iter=50):
    """Differentiable ZAMS solver: returns (log_L, log_Teff, y_conv) with IFT gradients."""
    from stellar_jax.structure import initial_guess, newton_solve_xprofile, build_model_on_mesh
    from stellar_jax.config.mesh_defaults import N_NEWTON_COLD

    Y = Y_BBN + DY_DZ * Z
    X_init = 1.0 - Y - Z
    X_init = jnp.where(X_init > 0.5, X_init, 0.5)
    X_profile = jnp.full(N_COMP, X_init)

    logL_g, logTe_g = initial_guess(lax.stop_gradient(M_solar))  # GP-9: CONVERGENCE_FLAGS
    logL_g = jnp.float64(logL_g)
    logTe_g = jnp.float64(logTe_g)

    logL, logTe = newton_solve_xprofile(
        M_solar, X_profile, Z, jnp.float64(0.0), logL_g, logTe_g, alpha_mlt, N_NEWTON_COLD)

    model = build_model_on_mesh(M_solar, logL, logTe, X_profile, Z, alpha_mlt, n_mesh)
    q_mesh = model['q']
    M_star = M_solar * Msun

    ln_r = jnp.log(jnp.maximum(model['r'], 1.0))
    ln_P = jnp.log(10.0) * model['logP']
    ln_T = jnp.log(10.0) * model['logT']
    ell = model['L'] / Lsun
    y_full = jnp.stack([ln_r, ln_P, ln_T, ell], axis=-1)
    N_s = q_mesh.shape[0] - 1
    y_init = y_full[:N_s]

    y_conv = _henyey_newton(y_init, q_mesh, M_star, X_profile, Z, alpha_mlt, n_iter)
    return logL, logTe, y_conv


# ═══════════════════════════════════════════════════════════════
# Static-equilibrium continuation + IFT @custom_vjp
# ═══════════════════════════════════════════════════════════════

@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _henyey_continuation_ift(y_prev, q_mesh, M_star, X_profile, Z, alpha_mlt,
                             n_iter, tol):
    """Newton continuation with IFT @custom_vjp (static equilibrium, dt=None).

    nondiff_argnums=(6, 7) → n_iter, tol are static.
    """
    def newton_step(carry, _):
        y, converged = carry
        R = _build_residual(y, q_mesh, M_star, X_profile, Z, alpha_mlt)
        R_norm = jnp.max(jnp.abs(R))
        A, B, C = _jacobian_blocks(y, q_mesh, M_star, X_profile, Z, alpha_mlt)
        dy = block_thomas_solve(A, B, C, -R)
        dy = _apply_per_zone_damping(dy, R_norm)

        # Residual-based backtracking: halve alpha if the trial step causes
        # the residual to more than double. 4 halvings → min alpha = 1/16.
        # This matches the algorithm on main. The MESA cosine taper
        # stiffens the Jacobian at the 1 M☉ core (logT~7.17), narrowing the
        # convergence basin from >1.6% to ~1.3% (measured). The step control
        # mechanism is unchanged; the test perturbation is adapted to the
        # correct (stiffer) physics.
        # Ref: MESA star_solver.f90:396-411; Paxton et al. (2011) §6.3.
        def _backtrack_step(alpha, _):
            y_trial = y + alpha * dy
            R_trial = _build_residual(y_trial, q_mesh, M_star, X_profile, Z, alpha_mlt)
            R_trial_norm = jnp.max(jnp.abs(R_trial))
            should_reduce = (R_trial_norm > 2.0 * R_norm) & (alpha > 1.0 / 16.0)
            alpha_new = jnp.where(should_reduce, alpha * 0.5, alpha)
            return alpha_new, None

        alpha_final, _ = lax.scan(_backtrack_step, jnp.float64(1.0), None, length=4)
        dy_final = lax.stop_gradient(alpha_final) * dy  # GP-9: CONVERGENCE_FLAGS
        dy_final = jnp.where(converged, 0.0, dy_final)
        y_new = y + dy_final
        converged_new = converged | (R_norm < tol)
        return (y_new, converged_new), R_norm

    (y_final, _), _ = lax.scan(newton_step, (y_prev, jnp.bool_(False)), None, length=n_iter)
    return y_final


def _henyey_continuation_ift_fwd(y_prev, q_mesh, M_star, X_profile, Z, alpha_mlt,
                                 n_iter, tol):
    y_final = _henyey_continuation_ift(y_prev, q_mesh, M_star, X_profile, Z, alpha_mlt,
                                       n_iter, tol)
    R_final = _build_residual(y_final, q_mesh, M_star, X_profile, Z, alpha_mlt)
    res_norm = jnp.max(jnp.abs(R_final))
    converged = res_norm < tol
    return y_final, (y_final, q_mesh, M_star, X_profile, Z, alpha_mlt, converged)


def _henyey_continuation_ift_bwd(n_iter, tol, res, g_y):
    """IFT backward pass with convergence guard."""
    y_final, q_mesh, M_star, X_profile, Z, alpha_mlt, converged = res
    N_s = y_final.shape[0]

    A, B, C = _jacobian_blocks(y_final, q_mesh, M_star, X_profile, Z, alpha_mlt)

    A_T, B_T, C_T = _transpose_block_tridiag(A, B, C)
    lam = block_thomas_solve(A_T, B_T, C_T, g_y)

    def residual_fn_split(y_interior, y_surface_23, qm, Ms, Xp, Zp, ap):
        y_full = y_interior.at[N_s - 1, 2].set(y_surface_23[0])
        y_full = y_full.at[N_s - 1, 3].set(y_surface_23[1])
        return _build_residual(y_full, qm, Ms, Xp, Zp, ap)

    y_surface_23 = y_final[N_s - 1, 2:4]
    _, vjp_fn = jax.vjp(residual_fn_split, y_final, y_surface_23,
                        q_mesh, M_star, X_profile, Z, alpha_mlt)
    _g_y_in, g_surf_23, g_qm, g_Ms, g_Xp, g_Zp, g_ap = vjp_fn(-lam)

    g_y_init = jnp.zeros_like(y_final)
    g_y_init = g_y_init.at[N_s - 1, 2].set(g_y[N_s - 1, 2] + g_surf_23[0])
    g_y_init = g_y_init.at[N_s - 1, 3].set(g_y[N_s - 1, 3] + g_surf_23[1])

    return _convergence_gate_simple(converged, g_y_init, g_qm, g_Ms, g_Xp, g_Zp, g_ap)


_henyey_continuation_ift.defvjp(_henyey_continuation_ift_fwd,
                                _henyey_continuation_ift_bwd)


# ═══════════════════════════════════════════════════════════════
# eps_grav-coupled Newton + IFT @custom_vjp
# ═══════════════════════════════════════════════════════════════

@functools.partial(jax.custom_vjp, nondiff_argnums=(8, 9))
def _henyey_eps_grav_ift_impl(y_prev, q_mesh, M_star, X_profile, Z, alpha_mlt,
                              y_prev_step, inv_dt, n_iter, tol):
    """Newton with eps_grav + IFT backward (prevents XLA graph explosion).

    nondiff_argnums=(8, 9) → n_iter, tol are static.
    """
    ln_T_prev_hist = y_prev_step[:, 2]
    ln_P_prev_hist = y_prev_step[:, 1]

    def newton_step(carry, _):
        y, converged = carry
        R = _build_residual(y, q_mesh, M_star, X_profile, Z, alpha_mlt,
                            ln_T_prev_hist, ln_P_prev_hist, inv_dt)
        R_norm = jnp.max(jnp.abs(R))
        A, B, C = _jacobian_blocks(y, q_mesh, M_star, X_profile, Z, alpha_mlt,
                                   ln_T_prev_hist, ln_P_prev_hist, inv_dt)
        dy = block_thomas_solve(A, B, C, -R)
        dy = _apply_per_zone_damping(dy, R_norm)

        # Residual-based backtracking (same as the continuation path).
        # Ref: MESA star_solver.f90:396-411; Paxton et al. (2011) §6.3.
        def _backtrack_step(alpha, _):
            y_trial = y + alpha * dy
            R_trial = _build_residual(y_trial, q_mesh, M_star, X_profile, Z, alpha_mlt,
                                      ln_T_prev_hist, ln_P_prev_hist, inv_dt)
            R_trial_norm = jnp.max(jnp.abs(R_trial))
            should_reduce = (R_trial_norm > 2.0 * R_norm) & (alpha > 1.0 / 16.0)
            alpha_new = jnp.where(should_reduce, alpha * 0.5, alpha)
            return alpha_new, None

        alpha_final, _ = lax.scan(_backtrack_step, jnp.float64(1.0), None, length=4)
        dy_final = lax.stop_gradient(alpha_final) * dy  # GP-9: CONVERGENCE_FLAGS
        dy_final = jnp.where(converged, 0.0, dy_final)
        y_new = y + dy_final
        converged_new = converged | (R_norm < tol)
        return (y_new, converged_new), R_norm

    (y_final, _), _ = lax.scan(newton_step, (y_prev, jnp.bool_(False)), None, length=n_iter)
    return y_final


def _henyey_eps_grav_ift_fwd(y_prev, q_mesh, M_star, X_profile, Z, alpha_mlt,
                             y_prev_step, inv_dt, n_iter, tol):
    y_final = _henyey_eps_grav_ift_impl(y_prev, q_mesh, M_star, X_profile, Z, alpha_mlt,
                                        y_prev_step, inv_dt, n_iter, tol)
    R_final = _build_residual(y_final, q_mesh, M_star, X_profile, Z, alpha_mlt,
                              y_prev_step[:, 2], y_prev_step[:, 1], inv_dt)
    res_norm = jnp.max(jnp.abs(R_final))
    converged = res_norm < tol
    return y_final, (y_final, q_mesh, M_star, X_profile, Z, alpha_mlt,
                     y_prev_step, inv_dt, converged)


def _henyey_eps_grav_ift_bwd(n_iter, tol, res, g_y):
    """IFT backward pass for eps_grav coupled Henyey Newton."""
    (y_final, q_mesh, M_star, X_profile, Z, alpha_mlt,
     y_prev_step, inv_dt, converged) = res
    N_s = y_final.shape[0]

    ln_T_prev_hist = y_prev_step[:, 2]
    ln_P_prev_hist = y_prev_step[:, 1]

    A, B, C = _jacobian_blocks(y_final, q_mesh, M_star, X_profile, Z, alpha_mlt,
                               ln_T_prev_hist, ln_P_prev_hist, inv_dt)

    # Energy-row scaling (MESA set_energy_eqn_scal, star_utils.f90:3678)
    energy_scale = _energy_row_scale_factors(y_final, q_mesh, X_profile, Z, inv_dt)
    scale_row = energy_scale[1:, None]
    A = A.at[1:, 0, :].multiply(scale_row)
    B = B.at[1:, 0, :].multiply(scale_row)
    C = C.at[1:, 0, :].multiply(scale_row)

    A_T, B_T, C_T = _transpose_block_tridiag(A, B, C)
    mu = block_thomas_solve(A_T, B_T, C_T, g_y)

    # Recover λ = S * μ
    lam = mu.at[1:, 0].multiply(energy_scale[1:])

    def residual_fn_all(y_interior, y_surface_23, y_prev_st, inv_dt_p,
                        qm, Ms, Xp, Zp, ap):
        y_full = y_interior.at[N_s - 1, 2].set(y_surface_23[0])
        y_full = y_full.at[N_s - 1, 3].set(y_surface_23[1])
        return _build_residual(y_full, qm, Ms, Xp, Zp, ap,
                               y_prev_st[:, 2], y_prev_st[:, 1], inv_dt_p)

    y_surface_23 = y_final[N_s - 1, 2:4]
    _, vjp_fn = jax.vjp(residual_fn_all, y_final, y_surface_23, y_prev_step,
                        inv_dt, q_mesh, M_star, X_profile, Z, alpha_mlt)
    (_g_y_in, g_surf_23, g_y_prev_step, g_inv_dt,
     g_qm, g_Ms, g_Xp, g_Zp, g_ap) = vjp_fn(-lam)

    g_y_init = jnp.zeros_like(y_final)
    g_y_init = g_y_init.at[N_s - 1, 2].set(g_y[N_s - 1, 2] + g_surf_23[0])
    g_y_init = g_y_init.at[N_s - 1, 3].set(g_y[N_s - 1, 3] + g_surf_23[1])

    return _convergence_gate_simple(converged, g_y_init, g_qm, g_Ms, g_Xp, g_Zp, g_ap,
                                    g_y_prev_step, g_inv_dt)


_henyey_eps_grav_ift_impl.defvjp(_henyey_eps_grav_ift_fwd,
                                 _henyey_eps_grav_ift_bwd)


# ═══════════════════════════════════════════════════════════════
# Legacy entry point (dispatches by dt)
# ═══════════════════════════════════════════════════════════════

def henyey_solve_from_state(y_prev, q_mesh, M_star, X_profile, Z, alpha_mlt,
                            n_iter=60, tol=1e-4, dt=None, y_prev_step=None):
    """Re-converge Henyey from a previous state (legacy, non-atm tests).

    Dispatches to _henyey_continuation_ift (dt=None) or
    _henyey_eps_grav_ift_impl (dt provided).
    """
    if dt is None or y_prev_step is None:
        y_final = _henyey_continuation_ift(y_prev, q_mesh, M_star, X_profile, Z,
                                           alpha_mlt, n_iter, tol)
        R_final = _build_residual(y_final, q_mesh, M_star, X_profile, Z, alpha_mlt)
        res_norm = jnp.max(jnp.abs(R_final))
        return {
            'y': y_final,
            'converged': res_norm < tol,
            'residual_norm': res_norm,
        }

    inv_dt = jnp.where(dt > 0.0, 1.0 / dt, 0.0)
    y_final = _henyey_eps_grav_ift_impl(y_prev, q_mesh, M_star, X_profile, Z,
                                        alpha_mlt, y_prev_step, inv_dt, n_iter, tol)
    R_final = _build_residual(y_final, q_mesh, M_star, X_profile, Z, alpha_mlt,
                              y_prev_step[:, 2], y_prev_step[:, 1], inv_dt)
    res_norm = jnp.max(jnp.abs(R_final))

    return {
        'y': y_final,
        'converged': res_norm < tol,
        'residual_norm': res_norm,
    }
