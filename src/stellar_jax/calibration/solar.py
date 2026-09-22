"""Solar calibration: residual functions + Newton-Raphson optimizers.

Extracted from evolution.py. This module is a leaf consumer of
evolution/ — it calls evolve_star/evolve_solar but is never imported by other
physics modules.

Differentiability contract (see contracts.py):
  - evolve_solar, solar_residual, solar_residual_henyey: fully differentiable
  - solar_calibrate*: NOT differentiable (Python Newton-Raphson loops)

JAX pitfall §5.5 (closure capture): evolve_solar's lax.scan body captures
~15 names from module scope. All captured names are explicitly imported below.
"""
import logging
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from stellar_jax.config.constants import (
    sigma_sb,
    Lsun, Rsun,
    SECONDS_PER_YEAR,
    Q_PER_G,
)
from stellar_jax.evolution.contracts import (
    SOLAR_CARRY_X, SOLAR_CARRY_Y, SOLAR_CARRY_Z,
)
from stellar_jax.config.mesh_defaults import (
    F_OV,
    N_NEWTON_COLD, N_NEWTON_WARM, N_COMP,
)
from stellar_jax.config.calibration import ALPHA_SOLAR, Y0_SOLAR
from stellar_jax.structure import (
    initial_guess, shoot_xprofile, newton_solve_xprofile,
    newton_solve_with_zprofile,
)

# Composition operations captured by evolve_solar's lax.scan body
from stellar_jax.composition import burn_composition, mix_composition, diffuse_composition

# Structural hard limits — centralized in config/physics_floors.py
from stellar_jax.config.physics_floors import DELTA_LGL_HARD_LIMIT, DELTA_LGTE_HARD_LIMIT

from stellar_jax.calibration.interpolation import lagrange_interp_3pt

logger = logging.getLogger(__name__)


# ======================================================================
# Shared Newton-Raphson 2D optimizer (DRY: was copy-pasted 3×)
# ======================================================================

def _newton_raphson_2d(residual_fn, x0, bounds, tol=1e-7, max_iter=30,
                       fd_scale_alpha=10.0, fd_scale_Y=1.0,
                       fd_min_alpha=1e-7, fd_max_alpha=0.005,
                       fd_min_Y=1e-8, fd_max_Y=0.0005,
                       max_step_alpha=0.5, max_step_Y=0.02,
                       label="Solar Calibration", verbose=False):
    """Newton-Raphson optimizer for 2D (α, Y₀) → (logL, logR) calibration.

    Uses forward finite-difference Jacobian with adaptive step sizing.
    The adaptive FD steps (scaled to residual magnitude) maintain the Jacobian
    in the linear regime for quadratic convergence.

    Parameters
    ----------
    residual_fn : callable
        (alpha, Y0) -> (logL, logR) as Python floats.
    x0 : tuple of (alpha_init, Y0_init)
    bounds : tuple of ((alpha_min, alpha_max), (Y_min, Y_max))
    tol : float
        Convergence tolerance on |logL| and |logR|.
    max_iter : int
        Maximum Newton iterations.
    fd_scale_alpha, fd_scale_Y : float
        FD step = clip(res_scale * fd_scale, fd_min, fd_max).
    max_step_alpha, max_step_Y : float
        Maximum Newton step per iteration.
    label : str
        Label for progress logging.
    verbose : bool
        If True, log per-iteration progress at INFO level.

    Returns
    -------
    dict with keys: alpha, Y0, log_L_residual, log_R_residual,
                    iterations, converged.

    Reference: Press et al. (2007), Numerical Recipes §5.7 (forward differences).
    """
    (alpha_min, alpha_max), (Y_min, Y_max) = bounds
    alpha, Y0 = x0

    best_loss = 1e10
    best_alpha, best_Y0 = alpha, Y0

    if verbose:
        logger.info("=" * 70)
        logger.info(f"{label}")
        logger.info(f"  Starting: α={alpha:.8f}, Y₀={Y0:.8f}")
        logger.info("=" * 70)

    for iteration in range(max_iter):
        logL, logR = residual_fn(alpha, Y0)
        loss = logL**2 + logR**2

        if loss < best_loss:
            best_loss = loss
            best_alpha, best_Y0 = alpha, Y0

        if verbose:
            logger.info(f"  Iter {iteration:2d}: α={alpha:.8f}, Y₀={Y0:.8f} "
                        f"→ log_L={logL:+.2e}, log_R={logR:+.2e}  loss={loss:.2e}")

        if abs(logL) < tol and abs(logR) < tol:
            if verbose:
                logger.info(f"\n  ✓ CONVERGED in {iteration} iterations!")
                logger.info(f"    α_solar = {alpha:.8f}")
                logger.info(f"    Y₀_solar = {Y0:.8f}")
            return {'alpha': alpha, 'Y0': Y0,
                    'log_L_residual': logL, 'log_R_residual': logR,
                    'iterations': iteration, 'converged': True}

        # Forward FD Jacobian with adaptive step sizes
        res_scale = max(abs(logL), abs(logR))
        da = float(np.clip(res_scale * fd_scale_alpha, fd_min_alpha, fd_max_alpha))
        dY = float(np.clip(res_scale * fd_scale_Y, fd_min_Y, fd_max_Y))

        logL_pa, logR_pa = residual_fn(alpha + da, Y0)
        logL_pY, logR_pY = residual_fn(alpha, Y0 + dY)

        J = np.array([
            [(logL_pa - logL) / da, (logL_pY - logL) / dY],
            [(logR_pa - logR) / da, (logR_pY - logR) / dY]
        ])

        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-20:
            if verbose:
                logger.warning("  Near-singular Jacobian, using gradient descent step")
            grad_a = 2 * logL * J[0, 0] + 2 * logR * J[1, 0]
            grad_Y = 2 * logL * J[0, 1] + 2 * logR * J[1, 1]
            norm = np.sqrt(grad_a**2 + grad_Y**2) + 1e-30
            alpha -= 0.01 * grad_a / norm
            Y0 -= 0.001 * grad_Y / norm
        else:
            residual = np.array([logL, logR])
            delta = np.linalg.solve(J, -residual)
            scale = min(1.0,
                        max_step_alpha / (abs(delta[0]) + 1e-30),
                        max_step_Y / (abs(delta[1]) + 1e-30))
            delta *= scale
            alpha += delta[0]
            Y0 += delta[1]

        alpha = float(np.clip(alpha, alpha_min, alpha_max))
        Y0 = float(np.clip(Y0, Y_min, Y_max))

    # Return best result on non-convergence
    alpha, Y0 = best_alpha, best_Y0
    logL, logR = residual_fn(alpha, Y0)
    if verbose:
        logger.warning(f"  Did not converge in {max_iter} iterations")
        logger.warning(f"    Best: α={alpha:.8f}, Y₀={Y0:.8f}")
        logger.warning(f"    Residuals: log_L={logL:+.7f}, log_R={logR:+.7f}")
    return {'alpha': alpha, 'Y0': Y0,
            'log_L_residual': logL, 'log_R_residual': logR,
            'iterations': max_iter, 'converged': False}


# ======================================================================
# evolve_solar — the JIT-compiled 1 M☉ lax.scan kernel (KEPT WHOLE)
# ======================================================================

# lint: scan-body exception — evolve_solar is a JIT boundary with an inner
# lax.scan step_fn; splitting would change the XLA trace graph.
@partial(jax.jit, static_argnames=('max_steps', 't_max', 'z_feedback'))
def evolve_solar(alpha_mlt, Y_init, Z, max_steps=500, t_max=None, z_feedback=False):
    """Evolve 1 M☉ with explicit Y_init (for solar calibration).

    Uses the shooting solver (newton_solve_xprofile + shoot_xprofile) for
    self-consistent structure at each timestep. This is the code path that
    the calibration constants ALPHA_SOLAR/Y0_SOLAR were derived from.

    Args:
        t_max: Optional maximum age in years. If set, evolution freezes once
               this age is reached (dt→0). Used by compare_model_s to get
               composition at exactly solar age.
        z_feedback: If True, track Z_profile from diffusion and feed it back
               into shoot_xprofile at each step (affects opacity/EOS). Used
               by compare_model_s for Model S density comparison.
    """
    mass = jnp.float64(1.0)
    Z = jnp.asarray(Z, dtype=jnp.float64)
    alpha_mlt = jnp.asarray(alpha_mlt, dtype=jnp.float64)
    Y_init = jnp.asarray(Y_init, dtype=jnp.float64)
    X_init = jnp.maximum(1.0 - Y_init - Z, 0.5)
    f_ov = jnp.asarray(F_OV, dtype=jnp.float64)
    t_max_sec = jnp.float64(t_max * SECONDS_PER_YEAR if t_max is not None else 1e30)

    X_profile_init = jnp.full(N_COMP, X_init)
    Y_profile_init = jnp.full(N_COMP, Y_init)

    log_L_g, log_Te_g = initial_guess(mass)
    logL0, logTe0 = newton_solve_xprofile(mass, X_profile_init, Z, jnp.float64(0.0),
                                           log_L_g, log_Te_g, alpha_mlt, N_NEWTON_COLD)

    _, shell_data_init, _ = shoot_xprofile(mass, logL0, logTe0, X_profile_init,
                                        Z, jnp.float64(0.0), alpha_mlt)
    eps_max_init = jnp.maximum(jnp.max(shell_data_init[:, 0]), 1e-10)
    t_nuc = X_init * Q_PER_G / eps_max_init
    dt_init = t_nuc / 50000.0

    varcontrol_target = jnp.float64(1e-3)
    dt_grow = jnp.float64(1.5)
    dt_shrink = jnp.float64(0.5)
    reject_threshold = jnp.float64(4e-3)

    logT_init = shell_data_init[:, 6]
    logRho_init = shell_data_init[:, 7]
    logP_init = shell_data_init[:, 9]

    def step_fn(carry, _):
        X_prof, Y_prof, Z_prof, logL, logTe, t, dt, prev_logT, prev_logP, prev_logRho = carry
        t_age = t / SECONDS_PER_YEAR

        logL_new, logTe_new = newton_solve_xprofile(
            mass, X_prof, Z, t_age, logL, logTe, alpha_mlt, N_NEWTON_WARM)
        if z_feedback:
            _, shell_data, _ = shoot_xprofile(
                mass, logL_new, logTe_new, X_prof, Z, t_age, alpha_mlt,
                Z_profile=Z_prof)
        else:
            _, shell_data, _ = shoot_xprofile(
                mass, logL_new, logTe_new, X_prof, Z, t_age, alpha_mlt)

        X_burned, Y_burned = burn_composition(X_prof, Y_prof, shell_data, dt)
        X_mixed = mix_composition(X_burned, shell_data, M_solar=mass, f_ov=f_ov)
        if z_feedback:
            X_diff, Y_diff, Z_diff = diffuse_composition(
                X_mixed, Y_burned, shell_data, dt, mass,
                Z_profile=Z_prof, use_exact_structure=True, use_4species=True)
            Z_new = Z_diff
        else:
            X_diff, Y_diff, _ = diffuse_composition(
                X_mixed, Y_burned, shell_data, dt, mass, use_exact_structure=True)
            Z_new = Z_prof
        Y_mixed = mix_composition(Y_diff, shell_data, M_solar=mass, f_ov=f_ov)

        L = 10.0**logL_new * Lsun
        Te = 10.0**logTe_new
        R_star = jnp.sqrt(L / (4.0 * jnp.pi * sigma_sb)) / Te**2
        log_R = jnp.log10(R_star / Rsun)

        # Full composition change for varcontrol sizing (see evolve_star)
        max_dX = jnp.max(jnp.abs(X_diff - X_prof))
        cur_logT = shell_data[:, 6]
        cur_logRho = shell_data[:, 7]
        cur_logP = shell_data[:, 9]
        max_delta_logT = jnp.max(jnp.abs(cur_logT - prev_logT))
        max_delta_logRho = jnp.max(jnp.abs(cur_logRho - prev_logRho))
        delta_logL_s = jnp.abs(logL_new - logL)
        delta_logTe_s = jnp.abs(logTe_new - logTe)
        varcontrol_reject = jnp.maximum(delta_logL_s, delta_logTe_s)
        varcontrol = jnp.maximum(varcontrol_reject,
                                 jnp.maximum(max_delta_logT, max_delta_logRho))
        varcontrol = jnp.maximum(varcontrol, max_dX)

        # Structural hard limits (MESA delta_lgL_limit analog)
        structural_reject = (jax.lax.stop_gradient(delta_logL_s) > DELTA_LGL_HARD_LIMIT) | (jax.lax.stop_gradient(delta_logTe_s) > DELTA_LGTE_HARD_LIMIT)  # GP-1: TIMESTEP (gradient policy)
        dt_at_floor = dt <= (1.1e5 * SECONDS_PER_YEAR)
        reject = ((varcontrol_reject > reject_threshold) | structural_reject) & (~dt_at_floor)
        dt_next_reject = dt * 0.5
        ratio = varcontrol_target / jnp.maximum(varcontrol, 1e-30)
        factor = jnp.clip(ratio, dt_shrink, dt_grow)
        dt_next_accept = dt * factor

        X_out = jnp.where(reject, X_prof, X_diff)
        Y_out = jnp.where(reject, Y_prof, Y_mixed)
        Z_out = jnp.where(reject, Z_prof, Z_new)
        logL_out = jnp.where(reject, logL, logL_new)
        logTe_out = jnp.where(reject, logTe, logTe_new)
        t_new = jnp.where(reject, t, t + dt)
        dt_next = jnp.where(reject, dt_next_reject, dt_next_accept)
        logT_out = jnp.where(reject, prev_logT, cur_logT)
        logRho_out = jnp.where(reject, prev_logRho, cur_logRho)
        logP_out = jnp.where(reject, prev_logP, cur_logP)

        eps_max = jnp.maximum(jnp.max(shell_data[:, 0]), 1e-10)
        dt_max = 0.10 * Q_PER_G / eps_max
        dt_next = jnp.minimum(dt_next, dt_max)
        dt_next = jnp.maximum(dt_next, 1e5 * SECONDS_PER_YEAR)

        X_center = X_out[0]
        dt_next = jnp.where(X_center < 0.005, 0.0, dt_next)

        # t_max freeze: once t >= t_max, set dt=0 to freeze composition
        dt_next = jnp.where(t_new >= t_max_sec, 0.0, dt_next)
        dt_next = jnp.minimum(dt_next, jnp.maximum(t_max_sec - t_new, 0.0))

        t_new_out = t_new
        log_R_out = jnp.where(reject, jnp.float64(0.0), log_R)
        logTe_out_val = jnp.where(reject, jnp.float64(0.0), logTe_new)
        return (X_out, Y_out, Z_out, logL_out, logTe_out, t_new_out, dt_next,
                logT_out, logP_out, logRho_out), \
               jnp.array([t_new_out, logL_out, log_R_out, logTe_out_val])

    Z_profile_init = jnp.full(N_COMP, Z)
    init_carry = (X_profile_init, Y_profile_init, Z_profile_init, logL0, logTe0,
                  jnp.float64(0.0), dt_init, logT_init, logP_init, logRho_init)
    final_carry, out = lax.scan(step_fn, init_carry, None, length=max_steps)

    X_final = final_carry[SOLAR_CARRY_X]
    Y_final = final_carry[SOLAR_CARRY_Y]
    Z_final = final_carry[SOLAR_CARRY_Z]
    return out[:, 0] / SECONDS_PER_YEAR, out[:, 1], out[:, 2], out[:, 3], X_final, Y_final, Z_final


# ======================================================================
# Solar residual functions (differentiable)
# ======================================================================

def solar_residual(alpha_mlt, Y_init, Z=0.0188, t_target=4.57e9, max_steps=500):
    """Compute (log_L, log_R) at t_target for 1 M☉.

    Uses quadratic (3-point Lagrange) interpolation in age for robustness
    against timestep-size variations.
    """
    ages, log_L, log_R, _logTe, _X, _Y, _Z = evolve_solar(alpha_mlt, Y_init, Z, max_steps=max_steps)
    idx = jnp.searchsorted(ages, t_target) - 1
    idx = jnp.clip(idx, 1, max_steps - 2)

    t0, t1, t2 = ages[idx - 1], ages[idx], ages[idx + 1]
    logL_at_t = lagrange_interp_3pt(t0, t1, t2,
                                     log_L[idx - 1], log_L[idx], log_L[idx + 1],
                                     t_target)
    logR_at_t = lagrange_interp_3pt(t0, t1, t2,
                                     log_R[idx - 1], log_R[idx], log_R[idx + 1],
                                     t_target)
    return logL_at_t, logR_at_t


def solar_residual_with_zprofile(alpha_mlt, Y_init, Z=0.0188, t_target=4.57e9, max_steps=500):
    """Compute (log_L, log_R) at t_target WITH Z_profile physics active.

    Evolves 1 M☉, computes the Z settling profile from multi-element diffusion,
    reconverges the structure with Z_profile, and returns residuals.

    Reference: Christensen-Dalsgaard et al. (1996); Turcotte et al. (1998).
    """

    # Step 1: Evolve (with He settling active during evolution, Z fixed)
    ages, log_L, log_R, _logTe, X_final, _Y, _Z = evolve_solar(alpha_mlt, Y_init, Z, max_steps=max_steps)

    # Step 2: Interpolate to t_target
    idx = jnp.searchsorted(ages, t_target) - 1
    idx = jnp.clip(idx, 1, max_steps - 2)
    t0, t1, t2 = ages[idx - 1], ages[idx], ages[idx + 1]
    logL_at_t = lagrange_interp_3pt(t0, t1, t2,
                                     log_L[idx - 1], log_L[idx], log_L[idx + 1],
                                     t_target)
    logR_at_t = lagrange_interp_3pt(t0, t1, t2,
                                     log_R[idx - 1], log_R[idx], log_R[idx + 1],
                                     t_target)

    # Step 3: Reconverge structure and compute Z_profile
    M_j = jnp.float64(1.0)
    Z_j = jnp.float64(Z)
    alpha_j = jnp.asarray(alpha_mlt, dtype=jnp.float64)
    t_age_j = jnp.float64(t_target)

    # Get logTe from logL and logR
    logR_val = float(logR_at_t)
    logL_val = float(logL_at_t)
    R_val = Rsun * 10.0**logR_val
    L_val = Lsun * 10.0**logL_val
    Te_val = (L_val / (4.0 * np.pi * float(sigma_sb) * R_val**2))**0.25
    logTe_val = np.log10(Te_val)

    # Reconverge Newton with scalar Z first
    logL_f, logTe_f = newton_solve_xprofile(
        M_j, X_final, Z_j, t_age_j, jnp.float64(logL_val), jnp.float64(logTe_val),
        alpha_j, N_NEWTON_WARM)

    # Get shell data for diffusion calculation
    _, shell_data_final, _ = shoot_xprofile(
        M_j, logL_f, logTe_f, X_final, Z_j, t_age_j, alpha_j)

    # Compute Z_profile from multi-element diffusion
    Y_final = 1.0 - X_final - float(Z)
    Y_final = jnp.maximum(Y_final, 0.01)

    N_DIFF_SUB = 10
    t_target_sec = t_target * SECONDS_PER_YEAR
    dt_diff = t_target_sec / N_DIFF_SUB
    Z_profile = jnp.full(N_COMP, float(Z))
    for _ in range(N_DIFF_SUB):
        _, _, Z_profile = diffuse_composition(
            X_final, Y_final, shell_data_final, jnp.float64(dt_diff),
            M_j, Z_profile=Z_profile, use_exact_structure=True,
            use_4species=True)

    # Step 4: Reconverge Newton WITH Z_profile
    logL_zp, logTe_zp = newton_solve_with_zprofile(
        M_j, X_final, Z_j, t_age_j, logL_f, logTe_f, alpha_j,
        Z_profile, n_iter=8)

    # Step 5: Compute log(R/Rsun) from the reconverged state
    L_zp = 10.0**float(logL_zp) * float(Lsun)
    Te_zp = 10.0**float(logTe_zp)
    R_zp = np.sqrt(L_zp / (4.0 * np.pi * float(sigma_sb))) / Te_zp**2
    logR_zp = np.log10(R_zp / float(Rsun))

    return jnp.float64(float(logL_zp)), jnp.float64(logR_zp)


def solar_residual_henyey(alpha_mlt, Y_init, Z=0.0196, t_target=4.57e9, max_steps=500):
    """Compute (log_L, log_R, X_c) at t_target using the Henyey-driven evolve_star.

    Reference: Christensen-Dalsgaard et al. (1996); Paxton et al. (2011, §4).
    """
    from stellar_jax.evolution import evolve_star  # lazy import to avoid circular dependency

    result = evolve_star(1.0, Z=Z, max_steps=max_steps,
                         alpha_mlt=float(alpha_mlt), t_max=t_target,
                         Y_init=float(Y_init), z_feedback=True)

    # Interpolate to t_target (quadratic Lagrange)
    ages = result['star_age']
    log_L = result['log_L']
    log_R = result['log_R']
    idx = int(jnp.searchsorted(ages, t_target)) - 1
    idx = max(1, min(idx, max_steps - 2))
    t0, t1, t2 = float(ages[idx - 1]), float(ages[idx]), float(ages[idx + 1])

    logL_at_t = lagrange_interp_3pt(t0, t1, t2,
                                     float(log_L[idx - 1]), float(log_L[idx]), float(log_L[idx + 1]),
                                     t_target)
    logR_at_t = lagrange_interp_3pt(t0, t1, t2,
                                     float(log_R[idx - 1]), float(log_R[idx]), float(log_R[idx + 1]),
                                     t_target)

    # X_c at solar age
    center_h1 = result['center_h1']
    Xc_at_t = lagrange_interp_3pt(t0, t1, t2,
                                    float(center_h1[idx - 1]), float(center_h1[idx]), float(center_h1[idx + 1]),
                                    t_target)

    return jnp.float64(logL_at_t), jnp.float64(logR_at_t), float(Xc_at_t)


# ======================================================================
# Solar calibration optimizers (NOT differentiable — Python loops)
# ======================================================================

def solar_calibrate(Z=0.0188, t_target=4.57e9, max_steps=500, tol=1e-7, max_iter=30,
                    verbose=False):
    """Find (α_MLT, Y₀) giving log(L/L☉)=0, log(R/R☉)=0 at t=4.57 Gyr.

    Uses Newton-Raphson with adaptive forward finite-difference Jacobian.

    Reference: Christensen-Dalsgaard et al. (1996), Science 272, 1286.
    """
    def residual_fn(alpha, Y0):
        logL, logR = solar_residual(jnp.float64(alpha), jnp.float64(Y0),
                                     Z=Z, t_target=t_target, max_steps=max_steps)
        return float(logL), float(logR)

    return _newton_raphson_2d(
        residual_fn,
        x0=(2.0, 0.27),
        bounds=((1.2, 3.5), (0.23, 0.32)),
        tol=tol, max_iter=max_iter,
        fd_scale_alpha=10.0, fd_scale_Y=1.0,
        fd_min_alpha=1e-7, fd_max_alpha=0.005,
        fd_min_Y=1e-8, fd_max_Y=0.0005,
        max_step_alpha=0.5, max_step_Y=0.02,
        label=f"Solar Calibration (Z={Z}, t={t_target/1e9:.2f} Gyr, tol={tol})",
        verbose=verbose,
    )


def solar_calibrate_with_zprofile(Z=0.0188, t_target=4.57e9, max_steps=500,
                                  tol=1e-7, max_iter=20, verbose=False):
    """Find (α_MLT, Y₀) giving log(L/L☉)=0, log(R/R☉)=0 WITH Z_profile physics.

    Reference: Christensen-Dalsgaard et al. (1996); Turcotte et al. (1998).
    """
    def residual_fn(alpha, Y0):
        logL, logR = solar_residual_with_zprofile(
            jnp.float64(alpha), jnp.float64(Y0), Z=Z,
            t_target=t_target, max_steps=max_steps)
        return float(logL), float(logR)

    return _newton_raphson_2d(
        residual_fn,
        x0=(ALPHA_SOLAR, Y0_SOLAR),
        bounds=((1.2, 3.5), (0.23, 0.32)),
        tol=tol, max_iter=max_iter,
        fd_scale_alpha=10.0, fd_scale_Y=1.0,
        fd_min_alpha=1e-7, fd_max_alpha=0.005,
        fd_min_Y=1e-8, fd_max_Y=0.0005,
        max_step_alpha=0.3, max_step_Y=0.015,
        label=f"Solar Calibration WITH Z_profile (Z={Z}, t={t_target/1e9:.2f} Gyr)",
        verbose=verbose,
    )


def solar_calibrate_henyey(Z=0.0188, t_target=4.57e9, max_steps=500,
                           tol=1e-7, max_iter=20, verbose=False):
    """Find (α_MLT, Y₀) giving log(L/L☉)=0, log(R/R☉)=0 on the Henyey path.

    Uses the shared _newton_raphson_2d optimizer (DRY with solar_calibrate and
    solar_calibrate_with_zprofile). After convergence, verifies that X_c at
    solar age is physical (~0.34) to detect spurious local minima.

    Reference: Christensen-Dalsgaard et al. (1996); Paxton et al. (2011, §4).
    """
    def residual_fn(alpha, Y0):
        logL, logR, _ = solar_residual_henyey(
            jnp.float64(alpha), jnp.float64(Y0), Z=Z,
            t_target=t_target, max_steps=max_steps)
        return float(logL), float(logR)

    result = _newton_raphson_2d(
        residual_fn,
        x0=(ALPHA_SOLAR, Y0_SOLAR),
        bounds=((1.5, 3.5), (0.20, 0.35)),
        tol=tol, max_iter=max_iter,
        fd_scale_alpha=10.0, fd_scale_Y=1.0,
        fd_min_alpha=5e-4, fd_max_alpha=0.002,
        fd_min_Y=5e-5, fd_max_Y=0.0002,
        max_step_alpha=0.3, max_step_Y=0.015,
        label=f"Solar Calibration on Henyey path (Z={Z}, t={t_target/1e9:.2f} Gyr)",
        verbose=verbose,
    )

    # Post-convergence X_c diagnostic: verify the solution is physical
    _, _, X_c = solar_residual_henyey(
        jnp.float64(result['alpha']), jnp.float64(result['Y0']), Z=Z,
        t_target=t_target, max_steps=max_steps)
    result['X_c'] = float(X_c)
    if result['converged'] and X_c > 0.40:
        logger.warning(f"  X_c={X_c:.4f} > 0.40 — likely a spurious local minimum!")
    elif result['converged'] and verbose:
        logger.info(f"  X_c = {X_c:.4f} (expected ~0.34 for solar)")

    return result
