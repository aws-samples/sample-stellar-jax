"""Adaptive forward pass: driver + step helpers + JIT kernels + ZAMS init.

Non-differentiable forward evolution that escapes lax.scan's fixed-shape
constraint via a Python/numpy outer loop with JIT-compiled inner Newton steps.
This module contains the complete adaptive forward pass implementation:

  - JIT-compiled inner kernels (Henyey step, shell data, composition update)
  - ZAMS model construction + cold-start reconvergence
  - Step helpers (timestep control, solve, composition, accept/reject,
    trajectory recording)
  - Public evolve_star_adaptive() facade

Moved from adaptive_forward.py into evolution/adaptive/ (#1115).

MESA references:
  - star_newton.f90: max_tries default ~100 (n_iter=100 here).
  - timestep.f90:check_change (line 732-766): soft limiters, hard limiters.
  - timestep.f90:check_XH_cntr (line 1740): delta_XH_cntr_limit=0.01.
  - controls.defaults:9704: max_timestep_factor=1.2.
  - evolve.f90:1882-1886: remesh in prepare_for_new_step, before solve
  - Paxton et al. 2011 ApJS 192, 3 §7; Paxton et al. 2013 ApJS 208, 4 §6.
"""
import functools
import logging
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from stellar_jax.config.constants import (
    a_rad, sigma_sb, k_B, m_H,
    Msun, Lsun,
    Y_BBN, DY_DZ,
    SECONDS_PER_YEAR, Q_PER_G,
)
from stellar_jax.config.mesh_defaults import (
    ALPHA_MLT, F_OV,
)
from stellar_jax.config.physics_floors import (
    PGAS_FRAC_FLOOR,
    DELTA_LGL_HARD_LIMIT, DELTA_LGTE_HARD_LIMIT,
)
from stellar_jax.henyey import (
    henyey_solve_from_state_atm,
    extract_shell_data_from_henyey,
)
from stellar_jax.evolution import (
    mix_composition, burn_cno, _mix_z_in_cz,
    diffuse_composition,
)
from stellar_jax.structure import (
    initial_guess, build_model_on_mesh,
    newton_solve_xprofile,
)
from stellar_jax.microphysics.nuclear import epsilon_nuclear, _eps_nuclear_np
from stellar_jax.evolution.adaptive.mesh_numpy import (
    make_unified_mesh, _step_remesh,
)
from stellar_jax.evolution.adaptive.trajectory import (
    TrajectoryStep, Trajectory,
)

logger = logging.getLogger(__name__)


def _writable_copy(jax_arr):
    """Return a writable NumPy copy of a JAX array.

    np.asarray(jax_array) returns a read-only view; this helper makes a
    writable copy so in-place NaN repair can safely mutate it (#1182).
    """
    return np.array(jax_arr)


# =============================================================================
# JIT-compiled inner kernels (Henyey step, shell data, composition update)
# =============================================================================

# =============================================================================
# JIT-compiled inner step (single Henyey solve)
# =============================================================================

@functools.partial(jax.jit, static_argnames=('helm_eos', 'bicubic_opacity'))
def _jit_henyey_step(y_prev, q_mesh, M_star_cgs, X_profile, Z, alpha_mlt,
                     atm_ratio, ln_T_prev, ln_P_prev, inv_dt,
                     N14_profile, comp_mfracs, gradL_composition_term=None,
                     ln_rho_guess=None, helm_eos=False, bicubic_opacity=True):
    """JIT-compiled single Henyey Newton step.

    This is the expensive inner kernel — compiled once per mesh shape,
    then fast (~1-2s per call). The outer python loop calls this each step.

    Parameters match henyey_solve_from_state_atm.
    comp_mfracs: positions where X_profile is defined (unified mesh cell centers).
    gradL_composition_term: Optional (N_c,) Ledoux stabilization term.
        Precomputed from composition profile in the Python outer loop.
        MESA turb_support.f90:273: gradL = grada + gradL_composition_term.

    n_iter=100: matches the production lax.scan step (evolution.py:1793).
    MESA ref: star_newton.f90 max_tries default ~100.
    """
    result = henyey_solve_from_state_atm(
        y_prev, q_mesh, M_star_cgs, X_profile, Z, alpha_mlt,
        None,  # R_phot deprecated
        n_iter=100, tol=1e-4,  # MESA: star_solver.f90:213
        ln_T_prev=ln_T_prev, ln_P_prev=ln_P_prev, inv_dt=inv_dt,
        atm_ratio=atm_ratio, N14_profile=N14_profile,
        comp_mfracs=comp_mfracs,
        bypass_conv_gate=False,
        gradL_composition_term=gradL_composition_term,
        ln_rho_guess=ln_rho_guess,
        helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
    return result


@jax.jit
def _jit_extract_shell_data(y_henyey, q_mesh, M_star_cgs, X_profile, Z, alpha_mlt, comp_mfracs):
    """JIT-compiled shell data extraction (eps_nuc, mixing info, etc.).

    comp_mfracs: (N,) array of mass-coordinate positions where X_profile
    is defined. Must be passed explicitly (not None) for JIT stability.
    """
    return extract_shell_data_from_henyey(
        y_henyey, q_mesh, M_star_cgs, X_profile, Z, alpha_mlt,
        comp_mfracs=comp_mfracs)


# =============================================================================
# JIT-compiled composition update (avoids per-step retracing)
# =============================================================================

@jax.jit
def _jit_composition_update(X_burned, Y_burned, C12, C13, N14, X_orig,
                            Z_profile, shell_data, dt_sec, mass, f_ov,
                            comp_mfracs, gradL_composition_term=None,
                            envelope_mixing_enabled=True):
    """JIT-compiled composition mixing + CNO burn + Z-mixing.

    Wrapping all composition operations in a single JIT avoids retracing
    the internal lax.scan calls on every step (measured: 1994 re-compilations
    without this, ~268s wasted). The shapes are fixed (N_zones=600) so this
    compiles once and is cached for all subsequent steps.

    gradL_composition_term: Optional (N_comp,) Ledoux stabilization on the
    composition grid. When provided, the envelope mixing boundary uses
    Ledoux (consistent with the solver's criterion) rather than
    Schwarzschild. On the RGB this prevents the mixing from extending past
    the mu-barrier at the H-burning shell.

    envelope_mixing_enabled: bool or JAX scalar (default True). When False,
    envelope mixing is suppressed via jnp.where (runtime selection, both
    paths compiled). This is a TRACED value (not static_argnums) so that
    the O2 mutation gate works regardless of JAX's persistent compilation
    cache state — the same compiled graph handles both True and False at
    runtime. The caller passes True in production; the mutation gate
    patches _envelope_zone_mixing directly (mechanism 2, trace-time).
    """
    # Mix X and Y
    X_mixed = mix_composition(
        X_burned, shell_data, M_solar=mass, f_ov=f_ov,
        comp_mfracs_in=comp_mfracs,
        gradL_composition_term=gradL_composition_term,
        envelope_mixing=envelope_mixing_enabled)
    Y_mixed = mix_composition(
        Y_burned, shell_data, M_solar=mass, f_ov=f_ov,
        comp_mfracs_in=comp_mfracs,
        gradL_composition_term=gradL_composition_term,
        envelope_mixing=envelope_mixing_enabled)

    # Burn CNO isotopes
    C12_b, C13_b, N14_b = burn_cno(
        C12, C13, N14, X_orig,
        shell_data, dt_sec,
        comp_mfracs_in=comp_mfracs)
    C12_mixed = mix_composition(
        C12_b, shell_data, M_solar=mass, f_ov=f_ov,
        comp_mfracs_in=comp_mfracs,
        gradL_composition_term=gradL_composition_term,
        envelope_mixing=envelope_mixing_enabled)
    C13_mixed = mix_composition(
        C13_b, shell_data, M_solar=mass, f_ov=f_ov,
        comp_mfracs_in=comp_mfracs,
        gradL_composition_term=gradL_composition_term,
        envelope_mixing=envelope_mixing_enabled)
    N14_mixed = mix_composition(
        N14_b, shell_data, M_solar=mass, f_ov=f_ov,
        comp_mfracs_in=comp_mfracs,
        gradL_composition_term=gradL_composition_term,
        envelope_mixing=envelope_mixing_enabled)

    # Z mixing in CZ
    Z_mixed = _mix_z_in_cz(
        Z_profile, shell_data, M_solar=mass, f_ov=f_ov,
        comp_mfracs_in=comp_mfracs,
        gradL_composition_term=gradL_composition_term,
        envelope_mixing=envelope_mixing_enabled)

    return X_mixed, Y_mixed, C12_mixed, C13_mixed, N14_mixed, Z_mixed



# =============================================================================
# ZAMS model construction + cold-start reconvergence
# =============================================================================

def _derive_surface_observables(y_henyey, atm_ratio, N_s, shooting_fallback,
                                verbose):
    """Derive logL0/logTe0 from reconverged Henyey state; fall back to shooting values.

    Parameters
    ----------
    shooting_fallback : dict — {logL0, logTe0, mass} for NaN fallback.
    """
    mass = shooting_fallback['mass']
    logL0_shooting = shooting_fallback['logL0']
    logTe0_shooting = shooting_fallback['logTe0']
    ell_surf_init = float(y_henyey[N_s - 1, 3])
    L_init_h = ell_surf_init * Lsun
    R_phot_obs_init = atm_ratio * np.exp(float(y_henyey[-1, 0]))
    L_init_safe = float(np.fmax(L_init_h, 1e-30))
    R_phot_safe = float(np.fmax(R_phot_obs_init, 1e-10))
    T_eff_init = (L_init_safe / (4.0 * np.pi * sigma_sb * R_phot_safe**2))**0.25
    logL0 = np.log10(L_init_safe / Lsun)
    logTe0 = np.log10(float(np.fmax(T_eff_init, 1.0)))

    if not np.isfinite(logL0) or not np.isfinite(logTe0):
        if verbose:
            logger.warning("Initial logL0=%.3f or logTe0=%.4f non-finite, "
                           "using shooting model values", logL0, logTe0)
        logL0 = float(np.asarray(logL0_shooting))
        logTe0 = float(np.asarray(logTe0_shooting))
        if not np.isfinite(logL0):
            logL0 = float(4.5 * np.log10(mass) - 0.18)
        if not np.isfinite(logTe0):
            logTe0 = float(3.752 + 0.65 * np.log10(mass))

    return logL0, logTe0


def _compute_initial_dt(y_henyey, X_init, Z, verbose):
    """Compute initial dt from nuclear timescale (MESA timestep.f90: dt_init = t_nuc/50000)."""
    T_center = np.exp(float(y_henyey[0, 2]))
    P_center = np.exp(float(y_henyey[0, 1]))
    P_rad_c = a_rad * T_center**4 / 3.0
    P_gas_c = max(P_center - P_rad_c, PGAS_FRAC_FLOOR * P_center)
    mu_center = 4.0 / (3.0 + 5.0 * X_init)
    rho_center = P_gas_c * mu_center * m_H / (k_B * T_center)
    eps_center = float(epsilon_nuclear(
        jnp.float64(rho_center), jnp.float64(T_center),
        jnp.float64(X_init), jnp.float64(Z)))
    eps_center = max(eps_center, 1e-10)
    t_nuc = X_init * Q_PER_G / eps_center
    dt_sec = t_nuc / 50000.0
    dt_max_init = 10.0e6 * SECONDS_PER_YEAR
    dt_sec = min(dt_sec, dt_max_init)
    if verbose:
        logger.info("Initial dt = %.3e yr (t_nuc/50000; eps_center=%.3e)",
                    dt_sec / SECONDS_PER_YEAR, eps_center)
    return dt_sec, rho_center


def _init_zams_state(stellar_params, mesh_state, init_comp, verbose):
    """Build the ZAMS model and reconverge on the unified mesh.

    Parameters
    ----------
    stellar_params : dict — {mass, M_star_cgs, Z, alpha_mlt}
    mesh_state : dict — {N_zones, q_cells, q_struct}
    init_comp : dict — {X_init, Y_init, X_profile}
    verbose : bool

    Returns a dict with the initial evolution state: y_henyey, atm_ratio,
    logL, logTe, ln_T_prev, ln_P_prev, ln_rho_prev, dt_sec, log_rhoc_prev.
    """
    mass = stellar_params['mass']
    M_star_cgs = stellar_params['M_star_cgs']
    Z = stellar_params['Z']
    alpha_mlt = stellar_params['alpha_mlt']
    N_zones = mesh_state['N_zones']
    q_cells = mesh_state['q_cells']
    q_struct = mesh_state['q_struct']
    X_init = init_comp['X_init']
    X_profile = init_comp['X_profile']
    # Shooting guess → cold-start on N_COMP=200 grid
    log_L_g, log_Te_g = initial_guess(mass)
    X_200 = np.full(200, X_init)
    logL0, logTe0 = newton_solve_xprofile(
        jnp.float64(mass), jnp.array(X_200), jnp.float64(Z),
        jnp.float64(0.0), log_L_g, log_Te_g,
        jnp.float64(alpha_mlt), 15)
    logL0_shooting = logL0
    logTe0_shooting = logTe0

    N_s = N_zones
    q_struct_jax = jnp.array(q_struct)
    model_init = build_model_on_mesh(
        mass, logL0, logTe0, jnp.array(X_200),
        Z, alpha_mlt, N_zones, q_mesh_in=q_struct_jax)

    # Initialize Henyey state from shooting model
    ln_r_init = np.log(np.maximum(np.asarray(model_init['r'][:N_s]), 1.0))
    ln_P_init = np.log(10.0) * np.asarray(model_init['logP'][:N_s])
    ln_T_init = np.log(10.0) * np.asarray(model_init['logT'][:N_s])
    ell_init = np.asarray(model_init['L'] / Lsun)[:N_s]
    y_henyey = np.stack([ln_r_init, ln_P_init, ln_T_init, ell_init], axis=-1)

    R_phot_init = float(model_init['r'][N_s])
    atm_ratio = R_phot_init / np.exp(y_henyey[-1, 0])

    # Reconverge with continuation solver (small XLA graph, reuses per-step JIT)
    if verbose:
        logger.info("Cold-start: reconverging shooting guess with continuation solver (N=600)...")
    X_600_init = np.full(N_zones, X_init)
    q_cells_jax = jnp.array(q_cells)
    y_henyey_shooting = y_henyey.copy()
    atm_ratio_shooting = atm_ratio

    # Deferred HELM activation: start with OPAL-only for MS speed.
    # Eager import fd_electron for JIT safety (avoids UnexpectedTracerError).
    import stellar_jax.microphysics.fd_electron  # noqa: F401
    # Deferred HELM activation: start with OPAL-only for MS speed.
    # helm_eos_active is threaded explicitly to _jit_henyey_step as a static arg.
    # Different values produce different JIT cache entries automatically.
    helm_eos_active = False
    warmup_result = _jit_henyey_step(
        jnp.array(y_henyey), q_struct_jax, jnp.float64(M_star_cgs),
        jnp.array(X_600_init), jnp.float64(Z), jnp.float64(alpha_mlt),
        jnp.float64(atm_ratio),
        jnp.array(y_henyey[:, 2]),
        jnp.array(y_henyey[:, 1]),
        jnp.float64(0.0),
        jnp.array(np.full(N_zones, 0.251 * Z)),
        q_cells_jax,
        None,
        helm_eos=helm_eos_active)
    y_warmup = np.asarray(warmup_result['y'])
    warmup_converged = bool(warmup_result.get('converged', True))
    warmup_finite = bool(np.all(np.isfinite(y_warmup)))
    if verbose:
        logger.info("Converged: %s, finite: %s", warmup_converged, warmup_finite)

    if warmup_converged and warmup_finite:
        y_henyey = y_warmup
        atm_ratio = R_phot_init / np.exp(y_henyey[-1, 0])
    else:
        if verbose:
            logger.warning("Warmup reconvergence failed — using shooting model state")
        y_henyey = y_henyey_shooting
        atm_ratio = atm_ratio_shooting

    # Derive logL0/logTe0 from reconverged state (eliminates startup transient)
    logL0, logTe0 = _derive_surface_observables(
        y_henyey, atm_ratio, N_s,
        {'logL0': logL0_shooting, 'logTe0': logTe0_shooting, 'mass': mass},
        verbose)

    ln_T_prev = y_henyey[:, 2].copy()
    ln_P_prev = y_henyey[:, 1].copy()

    # Warm-start density guess from EOS at warmup state
    _warmup_shell = np.asarray(_jit_extract_shell_data(
        jnp.array(y_henyey), jnp.array(q_struct), jnp.float64(M_star_cgs),
        jnp.array(X_profile), jnp.float64(Z), jnp.float64(alpha_mlt),
        comp_mfracs=jnp.array(q_cells)))
    ln_rho_prev = _warmup_shell[:, 7] * np.log(10.0)

    # Initial dt from nuclear timescale
    dt_sec, rho_center = _compute_initial_dt(y_henyey, X_init, Z, verbose)

    logL = float(logL0)
    logTe = float(logTe0)
    log_rhoc_prev = np.log10(max(rho_center, 1.0))

    assert np.isfinite(logL), f"Initial logL={logL} is non-finite!"
    assert np.isfinite(logTe), f"Initial logTe={logTe} is non-finite!"
    if verbose:
        logger.info("Initial state: logL=%.4f, logTe=%.4f (verified finite)", logL, logTe)

    return {
        'y_henyey': y_henyey, 'atm_ratio': atm_ratio,
        'logL': logL, 'logTe': logTe,
        'ln_T_prev': ln_T_prev, 'ln_P_prev': ln_P_prev,
        'ln_rho_prev': ln_rho_prev, 'dt_sec': dt_sec,
        'log_rhoc_prev': log_rhoc_prev,
    }



# =============================================================================
# Step helpers (timestep, solve, composition, accept/reject, trajectory)
# =============================================================================


def _adaptive_compute_dt_next(dt_sec, deltas, dt_config) -> float:
    """Compute dt for the next step using MESA soft limiters.

    Parameters
    ----------
    dt_sec : float — current timestep
    deltas : dict — {delta_logL, delta_xh, delta_lgT_cntr, delta_lgRho_cntr_est}
    dt_config : dict — {dt_grow, dt_shrink, dt_floor_sec, max_dt_sec,
        delta_lgL_limit, xh_cntr_limit, delta_lgT_cntr_limit,
        delta_lgRho_cntr_limit}

    MESA soft-limiter approach (timestep.f90:check_change, line 732-766):
      dt_next *= min(limit / |delta|) across all active limiters.

    Returns
    -------
    dt_next : float (seconds)
    """
    delta_logL = deltas['delta_logL']
    delta_xh = deltas['delta_xh']
    delta_lgT_cntr = deltas['delta_lgT_cntr']
    delta_lgRho_cntr_est = deltas['delta_lgRho_cntr_est']
    dt_grow = dt_config['dt_grow']
    dt_shrink = dt_config['dt_shrink']
    dt_floor_sec = dt_config['dt_floor_sec']
    max_dt_sec = dt_config['max_dt_sec']
    delta_lgL_limit = dt_config.get('delta_lgL_limit', 0.10)
    xh_cntr_limit = dt_config.get('xh_cntr_limit', 0.01)
    delta_lgT_cntr_limit = dt_config.get('delta_lgT_cntr_limit', 0.01)
    delta_lgRho_cntr_limit = dt_config.get('delta_lgRho_cntr_limit', 0.05)
    dt_limit_ratios = []
    if delta_logL > 1e-30:
        dt_limit_ratios.append(delta_logL / delta_lgL_limit)
    if delta_xh > 1e-30:
        dt_limit_ratios.append(delta_xh / xh_cntr_limit)
    if delta_lgT_cntr > 1e-30:
        dt_limit_ratios.append(delta_lgT_cntr / delta_lgT_cntr_limit)
    if delta_lgRho_cntr_est > 1e-30:
        dt_limit_ratios.append(delta_lgRho_cntr_est / delta_lgRho_cntr_limit)

    if dt_limit_ratios:
        max_ratio = max(dt_limit_ratios)
        factor = 1.0 / max(max_ratio, 1.0 / dt_grow)
        factor = min(factor, dt_grow)
        factor = max(factor, dt_shrink)
    else:
        factor = dt_grow

    dt_next = max(dt_sec * factor, dt_floor_sec)
    dt_next = min(dt_next, max_dt_sec)
    return dt_next

def _step_solve_structure(stellar_params, mesh_state, step_state,
                          helm_eos=False, bicubic_opacity=True):
    """Run the JIT-compiled Henyey Newton step and extract observables.

    Parameters
    ----------
    stellar_params : dict — {M_star_cgs, Z, alpha_mlt}
    mesh_state : dict — {q_struct, q_cells, N_zones, X_profile, N14_profile}
    step_state : dict — {y_henyey, atm_ratio, ln_T_prev, ln_P_prev,
        dt_sec, ln_rho_prev}
    helm_eos : bool — whether to include HELM+Coulomb blend in EOS (default False)
    bicubic_opacity : bool — whether to use bicubic opacity interpolation (default True)

    Returns (y_new, converged, y_new_finite, nonphysical_state,
             logL_new, logTe_new, shell_data, gradL_comp).
    """
    M_star_cgs = stellar_params['M_star_cgs']
    Z = stellar_params['Z']
    alpha_mlt = stellar_params['alpha_mlt']
    q_struct = mesh_state['q_struct']
    q_cells = mesh_state['q_cells']
    N_zones = mesh_state['N_zones']
    X_profile = mesh_state['X_profile']
    N14_profile = mesh_state['N14_profile']
    y_henyey = step_state['y_henyey']
    atm_ratio = step_state['atm_ratio']
    ln_T_prev = step_state['ln_T_prev']
    ln_P_prev = step_state['ln_P_prev']
    dt_sec = step_state['dt_sec']
    ln_rho_prev = step_state['ln_rho_prev']

    inv_dt = 1.0 / max(dt_sec, 1.0) if dt_sec > 0 else 0.0
    N_s = N_zones

    y_prev_jax = jnp.array(y_henyey)
    q_struct_jax = jnp.array(q_struct)
    X_jax = jnp.array(X_profile)
    N14_jax = jnp.array(N14_profile)
    ln_T_prev_jax = jnp.array(ln_T_prev)
    ln_P_prev_jax = jnp.array(ln_P_prev)
    q_cells_jax = jnp.array(q_cells)

    from stellar_jax.solver.residual import _compute_gradL_composition_term
    gradL_comp = _compute_gradL_composition_term(
        y_prev_jax, X_jax, Z, q_struct_jax, M_star_cgs, q_cells_jax)

    henyey_result = _jit_henyey_step(
        y_prev_jax, q_struct_jax, jnp.float64(M_star_cgs),
        X_jax, jnp.float64(Z), jnp.float64(alpha_mlt),
        jnp.float64(atm_ratio),
        ln_T_prev_jax, ln_P_prev_jax, jnp.float64(inv_dt),
        N14_jax, q_cells_jax, gradL_comp,
        ln_rho_guess=jnp.array(ln_rho_prev),
        helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)

    y_new = np.asarray(henyey_result['y'])
    converged = bool(henyey_result.get('converged', True))

    y_new_has_nan = bool(np.any(np.isnan(y_new)))
    y_new_has_inf = bool(np.any(np.isinf(y_new)))
    y_new_finite = not (y_new_has_nan or y_new_has_inf)
    if not y_new_finite:
        converged = False

    # Extract observables
    ell_surf = float(y_new[N_s - 1, 3])
    L_henyey = ell_surf * Lsun
    R_phot_obs = atm_ratio * np.exp(float(y_new[-1, 0]))

    nonphysical_state = False
    if not (np.isfinite(L_henyey) and L_henyey > 0 and
            np.isfinite(R_phot_obs) and R_phot_obs > 0):
        converged = False
        nonphysical_state = True
        logL_new = None
        logTe_new = None
    else:
        T_eff = (L_henyey / (4.0 * np.pi * sigma_sb * R_phot_obs**2))**0.25
        logL_new = np.log10(L_henyey / Lsun)
        logTe_new = np.log10(T_eff)
        if not (np.isfinite(logL_new) and np.isfinite(logTe_new)):
            converged = False
            nonphysical_state = True
            logL_new = None
            logTe_new = None

    # Extract shell data
    shell_data = np.asarray(_jit_extract_shell_data(
        jnp.array(y_new), jnp.array(q_struct), jnp.float64(M_star_cgs),
        jnp.array(X_profile), jnp.float64(Z), jnp.float64(alpha_mlt),
        comp_mfracs=jnp.array(q_cells)))

    return (y_new, converged, y_new_finite, nonphysical_state,
            logL_new, logTe_new, shell_data, gradL_comp)

def _step_composition_update(stellar_params, mesh_state, composition,
                             step_data, config):
    """Burn + mix + diffuse composition on the unified mesh.

    Parameters
    ----------
    stellar_params : dict — {mass, M_star_cgs, f_ov}
    mesh_state : dict — {q_struct, q_cells, N_zones}
    composition : dict — {X, Y, Z, C12, C13, N14}
    step_data : dict — {y_new, shell_data, dt_sec, Z, t_sec}
    config : dict — {diffusion, gradL_comp}

    Returns (X_out, Y_out, Z_mixed, C12_mixed, C13_mixed, N14_mixed).
    """
    mass, M_star_cgs, f_ov = stellar_params['mass'], stellar_params['M_star_cgs'], stellar_params['f_ov']
    q_struct, q_cells = mesh_state['q_struct'], mesh_state['q_cells']
    N_zones = mesh_state['N_zones']
    X_profile, Y_profile, Z_profile = composition['X'], composition['Y'], composition['Z']
    C12_profile, C13_profile = composition['C12'], composition['C13']
    N14_profile = composition['N14']
    y_new, shell_data, dt_sec = step_data['y_new'], step_data['shell_data'], step_data['dt_sec']
    Z_metal = step_data.get('Z', 0.014)
    t_sec_age = step_data.get('t_sec', 0.0)
    diffusion = config['diffusion']
    gradL_comp = config['gradL_comp']

    N_s = N_zones

    # Direct eps_nuc at composition grid points (MESA hydro_chem_eqns.f90:102-115
    # uses dxdt_nuc, NEVER dL/dm). Ideal-gas density for JIT-path consistency.
    # CONSTRAINT: MESA uses EOS density; ideal gas <1% in H-shell.
    lnT_struct = y_new[:, 2]
    lnP_struct = y_new[:, 1]
    q_struct_arr = q_struct[:N_s]
    T_at_cells = np.exp(np.interp(q_cells, q_struct_arr, lnT_struct))
    P_at_cells = np.exp(np.interp(q_cells, q_struct_arr, lnP_struct))
    P_rad_cells = a_rad * T_at_cells**4 / 3.0
    P_gas_cells = np.maximum(P_at_cells - P_rad_cells,
                             PGAS_FRAC_FLOOR * P_at_cells)
    Y_cells = 1.0 - X_profile - Z_metal
    mu_cells = 1.0 / (2.0 * X_profile + 0.75 * Y_cells + 0.5 * Z_metal)
    rho_cells = P_gas_cells * mu_cells * m_H / (k_B * T_at_cells)
    _T_BURN_MIN = 5e5  # K — below this, eps_nuc is physically zero (CF88)
    eps_on_mesh = np.zeros_like(X_profile)
    for i in range(len(X_profile)):
        if T_at_cells[i] > _T_BURN_MIN:
            eps_on_mesh[i] = _eps_nuclear_np(
                float(rho_cells[i]), float(T_at_cells[i]),
                float(X_profile[i]), float(Z_metal),
                float(t_sec_age), X_N14=float(N14_profile[i]))

    # Burn hydrogen
    dX = eps_on_mesh * dt_sec / Q_PER_G
    actual_dX = np.minimum(dX, X_profile)
    X_burned = np.maximum(X_profile - dX, 0.0)
    Y_burned = Y_profile + actual_dX

    # Mix + CNO burn + Z-mixing (single JIT call)
    shell_data_jax = jnp.array(shell_data)
    q_cells_jax = jnp.array(q_cells)

    if gradL_comp is not None:
        gradL_for_mix = jnp.concatenate([gradL_comp, jnp.zeros(1)])
    else:
        gradL_for_mix = None

    comp_result = _jit_composition_update(
        jnp.array(X_burned), jnp.array(Y_burned),
        jnp.array(C12_profile), jnp.array(C13_profile),
        jnp.array(N14_profile), jnp.array(X_profile),
        jnp.array(Z_profile), shell_data_jax,
        jnp.float64(dt_sec), jnp.float64(mass), jnp.float64(f_ov),
        q_cells_jax, gradL_for_mix,
        jnp.bool_(True))  # envelope_mixing: always True in production
    X_mixed = np.asarray(comp_result[0])
    Y_mixed = np.asarray(comp_result[1])
    # Writable copies: the NaN-recovery loop below mutates these in-place;
    # np.asarray returns a read-only view of JAX arrays.
    C12_mixed = _writable_copy(comp_result[2])
    C13_mixed = _writable_copy(comp_result[3])
    N14_mixed = _writable_copy(comp_result[4])
    Z_mixed = _writable_copy(comp_result[5])

    # Diffusion
    if diffusion:
        diff_result = diffuse_composition(
            jnp.array(X_mixed), jnp.array(Y_mixed),
            shell_data_jax, jnp.float64(dt_sec),
            jnp.float64(mass),
            Z_profile=jnp.array(Z_mixed),
            comp_mfracs_in=q_cells_jax)
        X_out = np.asarray(diff_result[0])
        Y_out = np.asarray(diff_result[1])
        Z_mixed = _writable_copy(diff_result[2])  # NaN recovery
    else:
        X_out = X_mixed
        Y_out = Y_mixed

    # NaN defense for CNO species (preserve step's X/Y changes)
    for arr, prev in [(N14_mixed, N14_profile), (C12_mixed, C12_profile),
                      (C13_mixed, C13_profile), (Z_mixed, Z_profile)]:
        nan_mask = ~np.isfinite(arr)
        if np.any(nan_mask):
            arr[nan_mask] = prev[nan_mask]

    return X_out, Y_out, Z_mixed, C12_mixed, C13_mixed, N14_mixed

def _init_timestep_config(max_dt_yr, delta_lgL_limit_override):
    """Build timestep control parameters matching MESA's architecture.

    MESA architecture (timestep.f90:check_change, line 732-766):
      - Soft limiters SIZE dt_next (varcontrol_target, delta_XH_cntr_limit)
      - Hard limiters REJECT the step (only when hard_lim > 0 AND |Δ| > hard_lim)
      - varcontrol_target (default 1e-3) is a SOFT adjuster, NEVER causes retry
      - delta_lgL_hard_limit = -1 (DISABLED in MESA by default)

    References:
      - MESA controls.defaults:11185 (delta_XH_cntr_limit=0.01)
      - MESA timestep.f90:check_XH_cntr (line 1740)
      - MESA controls.defaults:9704 (max_timestep_factor=1.2)
      - MESA timestep.f90:337-351 (max_timestep_factor at high T)
      - MESA timestep.f90:56, controls.defaults:9599 (max_years_for_timestep)
      - MESA controls.defaults:10838 (delta_lgT_cntr_limit)
      - MESA controls.defaults:10865 (delta_lgRho_cntr_limit)

    Returns
    -------
    dict with keys: dt_grow, dt_shrink, dt_floor_sec, max_dt_sec,
        xh_cntr_limit, delta_lgL_limit, delta_lgT_cntr_limit,
        delta_lgRho_cntr_limit
    """
    _max_dt_yr = max_dt_yr if max_dt_yr is not None else 3.0e8
    delta_lgL_limit = (delta_lgL_limit_override
                       if delta_lgL_limit_override is not None else 0.10)
    return {
        'dt_grow': 1.2,
        'dt_shrink': 0.5,
        'dt_floor_sec': 1e4 * SECONDS_PER_YEAR,
        'max_dt_sec': _max_dt_yr * SECONDS_PER_YEAR,
        'xh_cntr_limit': 0.01,
        'delta_lgL_limit': delta_lgL_limit,
        'delta_lgT_cntr_limit': 0.01,
        'delta_lgRho_cntr_limit': 0.05,
    }

def _check_loop_termination(loop_state, step_state, config):
    """Check whether the evolution loop should terminate.

    Parameters
    ----------
    loop_state : dict — {step, accepted_count, peak_logL_reached}
    step_state : dict — {t_sec, ln_T_prev, logL}
    config : dict — {t_max_sec, t_max_yr, verbose}

    Returns (should_break, log_Tc_current).
    """
    step = loop_state['step']
    accepted_count = loop_state['accepted_count']
    peak_logL_reached = loop_state['peak_logL_reached']
    t_sec = step_state['t_sec']
    ln_T_prev = step_state['ln_T_prev']
    logL = step_state['logL']
    t_max_sec = config['t_max_sec']
    t_max_yr = config['t_max_yr']
    verbose = config['verbose']

    # Time limit
    if t_sec >= t_max_sec:
        if verbose:
            logger.info("[step %d] Reached t_max = %.2e yr", step, t_max_yr)
        return True, 0.0

    log_Tc_current = float(ln_T_prev[0]) / np.log(10.0)

    # He ignition halt (logTc=8.2; see evolve_star_adaptive docstring for
    # the threshold justification — our operator-split eps_grav coupling
    # runs ~0.2 dex hotter than MESA at the same logL).
    if log_Tc_current > 8.2:
        if verbose:
            logger.info("[step %d] He ignition halt: log_Tc = %.3f", step, log_Tc_current)
        return True, log_Tc_current

    # Stability halt: logL declining >0.3 dex below peak on the upper RGB
    # indicates solver degeneration (degenerate core + thin H-shell stiffness).
    # Analogous to MESA struct_burn_mix.f90 retry exhaustion.
    if logL > 2.5 and accepted_count > 10:
        if logL < peak_logL_reached - 0.3:
            if verbose:
                logger.info("[step %d] Stability halt: logL=%.3f "
                            "dropped >0.3 dex below peak=%.3f",
                            step, logL, peak_logL_reached)
            return True, log_Tc_current

    return False, log_Tc_current

def _maybe_activate_helm(log_Tc_current, log_rhoc_prev, step, verbose, helm_eos_active):
    """Deferred HELM activation when the star approaches degenerate conditions.

    Returns True when logTc > 7.6 AND logRhoc > 5.3, indicating that the
    HELM+Coulomb EOS blend should be compiled into subsequent Henyey steps.
    The caller passes this as a static arg to _jit_henyey_step, which
    automatically produces a separate JIT cache entry — no jax.clear_caches()
    needed.

    References:
      - eos_lookup density gate: w_rho = sigmoid(20*(logRho-5.7))
      - HELM threshold 7.6 = 0.1 dex below blend midpoint (logT=7.7)
    """
    if log_Tc_current > 7.6 and log_rhoc_prev > 5.3 and not helm_eos_active:
        if verbose:
            logger.info("[step %d] HELM activation: logTc=%.3f > 7.6, "
                        "logRhoc=%.3f > 5.3 "
                        "— next _jit_henyey_step will compile with OPAL+HELM blend",
                        step, log_Tc_current, log_rhoc_prev)
        return True
    return helm_eos_active

def _evaluate_rejection(solve_result, step_state, composition_new):
    """Evaluate whether a step should be rejected.

    Parameters
    ----------
    solve_result : dict — {logL_new, logTe_new, converged, y_new_finite,
        nonphysical_state}
    step_state : dict — {logL, logTe, dt_sec, dt_floor_sec, X_profile}
    composition_new : dict — {X_out, Y_out, Z_mixed, C12_mixed, C13_mixed,
        N14_mixed}

    MESA architecture (timestep.f90:check_change, line 732-766):
      REJECT on: convergence failure, structural hard limits, composition NaN.
      Force-accept at dt_floor (min_timestep_limit) EXCEPT for non-physical
      or non-finite states.

    Returns (reject, delta_logL, delta_xh, dt_at_floor).
    """
    logL_new = solve_result['logL_new']
    logTe_new = solve_result['logTe_new']
    converged = solve_result['converged']
    y_new_finite = solve_result['y_new_finite']
    nonphysical_state = solve_result['nonphysical_state']
    logL, logTe = step_state['logL'], step_state['logTe']
    dt_sec, dt_floor_sec = step_state['dt_sec'], step_state['dt_floor_sec']
    X_profile = step_state['X_profile']
    X_out = composition_new['X_out']
    Y_out = composition_new['Y_out']
    N14_mixed = composition_new['N14_mixed']
    C12_mixed = composition_new['C12_mixed']
    C13_mixed = composition_new['C13_mixed']
    Z_mixed = composition_new['Z_mixed']

    delta_logL = abs(logL_new - logL)
    delta_logTe = abs(logTe_new - logTe)

    structural_reject = (delta_logL > DELTA_LGL_HARD_LIMIT or
                         delta_logTe > DELTA_LGTE_HARD_LIMIT)
    dt_at_floor = (dt_sec <= 1.1 * dt_floor_sec)
    convergence_reject = (not converged) and (not dt_at_floor)

    reject = (convergence_reject or structural_reject) and (not dt_at_floor)

    # Non-finite y is always rejected (NaN state is unrecoverable)
    if not y_new_finite:
        reject = True

    # Non-physical state (L≤0) always rejected — even at dt_floor
    if nonphysical_state:
        reject = True

    # NaN guard on observables (IEEE 754: NaN > limit → False, so the
    # structural check misses NaN)
    if not np.isfinite(logL_new) or not np.isfinite(logTe_new):
        reject = True

    # NaN guard on composition — all species must be finite
    if not reject:
        comp_finite = (np.all(np.isfinite(X_out)) and
                       np.all(np.isfinite(Y_out)) and
                       np.all(np.isfinite(N14_mixed)) and
                       np.all(np.isfinite(C12_mixed)) and
                       np.all(np.isfinite(C13_mixed)) and
                       np.all(np.isfinite(Z_mixed)))
        if not comp_finite:
            reject = True

    # Central abundance hard limit (MESA: delta_XH_cntr_hard_limit)
    X_center_new = X_out[0] if not reject else X_profile[0]
    delta_xh = abs(X_center_new - X_profile[0])
    xh_reject = (delta_xh > 0.05) and (not dt_at_floor)
    reject = reject or xh_reject

    return reject, delta_logL, delta_xh, dt_at_floor

def _record_step(record_ctx, prev_state, new_state):
    """Record a trajectory step (accepted or rejected).

    Parameters
    ----------
    record_ctx : dict — {trajectory, step, reject}
    prev_state : dict — {t_sec, dt_sec, q_struct, X_profile, Y_profile, Z,
        C12_profile, C13_profile, N14_profile, y_henyey, logL, logTe, N_zones}
    new_state : dict — {X_out, Y_out, C12_mixed, C13_mixed, N14_mixed,
        logL_new, logTe_new, y_new, shell_data}
    """
    trajectory = record_ctx['trajectory']
    step = record_ctx['step']
    reject = record_ctx['reject']
    t_sec, dt_sec = prev_state['t_sec'], prev_state['dt_sec']
    q_struct, N_zones = prev_state['q_struct'], prev_state['N_zones']
    X_profile, Y_profile, Z = prev_state['X_profile'], prev_state['Y_profile'], prev_state['Z']
    C12_profile, C13_profile = prev_state['C12_profile'], prev_state['C13_profile']
    N14_profile = prev_state['N14_profile']
    y_henyey, logL, logTe = prev_state['y_henyey'], prev_state['logL'], prev_state['logTe']
    X_out, Y_out = new_state['X_out'], new_state['Y_out']
    C12_mixed, C13_mixed = new_state['C12_mixed'], new_state['C13_mixed']
    N14_mixed = new_state['N14_mixed']
    logL_new, logTe_new = new_state['logL_new'], new_state['logTe_new']
    y_new, shell_data = new_state['y_new'], new_state['shell_data']

    t_yr = t_sec / SECONDS_PER_YEAR
    log_Tc = float(y_new[0, 2]) / np.log(10.0)
    # shell_data is surface-to-center; index -1 = center
    log_rhoc = float(shell_data[-1, 7]) if shell_data.shape[1] > 7 else 0.0

    traj_step = TrajectoryStep(
        step_index=step,
        accepted=not reject,
        t_yr=t_yr,
        dt_yr=dt_sec / SECONDS_PER_YEAR,
        q_mesh=q_struct.copy(),
        X_profile=X_profile.copy() if reject else X_out.copy(),
        Y_profile=Y_profile.copy() if reject else Y_out.copy(),
        Z_val=Z,
        C12_profile=C12_profile.copy() if reject else C12_mixed.copy(),
        C13_profile=C13_profile.copy() if reject else C13_mixed.copy(),
        N14_profile=N14_profile.copy() if reject else N14_mixed.copy(),
        y_henyey=y_new.copy() if not reject else y_henyey.copy(),
        logL=logL_new if not reject else logL,
        logTe=logTe_new if not reject else logTe,
        log_Tc=log_Tc,
        log_rhoc=log_rhoc,
        n_zones=N_zones,
    )
    trajectory.append(traj_step)
    return log_rhoc

def _handle_rejected_step(reject_state, dt_state, config):
    """Adjust dt after a rejected step.

    Parameters
    ----------
    reject_state : dict — {nonphysical_state, dt_at_floor, consecutive_nonphysical, step}
    dt_state : dict — {dt_sec, dt_floor_sec}
    config : dict — {verbose}

    SPECIAL CASE: non-physical state (L≤0) at dt_floor → GROW dt after 3
    consecutive rejections. The eps_grav = (T_new − T_prev)/dt term dominates
    at small dt; growing dt reduces its contribution (inv_dt decreases),
    allowing the solver to find a physical state.

    Returns (dt_sec, consecutive_nonphysical).
    """
    nonphysical_state = reject_state['nonphysical_state']
    dt_at_floor = reject_state['dt_at_floor']
    consecutive_nonphysical = reject_state['consecutive_nonphysical']
    step = reject_state['step']
    dt_sec = dt_state['dt_sec']
    dt_floor_sec = dt_state['dt_floor_sec']
    verbose = config['verbose']
    if nonphysical_state and dt_at_floor:
        consecutive_nonphysical += 1
        if consecutive_nonphysical >= 3:
            dt_sec = dt_sec * 4.0
            if verbose and consecutive_nonphysical == 3:
                logger.warning("[step %d] Non-physical state at dt_floor "
                               "(%d× consecutive) — "
                               "GROWING dt to %.3e yr "
                               "(eps_grav stiffness recovery)",
                               step, consecutive_nonphysical,
                               dt_sec / SECONDS_PER_YEAR)
        else:
            dt_sec = max(dt_sec * 0.5, dt_floor_sec)
    else:
        consecutive_nonphysical = 0
        dt_sec = max(dt_sec * 0.5, dt_floor_sec)
    return dt_sec, consecutive_nonphysical

def _build_evolution_summary(run_stats, stellar_params, composition):
    """Build the result dict from the trajectory (compatible with analysis tools).

    Parameters
    ----------
    run_stats : dict — {trajectory, elapsed, accepted_count, step_count}
    stellar_params : dict — {mass, Z, alpha_mlt}
    composition : dict — {X, Y, Z_profile, C12, C13, N14}
    """
    trajectory = run_stats['trajectory']
    mass = stellar_params['mass']
    Z = stellar_params['Z']
    alpha_mlt = stellar_params['alpha_mlt']
    elapsed = run_stats['elapsed']
    accepted_count = run_stats['accepted_count']
    step_count = run_stats['step_count']
    X_profile = composition['X']
    Y_profile = composition['Y']
    Z_profile = composition['Z_profile']
    C12_profile = composition['C12']
    C13_profile = composition['C13']
    N14_profile = composition['N14']

    summary = trajectory.summary()
    summary['mass'] = mass
    summary['Z'] = Z
    summary['alpha_mlt'] = alpha_mlt
    summary['elapsed_s'] = elapsed
    summary['n_accepted'] = accepted_count
    summary['n_total'] = step_count
    summary['X_profile'] = X_profile
    summary['Y_profile'] = Y_profile
    summary['Z_profile'] = Z_profile
    summary['C12_profile'] = C12_profile
    summary['C13_profile'] = C13_profile
    summary['N14_profile'] = N14_profile
    return summary



# =============================================================================
# Public API: evolve_star_adaptive
# =============================================================================

# lint: scan-body exception — Python-level adaptive loop (10+ named helpers)
def evolve_star_adaptive(mass, Z=0.014, alpha_mlt=None, N_zones=600,
                         max_steps=5000, t_max_yr=None,
                         Y_init=None, f_ov=None, diffusion=True,
                         varcontrol_target=1e-3,
                         max_dt_yr=None,
                         delta_lgL_limit_override=None,
                         helm_eos=False,
                         bicubic_opacity=True,
                         verbose=True):
    """Adaptive forward pass with unified mesh + trajectory recording.

    NON-DIFFERENTIABLE forward evolution that escapes lax.scan's fixed-shape
    constraint. The mesh adapts freely (MESA-style) each step, concentrating
    zones at the H-burning shell to resolve RGB ascent. The gradient replay
    re-traces this trajectory on frozen meshes.

    Parameters
    ----------
    helm_eos : bool — include HELM+Coulomb EOS blend (default False). When
        False, starts OPAL-only; _maybe_activate_helm may flip True mid-loop
        when logTc > 7.6 AND logRhoc > 5.3 (deferred activation, producing
        a separate JIT cache entry via static_argnames).
    bicubic_opacity : bool — use bicubic opacity interpolation (default True).

    Returns (trajectory, summary_dict).
    """
    # Float64 gate — stellar structure is unreachable in float32 (§11-A).
    assert jax.config.jax_enable_x64, (
        "JAX float64 is not enabled. stellar-jax requires 64-bit arithmetic. "
        "Import stellar_jax (which enables it) before any other JAX call, or "
        "set jax.config.update('jax_enable_x64', True) at program start."
    )
    if alpha_mlt is None:
        alpha_mlt = float(ALPHA_MLT)
    if f_ov is None:
        f_ov = float(F_OV)
    if Y_init is None:
        Y_init = float(Y_BBN + DY_DZ * Z)

    X_init = max(1.0 - Y_init - Z, 0.5)
    M_star_cgs = mass * Msun
    t_max_sec = (t_max_yr * SECONDS_PER_YEAR) if t_max_yr is not None else 1e30
    tc = _init_timestep_config(max_dt_yr, delta_lgL_limit_override)

    # --- Initialize unified mesh + composition ---
    q_cells, q_struct = make_unified_mesh(None, N_zones)
    X_profile = np.full(N_zones, X_init)
    Y_profile = np.full(N_zones, Y_init)
    Z_profile = np.full(N_zones, Z)
    # Initial CNO mass fractions from the Grevesse & Sauval (1998) solar mix:
    # C/Z ≈ 0.170, ¹²C/¹³C ≈ 89 (Anders & Grevesse 1989).
    # N14 init = 0.079*Z so that CN_total = C12+C13+N14 = 0.251*Z,
    # matching MESA ZAMS center X_N14 (includes PMS ON-cycle O16→N14).
    C12_profile = np.full(N_zones, 0.170 * Z)
    C13_profile = np.full(N_zones, 0.170 * Z / 89.0)
    N14_profile = np.full(N_zones, 0.079 * Z)

    # --- ZAMS model + cold-start reconvergence ---
    zams = _init_zams_state(
        {'mass': mass, 'M_star_cgs': M_star_cgs, 'Z': Z, 'alpha_mlt': alpha_mlt},
        {'N_zones': N_zones, 'q_cells': q_cells, 'q_struct': q_struct},
        {'X_init': X_init, 'Y_init': Y_init, 'X_profile': X_profile},
        verbose)
    y_henyey = zams['y_henyey']
    atm_ratio = zams['atm_ratio']
    logL = zams['logL']
    logTe = zams['logTe']
    ln_T_prev = zams['ln_T_prev']
    ln_P_prev = zams['ln_P_prev']
    ln_rho_prev = zams['ln_rho_prev']
    dt_sec = zams['dt_sec']
    log_rhoc_prev = zams['log_rhoc_prev']
    peak_logL_reached = logL

    trajectory = Trajectory(mass=mass, Z=Z, alpha_mlt=alpha_mlt)
    step_count = 0
    accepted_count = 0
    consecutive_nonphysical = 0
    t_sec = 0.0
    t_start = time.time()

    # Deferred HELM activation: start with the caller's helm_eos flag.
    # _maybe_activate_helm may flip this True mid-loop when the star
    # approaches degenerate conditions.
    helm_eos_active = helm_eos
    if helm_eos:
        # Eager import: fd_electron must be in sys.modules before the JIT
        # trace. Mirrors the pattern in evolution/_core.py.
        import stellar_jax.microphysics.fd_electron  # noqa: F401

    if verbose:
        logger.info("Starting adaptive evolution: M=%.2f Msun, Z=%s, "
                    "N_zones=%d, max_steps=%d", mass, Z, N_zones, max_steps)

    for step in range(max_steps):
        # --- Termination checks ---
        should_break, log_Tc_current = _check_loop_termination(
            {'step': step, 'accepted_count': accepted_count,
             'peak_logL_reached': peak_logL_reached},
            {'t_sec': t_sec, 'ln_T_prev': ln_T_prev, 'logL': logL},
            {'t_max_sec': t_max_sec, 't_max_yr': t_max_yr, 'verbose': verbose})
        if should_break:
            break

        helm_eos_active = _maybe_activate_helm(log_Tc_current, log_rhoc_prev, step, verbose, helm_eos_active)

        # --- Remesh + solve + composition (delegated to existing helpers) ---
        if accepted_count > 0:
            (q_cells, q_struct, X_profile, Y_profile, Z_profile,
             C12_profile, C13_profile, N14_profile,
             y_henyey, ln_T_prev, ln_P_prev, ln_rho_prev) = _step_remesh(
                {'X': X_profile, 'Y': Y_profile, 'Z': Z_profile,
                 'C12': C12_profile, 'C13': C13_profile, 'N14': N14_profile},
                {'y_henyey': y_henyey, 'ln_T_prev': ln_T_prev,
                 'ln_P_prev': ln_P_prev, 'ln_rho_prev': ln_rho_prev,
                 'Z_metal': Z},
                {'q_cells': q_cells, 'q_struct': q_struct, 'N_zones': N_zones})

        (y_new, converged, y_new_finite, nonphysical_state,
         logL_new, logTe_new, shell_data, gradL_comp) = _step_solve_structure(
            {'M_star_cgs': M_star_cgs, 'Z': Z, 'alpha_mlt': alpha_mlt},
            {'q_struct': q_struct, 'q_cells': q_cells, 'N_zones': N_zones,
             'X_profile': X_profile, 'N14_profile': N14_profile},
            {'y_henyey': y_henyey, 'atm_ratio': atm_ratio,
             'ln_T_prev': ln_T_prev, 'ln_P_prev': ln_P_prev,
             'dt_sec': dt_sec, 'ln_rho_prev': ln_rho_prev},
            helm_eos=helm_eos_active, bicubic_opacity=bicubic_opacity)

        if nonphysical_state:
            logL_new = logL
            logTe_new = logTe

        (X_out, Y_out, Z_mixed, C12_mixed, C13_mixed, N14_mixed
         ) = _step_composition_update(
            {'mass': mass, 'M_star_cgs': M_star_cgs, 'f_ov': f_ov},
            {'q_struct': q_struct, 'q_cells': q_cells, 'N_zones': N_zones},
            {'X': X_profile, 'Y': Y_profile, 'Z': Z_profile,
             'C12': C12_profile, 'C13': C13_profile, 'N14': N14_profile},
            {'y_new': y_new, 'shell_data': shell_data, 'dt_sec': dt_sec,
             'Z': Z, 't_sec': t_sec},
            {'diffusion': diffusion, 'gradL_comp': gradL_comp})

        # --- Accept/reject ---
        reject, delta_logL, delta_xh, dt_at_floor = _evaluate_rejection(
            {'logL_new': logL_new, 'logTe_new': logTe_new,
             'converged': converged, 'y_new_finite': y_new_finite,
             'nonphysical_state': nonphysical_state},
            {'logL': logL, 'logTe': logTe, 'dt_sec': dt_sec,
             'dt_floor_sec': tc['dt_floor_sec'], 'X_profile': X_profile},
            {'X_out': X_out, 'Y_out': Y_out,
             'N14_mixed': N14_mixed, 'C12_mixed': C12_mixed,
             'C13_mixed': C13_mixed, 'Z_mixed': Z_mixed})

        # --- Record trajectory ---
        log_rhoc = _record_step(
            {'trajectory': trajectory, 'step': step, 'reject': reject},
            {'t_sec': t_sec, 'dt_sec': dt_sec, 'q_struct': q_struct,
             'X_profile': X_profile, 'Y_profile': Y_profile, 'Z': Z,
             'C12_profile': C12_profile, 'C13_profile': C13_profile,
             'N14_profile': N14_profile, 'y_henyey': y_henyey,
             'logL': logL, 'logTe': logTe, 'N_zones': N_zones},
            {'X_out': X_out, 'Y_out': Y_out,
             'C12_mixed': C12_mixed, 'C13_mixed': C13_mixed,
             'N14_mixed': N14_mixed, 'logL_new': logL_new,
             'logTe_new': logTe_new, 'y_new': y_new,
             'shell_data': shell_data})

        if reject:
            dt_sec, consecutive_nonphysical = _handle_rejected_step(
                {'nonphysical_state': nonphysical_state, 'dt_at_floor': dt_at_floor,
                 'consecutive_nonphysical': consecutive_nonphysical, 'step': step},
                {'dt_sec': dt_sec, 'dt_floor_sec': tc['dt_floor_sec']},
                {'verbose': verbose})
        else:
            # Accept: update state
            accepted_count += 1
            consecutive_nonphysical = 0
            t_sec += dt_sec
            logL = logL_new
            logTe = logTe_new
            peak_logL_reached = max(peak_logL_reached, logL)

            # Central T/P changes BEFORE updating prev state (for soft limiters)
            delta_lgT_cntr = abs(float(y_new[0, 2]) - float(ln_T_prev[0])) / np.log(10.0)
            delta_lgP_cntr = abs(float(y_new[0, 1]) - float(ln_P_prev[0])) / np.log(10.0)
            ln_T_prev = y_new[:, 2].copy()
            ln_P_prev = y_new[:, 1].copy()
            ln_rho_prev = shell_data[:, 7] * np.log(10.0)
            log_rhoc_prev = log_rhoc
            y_henyey = y_new
            X_profile = X_out
            Y_profile = Y_out
            Z_profile = Z_mixed
            C12_profile = C12_mixed
            C13_profile = C13_mixed
            N14_profile = N14_mixed

            # Defensive NaN halt (should be impossible given guards above)
            if not np.isfinite(logL) or not np.isfinite(logTe):
                logger.error("FATAL: logL=%s or logTe=%s is NaN AFTER "
                             "acceptance at step %d (accepted %d)",
                             logL, logTe, step, accepted_count)
                logger.error("y_new range: [%.6e, %.6e], any NaN=%s",
                             np.min(y_new), np.max(y_new),
                             np.any(np.isnan(y_new)))
                break

            # Adapt dt for next step via MESA-style soft limiters
            delta_lgRho_cntr_est = abs(delta_lgP_cntr - delta_lgT_cntr)
            dt_sec = _adaptive_compute_dt_next(
                dt_sec,
                {'delta_logL': delta_logL, 'delta_xh': delta_xh,
                 'delta_lgT_cntr': delta_lgT_cntr,
                 'delta_lgRho_cntr_est': delta_lgRho_cntr_est},
                tc)

            # Report progress periodically
            if verbose and (accepted_count % 20 == 0 or accepted_count <= 5):
                t_yr = t_sec / SECONDS_PER_YEAR
                elapsed = time.time() - t_start
                shell_mask = (X_profile > 0.05) & (X_profile < 0.65)
                shell_zones = int(np.sum(shell_mask))
                Xc = float(X_profile[0])
                logger.info("step %4d (accepted %4d): "
                            "age=%.4e yr, logL=%.3f, logTe=%.4f, "
                            "dt=%.3e yr, Xc=%.4f, "
                            "shell_zones=%d, "
                            "elapsed=%.1fs",
                            step, accepted_count,
                            t_yr, logL, logTe,
                            dt_sec / SECONDS_PER_YEAR, Xc,
                            shell_zones,
                            elapsed)

        step_count += 1

    # --- Final summary ---
    elapsed = time.time() - t_start
    if verbose:
        logger.info("Evolution complete: %d accepted / %d total steps "
                    "in %.1fs (%.2f s/step)",
                    accepted_count, step_count, elapsed,
                    elapsed / max(accepted_count, 1))
        logger.info("Final: age=%.4e yr, logL=%.3f, logTe=%.4f",
                    t_sec / SECONDS_PER_YEAR, logL, logTe)

    summary = _build_evolution_summary(
        {'trajectory': trajectory, 'elapsed': elapsed,
         'accepted_count': accepted_count, 'step_count': step_count},
        {'mass': mass, 'Z': Z, 'alpha_mlt': alpha_mlt},
        {'X': X_profile, 'Y': Y_profile, 'Z_profile': Z_profile,
         'C12': C12_profile, 'C13': C13_profile, 'N14': N14_profile})

    # No global state cleanup needed — helm_eos_active is a local variable,
    # not a module-global switch. Each call is independent.

    return trajectory, summary


def shell_resolution(trajectory, step_idx=-1):
    """Count zones resolving the H-burning shell at a given step.

    The shell is where X transitions from ~0.7 (envelope) to ~0 (core).
    We count zones where 0.01 < X < 0.65 (the gradient region).

    Returns
    -------
    n_zones_in_shell : int
    shell_width_dq : float — total mass fraction spanned by the shell
    """
    if step_idx < 0:
        step_idx = len(trajectory.steps) + step_idx
    s = trajectory.steps[step_idx]
    X = s.X_profile
    shell_mask = (X > 0.01) & (X < 0.65)
    n_zones = int(np.sum(shell_mask))

    # Width in mass fraction
    q = 0.5 * (s.q_mesh[:-1] + s.q_mesh[1:])  # cell centers from face positions
    if len(q) == len(X):
        shell_q = q[shell_mask]
        width = float(shell_q[-1] - shell_q[0]) if len(shell_q) > 1 else 0.0
    else:
        width = 0.0
    return n_zones, width
