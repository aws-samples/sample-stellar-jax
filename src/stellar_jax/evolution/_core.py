"""Stellar evolution: composition, timestepping, ZAMS, solar calibration, I/O.

Orchestrator module — delegates to:
  evolution/zams.py  — ZAMS init + warmup
  evolution/step.py  — per-step helpers for the lax.scan step function
  evolution/scan.py  — scan orchestration (carry init, windowed scan, result assembly)
  evolution/timestep.py — H211b controller + central limiters
  evolution/config.py — frozen config dataclasses
  evolution/contracts.py — carry index constants + shape documentation

The step_fn closure stays in _evolve_star_jit — it is a thin dispatcher that
unpacks the carry, calls module-level helpers, and builds the output + carry.
The carry is a PLAIN TUPLE (conditional on adaptive_mesh) — NOT a NamedTuple.
Python function calls are invisible to XLA tracing: the compiled graph is
IDENTICAL to the monolithic version. See evolution/__init__.py docstring.
"""
import os
import warnings
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

# Gradient trustworthiness budget (F1): maximum max_steps for which
# Conservative default: full-grid AD-vs-FD validation runs at N=100.
# Post-F18: n500 test proves <1% bias for dlogL/dM at 1.0 M☉, but the full
# grid (all masses × observables) is validated only at N=100.
# See docs/reference/gradient-accuracy.md for the reconciliation.
GRAD_TRUST_STEPS = 100


class GradientTrustWarning(UserWarning):
    """Issued when max_steps exceeds the validated gradient-trust budget."""
    pass



class GradientTrustError(Exception):
    """Raised in strict mode when a gradient is taken above GRAD_TRUST_STEPS.

    Opt-in via ``STELLAR_STRICT_GRADIENT=1`` env var or the
    ``strict_gradient=True`` kwarg on ``evolve_star``.  Only fires when
    the caller is inside a JAX differentiation transform (jax.grad / jacrev /
    jacfwd) — forward-only evaluations at N > GRAD_TRUST_STEPS still run
    with the usual GradientTrustWarning.

    See docs/reference/gradient-accuracy.md §Strict mode.
    """
    pass


def _strict_gradient_enabled(kwarg_value):
    """Resolve whether strict-gradient enforcement is active.

    Precedence: explicit kwarg (True/False) > env var STELLAR_STRICT_GRADIENT.
    The env var is truthy when set to '1', 'true', or 'yes' (case-insensitive).
    Returns False when neither is set (backward-compatible default).
    """
    if kwarg_value is not None:
        return bool(kwarg_value)
    env = os.environ.get("STELLAR_STRICT_GRADIENT", "").strip().lower()
    return env in ("1", "true", "yes")

from stellar_jax.config.constants import (
    G, a_rad, sigma_sb,
    Msun, Lsun, Rsun,
    Y_BBN, DY_DZ, SECONDS_PER_YEAR,
    Q_PER_G,
)
from stellar_jax.config.mesh_defaults import (
    N_HENYEY, N_NEWTON_COLD, N_NEWTON_WARM, N_COMP, COMP_MFRACS,
)
from stellar_jax.config.calibration import ALPHA_SOLAR
from stellar_jax.config.physics_floors import PGAS_FRAC_FLOOR
from stellar_jax.microphysics.eos import eos_lookup
from stellar_jax.microphysics.nuclear import epsilon_nuclear, _eps_nuclear_np, he3_equilibrium
from stellar_jax.microphysics.neutrino import epsilon_neutrino
from stellar_jax.structure import (
    initial_guess, interp_X_at_mass,
    shoot_xprofile, newton_solve_xprofile,
    build_model_on_mesh,
)
from stellar_jax.mesh import initial_lagrangian_mesh as _lagrangian_mass_mesh
from stellar_jax.solver.residual import _compute_gradL_composition_term  #: replay Ledoux term
from stellar_jax.henyey import (henyey_solve_from_state_atm,
                    henyey_init_from_state_atm,
                    extract_shell_data_from_henyey, _henyey_newton)
# mesh/ package: consolidated mesh operations.
from stellar_jax.mesh import adapt_composition_mesh, conservative_remap
from stellar_jax.evolution.timestep import (
    h211b_filter_dt_next, compute_central_limiters, compute_worst_offender_ratio
)
from stellar_jax.evolution.config import (
    SolverNumerics, GradientOptions, PhysicsToggles, TracedConstants,
)
from stellar_jax.evolution.contracts import (
    CARRY_X_PROF, CARRY_Y_PROF, CARRY_Z_PROF, CARRY_C12_PROF,
    CARRY_C13_PROF, CARRY_N14_PROF, CARRY_X3_PROF,
    CARRY_LOGL, CARRY_LOGTE, CARRY_T, CARRY_DT,
    CARRY_PREV_LOGT, CARRY_PREV_LOGP, CARRY_PREV_LOGRHO,
    CARRY_Y_HENYEY, CARRY_COMP_MFRACS,
    CARRY_DT_OLD_ADAPTIVE, CARRY_DT_LIMIT_RATIO_OLD_ADAPTIVE,
    CARRY_DT_OLD_STATIC, CARRY_DT_LIMIT_RATIO_OLD_STATIC,
)

# --- Decomposed helpers (evolution/zams.py, step.py, scan.py) ---
# These module-level functions produce the SAME XLA graph as inline code.
from stellar_jax.evolution.zams import (
    _init_params_and_profiles,
    _init_zams_and_warmup,
    _egrav_warmup,
    _extract_warmup_observables,
)
from stellar_jax.evolution.step import (
    _henyey_step_solve,
    _step_remap_to_static,
    _step_solve_and_eps,
    _step_composition_update,
    _step_compute_eps_grav_frac,
    _step_adaptive_mesh_remap,
    _step_diagnostics,
    _step_timestep_decide,
    _step_dt_limiters,
)
from stellar_jax.evolution.scan import (
    _build_init_carry,
    _run_windowed_scan,
    _assemble_result,
)

# --- Structural hard limits (imported from config.physics_floors) ---
# See config/physics_floors.py for the rationale and MESA references.
from stellar_jax.config.physics_floors import (  # noqa: E402
    DELTA_LGL_HARD_LIMIT, DELTA_LGTE_HARD_LIMIT,
)

# --- Forward eps_grav coupling ---
EPS_GRAV_IN_STRUCTURE = True
DETACH_EPS_GRAV_CARRY = False

# --- Central-abundance timestep limiter (MESA timestep.f90:check_XH_cntr, line 1740) ---
_DELTA_XH_CNTR_LIMIT = 0.01
_DELTA_XH_CNTR_HARD_LIMIT = 0.05
_DELTA_LGT_CNTR_LIMIT = 0.01
_DELTA_LGRHO_CNTR_LIMIT = 0.05


# ==================================================================
# Composition evolution — extracted to composition/ package
# ==================================================================
from stellar_jax.composition.burn import burn_composition, burn_cno, relax_he3
from stellar_jax.composition.mix import mix_composition, _mix_z_in_cz
from stellar_jax.composition.diffuse import (
    diffuse_composition,
    _thoul_burgers_coefficients,
    _thoul_burgers_4species,
)


@jax.jit
def _comp_step_with_dt(state, comp, dt):
    """Single time step with explicit dt.

    Parameters
    ----------
    state : dict — {mass, Z, alpha_mlt, prev_L, prev_T, t_prev, f_ov}
    comp : dict — {X_prof, Y_prof}
    dt : scalar — timestep in seconds
    """
    mass, Z, alpha_mlt = state['mass'], state['Z'], state['alpha_mlt']
    prev_L, prev_T, t_prev = state['prev_L'], state['prev_T'], state['t_prev']
    f_ov = state['f_ov']
    X_prof, Y_prof = comp['X_prof'], comp['Y_prof']

    t_age = t_prev / SECONDS_PER_YEAR
    logL_f, logTe_f = newton_solve_xprofile(mass, X_prof, Z, t_age,
                                             prev_L, prev_T, alpha_mlt, N_NEWTON_WARM)
    _, shell_data, _ = shoot_xprofile(mass, logL_f, logTe_f, X_prof, Z, t_age, alpha_mlt)

    X_burned, _Y_burned = burn_composition(X_prof, Y_prof, shell_data, dt)
    X_new = mix_composition(X_burned, shell_data, M_solar=mass, f_ov=f_ov)

    max_dX = jnp.max(jnp.abs(X_new - X_prof))

    L = 10.0**logL_f * Lsun
    Te = 10.0**logTe_f
    R_star = jnp.sqrt(L / (4.0*jnp.pi*sigma_sb)) / Te**2
    log_R = jnp.log10(R_star / Rsun)
    t_new = t_prev + dt

    return logL_f, logTe_f, t_new, X_new, log_R, max_dX


@jax.jit
def _comp_step_diffusion(state, comp, dt):
    """Single step with burn + mix + diffusion.

    Parameters
    ----------
    state : dict — {mass, Z, alpha_mlt, prev_L, prev_T, t_prev, f_ov}
    comp : dict — {X_prof, Y_prof}
    dt : scalar — timestep in seconds
    """
    mass, Z, alpha_mlt = state['mass'], state['Z'], state['alpha_mlt']
    prev_L, prev_T, t_prev = state['prev_L'], state['prev_T'], state['t_prev']
    f_ov = state['f_ov']
    X_prof, Y_prof = comp['X_prof'], comp['Y_prof']

    t_age = t_prev / SECONDS_PER_YEAR
    logL_f, logTe_f = newton_solve_xprofile(mass, X_prof, Z, t_age,
                                             prev_L, prev_T, alpha_mlt, N_NEWTON_WARM)
    _, shell_data, _ = shoot_xprofile(mass, logL_f, logTe_f, X_prof, Z, t_age, alpha_mlt)

    X_burned, Y_burned = burn_composition(X_prof, Y_prof, shell_data, dt)
    X_mixed = mix_composition(X_burned, shell_data, M_solar=mass, f_ov=f_ov)

    X_diff, Y_diff, _ = diffuse_composition(X_mixed, Y_burned, shell_data, dt, mass, use_exact_structure=True)
    Y_mixed = mix_composition(Y_diff, shell_data, M_solar=mass, f_ov=f_ov)

    max_dX = jnp.max(jnp.abs(X_diff - X_prof))

    L = 10.0**logL_f * Lsun
    Te = 10.0**logTe_f
    R_star = jnp.sqrt(L / (4.0*jnp.pi*sigma_sb)) / Te**2
    log_R = jnp.log10(R_star / Rsun)
    t_new = t_prev + dt

    return logL_f, logTe_f, t_new, X_diff, Y_mixed, log_R, max_dX


@jax.jit
def _comp_estimate_dt(state, X_prof):
    """Estimate initial dt from nuclear timescale.

    Parameters
    ----------
    state : dict — {mass, Z, alpha_mlt, prev_L, prev_T, t_prev}
    X_prof : (N_COMP,) — hydrogen mass fraction profile
    """
    mass, Z, alpha_mlt = state['mass'], state['Z'], state['alpha_mlt']
    prev_L, prev_T, t_prev = state['prev_L'], state['prev_T'], state['t_prev']

    t_age = t_prev / SECONDS_PER_YEAR
    logL_f, logTe_f = newton_solve_xprofile(mass, X_prof, Z, t_age,
                                             prev_L, prev_T, alpha_mlt, N_NEWTON_WARM)
    _, shell_data, _ = shoot_xprofile(mass, logL_f, logTe_f, X_prof, Z, t_age, alpha_mlt)
    eps_center = jnp.maximum(shell_data[-1, 0], 1e-10)
    eps_max = jnp.maximum(jnp.max(shell_data[:, 0]), 1e-10)
    dt_nuc = 0.01 * Q_PER_G / eps_center
    dt_safety = 0.03 * Q_PER_G / eps_max
    dt = jnp.minimum(dt_nuc, dt_safety)
    return logL_f, logTe_f, dt


@jax.jit
def _comp_cold_start(mass, Z, alpha_mlt, X_profile):
    """Cold-start Newton solve for composition evolution."""
    log_L_g, log_Te_g = initial_guess(mass)
    logL0, logTe0 = newton_solve_xprofile(mass, X_profile, Z, jnp.float64(0.0),
                                           log_L_g, log_Te_g, alpha_mlt, N_NEWTON_COLD)
    return logL0, logTe0


# ==================================================================
# ZAMS properties
# ==================================================================

@jax.jit
def _zams_properties_jit(mass, Z, alpha_mlt):
    """JIT-compiled core of zams_properties."""
    mass = jnp.asarray(mass, dtype=jnp.float64)
    Z = jnp.asarray(Z, dtype=jnp.float64)
    alpha_mlt = jnp.asarray(alpha_mlt, dtype=jnp.float64)

    Y = Y_BBN + DY_DZ * Z
    X_init = jnp.maximum(1.0 - Y - Z, 0.5)
    X_profile = jnp.full(N_COMP, X_init)

    log_L_g, log_Te_g = initial_guess(mass)
    logL, logTe = newton_solve_xprofile(mass, X_profile, Z, jnp.float64(0.0),
                                         log_L_g, log_Te_g, alpha_mlt, N_NEWTON_COLD)

    _, shell_data, final_state = shoot_xprofile(mass, logL, logTe, X_profile, Z, jnp.float64(0.0), alpha_mlt)

    P_inner = final_state[0]
    T_inner = final_state[3]
    nabla = shell_data[-1, 8]

    logT_inner = jnp.log10(T_inner)
    P_rad_inner = a_rad * T_inner**4 / 3.0
    logPgas_inner = jnp.log10(jnp.maximum(P_inner - P_rad_inner, PGAS_FRAC_FLOOR * P_inner))
    X_c = X_profile[0]
    rho_inner, _, _, *_ = eos_lookup(logT_inner, logPgas_inner, X_c, Z)

    L_Lsun = 10.0**logL
    Te = 10.0**logTe
    L_cgs = L_Lsun * Lsun
    R_star = jnp.sqrt(L_cgs / (4.0 * jnp.pi * sigma_sb)) / Te**2
    r_inner = 0.005 * R_star

    coeff = (2.0 * jnp.pi / 3.0) * G * r_inner**2

    def _central_iter(_, rho_c_i):
        P_c_i = P_inner + coeff * rho_c_i**2
        T_c_i = T_inner * (P_c_i / P_inner)**nabla
        logT_c_i = jnp.log10(T_c_i)
        P_rad_c_i = a_rad * T_c_i**4 / 3.0
        logPgas_c_i = jnp.log10(jnp.maximum(P_c_i - P_rad_c_i, PGAS_FRAC_FLOOR * P_c_i))
        rho_new, _, _, *_ = eos_lookup(logT_c_i, logPgas_c_i, X_c, Z)
        return rho_new

    rho_c = lax.fori_loop(0, 8, _central_iter, rho_inner)
    P_c = P_inner + coeff * rho_c**2
    T_c = T_inner * (P_c / P_inner)**nabla

    log_Tc = jnp.log10(T_c)
    log_rhoc = jnp.log10(rho_c)
    R_Rsun = R_star / Rsun

    return log_Tc, log_rhoc, L_Lsun, R_Rsun, logL, logTe


def zams_properties(mass, Z=0.014, alpha_mlt=1.9):
    """Extract ZAMS structure properties for validation."""
    log_Tc, log_rhoc, L_Lsun, R_Rsun, logL, logTe = _zams_properties_jit(
        jnp.asarray(float(mass), dtype=jnp.float64),
        jnp.asarray(float(Z), dtype=jnp.float64),
        jnp.asarray(float(alpha_mlt), dtype=jnp.float64),
    )
    return {
        'log_Tc': float(log_Tc),
        'log_rhoc': float(log_rhoc),
        'L': float(L_Lsun),
        'R': float(R_Rsun),
        'log_L': float(logL),
        'log_Teff': float(logTe),
    }


# ==================================================================
# Time-domain evolution
# ==================================================================

def _check_gradient_trust(max_steps, mass, strict_gradient):
    """Validate max_steps against the gradient trust boundary."""
    if max_steps > GRAD_TRUST_STEPS:
        if _strict_gradient_enabled(strict_gradient) and isinstance(mass, jax.core.Tracer):
            raise GradientTrustError(
                f"max_steps={max_steps} exceeds GRAD_TRUST_STEPS={GRAD_TRUST_STEPS} "
                f"and strict gradient enforcement is active. AD gradients are not "
                f"validated above N={GRAD_TRUST_STEPS}. Either reduce max_steps or "
                f"disable strict mode (strict_gradient=False / unset "
                f"STELLAR_STRICT_GRADIENT). See docs/reference/gradient-accuracy.md."
            )
        warnings.warn(
            f"max_steps={max_steps} exceeds GRAD_TRUST_STEPS={GRAD_TRUST_STEPS}. "
            f"AD gradients (jax.grad) may have >5% bias vs finite difference. "
            f"See docs/reference/gradient-accuracy.md.",
            GradientTrustWarning,
            stacklevel=2,
        )


def _coerce_evolve_star_inputs(grad_window, max_steps, Z, helm_eos):
    """Coerce grad_window, bicubic opacity flag, and HELM gate."""
    if grad_window is not None:
        gw = (int(grad_window[0]), int(grad_window[1]))
    else:
        gw = (0, max_steps)
    try:
        bicubic = float(Z) >= 0.015
    except jax.errors.ConcretizationTypeError:
        bicubic = False
    if helm_eos:
        import stellar_jax.microphysics.fd_electron  # noqa: F401
    return gw, bicubic


def evolve_star(mass, Z=0.014, max_steps=500, alpha_mlt=1.9, t_max=None,
                Y_init=None, f_ov=None, diffusion=True,
                varcontrol_target=None, grad_window=None, z_feedback=False,
                fixed_dt=None, freeze_schedule=False, adaptive_mesh=True,
                dt_schedule=None, mesh_schedule=None, comp_schedule=None,
                opacity_factor=1.0, eps_nuc_factor=1.0,
                diffusion_factor=1.0, helm_eos=False, strict_gradient=None):
    """Unified differentiable stellar evolution solver.

    Adaptive timestepping (varcontrol scheme, Paxton et al. 2013 §4):
    -------------------------------------------------------------------
    Each step is accepted/rejected based on a dimensionless measure of
    structural change:

        varcontrol = max(ΔX_max, Δlog_L, Δlog_Teff, Δlog_T_max, Δlog_ρ_max)

    where Δ means |current - previous| over the timestep dt. The next dt is
    scaled by (varcontrol_target / varcontrol), clamped to [0.5, 1.5]×dt.
    A step is *rejected* (retried at dt/2) when varcontrol_reject > 4×target.
    The solver hard-stops at TAMS (MS swap scope); post-TAMS evolution
    (SGB/RGB ascent) is handled by adaptive_forward with Armijo line-search +
    eps_grav coupling.

    Default varcontrol_target = 1e-3 (same as MESA's varcontrol_target for
    main-sequence evolution, Paxton et al. 2013, ApJS 208, 4, §4.1).

    Parameters
    ----------
    mass : float
        Stellar mass in solar masses.
    Z : float
        Initial metallicity (default 0.014).
    max_steps : int
        Maximum number of evolution steps (default 500).
    alpha_mlt : float
        Mixing-length parameter (default 1.9).
    t_max : float or None
        Maximum age in years. None → 1e30.
    Y_init : float or None
        Initial helium mass fraction. None → Y_BBN + dY/dZ * Z.
    f_ov : float or None
        Overshooting parameter. None → F_OV (= 0.016, Herwig 2000).
    diffusion : bool
        Enable element diffusion (default True).
    varcontrol_target : float or None
        Adaptive dt target. None → 1e-3.
    grad_window : tuple (k0, k1) or None
        Gradient window for restricted autodiff.
    z_feedback : bool
        Z-profile feedback (default False).
    fixed_dt : float or None
        Fixed timestep in years (disables adaptive dt).
    freeze_schedule : bool
        Frozen-schedule adjoint (default False).
    adaptive_mesh : bool
        Adaptive composition mesh (default True).
    dt_schedule : array or None
        Prescribed timestep schedule (seconds).
    mesh_schedule : array or None
        Prescribed per-step structure mesh schedule, shape (max_steps, N+1).
        When provided together with dt_schedule, the differentiable replay
        uses both the recorded timesteps AND the recorded meshes from an
        adaptive forward trajectory. The mesh is frozen in the backward pass
        (stop_gradient'd, same GP-1/GP-7 pattern as dt_schedule).
        MESA ref: evolve.f90:1882-1886 — do_mesh() per step.
    comp_schedule : dict or None
        Prescribed per-step composition schedule from an adaptive forward,
        with keys 'X' shape (max_steps, N_HENYEY) and 'Y' shape
        (max_steps, N_HENYEY), plus 'N14', 'C12', 'C13' (same shape)
        and 'comp_mfracs' shape (max_steps, N_HENYEY) — all on the
        recorded 600-zone adaptive mesh (NO 600→200 remap).
        GRADSOLVE-faithful: the replay uses the same composition grid
        as the record, so replay-map = record-map (reproduction guarantee).
        The composition is stop_gradient'd in the backward pass (same
        pattern as dt and mesh).
        CONSTRAINT: MESA's operator split (struct_burn_mix.f90:82-100)
        only splits the burn, not the mix. Our freeze is stronger —
        justified by the fixed-shape lax.scan carry (gradient_policy.py).
    opacity_factor : float
        Multiplicative opacity knob (default 1.0).
    eps_nuc_factor : float
        Multiplicative nuclear rate knob (default 1.0).
    diffusion_factor : float
        Multiplicative diffusion knob (default 1.0).
    helm_eos : bool
        Enable HELM EOS for RGB (default False).
    strict_gradient : bool or None
        Strict gradient enforcement.
    """
    # Float64 gate — stellar structure is unreachable in float32 (§11-A).
    assert jax.config.jax_enable_x64, (
        "JAX float64 is not enabled. stellar-jax requires 64-bit arithmetic. "
        "Import stellar_jax (which enables it) before any other JAX call, or "
        "set jax.config.update('jax_enable_x64', True) at program start."
    )
    _check_gradient_trust(max_steps, mass, strict_gradient)
    gw, bicubic = _coerce_evolve_star_inputs(
        grad_window, max_steps, Z, helm_eos)
    # fixed_dt is a static argument (changes the control flow: with/without
    # accept/reject branch). Convert to a hashable value for JIT caching.
    _fixed_dt_static = float(fixed_dt) if fixed_dt is not None else None
    _replay_schedule = dt_schedule is not None
    _dt_schedule_arr = (jnp.asarray(dt_schedule, dtype=jnp.float64)
                        if _replay_schedule else None)
    _replay_mesh = mesh_schedule is not None
    _mesh_schedule_arr = (jnp.asarray(mesh_schedule, dtype=jnp.float64)
                          if _replay_mesh else None)
    _replay_comp = comp_schedule is not None
    if _replay_comp and not _replay_mesh:
        raise ValueError(
            "comp_schedule requires mesh_schedule: the composition replay "
            "builds a multi-element xs tuple in the scan, which "
            "requires mesh_schedule to be provided.")
    _comp_schedule_X = (jnp.asarray(comp_schedule['X'], dtype=jnp.float64)
                        if _replay_comp else None)
    _comp_schedule_Y = (jnp.asarray(comp_schedule['Y'], dtype=jnp.float64)
                        if _replay_comp else None)
    _comp_schedule_N14 = (jnp.asarray(comp_schedule['N14'], dtype=jnp.float64)
                          if _replay_comp and 'N14' in comp_schedule else None)
    _comp_schedule_C12 = (jnp.asarray(comp_schedule['C12'], dtype=jnp.float64)
                          if _replay_comp and 'C12' in comp_schedule else None)
    _comp_schedule_C13 = (jnp.asarray(comp_schedule['C13'], dtype=jnp.float64)
                          if _replay_comp and 'C13' in comp_schedule else None)
    _comp_schedule_mfracs = (jnp.asarray(comp_schedule['comp_mfracs'], dtype=jnp.float64)
                             if _replay_comp and 'comp_mfracs' in comp_schedule else None)
    _has_idle_guard = (t_max is not None) or _replay_schedule
    return _evolve_star_jit(mass, Z, alpha_mlt, t_max, Y_init, f_ov,
                            varcontrol_target, dt_schedule=_dt_schedule_arr,
                            mesh_schedule=_mesh_schedule_arr,
                            comp_schedule_X=_comp_schedule_X,
                            comp_schedule_Y=_comp_schedule_Y,
                            comp_schedule_N14=_comp_schedule_N14,
                            comp_schedule_C12=_comp_schedule_C12,
                            comp_schedule_C13=_comp_schedule_C13,
                            comp_schedule_mfracs=_comp_schedule_mfracs,
                            opacity_factor=opacity_factor,
                            eps_nuc_factor=eps_nuc_factor,
                            diffusion_factor=diffusion_factor,
                            numerics=SolverNumerics(
                                max_steps=max_steps,
                                fixed_dt=_fixed_dt_static,
                                adaptive_mesh=adaptive_mesh,
                                freeze_schedule=freeze_schedule,
                                replay_schedule=_replay_schedule,
                                replay_mesh=_replay_mesh,
                                replay_comp=_replay_comp,
                                has_idle_guard=_has_idle_guard,
                                xh_cntr_limit=_DELTA_XH_CNTR_LIMIT,
                                xh_cntr_hard_limit=_DELTA_XH_CNTR_HARD_LIMIT,
                                delta_lgT_cntr_limit=_DELTA_LGT_CNTR_LIMIT,
                                delta_lgRho_cntr_limit=_DELTA_LGRHO_CNTR_LIMIT,
                            ),
                            gradient=GradientOptions(
                                grad_window=gw,
                                eps_grav_flag=bool(EPS_GRAV_IN_STRUCTURE),
                                detach_eps_grav_carry=bool(DETACH_EPS_GRAV_CARRY),
                            ),
                            physics=PhysicsToggles(
                                diffusion=diffusion,
                                z_feedback=z_feedback,
                                bicubic_opacity=bicubic,
                                helm_eos=helm_eos,
                            ))


# ==================================================================
# JIT-compiled evolution implementation
# ==================================================================

# lint: scan-body exception — _evolve_star_jit is the JIT boundary wrapping the
# 3-phase windowed lax.scan; splitting would change the XLA trace graph.
@partial(jax.jit, static_argnames=('numerics', 'gradient', 'physics'))
def _evolve_star_jit(mass, Z, alpha_mlt, t_max, Y_init, f_ov,
                     varcontrol_target, dt_schedule=None, mesh_schedule=None,
                     comp_schedule_X=None, comp_schedule_Y=None,
                     comp_schedule_N14=None, comp_schedule_C12=None,
                     comp_schedule_C13=None, comp_schedule_mfracs=None, *,
                     opacity_factor=1.0, eps_nuc_factor=1.0, diffusion_factor=1.0,
                     numerics=SolverNumerics(), gradient=GradientOptions(),
                     physics=PhysicsToggles()):
    """JIT-compiled implementation of evolve_star.

    Traced args are individual positional/keyword args (NOT a dict) to preserve
    the XLA HLO input ordering from main.
    """
    # --- Unpack static config objects into local variables ---
    max_steps = numerics.max_steps
    fixed_dt = numerics.fixed_dt
    adaptive_mesh = numerics.adaptive_mesh
    freeze_schedule = numerics.freeze_schedule
    replay_schedule = numerics.replay_schedule
    replay_mesh = numerics.replay_mesh
    replay_comp = numerics.replay_comp
    has_idle_guard = numerics.has_idle_guard

    grad_window = gradient.grad_window
    _eps_grav_flag = gradient.eps_grav_flag
    _detach_eps_grav_carry = gradient.detach_eps_grav_carry

    _helm_eos = physics.helm_eos
    _bicubic_opacity = physics.bicubic_opacity

    # --- Parameter coercion + composition profiles + ZAMS cold start ---
    _numerics_static = {'fixed_dt': fixed_dt, 'freeze_schedule': freeze_schedule}
    _traced_inputs = {
        'mass': mass, 'Z': Z, 'alpha_mlt': alpha_mlt,
        't_max': t_max, 'Y_init': Y_init, 'f_ov': f_ov,
        'varcontrol_target': varcontrol_target,
        'opacity_factor': opacity_factor,
        'eps_nuc_factor': eps_nuc_factor,
        'diffusion_factor': diffusion_factor,
    }
    p = _init_params_and_profiles(_traced_inputs, _numerics_static)
    mass, Z, alpha_mlt = p['mass'], p['Z'], p['alpha_mlt']
    opacity_factor, eps_nuc_factor = p['opacity_factor'], p['eps_nuc_factor']
    diffusion_factor, f_ov, t_max_sec = p['diffusion_factor'], p['f_ov'], p['t_max_sec']
    X_profile_init, Y_profile_init = p['X_profile_init'], p['Y_profile_init']
    C12_profile_init = p['C12_profile_init']
    C13_profile_init, N14_profile_init = p['C13_profile_init'], p['N14_profile_init']
    X3_profile_init = p['X3_profile_init']
    logL0, logTe0 = p['logL0'], p['logTe0']
    varcontrol_target, reject_threshold = p['varcontrol_target'], p['reject_threshold']
    dt_grow, dt_shrink = p['dt_grow'], p['dt_shrink']
    _DELTA_LGL_HARD, _DELTA_LGTE_HARD = p['_DELTA_LGL_HARD'], p['_DELTA_LGTE_HARD']
    M_star_cgs, q_mesh, N_s = p['M_star_cgs'], p['q_mesh'], p['N_s']
    atm_ratio, y_warmup, dt_init = p['atm_ratio'], p['y_warmup'], p['dt_init']
    logT_init, logP_init, logRho_init = p['logT_init'], p['logP_init'], p['logRho_init']

    # Build traced-constants bundle for step helpers.
    tc = TracedConstants(
        mass=mass, Z=Z, alpha_mlt=alpha_mlt, f_ov=f_ov,
        M_star_cgs=M_star_cgs, atm_ratio=atm_ratio,
        q_mesh=q_mesh, N_s=N_s,
        varcontrol_target=varcontrol_target,
        reject_threshold=reject_threshold,
        t_max_sec=t_max_sec, dt_grow=dt_grow, dt_shrink=dt_shrink,
        delta_lgL_hard=_DELTA_LGL_HARD,
        delta_lgTe_hard=_DELTA_LGTE_HARD,
        opacity_factor=opacity_factor,
        eps_nuc_factor=eps_nuc_factor,
        diffusion_factor=diffusion_factor,
    )

    # y_henyey is a LIVE carry (not frozen from schedule) — per GRADSOLVE.
    # No _replay_y_henyey flag needed.

    _static_flags = {
        'eps_grav_flag': _eps_grav_flag,
        'detach_eps_grav_carry': _detach_eps_grav_carry,
        'replay_schedule': replay_schedule,
        'replay_mesh': replay_mesh,
        'replay_comp': replay_comp,
        'helm_eos': _helm_eos,
        'bicubic_opacity': _bicubic_opacity,
    }
    _henyey_step = partial(_henyey_step_solve, static_flags=_static_flags)

    # lint: scan-body exception — step_fn is the lax.scan body; splitting
    # would change the carry shape and XLA graph. Already delegates to
    # module-level helpers; only carry unpack/pack and accept/reject inline.
    def step_fn(carry, xs_step):
        """Per-step evolution: Henyey solve → composition → timestep control."""
        # --- Unpack carry (18- or 19-element tuple; q_mesh_prev is the tail) ---
        if adaptive_mesh:
            (X_prof, Y_prof, Z_prof, C12_prof, C13_prof, N14_prof, X3_prof,
             logL, logTe, t, dt, prev_logT, prev_logP, prev_logRho,
             y_henyey_prev, comp_mfracs_carry,
             dt_old, dt_limit_ratio_old, q_mesh_prev) = carry
        else:
            (X_prof, Y_prof, Z_prof, C12_prof, C13_prof, N14_prof, X3_prof,
             logL, logTe, t, dt, prev_logT, prev_logP, prev_logRho,
             y_henyey_prev,
             dt_old, dt_limit_ratio_old, q_mesh_prev) = carry

        # --- Unpack xs_step: scalar dt (replay_schedule only) or
        #     tuple (dt, q_mesh) when replay_mesh=True, or
        #     tuple (dt, q_mesh, X, Y, N14, C12, C13, comp_mfracs)
        #     when replay_comp=True (8-tuple, 600-zone compositions) ---
        # GRADSOLVE (arXiv:2609.02876 §3.1): compositions are on the
        # recorded 600-zone grid (no 600→200 remap).  comp_mfracs are
        # the 600-zone cell centers so interp_X_at_mass uses the right grid.
        q_mesh_step = None
        X_comp_step = None
        Y_comp_step = None
        N14_comp_step = None
        C12_comp_step = None
        C13_comp_step = None
        comp_mfracs_step = None
        if replay_comp:
            # xs_step is an 8-tuple:
            #   (dt, q_mesh, X, Y, N14, C12, C13, comp_mfracs)
            # All 600-zone: the recorded adaptive-mesh compositions.
            dt = xs_step[0]
            q_mesh_step = jax.lax.stop_gradient(xs_step[1])  # GP-11: MESH_SCHEDULE
            # Frozen 600-zone composition from the adaptive forward.
            # CONSTRAINT (not a MESA match): MESA only splits burn
            # (struct_burn_mix.f90:82-100); we freeze both burn and mix.
            X_comp_step = jax.lax.stop_gradient(xs_step[2])  # GP-12: COMP_SCHEDULE
            Y_comp_step = jax.lax.stop_gradient(xs_step[3])  # GP-12: COMP_SCHEDULE
            N14_comp_step = jax.lax.stop_gradient(xs_step[4])  # GP-12: COMP_SCHEDULE
            C12_comp_step = jax.lax.stop_gradient(xs_step[5])  # GP-12: COMP_SCHEDULE
            C13_comp_step = jax.lax.stop_gradient(xs_step[6])  # GP-12: COMP_SCHEDULE
            comp_mfracs_step = jax.lax.stop_gradient(xs_step[7])  # GP-12: COMP_SCHEDULE
        elif replay_mesh:
            # xs_step is a tuple: (dt_scalar, q_mesh_array)
            dt = xs_step[0]
            # GP-11: MESH_SCHEDULE — stop_gradient at the SOURCE so every
            # downstream consumer (_henyey_step_solve AND _step_solve_and_eps)
            # sees a frozen mesh. The mesh is an algorithmic decision recorded
            # from the adaptive forward; its derivative is not meaningful.
            # MESA ref: evolve.f90:1882-1886 (do_mesh per step).
            q_mesh_step = jax.lax.stop_gradient(xs_step[1])  # GP-11: MESH_SCHEDULE
        elif replay_schedule:
            dt = xs_step

        t_age = t / SECONDS_PER_YEAR

        # --- 1. Remap composition to static grid ---
        _comp_carry = {'X': X_prof, 'Y': Y_prof, 'Z': Z_prof,
                       'C12': C12_prof, 'C13': C13_prof, 'N14': N14_prof, 'X3': X3_prof}
        X_s, Y_s, Z_s, C12_s, C13_s, N14_s, X3_s = _step_remap_to_static(
            _comp_carry, comp_mfracs_carry if adaptive_mesh else None, Z, adaptive_mesh)

        # --- 1b. Composition for the Henyey solve ---
        # GRADSOLVE-faithful (arXiv:2609.02876 §3.1): the replay must use
        # the SAME one-step map as the record.  The recorded trajectory's
        # compositions live on the 600-zone adaptive mesh.  When
        # replay_comp=True, pass these 600-zone compositions DIRECTLY to
        # the Henyey solve (with the matching comp_mfracs so interp_X_at_mass
        # uses the correct grid).  This drops the 600→200→interpolate-to-600
        # round-trip that lost H-shell resolution and made replay ≠ record.
        #
        # ALSO override the 200-zone carry compositions (X_s/Y_s/...) with
        # values interpolated from the 600-zone schedule.  This keeps the
        # carry reasonable (preventing drift in the composition update,
        # which feeds back as the carry X_prof for the next step).  The
        # composition update is irrelevant in replay mode (the carry gets
        # overridden next step), but must stay numerically sane.
        X_solve = X_s      # 200-zone default
        N14_solve = N14_s
        X3_solve = X3_s
        comp_mfracs_solve = None  # None → COMP_MFRACS (200) default
        if replay_comp and X_comp_step is not None:
            X_solve = X_comp_step              # 600-zone from schedule
            N14_solve = N14_comp_step          # 600-zone from schedule
            comp_mfracs_solve = comp_mfracs_step  # 600-zone cell centers
            # X3 is not in the trajectory record — interpolate from the
            # 200-zone carry grid (COMP_MFRACS) to the 600-zone adaptive
            # grid so all species share the same comp_mfracs in the
            # Henyey residual (interp_X_at_mass requires matching shapes).
            X3_solve = jnp.interp(comp_mfracs_step,
                                  jnp.array(COMP_MFRACS), X3_s)
            # Override carry compositions (200-zone) from the 600-zone
            # schedule by interpolating to the static COMP_MFRACS grid.
            # This prevents carry drift in the composition update.
            # JANC-style bounds-clip (arXiv:2504.13750, chemical.py):
            # clip(ΔY, -Y, 1-Y) ≡ clip result to [0, 1] for mass fracs.
            _static_mfracs = jnp.array(COMP_MFRACS)
            X_s = jnp.clip(jnp.interp(_static_mfracs, comp_mfracs_step, X_comp_step), 0.0, 1.0)
            Y_s = jnp.clip(jnp.interp(_static_mfracs, comp_mfracs_step, Y_comp_step), 0.0, 1.0)
            N14_s = jnp.clip(jnp.interp(_static_mfracs, comp_mfracs_step, N14_comp_step), 0.0, 1.0)
            C12_s = jnp.clip(jnp.interp(_static_mfracs, comp_mfracs_step, C12_comp_step), 0.0, 1.0)
            C13_s = jnp.clip(jnp.interp(_static_mfracs, comp_mfracs_step, C13_comp_step), 0.0, 1.0)

        # --- 1c. y_henyey carry is LIVE (not frozen) ---
        # The y_henyey carry is kept LIVE per GRADSOLVE (arXiv:2609.02876):
        # freeze only the step-size schedule (dt + mesh), keep the STATE
        # carry live = exact discrete adjoint.  The energy-history terms
        # ln_T_prev / ln_P_prev (step.py:101-102) are read directly from
        # the y_henyey carry columns; the IFT backward differentiates
        # through them.  Freezing y_henyey_prev (the prior code) severed
        # this channel, producing ~97% AD-vs-FD error.
        # prev_logT / prev_logP stay from the carry (positions 11/12).

        # --- 1d. Differentiable mesh remap of the Henyey guess ---
        # ROOT CAUSE of the subgiant replay collapse: the replay fed the
        # carried y_henyey_prev (living on the PREVIOUS step's mesh q_{k-1})
        # as the Newton guess + energy history onto the CURRENT frozen mesh
        # q_k WITHOUT remapping.  The recorded adaptive forward remaps the
        # Henyey state onto every new mesh before each solve
        # (remap_henyey_state_numpy, mesh_numpy.py:480-508).  Post-TAMS the
        # mesh re-concentrates → a stale-mesh guess → Newton diverges →
        # ell_surf≤0 → L-floor (logL=-63.58) + Teff NaN.
        #
        # Fix: reproduce the record's one-step map — remap ALL 4 columns of
        # y_henyey_prev from q_{k-1} onto q_k with a DIFFERENTIABLE linear
        # interpolation BEFORE the solve, so both the guess AND the derived
        # ln_T_prev/ln_P_prev (step.py:101-102) live on the correct mesh.
        # jnp.interp is differentiable w.r.t. the y VALUES (the live-carry
        # channel that the IFT backward flows through); the x-grids are
        # stop_gradient'd because the mesh is a frozen algorithmic decision
        # (GP-11: MESH_SCHEDULE) — intended and correct.
        #
        # ONLY on the replay path (q_mesh_step is not None).  On the normal
        # adaptive/MS run (q_mesh_step is None) the guess is passed UNCHANGED
        # so the non-replay XLA graph is byte-identical to before.
        y_henyey_solve_in = y_henyey_prev
        if q_mesh_step is not None:
            N_s_remap = y_henyey_prev.shape[0]
            q_old_iface = jax.lax.stop_gradient(q_mesh_prev)[:N_s_remap]  # GP-11: MESH frozen
            q_new_iface = jax.lax.stop_gradient(q_mesh_step)[:N_s_remap]  # GP-11: MESH frozen
            _remapped = jax.vmap(
                lambda col: jnp.interp(q_new_iface, q_old_iface, col),
                in_axes=1, out_axes=1)(y_henyey_prev)
            # Mirror the recorded forward's one-step map EXACTLY (GRADSOLVE,
            # arXiv:2609.02876 — the exact discrete adjoint is of the RECORDED
            # steps). The record remaps the Henyey state ONLY when the mesh
            # actually moved (mesh_numpy.py: mesh_changed = max|Δq_cells| > 1e-8;
            # else it returns y_henyey / ln_T_prev / ln_P_prev UNCHANGED).
            # Applying the differentiable remap on EVERY step is value-≈-identity
            # where the mesh is static (AC-R still passes) but injects ~150
            # near-identity, 1/Δq-ill-conditioned interp-transposes into the
            # eps_grav backward channel (ln_T_prev/ln_P_prev = y[:,2]/[:,1]);
            # composed over N≈154 they explode the adjoint → wrong-sign, ~100×
            # ∂σ²/∂M at the subgiant (AD=+1.6e3 vs FD=−1.5e1). jnp.where routes
            # the gradient to the raw carry (identity) on static-mesh steps and
            # through the legitimate remap only where the mesh moved — matching
            # the record's discrete adjoint AND preserving the live eps_grav
            # channel (sign-correct at N=50/105; -P1 exact per-step).
            _mesh_moved = jax.lax.stop_gradient(  # GP-11: MESH frozen (algorithmic)
                jnp.max(jnp.abs(q_new_iface - q_old_iface)) > 1e-8)
            y_henyey_solve_in = jnp.where(_mesh_moved, _remapped, y_henyey_prev)

        # --- 1e. Ledoux composition-gradient term for the replay solve ---
        # The recorded forward passes gradL_composition_term into the Henyey
        # solve (forward.py:491-493); the replay omitted it.  Compute it from
        # the SAME inputs as the record: the (remapped) Henyey state, the
        # 600-zone composition, the frozen mesh, and the 600-zone cell
        # centers.  None on the non-replay path (unchanged graph).
        gradL_comp = None
        if replay_comp and X_comp_step is not None:
            gradL_comp = _compute_gradL_composition_term(
                y_henyey_solve_in, X_solve, Z,
                jax.lax.stop_gradient(q_mesh_step),  # GP-11: MESH frozen
                M_star_cgs, comp_mfracs_solve)

        # --- 2. Henyey solve + surface observables + eps ---
        _comp_solve = {'X': X_solve, 'N14': N14_solve, 'X3': X3_solve}
        _step_in = {'prev_logRho': prev_logRho, 'dt': dt, 't_age': t_age,
                    'helm_eos': _helm_eos, 'q_mesh_override': q_mesh_step,
                    'comp_mfracs': comp_mfracs_solve,
                    'gradL_composition_term': gradL_comp}
        y_henyey_out, shell_data, henyey_converged, L_henyey, logL_new, \
            logTe_new, shell_data_sg, eps_comp = _step_solve_and_eps(
                y_henyey_solve_in, _comp_solve, _step_in, tc, _henyey_step)

        # --- 3. Composition update (on 200-zone carry grid, irrelevant in replay) ---
        _comp_s = {'X': X_s, 'Y': Y_s, 'Z': Z_s,
                   'C12': C12_s, 'C13': C13_s, 'N14': N14_s, 'X3': X3_s}
        _comp_step_data = {'shell_data_sg': shell_data_sg, 'eps_comp': eps_comp, 'dt': dt}
        X_diff, Y_mixed, Z_mixed, C12_m, C13_m, N14_m, X3_m = \
            _step_composition_update(
                _comp_s, _comp_step_data, tc, physics)

        # Derived surface quantities.
        L = 10.0**logL_new * Lsun
        Te = 10.0**logTe_new
        R_star = jnp.sqrt(L / (4.0 * jnp.pi * sigma_sb)) / Te**2
        log_R = jnp.log10(R_star / Rsun)

        # --- 4. Diagnostics ---
        _diag_state = {
            'X_diff': X_diff, 'X_s': X_s,
            'logL_new': logL_new, 'logTe_new': logTe_new,
            'logL': logL, 'logTe': logTe,
            'shell_data': shell_data, 'dt': dt,
            'L_h': L_henyey, 'y_out': y_henyey_out,
        }
        _prev_struct = {
            'prev_logT': prev_logT, 'prev_logP': prev_logP,
            'prev_logRho': prev_logRho,
        }
        (cur_logT, cur_logP, cur_logRho, eps_grav_frac, log_Tc, log_rhoc,
         max_dX, delta_logL, delta_logTe, varcontrol, varcontrol_reject
         ) = _step_diagnostics(
            _diag_state, _prev_struct, tc, _eps_grav_flag)

        # --- 5. Accept/reject + H211b timestep controller ---
        _dt_state = {
            'varcontrol': varcontrol, 'varcontrol_reject': varcontrol_reject,
            'converged': henyey_converged, 'dt': dt,
            'delta_logL': delta_logL, 'delta_logTe': delta_logTe,
            'cur_logT': cur_logT, 'cur_logRho': cur_logRho,
            'prev_logT': prev_logT, 'prev_logRho': prev_logRho,
            'X_diff': X_diff, 'X_s': X_s,
            'dt_old': dt_old, 'dt_limit_ratio_old': dt_limit_ratio_old,
        }
        reject, dt_at_floor, dt_next_accept, worst_ratio = _step_timestep_decide(
            _dt_state, tc, numerics)

        # --- 6. State selection (accept/reject gate) ---
        X_out = jnp.where(reject, X_prof, X_diff)
        Y_out = jnp.where(reject, Y_prof, Y_mixed)
        Z_out = jnp.where(reject, Z_prof, Z_mixed)
        C12_out = jnp.where(reject, C12_prof, C12_m)
        C13_out = jnp.where(reject, C13_prof, C13_m)
        N14_out = jnp.where(reject, N14_prof, N14_m)
        X3_out = jnp.where(reject, X3_prof, X3_m)
        logL_out = jnp.where(reject, logL, logL_new)
        logTe_out = jnp.where(reject, logTe, logTe_new)
        t_new = jnp.where(reject, t, t + dt)
        dt_next = jnp.where(reject, dt * 0.5, dt_next_accept)
        logT_out = jnp.where(reject, prev_logT, cur_logT)
        logP_out = jnp.where(reject, prev_logP, cur_logP)
        logRho_out = jnp.where(reject, prev_logRho, cur_logRho)
        log_R_out = jnp.where(reject, jnp.float64(0.0), log_R)
        egf_out = jnp.where(reject, jnp.float64(0.0), eps_grav_frac)

        # --- 7. dt limiters ---
        _limiter_state = {
            'reject': reject, 'dt_at_floor': dt_at_floor, 'dt_next': dt_next,
            'X_out': X_out, 'X_s': X_s, 'Y_out': Y_out, 'Y_s': Y_s,
            'Z_out': Z_out, 'Z_s': Z_s,
            'C12_out': C12_out, 'C12_s': C12_s,
            'C13_out': C13_out, 'C13_s': C13_s,
            'N14_out': N14_out, 'N14_s': N14_s,
            'X3_out': X3_out, 'X3_s': X3_s,
            'logL_out': logL_out, 'logL': logL,
            'logTe_out': logTe_out, 'logTe': logTe,
            't_new': t_new, 't': t,
            'logT_out': logT_out, 'logP_out': logP_out, 'logRho_out': logRho_out,
            'log_R_out': log_R_out, 'egf_out': egf_out,
            'log_Tc': log_Tc, 'log_rhoc': log_rhoc,
            'cur_logT': cur_logT, 'cur_logRho': cur_logRho,
            'dt': dt, 't_max_sec': t_max_sec, 'Z': Z,
        }
        _prev_state = {
            'prev_logT': prev_logT, 'prev_logP': prev_logP, 'prev_logRho': prev_logRho,
        }
        (reject, dt_next, X_out, Y_out, Z_out, C12_out, C13_out, N14_out,
         X3_out,
         logL_out, logTe_out, t_new, logT_out, logP_out, logRho_out,
         log_R_out, egf_out, log_Tc_out, log_rhoc_out, xh_ratio_out
         ) = _step_dt_limiters(
            shell_data, _limiter_state, _prev_state, numerics,
            xs_step[0] if (replay_comp or replay_mesh) else xs_step)

        X_center = X_out[0]
        y_henyey_carry = jnp.where(reject, y_henyey_prev, y_henyey_out)

        # --- 8. Idle-step gradient guard ---
        if has_idle_guard:
            idle_step = (dt <= 0.0)
            y_henyey_carry_safe = jnp.where(
                idle_step,
                jax.lax.stop_gradient(y_henyey_carry),  # GP-6: IDLE_GUARD
                y_henyey_carry)
            y_henyey_carry = jnp.where(idle_step, y_henyey_prev, y_henyey_carry_safe)
            logL_safe = jnp.where(
                idle_step, jax.lax.stop_gradient(logL_out), logL_out)  # GP-6: IDLE_GUARD
            logL_out = jnp.where(idle_step, logL, logL_safe)
            logTe_safe = jnp.where(
                idle_step, jax.lax.stop_gradient(logTe_out), logTe_out)  # GP-6: IDLE_GUARD
            logTe_out = jnp.where(idle_step, logTe, logTe_safe)

        # --- 9. Build outputs + carry ---
        outputs = jnp.array([t_new, logL_out, logTe_out, log_R_out, X_center,
                             egf_out, log_Tc_out, log_rhoc_out, xh_ratio_out])

        dt_old_out = jax.lax.stop_gradient(jnp.where(reject, dt_old, dt))  # GP-1: TIMESTEP
        worst_ratio_out = jax.lax.stop_gradient(jnp.where(reject, dt_limit_ratio_old, worst_ratio))  # GP-1: TIMESTEP

        # --- q_mesh_prev carry-out ---
        # On the REPLAY path the accepted y_henyey_carry lives on THIS step's
        # frozen mesh q_mesh_step, so the next step must remap FROM q_mesh_step.
        # On REJECT (and idle) the carry falls back to y_henyey_prev, which
        # still lives on the incoming q_mesh_prev → keep q_mesh_prev so the
        # next step remaps the fallback state from the mesh it actually sits on.
        # Mesh is a frozen algorithmic decision → stop_gradient (GP-11).
        # On the NON-replay path (q_mesh_step is None) keep q_mesh_prev as an
        # inert constant carry so the graph is unchanged.
        if q_mesh_step is not None:
            _keep_prev_mesh = reject
            if has_idle_guard:
                _keep_prev_mesh = reject | (dt <= 0.0)
            q_mesh_prev_out = jax.lax.stop_gradient(  # GP-11: MESH_SCHEDULE
                jnp.where(_keep_prev_mesh, q_mesh_prev, q_mesh_step))
        else:
            q_mesh_prev_out = q_mesh_prev

        if adaptive_mesh:
            _comp_new = {'X': X_diff, 'Y': Y_mixed, 'Z': Z_mixed,
                         'C12': C12_m, 'C13': C13_m, 'N14': N14_m, 'X3': X3_out}
            _comp_prev = {'X': X_prof, 'Y': Y_prof, 'Z': Z_prof,
                          'C12': C12_prof, 'C13': C13_prof, 'N14': N14_prof}
            X_final, Y_final, Z_final, C12_final, C13_final, N14_final, X3_final, comp_mfracs_out = \
                _step_adaptive_mesh_remap(
                    _comp_new, _comp_prev,
                    reject, comp_mfracs_carry, Z)
            return (X_final, Y_final, Z_final,
                    C12_final, C13_final, N14_final, X3_final,
                    logL_out, logTe_out, t_new, dt_next,
                    logT_out, logP_out, logRho_out,
                    y_henyey_carry, comp_mfracs_out,
                    dt_old_out, worst_ratio_out, q_mesh_prev_out), outputs
        else:
            X_final = jnp.where(reject, X_prof, X_diff)
            Y_final = jnp.where(reject, Y_prof, Y_mixed)
            Z_final = jnp.where(reject, Z_prof, Z_mixed)
            C12_final = jnp.where(reject, C12_prof, C12_m)
            C13_final = jnp.where(reject, C13_prof, C13_m)
            N14_final = jnp.where(reject, N14_prof, N14_m)
            return (X_final, Y_final, Z_final,
                    C12_final, C13_final, N14_final, X3_out,
                    logL_out, logTe_out, t_new, dt_next,
                    logT_out, logP_out, logRho_out,
                    y_henyey_carry,
                    dt_old_out, worst_ratio_out, q_mesh_prev_out), outputs

    # --- Build initial carry and run the 3-phase windowed scan ---
    _init_profiles = {
        'X': X_profile_init, 'Y': Y_profile_init, 'Z': Z,
        'C12': C12_profile_init, 'C13': C13_profile_init,
        'N14': N14_profile_init, 'X3': X3_profile_init,
        'logL0': logL0, 'logTe0': logTe0, 'dt_init': dt_init,
        'logT_init': logT_init, 'logP_init': logP_init,
        'logRho_init': logRho_init,
    }
    init_carry = _build_init_carry(adaptive_mesh, _init_profiles, y_warmup, q_mesh)

    k0, k1 = grad_window
    _scan_config = {
        'max_steps': max_steps, 'k0': k0, 'k1': k1,
        'replay_schedule': replay_schedule, 'dt_schedule': dt_schedule,
        'replay_mesh': replay_mesh, 'mesh_schedule': mesh_schedule,
        'replay_comp': replay_comp,
        'comp_schedule_X': comp_schedule_X,
        'comp_schedule_Y': comp_schedule_Y,
        'comp_schedule_N14': comp_schedule_N14,
        'comp_schedule_C12': comp_schedule_C12,
        'comp_schedule_C13': comp_schedule_C13,
        'comp_schedule_mfracs': comp_schedule_mfracs,
    }
    final_carry, out = _run_windowed_scan(step_fn, init_carry, _scan_config)

    return _assemble_result(final_carry, out, adaptive_mesh, Z,
                           atm_ratio=atm_ratio)


def evolve_star_comp(mass, Z=0.014, max_steps=500, alpha_mlt=1.9, t_max=None, f_ov=None):
    """Alias for evolve_star (unified solver)."""
    return evolve_star(mass, Z=Z, max_steps=max_steps, alpha_mlt=alpha_mlt, t_max=t_max, f_ov=f_ov)


# ==================================================================
# Physical-target readout (GRAD-1)
# ==================================================================

def observable_at_target(result, observable='log_L', target_key='star_age',
                         target_value=None):
    """Interpolate an observable onto a fixed physical target (age or center_h1).

    Uses jnp.interp, which is fully differentiable (linear interpolation).

    Parameters
    ----------
    result : dict — output of evolve_star
    observable : str — key for the observable to interpolate
    target_key : str — 'star_age' or 'center_h1'
    target_value : float — fixed physical-coordinate value

    Returns
    -------
    jnp scalar — differentiable observable at target
    """
    target_value = jnp.float64(target_value)
    obs_arr = result[observable]
    coord_arr = result[target_key]

    if target_key == 'center_h1':
        coord_arr = coord_arr[::-1]
        obs_arr = obs_arr[::-1]

    return jnp.interp(target_value, coord_arr, obs_arr)


# ==================================================================
# Diagnostic evolution
# ==================================================================

def evolve_star_diagnostic(mass, Z=0.014, max_steps=500, alpha_mlt=1.9, f_ov=None, diffusion=True, varcontrol_target=None):
    """Non-differentiable diagnostic evolution. Moved to calibration.diagnostic."""
    from stellar_jax.calibration.diagnostic import evolve_star_diagnostic as _impl
    return _impl(mass, Z=Z, max_steps=max_steps, alpha_mlt=alpha_mlt,
                 f_ov=f_ov, diffusion=diffusion, varcontrol_target=varcontrol_target)


def plot_kippenhahn(diag, title=None, filename=None, ax=None):
    """Plot Kippenhahn diagram. Moved to calibration.diagnostic."""
    from stellar_jax.calibration.diagnostic import plot_kippenhahn as _impl
    return _impl(diag, title=title, filename=filename, ax=ax)


# ==================================================================
# Solar calibration
# ==================================================================


def evolve_solar(alpha_mlt, Y_init, Z, max_steps=500, t_max=None, z_feedback=False):
    """Evolve 1 M_sun with explicit Y_init. Moved to calibration.solar."""
    from stellar_jax.calibration.solar import evolve_solar as _impl
    return _impl(alpha_mlt, Y_init, Z, max_steps=max_steps, t_max=t_max, z_feedback=z_feedback)


def solar_residual(alpha_mlt, Y_init, Z=0.0188, t_target=4.57e9, max_steps=500):
    """Compute (log_L, log_R) at t_target. Moved to calibration.solar."""
    from stellar_jax.calibration.solar import solar_residual as _impl
    return _impl(alpha_mlt, Y_init, Z=Z, t_target=t_target, max_steps=max_steps)


def solar_residual_with_zprofile(alpha_mlt, Y_init, Z=0.0188, t_target=4.57e9, max_steps=500):
    """Compute (log_L, log_R) with Z_profile. Moved to calibration.solar."""
    from stellar_jax.calibration.solar import solar_residual_with_zprofile as _impl
    return _impl(alpha_mlt, Y_init, Z=Z, t_target=t_target, max_steps=max_steps)


def solar_calibrate_with_zprofile(Z=0.0188, t_target=4.57e9, max_steps=500, tol=1e-7, max_iter=20):
    """Find (alpha, Y0) with Z_profile. Moved to calibration.solar."""
    from stellar_jax.calibration.solar import solar_calibrate_with_zprofile as _impl
    return _impl(Z=Z, t_target=t_target, max_steps=max_steps, tol=tol, max_iter=max_iter)


def solar_residual_henyey(alpha_mlt, Y_init, Z=0.0196, t_target=4.57e9, max_steps=500):
    """Compute (log_L, log_R, X_c) via Henyey. Moved to calibration.solar."""
    from stellar_jax.calibration.solar import solar_residual_henyey as _impl
    return _impl(alpha_mlt, Y_init, Z=Z, t_target=t_target, max_steps=max_steps)


def solar_calibrate_henyey(Z=0.0188, t_target=4.57e9, max_steps=500, tol=1e-7, max_iter=20):
    """Find (alpha, Y0) on Henyey path. Moved to calibration.solar."""
    from stellar_jax.calibration.solar import solar_calibrate_henyey as _impl
    return _impl(Z=Z, t_target=t_target, max_steps=max_steps, tol=tol, max_iter=max_iter)


def solar_calibrate(Z=0.0188, t_target=4.57e9, max_steps=500, tol=1e-7, max_iter=30):
    """Find (alpha, Y0) via Newton-Raphson. Moved to calibration.solar."""
    from stellar_jax.calibration.solar import solar_calibrate as _impl
    return _impl(Z=Z, t_target=t_target, max_steps=max_steps, tol=tol, max_iter=max_iter)


# ==================================================================
# Model S comparison
# ==================================================================

def compare_model_s(M_solar=1.0, Z=None, alpha_mlt=None, max_steps=500, use_structure_density=True):
    """Compare sound speed and density vs Model S. Moved to calibration.comparison."""
    from stellar_jax.calibration.comparison import compare_model_s as _impl
    return _impl(M_solar=M_solar, Z=Z, alpha_mlt=alpha_mlt, max_steps=max_steps,
                 use_structure_density=use_structure_density)


# ==================================================================
# FGONG / GYRE output — extracted to fgong/ package
# ==================================================================

from stellar_jax.fgong.contracts import N_HIRES  # noqa: F401
from stellar_jax.fgong.hires_profile import hires_profile as _hires_profile  # noqa: F401
from stellar_jax.fgong.builder import structure_to_fgong_jax  # noqa: F401
from stellar_jax.fgong.io import write_fgong, write_gyre  # noqa: F401


# ==================================================================
# Kippenhahn diagram + MESA comparison
# ==================================================================

def kippenhahn_diagram(masses=(1.0, 2.0), Z=0.014, max_steps=500,
                       alpha_mlt=None, filename='kippenhahn.png'):
    """Generate Kippenhahn diagram. Moved to calibration.diagnostic."""
    from stellar_jax.calibration.diagnostic import kippenhahn_diagram as _impl
    return _impl(masses=masses, Z=Z, max_steps=max_steps,
                 alpha_mlt=alpha_mlt, filename=filename)


def read_mesa_history(filepath):
    """Parse MESA history.data file. Moved to calibration.comparison."""
    from stellar_jax.calibration.comparison import read_mesa_history as _impl
    return _impl(filepath)


def compare_mesa(masses=(1.0, 1.2, 1.5, 2.0), Z=0.014, alpha_mlt=None,
                 mesa_dir=None, max_steps=1000, xc_grid=None):
    """Compare tracks against MESA. Moved to calibration.comparison."""
    from stellar_jax.calibration.comparison import compare_mesa as _impl
    return _impl(masses=masses, Z=Z, alpha_mlt=alpha_mlt,
                 mesa_dir=mesa_dir, max_steps=max_steps, xc_grid=xc_grid)
