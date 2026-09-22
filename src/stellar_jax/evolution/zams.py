"""evolution/zams.py — ZAMS initialization, warmup, and parameter setup.

Extracted from _core.py (#1089). These are module-level helpers called during
the JIT-traced body of _evolve_star_jit. Python function calls are invisible
to XLA tracing — the compiled graph is IDENTICAL to the monolithic version.

Contains:
- _init_params_and_profiles: parameter coercion + initial composition profiles
- _init_zams_and_warmup: ZAMS model build + X3 equilibrium + eps_grav warmup
- _egrav_warmup: fixed-point iteration for eps_grav-consistent initial state
- _extract_warmup_observables: surface observables from warmup state
"""

import jax
import jax.numpy as jnp

from stellar_jax.config.constants import (
    a_rad, sigma_sb, Msun, Lsun,
    Y_BBN, DY_DZ, SECONDS_PER_YEAR, Q_PER_G,
)
from stellar_jax.config.mesh_defaults import (
    F_OV, N_HENYEY, N_NEWTON_COLD, N_COMP, COMP_MFRACS,
)
from stellar_jax.config.physics_floors import PGAS_FRAC_FLOOR
from stellar_jax.microphysics.eos import eos_lookup
from stellar_jax.microphysics.nuclear import epsilon_nuclear, he3_equilibrium
from stellar_jax.structure import (
    initial_guess, interp_X_at_mass, newton_solve_xprofile, build_model_on_mesh,
)
from stellar_jax.mesh import initial_lagrangian_mesh as _lagrangian_mass_mesh
from stellar_jax.henyey import (
    henyey_solve_from_state_atm, henyey_init_from_state_atm,
    extract_shell_data_from_henyey,
)


def _egrav_warmup(y_henyey_init: jnp.ndarray, warmup_params: dict,
                  n_iter: int = 4) -> jnp.ndarray:
    """eps_grav-consistent warmup — collapses the 4 copy-pasted iterations.

    Each iteration solves: y_next = henyey_solve(guess=y_prev, prev=y_prev)
    forming a fixed-point iteration f(x) = x. After n_iter iterations, the
    state IS what scan step 1 would converge to.

    This is a Python for-loop (NOT lax.fori_loop) — it runs at trace time
    and XLA unrolls it identically to the copy-pasted version.

    MESA parallel: init_model.f90 runs the initial model through the SAME
    solver configuration as the first step (same dt, same eps_grav treatment).

    Parameters
    ----------
    y_henyey_init : initial converged state from Pass 1
    warmup_params : dict — {q_mesh, M_star_cgs, X_profile_init, Z, alpha_mlt,
        R_phot_init, inv_dt_init, atm_ratio, N14_profile_init, X3_profile_init}
    n_iter : number of fixed-point iterations (default 4)

    Returns
    -------
    y_warmup : the eps_grav-consistent state (fixed point)
    """
    q_mesh = warmup_params['q_mesh']
    M_star_cgs = warmup_params['M_star_cgs']
    X_profile_init = warmup_params['X_profile_init']
    Z = warmup_params['Z']
    alpha_mlt = warmup_params['alpha_mlt']
    R_phot_init = warmup_params['R_phot_init']
    inv_dt_init = warmup_params['inv_dt_init']
    atm_ratio = warmup_params['atm_ratio']
    N14_profile_init = warmup_params['N14_profile_init']
    X3_profile_init = warmup_params['X3_profile_init']

    y = y_henyey_init
    for _ in range(n_iter):
        result = henyey_solve_from_state_atm(
            y, q_mesh, M_star_cgs, X_profile_init, Z, alpha_mlt,
            R_phot_init, n_iter=100, tol=1e-4, atm_ratio=atm_ratio,  # MESA: star_solver.f90:213
            ln_T_prev=y[:, 2], ln_P_prev=y[:, 1],
            inv_dt=jax.lax.stop_gradient(inv_dt_init),  # GP-1: TIMESTEP
            N14_profile=N14_profile_init, X3_profile=X3_profile_init)
        y = result['y']
    return y


def _extract_warmup_observables(y_warmup: jnp.ndarray,
                                warmup_params: dict) -> tuple:
    """Extract surface observables and shell data from the warmup state.

    Parameters
    ----------
    y_warmup : (N_s, 4) — Henyey state from eps_grav warmup
    warmup_params : dict — {N_s, atm_ratio, q_mesh, M_star_cgs,
        X_profile_init, Z, alpha_mlt}

    Returns
    -------
    logL0, logTe0, logT_init, logP_init, logRho_init
    """
    N_s = warmup_params['N_s']
    atm_ratio = warmup_params['atm_ratio']
    q_mesh = warmup_params['q_mesh']
    M_star_cgs = warmup_params['M_star_cgs']
    X_profile_init = warmup_params['X_profile_init']
    Z = warmup_params['Z']
    alpha_mlt = warmup_params['alpha_mlt']

    ell_surf_init = y_warmup[N_s - 1, 3]
    L_init_h = ell_surf_init * Lsun
    R_phot_obs_init = atm_ratio * jnp.exp(y_warmup[-1, 0])
    T_eff_init = (L_init_h / (4.0 * jnp.pi * sigma_sb * R_phot_obs_init**2))**0.25
    logL0 = jnp.log10(jnp.maximum(L_init_h, 1e-30) / Lsun)
    logTe0 = jnp.log10(jnp.maximum(T_eff_init, 1.0))

    # shell_data_init for the carry state (logT/logP/logRho profiles for
    # Form C eps_grav). stop_gradient'd — see annotation.
    shell_data_init = extract_shell_data_from_henyey(
        jax.lax.stop_gradient(y_warmup), q_mesh, M_star_cgs,  # GP-8: EPSGRAV_HISTORY
        X_profile_init, Z, alpha_mlt, 0.0)
    logT_init = shell_data_init[:, 6]
    logRho_init = shell_data_init[:, 7]
    logP_init = shell_data_init[:, 9]

    return logL0, logTe0, logT_init, logP_init, logRho_init


def _init_zams_and_warmup(stellar_params: dict, init_state: dict,
                          numerics_static: dict) -> dict:
    # lint: sequential-setup exception — 2-pass ZAMS + X3 eq + warmup is one atomic init sequence
    """Build ZAMS model, compute X3 equilibrium, run eps_grav warmup.

    Parameters
    ----------
    stellar_params : dict — {mass, Z, alpha_mlt, X_init}
    init_state : dict — {X_profile_init, N14_profile_init, logL0, logTe0}
    numerics_static : dict — {fixed_dt, freeze_schedule}

    Returns a dict with: M_star_cgs, q_mesh, N_s, atm_ratio,
    X3_profile_init, y_warmup, dt_init, logL0, logTe0, logT_init,
    logP_init, logRho_init.
    """
    mass = stellar_params['mass']
    Z = stellar_params['Z']
    alpha_mlt = stellar_params['alpha_mlt']
    X_init = stellar_params['X_init']
    X_profile_init = init_state['X_profile_init']
    N14_profile_init = init_state['N14_profile_init']
    logL0 = init_state['logL0']
    logTe0 = init_state['logTe0']
    fixed_dt = numerics_static['fixed_dt']
    freeze_schedule = numerics_static['freeze_schedule']
    M_star_cgs = mass * Msun
    q_mesh = _lagrangian_mass_mesh(N_HENYEY)
    model_init = build_model_on_mesh(mass, logL0, logTe0, X_profile_init,
                                     Z, alpha_mlt, N_HENYEY, q_mesh_in=q_mesh)
    N_s = N_HENYEY
    ln_r = jnp.log(jnp.maximum(model_init['r'][:N_s], 1.0))
    ln_P = jnp.log(10.0) * model_init['logP'][:N_s]
    ln_T = jnp.log(10.0) * model_init['logT'][:N_s]
    ell = (model_init['L'] / Lsun)[:N_s]
    y_henyey_raw = jnp.stack([ln_r, ln_P, ln_T, ell], axis=-1)
    R_phot_init = model_init['r'][N_s]

    # 2-pass ZAMS init for self-consistent local phi (MESA init_model.f90).
    # Pass 0: cold-start without X3.
    _init_pass0 = henyey_init_from_state_atm(
        y_henyey_raw, q_mesh, M_star_cgs, X_profile_init, Z, alpha_mlt,
        R_phot_init, n_iter=200, tol=1e-4)
    _y_pass0 = _init_pass0['y']

    # Compute X3_eq from Pass 0 structure (stop_gradient'd).
    _q_henyey = q_mesh[:N_s]
    _T_at_comp = jnp.exp(jnp.interp(COMP_MFRACS, _q_henyey, _y_pass0[:, 2]))
    _P_at_comp = jnp.exp(jnp.interp(COMP_MFRACS, _q_henyey, _y_pass0[:, 1]))
    _P_rad = a_rad * _T_at_comp**4 / 3.0
    _Pgas = jnp.maximum(_P_at_comp - _P_rad, PGAS_FRAC_FLOOR * _P_at_comp)
    _rho, _, _, _, _, _, _ = jax.vmap(
        lambda lt, lp, x: eos_lookup(lt, lp, x, Z)
    )(jnp.log10(_T_at_comp), jnp.log10(_Pgas), X_profile_init)
    _X3_eq, _ = jax.vmap(
        lambda T, rho, X: he3_equilibrium(T, rho, X)
    )(_T_at_comp, _rho, X_profile_init)
    X3_profile_init = jax.lax.stop_gradient(jnp.minimum(_X3_eq, 1.0))  # GP-4: Z_CNO_ISOTOPES

    # Pass 1: re-converge with correct X3 (phi≈1.0).
    henyey_init_result = henyey_init_from_state_atm(
        _y_pass0, q_mesh, M_star_cgs, X_profile_init, Z, alpha_mlt,
        R_phot_init, n_iter=200, tol=1e-4, X3_profile=X3_profile_init)
    y_henyey_init = henyey_init_result['y']
    atm_ratio = R_phot_init / jnp.exp(y_henyey_init[-1, 0])

    # Center-only eps for initial dt (avoids vmap backward NaN).
    _T_c = jnp.exp(y_henyey_init[0, 2])
    _P_c = jnp.exp(y_henyey_init[0, 1])
    _P_rad_c = a_rad * _T_c**4 / 3.0
    _P_gas_c = jnp.maximum(_P_c - _P_rad_c, PGAS_FRAC_FLOOR * _P_c)
    _X_c = interp_X_at_mass(X_profile_init, q_mesh[0])
    _rho_c, _, _, _, _, _, _ = eos_lookup(
        jnp.log10(_T_c), jnp.log10(_P_gas_c), _X_c, Z)
    eps_center = epsilon_nuclear(
        _rho_c, _T_c, _X_c, Z, jnp.float64(0.0),
        X_N14=N14_profile_init[0], X3=X3_profile_init[0])
    t_nuc = X_init * Q_PER_G / jnp.maximum(eps_center, 1e-10)
    dt_init = t_nuc / 50000.0

    if fixed_dt is not None:
        dt_init = jnp.float64(fixed_dt * SECONDS_PER_YEAR)
    if freeze_schedule:
        dt_init = jax.lax.stop_gradient(dt_init)  # GP-1: TIMESTEP

    # eps_grav-consistent warmup (MESA init_model.f90 pattern).
    _inv_dt = jnp.where(dt_init > 0.0, 1.0 / jnp.maximum(dt_init, 1.0), 0.0)
    _warmup_params = {
        'q_mesh': q_mesh, 'M_star_cgs': M_star_cgs,
        'X_profile_init': X_profile_init, 'Z': Z, 'alpha_mlt': alpha_mlt,
        'R_phot_init': R_phot_init, 'inv_dt_init': _inv_dt,
        'atm_ratio': atm_ratio, 'N14_profile_init': N14_profile_init,
        'X3_profile_init': X3_profile_init, 'N_s': N_s,
    }
    y_warmup = _egrav_warmup(y_henyey_init, _warmup_params, n_iter=4)
    logL0, logTe0, logT_init, logP_init, logRho_init = _extract_warmup_observables(
        y_warmup, _warmup_params)

    return {
        'M_star_cgs': M_star_cgs, 'q_mesh': q_mesh, 'N_s': N_s,
        'atm_ratio': atm_ratio, 'X3_profile_init': X3_profile_init,
        'y_warmup': y_warmup, 'dt_init': dt_init,
        'logL0': logL0, 'logTe0': logTe0,
        'logT_init': logT_init, 'logP_init': logP_init,
        'logRho_init': logRho_init,
    }


def _init_params_and_profiles(traced_inputs: dict,
                              numerics_static: dict) -> dict:
    """Coerce args, build initial composition profiles, ZAMS cold-start solve.

    Parameters
    ----------
    traced_inputs : dict — {mass, Z, alpha_mlt, opacity_factor, eps_nuc_factor,
        diffusion_factor, f_ov, t_max, Y_init, varcontrol_target}
    numerics_static : dict — {fixed_dt, freeze_schedule}

    Pure setup extracted from _evolve_star_jit so the JIT body stays under
    ~200 lines.  Python calls are invisible to XLA tracing — the compiled
    graph is identical to the monolithic version.
    """
    mass = traced_inputs['mass']
    Z = traced_inputs['Z']
    alpha_mlt = traced_inputs['alpha_mlt']
    opacity_factor = traced_inputs['opacity_factor']
    eps_nuc_factor = traced_inputs['eps_nuc_factor']
    diffusion_factor = traced_inputs['diffusion_factor']
    f_ov = traced_inputs['f_ov']
    t_max = traced_inputs['t_max']
    Y_init = traced_inputs['Y_init']
    varcontrol_target = traced_inputs['varcontrol_target']

    mass = jnp.asarray(mass, dtype=jnp.float64)
    Z = jnp.asarray(Z, dtype=jnp.float64)
    alpha_mlt = jnp.asarray(alpha_mlt, dtype=jnp.float64)
    opacity_factor = jnp.asarray(opacity_factor, dtype=jnp.float64)
    eps_nuc_factor = jnp.asarray(eps_nuc_factor, dtype=jnp.float64)
    diffusion_factor = jnp.asarray(diffusion_factor, dtype=jnp.float64)
    f_ov = jnp.asarray(F_OV if f_ov is None else f_ov, dtype=jnp.float64)
    t_max_sec = jnp.float64(t_max * SECONDS_PER_YEAR) if t_max is not None else jnp.float64(1e30)
    if Y_init is not None:
        Y_init = jnp.asarray(Y_init, dtype=jnp.float64)
    else:
        Y_init = Y_BBN + DY_DZ * Z
    X_init = jnp.maximum(1.0 - Y_init - Z, 0.5)
    X_profile_init = jnp.full(N_COMP, X_init)
    Y_profile_init = jnp.full(N_COMP, Y_init)
    # Initial CNO mass fractions from the Grevesse & Sauval (1998) solar mix:
    # C/Z ≈ 0.170, ¹²C/¹³C ≈ 89 (Anders & Grevesse 1989), N/Z ≈ 0.049.
    # (Full CNO-catalyst accounting incl. O16 is documented in microphysics/nuclear.py.)
    C12_profile_init = jnp.full(N_COMP, 0.170 * Z)
    C13_profile_init = jnp.full(N_COMP, 0.170 * Z / 89.0)
    # N14 initial: MESA ZAMS center X_N14 ≈ 0.251*Z (measured from FGONG col 23
    # across 1.0-2.0 Msun, all within 0.2507-0.2535*Z). This exceeds the CN-only
    # equilibrium (0.221*Z) by 0.030*Z because PMS evolution (10-30 Myr at T~15-20 MK)
    # converts ~6% of initial O16 to N14 via the ON sub-cycle before reaching ZAMS.
    # We set N14_init = 0.251*Z - C12_init - C13_init = 0.079*Z so that
    # CN_total = C12+C13+N14 = 0.251*Z. Since burn_cno conserves CN_total,
    # the tracked N14 at CN equilibrium ≈ 0.251*Z, matching MESA ZAMS.
    # MESA ref: net_approx21.f90:1116 (y(in14) set by PMS evolution);
    #   FGONG col 23 = tracked X_N14; pulse_fgong.f90 (FGONG storage).
    N14_profile_init = jnp.full(N_COMP, 0.079 * Z)
    log_L_g, log_Te_g = initial_guess(mass)
    logL0, logTe0 = newton_solve_xprofile(mass, X_profile_init, Z, jnp.float64(0.0),
                                           log_L_g, log_Te_g, alpha_mlt, N_NEWTON_COLD)

    # Single source of truth: config/physics_floors.py.
    from stellar_jax.config.physics_floors import DELTA_LGL_HARD_LIMIT, DELTA_LGTE_HARD_LIMIT

    _vct = 1e-3 if varcontrol_target is None else varcontrol_target
    varcontrol_target_val = jnp.float64(_vct)
    reject_threshold = jnp.float64(4.0 * _vct)
    dt_grow = jnp.float64(1.5)
    dt_shrink = jnp.float64(0.5)
    _DELTA_LGL_HARD = jnp.float64(DELTA_LGL_HARD_LIMIT)
    _DELTA_LGTE_HARD = jnp.float64(DELTA_LGTE_HARD_LIMIT)
    # ZAMS initialization + eps_grav warmup.
    _stellar_params = {'mass': mass, 'Z': Z, 'alpha_mlt': alpha_mlt, 'X_init': X_init}
    _init_state = {
        'X_profile_init': X_profile_init, 'N14_profile_init': N14_profile_init,
        'logL0': logL0, 'logTe0': logTe0,
    }
    _zams = _init_zams_and_warmup(_stellar_params, _init_state, numerics_static)
    return {
        'mass': mass, 'Z': Z, 'alpha_mlt': alpha_mlt,
        'opacity_factor': opacity_factor, 'eps_nuc_factor': eps_nuc_factor,
        'diffusion_factor': diffusion_factor, 'f_ov': f_ov,
        't_max_sec': t_max_sec, 'X_init': X_init,
        'X_profile_init': X_profile_init, 'Y_profile_init': Y_profile_init,
        'C12_profile_init': C12_profile_init,
        'C13_profile_init': C13_profile_init,
        'N14_profile_init': N14_profile_init,
        'logL0': _zams['logL0'], 'logTe0': _zams['logTe0'],
        'varcontrol_target': varcontrol_target_val,
        'reject_threshold': reject_threshold,
        'dt_grow': dt_grow, 'dt_shrink': dt_shrink,
        '_DELTA_LGL_HARD': _DELTA_LGL_HARD,
        '_DELTA_LGTE_HARD': _DELTA_LGTE_HARD,
        'M_star_cgs': _zams['M_star_cgs'], 'q_mesh': _zams['q_mesh'],
        'N_s': _zams['N_s'], 'atm_ratio': _zams['atm_ratio'],
        'X3_profile_init': _zams['X3_profile_init'],
        'y_warmup': _zams['y_warmup'], 'dt_init': _zams['dt_init'],
        'logT_init': _zams['logT_init'], 'logP_init': _zams['logP_init'],
        'logRho_init': _zams['logRho_init'],
    }
