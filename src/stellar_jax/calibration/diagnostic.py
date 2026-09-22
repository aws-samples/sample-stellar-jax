"""Diagnostic evolution + Kippenhahn diagram.

evolve_star_diagnostic is a self-contained lax.scan loop for
non-differentiable per-shell diagnostics (eps_nuc, convection maps,
eps_grav_frac). Kept as a cohesive kernel with its own carry shape.

NOT differentiable: uses numpy post-processing and Python-level branching.
"""
import os

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from stellar_jax.config.constants import (
    a_rad,
    Msun, Lsun,
    Y_BBN, DY_DZ, SECONDS_PER_YEAR,
    Q_PER_G,
)
from stellar_jax.config.mesh_defaults import (
    F_OV,
    N_NEWTON_COLD, N_NEWTON_WARM, N_COMP, COMP_MFRACS,
)
from stellar_jax.config.calibration import ALPHA_SOLAR
from stellar_jax.config.physics_floors import NAD_CP_FLOOR
from stellar_jax.microphysics.neutrino import epsilon_neutrino
from stellar_jax.structure import initial_guess, shoot_xprofile, newton_solve_xprofile
from stellar_jax.composition import burn_composition, mix_composition, diffuse_composition
from stellar_jax.evolution.contracts import DIAG_CARRY_X

# Hard limits — centralized in config/physics_floors.py
from stellar_jax.config.physics_floors import DELTA_LGL_HARD_LIMIT, DELTA_LGTE_HARD_LIMIT


def _init_diagnostic_state(mass, Z, alpha_mlt, f_ov, max_steps, varcontrol_target):
    """Cold-start setup for diagnostic evolution: ZAMS + initial timestep."""
    mass = jnp.asarray(mass, dtype=jnp.float64)
    Z = jnp.asarray(Z, dtype=jnp.float64)
    alpha_mlt = jnp.asarray(alpha_mlt, dtype=jnp.float64)
    f_ov = jnp.asarray(F_OV if f_ov is None else f_ov, dtype=jnp.float64)

    Y_init = Y_BBN + DY_DZ * Z
    X_init = jnp.maximum(1.0 - Y_init - Z, 0.5)
    X_profile_init = jnp.full(N_COMP, X_init)
    Y_profile_init = jnp.full(N_COMP, Y_init)

    t_start = jnp.float64(5e7 * SECONDS_PER_YEAR)
    t_age_init = jnp.float64(5e7)

    log_L_g, log_Te_g = initial_guess(mass)
    logL0, logTe0 = newton_solve_xprofile(mass, X_profile_init, Z, t_age_init,
                                           log_L_g, log_Te_g, alpha_mlt, N_NEWTON_COLD)

    _, shell_data_init, _ = shoot_xprofile(mass, logL0, logTe0, X_profile_init,
                                           Z, t_age_init, alpha_mlt)
    eps_max_init = jnp.maximum(jnp.max(shell_data_init[:, 0]), 1e-10)
    t_nuc = X_init * Q_PER_G / eps_max_init
    dt_init = 1.5 * t_nuc / max_steps

    varcontrol_target = jnp.float64(1e-3 if varcontrol_target is None else varcontrol_target)
    reject_threshold = jnp.float64(4e-3)

    logT_init = shell_data_init[:, 6]
    logRho_init = shell_data_init[:, 7]
    logP_init = shell_data_init[:, 9]

    return {
        'mass': mass, 'Z': Z, 'alpha_mlt': alpha_mlt, 'f_ov': f_ov,
        'X_profile_init': X_profile_init, 'Y_profile_init': Y_profile_init,
        'logL0': logL0, 'logTe0': logTe0,
        't_start': t_start, 'dt_init': dt_init,
        'varcontrol_target': varcontrol_target, 'reject_threshold': reject_threshold,
        'logT_init': logT_init, 'logRho_init': logRho_init, 'logP_init': logP_init,
    }


def _compute_diagnostic_eps_grav(shell_data, cur_logT, cur_logP, cur_logRho,
                                 prev_logT, prev_logP, dt, logL_out, mass, reject):
    """Compute the gravothermal energy fraction for diagnostic output."""
    mf_diag = shell_data[:, 1]
    nad_diag = shell_data[:, 3]
    dm_diag = jnp.abs(jnp.diff(mf_diag, prepend=mf_diag[0]))
    T_d = 10.0**cur_logT
    P_d = 10.0**cur_logP
    rho_d = 10.0**cur_logRho
    T_prev_d = 10.0**prev_logT
    P_prev_d = 10.0**prev_logP
    P_rad_d = a_rad * T_d**4 / 3.0
    beta_d = jnp.clip((P_d - P_rad_d) / P_d, 0.01, 1.0)
    delta_d = (4.0 - 3.0 * beta_d) / beta_d
    cp_d = P_d * delta_d / (T_d * rho_d * jnp.maximum(nad_diag, NAD_CP_FLOOR))
    dT_dt_d = (T_d - T_prev_d) / jnp.maximum(dt, 1.0)
    dP_dt_d = (P_d - P_prev_d) / jnp.maximum(dt, 1.0)
    eps_grav_d = -cp_d * dT_dt_d + (cp_d * T_d * nad_diag / P_d) * dP_dt_d
    M_star_cgs_d = mass * Msun
    L_grav_d = jnp.sum(eps_grav_d * dm_diag) * M_star_cgs_d
    L_surf_d = 10.0**logL_out * Lsun
    egf_d = L_grav_d / jnp.maximum(L_surf_d, 1e-30)
    return jnp.where(reject, jnp.float64(0.0), egf_d)


def _assemble_diagnostic_result(final_carry, scalars, eps_arr, conv_arr):
    """Pack scan outputs into the diagnostic result dict."""
    return {
        'star_age': scalars[:, 0] / SECONDS_PER_YEAR,
        'center_h1': scalars[:, 1],
        'log_L': scalars[:, 2],
        'log_Teff': scalars[:, 3],
        'log_Tc': scalars[:, 4],
        'eps_grav_frac': scalars[:, 5],
        'eps_nuc': eps_arr,
        'is_convective': conv_arr,
        'mass_coords': np.array(COMP_MFRACS),
        'X_profile': final_carry[DIAG_CARRY_X],
        'from_shooting': True,
        'from_henyey': False,
    }


def evolve_star_diagnostic(mass, Z=0.014, max_steps=500, alpha_mlt=1.9, f_ov=None, diffusion=True, varcontrol_target=None):
    """Non-differentiable diagnostic evolution returning per-shell data."""
    s = _init_diagnostic_state(mass, Z, alpha_mlt, f_ov, max_steps, varcontrol_target)
    mass = s['mass']
    Z = s['Z']
    alpha_mlt = s['alpha_mlt']
    f_ov = s['f_ov']
    varcontrol_target = s['varcontrol_target']
    reject_threshold = s['reject_threshold']
    dt_grow = jnp.float64(1.5)
    dt_shrink = jnp.float64(0.5)
    comp_mfracs = jnp.array(COMP_MFRACS)

    def step_fn(carry, _):
        X_prof, Y_prof, logL, logTe, t, dt, prev_logT, prev_logP, prev_logRho = carry
        t_age = t / SECONDS_PER_YEAR

        logL_new, logTe_new = newton_solve_xprofile(
            mass, X_prof, Z, t_age, logL, logTe, alpha_mlt, N_NEWTON_WARM)
        _, shell_data, _ = shoot_xprofile(
            mass, logL_new, logTe_new, X_prof, Z, t_age, alpha_mlt)

        mf_sorted = shell_data[:, 1][::-1]
        eps_sorted = shell_data[:, 0][::-1]
        eps_comp = jnp.interp(comp_mfracs, mf_sorted, eps_sorted,
                              left=eps_sorted[0], right=eps_sorted[-1])

        nrad = shell_data[:, 2]
        nad = shell_data[:, 3]
        is_conv = (nrad > nad).astype(jnp.float64)
        conv_sorted = is_conv[::-1]
        conv_comp = jnp.interp(comp_mfracs, mf_sorted, conv_sorted,
                               left=conv_sorted[0], right=conv_sorted[-1])
        is_conv_comp = (conv_comp > 0.5).astype(jnp.float64)

        X_burned, Y_burned = burn_composition(X_prof, Y_prof, shell_data, dt)
        X_mixed = mix_composition(X_burned, shell_data, M_solar=mass, f_ov=f_ov)
        if diffusion:
            X_diff, Y_diff, _ = diffuse_composition(X_mixed, Y_burned, shell_data, dt, mass, use_exact_structure=True)
        else:
            X_diff, Y_diff = X_mixed, Y_burned
        Y_mixed = mix_composition(Y_diff, shell_data, M_solar=mass, f_ov=f_ov)

        # varcontrol sizing
        max_dX = jnp.max(jnp.abs(X_diff - X_prof))
        cur_logT = shell_data[:, 6]
        cur_logRho = shell_data[:, 7]
        cur_logP = shell_data[:, 9]
        max_delta_logT = jnp.max(jnp.abs(cur_logT - prev_logT))
        max_delta_logRho = jnp.max(jnp.abs(cur_logRho - prev_logRho))
        delta_logL = jnp.abs(logL_new - logL)
        delta_logTe = jnp.abs(logTe_new - logTe)
        varcontrol_reject = jnp.maximum(delta_logL, delta_logTe)
        varcontrol = jnp.maximum(varcontrol_reject,
                                 jnp.maximum(max_delta_logT, max_delta_logRho))
        varcontrol = jnp.maximum(varcontrol, max_dX)

        # Post-TAMS relaxation
        X_center_cur = X_diff[0]
        w_post_tams = jax.nn.sigmoid((0.01 - X_center_cur) / 0.003)
        varcontrol_target_eff = varcontrol_target * (1.0 + 4.0 * w_post_tams)
        reject_threshold_eff = reject_threshold * (1.0 + 9.0 * w_post_tams)

        dt_at_floor = dt <= (1.1e5 * SECONDS_PER_YEAR)
        reject = ((varcontrol_reject > reject_threshold_eff) | (delta_logL > DELTA_LGL_HARD_LIMIT) | (delta_logTe > DELTA_LGTE_HARD_LIMIT)) & (~dt_at_floor)
        ratio = varcontrol_target_eff / jnp.maximum(varcontrol, 1e-30)
        factor = jnp.clip(ratio, dt_shrink, dt_grow)
        dt_next = jnp.where(reject, dt * 0.5, dt * factor)

        X_out = jnp.where(reject, X_prof, X_diff)
        Y_out = jnp.where(reject, Y_prof, Y_mixed)
        logL_out = jnp.where(reject, logL, logL_new)
        logTe_out = jnp.where(reject, logTe, logTe_new)
        t_new = jnp.where(reject, t, t + dt)
        logT_out = jnp.where(reject, prev_logT, cur_logT)
        logRho_out = jnp.where(reject, prev_logRho, cur_logRho)
        logP_out = jnp.where(reject, prev_logP, cur_logP)

        # Timestep limiter
        eps_nuc_max = jnp.maximum(jnp.max(shell_data[:, 0]), 1e-10)
        T_c = 10.0**cur_logT[-1]
        rho_c = 10.0**cur_logRho[-1]
        eps_nu_c = epsilon_neutrino(rho_c, T_c, X_out[0], Z)
        eps_max = jnp.maximum(eps_nuc_max, eps_nu_c)
        dt_max = 0.10 * Q_PER_G / eps_max
        dt_next = jnp.minimum(dt_next, dt_max)
        dt_next = jnp.maximum(dt_next, 1e5 * SECONDS_PER_YEAR)

        X_center = X_out[0]
        log_Tc_out = jnp.where(reject, prev_logT[-1], cur_logT[-1])

        egf_out = _compute_diagnostic_eps_grav(
            shell_data, cur_logT, cur_logP, cur_logRho,
            prev_logT, prev_logP, dt, logL_out, mass, reject)

        scalar_out = jnp.array([t_new, X_center, logL_out, logTe_out, log_Tc_out, egf_out])
        eps_out = jnp.where(reject, jnp.zeros(N_COMP), eps_comp)
        conv_out = jnp.where(reject, jnp.zeros(N_COMP), is_conv_comp)

        return (X_out, Y_out, logL_out, logTe_out, t_new, dt_next,
                logT_out, logP_out, logRho_out), (scalar_out, eps_out, conv_out)

    init_carry = (s['X_profile_init'], s['Y_profile_init'], s['logL0'], s['logTe0'],
                  s['t_start'], s['dt_init'], s['logT_init'], s['logP_init'], s['logRho_init'])
    final_carry, (scalars, eps_arr, conv_arr) = lax.scan(step_fn, init_carry, None, length=max_steps)

    return _assemble_diagnostic_result(final_carry, scalars, eps_arr, conv_arr)


def plot_kippenhahn(diag, title=None, filename=None, ax=None):
    """Plot a Kippenhahn diagram from evolve_star_diagnostic output."""
    import matplotlib.pyplot as plt

    age = np.array(diag['star_age'])
    mcoords = np.array(diag['mass_coords'])
    eps = np.array(diag['eps_nuc'])
    conv = np.array(diag['is_convective'])

    valid = age > 0
    age = age[valid]
    eps = eps[valid]
    conv = conv[valid]

    if len(age) < 2:
        raise ValueError("Not enough valid timesteps for Kippenhahn diagram")

    age_gyr = age / 1e9

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.figure

    eps_floor = 1e-2
    eps_plot = np.maximum(eps, eps_floor)

    pcm = ax.pcolormesh(age_gyr, mcoords, np.log10(eps_plot).T,
                        shading='auto', cmap='inferno',
                        vmin=np.log10(eps_floor), vmax=np.log10(np.maximum(eps.max(), 1.0)))
    fig.colorbar(pcm, ax=ax, label=r'$\log_{10}\,\varepsilon_{\rm nuc}$ [erg g$^{-1}$ s$^{-1}$]')

    ax.contourf(age_gyr, mcoords, conv.T, levels=[0.5, 1.5],
                colors='none', hatches=['///'], alpha=0)
    ax.contour(age_gyr, mcoords, conv.T, levels=[0.5],
               colors='cyan', linewidths=0.8)

    ax.set_xlabel('Age [Gyr]')
    ax.set_ylabel(r'Mass coordinate $m/M$')
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches='tight')

    return fig, ax


def kippenhahn_diagram(masses=(1.0, 2.0), Z=0.014, max_steps=500,
                       alpha_mlt=None, filename='kippenhahn.png'):
    """Generate Kippenhahn diagram using evolve_star_diagnostic.

    Reference: Kippenhahn, Weigert & Weiss (2012), ch. 34.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if alpha_mlt is None:
        alpha_mlt = ALPHA_SOLAR

    fig, axes = plt.subplots(len(masses), 1, figsize=(10, 4*len(masses)), squeeze=False)

    for ax_idx, mass in enumerate(masses):
        ax = axes[ax_idx, 0]
        diag = evolve_star_diagnostic(float(mass), Z=Z, max_steps=max_steps,
                                      alpha_mlt=float(alpha_mlt))
        age = np.array(diag['star_age'])
        if (age > 0).sum() < 2:
            ax.text(0.5, 0.5, f'{mass} M\u2609: insufficient steps',
                    transform=ax.transAxes, ha='center')
            continue
        plot_kippenhahn(diag, title=f'{mass:.1f} M$_\\odot$, Z={Z}', ax=ax)

    plt.tight_layout()
    outdir = os.path.dirname(filename)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    return filename
