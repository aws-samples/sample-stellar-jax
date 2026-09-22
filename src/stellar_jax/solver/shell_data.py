"""Post-solve diagnostic extraction from a converged Henyey state.

KEEP WHOLE (the module redesign spec §2): the internal structure
(EOS→opacity→nuclear→nabla_rad→MLT→stack) is a linear pipeline with no
branching logic. Each vmap call feeds the next; splitting would produce
functions that can't be called independently.
"""

import jax
import jax.numpy as jnp

from stellar_jax.config.constants import G, Lsun, a_rad, c_light
from stellar_jax.config.physics_floors import PGAS_FRAC_FLOOR
from stellar_jax.microphysics.eos import eos_lookup
from stellar_jax.microphysics.opacity import kappa
from stellar_jax.microphysics.nuclear import epsilon_nuclear
from stellar_jax.transport import mlt_nabla
from stellar_jax.structure import interp_X_at_mass


def extract_shell_data_from_henyey(y, q_mesh, M_star, X_profile, Z, alpha_mlt, t_age=None, comp_mfracs=None,
                                   helm_eos=False, bicubic_opacity=True):
    """Extract shell_data (N_s, 10) from a converged Henyey state.

    Computes the same per-shell diagnostics as shoot_xprofile:
      col 0: eps (nuclear energy generation rate, erg/g/s)
      col 1: mass fraction (q = m/M_star)
      col 2: nabla_rad (radiative temperature gradient)
      col 3: nabla_ad (adiabatic temperature gradient)
      col 4: H_p / R_star (pressure scale height / stellar radius)
      col 5: dm/dr normalized (= dm_dr * R_star / M_star)
      col 6: log10(T)
      col 7: log10(rho)
      col 8: nabla (actual temperature gradient, including MLT)
      col 9: log10(P)

    Parameters
    ----------
    y : (N_s, 4) array — [ln_r, ln_P, ln_T, ell] at each interface
    q_mesh : (N_s+1,) array — mass coordinates
    M_star : float — stellar mass in grams
    X_profile : (N_COMP,) array — hydrogen mass fraction profile
    Z : float — metal mass fraction
    alpha_mlt : float — mixing-length parameter
    t_age : float or None — stellar age in years (for nuclear rates)
    comp_mfracs : array or None — composition grid positions (default: COMP_MFRACS)

    Returns
    -------
    shell_data : (N_s, 10) array — same layout as shoot_xprofile output
    """
    N_s = y.shape[0]
    _t_age = jnp.float64(0.0) if t_age is None else jnp.asarray(t_age, dtype=jnp.float64)

    ln_r = y[:, 0]
    ln_P = y[:, 1]
    ln_T = y[:, 2]
    ell = y[:, 3]

    r = jnp.exp(ln_r)
    P = jnp.exp(ln_P)
    T = jnp.exp(ln_T)
    L = ell * Lsun

    logT = jnp.log10(T)
    P_rad = a_rad * T**4 / 3.0
    P_gas = jnp.maximum(P - P_rad, PGAS_FRAC_FLOOR * P)
    logPgas = jnp.log10(P_gas)

    # Mass coordinate at each interface
    q_interfaces = q_mesh[:N_s]  # (N_s,)
    m_interfaces = q_interfaces * M_star

    # Interpolate X at each interface
    def _get_X(q):
        return interp_X_at_mass(X_profile, q, comp_mfracs)
    X_at_q = jax.vmap(_get_X)(q_interfaces)

    # EOS at each interface
    def _eos_at_k(logT_k, logPg_k, X_k):
        return eos_lookup(logT_k, logPg_k, X_k, Z, helm_eos=helm_eos)
    rho_arr, mu_arr, nad_arr, S_arr, cp_arr, chi_rho_arr, chi_T_arr = jax.vmap(
        _eos_at_k)(logT, logPgas, X_at_q)

    logrho = jnp.log10(rho_arr)
    logP_total = jnp.log10(P)

    # Opacity
    def _kap_at_k(logT_k, logrho_k, X_k):
        return kappa(logT_k, logrho_k, X_k, Z, bicubic_opacity=bicubic_opacity)
    log_kap_arr = jax.vmap(_kap_at_k)(logT, logrho, X_at_q)
    kap_arr = 10.0**log_kap_arr

    # Nuclear energy
    def _eps_at_k(rho_k, T_k, X_k):
        return epsilon_nuclear(rho_k, T_k, X_k, Z, _t_age)
    eps_arr = jax.vmap(_eps_at_k)(rho_arr, T, X_at_q)

    # Radiative gradient: nabla_rad = 3κLP / (16π a c G m T^4)
    nabla_rad_arr = 3.0 * kap_arr * L * P / (
        16.0 * jnp.pi * a_rad * c_light * G * jnp.maximum(m_interfaces, 1e-10 * M_star) * T**4 + 1e-30)

    # Local gravity and MLT
    g_local = G * jnp.maximum(m_interfaces, 1e-10 * M_star) / (r**2 + 1e-30)

    def _nabla_at_k(nrad_k, nad_k, T_k, P_k, rho_k, kap_k, g_k, mu_k):
        return mlt_nabla(nrad_k, nad_k, T_k, P_k, rho_k, kap_k, g_k, mu_k, alpha_mlt)
    nabla_arr = jax.vmap(_nabla_at_k)(
        nabla_rad_arr, nad_arr, T, P, rho_arr, kap_arr, g_local, mu_arr)

    # Pressure scale height / R_star
    R_star = r[-1]  # outermost interface = surface
    H_p = P / (rho_arr * g_local + 1e-30)
    hp_over_R = H_p / R_star

    # dm/dr normalized: dm_dr = 4π r² ρ, normalized by R_star/M_star
    dm_dr = 4.0 * jnp.pi * r**2 * rho_arr
    dm_dr_norm = dm_dr * R_star / M_star

    # Stack into shell_data format — reversed to match shoot_xprofile convention
    # (index 0 = surface, index N-1 = center), since Henyey mesh goes center→surface.
    shell_data = jnp.stack([
        eps_arr,        # col 0
        q_interfaces,   # col 1: mass fraction
        nabla_rad_arr,  # col 2
        nad_arr,        # col 3
        hp_over_R,      # col 4
        dm_dr_norm,     # col 5
        logT,           # col 6
        logrho,         # col 7
        nabla_arr,      # col 8
        logP_total,     # col 9
    ], axis=-1)

    return shell_data[::-1]
