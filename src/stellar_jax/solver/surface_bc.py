"""Atmosphere bridge surface boundary conditions.

Contains:
  - _atm_bridge_endpoint: shared RK2 bridge integration (3000-step lax.scan)
  - _surface_bc_atm: full atmosphere bridge BC residuals
  - _surface_bc_atm_values: bridge endpoint (ln_P, ln_T) computation
  - _surface_bc_atm_values_with_ratio: bridge with R_phot = atm_ratio * r_surf

KEEP WHOLE decision (the module redesign spec §2): the full atmosphere
bridge is a single numerical integration (atmosphere_bc → RK2 scan inward).
The RK2 step body + derivative function share closure-captured state and the
3000-step scan is a single coupled integration.

JAX PITFALL (§5.2 XLA recompile): N_ENV=3000 and TAU_ATM=100.0 are module-level
constants. If they accidentally become traced values, lax.scan(length=N_ENV)
would fail loudly (length must be static). Keep as constants.

References:
  - Kippenhahn, Weigert & Weiss (2012), §11.1 (fitting-point method)
  - Paxton et al. (2013), ApJS 208, 4, App. A.5 (MESA atm module)
  - Krishna Swamy (1966), ApJ 145, 174
  - MESA star/private/hydro_eqns.f90:817 (get_PT_bc_ad)
"""

import jax.numpy as jnp
from jax import lax

from stellar_jax.config.constants import G, Lsun, sigma_sb, a_rad, c_light
from stellar_jax.microphysics.eos import eos_lookup
from stellar_jax.microphysics.opacity import kappa
from stellar_jax.transport import mlt_nabla
from stellar_jax.structure import atmosphere_bc

# Module-level constants — static for lax.scan length.
TAU_ATM = 100.0
N_ENV = 3000


def _atm_bridge_endpoint(y_surf, M_star, X_surf, Z, alpha_mlt, R_star,
                         helm_eos=False, bicubic_opacity=True):
    """Integrate the RK2 atmosphere bridge from photosphere to mesh surface.

    Shared core of both _surface_bc_atm and _surface_bc_atm_values. Returns
    the bridge endpoint (ln_P_end, ln_T_end) at the Henyey mesh surface.

    The integration proceeds in ln(r) space from R_star (photosphere) inward
    to r_surf (mesh surface at q≈0.993), using a 3000-step RK2 (midpoint)
    scheme with Krishna Swamy T(τ) atmosphere as the outer boundary.
    """
    ln_r_s = y_surf[0]
    ell_s = y_surf[3]
    r_surf = jnp.exp(ln_r_s)
    L_s = ell_s * Lsun
    T_eff = (L_s / (4.0 * jnp.pi * sigma_sb * R_star**2))**0.25
    g_phot = G * M_star / R_star**2
    P_atm, T_atm = atmosphere_bc(T_eff, g_phot, X_surf, Z, alpha_mlt, L_s, M_star,
                                  tau_base=TAU_ATM)

    ln_R_star = jnp.log(R_star)
    ln_r_fit = jnp.log(r_surf)
    d_ln_r = (ln_r_fit - ln_R_star) / N_ENV

    def _derivs(ln_r_val, P_val, T_val):
        r = jnp.exp(ln_r_val)
        P_val = jnp.maximum(P_val, 1.0)
        T_val = jnp.maximum(T_val, 1e3)
        logT = jnp.log10(T_val)
        logP = jnp.log10(P_val)
        rho, mu_l, nad, *_ = eos_lookup(logT, logP, X_surf, Z, helm_eos=helm_eos)
        log_kap = kappa(logT, jnp.log10(rho), X_surf, Z, bicubic_opacity=bicubic_opacity)
        kap = 10.0**log_kap
        g_local = G * M_star / r**2
        dln_P_dln_r = -rho * g_local * r / P_val
        nabla_rad = 3.0 * kap * L_s * P_val / (
            16.0 * jnp.pi * a_rad * c_light * G * M_star * T_val**4 + 1e-30)
        nabla = mlt_nabla(nabla_rad, nad, T_val, P_val, rho, kap, g_local, mu_l, alpha_mlt)
        dln_T_dln_r = nabla * dln_P_dln_r
        return dln_P_dln_r, dln_T_dln_r

    def _radial_step(carry, _):
        ln_r, ln_P, ln_T = carry
        P = jnp.exp(jnp.minimum(ln_P, 50.0))
        T = jnp.exp(jnp.minimum(ln_T, 30.0))
        k1_P, k1_T = _derivs(ln_r, P, T)
        ln_r_mid = ln_r + 0.5 * d_ln_r
        ln_P_mid = ln_P + 0.5 * d_ln_r * k1_P
        ln_T_mid = ln_T + 0.5 * d_ln_r * k1_T
        P_mid = jnp.exp(jnp.minimum(ln_P_mid, 50.0))
        T_mid = jnp.exp(jnp.minimum(ln_T_mid, 30.0))
        k2_P, k2_T = _derivs(ln_r_mid, P_mid, T_mid)
        ln_r_new = ln_r + d_ln_r
        ln_P_new = ln_P + d_ln_r * k2_P
        ln_T_new = ln_T + d_ln_r * k2_T
        return (ln_r_new, ln_P_new, ln_T_new), None

    (_, ln_P_end, ln_T_end), _ = lax.scan(
        _radial_step, (ln_R_star, jnp.log(P_atm), jnp.log(T_atm)), None, length=N_ENV)
    return ln_P_end, ln_T_end


def _surface_bc_atm(y_surf, M_star, X_surf, Z, alpha_mlt, T_eff_param=None, R_phot=None,
                    envelope_ratio=None, helm_eos=False, bicubic_opacity=True):
    """Two-equation surface BC at shallow fitting depth τ=10.

    Architecture:
      1. Atmosphere integrates in τ-space from τ≈0 to τ=100 (Krishna Swamy
         T(τ) + MLT), giving (P_atm, T_atm) at the photosphere.
      2. A radial RK2 bridge integrates inward from the photosphere (R_phot)
         to the Henyey mesh surface at r_surf (q≈0.993).
      3. The BC residual = lnP_mesh - lnP_bridge, lnT_mesh - lnT_bridge.

    References:
      - Kippenhahn, Weigert & Weiss 2012, §11.1 (fitting-point method)
      - Paxton et al. 2013, ApJS 208, 4, App. A.5 (MESA atm module)
      - Krishna Swamy 1966, ApJ 145, 174
    """
    ln_P_s, ln_T_s = y_surf[1], y_surf[2]
    ell_s = y_surf[3]
    L_s = ell_s * Lsun

    if T_eff_param is not None:
        T_eff = T_eff_param
        R_star = jnp.sqrt(L_s / (4.0 * jnp.pi * sigma_sb * T_eff**4))
    else:
        R_star = R_phot

    ln_P_end, ln_T_end = _atm_bridge_endpoint(y_surf, M_star, X_surf, Z, alpha_mlt, R_star,
                                               helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
    BC_P = ln_P_s - ln_P_end
    BC_T = ln_T_s - ln_T_end
    return BC_P, BC_T


def _surface_bc_atm_values(y_surf, M_star, X_surf, Z, alpha_mlt, R_phot,
                           helm_eos=False, bicubic_opacity=True):
    """Compute atmosphere bridge endpoint (ln_P, ln_T) at mesh surface.

    Returns the bridge endpoint values (not residuals). Called ONCE before
    Newton to pre-compute surface BC targets for the linearized approach.
    """
    return _atm_bridge_endpoint(y_surf, M_star, X_surf, Z, alpha_mlt, R_phot,
                                helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)


def _surface_bc_atm_values_with_ratio(y_surf, M_star, X_surf, Z, alpha_mlt, atm_ratio,
                                      helm_eos=False, bicubic_opacity=True):
    """Compute atmosphere bridge endpoint with R_phot derived from y_surf[0].

    R_phot = atm_ratio * exp(y_surf[0]) INTERNALLY, so jacfwd w.r.t. y_surf
    captures the full sensitivity including lnR → R_phot → bridge → (ln_P, ln_T).

    Reference: MESA star/private/hydro_eqns.f90:817 (get_PT_bc_ad) —
    dlnPsurf_dlnR and dlnTsurf_dlnR in the surface Jacobian.
    """
    R_phot = atm_ratio * jnp.exp(y_surf[0])
    return _surface_bc_atm_values(y_surf, M_star, X_surf, Z, alpha_mlt, R_phot,
                                  helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
