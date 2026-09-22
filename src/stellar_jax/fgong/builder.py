"""Differentiable FGONG builder — pure JAX, on the gradient path.

This module contains structure_to_fgong_jax, the differentiable bridge between
the Henyey solver state and the oscillation eigensolver. It takes the converged
Henyey state y_henyey and builds FGONG-layout JAX arrays (glob, var) that flow
into oscillations.build_oscillation_coeffs_jax.

Gradient chain: M → evolve_star → y_henyey → structure_to_fgong_jax →
                (glob, var) → build_oscillation_coeffs_jax → eigenfreq

No @custom_vjp: gradients flow through standard JAX AD (EOS lookups + array ops).
No stop_gradient owned by this module (policy lives in evolution/_core.py).

References
----------
Christensen-Dalsgaard (2008), Ap&SS 316, 113, §A.1 — FGONG format.
Paxton et al. (2013), ApJS 208, 4, §5 — MESA–ADIPLS coupling.
Kippenhahn, Weigert & Weiss (2012), ch. 13, eq. 13.22 — Gamma1.
MESA eospc_eval.f90:294-295 — Gamma1 = chi_rho / (1 - nabla_ad * chi_T).
"""

import jax.numpy as jnp
from jax import lax

from stellar_jax.config.constants import (
    G, a_rad, sigma_sb, Msun, Lsun,
)
from stellar_jax.config.mesh_defaults import (
    N_HENYEY, N_COMP,
)
from stellar_jax.config.physics_floors import PGAS_FRAC_FLOOR
from stellar_jax.fgong.contracts import (
    VAR_R, VAR_LN_Q, VAR_P, VAR_RHO, VAR_GAMMA1, VAR_A_STAR,
    GLOB_M, GLOB_R, GLOB_L, GLOB_Z, GLOB_X_SURF, GLOB_G,
)
from stellar_jax.microphysics.eos import eos_lookup
from stellar_jax.structure import interp_X_at_mass
from stellar_jax.mesh import initial_lagrangian_mesh


def structure_to_fgong_jax(M_solar, log_L, log_Te, X_profile, Z, t_age, alpha_mlt,
                           n_hires=None, y_henyey=None, q_mesh=None,
                           atm_ratio=None):
    """Build FGONG-layout structure arrays as live JAX arrays (no file I/O).

    This is the OSC-3 bridge: takes the converged Henyey state from evolve_star
    and builds FGONG-layout arrays as live JAX, so gradients flow through
    M → evolve_star → structure → ν(n,l).

    Returns (glob, var) where:
        glob: (15,) JAX array — FGONG global parameters [M, R, L, Z, X_surf, ...]
        var: (N_s, 15) JAX array — per-point structure (center-to-surface order)
            col 0: r, col 1: ln(m/M), col 3: P, col 4: rho, col 9: Gamma1, col 14: A*

    The column layout matches the standard FGONG format (Christensen-Dalsgaard 2008,
    Ap&SS 316, 113, §A.1) so that build_oscillation_coeffs_jax can process it.

    Works directly from the Henyey state y_henyey_final returned by evolve_star.
    Extracts (r, P, T) from the converged state, calls the EOS to get (rho, Gamma1),
    and builds the FGONG arrays — all differentiable.

    Self-consistency: when ``atm_ratio`` is provided, the FGONG radius
    ``glob[GLOB_R]`` and luminosity ``glob[GLOB_L]`` are derived DIRECTLY
    from ``y_henyey`` using the same formula as ``_step_solve_and_eps``.
    This makes the FGONG self-consistent by construction — ``x = r / R``
    is always physically bounded — regardless of whether the carry's
    ``log_L`` / ``log_Te`` are stale. Without ``atm_ratio``, the legacy
    Stefan-Boltzmann formula from ``log_L`` / ``log_Te`` is used (backward
    compat).

    MESA reference: pulse_fgong.f90:168 — ``r_outer = Rsun*s%photosphere_r``.
    MESA derives photosphere_r from the structural state via optical-depth
    interpolation (star_utils.f90:1093 set_phot_info → get_phot_info).
    CONSTRAINT: our ``atm_ratio`` is a constant computed once at ZAMS init
    (R_phot_init / r_surface_init); MESA recomputes photosphere_r at every
    step. This is a JAX constraint (avoids carrying the atmosphere solve
    through lax.scan).

    Parameters
    ----------
    M_solar : JAX scalar
        Stellar mass in solar masses.
    log_L : JAX scalar
        log10(L/L_sun). Used as fallback when atm_ratio is not provided.
    log_Te : JAX scalar
        log10(T_eff / K). Used as fallback when atm_ratio is not provided.
    X_profile : (N_COMP,) JAX array
        Hydrogen mass fraction profile.
    Z : JAX scalar
        Metal mass fraction.
    t_age : JAX scalar
        Stellar age in years (unused, kept for API compat).
    alpha_mlt : JAX scalar
        Mixing-length parameter (unused, kept for API compat).
    n_hires : int or None
        Ignored (uses N_HENYEY mesh). Kept for API compatibility.
    y_henyey : (N_s, 4) JAX array
        The converged Henyey state from evolve_star['y_henyey_final'].
        Columns: (ln_r, ln_P, ln_T, ell).
        REQUIRED for the differentiable path — this is the live JAX state.
    q_mesh : (N_s+1,) JAX array or None
        Lagrangian mass mesh face positions. When the Henyey solve used a
        non-default mesh (e.g. adapted mesh via mesh_schedule replay), pass
        that mesh here so the FGONG mass coordinates match the y_henyey
        structure. Defaults to the static Lagrangian mesh
        (initial_lagrangian_mesh(N_HENYEY)).
        MESA ref: pulse_fgong.f90 — FGONG uses the same mesh as the solver.
    atm_ratio : JAX scalar or None
        Photospheric radius ratio R_phot / r_surface. When provided, R_star
        and L_star are derived from y_henyey directly (self-consistent).
        Obtain from evolve_star(...)['atm_ratio'].

    References
    ----------
    Christensen-Dalsgaard (2008), Ap&SS 316, 113, §A.1 (FGONG format)
    Paxton et al. (2013), ApJS 208, 4, §5 (MESA–ADIPLS coupling)
    Kippenhahn, Weigert & Weiss (2012), ch. 13, eq. 13.22 (Gamma1)
    MESA pulse_fgong.f90:168 (r_outer = photosphere_r)
    """
    if y_henyey is None:
        raise ValueError(
            "y_henyey is required for the differentiable path (OSC-3). "
            "Pass evolve_star(...)['y_henyey_final'].")

    M_star_j = jnp.float64(M_solar) * Msun

    if atm_ratio is not None:
        # Self-consistent path: derive R_star and L_star from y_henyey
        # using the same formula as _step_solve_and_eps (evolution/step.py).
        # MESA reference: pulse_fgong.f90:168 — r_outer = Rsun*s%photosphere_r.
        # Note: atm_ratio is a stellar-jax concept (R_phot_init / r_surface),
        # held constant from ZAMS init. MESA recomputes photosphere_r each step
        # via star_utils.f90:1093 set_phot_info (CONSTRAINT: JAX/lax.scan).
        atm_ratio_j = jnp.float64(atm_ratio)
        R_star_j = atm_ratio_j * jnp.exp(y_henyey[-1, 0])  # R_phot from Henyey
        ell_surf = y_henyey[-1, 3]  # dimensionless luminosity L/L_sun
        L_star_j = ell_surf * Lsun
        Te_j = (L_star_j / (4.0 * jnp.pi * sigma_sb * R_star_j**2))**0.25
    else:
        # Legacy path: derive R_star from log_L and log_Te via Stefan-Boltzmann.
        # May be inconsistent with y_henyey if the carry coupling is broken.
        L_star_j = 10.0**jnp.float64(log_L) * Lsun
        Te_j = 10.0**jnp.float64(log_Te)
        R_star_j = jnp.sqrt(L_star_j / (4.0 * jnp.pi * sigma_sb)) / Te_j**2

    X_profile_j = jnp.asarray(X_profile, dtype=jnp.float64)
    Z_j = jnp.float64(Z)
    N_s = N_HENYEY

    # Extract structure from Henyey state: (ln_r, ln_P, ln_T, ell)
    r = jnp.exp(y_henyey[:, 0])       # radius (cm)
    P = jnp.exp(y_henyey[:, 1])       # total pressure
    T = jnp.exp(y_henyey[:, 2])       # temperature

    # Mass coordinate from the Lagrangian mesh — use provided mesh if given
    # (e.g. adapted mesh from mesh_schedule replay), else static default.
    # MESA ref: pulse_fgong.f90 — FGONG uses the same mesh as the solver.
    if q_mesh is None:
        q_mesh = initial_lagrangian_mesh(N_HENYEY)
    else:
        q_mesh = jnp.asarray(q_mesh, dtype=jnp.float64)
    Mr = q_mesh[:N_s] * M_star_j

    # EOS at each zone → (rho, Gamma1) via lax.scan
    Prad = a_rad * T**4 / 3.0
    Pgas = jnp.maximum(P - Prad, PGAS_FRAC_FLOOR * P)
    logT = jnp.log10(jnp.maximum(T, 1.0))
    logPgas = jnp.log10(jnp.maximum(Pgas, 1.0))

    def _scan_eos(carry, i):
        m_frac = q_mesh[i]
        X_l = interp_X_at_mass(X_profile_j, m_frac)
        rho_i, mu_i, nad_i, S_i, cp_i, chirho_i, chiT_i = eos_lookup(
            logT[i], logPgas[i], X_l, Z_j)
        # EOS-consistent Gamma1 from thermodynamic identity:
        #   Gamma1 = chi_rho / (1 - nabla_ad * chi_T)
        # Derived from MESA eospc_eval.f90:294-295.
        Gamma1_i = chirho_i / jnp.maximum(1.0 - nad_i * chiT_i, 1e-10)
        return carry, jnp.array([rho_i, Gamma1_i])

    _, eos_out = lax.scan(_scan_eos, None, jnp.arange(N_s))
    rho = eos_out[:, 0]
    Gamma1 = eos_out[:, 1]

    # Build FGONG-layout arrays
    X_surf = X_profile_j[N_COMP - 1]
    glob = jnp.zeros(15)
    glob = glob.at[GLOB_M].set(M_star_j)
    glob = glob.at[GLOB_R].set(R_star_j)
    glob = glob.at[GLOB_L].set(L_star_j)
    glob = glob.at[GLOB_Z].set(Z_j)
    glob = glob.at[GLOB_X_SURF].set(X_surf)
    glob = glob.at[GLOB_G].set(G)

    # var: (N_s, 15) center-to-surface
    var = jnp.zeros((N_s, 15))
    var = var.at[:, VAR_R].set(r)
    q_arr = jnp.clip(Mr / M_star_j, 1e-30, 1.0)
    var = var.at[:, VAR_LN_Q].set(jnp.log(q_arr))
    var = var.at[:, VAR_P].set(P)
    var = var.at[:, VAR_RHO].set(rho)
    var = var.at[:, VAR_GAMMA1].set(Gamma1)

    # Brunt-Väisälä discriminant A* = dlnP/dlnr / Gamma1 - dlnrho/dlnr
    lnP = jnp.log(jnp.maximum(P, 1e-30))
    lnrho = jnp.log(jnp.maximum(rho, 1e-30))
    dlnP_dr = jnp.gradient(lnP, r)
    dlnrho_dr = jnp.gradient(lnrho, r)
    A_star = dlnP_dr * r / Gamma1 - dlnrho_dr * r
    r_min_brunt = 1e-4 * R_star_j
    A_star = jnp.where(r > r_min_brunt, A_star, 0.0)
    var = var.at[:, VAR_A_STAR].set(A_star)

    return glob, var
