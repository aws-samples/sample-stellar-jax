"""High-resolution structure profile integrator — NOT on the gradient path.

RK4 integration on a 2500-zone high-res mesh (surface→center) for publication-
grade FGONG/GYRE files. Uses lax.scan for performance but outputs NumPy arrays.
Called by write_fgong and write_gyre only.

All thermodynamic quantities are EOS-consistent (not ideal-gas approximation).

References
----------
Kippenhahn, Weigert & Weiss (2012), ch. 13 — structure equations.
"""

import numpy as np
import jax.numpy as jnp
from jax import lax

from stellar_jax.config.constants import (
    G, a_rad, c_light, sigma_sb, Msun, Lsun,
)
from stellar_jax.config.mesh_defaults import (
    N_COMP,
)
from stellar_jax.config.physics_floors import PGAS_FRAC_FLOOR
from stellar_jax.microphysics.eos import eos_opal_only
# NOTE: uses eos_opal_only (pure OPAL) instead of the blended eos_lookup.
# hires_profile is a diagnostic FGONG writer called on converged models.
# The blended eos_lookup includes HELM (30-iter NR + jax.grad) which bloats
# the lax.scan XLA graph and causes compilation timeout for MS models where
# HELM adds nothing (w≈0 at logT<7.5). The production Henyey solver uses the
# full eos_lookup (with HELM blend) for RGB evolution; this diagnostic path
# does not need it. If RGB-tip FGONGs need HELM thermodynamics in the future,
# use the concrete-value eos_lookup in a non-scan path (eager loop).
from stellar_jax.microphysics.opacity import kappa
from stellar_jax.microphysics.nuclear import epsilon_nuclear
from stellar_jax.microphysics.neutrino import epsilon_neutrino
from stellar_jax.transport import mlt_nabla
from stellar_jax.structure import atmosphere_bc, interp_X_at_mass

from stellar_jax.fgong.contracts import N_HIRES


def hires_profile(M_solar, log_L, log_Te, X_profile, Z, t_age, alpha_mlt):
    """Integrate structure on high-res mesh via JAX lax.scan (fast).

    Returns dict of numpy arrays (length N_HIRES), ordered surface→center,
    plus M_star, R_star, L_star scalars.

    All thermodynamic quantities are EOS-consistent (not ideal-gas approx).
    Reference: Kippenhahn, Weigert & Weiss (2012), ch. 13.
    """
    M_star_j = jnp.float64(M_solar) * Msun
    L_star_j = 10.0**jnp.float64(log_L) * Lsun
    Te_j = 10.0**jnp.float64(log_Te)
    R_star_j = jnp.sqrt(L_star_j / (4.0 * jnp.pi * sigma_sb)) / Te_j**2

    X_profile_j = jnp.asarray(X_profile, dtype=jnp.float64)
    Z_j = jnp.float64(Z)
    t_age_j = jnp.float64(t_age)
    alpha_j = jnp.float64(alpha_mlt)

    X_surf = X_profile_j[N_COMP - 1]
    g_surf = G * M_star_j / R_star_j**2
    P_phot, T_phot = atmosphere_bc(
        Te_j, g_surf, X_surf, Z_j, alpha_j, L_star_j, M_star_j)

    # Build high-res radial grid (surface→center)
    r_inner = 0.005 * R_star_j
    xi = jnp.linspace(0.0, 1.0, N_HIRES + 1)
    r_grid = R_star_j - (R_star_j - r_inner) * xi**2

    state0 = jnp.array([P_phot, M_star_j, L_star_j, T_phot])

    def rk4_step(carry, i):
        st, _ = carry
        P, Mr, Lr, T = st
        r = r_grid[i]
        r_next = r_grid[i + 1]
        dr = r_next - r

        P = jnp.maximum(P, 1.0)
        T = jnp.maximum(T, 1.0e3)
        r_c = jnp.maximum(r, 1e-4 * R_star_j)
        Mr = jnp.maximum(Mr, 1e-10 * M_star_j)
        m_frac = Mr / M_star_j
        X_l = interp_X_at_mass(X_profile_j, m_frac)
        Prad = a_rad * T**4 / 3.0
        Pgas = jnp.maximum(P - Prad, PGAS_FRAC_FLOOR * P)
        logT_v = jnp.log10(T)
        rho_v, mu_v, nad_v, S_v, cp_v, chirho_v, chiT_v = eos_opal_only(
            logT_v, jnp.log10(Pgas), X_l, Z_j)
        log_kap = kappa(logT_v, jnp.log10(rho_v), X_l, Z_j)
        kap_v = 10.0**log_kap
        eps_v = epsilon_nuclear(rho_v, T, X_l, Z_j, t_age_j)
        eps_nu = epsilon_neutrino(rho_v, T, X_l, Z_j)
        g_local = G * Mr / r_c**2

        # EOS-consistent Gamma1 from thermodynamic identity:
        #   Gamma1 = chi_rho / (1 - nabla_ad * chi_T)
        # Derived from MESA's relation (eospc_eval.f90:294):
        #   gamma1 = chiT*(gamma3-1) + chiRho
        # combined with grad_ad = (gamma3-1)/gamma1 (line 295).
        # Avoids cv (not returned by eos_opal_only); uses only nad, chi_rho,
        # chi_T which are already available.
        # Ideal fully-ionized gas check: nad=0.4, chi_T=1, chi_rho=1
        #   => Gamma1 = 1/(1-0.4) = 5/3 ✓
        delta_v = chiT_v / jnp.maximum(chirho_v, 1e-6)
        Gamma1_v = chirho_v / jnp.maximum(1.0 - nad_v * chiT_v, 1e-10)

        # Temperature gradient
        dPdr = -rho_v * g_local
        dMdr = 4.0 * jnp.pi * rho_v * r_c**2
        dLdr = 4.0 * jnp.pi * rho_v * (eps_v - eps_nu) * r_c**2
        nabla_rad = 3.0 * kap_v * Lr * P / (
            16.0 * jnp.pi * a_rad * c_light * G * Mr * T**4 + 1e-30)
        nabla_v = mlt_nabla(nabla_rad, nad_v, T, P, rho_v, kap_v,
                            g_local, mu_v, alpha_j)
        dTdr = (T / P) * dPdr * nabla_v

        k1 = jnp.array([dPdr, dMdr, dLdr, dTdr])

        def _derivs(r_, st_):
            P_, Mr_, Lr_, T_ = st_
            P_ = jnp.maximum(P_, 1.0)
            T_ = jnp.maximum(T_, 1.0e3)
            r_ = jnp.maximum(r_, 1e-4 * R_star_j)
            Mr_ = jnp.maximum(Mr_, 1e-10 * M_star_j)
            mf_ = Mr_ / M_star_j
            X_ = interp_X_at_mass(X_profile_j, mf_)
            Prad_ = a_rad * T_**4 / 3.0
            Pgas_ = jnp.maximum(P_ - Prad_, PGAS_FRAC_FLOOR * P_)
            rho_, mu_, nad_, _, _, _, _ = eos_opal_only(
                jnp.log10(T_), jnp.log10(Pgas_), X_, Z_j)
            kap_ = 10.0**kappa(jnp.log10(T_), jnp.log10(rho_), X_, Z_j)
            eps_ = epsilon_nuclear(rho_, T_, X_, Z_j, t_age_j)
            enu_ = epsilon_neutrino(rho_, T_, X_, Z_j)
            g_ = G * Mr_ / r_**2
            dP_ = -rho_ * g_
            dM_ = 4.0 * jnp.pi * rho_ * r_**2
            dL_ = 4.0 * jnp.pi * rho_ * (eps_ - enu_) * r_**2
            nr_ = 3.0 * kap_ * Lr_ * P_ / (
                16.0 * jnp.pi * a_rad * c_light * G * Mr_ * T_**4 + 1e-30)
            n_ = mlt_nabla(nr_, nad_, T_, P_, rho_, kap_, g_, mu_, alpha_j)
            dT_ = (T_ / P_) * dP_ * n_
            return jnp.array([dP_, dM_, dL_, dT_])

        k2 = _derivs(r_c + 0.5 * dr, st + 0.5 * dr * k1)
        k3 = _derivs(r_c + 0.5 * dr, st + 0.5 * dr * k2)
        k4 = _derivs(r_c + dr, st + dr * k3)
        st_new = st + (dr / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        st_new = st_new.at[0].set(jnp.maximum(st_new[0], 1.0))
        st_new = st_new.at[3].set(jnp.maximum(st_new[3], 1.0e3))

        # Record all quantities at this shell
        shell_out = jnp.array([
            r_c, P, T, rho_v, Mr, jnp.maximum(Lr, 0.0),
            kap_v, eps_v, nad_v, nabla_v,
            Gamma1_v, delta_v, cp_v, chirho_v, chiT_v, X_l, mu_v])
        return (st_new, r_next), shell_out

    _, profile = lax.scan(rk4_step, (state0, r_grid[0]), jnp.arange(N_HIRES))

    # Convert to numpy dict
    profile_np = np.asarray(profile)
    keys = ['r', 'P', 'T', 'rho', 'Mr', 'Lr', 'kappa_val', 'eps_nuc',
            'nabla_ad', 'nabla', 'Gamma1', 'delta', 'cp', 'chi_rho',
            'chi_T', 'X_local', 'mu']
    out = {k: profile_np[:, i] for i, k in enumerate(keys)}

    M_star = float(M_star_j)
    R_star = float(R_star_j)
    L_star = float(L_star_j)
    return out, M_star, R_star, L_star
