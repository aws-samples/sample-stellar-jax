"""Analytic HELM-like EOS for fully-ionized degenerate matter with Coulomb corrections.

All thermodynamic quantities derived from a single Helmholtz free energy F(rho, T):
  P = rho^2 * dF/drho,  S = -dF/dT
guaranteeing Maxwell-relation consistency (Timmes & Swesty 2000; Jermyn+2021 Skye).

Components:
  1. Electrons: unified free energy interpolating between Chandrasekhar (degenerate)
     and ideal gas (non-degenerate), with Sommerfeld thermal corrections.
  2. Classical ideal ion gas
  3. Radiation (photon gas)
  4. Coulomb corrections: OCP free energy (Potekhin & Chabrier 2000)

References:
  - Timmes & Swesty (2000), ApJS 126, 501 (HELM EOS)
  - Chabrier & Potekhin (1998), Phys. Rev. E 58, 4941
  - Potekhin & Chabrier (2000), Phys. Rev. E 62, 8554
  - Jermyn et al. (2021), ApJ 913, 72 (Skye — differentiable EOS framework)
  - Chandrasekhar (1939), Introduction to Stellar Structure, Ch. 10
  - Kippenhahn, Weigert & Weiss (2012), Stellar Structure and Evolution, Ch. 15
"""
import jax
import jax.numpy as jnp

# --- Physical constants (CGS) — imported from single source ---
from stellar_jax.config.constants import (
    k_B_helm as k_B, m_H_helm as m_H, m_e_helm as m_e,
    hbar_helm as hbar, c_light_helm as c_light,
    e_charge_helm as e_charge, a_rad_helm as a_rad,
)
pi = jnp.pi


def _f_coul(Gamma):
    """OCP free energy per ion in units of kT: f(Gamma).

    Fit from Potekhin & Chabrier (2000), Phys. Rev. E 62, 8554, eq. 16.
    Valid for both weak (Gamma << 1, Debye-Hückel) and strong (Gamma >> 1) coupling.
    Single analytic expression — no regime-switching sigmoid needed.
    """
    Gamma_safe = jnp.maximum(Gamma, 1e-10)
    A1 = -0.9070
    A2 = 0.62849
    A3 = -jnp.sqrt(3.0) / 2.0  # Debye-Hückel coefficient
    B1 = -0.0045
    B2 = 170.0
    B3 = -8.4e-5

    term1 = A1 * (jnp.sqrt(Gamma_safe * (A2 + Gamma_safe))
                  - A2 * jnp.log(jnp.sqrt(Gamma_safe / A2) + jnp.sqrt(1.0 + Gamma_safe / A2)))
    sqrt_G = jnp.sqrt(Gamma_safe)
    term2 = 2.0 * A3 * (sqrt_G - jnp.arctan(sqrt_G))
    term3 = B1 * (Gamma_safe - B2 * jnp.log(Gamma_safe)) + B3
    return term1 + term2 + term3


_df_coul_dGamma = jax.grad(_f_coul)


def _composition(X, Z):
    """Mean molecular weights for fully ionized plasma.
    Reference: KWW (2012) eq. 13.5
    """
    Y = 1.0 - X - Z
    mu_e = 2.0 / (1.0 + X)
    mu_ion = 1.0 / (X + Y / 4.0 + Z / 16.0)
    mu = 1.0 / (1.0 / mu_e + 1.0 / mu_ion)
    Z_bar = mu_ion / mu_e
    A_bar = mu_ion
    return mu_e, mu_ion, mu, Z_bar, A_bar


def _F_electron(ln_rho, ln_T, X, Z):
    """Electron Helmholtz free energy per unit mass [erg/g].

    Unified free energy that smoothly interpolates between:
      - Degenerate limit (Theta << 1): Chandrasekhar + Sommerfeld correction
      - Non-degenerate limit (Theta >> 1): ideal classical electron gas

    Reference: Timmes & Swesty (2000) §2; Jermyn+2021 Skye §2.3
    """
    rho = jnp.exp(ln_rho)
    T = jnp.exp(ln_T)
    mu_e = 2.0 / (1.0 + X)

    n_e = rho / (mu_e * m_H)
    pF = hbar * (3.0 * pi**2 * n_e) ** (1.0 / 3.0)
    x = pF / (m_e * c_light)
    gamma_F = jnp.sqrt(1.0 + x**2)

    # --- Degenerate free energy ---
    A_ch = m_e**4 * c_light**5 / (24.0 * pi**2 * hbar**3)
    f_x = x * (2.0 * x**2 - 3.0) * gamma_F + 3.0 * jnp.arcsinh(x)
    g_x = 8.0 * x**3 * (gamma_F - 1.0) - f_x
    E_0_vol = A_ch * g_x
    e_0 = E_0_vol / rho  # per unit mass

    # Sommerfeld thermal correction
    E_F = m_e * c_light**2 * (gamma_F - 1.0)
    # Guard: E_F → 0 in the low-degeneracy (high-T, low-ρ) limit; 1e-50 keeps
    # Θ = k_B T / E_F finite for the Sommerfeld thermal correction and its AD gradient.
    E_F_safe = jnp.maximum(E_F, 1e-50)
    Theta = k_B * T / E_F_safe
    x_safe = jnp.maximum(x, 1e-10)
    rel_factor = jnp.minimum(gamma_F / x_safe**2, 100.0)
    w_som = jax.nn.sigmoid(3.0 * jnp.log(jnp.maximum(Theta, 1e-30)))
    F_som = -(pi**2 / 12.0) * (n_e * k_B * T / rho) * Theta * rel_factor * (1.0 - w_som)

    F_degen = e_0 + F_som

    # --- Non-degenerate (ideal) free energy ---
    nQ_e = (2.0 * pi * m_e * k_B * T / (2.0 * pi * hbar)**2) ** 1.5
    ln_ratio = jnp.log(jnp.maximum(n_e / nQ_e, 1e-30))
    F_ideal = (k_B * T / (mu_e * m_H)) * (ln_ratio - 1.0)

    # --- Smooth blend ---
    w = jax.nn.sigmoid(3.0 * jnp.log(jnp.maximum(Theta, 1e-30)))
    return (1.0 - w) * F_degen + w * F_ideal


def _F_ion(ln_rho, ln_T, X, Z):
    """Ion free energy per unit mass [erg/g] — classical ideal gas.
    Reference: KWW (2012) §13.1
    """
    rho = jnp.exp(ln_rho)
    T = jnp.exp(ln_T)
    _, _, _, _, A_bar = _composition(X, Z)
    n_ion = rho / (A_bar * m_H)
    m_ion = A_bar * m_H
    nQ_ion = (2.0 * pi * m_ion * k_B * T / (2.0 * pi * hbar)**2) ** 1.5
    ln_ratio = jnp.log(jnp.maximum(n_ion / nQ_ion, 1e-30))
    return (k_B * T / (A_bar * m_H)) * (ln_ratio - 1.0)


def _F_rad(ln_rho, ln_T, X, Z):
    """Radiation free energy per unit mass [erg/g].
    Reference: KWW (2012) §13.2
    """
    rho = jnp.exp(ln_rho)
    T = jnp.exp(ln_T)
    return -a_rad * T**4 / (3.0 * rho)


def _F_coul(ln_rho, ln_T, X, Z):
    """Coulomb free energy per unit mass [erg/g].
    F_coul = (kT / A_bar*mH) * f(Gamma)
    Reference: Potekhin & Chabrier (2000); Jermyn+2021 eq 13.
    """
    rho = jnp.exp(ln_rho)
    T = jnp.exp(ln_T)
    mu_e, _, _, Z_bar, A_bar = _composition(X, Z)
    n_e = rho / (mu_e * m_H)
    a_e = (3.0 / (4.0 * pi * n_e)) ** (1.0 / 3.0)
    Gamma_e = e_charge**2 / (a_e * k_B * T)
    Gamma = Gamma_e * Z_bar ** (5.0 / 3.0)
    Gamma_safe = jnp.maximum(Gamma, 1e-10)
    return (k_B * T / (A_bar * m_H)) * _f_coul(Gamma_safe)


def _F_total(ln_rho, ln_T, X, Z):
    """Total Helmholtz free energy per unit mass [erg/g]."""
    return (_F_electron(ln_rho, ln_T, X, Z) +
            _F_ion(ln_rho, ln_T, X, Z) +
            _F_rad(ln_rho, ln_T, X, Z) +
            _F_coul(ln_rho, ln_T, X, Z))


def _total_pressure(ln_rho, ln_T, X, Z):
    """Total pressure: P = rho * dF/d(ln_rho)."""
    rho = jnp.exp(ln_rho)
    dF_dlnrho = jax.grad(_F_total, argnums=0)(ln_rho, ln_T, X, Z)
    return rho * dF_dlnrho


def _total_entropy(ln_rho, ln_T, X, Z):
    """Total specific entropy: S = -(1/T)*dF/d(ln_T)."""
    T = jnp.exp(ln_T)
    dF_dlnT = jax.grad(_F_total, argnums=1)(ln_rho, ln_T, X, Z)
    return jnp.maximum(-dF_dlnT / T, 1e-10)


def _helm_composition_mu(X, Z):
    """Return mean molecular weight mu for the helm EOS ideal-gas initial guess."""
    _, _, mu, _, _ = _composition(X, Z)
    return mu


# --- Density inversion: delegates to shared density_inversion module ---
# helm.py is NOT on the per-step Newton hot path (eos_lookup calls fd_electron.py
# for the FD table path; helm_eos_full is only called in tests/standalone). The
# delegation does not affect XLA trace identity on any production gradient path.
from stellar_jax.microphysics.density_inversion import (
    _solve_density_core, make_solve_density_ift, compute_thermo_from_free_energy
)


def _solve_density(logT, logP_target, X, Z):
    """Newton-Raphson density inversion (forward only, no custom gradient).

    Given (logT, logP, X, Z), find ln_rho such that P(rho, T) = P_target.
    30 iterations with jax.grad Jacobian (exact, no FD truncation error).
    Reference: Timmes & Swesty (2000) §3.
    """
    return _solve_density_core(logT, logP_target, X, Z,
                               _total_pressure, _helm_composition_mu)


_solve_density_ift = make_solve_density_ift(_total_pressure, _helm_composition_mu)


def helm_eos_full(logT, logP_target, X, Z):
    """HELM EOS with P-based lookup (invert rho from P).

    Given (logT, logP, X, Z), find rho such that P(rho, T) = P_target,
    then return all thermodynamic quantities derived from the free energy.

    The NR density inversion uses @custom_jvp with the implicit function
    theorem (IFT). custom_jvp is compatible with BOTH jax.jacfwd (used by the
    Henyey block-tridiagonal Jacobian) and jax.grad/jacrev (used by evolve_star
    reverse-mode AD). No unrolling through 30 NR iterations in the backward pass.

    All output thermodynamic derivatives (chi_rho, chi_T, cv, cp, nad) are
    computed via jax.grad of the free energy at the converged point.

    Reference: Timmes & Swesty (2000); Jermyn+2021 Skye §2.3
    """
    T = 10.0**logT
    ln_T = jnp.log(T)

    _, _, mu, _, _ = _composition(X, Z)

    # Solve for density using IFT-based custom_jvp (works in both forward and
    # reverse mode AD). The custom_jvp rule applies the implicit function theorem
    # at convergence, avoiding gradient unrolling through NR iterations.
    ln_rho = _solve_density_ift(logT, logP_target, X, Z)

    return compute_thermo_from_free_energy(
        ln_rho, ln_T, X, Z, mu, _total_pressure, _total_entropy)
