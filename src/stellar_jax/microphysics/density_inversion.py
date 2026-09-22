"""Shared density-inversion machinery for Helmholtz free-energy EOS modules.

Provides:
  - _solve_density_core: Newton-Raphson P → ρ inversion (forward only)
  - make_solve_density_ift: factory that creates a custom_jvp-wrapped NR solver
                            with IFT gradient rule
  - compute_thermo_from_free_energy: Cox & Giuli thermodynamic relations from
                                     a converged (ln_rho, ln_T)

These are parameterized by the thermodynamic function set (total_pressure_fn,
total_entropy_fn) and avoid the 100% copy-paste between helm.py's analytic
EOS and any future HELM-like modules (issue #790).

NOTE: fd_electron.py keeps its NR solver INLINE (not delegated here) to
preserve XLA trace identity on the per-step Newton hot path. Only helm.py
(which is NOT on the per-step Newton path) delegates to this module.

References:
  - Timmes & Swesty (2000), ApJS 126, 501 (HELM EOS — NR density inversion §3)
  - Griewank & Walther (2008) §15 (IFT for implicit solves)
  - Cox & Giuli (1968), Ch. 9 (Gamma3, Gamma1, nad, cp relations)
"""
import functools
from typing import Callable

import jax
import jax.numpy as jnp

# Physical constants — imported from single source
from stellar_jax.config.constants import k_B_helm as k_B, m_H_helm as m_H


def _solve_density_core(
    logT: jnp.ndarray,
    logP_target: jnp.ndarray,
    X: jnp.ndarray,
    Z: jnp.ndarray,
    total_pressure_fn: Callable,
    composition_fn: Callable,
) -> jnp.ndarray:
    """Newton-Raphson density inversion (forward only, no custom gradient).

    Given (logT, logP, X, Z), find ln_rho such that P(rho, T) = P_target.
    30 iterations with jax.grad Jacobian (exact, no FD truncation error).

    Parameters
    ----------
    logT : scalar — log10(T [K])
    logP_target : scalar — log10(P_target [dyn/cm²])
    X : scalar — hydrogen mass fraction
    Z : scalar — metallicity
    total_pressure_fn : callable (ln_rho, ln_T, X, Z) → P
    composition_fn : callable (X, Z) → mu (mean molecular weight)

    Returns
    -------
    ln_rho : scalar — natural log of converged density

    Reference: Timmes & Swesty (2000) §3.
    """
    T = 10.0**logT
    P_target = 10.0**logP_target
    ln_T = jnp.log(T)

    mu = composition_fn(X, Z)

    # Initial guess from ideal gas
    rho_ideal = P_target * mu * m_H / (k_B * T)
    ln_rho = jnp.log(jnp.maximum(rho_ideal, 1e-10))
    ln_rho = jnp.clip(ln_rho, jnp.log(1e-2), jnp.log(1e10))

    _dP_dlnrho = jax.grad(total_pressure_fn, argnums=0)

    def _nr_step(_, ln_rho_i):
        P_i = total_pressure_fn(ln_rho_i, ln_T, X, Z)
        dP = _dP_dlnrho(ln_rho_i, ln_T, X, Z)
        dP_safe = jnp.where(jnp.abs(dP) > 1e-30, dP, 1e-30)
        delta = jnp.clip(-(P_i - P_target) / dP_safe, -2.0, 2.0)
        return ln_rho_i + delta

    ln_rho = jax.lax.fori_loop(0, 30, _nr_step, ln_rho)
    return ln_rho


def make_solve_density_ift(
    total_pressure_fn: Callable,
    composition_fn: Callable,
):
    """Factory: creates an IFT-backed density-inversion function with custom_jvp.

    The returned function has signature (logT, logP_target, X, Z) → ln_rho
    and carries an IFT-based custom_jvp rule that avoids unrolling through the
    30 NR iterations in the backward/forward-mode pass.

    Parameters
    ----------
    total_pressure_fn : callable (ln_rho, ln_T, X, Z) → P
    composition_fn : callable (X, Z) → mu

    Returns
    -------
    solve_density_ift : function with @custom_jvp

    References:
      - Griewank & Walther (2008) §15
      - Blondel et al. (2022)
    """
    @functools.partial(jax.custom_jvp, nondiff_argnums=())
    def _solve_density_ift(logT, logP_target, X, Z):
        """Density inversion with IFT-based custom_jvp."""
        return _solve_density_core(logT, logP_target, X, Z,
                                   total_pressure_fn, composition_fn)

    @_solve_density_ift.defjvp
    def _solve_density_ift_jvp(primals, tangents):
        """Forward-mode IFT rule for density inversion.

        At convergence: P(ln_rho*, ln_T, X, Z) = P_target
        Differentiating implicitly:
          (∂P/∂ln_rho) * d(ln_rho) = dP_target - ∂P/∂ln_T * d(ln_T)
                                                 - ∂P/∂X * dX - ∂P/∂Z * dZ
        """
        logT, logP_target, X, Z = primals
        d_logT, d_logP, d_X, d_Z = tangents

        ln_rho = _solve_density_ift(logT, logP_target, X, Z)

        T = 10.0**logT
        ln_T = jnp.log(T)
        P_target = 10.0**logP_target

        dP_dlnrho = jax.grad(total_pressure_fn, argnums=0)(ln_rho, ln_T, X, Z)
        dP_dlnT = jax.grad(total_pressure_fn, argnums=1)(ln_rho, ln_T, X, Z)
        dP_dX = jax.grad(total_pressure_fn, argnums=2)(ln_rho, ln_T, X, Z)
        dP_dZ = jax.grad(total_pressure_fn, argnums=3)(ln_rho, ln_T, X, Z)

        dP_dlnrho_safe = jnp.where(jnp.abs(dP_dlnrho) > 1e-30, dP_dlnrho, 1e-30)

        d_ln_rho = (P_target * jnp.log(10.0) * d_logP
                    - dP_dlnT * jnp.log(10.0) * d_logT
                    - dP_dX * d_X
                    - dP_dZ * d_Z) / dP_dlnrho_safe

        return ln_rho, d_ln_rho

    return _solve_density_ift


def compute_thermo_from_free_energy(
    ln_rho: jnp.ndarray,
    ln_T: jnp.ndarray,
    X: jnp.ndarray,
    Z: jnp.ndarray,
    mu: jnp.ndarray,
    total_pressure_fn: Callable,
    total_entropy_fn: Callable,
) -> tuple:
    """Compute full thermodynamic quantities via Cox & Giuli (1968) relations.

    Given a converged (ln_rho, ln_T) and the free-energy-derived P/S functions,
    compute (rho, mu, nad, S, cp, chi_rho, chi_T).

    Parameters
    ----------
    ln_rho : scalar — natural log of density
    ln_T : scalar — natural log of temperature
    X : scalar — hydrogen mass fraction
    Z : scalar — metallicity
    mu : scalar — mean molecular weight
    total_pressure_fn : callable (ln_rho, ln_T, X, Z) → P
    total_entropy_fn : callable (ln_rho, ln_T, X, Z) → S

    Returns
    -------
    tuple: (rho, mu, nad, S_total, cp, chi_rho, chi_T)

    Reference: Cox & Giuli (1968), Ch. 9, §9.83-9.93
    """
    rho = jnp.exp(ln_rho)
    T = jnp.exp(ln_T)

    P_total = total_pressure_fn(ln_rho, ln_T, X, Z)
    dP_dlnrho = jax.grad(total_pressure_fn, argnums=0)(ln_rho, ln_T, X, Z)
    chi_rho = dP_dlnrho / jnp.maximum(P_total, 1e-30)
    dP_dlnT = jax.grad(total_pressure_fn, argnums=1)(ln_rho, ln_T, X, Z)
    chi_T = dP_dlnT / jnp.maximum(P_total, 1e-30)

    S_total = total_entropy_fn(ln_rho, ln_T, X, Z)
    dS_dlnT = jax.grad(total_entropy_fn, argnums=1)(ln_rho, ln_T, X, Z)
    # The maxima below (cv≥1e5, Γ₁≥0.1, χ_ρ≥0.01, cp≥1e5) are LOOSE NON-PHYSICAL
    # stability guards, NOT tuned physics: they only keep the thermodynamic
    # derivatives finite/positive where the differentiated free-energy fit is noisy
    # (true values are far from these floors: cv,cp ≳ 1e7 erg/g/K, Γ₁ ~ 1.0–1.67,
    # χ_ρ ~ 0.3–2). They never bind in the physical regime.
    cv = jnp.maximum(dS_dlnT, 1e5)

    # Cox & Giuli (1968) thermodynamic relations
    Gamma3_m1 = P_total * chi_T / (rho * T * jnp.maximum(cv, 1e5))
    Gamma1 = jnp.maximum(chi_rho + Gamma3_m1 * chi_T, 0.1)
    nad = Gamma3_m1 / Gamma1
    cp = cv * Gamma1 / jnp.maximum(chi_rho, 0.01)
    cp = jnp.maximum(cp, 1e5)

    return rho, mu, nad, S_total, cp, chi_rho, chi_T
