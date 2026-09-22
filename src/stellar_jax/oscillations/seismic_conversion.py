"""Canonical σ² ↔ ν conversion helpers — ONE definition, used everywhere.

The dimensionless eigenfrequency σ² relates to the cyclic frequency ν [µHz] via:

    σ² = ν² · factor,     factor = (2π · 1e-6)² · R³ / (G · M)

where R is the stellar radius [cm], M the stellar mass [g], and G the gravitational
constant [cm³ g⁻¹ s⁻²].  The factor carries units of s² and depends on the global
stellar parameters; when M or R change (e.g. under jax.grad w.r.t. mass), the factor
changes too.

**Gradient correctness:** the IFT adjoint in eigenfreq_from_coeffs correctly treats
`factor` as nondiff (it's the dimensionless eigenvalue σ² = ω²R³/(GM) that the
Cowling equations determine). The R³/(GM) scaling belongs in the *conversion* from
σ² to ν, outside the custom_vjp, where it must be **live** (traced) so that
jax.grad captures ∂factor/∂θ.  Freezing factor via stop_gradient drops the dominant
scaling term ν²·∂factor/∂θ from ∂(ν²)/∂θ.

This module provides the SINGLE canonical definitions used by all oscillation,
seismic, and inference code.  DRY: no hand-written copies of the formula.

References:
    Christensen-Dalsgaard (2008), Ap&SS 316, 113, §2.1 — dimensionless σ²
    Kjeldsen & Bedding (1995), A&A 293, 87 — Δν ∝ √(M/R³) scaling
"""

import jax.numpy as jnp
import numpy as np

from stellar_jax.config.constants import G as _G_CGS

# The constant pre-factor: (2π · 1e-6)² [s²], converting µHz to rad/s then squaring.
_TWO_PI_MICRO_SQ = (2.0 * np.pi * 1e-6) ** 2


def compute_factor(R, M, G=_G_CGS):
    """Compute the ν² → σ² conversion factor: (2π·1e-6)² · R³ / (G·M).

    Works with both NumPy scalars/arrays and JAX tracers. When R or M are
    JAX tracers (inside jax.grad), the returned factor is a tracer too,
    preserving the gradient path.

    Args:
        R: stellar radius [cm] — scalar or JAX tracer.
        M: stellar mass [g] — scalar or JAX tracer.
        G: gravitational constant [cm³ g⁻¹ s⁻²].  Default: config/constants.py G.

    Returns:
        factor (same type as R, M) — dimensionless conversion factor [s²].
            σ² = ν² · factor,  ν = √(σ² / factor).

    References:
        Christensen-Dalsgaard (2008), Ap&SS 316, 113, Eq. 2.1
    """
    return _TWO_PI_MICRO_SQ * R ** 3 / (G * M)


def nu_from_sigma2(sigma2, factor):
    """Convert dimensionless σ² to cyclic frequency ν [µHz].

    ν = √(σ² / factor).

    Both arguments may be JAX tracers; the gradient ∂ν/∂(σ², factor) is
    preserved.  When `factor` is live (not stop_gradient'd), jax.grad
    correctly captures the ν²·∂factor/∂θ scaling term.

    Args:
        sigma2: dimensionless eigenfrequency squared — scalar or array.
        factor: conversion factor from compute_factor() — scalar.

    Returns:
        ν [µHz] — same shape as sigma2.
    """
    return jnp.sqrt(sigma2 / factor)


def sigma2_from_nu(nu, factor):
    """Convert cyclic frequency ν [µHz] to dimensionless σ².

    σ² = ν² · factor.

    Args:
        nu: cyclic frequency [µHz] — scalar or array.
        factor: conversion factor from compute_factor() — scalar.

    Returns:
        σ² — same shape as nu.
    """
    return nu ** 2 * factor
