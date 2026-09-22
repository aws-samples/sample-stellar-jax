"""IFT adjoint for eigenfrequencies via @custom_vjp.

This is the FREQUENCY DIFFERENTIABILITY BOUNDARY. The eigenfrequency σ² is
found by Brent root-finding (non-differentiable), but the implicit function
theorem gives:
    ∂σ²*/∂θ = −(∂D/∂θ) / (∂D/∂σ²)

where D(σ², θ) = 0 at the converged eigenfrequency and θ = (coeffs, x_grid).

CRITICAL JAX PITFALL — DO NOT reorder parameters:
  eigenfreq_radial: nondiff_argnums=(3, 4, 5) → args 3,4,5 = x_steps, h_steps, factor
  eigenfreq_nonradial: nondiff_argnums=(3, 4, 5, 6) → args 3,4,5,6 = l, x_steps, h_steps, factor

The @custom_vjp decorator + .defvjp() registration MUST stay in the same module
as the decorated function (JAX registers by function identity).

References:
    Christensen-Dalsgaard (2008), Ap&SS 316, 113 (ADIPLS: the forward eqns)
    Christensen-Dalsgaard, Lecture Notes on Stellar Oscillations, Ch. 5
        (variational principle / frequency sensitivity kernels)
    Blondel et al. (2022), "Efficient and Modular Implicit Differentiation"
"""
import functools

import jax
import jax.numpy as jnp

from .determinant import _radial_det_for_grad, _nonradial_det_for_grad


# ═══════════════════════════════════════════════════════════════════════════════
# Radial (l=0) IFT adjoint
# ═══════════════════════════════════════════════════════════════════════════════

@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def eigenfreq_radial(coeffs, x_grid, sigma2_converged, x_steps, h_steps, factor):
    """Return the converged σ² for a radial (l=0) mode with IFT adjoint.

    Forward: returns sigma2_converged unchanged.
    Backward: uses IFT ∂σ²*/∂θ = −(∂D/∂θ) / (∂D/∂σ²) where θ = coeffs.

    Args:
        coeffs: structure coefficients (5, N_grid) JAX array [Vg, A1, A_bv, U, q]
        x_grid: fractional radius grid (N_grid,) JAX array
        sigma2_converged: the eigenvalue found by Brent (scalar, JAX)
        x_steps: integration grid x-values (nondiff, static)
        h_steps: integration grid step sizes (nondiff, static)
        factor: ν² → σ² conversion factor (nondiff, static)

    Returns:
        sigma2_converged (scalar) — passes through the forward value, but
        backward pass provides ∂σ²/∂coeffs via IFT.
    """
    return sigma2_converged


def _eigenfreq_radial_fwd(coeffs, x_grid, sigma2_converged, x_steps, h_steps, factor):
    """Forward pass: return σ² and save residuals for backward."""
    sigma2 = eigenfreq_radial(coeffs, x_grid, sigma2_converged, x_steps, h_steps, factor)
    return sigma2, (coeffs, x_grid, sigma2_converged)


def _eigenfreq_radial_bwd(x_steps, h_steps, factor, res, g_sigma2):
    """Backward pass: IFT adjoint for the radial eigenfrequency.

    Given cotangent g_σ² on σ²*, returns cotangent on (coeffs, x_grid, σ²_init):
      ∂σ²*/∂θ = −(∂D/∂θ) / (∂D/∂σ²)  where θ = (coeffs, x_grid)
      g_θ = g_σ² · ∂σ²*/∂θ

    The x_grid gradient is essential for end-to-end ∂ν/∂M: when M changes,
    the structure radii r shift, so x = r/R changes, which shifts where the
    coefficients are interpolated onto the fixed integration mesh. Dropping
    this path loses ~75% of the gradient (diagnosed in OSC-3, issue #373).
    """
    coeffs, x_grid, sigma2_converged = res

    # Shape assertions (fail-fast on bad refactor — §5.1 of refactoring best practices)
    assert coeffs.ndim == 2 and coeffs.shape[0] == 5, (
        f"adjoint._eigenfreq_radial_bwd: coeffs shape {coeffs.shape}, expected (5, N)")
    assert x_grid.ndim == 1 and x_grid.shape[0] == coeffs.shape[1], (
        f"adjoint._eigenfreq_radial_bwd: x_grid shape {x_grid.shape} vs coeffs {coeffs.shape}")

    # ∂D/∂σ² at the converged point
    dD_dsigma2 = jax.grad(_radial_det_for_grad, argnums=0)(
        sigma2_converged, coeffs, x_grid, x_steps, h_steps)

    # ∂D/∂(coeffs, x_grid) jointly via vjp
    _, vjp_fn = jax.vjp(
        lambda c, xg: _radial_det_for_grad(sigma2_converged, c, xg, x_steps, h_steps),
        coeffs, x_grid)

    # IFT: ∂σ²*/∂θ = −(∂D/∂θ) / (∂D/∂σ²)
    # VJP: g_θ = g_σ² · ∂σ²*/∂θ = −g_σ² / (∂D/∂σ²) · ∂D/∂θ
    scale = -g_sigma2 / jnp.where(jnp.abs(dD_dsigma2) > 1e-30, dD_dsigma2, 1e-30)
    g_coeffs, g_x_grid = vjp_fn(scale)

    # No gradient on sigma2_converged (it's the input seed, overwritten by the root)
    g_sigma2_in = jnp.zeros_like(sigma2_converged)

    return g_coeffs, g_x_grid, g_sigma2_in


eigenfreq_radial.defvjp(_eigenfreq_radial_fwd, _eigenfreq_radial_bwd)


# ═══════════════════════════════════════════════════════════════════════════════
# Nonradial (l≥1) IFT adjoint
# ═══════════════════════════════════════════════════════════════════════════════

@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def eigenfreq_nonradial(coeffs, x_grid, sigma2_converged, l, x_steps, h_steps, factor):
    """Return the converged σ² for a nonradial (l≥1) mode with IFT adjoint.

    Forward: returns sigma2_converged unchanged.
    Backward: uses IFT ∂σ²*/∂θ = −(∂D/∂θ) / (∂D/∂σ²) where θ = coeffs.

    Args:
        coeffs: structure coefficients (5, N_grid) JAX array [Vg, A1, A_bv, U, q]
        x_grid: fractional radius grid (N_grid,) JAX array
        sigma2_converged: the eigenvalue found by Brent (scalar, JAX)
        l: angular degree (nondiff, static float)
        x_steps: integration grid x-values (nondiff, static)
        h_steps: integration grid step sizes (nondiff, static)
        factor: ν² → σ² conversion factor (nondiff, static)

    Returns:
        sigma2_converged (scalar) — passes through the forward value, but
        backward pass provides ∂σ²/∂coeffs via IFT.
    """
    return sigma2_converged


def _eigenfreq_nonradial_fwd(coeffs, x_grid, sigma2_converged, l, x_steps, h_steps, factor):
    """Forward pass: return σ² and save residuals for backward."""
    sigma2 = eigenfreq_nonradial(coeffs, x_grid, sigma2_converged, l, x_steps, h_steps, factor)
    return sigma2, (coeffs, x_grid, sigma2_converged)


def _eigenfreq_nonradial_bwd(l, x_steps, h_steps, factor, res, g_sigma2):
    """Backward pass: IFT adjoint for the nonradial eigenfrequency.

    Same IFT formula as radial, but uses nonradial_determinant.
    Includes x_grid gradient (same reasoning as radial — see OSC-3, #373).
    """
    coeffs, x_grid, sigma2_converged = res

    # Shape assertions (fail-fast on bad refactor — §5.1 of refactoring best practices)
    assert coeffs.ndim == 2 and coeffs.shape[0] == 5, (
        f"adjoint._eigenfreq_nonradial_bwd: coeffs shape {coeffs.shape}, expected (5, N)")
    assert x_grid.ndim == 1 and x_grid.shape[0] == coeffs.shape[1], (
        f"adjoint._eigenfreq_nonradial_bwd: x_grid shape {x_grid.shape} vs coeffs {coeffs.shape}")

    # ∂D/∂σ² at the converged point
    dD_dsigma2 = jax.grad(_nonradial_det_for_grad, argnums=0)(
        sigma2_converged, l, coeffs, x_grid, x_steps, h_steps)

    # ∂D/∂(coeffs, x_grid) jointly via vjp
    _, vjp_fn = jax.vjp(
        lambda c, xg: _nonradial_det_for_grad(sigma2_converged, l, c, xg, x_steps, h_steps),
        coeffs, x_grid)

    # IFT: g_θ = −g_σ² / (∂D/∂σ²) · (∂D/∂θ)^T
    scale = -g_sigma2 / jnp.where(jnp.abs(dD_dsigma2) > 1e-30, dD_dsigma2, 1e-30)
    g_coeffs, g_x_grid = vjp_fn(scale)

    g_sigma2_in = jnp.zeros_like(sigma2_converged)

    return g_coeffs, g_x_grid, g_sigma2_in


eigenfreq_nonradial.defvjp(_eigenfreq_nonradial_fwd, _eigenfreq_nonradial_bwd)


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch: radial vs nonradial
# ═══════════════════════════════════════════════════════════════════════════════

def eigenfreq_from_coeffs(coeffs, x_grid, sigma2_converged, l, x_steps, h_steps, factor):
    """Differentiable eigenfrequency: σ²(coeffs) with IFT adjoint.

    This is the function you differentiate through. Given structure coefficients
    and the converged σ², it returns σ² with a custom_vjp that provides
    ∂σ²/∂coeffs via the implicit function theorem.

    Usage:
        info = compute_eigenfreq_differentiable(glob, var, l=0, ...)
        grad_fn = jax.grad(lambda c: eigenfreq_from_coeffs(
            c, info['x_grid'], info['sigma2'], info['l'],
            info['x_steps'], info['h_steps'], info['factor']))
        dsigma2_dcoeffs = grad_fn(info['coeffs'])

    Args:
        coeffs: structure coefficients (5, N) — the differentiable input
        x_grid: fractional radius grid (N,)
        sigma2_converged: converged eigenvalue from Brent
        l: angular degree
        x_steps, h_steps: integration grid (static)
        factor: ν²→σ² conversion

    Returns:
        sigma2 (scalar) with IFT gradient attached
    """
    if l == 0:
        return eigenfreq_radial(coeffs, x_grid, sigma2_converged,
                                x_steps, h_steps, factor)
    else:
        return eigenfreq_nonradial(coeffs, x_grid, sigma2_converged,
                                   float(l), x_steps, h_steps, factor)
