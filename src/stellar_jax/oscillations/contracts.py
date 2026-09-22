"""Oscillations module contracts — types + differentiability boundary.

Defines the typed interfaces for the oscillation pipeline:
  structure → coefficients → integration → determinant → eigenvalue (IFT adjoint)

References:
    Christensen-Dalsgaard (2008), Ap&SS 316, 113 (ADIPLS: FGONG layout)
    Aerts, Christensen-Dalsgaard & Kurtz (2010), §3.3 (structure coefficients)
"""
from typing import NamedTuple

import jax.numpy as jnp


class OscillationCoeffs(NamedTuple):
    """Dimensionless structure coefficients for the pulsation equations.

    Fields:
        x_grid: (N,) fractional radius — DIFFERENTIABLE (∂ν/∂structure)
        coeffs: (5, N) [Vg, A1, A_bv, U, q] — DIFFERENTIABLE
        factor: ν² → σ² conversion scalar (non-diff, depends only on M, R, G)
    """
    x_grid: jnp.ndarray   # (N,)
    coeffs: jnp.ndarray   # (5, N)
    factor: float          # scalar


class EigenfreqResult(NamedTuple):
    """Result of a differentiable eigenfrequency computation."""
    nu: float              # frequency in μHz (Python float, for display)
    sigma2: jnp.ndarray   # dimensionless σ² (JAX scalar) — THE differentiable output
    l: int                 # angular degree (static)


class IntegrationGrid(NamedTuple):
    """Fixed RK4 integration grid (static, nondiff)."""
    x_steps: jnp.ndarray  # (N_steps,) starting x-positions
    h_steps: jnp.ndarray  # (N_steps,) step sizes


# ─── Shape validation helpers ──────────────────────────────────────────────────

def validate_coeffs_shape(coeffs, x_grid):
    """Assert shape contract for oscillation coefficients.

    Raises AssertionError if the shapes don't match the contract:
      coeffs: (5, N) and x_grid: (N,) with matching N.
    """
    assert coeffs.ndim == 2, f"coeffs must be 2D, got shape {coeffs.shape}"
    assert coeffs.shape[0] == 5, (
        f"coeffs must have 5 rows [Vg, A1, A_bv, U, q], got {coeffs.shape[0]}")
    assert x_grid.ndim == 1, f"x_grid must be 1D, got shape {x_grid.shape}"
    assert coeffs.shape[1] == x_grid.shape[0], (
        f"coeffs columns ({coeffs.shape[1]}) must match x_grid length ({x_grid.shape[0]})")


def validate_fgong_arrays(glob, var):
    """Assert shape contract for FGONG arrays.

    FGONG layout (Christensen-Dalsgaard 2008, Ap&SS 316, §A.1):
      glob: shape (≥15,) — global parameters [M, R, L, Z, X_surf, ...]
      var: shape (N, ≥15) — per-point center-to-surface
           columns [r, ln(m/M), T, P, ρ, X, L, κ, ε_nuc, Γ₁, ..., A*=col14]
    """
    assert glob.ndim == 1, f"glob must be 1D, got shape {glob.shape}"
    assert glob.shape[0] >= 15, (
        f"glob must have ≥15 elements (FGONG standard), got {glob.shape[0]}")
    assert var.ndim == 2, f"var must be 2D, got shape {var.shape}"
    assert var.shape[1] >= 15, (
        f"var must have ≥15 columns (FGONG standard), got {var.shape[1]}")
