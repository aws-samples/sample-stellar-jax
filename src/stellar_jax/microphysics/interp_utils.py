"""Shared grid-location utilities for table interpolation (DRY extraction).

These helpers encapsulate the repeated clamp → searchsorted → fraction pattern
used across EOS and opacity 4D table lookups. Behavior-preserving: the numerical
results are bit-identical to the inlined versions they replace.

The pattern (common to MESA's kap/eos table lookup infrastructure):
  1. Clip point to [grid[0], grid[-1]]
  2. Index = clip(searchsorted(grid, point) - 1, 0, len(grid) - 2)
  3. Fraction = clip((point - grid[index]) / (grid[index+1] - grid[index]), 0, 1)

Reference: MESA kap/private/kap_eval_support.f90 (Do_Kap_Interpolations)
           MESA eos/private/eosdt_eval.f90 (locate grid cell + fraction)
"""
import jax.numpy as jnp


def _locate_1d(grid, point):
    """Locate a single point in a 1D grid: clamp, find cell index, compute fraction.

    Args:
        grid: 1D sorted grid array (jnp.ndarray).
        point: scalar value to locate.

    Returns:
        (clamped_point, index, fraction) where:
          - clamped_point: point clipped to [grid[0], grid[-1]]
          - index: integer cell index in [0, len(grid)-2]
          - fraction: normalized position within cell, clipped to [0, 1]
    """
    pt = jnp.clip(point, grid[0], grid[-1])
    n = grid.shape[0] - 2  # max valid index
    idx = jnp.clip(jnp.searchsorted(grid, pt) - 1, 0, n)
    frac = jnp.clip((pt - grid[idx]) / (grid[idx + 1] - grid[idx]), 0.0, 1.0)
    return pt, idx, frac


def _locate_4d(grids, points):
    """Locate a point in a 4D grid: clamp, find cell indices, compute fractions.

    This is the DRY extraction of the repeated preamble in _interp4d_logkappa,
    _interp4d_logkappa_bicubic, eos_lookup, and eos_lookup_bicubic.

    Args:
        grids: tuple of 4 sorted 1D grid arrays (g0, g1, g2, g3).
        points: tuple of 4 scalar values (p0, p1, p2, p3) to locate.

    Returns:
        (indices, fractions) where:
          - indices: tuple of 4 integer cell indices (i0, i1, i2, i3),
            each in [0, len(grid_k) - 2]
          - fractions: tuple of 4 normalized positions (t0, t1, t2, t3),
            each clipped to [0, 1]
    """
    g0, g1, g2, g3 = grids
    p0, p1, p2, p3 = points

    _, i0, t0 = _locate_1d(g0, p0)
    _, i1, t1 = _locate_1d(g1, p1)
    _, i2, t2 = _locate_1d(g2, p2)
    _, i3, t3 = _locate_1d(g3, p3)

    return (i0, i1, i2, i3), (t0, t1, t2, t3)
