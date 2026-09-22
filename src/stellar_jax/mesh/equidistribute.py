"""Unified equidistribution kernel — shared by structure and composition meshes.

The core numerical operation (trapezoidal cumulative integral → linspace target →
interp inversion) is IDENTICAL between both meshes. Only what drives the density
differs (P+T gvals vs log(X+param)).

Algorithm (Paxton et al. 2011, ApJS 192, 3, §7):
  MESA redistributes mesh points so each zone spans approximately equal change
  in the mesh function. For fixed-N (JIT-compatible):
    1. Trapezoidal integration of density over the grid: G(q) = ∫₀^q ρ(q') dq'
    2. Target values: G_target(k) = k/(N-1) * G(1) for k=0..N-1
    3. Invert: q_new(k) = interp(G_target(k), G, grid_old)

CONSTRAINT label: MESA uses a variable number of zones (split/merge cells to
bound delta_gval_max). We use fixed N (JAX requires static array shapes for
jit/lax.scan). The equidistribution achieves the same quality: each cell spans
approximately equal gval change.

Reference:
  - MESA mesh_plan.f90:pick1_dq (variable-N delta_gval_max bounding)
  - Paxton et al. (2011), ApJS 192, 3, §7
  - Paxton et al. (2013), ApJS 208, 4, §6
"""
import jax.numpy as jnp


def equidistribute_fixed_N(density, grid_old, N):
    """Equidistribute N nodes based on the given density function.

    Places N nodes such that the integral of density between consecutive
    nodes is equal. This is the fixed-N analogue of MESA's "each cell spans
    approximately equal change in gval" principle.

    Parameters
    ----------
    density : (N,) — mesh density at each node position.
    grid_old : (N,) or (N+1,) — current grid positions.
        For composition mesh: (N,) where N = N_COMP.
        For structure mesh: (N_s+1,) where the density has N_s values.
    N : int — number of output nodes (same as input for composition;
        N_s+1 for structure mesh).

    Returns
    -------
    grid_new : same shape as grid_old — new equidistributed positions on [0, 1].

    Notes
    -----
    The density must be strictly positive at all nodes (apply a floor before
    calling this function). The function uses trapezoidal integration for G(q)
    and linear interpolation for the inversion — both are standard for MESA-style
    mesh equidistribution on a fixed grid.
    """
    n_pts = grid_old.shape[0]
    n_density = density.shape[0]

    # Handle the case where density has one fewer element than grid_old
    # (structure mesh: N_s density values for N_s+1 grid points)
    if n_density < n_pts:
        # Extend density to match grid by repeating the last value
        density_full = jnp.concatenate([density, density[-1:]])
    else:
        density_full = density

    # Trapezoidal integration of density over the grid
    dq = jnp.diff(grid_old)  # (n_pts - 1,)
    avg_density = 0.5 * (density_full[:-1] + density_full[1:])  # (n_pts - 1,)
    dG = avg_density * dq  # (n_pts - 1,)
    G = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dG)])  # (n_pts,)

    # Target: equal spacing in G-space
    G_total = G[-1]
    G_total = jnp.maximum(G_total, 1e-30)  # safety for uniform profiles
    G_target = jnp.linspace(0.0, G_total, n_pts)

    # Invert by interpolation: grid_new(k) = interp(G_target(k), G, grid_old)
    grid_new = jnp.interp(G_target, G, grid_old)

    # Pin boundaries exactly
    grid_new = grid_new.at[0].set(0.0)
    grid_new = grid_new.at[-1].set(1.0)

    return grid_new
