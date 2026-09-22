"""Mesh density computation for structure and composition meshes.

Computes the density function (|dg/dq| or |dgval/dm|) that drives the
equidistribution — where density is high, more mesh nodes concentrate.

The density COMPUTATION differs between structure and composition meshes
(different mesh functions), but both feed into the same unified
equidistribution kernel (mesh/equidistribute.py).

Reference:
  - MESA mesh_functions.f90:307-321 (do1_xa_function: weight*log10(xa+param))
  - MESA controls.defaults: P_function_weight=40, T_function1_weight=110
  - Paxton et al. (2011), ApJS 192, 3, §7 (mesh functions)
"""
import jax.numpy as jnp

from stellar_jax.mesh.mesh_params import (
    MESH_WEIGHT_P, MESH_WEIGHT_T, MESH_DENSITY_FLOOR_FRAC,
    COMP_MESH_WEIGHT, COMP_MESH_PARAM, COMP_MESH_FLOOR_FRAC,
)


def compute_structure_density(y, q_mesh, w_P=MESH_WEIGHT_P, w_T=MESH_WEIGHT_T):
    """Compute mesh density from the current Henyey state.

    The mesh density ρ(q) is proportional to |dg/dq| where g is the
    MESA-weighted combination of log₁₀(P) and log₁₀(T):
      g(q) = w_P * log₁₀(P(q)) + w_T * log₁₀(T(q))

    This concentrates points where pressure and temperature change rapidly.

    Parameters
    ----------
    y : (N_s, 4) — Henyey state [ln_r, ln_P, ln_T, ell] at each interface.
    q_mesh : (N_s+1,) — current mass-coordinate mesh.
    w_P : float — weight for log₁₀(P) mesh function (MESA default: 40).
    w_T : float — weight for log₁₀(T) mesh function (MESA default: 110).

    Returns
    -------
    density : (N_s,) — mesh density at each interface (not normalized).
    """
    N_s = y.shape[0]
    ln10 = jnp.log(10.0)

    # Mesh function: g_k = w_P * log10(P_k) + w_T * log10(T_k)
    g = w_P * y[:, 1] / ln10 + w_T * y[:, 2] / ln10  # (N_s,)

    # Mesh density via centered finite differences of g w.r.t. q.
    dg = jnp.zeros(N_s)

    # Interior: centered difference
    dq_fwd = q_mesh[2:N_s] - q_mesh[:N_s - 2]
    dg_interior = jnp.abs(g[2:] - g[:-2]) / jnp.maximum(dq_fwd, 1e-30)

    # Boundary: one-sided
    dq_0 = q_mesh[1] - q_mesh[0]
    dg_0 = jnp.abs(g[1] - g[0]) / jnp.maximum(dq_0, 1e-30)

    dq_end = q_mesh[N_s - 1] - q_mesh[N_s - 2]
    dg_end = jnp.abs(g[N_s - 1] - g[N_s - 2]) / jnp.maximum(dq_end, 1e-30)

    dg = jnp.concatenate([dg_0[None], dg_interior, dg_end[None]])

    # Floor: minimum density = MESH_DENSITY_FLOOR_FRAC * mean(density)
    mean_dg = jnp.mean(dg)
    floor = MESH_DENSITY_FLOOR_FRAC * mean_dg
    density = jnp.maximum(dg, floor)

    return density


def compute_gval(X_prof):
    """Compute the MESA log-gval mesh function from hydrogen mass fraction.

    gval(k) = COMP_MESH_WEIGHT * log10(X(k) + COMP_MESH_PARAM)

    This is the composition mesh function from MESA's do1_xa_function
    (mesh_functions.f90:307-321). The log transform concentrates zones
    where X is small (H-burning shell has X→0).

    Parameters
    ----------
    X_prof : (N_COMP,) — hydrogen mass fraction profile on current grid.

    Returns
    -------
    gval : (N_COMP,) — mesh function values at each zone.
    """
    return COMP_MESH_WEIGHT * jnp.log10(X_prof + COMP_MESH_PARAM)


def compute_composition_density(X_prof, comp_mfracs):
    """Compute mesh density from the composition profile (log-gval gradient).

    Density = |dgval/dm| at each node, concentrating zones where the
    hydrogen mass fraction changes rapidly (burning shells).

    Parameters
    ----------
    X_prof : (N_COMP,) — hydrogen mass fraction on current grid.
    comp_mfracs : (N_COMP,) — current grid positions (m/M coordinates).

    Returns
    -------
    density : (N_COMP,) — composition mesh density at each zone.
    """
    N = X_prof.shape[0]

    # gval = weight * log10(X + param)
    gval = compute_gval(X_prof)

    # |dgval/dm| via finite differences
    dm = jnp.diff(comp_mfracs)  # (N-1,)
    dm = jnp.maximum(dm, 1e-30)

    dgval = jnp.diff(gval)  # (N-1,)
    density_mid = jnp.abs(dgval) / dm  # (N-1,)

    # Interpolate to nodes: average adjacent midpoint densities
    density = jnp.zeros(N)
    density = density.at[0].set(density_mid[0])
    density = density.at[1:-1].set(0.5 * (density_mid[:-1] + density_mid[1:]))
    density = density.at[-1].set(density_mid[-1])

    # Floor: use max(mean, 1.0) so flat profiles produce uniform grids
    mean_density = jnp.mean(density)
    floor = COMP_MESH_FLOOR_FRAC * jnp.maximum(mean_density, 1.0)
    density = jnp.maximum(density, floor)

    return density


def smooth(density, n_iters):
    """Smooth mesh density with 3-point neighbor averaging.

    MESA smooths delta_gval_max with neighbor averaging
    (adjust_mesh_support.f90, mesh_delta_coeff_factor_smooth_iters).
    We smooth the density itself, achieving the same effect.

    Uses an unrolled loop (not lax.scan) to avoid nested scan inside
    the outer evolution lax.scan — prevents XLA graph inflation.

    Parameters
    ----------
    density : (N,) — raw mesh density.
    n_iters : int — number of smoothing passes (MESA default: 3).

    Returns
    -------
    smoothed : (N,) — smoothed mesh density.
    """
    d = density
    for _ in range(n_iters):
        d_left = jnp.concatenate([d[:1], d[:-1]])
        d_right = jnp.concatenate([d[1:], d[-1:]])
        d = (d_left + d + d_right) / 3.0
    return d
