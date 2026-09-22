"""Composition mesh: per-step in-scan equidistribution + conservative remap.

Implements MESA's composition-driven mesh redistribution for the fixed
N_COMP=200 composition grid, called inside the lax.scan step body.

Grid positions are stop_gradient'd (policy 2: algorithmic decisions);
species values stay differentiable through the conservative remap.

Reference:
  - MESA mesh_functions.f90:307-321 (do1_xa_function)
  - MESA mesh_plan.f90:1081-1246 (pick1_dq)
  - MESA mesh_adjust.f90:1203-1324 (do_xa)
  - Paxton et al. (2011), ApJS 192, 3, §7
  - Paxton et al. (2013), ApJS 208, 4, §6
"""
import jax
import jax.numpy as jnp

from stellar_jax.mesh.mesh_params import (
    COMP_MESH_SMOOTH_ITERS,
    COMP_MESH_GVAL_RANGE_THRESHOLD,
    COMP_MESH_CONCENTRATION_K,
    COMP_MESH_CONCENTRATION_THRESHOLD,
    COMP_MESH_DISPLACEMENT_THRESHOLD,
)
from stellar_jax.mesh.density import compute_gval, compute_composition_density, smooth
from stellar_jax.mesh.equidistribute import equidistribute_fixed_N
from stellar_jax.mesh.remap import remap_all_species


def equidistribute_comp_mesh(X_prof, comp_mfracs):
    """Compute equidistributed composition mesh from the hydrogen profile.

    Places N_COMP nodes so that each cell spans approximately equal change
    in gval = weight * log10(X + param). This is the fixed-N analogue of
    MESA's delta_gval_max criterion.

    Parameters
    ----------
    X_prof : (N_COMP,) — hydrogen mass fraction on current grid.
    comp_mfracs : (N_COMP,) — current grid positions (m/M coordinates).

    Returns
    -------
    comp_mfracs_new : (N_COMP,) — new equidistributed grid positions.
    """
    N = X_prof.shape[0]
    density = compute_composition_density(X_prof, comp_mfracs)
    density = smooth(density, n_iters=COMP_MESH_SMOOTH_ITERS)
    return equidistribute_fixed_N(density, comp_mfracs, N)


def should_remap(X_prof, comp_mfracs_old, comp_mfracs_new):
    """Determine whether the composition mesh remap should fire.

    Two criteria must BOTH be met:
    1. The composition has a sharp feature worth resolving (gval variation
       is concentrated in a few cell pairs, not distributed smoothly).
    2. The new grid differs meaningfully from the current grid (avoids
       cumulative numerical diffusion from near-identity remaps).

    Parameters
    ----------
    X_prof : (N_COMP,) — hydrogen mass fraction profile.
    comp_mfracs_old : (N_COMP,) — current grid positions.
    comp_mfracs_new : (N_COMP,) — proposed new grid positions.

    Returns
    -------
    do_remap : scalar bool — whether to apply the remap.
    """
    # Criterion 1: gval concentration
    gval = compute_gval(X_prof)
    dgval = jnp.abs(jnp.diff(gval))
    max_dgval_per_cell = jnp.max(dgval)

    # Concentration: fraction of total |Δgval| in the top K cell pairs
    total_dgval = jnp.sum(dgval) + 1e-30
    sorted_dgval = jnp.sort(dgval)[::-1]  # descending
    top_k_sum = jnp.sum(sorted_dgval[:COMP_MESH_CONCENTRATION_K])
    concentration = top_k_sum / total_dgval

    gval_sufficient = ((max_dgval_per_cell > COMP_MESH_GVAL_RANGE_THRESHOLD) &
                       (concentration > COMP_MESH_CONCENTRATION_THRESHOLD))

    # Criterion 2: grid displacement
    grid_displacement = jnp.max(jnp.abs(comp_mfracs_new - comp_mfracs_old))
    grid_moves_enough = grid_displacement > COMP_MESH_DISPLACEMENT_THRESHOLD

    return gval_sufficient & grid_moves_enough


def adapt_composition_mesh(X_prof, Y_prof, Z_prof, C12_prof, C13_prof, N14_prof,
                           comp_mfracs):
    """Full adaptive composition mesh step: equidistribute + remap.

    Called inside lax.scan on accepted steps (after burn+mix+diffusion).

    1. Compute equidistributed grid from X_prof (the driver species).
    2. Apply stop_gradient to new grid positions (POLICY 2: algorithmic).
    3. Check whether the remap should fire (concentration + displacement).
    4. Conservatively remap all species to the new grid (if firing).

    Parameters
    ----------
    X_prof, Y_prof, Z_prof, C12_prof, C13_prof, N14_prof : (N_COMP,)
        Composition profiles after burn+mix+diffusion. DIFFERENTIABLE.
    comp_mfracs : (N_COMP,) — current grid positions.

    Returns
    -------
    X_new, Y_new, Z_new, C12_new, C13_new, N14_new : (N_COMP,)
        Remapped composition profiles. DIFFERENTIABLE w.r.t. input profiles.
    comp_mfracs_new : (N_COMP,) — new grid (stop_gradient APPLIED, Policy 2).
    """
    # Step 1: equidistribute via the named function (patchable by mutations)
    comp_mfracs_new = equidistribute_comp_mesh(X_prof, comp_mfracs)

    # Step 2: stop_gradient on grid positions (POLICY 2)
    comp_mfracs_new = jax.lax.stop_gradient(comp_mfracs_new)  # GP-2: MESH_POSITIONS

    # Step 3: decide whether to remap
    do_remap = should_remap(X_prof, comp_mfracs, comp_mfracs_new)

    # Step 4: conservative remap of all species
    X_remap, Y_remap, Z_remap, C12_remap, C13_remap, N14_remap = remap_all_species(
        X_prof, Y_prof, Z_prof, C12_prof, C13_prof, N14_prof,
        comp_mfracs, comp_mfracs_new)

    # Select: remap only if criteria are met, else keep originals
    X_new = jnp.where(do_remap, X_remap, X_prof)
    Y_new = jnp.where(do_remap, Y_remap, Y_prof)
    Z_new = jnp.where(do_remap, Z_remap, Z_prof)
    C12_new = jnp.where(do_remap, C12_remap, C12_prof)
    C13_new = jnp.where(do_remap, C13_remap, C13_prof)
    N14_new = jnp.where(do_remap, N14_remap, N14_prof)
    comp_mfracs_out = jnp.where(do_remap, comp_mfracs_new, comp_mfracs)

    return X_new, Y_new, Z_new, C12_new, C13_new, N14_new, comp_mfracs_out
