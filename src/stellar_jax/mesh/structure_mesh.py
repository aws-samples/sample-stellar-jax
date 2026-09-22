"""Structure mesh: orchestration + initial Lagrangian mesh generator.

The structure mesh adapts the Henyey-solver grid positions based on P+T
gradients (MESA mesh functions). Grid positions are stop_gradient'd (policy 2:
algorithmic decisions); physics values on the mesh stay differentiable.

Reference:
  - MESA mesh_functions.f90 (P_function, T_function1)
  - MESA mesh_plan.f90 (equidistribution via delta_gval_max)
  - Paxton et al. (2011), ApJS 192, 3, §7
  - initial cubic mesh polynomial
"""
import jax
import jax.numpy as jnp

from stellar_jax.mesh.mesh_params import MESH_WEIGHT_P, MESH_WEIGHT_T, MESH_SMOOTH_ITERS
from stellar_jax.mesh.density import compute_structure_density, smooth
from stellar_jax.mesh.equidistribute import equidistribute_fixed_N
from stellar_jax.mesh.remap import remap_linear


def initial_lagrangian_mesh(N):
    """Generate the initial static Lagrangian mass mesh.

    q(ξ) = 0.30·ξ³ + 0.15·[1-(1-ξ)³] + 0.55·ξ,  ξ = k/N, k=0..N

    This provides surface-concentrated spacing: more zones near q=1 (surface)
    where the superadiabatic layer and atmosphere reside.

    Parameters
    ----------
    N : int — number of interior interfaces (returns N+1 total nodes).

    Returns
    -------
    q_mesh : (N+1,) — mass-coordinate mesh on [0, 1].
    """
    xi = jnp.linspace(0.0, 1.0, N + 1)
    q = 0.30 * xi**3 + 0.15 * (1.0 - (1.0 - xi)**3) + 0.55 * xi
    q = q.at[0].set(0.0)
    q = q.at[-1].set(1.0)
    return q


def adapt_structure_mesh(y, q_mesh, w_P=MESH_WEIGHT_P, w_T=MESH_WEIGHT_T,
                         smooth_iters=MESH_SMOOTH_ITERS,
                         relax_factor=0.005):
    """Adaptive structure mesh reparameterization: equidistribute + remap.

    Computes the ideal mesh for the current structure and returns both the
    new mesh coordinates and the state remapped onto them.

    Algorithm:
      1. Compute mesh density from the converged structure (gval gradient).
      2. Smooth the density (3-pass neighbor average, MESA-style).
      3. Equidistribute N points via cumulative-inverse.
      4. Relax toward the target (limits per-step mesh movement).
      5. Apply stop_gradient to new mesh coordinates (POLICY 2).
      6. Remap y onto the new mesh (linear interpolation in q).

    The relaxation (step 4) limits how fast the mesh converges to the
    equidistributed target. MESA uses mesh_delta_coeff to limit per-step
    gval changes analogously (adjust_mesh_support.f90).

    Parameters
    ----------
    y : (N_s, 4) — converged Henyey state. DIFFERENTIABLE.
    q_mesh : (N_s+1,) — current mesh (already stop_gradient from prior step).
    w_P : float — P_function weight (MESA default: 40).
    w_T : float — T_function1 weight (MESA default: 110).
    smooth_iters : int — density smoothing iterations (MESA default: 3).
    relax_factor : float — relaxation toward target (0=no change, 1=full).

    Returns
    -------
    y_new : (N_s, 4) — state remapped to new mesh. DIFFERENTIABLE w.r.t. y.
    q_mesh_new : (N_s+1,) — new mesh (stop_gradient APPLIED, Policy 2).
    """
    N_s = y.shape[0]

    # Step 1: compute mesh density
    density = compute_structure_density(y, q_mesh, w_P=w_P, w_T=w_T)

    # Step 2: smooth
    density = smooth(density, n_iters=smooth_iters)

    # Step 3: equidistribute to get TARGET mesh
    q_mesh_target = equidistribute_fixed_N(density, q_mesh, N_s + 1)

    # Step 4: relax toward target
    q_mesh_new = q_mesh + relax_factor * (q_mesh_target - q_mesh)
    q_mesh_new = q_mesh_new.at[0].set(0.0)
    q_mesh_new = q_mesh_new.at[-1].set(1.0)

    # Step 5: stop_gradient on node locations (POLICY 2)
    # Grid positions are algorithmic decisions, not physical derivatives.
    q_mesh_new = jax.lax.stop_gradient(q_mesh_new)  # GP-2: MESH_POSITIONS

    # Step 6: remap y to new mesh
    # Values stay differentiable through the interpolation.
    y_new = remap_linear(y, q_mesh, q_mesh_new)

    return y_new, q_mesh_new
