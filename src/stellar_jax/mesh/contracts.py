"""Contracts for the mesh/ public API.

Documents the typed signatures, differentiability contract, and invariants
for all public mesh functions. This is the interface specification — callers
depend on this contract, not on internals.

Differentiability Contract (Policy 2):
  - Grid positions (q_mesh, comp_mfracs) are stop_gradient'd: node placement
    is an algorithmic optimization, not a physical derivative.
  - VALUES on the mesh (y, X_prof, ...) remain fully differentiable: gradients
    flow through the interpolation / remap operations.
  - No @custom_vjp / @custom_jvp in this module. The stop_gradient on grid
    positions is a simple jax.lax.stop_gradient() call.

Why mesh/ owns stop_gradient (exception to orchestrator-owns rule):
  Grid equidistribution is an iterative algorithmic choice (not physics).
  The derivative of "where to place nodes" w.r.t. physics quantities is
  meaningless. The orchestrator shouldn't need mesh internals to apply the
  detachment — mesh/ IS the authority on what is algorithmic vs what carries
  gradient.
"""
from jax import Array


def adapt_structure_mesh(
    y: Array,             # (N_s, 4) — Henyey state. DIFFERENTIABLE.
    q_mesh: Array,        # (N_s+1,) — current mesh. stop_gradient from prior step.
    relax_factor: float = 0.005,
    w_P: float = 40.0,
    w_T: float = 110.0,
) -> tuple:
    """Adapt the structure mesh and remap Henyey state.

    Returns (y_new, q_mesh_new):
      y_new : (N_s, 4) — DIFFERENTIABLE w.r.t. y.
      q_mesh_new : (N_s+1,) — stop_gradient APPLIED (Policy 2).
    """
    ...


def adapt_composition_mesh(
    X_prof: Array,        # (N_COMP,) DIFFERENTIABLE
    Y_prof: Array,        # (N_COMP,) DIFFERENTIABLE
    Z_prof: Array,        # (N_COMP,) DIFFERENTIABLE
    C12_prof: Array,      # (N_COMP,) DIFFERENTIABLE
    C13_prof: Array,      # (N_COMP,) DIFFERENTIABLE
    N14_prof: Array,      # (N_COMP,) DIFFERENTIABLE
    comp_mfracs: Array,   # (N_COMP,) — current grid. stop_gradient from prior step.
) -> tuple:
    """Equidistribute composition mesh + conservative remap.

    Returns (X_new, Y_new, Z_new, C12_new, C13_new, N14_new, comp_mfracs_new):
      All species: (N_COMP,) DIFFERENTIABLE w.r.t. input profiles.
      comp_mfracs_new: (N_COMP,) — stop_gradient APPLIED (Policy 2).
    """
    ...


def initial_lagrangian_mesh(N: int) -> Array:
    """Generate the initial static Lagrangian mass mesh.

    q(ξ) = 0.30·ξ³ + 0.15·[1-(1-ξ)³] + 0.55·ξ,  ξ = k/N, k=0..N
    Returns (N+1,) on [0, 1].
    """
    ...


def conservative_remap(
    profile: Array,       # (N,) — values. DIFFERENTIABLE.
    grid_old: Array,      # (N,) — old positions. stop_gradient'd by caller.
    grid_new: Array,      # (N,) — new positions. stop_gradient'd by caller.
) -> Array:
    """Order-1 (piecewise-linear) conservative remap. Mass-preserving.

    Uses slope-limited linear reconstruction (MESA get1_lpp, order=1)
    and conservative integration across new cell boundaries.

    Returns (N,) — DIFFERENTIABLE w.r.t. profile.
    """
    ...
