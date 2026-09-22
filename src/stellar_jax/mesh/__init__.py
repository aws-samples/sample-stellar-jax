"""mesh/ — Structure-mesh + composition-mesh equidistribution + conservative remap.

Public API:
  adapt_structure_mesh      — structure mesh adaptation (P+T gvals, relax, remap)
  adapt_composition_mesh    — composition mesh adaptation (log-gval, conservative remap)
  initial_lagrangian_mesh   — initial static Lagrangian q-mesh
  conservative_remap        — order-1 (linear) mass-conserving remap

Legacy aliases (used by tests and evolution.py under old names):
  adaptive_mesh_reparameterize — old name for adapt_structure_mesh
  compute_mesh_density         — old name for compute_structure_density
  adaptive_comp_mesh_remap     — old name for adapt_composition_mesh

Differentiability contract (Policy 2):
  Grid POSITIONS are stop_gradient'd (algorithmic). VALUES on the mesh stay
  fully differentiable. See contracts.py for the full specification.
"""
from stellar_jax.mesh.structure_mesh import adapt_structure_mesh, initial_lagrangian_mesh
from stellar_jax.mesh.composition_mesh import adapt_composition_mesh
from stellar_jax.mesh.remap import conservative_remap
from stellar_jax.mesh.density import compute_structure_density, compute_gval

# Legacy aliases — preserve backward compatibility during transition
adaptive_mesh_reparameterize = adapt_structure_mesh
compute_mesh_density = compute_structure_density
adaptive_comp_mesh_remap = adapt_composition_mesh

__all__ = [
    'adapt_structure_mesh',
    'adapt_composition_mesh',
    'initial_lagrangian_mesh',
    'conservative_remap',
    'compute_structure_density',
    'compute_gval',
    # Legacy aliases
    'adaptive_mesh_reparameterize',
    'compute_mesh_density',
    'adaptive_comp_mesh_remap',
]
