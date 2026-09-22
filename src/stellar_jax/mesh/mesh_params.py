"""Mesh-specific constants for structure and composition meshes.

Consolidated from mesh.py:40-64 and composition_mesh.py:56-108.
Constants specific to the mesh algorithms live here; N_COMP and COMP_MFRACS
remain in config (constants.py) since they define the global grid shape.

Reference:
  - MESA controls.defaults: P_function_weight=40, T_function1_weight=110
  - MESA mesh_delta_coeff_factor_smooth_iters=3
  - MESA mesh_functions.f90:307-321 (xa_function_weight, xa_function_param)
  - MESA adjust_mesh_support.f90:96 (delta_gval_max=1.0)
"""

# === Structure mesh constants ===

# MESA default weights for the structure mesh functions.
# P_function_weight=40, T_function1_weight=110.
# Reference: mesa/star/defaults/controls.defaults L6086,L6099.
MESH_WEIGHT_P = 40.0
MESH_WEIGHT_T = 110.0

# Minimum mesh density floor (fraction of mean density).
# Prevents degenerate cells in uniform regions.
# MESA analog: max_dq / min_dq controls.
MESH_DENSITY_FLOOR_FRAC = 0.1

# Smoothing iterations for mesh density.
# MESA default: mesh_delta_coeff_factor_smooth_iters=3.
MESH_SMOOTH_ITERS = 3


# === Composition mesh constants ===

# MESA xa_function_weight (He4 default = 30) and xa_function_param (0.01).
# We use hydrogen (X) since it drives the H-burning shell structure.
# Reference: mesh_functions.f90:307-321 (do1_xa_function).
COMP_MESH_WEIGHT = 30.0
COMP_MESH_PARAM = 0.01

# Density floor for composition mesh (fraction of mean density).
COMP_MESH_FLOOR_FRAC = 0.05

# Smoothing iterations for composition mesh density.
COMP_MESH_SMOOTH_ITERS = 3

# Threshold for skipping the remap: maximum per-cell-pair |Δgval|
# must exceed this value for the profile to have a feature worth resolving.
# Prevents numerical diffusion from remaps on smooth profiles.
# MESA grounding: adjust_mesh_support.f90:96 (delta_gval_max=1.0 per cell pair).
COMP_MESH_GVAL_RANGE_THRESHOLD = 0.01

# Concentration criterion: fraction of total |Δgval| in top K cell pairs
# must exceed the threshold. Distinguishes step functions from smooth gradients.
# MESA analog: delta_gval_max per cell, mesh_plan.f90:1122.
COMP_MESH_CONCENTRATION_K = 5
COMP_MESH_CONCENTRATION_THRESHOLD = 0.5

# Grid displacement threshold for skipping near-identity remaps.
# Even when composition has structure, skip if the new grid barely differs
# from the old. Value: 0.002 in mass fraction (~40% of mean cell width).
# MESA analog: mesh_plan.f90 pick1_dq only moves cells violating delta_gval_max.
COMP_MESH_DISPLACEMENT_THRESHOLD = 0.002
