"""Composition operators: burn, mix, diffuse.

Extracted from evolution.py (issue #479, refactor P2). These are PURE functions —
they contain NO stop_gradient calls. The gradient policy (detaching structure,
freezing diffusion output, etc.) is enforced by the CALLER (step_fn in
evolution.py), matching MESA's operator-split architecture:

  - MESA struct_burn_mix.f90:82-100 — op_split_burn zeros d_epsnuc_dlnT/dlnd in
    the Jacobian, then calls do_burn with converged T/rho as frozen parameters.
  - MESA evolve.f90:634-639 — element diffusion called after do_struct_burn_mix,
    outside the Newton Jacobian.

See composition/contracts.py for the full VJP contract documentation.
"""

from stellar_jax.composition.burn import burn_composition, burn_cno
from stellar_jax.composition.mix import mix_composition, mix_z_in_cz
from stellar_jax.composition.diffuse import (
    diffuse_composition,
    thoul_burgers_coefficients,
    thoul_burgers_4species,
)
from stellar_jax.composition.burgers import solve_3species, solve_4species

__all__ = [
    "burn_composition",
    "burn_cno",
    "mix_composition",
    "mix_z_in_cz",
    "diffuse_composition",
    "thoul_burgers_coefficients",
    "thoul_burgers_4species",
    "solve_3species",
    "solve_4species",
]
