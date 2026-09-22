"""solver/ — Newton structure solver + IFT adjoint.

Module responsibility: Block-tridiagonal Henyey relaxation solver + IFT adjoint
at the converged fixed point. This is the STRUCTURE DIFFERENTIABILITY BOUNDARY.

Public API:
  - block_thomas_solve: O(N) block-tridiagonal solver
  - henyey_solve_from_state_atm: Production continuation solver (per-step)
  - henyey_init_from_state_atm: Cold-start ZAMS solver
  - henyey_solve_from_state: Legacy continuation (non-atm tests)
  - henyey_solve_differentiable: ZAMS differentiable solver
  - extract_shell_data_from_henyey: Post-solve diagnostic extraction
"""

# Phase P4-1: leaf extractions
from stellar_jax.solver.thomas import block_thomas_solve, _adjoint_thomas_solve, _transpose_block_tridiag
from stellar_jax.solver.eps_grav import _eps_grav_form_c, _energy_row_scale_factors
from stellar_jax.solver.damping import _apply_per_zone_damping
from stellar_jax.solver.conditioning import (
    conditioned_solve, armijo_line_search,
    _condition_system, _block_thomas_solve_conditioned,
    _TOL_MAX_CORRECTION, _TOL_RESIDUAL,
)

# Phase P4-2: residual + jacobian extraction
from stellar_jax.solver.residual import (
    _cell_residual, _cell_residual_raw,
    _build_residual, _build_residual_fixed_bc, _build_residual_atm,
    _prepare_cell_inputs, CellInputs,
)
from stellar_jax.solver.jacobian import (
    _jacobian_blocks, _jacobian_blocks_fixed_bc, _jacobian_blocks_atm,
)

# Phase P4-3: surface BC extraction
from stellar_jax.solver.surface_bc import (
    _surface_bc_atm, _surface_bc_atm_values, _surface_bc_atm_values_with_ratio,
)

# Phase P4-4: Newton + continuation (@custom_vjp boundaries)
from stellar_jax.solver.newton import (
    henyey_solve, _henyey_newton, henyey_solve_differentiable,
    _henyey_continuation_ift, _henyey_eps_grav_ift_impl,
    henyey_solve_from_state,
)
from stellar_jax.solver.continuation import (
    _henyey_continuation_atm, henyey_solve_from_state_atm,
    henyey_init_from_state_atm,
)

# Phase P4-5: adjoint helpers
from stellar_jax.solver.adjoint import (
    _solve_adjoint_system, _vjp_residual_params,
    _atmosphere_correction, _convergence_gate_outputs,
    _convergence_gate_simple,
)

# Phase P4-6: shell data extraction
from stellar_jax.solver.shell_data import extract_shell_data_from_henyey
