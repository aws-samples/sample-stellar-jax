"""Block-tridiagonal Newton (Henyey) solver for stellar structure.

BACKWARD-COMPATIBILITY SHIM — the implementation has moved to solver/.

All public APIs are re-exported from their new locations in the solver/ package
(issue #503, Phase P4-4). This file exists so that existing imports from
`evolution.py`, `tests/validate.py`, and other callers continue to work without
modification. New code should import from `solver` or its submodules directly.

Module structure (per the module redesign spec):
  solver/
  ├── __init__.py         — Re-exports public API
  ├── contracts.py        — Differentiability contract + param-name assertion
  ├── thomas.py           — Block-Thomas linear algebra
  ├── residual.py         — Cell residual + full residual assembly
  ├── jacobian.py         — Jacobian blocks (jacfwd-based)
  ├── damping.py          — Per-zone damping
  ├── conditioning.py     — Row equilibration + Levenberg + Armijo
  ├── surface_bc.py       — Atmosphere bridge + linearized BCs
  ├── newton.py           — Newton loops + @custom_vjp (ZAMS, static, eps_grav)
  ├── adjoint.py          — IFT backward pass helpers (decomposed)
  ├── continuation.py     — Production @custom_vjp boundary + entry points
  ├── shell_data.py       — Post-solve extraction
  └── eps_grav.py         — eps_grav Form C + energy-row scaling
"""

# ═══════════════════════════════════════════════════════════════
# Re-exports from solver/ package
# ═══════════════════════════════════════════════════════════════

# Phase P4-1: leaf functions
from stellar_jax.solver.thomas import block_thomas_solve, _adjoint_thomas_solve
from stellar_jax.solver.eps_grav import _eps_grav_form_c, _energy_row_scale_factors
from stellar_jax.solver.damping import _apply_per_zone_damping
from stellar_jax.solver.conditioning import conditioned_solve, armijo_line_search, _TOL_MAX_CORRECTION

# Phase P4-2: residual + jacobian
from stellar_jax.solver.residual import (
    _cell_residual, _cell_residual_raw,
    _build_residual, _build_residual_fixed_bc, _build_residual_atm,
)
from stellar_jax.solver.jacobian import (
    _jacobian_blocks, _jacobian_blocks_fixed_bc, _jacobian_blocks_atm,
)

# Phase P4-3: surface BC
from stellar_jax.solver.surface_bc import (
    _surface_bc_atm, _surface_bc_atm_values, _surface_bc_atm_values_with_ratio,
)

# Phase P4-4: Newton + continuation (the @custom_vjp boundaries)
from stellar_jax.solver.newton import (
    henyey_solve, _henyey_newton, henyey_solve_differentiable,
    _henyey_continuation_ift, _henyey_eps_grav_ift_impl,
    henyey_solve_from_state,
)
from stellar_jax.solver.continuation import (
    _henyey_continuation_atm, henyey_solve_from_state_atm,
    henyey_init_from_state_atm,
)

# Phase P4-5: adjoint helpers (exposed for tests)
from stellar_jax.solver.adjoint import (
    _solve_adjoint_system, _vjp_residual_params,
    _atmosphere_correction, _convergence_gate_outputs,
)

# Phase P4-6: shell data
from stellar_jax.solver.shell_data import extract_shell_data_from_henyey

# Expose _HENYEY_CONV_TOL for tests that reference it
from stellar_jax.solver.newton import _HENYEY_CONV_TOL
