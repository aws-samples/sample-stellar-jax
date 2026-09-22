"""Adaptive forward pass subpackage — non-differentiable NumPy-outer-loop
RGB forward path.

Public surface:
  evolve_star_adaptive, Trajectory, TrajectoryStep, shell_resolution,
  and the numpy mesh kernels that tests use.

Moved from package-root adaptive_forward.py into evolution/adaptive/ (#1115).
"""

from stellar_jax.evolution.adaptive.forward import (  # noqa: F401
    evolve_star_adaptive,
    # JIT kernels (used by tests)
    _jit_henyey_step,
    _jit_extract_shell_data,
    _jit_composition_update,
    # ZAMS init helpers (decomposition)
    _derive_surface_observables,
    _compute_initial_dt,
    _init_zams_state,
    # Step helpers (used by evolve_star_adaptive — re-exported for completeness)
    DELTA_LGL_HARD_LIMIT,
    DELTA_LGTE_HARD_LIMIT,
    _adaptive_compute_dt_next,
    _step_solve_structure,
    _step_composition_update,
    _init_timestep_config,
    _check_loop_termination,
    _maybe_activate_helm,
    _evaluate_rejection,
    _record_step,
    _handle_rejected_step,
    _build_evolution_summary,
)

from stellar_jax.evolution.adaptive.trajectory import (  # noqa: F401
    TrajectoryStep,
    Trajectory,
    shell_resolution,
)

from stellar_jax.evolution.adaptive.mesh_numpy import (  # noqa: F401
    MESH_WEIGHT,
    MESH_PARAM,
    MESH_T_FUNCTION1_WEIGHT,
    MESH_P_FUNCTION_WEIGHT,
    MESH_MIN_DQ,
    MESH_DELTA_GVAL_MAX,
    MESH_SMOOTH_ITERS,
    MAX_CONCENTRATION_MS,
    MAX_CONCENTRATION_RGB,
    compute_gval_numpy,
    equidistribute_mesh_numpy,
    _compute_slopes_mesa,
    conservative_remap_numpy,
    remap_henyey_state_numpy,
    make_unified_mesh,
    _step_remesh,
)
