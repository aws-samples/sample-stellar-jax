"""Backward-compatible re-export shim for the adaptive forward pass.

The implementation moved to ``stellar_jax.evolution.adaptive`` (#1115).
This module re-exports all public symbols so that existing
``from stellar_jax.adaptive_forward import X`` statements continue to work.

Prefer importing from ``stellar_jax.evolution.adaptive`` directly.
"""

from stellar_jax.evolution.adaptive import (  # noqa: F401 — re-exported
    # Public API
    evolve_star_adaptive,
    shell_resolution,
    # Trajectory
    TrajectoryStep,
    Trajectory,
    # Mesh constants
    MESH_WEIGHT,
    MESH_PARAM,
    MESH_T_FUNCTION1_WEIGHT,
    MESH_P_FUNCTION_WEIGHT,
    MESH_MIN_DQ,
    MESH_DELTA_GVAL_MAX,
    MESH_SMOOTH_ITERS,
    MAX_CONCENTRATION_MS,
    MAX_CONCENTRATION_RGB,
    # Mesh functions
    compute_gval_numpy,
    equidistribute_mesh_numpy,
    _compute_slopes_mesa,
    conservative_remap_numpy,
    remap_henyey_state_numpy,
    make_unified_mesh,
    _step_remesh,
    # JIT kernels
    _jit_henyey_step,
    _jit_extract_shell_data,
    _jit_composition_update,
    # ZAMS init helpers
    _derive_surface_observables,
    _compute_initial_dt,
    _init_zams_state,
    # Step helpers
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
