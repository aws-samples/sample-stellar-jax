"""evolution/ — lax.scan driver + per-step orchestration + adaptive timestepping.

This package owns:
- The 3-phase gradient-windowed scan driver (scan.py)
- The per-step orchestrator (step_fn closure inside _core._evolve_star_jit)
- Per-step helpers: Henyey solve, composition, diagnostics, timestep (step.py)
- ZAMS initialization + warmup (zams.py)
- ALL stop_gradient policy annotations (except mesh grid-position policy 2)
- CarryState shape contract (in contracts.py, for typing/documentation)
- Adaptive timestepping (varcontrol, limiters, accept/reject) (timestep.py)
- Frozen config dataclasses (config.py)

Module structure (#1089 decomposition):
  _core.py      — orchestrator: evolve_star, _evolve_star_jit + thin step_fn
                   closure, observable_at_target, re-exports, composition helpers
  zams.py       — ZAMS init + warmup: _init_params_and_profiles,
                   _init_zams_and_warmup, _egrav_warmup, _extract_warmup_observables
  step.py       — per-step helpers: _henyey_step_solve, _step_composition_update,
                   _step_remap_to_static, _step_solve_and_eps, _step_diagnostics,
                   _step_timestep_decide, _step_dt_limiters, _step_compute_eps_grav_frac,
                   _step_adaptive_mesh_remap
  scan.py       — scan orchestration: _build_init_carry, _run_windowed_scan,
                   _assemble_result
  timestep.py   — H211b controller + central limiters
  config.py     — frozen config dataclasses (SolverNumerics, GradientOptions, etc.)
  contracts.py  — carry index constants + CarryState/StepContext documentation

CARRY SHAPE NOTE: _core.py uses a CONDITIONAL carry tuple to match main's XLA
graph structure exactly: 17-element tuple when adaptive_mesh=False, 18-element
tuple when adaptive_mesh=True. Using a SINGLE 18-field CarryState NamedTuple
changes the XLA compiled graph for adaptive_mesh=False, shifting FP instruction
scheduling enough to cause borderline Newton convergence failures over 1000+ steps.
Python function calls are invisible to XLA tracing — the decomposition into
zams.py/step.py/scan.py produces the SAME compiled graph as the monolithic version.
"""
from stellar_jax.evolution.contracts import CarryState, StepContext

# _core.py is the LIVE execution path — its closure-based step_fn produces
# the proven numerical trajectory matching main. Import evolve_star from it.
import stellar_jax.evolution._core as _core

evolve_star = _core.evolve_star
GradientTrustWarning = _core.GradientTrustWarning
GradientTrustError = _core.GradientTrustError
GRAD_TRUST_STEPS = _core.GRAD_TRUST_STEPS

# Re-export everything else from _core.py for backward compat
# (evolve_star_comp, evolve_star_diagnostic, observable_at_target, etc.)
import sys as _sys

_this_module = _sys.modules[__name__]
for _name in dir(_core):
    if not _name.startswith('__') and _name != 'evolve_star':
        if not hasattr(_this_module, _name):
            setattr(_this_module, _name, getattr(_core, _name))

# Clean up temporary names
del _sys, _this_module, _name
