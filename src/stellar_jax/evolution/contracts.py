"""evolution/contracts.py — Shape contracts for the lax.scan carry and step context.

CarryState: NamedTuple documenting the scan carry shape — FOR DOCUMENTATION ONLY.
StepContext: frozen dataclass holding all former closure-captured variables.
Carry index constants: named integers replacing bare carry[N]/final_carry[N].

IMPORTANT: The live execution path in _core.py uses a PLAIN TUPLE (not CarryState)
because the carry is CONDITIONAL:
  - adaptive_mesh=True:  18-element tuple (with comp_mfracs at position 15,
    dt_old at 16, dt_limit_ratio_old at 17)
  - adaptive_mesh=False: 17-element tuple (no comp_mfracs; dt_old at 15,
    dt_limit_ratio_old at 16)

Using a SINGLE NamedTuple with always-18 fields changes the XLA compiled graph
for adaptive_mesh=False, shifting FP instruction scheduling enough to cause
borderline Newton convergence failures over 1000+ steps. This is the root cause
of why #525's decomposition was never adopted.

The CarryState below matches the adaptive_mesh=True shape (18 fields) and serves
as DOCUMENTATION + TYPE REFERENCE for tooling/tests. It is NOT used in the hot path.
"""
from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp


# =========================================================================
# Carry field indices — the live carry is a PLAIN TUPLE (see module docstring).
# These named constants replace bare carry[N]/final_carry[N] magic indices.
# The carry is CONDITIONAL on adaptive_mesh; H211b history fields shift.
# =========================================================================

# --- Common fields (same position regardless of adaptive_mesh) ---
CARRY_X_PROF = 0
CARRY_Y_PROF = 1
CARRY_Z_PROF = 2
CARRY_C12_PROF = 3
CARRY_C13_PROF = 4
CARRY_N14_PROF = 5
CARRY_X3_PROF = 6
CARRY_LOGL = 7
CARRY_LOGTE = 8
CARRY_T = 9
CARRY_DT = 10
CARRY_PREV_LOGT = 11
CARRY_PREV_LOGP = 12
CARRY_PREV_LOGRHO = 13
CARRY_Y_HENYEY = 14

# --- adaptive_mesh=True: 18-element carry (19 with q_mesh_prev tail) ---
CARRY_COMP_MFRACS = 15          # ONLY present when adaptive_mesh=True
CARRY_DT_OLD_ADAPTIVE = 16      # H211b history (adaptive)
CARRY_DT_LIMIT_RATIO_OLD_ADAPTIVE = 17

# --- adaptive_mesh=False: 17-element carry (18 with q_mesh_prev tail) ---
CARRY_DT_OLD_STATIC = 15        # H211b history (no adaptive mesh)
CARRY_DT_LIMIT_RATIO_OLD_STATIC = 16

# --- q_mesh_prev tail field (differentiable-replay mesh remap) ---
# Appended as the LAST carry element in BOTH branches so it is a safe
# tail-append (no existing named index shifts; _assemble_result reads only
# indices 0..15 by name and never touches the tail). Carries the PREVIOUS
# step's structure mesh q_{k-1} so the replay can remap the live y_henyey
# guess FROM q_{k-1} ONTO the current frozen mesh q_k before the solve,
# matching remap_henyey_state_numpy in the recorded adaptive forward.
CARRY_Q_MESH_PREV_ADAPTIVE = 18  # adaptive_mesh=True: 19-element carry
CARRY_Q_MESH_PREV_STATIC = 17    # adaptive_mesh=False: 18-element carry

# =========================================================================
# evolve_solar carry indices (calibration/solar.py — 10-element carry).
# Separate carry from the main evolution loop.
# =========================================================================
SOLAR_CARRY_X = 0
SOLAR_CARRY_Y = 1
SOLAR_CARRY_Z = 2
SOLAR_CARRY_LOGL = 3
SOLAR_CARRY_LOGTE = 4
SOLAR_CARRY_T = 5
SOLAR_CARRY_DT = 6
SOLAR_CARRY_LOGT = 7
SOLAR_CARRY_LOGP = 8
SOLAR_CARRY_LOGRHO = 9

# =========================================================================
# evolve_star_diagnostic carry indices (calibration/diagnostic.py — 9-element).
# =========================================================================
DIAG_CARRY_X = 0
DIAG_CARRY_Y = 1
DIAG_CARRY_LOGL = 2
DIAG_CARRY_LOGTE = 3
DIAG_CARRY_T = 4
DIAG_CARRY_DT = 5
DIAG_CARRY_LOGT = 6
DIAG_CARRY_LOGP = 7
DIAG_CARRY_LOGRHO = 8


class CarryState(NamedTuple):
    """Scan carry shape — DOCUMENTATION ONLY (see module docstring).

    18 fields matching the adaptive_mesh=True carry tuple.
    Field order matches the live code positions exactly:
      0: X_prof, 1: Y_prof, 2: Z_prof, 3: C12_prof, 4: C13_prof,
      5: N14_prof, 6: X3_prof, 7: logL, 8: logTe, 9: t, 10: dt,
      11: prev_logT, 12: prev_logP, 13: prev_logRho, 14: y_henyey,
      15: comp_mfracs (ONLY when adaptive_mesh=True),
      16: dt_old (H211b history), 17: dt_limit_ratio_old (H211b history)
    """
    X_prof: jnp.ndarray          # (N_COMP,) hydrogen mass fraction profile
    Y_prof: jnp.ndarray          # (N_COMP,) helium mass fraction profile
    Z_prof: jnp.ndarray          # (N_COMP,) metals mass fraction profile
    C12_prof: jnp.ndarray        # (N_COMP,) carbon-12 mass fraction
    C13_prof: jnp.ndarray        # (N_COMP,) carbon-13 mass fraction
    N14_prof: jnp.ndarray        # (N_COMP,) nitrogen-14 mass fraction
    X3_prof: jnp.ndarray         # (N_COMP,) helium-3 mass fraction
    logL: jnp.ndarray            # scalar — log10(L/L_sun)
    logTe: jnp.ndarray           # scalar — log10(T_eff)
    t: jnp.ndarray               # scalar — age in seconds
    dt: jnp.ndarray              # scalar — current timestep in seconds
    prev_logT: jnp.ndarray       # (N_s,) — previous step log10(T) for eps_grav Form C
    prev_logP: jnp.ndarray       # (N_s,) — previous step log10(P)
    prev_logRho: jnp.ndarray     # (N_s,) — previous step log10(rho)
    y_henyey: jnp.ndarray        # (N_s, 4) — Henyey state vector [ln_r, ln_P, ln_T, ell]
    comp_mfracs: jnp.ndarray     # (N_COMP,) — composition mesh grid positions
    # When adaptive_mesh=False, this field is ABSENT from the live tuple.
    # Its presence here is for the 18-field documentation shape only.
    dt_old: jnp.ndarray          # scalar — previous step's dt (for H211b controller)
    # MESA timestep.f90:2381 (filter_dt_next): s% dt_old = dt of previous step.
    # Initialized to 0 → triggers 1st-order fallback on first step.
    dt_limit_ratio_old: jnp.ndarray  # scalar — previous worst-offender ratio (for H211b)
    # MESA timestep.f90:2381: s% dt_limit_ratio_old. Initialized to 0.


@dataclass(frozen=True)
class StepContext:
    """All former closure-captured variables — DOCUMENTATION ONLY.

    The live step_fn is a closure that captures these from _evolve_star_jit's
    local scope. They are NOT passed explicitly (that would change the XLA graph).
    This dataclass documents what the closure captures, for tooling/reference.
    """
    # --- Traced (differentiable) parameters ---
    mass: jnp.ndarray              # M/M_sun
    Z: jnp.ndarray                 # metallicity
    alpha_mlt: jnp.ndarray         # mixing-length parameter
    f_ov: jnp.ndarray              # overshooting parameter
    M_star_cgs: jnp.ndarray        # mass in grams
    atm_ratio: jnp.ndarray         # R_phot / r_surf geometry ratio
    varcontrol_target: jnp.ndarray # adaptive dt target
    reject_threshold: jnp.ndarray  # = 4 × varcontrol_target
    t_max_sec: jnp.ndarray         # max age (seconds)
    dt_grow: jnp.ndarray           # timestep growth factor (= 1.5)
    dt_shrink: jnp.ndarray         # timestep shrink factor (= 0.5)
    delta_lgL_hard: jnp.ndarray    # structural hard limit |ΔlogL|
    delta_lgTe_hard: jnp.ndarray   # structural hard limit |ΔlogTeff|
    opacity_factor: jnp.ndarray    # multiplicative opacity knob (default 1.0)
    eps_nuc_factor: jnp.ndarray    # multiplicative nuclear rate knob (default 1.0)
    diffusion_factor: jnp.ndarray  # multiplicative diffusion knob (default 1.0)

    # --- Static (compile-time) parameters — NOT traced ---
    q_mesh: jnp.ndarray            # (N_s+1,) Lagrangian mass mesh (constant)
    N_s: int                       # number of Henyey shells
    fixed_dt: object               # float or None — fixed timestep mode
    freeze_schedule: bool          # frozen-schedule adjoint
    replay_schedule: bool          # schedule replay mode
    adaptive_mesh: bool            # adaptive composition mesh
    diffusion: bool                # element diffusion enabled
    z_feedback: bool               # Z-profile feedback
    has_idle_guard: bool           # idle-step gradient guard
    eps_grav_flag: bool            # eps_grav in structure (forward coupling)
    detach_eps_grav_carry: bool    # sever cross-step eps_grav gradient
    xh_cntr_limit: float           # XH soft limiter threshold (static)
    xh_cntr_hard_limit: float      # XH hard limiter threshold (static)
    delta_lgT_cntr_limit: float    # central logT change limit (MESA default: 0.01)
    # MESA timestep.f90:1367 (check_dlgT_cntr_change); controls.defaults:10878.
    delta_lgRho_cntr_limit: float  # central logRho change limit (MESA default: 0.05)
    # MESA timestep.f90:1412 (check_dlgRho_cntr_change); controls.defaults:10908.
