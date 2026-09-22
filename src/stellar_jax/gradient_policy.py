"""Single-source gradient-policy contract for stellar-jax.

Every jax.lax.stop_gradient call in the codebase implements ONE of these 12
policies. The policy documents WHAT is detached and WHY. The enforcement rule:
each stop_gradient call MUST cite its policy number in a comment.

DIFFERENTIABLE (gradients flow):
    mass, Z, alpha_mlt, f_ov, Y_init, opacity_factor, eps_nuc_factor,
    diffusion_factor → logL, logTe, structure state, eigenfrequencies σ²

FROZEN (stop_gradient'd):
    Policies 1–9 below.

OWNERSHIP: stop_gradient calls belong to the ORCHESTRATOR (evolution/_core.py),
NOT inside leaf physics modules. SINGLE EXCEPTION: mesh/ owns Policy 2 (grid
positions are algorithmic by definition).

Usage:
    from stellar_jax.gradient_policy import GradPolicy

    # In evolution/_core.py:
    dt_next = jax.lax.stop_gradient(dt_next)  # Policy 1: TIMESTEP
"""

from enum import IntEnum


class GradPolicy(IntEnum):
    """The 12 stop_gradient policies governing stellar-jax differentiability.

    Each policy documents a category of quantities that are detached from the
    gradient tape via jax.lax.stop_gradient. Together they define the
    differentiability contract: what flows through grad, and what is frozen.
    """

    TIMESTEP = 1
    """dt decisions, varcontrol, reject flags, dt_next.

    Rationale: adaptive dt is a discrete algorithm; MESA never differentiates
    through its timestep controller (timestep.f90, Paxton+2013 §5.2).
    Location: evolution/_core.py (_step_timestep_decide, _step_dt_limiters, _init_zams_and_warmup).
    """

    MESH_POSITIONS = 2
    """q_mesh grid points, comp_mfracs grid node locations.

    Rationale: grid is algorithmic; species VALUES on the grid remain live.
    Location: mesh/ (self-owned — the exception), composition_mesh.py.
    """

    DIFFUSION = 3
    """Full diffuse_composition output (operator-split).

    Rationale: MESA operator-splits diffusion outside the Newton Jacobian
    (evolve.f90:634; Thoul, Bahcall & Loeb 1994). Straight-through estimator.
    Location: evolution/_core.py (_step_composition_update).
    """

    Z_CNO_ISOTOPES = 4
    """Z_mixed, C12_m, C13_m, N14_m carries.

    Rationale: secondary species; gradient path is through X, Y only.
    Location: evolution/_core.py (_step_composition_update, _init_zams_and_warmup).
    """

    STRUCTURE_TO_COMP = 5
    """shell_data_sg passed to composition operators.

    Rationale: one-way coupling (structure → composition, not reverse).
    Issue #447 targets making this two-way — policy would be REMOVED then.
    Location: evolution/_core.py (_henyey_step_solve, _step_solve_and_eps).
    """

    IDLE_GUARD = 6
    """Detach logL/logTe when step is idle (t_max reached, He-halt).

    Rationale: no physics happens; detaching prevents spurious gradients.
    Location: evolution/_core.py (step_fn in _evolve_star_jit, idle-step branch).
    """

    WINDOWED_SCAN = 7
    """Phases outside [k0, k1) fully detached.

    Rationale: gradient-budget control — backward cost ∝ (k1 - k0).
    Location: evolution/_core.py (_run_windowed_scan, 3-phase scan implementation).
    """

    EPSGRAV_HISTORY = 8
    """cur_logT, cur_logP, cur_logRho extraction from shell_data into carry.

    Rationale: the extraction of current-step profiles from shell_data (the
    Henyey solution) into the scan carry is detached. This prevents NaN from
    extract_shell_data_from_henyey's backward (through mlt_nabla/EOS on some
    XLA platforms) from leaking into the carry gradient. The prev_logT/logP/
    logRho carry values themselves are LIVE — the IFT backward propagates
    ∂eps_grav/∂T_prev through the scan carry, matching MESA's coupled Newton
    solve (hydro_energy.f90:get1_energy_eqn). Issue #407 removed the earlier
    blanket stop_gradient on history terms; this residual detachment is a
    targeted NaN guard on the extraction path only.
    Location: evolution/_core.py (_egrav_warmup, _henyey_step_solve, _step_compute_eps_grav_frac, _step_solve_and_eps).
    """

    CONVERGENCE_FLAGS = 9
    """solver_converged, henyey_converged flags.

    Rationale: Boolean algorithmic decisions (converged/not), not physics.
    Location: evolution/_core.py (_henyey_step_solve).
    """

    WARM_START_GUESS = 10
    """ln_rho_guess warm-start for HELM density inversion.

    Rationale: the warm-start guess is a numerical convenience from the
    previous step's density — it speeds convergence but its derivative
    is not part of the physics chain. Detaching prevents gradient flow
    through the guess into the previous-step carry.
    Location: evolution/step.py (_step_solve_and_eps).
    """

    MESH_SCHEDULE = 11
    """Per-step structure mesh from a frozen replay schedule.

    Rationale: the structure mesh is an algorithmic decision recorded from
    the adaptive forward pass (evolve_star_adaptive). During differentiable
    replay, the recorded mesh sequence is replayed as scan xs. The mesh is
    frozen (stop_gradient'd) in the backward pass — the adaptive mesh
    adaptation algorithm is not differentiable (CONSTRAINT: JAX fixed-shape
    lax.scan). The mesh VALUES match the adaptive forward's choices; only
    the mesh-adaptation DECISION is frozen. Same pattern as dt in the
    frozen-schedule adjoint (GP-1/GP-7).
    MESA ref: evolve.f90:1882-1886 — do_mesh() per step before solve.
    Location: evolution/_core.py (step_fn xs unpack), evolution/step.py
    (_henyey_step_solve, _step_solve_and_eps).
    """

    COMP_SCHEDULE = 12
    """Per-step frozen composition from the adaptive forward trajectory.

    CONSTRAINT (not a MESA match): the 200-zone static-grid lax.scan
    replay cannot evolve composition on the 600-zone adaptive mesh
    recorded by the adaptive forward. Over ~1000 SGB steps the
    composition-path divergence is catastrophic (logL=-63.6 without
    freezing). Freezing the recorded per-step compositions as scan xs
    (remapped to the 200-zone grid) ensures the Henyey solve sees the
    correct composition at each step. The deviation is bounded by the
    600→200 zone remap error.

    Note: MESA's operator split (struct_burn_mix.f90:82-100) only
    splits the BURN (eps_nuc derivatives zeroed when op_split_burn=True).
    The MIXING remains coupled in the Newton Jacobian (line 144:
    do_chem=True → nvar=nvar_total, composition variables included).
    Our GP-12 freeze is therefore NOT analogous to MESA's operator split
    — it is a stronger freeze (both burn AND mix paths frozen), justified
    solely by the JAX grid-mismatch constraint above.

    The composition is stop_gradient'd in the backward pass — same
    pattern as dt (GP-1) and mesh (GP-11). The gradient flows through
    M → Henyey IFT → structure → σ²; composition is frozen context.

    AD-vs-FD neutrality: GP-12 does NOT bias the AD-vs-FD comparison.
    Both AD and FD use the same frozen composition schedule, so the
    composition detach creates no AD-vs-FD gap. The bias is only in
    the PHYSICAL gradient (missing ∂composition/∂M sensitivity) — a
    known limitation, not an AD correctness issue.
    Location: evolution/_core.py (step_fn xs unpack, composition override).
    """
