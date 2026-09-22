"""evolution/scan.py — lax.scan orchestration: carry init, windowed scan, result assembly.

Extracted from _core.py (#1089). These are module-level helpers called by
_evolve_star_jit. Python function calls are invisible to XLA tracing —
the compiled graph is IDENTICAL to the monolithic version.

Contains:
- _build_init_carry: build the initial carry tuple for lax.scan
- _run_windowed_scan: 3-phase gradient-windowed scan
- _assemble_result: assemble the result dict from scan outputs

CARRY SHAPE NOTE: the carry is a CONDITIONAL plain tuple:
  - adaptive_mesh=True:  18-element tuple
  - adaptive_mesh=False: 17-element tuple
Using a single NamedTuple changes the XLA graph and causes FP drift (#525).
"""

import jax
import jax.numpy as jnp
from jax import lax

from stellar_jax.config.constants import SECONDS_PER_YEAR
from stellar_jax.config.mesh_defaults import N_COMP, COMP_MFRACS
from stellar_jax.mesh import conservative_remap
from stellar_jax.evolution.contracts import (
    CARRY_X_PROF, CARRY_Y_PROF, CARRY_Z_PROF, CARRY_C12_PROF,
    CARRY_C13_PROF, CARRY_N14_PROF,
    CARRY_LOGL, CARRY_LOGTE, CARRY_Y_HENYEY, CARRY_COMP_MFRACS,
)


def _build_init_carry(adaptive_mesh: bool, init_profiles: dict,
                      y_warmup: jnp.ndarray,
                      q_mesh_init: jnp.ndarray) -> tuple:
    """Build the initial carry tuple for lax.scan.

    Parameters
    ----------
    adaptive_mesh : bool — conditional carry shape
    init_profiles : dict — {X, Y, Z, C12, C13, N14, X3,
        logL0, logTe0, dt_init, logT_init, logP_init, logRho_init}
    y_warmup : (N_s, 4) — Henyey state from eps_grav warmup
    q_mesh_init : (N_s+1,) — the ZAMS/warmup structure mesh that y_warmup
        was solved on. Appended as the q_mesh_prev tail field so the
        first replay step remaps y_warmup FROM this mesh onto q_mesh[0]
        (#1165). On the non-replay path it is an inert constant carry.

    Conditional shape — MUST match the XLA graph structure:
    - adaptive_mesh=True: 19-element tuple (with comp_mfracs + H211b history
      + q_mesh_prev tail).
    - adaptive_mesh=False: 18-element tuple (no comp_mfracs + H211b history
      + q_mesh_prev tail).

    H211b history (dt_old, dt_limit_ratio_old) initialized to 0.0:
    MESA timestep.f90:2381 — falls back to 1st-order when dt_old=0 or
    dt_limit_ratio_old=0 (no history available on first step).
    """
    X_profile_init = init_profiles['X']
    Y_profile_init = init_profiles['Y']
    Z = init_profiles['Z']
    C12_profile_init = init_profiles['C12']
    C13_profile_init = init_profiles['C13']
    N14_profile_init = init_profiles['N14']
    X3_profile_init = init_profiles['X3']
    logL0 = init_profiles['logL0']
    logTe0 = init_profiles['logTe0']
    dt_init = init_profiles['dt_init']
    logT_init = init_profiles['logT_init']
    logP_init = init_profiles['logP_init']
    logRho_init = init_profiles['logRho_init']

    Z_profile_init = jnp.full(N_COMP, Z)
    if adaptive_mesh:
        return (X_profile_init, Y_profile_init, Z_profile_init,
                C12_profile_init, C13_profile_init, N14_profile_init, X3_profile_init,
                logL0, logTe0,
                jnp.float64(0.0), dt_init, logT_init, logP_init, logRho_init,
                y_warmup, jnp.array(COMP_MFRACS),
                jnp.float64(0.0), jnp.float64(0.0),  # dt_old, dt_limit_ratio_old
                q_mesh_init)  #: q_mesh_prev tail (mesh y_warmup solved on)
    else:
        return (X_profile_init, Y_profile_init, Z_profile_init,
                C12_profile_init, C13_profile_init, N14_profile_init, X3_profile_init,
                logL0, logTe0,
                jnp.float64(0.0), dt_init, logT_init, logP_init, logRho_init,
                y_warmup,
                jnp.float64(0.0), jnp.float64(0.0),  # dt_old, dt_limit_ratio_old
                q_mesh_init)  #: q_mesh_prev tail (mesh y_warmup solved on)


def _run_windowed_scan(step_fn, init_carry: tuple,
                       scan_config: dict) -> tuple:
    """Run the 3-phase gradient-windowed scan.

    Parameters
    ----------
    step_fn : callable — per-step function for lax.scan
    init_carry : tuple — initial scan carry
    scan_config : dict — {max_steps, k0, k1, replay_schedule, dt_schedule,
        replay_mesh, mesh_schedule, replay_comp, comp_schedule_X, comp_schedule_Y,
        comp_schedule_N14, comp_schedule_C12, comp_schedule_C13}

    Phase 1 [0, k0): forward only, outputs+carry detached.
    Phase 2 [k0, k1): full autodiff.
    Phase 3 [k1, max_steps): forward only, outputs+carry detached.

    When replay_comp=True, xs is a tuple
    (dt, mesh, X_comp, Y_comp, N14_comp, C12_comp, C13_comp, comp_mfracs);
    when replay_mesh=True (no comp), xs is (dt, mesh);
    when only replay_schedule=True, xs is dt_step (scalar).
    All frozen xs are stop_gradient'd in phases 1 & 3 (GP-7: WINDOWED_SCAN).

    Reference: Griewank & Walther (2008) §13 on adjoint checkpointing.
    MESA: evolve.f90:1882-1886 — do_mesh() before each Newton solve.

    Returns
    -------
    final_carry, out (concatenated from all phases)
    """
    max_steps = scan_config['max_steps']
    k0 = scan_config['k0']
    k1 = scan_config['k1']
    replay_schedule = scan_config['replay_schedule']
    dt_schedule = scan_config['dt_schedule']
    replay_mesh = scan_config.get('replay_mesh', False)
    mesh_schedule = scan_config.get('mesh_schedule', None)
    replay_comp = scan_config.get('replay_comp', False)
    comp_schedule_X = scan_config.get('comp_schedule_X', None)
    comp_schedule_Y = scan_config.get('comp_schedule_Y', None)
    comp_schedule_N14 = scan_config.get('comp_schedule_N14', None)
    comp_schedule_C12 = scan_config.get('comp_schedule_C12', None)
    comp_schedule_C13 = scan_config.get('comp_schedule_C13', None)

    k0 = max(0, min(k0, max_steps))
    k1 = max(k0, min(k1, max_steps))

    # Build per-step xs for the scan.
    # When replay_comp=True: xs = (dt, mesh, X, Y, N14, C12, C13, comp_mfracs) — 8-tuple.
    #   All compositions are 600-zone (recorded adaptive mesh, no remap).
    #   comp_mfracs are the 600-zone cell centers for interp_X_at_mass.
    # When replay_mesh=True: xs = (dt_array, mesh_array) — 2-tuple.
    # When only replay_schedule=True: xs = dt_array (scalar per step).
    # When neither: xs = None.
    if replay_comp and comp_schedule_X is not None:
        # xs is an 8-tuple: (dt, mesh, X, Y, N14, C12, C13, comp_mfracs)
        # GRADSOLVE-faithful: 600-zone compositions on the recorded grid.
        comp_schedule_mfracs = scan_config.get('comp_schedule_mfracs', None)
        def _comp_xs(sl):
            parts = [dt_schedule[sl], mesh_schedule[sl],
                     comp_schedule_X[sl], comp_schedule_Y[sl]]
            if comp_schedule_N14 is not None:
                parts.append(comp_schedule_N14[sl])
            else:
                parts.append(comp_schedule_X[sl] * 0)  # placeholder zeros
            if comp_schedule_C12 is not None:
                parts.append(comp_schedule_C12[sl])
            else:
                parts.append(comp_schedule_X[sl] * 0)
            if comp_schedule_C13 is not None:
                parts.append(comp_schedule_C13[sl])
            else:
                parts.append(comp_schedule_X[sl] * 0)
            if comp_schedule_mfracs is not None:
                parts.append(comp_schedule_mfracs[sl])
            else:
                parts.append(comp_schedule_X[sl] * 0)  # placeholder
            return tuple(parts)
        xs_p1 = _comp_xs(slice(0, k0)) if k0 > 0 else None
        xs_p2 = _comp_xs(slice(k0, k1)) if (k1 - k0) > 0 else None
        xs_p3 = _comp_xs(slice(k1, None)) if (max_steps - k1) > 0 else None
    elif replay_mesh and mesh_schedule is not None:
        # xs is a tuple: (dt_schedule[slice], mesh_schedule[slice])
        xs_p1 = (dt_schedule[:k0], mesh_schedule[:k0]) if k0 > 0 else None
        xs_p2 = (dt_schedule[k0:k1], mesh_schedule[k0:k1]) if (k1 - k0) > 0 else None
        xs_p3 = (dt_schedule[k1:], mesh_schedule[k1:]) if (max_steps - k1) > 0 else None
    elif replay_schedule:
        xs_p1 = dt_schedule[:k0] if k0 > 0 else None
        xs_p2 = dt_schedule[k0:k1] if (k1 - k0) > 0 else None
        xs_p3 = dt_schedule[k1:] if (max_steps - k1) > 0 else None
    else:
        xs_p1 = xs_p2 = xs_p3 = None

    # Phase 1: steps [0, k0) — forward only, stop_gradient at exit.
    if k0 > 0:
        carry_p1, out_p1 = lax.scan(step_fn, init_carry, xs_p1, length=k0)
        carry_win = jax.lax.stop_gradient(carry_p1)  # GP-7: WINDOWED_SCAN
        out_p1 = jax.lax.stop_gradient(out_p1)  # GP-7: WINDOWED_SCAN
    else:
        carry_win = init_carry
        out_p1 = None

    # Phase 2: steps [k0, k1) — full autodiff with remat.
    # jax.checkpoint (remat) on the Phase-2 scan body: O(N) backward
    # memory instead of O(N²).  JANC (arXiv:2504.13750) needs none
    # only because it is EXPLICIT/cheap-per-step; we are IMPLICIT/
    # expensive (block-tridiagonal Henyey solve per step). P5
    # measured a compilation OOM at N≈154 without remat.
    # Cost: ~2× forward FLOPs in the gradient window (rematerialization).
    # Ref: GRADSOLVE arXiv:2609.02876 (remat on-by-default for
    # large/stiff systems); JAX docs on lax.scan + checkpoint.
    n_win = k1 - k0
    if n_win > 0:
        step_fn_remat = jax.checkpoint(step_fn)
        carry_p2, out_p2 = lax.scan(step_fn_remat, carry_win, xs_p2, length=n_win)
    else:
        carry_p2 = carry_win
        out_p2 = None

    # Phase 3: steps [k1, max_steps) — forward only again.
    n_tail = max_steps - k1
    if n_tail > 0:
        carry_tail_in = jax.lax.stop_gradient(carry_p2)  # GP-7: WINDOWED_SCAN
        final_carry, out_p3 = lax.scan(step_fn, carry_tail_in, xs_p3, length=n_tail)
        out_p3 = jax.lax.stop_gradient(out_p3)  # GP-7: WINDOWED_SCAN
    else:
        final_carry = carry_p2
        out_p3 = None

    # Concatenate outputs from all three phases in order.
    parts = [p for p in (out_p1, out_p2, out_p3) if p is not None]
    out = jnp.concatenate(parts, axis=0)

    return final_carry, out


def _assemble_result(final_carry: tuple, out: jnp.ndarray,
                     adaptive_mesh: bool, Z: jnp.ndarray,
                     atm_ratio: jnp.ndarray = None) -> dict:
    """Assemble the result dictionary from scan outputs.

    Handles adaptive-mesh remap of final profiles back to the static grid.

    Returns
    -------
    result : dict with star_age, log_L, log_Teff, profiles, etc.
    """
    X_final = final_carry[CARRY_X_PROF]
    Y_final = final_carry[CARRY_Y_PROF]
    Z_final = final_carry[CARRY_Z_PROF]
    C12_final = final_carry[CARRY_C12_PROF]
    C13_final = final_carry[CARRY_C13_PROF]
    N14_final = final_carry[CARRY_N14_PROF]
    logL_final = final_carry[CARRY_LOGL]
    logTe_final = final_carry[CARRY_LOGTE]
    y_henyey_final = final_carry[CARRY_Y_HENYEY]

    if adaptive_mesh:
        # Remap final profiles from the adapted composition grid back to the
        # static COMP_MFRACS for backward compatibility.
        comp_mfracs_final = final_carry[CARRY_COMP_MFRACS]
        static_grid = jnp.array(COMP_MFRACS)
        grid_displaced = jnp.max(jnp.abs(comp_mfracs_final - static_grid)) > 1e-10
        X_final = jnp.where(grid_displaced,
                            conservative_remap(X_final, comp_mfracs_final, static_grid),
                            X_final)
        Y_final = jnp.where(grid_displaced,
                            conservative_remap(Y_final, comp_mfracs_final, static_grid),
                            Y_final)
        Z_final = jnp.where(grid_displaced,
                            conservative_remap(Z_final, comp_mfracs_final, static_grid),
                            Z_final)
        C12_final = jnp.where(grid_displaced,
                              conservative_remap(C12_final, comp_mfracs_final, static_grid),
                              C12_final)
        C13_final = jnp.where(grid_displaced,
                              conservative_remap(C13_final, comp_mfracs_final, static_grid),
                              C13_final)
        N14_final = jnp.where(grid_displaced,
                              conservative_remap(N14_final, comp_mfracs_final, static_grid),
                              N14_final)
        # Per-zone CN normalization for the final output remap
        CN_target_out = (0.170 + 0.170 / 89.0 + 0.079) * Z
        CN_total_post = C12_final + C13_final + N14_final
        cn_ratio_out = jnp.where(
            grid_displaced & (CN_total_post > 1e-30),
            CN_target_out / CN_total_post, 1.0)
        C12_final = C12_final * cn_ratio_out
        C13_final = C13_final * cn_ratio_out
        N14_final = N14_final * cn_ratio_out
        comp_mfracs_adapted = comp_mfracs_final
    else:
        static_grid = jnp.array(COMP_MFRACS)
        comp_mfracs_adapted = static_grid

    return {
        'star_age':  out[:, 0] / SECONDS_PER_YEAR,
        'log_L':     out[:, 1],
        'log_Teff':  out[:, 2],
        'log_R':     out[:, 3],
        'center_h1': out[:, 4],
        'eps_grav_frac': out[:, 5],
        'log_Tc':    out[:, 6],
        'log_rhoc':  out[:, 7],
        'xh_ratio':  out[:, 8],
        'X_profile': X_final,
        'Y_profile': Y_final,
        'Z_profile': Z_final,
        'C12_profile': C12_final,
        'C13_profile': C13_final,
        'N14_profile': N14_final,
        'log_L_final': logL_final,
        'log_Teff_final': logTe_final,
        'from_henyey': True,
        'from_shooting': False,
        'y_henyey_final': y_henyey_final,
        'atm_ratio': atm_ratio,
        'comp_mfracs': static_grid,
        'comp_mfracs_adapted': comp_mfracs_adapted,
    }
