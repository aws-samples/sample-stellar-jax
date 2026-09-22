"""evolution/step.py — Per-step helpers for the lax.scan step function.

Extracted from _core.py (#1089). These are module-level helpers called by
step_fn inside _evolve_star_jit. Python function calls are invisible to XLA
tracing — the compiled graph is IDENTICAL to the monolithic version.

Contains:
- _henyey_step_solve: Henyey solve with atmosphere BC
- _step_remap_to_static: composition remap to static grid
- _step_solve_and_eps: Henyey solve + surface obs + eps on composition grid
- _step_composition_update: burn + mix + diffusion + CNO + X3
- _step_compute_eps_grav_frac: gravothermal energy fraction (Form C)
- _step_adaptive_mesh_remap: adaptive composition mesh remap
- _step_diagnostics: eps_grav, central state, varcontrol
- _step_timestep_decide: accept/reject + H211b controller
- _step_dt_limiters: nuclear/neutrino dt_max, XH limiter, He halt
"""

import jax
import jax.numpy as jnp

from stellar_jax.config.constants import (
    a_rad, sigma_sb, Lsun, SECONDS_PER_YEAR, Q_PER_G,
)
from stellar_jax.config.mesh_defaults import COMP_MFRACS
from stellar_jax.config.physics_floors import NAD_CP_FLOOR
from stellar_jax.microphysics.neutrino import epsilon_neutrino
from stellar_jax.henyey import (
    henyey_solve_from_state_atm, extract_shell_data_from_henyey,
)
from stellar_jax.mesh import adapt_composition_mesh, conservative_remap
from stellar_jax.composition.burn import burn_cno, relax_he3
from stellar_jax.composition.mix import mix_composition, _mix_z_in_cz
from stellar_jax.composition.diffuse import diffuse_composition
from stellar_jax.evolution.timestep import (
    h211b_filter_dt_next, compute_central_limiters, compute_worst_offender_ratio,
)


def _henyey_step_solve(y_henyey_prev_arg: jnp.ndarray,
                       tc, step_args: dict,
                       static_flags: dict) -> tuple:
    """Henyey step with atmosphere BC — module-level helper.

    Performs one Newton-solve step of the stellar structure equations using
    the atmosphere-BC continuation solver. R_phot is solved INSIDE the Newton
    system via atm_ratio — the ratio R_phot_init / r_surface computed once at
    ZAMS init (a stellar-jax concept). MESA derives photosphere_r at each step
    from the structural state (star_utils.f90:1093 set_phot_info); our constant
    atm_ratio is a CONSTRAINT (JAX lax.scan avoids carrying the atmosphere
    solve through the scan body).

    Parameters
    ----------
    y_henyey_prev_arg : (N_s, 4) — Henyey state from previous step
    tc : TracedConstants — traced step-invariant values
    step_args : dict — per-step traced values {X_prof, t_age, dt, N14_prof,
        comp_mfracs, X3_prof, ln_rho_guess, q_mesh_override,
        gradL_composition_term}
    static_flags : dict — {eps_grav_flag, detach_eps_grav_carry, replay_schedule,
        replay_mesh, helm_eos, bicubic_opacity}

    When replay_mesh=True, step_args['q_mesh_override'] carries the per-step
    mesh from the mesh_schedule (already stop_gradient'd at the source in
    step_fn, GP-11: MESH_SCHEDULE). Falls back to tc.q_mesh when absent or None.

    MESA ref: evolve.f90:1882-1886 — do_mesh() in prepare_for_new_step sets
    the structure mesh BEFORE each Newton solve.

    Returns
    -------
    y_out : (N_s, 4) — converged Henyey state (or prev on divergence)
    shell_data : (N_s, ...) — shell diagnostics (stop_gradient'd)
    converged : bool — Newton convergence flag
    """
    eps_grav_flag = static_flags['eps_grav_flag']
    detach_eps_grav_carry = static_flags['detach_eps_grav_carry']
    replay_schedule = static_flags['replay_schedule']
    replay_mesh = static_flags.get('replay_mesh', False)
    helm_eos = static_flags.get('helm_eos', False)
    bicubic_opacity = static_flags.get('bicubic_opacity', True)
    atm_ratio_arg = tc.atm_ratio
    q_mesh_arg = tc.q_mesh
    # Per-step mesh override: when replay_mesh=True, use the recorded mesh.
    # Already stop_gradient'd at the source (step_fn in _core.py, GP-11).
    q_mesh_override = step_args.get('q_mesh_override', None)
    if replay_mesh and q_mesh_override is not None:
        q_mesh_arg = q_mesh_override  # GP-11: MESH_SCHEDULE (frozen at source)
    M_star_cgs_arg = tc.M_star_cgs
    X_prof_arg = step_args.get('X_prof', None)
    Z_arg = tc.Z
    alpha_mlt_arg = tc.alpha_mlt
    t_age_arg = step_args.get('t_age', jnp.float64(0.0))
    dt_arg = step_args.get('dt', jnp.float64(0.0))
    N14_prof_arg = step_args.get('N14_prof', None)
    comp_mfracs_arg = step_args.get('comp_mfracs', None)
    opacity_factor_arg = tc.opacity_factor
    eps_nuc_factor_arg = tc.eps_nuc_factor
    X3_prof_arg = step_args.get('X3_prof', None)
    ln_rho_guess_arg = step_args.get('ln_rho_guess', None)
    #: Ledoux composition-gradient term (replay path only; None on the
    # normal lax.scan MS path → XLA trace identity preserved).
    gradL_composition_term_arg = step_args.get('gradL_composition_term', None)

    ln_T_prev_h = y_henyey_prev_arg[:, 2]
    ln_P_prev_h = y_henyey_prev_arg[:, 1]

    # MUTATION GATE: sever cross-step gradient when detach=True.
    if detach_eps_grav_carry:
        ln_T_prev_h = jax.lax.stop_gradient(ln_T_prev_h)  # GP-8: EPSGRAV_HISTORY
        ln_P_prev_h = jax.lax.stop_gradient(ln_P_prev_h)  # GP-8: EPSGRAV_HISTORY

    # eps_grav coupling: inv_dt activates Form C (KWW 4.18).
    # Static branch: eps_grav_flag=False → separate XLA graph.
    if eps_grav_flag:
        inv_dt_h = jax.lax.stop_gradient(  # GP-1: TIMESTEP
            jnp.where(dt_arg > 0.0, 1.0 / jnp.maximum(dt_arg, 1.0), 0.0))
    else:
        inv_dt_h = jnp.float64(0.0)

    henyey_result = henyey_solve_from_state_atm(
        y_henyey_prev_arg, q_mesh_arg, M_star_cgs_arg, X_prof_arg, Z_arg,
        alpha_mlt_arg, None, n_iter=100, tol=1e-4,  # MESA: star_solver.f90:213 (tol_residual_norm1=1e-10); 1e-4 matches _HENYEY_CONV_TOL
        ln_T_prev=ln_T_prev_h, ln_P_prev=ln_P_prev_h, inv_dt=inv_dt_h,
        atm_ratio=atm_ratio_arg, N14_profile=N14_prof_arg,
        comp_mfracs=comp_mfracs_arg,
        bypass_conv_gate=replay_schedule,
        opacity_factor=opacity_factor_arg,
        eps_nuc_factor=eps_nuc_factor_arg,
        X3_profile=X3_prof_arg,
        ln_rho_guess=ln_rho_guess_arg,
        gradL_composition_term=gradL_composition_term_arg,  #: replay Ledoux term
        helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
    y_new = henyey_result['y']

    # Divergence guard: NaN/Inf or structure equations violated >100%.
    res_norm = henyey_result.get('residual_norm', jnp.float64(0.0))
    diverged = ~jnp.all(jnp.isfinite(y_new)) | (res_norm > 1.0)

    # AD-safe non-finite guard: sanitize BEFORE jnp.where (JAX evaluates
    # gradients through BOTH branches regardless of forward selection).
    y_safe = jnp.where(~jnp.isfinite(y_new), y_henyey_prev_arg, y_new)
    y_out = jnp.where(diverged, y_henyey_prev_arg, y_safe)

    h_shell = extract_shell_data_from_henyey(
        y_out, q_mesh_arg, M_star_cgs_arg, X_prof_arg, Z_arg,
        alpha_mlt_arg, t_age_arg, comp_mfracs=comp_mfracs_arg,
        helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
    h_shell = jax.lax.stop_gradient(h_shell)  # GP-5: STRUCTURE_TO_COMP

    # Convergence flag (MESA discipline: struct_burn_mix.f90:580-603).
    solver_converged = henyey_result.get('converged', True)
    converged = jax.lax.stop_gradient(solver_converged)  # GP-9: CONVERGENCE_FLAGS

    return y_out, h_shell, converged


def _step_composition_update(comp: dict, step_data: dict,
                             tc: 'TracedConstants',
                             physics: 'PhysicsToggles') -> tuple:
    """Per-step composition update: burn + mix + diffusion + CNO + X3.

    Parameters
    ----------
    comp : dict — composition profiles {X, Y, Z, C12, C13, N14, X3}
    step_data : dict — {shell_data_sg, eps_comp, dt}
    tc : TracedConstants — for mass, f_ov, Z, diffusion_factor
    physics : PhysicsToggles — for diffusion, z_feedback

    Returns
    -------
    X_diff, Y_mixed, Z_mixed, C12_m, C13_m, N14_m, X3_m
    """
    X_s, Y_s, Z_s = comp['X'], comp['Y'], comp['Z']
    C12_s, C13_s, N14_s, X3_s = comp['C12'], comp['C13'], comp['N14'], comp['X3']
    shell_data_sg = step_data['shell_data_sg']
    eps_comp = step_data['eps_comp']
    dt = step_data['dt']
    mass = tc.mass
    f_ov = tc.f_ov
    diffusion_factor = tc.diffusion_factor
    diffusion = physics.diffusion
    z_feedback = physics.z_feedback
    # Burn: deplete H and deposit into He (baryon conservation, ΔY = −ΔX).
    # MESA net_eval.f90:274 — sum(dxdt) = 0 by molar baryon conservation.
    # eps_comp (dL/dm from the Henyey grid) is LIVE w.r.t. the backward pass.
    # MESA's default is op_split_burn = .false. (controls.defaults:9501),
    # meaning the burn composition derivatives ARE included in the structure
    # Jacobian. Our code matches this: the gradient M → Henyey → eps_comp →
    # dX → X_burned → center_h1 flows through the full burn path.
    dX = eps_comp * dt / Q_PER_G
    actual_dX = jnp.minimum(dX, X_s)  # can't burn more H than exists
    X_burned = jnp.maximum(X_s - dX, 0.0)
    # The Henyey solver computes Y = 1 - X - Z (solver/residual.py:67),
    # NOT from the tracked Y_profile in the carry. So the backward path
    # through Y_burned → actual_dX → eps_comp → Henyey is a DEAD
    # gradient path (zero cotangent) that only expands the XLA backward
    # graph. CONSTRAINT: when Y is wired into the Henyey solve (μ from
    # Y directly), this stop_gradient must be REMOVED.
    Y_burned = Y_s + jax.lax.stop_gradient(actual_dX)  # GP-4: Z_CNO_ISOTOPES

    X_mixed = mix_composition(X_burned, shell_data_sg, M_solar=mass, f_ov=f_ov)

    if diffusion:
        _diff_result = jax.lax.stop_gradient(  # GP-3: DIFFUSION
            diffuse_composition(
                X_mixed, Y_burned, shell_data_sg, dt, mass,
                Z_profile=Z_s,
                use_exact_structure=z_feedback,
                use_4species=z_feedback))
        X_diffused, Y_diffused, Z_diffused = _diff_result
        # Straight-through estimator with diffusion_factor
        X_diff = X_mixed + diffusion_factor * (X_diffused - jax.lax.stop_gradient(X_mixed))  # GP-3: DIFFUSION
        Y_diff = Y_burned + diffusion_factor * (Y_diffused - jax.lax.stop_gradient(Y_burned))  # GP-3: DIFFUSION
        Z_new = Z_s + diffusion_factor * (Z_diffused - jax.lax.stop_gradient(Z_s))  # GP-3: DIFFUSION
    else:
        X_diff = X_mixed
        Y_diff = Y_burned
        Z_new = Z_s

    Y_mixed = mix_composition(Y_diff, shell_data_sg, M_solar=mass, f_ov=f_ov)

    # Z-mixing in convective zones (MESA: mix_info.f90)
    if diffusion:
        Z_mixed = jax.lax.stop_gradient(  # GP-4: Z_CNO_ISOTOPES
            _mix_z_in_cz(Z_new, shell_data_sg, mass, f_ov))
    else:
        Z_mixed = Z_new

    # CNO isotope tracking (operator-split burn + mix)
    # X_s is LIVE w.r.t. the backward pass for burn_cno. The CN equilibration
    # rate depends on X_H, and the gradient M → X_s → burn_cno → N14 → eps_cno
    # is the primary path for ∂center_h1/∂M through the composition carry.
    # MESA's default op_split_burn = .false. (controls.defaults:9501) includes
    # the burn composition derivatives in the structure Jacobian.
    C12_b, C13_b, N14_b = burn_cno(C12_s, C13_s, N14_s,
                                    X_s,
                                    shell_data_sg, dt)
    C12_m = mix_composition(C12_b, shell_data_sg, M_solar=mass, f_ov=f_ov)  # GP-4 relaxed: C12 gradient now live for ∂eps_cno/∂Z
    C13_m = mix_composition(C13_b, shell_data_sg, M_solar=mass, f_ov=f_ov)  # GP-4 relaxed: C13 gradient now live for ∂eps_cno/∂Z
    N14_m = mix_composition(N14_b, shell_data_sg, M_solar=mass, f_ov=f_ov)

    # ³He equilibrium relaxation (Clayton 1968 §5.6, operator-split)
    X3_b = jax.lax.stop_gradient(relax_he3(X3_s, X_s, shell_data_sg, dt))  # GP-4: Z_CNO_ISOTOPES
    X3_m = jax.lax.stop_gradient(mix_composition(X3_b, shell_data_sg, M_solar=mass, f_ov=f_ov))  # GP-4: Z_CNO_ISOTOPES

    # Per-zone X+Y+Z=1 normalization (MESA hydro_vars.f90:751-754).
    # stop_gradient on xa_sum: the normalization corrects O(1e-15) drift,
    # so differentiating through the divisor adds gradient graph complexity
    # (cross-species coupling ∂(X/s)/∂Y = -X/s²) for zero physical signal.
    xa_sum = jax.lax.stop_gradient(X_diff + Y_mixed + Z_mixed)  # GP-4: Z_CNO_ISOTOPES
    X_diff = X_diff / xa_sum
    Y_mixed = Y_mixed / xa_sum
    Z_mixed = Z_mixed / xa_sum

    return X_diff, Y_mixed, Z_mixed, C12_m, C13_m, N14_m, X3_m


def _step_compute_eps_grav_frac(shell_data: jnp.ndarray, prev_struct: dict,
                                step_vals: dict, tc: 'TracedConstants',
                                eps_grav_flag: bool) -> jnp.ndarray:
    """Compute gravothermal energy fraction — Form C (KWW 4.18).

    Parameters
    ----------
    prev_struct : dict — {prev_logT, prev_logP, prev_logRho}
    step_vals : dict — {dt, L_henyey}
    tc : TracedConstants — for M_star_cgs
    """
    prev_logT = prev_struct['prev_logT']
    prev_logP = prev_struct['prev_logP']
    dt = step_vals['dt']
    L_henyey = step_vals['L_henyey']
    M_star_cgs = tc.M_star_cgs

    cur_logT = jax.lax.stop_gradient(shell_data[:, 6])  # GP-8: EPSGRAV_HISTORY
    cur_logP = jax.lax.stop_gradient(shell_data[:, 9])  # GP-8: EPSGRAV_HISTORY
    cur_logRho = jax.lax.stop_gradient(shell_data[:, 7])  # GP-8: EPSGRAV_HISTORY
    mf_cur = jax.lax.stop_gradient(shell_data[:, 1])  # GP-8: EPSGRAV_HISTORY
    nad_cur = jax.lax.stop_gradient(shell_data[:, 3])  # GP-8: EPSGRAV_HISTORY

    dm_shells = jnp.abs(jnp.diff(mf_cur, prepend=mf_cur[0]))
    T_cur = 10.0**cur_logT
    P_cur = 10.0**cur_logP
    rho_cur = 10.0**cur_logRho
    T_prev = 10.0**prev_logT
    P_prev = 10.0**prev_logP

    # cp from ideal gas + radiation EOS
    P_rad_cur = a_rad * T_cur**4 / 3.0
    beta_cur = jnp.clip((P_cur - P_rad_cur) / P_cur, 0.01, 1.0)
    delta_cur = (4.0 - 3.0 * beta_cur) / beta_cur
    cp_cur = P_cur * delta_cur / (T_cur * rho_cur * jnp.maximum(nad_cur, NAD_CP_FLOOR))

    # Time derivatives from consecutive Henyey equilibria
    dT_dt = (T_cur - T_prev) / jnp.maximum(dt, 1.0)
    dP_dt = (P_cur - P_prev) / jnp.maximum(dt, 1.0)

    # Form C per shell
    eps_grav_per_shell = -cp_cur * dT_dt + (cp_cur * T_cur * nad_cur / P_cur) * dP_dt
    L_grav = jnp.sum(eps_grav_per_shell * dm_shells) * M_star_cgs

    if eps_grav_flag:
        return L_grav / jnp.maximum(L_henyey, 1e-30)
    else:
        return jnp.float64(0.0)


def _step_adaptive_mesh_remap(comp_new: dict, comp_carry: dict,
                              reject: jnp.ndarray,
                              comp_mfracs_carry: jnp.ndarray,
                              Z: jnp.ndarray) -> tuple:
    """Apply adaptive composition mesh remapping on accepted steps.

    Parameters
    ----------
    comp_new : dict — new-step composition {X, Y, Z, C12, C13, N14, X3}
    comp_carry : dict — carry (previous) composition {X, Y, Z, C12, C13, N14}
    """
    X_diff, Y_mixed, Z_mixed = comp_new['X'], comp_new['Y'], comp_new['Z']
    C12_m, C13_m, N14_m = comp_new['C12'], comp_new['C13'], comp_new['N14']
    X3_out = comp_new['X3']
    X_prof, Y_prof, Z_prof = comp_carry['X'], comp_carry['Y'], comp_carry['Z']
    C12_prof, C13_prof, N14_prof = comp_carry['C12'], comp_carry['C13'], comp_carry['N14']

    static_grid_end = jnp.array(COMP_MFRACS)
    X_remapped, Y_remapped, Z_remapped, C12_remapped, C13_remapped, N14_remapped, comp_mfracs_new = \
        adapt_composition_mesh(X_diff, Y_mixed, Z_mixed, C12_m, C13_m, N14_m, static_grid_end)

    # On rejected steps, keep old grid and profiles unchanged.
    X_final = jnp.where(reject, X_prof, X_remapped)
    Y_final = jnp.where(reject, Y_prof, Y_remapped)
    Z_final = jnp.where(reject, Z_prof, Z_remapped)
    C12_final = jnp.where(reject, C12_prof, C12_remapped)
    C13_final = jnp.where(reject, C13_prof, C13_remapped)
    N14_final = jnp.where(reject, N14_prof, N14_remapped)
    X3_final = X3_out  # stop_gradient'd, no remap needed
    comp_mfracs_out = jnp.where(reject, comp_mfracs_carry, comp_mfracs_new)

    return X_final, Y_final, Z_final, C12_final, C13_final, N14_final, X3_final, comp_mfracs_out


def _step_remap_to_static(comp: dict,
                          comp_mfracs_carry: jnp.ndarray | None,
                          Z_metal: jnp.ndarray,
                          adaptive_mesh: bool) -> tuple:
    """Remap composition to the static grid at step start (adaptive mesh).

    Parameters
    ----------
    comp : dict — composition profiles {X, Y, Z, C12, C13, N14, X3}
    """
    X_prof, Y_prof, Z_prof = comp['X'], comp['Y'], comp['Z']
    C12_prof, C13_prof, N14_prof, X3_prof = comp['C12'], comp['C13'], comp['N14'], comp['X3']
    if adaptive_mesh:
        static_grid = jnp.array(COMP_MFRACS)
        grid_changed = jnp.max(jnp.abs(comp_mfracs_carry - static_grid)) > 1e-10
        X_s = jnp.where(grid_changed, conservative_remap(X_prof, comp_mfracs_carry, static_grid), X_prof)
        Y_s = jnp.where(grid_changed, conservative_remap(Y_prof, comp_mfracs_carry, static_grid), Y_prof)
        Z_s = jnp.where(grid_changed, conservative_remap(Z_prof, comp_mfracs_carry, static_grid), Z_prof)
        C12_s = jnp.where(grid_changed, conservative_remap(C12_prof, comp_mfracs_carry, static_grid), C12_prof)
        C13_s = jnp.where(grid_changed, conservative_remap(C13_prof, comp_mfracs_carry, static_grid), C13_prof)
        N14_s = jnp.where(grid_changed, conservative_remap(N14_prof, comp_mfracs_carry, static_grid), N14_prof)
        X3_s = jnp.where(grid_changed, conservative_remap(X3_prof, comp_mfracs_carry, static_grid), X3_prof)
        CN_target = (0.170 + 0.170 / 89.0 + 0.079) * Z_metal
        CN_total_new = C12_s + C13_s + N14_s
        cn_ratio = jnp.where(grid_changed & (CN_total_new > 1e-30), CN_target / CN_total_new, 1.0)
        C12_s = C12_s * cn_ratio
        C13_s = C13_s * cn_ratio
        N14_s = N14_s * cn_ratio
    else:
        X_s, Y_s, Z_s = X_prof, Y_prof, Z_prof
        C12_s, C13_s, N14_s, X3_s = C12_prof, C13_prof, N14_prof, X3_prof
    return X_s, Y_s, Z_s, C12_s, C13_s, N14_s, X3_s


def _step_solve_and_eps(y_henyey_prev: jnp.ndarray, comp_s: dict,
                        step_in: dict, tc: 'TracedConstants',
                        henyey_step_fn: callable) -> tuple:
    """Henyey solve + surface observables + eps on composition grid.

    Parameters
    ----------
    comp_s : dict — static-grid composition {X, N14, X3}
    step_in : dict — {prev_logRho, dt, t_age, helm_eos, q_mesh_override, comp_mfracs, gradL_composition_term}
    tc : TracedConstants — traced step-invariant values
    """
    X_s = comp_s['X']
    N14_s = comp_s['N14']
    X3_s = comp_s['X3']
    prev_logRho = step_in['prev_logRho']
    dt = step_in['dt']
    t_age = step_in['t_age']
    N_s = tc.N_s
    atm_ratio = tc.atm_ratio
    q_mesh = tc.q_mesh
    M_star_cgs = tc.M_star_cgs
    # Per-step mesh override for replay (GP-11: MESH_SCHEDULE).
    # stop_gradient'd at the source (step_fn in _core.py) so both
    # _henyey_step_solve and this function see a frozen mesh.
    q_mesh_override = step_in.get('q_mesh_override', None)
    if q_mesh_override is not None:
        q_mesh = q_mesh_override  # GP-11: frozen at source (_core.py)

    if step_in.get('helm_eos', False):
        ln_rho_guess = jax.lax.stop_gradient(prev_logRho * jnp.log(10.0))  # GP-10: WARM_START_GUESS
    else:
        ln_rho_guess = None
    _step_henyey_args = {
        'X_prof': X_s, 't_age': t_age, 'dt': dt,
        'N14_prof': N14_s, 'comp_mfracs': step_in.get('comp_mfracs', None),
        'X3_prof': X3_s, 'ln_rho_guess': ln_rho_guess,
        'q_mesh_override': q_mesh_override,
        #: Ledoux composition-gradient term for the replay solve
        # (None on the non-replay path → unchanged graph).
        'gradL_composition_term': step_in.get('gradL_composition_term', None),
    }
    y_out, shell_data, converged = henyey_step_fn(
        y_henyey_prev, tc, _step_henyey_args)
    ell_surf = y_out[N_s - 1, 3]
    L_h = ell_surf * Lsun
    R_phot = atm_ratio * jnp.exp(y_out[-1, 0])
    T_eff = (L_h / (4.0 * jnp.pi * sigma_sb * R_phot**2))**0.25
    logL_new = jnp.log10(jnp.maximum(L_h, 1e-30) / Lsun)
    logTe_new = jnp.log10(jnp.maximum(T_eff, 1.0))
    shell_sg = jax.lax.stop_gradient(shell_data)  # GP-5: STRUCTURE_TO_COMP

    # ── eps_comp from dL/dm (the Henyey luminosity gradient) ──
    #
    # On the MS, dL/dm = eps_nuc − eps_ν + eps_grav ≈ eps_nuc (eps_grav ≈ 0),
    # validated by test_structure_burn_split_error_ms. On the RGB, eps_grav
    # contaminates this proxy (median 58% overestimate at the RGB tip, issue
    #). The RGB fix lives in adaptive/forward.py (Python loop), which
    # uses direct eps_nuc. The JIT path retains dL/dm because:
    #   (a) evolve_star (JIT) is used for MS runs + gradient tests, where the
    #       proxy is accurate;
    #   (b) evolve_star_adaptive (Python loop) is used for RGB runs, where the
    #       direct eps_nuc fix is active;
    #   (c) CONSTRAINT: adding vmap(epsilon_nuclear) to the lax.scan body
    #       enlarges the XLA HLO graph enough to perturb floating-point
    #       accumulation order in the adjoint cotangents, breaking the r₀₂
    #       ratio gradient (test_seismic_gradient_r02).
    #
    # MESA reference: hydro_chem_eqns.f90:102-115 uses dxdt_nuc from the
    # nuclear network, never dL/dm. Our dL/dm proxy is a CONSTRAINT, not a
    # physics match — see adaptive/forward.py for the MESA-matching path.
    ell_arr = y_out[:, 3]
    dq = jnp.diff(q_mesh[:N_s])
    dm = M_star_cgs * dq
    dell = jnp.diff(ell_arr)
    eps_cells = jnp.maximum(dell * Lsun / jnp.maximum(dm, 1e-30), 0.0)
    q_mid = 0.5 * (q_mesh[:N_s-1] + q_mesh[1:N_s])
    eps_comp = jnp.interp(jnp.array(COMP_MFRACS), q_mid, eps_cells,
                          left=eps_cells[0], right=eps_cells[-1])
    return y_out, shell_data, converged, L_h, logL_new, logTe_new, shell_sg, eps_comp


def _step_diagnostics(step_state: dict, prev_struct: dict,
                      tc: 'TracedConstants',
                      eps_grav_flag: bool) -> tuple:
    """Compute eps_grav, central T/rho, varcontrol quantities.

    Parameters
    ----------
    step_state : dict — current step values:
        X_diff, X_s, logL_new, logTe_new, logL, logTe,
        shell_data, dt, L_h, y_out
    prev_struct : dict — previous-step structure: prev_logT, prev_logP, prev_logRho
    tc : TracedConstants — for M_star_cgs
    eps_grav_flag : bool — static flag
    """
    X_diff = step_state['X_diff']
    X_s = step_state['X_s']
    logL_new = step_state['logL_new']
    logTe_new = step_state['logTe_new']
    logL = step_state['logL']
    logTe = step_state['logTe']
    shell_data = step_state['shell_data']
    dt = step_state['dt']
    L_h = step_state['L_h']
    y_out = step_state['y_out']

    cur_logT = jax.lax.stop_gradient(shell_data[:, 6])  # GP-8: EPSGRAV_HISTORY
    cur_logP = jax.lax.stop_gradient(shell_data[:, 9])  # GP-8: EPSGRAV_HISTORY
    cur_logRho = jax.lax.stop_gradient(shell_data[:, 7])  # GP-8: EPSGRAV_HISTORY
    eps_grav_frac = _step_compute_eps_grav_frac(
        shell_data, prev_struct, {'dt': dt, 'L_henyey': L_h}, tc, eps_grav_flag)
    log_Tc = y_out[0, 2] / jnp.log(10.0)
    log_rhoc = cur_logRho[-1]
    max_dX = jnp.max(jnp.abs(X_diff - X_s))
    delta_logL = jnp.abs(logL_new - logL)
    delta_logTe = jnp.abs(logTe_new - logTe)
    varcontrol_reject = jnp.maximum(delta_logL, delta_logTe)
    varcontrol = jnp.maximum(max_dX, varcontrol_reject)
    return (cur_logT, cur_logP, cur_logRho, eps_grav_frac, log_Tc, log_rhoc,
            max_dX, delta_logL, delta_logTe, varcontrol, varcontrol_reject)


def _step_timestep_decide(dt_state: dict, tc: 'TracedConstants',
                          numerics: 'SolverNumerics') -> tuple:
    """Accept/reject decision + H211b dt controller.

    Parameters
    ----------
    dt_state : dict — varcontrol, varcontrol_reject, converged, dt,
        delta_logL, delta_logTe, cur_logT, cur_logRho,
        prev_logT, prev_logRho, X_diff, X_s, dt_old, dt_limit_ratio_old
    tc : TracedConstants — traced step-invariant values
    numerics : SolverNumerics — static solver config
    """
    varcontrol = dt_state['varcontrol']
    varcontrol_reject = dt_state['varcontrol_reject']
    converged = dt_state['converged']
    dt = dt_state['dt']
    delta_logL = dt_state['delta_logL']
    delta_logTe = dt_state['delta_logTe']
    cur_logT = dt_state['cur_logT']
    cur_logRho = dt_state['cur_logRho']
    prev_logT = dt_state['prev_logT']
    prev_logRho = dt_state['prev_logRho']
    X_diff = dt_state['X_diff']
    X_s = dt_state['X_s']
    dt_old = dt_state['dt_old']
    dt_limit_ratio_old = dt_state['dt_limit_ratio_old']

    reject_threshold = tc.reject_threshold
    _DELTA_LGL_HARD = tc.delta_lgL_hard
    _DELTA_LGTE_HARD = tc.delta_lgTe_hard
    varcontrol_target = tc.varcontrol_target
    dt_shrink = tc.dt_shrink
    dt_grow = tc.dt_grow

    _xh_cntr_limit = numerics.xh_cntr_limit
    _delta_lgT_cntr_limit = numerics.delta_lgT_cntr_limit
    _delta_lgRho_cntr_limit = numerics.delta_lgRho_cntr_limit
    fixed_dt = numerics.fixed_dt
    replay_schedule = numerics.replay_schedule
    freeze_schedule = numerics.freeze_schedule

    dt_at_floor = dt <= (1.1e5 * SECONDS_PER_YEAR)
    convergence_rej = (~converged) & (~dt_at_floor)
    structural_rej = (
        (jax.lax.stop_gradient(delta_logL) > _DELTA_LGL_HARD) |  # GP-1: TIMESTEP
        (jax.lax.stop_gradient(delta_logTe) > _DELTA_LGTE_HARD))  # GP-1: TIMESTEP
    reject = ((varcontrol_reject > reject_threshold) | convergence_rej | structural_rej) & (~dt_at_floor)
    if fixed_dt is None:
        logT_c = jax.lax.stop_gradient(cur_logT[-1])  # GP-1: TIMESTEP
        logT_c_old = jax.lax.stop_gradient(prev_logT[-1])  # GP-1: TIMESTEP
        logRho_c = jax.lax.stop_gradient(cur_logRho[-1])  # GP-1: TIMESTEP
        logRho_c_old = jax.lax.stop_gradient(prev_logRho[-1])  # GP-1: TIMESTEP
        r_lgT, r_lgRho = compute_central_limiters(
            logT_c, logT_c_old, logRho_c, logRho_c_old,
            _delta_lgT_cntr_limit, _delta_lgRho_cntr_limit)
        dXH_wo = jax.lax.stop_gradient(jnp.abs(X_diff[0] - X_s[0]))  # GP-1: TIMESTEP
        worst = compute_worst_offender_ratio(
            varcontrol, varcontrol_target, dXH_wo, _xh_cntr_limit, r_lgT, r_lgRho)
        dt_next_a = h211b_filter_dt_next(
            dt, jax.lax.stop_gradient(dt_old), worst, jax.lax.stop_gradient(dt_limit_ratio_old))  # GP-1: TIMESTEP
        dt_next_a = jnp.clip(dt_next_a, dt * dt_shrink, dt * dt_grow)
    else:
        worst = jnp.float64(0.0)
        dt_next_a = dt
    if fixed_dt is not None:
        reject = jnp.bool_(False)
    if replay_schedule:
        reject = jnp.bool_(False)
    if freeze_schedule:
        reject = jax.lax.stop_gradient(reject)  # GP-1: TIMESTEP
    return reject, dt_at_floor, dt_next_a, worst


def _step_dt_limiters(shell_data: jnp.ndarray, state: dict, prev_state: dict,
                      numerics: 'SolverNumerics',
                      xs_step: jnp.ndarray) -> tuple:
    # lint: limiter-chain exception — 6 coupled dt limiters applied in sequence; splitting breaks ordering
    """Nuclear/neutrino dt_max, XH limiter, He halt, floor, mode overrides.

    Parameters
    ----------
    shell_data : (N_s, ...) — shell diagnostics
    state : dict — mutable output state from accept/reject gate (includes Z)
    prev_state : dict — previous-step structure state
    numerics : SolverNumerics — static solver config
    xs_step : scalar — dt schedule entry (for replay mode)
    """
    reject = state['reject']
    dt_at_floor = state['dt_at_floor']
    dt_next = state['dt_next']
    X_out, X_s = state['X_out'], state['X_s']
    Y_out, Y_s = state['Y_out'], state['Y_s']
    Z_out, Z_s = state['Z_out'], state['Z_s']
    C12_out, C12_s = state['C12_out'], state['C12_s']
    C13_out, C13_s = state['C13_out'], state['C13_s']
    N14_out, N14_s = state['N14_out'], state['N14_s']
    X3_out, X3_s = state['X3_out'], state['X3_s']
    logL_out, logL = state['logL_out'], state['logL']
    logTe_out, logTe = state['logTe_out'], state['logTe']
    t_new, t = state['t_new'], state['t']
    Z = state['Z']
    prev_logT = prev_state['prev_logT']
    prev_logP = prev_state['prev_logP']
    prev_logRho = prev_state['prev_logRho']
    logT_out = state['logT_out']
    logP_out = state['logP_out']
    logRho_out = state['logRho_out']
    log_R_out = state['log_R_out']
    egf_out = state['egf_out']
    log_Tc = state['log_Tc']
    log_rhoc = state['log_rhoc']
    cur_logT = state['cur_logT']
    cur_logRho = state['cur_logRho']
    dt = state['dt']
    t_max_sec = state['t_max_sec']

    _xh_cntr_limit = numerics.xh_cntr_limit
    _xh_cntr_hard_limit = numerics.xh_cntr_hard_limit
    fixed_dt = numerics.fixed_dt
    replay_schedule = numerics.replay_schedule
    freeze_schedule = numerics.freeze_schedule

    eps_nuc_max = jnp.maximum(jnp.max(jax.lax.stop_gradient(shell_data[:, 0])), 1e-10)  # GP-1: TIMESTEP
    T_c = 10.0**jax.lax.stop_gradient(cur_logT[-1])  # GP-1: TIMESTEP
    rho_c = 10.0**jax.lax.stop_gradient(cur_logRho[-1])  # GP-1: TIMESTEP
    X_c_val = jax.lax.stop_gradient(X_out[0])  # GP-1: TIMESTEP
    eps_nu_c = epsilon_neutrino(rho_c, T_c, X_c_val, Z)
    dt_max = 0.10 * Q_PER_G / jnp.maximum(eps_nuc_max, eps_nu_c)
    dt_next = jnp.minimum(dt_next, dt_max)
    delta_XH = jax.lax.stop_gradient(jnp.abs(X_out[0] - X_s[0]))  # GP-1: TIMESTEP
    xh_ratio = delta_XH / _xh_cntr_limit
    dt_next = dt_next * jnp.where(xh_ratio > 1.0, _xh_cntr_limit / jnp.maximum(delta_XH, 1e-30), 1.0)
    xh_hard = jax.lax.stop_gradient(delta_XH > _xh_cntr_hard_limit) & (~dt_at_floor)  # GP-1: TIMESTEP
    # In frozen-schedule replay, suppress xh_hard BEFORE it applies its
    # jnp.where reversions. The adaptive forward already accepted these
    # steps — the schedule IS the accepted sequence. Without this guard,
    # xh_hard reverts t_new/logL_out/composition BEFORE the reject=False
    # guard below can take effect, desynchronizing the carry from the scan
    # index (time doesn't advance but lax.scan moves to the next xs entry).
    # MESA ref: timestep.f90 — retry logic is meaningful only when the
    # code controls the timestep.
    if replay_schedule:
        xh_hard = jnp.bool_(False)
    reject = reject | xh_hard
    X_out = jnp.where(xh_hard, X_s, X_out)
    Y_out = jnp.where(xh_hard, Y_s, Y_out)
    Z_out = jnp.where(xh_hard, Z_s, Z_out)
    C12_out = jnp.where(xh_hard, C12_s, C12_out)
    C13_out = jnp.where(xh_hard, C13_s, C13_out)
    N14_out = jnp.where(xh_hard, N14_s, N14_out)
    X3_out = jnp.where(xh_hard, X3_s, X3_out)
    logL_out = jnp.where(xh_hard, logL, logL_out)
    logTe_out = jnp.where(xh_hard, logTe, logTe_out)
    t_new = jnp.where(xh_hard, t, t_new)
    dt_next = jnp.where(xh_hard, dt * 0.5, dt_next)
    logT_out = jnp.where(xh_hard, prev_logT, logT_out)
    logP_out = jnp.where(xh_hard, prev_logP, logP_out)
    logRho_out = jnp.where(xh_hard, prev_logRho, logRho_out)
    log_R_out = jnp.where(xh_hard, jnp.float64(0.0), log_R_out)
    egf_out = jnp.where(xh_hard, jnp.float64(0.0), egf_out)
    # Also suppress reject for all remaining jnp.where calls below.
    if replay_schedule:
        reject = jnp.bool_(False)
    dt_next = jnp.maximum(dt_next, 1e5 * SECONDS_PER_YEAR)
    log_Tc_out = jnp.where(reject, prev_logT[-1], log_Tc)
    log_rhoc_out = jnp.where(reject, prev_logRho[-1], log_rhoc)
    dt_next = dt_next * (1.0 - jax.nn.sigmoid((log_Tc_out - 7.9) / 0.02))
    dt_next = jnp.where(t_new >= t_max_sec, 0.0, dt_next)
    dt_next = jnp.minimum(dt_next, jnp.maximum(t_max_sec - t_new, 0.0))
    if fixed_dt is not None:
        fdt = jnp.float64(fixed_dt * SECONDS_PER_YEAR)
        dt_next = jnp.where(t_new >= t_max_sec, 0.0, fdt)
        dt_next = jnp.minimum(dt_next, jnp.maximum(t_max_sec - t_new, 0.0))
    if replay_schedule:
        dt_next = jnp.where(xs_step > 0.0, xs_step, 0.0)
    if freeze_schedule:
        dt_next = jax.lax.stop_gradient(dt_next)  # GP-1: TIMESTEP
    xh_ratio_out = jax.lax.stop_gradient(xh_ratio)  # GP-1: TIMESTEP
    return (reject, dt_next, X_out, Y_out, Z_out, C12_out, C13_out, N14_out,
            X3_out,
            logL_out, logTe_out, t_new, logT_out, logP_out, logRho_out,
            log_R_out, egf_out, log_Tc_out, log_rhoc_out, xh_ratio_out)
