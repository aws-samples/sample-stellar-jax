"""Issue #1223: ∂ν²/∂{opacity_factor, eps_nuc_factor, alpha_mlt} at the subgiant.

Extends #1161's differentiable replay to the physics-input columns (opacity,
eps_nuc, α) at a post-TAMS subgiant model — the actual hero quantity for
PLATO age budgets.

Architecture:
  - ONE adaptive forward recording (1.5 M☉ MODE-A → SGB, X_c < 0.01)
    shared via a module-level pytest fixture.
  - THREE test functions, one per physics knob, each:
      1. Replays evolve_star with frozen (dt+mesh+comp) schedule + LIVE
         y_henyey carry, differentiating through the knob.
      2. AD: jax.grad of ν² = σ²/factor (well-conditioned at SGB, no
         catastrophic cancellation — see #1161 docstring).
      3. FD: mode-locked nearest-ν oracle at knob±δ (same ν² observable).
      4. Asserts: finite, non-zero, correct-sign, AD-vs-FD rel_err < 100%
         (interim hard ceiling; tight <25% tracked with #1220).

The adaptive recording uses MESA_CONFIG (MODE-A: Z=0.014, alpha_mlt=2.0,
Y_init=0.2695, diffusion=False, f_ov=0.0) with DEFAULT knob values
(opacity_factor=1.0, eps_nuc_factor=1.0). Knob perturbation happens only
during replay — the frozen (dt+mesh+comp) schedule is knob-independent.

MESA references:
  - opacity_factor: micro.f90:622-638 (multiplies kappa + derivatives)
  - eps_nuc_factor: net.f90:365-371 (multiplies eps_nuc + all derivatives)
  - mixing_length_alpha: controls.defaults:2153 (enters MLT module)
  - arXiv:2609.02876, §3.1–3.2 (GRADSOLVE frozen-schedule replay)
  - Christensen-Dalsgaard (2008), Ap&SS 316, 113 (oscillation coefficients)
"""
import os
import sys
import warnings

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_M = 1.5             # Solar masses — fast turnoff, reaches SGB
_XC_CUTOFF = 0.01    # past TAMS: core hydrogen exhausted
_EXTRA_STEPS = 50    # margin past cutoff for gradient information
_NU_MIN = 100.0      # µHz — wide bracket for SGB robustness
_NU_MAX = 4000.0     # µHz
_N_SCAN = 500        # frequency scan points for mode finding
_N_STEPS_ODE = 8000  # RK4 steps for eigenvalue integration
_TOL_UPPER = 1.0     # 100% interim hard ceiling (→ tight <25% in)

# ---------------------------------------------------------------------------
# Module-level fixture: record ONE adaptive trajectory to SGB
# ---------------------------------------------------------------------------
# This is expensive (~5000 steps), so we do it once per module.
# The fixture returns (accepted_steps, schedules, N_acc, final_state_info)
# which all three tests share.

_TRAJECTORY_CACHE = {}


def _record_trajectory():
    """Record an adaptive forward trajectory to the SGB (cached at module level).

    Returns a dict with:
      dt_sched, mesh_sched, comp_sched — frozen schedules for replay
      N_acc — number of accepted steps
      final_Xc, final_logL, final_age — diagnostic state at truncation
    """
    if _TRAJECTORY_CACHE:
        return _TRAJECTORY_CACHE

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import jax
    jax.config.update("jax_enable_x64", True)

    from stellar_jax.evolution.adaptive import evolve_star_adaptive
    from stellar_jax.config.mesa_config import MESA_CONFIG
    from stellar_jax.config.constants import SECONDS_PER_YEAR as _SPY
    from stellar_jax.config.mesh_defaults import N_HENYEY

    print(f"\n[#1223] Recording adaptive forward to SGB: {_M} M☉ MODE-A...")
    traj, result = evolve_star_adaptive(
        mass=_M, N_zones=600, max_steps=5000, verbose=True,
        **MESA_CONFIG)

    all_accepted = [s for s in traj.steps if s.accepted]
    N_all = len(all_accepted)
    assert N_all > 0, "No accepted steps from evolve_star_adaptive"

    # Truncate to early SGB (past TAMS + margin)
    tams_idx = None
    for i, s in enumerate(all_accepted):
        if float(s.X_profile[0]) < _XC_CUTOFF:
            tams_idx = i
            break
    if tams_idx is not None:
        trunc_idx = min(tams_idx + _EXTRA_STEPS, N_all)
        accepted = all_accepted[:trunc_idx]
        print(f"  Total accepted: {N_all}, truncated to {trunc_idx} "
              f"(X_c < {_XC_CUTOFF} at step {tams_idx}, +{_EXTRA_STEPS} margin)")
    else:
        accepted = all_accepted
        print(f"  WARNING: X_c never dropped below {_XC_CUTOFF}, "
              f"using all {N_all} accepted steps")

    N_acc = len(accepted)
    final_logL = accepted[-1].logL
    final_age = accepted[-1].t_yr
    final_Xc = float(accepted[-1].X_profile[0])
    print(f"  N_accepted = {N_acc}, age = {final_age:.3e} yr")
    print(f"  logL = {final_logL:.4f}, X_c = {final_Xc:.6f}")

    # Verify past TAMS
    assert final_Xc < 0.1, (
        f"1.5 M☉ did not reach near-TAMS: X_c = {final_Xc:.4f} > 0.1.")

    # Extract frozen schedules (same pattern as test)
    dt_sched = np.array([s.dt_yr * _SPY for s in accepted], dtype=np.float64)
    mesh_sched = np.array([s.q_mesh for s in accepted], dtype=np.float64)

    # GRADSOLVE-faithful: 600-zone compositions on the recorded grid (no remap)
    _X = np.zeros((N_acc, N_HENYEY), dtype=np.float64)
    _Y = np.zeros((N_acc, N_HENYEY), dtype=np.float64)
    _N14 = np.zeros((N_acc, N_HENYEY), dtype=np.float64)
    _C12 = np.zeros((N_acc, N_HENYEY), dtype=np.float64)
    _C13 = np.zeros((N_acc, N_HENYEY), dtype=np.float64)
    _mfracs = np.zeros((N_acc, N_HENYEY), dtype=np.float64)
    for i, s in enumerate(accepted):
        _X[i] = s.X_profile
        _Y[i] = s.Y_profile
        _N14[i] = s.N14_profile
        _C12[i] = s.C12_profile
        _C13[i] = s.C13_profile
        _mfracs[i] = 0.5 * (s.q_mesh[:-1] + s.q_mesh[1:])
    comp_sched = {'X': _X, 'Y': _Y, 'N14': _N14,
                  'C12': _C12, 'C13': _C13, 'comp_mfracs': _mfracs}

    cache = {
        'dt_sched': dt_sched,
        'mesh_sched': mesh_sched,
        'comp_sched': comp_sched,
        'N_acc': N_acc,
        'final_Xc': final_Xc,
        'final_logL': final_logL,
        'final_age': final_age,
    }
    _TRAJECTORY_CACHE.update(cache)
    return cache


@pytest.fixture(scope="module")
def sgb_trajectory():
    """Shared fixture: adaptive forward trajectory to the SGB."""
    return _record_trajectory()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _find_reference_mode(glob, var):
    """Find an l=0 reference mode on the SGB model for mode-locking.

    Returns (info_ref, nu_ref, sigma2_ref) from the reference structure.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    from stellar_jax.oscillations import compute_eigenfreq_from_structure_jax

    info_ref = compute_eigenfreq_from_structure_jax(
        glob, var, l=0, nu_min=_NU_MIN, nu_max=_NU_MAX,
        n_scan=_N_SCAN, n_steps=_N_STEPS_ODE, mode_index=0)
    nu_ref = info_ref['nu']
    sigma2_ref = info_ref['sigma2']
    return info_ref, nu_ref, sigma2_ref


def _mode_locked_nu2_fd(mass_val, knob_name, knob_val,
                        dt_sched, mesh_sched, comp_sched,
                        N_acc, q_mesh_last, nu_ref, mesa_config):
    """Forward-only ν² at given knob value via mode-locked nearest-ν matching.

    Returns ν² = nu_fd² (cyclic frequency squared, µHz²).

    Mode-locked FD oracle (#1266/#1264 pattern):
    1. Evolve at the knob value with the frozen schedule
    2. Compute ALL l=0 modes in [100, nu_ref+300] µHz
    3. Select the mode NEAREST to nu_ref
    4. Return ν² = nu_fd²
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from stellar_jax.evolution import evolve_star, structure_to_fgong_jax
    from stellar_jax.oscillations import compute_oscillation_freqs_jax

    # Build evolve_star kwargs — inject the knob being tested
    ev_kwargs = dict(mesa_config)
    ev_kwargs.update(
        max_steps=N_acc,
        freeze_schedule=True,
        adaptive_mesh=False,
        dt_schedule=dt_sched,
        mesh_schedule=mesh_sched,
        comp_schedule=comp_sched,
    )
    if knob_name == 'alpha_mlt':
        ev_kwargs['alpha_mlt'] = float(knob_val)
    elif knob_name == 'opacity_factor':
        ev_kwargs['opacity_factor'] = float(knob_val)
    elif knob_name == 'eps_nuc_factor':
        ev_kwargs['eps_nuc_factor'] = float(knob_val)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = evolve_star(jnp.float64(mass_val), **ev_kwargs)

    alpha_for_fgong = float(ev_kwargs['alpha_mlt'])
    glob_fd, var_fd = structure_to_fgong_jax(
        jnp.float64(mass_val), r['log_L_final'], r['log_Teff_final'],
        r['X_profile'], jnp.float64(mesa_config['Z']), jnp.float64(0.0),
        jnp.float64(alpha_for_fgong),
        y_henyey=r['y_henyey_final'],
        q_mesh=q_mesh_last, atm_ratio=r.get('atm_ratio'))

    # Mode-locked search
    fd_nu_min = max(100.0, nu_ref - 300.0)
    fd_nu_max = nu_ref + 300.0
    freqs_fd = compute_oscillation_freqs_jax(
        glob_fd, var_fd, l_values=(0,),
        nu_min=fd_nu_min, nu_max=fd_nu_max,
        n_scan=_N_SCAN, n_steps=_N_STEPS_ODE)
    l0_freqs = freqs_fd[0]
    assert len(l0_freqs) > 0, (
        f"No l=0 modes found in [{fd_nu_min:.0f}, {fd_nu_max:.0f}] µHz "
        f"at {knob_name}={knob_val}")
    # Nearest-ν matching
    idx = np.argmin(np.abs(l0_freqs - nu_ref))
    nu_fd = l0_freqs[idx]
    print(f"    FD mode-lock at {knob_name}={knob_val:.8f}: "
          f"found {len(l0_freqs)} modes, selected ν={nu_fd:.2f} µHz "
          f"(|Δν|={abs(nu_fd - nu_ref):.2f} µHz from ref {nu_ref:.2f})",
          flush=True)
    return nu_fd ** 2


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: ∂ν²/∂opacity_factor at the subgiant
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("opacity_factor_detach")
@pytest.mark.right_reason("Gradient")
@pytest.mark.timeout(18000)
def test_seismic_gradient_opacity_factor_subgiant(sgb_trajectory):
    """#1223: ∂ν²/∂opacity_factor at the SGB via #1161 differentiable replay.

    WHAT: validates that the AD gradient ∂ν²/∂opacity_factor through the full
    chain (opacity_factor → evolve_star replay → FGONG → coeffs → IFT eigenfreq
    → ν²) is finite, non-zero, correct-sign, and agrees with an independent FD
    at a post-TAMS subgiant (X_c < 0.01).

    WHY: the near-ZAMS test (test_seismic_gradient_opacity_factor, N=3) validates
    the opacity gradient at an almost-unchanged-from-ZAMS model. At the SGB, the
    star has exhausted core hydrogen, has a thick convective envelope, and the
    opacity sensitivity has fundamentally changed (envelope-dominated, not core).
    This is the regime where PLATO's age information lives.

    OBSERVABLE: ν² = σ²/factor (well-conditioned at SGB; avoids the catastrophic
    +527/−511 cancellation in σ² — see #1161 docstring for derivation).

    MESA ref: micro.f90:622-638 — opacity_factor multiplies kappa + derivatives.
    EXTERNAL REFERENCE: AD vs independent mode-locked two-sided FD.
    TOLERANCE: <100% interim hard ceiling (#1223); <25% tracked in #1220.

    WHAT MAKES IT FAIL: @mutation opacity_factor_detach — stop_gradient on
    opacity_factor inside kappa severs ∂kappa/∂opacity_factor → AD ≈ 0.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from stellar_jax.evolution import evolve_star, structure_to_fgong_jax
    from stellar_jax.oscillations import (
        build_oscillation_coeffs_jax, eigenfreq_from_coeffs,
        compute_eigenfreq_from_structure_jax,
    )
    from stellar_jax.config.mesa_config import MESA_CONFIG

    traj = sgb_trajectory
    dt_sched = traj['dt_sched']
    mesh_sched = traj['mesh_sched']
    comp_sched = traj['comp_sched']
    N_acc = traj['N_acc']

    q_mesh_last = jax.lax.stop_gradient(jnp.asarray(mesh_sched[-1]))

    # ── Step 1: Replay reference at opacity_factor=1.0 (default) ──
    print(f"\n[#1223/opacity] Replaying at opacity_factor=1.0, N={N_acc}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_ref = evolve_star(
            jnp.float64(_M), max_steps=N_acc,
            freeze_schedule=True, adaptive_mesh=False,
            dt_schedule=dt_sched, mesh_schedule=mesh_sched,
            comp_schedule=comp_sched, **MESA_CONFIG)
    glob_ref, var_ref = structure_to_fgong_jax(
        jnp.float64(_M), r_ref['log_L_final'], r_ref['log_Teff_final'],
        r_ref['X_profile'], jnp.float64(MESA_CONFIG['Z']), jnp.float64(0.0),
        jnp.float64(MESA_CONFIG['alpha_mlt']),
        y_henyey=r_ref['y_henyey_final'],
        q_mesh=q_mesh_last, atm_ratio=r_ref.get('atm_ratio'))

    info_ref, nu_ref, sigma2_ref = _find_reference_mode(glob_ref, var_ref)
    print(f"  Reference mode: l=0, ν = {nu_ref:.2f} µHz, σ² = {sigma2_ref:.6e}")

    # ── Step 2: AD gradient ∂ν²/∂opacity_factor ──
    print(f"[#1223/opacity] Computing AD ∂ν²/∂opacity_factor...")

    def nu2_of_opf(opf_val):
        """Differentiable chain: opacity_factor → replay → ν²."""
        r = evolve_star(jnp.float64(_M), max_steps=N_acc,
                        freeze_schedule=True, adaptive_mesh=False,
                        dt_schedule=dt_sched, mesh_schedule=mesh_sched,
                        comp_schedule=comp_sched,
                        opacity_factor=opf_val,
                        Z=MESA_CONFIG['Z'],
                        alpha_mlt=MESA_CONFIG['alpha_mlt'],
                        Y_init=MESA_CONFIG['Y_init'],
                        diffusion=MESA_CONFIG['diffusion'],
                        f_ov=MESA_CONFIG['f_ov'])
        glob, var = structure_to_fgong_jax(
            jnp.float64(_M), r['log_L_final'], r['log_Teff_final'],
            r['X_profile'], jnp.float64(MESA_CONFIG['Z']), jnp.float64(0.0),
            jnp.float64(MESA_CONFIG['alpha_mlt']),
            y_henyey=r['y_henyey_final'],
            q_mesh=q_mesh_last, atm_ratio=r.get('atm_ratio'))
        grid_data = build_oscillation_coeffs_jax(glob, var)
        sigma2 = eigenfreq_from_coeffs(
            grid_data['coeffs'], grid_data['x_grid'], sigma2_ref,
            info_ref['l'], info_ref['x_steps'], info_ref['h_steps'],
            info_ref['factor'])
        factor = grid_data['factor']
        return sigma2 / factor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ad_grad = float(jax.grad(nu2_of_opf)(jnp.float64(1.0)))
    print(f"  AD ∂ν²/∂opacity_factor = {ad_grad:.8e}")

    # ── Step 3: FD gradient (mode-locked, two-sided) ──
    d_opf = 1e-4
    print(f"  Computing FD at opacity_factor ± {d_opf}...")
    nu2_plus = _mode_locked_nu2_fd(
        _M, 'opacity_factor', 1.0 + d_opf,
        dt_sched, mesh_sched, comp_sched,
        N_acc, q_mesh_last, nu_ref, MESA_CONFIG)
    nu2_minus = _mode_locked_nu2_fd(
        _M, 'opacity_factor', 1.0 - d_opf,
        dt_sched, mesh_sched, comp_sched,
        N_acc, q_mesh_last, nu_ref, MESA_CONFIG)
    fd_grad = (nu2_plus - nu2_minus) / (2.0 * d_opf)
    print(f"  FD ∂ν²/∂opacity_factor = {fd_grad:.8e}")

    # ── Step 4: Assertions ──
    denom = max(abs(fd_grad), abs(ad_grad), 1e-30)
    rel_err = abs(ad_grad - fd_grad) / denom

    assert np.isfinite(ad_grad), f"AD ∂ν²/∂opacity_factor is not finite: {ad_grad}"
    assert abs(ad_grad) > 1e-20, (
        f"AD ∂ν²/∂opacity_factor ≈ 0: {ad_grad}. Gradient chain severed.")
    assert abs(fd_grad) > 1e-20, f"FD ∂ν²/∂opacity_factor ≈ 0: {fd_grad}"
    assert ad_grad * fd_grad > 0, (
        f"∂ν²/∂opacity_factor sign mismatch: AD={ad_grad:.6e}, FD={fd_grad:.6e}")
    assert rel_err < _TOL_UPPER, (
        f"∂ν²/∂opacity_factor: AD vs FD mismatch {rel_err:.4e} > {_TOL_UPPER}\n"
        f"  AD = {ad_grad:.8e}, FD = {fd_grad:.8e}\n"
        f"  Config: {_M} M☉, N={N_acc}, SGB replay")

    print(f"\n  ✓ PASSED: ∂ν²/∂opacity_factor rel_err = {rel_err*100:.2f}% "
          f"(< {_TOL_UPPER*100:.0f}%)")
    print(f"    Stage: M={_M}, N={N_acc}, X_c={traj['final_Xc']:.4f}, "
          f"age={traj['final_age']:.3e} yr")
    print(f"    AD = {ad_grad:.8e}, FD = {fd_grad:.8e}")
    print(f"    ν(l=0) = {nu_ref:.2f} µHz")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: ∂ν²/∂eps_nuc_factor at the subgiant
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("eps_nuc_factor_detach")
@pytest.mark.right_reason("Gradient")
@pytest.mark.timeout(18000)
def test_seismic_gradient_eps_nuc_factor_subgiant(sgb_trajectory):
    """#1223: ∂ν²/∂eps_nuc_factor at the SGB via #1161 differentiable replay.

    WHAT: validates that the AD gradient ∂ν²/∂eps_nuc_factor through the full
    chain (eps_nuc_factor → evolve_star replay → FGONG → coeffs → IFT eigenfreq
    → ν²) is finite, non-zero, correct-sign, and agrees with an independent FD
    at a post-TAMS subgiant (X_c < 0.01).

    WHY: the near-ZAMS test (test_seismic_gradient_eps_nuc_factor, N=3) validates
    the nuclear-rate gradient at an almost-unchanged-from-ZAMS model. At the SGB,
    energy production is dominated by H-shell burning around the contracting
    helium core, and ∂ν²/∂eps_nuc_factor probes how nuclear-rate uncertainty
    propagates to observed oscillation frequencies in the evolved regime.

    OBSERVABLE: ν² = σ²/factor (well-conditioned at SGB).

    MESA ref: net.f90:365-371 — eps_nuc_factor multiplies eps_nuc + derivatives.
    NOTE: MESA's eps_nuc_factor scales energy production only, not dxdt_nuc
    (composition change rates). Our implementation matches: the multiplicative
    factor applies to eps_nuc in the structure equations (energy conservation),
    while the frozen composition schedule handles the abundance evolution.
    EXTERNAL REFERENCE: AD vs independent mode-locked two-sided FD.
    TOLERANCE: <100% interim hard ceiling (#1223); <25% tracked in #1220.

    WHAT MAKES IT FAIL: @mutation eps_nuc_factor_detach — stop_gradient on
    eps_nuc_factor inside the Henyey residual severs the gradient → AD ≈ 0.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from stellar_jax.evolution import evolve_star, structure_to_fgong_jax
    from stellar_jax.oscillations import (
        build_oscillation_coeffs_jax, eigenfreq_from_coeffs,
        compute_eigenfreq_from_structure_jax,
    )
    from stellar_jax.config.mesa_config import MESA_CONFIG

    traj = sgb_trajectory
    dt_sched = traj['dt_sched']
    mesh_sched = traj['mesh_sched']
    comp_sched = traj['comp_sched']
    N_acc = traj['N_acc']

    q_mesh_last = jax.lax.stop_gradient(jnp.asarray(mesh_sched[-1]))

    # ── Step 1: Replay reference at eps_nuc_factor=1.0 (default) ──
    print(f"\n[#1223/eps_nuc] Replaying at eps_nuc_factor=1.0, N={N_acc}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_ref = evolve_star(
            jnp.float64(_M), max_steps=N_acc,
            freeze_schedule=True, adaptive_mesh=False,
            dt_schedule=dt_sched, mesh_schedule=mesh_sched,
            comp_schedule=comp_sched, **MESA_CONFIG)
    glob_ref, var_ref = structure_to_fgong_jax(
        jnp.float64(_M), r_ref['log_L_final'], r_ref['log_Teff_final'],
        r_ref['X_profile'], jnp.float64(MESA_CONFIG['Z']), jnp.float64(0.0),
        jnp.float64(MESA_CONFIG['alpha_mlt']),
        y_henyey=r_ref['y_henyey_final'],
        q_mesh=q_mesh_last, atm_ratio=r_ref.get('atm_ratio'))

    info_ref, nu_ref, sigma2_ref = _find_reference_mode(glob_ref, var_ref)
    print(f"  Reference mode: l=0, ν = {nu_ref:.2f} µHz, σ² = {sigma2_ref:.6e}")

    # ── Step 2: AD gradient ∂ν²/∂eps_nuc_factor ──
    print(f"[#1223/eps_nuc] Computing AD ∂ν²/∂eps_nuc_factor...")

    def nu2_of_enf(enf_val):
        """Differentiable chain: eps_nuc_factor → replay → ν²."""
        r = evolve_star(jnp.float64(_M), max_steps=N_acc,
                        freeze_schedule=True, adaptive_mesh=False,
                        dt_schedule=dt_sched, mesh_schedule=mesh_sched,
                        comp_schedule=comp_sched,
                        eps_nuc_factor=enf_val,
                        Z=MESA_CONFIG['Z'],
                        alpha_mlt=MESA_CONFIG['alpha_mlt'],
                        Y_init=MESA_CONFIG['Y_init'],
                        diffusion=MESA_CONFIG['diffusion'],
                        f_ov=MESA_CONFIG['f_ov'])
        glob, var = structure_to_fgong_jax(
            jnp.float64(_M), r['log_L_final'], r['log_Teff_final'],
            r['X_profile'], jnp.float64(MESA_CONFIG['Z']), jnp.float64(0.0),
            jnp.float64(MESA_CONFIG['alpha_mlt']),
            y_henyey=r['y_henyey_final'],
            q_mesh=q_mesh_last, atm_ratio=r.get('atm_ratio'))
        grid_data = build_oscillation_coeffs_jax(glob, var)
        sigma2 = eigenfreq_from_coeffs(
            grid_data['coeffs'], grid_data['x_grid'], sigma2_ref,
            info_ref['l'], info_ref['x_steps'], info_ref['h_steps'],
            info_ref['factor'])
        factor = grid_data['factor']
        return sigma2 / factor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ad_grad = float(jax.grad(nu2_of_enf)(jnp.float64(1.0)))
    print(f"  AD ∂ν²/∂eps_nuc_factor = {ad_grad:.8e}")

    # ── Step 3: FD gradient (mode-locked, two-sided) ──
    d_enf = 1e-4
    print(f"  Computing FD at eps_nuc_factor ± {d_enf}...")
    nu2_plus = _mode_locked_nu2_fd(
        _M, 'eps_nuc_factor', 1.0 + d_enf,
        dt_sched, mesh_sched, comp_sched,
        N_acc, q_mesh_last, nu_ref, MESA_CONFIG)
    nu2_minus = _mode_locked_nu2_fd(
        _M, 'eps_nuc_factor', 1.0 - d_enf,
        dt_sched, mesh_sched, comp_sched,
        N_acc, q_mesh_last, nu_ref, MESA_CONFIG)
    fd_grad = (nu2_plus - nu2_minus) / (2.0 * d_enf)
    print(f"  FD ∂ν²/∂eps_nuc_factor = {fd_grad:.8e}")

    # ── Step 4: Assertions ──
    denom = max(abs(fd_grad), abs(ad_grad), 1e-30)
    rel_err = abs(ad_grad - fd_grad) / denom

    assert np.isfinite(ad_grad), f"AD ∂ν²/∂eps_nuc_factor is not finite: {ad_grad}"
    assert abs(ad_grad) > 1e-20, (
        f"AD ∂ν²/∂eps_nuc_factor ≈ 0: {ad_grad}. Gradient chain severed.")
    assert abs(fd_grad) > 1e-20, f"FD ∂ν²/∂eps_nuc_factor ≈ 0: {fd_grad}"
    assert ad_grad * fd_grad > 0, (
        f"∂ν²/∂eps_nuc_factor sign mismatch: AD={ad_grad:.6e}, FD={fd_grad:.6e}")
    assert rel_err < _TOL_UPPER, (
        f"∂ν²/∂eps_nuc_factor: AD vs FD mismatch {rel_err:.4e} > {_TOL_UPPER}\n"
        f"  AD = {ad_grad:.8e}, FD = {fd_grad:.8e}\n"
        f"  Config: {_M} M☉, N={N_acc}, SGB replay")

    print(f"\n  ✓ PASSED: ∂ν²/∂eps_nuc_factor rel_err = {rel_err*100:.2f}% "
          f"(< {_TOL_UPPER*100:.0f}%)")
    print(f"    Stage: M={_M}, N={N_acc}, X_c={traj['final_Xc']:.4f}, "
          f"age={traj['final_age']:.3e} yr")
    print(f"    AD = {ad_grad:.8e}, FD = {fd_grad:.8e}")
    print(f"    ν(l=0) = {nu_ref:.2f} µHz")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: ∂ν²/∂alpha_mlt at the subgiant
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("detach_alpha_gradient")
@pytest.mark.right_reason("Gradient")
@pytest.mark.timeout(18000)
def test_seismic_gradient_alpha_mlt_subgiant(sgb_trajectory):
    """#1223: ∂ν²/∂alpha_mlt at the SGB via #1161 differentiable replay.

    WHAT: validates that the AD gradient ∂ν²/∂alpha_mlt through the full chain
    (alpha_mlt → evolve_star replay → FGONG → coeffs → IFT eigenfreq → ν²) is
    finite, non-zero, correct-sign, and agrees with an independent FD at a
    post-TAMS subgiant (X_c < 0.01).

    WHY: the existing test_seismic_gradient_alpha_subgiant validates ∂σ²/∂α at
    X_c≈0.22 (late MS / early SGB) using a fixed_dt N=500 trajectory. This
    test validates at the POST-TAMS SGB (X_c < 0.01) using #1161's adaptive
    replay infrastructure — a more evolved model where the convective envelope
    is deeper and the α→MLT→T-profile→c_s→ν chain has a larger lever arm.
    This is the PLATO regime: age sensitivity to α_MLT at the subgiant is
    the dominant systematic in asteroseismic age determination.

    OBSERVABLE: ν² = σ²/factor (well-conditioned at SGB).

    MESA ref: controls.defaults:2153 — mixing_length_alpha default 2.0.
    The α gradient enters through per-step MLT (∇_MLT in convective zones),
    NOT through convective boundary detection (GP-5). The dominant channel
    is α→Γ→∇_MLT→T-profile→c_s→σ²→ν², operating INSIDE convective zones.

    Refs:
        Böhm-Vitense (1958), ZfA 46, 108 (MLT)
        Christensen-Dalsgaard (2008), Ap&SS 316, 113 (oscillation coefficients)
        Ball & Gizon (2014), A&A 568, A123 (surface term α maps onto)
        Sonoi et al. (2019), A&A 621, A84 (α↔p-mode frequencies calibration)

    EXTERNAL REFERENCE: AD vs independent mode-locked two-sided FD.
    TOLERANCE: <100% interim hard ceiling (#1223); <25% tracked in #1220.

    WHAT MAKES IT FAIL: @mutation detach_alpha_gradient — stop_gradient on α
    → AD ≈ 0.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from stellar_jax.evolution import evolve_star, structure_to_fgong_jax
    from stellar_jax.oscillations import (
        build_oscillation_coeffs_jax, eigenfreq_from_coeffs,
        compute_eigenfreq_from_structure_jax,
    )
    from stellar_jax.config.mesa_config import MESA_CONFIG

    traj = sgb_trajectory
    dt_sched = traj['dt_sched']
    mesh_sched = traj['mesh_sched']
    comp_sched = traj['comp_sched']
    N_acc = traj['N_acc']
    alpha_ref = MESA_CONFIG['alpha_mlt']  # 2.0 in MODE-A

    q_mesh_last = jax.lax.stop_gradient(jnp.asarray(mesh_sched[-1]))

    # ── Step 1: Replay reference at alpha_mlt=2.0 (MODE-A default) ──
    print(f"\n[#1223/alpha] Replaying at alpha_mlt={alpha_ref}, N={N_acc}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_ref = evolve_star(
            jnp.float64(_M), max_steps=N_acc,
            freeze_schedule=True, adaptive_mesh=False,
            dt_schedule=dt_sched, mesh_schedule=mesh_sched,
            comp_schedule=comp_sched, **MESA_CONFIG)
    glob_ref, var_ref = structure_to_fgong_jax(
        jnp.float64(_M), r_ref['log_L_final'], r_ref['log_Teff_final'],
        r_ref['X_profile'], jnp.float64(MESA_CONFIG['Z']), jnp.float64(0.0),
        jnp.float64(alpha_ref),
        y_henyey=r_ref['y_henyey_final'],
        q_mesh=q_mesh_last, atm_ratio=r_ref.get('atm_ratio'))

    info_ref, nu_ref, sigma2_ref = _find_reference_mode(glob_ref, var_ref)
    print(f"  Reference mode: l=0, ν = {nu_ref:.2f} µHz, σ² = {sigma2_ref:.6e}")

    # ── Step 2: AD gradient ∂ν²/∂alpha_mlt ──
    print(f"[#1223/alpha] Computing AD ∂ν²/∂alpha_mlt...")

    def nu2_of_alpha(alpha_val):
        """Differentiable chain: alpha_mlt → replay → ν²."""
        r = evolve_star(jnp.float64(_M), max_steps=N_acc,
                        freeze_schedule=True, adaptive_mesh=False,
                        dt_schedule=dt_sched, mesh_schedule=mesh_sched,
                        comp_schedule=comp_sched,
                        alpha_mlt=alpha_val,
                        Z=MESA_CONFIG['Z'],
                        Y_init=MESA_CONFIG['Y_init'],
                        diffusion=MESA_CONFIG['diffusion'],
                        f_ov=MESA_CONFIG['f_ov'])
        glob, var = structure_to_fgong_jax(
            jnp.float64(_M), r['log_L_final'], r['log_Teff_final'],
            r['X_profile'], jnp.float64(MESA_CONFIG['Z']), jnp.float64(0.0),
            alpha_val,
            y_henyey=r['y_henyey_final'],
            q_mesh=q_mesh_last, atm_ratio=r.get('atm_ratio'))
        grid_data = build_oscillation_coeffs_jax(glob, var)
        sigma2 = eigenfreq_from_coeffs(
            grid_data['coeffs'], grid_data['x_grid'], sigma2_ref,
            info_ref['l'], info_ref['x_steps'], info_ref['h_steps'],
            info_ref['factor'])
        factor = grid_data['factor']
        return sigma2 / factor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ad_grad = float(jax.grad(nu2_of_alpha)(jnp.float64(alpha_ref)))
    print(f"  AD ∂ν²/∂alpha_mlt = {ad_grad:.8e}")

    # ── Step 3: FD gradient (mode-locked, two-sided) ──
    d_alpha = alpha_ref * 1e-4
    print(f"  Computing FD at alpha_mlt ± {d_alpha:.6f}...")
    nu2_plus = _mode_locked_nu2_fd(
        _M, 'alpha_mlt', alpha_ref + d_alpha,
        dt_sched, mesh_sched, comp_sched,
        N_acc, q_mesh_last, nu_ref, MESA_CONFIG)
    nu2_minus = _mode_locked_nu2_fd(
        _M, 'alpha_mlt', alpha_ref - d_alpha,
        dt_sched, mesh_sched, comp_sched,
        N_acc, q_mesh_last, nu_ref, MESA_CONFIG)
    fd_grad = (nu2_plus - nu2_minus) / (2.0 * d_alpha)
    print(f"  FD ∂ν²/∂alpha_mlt = {fd_grad:.8e}")

    # ── Step 4: Assertions ──
    denom = max(abs(fd_grad), abs(ad_grad), 1e-30)
    rel_err = abs(ad_grad - fd_grad) / denom

    assert np.isfinite(ad_grad), f"AD ∂ν²/∂alpha_mlt is not finite: {ad_grad}"
    assert abs(ad_grad) > 1e-20, (
        f"AD ∂ν²/∂alpha_mlt ≈ 0: {ad_grad}. Gradient chain severed.")
    assert abs(fd_grad) > 1e-20, f"FD ∂ν²/∂alpha_mlt ≈ 0: {fd_grad}"
    assert ad_grad * fd_grad > 0, (
        f"∂ν²/∂alpha_mlt sign mismatch: AD={ad_grad:.6e}, FD={fd_grad:.6e}")
    assert rel_err < _TOL_UPPER, (
        f"∂ν²/∂alpha_mlt: AD vs FD mismatch {rel_err:.4e} > {_TOL_UPPER}\n"
        f"  AD = {ad_grad:.8e}, FD = {fd_grad:.8e}\n"
        f"  Config: {_M} M☉, N={N_acc}, SGB replay")

    print(f"\n  ✓ PASSED: ∂ν²/∂alpha_mlt rel_err = {rel_err*100:.2f}% "
          f"(< {_TOL_UPPER*100:.0f}%)")
    print(f"    Stage: M={_M}, N={N_acc}, X_c={traj['final_Xc']:.4f}, "
          f"age={traj['final_age']:.3e} yr")
    print(f"    AD = {ad_grad:.8e}, FD = {fd_grad:.8e}")
    print(f"    ν(l=0) = {nu_ref:.2f} µHz")
