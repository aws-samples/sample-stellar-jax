#!/usr/bin/env python3
"""Generate golden-master reference values for the refactor characterization net.

Usage:
    python scripts/generate_golden_master.py [--fast-only]

This script runs the code at its current state and records exact outputs for:
  1. Composition operators (burn_cno, mix_composition, diffuse_composition)
  2. Oscillation solver (eigenfreq from fixed FGONG)
  3. Forward evolution (evolve_star at fixed_dt)
  4. AD gradients (∂logL/∂M, ∂logTe/∂α)

The output is saved to tests/golden_master_ref.npz.

Flags:
  --fast-only   Generate only the fast references (composition + oscillation).
                Skip the heavy evolve_star + gradient computations.

WHEN TO RE-BASELINE: Only when a deliberate, justified behavioral change is made
(e.g. a bug fix that changes numerical output). The commit message MUST document
what changed and why. Never re-baseline to "make tests pass" without understanding
the root cause.
"""

import os
import sys
import gzip
import time
import argparse
import tempfile

import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

from stellar_jax.config.constants import SECONDS_PER_YEAR
from stellar_jax.config.mesh_defaults import N_COMP, COMP_MFRACS


def make_synthetic_shell_data():
    """Create the same fixed synthetic shell_data used in the tests.

    Layout (henyey.py:2753 extract_shell_data_from_henyey):
      col 0: eps_nuc, col 1: q, col 2: nabla_rad, col 3: nabla_ad,
      col 4: Hp/R, col 5: dm/dr_norm, col 6: log10(T), col 7: log10(rho),
      col 8: nabla, col 9: log10(P)
    """
    n = N_COMP
    mf = np.array(COMP_MFRACS)
    eps_nuc = 10.0 * np.exp(-mf / 0.05)
    nabla_ad = np.full(n, 0.4)
    nabla_rad = np.where(mf < 0.10, 0.8, 0.25)
    hp_over_R = 0.1 * (1.0 - mf) + 0.01
    dm_dr_norm = np.ones(n) * 3.0
    logT = np.linspace(7.17, 3.78, n)
    logrho = np.linspace(2.18, -7.0, n)
    nabla = np.where(mf < 0.10, nabla_ad, nabla_rad)
    logP = np.linspace(17.3, 5.0, n)
    shell_data = np.column_stack([eps_nuc, mf, nabla_rad, nabla_ad, hp_over_R,
                                  dm_dr_norm, logT, logrho, nabla, logP])
    return shell_data.astype(np.float64)


def generate_composition_refs():
    """Generate burn_cno, mix_composition, diffuse_composition references."""
    import stellar_jax.evolution as evolution

    print("=" * 60)
    print("§1. Composition operator references")
    print("=" * 60)

    shell_data = make_synthetic_shell_data()
    Z = 0.014

    # burn_cno
    print("  burn_cno...", end=" ", flush=True)
    t0 = time.time()
    X = np.full(N_COMP, 0.7)
    C12_init = np.full(N_COMP, 0.142 * Z)
    C13_init = np.full(N_COMP, 0.0036 * Z)
    N14_init = np.full(N_COMP, 0.049 * Z)
    dt = 2e6 * SECONDS_PER_YEAR

    C12_out, C13_out, N14_out = evolution.burn_cno(
        jnp.array(C12_init), jnp.array(C13_init), jnp.array(N14_init),
        jnp.array(X), jnp.array(shell_data), jnp.float64(dt)
    )
    print(f"done ({time.time() - t0:.1f}s)")

    # mix_composition
    print("  mix_composition...", end=" ", flush=True)
    t0 = time.time()
    X_mix = np.linspace(0.35, 0.70, N_COMP)
    X_mixed = evolution.mix_composition(
        jnp.array(X_mix), jnp.array(shell_data), M_solar=1.0, f_ov=0.016
    )
    print(f"done ({time.time() - t0:.1f}s)")

    # diffuse_composition
    print("  diffuse_composition...", end=" ", flush=True)
    t0 = time.time()
    X_diff = np.full(N_COMP, 0.70)
    Y_diff = np.full(N_COMP, 0.28)
    diff_result = evolution.diffuse_composition(
        jnp.array(X_diff), jnp.array(Y_diff), jnp.array(shell_data),
        jnp.float64(dt), M_solar=1.0
    )
    if isinstance(diff_result, tuple):
        diff_X, diff_Y = np.array(diff_result[0]), np.array(diff_result[1])
    else:
        diff_X = np.array(diff_result)
        diff_Y = None
    print(f"done ({time.time() - t0:.1f}s)")

    refs = {
        'burn_cno_C12': np.array(C12_out),
        'burn_cno_C13': np.array(C13_out),
        'burn_cno_N14': np.array(N14_out),
        'mix_X': np.array(X_mixed),
        'diffuse_X': diff_X,
    }
    if diff_Y is not None:
        refs['diffuse_Y'] = diff_Y

    return refs


def generate_oscillation_ref():
    """Generate eigenfrequency reference from a fixed FGONG."""
    import stellar_jax.oscillations as oscillations
    import stellar_jax.oscillations as osc_mod

    print("=" * 60)
    print("§2. Oscillation solver reference")
    print("=" * 60)

    fgong_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "stellar_jax", "data", "mesa_comparison", "profiles", "1.0Msun", "midMS.FGONG.gz"
    )
    if not os.path.isfile(fgong_path):
        print(f"  WARNING: FGONG not found at {fgong_path}, skipping.")
        return {}

    print("  compute_eigenfreq_differentiable (l=0)...", end=" ", flush=True)
    t0 = time.time()

    with gzip.open(fgong_path, 'rt') as gz:
        content = gz.read()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.FGONG', delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        glob, var = oscillations.read_fgong(tmp_path)
    finally:
        os.unlink(tmp_path)

    result = osc_mod.compute_eigenfreq_differentiable(
        glob, var, l=0, nu_min=2500.0, nu_max=3500.0
    )
    sigma2 = float(result['sigma2'])
    print(f"done ({time.time() - t0:.1f}s), σ²={sigma2:.10e}")

    return {'eigenfreq_sigma2': np.float64(sigma2)}


def generate_evolution_ref():
    """Generate evolve_star forward reference."""
    import stellar_jax.evolution as evolution

    print("=" * 60)
    print("§3. Forward evolution reference (WARNING: ~10-16 min JIT compile)")
    print("=" * 60)

    print("  evolve_star(1.0, fixed_dt=2e6, max_steps=200)...", flush=True)
    t0 = time.time()

    result = evolution.evolve_star(
        1.0, Z=0.014, max_steps=200, fixed_dt=2e6,
        diffusion=True, adaptive_mesh=False
    )
    # Block until complete
    jax.block_until_ready(result['log_L'])
    elapsed = time.time() - t0
    print(f"  done ({elapsed:.0f}s)")
    print(f"    log_L[-1]    = {float(result['log_L'][-1]):.15e}")
    print(f"    log_Teff[-1] = {float(result['log_Teff'][-1]):.15e}")
    print(f"    log_Tc[-1]   = {float(result['log_Tc'][-1]):.15e}")
    print(f"    log_rhoc[-1] = {float(result['log_rhoc'][-1]):.15e}")

    return {
        'evolve_logL_final': np.float64(float(result['log_L'][-1])),
        'evolve_logTe_final': np.float64(float(result['log_Teff'][-1])),
        'evolve_logTc_final': np.float64(float(result['log_Tc'][-1])),
        'evolve_logrhoc_final': np.float64(float(result['log_rhoc'][-1])),
        'evolve_X_profile': np.array(result['X_profile']),
        'evolve_y_henyey_final': np.array(result['y_henyey_final']),
    }


def generate_gradient_refs():
    """Generate AD gradient references (∂logL/∂M, ∂logTe/∂α)."""
    import stellar_jax.evolution as evolution

    print("=" * 60)
    print("§4. Gradient references (WARNING: ~10-20 min backward JIT compile)")
    print("=" * 60)

    # ∂logL/∂M
    print("  jax.grad(logL[-1])(M=1.0)...", flush=True)
    t0 = time.time()

    def logL_final(mass):
        r = evolution.evolve_star(
            mass, Z=0.014, max_steps=200, fixed_dt=2e6,
            diffusion=True, adaptive_mesh=False
        )
        return r['log_L'][-1]

    grad_M = float(jax.grad(logL_final)(jnp.float64(1.0)))
    elapsed = time.time() - t0
    print(f"  done ({elapsed:.0f}s), ∂logL/∂M = {grad_M:.15e}")

    # ∂logTe/∂α
    print("  jax.grad(logTe[-1])(α=1.9)...", flush=True)
    t0 = time.time()

    def logTe_final(alpha):
        r = evolution.evolve_star(
            1.0, Z=0.014, max_steps=200, fixed_dt=2e6, alpha_mlt=alpha,
            diffusion=True, adaptive_mesh=False
        )
        return r['log_Teff'][-1]

    grad_alpha = float(jax.grad(logTe_final)(jnp.float64(1.9)))
    elapsed = time.time() - t0
    print(f"  done ({elapsed:.0f}s), ∂logTe/∂α = {grad_alpha:.15e}")

    return {
        'grad_dlogL_dM': np.float64(grad_M),
        'grad_dlogTe_dalpha': np.float64(grad_alpha),
    }


def generate_seismic_gradient_ref():
    """Generate AD seismic gradient reference (∂σ²/∂M through full E2E chain)."""
    import stellar_jax.evolution as evolution
    import warnings
    from stellar_jax.evolution import structure_to_fgong_jax
    from stellar_jax.oscillations import (
        build_oscillation_coeffs_jax, compute_eigenfreq_from_structure_jax,
        eigenfreq_from_coeffs
    )

    print("=" * 60)
    print("§5. Seismic gradient reference (WARNING: ~20-40 min E2E backward compile)")
    print("=" * 60)

    M = 1.0
    Z = 0.014
    alpha = 1.9
    N_steps = 50
    dt_fixed = 2e6
    nu_min, nu_max = 2500.0, 3500.0

    # Forward-only run to find the reference eigenfrequency
    print("  Forward run (max_steps=50, fixed_dt=2e6)...", end=" ", flush=True)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_ref = evolution.evolve_star(
            jnp.float64(M), Z=Z, max_steps=N_steps, fixed_dt=dt_fixed,
            alpha_mlt=alpha, diffusion=False, adaptive_mesh=False
        )
    print(f"done ({time.time() - t0:.0f}s)")

    # Find eigenfrequency
    print("  Finding eigenfrequency (l=0, Brent)...", end=" ", flush=True)
    t0 = time.time()
    glob_ref, var_ref = structure_to_fgong_jax(
        jnp.float64(M), r_ref['log_L_final'], r_ref['log_Teff_final'],
        r_ref['X_profile'], jnp.float64(Z), jnp.float64(0.0),
        jnp.float64(alpha), y_henyey=r_ref['y_henyey_final'],
        atm_ratio=r_ref.get('atm_ratio'),
    )
    info = compute_eigenfreq_from_structure_jax(
        glob_ref, var_ref, l=0, nu_min=nu_min, nu_max=nu_max,
        n_scan=100, n_steps=8000, mode_index=0
    )
    sigma2_ref = info['sigma2']
    print(f"done ({time.time() - t0:.0f}s), σ²={float(sigma2_ref):.10e}")

    # AD gradient ∂σ²/∂M
    print("  jax.grad(σ²_of_mass)(M=1.0)...", flush=True)
    t0 = time.time()

    def sigma2_of_mass(mass_val):
        r = evolution.evolve_star(
            mass_val, Z=Z, max_steps=N_steps, fixed_dt=dt_fixed,
            alpha_mlt=alpha, diffusion=False, adaptive_mesh=False
        )
        glob, var = structure_to_fgong_jax(
            mass_val, r['log_L_final'], r['log_Teff_final'],
            r['X_profile'], jnp.float64(Z), jnp.float64(0.0),
            jnp.float64(alpha), y_henyey=r['y_henyey_final'],
            atm_ratio=r.get('atm_ratio'),
        )
        grid_data = build_oscillation_coeffs_jax(glob, var)
        return eigenfreq_from_coeffs(
            grid_data['coeffs'], grid_data['x_grid'], sigma2_ref,
            info['l'], info['x_steps'], info['h_steps'], info['factor']
        )

    grad_sigma2_M = float(jax.grad(sigma2_of_mass)(jnp.float64(M)))
    elapsed = time.time() - t0
    print(f"  done ({elapsed:.0f}s), ∂σ²/∂M = {grad_sigma2_M:.15e}")

    return {'grad_dsigma2_dM': np.float64(grad_sigma2_M)}


def generate_mlt_nabla_ref():
    """Generate mlt_nabla value + grad reference on fixed solar-interior inputs.

    Inputs: physically valid mid-convection-zone conditions (T~1e6 K, rho~0.5,
    P~1e16 dyne/cm², kappa~1.0, nabla_rad=0.6 > nad=0.4 → convective).
    Pins: nabla (scalar), grad_nabla_dalpha (scalar ∂nabla/∂alpha_mlt).
    """
    from stellar_jax.transport.mlt import mlt_nabla
    from stellar_jax.config.constants import G, Msun, Rsun, k_B, m_H

    print("=" * 60)
    print("§6. mlt_nabla value + grad reference")
    print("=" * 60)

    # Fixed physically-valid solar CZ inputs (mid-convection zone, ~0.85 R_sun)
    nabla_rad = jnp.float64(0.6)   # superadiabatic → convective
    nad = jnp.float64(0.4)         # ideal-gas ∇_ad
    T = jnp.float64(1e6)           # ~1 MK (outer CZ)
    P = jnp.float64(1e16)          # ~10^16 dyne/cm²
    rho = jnp.float64(0.5)         # ~0.5 g/cm³
    kappa = jnp.float64(1.0)       # ~1 cm²/g (H⁻ opacity regime)
    g = jnp.float64(G * Msun / Rsun**2)  # surface-ish gravity
    mu = jnp.float64(0.62)         # mean molecular weight (ionized solar mix)
    alpha_mlt = jnp.float64(1.9)   # standard solar α

    print("  mlt_nabla(convective)...", end=" ", flush=True)
    t0 = time.time()
    nabla_val = float(mlt_nabla(nabla_rad, nad, T, P, rho, kappa, g, mu, alpha_mlt))
    print(f"done ({time.time() - t0:.1f}s), nabla={nabla_val:.15e}")

    # Gradient ∂nabla/∂alpha_mlt
    print("  jax.grad(mlt_nabla)(alpha)...", end=" ", flush=True)
    t0 = time.time()
    grad_fn = jax.grad(lambda a: mlt_nabla(nabla_rad, nad, T, P, rho, kappa, g, mu, a), argnums=0)
    grad_val = float(grad_fn(alpha_mlt))
    print(f"done ({time.time() - t0:.1f}s), ∂nabla/∂α={grad_val:.15e}")

    return {
        'mlt_nabla_value': np.float64(nabla_val),
        'mlt_nabla_grad_dalpha': np.float64(grad_val),
        # Also store the inputs for reproducibility
        'mlt_nabla_inputs': np.array([
            float(nabla_rad), float(nad), float(T), float(P), float(rho),
            float(kappa), float(g), float(mu), float(alpha_mlt)]),
    }


def generate_cell_residual_ref():
    """Generate _cell_residual reference on fixed solar-interior cell inputs.

    Inputs: two adjacent Henyey mesh points with physically valid ln(r), ln(P),
    ln(T), ell values for a solar-mass star. Pins the 4-vector [F1,F2,F3,F4].
    """
    from stellar_jax.solver.residual import _cell_residual
    from stellar_jax.config.constants import Msun, Rsun, Lsun

    print("=" * 60)
    print("§7. _cell_residual reference")
    print("=" * 60)

    # Fixed physically-valid Henyey cell inputs (solar interior, ~0.5 R_sun)
    # y_k = [ln_r, ln_P, ln_T, ell] at inner boundary
    # y_kp1 = [ln_r, ln_P, ln_T, ell] at outer boundary
    r_inner = 0.3 * Rsun
    r_outer = 0.35 * Rsun
    P_inner = 10**(16.5)  # dyne/cm²
    P_outer = 10**(16.3)
    T_inner = 10**(7.0)   # K
    T_outer = 10**(6.95)
    L_inner = 0.8 * Lsun
    L_outer = 0.85 * Lsun
    ell_inner = L_inner / Lsun  # = 0.8
    ell_outer = L_outer / Lsun  # = 0.85

    y_k = jnp.array([np.log(r_inner), np.log(P_inner), np.log(T_inner), ell_inner])
    y_kp1 = jnp.array([np.log(r_outer), np.log(P_outer), np.log(T_outer), ell_outer])
    dm = jnp.float64(0.05 * Msun)          # 5% of solar mass per cell
    m_mid = jnp.float64(0.325 * Msun)      # midpoint mass
    M_star = jnp.float64(1.0 * Msun)
    X_mid = jnp.float64(0.70)
    Z = jnp.float64(0.014)
    alpha_mlt = jnp.float64(1.9)

    print("  _cell_residual(solar interior cell)...", end=" ", flush=True)
    t0 = time.time()
    F = _cell_residual(y_k, y_kp1, dm, m_mid, M_star, X_mid, Z, alpha_mlt)
    F_np = np.array(F)
    print(f"done ({time.time() - t0:.1f}s)")
    print(f"    F = [{F_np[0]:.6e}, {F_np[1]:.6e}, {F_np[2]:.6e}, {F_np[3]:.6e}]")

    return {
        'cell_residual_F': F_np,
        'cell_residual_y_k': np.array(y_k),
        'cell_residual_y_kp1': np.array(y_kp1),
    }


def generate_eigenfreq_radial_vjp_ref():
    """Generate eigenfreq_radial primal + all cotangent reference via jax.vjp.

    Uses the existing FGONG pipeline to build realistic coefficients, then
    calls eigenfreq_radial through jax.vjp to pin the primal AND all three
    cotangents (coeffs, x_grid, sigma2_converged).
    """
    from stellar_jax.oscillations import (
        build_oscillation_coeffs_jax, compute_eigenfreq_from_structure_jax,
    )
    from stellar_jax.oscillations.adjoint import eigenfreq_radial
    import stellar_jax.oscillations as oscillations

    print("=" * 60)
    print("§8. eigenfreq_radial vjp primal + cotangent reference")
    print("=" * 60)

    # Load fixed FGONG
    fgong_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "mesa_comparison", "profiles", "1.0Msun", "midMS.FGONG.gz"
    )
    if not os.path.isfile(fgong_path):
        print(f"  WARNING: FGONG not found at {fgong_path}, skipping.")
        return {}

    with gzip.open(fgong_path, 'rt') as gz:
        content = gz.read()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.FGONG', delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        glob, var = oscillations.read_fgong(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Build coefficients and find eigenfrequency
    print("  Finding l=0 eigenfrequency from FGONG...", end=" ", flush=True)
    t0 = time.time()
    info = compute_eigenfreq_from_structure_jax(
        glob, var, l=0, nu_min=2500.0, nu_max=3500.0,
        n_scan=100, n_steps=8000, mode_index=0
    )
    print(f"done ({time.time() - t0:.1f}s), σ²={float(info['sigma2']):.10e}")

    # Build oscillation coeffs from the FGONG
    grid_data = build_oscillation_coeffs_jax(glob, var)
    coeffs = grid_data['coeffs']    # (5, N_grid) JAX array
    x_grid = grid_data['x_grid']    # (N_grid,) JAX array

    sigma2_converged = info['sigma2']
    x_steps = info['x_steps']
    h_steps = info['h_steps']
    factor = info['factor']

    # jax.vjp through eigenfreq_radial
    print("  jax.vjp(eigenfreq_radial)...", end=" ", flush=True)
    t0 = time.time()
    primals, vjp_fn = jax.vjp(
        lambda c, xg, s2: eigenfreq_radial(c, xg, s2, x_steps, h_steps, factor),
        coeffs, x_grid, sigma2_converged
    )
    # Pull back with cotangent g_sigma2 = 1.0
    g_coeffs, g_x_grid, g_sigma2_in = vjp_fn(jnp.float64(1.0))
    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)")
    print(f"    primal σ²  = {float(primals):.15e}")
    print(f"    |g_coeffs| = {float(jnp.linalg.norm(g_coeffs)):.6e}")
    print(f"    |g_x_grid| = {float(jnp.linalg.norm(g_x_grid)):.6e}")
    print(f"    g_sigma2_in = {float(g_sigma2_in):.6e}")

    return {
        'eigenfreq_radial_primal': np.float64(float(primals)),
        'eigenfreq_radial_g_coeffs': np.array(g_coeffs),
        'eigenfreq_radial_g_x_grid': np.array(g_x_grid),
        'eigenfreq_radial_g_sigma2_in': np.float64(float(g_sigma2_in)),
    }


def generate_henyey_newton_vjp_ref():
    """Generate _henyey_newton primal + all cotangent reference via jax.vjp.

    Constructs realistic ZAMS inputs (M=1.0 M_sun) using the shooting solver
    for an initial guess, then pins the converged state + cotangents.
    """
    from stellar_jax.solver.newton import _henyey_newton
    from stellar_jax.structure import initial_guess, newton_solve_xprofile, build_model_on_mesh
    from stellar_jax.config.constants import Msun, Lsun, Y_BBN, DY_DZ
    from stellar_jax.config.mesh_defaults import N_COMP, N_NEWTON_COLD

    print("=" * 60)
    print("§9. _henyey_newton vjp primal + cotangent reference")
    print("=" * 60)

    M_solar = 1.0
    Z = 0.014
    alpha_mlt = 1.9
    n_iter = 15  # N_NEWTON_COLD

    Y = Y_BBN + DY_DZ * Z
    X_init = 1.0 - Y - Z
    X_profile = jnp.full(N_COMP, X_init)

    # Use shooting solver for initial guess (same as henyey_solve_differentiable)
    # n_mesh=200: smaller than production (600) to keep npz size manageable,
    # but fully exercises the block-Thomas solve + IFT adjoint path.
    print("  Building initial guess via shooting...", end=" ", flush=True)
    t0 = time.time()
    logL_g, logTe_g = initial_guess(jnp.float64(M_solar))
    logL, logTe = newton_solve_xprofile(
        jnp.float64(M_solar), X_profile, jnp.float64(Z), jnp.float64(0.0),
        jnp.float64(logL_g), jnp.float64(logTe_g), jnp.float64(alpha_mlt),
        N_NEWTON_COLD)
    n_mesh = 200
    model = build_model_on_mesh(jnp.float64(M_solar), logL, logTe, X_profile,
                                jnp.float64(Z), jnp.float64(alpha_mlt), n_mesh)
    q_mesh = model['q']
    M_star = jnp.float64(M_solar * Msun)

    ln_r = jnp.log(jnp.maximum(model['r'], 1.0))
    ln_P = jnp.log(10.0) * model['logP']
    ln_T = jnp.log(10.0) * model['logT']
    ell = model['L'] / Lsun
    y_full = jnp.stack([ln_r, ln_P, ln_T, ell], axis=-1)
    N_s = q_mesh.shape[0] - 1
    y_init = y_full[:N_s]
    print(f"done ({time.time() - t0:.1f}s), y_init shape={y_init.shape}")

    # vjp through _henyey_newton
    print("  jax.vjp(_henyey_newton)...", flush=True)
    t0 = time.time()
    primals, vjp_fn = jax.vjp(
        lambda yi, qm, ms, xp, z, a: _henyey_newton(yi, qm, ms, xp, z, a, n_iter),
        y_init, q_mesh, M_star, X_profile, jnp.float64(Z), jnp.float64(alpha_mlt)
    )
    # Pull back with unit cotangent on y_final
    g_out = jnp.ones_like(primals)
    g_y_init, g_q_mesh, g_M_star, g_X_profile, g_Z, g_alpha = vjp_fn(g_out)
    elapsed = time.time() - t0
    print(f"  done ({elapsed:.0f}s)")
    print(f"    |y_final|    = {float(jnp.linalg.norm(primals)):.6e}")
    print(f"    |g_y_init|   = {float(jnp.linalg.norm(g_y_init)):.6e}")
    print(f"    |g_q_mesh|   = {float(jnp.linalg.norm(g_q_mesh)):.6e}")
    print(f"    g_M_star     = {float(g_M_star):.6e}")
    print(f"    |g_X_profile|= {float(jnp.linalg.norm(g_X_profile)):.6e}")
    print(f"    g_Z          = {float(g_Z):.6e}")
    print(f"    g_alpha      = {float(g_alpha):.6e}")

    return {
        'henyey_newton_y_final': np.array(primals),
        'henyey_newton_g_y_init': np.array(g_y_init),
        'henyey_newton_g_q_mesh': np.array(g_q_mesh),
        'henyey_newton_g_M_star': np.float64(float(g_M_star)),
        'henyey_newton_g_X_profile': np.array(g_X_profile),
        'henyey_newton_g_Z': np.float64(float(g_Z)),
        'henyey_newton_g_alpha': np.float64(float(g_alpha)),
    }


def generate_evolve_one_step_ref():
    """Generate evolve_star(max_steps=1) forward + gradient reference.

    A single evolution step (no adaptive dt) — fast to compile and pins
    the scan body + ZAMS initialization.
    """
    import stellar_jax.evolution as evolution

    print("=" * 60)
    print("§10. evolve_star(max_steps=1) forward + gradient reference")
    print("=" * 60)

    # Forward
    print("  evolve_star(M=1.0, max_steps=1, fixed_dt=2e6)...", flush=True)
    t0 = time.time()
    r = evolution.evolve_star(
        1.0, Z=0.014, max_steps=1, fixed_dt=2e6,
        diffusion=True, adaptive_mesh=False
    )
    jax.block_until_ready(r['log_L'])
    elapsed = time.time() - t0
    print(f"  done ({elapsed:.0f}s)")
    print(f"    log_L[0]    = {float(r['log_L'][0]):.15e}")
    print(f"    log_Teff[0] = {float(r['log_Teff'][0]):.15e}")

    # Gradient ∂logL[0]/∂M (one-step)
    print("  jax.grad(logL[0])(M=1.0, max_steps=1)...", flush=True)
    t0 = time.time()

    def logL_one_step(mass):
        result = evolution.evolve_star(
            mass, Z=0.014, max_steps=1, fixed_dt=2e6,
            diffusion=True, adaptive_mesh=False
        )
        return result['log_L'][0]

    grad_val = float(jax.grad(logL_one_step)(jnp.float64(1.0)))
    elapsed = time.time() - t0
    print(f"  done ({elapsed:.0f}s), ∂logL[0]/∂M = {grad_val:.15e}")

    return {
        'evolve_1step_logL': np.float64(float(r['log_L'][0])),
        'evolve_1step_logTe': np.float64(float(r['log_Teff'][0])),
        'evolve_1step_grad_dlogL_dM': np.float64(grad_val),
    }


def generate_evolve_solar_ref():
    """Generate evolve_solar forward + gradient reference.

    Uses max_steps=5 (small, fast) with fixed composition. Pins forward
    log_L at the final accepted step and ∂logL[-1]/∂alpha_mlt.
    """
    from stellar_jax.calibration.solar import evolve_solar

    print("=" * 60)
    print("§11. evolve_solar forward + gradient reference")
    print("=" * 60)

    alpha_mlt = 1.9
    Y_init = 0.28
    Z = 0.014

    # Forward
    print("  evolve_solar(alpha=1.9, Y=0.28, Z=0.014, max_steps=5)...", flush=True)
    t0 = time.time()
    ages, log_L, log_R, log_Te, X_final, Y_final, Z_final = evolve_solar(
        jnp.float64(alpha_mlt), jnp.float64(Y_init), jnp.float64(Z),
        max_steps=5
    )
    jax.block_until_ready(log_L)
    elapsed = time.time() - t0
    print(f"  done ({elapsed:.0f}s)")
    # Take last step's logL (which may be 0 if rejected — take max valid)
    logL_final = float(log_L[-1])
    print(f"    log_L[-1]  = {logL_final:.15e}")
    print(f"    log_Te[-1] = {float(log_Te[-1]):.15e}")

    # Gradient ∂logL[-1]/∂alpha_mlt
    print("  jax.grad(logL[-1] from evolve_solar)(alpha=1.9)...", flush=True)
    t0 = time.time()

    def solar_logL_final(alpha):
        _ages, _logL, _logR, _logTe, _X, _Y, _Z = evolve_solar(
            alpha, jnp.float64(Y_init), jnp.float64(Z), max_steps=5
        )
        return _logL[-1]

    grad_val = float(jax.grad(solar_logL_final)(jnp.float64(alpha_mlt)))
    elapsed = time.time() - t0
    print(f"  done ({elapsed:.0f}s), ∂logL[-1]/∂α = {grad_val:.15e}")

    return {
        'evolve_solar_logL_final': np.float64(logL_final),
        'evolve_solar_logTe_final': np.float64(float(log_Te[-1])),
        'evolve_solar_ages': np.array(ages),
        'evolve_solar_grad_dlogL_dalpha': np.float64(grad_val),
    }


def _get_git_sha():
    """Return the HEAD git SHA, or 'unknown' if not in a git repo."""
    import subprocess
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fast-only', action='store_true',
                        help='Only generate fast refs (composition + oscillation + mlt + residual)')
    args = parser.parse_args()

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "golden_master_ref.npz"
    )

    print(f"Generating golden-master references...")
    print(f"Output: {out_path}")
    print(f"Platform: JAX {jax.__version__}, x64={jax.config.jax_enable_x64}")
    print()

    all_refs = {}

    # §1: Composition (fast)
    all_refs.update(generate_composition_refs())

    # §2: Oscillation (fast)
    all_refs.update(generate_oscillation_ref())

    # §6: mlt_nabla value + grad (fast)
    all_refs.update(generate_mlt_nabla_ref())

    # §7: _cell_residual (fast)
    all_refs.update(generate_cell_residual_ref())

    if not args.fast_only:
        # §3: Forward evolution (heavy)
        all_refs.update(generate_evolution_ref())

        # §4: Gradients (heavy)
        all_refs.update(generate_gradient_refs())

        # §5: Seismic gradient E2E (heaviest)
        all_refs.update(generate_seismic_gradient_ref())

        # §8: eigenfreq_radial vjp (medium — one compile)
        all_refs.update(generate_eigenfreq_radial_vjp_ref())

        # §9: _henyey_newton vjp (heavy — Newton + IFT backward)
        all_refs.update(generate_henyey_newton_vjp_ref())

        # §10: evolve_star one-step forward + grad (heavy)
        all_refs.update(generate_evolve_one_step_ref())

        # §11: evolve_solar forward + grad (heavy)
        all_refs.update(generate_evolve_solar_ref())

    # Metadata: jax version + git SHA
    git_sha = _get_git_sha()
    all_refs['_metadata_jax_version'] = np.array(jax.__version__, dtype=object)
    all_refs['_metadata_git_sha'] = np.array(git_sha, dtype=object)

    # In --fast-only mode, merge new keys into the existing npz (preserve heavy keys)
    if args.fast_only and os.path.isfile(out_path):
        print()
        print("--fast-only: merging new keys into existing npz...")
        existing = dict(np.load(out_path, allow_pickle=True))
        existing.update(all_refs)
        all_refs = existing

    # Save
    np.savez(out_path, **all_refs)
    print()
    print(f"Saved {len(all_refs)} reference arrays to {out_path}")
    print(f"Keys: {sorted(all_refs.keys())}")
    print(f"Metadata: jax={jax.__version__}, git={git_sha[:12]}")

    # Verify roundtrip
    loaded = np.load(out_path, allow_pickle=True)
    for key in all_refs:
        assert key in loaded, f"Key {key} missing after save/load"
    print("Roundtrip verification: OK")


if __name__ == '__main__':
    main()
