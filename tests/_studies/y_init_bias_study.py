#!/usr/bin/env python3
"""Research study for #1119: ∂σ²/∂Y_init magnitude bias.

Measures:
  Experiment 1: Baseline AD-vs-FD bias (current code)
  Experiment 2: Size of the missing ∂(atm)/∂X_surf term
  Experiment 3: AD-vs-FD bias after adding X_surf to atm_param_grads
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', '/cache')

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import warnings

print("=" * 70, flush=True)
print("EXPERIMENT 1: Baseline ∂σ²/∂Y_init AD-vs-FD", flush=True)
print("=" * 70, flush=True)

from stellar_jax.evolution import evolve_star, structure_to_fgong_jax
from stellar_jax.oscillations import (
    build_oscillation_coeffs_jax, compute_eigenfreq_from_structure_jax,
    eigenfreq_from_coeffs, radial_determinant
)
from scipy.optimize import brentq

# Parameters — match the existing test
M = 1.0; Z = 0.014; Y_init = 0.27; alpha = 1.9
N_steps = 3; dt_fixed = 1e7
dY = Y_init * 1e-3
nu_min, nu_max = 2600.0, 2900.0

print(f"Parameters: M={M}, Z={Z}, Y_init={Y_init}, alpha={alpha}", flush=True)
print(f"N_steps={N_steps}, dt_fixed={dt_fixed}", flush=True)
print(f"FD step: dY={dY}", flush=True)
print(flush=True)

# ── Reference structure + eigenfrequency ──
print("Computing reference at Y_init=0.27...", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r_ref = evolve_star(jnp.float64(M), Z=Z, Y_init=Y_init, max_steps=N_steps,
                        alpha_mlt=alpha, diffusion=False, fixed_dt=dt_fixed,
                        adaptive_mesh=False)
print("  evolve_star done.", flush=True)
glob_ref, var_ref = structure_to_fgong_jax(
    jnp.float64(M), r_ref['log_L_final'], r_ref['log_Teff_final'],
    r_ref['X_profile'], jnp.float64(Z), jnp.float64(0.0),
    jnp.float64(alpha), y_henyey=r_ref['y_henyey_final'],
    atm_ratio=r_ref.get('atm_ratio'))
print("  FGONG done.", flush=True)
info = compute_eigenfreq_from_structure_jax(
    glob_ref, var_ref, l=0, nu_min=nu_min, nu_max=nu_max,
    n_scan=100, n_steps=8000, mode_index=0)
sigma2_ref = info['sigma2']
nu_ref = info['nu']
print(f"  ν = {float(nu_ref):.4f} μHz, σ² = {float(sigma2_ref):.6f}", flush=True)

# ── AD gradient ∂σ²/∂Y_init ──
print("\nComputing AD gradient ∂σ²/∂Y_init...", flush=True)

def sigma2_of_Y(y_val):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = evolve_star(jnp.float64(M), Z=Z, Y_init=y_val, max_steps=N_steps,
                        alpha_mlt=alpha, diffusion=False, fixed_dt=dt_fixed,
                        adaptive_mesh=False)
    glob, var = structure_to_fgong_jax(
        jnp.float64(M), r['log_L_final'], r['log_Teff_final'],
        r['X_profile'], jnp.float64(Z), jnp.float64(0.0),
        jnp.float64(alpha), y_henyey=r['y_henyey_final'],
        atm_ratio=r.get('atm_ratio'))
    grid_data = build_oscillation_coeffs_jax(glob, var)
    return eigenfreq_from_coeffs(
        grid_data['coeffs'], grid_data['x_grid'], sigma2_ref,
        info['l'], info['x_steps'], info['h_steps'], info['factor'])

print("  JIT compiling AD gradient (this takes a while)...", flush=True)
ad_grad = float(jax.grad(sigma2_of_Y)(jnp.float64(Y_init)))
print(f"  AD: ∂σ²/∂Y_init = {ad_grad:.6e}", flush=True)

# ── Independent FD gradient ──
print("\nComputing independent FD gradient...", flush=True)

def sigma2_fd(y_float):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = evolve_star(jnp.float64(M), Z=Z, Y_init=y_float, max_steps=N_steps,
                        alpha_mlt=alpha, diffusion=False, fixed_dt=dt_fixed,
                        adaptive_mesh=False)
    glob_fd, var_fd = structure_to_fgong_jax(
        jnp.float64(M), r['log_L_final'], r['log_Teff_final'],
        r['X_profile'], jnp.float64(Z), jnp.float64(0.0),
        jnp.float64(alpha), y_henyey=r['y_henyey_final'],
        atm_ratio=r.get('atm_ratio'))
    grid_data = build_oscillation_coeffs_jax(glob_fd, var_fd)
    coeffs_fd = grid_data['coeffs']
    x_grid_fd = grid_data['x_grid']

    @jax.jit
    def det_fn(s2):
        return radial_determinant(s2, info['x_steps'], info['h_steps'],
                                  x_grid_fd, coeffs_fd)

    s2_lo = float(sigma2_ref) * 0.9
    s2_hi = float(sigma2_ref) * 1.1
    try:
        s2_root = brentq(lambda s: float(det_fn(jnp.float64(s))),
                         s2_lo, s2_hi, rtol=1e-10)
    except ValueError:
        s2_lo = float(sigma2_ref) * 0.5
        s2_hi = float(sigma2_ref) * 1.5
        s2_root = brentq(lambda s: float(det_fn(jnp.float64(s))),
                         s2_lo, s2_hi, rtol=1e-10)
    return s2_root

print("  Computing σ²(Y+dY)...", flush=True)
s2_plus = sigma2_fd(Y_init + dY)
print(f"    σ²(Y+dY) = {s2_plus:.8e}", flush=True)
print("  Computing σ²(Y-dY)...", flush=True)
s2_minus = sigma2_fd(Y_init - dY)
print(f"    σ²(Y-dY) = {s2_minus:.8e}", flush=True)
fd_grad = (s2_plus - s2_minus) / (2 * dY)
print(f"  FD: ∂σ²/∂Y_init = {fd_grad:.6e}", flush=True)

rel_err = abs(ad_grad - fd_grad) / max(abs(fd_grad), abs(ad_grad))
print(f"\n  BASELINE: AD={ad_grad:.6e}, FD={fd_grad:.6e}", flush=True)
print(f"  rel_err = {rel_err*100:.2f}%", flush=True)
print(f"  AD/FD ratio = {ad_grad/fd_grad:.4f}", flush=True)

# ── Experiment 2: Measure the atmosphere X_surf sensitivity ──
print("\n" + "=" * 70, flush=True)
print("EXPERIMENT 2: Measure ∂(ln_P_atm, ln_T_atm)/∂X_surf", flush=True)
print("=" * 70, flush=True)

from stellar_jax.solver.surface_bc import _surface_bc_atm_values

# Get the final Henyey state
y_henyey = r_ref['y_henyey_final']
N_s = y_henyey.shape[0]
y_surf_ref_val = y_henyey[N_s - 1]
M_star_cgs = float(M) * 1.989e33
atm_ratio_val = float(r_ref.get('atm_ratio'))
R_phot = atm_ratio_val * jnp.exp(y_surf_ref_val[0])
X_surf = float(1.0 - Y_init - Z)

print(f"  y_surf_ref = {y_surf_ref_val}", flush=True)
print(f"  atm_ratio = {atm_ratio_val}", flush=True)
print(f"  X_surf = {X_surf}", flush=True)

# Compute ∂(ln_P_atm, ln_T_atm)/∂X_surf via jacfwd
def atm_of_X(X_val):
    R_p = atm_ratio_val * jnp.exp(y_surf_ref_val[0])
    lnP, lnT = _surface_bc_atm_values(y_surf_ref_val, jnp.float64(M_star_cgs),
                                       X_val, jnp.float64(Z), jnp.float64(alpha), R_p)
    return jnp.array([lnP, lnT])

print("  Computing jacfwd of atm w.r.t. X_surf...", flush=True)
atm_jac_X = jax.jacfwd(atm_of_X)(jnp.float64(X_surf))
dlnP_dX = float(atm_jac_X[0])
dlnT_dX = float(atm_jac_X[1])
print(f"  ∂ln_P_atm/∂X_surf = {dlnP_dX:.6e}", flush=True)
print(f"  ∂ln_T_atm/∂X_surf = {dlnT_dX:.6e}", flush=True)

# For comparison, compute the existing Z partials
def atm_of_Z(Z_val):
    R_p = atm_ratio_val * jnp.exp(y_surf_ref_val[0])
    lnP, lnT = _surface_bc_atm_values(y_surf_ref_val, jnp.float64(M_star_cgs),
                                       jnp.float64(X_surf), Z_val, jnp.float64(alpha), R_p)
    return jnp.array([lnP, lnT])

print("  Computing jacfwd of atm w.r.t. Z (comparison)...", flush=True)
atm_jac_Z = jax.jacfwd(atm_of_Z)(jnp.float64(Z))
dlnP_dZ = float(atm_jac_Z[0])
dlnT_dZ = float(atm_jac_Z[1])
print(f"  ∂ln_P_atm/∂Z = {dlnP_dZ:.6e}", flush=True)
print(f"  ∂ln_T_atm/∂Z = {dlnT_dZ:.6e}", flush=True)

# Scale: since ∂X/∂Y = -1, the missing atm term for ∂σ²/∂Y is:
# Δg_Y ≈ -(lam[-1,2]*dlnP_dX + lam[-1,3]*dlnT_dX)
# We can estimate the order of magnitude
print(f"\n  Since ∂X/∂Y = -1, the missing atmosphere contribution is:", flush=True)
print(f"  lam[-1,2]*{dlnP_dX:.4e} + lam[-1,3]*{dlnT_dX:.4e}", flush=True)
print(f"  (Need the adjoint vector lam to quantify the correction)", flush=True)

# ── Experiment 3: FD verification of the atmosphere X sensitivity ──
print("\n" + "=" * 70, flush=True)
print("EXPERIMENT 3: FD verification of atmosphere X_surf sensitivity", flush=True)
print("=" * 70, flush=True)

dX_fd = 1e-5
lnP_ref, lnT_ref = atm_of_X(jnp.float64(X_surf))
lnP_plus, lnT_plus = atm_of_X(jnp.float64(X_surf + dX_fd))
lnP_minus, lnT_minus = atm_of_X(jnp.float64(X_surf - dX_fd))

dlnP_dX_fd = float(lnP_plus - lnP_minus) / (2 * dX_fd)
dlnT_dX_fd = float(lnT_plus - lnT_minus) / (2 * dX_fd)

print(f"  AD:  ∂ln_P_atm/∂X = {dlnP_dX:.6e}, ∂ln_T_atm/∂X = {dlnT_dX:.6e}", flush=True)
print(f"  FD:  ∂ln_P_atm/∂X = {dlnP_dX_fd:.6e}, ∂ln_T_atm/∂X = {dlnT_dX_fd:.6e}", flush=True)
print(f"  Match: lnP err={abs(dlnP_dX-dlnP_dX_fd)/max(abs(dlnP_dX_fd),1e-30):.4e}, "
      f"lnT err={abs(dlnT_dX-dlnT_dX_fd)/max(abs(dlnT_dX_fd),1e-30):.4e}", flush=True)

# Compare magnitude to the Z partial that added
print(f"\n  Comparison to Z partial (already corrected by #1211):", flush=True)
print(f"  |∂ln_P_atm/∂X| / |∂ln_P_atm/∂Z| = {abs(dlnP_dX)/max(abs(dlnP_dZ),1e-30):.4f}", flush=True)
print(f"  |∂ln_T_atm/∂X| / |∂ln_T_atm/∂Z| = {abs(dlnT_dX)/max(abs(dlnT_dZ),1e-30):.4f}", flush=True)
print(f"  The X partial is comparable to the Z partial — it should matter for the gradient.", flush=True)

print("\n" + "=" * 70, flush=True)
print("DONE — results above. Use these to decide if the X_surf atmosphere", flush=True)
print("correction is the dominant missing term in ∂σ²/∂Y_init.", flush=True)
print("=" * 70, flush=True)
