#!/usr/bin/env python3
"""Research study: AD ∂σ²/∂α at the evolved subgiant (X_c≈0.15–0.22).

Issue #1206: Validate AD ∂σ²/∂α via recorded replay + live-factor.

This script:
  1. Evolves to X_c≈0.22 via evolve_star(max_steps=500, freeze_schedule=True)
  2. Extracts the dt_schedule for replay
  3. Measures AD ∂σ²/∂α (frozen-factor, via eigenfreq_from_coeffs)
  4. Measures AD ∂ν²/∂α (live-factor, via nu_from_sigma2 with traced factor)
  5. Computes mode-locked centered FD as oracle for both σ² and ν²
  6. Compares sign + magnitude, measures rel_err

All results streamed to stdout for liveness.
"""
import os
import sys
import time
import warnings

# Environment
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', '/cache')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from stellar_jax.evolution import evolve_star, structure_to_fgong_jax
from stellar_jax.oscillations import (
    build_oscillation_coeffs_jax, compute_eigenfreq_from_structure_jax,
    eigenfreq_from_coeffs, radial_determinant,
    compute_factor, nu_from_sigma2,
)
from stellar_jax.config.constants import SECONDS_PER_YEAR
from scipy.optimize import brentq


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════
M = 1.0           # Solar mass
Z = 0.014         # MODE-A metallicity
ALPHA = 1.9       # MLT parameter
D_ALPHA = ALPHA * 1e-4   # FD step (centered)
N_STEPS = 500     # lax.scan steps — reaches X_c≈0.22 (late MS / early SGB)
NU_MIN = 1500.0   # Wider bracket for evolved model (lower ν than ZAMS)
NU_MAX = 4500.0


def banner(msg):
    print(f"\n{'='*72}\n  {msg}\n{'='*72}", flush=True)


def progress(msg):
    print(f"  {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Reference evolution (freeze_schedule → extract dt_schedule)
# ═══════════════════════════════════════════════════════════════════
banner("Phase 1: Reference evolution (max_steps=500, freeze_schedule=True)")
t0 = time.time()

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    r_ref = evolve_star(
        jnp.float64(M), Z=Z, max_steps=N_STEPS, alpha_mlt=ALPHA,
        diffusion=False, freeze_schedule=True, adaptive_mesh=False)

elapsed = time.time() - t0
progress(f"Done in {elapsed:.1f}s")

# Extract schedule
ages_yr = np.array(r_ref['star_age'])
ages_sec = ages_yr * SECONDS_PER_YEAR
dt_sched = np.zeros(N_STEPS, dtype=np.float64)
dt_sched[0] = ages_sec[0]
dt_sched[1:] = np.diff(ages_sec)

xc_final = float(r_ref['center_h1'][-1])
age_final = ages_yr[-1]
progress(f"X_c = {xc_final:.4f}, age = {age_final:.4e} yr")
progress(f"logL = {float(r_ref['log_L_final']):.4f}, logTe = {float(r_ref['log_Teff_final']):.4f}")
progress(f"Schedule: {N_STEPS} steps, dt range [{np.min(dt_sched[dt_sched>0])/SECONDS_PER_YEAR:.3e}, {np.max(dt_sched)/SECONDS_PER_YEAR:.3e}] yr")

# Adaptive guard
accepted_dts = np.diff(ages_yr)
accepted_dts = accepted_dts[accepted_dts > 0]
if len(accepted_dts) > 1:
    dt_ratio = float(np.max(accepted_dts) / (np.min(accepted_dts) + 1e-30))
else:
    dt_ratio = 1.0
progress(f"Adaptive guard: dt_ratio = {dt_ratio:.1f}")


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Build FGONG + find reference eigenfrequency
# ═══════════════════════════════════════════════════════════════════
banner("Phase 2: Build FGONG and find eigenfrequency")
t0 = time.time()

glob_ref, var_ref = structure_to_fgong_jax(
    jnp.float64(M), r_ref['log_L_final'], r_ref['log_Teff_final'],
    r_ref['X_profile'], jnp.float64(Z), jnp.float64(0.0),
    jnp.float64(ALPHA), y_henyey=r_ref['y_henyey_final'],
    atm_ratio=r_ref.get('atm_ratio'))
R_star = float(glob_ref[1])
progress(f"R_star = {R_star:.3e} cm")

info_ref = compute_eigenfreq_from_structure_jax(
    glob_ref, var_ref, l=0, nu_min=NU_MIN, nu_max=NU_MAX,
    n_scan=200, n_steps=8000, mode_index=0)
sigma2_ref = info_ref['sigma2']
nu_ref = info_ref['nu']
factor_ref = info_ref['factor']
progress(f"ν = {nu_ref:.4f} μHz, σ² = {float(sigma2_ref):.6f}, factor = {float(factor_ref):.6e}")
elapsed = time.time() - t0
progress(f"Done in {elapsed:.1f}s")


# ═══════════════════════════════════════════════════════════════════
# Phase 3: AD gradient ∂σ²/∂α (frozen-factor — eigenfreq_from_coeffs)
# ═══════════════════════════════════════════════════════════════════
banner("Phase 3: AD ∂σ²/∂α (frozen-factor)")
t0 = time.time()

def sigma2_of_alpha(alpha_val):
    """Differentiable: α → evolve → FGONG → σ² (frozen factor in nondiff_argnums)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = evolve_star(jnp.float64(M), Z=Z, max_steps=N_STEPS, alpha_mlt=alpha_val,
                        diffusion=False, freeze_schedule=True,
                        dt_schedule=dt_sched, adaptive_mesh=False)
    glob, var = structure_to_fgong_jax(
        jnp.float64(M), r['log_L_final'], r['log_Teff_final'],
        r['X_profile'], jnp.float64(Z), jnp.float64(0.0),
        alpha_val, y_henyey=r['y_henyey_final'],
        atm_ratio=r.get('atm_ratio'))
    grid_data = build_oscillation_coeffs_jax(glob, var)
    return eigenfreq_from_coeffs(
        grid_data['coeffs'], grid_data['x_grid'], sigma2_ref,
        info_ref['l'], info_ref['x_steps'], info_ref['h_steps'], info_ref['factor'])

progress("Computing jax.grad(sigma2_of_alpha)...")
ad_dsigma2_dalpha = float(jax.grad(sigma2_of_alpha)(jnp.float64(ALPHA)))
elapsed = time.time() - t0
progress(f"AD ∂σ²/∂α (frozen-factor) = {ad_dsigma2_dalpha:.6f} (in {elapsed:.1f}s)")


# ═══════════════════════════════════════════════════════════════════
# Phase 4: AD gradient ∂ν²/∂α (live-factor — captures ∂factor/∂α)
# ═══════════════════════════════════════════════════════════════════
banner("Phase 4: AD ∂ν²/∂α (live-factor)")
t0 = time.time()

def nu2_of_alpha(alpha_val):
    """Differentiable: α → evolve → FGONG → σ² → ν² with LIVE factor."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = evolve_star(jnp.float64(M), Z=Z, max_steps=N_STEPS, alpha_mlt=alpha_val,
                        diffusion=False, freeze_schedule=True,
                        dt_schedule=dt_sched, adaptive_mesh=False)
    glob, var = structure_to_fgong_jax(
        jnp.float64(M), r['log_L_final'], r['log_Teff_final'],
        r['X_profile'], jnp.float64(Z), jnp.float64(0.0),
        alpha_val, y_henyey=r['y_henyey_final'],
        atm_ratio=r.get('atm_ratio'))
    grid_data = build_oscillation_coeffs_jax(glob, var)
    # σ² from IFT with frozen factor (in nondiff_argnums)
    sigma2 = eigenfreq_from_coeffs(
        grid_data['coeffs'], grid_data['x_grid'], sigma2_ref,
        info_ref['l'], info_ref['x_steps'], info_ref['h_steps'], info_ref['factor'])
    # Convert to ν² with LIVE factor (captures ∂factor/∂α via R(α))
    live_factor = grid_data['factor']
    nu = nu_from_sigma2(sigma2, live_factor)
    return nu ** 2  # ν²

progress("Computing jax.grad(nu2_of_alpha)...")
ad_dnu2_dalpha = float(jax.grad(nu2_of_alpha)(jnp.float64(ALPHA)))
elapsed = time.time() - t0
progress(f"AD ∂ν²/∂α (live-factor) = {ad_dnu2_dalpha:.6e} (in {elapsed:.1f}s)")


# ═══════════════════════════════════════════════════════════════════
# Phase 5: Mode-locked centered FD (oracle for both σ² and ν²)
# ═══════════════════════════════════════════════════════════════════
banner("Phase 5: Mode-locked centered FD")

def compute_sigma2_and_nu2_fd(alpha_float):
    """Forward-only: α → evolve → FGONG → mode-locked σ² and ν²."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = evolve_star(jnp.float64(M), Z=Z, max_steps=N_STEPS,
                        alpha_mlt=alpha_float, diffusion=False,
                        freeze_schedule=True,
                        dt_schedule=dt_sched, adaptive_mesh=False)
    glob_fd, var_fd = structure_to_fgong_jax(
        jnp.float64(M), r['log_L_final'], r['log_Teff_final'],
        r['X_profile'], jnp.float64(Z), jnp.float64(0.0),
        jnp.float64(alpha_float), y_henyey=r['y_henyey_final'],
        atm_ratio=r.get('atm_ratio'))
    grid_data = build_oscillation_coeffs_jax(glob_fd, var_fd)
    coeffs_fd = grid_data['coeffs']
    x_grid_fd = grid_data['x_grid']
    factor_fd = float(grid_data['factor'])

    # Mode-locked root-finding: brentq in a bracket around the reference σ²
    @jax.jit
    def det_fn(s2):
        return radial_determinant(s2, info_ref['x_steps'], info_ref['h_steps'],
                                  x_grid_fd, coeffs_fd)

    s2_lo = float(sigma2_ref) * 0.85
    s2_hi = float(sigma2_ref) * 1.15
    try:
        s2_root = brentq(lambda s: float(det_fn(jnp.float64(s))),
                         s2_lo, s2_hi, rtol=1e-10)
    except ValueError:
        s2_lo = float(sigma2_ref) * 0.5
        s2_hi = float(sigma2_ref) * 1.5
        s2_root = brentq(lambda s: float(det_fn(jnp.float64(s))),
                         s2_lo, s2_hi, rtol=1e-10)

    nu2 = s2_root / factor_fd  # ν² = σ²/factor
    return s2_root, nu2, factor_fd

# α + δα
progress(f"Computing FD at α+δα = {ALPHA + D_ALPHA:.6f}...")
t0 = time.time()
s2_plus, nu2_plus, factor_plus = compute_sigma2_and_nu2_fd(ALPHA + D_ALPHA)
progress(f"  σ²+ = {s2_plus:.6f}, ν²+ = {nu2_plus:.6e}, factor+ = {factor_plus:.6e} ({time.time()-t0:.1f}s)")

# α - δα
progress(f"Computing FD at α-δα = {ALPHA - D_ALPHA:.6f}...")
t0 = time.time()
s2_minus, nu2_minus, factor_minus = compute_sigma2_and_nu2_fd(ALPHA - D_ALPHA)
progress(f"  σ²- = {s2_minus:.6f}, ν²- = {nu2_minus:.6e}, factor- = {factor_minus:.6e} ({time.time()-t0:.1f}s)")

# Centered FD
fd_dsigma2_dalpha = (s2_plus - s2_minus) / (2 * D_ALPHA)
fd_dnu2_dalpha = (nu2_plus - nu2_minus) / (2 * D_ALPHA)

progress(f"FD ∂σ²/∂α = {fd_dsigma2_dalpha:.6f}")
progress(f"FD ∂ν²/∂α = {fd_dnu2_dalpha:.6e}")
progress(f"FD ∂factor/∂α = {(factor_plus - factor_minus) / (2 * D_ALPHA):.6e}")


# ═══════════════════════════════════════════════════════════════════
# Phase 6: Results summary
# ═══════════════════════════════════════════════════════════════════
banner("RESULTS SUMMARY")

print(f"\n  Operating point:", flush=True)
print(f"    M = {M} M☉, Z = {Z}, α = {ALPHA}", flush=True)
print(f"    X_c = {xc_final:.4f}, age = {age_final:.4e} yr", flush=True)
print(f"    ν_ref = {nu_ref:.4f} μHz, σ²_ref = {float(sigma2_ref):.6f}", flush=True)
print(f"    factor = {float(factor_ref):.6e}", flush=True)

# ── σ² gradient (frozen-factor) ──
print(f"\n  ∂σ²/∂α (frozen-factor, IFT only):", flush=True)
print(f"    AD  = {ad_dsigma2_dalpha:.6f}", flush=True)
print(f"    FD  = {fd_dsigma2_dalpha:.6f}", flush=True)

if abs(fd_dsigma2_dalpha) > 1e-10:
    sigma2_sign_ok = ad_dsigma2_dalpha * fd_dsigma2_dalpha > 0
    sigma2_rel_err = abs(ad_dsigma2_dalpha - fd_dsigma2_dalpha) / abs(fd_dsigma2_dalpha)
    print(f"    Sign: {'CORRECT ✓' if sigma2_sign_ok else 'MISMATCH ✗'}", flush=True)
    print(f"    rel_err = {sigma2_rel_err*100:.2f}%", flush=True)
else:
    print(f"    FD is zero — cannot compute rel_err", flush=True)
    sigma2_sign_ok = False
    sigma2_rel_err = float('inf')

# ── ν² gradient (live-factor) ──
print(f"\n  ∂ν²/∂α (live-factor, captures ∂factor/∂α):", flush=True)
print(f"    AD  = {ad_dnu2_dalpha:.6e}", flush=True)
print(f"    FD  = {fd_dnu2_dalpha:.6e}", flush=True)

if abs(fd_dnu2_dalpha) > 1e-10:
    nu2_sign_ok = ad_dnu2_dalpha * fd_dnu2_dalpha > 0
    nu2_rel_err = abs(ad_dnu2_dalpha - fd_dnu2_dalpha) / abs(fd_dnu2_dalpha)
    print(f"    Sign: {'CORRECT ✓' if nu2_sign_ok else 'MISMATCH ✗'}", flush=True)
    print(f"    rel_err = {nu2_rel_err*100:.2f}%", flush=True)
else:
    print(f"    FD is zero — cannot compute rel_err", flush=True)
    nu2_sign_ok = False
    nu2_rel_err = float('inf')

# ── Assessment ──
print(f"\n  Assessment:", flush=True)

# AC1: sign-correct
ac1_pass = sigma2_sign_ok  # σ² sign-correct vs mode-locked FD
print(f"    AC1 (sign-correct σ²): {'PASS ✓' if ac1_pass else 'FAIL ✗'}", flush=True)

# AC2: rel_err < 25% with live-factor
ac2_pass = nu2_sign_ok and nu2_rel_err < 0.25
print(f"    AC2 (live-factor ∂ν²/∂α rel_err < 25%): {'PASS ✓' if ac2_pass else 'FAIL ✗'} ({nu2_rel_err*100:.2f}%)", flush=True)

# Frozen-factor σ² rel_err for comparison
print(f"    σ² frozen-factor rel_err: {sigma2_rel_err*100:.2f}%", flush=True)
print(f"    ν² live-factor rel_err: {nu2_rel_err*100:.2f}%", flush=True)

# Factor sensitivity analysis
df_dalpha = (factor_plus - factor_minus) / (2 * D_ALPHA)
print(f"\n  Factor sensitivity:", flush=True)
print(f"    ∂factor/∂α = {df_dalpha:.6e}", flush=True)
print(f"    σ²·∂(1/factor)/∂α = {-float(sigma2_ref)/float(factor_ref)**2 * df_dalpha:.6e}", flush=True)
print(f"    (fractional): ∂ln(factor)/∂α = {df_dalpha/float(factor_ref):.6f}", flush=True)

print(f"\n{'='*72}", flush=True)
print(f"  Script completed.", flush=True)
print(f"{'='*72}", flush=True)
