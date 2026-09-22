"""Independent reference structure kernels via finite differences.

Computes the Γ₁ kernel K_{Γ₁,ρ}(r) by direct numerical differentiation:
perturb Γ₁ at each FGONG grid point, recompute the eigenfrequency from
scratch, and take the finite-difference derivative. This is the most
independent possible reference — it shares only the eigenvalue solver with
the AD kernel, and uses a DIFFERENT differentiation method (FD vs AD/IFT).

The AD kernel (kernels.py) differentiates through the IFT adjoint:
    Γ₁ → coeffs → eigenfreq (IFT @custom_vjp) → ν → jax.grad → K

This module perturbs each Γ₁(r_i) independently and recomputes ν:
    K(r_i) = (Γ₁(r_i)/ν) · [ν(Γ₁+δΓ₁) - ν(Γ₁-δΓ₁)] / (2δΓ₁) / (Δr_i/R)

Both must agree. Disagreement proves a bug in either the AD/IFT chain or
the FD step size / mode tracking.

This is expensive (two eigenvalue solves per perturbed grid point), so we
subsample the grid — typically every ~100th point (giving ~20 kernel values
across the star, sufficient for integral and weighted pointwise comparison).
With optimized frequency search (narrow window, few scan points), each solve
takes ~2s, so 20 points ≈ 80s.

References:
    Christensen-Dalsgaard (2008), Ap&SS 316, 113
    Basu & Christensen-Dalsgaard (1997), astro-ph/9702162
"""
import numpy as np

from .eigenfunction import _select_mode_by_npg
def _compute_reference_frequency(glob, var, l, n_pg, n_scan, n_steps):
    """Compute reference eigenfrequency for a given (l, n_pg) if not provided."""
    from stellar_jax.oscillations.eigenvalue import compute_oscillation_freqs_jax
    from .seismic_quantities import large_separation

    dnu_star = large_separation(glob, var)
    nu_est = dnu_star * (n_pg + l / 2.0 + 1.2)
    auto_nu_min = max(100.0, nu_est - 5 * dnu_star)
    auto_nu_max = min(8000.0, nu_est + 5 * dnu_star)
    freqs_ref = compute_oscillation_freqs_jax(
        glob, var, l_values=(l,),
        nu_min=auto_nu_min, nu_max=auto_nu_max,
        n_scan=n_scan, n_steps=n_steps)
    ref_freqs = freqs_ref[l]
    if len(ref_freqs) == 0:
        raise ValueError(f"No modes found for l={l}")
    mode_idx = _select_mode_by_npg(ref_freqs, glob, var, l, n_pg)
    return ref_freqs[mode_idx]


def compute_fd_kernel(glob, var, l, n_pg, nu_ref=None,
                      n_scan=500, n_steps=8000, eps=1e-4, stride=100):
    """Compute the structure kernel K_{Γ₁,ρ}(r) by finite differences.

    For a subset of FGONG grid points (every `stride`-th), perturbs Γ₁ by
    ±eps fraction and recomputes the eigenfrequency to obtain:

        ∂ν/∂Γ₁(r_i) ≈ [ν(Γ₁+eps·Γ₁) - ν(Γ₁-eps·Γ₁)] / (2·eps·Γ₁)

    Then converts to the standard kernel convention:
        K(r_i) = (Γ₁(r_i)/ν) · ∂ν/∂Γ₁(r_i) / (Δr_i/R)

    Args:
        glob: FGONG global parameters
        var: FGONG per-point variables (center-to-surface)
        l: angular degree
        n_pg: target radial order
        nu_ref: reference eigenfrequency (µHz); computed if None
        n_scan: scan points for initial bracket detection
        n_steps: RK4 integration steps
        eps: relative perturbation amplitude (default 1e-4)
        stride: subsample every N-th grid point (default 100)

    Returns:
        dict with:
            'x': (M,) fractional radii of the subsampled points
            'K_gamma1_rho': (M,) FD kernel values
            'K_c2_rho': (M,) same (= K_gamma1_rho at fixed ρ)
            'dr_over_R': (M,) grid spacing at subsampled points
            'nu': reference eigenfrequency in µHz
            'l': angular degree
            'n_pg': radial order
            'indices': (M,) indices into the physical FGONG grid
            'integral': float, partial kernel integral (subsampled)
    """
    from stellar_jax.oscillations.eigenvalue import compute_oscillation_freqs_jax

    R = float(glob[1])
    r_fgong = var[:, 0]
    x_fgong = r_fgong / R

    # Physical grid (exclude center)
    mask = x_fgong > 1e-4
    x_full = x_fgong[mask]
    r_full = r_fgong[mask]
    N_full = len(x_full)

    # Compute reference frequency if not provided
    if nu_ref is None:
        nu_ref = _compute_reference_frequency(glob, var, l, n_pg, n_scan, n_steps)

    # Subsample indices (exclude extreme center and surface)
    valid = (x_full > 0.05) & (x_full < 0.98)
    all_valid = np.where(valid)[0]
    sub_indices = all_valid[::stride]
    M = len(sub_indices)

    # Grid spacing
    dr = np.zeros(N_full)
    dr[1:-1] = (r_full[2:] - r_full[:-2]) / 2.0
    dr[0] = r_full[1] - r_full[0]
    dr[-1] = r_full[-1] - r_full[-2]
    dr_over_R_full = dr / R

    # Map from physical grid to full FGONG array
    full_to_fgong = np.where(mask)[0]

    # FD kernel at subsampled points
    K_fd = np.zeros(M)
    x_sub = np.zeros(M)
    dr_sub = np.zeros(M)

    # Narrow search window for fast per-point solves
    search_half = 20.0  # µHz around nu_ref

    for k, idx in enumerate(sub_indices):
        fgong_idx = full_to_fgong[idx]
        x_sub[k] = x_full[idx]
        dr_sub[k] = dr_over_R_full[idx]

        g1_val = var[fgong_idx, 9]
        delta = eps * g1_val

        # +eps perturbation
        var_plus = var.copy()
        var_plus[fgong_idx, 9] = g1_val + delta
        freqs_plus = compute_oscillation_freqs_jax(
            glob, var_plus, l_values=(l,),
            nu_min=nu_ref - search_half, nu_max=nu_ref + search_half,
            n_scan=50, n_steps=n_steps)
        freqs_p = freqs_plus[l]
        if len(freqs_p) == 0:
            K_fd[k] = 0.0
            continue
        nu_plus = freqs_p[int(np.argmin(np.abs(np.array(freqs_p) - nu_ref)))]

        # -eps perturbation
        var_minus = var.copy()
        var_minus[fgong_idx, 9] = g1_val - delta
        freqs_minus = compute_oscillation_freqs_jax(
            glob, var_minus, l_values=(l,),
            nu_min=nu_ref - search_half, nu_max=nu_ref + search_half,
            n_scan=50, n_steps=n_steps)
        freqs_m = freqs_minus[l]
        if len(freqs_m) == 0:
            K_fd[k] = 0.0
            continue
        nu_minus = freqs_m[int(np.argmin(np.abs(np.array(freqs_m) - nu_ref)))]

        # Central FD derivative
        dnu_dgamma1 = (nu_plus - nu_minus) / (2.0 * delta)

        # Kernel convention: K = (Γ₁/ν) · (∂ν/∂Γ₁) / (Δr/R)
        dr_safe = max(dr_sub[k], 1e-30)
        K_fd[k] = (g1_val / nu_ref) * dnu_dgamma1 / dr_safe

    integral = float(np.sum(K_fd * dr_sub))

    return {
        'x': x_sub,
        'K_gamma1_rho': K_fd,
        'K_c2_rho': K_fd.copy(),
        'dr_over_R': dr_sub,
        'nu': nu_ref,
        'l': l,
        'n_pg': n_pg,
        'indices': sub_indices,
        'integral': integral,
    }
