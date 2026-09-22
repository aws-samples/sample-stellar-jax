"""Eigenvalue search: frequency scan + Brent root-finding.

The eigenvalue search loop is forward-only (Python/NumPy), NOT in the JAX
autodiff graph. It scans σ² values to bracket sign changes in the shooting
determinant, then refines with Brent's method.

The differentiable path is via adjoint.eigenfreq_from_coeffs (IFT @custom_vjp).

References:
    Christensen-Dalsgaard (2008), Ap&SS 316, 113 (ADIPLS)
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from .coefficients import _build_oscillation_grid_jax, _prepare_coeffs, build_oscillation_coeffs_jax
from .determinant import (
    radial_determinant, nonradial_determinant,
    radial_determinant_jcd, nonradial_determinant_jcd,
)
from .integrator import _make_integration_grid


def _bracket_and_refine(det_fn, nu_arr, factor, rtol=1e-10, maxiter=100):
    """Scan for sign changes in the shooting determinant and refine roots via Brent's method.

    This is the shared eigenvalue-search kernel used by all three public functions
    in this module. It implements the standard approach: evaluate the determinant
    on a frequency grid, identify brackets (adjacent points with opposite signs),
    and refine each bracket with scipy.optimize.brentq.

    The search is forward-only (Python/NumPy loop around a JIT-compiled determinant),
    NOT in the JAX autodiff graph.

    Args:
        det_fn: callable(sigma2: jnp.float64) -> float
            JIT-compiled determinant function. Takes dimensionless σ² and returns
            the surface-BC residual (sign change = eigenvalue bracket).
        nu_arr: np.ndarray, shape (n_scan,)
            Trial frequencies in μHz for the bracket scan.
        factor: float
            Conversion factor ν² → σ² (depends on M, R, G).
        rtol: float
            Relative tolerance for Brent's method (default 1e-10).
        maxiter: int
            Maximum iterations for Brent's method (default 100).

    Returns:
        list of float: sorted eigenfrequencies in μHz found within the scan range.

    References:
        the module redesign spec §6 (DRY extraction target)
        the dev guidelines §4 (DRY principle)
    """
    sigma2_arr = nu_arr**2 * factor
    det_values = np.array([float(det_fn(jnp.float64(s2))) for s2 in sigma2_arr])

    roots_nu = []
    for i in range(len(det_values) - 1):
        if np.isnan(det_values[i]) or np.isnan(det_values[i + 1]):
            continue
        if det_values[i] * det_values[i + 1] < 0:
            def f_root(nu):
                s2 = nu**2 * factor
                return float(det_fn(jnp.float64(s2)))
            try:
                nu_root = brentq(f_root, nu_arr[i], nu_arr[i + 1],
                                 rtol=rtol, maxiter=maxiter)
                roots_nu.append(nu_root)
            except (ValueError, RuntimeError):
                continue

    return sorted(roots_nu)


def compute_oscillation_freqs_jax(glob, var, l_values=(0, 1, 2, 3),
                                  nu_min=1000.0, nu_max=4500.0, n_scan=500,
                                  n_steps=8000, outer_bc='VACUUM'):
    """Compute adiabatic p-mode frequencies using the full 4th-order equations in JAX.

    This is the JAX equivalent of the NumPy compute_oscillation_freqs_full.
    Same physics (ADIPLS Eqs. 11-14 nonradial, Eqs. 17-18 radial), same BCs,
    same grid domain — different integration backend (fixed-step RK4 via lax.scan
    instead of scipy adaptive RK45).

    The eigenvalue search (bracket + Brent refinement) remains in Python/NumPy.
    The forward ODE integration is in JAX and is JIT-compiled.

    Args:
        glob: FGONG global parameters
        var: FGONG per-point variables (center-to-surface)
        l_values: angular degrees to compute
        nu_min: minimum frequency in μHz
        nu_max: maximum frequency in μHz
        n_scan: number of trial frequencies for root bracketing
        n_steps: number of RK4 steps for the integration (default 8000)
        outer_bc: outer boundary condition — 'VACUUM' (default) or 'JCD'
            (Christensen-Dalsgaard 2008 isothermal atmosphere).

    Returns:
        dict mapping l -> array of frequencies in μHz
    """
    if outer_bc not in ('VACUUM', 'JCD'):
        raise ValueError(f"outer_bc must be 'VACUUM' or 'JCD', got '{outer_bc}'")

    # Build structure grid (NumPy extraction from FGONG)
    grid_data = _build_oscillation_grid_jax(glob, var)
    x_grid_jax, coeffs_jax = _prepare_coeffs(grid_data)
    factor = grid_data['factor']
    gamma1_s = jnp.float64(grid_data['gamma1_s'])

    # Fixed integration grid
    x_steps, h_steps = _make_integration_grid(grid_data['x_grid'], n_steps=n_steps)

    # JIT-compile the determinant functions
    if outer_bc == 'VACUUM':
        @jax.jit
        def _radial_det_jit(sigma2):
            return radial_determinant(sigma2, x_steps, h_steps, x_grid_jax, coeffs_jax)

        @jax.jit
        def _nonradial_det_jit(sigma2, l_float):
            return nonradial_determinant(sigma2, l_float, x_steps, h_steps,
                                         x_grid_jax, coeffs_jax)
    else:  # JCD
        @jax.jit
        def _radial_det_jit(sigma2):
            return radial_determinant_jcd(sigma2, x_steps, h_steps,
                                          x_grid_jax, coeffs_jax, gamma1_s)

        @jax.jit
        def _nonradial_det_jit(sigma2, l_float):
            return nonradial_determinant_jcd(sigma2, l_float, x_steps, h_steps,
                                             x_grid_jax, coeffs_jax, gamma1_s)

    results = {}
    for l in l_values:
        nu_arr = np.linspace(nu_min, nu_max, n_scan)

        if l == 0:
            det_fn = _radial_det_jit
        else:
            l_float = float(l)
            def det_fn(sigma2, _l=l_float):
                return _nonradial_det_jit(sigma2, jnp.float64(_l))

        freqs = _bracket_and_refine(det_fn, nu_arr, factor,
                                    rtol=1e-8, maxiter=50)
        results[l] = np.array(freqs)
    return results


def compute_eigenfreq_differentiable(glob, var, l, nu_min, nu_max,
                                     n_scan=200, n_steps=8000, mode_index=0):
    """Compute a single eigenfrequency with IFT adjoint (differentiable w.r.t. structure).

    Given a stellar structure (FGONG format) and a mode (l, frequency bracket), it:
    1. Builds the JAX integration grid from the FGONG structure
    2. Finds the eigenfrequency via Brent (forward-only, outside JAX)
    3. Returns σ² wrapped with @custom_vjp so that jax.grad through it uses
       the IFT formula: ∂σ²/∂θ = −(∂D/∂θ)/(∂D/∂σ²)

    Args:
        glob: FGONG global parameters (numpy)
        var: FGONG per-point variables center-to-surface (numpy)
        l: angular degree (0, 1, 2, 3, ...)
        nu_min: lower frequency bound for root search (μHz)
        nu_max: upper frequency bound for root search (μHz)
        n_scan: frequency scan points for bracket detection
        n_steps: RK4 integration steps
        mode_index: which root to return (0 = lowest in bracket)

    Returns:
        dict with:
            'nu': frequency in μHz (float)
            'sigma2': dimensionless σ² (float)
            'factor': ν²→σ² conversion factor
            'x_grid': JAX array of structure grid points
            'coeffs': JAX array of structure coefficients (5, N)
            'x_steps': integration grid x-values
            'h_steps': integration step sizes
            'l': angular degree
    """
    # Build structure grid
    grid_data = _build_oscillation_grid_jax(glob, var)
    x_grid_jax, coeffs_jax = _prepare_coeffs(grid_data)
    factor = grid_data['factor']

    # Fixed integration grid
    x_steps, h_steps = _make_integration_grid(grid_data['x_grid'], n_steps=n_steps)

    # JIT-compile determinant
    if l == 0:
        @jax.jit
        def det_fn(s2):
            return radial_determinant(s2, x_steps, h_steps, x_grid_jax, coeffs_jax)
    else:
        l_float = float(l)
        @jax.jit
        def det_fn(s2):
            return nonradial_determinant(s2, jnp.float64(l_float),
                                         x_steps, h_steps, x_grid_jax, coeffs_jax)

    # Find eigenfrequencies via shared helper
    nu_arr = np.linspace(nu_min, nu_max, n_scan)
    roots_nu = _bracket_and_refine(det_fn, nu_arr, factor,
                                   rtol=1e-10, maxiter=100)

    if len(roots_nu) <= mode_index:
        raise ValueError(
            f"Only {len(roots_nu)} modes found in [{nu_min}, {nu_max}] μHz "
            f"for l={l}, but mode_index={mode_index} requested")

    nu_converged = roots_nu[mode_index]
    sigma2_converged = jnp.float64(nu_converged**2 * factor)

    return {
        'nu': nu_converged,
        'sigma2': sigma2_converged,
        'factor': factor,
        'x_grid': x_grid_jax,
        'coeffs': coeffs_jax,
        'x_steps': x_steps,
        'h_steps': h_steps,
        'l': l,
    }


def compute_eigenfreq_from_structure_jax(glob_jax, var_jax, l, nu_min, nu_max,
                                         n_scan=200, n_steps=8000, mode_index=0):
    """Compute an eigenfrequency from live JAX structure arrays (end-to-end differentiable).

    This is the OSC-3 public API: takes FGONG-layout JAX arrays (from
    evolution.structure_to_fgong_jax) and returns the eigenfrequency with
    gradients flowing all the way back to the stellar parameters (M, Z, α, ...).

    The chain: evolve_star → structure_to_fgong_jax → this function → eigenfreq_from_coeffs.

    The root-finding (Brent) runs forward-only (outside JAX); the gradient at the
    converged root is provided by the IFT @custom_vjp from adjoint.py.

    Args:
        glob_jax: (15,) JAX array — FGONG global parameters
        var_jax: (N, 15) JAX array — per-point structure (center-to-surface)
        l: angular degree (0, 1, 2, 3, ...)
        nu_min: lower frequency bound for root search (μHz)
        nu_max: upper frequency bound for root search (μHz)
        n_scan: frequency scan points for bracket detection
        n_steps: RK4 integration steps
        mode_index: which root to return (0 = lowest in bracket)

    Returns:
        dict with:
            'nu': frequency in μHz (float)
            'sigma2': dimensionless σ² (JAX scalar)
            'factor': ν²→σ² conversion factor
            'x_grid': JAX array of structure grid points
            'coeffs': JAX array of structure coefficients (5, N)
            'x_steps': integration grid x-values
            'h_steps': integration step sizes
            'l': angular degree
    """
    # Build oscillation coefficients from live JAX structure
    grid_data = build_oscillation_coeffs_jax(glob_jax, var_jax)
    x_grid_jax = grid_data['x_grid']
    coeffs_jax = grid_data['coeffs']
    factor = float(grid_data['factor'])
    n_center = grid_data['n_center']

    # Fixed integration grid — use PHYSICAL points only (skip synthetic center
    # extension points). The synthetic center points exist for jnp.interp
    # coverage near x→0, but the integration grid must start at the first
    # physical mesh point to preserve gradient flow (synthetic points are
    # stop_gradient'd and would cut the AD gradient path).
    # After skipping n_center synthetic points, also exclude any remaining
    # x <= 1e-4 (the x=0 center point in MESA FGONGs).
    # ADIPLS starts at nibc=2 when x(1)=0 (adipls.c.d.f:1640-1641).
    x_grid_np = np.asarray(x_grid_jax)
    x_grid_physical = x_grid_np[n_center:]  # skip synthetic center points
    x_grid_for_integration = x_grid_physical[x_grid_physical > 1e-4]
    if len(x_grid_for_integration) == 0:
        R_val = float(grid_data.get('R', np.nan))
        r_max = float(np.max(np.asarray(var_jax[:, 0]))) if var_jax.shape[0] > 0 else 0.0
        raise ValueError(
            f"Empty oscillation grid: no physical points with x = r/R > 1e-4. "
            f"R_star={R_val:.3e} cm, max(r)={r_max:.3e} cm, "
            f"x_max={r_max/max(R_val, 1e-30):.3e}. "
            f"This means R_star (glob[1], from log_L/log_Te) is vastly "
            f"larger than the radii in y_henyey — the structure is "
            f"inconsistent. Ensure log_L_final and log_Teff_final are "
            f"derived from the same Henyey solution as y_henyey_final."
        )
    x_steps, h_steps = _make_integration_grid(
        x_grid_for_integration, n_steps=n_steps)

    # JIT-compile determinant
    if l == 0:
        @jax.jit
        def det_fn(s2):
            return radial_determinant(s2, x_steps, h_steps, x_grid_jax, coeffs_jax)
    else:
        l_float = float(l)
        @jax.jit
        def det_fn(s2):
            return nonradial_determinant(s2, jnp.float64(l_float),
                                         x_steps, h_steps, x_grid_jax, coeffs_jax)

    # Find eigenfrequencies via shared helper
    nu_arr = np.linspace(nu_min, nu_max, n_scan)
    roots_nu = _bracket_and_refine(det_fn, nu_arr, factor,
                                   rtol=1e-10, maxiter=100)

    if len(roots_nu) <= mode_index:
        raise ValueError(
            f"Only {len(roots_nu)} modes found in [{nu_min}, {nu_max}] μHz "
            f"for l={l}, but mode_index={mode_index} requested")

    nu_converged = roots_nu[mode_index]
    sigma2_converged = jnp.float64(nu_converged**2 * factor)

    return {
        'nu': nu_converged,
        'sigma2': sigma2_converged,
        'factor': factor,
        'x_grid': x_grid_jax,
        'coeffs': coeffs_jax,
        'x_steps': x_steps,
        'h_steps': h_steps,
        'l': l,
    }
