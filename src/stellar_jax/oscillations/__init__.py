"""Oscillations package — adiabatic pulsation eigensolver + IFT adjoint.

Responsibility: compute eigenfrequencies of stellar oscillation modes and
provide analytic gradients (∂ν/∂structure) via the implicit function theorem.

Public API:
  - Differentiable path (JAX, in the AD graph):
      build_oscillation_coeffs_jax, eigenfreq_from_coeffs,
      compute_eigenfreq_differentiable, compute_eigenfreq_from_structure_jax,
      compute_oscillation_freqs_jax
  - Diagnostic path (NumPy/SciPy, NOT in AD graph):
      compute_oscillation_freqs, compute_oscillation_freqs_full
  - I/O:
      read_fgong, fgong_components
  - Seismic utilities:
      echelle_data, estimate_delta_nu, compute_solar_model_freqs,
      compute_evolved_solar_model_freqs, compare_echelle
  - Types/contracts:
      OscillationCoeffs, EigenfreqResult, IntegrationGrid,
      validate_coeffs_shape, validate_fgong_arrays

References:
    Christensen-Dalsgaard (2008), Ap&SS 316, 113 (ADIPLS)
"""
# ─── Types & contracts ─────────────────────────────────────────────────────────
from .contracts import (
    OscillationCoeffs,
    EigenfreqResult,
    IntegrationGrid,
    validate_coeffs_shape,
    validate_fgong_arrays,
)

# ─── I/O ───────────────────────────────────────────────────────────────────────
from stellar_jax.fgong.io import read_fgong, fgong_components

# ─── Differentiable path (JAX) ────────────────────────────────────────────────
from .coefficients import (
    build_oscillation_coeffs_jax,
    _build_oscillation_grid_jax,
    _prepare_coeffs,
)
from .integrator import _make_integration_grid
from .adjoint import eigenfreq_from_coeffs, eigenfreq_radial, eigenfreq_nonradial
from .eigenvalue import (
    _bracket_and_refine,
    compute_oscillation_freqs_jax,
    compute_eigenfreq_differentiable,
    compute_eigenfreq_from_structure_jax,
)
from .determinant import (
    radial_determinant, nonradial_determinant,
    radial_determinant_jcd, nonradial_determinant_jcd,
)

# ─── Diagnostic path (NumPy) ──────────────────────────────────────────────────
from .diagnostic import (
    _build_oscillation_grid,
    compute_oscillation_freqs,
    compute_oscillation_freqs_full,
)

# ─── σ² ↔ ν conversion (canonical, DRY —) ──────────────────────────────
from .seismic_conversion import compute_factor, nu_from_sigma2, sigma2_from_nu

# ─── Eigenfunction output ───────────────────────────────────────────────
from .eigenfunction import compute_eigenfunction, compute_mode_inertia

# ─── Structure kernels ──────────────────────────────────────────────────
from .kernels import compute_structure_kernels, kernel_integral

# ─── Composition kernels ──────────────────────────────────────────────
from .kernels import compute_composition_kernels, composition_kernel_integral

# ─── Independent reference kernels (FD — for validating AD kernels) ───────────
from .analytic_kernels import compute_fd_kernel

# ─── Structure inversions ──────────────────────────────────────────────
from .inversions import sola_inversion

# ─── Seismic utilities ─────────────────────────────────────────────────────────
from .seismic_quantities import (
    # n_pg identification
    assign_n_pg,
    identify_modes,
    # Small separations
    delta_nu_02,
    delta_nu_01,
    # Frequency ratios (Roxburgh & Vorontsov 2003)
    r02,
    r01,
    r10,
    # Differentiable frequency ratios
    r02_differentiable,
    # Differentiable r₀₁ ratio
    r01_differentiable,
    # Structure-based quantities
    large_separation,
    nu_max_scaling,
    # Asymptotic phase
    epsilon_fit,
)

# Legacy / utility — moved from seismic_quantities to diagnostic
from .diagnostic import (
    echelle_data,
    estimate_delta_nu,
    compute_solar_model_freqs,
    compute_evolved_solar_model_freqs,
    compare_echelle,
)



# ─── Surface correction (Ball & Gizon 2014) ──────────────────────────────────
from .surface_correction import (
    acoustic_cutoff_frequency,
    surface_correction_bg14,
    chi2_with_surface_correction,
)

# ─── Acoustic glitch signatures ───────────────────────────────────────────────
from .glitch import (
    compute_acoustic_depth_profile,
    compute_acoustic_depth_jax,
    second_differences,
    fit_glitch_signatures,
)

__all__ = [
    # Types
    'OscillationCoeffs', 'EigenfreqResult', 'IntegrationGrid',
    'validate_coeffs_shape', 'validate_fgong_arrays',
    # I/O
    'read_fgong', 'fgong_components',
    # Differentiable path
    'build_oscillation_coeffs_jax', 'eigenfreq_from_coeffs',
    'eigenfreq_radial', 'eigenfreq_nonradial',
    'compute_oscillation_freqs_jax', 'compute_eigenfreq_differentiable',
    'compute_eigenfreq_from_structure_jax',
    'radial_determinant', 'nonradial_determinant',
    'radial_determinant_jcd', 'nonradial_determinant_jcd',
    # Diagnostic
    'compute_oscillation_freqs', 'compute_oscillation_freqs_full',
    # Eigenfunction
    'compute_eigenfunction',
    # Mode inertia
    'compute_mode_inertia',
    # Structure kernels
    'compute_structure_kernels', 'kernel_integral',
    # Composition kernels
    'compute_composition_kernels', 'composition_kernel_integral',
    # Analytic kernels (ADIPLS variational formula — independent reference)
    'compute_fd_kernel',
    # Structure inversions
    'sola_inversion',
    # Seismic quantities — n_pg identification
    'assign_n_pg', 'identify_modes',
    # Seismic quantities — separations
    'delta_nu_02', 'delta_nu_01',
    # Seismic quantities — ratios (Roxburgh & Vorontsov 2003)
    'r02', 'r01', 'r10',
    # Seismic quantities — differentiable ratios
    'r02_differentiable',
    # Seismic quantities — differentiable r₀₁ ratio
    'r01_differentiable',
    # Seismic quantities — structure-based
    'large_separation', 'nu_max_scaling',
    # Seismic quantities — asymptotic phase
    'epsilon_fit',
    # Seismic utilities (legacy)
    'echelle_data', 'estimate_delta_nu', 'compute_solar_model_freqs',
    'compute_evolved_solar_model_freqs', 'compare_echelle',
    # Surface correction
    'acoustic_cutoff_frequency', 'surface_correction_bg14',
    'chi2_with_surface_correction',
    # Acoustic glitch signatures
    'compute_acoustic_depth_profile', 'compute_acoustic_depth_jax',
    'second_differences', 'fit_glitch_signatures',
]
