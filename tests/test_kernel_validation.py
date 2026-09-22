"""Validate AD structure kernels against independent finite-difference reference.

This is the AC1 deliverable for issue #782: an independent pointwise comparison
of our AD kernels (jax.grad through the IFT adjoint) against a kernel computed
by direct numerical finite-difference differentiation of the eigenfrequency.

WHAT: for solar p-modes on Model S, compare K_{Γ₁,ρ}(r) computed by two
fundamentally different methods:
  (a) AD kernel (kernels.py): jax.grad of ν w.r.t. Γ₁, through the IFT
      adjoint ∂σ²/∂coeffs and the coefficient-building chain.
  (b) FD kernel (analytic_kernels.py): perturb Γ₁(r_i) by ±ε, recompute ν
      from scratch, take the centered finite-difference derivative. No AD,
      no IFT, no coefficient-chain gradient — just forward eigenvalue solves.

WHY: these two methods share ONLY the forward eigenvalue solver (Brent root-
find on the oscillation determinant). The AD kernel differentiates through
the coefficient-building, the IFT adjoint (@custom_vjp), and the Γ₁→Vg chain.
The FD kernel recomputes the eigenfrequency from scratch for each perturbation.
Agreement proves the entire AD/IFT chain is correct; disagreement exposes a bug
in either the AD chain or the FD mode tracking.

EXTERNAL REFERENCE: the Model S FGONG (Christensen-Dalsgaard et al. 1996).
The FD method is the standard numerical-differentiation approach used across
computational physics. The ADIPLS gm1ker formula (Christensen-Dalsgaard 2008)
is the field-standard analytic kernel; we verify consistency with the
variational principle by construction (the FD kernel IS the derivative).

TOLERANCE: pointwise relative error < 1% at each FD sample point in the
acoustic cavity (0.1 < r/R < 0.95). The FD uses a centered difference with
ε = 1e-4 (relative), giving O(ε²) = O(1e-8) truncation error — far below
the 1% tolerance. The kernel integral is not directly compared because the
FD kernel is subsampled (~20 points), but the pointwise agreement + the
existing variational self-consistency test (test_structure_kernels.py, 5%)
together validate the full kernel.

MUTATION: corrupt_gamma1 — scales Γ₁ by 50%. The AD kernel linearizes at
the wrong eigenvalue (IFT at wrong σ²), while the FD kernel uses the wrong
structure. Both produce kernels, but they will be different because the AD
chain compounds the Γ₁ corruption differently (through coefficient building)
than the FD recomputation.

References:
    Christensen-Dalsgaard (2008), Ap&SS 316, 113 (ADIPLS)
    Basu & Christensen-Dalsgaard (1997), astro-ph/9702162
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('JAX_ENABLE_X64', '1')


@pytest.fixture(scope="module")
def model_s_fgong():
    """Load the Model S FGONG file (committed reference data)."""
    fgong_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "model_s", "fgong.l5bi.d.15c"
    )
    assert os.path.isfile(fgong_path), f"Model S FGONG not found: {fgong_path}"
    from stellar_jax.fgong.io import read_fgong
    return read_fgong(fgong_path)


@pytest.mark.validation
@pytest.mark.mutation("corrupt_gamma1")
@pytest.mark.integration
class TestADvsFDKernel:
    """AD kernel vs finite-difference kernel on Model S.

    WHAT: pointwise comparison of two independent kernel computations —
    AD (jax.grad through IFT) vs FD (centered numerical derivative by
    recomputing the eigenfrequency for each perturbed Γ₁ point).

    WHY: proves the AD/IFT chain (adjoint.py + coefficients.py + kernels.py)
    correctly computes the per-point frequency sensitivity ∂ν/∂Γ₁(r). This
    is the strongest possible validation — the FD kernel is ground truth
    by construction.

    EXTERNAL REFERENCE: Model S FGONG (Christensen-Dalsgaard et al. 1996).
    FD with ε=1e-4 gives O(1e-8) truncation error.

    TOLERANCE: 1% relative at each point where the kernel is significant.

    MUTATION: corrupt_gamma1 breaks the agreement because the AD linearizes
    at the wrong eigenvalue while FD recomputes from the corrupted structure.
    """

    def test_radial_mode_ad_vs_fd(self, model_s_fgong):
        """AD kernel matches FD for l=0, n_pg=20 (~2900 µHz)."""
        from stellar_jax.oscillations.kernels import compute_structure_kernels
        from stellar_jax.oscillations.analytic_kernels import compute_fd_kernel

        glob, var = model_s_fgong

        # Compute AD kernel
        kr_ad = compute_structure_kernels(
            glob, var, l=0, n_pg=20,
            nu_min=2000.0, nu_max=3500.0, n_scan=400, n_steps=8000)

        # Compute FD kernel at subsampled points
        kr_fd = compute_fd_kernel(
            glob, var, l=0, n_pg=20,
            nu_ref=kr_ad['nu'], n_steps=8000,
            eps=1e-4, stride=120)

        self._compare_ad_fd(kr_ad, kr_fd, mode_label="l=0, n_pg=20")

    def test_nonradial_mode_ad_vs_fd(self, model_s_fgong):
        """AD kernel matches FD for l=2, n_pg=17 (~2500 µHz)."""
        from stellar_jax.oscillations.kernels import compute_structure_kernels
        from stellar_jax.oscillations.analytic_kernels import compute_fd_kernel

        glob, var = model_s_fgong

        kr_ad = compute_structure_kernels(
            glob, var, l=2, n_pg=17,
            nu_min=2000.0, nu_max=3500.0, n_scan=400, n_steps=8000)

        kr_fd = compute_fd_kernel(
            glob, var, l=2, n_pg=17,
            nu_ref=kr_ad['nu'], n_steps=8000,
            eps=1e-4, stride=120)

        self._compare_ad_fd(kr_ad, kr_fd, mode_label="l=2, n_pg=17")

    @staticmethod
    def _compare_ad_fd(kr_ad, kr_fd, mode_label):
        """Compare AD kernel to FD kernel at the FD sample points.

        Args:
            kr_ad: dict from compute_structure_kernels (AD kernel, full grid)
            kr_fd: dict from compute_fd_kernel (FD kernel, subsampled)
            mode_label: string for error messages
        """
        x_fd = kr_fd['x']
        K_fd = kr_fd['K_gamma1_rho']
        x_ad = kr_ad['x']
        K_ad = kr_ad['K_gamma1_rho']

        # Interpolate AD kernel to FD sample points
        K_ad_at_fd = np.interp(x_fd, x_ad, K_ad)

        # Compare only in the acoustic cavity (0.1 < r/R < 0.95)
        # and where the kernel is significant (> 1% of peak)
        cavity = (x_fd > 0.1) & (x_fd < 0.95)
        peak = max(np.max(np.abs(K_ad_at_fd[cavity])),
                   np.max(np.abs(K_fd[cavity])))
        significant = cavity & (np.maximum(np.abs(K_ad_at_fd),
                                           np.abs(K_fd)) > 0.01 * peak)

        n_sig = int(np.sum(significant))
        assert n_sig >= 5, (
            f"Too few significant points ({n_sig}) for comparison — "
            f"check mode or stride")

        # Pointwise relative error at each significant point
        K_ad_sig = K_ad_at_fd[significant]
        K_fd_sig = K_fd[significant]
        x_sig = x_fd[significant]

        rel_errors = np.abs(K_ad_sig - K_fd_sig) / np.maximum(np.abs(K_fd_sig), 1e-30)
        max_rel = float(np.max(rel_errors))
        worst_idx = int(np.argmax(rel_errors))

        assert max_rel < 0.01, (
            f"AD vs FD kernel disagreement for {mode_label}:\n"
            f"  Max relative error = {max_rel:.4%} at x={x_sig[worst_idx]:.3f}\n"
            f"  AD = {K_ad_sig[worst_idx]:.6f}, FD = {K_fd_sig[worst_idx]:.6f}\n"
            f"  N significant points = {n_sig}\n"
            f"  Tolerance: 1% pointwise")

        # Also check that the AD kernel is non-trivial
        assert np.max(np.abs(K_ad)) > 0.01, (
            f"AD kernel is essentially zero for {mode_label}")

        # Check kernel peaks are at similar locations
        peak_x_ad = x_ad[np.argmax(np.abs(K_ad))]
        peak_x_fd = x_fd[np.argmax(np.abs(K_fd))]
        assert abs(peak_x_ad - peak_x_fd) < 0.1, (
            f"Kernel peaks at different locations for {mode_label}:\n"
            f"  AD peak at x={peak_x_ad:.3f}, FD peak at x={peak_x_fd:.3f}")
