"""Tests for AD structure kernels K^{c²,ρ}(r) and K^{Γ₁,ρ}(r) (#781).

Validates the AD structure kernels by checking the variational consistency:
the kernel integral ∫K·δΓ₁/Γ₁·dr/R must predict the actual δν/ν from a
perturbed model to within the linearization error.

WHAT: AD structure kernels — sensitivity of oscillation frequencies to radial
perturbations in c²(r) and Γ₁(r) at fixed ρ.

WHY: these kernels are the fundamental input to SOLA/RLS seismic inversions
(Basu & Christensen-Dalsgaard 1997; Dziembowski et al. 1990). Without them
the asteroseismic inversion pipeline cannot be validated. This test ensures
the AD kernels are consistent with the eigenfrequency solver to linear order.

EXTERNAL REFERENCE: the variational principle itself (Basu & CD 1997 Eq. 1):
    δν/ν = ∫ K_{Γ₁,ρ}(r) · δΓ₁/Γ₁ · dr/R
applied with a KNOWN perturbation δΓ₁/Γ₁ = 1% Gaussian bump at r/R = 0.5
(width 0.05). The "reference" is not an external dataset but the SELF-CONSISTENT
forward recomputation with the perturbed model — the linearization error is the
physics being tested, bounded by (δΓ₁/Γ₁)² ≈ O(1e-4) relative to the linear term.

TOLERANCE: 5% relative agreement between the kernel-predicted δν/ν and the
actual (re-solved) δν/ν. This is generous for a 1% perturbation (linearization
error is O(1e-4), so the kernel should agree to <1%), but allows for:
- Brent root-finding tolerance (~1e-8 relative in σ²)
- Grid discretization effects (trapezoidal vs continuous integral)
- Center-extension artifacts in the coefficient grid

WHAT MAKES IT FAIL: @mutation corrupt_gamma1 — corrupting Γ₁ by 50% moves the
eigenfrequency by ~22%, so the IFT adjoint (linearized at the WRONG σ²) produces
a drastically wrong kernel. The kernel integral no longer predicts δν/ν correctly.

References:
    Basu & Christensen-Dalsgaard (1997), astro-ph/9702162, Eq. 1
    Gough & Thompson (1991), "Solar Interior and Atmosphere"
    Christensen-Dalsgaard (2008), Ap&SS 316, 113 (ADIPLS)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('JAX_ENABLE_X64', '1')


# ─── Fixture: Model S FGONG ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_s_fgong():
    """Load the Model S FGONG file (committed reference data)."""
    fgong_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "model_s", "fgong.l5bi.d.15c"
    )
    assert os.path.isfile(fgong_path), (
        f"Model S FGONG not found: {fgong_path}")
    from stellar_jax.fgong.io import read_fgong
    glob, var = read_fgong(fgong_path)
    return glob, var


# ════════════════════════════════════════════════════════════════════════════════
# Test 1: Variational consistency — kernel integral predicts δν/ν
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.validation
@pytest.mark.mutation("corrupt_gamma1")
@pytest.mark.integration
class TestStructureKernelVariational:
    """Variational consistency: ∫K·δΓ₁/Γ₁·dr/R ≈ δν/ν from re-solved perturbed model.

    WHAT: verifies the AD kernel K_{Γ₁,ρ}(r) satisfies the linearized variational
    relation to within the expected linearization error.

    WHY: this is the fundamental correctness check — if the kernel doesn't predict
    the actual frequency shift, it's wrong and unusable for inversions.

    EXTERNAL REFERENCE: Basu & CD (1997) Eq. 1. The test uses the variational
    principle itself as the consistency condition. The "actual δν" comes from
    re-solving the perturbed model with the same eigenvalue solver.

    TOLERANCE: 5% relative (generous; linearization error for 1% perturbation
    is O(1e-4), so kernel should agree to <1% for smooth perturbations).

    MUTATION: corrupt_gamma1 scales Γ₁ by 1.5 → IFT linearizes at the wrong
    eigenvalue → kernel integral off by >>5%.
    """

    def test_radial_mode_kernel_variational(self, model_s_fgong):
        """Kernel integral predicts δν/ν for a radial (l=0) mode."""
        from stellar_jax.oscillations.kernels import compute_structure_kernels, kernel_integral
        from stellar_jax.oscillations.eigenvalue import compute_oscillation_freqs_jax

        glob, var = model_s_fgong
        R = glob[1]

        # Compute kernel for l=0, n_pg~20 (solar-like p-mode, ~2700 µHz)
        kernel_result = compute_structure_kernels(
            glob, var, l=0, n_pg=20,
            nu_min=2000.0, nu_max=3500.0, n_scan=400, n_steps=8000)

        nu_ref = kernel_result['nu']
        x_kernel = kernel_result['x']

        # Apply a 1% Gaussian perturbation in Γ₁ centered at r/R = 0.5
        x_center = 0.5
        width = 0.05
        amplitude = 0.01  # 1% perturbation
        delta_gamma1_over_gamma1 = amplitude * np.exp(
            -0.5 * ((x_kernel - x_center) / width) ** 2)

        # Kernel-predicted δν/ν
        dnu_over_nu_predicted = kernel_integral(kernel_result, delta_gamma1_over_gamma1)

        # Actual δν/ν: re-solve with perturbed Γ₁
        var_perturbed = var.copy()
        x_fgong = var[:, 0] / R
        perturbation_full = amplitude * np.exp(
            -0.5 * ((x_fgong - x_center) / width) ** 2)
        var_perturbed[:, 9] = var[:, 9] * (1.0 + perturbation_full)

        # Re-solve for the perturbed frequency
        freqs_perturbed = compute_oscillation_freqs_jax(
            glob, var_perturbed, l_values=(0,),
            nu_min=nu_ref - 50.0, nu_max=nu_ref + 50.0,
            n_scan=200, n_steps=8000)

        # Find the closest mode to nu_ref in the perturbed spectrum
        perturbed_freqs = freqs_perturbed[0]
        assert len(perturbed_freqs) > 0, "No modes found in perturbed model"
        idx = np.argmin(np.abs(perturbed_freqs - nu_ref))
        nu_perturbed = perturbed_freqs[idx]

        dnu_over_nu_actual = (nu_perturbed - nu_ref) / nu_ref

        # Both should be small and have the same sign
        assert dnu_over_nu_actual != 0.0, (
            "Perturbed frequency identical to reference — perturbation not applied?")

        # Relative agreement between kernel prediction and actual
        rel_error = abs(dnu_over_nu_predicted - dnu_over_nu_actual) / abs(dnu_over_nu_actual)

        assert rel_error < 0.05, (
            f"Kernel variational consistency failed for l=0 mode (ν={nu_ref:.1f} µHz):\n"
            f"  Kernel-predicted δν/ν = {dnu_over_nu_predicted:.6e}\n"
            f"  Actual δν/ν (re-solved) = {dnu_over_nu_actual:.6e}\n"
            f"  Relative error = {rel_error:.2%} (tolerance: 5%)")

    def test_nonradial_mode_kernel_variational(self, model_s_fgong):
        """Kernel integral predicts δν/ν for a nonradial (l=2) mode."""
        from stellar_jax.oscillations.kernels import compute_structure_kernels, kernel_integral
        from stellar_jax.oscillations.eigenvalue import compute_oscillation_freqs_jax

        glob, var = model_s_fgong
        R = glob[1]

        # Compute kernel for l=2, n_pg~20 (solar-like p-mode)
        kernel_result = compute_structure_kernels(
            glob, var, l=2, n_pg=20,
            nu_min=2000.0, nu_max=3500.0, n_scan=400, n_steps=8000)

        nu_ref = kernel_result['nu']
        x_kernel = kernel_result['x']

        # Apply a 0.5% Gaussian perturbation centered at r/R = 0.3
        # (radiative interior — where p-mode kernels are sensitive)
        x_center = 0.3
        width = 0.05
        amplitude = 0.005  # 0.5%
        delta_gamma1_over_gamma1 = amplitude * np.exp(
            -0.5 * ((x_kernel - x_center) / width) ** 2)

        # Kernel-predicted δν/ν
        dnu_over_nu_predicted = kernel_integral(kernel_result, delta_gamma1_over_gamma1)

        # Actual δν/ν from re-solving
        var_perturbed = var.copy()
        x_fgong = var[:, 0] / R
        perturbation_full = amplitude * np.exp(
            -0.5 * ((x_fgong - x_center) / width) ** 2)
        var_perturbed[:, 9] = var[:, 9] * (1.0 + perturbation_full)

        freqs_perturbed = compute_oscillation_freqs_jax(
            glob, var_perturbed, l_values=(2,),
            nu_min=nu_ref - 50.0, nu_max=nu_ref + 50.0,
            n_scan=200, n_steps=8000)

        perturbed_freqs = freqs_perturbed[2]
        assert len(perturbed_freqs) > 0, "No l=2 modes found in perturbed model"
        idx = np.argmin(np.abs(perturbed_freqs - nu_ref))
        nu_perturbed = perturbed_freqs[idx]

        dnu_over_nu_actual = (nu_perturbed - nu_ref) / nu_ref

        assert dnu_over_nu_actual != 0.0, (
            "Perturbed frequency identical to reference — perturbation not applied?")

        rel_error = abs(dnu_over_nu_predicted - dnu_over_nu_actual) / abs(dnu_over_nu_actual)

        assert rel_error < 0.05, (
            f"Kernel variational consistency failed for l=2 mode (ν={nu_ref:.1f} µHz):\n"
            f"  Kernel-predicted δν/ν = {dnu_over_nu_predicted:.6e}\n"
            f"  Actual δν/ν (re-solved) = {dnu_over_nu_actual:.6e}\n"
            f"  Relative error = {rel_error:.2%} (tolerance: 5%)")


# ════════════════════════════════════════════════════════════════════════════════
# Test 2: Kernel basic properties (sign, shape, finiteness)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.validation
@pytest.mark.mutation("corrupt_gamma1")
@pytest.mark.integration
class TestStructureKernelProperties:
    """Basic physical properties of the structure kernel.

    WHAT: K_{Γ₁,ρ}(r) must be finite, predominantly positive for p-modes
    (compressive energy dominates), and concentrated in the acoustic cavity.
    The integral ∫K·dr/R should be ~0.5 (since ν ∝ c ∝ Γ₁^{1/2}).

    WHY: a kernel with NaN/Inf or zero everywhere is unphysical and indicates
    a bug in the AD chain or normalization. The integral being ~0.5 validates
    the normalization convention against the asymptotic expectation.

    EXTERNAL REFERENCE: Christensen-Dalsgaard (2008) §5.1 — for high-order
    p-modes, the kernel is dominated by the positive compressive-energy
    contribution. Near-surface negative regions are physical (acoustic cutoff,
    atmosphere effects). The integral ∫K·dr/R ≈ ∂lnν/∂lnΓ₁ ≈ 0.5 (since
    ν ∝ c ∝ √(Γ₁P/ρ) and the mode samples the whole cavity).

    TOLERANCE: integral within [0.2, 0.8] — bracketing the ~0.5 asymptotic
    expectation with margin for mode-specific variations.

    MUTATION: corrupt_gamma1 shifts the eigenfrequency, so the IFT linearization
    is at the wrong root → kernel integral deviates grossly from 0.5.
    """

    def test_kernel_finite_and_nonzero(self, model_s_fgong):
        """K_{Γ₁,ρ}(r) is finite and non-trivial for a solar p-mode."""
        from stellar_jax.oscillations.kernels import compute_structure_kernels

        glob, var = model_s_fgong

        kernel_result = compute_structure_kernels(
            glob, var, l=0, n_pg=20,
            nu_min=2000.0, nu_max=3500.0, n_scan=400, n_steps=8000)

        K = kernel_result['K_gamma1_rho']

        # Finiteness
        assert np.all(np.isfinite(K)), (
            f"Kernel contains {np.sum(~np.isfinite(K))} non-finite values")

        # Not identically zero (the kernel must have structure)
        max_K = np.max(np.abs(K))
        assert max_K > 0.01, (
            f"Kernel is essentially zero: max(|K|) = {max_K:.4e}")

        # Predominantly positive (compressive energy dominates for p-modes)
        # The interior (x < 0.95) should be mostly positive
        x = kernel_result['x']
        interior = x < 0.95
        K_interior = K[interior]
        positive_fraction = np.mean(K_interior > 0)
        assert positive_fraction > 0.7, (
            f"Interior kernel mostly negative: {positive_fraction:.0%} positive "
            f"(expected >70% for a p-mode)")

    def test_kernel_integral_order_of_magnitude(self, model_s_fgong):
        """∫K_{Γ₁,ρ}·dr/R ≈ 0.5 for a solar p-mode (normalization check)."""
        from stellar_jax.oscillations.kernels import compute_structure_kernels

        glob, var = model_s_fgong

        kernel_result = compute_structure_kernels(
            glob, var, l=0, n_pg=20,
            nu_min=2000.0, nu_max=3500.0, n_scan=400, n_steps=8000)

        K = kernel_result['K_gamma1_rho']
        dr_over_R = kernel_result['dr_over_R']

        # The total integral ∫K·dr/R (= δν/ν for a uniform δΓ₁/Γ₁ = 1)
        total = float(np.sum(K * dr_over_R))

        # For p-modes, ν ∝ c ∝ Γ₁^{1/2}, so ∂lnν/∂lnΓ₁ ≈ 0.5.
        # The exact value depends on mode-specific turning point effects.
        assert 0.2 < total < 0.8, (
            f"∫K·dr/R = {total:.4f} — expected ~0.5 for a p-mode. "
            f"Normalization may be wrong.")

    # test_kernel_c2_equals_gamma1 REMOVED (sub-case A).
    #
    # The identity K_{c²,ρ} ≡ K_{Γ₁,ρ} at fixed ρ is ALGEBRAIC, not physical:
    # c² = Γ₁P/ρ, so at fixed ρ the chain rule gives
    #   K_{c²,ρ} = (c²/ν)·∂ν/∂c² = (Γ₁P/ρ)/(ν) · (∂ν/∂Γ₁)·(ρ/P) = (Γ₁/ν)·∂ν/∂Γ₁ = K_{Γ₁,ρ}
    # regardless of Γ₁ corruption. The implementation sets K_c2_rho = K_gamma1_rho.copy()
    # (kernels.py line ~232), so assert_array_equal was testing a copy equals its source —
    # vacuous by construction, insensitive to ANY mutation.
    #
    # Surviving tests in this class (test_kernel_finite_and_nonzero,
    # test_kernel_integral_order_of_magnitude) cover the kernel's physical correctness
    # and ARE sensitive to corrupt_gamma1 (the IFT adjoint linearizes at the wrong σ²
    # under mutation, producing wrong kernel values and integrals).
    # The AD-vs-FD kernel identity is separately tested by TestADvsFDKernel.
