"""Validate the intrinsic-EOS composition kernel K_{Z,ρ}(r) (#1212).

WHAT: AD composition kernel K_{Z,ρ}(r) — the per-zone sensitivity of
oscillation frequencies to metal mass fraction Z through the EOS dependence
of Γ₁ at fixed (P, ρ): Z → Γ₁(P,ρ,X,Z) → Vg → σ² → ν.

WHY: the standard kernel K_{Γ₁,ρ} treats Γ₁ as a free variable and does NOT
capture how composition changes affect ν through the EOS (Basu & Christensen-
Dalsgaard 1997 Eq. 2). The composition kernel adds the missing intrinsic-EOS
term: K_{Z,ρ}(r) = K_{Γ₁,ρ}(r) · (∂lnΓ₁/∂Z)|_{P,ρ}. Without it, ∂σ²/∂Z
is incomplete — the He-ionization Γ₁ glitch sensitivity is dropped.

EXTERNAL REFERENCE: finite-difference computation of δΓ₁ at fixed (P, ρ),
matching the Basu & CD 1997 Eq. 2 thermodynamic holding. At each zone,
Z is perturbed while T is adjusted (Newton) to keep ρ constant at fixed P.
The Model S FGONG (Christensen-Dalsgaard et al. 1996) provides the structure.

TOLERANCE: the composition kernel integral ∫K_{Z,ρ}·δZ·dr/R predicts δν/ν
from a localized Z-perturbation to within 5% relative of the actual (re-solved)
δν/ν. This is the same variational-consistency tolerance used for K_{Γ₁,ρ}
(test_structure_kernels.py). The linearization error for a 5% Z perturbation
is O((δZ)²) ≈ O(1e-4), well below 5%.

MUTATION: freeze_composition_kernel — applies stop_gradient on Z in the OPAL
EOS lookup, zeroing ∂Γ₁/∂Z while keeping Γ₁ values unchanged. This kills
K_{Z,ρ}, making the kernel integral zero and failing the variational check.

References:
    Basu & Christensen-Dalsgaard (1997), astro-ph/9702162, Eq. 2-3
    Buldgen, Reese & Dupret (2017), arXiv:1610.01460 (structural pairs)
    Verma et al. (2019), MNRAS 483, 4678 (He-glitch Y sensitivity)
    MESA pulse_fgong.f90:345-375 (thermodynamic transformation)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('JAX_ENABLE_X64', '1')


_FGONG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "model_s", "fgong.l5bi.d.15c"
)


@pytest.fixture(scope="module")
def model_s_fgong():
    """Load the Model S FGONG file (committed reference data)."""
    assert os.path.isfile(_FGONG_PATH), f"Model S FGONG not found: {_FGONG_PATH}"
    from stellar_jax.fgong.io import read_fgong
    return read_fgong(_FGONG_PATH)


def _gamma1_at_fixed_P_rho(logT_ref, logPgas, X, Z_new, rho_target,
                            _eos_fn, a_rad, PGAS_FRAC_FLOOR,
                            max_iter=20, tol=1e-10):
    """Compute Γ₁ at (Z_new) holding (P, ρ) fixed via Newton adjustment of T.

    At fixed P_gas, changing Z changes ρ = ρ(T, P, X, Z). To keep ρ constant,
    we adjust T via Newton iteration: find T' such that
    ρ(T', P_gas, X, Z_new) = ρ_target.

    Returns (Γ₁_new, logT_new) or (None, None) if Newton fails.

    This matches MESA's approach: pulse_fgong.f90:345-375 computes the derivative
    at fixed (P, ρ) by transforming from (ρ, T)-fixed using the χ_T correction.
    Here we do the equivalent numerically: perturb Z while adjusting T to maintain
    constant ρ.

    Uses concrete Python floats to avoid repeated JAX JIT compilations.
    """
    logT_i = float(logT_ref)
    logPgas_f = float(logPgas)
    X_f = float(X)
    Z_f = float(Z_new)
    rho_tgt = float(rho_target)

    for _ in range(max_iter):
        result = _eos_fn(logT_i, logPgas_f, X_f, Z_f)
        rho_e = float(result[0])
        residual = np.log(rho_e) - np.log(rho_tgt)

        if abs(residual) < tol:
            # Converged — compute Γ₁ at the adjusted T
            nad_e = float(result[2])
            cr_e = float(result[5])
            ct_e = float(result[6])
            gamma1_new = cr_e / max(1.0 - nad_e * ct_e, 1e-10)
            return gamma1_new, logT_i

        # Newton step: dln(ρ)/dlogT via FD (centered, small step)
        h = 1e-6
        rho_hi = float(_eos_fn(logT_i + h, logPgas_f, X_f, Z_f)[0])
        rho_lo = float(_eos_fn(logT_i - h, logPgas_f, X_f, Z_f)[0])
        dlnrho_dlogT = (np.log(rho_hi) - np.log(rho_lo)) / (2.0 * h)

        if abs(dlnrho_dlogT) < 1e-30:
            return None, None  # degenerate — shouldn't happen

        # logT correction: residual is in ln(ρ), need change in logT
        # ln(ρ) = ln(ρ)(logT) → dln(ρ) = dlnrho_dlogT · dlogT
        dlogT = -residual / dlnrho_dlogT
        logT_i += dlogT

    return None, None  # failed to converge


@pytest.mark.validation
@pytest.mark.mutation("freeze_composition_kernel")
@pytest.mark.integration
class TestCompositionKernel:
    """Validate the intrinsic-EOS composition kernel K_{Z,ρ}(r).

    WHAT: K_{Z,ρ}(r) computed via AD (jax.grad through the EOS → Γ₁ chain)
    must predict the actual frequency shift from a localized Z perturbation,
    validated against re-solved eigenfrequencies (variational consistency).
    The derivative is at fixed (P, ρ), matching Basu & CD 1997 Eq. 2 and
    MESA pulse_fgong.f90:371-375.

    WHY: proves the composition kernel captures the real Z → Γ₁ → ν sensitivity
    at the correct thermodynamic holding.

    EXTERNAL REFERENCE: Model S FGONG (Christensen-Dalsgaard et al. 1996).
    The FD recomputation holds (P, ρ) fixed by Newton-adjusting T at each zone.

    TOLERANCE: 5% relative agreement (variational, same as K_{Γ₁,ρ} test).

    MUTATION: freeze_composition_kernel zeros ∂Γ₁/∂Z via stop_gradient → K_{Z,ρ}
    vanishes → kernel integral predicts zero frequency shift → fails vs actual.
    """

    def test_composition_kernel_variational_radial(self, model_s_fgong):
        """K_{Z,ρ} integral predicts δν/ν from a (P,ρ)-fixed Z perturbation.

        WHAT: variational consistency of the composition kernel for l=0, n=20.
        WHY: validates that K_{Z,ρ}(r) correctly captures Z → Γ₁(P,ρ,X,Z) → ν.
        EXTERNAL REFERENCE: Model S + FD re-solved eigenfrequency with (P,ρ)-
            fixed δΓ₁ (Newton T-adjustment at each zone to keep ρ constant).
        TOLERANCE: 5% relative — linearization error for 5% δZ is O(2.5e-3).
        MUTATION: freeze_composition_kernel → K_{Z,ρ}=0 → integral=0 → fails.
        """
        from stellar_jax.oscillations.kernels import (
            compute_composition_kernels, composition_kernel_integral)
        from stellar_jax.oscillations.eigenvalue import compute_oscillation_freqs_jax

        glob, var = model_s_fgong
        R = glob[1]

        # Compute composition kernel for l=0, n=20
        ck = compute_composition_kernels(
            glob, var, l=0, n_pg=20,
            nu_min=2000.0, nu_max=3500.0,
            n_scan=400, n_steps=8000)

        nu_ref = ck['nu']
        x_kernel = ck['x']

        # Apply a 5% Gaussian Z perturbation centered at x/R = 0.65
        # (CZ base — where dlnGamma1/dZ is significant)
        x_center = 0.65
        width = 0.05
        amplitude = 0.05  # 5% of Z (~0.001 in absolute Z)

        # Z at each kernel grid point
        mask = (var[:, 0] / R) > 1e-4
        Z_profile = var[mask, 16] if var.shape[1] > 16 else np.full(len(x_kernel), float(glob[3]))

        delta_Z = amplitude * Z_profile * np.exp(
            -0.5 * ((x_kernel - x_center) / width) ** 2)

        # Kernel-predicted δν/ν
        dnu_over_nu_predicted = composition_kernel_integral(ck, delta_Z)

        # ─── Actual δν/ν: (P,ρ)-fixed perturbation ─────────────────────────
        # At each zone where δZ is non-negligible, perturb Z while adjusting T
        # to keep ρ constant at fixed P (matching Basu & CD 1997 Eq. 2).
        # This is the key difference from the prior (T,P)-fixed test.
        from stellar_jax.microphysics.eos import _eos_opal_table
        from stellar_jax.config.constants import a_rad
        from stellar_jax.config.physics_floors import PGAS_FRAC_FLOOR

        var_perturbed = var.copy()
        x_fgong = var[:, 0] / R
        delta_Z_full = amplitude * var[:, 16] * np.exp(
            -0.5 * ((x_fgong - x_center) / width) ** 2)

        # Only perturb zones where δZ is non-negligible (> 1e-6 × Z).
        # The Gaussian is effectively zero outside ±4σ of the center,
        # so this reduces the Newton loop from ~2500 to ~200 zones.
        sig_threshold = 1e-6 * np.max(var[:, 16])
        for i in range(len(var)):
            if abs(delta_Z_full[i]) < sig_threshold:
                continue  # no meaningful perturbation at this zone

            T_i = var[i, 2]
            P_total_i = var[i, 3]
            X_i = var[i, 5]
            Z_orig = var[i, 16]
            Z_new = Z_orig + delta_Z_full[i]

            logT_i = np.log10(max(T_i, 1.0))
            P_rad_i = a_rad * T_i**4 / 3.0
            P_gas_i = max(P_total_i - P_rad_i, PGAS_FRAC_FLOOR * P_total_i)
            logPgas_i = np.log10(max(P_gas_i, 1.0))

            # Original Gamma1 from our EOS at (T, P, X, Z_orig).
            # Use concrete Python floats to avoid JAX JIT overhead in the loop.
            result_orig = _eos_opal_table(
                float(logT_i), float(logPgas_i), float(X_i), float(Z_orig))
            g1_orig = float(result_orig[5]) / max(
                1.0 - float(result_orig[2]) * float(result_orig[6]), 1e-10)
            # Target ρ: what OUR EOS gives at (T, P, X, Z_orig) — NOT the FGONG ρ.
            # This ensures δΓ₁ measures only the Z-change, not the EOS-FGONG offset.
            rho_target = float(result_orig[0])

            # Γ₁ at new Z, T adjusted to keep ρ constant at fixed P_gas
            g1_new, _ = _gamma1_at_fixed_P_rho(
                float(logT_i), float(logPgas_i), float(X_i), float(Z_new),
                rho_target, _eos_opal_table, a_rad, PGAS_FRAC_FLOOR)

            if g1_new is None:
                continue  # Newton failed at this zone — skip

            # Perturb the FGONG Γ₁ by the (P,ρ)-fixed EOS difference
            var_perturbed[i, 9] = var[i, 9] + (g1_new - g1_orig)

        # Re-solve for frequencies with the perturbed Γ₁ profile
        freqs_perturbed = compute_oscillation_freqs_jax(
            glob, var_perturbed, l_values=(0,),
            nu_min=nu_ref - 50.0, nu_max=nu_ref + 50.0,
            n_scan=200, n_steps=8000)

        perturbed_freqs = freqs_perturbed[0]
        assert len(perturbed_freqs) > 0, "No modes found in perturbed model"
        idx = np.argmin(np.abs(perturbed_freqs - nu_ref))
        nu_perturbed = perturbed_freqs[idx]

        dnu_over_nu_actual = (nu_perturbed - nu_ref) / nu_ref

        # ─── Checks ────────────────────────────────────────────────────────
        assert dnu_over_nu_actual != 0.0, (
            "Perturbed frequency identical to reference — Z perturbation has no effect?")

        assert abs(dnu_over_nu_predicted) > 1e-10, (
            "Composition kernel integral is zero — K_{Z,ρ} was not computed "
            "(possible freeze_composition_kernel mutation?)")

        # Both should have the same sign
        assert np.sign(dnu_over_nu_predicted) == np.sign(dnu_over_nu_actual), (
            f"Sign mismatch: predicted {dnu_over_nu_predicted:.6e}, "
            f"actual {dnu_over_nu_actual:.6e}")

        rel_error = abs(dnu_over_nu_predicted - dnu_over_nu_actual) / abs(dnu_over_nu_actual)

        assert rel_error < 0.05, (
            f"Composition kernel variational consistency failed (l=0, n=20, "
            f"ν={nu_ref:.1f} µHz):\n"
            f"  Kernel-predicted δν/ν = {dnu_over_nu_predicted:.6e}\n"
            f"  Actual δν/ν (re-solved, P,ρ-fixed) = {dnu_over_nu_actual:.6e}\n"
            f"  Relative error = {rel_error:.2%} (tolerance: 5%)\n"
            f"  Perturbation: 5% Gaussian δZ at x/R=0.65, width=0.05")

    def test_composition_kernel_nonzero(self, model_s_fgong):
        """K_{Z,ρ}(r) is finite, non-trivial, and has physical structure.

        WHAT: basic sanity checks on the composition kernel shape and values.
        WHY: catches implementation bugs (NaN, all-zero, wrong sign) and
        verifies the EOS Z-sensitivity is resolved.
        EXTERNAL REFERENCE: Model S structure (Christensen-Dalsgaard et al. 1996).
        TOLERANCE: K_Z_rho max > 0.01 (non-trivial); integral physically reasonable.
        MUTATION: freeze_composition_kernel → K_{Z,ρ}=0 → fails non-trivial check.
        """
        from stellar_jax.oscillations.kernels import compute_composition_kernels

        glob, var = model_s_fgong

        ck = compute_composition_kernels(
            glob, var, l=0, n_pg=20,
            nu_min=2000.0, nu_max=3500.0,
            n_scan=400, n_steps=8000)

        K_Z = ck['K_Z_rho']
        K_G1 = ck['K_gamma1_rho']
        dlnG1_dZ = ck['dlnGamma1_dZ']
        dlnG1_dZ_TP = ck['dlnGamma1_dZ_TP']
        correction = ck['correction_term']

        # Finiteness
        assert np.all(np.isfinite(K_Z)), (
            f"K_Z_rho has {np.sum(~np.isfinite(K_Z))} non-finite values")

        # Non-trivial: the composition kernel must have structure
        max_K_Z = np.max(np.abs(K_Z))
        assert max_K_Z > 0.01, (
            f"Composition kernel is essentially zero: max(|K_Z|) = {max_K_Z:.4e}. "
            f"Possible freeze_composition_kernel mutation?")

        # dlnGamma1_dZ (at P,ρ) must be non-trivial
        max_dlnG1_dZ = np.max(np.abs(dlnG1_dZ))
        assert max_dlnG1_dZ > 0.001, (
            f"dlnGamma1_dZ is trivial: max = {max_dlnG1_dZ:.6e}")

        # K_{Z,ρ} = K_{Γ₁,ρ} · dlnΓ₁/dZ — verify this identity holds
        K_Z_expected = K_G1 * dlnG1_dZ
        np.testing.assert_allclose(K_Z, K_Z_expected, rtol=1e-12,
            err_msg="K_Z_rho != K_gamma1_rho * dlnGamma1_dZ (composition identity)")

        # The correction term should be non-negligible (first-order, not
        # second-order as the old docstring falsely claimed).
        # MESA pulse_fgong.f90:371-375 explicitly computes this term.
        max_correction = np.max(np.abs(correction))
        assert max_correction > 1e-4, (
            f"Thermodynamic correction is negligible: max = {max_correction:.6e}. "
            f"The (T,P)→(P,ρ) correction should be first-order.")

        # The corrected dlnG1/dZ should differ from the raw (T,P)-fixed value
        # by a measurable amount — this proves the correction is active.
        diff_TP_vs_Prho = np.max(np.abs(dlnG1_dZ - dlnG1_dZ_TP))
        assert diff_TP_vs_Prho > 1e-4, (
            f"(P,ρ)-fixed and (T,P)-fixed derivatives are identical: "
            f"max diff = {diff_TP_vs_Prho:.6e}. Correction not active?")

        # Physical check: the composition kernel should be smaller than K_{Γ₁,ρ}
        # because |dlnΓ₁/dZ| < 1 for solar conditions
        ratio = max_K_Z / np.max(np.abs(K_G1))
        assert ratio < 0.5, (
            f"K_Z_rho / K_gamma1_rho ratio = {ratio:.2f} — unexpectedly large. "
            f"dlnGamma1/dZ should be << 1 for solar Z")

        # The total integral ∫K_Z·dr/R (= δν/ν per unit uniform δZ)
        # should be order ~0.01 for solar conditions
        total = float(np.sum(K_Z * ck['dr_over_R']))
        assert abs(total) < 1.0, (
            f"∫K_Z·dr/R = {total:.4f} — unreasonably large")
        assert abs(total) > 1e-4, (
            f"∫K_Z·dr/R = {total:.4f} — unreasonably small")
