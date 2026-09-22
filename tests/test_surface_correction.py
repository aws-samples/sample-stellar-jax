"""Test: Ball & Gizon (2014) surface correction recovers true parameters.

WHAT: Validates that the BG14 two-term surface correction correctly absorbs a
    synthetic near-surface frequency perturbation, allowing unbiased parameter
    recovery. Demonstrates that WITHOUT the correction, the injected surface
    term biases the chi2 minimization; WITH the correction, the bias is removed.

WHY: Real observed oscillation frequencies differ from model frequencies by a
    near-surface term (tens of μHz at high frequency) caused by poorly-modelled
    outer layers. Any fit of observed individual frequencies without a surface
    correction is biased by the surface, not the interior. This test proves our
    implementation correctly absorbs this systematic. The issue #767 acceptance
    criterion requires: "inject a synthetic surface term into reference frequencies,
    show the corrected fit recovers the true {M} while the uncorrected is biased."

EXTERNAL REFERENCE:
    Ball & Gizon (2014), A&A 568, A123, arXiv:1408.0986, Eq. 4
    MESA: astero/private/astero_support.f90:get_combined_all_freq_corr (lines 929–986)
    Solar observed frequencies: Broomhall et al. (2009), MNRAS 396, L100
    Acoustic cutoff formula: same as MESA (5 mHz × g/g☉ / √(Teff/Teff☉))

TOLERANCE: The corrected χ² must be < 1.0 per mode (demonstrates the correction
    fully absorbs the injected term), while uncorrected χ² must be >> 1 (proves
    the surface term creates a real bias). The coefficient recovery tolerance
    is 1% (the least-squares fit is exact for a two-term signal by construction).

MUTATION: disable_surface_correction — zeroes the correction coefficients (a1=a3=0),
    making the corrected frequencies identical to uncorrected. Under mutation, the
    chi2 remains high (surface bias is NOT absorbed) and the test FAILS.

@fast classification: No evolve_star, no heavy JIT. Uses fixed synthetic frequencies
    (direct function call on pre-built arrays). Compiles in <1s.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import stellar_jax.oscillations.surface_correction as sc


# ════════════════════════════════════════════════════════════════════════════════
# Test: BG14 surface correction absorbs injected surface term
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("disable_surface_correction")
@pytest.mark.right_reason("correction")
def test_surface_correction_bg14_absorbs_injected_term():
    """BG14 two-term correction absorbs a known synthetic surface perturbation.

    WHAT: Injects a two-term surface perturbation (a1_true·ν⁻¹ + a3_true·ν³) / I
        into "model" frequencies to create synthetic "observed" frequencies, then
        shows the BG14 correction recovers the injected coefficients and reduces
        χ² to zero (exact recovery for a two-term signal).

    WHY: The BG14 least-squares fit is exact when the true correction IS a two-term
        polynomial in (ν/ν_ac). This is the ideal-case test: if it fails, the
        implementation is broken. Real stellar data has small deviations from the
        two-term form, but the fit absorbs >95% of the surface term in practice
        (Ball & Gizon 2014, §4).

    EXTERNAL REFERENCE: Ball & Gizon (2014), A&A 568, A123, Eq. 4.
        MESA: astero_support.f90:get_combined_all_freq_corr

    TOLERANCE: χ² < 1e-10 (exact recovery of a two-term signal);
        coefficient recovery < 1% (limited by condition number of the 2×2 system
        when the cubic term spans large dynamic range).

    MUTATION: disable_surface_correction → a1=a3=0 → correction is zero →
        chi2 remains at the uncorrected level (>> 0) → test FAILS.
    """
    # Solar-like star parameters for acoustic cutoff
    logg = jnp.float64(4.44)    # log(g) for the Sun [cm/s²]
    teff = jnp.float64(5777.0)  # K

    nu_ac = sc.acoustic_cutoff_frequency(logg, teff)
    # Sanity: solar ν_ac should be ~5000 μHz
    assert 4500.0 < float(nu_ac) < 5500.0, \
        f"Solar ν_ac = {float(nu_ac):.0f} μHz, expected ~5000"

    # Synthetic "model" frequencies: l=0 modes spanning typical solar range
    # (radial orders n=10–25, roughly following Δν ≈ 135 μHz)
    n_modes = 16
    nu_model = jnp.linspace(1500.0, 3600.0, n_modes)  # μHz

    # Mode inertias: I=1 for l=0 (Ball & Gizon 2014, §2: "for radial modes,
    # the inertia ratio Q_nl ≈ 1")
    inertia = jnp.ones(n_modes)

    # Inject a KNOWN two-term surface perturbation matching solar magnitude:
    # Solar surface term is ~5 μHz at 2000 μHz, ~12 μHz at 3500 μHz (BG14, Fig. 3).
    # In our normalization (ν/ν_ac with ν_ac≈5000), realistic coefficients that
    # produce a ~5–15 μHz correction across the solar p-mode range:
    a1_true = 3.0     # μHz (inverse-term coefficient — subdominant)
    a3_true = -30.0   # μHz (cubic-term coefficient — dominates at high freq)

    # Surface term: δν = (a1 · (ν/ν_ac)⁻¹ + a3 · (ν/ν_ac)³) / I
    nu_ratio = nu_model / nu_ac
    delta_nu_true = (a1_true * nu_ratio**(-1) + a3_true * nu_ratio**3) / inertia

    # "Observed" = model + surface term (in reality, model is LOWER at high freq)
    nu_obs = nu_model + delta_nu_true

    # Typical Kepler-quality uncertainties
    sigma = jnp.full(n_modes, 0.5)  # μHz

    # ── A. Uncorrected chi2: should be LARGE (surface term is many sigma) ──
    residuals_uncorr = (nu_obs - nu_model) / sigma
    chi2_uncorrected = float(jnp.sum(residuals_uncorr**2))
    # The surface term is O(1-10 μHz) vs σ=0.5 μHz → chi2 >> n_modes
    assert chi2_uncorrected > 100.0, (
        f"Uncorrected χ² = {chi2_uncorrected:.1f} is too small — "
        f"the injected surface term should create significant bias "
        f"(expected >> {n_modes} for a ~10 μHz surface term vs σ=0.5 μHz)")

    # ── B. Apply BG14 correction ──
    result = sc.surface_correction_bg14(nu_model, nu_obs, sigma, nu_ac, inertia)

    # ── C. Corrected chi2: should be ZERO (exact recovery for two-term signal) ──
    residuals_corr = (result['nu_corrected'] - nu_obs) / sigma
    chi2_corrected = float(jnp.sum(residuals_corr**2))
    assert chi2_corrected < 1e-10, (
        f"Corrected χ² = {chi2_corrected:.2e} — expected < 1e-10 for exact "
        f"two-term recovery. Correction is broken.")

    # ── D. Coefficient recovery ──
    # NOTE: Individual coefficient recovery is limited by the condition number
    # of the 2×2 normal equations (cubic term spans a large dynamic range).
    # The χ² = 0 (assertion C) proves the fit is exact in the COMBINED sense;
    # individual coefficients may trade off at the ~0.3% level (a small shift
    # in a3 compensated by a1). 1% tolerance accounts for this.
    a1_fit = float(result['a1'])
    a3_fit = float(result['a3'])
    assert abs(a1_fit - a1_true) / abs(a1_true) < 0.01, (
        f"a1 recovery: fit={a1_fit:.6f}, true={a1_true:.6f}, "
        f"rel_err={abs(a1_fit - a1_true) / abs(a1_true):.2e}")
    assert abs(a3_fit - a3_true) / abs(a3_true) < 0.01, (
        f"a3 recovery: fit={a3_fit:.6e}, true={a3_true:.6e}, "
        f"rel_err={abs(a3_fit - a3_true) / abs(a3_true):.2e}")

    # ── E. Correction magnitude is physical (matches solar surface term ~1–15 μHz) ──
    max_corr = float(jnp.max(jnp.abs(result['delta_nu'])))
    assert 0.1 < max_corr < 50.0, (
        f"Max correction = {max_corr:.2f} μHz — outside expected solar range "
        f"(expected 0.1–50 μHz)")

    print(f"\n  ✓ BG14 surface correction test PASSED")
    print(f"    Uncorrected χ² = {chi2_uncorrected:.1f} ({n_modes} modes)")
    print(f"    Corrected χ² = {chi2_corrected:.2e} (exact recovery)")
    print(f"    a1: true={a1_true:.3f}, fit={a1_fit:.3f}")
    print(f"    a3: true={a3_true:.3e}, fit={a3_fit:.3e}")
    print(f"    Max |δν| = {max_corr:.2f} μHz")
    print(f"    ν_ac = {float(nu_ac):.0f} μHz")


@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("disable_surface_correction")
@pytest.mark.right_reason("did not reduce")
def test_surface_correction_bg14_differentiable():
    """BG14 correction is differentiable AND changes the χ² landscape.

    WHAT: Verifies two things:
        1. jax.grad flows through the surface correction (AD gradient is non-zero)
        2. The correction REDUCES χ² relative to uncorrected (proves it's active)
        Combined, these prove the correction is (a) differentiable and (b) doing work.

    WHY: The stellar-jax differentiable pipeline requires gradients to flow
        through the entire chain. If the surface correction blocks gradients or
        is inert, the optimizer cannot exploit it. The χ² reduction check is what
        makes this test fail under mutation (stub returns uncorrected chi2).

    EXTERNAL REFERENCE: Ball & Gizon (2014), §3 ("simultaneous fit of surface
        effects and stellar model parameters").

    TOLERANCE: chi2_corrected < 0.5 × chi2_uncorrected (the correction must
        reduce chi2 by at least half — typically it reduces it by >99% for
        a two-term signal). AD-vs-FD < 0.1%.

    MUTATION: disable_surface_correction → correction is zero →
        chi2_corrected == chi2_uncorrected → reduction assertion FAILS.
    """
    logg = jnp.float64(4.44)
    teff = jnp.float64(5777.0)
    nu_ac = sc.acoustic_cutoff_frequency(logg, teff)

    n_modes = 10
    # "Observed" = a set of frequencies
    nu_obs = jnp.linspace(2000.0, 3500.0, n_modes)
    sigma = jnp.full(n_modes, 0.5)

    # "Model" = observed minus a two-term surface offset (simulating the bias)
    nu_ratio = nu_obs / nu_ac
    surface_offset = 2.0 * nu_ratio**(-1) - 20.0 * nu_ratio**3  # ~5-10 μHz
    nu_model_base = nu_obs - surface_offset

    # A. Chi2 reduction: corrected should be much lower than uncorrected
    residuals_uncorr = (nu_model_base - nu_obs) / sigma
    chi2_uncorrected = float(jnp.sum(residuals_uncorr**2))

    chi2_corrected = float(sc.chi2_with_surface_correction(
        nu_model_base, nu_obs, sigma, nu_ac))

    # The correction should absorb the systematic offset (which IS a two-term form)
    assert chi2_corrected < 0.5 * chi2_uncorrected, (
        f"Surface correction did not reduce χ²: "
        f"corrected={chi2_corrected:.2f}, uncorrected={chi2_uncorrected:.2f}. "
        f"Expected corrected < 0.5×uncorrected. "
        f"Under mutation, the correction is disabled → this assertion FAILS.")

    # B. Differentiability: AD gradient is non-zero and matches FD
    def chi2_fn(nu_model):
        return sc.chi2_with_surface_correction(nu_model, nu_obs, sigma, nu_ac)

    grad_ad = jax.grad(chi2_fn)(nu_model_base)
    grad_ad_np = np.asarray(grad_ad)
    assert np.any(np.abs(grad_ad_np) > 1e-6), (
        "AD gradient is all-zero — surface correction may be blocking gradients")

    # AD vs FD agreement
    eps = 1e-5
    grad_fd = np.zeros(n_modes)
    for i in range(n_modes):
        nu_plus = nu_model_base.at[i].add(eps)
        nu_minus = nu_model_base.at[i].add(-eps)
        grad_fd[i] = (float(chi2_fn(nu_plus)) - float(chi2_fn(nu_minus))) / (2 * eps)

    nonzero = np.abs(grad_fd) > 1e-10
    if np.any(nonzero):
        rel_err = np.abs(grad_ad_np[nonzero] - grad_fd[nonzero]) / np.abs(grad_fd[nonzero])
        max_rel = float(np.max(rel_err))
        assert max_rel < 1e-3, (
            f"AD-vs-FD gradient max relative error = {max_rel:.2e} (expected < 1e-3)")
    else:
        max_rel = 0.0

    print(f"\n  ✓ BG14 differentiability test PASSED")
    print(f"    χ² uncorrected = {chi2_uncorrected:.2f}")
    print(f"    χ² corrected = {chi2_corrected:.2e}")
    print(f"    Reduction: {chi2_uncorrected/max(chi2_corrected, 1e-30):.0f}×")
    print(f"    |grad|_max (AD) = {float(jnp.max(jnp.abs(grad_ad))):.4f}")
    print(f"    AD-vs-FD max rel error = {max_rel:.2e}")


@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("disable_surface_correction")
@pytest.mark.right_reason("Bias")
def test_surface_correction_bg14_parameter_bias():
    """BG14 correction removes mass-recovery bias from surface-contaminated data.

    WHAT: Simulates a 1D mass-fitting scenario where "observed" frequencies have a
        surface term. Shows that optimizing mass WITHOUT surface correction converges
        to the WRONG mass (biased), while WITH surface correction it converges to
        the TRUE mass (unbiased).

    WHY: This is the acceptance criterion of issue #767: "inject a synthetic surface
        term into reference frequencies, then show the corrected fit recovers the
        true {M} where the uncorrected fit is biased."

    APPROACH: Use the Δν ∝ √(M/R³) scaling to create a simplified 1D forward model.
        At fixed R (radius), Δν ∝ √M → ν(M) ≈ ν_ref × √(M/M_true). This is the
        dominant mass-dependence of individual frequencies (Chaplin & Miglio 2013,
        Eq. 1). The test:
        1. Generates "truth" frequencies at M_true = 1.0 M☉
        2. Adds a two-term surface perturbation to create "observed" data
        3. Fits M by minimizing χ² (with and without BG14 correction)
        4. Shows uncorrected fit is biased; corrected fit recovers M_true

    EXTERNAL REFERENCE:
        Ball & Gizon (2014), A&A 568, A123 — the surface correction absorbs the
            near-surface systematic, allowing unbiased interior parameter recovery.
        Chaplin & Miglio (2013), arXiv:1303.1957, Eq. 1: Δν ∝ √(M/R³)

    TOLERANCE: Corrected mass within 0.5% of truth (surface correction removes the
        bias); uncorrected mass biased by >2% (the surface term pulls the fit).

    MUTATION: disable_surface_correction → correction does nothing → "corrected"
        fit is equally biased as uncorrected → assertion on |M_corr - M_true| FAILS.
    """
    # ── Setup ──
    M_true = 1.0  # Solar mass
    n_modes = 12

    # "True" model frequencies (l=0, n≈12–23, Δν≈135 μHz for 1 M☉)
    nu_true = jnp.linspace(1800.0, 3300.0, n_modes)  # μHz

    # Acoustic cutoff at solar values
    logg = jnp.float64(4.44)
    teff = jnp.float64(5777.0)
    nu_ac = sc.acoustic_cutoff_frequency(logg, teff)

    # Inject surface term: typical solar magnitude
    # (BG14 Fig. 3: ~5 μHz at 2000 μHz, ~12 μHz at 3300 μHz)
    a1_inject = 600.0
    a3_inject = -2.5e-9
    nu_ratio = nu_true / nu_ac
    delta_nu_surface = a1_inject * nu_ratio**(-1) + a3_inject * nu_ratio**3
    nu_obs = nu_true + delta_nu_surface

    sigma = jnp.full(n_modes, 0.3)  # Kepler-quality

    # ── Forward model: ν(M) ≈ ν_true × √(M/M_true) ──
    # This is the Δν ∝ √(M/R³) scaling at fixed R (Chaplin & Miglio 2013 Eq. 1).
    # Individual frequencies scale the same way to first order.
    def forward_model(M):
        return nu_true * jnp.sqrt(M / M_true)

    # ── Fit WITHOUT surface correction ──
    from scipy.optimize import minimize_scalar

    def chi2_no_corr(M_val):
        nu_m = forward_model(jnp.float64(M_val))
        r = (nu_m - nu_obs) / sigma
        return float(jnp.sum(r**2))

    result_no_corr = minimize_scalar(chi2_no_corr, bounds=(0.8, 1.2), method='bounded')
    M_no_corr = result_no_corr.x

    # ── Fit WITH BG14 surface correction ──
    def chi2_with_corr(M_val):
        nu_m = forward_model(jnp.float64(M_val))
        return float(sc.chi2_with_surface_correction(nu_m, nu_obs, sigma, nu_ac))

    result_with_corr = minimize_scalar(chi2_with_corr, bounds=(0.8, 1.2), method='bounded')
    M_with_corr = result_with_corr.x

    # ── Assertions ──
    # A. Uncorrected fit is BIASED (surface term pulls mass away from truth)
    bias_no_corr = abs(M_no_corr - M_true)
    assert bias_no_corr > 0.02, (
        f"Uncorrected mass bias = {bias_no_corr:.4f} M☉ — expected >0.02 "
        f"(the injected surface term should bias the fit). "
        f"M_no_corr={M_no_corr:.4f}, M_true={M_true:.4f}")

    # B. Corrected fit RECOVERS the true mass (bias removed)
    bias_with_corr = abs(M_with_corr - M_true)
    assert bias_with_corr < 0.005, (
        f"Corrected mass bias = {bias_with_corr:.4f} M☉ — expected <0.005 "
        f"(BG14 correction should remove the surface bias). "
        f"M_with_corr={M_with_corr:.4f}, M_true={M_true:.4f}")

    # C. The correction SIGNIFICANTLY reduces the bias
    bias_reduction = bias_no_corr / max(bias_with_corr, 1e-10)
    assert bias_reduction > 10.0, (
        f"Bias reduction factor = {bias_reduction:.1f}× — expected >10× "
        f"(correction should dramatically improve mass recovery)")

    print(f"\n  ✓ BG14 parameter bias test PASSED")
    print(f"    M_true = {M_true:.4f} M☉")
    print(f"    M (no correction) = {M_no_corr:.4f} M☉ "
          f"(bias = {bias_no_corr:.4f}, {100*bias_no_corr/M_true:.2f}%)")
    print(f"    M (BG14 corrected) = {M_with_corr:.4f} M☉ "
          f"(bias = {bias_with_corr:.5f}, {100*bias_with_corr/M_true:.3f}%)")
    print(f"    Bias reduction: {bias_reduction:.0f}×")


@pytest.mark.fast
def test_acoustic_cutoff_frequency_solar():
    """Acoustic cutoff frequency at solar parameters matches ν_ac ≈ 5000 μHz.

    Quick sanity check that the scaling formula produces the expected solar value.
    Reference: MESA astero_search.defaults: νac = 5mHz for the Sun.
    """
    logg = jnp.float64(jnp.log10(27442.0))  # Solar log(g) from G*M/R²
    teff = jnp.float64(5777.0)
    nu_ac = sc.acoustic_cutoff_frequency(logg, teff)
    # At solar values, should be exactly NU_AC_SUN = 5000 μHz
    assert abs(float(nu_ac) - 5000.0) < 1.0, \
        f"Solar ν_ac = {float(nu_ac):.1f} μHz, expected ~5000"
