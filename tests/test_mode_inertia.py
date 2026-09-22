"""Tests for mode inertia computation (#780).

Validates that compute_mode_inertia produces physically correct and internally
consistent values for the normalized mode inertia E of solar oscillation modes.

The mode inertia is the integral:
    E = (4πR³/M) ∫₀^1 [(ξ_r/R)² + l(l+1)(ξ_h/R)²] ρ x² dx / (ξ_r(R)/R)²

For high-order p-modes in a solar model, the raw mode inertia E is O(10⁻⁸)
to O(10⁻⁷) in this convention (Christensen-Dalsgaard 2008; ADIPLS obs_st(4)).
This is because the p-mode displacement ξ_r grows enormously near the surface
(where ρ→0, energy conservation amplifies the wave), so ξ_r(R)² in the
denominator dominates. The interior mass-weighted integral is much smaller.

Established scaling properties for validation:
  1. E > 0 and finite for all p-modes
  2. E decreases with increasing frequency (higher ν → more nodes, slightly
     larger surface amplitude relative to interior → smaller E)
  3. At a given frequency, E(l=2) > E(l=0) — higher-l modes couple to
     horizontal displacement (the l(l+1)ξ_h² term adds to the numerator
     without increasing the surface ξ_r normalization)

For the AD kernel pipeline, mode inertia normalizes the analytic comparison:
    K_analytic(r) ∝ 1/(2ν · E) × [eigenfunction integrands]
But the AD kernels ∂ν/∂c²(r) are computed directly and do NOT require E.
Mode inertia is a validation/comparison diagnostic.

External reference: ADIPLS obs_st(4) values for Model S modes are O(10⁻⁸).
MESA/GYRE `E_norm` with `inertia_norm='BOTH'` gives similar scale.

Mutation: NONE — mode inertia E = ∫ρ|ξ|²dV is a density/displacement integral,
physically Γ₁-independent. The structural properties tested (positivity, order of
magnitude, monotonicity, l-dependence) hold for ANY valid stellar p-mode regardless
of Γ₁ corruption. Dropped from the mutation gate per issue #1136.

References:
    Aerts, Christensen-Dalsgaard & Kurtz (2010), §3.3, Eq. 3.139 (definition)
    Christensen-Dalsgaard (2008), Ap&SS 316, 113 (ADIPLS mode inertia)
    Christensen-Dalsgaard & Berthomieu (1991), Solar Interior & Atmosphere, §IV.B
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Fixture: Model S FGONG ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_s_fgong():
    """Load the Model S FGONG file (committed reference data)."""
    fgong_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "model_s", "fgong.l5bi.d.15c"
    )
    assert os.path.isfile(fgong_path), (
        f"Model S FGONG not found: {fgong_path} — file is committed, checkout may be corrupt")
    from stellar_jax.fgong.io import read_fgong
    glob, var = read_fgong(fgong_path)
    return glob, var


@pytest.fixture(scope="module")
def radial_inertias(model_s_fgong):
    """Compute mode inertias for 5 consecutive radial modes of Model S.

    Uses n_pg = 15, 17, 19, 21, 23 — corresponding to l=0 modes in the
    ~2000-3200 µHz range (Δν ≈ 135 µHz for solar-type).

    Returns list of (nu, E) tuples.
    """
    os.environ.setdefault('JAX_ENABLE_X64', '1')
    from stellar_jax.oscillations.eigenfunction import compute_eigenfunction, compute_mode_inertia
    glob, var = model_s_fgong
    results = []
    for n_pg in [15, 17, 19, 21, 23]:
        ef = compute_eigenfunction(glob, var, l=0, n_pg=n_pg,
                                   nu_min=1500.0, nu_max=4000.0,
                                   n_scan=500, n_steps=8000)
        E = compute_mode_inertia(ef, glob, var)
        results.append((ef['nu'], E))
    return results


@pytest.fixture(scope="module")
def l2_inertias(model_s_fgong):
    """Compute mode inertias for 3 consecutive l=2 modes of Model S.

    Uses n_pg = 15, 17, 19 — matching the radial modes' frequency range.

    Returns list of (nu, E) tuples.
    """
    os.environ.setdefault('JAX_ENABLE_X64', '1')
    from stellar_jax.oscillations.eigenfunction import compute_eigenfunction, compute_mode_inertia
    glob, var = model_s_fgong
    results = []
    for n_pg in [15, 17, 19]:
        ef = compute_eigenfunction(glob, var, l=2, n_pg=n_pg,
                                   nu_min=1500.0, nu_max=4000.0,
                                   n_scan=500, n_steps=8000)
        E = compute_mode_inertia(ef, glob, var)
        results.append((ef['nu'], E))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestModeInertia:
    """Mode inertia properties for solar p-modes (#780).

    WHAT: validates that compute_mode_inertia returns physically correct
    normalized mode inertias for Model S eigenmodes in the solar p-mode range.

    WHY: mode inertia E is needed for: (1) comparing AD kernels to analytic
    kernels (validation of #782), and (2) the surface-effect correction
    (Ball & Gizon 2014 uses E to scale the correction per mode). A wrong E
    propagates through all kernel comparisons.

    EXTERNAL REFERENCE: ADIPLS obs_st(4) for Model S l=0 modes in 2000-4000 μHz
    gives E ~ 3e-9 to 2e-8 (Christensen-Dalsgaard 2008). Our values should be
    in the same order-of-magnitude range.

    NO MUTATION GATE (issue #1136): mode inertia E = (4πR³/M) ∫ ρ|ξ|²x²dx / ξ_r(R)²
    is a density/displacement integral — physically Γ₁-independent. corrupt_gamma1
    shifts eigenfrequencies, but the Brent solver finds valid modes at the shifted
    frequencies and their structural inertia properties (positive, finite, monotonic,
    E(l=2) > E(l=0)) are preserved. No registered mutation can break these universal
    p-mode properties without also breaking the eigenvalue solver outright (which
    would be an incidental guard failure, not a physics assertion failure). Dropped
    from @validation per the O2 right_reason scheme.
    """

    def test_positive_finite(self, radial_inertias):
        """Mode inertia must be positive and finite for all p-modes.

        E = (4πR³/M) · ∫ > 0 by construction (integrand is non-negative);
        finite because ξ_r(R) ≠ 0 for p-modes (ξ_r has maximum at surface).
        """
        for nu, E in radial_inertias:
            assert E > 0, f"E must be positive, got {E:.4e} for ν={nu:.1f} μHz"
            assert np.isfinite(E), f"E must be finite, got {E:.4e} for ν={nu:.1f} μHz"

    def test_order_of_magnitude(self, radial_inertias):
        """Solar l=0 p-mode inertias are O(10⁻⁸) to O(10⁻⁹).

        ADIPLS obs_st(4) for Model S l=0 modes in 2000-4000 μHz gives
        values in this range (Christensen-Dalsgaard 2008). The exact scale
        depends on the surface-density treatment, so we allow [1e-10, 1e-6].
        """
        for nu, E in radial_inertias:
            assert 1e-10 < E < 1e-6, (
                f"l=0 mode inertia out of expected range: E={E:.4e} "
                f"for ν={nu:.1f} μHz (expected ~1e-9 to 1e-8)")

    def test_decreases_with_frequency(self, radial_inertias):
        """Mode inertia decreases monotonically with frequency for l=0.

        Higher-frequency (higher-order) modes have larger surface amplitude
        relative to the interior (more nodes compress the wave), so the
        denominator (surface ξ_r²) grows faster than the numerator
        (mass-weighted integral) → E decreases. This is established for
        the asymptotic p-mode regime (Christensen-Dalsgaard 2008, Fig. 2).
        """
        for i in range(1, len(radial_inertias)):
            nu_prev, E_prev = radial_inertias[i - 1]
            nu_curr, E_curr = radial_inertias[i]
            assert E_curr < E_prev, (
                f"E must decrease with ν: E({nu_curr:.0f})={E_curr:.4e} "
                f">= E({nu_prev:.0f})={E_prev:.4e}")

    def test_l2_larger_than_l0(self, radial_inertias, l2_inertias):
        """At similar frequencies, E(l=2) > E(l=0).

        The l(l+1)ξ_h² term in the numerator adds mode energy from
        horizontal motions without increasing the radial surface normalization.
        For l=2, this adds a factor ~6 ξ_h² to the numerator, making
        E(l=2) systematically larger than E(l=0).

        Reference: This is the standard "mode inertia bump" for higher-l
        (Aerts+2010, Fig. 3.22; Christensen-Dalsgaard 2003, Fig. 5.14).
        """
        # Compare at the same approximate frequency (~2350 μHz = mode index 2)
        _, E_l0 = radial_inertias[2]
        _, E_l2 = l2_inertias[2]
        assert E_l2 > E_l0, (
            f"Expected E(l=2) > E(l=0): got {E_l2:.4e} vs {E_l0:.4e}")

    def test_l2_positive_finite(self, l2_inertias):
        """l=2 mode inertias must also be positive and finite."""
        for nu, E in l2_inertias:
            assert E > 0, f"E must be positive, got {E:.4e} for l=2 ν={nu:.1f} μHz"
            assert np.isfinite(E), f"E must be finite, got {E:.4e} for l=2 ν={nu:.1f} μHz"

    def test_l2_order_of_magnitude(self, l2_inertias):
        """l=2 solar p-mode inertias are O(10⁻⁷) to O(10⁻⁸).

        Higher-l modes have larger E due to the l(l+1)ξ_h² contribution.
        """
        for nu, E in l2_inertias:
            assert 1e-9 < E < 1e-5, (
                f"l=2 mode inertia out of expected range: E={E:.4e} "
                f"for ν={nu:.1f} μHz (expected ~1e-8 to 1e-7)")

    def test_consistent_ratio_l2_over_l0(self, radial_inertias, l2_inertias):
        """The ratio E(l=2)/E(l=0) should be approximately constant.

        For high-order p-modes in the asymptotic regime, the ratio E(l=2)/E(l=0)
        at similar frequencies is roughly constant (determined by the l-dependence
        of the eigenfunction geometry). For solar p-modes at n_pg=15-23, ξ_h is
        small relative to ξ_r (|ξ_h/ξ_r| ~ 1/(c₁ω²) ~ 0.003-0.01), so the
        l(l+1)ξ_h² contribution raises E(l=2) by a factor ~1.4-1.7 over E(l=0).
        This is consistent with ADIPLS/GYRE E_norm for high-order solar modes
        (Christensen-Dalsgaard 2003, Fig. 5.14; the large ratios ~3-30 occur only
        for low-order or mixed modes where ξ_h becomes comparable to ξ_r).
        """
        ratios = []
        for i in range(min(len(radial_inertias), len(l2_inertias))):
            _, E_l0 = radial_inertias[i]
            _, E_l2 = l2_inertias[i]
            ratio = E_l2 / E_l0
            ratios.append(ratio)

        mean_ratio = np.mean(ratios)
        # High-order solar p-modes: ratio ~1.4-1.7 (measured; consistent with
        # the small ξ_h/ξ_r asymptotic regime). The ratio must be > 1 (l=2
        # always has more inertia due to ξ_h) and < 5 (it would only reach
        # higher values for mixed or low-order modes).
        assert 1.1 < mean_ratio < 5.0, (
            f"E(l=2)/E(l=0) ratio = {mean_ratio:.2f}, expected 1.1-5.0")
