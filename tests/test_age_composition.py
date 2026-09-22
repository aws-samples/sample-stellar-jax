"""Validation test for the ∂age/∂physics age-systematic budget (issue #1084).

Tests the age-bias attribution via the X_c composition clock: how much does
inferred age shift per unit change in a physics parameter? This is the
PLATO hero deliverable — the physics-systematic age floor that grids/MCMC
cannot produce.

HONESTY DISCLAIMER: This is an INFERENCE SENSITIVITY (age-bias attribution),
NOT a forward ∂age/∂θ. It measures how much a physics perturbation biases
inferred age when the observer fits to X_c-parameterized tracks. Validated
at the near-ZAMS operating point (N=3, 1.0 Msun, MODE-A: alpha_mlt=2.0,
Z=0.014) — a method demonstration. No hardcoded f_age proxy anywhere on
this path.

EXTERNAL REFERENCES:
- Cunha et al. (2021), arXiv:2110.03332 — PLATO H&H: largest age bias 8.66%
  from unaccounted gravitational settling.
- Issue #1222 (GREENLIT): ∂σ²/∂X_c = −0.599, rel_err = 1.7e-9.
- Validated ∂ν/∂{opacity_factor, eps_nuc_factor, alpha_mlt} in
  test_oscillations.py (N=3, dt=1e7, MODE-A).

CLASSIFICATION: @integration — the full chain requires evolve_star + IFT
eigenfrequency compilation (multi-minute JIT). MUST NOT be @fast.

MUTATION: opacity_factor_detach — severs the opacity_factor → kappa gradient
path. Under this mutation, ∂ν/∂opacity_factor = 0 → δage/δ(opacity_factor) = 0
while δage/δ(eps_nuc_factor) and δage/δ(alpha_mlt) remain nonzero (their
gradient paths are unaffected). The test asserts that opacity_factor's age
sensitivity drops below a small threshold.

MEMORY OPTIMIZATION: The OLS numerator Σ_i w_i · ∂ν_i/∂θ is computed via a
single weighted-sum backward pass per parameter (3 total), NOT per-mode
backward passes (which would be N_modes × 3 = ~15-30 passes, each ~30+ GB,
causing OOM on ci-mega 120 GB).
"""

import pytest
import numpy as np

from stellar_jax.config.mesa_config import MESA_CONFIG


@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("opacity_factor_detach")
@pytest.mark.right_reason("opacity_factor age sensitivity")
def test_age_bias_attribution(stellar):
    """Validate ∂age/∂physics age-systematic budget via X_c composition.

    WHAT: Computes δage/δθ for θ ∈ {opacity_factor, eps_nuc_factor, alpha_mlt}
    via the OLS projection of ∂ν/∂θ onto ∂ν/∂X_c, composed with dage/dX_c.
    The OLS numerator Σ(w_i · ∂ν_i/∂θ) is computed in a single backward pass
    per parameter (w_i = ∂ν_i/∂X_c from the cheap fixed-structure path).

    WHY: This is the near-term PLATO hero — the physics-systematic age budget
    that nobody else can produce analytically. Grids/MCMC give statistical σ(age);
    this gives the physics-systematic floor that dominates the ~10% budget
    (Cunha 2021: largest bias 8.66% from unaccounted diffusion).

    EXTERNAL REFERENCE: Composition formula grounded in:
    - #1222: ∂σ²/∂X_c validated to machine precision (rel_err = 1.7e-9).
    - test_seismic_gradient_opacity_factor: ∂σ²/∂opacity_factor validated < 5%.
    - test_seismic_gradient_eps_nuc_factor: ∂σ²/∂eps_nuc_factor validated < 5%.
    - test_seismic_gradient_alpha_mlt: ∂ν/∂alpha_mlt validated (CI-gated).
    - dage/dX_c from the evolution track: age(X_c) monotonic during MS
      (MESA history.f90:1397,2973).
    The composed result is an INFERENCE sensitivity — it answers "if opacity is
    1% wrong, how many Myr is age biased?" — not directly comparable to Cunha's
    grid-recovery biases, which include fitting degeneracies we do not model.

    TOLERANCE: Each parameter must produce a nonzero age sensitivity, and the
    composition ingredients (∂ν/∂X_c, OLS numerator, dage/dX_c) must all be
    finite and nonzero. No hardcoded numerical tolerance on the composed value —
    the value is REPORTED (printed), not asserted against a specific number,
    because this is a method demonstration at near-ZAMS. The assertion is
    LIVENESS + SIGN + MUTATION-SENSITIVITY.

    WHAT MAKES IT FAIL: @mutation opacity_factor_detach severs the
    opacity_factor → kappa gradient → ∂ν/∂opacity_factor ≈ 0 → the OLS
    numerator for opacity_factor ≈ 0 → the assertion that
    |δage/δ(opacity_factor)| > threshold fails.
    """
    from stellar_jax.inference.age_composition import compute_age_bias_attribution

    # Source physics from MESA_CONFIG (MODE-A canonical inlist) — no hardcoded
    # alpha_mlt/Z at call site. Defaults in compute_age_bias_attribution also
    # source from MESA_CONFIG.
    result = compute_age_bias_attribution(
        physics_params=('opacity_factor', 'eps_nuc_factor', 'alpha_mlt'),
        mass=1.0,
        Z=MESA_CONFIG['Z'],
        alpha_mlt=MESA_CONFIG['alpha_mlt'],
        Y_init=MESA_CONFIG['Y_init'],
        f_ov=MESA_CONFIG['f_ov'],
        n_steps=3, dt_fixed=1e7,
        l_value=0, nu_min=2000.0, nu_max=4500.0,
        n_scan=250, n_steps_osc=8000,
    )

    # ── Print the achieved values (required by acceptance criteria) ──
    print("\n" + "=" * 70)
    print("∂age/∂physics — age-systematic budget via X_c clock")
    print("=" * 70)
    print(f"HONESTY: {result['honesty']}")
    print(f"\nOperating point: age = {result['age_ref']:.2e} yr, "
          f"X_c = {result['Xc_ref']:.6f}")
    print(f"Modes: {len(result['modes'])} l=0 modes, "
          f"ν = {result['nu_ref']} μHz")
    print(f"dage/dX_c = {result['dage_dXc']:.4e} yr")
    print(f"‖∂ν/∂X_c‖² = {result['norm_sq']:.6e}")
    print(f"\n{'Parameter':<20} {'δX_c/δθ':>14} {'δage/δθ [yr]':>14} "
          f"{'δage/age [%]':>14}")
    print("-" * 70)
    for pname in ('opacity_factor', 'eps_nuc_factor', 'alpha_mlt'):
        print(f"{pname:<20} {result['dXc_dtheta'][pname]:>14.6e} "
              f"{result['dage_dtheta'][pname]:>14.4e} "
              f"{result['dage_dtheta_pct'][pname]:>13.4f}%")
    print("-" * 70)

    # ── Assertion 1: Ingredients are finite and nonzero ──
    assert np.all(np.isfinite(result['dnu_dXc'])), (
        f"∂ν/∂X_c contains non-finite values: {result['dnu_dXc']}")
    assert np.any(np.abs(result['dnu_dXc']) > 1e-10), (
        f"∂ν/∂X_c is effectively zero: {result['dnu_dXc']}")

    assert np.isfinite(result['dage_dXc']), (
        f"dage/dX_c is non-finite: {result['dage_dXc']}")
    assert abs(result['dage_dXc']) > 1.0, (
        f"dage/dX_c is suspiciously small: {result['dage_dXc']}")

    # ── Assertion 2: Sign checks (physics-grounded) ──
    # dage/dX_c < 0: X_c decreases as star ages (H burns to He)
    assert result['dage_dXc'] < 0, (
        f"dage/dX_c should be negative (X_c decreases with age), "
        f"got {result['dage_dXc']}")

    # ── Assertion 3: OLS numerator is finite and each parameter's composed ──
    #    age sensitivity is nonzero. (The OLS numerator replaces per-mode
    #    ∂ν/∂θ checks — the weighted sum Σ w_i · ∂ν_i/∂θ is computed in a
    #    single backward pass for memory efficiency.)
    for pname in ('opacity_factor', 'eps_nuc_factor', 'alpha_mlt'):
        ols_val = result['ols_numerator'][pname]
        assert np.isfinite(ols_val), (
            f"OLS numerator for {pname} is non-finite: {ols_val}")

        # Each parameter's COMPOSED age sensitivity must be nonzero (liveness).
        # Threshold: 1 yr — generous floor, catches a broken composition path.
        # The mutation-sensitive assertion (Assertion 4) uses a tighter threshold
        # for opacity_factor specifically.
        age_sens = abs(result['dage_dtheta'][pname])
        assert age_sens > 1.0, (
            f"{pname} composed age sensitivity is too small: "
            f"|δage/δ({pname})| = {age_sens:.2e} yr < 1 yr. "
            f"The gradient path for {pname} may be broken.")

    # ── Assertion 4: MUTATION-SENSITIVE — opacity_factor age sensitivity ──
    # Under opacity_factor_detach: ∂ν/∂opacity_factor ≈ 0 → δage/δ(opacity_factor) ≈ 0
    # Clean: the opacity_factor age sensitivity must be nonzero.
    # Threshold: 1e3 yr — generous, but catches the zero-gradient mutation.
    # At near-ZAMS (30 Myr), even a small opacity perturbation shifts
    # age by ~1e4-1e6 yr (from the OLS composition).
    opf_age_sens = abs(result['dage_dtheta']['opacity_factor'])
    assert opf_age_sens > 1e3, (
        f"opacity_factor age sensitivity is too small: "
        f"|δage/δ(opacity_factor)| = {opf_age_sens:.2e} yr < 1e3 yr. "
        f"The ∂ν/∂opacity_factor gradient path may be severed.")

    # ── Assertion 5: At least 2 modes found (OLS projection needs ≥2 data points) ──
    assert len(result['modes']) >= 2, (
        f"Expected ≥2 modes for OLS projection, found {len(result['modes'])}")

    print("\n✓ All assertions passed.")
    print("  - Ingredients (∂ν/∂X_c, OLS numerator, dage/dX_c) all finite and nonzero")
    print("  - dage/dX_c < 0 (physics-correct sign)")
    print("  - Each parameter produces a nonzero age sensitivity (|δage/δθ| > 1 yr)")
    print(f"  - opacity_factor sensitivity: {opf_age_sens:.2e} yr > 1e3 yr")
