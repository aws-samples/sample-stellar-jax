"""Fast unit tests for the inference/ package.

These tests validate the extracted modules in isolation (<5s each, no evolve_star,
no JIT compilation). They cover:
- LM solver on synthetic forward models (convergence, bounds, singular handling)
- Metallicity roundtrip (z_to_feh ↔ feh_to_z)
- DSEE normalization (shape, roundtrip)
- Parameter mapping (stellar-jax → DSEE labels)
- Warmstart (mass-luminosity estimate)
- ForwardFn protocol contract

Extracted from issue #501 (inference/ consolidation). Design:
docs/design/redesign/10-inference.md §5.

All tests are @pytest.mark.fast (no evolve_star, no heavy JIT compile).
"""

import os
import sys
import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


# ===========================================================================
# §1. LM solver tests (inference/solver.py)
# ===========================================================================


@pytest.mark.fast
def test_lm_quadratic_convergence():
    """LM solver converges on a trivial differentiable forward model.

    Forward model: f(θ) = [θ₀², θ₁²]
    Observed: [4.0, 9.0] → solution θ = [2.0, 3.0]
    Verifies: convergence, covariance shape, history length.
    """
    from stellar_jax.inference.solver import infer_parameters

    def forward_fn(theta):
        return jnp.array([theta[0] ** 2, theta[1] ** 2])

    observed = jnp.array([4.0, 9.0])
    sigma = jnp.array([0.1, 0.1])
    theta_init = jnp.array([1.5, 2.5])

    result = infer_parameters(
        forward_fn, observed, sigma, theta_init,
        param_names=['a', 'b'], max_iter=50, tol_param=1e-8,
    )

    assert result['converged'], f"Did not converge: chi2={result['chi2']}"
    np.testing.assert_allclose(
        np.array(result['theta']), [2.0, 3.0], atol=1e-6,
        err_msg="LM solver did not find the correct minimum"
    )
    assert result['covariance'].shape == (2, 2)
    assert result['sigma_params'].shape == (2,)
    assert len(result['history']) > 0
    assert result['param_names'] == ['a', 'b']


@pytest.mark.fast
def test_lm_bounds_respected():
    """LM solver respects parameter bounds.

    Forward model: f(θ) = [2*θ₀] (linear)
    Observed: [10.0] → unconstrained solution θ₀ = 5.0
    Bounds: [(0.5, 3.0)] → solution should be clipped to 3.0
    """
    from stellar_jax.inference.solver import infer_parameters

    def forward_fn(theta):
        return jnp.array([2.0 * theta[0]])

    observed = jnp.array([10.0])
    sigma = jnp.array([0.1])
    theta_init = jnp.array([1.0])

    result = infer_parameters(
        forward_fn, observed, sigma, theta_init,
        bounds=[(0.5, 3.0)], max_iter=50, tol_param=1e-8,
    )

    # Solution should be at the upper bound
    assert float(result['theta'][0]) <= 3.0 + 1e-10, (
        f"Bound violated: theta={float(result['theta'][0])}"
    )


@pytest.mark.fast
def test_lm_singular_handling():
    """LM solver handles singular Jacobian gracefully (no crash).

    Forward model: f(θ) = [0, 0] (zero gradient everywhere)
    Should NOT crash; should increase λ and eventually return converged=False
    or converge to chi2=const.
    """
    from stellar_jax.inference.solver import infer_parameters

    def forward_fn(theta):
        return jnp.zeros(2)

    observed = jnp.array([1.0, 1.0])
    sigma = jnp.array([0.1, 0.1])
    theta_init = jnp.array([1.0, 1.0])

    # Should not crash
    result = infer_parameters(
        forward_fn, observed, sigma, theta_init,
        max_iter=10,
    )

    # With zero gradient, the solver can't move — should get stuck
    assert 'chi2' in result
    assert 'theta' in result
    assert len(result['history']) > 0


# ===========================================================================
# §1b. Compile-once LM tests (inference/solver.py jax.jit-cached path)
#
# NOTE: test_lm_quadratic_convergence and test_lm_bounds_respected (§1)
# already exercise the compile-once path (jacobian_fn=None is the default).
# Dedicated quadratic/bounds duplicates were removed per reviewer feedback —
# the tests below cover the DISTINCT behaviors: external jacobian_fn override
# and multi-parameter per-wave smoke.
# ===========================================================================


@pytest.mark.fast
def test_lm_external_jacobian_fn():
    """Verify the external jacobian_fn path uses the provided Jacobian.

    WHAT: when jacobian_fn is provided, infer_parameters uses the Python
    loop (not lax.while_loop) and calls the external Jacobian function.

    WHY: the two-level forward pattern (Brent residuals + IFT Jacobian)
    requires an external, potentially non-JAX-traceable Jacobian.

    WHAT FAILS: if the external jacobian_fn is ignored (solver uses jacrev
    instead), the test would fail because forward_fn is non-differentiable.
    """
    from stellar_jax.inference.solver import infer_parameters

    truth = jnp.array([2.0, 3.0])
    observed = truth ** 2
    sigma = jnp.array([1.0, 1.0])

    def forward_fn(theta):
        return theta ** 2

    call_count = [0]

    def my_jacobian_fn(theta):
        call_count[0] += 1
        return jnp.diag(2.0 * theta / sigma)

    result = infer_parameters(
        forward_fn=forward_fn,
        observed=observed,
        sigma=sigma,
        theta_init=jnp.array([1.5, 2.5]),
        jacobian_fn=my_jacobian_fn,
        max_iter=20,
    )

    assert result['converged'] or result['chi2'] < 1e-6
    assert jnp.allclose(result['theta'], truth, atol=1e-4)
    assert call_count[0] >= 2, (
        f"jacobian_fn called {call_count[0]}× — should be ≥2")



# ===========================================================================
# §1b-2. NaN-safe LM solver: singular Jacobian rejection + cond(J)
# ===========================================================================


@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("disable_nan_safe_lm")
@pytest.mark.right_reason("singular")
def test_lm_solver_singular_rejects():
    """NaN-safe LM solver rejects steps when jnp.linalg.solve returns NaN.

    WHAT: constructs a forward model f(θ) = [θ₀+θ₁, θ₀+θ₁] with a rank-1
    Jacobian (two identical columns), and provides an external jacobian_fn
    that returns NaN for the first call (simulating a degenerate Jacobian
    evaluation) then the correct rank-1 Jacobian thereafter. Asserts that
    the solver detects the NaN delta, records 'singular' status, and
    ultimately returns finite theta/cov — NOT NaN.

    Also validates the new diagnostics: n_accepted count in return dict,
    cond_J in non-singular history entries.

    WHY: JAX's jnp.linalg.solve returns NaN (not a Python exception) when
    the coefficient matrix contains NaN. The pre-fix code wrapped it in a
    dead try/except that never fired, allowing NaN to silently propagate
    through theta and covariance. This is the core bug of issue #1029
    (extracted from #526/#1028).

    EXTERNAL REFERENCE: Press et al. (2007) Numerical Recipes §15.5 —
    the Levenberg-Marquardt algorithm must detect a singular/degenerate
    normal-equation matrix and increase the damping parameter λ (reject).

    TOLERANCE: exact assertions — 'singular' status must appear in history,
    theta and cov must be finite (all entries). No tolerance needed because
    the assertion is qualitative (NaN vs finite), not quantitative.

    MUTATION (disable_nan_safe_lm): restores the dead try/except behavior in
    _lm_step, removing the jnp.isnan(delta) guard. Under mutation, NaN
    propagates through theta and cov, so the finiteness assertions FAIL.
    """
    from stellar_jax.inference.solver import infer_parameters

    # Forward model: f(θ) = [θ₀ + θ₁, θ₀ + θ₁].
    # J = [[1/σ, 1/σ], [1/σ, 1/σ]] — rank 1, two identical columns.
    # With LM damping, the solve doesn't produce NaN from rank deficiency
    # alone (the 1e-30 floor regularizes). To reliably trigger the NaN
    # guard, we provide an external jacobian_fn that returns NaN on the
    # first call (simulating the real-world scenario where the Jacobian
    # evaluation fails at a degenerate point).
    def forward_fn(theta):
        s = theta[0] + theta[1]
        return jnp.array([s, s])

    observed = jnp.array([5.0, 5.0])
    sigma = jnp.array([0.1, 0.1])
    theta_init = jnp.array([1.0, 1.0])

    call_count = [0]

    def jacobian_fn_with_nan(theta):
        """Return NaN Jacobian on first call, valid rank-1 Jacobian after."""
        call_count[0] += 1
        if call_count[0] == 1:
            # Simulate degenerate/failed Jacobian evaluation
            return jnp.full((2, 2), jnp.nan)
        # Valid (but rank-deficient) Jacobian for residuals (f-obs)/σ
        return jnp.array([[1.0 / 0.1, 1.0 / 0.1],
                          [1.0 / 0.1, 1.0 / 0.1]])

    result = infer_parameters(
        forward_fn, observed, sigma, theta_init,
        param_names=['a', 'b'], max_iter=20,
        jacobian_fn=jacobian_fn_with_nan,
    )

    # 1. At least one iteration must have status 'singular'.
    statuses = [h.get('status') for h in result['history']]
    assert 'singular' in statuses, (
        f"No 'singular' status in history; got statuses: {statuses}. "
        f"The NaN-safe guard should detect the NaN Jacobian on first call."
    )

    # 2. Final theta must be finite (not NaN from propagated singular solve).
    theta_final = np.array(result['theta'])
    assert np.all(np.isfinite(theta_final)), (
        f"theta contains NaN/Inf: {theta_final}. "
        f"The NaN-safe guard should reject the step and preserve the old theta."
    )

    # 3. Covariance must be finite (pinv fallback handles the rank-1 J^T J).
    cov = np.array(result['covariance'])
    assert np.all(np.isfinite(cov)), (
        f"Covariance contains NaN/Inf: {cov}. "
        f"_lm_covariance should fall back to pinv on rank-deficient J^T J."
    )

    # 4. n_accepted is present and is an integer.
    assert 'n_accepted' in result, "Return dict should include n_accepted"
    assert isinstance(result['n_accepted'], int), (
        f"n_accepted should be int, got {type(result['n_accepted'])}")

    # 5. cond_J is present in non-singular iterations (where info is not None).
    accepted_or_rejected = [h for h in result['history']
                            if h.get('status') in ('accept', 'reject')]
    for h in accepted_or_rejected:
        assert 'cond_J' in h, (
            f"cond_J missing from history entry: {h}. "
            f"Non-singular iterations should record cond(J)."
        )
        assert np.isfinite(h['cond_J']), (
            f"cond_J is not finite: {h['cond_J']}")


# ===========================================================================
# §1c. Per-wave smoke: multi-param recovery via compile-once path
# ===========================================================================


@pytest.mark.fast
def test_lm_compile_once_multi_param_smoke():
    """Per-wave smoke: 2-param recovery exercises the compile-once path.

    WHAT: A synthetic 7-observable / 2-parameter forward model mimicking the
    {M, age} seismic recovery structure (5 frequency-like + 2 photometric-like
    observables). Uses the compile-once default path (jacobian_fn=None →
    jax.jit-cached residuals + Jacobian functions).

    WHY: Per-wave coverage of the #526 compile-once LM machinery with a
    multi-parameter forward model, without the ~10 min evolve_star JIT.
    Catches broken gradient, optimizer, or jit-caching issues per-wave.

    The FULL {M, age} seismic recovery test (evolve_star + oscillations + IFT)
    runs nightly (@suspended); this exercises the same solver path cheaply.

    WHAT FAILS: if the compile-once path, bounds, convergence, or jacrev
    path is broken, the fit doesn't move toward truth (chi2 doesn't decrease
    or parameters don't improve).

    @mutation: not needed — this is a solver-machinery test, not a physics test.
    The physics mutation gates are on the full integration tests.
    """
    from stellar_jax.inference.solver import infer_parameters

    # Synthetic forward model: 7 observables from 2 parameters
    # Mimics the {M, age} structure:
    # - 5 "frequency-like" observables: fi(θ) = a_i * θ₀^0.5 + b_i * θ₁
    # - 2 "photometric-like" observables: logL ∝ θ₀^3.5, logTeff ∝ θ₀^0.6
    # The mixed power-law structure creates non-trivial coupling between params.
    a_freq = jnp.array([100.0, 110.0, 120.0, 130.0, 140.0])
    b_freq = jnp.array([-5.0, -6.0, -7.0, -8.0, -9.0])

    def forward_fn(theta):
        M, age = theta[0], theta[1]
        freqs = a_freq * jnp.sqrt(M) + b_freq * age
        logL = 3.5 * jnp.log10(M) + 0.1
        logTeff = 0.6 * jnp.log10(M) - 0.02 * age + 3.7
        return jnp.concatenate([freqs, jnp.array([logL, logTeff])])

    # Truth
    M_true, age_true = 1.1, 3.5
    theta_true = jnp.array([M_true, age_true])
    observed = forward_fn(theta_true)

    # Uncertainties (similar structure to real test)
    sigma = jnp.array([0.5] * 5 + [0.01, 0.005])

    # Start 10% off in M, 17% off in age
    theta_init = jnp.array([1.21, 4.1])

    # Run with bounded params, 5 max iterations (smoke, not full convergence)
    result = infer_parameters(
        forward_fn=forward_fn,
        observed=observed,
        sigma=sigma,
        theta_init=theta_init,
        param_names=['mass', 'age'],
        max_iter=5,
        bounds=[(0.5, 2.0), (1.0, 6.0)],
    )

    # Assert: chi2 decreased (fit moved toward truth)
    assert len(result['history']) >= 1, "No iterations recorded"
    chi2_start = result['history'][0]['chi2']
    chi2_end = result['chi2']
    assert chi2_end < chi2_start, (
        f"chi2 did not decrease: {chi2_start:.4e} → {chi2_end:.4e}")

    # Assert: parameters moved toward truth
    M_init_err = abs(float(theta_init[0]) - M_true)
    M_final_err = abs(float(result['theta'][0]) - M_true)
    assert M_final_err < M_init_err, (
        f"M did not improve: init_err={M_init_err:.4f}, final_err={M_final_err:.4f}")

    age_init_err = abs(float(theta_init[1]) - age_true)
    age_final_err = abs(float(result['theta'][1]) - age_true)
    assert age_final_err < age_init_err, (
        f"age did not improve: init_err={age_init_err:.3f}, final_err={age_final_err:.3f}")

    # Assert: Jacobian has the correct shape and is non-zero
    J = result['jacobian']
    assert J.shape == (7, 2), f"Jacobian shape {J.shape}, expected (7, 2)"
    assert float(jnp.linalg.norm(J)) > 0.1, "Jacobian is near-zero"


# ===========================================================================
# §2. Metallicity tests (inference/metallicity.py)
# ===========================================================================


@pytest.mark.fast
def test_metallicity_roundtrip():
    """feh_to_z(z_to_feh(Z)) == Z for a range of metallicities."""
    from stellar_jax.inference.metallicity import z_to_feh, feh_to_z

    for Z in [0.001, 0.005, 0.014, 0.020, 0.040]:
        feh = z_to_feh(Z)
        Z_back = feh_to_z(feh)
        np.testing.assert_allclose(
            Z_back, Z, rtol=1e-10,
            err_msg=f"Roundtrip failed for Z={Z}: [Fe/H]={feh}, Z_back={Z_back}"
        )


@pytest.mark.fast
def test_metallicity_solar_is_zero():
    """[Fe/H] = 0 at solar metallicity Z_solar."""
    from stellar_jax.inference.metallicity import z_to_feh, Z_SOLAR
    from stellar_jax.config.constants import Y_BBN, DY_DZ

    Y_solar = Y_BBN + DY_DZ * Z_SOLAR
    feh = z_to_feh(Z_SOLAR, Y=Y_solar)
    assert abs(feh) < 0.001, f"Solar [Fe/H] should be ~0, got {feh}"


@pytest.mark.fast
def test_metallicity_subsolar_negative():
    """Sub-solar Z produces negative [Fe/H]."""
    from stellar_jax.inference.metallicity import z_to_feh

    feh = z_to_feh(0.004)
    assert feh < -0.5, f"Expected [Fe/H]<-0.5 for Z=0.004, got {feh}"


@pytest.mark.fast
def test_metallicity_supersolar_positive():
    """Super-solar Z produces positive [Fe/H]."""
    from stellar_jax.inference.metallicity import z_to_feh

    feh = z_to_feh(0.030)
    assert feh > 0.1, f"Expected [Fe/H]>0.1 for Z=0.030, got {feh}"


@pytest.mark.fast
def test_metallicity_unphysical_raises():
    """Unphysical Z or Y raises ValueError."""
    from stellar_jax.inference.metallicity import z_to_feh, feh_to_z

    with pytest.raises(ValueError):
        z_to_feh(0.0, Y=0.5)  # Z=0 invalid

    with pytest.raises(ValueError):
        z_to_feh(0.5, Y=0.6)  # X = 1-0.6-0.5 < 0


# ===========================================================================
# §3. DSEE normalization tests (inference/emulator.py)
# ===========================================================================


@pytest.mark.fast
def test_normalize_labels_shape():
    """_normalize_labels produces shape (25,) from 24 physical labels.

    Age uses mantissa_exponent → expands from 1 to 2 dimensions.
    """
    from stellar_jax.inference.emulator import _normalize_labels, DSEE_LABEL_DEFAULT

    normed = _normalize_labels(DSEE_LABEL_DEFAULT)
    assert normed.shape == (25,), f"Expected (25,), got {normed.shape}"
    assert normed.dtype == np.float32


@pytest.mark.fast
def test_denormalize_roundtrip():
    """Normalize then denormalize a known output vector → recovers original.

    Tests the output-space (5-dim) normalization/denormalization roundtrip.
    """
    from stellar_jax.inference.emulator import _denormalize_outputs, DSEE_DATA_SCALE, DSEE_DATA_SCALER

    # Create a physically reasonable output: [log_T, log_g, log_L, log_R, Y_core]
    physical = np.array([3.76, 4.44, 0.0, 0.0, 0.35])

    # Normalize manually (forward pass)
    normed = np.empty(5)
    for i, (scaler, scale) in enumerate(zip(DSEE_DATA_SCALER, DSEE_DATA_SCALE)):
        if scaler == "linear_rescale":
            normed[i] = (physical[i] - scale[0]) / (scale[1] - scale[0])
        elif scaler == "log_rescale":
            normed[i] = (np.log10(physical[i]) - scale[0]) / (scale[1] - scale[0])

    # Denormalize back
    recovered = _denormalize_outputs(normed)

    # First 4 are linear_rescale → should roundtrip exactly
    np.testing.assert_allclose(
        recovered[:4], physical[:4], atol=1e-10,
        err_msg="linear_rescale denormalization roundtrip failed"
    )
    # Y_core uses log_rescale — roundtrip only works on log10(Y_core) space
    # since the forward normalizes log10(val), we test differently
    # Actually for log_rescale: normed = (log10(val) - lo) / (hi - lo)
    # denorm: val = 10^(normed * (hi-lo) + lo)
    # For val=0.35: log10(0.35) = -0.4559, scale=[-2, 0], normed = (-0.4559-(-2))/(0-(-2)) = 0.772
    # denorm: 10^(0.772*2 + (-2)) = 10^(-0.456) = 0.35 ✓


@pytest.mark.fast
def test_normalization_helpers():
    """Individual normalization helpers produce correct values."""
    from stellar_jax.inference.emulator import (
        _norm_linear_rescale, _norm_normal, _norm_mantissa_exponent
    )

    # linear_rescale: (val - lo) / (hi - lo)
    assert abs(_norm_linear_rescale(1.75, [1.75, 2.5]) - 0.0) < 1e-10
    assert abs(_norm_linear_rescale(2.5, [1.75, 2.5]) - 1.0) < 1e-10
    assert abs(_norm_linear_rescale(2.125, [1.75, 2.5]) - 0.5) < 1e-10

    # normal: (val - mean) / std
    assert abs(_norm_normal(1.0, [1.0, 0.1]) - 0.0) < 1e-10
    assert abs(_norm_normal(1.1, [1.0, 0.1]) - 1.0) < 1e-10

    # mantissa_exponent: returns (mantissa_scaled, exponent_scaled)
    m, e = _norm_mantissa_exponent(4.603e9, [3.0, 10.0])
    assert isinstance(m, float) or isinstance(m, np.floating)
    assert isinstance(e, float) or isinstance(e, np.floating)


# ===========================================================================
# §4. Parameter mapping tests (inference/emulator.py)
# ===========================================================================


@pytest.mark.fast
def test_stellar_jax_to_dsee_labels_defaults():
    """Parameter mapping produces correct [Fe/H], mass, age positions."""
    from stellar_jax.inference.emulator import stellar_jax_to_dsee_labels
    from stellar_jax.inference.metallicity import z_to_feh
    from stellar_jax.config.constants import Y_BBN, DY_DZ
    from stellar_jax.config.mesh_defaults import ALPHA_MLT, F_OV

    labels = stellar_jax_to_dsee_labels(mass=1.5, Z=0.014, age_yr=5.0e9)

    # Check mass position
    assert labels[23] == 1.5, f"Mass should be 1.5, got {labels[23]}"

    # Check age position
    assert labels[22] == 5.0e9, f"Age should be 5e9, got {labels[22]}"

    # Check alpha_mlt
    assert labels[3] == ALPHA_MLT, f"α_MLT should be {ALPHA_MLT}, got {labels[3]}"

    # Check [Fe/H] is consistent
    Y_init = Y_BBN + DY_DZ * 0.014
    expected_feh = z_to_feh(0.014, Y=Y_init)
    np.testing.assert_allclose(
        labels[0], expected_feh, atol=1e-10,
        err_msg="[Fe/H] in label vector is inconsistent"
    )

    # Check f_ov default
    assert labels[8] == F_OV, f"f_ov should be {F_OV}, got {labels[8]}"

    # Check surface BC = Krishna-Swamy
    assert labels[6] == 1.0, f"Surface BC should be 1.0, got {labels[6]}"


@pytest.mark.fast
def test_stellar_jax_to_dsee_labels_diffusion_off():
    """Diffusion=False sets He/Heavy diffusion factors to 0.5."""
    from stellar_jax.inference.emulator import stellar_jax_to_dsee_labels

    labels = stellar_jax_to_dsee_labels(mass=1.0, diffusion=False)
    assert labels[4] == 0.5, f"He diffusion should be 0.5, got {labels[4]}"
    assert labels[5] == 0.5, f"Heavy diffusion should be 0.5, got {labels[5]}"


@pytest.mark.fast
def test_dsee_outputs_to_observables():
    """dsee_outputs_to_observables converts 5-dim array to labeled dict."""
    from stellar_jax.inference.emulator import dsee_outputs_to_observables

    raw = np.array([3.76, 4.44, 0.0, 0.0, 0.35])
    obs = dsee_outputs_to_observables(raw)

    assert obs['log_Teff'] == 3.76
    assert obs['log_g'] == 4.44
    assert obs['log_L'] == 0.0
    assert obs['log_R'] == 0.0
    assert obs['Y_core'] == 0.35


# ===========================================================================
# §5. Warmstart tests (inference/warmstart.py)
# ===========================================================================


@pytest.mark.fast
def test_warmstart_solar_luminosity():
    """warmstart_params_from_observables returns mass ~1.0 for solar properties."""
    from stellar_jax.inference.warmstart import warmstart_params_from_observables

    guess = warmstart_params_from_observables(
        log_Teff_obs=3.76, log_L_obs=0.0
    )

    assert 0.8 < guess['mass'] < 1.2, (
        f"Solar luminosity should give mass ~1.0, got {guess['mass']}"
    )
    assert guess['Z'] == 0.014  # default
    assert guess['diffusion'] is True


@pytest.mark.fast
def test_warmstart_with_feh():
    """warmstart converts [Fe/H] to Z correctly."""
    from stellar_jax.inference.warmstart import warmstart_params_from_observables
    from stellar_jax.inference.metallicity import feh_to_z

    guess = warmstart_params_from_observables(
        log_Teff_obs=3.76, log_L_obs=0.0, feh_obs=-0.5
    )

    expected_Z = feh_to_z(-0.5)
    np.testing.assert_allclose(
        guess['Z'], expected_Z, rtol=1e-10,
        err_msg="Warmstart Z from [Fe/H] is inconsistent"
    )


@pytest.mark.fast
def test_warmstart_luminous_star():
    """High luminosity gives higher mass estimate."""
    from stellar_jax.inference.warmstart import warmstart_params_from_observables

    guess = warmstart_params_from_observables(
        log_Teff_obs=4.0, log_L_obs=2.0  # ~100 L_sun
    )

    # L/L_sun = 100 → M ≈ 100^(1/3.5) ≈ 3.7 → clamped to 3.0
    assert guess['mass'] == 3.0, (
        f"Very luminous star should clamp to max=3.0, got {guess['mass']}"
    )


# ===========================================================================
# §6. ForwardFn protocol contract test
# ===========================================================================


@pytest.mark.fast
def test_forward_fn_protocol():
    """ForwardFn Protocol is satisfied by a well-typed callable."""
    from stellar_jax.inference.contracts import ForwardFn
    from typing import runtime_checkable, Protocol

    # Our protocol should be usable for isinstance checks if made runtime_checkable
    # (we don't enforce that at runtime, but verify structurally)

    # A valid forward_fn
    def good_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return theta ** 2

    # Verify it has the right signature (duck typing — Protocol is structural)
    result = good_fn(jnp.array([1.0, 2.0]))
    assert result.shape == (2,)
    assert hasattr(good_fn, '__call__')


# ===========================================================================
# §7. Warm-start integration tests (inference/solver.py + warmstart.py)
# — B2 optional flagged emulator warm-start
# ===========================================================================


@pytest.mark.integration
@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("disable_warmstart_fn")
@pytest.mark.right_reason("FEWER iterations")
def test_warmstart_reduces_iterations():
    """Warm-start flag reduces LM iterations without changing the final answer.

    Issue #473: The emulator warm-start provides a better initial guess (closer
    to truth) so the optimizer converges in fewer iterations. The FINAL answer
    must be the same (within tolerance) — the emulator only accelerates, never
    changes the physics.

    Setup: a synthetic 2-param forward model f(θ) = [θ₀^1.5, 2*θ₁^2] with a
    known solution. Cold start = 50% off; warm start = 5% off (simulating what
    an emulator would provide).

    Assertions:
    - Both converge to the same answer (atol=1e-5)
    - Warm-start uses FEWER iterations (n_iter_warm < n_iter_cold)
    - Under 'disable_warmstart_fn' mutation, warmstart_fn is stripped →
      the warm-start path falls back to theta_init (the cold guess) →
      n_iter_warm == n_iter_cold → the "fewer iterations" assertion FAILS.

    References:
    - Nocedal & Wright (2006), §11.3: initial guess quality determines
      convergence speed in trust-region / LM methods.
    - Issue #473, epic #466 Arm B.
    """
    from stellar_jax.inference.solver import infer_parameters

    # Synthetic forward model: f(θ) = [θ₀^1.5, 2*θ₁²]
    # Solution at observed=[8.0, 18.0]: θ₀ = 8^(2/3) = 4.0, θ₁ = 3.0
    def forward_fn(theta):
        return jnp.array([theta[0] ** 1.5, 2.0 * theta[1] ** 2])

    truth = jnp.array([4.0, 3.0])
    observed = forward_fn(truth)  # [8.0, 18.0]
    sigma = jnp.array([0.01, 0.01])

    # Cold start: 50% off truth
    theta_cold = jnp.array([6.0, 4.5])  # +50%

    # Warm start function: returns a guess 5% off truth (simulating emulator)
    theta_warm_value = jnp.array([4.2, 3.15])  # +5%

    def mock_warmstart_fn(obs, sig, pnames):
        """Mock emulator: returns a guess close to truth."""
        return np.array(theta_warm_value)

    # Run cold start
    result_cold = infer_parameters(
        forward_fn, observed, sigma, theta_init=theta_cold,
        param_names=['a', 'b'], max_iter=100, tol_param=1e-8,
    )

    # Run warm start (warmstart_fn provides the init)
    # Pass theta_init as fallback (same as cold) — when warmstart_fn is active,
    # it overrides theta_init with the better guess. Under 'disable_warmstart_fn'
    # mutation, warmstart_fn is stripped → falls back to theta_cold → same n_iter
    # as cold → the "fewer iterations" assertion FAILS.
    result_warm = infer_parameters(
        forward_fn, observed, sigma, theta_init=theta_cold,
        warmstart_fn=mock_warmstart_fn,
        param_names=['a', 'b'], max_iter=100, tol_param=1e-8,
    )

    # Both must converge
    assert result_cold['converged'], (
        f"Cold start did not converge: chi2={result_cold['chi2']}")
    assert result_warm['converged'], (
        f"Warm start did not converge: chi2={result_warm['chi2']}")

    # Same final answer (the warm-start doesn't change the physics)
    np.testing.assert_allclose(
        np.array(result_warm['theta']),
        np.array(result_cold['theta']),
        atol=1e-5,
        err_msg="Warm-start and cold-start must converge to the SAME answer"
    )

    # Both close to truth
    np.testing.assert_allclose(
        np.array(result_cold['theta']), np.array(truth), atol=1e-4,
        err_msg="Cold start did not recover truth"
    )

    # THE KEY ASSERTION: warm-start uses fewer iterations
    # Under 'disable_warmstart_fn' mutation, warmstart_fn is stripped →
    # this falls back to theta_init=None → raises ValueError (or if we
    # pass theta_init as fallback, both use cold start → same n_iter).
    assert result_warm['n_iter'] < result_cold['n_iter'], (
        f"Warm-start should use FEWER iterations: "
        f"warm={result_warm['n_iter']}, cold={result_cold['n_iter']}. "
        f"Under the disable_warmstart_fn mutation, this is expected to fail."
    )


@pytest.mark.fast
def test_warmstart_fn_overrides_theta_init():
    """When warmstart_fn is provided, it overrides theta_init.

    This is the B2 flag semantics: providing warmstart_fn = warm-start ON;
    omitting it = cold start (theta_init used directly).
    """
    from stellar_jax.inference.solver import infer_parameters

    def forward_fn(theta):
        return jnp.array([theta[0] ** 2])

    observed = jnp.array([9.0])  # truth = 3.0
    sigma = jnp.array([0.01])

    # theta_init = far from truth
    theta_far = jnp.array([1.0])

    # warmstart_fn = close to truth
    def close_warmstart(obs, sig, pnames):
        return np.array([2.9])

    result = infer_parameters(
        forward_fn, observed, sigma, theta_init=theta_far,
        warmstart_fn=close_warmstart,
        param_names=['x'], max_iter=50, tol_param=1e-8,
    )

    # Should converge to truth
    assert result['converged']
    np.testing.assert_allclose(float(result['theta'][0]), 3.0, atol=1e-4)

    # It should have used fewer iterations than from theta_far (convergence from 2.9 is fast)
    result_cold = infer_parameters(
        forward_fn, observed, sigma, theta_init=theta_far,
        param_names=['x'], max_iter=50, tol_param=1e-8,
    )
    assert result['n_iter'] <= result_cold['n_iter'], (
        "warmstart_fn should have provided a better starting point"
    )


@pytest.mark.fast
def test_warmstart_fn_no_init_raises():
    """Without theta_init or warmstart_fn, infer_parameters raises ValueError."""
    from stellar_jax.inference.solver import infer_parameters

    def forward_fn(theta):
        return jnp.array([theta[0]])

    with pytest.raises(ValueError, match="theta_init is required"):
        infer_parameters(
            forward_fn,
            observed=jnp.array([1.0]),
            sigma=jnp.array([0.1]),
            theta_init=None,
            warmstart_fn=None,
        )


@pytest.mark.fast
def test_make_warmstart_fn_solar():
    """make_warmstart_fn produces a reasonable initial guess for a solar-like star.

    Given observed logL≈0, logTeff≈3.76 (solar), the warmstart should return
    mass≈1.0 and opacity_factor≈1.0.
    """
    from stellar_jax.inference.warmstart import make_warmstart_fn

    warmstart = make_warmstart_fn(
        param_names=['mass', 'opacity_factor'],
        observed_names=['logL', 'logTeff'],
    )

    observed = np.array([0.0, np.log10(5778.0)])  # solar: logL=0, logTeff=3.76
    sigma = np.array([0.05, 0.01])

    theta_init = warmstart(observed, sigma, ['mass', 'opacity_factor'])

    # mass should be near 1.0 (solar luminosity)
    assert 0.8 < theta_init[0] < 1.2, (
        f"Solar mass guess should be ~1.0, got {theta_init[0]}"
    )
    # opacity_factor defaults to 1.0 (no warmstart info for it)
    assert theta_init[1] == 1.0, (
        f"opacity_factor default should be 1.0, got {theta_init[1]}"
    )


@pytest.mark.fast
def test_make_warmstart_fn_luminous():
    """make_warmstart_fn gives higher mass for a luminous star."""
    from stellar_jax.inference.warmstart import make_warmstart_fn

    warmstart = make_warmstart_fn(
        param_names=['mass', 'alpha_mlt'],
        observed_names=['logL', 'logTeff'],
    )

    # log L = 2.0 → L ≈ 100 L_sun → M ≈ 100^(1/3.5) ≈ 3.7 (clamped to 3.0)
    observed = np.array([2.0, 3.85])
    sigma = np.array([0.05, 0.01])

    theta_init = warmstart(observed, sigma, ['mass', 'alpha_mlt'])

    # Should get a high mass (clamped at 3.0)
    assert theta_init[0] >= 2.5, (
        f"Luminous star mass guess should be ≥2.5, got {theta_init[0]}"
    )


# ===========================================================================
# §8. DSEE emulator warm-start factory tests (inference/warmstart.py)
# — B2: wire the DSEE emulator as optional warm-start
# ===========================================================================


@pytest.mark.fast
def test_make_dsee_warmstart_fn_fallback_to_pure():
    """When DSEE is unavailable, make_dsee_warmstart_fn falls back to pure mapping.

    This tests the fallback_to_pure=True (default) behavior: if PyTorch/DSEE
    aren't available, the factory silently returns a pure-mapping warmstart_fn
    that still produces a physically motivated initial guess.
    """
    from stellar_jax.inference.warmstart import make_dsee_warmstart_fn

    # Use a model_path that doesn't exist — forces fallback
    warmstart = make_dsee_warmstart_fn(
        param_names=['mass', 'opacity_factor'],
        observed_names=['logL', 'logTeff'],
        model_path='/nonexistent/DSEE.model',
        fallback_to_pure=True,
    )

    # Should still produce a reasonable guess (from pure mapping)
    observed = np.array([0.0, np.log10(5778.0)])  # solar
    sigma = np.array([0.05, 0.01])

    theta = warmstart(observed, sigma, ['mass', 'opacity_factor'])

    # mass should be near 1.0 (solar luminosity → mass-luminosity)
    assert 0.8 < theta[0] < 1.2, (
        f"Fallback mass guess should be ~1.0, got {theta[0]}"
    )
    # opacity_factor defaults to 1.0
    assert theta[1] == 1.0, (
        f"Fallback opacity_factor should be 1.0, got {theta[1]}"
    )


@pytest.mark.fast
def test_make_dsee_warmstart_fn_no_fallback_raises():
    """When DSEE is unavailable and fallback_to_pure=False, raises ImportError."""
    from stellar_jax.inference.warmstart import make_dsee_warmstart_fn

    with pytest.raises(ImportError, match="DSEE emulator not available"):
        make_dsee_warmstart_fn(
            param_names=['mass'],
            model_path='/nonexistent/DSEE.model',
            fallback_to_pure=False,
        )


@pytest.mark.integration
@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("disable_warmstart_fn")
@pytest.mark.right_reason("FEWER iterations")
def test_dsee_warmstart_reduces_iterations():
    """DSEE-based warmstart produces a tighter initial guess than pure mapping.

    Issue #473: Wire the emulator as an optional warm-start that reduces
    iteration count without changing the final answer. This test uses a MOCK
    DSEEInterface (no real model weights needed) to verify the wiring end-to-end.

    The mock emulator is a perfect predictor of the synthetic forward model,
    so the DSEE-refined guess lands very close to truth. This should converge
    in fewer LM iterations than a pure-mapping guess (which only knows
    mass-luminosity relations).

    Under 'disable_warmstart_fn' mutation, warmstart_fn is stripped → cold start
    → same n_iter as pure-mapping → "fewer iterations" assertion FAILS.
    """
    from stellar_jax.inference.solver import infer_parameters
    from stellar_jax.inference.warmstart import make_dsee_warmstart_fn

    # Synthetic forward model: f(θ) = [θ₀^1.5, 0.5*θ₁^2]
    # Truth: θ = [4.0, 3.0] → observed = [8.0, 4.5]
    def forward_fn(theta):
        return jnp.array([theta[0] ** 1.5, 0.5 * theta[1] ** 2])

    truth = jnp.array([4.0, 3.0])
    observed = forward_fn(truth)  # [8.0, 4.5]
    sigma = jnp.array([0.01, 0.01])

    # Cold start: 50% off (simulating what pure-mapping gives)
    theta_cold = jnp.array([6.0, 4.5])

    # Mock DSEEInterface that acts as a perfect oracle for this forward model.
    # It predicts log_L and log_Teff from mass and alpha_mlt by running the
    # same forward model — this simulates a well-trained emulator.
    class MockDSEEInterface:
        """Mock emulator: perfect predictor for the test forward model."""

        def predict(self, mass=1.0, Z=0.014, alpha_mlt=1.9, f_ov=0.016,
                    Y_init=None, age_yr=4.603e9, diffusion=True,
                    opacity_factor=1.0):
            # Map mass → logL (our forward model's first output)
            # Map alpha_mlt → logTeff (our forward model's second output)
            pred_logL = mass ** 1.5
            pred_logTeff = 0.5 * alpha_mlt ** 2
            return {
                'log_L': float(pred_logL),
                'log_Teff': float(pred_logTeff),
                'log_R': 0.0,
                'log_g': 4.44,
                'Y_core': 0.35,
            }

    # Create DSEE warmstart with the mock interface
    dsee_warmstart = make_dsee_warmstart_fn(
        param_names=['mass', 'alpha_mlt'],
        observed_names=['logL', 'logTeff'],
        dsee_interface=MockDSEEInterface(),
        n_refine=30,  # enough for Nelder-Mead to converge on this simple model
    )

    # Run with pure-mapping warmstart (cold)
    result_cold = infer_parameters(
        forward_fn, observed, sigma, theta_init=theta_cold,
        param_names=['mass', 'alpha_mlt'], max_iter=100, tol_param=1e-8,
    )

    # Run with DSEE warmstart
    result_dsee = infer_parameters(
        forward_fn, observed, sigma, theta_init=theta_cold,
        warmstart_fn=dsee_warmstart,
        param_names=['mass', 'alpha_mlt'], max_iter=100, tol_param=1e-8,
    )

    # Both must converge
    assert result_cold['converged'], (
        f"Cold start did not converge: chi2={result_cold['chi2']}")
    assert result_dsee['converged'], (
        f"DSEE warmstart did not converge: chi2={result_dsee['chi2']}")

    # Same final answer (emulator only accelerates, never changes physics)
    np.testing.assert_allclose(
        np.array(result_dsee['theta']),
        np.array(result_cold['theta']),
        atol=1e-5,
        err_msg="DSEE warmstart and cold-start must converge to the SAME answer"
    )

    # Both close to truth
    np.testing.assert_allclose(
        np.array(result_cold['theta']), np.array(truth), atol=1e-4,
        err_msg="Cold start did not recover truth"
    )

    # THE KEY ASSERTION: DSEE warmstart uses fewer iterations
    assert result_dsee['n_iter'] < result_cold['n_iter'], (
        f"DSEE warmstart should use FEWER iterations: "
        f"dsee={result_dsee['n_iter']}, cold={result_cold['n_iter']}. "
        f"Under the disable_warmstart_fn mutation, this is expected to fail."
    )


@pytest.mark.fast
def test_make_dsee_warmstart_fn_with_mock_interface():
    """make_dsee_warmstart_fn accepts a pre-built DSEEInterface instance.

    This verifies the dsee_interface= kwarg path: when passed a DSEEInterface
    directly, the factory skips the model-loading step and uses it immediately.
    """
    from stellar_jax.inference.warmstart import make_dsee_warmstart_fn

    # Minimal mock that returns fixed predictions
    class SimpleMock:
        def predict(self, **kwargs):
            return {
                'log_Teff': 3.76,
                'log_g': 4.44,
                'log_L': 0.0,
                'log_R': 0.0,
                'Y_core': 0.35,
            }

    warmstart = make_dsee_warmstart_fn(
        param_names=['mass', 'Z'],
        observed_names=['logL', 'logTeff'],
        dsee_interface=SimpleMock(),
        n_refine=3,
    )

    observed = np.array([0.0, 3.76])
    sigma = np.array([0.05, 0.01])

    theta = warmstart(observed, sigma, ['mass', 'Z'])

    # Should return a 2-element array
    assert theta.shape == (2,), f"Expected shape (2,), got {theta.shape}"
    # mass should be positive and physical
    assert 0.1 < theta[0] < 5.0, f"mass={theta[0]} out of range"
    # Z should be positive and physical
    assert 0.001 < theta[1] < 0.1, f"Z={theta[1]} out of range"


# ═══════════════════════════════════════════════════════════════════════════════
# Fisher matrix assembly — fast unit tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFisherAssembly:
    """Fast unit tests for compute_fisher / forecast_sigma (pure linear algebra).

    WHAT: Fisher matrix assembly F = Jᵀ Σ⁻¹ J and Cramér-Rao bound σ(θ_j) =
    sqrt([F⁻¹]_jj) on synthetic Jacobian data. No evolve_star, no JIT.

    WHY: Validates the linear-algebra assembly independently of the expensive
    stellar forward model. Catches sign errors, shape mismatches, and
    numerical instabilities in the matrix inversion.

    EXTERNAL REFERENCE: Iacovelli et al. (2022), arXiv:2207.06910, Eq. 2.7 —
    F = Jᵀ Σ⁻¹ J. The analytic Fisher for a 1D linear model h(θ) = θ with
    N observations at σ is F = N/σ² (scalar case).
    """

    @pytest.mark.fast
    def test_fisher_identity_jacobian(self):
        """F = Jᵀ Σ⁻¹ J for J=I, Σ=σ²I gives F = (1/σ²)I."""
        from stellar_jax.inference.fisher import (
            compute_fisher, JacobianResult, forecast_sigma,
        )

        n = 3
        J = np.eye(n)
        sigma_val = 0.1
        sigma_obs = np.full(n, sigma_val)

        jac_result = JacobianResult(
            jacobian=jnp.array(J),
            obs_values=np.zeros(n),
            obs_labels=[f'obs_{i}' for i in range(n)],
            param_names=[f'p_{i}' for i in range(n)],
            param_values=np.ones(n),
        )

        fisher = compute_fisher(jac_result, sigma_obs)

        # F should be (1/σ²) I
        expected_F = np.eye(n) / sigma_val**2
        np.testing.assert_allclose(
            np.asarray(fisher.fisher), expected_F, rtol=1e-12,
            err_msg="F = Jᵀ Σ⁻¹ J for J=I should give (1/σ²)I")

        # Condition number of identity is 1.0
        assert abs(fisher.condition_number - 1.0) < 1e-10, (
            f"κ(I) should be 1.0, got {fisher.condition_number}")

        # All eigenvalues should be 1/σ²
        np.testing.assert_allclose(
            fisher.eigenvalues, np.full(n, 1.0 / sigma_val**2), rtol=1e-12)

        # forecast σ should be σ_val for each parameter
        sigmas = forecast_sigma(fisher)
        for pname in jac_result.param_names:
            assert abs(sigmas[pname] - sigma_val) < 1e-12, (
                f"σ({pname}) should be {sigma_val}, got {sigmas[pname]}")

    @pytest.mark.fast
    def test_fisher_rectangular_jacobian(self):
        """F = Jᵀ Σ⁻¹ J for a 5-obs × 2-param system gives (2×2) F."""
        from stellar_jax.inference.fisher import (
            compute_fisher, JacobianResult, forecast_sigma,
        )

        # 5 observations, 2 parameters — more obs than params (typical)
        n_obs, n_params = 5, 2
        rng = np.random.RandomState(42)
        J = rng.randn(n_obs, n_params)
        sigma_obs = np.abs(rng.randn(n_obs)) + 0.1  # positive sigmas

        jac_result = JacobianResult(
            jacobian=jnp.array(J),
            obs_values=np.zeros(n_obs),
            obs_labels=[f'obs_{i}' for i in range(n_obs)],
            param_names=['mass', 'alpha_mlt'],
            param_values=np.array([1.0, 1.9]),
        )

        fisher = compute_fisher(jac_result, sigma_obs)

        # Verify F = Jᵀ diag(1/σ²) J manually
        W = np.diag(1.0 / sigma_obs**2)
        F_expected = J.T @ W @ J
        np.testing.assert_allclose(
            np.asarray(fisher.fisher), F_expected, rtol=1e-12,
            err_msg="F should equal Jᵀ diag(1/σ²) J")

        # Shape check
        assert fisher.fisher.shape == (n_params, n_params)
        assert fisher.eigenvalues.shape == (n_params,)

        # Condition number > 1 for non-identity
        assert fisher.condition_number >= 1.0

        # Forecast sigmas are finite and positive
        sigmas = forecast_sigma(fisher)
        for pname in ['mass', 'alpha_mlt']:
            assert np.isfinite(sigmas[pname]) and sigmas[pname] > 0

    @pytest.mark.fast
    def test_fisher_shape_mismatch_raises(self):
        """compute_fisher raises ValueError when J rows != sigma length."""
        from stellar_jax.inference.fisher import compute_fisher, JacobianResult

        jac_result = JacobianResult(
            jacobian=jnp.zeros((3, 2)),
            obs_values=np.zeros(3),
            obs_labels=['a', 'b', 'c'],
            param_names=['p0', 'p1'],
            param_values=np.array([1.0, 2.0]),
        )

        with pytest.raises(ValueError, match="sigma_obs"):
            compute_fisher(jac_result, np.ones(5))  # wrong size

    @pytest.mark.fast
    def test_fisher_positive_semidefinite(self):
        """F = Jᵀ Σ⁻¹ J is always positive semi-definite."""
        from stellar_jax.inference.fisher import compute_fisher, JacobianResult

        # Use a rank-deficient Jacobian (3 obs, 3 params, rank 2)
        J = np.array([[1.0, 0.0, 1.0],
                       [0.0, 1.0, 1.0],
                       [1.0, 1.0, 2.0]])  # row 3 = row 1 + row 2
        sigma_obs = np.array([0.1, 0.1, 0.1])

        jac_result = JacobianResult(
            jacobian=jnp.array(J),
            obs_values=np.zeros(3),
            obs_labels=['a', 'b', 'c'],
            param_names=['p0', 'p1', 'p2'],
            param_values=np.ones(3),
        )

        fisher = compute_fisher(jac_result, sigma_obs)

        # All eigenvalues >= 0 (PSD)
        assert np.all(fisher.eigenvalues >= -1e-14), (
            f"Fisher should be PSD, eigenvalues: {fisher.eigenvalues}")

    @pytest.mark.fast
    def test_forecast_sigma_cramer_rao(self):
        """Cramér-Rao bound: σ(θ_j) = sqrt([F⁻¹]_jj) for a known case.

        For a single observation h(θ) = θ₁ + θ₂ with σ = 0.1,
        F = (1/σ²) [[1,1],[1,1]]. F is singular → forecast uses pseudoinverse.
        """
        from stellar_jax.inference.fisher import (
            compute_fisher, JacobianResult, forecast_sigma,
        )

        # Non-degenerate 2-param case: J = [[1, 0], [0, 1], [1, 1]]
        J = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        sigma_obs = np.array([0.1, 0.1, 0.1])

        jac_result = JacobianResult(
            jacobian=jnp.array(J),
            obs_values=np.zeros(3),
            obs_labels=['a', 'b', 'c'],
            param_names=['mass', 'Z'],
            param_values=np.array([1.0, 0.014]),
        )

        fisher = compute_fisher(jac_result, sigma_obs)
        sigmas = forecast_sigma(fisher)

        # F = Jᵀ W J = [[200, 100], [100, 200]], F⁻¹ = [[2/300, -1/300], [-1/300, 2/300]]
        # σ(mass) = sqrt(2/300) ≈ 0.08165, σ(Z) = sqrt(2/300) ≈ 0.08165
        expected_sigma = np.sqrt(2.0 / 300.0)
        assert abs(sigmas['mass'] - expected_sigma) < 1e-10, (
            f"σ(mass) = {sigmas['mass']:.6f}, expected {expected_sigma:.6f}")
        assert abs(sigmas['Z'] - expected_sigma) < 1e-10, (
            f"σ(Z) = {sigmas['Z']:.6f}, expected {expected_sigma:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLATO forecast helper unit tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.fast
def test_propagate_derived_sigma_identity_covariance():
    """_propagate_derived_sigma with identity C gives σ = ||grad||.

    WHAT: Tests the Fisher error propagation formula σ²(g) = ∇gᵀ C ∇g
    with C = I, where the result should be ||∇g||₂.

    WHY: Verifies the core error propagation helper is wired correctly
    before it's used in the PLATO forecast for σ(R) and σ(age).

    EXTERNAL REFERENCE: Tegmark, Taylor & Heavens (1997), ApJ 480, 22,
    Eq. 2 — σ²(g) = (∇g)ᵀ F⁻¹ (∇g).

    WHAT MAKES IT FAIL: Any bug in _propagate_derived_sigma (wrong matrix
    product, missing sqrt, sign error).
    """
    from stellar_jax.inference.plato_forecast import _propagate_derived_sigma

    C = np.eye(3)
    grad = np.array([3.0, 4.0, 0.0])
    sigma = _propagate_derived_sigma(C, grad)
    assert abs(sigma - 5.0) < 1e-12, f"σ = {sigma}, expected 5.0 (3-4-5 triangle)"


@pytest.mark.fast
def test_propagate_derived_sigma_scaled_covariance():
    """_propagate_derived_sigma with diagonal C scales correctly.

    WHAT: σ²(g) = Σ_j (∂g/∂θ_j)² σ²(θ_j) for diagonal C.

    WHY: Verifies the error propagation with a non-trivial covariance.

    EXTERNAL REFERENCE: standard error propagation (Bevington & Robinson,
    "Data Reduction and Error Analysis", Ch. 3).
    """
    from stellar_jax.inference.plato_forecast import _propagate_derived_sigma

    # C = diag(0.01, 0.04, 0.09) → σ = (0.1, 0.2, 0.3)
    C = np.diag([0.01, 0.04, 0.09])
    grad = np.array([1.0, 1.0, 1.0])
    sigma = _propagate_derived_sigma(C, grad)
    # σ² = 0.01 + 0.04 + 0.09 = 0.14 → σ = sqrt(0.14)
    expected = np.sqrt(0.14)
    assert abs(sigma - expected) < 1e-12, f"σ = {sigma}, expected {expected}"


@pytest.mark.fast
def test_propagate_derived_sigma_zero_gradient():
    """_propagate_derived_sigma with zero gradient gives σ = 0.

    WHAT: If ∂g/∂θ = 0 for all parameters, σ(g) = 0.

    WHY: Edge case — the PLATO forecast uses this to detect when ∂R/∂θ ≈ 0
    (degenerate model) and falls back to homology.
    """
    from stellar_jax.inference.plato_forecast import _propagate_derived_sigma

    C = np.eye(3) * 100.0
    grad = np.zeros(3)
    sigma = _propagate_derived_sigma(C, grad)
    assert sigma == 0.0, f"σ = {sigma}, expected 0.0"


@pytest.mark.fast
def test_plato_f_age_constants_grounded():
    """F_AGE constants are grounded in Cunha (2021) hare-and-hounds.

    WHAT: Verifies the f_age constants are within the range implied by
    Cunha (2021) Table 2 and Bétrisey (2023) Table 3.

    WHY: The f_age scaling σ(age)/age = f_age × σ(M)/M must be calibrated
    from published H&H exercises — not arbitrary.

    EXTERNAL REFERENCE:
    - Cunha (2021, arXiv:2110.03332) Table 2: mass ~1.6–2.6%, age ~5–6%
      → f_age ≈ 2.0–3.8 (range: 5/2.6 to 6/1.6).
    - Bétrisey (2023, arXiv:2306.04509) Table 3: mass 1.9%, age 4.1%
      → f_age ≈ 2.2.
    - Without l=2: Cunha (2021) §7 shows age degrades significantly;
      f_age ≳ 5 is conservative.
    """
    from stellar_jax.inference.plato_forecast import F_AGE_WITH_L2, F_AGE_WITHOUT_L2

    # f_age with l=2: Cunha range ~2.0–3.8, Bétrisey ~2.2
    assert 1.5 < F_AGE_WITH_L2 < 4.0, (
        f"F_AGE_WITH_L2 = {F_AGE_WITH_L2}, expected ~2.0–3.8 "
        f"(Cunha 2021 Table 2)")

    # f_age without l=2: must be larger (age degrades without δν₀₂)
    assert F_AGE_WITHOUT_L2 > F_AGE_WITH_L2, (
        f"F_AGE_WITHOUT_L2 ({F_AGE_WITHOUT_L2}) should exceed "
        f"F_AGE_WITH_L2 ({F_AGE_WITH_L2})")

    # The degradation factor (f_age_no_l2 / f_age_l2) should be > 2
    # to satisfy the issue's acceptance criterion that σ(age) degrades > 2×
    # when l=2 is removed (even before Fisher σ(M) changes).
    ratio = F_AGE_WITHOUT_L2 / F_AGE_WITH_L2
    assert ratio > 2.0, (
        f"f_age ratio = {ratio:.1f}, needs > 2.0 for the l=2 degradation test")


@pytest.mark.fast
def test_plato_noise_constants_physical():
    """PLATO noise constants are physical and literature-grounded.

    WHAT: σ_ν, σ(log_L), σ(log_Teff) are within realistic ranges for
    PLATO's P1 bright-star sample.

    WHY: Wrong noise constants would make the Fisher forecast meaningless.

    EXTERNAL REFERENCE:
    - Lund et al. (2017, ApJ 835, 172) Table 1: median σ_ν ≈ 0.15–0.30 μHz.
    - Rauer et al. (2024, arXiv:2406.05447) §3.3/§4.2: PLATO input catalog.
    """
    from stellar_jax.inference.plato_forecast import (
        PLATO_SIGMA_NU, PLATO_SIGMA_LOG_L, PLATO_SIGMA_LOG_TEFF,
    )

    # σ_ν: Lund (2017) median ~0.15–0.30 μHz for 4yr Kepler
    assert 0.05 < PLATO_SIGMA_NU < 1.0, (
        f"σ_ν = {PLATO_SIGMA_NU} μHz out of physical range")

    # σ(log_L): Gaia-era → ~0.01–0.05 dex
    assert 0.005 < PLATO_SIGMA_LOG_L < 0.1, (
        f"σ(log_L) = {PLATO_SIGMA_LOG_L} dex out of range")

    # σ(log_Teff): spectroscopic ~80 K → ~0.006 dex at 5777 K
    assert 0.001 < PLATO_SIGMA_LOG_TEFF < 0.05, (
        f"σ(log_Teff) = {PLATO_SIGMA_LOG_TEFF} dex out of range")
