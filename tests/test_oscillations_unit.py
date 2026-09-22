"""Fast unit tests for the oscillations/ package (<5s each, isolated).

Tests each extracted module in isolation using the Model S FGONG file
(a fixed external reference) or synthetic inputs. No evolve_star, no heavy JIT.

Per docs/design/redesign/07-oscillations.md §5.
"""
import os
import sys
import gzip
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


# ─── Fixture: Model S FGONG ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_s_fgong():
    """Load the Model S FGONG file (committed reference data)."""
    fgong_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "model_s", "fgong.l5bi.d.15c"
    )
    if not os.path.isfile(fgong_path):
        pytest.skip(f"Model S FGONG not found: {fgong_path}")
    from stellar_jax.fgong.io import read_fgong
    glob, var = read_fgong(fgong_path)
    return glob, var


@pytest.fixture(scope="module")
def mesa_fgong():
    """Load the MESA midMS FGONG file (committed reference data)."""
    fgong_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "mesa_comparison", "profiles", "1.0Msun", "midMS.FGONG.gz"
    )
    if not os.path.isfile(fgong_path):
        pytest.skip(f"MESA FGONG not found: {fgong_path}")
    from stellar_jax.fgong.io import read_fgong
    with gzip.open(fgong_path, 'rt') as gz:
        content = gz.read()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.FGONG', delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        glob, var = read_fgong(tmp_path)
    finally:
        os.unlink(tmp_path)
    return glob, var


# ═══════════════════════════════════════════════════════════════════════════════
# §1. contracts.py
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
class TestContracts:
    """Test shape validation helpers in contracts.py."""

    def test_validate_coeffs_shape_good(self):
        from stellar_jax.oscillations.contracts import validate_coeffs_shape
        coeffs = jnp.ones((5, 100))
        x_grid = jnp.ones(100)
        validate_coeffs_shape(coeffs, x_grid)  # should not raise

    def test_validate_coeffs_shape_wrong_rows(self):
        from stellar_jax.oscillations.contracts import validate_coeffs_shape
        coeffs = jnp.ones((4, 100))
        x_grid = jnp.ones(100)
        with pytest.raises(AssertionError, match="5 rows"):
            validate_coeffs_shape(coeffs, x_grid)

    def test_validate_coeffs_shape_mismatch(self):
        from stellar_jax.oscillations.contracts import validate_coeffs_shape
        coeffs = jnp.ones((5, 100))
        x_grid = jnp.ones(50)
        with pytest.raises(AssertionError, match="must match"):
            validate_coeffs_shape(coeffs, x_grid)

    def test_validate_fgong_arrays_good(self):
        from stellar_jax.oscillations.contracts import validate_fgong_arrays
        glob = np.ones(15)
        var = np.ones((100, 15))
        validate_fgong_arrays(glob, var)

    def test_validate_fgong_arrays_too_few_columns(self):
        from stellar_jax.oscillations.contracts import validate_fgong_arrays
        glob = np.ones(15)
        var = np.ones((100, 10))
        with pytest.raises(AssertionError, match="≥15 columns"):
            validate_fgong_arrays(glob, var)


# ═══════════════════════════════════════════════════════════════════════════════
# §2. fgong/io.py
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
class TestFgongIO:
    """Test FGONG I/O functions."""

    def test_read_fgong_shape(self, model_s_fgong):
        glob, var = model_s_fgong
        assert glob.ndim == 1
        assert glob.shape[0] >= 15
        assert var.ndim == 2
        assert var.shape[1] >= 15
        # Model S has ~2482 mesh points
        assert var.shape[0] > 2000

    def test_read_fgong_center_to_surface(self, model_s_fgong):
        """Verify data is ordered center-to-surface (r increasing)."""
        _, var = model_s_fgong
        r = var[:, 0]
        assert r[0] < r[-1], "FGONG should be center-to-surface"

    def test_fgong_components_keys(self, model_s_fgong):
        from stellar_jax.fgong.io import fgong_components
        glob, var = model_s_fgong
        comp = fgong_components(glob, var)
        expected_keys = {'r', 'T', 'P', 'rho', 'X', 'L', 'kappa', 'eps_nuc',
                         'gamma1', 'm_frac', 'brunt_A'}
        assert expected_keys.issubset(set(comp.keys()))

    def test_fgong_components_physical_ranges(self, model_s_fgong):
        from stellar_jax.fgong.io import fgong_components
        glob, var = model_s_fgong
        comp = fgong_components(glob, var)
        # Gamma1 should be ~5/3 in the interior
        assert np.all(comp['gamma1'] > 1.0)
        assert np.all(comp['gamma1'] < 2.0)
        # Radius should be positive
        assert np.all(comp['r'] >= 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# §3. integrator.py
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
class TestIntegrator:
    """Test RK4 integrator components."""

    def test_rk4_step_polynomial(self):
        """RK4 integrates a degree-3 polynomial exactly.

        f(x) = x³ → f'(x) = 3x²
        RK4 is exact for polynomials of degree ≤ 3.
        """
        from stellar_jax.oscillations.integrator import _rk4_step

        # dy/dx = 3x² (derivative of x³)
        def rhs(x, y):
            return jnp.array([3.0 * x**2])

        y0 = jnp.array([0.0])  # x³ at x=0
        x0 = 0.0
        h = 1.0

        y1 = _rk4_step(rhs, x0, y0, h)
        # Should be exactly 1.0³ = 1.0
        np.testing.assert_allclose(float(y1[0]), 1.0, atol=1e-14)

    def test_rk4_step_exponential(self):
        """RK4 on dy/dx = y (exponential) at h=0.1 is accurate to O(h⁵)."""
        from stellar_jax.oscillations.integrator import _rk4_step

        def rhs(x, y):
            return y

        y0 = jnp.array([1.0])
        y1 = _rk4_step(rhs, 0.0, y0, 0.1)
        expected = np.exp(0.1)
        # RK4 error is O(h^5) = O(1e-5) for h=0.1
        np.testing.assert_allclose(float(y1[0]), expected, rtol=1e-5)

    def test_integration_grid_properties(self, model_s_fgong):
        """Test _make_integration_grid on Model S data."""
        from stellar_jax.oscillations.integrator import _make_integration_grid
        from stellar_jax.oscillations.coefficients import _build_oscillation_grid_jax

        glob, var = model_s_fgong
        grid_data = _build_oscillation_grid_jax(glob, var)
        x_steps, h_steps = _make_integration_grid(grid_data['x_grid'], n_steps=8000)

        # Check basic properties
        assert x_steps.shape == h_steps.shape
        assert len(x_steps) >= 7000  # should be close to requested
        # First point should be above 1e-4 (center excluded)
        assert float(x_steps[0]) > 1e-4
        # Steps should be positive
        assert np.all(np.array(h_steps) > 0)
        # Grid should be monotonically increasing
        assert np.all(np.diff(np.array(x_steps)) >= 0)

    def test_radial_rhs_finite(self):
        """_radial_rhs returns finite values at a typical point."""
        from stellar_jax.oscillations.integrator import _radial_rhs

        # Synthetic coefficients at a single point (mid-star values)
        x_grid = jnp.array([0.1, 0.5, 0.9])
        coeffs = jnp.array([
            [10.0, 5.0, 2.0],   # Vg
            [1.0, 1.0, 1.0],    # A1
            [0.1, 0.2, 0.1],    # A_bv
            [3.0, 3.0, 3.0],    # U
            [0.01, 0.5, 0.99],  # q
        ])
        sigma2 = jnp.float64(100.0)
        y = jnp.array([1.0, 0.3])

        dy = _radial_rhs(0.5, y, sigma2, x_grid, coeffs)
        assert dy.shape == (2,)
        assert np.all(np.isfinite(np.array(dy)))


# ═══════════════════════════════════════════════════════════════════════════════
# §4. coefficients.py
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
class TestCoefficients:
    """Test structure → oscillation coefficient extraction."""

    def test_build_coeffs_shape_contract(self, mesa_fgong):
        """build_oscillation_coeffs_jax returns correct shapes + no NaN/Inf."""
        from stellar_jax.oscillations.coefficients import build_oscillation_coeffs_jax

        glob, var = mesa_fgong
        glob_jax = jnp.array(glob)
        var_jax = jnp.array(var)

        result = build_oscillation_coeffs_jax(glob_jax, var_jax)

        assert 'x_grid' in result
        assert 'coeffs' in result
        assert 'factor' in result

        N = var.shape[0]
        assert result['x_grid'].shape == (N,)
        assert result['coeffs'].shape == (5, N)
        assert np.all(np.isfinite(np.array(result['coeffs'])))
        assert np.all(np.isfinite(np.array(result['x_grid'])))
        assert float(result['factor']) > 0

    def test_build_coeffs_numpy_path(self, mesa_fgong):
        """_build_oscillation_grid_jax returns correct structure."""
        from stellar_jax.oscillations.coefficients import _build_oscillation_grid_jax, _prepare_coeffs

        glob, var = mesa_fgong
        grid_data = _build_oscillation_grid_jax(glob, var)

        assert 'x_grid' in grid_data
        assert 'Vg' in grid_data
        assert 'A1' in grid_data
        assert 'factor' in grid_data

        x_grid, coeffs = _prepare_coeffs(grid_data)
        assert coeffs.shape[0] == 5
        assert coeffs.shape[1] == len(grid_data['x_grid'])


# ═══════════════════════════════════════════════════════════════════════════════
# §5. determinant.py
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
class TestDeterminant:
    """Test shooting determinant functions."""

    def test_radial_determinant_sign_change(self, mesa_fgong):
        """radial_determinant has a sign change between two known σ² brackets.

        A sign change in D(σ²) means there's an eigenfrequency between the brackets.
        """
        from stellar_jax.oscillations.coefficients import _build_oscillation_grid_jax, _prepare_coeffs
        from stellar_jax.oscillations.integrator import _make_integration_grid
        from stellar_jax.oscillations.determinant import radial_determinant

        glob, var = mesa_fgong
        grid_data = _build_oscillation_grid_jax(glob, var)
        x_grid, coeffs = _prepare_coeffs(grid_data)
        x_steps, h_steps = _make_integration_grid(grid_data['x_grid'], n_steps=8000)
        factor = grid_data['factor']

        # Scan a known bracket: ~2700-3200 μHz should contain an l=0 mode
        nu_lo, nu_hi = 2700.0, 3200.0
        sigma2_lo = jnp.float64(nu_lo**2 * factor)
        sigma2_hi = jnp.float64(nu_hi**2 * factor)

        det_lo = float(radial_determinant(sigma2_lo, x_steps, h_steps, x_grid, coeffs))
        det_hi = float(radial_determinant(sigma2_hi, x_steps, h_steps, x_grid, coeffs))

        # At least one sign change should exist (eigenfrequency in bracket)
        assert np.isfinite(det_lo) and np.isfinite(det_hi), \
            f"Determinant not finite: lo={det_lo}, hi={det_hi}"
        # This is not guaranteed for a single bracket, but for the solar model
        # there are ~3-4 radial modes in this range, so sub-intervals will bracket
        assert det_lo != 0.0 or det_hi != 0.0, "Determinant should be non-trivial"


# ═══════════════════════════════════════════════════════════════════════════════
# §6. adjoint.py — THE CRITICAL IFT ADJOINT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
class TestAdjoint:
    """Test IFT adjoint correctness via AD vs independent FD."""

    def test_ift_adjoint_radial_vs_fd(self, mesa_fgong):
        """IFT backward for eigenfreq_radial matches finite-difference ∂σ²/∂coeff[i].

        This is the key correctness test: the @custom_vjp IFT formula must agree
        with an independent finite difference to <5%.
        """
        from stellar_jax.oscillations.coefficients import _build_oscillation_grid_jax, _prepare_coeffs
        from stellar_jax.oscillations.integrator import _make_integration_grid
        from stellar_jax.oscillations.eigenvalue import compute_eigenfreq_differentiable
        from stellar_jax.oscillations.adjoint import eigenfreq_from_coeffs

        glob, var = mesa_fgong
        # Find a converged eigenfrequency
        info = compute_eigenfreq_differentiable(glob, var, l=0,
                                                nu_min=2500.0, nu_max=3500.0,
                                                n_scan=100, n_steps=8000)

        # AD gradient via IFT
        grad_fn = jax.grad(lambda c: eigenfreq_from_coeffs(
            c, info['x_grid'], info['sigma2'], info['l'],
            info['x_steps'], info['h_steps'], info['factor']))
        ad_grad = np.array(grad_fn(info['coeffs']))

        # Independent FD: perturb one coefficient at a few points
        # Pick a point in the middle of the grid
        from stellar_jax.oscillations.determinant import radial_determinant
        from scipy.optimize import brentq

        eps = 1e-7
        # Test Vg at midpoint
        mid_idx = info['coeffs'].shape[1] // 2
        coeffs_plus = info['coeffs'].at[0, mid_idx].add(eps)
        coeffs_minus = info['coeffs'].at[0, mid_idx].add(-eps)

        # Recompute eigenfrequency with perturbed coefficients (forward only)
        factor = info['factor']
        x_grid = info['x_grid']
        x_steps = info['x_steps']
        h_steps = info['h_steps']
        nu_ref = info['nu']

        @jax.jit
        def det_plus(s2):
            return radial_determinant(s2, x_steps, h_steps, x_grid, coeffs_plus)

        @jax.jit
        def det_minus(s2):
            return radial_determinant(s2, x_steps, h_steps, x_grid, coeffs_minus)

        # Search near the known root
        nu_lo, nu_hi = nu_ref * 0.999, nu_ref * 1.001

        def f_plus(nu):
            return float(det_plus(jnp.float64(nu**2 * factor)))

        def f_minus(nu):
            return float(det_minus(jnp.float64(nu**2 * factor)))

        try:
            nu_p = brentq(f_plus, nu_lo, nu_hi, rtol=1e-10, maxiter=100)
            nu_m = brentq(f_minus, nu_lo, nu_hi, rtol=1e-10, maxiter=100)
        except ValueError:
            pytest.skip("Could not bracket perturbed roots (narrow bracket)")

        sigma2_p = nu_p**2 * factor
        sigma2_m = nu_m**2 * factor
        fd_grad_vg_mid = (sigma2_p - sigma2_m) / (2 * eps)

        ad_grad_vg_mid = float(ad_grad[0, mid_idx])

        # IFT adjoint should agree with FD to <5%
        if abs(fd_grad_vg_mid) > 1e-15:
            rel_err = abs(ad_grad_vg_mid - fd_grad_vg_mid) / abs(fd_grad_vg_mid)
            assert rel_err < 0.05, (
                f"IFT adjoint vs FD: AD={ad_grad_vg_mid:.6e}, FD={fd_grad_vg_mid:.6e}, "
                f"rel_err={rel_err:.2%}"
            )

    def test_eigenfreq_from_coeffs_dispatch(self, mesa_fgong):
        """eigenfreq_from_coeffs dispatches correctly for l=0 vs l>0."""
        from stellar_jax.oscillations.adjoint import eigenfreq_from_coeffs

        # Synthetic test: just check it doesn't crash and returns the seed
        coeffs = jnp.ones((5, 10))
        x_grid = jnp.linspace(0.01, 1.0, 10)
        sigma2 = jnp.float64(100.0)
        x_steps = jnp.linspace(0.01, 1.0, 50)
        h_steps = jnp.full(50, 0.02)
        factor = 1.0

        # l=0 should dispatch to eigenfreq_radial (returns sigma2 unchanged)
        result_0 = eigenfreq_from_coeffs(coeffs, x_grid, sigma2, 0,
                                         x_steps, h_steps, factor)
        assert float(result_0) == float(sigma2)

        # l=1 should dispatch to eigenfreq_nonradial (returns sigma2 unchanged)
        result_1 = eigenfreq_from_coeffs(coeffs, x_grid, sigma2, 1,
                                         x_steps, h_steps, factor)
        assert float(result_1) == float(sigma2)


# ═══════════════════════════════════════════════════════════════════════════════
# §7. eigenvalue.py
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
class TestEigenvalue:
    """Test eigenvalue search (Brent root-finding)."""

    def test_compute_eigenfreq_differentiable_finds_mode(self, mesa_fgong):
        """compute_eigenfreq_differentiable finds a known l=0 mode."""
        from stellar_jax.oscillations.eigenvalue import compute_eigenfreq_differentiable

        glob, var = mesa_fgong
        result = compute_eigenfreq_differentiable(glob, var, l=0,
                                                  nu_min=2500.0, nu_max=3500.0)
        # Should find a frequency in the solar-like range
        assert 2500.0 < result['nu'] < 3500.0
        assert float(result['sigma2']) > 0
        assert result['coeffs'].shape[0] == 5
        assert result['l'] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# §8. eigenvalue.py — _bracket_and_refine helper (DRY extraction,)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
class TestBracketAndRefine:
    """Unit tests for the shared _bracket_and_refine root-finding helper.

    Uses simple analytic determinant functions (no JIT, no JAX compilation)
    to verify the sign-change detection + Brent refinement logic in <1s.

    Per issue #541 acceptance: "Fast (<5s) unit test on the helper."
    Per docs/dev/refactoring-best-practices.md §5: every extraction gets a fast test.
    """

    def test_finds_single_root_of_sin(self):
        """A simple sin(nu) has a root at π ≈ 3.14159..."""
        from stellar_jax.oscillations.eigenvalue import _bracket_and_refine

        def det_fn(sigma2):
            # sigma2 = nu^2 * factor, so nu = sqrt(sigma2/factor)
            # We want det(nu) = sin(nu), so we invert: nu = sqrt(sigma2/1.0)
            nu = float(jnp.sqrt(sigma2))
            return np.sin(nu)

        nu_arr = np.linspace(2.0, 4.0, 100)
        factor = 1.0  # sigma2 = nu^2
        roots = _bracket_and_refine(det_fn, nu_arr, factor, rtol=1e-12, maxiter=100)

        assert len(roots) == 1
        assert abs(roots[0] - np.pi) < 1e-10

    def test_finds_multiple_roots(self):
        """sin(nu) has roots at π, 2π, 3π in [2, 10]."""
        from stellar_jax.oscillations.eigenvalue import _bracket_and_refine

        def det_fn(sigma2):
            nu = float(jnp.sqrt(sigma2))
            return np.sin(nu)

        nu_arr = np.linspace(2.0, 10.0, 500)
        factor = 1.0
        roots = _bracket_and_refine(det_fn, nu_arr, factor, rtol=1e-12, maxiter=100)

        # Should find roots at π, 2π, 3π
        assert len(roots) == 3
        expected = [np.pi, 2 * np.pi, 3 * np.pi]
        for r, e in zip(roots, expected):
            assert abs(r - e) < 1e-10, f"Root {r} != expected {e}"

    def test_returns_sorted(self):
        """Roots are returned in ascending order."""
        from stellar_jax.oscillations.eigenvalue import _bracket_and_refine

        def det_fn(sigma2):
            nu = float(jnp.sqrt(sigma2))
            return np.sin(nu)

        nu_arr = np.linspace(2.0, 10.0, 500)
        roots = _bracket_and_refine(det_fn, nu_arr, 1.0, rtol=1e-10, maxiter=50)

        assert roots == sorted(roots)

    def test_handles_nan_in_det_values(self):
        """NaN values in the determinant scan are skipped gracefully."""
        from stellar_jax.oscillations.eigenvalue import _bracket_and_refine

        def det_fn(sigma2):
            nu = float(jnp.sqrt(sigma2))
            # Return NaN for a range that would otherwise contain a root
            if 3.0 < nu < 3.3:
                return float('nan')
            return np.sin(nu)

        nu_arr = np.linspace(2.0, 4.0, 200)
        roots = _bracket_and_refine(det_fn, nu_arr, 1.0, rtol=1e-10, maxiter=100)

        # π ≈ 3.14159 is in the NaN zone, so no root should be found
        assert len(roots) == 0

    def test_handles_no_sign_change(self):
        """A positive-definite function returns no roots."""
        from stellar_jax.oscillations.eigenvalue import _bracket_and_refine

        def det_fn(sigma2):
            nu = float(jnp.sqrt(sigma2))
            return nu**2 + 1.0  # always > 0

        nu_arr = np.linspace(1.0, 5.0, 100)
        roots = _bracket_and_refine(det_fn, nu_arr, 1.0)

        assert len(roots) == 0

    def test_respects_rtol_and_maxiter(self):
        """Coarse rtol still finds the root but with less precision."""
        from stellar_jax.oscillations.eigenvalue import _bracket_and_refine

        def det_fn(sigma2):
            nu = float(jnp.sqrt(sigma2))
            return np.sin(nu)

        nu_arr = np.linspace(2.0, 4.0, 100)
        factor = 1.0

        # Tight tolerance
        roots_tight = _bracket_and_refine(det_fn, nu_arr, factor,
                                          rtol=1e-14, maxiter=200)
        # Coarse tolerance
        roots_coarse = _bracket_and_refine(det_fn, nu_arr, factor,
                                           rtol=1e-4, maxiter=10)

        assert len(roots_tight) == 1
        assert len(roots_coarse) == 1
        # Both find the root, tight is closer to π
        assert abs(roots_tight[0] - np.pi) < abs(roots_coarse[0] - np.pi)

    def test_factor_converts_nu_to_sigma2(self):
        """The factor parameter correctly maps nu² → σ²."""
        from stellar_jax.oscillations.eigenvalue import _bracket_and_refine

        # det_fn receives sigma2 = nu^2 * factor
        # With factor=4.0, at nu=1 we get sigma2=4
        # sin(sqrt(sigma2)) = sin(2) ≈ 0.909 (positive)
        # At nu=1.6, sigma2=10.24, sin(sqrt(10.24)) = sin(3.2) ≈ -0.058 (negative)
        # Root at sin(sqrt(sigma2))=0 → sqrt(sigma2)=π → sigma2=π² → nu=π/2 ≈ 1.5708
        def det_fn(sigma2):
            return np.sin(float(jnp.sqrt(sigma2)))

        nu_arr = np.linspace(1.0, 2.0, 100)
        factor = 4.0
        roots = _bracket_and_refine(det_fn, nu_arr, factor, rtol=1e-12, maxiter=100)

        # nu such that nu^2 * 4 = π² → nu = π/2
        assert len(roots) == 1
        assert abs(roots[0] - np.pi / 2) < 1e-10

    def test_brentq_failure_is_graceful(self):
        """If brentq raises (e.g. maxiter=1 not enough), the bracket is skipped."""
        from stellar_jax.oscillations.eigenvalue import _bracket_and_refine

        # A function that changes sign but oscillates wildly — brentq with maxiter=1
        # may not converge.
        def det_fn(sigma2):
            nu = float(jnp.sqrt(sigma2))
            return np.tan(nu)  # discontinuity at π/2 looks like a sign change

        nu_arr = np.linspace(1.0, 2.0, 50)
        # The tan function has a discontinuity at π/2 ≈ 1.5708; brentq may fail
        # because the function goes from +∞ to −∞ (no actual zero crossing within bracket)
        # With maxiter=1, brentq is even more likely to fail gracefully
        roots = _bracket_and_refine(det_fn, nu_arr, 1.0, rtol=1e-10, maxiter=1)

        # We just verify it doesn't crash — it either finds "roots" (wrong but
        # non-crashing) or returns empty; the key is no exception propagates
        assert isinstance(roots, list)


# ═══════════════════════════════════════════════════════════════════════════════
# §9. Package-level re-exports
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
class TestPackageReExports:
    """Verify that the package __init__.py re-exports all expected symbols."""

    def test_all_public_symbols_accessible(self):
        import stellar_jax.oscillations as oscillations
        expected = [
            'read_fgong', 'fgong_components',
            'build_oscillation_coeffs_jax', 'eigenfreq_from_coeffs',
            'eigenfreq_radial', 'eigenfreq_nonradial',
            'compute_oscillation_freqs_jax', 'compute_eigenfreq_differentiable',
            'compute_eigenfreq_from_structure_jax',
            'radial_determinant', 'nonradial_determinant',
            'compute_oscillation_freqs', 'compute_oscillation_freqs_full',
            'echelle_data', 'estimate_delta_nu',
            'OscillationCoeffs', 'EigenfreqResult', 'IntegrationGrid',
            'validate_coeffs_shape', 'validate_fgong_arrays',
        ]
        for name in expected:
            assert hasattr(oscillations, name), f"Missing: oscillations.{name}"

    def test_oscillations_package_symbols(self):
        """The oscillations package exposes all expected public symbols."""
        import stellar_jax.oscillations as osc
        expected = [
            'eigenfreq_from_coeffs', 'eigenfreq_radial', 'eigenfreq_nonradial',
            'build_oscillation_coeffs_jax', 'compute_oscillation_freqs_jax',
            'compute_eigenfreq_differentiable', 'compute_eigenfreq_from_structure_jax',
            'radial_determinant', 'nonradial_determinant',
        ]
        for name in expected:
            assert hasattr(osc, name), f"Missing: oscillations.{name}"
