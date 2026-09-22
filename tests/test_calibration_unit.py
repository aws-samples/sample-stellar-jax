"""Fast isolated unit tests for the calibration/ module (issue #500).

Each test exercises an extracted function in isolation with synthetic/tiny
inputs — no evolve_star call, no heavy JIT compilation.

These are refactor-guard tests: they verify the extraction didn't break the
module's contract (correct imports, correct signatures, correct numerical
behavior on simple inputs).

Per the L2 design doc §5, all tests should run in <5s.
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# §1. lagrange_interp_3pt
# ===========================================================================

@pytest.mark.fast
class TestLagrangeInterp3pt:
    """Test the shared quadratic Lagrange interpolation utility."""

    def test_exact_quadratic(self):
        """Exact reconstruction of a known quadratic y = t² + 2t + 1."""
        import jax.numpy as jnp
        from stellar_jax.calibration.interpolation import lagrange_interp_3pt

        # Knots
        t0, t1, t2 = 1.0, 2.0, 3.0
        y0 = t0**2 + 2*t0 + 1  # 4
        y1 = t1**2 + 2*t1 + 1  # 9
        y2 = t2**2 + 2*t2 + 1  # 16

        # Evaluate at t=2.5 → expected: 2.5² + 2*2.5 + 1 = 12.25
        result = float(lagrange_interp_3pt(t0, t1, t2, y0, y1, y2, 2.5))
        np.testing.assert_allclose(result, 12.25, atol=1e-12)

    def test_at_knots(self):
        """Interpolant equals the knot values at the knot positions."""
        import jax.numpy as jnp
        from stellar_jax.calibration.interpolation import lagrange_interp_3pt

        t0, t1, t2 = 0.0, 1.0, 3.0
        y0, y1, y2 = 5.0, 7.0, -1.0

        assert abs(float(lagrange_interp_3pt(t0, t1, t2, y0, y1, y2, t0)) - y0) < 1e-12
        assert abs(float(lagrange_interp_3pt(t0, t1, t2, y0, y1, y2, t1)) - y1) < 1e-12
        assert abs(float(lagrange_interp_3pt(t0, t1, t2, y0, y1, y2, t2)) - y2) < 1e-12

    def test_jax_differentiable(self):
        """Verify the interpolation is differentiable w.r.t. t_target."""
        import jax
        import jax.numpy as jnp
        from stellar_jax.calibration.interpolation import lagrange_interp_3pt

        jax.config.update('jax_enable_x64', True)

        def f(t):
            return lagrange_interp_3pt(1.0, 2.0, 3.0, 4.0, 9.0, 16.0, t)

        grad_val = float(jax.grad(f)(jnp.float64(2.5)))
        # Analytical derivative of t² + 2t + 1 at t=2.5 is 2*2.5 + 2 = 7.0
        np.testing.assert_allclose(grad_val, 7.0, atol=1e-10)


# ===========================================================================
# §2. _newton_raphson_2d
# ===========================================================================

@pytest.mark.fast
class TestNewtonRaphson2D:
    """Test the shared Newton-Raphson 2D optimizer."""

    def test_convergence_simple(self):
        """Converges on a simple quadratic system."""
        from stellar_jax.calibration.solar import _newton_raphson_2d

        # f(x, y) = (x² - 1, y² - 4) → root at (1, 2)
        def residual_fn(x, y):
            return (x**2 - 1.0, y**2 - 4.0)

        result = _newton_raphson_2d(
            residual_fn,
            x0=(0.5, 1.0),
            bounds=((0.01, 10.0), (0.01, 10.0)),
            tol=1e-10, max_iter=20,
            fd_scale_alpha=1.0, fd_scale_Y=1.0,
            fd_min_alpha=1e-6, fd_max_alpha=0.1,
            fd_min_Y=1e-6, fd_max_Y=0.1,
            max_step_alpha=5.0, max_step_Y=5.0,
            label="Test"
        )

        assert result['converged']
        np.testing.assert_allclose(result['alpha'], 1.0, atol=1e-8)
        np.testing.assert_allclose(result['Y0'], 2.0, atol=1e-8)

    def test_bounds_respected(self):
        """Solution stays within specified bounds."""
        from stellar_jax.calibration.solar import _newton_raphson_2d

        # System with root at (-5, -5) — outside bounds [1, 3] × [1, 3]
        def residual_fn(x, y):
            return (x + 5.0, y + 5.0)

        result = _newton_raphson_2d(
            residual_fn,
            x0=(2.0, 2.0),
            bounds=((1.0, 3.0), (1.0, 3.0)),
            tol=1e-10, max_iter=5,
            label="Bounds test"
        )

        # Should NOT converge (root is outside bounds), but should stay in bounds
        assert result['alpha'] >= 1.0
        assert result['alpha'] <= 3.0
        assert result['Y0'] >= 1.0
        assert result['Y0'] <= 3.0

    def test_singular_jacobian_fallback(self):
        """Doesn't crash on near-singular Jacobian (uses gradient descent)."""
        from stellar_jax.calibration.solar import _newton_raphson_2d

        # A system where the Jacobian is ill-conditioned at the start
        call_count = [0]
        def residual_fn(x, y):
            call_count[0] += 1
            # Near-degenerate: both residuals depend only on x
            return (x - 1.0, x - 1.0 + 1e-15 * (y - 2.0))

        result = _newton_raphson_2d(
            residual_fn,
            x0=(0.5, 1.5),
            bounds=((0.01, 5.0), (0.01, 5.0)),
            tol=1e-6, max_iter=10,
            fd_scale_alpha=1.0, fd_scale_Y=1.0,
            fd_min_alpha=0.01, fd_max_alpha=0.1,
            fd_min_Y=0.01, fd_max_Y=0.1,
            max_step_alpha=2.0, max_step_Y=2.0,
            label="Singular test"
        )
        # Should not crash; may or may not converge
        assert 'alpha' in result
        assert 'Y0' in result


# ===========================================================================
# §2b. DRY verification — all calibrators use _newton_raphson_2d
# ===========================================================================

@pytest.mark.fast
class TestNewtonRaphson2DDRY:
    """Verify that all three calibrators use the shared optimizer (DRY)."""

    def test_all_calibrators_use_shared_utility(self):
        """solar_calibrate, solar_calibrate_with_zprofile, and
        solar_calibrate_henyey all delegate to _newton_raphson_2d."""
        import inspect
        from stellar_jax.calibration import solar

        # Check that _newton_raphson_2d is called in each calibrator's source
        for fn_name in ('solar_calibrate', 'solar_calibrate_with_zprofile',
                        'solar_calibrate_henyey'):
            fn = getattr(solar, fn_name)
            source = inspect.getsource(fn)
            assert '_newton_raphson_2d' in source, (
                f"{fn_name} does not use the shared _newton_raphson_2d utility "
                f"(DRY violation per L2 design doc §2.1)"
            )


# ===========================================================================
# §3. Module import and API contract tests
# ===========================================================================

@pytest.mark.fast
class TestCalibrationImports:
    """Verify the calibration module imports and exports correctly."""

    def test_package_imports(self):
        """All public API symbols are importable from calibration."""
        import stellar_jax.calibration as calibration
        expected = [
            'evolve_solar', 'solar_residual', 'solar_residual_henyey',
            'solar_residual_with_zprofile', 'solar_calibrate',
            'solar_calibrate_henyey', 'solar_calibrate_with_zprofile',
            'compare_model_s', 'compare_mesa', 'read_mesa_history',
            'evolve_star_diagnostic', 'plot_kippenhahn', 'kippenhahn_diagram',
            'lagrange_interp_3pt',
        ]
        for name in expected:
            assert hasattr(calibration, name), f"Missing export: {name}"

    def test_backward_compat_evolution(self):
        """evolution.py still exports all moved functions."""
        import stellar_jax.evolution as evolution
        moved = [
            'evolve_solar', 'solar_residual', 'solar_calibrate',
            'solar_residual_with_zprofile', 'solar_calibrate_with_zprofile',
            'compare_model_s', 'evolve_star_diagnostic', 'plot_kippenhahn',
            'kippenhahn_diagram', 'read_mesa_history', 'compare_mesa',
        ]
        for name in moved:
            assert hasattr(evolution, name), f"evolution.{name} missing (backward compat broken)"

    def test_backward_compat_stellar(self):
        """stellar.py facade still exports all moved functions."""
        import stellar_jax.stellar as stellar
        facade_exports = [
            'evolve_solar', 'solar_residual', 'solar_calibrate',
            'solar_residual_with_zprofile', 'solar_calibrate_with_zprofile',
            'compare_model_s', 'evolve_star_diagnostic', 'plot_kippenhahn',
            'kippenhahn_diagram', 'read_mesa_history', 'compare_mesa',
        ]
        for name in facade_exports:
            assert hasattr(stellar, name), f"stellar.{name} missing (facade broken)"


# ===========================================================================
# §3b. solar_residual differentiability contract (L2 design doc §5.4)
# ===========================================================================

@pytest.mark.fast
class TestSolarResidualDifferentiable:
    """Verify solar_residual's differentiable contract is intact.

    NOTE: Full jax.grad through evolve_solar triggers lax.scan compilation
    (~10 min even at max_steps=5 — trip-count-independent). That path is
    already tested by the golden-master gradient test (test_grad_logL_dM_golden).
    Here we verify the lighter contract: the Lagrange interpolation layer
    on top of evolve_solar is differentiable (it's the part WE extracted).
    """

    def test_lagrange_layer_differentiable(self):
        """The Lagrange interpolation layer (our extraction) is differentiable."""
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        from stellar_jax.calibration.interpolation import lagrange_interp_3pt

        # Simulate the interpolation layer of solar_residual:
        # given arrays of ages and log_L, compute the interpolated value and
        # differentiate w.r.t. an input that flows through the Lagrange weights.
        def interpolated_logL(alpha):
            # Synthetic evolve_solar output (alpha affects the values)
            ages = jnp.array([3.5e9, 4.5e9, 5.5e9])
            log_L = jnp.array([-0.01 + 0.1 * alpha,
                               0.005 + 0.05 * alpha,
                               0.02 + 0.01 * alpha])
            t_target = 4.57e9
            return lagrange_interp_3pt(ages[0], ages[1], ages[2],
                                       log_L[0], log_L[1], log_L[2],
                                       t_target)

        alpha = jnp.float64(2.0)
        grad_val = jax.grad(interpolated_logL)(alpha)
        assert jnp.isfinite(grad_val), f"grad is not finite: {grad_val}"
        assert abs(float(grad_val)) > 0.0, "grad is zero"


# ===========================================================================
# §3c. compare_mesa missing file graceful (L2 design doc §5.5)
# ===========================================================================

@pytest.mark.fast
class TestCompareMesaMissingFile:
    """Verify compare_mesa handles missing data gracefully."""

    def test_missing_dir_returns_empty(self):
        """compare_mesa with a non-existent mesa_dir returns empty results."""
        from stellar_jax.calibration.comparison import compare_mesa
        # With a bogus directory, no MESA files are found — should warn + skip
        result = compare_mesa(masses=(1.0,), mesa_dir='/nonexistent/path',
                              max_steps=5)
        # No files found → empty dict (no evolve_star call triggered)
        assert result == {}


# ===========================================================================
# §4. read_mesa_history format test
# ===========================================================================

@pytest.mark.fast
class TestReadMesaHistory:
    """Test MESA history file parsing."""

    def test_parse_synthetic(self, tmp_path):
        """Parse a synthetic MESA history.data file."""
        from stellar_jax.calibration.comparison import read_mesa_history

        # MESA format: 6 header lines, then data
        content = """# header line 1
# header line 2
# header line 3
# header line 4
# header line 5
star_age log_L log_Teff center_h1 log_R
0.0 -0.010 3.750 0.700 -0.050
1.0e8 0.001 3.752 0.690 -0.048
5.0e8 0.010 3.755 0.650 -0.045
"""
        fpath = tmp_path / "history.data"
        fpath.write_text(content)

        cols = read_mesa_history(str(fpath))
        assert 'star_age' in cols
        assert 'log_L' in cols
        assert 'center_h1' in cols
        assert len(cols['star_age']) == 3
        np.testing.assert_allclose(cols['center_h1'][0], 0.700)


# ===========================================================================
# §5. Structure archival verification
# ===========================================================================

@pytest.mark.fast
class TestStructureArchival:
    """Verify structure.py archival is correct."""

    def test_archive_exists(self):
        """The archival copy exists."""
        archive_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'archive', 'structure_shooting.py'
        )
        assert os.path.isfile(archive_path)

    def test_structure_still_importable(self):
        """structure.py still works (backward compat)."""
        import stellar_jax.structure as structure
        assert hasattr(structure, 'initial_guess')
        assert hasattr(structure, 'shoot_xprofile')
        assert hasattr(structure, 'newton_solve_xprofile')
        assert hasattr(structure, 'zams_initial_model')

    def test_deprecation_notice(self):
        """structure.py has a deprecation notice."""
        structure_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'structure.py'
        )
        with open(structure_path) as f:
            header = f.read(500)
        assert 'LEGACY' in header or 'legacy' in header
