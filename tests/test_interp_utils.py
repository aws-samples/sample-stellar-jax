"""Fast unit tests for _locate_1d and _locate_4d grid helpers.

Tests the DRY extraction (issue #533) — validates index/fraction correctness
against known inputs. All tests <1s each; no evolve_star, no heavy JIT.
"""
import pytest
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from stellar_jax.microphysics.interp_utils import _locate_1d, _locate_4d


@pytest.mark.fast
class TestLocate1d:
    """Unit tests for _locate_1d grid-cell locator."""

    def test_interior_point(self):
        """Point in the middle of a cell returns correct index and fraction."""
        grid = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        pt, idx, frac = _locate_1d(grid, 1.5)
        assert int(idx) == 1
        assert abs(float(frac) - 0.5) < 1e-14

    def test_at_grid_point(self):
        """Point exactly on a grid node returns fraction 0 or 1."""
        grid = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        pt, idx, frac = _locate_1d(grid, 2.0)
        # At grid[2], could be idx=1,frac=1.0 or idx=2,frac=0.0
        # Both are valid; just check consistency.
        val = float(grid[int(idx)]) + float(frac) * float(grid[int(idx) + 1] - grid[int(idx)])
        assert abs(val - 2.0) < 1e-14

    def test_below_range_clamps(self):
        """Point below grid[0] is clamped; index=0, fraction=0."""
        grid = jnp.array([1.0, 2.0, 3.0, 4.0])
        pt, idx, frac = _locate_1d(grid, -5.0)
        assert float(pt) == 1.0
        assert int(idx) == 0
        assert float(frac) == 0.0

    def test_above_range_clamps(self):
        """Point above grid[-1] is clamped; index=n-2, fraction=1."""
        grid = jnp.array([1.0, 2.0, 3.0, 4.0])
        pt, idx, frac = _locate_1d(grid, 10.0)
        assert float(pt) == 4.0
        assert int(idx) == 2  # len(grid)-2 = 2
        assert float(frac) == 1.0

    def test_first_cell(self):
        """Point in first cell works correctly."""
        grid = jnp.array([0.0, 1.0, 2.0, 3.0])
        pt, idx, frac = _locate_1d(grid, 0.25)
        assert int(idx) == 0
        assert abs(float(frac) - 0.25) < 1e-14

    def test_last_cell(self):
        """Point in last cell works correctly."""
        grid = jnp.array([0.0, 1.0, 2.0, 3.0])
        pt, idx, frac = _locate_1d(grid, 2.75)
        assert int(idx) == 2
        assert abs(float(frac) - 0.75) < 1e-14

    def test_non_uniform_grid(self):
        """Works correctly on a non-uniform grid."""
        grid = jnp.array([0.0, 0.1, 0.5, 2.0, 10.0])
        pt, idx, frac = _locate_1d(grid, 1.25)
        assert int(idx) == 2  # cell [0.5, 2.0]
        expected_frac = (1.25 - 0.5) / (2.0 - 0.5)
        assert abs(float(frac) - expected_frac) < 1e-14

    def test_differentiable(self):
        """Gradient flows through _locate_1d fraction."""
        grid = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        # d(frac)/d(point) at point=1.5 in cell [1,2] is 1/(2-1) = 1.0
        grad_fn = jax.grad(lambda p: _locate_1d(grid, p)[2])
        g = grad_fn(1.5)
        assert abs(float(g) - 1.0) < 1e-12


@pytest.mark.fast
class TestLocate4d:
    """Unit tests for _locate_4d 4D grid-cell locator."""

    def test_interior_point(self):
        """All four dimensions are located correctly in interior."""
        g0 = jnp.array([0.0, 1.0, 2.0, 3.0])
        g1 = jnp.array([10.0, 20.0, 30.0])
        g2 = jnp.array([100.0, 200.0, 300.0, 400.0, 500.0])
        g3 = jnp.array([-1.0, 0.0, 1.0, 2.0])

        (i0, i1, i2, i3), (t0, t1, t2, t3) = _locate_4d(
            (g0, g1, g2, g3), (1.5, 15.0, 350.0, 0.5))

        assert int(i0) == 1
        assert abs(float(t0) - 0.5) < 1e-14
        assert int(i1) == 0
        assert abs(float(t1) - 0.5) < 1e-14
        assert int(i2) == 2
        assert abs(float(t2) - 0.5) < 1e-14
        assert int(i3) == 1
        assert abs(float(t3) - 0.5) < 1e-14

    def test_clamping_all_dims(self):
        """Out-of-range points are clamped in all dimensions."""
        g0 = jnp.array([0.0, 1.0, 2.0])
        g1 = jnp.array([0.0, 1.0, 2.0])
        g2 = jnp.array([0.0, 1.0, 2.0])
        g3 = jnp.array([0.0, 1.0, 2.0])

        (i0, i1, i2, i3), (t0, t1, t2, t3) = _locate_4d(
            (g0, g1, g2, g3), (-5.0, -5.0, -5.0, -5.0))

        # All clamped to lower bound
        for idx in (i0, i1, i2, i3):
            assert int(idx) == 0
        for frac in (t0, t1, t2, t3):
            assert float(frac) == 0.0

    def test_matches_inline_pattern(self):
        """Result matches the hand-inlined clamp/searchsorted/fraction pattern."""
        # Use the actual OPAL grids to verify against the original pattern
        from stellar_jax.microphysics.opacity import OPAL_X, OPAL_Z, OPAL_LOGT, OPAL_LOGR

        X, Z, logT, logR = 0.72, 0.015, 6.3, -1.5

        # Original pattern (inlined):
        Xc = jnp.clip(X, OPAL_X[0], OPAL_X[-1])
        Zc = jnp.clip(Z, OPAL_Z[0], OPAL_Z[-1])
        lt = jnp.clip(logT, OPAL_LOGT[0], OPAL_LOGT[-1])
        lr = jnp.clip(logR, OPAL_LOGR[0], OPAL_LOGR[-1])

        nX = OPAL_X.shape[0] - 2
        nZ = OPAL_Z.shape[0] - 2
        nT = OPAL_LOGT.shape[0] - 2
        nR = OPAL_LOGR.shape[0] - 2

        ix_ref = jnp.clip(jnp.searchsorted(OPAL_X, Xc) - 1, 0, nX)
        iz_ref = jnp.clip(jnp.searchsorted(OPAL_Z, Zc) - 1, 0, nZ)
        it_ref = jnp.clip(jnp.searchsorted(OPAL_LOGT, lt) - 1, 0, nT)
        ir_ref = jnp.clip(jnp.searchsorted(OPAL_LOGR, lr) - 1, 0, nR)

        tx_ref = jnp.clip((Xc - OPAL_X[ix_ref]) / (OPAL_X[ix_ref+1] - OPAL_X[ix_ref]), 0., 1.)
        tz_ref = jnp.clip((Zc - OPAL_Z[iz_ref]) / (OPAL_Z[iz_ref+1] - OPAL_Z[iz_ref]), 0., 1.)
        tt_ref = jnp.clip((lt - OPAL_LOGT[it_ref]) / (OPAL_LOGT[it_ref+1] - OPAL_LOGT[it_ref]), 0., 1.)
        tr_ref = jnp.clip((lr - OPAL_LOGR[ir_ref]) / (OPAL_LOGR[ir_ref+1] - OPAL_LOGR[ir_ref]), 0., 1.)

        # New helper:
        (ix, iz, it, ir), (tx, tz, tt, tr) = _locate_4d(
            (OPAL_X, OPAL_Z, OPAL_LOGT, OPAL_LOGR), (X, Z, logT, logR))

        # Bit-exact match
        assert int(ix) == int(ix_ref)
        assert int(iz) == int(iz_ref)
        assert int(it) == int(it_ref)
        assert int(ir) == int(ir_ref)
        assert float(tx) == float(tx_ref)
        assert float(tz) == float(tz_ref)
        assert float(tt) == float(tt_ref)
        assert float(tr) == float(tr_ref)

    def test_gradient_flows(self):
        """Gradients propagate through _locate_4d fractions."""
        g0 = jnp.array([0.0, 1.0, 2.0, 3.0])
        g1 = jnp.array([0.0, 1.0, 2.0, 3.0])
        g2 = jnp.array([0.0, 1.0, 2.0, 3.0])
        g3 = jnp.array([0.0, 1.0, 2.0, 3.0])

        # Gradient of fraction[0] wrt point[0] should be 1/(grid spacing) = 1.0
        def f(p0):
            _, fracs = _locate_4d((g0, g1, g2, g3), (p0, 1.5, 1.5, 1.5))
            return fracs[0]

        g = jax.grad(f)(1.5)
        assert abs(float(g) - 1.0) < 1e-12

    def test_eos_grids(self):
        """Verify on actual EOS grids with a solar-core test point."""
        from stellar_jax.microphysics.eos import EOS_X, EOS_Z, EOS_LOGT, EOS_LOGP

        (ix, iz, it, ip), (tx, tz, tt, tp) = _locate_4d(
            (EOS_X, EOS_Z, EOS_LOGT, EOS_LOGP), (0.34, 0.02, 7.2, 17.3))

        # Indices must be valid
        assert 0 <= int(ix) <= EOS_X.shape[0] - 2
        assert 0 <= int(iz) <= EOS_Z.shape[0] - 2
        assert 0 <= int(it) <= EOS_LOGT.shape[0] - 2
        assert 0 <= int(ip) <= EOS_LOGP.shape[0] - 2

        # Fractions must be in [0, 1]
        for frac in (tx, tz, tt, tp):
            assert 0.0 <= float(frac) <= 1.0
