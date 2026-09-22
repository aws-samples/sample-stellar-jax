"""Gradient correctness tests: analytic (jax.grad) vs finite-difference.

Issue #5: Validate ∂logL/∂M and ∂logTeff/∂α match FD within 5%.

The adaptive timestepper creates non-smooth boundaries (via jnp.clip on dt)
that make large FD perturbations unreliable. The analytic gradient through
lax.scan correctly differentiates along the smooth local path. We use small
perturbations (relative 1e-6*M for mass, 1e-4 for alpha) to stay on the same
branch while avoiding roundoff at longer evolutions (50 steps).

References:
  - Griewank & Walther (2008), Evaluating Derivatives, Ch. 8: FD step selection
    h ~ eps^(1/3) * |x| ≈ 6e-6 for f64
  - Kidger (2022), diffrax: adaptive stepping inside scan
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

import stellar_jax.stellar as stellar


# JIT compilation takes ~5 min on first call; subsequent calls reuse cache.
pytestmark = pytest.mark.timeout(1800)

MAX_STEPS = 50  # Test gradients through real evolution (burning, mixing, diffusion)


class TestGradientLogL_Mass:
    """∂log_L/∂M: analytic vs finite-difference."""

    def test_fd_match_within_5_percent(self):
        """Core requirement: analytic matches FD to < 5%."""
        M = 1.0
        dM = 1e-6 * M  # Relative perturbation per Griewank & Walther: h ~ eps^(1/3)*|x|
        f = lambda m: stellar.evolve_star(m, Z=0.014, max_steps=MAX_STEPS)['log_L'][-1]
        ad = float(jax.grad(f)(jnp.float64(M)))
        fd = (float(f(M + dM)) - float(f(M - dM))) / (2 * dM)
        rel_err = abs(ad - fd) / (abs(fd) + 1e-10)
        assert rel_err < 0.05, f"analytic={ad:.4f}, FD={fd:.4f}, err={rel_err:.3f}"

    def test_gradient_positive(self):
        """∂logL/∂M > 0: more massive stars are more luminous."""
        g = float(jax.grad(lambda m: stellar.evolve_star(m, Z=0.014, max_steps=MAX_STEPS)['log_L'][-1])(jnp.float64(1.0)))
        assert g > 0, f"Gradient should be positive, got {g}"

    def test_gradient_physical_range(self):
        """∂logL/∂M should be ~2-8 (mass-luminosity relation L~M^3.5 → dlogL/dM ~ 3.5/M)."""
        g = float(jax.grad(lambda m: stellar.evolve_star(m, Z=0.014, max_steps=MAX_STEPS)['log_L'][-1])(jnp.float64(1.0)))
        assert 0.5 < g < 10.0, f"Gradient {g} outside physical range [0.5, 10]"

    def test_not_nan(self):
        """Gradient must not be NaN."""
        g = float(jax.grad(lambda m: stellar.evolve_star(m, Z=0.014, max_steps=MAX_STEPS)['log_L'][-1])(jnp.float64(1.0)))
        assert not jnp.isnan(g), "Gradient is NaN"


class TestGradientLogTeff_Alpha:
    """∂log_Teff/∂α: analytic vs finite-difference."""

    def test_fd_match_within_5_percent(self):
        """Core requirement: analytic matches FD to < 5%."""
        alpha = 1.9
        da = 5e-5  # Small enough to stay on same adaptive-step branch at 50 steps
        f = lambda a: stellar.evolve_star(1.0, Z=0.014, max_steps=MAX_STEPS, alpha_mlt=a)['log_Teff'][-1]
        ad = float(jax.grad(f)(jnp.float64(alpha)))
        fd = (float(f(alpha + da)) - float(f(alpha - da))) / (2 * da)
        rel_err = abs(ad - fd) / (abs(fd) + 1e-10)
        assert rel_err < 0.05, f"analytic={ad:.6f}, FD={fd:.6f}, err={rel_err:.3f}"

    def test_gradient_positive(self):
        """∂logTeff/∂α > 0: higher MLT α → more efficient convection → hotter."""
        g = float(jax.grad(lambda a: stellar.evolve_star(1.0, Z=0.014, max_steps=MAX_STEPS, alpha_mlt=a)['log_Teff'][-1])(jnp.float64(1.9)))
        assert g > 0, f"Gradient should be positive, got {g}"

    def test_not_nan(self):
        """Gradient must not be NaN."""
        g = float(jax.grad(lambda a: stellar.evolve_star(1.0, Z=0.014, max_steps=MAX_STEPS, alpha_mlt=a)['log_Teff'][-1])(jnp.float64(1.9)))
        assert not jnp.isnan(g), "Gradient is NaN"
