"""Fast unit tests for the transport/ package (issue #497).

Tests the DRYed cubic solver, the @custom_jvp switch, and the dual-variant
contract. All tests are <5s, isolated (no evolve_star, no heavy JIT).

Marks: @pytest.mark.fast (component-level, no lax.scan/evolve_star).
"""

import os
import sys

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from stellar_jax.transport import mlt_nabla, mlt_nabla_raw, _mlt_switch, _mlt_cubic_solve
from stellar_jax.transport._constants import FF1, FF2, FF3, FF4, NU_PARAM, Y_PARAM


# ===========================================================================
# Helpers: physically-meaningful MLT inputs
# ===========================================================================

def _solar_convective_inputs():
    """Solar-like CZ conditions: nabla_rad >> nad → convective."""
    return dict(
        nabla_rad=jnp.float64(3.5),
        nad=jnp.float64(0.4),
        T=jnp.float64(1e6),
        P=jnp.float64(1e17),
        rho=jnp.float64(0.1),
        kappa=jnp.float64(1.0),
        g=jnp.float64(2.7e4),
        mu=jnp.float64(0.6),
        alpha_mlt=jnp.float64(2.0),
        cp=None,
        Q=None,
    )


def _radiative_inputs():
    """Radiative zone: nabla_rad < nad."""
    d = _solar_convective_inputs()
    d['nabla_rad'] = jnp.float64(0.2)  # well below nad=0.4
    return d


# ===========================================================================
# §1. Constants validation
# ===========================================================================

@pytest.mark.fast
def test_mlt_constants_match_mesa_henyey_option():
    """MLT geometry constants match MESA turb/private/mlt.f90 lines 90-95 (Henyey)."""
    # MESA: ff1=1/nu, ff2=0.5, ff3=8/y, ff4=1/y with nu=8, y=1/3
    assert NU_PARAM == 8.0
    assert Y_PARAM == pytest.approx(1.0 / 3.0)
    assert FF1 == pytest.approx(1.0 / 8.0)
    assert FF2 == 0.5
    assert FF3 == pytest.approx(24.0)
    assert FF4 == pytest.approx(3.0)


# ===========================================================================
# §2. _mlt_switch forward + JVP tests
# ===========================================================================

@pytest.mark.fast
def test_mlt_switch_forward_exact():
    """_mlt_switch returns exact jnp.where result (no smoothing in forward)."""
    nad = jnp.float64(0.4)
    grad_conv = jnp.float64(0.42)

    # Convective: nabla_rad > nad → returns grad_conv
    nrad_conv = jnp.float64(0.5)
    result = _mlt_switch(nrad_conv, nad, grad_conv)
    assert float(result) == float(grad_conv)

    # Radiative: nabla_rad < nad → returns nabla_rad
    nrad_rad = jnp.float64(0.35)
    result_r = _mlt_switch(nrad_rad, nad, grad_conv)
    assert float(result_r) == float(nrad_rad)

    # Boundary: nabla_rad == nad → returns nabla_rad (jnp.where uses >)
    result_b = _mlt_switch(nad, nad, grad_conv)
    assert float(result_b) == float(nad)


@pytest.mark.fast
def test_mlt_switch_jvp_sigmoid_formula():
    """JVP tangent matches w*d_conv + (1-w)*d_rad with eps=0.005 at 3 test points."""
    _eps = 0.005
    nad = jnp.float64(0.4)
    grad_conv = jnp.float64(0.39)

    test_cases = [
        # (nabla_rad, description, expected_w_approx)
        (0.5, "deep convective", 1.0),    # (0.5-0.4)/0.005 = 20 → w≈1
        (0.2, "deep radiative", 0.0),      # (0.2-0.4)/0.005 = -40 → w≈0
        (0.4, "boundary", 0.5),            # (0.4-0.4)/0.005 = 0 → w=0.5
    ]

    for nrad_val, desc, w_approx in test_cases:
        nrad = jnp.float64(nrad_val)
        d_rad, d_conv = jnp.float64(1.0), jnp.float64(2.0)
        primals = (nrad, nad, grad_conv)
        tangents = (d_rad, jnp.float64(0.0), d_conv)

        _, t_out = jax.jvp(_mlt_switch, primals, tangents)
        t_out = float(t_out)

        w = float(jax.nn.sigmoid((nrad - nad) / _eps))
        expected = w * float(d_conv) + (1.0 - w) * float(d_rad)

        assert abs(t_out - expected) < 1e-12, (
            f"{desc}: JVP {t_out:.10e} != formula {expected:.10e}")
        # Also check w is in expected range
        assert abs(w - w_approx) < 0.01, (
            f"{desc}: w={w:.4f}, expected ~{w_approx}")


# ===========================================================================
# §3. _mlt_cubic_solve tests
# ===========================================================================

@pytest.mark.fast
def test_mlt_cubic_solve_known_values():
    """_mlt_cubic_solve produces physically valid output in a convective zone."""
    d = _solar_convective_inputs()
    # Compute cp/Q defaults matching what mlt_nabla would use
    from stellar_jax.config.constants import k_B, m_H
    cp = (k_B / (d['mu'] * m_H)) / jnp.maximum(d['nad'], 1e-2)
    Q = jnp.float64(1.0)

    grad_conv = _mlt_cubic_solve(
        d['nabla_rad'], d['nad'], d['T'], d['P'], d['rho'],
        d['kappa'], d['g'], d['mu'], d['alpha_mlt'], cp, Q
    )
    gc = float(grad_conv)

    # Physical constraints: grad_conv must be between nad and nabla_rad
    # (efficient convection brings grad toward nad; inefficient → toward nabla_rad)
    assert gc >= float(d['nad']), f"grad_conv={gc} < nad={float(d['nad'])}"
    assert gc <= float(d['nabla_rad']), f"grad_conv={gc} > nabla_rad={float(d['nabla_rad'])}"


@pytest.mark.fast
def test_mlt_cubic_solve_optically_thick_limit():
    """At large optical depth (omega → ∞), efficient convection → grad_conv → nad.

    When the mixing length is much larger than the photon mean free path,
    convection is maximally efficient and the gradient approaches adiabatic.
    This is the A → ∞ limit: Bcubed → ∞, Gamma → ∞, Zeta → 1 → gradT = nad.
    """
    from stellar_jax.config.constants import k_B, m_H

    nad = jnp.float64(0.4)
    nabla_rad = jnp.float64(2.0)
    T = jnp.float64(1e7)       # hot → large omega
    P = jnp.float64(1e18)      # high P
    rho = jnp.float64(10.0)    # dense → large omega = Lambda*rho*kappa
    kappa = jnp.float64(10.0)  # high opacity → large omega
    g = jnp.float64(2.7e4)
    mu = jnp.float64(0.6)
    alpha_mlt = jnp.float64(2.0)
    cp = (k_B / (mu * m_H)) / jnp.maximum(nad, 1e-2)
    Q = jnp.float64(1.0)

    grad_conv = _mlt_cubic_solve(nabla_rad, nad, T, P, rho, kappa, g, mu, alpha_mlt, cp, Q)
    gc = float(grad_conv)

    # In the highly efficient limit, grad_conv should be very close to nad
    assert abs(gc - float(nad)) < 0.01, (
        f"Optically thick: grad_conv={gc:.6f} not close to nad={float(nad):.6f}")


# ===========================================================================
# §4. Dual-variant contract tests
# ===========================================================================

@pytest.mark.fast
def test_mlt_nabla_vs_raw_forward_identical():
    """mlt_nabla and mlt_nabla_raw produce identical forward outputs.

    They differ ONLY in JVP behavior (sigmoid vs hard-where tangent).
    """
    for inputs in [_solar_convective_inputs(), _radiative_inputs()]:
        result = mlt_nabla(**inputs)
        result_raw = mlt_nabla_raw(**inputs)
        assert float(result) == float(result_raw), (
            f"Forward mismatch: mlt_nabla={float(result)}, raw={float(result_raw)}")


@pytest.mark.fast
def test_mlt_nabla_alpha_gradient_nonzero_convective():
    """In a convective zone, d(mlt_nabla)/d(alpha_mlt) is nonzero.

    This validates the custom_jvp path provides alpha sensitivity through
    the cubic solver. Without the _mlt_switch custom_jvp, the gradient would
    still be nonzero here (since nabla_rad > nad selects the convective branch
    in both hard-where and sigmoid-blend), but the VALUE would differ at
    boundary zones.
    """
    d = _solar_convective_inputs()
    grad_fn = jax.grad(lambda a: mlt_nabla(
        d['nabla_rad'], d['nad'], d['T'], d['P'], d['rho'],
        d['kappa'], d['g'], d['mu'], a, d['cp'], d['Q']))
    grad_alpha = grad_fn(d['alpha_mlt'])
    assert abs(float(grad_alpha)) > 1e-15, (
        f"Alpha gradient must be nonzero in CZ, got {float(grad_alpha)}")


@pytest.mark.fast
def test_mlt_nabla_raw_alpha_grad_zero_radiative():
    """In a radiative zone, mlt_nabla_raw gives ZERO alpha gradient (hard where).

    The hard jnp.where selects nabla_rad when nabla_rad < nad, so the tangent
    w.r.t. alpha_mlt is exactly zero (alpha only flows through grad_conv which
    is not selected). Meanwhile mlt_nabla gives nonzero due to sigmoid bleed
    (at a MARGINAL zone near the Schwarzschild boundary, where w is appreciable).
    """
    d = _solar_convective_inputs()
    # Use a MARGINAL radiative zone: nabla_rad just below nad.
    # At (nabla_rad - nad)/eps = -1 → w = sigmoid(-1) ≈ 0.27 (appreciable bleed).
    d['nabla_rad'] = jnp.float64(0.395)  # 0.395 < 0.4 → radiative, but marginal

    # Raw: zero gradient (hard where kills alpha path)
    grad_raw = jax.grad(lambda a: mlt_nabla_raw(
        d['nabla_rad'], d['nad'], d['T'], d['P'], d['rho'],
        d['kappa'], d['g'], d['mu'], a, d['cp'], d['Q']))
    g_raw = grad_raw(d['alpha_mlt'])
    assert abs(float(g_raw)) < 1e-15, (
        f"mlt_nabla_raw must have zero alpha grad in radiative zone, got {float(g_raw)}")

    # Custom JVP: nonzero gradient (sigmoid bleed at w≈0.27)
    grad_custom = jax.grad(lambda a: mlt_nabla(
        d['nabla_rad'], d['nad'], d['T'], d['P'], d['rho'],
        d['kappa'], d['g'], d['mu'], a, d['cp'], d['Q']))
    g_custom = grad_custom(d['alpha_mlt'])
    assert abs(float(g_custom)) > 1e-15, (
        f"mlt_nabla must have nonzero alpha grad (sigmoid bleed), got {float(g_custom)}")


@pytest.mark.fast
def test_mlt_cubic_solve_gradient_vs_fd():
    """AD gradient of _mlt_cubic_solve w.r.t. alpha_mlt matches independent FD.

    This is the cheapest gradient correctness check — a single function call,
    no evolve_star. Confirms the cubic solver algebra is correctly differentiated.
    """
    from stellar_jax.config.constants import k_B, m_H

    # Use conditions with INEFFICIENT convection (low omega) so gradient is
    # appreciable (~1e-8) rather than near numerical floor. Inefficient convection
    # is the regime where alpha_mlt sensitivity matters physically.
    nabla_rad = jnp.float64(0.6)    # moderately superadiabatic
    nad = jnp.float64(0.4)
    T = jnp.float64(1e4)            # cooler
    P = jnp.float64(1e12)           # lower pressure
    rho = jnp.float64(1e-5)         # low density → small omega
    kappa = jnp.float64(0.1)        # low opacity
    g = jnp.float64(2.7e4)
    mu = jnp.float64(0.6)
    alpha_mlt = jnp.float64(2.0)
    cp = (k_B / (mu * m_H)) / jnp.maximum(nad, 1e-2)
    Q = jnp.float64(1.0)

    def f(alpha):
        return _mlt_cubic_solve(nabla_rad, nad, T, P, rho, kappa, g, mu, alpha, cp, Q)

    ad_grad = float(jax.grad(f)(alpha_mlt))

    # Independent FD (central difference)
    # h=1e-3 gives good FD convergence for this gradient magnitude (~5e-8)
    h = 1e-3
    f_plus = float(f(alpha_mlt + h))
    f_minus = float(f(alpha_mlt - h))
    fd_grad = (f_plus - f_minus) / (2 * h)

    # Both should be nonzero and agree to ~1e-5 relative
    assert abs(ad_grad) > 1e-15, f"AD gradient is unexpectedly zero: {ad_grad}"
    assert abs(fd_grad) > 1e-15, f"FD gradient is unexpectedly zero: {fd_grad}"

    rel_err = abs(ad_grad - fd_grad) / (abs(fd_grad) + 1e-30)
    assert rel_err < 1e-4, (
        f"AD vs FD: AD={ad_grad:.8e}, FD={fd_grad:.8e}, rel_err={rel_err:.2e}")
