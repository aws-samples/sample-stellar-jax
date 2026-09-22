"""MLT cubic solver: Cox & Giuli (1968) + HVB65 optically-thin correction.

Implements the 'Henyey' MLT option from MESA turb/private/mlt.f90 lines 62–168.
The cubic solve for the convective efficiency Gamma (C&G 14.82) is factored
into _mlt_cubic_solve — the single authoritative implementation used by both
mlt_nabla (custom_jvp switch) and mlt_nabla_raw (hard jnp.where switch).

MESA divergence (CONSTRAINT — JAX/differentiability):
  MESA uses exp(3*log(Gamma) - log(Bcubed)) to compute Zeta (avoids overflow).
  We use Gamma**3 / (Bcubed + 1e-300) with jnp.clip to [0,1].
  Reason: log(0) has infinite gradient in JAX; the +1e-300 floor keeps the
  backward pass finite. The forward value is identical within f64 precision
  for all physically-meaningful Gamma > 0.
"""

import jax.numpy as jnp

from stellar_jax.config.constants import k_B, m_H, a_rad, c_light
from stellar_jax.config.physics_floors import NAD_CP_FLOOR
from stellar_jax.transport._constants import FF1, FF2, FF3, FF4
from stellar_jax.transport.switch import _mlt_switch


def _mlt_cubic_solve(nabla_rad, nad, T, P, rho, kappa, g, mu, alpha_mlt, cp, Q,
                     gradL_composition_term=0.0):
    """Solve the MLT cubic for grad_conv (the convective temperature gradient).

    Implements C&G 14.79–14.86 with HVB65 optically-thin correction (ff4 term).
    Matches MESA turb/private/mlt.f90 lines 97–165 (Henyey option).

    Args:
        nabla_rad: Radiative temperature gradient (scalar per zone).
        nad: Adiabatic gradient from EOS (= gradL for Schwarzschild).
        T: Temperature [K].
        P: Total pressure [dyne/cm²].
        rho: Density [g/cm³].
        kappa: Opacity [cm²/g].
        g: Local gravity [cm/s²].
        mu: Mean molecular weight.
        alpha_mlt: MLT mixing-length parameter.
        cp: Specific heat at constant pressure [erg/g/K].
        Q: Thermal expansion coefficient chiT/chiRho (d ln rho / d ln T |_P).
        gradL_composition_term: Ledoux composition gradient term (≥0).
            gradL = nad + gradL_composition_term (MESA turb_support.f90:273).
            Default 0.0 = Schwarzschild (no composition gradient).

    Returns:
        grad_conv: Convective temperature gradient = (1 - Zeta)*nabla_rad + Zeta*gradL.
    """
    H_p = P / (rho * g + 1e-30)
    Lambda = alpha_mlt * H_p

    omega = jnp.maximum(Lambda * rho * kappa, 1e-30)
    ff4_omega2_plus_1 = FF4 / (omega * omega) + 1.0
    a0 = jnp.maximum((3.0 / 16.0) * FF2 * FF3 / ff4_omega2_plus_1, 1e-30)

    A = (4.0 * cp * jnp.sqrt(FF1 * P * Q * rho) * alpha_mlt * omega * ff4_omega2_plus_1) / \
        (FF3 * a_rad * c_light * T**3 + 1e-30)

    # Ledoux gradient: gradL = nad + composition term (MESA turb_support.f90:273).
    # When gradL_composition_term is a Python literal 0/0.0, avoid adding a JAX op
    # (keeps the XLA trace bit-identical to main for the differentiable path).
    if isinstance(gradL_composition_term, (int, float)) and gradL_composition_term == 0:
        gradL = nad
    else:
        gradL = nad + gradL_composition_term

    # MESA mlt.f90:122: Bcubed = (A²/a0)*(gradr - gradL)
    W = jnp.maximum(nabla_rad - gradL, 1e-30)
    Bcubed = (A * A / a0) * W

    # Cubic solver for Gamma (MESA turb/private/mlt.f90 lines 113-143).
    # Solves: a0*Gamma^3 + Gamma^2 + Gamma - a0*Bcubed == 0  (C&G 14.82)
    delta_c = a0 * Bcubed
    f = -2.0 + 9.0 * a0 + 27.0 * a0 * a0 * delta_c
    # disc = f² + 4*(3*a0 - 1)³ is the discriminant of the cubic.
    # MESA's branches: if f > 1e100 or disc <= 0, set f0 = f (instead of sqrt(disc)).
    # AD-safe: use a dummy positive disc in the sqrt branch when disc<=0 to avoid
    # sqrt(0) gradient (=inf) leaking through jnp.where (JAX evaluates both branches).
    disc = f * f + 4.0 * (-1.0 + 3.0 * a0) ** 3
    safe_disc = jnp.where(disc > 0.0, disc, 1.0)
    f0_sqrt = jnp.sqrt(safe_disc)
    f0 = jnp.where((disc > 0.0) & (f <= 1e100), f0_sqrt, f)
    f1 = jnp.maximum(f + f0, 1e-30) ** (1.0 / 3.0)
    f2 = 2.0 * (2.0 ** (1.0 / 3.0)) * (1.0 - 3.0 * a0) / f1
    Gamma = jnp.maximum(((4.0 ** (1.0 / 3.0)) * f1 + f2 - 2.0) / (6.0 * a0), 0.0)

    # Zeta: MESA uses exp(3*log(Gamma) - log(Bcubed)) to avoid overflow.
    # CONSTRAINT (JAX): we use Gamma**3 / Bcubed + clip — log(0) has inf gradient.
    Zeta = jnp.clip(Gamma ** 3 / (Bcubed + 1e-300), 0.0, 1.0)
    # MESA mlt.f90:174: gradT = (1 - Zeta)*gradr + Zeta*gradL
    grad_conv = (1.0 - Zeta) * nabla_rad + Zeta * gradL
    return grad_conv


def mlt_nabla(nabla_rad, nad, T, P, rho, kappa, g, mu, alpha_mlt, cp=None, Q=None,
             gradL_composition_term=0.0):
    """Actual temperature gradient via MLT with smooth alpha_mlt adjoint.

    Uses _mlt_cubic_solve for the physics + _mlt_switch (custom_jvp) for the
    Schwarzschild/Ledoux selection. The sigmoid value-blend JVP ensures nonzero
    alpha_mlt gradient across the convective/radiative boundary.

    Args:
        cp: Optional specific heat at constant pressure [erg/g/K]. If None,
            computed from ideal-gas approximation cp = k_B/(mu*m_H*nabla_ad).
        Q: Optional thermal expansion coefficient chiT/chiRho.
            Defaults to 1.0 (ideal gas).
        gradL_composition_term: Ledoux composition gradient (≥0, default 0.0).
            MESA turb_support.f90:273: gradL = grada + gradL_composition_term.
    """
    if Q is None:
        Q = 1.0
    if cp is None:
        cp = (k_B / (mu * m_H)) / jnp.maximum(nad, NAD_CP_FLOOR)
    grad_conv = _mlt_cubic_solve(nabla_rad, nad, T, P, rho, kappa, g, mu, alpha_mlt, cp, Q,
                                 gradL_composition_term)
    # Ledoux switch: convective when nabla_rad > gradL (= nad + comp_term).
    # MESA turb_support.f90:386: if (gradr > gradL) then ...
    if isinstance(gradL_composition_term, (int, float)) and gradL_composition_term == 0:
        gradL = nad
    else:
        gradL = nad + gradL_composition_term
    nabla = _mlt_switch(nabla_rad, gradL, grad_conv)
    return nabla


def mlt_nabla_raw(nabla_rad, nad, T, P, rho, kappa, g, mu, alpha_mlt, cp=None, Q=None,
                  gradL_composition_term=0.0):
    """Same MLT physics but with plain jnp.where (no custom_jvp).

    Used by the Henyey Newton Jacobian (via jacfwd) where the exact
    hard-switch derivative is needed for convergence. The custom_jvp
    sigmoid in _mlt_switch corrupts the Jacobian for radiative-envelope
    stars, slowing convergence from ~60 to ~200 iterations.

    Args:
        gradL_composition_term: Ledoux composition gradient (≥0, default 0.0).
            MESA turb_support.f90:273: gradL = grada + gradL_composition_term.
    """
    if Q is None:
        Q = 1.0
    if cp is None:
        cp = (k_B / (mu * m_H)) / jnp.maximum(nad, NAD_CP_FLOOR)
    grad_conv = _mlt_cubic_solve(nabla_rad, nad, T, P, rho, kappa, g, mu, alpha_mlt, cp, Q,
                                 gradL_composition_term)
    # Ledoux switch: convective when nabla_rad > gradL (= nad + comp_term).
    # MESA turb_support.f90:386: if (gradr > gradL) then ...
    if isinstance(gradL_composition_term, (int, float)) and gradL_composition_term == 0:
        gradL = nad
    else:
        gradL = nad + gradL_composition_term
    nabla = jnp.where(nabla_rad > gradL, grad_conv, nabla_rad)
    return nabla
