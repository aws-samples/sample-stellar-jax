"""AD-safe math primitives for microphysics.

Fractional powers x^p (0 < p < 1) have gradients p*x^(p-1) that diverge
as x → 0.  In stellar microphysics this arises in:
  - Nuclear: T6^(1/3)  →  gradient (1/3)*T6^(-2/3) diverges
  - Neutrino (plasma): a1^(2/3)  →  gradient (2/3)*a1^(-1/3) diverges
  - Neutrino (photo): a0^(1/3)  →  gradient (1/3)*a0^(-2/3) diverges

MESA handles these via compiler-level safe_exp/safe_log and by never
differentiating through them (no AD in Fortran).  For JAX AD we need
the gradient to be finite everywhere.

Solution: `safe_power(x, p, x_floor)` — a @custom_jvp that:
  - Forward: computes x^p exactly (same as jnp.power)
  - Backward: uses max(x, x_floor) in the gradient computation, i.e.
    d/dx[x^p] = p * max(x, x_floor)^(p-1)

This floors the gradient at p * x_floor^(p-1) instead of letting it
diverge to infinity.  The forward value is unaffected — the function
returns the exact x^p.  The gradient is exact whenever x > x_floor
and gracefully bounded otherwise.

References:
  - Itoh et al. (1996, ApJS 102, 411): neutrino small-argument limits
  - Caughlan & Fowler (1988, ADNDT 40, 283): nuclear rate low-T behaviour
  - MESA mod_neu.f90, net/private/net_approx21.f90: safe exp/log patterns
"""
import jax
import jax.numpy as jnp


@jax.custom_jvp
def safe_power(x, p, x_floor):
    """Compute x^p with AD-safe gradient (bounded near x=0).

    Parameters
    ----------
    x : array_like
        Base (must be >= 0 in the forward pass).
    p : float
        Exponent (typically a fraction like 1/3, 2/3).
    x_floor : float
        Minimum value of x used in the gradient computation.
        Should be set to a physically meaningful lower bound
        (e.g. 0.5 for T6, 1e-6 for density-derived quantities).

    Returns
    -------
    x^p : same shape as x
    """
    return jnp.power(x, p)


@safe_power.defjvp
def _safe_power_jvp(primals, tangents):
    """Custom JVP: gradient uses max(x, x_floor) to prevent divergence.

    Only the tangent w.r.t. x is computed.  p and x_floor are always static
    Python scalars (never differentiated), so their tangents are Python 0.0.
    We do NOT trace through dp because `y * log(x) * 0.0` can produce NaN
    via inf*0 when x is very small (log → -inf, y*log → -inf, -inf*0 = NaN
    under IEEE 754 in traced XLA graphs that don't short-circuit).
    """
    x, p, x_floor = primals
    dx, dp, dx_floor = tangents

    # Forward pass (exact)
    y = jnp.power(x, p)

    # Tangent: dy/dx = p * max(x, x_floor)^(p-1)
    x_safe = jnp.maximum(x, x_floor)
    dy = p * jnp.power(x_safe, p - 1.0) * dx

    return y, dy
