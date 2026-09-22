"""MESA neutrino adapter: forward-only epsilon_neutrino() via jax.pure_callback.

Same signature as microphysics.neutrino.epsilon_neutrino:
    epsilon_neutrino(rho, T, X, Z) -> eps_neu [erg/g/s]

This is a VALIDATION backend — forward-only, not differentiable.

References:
  - Itoh et al. (1996), ApJS 102, 411 (thermal neutrino losses)
  - Haft, Raffelt & Weiss (1994), ApJ 425, 222 (plasma neutrinos)
  - MESA neu/public/neu_lib.f90
"""
import jax
import jax.numpy as jnp
import numpy as np

from stellar_jax.microphysics.mesa.bindings import call_mesa_neu


def _neu_callback(args):
    """NumPy callback for MESA neutrino loss rate.

    NaN/Inf protection: if inputs are non-finite or MESA returns non-finite
    neutrino losses, returns 0.0 (no neutrino losses). This is safe because
    neutrino losses are a small correction to the energy budget — zero is
    better than NaN propagating through the timestep controller.
    """
    logT_f = float(args[0])
    logRho_f = float(args[1])
    X_f = float(args[2])
    Z_f = float(args[3])

    # Guard: non-finite inputs → return 0 (no neutrino losses)
    if not (np.isfinite(logT_f) and np.isfinite(logRho_f)):
        return np.array(0.0, dtype=np.float64)

    try:
        eps_neu = call_mesa_neu(logT_f, logRho_f, X_f, Z_f)
    except RuntimeError:
        eps_neu = 0.0

    # NaN/Inf guard: replace non-finite with zero (safe — neutrino losses
    # are a small correction; zero prevents NaN propagation)
    if not np.isfinite(eps_neu):
        eps_neu = 0.0

    return np.array(eps_neu, dtype=np.float64)


# Relative step size for finite-difference JVP.
# (Retained for reference; no longer used — JAX-proxy JVP replaced FD.)


def _call_neu_forward(args):
    """Invoke the MESA neutrino callback via jax.pure_callback."""
    return jax.pure_callback(
        _neu_callback,
        np.zeros((), dtype=np.float64),
        args,
        vmap_method='sequential',
    )


# Import JAX neutrino for use as Jacobian proxy in custom_jvp.
from stellar_jax.microphysics.neutrino import epsilon_neutrino as _jax_epsilon_neutrino


@jax.custom_jvp
def epsilon_neutrino(rho, T, X, Z):
    """MESA neutrino energy loss (same signature as JAX backend).

    Equipped with a custom_jvp that uses JAX neutrino for tangent computation.
    PRIMAL from MESA, TANGENT from JAX (fast AD, no callbacks).

    Returns: eps_neu [erg/g/s] (positive = energy lost from star)
    """
    # Sanitize: replace NaN/inf with safe defaults (prevents propagation from
    # RK4 state overflow — jnp.maximum(NaN, x) = NaN in IEEE 754)
    T_safe = jnp.where(jnp.isfinite(T), jnp.maximum(T, 1.0), jnp.float64(1e6))
    rho_safe = jnp.where(jnp.isfinite(rho), jnp.maximum(rho, 1e-30), jnp.float64(1.0))
    logT = jnp.log10(T_safe)
    logRho = jnp.log10(rho_safe)
    args = jnp.stack([logT, logRho, X, Z])
    result = _call_neu_forward(args)
    # Final NaN guard: neutrino losses must be finite and non-negative.
    # 0.0 is safe (neutrino losses are a small correction to the energy budget).
    return jnp.where(jnp.isfinite(result) & (result >= 0.0), result, jnp.float64(0.0))


@epsilon_neutrino.defjvp
def _neu_jvp(primals, tangents):
    """JVP using JAX neutrino as a differentiable proxy.

    Primals: computed via MESA callback with input sanitization (matches forward).
    Tangents: computed via JAX neutrino AD (fast, no callbacks).
    """
    rho, T, X, Z = primals
    d_rho, d_T, d_X, d_Z = tangents

    # Primal: replicate the forward function's sanitization logic.
    # Must match epsilon_neutrino() forward exactly (custom_jvp contract).
    T_safe = jnp.where(jnp.isfinite(T), jnp.maximum(T, 1.0), jnp.float64(1e6))
    rho_safe = jnp.where(jnp.isfinite(rho), jnp.maximum(rho, 1e-30), jnp.float64(1.0))
    logT = jnp.log10(T_safe)
    logRho = jnp.log10(rho_safe)
    args = jnp.stack([logT, logRho, X, Z])
    f_x = _call_neu_forward(args)
    # Final NaN guard (must match forward function)
    f_x = jnp.where(jnp.isfinite(f_x) & (f_x >= 0.0), f_x, jnp.float64(0.0))

    # Tangent: use JAX neutrino (differentiable) as proxy
    _, tangent_out = jax.jvp(
        lambda r, t, x, z: _jax_epsilon_neutrino(r, t, x, z),
        (rho, T, X, Z),
        (d_rho, d_T, d_X, d_Z),
    )

    return f_x, tangent_out
