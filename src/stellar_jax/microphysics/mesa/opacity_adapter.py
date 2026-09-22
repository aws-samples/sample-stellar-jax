"""MESA opacity adapter: forward-only kappa() via jax.pure_callback.

Same signature as microphysics.opacity.kappa:
    kappa(logT, logRho, X, Z) -> log10(kappa)

This is a VALIDATION backend — forward-only, not differentiable.

No disagreement-fallback: the adapter returns exactly what MESA computes.
The only fallbacks are for non-finite inputs (guard) and Fortran shim
crashes (try/except). A validation backend must NEVER substitute JAX
values where MESA disagrees — that masks the very divergence it exists
to detect (#333).

References:
  - Iglesias & Rogers (1996), ApJ 464, 943 (OPAL opacity)
  - Ferguson et al. (2005), ApJ 623, 585 (low-T opacity)
  - Paxton et al. (2011), ApJS 192, 3, §4 (MESA kap module)
"""
import jax
import jax.numpy as jnp
import numpy as np

from stellar_jax.microphysics.mesa.bindings import call_mesa_kap


def _kap_callback(args):
    """NumPy callback for MESA opacity. Returns log10(kappa).

    Guards:
    - Non-finite inputs → return 0.0 (electron scattering floor, safe)
    - Fortran shim crash → return 0.0
    - Non-finite output → return 0.0
    """
    logT_f = float(args[0])
    logRho_f = float(args[1])
    X_f = float(args[2])
    Z_f = float(args[3])

    # Non-finite input guard
    if not (np.isfinite(logT_f) and np.isfinite(logRho_f)
            and np.isfinite(X_f) and np.isfinite(Z_f)):
        return np.array(0.0, dtype=np.float64)

    try:
        log_kappa, _, _ = call_mesa_kap(logT_f, logRho_f, X_f, Z_f)
    except (RuntimeError, ValueError, OSError):
        return np.array(0.0, dtype=np.float64)

    # Non-finite output guard
    if not np.isfinite(log_kappa):
        return np.array(0.0, dtype=np.float64)

    return np.array(log_kappa, dtype=np.float64)


# Relative step size for finite-difference JVP.
# (Retained for reference; no longer used — JAX-proxy JVP replaced FD.)


def _call_kap_forward(args):
    """Invoke the MESA opacity callback via jax.pure_callback."""
    return jax.pure_callback(
        _kap_callback,
        np.zeros((), dtype=np.float64),
        args,
        vmap_method='sequential',
    )


# Import JAX opacity for use as Jacobian proxy in custom_jvp.
from stellar_jax.microphysics.opacity import kappa as _jax_kappa


@jax.custom_jvp
def _kappa_mesa(logT, logRho, X, Z):
    """Internal MESA opacity lookup (4-arg, traced by custom_jvp).

    Equipped with a custom_jvp that uses JAX opacity for tangent computation.
    PRIMAL from MESA, TANGENT from JAX (fast AD, no callbacks).

    Returns: log10(kappa) [cm²/g]
    """
    args = jnp.stack([logT, logRho, X, Z])
    return _call_kap_forward(args)


def kappa(logT, logRho, X, Z, opacity_factor=None):
    """MESA opacity (same signature as JAX backend).

    opacity_factor is accepted for signature compatibility with the JAX backend
    (microphysics/contracts.py) but IGNORED — the MESA backend is forward-only
    validation and does not support physics-calibration knobs.

    Returns: log10(kappa) [cm²/g]
    """
    return _kappa_mesa(logT, logRho, X, Z)


@_kappa_mesa.defjvp
def _kappa_jvp(primals, tangents):
    """JVP using JAX opacity as a differentiable proxy.

    Primals: computed via MESA callback (exact MESA physics).
    Tangents: computed via JAX opacity AD (fast, no callbacks).
    """
    logT, logRho, X, Z = primals
    d_logT, d_logRho, d_X, d_Z = tangents

    # Primal: use MESA callback
    args = jnp.stack([logT, logRho, X, Z])
    f_x = _call_kap_forward(args)

    # Tangent: use JAX opacity (differentiable) as proxy
    _, tangent_out = jax.jvp(
        lambda lt, lr, x, z: _jax_kappa(lt, lr, x, z),
        (logT, logRho, X, Z),
        (d_logT, d_logRho, d_X, d_Z),
    )

    return f_x, tangent_out
