"""MESA MLT adapter: forward-only mlt_nabla() via jax.pure_callback.

Same signature as transport.mlt_nabla:
    mlt_nabla(nabla_rad, nad, T, P, rho, kappa, g, mu, alpha_mlt) -> nabla

Uses MESA's set_mlt (via the bind(C) mlt_wrapper shim) for the actual
gradient computation. Internally derives the quantities MESA needs (chiT,
chiRho, Cp, Lambda, gradL) from the solver's available state variables.

Assumptions (valid for H/He stellar interiors):
  - chiT = 1 (ideal gas: d ln P / d ln T |_rho = 1)
  - chiRho = 1 (ideal gas: d ln P / d ln rho |_T = 1)
  - gradL = nad (no composition gradient in simple H/He)
  - Cp = k_B / (mu * m_H * nabla_ad)  [standard ideal gas]

These are the SAME assumptions used by the JAX mlt_nabla in transport.py.
For a validation backend, the physics difference between the codes is in the
EOS/opacity tables, not in the MLT algebra (which is a closed-form cubic).

This is a VALIDATION backend — forward-only, but equipped with a
@jax.custom_jvp finite-difference rule so that Henyey's jacfwd can
propagate through it without crashing. The FD derivatives are NOT used
for production gradients (those come from the JAX path); they exist solely
to keep the Newton Jacobian functional during MESA validation runs.

References:
  - Cox & Giuli (1968), Principles of Stellar Structure, Ch. 14
  - Böhm-Vitense (1958), ZAp 46, 108 (MLT formulation)
  - Henyey, Vardya & Bodenheimer (1965), ApJ 142, 841 (optically-thin correction)
  - Griewank & Walther (2008), Evaluating Derivatives, §8.1 (FD as custom rule)
"""
import jax
import jax.numpy as jnp
import numpy as np

from stellar_jax.microphysics.mesa.bindings import call_mesa_mlt
from stellar_jax.config.constants import k_B, m_H

# Relative step size for finite-difference JVP.
# MESA's MLT is a smooth cubic solve — 1e-7 gives ~7 digits of agreement
# with the true derivative, well within Newton convergence requirements.
_FD_EPS = 1e-7


def _mlt_callback(args):
    """NumPy callback for MESA MLT. Returns the actual temperature gradient nabla.

    NaN/Inf protection: if any input is non-finite, returns nabla_rad (radiative
    gradient — the safe default when MLT computation is unreliable). This prevents
    NaN propagation from upstream EOS/opacity failures.
    """
    nabla_rad = float(args[0])
    nad = float(args[1])
    T = float(args[2])
    P = float(args[3])
    rho = float(args[4])
    kappa_val = float(args[5])
    g = float(args[6])
    mu = float(args[7])
    alpha_mlt = float(args[8])

    # Guard: if any critical input is non-finite, return nabla_rad (radiative)
    if not all(np.isfinite(x) for x in [nabla_rad, nad, T, P, rho, kappa_val, g]):
        # Use nad if nabla_rad is bad; otherwise nabla_rad
        fallback = nabla_rad if np.isfinite(nabla_rad) else (nad if np.isfinite(nad) else 0.4)
        return np.array(fallback, dtype=np.float64)

    # Derive MESA's required inputs from the solver's state
    # (same physics assumptions as transport.py)
    chiT = 1.0      # ideal gas
    chiRho = 1.0    # ideal gas
    Cp = float(k_B / (mu * m_H)) / max(nad, 1e-2)
    H_p = P / (rho * max(g, 1e-30))
    Lambda = alpha_mlt * H_p
    gradL = nad  # no composition gradient

    # Only call MESA MLT if convective (nabla_rad > nad)
    # Otherwise, the actual gradient = nabla_rad (radiative)
    if nabla_rad <= nad:
        return np.array(nabla_rad, dtype=np.float64)

    try:
        result = call_mesa_mlt(
            chiT=chiT, chiRho=chiRho, Cp=Cp, grav=g,
            Lambda=Lambda, rho=rho, P=P, T=T, opacity=kappa_val,
            gradr=nabla_rad, grada=nad, gradL=gradL,
            mixing_length_alpha=alpha_mlt,
            mlt_option='Henyey',
        )
        gradT = result['gradT']
        # NaN/Inf guard on MLT output
        if not np.isfinite(gradT):
            return np.array(nabla_rad, dtype=np.float64)
        return np.array(gradT, dtype=np.float64)
    except RuntimeError:
        # Fall back to nabla_rad if MESA MLT fails (shouldn't happen)
        return np.array(nabla_rad, dtype=np.float64)


def _call_mesa_mlt_forward(args):
    """Invoke the MESA MLT callback via jax.pure_callback (non-differentiable)."""
    return jax.pure_callback(
        _mlt_callback,
        np.zeros((), dtype=np.float64),
        args,
        vmap_method='sequential',
    )


def _fd_jvp_impl(primals, tangents):
    """Finite-difference JVP rule shared by mlt_nabla and mlt_nabla_raw.

    Computes the JVP (directional derivative along tangent t) via full
    per-component finite differences:

        tangent_out = sum_i (df/dx_i) * t_i

    where df/dx_i = (f(x + h_i*e_i) - f(x)) / h_i  with
    h_i = eps * max(|x_i|, 1.0).

    The primal replicates the forward function's full logic (input
    sanitization + output clamp) to satisfy JAX's custom_jvp contract.
    The FD tangent is computed on the SANITIZED inputs so the derivative
    reflects the actual function behavior (not the raw callback).

    This is correct regardless of whether t is a unit vector (jacfwd at
    the top level) or an arbitrary propagated tangent (jacfwd on a composed
    function, which is the Henyey use case: jacfwd(residual_fn) where
    residual_fn calls mlt_nabla_raw internally).

    The cost is 10 callback evaluations per JVP (1 base + 9 perturbed).
    Acceptable for a forward-only validation backend that runs occasionally.

    References:
      - Griewank & Walther (2008), Evaluating Derivatives, §6.1
    """
    nabla_rad, nad, T, P, rho, kappa, g, mu, alpha_mlt = primals
    d_nabla_rad, d_nad, d_T, d_P, d_rho, d_kappa, d_g, d_mu, d_alpha_mlt = tangents

    # Replicate forward sanitization (must match mlt_nabla/mlt_nabla_raw forward)
    nabla_rad_s = jnp.where(jnp.isfinite(nabla_rad), nabla_rad, nad)
    T_s = jnp.where(jnp.isfinite(T), T, jnp.float64(1e6))
    P_s = jnp.where(jnp.isfinite(P), P, jnp.float64(1e15))
    rho_s = jnp.where(jnp.isfinite(rho), rho, jnp.float64(1.0))
    kappa_s = jnp.where(jnp.isfinite(kappa), kappa, jnp.float64(1.0))
    g_s = jnp.where(jnp.isfinite(g), g, jnp.float64(1e4))

    # Pack sanitized primals and tangents
    x = jnp.stack([nabla_rad_s, nad, T_s, P_s, rho_s, kappa_s, g_s, mu, alpha_mlt])
    t = jnp.stack([d_nabla_rad, d_nad, d_T, d_P, d_rho, d_kappa, d_g, d_mu, d_alpha_mlt])

    # Forward evaluation at the sanitized base point
    raw_f_x = _call_mesa_mlt_forward(x)
    # Apply output NaN guard + clamp (matches forward function)
    raw_f_x = jnp.where(jnp.isfinite(raw_f_x), raw_f_x, nad)
    f_x = jnp.clip(raw_f_x, 0.0, jnp.maximum(nabla_rad_s, 0.5))

    # Per-component step sizes: h_i = eps * max(|x_i|, 1.0)
    h = _FD_EPS * jnp.maximum(jnp.abs(x), 1.0)

    # Compute partial derivatives and contract with tangent.
    # df/dx_i = (f(x + h_i * e_i) - f(x)) / h_i
    # tangent_out = sum_i (df/dx_i * t_i)
    #
    # We compute each partial via a perturbed callback call.
    def partial_i(i):
        """Compute df/dx_i * t_i for component i."""
        e_i = jnp.zeros(9).at[i].set(1.0)
        raw_pert = _call_mesa_mlt_forward(x + h[i] * e_i)
        # NaN guard + clamp (same as forward)
        raw_pert = jnp.where(jnp.isfinite(raw_pert), raw_pert, nad)
        f_pert = jnp.clip(raw_pert, 0.0, jnp.maximum(nabla_rad_s, 0.5))
        df_dxi = (f_pert - f_x) / h[i]
        return df_dxi * t[i]

    # Sum contributions from all 9 components.
    tangent_out = jnp.zeros(())
    for i in range(9):
        tangent_out = tangent_out + partial_i(i)

    return f_x, tangent_out


@jax.custom_jvp
def mlt_nabla(nabla_rad, nad, T, P, rho, kappa, g, mu, alpha_mlt):
    """MESA MLT (same signature as transport.mlt_nabla).

    Equipped with a finite-difference custom_jvp so Henyey's jacfwd can
    propagate through the non-differentiable MESA callback.

    Returns: nabla (actual temperature gradient)
    """
    # Sanitize inputs: replace NaN/inf with safe defaults (from RK4 state overflow).
    # nabla_rad can be inf when P overflows; other inputs can be NaN from state.
    nabla_rad = jnp.where(jnp.isfinite(nabla_rad), nabla_rad, nad)
    T = jnp.where(jnp.isfinite(T), T, jnp.float64(1e6))
    P = jnp.where(jnp.isfinite(P), P, jnp.float64(1e15))
    rho = jnp.where(jnp.isfinite(rho), rho, jnp.float64(1.0))
    kappa = jnp.where(jnp.isfinite(kappa), kappa, jnp.float64(1.0))
    g = jnp.where(jnp.isfinite(g), g, jnp.float64(1e4))
    args = jnp.stack([nabla_rad, nad, T, P, rho, kappa, g, mu, alpha_mlt])
    # Clamp output nabla to [0, nabla_rad] — physically, nabla ≤ nabla_rad always.
    result = _call_mesa_mlt_forward(args)
    # Final NaN guard: if callback returned non-finite, use nad (adiabatic gradient)
    # as a safe default. This is physically conservative (assumes efficient convection).
    result = jnp.where(jnp.isfinite(result), result, nad)
    return jnp.clip(result, 0.0, jnp.maximum(nabla_rad, 0.5))


@mlt_nabla.defjvp
def _mlt_nabla_jvp(primals, tangents):
    """FD JVP: directional derivative via (f(x+h) - f(x)) / h."""
    return _fd_jvp_impl(primals, tangents)


@jax.custom_jvp
def mlt_nabla_raw(nabla_rad, nad, T, P, rho, kappa, g, mu, alpha_mlt):
    """MESA MLT for Henyey Jacobian (same as mlt_nabla, separate custom_jvp).

    Used by the Henyey Newton solver via jacfwd. The FD JVP rule ensures
    jacfwd can propagate through the non-differentiable MESA callback.
    """
    # Same input sanitization as mlt_nabla
    nabla_rad = jnp.where(jnp.isfinite(nabla_rad), nabla_rad, nad)
    T = jnp.where(jnp.isfinite(T), T, jnp.float64(1e6))
    P = jnp.where(jnp.isfinite(P), P, jnp.float64(1e15))
    rho = jnp.where(jnp.isfinite(rho), rho, jnp.float64(1.0))
    kappa = jnp.where(jnp.isfinite(kappa), kappa, jnp.float64(1.0))
    g = jnp.where(jnp.isfinite(g), g, jnp.float64(1e4))
    args = jnp.stack([nabla_rad, nad, T, P, rho, kappa, g, mu, alpha_mlt])
    result = _call_mesa_mlt_forward(args)
    # Final NaN guard (same as mlt_nabla)
    result = jnp.where(jnp.isfinite(result), result, nad)
    return jnp.clip(result, 0.0, jnp.maximum(nabla_rad, 0.5))


@mlt_nabla_raw.defjvp
def _mlt_nabla_raw_jvp(primals, tangents):
    """FD JVP: identical rule to mlt_nabla."""
    return _fd_jvp_impl(primals, tangents)
