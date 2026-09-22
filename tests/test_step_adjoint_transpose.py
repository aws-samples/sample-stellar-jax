"""Tier-0 per-step adjoint-transpose test (#1161).

Fast local gate (seconds-to-minutes) that validates the per-step adjoint of the
differentiable Henyey structure solve on ONE committed subgiant step, instead of
the 95-min end-to-end evolution.

Bug it catches (#1161): a through-evolution `d sigma^2/dM` came out wrong-sign
because the replay custom_vjp BACKWARD rebuilt its Jacobian/VJP on the ideal-gas
branch while the FORWARD solved on the adaptive branch (frozen cp/Q + non-zero
Ledoux gradL). The A+C fix threads the frozen gradL into the backward so it is the
EXACT transpose of the forward. Here we differentiate the SAME custom_vjp solve
w.r.t. M_star at step 118 (deep subgiant, max|gradL|~86) and require AD to match a
central finite difference in sign AND magnitude.

Signature matched at src/stellar_jax/solver/continuation.py:352
(henyey_solve_from_state_atm(y_prev, q_mesh, M_star, X_profile, Z, alpha_mlt,
 R_phot, n_iter=30, tol=1e-4, ln_T_prev=..., ln_P_prev=..., inv_dt=...,
 atm_ratio=..., N14_profile=..., comp_mfracs=..., bypass_conv_gate=...,
 gradL_composition_term=..., helm_eos=..., bicubic_opacity=...)).
Production solve mirrored from evolution/step.py (see /tmp/tier0_brief.md).
"""
import os
import time

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stellar_jax.solver.continuation import henyey_solve_from_state_atm

_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "subgiant_step_fixture.npz")


def _load_step(prefix="step_118_"):
    d = np.load(_FIXTURE, allow_pickle=True)
    g = {k[len(prefix):]: d[k] for k in d.files if k.startswith(prefix)}
    return g


def _solve(step, M_star, X_profile, alpha_mlt):
    """Mirror the exact production replay solve call (evolution/step.py)."""
    dt = float(step["dt_sec"])
    inv_dt = jax.lax.stop_gradient(
        jnp.where(dt > 0.0, 1.0 / jnp.maximum(dt, 1.0), 0.0))
    res = henyey_solve_from_state_atm(
        jnp.asarray(step["y_prev"]),
        jnp.asarray(step["q_mesh"]),
        M_star,
        X_profile,
        float(step["Z"]),
        alpha_mlt,
        None,                      # R_phot (7th positional) -> None, atm_ratio kw below
        n_iter=100, tol=1e-4,
        ln_T_prev=jnp.asarray(step["ln_T_prev"]),
        ln_P_prev=jnp.asarray(step["ln_P_prev"]),
        inv_dt=inv_dt,
        atm_ratio=float(step["atm_ratio"]),
        N14_profile=jnp.asarray(step["N14_profile"]),
        comp_mfracs=jnp.asarray(step["comp_mfracs"]),
        bypass_conv_gate=True,
        gradL_composition_term=jnp.asarray(step["gradL_composition_term"]),
        helm_eos=False, bicubic_opacity=True)
    return res


@pytest.mark.smoke
def test_step118_adjoint_transpose_vs_fd():
    step = _load_step("step_118_")

    M0 = float(step["M_star_cgs"])
    X_profile = jnp.asarray(step["X_profile"])
    alpha0 = float(step["alpha_mlt"])
    y_conv = np.asarray(step["y_converged"])
    gradL_max = float(np.max(np.abs(step["gradL_composition_term"])))

    # --- AC1: fixture converged state is reproduced, and bug-A witness present ---
    t0 = time.time()
    res0 = _solve(step, M0, X_profile, alpha0)
    y_solved = np.asarray(res0["y"])
    ac1 = float(np.max(np.abs(y_solved - y_conv)))
    print(f"\n[AC1] max|y_solved - y_converged| = {ac1:.3e} "
          f"(converged={bool(res0['converged'])}, res_norm={float(res0['residual_norm']):.3e})")
    print(f"[AC1] max|gradL| = {gradL_max:.3e} (bug-A witness; adaptive branch active)")
    assert gradL_max > 1e-3, "gradL is ~0 -> vacuous (wrong branch)"
    assert ac1 < 1e-3, f"AC1 reconstruction failed: max|y_solved-y_converged|={ac1:.3e}"

    # --- Core: per-step adjoint AD vs central FD w.r.t. M_star ---
    rng = np.random.default_rng(1161)
    cot_np = rng.standard_normal(y_conv.shape)
    cot_np /= np.linalg.norm(cot_np)          # unit cotangent (600,4), seeded
    cot = jnp.asarray(cot_np)

    def f(M):
        return jnp.vdot(cot, _solve(step, M, X_profile, alpha0)["y"])

    AD = float(jax.grad(f)(M0))

    h = 1e-6 * M0
    fp = float(f(M0 + h))
    fm = float(f(M0 - h))
    FD = (fp - fm) / (2.0 * h)

    rel_err = abs(AD - FD) / max(abs(FD), abs(AD), 1e-30)
    wall = time.time() - t0
    print(f"[M-channel] AD = {AD:.8e}")
    print(f"[M-channel] FD = {FD:.8e}")
    print(f"[M-channel] rel_err = {rel_err:.3e}   AD*FD = {AD*FD:.3e}   (h={h:.3e})")
    print(f"[wall] {wall:.1f}s")

    assert np.isfinite(AD), f"AD not finite: {AD}"
    assert AD * FD > 0, f"SIGN MISMATCH: AD={AD:.6e} FD={FD:.6e}"
    assert rel_err < 0.02, f"rel_err too large: {rel_err:.3e} (AD={AD:.6e} FD={FD:.6e})"


@pytest.mark.smoke
@pytest.mark.xfail(
    reason="alpha_mlt channel: AD and FD agree in SIGN but AD magnitude is ~15.6% "
           "below FD (AD=-1.36e-3 vs FD=-1.61e-3), STABLE across FD h in 1e-4..1e-7 "
           "(so NOT FD truncation). alpha flows through the frozen cp/Q EOS branch the "
           "backward treats as constant (documented constraint, forced by 4-D-EOS jacfwd "
           "OOM; tracked in #1232). The DECISIVE #1161 wrong-sign witness is the M_star "
           "channel (test_step118_adjoint_transpose_vs_fd), which is exact (rel_err 5e-6). "
           "Leaving this as a real, non-weakened signal.",
    strict=False)
def test_step118_adjoint_transpose_vs_fd_alpha_channel():
    """Second input channel: differentiate w.r.t. a scalar scaling of alpha_mlt.

    NOTE: the 2% tolerance is intentionally NOT relaxed; this xfails on magnitude
    because alpha routes through the frozen-cp/Q Jacobian approximation (#1232)."""
    step = _load_step("step_118_")
    M0 = float(step["M_star_cgs"])
    X_profile = jnp.asarray(step["X_profile"])
    alpha0 = float(step["alpha_mlt"])
    y_conv = np.asarray(step["y_converged"])

    rng = np.random.default_rng(11611)
    cot_np = rng.standard_normal(y_conv.shape)
    cot_np /= np.linalg.norm(cot_np)
    cot = jnp.asarray(cot_np)

    def f(a):
        return jnp.vdot(cot, _solve(step, M0, X_profile, a)["y"])

    AD = float(jax.grad(f)(alpha0))
    h = 1e-6 * alpha0
    FD = (float(f(alpha0 + h)) - float(f(alpha0 - h))) / (2.0 * h)
    rel_err = abs(AD - FD) / max(abs(FD), abs(AD), 1e-30)
    print(f"\n[alpha-channel] AD = {AD:.8e}  FD = {FD:.8e}  rel_err = {rel_err:.3e}  AD*FD = {AD*FD:.3e}")

    assert np.isfinite(AD)
    assert AD * FD > 0, f"alpha SIGN MISMATCH: AD={AD:.6e} FD={FD:.6e}"
    assert rel_err < 0.02, f"alpha rel_err too large: {rel_err:.3e}"
