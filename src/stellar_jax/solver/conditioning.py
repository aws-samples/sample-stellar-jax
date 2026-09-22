"""Jacobian conditioning: row equilibration + Levenberg + Armijo line-search.

This module provides robust conditioning for the Newton solver:
  1. Row equilibration (Duff, Erisman & Reid 1986, §5.3)
  2. Levenberg regularization (Nocedal & Wright 2006, §10.2)
  3. Armijo backtracking line-search (Nocedal & Wright 2006, §3.1;
     MESA star_solver.f90:739-953)
  4. Dual convergence (MESA star_solver.f90:53-64, 496-509)

References:
  - Duff, Erisman & Reid (1986), "Direct Methods for Sparse Matrices", §5.3
  - Nocedal & Wright (2006), "Numerical Optimization", §3.1, §10.2
  - Paxton et al. (2011), ApJS 192, 3, §6.3 (MESA: backtracking without Levenberg)
  - MESA star_solver.f90:739-953 (adjust_correction)
  - MESA solver_support.f90:42-86 (set_xscale_info), 316-617 (sizeB)
  - MESA star_solver.f90:53-64, 496-509 (dual convergence criterion)
  - MESA controls.defaults:8676-8892 (tolerance defaults)
"""

import jax.numpy as jnp
import jax.lax as lax


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Levenberg regularization base coefficient.
_LEVENBERG_BASE = 1e-4

# Dual convergence tolerances (MESA-grounded):
_TOL_RESIDUAL = 1e-8
_TOL_MAX_CORRECTION = 3e-3

# Armijo line-search parameters (MESA star_solver.f90:739-953):
_ARMIJO_ALF = 1e-2
_ARMIJO_FACTOR = 0.2
_ARMIJO_MAX_ITER = 8
_ARMIJO_MIN_ALPHA = 1.0 / 64.0


# ─────────────────────────────────────────────────────────────────────────────
# Row equilibration + Levenberg regularization
# ─────────────────────────────────────────────────────────────────────────────

def _condition_system(A, B, C, rhs, R_norm, levenberg=True):
    """Per-equation row scaling + Levenberg regularization of the block system.

    Transforms:
        A_k dy_{k-1} + B_k dy_k + C_k dy_{k+1} = rhs_k
    into:
        S_k A_k dy_{k-1} + (S_k B_k + λI) dy_k + S_k C_k dy_{k+1} = S_k rhs_k

    where S_k = diag(s_i) with s_i = 1/max_j|B_k[i,j]| (row equilibration).
    """
    # Row equilibration: s_i = 1 / max_j |B_k[i,j]|
    row_max = jnp.max(jnp.abs(B), axis=-1)  # (N, 4)
    row_scale = 1.0 / jnp.maximum(row_max, 1e-30)  # (N, 4)
    D = row_scale[:, :, None]  # (N, 4, 1) for broadcasting

    A_s = A * D
    B_s = B * D
    C_s = C * D
    rhs_s = rhs * row_scale  # (N, 4)

    if levenberg:
        lam = _LEVENBERG_BASE * jnp.minimum(jnp.maximum(R_norm, 0.01), 1.0)
        eye4 = jnp.eye(4)
        B_reg = B_s + lam * eye4[None, :, :]
    else:
        B_reg = B_s

    return A_s, B_reg, C_s, rhs_s


# ─────────────────────────────────────────────────────────────────────────────
# Block-Thomas solve for pre-conditioned system
# ─────────────────────────────────────────────────────────────────────────────

def _block_thomas_solve_conditioned(A, B, C, rhs):
    """Block-Thomas solve for an ALREADY-conditioned system.

    Same algorithm as block_thomas_solve but skips internal row equilibration
    (the caller has already applied _condition_system).

    NO per-block NaN guards — the Levenberg floor ensures all diagonal pivots
    are well above subnormal thresholds on AVX-512 FTZ/DAZ hardware.
    """
    N = B.shape[0]

    def forward_step(carry, k):
        B_prev, F_prev = carry
        factor = jnp.linalg.solve(B_prev.T, A[k].T).T
        B_new = B[k] - factor @ C[k - 1]
        F_new = rhs[k] - factor @ F_prev
        return (B_new, F_new), (B_new, F_new)

    init_carry = (B[0], rhs[0])
    _, (B_rest, F_rest) = lax.scan(forward_step, init_carry, jnp.arange(1, N))

    B_all = jnp.concatenate([B[0:1], B_rest], axis=0)
    F_all = jnp.concatenate([rhs[0:1], F_rest], axis=0)

    x_last = jnp.linalg.solve(B_all[N - 1], F_all[N - 1])

    def back_step(x_next, k):
        x_k = jnp.linalg.solve(B_all[k], F_all[k] - C[k] @ x_next)
        return x_k, x_k

    _, x_rev = lax.scan(back_step, x_last, jnp.arange(N - 2, -1, -1))
    x = jnp.concatenate([x_rev[::-1], x_last[None]], axis=0)
    return x


def conditioned_solve(A, B, C, rhs, R_norm, levenberg=True):
    """Condition the block system then solve — convenience wrapper."""
    return _block_thomas_solve_conditioned(
        *_condition_system(A, B, C, rhs, R_norm, levenberg=levenberg)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Armijo backtracking line-search (KEEP WHOLE)
# ─────────────────────────────────────────────────────────────────────────────

def armijo_line_search(y, dy, R, residual_fn, R_norm):
    """Armijo backtracking line-search on the merit function f = ½‖F‖².

    Finds α ∈ [_ARMIJO_MIN_ALPHA, 1] such that:
        f(y + α dy) ≤ f(y) + alf * α * slope

    Uses quadratic interpolation on first backtrack, then cubic using two
    prior function values (same as MESA star_solver.f90:739-953).

    The step-size α is wrapped in stop_gradient: it controls the convergence
    PATH but does not carry gradient information. The IFT adjoint operates at
    the converged fixed point where F(y*)=0 (α is irrelevant there).

    Implementation: fixed-length scan (8 iterations) with freeze-on-done
    semantics. The full step (α=1) is typically accepted on the first
    iteration, so the remaining 7 iterations are effectively no-ops (state
    frozen). Kept as scan rather than while_loop to avoid nested while_loops
    inside the Newton while_loop — nested while_loops inside lax.scan produce
    an XLA graph that crashes the compiler on the heaviest gradient tests
    (SIGSEGV during compilation at ~64 GB; issue #805).
    MESA star_solver.f90:739-953 uses early return; our scan-freeze is the
    equivalent within JAX's fixed-shape constraint.

    Parameters
    ----------
    y : (N, 4) — current state.
    dy : (N, 4) — Newton direction.
    R : (N, 4) — current residual F(y).
    residual_fn : callable (N,4) -> (N,4).
    R_norm : scalar — max|R|.

    Returns
    -------
    alpha : scalar (stop_gradient wrapped).
    """
    fold = 0.5 * jnp.sum(R ** 2)
    slope = -2.0 * fold

    def backtrack_step(carry, _):
        alam, alam2, f2, first_time, done = carry

        y_trial = y + alam * dy
        R_trial = residual_fn(y_trial)
        f_trial = 0.5 * jnp.sum(R_trial ** 2)

        # Sufficient decrease (MESA-style: also accept any f ≤ fold/2)
        f_target = jnp.maximum(fold * 0.5, fold + _ARMIJO_ALF * alam * slope)
        sufficient = f_trial <= f_target
        new_done = done | sufficient

        # Quadratic interpolation (first time)
        tmplam_quad = -slope / (2.0 * (f_trial - fold - slope + 1e-30))

        # Cubic interpolation (subsequent)
        rhs1 = f_trial - fold - alam * slope
        rhs2 = f2 - fold - alam2 * slope
        a1 = (rhs1 / (alam * alam) - rhs2 / (alam2 * alam2 + 1e-30)) / (alam - alam2 + 1e-30)
        a2 = (-alam2 * rhs1 / (alam * alam) + alam * rhs2 / (alam2 * alam2 + 1e-30)) / (alam - alam2 + 1e-30)

        disc = a2 * a2 - 3.0 * a1 * slope
        tmplam_cubic = jnp.where(
            jnp.abs(a1) < 1e-30,
            -slope / (2.0 * a2 + 1e-30),
            jnp.where(
                disc < 0.0,
                alam * _ARMIJO_FACTOR,
                jnp.where(
                    a2 <= 0.0,
                    (-a2 + jnp.sqrt(jnp.maximum(disc, 0.0))) / (3.0 * a1 + 1e-30),
                    -slope / (a2 + jnp.sqrt(jnp.maximum(disc, 0.0)) + 1e-30)
                )
            )
        )

        tmplam = jnp.where(first_time, tmplam_quad, tmplam_cubic)
        tmplam = jnp.maximum(tmplam, alam * _ARMIJO_FACTOR)
        tmplam = jnp.minimum(tmplam, alam * 0.5)

        new_alam = jnp.where(new_done, alam, jnp.maximum(tmplam, _ARMIJO_MIN_ALPHA))
        new_alam2 = jnp.where(new_done, alam2, alam)
        new_f2 = jnp.where(new_done, f2, f_trial)
        new_first_time = jnp.where(new_done, first_time, jnp.bool_(False))

        return (new_alam, new_alam2, new_f2, new_first_time, new_done), None

    init = (jnp.float64(1.0), jnp.float64(0.5), fold, jnp.bool_(True), jnp.bool_(False))
    (alpha_final, _, _, _, _), _ = lax.scan(backtrack_step, init, None, length=_ARMIJO_MAX_ITER)

    return lax.stop_gradient(alpha_final)  # GP-9: CONVERGENCE_FLAGS
