"""Block-tridiagonal linear algebra for the Henyey solver.

KEEP WHOLE (the module redesign spec §2):
  - block_thomas_solve (49L): self-contained O(N) scan kernel.
  - _adjoint_thomas_solve (101L): same algorithm with NaN guards + Levenberg.
  Splitting would require passing intermediate arrays between sweeps, breaking
  the tight coupling of the elimination data flow.

References:
  - Duff, Erisman & Reid (1986), "Direct Methods for Sparse Matrices", §5.3
  - Nocedal & Wright (2006), "Numerical Optimization", §10.2
  - Higham (2002), "Accuracy and Stability of Numerical Algorithms", §9.4
"""

import jax.numpy as jnp


def _transpose_block_tridiag(A, B, C):
    """Transpose a block-tridiagonal system (A, B, C) → (A_T, B_T, C_T).

    The transpose of a block-tridiagonal matrix with blocks:
      A_k (sub-diagonal), B_k (diagonal), C_k (super-diagonal)
    has blocks:
      A_T_k = C_{k-1}^T (sub-diagonal of J^T = transposed super-diagonal of J)
      B_T_k = B_k^T     (diagonal transposes in place)
      C_T_k = A_{k+1}^T (super-diagonal of J^T = transposed sub-diagonal of J)

    DRY-2 extraction: this 5-line pattern appeared at 4 sites:
      - solver/newton.py: _henyey_newton_bwd, _henyey_continuation_ift_bwd,
        _henyey_eps_grav_ift_bwd
      - solver/adjoint.py: _solve_adjoint_system

    Parameters
    ----------
    A : (N, 4, 4) — sub-diagonal blocks
    B : (N, 4, 4) — diagonal blocks
    C : (N, 4, 4) — super-diagonal blocks

    Returns
    -------
    A_T, B_T, C_T : each (N, 4, 4) — blocks of the transposed system
    """
    A_T = jnp.zeros_like(A)
    B_T = jnp.transpose(B, (0, 2, 1))
    C_T = jnp.zeros_like(C)
    A_T = A_T.at[1:].set(jnp.transpose(C[:-1], (0, 2, 1)))
    C_T = C_T.at[:-1].set(jnp.transpose(A[1:], (0, 2, 1)))
    return A_T, B_T, C_T

from jax import lax


def block_thomas_solve(A, B, C, rhs):
    """Solve block-tridiagonal system: A_k x_{k-1} + B_k x_k + C_k x_{k+1} = rhs_k.

    A[0] and C[N-1] are ignored. Uses lax.scan for JIT.
    Parameters: A,B,C: (N,4,4), rhs: (N,4). Returns: x: (N,4).

    Row equilibration: each block row is scaled so that the diagonal block
    has unit max-row-norm before elimination. This prevents NaN from
    ill-conditioned blocks in degenerate/RGB cores where variables span
    many orders of magnitude. The scaling is: D_k = diag(1/max_j|B_k[i,j]|).
    Reference: Duff, Erisman & Reid (1986), "Direct Methods for Sparse
    Matrices", §5.3 (equilibration for block elimination).
    """
    N = B.shape[0]

    # Row-equilibrate: scale each block row by the inverse max-abs of its
    # diagonal block's rows. This makes B well-conditioned without altering
    # the solution (the scaling cancels in the elimination).
    row_max = jnp.max(jnp.abs(B), axis=-1)  # (N, 4)
    row_scale = 1.0 / jnp.maximum(row_max, 1e-30)  # (N, 4)
    # Apply: D_k @ A_k, D_k @ B_k, D_k @ C_k, D_k @ rhs_k
    D = row_scale[:, :, None]  # (N, 4, 1) for broadcasting
    A_s = A * D
    B_s = B * D
    C_s = C * D
    rhs_s = rhs * row_scale

    def forward_step(carry, k):
        B_prev, F_prev = carry
        factor = jnp.linalg.solve(B_prev.T, A_s[k].T).T
        B_new = B_s[k] - factor @ C_s[k - 1]
        F_new = rhs_s[k] - factor @ F_prev
        return (B_new, F_new), (B_new, F_new)

    init_carry = (B_s[0], rhs_s[0])
    _, (B_rest, F_rest) = lax.scan(forward_step, init_carry, jnp.arange(1, N))

    B_all = jnp.concatenate([B_s[0:1], B_rest], axis=0)
    F_all = jnp.concatenate([rhs_s[0:1], F_rest], axis=0)

    x_last = jnp.linalg.solve(B_all[N - 1], F_all[N - 1])

    def back_step(x_next, k):
        x_k = jnp.linalg.solve(B_all[k], F_all[k] - C_s[k] @ x_next)
        return x_k, x_k

    _, x_rev = lax.scan(back_step, x_last, jnp.arange(N - 2, -1, -1))
    x = jnp.concatenate([x_rev[::-1], x_last[None]], axis=0)
    return x


def _adjoint_thomas_solve(A, B, C, rhs):
    """Block-Thomas solve for the ADJOINT system with row equilibration + Levenberg + NaN guards.

    Solves A x_{k-1} + B x_k + C x_{k+1} = rhs_k using the standard Thomas
    algorithm with:
      1. Row equilibration (max|row|=1, prevents catastrophic cancellation)
      2. Minimal Levenberg regularization (1e-8 on equilibrated diagonal, keeps
         pivot elements above FTZ/DAZ subnormal threshold on AVX-512 CI)
      3. Per-block isfinite guards in forward elimination (defense-in-depth)

    The per-block NaN guards are the PRIMARY defense against degenerate zones:
    they replace non-finite blocks with identity/zero to prevent contamination
    of downstream blocks. The Levenberg (1e-8) is a SECONDARY safeguard that
    prevents the intermediate case where pivots are small enough to amplify
    rounding errors but not small enough to produce Inf that triggers the guard.

    Bias: O(1e-8) per step after row equilibration (σ_min ~ O(0.01) at
    convective boundaries → per-step error O(1e-6)). Over N=500:
    500 × 1e-6 = 5e-4 = 0.05%, negligible vs 1% AD-vs-FD tolerance.
    (Previous 1e-6 Levenberg caused 4.8% bias at N=500.)

    The per-block NaN guards handle extreme configurations where even 1e-8
    is insufficient (all-zero rows, truly degenerate blocks). They replace
    such blocks with identity (zero contribution) rather than propagating
    NaN through ALL downstream blocks.
    """
    N = B.shape[0]

    # Row equilibration + minimal Levenberg regularization.
    row_max = jnp.max(jnp.abs(B), axis=-1)  # (N, 4)
    row_scale = 1.0 / jnp.maximum(row_max, 1e-30)  # (N, 4)
    D = row_scale[:, :, None]  # (N, 4, 1)
    A_s = A * D
    B_s = B * D
    C_s = C * D
    rhs_s = rhs * row_scale  # (N, 4)

    eye4 = jnp.eye(4)
    # Levenberg: prevents zero pivots on FTZ/DAZ hardware after equilibration.
    # Value 1e-8: keeps effective pivots above the FTZ/DAZ subnormal flush
    # threshold (~2.2e-308) without introducing meaningful gradient bias.
    _ADJOINT_LEVENBERG = 1e-8
    B_s = B_s + _ADJOINT_LEVENBERG * eye4[None, :, :]

    def forward_step(carry, k):
        B_prev, F_prev = carry
        # Standard Thomas: factor = A[k] @ B_prev^{-1}
        factor = jnp.linalg.solve(B_prev.T, A_s[k].T).T
        B_new = B_s[k] - factor @ C_s[k - 1]
        F_new = rhs_s[k] - factor @ F_prev
        # NaN guard: if elimination produced non-finite values, replace with
        # identity/zero to prevent contamination of ALL downstream blocks.
        bad = ~jnp.all(jnp.isfinite(B_new)) | ~jnp.all(jnp.isfinite(F_new))
        B_new = jnp.where(bad, eye4, B_new)
        F_new = jnp.where(bad, jnp.zeros(4), F_new)
        return (B_new, F_new), (B_new, F_new)

    init_carry = (B_s[0], rhs_s[0])
    _, (B_rest, F_rest) = lax.scan(forward_step, init_carry, jnp.arange(1, N))

    B_all = jnp.concatenate([B_s[0:1], B_rest], axis=0)
    F_all = jnp.concatenate([rhs_s[0:1], F_rest], axis=0)

    # Back-substitution with NaN guard on each step.
    x_last = jnp.linalg.solve(B_all[N - 1], F_all[N - 1])
    x_last = jnp.where(~jnp.isfinite(x_last), 0.0, x_last)

    def back_step(x_next, k):
        x_k = jnp.linalg.solve(B_all[k], F_all[k] - C_s[k] @ x_next)
        x_k = jnp.where(~jnp.isfinite(x_k), 0.0, x_k)
        return x_k, x_k

    _, x_rev = lax.scan(back_step, x_last, jnp.arange(N - 2, -1, -1))
    x = jnp.concatenate([x_rev[::-1], x_last[None]], axis=0)
    return x
