"""Standalone robust Henyey solver with Jacobian conditioning.

.. note:: **LEGACY** — the production solver is ``solver/continuation.py``
   (continuation-based, atmosphere BC, Armijo line search). This module
   is retained for diagnostic/standalone use and as a reference for the
   conditioning algorithm. Relocated from top-level henyey_conditioning.py
   to solver/ in #1146 (cohesion cleanup).

It is a STANDALONE module:
  - Production code (evolution.py, the continuation solvers in henyey.py) does
    NOT import from here. This preserves henyey.py line numbers (JAX cache-key
    stability) and ensures zero production regression.
  - This module imports FROM solver/conditioning.py for the shared conditioning
    helpers (_condition_system, _block_thomas_solve_conditioned, conditioned_solve,
    armijo_line_search) — one source, no duplication.
  - Tests call conditioned_newton_armijo directly — never evolve_star.

Key capabilities beyond the base Newton in henyey.py:
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
# Shared conditioning helpers — one source in solver/conditioning.py
# Access via module reference (not from-import) so that monkeypatch in
# tests/mutations.py can intercept the call at runtime.
# ─────────────────────────────────────────────────────────────────────────────
import stellar_jax.solver.conditioning as _cond

# Re-export for backward compat (test imports from henyey_conditioning)
_condition_system = _cond._condition_system
_block_thomas_solve_conditioned = _cond._block_thomas_solve_conditioned
conditioned_solve = _cond.conditioned_solve
armijo_line_search = _cond.armijo_line_search

# Constants from solver/conditioning.py
_TOL_RESIDUAL = _cond._TOL_RESIDUAL
_TOL_MAX_CORRECTION = _cond._TOL_MAX_CORRECTION


# ─────────────────────────────────────────────────────────────────────────────
# Full conditioned Newton solver with Armijo + dual convergence
# ─────────────────────────────────────────────────────────────────────────────

def conditioned_newton_armijo(y_init, q_mesh, M_star, X_profile, Z, alpha_mlt,
                              n_iter=50, tol=_TOL_RESIDUAL, tol_c=_TOL_MAX_CORRECTION):
    """Newton solver with conditioning + Armijo + dual convergence.

    Combines:
      1. Row equilibration + Levenberg (vanishes at convergence → IFT exact)
      2. Per-zone damping (henyey._apply_per_zone_damping)
      3. Armijo backtracking line-search (guaranteed descent)
      4. Dual convergence: max|R| < tol AND max|dy/xscale| < tol_c

    Parameters
    ----------
    y_init : (N, 4) — initial guess [ln_r, ln_P, ln_T, ℓ].
    q_mesh : (N+1,) — mass-fraction mesh.
    M_star : scalar — stellar mass in grams.
    X_profile : (N_COMP,) — hydrogen composition profile.
    Z : scalar — metallicity.
    alpha_mlt : scalar — MLT parameter.
    n_iter : int — max Newton iterations.
    tol : float — residual tolerance (default 1e-8).
    tol_c : float — correction tolerance (default 3e-3).

    Returns
    -------
    dict with 'y', 'converged', 'residual_norm', 'n_iter_used'.
    """
    from stellar_jax.henyey import _build_residual, _jacobian_blocks, _apply_per_zone_damping

    def residual_fn(y):
        return _build_residual(y, q_mesh, M_star, X_profile, Z, alpha_mlt)

    def newton_step(carry, _):
        y, converged, n_done = carry
        R = residual_fn(y)
        R_norm = jnp.max(jnp.abs(R))

        # Conditioned solve
        A, B, C = _jacobian_blocks(y, q_mesh, M_star, X_profile, Z, alpha_mlt)
        dy = _cond.conditioned_solve(A, B, C, -R, R_norm, levenberg=True)

        # Per-zone damping
        dy = _apply_per_zone_damping(dy, R_norm)

        # Armijo backtracking
        alpha = _cond.armijo_line_search(y, dy, R, residual_fn, R_norm)
        dy_final = alpha * dy

        # Dual convergence (MESA star_solver.f90:496-509)
        xscale = jnp.maximum(1.0, jnp.abs(y))
        max_corr = jnp.max(jnp.abs(dy_final) / xscale)
        converged_new = converged | ((R_norm < tol) & (max_corr < tol_c))

        # Freeze once converged
        dy_final = jnp.where(converged, 0.0, dy_final)
        y_new = y + dy_final
        n_done_new = n_done + jnp.where(converged, 0, 1)

        return (y_new, converged_new, n_done_new), R_norm

    init = (y_init, jnp.bool_(False), jnp.int32(0))
    (y_final, converged_final, n_used), residuals = lax.scan(
        newton_step, init, None, length=n_iter)

    # Final residual for diagnostics
    R_final = residual_fn(y_final)
    res_norm = jnp.max(jnp.abs(R_final))

    return {
        'y': y_final,
        'converged': converged_final | (res_norm < tol),
        'residual_norm': res_norm,
        'n_iter_used': n_used,
    }
