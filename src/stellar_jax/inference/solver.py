"""Gradient-based inverse solver: observables → stellar parameters.

Implements Levenberg-Marquardt (damped Gauss-Newton) optimization to infer
stellar parameters (M, Z, α_MLT, Y₀) from observed quantities (L, Teff, R,
oscillation frequencies) using the analytic gradients that flow through the
differentiable forward model.

The key advantage over traditional grid-search or finite-difference approaches
is that jax.jacrev provides EXACT Jacobians at the cost of a single backward
pass — no FD noise, no grid resolution limits.

COMPILE-ONCE ARCHITECTURE (issue #526):
---------------------------------------
When forward_fn is differentiable and jacobian_fn is None (the default path),
the LM iteration uses jit-compiled residual and Jacobian functions:
- residuals_fn compiles ONCE (first jit call), then XLA reuses the compiled code
- jax.jacrev(residuals_fn) compiles ONCE
- The Python-level for-loop does NOT cause recompilation — jax.jit caches by
  function identity + input shapes/types, both of which are stable across iters
- Target: compiles ~22k → O(10) for the seismic inversion test

We use a PYTHON loop (not lax.while_loop) because lax.while_loop absorbs
jax.jit calls into its body trace, compiling the FULL evolve_star forward +
backward into one monolithic XLA program that OOMs on heavy forward_fn.

Algorithm: Levenberg-Marquardt (damped Gauss-Newton)
---------------------------------------------------
Given observed quantities y_obs with uncertainties σ, and a forward model f(θ):

1. Compute residuals: r(θ) = (f(θ) - y_obs) / σ
2. Compute Jacobian: J = ∂r/∂θ  (via jax.jacrev — exact, automatic)
3. Solve the damped normal equations: (J^T J + λ I) δθ = -J^T r
4. Update θ ← θ + δθ
5. Adjust λ based on gain ratio ρ = (χ²_old - χ²_new) / predicted_reduction
6. Converge when ||δθ/θ|| < tol or ||r||² < χ²_tol

Returns the Jacobian at solution for error-bar estimation:
    Cov(θ) ≈ (J^T J)^{-1} · s²  where s² = χ²/(n_obs - n_params)

References:
    Deheuvels, S. — "Automatic search for optimal models using Levenberg-Marquardt"
        (PLATO-STESCI, https://plato-stesci.lesia.obspm.fr/sites/plato-stesci/IMG/pdf/deheuvels.pdf)
    Transtrum, M., Machta, B., Sethna, J. (2012) — "Improvements to the
        Levenberg-Marquardt algorithm", arXiv:1201.5885
    Chaplin, W. & Miglio, A. (2013) — "Asteroseismology of Solar-Type and
        Red-Giant Stars", ARA&A 51, 353, arXiv:1303.1957
    Press et al. (2007) — Numerical Recipes, §15.5 (Levenberg-Marquardt method)
"""

import jax
import jax.numpy as jnp
import numpy as np


def _lm_step(theta, r, chi2, lam, jacobian_fn, residuals_fn, bounds_lo,
             bounds_hi, lambda_up, lambda_down):
    """Single Levenberg-Marquardt iteration. Returns updated state.

    NaN-safe: jnp.linalg.solve returns NaN (not a Python exception) on
    singular matrices. We detect NaN in delta and treat it as a rejected
    step with increased λ — equivalent to the singular-matrix branch in
    Press NR §15.5.

    Parameters
    ----------
    bounds_lo, bounds_hi : jnp.ndarray
        Vectorized lower/upper bounds for jnp.clip (shape n_params).
        Use -inf/inf for unbounded.
    """
    J = jacobian_fn(theta)
    JtJ = J.T @ J
    Jtr = J.T @ r
    diag_JtJ = jnp.diag(jnp.maximum(jnp.diag(JtJ), 1e-30))
    A = JtJ + lam * diag_JtJ

    delta = jnp.linalg.solve(A, -Jtr)

    # NaN from singular (J^T J + λ diag) → reject step, increase λ.
    # jnp.linalg.solve returns NaN on singular matrices (no Python exception).
    if bool(jnp.any(jnp.isnan(delta))):
        return theta, r, chi2, lam * lambda_up, None, 'singular'

    theta_trial = jnp.clip(theta + delta, bounds_lo, bounds_hi)
    r_trial = residuals_fn(theta_trial)
    chi2_trial = float(jnp.sum(r_trial ** 2))

    # NaN from forward model failure at trial point → reject.
    if not np.isfinite(chi2_trial):
        return theta, r, chi2, lam * lambda_up, None, 'reject'

    predicted = float(-2.0 * delta @ Jtr - delta @ JtJ @ delta)
    rho = ((chi2 - chi2_trial) / predicted) if predicted > 0 else (
        0.0 if chi2_trial < chi2 else -1.0)
    rel_step = float(jnp.max(jnp.abs(delta) / jnp.maximum(jnp.abs(theta), 1e-30)))

    # Jacobian condition number for diagnostics (ill-conditioning detection).
    j_cond = float(jnp.linalg.cond(J))

    info = {'chi2_trial': chi2_trial, 'rho': rho, 'rel_step': rel_step,
            'cond_J': j_cond}
    if chi2_trial < chi2:
        if rho > 0.75:
            lam *= lambda_down
        elif rho < 0.25:
            lam *= lambda_up
        return theta_trial, r_trial, chi2_trial, lam, info, 'accept'
    return theta, r, chi2, lam * lambda_up, info, 'reject'


def _lm_covariance(theta, chi2, n_obs, n_params, jacobian_fn):
    """Compute parameter covariance from the final Jacobian.

    Uses pseudoinverse when J^T J is singular or ill-conditioned
    (jnp.linalg.inv returns NaN or inf, not a Python exception).
    Ref: Press NR §15.5, Cov(θ) ≈ (J^T J)^{-1} · s².
    """
    J_final = jacobian_fn(theta)
    JtJ = J_final.T @ J_final
    dof = max(n_obs - n_params, 1)
    s2 = chi2 / dof
    cov = jnp.linalg.inv(JtJ) * s2
    # jnp.linalg.inv returns NaN or inf on singular matrices → fall back to pinv.
    if not bool(jnp.all(jnp.isfinite(cov))):
        cov = jnp.linalg.pinv(JtJ) * s2
    sigma_params = jnp.sqrt(jnp.maximum(jnp.diag(cov), 0.0))
    return J_final, cov, sigma_params


def _build_bounds_arrays(bounds, n_params):
    """Build vectorized bounds arrays for jnp.clip."""
    if bounds is not None:
        lo_list = [b[0] if b[0] is not None else -jnp.inf for b in bounds]
        hi_list = [b[1] if b[1] is not None else jnp.inf for b in bounds]
        bounds_lo = jnp.array(lo_list, dtype=jnp.float64)
        bounds_hi = jnp.array(hi_list, dtype=jnp.float64)
    else:
        bounds_lo = jnp.full(n_params, -jnp.inf)
        bounds_hi = jnp.full(n_params, jnp.inf)
    return bounds_lo, bounds_hi


def _setup_lm_functions(residuals_fn, jacobian_fn):
    """Set up JIT-compiled residual and Jacobian functions for the LM loop.

    COMPILE-ONCE (issue #526): when jacobian_fn is None, wrap both
    residuals_fn and jacrev(residuals_fn) in jax.jit so each compiles
    ONCE on first call and is cached for all subsequent LM iterations.
    """
    if jacobian_fn is None:
        _jacobian_fn = jax.jit(jax.jacrev(residuals_fn))
        _residuals_fn = jax.jit(residuals_fn)
    else:
        _jacobian_fn = jacobian_fn
        _residuals_fn = residuals_fn
    return _jacobian_fn, _residuals_fn


def _run_lm_loop(theta, observed, sigma, n_obs, n_params, param_names,
                  _jacobian_fn, _residuals_fn, bounds_lo, bounds_hi,
                  max_iter, lambda_init, lambda_up, lambda_down,
                  tol_param, tol_chi2):
    """Execute the Levenberg-Marquardt iteration loop and return results."""
    lam = lambda_init
    history = []
    converged = False

    # Evaluate at initial point
    r = _residuals_fn(theta)
    chi2 = float(jnp.sum(r ** 2))

    for iteration in range(max_iter):
        chi2_old = chi2
        theta, r, chi2, lam, info, status = _lm_step(
            theta, r, chi2, lam, _jacobian_fn, _residuals_fn, bounds_lo,
            bounds_hi, lambda_up, lambda_down)
        rel_step = info['rel_step'] if info else 0.0
        record = {'iter': iteration, 'chi2': chi2_old, 'lambda': lam,
                  'theta': np.array(theta)}
        if info:
            record.update({k: info[k] for k in
                           ('chi2_trial', 'rho', 'rel_step', 'cond_J')
                           if k in info})
        record['status'] = status
        history.append(record)
        if status == 'reject' or status == 'singular':
            continue
        if rel_step < tol_param or chi2 < tol_chi2:
            converged = True
            break

    # Final Jacobian at solution (for error bars)
    J_final, cov, sigma_params = _lm_covariance(
        theta, chi2, n_obs, n_params, _jacobian_fn)

    n_accepted = sum(1 for h in history if h.get('status') == 'accept')

    return {
        'theta': theta,
        'chi2': chi2,
        'jacobian': J_final,
        'covariance': cov,
        'sigma_params': sigma_params,
        'converged': converged,
        'n_iter': iteration + 1,
        'n_accepted': n_accepted,
        'history': history,
        'residuals': r,
        'param_names': param_names,
    }


def infer_parameters(
    forward_fn,
    observed,
    sigma,
    theta_init=None,
    param_names=None,
    max_iter=50,
    lambda_init=1e-3,
    lambda_up=10.0,
    lambda_down=0.1,
    tol_param=1e-4,
    tol_chi2=1e-6,
    bounds=None,
    warmstart_fn=None,
    jacobian_fn=None,
):
    """Infer stellar parameters from observed quantities using analytic gradients.

    Levenberg-Marquardt optimizer exploiting the end-to-end differentiability
    of the forward stellar model (evolve_star → observables).

    COMPILE-ONCE (issue #526): when jacobian_fn is None (fully JAX-traceable
    pipeline), residuals and Jacobian are wrapped in jax.jit so each compiles
    ONCE on first call and is cached for all subsequent LM iterations.
    Compiles O(1) times regardless of max_iter — no re-tracing per iteration.
    When jacobian_fn is provided externally (two-level forward: Brent + IFT),
    falls back to a Python loop with un-jitted residuals.

    Parameters
    ----------
    forward_fn : callable
        Maps parameter vector θ (1D JAX array) → predicted observables (1D JAX array).
        Must be differentiable via jax.jacrev (unless jacobian_fn is provided).
    observed : array-like, shape (n_obs,)
        Observed values (e.g., [log_L, log_Teff] at a known age).
    sigma : array-like, shape (n_obs,)
        1-σ uncertainties on the observables.
    theta_init : array-like, shape (n_params,)
        Initial guess for the parameters.
    param_names : list of str, optional
        Names for each parameter (for diagnostics).
    max_iter : int
        Maximum number of LM iterations.
    lambda_init : float
        Initial damping parameter λ. Large λ → gradient descent; small → Gauss-Newton.
    lambda_up : float
        Factor to increase λ on a rejected step (ρ < 0.25).
    lambda_down : float
        Factor to decrease λ on an accepted step (ρ > 0.75).
    tol_param : float
        Convergence tolerance on relative parameter change ||δθ/θ||.
    tol_chi2 : float
        Convergence tolerance on absolute χ² value.
    bounds : list of tuples or None
        Parameter bounds [(lo, hi), ...] for each parameter. None = unbounded.
    warmstart_fn : callable or None
        Optional warm-start function. If provided, called as
        warmstart_fn(observed, sigma, param_names) → 1D array to use as
        the initial guess, OVERRIDING theta_init. This enables emulator-guided
        warm-starts (e.g. via DSEE) that reduce iteration count without
        changing the final answer. The warmstart_fn is NOT in the gradient
        path — it runs once, before the LM loop, to produce a better
        starting point.
        When warmstart_fn is None (the default), theta_init is used directly
        (cold start / existing behavior).
    jacobian_fn : callable or None
        Optional external Jacobian function. If provided, called as
        jacobian_fn(theta) → array of shape (n_obs, n_params), replacing
        the internal jax.jacrev(residuals_fn). This enables two-level forward
        architectures where the Jacobian is computed by a separate differentiable
        path (e.g. IFT per-mode jax.grad with frozen structure anchors) while
        residuals come from non-JAX root-finding (Brent).
        When None (default), uses jax.jacrev + jax.jit (compile-once).

    Returns
    -------
    dict with keys:
        'theta' : final parameter estimates (JAX array)
        'chi2' : final χ² value
        'jacobian' : Jacobian at solution, shape (n_obs, n_params)
        'covariance' : estimated parameter covariance matrix (n_params, n_params)
        'sigma_params' : 1-σ parameter uncertainties (diagonal of sqrt(cov))
        'converged' : bool
        'n_iter' : number of iterations used
        'n_accepted' : number of accepted LM steps (chi2 decreased)
        'history' : list of dicts with per-iteration diagnostics, including
            'cond_J' (Jacobian condition number) for ill-conditioning detection
        'residuals' : final residuals (f(θ) - observed) / σ
    """
    observed = jnp.asarray(observed, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)

    # Resolve initial guess: warmstart_fn overrides theta_init when present.
    # This is the B2 warm-start flag: providing warmstart_fn enables the emulator
    # warm-start; omitting it (or setting it to None) = cold start from theta_init.
    if warmstart_fn is not None:
        theta_init = warmstart_fn(observed, sigma, param_names)
    if theta_init is None:
        raise ValueError(
            "theta_init is required when warmstart_fn is not provided. "
            "Either pass theta_init explicitly or provide a warmstart_fn."
        )

    theta = jnp.asarray(theta_init, dtype=jnp.float64)
    n_obs = observed.shape[0]
    n_params = theta.shape[0]

    if param_names is None:
        param_names = [f"p{i}" for i in range(n_params)]

    # Residual function: r(θ) = (f(θ) - y_obs) / σ
    def residuals_fn(th):
        pred = forward_fn(th)
        return (pred - observed) / sigma

    bounds_lo, bounds_hi = _build_bounds_arrays(bounds, n_params)

    _jacobian_fn, _residuals_fn = _setup_lm_functions(
        residuals_fn, jacobian_fn)

    return _run_lm_loop(
        theta, observed, sigma, n_obs, n_params, param_names,
        _jacobian_fn, _residuals_fn, bounds_lo, bounds_hi,
        max_iter, lambda_init, lambda_up, lambda_down,
        tol_param, tol_chi2)
