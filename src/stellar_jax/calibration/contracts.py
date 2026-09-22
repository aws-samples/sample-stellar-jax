"""Differentiability contract for calibration/

This module documents which functions are JAX-differentiable and which are not,
and the stop_gradient/custom_vjp policy for the calibration package.

DIFFERENTIABLE (jax.grad-compatible):
  - solar_residual(alpha_mlt, Y_init, ...) -> (logL, logR)
    Gradient flows: alpha_mlt -> evolve_solar -> lax.scan body -> logL/logR
    via quadratic Lagrange interpolation (fully differentiable).
  - solar_residual_henyey(alpha_mlt, Y_init, ...) -> (logL, logR, X_c)
    Gradient flows: alpha_mlt, Y_init -> evolve_star -> lax.scan body -> logL/logR
    via jnp.interp (linear) or Lagrange (quadratic).
  - evolve_solar(...) -- the JIT-compiled lax.scan kernel

NOT DIFFERENTIABLE (Python control flow, I/O, or numpy):
  - solar_calibrate* -- Python Newton-Raphson loop with print statements
  - compare_model_s -- loads Model S .dat files, numpy operations
  - compare_mesa -- loads MESA history.data, numpy interpolation
  - evolve_star_diagnostic -- non-JIT diagnostic path with numpy post-processing

stop_gradient: NONE in this module. This module does not insert any stop_gradient;
    the gradient-window policy is enforced by evolution/ (the orchestrator).
    evolve_solar's internal step_fn uses stop_gradient on structural_reject only
    (algorithmic control flow, matching the orchestrator pattern).

@custom_vjp / @custom_jvp: NONE. This module has no custom differentiation rules.
    It relies on the upstream IFT boundaries in solver/ and oscillations/.
"""
