"""Differentiability contract for the transport module.

The transport module exposes two calling conventions for the same MLT physics:

1. mlt_nabla — DIFFERENTIABLE w.r.t. all inputs including alpha_mlt.
   Uses @custom_jvp (_mlt_switch) to provide a smooth alpha_mlt adjoint
   across the Schwarzschild boundary via sigmoid value-blend (eps=0.005).
   USE: IFT adjoint path, outer differentiable chain.

2. mlt_nabla_raw — DIFFERENTIABLE but with EXACT hard-switch derivative.
   Standard jnp.where AD (tangent = d_conv if convective, d_rad if radiative).
   USE: Newton Jacobian (jacfwd) where exact derivatives improve convergence.

Neither function contains stop_gradient. The stop_gradient policy is
owned by the orchestrator (evolution/_core.py), not by transport.

The @custom_jvp on _mlt_switch:
  - Forward: jnp.where(nabla_rad > nad, grad_conv, nabla_rad)
  - JVP: w*d_conv + (1-w)*d_rad, w = sigmoid((nabla_rad - nad)/eps), eps=0.005
  - Invariant: primal_out is EXACT (no smoothing in forward); only the tangent is blended.
  - Positional args: (nabla_rad, nad, grad_conv) — ALL are traced JAX values.
    No nondiff_argnums fragility here (unlike @custom_vjp in solver/).

MESA correspondence:
  MESA turb/private/mlt.f90 (calc_MLT subroutine) uses auto_diff_real_star_order1
  types for automatic partial derivatives. Our @custom_jvp is the JAX analog —
  it defines how tangents propagate through the convective/radiative selection,
  which is the only non-smooth operation in the MLT algebra.
"""
