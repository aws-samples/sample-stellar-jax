"""Schwarzschild/Ledoux convective/radiative switch with custom_jvp for alpha_mlt adjoint.

Architecture:
  - Forward: exact jnp.where (sharp boundary per KWW §6.2 / MESA turb_support.f90:386).
  - JVP: sigmoid value-blend so alpha_mlt adjoint flows through the
    convective-radiative boundary even when a zone is marginally radiative.

The @custom_jvp decorator and .defjvp registration MUST stay in this file
(JAX registers by function identity — splitting them across files breaks
registration silently). See the module redesign spec §2.

References:
  - Griewank & Walther 2008, §14 (nonsmooth AD regularization).
  - Kippenhahn, Weigert & Weiss §6.2 (Schwarzschild criterion).
  - MESA turb/private/mlt.f90 (Henyey option, gradT selection logic).
  - MESA star/private/turb_support.f90:386 (if gradr > gradL then).
"""

import jax
import jax.numpy as jnp


@jax.custom_jvp
def _mlt_switch(nabla_rad, gradL, grad_conv):
    """Schwarzschild/Ledoux switch for interior MLT: hard where forward, value-blend JVP.

    Forward: exact jnp.where (sharp boundary per KWW §6.2 / MESA turb_support.f90:386).
    JVP: sigmoid value-blend so alpha_mlt adjoint flows through the
    convective-radiative boundary even when a zone is marginally radiative.
    This is needed because the RK2 atmosphere upgrade shifts interior zone
    positions across the boundary for hot stars (M≥2 M☉), severing the
    alpha path through the hard where. The value-blend keeps a nonzero
    alpha adjoint via w*d_conv (w small but nonzero on the radiative side).

    The second argument is gradL (= nad + gradL_composition_term for Ledoux,
    or just nad for Schwarzschild). Convection when nabla_rad > gradL.

    No dw_dx boundary-position terms: for the interior MLT switch,
    delta = grad_conv - nabla_rad → 0 at the boundary
    (MLT gives grad_conv → nabla_rad as nabla_rad → gradL), so they would
    contribute negligibly and only add noise.
    """
    return jnp.where(nabla_rad > gradL, grad_conv, nabla_rad)


@_mlt_switch.defjvp
def _mlt_switch_jvp(primals, tangents):
    nabla_rad, gradL, grad_conv = primals
    d_rad, d_gradL, d_conv = tangents
    primal_out = jnp.where(nabla_rad > gradL, grad_conv, nabla_rad)
    # eps=0.005: ~1.25% of gradL (~0.4). Tight enough that zones clearly
    # convective (nabla_rad - gradL > 0.01) get w>0.88 (close to hard-where's
    # 1.0), reducing AD-vs-FD mismatch for M=1.5 with its thin envelope.
    # Still provides alpha adjoint for M=2.0: zones at nabla_rad - gradL = -0.005
    # get w=sigmoid(-1)=0.27, keeping a nonzero alpha path.
    _eps = 0.005
    w = jax.nn.sigmoid((nabla_rad - gradL) / _eps)
    tangent_out = w * d_conv + (1.0 - w) * d_rad
    return primal_out, tangent_out
