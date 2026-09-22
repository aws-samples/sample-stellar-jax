"""Differentiability contract for the solver module.

The solver is the STRUCTURE DIFFERENTIABILITY BOUNDARY. It provides:
  ∂y*/∂θ where y* satisfies F(y*, θ) = 0
  via the Implicit Function Theorem: ∂y*/∂θ = −(∂F/∂y)⁻¹ (∂F/∂θ)

DIFFERENTIABLE inputs (gradients flow through IFT):
  - y_prev: (N_s, 4) — initial state [ln_r, ln_P, ln_T, ell]
  - q_mesh: (N_s+1,) — mass mesh [differentiable in principle; currently dead]
  - M_star: scalar — stellar mass in grams
  - X_profile: (N_COMP,) — hydrogen composition
  - Z: scalar — metallicity
  - alpha_mlt: scalar — MLT parameter
  - atm_ratio: scalar — R_phot/r_surf geometry constant
  - ln_T_prev: (N_s,) — thermal history (eps_grav backward)
  - ln_P_prev: (N_s,) — pressure history (eps_grav backward)
  - inv_dt: scalar — 1/dt for eps_grav coupling
  - N14_profile: (N_COMP,) — tracked nitrogen (#393 CNO feedback)
  - comp_mfracs: (N_COMP,) — composition grid positions
    [STOP_GRADIENT'D UPSTREAM — zero gradient by construction; closed over
     in backward VJP, not positional]
  - opacity_factor: scalar — multiplicative opacity knob (#467)

NONDIFF inputs (static Python values, NOT traced):
  - n_iter: int — max Newton iterations (nondiff_argnums position 7)
  - tol: float — convergence tolerance (nondiff_argnums position 8)
  - bypass_conv_gate: bool — skip convergence gating (nondiff_argnums position 9)

@custom_vjp BOUNDARY:
  _henyey_continuation_atm
    nondiff_argnums = (7, 8, 9) → n_iter, tol, bypass_conv_gate
    Positional signature (FRAGILE — must not reorder):
      (y_prev, q_mesh, M_star, X_profile, Z, alpha_mlt, atm_ratio,
       n_iter, tol, bypass_conv_gate,
       ln_T_prev, ln_P_prev, inv_dt, N14_profile, comp_mfracs, opacity_factor)
    Positions 0-6: differentiable
    Positions 7-9: nondiff (static)
    Positions 10-15: differentiable

CONVERGENCE CONTRACT:
  The IFT is valid iff max|F(y*, θ)| < 10*tol at the converged state.
  If not converged, all output gradients are ZEROED (convergence gate).
  bypass_conv_gate=True skips the gate (for schedule-replay mode, #374).

OUTPUT: (y_final: (N_s, 4), converged: bool)
  y_final is differentiable via IFT; converged is not.
"""

import inspect


# Expected parameter names for _henyey_continuation_atm in exact positional order.
# This is the COMPILE-TIME assertion that catches any accidental reorder.
_CONTINUATION_ATM_PARAM_NAMES = (
    'y_prev', 'q_mesh', 'M_star', 'X_profile', 'Z', 'alpha_mlt', 'atm_ratio',
    'n_iter', 'tol', 'bypass_conv_gate',
    'ln_T_prev', 'ln_P_prev', 'inv_dt', 'N14_profile', 'comp_mfracs', 'opacity_factor',
    'eps_nuc_factor',
)

# Expected nondiff_argnums for each @custom_vjp boundary.
_NONDIFF_ARGNUMS = {
    '_henyey_continuation_atm': (7, 8, 9, 21, 22),
    '_henyey_newton': (6,),
    '_henyey_eps_grav_ift_impl': (8, 9),
}


def assert_continuation_atm_signature(fn):
    """Compile-time assertion: verify _henyey_continuation_atm's parameter order.

    Call this at module load time (in continuation.py) to catch any accidental
    signature reorder. A 1-position shift in nondiff_argnums silently produces
    wrong gradients with NO error — this is the ONLY defense besides golden-master.

    Parameters
    ----------
    fn : the _henyey_continuation_atm function (before or after @custom_vjp wrapping)

    Raises
    ------
    AssertionError if the parameter names don't match the expected order.
    """
    # Get the underlying function if wrapped by custom_vjp
    target = fn
    if hasattr(fn, '__wrapped__'):
        target = fn.__wrapped__
    elif hasattr(fn, 'fun'):
        # JAX custom_vjp stores the original as .fun
        target = fn.fun

    sig = inspect.signature(target)
    param_names = tuple(sig.parameters.keys())

    # Check the first len(_CONTINUATION_ATM_PARAM_NAMES) params match
    expected = _CONTINUATION_ATM_PARAM_NAMES
    actual = param_names[:len(expected)]

    assert actual == expected, (
        f"@custom_vjp signature MISMATCH — nondiff_argnums=(7,8,9,21,22) will be WRONG!\n"
        f"Expected: {expected}\n"
        f"Actual:   {actual}\n"
        f"This SILENTLY corrupts all gradients. Fix the signature order."
    )
