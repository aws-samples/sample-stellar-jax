"""MESA nuclear adapter: forward-only epsilon_nuclear() via jax.pure_callback.

Same signature as microphysics.nuclear.epsilon_nuclear:
    epsilon_nuclear(rho, T, X, Z, t_age=1e9) -> eps_nuc [erg/g/s]

Uses the generic net path (pp_cno_extras_o18_ne22.net) via the bind(C)
net_wrapper shim, which properly initializes the full MESA nuclear network
(net_start_def → read_net_file → net_finish_def → net_setup_tables).

This is a VALIDATION backend — forward-only, not differentiable.

References:
  - Caughlan & Fowler (1988), ADNDT 40, 283 (reaction rates)
  - Angulo et al. (1999), Nucl. Phys. A 656, 3 (NACRE rates)
  - Paxton et al. (2011), ApJS 192, 3, §3 (MESA net module)
"""
import jax
import jax.numpy as jnp
import numpy as np

from stellar_jax.microphysics.mesa.bindings import call_mesa_net


def _net_callback(args):
    """NumPy callback for MESA nuclear energy generation."""
    rho_f = float(args[0])
    T_f = float(args[1])
    X_f = float(args[2])
    Z_f = float(args[3])

    eps_nuc = call_mesa_net(rho_f, T_f, X_f, Z_f)
    return np.array(eps_nuc, dtype=np.float64)


def epsilon_nuclear(rho, T, X, Z, t_age=1e9, X_N14=None, eps_nuc_factor=None):
    """MESA nuclear energy generation (same signature as JAX backend).

    Forward-only — NOT differentiable (validation backend).

    Note: t_age, X_N14, and eps_nuc_factor are accepted for signature
    compatibility with the JAX backend (microphysics/contracts.py) but:
    - t_age: MESA's network does not use it directly
    - X_N14: MESA's full network handles N14 internally
    - eps_nuc_factor: physics-calibration knob, ignored in validation backend

    Returns: eps_nuc [erg/g/s]
    """
    args = jnp.stack([rho, T, X, Z])

    result = jax.pure_callback(
        _net_callback,
        np.zeros((), dtype=np.float64),
        args,
        vmap_method='sequential',
    )
    return result
