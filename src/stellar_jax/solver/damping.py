"""Per-zone Newton correction damping.

References:
  - Paxton et al. (2011), ApJS 192, 3, §6.3 (MESA per-cell correction limiting)
  - Kippenhahn, Weigert & Weiss (2012), §11.2 (Henyey method, correction limiting)
"""

import jax.numpy as jnp


def _apply_per_zone_damping(dy, R_norm=None):
    """Per-zone (shell-aware) Newton correction limiting.

    Residual-gated hybrid: uses per-zone damping near the fixed point
    (quadratic convergence regime) and conservative global damping far
    from it (initial approach / continuation jumps).

    Near convergence (R_norm < 0.1):
        alpha_k = max(per_zone_k, global)
      - Well-behaved zones: per_zone=1.0 > global → FULL Newton step
      - Stiff zones: per_zone < global → global factor (inter-zone consistency)

    Far from convergence (R_norm >= 0.1):
        alpha_k = global  (same as old code — safe, linear convergence)

    The per-zone benefit: when the H-burning shell has large corrections
    (ε ∝ X²T¹⁶ for CNO), global damping throttles ALL ~1000 zones equally,
    reducing convergence to linear rate ~0.02. Near the solution, per-zone
    lets well-behaved zones (He core, envelope) take full steps and converge
    quadratically, while only the stiff shell is throttled.

    The residual gate prevents instability far from the solution where the
    linearization is unreliable: "well-behaved-looking" zones given full
    steps can drive the coupled system out of the convergence basin. This
    matches MESA practice where per-cell limiting operates within the
    quadratic regime only (Paxton et al. 2011, §6.3).

    Parameters
    ----------
    dy : (N_s, 4) array
        Raw Newton correction.
    R_norm : scalar or None
        Current residual norm. Activates per-zone benefit when < 0.1.

    References:
      - Paxton et al. (2011), ApJS 192, 3, §6.3 (MESA per-cell correction limiting)
      - Kippenhahn, Weigert & Weiss (2012), §11.2 (Henyey method, correction limiting)
    """
    R_THRESHOLD = 1e-1
    max_corr_zone = 1.0
    max_corr_global = 2.0
    max_struct_per_zone = jnp.max(jnp.abs(dy[:, :3]), axis=1)
    max_struct_global = jnp.max(max_struct_per_zone)
    alpha_per_zone = jnp.minimum(1.0, max_corr_zone / (max_struct_per_zone + 1e-30))
    alpha_global = jnp.minimum(1.0, max_corr_global / (max_struct_global + 1e-30))
    alpha_hybrid = jnp.maximum(alpha_per_zone, alpha_global)
    near_convergence = R_norm < R_THRESHOLD if R_norm is not None else False
    alpha = jnp.where(near_convergence, alpha_hybrid, alpha_global)
    return dy * alpha[:, None]
