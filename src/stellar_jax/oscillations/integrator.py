"""Fixed-step RK4 integrator via lax.scan for adiabatic oscillation equations.

Implements the radial (l=0) and nonradial (l≥1) adiabatic oscillation equations in
GYRE's 'GYRE' variable set (Townsend & Teitler 2013), using classical RK4 on a fixed
integration grid. The grid follows the FGONG point distribution (concentrating points
near the center) subdivided to the target step count.

NOTE: these were ported from the ADIPLS A-formulation (CD2008 Eqs. 11-14, 17-18)
to GYRE's formulation. In the A-formulation the Brunt term A entered frequency-amplified
(−ηA, η = l(l+1)/(c₁σ²)), which is numerically explosive for sharp evolved-core
composition gradients and biased δν₀₂ 5–40% low; GYRE carries the Brunt term un-amplified.

The RK4 integration kernels are kept whole (single-responsibility cohesive units):
  _rk4_step: generic RK4 stepper
  _radial_rhs / _nonradial_rhs: physics equations (GYRE ad_eqns_m/rad_eqns_m)
  _integrate_radial / _integrate_nonradial: lax.scan drivers
  _make_integration_grid: grid construction

References:
    Townsend & Teitler (2013), MNRAS 435, 3406 (GYRE; src/ad/ad_eqns_m.fypp, rad_eqns_m)
    Christensen-Dalsgaard (2008), Ap&SS 316, 113 (ADIPLS; prior formulation, superseded)
"""
import jax.numpy as jnp
from jax import lax
import numpy as np


# ─── Coefficient interpolation ─────────────────────────────────────────────────

def _interp_coeffs(x_val, x_grid, coeffs):
    """Interpolate structure coefficients at a single point.

    Uses jnp.interp (linear interpolation), matching np.interp in the NumPy solver.

    Args:
        x_val: scalar x position
        x_grid: (N,) fractional radius grid
        coeffs: (5, N) coefficient stack [Vg, A1, A_bv, U, q]

    Returns:
        Vg, A1, A_bv, U, q at x_val
    """
    Vg = jnp.interp(x_val, x_grid, coeffs[0])
    A1 = jnp.interp(x_val, x_grid, coeffs[1])
    A_bv = jnp.interp(x_val, x_grid, coeffs[2])
    U = jnp.interp(x_val, x_grid, coeffs[3])
    q = jnp.interp(x_val, x_grid, coeffs[4])
    return Vg, A1, A_bv, U, q


# ─── RHS functions (physics equations) ─────────────────────────────────────────

def _radial_rhs(x_val, y, sigma2, x_grid, coeffs):
    """RHS of radial (l=0) oscillation equations — GYRE formulation.

    GYRE's 'GYRE' variable set (src/rad/rad_eqns_m.fypp), consistent with the
    nonradial GYRE formulation in _nonradial_rhs. Replaces the
    ADIPLS radial form (Eqs. 17-18) so that l=0 and l=2 share one formulation
    and surface treatment — required for δν₀₂ to match GYRE (the ADIPLS/GYRE
    surface offset otherwise fails to cancel between l=0 and l=2).

    Coefficients: Vg = V/Γ₁, As = Brunt A (= A_bv), U, c₁ = 1/A1.
    """
    Vg, A1, A_bv, U, _ = _interp_coeffs(x_val, x_grid, coeffs)
    As = A_bv
    c1w2 = sigma2 / jnp.maximum(A1, 1e-30)  # c₁σ²
    y1, y2 = y[0], y[1]

    # GYRE xA matrix (x·y' = xA·y); A = xA/x
    dy1 = ((Vg - 1.0) * y1
           - Vg * y2) / x_val
    dy2 = ((c1w2 + U - As) * y1
           + (As - U + 3.0) * y2) / x_val
    return jnp.array([dy1, dy2])


def _nonradial_rhs(x_val, y, sigma2, l, x_grid, coeffs):
    """RHS of nonradial (l≥1) oscillation equations — GYRE formulation.

    4th-order system with gravitational perturbation Φ', using GYRE's
    'GYRE' variable set (src/ad/ad_eqns_m.fypp). This REPLACES the ADIPLS
    A-formulation (Eqs. 11-14), in which the Brunt term A entered multiplied
    by η = l(l+1)/(c₁σ²) — a frequency-amplified factor that made A's
    composition-gradient sharpness numerically explosive in evolved cores,
    biasing δν₀₂ low by 5–40%. In GYRE's formulation the Brunt
    term (As) enters un-amplified (coefficient ~1), fixing the bias
    (validated vs GYRE across the MS X_c series to <0.1 µHz at low order).

    Variables (GYRE set): y = (ξr/r, ξh-related, Φ'/(gr)-related, dΦ'/dr).
    Coefficients: Vg = V/Γ₁, As = Brunt A (= A_bv), U, c₁ = 1/A1, λ = l(l+1).
    Reference: Townsend & Teitler (2013) MNRAS 435, 3406 (GYRE); matrix from
    ad_eqns_m.fypp with variables_set='GYRE' (identity transform).
    """
    Vg, A1, A_bv, U, _ = _interp_coeffs(x_val, x_grid, coeffs)
    ll1 = l * (l + 1.0)
    As = A_bv
    lam_c1w2 = ll1 * A1 / sigma2          # λ/(c₁σ²)  (since c₁ = 1/A1)
    c1w2 = sigma2 / jnp.maximum(A1, 1e-30)  # c₁σ²

    y1, y2, y3, y4 = y[0], y[1], y[2], y[3]

    # GYRE xA matrix (x·y' = xA·y); A = xA/x
    dy1 = ((Vg - 1.0 - l) * y1
           + (lam_c1w2 - Vg) * y2
           + lam_c1w2 * y3) / x_val
    dy2 = ((c1w2 - As) * y1
           + (As - U + 3.0 - l) * y2
           - y4) / x_val
    dy3 = ((3.0 - U - l) * y3
           + y4) / x_val
    dy4 = (U * As * y1
           + U * Vg * y2
           + ll1 * y3
           + (2.0 - U - l) * y4) / x_val
    return jnp.array([dy1, dy2, dy3, dy4])


# ─── RK4 step ─────────────────────────────────────────────────────────────────

def _rk4_step(rhs_fn, x, y, h, *args):
    """Single classical RK4 step: y(x+h) from y(x)."""
    k1 = rhs_fn(x, y, *args)
    k2 = rhs_fn(x + 0.5 * h, y + 0.5 * h * k1, *args)
    k3 = rhs_fn(x + 0.5 * h, y + 0.5 * h * k2, *args)
    k4 = rhs_fn(x + h, y + h * k3, *args)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ─── lax.scan integration drivers ─────────────────────────────────────────────

def _integrate_radial(sigma2, x_steps, h_steps, y0, x_grid, coeffs):
    """Integrate the radial system center→surface via lax.scan RK4.

    Args:
        sigma2: dimensionless frequency squared
        x_steps: array of x starting points for each step
        h_steps: array of step sizes
        y0: initial conditions [y1, y2] at x_steps[0]
        x_grid, coeffs: structure coefficient data

    Returns:
        Final state [y1, y2] at surface
    """
    def scan_body(y, step_data):
        x_i, h_i = step_data
        y_new = _rk4_step(_radial_rhs, x_i, y, h_i, sigma2, x_grid, coeffs)
        return y_new, None

    y_final, _ = lax.scan(scan_body, y0, (x_steps, h_steps))
    return y_final


def _integrate_radial_trajectory(sigma2, x_steps, h_steps, y0, x_grid, coeffs):
    """Integrate the radial system center→surface, stacking the full trajectory.

    Same physics as _integrate_radial but stores y(x) at every grid point
    for eigenfunction extraction. Separated to avoid the memory cost of
    stacking 16k×2 floats during the frequency scan (hundreds of evaluations).

    Args:
        sigma2: dimensionless frequency squared
        x_steps: array of x starting points for each step
        h_steps: array of step sizes
        y0: initial conditions [y1, y2] at x_steps[0]
        x_grid, coeffs: structure coefficient data

    Returns:
        (y_final, y_trajectory): final state and shape (N_steps, 2) trajectory
    """
    def scan_body(y, step_data):
        x_i, h_i = step_data
        y_new = _rk4_step(_radial_rhs, x_i, y, h_i, sigma2, x_grid, coeffs)
        return y_new, y_new

    y_final, y_traj = lax.scan(scan_body, y0, (x_steps, h_steps))
    return y_final, y_traj


def _integrate_nonradial(sigma2, l, x_steps, h_steps, y0, x_grid, coeffs):
    """Integrate the nonradial system center→surface via lax.scan RK4.

    Args:
        sigma2: dimensionless frequency squared
        l: angular degree (float)
        x_steps: array of x starting points for each step
        h_steps: array of step sizes
        y0: initial conditions [y1, y2, y3, y4] at x_steps[0]
        x_grid, coeffs: structure coefficient data

    Returns:
        Final state [y1, y2, y3, y4] at surface
    """
    def scan_body(y, step_data):
        x_i, h_i = step_data
        y_new = _rk4_step(_nonradial_rhs, x_i, y, h_i, sigma2, l, x_grid, coeffs)
        return y_new, None

    y_final, _ = lax.scan(scan_body, y0, (x_steps, h_steps))
    return y_final


def _integrate_nonradial_trajectory(sigma2, l, x_steps, h_steps, y0,
                                    x_grid, coeffs):
    """Integrate the nonradial system center→surface, stacking the full trajectory.

    Same physics as _integrate_nonradial but stores y(x) at every grid point
    for eigenfunction extraction. Separated to avoid the memory cost of
    stacking 16k×4 floats during the frequency scan.

    Args:
        sigma2: dimensionless frequency squared
        l: angular degree (float)
        x_steps: array of x starting points for each step
        h_steps: array of step sizes
        y0: initial conditions [y1, y2, y3, y4] at x_steps[0]
        x_grid, coeffs: structure coefficient data

    Returns:
        (y_final, y_trajectory): final state and shape (N_steps, 4) trajectory
    """
    def scan_body(y, step_data):
        x_i, h_i = step_data
        y_new = _rk4_step(_nonradial_rhs, x_i, y, h_i, sigma2, l, x_grid, coeffs)
        return y_new, y_new

    y_final, y_traj = lax.scan(scan_body, y0, (x_steps, h_steps))
    return y_final, y_traj


# ─── Integration grid construction ────────────────────────────────────────────

def _make_integration_grid(x_grid_np, n_steps=16000):
    """Create a fixed integration grid from the FGONG x-grid.

    Uses the FGONG grid points themselves as the integration grid, subdivided
    to achieve n_steps total steps. This naturally concentrates points near the
    center where the FGONG has denser spacing (matching the adaptive solver's
    behavior near the 1/x singularity).

    The approach: subdivide each FGONG interval into sub-steps proportional to
    interval size, ensuring at least 1 sub-step per interval, then distribute
    remaining steps proportional to interval width.

    Args:
        x_grid_np: numpy array of FGONG x-grid points
        n_steps: target number of RK4 steps

    Returns:
        (x_steps, h_steps): JAX arrays for lax.scan integration
    """
    if len(x_grid_np) == 0:
        raise ValueError(
            "Empty integration grid: no FGONG points with x > 1e-4. This "
            "typically means R_star (from log_L/log_Te) is inconsistent with "
            "the Henyey radial grid (y_henyey[:, 0]): R_star >> max(r), so "
            "all x = r/R < 1e-4. Check that the evolve_star result's "
            "log_L_final / log_Teff_final are derived from the same y_henyey "
            "as y_henyey_final."
        )
    x_end = float(x_grid_np[-1])
    intervals = np.diff(x_grid_np)

    # Allocate sub-steps: proportional to interval width, minimum 1 per interval
    n_intervals = len(intervals)
    remaining = n_steps - n_intervals  # after giving 1 to each

    if remaining > 0:
        # Distribute proportional to interval width
        weights = intervals / intervals.sum()
        extra = (weights * remaining).astype(int)
        # Fix rounding
        shortfall = remaining - extra.sum()
        if shortfall > 0:
            biggest = np.argsort(intervals)[-shortfall:]
            extra[biggest] += 1
        n_sub = 1 + extra
    else:
        n_sub = np.ones(n_intervals, dtype=int)

    # Build the full grid
    x_points = []
    for i in range(n_intervals):
        sub_x = np.linspace(x_grid_np[i], x_grid_np[i + 1], n_sub[i] + 1)
        x_points.append(sub_x[:-1])
    x_all = np.concatenate(x_points)

    h_all = np.diff(np.append(x_all, x_end))  # h for each step
    return jnp.array(x_all), jnp.array(h_all)
