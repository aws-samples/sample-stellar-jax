"""Shared interpolation utilities for the calibration module.

Extracted from evolution.py where the 3-point Lagrange interpolation was
copy-pasted 4 times (solar_residual, solar_residual_with_zprofile,
solar_residual_henyey, compare_model_s).

Reference: Press et al. (2007), Numerical Recipes §3.1 — polynomial interpolation.
"""


def lagrange_interp_3pt(t0, t1, t2, y0, y1, y2, t_target):
    """Quadratic (3-point) Lagrange interpolation.

    Given three knots (t0, y0), (t1, y1), (t2, y2), interpolate the value
    at t_target using the unique quadratic polynomial through the points.

    This reduces truncation error from O(dt²) to O(dt³), making interpolated
    residuals insensitive to adaptive-step changes (issue #155/F14).

    Parameters
    ----------
    t0, t1, t2 : scalar (JAX-compatible)
        The three knot positions (need not be equidistant).
    y0, y1, y2 : scalar (JAX-compatible)
        Values at the knots.
    t_target : scalar (JAX-compatible)
        Position at which to evaluate the interpolant.

    Returns
    -------
    scalar
        Interpolated value at t_target. Fully differentiable.
    """
    d01 = t0 - t1
    d02 = t0 - t2
    d12 = t1 - t2
    # Lagrange basis weights (1e-30 prevents division by zero for degenerate spacing)
    w0 = (t_target - t1) * (t_target - t2) / (d01 * d02 + 1e-30)
    w1 = (t_target - t0) * (t_target - t2) / (-d01 * d12 + 1e-30)
    w2 = (t_target - t0) * (t_target - t1) / (d02 * d12 + 1e-30)
    return w0 * y0 + w1 * y1 + w2 * y2
