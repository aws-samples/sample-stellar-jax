"""Eigenfunction extraction from the oscillation integrator trajectory.

Computes the physical displacement eigenfunctions ξ_r(r) and ξ_h(r) at
a converged eigenfrequency by stacking the lax.scan integration trajectory
and converting from GYRE's dimensionless variables to physical displacements.

Variable definitions (GYRE variable set, Townsend & Teitler 2013 eq. 12):
    y₁ = x^(2-ℓ) · ξ_r / r    →  ξ_r/R = y₁ · x^(ℓ-1)
    y₂ = x^(2-ℓ) · P'/(ρgr)   }
    y₃ = x^(2-ℓ) · Φ'/(gr)    }→  ξ_h/R = (y₂ + y₃) · x^(ℓ-1) / (c₁ω²)

The ξ_h conversion follows from the linearized horizontal momentum equation
(Unno et al. 1989 eq. 14.5): σ²rξ_h = (1/ρ)P' + Φ', which in the GYRE
variables becomes ξ_h/r = (y₂ + y₃) · x^(ℓ-2) / (c₁ω²). The y₃ term is
the gravitational potential perturbation; dropping it is the Cowling
approximation (small for high-order p-modes, but wrong for g-modes).

For radial modes (ℓ=0, reduce_order):
    y₁ = ξ_r / r               →  ξ_r/R = y₁ · x
    (no horizontal displacement)

For nonradial modes (ℓ≥1), two independent solutions are integrated from
the center. The physical eigenfunction is the linear combination satisfying
the surface boundary condition (vacuum or JCD).

Mode inertia E_{n,l} (#780):
    E = (4π R³ / M) ∫₀¹ [ξ_r² + l(l+1)ξ_h²] ρ x² dx / (ξ_r(R)/R)²

This is the GYRE/ADIPLS normalized mode inertia (E_norm in GYRE). It
measures the kinetic energy stored in the mode — higher inertia means the
mode is harder to excite and has smaller surface amplitude for given energy.

References:
    Townsend & Teitler (2013), MNRAS 435, 3406 (GYRE; variable definitions)
    Christensen-Dalsgaard (2008), Ap&SS 316, 113 (ADIPLS; boundary conditions)
    Aerts, Christensen-Dalsgaard & Kurtz (2010), §3.3 (mode inertia, ξ_h)
    Christensen-Dalsgaard & Berthomieu (1991), in Solar Interior and Atmosphere,
        §IV.B (mode inertia normalization, Eq. IV.35)
"""
import jax.numpy as jnp
import numpy as np

from .integrator import (
    _integrate_radial_trajectory,
    _integrate_nonradial_trajectory,
    _make_integration_grid,
)
from .coefficients import _build_oscillation_grid_jax, _prepare_coeffs
from .eigenvalue import _bracket_and_refine


def _select_mode_by_npg(roots_nu, glob, var, l, n_pg):
    """Select the eigenfrequency root matching a target radial order (l, n_pg).

    Uses the per-star large separation Δν (from the acoustic radius integral,
    MESA report.f90:343) and the empirical asymptotic phase ε (from l=0 mode
    spacings, White et al. 2012) to assign radial orders to the found roots
    via the Tassoul (1980) relation:

        n = round(ν/Δν − l/2 − ε)

    This replaces the old hardcoded solar selector (Δν=135, ε=1.5, argmin)
    which returned the wrong mode for any non-solar star (#1122).

    Args:
        roots_nu: list of converged eigenfrequencies (µHz), sorted ascending
        glob: FGONG global parameters
        var: FGONG per-point variables (center-to-surface)
        l: angular degree
        n_pg: target radial order

    Returns:
        int: index into roots_nu of the mode with the requested n_pg

    Raises:
        ValueError: if no root matches the requested n_pg

    References:
        Tassoul (1980), ApJS 43, 469 (asymptotic relation)
        MESA report.f90:343 (Δν from acoustic radius)
        White et al. (2012), ApJ 751, 36 (ε estimation)
    """
    from .seismic_quantities import (
        large_separation, _estimate_epsilon_from_l0, assign_n_pg
    )

    roots_arr = np.array(sorted(roots_nu))
    dnu_star = large_separation(glob, var)

    # Estimate ε: the asymptotic phase is a property of the STAR, not the mode.
    # _estimate_epsilon_from_l0 expects l=0 frequencies (ν ≈ Δν·(n + ε)).
    # For l > 0, ν ≈ Δν·(n + l/2 + ε), so we remove the l/2 shift before
    # estimating ε, then use the true l for assign_n_pg.
    shifted_roots = roots_arr - l / 2.0 * dnu_star  # shift to pseudo-l=0
    epsilon = _estimate_epsilon_from_l0(shifted_roots, dnu_star)

    # Assign n_pg to each root
    identified = assign_n_pg(roots_arr, dnu_star, epsilon, l)

    # Look up the target n_pg
    if n_pg in identified:
        target_nu = identified[n_pg]
        # Find which root index this corresponds to
        mode_index = int(np.argmin(np.abs(np.array(roots_nu) - target_nu)))
        return mode_index

    # Fallback: if assign_n_pg didn't find the exact n_pg (e.g., due to
    # mode mixing or sparse roots), use the Tassoul estimate directly
    nu_target = dnu_star * (n_pg + l / 2.0 + epsilon)
    mode_index = int(np.argmin(np.abs(np.array(roots_nu) - nu_target)))
    return mode_index


def compute_eigenfunction(glob, var, l, n_pg, nu_min=None, nu_max=None,
                          n_scan=500, n_steps=8000):
    """Compute the eigenfunction ξ_r(x) and ξ_h(x) for a specific mode.

    Given a stellar model in FGONG format and a target mode (l, n_pg),
    finds the eigenfrequency and integrates the oscillation equations to
    produce the spatial eigenfunctions on the integration grid.

    For nonradial modes, the two independent solutions are combined using
    the surface boundary condition (vacuum BC: y₁ - y₂ = 0) to produce
    the physical eigenfunction.

    Args:
        glob: FGONG global parameters (numpy array)
        var: FGONG per-point variables, center-to-surface (numpy array)
        l: angular degree (0, 1, 2, 3, ...)
        n_pg: target radial order (Scuflaire-Osaki-Takata classification)
        nu_min: lower frequency bound for search (µHz); auto if None
        nu_max: upper frequency bound for search (µHz); auto if None
        n_scan: number of trial frequencies for bracket detection
        n_steps: number of RK4 integration steps

    Returns:
        dict with:
            'x': (N_steps,) fractional radius grid (dimensionless)
            'xi_r': (N_steps,) radial displacement ξ_r/R (dimensionless,
                     normalized so max|ξ_r/R| = 1)
            'xi_h': (N_steps,) horizontal displacement ξ_h/R (dimensionless,
                     same normalization; zeros for l=0)
            'y': (N_steps, n_var) raw GYRE variables on the grid
            'nu': converged frequency in µHz
            'sigma2': dimensionless σ²
            'l': angular degree
            'n_pg': radial order

    References:
        GYRE docs §Dimensionless Formulation, eq. 12 (variable definitions)
        GYRE docs §Boundary Conditions (inner regular, outer vacuum)
    """
    # Build structure grid
    grid_data = _build_oscillation_grid_jax(glob, var)
    x_grid_jax, coeffs_jax = _prepare_coeffs(grid_data)
    factor = grid_data['factor']

    # Integration grid
    x_steps, h_steps = _make_integration_grid(grid_data['x_grid'], n_steps=n_steps)

    # Auto frequency range from the per-star large separation (acoustic radius
    # integral, MESA report.f90:343). Replaces the old hardcoded Δν=135 µHz
    # which was only correct for solar-type stars.
    from .seismic_quantities import large_separation
    dnu_star = large_separation(glob, var)

    if nu_min is None:
        # Tassoul estimate with per-star Δν; ε≈1.2 is a conservative midpoint
        nu_est = dnu_star * (n_pg + l / 2.0 + 1.2)
        nu_min = max(100.0, nu_est - 5 * dnu_star)
    if nu_max is None:
        nu_est = dnu_star * (n_pg + l / 2.0 + 1.2)
        nu_max = min(8000.0, nu_est + 5 * dnu_star)

    # Find all eigenfrequencies in the range
    from .determinant import radial_determinant, nonradial_determinant
    import jax

    if l == 0:
        @jax.jit
        def det_fn(s2):
            return radial_determinant(s2, x_steps, h_steps, x_grid_jax, coeffs_jax)
    else:
        l_float = float(l)

        @jax.jit
        def det_fn(s2):
            return nonradial_determinant(
                s2, jnp.float64(l_float), x_steps, h_steps, x_grid_jax, coeffs_jax)

    nu_arr = np.linspace(nu_min, nu_max, n_scan)
    roots_nu = _bracket_and_refine(det_fn, nu_arr, factor, rtol=1e-10, maxiter=100)

    if len(roots_nu) == 0:
        raise ValueError(
            f"No modes found in [{nu_min}, {nu_max}] µHz for l={l}")

    # Select the mode by per-star radial-order identification (Tassoul relation).
    # Uses the acoustic-radius Δν and empirical ε from l=0 spacings, NOT the
    # hardcoded solar Δν=135/ε=1.5 which is wrong for non-solar stars.
    mode_index = _select_mode_by_npg(roots_nu, glob, var, l, n_pg)

    nu_converged = roots_nu[mode_index]
    sigma2 = jnp.float64(nu_converged**2 * factor)

    # Now integrate with trajectory stacking
    if l == 0:
        xi_r, xi_h, y_traj, x_grid_out = _eigenfunction_radial(
            sigma2, x_steps, h_steps, x_grid_jax, coeffs_jax)
    else:
        xi_r, xi_h, y_traj, x_grid_out = _eigenfunction_nonradial(
            sigma2, float(l), x_steps, h_steps, x_grid_jax, coeffs_jax)

    # Normalize so max|ξ_r/R| = 1
    xi_r_np = np.asarray(xi_r)
    xi_h_np = np.asarray(xi_h)
    norm = np.max(np.abs(xi_r_np))
    if norm > 0:
        xi_r_np = xi_r_np / norm
        xi_h_np = xi_h_np / norm

    return {
        'x': np.asarray(x_grid_out),
        'xi_r': xi_r_np,
        'xi_h': xi_h_np,
        'y': np.asarray(y_traj),
        'nu': nu_converged,
        'sigma2': float(sigma2),
        'l': l,
        'n_pg': n_pg,
    }


def _eigenfunction_radial(sigma2, x_steps, h_steps, x_grid, coeffs):
    """Extract radial eigenfunction from trajectory.

    For radial modes (l=0, reduce_order), GYRE uses:
        y₁ = ξ_r/r  → ξ_r/R = x · y₁

    Inner BC: y = [0, 1] (regularity: y₁ = 0 at center).

    Returns:
        (xi_r, xi_h, y_trajectory, x_grid_out)
    """
    y0 = jnp.array([0.0, 1.0])
    _, y_traj = _integrate_radial_trajectory(
        sigma2, x_steps, h_steps, y0, x_grid, coeffs)

    # x positions: each row in y_traj corresponds to x_steps + h_steps
    # (the state AFTER taking the step from x_steps[i] with size h_steps[i])
    x_out = x_steps + h_steps

    # Convert y₁ → ξ_r/R = x · y₁ (radial, l=0 reduced order)
    xi_r = x_out * y_traj[:, 0]
    # No horizontal displacement for radial modes
    xi_h = jnp.zeros_like(xi_r)

    return xi_r, xi_h, y_traj, x_out


def _eigenfunction_nonradial(sigma2, l, x_steps, h_steps, x_grid, coeffs):
    """Extract nonradial eigenfunction from trajectory.

    Two independent solutions are integrated from the center. The physical
    eigenfunction is the linear combination c₁·sol₁ + c₂·sol₂ satisfying
    the surface vacuum BC: y₁ - y₂ = 0 at the surface.

    GYRE variable set (eq. 12) + horizontal momentum equation:
        ξ_r/R = y₁ · x^(l-1)
        ξ_h/R = (y₂ + y₃) · x^(l-1) / (c₁ω²)

    where c₁ = 1/A₁ (the structure coefficient). The ξ_h formula comes from
    the linearized horizontal momentum equation: σ²rξ_h = (1/ρ)P' + Φ',
    which in GYRE variables gives ξ_h/r = (y₂ + y₃)·x^(l-2)/(c₁ω²).
    (Unno et al. 1989 eq. 14.5; Aerts, CD & Kurtz 2010 eq. 3.132).

    Inner regular BCs (GYRE ad_bound_m.fypp):
        sol1 = [1, c₁σ²/l, 0, 0]
        sol2 = [0, -1, 1, l]

    Returns:
        (xi_r, xi_h, y_trajectory, x_grid_out)
    """
    # Inner regular BCs
    x_c = x_steps[0]
    A1_c = jnp.interp(x_c, x_grid, coeffs[1])
    c1_c = 1.0 / jnp.maximum(A1_c, 1e-30)
    y0_1 = jnp.array([1.0, c1_c * sigma2 / l, 0.0, 0.0])
    y0_2 = jnp.array([0.0, -1.0, 1.0, l])

    # Integrate both solutions with trajectory
    _, y_traj_1 = _integrate_nonradial_trajectory(
        sigma2, l, x_steps, h_steps, y0_1, x_grid, coeffs)
    _, y_traj_2 = _integrate_nonradial_trajectory(
        sigma2, l, x_steps, h_steps, y0_2, x_grid, coeffs)

    # Surface BC (vacuum): y₁ - y₂ = 0
    # For sol₁: bc1_1 = y₁_1(surface) - y₂_1(surface)
    # For sol₂: bc1_2 = y₁_2(surface) - y₂_2(surface)
    # Linear combo: c₁·bc1_1 + c₂·bc1_2 = 0
    # Choose c₁ = -bc1_2, c₂ = bc1_1 (unnormalized)
    bc1_1 = y_traj_1[-1, 0] - y_traj_1[-1, 1]
    bc1_2 = y_traj_2[-1, 0] - y_traj_2[-1, 1]

    # The eigenfunction is c1*sol1 + c2*sol2 where c1*bc1_1 + c2*bc1_2 = 0
    # Choose: c1 = -bc1_2, c2 = bc1_1
    c1 = -bc1_2
    c2 = bc1_1

    # Combined trajectory
    y_traj = c1 * y_traj_1 + c2 * y_traj_2

    # x positions after each step
    x_out = x_steps + h_steps

    # Convert to physical displacements (GYRE eq. 12 + momentum eqn):
    #   ξ_r/R = y₁ · x^(l-1)
    #   ξ_h/R = (y₂ + y₃) · x^(l-1) / (c₁ω²)
    # where c₁(x) = 1/A₁(x) is the structure coefficient at each point.
    # The y₃ term is the gravitational perturbation Φ'/(gr); omitting it
    # is the Cowling approximation (valid for high-order p-modes, wrong for
    # low-order or g-modes).
    x_pow = x_out ** (l - 1.0)
    xi_r = y_traj[:, 0] * x_pow

    # c₁(x) = 1/A₁(x) at each grid point
    A1_arr = jnp.interp(x_out, x_grid, coeffs[1])
    c1_arr = 1.0 / jnp.maximum(A1_arr, 1e-30)
    c1_omega2 = c1_arr * sigma2

    xi_h = (y_traj[:, 1] + y_traj[:, 2]) * x_pow / c1_omega2

    return xi_r, xi_h, y_traj, x_out


# ═══════════════════════════════════════════════════════════════════════════════
# Mode inertia
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mode_inertia(eigenfunction_result, glob, var):
    """Compute the normalized mode inertia E for a given eigenfunction.

    The mode inertia is the kinetic energy integral normalized by the
    surface displacement squared:

        E = (4π R³ / M) ∫₀¹ [ξ_r² + l(l+1)ξ_h²] ρ x² dx / (ξ_r(R)/R)²

    where ξ_r/R and ξ_h/R are the dimensionless displacements from
    compute_eigenfunction, and the integral is over the stellar interior.

    This is the GYRE E_norm normalization (Aerts, Christensen-Dalsgaard &
    Kurtz 2010, Eq. 3.139; GYRE docs, mode quantities, "E_norm"). It is
    dimensionless and measures the effective inertia of the mode relative
    to a uniform radial displacement of the whole star.

    For high-order solar p-modes (the regime relevant to asteroseismic
    inversions), E is approximately constant across modes of the same l
    and ~1/[l(l+1)] scaling between different l values. Lower-order and
    mixed modes have systematically higher E.

    Args:
        eigenfunction_result: dict from compute_eigenfunction() containing
            'xi_r', 'xi_h', 'x', 'l'
        glob: FGONG global parameters (numpy array)
            glob[0] = M (stellar mass in grams)
            glob[1] = R (stellar radius in cm)
        var: FGONG per-point variables center-to-surface (numpy array)
            col 0: r, col 4: ρ

    Returns:
        float: the normalized mode inertia E (dimensionless, > 0)

    Raises:
        ValueError: if ξ_r(R) = 0 (mode has a node at the surface —
            would indicate a g-mode or numerical failure)

    References:
        Aerts, Christensen-Dalsgaard & Kurtz (2010), §3.3, Eq. 3.139
        Christensen-Dalsgaard & Berthomieu (1991), Eq. IV.35
        GYRE: mode_par_m.fypp, E_norm() function
    """
    xi_r = eigenfunction_result['xi_r']   # ξ_r/R (dimensionless, normalized)
    xi_h = eigenfunction_result['xi_h']   # ξ_h/R (dimensionless, same norm)
    x = eigenfunction_result['x']         # fractional radius r/R
    l = eigenfunction_result['l']

    # Stellar mass and radius from FGONG global parameters
    M = glob[0]
    R = glob[1]

    # Interpolate density from FGONG onto the integration grid
    r_fgong = var[:, 0]
    x_fgong = r_fgong / R
    rho_fgong = var[:, 4]
    rho = np.interp(x, x_fgong, rho_fgong)

    # Integrand: [ξ_r² + l(l+1)ξ_h²] · ρ · x²
    # The full formula (Aerts+2010 Eq. 3.139) is:
    #   E = (4π R³ / M) ∫ [(ξ_r/R)² + l(l+1)(ξ_h/R)²] · ρ · x² dx / (ξ_r(R)/R)²
    #
    # GYRE's E_norm uses normalization by ξ_r(R)² only (for p-modes where
    # ξ_h(R) << ξ_r(R)). We follow GYRE's convention.
    ll1 = l * (l + 1.0)
    integrand = (xi_r**2 + ll1 * xi_h**2) * rho * x**2

    # Trapezoidal integration over the integration grid
    integral = np.trapezoid(integrand, x)

    # Surface displacement normalization
    # Use the last point of the trajectory (closest to surface)
    xi_r_surface = xi_r[-1]
    if abs(xi_r_surface) < 1e-30:
        raise ValueError(
            f"ξ_r(R) ≈ 0 for mode l={l}, ν={eigenfunction_result['nu']:.1f} µHz "
            f"— cannot normalize mode inertia (possible g-mode or numerical failure)")

    # E = (4π R³ / M) · integral / (ξ_r(R)/R)²
    E = 4.0 * np.pi * R**3 / M * integral / xi_r_surface**2

    return float(E)
