"""Structure → oscillation coefficients mapping.

Extracts dimensionless structure coefficients [Vg, A1, A_bv, U, q] from FGONG
arrays. These are the DIFFERENTIABLE inputs to the pulsation eigensolver.

Two paths:
  - _build_oscillation_grid_jax: NumPy FGONG → NumPy extraction → JAX arrays
    (for compute_eigenfreq_differentiable with a file-loaded FGONG)
  - build_oscillation_coeffs_jax: live JAX arrays → JAX coefficients
    (for the end-to-end differentiable chain: evolve_star → FGONG → eigenfreq)

Shared helper:
  - _extract_fgong_numpy: one-source FGONG column extraction + BV frequency
    (NumPy). Used by both _build_oscillation_grid_jax (this module) and
    _build_oscillation_grid (diagnostic.py), eliminating the copy-pasted
    BV branch + guards + nan_to_num defaults.

References:
    Christensen-Dalsgaard (2008), Ap&SS 316, 113 (ADIPLS: §2, Eq. 1)
    Aerts, Christensen-Dalsgaard & Kurtz (2010), §3.3 (structure coefficients)
"""
import jax
import jax.numpy as jnp
import numpy as np

from stellar_jax.config.constants import G as _G
from .seismic_conversion import compute_factor as _compute_factor


def _extract_fgong_numpy(glob, var):
    """Extract core FGONG structure arrays in NumPy (shared helper, DRY).

    This is the ONE source for FGONG column extraction + Brunt-Väisälä
    frequency computation from NumPy arrays. Both _build_oscillation_grid_jax
    (this module) and _build_oscillation_grid (diagnostic.py) delegate here.

    Args:
        glob: FGONG global parameters (numpy array)
        var: FGONG per-point variables center-to-surface (numpy array)

    Returns:
        dict with numpy arrays:
            x, r, m, P, rho, gamma1, g, N2, c_s, M, R, G
    """
    M = float(glob[0])
    R = float(glob[1])
    G = _G  # CGS, from config/constants.py

    r = var[:, 0]
    x = r / R
    ln_q = var[:, 1]  # ln(m/M)
    P = var[:, 3]
    rho = var[:, 4]
    gamma1 = var[:, 9]

    m = np.exp(ln_q) * M
    # --- Guard roles in this section (center-point singularities) ---
    # • Relative floor (1e-10 * M): prevents m[0]=0 at the center; the magnitude
    #   scales with the star so the guard is physically meaningful, not arbitrary.
    # • Absolute floor (1.0 cm on r): prevents r=0 in g = GM/r² at the center.
    # • Underflow guards (1e-30 on P, rho): keep log() and sqrt() finite; the
    #   magnitude is immaterial (any tiny positive value works).
    m[0] = max(m[0], 1e-10 * M)

    g = G * m / np.maximum(r, 1.0)**2
    c_s = np.sqrt(gamma1 * P / np.maximum(rho, 1e-30))

    # Brunt-Väisälä: use FGONG A* (column 14) if available.
    # Col-14 includes the full Ledoux composition term when written by our
    # write_fgong (or by MESA's pulse output). Fallback is Schwarzschild-only
    # (acceptable for FGONGs without composition data, e.g. bare Model S).
    ivar = var.shape[1]
    if ivar > 14 and np.any(var[:, 14] != 0):
        A_star = var[:, 14]
        N2 = A_star * g / np.maximum(r, 1.0)
    else:
        lnP = np.log(np.maximum(P, 1e-30))
        lnrho = np.log(np.maximum(rho, 1e-30))
        dlnP_dr = np.gradient(lnP, r)
        dlnrho_dr = np.gradient(lnrho, r)
        N2 = g * (dlnP_dr / gamma1 - dlnrho_dr)

    return dict(x=x, r=r, m=m, P=P, rho=rho, gamma1=gamma1,
                g=g, N2=N2, c_s=c_s, M=M, R=R, G=G)


def _build_oscillation_grid_jax(glob, var):
    """Extract dimensionless structure variables for oscillation computation.

    Identical to oscillations._build_oscillation_grid but returns JAX-ready data.
    Uses NumPy for the initial extraction (FGONG is a NumPy file), then
    the caller converts to JAX for the integration phase.

    Delegates FGONG column extraction + BV frequency to _extract_fgong_numpy
    (the shared DRY helper), then computes ADIPLS structure coefficients.

    Args:
        glob: FGONG global parameters (numpy array)
        var: FGONG per-point variables center-to-surface (numpy array)

    Returns:
        dict with numpy arrays: x_grid, Vg, A1, A_bv, U, q
        plus scalars: M, R, G, factor
    """
    fgong = _extract_fgong_numpy(glob, var)
    M, R, G = fgong['M'], fgong['R'], fgong['G']
    x, r, m, P, rho, gamma1, g, N2 = (
        fgong['x'], fgong['r'], fgong['m'], fgong['P'],
        fgong['rho'], fgong['gamma1'], fgong['g'], fgong['N2'],
    )

    # Structure coefficients (ADIPLS notation):
    # --- Guard roles ---
    # • Underflow guards (1e-30 on P, x³): magnitude immaterial; prevents /0.
    # • Absolute floor (1.0 cm on r): prevents r=0 in Vg, A_bv at the center.
    # • Relative floor (1e-10 * M on m): scales with the star; prevents U→∞
    #   at the center where 4πr³ρ/m is 0/0.
    # • g floor (1e-10): prevents A_bv = N² r/g from blowing up; g→0 only
    #   at the exact center (already excluded by the mask below).
    Vg = G * m * rho / (np.maximum(r, 1.0) * gamma1 * np.maximum(P, 1e-30))
    A1 = (m / M) / np.maximum(x**3, 1e-30)
    A_bv = N2 * np.maximum(r, 1.0) / np.maximum(g, 1e-10)
    U = 4.0 * np.pi * r**3 * rho / np.maximum(m, 1e-10 * M)
    q = m / M

    # Working grid: exclude singular center; use full model extent.
    mask = (x > 1e-4)
    x_grid = x[mask]

    # Sanitize residual NaN/inf from center-adjacent points.
    # Replacement values are physically motivated defaults:
    #   Vg → 0 (vanishes at center), A1 → 1 (m/M ≈ x³ near center),
    #   A_bv → 0 (N²→0 at center), U → 3 (uniform-density sphere: 4πr³ρ/m→3),
    #   q → small positive (avoids log(0) downstream).
    Vg_grid = np.nan_to_num(Vg[mask], nan=0.0, posinf=0.0, neginf=0.0)
    A1_grid = np.nan_to_num(A1[mask], nan=1.0, posinf=1e10, neginf=1.0)
    A_grid = np.nan_to_num(A_bv[mask], nan=0.0, posinf=0.0, neginf=0.0)
    U_grid = np.nan_to_num(U[mask], nan=3.0, posinf=3.0, neginf=3.0)
    q_grid = np.nan_to_num(q[mask], nan=1e-10, posinf=1.0, neginf=1e-10)

    # Dimensionless frequency factor: σ² = (2πν·1e-6)²·R³/(GM)
    # Canonical helper (DRY —).
    factor = _compute_factor(R, M, G)

    # Surface Γ₁ — needed for JCD isothermal atmosphere outer BC
    if len(x_grid) == 0:
        raise ValueError(
            f"Empty oscillation grid: no points with x = r/R > 1e-4. "
            f"R={R:.3e} cm, max(r)={np.max(r):.3e} cm, "
            f"x_max={np.max(x):.6f}. Structure is inconsistent."
        )
    gamma1_s = float(gamma1[mask][-1])

    return {
        'x_grid': x_grid,
        'Vg': Vg_grid,
        'A1': A1_grid,
        'A_bv': A_grid,
        'U': U_grid,
        'q': q_grid,
        'gamma1_s': gamma1_s,
        'M': M, 'R': R, 'G': G,
        'factor': factor,
    }


def _prepare_coeffs(grid_data):
    """Package structure coefficients as a JAX array stack for interpolation.

    Args:
        grid_data: dict from _build_oscillation_grid_jax

    Returns:
        (x_grid_jax, coeffs_jax): JAX arrays ready for the integrator
    """
    x_grid = jnp.array(grid_data['x_grid'])
    coeffs = jnp.array([
        grid_data['Vg'],
        grid_data['A1'],
        grid_data['A_bv'],
        grid_data['U'],
        grid_data['q'],
    ])
    return x_grid, coeffs


def build_oscillation_coeffs_jax(glob_jax, var_jax):
    """Build oscillation structure coefficients from JAX FGONG arrays.

    Same physics as _build_oscillation_grid_jax but operates on live JAX arrays
    (not numpy), preserving the gradient graph from structure → frequencies.

    This is the OSC-3 bridge function: structure computed via evolve_star stays
    as JAX arrays and flows directly into eigenfreq_from_coeffs.

    When the structure grid doesn't extend to the center (x_min > 0.01, typical
    of the Henyey mesh which starts at the innermost mass shell ~3% of R), this
    function prepends synthetic center points with physically correct limiting
    values. This ensures the nonradial (l≥1) mode integrator can apply its center
    regularity BCs at x→0, matching what MESA's FGONG output provides naturally.

    Center limits (Kippenhahn, Weigert & Weiss 2012, §10.1; Christensen-Dalsgaard
    2008, Ap&SS 316, 113, §2.3; Unno et al. 1989, §14.1):
      - Vg → 0  (Gmρ/rΓ₁P → 0 as r→0 since m~r³)
      - A1 → ρ_c/ρ̄  (~1 for homogeneous, ~constant near center)
      - A_bv → 0  (N²r/g → 0; convective core or smooth gradient)
      - U → 3  (4πr³ρ/m → 3 when ρ=ρ_c and m=(4/3)πr³ρ_c)
      - q → 0  (m/M → 0)

    CONSTRAINT (JAX/differentiability): the prepended center points use
    stop_gradient values since there is no physical structure data below the
    innermost Henyey shell — gradients flow through the actual mesh points only.
    The deviation from MESA's approach (which has real center data in its FGONG)
    is bounded: center coefficients affect only the first few integration steps
    where the solution is dominated by the analytic regularity BC, not by the
    ODE's RHS (ADIPLS §2.3: the leading-order solution ~x^l is set by the BC,
    not by the numerical integration near center).

    Args:
        glob_jax: (15,) JAX array — FGONG global params [M, R, ...]
        var_jax: (N, ≥15) JAX array — per-point structure center-to-surface
            col 0: r, col 1: ln(m/M), col 3: P, col 4: rho, col 9: Gamma1,
            col 14: A* (Brunt-Väisälä discriminant)

    Returns:
        dict with:
            'x_grid': (N_out,) JAX array — fractional radius grid (with center)
            'coeffs': (5, N_out) JAX array — [Vg, A1, A_bv, U, q]
            'factor': float — ν² → σ² conversion factor
            'M': float, 'R': float, 'G': float

    References:
        Christensen-Dalsgaard (2008), Ap&SS 316, 113 (ADIPLS: §2, Eq. 1)
        Aerts, Christensen-Dalsgaard & Kurtz (2010), §3.3 (structure coefficients)
        Unno et al. (1989), "Nonradial oscillations of stars", §14.1 (center BCs)
    """
    G = _G  # CGS

    M = glob_jax[0]
    R = glob_jax[1]

    r = var_jax[:, 0]
    ln_q = var_jax[:, 1]  # ln(m/M)
    P = var_jax[:, 3]
    rho = var_jax[:, 4]
    gamma1 = var_jax[:, 9]

    x = r / R
    m = jnp.exp(ln_q) * M
    # Protect center: m must be > 0
    m = jnp.maximum(m, 1e-10 * M)

    g = G * m / jnp.maximum(r, 1.0)**2

    # Brunt-Väisälä: use FGONG A* (column 14) if available
    # Col-14 includes the full Ledoux composition term when written by our
    # write_fgong (or by MESA's pulse output). Fallback is Schwarzschild-only
    # (acceptable for FGONGs without composition data, e.g. bare Model S).
    A_star = var_jax[:, 14]
    has_A_star = jnp.any(A_star != 0.0)

    # If A* column is populated, use it; otherwise compute from structure
    N2_from_A = A_star * g / jnp.maximum(r, 1.0)

    lnP = jnp.log(jnp.maximum(P, 1e-30))
    lnrho = jnp.log(jnp.maximum(rho, 1e-30))
    dlnP_dr = jnp.gradient(lnP, r)
    dlnrho_dr = jnp.gradient(lnrho, r)
    N2_computed = g * (dlnP_dr / gamma1 - dlnrho_dr)

    # Use A* column when available (it's more accurate — includes composition gradient)
    N2 = jnp.where(has_A_star, N2_from_A, N2_computed)

    # Structure coefficients (ADIPLS notation, Eq. 1):
    # Vg = Gmρ/(rΓ₁P)
    Vg = G * m * rho / (jnp.maximum(r, 1.0) * gamma1 * jnp.maximum(P, 1e-30))
    # A1 = (m/M) / x³
    A1 = (m / M) / jnp.maximum(x**3, 1e-30)
    # A_bv = N²r/g (Brunt-Väisälä parameter)
    A_bv = N2 * jnp.maximum(r, 1.0) / jnp.maximum(g, 1e-10)
    # U = 4πr³ρ/m
    U = 4.0 * jnp.pi * r**3 * rho / jnp.maximum(m, 1e-10 * M)
    # q = m/M
    q = m / M

    # Working grid: exclude singular center (x > 1e-4)
    # For JAX traceability, we mask center points by setting coefficients to safe
    # defaults. The integration grid starts above x=1e-4 anyway.
    center_mask = x <= 1e-4  # True for points to zero out

    Vg_safe = jnp.where(center_mask, 0.0, Vg)
    A1_safe = jnp.where(center_mask, 1.0, A1)
    A_bv_safe = jnp.where(center_mask, 0.0, A_bv)
    U_safe = jnp.where(center_mask, 3.0, U)
    q_safe = jnp.where(center_mask, 1e-10, q)

    # NaN/Inf cleanup (same as original numpy path)
    Vg_grid = jnp.nan_to_num(Vg_safe, nan=0.0, posinf=0.0, neginf=0.0)
    A1_grid = jnp.nan_to_num(A1_safe, nan=1.0, posinf=1e10, neginf=1.0)
    A_bv_grid = jnp.nan_to_num(A_bv_safe, nan=0.0, posinf=0.0, neginf=0.0)
    U_grid = jnp.nan_to_num(U_safe, nan=3.0, posinf=3.0, neginf=3.0)
    q_grid = jnp.nan_to_num(q_safe, nan=1e-10, posinf=1.0, neginf=1e-10)

    coeffs = jnp.stack([Vg_grid, A1_grid, A_bv_grid, U_grid, q_grid], axis=0)

    factor = _compute_factor(R, M, G)  # canonical helper (DRY —)

    # ─── Center extension for truncated models (Henyey mesh,) ──────────
    # When the structure starts far from center (x_min > 0.01), the nonradial
    # integrator cannot apply regularity BCs. Prepend synthetic center points
    # with physically correct limits so jnp.interp provides smooth coefficients
    # from x~0 outward. This matches what a MESA FGONG provides naturally.
    #
    # Reference: ADIPLS (CD2008 §2.3) requires center data for the regularity
    # expansion; when x(1)=0 it sets nibc=2 (integrates from point 2 with BCs
    # from the center expansion). Our integration grid starts at x > 1e-4, so
    # we need interpolation-accessible coefficient values down to that level.
    #
    # Center limits:
    #   Vg(0) = 0, A1(0) = ρ_c/ρ̄ ≈ A1(x_min), A_bv(0) = 0, U(0) = 3, q(0) = 0
    # We add points at x = [0, x_min/4, x_min/2] for smooth interpolation.
    x_out, coeffs_out, n_center = _extend_to_center(x, coeffs, A1_grid)

    # Surface Γ₁ — needed for JCD isothermal atmosphere outer BC
    # (Christensen-Dalsgaard 2008, §2.2; ADIPLS setbcs.n.d.f label 45)
    gamma1_s = gamma1[-1]

    return {
        'x_grid': x_out,
        'coeffs': coeffs_out,
        'n_center': n_center,
        'gamma1_s': gamma1_s,
        'factor': factor,
        'M': M, 'R': R, 'G': G,
    }


def _extend_to_center(x_grid, coeffs, A1_grid):
    """Prepend synthetic center points for truncated Henyey-mesh structures.

    The Henyey mesh starts at x ≈ 0.03 (the innermost mass shell, ~3% of R),
    not at center. Nonradial (l≥1) modes require regularity BCs near x→0
    (ADIPLS §2.3). This function prepends 4 synthetic center points with
    physically correct limiting values, extending the grid from x=0 to x_min.

    Also fixes the pathological U coefficient at the innermost Henyey shell
    (where q is clamped → U~5e7) by replacing U[0] with U[1].

    CONSISTENCY REQUIREMENT: this function must produce IDENTICAL output
    whether called from Python (non-traced, e.g. _find_modes) or from inside
    @jax.jit/@jax.jacrev (traced, e.g. _forward_all_obs in _jit_jacobian).
    Any difference between contexts corrupts the IFT eigenfrequency gradient:
    sigma2_anchors computed in the non-traced context must be valid roots in
    the traced context (same coefficients at the same grid positions).

    To guarantee this: x_min_phys is extracted via float(stop_gradient(x_grid[0]))
    when possible (bare jax.grad — succeeds because the primal is concrete).
    Under @jax.jit (where extraction fails with ConcretizationTypeError), a
    fixed fallback constant (0.03) is used. This is safe because the center-
    point positions (all below x=0.015) do not affect eigenfrequency computation
    — the integration grid starts at x_min (≥0.029) and jnp.interp at those
    positions is determined entirely by the physical mesh points.

    GRADIENT PRESERVATION: the original grid point at index 0 is kept, but
    its U coefficient is replaced with U[1] (the NEXT grid point's value).
    U[1] is well-resolved (not clamped) and carries the full gradient from
    the stellar structure (opacity_factor, M, etc.).

    Args:
        x_grid: (N,) JAX array of fractional radii (may be a tracer under JIT)
        coeffs: (5, N) JAX array of [Vg, A1, A_bv, U, q] (may be traced)
        A1_grid: (N,) JAX array — A1 values (unused, kept for API compat)

    Returns:
        (x_out, coeffs_out, n_center): extended arrays with center coverage
            and the count of prepended synthetic points (4 when extended, 0
            if the grid already starts near center)

    References:
        ADIPLS (CD2008 §2.3): center regularity BCs, nibc=2 when x(1)=0
        KWW (2012) §10.1: U → 3, Vg → 0, q → 0 at center
        Unno et al. (1989) §14.1: center expansion coefficients
    """
    x_min = x_grid[0]

    # Extract the concrete x_min value for the early-return check and for
    # computing synthetic center-point positions. Under bare jax.grad (e.g.
    # the seismic gradient tests), stop_gradient + float() succeeds and
    # returns the actual value. Under @jax.jit (e.g. the multimode test's
    # _jit_jacobian via jacrev), it raises ConcretizationTypeError — fall
    # back to a fixed constant. Using the actual value when available is
    # important: it ensures the coefficient grid is identical between the
    # forward reference computation and the AD gradient computation (both
    # running under jax.grad without JIT), preserving the IFT linearization.
    try:
        x_min_phys = float(jax.lax.stop_gradient(x_min))  # GP-2: MESH_POSITIONS
        if x_min_phys < 0.01:
            # Grid already has center coverage (e.g. FGONG with x=0)
            return x_grid, coeffs, 0
    except jax.errors.ConcretizationTypeError:
        # Under @jax.jit: cannot extract concrete value. Use a fixed constant.
        # The actual x_min for Henyey structures is ~0.029-0.032; the exact
        # value only affects synthetic center-point positions (x < 0.015)
        # which are below the integration domain (x >= x_min) and thus do
        # not affect eigenfrequency computation or its gradient.
        x_min_phys = 0.03

    # Synthetic center points: x = [0, x_min_phys/8, x_min_phys/4, x_min_phys/2]
    x_center = jnp.array([0.0, x_min_phys / 8.0, x_min_phys / 4.0, x_min_phys / 2.0])

    # Center coefficient values — FIXED physical limits (stop_gradient'd).
    # Center limits (KWW §10.1; CD2008 §2.3; Unno et al. 1989 §14.1):
    #   Vg(0) = 0, A1 ≈ ρ_c/ρ̄ (≈50 for MS), A_bv(0) = 0, U(0) = 3, q(0) = 0
    A1_center = 50.0
    frac = x_center / x_min_phys  # [0, 0.125, 0.25, 0.5]

    center_Vg = jax.lax.stop_gradient(frac * 0.0)  # GP-2: MESH_POSITIONS
    center_A1 = jax.lax.stop_gradient(jnp.full(4, A1_center))  # GP-2: MESH_POSITIONS
    center_A_bv = jax.lax.stop_gradient(frac * 0.0)  # GP-2: MESH_POSITIONS
    center_U = jax.lax.stop_gradient(3.0 + frac * 3.0)  # GP-2: MESH_POSITIONS
    center_q = jax.lax.stop_gradient(frac * 0.002)  # GP-2: MESH_POSITIONS

    center_coeffs = jnp.stack([center_Vg, center_A1, center_A_bv,
                               center_U, center_q], axis=0)  # (5, 4)

    # Fix pathological U[0]: the Henyey mesh's innermost shell has q clamped
    # to 1e-10*M, making U = 4πr³ρ/m diverge to ~5e7. Replace with U[1]
    # (well-resolved, physical ≈ 3-5). Preserves gradient flow (U[1] depends
    # on opacity_factor through ρ[1]).
    coeffs_fixed = coeffs.at[3, 0].set(coeffs[3, 1])

    x_out = jnp.concatenate([x_center, x_grid])
    coeffs_out = jnp.concatenate([center_coeffs, coeffs_fixed], axis=1)

    return x_out, coeffs_out, 4  # 4 synthetic center points prepended
