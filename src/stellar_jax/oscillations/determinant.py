"""Surface-BC determinant functions for eigenvalue search.

Composes the RK4 integrator with boundary conditions to produce the shooting
determinant D(σ²) whose zeros are eigenfrequencies.

  radial: D = y₁ − y₂ at surface (GYRE vacuum outer BC; l=0)
  nonradial: det of 2×2 surface BC matrix (GYRE vacuum outer: y₁−y₂ and U·y₁+(l+1)y₃+y₄)

Also provides JCD (Christensen-Dalsgaard 2008) isothermal-atmosphere outer BCs
that match to the decaying solution in an isothermal atmosphere above the
photosphere. This shifts absolute frequencies downward by ~10–17 µHz for solar-type
stars, matching GYRE's `outer_bound='JCD'` option.

References:
    Townsend & Teitler (2013), MNRAS 435, 3406 (GYRE; src/ad/ad_bound_m.fypp, rad_bound_m)
    Christensen-Dalsgaard (2008), Ap&SS 316, 113 (ADIPLS; setbcs.n.d.f labels 45+54+57)
"""
import jax.numpy as jnp

from .integrator import _integrate_radial, _integrate_nonradial


# ─── JCD isothermal-atmosphere surface coefficient ─────────────────────────────

def _jcd_surface_coeffs(sigma2, l, Vg_s, A1_s, gamma1_s):
    """Compute the JCD isothermal-atmosphere BC coefficients at the surface.

    Implements GYRE's JCD outer boundary condition (src/eqns/ad/gyre/OB_jcd.inc)
    in the GYRE variable set. The atmospheric eigenvalue chi comes from GYRE's
    atmos_chi (src/common/atmos_m.fypp), which solves the characteristic equation
    for an isothermal atmosphere.

    The BC replaces the vacuum first row [1, -1, 0, 0] with:
        [chi - b_11, -b_12, b_12*(-l-1+λ/(c₁ω²))/(As+Vg), 0]

    where b_11 = Vg-3, b_12 = λ/(c₁ω²)-Vg, chi = atmospheric eigenvalue,
    As = Vg*(Γ₁-1) (isothermal atmosphere), and λ = l(l+1).

    For radial modes (l=0): the equation reduces to the 2×2 case with
    bc1 = (chi-b_11)*y₁ + (-b_12)*y₂ = 0, where b_12 = -Vg (since λ=0).

    Args:
        sigma2: dimensionless σ² (= ω² in GYRE notation, since ω = σ√(R³/GM))
        l: angular degree (0 for radial)
        Vg_s: V/Γ₁ at surface (= V_g in GYRE)
        A1_s: (m/M)/x³ at surface (= 1/c₁ in GYRE, since c₁ = x³M/(m))
        gamma1_s: Γ₁ at surface (adiabatic exponent)

    Returns:
        (bc1_coeffs): tuple (coeff_y1, coeff_y2, coeff_y3) for the first BC row.
        BC equation: coeff_y1*y₁ + coeff_y2*y₂ + coeff_y3*y₃ = 0

    References:
        GYRE src/eqns/ad/gyre/OB_jcd.inc (the BC matrix)
        GYRE src/common/atmos_m.fypp (atmos_chi: atmospheric eigenvalue)
        Christensen-Dalsgaard (2008), Ap&SS 316, 113, §2.2
    """
    ll1 = l * (l + 1.0)
    c1_s = 1.0 / jnp.maximum(A1_s, 1e-30)  # c₁ = 1/A1
    omega2 = sigma2  # ω² = σ² in our normalization

    # Isothermal atmosphere coefficients (GYRE eval_atmos_coeffs_isothrm):
    # V_g stays as is; As = V_g*(Γ₁-1) = V*(1-1/Γ₁)
    As = Vg_s * (gamma1_s - 1.0)

    # Elements of the Jacobian matrix at the surface (GYRE atmos_chi):
    a_11 = Vg_s - 3.0
    a_12 = ll1 / (c1_s * jnp.maximum(omega2, 1e-30)) - Vg_s
    a_21 = c1_s * omega2 - As
    a_22 = As + 1.0

    # Characteristic equation: chi² + b*chi + c = 0
    b = -(a_11 + a_22)
    c = a_11 * a_22 - a_12 * a_21
    psi2 = b * b - 4.0 * c

    # Guard against negative discriminant (above acoustic cutoff)
    psi2_safe = jnp.maximum(psi2, 0.0)
    psi = jnp.sqrt(psi2_safe)

    # chi = (-b - psi)/2 when b >= 0, else 2c/(-b + psi)
    # (numerically stable selection per GYRE atmos_chi_r_)
    chi_pos_b = (-b - psi) / 2.0
    # Guard against division by zero in the b<0 branch
    denom_neg_b = jnp.where(jnp.abs(-b + psi) < 1e-30, 1e-30, -b + psi)
    chi_neg_b = 2.0 * c / denom_neg_b
    chi = jnp.where(b >= 0.0, chi_pos_b, chi_neg_b)

    # When psi2 < 0, fall back to vacuum (chi = -b/2, but we use vacuum directly)
    # Actually GYRE warns and uses chi = -b/2 when psi2 < 0
    chi = jnp.where(psi2 < 0.0, -b / 2.0, chi)

    # BC coefficients from OB_jcd.inc (GYRE variable set):
    # Row 1: [chi - b_11, -b_12, b_12*(-l-1+λ/(c₁ω²))/(As+Vg), 0]
    b_11 = Vg_s - 3.0  # same as a_11
    b_12 = ll1 / (c1_s * jnp.maximum(omega2, 1e-30)) - Vg_s  # same as a_12

    coeff_y1 = chi - b_11
    coeff_y2 = -b_12
    # The y₃ coefficient (gravitational coupling term)
    lam_over_c1w2 = ll1 / (c1_s * jnp.maximum(omega2, 1e-30))
    As_plus_Vg = As + Vg_s
    As_plus_Vg_safe = jnp.where(jnp.abs(As_plus_Vg) < 1e-20, 1e-20, As_plus_Vg)
    coeff_y3 = b_12 * (-l - 1.0 + lam_over_c1w2) / As_plus_Vg_safe

    return coeff_y1, coeff_y2, coeff_y3


# ─── Vacuum boundary condition determinants ────────────────────────────────────


def radial_determinant(sigma2, x_steps, h_steps, x_grid, coeffs):
    """Compute the radial (l=0) surface BC residual for a given σ².

    GYRE formulation (src/rad/rad_bound_m.fypp):
      inner regular: [c₁σ², 0]·y = 0 → y₁ = 0 at the innermost point
      outer vacuum:  [1, -1]·y = 0 → y₁ - y₂ = 0 at the surface

    Args:
        sigma2: dimensionless frequency squared (JAX scalar)
        x_steps: (N_steps,) integration grid x-positions
        h_steps: (N_steps,) integration step sizes
        x_grid: (N,) structure grid
        coeffs: (5, N) structure coefficients

    Returns:
        Scalar determinant value (zero at eigenfrequency)
    """
    # Inner regular BC (GYRE): y₁ = 0 at the innermost point → seed [0, 1]
    y0 = jnp.array([0.0, 1.0])

    y_final = _integrate_radial(sigma2, x_steps, h_steps, y0, x_grid, coeffs)
    y1_s, y2_s = y_final[0], y_final[1]

    # Outer vacuum BC (GYRE): y₁ - y₂ = 0
    return y1_s - y2_s


def nonradial_determinant(sigma2, l, x_steps, h_steps, x_grid, coeffs):
    """Compute the nonradial (l≥1) surface BC determinant for a given σ².

    Two independent solutions from center, surface determinant from 2×2 BCs
    (GYRE formulation; src/ad/ad_bound_m.fypp):
      inner regular: B_i·y = 0 with rows [c₁σ², -l, -l, 0] and [0, 0, l, -1]
      outer vacuum:  B_o·y = 0 with rows [1, -1, 0, 0] and [U, 0, l+1, 1]

    Args:
        sigma2: dimensionless frequency squared (JAX scalar)
        l: angular degree (float)
        x_steps: (N_steps,) integration grid x-positions
        h_steps: (N_steps,) integration step sizes
        x_grid: (N,) structure grid
        coeffs: (5, N) structure coefficients

    Returns:
        Scalar determinant value (zero at eigenfrequency)
    """

    # Inner regular BCs (GYRE, ad_bound_m.fypp build_regular_i_): the two
    # regular solutions span null(B_i). Parameterizing by (y₁, y₃):
    #   y₂ = (c₁σ²·y₁ - l·y₃)/l = (c₁σ²/l)·y₁ - y₃ ,  y₄ = l·y₃
    # Solution 1 (y₁=1, y₃=0); Solution 2 (y₁=0, y₃=1).
    x_c = x_steps[0]
    A1_c = jnp.interp(x_c, x_grid, coeffs[1])
    c1_c = 1.0 / jnp.maximum(A1_c, 1e-30)
    y0_1 = jnp.array([1.0, c1_c * sigma2 / l, 0.0, 0.0])
    y0_2 = jnp.array([0.0, -1.0, 1.0, l])

    y_final_1 = _integrate_nonradial(sigma2, l, x_steps, h_steps, y0_1, x_grid, coeffs)
    y_final_2 = _integrate_nonradial(sigma2, l, x_steps, h_steps, y0_2, x_grid, coeffs)

    y1_1, y2_1, y3_1, y4_1 = y_final_1[0], y_final_1[1], y_final_1[2], y_final_1[3]
    y1_2, y2_2, y3_2, y4_2 = y_final_2[0], y_final_2[1], y_final_2[2], y_final_2[3]

    # U at surface (for the outer vacuum BC)
    x_s = x_steps[-1] + h_steps[-1]
    U_s = jnp.interp(x_s, x_grid, coeffs[3])

    # Surface BC determinant (2×2), GYRE vacuum outer:
    #   bc1 = y₁ - y₂ ;  bc2 = U·y₁ + (l+1)·y₃ + y₄
    bc1_1 = y1_1 - y2_1
    bc1_2 = y1_2 - y2_2
    bc2_1 = U_s * y1_1 + (l + 1.0) * y3_1 + y4_1
    bc2_2 = U_s * y1_2 + (l + 1.0) * y3_2 + y4_2

    return bc1_1 * bc2_2 - bc1_2 * bc2_1


# ─── Wrappers with argument ordering for jax.grad (used by IFT adjoint) ────────

def _radial_det_for_grad(sigma2, coeffs, x_grid, x_steps, h_steps):
    """Radial determinant D(σ², θ) — arguments ordered for jax.grad.

    This is a thin wrapper that calls radial_determinant with the interface
    expected by the IFT: sigma2 as first arg, structure coefficients as second.
    """
    return radial_determinant(sigma2, x_steps, h_steps, x_grid, coeffs)


def _nonradial_det_for_grad(sigma2, l, coeffs, x_grid, x_steps, h_steps):
    """Nonradial determinant D(σ², l, θ) — arguments ordered for jax.grad."""
    return nonradial_determinant(sigma2, l, x_steps, h_steps, x_grid, coeffs)


# ─── JCD isothermal-atmosphere boundary condition determinants ─────────────────

def radial_determinant_jcd(sigma2, x_steps, h_steps, x_grid, coeffs, gamma1_s):
    """Compute the radial (l=0) surface BC residual with JCD outer boundary.

    Same as radial_determinant but replaces the vacuum condition (y₁ - y₂ = 0)
    with the Christensen-Dalsgaard (2008) isothermal-atmosphere matching.

    In the GYRE variable set (src/eqns/ad/gyre/OB_jcd.inc), the JCD BC for
    radial modes (l=0, λ=0) reduces to:
        (chi - b_11)*y₁ + (-b_12)*y₂ = 0
    where b_11 = Vg-3, b_12 = -Vg (since λ/(c₁ω²) = 0 for l=0), and chi is
    the atmospheric eigenvalue from atmos_chi.

    Args:
        sigma2: dimensionless frequency squared (JAX scalar)
        x_steps: (N_steps,) integration grid x-positions
        h_steps: (N_steps,) integration step sizes
        x_grid: (N,) structure grid
        coeffs: (5, N) structure coefficients [Vg, A1, A_bv, U, q]
        gamma1_s: Γ₁ at surface (scalar)

    Returns:
        Scalar determinant value (zero at eigenfrequency)

    References:
        GYRE src/eqns/ad/gyre/OB_jcd.inc
        GYRE src/common/atmos_m.fypp (atmos_chi)
        Christensen-Dalsgaard (2008), Ap&SS 316, 113, §2.2
    """
    y0 = jnp.array([0.0, 1.0])
    y_final = _integrate_radial(sigma2, x_steps, h_steps, y0, x_grid, coeffs)
    y1_s, y2_s = y_final[0], y_final[1]

    # Surface coefficients for JCD BC
    x_s = x_steps[-1] + h_steps[-1]
    Vg_s = jnp.interp(x_s, x_grid, coeffs[0])
    A1_s = jnp.interp(x_s, x_grid, coeffs[1])

    coeff_y1, coeff_y2, _ = _jcd_surface_coeffs(sigma2, 0.0, Vg_s, A1_s, gamma1_s)

    # JCD outer BC (radial): coeff_y1*y₁ + coeff_y2*y₂ = 0
    return coeff_y1 * y1_s + coeff_y2 * y2_s


def nonradial_determinant_jcd(sigma2, l, x_steps, h_steps, x_grid, coeffs, gamma1_s):
    """Compute the nonradial (l≥1) surface BC determinant with JCD outer boundary.

    Same as nonradial_determinant but replaces the vacuum first condition
    (y₁ - y₂ = 0) with the GYRE JCD isothermal-atmosphere matching
    (src/eqns/ad/gyre/OB_jcd.inc):
        (chi-b_11)*y₁ + (-b_12)*y₂ + (b_12*(-l-1+λ/(c₁ω²))/(As+Vg))*y₃ = 0
    The gravitational BC (second condition) is unchanged:
        U*y₁ + (l+1)*y₃ + y₄ = 0

    Args:
        sigma2: dimensionless frequency squared (JAX scalar)
        l: angular degree (float)
        x_steps: (N_steps,) integration grid x-positions
        h_steps: (N_steps,) integration step sizes
        x_grid: (N,) structure grid
        coeffs: (5, N) structure coefficients [Vg, A1, A_bv, U, q]
        gamma1_s: Γ₁ at surface (scalar)

    Returns:
        Scalar determinant value (zero at eigenfrequency)

    References:
        GYRE src/eqns/ad/gyre/OB_jcd.inc
        GYRE src/common/atmos_m.fypp (atmos_chi)
        Christensen-Dalsgaard (2008), Ap&SS 316, 113, §2.2
    """

    # Inner regular BCs — same as nonradial_determinant
    x_c = x_steps[0]
    A1_c = jnp.interp(x_c, x_grid, coeffs[1])
    c1_c = 1.0 / jnp.maximum(A1_c, 1e-30)
    y0_1 = jnp.array([1.0, c1_c * sigma2 / l, 0.0, 0.0])
    y0_2 = jnp.array([0.0, -1.0, 1.0, l])

    y_final_1 = _integrate_nonradial(sigma2, l, x_steps, h_steps, y0_1, x_grid, coeffs)
    y_final_2 = _integrate_nonradial(sigma2, l, x_steps, h_steps, y0_2, x_grid, coeffs)

    y1_1, y2_1, y3_1, y4_1 = y_final_1[0], y_final_1[1], y_final_1[2], y_final_1[3]
    y1_2, y2_2, y3_2, y4_2 = y_final_2[0], y_final_2[1], y_final_2[2], y_final_2[3]

    # Surface structure coefficients
    x_s = x_steps[-1] + h_steps[-1]
    U_s = jnp.interp(x_s, x_grid, coeffs[3])
    Vg_s = jnp.interp(x_s, x_grid, coeffs[0])
    A1_s = jnp.interp(x_s, x_grid, coeffs[1])

    # JCD surface BC coefficients
    coeff_y1, coeff_y2, coeff_y3 = _jcd_surface_coeffs(sigma2, l, Vg_s, A1_s, gamma1_s)

    # Surface BC determinant (2×2):
    #   bc1 (JCD): coeff_y1*y₁ + coeff_y2*y₂ + coeff_y3*y₃
    #   bc2 (gravitational, unchanged): U*y₁ + (l+1)*y₃ + y₄
    bc1_1 = coeff_y1 * y1_1 + coeff_y2 * y2_1 + coeff_y3 * y3_1
    bc1_2 = coeff_y1 * y1_2 + coeff_y2 * y2_2 + coeff_y3 * y3_2
    bc2_1 = U_s * y1_1 + (l + 1.0) * y3_1 + y4_1
    bc2_2 = U_s * y1_2 + (l + 1.0) * y3_2 + y4_2

    return bc1_1 * bc2_2 - bc1_2 * bc2_1
