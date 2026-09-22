"""AD structure kernels K^{c²,ρ}(r), K^{Γ₁,ρ}(r), and K^{Z,ρ}(r) for asteroseismic inversions.

Computes the sensitivity of oscillation frequencies to radial perturbations in
the squared sound speed c²(r) and the adiabatic exponent Γ₁(r), both at fixed
density ρ. These are the standard (c²,ρ) and (Γ₁,ρ) kernel pairs consumed by
SOLA/RLS inversions (Dziembowski et al. 1990; Basu & Christensen-Dalsgaard 1997).

Additionally computes the intrinsic-EOS composition kernel K_{Z,ρ}(r), which
captures the sensitivity of ν to changes in metal mass fraction Z through the
EOS dependence of Γ₁: Z → Γ₁(T,P,X,Z) → Vg → σ² → ν. This is the term
dropped by the K_{c²,ρ} = K_{Γ₁,ρ} identity at fixed composition (Basu & CD
1997 Eq. 2; Buldgen, Reese & Dupret 2017; issue #1212).

The AD approach: rather than constructing kernels analytically from eigenfunctions
(as ADIPLS gm1ker.n.d.f does), we differentiate the full eigenfrequency chain
using JAX autodiff:

    Γ₁(r) → build_oscillation_coeffs_jax → eigenfreq_from_coeffs (IFT) → σ² → ν

This gives ∂ν/∂Γ₁(r_i) exactly, via the implicit function theorem adjoint.
The structure kernel is then:

    K_{Γ₁,ρ}(r_i) = (Γ₁(r_i) / ν) · ∂ν/∂Γ₁(r_i)

so that the variational relation holds:

    δν/ν = Σ_i K_{Γ₁,ρ}(r_i) · δΓ₁(r_i)/Γ₁(r_i) · Δr_i / R

At fixed ρ, c² = Γ₁·P/ρ so δc²/c² = δΓ₁/Γ₁ (P is determined by hydrostatic
equilibrium from ρ alone). Therefore K_{c²,ρ} = K_{Γ₁,ρ}.

CONSTRAINT (JAX/differentiability): uses stop_gradient on the FGONG columns not
being differentiated (ρ, P, r, m) to ensure the gradient flows only through
the Γ₁ → Vg path. The center-extension synthetic points also use stop_gradient
(they carry no physical structure data below the innermost Henyey shell).

Normalization convention:
    Our K_{Γ₁,ρ} uses RELATIVE δΓ₁/Γ₁:
        δν/ν = ∫ K_{Γ₁,ρ}(r) · (δΓ₁/Γ₁)(r) · dr/R

    ADIPLS gm1ker uses ABSOLUTE δΓ₁ (adiab.prg.c.tex eq 4.8):
        δω/ω = ∫ K_{ADIPLS}(x) · δΓ₁(x) · dx

    Conversion: K_{ours}(x) = K_{ADIPLS}(x) × Γ₁(x).
    In Model S, Γ₁ ≈ 5/3 throughout the interior, so the kernels differ
    by a near-constant factor ~1.61–1.67 (the kernel-weighted Γ₁ average).

References:
    Basu & Christensen-Dalsgaard (1997), astro-ph/9702162, Eq. 1-3
    Dziembowski, Pamyatnykh & Sienkiewicz (1990), MNRAS 244, 542
    Gough & Thompson (1991), in "Solar Interior and Atmosphere"
    Buldgen, Reese & Dupret (2017), A&A 601, A67 (arXiv:1610.01460)
    Christensen-Dalsgaard (2008), Ap&SS 316, 113 (ADIPLS)
    ADIPLS gm1ker.n.d.f: analytic Γ₁ kernel for comparison
    Verma et al. (2019), MNRAS 483, 4678 (He-glitch Y sensitivity)
"""
import jax
import jax.numpy as jnp
import numpy as np

from .coefficients import _build_oscillation_grid_jax, build_oscillation_coeffs_jax
from .integrator import _make_integration_grid
from .adjoint import eigenfreq_from_coeffs
from .eigenfunction import compute_eigenfunction, compute_mode_inertia


def compute_structure_kernels(glob, var, l, n_pg, nu_min=None, nu_max=None,
                              n_scan=500, n_steps=8000):
    """Compute AD structure kernels K^{c²,ρ}(r) and K^{Γ₁,ρ}(r) for a mode.

    Uses JAX autodiff through the full eigenfrequency chain to obtain the
    per-grid-point sensitivity ∂ν/∂Γ₁(r_i), then normalizes to produce the
    standard kernel convention:

        δν/ν = ∫₀^R K_{Γ₁,ρ}(r) · δΓ₁(r)/Γ₁(r) · dr/R

    The AD gradient is computed via jax.grad through:
        var_jax[:, 9] (Γ₁) → build_oscillation_coeffs_jax → coeffs
        → eigenfreq_from_coeffs (IFT @custom_vjp) → σ² → ν

    At fixed ρ, K_{c²,ρ} = K_{Γ₁,ρ} (since c² = Γ₁P/ρ and P is determined
    by hydrostatic equilibrium from ρ alone at fixed composition).

    Args:
        glob: FGONG global parameters (numpy array, shape ≥15)
        var: FGONG per-point variables center-to-surface (numpy array, shape N×≥15)
        l: angular degree (0, 1, 2, 3, ...)
        n_pg: target radial order (for mode identification)
        nu_min: lower frequency bound for search (µHz); auto if None
        nu_max: upper frequency bound for search (µHz); auto if None
        n_scan: number of trial frequencies for bracket detection
        n_steps: number of RK4 integration steps

    Returns:
        dict with:
            'r': (N,) radial grid in cm (FGONG grid, center-excluded)
            'x': (N,) fractional radius r/R
            'K_c2_rho': (N,) kernel K^{c²,ρ}(r) — per unit dr/R
            'K_gamma1_rho': (N,) kernel K^{Γ₁,ρ}(r) — per unit dr/R
            'nu': float, converged frequency in µHz
            'sigma2': float, dimensionless σ²
            'E': float, normalized mode inertia
            'l': int, angular degree
            'n_pg': int, radial order

    References:
        Basu & Christensen-Dalsgaard (1997), astro-ph/9702162, Eq. 1
        Gough & Thompson (1991), in "Solar Interior and Atmosphere"
        ADIPLS gm1ker.n.d.f (Christensen-Dalsgaard, 1993)
    """
    # ─── Step 1: Find the eigenfrequency and validate mode identification ───
    # Use compute_eigenfunction to get the mode (handles bracket + Brent)
    ef_result = compute_eigenfunction(glob, var, l=l, n_pg=n_pg,
                                      nu_min=nu_min, nu_max=nu_max,
                                      n_scan=n_scan, n_steps=n_steps)
    nu_converged = ef_result['nu']
    sigma2_converged = ef_result['sigma2']

    # Mode inertia for the comparison/normalization diagnostic
    E = compute_mode_inertia(ef_result, glob, var)

    # ─── Step 2: Prepare the differentiable chain ───────────────────────────
    # Build the numpy grid data (for the integration grid, which is static)
    grid_data = _build_oscillation_grid_jax(glob, var)
    x_grid_np = grid_data['x_grid']
    factor = grid_data['factor']

    # Integration grid (static, not differentiated)
    x_steps, h_steps = _make_integration_grid(x_grid_np, n_steps=n_steps)

    # Mask: grid points with x > 1e-4 (same as coefficients.py)
    R = float(glob[1])
    r_fgong = var[:, 0]
    x_fgong = r_fgong / R
    mask = x_fgong > 1e-4
    N_grid = int(np.sum(mask))

    # The FGONG variables as JAX arrays (the differentiable inputs)
    glob_jax = jnp.array(glob, dtype=jnp.float64)
    var_jax = jnp.array(var, dtype=jnp.float64)

    # ─── Step 3: Compute ∂ν/∂Γ₁(r_i) via JAX autodiff ─────────────────────
    # We differentiate ν with respect to var_jax[:, 9] (the Γ₁ column).
    # The chain is: Γ₁ → build_oscillation_coeffs_jax → coeffs → eigenfreq → σ² → ν
    #
    # Since we need per-element gradients ∂ν/∂Γ₁(r_i), we use jax.grad on
    # a function that takes the full Γ₁ array and returns ν.

    sigma2_jax = jnp.float64(sigma2_converged)

    def _nu_from_gamma1(gamma1_arr):
        """Compute ν given a Γ₁ profile (all other structure fixed).

        Replaces var_jax[:, 9] with gamma1_arr, recomputes coefficients,
        and returns the eigenfrequency through the IFT adjoint.
        """
        # Substitute the Γ₁ column
        var_modified = var_jax.at[:, 9].set(gamma1_arr)

        # Build oscillation coefficients (differentiable)
        osc_data = build_oscillation_coeffs_jax(glob_jax, var_modified)
        coeffs = osc_data['coeffs']
        x_grid = osc_data['x_grid']

        # Get σ² through the IFT adjoint
        sigma2 = eigenfreq_from_coeffs(
            coeffs, x_grid, sigma2_jax,
            l, x_steps, h_steps, factor)

        # Convert to ν [µHz] via canonical helper (DRY —).
        # factor is a numpy float here (kernels only vary Γ₁, not M/R), so
        # the gradient through factor is zero anyway — but using the canonical
        # helper prevents formula duplication.
        from stellar_jax.oscillations.seismic_conversion import nu_from_sigma2
        nu = nu_from_sigma2(sigma2, factor)
        return nu

    # Compute the gradient ∂ν/∂Γ₁(r_i) for all grid points at once
    gamma1_arr = var_jax[:, 9]
    dnu_dgamma1 = jax.grad(_nu_from_gamma1)(gamma1_arr)

    # ─── Step 4: Convert to the standard kernel convention ──────────────────
    # K_{Γ₁,ρ}(r_i) is defined such that:
    #   δν/ν = ∫ K_{Γ₁,ρ}(r) · δΓ₁(r)/Γ₁(r) · dr/R
    #
    # In discrete form on the FGONG grid:
    #   δν/ν ≈ Σ_i K_{Γ₁,ρ}(r_i) · δΓ₁(r_i)/Γ₁(r_i) · Δr_i/R
    #
    # From the chain rule:
    #   δν = Σ_i (∂ν/∂Γ₁(r_i)) · δΓ₁(r_i)
    #      = Σ_i (∂ν/∂Γ₁(r_i)) · Γ₁(r_i) · (δΓ₁(r_i)/Γ₁(r_i))
    #
    # To get the per-unit-dr/R form, divide by Δr_i/R:
    #   K_{Γ₁,ρ}(r_i) = (Γ₁(r_i) / ν) · (∂ν/∂Γ₁(r_i)) · R / Δr_i
    #
    # But for a grid-function kernel (value at each grid point), the natural
    # convention is the DISCRETE one:
    #   K_{Γ₁,ρ}(r_i) · Δr_i/R = (Γ₁(r_i) / ν) · (∂ν/∂Γ₁(r_i))
    #
    # We output the per-unit-dr/R form (continuous kernel density).

    gamma1_np = np.array(gamma1_arr)
    dnu_dgamma1_np = np.asarray(dnu_dgamma1)

    # Extract the physical grid (excluding center x <= 1e-4)
    r_out = r_fgong[mask]
    x_out = x_fgong[mask]
    gamma1_out = gamma1_np[mask]
    dnu_dgamma1_out = dnu_dgamma1_np[mask]

    # Grid spacing for the kernel density
    # Use centered differences for interior, one-sided at boundaries
    dr = np.zeros(N_grid)
    dr[1:-1] = (r_out[2:] - r_out[:-2]) / 2.0
    dr[0] = r_out[1] - r_out[0]
    dr[-1] = r_out[-1] - r_out[-2]
    dr_over_R = dr / R

    # The kernel density: K(r_i) such that δν/ν = Σ K(r_i) · (δΓ₁/Γ₁)_i · Δr_i/R
    # K(r_i) = (Γ₁(r_i) / ν) · ∂ν/∂Γ₁(r_i) / (Δr_i/R)
    # But we need to be careful: ∂ν/∂Γ₁ is the per-element derivative.
    # The continuous kernel density is:
    #   K(r) = (Γ₁(r) / ν) · (∂ν/∂Γ₁(r)) · (R / Δr)
    #
    # where (∂ν/∂Γ₁(r)) is the partial derivative of ν w.r.t. Γ₁ at grid point r.

    # Avoid division by zero at boundaries where dr might be small
    dr_over_R_safe = np.maximum(dr_over_R, 1e-30)

    K_gamma1_rho = (gamma1_out / nu_converged) * dnu_dgamma1_out / dr_over_R_safe

    # K_{c²,ρ} = K_{Γ₁,ρ} at fixed ρ
    # (since c² = Γ₁P/ρ, at fixed ρ: δc²/c² = δΓ₁/Γ₁ when P is in HSE)
    K_c2_rho = K_gamma1_rho.copy()

    return {
        'r': r_out,
        'x': x_out,
        'K_c2_rho': K_c2_rho,
        'K_gamma1_rho': K_gamma1_rho,
        'nu': nu_converged,
        'sigma2': sigma2_converged,
        'E': E,
        'l': l,
        'n_pg': n_pg,
        'dnu_dgamma1': dnu_dgamma1_out,
        'dr_over_R': dr_over_R,
    }


def compute_composition_kernels(glob, var, l, n_pg, nu_min=None, nu_max=None,
                                n_scan=500, n_steps=8000):
    """Compute the intrinsic-EOS composition kernel K_{Z,ρ}(r) for a mode.

    Captures how changes in metal mass fraction Z affect oscillation frequencies
    through the EOS dependence of Γ₁ at fixed (P, ρ):

        K_{Z,ρ}(r) = K_{Γ₁,ρ}(r) · (∂lnΓ₁/∂Z)|_{P,ρ}

    so that: δν/ν = ∫ K_{Z,ρ}(r) · δZ(r) · dr/R

    This is the term dropped by the current K_{c²,ρ} = K_{Γ₁,ρ}.copy() formulation.
    Per Basu & Christensen-Dalsgaard (1997, Eq. 2), the full δΓ₁/Γ₁ contains
    the intrinsic-EOS composition term (∂lnΓ₁/∂Y)|_{P,ρ} · δY (or equivalently
    for Z), weighted by K_{Γ₁,ρ}, producing the composition kernel.

    Thermodynamic holding — matching MESA pulse_fgong.f90:371-375:

    Our EOS takes (logT, logP, X, Z), so AD naturally gives (∂Γ₁/∂Z)|_{T,P}.
    Basu & CD 1997 Eq. 2 requires the composition derivative at fixed (P, ρ).
    The transformation is (general thermodynamic identity):

        (∂Γ₁/∂Z)|_{P,ρ} = (∂Γ₁/∂Z)|_{T,P} + (∂Γ₁/∂lnT)|_{P,Z} × (∂lnT/∂Z)|_{P,ρ}

    where (∂lnT/∂Z)|_{P,ρ} = −(∂lnρ/∂Z)|_{T,P} / (∂lnρ/∂lnT)|_{P,Z}.

    This is the FIRST-ORDER correction MESA applies (pulse_fgong.f90:371-375):
        dlnGamma1_dY = dres_dxa(i_Gamma1,1)
           - dres_dlnT(i_Gamma1)*dres_dxa(i_lnPgas,1)/res(i_chiT)
    (MESA transforms from (ρ,T)-fixed; we transform from (T,P)-fixed. Both
    arrive at the same (P,ρ)-fixed result.)

    All three correction quantities are computed via jax.grad through the
    same _eos_opal_table — no implicit solve needed, pure chain-rule.

    MESA reference: pulse_fgong.f90:345-375 (dlnGamma1_dY at fixed P,ρ).
    ADIPLS gm1ker.n.d.f computes K_{Γ₁} only — no composition kernel.
    The composition kernel is novel AD computation, but uses MESA's
    thermodynamic transformation.

    Args:
        glob: FGONG global parameters (numpy array, shape ≥15)
        var: FGONG per-point variables (numpy, N×≥17). Requires columns:
            0: r, 1: ln(m/M), 2: T, 3: P, 4: ρ, 5: X, 9: Γ₁, 16: Z
        l: angular degree (0, 1, 2, 3, ...)
        n_pg: target radial order
        nu_min, nu_max: frequency bounds for search (µHz); auto if None
        n_scan: number of trial frequencies for bracket detection
        n_steps: number of RK4 integration steps

    Returns:
        dict with:
            'r': (N,) radial grid in cm
            'x': (N,) fractional radius r/R
            'K_Z_rho': (N,) composition kernel K_{Z,ρ}(r) at fixed (P,ρ)
            'K_gamma1_rho': (N,) standard Γ₁ kernel (from compute_structure_kernels)
            'dlnGamma1_dZ': (N,) per-zone ∂lnΓ₁/∂Z at fixed (P,ρ)
            'dlnGamma1_dZ_TP': (N,) per-zone ∂lnΓ₁/∂Z at fixed (T,P) [before correction]
            'correction_term': (N,) the (T,P)→(P,ρ) thermodynamic correction
            'nu': float, converged frequency in µHz
            'sigma2': float, dimensionless σ²
            'E': float, normalized mode inertia
            'l': int, angular degree
            'n_pg': int, radial order
            'dr_over_R': (N,) grid spacing / R

    References:
        Basu & Christensen-Dalsgaard (1997), astro-ph/9702162, Eq. 2
        Buldgen, Reese & Dupret (2017), arXiv:1610.01460 (structural pairs)
        Gough & Thompson (1991), in "Solar Interior and Atmosphere"
        Verma et al. (2019), MNRAS 483, 4678 (He-glitch Y sensitivity)
        MESA pulse_fgong.f90:345-375 (thermodynamic transformation)
    """
    from stellar_jax.microphysics.eos import _eos_opal_table

    # ─── Step 1: Get the standard Γ₁ kernel ─────────────────────────────────
    sk = compute_structure_kernels(
        glob, var, l=l, n_pg=n_pg,
        nu_min=nu_min, nu_max=nu_max,
        n_scan=n_scan, n_steps=n_steps)

    # ─── Step 2: Compute ∂lnΓ₁/∂Z|_{P,ρ} at each zone via the EOS chain ───
    # Extract the structural variables needed for the EOS lookup.
    # Required FGONG columns: T (2), P (3), rho (4), X (5), Gamma1 (9), Z (16)
    R = float(glob[1])
    r_fgong = var[:, 0]
    x_fgong = r_fgong / R
    mask = x_fgong > 1e-4  # same mask as compute_structure_kernels
    N_grid = int(np.sum(mask))

    T_arr = var[mask, 2]
    P_arr = var[mask, 3]
    X_arr = var[mask, 5]
    gamma1_arr = var[mask, 9]

    # Z: from column 16 if available, else from glob[3] (uniform)
    if var.shape[1] > 16 and np.any(var[:, 16] != 0):
        Z_arr = var[mask, 16]
    else:
        Z_arr = np.full(N_grid, float(glob[3]))

    from stellar_jax.config.constants import a_rad
    from stellar_jax.config.physics_floors import PGAS_FRAC_FLOOR

    # Prepare JAX arrays for vectorized computation
    T_jax = jnp.array(T_arr, dtype=jnp.float64)
    P_total_jax = jnp.array(P_arr, dtype=jnp.float64)
    X_jax = jnp.array(X_arr, dtype=jnp.float64)
    Z_jax = jnp.array(Z_arr, dtype=jnp.float64)
    gamma1_jax_arr = jnp.array(gamma1_arr, dtype=jnp.float64)

    # Compute logT and logPgas from FGONG T and P_total
    logT_jax = jnp.log10(jnp.maximum(T_jax, 1.0))
    P_rad_jax = a_rad * T_jax**4 / 3.0
    P_gas_jax = jnp.maximum(P_total_jax - P_rad_jax,
                            PGAS_FRAC_FLOOR * P_total_jax)
    logPgas_jax = jnp.log10(jnp.maximum(P_gas_jax, 1.0))

    # ─── Step 2a: Scalar helpers for AD through the EOS ─────────────────────
    # CONSTRAINT (JAX): eos_lookup's try/except concrete-value check is
    # not compatible with vmap/jit; _eos_opal_table is a pure function.

    def _gamma1_scalar(logT_i, logPgas_i, X_i, Z_i):
        """Compute Γ₁ for a single zone from EOS.
        Γ₁ = χ_ρ / max(1 − ∇_ad · χ_T, 1e-10).
        """
        rho_e, mu_e, nad_e, S_e, cp_e, chirho_e, chiT_e = _eos_opal_table(
            logT_i, logPgas_i, X_i, Z_i)
        return chirho_e / jnp.maximum(1.0 - nad_e * chiT_e, 1e-10)

    def _lnrho_scalar(logT_i, logPgas_i, X_i, Z_i):
        """Compute ln(ρ) for a single zone from EOS."""
        rho_e, mu_e, nad_e, S_e, cp_e, chirho_e, chiT_e = _eos_opal_table(
            logT_i, logPgas_i, X_i, Z_i)
        return jnp.log(jnp.maximum(rho_e, 1e-30))

    # ─── Step 2b: (∂Γ₁/∂Z)|_{T,P} — the raw (T,P)-fixed derivative ────────
    def _dGamma1_dZ_TP(logT_i, logPgas_i, X_i, Z_i):
        """∂Γ₁/∂Z at fixed (T, P_gas, X) for one zone, via AD."""
        return jax.grad(_gamma1_scalar, argnums=3)(
            logT_i, logPgas_i, X_i, Z_i)

    dGamma1_dZ_TP_jax = jax.vmap(_dGamma1_dZ_TP)(
        logT_jax, logPgas_jax, X_jax, Z_jax)

    # ─── Step 2c: Thermodynamic correction (T,P)-fixed → (P,ρ)-fixed ───────
    # Identity (Basu & CD 1997; MESA pulse_fgong.f90:371-375):
    #   (∂Γ₁/∂Z)|_{P,ρ} = (∂Γ₁/∂Z)|_{T,P}
    #                       + (∂Γ₁/∂lnT)|_{P,Z} × (∂lnT/∂Z)|_{P,ρ}
    #
    # where (∂lnT/∂Z)|_{P,ρ} = −(∂lnρ/∂Z)|_{T,P} / (∂lnρ/∂lnT)|_{P,Z}
    #
    # All three AD derivatives are through the same _eos_opal_table.

    # (∂Γ₁/∂lnT)|_{P,Z}: derivative of Γ₁ w.r.t. logT at fixed (logP, X, Z).
    # Note: logT = log10(T), so ∂/∂lnT = (1/ln10) · ∂/∂logT.
    def _dGamma1_dlogT(logT_i, logPgas_i, X_i, Z_i):
        """∂Γ₁/∂logT at fixed (P_gas, X, Z) for one zone."""
        return jax.grad(_gamma1_scalar, argnums=0)(
            logT_i, logPgas_i, X_i, Z_i)

    dGamma1_dlogT_jax = jax.vmap(_dGamma1_dlogT)(
        logT_jax, logPgas_jax, X_jax, Z_jax)
    # Convert ∂Γ₁/∂logT to ∂Γ₁/∂lnT: ∂/∂lnT = (1/ln10) · ∂/∂logT
    ln10 = jnp.log(10.0)
    dGamma1_dlnT_jax = dGamma1_dlogT_jax / ln10

    # (∂lnρ/∂Z)|_{T,P}: derivative of ln(ρ) w.r.t. Z at fixed (logT, logP, X)
    def _dlnrho_dZ(logT_i, logPgas_i, X_i, Z_i):
        """∂ln(ρ)/∂Z at fixed (T, P_gas, X) for one zone."""
        return jax.grad(_lnrho_scalar, argnums=3)(
            logT_i, logPgas_i, X_i, Z_i)

    dlnrho_dZ_jax = jax.vmap(_dlnrho_dZ)(
        logT_jax, logPgas_jax, X_jax, Z_jax)

    # (∂lnρ/∂lnT)|_{P,Z}: derivative of ln(ρ) w.r.t. lnT at fixed (logP, X, Z)
    def _dlnrho_dlogT(logT_i, logPgas_i, X_i, Z_i):
        """∂ln(ρ)/∂logT at fixed (P_gas, X, Z) for one zone."""
        return jax.grad(_lnrho_scalar, argnums=0)(
            logT_i, logPgas_i, X_i, Z_i)

    dlnrho_dlogT_jax = jax.vmap(_dlnrho_dlogT)(
        logT_jax, logPgas_jax, X_jax, Z_jax)
    # Convert: ∂lnρ/∂lnT = (1/ln10) · ∂lnρ/∂logT
    dlnrho_dlnT_jax = dlnrho_dlogT_jax / ln10

    # Assemble: (∂lnT/∂Z)|_{P,ρ} = −(∂lnρ/∂Z)|_{T,P} / (∂lnρ/∂lnT)|_{P,Z}
    # Guard against division by zero (dlnrho_dlnT should be ~ -chi_rho/chi_T < 0)
    dlnT_dZ_Prho_jax = -dlnrho_dZ_jax / jnp.where(
        jnp.abs(dlnrho_dlnT_jax) > 1e-30,
        dlnrho_dlnT_jax,
        -1.0)  # fallback: ideal gas → dlnrho/dlnT ~ -1

    # Correction to Γ₁: (∂Γ₁/∂lnT)|_{P,Z} × (∂lnT/∂Z)|_{P,ρ}
    correction_gamma1 = dGamma1_dlnT_jax * dlnT_dZ_Prho_jax

    # Full (P,ρ)-fixed derivative:
    #   (∂Γ₁/∂Z)|_{P,ρ} = (∂Γ₁/∂Z)|_{T,P} + correction
    dGamma1_dZ_Prho_jax = dGamma1_dZ_TP_jax + correction_gamma1

    # Convert to logarithmic: ∂lnΓ₁/∂Z = (1/Γ₁) · ∂Γ₁/∂Z
    dlnGamma1_dZ_TP = np.asarray(
        dGamma1_dZ_TP_jax / jnp.maximum(gamma1_jax_arr, 1e-30))
    dlnGamma1_dZ_Prho = np.asarray(
        dGamma1_dZ_Prho_jax / jnp.maximum(gamma1_jax_arr, 1e-30))
    correction_term = np.asarray(
        correction_gamma1 / jnp.maximum(gamma1_jax_arr, 1e-30))

    # ─── Step 3: Compose the composition kernel at (P,ρ)-fixed ──────────────
    # K_{Z,ρ}(r) = K_{Γ₁,ρ}(r) · (∂lnΓ₁/∂Z)|_{P,ρ}
    # Convention: δν/ν = ∫ K_{Z,ρ}(r) · δZ(r) · dr/R
    K_Z_rho = sk['K_gamma1_rho'] * dlnGamma1_dZ_Prho

    return {
        'r': sk['r'],
        'x': sk['x'],
        'K_Z_rho': K_Z_rho,
        'K_gamma1_rho': sk['K_gamma1_rho'],
        'dlnGamma1_dZ': dlnGamma1_dZ_Prho,
        'dlnGamma1_dZ_TP': dlnGamma1_dZ_TP,
        'correction_term': correction_term,
        'nu': sk['nu'],
        'sigma2': sk['sigma2'],
        'E': sk['E'],
        'l': sk['l'],
        'n_pg': sk['n_pg'],
        'dr_over_R': sk['dr_over_R'],
        'dnu_dgamma1': sk['dnu_dgamma1'],
    }


def composition_kernel_integral(kernel_result, delta_Z):
    """Compute δν/ν from the composition kernel and a δZ profile.

    Evaluates:
        δν/ν = ∫ K_{Z,ρ}(r) · δZ(r) · dr/R

    Args:
        kernel_result: dict from compute_composition_kernels
        delta_Z: (N,) array — perturbation δZ on the same grid

    Returns:
        float: δν/ν (fractional frequency shift from composition change)

    References:
        Basu & Christensen-Dalsgaard (1997), astro-ph/9702162, Eq. 3
    """
    K = kernel_result['K_Z_rho']
    dr_over_R = kernel_result['dr_over_R']
    return float(np.sum(K * delta_Z * dr_over_R))


def kernel_integral(kernel_result, delta_X_over_X):
    """Compute δν/ν from a kernel and a perturbation profile.

    Evaluates the discrete version of:
        δν/ν = ∫ K(r) · δX(r)/X(r) · dr/R

    Args:
        kernel_result: dict from compute_structure_kernels
        delta_X_over_X: (N,) array — relative perturbation δΓ₁/Γ₁ or δc²/c²
            on the same grid as kernel_result['x']

    Returns:
        float: δν/ν (fractional frequency shift)
    """
    K = kernel_result['K_gamma1_rho']
    dr_over_R = kernel_result['dr_over_R']
    return float(np.sum(K * delta_X_over_X * dr_over_R))
