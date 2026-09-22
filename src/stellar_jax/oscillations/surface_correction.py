"""Ball & Gizon (2014) two-term surface correction for oscillation frequencies.

Implements the "combined" (two-term) near-surface frequency correction from
Ball & Gizon (2014, A&A 568, A123, arXiv:1408.0986), Eq. 4:

    δν_nl = (a₁ · (ν/ν_ac)⁻¹ + a₃ · (ν/ν_ac)³) / I_nl

where:
    ν_ac = acoustic cutoff frequency, scaled from solar:
           ν_ac = 5000 μHz × (g/g☉) / √(Teff/Teff☉)
    I_nl = normalized mode inertia (dimensionless; I ≈ 1 for radial modes)
    a₁, a₃ = nuisance coefficients fit by weighted least-squares to the
              (observed - model) frequency residuals.

The coefficients (a₁, a₃) are determined by solving the 2×2 normal equations:
    (X^T X) [a₁, a₃]^T = X^T y
where the design matrix columns are:
    X₁ᵢ = (νᵢ/ν_ac)⁻¹ / Iᵢ / σᵢ
    X₂ᵢ = (νᵢ/ν_ac)³  / Iᵢ / σᵢ
and yᵢ = (ν_obs,ᵢ - ν_model,ᵢ) / σᵢ.

This matches MESA's implementation in astero_support.f90:get_combined_all_freq_corr
(lines 929–986, r26.04.1), where correction_factor=1 and the Ball & Gizon scheme
uses the normalized inertia I (not the relative inertia Q used by Kjeldsen 2008).

DIFFERENTIABILITY: The corrected frequencies are differentiable w.r.t. the model
frequencies (via JAX). The a₁, a₃ coefficients depend linearly on the residuals,
which depend on the model frequencies → full gradient flow.

References:
    Ball & Gizon (2014), A&A 568, A123, arXiv:1408.0986 (Eq. 3–4)
    MESA: astero/private/astero_support.f90:get_combined_all_freq_corr (r26.04.1)
    Gough (1990), in "Progress of Seismology of the Sun and Stars" (physical basis)
"""

import jax.numpy as jnp

# Solar reference values for acoustic cutoff scaling
# MESA uses nu_ac = 5000 μHz × (g/g☉) / √(Teff/Teff☉)
# (astero_search.defaults, §Surface corrections)
NU_AC_SUN = 5000.0  # μHz — solar acoustic cutoff (Jiménez 2006, ApJ 646, 1398)
G_SUN = 27442.0     # cm/s² — solar surface gravity (IAU 2015: GM/R² → g = 2.74e4)
TEFF_SUN = 5777.0   # K — solar effective temperature (IAU 2015 nominal)


def acoustic_cutoff_frequency(logg, teff):
    """Compute the acoustic cutoff frequency by scaling from solar.

    ν_ac = ν_ac,☉ × (g/g☉) / √(Teff/Teff☉)

    This is the same formula MESA uses (astero_search.defaults, §Surface corrections):
    "νac is the acoustic cutoff frequency, computed by scaling the solar value of
    5mHz in the same way as νmax (i.e. νac = 5mHz * (g/gsun)/sqrt(Teff/Teff_sun))."

    Args:
        logg: log10(g [cm/s²]) — surface gravity
        teff: effective temperature [K]

    Returns:
        ν_ac in μHz
    """
    g = 10.0**logg
    return NU_AC_SUN * (g / G_SUN) / jnp.sqrt(teff / TEFF_SUN)


def surface_correction_bg14(nu_model, nu_obs, sigma, nu_ac, inertia=None):
    """Compute the Ball & Gizon (2014) two-term surface correction.

    Fits the combined (two-term) correction coefficients (a₁, a₃) via weighted
    least-squares, then returns the corrected model frequencies.

    Matches MESA astero_support.f90:get_combined_all_freq_corr (lines 929–986).

    Args:
        nu_model: (N,) model frequencies in μHz (JAX array)
        nu_obs: (N,) observed frequencies in μHz (JAX array)
        sigma: (N,) frequency uncertainties in μHz (JAX array)
        nu_ac: acoustic cutoff frequency in μHz (scalar)
        inertia: (N,) normalized mode inertias (dimensionless).
                 If None, assumes I=1 for all modes (valid for l=0 radial modes).
                 For l>0, proper mode inertias must be provided.

    Returns:
        dict with:
            'nu_corrected': (N,) corrected model frequencies [μHz]
            'a1': inverse-term coefficient
            'a3': cubic-term coefficient
            'delta_nu': (N,) frequency corrections applied [μHz]

    References:
        Ball & Gizon (2014), A&A 568, A123, Eq. 4
        MESA: astero/private/astero_support.f90:get_combined_all_freq_corr
    """
    nu_model = jnp.asarray(nu_model, dtype=jnp.float64)
    nu_obs = jnp.asarray(nu_obs, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)

    if inertia is None:
        inertia = jnp.ones_like(nu_model)
    else:
        inertia = jnp.asarray(inertia, dtype=jnp.float64)

    # Frequency ratio ν/ν_ac
    nu_ratio = nu_model / nu_ac

    # Design matrix columns (matching MESA: X(1) = powm1/I/sigma, X(2) = pow3/I/sigma)
    x1 = (nu_ratio**(-1)) / inertia / sigma  # inverse term
    x2 = (nu_ratio**3) / inertia / sigma     # cubic term

    # Weighted residuals
    y = (nu_obs - nu_model) / sigma

    # Normal equations: (X^T X) [a1, a3]^T = X^T y
    # 2×2 system, solve analytically (matching MESA's explicit inverse)
    xtx_11 = jnp.sum(x1 * x1)
    xtx_12 = jnp.sum(x1 * x2)
    xtx_22 = jnp.sum(x2 * x2)

    xty_1 = jnp.sum(x1 * y)
    xty_2 = jnp.sum(x2 * y)

    # 2×2 matrix inverse via determinant
    det = xtx_11 * xtx_22 - xtx_12 * xtx_12
    # Guard against singular matrix (e.g., single mode)
    det = jnp.maximum(det, 1e-30)

    a1 = (xtx_22 * xty_1 - xtx_12 * xty_2) / det
    a3 = (xtx_11 * xty_2 - xtx_12 * xty_1) / det

    # Apply correction: δν = (a₁·(ν/ν_ac)⁻¹ + a₃·(ν/ν_ac)³) / I
    # (MESA: freq_corr = freq + correction_factor*(a1*powm1(freq)+a3*pow3(freq))/inertia)
    delta_nu = (a1 * nu_ratio**(-1) + a3 * nu_ratio**3) / inertia
    nu_corrected = nu_model + delta_nu

    return {
        'nu_corrected': nu_corrected,
        'a1': a1,
        'a3': a3,
        'delta_nu': delta_nu,
    }


def chi2_with_surface_correction(nu_model, nu_obs, sigma, nu_ac, inertia=None):
    """Compute χ² of model vs observed frequencies WITH the BG14 surface correction.

    This is the key function for frequency fitting: it applies the optimal surface
    correction (fit by least-squares) and returns the χ² of the corrected residuals.
    The correction absorbs the near-surface systematic so the χ² reflects only
    the INTERIOR structure mismatch.

    Differentiable w.r.t. nu_model (for use in the LM optimizer).

    Args:
        nu_model: (N,) model frequencies [μHz]
        nu_obs: (N,) observed frequencies [μHz]
        sigma: (N,) uncertainties [μHz]
        nu_ac: acoustic cutoff frequency [μHz]
        inertia: (N,) normalized mode inertias (default: I=1 for l=0)

    Returns:
        Scalar χ² (sum of squared standardized residuals after correction)
    """
    result = surface_correction_bg14(nu_model, nu_obs, sigma, nu_ac, inertia)
    residuals = (result['nu_corrected'] - nu_obs) / sigma
    return jnp.sum(residuals**2)
