"""Neutrino energy loss rates (plasma + photo + pair).

References:
  - Plasma: Haft, Raffelt & Weiss (1994, ApJ 425, 222) Eq. 23-27
    via the MESA implementation (Timmes). Normalization constant
    0.93153 from electroweak couplings (Itoh et al. 1996 §4).
  - Photo: Itoh et al. (1996, ApJS 102, 411) Eq. 3.1–3.8, Table 2.
    Fourier-Chebyshev expansion with temperature-regime-dependent
    coefficients, following MESA's mod_neu.f90 (Timmes).
  - Pair: Itoh et al. (1996, ApJS 102, 411) Eq. 2.1–2.8.
    Following MESA's mod_neu.f90 implementation exactly.

All rates returned in erg/g/s (specific energy loss rate).
"""
import jax
import jax.numpy as jnp

from stellar_jax.microphysics.safe_math import safe_power


# Physical constants (CGS)
_ME_C2_K = 5.9302e9  # m_e c^2 / k_B in K
_CON1 = 1.0 / 5.9302  # kT/(m_e c^2) = T9 * con1

# Weinberg angle sin²θ_W = 0.2229 (MESA const_def.f90; consistent with
# Itoh et al. 1996, ApJS 102, 411 calibration)
_SIN2_TW = 0.2229
_CV = 0.5 + 2.0 * _SIN2_TW   # 0.9458
_CVP = 1.0 - _CV               # 0.0542
_CA = 0.5
_CAP = 1.0 - _CA               # 0.5
# Itoh et al. (1996) Eq. 2.4, 2.5 coupling factors (3 neutrino flavors)
_TFAC1 = _CV**2 + _CA**2 + 2.0 * (_CVP**2 + _CAP**2)  # ~1.6504
_TFAC2 = _CV**2 - _CA**2 + 2.0 * (_CVP**2 - _CAP**2)  # ~0.1504
_TFAC3 = _TFAC2 / _TFAC1
_TFAC4 = 0.5 * _TFAC1
# Fourier-Chebyshev frequencies (Itoh 1996 Eq. 3.7)
_FAC1 = 5.0 * jnp.pi / 3.0
_FAC2 = 10.0 * jnp.pi

# Photo neutrino Fourier-Chebyshev coefficients (Itoh et al. 1996, Table 2).
# Each regime tuple: (c00-c06, c10-c16, c20-c26, dd01-dd05, dd11-dd15, dd21-dd25)
_PHOTO_COEFF_R1 = (  # Regime 1: 10^7 - 10^8 K
    1.008e11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    8.156e10, 9.728e8, -3.806e9, -4.384e9, -5.774e9, -5.249e9, -5.153e9,
    1.067e11, -9.782e9, -7.193e9, -6.936e9, -6.893e9, -7.041e9, -7.193e9,
    0.0, 0.0, 0.0, 0.0, 0.0,
    -1.879e10, -9.667e9, -5.602e9, -3.370e9, -1.825e9,
    -2.919e10, -1.185e10, -7.270e9, -4.222e9, -1.560e9,
)
_PHOTO_COEFF_R2 = (  # Regime 2: 10^8 - 10^9 K
    9.889e10, -4.524e8, -6.088e6, 4.269e7, 5.172e7, 4.910e7, 4.388e7,
    1.813e11, -7.556e9, -3.304e9, -1.031e9, -1.764e9, -1.851e9, -1.928e9,
    9.750e10, 3.484e10, 5.199e9, -1.695e9, -2.865e9, -3.395e9, -3.418e9,
    -1.135e8, 1.256e8, 5.149e7, 3.436e7, 1.005e7,
    1.652e9, -3.119e9, -1.839e9, -1.458e9, -8.956e8,
    -1.548e10, -9.338e9, -5.899e9, -3.035e9, -1.598e9,
)
_PHOTO_COEFF_R3 = (  # Regime 3: T >= 10^9 K
    9.581e10, 4.107e8, 2.305e8, 2.236e8, 1.580e8, 2.165e8, 1.721e8,
    1.459e12, 1.314e11, -1.169e11, -1.765e11, -1.867e11, -1.983e11, -1.896e11,
    2.424e11, -3.669e9, -8.691e9, -7.967e9, -7.932e9, -7.987e9, -8.333e9,
    4.724e8, 2.976e8, 2.242e8, 7.937e7, 4.859e7,
    -7.094e11, -3.697e11, -2.189e11, -1.273e11, -5.705e10,
    -2.254e10, -1.551e10, -7.793e9, -4.489e9, -2.185e9,
)


def _fourier_chebyshev(tau_r, c00r, c01r, c02r, c03r, c04r, c05r, c06r,
                       c10r, c11r, c12r, c13r, c14r, c15r, c16r,
                       c20r, c21r, c22r, c23r, c24r, c25r, c26r,
                       dd01r, dd02r, dd03r, dd04r, dd05r,
                       dd11r, dd12r, dd13r, dd14r, dd15r,
                       dd21r, dd22r, dd23r, dd24r, dd25r):
    """Evaluate a 3-row Fourier-Chebyshev expansion (Itoh et al. 1996, Eq. 3.7)."""
    cos1 = jnp.cos(_FAC1 * tau_r)
    sin1 = jnp.sin(_FAC1 * tau_r)
    sin2 = 2.0 * sin1 * cos1
    cos2 = 2.0 * cos1 * cos1 - 1.0
    sin3 = sin1 * (3.0 - 4.0 * sin1 * sin1)
    cos3 = cos1 * (4.0 * cos1 * cos1 - 3.0)
    sin4 = 2.0 * sin2 * cos2
    cos4 = 2.0 * cos2 * cos2 - 1.0
    sin5 = sin1 * (5.0 - sin1 * sin1 * (20.0 - 16.0 * sin1 * sin1))
    cos5 = cos1 * (cos1 * cos1 * (16.0 * cos1 * cos1 - 20.0) + 5.0)
    last = jnp.cos(_FAC2 * tau_r)
    a0 = (0.5 * c00r + c01r * cos1 + dd01r * sin1
          + c02r * cos2 + dd02r * sin2 + c03r * cos3 + dd03r * sin3
          + c04r * cos4 + dd04r * sin4 + c05r * cos5 + dd05r * sin5
          + 0.5 * c06r * last)
    a1 = (0.5 * c10r + c11r * cos1 + dd11r * sin1
          + c12r * cos2 + dd12r * sin2 + c13r * cos3 + dd13r * sin3
          + c14r * cos4 + dd14r * sin4 + c15r * cos5 + dd15r * sin5
          + 0.5 * c16r * last)
    a2 = (0.5 * c20r + c21r * cos1 + dd21r * sin1
          + c22r * cos2 + dd22r * sin2 + c23r * cos3 + dd23r * sin3
          + c24r * cos4 + dd24r * sin4 + c25r * cos5 + dd25r * sin5
          + 0.5 * c26r * last)
    return a0, a1, a2


def _neutrino_taper_factor(T):
    """MESA low-T cosine taper factor for neutrino losses.

    Smoothly ramps all neutrino loss rates from 0 to 1 between
    log10(T) = 7.0 and log10(T) = 7.5 using a cosine taper (C1-smooth).
    Below log10(T) = 7.0, the factor is 0 (all rates vanish).
    Above log10(T) = 7.5, the factor is 1 (rates unmodified).

    Matches MESA mod_neu.f90:366-369 exactly:
      tcutoff_factor = 0.5 * (1 - cos(pi * (logT - log10Tmin_neu)
                                        / (log10_Tlim - log10Tmin_neu)))
    where log10Tmin_neu = 7.0 (neu_def.f90:24) and log10_Tlim = 7.5
    (star/private/neu.f90:60).

    CONSTRAINT: uses jnp.where for the regime boundaries (instead of
    MESA's early-return) to preserve JAX traceability. The cosine itself
    is C1-smooth, so gradients are well-behaved everywhere.
    """
    # MESA constants (neu_def.f90:24, star/private/neu.f90:60)
    _LOG10_TMIN_NEU = 7.0
    _LOG10_TLIM = 7.5

    logT = jnp.log10(jnp.maximum(T, 1.0))  # safe log10
    x = (logT - _LOG10_TMIN_NEU) / (_LOG10_TLIM - _LOG10_TMIN_NEU)

    # Cosine taper: 0.5*(1 - cos(pi*x)) for x in [0, 1]
    factor = 0.5 * (1.0 - jnp.cos(jnp.pi * jnp.clip(x, 0.0, 1.0)))

    # Below logTmin: hard zero (MESA mod_neu.f90:224-228)
    factor = jnp.where(logT <= _LOG10_TMIN_NEU, 0.0, factor)

    return factor


def epsilon_neutrino(rho, T, X, Z):
    """Total neutrino energy loss rate (erg/g/s).

    Computes plasma + photo + pair neutrino losses per gram, then applies
    MESA's low-T cosine taper (mod_neu.f90:366-399) to smoothly ramp all
    rates to zero between logT=7.0 and logT=7.5. Below logT=7.0, all
    rates are exactly zero (MESA mod_neu.f90:224-228).

    Bremsstrahlung and recombination omitted (negligible pre-flash
    for H/He composition).

    Parameters
    ----------
    rho : density (g/cm³)
    T : temperature (K)
    X : hydrogen mass fraction
    Z : metal mass fraction

    Returns
    -------
    eps_nu : total neutrino loss rate (erg/g/s), always >= 0
    """
    rho = jnp.maximum(rho, 1e-10)
    T = jnp.maximum(T, 1e4)

    eps_plas = _plasma_neutrino(rho, T, X, Z)
    eps_phot = _photo_neutrino(rho, T, X, Z)
    eps_pair = _pair_neutrino(rho, T, X, Z)

    # Apply MESA's low-T cosine taper to the sum of all sources
    # (equivalent to applying it per-source as MESA does, since it's
    # the same multiplicative factor for all). This replaces the previous
    # hard photo step at T<1e7 with a smooth C1 ramp.
    # Ref: MESA mod_neu.f90:366-399 (tcutoff_factor applied to all sources).
    taper = _neutrino_taper_factor(T)

    return jnp.maximum((eps_plas + eps_phot + eps_pair) * taper, 0.0)


def _mu_e(X, Z):
    """Mean molecular weight per electron: μ_e = 2/(1+X).

    Kippenhahn & Weigert (1990) Eq. 13.7.
    """
    return 2.0 / (1.0 + X)


def _plasma_neutrino(rho, T, X, Z):
    """Plasma neutrino emission rate (erg/g/s).

    MESA's implementation of Haft, Raffelt & Weiss (1994) /
    Itoh et al. (1996) §4. Formula:
      Q_plas = 0.93153 * 3.0e21 * λ^9 * γ^6 * exp(-γ) * (fT + fL) * fxy
    in erg/cm³/s, divided by ρ → erg/g/s.

    The normalization 0.93153 is the electroweak coupling factor
    from Itoh et al. (1996) Eq. 4.1 (CV² + CA² summed over flavors).
    """
    mu_e = _mu_e(X, Z)
    lam = T / _ME_C2_K  # = xl in MESA

    # γ² (plasma frequency parameter, Itoh Eq. 4.6)
    rm = rho / mu_e
    a1 = 1.019e-6 * rm
    # AD-safe a1^(2/3): gradient (2/3)*a1^(-1/3) diverges as a1→0.
    # Floor at 1e-4: below this, plasma neutrinos are negligible (Itoh et al.
    # 1996 §4 validates for ρ/μ_e > 10 g/cm³ → a1 > 1e-5). The floor makes
    # the gradient bounded at (2/3)*(1e-4)^(-1/3) ≈ 14.3.
    a2 = safe_power(a1, 2.0 / 3.0, 1e-4)
    b1 = jnp.sqrt(1.0 + a2)
    gamma2 = 1.1095e11 * rm / (T * T * b1)
    gamma = jnp.sqrt(jnp.maximum(gamma2, 1e-30))

    # fT (Itoh Eq. 4.7)
    gl12 = jnp.sqrt(gamma)
    gl32 = gamma * gl12
    f_T = 2.4 + 0.6 * gl12 + 0.51 * gamma + 1.25 * gl32

    # fL (Itoh Eq. 4.8)
    f_L = (8.6 * gamma2 + 1.35 * gamma2 * gl32) / (225.0 - 17.0 * gamma + gamma2)

    # Correction factor fxy (Itoh Eq. 4.9–4.11)
    log_2rm = jnp.log10(jnp.maximum(2.0 * rm, 1e-30))
    logT = jnp.log10(T)
    xnum = (17.5 + log_2rm - 3.0 * logT) / 6.0
    xden = (-24.5 + log_2rm + 3.0 * logT) / 6.0

    # Eq. 4.11: fxy applied only when |xnum| <= 0.7 and xden >= 0
    a1_inner = 0.39 - 1.25 * xnum - 0.35 * jnp.sin(4.5 * xnum)
    b1_inner = 0.3 * jnp.exp(-1.0 * (4.5 * xnum + 0.9) ** 2)
    c_inner = jnp.minimum(0.0, xden - 1.6 + 1.25 * xnum)
    d_inner = 0.57 - 0.25 * xnum
    a3_inner = c_inner / d_inner
    c00_inner = jnp.exp(-1.0 * a3_inner * a3_inner)
    fxy_raw = 1.05 + (a1_inner - b1_inner) * c00_inner
    use_fxy = (jnp.abs(xnum) <= 0.7) & (xden >= 0.0)
    f_xy = jnp.where(use_fxy, fxy_raw, 1.0)

    # Q_plas in erg/cm³/s (Itoh Eq. 4.1, 4.5)
    lam9 = lam ** 9
    gamma6 = gamma2 ** 3
    Q_plas = 0.93153 * 3.0e21 * lam9 * gamma6 * jnp.exp(-gamma) * (f_T + f_L) * f_xy

    return Q_plas / rho


def _photo_neutrino(rho, T, X, Z):
    """Photoneutrino emission rate (erg/g/s).

    Itoh et al. (1996, ApJS 102, 411) Eq. 3.1–3.8, Table 2.
    Uses the Fourier-Chebyshev expansion with temperature-regime-dependent
    coefficients, following MESA's mod_neu.f90 (Timmes) exactly.

    The photo process γ + e⁻ → e⁻ + ν + ν̄.
    """
    mu_e = _mu_e(X, Z)
    rm = rho / mu_e

    # Dimensionless quantities
    t9 = T * 1e-9
    xl = t9 * _CON1   # kT/(m_e c²)
    xl2 = xl * xl
    xl3 = xl2 * xl
    xl4 = xl3 * xl
    xl5 = xl4 * xl
    xlm1 = 1.0 / jnp.maximum(xl, 1e-30)
    xlm2 = xlm1 * xlm1
    xlm3 = xlm2 * xlm1

    # ζ = (ρ_e/10⁹)^(1/3) / λ  (Itoh Eq. 2.9)
    a0 = rm * 1e-9
    # AD-safe a0^(1/3): gradient (1/3)*a0^(-2/3) diverges as a0→0.
    # Floor at 1e-12: photo neutrinos are negligible below T<1e7 K (enforced
    # at line 318) and at densities ρ/μ_e < 1e-3 g/cm³. Itoh et al. (1996)
    # §3 validates for log(ρ/μ_e) ∈ [-1, 14].
    a1 = safe_power(jnp.maximum(a0, 1e-30), 1.0 / 3.0, 1e-12)
    zeta = a1 * xlm1
    zeta2 = zeta * zeta
    zeta3 = zeta2 * zeta

    # Temperature-dependent coefficients (Table 2 of Itoh et al. 1996)
    # Smooth sigmoid blending at regime boundaries for JAX autodiff.
    logT = jnp.log10(T)
    _W = 100.0  # steepness: transition over ~0.02 dex
    w1 = jax.nn.sigmoid(_W * (logT - 7.0)) * jax.nn.sigmoid(_W * (8.0 - logT))
    w2 = jax.nn.sigmoid(_W * (logT - 8.0)) * jax.nn.sigmoid(_W * (9.0 - logT))
    w3 = jax.nn.sigmoid(_W * (logT - 9.0))
    w_sum = w1 + w2 + w3 + 1e-30
    w1 = w1 / w_sum
    w2 = w2 / w_sum
    w3 = w3 / w_sum

    tau_1 = jnp.clip(logT - 7.0, 0.0, 1.0)
    tau_2 = jnp.clip(logT - 8.0, 0.0, 1.0)
    tau_3 = jnp.clip(logT - 9.0, 0.0, 1.0)
    cc_1 = 0.5654 + tau_1
    cc_2 = 1.5654
    cc_3 = 1.5654

    # Coefficients stored as module-level tuples (_PHOTO_COEFF_R1/R2/R3).

    a0_1, a1_1, a2_1 = _fourier_chebyshev(tau_1, *_PHOTO_COEFF_R1)
    a0_2, a1_2, a2_2 = _fourier_chebyshev(tau_2, *_PHOTO_COEFF_R2)
    a0_3, a1_3, a2_3 = _fourier_chebyshev(tau_3, *_PHOTO_COEFF_R3)

    # Blend: a0 = w1*a0_1 + w2*a0_2 + w3*a0_3
    a0_fc = w1 * a0_1 + w2 * a0_2 + w3 * a0_3
    a1_fc = w1 * a1_1 + w2 * a1_2 + w3 * a1_3
    a2_fc = w1 * a2_1 + w2 * a2_2 + w3 * a2_3
    cc = w1 * cc_1 + w2 * cc_2 + w3 * cc_3

    # fphot (Eq. 3.4): numerator / denominator
    dum = a0_fc + a1_fc * zeta + a2_fc * zeta2
    xnum_fc = dum * jnp.exp(-cc * zeta)
    xden_fc = zeta3 + 6.290e-3 * xlm1 + 7.483e-3 * xlm2 + 3.061e-4 * xlm3
    fphot = xnum_fc / jnp.maximum(xden_fc, 1e-30)

    # qphot (Eq. 3.3)
    a0_q = 1.0 + 2.045 * xl
    xnum_q = 0.666 * jnp.power(jnp.maximum(a0_q, 1e-30), -2.066)
    dum_q = 1.875e8 * xl + 1.653e8 * xl2 + 8.499e8 * xl3 - 1.604e8 * xl4
    xden_q = 1.0 + rm / jnp.maximum(dum_q, 1e-30)
    qphot = xnum_q / jnp.maximum(xden_q, 1e-30)

    # sphot (Eq. 3.2): rm * xl^5 * fphot * tfac4 * (1 - tfac3*qphot)
    sphot = rm * xl5 * fphot * _TFAC4 * (1.0 - _TFAC3 * qphot)

    # Low-T guard: the MESA cosine taper (applied in epsilon_neutrino) smoothly
    # ramps all rates to zero below logT=7.5. The previous hard step at T<1e7
    # is removed — the taper is C1-smooth and handles this regime correctly.

    return jnp.maximum(sphot / rho, 0.0)


def _pair_neutrino(rho, T, X, Z):
    """Pair-annihilation neutrino emission rate (erg/g/s).

    Itoh et al. (1996, ApJS 102, 411) Eq. 2.1–2.8.
    Following MESA's mod_neu.f90 implementation exactly.

    The pair process e⁺ + e⁻ → ν + ν̄ requires T ≳ 10⁹ K to produce
    significant e⁺e⁻ pairs. Negligible on MS and RGB pre-flash.
    """
    mu_e = _mu_e(X, Z)
    rm = rho / mu_e

    # Dimensionless temperature
    t9 = T * 1e-9
    xl = t9 * _CON1   # kT/(m_e c²)
    xlp5 = jnp.sqrt(jnp.maximum(xl, 1e-30))
    xl2 = xl * xl
    xl3 = xl2 * xl
    xl4 = xl3 * xl
    xl5 = xl4 * xl
    xl6 = xl5 * xl
    xl7 = xl6 * xl
    xl8 = xl7 * xl
    xlm1 = 1.0 / jnp.maximum(xl, 1e-30)
    xlm2 = xlm1 * xlm1
    xlm3 = xlm2 * xlm1

    # ζ = (ρ_e/10⁹)^(1/3) / λ  (Itoh Eq. 2.9)
    a0 = rm * 1e-9
    # AD-safe a0^(1/3): same pattern as photo neutrino (floor at 1e-12).
    # Pair neutrinos require T > 10^9 K to produce significant e⁺e⁻ pairs,
    # so the gradient contribution at low density is negligible.
    a1_z = safe_power(jnp.maximum(a0, 1e-30), 1.0 / 3.0, 1e-12)
    zeta = a1_z * xlm1
    zeta2 = zeta * zeta
    zeta3 = zeta2 * zeta

    # gl polynomial (Eq. 2.8) — NO CLAMP, same as MESA.
    # At low xl where gl<0, exp(-2/xl) makes the product negligible.
    gl = 1.0 - 13.04 * xl2 + 133.5 * xl4 + 1534.0 * xl6 + 918.6 * xl8

    # fpair (Eq. 2.7): numerator with regime-dependent exponential
    # MESA: low-T uses exp(-5.5924*zeta), high-T uses exp(-4.9924*zeta)
    a1_coeff = 6.002e19 + 2.084e20 * zeta + 1.872e21 * zeta2
    b1_lo = jnp.exp(-5.5924 * zeta)
    b1_hi = jnp.exp(-4.9924 * zeta)
    b1 = jnp.where(t9 < 10.0, b1_lo, b1_hi)
    xnum_p = a1_coeff * b1

    # Denominator (Eq. 2.7)
    # MESA: low-T uses 9.383e-1*xlm1 - 4.141e-1*xlm2 + 5.829e-2*xlm3
    #        high-T uses 1.2383*xlm1 - 8.141e-1*xlm2
    a1_lo = 9.383e-1 * xlm1 - 4.141e-1 * xlm2 + 5.829e-2 * xlm3
    a1_hi = 1.2383 * xlm1 - 8.141e-1 * xlm2
    a1_den = jnp.where(t9 < 10.0, a1_lo, a1_hi)
    xden_p = zeta3 + a1_den
    fpair = xnum_p / jnp.maximum(xden_p, 1e-30)

    # qpair (Eq. 2.6)
    a1_q = 10.7480 * xl2 + 0.3967 * xlp5 + 1.005
    xnum_q = 1.0 / jnp.maximum(a1_q, 1e-30)
    a1_qd = 7.692e7 * xl3 + 9.715e6 * xlp5
    b1_q = 1.0 + rm / jnp.maximum(a1_qd, 1e-30)
    xden_q = jnp.power(b1_q, -0.3)
    qpair = xnum_q * xden_q

    # spair (Eq. 2.5): exp(-2/xl) * fpair * gl * tfac4*(1 + tfac3*qpair)
    exp_2xl = jnp.exp(-2.0 * xlm1)
    spair = exp_2xl * fpair * gl * _TFAC4 * (1.0 + _TFAC3 * qpair)

    return jnp.maximum(spair / rho, 0.0)
