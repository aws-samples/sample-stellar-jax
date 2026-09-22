"""Acoustic glitch signatures — He II ionization zone and base of convection zone.

Computes:
  1. Acoustic depth τ(r) = ∫_r^R dr'/c(r') and acoustic radius T = τ(0)
  2. Second frequency differences Δ₂ν_{n,l} = ν_{n-1,l} - 2ν_{n,l} + ν_{n+1,l}
  3. Houdek & Gough (2007) glitch fitting: extracts τ_He, A_He, Δ_He, τ_cz, A_cz

Method:
  - Acoustic depth: cumulative trapezoidal integration of dr/c from surface inward,
    matching MESA's profile_getval.f90:672 (sum of dr_div_csound from surface to point k).
  - Sound speed: c = sqrt(Γ₁ · P / ρ), matching MESA's star_utils.f90:370 (eval_csound).
  - Fitting: Houdek & Gough (2007), MNRAS 375, 861, Eq. 24 (D₀ diagnostic, simplified form
    without acoustic-cutoff correction for robustness on real data).

References:
    Houdek & Gough (2007), MNRAS 375, 861 (asteroseismic signature of He ionization)
    Verma et al. (2014), ApJ 794, 114 (theoretical study of acoustic glitches)
    Verma et al. (2017), ApJ 837, 134 (Kepler LEGACY glitch analysis)
    Christensen-Dalsgaard (2008), Ap&SS 316, 113 (FGONG format)
    MESA star_utils.f90:370-384 (eval_csound = sqrt(gamma1 * P / rho))
    MESA profile_getval.f90:672-677 (acoustic_depth = sum(dr_div_csound(1:k-1)))
"""
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares


def compute_acoustic_depth_profile(glob, var):
    """Compute acoustic depth τ(r) from an FGONG structure.

    Acoustic depth at point k = ∫ from surface (k=N-1) inward to point k of dr/c.
    This matches MESA profile_getval.f90:675: acoustic_depth(k) = sum(dr_div_csound(1:k-1))
    where MESA indexes surface-to-center (we index center-to-surface, so we reverse).

    Args:
        glob: FGONG global parameters (numpy array, shape >= 15)
        var: FGONG per-point variables (numpy array, shape (N, >=15)), center-to-surface

    Returns:
        dict with:
            'r': radius array (center-to-surface), cm
            'csound': adiabatic sound speed c(r) = sqrt(Γ₁·P/ρ), cm/s
            'tau': acoustic depth τ(r) from surface, seconds (center-to-surface)
            'acoustic_radius': T = total acoustic radius = τ(r=0), seconds
            'x': fractional radius r/R
    """
    r = var[:, 0]       # radius, cm
    P = var[:, 3]       # pressure, dyn/cm²
    rho = var[:, 4]     # density, g/cm³
    gamma1 = var[:, 9]  # first adiabatic exponent Γ₁

    R = glob[1]  # stellar radius

    # Sound speed: c = sqrt(Γ₁ · P / ρ)  [MESA star_utils.f90:370]
    cs2 = gamma1 * P / np.maximum(rho, 1e-30)
    csound = np.sqrt(np.maximum(cs2, 0.0))

    # Acoustic depth: τ(r) = ∫_r^R dr'/c(r')
    # Integrate from surface inward using trapezoidal rule on the radial grid.
    # var is center-to-surface, so r is increasing. We compute cumulative integral
    # from the surface (index N-1) backwards.
    N = len(r)
    dr = np.diff(r)  # dr[i] = r[i+1] - r[i], length N-1

    # dr_div_csound for each interval [i, i+1]: use midpoint sound speed
    cs_mid = 0.5 * (csound[:-1] + csound[1:])
    dt = dr / np.maximum(cs_mid, 1.0)  # time to cross each shell, length N-1

    # Acoustic depth at each point: sum from surface inward
    # tau[N-1] = 0 (surface), tau[k] = sum of dt[k:N-1]
    tau = np.zeros(N)
    tau_cumulative = np.cumsum(dt[::-1])[::-1]  # cumsum from surface
    tau[:-1] = tau_cumulative
    # tau[-1] = 0 (surface)

    acoustic_radius = tau[0]  # total acoustic radius T

    return {
        'r': r,
        'csound': csound,
        'tau': tau,
        'acoustic_radius': acoustic_radius,
        'x': r / R,
    }


def compute_acoustic_depth_jax(r, P, rho, gamma1, R):
    """JAX-differentiable acoustic depth computation.

    Same physics as compute_acoustic_depth_profile but operates on JAX arrays
    for end-to-end differentiability.

    Args:
        r: radius array (center-to-surface), shape (N,)
        P: pressure array, shape (N,)
        rho: density array, shape (N,)
        gamma1: Γ₁ array, shape (N,)
        R: stellar radius (scalar)

    Returns:
        dict with JAX arrays: 'csound', 'tau', 'acoustic_radius'
    """
    # Sound speed: c = sqrt(Γ₁ · P / ρ)
    cs2 = gamma1 * P / jnp.maximum(rho, 1e-30)
    csound = jnp.sqrt(jnp.maximum(cs2, 0.0))

    # Shell widths and midpoint sound speeds
    dr = jnp.diff(r)
    cs_mid = 0.5 * (csound[:-1] + csound[1:])
    dt = dr / jnp.maximum(cs_mid, 1.0)

    # Acoustic depth: cumulative sum from surface inward
    tau_inner = jnp.cumsum(dt[::-1])[::-1]
    tau = jnp.concatenate([tau_inner, jnp.zeros(1)])

    return {
        'csound': csound,
        'tau': tau,
        'acoustic_radius': tau[0],
    }


def second_differences(freqs):
    """Compute second frequency differences Δ₂ν_n = ν_{n-1} - 2ν_n + ν_{n+1}.

    The second difference is the standard diagnostic for acoustic glitches
    (Gough 1990; Houdek & Gough 2007, Eq. 2).

    Args:
        freqs: 1D array of frequencies ordered by radial order n (μHz)

    Returns:
        dict with:
            'nu': central frequencies ν_n (μHz), length N-2
            'delta2': second differences Δ₂ν (μHz), length N-2
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    if len(freqs) < 3:
        raise ValueError("Need at least 3 consecutive frequencies for second differences")

    delta2 = freqs[:-2] - 2.0 * freqs[1:-1] + freqs[2:]
    nu_central = freqs[1:-1]

    return {'nu': nu_central, 'delta2': delta2}


def _glitch_model_D0(nu, params):
    """Houdek & Gough (2007) D₀ diagnostic (Eq. 24, simplified form).

    Model for second differences (all in μHz):
        Δ₂ν ≈ A_He · ν · exp(-8π²Δ²_He·ν²_Hz) · cos(4πτ_He·ν_Hz + φ_He)
             + A_cz / ν² · cos(4πτ_cz·ν_Hz + φ_cz)
             + a₀ + a₁/ν + a₂/ν²

    where ν_Hz = ν × 1e-6 (convert μHz to Hz for phase/damping terms).
    A_He has units such that A_He × ν(μHz) × exp(...) gives μHz (dimensionless-ish).
    A_cz has units μHz³ (so that A_cz/ν² gives μHz).

    This is the simplified form without the second-difference amplitude corrections
    F_II, δ_II (those are ~1.5-3.4× and vary weakly; absorbed into fitted A).

    Args:
        nu: frequencies in μHz (1D array)
        params: dict with keys:
            'A_He': He II glitch amplitude (dimensionless)
            'Delta_He': He II Gaussian width (s)
            'tau_He': acoustic depth of He II zone (s)
            'phi_He': He II phase (rad)
            'A_cz': BCZ glitch amplitude (μHz³)
            'tau_cz': acoustic depth of BCZ (s)
            'phi_cz': BCZ phase (rad)
            'a0', 'a1', 'a2': smooth polynomial coefficients (μHz, μHz², μHz³)

    Returns:
        model Δ₂ν values (μHz)
    """
    nu_Hz = nu * 1e-6  # convert μHz to Hz for phase/damping terms

    # He II ionization zone component (Houdek & Gough 2007, Eq. 15/24)
    # Oscillatory signal: amplitude decays as Gaussian in ν
    A_He = params['A_He']
    Delta_He = params['Delta_He']
    tau_He = params['tau_He']
    phi_He = params['phi_He']
    he_term = (A_He * nu *
               np.exp(-8.0 * np.pi**2 * Delta_He**2 * nu_Hz**2) *
               np.cos(4.0 * np.pi * tau_He * nu_Hz + phi_He))

    # Base of convection zone component (Houdek & Gough 2007, Eq. 17-18)
    # BCZ signal decays as 1/ν² (algebraic, not exponential)
    A_cz = params['A_cz']
    tau_cz = params['tau_cz']
    phi_cz = params['phi_cz']
    cz_term = (A_cz / nu**2 *
               np.cos(4.0 * np.pi * tau_cz * nu_Hz + phi_cz))

    # Smooth polynomial background
    smooth = params['a0'] + params['a1'] / nu + params['a2'] / nu**2

    return he_term + cz_term + smooth


def fit_glitch_signatures(freqs_dict, delta_nu=None, nu_min=1500.0, nu_max=4000.0):
    """Fit acoustic glitch signatures from oscillation frequencies.

    Computes second differences from l=0 (and optionally l=1,2) frequencies,
    then fits the Houdek & Gough (2007) D₀ model to extract He II and BCZ
    glitch parameters.

    Args:
        freqs_dict: dict {l: array of frequencies in μHz}, ordered by radial order.
                    At minimum, l=0 must be present with ≥5 modes.
        delta_nu: large frequency separation (μHz). If None, estimated from l=0.
        nu_min: minimum frequency for fitting (μHz)
        nu_max: maximum frequency for fitting (μHz)

    Returns:
        dict with:
            'tau_He': acoustic depth of He II zone (s)
            'A_He': He II amplitude (μHz)
            'Delta_He': He II Gaussian width (s)
            'phi_He': He II phase (rad)
            'tau_cz': acoustic depth of BCZ (s)
            'A_cz': BCZ amplitude (μHz⁴)
            'phi_cz': BCZ phase (rad)
            'delta_nu': large separation used (μHz)
            'smooth_coeffs': (a0, a1, a2)
            'residual_rms': RMS of fit residuals (μHz)
            'nu_data': frequencies used in fit
            'delta2_data': second differences used in fit
            'delta2_model': model values at nu_data
    """
    # Compute second differences for available degrees
    all_nu = []
    all_d2 = []

    for l_deg in sorted(freqs_dict.keys()):
        freqs = np.sort(np.asarray(freqs_dict[l_deg], dtype=np.float64))
        if len(freqs) < 3:
            continue
        sd = second_differences(freqs)
        all_nu.append(sd['nu'])
        all_d2.append(sd['delta2'])

    if not all_nu:
        raise ValueError("No degree with ≥3 frequencies found")

    nu_all = np.concatenate(all_nu)
    d2_all = np.concatenate(all_d2)

    # Apply frequency window
    mask = (nu_all >= nu_min) & (nu_all <= nu_max)
    if mask.sum() < 6:
        raise ValueError(f"Only {mask.sum()} points in [{nu_min}, {nu_max}] μHz; need ≥6")
    nu_fit = nu_all[mask]
    d2_fit = d2_all[mask]

    # Sort by frequency
    order = np.argsort(nu_fit)
    nu_fit = nu_fit[order]
    d2_fit = d2_fit[order]

    # Estimate delta_nu if not provided
    if delta_nu is None:
        if 0 in freqs_dict and len(freqs_dict[0]) >= 3:
            delta_nu = float(np.median(np.diff(np.sort(freqs_dict[0]))))
        else:
            delta_nu = 135.0  # solar-ish default

    # Initial parameter guesses (solar-like)
    # τ_He ~ 700-900 s, τ_cz ~ 2000-2500 s for solar-like stars
    acoustic_radius_est = 1.0 / (2.0 * delta_nu * 1e-6)  # T ≈ 1/(2Δν) in seconds
    tau_He_init = 0.22 * acoustic_radius_est   # ~22% of T for He II (solar ~800/3600)
    tau_cz_init = 0.63 * acoustic_radius_est   # ~63% of T for BCZ (solar ~2270/3600)

    # Amplitude scaling: at ν=2500 μHz with Δ=70s, the He term gives
    # A_He × 2500 × exp(-8π² × 70² × (2.5e-3)²) = A_He × 2500 × 0.10
    # For ~1 μHz second-diff signal: A_He ~ 0.004
    A_He_init = 0.004

    # BCZ: A_cz / ν² at ν=2500 gives A_cz / 6.25e6. For ~0.3 μHz: A_cz ~ 2e6
    A_cz_init = 2.0e6

    # Pack into array for optimizer: [A_He, Delta_He, tau_He, phi_He,
    #                                  A_cz, tau_cz, phi_cz, a0, a1, a2]
    x0 = np.array([
        A_He_init, 70.0, tau_He_init, 0.0,
        A_cz_init, tau_cz_init, 0.0,
        0.0, 0.0, 0.0  # smooth polynomial
    ])

    def _residual(x):
        params = {
            'A_He': x[0], 'Delta_He': x[1], 'tau_He': x[2], 'phi_He': x[3],
            'A_cz': x[4], 'tau_cz': x[5], 'phi_cz': x[6],
            'a0': x[7], 'a1': x[8], 'a2': x[9],
        }
        model = _glitch_model_D0(nu_fit, params)
        return model - d2_fit

    # Bounds to keep parameters physical
    bounds_lower = [0.0,    10.0,  200.0,  -2*np.pi, -1e9,   500.0,  -2*np.pi, -10, -1e7, -1e13]
    bounds_upper = [0.1,   300.0, 1500.0,   2*np.pi,  1e9,  4000.0,   2*np.pi,  10,  1e7,  1e13]

    result = least_squares(_residual, x0, bounds=(bounds_lower, bounds_upper),
                           method='trf', max_nfev=5000, ftol=1e-12, xtol=1e-12)

    xopt = result.x
    params_opt = {
        'A_He': xopt[0], 'Delta_He': xopt[1], 'tau_He': xopt[2], 'phi_He': xopt[3],
        'A_cz': xopt[4], 'tau_cz': xopt[5], 'phi_cz': xopt[6],
        'a0': xopt[7], 'a1': xopt[8], 'a2': xopt[9],
    }
    model_values = _glitch_model_D0(nu_fit, params_opt)
    residual_rms = float(np.sqrt(np.mean((model_values - d2_fit)**2)))

    return {
        'tau_He': float(xopt[2]),
        'A_He': float(xopt[0]),
        'Delta_He': float(xopt[1]),
        'phi_He': float(xopt[3]),
        'tau_cz': float(xopt[5]),
        'A_cz': float(xopt[4]),
        'phi_cz': float(xopt[6]),
        'delta_nu': delta_nu,
        'smooth_coeffs': (float(xopt[7]), float(xopt[8]), float(xopt[9])),
        'residual_rms': residual_rms,
        'nu_data': nu_fit,
        'delta2_data': d2_fit,
        'delta2_model': model_values,
    }
