"""Seismic quantities: n_pg-identified ratios, separations, ν_max, and ε.

First-class API for surface-independent asteroseismic diagnostics. All functions
operate on n_pg-identified modes (radial order), not nearest-frequency matching.

Public API (n_pg-identified diagnostics):
  - assign_n_pg: assign radial order to p-mode frequencies via the asymptotic relation
  - delta_nu_02, delta_nu_01: small frequency separations δν₀₂(n), δν₀₁(n)
  - r02, r01, r10: frequency ratios (Roxburgh & Vorontsov 2003)
  - r02_differentiable: differentiable r₀₂ (IFT adjoint, #1150)
  - r01_differentiable: differentiable r₀₁ (IFT adjoint, #1162)
  - large_separation: Δν from the acoustic radius integral (MESA report.f90:343)
  - nu_max_scaling: ν_max from the Brown-Kjeldsen-Bedding scaling relation
  - epsilon_fit: asymptotic phase ε from a linear fit to the Tassoul relation

Legacy/utility functions (echelle_data, estimate_delta_nu, compute_solar_model_freqs,
compute_evolved_solar_model_freqs, compare_echelle) moved to diagnostic.py (#1146).

References:
    Roxburgh & Vorontsov (2003), A&A 411, 215 (frequency ratios)
    Tassoul (1980), ApJS 43, 469 (asymptotic theory, ε)
    Brown et al. (1991), ApJ 368, 599 (ν_max scaling)
    Kjeldsen & Bedding (1995), A&A 293, 87 (ν_max scaling)
    Christensen-Dalsgaard (2008), Ap&SS 316, 113 (ADIPLS, FGONG)
    Christensen-Dalsgaard et al. (1996), Science 272, 1286 (Model S)
    MESA report.f90:343 — delta_nu = 1e6/(2*photosphere_acoustic_r)
    MESA report.f90:353 — nu_max = nu_max_sun*M/(R^2*sqrt(Teff/Teff_sun))
"""
import numpy as np


# ════════════════════════════════════════════════════════════════════════════════
# n_pg identification
# ════════════════════════════════════════════════════════════════════════════════

def assign_n_pg(freqs_l, delta_nu, epsilon, l):
    """Assign radial order n_pg to p-mode frequencies using the Tassoul relation.

    For pure p-modes in the asymptotic regime:
        ν_{n,l} ≈ Δν·(n + l/2 + ε)   [Tassoul 1980, ApJS 43, 469, Eq. 66]

    Solves for n = round(ν/Δν − l/2 − ε) for each frequency.

    Args:
        freqs_l: 1D array of sorted frequencies (µHz) for a single l
        delta_nu: large separation estimate (µHz)
        epsilon: asymptotic phase (dimensionless, typically 1.0–1.6 for solar-type)
        l: angular degree

    Returns:
        dict: {n_pg: freq_uHz} mapping radial order → frequency.
              Only returns n_pg ≥ 1 (excludes f-mode / g-modes).
    """
    freqs_l = np.asarray(freqs_l)
    if len(freqs_l) == 0 or delta_nu <= 0:
        return {}
    n_float = freqs_l / delta_nu - l / 2.0 - epsilon
    n_int = np.rint(n_float).astype(int)
    result = {}
    for i, n in enumerate(n_int):
        if n >= 1 and n not in result:  # first (closest) wins for duplicates
            result[int(n)] = float(freqs_l[i])
    return result


def _estimate_epsilon_from_l0(freqs_l0, delta_nu):
    """Estimate ε from l=0 frequencies via the asymptotic relation.

    For l=0: ν_{n,0} ≈ Δν·(n + ε), so ε = (ν/Δν) mod 1 gives the fractional
    shift. However, ε is conventionally in [0, 2) — for solar-type stars it is
    typically 1.0–1.6, but for hotter stars (convective cores) it can be 0.5–1.0.

    We estimate by taking the median fractional part of ν/Δν. The fractional
    part gives ε mod 1; we then decide whether the true ε is frac or frac + 1
    based on where the frac falls relative to 0.5.

    Physical basis (White et al. 2012, ApJ 751, 36, Fig. 5):
      - Cool MS stars (Teff ≲ 6000 K, solar-type): ε ≈ 1.0–1.5 → frac < 0.5
      - Hot MS stars (Teff ≳ 6200 K, F-type): ε ≈ 0.5–1.0 → frac ≥ 0.5
    The threshold at 0.5 is the natural midpoint: frac < 0.5 → true ε is
    frac + 1 (cooler stars); frac ≥ 0.5 → true ε is frac (hotter stars).

    Ref: Christensen-Dalsgaard, Lecture Notes on Stellar Oscillations, §5.1
         White et al. (2012), ApJ 751, 36 (ε–Teff relation)
    """
    freqs_l0 = np.asarray(freqs_l0)
    if len(freqs_l0) < 2 or delta_nu <= 0:
        return 1.4  # reasonable default for solar-type stars
    # The fractional part of ν/Δν for l=0 modes clusters around ε mod 1
    frac = (freqs_l0 / delta_nu) % 1.0
    eps_raw = float(np.median(frac))
    # ε is typically in [0.5, 1.7]. The fractional part gives ε mod 1.
    # If frac < 0.5, the true ε is frac + 1 (e.g., solar ε≈1.2 → frac≈0.2)
    # If frac ≥ 0.5, the true ε is frac itself (e.g., hot star ε≈0.7 → frac=0.7)
    # Threshold = 0.5 (the natural boundary between ε<1 and ε>1 regime).
    # Verified: 100% n_pg agreement with GYRE 8.1 across all 7 committed models
    # (1.0Msun ZAMS/midMS/Xc0.30, 1.2Msun ZAMS, 1.5Msun ZAMS, 2.0Msun ZAMS/Xc0.40).
    if eps_raw < 0.5:
        return eps_raw + 1.0
    return eps_raw


def identify_modes(freqs_dict):
    """Assign n_pg to all modes in a frequency dict using the asymptotic relation.

    This is the primary n_pg identification entry point. It estimates Δν from l=0
    spacings, then bootstraps ε, and assigns n_pg to all l values.

    Args:
        freqs_dict: dict {l: array of frequencies in µHz} (sorted ascending per l)

    Returns:
        dict: {l: {n_pg: freq_uHz}} — identified modes by (l, n_pg).

    Raises:
        ValueError: if l=0 modes are insufficient for identification.
    """
    if 0 not in freqs_dict or len(freqs_dict[0]) < 3:
        raise ValueError(
            "identify_modes requires ≥3 l=0 modes for Δν/ε estimation")
    freqs_l0 = np.sort(freqs_dict[0])
    delta_nu = float(np.median(np.diff(freqs_l0)))
    epsilon = _estimate_epsilon_from_l0(freqs_l0, delta_nu)
    result = {}
    for l, freqs in freqs_dict.items():
        result[l] = assign_n_pg(np.sort(freqs), delta_nu, epsilon, l)
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Small frequency separations
# ════════════════════════════════════════════════════════════════════════════════

def delta_nu_02(identified_modes):
    """Compute the small separation δν₀₂(n) = ν(n,0) − ν(n−1,2).

    Definition: Tassoul (1980), ApJS 43, Eq. 69. Sensitive to the sound-speed
    gradient near the core — the primary age diagnostic.

    Args:
        identified_modes: dict {l: {n_pg: freq_uHz}} from identify_modes()

    Returns:
        dict: {n: δν₀₂} for all valid n where both ν(n,0) and ν(n−1,2) exist.
    """
    if 0 not in identified_modes or 2 not in identified_modes:
        return {}
    freq_l0 = identified_modes[0]
    freq_l2 = identified_modes[2]
    result = {}
    for n, f0 in freq_l0.items():
        if (n - 1) in freq_l2:
            d02 = f0 - freq_l2[n - 1]
            if d02 > 0:
                result[n] = d02
    return result


def delta_nu_01(identified_modes):
    """Compute the small separation δν₀₁(n) = ν(n,0) − [ν(n−1,1) + ν(n,1)]/2.

    Definition: half the distance between l=0 and the midpoint of adjacent l=1
    modes. Also written as δν₀₁(n) = (1/8)[ν(n−1,0) − 4ν(n−1,1) + 6ν(n,0)
    − 4ν(n,1) + ν(n+1,0)] in the 5-point form (Roxburgh & Vorontsov 2003,
    Eq. 3), but the 3-point form above is standard.

    Args:
        identified_modes: dict {l: {n_pg: freq_uHz}} from identify_modes()

    Returns:
        dict: {n: δν₀₁} for all valid n.
    """
    if 0 not in identified_modes or 1 not in identified_modes:
        return {}
    freq_l0 = identified_modes[0]
    freq_l1 = identified_modes[1]
    result = {}
    for n, f0 in freq_l0.items():
        if (n - 1) in freq_l1 and n in freq_l1:
            d01 = f0 - 0.5 * (freq_l1[n - 1] + freq_l1[n])
            result[n] = d01
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Frequency ratios (Roxburgh & Vorontsov 2003, A&A 411, 215)
# ════════════════════════════════════════════════════════════════════════════════

def r02(identified_modes):
    """Compute the frequency ratio r₀₂(n) = δν₀₂(n) / Δν₁(n).

    Definition: Roxburgh & Vorontsov (2003), A&A 411, 215, Eq. 5.
        r₀₂(n) = [ν(n,0) − ν(n−1,2)] / [ν(n,1) − ν(n−1,1)]

    The ratio is surface-independent to first order — the dominant diagnostic
    for core conditions without needing a surface-term model.

    Args:
        identified_modes: dict {l: {n_pg: freq_uHz}} from identify_modes()

    Returns:
        dict: {n: r₀₂} for all valid n.
    """
    if not all(l in identified_modes for l in (0, 1, 2)):
        return {}
    freq_l0 = identified_modes[0]
    freq_l1 = identified_modes[1]
    freq_l2 = identified_modes[2]
    result = {}
    for n, f0 in freq_l0.items():
        if n < 2:
            continue
        if (n - 1) not in freq_l2:
            continue
        if n not in freq_l1 or (n - 1) not in freq_l1:
            continue
        d02 = f0 - freq_l2[n - 1]
        dnu1 = freq_l1[n] - freq_l1[n - 1]
        if dnu1 > 0 and d02 > 0:
            result[n] = d02 / dnu1
    return result


def r01(identified_modes):
    """Compute the frequency ratio r₀₁(n) = δν₀₁(n) / Δν₁(n).

    Definition: Roxburgh & Vorontsov (2003), A&A 411, 215, Eq. 4.
        r₀₁(n) = [ν(n,0) − (ν(n−1,1) + ν(n,1))/2] / [ν(n,1) − ν(n−1,1)]

    Surface-independent ratio probing l=0/l=1 interior differences.

    Args:
        identified_modes: dict {l: {n_pg: freq_uHz}} from identify_modes()

    Returns:
        dict: {n: r₀₁} for all valid n.
    """
    if not all(l in identified_modes for l in (0, 1)):
        return {}
    freq_l0 = identified_modes[0]
    freq_l1 = identified_modes[1]
    result = {}
    for n, f0 in freq_l0.items():
        if n < 2:
            continue
        if n not in freq_l1 or (n - 1) not in freq_l1:
            continue
        d01 = f0 - 0.5 * (freq_l1[n - 1] + freq_l1[n])
        dnu1 = freq_l1[n] - freq_l1[n - 1]
        if dnu1 > 0:
            result[n] = d01 / dnu1
    return result


def r10(identified_modes):
    """Compute the frequency ratio r₁₀(n) = δν₁₀(n) / Δν₀(n+1).

    Definition: Roxburgh & Vorontsov (2003), A&A 411, 215, Eq. 6.
        r₁₀(n) = −[ν(n,1) − (ν(n,0) + ν(n+1,0))/2] / [ν(n+1,0) − ν(n,0)]

    The l=1 analogue of r₀₁, using the l=0 large separation as denominator.

    Args:
        identified_modes: dict {l: {n_pg: freq_uHz}} from identify_modes()

    Returns:
        dict: {n: r₁₀} for all valid n.
    """
    if not all(l in identified_modes for l in (0, 1)):
        return {}
    freq_l0 = identified_modes[0]
    freq_l1 = identified_modes[1]
    result = {}
    for n, f1 in freq_l1.items():
        if n not in freq_l0 or (n + 1) not in freq_l0:
            continue
        d10 = -(f1 - 0.5 * (freq_l0[n] + freq_l0[n + 1]))
        dnu0 = freq_l0[n + 1] - freq_l0[n]
        if dnu0 > 0:
            result[n] = d10 / dnu0
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Large separation and ν_max
# ════════════════════════════════════════════════════════════════════════════════

def large_separation(glob, var):
    """Compute Δν from the acoustic radius integral.

    MESA report.f90:343:
        delta_nu = 1d6 / (2 * photosphere_acoustic_r)  [µHz]
    where acoustic_r = ∫₀ᴿ dr/c_s  [seconds].

    Computes the sound speed from the FGONG structure:
        c_s² = Γ₁·P/ρ   (adiabatic sound speed)

    Args:
        glob: FGONG global parameters (glob[1] = R in cm)
        var: FGONG per-point variables (center-to-surface ordering)
            var[:,0] = r, var[:,3] = P, var[:,4] = ρ, var[:,9] = Γ₁

    Returns:
        float: Δν in µHz
    """
    r = var[:, 0]
    P = var[:, 3]
    rho = var[:, 4]
    gamma1 = var[:, 9]

    # Sound speed
    c_s = np.sqrt(gamma1 * P / rho)

    # Acoustic radius via trapezoidal integration: τ = ∫ dr/c_s
    dr = np.diff(r)
    inv_cs = 1.0 / c_s
    avg_inv_cs = 0.5 * (inv_cs[:-1] + inv_cs[1:])
    acoustic_radius = float(np.sum(avg_inv_cs * dr))  # seconds

    if acoustic_radius <= 0:
        raise ValueError("acoustic_radius ≤ 0: invalid FGONG structure")

    # MESA report.f90:343
    return 1.0e6 / (2.0 * acoustic_radius)


def nu_max_scaling(glob, var):
    """Compute ν_max from the Brown-Kjeldsen-Bedding scaling relation.

    MESA report.f90:353:
        nu_max = nu_max_sun * M / (R^2 * sqrt(Teff / Teff_sun))

    where Teff is derived from L = 4πR²σT⁴.

    Solar reference values (MESA pgstar_mode_prop.f90:153):
        ν_max,☉ = 3100 µHz, T_eff,☉ = 5777 K

    Args:
        glob: FGONG global parameters
            glob[0] = M (g), glob[1] = R (cm), glob[2] = L (erg/s)

    Returns:
        float: ν_max in µHz

    References:
        Brown et al. (1991), ApJ 368, 599
        Kjeldsen & Bedding (1995), A&A 293, 87
        MESA report.f90:353; pgstar_mode_prop.f90:153-164
    """
    from stellar_jax.config.constants import Msun, Rsun, sigma_sb

    # Solar reference values (MESA pgstar_mode_prop.f90:153)
    NU_MAX_SUN = 3100.0   # µHz
    TEFF_SUN = 5777.0     # K

    M = glob[0]
    R = glob[1]
    L = glob[2]

    T_eff = (L / (4.0 * np.pi * R**2 * sigma_sb))**0.25
    M_ratio = M / Msun
    R_ratio = R / Rsun

    return float(NU_MAX_SUN * M_ratio / (R_ratio**2 * np.sqrt(T_eff / TEFF_SUN)))


# ════════════════════════════════════════════════════════════════════════════════
# Asymptotic phase ε
# ════════════════════════════════════════════════════════════════════════════════

def epsilon_fit(freqs_dict_or_identified, delta_nu=None):
    """Compute the asymptotic phase ε from a fit to the Tassoul relation.

    Tassoul (1980), ApJS 43, 469, Eq. 66:
        ν_{n,0} ≈ Δν·(n + ε)

    Fits ε = median(ν_{n,0}/Δν mod 1) with correction for ε > 1.

    If a delta_nu is provided, uses it; otherwise estimates from l=0 spacings.

    Args:
        freqs_dict_or_identified: either:
            - dict {l: array} (raw freqs, will estimate Δν internally)
            - dict {l: {n_pg: freq}} (identified modes)
        delta_nu: Δν in µHz (optional; if None, estimated from l=0 spacings)

    Returns:
        float: asymptotic phase ε (typically 1.0–1.6 for solar-type stars)

    References:
        Tassoul (1980), ApJS 43, 469
        Christensen-Dalsgaard, Lecture Notes on Stellar Oscillations, §5.1
    """
    # Extract l=0 frequencies
    if 0 not in freqs_dict_or_identified:
        raise ValueError("epsilon_fit requires l=0 modes")

    l0_data = freqs_dict_or_identified[0]
    if isinstance(l0_data, dict):
        # identified modes: {n_pg: freq}
        freqs_l0 = np.array(sorted(l0_data.values()))
    else:
        freqs_l0 = np.sort(np.asarray(l0_data))

    if len(freqs_l0) < 3:
        raise ValueError("epsilon_fit requires ≥3 l=0 modes")

    if delta_nu is None:
        delta_nu = float(np.median(np.diff(freqs_l0)))

    return _estimate_epsilon_from_l0(freqs_l0, delta_nu)


# ════════════════════════════════════════════════════════════════════════════════
# Differentiable frequency ratios
#
# The non-differentiable r02/r01/r10 above operate on pre-identified modes
# (Python dicts, np.rint). The functions below route differentiable frequencies
# through the same ratio arithmetic, keeping mode labels (n_pg, l) static.
#
# Architecture:
#   1. Mode identification: compute_oscillation_freqs_jax → identify_modes()
#      → (n, l) pairs.  This is NumPy/brentq — non-differentiable, and
#      intentionally so: the integer labels are discrete.
#   2. For each needed eigenfrequency, call eigenfreq_from_coeffs (custom_vjp
#      IFT adjoint) on the SAME live coefficients.  The coefficients flow from
#      build_oscillation_coeffs_jax(glob_jax, var_jax) — differentiable.
#   3. Compute the ratio in JAX: r₀₂(n) = (ν(n,0) − ν(n−1,2)) / (ν(n,1) − ν(n−1,1)).
#
# The result: ∂r₀₂/∂θ is computable via jax.grad for any θ that flows through
# the stellar structure (M, Z, α_mlt, opacity_factor, ...).
#
# References:
#   Roxburgh & Vorontsov (2003), A&A 411, 215 (frequency ratios)
#   Christensen-Dalsgaard (2008), Ap&SS 316, 113 (IFT for eigenvalues)
# ════════════════════════════════════════════════════════════════════════════════


def _find_valid_r02_orders(identified):
    """Find radial orders with valid r₀₂ triples from identified modes.

    A valid r₀₂ triple at order n requires: ν(n,0), ν(n−1,2), ν(n,1), ν(n−1,1),
    with δν₀₂ > 0 and Δν₁ > 0.

    Args:
        identified: dict {l: {n_pg: freq_uHz}} from identify_modes()

    Returns:
        list of valid radial orders (sorted ascending).
    """
    if not all(l in identified for l in (0, 1, 2)):
        return []
    freq_l0, freq_l1, freq_l2 = identified[0], identified[1], identified[2]
    valid_n = []
    for n in sorted(freq_l0.keys()):
        if n < 2:
            continue
        if (n - 1) not in freq_l2:
            continue
        if n not in freq_l1 or (n - 1) not in freq_l1:
            continue
        d02 = freq_l0[n] - freq_l2[n - 1]
        dnu1 = freq_l1[n] - freq_l1[n - 1]
        if dnu1 > 0 and d02 > 0:
            valid_n.append(n)
    return valid_n


def _find_valid_r01_orders(identified):
    """Find radial orders with valid r₀₁ quintuples from identified modes.

    The 5-point r₀₁ at order n requires: ν(n−1,0), ν(n,0), ν(n+1,0),
    ν(n−1,1), ν(n,1), with Δν₁ = ν(n,1) − ν(n−1,1) > 0.
    Matches MESA astero_support.f90:396 (5-point d01 formula).

    Args:
        identified: dict {l: {n_pg: freq_uHz}} from identify_modes()

    Returns:
        list of valid radial orders (sorted ascending).
    """
    if not all(l in identified for l in (0, 1)):
        return []
    freq_l0, freq_l1 = identified[0], identified[1]
    valid_n = []
    for n in sorted(freq_l0.keys()):
        if n < 2:
            continue
        # 5-point formula needs l=0 at n-1, n, n+1 and l=1 at n-1, n
        if (n - 1) not in freq_l0 or (n + 1) not in freq_l0:
            continue
        if n not in freq_l1 or (n - 1) not in freq_l1:
            continue
        dnu1 = freq_l1[n] - freq_l1[n - 1]
        if dnu1 > 0:
            valid_n.append(n)
    return valid_n


def _find_eigenvalue_anchor(l, nu_approx, delta_nu_est,
                            coeffs_np, x_grid_np, x_steps, h_steps,
                            factor, nu_min, nu_max):
    """Find the eigenvalue σ² near nu_approx via brentq (non-differentiable).

    Uses concrete NumPy/JIT arrays for the root-finding. This is the
    non-differentiable anchor step: brentq finds the precise eigenvalue,
    which is then passed to eigenfreq_from_coeffs as a static anchor.

    Args:
        l: angular degree (0, 1, 2)
        nu_approx: approximate frequency from mode identification (µHz)
        delta_nu_est: estimated large separation for bracketing (µHz)
        coeffs_np: concrete (non-traced) coefficients (5, N) — NumPy or JAX array
        x_grid_np: concrete fractional radius grid (N,)
        x_steps, h_steps: integration grid (static)
        factor: ν²→σ² conversion factor (float)
        nu_min, nu_max: global frequency bounds (µHz)

    Returns:
        float: converged σ² (concrete, not traced)
    """
    import jax
    import jax.numpy as jnp
    from .eigenvalue import _bracket_and_refine
    from .determinant import radial_determinant, nonradial_determinant

    # Convert to JAX arrays if needed (for the JIT-compiled determinant)
    x_grid_jax = jnp.asarray(x_grid_np)
    coeffs_jax = jnp.asarray(coeffs_np)

    # Bracket: narrow window around the known approximate frequency
    bracket_half = 0.4 * delta_nu_est
    nu_lo = max(nu_approx - bracket_half, nu_min)
    nu_hi = min(nu_approx + bracket_half, nu_max)
    nu_arr_local = np.linspace(nu_lo, nu_hi, 50)

    if l == 0:
        @jax.jit
        def det_fn(s2):
            return radial_determinant(s2, x_steps, h_steps,
                                      x_grid_jax, coeffs_jax)
    else:
        l_float = float(l)
        @jax.jit
        def det_fn(s2):
            return nonradial_determinant(s2, jnp.float64(l_float),
                                         x_steps, h_steps,
                                         x_grid_jax, coeffs_jax)

    roots_nu = _bracket_and_refine(det_fn, nu_arr_local, factor,
                                   rtol=1e-10, maxiter=100)
    if not roots_nu:
        # Fall back to wider bracket
        nu_arr_wide = np.linspace(
            max(nu_approx - delta_nu_est, nu_min),
            min(nu_approx + delta_nu_est, nu_max), 100)
        roots_nu = _bracket_and_refine(det_fn, nu_arr_wide, factor,
                                       rtol=1e-10, maxiter=100)

    if not roots_nu:
        raise ValueError(
            f"_find_eigenvalue_anchor: cannot find mode l={l} "
            f"near ν≈{nu_approx:.1f} µHz")

    best_idx = int(np.argmin(np.abs(np.array(roots_nu) - nu_approx)))
    nu_converged = roots_nu[best_idx]
    return nu_converged**2 * factor


def _find_all_modes_from_coeffs(coeffs_np, x_grid_np, x_steps, h_steps,
                                factor, nu_min, nu_max, n_scan):
    """Find eigenfrequencies for l=0,1,2 using concrete oscillation coefficients.

    Uses the same coefficients as the differentiable path (build_oscillation_coeffs_jax),
    ensuring consistent mode identification between the non-differentiable mode-finding
    step and the IFT gradient evaluation.

    Args:
        coeffs_np: (5, N) numpy array — concrete oscillation coefficients
        x_grid_np: (N,) numpy array — fractional radius grid
        x_steps, h_steps: integration grid from _make_integration_grid
        factor: ν²→σ² conversion factor (float)
        nu_min, nu_max: frequency bounds for root search (µHz)
        n_scan: number of trial frequencies for bracket detection

    Returns:
        dict: {l: np.array of frequencies in µHz} for l=0,1,2
    """
    import jax
    import jax.numpy as jnp
    from .determinant import radial_determinant, nonradial_determinant
    from .eigenvalue import _bracket_and_refine

    x_grid_c = jnp.asarray(x_grid_np)
    coeffs_c = jnp.asarray(coeffs_np)

    all_freqs = {}
    for l in (0, 1, 2):
        nu_arr = np.linspace(nu_min, nu_max, n_scan)
        if l == 0:
            @jax.jit
            def _det_l0(s2):
                return radial_determinant(
                    s2, x_steps, h_steps, x_grid_c, coeffs_c)
            det_fn = _det_l0
        else:
            l_float = float(l)
            @jax.jit
            def _det_nl(s2, _l=jnp.float64(l_float)):
                return nonradial_determinant(
                    s2, _l, x_steps, h_steps, x_grid_c, coeffs_c)
            det_fn = _det_nl
        freqs = _bracket_and_refine(det_fn, nu_arr, factor,
                                    rtol=1e-8, maxiter=50)
        all_freqs[l] = np.array(freqs)
    return all_freqs


def r02_differentiable(glob_jax, var_jax, n_target=None,
                       nu_min=1500.0, nu_max=4500.0, n_scan=400, n_steps=8000):
    """Compute r₀₂(n) with gradients flowing through the structure.

    r₀₂(n) = [ν(n,0) − ν(n−1,2)] / [ν(n,1) − ν(n−1,1)]
    (Roxburgh & Vorontsov 2003, A&A 411, 215, Eq. 5)

    The mode labels (n, l) are determined by non-differentiable mode
    identification (np.rint via Tassoul); the frequencies at those labels
    are re-evaluated through the differentiable IFT eigenvalue solver.
    The gradient flows: structure → coefficients → eigenfreq_from_coeffs → ratio.

    Architecture (gradient-safe under jax.grad):
      1. Build oscillation coefficients from the TRACED glob_jax/var_jax via
         build_oscillation_coeffs_jax (includes center extension for truncated
         Henyey meshes). Extract concrete copies (stop_gradient) for mode
         identification and eigenvalue anchoring.
      2. Use the concrete coefficients to find all l=0,1,2 eigenfrequencies
         (bracket + Brent), identify modes (np.rint, Tassoul), and find σ²
         anchors for the 4 modes needed for r₀₂.
      3. For each needed mode, pass the concrete σ² anchor + TRACED coefficients
         to eigenfreq_from_coeffs (IFT custom_vjp). The gradient flows through
         the traced coefficients only, at fixed eigenvalue anchor — this is the
         standard IFT approach.

    Args:
        glob_jax: (15,) JAX array — FGONG global parameters
        var_jax: (N, 15) JAX array — per-point structure (center-to-surface)
        n_target: specific radial order to compute (int). If None, picks the
                  middle valid order.
        nu_min, nu_max: frequency bounds for mode search (µHz)
        n_scan: trial frequencies for root bracketing
        n_steps: RK4 integration steps

    Returns:
        JAX scalar: r₀₂ at the selected radial order (differentiable w.r.t.
                    glob_jax, var_jax through the traced computation graph).
    """
    import jax
    import jax.numpy as jnp
    from .adjoint import eigenfreq_from_coeffs
    from .coefficients import build_oscillation_coeffs_jax
    from .integrator import _make_integration_grid
    from .seismic_conversion import nu_from_sigma2

    # ── Step 1: Build coefficients and integration grid (concrete) ──
    grid_data = build_oscillation_coeffs_jax(glob_jax, var_jax)
    x_grid_jax = grid_data['x_grid']
    coeffs_jax = grid_data['coeffs']
    n_center = grid_data['n_center']
    # LIVE factor for the σ²→ν conversion — must NOT be stop_gradient'd so
    # that jax.grad captures ∂factor/∂θ (the ν²·∂factor/∂θ scaling term).
    # This is the fisher.py:241-242 pattern (/).
    live_factor = grid_data['factor']
    # Frozen float for eigenfreq_from_coeffs nondiff_argnums and mode-finding.
    factor = float(jax.lax.stop_gradient(live_factor))  # GP-2: MESH_POSITIONS (nondiff extraction)

    # Concrete copies for mode-finding (non-differentiable Python operations)
    x_grid_np = np.asarray(jax.lax.stop_gradient(x_grid_jax))  # GP-2: MESH_POSITIONS
    coeffs_np = np.asarray(jax.lax.stop_gradient(coeffs_jax))  # GP-2: MESH_POSITIONS

    # Integration grid from physical mesh points (skip synthetic center)
    x_phys = x_grid_np[n_center:]
    x_steps, h_steps = _make_integration_grid(
        x_phys[x_phys > 1e-4], n_steps=n_steps)

    # ── Step 1b: Non-differentiable mode identification ──
    # Find all eigenfrequencies for l=0,1,2 using the CONCRETE coefficients.
    # Mode identification is intentionally non-differentiable (integer n_pg via
    # np.rint) — we differentiate only the frequencies at fixed labels (Step 3).
    all_freqs = _find_all_modes_from_coeffs(
        coeffs_np, x_grid_np, x_steps, h_steps, factor,
        nu_min, nu_max, n_scan)

    freqs_dict = {l: all_freqs[l] for l in (0, 1, 2)}
    identified = identify_modes(freqs_dict)

    if not all(l in identified for l in (0, 1, 2)):
        raise ValueError("r02_differentiable: need l=0,1,2 modes; "
                         f"got l={list(identified.keys())}")

    valid_n = _find_valid_r02_orders(identified)
    if not valid_n:
        raise ValueError("r02_differentiable: no valid (n,l) triples found")

    if n_target is not None:
        if n_target not in valid_n:
            raise ValueError(
                f"r02_differentiable: n_target={n_target} not in valid orders {valid_n}")
        n_sel = n_target
    else:
        n_sel = valid_n[len(valid_n) // 2]

    # ── Step 2: Find eigenvalue anchors using CONCRETE coefficients ──
    # brentq root-finding requires concrete floats — it cannot trace through
    # JAX transformations. We use the concrete coefficients from Step 1 and
    # find σ² anchors for each of the 4 modes. The anchors are then passed
    # as static values to eigenfreq_from_coeffs (Step 3).
    delta_nu_est = float(np.median(np.diff(np.sort(all_freqs[0]))))
    freq_l0, freq_l1, freq_l2 = identified[0], identified[1], identified[2]
    anchor_common = dict(delta_nu_est=delta_nu_est, coeffs_np=coeffs_np,
                         x_grid_np=x_grid_np, x_steps=x_steps,
                         h_steps=h_steps, factor=factor,
                         nu_min=nu_min, nu_max=nu_max)

    s2_anchor_n0 = _find_eigenvalue_anchor(0, freq_l0[n_sel], **anchor_common)
    s2_anchor_nm1_2 = _find_eigenvalue_anchor(2, freq_l2[n_sel - 1],
                                               **anchor_common)
    s2_anchor_n1 = _find_eigenvalue_anchor(1, freq_l1[n_sel], **anchor_common)
    s2_anchor_nm1_1 = _find_eigenvalue_anchor(1, freq_l1[n_sel - 1],
                                               **anchor_common)

    # ── Step 3: Evaluate 4 frequencies differentiably via IFT, compute ratio ──
    # eigenfreq_from_coeffs has custom_vjp: the forward returns σ² at the
    # anchor, and the backward provides ∂σ²/∂coeffs via the implicit function
    # theorem. The gradient flows through coeffs_jax (traced) → ratio.
    s2_n0 = eigenfreq_from_coeffs(
        coeffs_jax, x_grid_jax, jnp.float64(s2_anchor_n0),
        0, x_steps, h_steps, factor)
    s2_nm1_2 = eigenfreq_from_coeffs(
        coeffs_jax, x_grid_jax, jnp.float64(s2_anchor_nm1_2),
        2, x_steps, h_steps, factor)
    s2_n1 = eigenfreq_from_coeffs(
        coeffs_jax, x_grid_jax, jnp.float64(s2_anchor_n1),
        1, x_steps, h_steps, factor)
    s2_nm1_1 = eigenfreq_from_coeffs(
        coeffs_jax, x_grid_jax, jnp.float64(s2_anchor_nm1_1),
        1, x_steps, h_steps, factor)

    # Convert σ² → ν using the LIVE factor so ∂factor/∂θ flows.
    nu_n0 = nu_from_sigma2(s2_n0, live_factor)
    nu_nm1_2 = nu_from_sigma2(s2_nm1_2, live_factor)
    nu_n1 = nu_from_sigma2(s2_n1, live_factor)
    nu_nm1_1 = nu_from_sigma2(s2_nm1_1, live_factor)

    return (nu_n0 - nu_nm1_2) / (nu_n1 - nu_nm1_1)


def r01_differentiable(glob_jax, var_jax, n_target=None,
                       nu_min=1500.0, nu_max=4500.0, n_scan=400, n_steps=8000):
    """Compute r₀₁(n) with gradients flowing through the structure.

    r₀₁(n) = [ν(n−1,0) − 4ν(n−1,1) + 6ν(n,0) − 4ν(n,1) + ν(n+1,0)]
              / [8 · (ν(n,1) − ν(n−1,1))]

    This is the 5-point smoothed small separation ratio from Roxburgh &
    Vorontsov (2003, A&A 411, 215) as used by MESA (astero_support.f90:396)
    and the PLATO/SAS pipeline. It is a smoothed approximation to the
    phase-shift difference ε₀(ν) − ε₁(ν), independent of the near-surface
    structure by construction.

    Note: R&V also define r₀₁*(n) = [ν(n,0) − (ν(n−1,1) + ν(n,1))/2]
    / [ν(n,1) − ν(n−1,1)], the simpler 3-point form. We implement the
    5-point form to match MESA's convention.

    Architecture (gradient-safe under jax.grad) — mirrors r02_differentiable:
      1. Build oscillation coefficients from the TRACED glob_jax/var_jax via
         build_oscillation_coeffs_jax (includes center extension for truncated
         Henyey meshes). Extract concrete copies (stop_gradient) for mode
         identification and eigenvalue anchoring.
      2. Use the concrete coefficients to find l=0,1 eigenfrequencies
         (bracket + Brent), identify modes (np.rint, Tassoul), and find σ²
         anchors for the 5 modes needed for r₀₁.
      3. For each needed mode, pass the concrete σ² anchor + TRACED coefficients
         to eigenfreq_from_coeffs (IFT custom_vjp). The gradient flows through
         the traced coefficients only, at fixed eigenvalue anchor.

    MESA ref: astero/private/astero_support.f90:396
        d01 = (l0(i0-1) - 4*l1(i1-1) + 6*l0(i0) - 4*l1(i1) + l0(i0+1))/8d0
        r01(i) = d01/(l1(i1) - l1(i1-1))

    Args:
        glob_jax: (15,) JAX array — FGONG global parameters
        var_jax: (N, 15) JAX array — per-point structure (center-to-surface)
        n_target: specific radial order to compute (int). If None, picks the
                  middle valid order.
        nu_min, nu_max: frequency bounds for mode search (µHz)
        n_scan: trial frequencies for root bracketing
        n_steps: RK4 integration steps

    Returns:
        JAX scalar: r₀₁ at the selected radial order (differentiable w.r.t.
                    glob_jax, var_jax through the traced computation graph).
    """
    import jax
    import jax.numpy as jnp
    from .adjoint import eigenfreq_from_coeffs
    from .coefficients import build_oscillation_coeffs_jax
    from .integrator import _make_integration_grid
    from .seismic_conversion import nu_from_sigma2

    # ── Step 1: Build coefficients and integration grid (concrete) ──
    grid_data = build_oscillation_coeffs_jax(glob_jax, var_jax)
    x_grid_jax = grid_data['x_grid']
    coeffs_jax = grid_data['coeffs']
    n_center = grid_data['n_center']
    # LIVE factor for the σ²→ν conversion.
    live_factor = grid_data['factor']
    # Frozen float for eigenfreq_from_coeffs nondiff_argnums and mode-finding.
    factor = float(jax.lax.stop_gradient(live_factor))  # GP-2: MESH_POSITIONS (nondiff extraction)

    # Concrete copies for mode-finding (non-differentiable Python operations)
    x_grid_np = np.asarray(jax.lax.stop_gradient(x_grid_jax))  # GP-2: MESH_POSITIONS
    coeffs_np = np.asarray(jax.lax.stop_gradient(coeffs_jax))  # GP-2: MESH_POSITIONS

    # Integration grid from physical mesh points (skip synthetic center)
    x_phys = x_grid_np[n_center:]
    x_steps, h_steps = _make_integration_grid(
        x_phys[x_phys > 1e-4], n_steps=n_steps)

    # ── Step 1b: Non-differentiable mode identification ──
    # r₀₁ needs only l=0,1 but _find_all_modes_from_coeffs finds l=0,1,2.
    # Reuse it to keep mode identification consistent with r02_differentiable.
    all_freqs = _find_all_modes_from_coeffs(
        coeffs_np, x_grid_np, x_steps, h_steps, factor,
        nu_min, nu_max, n_scan)

    freqs_dict = {l: all_freqs[l] for l in all_freqs if l in (0, 1, 2)}
    identified = identify_modes(freqs_dict)

    if not all(l in identified for l in (0, 1)):
        raise ValueError("r01_differentiable: need l=0,1 modes; "
                         f"got l={list(identified.keys())}")

    valid_n = _find_valid_r01_orders(identified)
    if not valid_n:
        raise ValueError("r01_differentiable: no valid (n,l) triples found")

    if n_target is not None:
        if n_target not in valid_n:
            raise ValueError(
                f"r01_differentiable: n_target={n_target} not in "
                f"valid orders {valid_n}")
        n_sel = n_target
    else:
        n_sel = valid_n[len(valid_n) // 2]

    # ── Step 2: Find eigenvalue anchors using CONCRETE coefficients ──
    delta_nu_est = float(np.median(np.diff(np.sort(all_freqs[0]))))
    freq_l0, freq_l1 = identified[0], identified[1]
    anchor_common = dict(delta_nu_est=delta_nu_est, coeffs_np=coeffs_np,
                         x_grid_np=x_grid_np, x_steps=x_steps,
                         h_steps=h_steps, factor=factor,
                         nu_min=nu_min, nu_max=nu_max)

    # 5 modes for r₀₁(n): (n−1,0), (n,0), (n+1,0), (n−1,1), (n,1)
    # MESA astero_support.f90:396: d01 = (l0(i0-1) - 4*l1(i1-1) + 6*l0(i0) - 4*l1(i1) + l0(i0+1))/8
    s2_anchor_nm1_0 = _find_eigenvalue_anchor(0, freq_l0[n_sel - 1],
                                               **anchor_common)
    s2_anchor_n0 = _find_eigenvalue_anchor(0, freq_l0[n_sel], **anchor_common)
    s2_anchor_np1_0 = _find_eigenvalue_anchor(0, freq_l0[n_sel + 1],
                                               **anchor_common)
    s2_anchor_nm1_1 = _find_eigenvalue_anchor(1, freq_l1[n_sel - 1],
                                               **anchor_common)
    s2_anchor_n1 = _find_eigenvalue_anchor(1, freq_l1[n_sel], **anchor_common)

    # ── Step 3: Evaluate 5 frequencies differentiably via IFT, compute ratio ──
    s2_nm1_0 = eigenfreq_from_coeffs(
        coeffs_jax, x_grid_jax, jnp.float64(s2_anchor_nm1_0),
        0, x_steps, h_steps, factor)
    s2_n0 = eigenfreq_from_coeffs(
        coeffs_jax, x_grid_jax, jnp.float64(s2_anchor_n0),
        0, x_steps, h_steps, factor)
    s2_np1_0 = eigenfreq_from_coeffs(
        coeffs_jax, x_grid_jax, jnp.float64(s2_anchor_np1_0),
        0, x_steps, h_steps, factor)
    s2_nm1_1 = eigenfreq_from_coeffs(
        coeffs_jax, x_grid_jax, jnp.float64(s2_anchor_nm1_1),
        1, x_steps, h_steps, factor)
    s2_n1 = eigenfreq_from_coeffs(
        coeffs_jax, x_grid_jax, jnp.float64(s2_anchor_n1),
        1, x_steps, h_steps, factor)

    # Convert σ² → ν using the LIVE factor so ∂factor/∂θ flows.
    nu_nm1_0 = nu_from_sigma2(s2_nm1_0, live_factor)
    nu_n0 = nu_from_sigma2(s2_n0, live_factor)
    nu_np1_0 = nu_from_sigma2(s2_np1_0, live_factor)
    nu_nm1_1 = nu_from_sigma2(s2_nm1_1, live_factor)
    nu_n1 = nu_from_sigma2(s2_n1, live_factor)

    # r₀₁(n) = [ν(n−1,0) − 4ν(n−1,1) + 6ν(n,0) − 4ν(n,1) + ν(n+1,0)]
    #           / [8 · (ν(n,1) − ν(n−1,1))]
    # MESA astero_support.f90:396
    d01 = (nu_nm1_0 - 4.0 * nu_nm1_1 + 6.0 * nu_n0
           - 4.0 * nu_n1 + nu_np1_0) / 8.0
    return d01 / (nu_n1 - nu_nm1_1)


# Legacy / utility functions (echelle_data, estimate_delta_nu,
# compute_solar_model_freqs, compute_evolved_solar_model_freqs, compare_echelle)
# moved to diagnostic.py in — they drag I/O deps (tempfile, matplotlib,
# evolve_star) that don't belong in a pure-math seismic module.
