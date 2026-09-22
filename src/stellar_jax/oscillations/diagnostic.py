"""Diagnostic NumPy/SciPy oscillation solvers — NOT in the JAX autodiff graph.

Post-processing diagnostic module for eigenfrequency computation on FGONG output.
Uses NumPy/SciPy (adaptive RK + Brent). Intentionally outside the differentiable
solver: these are observational validation/comparison tools.

Solvers:
- compute_oscillation_freqs: Cowling 2nd-order (Aerts et al. 2010, eq. 3.138-3.139)
- compute_oscillation_freqs_full: Full 4th-order, GYRE formulation
  (Townsend & Teitler 2013, MNRAS 435, 3406; ad_eqns_m.fypp / rad_eqns_m.fypp)

References:
    Cowling (1941), MNRAS 101, 367
    Aerts, Christensen-Dalsgaard & Kurtz (2010), "Asteroseismology", Springer
    Christensen-Dalsgaard & Mullan (1994), MNRAS 270, 921 (CM94)
    Townsend & Teitler (2013), MNRAS 435, 3406 (GYRE formulation)
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from stellar_jax.oscillations.coefficients import _extract_fgong_numpy
from stellar_jax.fgong.io import read_fgong

def _build_oscillation_grid(glob, var):
    """Extract dimensionless structure variables for oscillation computation.

    Delegates FGONG column extraction + BV frequency to _extract_fgong_numpy
    (the shared DRY helper in coefficients.py).

    Returns dict with arrays (center-to-surface):
        x, m_frac, P, rho, gamma1, g, N2, c_s, M, R, G
    """
    fgong = _extract_fgong_numpy(glob, var)
    return dict(x=fgong['x'], m_frac=fgong['m'] / fgong['M'],
                P=fgong['P'], rho=fgong['rho'], gamma1=fgong['gamma1'],
                g=fgong['g'], N2=fgong['N2'], c_s=fgong['c_s'],
                M=fgong['M'], R=fgong['R'], G=fgong['G'])



def compute_oscillation_freqs(glob, var, l_values=(0, 1, 2, 3),
                              nu_min=1000.0, nu_max=4500.0, n_scan=500):
    """Compute adiabatic p-mode frequencies using the Cowling approximation.

    Solves the dimensionless Cowling equations (Aerts et al. 2010, eq. 3.138–3.139)
    via shooting from center to surface with brentq root-finding.

    Args:
        glob: FGONG global parameters
        var: FGONG per-point variables (center-to-surface)
        l_values: angular degrees to compute
        nu_min: minimum frequency in μHz
        nu_max: maximum frequency in μHz
        n_scan: number of trial frequencies for root bracketing

    Returns:
        dict mapping l -> array of frequencies in μHz
    """
    grid = _build_oscillation_grid(glob, var)
    x = grid['x']
    M, R, G_val = grid['M'], grid['R'], grid['G']

    r = x * R
    m = grid['m_frac'] * M
    P = grid['P']
    rho = grid['rho']
    gamma1 = grid['gamma1']
    g = grid['g']

    # Structure coefficients (Aerts et al. 2010, §3.3):
    # V_g = V/Γ₁ = Gm/(r·Γ₁P/ρ) = Gmρ/(rΓ₁P)
    Vg = G_val * m * rho / (np.maximum(r, 1.0) * gamma1 * np.maximum(P, 1e-30))
    # c₁ = (r/R)³·(M/m)
    c1 = (r / R)**3 * M / np.maximum(m, 1e-10 * M)
    # A* = r·N²/g (Brunt-Väisälä parameter)
    A_star = grid['N2'] * np.maximum(r, 1.0) / np.maximum(g, 1e-10)
    # U = d ln m / d ln r = 4πr³ρ/m
    U = 4.0 * np.pi * r**3 * rho / np.maximum(m, 1e-10 * M)

    # Working grid: exclude singular center; use the full model extent.
    #
    # ADIPLS (CD2008, Ap&SS 316, 113, §2.1) defines the outermost integration
    # point as x_s = R_s/R where R_s "is the surface radius, including the
    # atmosphere; thus, typically, x_s > 1." The δP=0 surface BC is applied at
    # this outermost point. We do the same.
    #
    # Empirical validation (Model S FGONG, Δν from l=0 modes 1300-4050 μHz):
    #   With x<1.0 cutoff (OLD):  Cowling 140.09 μHz (+3.3% vs ADIPLS 135.6)
    #                              Full solver 137.92 μHz (+1.7%)
    #   With full extent (FIX):   Cowling 133.65 μHz (-1.4%)
    #                              Full solver 136.15 μHz (+0.4%) ← matches ADIPLS
    # The ~2% acoustic-cavity extension (∫dr/c_s grows 1.98%) explains the shift.
    mask = (x > 1e-4)
    x_grid = x[mask]
    Vg_grid = np.nan_to_num(Vg[mask], nan=0.0, posinf=0.0, neginf=0.0)
    c1_grid = np.nan_to_num(c1[mask], nan=1.0, posinf=1e10, neginf=1.0)
    A_star_grid = np.nan_to_num(A_star[mask], nan=0.0, posinf=0.0, neginf=0.0)
    U_grid = np.nan_to_num(U[mask], nan=3.0, posinf=3.0, neginf=3.0)

    # Dimensionless frequency: ω² = (2πν)² R³/(GM)
    from .seismic_conversion import compute_factor
    factor = compute_factor(R, M, G_val)

    def shooting_determinant(omega2, l):
        """Integrate Cowling equations from center outward, return surface BC residual."""
        def rhs(x_val, y):
            Vg_x = np.interp(x_val, x_grid, Vg_grid)
            c1_x = np.interp(x_val, x_grid, c1_grid)
            As_x = np.interp(x_val, x_grid, A_star_grid)
            U_x = np.interp(x_val, x_grid, U_grid)
            y1, y2 = y
            # Cowling equations (Aerts et al. 2010, eq. 3.138–3.139):
            #   x dy₁/dx = (V_g - 1 - l)y₁ + [l(l+1)/(c₁ω²) - V_g]y₂
            #   x dy₂/dx = (c₁ω² - A*)y₁ + (A* + 1 - U)y₂
            dy1 = ((Vg_x - 1.0 - l) * y1 +
                   (l * (l + 1) / (c1_x * omega2) - Vg_x) * y2) / x_val
            dy2 = ((c1_x * omega2 - As_x) * y1 +
                   (As_x + 1.0 - U_x) * y2) / x_val
            return [dy1, dy2]

        # Central boundary conditions (Aerts et al. 2010, eq. 3.147):
        # At center: U→3, A*→0, c₁→1 (for well-resolved models).
        # Regular solution: y₁ ~ x^(l-1), y₂ ~ x^l
        # Leading-order ratio: y₂/y₁ = c₁ω²/[l(l+1)] for l>0
        #                       y₂/y₁ = c₁ω²/3        for l=0
        # (Unno et al. 1989, eq. 14.12–14.13)
        x_in = x_grid[0]
        c1_in = np.interp(x_in, x_grid, c1_grid)
        y1_in = 1.0
        if l == 0:
            y2_in = c1_in * omega2 / 3.0
        else:
            y2_in = c1_in * omega2 / (l * (l + 1))

        # Integrate outward. Use DOP853 (8th-order) with relaxed tolerances:
        # RK45 gets stuck on Model S FGONG (2400 pts, stiff-like 1/x coefficients).
        # DOP853 takes far fewer steps for smooth ODE with rapidly-varying
        # interpolated coefficients. max_step prevents getting stuck near center.
        sol = solve_ivp(rhs, [x_in, x_grid[-1]], [y1_in, y2_in],
                        method='DOP853', rtol=1e-5, atol=1e-8,
                        max_step=0.05)
        if not sol.success:
            return np.nan

        y1_s, y2_s = sol.y[:, -1]

        # Surface BC: δP = 0 at free surface (zero-pressure condition).
        # For high-order p-modes this is accurate to <0.3% for n≥5
        # (Christensen-Dalsgaard & Mullan 1994, MNRAS 270, 921, Table 1).
        # Eigenfrequencies are zeros of y₂(surface) as a function of ω².
        return y2_s

    results = {}
    for l in l_values:
        nu_arr = np.linspace(nu_min, nu_max, n_scan)
        omega2_arr = nu_arr**2 * factor
        det_values = np.array([shooting_determinant(w2, l) for w2 in omega2_arr])

        freqs = []
        for i in range(len(det_values) - 1):
            if np.isnan(det_values[i]) or np.isnan(det_values[i + 1]):
                continue
            if det_values[i] * det_values[i + 1] < 0:
                def f_root(nu, _l=l):
                    return shooting_determinant(nu**2 * factor, _l)
                try:
                    nu_root = brentq(f_root, nu_arr[i], nu_arr[i + 1],
                                     rtol=1e-8, maxiter=50)
                    freqs.append(nu_root)
                except (ValueError, RuntimeError):
                    continue
        results[l] = np.array(sorted(freqs))
    return results


def _radial_det_full(sigma2, x_grid, Vg_grid, A1_grid, A_grid, U_grid):
    """Radial determinant via full (non-Cowling) shooting — GYRE formulation."""
    def rhs(xv, y):
        Vg_x = np.interp(xv, x_grid, Vg_grid)
        A1_x = np.interp(xv, x_grid, A1_grid)
        As_x = np.interp(xv, x_grid, A_grid)
        U_x = np.interp(xv, x_grid, U_grid)
        c1w2 = sigma2 / max(A1_x, 1e-30)
        y1, y2 = y
        dy1 = ((Vg_x - 1.0) * y1 - Vg_x * y2) / xv
        dy2 = ((c1w2 + U_x - As_x) * y1 + (As_x - U_x + 3.0) * y2) / xv
        return [dy1, dy2]
    sol = solve_ivp(rhs, [x_grid[0], x_grid[-1]], [0.0, 1.0],
                    method='DOP853', rtol=1e-8, atol=1e-10, max_step=0.05)
    if not sol.success:
        return np.nan
    return sol.y[0, -1] - sol.y[1, -1]


def _nonradial_det_full(sigma2, l, x_grid, Vg_grid, A1_grid, A_grid,
                        U_grid):
    """Nonradial 4th-order determinant via full shooting — GYRE formulation.

    GYRE's nonradial variable set (src/ad/ad_eqns_m.fypp):
      x·dy₁/dx = (Vg - 1 - l)·y₁ + (λ/(c₁σ²) - Vg)·y₂ + λ/(c₁σ²)·y₃
      x·dy₂/dx = (c₁σ² - As)·y₁ + (As - U + 3 - l)·y₂ - y₄
      x·dy₃/dx = (3 - U - l)·y₃ + y₄
      x·dy₄/dx = U·As·y₁ + U·Vg·y₂ + l(l+1)·y₃ + (2 - U - l)·y₄

    where λ = l(l+1), c₁ = 1/A1, λ/(c₁σ²) = ll1·A1/σ².

    Inner regular BCs (ad_bound_m.fypp, build_regular_i_):
      sol1 = [1, c₁·σ²/l, 0, 0]
      sol2 = [0, -1, 1, l]

    Outer vacuum BCs (ad_bound_m.fypp):
      bc1 = y₁ - y₂
      bc2 = U·y₁ + (l+1)·y₃ + y₄
    """
    ll1 = l * (l + 1)

    def rhs(xv, y):
        Vg_x = np.interp(xv, x_grid, Vg_grid)
        A1_x = np.interp(xv, x_grid, A1_grid)
        As_x = np.interp(xv, x_grid, A_grid)
        U_x = np.interp(xv, x_grid, U_grid)
        lam_c1w2 = ll1 * A1_x / sigma2    # λ/(c₁σ²)
        c1w2 = sigma2 / max(A1_x, 1e-30)  # c₁·σ²
        y1, y2, y3, y4 = y
        dy1 = ((Vg_x - 1.0 - l) * y1
               + (lam_c1w2 - Vg_x) * y2
               + lam_c1w2 * y3) / xv
        dy2 = ((c1w2 - As_x) * y1
               + (As_x - U_x + 3.0 - l) * y2
               - y4) / xv
        dy3 = ((3.0 - U_x - l) * y3
               + y4) / xv
        dy4 = (U_x * As_x * y1
               + U_x * Vg_x * y2
               + ll1 * y3
               + (2.0 - U_x - l) * y4) / xv
        return [dy1, dy2, dy3, dy4]

    x_in = x_grid[0]
    # Inner regular BCs (GYRE ad_bound_m.fypp):
    # c₁ at center ≈ 1/A1 at x_in
    A1_c = np.interp(x_in, x_grid, A1_grid)
    c1_c = 1.0 / max(A1_c, 1e-30)
    # Solution 1: y₁=1, y₂=c₁σ²/l, y₃=0, y₄=0
    ic1 = [1.0, c1_c * sigma2 / l, 0.0, 0.0]
    # Solution 2: y₁=0, y₂=-1, y₃=1, y₄=l
    ic2 = [0.0, -1.0, 1.0, float(l)]
    sol1 = solve_ivp(rhs, [x_in, x_grid[-1]], ic1,
                     method='DOP853', rtol=1e-8, atol=1e-10, max_step=0.05)
    sol2 = solve_ivp(rhs, [x_in, x_grid[-1]], ic2,
                     method='DOP853', rtol=1e-8, atol=1e-10, max_step=0.05)
    if not sol1.success or not sol2.success:
        return np.nan

    y1_1, y2_1, y3_1, y4_1 = sol1.y[:, -1]
    y1_2, y2_2, y3_2, y4_2 = sol2.y[:, -1]

    # U at surface (for outer vacuum BC)
    x_s = x_grid[-1]
    U_s = np.interp(x_s, x_grid, U_grid)

    # Outer vacuum BC determinant (2×2), GYRE ad_bound_m.fypp:
    #   bc1 = y₁ - y₂
    #   bc2 = U·y₁ + (l+1)·y₃ + y₄
    bc1_1 = y1_1 - y2_1
    bc1_2 = y1_2 - y2_2
    bc2_1 = U_s * y1_1 + (l + 1) * y3_1 + y4_1
    bc2_2 = U_s * y1_2 + (l + 1) * y3_2 + y4_2
    return bc1_1 * bc2_2 - bc1_2 * bc2_1


def compute_oscillation_freqs_full(glob, var, l_values=(0, 1, 2, 3),
                                   nu_min=1000.0, nu_max=4500.0, n_scan=500):
    """Compute adiabatic p-mode frequencies using the FULL 4th-order equations.

    Includes the gravitational perturbation Φ' (Poisson equation coupling),
    which the Cowling approximation neglects. This reduces the ~2-3% systematic
    overestimate of individual mode frequencies present in the Cowling solver.

    Method — GYRE formulation (Townsend & Teitler 2013, MNRAS 435, 3406):
    - l=0 (radial): GYRE's radial equations (src/rad/rad_eqns_m.fypp). The
      Brunt term enters un-amplified. Inner regular BC: y=[0,1].
      Outer vacuum BC: y₁ - y₂ = 0.
    - l≥1 (nonradial): GYRE's 4th-order system (src/ad/ad_eqns_m.fypp). Two
      linearly independent solutions integrated from center satisfying inner
      regular BCs (ad_bound_m.fypp); eigenvalue from 2×2 determinant of outer
      vacuum BCs: bc1 = y₁ - y₂, bc2 = U·y₁ + (l+1)·y₃ + y₄.

    This formulation replaces the ADIPLS A-formulation (CD2008 Eqs. 11-14,
    17-18) where the Brunt term A entered multiplied by η = l(l+1)/(c₁σ²) —
    a frequency-amplified factor that made the composition-gradient sharpness
    in evolved cores numerically explosive, biasing δν₀₂ low by 5–40%
    (issue #682). Mirrors the JAX production path (PR #695) exactly.

    Args:
        glob: FGONG global parameters
        var: FGONG per-point variables (center-to-surface)
        l_values: angular degrees to compute
        nu_min: minimum frequency in μHz
        nu_max: maximum frequency in μHz
        n_scan: number of trial frequencies for root bracketing

    Returns:
        dict mapping l -> array of frequencies in μHz
    """
    grid = _build_oscillation_grid(glob, var)
    x = grid['x']
    M, R, G_val = grid['M'], grid['R'], grid['G']

    r = x * R
    m = grid['m_frac'] * M
    P = grid['P']
    rho = grid['rho']
    gamma1 = grid['gamma1']
    g = grid['g']

    # Structure coefficients (GYRE notation):
    # Vg = V/Γ₁ = G·m·ρ/(r·Γ₁·P)
    Vg = G_val * m * rho / (np.maximum(r, 1.0) * gamma1 * np.maximum(P, 1e-30))
    # A1 = (m/M)/x³  (inverse c₁: c₁ = 1/A1)
    A1 = (m / M) / np.maximum(x**3, 1e-30)
    # As = Brunt A* = r·N²/g  (enters UN-amplified in the GYRE formulation)
    A_bv = grid['N2'] * np.maximum(r, 1.0) / np.maximum(g, 1e-10)
    # U = d ln m / d ln r = 4πr³ρ/m
    U = 4.0 * np.pi * r**3 * rho / np.maximum(m, 1e-10 * M)

    # Working grid: exclude singular center; use full model extent.
    # See Cowling solver comment for full rationale and empirical evidence.
    # GYRE integrates to x_s (outermost model point); we match that domain.
    mask = (x > 1e-4)
    x_grid = x[mask]
    Vg_grid = np.nan_to_num(Vg[mask], nan=0.0, posinf=0.0, neginf=0.0)
    A1_grid = np.nan_to_num(A1[mask], nan=1.0, posinf=1e10, neginf=1.0)
    A_grid = np.nan_to_num(A_bv[mask], nan=0.0, posinf=0.0, neginf=0.0)
    U_grid = np.nan_to_num(U[mask], nan=3.0, posinf=3.0, neginf=3.0)

    # Dimensionless frequency: σ² = ω²R³/(GM) = (2πν·1e-6)²·R³/(GM)
    from .seismic_conversion import compute_factor
    factor = compute_factor(R, M, G_val)

    def _radial_shooting(sigma2):
        return _radial_det_full(sigma2, x_grid, Vg_grid, A1_grid, A_grid, U_grid)

    def _nonradial_shooting(sigma2, l):
        return _nonradial_det_full(sigma2, l, x_grid, Vg_grid, A1_grid, A_grid,
                                   U_grid)


    results = {}
    for l in l_values:
        nu_arr = np.linspace(nu_min, nu_max, n_scan)
        sigma2_arr = nu_arr**2 * factor

        if l == 0:
            det_fn = _radial_shooting
        else:
            def det_fn(s2, _l=l):
                return _nonradial_shooting(s2, _l)

        det_values = np.array([det_fn(s2) for s2 in sigma2_arr])

        freqs = []
        for i in range(len(det_values) - 1):
            if np.isnan(det_values[i]) or np.isnan(det_values[i + 1]):
                continue
            if det_values[i] * det_values[i + 1] < 0:
                def f_root(nu, _l=l):
                    s2 = nu**2 * factor
                    if _l == 0:
                        return _radial_shooting(s2)
                    return _nonradial_shooting(s2, _l)
                try:
                    nu_root = brentq(f_root, nu_arr[i], nu_arr[i + 1],
                                     rtol=1e-8, maxiter=50)
                    freqs.append(nu_root)
                except (ValueError, RuntimeError):
                    continue
        results[l] = np.array(sorted(freqs))
    return results




# ════════════════════════════════════════════════════════════════════════════════
# Legacy / utility functions — moved from seismic_quantities.py
# These drag I/O deps (tempfile, matplotlib, evolve_star) that don't belong in
# a pure-math seismic module.
# ════════════════════════════════════════════════════════════════════════════════

def echelle_data(freqs_dict, delta_nu):
    """Compute échelle diagram coordinates: (ν mod Δν, ν) for each l."""
    result = {}
    for l, freqs in freqs_dict.items():
        result[l] = (freqs % delta_nu, freqs)
    return result


def estimate_delta_nu(freqs_dict, l=0):
    """Estimate Δν (large separation) from median consecutive l=0 spacing.

    Note: for a structure-based Δν, use large_separation(glob, var) which
    computes the acoustic radius integral (MESA report.f90:343).
    """
    if l not in freqs_dict or len(freqs_dict[l]) < 3:
        all_freqs = np.sort(np.concatenate(list(freqs_dict.values())))
        if len(all_freqs) < 3:
            raise ValueError(
                "Cannot estimate Δν: fewer than 3 modes available. "
                "Use large_separation(glob, var) for structure-based Δν.")
        return float(np.median(np.diff(all_freqs)))
    return float(np.median(np.diff(freqs_dict[l])))


def compute_solar_model_freqs(stellar_module, Z=0.0188, max_steps=500):
    """Compute oscillation frequencies for our solar-calibrated ZAMS model.

    Note: This uses the ZAMS structure (t=0), NOT the evolved 4.57 Gyr model.
    The ZAMS radius of a 1 M☉ star is ~13% smaller than the present Sun
    (R_ZAMS ≈ 0.87 R☉), giving Δν_ZAMS ≈ 135·(0.87)^{-3/2} ≈ 180 μHz.
    This is expected and NOT a validation failure — it reflects the ZAMS-only
    structure (the evolved solar model is computed separately).

    Args:
        stellar_module: the imported stellar module
        Z: metallicity
        max_steps: unused (ZAMS only)

    Returns:
        (glob, var, freqs_dict): FGONG data and frequencies
    """
    import tempfile
    import os
    import jax.numpy as jnp

    alpha = float(stellar_module.ALPHA_SOLAR)
    Y_init = float(stellar_module.Y0_SOLAR)
    X_init = 1.0 - Y_init - Z

    N_COMP = int(stellar_module.N_COMP)
    X_profile = jnp.full(N_COMP, X_init)

    mass = jnp.float64(1.0)
    log_L_g, log_Te_g = stellar_module.initial_guess(mass)
    logL, logTe = stellar_module.newton_solve_xprofile(
        mass, X_profile, jnp.float64(Z), jnp.float64(0.0),
        log_L_g, log_Te_g, jnp.float64(alpha),
        int(stellar_module.N_NEWTON_COLD))

    fd, tmpfile = tempfile.mkstemp(suffix='.fgong')
    os.close(fd)
    try:
        stellar_module.write_fgong(
            tmpfile, 1.0, float(logL), float(logTe),
            X_profile, Z, 0.0, alpha)
        glob, var = read_fgong(tmpfile)
        freqs = compute_oscillation_freqs(glob, var)
    finally:
        os.unlink(tmpfile)
    return glob, var, freqs


def compute_evolved_solar_model_freqs(stellar_module, Z=0.0188, t_max=4.57e9,
                                      max_steps=500, l_values=(0, 1, 2, 3),
                                      nu_min=1000.0, nu_max=4500.0, n_scan=150):
    """Compute l=0–3 oscillation frequencies for the evolved solar-calibrated model.

    Evolves a 1 M☉ star to t_max (default 4.57 Gyr = solar age) using the
    calibrated ALPHA_SOLAR and Y0_SOLAR, then computes adiabatic p-mode
    frequencies with the full 4th-order solver.

    This is the "our model" side of the M3 validation deliverable: the evolved
    solar model frequencies to overlay vs Model S on an échelle diagram.

    Args:
        stellar_module: the imported stellar module (stellar.py)
        Z: metallicity (default 0.0188)
        t_max: target age in years (default 4.57e9 = solar age)
        max_steps: evolution timesteps
        l_values: angular degrees to compute
        nu_min, nu_max: frequency scan range (μHz)
        n_scan: scan points for root-finding

    Returns:
        dict with keys:
            'glob': FGONG global parameters
            'var': FGONG per-point variables
            'freqs': dict l -> array of frequencies in μHz
            'delta_nu': large separation (μHz)
            'R_over_Rsun': stellar radius in solar units
            'age_gyr': final age reached (Gyr)

    References:
        Christensen-Dalsgaard (2008), Ap&SS 316, 113 (ADIPLS)
        Christensen-Dalsgaard et al. (1996), Science 272, 1286 (Model S)
    """
    import tempfile
    import os
    import jax.numpy as jnp

    alpha = float(stellar_module.ALPHA_SOLAR)

    # Evolve to solar age
    result = stellar_module.evolve_star(
        1.0, Z=Z, max_steps=max_steps, alpha_mlt=stellar_module.ALPHA_SOLAR,
        t_max=t_max, Y_init=stellar_module.Y0_SOLAR)
    X_final = result['X_profile']
    age_gyr = float(result['star_age'][-1])

    # Newton re-solve for self-consistent structure at final state
    mass = jnp.float64(1.0)
    logL, logTe = stellar_module.newton_solve_xprofile(
        mass, X_final, jnp.float64(Z), jnp.float64(t_max),
        result['log_L_final'], result['log_Teff_final'],
        jnp.float64(alpha), stellar_module.N_NEWTON_WARM)

    # Write FGONG and compute frequencies
    fd, tmpfile = tempfile.mkstemp(suffix='.fgong')
    os.close(fd)
    try:
        stellar_module.write_fgong(
            tmpfile, 1.0, float(logL), float(logTe),
            X_final, Z, t_max, alpha)
        glob, var = read_fgong(tmpfile)
    finally:
        if os.path.exists(tmpfile):
            os.unlink(tmpfile)

    R_over_Rsun = glob[1] / 6.96e10
    freqs = compute_oscillation_freqs_full(
        glob, var, l_values=l_values, nu_min=nu_min, nu_max=nu_max, n_scan=n_scan)
    delta_nu = estimate_delta_nu(freqs, l=0)

    return {
        'glob': glob,
        'var': var,
        'freqs': freqs,
        'delta_nu': delta_nu,
        'R_over_Rsun': R_over_Rsun,
        'age_gyr': age_gyr,
    }


def compare_echelle(model_s_fgong, stellar_module=None, Z=0.0188,
                    nu_min=1500, nu_max=3500, n_scan=200, save_path=None):
    """Compare échelle diagrams of Model S and our solar model.

    Args:
        model_s_fgong: path to Model S FGONG file
        stellar_module: imported stellar module (if None, only Model S)
        Z: metallicity for our model
        nu_min, nu_max: frequency range (μHz)
        n_scan: scan points for root finding
        save_path: save plot path (requires matplotlib)

    Returns:
        dict with model_s_freqs, model_s_delta_nu, our_freqs, our_delta_nu
    """
    glob_ms, var_ms = read_fgong(model_s_fgong)
    freqs_ms = compute_oscillation_freqs(glob_ms, var_ms, l_values=(0, 1, 2, 3),
                                         nu_min=nu_min, nu_max=nu_max, n_scan=n_scan)
    delta_nu_ms = estimate_delta_nu(freqs_ms, l=0)

    result = {
        'model_s_freqs': freqs_ms,
        'model_s_delta_nu': delta_nu_ms,
        'our_freqs': None,
        'our_delta_nu': None,
    }

    if stellar_module is not None:
        _, _, freqs_ours = compute_solar_model_freqs(stellar_module, Z=Z)
        delta_nu_ours = estimate_delta_nu(freqs_ours, l=0)
        result['our_freqs'] = freqs_ours
        result['our_delta_nu'] = delta_nu_ours

    if save_path is not None:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            n_panels = 2 if result['our_freqs'] else 1
            fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 8))
            if not isinstance(axes, np.ndarray):
                axes = [axes]

            colors = {0: 'C0', 1: 'C1', 2: 'C2', 3: 'C3'}
            markers = {0: 'o', 1: 's', 2: '^', 3: 'v'}

            ax = axes[0]
            ech_ms = echelle_data(freqs_ms, delta_nu_ms)
            for l_deg in range(4):
                if l_deg in ech_ms:
                    xv, yv = ech_ms[l_deg]
                    ax.scatter(xv, yv, c=colors[l_deg], marker=markers[l_deg],
                              s=30, label=f'l={l_deg}', alpha=0.8)
            ax.set_xlabel(f'ν mod Δν (μHz), Δν = {delta_nu_ms:.1f}')
            ax.set_ylabel('ν (μHz)')
            ax.set_title('Model S (Cowling approx.)')
            ax.legend()
            ax.set_xlim(0, delta_nu_ms)

            if result['our_freqs'] and len(axes) > 1:
                ax = axes[1]
                ech_ours = echelle_data(result['our_freqs'], delta_nu_ours)
                for l_deg in range(4):
                    if l_deg in ech_ours:
                        xv, yv = ech_ours[l_deg]
                        ax.scatter(xv, yv, c=colors[l_deg], marker=markers[l_deg],
                                  s=30, label=f'l={l_deg}', alpha=0.8)
                ax.set_xlabel(f'ν mod Δν (μHz), Δν = {delta_nu_ours:.1f}')
                ax.set_ylabel('ν (μHz)')
                ax.set_title('stellar-jax ZAMS (Cowling approx.)')
                ax.legend()
                ax.set_xlim(0, delta_nu_ours)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        except ImportError:
            pass

    return result
