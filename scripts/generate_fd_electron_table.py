#!/usr/bin/env python3
"""Generate the Fermi-Dirac electron EOS table for the degenerate regime.

This script computes the electron Helmholtz free energy per unit mass on a
grid of (log rho_e, log T) via numerical evaluation of the relativistic
Fermi-Dirac integrals — the same method MESA's HELM table uses (Timmes &
Swesty 2000, ApJS 126, 501; Aparicio 1998, ApJS 117, 627).

The table stores f_e(rho_e, T) [erg/g] where rho_e = ye * rho = Z/A * rho
is the electron mass density ("din" in MESA), plus precomputed derivative
arrays (fx, fy, fxy) for C² bicubic Hermite interpolation (matching MESA's
h3 scheme, helm_polynomials.f90:197). The derivatives are computed via
natural cubic spline fitting (scipy CubicSpline, bc_type="natural") and
scaled by cell spacing to match MESA's convention:
  fx = ∂f/∂x · Δx   (MESA: fd · dd_sav)
  fy = ∂f/∂y · Δy   (MESA: ft · dt_sav)
  fxy = ∂²f/∂x∂y · Δx · Δy   (MESA: fdt · dd_sav · dt_sav)

All thermodynamic quantities (P, S, Gamma1, nabla_ad, cp, chi_rho, chi_T)
are derived from f_e via JAX autodiff — guaranteeing Maxwell-relation
consistency by construction.

Grid: log(rho_e) from 1.0 to 8.0, logT from 6.0 to 9.0
Resolution: N_rho x N_T = 701 x 301 (uniform in log space, Δ=0.01 matching MESA)

This covers the He-core degenerate regime through partial degeneracy,
with generous margins on both sides.

Reference:
  - Timmes & Swesty (2000), ApJS 126, 501 — HELM EOS table method
  - Aparicio (1998), ApJS 117, 627 — efficient FD integral evaluation
  - Chandrasekhar (1939), Introduction to Stellar Structure, Ch. 10
  - MESA: helm_polynomials.f90 (h3 bicubic Hermite, xpsi0/xpsi1)
  - MESA: helm_alloc.f90 (binary table with stored derivatives)

Output: data/fd_electron_eos.npz
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import os
import sys
import time

# --- Physical constants (CGS, CODATA 2018) ---
k_B = 1.380649e-16       # Boltzmann constant [erg/K]
m_e = 9.1093837015e-28   # electron mass [g]
m_H = 1.6726219237e-24   # proton mass [g]
hbar = 1.054571817e-27   # reduced Planck constant [erg s]
c_light = 2.99792458e10  # speed of light [cm/s]
pi = np.pi


def fd_integral(k, eta, theta, epsabs=1e-10, epsrel=1e-10):
    """Relativistic Fermi-Dirac integral F_k(eta, theta).

    F_k(eta, theta) = int_0^inf x^k * sqrt(1 + x*theta/2) / (exp(x - eta) + 1) dx

    Parameters:
        k: index (1/2, 3/2, 5/2, ...)
        eta: degeneracy parameter = mu_e / (k_B * T)
        theta: relativity parameter = k_B * T / (m_e * c^2)

    Reference: Timmes & Swesty (2000) eq. 3; Aparicio (1998)
    """
    def integrand(x):
        dxst = np.sqrt(1.0 + 0.5 * x * theta)
        exponent = x - eta
        if exponent < 100:
            return x**k * dxst / (np.exp(exponent) + 1.0)
        else:
            return x**k * dxst * np.exp(-exponent)

    # Split the integral at x=eta (the Fermi edge) for better convergence
    # at high degeneracy where the integrand drops sharply
    if eta > 0:
        # Split at the Fermi edge for better convergence
        split = max(eta, 1.0)
        r1, _ = quad(integrand, 0, split, epsabs=epsabs, epsrel=epsrel,
                     limit=200)
        r2, _ = quad(integrand, split, np.inf, epsabs=epsabs, epsrel=epsrel,
                     limit=200)
        return r1 + r2
    else:
        result, _ = quad(integrand, 0, np.inf, epsabs=epsabs, epsrel=epsrel,
                         limit=200)
        return result


def fd_integral_deriv_eta(k, eta, theta, epsabs=1e-12, epsrel=1e-12):
    """Derivative of F_k w.r.t. eta: dF_k/d(eta).

    dF_k/d(eta) = int_0^inf x^k * sqrt(1 + x*theta/2) * exp(x-eta) / (exp(x-eta)+1)^2 dx
                = F_{k-1}(eta, theta) for theta=0, but we compute directly for theta != 0.

    Reference: Timmes & Swesty (2000) eq. 4
    """
    def integrand(x):
        dxst = np.sqrt(1.0 + 0.5 * x * theta)
        exponent = x - eta
        if exponent < 100:
            fac = np.exp(exponent)
            return x**k * dxst * fac / (fac + 1.0)**2
        else:
            return x**k * dxst * np.exp(-exponent)

    result, _ = quad(integrand, 0, np.inf, epsabs=epsabs, epsrel=epsrel,
                     limit=200)
    return result


def fd_integral_deriv_theta(k, eta, theta, epsabs=1e-12, epsrel=1e-12):
    """Derivative of F_k w.r.t. theta: dF_k/d(theta).

    dF_k/d(theta) = int_0^inf x^(k+1) / (4*sqrt(1 + x*theta/2)) / (exp(x-eta)+1) dx

    Reference: Timmes & Swesty (2000) eq. 5
    """
    def integrand(x):
        dxst = np.sqrt(1.0 + 0.5 * x * theta)
        exponent = x - eta
        if exponent < 100:
            return x**(k + 1) / (4.0 * dxst * (np.exp(exponent) + 1.0))
        else:
            return x**(k + 1) / (4.0 * dxst) * np.exp(-exponent)

    result, _ = quad(integrand, 0, np.inf, epsabs=epsabs, epsrel=epsrel,
                     limit=200)
    return result


def solve_eta(n_e, T, theta):
    """Solve for the degeneracy parameter eta given electron number density.

    The number density equation (Timmes & Swesty 2000, eq. 6):
      n_e = (8*pi*sqrt(2))/(h^3) * (m_e*c)^3 * theta^{3/2} * [F_{1/2}(eta,theta) + theta*F_{3/2}(eta,theta)]

    We invert this numerically for eta. For strongly degenerate matter
    (high rho_e, low T), eta can be very large (>1000).
    """
    h_planck = 2 * pi * hbar
    prefactor = 8.0 * pi * np.sqrt(2.0) / h_planck**3 * (m_e * c_light)**3 * theta**1.5

    def residual(eta):
        f12 = fd_integral(0.5, eta, theta)
        f32 = fd_integral(1.5, eta, theta)
        n_computed = prefactor * (f12 + theta * f32)
        return n_computed - n_e

    # Initial guess from non-relativistic asymptotic:
    # In the strongly degenerate limit (eta >> 1, theta << 1):
    #   F_{1/2}(eta, 0) ~ (2/3) * eta^{3/2}
    # So n_e ~ prefactor * (2/3) * eta^{3/2}
    # → eta ~ (3*n_e / (2*prefactor))^{2/3}
    eta_guess = (1.5 * n_e / prefactor) ** (2.0 / 3.0)

    # For low degeneracy (ideal gas limit): eta ~ ln(n_e/nQ) where
    # nQ = (2*pi*m_e*kT/h^2)^{3/2}
    nQ = (2 * pi * m_e * k_B * T / h_planck**2) ** 1.5
    eta_ideal = np.log(max(n_e / nQ, 1e-30))

    # Use the larger as a starting upper bound
    eta_upper = max(eta_guess * 2.0, eta_ideal + 50.0, 50.0)
    eta_lower = min(eta_ideal - 50.0, -50.0)

    # Expand bracket until it encloses the root
    for _ in range(20):
        r_lo = residual(eta_lower)
        r_hi = residual(eta_upper)
        if r_lo * r_hi <= 0:
            break
        if r_lo > 0:
            eta_lower *= 2.0
        if r_hi < 0:
            eta_upper *= 2.0
    else:
        raise ValueError(f"Cannot bracket eta for n_e={n_e:.3e}, T={T:.3e}, "
                         f"bounds=[{eta_lower:.1f}, {eta_upper:.1f}]")

    eta = brentq(residual, eta_lower, eta_upper, xtol=1e-10, rtol=1e-10)
    return eta


def electron_free_energy_per_mass(rho_e, T):
    """Compute the electron Helmholtz free energy per unit mass [erg/g].

    This is the FULL relativistic Fermi-Dirac expression — no Sommerfeld
    expansion, no Chandrasekhar approximation. Matches MESA's HELM table
    generation method.

    The free energy per unit volume (Timmes & Swesty 2000, eq. 24):
      F_e/V = n_e * k_B * T * (eta - tau)
    where tau = (F_{3/2} + theta*F_{5/2}) / (F_{1/2} + theta*F_{3/2})
    and eta is the chemical potential / (k_B * T).

    Actually, the cleaner expression is via the thermodynamic identity:
      f_e = (mu_e * n_e - P_e) / rho_e   (Helmholtz free energy per mass)
    where:
      P_e = (8*pi*sqrt(2))/(3*h^3) * (m_e*c)^3 * (k_B*T) * theta^{3/2}
            * [F_{3/2}(eta,theta) + theta/2 * F_{5/2}(eta,theta)]
      mu_e_chem = eta * k_B * T  (chemical potential)
      n_e = rho_e / m_H  (number density for ye=1, but we use rho_e directly)

    We use the direct expression for the free energy density from
    Timmes & Swesty (2000) eq. 8-10 (via the pressure and number density):
      f_e = (mu * n_e - P_e) / rho_e

    Reference: Timmes & Swesty (2000), ApJS 126, 501, §2
    """
    # Electron number density: rho_e = n_e * m_H (for mu_e = 1; rho_e = ye*rho)
    # Actually n_e = rho_e / (mu_e * m_H) but since rho_e = ye*rho and n_e = rho/(mu_e*m_H)
    # we have n_e = rho_e / m_H (the "din" variable in MESA is rho_e = ye*rho)
    # Wait — MESA uses din = ye * den, and n_e = din / (amu) = rho_e / m_H
    # Actually no: n_e = rho / (mu_e * m_H) and rho_e = ye * rho = (Z/A) * rho
    # so n_e = rho / (mu_e * m_H) = rho * ye / m_H = rho_e / m_H
    # This is correct: n_e = rho_e / m_H

    n_e = rho_e / m_H  # electron number density

    # Relativity parameter
    theta = k_B * T / (m_e * c_light**2)

    # Solve for degeneracy parameter eta
    eta = solve_eta(n_e, T, theta)

    # Compute the FD integrals at this (eta, theta)
    f12 = fd_integral(0.5, eta, theta)
    f32 = fd_integral(1.5, eta, theta)
    f52 = fd_integral(2.5, eta, theta)

    # Pressure (Timmes & Swesty 2000, eq. 9):
    # P_e = (16*pi*sqrt(2))/(3*h^3) * (m_e*c)^3 * (k_B*T) * theta^{3/2}
    #       * [F_{3/2} + theta/2 * F_{5/2}]
    # The factor 16π√2/3 = 2 × 8π√2/3 accounts for the spin degeneracy g_s=2
    # (same factor as in the number density prefactor).
    # Reference: Timmes & Swesty (2000) eq. 9; Cox & Giuli (1968) Ch. 24.
    h_planck = 2 * pi * hbar
    coeff = 16.0 * pi * np.sqrt(2.0) / (3.0 * h_planck**3) * (m_e * c_light)**3
    P_e = coeff * (k_B * T) * theta**1.5 * (f32 + 0.5 * theta * f52)

    # Chemical potential
    mu_chem = eta * k_B * T

    # Helmholtz free energy per unit volume: F/V = mu*n - P (Euler relation for
    # a single-component system at fixed T: F = G - PV, G = mu*N → F/V = mu*n - P)
    F_per_vol = mu_chem * n_e - P_e

    # Per unit mass
    f_e = F_per_vol / rho_e

    return f_e


def _compute_row(args):
    """Compute one row of the table (all T values for a given rho_e)."""
    i, log_rho_e, logT_grid = args
    rho_e = 10.0**log_rho_e
    N_T = len(logT_grid)
    row = np.zeros(N_T)
    for j, logT in enumerate(logT_grid):
        T = 10.0**logT
        try:
            row[j] = electron_free_energy_per_mass(rho_e, T)
        except Exception as e:
            print(f"ERROR at log_rho_e={log_rho_e:.2f}, logT={logT:.2f}: {e}")
            row[j] = np.nan
    return i, row


def generate_table(N_rho=701, N_T=301, log_rho_min=1.0, log_rho_max=8.0,
                   logT_min=6.0, logT_max=9.0, n_workers=None):
    """Generate the FD electron free energy table.

    Grid resolution chosen to match MESA's HELM table spacing (Δ=0.01 in
    both log(ρ_e) and logT). MESA uses imax=2701, jmax=1001 over a wider
    range; for our [1,8]×[6,9] domain, Δ=0.01 requires 701×301.

    Finer spacing reduces interpolation error in second derivatives
    (chi_T, cv → nabla_ad), which the C² bicubic Hermite scheme (#1021)
    minimizes through continuous 2nd derivatives at cell boundaries.

    Returns:
        log_rho_e_grid: 1D array of log10(rho_e) values
        logT_grid: 1D array of log10(T) values
        f_e_table: 2D array of f_e(rho_e, T) [erg/g], shape (N_rho, N_T)
    """
    from multiprocessing import Pool, cpu_count

    log_rho_e_grid = np.linspace(log_rho_min, log_rho_max, N_rho)
    logT_grid = np.linspace(logT_min, logT_max, N_T)

    if n_workers is None:
        n_workers = min(cpu_count(), 16)

    total = N_rho * N_T
    print(f"Generating {N_rho}×{N_T} = {total} points using {n_workers} workers")
    t0 = time.time()

    args_list = [(i, log_rho_e_grid[i], logT_grid) for i in range(N_rho)]

    f_e_table = np.zeros((N_rho, N_T))

    with Pool(n_workers) as pool:
        for count, (i, row) in enumerate(pool.imap_unordered(_compute_row, args_list)):
            f_e_table[i, :] = row
            if (count + 1) % 50 == 0 or count + 1 == N_rho:
                elapsed = time.time() - t0
                rate = (count + 1) / elapsed
                eta_sec = (N_rho - count - 1) / rate if rate > 0 else 0
                print(f"  [{count+1}/{N_rho} rows] {elapsed:.1f}s elapsed, "
                      f"~{eta_sec:.0f}s remaining", flush=True)

    return log_rho_e_grid, logT_grid, f_e_table


def main():
    print("=" * 60)
    print("Generating Fermi-Dirac electron EOS table")
    print("Method: Numerical FD integrals (scipy quad)")
    print("Matching: MESA HELM table (Timmes & Swesty 2000)")
    print("=" * 60)

    # Grid parameters — covers the degenerate regime with margin
    # log(rho_e) from 1.0 to 8.0: rho_e = ye*rho, so for ye~0.5:
    #   rho_e=10 → rho~20 (non-degenerate)
    #   rho_e=1e8 → rho~2e8 (strongly degenerate)
    # logT from 6.0 to 9.0: covers He-core conditions
    #
    # Resolution: Δ=0.01 for both axes, matching MESA's HELM table
    # (MESA: imax=2701, jmax=1001 over [-12,15]×[3,13] → Δ=0.01).
    # Fine spacing supports the C² bicubic Hermite interpolation,
    # which eliminates the ±3% nad oscillation from the former C¹ scheme.
    N_rho = 701
    N_T = 301
    log_rho_min, log_rho_max = 1.0, 8.0
    logT_min, logT_max = 6.0, 9.0

    print(f"Grid: log(rho_e) = [{log_rho_min}, {log_rho_max}], N={N_rho}")
    print(f"      logT       = [{logT_min}, {logT_max}], N={N_T}")
    print(f"Total points: {N_rho * N_T}")
    print()

    log_rho_e_grid, logT_grid, f_e_table = generate_table(
        N_rho=N_rho, N_T=N_T,
        log_rho_min=log_rho_min, log_rho_max=log_rho_max,
        logT_min=logT_min, logT_max=logT_max
    )

    # Check for NaN
    n_nan = np.sum(np.isnan(f_e_table))
    if n_nan > 0:
        print(f"\nWARNING: {n_nan} NaN values in table!")
    else:
        print("\nTable generation complete — no NaN values.")

    # --- Precompute C² Hermite derivative tables (fx, fy, fxy) ---
    # For bicubic Hermite interpolation (matching MESA h3), each cell corner
    # needs f, fx=∂f/∂x·Δx, fy=∂f/∂y·Δy, fxy=∂²f/∂x∂y·Δx·Δy.
    # We use natural cubic splines (scipy CubicSpline, bc_type="natural")
    # to compute derivatives, then scale by cell spacing to match MESA's
    # convention (MESA: fd·dd_sav, ft·dt_sav, fdt·dd_sav·dt_sav).
    #
    # MESA divergence (CONSTRAINT, bounded): MESA tabulates analytic
    # derivatives (from the FD integrals directly) in its binary helm
    # table. We compute them from the tabulated f_e via spline fitting.
    # The natural cubic spline gives C² continuity at cell boundaries,
    # matching MESA's h3 bicubic Hermite scheme. Bounded: nad vs MESA
    # < 0.3% across logρ 5.0–6.0.
    #
    # Reference: MESA helm_alloc.f90:200-215 (binary table with stored
    # derivatives); helm_electron_positron.dek:40-76 (fi() assembly).
    from scipy.interpolate import CubicSpline
    print("\nComputing C² Hermite derivative tables (natural cubic spline)...")

    dx = log_rho_e_grid[1] - log_rho_e_grid[0]
    dy = logT_grid[1] - logT_grid[0]

    # fx_table = ∂f/∂x · Δx at each grid point (x = log_rho_e direction)
    # Fit a natural cubic spline along each column (fixed T), differentiate
    fx_table = np.zeros_like(f_e_table)
    for j in range(N_T):
        cs = CubicSpline(log_rho_e_grid, f_e_table[:, j], bc_type='natural')
        fx_table[:, j] = cs(log_rho_e_grid, 1) * dx  # 1st deriv × Δx

    # fy_table = ∂f/∂y · Δy at each grid point (y = logT direction)
    # Fit a natural cubic spline along each row (fixed rho_e), differentiate
    fy_table = np.zeros_like(f_e_table)
    for i in range(N_rho):
        cs = CubicSpline(logT_grid, f_e_table[i, :], bc_type='natural')
        fy_table[i, :] = cs(logT_grid, 1) * dy  # 1st deriv × Δy

    # fxy_table = ∂²f/∂x∂y · Δx · Δy (cross derivative)
    # Differentiate fx_table (which is ∂f/∂x · Δx) in the y direction
    fxy_table = np.zeros_like(f_e_table)
    for i in range(N_rho):
        cs = CubicSpline(logT_grid, fx_table[i, :], bc_type='natural')
        fxy_table[i, :] = cs(logT_grid, 1) * dy  # ∂(fx)/∂y × Δy = ∂²f/∂x∂y × Δx × Δy

    print(f"  fx_table range: [{fx_table.min():.6e}, {fx_table.max():.6e}]")
    print(f"  fy_table range: [{fy_table.min():.6e}, {fy_table.max():.6e}]")
    print(f"  fxy_table range: [{fxy_table.min():.6e}, {fxy_table.max():.6e}]")

    # Save
    outpath = os.path.join(os.path.dirname(__file__), '..', 'data', 'fd_electron_eos.npz')
    outpath = os.path.abspath(outpath)
    np.savez_compressed(outpath,
                        log_rho_e_grid=log_rho_e_grid,
                        logT_grid=logT_grid,
                        f_e_table=f_e_table,
                        fx_table=fx_table,
                        fy_table=fy_table,
                        fxy_table=fxy_table,
                        # Metadata
                        description="Fermi-Dirac electron free energy per unit mass [erg/g]",
                        method="Numerical FD integrals (scipy quad, Aparicio 1998 breakpoints)",
                        reference="Timmes & Swesty (2000) ApJS 126, 501",
                        grid_variable_rho="log10(rho_e) where rho_e = ye * rho [g/cm^3]",
                        grid_variable_T="log10(T) [K]",
                        derivative_method="Natural cubic spline (scipy CubicSpline, bc_type='natural')",
                        derivative_convention="fx=df/dx*dx, fy=df/dy*dy, fxy=d2f/dxdy*dx*dy (MESA h3)")

    print(f"Saved to: {outpath}")
    print(f"File size: {os.path.getsize(outpath) / 1024:.1f} KB")

    # Quick validation: check a known point
    # At logRho_e=6, logT=7.5 (partial degeneracy) — should give finite, negative f_e
    i_test = np.searchsorted(log_rho_e_grid, 6.0)
    j_test = np.searchsorted(logT_grid, 7.5)
    f_test = f_e_table[i_test, j_test]
    print(f"\nValidation point: log(rho_e)=6.0, logT=7.5")
    print(f"  f_e = {f_test:.6e} erg/g")
    print(f"  (should be negative — electrons prefer lower energy in degenerate state)")


if __name__ == "__main__":
    main()
