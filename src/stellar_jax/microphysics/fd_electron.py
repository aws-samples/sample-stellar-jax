"""Fermi-Dirac electron EOS via pre-computed table + bicubic Hermite interpolation.

Provides a full Fermi-Dirac electron EOS that is:
  1. Accurate at ALL degeneracy levels (partial AND strong) — no Sommerfeld expansion
  2. Fully JAX-differentiable (autodiff through the spline interpolation)
  3. Thermodynamically consistent (all quantities derived from a single free energy)

The table stores the electron Helmholtz free energy per unit mass f_e(rho_e, T) computed
via numerical evaluation of relativistic Fermi-Dirac integrals (Timmes & Swesty 2000;
Aparicio 1998). Thermodynamic quantities are derived via JAX autodiff:
  P_e = rho_e^2 * df_e/d(rho_e)   [pressure]
  S_e = -df_e/dT                    [entropy]
guaranteeing Maxwell-relation consistency by construction.

Interpolation uses a C² bicubic Hermite scheme matching MESA's h3 function
(helm_polynomials.f90:197): at each cell corner, the table stores f, fx=∂f/∂x·Δx,
fy=∂f/∂y·Δy, fxy=∂²f/∂x∂y·Δx·Δy (precomputed via natural cubic spline,
scipy.interpolate.CubicSpline bc_type="natural"). The 16-term tensor-product
Hermite basis (xpsi0/xpsi1, helm_polynomials.f90:127–151) guarantees C²
continuity at cell boundaries — eliminating the ±3% nabla_ad oscillation that
the former Catmull-Rom (C¹) scheme produced.

MESA divergence (CONSTRAINT, bounded): MESA uses biquintic Hermite (h5, 36 terms)
for the main free energy table; we use bicubic (h3, 16 terms). The quintic form
additionally matches 2nd derivatives at corners, giving C⁴ continuity. Our C²
bicubic is sufficient because JAX autodiff computes thermodynamic derivatives
analytically through the Hermite basis (not from the stored derivative arrays),
so the stored 1st derivatives serve only as interpolation coefficients. Bounded:
nad vs MESA < 0.3% across logρ 5.0–6.0 (research #1018).

Table grid:
  - log(rho_e) from 1.0 to 8.0 (rho_e = ye * rho)
  - logT from 6.0 to 9.0
  - Resolution: 701 x 301 (uniform in log space, Δlog ρ_e = Δlog T = 0.01,
    matching MESA's HELM table spacing)

References:
  - Timmes & Swesty (2000), ApJS 126, 501 (HELM EOS — table method)
  - Aparicio (1998), ApJS 117, 627 (FD integral evaluation)
  - Yakovlev & Shalybkov (1989), Soviet Scientific Reviews E 7, 311 (Coulomb)
  - Chandrasekhar (1939), Introduction to Stellar Structure, Ch. 10
  - Cox & Giuli (1968), Principles of Stellar Structure, Ch. 24
  - MESA: mesa/eos/private/helm_polynomials.f90 (h3 bicubic Hermite, xpsi0/xpsi1)
  - MESA: mesa/eos/private/helm_electron_positron.dek (table lookup + assembly)
  - MESA: mesa/eos/private/helm_coulomb2.dek (Coulomb correction)
"""
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np


# --- Physical constants (CGS) — imported from single source ---
from stellar_jax.config.constants import (
    k_B_fd as k_B, m_H_fd as m_H,
    hbar_fd as hbar,
    e_charge_fd as e_charge, a_rad_fd as a_rad,
)


# --- Table loading (at import time) ---
def _init_table():
    """Load the FD table with precomputed C² Hermite derivatives.

    The table stores f_e plus three derivative arrays (fx, fy, fxy)
    precomputed via natural cubic spline (scipy CubicSpline, bc_type="natural")
    and scaled by cell spacing to match MESA's h3 convention:
      fx_table = ∂f/∂x · Δx   (MESA: fd · dd_sav)
      fy_table = ∂f/∂y · Δy   (MESA: ft · dt_sav)
      fxy_table = ∂²f/∂x∂y · Δx · Δy   (MESA: fdt · dd_sav · dt_sav)

    MESA reference: helm_alloc.f90:200-215 (binary table read);
    helm_electron_positron.dek:40-76 (fi() corner data assembly).
    """
    table_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                              'fd_electron_eos.npz')
    table_path = os.path.abspath(table_path)
    if not os.path.exists(table_path):
        _empty = (None, None, None, None, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  False)
        return _empty
    data = np.load(table_path)
    log_rho_e_grid = data['log_rho_e_grid']
    logT_grid = data['logT_grid']
    f_e_table = data['f_e_table']

    # Grid parameters as plain Python scalars (never traced by JAX)
    nx = int(log_rho_e_grid.shape[0])
    ny = int(logT_grid.shape[0])
    dx = float(log_rho_e_grid[1] - log_rho_e_grid[0])
    dy = float(logT_grid[1] - logT_grid[0])
    x0 = float(log_rho_e_grid[0])
    y0 = float(logT_grid[0])
    x_max = float(log_rho_e_grid[-1])
    y_max = float(logT_grid[-1])

    # Tables as JAX arrays (static constants, not traced)
    table_jax = jnp.array(f_e_table, dtype=jnp.float64)

    # Derivative arrays for C² bicubic Hermite (scaled by cell spacing)
    fx_jax = jnp.array(data['fx_table'], dtype=jnp.float64)
    fy_jax = jnp.array(data['fy_table'], dtype=jnp.float64)
    fxy_jax = jnp.array(data['fxy_table'], dtype=jnp.float64)

    return (table_jax, fx_jax, fy_jax, fxy_jax,
            nx, ny, dx, dy, x0, y0, x_max, y_max, True)


# Module-level constants — loaded once at import
(_FD_TABLE, _FX_TABLE, _FY_TABLE, _FXY_TABLE,
 _NX, _NY, _DX, _DY, _X0, _Y0,
 _X_MAX, _Y_MAX, _TABLE_AVAILABLE) = _init_table()


# --- Bicubic Hermite interpolation (C², JAX-differentiable) ---
def _bicubic_hermite_interp(log_rho_e, logT):
    """C² bicubic Hermite interpolation of the electron free energy table.

    Matches MESA's h3 function (helm_polynomials.f90:197): 16-term tensor
    product of cubic Hermite basis functions xpsi0/xpsi1 at the 4 cell
    corners, using precomputed derivatives (fx, fy, fxy) from natural cubic
    splines. Gives C² continuity at cell boundaries — 2nd derivatives are
    continuous, eliminating the ±3% nad oscillation from the former
    Catmull-Rom (C¹) scheme.

    MESA reference:
      - helm_polynomials.f90:127-151 (xpsi0, xpsi1 basis functions)
      - helm_polynomials.f90:197-212 (h3 assembly)
      - helm_electron_positron.dek:140-180 (offset + weight computation)
      - helm_electron_positron.dek:255-330 (bicubic auxiliary tables)

    Parameters:
        log_rho_e: log10(rho_e) [scalar, traced]
        logT: log10(T) [scalar, traced]

    Returns:
        Interpolated f_e [erg/g]
    """
    # Normalized coordinates (constants are Python floats — never traced)
    sx = (log_rho_e - _X0) / _DX
    sy = (logT - _Y0) / _DY

    # Integer indices (clamped so 2-point stencil stays in-bounds)
    # Hermite uses 2x2 stencil (ix, ix+1) × (iy, iy+1), not 4x4
    ix = jnp.clip(jnp.floor(sx).astype(jnp.int32), 0, _NX - 2)
    iy = jnp.clip(jnp.floor(sy).astype(jnp.int32), 0, _NY - 2)

    # Fractional parts within the cell: xt, yt ∈ [0, 1]
    # MESA: helm_electron_positron.dek:140-146
    xt = sx - ix.astype(jnp.float64)
    yt = sy - iy.astype(jnp.float64)

    # --- Cubic Hermite basis functions (MESA helm_polynomials.f90:127-151) ---
    # xpsi0(z) = z²(2z - 3) + 1   [value basis, node 0]
    # xpsi1(z) = z(z(z - 2) + 1)  [derivative basis, node 0]
    # At z=0: xpsi0=1, xpsi1=0; at z=1: xpsi0=0, xpsi1=0
    # xpsi0(1-z) is the value basis for node 1
    # -xpsi1(1-z) is the derivative basis for node 1 (sign flip for chain rule)

    # Density direction (x = log_rho_e)
    mxt = 1.0 - xt
    # MESA: helm_electron_positron.dek:260-266
    w0d = xt * xt * (2.0 * xt - 3.0) + 1.0       # xpsi0(xt)  → node ix
    w1d = xt * (xt * (xt - 2.0) + 1.0)            # xpsi1(xt)  → deriv at ix
    w0md = mxt * mxt * (2.0 * mxt - 3.0) + 1.0    # xpsi0(mxt) → node ix+1
    w1md = -(mxt * (mxt * (mxt - 2.0) + 1.0))     # -xpsi1(mxt) → deriv at ix+1
    # Note: MESA scales w1d by dd_sav and w1md by -dd_sav (helm_electron_positron.dek:261-266).
    # We absorb the dd_sav into the stored derivative arrays (fx_table = ∂f/∂x · Δx).

    # Temperature direction (y = logT)
    myt = 1.0 - yt
    # MESA: helm_electron_positron.dek:255-258
    w0t = yt * yt * (2.0 * yt - 3.0) + 1.0        # xpsi0(yt)
    w1t = yt * (yt * (yt - 2.0) + 1.0)             # xpsi1(yt)
    w0mt = myt * myt * (2.0 * myt - 3.0) + 1.0     # xpsi0(myt)
    w1mt = -(myt * (myt * (myt - 2.0) + 1.0))      # -xpsi1(myt)

    # --- Gather corner data (2×2 patch) via dynamic_slice ---
    # MESA: helm_electron_positron.dek:40-76 loads fi(1..16) from 4 corners.
    # Corners: (ix,iy), (ix+1,iy), (ix,iy+1), (ix+1,iy+1)
    f00 = jax.lax.dynamic_slice(_FD_TABLE, (ix, iy), (1, 1))[0, 0]
    f10 = jax.lax.dynamic_slice(_FD_TABLE, (ix + 1, iy), (1, 1))[0, 0]
    f01 = jax.lax.dynamic_slice(_FD_TABLE, (ix, iy + 1), (1, 1))[0, 0]
    f11 = jax.lax.dynamic_slice(_FD_TABLE, (ix + 1, iy + 1), (1, 1))[0, 0]

    fx00 = jax.lax.dynamic_slice(_FX_TABLE, (ix, iy), (1, 1))[0, 0]
    fx10 = jax.lax.dynamic_slice(_FX_TABLE, (ix + 1, iy), (1, 1))[0, 0]
    fx01 = jax.lax.dynamic_slice(_FX_TABLE, (ix, iy + 1), (1, 1))[0, 0]
    fx11 = jax.lax.dynamic_slice(_FX_TABLE, (ix + 1, iy + 1), (1, 1))[0, 0]

    fy00 = jax.lax.dynamic_slice(_FY_TABLE, (ix, iy), (1, 1))[0, 0]
    fy10 = jax.lax.dynamic_slice(_FY_TABLE, (ix + 1, iy), (1, 1))[0, 0]
    fy01 = jax.lax.dynamic_slice(_FY_TABLE, (ix, iy + 1), (1, 1))[0, 0]
    fy11 = jax.lax.dynamic_slice(_FY_TABLE, (ix + 1, iy + 1), (1, 1))[0, 0]

    fxy00 = jax.lax.dynamic_slice(_FXY_TABLE, (ix, iy), (1, 1))[0, 0]
    fxy10 = jax.lax.dynamic_slice(_FXY_TABLE, (ix + 1, iy), (1, 1))[0, 0]
    fxy01 = jax.lax.dynamic_slice(_FXY_TABLE, (ix, iy + 1), (1, 1))[0, 0]
    fxy11 = jax.lax.dynamic_slice(_FXY_TABLE, (ix + 1, iy + 1), (1, 1))[0, 0]

    # --- 16-term bicubic Hermite (MESA h3, helm_polynomials.f90:205-212) ---
    # h3 = Σ fi(k) · w_d(k) · w_t(k)
    # Mapping to MESA fi() indices:
    #   fi(1-4)   = f   at corners  → f00, f10, f01, f11
    #   fi(5-8)   = ft  at corners  → fy00, fy10, fy01, fy11  (scaled by dy)
    #   fi(9-12)  = fd  at corners  → fx00, fx10, fx01, fx11  (scaled by dx)
    #   fi(13-16) = fdt at corners  → fxy00, fxy10, fxy01, fxy11 (scaled by dx·dy)
    result = (f00  * w0d * w0t  + f10  * w0md * w0t
            + f01  * w0d * w0mt + f11  * w0md * w0mt
            + fy00 * w0d * w1t  + fy10 * w0md * w1t
            + fy01 * w0d * w1mt + fy11 * w0md * w1mt
            + fx00 * w1d * w0t  + fx10 * w1md * w0t
            + fx01 * w1d * w0mt + fx11 * w1md * w0mt
            + fxy00 * w1d * w1t  + fxy10 * w1md * w1t
            + fxy01 * w1d * w1mt + fxy11 * w1md * w1mt)

    return result


# --- Free energy components ---
def _F_electron_fd(ln_rho, ln_T, X, Z):
    """Electron free energy per unit mass from the FD table [erg/g].

    Drop-in replacement for helm._F_electron in the partial degeneracy regime.
    """
    if not _TABLE_AVAILABLE:
        raise RuntimeError("FD electron EOS table not loaded. "
                           "Run scripts/generate_fd_electron_table.py first.")

    rho = jnp.exp(ln_rho)
    T = jnp.exp(ln_T)

    # Electron mass density: rho_e = ye * rho
    mu_e = 2.0 / (1.0 + X)
    ye = 1.0 / mu_e
    rho_e = ye * rho

    # Convert to log10 for table lookup
    log_rho_e = jnp.log10(rho_e)
    log_T = jnp.log10(T)

    # Clamp to table interior (Hermite uses 2-point stencil, so the
    # outermost grid cell is the boundary; clamp to [x0, x_max - dx] for
    # the ix index to stay in [0, NX-2])
    log_rho_e = jnp.clip(log_rho_e, _X0, _X_MAX - _DX)
    log_T = jnp.clip(log_T, _Y0, _Y_MAX - _DY)

    # Table stores f_e per unit electron mass density (per rho_e).
    # Convert to per total mass: F_e/rho = (F_e/rho_e) * (rho_e/rho) = f_table * ye
    f_e_per_rho_e = _bicubic_hermite_interp(log_rho_e, log_T)
    return f_e_per_rho_e * ye


def _F_ion(ln_rho, ln_T, X, Z):
    """Ion free energy per unit mass [erg/g] — classical ideal gas.
    Reference: KWW (2012) §13.1
    """
    rho = jnp.exp(ln_rho)
    T = jnp.exp(ln_T)
    Y = 1.0 - X - Z
    A_bar = 1.0 / (X + Y / 4.0 + Z / 16.0)
    n_ion = rho / (A_bar * m_H)
    m_ion = A_bar * m_H
    h_planck = 2.0 * jnp.pi * hbar
    nQ_ion = (2.0 * jnp.pi * m_ion * k_B * T / h_planck**2) ** 1.5
    ln_ratio = jnp.log(jnp.maximum(n_ion / nQ_ion, 1e-30))
    return (k_B * T / (A_bar * m_H)) * (ln_ratio - 1.0)


def _F_rad(ln_rho, ln_T, X, Z):
    """Radiation free energy per unit mass [erg/g].
    Reference: KWW (2012) §13.2
    """
    rho = jnp.exp(ln_rho)
    T = jnp.exp(ln_T)
    return -a_rad * T**4 / (3.0 * rho)


def _F_coul(ln_rho, ln_T, X, Z):
    """Coulomb free energy per unit mass [erg/g].

    Implements MESA's Yakovlev & Shalybkov (1989) uniform background Coulomb
    correction, matching helm_coulomb2.dek exactly. Two regimes (Gamma >= 1:
    solid-like, eq. 82/85/86/87; Gamma < 1: liquid/Debye-Hückel, eq. 102-104).

    The free energy is F = E - T*S where E (ecoul) and S (scoul) use the
    Y&S functionals with coefficients from helm_declare_local_variables.dek.

    Reference: Yakovlev & Shalybkov (1989), Soviet Scientific Reviews E 7, 311
    MESA source: mesa/eos/private/helm_coulomb2.dek
    MESA coefficients: mesa/eos/private/helm_declare_local_variables.dek:242-249
    """
    rho = jnp.exp(ln_rho)
    T = jnp.exp(ln_T)

    # Yakovlev & Shalybkov coefficients (MESA helm_declare_local_variables.dek:242-249)
    a1 = -0.898004
    b1 = 0.96786
    c1 = 0.220703
    d1 = -0.86097
    e1 = 2.5269
    a2 = 0.29561
    b2 = 1.9885
    c2 = 0.288675   # = sqrt(3)/6, Debye-Hückel limit

    # Composition
    Y_comp = 1.0 - X - Z
    A_bar = 1.0 / (X + Y_comp / 4.0 + Z / 16.0)
    mu_e = 2.0 / (1.0 + X)
    Z_bar = A_bar / mu_e

    # Ion number density: n_i = rho / (A_bar * m_H)
    # (MESA: xni = avo * ytot1 * den, where ytot1=1/abar, and avo*amu≈1)
    n_i = rho / (A_bar * m_H)

    # Electron-sphere radius from ion density (MESA: s = forthpi*zbar*xni;
    # aele = s^(-1/3)), which gives a_e = (3/(4π*Z_bar*n_i))^(1/3)
    a_e = (3.0 / (4.0 * jnp.pi * Z_bar * n_i)) ** (1.0 / 3.0)

    # Electron coupling parameter: eplasg = e²/(a_e * kT)
    eplasg = e_charge**2 / (a_e * k_B * T)

    # Ion coupling parameter: Gamma = Z_bar^(5/3) * eplasg
    Gamma = Z_bar ** (5.0 / 3.0) * eplasg
    Gamma_safe = jnp.maximum(Gamma, 1e-30)

    # --- Gamma >= 1 regime (solid-like): Y&S eq. 82, 85-87 ---
    # Energy per ion / kT: u0_e = a1*G + b1*G^(1/4) + c1*G^(-1/4) + d1
    p1_solid = Gamma_safe ** 0.25    # G^(1/4)
    p2_solid = Gamma_safe ** (-0.25)  # G^(-1/4)
    u0_e_solid = a1 * Gamma_safe + b1 * p1_solid + c1 * p2_solid + d1

    # Entropy per ion / k_B: u0_s = 3*b1*G^(1/4) - 5*c1*G^(-1/4) + d1*(ln(G)-1) - e1
    u0_s_solid = (3.0 * b1 * p1_solid - 5.0 * c1 * p2_solid
                  + d1 * (jnp.log(Gamma_safe) - 1.0) - e1)

    # Free energy = Energy - T*Entropy (per ion / kT)
    # F/(n_i*kT) = u0_e - (-u0_s) = u0_e + u0_s
    # (scoul = -k_B/(Abar*m_H) * u0_s → S = -k_B/(Am_H)*u0_s → -TS/kT = u0_s per ion)
    u0_f_solid = u0_e_solid + u0_s_solid

    # --- Gamma < 1 regime (liquid/Debye): Y&S eq. 102-104 ---
    # Pressure functional: pcoul = -pion * u0_p
    # where u0_p = c2*G^(3/2) - (1/3)*a2*G^b2
    p1_liquid = Gamma_safe ** 1.5    # G^(3/2)
    p2_liquid = Gamma_safe ** b2     # G^b2

    # Energy: ecoul = 3*pcoul/den = -3*(kT/(Abar*m_H))*(c2*G^(3/2) - (1/3)*a2*G^b2)
    u0_e_liquid = -(3.0 * c2 * p1_liquid - a2 * p2_liquid)

    # Entropy functional: u0_s = c2*G^(3/2) - a2/b2*(b2-1)*G^b2
    u0_s_liquid = c2 * p1_liquid - a2 / b2 * (b2 - 1.0) * p2_liquid

    # Free energy: F = E - TS → u0_f = u0_e + u0_s (same sign convention)
    u0_f_liquid = u0_e_liquid + u0_s_liquid

    # Smooth piecewise selection (JAX-differentiable via jnp.where)
    u0_f = jnp.where(Gamma_safe >= 1.0, u0_f_solid, u0_f_liquid)

    # Free energy per unit mass: F_coul = (kT/(Abar*m_H)) * u0_f
    return (k_B * T / (A_bar * m_H)) * u0_f


# --- Total free energy and thermodynamic functions ---
def _F_total_fd(ln_rho, ln_T, X, Z):
    """Total Helmholtz free energy per unit mass [erg/g] using FD electron table."""
    return (_F_electron_fd(ln_rho, ln_T, X, Z) +
            _F_ion(ln_rho, ln_T, X, Z) +
            _F_rad(ln_rho, ln_T, X, Z) +
            _F_coul(ln_rho, ln_T, X, Z))


def _total_pressure_fd(ln_rho, ln_T, X, Z):
    """Total pressure: P = rho * dF/d(ln_rho)."""
    rho = jnp.exp(ln_rho)
    dF_dlnrho = jax.grad(_F_total_fd, argnums=0)(ln_rho, ln_T, X, Z)
    return rho * dF_dlnrho


def _total_entropy_fd(ln_rho, ln_T, X, Z):
    """Total specific entropy: S = -(1/T)*dF/d(ln_T)."""
    T = jnp.exp(ln_T)
    dF_dlnT = jax.grad(_F_total_fd, argnums=1)(ln_rho, ln_T, X, Z)
    return jnp.maximum(-dF_dlnT / T, 1e-10)


# --- Density inversion (P → rho) ---
# NOTE: These functions (_solve_density_fd, _solve_density_fd_ift, helm_eos_fd)
# are kept INLINE (not delegated to density_inversion.py) to preserve XLA trace
# identity on the Newton per-step hot path. helm_eos_fd is called via eos_lookup
# at EVERY Newton iteration during RGB evolution (logT>7.7, logRho>5.9 activates
# the HELM blend). The adaptive forward RGB tests accumulate float error over
# thousands of steps and are sensitive to XLA graph restructuring (~0.02-0.03 dex
# shift). Inlining ensures bit-identical HLO vs the pre-refactoring code.

# Convergence tolerance for Newton-Raphson density inversion.
# MESA reference: eosdt_eval.f90:2359 do_safe_get_Rho_T uses logRho_tol=1e-8
# (in log10 space ≈ 2.3e-8 in ln space). Our tolerance is |Δln_ρ| per step,
# set tighter at 1e-10 to guarantee bit-level convergence for the IFT gradient.
_NR_TOL = 1e-10
_NR_MAX_ITER = 30  # Safety cap (MESA uses max_iter=20)


def _solve_density_fd(logT, logP_target, X, Z, ln_rho_guess=None):
    """Newton-Raphson density inversion using the FD table.

    Uses lax.while_loop that terminates on convergence (|Δln_ρ| < tol)
    instead of a fixed iteration count. Newton converges quadratically
    from a good guess — typically 6-8 iterations from ideal-gas, 2-4
    from a warm-start guess.

    MESA reference: eosdt_eval.f90:2359 do_safe_get_Rho_T —
    safe_root_with_guess with convergence tolerance, max_iter=20.

    Parameters
    ----------
    logT : scalar — log10(T) [K]
    logP_target : scalar — log10(P_target) [dyn/cm²]
    X : scalar — hydrogen mass fraction
    Z : scalar — metals mass fraction
    ln_rho_guess : scalar or None — initial guess for ln(rho) [g/cm³].
        If None, uses the ideal-gas estimate. When the previous step's
        converged density is available, passing it here reduces iterations
        from ~7 to ~3 (warm-start, matching MESA's s%lnd(k)/ln10 pattern).
    """
    T = 10.0**logT
    P_target = 10.0**logP_target
    ln_T = jnp.log(T)

    mu_e = 2.0 / (1.0 + X)
    Y = 1.0 - X - Z
    A_bar = 1.0 / (X + Y / 4.0 + Z / 16.0)
    mu = 1.0 / (1.0 / mu_e + 1.0 / A_bar)

    # Initial guess: use provided warm-start or fall back to ideal gas.
    # MESA: micro.f90:333 passes s%lnd(k)/ln10 as logRho_guess (previous
    # step converged density). Our ideal-gas fallback matches MESA's
    # eospt_eval.f90:186: rho_guess = Pgas*abar*mp/(kerg*T*(1+zbar)).
    if ln_rho_guess is None:
        rho_ideal = P_target * mu * m_H / (k_B * T)
        ln_rho_init = jnp.log(jnp.maximum(rho_ideal, 1e-10))
    else:
        ln_rho_init = ln_rho_guess
    ln_rho_init = jnp.clip(ln_rho_init, jnp.log(1e-2), jnp.log(1e10))

    _dP_dlnrho = jax.grad(_total_pressure_fd, argnums=0)

    # while_loop state: (iteration_count, ln_rho, converged)
    # Convergence criterion: |Δln_ρ| < tol (step size in ln-space).
    # This matches MESA's logRho_tol criterion (eosdt_eval.f90:2383 xacc)
    # and avoids an extra pressure evaluation per iteration.
    def _cond(state):
        i, _, converged = state
        return jnp.logical_and(i < _NR_MAX_ITER, jnp.logical_not(converged))

    def _body(state):
        i, ln_rho_i, _ = state
        P_i = _total_pressure_fd(ln_rho_i, ln_T, X, Z)
        dP = _dP_dlnrho(ln_rho_i, ln_T, X, Z)
        dP_safe = jnp.where(jnp.abs(dP) > 1e-30, dP, 1e-30)
        delta = jnp.clip(-(P_i - P_target) / dP_safe, -2.0, 2.0)
        ln_rho_new = ln_rho_i + delta
        # Convergence: |Δln_ρ| < tol (quadratic convergence → step size
        # drops below tol iff the root is found to that precision)
        converged = jnp.abs(delta) < _NR_TOL
        return (i + 1, ln_rho_new, converged)

    init_state = (jnp.int32(0), ln_rho_init, jnp.bool_(False))
    _, ln_rho_final, _ = jax.lax.while_loop(_cond, _body, init_state)
    return ln_rho_final


@functools.partial(jax.custom_jvp, nondiff_argnums=())
def _solve_density_fd_ift(logT, logP_target, X, Z, ln_rho_guess):
    """Density inversion with IFT-based custom_jvp (FD table version).

    The ln_rho_guess parameter enables warm-start iteration (fewer Newton
    steps) without affecting the converged result or gradient. The gradient
    is computed analytically via the implicit function theorem — it depends
    only on the CONVERGED root, not on how many iterations were taken to
    reach it.
    """
    return _solve_density_fd(logT, logP_target, X, Z, ln_rho_guess)


@_solve_density_fd_ift.defjvp
def _solve_density_fd_ift_jvp(primals, tangents):
    """Forward-mode IFT rule for density inversion (FD table version).

    The gradient is independent of the Newton iteration path — it uses the
    implicit function theorem at the converged root. ln_rho_guess affects
    only convergence speed, not the derivative.
    """
    logT, logP_target, X, Z, ln_rho_guess = primals
    d_logT, d_logP, d_X, d_Z, _d_guess = tangents

    ln_rho = _solve_density_fd_ift(logT, logP_target, X, Z, ln_rho_guess)

    T = 10.0**logT
    ln_T = jnp.log(T)
    P_target = 10.0**logP_target

    dP_dlnrho = jax.grad(_total_pressure_fd, argnums=0)(ln_rho, ln_T, X, Z)
    dP_dlnT = jax.grad(_total_pressure_fd, argnums=1)(ln_rho, ln_T, X, Z)
    dP_dX = jax.grad(_total_pressure_fd, argnums=2)(ln_rho, ln_T, X, Z)
    dP_dZ = jax.grad(_total_pressure_fd, argnums=3)(ln_rho, ln_T, X, Z)

    dP_dlnrho_safe = jnp.where(jnp.abs(dP_dlnrho) > 1e-30, dP_dlnrho, 1e-30)

    d_ln_rho = (P_target * jnp.log(10.0) * d_logP
                - dP_dlnT * jnp.log(10.0) * d_logT
                - dP_dX * d_X
                - dP_dZ * d_Z) / dP_dlnrho_safe

    return ln_rho, d_ln_rho


# --- Public API ---
def helm_eos_fd(logT, logP_target, X, Z, ln_rho_guess=None):
    """Full Fermi-Dirac HELM EOS with P-based lookup.

    Drop-in replacement for helm.helm_eos_full that uses the FD electron
    table instead of the analytic Chandrasekhar+Sommerfeld approximation.
    Accurate at ALL degeneracy levels including partial degeneracy
    (logRho 5.0-5.5, Theta 0.1-0.2) where the analytic form fails.

    Parameters
    ----------
    logT : scalar — log10(T) [K]
    logP_target : scalar — log10(P) [dyn/cm²]
    X : scalar — hydrogen mass fraction
    Z : scalar — metals mass fraction
    ln_rho_guess : scalar or None — warm-start initial guess for the
        density Newton solve (ln(rho) in cgs). If None, uses ideal-gas.
        Passing the previous step's converged density reduces iterations
        from ~7 to ~3 (MESA pattern: eospt_eval.f90:186, micro.f90:333).

    Returns the same 7-tuple: (rho, mu, nad, S, cp, chi_rho, chi_T)

    Reference: Timmes & Swesty (2000), ApJS 126, 501
    """
    T = 10.0**logT
    ln_T = jnp.log(T)

    mu_e = 2.0 / (1.0 + X)
    Y = 1.0 - X - Z
    A_bar = 1.0 / (X + Y / 4.0 + Z / 16.0)
    mu = 1.0 / (1.0 / mu_e + 1.0 / A_bar)

    # Compute default guess if none provided (ideal gas, matching MESA
    # eospt_eval.f90:186: rho_guess = Pgas*abar*mp/(kerg*T*(1+zbar)))
    P_target = 10.0**logP_target
    default_guess = jnp.log(jnp.maximum(P_target * mu * m_H / (k_B * T), 1e-10))
    guess = default_guess if ln_rho_guess is None else ln_rho_guess

    # Solve for density via IFT-backed NR with warm-start
    ln_rho = _solve_density_fd_ift(logT, logP_target, X, Z, guess)
    rho = jnp.exp(ln_rho)

    # Thermodynamic derivatives via autodiff of the free energy
    P_total = _total_pressure_fd(ln_rho, ln_T, X, Z)
    dP_dlnrho = jax.grad(_total_pressure_fd, argnums=0)(ln_rho, ln_T, X, Z)
    chi_rho = dP_dlnrho / jnp.maximum(P_total, 1e-30)
    dP_dlnT = jax.grad(_total_pressure_fd, argnums=1)(ln_rho, ln_T, X, Z)
    chi_T = dP_dlnT / jnp.maximum(P_total, 1e-30)

    S_total = _total_entropy_fd(ln_rho, ln_T, X, Z)
    dS_dlnT = jax.grad(_total_entropy_fd, argnums=1)(ln_rho, ln_T, X, Z)
    # Loose NON-PHYSICAL stability guards (cv≥1e5, Γ₁≥0.1, χ_ρ≥0.01, cp≥1e5), NOT
    # tuned physics — they only keep the differentiated free-energy derivatives
    # finite where the fit is noisy; true values are far from these floors.
    cv = jnp.maximum(dS_dlnT, 1e5)

    # Cox & Giuli (1968) thermodynamic relations
    Gamma3_m1 = P_total * chi_T / (rho * T * jnp.maximum(cv, 1e5))
    Gamma1 = jnp.maximum(chi_rho + Gamma3_m1 * chi_T, 0.1)
    nad = Gamma3_m1 / Gamma1
    cp = cv * Gamma1 / jnp.maximum(chi_rho, 0.01)
    cp = jnp.maximum(cp, 1e5)

    return rho, mu, nad, S_total, cp, chi_rho, chi_T
