"""FGONG/GYRE file I/O — side-effecting, off the gradient path.

Contains:
- read_fgong: parse FGONG file → (glob, var) NumPy arrays
- fgong_components: named dict from FGONG arrays
- write_fgong: write publication-grade FGONG v3.00 file
- write_gyre: write GYRE MESA v1.01 format file

All functions are pure NumPy + file operations. NEVER inside JIT/grad.

References
----------
Christensen-Dalsgaard (2008), Ap&SS 316, 113, §A.1 — FGONG format.
Townsend & Teitler (2013), MNRAS 435, 3406 — GYRE format.
"""

import re
import numpy as np

from stellar_jax.config.constants import G, SECONDS_PER_YEAR
from stellar_jax.fgong.contracts import (
    N_HIRES,
    VAR_R, VAR_LN_Q, VAR_T, VAR_P, VAR_RHO, VAR_X, VAR_L, VAR_KAPPA,
    VAR_EPS_NUC, VAR_GAMMA1, VAR_NABLA_AD, VAR_DELTA, VAR_CP, VAR_NABLA,
    VAR_A_STAR, VAR_Z, VAR_DIST_SURF,
    GLOB_M, GLOB_R, GLOB_L, GLOB_Z, GLOB_X_SURF, GLOB_ALPHA, GLOB_AGE, GLOB_G,
)
from stellar_jax.fgong.hires_profile import hires_profile


def read_fgong(filename):
    """Parse an FGONG file and return (glob, var) arrays.

    Returns:
        glob: array of shape (iconst,) with global parameters
        var: array of shape (nn, ivar) with per-point variables
             Ordered center-to-surface (r increasing).
    """
    with open(filename, 'r') as f:
        for _ in range(4):
            f.readline()
        dims = f.readline().split()
        nn, iconst, ivar, _ = int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3])
        numbers = []
        for line in f:
            line = line.replace('D', 'E').replace('d', 'e')
            tokens = re.findall(r'[+-]?\d+\.\d+[Ee][+-]?\d+|[+-]?\d+\.\d+|[+-]?\d+', line)
            numbers.extend(tokens)
    data = np.array([float(x) for x in numbers])
    glob = data[:iconst]
    var = data[iconst:iconst + nn * ivar].reshape(nn, ivar)
    if var[0, VAR_R] > var[-1, VAR_R]:
        var = var[::-1]
    return glob, var


def fgong_components(glob, var):
    """Named per-zone components from a loaded MESA FGONG (ordered center->surface).

    A loaded FGONG is a REAL external MODE-A reference (identical physics: Z=0.014, Y=0.2695,
    alpha_MLT=2.0 Cox, Krishna-Swamy T(tau), diffusion/overshoot/rotation OFF) -- NOT a
    synthetic/hand-built profile. Use it to validate one microphysics/operator component pointwise
    against MESA at its own (rho, T, X); run the component with MESA_CONFIG (MODE A). Never
    cross-compare with Model S (MODE B).

    Standard MESAstar FGONG layout (0-indexed per-zone column):
        r=0, ln(m/M)=1, T=2, P=3, rho=4, X=5, L=6, kappa=7, eps_nuc=8, Gamma1=9, ... Brunt A*=14.
    kappa/eps_nuc are populated by MESAstar pulse output; verify non-zero in-core before relying on
    a column (eps_nuc is correctly ~0 in the envelope). Returns a dict of 1-D numpy arrays.
    """
    v = np.asarray(var)
    out = dict(r=v[:, VAR_R], T=v[:, VAR_T], P=v[:, VAR_P], rho=v[:, VAR_RHO],
               X=v[:, VAR_X], L=v[:, VAR_L], kappa=v[:, VAR_KAPPA],
               eps_nuc=v[:, VAR_EPS_NUC], gamma1=v[:, VAR_GAMMA1])
    if v.shape[1] > 1:
        out["m_frac"] = np.exp(np.clip(v[:, VAR_LN_Q], -700, 0))   # m/M = exp(ln(m/M))
    if v.shape[1] > 14:
        out["brunt_A"] = v[:, VAR_A_STAR]
    return out


def _compute_brunt_vaisala(r, P, rho, Gamma1, R_star, mu=None,
                           chi_rho=None, chi_T=None):
    """Compute the Brunt-Väisälä discriminant A* on a non-uniform mesh.

    A* = (1/Γ₁) dlnP/dlnr − dlnρ/dlnr   (dimensionless)

    On self-consistent structure (where ρ is computed from EOS with the correct
    local composition at every point), this Schwarzschild form IS the full Ledoux
    discriminant: the composition effect on buoyancy is already encoded in ρ(r).
    No explicit (φ/δ)·dlnμ/dlnr term is needed (adding it would double-count).

    MESA equivalence: MESA writes A* = N²r/g to FGONG col-14 using its full MHM
    brunt_B (brunt.f90:get_brunt_B — two EOS calls per face at different
    compositions, same (T,ρ)). That MHM form decomposes N² into thermal +
    composition terms for diagnostic purposes, but the total A* it produces equals
    (1/Γ₁)dlnP/dlnr − dlnρ/dlnr evaluated on the self-consistent structure to
    <0.2% RMS across all masses and stages (verified against committed MODE-A
    FGONG library; issue #683).

    Uses proper unequal-spacing 3-point centered finite differences on the
    quadratic-stretched mesh (Fornberg 1988, Math. Comp. 51, 699):
        df/dr|_i = [h_-² f_{i+1} + (h_+² - h_-²) f_i - h_+² f_{i-1}]
                   / [h_+ h_- (h_+ + h_-)]
    where h_+ = r[i+1]-r[i], h_- = r[i]-r[i-1]. This is O(h²) on non-uniform
    meshes (the symmetric formula (r[i+1]-r[i-1]) is only O(h) when spacing varies).

    Guards: set A* = 0 near center (r < 1e-4 * R_star) where log-derivatives
    are numerically unstable.

    Parameters
    ----------
    r : (N,) ndarray — radius array (same order as profile)
    P : (N,) ndarray — total pressure
    rho : (N,) ndarray — density
    Gamma1 : (N,) ndarray — first adiabatic exponent
    R_star : float — stellar radius
    mu : (N,) ndarray or None — UNUSED (kept for API compatibility)
    chi_rho : (N,) ndarray or None — UNUSED (kept for API compatibility)
    chi_T : (N,) ndarray or None — UNUSED (kept for API compatibility)

    Returns
    -------
    AA : (N,) ndarray — Brunt-Väisälä discriminant A*
    """
    nn = len(r)
    r_min_brunt = 1e-4 * R_star
    AA = np.zeros(nn)

    lnP = np.log(np.maximum(P, 1.0))
    lnrho = np.log(np.maximum(rho, 1e-30))

    for i in range(1, nn - 1):
        if r[i] <= r_min_brunt:
            continue

        # Unequal-spacing centered difference (Fornberg 1988)
        h_plus = r[i+1] - r[i]
        h_minus = r[i] - r[i-1]
        denom = h_plus * h_minus * (h_plus + h_minus)
        if abs(denom) < 1e-30:
            continue

        # d(lnf)/dr using Fornberg's O(h²) formula
        w_plus = h_minus**2
        w_zero = h_plus**2 - h_minus**2
        w_minus = -(h_plus**2)
        dlnP_dr = (w_plus * lnP[i+1] + w_zero * lnP[i]
                   + w_minus * lnP[i-1]) / denom
        dlnrho_dr = (w_plus * lnrho[i+1] + w_zero * lnrho[i]
                     + w_minus * lnrho[i-1]) / denom

        # Convert d/dr to d/dlnr by multiplying by r
        dlnP_dlnr = dlnP_dr * r[i]
        dlnrho_dlnr = dlnrho_dr * r[i]

        AA[i] = dlnP_dlnr / max(Gamma1[i], 0.1) - dlnrho_dlnr

    return AA


def _assemble_fgong_arrays(prof, M_star, R_star, L_star, Z, X_profile,
                           alpha_mlt, t_age):
    """Assemble the full 25-column FGONG (glob, var) arrays from a profile dict.

    Parameters
    ----------
    prof : dict — output of hires_profile (keys: r, P, T, rho, Mr, Lr, ...)
    M_star, R_star, L_star : float — stellar parameters in CGS
    Z, alpha_mlt, t_age : float — metallicity, MLT param, age in years
    X_profile : array — hydrogen mass fraction profile

    Returns
    -------
    glob : (15,) ndarray — FGONG global parameters
    var : (N, 25) ndarray — FGONG per-point variables
    """
    nn = len(prof['r'])
    r = prof['r']
    P = prof['P']
    T = prof['T']
    rho = prof['rho']
    Mr = prof['Mr']
    Lr = prof['Lr']
    Gamma1 = prof['Gamma1']
    nad = prof['nabla_ad']
    delta = prof['delta']
    cp = prof['cp']
    nabla = prof['nabla']
    kap = prof['kappa_val']
    eps = prof['eps_nuc']
    X_l = prof['X_local']

    AA = _compute_brunt_vaisala(r, P, rho, Gamma1, R_star)

    ivar = 25
    iconst = 15
    var = np.zeros((nn, ivar))
    q = np.clip(Mr / M_star, 1e-30, 1.0)
    var[:, VAR_R] = r
    var[:, VAR_LN_Q] = np.log(q)
    var[:, VAR_T] = T
    var[:, VAR_P] = P
    var[:, VAR_RHO] = rho
    var[:, VAR_X] = X_l
    var[:, VAR_L] = Lr
    var[:, VAR_KAPPA] = kap
    var[:, VAR_EPS_NUC] = eps
    var[:, VAR_GAMMA1] = Gamma1
    var[:, VAR_NABLA_AD] = nad
    var[:, VAR_DELTA] = delta
    var[:, VAR_CP] = cp
    var[:, VAR_NABLA] = nabla
    var[:, VAR_A_STAR] = AA
    var[:, VAR_Z] = float(Z)
    var[:, VAR_DIST_SURF] = R_star - r  # distance from surface

    # Guard against NaN/Inf from numerical issues in the high-resolution
    # profile integration. Replace with 0.0 — the oscillation solver uses
    # only columns 0-9 which should always be finite for a converged model.
    nan_mask = ~np.isfinite(var)
    if np.any(nan_mask):
        var[nan_mask] = 0.0

    X_prof_np = np.asarray(X_profile)
    X_surf = float(X_prof_np[-1])

    glob = np.zeros(iconst)
    glob[GLOB_M] = M_star
    glob[GLOB_R] = R_star
    glob[GLOB_L] = L_star
    glob[GLOB_Z] = float(Z)
    glob[GLOB_X_SURF] = X_surf
    glob[GLOB_ALPHA] = float(alpha_mlt)
    glob[GLOB_AGE] = t_age * SECONDS_PER_YEAR
    glob[GLOB_G] = float(G)

    return glob, var


def write_fgong(filename, M_solar, log_L, log_Te, X_profile, Z, t_age,
                alpha_mlt, description=None):
    """Write a converged stellar model in FGONG format (publication-grade).

    All FGONG variables populated with EOS-consistent quantities:
    - Gamma1 from chi_rho + (Gamma3-1)*chi_T [KW 2012 eq 13.22; HELM]
    - cp, delta, nabla_ad from OPAL EOS (not ideal-gas approximation)
    - Brunt-Väisälä discriminant A* on self-consistent structure (includes
      composition effect through ρ; equivalent to full Ledoux — see
      _compute_brunt_vaisala docstring for the MESA equivalence proof)

    Resolution: N_HIRES=2500 mesh points (quadratic-stretched, surface-dense).
    Format: FGONG v3.00 (Christensen-Dalsgaard 2008, Ap&SS 316, 113).

    Internally delegates to:
    - hires_profile() for high-res structure integration
    - _assemble_fgong_arrays() for FGONG array construction
    - _compute_brunt_vaisala() for the A* discriminant
    """
    prof, M_star, R_star, L_star = hires_profile(
        M_solar, log_L, log_Te, X_profile, Z, t_age, alpha_mlt)

    glob, var = _assemble_fgong_arrays(
        prof, M_star, R_star, L_star, Z, X_profile, alpha_mlt, t_age)

    nn = len(prof['r'])
    ivar = 25
    iconst = 15
    ivers = 300

    X_prof_np = np.asarray(X_profile)
    X_c_val = float(X_prof_np[0])

    if description is None:
        description = [
            f'stellar-jax {M_solar:.2f} Msun Z={Z:.4f} alpha={alpha_mlt:.3f}',
            f'age={t_age:.3e} yr  X_c={X_c_val:.4f}',
            'FGONG format v3.00 (Christensen-Dalsgaard 2008, ApSS 316, 113)',
            'EOS-consistent Gamma1/cp/delta; Ledoux Brunt-Vaisala',
        ]

    with open(filename, 'w') as f:
        for line in description[:4]:
            f.write(line + '\n')
        f.write('%10d%10d%10d%10d\n' % (nn, iconst, ivar, ivers))
        for i, val in enumerate(glob):
            f.write('%16.9E' % val)
            if (i + 1) % 5 == 0:
                f.write('\n')
        if iconst % 5 != 0:
            f.write('\n')
        for row in var:
            for i, val in enumerate(row):
                f.write('%16.9E' % val)
                if (i + 1) % 5 == 0:
                    f.write('\n')
            if ivar % 5 != 0:
                f.write('\n')

    return filename


def _compute_n2_brunt(r, P, rho, Gamma1, Mr, R_star):
    """Compute Brunt-Väisälä frequency squared N² from structure arrays.

    N² = g/r * A* where A* = (1/Γ₁)dlnP/dlnr − dlnρ/dlnr on self-consistent
    structure (composition already in ρ; see _compute_brunt_vaisala docstring).

    Parameters
    ----------
    r, P, rho, Gamma1, Mr : (N,) ndarray — structure arrays
    R_star : float — stellar radius

    Returns
    -------
    N2 : (N,) ndarray — Brunt-Väisälä frequency squared (rad²/s²)
    """
    AA = _compute_brunt_vaisala(r, P, rho, Gamma1, R_star)
    g = float(G) * Mr / np.maximum(r, 1.0)**2
    N2 = np.where(r > 1e-4 * R_star, g / np.maximum(r, 1.0) * AA, 0.0)
    return N2


def write_gyre(filename, M_solar, log_L, log_Te, X_profile, Z, t_age,
               alpha_mlt):
    """Write a converged stellar model in GYRE MESA format v1.01.

    Format: Townsend & Teitler (2013), MNRAS 435, 3406.
    Header: N  M_star  R_star  L_star  101
    Data (N lines, center→surface): k r M_r L_r P T rho nabla N² Gamma1
        nabla_ad delta kappa kappa_T kappa_rho eps_nuc eps_T eps_rho Omega_rot

    Uses same EOS-consistent quantities as write_fgong.

    Internally delegates to:
    - hires_profile() for high-res structure integration
    - _compute_n2_brunt() for N² (via _compute_brunt_vaisala for A*)
    """
    prof, M_star, R_star, L_star = hires_profile(
        M_solar, log_L, log_Te, X_profile, Z, t_age, alpha_mlt)

    nn = N_HIRES
    r = prof['r']
    P = prof['P']
    rho = prof['rho']
    Mr = prof['Mr']
    Lr = prof['Lr']
    Gamma1 = prof['Gamma1']
    nad = prof['nabla_ad']
    delta = prof['delta']
    nabla = prof['nabla']
    kap = prof['kappa_val']
    eps = prof['eps_nuc']

    N2 = _compute_n2_brunt(r, P, rho, Gamma1, Mr, R_star)

    # GYRE format is center→surface (reverse our surface→center arrays)
    idx = np.arange(nn - 1, -1, -1)

    with open(filename, 'w') as f:
        # Header
        f.write(f'{nn:6d} {M_star:26.16E} {R_star:26.16E} '
                f'{L_star:26.16E} {101:4d}\n')
        # Data lines
        for j in range(nn):
            k = j + 1
            ii = idx[j]
            f.write(f'{k:6d}' +
                    ''.join(f' {v:26.16E}' for v in [
                        r[ii], Mr[ii], Lr[ii], P[ii], prof['T'][ii],
                        rho[ii], nabla[ii], N2[ii], Gamma1[ii], nad[ii],
                        delta[ii], kap[ii], 0.0, 0.0,  # kappa_T, kappa_rho
                        eps[ii], 0.0, 0.0,  # eps_T, eps_rho
                        0.0  # Omega_rot
                    ]) + '\n')

    return filename
