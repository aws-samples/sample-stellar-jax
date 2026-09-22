"""Burgers equation solvers for element diffusion coefficients.

Extracted from composition/diffuse.py per the L2 spec (issue #530).
These are pure numerical kernels — self-contained linear-algebra solves.

Implements the multi-species Burgers equations following Thoul, Bahcall & Loeb
(1994, ApJ 421, 828) and the 4-species extension of Turcotte et al. (1998,
ApJ 504, 539).

MESA reference:
  - star/private/diffusion_support.f90: do1_solve_thoul_hu()
  - Solves the resistance-coefficient system for AP, AT, AX coefficients.

References:
  - Thoul, Bahcall & Loeb (1994), ApJ 421, 828: H-He-e diffusion
  - Paquette et al. (1986), ApJS 61, 177: Coulomb logarithm fit
  - Turcotte et al. (1998), ApJ 504, 539: multi-element extension
"""
import jax.numpy as jnp


def _solve_burgers_system(A, Z_ion, C, T, rho, n_gamma_rhs):
    """Solve the multi-species Burgers resistance-matrix system.

    Parameterized core for both 3-species (H-He-e) and 4-species (H-He-Z-e).
    Constructs the (2M+2)×(2M+2) DELTA matrix and solves for AP, AT, and AX
    coefficients.

    Parameters
    ----------
    A : (M,) array — atomic mass numbers [amu]
    Z_ion : (M,) array — ionic charges (electrons = -1)
    C : (M,) array — concentration fractions (TBL eq. 5)
    T : scalar — temperature [K]
    rho : scalar — density [g/cm³]
    n_gamma_rhs : int — number of GAMMA concentration-gradient RHS vectors
        (= number of non-electron species minus 1, i.e. M-2 for 3-species, M-2+1 for 4-species;
         concretely 2 for 3-species [H, He], 3 for 4-species [H, He, Z])

    Returns
    -------
    SOL : (2M+2, 2+n_gamma_rhs) array — solution columns: [ALPHA, NU, GAMMA_0, ...]
    norm : scalar — K0 * AC * CC normalization factor

    References:
      - Thoul, Bahcall & Loeb (1994), ApJ 421, 828, eq. 5-12, 21
      - MESA star/private/diffusion_support.f90: do1_solve_thoul_hu()
    """
    M = A.shape[0]
    N = 2 * M + 2

    CC = jnp.sum(C)
    AC = jnp.sum(A * C)

    # Coulomb logarithm matrix CL(i,j) — TBL eq. 9 + README errata
    NE = rho / (1.6726e-24 * AC)
    # NI = sum of ion concentrations (exclude electron, last species)
    NI = jnp.sum(C[:M-1]) * NE
    AO = (0.23873 / (NI + 1e-10)) ** (1.0 / 3.0)
    CZ = jnp.sum(C * Z_ion**2)
    LAMBDAD = 6.9010 * jnp.sqrt(T / (NE * CZ + 1e-10))
    LAMBDA = jnp.maximum(LAMBDAD, AO)
    ZiZj = jnp.abs(Z_ion[:, None] * Z_ion[None, :])
    XIJ = 2.3939e3 * T * LAMBDA / (ZiZj + 1e-30)
    CL = 0.81245 * jnp.log(1.0 + 0.18769 * XIJ**1.2)

    # Resistance coefficients K(i,j) (TBL eq. 10-12, routine.f)
    Aij = A[:, None] * A[None, :]
    Asum = A[:, None] + A[None, :]
    K = CL * jnp.sqrt(Aij / Asum) * (C[:, None] * C[None, :]) * (Z_ion[:, None]**2 * Z_ion[None, :]**2)
    XX = A[None, :] / Asum  # XX[i,j] = A_j / (A_i + A_j)
    YC = A[:, None] / Asum  # YC[i,j] = A_i / (A_i + A_j)

    # --- Momentum equations (rows 0..M-1) ---
    K_row_sum = jnp.sum(K, axis=1)
    mom_vel = K - jnp.diag(K_row_sum)

    mom_heat_sum = jnp.sum(0.6 * XX * K, axis=1)
    mom_heat_self = 0.6 * jnp.diag(XX) * jnp.diag(K)
    mom_heat_diag = mom_heat_sum - mom_heat_self
    mom_heat = -0.6 * YC * K
    mom_heat = mom_heat - jnp.diag(jnp.diag(mom_heat)) + jnp.diag(mom_heat_diag)

    mom_E = C * Z_ion
    mom_g = -C * A

    # --- Energy equations (rows M..2M-1) ---
    en_vel_offdiag = -1.5 * XX * K
    en_vel_diag_sum = jnp.sum(1.5 * XX * K, axis=1) - 1.5 * jnp.diag(XX) * jnp.diag(K)
    en_vel = en_vel_offdiag - jnp.diag(jnp.diag(en_vel_offdiag)) + jnp.diag(en_vel_diag_sum)

    YY_full = 3.0 * YC + 1.3 * XX * (A[None, :] / A[:, None])

    en_heat_diag_terms = YC * K * (1.6 * XX + YY_full)
    en_heat_diag_sum = jnp.sum(en_heat_diag_terms, axis=1) - jnp.diag(en_heat_diag_terms)
    en_heat_diag = -en_heat_diag_sum - 0.8 * jnp.diag(K)

    en_heat_offdiag = 2.7 * K * XX * YC
    en_heat = en_heat_offdiag - jnp.diag(jnp.diag(en_heat_offdiag)) + jnp.diag(en_heat_diag)

    # --- Constraints ---
    row_mom = jnp.concatenate([mom_vel, mom_heat, mom_E[:, None], mom_g[:, None]], axis=1)
    row_en = jnp.concatenate([en_vel, en_heat, jnp.zeros((M, 1)), jnp.zeros((M, 1))], axis=1)
    row_cn = jnp.concatenate([C * Z_ion, jnp.zeros(M), jnp.zeros(2)])[None, :]
    row_mf = jnp.concatenate([C * A, jnp.zeros(M), jnp.zeros(2)])[None, :]

    DELTA = jnp.concatenate([row_mom, row_en, row_cn, row_mf], axis=0)

    # RHS vectors (TBL eq. 6-7)
    ALPHA = jnp.zeros(N).at[:M].set(C[:M] / CC)
    NU = jnp.zeros(N).at[M:2*M].set(2.5 * C[:M] / CC)

    # Concentration-gradient RHS vectors (TBL 1994, eq. 21).
    C_over_CC = C / CC

    def _make_gamma(species_idx):
        """Build GAMMA vector for concentration gradient of species_idx."""
        gamma = jnp.zeros(N)
        for j in range(M):
            if j == species_idx:
                gamma = gamma.at[j].set(C_over_CC[j] * (1.0 - C_over_CC[species_idx]))
            else:
                gamma = gamma.at[j].set(C_over_CC[j] * (0.0 - C_over_CC[species_idx]))
        return gamma

    # Build GAMMA vectors for each non-electron diffusing species
    # For 3-species: [H, He]; for 4-species: [H, He, Z]
    gamma_list = [_make_gamma(i) for i in range(n_gamma_rhs)]

    # Batch all RHS into a single matrix solve
    RHS = jnp.stack([ALPHA, NU] + gamma_list, axis=1)
    SOL = jnp.linalg.solve(DELTA, RHS)

    # Normalization: KO × AC × CC (TBL routine.f line: KO=2.)
    K0 = 2.0
    norm = K0 * AC * CC

    return SOL, norm


def solve_3species(X, Y, T, rho):
    """Solve the Burgers equations for a H-He-e plasma (Thoul+ 1994, ApJ 421, 828).

    Returns AP_He, AT_He — the pressure and thermal diffusion coefficients
    for helium. These are composition-dependent via the resistance matrix.

    Uses the 3-species (H, He, e⁻) approximation. For solar-metallicity plasmas
    (Z < 0.03), metals contribute < 5% to the resistance matrix and can be
    treated as a trace species absorbed into the electron background (Thoul,
    Bahcall & Loeb 1994, §4; their Table 3 shows <5% error vs full multi-species).

    Matrix construction follows TBL's IAS Fortran code (routine.f) EXACTLY:
      https://www.sns.ias.edu/~jnb/SNdata/Export/Diffusion/routine.f
    Cross-validated against MESA's do1_solve_thoul_hu() in
      star/private/diffusion_support.f90
    which uses Zdiff/Zdiff1/Zdiff2 arrays (for pure Coulomb: 0.6, 1.3, 2.0).

    Parameters
    ----------
    X, Y : scalar (float)
        Hydrogen and helium mass fractions at the zone.
    T : scalar (float)
        Temperature [K] at the zone.
    rho : scalar (float)
        Density [g/cm³] at the zone.

    Returns
    -------
    AP_He : scalar
        Pressure diffusion coefficient for He.
    AT_He : scalar
        Thermal diffusion coefficient for He.
    AX_He_H : scalar
        Concentration-gradient coefficient: He response to H gradient.
    AX_He_He : scalar
        Concentration-gradient coefficient: He response to He gradient.
    """
    # Species: 0=H, 1=He, 2=electrons
    m_e_over_m_u = 5.4858e-4
    A = jnp.array([1.0, 4.0, m_e_over_m_u])
    Z_ion = jnp.array([1.0, 2.0, -1.0])

    # Concentrations (TBL eq. 5)
    ZXA = jnp.maximum(Z_ion[0] * X / A[0] + Z_ion[1] * Y / A[1], 1e-30)
    C = jnp.array([X / (A[0] * ZXA), Y / (A[1] * ZXA), 1.0])

    SOL, norm = _solve_burgers_system(A, Z_ion, C, T, rho, n_gamma_rhs=2)

    AP_He = SOL[1, 0] * norm
    AT_He = SOL[1, 1] * norm
    AX_He_H = SOL[1, 2] * norm
    AX_He_He = SOL[1, 3] * norm

    return AP_He, AT_He, AX_He_H, AX_He_He


def solve_4species(X, Y, Z_mass, T, rho):
    """Solve the Burgers equations for a H-He-Z-e plasma (4 species).

    Extends TBL (1994) to include a mean metal species with mass-weighted
    effective settling properties (A_eff=41, Z_eff=12) following Turcotte et al.
    (1998, ApJ 504, 539) and MESA's do1_solve_thoul_hu with nsmall=4
    (Paxton+ 2015, §5.3).

    Parameters
    ----------
    X, Y, Z_mass : scalar (float)
        Hydrogen, helium, metal mass fractions at the zone.
    T : scalar (float)
        Temperature [K] at the zone.
    rho : scalar (float)
        Density [g/cm³] at the zone.

    Returns
    -------
    tuple of 10 scalars:
        (AP_He, AT_He, AP_Z, AT_Z,
         AX_He_H, AX_He_He, AX_He_Z, AX_Z_H, AX_Z_He, AX_Z_Z)
    """
    m_e_over_m_u = 5.4858e-4
    A = jnp.array([1.0, 4.0, 41.0, m_e_over_m_u])
    Z_ion = jnp.array([1.0, 2.0, 12.0, -1.0])

    Z_mass_safe = jnp.maximum(Z_mass, 1e-10)
    TEMP = Z_ion[0] * X / A[0] + Z_ion[1] * Y / A[1] + Z_ion[2] * Z_mass_safe / A[2]
    TEMP = jnp.maximum(TEMP, 1e-30)
    C = jnp.array([X / A[0] / TEMP, Y / A[1] / TEMP,
                   Z_mass_safe / A[2] / TEMP, 1.0])

    SOL, norm = _solve_burgers_system(A, Z_ion, C, T, rho, n_gamma_rhs=3)

    AP_He = SOL[1, 0] * norm
    AT_He = SOL[1, 1] * norm
    AP_Z = SOL[2, 0] * norm
    AT_Z = SOL[2, 1] * norm

    AX_He_H = SOL[1, 2] * norm
    AX_He_He = SOL[1, 3] * norm
    AX_He_Z = SOL[1, 4] * norm

    AX_Z_H = SOL[2, 2] * norm
    AX_Z_He = SOL[2, 3] * norm
    AX_Z_Z = SOL[2, 4] * norm

    return (AP_He, AT_He, AP_Z, AT_Z,
            AX_He_H, AX_He_He, AX_He_Z, AX_Z_H, AX_Z_He, AX_Z_Z)
