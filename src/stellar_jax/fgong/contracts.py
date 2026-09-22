"""fgong/ module contracts: types, shapes, and differentiability boundaries.

This module documents the interface contracts for the FGONG bridge between the
Henyey solver (evolution/) and the oscillation eigensolver (oscillations/).

References
----------
Christensen-Dalsgaard (2008), Ap&SS 316, 113, §A.1 — FGONG format specification.
Paxton et al. (2013), ApJS 208, 4, §5 — MESA–ADIPLS coupling.
"""

# --- Shape contracts ---
# FGONG global parameters: (15,) — [M, R, L, Z, X_surf, alpha, ...padding..., G]
GLOB_SIZE = 15

# FGONG per-point variable columns
VAR_COLS_DIFF = 15    # differentiable path (structure_to_fgong_jax)
VAR_COLS_FULL = 25    # publication path (write_fgong)

# High-resolution mesh for publication-grade I/O
N_HIRES = 2500


# --- Named FGONG column indices ---
# Per-point variable columns (var[:, N]):
# Reference: Christensen-Dalsgaard (2008), Ap&SS 316, 113, Table A.1.
VAR_R = 0             # radius [cm]
VAR_LN_Q = 1         # ln(m/M) — natural log of mass fraction
VAR_T = 2            # temperature [K]
VAR_P = 3            # total pressure [dyn/cm²]
VAR_RHO = 4          # density [g/cm³]
VAR_X = 5            # hydrogen mass fraction
VAR_L = 6            # luminosity [erg/s]
VAR_KAPPA = 7        # opacity [cm²/g]
VAR_EPS_NUC = 8      # nuclear energy generation rate [erg/g/s]
VAR_GAMMA1 = 9       # first adiabatic exponent Γ₁
VAR_NABLA_AD = 10    # adiabatic temperature gradient ∇_ad
VAR_DELTA = 11       # -(∂lnρ/∂lnT)_P — thermal expansion coefficient
VAR_CP = 12          # specific heat at constant pressure [erg/g/K]
VAR_NABLA = 13       # actual temperature gradient ∇
VAR_A_STAR = 14      # Brunt-Väisälä discriminant A* = N²r/g
# col 15: unused (reserved)
VAR_Z = 16           # metal mass fraction
VAR_DIST_SURF = 17   # R_star - r: distance from surface [cm]

# Global parameter indices (glob[N]):
GLOB_M = 0           # stellar mass [g]
GLOB_R = 1           # stellar radius [cm]
GLOB_L = 2           # stellar luminosity [erg/s]
GLOB_Z = 3           # surface metal mass fraction
GLOB_X_SURF = 4      # surface hydrogen mass fraction
GLOB_ALPHA = 5       # mixing-length parameter α_MLT
GLOB_AGE = 12        # stellar age [s] (age_yr * SECONDS_PER_YEAR)
GLOB_G = 14          # gravitational constant G [dyn cm² g⁻²]


# --- Differentiability contract ---
# structure_to_fgong_jax:
#   DIFFERENTIABLE w.r.t.: y_henyey (the live Henyey state), M_solar, log_L,
#                          log_Te, X_profile, Z
#   NOT differentiated through: q_mesh (from initial_lagrangian_mesh —
#                                deterministic, fixed)
#   No @custom_vjp — relies on standard JAX AD through EOS lookups + array ops
#   Gradient flow: M → evolve_star → y_henyey → structure_to_fgong_jax →
#                  (glob, var) → oscillations → eigenfreq
#
# hires_profile:
#   NOT on the gradient path (output converted to NumPy)
#   Uses lax.scan internally for speed but is a leaf computation for I/O only
#
# read_fgong, write_fgong, write_gyre:
#   Side-effecting (file I/O) — NEVER inside JIT/grad
