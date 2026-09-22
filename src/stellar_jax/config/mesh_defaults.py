"""Mesh defaults and solver-tuning constants for stellar-jax.

Contains the composition mesh (N_COMP, COMP_MFRACS, COMP_ZONE_MASSES),
the structure mesh size (N_HENYEY), shooting grid (N_SHOOT), Newton
iteration limits, overshooting parameters, and the finite-difference step.
"""

import numpy as np


# Default mixing-length parameter
ALPHA_MLT = 1.9

# Shooting grid and finite-difference step
N_SHOOT = 300
N_HENYEY = 600  # Henyey mesh zones (increased from 300 for CNO core resolution)
EPS_FD = 1e-5
N_NEWTON_COLD = 15
N_NEWTON_WARM = 4

# Mass grid for composition: N_COMP non-uniformly spaced points in m/M from 0 to 1
# Coordinate transform: q uniform on [0,1], m/M = q^3.
N_COMP = 200
_xi = np.linspace(0.0, 1.0, N_COMP)
COMP_MFRACS = _xi ** 3
COMP_MFRACS[0] = 0.0
COMP_MFRACS[-1] = 1.0

# Conservative finite-volume zone masses (sum to exactly 1.0).
# Cell boundaries at the midpoints between grid nodes, pinned at 0 and 1:
#   bnd[0] = 0.0, bnd[k] = 0.5*(COMP_MFRACS[k-1] + COMP_MFRACS[k]) for k=1..N-1, bnd[N] = 1.0
# Zone mass[k] = bnd[k+1] - bnd[k]  →  sum = bnd[N] - bnd[0] = 1.0 - 0.0 = 1.0 (exact).
# This matches MESA's convention where dq(k) are cell widths summing to 1
# (star_utils.f90:672-713, normalize_dqs).
_midpoints = 0.5 * (COMP_MFRACS[:-1] + COMP_MFRACS[1:])  # N-1 interior boundaries
_bnd = np.concatenate([[0.0], _midpoints, [1.0]])  # N+1 boundaries
COMP_ZONE_MASSES = np.diff(_bnd)  # N zone masses, sum = 1.0 exactly
assert abs(COMP_ZONE_MASSES.sum() - 1.0) < 1e-15, \
    f"Zone masses must sum to 1, got {COMP_ZONE_MASSES.sum()}"


def compute_zone_masses(comp_mfracs):
    """Compute conservative finite-volume zone masses for an arbitrary grid.

    Cell boundaries are placed at the midpoints between adjacent grid nodes,
    pinned at 0 and 1 at the domain edges.  This guarantees sum(zone_masses)=1.0
    exactly (telescoping sum: bnd[N]-bnd[0]=1-0=1).

    Matches MESA's convention: dq(k) are cell widths summing to 1
    (star_utils.f90:672-713, normalize_dqs; set_m_and_dm L635-648).

    Parameters
    ----------
    comp_mfracs : array (N,) — grid node positions in [0, 1].

    Returns
    -------
    zone_masses : array (N,) — cell masses summing to 1.0.
    """
    import jax.numpy as jnp
    midpoints = 0.5 * (comp_mfracs[:-1] + comp_mfracs[1:])
    bnd = jnp.concatenate([jnp.zeros(1), midpoints, jnp.ones(1)])
    return jnp.diff(bnd)


# Step overshooting parameter (Herwig 2000 f_ov convention).
# The mix_composition formula is delta_m_ov = f_ov · Hp · dm/dr, matching
# MESA overshoot_step.f90:131 (dr < f * Hp_cb). The solar-calibrated value
# is f_ov ≈ 0.016 (Herwig 2000, A&A 360, 952).
#
# HISTORY: the codebase previously carried ALPHA_OV = 0.2 alongside
# F_OV = 0.016, with the evolution/calibration path defaulting to ALPHA_OV
# and the inference path to F_OV. ALPHA_OV = 0.2 is a valid α_ov in the
# α_ov·Hp-DISTANCE convention (MIST/Dotter 2016; Claret & Torres 2016) but
# is NOT numerically equal to f_ov — passing 0.2 as f_ov was a 12.5×
# parametrization error, fully active at M ≥ 1.5 M☉. All paths now use F_OV.
F_OV = 0.016
