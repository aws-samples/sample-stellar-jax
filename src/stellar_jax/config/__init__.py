"""config/ — Frozen dataclasses, physical constants, and configuration for stellar-jax.

Public API:
    StellarParams  — physics parameters (all traced/differentiable)
    SolverConfig   — algorithmic settings (all static/recompile triggers)
    HasPhysicsKnobs — protocol for knob consumers

    Physical constants: G, c_light, sigma_sb, a_rad, k_B, m_H, Msun, Lsun, Rsun, ...
    Calibration: ALPHA_SOLAR, Y0_SOLAR
    Mesh defaults: N_COMP, COMP_MFRACS, COMP_ZONE_MASSES, compute_zone_masses, ...
    MESA config: MESA_CONFIG, Z_MESA, ALPHA_MLT_MESA, ...
"""

# --- Dataclasses ---
from stellar_jax.config.params import StellarParams
from stellar_jax.config.solver_config import SolverConfig
from stellar_jax.config.contracts import HasPhysicsKnobs, N_COMP_DEFAULT, N_HENYEY_DEFAULT

# --- Physical constants (immutable CGS) ---
from stellar_jax.config.constants import (
    G, e_cgs, c_light, sigma_sb, a_rad, k_B, m_H,
    Msun, Lsun, Rsun,
    Y_BBN, DY_DZ, SECONDS_PER_YEAR, Q_PER_G,
)

# --- Calibration results ---
from stellar_jax.config.calibration import ALPHA_SOLAR, Y0_SOLAR

# --- Mesh defaults and solver tuning ---
from stellar_jax.config.mesh_defaults import (
    ALPHA_MLT, N_SHOOT, N_HENYEY, EPS_FD, N_NEWTON_COLD, N_NEWTON_WARM,
    N_COMP, COMP_MFRACS, COMP_ZONE_MASSES, compute_zone_masses,
    F_OV,
)

# --- MESA comparison config ---
from stellar_jax.config.mesa_config import (
    MESA_CONFIG, Z_MESA, ALPHA_MLT_MESA, Y_MESA, DIFFUSION_MESA, F_OV_MESA,
)
