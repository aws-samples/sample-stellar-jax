"""Differentiable stellar evolution in JAX — v10: MLT in atmosphere.

Changes from v4 (Krishna Swamy):
  - alpha_mlt threaded as explicit argument (not a JIT-captured global)
  - atmosphere_bc extended to τ_base=100 with MLT convective gradient
  - α sensitivity: comes from mlt_nabla() at τ~1-10 where Γ~O(1)
  - At deep τ, MLT gives ∇≈∇_ad (Γ>>1) regardless of α — correct physics

This module is a thin facade re-exporting the public API from submodules.
The implementation lives in constants.py, microphysics/, transport.py,
structure.py, and evolution.py.

Mass mesh scheme (N_MESH = 600):
  The structure solver uses N_MESH=600 radial shells with quadratic stretching
  (exponent p=2): r_i = R - (R - r_inner)*(i/N)^2, i=0..N. Step size |Δr|
  grows linearly from surface to center, placing ~104 shells in the outer 3%
  of R (resolving the SAL/photosphere where H_p ~ 0.01*R) and ~496 in the
  interior. r_inner = 0.005*R (avoids the coordinate singularity at r=0).

  Richardson extrapolation at N=300/600/1200 confirms convergence order p≈2
  (consistent with RK4 on a quadratically-stretched grid; LeVeque 2007 §6.1).
  At N=600 the relative error in P_c and T_c is < 0.1% vs the Richardson
  extrapolant (see tests/test_mesh.py::test_mesh_richardson).

  Reference: Christensen-Dalsgaard 2008, ApSS 316, 13 (surface concentration).

Adaptive timestepping (varcontrol scheme, Paxton et al. 2013 §4):
  Each timestep is accepted/rejected based on the dimensionless structural
  change varcontrol = max(ΔX_max, Δlog_L, Δlog_Teff, Δlog_T_max, Δlog_ρ_max).
  The next dt is scaled by (target / varcontrol), clamped to [0.5, 1.5]×dt.
  Steps with varcontrol > 4×target are rejected and retried at dt/2.

  Default varcontrol_target = 1e-3 (MESA's MS setting). At this target, the
  timestepping error contributes < 0.005 dex to logTeff — well below the
  ~0.02 dex residual vs MESA, which is dominated by atmosphere/opacity
  interpolation differences (see tests/test_evolution.py::test_mesa_residual_attribution).

  Reference: Paxton et al. 2013, ApJS 208, 4, §4.1.
"""
import os
import time
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

jax.config.update('jax_enable_x64', True)

# ------------------------------------------------------------------
# Re-export everything from submodules to preserve public API
# ------------------------------------------------------------------

# Constants
from stellar_jax.config.constants import (
    G, c_light, sigma_sb, a_rad, k_B, m_H,
    Msun, Lsun, Rsun, Y_BBN, DY_DZ, SECONDS_PER_YEAR,
    Q_PER_G,
)
from stellar_jax.config.mesh_defaults import (
    ALPHA_MLT, N_SHOOT, EPS_FD, N_NEWTON_COLD, N_NEWTON_WARM,
    N_COMP, COMP_MFRACS, COMP_ZONE_MASSES, compute_zone_masses, F_OV,
)
from stellar_jax.config.calibration import ALPHA_SOLAR, Y0_SOLAR

# Microphysics
from stellar_jax.microphysics.opacity import (
    OPAL_X, OPAL_Z, OPAL_LOGT, OPAL_LOGR, OPAL_LK,
    FERG_X, FERG_Z, FERG_LOGT, FERG_LOGR, FERG_LK,
    _interp4d_logkappa, opal_kappa, ferguson_kappa, _ferguson_weight, kappa,
    compton_kappa, _compton_blend_weight,
)
from stellar_jax.microphysics.eos import (
    _build_eos_pgrid, EOS_X, EOS_Z, EOS_LOGT, EOS_LOGP,
    EOS_LOGRHO, EOS_MU, EOS_NAD, EOS_CHI_RHO, EOS_CHI_T,
    eos_lookup, eos_lookup_bicubic,
)
from stellar_jax.microphysics.nuclear import epsilon_nuclear, _eps_nuclear_np
from stellar_jax.microphysics.neutrino import epsilon_neutrino

# Transport
from stellar_jax.transport import mlt_nabla

# Structure
from stellar_jax.structure import (
    initial_guess, interp_X_at_mass, atmosphere_bc,
    shoot_xprofile, shoot_xprofile_residual, newton_solve_xprofile,
    get_structure_profile, sound_speed_profile, shoot_at_resolution,
    newton_solve_at_resolution, zams_initial_model,
)

# Inverse solver (gradient-based parameter inference)
from stellar_jax.inference import infer_parameters

# Evolution (composition, timestepping, ZAMS, solar, I/O)
from stellar_jax.evolution import (
    burn_composition, burn_cno, mix_composition, diffuse_composition,
    _comp_step_with_dt, _comp_step_diffusion, _comp_estimate_dt, _comp_cold_start,
    _zams_properties_jit, zams_properties,
    evolve_star, evolve_star_comp, evolve_star_diagnostic, plot_kippenhahn,
    evolve_solar, solar_residual, solar_calibrate,
    solar_residual_with_zprofile, solar_calibrate_with_zprofile,
    compare_model_s, write_fgong, write_gyre, structure_to_fgong_jax,
    kippenhahn_diagram, read_mesa_history, compare_mesa,
    GRAD_TRUST_STEPS, GradientTrustWarning, GradientTrustError,
    observable_at_target,
)

# NOTE: henyey.py is internal scaffolding for (Henyey relaxation
# structure solver). It is NOT re-exported here because it does not yet replace
# the shooting solver — re-exporting would violate the "one code path" rule.
# It will become the public API when integrates it into
# evolve_star as THE structure solver.

# NOTE: oscillations.py is a post-processing diagnostic module (NumPy/SciPy)
# that is intentionally OUTSIDE the JAX autodiff graph. It is not re-exported
# here to maintain the single-code-path rule: stellar.py is the differentiable
# API; oscillation analysis is a separate validation tool.
# Usage: `from oscillations import compute_oscillation_freqs, ...`
