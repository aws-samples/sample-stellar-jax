"""calibration/ — Solar calibration + MESA/Model S comparison.

A leaf consumer of evolution/ (calls evolve_star/evolve_solar).
Never imported by other physics modules.

Public API:
  Solar:       evolve_solar, solar_residual, solar_residual_henyey,
               solar_residual_with_zprofile, solar_calibrate,
               solar_calibrate_henyey, solar_calibrate_with_zprofile
  Comparison:  compare_model_s, compare_mesa, read_mesa_history
  Diagnostic:  evolve_star_diagnostic, plot_kippenhahn, kippenhahn_diagram
  Utility:     lagrange_interp_3pt
"""

from stellar_jax.calibration.solar import (
    evolve_solar,
    solar_residual,
    solar_residual_henyey,
    solar_residual_with_zprofile,
    solar_calibrate,
    solar_calibrate_henyey,
    solar_calibrate_with_zprofile,
)
from stellar_jax.calibration.comparison import (
    compare_model_s,
    compare_mesa,
    read_mesa_history,
)
from stellar_jax.calibration.diagnostic import (
    evolve_star_diagnostic,
    plot_kippenhahn,
    kippenhahn_diagram,
)
from stellar_jax.calibration.interpolation import lagrange_interp_3pt

__all__ = [
    'evolve_solar',
    'solar_residual',
    'solar_residual_henyey',
    'solar_residual_with_zprofile',
    'solar_calibrate',
    'solar_calibrate_henyey',
    'solar_calibrate_with_zprofile',
    'compare_model_s',
    'compare_mesa',
    'read_mesa_history',
    'evolve_star_diagnostic',
    'plot_kippenhahn',
    'kippenhahn_diagram',
    'lagrange_interp_3pt',
]
