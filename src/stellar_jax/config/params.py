"""StellarParams — frozen dataclass for physics parameters.

All fields are JAX-traced (differentiable). Changing a value does NOT trigger
XLA recompilation — only SolverConfig (static) changes do that.

Open/Closed extension point: adding a new physics-calibration knob requires
only 1 new field here (default 1.0) + 1 multiply at the consumption point
in the microphysics module. No signature change to evolve_star, no carry-shape
change, no recompile trigger.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StellarParams:
    """Physics parameters — ALL fields are JAX-traced (differentiable).

    These flow through jit/grad as traced values. Changing a value does NOT
    trigger recompilation — only changes to SolverConfig (static) do that.

    Attributes
    ----------
    mass : float
        Stellar mass in solar masses [M_sun].
    Z : float
        Initial metallicity (metal mass fraction).
    alpha_mlt : float
        Mixing-length parameter (code-dependent; calibrated via solar_calibrate).
    f_ov : float
        Step-overshoot parameter (fraction of H_P).
    Y_init : float or None
        Initial helium mass fraction. None → Y_BBN + DY_DZ * Z.
    t_max : float or None
        Maximum evolution age [years]. None → effectively infinite (1e30 yr).
    opacity_factor : float
        Multiplies kappa(T, rho, X, Z) output. Default 1.0 = standard physics.
    eps_nuc_factor : float
        Multiplies epsilon_nuclear output. Default 1.0 = standard physics.
    diffusion_factor : float
        Multiplies diffusion velocities. Default 1.0 = standard physics.

    Differentiability contract
    --------------------------
    ALL fields are TRACED by JAX (flow through jit, differentiable via grad).
    jax.grad(lambda p: f(p).log_L[-1])(params) gives ∂logL/∂mass, etc.
    No @custom_vjp boundary here — this is a pure data container.
    """
    mass: float = 1.0
    Z: float = 0.014
    alpha_mlt: float = 1.9
    f_ov: float = 0.016
    Y_init: Optional[float] = None
    t_max: Optional[float] = None

    # Physics-calibration knobs (Open/Closed extension point):
    # Each multiplies the corresponding microphysics output at ONE consumption
    # point. Default 1.0 = standard physics. New knobs: add field + 1 multiply.
    opacity_factor: float = 1.0
    eps_nuc_factor: float = 1.0
    diffusion_factor: float = 1.0
