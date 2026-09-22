"""Structural type contracts for config/ consumers.

These define the interface contracts that downstream modules depend on,
enforcing structural expectations via Protocols and shape constants.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class HasPhysicsKnobs(Protocol):
    """Any object providing the physics-calibration knob interface.

    Downstream microphysics modules can type-check against this protocol
    to ensure they receive an object with the required knob fields.

    Example usage in a microphysics module:
        def kappa(T, rho, X, Z, knobs: HasPhysicsKnobs):
            return _raw_kappa(T, rho, X, Z) * knobs.opacity_factor
    """
    @property
    def opacity_factor(self) -> float: ...
    @property
    def eps_nuc_factor(self) -> float: ...
    @property
    def diffusion_factor(self) -> float: ...


# Shape contracts (documentation; enforced by tests)
N_COMP_DEFAULT: int = 200
N_HENYEY_DEFAULT: int = 600
