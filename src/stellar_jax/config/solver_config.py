"""SolverConfig — frozen dataclass for algorithmic (static) settings.

All fields are STATIC — they are passed via static_argnames to jax.jit.
Changing any field produces a new XLA compilation (~16-min penalty). Use
different SolverConfig instances sparingly (one per experiment, not per star).

NOTE: As of #1063, the internal _evolve_star_jit uses focused frozen config
objects (SolverNumerics, GradientOptions, PhysicsToggles in evolution/config.py)
as static_argnames directly. SolverConfig remains the CALLER-FACING API; the
evolve_star public facade constructs the internal configs from its kwargs.
The to_jit_kwargs() method is DEPRECATED — it is no longer used by evolve_star.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SolverConfig:
    """Algorithmic settings — ALL fields are STATIC (trigger recompile on change).

    These are Python values passed via static_argnums/static_argnames to jax.jit.
    Changing any field produces a new XLA compilation (16-min penalty). Use
    different SolverConfig instances sparingly (one per experiment, not per star).

    Attributes
    ----------
    max_steps : int
        Maximum number of evolution steps (lax.scan trip count).
    diffusion : bool
        Enable element diffusion (Thoul et al. 1994).
    adaptive_mesh : bool
        Enable adaptive composition mesh reparameterization.
    z_feedback : bool
        Enable CNO Z-feedback into opacity.
    bicubic_opacity : bool
        Use bicubic T/R + quadratic Z opacity interpolation.
    freeze_schedule : bool
        Detach adaptive-dt schedule from gradient tape (frozen-schedule adjoint).
    replay_schedule : bool
        Replay a recorded dt schedule instead of adaptive timestepping.
    has_idle_guard : bool
        Enable stop_gradient on idle steps (when t_max reached).
    eps_grav_enabled : bool
        Enable gravitational energy source term (eps_grav Form C).
    xh_cntr_limit : float
        Central H depletion soft limit for dt control.
    xh_cntr_hard_limit : float
        Central H depletion hard limit (halt evolution).
    grad_window : tuple of (int, int) or None
        Gradient windowing bounds [k0, k1). None → full differentiation.
    fixed_dt : float or None
        Fixed timestep [years]. None → adaptive timestepping.
    varcontrol_target : float
        Dimensionless accuracy target for adaptive timestep controller.
        NOTE: despite living in SolverConfig (semantically a solver setting),
        this field is currently TRACED (not static_argnames) in _evolve_star_jit
        — it's converted to jnp.float64 and passed as a traced value. The
        threshold computation (reject = 4*vct) uses jnp.where, not Python if.
        It lives here for grouping convenience; it does NOT trigger recompile.

    XLA recompile contract
    -----------------------
    Every field here except varcontrol_target is a static_argname. The set of
    distinct SolverConfig values used in a session = the set of XLA compilations.
    Keep this small.
    """
    max_steps: int = 500
    diffusion: bool = True
    adaptive_mesh: bool = True
    z_feedback: bool = False
    bicubic_opacity: bool = True
    freeze_schedule: bool = False
    replay_schedule: bool = False
    has_idle_guard: bool = False
    eps_grav_enabled: bool = True
    xh_cntr_limit: float = 0.01
    xh_cntr_hard_limit: float = 0.05
    grad_window: Optional[Tuple[int, int]] = None
    fixed_dt: Optional[float] = None
    varcontrol_target: float = 1e-3

    def static_argnames(self) -> tuple:
        """Return the tuple of field names that are static_argnames at the JIT boundary."""
        return (
            'max_steps', 'diffusion', 'grad_window', 'z_feedback',
            'bicubic_opacity', 'fixed_dt', 'freeze_schedule', 'replay_schedule',
            'adaptive_mesh', 'has_idle_guard', '_eps_grav_flag',
            '_xh_cntr_limit', '_xh_cntr_hard_limit',
        )

    def to_jit_kwargs(self) -> dict:
        """Unpack to keyword arguments for _evolve_star_jit's static args.

        Maps dataclass field names to the legacy _evolve_star_jit parameter names.
        This is the bridge between the structured config and the flat JIT boundary.
        """
        return {
            'max_steps': self.max_steps,
            'diffusion': self.diffusion,
            'adaptive_mesh': self.adaptive_mesh,
            'z_feedback': self.z_feedback,
            'bicubic_opacity': self.bicubic_opacity,
            'freeze_schedule': self.freeze_schedule,
            'replay_schedule': self.replay_schedule,
            'has_idle_guard': self.has_idle_guard,
            '_eps_grav_flag': self.eps_grav_enabled,
            '_xh_cntr_limit': self.xh_cntr_limit,
            '_xh_cntr_hard_limit': self.xh_cntr_hard_limit,
            'grad_window': self.grad_window,
            'fixed_dt': self.fixed_dt,
        }
