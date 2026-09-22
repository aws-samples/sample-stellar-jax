"""evolution/config.py — Focused frozen dataclasses for _evolve_star_jit static args.

These group the 16 individual static_argnames of _evolve_star_jit into 3 focused,
hashable config objects. Because @dataclass(frozen=True) is hashable, JAX's JIT
accepts them directly as static_argnames — no unpacking needed.

The set of values that are STATIC vs TRACED is IDENTICAL to before this refactor:
  STATIC (in these configs):  max_steps, diffusion, grad_window, z_feedback,
      bicubic_opacity, fixed_dt, freeze_schedule, replay_schedule, replay_mesh,
      replay_comp, adaptive_mesh, has_idle_guard, eps_grav_flag, detach_eps_grav_carry,
      xh_cntr_limit, xh_cntr_hard_limit, delta_lgT_cntr_limit, delta_lgRho_cntr_limit
  TRACED (explicit args):  mass, Z, alpha_mlt, Y_init, f_ov, varcontrol_target,
      opacity_factor, eps_nuc_factor, diffusion_factor, t_max, dt_schedule,
      mesh_schedule, comp_schedule

Changing any field in these configs produces a new XLA compilation (~16-min penalty).
Keep the set of distinct config instances small (one per experiment, not per star).
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SolverNumerics:
    """Solver algorithmic settings — ALL fields are static (trigger recompile).

    Groups the timestepping mode, mesh mode, limiter thresholds, and schedule
    flags that were previously individual static_argnames.
    """
    max_steps: int = 500
    fixed_dt: Optional[float] = None
    adaptive_mesh: bool = True
    freeze_schedule: bool = False
    replay_schedule: bool = False
    replay_mesh: bool = False
    replay_comp: bool = False
    has_idle_guard: bool = False
    xh_cntr_limit: float = 0.01
    xh_cntr_hard_limit: float = 0.05
    delta_lgT_cntr_limit: float = 0.01
    delta_lgRho_cntr_limit: float = 0.05


@dataclass(frozen=True)
class GradientOptions:
    """Gradient/adjoint settings — ALL fields are static.

    Controls the gradient windowing and eps_grav coupling flags.
    """
    grad_window: Optional[Tuple[int, int]] = None
    eps_grav_flag: bool = True
    detach_eps_grav_carry: bool = False


@dataclass(frozen=True)
class PhysicsToggles:
    """Physics feature toggles — ALL fields are static.

    Controls which physics modules are compiled into the step function.
    Changing any field produces a new XLA compilation (different cache key).
    """
    diffusion: bool = True
    z_feedback: bool = False
    bicubic_opacity: bool = True
    helm_eos: bool = False


class TracedConstants:
    """Traced (differentiable) values that are constant across all steps.

    NOT a frozen dataclass — these are JAX arrays, not hashable Python values.
    This is a plain class used purely for argument grouping; it is never a
    static_argname. The step_fn closure captures these from the enclosing
    _evolve_star_jit scope.

    All fields are jnp.float64 scalars or arrays, differentiable w.r.t. the
    evolve_star input parameters (mass, Z, alpha_mlt, etc.).
    """
    __slots__ = (
        'mass', 'Z', 'alpha_mlt', 'f_ov', 'M_star_cgs', 'atm_ratio',
        'q_mesh', 'N_s', 'varcontrol_target', 'reject_threshold',
        't_max_sec', 'dt_grow', 'dt_shrink', 'delta_lgL_hard',
        'delta_lgTe_hard', 'opacity_factor', 'eps_nuc_factor',
        'diffusion_factor',
    )

    def __init__(self, *, mass, Z, alpha_mlt, f_ov, M_star_cgs, atm_ratio,
                 q_mesh, N_s, varcontrol_target, reject_threshold,
                 t_max_sec, dt_grow, dt_shrink, delta_lgL_hard,
                 delta_lgTe_hard, opacity_factor, eps_nuc_factor,
                 diffusion_factor):
        self.mass = mass
        self.Z = Z
        self.alpha_mlt = alpha_mlt
        self.f_ov = f_ov
        self.M_star_cgs = M_star_cgs
        self.atm_ratio = atm_ratio
        self.q_mesh = q_mesh
        self.N_s = N_s
        self.varcontrol_target = varcontrol_target
        self.reject_threshold = reject_threshold
        self.t_max_sec = t_max_sec
        self.dt_grow = dt_grow
        self.dt_shrink = dt_shrink
        self.delta_lgL_hard = delta_lgL_hard
        self.delta_lgTe_hard = delta_lgTe_hard
        self.opacity_factor = opacity_factor
        self.eps_nuc_factor = eps_nuc_factor
        self.diffusion_factor = diffusion_factor
