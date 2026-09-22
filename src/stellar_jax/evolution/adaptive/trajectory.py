"""Trajectory records for the adaptive forward pass.

TrajectoryStep and Trajectory dataclasses that store the per-step state
for gradient replay and analysis. Also contains shell_resolution() for
counting zones in the H-burning shell.

Moved from adaptive_forward.py into evolution/adaptive/ (#1115).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrajectoryStep:
    """Per-step record for gradient replay."""
    step_index: int
    accepted: bool
    t_yr: float           # age in years at start of step
    dt_yr: float          # timestep in years
    q_mesh: np.ndarray    # structure mesh (N+1,) — face positions
    X_profile: np.ndarray # hydrogen on unified mesh (N,)
    Y_profile: np.ndarray
    Z_val: float
    C12_profile: np.ndarray
    C13_profile: np.ndarray
    N14_profile: np.ndarray
    y_henyey: np.ndarray  # converged state (N, 4)
    logL: float
    logTe: float
    log_Tc: float
    log_rhoc: float
    n_zones: int          # number of zones at this step


@dataclass
class Trajectory:
    """Full trajectory for gradient replay."""
    mass: float
    Z: float
    alpha_mlt: float
    steps: list = field(default_factory=list)

    def append(self, step: TrajectoryStep):
        self.steps.append(step)

    @property
    def n_accepted(self):
        return sum(1 for s in self.steps if s.accepted)

    def dt_schedule(self) -> np.ndarray:
        """Extract the per-step dt schedule (accepted steps only).

        Returns
        -------
        dt_schedule : (N_accepted,) float64 — timestep in seconds per accepted step.
            Suitable for evolve_star(dt_schedule=...) replay.

        MESA ref: timestep.f90 — the recorded dt is the *consumed* timestep, not
        the proposed dt_next; replay replays the consumed sequence.
        """
        from stellar_jax.config.constants import SECONDS_PER_YEAR
        accepted = [s for s in self.steps if s.accepted]
        return np.array([s.dt_yr * SECONDS_PER_YEAR for s in accepted],
                        dtype=np.float64)

    def mesh_schedule(self) -> np.ndarray:
        """Extract the per-step structure mesh schedule (accepted steps only).

        Returns
        -------
        mesh_schedule : (N_accepted, N_zones+1) float64 — fractional-mass face
            positions per accepted step. N_zones is fixed (600), so mesh_schedule
            is always (N_accepted, 601).

        MESA ref: evolve.f90:1882-1886 — do_mesh() in prepare_for_new_step sets
        the structure mesh BEFORE each Newton solve. The adaptive forward records
        the per-step mesh in TrajectoryStep.q_mesh; this stacks them for replay.
        """
        accepted = [s for s in self.steps if s.accepted]
        return np.array([s.q_mesh for s in accepted], dtype=np.float64)

    def comp_schedule(self) -> dict:
        """Extract per-step compositions on the recorded 600-zone grid.

        Returns the compositions directly on the adaptive forward's own
        600-zone mesh — NO 600→200 remap.  This is the GRADSOLVE-faithful
        approach (arXiv:2609.02876 §3.1): the replay must use the same
        one-step map as the record.  The 600→200 conservative remap lost
        H-shell resolution and made replay-map ≠ record-map.

        Returns
        -------
        comp_schedule : dict with keys
            'X'    : (N_accepted, N_zones) float64 — hydrogen mass fraction
            'Y'    : (N_accepted, N_zones) float64 — helium mass fraction
            'N14'  : (N_accepted, N_zones) float64 — nitrogen-14 mass fraction
            'C12'  : (N_accepted, N_zones) float64 — carbon-12 mass fraction
            'C13'  : (N_accepted, N_zones) float64 — carbon-13 mass fraction
            'comp_mfracs' : (N_accepted, N_zones) float64 — per-step cell-
                center fractional mass positions (from 0.5*(q[:-1]+q[1:])).
                The Henyey solver uses these as the composition grid for
                interp_X_at_mass, replacing the static COMP_MFRACS default.

        N14 is critical for SGB replay fidelity: the CNO shell-burning rate
        is directly proportional to X_N14.  C12/C13 complete the CNO set.

        The compositions are stop_gradient'd in the replay (GP-12:
        COMP_SCHEDULE).  The gradient flows through M → Henyey IFT →
        structure → σ²; composition is frozen context.

        MESA ref: struct_burn_mix.f90:82-100 — MESA's operator split
        only covers the BURN; MIXING remains coupled (line 144: nvar=
        nvar_total).  CONSTRAINT: our GP-12 freeze is stronger — both
        burn and mix frozen — because the lax.scan replay cannot evolve
        composition on the adaptive mesh (JAX fixed-shape scan carry).
        """
        from stellar_jax.config.mesh_defaults import N_HENYEY

        accepted = [s for s in self.steps if s.accepted]
        N_acc = len(accepted)

        X_all = np.zeros((N_acc, N_HENYEY), dtype=np.float64)
        Y_all = np.zeros((N_acc, N_HENYEY), dtype=np.float64)
        N14_all = np.zeros((N_acc, N_HENYEY), dtype=np.float64)
        C12_all = np.zeros((N_acc, N_HENYEY), dtype=np.float64)
        C13_all = np.zeros((N_acc, N_HENYEY), dtype=np.float64)
        mfracs_all = np.zeros((N_acc, N_HENYEY), dtype=np.float64)

        for i, s in enumerate(accepted):
            X_all[i] = s.X_profile
            Y_all[i] = s.Y_profile
            N14_all[i] = s.N14_profile
            C12_all[i] = s.C12_profile
            C13_all[i] = s.C13_profile
            # Cell-center grid from the (N_zones+1,) face positions.
            mfracs_all[i] = 0.5 * (s.q_mesh[:-1] + s.q_mesh[1:])

        return {'X': X_all, 'Y': Y_all, 'N14': N14_all,
                'C12': C12_all, 'C13': C13_all,
                'comp_mfracs': mfracs_all}

    def summary(self):
        """Return a dict of arrays for easy analysis."""
        accepted = [s for s in self.steps if s.accepted]
        return {
            'star_age': np.array([s.t_yr for s in accepted]),
            'log_L': np.array([s.logL for s in accepted]),
            'log_Teff': np.array([s.logTe for s in accepted]),
            'log_Tc': np.array([s.log_Tc for s in accepted]),
            'log_rhoc': np.array([s.log_rhoc for s in accepted]),
            'n_zones': np.array([s.n_zones for s in accepted]),
        }


def shell_resolution(trajectory, step_idx=-1):
    """Count zones resolving the H-burning shell at a given step.

    The shell is where X transitions from ~0.7 (envelope) to ~0 (core).
    We count zones where 0.01 < X < 0.65 (the gradient region).

    Returns
    -------
    n_zones_in_shell : int
    shell_width_dq : float — total mass fraction spanned by the shell
    """
    if step_idx < 0:
        step_idx = len(trajectory.steps) + step_idx
    s = trajectory.steps[step_idx]
    X = s.X_profile
    shell_mask = (X > 0.01) & (X < 0.65)
    n_zones = int(np.sum(shell_mask))

    # Width in mass fraction
    q = 0.5 * (s.q_mesh[:-1] + s.q_mesh[1:])  # cell centers from face positions
    if len(q) == len(X):
        shell_q = q[shell_mask]
        width = float(shell_q[-1] - shell_q[0]) if len(shell_q) > 1 else 0.0
    else:
        width = 0.0
    return n_zones, width
