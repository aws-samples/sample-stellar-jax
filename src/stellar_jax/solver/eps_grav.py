"""Gravothermal energy rate + energy-row scaling for the IFT adjoint.

References:
  - MESA eps_grav.f90:113-182 (do_std_eps_grav, use_time_centered_eps_grav, θ=0.5)
  - MESA eps_grav.f90:262-395 (eval_eps_grav_composition)
  - Paxton et al. (2019), ApJS 243, 10, §4 (energy equation time-centering)
  - Kippenhahn, Weigert & Weiss (2012), §4.1, eq. 4.18
  - MESA star_utils.f90:3678 (set_energy_eqn_scal)
"""

import jax
import jax.numpy as jnp

from stellar_jax.config.constants import a_rad, k_B, m_H
from stellar_jax.config.physics_floors import PGAS_FRAC_FLOOR
from stellar_jax.microphysics.eos import eos_lookup
from stellar_jax.structure import interp_X_at_mass

# Module-level flag: when True, eps_grav uses θ=0.5 time-centering
# (MESA eps_grav.f90:130, use_time_centered_eps_grav). When False (default),
# uses θ=1 (end-of-step only — log-space derivatives still active).
#
# Default False: time-centering unmasked a composition remap error at 1.2 M☉
# (median logTeff > 0.08 dex) that has since been fixed (order-1 remap,
# now the production path in mesh/remap.py). Enabling θ=0.5 requires re-
# validation of 1.2 M☉ RGB with the order-1 remap to confirm no regression.
# The log-space derivative switch (this module) is safe independently.
# Tracked as. Used by the O2 mutation gate to validate the formula.
EPS_GRAV_TIME_CENTERED = False


def _eps_grav_composition(T_mid, X_mid, X_prev_mid, Z, inv_dt):
    """Composition contribution to eps_grav (MESA eval_eps_grav_composition).

    Computes the energy change at constant (T, ρ) due to composition change:
        eps_grav_composition = -de/dt

    where de = e(X_now, T, ρ) - e(X_prev, T, ρ).

    For a fully-ionized ideal gas (appropriate for stellar interiors where
    eps_grav matters — the H-burning shell and contracting core):
        e_gas = (3/2) * k_B * T / (μ * m_H)
        1/μ = (5X + 3 - Z) / 4  (fully-ionized H+He+metals)

    Therefore:
        de = (3/2) * (k_B * T / m_H) * (1/μ_now - 1/μ_prev)
           = (3/2) * (k_B * T / m_H) * (5/4) * (X_now - X_prev)
           = (15/8) * (k_B * T / m_H) * (X_now - X_prev)

    Since X decreases with time (H burns), dX < 0, so de < 0, and
    eps_grav_composition = -de/dt > 0 (energy is released).

    This matches MESA's general EOS-based approach (eps_grav.f90:262-395)
    specialized to fully-ionized conditions. MESA uses a full EOS call with
    xa_start; we use the analytic ideal-gas formula because:
    (a) CONSTRAINT: our EOS is parametrized by (T, P_gas, X, Z), not (T, ρ, X, Z),
        so a constant-(T,ρ) EOS call at different X requires an iterative solve.
    (b) In the deep interior (T > 10^6 K) where this term matters, the
        fully-ionized ideal gas is exact to < 0.1%.

    References:
      - MESA eps_grav.f90:262-395 (eval_eps_grav_composition)
      - MESA controls.defaults:8095 (include_composition_in_eps_grav = .true.)
      - Kippenhahn, Weigert & Weiss (2012), §13.1 (ideal gas EOS, mean molecular weight)
    """
    # dX = X_now - X_prev (negative when H burns: X decreases)
    dX = X_mid - X_prev_mid

    # de = (15/8) * (k_B * T / m_H) * dX  [erg/g]
    # eps_grav_composition = -de/dt = -(15/8) * (k_B * T / m_H) * dX * inv_dt
    de = (15.0 / 8.0) * (k_B / m_H) * T_mid * dX
    return -de * inv_dt


def _eps_grav_form_c(T_mid, P_mid, cp, nad, ln_T_prev_mid, ln_P_prev_mid, inv_dt,
                    cp_start=None, nad_start=None):
    """Gravothermal energy generation rate — Form C (T,P basis).

    Two modes depending on whether time-centering is active:

    θ=1 (no time-centering, cp_start/nad_start = None):
      Uses the original linear (T,P) Form C (KWW 2012, eq. 4.18):
        eps_grav = -cp*dT/dt + (cp*T*∇_ad/P)*dP/dt
      where dT/dt = (T - T_prev)/dt, dP/dt = (P - P_prev)/dt.

    θ=0.5 (time-centering active, cp_start/nad_start provided):
      Uses MESA's log-space (lnT, lnP) basis (eps_grav.f90:130-182):
        dlnT_dt = (lnT - lnT_start)/dt
        dlnP_dt = (lnP - lnP_start)/dt
        eps_grav_end   = -Cp*T*dlnT_dt + Cp*T*∇_ad*dlnP_dt
        eps_grav_start = -Cp_start*T_start*dlnT_dt + Cp_start*T_start*∇_ad_start*dlnP_dt
        eps_grav = 0.5*eps_grav_end + 0.5*eps_grav_start

      The log-space discretization is required with time-centering because the
      linear form introduces O(Δt²) errors that are amplified by the θ-blend at
      large steps (measured: 0.17 dex for 2.0 M☉ on the RGB with linear+θ=0.5).

    The _start quantities (cp_start, nad_start) are from the PREVIOUS step's
    converged state — constants during Newton iteration. The time derivatives
    are the SAME in both terms (MESA eps_grav.f90:152-153).

    References:
      - MESA eps_grav.f90:113-182 (do_std_eps_grav, use_time_centered_eps_grav)
      - Paxton et al. (2019), ApJS 243, 10, §4 (energy equation time-centering)
      - Kippenhahn, Weigert & Weiss (2012), §4.1, eq. 4.18
    """
    T_prev = jnp.exp(ln_T_prev_mid)

    if cp_start is not None and nad_start is not None:
        # Time-centering active (θ=0.5): use MESA log-space formula
        # MESA eps_grav.f90:152-153: dlnT_dt = wrap_dxh_lnT/dt
        ln_T_mid = jnp.log(T_mid)
        ln_P_mid = jnp.log(P_mid)
        dlnT_dt = (ln_T_mid - ln_T_prev_mid) * inv_dt
        dlnP_dt = (ln_P_mid - ln_P_prev_mid) * inv_dt

        # End-of-step: -Cp*T*dlnT_dt + Cp*T*grada*dlnP_dt (MESA line 163 in (T,P) basis)
        eps_grav_end = -cp * T_mid * dlnT_dt + cp * T_mid * nad * dlnP_dt
        # Start-of-step: same formula with _start coefficients (MESA lines 176-177)
        eps_grav_start = -cp_start * T_prev * dlnT_dt + cp_start * T_prev * nad_start * dlnP_dt
        theta = 0.5
        return theta * eps_grav_end + (1.0 - theta) * eps_grav_start
    else:
        # No time-centering (θ=1): use original linear Form C (KWW eq. 4.18)
        # This is the formula that was on main before.
        P_prev = jnp.exp(ln_P_prev_mid)
        dT_dt = (T_mid - T_prev) * inv_dt
        dP_dt = (P_mid - P_prev) * inv_dt
        return -cp * dT_dt + (cp * T_mid * nad / P_mid) * dP_dt


def _energy_row_scale_factors(y, q_mesh, X_profile, Z, inv_dt, comp_mfracs=None,
                              helm_eos=False):
    """Compute per-cell F3 (energy equation) row-scaling factors.

    MESA reference: star_utils.f90:3678 set_energy_eqn_scal — MESA scales the
    energy equation residual (and its Jacobian row, via auto_diff partials) by
    ``scal = dt / energy_start(k)`` where ``energy_start(k) = s% energy(k)``
    (specific internal energy from EOS at start-of-step). This conditions the
    eps_grav-dominated energy equation so its residual entries are O(1).

    Applied always-on in both the forward Newton (solver/continuation.py) and
    the IFT adjoint backward pass (solver/adjoint.py). In the forward, row 0
    of each interior block (the F3 equation) of both the residual AND Jacobian
    is multiplied by the scale factor. In the backward, the transpose system
    is scaled identically before solving.

    CONSTRAINT deviations from MESA (bounded, labeled):

      1. **cp*T instead of energy_start (~5/3 factor for ideal gas).**
         MESA uses ``energy_start(k)`` = total specific internal energy from
         EOS (≈ cv*T for ideal gas). We use ``cp*T`` because the EOS lookup
         already computes cp (used in the eps_grav Form C residual) but does
         NOT return the total specific internal energy directly. For a
         fully-ionized ideal gas, e = (3/2) kT/(μ mH) while cp*T = (5/2) kT/
         (μ mH), so our scale is ~3/5 of MESA's (more conservative — we scale
         LESS aggressively). The conditioning effect is qualitatively identical:
         the F3 row becomes O(1) in both cases. The fixed point is invariant
         (scaling both sides of J·dy = -R preserves dy). Bounded: factor 5/3
         in the denominator, same order, same sign.

      2. **No center-cell (k=1) special treatment.**
         MESA uses ``scal = 1d-6`` as the base for k=1, yielding
         ``scal = 1d-6 * dt / energy_start(1)`` — 1000× more aggressive
         scaling at the center. Our formulation treats the center cell
         uniformly (same formula as all interior cells). This is because our
         block-0 row is the boundary-condition block (not a cell equation), so
         our first F3 entry (at block index 1) corresponds to the first
         interior cell, which in our grid is already at small but finite mass
         fraction — not the degenerate center point. Deviation bounded: the
         center cell's F3 is still scaled by 1/(cp*T*inv_dt), just not by an
         additional 1e-6 factor.

      3. **dedt_eqn_r_scale cap omitted.**
         MESA caps scal at ``cell_energy_fraction * 1d8`` (star_utils.f90:3691-
         3694), requiring ``total_internal_energy_old`` (a global sum over all
         cells). This introduces a global-to-cell dependency incompatible with
         JAX's per-cell vectorization. The cap only activates for cells with
         < 1e-8 of total energy (practically never for MS/RGB). Omitted as
         CONSTRAINT; deviation negligible in practice.

    The scaling is a no-op when inv_dt = 0 (sequence-of-equilibria, eps_grav off):
    1/max(1, 0) = 1, so the system is unmodified.

    Returns:
        scale: (N_s,) array of scale factors. scale[0] = 1 (BC block, not F3).
               scale[k] = 1/max(1, cp_{cell k-1} * T_mid_{cell k-1} * inv_dt) for k≥1.
    """
    N_s = y.shape[0]
    N_c = N_s - 1

    # Cell midpoint temperatures and pressures (same as _cell_residual)
    ln_T_mid = 0.5 * (y[:N_c, 2] + y[1:N_s, 2])  # (N_c,)
    ln_P_mid = 0.5 * (y[:N_c, 1] + y[1:N_s, 1])  # (N_c,)
    T_mid = jnp.exp(ln_T_mid)
    P_mid = jnp.exp(ln_P_mid)

    # EOS lookup at midpoints (same as _cell_residual for cp)
    logT_mid = jnp.log10(T_mid)
    P_rad_mid = a_rad * T_mid**4 / 3.0
    P_gas_mid = jnp.maximum(P_mid - P_rad_mid, PGAS_FRAC_FLOOR * P_mid)
    logPgas_mid = jnp.log10(P_gas_mid)

    # Interpolate composition at midpoints
    q_mesh_arr = q_mesh[:N_s]
    q_mid = 0.5 * (q_mesh_arr[:N_c] + q_mesh_arr[1:N_s])

    X_mid = jax.vmap(lambda q: interp_X_at_mass(X_profile, q, comp_mfracs))(q_mid)
    # Vectorized EOS lookup for cp at all cell midpoints
    _, _, _, _, cp_cells, _, _ = jax.vmap(
        lambda logT, logP, X: eos_lookup(logT, logP, X, Z, helm_eos=helm_eos)
    )(logT_mid, logPgas_mid, X_mid)  # cp_cells: (N_c,)

    # Scale factor per cell: 1/max(1, cp * T * inv_dt)
    # When inv_dt=0 (eps_grav off), cp*T*0 = 0, max(1, 0) = 1, scale = 1 (no-op).
    eps_grav_stiffness = cp_cells * T_mid * inv_dt  # (N_c,)
    scale_cells = 1.0 / jnp.maximum(1.0, eps_grav_stiffness)  # (N_c,)

    # Build full (N_s,) scale array: block 0 gets scale=1 (BC, not F3)
    scale = jnp.ones(N_s)
    scale = scale.at[1:].set(scale_cells)  # blocks 1..N_s-1 get cell 0..N_c-1 scales

    return scale
