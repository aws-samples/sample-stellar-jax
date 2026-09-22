"""evolution/timestep.py — Adaptive timestep control: varcontrol, accept/reject, limiters.

Pure functions: compute values and decisions. All stop_gradient annotations
are at the CALL SITE in step.py (the orchestrator), never here.

Design ref: the module redesign spec §2.
MESA ref: star/private/timestep.f90 (varcontrol, check_change, limiters).

H211b low-pass controller (Söderlind & Wang 2006, JCAM 185:225-243):
  MESA ref: timestep.f90:2362-2430 (filter_dt_next subroutine).
  β1=β2=0.25/order, α2=0.25. Atan limiter κ=10.
  Falls back to 1st-order when no history (dt_old=0 or dt_limit_ratio_old=0).

Central limiters (δlgT_cntr, δlgRho_cntr):
  MESA ref: timestep.f90:1367-1440 (check_dlgT_cntr_change, check_dlgRho_cntr_change).
  Each calls check_change() → dt_limit_ratio = |Δ| / limit (0 if ≤ 1).
  MESA defaults: delta_lgT_cntr_limit=0.01, delta_lgRho_cntr_limit=0.05, hard_limits disabled.

Worst-offender arbitration:
  MESA ref: timestep.f90:340 — maxloc(dt_limit_ratio(1:numTlim)).
  Single worst ratio feeds the H211b controller (not sequential multiplicative shrinks).
  NOTE: varcontrol passes its raw ratio (can be < 1); δlg limiters + XH zero when <= 1
  (check_change pattern, timestep.f90:763). The maxloc picks the maximum of all.
"""
import jax.numpy as jnp

from stellar_jax.config.constants import Q_PER_G, SECONDS_PER_YEAR
from stellar_jax.microphysics.neutrino import epsilon_neutrino


# ======================================================================
# H211b LOW-PASS CONTROLLER — Söderlind & Wang (2006)
# MESA ref: timestep.f90:2362-2430 (filter_dt_next)
# ======================================================================

def atan_limiter(x, kappa=10.0):
    """Smooth nonlinear limiter matching MESA's `limiter` function.

    MESA ref: timestep.f90:2420 — `limiter = 1 + kappa * atan((x-1)/kappa)`.
    For x >= 0, κ=10: output ∈ [0.003, 16.7]; limiter(1) = 1.

    Parameters
    ----------
    x : scalar — input ratio (typically target/dt_limit_ratio)
    kappa : float — limiter bandwidth (MESA default: 10)

    Returns
    -------
    scalar — limited value, smooth and bounded
    """
    return 1.0 + kappa * jnp.arctan((x - 1.0) / kappa)


def h211b_filter_dt_next(dt, dt_old, dt_limit_ratio, dt_limit_ratio_old,
                         order=1.0):
    """H211b 2nd-order digital filter for dt_next.

    MESA ref: timestep.f90:2381-2414 (filter_dt_next body).
    Uses the atan limiter for ratio smoothing. Falls back to 1st-order
    controller when no history is available (dt_old <= 0 or
    dt_limit_ratio_old <= 0), matching MESA's guard at line 2381.

    Parameters
    ----------
    dt : scalar — timestep just completed (seconds)
    dt_old : scalar — timestep of the PREVIOUS step (seconds); 0 = no history
    dt_limit_ratio : scalar — worst-offender ratio this step (varcontrol raw ∈ (0,∞);
        central/XH limiters zero when not binding. Value < 1 = step quality was good)
    dt_limit_ratio_old : scalar — worst-offender ratio from previous step; 0 = no history
    order : float — controller order (MESA uses 1.0 with varcontrol)

    Returns
    -------
    dt_next : scalar — filtered timestep for next step (seconds)
    """
    beta1 = 0.25 / order
    beta2 = 0.25 / order
    alpha2 = 0.25

    # Ensure non-zero ratio (MESA line 2378: max(1e-10, dt_limit_ratio_in))
    dt_limit_ratio_safe = jnp.maximum(dt_limit_ratio, 1e-10)
    dt_limit_ratio_target = 1.0  # MESA: target is always 1.0

    # Check for history availability (MESA line 2381)
    has_history = (dt_limit_ratio_old > 0.0) & (dt_old > 0.0)

    # 2nd-order H211b controller (MESA lines 2382-2394)
    ratio = atan_limiter(dt_limit_ratio_target / dt_limit_ratio_safe)
    ratio_prev = atan_limiter(dt_limit_ratio_target /
                              jnp.maximum(dt_limit_ratio_old, 1e-10))
    limtr = atan_limiter(
        jnp.power(ratio, beta1) *
        jnp.power(ratio_prev, beta2) *
        jnp.power(dt / jnp.maximum(dt_old, 1e-10), -alpha2)
    )
    dt_next_h211b = dt * limtr

    # 1st-order fallback (MESA lines 2406-2412)
    dt_next_first_order = dt * dt_limit_ratio_target / dt_limit_ratio_safe

    return jnp.where(has_history, dt_next_h211b, dt_next_first_order)


# ======================================================================
# CENTRAL LIMITERS — δlgT_cntr, δlgRho_cntr
# MESA ref: timestep.f90:1367-1440 (check_dlgT_cntr_change, check_dlgRho_cntr_change)
# ======================================================================

def compute_central_limiter_ratio(change, limit):
    """Compute dt_limit_ratio for a single central quantity change.

    MESA ref: timestep.f90:755-766 (check_change body).
    Returns abs_change / limit if > 1 (limiter binding), else 0.
    If limit <= 0, the limiter is disabled (returns 0).

    Parameters
    ----------
    change : scalar — Δ(log10 quantity) at the center this step
    limit : float — soft limit (MESA: delta_lgT_cntr_limit or delta_lgRho_cntr_limit)

    Returns
    -------
    dt_limit_ratio : scalar — 0 if not binding, > 1 if binding
    """
    abs_change = jnp.abs(change)
    ratio = abs_change / jnp.maximum(limit, 1e-30)
    # MESA: dt_limit_ratio = 0 if ratio <= 1 (line 766)
    return jnp.where(ratio > 1.0, ratio, 0.0)


def compute_central_limiters(logT_cntr, logT_cntr_old, logRho_cntr, logRho_cntr_old,
                             delta_lgT_cntr_limit, delta_lgRho_cntr_limit):
    """Compute dt_limit_ratios for central T and ρ changes.

    MESA ref: timestep.f90:1367-1395 (check_dlgT_cntr_change) and
    timestep.f90:1412-1440 (check_dlgRho_cntr_change). Each evaluates
    check_change(Δ, limit) → abs(Δ)/limit if >1, else 0.

    Known omission: MESA guards delta_lgT_cntr with
    ``delta_lgT_cntr_limit_only_after_near_zams`` (default .true.,
    timestep.f90:1379-1381) — the T-center limiter is disabled while
    the star is still near ZAMS (X_center > 0.1 AND L_nuc/L_phot < limit).
    This guard does NOT apply to delta_lgRho_cntr (which fires unconditionally).
    We omit the near_zams guard: the limiter fires unconditionally for both
    T and ρ.  This is harmless on the MS (central T changes are small, so the
    limiter never binds) and conservative on the SGB/RGB (fires earlier than
    MESA would).  CONSTRAINT: lax.scan fixed control-flow — the guard requires
    an evolutionary-phase branch we avoid for differentiability.

    Parameters
    ----------
    logT_cntr : scalar — log10(T_center) this step
    logT_cntr_old : scalar — log10(T_center) previous step
    logRho_cntr : scalar — log10(ρ_center) this step
    logRho_cntr_old : scalar — log10(ρ_center) previous step
    delta_lgT_cntr_limit : float — soft limit (MESA default: 0.01)
    delta_lgRho_cntr_limit : float — soft limit (MESA default: 0.05)

    Returns
    -------
    ratio_lgT_cntr : scalar — dt_limit_ratio for central T
    ratio_lgRho_cntr : scalar — dt_limit_ratio for central ρ
    """
    delta_lgT = logT_cntr - logT_cntr_old
    delta_lgRho = logRho_cntr - logRho_cntr_old

    ratio_lgT = compute_central_limiter_ratio(delta_lgT, delta_lgT_cntr_limit)
    ratio_lgRho = compute_central_limiter_ratio(delta_lgRho, delta_lgRho_cntr_limit)

    return ratio_lgT, ratio_lgRho


# ======================================================================
# WORST-OFFENDER ARBITRATION
# MESA ref: timestep.f90:340 — i_limit = maxloc(dt_limit_ratio(1:numTlim))
# ======================================================================

def compute_worst_offender_ratio(varcontrol, varcontrol_target,
                                 delta_XH_cntr, xh_cntr_limit,
                                 ratio_lgT_cntr, ratio_lgRho_cntr):
    """Compute the single worst-offender dt_limit_ratio across all limiters.

    MESA ref: timestep.f90:340 — `maxloc(dt_limit_ratio(1:numTlim), dim=1)`.
    Collects ratios from varcontrol, XH central, δlgT_cntr, δlgRho_cntr.
    The maximum feeds the H211b controller (single-input, not sequential).

    Parameters
    ----------
    varcontrol : scalar — current varcontrol quality factor
    varcontrol_target : scalar — target value
    delta_XH_cntr : scalar — |ΔX_H| at center this step
    xh_cntr_limit : float — XH soft limiter threshold
    ratio_lgT_cntr : scalar — from compute_central_limiters
    ratio_lgRho_cntr : scalar — from compute_central_limiters

    Returns
    -------
    worst_ratio : scalar — max dt_limit_ratio (feeds H211b controller)
    """
    # Varcontrol ratio: varcontrol / target (MESA: check_varcontrol_limit, line 2273).
    # MESA passes the RAW ratio — no zeroing when <= 1. When < 1 (good step), the
    # H211b controller uses it to compute proportional growth (limiter(target/0.5) ≈ 2).
    # This is distinct from the δlg limiters which use check_change (zeros when <= 1).
    varcontrol_ratio = varcontrol / jnp.maximum(varcontrol_target, 1e-30)

    # XH central ratio: delta_XH / limit (check_change pattern — zeros when <= 1)
    xh_ratio = delta_XH_cntr / jnp.maximum(xh_cntr_limit, 1e-30)
    xh_ratio = jnp.where(xh_ratio > 1.0, xh_ratio, 0.0)

    # Max across all limiter ratios (worst offender wins)
    worst = jnp.maximum(varcontrol_ratio, xh_ratio)
    worst = jnp.maximum(worst, ratio_lgT_cntr)
    worst = jnp.maximum(worst, ratio_lgRho_cntr)

    return worst


def compute_varcontrol(delta_logL, delta_logTe, max_dX):
    """Compute the varcontrol quality factor for this step.

    The varcontrol is the maximum of three scaled changes. Larger values
    mean the step took too large a bite.

    MESA ref: timestep.f90 — varcontrol_target mechanism (Paxton+2013 §4.2).

    Parameters
    ----------
    delta_logL : scalar — |Δlog10(L)| this step
    delta_logTe : scalar — |Δlog10(Teff)| this step
    max_dX : scalar — max |ΔX| across composition grid

    Returns
    -------
    varcontrol : scalar — quality factor for this step
    """
    return jnp.maximum(jnp.maximum(delta_logL, delta_logTe), max_dX)


def apply_xh_limiter(dt_next, delta_XH_cntr, xh_cntr_limit, xh_cntr_hard_limit,
                     dt_at_floor):
    """Apply central-hydrogen timestep limiter.

    MESA ref: timestep.f90:1740 (check_XH_cntr), line 732 (check_change).
    Soft: scale dt proportionally. Hard: reject step.

    Parameters
    ----------
    dt_next : scalar — proposed dt for next step
    delta_XH_cntr : scalar — |ΔX_c| this step (already stop_gradient'd by caller)
    xh_cntr_limit : float — soft limit
    xh_cntr_hard_limit : float — hard limit
    dt_at_floor : bool — whether dt is already at the floor

    Returns
    -------
    dt_next : scalar — adjusted dt
    xh_hard_reject : bool — whether the hard limit triggers rejection
    """
    # Soft limiter: scale dt_next proportionally when δXH > limit
    xh_ratio = delta_XH_cntr / xh_cntr_limit
    xh_shrink = jnp.where(xh_ratio > 1.0,
                           xh_cntr_limit / jnp.maximum(delta_XH_cntr, 1e-30),
                           1.0)
    dt_next = dt_next * xh_shrink

    # Hard rejection: if |ΔXc| exceeds hard limit (not at dt floor)
    xh_hard_reject = (delta_XH_cntr > xh_cntr_hard_limit) & (~dt_at_floor)

    return dt_next, xh_hard_reject


def apply_eps_nuc_limiter(dt_next, shell_data_sg, cur_logT, cur_logRho,
                          X_out, Z):
    """Apply nuclear-burning dt_max safety limiter.

    MESA ref: timestep.f90, dt_div_dt_cell (Paxton+2013 §5.2).
    stop_gradient: all inputs should be detached by the caller (safety limiter,
    not a physical sensitivity path).

    Parameters
    ----------
    dt_next : scalar — proposed dt
    shell_data_sg : (N_s, ...) — shell data (stop_gradient'd)
    cur_logT : (N_s,) — current log10(T) (stop_gradient'd)
    cur_logRho : (N_s,) — current log10(rho) (stop_gradient'd)
    X_out : (N_COMP,) — hydrogen profile (stop_gradient'd)
    Z : scalar — metallicity

    Returns
    -------
    dt_next : scalar — limited dt
    """
    eps_nuc_max = jnp.maximum(jnp.max(shell_data_sg[:, 0]), 1e-10)
    T_center = 10.0**cur_logT[-1]
    rho_center = 10.0**cur_logRho[-1]
    X_center_val = X_out[0]
    eps_nu_center = epsilon_neutrino(rho_center, T_center, X_center_val, Z)
    eps_max = jnp.maximum(eps_nuc_max, eps_nu_center)
    dt_max = 0.10 * Q_PER_G / eps_max
    return jnp.minimum(dt_next, dt_max)


def apply_he_ignition_halt(dt_next, log_Tc):
    """Smooth sigmoid halt at log_Tc > 7.9 (He ignition threshold).

    Width 0.02 dex gives a smooth transition for differentiability.
    Ref: He ignites at log T_c ≈ 7.9-8.0 (Kippenhahn & Weigert §33;
    Salaris & Cassisi 2005 §5.9).

    Parameters
    ----------
    dt_next : scalar — proposed dt
    log_Tc : scalar — log10(T_center) for this step

    Returns
    -------
    dt_next : scalar — halted dt (→ 0 smoothly at ignition)
    """
    he_ignition_halt = jnp.float64(1.0) / (1.0 + jnp.exp(-(log_Tc - 7.9) / 0.02))
    return dt_next * (1.0 - he_ignition_halt)


def apply_t_max_cap(dt_next, t_new, t_max_sec):
    """Cap dt to not overshoot t_max.

    Parameters
    ----------
    dt_next : scalar — proposed dt
    t_new : scalar — current age (seconds)
    t_max_sec : scalar — maximum age (seconds)

    Returns
    -------
    dt_next : scalar — capped dt
    """
    dt_next = jnp.where(t_new >= t_max_sec, 0.0, dt_next)
    dt_next = jnp.minimum(dt_next, jnp.maximum(t_max_sec - t_new, 0.0))
    return dt_next


def apply_dt_floor(dt_next):
    """Clamp dt to a physical minimum (prevents adaptive cascade collapse).

    MESA: dt < min_timestep_limit → terminate. In lax.scan: clamp.

    Parameters
    ----------
    dt_next : scalar — proposed dt

    Returns
    -------
    dt_next : scalar — floored dt
    dt_at_floor : bool — True if dt was at or below the floor before this call
    """
    dt_floor = 1e5 * SECONDS_PER_YEAR
    dt_floored = jnp.maximum(dt_next, dt_floor)
    return dt_floored
