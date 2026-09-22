"""Residual assembly for the Henyey Newton solver.

Contains:
  - _cell_residual: 4 structure equations for one cell (KEEP WHOLE)
  - _cell_residual_raw: same with plain jnp.where MLT switch (for jacfwd)
  - _build_residual: full residual with Dirichlet BCs
  - _build_residual_fixed_bc: residual with linearized atmosphere BCs
  - _build_residual_atm: residual with full atmosphere bridge

KEEP WHOLE decision (the module redesign spec §2): _cell_residual
and _cell_residual_raw implement the 4 coupled structure equations at one
midpoint — splitting by equation would produce 4 trivial functions that share
EOS/opacity/nuclear lookups.

References:
  - Kippenhahn, Weigert & Weiss (2012), §4.1, §10.3
  - Paxton et al. (2011), ApJS 192, 3, eq. 12
  - Henyey, Forbes & Gould (1964), ApJ 139, 306
"""

import jax
import jax.numpy as jnp
from jax import lax

from stellar_jax.config.constants import G, Lsun, a_rad, c_light
from stellar_jax.config.physics_floors import PGAS_FRAC_FLOOR
from stellar_jax.microphysics.eos import eos_lookup
from stellar_jax.microphysics.opacity import kappa
from stellar_jax.microphysics.nuclear import epsilon_nuclear
from stellar_jax.microphysics.neutrino import epsilon_neutrino
from stellar_jax.transport import mlt_nabla, mlt_nabla_raw
from stellar_jax.structure import interp_X_at_mass
from stellar_jax.solver.eps_grav import _eps_grav_form_c
import stellar_jax.solver.eps_grav as _eps_grav_mod


# ═══════════════════════════════════════════════════════════════
# DRY-1: Per-cell input preparation (shared by residual + Jacobian assembly)
# ═══════════════════════════════════════════════════════════════

from typing import NamedTuple, Optional


def _compute_gradL_composition_term(y, X_profile, Z, q_mesh, M_star, comp_mfracs):
    """Compute the Ledoux composition-gradient term per cell.

    Uses the textbook form (KWW §6.1): gradL_comp = max(d ln μ / d ln P, 0).
    CONSTRAINT: simpler than MESA's full MHM brunt_B (brunt.f90:get_brunt_B,
    which calls EOS twice per cell with different compositions). The MHM form
    would double EOS evaluations per cell, inflating the JIT graph. The simple
    form captures the stabilizing mu-barrier at the H-shell correctly.

    MESA ref: star/private/turb_support.f90:273:
      gradL = grada + gradL_composition_term
    where gradL_composition_term = smoothed_brunt_B (hydro_vars.f90:1091).

    Returns:
        (N_c,) array of composition gradient terms (≥0, per cell midpoint).
        On the MS with uniform composition, all values are 0.0.
    """
    N_s = y.shape[0]
    N_c = N_s - 1

    # Compute mean molecular weight at cell interfaces from composition.
    # mu = 1 / (2X + 3Y/4 + Z/2) is the standard ionized-gas approximation.
    def mu_at_q(q):
        X_val = interp_X_at_mass(X_profile, q, comp_mfracs)
        Y_val = 1.0 - X_val - Z
        return 1.0 / (2.0 * X_val + 0.75 * Y_val + 0.5 * Z + 1e-30)

    # mu at each solved interface
    q_interfaces = q_mesh[:N_s]
    mu_interfaces = jax.vmap(mu_at_q)(q_interfaces)

    # d ln mu / d ln P per cell = (ln mu_{k+1} - ln mu_k) / (ln P_{k+1} - ln P_k)
    ln_mu = jnp.log(jnp.maximum(mu_interfaces, 1e-30))
    ln_P = y[:, 1]  # ln_P at interfaces

    d_ln_mu = ln_mu[1:N_s] - ln_mu[:N_c]   # (N_c,)
    d_ln_P = ln_P[1:N_s] - ln_P[:N_c]      # (N_c,) — negative (P decreases outward)

    # d ln mu / d ln P: positive in regions where mu increases inward (stabilizing).
    # P decreases outward, mu decreases outward past the H-shell → both d_ln_mu and d_ln_P
    # are negative → ratio is positive → stabilizing.
    grad_mu = d_ln_mu / jnp.where(jnp.abs(d_ln_P) > 1e-30, d_ln_P, -1e-30)

    # Ledoux term: only the stabilizing (positive) part matters.
    # MESA turb_support.f90:421: if (gradL_composition_term < 0) then → thermohaline
    # We set negative values to zero (thermohaline is out of scope).
    gradL_comp = jnp.maximum(grad_mu, 0.0)
    return gradL_comp


class CellInputs(NamedTuple):
    """Pre-computed per-cell setup for residual/Jacobian assembly.

    All arrays are (N_c,) except where noted, where N_c = N_s - 1
    is the number of cells (inter-interface intervals).
    """
    N_s: int              # number of solved interfaces
    N_c: int              # number of cells = N_s - 1
    dq: jnp.ndarray       # (N_c,) fractional mass per cell
    dm: jnp.ndarray       # (N_c,) mass per cell [g]
    q_mid: jnp.ndarray    # (N_c,) fractional mass at cell midpoints
    m_mid: jnp.ndarray    # (N_c,) mass at midpoints [g]
    X_mid: jnp.ndarray    # (N_c,) hydrogen mass fraction at midpoints
    ln_T_prev_mid: jnp.ndarray  # (N_c,) ln(T) from previous step at midpoints
    ln_P_prev_mid: jnp.ndarray  # (N_c,) ln(P) from previous step at midpoints
    inv_dt: jnp.ndarray   # scalar: 1/dt for eps_grav coupling
    N14_mid: Optional[jnp.ndarray]  # (N_c,) or None: N14 at midpoints
    X_prev_mid: Optional[jnp.ndarray]  # (N_c,) or None: previous step X at midpoints
    X3_mid: Optional[jnp.ndarray] = None  # (N_c,) or None: ³He at midpoints
    ln_rho_guess_mid: Optional[jnp.ndarray] = None  # (N_c,) or None: warm-start density guess


def _prepare_cell_inputs(y, q_mesh, M_star, X_profile, Z,
                         ln_T_prev=None, ln_P_prev=None, inv_dt=None,
                         N14_profile=None, comp_mfracs=None,
                         X_prev_profile=None, X3_profile=None,
                         ln_rho_guess=None):
    """Compute the per-cell setup shared by all residual/Jacobian assembly functions.

    This is the DRY-1 extraction prescribed by the module redesign spec §2:
    the identical ~15-line interpolation/setup block was duplicated across
    _build_residual, _build_residual_fixed_bc, _build_residual_atm,
    _jacobian_blocks, _jacobian_blocks_fixed_bc, and _jacobian_blocks_atm.

    Parameters
    ----------
    y : (N_s, 4) — state at N_s interfaces [ln_r, ln_P, ln_T, ell]
    q_mesh : (N_s+1,) — fractional mass mesh
    M_star : scalar — stellar mass [g]
    X_profile : (N_COMP,) — hydrogen composition on the composition grid
    Z : scalar — metallicity
    ln_T_prev : (N_s,) or None — ln(T) from previous timestep
    ln_P_prev : (N_s,) or None — ln(P) from previous timestep
    inv_dt : scalar or None — 1/dt (0 if no eps_grav)
    N14_profile : (N_COMP,) or None — nitrogen profile for CNO feedback
    comp_mfracs : (N_COMP,) or None — composition grid positions
    X_prev_profile : (N_COMP,) or None — previous step hydrogen profile
    ln_rho_guess : (N_s,) or None — warm-start density guess [ln(rho)] for HELM.
        MESA ref: hydro_vars.f90:723 — lnd_start(k) = lnd(k) from previous step.

    Returns
    -------
    CellInputs NamedTuple with all per-cell quantities.
    """
    N_s = y.shape[0]
    N_c = N_s - 1
    dq = jnp.diff(q_mesh[:N_s])
    dm = M_star * dq
    q_mid = 0.5 * (q_mesh[:N_c] + q_mesh[1:N_s])
    m_mid = M_star * q_mid
    X_mid = jax.vmap(lambda q: interp_X_at_mass(X_profile, q, comp_mfracs))(q_mid)

    if N14_profile is not None:
        N14_mid = jax.vmap(lambda q: interp_X_at_mass(N14_profile, q, comp_mfracs))(q_mid)
    else:
        N14_mid = None

    _inv_dt = inv_dt if inv_dt is not None else jnp.float64(0.0)
    if ln_T_prev is not None:
        ln_T_prev_mid = 0.5 * (ln_T_prev[:N_c] + ln_T_prev[1:N_s])
        ln_P_prev_mid = 0.5 * (ln_P_prev[:N_c] + ln_P_prev[1:N_s])
    else:
        ln_T_prev_mid = 0.5 * (y[:N_c, 2] + y[1:N_s, 2])
        ln_P_prev_mid = 0.5 * (y[:N_c, 1] + y[1:N_s, 1])
        _inv_dt = jnp.float64(0.0)

    if X_prev_profile is not None:
        X_prev_mid = jax.vmap(lambda q: interp_X_at_mass(X_prev_profile, q, comp_mfracs))(q_mid)
    else:
        X_prev_mid = None

    if X3_profile is not None:
        X3_mid = jax.vmap(lambda q: interp_X_at_mass(X3_profile, q, comp_mfracs))(q_mid)
    else:
        X3_mid = None

    # Warm-start density guess: midpoint average of per-zone guesses.
    # MESA ref: hydro_vars.f90:723 — lnd_start(k) = lnd(k) from previous step.
    if ln_rho_guess is not None:
        ln_rho_guess_mid = 0.5 * (ln_rho_guess[:N_c] + ln_rho_guess[1:N_s])
    else:
        ln_rho_guess_mid = None

    return CellInputs(
        N_s=N_s, N_c=N_c, dq=dq, dm=dm, q_mid=q_mid, m_mid=m_mid,
        X_mid=X_mid, ln_T_prev_mid=ln_T_prev_mid, ln_P_prev_mid=ln_P_prev_mid,
        inv_dt=_inv_dt, N14_mid=N14_mid, X_prev_mid=X_prev_mid, X3_mid=X3_mid,
        ln_rho_guess_mid=ln_rho_guess_mid,
    )


def _center_eps_grav(T_c, P_c, cp_c, nad_c, ln_T_prev, ln_P_prev, inv_dt):
    """Compute eps_grav at the center boundary using Form C.

    MESA includes eps_grav at ALL cells uniformly (hydro_energy.f90:108,
    eps_grav.f90:35 — eval_eps_grav_and_partials called for every k including
    k==nz). Our center BC must do the same.

    Uses point values at the center (y[0]), not midpoint averages — the center
    cell has no inner neighbor to average with.

    Parameters
    ----------
    T_c : scalar — temperature at center (K)
    P_c : scalar — total pressure at center (dyn/cm²)
    cp_c : scalar — specific heat at center (erg/g/K)
    nad_c : scalar — adiabatic gradient at center
    ln_T_prev : (N_s,) or None — ln(T) from previous timestep
    ln_P_prev : (N_s,) or None — ln(P) from previous timestep
    inv_dt : scalar — 1/dt (0 → no eps_grav contribution)

    Returns
    -------
    eps_grav_c : scalar — gravothermal energy rate at center (erg/g/s)
    """
    if ln_T_prev is None or inv_dt is None:
        return jnp.float64(0.0)
    _inv_dt = inv_dt
    _ln_T_prev_c = ln_T_prev[0]
    _ln_P_prev_c = ln_P_prev[0]

    # Guard: time-centering at the center cell requires X_c to be threaded
    # through for a consistent start-of-step EOS call (cp_start, nad_start).
    # Without it, cp_start/nad_start stay None → the center cell silently
    # uses θ=1 while interior cells use θ=0.5.  Block this until X_c
    # threading is implemented as part of eps_grav time-centering validation.
    if _eps_grav_mod.EPS_GRAV_TIME_CENTERED:
        raise NotImplementedError(
            "EPS_GRAV_TIME_CENTERED at the center cell needs X_c threading "
            "for a consistent start-of-step EOS call — see the eps_grav "
            "time-centering validation issue."
        )

    return _eps_grav_form_c(T_c, P_c, cp_c, nad_c, _ln_T_prev_c, _ln_P_prev_c,
                            _inv_dt, cp_start=None, nad_start=None)


# ═══════════════════════════════════════════════════════════════
# Cell residual functions (KEEP WHOLE — 4 coupled structure equations)
# ═══════════════════════════════════════════════════════════════

def _cell_residual(y_k, y_kp1, dm, m_mid, M_star, X_mid, Z, alpha_mlt,
                   ln_T_prev_mid=None, ln_P_prev_mid=None, inv_dt=None, t_age=None,
                   X_N14_mid=None, opacity_factor=None, eps_nuc_factor=None,
                   X_prev_mid=None, gradL_composition_term=0.0, X3_mid=None,
                   ln_rho_guess_mid=None, on_adaptive_path=False,
                   helm_eos=False, bicubic_opacity=True):
    """4 structure equations for one cell at the midpoint.

    When ln_T_prev_mid, ln_P_prev_mid, and inv_dt are provided (and inv_dt > 0),
    ε_grav is coupled into the luminosity equation F3:
        dL/dm = ε_nuc + ε_grav
    where ε_grav uses Form C (KWW 2012, eq. 4.18; Paxton+2011 eq. 12):
        ε_grav = −cp (T − T_prev)/dt + (cp T ∇_ad / P)(P − P_prev)/dt

    The history terms (T_prev, P_prev) come from the previous timestep and are
    treated as PARAMETERS (not variables). The IFT backward pass propagates
    gradients through them automatically.

    Part A (#409): Real EOS cp and Q=chi_T/chi_rho are passed to mlt_nabla,
    matching MESA turb/private/mlt.f90:76 (Q=chiT/chiRho), :114 (A_1=4*Cp*sqrt(ff1*P*Q*rho)).
    Part B (#409): gradL_composition_term (Ledoux, MESA turb_support.f90:273).

    Parameters
    ----------
    on_adaptive_path : bool
        Explicit flag selecting the residual/Jacobian code path.
        True on the non-differentiable adaptive_forward RGB path: uses real
        EOS cp/Q (stop_gradient'd) and Ledoux gradL_composition_term.
        False (default) on the differentiable lax.scan path: uses ideal-gas
        cp/Q defaults, keeping the XLA trace and forward fixed point identical
        to the Jacobian path (_cell_residual_raw). This ensures IFT adjoint
        consistency (R_raw(y*) ≈ 0).

    References:
      - Kippenhahn, Weigert & Weiss (2012), §4.1, eq. 4.18
      - Paxton et al. (2011), ApJS 192, 3, eq. 12
      - MESA turb/private/mlt.f90:76,114 (Q, cp in the A factor)
    """
    ln_r_k, ln_P_k, ln_T_k, ell_k = y_k
    ln_r_kp1, ln_P_kp1, ln_T_kp1, ell_kp1 = y_kp1

    ln_r_mid = 0.5 * (ln_r_k + ln_r_kp1)
    ln_P_mid = 0.5 * (ln_P_k + ln_P_kp1)
    ln_T_mid = 0.5 * (ln_T_k + ln_T_kp1)
    ell_mid = 0.5 * (ell_k + ell_kp1)

    r_mid = jnp.exp(ln_r_mid)
    P_mid = jnp.exp(ln_P_mid)
    T_mid = jnp.exp(ln_T_mid)
    L_mid = ell_mid * Lsun

    logT_mid = jnp.log10(T_mid)
    P_rad_mid = a_rad * T_mid**4 / 3.0
    P_gas_mid = jnp.maximum(P_mid - P_rad_mid, PGAS_FRAC_FLOOR * P_mid)
    logPgas_mid = jnp.log10(P_gas_mid)

    rho, mu, nad, S, cp, chi_rho, chi_T = eos_lookup(logT_mid, logPgas_mid, X_mid, Z,
                                                      ln_rho_guess=ln_rho_guess_mid,
                                                      helm_eos=helm_eos)
    log_kap = kappa(logT_mid, jnp.log10(rho), X_mid, Z, opacity_factor=opacity_factor,
                    bicubic_opacity=bicubic_opacity)
    kap = 10.0**log_kap
    _t_age = t_age if t_age is not None else jnp.float64(0.0)
    # Pass X3 as its own X3_eq_frozen: since X3 from the carry IS X3_eq (relaxed
    # at the previous step with τ₃ ≪ dt), phi = X3²/X3² = 1.0 at equilibrium.
    # This prevents the spurious phi↔T coupling inside Newton (MESA operator-split).
    eps = epsilon_nuclear(rho, T_mid, X_mid, Z, _t_age, X_N14=X_N14_mid, X3=X3_mid,
                          X3_eq_frozen=X3_mid)
    # eps_nuc_factor: multiplicative knob on nuclear burning rate.
    # Default 1.0 = no change. Applied AFTER epsilon_nuclear, not inside it.
    if eps_nuc_factor is not None:
        eps = eps * eps_nuc_factor
    eps_nu = epsilon_neutrino(rho, T_mid, X_mid, Z)
    g_local = G * m_mid / (r_mid**2 + 1e-30)

    nabla_rad = 3.0 * kap * L_mid * P_mid / (
        16.0 * jnp.pi * a_rad * c_light * G * m_mid * T_mid**4 + 1e-30)
    # Part A: Pass real EOS cp and Q=chi_T/chi_rho to MLT.
    # MESA turb/private/mlt.f90:76: Q = chiT/chiRho
    # Guard chi_rho > 0 to avoid division by zero in the deep interior.
    Q_mlt = chi_T / jnp.maximum(chi_rho, 1e-30)
    # CONSTRAINT (GP-9 IFT_CONSISTENCY): Real cp/Q are only used on the
    # NON-DIFFERENTIABLE adaptive_forward path (on_adaptive_path=True). On the
    # differentiable lax.scan path (on_adaptive_path=False), we pass cp=None,
    # Q=None so mlt_nabla uses ideal-gas defaults — keeping the forward
    # fixed point y* IDENTICAL to main. This ensures R_raw(y*) ≈ 0 (the
    # Jacobian path _cell_residual_raw also uses ideal-gas), so the IFT
    # adjoint has no J/VJP mismatch. jacfwd through 4D EOS → OOM, so
    # the Jacobian CANNOT use real cp/Q in any case.
    # When the adaptive_forward path calls with on_adaptive_path=True,
    # real cp/Q are stop_gradient'd to suppress derivative flow while
    # giving the correct forward nabla value (RGB physics improvement).
    if on_adaptive_path:
        # CONSTRAINT (reviewer ask): cp and Q are treated as locally
        # constant w.r.t. the Newton iterate (and w.r.t. M) — i.e. we freeze the
        # EOS branch in the Jacobian. This is a stronger approximation than
        # GRADSOLVE strictly mandates, but it is FORCED: jacfwd through the 4-D
        # EOS to get live ∂cp/∂y, ∂Q/∂y OOMs (see the note above). It is also
        # SELF-CONSISTENT: the backward Jacobian/VJP freeze the SAME cp/Q/gradL
        # (jacobian.py/adjoint.py,), so the adjoint is the exact transpose
        # of the map the forward actually solves. Upgrade to a live path only
        # with FD evidence — tracked in the follow-up issue.
        cp_sg = lax.stop_gradient(cp)  # GP-9: IFT_CONSISTENCY
        Q_sg = lax.stop_gradient(Q_mlt)  # GP-9: IFT_CONSISTENCY
        gradL_sg = lax.stop_gradient(gradL_composition_term)  # GP-9: IFT_CONSISTENCY
    else:
        # lax.scan differentiable path: use ideal-gas defaults (same as main).
        # This preserves XLA trace identity AND forward fixed-point identity.
        cp_sg = None   # → mlt_nabla uses ideal-gas cp = k_B/(mu*m_H*nad)
        Q_sg = None    # → mlt_nabla uses Q = 1.0 (ideal gas)
        gradL_sg = 0.0  # Python literal → isinstance guard fires in mlt_nabla
    nabla = mlt_nabla(nabla_rad, nad, T_mid, P_mid, rho, kap, g_local, mu, alpha_mlt,
                      cp=cp_sg, Q=Q_sg, gradL_composition_term=gradL_sg)

    _inv_dt = inv_dt if inv_dt is not None else jnp.float64(0.0)
    _ln_T_prev = ln_T_prev_mid if ln_T_prev_mid is not None else ln_T_mid
    _ln_P_prev = ln_P_prev_mid if ln_P_prev_mid is not None else ln_P_mid

    # Time-centering (θ=0.5, MESA eps_grav.f90:130-175): compute start-of-step
    # EOS quantities from previous step's (T, P) — constants during Newton.
    # The EOS call is on the PREVIOUS step's state, so it does not depend on the
    # current Newton iterate — it gives identical values every iteration.
    # Controlled by EPS_GRAV_TIME_CENTERED flag (mutation gate).
    cp_start = None
    nad_start = None
    if _eps_grav_mod.EPS_GRAV_TIME_CENTERED and ln_T_prev_mid is not None and inv_dt is not None:
        T_prev_val = jnp.exp(_ln_T_prev)
        P_prev_val = jnp.exp(_ln_P_prev)
        logT_prev = jnp.log10(T_prev_val)
        P_rad_prev = a_rad * T_prev_val**4 / 3.0
        P_gas_prev = jnp.maximum(P_prev_val - P_rad_prev, PGAS_FRAC_FLOOR * P_prev_val)
        logPgas_prev = jnp.log10(P_gas_prev)
        _, _, nad_start, _, cp_start, _, _ = eos_lookup(logT_prev, logPgas_prev, X_mid, Z,
                                                              helm_eos=helm_eos)

    eps_grav = _eps_grav_form_c(T_mid, P_mid, cp, nad, _ln_T_prev, _ln_P_prev, _inv_dt,
                                cp_start=cp_start, nad_start=nad_start)

    # Composition term (MESA eps_grav.f90:187-189, eval_eps_grav_composition).
    # Adds the energy change from composition evolution: eps_grav_comp = -de/dt.
    # Only active when X_prev_mid is provided (i.e. composition carry is wired).
    # Reference: MESA controls.defaults:8095 (include_composition_in_eps_grav = .true.)
    if X_prev_mid is not None:
        from stellar_jax.solver.eps_grav import _eps_grav_composition
        eps_grav = eps_grav + _eps_grav_composition(T_mid, X_mid, X_prev_mid, Z, _inv_dt)

    F1 = (ln_r_kp1 - ln_r_k) - dm / (4.0 * jnp.pi * r_mid**3 * rho)
    F2 = (ln_P_kp1 - ln_P_k) + G * m_mid * dm / (4.0 * jnp.pi * r_mid**4 * P_mid)
    F3 = (ell_kp1 - ell_k) - (dm / Lsun) * (eps - eps_nu + eps_grav)
    F4 = (ln_T_kp1 - ln_T_k) - nabla * (ln_P_kp1 - ln_P_k)

    return jnp.array([F1, F2, F3, F4])


def _cell_residual_raw(y_k, y_kp1, dm, m_mid, M_star, X_mid, Z, alpha_mlt,
                       ln_T_prev_mid=None, ln_P_prev_mid=None, inv_dt=None, t_age=None,
                       X_N14_mid=None, opacity_factor=None, eps_nuc_factor=None,
                       X_prev_mid=None, X3_mid=None, ln_rho_guess_mid=None,
                       gradL_composition_term=0.0, on_adaptive_path=False,
                       helm_eos=False, bicubic_opacity=True):
    """Same as _cell_residual but with plain jnp.where MLT switch (no custom_jvp).

    Used by jax.jacfwd in _jacobian_blocks_fixed_bc where the exact hard-switch
    derivative is needed for Newton convergence of radiative-envelope stars.
    The sigmoid custom_jvp in _mlt_switch gives spurious nabla sensitivity in
    radiative zones, corrupting the Jacobian for fully-radiative envelopes.
    """
    ln_r_k, ln_P_k, ln_T_k, ell_k = y_k
    ln_r_kp1, ln_P_kp1, ln_T_kp1, ell_kp1 = y_kp1

    ln_r_mid = 0.5 * (ln_r_k + ln_r_kp1)
    ln_P_mid = 0.5 * (ln_P_k + ln_P_kp1)
    ln_T_mid = 0.5 * (ln_T_k + ln_T_kp1)
    ell_mid = 0.5 * (ell_k + ell_kp1)

    r_mid = jnp.exp(ln_r_mid)
    P_mid = jnp.exp(ln_P_mid)
    T_mid = jnp.exp(ln_T_mid)
    L_mid = ell_mid * Lsun

    logT_mid = jnp.log10(T_mid)
    P_rad_mid = a_rad * T_mid**4 / 3.0
    P_gas_mid = jnp.maximum(P_mid - P_rad_mid, PGAS_FRAC_FLOOR * P_mid)
    logPgas_mid = jnp.log10(P_gas_mid)

    rho, mu, nad, S, cp, chi_rho, chi_T = eos_lookup(logT_mid, logPgas_mid, X_mid, Z,
                                                      ln_rho_guess=ln_rho_guess_mid,
                                                      helm_eos=helm_eos)
    log_kap = kappa(logT_mid, jnp.log10(rho), X_mid, Z, opacity_factor=opacity_factor,
                    bicubic_opacity=bicubic_opacity)
    kap = 10.0**log_kap
    _t_age = t_age if t_age is not None else jnp.float64(0.0)
    eps = epsilon_nuclear(rho, T_mid, X_mid, Z, _t_age, X_N14=X_N14_mid, X3=X3_mid,
                          X3_eq_frozen=X3_mid)
    if eps_nuc_factor is not None:
        eps = eps * eps_nuc_factor
    eps_nu = epsilon_neutrino(rho, T_mid, X_mid, Z)
    g_local = G * m_mid / (r_mid**2 + 1e-30)

    nabla_rad = 3.0 * kap * L_mid * P_mid / (
        16.0 * jnp.pi * a_rad * c_light * G * m_mid * T_mid**4 + 1e-30)
    # GP-9 IFT_CONSISTENCY: on the replay/adaptive path the FORWARD solve
    # computes nabla with real EOS cp/Q + Ledoux gradL (stop_gradient'd). The IFT
    # adjoint requires this Jacobian (jax.jacfwd of this fn) to be the EXACT
    # transpose of that forward map, so mirror the same frozen cp/Q/gradL here.
    # Gated on on_adaptive_path so non-replay callers stay byte-identical.
    # CONSTRAINT: cp/Q frozen (locally constant w.r.t. y and M) — see the fuller
    # note in _cell_residual; forced by 4-D-EOS jacfwd OOM, self-consistent with
    # the forward. FD-evidence upgrade tracked in the follow-up issue.
    if on_adaptive_path:
        Q_mlt = chi_T / jnp.maximum(chi_rho, 1e-30)
        cp_sg = lax.stop_gradient(cp)  # GP-9: IFT_CONSISTENCY
        Q_sg = lax.stop_gradient(Q_mlt)  # GP-9: IFT_CONSISTENCY
        gradL_sg = lax.stop_gradient(gradL_composition_term)  # GP-9: IFT_CONSISTENCY
    else:
        cp_sg = None   # ideal-gas defaults (unchanged behavior)
        Q_sg = None
        gradL_sg = 0.0
    nabla = mlt_nabla_raw(nabla_rad, nad, T_mid, P_mid, rho, kap, g_local, mu, alpha_mlt,
                          cp=cp_sg, Q=Q_sg, gradL_composition_term=gradL_sg)

    _inv_dt = inv_dt if inv_dt is not None else jnp.float64(0.0)
    _ln_T_prev = ln_T_prev_mid if ln_T_prev_mid is not None else ln_T_mid
    _ln_P_prev = ln_P_prev_mid if ln_P_prev_mid is not None else ln_P_mid

    # Time-centering (θ=0.5): same as _cell_residual — start-of-step EOS.
    cp_start = None
    nad_start = None
    if _eps_grav_mod.EPS_GRAV_TIME_CENTERED and ln_T_prev_mid is not None and inv_dt is not None:
        T_prev_val = jnp.exp(_ln_T_prev)
        P_prev_val = jnp.exp(_ln_P_prev)
        logT_prev = jnp.log10(T_prev_val)
        P_rad_prev = a_rad * T_prev_val**4 / 3.0
        P_gas_prev = jnp.maximum(P_prev_val - P_rad_prev, PGAS_FRAC_FLOOR * P_prev_val)
        logPgas_prev = jnp.log10(P_gas_prev)
        _, _, nad_start, _, cp_start, _, _ = eos_lookup(logT_prev, logPgas_prev, X_mid, Z,
                                                              helm_eos=helm_eos)

    eps_grav = _eps_grav_form_c(T_mid, P_mid, cp, nad, _ln_T_prev, _ln_P_prev, _inv_dt,
                                cp_start=cp_start, nad_start=nad_start)

    # Composition term (same as _cell_residual — MESA eps_grav.f90:187-189).
    if X_prev_mid is not None:
        from stellar_jax.solver.eps_grav import _eps_grav_composition
        eps_grav = eps_grav + _eps_grav_composition(T_mid, X_mid, X_prev_mid, Z, _inv_dt)

    F1 = (ln_r_kp1 - ln_r_k) - dm / (4.0 * jnp.pi * r_mid**3 * rho)
    F2 = (ln_P_kp1 - ln_P_k) + G * m_mid * dm / (4.0 * jnp.pi * r_mid**4 * P_mid)
    F3 = (ell_kp1 - ell_k) - (dm / Lsun) * (eps - eps_nu + eps_grav)
    F4 = (ln_T_kp1 - ln_T_k) - nabla * (ln_P_kp1 - ln_P_k)

    return jnp.array([F1, F2, F3, F4])


def _build_residual(y, q_mesh, M_star, X_profile, Z, alpha_mlt,
                    ln_T_prev=None, ln_P_prev=None, inv_dt=None, N14_profile=None,
                    comp_mfracs=None, opacity_factor=None, eps_nuc_factor=None,
                    X3_profile=None, helm_eos=False, bicubic_opacity=True):
    """Build (N_s, 4) residual for block-Thomas with Dirichlet BCs.

    NOTE: This function is kept INLINE (not delegated to _assemble_residual)
    to preserve XLA trace identity on the ZAMS Newton path. The ZAMS solve is
    part of evolve_star, which jacrev(evolve_star) differentiates through.
    Delegating to _assemble_residual restructures the XLA HLO graph enough to
    cause OOM during backward-pass compilation of the MS inversion test
    (test_ms_inversion_real_evolve_star uses jacrev through the full ZAMS +
    lax.scan evolution). Bit-identical to main.

    y: (N_s, 4) state at N_s interfaces.
    Surface BC: Dirichlet (y[N_s-1] is pinned externally — residual = 0).
    Center BC: regularity of r and L (KWW 2012, §10.3).
    """
    ci = _prepare_cell_inputs(y, q_mesh, M_star, X_profile, Z,
                              ln_T_prev, ln_P_prev, inv_dt,
                              N14_profile, comp_mfracs,
                              X3_profile=X3_profile)
    N_s, N_c = ci.N_s, ci.N_c

    # Part B: Ledoux composition-gradient term.
    # NOT computed here — on the differentiable lax.scan path (MS), composition
    # is uniform so gradL_comp=0 everywhere. Computing it would add JAX traced
    # ops (interp_X_at_mass, log, division) to the XLA graph even though the
    # result is stop_gradient'd inside _cell_residual. These dead ops change XLA
    # instruction scheduling, shifting float accumulation order enough to push
    # the bounded IFT approximation error past tolerance over N=200 steps.
    # The Ledoux term only matters on the RGB (non-differentiable adaptive_forward
    # path), where it should be precomputed and passed via a future parameter.
    # MESA turb_support.f90:273: gradL = grada + gradL_composition_term

    def cell_k(k):
        _n14_k = ci.N14_mid[k] if ci.N14_mid is not None else None
        _xprev_k = ci.X_prev_mid[k] if ci.X_prev_mid is not None else None
        _x3_k = ci.X3_mid[k] if ci.X3_mid is not None else None
        return _cell_residual(y[k], y[k + 1], ci.dm[k], ci.m_mid[k], M_star, ci.X_mid[k], Z, alpha_mlt,
                              ci.ln_T_prev_mid[k], ci.ln_P_prev_mid[k], ci.inv_dt,
                              X_N14_mid=_n14_k,
                              opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                              X_prev_mid=_xprev_k, X3_mid=_x3_k,
                              helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
    F_cells = jax.vmap(cell_k)(jnp.arange(N_c))

    # Center BCs (KWW 2012, §10.3, Eqs. 10.8–10.10)
    ln_r_0, ln_P_0, ln_T_0, ell_0 = y[0]
    T_c = jnp.exp(ln_T_0)
    P_c = jnp.exp(ln_P_0)
    P_rad_c = a_rad * T_c**4 / 3.0
    P_gas_c = jnp.maximum(P_c - P_rad_c, PGAS_FRAC_FLOOR * P_c)
    rho_c, _mu_c, nad_c, _S_c, cp_c, _chi_rho_c, _chi_T_c = eos_lookup(
        jnp.log10(T_c), jnp.log10(P_gas_c), X_profile[0], Z, helm_eos=helm_eos)
    m1 = M_star * q_mesh[1]
    r0_exp = (3.0 * m1 / (4.0 * jnp.pi * rho_c))**(1.0 / 3.0)
    BC1 = ln_r_0 - jnp.log(jnp.maximum(r0_exp, 1.0))
    _n14_c = N14_profile[0] if N14_profile is not None else None
    _x3_c = X3_profile[0] if X3_profile is not None else None
    eps_c = epsilon_nuclear(rho_c, T_c, X_profile[0], Z, jnp.float64(0.0), X_N14=_n14_c, X3=_x3_c)
    if eps_nuc_factor is not None:
        eps_c = eps_c * eps_nuc_factor
    eps_nu_c = epsilon_neutrino(rho_c, T_c, X_profile[0], Z)
    # eps_grav at center: MESA hydro_energy.f90:108 includes eps_grav at ALL
    # cells (k==nz is the center). Form C via _eps_grav_form_c, same as interior.
    eps_grav_c = _center_eps_grav(T_c, P_c, cp_c, nad_c,
                                  ln_T_prev, ln_P_prev, ci.inv_dt)
    BC2 = ell_0 - (eps_c - eps_nu_c + eps_grav_c) * m1 / Lsun

    R = jnp.zeros((N_s, 4))
    R = R.at[0].set(jnp.array([BC1, BC2, F_cells[0, 0], F_cells[0, 1]]))

    def interior_row(k):
        return jnp.array([F_cells[k - 1, 2], F_cells[k - 1, 3], F_cells[k, 0], F_cells[k, 1]])
    R = R.at[1:N_c].set(jax.vmap(interior_row)(jnp.arange(1, N_c)))

    R = R.at[N_s - 1].set(jnp.array([F_cells[N_c - 1, 2], F_cells[N_c - 1, 3], 0.0, 0.0]))
    return R


def _build_residual_fixed_bc(y, q_mesh, M_star, X_profile, Z, alpha_mlt,
                              ln_P_atm, ln_T_atm, atm_jac, y_surf_ref,
                              ln_T_prev=None, ln_P_prev=None, inv_dt=None,
                              bc1_floor=1.0, subtract_neutrinos=True,
                              N14_profile=None, comp_mfracs=None,
                              opacity_factor=None, eps_nuc_factor=None,
                              X_prev_profile=None, gradL_composition_term=None, X3_profile=None,
                              ln_rho_guess=None, helm_eos=False, bicubic_opacity=True):
    """Build residual with linearized atmosphere BCs (no bridge call).

    NOTE: This function is kept INLINE (not delegated to _assemble_residual)
    to preserve XLA trace identity on the Newton per-step path. The adaptive
    forward RGB tests (test_adaptive_forward_rgb_*) accumulate float error over
    thousands of steps and are sensitive to XLA graph restructuring (~0.02-0.03
    dex shift). Inlining ensures bit-identical HLO vs the pre-refactoring code.

    Surface BCs use the linearized atmosphere response:
      BC_P = ln_P_s - (ln_P_atm + atm_jac[0] @ (y_surf - y_surf_ref))
      BC_T = ln_T_s - (ln_T_atm + atm_jac[1] @ (y_surf - y_surf_ref))

    gradL_composition_term: Optional (N_c,) array of per-cell Ledoux composition
        gradient terms. When None (default, lax.scan MS path), no Ledoux term is
        used and the XLA trace stays bit-identical to main. When an array (the
        adaptive_forward RGB path), each cell receives its Ledoux stabilization.
        MESA turb_support.f90:273: gradL = grada + gradL_composition_term.
    """
    ci = _prepare_cell_inputs(y, q_mesh, M_star, X_profile, Z,
                              ln_T_prev, ln_P_prev, inv_dt,
                              N14_profile, comp_mfracs,
                              X_prev_profile=X_prev_profile,
                              X3_profile=X3_profile,
                              ln_rho_guess=ln_rho_guess)
    N_s, N_c = ci.N_s, ci.N_c

    # Part B: Ledoux composition-gradient term (MESA turb_support.f90:273).
    # When gradL_composition_term is None (the differentiable lax.scan path),
    # _cell_residual receives the default 0.0 and on_adaptive_path=False →
    # ideal-gas cp/Q defaults, preserving XLA trace identity.
    # When it's an array (the adaptive_forward RGB path, precomputed externally),
    # on_adaptive_path=True activates real EOS cp/Q + Ledoux stabilization.

    # Determine path-select once (Python bool, not traced) — True when the
    # caller provided a real composition-gradient array (adaptive_forward RGB).
    _on_adaptive_path = gradL_composition_term is not None

    def cell_k(k):
        _n14_k = ci.N14_mid[k] if ci.N14_mid is not None else None
        _xprev_k = ci.X_prev_mid[k] if ci.X_prev_mid is not None else None
        _x3_k = ci.X3_mid[k] if ci.X3_mid is not None else None
        _gradL_k = gradL_composition_term[k] if gradL_composition_term is not None else 0.0
        _rho_guess_k = ci.ln_rho_guess_mid[k] if ci.ln_rho_guess_mid is not None else None
        return _cell_residual(y[k], y[k + 1], ci.dm[k], ci.m_mid[k], M_star, ci.X_mid[k], Z, alpha_mlt,
                              ci.ln_T_prev_mid[k], ci.ln_P_prev_mid[k], ci.inv_dt,
                              X_N14_mid=_n14_k, opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                              X_prev_mid=_xprev_k, gradL_composition_term=_gradL_k, X3_mid=_x3_k,
                              ln_rho_guess_mid=_rho_guess_k,
                              on_adaptive_path=_on_adaptive_path,
                              helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
    F_cells = jax.vmap(cell_k)(jnp.arange(N_c))

    ln_r_0, ln_P_0, ln_T_0, ell_0 = y[0]
    T_c = jnp.exp(ln_T_0)
    P_c = jnp.exp(ln_P_0)
    P_rad_c = a_rad * T_c**4 / 3.0
    P_gas_c = jnp.maximum(P_c - P_rad_c, PGAS_FRAC_FLOOR * P_c)
    rho_c, _mu_c, nad_c, _S_c, cp_c, _chi_rho_c, _chi_T_c = eos_lookup(
        jnp.log10(T_c), jnp.log10(P_gas_c), X_profile[0], Z, helm_eos=helm_eos)
    m1 = M_star * q_mesh[1]
    r0_exp = (3.0 * m1 / (4.0 * jnp.pi * rho_c))**(1.0 / 3.0)
    BC1 = ln_r_0 - jnp.log(jnp.maximum(r0_exp, bc1_floor))
    _n14_c = N14_profile[0] if N14_profile is not None else None
    _x3_c = X3_profile[0] if X3_profile is not None else None
    eps_c = epsilon_nuclear(rho_c, T_c, X_profile[0], Z, jnp.float64(0.0), X_N14=_n14_c, X3=_x3_c)
    if eps_nuc_factor is not None:
        eps_c = eps_c * eps_nuc_factor
    eps_nu_c = epsilon_neutrino(rho_c, T_c, X_profile[0], Z)
    # eps_grav at center (MESA hydro_energy.f90:108, eps_grav.f90:35)
    eps_grav_c = _center_eps_grav(T_c, P_c, cp_c, nad_c,
                                  ln_T_prev, ln_P_prev, ci.inv_dt)
    BC2 = ell_0 - jnp.where(subtract_neutrinos,
                             (eps_c - eps_nu_c + eps_grav_c),
                             (eps_c + eps_grav_c)) * m1 / Lsun

    y_surf = y[N_s - 1]
    dy_surf = y_surf - y_surf_ref
    ln_P_target = ln_P_atm + jnp.dot(atm_jac[0], dy_surf)
    ln_T_target = ln_T_atm + jnp.dot(atm_jac[1], dy_surf)
    BC_P = y_surf[1] - ln_P_target
    BC_T = y_surf[2] - ln_T_target

    R = jnp.zeros((N_s, 4))
    R = R.at[0].set(jnp.array([BC1, BC2, F_cells[0, 0], F_cells[0, 1]]))
    def interior_row(k):
        return jnp.array([F_cells[k - 1, 2], F_cells[k - 1, 3], F_cells[k, 0], F_cells[k, 1]])
    R = R.at[1:N_c].set(jax.vmap(interior_row)(jnp.arange(1, N_c)))
    R = R.at[N_s - 1].set(jnp.array([F_cells[N_c - 1, 2], F_cells[N_c - 1, 3], BC_P, BC_T]))
    return R


def _build_residual_atm(y, q_mesh, M_star, X_profile, Z, alpha_mlt,
                         ln_T_prev=None, ln_P_prev=None, inv_dt=None, T_eff=None, R_phot=None,
                         bc1_floor=1.0, subtract_neutrinos=True, comp_mfracs=None,
                         opacity_factor=None, eps_nuc_factor=None,
                         N14_profile=None, X3_profile=None,
                         helm_eos=False, bicubic_opacity=True):
    """Build residual with atmosphere BC at the surface (replaces Dirichlet).

    NOTE: This function is kept INLINE (not delegated to a shared helper) to
    preserve XLA trace identity. It is called from henyey_init_from_state_atm
    and henyey_solve_from_state_atm, both of which are part of the evolve_star
    forward trace compiled by jacrev(evolve_star). Even though the results are
    stop_gradient'd, the XLA compiler still traces the full forward graph —
    delegating to a callback-parameterized helper produces a different/larger
    HLO graph that causes OOM during backward-pass compilation.

    Same solve domain as _build_residual, but surface block uses _surface_bc_atm
    instead of zeros (Dirichlet pinning).
    """
    from stellar_jax.solver.surface_bc import _surface_bc_atm

    ci = _prepare_cell_inputs(y, q_mesh, M_star, X_profile, Z,
                              ln_T_prev, ln_P_prev, inv_dt,
                              N14_profile, comp_mfracs,
                              X3_profile=X3_profile)
    N_s, N_c = ci.N_s, ci.N_c

    # Part B: Ledoux composition-gradient term — NOT computed here.
    # See _build_residual comment for rationale (XLA graph identity on the MS).

    def cell_k(k):
        _n14_k = ci.N14_mid[k] if ci.N14_mid is not None else None
        _xprev_k = ci.X_prev_mid[k] if ci.X_prev_mid is not None else None
        _x3_k = ci.X3_mid[k] if ci.X3_mid is not None else None
        return _cell_residual(y[k], y[k + 1], ci.dm[k], ci.m_mid[k], M_star, ci.X_mid[k], Z, alpha_mlt,
                              ci.ln_T_prev_mid[k], ci.ln_P_prev_mid[k], ci.inv_dt,
                              X_N14_mid=_n14_k, opacity_factor=opacity_factor,
                              X_prev_mid=_xprev_k, X3_mid=_x3_k,
                              helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
    F_cells = jax.vmap(cell_k)(jnp.arange(N_c))

    # Center BCs
    ln_r_0, ln_P_0, ln_T_0, ell_0 = y[0]
    T_c = jnp.exp(ln_T_0)
    P_c = jnp.exp(ln_P_0)
    P_rad_c = a_rad * T_c**4 / 3.0
    P_gas_c = jnp.maximum(P_c - P_rad_c, PGAS_FRAC_FLOOR * P_c)
    rho_c, _mu_c, nad_c, _S_c, cp_c, _chi_rho_c, _chi_T_c = eos_lookup(
        jnp.log10(T_c), jnp.log10(P_gas_c), X_profile[0], Z, helm_eos=helm_eos)
    m1 = M_star * q_mesh[1]
    r0_exp = (3.0 * m1 / (4.0 * jnp.pi * rho_c))**(1.0 / 3.0)
    BC1 = ln_r_0 - jnp.log(jnp.maximum(r0_exp, bc1_floor))
    _n14_c = N14_profile[0] if N14_profile is not None else None
    _x3_c = X3_profile[0] if X3_profile is not None else None
    eps_c = epsilon_nuclear(rho_c, T_c, X_profile[0], Z, jnp.float64(0.0), X_N14=_n14_c, X3=_x3_c)
    eps_nu_c = epsilon_neutrino(rho_c, T_c, X_profile[0], Z)
    # eps_grav at center (MESA hydro_energy.f90:108, eps_grav.f90:35)
    eps_grav_c = _center_eps_grav(T_c, P_c, cp_c, nad_c,
                                  ln_T_prev, ln_P_prev, ci.inv_dt)
    BC2 = ell_0 - jnp.where(subtract_neutrinos,
                             (eps_c - eps_nu_c + eps_grav_c),
                             (eps_c + eps_grav_c)) * m1 / Lsun

    # Surface atmosphere BCs
    X_surf = X_profile[-1]
    BC_P, BC_T = _surface_bc_atm(y[N_s - 1], M_star, X_surf, Z, alpha_mlt,
                                  T_eff_param=T_eff, R_phot=R_phot)

    R = jnp.zeros((N_s, 4))
    R = R.at[0].set(jnp.array([BC1, BC2, F_cells[0, 0], F_cells[0, 1]]))
    def interior_row(k):
        return jnp.array([F_cells[k - 1, 2], F_cells[k - 1, 3], F_cells[k, 0], F_cells[k, 1]])
    R = R.at[1:N_c].set(jax.vmap(interior_row)(jnp.arange(1, N_c)))
    R = R.at[N_s - 1].set(jnp.array([F_cells[N_c - 1, 2], F_cells[N_c - 1, 3], BC_P, BC_T]))
    return R
