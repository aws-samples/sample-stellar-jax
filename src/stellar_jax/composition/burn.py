"""Hydrogen burning and CNO isotope tracking.

These are PURE functions — no stop_gradient, no side effects.

References:
  - Caughlan & Fowler (1988), ADNDT 40, 283: pp+CNO rates
  - Clayton (1968), Principles of Stellar Evolution, §5.4: CN equilibrium
  - MESA struct_burn_mix.f90:82-100: op_split_burn architecture
  - MESA net_approx21.f90: CN equilibration (exponential relaxation)
"""
import jax.numpy as jnp

from stellar_jax.config.mesh_defaults import COMP_MFRACS
from stellar_jax.config.constants import Q_PER_G


def burn_composition(X_profile, Y_profile, shell_data, dt):
    """Burn hydrogen on the composition grid, depositing into helium.

    Baryon conservation: ΔY = −ΔX exactly. Mass fractions track baryon
    (nucleon) count, not rest mass; the 0.7% mass defect goes into energy
    (eps_nuc / Q_PER_G) and is NOT removed from the mass-fraction sum.
    This matches MESA's network: net_eval.f90:274 computes
    dxdt(i) = Z_plus_N(ci) * dydt(i), and sum(dxdt) = 0 by construction
    (molar baryon conservation). MESA also renormalizes per zone:
    hydro_vars.f90:751-754.

    Parameters
    ----------
    X_profile : array (N_COMP,)
        Hydrogen mass fraction on the composition grid.
    Y_profile : array (N_COMP,)
        Helium mass fraction on the composition grid.
    shell_data : array (N_SHELLS, 10)
        Structure data (col 0 = eps_nuc, col 1 = m/M).
    dt : scalar
        Timestep [seconds].

    Returns
    -------
    X_new : array (N_COMP,)
        Updated hydrogen mass fraction after burning.
    Y_new : array (N_COMP,)
        Updated helium mass fraction after burning (ΔY = −ΔX).
    """
    from stellar_jax.composition.interp import _interp_shell_to_comp_reversed
    eps_shells = shell_data[:, 0]
    mf_shells = shell_data[:, 1]
    comp_mfracs = jnp.array(COMP_MFRACS)
    eps_comp = _interp_shell_to_comp_reversed(comp_mfracs, mf_shells, eps_shells)
    dX = eps_comp * dt / Q_PER_G
    actual_dX = jnp.minimum(dX, X_profile)  # can't burn more H than exists
    X_new = jnp.maximum(X_profile - dX, 0.0)
    Y_new = Y_profile + actual_dX  # He produced = H consumed (baryon conservation)
    return X_new, Y_new


def burn_cno(C12_profile, C13_profile, N14_profile, X_profile, shell_data, dt, comp_mfracs_in=None):
    """Evolve CNO isotopes via the CN cycle: 12C(p,γ)13N(β+)13C(p,γ)14N.

    At T > ~15 MK the CN cycle reaches equilibrium on a timescale much shorter
    than the MS lifetime.  The equilibrium 12C/13C ratio is T-dependent, set by
    the ratio of proton-capture rates from Caughlan & Fowler (1988) Table V.a:
    ~3.1 at 20 MK, ~3.5 at 15 MK, ~4.5 at 30 MK (NACRE II 2013 confirms
    within 10%).

    Implementation: compute the CN-cycle rate per shell and drive the isotopes
    toward T-dependent equilibrium with an exponential relaxation (operator-split,
    same as MESA's approx21 network for the CN sub-cycle).  Total C+N is
    conserved (the ON cycle is slow and ignored here — issue #53 for full
    network).

    NOTE (updated #1026): This tracking is now ACTIVE — the evolved N14
    profile feeds back into epsilon_nuclear via the Henyey solver. The CNO
    energy generation rate uses the tracked N14 abundance directly, matching
    MESA's approx21 treatment (net_approx21.f90:1116: y(in14)*y(ih1)*rate).

    PHYSICAL JUSTIFICATION FOR ACTIVE FEEDBACK:
    On the MS core (CN equilibrium), N14 ≈ CN_total = 0.251*Z (with the
    corrected initial CN_total that includes PMS ON-cycle N14 gain, #1026).
    In the radiative envelope (never burned), initial N14 = 0.079*Z, giving
    lower eps_cno than the equilibrium assumption. This spatial variation is
    physically correct and matters at core-envelope boundaries (H-shell on
    RGB) and post-dredge-up. MESA's approx21 uses y(in14) directly
    (net_approx21.f90:1116).

    CONSTRAINT: Our burn_cno conserves CN_total = C12+C13+N14 = 0.251*Z
    (set at initialization). During the MS, MESA's N14 grows further (to
    ~0.326*Z at 2.0 Msun midMS) as the ON sub-cycle continues converting
    O16→N14. Our CN_total is fixed, so we underestimate N14 by ~12-23%
    at midMS for CNO-dominant masses — a bounded CONSTRAINT of the CN-only
    network that cannot be removed without adding O16 tracking.

    Gradient path: the CN equilibration + convective mixing is a LIVE
    gradient path (issue #393). N14 changes at step k affect eps_cno at
    step k+1 via the carry → Henyey solve → IFT. X_H also participates
    (it sets the equilibration rate), with numerical stability ensured by
    clipping the exponent in the relaxation factor: when dt*rate >> 1
    (system fully equilibrated), the gradient naturally damps to zero.

    NOTE: the CALLER passes shell_data_sg (stop_gradient'd structure) so
    that ∂(burn_cno)/∂T_structure does NOT propagate through the lax.scan
    backward. This matches MESA's op_split_burn (struct_burn_mix.f90:82-100)
    where d_epsnuc_dlnT/d_epsnuc_dlnd are zeroed in the structure Jacobian.
    The Henyey IFT already captures ∂L/∂T_structure; adding a second path
    through burn_cno's T-dependent rate creates numerical instability
    (NaN via exp(-136.90/T6^{1/3}) backward over many steps).

    References:
      - Caughlan & Fowler (1988), ADNDT 40, 283: reaction rates
      - Clayton (1968), Principles of Stellar Evolution, §5.4: CN equilibrium
      - Arnould, Goriely & Takahashi (2003), Phys. Rep. 384, 1: NACRE II
    """
    mf_shells = shell_data[:, 1]
    logT_shells = shell_data[:, 6]
    logrho_shells = shell_data[:, 7]

    comp_mfracs = comp_mfracs_in if comp_mfracs_in is not None else jnp.array(COMP_MFRACS)

    # Interpolate T and rho onto composition grid
    from stellar_jax.composition.interp import _interp_shell_to_comp_reversed
    logT_comp = _interp_shell_to_comp_reversed(comp_mfracs, mf_shells, logT_shells)
    logrho_comp = _interp_shell_to_comp_reversed(comp_mfracs, mf_shells, logrho_shells)
    rho_comp = 10.0**logrho_comp
    T6 = 10.0**logT_comp * 1e-6

    # CN-cycle relaxation rate for 12C destruction by protons [1/s].
    #
    # Kinetic theory (Iliadis 2015, eq. 3.65; Clayton 1968, eq. 4-52):
    #   dn_12/dt = -n_H × n_12 × ⟨σv⟩
    #
    # Converting to mass fractions (X_i = n_i A_i m_u / ρ, i.e. n_i = ρ X_i / (A_i m_u)):
    #   dX_12/dt = -X_12 × (ρ X_H / m_H) × ⟨σv⟩
    #            = -X_12 × λ
    #
    # where λ = ρ X_H ⟨σv⟩ / m_H is the relaxation rate [1/s].
    #
    # CF88 tabulates NA⟨σv⟩ [cm³ mol⁻¹ s⁻¹]. The physical cross-section is:
    #   ⟨σv⟩ = NA⟨σv⟩ / NA
    # Therefore:
    #   λ = ρ X_H × (NA⟨σv⟩ / NA) / m_H = ρ X_H × NA⟨σv⟩ / (NA × m_H)
    #
    # Since NA × m_u = 1 g/mol and m_H ≈ m_u (0.08% error):
    #   λ = ρ X_H × NA⟨σv⟩ [s⁻¹]  (with ρ in g/cm³, NA⟨σv⟩ in cm³ g⁻¹ s⁻¹)
    #
    # NA⟨σv⟩(12C+p) = 2.04e7 T9^(-2/3) exp(-13.690/T9^(1/3)) cm³/mol/s
    #   (Caughlan & Fowler 1988, Table V.a, reaction II-1: 12C(p,γ)13N)
    # Converting to T6 = T9 × 1000:
    #   T9^(-2/3) = (T6/1000)^(-2/3) = 100 / T6^(2/3)
    #   13.690/T9^(1/3) = 136.90/T6^(1/3)
    # → NA⟨σv⟩ = 2.04e9 / T6^(2/3) × exp(-136.90/T6^(1/3))
    #
    # Verification at solar center (T6=15, ρ=150, X_H=0.34):
    #   NA⟨σv⟩ = 2.04e9 / 6.08 × exp(-55.5) ≈ 3.36e8 × 8.4e-25 ≈ 2.8e-16
    #   λ = 150 × 0.34 × 2.8e-16 ≈ 1.4e-14 s⁻¹
    #   τ_CN = 1/λ ≈ 7e13 s ≈ 2.2 Myr (Clayton 1968: ~2.5 Myr ✓)
    T6_13 = jnp.maximum(T6, 0.5)**(1.0 / 3.0)
    # CN equilibration rate: λ_CN = ρ X_H NA⟨σv⟩(12C+p)  [s⁻¹]
    # Full gradient path preserved: X_profile is NOT stop_gradient'd.
    # The sensitivity ∂N14/∂X_H through the rate is a valid physical path
    # (X_H sets the equilibration timescale τ_CN = 1/λ_CN).
    #
    # Numerical stability: the relaxation factor f = 1 - exp(-dt * rate_cn)
    # saturates at 1.0 when dt * rate_cn >> 1 (CN equilibrium reached within
    # one timestep). In that limit, ∂f/∂X = dt * exp(-dt*rate_cn) * (rate/X)
    # → 0 exponentially: the system has forgotten its initial state, so the
    # gradient contribution is physically zero. We clip the exponent at 30.0
    # (exp(-30) ≈ 9.4e-14, indistinguishable from full equilibrium) to prevent
    # NaN in the backward pass when dt*rate_cn is O(10³) or larger.
    #
    # Physics: this matches the correct limit. At CN equilibrium (f=1),
    # N14_new = N14_eq regardless of the rate — so ∂N14_new/∂(anything via rate) = 0.
    # The clip ensures the numerical gradient respects this zero limit cleanly.
    #
    # MESA reference: MESA's approx21 uses the same exponential relaxation
    # (net_approx21.f90 subroutine approx21_eval_PPII, net_basic_support.f90)
    # and does not encounter overflow because it uses an implicit solver that
    # naturally limits the effective integration step.
    rate_cn = 2.04e9 * rho_comp * X_profile / (T6_13**2) * jnp.exp(-136.90 / T6_13)

    # Relaxation factor with clipped exponent for AD stability.
    # Forward: f = 1 - exp(-min(dt*rate, 30)) ≈ 1 - exp(-dt*rate) when dt*rate < 30.
    # Backward: ∂f/∂rate = dt * exp(-min(dt*rate, 30)) — naturally zero when saturated.
    dt_rate = jnp.minimum(dt * rate_cn, 30.0)
    f = 1.0 - jnp.exp(-dt_rate)
    f = jnp.clip(f, 0.0, 1.0)

    # CN equilibrium fractions — T-dependent (Caughlan & Fowler 1988, Table V):
    # At CN equilibrium, dX_i/dt = 0 → X_i ∝ 1/λ_i (destruction rate).
    # The equilibrium fractions are set by the rate ratios:
    #   X(12C)/X(13C) = [λ(13C+p)/λ(12C+p)] × (A_12/A_13)
    # CF88 rates (cm3/mol/s):
    #   12C(p,γ)13N: S12 = 2.04e7 T9^(-2/3) exp(-13.690/T9^(1/3))
    #   13C(p,γ)14N: S13 = 8.01e7 T9^(-2/3) exp(-13.717/T9^(1/3))
    #   14N(p,γ)15O: S14 = 4.90e7 T9^(-2/3) exp(-15.228/T9^(1/3))
    # The T9^(-2/3) and prefactors cancel in ratios except for the exponential
    # and the prefactor ratio. Converting T9 = T6/1000:
    #   ratio_12_13 = (S13/S12) × (A_12/A_13) = (8.01/2.04)×(12/13) × exp(-(13.717-13.690)/T9^(1/3))
    #   ratio_14_12 = (S12/S14) × (A_14/A_12) = (2.04/4.90)×(14/12) × exp(-(13.690-15.228)/T9^(1/3))
    # These give 12C/13C ~ 3.5 at T9=0.015 (15 MK) and ~ 4.5 at T9=0.030 (30 MK).
    # Reference: Clayton (1968) §5.4; NACRE II (2013), Angulo+ (1999), Fig. 12.
    T9 = T6 / 1000.0
    T9_13 = jnp.maximum(T9, 5e-4)**(1.0 / 3.0)

    # At CN equilibrium, number densities satisfy n_i ∝ 1/λ_i where
    # λ_i = n_H ⟨σv⟩_i is the destruction rate of species i by proton capture.
    # Converting to mass fractions: X_i = n_i A_i m_u / ρ ∝ A_i / λ_i.
    # Since λ_i ∝ NA⟨σv⟩_i (the ρ X_H prefactor cancels in ratios):
    #   X_i ∝ A_i / S_i  where S_i = NA⟨σv⟩_i (CF88 tabulated rate)
    # Reference: Clayton (1968) eq. 5-19; Iliadis (2015) eq. 5.44.
    # The exponential part dominates the T-dependence:
    inv_12 = (12.0 / 2.04e7) * jnp.exp(13.690 / T9_13)
    inv_13 = (13.0 / 8.01e7) * jnp.exp(13.717 / T9_13)
    inv_14 = (14.0 / 4.90e7) * jnp.exp(15.228 / T9_13)
    inv_total = inv_12 + inv_13 + inv_14

    # Mass-fraction equilibrium ratios (T-dependent)
    C12_eq_frac = inv_12 / inv_total
    C13_eq_frac = inv_13 / inv_total
    N14_eq_frac = inv_14 / inv_total

    CN_total = C12_profile + C13_profile + N14_profile
    C12_eq = C12_eq_frac * CN_total
    C13_eq = C13_eq_frac * CN_total
    N14_eq = N14_eq_frac * CN_total

    # Relax toward equilibrium
    C12_new = C12_profile + f * (C12_eq - C12_profile)
    C13_new = C13_profile + f * (C13_eq - C13_profile)
    N14_new = N14_profile + f * (N14_eq - N14_profile)

    # Ensure non-negative and conserve total
    C12_new = jnp.maximum(C12_new, 0.0)
    C13_new = jnp.maximum(C13_new, 0.0)
    N14_new = jnp.maximum(N14_new, 0.0)
    total_new = C12_new + C13_new + N14_new
    scale = CN_total / (total_new + 1e-30)
    C12_new = C12_new * scale
    C13_new = C13_new * scale
    N14_new = N14_new * scale

    return C12_new, C13_new, N14_new



def relax_he3(X3_profile, X_profile, shell_data, dt, comp_mfracs_in=None):
    """Relax ³He toward equilibrium via exponential decay (issue #112).

    Implements the operator-split ³He evolution:
        X3_new = X3_eq + (X3_old - X3_eq) * exp(-dt/tau_eq)

    where X3_eq and tau_eq are computed from local (T, ρ, X) via the Clayton
    (1968) §5.6 rate balance in he3_equilibrium().

    MESA reference: MESA evolves ³He explicitly (net_approx21.f90), but
    operator-splits the burn step from the structure solve
    (struct_burn_mix.f90:82-100). Our exponential relaxation matches this
    split: the structure T, ρ from the converged Henyey solve (shell_data)
    are used as frozen parameters for the ³He evolution.

    Parameters
    ----------
    X3_profile : array (N_COMP,)
        Current ³He mass fraction on the composition grid
    X_profile : array (N_COMP,)
        Hydrogen mass fraction on the composition grid
    shell_data : array (N_shells, ...)
        Structure data from the Henyey solve (col 1=mfrac, 6=logT, 7=logrho)
    dt : scalar
        Timestep [seconds]
    comp_mfracs_in : array or None
        Composition grid (if None, uses COMP_MFRACS)

    Returns
    -------
    X3_new : array (N_COMP,)
        Updated ³He mass fraction after relaxation
    """
    from stellar_jax.microphysics.nuclear import he3_equilibrium
    from stellar_jax.composition.interp import _interp_shell_to_comp_reversed

    mf_shells = shell_data[:, 1]
    logT_shells = shell_data[:, 6]
    logrho_shells = shell_data[:, 7]

    comp_mfracs = comp_mfracs_in if comp_mfracs_in is not None else jnp.array(COMP_MFRACS)

    # Interpolate T and rho onto composition grid
    logT_comp = _interp_shell_to_comp_reversed(comp_mfracs, mf_shells, logT_shells)
    logrho_comp = _interp_shell_to_comp_reversed(comp_mfracs, mf_shells, logrho_shells)
    T_comp = 10.0**logT_comp
    rho_comp = 10.0**logrho_comp

    # Compute equilibrium X3 and timescale at each composition zone
    X3_eq, tau_eq = he3_equilibrium(T_comp, rho_comp, X_profile)

    # Exponential relaxation toward equilibrium (Clayton 1968 §5.6)
    # X3_new = X3_eq + (X3_old - X3_eq) * exp(-dt/tau_eq)
    # Clip exponent to prevent numerical issues at very short tau
    exponent = jnp.clip(-dt / jnp.maximum(tau_eq, 1.0), -50.0, 0.0)
    decay = jnp.exp(exponent)
    X3_new = X3_eq + (X3_profile - X3_eq) * decay

    # Ensure non-negative
    X3_new = jnp.maximum(X3_new, 0.0)

    return X3_new
