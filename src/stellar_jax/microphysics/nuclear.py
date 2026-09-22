"""Nuclear energy generation (PP + CNO).

Neutrino convention:
  The parametric CF88 energy constants (2.38e6 for pp, 8.67e27 for CNO)
  encode **neutrino-net** Q-values (Clayton 1968 §5.4: Q_eff = Q_total − Q_ν).
  For pp-I: Q_eff = (26.73 − 2×0.26)/2 ≈ 13.1 MeV per pp reaction.
  For CNO-I: Q_eff accounts for the 13N and 15O β+ neutrino losses (~2.9 MeV/cycle).
  MESA uses a different but equivalent approach: raw rates × total Q → eps_total,
  then subtracts eps_neu separately (net_eval.f90: eps_nuc = eps_total − eps_neu_total).
  Both yield the same nuclear energy deposited in the gas. The downstream thermal
  neutrino subtraction (neutrino.py: plasma/photo/pair) is a separate loss channel
  and correctly applied on top.

References:
  - Caughlan & Fowler (1988, ADNDT 40, 283): PP + CNO reaction rates
  - Adelberger et al. (2011, Rev. Mod. Phys. 83, 195): updated S-factors
  - Clayton (1968), Principles of Stellar Evolution and Nucleosynthesis
  - MESA: net/private/net_eval.f90, net_approx21.f90 (eps_nuc = eps_total − eps_neu)
"""
import numpy as np
import jax
import jax.numpy as jnp

from stellar_jax.microphysics.safe_math import safe_power
from stellar_jax.config.physics_floors import X_MIN_PP

# ═══════════════════════════════════════════════════════════════════════════════
# Physical constants in CGS (CODATA 2018, matching MESA const_def)
# ═══════════════════════════════════════════════════════════════════════════════
_QE = 4.8032047e-10       # elementary charge [esu = statcoulomb]
_KERG = 1.380649e-16      # Boltzmann constant [erg/K]
_HBAR = 1.054571817e-27   # reduced Planck constant [erg·s]
_AMU = 1.6605390666e-24   # atomic mass unit [g]
_PI = jnp.pi
_PI4 = 4.0 * _PI

# ═══════════════════════════════════════════════════════════════════════════════
# Chugunov (2007) fitting coefficients (eqs. 19, 21, and text after eq. 21)
# ═══════════════════════════════════════════════════════════════════════════════
_C_A1 = 2.7822
_C_A2 = 98.34
_C_A3 = jnp.sqrt(3.0) - _C_A1 / jnp.sqrt(_C_A2)
_C_B1 = -1.7476
_C_B2 = 66.07
_C_B3 = 1.12
_C_B4 = 65.0
_ALFA = 0.022             # coefficient of linear ζ term (Chugunov 2007 eq. 21)

# Domain limits (matching MESA screen_chugunov.f90)
_GAMFITLIM = 600.0        # upper limit for H(Γ̃) fit applicability
_G0 = 590.0               # blending starts here
_TP2 = 0.2                # minimum T/T_p before fading
_TP1 = 0.1                # floor T/T_p
_H0FITLIM = 300.0         # cap on H to prevent overflow


# ═══════════════════════════════════════════════════════════════════════════════
# AD-safe x^{3/2} via @custom_jvp.
#
# JAX's product-rule AD of x*sqrt(x) gives sqrt(x) + x/(2*sqrt(x)); at x=0
# (FTZ/DAZ hardware flushes subnormal gamtild to exactly 0) this evaluates as
# 0 + 0*inf = NaN. The mathematical derivative d/dx[x^{3/2}] = (3/2)*sqrt(x)
# is well-defined (= 0) at x=0.
#
# MESA screen_chugunov.f90:258-260 hand-codes the same derivative:
#   dAdt = x32 * (A / gamtild) * dgamtilddt  [= (3/2)*sqrt(gamtild)*dgamtilddt]
#
# The @custom_jvp preserves the forward XLA HLO byte-for-byte (forward body IS
# x*sqrt(x)), so XLA's whole-program optimization of the backward pass is not
# disrupted by new ops. Only the differentiation rule changes.
# ═══════════════════════════════════════════════════════════════════════════════
@jax.custom_jvp
def _pow_3_2(x):
    """Compute x^{3/2} = x * sqrt(x), AD-safe at x=0."""
    return x * jnp.sqrt(x)


@_pow_3_2.defjvp
def _pow_3_2_jvp(primals, tangents):
    (x,) = primals
    (t,) = tangents
    primal_out = x * jnp.sqrt(x)
    # d/dx[x^{3/2}] = (3/2)*sqrt(x). Use maximum(x, 0) to handle x < 0 from
    # numerical noise (sqrt of negative → NaN). For x >= 0 this is a no-op.
    tangent_out = 1.5 * jnp.sqrt(jnp.maximum(x, 0.0)) * t
    return primal_out, tangent_out


def screen_chugunov(z1, z2, a1, a2, rho, T, zbar, abar):
    """Compute Chugunov+DeWitt+Yakovlev (2007) screening enhancement factor.

    Implements MESA's chugunov_screening mode (rates/private/screen_chugunov.f90).
    Handles weak/intermediate/strong regimes seamlessly via the effective Coulomb
    coupling parameter Γ̃.

    Uses native JAX AD for both forward-mode (jacfwd in Henyey Jacobian) and
    reverse-mode (jax.grad for composition gradients). The XLA graph for a single
    screening evaluation (~50 operations) is manageable because the Henyey IFT
    @custom_vjp (commit decea942) collapses the backward pass to ONE Jacobian +
    ONE VJP at the converged state — no unrolled AD through 40 Newton iterations.

    The earlier @custom_jvp with finite-difference derivatives was removed because:
    (1) The IFT backward makes it redundant (no graph explosion risk).
    (2) The FD approach introduced subtle gradient inconsistencies (e.g., leakage
        below internal floors, ~0.025% relative error) that accumulated over 50
        lax.scan steps × 600 RK4 steps in the backward pass, causing NaN in the
        composition gradient test (test_gradient_composition_1p5Msun_50steps).
    (3) Native AD produces exact, finite derivatives at all stellar conditions
        (verified at rho=1e-10..200, T=5e3..3e7).

    Parameters
    ----------
    z1, z2 : float
        Charge numbers of reactants (e.g., 1 for proton, 7 for ¹⁴N)
    a1, a2 : float
        Mass numbers of reactants (e.g., 1 for proton, 14 for ¹⁴N)
    rho : float
        Density [g/cc]
    T : float
        Temperature [K]
    zbar : float
        Mean charge per ion in the plasma
    abar : float
        Mean mass number per ion in the plasma

    Returns
    -------
    screen : float
        Screening enhancement factor exp(H) ≥ 1. Multiply the bare rate by this.

    References
    ----------
    Chugunov, DeWitt & Yakovlev (2007), PhRvD 76, 025028
    MESA: rates/private/screen_chugunov.f90
    """
    return _screen_chugunov_impl(z1, z2, a1, a2, rho, T, zbar, abar)


def _screen_chugunov_impl(z1, z2, a1, a2, rho, T, zbar, abar):
    """Chugunov (2007) screening computation — differentiated by native JAX AD.

    This is the full implementation used by screen_chugunov(). JAX traces
    through it in both forward mode (jacfwd for Henyey Jacobian) and reverse
    mode (jax.grad for composition/parameter gradients). All operations are
    smooth and differentiable within the floored domain (rho >= 1e-30, T >= 100).
    """
    z1 = jnp.float64(z1)
    z2 = jnp.float64(z2)
    a1 = jnp.float64(a1)
    a2 = jnp.float64(a2)
    rho = jnp.float64(jnp.maximum(rho, 1e-30))
    T = jnp.float64(jnp.maximum(T, 100.0))
    zbar = jnp.float64(jnp.maximum(zbar, 1.0))
    abar = jnp.float64(jnp.maximum(abar, 1.0))

    # No screening for neutral particles
    z1z2 = z1 * z2
    # If z1 or z2 is 0, return 1.0 (handled by clamping h0fit to >= 0)

    # Total number density of ions
    ntot = rho / (_AMU * abar)

    # Electron sphere radius (Itoh 1979 eq. 1-3)
    a_e = (3.0 / (_PI4 * zbar * ntot)) ** (1.0 / 3.0)

    # Ion sphere radii for species 1 and 2
    a_1 = a_e * z1 ** (1.0 / 3.0)
    a_2 = a_e * z2 ** (1.0 / 3.0)
    a_av = 0.5 * (a_1 + a_2)

    # Average ion mass for plasma frequency
    mav = abar * _AMU

    # Plasma frequency and temperature (Chugunov 2007 eq. 2)
    wp = jnp.sqrt(_PI4 * z1z2 * _QE * _QE * ntot / mav)
    tp = _HBAR * wp / _KERG

    # Normalized temperature tn = T / T_p
    tn = T / tp

    # Effective temperature tk, with blending for very low T/T_p
    # (matching MESA's floor behavior to prevent numerical issues)
    tk = jnp.where(
        tn <= _TP1,
        _TP1 * tp,
        jnp.where(
            tn <= _TP2,
            # Cosine blend between tp1*tp and tp2*tp
            (1.0 - 0.5 * (1.0 - jnp.cos(_PI * (tn - _TP1) / (_TP2 - _TP1)))) * _TP1 * tp
            + 0.5 * (1.0 - jnp.cos(_PI * (tn - _TP1) / (_TP2 - _TP1))) * _TP2 * tp,
            T
        )
    )

    # Zeta: quantum correction parameter (Chugunov 2007 eq. 3)
    # ζ = (4 T_p² / (3 π² T_k²))^(1/3)
    U = (4.0 / 3.0) * (tp * tp) / (_PI * _PI * tk * tk)
    # AD-safe: U^(1/3) has gradient (1/3)*U^(-2/3) → ∞ as U → 0 at
    # low-density surface zones (small ntot → small wp → small tp → U ≈ 0).
    # The VJP through _build_residual_fixed_bc evaluates epsilon_nuclear at
    # ALL zones; surface zones with U ≈ 0 produce NaN via 0 × ∞ (IEEE 754).
    # Floor = 1e-30: screening is negligible (ζ ≈ 0, denom ≈ 1, Γ̃ ≈ Γ,
    # H ≈ 0) at U < 1e-30 (Chugunov 2007 eq. 3: ζ < 1e-10).
    zeta = safe_power(U, 1.0 / 3.0, 1e-30)

    # Coulomb coupling parameter Γ (Itoh 1979 eq. 4)
    gam = z1z2 * _QE * _QE / (a_av * tk * _KERG)

    # Limit Γ to gamfitlim with blending (MESA screen_chugunov.f90)
    gam = jnp.where(
        gam >= _GAMFITLIM,
        _GAMFITLIM,
        jnp.where(
            gam >= _G0,
            # Cosine blend
            (1.0 - 0.5 * (1.0 - jnp.cos(_PI * (gam - _G0) / (_GAMFITLIM - _G0)))) * gam
            + 0.5 * (1.0 - jnp.cos(_PI * (gam - _G0) / (_GAMFITLIM - _G0))) * _GAMFITLIM,
            gam
        )
    )

    # Coefficients β and γ dependent on Γ (Chugunov 2007, after eq. 21)
    beta = 0.41 - 0.6 / gam
    gama = 0.06 + 2.2 / gam

    # Effective Coulomb parameter Γ̃ (Chugunov 2007 eq. 21)
    zeta2 = zeta * zeta
    zeta3 = zeta2 * zeta
    denom = (1.0 + _ALFA * zeta + beta * zeta2 + gama * zeta3) ** (-1.0 / 3.0)
    gamtild = gam * denom

    # Mean-field potential H(Γ̃) fit (Chugunov 2007 eq. 19)
    # H = Γ̃^(3/2) * [a1/sqrt(a2 + Γ̃) + a3/(1 + Γ̃)]
    #   + b1 * Γ̃² / (b2 + Γ̃)
    #   + b3 * Γ̃² / (b4 + Γ̃²)
    gamtild_32 = _pow_3_2(gamtild)
    t1 = gamtild_32 * (_C_A1 / jnp.sqrt(_C_A2 + gamtild) + _C_A3 / (1.0 + gamtild))

    gamtild2 = gamtild * gamtild
    t2 = _C_B1 * gamtild2 / (_C_B2 + gamtild)
    t3 = _C_B3 * gamtild2 / (_C_B4 + gamtild2)

    h0fit = t1 + t2 + t3

    # Cap H to prevent overflow (matching MESA)
    h0fit = jnp.minimum(h0fit, _H0FITLIM)
    # Floor at 0 (screening never reduces the rate; handles z1=0 or z2=0 case)
    h0fit = jnp.maximum(h0fit, 0.0)

    # Screening factor
    screen = jnp.exp(h0fit)

    return screen


def _screen_pp(rho, T, zbar, abar):
    """Screening factor for p + p reaction (Z1=Z2=1, A1=A2=1).

    Chugunov (2007) at solar-core conditions gives f_pp ≈ 1.05,
    consistent with Salpeter (1954) weak screening.
    """
    return screen_chugunov(1.0, 1.0, 1.0, 1.0, rho, T, zbar, abar)


def _screen_cno(rho, T, zbar, abar):
    """Screening factor for 14N(p,γ)15O — the rate-limiting CNO step.

    Z1=1 (proton), Z2=7 (nitrogen-14), A1=1, A2=14.
    At solar-core conditions (Γ ≈ 0.5), this is in the intermediate regime
    and gives f_CNO ≈ 1.05-1.20.

    Reference: the 14N(p,γ)15O reaction has the smallest cross-section in
    the CN cycle and therefore controls the CNO energy generation rate
    (Adelberger+ 2011, Rev. Mod. Phys. 83, 195).
    """
    return screen_chugunov(1.0, 7.0, 1.0, 14.0, rho, T, zbar, abar)


def _screen_he3_he3(rho, T, zbar, abar):
    """Screening factor for ³He + ³He → ⁴He + 2p.

    Z1=Z2=2 (helium-3), A1=A2=3.
    At solar-core conditions gives f₃₃ ≈ 1.16 (Chugunov 2007).

    MESA screens this reaction via the automatic reaction_screening_info
    mechanism (rates/private/rates_initialize.f90:334: any reaction with
    2+ charged inputs gets screening). Previously missing in our code.

    Reference: MESA rates/private/screen_chugunov.f90
    """
    return screen_chugunov(2.0, 2.0, 3.0, 3.0, rho, T, zbar, abar)


def _screen_he3_he4(rho, T, zbar, abar):
    """Screening factor for ³He + ⁴He → ⁷Be + γ.

    Z1=2 (helium-3), Z2=2 (helium-4), A1=3, A2=4.
    At solar-core conditions gives f₃₄ ≈ 1.16 (Chugunov 2007).
    Equal to f₃₃ because Chugunov screening depends only on Z₁·Z₂,
    not on mass numbers (mass enters only through the plasma frequency).

    MESA screens this reaction via the automatic reaction_screening_info
    mechanism. Previously missing in our code.

    Reference: MESA rates/private/screen_chugunov.f90
    """
    return screen_chugunov(2.0, 2.0, 3.0, 4.0, rho, T, zbar, abar)


def _plasma_composition(X, Z):
    """Compute mean charge (zbar) and mean mass number (abar) for the plasma.

    Assumes solar-like composition: H + He + metals (approximated as ¹⁴N-like).
    Y = 1 - X - Z.

    Returns (zbar, abar).
    """
    Y = 1.0 - X - Z
    # Number fractions: n_i ∝ X_i / A_i
    # H: X/1, He: Y/4, metals: Z/14 (approximating as N-14)
    n_H = X / 1.0
    n_He = Y / 4.0
    n_Z = Z / 14.0
    n_total = n_H + n_He + n_Z

    # Mean mass number: abar = 1 / (sum X_i/A_i)
    abar = 1.0 / jnp.maximum(n_total, 1e-30)

    # Mean charge: zbar = sum(n_i * Z_i) / sum(n_i)
    # H: Z=1, He: Z=2, metals (N-like): Z=7
    zbar = (n_H * 1.0 + n_He * 2.0 + n_Z * 7.0) / jnp.maximum(n_total, 1e-30)

    return zbar, abar


def epsilon_nuclear(rho, T, X, Z, t_age=1e9, X_N14=None, eps_nuc_factor=None, X3=None, X3_eq_frozen=None):
    """Nuclear energy generation rate with Chugunov (2007) regime-aware screening.

    PP chain + CNO cycle with screening applied to BOTH reactions via the
    Chugunov+DeWitt+Yakovlev (2007) prescription. This handles weak (Γ << 1),
    intermediate, and strong (Γ >> 1) screening regimes seamlessly.

    The Chugunov formula is MESA's default since r15140 and supersedes the
    older Graboske/Salpeter/screen5 approach.

    Neutrino convention (verified against MESA net_eval.f90, issue #990):
        The CF88 parametric constants (2.38e6 for pp, 8.67e27 for CNO) encode
        neutrino-NET Q-values (Clayton 1968 §5.4). The pp constant gives the
        energy deposited in the gas per pp-initiated chain, already subtracting
        the pp-neutrino losses (~0.26 MeV average per pp reaction). MESA uses
        a different path (raw rates × total Q from mass excesses, then subtracts
        eps_neu_total from reaction_neuQs) but arrives at the same eps_nuc.
        The thermal neutrinos (neutrino.py: plasma/photo/pair) are a separate
        loss channel, correctly subtracted downstream.

    Parameters
    ----------
    rho : float
        Density [g/cc]
    T : float
        Temperature [K]
    X : float
        Hydrogen mass fraction
    Z : float
        Metal mass fraction
    t_age : float
        Stellar age [years] (for pp-chain ³He equilibrium placeholder)
    X_N14 : float or None
        ¹⁴N mass fraction (tracked). When provided, this is used directly as the
        CNO catalyst abundance for the rate-limiting ¹⁴N(p,γ)¹⁵O reaction, matching
        MESA's approx21 network which uses the actual y(in14) (net_approx21.f90:1116).
        When None, falls back to ZAMS CN+ON equilibrium X_CNO = 0.251*Z
        (Asplund 2009; valid in CN-equilibrated cores). See #1026 for rationale.
    eps_nuc_factor : float or None
        Multiplicative factor applied to the total eps_nuc (MESA control
        ``eps_nuc_factor``, default 1.0). Applied AFTER computing eps_nuc,
        matching MESA star/private/net.f90:365-372:
            eps_nuc_factor = s% eps_nuc_factor
            if (eps_nuc_factor /= 1d0) then
                s% eps_nuc(k) = s% eps_nuc(k) * eps_nuc_factor
                s% d_epsnuc_dlnd(k) = ... * eps_nuc_factor
                s% d_epsnuc_dlnT(k) = ... * eps_nuc_factor
        Because this is a differentiable code, all partial derivatives are
        handled automatically by JAX AD — no need to scale them manually.
        Default None → 1.0 (no modification, bit-identical to previous behavior).
    X3 : float or None
        Local ³He mass fraction (from the scan carry, issue #112). When provided,
        phi is computed locally as min(1, X3²/X3_eq²) — the pp-chain branching
        depends on (T, ρ, X) per shell. When None, falls back to the global-age
        placeholder phi = 1 - 0.3*exp(-t_age/5e6) for backward compatibility.

    Returns
    -------
    eps : float
        Total nuclear energy generation rate [erg/g/s]
    """
    rho = jnp.maximum(rho, 1e-10)
    T6 = jnp.maximum(T * 1e-6, 0.5)
    # AD-safe T6^(1/3): gradient (1/3)*T6^(-2/3) diverges as T6→0.
    # Floor at 0.5 (matching the forward-pass floor above) ensures the
    # gradient stays bounded. CF88 §IV: rates are negligible below T6=0.5
    # (Gamow peak energy >> kT), so the gradient there is physically zero.
    T6_13 = safe_power(T6, 1.0/3.0, 0.5)
    T6_23 = T6_13 * T6_13

    # Effective temperature for screening: use the SAME floored T as the rate
    # computation (T_eff = T6 * 1e6 = max(T, 5e5 K)). The T6 floor exists because
    # the Gamow-peak rate formulae are physically meaningless below ~500 kK. The
    # screening must use the same domain.
    # Reference: MESA evaluates screening only at the same T used for the rate.
    T_eff = T6 * 1e6

    # Compute plasma composition for screening
    zbar, abar = _plasma_composition(X, Z)

    # ═══════════════════════════════════════════════════════════════════════
    # PP chain energy generation (Caughlan & Fowler 1988)
    # ═══════════════════════════════════════════════════════════════════════
    # Chugunov screening for p + p (replaces old Salpeter fpp)
    f_pp = _screen_pp(rho, T_eff, zbar, abar)

    # Higher-order electrostatic correction (psipp: pp-chain electron screening)
    # ψ_pp and the C_pp rate polynomial: Caughlan & Fowler (1988), ADNDT 40, 283
    # (the pp reaction, their λ_pp fit; coefficients are CF88's, not fitted here).
    psipp = 1.0 + 1.412e8 * (1.0 / jnp.maximum(X, X_MIN_PP) - 1.0) * jnp.exp(-49.98 / T6_13)
    Cpp = 1.0 + 0.0123 * T6_13 + 0.0109 * T6_23 + 0.000938 * T6

    eps_pp = 2.38e6 * rho * X * X * f_pp * psipp * Cpp / T6_23 * jnp.exp(-33.80 / T6_13)
    # pp-I → pp-II transition via ³He equilibrium.
    # KNOWN-INCORRECT PLACEHOLDER (gated behind):
    #   Clayton (1968) §5.4 eq. 5.67 gives φ as a function of LOCAL ³He
    # pp-I → pp-II/III branching via ³He equilibrium.
    #
    # phi = fraction of pp-chain energy going through pp-I (³He+³He completion).
    # When X3 < X3_eq, pp-I dominates (phi → 1). When X3 ≈ X3_eq, pp-II/III
    # contribute their full equilibrium share (phi < 1).
    #
    # The simplified form (Clayton 1968 §5.6, spec):
    #   phi_local = min(1, X3² / X3_eq²)
    # MESA computes this rate-by-rate using actual abundances (net_approx21.f90:
    # 1547-1558). Our equilibrium approach gives the same result at equilibrium.
    #
    # When X3 is None (backward compatibility), fall back to the global-age
    # placeholder.
    if X3 is not None:
        # Local phi from ³He equilibrium
        # CONSTRAINT (MESA operator-split, struct_burn_mix.f90:82-100):
        # X3_eq is a START-OF-STEP quantity — MESA's burn uses frozen structure.
        # When X3_eq_frozen is provided (solver path), use it as the reference:
        # phi = X3²/X3_eq_frozen². Since the solver always has X3 = X3_eq from
        # the previous step's relaxation (τ₃ ≪ dt on the MS), and X3_eq_frozen
        # is the same value (also from the carry), phi = 1.0 — which is the
        # correct equilibrium value. Without freezing, X3_eq recomputed from
        # the Newton iterate's T creates a spurious phi↔T coupling (eps_grav
        # ≈ −0.10) because the steep T-dependence of X3_eq (T^{-16.7} via
        # Gamow peak) amplifies tiny Newton-convergence T shifts into large
        # phi variations.
        # Unit tests (no X3_eq_frozen) use the live X3_eq(T) to verify spatial
        # variation of phi in non-equilibrium conditions.
        if X3_eq_frozen is not None:
            X3_eq_local = X3_eq_frozen
        else:
            X3_eq_local, _ = he3_equilibrium(T_eff, rho, X, zbar=zbar, abar=abar, Z=Z)
        phi = jnp.minimum(1.0, X3**2 / jnp.maximum(X3_eq_local**2, 1e-40))
    else:
        # LEGACY placeholder (pre-): global-age approximation.
        phi = 1.0 - 0.3 * jnp.exp(-t_age / 5.0e6)

    # ═══════════════════════════════════════════════════════════════════════
    # CNO cycle energy generation (Caughlan & Fowler 1988)
    # Rate-limiting step: ¹⁴N(p,γ)¹⁵O
    # ═══════════════════════════════════════════════════════════════════════
    # Chugunov screening for p + ¹⁴N (NEW — previously unscreened)
    f_cno = _screen_cno(rho, T_eff, zbar, abar)

    # CNO rate polynomial correction g_cno for ¹⁴N(p,γ)¹⁵O — Caughlan & Fowler
    # (1988), ADNDT 40, 283 (their λ_14N fit; coefficients are CF88's).
    g_cno = 1.0 + 0.0027 * T6_13 - 0.00778 * T6_23 - 0.000149 * T6
    # CNO catalyst abundance: use tracked ¹⁴N when available.
    # MESA's approx21 network uses the actual N14 number density y(in14) directly
    # in the rate computation (net_approx21.f90:1116):
    #   eps_cno ∝ y(in14) * y(ih1) * rate(irnpg)
    # The rate-limiting step is ¹⁴N(p,γ)¹⁵O, so eps_cno ∝ X_N14, not X_CNO_total.
    #
    # When X_N14 is provided (from the solver's burn_cno), use it directly as
    # the catalyst abundance — matching MESA's treatment. Our burn_cno conserves
    # CN_total = (C12+C13+N14) = 0.251*Z (initialized to match MESA ZAMS,
    # including PMS ON-cycle N14 gain,); at CN equilibrium, tracked
    # N14 ≈ CN_total ≈ 0.251*Z. We do NOT add the O16 fraction because MESA
    # uses only y(in14) in the rate (net_approx21.f90:1116).
    # CONSTRAINT: during the MS, MESA's N14 grows further (to ~0.326*Z at
    # 2.0 Msun midMS) as the ON sub-cycle continues converting O16→N14. Our
    # CN_total is fixed at 0.251*Z, so we underestimate N14 by ~12-23% at
    # midMS for CNO-dominant masses — a bounded CONSTRAINT of the CN-only
    # network (adding O16 tracking requires a new scan-carry species).
    # MESA ref: net_approx21.f90:1116 (y(in14)*y(ih1)*rate(irnpg)).
    #
    # Fallback (X_N14=None): ZAMS N14 after PMS CN+ON equilibration = 0.251*Z.
    # MESA's ZAMS X_N14 is ~0.00351 across 1.0-2.0 Msun (measured from FGONG
    # col 23). This exceeds the CN-only equilibrium (0.221*Z = 0.00309) because
    # PMS evolution (~10-30 Myr at T~15-20 MK) converts ~6% of initial O16 to
    # N14 via the ON sub-cycle before reaching ZAMS. The 0.251*Z = 0.00351
    # matches MESA's ZAMS N14 to <1% across all masses. The prior CN-only
    # fallback (0.221*Z) underestimated by 12-13%, causing ~10% eps deficit at
    # 2.0 Msun (86% CNO) — enough to shift the ZAMS structure beyond the
    # 0.01 dex logTeff tolerance.
    # MESA ref: net_approx21.f90:1116 (y(in14) set by PMS evolution).
    _CN_EQ_FRAC = 0.251  # ZAMS N14/Z after PMS CN+ON equilibration (Asplund 2009 + ON)
    X_CNO = X_N14 if X_N14 is not None else _CN_EQ_FRAC * Z
    # CNO rate constant: CF88 (Caughlan & Fowler 1988, ADNDT 40, 283).
    # The constant 8.67e27 encodes the neutrino-net cycle Q-value (Clayton 1968
    # §5.4: Q_eff ≈ 25.0 MeV per CNO-I cycle, minus ~1.7 MeV ν losses) and the
    # CF88 S(0) = 3.32 keV·barn for ¹⁴N(p,γ)¹⁵O. MESA's default rate provider
    # rate_n14pg_nacre (ratelib.f90:1506-1533, a0=4.83e7) uses the NACRE
    # (Angulo+ 1999) S-factor S(0) ≈ 3.2 keV·barn — within 4% of CF88. The
    # parametric rate shape (Gamow peak + g_cno polynomial) matches the NACRE
    # analytic form to <2% across 10-110 MK (verified against MESA ratelib.f90).
    #
    # Note: LUNA (Formicola+ 2004, Marta+ 2008) measured S(0) = 1.57 ± 0.13 and
    # NACRE II (Xu+ 2013) adopted this lower value. But MESA's DEFAULT dispatch
    # (raw_rates.f90:233: `call do1(rate_n14pg_nacre)`) uses the analytic NACRE
    # (1999) formula, not the NACRE II table. We match MESA's default for MODE-A.
    eps_cno = 8.67e27 * rho * X * X_CNO * f_cno / T6_23 * jnp.exp(-152.28 / T6_13) * g_cno

    eps_total = eps_pp * phi + eps_cno
    # Apply eps_nuc_factor (MESA star/private/net.f90:367):
    #   s% eps_nuc(k) = s% eps_nuc(k) * eps_nuc_factor
    # JAX AD automatically propagates the factor into all partial derivatives.
    if eps_nuc_factor is not None:
        eps_total = eps_total * eps_nuc_factor
    return eps_total


def _eps_nuclear_np(rho, T, X, Z, t_age, X_N14=None):
    """NumPy version of epsilon_nuclear for write_fgong loops (with Chugunov screening)."""
    rho = max(rho, 1e-10)
    T6 = max(T * 1e-6, 0.5)
    T6_13 = T6 ** (1.0 / 3.0)
    T6_23 = T6_13 * T6_13

    # Plasma composition
    Y = 1.0 - X - Z
    n_H = X / 1.0
    n_He = Y / 4.0
    n_Z = Z / 14.0
    n_total = n_H + n_He + n_Z
    abar = 1.0 / max(n_total, 1e-30)
    zbar = (n_H * 1.0 + n_He * 2.0 + n_Z * 7.0) / max(n_total, 1e-30)

    # Screening factors (use T_eff = T6*1e6 consistent with the rate's T domain)
    T_eff = T6 * 1e6
    f_pp = float(_screen_pp(rho, T_eff, zbar, abar))
    f_cno = float(_screen_cno(rho, T_eff, zbar, abar))

    # PP chain
    # ψ_pp / C_pp: Caughlan & Fowler (1988), ADNDT 40, 283 (pp rate fit).
    psipp = 1.0 + 1.412e8 * (1.0 / max(X, 1e-3) - 1.0) * np.exp(-49.98 / T6_13)
    Cpp = 1.0 + 0.0123 * T6_13 + 0.0109 * T6_23 + 0.000938 * T6
    e_pp = 2.38e6 * f_pp * X ** 2 * rho * psipp * Cpp / T6_23 * np.exp(-33.80 / T6_13)

    # CNO cycle (CF88 constant 8.67e27, matching MESA's NACRE S-factor;)
    # Catalyst: X_N14 directly when tracked; ZAMS CN+ON-eq fallback (0.251*Z).
    # See epsilon_nuclear docstring for rationale.
    _CN_EQ_FRAC = 0.251
    X_CNO = X_N14 if X_N14 is not None else _CN_EQ_FRAC * Z
    g_cno = 1.0 + 0.0027 * T6_13 - 0.00778 * T6_23 - 0.000149 * T6
    e_cno = 8.67e27 * X_CNO * X * rho * f_cno * g_cno / T6_23 * np.exp(-152.28 / T6_13)

    return e_pp + e_cno


# ═══════════════════════════════════════════════════════════════════════════════
# ³He Equilibrium (Clayton 1968 §5.6,)
#
# MESA reference: MESA evolves ³He explicitly in the approx21 network
# (net/private/net_approx21.f90:971-978), computing fII from actual rates.
#
# CONSTRAINT (JAX/differentiability): Our simplified network uses the Clayton
# (1968) equilibrium approximation rather than evolving ³He as a full species.
# Valid on the MS where τ₃(core) ≪ τ_evolution.
# ═══════════════════════════════════════════════════════════════════════════════


def he3_equilibrium(T, rho, X, zbar=None, abar=None, Z=0.014):
    """Compute equilibrium ³He mass fraction and relaxation timescale.

    Solves the rate balance (Clayton 1968 §5.6 eq. 5.62):
        production(pp) = destruction(³He+³He) + destruction(³He+⁴He)

    The production AND destruction rates are screened with Chugunov (2007)
    factors (f_pp for production, f₃₃/f₃₄ for destruction), matching MESA's
    approach where ALL charged-particle reactions are screened
    (MESA rates/private/rates_initialize.f90:333: particles_in > 1).
    At solar-core conditions, f_pp ≈ 1.05, f₃₃ ≈ f₃₄ ≈ 1.16, shifting
    X3_eq by ~7% relative to unscreened rates.

    Rates used:
        pp:  CF88/FXT (MESA ratelib.f90:81-82)
        33:  ³He + ³He → ⁴He + 2p  (NACRE; MESA ratelib.f90:365-375)
        34:  ³He + ⁴He → ⁷Be + γ   (NACRE; MESA ratelib.f90:415-420)

    Parameters
    ----------
    T : array_like
        Temperature [K]
    rho : array_like
        Density [g/cm³]
    X : array_like
        Hydrogen mass fraction
    zbar : float or None
        Mean charge per ion. If None, computed from (X, Z) via _plasma_composition.
    abar : float or None
        Mean mass number per ion. If None, computed from (X, Z) via _plasma_composition.
    Z : float
        Metal mass fraction (used only if zbar/abar are None). Default 0.014.

    Returns
    -------
    X3_eq : array_like
        Equilibrium ³He mass fraction
    tau_eq : array_like
        Relaxation timescale [seconds] for X3 → X3_eq
    """
    # Temperature in units of 10⁹ K (standard for thermonuclear rates)
    T9 = jnp.maximum(T * 1e-9, 5e-4)  # floor at 0.5 MK
    T9_13 = safe_power(T9, 1.0 / 3.0, 1e-30)
    T9_23 = T9_13 * T9_13

    # -----------------------------------------------------------------------
    # Screening factors for ³He reactions (Chugunov 2007, MESA-matched)
    # MESA screens every charged-particle reaction automatically
    # (rates_initialize.f90:334: reaction_screening_info for particles_in > 1).
    # -----------------------------------------------------------------------
    if zbar is None or abar is None:
        zbar_local, abar_local = _plasma_composition(X, Z)
    else:
        zbar_local, abar_local = zbar, abar

    # Use effective T for screening, floored at 0.5 MK (matching eps_nuclear's domain)
    T_eff = jnp.maximum(T, 5e5)
    f_33 = _screen_he3_he3(rho, T_eff, zbar_local, abar_local)
    # f_34 ≡ f_33: Chugunov screening depends only on z1·z2 (=4 for both
    # ³He+³He and ³He+⁴He, since both have Z₁=Z₂=2). Mass numbers a1, a2
    # are accepted by screen_chugunov but unused in the formula (verified:
    # MESA screen_chugunov.f90 lines 68-72 declare a1/a2 but never reference
    # them in the computation). Computing one call instead of two reduces
    # the forward XLA graph inside lax.scan.
    f_34 = f_33

    # -----------------------------------------------------------------------
    # Thermonuclear reaction rates <σv>·N_A [cm³/(mol·s)]
    # pp: CF88/FXT (MESA ratelib.f90:81-82)
    # ³He+³He, ³He+⁴He: NACRE (MESA ratelib.f90:365-375, 415-420)
    # -----------------------------------------------------------------------
    # pp: p + p → d + e⁺ + ν (CF88/FXT; MESA ratelib.f90:81-82)
    # Coefficients in T9 units: bb = 1 + 0.123*T9^(1/3) + 1.09*T9^(2/3) + 0.938*T9
    Cpp = 1.0 + 0.123 * T9_13 + 1.09 * T9_23 + 0.938 * T9
    rate_pp = 4.01e-15 / T9_23 * jnp.exp(-3.380 / T9_13) * Cpp

    # ³He+³He → ⁴He + 2p (NACRE; MESA ratelib.f90:356-375)
    C33 = 1.0 - 0.135 * T9 + 2.54e-2 * T9**2 - 1.29e-3 * T9**3
    rate_33 = 5.59e10 / T9_23 * jnp.exp(-12.277 / T9_13) * C33

    # ³He+⁴He → ⁷Be + γ (NACRE; MESA ratelib.f90:413-430)
    C34 = 1.0 - 0.307 * T9 + 8.81e-2 * T9**2 - 1.06e-2 * T9**3 + 4.46e-4 * T9**4
    rate_34 = 5.46e6 / T9_23 * jnp.exp(-12.827 / T9_13) * C34

    # -----------------------------------------------------------------------
    # Rate balance for equilibrium X₃ (Clayton 1968 §5.6 eq. 5.62)
    # -----------------------------------------------------------------------
    # dX₃/dt per unit mass:
    #   production:     P = (3/2)·ρ·X²·rate_pp·f_pp
    #   destruction₃₃: D₃₃ = (1/3)·ρ·X₃²·rate_33 · f₃₃
    #   destruction₃₄: D₃₄ = (1/4)·ρ·X₃·Y·rate_34 · f₃₄
    #
    # At equilibrium P = D₃₃ + D₃₄:
    #   (3/2)·X²·rate_pp·f_pp = (1/3)·X₃²·rate_33·f₃₃ + (1/4)·X₃·Y·rate_34·f₃₄
    #
    # MESA screens ALL charged-particle reactions (rates_initialize.f90:333:
    # particles_in > 1), including p+p. The production rate must carry f_pp
    # so that the equilibrium balances screened rates on both sides.
    #
    # Quadratic in X₃: a·X₃² + b·X₃ - c = 0
    Y = jnp.maximum(1.0 - X - Z, 0.0)  # He-4 mass fraction

    f_pp = _screen_pp(rho, T_eff, zbar_local, abar_local)
    a_coeff = rate_33 / 3.0 * f_33
    b_coeff = Y * rate_34 / 4.0 * f_34
    c_coeff = 1.5 * X * X * rate_pp * f_pp

    # Solve: X3_eq = (-b + sqrt(b² + 4ac)) / (2a)
    discriminant = b_coeff**2 + 4.0 * a_coeff * c_coeff
    # Guard: a_coeff ∝ rate_33 can underflow to 0 at low T; 1e-100 keeps the
    # quadratic-formula denominator finite (and its AD gradient bounded).
    X3_eq = (-b_coeff + jnp.sqrt(jnp.maximum(discriminant, 0.0))) / (
        2.0 * jnp.maximum(a_coeff, 1e-100))

    # Floor: ³He cannot vanish in pp-burning regions
    X3_eq = jnp.maximum(X3_eq, 1e-20)

    # -----------------------------------------------------------------------
    # Relaxation timescale τ₃ (Clayton 1968 eq. 5.65)
    # -----------------------------------------------------------------------
    # Linearize dX₃/dt = P - D(X₃) around X3_eq:
    #   dX₃/dt ≈ -∂D/∂X₃|_{eq} · (X₃ - X3_eq) ≡ -(X₃ - X3_eq)/τ₃
    #
    # ∂D/∂X₃ = (2/3)·ρ·X3_eq·rate_33·f₃₃ + (1/4)·ρ·Y·rate_34·f₃₄
    # τ₃ = 1 / (ρ · ((2/3)·X3_eq·rate_33·f₃₃ + (1/4)·Y·rate_34·f₃₄))
    dD_dX3 = rho * ((2.0 / 3.0) * X3_eq * rate_33 * f_33 + 0.25 * Y * rate_34 * f_34)
    # Guard: dD_dX3 can underflow outside pp-burning regions; 1e-50 keeps
    # the reciprocal (τ₃ = 1/dD_dX3) finite for AD.
    tau_eq = 1.0 / jnp.maximum(dD_dX3, 1e-50)

    return X3_eq, tau_eq
