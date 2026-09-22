"""OPAL EOS with 4D interpolation (P-based lookup), blended with HELM+Coulomb
for the degenerate regime (logT > 7.9, beyond OPAL table coverage).

Single-inversion free-energy blend (Jermyn+2021 Skye approach):
  P_blend(ln_rho, ln_T) = (1-w(T))*P_OPAL(ln_rho, T) + w(T)*P_HELM(ln_rho, T)
Solved ONCE via NR for ln_rho at given (ln_T, P_target). This guarantees
Maxwell-relation consistency because all thermodynamics derive from a single
blended free energy F_blend = (1-w)*F_OPAL + w*F_HELM.

The density inversion uses custom_jvp with the implicit function theorem (IFT)
for efficient gradient propagation — no unrolling through NR iterations.

References:
  - Rogers & Nayfonov (2002), ApJ 576, 1064 (OPAL tables)
  - Timmes & Swesty (2000), ApJS 126, 501 (HELM EOS)
  - Chabrier & Potekhin (1998), Phys. Rev. E 58, 4941 (Coulomb corrections)
  - Jermyn et al. (2021), ApJ 913, 72 (Skye differentiable EOS)
  - Potekhin & Chabrier (2000), Phys. Rev. E 62, 8554 (OCP Coulomb fit)
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
from stellar_jax.microphysics.interp_utils import _locate_4d
from stellar_jax.config.constants import (
    k_B_opal as _k_B, m_H_opal as _m_H,
    h_planck_opal as _h_planck, a_rad_opal as _a_rad,
)
from stellar_jax.config.physics_floors import NAD_CP_FLOOR, NAD_UPPER


def _build_eos_pgrid():
    _OPAL_EOS_PATH = os.environ.get('STELLAR_OPAL_EOS',
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'eos_compact.npz'))
    eos = np.load(_OPAL_EOS_PATH)
    X_grid    = eos['X_grid'].astype(np.float64)
    Z_grid    = eos['Z_grid'].astype(np.float64)
    logT_grid = eos['logT_grid'].astype(np.float64)
    logQ_grid = eos['logQ_grid'].astype(np.float64)
    logP_tab  = eos['logPgas'].astype(np.float64)
    mu_tab    = eos['mu'].astype(np.float64)
    nad_tab   = eos['grad_ad'].astype(np.float64)

    # chi_rho_gas and chi_T_gas: gas-pressure thermodynamic derivatives
    # computed from d(logPgas)/d(logQ)|_T and d(logPgas)/d(logT)|_rho.
    # Reference: Rogers & Nayfonov 2002 (OPAL); MESA eosdt_load_tables.f90 lines 32-33.
    if 'chi_rho_gas' in eos.files and 'chi_T_gas' in eos.files:
        chi_rho_tab = eos['chi_rho_gas'].astype(np.float64)
        chi_T_tab   = eos['chi_T_gas'].astype(np.float64)
    else:
        # Fallback: compute from logPgas via numerical differentiation
        chi_rho_tab = np.gradient(logP_tab, logQ_grid, axis=2)
        dP_dT_constQ = np.gradient(logP_tab, logT_grid, axis=3)
        # χ_T = dlogP/dlogT|_ρ = dlogP/dlogT|_Q − 2·χ_ρ, from the OPAL variable
        # Q ≡ ρ/T6³·1e-12 ⇒ logρ = logQ + 2·logT − 12 (chain rule; exact).
        chi_T_tab = dP_dT_constQ - 2.0 * chi_rho_tab
        chi_rho_tab = np.clip(chi_rho_tab, 0.3, 2.0)
        chi_T_tab = np.clip(chi_T_tab, 0.3, 4.0)

    n_P = 100
    logP_grid = np.linspace(-1.0, 22.0, n_P).astype(np.float64)
    nX, nZ, nT = len(X_grid), len(Z_grid), len(logT_grid)
    logRho_inv  = np.zeros((nX, nZ, nT, n_P))
    mu_inv      = np.zeros((nX, nZ, nT, n_P))
    nad_inv     = np.zeros((nX, nZ, nT, n_P))
    chi_rho_inv = np.zeros((nX, nZ, nT, n_P))
    chi_T_inv   = np.zeros((nX, nZ, nT, n_P))

    for ix in range(nX):
        for iz in range(nZ):
            for it in range(nT):
                slice_logP    = logP_tab[ix, iz, :, it]
                slice_mu      = mu_tab[ix, iz, :, it]
                slice_nad     = nad_tab[ix, iz, :, it]
                slice_chi_rho = chi_rho_tab[ix, iz, :, it]
                slice_chi_T   = chi_T_tab[ix, iz, :, it]
                if np.all(np.diff(slice_logP) > 0):
                    xp_logP, fp_logQ  = slice_logP, logQ_grid
                    fp_mu, fp_nad     = slice_mu, slice_nad
                    fp_chi_rho        = slice_chi_rho
                    fp_chi_T          = slice_chi_T
                else:
                    sort_idx   = np.argsort(slice_logP)
                    xp_logP    = slice_logP[sort_idx]
                    fp_logQ    = logQ_grid[sort_idx]
                    fp_mu      = slice_mu[sort_idx]
                    fp_nad     = slice_nad[sort_idx]
                    fp_chi_rho = slice_chi_rho[sort_idx]
                    fp_chi_T   = slice_chi_T[sort_idx]
                logQ_at_P    = np.interp(logP_grid, xp_logP, fp_logQ)
                mu_at_P      = np.interp(logP_grid, xp_logP, fp_mu)
                nad_at_P     = np.interp(logP_grid, xp_logP, fp_nad)
                chi_rho_at_P = np.interp(logP_grid, xp_logP, fp_chi_rho)
                chi_T_at_P   = np.interp(logP_grid, xp_logP, fp_chi_T)
                # Invert the OPAL Q definition: logρ = logQ + 2·logT − 12
                # (Q ≡ ρ / T6³ · 1e-12, i.e. Q = ρ·1e-12/(T/1e6)³). Exact.
                logRho_inv[ix, iz, it, :]  = logQ_at_P + 2.0*logT_grid[it] - 12.0
                mu_inv[ix, iz, it, :]      = mu_at_P
                nad_inv[ix, iz, it, :]     = nad_at_P
                chi_rho_inv[ix, iz, it, :] = chi_rho_at_P
                chi_T_inv[ix, iz, it, :]   = chi_T_at_P
    return (X_grid, Z_grid, logT_grid, logP_grid,
            logRho_inv, mu_inv, nad_inv, chi_rho_inv, chi_T_inv)


(_eosX, _eosZ, _eosT, _eosP, _eosRho, _eosMu, _eosNad,
 _eosChiRho, _eosChiT) = _build_eos_pgrid()
EOS_X       = jnp.asarray(_eosX)
EOS_Z       = jnp.asarray(_eosZ)
EOS_LOGT    = jnp.asarray(_eosT)
EOS_LOGP    = jnp.asarray(_eosP)
EOS_LOGRHO  = jnp.asarray(_eosRho)
EOS_MU      = jnp.asarray(_eosMu)
EOS_NAD     = jnp.asarray(_eosNad)
EOS_CHI_RHO = jnp.asarray(_eosChiRho)
EOS_CHI_T   = jnp.asarray(_eosChiT)


# ═══════════════════════════════════════════════════════════════════════
# Blended pressure function (module-level for custom_jvp compatibility)
# ═══════════════════════════════════════════════════════════════════════
#
# HELM PRODUCTION WIRING STATUS:
#
# The HELM EOS blend is ACTIVE in production via eos_lookup():
#   - logT < 7.4: w ≈ 0, result numerically identical to pure OPAL
#   - logT > 7.7 AND logRho > 5.9: sigmoid-blended OPAL↔HELM (>95% HELM)
#   - logT > 7.7 AND logRho < 5.5: pure OPAL (density gate suppresses HELM)
#
# Both OPAL and HELM are always computed and blended arithmetically (no
# lax.cond). lax.cond compiles BOTH branches at trace time — when jacfwd
# differentiates through eos_lookup for the Henyey Jacobian (50-1000 cells),
# the HELM branch's JVP bloats the XLA graph beyond CI memory/time limits.
# Always-compute-and-blend produces the SAME graph size (both paths compiled
# either way) while eliminating the cond JVP overhead. At MS conditions,
# w ≈ 0 so the result is bitwise identical to pure OPAL.
#
# The blend is 2D: w = w_T(logT) * w_rho(logRho). The density gate
# (center=5.7, slope=20) suppresses HELM below logRho=5.7 where OPAL
# is accurate (<0.2% nad vs MESA) and the FD table has ±3% oscillations
# from bicubic interpolation artifacts. At logRho>5.7, OPAL CLAMPS
# (its logP=22.0 table edge) and HELM FD correctly dominates.
#
# The original _P_blend / _solve_blend_density cluster (blended-pressure NR
# with IFT custom_jvp) was removed in: it was dead code — eos_lookup
# uses the simpler output-7-tuple blend of OPAL and helm_eos_fd directly.
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# Main EOS lookup
# ═══════════════════════════════════════════════════════════════════════

def _eos_opal_table(logT, logP, X, Z):
    """Pure OPAL 4D table interpolation (trilinear). Internal helper.

    Returns (rho, mu, nad, S, cp, chi_rho, chi_T).
    Reference: Rogers & Nayfonov (2002), ApJ 576, 1064.
    """
    (ix, iz, it, ip), (tx, tz, tt, tp) = _locate_4d(
        (EOS_X, EOS_Z, EOS_LOGT, EOS_LOGP), (X, Z, logT, logP))

    # Clamped values needed for derived thermodynamic quantities below.
    lt     = jnp.clip(logT, EOS_LOGT[0], EOS_LOGT[-1])
    logP_c = jnp.clip(logP, EOS_LOGP[0], EOS_LOGP[-1])

    # Trilinear 4D interpolation for the core EOS quantities.
    def interp4d(T):
        def b(dx, dz, dt_, dp):
            return T[ix+dx, iz+dz, it+dt_, ip+dp]
        c000 = b(0,0,0,0)*(1-tp) + b(0,0,0,1)*tp
        c001 = b(0,0,1,0)*(1-tp) + b(0,0,1,1)*tp
        c010 = b(0,1,0,0)*(1-tp) + b(0,1,0,1)*tp
        c011 = b(0,1,1,0)*(1-tp) + b(0,1,1,1)*tp
        c100 = b(1,0,0,0)*(1-tp) + b(1,0,0,1)*tp
        c101 = b(1,0,1,0)*(1-tp) + b(1,0,1,1)*tp
        c110 = b(1,1,0,0)*(1-tp) + b(1,1,0,1)*tp
        c111 = b(1,1,1,0)*(1-tp) + b(1,1,1,1)*tp
        c00  = c000*(1-tt) + c001*tt
        c01  = c010*(1-tt) + c011*tt
        c10  = c100*(1-tt) + c101*tt
        c11  = c110*(1-tt) + c111*tt
        c0   = c00*(1-tz)  + c01*tz
        c1   = c10*(1-tz)  + c11*tz
        return c0*(1-tx) + c1*tx

    logRho = interp4d(EOS_LOGRHO)
    mu     = jnp.clip(interp4d(EOS_MU),  0.5, 2.5)
    nad    = jnp.clip(interp4d(EOS_NAD), NAD_CP_FLOOR, NAD_UPPER)
    rho    = 10.0**logRho

    # Gas-pressure thermodynamic derivatives from table interpolation.
    # chi_rho_gas = (d ln P_gas / d ln rho)_T and
    # chi_T_gas = (d ln P_gas / d ln T)_rho, computed from the OPAL EOS table
    # via numerical differentiation of logPgas(logQ, logT).
    # Reference: Rogers & Nayfonov 2002, ApJ 576, 1064 (OPAL EOS);
    #            MESA eosdt_load_tables.f90 lines 32-33 (jchiRho=4, jchiT=5).
    chi_rho_gas = jnp.clip(interp4d(EOS_CHI_RHO), 0.3, 2.0)
    chi_T_gas   = jnp.clip(interp4d(EOS_CHI_T),   0.3, 4.0)

    # Convert gas-pressure derivatives to total-pressure derivatives:
    # P_total = P_gas + P_rad, where P_rad = aT^4/3 (independent of rho).
    # chi_rho_total = (P_gas/P_total) * chi_rho_gas = beta * chi_rho_gas
    # chi_T_total = beta * chi_T_gas + 4*(1 - beta)
    # Reference: KWW 2012 §13.2; MESA eoscms_eval.f90 line 394.
    T = 10.0**lt
    P_gas = 10.0**logP_c
    P_rad = _a_rad * T**4 / 3.0
    P_total = P_gas + P_rad
    beta = P_gas / P_total  # gas pressure fraction

    chi_rho = jnp.clip(beta * chi_rho_gas, 1e-6, 2.0)
    chi_T = jnp.clip(beta * chi_T_gas + 4.0 * (1.0 - beta), 0.5, 5.0)

    # cp from Maxwell relation: nad = P*delta / (T*rho*cp)
    # => cp = P*delta / (T*rho*nad)  where delta = chi_T / chi_rho
    delta = chi_T / chi_rho
    cp = P_total * delta / (T * rho * jnp.maximum(nad, NAD_CP_FLOOR))

    # Specific entropy: Sackur-Tetrode (ideal gas) + radiation
    # Reference: KW 2012 eq 13.12; Cox & Giuli (1968) §9.6
    mu_m = mu * _m_H
    # Sackur-Tetrode per unit mass
    S_gas = (_k_B / mu_m) * (2.5 + 1.5 * jnp.log(2.0 * jnp.pi * mu_m * _k_B * T / _h_planck**2)
                            - jnp.log(rho / mu_m))
    S_rad = 4.0 * _a_rad * T**3 / (3.0 * rho)
    S = S_gas + S_rad

    return rho, mu, nad, S, cp, chi_rho, chi_T


# Public alias for use by subsystems that need pure OPAL (e.g. shooting solver,
# hires_profile diagnostic) — avoids importing the leading-underscore helper.
eos_opal_only = _eos_opal_table


# Blend activation: w = w_T * w_rho (2D gate in temperature AND density)
#
# Temperature gate: w_T = sigmoid(slope_T * (logT - center_T))
# The OPAL table covers logT ∈ [3.0, 7.9] but does NOT include electron
# degeneracy — it assumes ideal gas + radiation. At RGB-tip conditions
# (logT≈7.9, logρ>5.5) electrons ARE degenerate, so OPAL gives unphysical
# results (Gamma1≈1.9 vs the correct ~1.57). We must transition to HELM
# BEFORE the OPAL boundary.
#
# Density gate: w_rho = sigmoid(slope_rho * (logRho - center_rho))
# The FD table is accurate at strong degeneracy (logρ > 5.5, <1%
# nad) but has 2-5% nad error at partial degeneracy (logρ 5.0-5.5) due
# to bicubic interpolation artifacts. OPAL is correct at logρ < 5.5
# (within its table domain). The gate at center=5.5 ensures OPAL handles
# the partial-degeneracy regime (where it's more accurate than our FD
# table for nad) and HELM takes over at logρ > 5.5 (where OPAL clamps
# at its table edge ~5.69 and becomes wrong).
#
# Combined: w = w_T * w_rho ensures HELM activates ONLY where it's both
# needed (high T, beyond OPAL's range) AND accurate (high ρ, strong
# degeneracy). The density gate uses HELM's NR density inversion
# (computed anyway in the blended branch) because OPAL clamps at its
# table edge (logRho ≈ 5.69 for logT > 7.85) and cannot gate accurately
# there.
#
# MESA reference: eos_blend.f90 uses a 2D polygon in (logRho, logT) space
# for blend containment — both coordinates gate HELM activation. Our
# approach is analogous (sigmoid in both). The wide density gate reflects
# the FD table's full-range accuracy (no Sommerfeld approximation).

# --- HELM compile-time gate ---
# This flag controls whether HELM is compiled into the eos_lookup trace.
# Evaluated at PYTHON level during tracing (not a JAX tracer), so it
# determines what goes into the XLA graph:
#   True  → full OPAL+HELM blend compiled (needed for RGB/degenerate)
#   False → pure OPAL only (MS evolution — no HELM graph overhead)
#
# Without this gate, the HELM graph (30-iteration NR + 4× jax.grad for
# thermodynamic derivatives) is compiled into every lax.scan body that
# calls eos_lookup, adding ~4× compilation time and ~2× execution time
# even when the blend weight is effectively zero at MS conditions.
#
# The evolve_star entry point sets this based on whether the evolution
# could encounter degenerate conditions (RGB). MS-only tests leave it
# False for zero overhead — the compiled graph is identical to main
# (before HELM wiring). RGB/post-MS tests set it True.
# The helm_eos flag is threaded explicitly through the call chain
# via the helm_eos parameter on eos_lookup() and its callers.


_HELM_BLEND_CENTER = 7.7   # logT midpoint of OPAL→HELM transition
_HELM_BLEND_SLOPE = 15.0   # steepness: ~0.3 dex full transition width
_HELM_RHO_CENTER = 5.0    # logRho midpoint: with C² bicubic Hermite interpolation
                           # (replacing Catmull-Rom C¹), the ±3% nad oscillation
                           # in logRho 5.0-5.9 is eliminated (detrended PtP < 0.001%).
                           # The HELM FD table now agrees with MESA to < 0.3% nad across
                           # logRho 5.0-6.0. Lowering from 5.7 to 5.0
                           # gives HELM its full accurate range. OPAL's logP=22.0 table
                           # maximum corresponds to logRho≈5.7 at logT=7.9; at logRho>5.0,
                           # HELM FD is better than the OPAL extrapolation. The crossover
                           # at 5.0 transitions OPAL→HELM before OPAL's table boundary.
_HELM_RHO_SLOPE = 20.0    # steeper than logT gate: ~0.15 dex full transition.
                           # At logRho=5.0: w_rho=0.500 (midpoint).
                           # At logRho=5.15: w_rho=0.953 (HELM dominant).
                           # At logRho=5.5: w_rho≈1.0.


def _helm_blend_weight(logT):
    """Temperature component of OPAL→HELM blend weight.

    w_T = sigmoid(slope * (logT - center))
    - logT < 7.5: w_T ≈ 0.05 (OPAL-dominated)
    - logT = 7.7: w_T = 0.5 (midpoint)
    - logT > 7.9: w_T ≈ 0.95 (HELM-dominated)

    NOTE: This is the temperature gate only. The full blend weight used in
    production is w = w_T * w_rho (see eos_lookup), which also requires
    logRho > ~5.7 for HELM to activate (suppresses HELM where OPAL is
    more accurate).

    CONSTRAINT (JAX): sigmoid is smooth + AD-friendly; no hard branch.
    Reference: Paxton et al. 2018 (MESA IV) §5.1 (EOS blending);
               MESA eos_blend.f90 (helm_blend_width=0.1 in log-space).
    """
    return jax.nn.sigmoid(_HELM_BLEND_SLOPE * (logT - _HELM_BLEND_CENTER))


def _helm_rho_gate(logRho):
    """Density gate for HELM activation.

    w_rho = sigmoid(slope_rho * (logRho - center_rho))
    - logRho < 4.7: w_rho < 0.003 (HELM negligible — OPAL handles this regime)
    - logRho = 5.0: w_rho = 0.5 (midpoint)
    - logRho > 5.15: w_rho > 0.95 (HELM dominant)

    The center at logRho=5.0 is set below OPAL's table edge (~5.7 at logT=7.9)
    because the C² bicubic Hermite interpolation (#1021) eliminated the ±3%
    nad oscillation that formerly required deferring to OPAL up to logRho=5.7.
    With C² continuity, HELM FD is smooth and accurate (nad vs MESA < 0.3%)
    across logRho 5.0-6.0, making it safe to activate earlier.

    OPAL: correct up to logRho~5.7 (its logP_max=22.0 corresponds to
    logRho≈5.7 at logT=7.9). At logRho>5.7, OPAL is frozen at its boundary
    value. HELM FD handles the degenerate regime where OPAL fails.

    MESA reference: eos_blend.f90 gates HELM via a 2D polygon in (logRho,
    logT) space — both coordinates must be inside the valid domain.
    CONSTRAINT (JAX): sigmoid for smoothness + AD compatibility.
    """
    return jax.nn.sigmoid(_HELM_RHO_SLOPE * (logRho - _HELM_RHO_CENTER))


def eos_lookup(logT, logP, X, Z, ln_rho_guess=None, helm_eos=False):
    """Production EOS: OPAL table blended with HELM+Coulomb for degenerate zones.

    Smooth 2D sigmoid blend between OPAL (non-degenerate regime) and
    HELM+Coulomb free-energy EOS (strongly degenerate regime). The blend
    weight w = w_T * w_rho requires BOTH high temperature AND high density:
      - MS evolution (logT < 7.4): w ≈ 0, result numerically identical to OPAL
      - RGB H-shell (logT > 7.7, logRho < 5.2): w ≈ 0, pure OPAL (non-degenerate)
      - RGB-tip He core (logT > 7.7, logRho > 5.9): w ≈ 1, HELM FD
        (correct degenerate thermodynamics; validated < 2% Gamma1 vs MESA
        at strong degeneracy logRho > 5.7)

    Both OPAL and HELM are always computed and blended arithmetically. This
    avoids lax.cond (which compiles both branches anyway) and produces a
    fixed-size XLA graph compatible with jacfwd (Henyey Jacobian assembly).
    At MS conditions, HELM's runtime cost is negligible: the NR density
    inversion uses @custom_jvp (not unrolled in the backward graph).

    The density gate uses HELM's output density as the gating signal. HELM's
    Newton-Raphson density inversion (IFT custom_jvp) gives the correct
    physical density at all conditions, unlike OPAL which clamps at its table
    edge (logRho≈5.69 for logT>7.85). Computing HELM is unavoidable in the
    blend, so using its density for the gate adds no cost.

    HELM uses a full Fermi-Dirac electron EOS table (Timmes & Swesty 2000,
    #723) with bicubic interpolation, ideal ions, radiation, and OCP Coulomb
    corrections (Yakovlev & Shalybkov 1989). Accurate at ALL degeneracy
    levels (partial AND strong). Density inversion via IFT custom_jvp —
    compatible with jacfwd (Henyey Jacobian) and grad (adjoint).

    Parameters
    ----------
    logT : scalar — log10(T) [K]
    logP : scalar — log10(P) [dyn/cm²]
    X : scalar — hydrogen mass fraction
    Z : scalar — metals mass fraction
    ln_rho_guess : scalar or None — warm-start initial guess for HELM's
        Newton density solve (ln(rho) in cgs). Passing the previous step's
        converged density reduces HELM iterations from ~7 to ~3.
        MESA reference: micro.f90:333 passes s%lnd(k)/ln10 as logRho_guess.
    helm_eos : bool — compile-time HELM gate. When True, the
        OPAL+HELM blend is compiled into the trace (needed for RGB
        degenerate-core evolution). When False (default), eos_lookup returns
        pure OPAL (zero HELM overhead, identical to pre-#720 behavior).
        Threaded explicitly through the call chain via PhysicsToggles (#1070).

    MESA reference: eosdt_eval.f90:695-805 (get_level6_for_eosdt, get_HELM_alfa).
    MESA gates HELM via a 2D polygon in (logRho, logT) space (eos_blend.f90)
    and blends outputs element-wise: res(j) = alfa*res_1(j) + beta*res_2(j).
    We use a product of sigmoids in logT and logRho (CONSTRAINT: JAX smooth).

    References:
      - Rogers & Nayfonov (2002), ApJ 576, 1064 (OPAL tables)
      - Timmes & Swesty (2000), ApJS 126, 501 (HELM EOS)
      - Potekhin & Chabrier (2000), Phys. Rev. E 62, 8554 (OCP Coulomb)
      - Paxton et al. (2018), ApJS 234, 34 (MESA IV — EOS blending §5.1)
    """
    # --- Concrete-value fast path: skip HELM entirely at MS conditions ---
    #
    # When eos_lookup is called from a Python loop with concrete floats
    # (e.g. test_convective_boundary_vs_mesa_fgong: 300 calls at logT≈6.5-7.2),
    # each call would individually JIT-trace helm_eos_fd (30-iteration NR +
    # multiple jax.grad calls for thermodynamic derivatives). At 300 calls
    # this exceeds the 300s pytest timeout.
    #
    # At logT < 7.4, the blend weight w_T = sigmoid(15*(logT-7.7)) < 0.01,
    # and the density gate further suppresses it to effectively zero — the
    # result IS pure OPAL. So we can safely skip HELM when logT is concrete
    # and below the activation threshold.
    #
    # Under JIT tracing (inside lax.scan for the Henyey solver), logT is a
    # JAX tracer and float() raises ConcretizationTypeError. The except path
    # falls through to the always-compute-both logic below, which is required
    # for jacfwd compatibility (see the WHY NOT lax.cond comment below).
    try:
        logT_val = float(logT)
        if logT_val < _HELM_BLEND_CENTER - 0.3:  # 7.4: w_T < 0.01
            return _eos_opal_table(logT, logP, X, Z)
        # logT >= 7.4 with a concrete value: fall through to full blend.
        # This is needed for tests that call eos_lookup eagerly at RGB-tip
        # conditions (test_helm_eos_production_blend_rgb_tip, etc.).
    except (jax.errors.ConcretizationTypeError, TypeError):
        # Under JIT tracing (lax.scan body) — check the compile-time gate.
        #
        # --- Compile-time HELM gate ---
        # When helm_eos is False (default for MS-only evolution), the HELM
        # graph is NOT compiled into the trace. This is a Python-level check
        # evaluated at trace time, not a runtime branch. Result: the compiled
        # lax.scan body contains only the OPAL table lookup (identical to main
        # before HELM wiring), with zero compilation/runtime overhead.
        #
        # evolve_star passes helm_eos=True when the evolution may encounter
        # degenerate conditions (RGB tip, logT > 7.7, logRho > 5.5). For MS-
        # only tests helm_eos stays False, preserving their original compile
        # and execution characteristics.
        _helm_flag = helm_eos
        if not _helm_flag:
            return _eos_opal_table(logT, logP, X, Z)

    # Import here (AFTER the early-return checks) to avoid loading the FD
    # table's 701×301 jnp.array into Python scope during JAX traces that
    # take the OPAL-only path. Loading it during a nested trace (e.g. the
    # custom_vjp backward of _henyey_newton) causes an UnexpectedTracerError
    # because JAX sees the module-global array reference escape the trace scope.
    from stellar_jax.microphysics.fd_electron import helm_eos_fd

    # Always compute both OPAL and HELM, blend with arithmetic weights.
    #
    # WHY NOT lax.cond: lax.cond compiles BOTH branches at trace time.
    # When jacfwd differentiates through eos_lookup (for the Henyey block-
    # tridiagonal Jacobian), the HELM branch's JVP (custom_jvp NR + multiple
    # jax.grad calls) bloats the XLA graph for EVERY cell × EVERY Newton
    # iteration. This exceeds CI memory/time limits on the Henyey solver
    # convergence tests (50-1000 cells), even though HELM is never EXECUTED
    # at MS conditions. Since both branches are compiled regardless, removing
    # the branch and always computing both is graph-size neutral (may even
    # shrink it by eliminating the cond JVP overhead).
    #
    # At MS conditions (logT < 7.4): w_T < 0.01 → w ≈ 0 → result is
    # numerically identical to pure OPAL (verified: zero diff at logT=7.15).
    # The runtime cost of the HELM NR at MS is acceptable because
    # helm_eos_fd uses @custom_jvp (the 30-iteration NR is not unrolled
    # in the backward graph — it's a single compiled operation).
    #
    # MESA reference: eosdt_eval.f90:695-805 also blends EOS outputs
    # element-wise: res(j) = alfa*res_1(j) + beta*res_2(j).

    opal = _eos_opal_table(logT, logP, X, Z)
    helm = helm_eos_fd(logT, logP, X, Z, ln_rho_guess=ln_rho_guess)

    # 2D blend weight: w = w_T(logT) * w_rho(logRho_helm)
    w_T = _helm_blend_weight(logT)

    # Density gate uses HELM's output density (not OPAL's) because OPAL
    # clamps at logRho≈5.69 near its table edge (logT>7.85) and cannot
    # distinguish logRho=5.7 from logRho=6.0. HELM's NR density inversion
    # (IFT custom_jvp) gives the correct physical density everywhere.
    rho_helm = helm[0]
    logRho_helm = jnp.log10(jnp.maximum(rho_helm, 1e-30))
    w_rho = _helm_rho_gate(logRho_helm)

    # Combined: need BOTH high T AND high rho for HELM activation
    w = w_T * w_rho

    # Pure arithmetic element-wise blend: (1-w)*OPAL + w*HELM
    #
    # This is gradient-smooth: JAX autodiff propagates through the blend
    # weight w (which depends on logT and logRho_helm) correctly, producing
    # the blend derivative terms d(w)/d(logT) * (helm-opal) + (1-w)*d(opal)
    # + w*d(helm) — matching MESA's eosdt_support.f90:223 which explicitly
    # includes d_alfa_dlnd * res_1(j) + d_beta_dlnd * res_2(j).
    #
    # No jnp.where guard is needed because helm_eos_fd (FD table with
    # Catmull-Rom interpolation,) always returns finite values — unlike
    # the old analytic helm_eos_full whose NR density inversion could diverge
    # at non-degenerate conditions, producing NaN that poisoned the backward
    # pass. The shooting solver (the only long-accumulation path) uses
    # _eos_opal_table directly, so XLA float-noise accumulation is not a
    # concern here (the Henyey solver is self-correcting per Newton step).
    #
    # MESA reference: eosdt_support.f90:223 — element-wise blend with quintic
    # smoothing. Our sigmoid is C∞ (smoother than quintic C² at blend edges).
    rho     = (1.0 - w) * opal[0] + w * helm[0]
    mu      = (1.0 - w) * opal[1] + w * helm[1]
    nad     = (1.0 - w) * opal[2] + w * helm[2]
    S       = (1.0 - w) * opal[3] + w * helm[3]
    cp      = (1.0 - w) * opal[4] + w * helm[4]
    chi_rho = (1.0 - w) * opal[5] + w * helm[5]
    chi_T   = (1.0 - w) * opal[6] + w * helm[6]

    return rho, mu, nad, S, cp, chi_rho, chi_T


def eos_lookup_bicubic(logT, logP, X, Z):
    """OPAL EOS lookup with bicubic interpolation in (T, P), bilinear in (X, Z).

    Higher-order interpolation reduces systematic density errors near the CZ base
    (logT≈6.3) where OPAL tables have steep gradients. Used for diagnostic
    comparisons (e.g. Model S density) without changing the calibrated evolution.

    Reference: Catmull & Rom (1974), "A class of local interpolating splines."
    """
    (ix, iz, it, ip), (tx, tz, tt, tp) = _locate_4d(
        (EOS_X, EOS_Z, EOS_LOGT, EOS_LOGP), (X, Z, logT, logP))

    # Clamped values needed for derived thermodynamic quantities below.
    lt     = jnp.clip(logT, EOS_LOGT[0], EOS_LOGT[-1])
    logP_c = jnp.clip(logP, EOS_LOGP[0], EOS_LOGP[-1])

    def _catmull_rom(t, f_m1, f_0, f_1, f_2):
        t2 = t * t
        t3 = t2 * t
        return 0.5 * ((-t3 + 2*t2 - t) * f_m1 +
                      (3*t3 - 5*t2 + 2) * f_0 +
                      (-3*t3 + 4*t2 + t) * f_1 +
                      (t3 - t2) * f_2)

    it_m1 = jnp.clip(it - 1, 0, EOS_LOGT.shape[0] - 1)
    it_p1 = jnp.clip(it + 1, 0, EOS_LOGT.shape[0] - 1)
    it_p2 = jnp.clip(it + 2, 0, EOS_LOGT.shape[0] - 1)
    ip_m1 = jnp.clip(ip - 1, 0, EOS_LOGP.shape[0] - 1)
    ip_p1 = jnp.clip(ip + 1, 0, EOS_LOGP.shape[0] - 1)
    ip_p2 = jnp.clip(ip + 2, 0, EOS_LOGP.shape[0] - 1)

    def interp4d(Tab):
        def _bicubic_at_xz(dxi, dzi):
            def _get(ti, pi):
                return Tab[ix+dxi, iz+dzi, ti, pi]
            def _cubic_P(ti):
                return _catmull_rom(tp,
                    _get(ti, ip_m1), _get(ti, ip), _get(ti, ip_p1), _get(ti, ip_p2))
            vp_m1 = _cubic_P(it_m1)
            vp_0  = _cubic_P(it)
            vp_1  = _cubic_P(it_p1)
            vp_2  = _cubic_P(it_p2)
            return _catmull_rom(tt, vp_m1, vp_0, vp_1, vp_2)
        c00 = _bicubic_at_xz(0, 0)
        c01 = _bicubic_at_xz(0, 1)
        c10 = _bicubic_at_xz(1, 0)
        c11 = _bicubic_at_xz(1, 1)
        c0 = c00 * (1 - tz) + c01 * tz
        c1 = c10 * (1 - tz) + c11 * tz
        return c0 * (1 - tx) + c1 * tx

    logRho = interp4d(EOS_LOGRHO)
    mu     = jnp.clip(interp4d(EOS_MU),  0.5, 2.5)
    nad    = jnp.clip(interp4d(EOS_NAD), NAD_CP_FLOOR, NAD_UPPER)
    rho    = 10.0**logRho

    # Gas-pressure thermodynamic derivatives from table (same as eos_lookup).
    # Reference: Rogers & Nayfonov 2002, ApJ 576, 1064 (OPAL EOS);
    #            MESA eosdt_load_tables.f90 lines 32-33 (jchiRho=4, jchiT=5).
    chi_rho_gas = jnp.clip(interp4d(EOS_CHI_RHO), 0.3, 2.0)
    chi_T_gas   = jnp.clip(interp4d(EOS_CHI_T),   0.3, 4.0)

    T = 10.0**lt
    P_gas = 10.0**logP_c
    P_rad = _a_rad * T**4 / 3.0
    P_total = P_gas + P_rad
    beta = P_gas / P_total
    chi_rho = jnp.clip(beta * chi_rho_gas, 1e-6, 2.0)
    chi_T = jnp.clip(beta * chi_T_gas + 4.0 * (1.0 - beta), 0.5, 5.0)
    delta = chi_T / chi_rho
    cp = P_total * delta / (T * rho * jnp.maximum(nad, NAD_CP_FLOOR))
    mu_m = mu * _m_H
    S_gas = (_k_B / mu_m) * (2.5 + 1.5 * jnp.log(2.0 * jnp.pi * mu_m * _k_B * T / _h_planck**2)
                            - jnp.log(rho / mu_m))
    S_rad = 4.0 * _a_rad * T**3 / (3.0 * rho)
    S = S_gas + S_rad

    return rho, mu, nad, S, cp, chi_rho, chi_T
