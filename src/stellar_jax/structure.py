"""Stellar structure solver: shooting on a surface-concentrated radial mesh.

Core module providing the shooting-based structure integration, atmosphere
boundary conditions, Newton solver, and model construction. Widely imported
by solver/, evolution/, fgong/, calibration/, and the stellar.py facade.

The Henyey relaxation solver (solver/) has taken over the production
evolution loop, but this module remains the home of:
  - atmosphere_bc (T(τ) outer boundary, used by solver/surface_bc)
  - interp_X_at_mass (composition interpolation, used by solver/fgong/evolution)
  - shoot_xprofile / newton_solve_xprofile (shooting solver, used by ZAMS +
    calibration + diagnostics)
  - build_model_on_mesh (FGONG-ready mesh interpolation)
  - initial_guess (mass-dependent starting point)
  - sound_speed_profile (Model S comparison)

Method (Kippenhahn, Weigert & Weiss 2012, §11.1):
  1. Distribute N_MESH=600 shells on a quadratic-stretched radial grid:
     r_i = R - (R - r_inner) * (i/N)^2.  Step size |Δr| ∝ i/N, so the
     smallest steps are at the surface (i=0) and grow toward the center.
  2. Shoot inward with RK4 on this mesh.
  3. Newton-Raphson iteration on (logL, logTe) to match central BCs.

Resolution: outer 3% of R has ~104 points (i < N*sqrt(0.03)), vs ~9 for
the old uniform N_SHOOT=300 mesh.  This resolves the steep ∇ transition
in the SAL (Christensen-Dalsgaard 2008, ApSS 316, 13).

Known limitation: the shooting solver over-estimates the 1 M☉ ZAMS radius
by ~5-7% because the SAL stratification (∇_ad → ∇_rad transition over a
few H_p) remains under-resolved — the entropy it sets propagates into the
deep convective envelope and inflates the radius. The Henyey relaxation
solver (henyey.py) eliminates this by solving all mesh points simultaneously.
Higher-mass stars (≥1.5 M☉) with convective cores are unaffected.
"""
import functools

import jax
import jax.numpy as jnp
from jax import lax

from stellar_jax.config.constants import (
    G, a_rad, c_light, sigma_sb,
    Msun, Lsun,
)
from stellar_jax.config.mesh_defaults import (
    N_SHOOT, EPS_FD, N_COMP, COMP_MFRACS,
)
from stellar_jax.config.physics_floors import PGAS_FRAC_FLOOR
from stellar_jax.microphysics.eos import eos_lookup, eos_opal_only
from stellar_jax.microphysics.opacity import kappa
from stellar_jax.microphysics.nuclear import epsilon_nuclear
from stellar_jax.microphysics.neutrino import epsilon_neutrino
from stellar_jax.transport import mlt_nabla

# Number of shells for the structure mesh (up from 300 uniform)
N_MESH = 600


def initial_guess(M_solar):
    """Mass-dependent first guess for ZAMS (logL, logTe) Newton solve."""
    logM = jnp.log10(M_solar)
    return 4.5 * logM - 0.18, 3.752 + 0.65 * logM


def interp_X_at_mass(X_profile, m_frac, comp_mfracs=None):
    """Interpolate X from composition grid at mass fraction m_frac.

    Parameters
    ----------
    X_profile : (N_COMP,) — species profile on the composition grid
    m_frac : float — mass fraction coordinate to interpolate at
    comp_mfracs : (N_COMP,) or None — composition grid positions.
        If None, uses the static COMP_MFRACS (backward compatible).
        When the adaptive composition mesh (#405) is active, pass the
        adapted grid so interpolation uses correct coordinates.
    """
    m_frac = jnp.clip(m_frac, 0.0, 1.0)
    if comp_mfracs is None:
        comp_mfracs = jnp.array(COMP_MFRACS)
    return jnp.interp(m_frac, comp_mfracs, X_profile)


def _schwarz_blend(nabla_rad, nad, nabla_mlt, nabla_ks):
    """Schwarzschild switch in the atmosphere: plain jnp.where.

    Uses the standard jnp.where (KWW §6.2 — sharp Schwarzschild boundary).
    No custom_jvp here: the alpha_mlt adjoint for hot stars (M≥2 M☉) is
    provided by transport._mlt_switch in the interior MLT, which is where
    the physically meaningful alpha sensitivity lives. A sigmoid blend here
    in the atmosphere would inject phantom sensitivity for intermediate
    masses (M=1.5) where many atmosphere points are near the boundary,
    creating AD-vs-FD mismatch that compounds over 100 timesteps × 400 steps.
    """
    return jnp.where(nabla_rad > nad, nabla_mlt, nabla_ks)


def atmosphere_bc(Te, g_surf, X, Z, alpha_mlt, L_star, M_star, tau_base=100.0):
    """Integrate T(τ) + hydrostatic equilibrium from τ≈0 to τ_base.

    Returns (P, T) at τ_base.
    Krishna Swamy (1966) T(τ) relation, extended with MLT.

    Radiation-pressure boundary condition (Cox & Giuli 1968, §20.1; MESA
    atm/private/atm_t_tau_uniform.f90:eval_data lines 622-627): includes
    nonzero Prad at τ=0, giving P(τ) = τ·g/κ + g·L/(6π·c·G·M). The second
    term is the radiation pressure from the emergent flux at the surface.
    """
    N_PHASE1 = 20
    N_PHASE2 = 180

    tau_start = 1e-4
    tau_mid = 1.0

    ln_tau_start = jnp.log(tau_start)
    ln_tau_mid = jnp.log(tau_mid)
    ln_tau_end = jnp.log(tau_base)
    d_ln_tau_1 = (ln_tau_mid - ln_tau_start) / N_PHASE1
    d_ln_tau_2 = (ln_tau_end - ln_tau_mid) / N_PHASE2

    # Krishna Swamy (1966), ApJ 145, 174 — solar T(τ) relation q(τ):
    # q = 1.39 − 0.815 e^(−2.54τ) − 0.025 e^(−30τ) (his fit coefficients).
    q_init = 1.39 - 0.815 * jnp.exp(-2.54 * tau_start) - 0.025 * jnp.exp(-30.0 * tau_start)
    T_init = Te * (0.75 * (tau_start + q_init)) ** 0.25

    logT_init = jnp.log10(T_init)
    rho_guess = 1e-9
    log_kap_init = kappa(logT_init, jnp.log10(jnp.float64(rho_guess)), X, Z)
    kap_init = 10.0 ** log_kap_init
    # Radiation-pressure boundary condition: P(τ) = τ·g/κ + P_rad_surf
    # where P_rad_surf = g·L/(6π·c·G·M) — the nonzero radiation pressure
    # at τ=0 from the emergent flux (Cox & Giuli 1968, §20.1; MESA
    # atm/private/atm_t_tau_uniform.f90:622-627, Pextra_factor=1).
    P_hydro = tau_start * g_surf / kap_init
    P_rad_surf = g_surf * L_star / (6.0 * jnp.pi * c_light * G * M_star)
    P_init = jnp.maximum(P_hydro + P_rad_surf, 1.0)

    def _atm_step(carry, _, d_ln_tau):
        ln_P, ln_T, ln_tau = carry
        tau = jnp.exp(ln_tau)
        P = jnp.exp(ln_P)
        T = jnp.exp(ln_T)
        logT = jnp.log10(T)
        logP = jnp.log10(P)
        rho, mu_l, nad, *_ = eos_opal_only(logT, logP, X, Z)
        log_kap = kappa(logT, jnp.log10(rho), X, Z)
        kap = 10.0 ** log_kap

        dlnP_dlntau = tau * g_surf / (kap * P + 1e-30)
        nabla_rad = 3.0 * kap * L_star * P / \
                    (16.0 * jnp.pi * a_rad * c_light * G * M_star * T ** 4 + 1e-30)
        # Energy transport equation: dlnT/dlnτ = (dlnT/dlnP) * (dlnP/dlnτ)
        # mlt_nabla returns the actual gradient ∇ = dlnT/dlnP:
        #   - radiative (nabla_rad ≤ nad): returns nabla_rad
        #   - convective (nabla_rad > nad): returns nabla_mlt from MLT
        # This is the SAME equation the interior solver uses, ensuring
        # consistency at the atmosphere-interior boundary.
        # The KS T(τ) relation sets only the INITIAL condition (T_init at τ_start);
        # the integration follows the local energy transport, not KS.
        nabla = mlt_nabla(nabla_rad, nad, T, P, rho, kap, g_surf, mu_l, alpha_mlt)
        dlnT_dlntau = nabla * dlnP_dlntau

        ln_P_new = ln_P + dlnP_dlntau * d_ln_tau
        ln_T_new = ln_T + dlnT_dlntau * d_ln_tau
        ln_tau_new = ln_tau + d_ln_tau
        return (ln_P_new, ln_T_new, ln_tau_new), None

    step1 = lambda carry, x: _atm_step(carry, x, d_ln_tau_1)
    init = (jnp.log(P_init), jnp.log(T_init), ln_tau_start)
    (ln_P_mid, ln_T_mid, ln_tau_at_mid), _ = lax.scan(step1, init, None, length=N_PHASE1)

    step2 = lambda carry, x: _atm_step(carry, x, d_ln_tau_2)
    (ln_P_final, ln_T_final, _), _ = lax.scan(step2, (ln_P_mid, ln_T_mid, ln_tau_at_mid),
                                                None, length=N_PHASE2)

    P_base = jnp.maximum(jnp.exp(ln_P_final), 1.0)
    T_base = jnp.exp(ln_T_final)
    return P_base, T_base




def _radial_mesh(R_star):
    """Non-uniform radial mesh concentrated at the stellar surface.

    Grid: r_i = R_star - (R_star - r_inner) * (i/N)^p,  i = 0..N_MESH.
    With p=2 the step size |Δr| ∝ i/N grows linearly from surface to center,
    placing the densest sampling at the photosphere/SAL where H_p ~ 0.01*R.

    Points in outer 3%: the outer 3% of R corresponds to r > 0.97*R, i.e.
    (i/N)^2 < 0.03*(1 - r_inner_frac) ≈ 0.03, so i < N*sqrt(0.03) ≈ 104.
    (Christensen-Dalsgaard 2008, ApSS 316, 13.)

    **Mass mesh scheme (N_MESH = 600):**
    The production solver uses N_MESH=600 radial shells with quadratic stretching
    (p=2). This concentrates ~104 shells in the outer 3% of the radius, resolving
    the steep convective-to-radiative transition and the Krishna Swamy atmosphere.
    The inner 99.5% is covered by the remaining ~496 shells, providing adequate
    resolution of the core where H_p/R ~ 0.1.

    Richardson extrapolation (verified in tests/test_mesh.py::test_mesh_richardson):
    The effective convergence order of the BVP under quadratic refinement is p≈2,
    consistent with the RK4 integrator used on a non-uniform grid with stretching
    exponent 2 (LeVeque 2007, Finite Difference Methods, §6.1). At N=600 the
    relative error in P_c and T_c is < 0.1%, verified by the 3-level Richardson
    test at 300/600/1200 shells.

    Returns array of radii from R_star (surface) to r_inner (decreasing),
    shape (N_MESH+1,).
    """
    r_inner_frac = 0.005
    r_inner = r_inner_frac * R_star
    xi = jnp.linspace(0.0, 1.0, N_MESH + 1)
    p = 2.0  # quadratic: steps smallest at surface (xi=0), largest at center (xi=1)
    r_grid = R_star - (R_star - r_inner) * xi ** p
    return r_grid


def _radial_mesh_n(R_star, n_mesh):
    """Like _radial_mesh but with configurable resolution (for convergence tests)."""
    r_inner_frac = 0.005
    r_inner = r_inner_frac * R_star
    xi = jnp.linspace(0.0, 1.0, n_mesh + 1)
    r_grid = R_star - (R_star - r_inner) * xi ** 2.0
    return r_grid


def shoot_xprofile(M_solar, log_L, log_Te, X_profile, Z, t_age, alpha_mlt,
                   Z_profile=None):
    """Shoot inward on a non-uniform radial mesh (N_MESH=600, quadratic stretching).

    Integrates the 4 stellar structure ODEs (dP/dr, dM/dr, dL/dr, dT/dr) from
    the photosphere (r=R★) to the innermost shell (r=0.005*R★) using RK4.
    The shooting BVP: (logL, logTe) are tuned by newton_solve_xprofile until
    the central boundary conditions (M→0, L→0) are satisfied.

    Parameters
    ----------
    Z_profile : array or None
        If provided (shape N_COMP), per-zone metal mass fraction from
        gravitational settling. Used in EOS/opacity lookups at each mass shell
        for self-consistent structure with multi-element diffusion.
        Reference: Turcotte et al. (1998, ApJ 504, 539).

    Returns (residual, shell_data, final_state).
      residual: 2-vector [log(M_c/M★), log(L_c/L★)] — should be near 0.
      shell_data: (N_SHOOT, 11) array of per-shell diagnostics.
      final_state: [P_c, M_c, L_c, T_c] at the innermost shell.
    """
    M_star = M_solar * Msun
    L_star = 10.0 ** log_L * Lsun
    Te = 10.0 ** log_Te
    R_star = jnp.sqrt(L_star / (4.0 * jnp.pi * sigma_sb)) / Te ** 2
    g_surf = G * M_star / R_star ** 2

    X_surf = X_profile[N_COMP - 1]
    Z_surf = Z if Z_profile is None else Z_profile[N_COMP - 1]
    P_phot, T_phot = atmosphere_bc(Te, g_surf, X_surf, Z_surf, alpha_mlt, L_star, M_star)

    state0 = jnp.array([P_phot, M_star, L_star, T_phot])
    r_grid = _radial_mesh(R_star)

    # Per-zone Z: when Z_profile is provided, interpolate at each mass shell.
    # When None, use scalar Z directly (avoids unnecessary jnp.interp in the
    # trace, keeping the XLA graph size identical to the pre- code).
    _use_Z_profile = Z_profile is not None
    if _use_Z_profile:
        _Z_prof = Z_profile
        def _interp_Z(m_frac):
            return interp_X_at_mass(_Z_prof, m_frac)
    else:
        def _interp_Z(m_frac):
            return Z

    # Use OPAL-only EOS for the shooting solver: it operates exclusively at
    # ZAMS/MS conditions (logT < 7.5) where the HELM blend weight is < 1e-60.
    # Using eos_opal_only directly avoids tracing the full HELM computation
    # (30-iteration NR + multiple jax.grad) into the 600-step lax.scan body,
    # which would bloat the XLA graph beyond compile-time/memory limits for the
    # backward pass (jax.jacobian in the IFT custom_vjp).
    # The production Henyey solver uses the full eos_lookup (with HELM blend)
    # for RGB evolution where degenerate conditions are encountered.
    _eos_fn = eos_opal_only

    def derivs_with_eps(r, st):
        P, Mr, Lr, T = st
        P = jnp.maximum(P, 1.0)
        T = jnp.maximum(T, 1.0e3)
        r = jnp.maximum(r, 1e-4 * R_star)
        Mr = jnp.maximum(Mr, 1e-10 * M_star)
        Lr = jnp.maximum(Lr, 1e-10 * L_star)
        m_frac = Mr / M_star
        X_l = interp_X_at_mass(X_profile, m_frac)
        Z_l = _interp_Z(m_frac)
        Prad = a_rad * T ** 4 / 3.0
        Pgas = jnp.maximum(P - Prad, PGAS_FRAC_FLOOR * P)
        logT = jnp.log10(T)
        rho, mu_l, nad, *_ = _eos_fn(logT, jnp.log10(Pgas), X_l, Z_l)
        log_kap = kappa(logT, jnp.log10(rho), X_l, Z_l)
        kap = 10.0 ** log_kap
        eps = epsilon_nuclear(rho, T, X_l, Z_l, t_age)
        eps_nu = epsilon_neutrino(rho, T, X_l, Z_l)
        g_local = G * Mr / r ** 2
        dPdr = -rho * g_local
        dMdr = 4.0 * jnp.pi * rho * r ** 2
        dLdr = 4.0 * jnp.pi * rho * (eps - eps_nu) * r ** 2
        nabla_rad = 3.0 * kap * Lr * P / (16.0 * jnp.pi * a_rad * c_light * G * Mr * T ** 4 + 1e-30)
        nabla = mlt_nabla(nabla_rad, nad, T, P, rho, kap, g_local, mu_l, alpha_mlt)
        dTdr = (T / P) * dPdr * nabla
        H_p = P / (rho * g_local + 1e-30)
        H_p_over_R = H_p / R_star
        dm_dr_norm = dMdr * R_star / M_star
        return jnp.array([dPdr, dMdr, dLdr, dTdr]), eps, m_frac, nabla_rad, nad, H_p_over_R, dm_dr_norm, jnp.log10(T), jnp.log10(rho), nabla, jnp.log10(P)

    def rk4_step(carry, i):
        st, _ = carry
        r = r_grid[i]
        r_next = r_grid[i + 1]
        dr = r_next - r  # negative (inward)

        k1, eps1, mf1, nrad1, nad1, hp1, dmdr1, logT1, logrho1, nabla1, logP1 = derivs_with_eps(r, st)
        k2, _, _, _, _, _, _, _, _, _, _ = derivs_with_eps(r + 0.5 * dr, st + 0.5 * dr * k1)
        k3, _, _, _, _, _, _, _, _, _, _ = derivs_with_eps(r + 0.5 * dr, st + 0.5 * dr * k2)
        k4, _, _, _, _, _, _, _, _, _, _ = derivs_with_eps(r + dr, st + dr * k3)
        st_new = st + (dr / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        st_new = st_new.at[0].set(jnp.maximum(st_new[0], 1.0))
        st_new = st_new.at[3].set(jnp.maximum(st_new[3], 1.0e3))
        return (st_new, r_next), jnp.array([eps1, mf1, nrad1, nad1, hp1, dmdr1, logT1, logrho1, nabla1, logP1])

    (final, _), shell_data = lax.scan(rk4_step, (state0, r_grid[0]), jnp.arange(N_MESH))
    residual = jnp.array([final[1] / M_star, final[2] / L_star])

    # Subsample to N_SHOOT points, uniformly spaced in radius (not index).
    # From r_i = R - (R-r_inner)*(i/N)^2, uniform-in-r targets correspond to
    # i = N * sqrt(j/(N_SHOOT-1)) for j=0..N_SHOOT-1.
    j = jnp.arange(N_SHOOT)
    sub_idx = (N_MESH * jnp.sqrt(j / (N_SHOOT - 1))).astype(jnp.int32)
    sub_idx = jnp.clip(sub_idx, 0, N_MESH - 1)
    shell_data_sub = shell_data[sub_idx]

    return residual, shell_data_sub, final


def shoot_xprofile_residual(M_solar, log_L, log_Te, X_profile, Z, t_age, alpha_mlt):
    """Return only residual (for Newton-Raphson)."""
    res, _, _ = shoot_xprofile(M_solar, log_L, log_Te, X_profile, Z, t_age, alpha_mlt)
    return res


def newton_solve_with_zprofile(M_solar, X_profile, Z, t_age, logL, logTe, alpha_mlt,
                               Z_profile, n_iter=2):
    """Correct (logL, logTe) for the Z_profile perturbation via linear correction.

    Uses the scalar-Z Jacobian (already available from newton_solve_xprofile) to
    apply a first-order correction for the Z gradient's effect on the shooting
    residual. This avoids retracing shoot_xprofile with Z_profile (expensive JIT)
    while being accurate for the small Z perturbation (ΔZ/Z ~ 2-5%).

    The correction is: Δ(logL, logTe) = -J_scalar⁻¹ × [R(Z_profile) - R(scalar_Z)]
    iterated n_iter times for convergence.

    Reference: Turcotte et al. (1998, ApJ 504, 539) — Z gradient perturbation.
    """
    eps_fd = 1e-5
    lL, lT = float(logL), float(logTe)
    for _ in range(n_iter):
        lL_j = jnp.float64(lL)
        lT_j = jnp.float64(lT)
        # Residual with Z_profile
        R_zp, _, _ = shoot_xprofile(M_solar, lL_j, lT_j, X_profile, Z, t_age,
                                    alpha_mlt, Z_profile=Z_profile)
        # Jacobian from scalar-Z (fast, already traced)
        R0, _, _ = shoot_xprofile(M_solar, lL_j, lT_j, X_profile, Z, t_age,
                                  alpha_mlt)
        R_dL, _, _ = shoot_xprofile(M_solar, lL_j + eps_fd, lT_j, X_profile, Z,
                                    t_age, alpha_mlt)
        R_dT, _, _ = shoot_xprofile(M_solar, lL_j, lT_j + eps_fd, X_profile, Z,
                                    t_age, alpha_mlt)
        Jmat = jnp.column_stack([(R_dL - R0) / eps_fd, (R_dT - R0) / eps_fd])
        Jmat = Jmat + 1e-8 * jnp.eye(2)
        # Solve for correction: make R_zp → 0
        dx = jnp.linalg.solve(Jmat, -R_zp)
        dx = jnp.clip(dx, -0.05, 0.05)
        lL = float(jnp.clip(lL + dx[0], -3.0, 5.0))
        lT = float(jnp.clip(lT + dx[1], 3.5, 4.2))
    return jnp.float64(lL), jnp.float64(lT)


def shoot_at_resolution(M_solar, log_L, log_Te, X_profile, Z, t_age, alpha_mlt, n_mesh):
    """Shoot at arbitrary resolution (for convergence tests).

    Returns final state [P_center, M_center, L_center, T_center].
    n_mesh must be a Python int (static for JIT).
    """
    M_star = M_solar * Msun
    L_star = 10.0 ** log_L * Lsun
    Te = 10.0 ** log_Te
    R_star = jnp.sqrt(L_star / (4.0 * jnp.pi * sigma_sb)) / Te ** 2
    g_surf = G * M_star / R_star ** 2
    X_surf = X_profile[N_COMP - 1]
    P_phot, T_phot = atmosphere_bc(Te, g_surf, X_surf, Z, alpha_mlt, L_star, M_star)
    state0 = jnp.array([P_phot, M_star, L_star, T_phot])
    r_grid = _radial_mesh_n(R_star, n_mesh)

    def derivs(r, st):
        P, Mr, Lr, T = st
        P = jnp.maximum(P, 1.0)
        T = jnp.maximum(T, 1.0e3)
        r = jnp.maximum(r, 1e-4 * R_star)
        Mr = jnp.maximum(Mr, 1e-10 * M_star)
        Lr = jnp.maximum(Lr, 1e-10 * L_star)
        m_frac = Mr / M_star
        X_l = interp_X_at_mass(X_profile, m_frac)
        Prad = a_rad * T ** 4 / 3.0
        Pgas = jnp.maximum(P - Prad, PGAS_FRAC_FLOOR * P)
        logT = jnp.log10(T)
        rho, mu_l, nad, *_ = eos_opal_only(logT, jnp.log10(Pgas), X_l, Z)
        log_kap = kappa(logT, jnp.log10(rho), X_l, Z)
        kap = 10.0 ** log_kap
        eps = epsilon_nuclear(rho, T, X_l, Z, t_age)
        eps_nu = epsilon_neutrino(rho, T, X_l, Z)
        g_local = G * Mr / r ** 2
        dPdr = -rho * g_local
        dMdr = 4.0 * jnp.pi * rho * r ** 2
        dLdr = 4.0 * jnp.pi * rho * (eps - eps_nu) * r ** 2
        nabla_rad = 3.0 * kap * Lr * P / (16.0 * jnp.pi * a_rad * c_light * G * Mr * T ** 4 + 1e-30)
        nabla = mlt_nabla(nabla_rad, nad, T, P, rho, kap, g_local, mu_l, alpha_mlt)
        dTdr = (T / P) * dPdr * nabla
        return jnp.array([dPdr, dMdr, dLdr, dTdr])

    def rk4_step(carry, i):
        st, _ = carry
        r = r_grid[i]
        r_next = r_grid[i + 1]
        dr = r_next - r
        k1 = derivs(r, st)
        k2 = derivs(r + 0.5 * dr, st + 0.5 * dr * k1)
        k3 = derivs(r + 0.5 * dr, st + 0.5 * dr * k2)
        k4 = derivs(r + dr, st + dr * k3)
        st_new = st + (dr / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        st_new = st_new.at[0].set(jnp.maximum(st_new[0], 1.0))
        st_new = st_new.at[3].set(jnp.maximum(st_new[3], 1.0e3))
        return (st_new, r_next), None

    (final, _), _ = lax.scan(rk4_step, (state0, r_grid[0]), jnp.arange(n_mesh))
    return final


def newton_solve_at_resolution(M_solar, X_profile, Z, t_age, logL, logTe, alpha_mlt, n_iter, n_mesh):
    """Newton-Raphson on (logL, logTe) using a specified mesh resolution.

    Like newton_solve_xprofile but shoots at n_mesh shells instead of the
    default N_MESH=600. Used for Richardson convergence tests where the BVP
    must be re-converged independently at each resolution (not reusing BCs
    from the production mesh). Not in the gradient path (no custom_vjp).
    """
    M_star = M_solar * Msun

    def residual_at_n(lL, lT):
        final = shoot_at_resolution(M_solar, lL, lT, X_profile, Z, t_age, alpha_mlt, n_mesh)
        P_c, M_c, L_c, T_c = final
        L_star = 10.0 ** lL * Lsun
        return jnp.array([M_c / M_star, L_c / L_star])

    def step(state, _):
        lL, lT = state
        R0 = residual_at_n(lL, lT)
        R_dL = residual_at_n(lL + EPS_FD, lT)
        R_dT = residual_at_n(lL, lT + EPS_FD)
        Jmat = jnp.column_stack([(R_dL - R0) / EPS_FD, (R_dT - R0) / EPS_FD]) + 1e-8 * jnp.eye(2)
        dx = jnp.linalg.solve(Jmat, -R0)
        dx = jnp.clip(dx, -0.05, 0.05)
        return (jnp.clip(lL + dx[0], -3.0, 5.0),
                jnp.clip(lT + dx[1], 3.5, 4.2)), None

    (lL_out, lT_out), _ = lax.scan(step, (logL, logTe), None, length=n_iter)
    return lL_out, lT_out


@functools.partial(jax.custom_vjp, nondiff_argnums=(7,))
def newton_solve_xprofile(M_solar, X_profile, Z, t_age, logL, logTe, alpha_mlt, n_iter):
    """Newton-Raphson on (logL, logTe) with fixed X_profile.  [FROZEN — F11 #152]"""
    def step(state, _):
        lL, lT = state
        R0 = shoot_xprofile_residual(M_solar, lL, lT, X_profile, Z, t_age, alpha_mlt)
        R_dL = shoot_xprofile_residual(M_solar, lL + EPS_FD, lT, X_profile, Z, t_age, alpha_mlt)
        R_dT = shoot_xprofile_residual(M_solar, lL, lT + EPS_FD, X_profile, Z, t_age, alpha_mlt)
        Jmat = jnp.column_stack([(R_dL - R0) / EPS_FD, (R_dT - R0) / EPS_FD]) + 1e-8 * jnp.eye(2)
        dx = jnp.linalg.solve(Jmat, -R0)
        dx = jnp.clip(dx, -0.05, 0.05)
        return (jnp.clip(lL + dx[0], -3.0, 5.0),
                jnp.clip(lT + dx[1], 3.5, 4.2)), None

    (lL_out, lT_out), _ = lax.scan(step, (logL, logTe), None, length=n_iter)
    return lL_out, lT_out


def _newton_solve_xprofile_fwd(M_solar, X_profile, Z, t_age, logL, logTe, alpha_mlt, n_iter):
    lL, lT = newton_solve_xprofile(M_solar, X_profile, Z, t_age, logL, logTe, alpha_mlt, n_iter)
    return (lL, lT), (M_solar, X_profile, Z, t_age, alpha_mlt, lL, lT)


def _newton_solve_xprofile_bwd(n_iter, res, g):
    """IFT backward pass: d(logL*,logTe*)/d(params) = -Jx^{-1} @ Jp.

    Collapses the backward pass to a single 2x2 linear solve at the converged
    fixed point — no unrolling through Newton iterations.

    Uses AD (jax.jacobian) to compute Jx exactly, eliminating the O(EPS_FD)
    truncation error that caused ~0.02%/step systematic gradient bias,
    accumulating to ~10% at N=500 (issue #158).

    Reference: Griewank & Walther (2008) §15; Blondel et al. (2022).
    """
    M_solar, X_profile, Z, t_age, alpha_mlt, lL, lT = res
    g_lL, g_lT = g

    # Jx: 2x2 Jacobian of residual w.r.t. (logL, logTe) at converged point.
    # Computed via AD (exact) instead of FD (O(EPS_FD) error per step).
    def residual_x(lL_lT):
        return shoot_xprofile_residual(M_solar, lL_lT[0], lL_lT[1], X_profile, Z, t_age, alpha_mlt)

    Jx = jax.jacobian(residual_x)(jnp.array([lL, lT]))

    # Adjoint: lambda = Jx^{-T} @ g
    lam = jnp.linalg.solve(Jx.T, jnp.array([g_lL, g_lT]))

    # Jp via jax.vjp of residual w.r.t. differentiable params, contracted with -lambda
    def residual_fn(M, X, Z_, ta, alp):
        return shoot_xprofile_residual(M, lL, lT, X, Z_, ta, alp)

    _, vjp_fn = jax.vjp(residual_fn, M_solar, X_profile, Z, t_age, alpha_mlt)
    g_M, g_X, g_Z, g_ta, g_alp = vjp_fn(-lam)

    # Return gradients for all 7 differentiable args (n_iter is nondiff)
    # Order: M_solar, X_profile, Z, t_age, logL, logTe, alpha_mlt
    return g_M, g_X, g_Z, g_ta, jnp.zeros_like(lL), jnp.zeros_like(lT), g_alp


newton_solve_xprofile.defvjp(_newton_solve_xprofile_fwd, _newton_solve_xprofile_bwd)


def get_structure_profile(M_solar, log_L, log_Te, X_profile, Z, t_age, alpha_mlt,
                          Z_profile=None):
    """Integrate converged structure on non-uniform radial mesh.

    Returns profile: (N_SHOOT, 5) = [r/R, P, T, rho, nabla_ad]

    If Z_profile is provided (shape N_COMP), the local Z at each mass shell is
    interpolated from this array (for models with gravitational settling of metals).
    """
    M_star = M_solar * Msun
    L_star = 10.0 ** log_L * Lsun
    Te = 10.0 ** log_Te
    R_star = jnp.sqrt(L_star / (4.0 * jnp.pi * sigma_sb)) / Te ** 2
    g_surf = G * M_star / R_star ** 2

    X_surf = X_profile[N_COMP - 1]
    Z_surf = Z if Z_profile is None else Z_profile[N_COMP - 1]
    P_phot, T_phot = atmosphere_bc(Te, g_surf, X_surf, Z_surf, alpha_mlt, L_star, M_star)
    state0 = jnp.array([P_phot, M_star, L_star, T_phot])
    r_grid = _radial_mesh(R_star)

    # If Z_profile provided, interpolate local Z at mass fraction; else use scalar
    _use_Z_profile = Z_profile is not None
    if _use_Z_profile:
        _Z_prof = Z_profile
        def _interp_Z(m_frac):
            return interp_X_at_mass(_Z_prof, m_frac)
    else:
        def _interp_Z(m_frac):
            return Z

    _eos_fn = eos_lookup

    def rk4_step(carry, i):
        st, _ = carry
        P, Mr, Lr, T = st
        r = r_grid[i]
        r_next = r_grid[i + 1]
        dr = r_next - r

        P = jnp.maximum(P, 1.0)
        T = jnp.maximum(T, 1.0e3)
        r_c = jnp.maximum(r, 1e-4 * R_star)
        Mr = jnp.maximum(Mr, 1e-10 * M_star)
        m_frac = Mr / M_star
        X_l = interp_X_at_mass(X_profile, m_frac)
        Z_l = _interp_Z(m_frac)
        Prad = a_rad * T ** 4 / 3.0
        Pgas = jnp.maximum(P - Prad, PGAS_FRAC_FLOOR * P)
        logT = jnp.log10(T)
        rho, mu_l, nad, *_ = _eos_fn(logT, jnp.log10(Pgas), X_l, Z_l)
        log_kap = kappa(logT, jnp.log10(rho), X_l, Z_l)
        kap = 10.0 ** log_kap
        eps = epsilon_nuclear(rho, T, X_l, Z_l, t_age)
        eps_nu = epsilon_neutrino(rho, T, X_l, Z_l)
        g_local = G * Mr / r_c ** 2

        dPdr = -rho * g_local
        dMdr = 4.0 * jnp.pi * rho * r_c ** 2
        dLdr = 4.0 * jnp.pi * rho * (eps - eps_nu) * r_c ** 2
        nabla_rad = 3.0 * kap * Lr * P / (16.0 * jnp.pi * a_rad * c_light * G * Mr * T ** 4 + 1e-30)
        nabla = mlt_nabla(nabla_rad, nad, T, P, rho, kap, g_local, mu_l, alpha_mlt)
        dTdr = (T / P) * dPdr * nabla

        k1 = jnp.array([dPdr, dMdr, dLdr, dTdr])

        def _derivs(r_, st_):
            P_, Mr_, Lr_, T_ = st_
            P_ = jnp.maximum(P_, 1.0)
            T_ = jnp.maximum(T_, 1.0e3)
            r_ = jnp.maximum(r_, 1e-4 * R_star)
            Mr_ = jnp.maximum(Mr_, 1e-10 * M_star)
            mf_ = Mr_ / M_star
            X_l_ = interp_X_at_mass(X_profile, mf_)
            Z_l_ = _interp_Z(mf_)
            Prad_ = a_rad * T_ ** 4 / 3.0
            Pgas_ = jnp.maximum(P_ - Prad_, PGAS_FRAC_FLOOR * P_)
            rho_, mu_, nad_, *_ = _eos_fn(jnp.log10(T_), jnp.log10(Pgas_), X_l_, Z_l_)
            kap_ = 10.0 ** kappa(jnp.log10(T_), jnp.log10(rho_), X_l_, Z_l_)
            eps_ = epsilon_nuclear(rho_, T_, X_l_, Z_l_, t_age)
            enu_ = epsilon_neutrino(rho_, T_, X_l_, Z_l_)
            g_ = G * Mr_ / r_ ** 2
            dP_ = -rho_ * g_
            dM_ = 4.0 * jnp.pi * rho_ * r_ ** 2
            dL_ = 4.0 * jnp.pi * rho_ * (eps_ - enu_) * r_ ** 2
            nr_ = 3.0 * kap_ * Lr_ * P_ / (16.0 * jnp.pi * a_rad * c_light * G * Mr_ * T_ ** 4 + 1e-30)
            n_ = mlt_nabla(nr_, nad_, T_, P_, rho_, kap_, g_, mu_, alpha_mlt)
            dT_ = (T_ / P_) * dP_ * n_
            return jnp.array([dP_, dM_, dL_, dT_])

        k2 = _derivs(r_c + 0.5 * dr, st + 0.5 * dr * k1)
        k3 = _derivs(r_c + 0.5 * dr, st + 0.5 * dr * k2)
        k4 = _derivs(r_c + dr, st + dr * k3)
        st_new = st + (dr / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        st_new = st_new.at[0].set(jnp.maximum(st_new[0], 1.0))
        st_new = st_new.at[3].set(jnp.maximum(st_new[3], 1.0e3))
        return (st_new, r_next), jnp.array([r_c / R_star, P, T, rho, nad])

    _, profile = lax.scan(rk4_step, (state0, r_grid[0]), jnp.arange(N_MESH))

    # Subsample to N_SHOOT points, uniformly spaced in radius (not index).
    # From r_i = R - (R-r_inner)*(i/N)^2: uniform-in-r → i = N*sqrt(j/(N_SHOOT-1)).
    j = jnp.arange(N_SHOOT)
    sub_idx = (N_MESH * jnp.sqrt(j / (N_SHOOT - 1))).astype(jnp.int32)
    sub_idx = jnp.clip(sub_idx, 0, N_MESH - 1)
    return profile[sub_idx]


def sound_speed_profile(M_solar, log_L, log_Te, X_profile, Z, t_age, alpha_mlt,
                        Z_profile=None):
    """Compute c_s(r/R) for a converged stellar model."""
    profile = get_structure_profile(M_solar, log_L, log_Te, X_profile, Z, t_age, alpha_mlt,
                                    Z_profile=Z_profile)
    r_over_R = profile[:, 0]
    P = profile[:, 1]
    rho = profile[:, 3]
    T = profile[:, 2]
    P_rad = a_rad * T ** 4 / 3.0
    P_gas = jnp.maximum(P - P_rad, PGAS_FRAC_FLOOR * P)
    beta = P_gas / P
    # Eddington Γ₁ for an ideal gas + radiation mixture, β = P_gas/P_total
    # (Chandrasekhar 1939, §56; Cox & Giuli 1968, §9.17). Coefficients are exact
    # (analytic thermodynamics), not fitted.
    gamma1 = (32.0 - 24.0 * beta - 3.0 * beta ** 2) / (24.0 - 21.0 * beta)
    c_s = jnp.sqrt(gamma1 * P / rho)
    return r_over_R, c_s




def _interpolate_profile_to_mesh(profile, q_mesh):
    """Interpolate a surface→center radial profile onto the Lagrangian mass mesh.

    Parameters
    ----------
    profile : (N_MESH, 6) array
        Columns: [r, m_frac, logP, logT, logrho, L], surface→center ordering.
    q_mesh : (N_mesh+1,) array
        Target Lagrangian mass coordinates.

    Returns
    -------
    dict with 'q', 'logP', 'logT', 'logrho', 'r', 'L' on the mesh.
    """
    # Reverse to center→surface for monotonic interpolation
    m_frac_arr = profile[:, 1][::-1]
    logP_arr = profile[:, 2][::-1]
    logT_arr = profile[:, 3][::-1]
    logrho_arr = profile[:, 4][::-1]
    r_arr = profile[:, 0][::-1]
    L_arr = profile[:, 5][::-1]

    return {
        'q': q_mesh,
        'logP': jnp.interp(q_mesh, m_frac_arr, logP_arr),
        'logT': jnp.interp(q_mesh, m_frac_arr, logT_arr),
        'logrho': jnp.interp(q_mesh, m_frac_arr, logrho_arr),
        'r': jnp.interp(q_mesh, m_frac_arr, r_arr),
        'L': jnp.interp(q_mesh, m_frac_arr, L_arr),
    }


def build_model_on_mesh(M_solar, logL, logTe, X_profile, Z, alpha_mlt, N_mesh, q_mesh_in=None, t_age=None):
    """Build a stellar model on the Lagrangian mass mesh from given (logL, logTe).

    Integrates structure equations surface→center on the standard N_MESH=600
    radial grid, then interpolates onto the Lagrangian mass mesh.

    Parameters
    ----------
    M_solar : float or JAX scalar
        Stellar mass in solar masses.
    logL : float or JAX scalar
        log10(L/L_sun).
    logTe : float or JAX scalar
        log10(T_eff / K).
    X_profile : (N_COMP,) array
        Hydrogen mass fraction profile.
    Z : float or JAX scalar
        Metal mass fraction.
    alpha_mlt : float or JAX scalar
        Mixing-length parameter.
    N_mesh : int
        Number of Lagrangian mass zones.

    Returns
    -------
    dict with keys:
        'q' : (N_mesh+1,) mass coordinate array
        'logP', 'logT', 'logrho' : (N_mesh+1,) thermodynamic state
        'r' : (N_mesh+1,) radius in cm
        'L' : (N_mesh+1,) luminosity in erg/s
    """
    M_star = jnp.asarray(M_solar, dtype=jnp.float64) * Msun
    L_star = 10.0**logL * Lsun
    Te = 10.0**logTe
    R_star = jnp.sqrt(L_star / (4.0 * jnp.pi * sigma_sb)) / Te**2
    g_surf = G * M_star / R_star**2
    X_surf = X_profile[N_COMP - 1]
    Z_j = jnp.asarray(Z, dtype=jnp.float64)
    alpha_j = jnp.asarray(alpha_mlt, dtype=jnp.float64)
    P_phot, T_phot = atmosphere_bc(Te, g_surf, X_surf, Z_j, alpha_j, L_star, M_star)

    state0 = jnp.array([P_phot, M_star, L_star, T_phot])
    r_grid = _radial_mesh(R_star)
    _t_age_j = jnp.float64(0.0) if t_age is None else jnp.asarray(t_age, dtype=jnp.float64)

    def _derivs(r, st):
        P, Mr, Lr, T = st
        P = jnp.maximum(P, 1.0)
        T = jnp.maximum(T, 1.0e3)
        r = jnp.maximum(r, 1e-4 * R_star)
        Mr = jnp.maximum(Mr, 1e-10 * M_star)
        Lr = jnp.maximum(Lr, 1e-10 * L_star)
        m_frac = Mr / M_star
        X_l = interp_X_at_mass(X_profile, m_frac)
        Prad = a_rad * T**4 / 3.0
        Pgas = jnp.maximum(P - Prad, PGAS_FRAC_FLOOR * P)
        logT_ = jnp.log10(T)
        rho, mu_l, nad, *_ = eos_lookup(logT_, jnp.log10(Pgas), X_l, Z_j)
        log_kap = kappa(logT_, jnp.log10(rho), X_l, Z_j)
        kap = 10.0**log_kap
        eps = epsilon_nuclear(rho, T, X_l, Z_j, _t_age_j)
        eps_nu = epsilon_neutrino(rho, T, X_l, Z_j)
        g_local = G * Mr / r**2
        dPdr = -rho * g_local
        dMdr = 4.0 * jnp.pi * rho * r**2
        dLdr = 4.0 * jnp.pi * rho * (eps - eps_nu) * r**2
        nabla_rad = 3.0 * kap * Lr * P / (
            16.0 * jnp.pi * a_rad * c_light * G * Mr * T**4 + 1e-30)
        nabla = mlt_nabla(nabla_rad, nad, T, P, rho, kap, g_local, mu_l, alpha_j)
        dTdr = (T / P) * dPdr * nabla
        return jnp.array([dPdr, dMdr, dLdr, dTdr]), rho

    def _rk4_step(carry, i):
        st, _ = carry
        r = r_grid[i]
        r_next = r_grid[i + 1]
        dr = r_next - r
        k1, _ = _derivs(r, st)
        k2, _ = _derivs(r + 0.5 * dr, st + 0.5 * dr * k1)
        k3, _ = _derivs(r + 0.5 * dr, st + 0.5 * dr * k2)
        k4, _ = _derivs(r + dr, st + dr * k3)
        st_new = st + (dr / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        st_new = st_new.at[0].set(jnp.maximum(st_new[0], 1.0))
        st_new = st_new.at[3].set(jnp.maximum(st_new[3], 1.0e3))
        _, rho_converged = _derivs(r_next, st_new)
        P_rec = st_new[0]
        T_rec = st_new[3]
        Mr_rec = st_new[1]
        Lr_rec = st_new[2]
        return (st_new, r_next), jnp.array([
            r_next, Mr_rec / M_star, jnp.log10(P_rec), jnp.log10(T_rec),
            jnp.log10(rho_converged), Lr_rec
        ])

    (_, _), profile = lax.scan(
        _rk4_step, (state0, r_grid[0]), jnp.arange(N_MESH)
    )

    if q_mesh_in is not None:
        q_mesh = q_mesh_in
    else:
        from stellar_jax.mesh import initial_lagrangian_mesh
        q_mesh = initial_lagrangian_mesh(N_mesh)
    return _interpolate_profile_to_mesh(profile, q_mesh)


def zams_initial_model(M_solar, Z=0.014, alpha_mlt=1.9, N_mesh=1000):
    """Build a ZAMS model on the Lagrangian mass mesh (issue #95).

    Approach (Kippenhahn, Weigert & Weiss 2012, §22.1):
      1. Solve ZAMS with the shooting solver → converged (logL, logTe)
      2. Integrate full structure surface→center via build_model_on_mesh
      3. Interpolate onto the Lagrangian mass mesh q_k (from #92)

    Parameters
    ----------
    M_solar : float
        Stellar mass in solar masses.
    Z : float
        Metal mass fraction.
    alpha_mlt : float
        Mixing-length parameter.
    N_mesh : int
        Number of mass zones (default 1000 from #92 recommendation).

    Returns
    -------
    dict with keys:
        'log_L', 'log_Teff' : scalar ZAMS values (from shooting solver)
        'q' : (N_mesh+1,) mass coordinate array
        'logP', 'logT', 'logrho' : (N_mesh+1,) thermodynamic state
        'r' : (N_mesh+1,) radius in cm
        'L' : (N_mesh+1,) luminosity in erg/s
    """
    from stellar_jax.config.constants import Y_BBN, DY_DZ
    from stellar_jax.config.mesh_defaults import N_NEWTON_COLD

    M_solar_j = jnp.asarray(float(M_solar), dtype=jnp.float64)
    Z_j = jnp.asarray(float(Z), dtype=jnp.float64)
    alpha_j = jnp.asarray(float(alpha_mlt), dtype=jnp.float64)

    # Homogeneous ZAMS composition
    Y = Y_BBN + DY_DZ * Z
    X_init = max(1.0 - Y - Z, 0.5)
    X_profile = jnp.full(N_COMP, X_init)

    # Step 1: Newton-solve for converged (logL, logTe)
    log_L_g, log_Te_g = initial_guess(M_solar_j)
    logL, logTe = newton_solve_xprofile(
        M_solar_j, X_profile, Z_j, jnp.float64(0.0),
        log_L_g, log_Te_g, alpha_j, N_NEWTON_COLD
    )

    # Step 2+3: Build model on mesh
    model = build_model_on_mesh(M_solar_j, logL, logTe, X_profile, Z_j, alpha_j, N_mesh)
    model['log_L'] = logL
    model['log_Teff'] = logTe
    return model
