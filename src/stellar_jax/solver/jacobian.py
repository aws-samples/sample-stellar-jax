"""Jacobian block assembly for the Henyey Newton solver.

Contains:
  - _jacobian_blocks: Jacobian with Dirichlet BCs (pinned surface)
  - _jacobian_blocks_fixed_bc: Jacobian with linearized atmosphere BCs
  - _jacobian_blocks_atm: Jacobian with full atmosphere bridge

All use jax.jacfwd to compute 4×4 blocks of ∂F/∂y per interface.
Interior blocks use _cell_residual_raw (plain jnp.where MLT switch)
for exact hard-switch derivatives in radiative zones.

References:
  - Henyey, Forbes & Gould (1964), ApJ 139, 306
  - Paxton et al. (2011), ApJS 192, 3, §6.3
"""

import jax
import jax.numpy as jnp

from stellar_jax.config.constants import Lsun, a_rad
from stellar_jax.config.physics_floors import PGAS_FRAC_FLOOR
from stellar_jax.microphysics.eos import eos_lookup
from stellar_jax.microphysics.nuclear import epsilon_nuclear
from stellar_jax.microphysics.neutrino import epsilon_neutrino
from stellar_jax.solver.residual import _cell_residual, _cell_residual_raw, _prepare_cell_inputs, _center_eps_grav


# lint: scan-body exception — inline Jacobian for XLA trace identity
def _jacobian_blocks(y, q_mesh, M_star, X_profile, Z, alpha_mlt,
                     ln_T_prev=None, ln_P_prev=None, inv_dt=None,
                     comp_mfracs=None, opacity_factor=None, eps_nuc_factor=None,
                     helm_eos=False, bicubic_opacity=True):
    """Compute (A, B, C) Jacobian blocks with Dirichlet BCs. Each (N_s, 4, 4).

    NOTE: This function is kept INLINE (not delegated to _jacobian_blocks_core)
    to preserve XLA trace identity on the ZAMS Newton path. The ZAMS solve is
    part of evolve_star, which jacrev(evolve_star) differentiates through.
    Delegating to _jacobian_blocks_core (with callback closures) restructures
    the XLA HLO graph enough to cause OOM during backward-pass compilation of
    the MS inversion test (test_ms_inversion_real_evolve_star).
    Bit-identical to main.
    """
    ci = _prepare_cell_inputs(y, q_mesh, M_star, X_profile, Z,
                              ln_T_prev, ln_P_prev, inv_dt,
                              comp_mfracs=comp_mfracs)
    N_s, N_c = ci.N_s, ci.N_c

    # Block 0: depends on (y_0, y_1)
    def block0_fn(y_pair):
        y_0, y_1 = y_pair[0], y_pair[1]
        ln_r_0, ln_P_0, ln_T_0, ell_0 = y_0
        T_c = jnp.exp(ln_T_0)
        P_c = jnp.exp(ln_P_0)
        P_rad_c = a_rad * T_c**4 / 3.0
        P_gas_c = jnp.maximum(P_c - P_rad_c, PGAS_FRAC_FLOOR * P_c)
        rho_c, _mu_c, nad_c, _S_c, cp_c, _chi_rho_c, _chi_T_c = eos_lookup(
            jnp.log10(T_c), jnp.log10(P_gas_c), X_profile[0], Z, helm_eos=helm_eos)
        m1 = M_star * q_mesh[1]
        r0_exp = (3.0 * m1 / (4.0 * jnp.pi * rho_c))**(1.0 / 3.0)
        BC1 = ln_r_0 - jnp.log(jnp.maximum(r0_exp, 1.0))
        eps_c = epsilon_nuclear(rho_c, T_c, X_profile[0], Z, jnp.float64(0.0))
        eps_nu_c = epsilon_neutrino(rho_c, T_c, X_profile[0], Z)
        # eps_grav at center (MESA hydro_energy.f90:108) — ln_T_prev/ln_P_prev
        # are constants (previous step) captured from the outer scope.
        eps_grav_c = _center_eps_grav(T_c, P_c, cp_c, nad_c,
                                      ln_T_prev, ln_P_prev, ci.inv_dt)
        BC2 = ell_0 - (eps_c - eps_nu_c + eps_grav_c) * m1 / Lsun
        F0 = _cell_residual(y_0, y_1, ci.dm[0], ci.m_mid[0], M_star, ci.X_mid[0], Z, alpha_mlt,
                            ci.ln_T_prev_mid[0], ci.ln_P_prev_mid[0], ci.inv_dt,
                            opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                            helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
        return jnp.array([BC1, BC2, F0[0], F0[1]])

    J0 = jax.jacfwd(block0_fn)(jnp.stack([y[0], y[1]]))
    B_0 = J0[:, 0, :]
    C_0 = J0[:, 1, :]

    # Last block: depends on (y_{N_s-2}, y_{N_s-1}); surface is pinned
    def blockN_fn(y_pair):
        y_Nm1, y_N_ = y_pair[0], y_pair[1]
        F_last = _cell_residual(y_Nm1, y_N_, ci.dm[N_c - 1], ci.m_mid[N_c - 1], M_star,
                                ci.X_mid[N_c - 1], Z, alpha_mlt,
                                ci.ln_T_prev_mid[N_c - 1], ci.ln_P_prev_mid[N_c - 1], ci.inv_dt,
                                opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                            helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
        return jnp.array([F_last[2], F_last[3], 0.0, 0.0])

    JN = jax.jacfwd(blockN_fn)(jnp.stack([y[N_s - 2], y[N_s - 1]]))
    A_N = JN[:, 0, :]
    B_N = JN[:, 1, :]
    B_N = B_N.at[2, :].set(jnp.array([0.0, 0.0, 1.0, 0.0]))
    B_N = B_N.at[3, :].set(jnp.array([0.0, 0.0, 0.0, 1.0]))
    A_N = A_N.at[2, :].set(0.0)
    A_N = A_N.at[3, :].set(0.0)

    # Interior blocks k=1..N_c-1
    def compute_one(k):
        y_triple = jnp.stack([y[k - 1], y[k], y[k + 1]])
        def res_fn(yt):
            y_km1, y_k_, y_kp1 = yt[0], yt[1], yt[2]
            F_left = _cell_residual(y_km1, y_k_, ci.dm[k - 1], ci.m_mid[k - 1], M_star,
                                    ci.X_mid[k - 1], Z, alpha_mlt,
                                    ci.ln_T_prev_mid[k - 1], ci.ln_P_prev_mid[k - 1], ci.inv_dt,
                                    opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                            helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
            F_right = _cell_residual(y_k_, y_kp1, ci.dm[k], ci.m_mid[k], M_star,
                                     ci.X_mid[k], Z, alpha_mlt,
                                     ci.ln_T_prev_mid[k], ci.ln_P_prev_mid[k], ci.inv_dt,
                                     opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                            helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
            return jnp.array([F_left[2], F_left[3], F_right[0], F_right[1]])
        J = jax.jacfwd(res_fn)(y_triple)
        return J[:, 0, :], J[:, 1, :], J[:, 2, :]

    A_int, B_int, C_int = jax.vmap(compute_one)(jnp.arange(1, N_c))

    A = jnp.zeros((N_s, 4, 4))
    B = jnp.zeros((N_s, 4, 4))
    C = jnp.zeros((N_s, 4, 4))
    B = B.at[0].set(B_0)
    C = C.at[0].set(C_0)
    A = A.at[1:N_c].set(A_int)
    B = B.at[1:N_c].set(B_int)
    C = C.at[1:N_c].set(C_int)
    A = A.at[N_s - 1].set(A_N)
    B = B.at[N_s - 1].set(B_N)
    return A, B, C


# lint: scan-body exception — inline Jacobian for XLA trace identity
def _jacobian_blocks_fixed_bc(y, q_mesh, M_star, X_profile, Z, alpha_mlt,
                               ln_P_atm, ln_T_atm, atm_jac, y_surf_ref,
                               ln_T_prev=None, ln_P_prev=None, inv_dt=None,
                               bc1_floor=1.0, subtract_neutrinos=True,
                               comp_mfracs=None, opacity_factor=None, eps_nuc_factor=None,
                               X3_profile=None, ln_rho_guess=None,
                               gradL_composition_term=None,
                               helm_eos=False, bicubic_opacity=True):
    """Compute (A, B, C) Jacobian blocks with linearized atmosphere BC.

    NOTE: This function is kept INLINE (not delegated to _jacobian_blocks_core)
    to preserve XLA trace identity on the Newton per-step path. The adaptive
    forward RGB tests accumulate float error over thousands of steps and are
    sensitive to XLA graph restructuring (~0.02-0.03 dex shift). Inlining
    ensures bit-identical HLO vs the pre-refactoring code.

    NOTE (#393): This Jacobian does NOT thread N14_profile into epsilon_nuclear
    (uses the ZAMS CN+ON-eq fallback X_CNO=0.251*Z, #1026). This is an acceptable
    approximate Jacobian — Newton still converges to the correct fixed point.
    """
    ci = _prepare_cell_inputs(y, q_mesh, M_star, X_profile, Z,
                              ln_T_prev, ln_P_prev, inv_dt,
                              comp_mfracs=comp_mfracs, X3_profile=X3_profile,
                              ln_rho_guess=ln_rho_guess)
    N_s, N_c = ci.N_s, ci.N_c

    # GP-9 IFT_CONSISTENCY: when a per-cell Ledoux gradL is supplied
    # (replay/adaptive path), this Jacobian is the jacfwd of the SAME adaptive
    # residual the forward solved, so the IFT adjoint is the exact transpose.
    # None (all non-replay callers) → on_adaptive_path=False → byte-identical.
    _on_adaptive_path = gradL_composition_term is not None

    # X3 at center for the center BC eps_nuclear call
    _x3_c = X3_profile[0] if X3_profile is not None else None

    def block0_fn(y_pair):
        y_0, y_1 = y_pair[0], y_pair[1]
        ln_r_0, ln_P_0, ln_T_0, ell_0 = y_0
        T_c = jnp.exp(ln_T_0)
        P_c = jnp.exp(ln_P_0)
        P_rad_c = a_rad * T_c**4 / 3.0
        P_gas_c = jnp.maximum(P_c - P_rad_c, PGAS_FRAC_FLOOR * P_c)
        rho_c, _mu_c, nad_c, _S_c, cp_c, _chi_rho_c, _chi_T_c = eos_lookup(
            jnp.log10(T_c), jnp.log10(P_gas_c), X_profile[0], Z, helm_eos=helm_eos)
        m1 = M_star * q_mesh[1]
        r0_exp = (3.0 * m1 / (4.0 * jnp.pi * rho_c))**(1.0 / 3.0)
        BC1 = ln_r_0 - jnp.log(jnp.maximum(r0_exp, bc1_floor))
        eps_c = epsilon_nuclear(rho_c, T_c, X_profile[0], Z, jnp.float64(0.0),
                                X3=_x3_c, X3_eq_frozen=_x3_c)
        eps_nu_c = epsilon_neutrino(rho_c, T_c, X_profile[0], Z)
        # eps_grav at center (MESA hydro_energy.f90:108)
        eps_grav_c = _center_eps_grav(T_c, P_c, cp_c, nad_c,
                                      ln_T_prev, ln_P_prev, ci.inv_dt)
        BC2 = ell_0 - jnp.where(subtract_neutrinos,
                                 (eps_c - eps_nu_c + eps_grav_c),
                                 (eps_c + eps_grav_c)) * m1 / Lsun
        _x3_mid_0 = ci.X3_mid[0] if ci.X3_mid is not None else None
        _rho_guess_0 = ci.ln_rho_guess_mid[0] if ci.ln_rho_guess_mid is not None else None
        _gradL_0 = gradL_composition_term[0] if gradL_composition_term is not None else 0.0
        F0 = _cell_residual_raw(y_0, y_1, ci.dm[0], ci.m_mid[0], M_star, ci.X_mid[0], Z, alpha_mlt,
                            ci.ln_T_prev_mid[0], ci.ln_P_prev_mid[0], ci.inv_dt,
                            opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                            X3_mid=_x3_mid_0, ln_rho_guess_mid=_rho_guess_0,
                            gradL_composition_term=_gradL_0, on_adaptive_path=_on_adaptive_path,
                            helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
        return jnp.array([BC1, BC2, F0[0], F0[1]])

    J0 = jax.jacfwd(block0_fn)(jnp.stack([y[0], y[1]]))
    B_0 = J0[:, 0, :]
    C_0 = J0[:, 1, :]

    _x3_mid_last = ci.X3_mid[N_c - 1] if ci.X3_mid is not None else None
    _rho_guess_last = ci.ln_rho_guess_mid[N_c - 1] if ci.ln_rho_guess_mid is not None else None
    _gradL_last = gradL_composition_term[N_c - 1] if gradL_composition_term is not None else 0.0

    def blockN_fn(y_pair):
        y_Nm1, y_N = y_pair[0], y_pair[1]
        F_last = _cell_residual_raw(y_Nm1, y_N, ci.dm[N_c - 1], ci.m_mid[N_c - 1], M_star,
                                ci.X_mid[N_c - 1], Z, alpha_mlt,
                                ci.ln_T_prev_mid[N_c - 1], ci.ln_P_prev_mid[N_c - 1], ci.inv_dt,
                                opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                                X3_mid=_x3_mid_last, ln_rho_guess_mid=_rho_guess_last,
                                gradL_composition_term=_gradL_last, on_adaptive_path=_on_adaptive_path,
                                helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
        dy_surf = y_N - y_surf_ref
        ln_P_target = ln_P_atm + jnp.dot(atm_jac[0], dy_surf)
        ln_T_target = ln_T_atm + jnp.dot(atm_jac[1], dy_surf)
        BC_P = y_N[1] - ln_P_target
        BC_T = y_N[2] - ln_T_target
        return jnp.array([F_last[2], F_last[3], BC_P, BC_T])

    JN = jax.jacfwd(blockN_fn)(jnp.stack([y[N_s - 2], y[N_s - 1]]))
    A_N = JN[:, 0, :]
    B_N = JN[:, 1, :]

    def compute_one(k):
        y_triple = jnp.stack([y[k - 1], y[k], y[k + 1]])
        _x3_left = ci.X3_mid[k - 1] if ci.X3_mid is not None else None
        _x3_right = ci.X3_mid[k] if ci.X3_mid is not None else None
        _rho_guess_left = ci.ln_rho_guess_mid[k - 1] if ci.ln_rho_guess_mid is not None else None
        _rho_guess_right = ci.ln_rho_guess_mid[k] if ci.ln_rho_guess_mid is not None else None
        _gradL_left = gradL_composition_term[k - 1] if gradL_composition_term is not None else 0.0
        _gradL_right = gradL_composition_term[k] if gradL_composition_term is not None else 0.0
        def res_fn(yt):
            y_km1, y_k_, y_kp1 = yt[0], yt[1], yt[2]
            F_left = _cell_residual_raw(y_km1, y_k_, ci.dm[k - 1], ci.m_mid[k - 1], M_star,
                                    ci.X_mid[k - 1], Z, alpha_mlt,
                                    ci.ln_T_prev_mid[k - 1], ci.ln_P_prev_mid[k - 1], ci.inv_dt,
                                    opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                                    X3_mid=_x3_left, ln_rho_guess_mid=_rho_guess_left,
                                    gradL_composition_term=_gradL_left, on_adaptive_path=_on_adaptive_path,
                                    helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
            F_right = _cell_residual_raw(y_k_, y_kp1, ci.dm[k], ci.m_mid[k], M_star,
                                     ci.X_mid[k], Z, alpha_mlt,
                                     ci.ln_T_prev_mid[k], ci.ln_P_prev_mid[k], ci.inv_dt,
                                     opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                                     X3_mid=_x3_right, ln_rho_guess_mid=_rho_guess_right,
                                     gradL_composition_term=_gradL_right, on_adaptive_path=_on_adaptive_path,
                                     helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
            return jnp.array([F_left[2], F_left[3], F_right[0], F_right[1]])
        J = jax.jacfwd(res_fn)(y_triple)
        return J[:, 0, :], J[:, 1, :], J[:, 2, :]

    A_int, B_int, C_int = jax.vmap(compute_one)(jnp.arange(1, N_c))

    A = jnp.zeros((N_s, 4, 4))
    B = jnp.zeros((N_s, 4, 4))
    C = jnp.zeros((N_s, 4, 4))
    B = B.at[0].set(B_0)
    C = C.at[0].set(C_0)
    A = A.at[1:N_c].set(A_int)
    B = B.at[1:N_c].set(B_int)
    C = C.at[1:N_c].set(C_int)
    A = A.at[N_s - 1].set(A_N)
    B = B.at[N_s - 1].set(B_N)
    return A, B, C



# lint: scan-body exception — inline Jacobian for XLA trace identity
def _jacobian_blocks_atm(y, q_mesh, M_star, X_profile, Z, alpha_mlt,
                          ln_T_prev=None, ln_P_prev=None, inv_dt=None, T_eff=None, R_phot=None,
                          bc1_floor=1.0, subtract_neutrinos=True, comp_mfracs=None,
                          opacity_factor=None, eps_nuc_factor=None,
                          helm_eos=False, bicubic_opacity=True):
    """Compute (A, B, C) Jacobian blocks with atmosphere surface BC.

    NOTE: This function is kept INLINE (not delegated to _jacobian_blocks_core)
    to preserve XLA trace identity. It is called from _henyey_init_atm, which
    runs as part of the evolve_star forward trace compiled by jacrev(evolve_star).
    Even though the result is stop_gradient'd, the XLA compiler still traces
    the full forward graph — delegating to a callback-parameterized helper
    produces a different/larger HLO graph that causes OOM during backward-pass
    compilation.
    """
    from stellar_jax.solver.surface_bc import _surface_bc_atm

    ci = _prepare_cell_inputs(y, q_mesh, M_star, X_profile, Z,
                              ln_T_prev, ln_P_prev, inv_dt,
                              comp_mfracs=comp_mfracs)
    N_s, N_c = ci.N_s, ci.N_c

    # Block 0 (center)
    def block0_fn(y_pair):
        y_0, y_1 = y_pair[0], y_pair[1]
        ln_r_0, ln_P_0, ln_T_0, ell_0 = y_0
        T_c = jnp.exp(ln_T_0)
        P_c = jnp.exp(ln_P_0)
        P_rad_c = a_rad * T_c**4 / 3.0
        P_gas_c = jnp.maximum(P_c - P_rad_c, PGAS_FRAC_FLOOR * P_c)
        rho_c, _mu_c, nad_c, _S_c, cp_c, _chi_rho_c, _chi_T_c = eos_lookup(
            jnp.log10(T_c), jnp.log10(P_gas_c), X_profile[0], Z, helm_eos=helm_eos)
        m1 = M_star * q_mesh[1]
        r0_exp = (3.0 * m1 / (4.0 * jnp.pi * rho_c))**(1.0 / 3.0)
        BC1 = ln_r_0 - jnp.log(jnp.maximum(r0_exp, bc1_floor))
        eps_c = epsilon_nuclear(rho_c, T_c, X_profile[0], Z, jnp.float64(0.0))
        eps_nu_c = epsilon_neutrino(rho_c, T_c, X_profile[0], Z)
        # eps_grav at center (MESA hydro_energy.f90:108)
        eps_grav_c = _center_eps_grav(T_c, P_c, cp_c, nad_c,
                                      ln_T_prev, ln_P_prev, ci.inv_dt)
        BC2 = ell_0 - jnp.where(subtract_neutrinos,
                                 (eps_c - eps_nu_c + eps_grav_c),
                                 (eps_c + eps_grav_c)) * m1 / Lsun
        F0 = _cell_residual_raw(y_0, y_1, ci.dm[0], ci.m_mid[0], M_star, ci.X_mid[0], Z, alpha_mlt,
                            ci.ln_T_prev_mid[0], ci.ln_P_prev_mid[0], ci.inv_dt,
                            opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                            helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
        return jnp.array([BC1, BC2, F0[0], F0[1]])

    J0 = jax.jacfwd(block0_fn)(jnp.stack([y[0], y[1]]))
    B_0 = J0[:, 0, :]
    C_0 = J0[:, 1, :]

    # Last block (surface): cell_{N_c-1} F3,F4 + atmosphere BC
    X_surf = X_profile[-1]

    def blockN_fn(y_pair):
        y_Nm1, y_N = y_pair[0], y_pair[1]
        F_last = _cell_residual_raw(y_Nm1, y_N, ci.dm[N_c - 1], ci.m_mid[N_c - 1], M_star,
                                ci.X_mid[N_c - 1], Z, alpha_mlt,
                                ci.ln_T_prev_mid[N_c - 1], ci.ln_P_prev_mid[N_c - 1], ci.inv_dt,
                                opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                            helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
        BC_P, BC_T = _surface_bc_atm(y_N, M_star, X_surf, Z, alpha_mlt, T_eff, R_phot,
                                      helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
        return jnp.array([F_last[2], F_last[3], BC_P, BC_T])

    JN = jax.jacfwd(blockN_fn)(jnp.stack([y[N_s - 2], y[N_s - 1]]))
    A_N = JN[:, 0, :]
    B_N = JN[:, 1, :]

    # Interior blocks
    def compute_one(k):
        y_triple = jnp.stack([y[k - 1], y[k], y[k + 1]])
        def res_fn(yt):
            y_km1, y_k_, y_kp1 = yt[0], yt[1], yt[2]
            F_left = _cell_residual_raw(y_km1, y_k_, ci.dm[k - 1], ci.m_mid[k - 1], M_star,
                                    ci.X_mid[k - 1], Z, alpha_mlt,
                                    ci.ln_T_prev_mid[k - 1], ci.ln_P_prev_mid[k - 1], ci.inv_dt,
                                    opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                            helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
            F_right = _cell_residual_raw(y_k_, y_kp1, ci.dm[k], ci.m_mid[k], M_star,
                                     ci.X_mid[k], Z, alpha_mlt,
                                     ci.ln_T_prev_mid[k], ci.ln_P_prev_mid[k], ci.inv_dt,
                                     opacity_factor=opacity_factor, eps_nuc_factor=eps_nuc_factor,
                            helm_eos=helm_eos, bicubic_opacity=bicubic_opacity)
            return jnp.array([F_left[2], F_left[3], F_right[0], F_right[1]])
        J = jax.jacfwd(res_fn)(y_triple)
        return J[:, 0, :], J[:, 1, :], J[:, 2, :]

    A_int, B_int, C_int = jax.vmap(compute_one)(jnp.arange(1, N_c))

    A = jnp.zeros((N_s, 4, 4))
    B = jnp.zeros((N_s, 4, 4))
    C = jnp.zeros((N_s, 4, 4))
    B = B.at[0].set(B_0)
    C = C.at[0].set(C_0)
    A = A.at[1:N_c].set(A_int)
    B = B.at[1:N_c].set(B_int)
    C = C.at[1:N_c].set(C_int)
    A = A.at[N_s - 1].set(A_N)
    B = B.at[N_s - 1].set(B_N)
    return A, B, C
