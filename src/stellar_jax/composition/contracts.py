"""VJP contracts for the composition operators.

This module documents which inputs to each composition function carry live
gradients and which are detached (stop_gradient'd) by the caller. The gradient
policy is enforced by the ORCHESTRATOR (step_fn in evolution/_core.py), NOT by the
composition modules themselves — the modules are PURE functions that accept
whatever they are given and produce differentiable outputs.

MESA Architectural Reference:
  - struct_burn_mix.f90:82-100 — op_split_burn: when active, MESA zeros
    d_epsnuc_dlnT, d_epsnuc_dlnd, d_epsnuc_dx in the structure Jacobian before
    calling the burn operator. The burn sees converged T/rho as FROZEN parameters.
  - evolve.f90:634-639 — element diffusion is called AFTER do_struct_burn_mix,
    fully outside the Newton Jacobian.
  - mix_info.f90:112 — CZ boundaries are integer classifications
    (mixing_type = mlt_mixing_type), not smooth functions. MESA never
    differentiates through them.

Our code mirrors this operator-split architecture via stop_gradient:
  - shell_data_sg = stop_gradient(shell_data) before passing to ANY composition op
    → structure does not differentiate backward through composition
  - diffuse_composition is fully stop_gradient'd
    → diffusion is detached (operator-split, matching evolve.f90:634)
  - C12_m, C13_m are stop_gradient'd after mixing
    → only N14 carry is live (feeds eps_nuclear)
  - Z_mixed is stop_gradient'd
    → no gradient path through Z_profile

KEY PRINCIPLE: These stop_gradients belong to step_fn, NOT to the composition
modules. This design allows #447 (composition adjoint regularization) to later
make some of these paths live by removing the caller-side stop_gradient —
without modifying the composition package at all.

===========================================================================
VJP CONTRACT PER FUNCTION
===========================================================================

burn_composition(X_profile, Y_profile, shell_data, dt)
-------------------------------------------
LIVE inputs (caller passes WITHOUT stop_gradient):
  - X_profile: ∂burn/∂X flows backward (hydrogen depletion gradient)
  - Y_profile: ∂burn/∂Y flows backward (helium deposition gradient)
  NOTE: in step_fn, burn_composition's logic is INLINED as
        (dX = eps_comp * dt / Q_PER_G; actual_dX = min(dX, X);
         X_burned = max(X - dX, 0); Y_burned = Y + actual_dX)
        using eps derived from the live Henyey y_henyey_out. So the gradient
        flows through Henyey → eps → X/Y, NOT through burn_composition's
        shell_data[:, 0] path. The standalone burn_composition is used in
        the legacy shooting path (_comp_step_with_dt / _comp_step_diffusion).

DETACHED inputs (caller passes with stop_gradient):
  - shell_data: always shell_data_sg in step_fn [GP-5: structure→comp coupling]
  - dt: algorithmic (timestep control) [GP-1]

Output gradient:
  - ∂X_burned/∂X_profile = indicator(X - dX > 0) [standard AD through max]
  - ∂Y_burned/∂Y_profile = 1 (identity — He receives the burned H)
  - ∂Y_burned/∂X_profile = indicator(dX < X) (actual_dX = min(dX, X))
  NOTE: in step_fn, actual_dX entering Y_burned is stop_gradient'd [GP-Y]
  because the Henyey solver derives Y = 1-X-Z (solver/residual.py:67),
  NOT from the tracked Y_profile. The backward path through Y_burned →
  actual_dX → eps_comp → Henyey is a dead gradient path. The standalone
  burn_composition (used in calibration/shooting) retains the live path.


burn_cno(C12_profile, C13_profile, N14_profile, X_profile, shell_data, dt, ...)
--------------------------------------------------------------------------------
LIVE inputs:
  - X_profile: ∂rate_cn/∂X_H is a valid physical gradient path
    (X_H sets the CN equilibration timescale τ_CN = 1/λ_CN).
  - N14_profile: N14 feeds back into eps_nuclear at the next timestep
    via the carry → Henyey solve. Only N14_m (after mix) stays live;
    C12_m, C13_m are stop_gradient'd by the caller because they don't
    enter epsilon_nuclear (∂L/∂C12 = 0).

DETACHED inputs (caller passes with stop_gradient):
  - shell_data: shell_data_sg [GP-5]
    Rationale: T_structure → rate_cn → N14 would produce NaN over many
    steps via ∂exp(-136.90/T6^{1/3})/∂T6^{1/3} accumulating. The Henyey
    IFT already captures ∂L/∂T_structure redundantly.
  - C12_profile, C13_profile: live into burn_cno BUT the OUTPUT C12_m/C13_m
    is stop_gradient'd by the caller (they don't affect eps_nuclear).
  - dt: algorithmic [GP-1]

Output gradient:
  - Only N14_b → N14_m (after mix) flows live in the carry.
  - C12_new, C13_new are stop_gradient'd by caller post-mix [GP-4].


mix_composition(X_profile, shell_data, M_solar, f_ov, ...)
-----------------------------------------------------------
LIVE inputs:
  - X_profile (or Y_diff, N14_b): the composition being mixed.
    Gradient flows through the zone-averaging operation.

DETACHED inputs (caller passes with stop_gradient):
  - shell_data: shell_data_sg [GP-5]
    The sigmoid CZ-boundary position is differentiable in principle, but
    accumulated over N=200 steps × 600 zones the gradient noise exceeds 2%.
    MESA classifies CZ boundaries as integers (mix_info.f90:112) — never
    differentiated. CONSTRAINT: our sigmoid provides a smooth forward but
    the backward is detached for stability.
  - M_solar: NOT stop_gradient'd explicitly, but mass enters mix only
    through f_ov_effective (the overshoot ramp). The gradient of mass through
    the structure solve (Henyey IFT) is the primary path; mix's mass
    dependence is secondary and well-behaved.
  - f_ov: NOT stop_gradient'd. Carries live gradient — validated as a
    first-class differentiable input (#1031, test_f_ov_gradient_ad_vs_fd).
    ∂(center_h1)/∂f_ov flows through the sigmoid overshoot mask
    (f_ov → f_ov_effective → delta_m_ov → core_mix_mask → zone-averaged
    composition → carry → Henyey solve). The straight-through estimator
    provides bounded, non-zero gradient at the CZ boundary.
    MESA ref: overshoot_step.f90:eval_overshoot_step (f parameter).

Output gradient:
  - ∂X_mixed/∂X_profile flows through the mass-weighted average.
  - The sigmoid straight-through estimator (Bengio 2013) gives
    bounded, non-zero gradient at the CZ boundary in the forward pass,
    but the accumulated noise over many steps is why shell_data is detached.


diffuse_composition(X_profile, Y_profile, shell_data, dt, M_solar, ...)
------------------------------------------------------------------------
FULLY DETACHED by the caller:
  step_fn wraps the entire call in stop_gradient:
    _diff_result = jax.lax.stop_gradient(diffuse_composition(...))

Then uses a straight-through estimator:
    X_diff = X_mixed + (X_diffused - stop_gradient(X_mixed))
  which gives ∂X_diff/∂X_mixed = 1 (identity gradient through diffusion).

Rationale:
  - MESA evolve.f90:634 — diffusion is operator-split, outside the Jacobian.
  - Physical: τ_diff ≈ 6×10^13 yr vs τ_nuc ≈ 10^10 yr;
    per-step ΔX from diffusion ≈ O(10^-5) vs burning ΔX ≈ O(10^-3).
  - Contribution to ∂X_c/∂M is 100× smaller than nuclear burning.
  - Verified: test_ad_vs_fd_gradient_correctness_n500 confirms <1% AD-vs-FD
    agreement with diffusion ON and operator-split.

The standalone test_diffusion_gradient_ad_vs_fd validates diffuse_composition's
internal gradients (when called directly, not through evolve_star's wrapper).


_mix_z_in_cz / mix_z_in_cz(Z_profile, shell_data, M_solar, f_ov, ...)
-----------------------------------------------------------------------
FULLY DETACHED by the caller:
  Z_mixed = jax.lax.stop_gradient(_mix_z_in_cz(Z_new, shell_data_sg, ...))

Rationale:
  - Z_profile has no gradient path in the Henyey solver (which uses scalar Z
    for EOS/opacity, not Z_profile). Both inputs (Z_new from diffusion,
    shell_data_sg) are already detached.
  - Without explicit stop_gradient, XLA still traces through mix_composition
    when building the backward graph, inflating HLO + compilation time ~10 min.
  - The evolution's gradient w.r.t. mass/alpha flows through X (burn/mix →
    structure) and Y (mix → structure), not Z_profile.

===========================================================================
GRADIENT POLICY NUMBERS (cross-reference with gradient_policy.py, future P6)
===========================================================================
GP-1: TIMESTEP — dt decisions, varcontrol, reject flags
GP-2: MESH_POSITIONS — grid points (values ON grid are live)
GP-3: DIFFUSION — full output stop_gradient'd (MESA operator-split)
GP-4: Z_CNO_ISOTOPES — Z_mixed, C12_m, C13_m (don't enter eps_nuclear)
GP-5: STRUCTURE_TO_COMP — shell_data_sg (one-way coupling: structure drives
      composition, not vice-versa for the current adjoint)
"""
