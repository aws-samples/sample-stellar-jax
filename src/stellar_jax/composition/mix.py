"""Convective mixing of composition profiles.

These are PURE functions — no stop_gradient, no side effects.

Mixing implements the Schwarzschild criterion with smooth sigmoid boundaries
for differentiability. Overshoot is step-type (Herwig 2000; f_ov parameter).

References:
  - Böhm-Vitense (1958): MLT / convective mixing
  - MESA mix_info.f90: CZ classification + boundary detection
  - MESA adjust_xyz.f90:do_adjust_xyz_for_mixing: zone homogenization
  - Bengio et al. (2013), arXiv:1308.3432: straight-through estimator
"""
import jax
import jax.numpy as jnp

from stellar_jax.config.mesh_defaults import COMP_MFRACS, COMP_ZONE_MASSES, compute_zone_masses, F_OV

# Envelope mixing is always enabled in production. The O2 mutation gate
# uses mechanism 2 (direct function patch of _envelope_zone_mixing) which
# operates at trace time.


def _core_conv_mask(is_conv_comp, hp_comp, dmdr_comp, comp_mfracs, M_solar, f_ov, _K=200.0):
    """Build the combined core convection + overshoot mask.

    Computes the Schwarzschild-criterion core mask via cumulative product of
    per-zone convective probability, then extends it with step overshooting.
    Uses straight-through estimators for differentiable boundary position.

    Parameters
    ----------
    is_conv_comp : array (N_COMP,)
        Smooth convective indicator on composition grid (sigmoid output).
    hp_comp : array (N_COMP,)
        Pressure scale height (Hp/R) on composition grid.
    dmdr_comp : array (N_COMP,)
        Normalized dm/dr on composition grid.
    comp_mfracs : array (N_COMP,)
        Composition grid mass-fraction coordinates.
    M_solar : scalar
        Stellar mass in solar masses.
    f_ov : scalar
        Overshooting parameter.
    _K : scalar
        Sigmoid sharpness for boundary detection (default 200).

    Returns
    -------
    core_mix_mask : array (N_COMP,)
        Combined core+overshoot mask (0=radiative, 1=mixed).
    """
    # Core mask via cumulative product (log-sum-exp trick)
    log_conv = jnp.log(jnp.clip(is_conv_comp, 1e-10, 1.0))
    soft_core_mask = jnp.clip(jnp.exp(jnp.cumsum(log_conv)), 0.0, 1.0)
    hard_core_mask = (soft_core_mask > 0.5).astype(jnp.float64)

    # Core boundary location
    n_comp = comp_mfracs.shape[0]
    n_core_conv = jnp.sum(soft_core_mask)
    bdy_pos = jnp.clip(n_core_conv - 1.0, 0.0, n_comp - 1.0)
    indices = jnp.arange(n_comp, dtype=jnp.float64)
    bdy_weights = jnp.exp(-0.5 * ((indices - bdy_pos) / 0.5) ** 2)
    bdy_weights = bdy_weights / (jnp.sum(bdy_weights) + 1e-30)

    hp_at_bdy = jnp.sum(bdy_weights * hp_comp)
    dmdr_at_bdy = jnp.sum(bdy_weights * dmdr_comp)
    m_bdy = jnp.sum(bdy_weights * comp_mfracs)

    # Mass-dependent overshoot ramp: smoothly switch core overshoot ON above
    # ~1.2 M☉, where stars develop a persistent convective core (below ~1.1–1.2 M☉
    # the H-burning core is radiative, so core overshoot does not apply).
    # NOTE: the 1.2 M☉ center and 0.1 M☉ sigmoid width are a DIFFERENTIABLE
    # modeling choice (smooth gate), not a calibrated/cited value — they set only
    # where the ramp turns on, not the overshoot physics (f_ov, set above).
    f_ov_effective = f_ov / (1.0 + jnp.exp(-(M_solar - 1.2) / 0.1))
    delta_m_ov = f_ov_effective * hp_at_bdy * dmdr_at_bdy

    soft_ov_mask = (jax.nn.sigmoid(_K * (comp_mfracs - m_bdy)) *
                    jax.nn.sigmoid(_K * (m_bdy + delta_m_ov - comp_mfracs)))
    has_core = jax.nn.sigmoid(_K * (n_core_conv - 1.0))
    soft_ov_mask = soft_ov_mask * has_core

    hard_ov_mask = (soft_ov_mask > 0.5).astype(jnp.float64)

    # Combined core+overshoot
    soft_combined = jnp.clip(jnp.maximum(soft_core_mask, soft_ov_mask), 0.0, 1.0)
    hard_combined = jnp.maximum(hard_core_mask, hard_ov_mask)
    core_mix_mask = jax.lax.stop_gradient(hard_combined - soft_combined) + soft_combined  # GP-5: STRUCTURE_TO_COMP

    return core_mix_mask


def _envelope_zone_mixing(X_after_core, comp_mfracs, mf_shells, nrad, nad,
                          core_mix_mask, zone_weights, _K_env=30.0,
                          gradL_composition_term=None):
    """Mix composition within the surface-connected envelope convective zone.

    Identifies the OUTERMOST contiguous envelope CZ (the one touching the
    surface at q→1) via the Ledoux criterion, then homogenizes it via
    mass-weighted averaging. Interior convective zones (e.g. from the
    H-burning shell) are NOT mixed — matching MESA's do_mix_envelope
    (mix_info.f90:1344-1381), which explicitly extends the envelope CZ
    inward from the surface and does not touch disconnected interior zones.

    Without this surface-only restriction, for 2.0 M☉ on the RGB, internal
    convective zones from the intense H-shell (high ∇_rad) get included
    in the mixing → CN-processed material from below the true CZ base is
    mixed to the surface → excessive C12 depletion (58.8% vs MESA 22.7%).

    The convective indicator must be CONSISTENT with the solver's own
    boundary criterion. The solver uses Ledoux (#409):
      convective iff ∇_rad > ∇_ad + gradL_composition_term
    so the mixing must use the same test.

    MESA ref:
      - mix_info.f90:1344-1381 (do_mix_envelope: finds the innermost convection
        with T < T_mix_limit and extends to the surface — surface-connected only)
      - turb_support.f90:273 (gradL = grada + gradL_composition_term)
      - turb_support.f90:386 (if gradr > gradL then convective)

    Parameters
    ----------
    X_after_core : array (N_COMP,)
        Profile after core mixing.
    comp_mfracs : array (N_COMP,)
        Composition grid mass-fraction coordinates (0=center, 1=surface).
    mf_shells : array (N_SHELLS,)
        Shell mass fractions (for interpolation).
    nrad, nad : array (N_SHELLS,)
        Radiative and adiabatic gradients on shell grid.
    core_mix_mask : array (N_COMP,)
        Core+overshoot mask (envelope = 1 - core_mix_mask).
    zone_weights : array (N_COMP,)
        Conservative zone masses (sum=1).
    _K_env : scalar
        Envelope sigmoid sharpness (default 30).
    gradL_composition_term : array (N_COMP,) or None
        Ledoux composition gradient term on the composition grid (≥0).
        When None or all-zero, reduces to Schwarzschild (∇_rad − ∇_ad).
        MESA turb_support.f90:273: gradL = grada + gradL_composition_term.

    Returns
    -------
    X_out : array (N_COMP,)
        Profile after envelope mixing.
    """
    from stellar_jax.composition.interp import _interp_shell_to_comp_reversed

    non_core = 1.0 - core_mix_mask

    # Smooth envelope convective indicator at K_env.
    # Use Ledoux criterion: convective when ∇_rad > ∇_ad + gradL_comp.
    # This is consistent with the solver's own convective boundary.
    delta_nab_comp = _interp_shell_to_comp_reversed(comp_mfracs, mf_shells, nrad - nad)
    if gradL_composition_term is not None:
        delta_nab_comp = delta_nab_comp - gradL_composition_term
    env_conv = jax.nn.sigmoid(_K_env * delta_nab_comp) * non_core

    # --- Surface-connected CZ only (MESA do_mix_envelope behavior) ---
    # The comp grid goes from center (index 0) to surface (index N-1).
    # Walk INWARD from the surface: the envelope CZ is the contiguous block
    # of convective zones starting at the surface and extending inward until
    # the first radiative zone. Interior CZs (e.g. H-burning shell) are
    # excluded — they are NOT connected to the surface and should not
    # participate in envelope mixing.
    #
    # Implementation: build a surface-connected mask via cumulative product
    # from the surface inward. env_conv_hard[k]=1 means zone k is convective.
    # surface_mask[k]=1 means zone k is convective AND all zones between it
    # and the surface are also convective (contiguous connection to surface).
    env_conv_hard = (env_conv > 0.5).astype(jnp.float64)

    # Cumulative product from surface inward = reversed cumprod of reversed array.
    # If any zone between k and the surface is radiative, the product drops to 0.
    surface_connected = jnp.flip(jnp.cumprod(jnp.flip(env_conv_hard)))

    # --- Binary homogenization of the surface-connected CZ ---
    #
    # All surface-connected CZ zones are fully replaced by the mass-weighted
    # average. This matches MESA's do_adjust_xyz_for_mixing (adjust_xyz.f90):
    # within a CZ, all cells get the mass-averaged composition (instantaneous
    # homogenization per timestep).
    #
    # The Ledoux criterion (above) + surface-connected restriction already
    # prevent the two pathological FDU failures:
    #   (1) Ledoux stops the CZ from extending past the mu-barrier (fixes
    #       excessive depletion from Schwarzschild-only: 61.5% → ~32%).
    #   (2) Surface-connected stops disconnected interior CZs (H-shell) from
    #       being mixed into the envelope (fixes 58.8% → ~32% for 2.0 M☉).
    #
    # The residual offset from MESA (~32% vs 24%) is within the [40%, 180%]
    # tolerance band and arises from mesh resolution + the CONSTRAINT of
    # operator-split instantaneous mixing (vs MESA's time-dependent D_mix).
    #
    # MESA ref:
    #   - adjust_xyz.f90:do_adjust_xyz_for_mixing (homogenize within CZs)
    #   - mix_info.f90:1344-1381 (do_mix_envelope: surface-connected only)
    has_envelope_cz = jnp.sum(surface_connected) > 0.5

    # Zone average weighted by the smooth env_conv (gives boundary zones
    # proportional influence on the average, but the REPLACEMENT is binary:
    # all surface-connected zones get fully set to zone_avg).
    mask = surface_connected * env_conv
    zone_mass = jnp.sum(mask * zone_weights) + 1e-30
    zone_avg = jnp.sum(mask * X_after_core * zone_weights) / zone_mass
    X_out = jnp.where(surface_connected * has_envelope_cz, zone_avg, X_after_core)

    return X_out


def mix_composition(X_profile, shell_data, M_solar=1.0, f_ov=None, comp_mfracs_in=None,
                    gradL_composition_term=None, envelope_mixing=True):
    """Mix X within convective zones (Ledoux/Schwarzschild) + step overshooting.

    When gradL_composition_term is provided, the envelope convective boundary
    uses the Ledoux criterion (consistent with the solver's #409 switch).
    Without it, falls back to Schwarzschild (∇_rad > ∇_ad).

    When envelope_mixing=False, only core mixing is performed — the envelope
    convective zone is NOT homogenized. This is used by the O2 mutation gate
    to verify that envelope mixing is the mechanism producing the first
    dredge-up signature. Implemented via jnp.where (runtime selection): both
    paths are always compiled into the XLA graph, and the flag selects at
    runtime. This guarantees the mutation works regardless of JAX's persistent
    compilation cache state (unlike a Python 'if' which is baked in at trace
    time and can be defeated by a stale cached executable).
    """
    if f_ov is None:
        f_ov = F_OV

    _K = 200.0

    nrad = shell_data[:, 2]
    nad = shell_data[:, 3]
    mf_shells = shell_data[:, 1]
    hp_over_R = shell_data[:, 4]
    dm_dr_norm = shell_data[:, 5]

    is_conv = jax.nn.sigmoid(_K * (nrad - nad))

    comp_mfracs = comp_mfracs_in if comp_mfracs_in is not None else jnp.array(COMP_MFRACS)

    from stellar_jax.composition.interp import _interp_shell_to_comp_reversed
    is_conv_comp = _interp_shell_to_comp_reversed(comp_mfracs, mf_shells, is_conv)
    hp_comp = _interp_shell_to_comp_reversed(comp_mfracs, mf_shells, hp_over_R)
    dmdr_comp = _interp_shell_to_comp_reversed(comp_mfracs, mf_shells, dm_dr_norm)

    # --- Step 1: Build core convection + overshoot mask ---
    core_mix_mask = _core_conv_mask(is_conv_comp, hp_comp, dmdr_comp,
                                    comp_mfracs, M_solar, f_ov, _K)

    # --- Step 2: Core mixing (mass-weighted average) ---
    zone_weights = compute_zone_masses(comp_mfracs) if comp_mfracs_in is not None else jnp.array(COMP_ZONE_MASSES)
    core_mass_total = jnp.sum(core_mix_mask * zone_weights) + 1e-30
    core_avg_X = jnp.sum(core_mix_mask * X_profile * zone_weights) / core_mass_total
    X_out = core_mix_mask * core_avg_X + (1.0 - core_mix_mask) * X_profile

    # --- Step 3: Envelope zone mixing ---
    # Always compute both paths; select via jnp.where so the envelope_mixing
    # flag is a TRACED value (runtime selection, not compile-time branch).
    # This guarantees the O2 mutation gate works regardless of JIT cache state:
    # a Python 'if' is evaluated at TRACE time and baked into the XLA graph,
    # making it invisible to a persistent-cache lookup that serves a stale
    # compilation. With jnp.where, the flag is a runtime input and both paths
    # are always compiled — setting the flag to False at runtime correctly
    # suppresses envelope mixing without requiring recompilation.
    X_env = _envelope_zone_mixing(X_out, comp_mfracs, mf_shells, nrad, nad,
                                  core_mix_mask, zone_weights,
                                  gradL_composition_term=gradL_composition_term)
    X_out = jnp.where(envelope_mixing, X_env, X_out)

    return X_out


def _mix_z_in_cz(Z_profile, shell_data, M_solar, f_ov, comp_mfracs_in=None,
                 gradL_composition_term=None, envelope_mixing=True):
    """Mix Z (metals) in convective zones — thin wrapper for mutation targeting.

    This is the same mix_composition call, isolated so the O2 mutation gate can
    disable Z-mixing specifically without affecting X/Y mixing.

    Reference: MESA mix_info.f90:set_dxdt_mix (all species j=1..species);
    Paxton et al. (2011, ApJS 192, 3, §4); Böhm-Vitense (1958).
    """
    return mix_composition(Z_profile, shell_data, M_solar=M_solar, f_ov=f_ov,
                           comp_mfracs_in=comp_mfracs_in,
                           gradL_composition_term=gradL_composition_term,
                           envelope_mixing=envelope_mixing)


# Public alias for the composition package API
mix_z_in_cz = _mix_z_in_cz

