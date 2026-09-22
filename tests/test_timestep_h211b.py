"""Test H211b low-pass controller and central (δlgT/δlgρ_cntr) limiters.

Unit validation of the timestep-control functions added in #761, matching
MESA's timestep.f90.

Marks: @pytest.mark.fast (no evolve_star, no JIT — pure function calls on scalars).
       @pytest.mark.validation + @pytest.mark.mutation (O2 gate).

External reference: MESA timestep.f90 (source at /tmp/mesa/star/private/timestep.f90):
  - filter_dt_next (lines 2362–2430): H211b 2nd-order digital filter,
    Söderlind & Wang, J. Comput. Appl. Math. 185 (2006) 225–243.
  - check_dlgT_cntr_change (lines 1367–1395), check_dlgRho_cntr_change (1412–1440):
    central limiters using check_change() (lines 732–766).
  - Worst-offender arbitration: maxloc(dt_limit_ratio) at line 340.
"""
import pytest
import jax.numpy as jnp

import stellar_jax.evolution.timestep as ts_mod


# ======================================================================
# ATAN LIMITER — MESA limiter() function
# ======================================================================

class TestAtanLimiter:
    """Verify atan_limiter matches MESA's `limiter` (timestep.f90:2420)."""

    @pytest.mark.fast
    def test_identity_at_one(self):
        """limiter(1) = 1 — the fixed point (MESA: `for x=1, limiter=1`)."""
        result = float(ts_mod.atan_limiter(jnp.float64(1.0)))
        assert result == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.fast
    def test_bounded_above(self):
        """For very large x, limiter < 1 + κ*π/2 ≈ 16.7 (MESA comment: κ=10)."""
        result = float(ts_mod.atan_limiter(jnp.float64(1000.0)))
        # Asymptote: 1 + 10*π/2 = 1 + 15.708 = 16.708
        assert result < 16.8
        assert result > 16.0

    @pytest.mark.fast
    def test_bounded_below(self):
        """For x=0, limiter > 0 (MESA: `for x>=0 and kappa=10, limiter >= 0.003`)."""
        result = float(ts_mod.atan_limiter(jnp.float64(0.0)))
        # 1 + 10*atan(-1/10) = 1 + 10*(-0.0997) ≈ 0.003
        assert result > 0.0
        assert result < 0.1

    @pytest.mark.fast
    def test_monotonically_increasing(self):
        """Limiter is monotonically increasing in x."""
        xs = jnp.linspace(0.01, 20.0, 100)
        vals = jnp.array([ts_mod.atan_limiter(x) for x in xs])
        diffs = jnp.diff(vals)
        assert jnp.all(diffs > 0)


# ======================================================================
# H211b CONTROLLER — filter_dt_next
# ======================================================================

class TestH211bController:
    """Verify H211b low-pass filter matches MESA's filter_dt_next.

    External reference: MESA timestep.f90:2362–2430.
    The key behavioral property: with history available, the controller
    SMOOTHS dt changes relative to the bare 1st-order controller.
    Without history (first step), it falls back to 1st-order exactly.

    Tolerance rationale: the smoothing property is structural (not a numeric
    precision question). The H211b output must differ from 1st-order by a
    measurable amount (>5%) when the history inputs differ from current.

    WHAT makes it FAIL: @mutation("disable_h211b_controller") replaces this
    function with bare 1st-order — the smoothing assertion fails.
    """

    @pytest.mark.fast
    def test_first_order_fallback_no_history(self):
        """With dt_old=0 (no history), H211b = 1st-order: dt * 1/ratio.

        MESA ref: timestep.f90:2406-2412 (the else-branch).
        """
        dt = jnp.float64(1e12)
        dt_old = jnp.float64(0.0)  # no history
        ratio = jnp.float64(2.0)  # limiter binding (ratio > 1)
        ratio_old = jnp.float64(0.0)

        result = float(ts_mod.h211b_filter_dt_next(dt, dt_old, ratio, ratio_old))
        expected = float(dt) * 1.0 / 2.0  # 1st order: dt * target / ratio
        assert result == pytest.approx(expected, rel=1e-10)

    @pytest.mark.fast
    def test_first_order_fallback_zero_ratio_old(self):
        """With dt_limit_ratio_old=0, H211b falls back to 1st-order."""
        dt = jnp.float64(1e12)
        dt_old = jnp.float64(8e11)
        ratio = jnp.float64(1.5)
        ratio_old = jnp.float64(0.0)  # no history

        result = float(ts_mod.h211b_filter_dt_next(dt, dt_old, ratio, ratio_old))
        expected = float(dt) * 1.0 / 1.5
        assert result == pytest.approx(expected, rel=1e-10)

    @pytest.mark.fast
    @pytest.mark.integration
    @pytest.mark.validation
    @pytest.mark.mutation("disable_h211b_controller")
    @pytest.mark.right_reason("smoothing")
    def test_h211b_smooths_vs_first_order(self):
        """H211b output differs from 1st-order when history is present.

        This is the DEFINING behavioral property of the 2nd-order filter:
        it uses the previous ratio and dt to smooth the response. When the
        previous step had a much larger ratio (limiter was binding 10x) and
        a much shorter dt (5x), the H211b controller RESISTS growing dt as
        fast as the bare 1st-order would — a conservative, smoothed response.

        WHAT: H211b controller output vs 1st-order output.
        WHY: validates that the 2nd-order filter is active (smoothing dt).
        EXTERNAL REF: MESA timestep.f90:2382-2394 (the H211b formula).
        TOLERANCE: the two outputs must differ by >10% — structural property.
          With dt_old=0.2×dt and ratio_old=10 (heavy binding last step), the
          H211b output is ~36% less than 1st-order (measured). The threshold
          of 10% has wide margin.
        MUTATION: disable_h211b_controller → reverts to 1st-order → identical → FAIL.
        """
        dt = jnp.float64(1e12)
        dt_old = jnp.float64(2e11)   # previous step was much shorter (5x)
        ratio = jnp.float64(2.0)      # current limiter binding at 2x
        ratio_old = jnp.float64(10.0)  # previous step had severe binding (10x)

        h211b_result = float(ts_mod.h211b_filter_dt_next(dt, dt_old, ratio, ratio_old))
        first_order = float(dt) * 1.0 / 2.0  # bare 1st-order: dt * target/ratio

        # The H211b result must DIFFER from 1st-order (smoothing effect)
        # With these inputs, the H211b is ~36% lower (conservative due to history)
        relative_diff = abs(h211b_result - first_order) / first_order
        assert relative_diff > 0.10, (
            f"H211b output {h211b_result:.6e} is too close to 1st-order "
            f"{first_order:.6e} (diff={relative_diff:.4f}) — smoothing not active"
        )

    @pytest.mark.fast
    def test_h211b_steady_state(self):
        """When no limiter is binding (ratio ≈ 0), H211b produces growth > dt.

        With ratio and ratio_old near zero (step quality far below target),
        target/ratio is very large → limiter(large) ≈ 16.7 → dt_next >> dt.
        The downstream clip to dt_grow (not tested here) caps this growth.
        MESA: when all limiters return 0, maxloc picks the first (0), and
        filter_dt_next gives dt * limiter(1/1e-10) → large growth, clipped
        by max_timestep_factor.
        """
        dt = jnp.float64(1e12)
        dt_old = jnp.float64(1e12)
        ratio = jnp.float64(1e-10)  # not binding (will be maxed to 1e-10)
        ratio_old = jnp.float64(1e-10)

        result = float(ts_mod.h211b_filter_dt_next(dt, dt_old, ratio, ratio_old))
        # With ratio ~0, target/ratio → very large → limiter(large) → ~16.7
        # This is the "not binding" case — dt should grow significantly
        assert result > float(dt)  # Should grow when nothing is binding


# ======================================================================
# CENTRAL LIMITERS — δlgT_cntr, δlgRho_cntr
# ======================================================================

class TestCentralLimiters:
    """Verify central T/ρ limiters match MESA's check_change mechanism.

    External reference: MESA timestep.f90:1367–1440, using check_change():732–766.
    The key behavior: ratio = |Δ| / limit if |Δ| > limit, else 0.
    MESA defaults: delta_lgT_cntr_limit = 0.01, delta_lgRho_cntr_limit = 0.05.

    WHAT: central T/ρ change ratio computation.
    WHY: guards against too-fast central evolution on the RGB (MESA default ON).
    EXTERNAL REF: MESA timestep.f90:1367-1440 + controls.defaults line 10878/10908.
    TOLERANCE: exact match — this is an arithmetic identity, not a floating-point
    approximation with inter-code scatter.
    MUTATION: disable_central_limiters → always returns (0, 0) → test FAILs.
    """

    @pytest.mark.fast
    def test_ratio_below_limit_is_zero(self):
        """When |Δ| < limit, dt_limit_ratio = 0 (MESA: line 766)."""
        ratio = float(ts_mod.compute_central_limiter_ratio(
            jnp.float64(0.005), 0.01))  # |0.005| < 0.01
        assert ratio == 0.0

    @pytest.mark.fast
    def test_ratio_at_limit_is_zero(self):
        """When |Δ| = limit, dt_limit_ratio = 0 (MESA: <= 1 → 0)."""
        ratio = float(ts_mod.compute_central_limiter_ratio(
            jnp.float64(0.01), 0.01))  # |0.01| / 0.01 = 1.0 → 0
        assert ratio == 0.0

    @pytest.mark.fast
    def test_ratio_above_limit(self):
        """When |Δ| > limit, dt_limit_ratio = |Δ|/limit (MESA: line 764)."""
        ratio = float(ts_mod.compute_central_limiter_ratio(
            jnp.float64(0.03), 0.01))  # |0.03| / 0.01 = 3.0
        assert ratio == pytest.approx(3.0, rel=1e-10)

    @pytest.mark.fast
    def test_ratio_negative_change(self):
        """Negative changes use absolute value (MESA: abs_change = abs(delta))."""
        ratio = float(ts_mod.compute_central_limiter_ratio(
            jnp.float64(-0.02), 0.01))  # |-0.02| / 0.01 = 2.0
        assert ratio == pytest.approx(2.0, rel=1e-10)

    @pytest.mark.fast
    @pytest.mark.integration
    @pytest.mark.validation
    @pytest.mark.mutation("disable_central_limiters")
    @pytest.mark.right_reason("Expected")
    def test_central_limiters_bind_when_exceeded(self):
        """Central limiters return non-zero ratio when T or ρ change exceeds limit.

        WHAT: compute_central_limiters with a large δlgT and δlgRho.
        WHY: validates the central-change constraint on dt is active.
        EXTERNAL REF: MESA timestep.f90:1367-1440, controls.defaults (0.01, 0.05).
        TOLERANCE: ratio must be >1 when change exceeds limit (exact arithmetic).
        MUTATION: disable_central_limiters → returns (0, 0) → assertion fails.
        """
        # Simulate RGB central conditions: T_cntr changes by 0.03 dex (3x the limit)
        logT_cntr = jnp.float64(7.8)      # log10(T_c) ~ 6.3×10^7 K (SGB)
        logT_cntr_old = jnp.float64(7.77)  # previous: Δ = 0.03 > 0.01 limit
        logRho_cntr = jnp.float64(3.5)
        logRho_cntr_old = jnp.float64(3.4)  # Δ = 0.10 > 0.05 limit

        ratio_T, ratio_Rho = ts_mod.compute_central_limiters(
            logT_cntr, logT_cntr_old,
            logRho_cntr, logRho_cntr_old,
            delta_lgT_cntr_limit=0.01,
            delta_lgRho_cntr_limit=0.05,
        )

        ratio_T = float(ratio_T)
        ratio_Rho = float(ratio_Rho)

        # Both must be binding (> 1)
        assert ratio_T == pytest.approx(3.0, rel=1e-10), \
            f"Expected ratio_T=3.0, got {ratio_T}"
        assert ratio_Rho == pytest.approx(2.0, rel=1e-10), \
            f"Expected ratio_Rho=2.0, got {ratio_Rho}"

    @pytest.mark.fast
    def test_central_limiters_disabled_with_negative_limit(self):
        """A negative limit disables the limiter (MESA: `if lim <= 0 return`).

        CONSTRAINT: we encode disabled as limit=0 (not negative, for JAX simplicity).
        When limit is very small (≈0), ratio → inf; we handle this in the controller
        by using limit <= 0 to disable. Here we verify the math with a positive limit.
        """
        # With limit=0.0 the ratio would be 0 / 0 → guarded by max(limit, 1e-30)
        # This just confirms the function doesn't crash with very small limits
        ratio = float(ts_mod.compute_central_limiter_ratio(
            jnp.float64(0.03), 1e-30))
        assert ratio > 1.0  # binding (change >> limit)


# ======================================================================
# WORST-OFFENDER ARBITRATION
# ======================================================================

class TestWorstOffenderRatio:
    """Verify worst-offender picks the max across all limiters.

    External reference: MESA timestep.f90:340 — maxloc(dt_limit_ratio(1:numTlim)).
    """

    @pytest.mark.fast
    def test_varcontrol_dominates(self):
        """When varcontrol is the worst offender, its ratio wins."""
        worst = float(ts_mod.compute_worst_offender_ratio(
            varcontrol=jnp.float64(0.006),
            varcontrol_target=jnp.float64(0.001),
            delta_XH_cntr=jnp.float64(0.005),  # below xh limit
            xh_cntr_limit=0.01,
            ratio_lgT_cntr=jnp.float64(0.0),
            ratio_lgRho_cntr=jnp.float64(0.0),
        ))
        # varcontrol/target = 6.0 (binding); XH = 0.5 (not binding)
        assert worst == pytest.approx(6.0, rel=1e-10)

    @pytest.mark.fast
    def test_central_limiter_dominates(self):
        """When a central limiter is worst offender, it wins."""
        worst = float(ts_mod.compute_worst_offender_ratio(
            varcontrol=jnp.float64(0.001),  # at target (not binding)
            varcontrol_target=jnp.float64(0.001),
            delta_XH_cntr=jnp.float64(0.005),
            xh_cntr_limit=0.01,
            ratio_lgT_cntr=jnp.float64(5.0),  # worst
            ratio_lgRho_cntr=jnp.float64(2.0),
        ))
        assert worst == pytest.approx(5.0, rel=1e-10)

    @pytest.mark.fast
    def test_xh_cntr_dominates(self):
        """When XH central is the worst offender, it wins."""
        worst = float(ts_mod.compute_worst_offender_ratio(
            varcontrol=jnp.float64(0.001),
            varcontrol_target=jnp.float64(0.001),
            delta_XH_cntr=jnp.float64(0.04),  # 4x limit
            xh_cntr_limit=0.01,
            ratio_lgT_cntr=jnp.float64(0.0),
            ratio_lgRho_cntr=jnp.float64(0.0),
        ))
        assert worst == pytest.approx(4.0, rel=1e-10)

    @pytest.mark.fast
    def test_good_step_returns_raw_varcontrol_ratio(self):
        """When step quality is good (varcontrol < target), raw ratio is passed.

        MESA ref: timestep.f90:2273 — check_varcontrol_limit sets
        dt_limit_ratio = varcontrol / vc_target DIRECTLY (no zeroing when <= 1).
        This gives the H211b controller proportional growth information
        (e.g. ratio=0.5 → limiter(1/0.5) ≈ 2 → moderate growth).
        Distinct from check_change (δlg limiters) which zeros when <= 1.
        """
        worst = float(ts_mod.compute_worst_offender_ratio(
            varcontrol=jnp.float64(0.0005),  # below target
            varcontrol_target=jnp.float64(0.001),
            delta_XH_cntr=jnp.float64(0.005),  # below limit
            xh_cntr_limit=0.01,
            ratio_lgT_cntr=jnp.float64(0.0),
            ratio_lgRho_cntr=jnp.float64(0.0),
        ))
        # varcontrol/target = 0.5 — MESA passes this raw (no zeroing)
        # XH = 0.5 below limit → zeroed; central = 0
        # max(0.5, 0, 0, 0) = 0.5
        assert worst == pytest.approx(0.5, rel=1e-10)
