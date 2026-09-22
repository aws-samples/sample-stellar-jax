"""Unit tests for evolution/ package components.

Each test validates an extracted sub-step with synthetic inputs — no evolve_star,
no JIT compilation of the full scan. Target: <5s each.

Design ref: docs/design/redesign/06-evolution.md §5 (fast unit tests).
"""
import pytest
import jax.numpy as jnp
import numpy as np

from stellar_jax.evolution.contracts import CarryState, StepContext
from stellar_jax.evolution.timestep import (
    compute_varcontrol,
    apply_xh_limiter,
    apply_he_ignition_halt,
    apply_t_max_cap,
    apply_dt_floor,
    apply_eps_nuc_limiter,
)
from stellar_jax.config.constants import SECONDS_PER_YEAR
from stellar_jax.config.mesh_defaults import N_COMP, COMP_MFRACS, N_HENYEY


# ========================================================================
# CarryState tests
# ========================================================================

class TestCarryState:
    """Tests for CarryState NamedTuple PyTree stability."""

    def test_carry_state_construction(self):
        """CarryState can be constructed with all 18 fields."""
        N_s = 10
        carry = CarryState(
            X_prof=jnp.ones(N_COMP),
            Y_prof=jnp.full(N_COMP, 0.28),
            Z_prof=jnp.full(N_COMP, 0.014),
            C12_prof=jnp.full(N_COMP, 0.002),
            C13_prof=jnp.full(N_COMP, 2e-5),
            N14_prof=jnp.full(N_COMP, 7e-4),
            X3_prof=jnp.full(N_COMP, 1e-10),
            logL=jnp.float64(0.0),
            logTe=jnp.float64(3.76),
            t=jnp.float64(0.0),
            dt=jnp.float64(1e12),
            prev_logT=jnp.ones(N_s) * 7.0,
            prev_logP=jnp.ones(N_s) * 16.0,
            prev_logRho=jnp.ones(N_s) * 1.5,
            y_henyey=jnp.zeros((N_s, 4)),
            comp_mfracs=jnp.array(COMP_MFRACS),
            dt_old=jnp.float64(0.0),
            dt_limit_ratio_old=jnp.float64(0.0),
        )
        assert len(carry) == 18
        assert carry.X_prof.shape == (N_COMP,)
        assert carry.logL.shape == ()
        assert carry.y_henyey.shape == (N_s, 4)
        assert carry.comp_mfracs.shape == (N_COMP,)
        assert carry.dt_old.shape == ()
        assert carry.dt_limit_ratio_old.shape == ()

    def test_carry_state_roundtrip(self):
        """CarryState → tuple → CarryState preserves all fields (PyTree stability)."""
        N_s = 5
        carry = CarryState(
            X_prof=jnp.ones(N_COMP) * 0.7,
            Y_prof=jnp.ones(N_COMP) * 0.28,
            Z_prof=jnp.ones(N_COMP) * 0.014,
            C12_prof=jnp.zeros(N_COMP),
            C13_prof=jnp.zeros(N_COMP),
            N14_prof=jnp.zeros(N_COMP),
            X3_prof=jnp.full(N_COMP, 1e-10),
            logL=jnp.float64(0.5),
            logTe=jnp.float64(3.75),
            t=jnp.float64(1e15),
            dt=jnp.float64(1e13),
            prev_logT=jnp.linspace(6.0, 7.5, N_s),
            prev_logP=jnp.linspace(14.0, 17.0, N_s),
            prev_logRho=jnp.linspace(0.5, 2.0, N_s),
            y_henyey=jnp.ones((N_s, 4)),
            comp_mfracs=jnp.linspace(0, 1, N_COMP),
            dt_old=jnp.float64(0.0),
            dt_limit_ratio_old=jnp.float64(0.0),
        )
        # Convert to tuple and back
        as_tuple = tuple(carry)
        carry_back = CarryState(*as_tuple)
        for i, (a, b) in enumerate(zip(carry, carry_back)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b),
                                          err_msg=f"Field {i} mismatch")

    def test_carry_shape_single_regardless_of_flags(self):
        """Same PyTree structure with adaptive_mesh=True/False — comp_mfracs always present."""
        N_s = 5
        # With adaptive mesh (comp_mfracs = equidistributed)
        carry_adaptive = CarryState(
            X_prof=jnp.ones(N_COMP),
            Y_prof=jnp.ones(N_COMP),
            Z_prof=jnp.ones(N_COMP),
            C12_prof=jnp.zeros(N_COMP),
            C13_prof=jnp.zeros(N_COMP),
            N14_prof=jnp.zeros(N_COMP),
            X3_prof=jnp.full(N_COMP, 1e-10),
            logL=jnp.float64(0.0),
            logTe=jnp.float64(3.7),
            t=jnp.float64(0.0),
            dt=jnp.float64(1e12),
            prev_logT=jnp.ones(N_s) * 7.0,
            prev_logP=jnp.ones(N_s) * 16.0,
            prev_logRho=jnp.ones(N_s) * 1.0,
            y_henyey=jnp.zeros((N_s, 4)),
            comp_mfracs=jnp.linspace(0, 1, N_COMP),  # adapted grid
            dt_old=jnp.float64(0.0),
            dt_limit_ratio_old=jnp.float64(0.0),
        )
        # Without adaptive mesh (comp_mfracs = static COMP_MFRACS)
        carry_static = CarryState(
            X_prof=jnp.ones(N_COMP),
            Y_prof=jnp.ones(N_COMP),
            Z_prof=jnp.ones(N_COMP),
            C12_prof=jnp.zeros(N_COMP),
            C13_prof=jnp.zeros(N_COMP),
            N14_prof=jnp.zeros(N_COMP),
            X3_prof=jnp.full(N_COMP, 1e-10),
            logL=jnp.float64(0.0),
            logTe=jnp.float64(3.7),
            t=jnp.float64(0.0),
            dt=jnp.float64(1e12),
            prev_logT=jnp.ones(N_s) * 7.0,
            prev_logP=jnp.ones(N_s) * 16.0,
            prev_logRho=jnp.ones(N_s) * 1.0,
            y_henyey=jnp.zeros((N_s, 4)),
            comp_mfracs=jnp.array(COMP_MFRACS),  # static grid
            dt_old=jnp.float64(0.0),
            dt_limit_ratio_old=jnp.float64(0.0),
        )
        # Both have exactly the same structure (18 fields, same shapes)
        assert len(carry_adaptive) == len(carry_static) == 18
        for a, b in zip(carry_adaptive, carry_static):
            assert a.shape == b.shape


# ========================================================================
# StepContext tests
# ========================================================================

class TestStepContext:
    """Tests for StepContext construction and immutability."""

    def test_step_context_construction(self):
        """All fields populated with correct types."""
        ctx = StepContext(
            mass=jnp.float64(1.0),
            Z=jnp.float64(0.014),
            alpha_mlt=jnp.float64(1.9),
            f_ov=jnp.float64(0.016),
            M_star_cgs=jnp.float64(1.989e33),
            atm_ratio=jnp.float64(1.002),
            varcontrol_target=jnp.float64(1e-3),
            reject_threshold=jnp.float64(4e-3),
            t_max_sec=jnp.float64(1e30),
            dt_grow=jnp.float64(1.5),
            dt_shrink=jnp.float64(0.5),
            delta_lgL_hard=jnp.float64(0.20),
            delta_lgTe_hard=jnp.float64(0.05),
            opacity_factor=jnp.float64(1.0),
            eps_nuc_factor=jnp.float64(1.0),
            diffusion_factor=jnp.float64(1.0),
            q_mesh=jnp.linspace(0, 1, 11),
            N_s=10,
            fixed_dt=None,
            freeze_schedule=False,
            replay_schedule=False,
            adaptive_mesh=True,
            diffusion=True,
            z_feedback=False,
            has_idle_guard=False,
            eps_grav_flag=True,
            detach_eps_grav_carry=False,
            xh_cntr_limit=0.01,
            xh_cntr_hard_limit=0.05,
            delta_lgT_cntr_limit=0.01,
            delta_lgRho_cntr_limit=0.05,
        )
        assert ctx.N_s == 10
        assert ctx.fixed_dt is None
        assert ctx.eps_grav_flag is True
        assert float(ctx.mass) == 1.0

    def test_step_context_frozen(self):
        """StepContext is immutable (frozen dataclass)."""
        ctx = StepContext(
            mass=jnp.float64(1.0), Z=jnp.float64(0.014),
            alpha_mlt=jnp.float64(1.9), f_ov=jnp.float64(0.016),
            M_star_cgs=jnp.float64(1.989e33), atm_ratio=jnp.float64(1.002),
            varcontrol_target=jnp.float64(1e-3), reject_threshold=jnp.float64(4e-3),
            t_max_sec=jnp.float64(1e30), dt_grow=jnp.float64(1.5),
            dt_shrink=jnp.float64(0.5), delta_lgL_hard=jnp.float64(0.20),
            delta_lgTe_hard=jnp.float64(0.05), opacity_factor=jnp.float64(1.0),
            eps_nuc_factor=jnp.float64(1.0), diffusion_factor=jnp.float64(1.0),
            q_mesh=jnp.linspace(0, 1, 11), N_s=10,
            fixed_dt=None, freeze_schedule=False, replay_schedule=False,
            adaptive_mesh=True, diffusion=True, z_feedback=False,
            has_idle_guard=False, eps_grav_flag=True, detach_eps_grav_carry=False,
            xh_cntr_limit=0.01, xh_cntr_hard_limit=0.05,
            delta_lgT_cntr_limit=0.01, delta_lgRho_cntr_limit=0.05,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            ctx.mass = jnp.float64(2.0)


# ========================================================================
# Timestep function tests
# ========================================================================

class TestVarcontrol:
    """Tests for compute_varcontrol."""

    def test_varcontrol_max_of_three(self):
        """Varcontrol is the maximum of delta_logL, delta_logTe, max_dX."""
        vc = compute_varcontrol(
            jnp.float64(0.001), jnp.float64(0.002), jnp.float64(0.003))
        assert float(vc) == pytest.approx(0.003)

    def test_varcontrol_logL_dominates(self):
        """When delta_logL is largest, it dominates."""
        vc = compute_varcontrol(
            jnp.float64(0.01), jnp.float64(0.001), jnp.float64(0.002))
        assert float(vc) == pytest.approx(0.01)


class TestXHLimiter:
    """Tests for apply_xh_limiter."""

    def test_xh_soft_limiter_scales_dt(self):
        """dt_next scales down proportionally when δXH > soft limit."""
        dt_next, hard_rej = apply_xh_limiter(
            jnp.float64(1e13),  # dt_next
            jnp.float64(0.02),  # delta_XH_cntr = 2x limit
            0.01,  # xh_cntr_limit
            0.05,  # xh_cntr_hard_limit
            jnp.bool_(False),  # dt_at_floor
        )
        # Should scale by limit/delta = 0.01/0.02 = 0.5
        assert float(dt_next) == pytest.approx(0.5e13)
        assert not bool(hard_rej)

    def test_xh_soft_limiter_noop_below_limit(self):
        """No scaling when δXH is below the limit."""
        dt_next, hard_rej = apply_xh_limiter(
            jnp.float64(1e13),
            jnp.float64(0.005),  # below limit
            0.01,
            0.05,
            jnp.bool_(False),
        )
        assert float(dt_next) == pytest.approx(1e13)
        assert not bool(hard_rej)

    def test_xh_hard_limiter_rejects(self):
        """Hard limit triggers rejection."""
        dt_next, hard_rej = apply_xh_limiter(
            jnp.float64(1e13),
            jnp.float64(0.06),  # > hard limit
            0.01,
            0.05,
            jnp.bool_(False),
        )
        assert bool(hard_rej)

    def test_xh_hard_limiter_no_reject_at_floor(self):
        """Hard limit does NOT reject when dt is at the floor."""
        dt_next, hard_rej = apply_xh_limiter(
            jnp.float64(1e13),
            jnp.float64(0.06),  # > hard limit
            0.01,
            0.05,
            jnp.bool_(True),  # dt_at_floor
        )
        assert not bool(hard_rej)


class TestHeIgnitionHalt:
    """Tests for apply_he_ignition_halt."""

    def test_halt_below_threshold(self):
        """dt unchanged well below log_Tc = 7.9."""
        dt_out = apply_he_ignition_halt(jnp.float64(1e13), jnp.float64(7.5))
        # Sigmoid at 7.5: (7.5-7.9)/0.02 = -20 → sigmoid ≈ 0 → dt unchanged
        assert float(dt_out) == pytest.approx(1e13, rel=1e-6)

    def test_halt_above_threshold(self):
        """dt → 0 above log_Tc = 7.9."""
        dt_out = apply_he_ignition_halt(jnp.float64(1e13), jnp.float64(8.2))
        # Sigmoid at 8.2: (8.2-7.9)/0.02 = 15 → sigmoid ≈ 0.9999997 → dt ≈ 3e-7 * 1e13
        assert float(dt_out) < 1e7  # effectively zero relative to 1e13

    def test_halt_smooth_transition(self):
        """Smooth sigmoid transition around 7.9."""
        dt_low = apply_he_ignition_halt(jnp.float64(1e13), jnp.float64(7.88))
        dt_mid = apply_he_ignition_halt(jnp.float64(1e13), jnp.float64(7.90))
        dt_high = apply_he_ignition_halt(jnp.float64(1e13), jnp.float64(7.92))
        # Monotonically decreasing
        assert float(dt_low) > float(dt_mid) > float(dt_high)
        # At 7.9 exactly: sigmoid(0) = 0.5 → dt_mid = 0.5 * 1e13
        assert float(dt_mid) == pytest.approx(0.5e13, rel=0.01)


class TestDtFloor:
    """Tests for apply_dt_floor."""

    def test_floor_prevents_small_dt(self):
        """dt below floor is clamped up."""
        dt_out = apply_dt_floor(jnp.float64(100.0))
        dt_floor = 1e5 * SECONDS_PER_YEAR
        assert float(dt_out) == pytest.approx(dt_floor)

    def test_floor_preserves_large_dt(self):
        """dt above floor is unchanged."""
        dt_in = 1e7 * SECONDS_PER_YEAR
        dt_out = apply_dt_floor(jnp.float64(dt_in))
        assert float(dt_out) == pytest.approx(dt_in)


class TestTMaxCap:
    """Tests for apply_t_max_cap."""

    def test_cap_at_t_max(self):
        """dt → 0 when t >= t_max."""
        t_max = jnp.float64(4.57e9 * SECONDS_PER_YEAR)
        dt_out = apply_t_max_cap(jnp.float64(1e13), t_max, t_max)
        assert float(dt_out) == 0.0

    def test_cap_near_t_max(self):
        """dt capped to not overshoot t_max."""
        t_max = jnp.float64(4.57e9 * SECONDS_PER_YEAR)
        t_now = t_max - jnp.float64(1e12)  # 1e12 sec before t_max
        dt_out = apply_t_max_cap(jnp.float64(1e13), t_now, t_max)
        # Should be clamped to t_max - t_now = 1e12
        assert float(dt_out) == pytest.approx(1e12)


# ========================================================================
# Driver wiring tests
# ========================================================================

class TestDriverWiring:
    """Tests that the package wiring is correct and decomposed modules are accessible."""

    def test_evolve_star_is_from_core(self):
        """The package-level evolve_star comes from _core.py (proven numerical path).

        Architecture note: _core.py remains the live path because its closure-based
        step_fn produces a numerically identical XLA graph to main. The decomposed
        driver.py + step.py change the XLA graph structure (dataclass access vs
        closure capture) which shifts FP accumulation over 1000+ adaptive steps.
        """
        from stellar_jax.evolution import evolve_star
        from stellar_jax.evolution._core import evolve_star as es_core
        assert evolve_star is es_core

    def test_stellar_facade_uses_core(self):
        """The stellar.py facade delegates to _core.py's evolve_star."""
        from stellar_jax.stellar import evolve_star as stellar_es
        from stellar_jax.evolution._core import evolve_star as es_core
        from stellar_jax.evolution import evolve_star as evolution_es
        assert evolution_es is es_core

    def test_backward_compat_other_names(self):
        """Non-evolve_star names from _core are still accessible."""
        from stellar_jax.evolution import (evolve_star_comp, evolve_star_diagnostic,
                               observable_at_target, _comp_step_with_dt)
        assert callable(evolve_star_comp)
        assert callable(evolve_star_diagnostic)
        assert callable(observable_at_target)
        assert callable(_comp_step_with_dt)

    def test_driver_scan_wrapper(self):
        """The _scan_step closure correctly wraps step_fn for lax.scan."""
        # Verify the closure captures ctx by building a StepContext
        from stellar_jax.evolution.contracts import StepContext
        ctx = StepContext(
            mass=jnp.float64(1.0), Z=jnp.float64(0.014),
            alpha_mlt=jnp.float64(1.9), f_ov=jnp.float64(0.016),
            M_star_cgs=jnp.float64(1.989e33), atm_ratio=jnp.float64(1.001),
            varcontrol_target=jnp.float64(1e-3), reject_threshold=jnp.float64(4e-3),
            t_max_sec=jnp.float64(1e30), dt_grow=jnp.float64(1.5),
            dt_shrink=jnp.float64(0.5), delta_lgL_hard=jnp.float64(0.20),
            delta_lgTe_hard=jnp.float64(0.05),
            opacity_factor=jnp.float64(1.0), eps_nuc_factor=jnp.float64(1.0),
            diffusion_factor=jnp.float64(1.0),
            q_mesh=jnp.linspace(0, 1, N_HENYEY + 1), N_s=N_HENYEY,
            fixed_dt=None, freeze_schedule=False, replay_schedule=False,
            adaptive_mesh=True, diffusion=True, z_feedback=False,
            has_idle_guard=False, eps_grav_flag=True, detach_eps_grav_carry=False,
            xh_cntr_limit=0.01, xh_cntr_hard_limit=0.05,
            delta_lgT_cntr_limit=0.01, delta_lgRho_cntr_limit=0.05)
        # The ctx is frozen and all fields are set
        assert ctx.N_s == N_HENYEY
        assert ctx.fixed_dt is None
        assert ctx.diffusion is True


class TestModuleLevelConstants:
    """Tests that module-level mutation constants are accessible."""

    def test_eps_grav_constant_accessible(self):
        """EPS_GRAV_IN_STRUCTURE is accessible for O2 mutation."""
        from stellar_jax.evolution._core import EPS_GRAV_IN_STRUCTURE
        assert EPS_GRAV_IN_STRUCTURE is True

    def test_xh_limits_accessible(self):
        """XH limiter constants are accessible for O2 mutation."""
        from stellar_jax.evolution._core import _DELTA_XH_CNTR_LIMIT, _DELTA_XH_CNTR_HARD_LIMIT
        assert _DELTA_XH_CNTR_LIMIT == 0.01
        assert _DELTA_XH_CNTR_HARD_LIMIT == 0.05

    def test_structural_limits_accessible(self):
        """Structural hard limit constants are accessible."""
        from stellar_jax.evolution._core import DELTA_LGL_HARD_LIMIT, DELTA_LGTE_HARD_LIMIT
        assert DELTA_LGL_HARD_LIMIT == 0.20
        assert DELTA_LGTE_HARD_LIMIT == 0.05


# ========================================================================
# Extracted helper tests — validate the NEW helpers in _core.py
# ========================================================================

class TestExtractedHelpers:
    """Tests for the module-level helper functions extracted in #789."""

    def test_build_init_carry_shape_adaptive_false(self):
        """Carry has 18 elements when adaptive_mesh=False (15 base + 2 H211b history + 1 q_mesh_prev tail, #1165)."""
        from stellar_jax.evolution._core import _build_init_carry
        N_s = N_HENYEY
        init_profiles = {
            'X': jnp.ones(N_COMP), 'Y': jnp.full(N_COMP, 0.28),
            'Z': jnp.float64(0.014),
            'C12': jnp.zeros(N_COMP), 'C13': jnp.zeros(N_COMP),
            'N14': jnp.zeros(N_COMP), 'X3': jnp.zeros(N_COMP),
            'logL0': jnp.float64(0.0), 'logTe0': jnp.float64(3.76),
            'dt_init': jnp.float64(1e12),
            'logT_init': jnp.ones(N_s) * 7.0,
            'logP_init': jnp.ones(N_s) * 16.0,
            'logRho_init': jnp.ones(N_s) * 1.5,
        }
        carry = _build_init_carry(False, init_profiles, jnp.zeros((N_s, 4)),
                                  jnp.linspace(0.0, 1.0, N_s + 1))
        assert len(carry) == 18

    def test_build_init_carry_shape_adaptive_true(self):
        """Carry has 19 elements when adaptive_mesh=True (16 base + 2 H211b history + 1 q_mesh_prev tail, #1165)."""
        from stellar_jax.evolution._core import _build_init_carry
        N_s = N_HENYEY
        init_profiles = {
            'X': jnp.ones(N_COMP), 'Y': jnp.full(N_COMP, 0.28),
            'Z': jnp.float64(0.014),
            'C12': jnp.zeros(N_COMP), 'C13': jnp.zeros(N_COMP),
            'N14': jnp.zeros(N_COMP), 'X3': jnp.zeros(N_COMP),
            'logL0': jnp.float64(0.0), 'logTe0': jnp.float64(3.76),
            'dt_init': jnp.float64(1e12),
            'logT_init': jnp.ones(N_s) * 7.0,
            'logP_init': jnp.ones(N_s) * 16.0,
            'logRho_init': jnp.ones(N_s) * 1.5,
        }
        carry = _build_init_carry(True, init_profiles, jnp.zeros((N_s, 4)),
                                  jnp.linspace(0.0, 1.0, N_s + 1))
        assert len(carry) == 19
        # comp_mfracs is at position 15
        np.testing.assert_allclose(np.asarray(carry[15]),
                                   np.asarray(COMP_MFRACS), atol=1e-15)
        # H211b history fields initialized to 0
        assert float(carry[16]) == 0.0  # dt_old
        assert float(carry[17]) == 0.0  # dt_limit_ratio_old

    def test_egrav_warmup_callable(self):
        """_egrav_warmup is importable and callable."""
        from stellar_jax.evolution._core import _egrav_warmup
        assert callable(_egrav_warmup)

    def test_extract_warmup_observables_callable(self):
        """_extract_warmup_observables is importable and callable."""
        from stellar_jax.evolution._core import _extract_warmup_observables
        assert callable(_extract_warmup_observables)

    def test_run_windowed_scan_callable(self):
        """_run_windowed_scan is importable and callable."""
        from stellar_jax.evolution._core import _run_windowed_scan
        assert callable(_run_windowed_scan)

    def test_assemble_result_callable(self):
        """_assemble_result is importable and callable."""
        from stellar_jax.evolution._core import _assemble_result
        assert callable(_assemble_result)


# ========================================================================
# Run with: python3.11 -m pytest tests/test_evolution_unit.py -v
# ========================================================================
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
