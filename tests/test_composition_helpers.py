"""Fast unit tests for composition/ decomposition helpers (issue #530).

Tests the helpers extracted from diffuse_composition and mix_composition:
- composition.interp: _interp_shell_to_comp, _interp_shell_to_comp_reversed
- composition.diffuse: _compute_diffusion_structure, _advective_settling_flux,
                       _concentration_gradient_flux_3sp
- composition.mix: _core_conv_mask, _envelope_zone_mixing

All tests use SYNTHETIC inputs (no evolve_star, no heavy JIT) and run in <5s.
Mark: @pytest.mark.fast
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from stellar_jax.config.mesh_defaults import N_COMP, COMP_MFRACS, COMP_ZONE_MASSES


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def small_comp_mfracs():
    """Small 20-zone composition grid for fast tests."""
    return jnp.linspace(0.0, 1.0, 20)


@pytest.fixture
def small_shell_data():
    """Small synthetic shell_data (20 shells, surface-to-center).

    Columns: [eps_nuc, m/M, nabla_rad, nabla_ad, Hp/R, dm_dr_norm, logT, logrho]
    - Convective core for inner 30% by mass (nabla_rad > nabla_ad)
    - Radiative envelope otherwise
    """
    n = 20
    # Surface-to-center: m/M goes from ~1 to ~0
    mf = np.linspace(0.95, 0.01, n)  # descending (surface → center)
    eps_nuc = 10.0 * np.exp(-mf / 0.05)
    nabla_ad = np.full(n, 0.4)
    # Convective where mf < 0.30
    nabla_rad = np.where(mf < 0.30, 0.8, 0.25)
    hp_over_R = 0.1 * (1.0 - mf) + 0.01
    dm_dr_norm = np.ones(n) * 3.0
    logT = 3.78 + (7.17 - 3.78) * (1.0 - mf)  # hotter toward center
    logrho = -7.0 + (2.18 + 7.0) * (1.0 - mf)  # denser toward center

    shell_data = np.column_stack([eps_nuc, mf, nabla_rad, nabla_ad,
                                  hp_over_R, dm_dr_norm, logT, logrho])
    return jnp.array(shell_data)


# ===========================================================================
# §1. Interpolation helpers
# ===========================================================================

@pytest.mark.fast
class TestInterp:
    """Tests for composition.interp helpers."""

    def test_interp_shell_to_comp_basic(self, small_comp_mfracs, small_shell_data):
        """Interpolation maps shell quantities onto comp grid correctly."""
        from stellar_jax.composition.interp import _interp_shell_to_comp
        mf_shells = small_shell_data[:, 1]
        logT_shells = small_shell_data[:, 6]
        result = _interp_shell_to_comp(small_comp_mfracs, mf_shells, logT_shells)
        # Result should be monotonically increasing (hotter toward center = mf→0)
        assert result.shape == (20,)
        # At the center (comp_mfracs=0), T should be highest
        assert float(result[0]) > float(result[-1])

    def test_interp_reversed_matches_argsort(self, small_comp_mfracs, small_shell_data):
        """Both interpolation variants produce identical results on descending data."""
        from stellar_jax.composition.interp import _interp_shell_to_comp, _interp_shell_to_comp_reversed
        mf_shells = small_shell_data[:, 1]  # descending
        logT_shells = small_shell_data[:, 6]
        result_argsort = _interp_shell_to_comp(small_comp_mfracs, mf_shells, logT_shells)
        result_reversed = _interp_shell_to_comp_reversed(small_comp_mfracs, mf_shells, logT_shells)
        np.testing.assert_allclose(result_argsort, result_reversed, atol=1e-14)

    def test_interp_preserves_range(self, small_comp_mfracs, small_shell_data):
        """Interpolated values stay within the input range (no extrapolation overshoot)."""
        from stellar_jax.composition.interp import _interp_shell_to_comp
        mf_shells = small_shell_data[:, 1]
        eps = small_shell_data[:, 0]
        result = _interp_shell_to_comp(small_comp_mfracs, mf_shells, eps)
        assert float(jnp.min(result)) >= float(jnp.min(eps)) - 1e-10
        assert float(jnp.max(result)) <= float(jnp.max(eps)) + 1e-10


# ===========================================================================
# §2. Diffuse helpers
# ===========================================================================

@pytest.mark.fast
class TestDiffuseHelpers:
    """Tests for composition.diffuse decomposition helpers."""

    def test_compute_diffusion_structure_shapes(self, small_shell_data):
        """Structure computation returns all required fields with correct shapes."""
        from stellar_jax.composition.diffuse import _compute_diffusion_structure
        n = 20
        comp_mfracs = jnp.linspace(0.0, 1.0, n)
        X = jnp.full(n, 0.70)
        Y = jnp.full(n, 0.28)
        Z = jnp.full(n, 0.02)
        struct = _compute_diffusion_structure(X, Y, Z, small_shell_data,
                                             1.0, comp_mfracs, False)
        assert struct['T_prof'].shape == (n,)
        assert struct['rho_prof'].shape == (n,)
        assert struct['dlnP_dr'].shape == (n,)
        assert struct['dlnT_dr'].shape == (n,)
        assert struct['flux_area'].shape == (n,)
        assert struct['dm'].shape == (n,)
        assert struct['cz_taper'].shape == (n,)
        # T should be positive
        assert float(jnp.min(struct['T_prof'])) > 0
        # dm should sum to M_star
        from stellar_jax.config.constants import Msun
        np.testing.assert_allclose(float(jnp.sum(struct['dm'])),
                                   1.0 * Msun, rtol=1e-10)

    def test_advective_settling_flux_conservation(self):
        """Advective settling flux conserves total mass (sum of dY × dm = 0)."""
        from stellar_jax.composition.diffuse import _advective_settling_flux
        n = 50
        # Uniform Y, uniform inward velocity
        Y = jnp.full(n, 0.28)
        v_He = jnp.full(n, -1e-7)  # inward settling
        cz_taper = jnp.ones(n)  # fully radiative (no suppression)
        flux_area = jnp.ones(n) * 1e20
        dm = jnp.ones(n) * 1e30
        dt = 1e13
        v_max = 1e-6
        dY = _advective_settling_flux(v_He, Y, cz_taper, flux_area, dm, dt, v_max)
        # Total He mass change should be zero (flux in = flux out, no boundaries)
        # Not exactly zero due to boundary conditions, but integral should be small
        assert dY.shape == (n,)
        # Interior zones should see settling (dY < 0 at surface, > 0 at center)
        # The sum is 0 because no flux crosses outer boundaries
        total_mass_change = float(jnp.sum(dY * dm))
        assert abs(total_mass_change) < 1e-10 * float(jnp.sum(jnp.abs(dY * dm)) + 1e-30)

    def test_advective_settling_flux_suppressed_in_cz(self):
        """Settling velocity is zero in convective zones."""
        from stellar_jax.composition.diffuse import _advective_settling_flux
        n = 20
        Y = jnp.full(n, 0.28)
        v_He = jnp.full(n, -5e-7)  # strong settling
        # All zones convective (taper = 0 → fully suppressed)
        cz_taper = jnp.zeros(n)
        flux_area = jnp.ones(n) * 1e20
        dm = jnp.ones(n) * 1e30
        dt = 1e13
        v_max = 1e-6
        dY = _advective_settling_flux(v_He, Y, cz_taper, flux_area, dm, dt, v_max)
        # All convective → no settling → dY = 0
        np.testing.assert_allclose(dY, 0.0, atol=1e-30)

    def test_concentration_gradient_flux_3sp_shapes(self):
        """3-species concentration gradient flux has correct output shape."""
        from stellar_jax.composition.diffuse import _concentration_gradient_flux_3sp
        n = 20
        X = jnp.linspace(0.35, 0.70, n)
        Y = jnp.linspace(0.30, 0.28, n)
        sig_He_H = jnp.ones(n) * 1e-7
        sig_He_He = jnp.ones(n) * (-1e-7)
        cz_taper = jnp.ones(n)  # fully radiative
        from stellar_jax.config.constants import Rsun
        r_prof = jnp.linspace(0.01 * Rsun, 0.9 * Rsun, n)
        flux_area = jnp.ones(n) * 1e20
        dm = jnp.ones(n) * 1e30
        dt = 1e13
        v_max = 1e-6
        dY_conc = _concentration_gradient_flux_3sp(
            X, Y, sig_He_H, sig_He_He, cz_taper, r_prof, flux_area, dm, dt, v_max)
        assert dY_conc.shape == (n,)
        # Should be non-zero since X/Y have gradients
        assert float(jnp.max(jnp.abs(dY_conc))) > 0


    def test_concentration_gradient_flux_4sp_shapes(self):
        """4-species concentration gradient flux returns correct shapes for dY and dZ."""
        from stellar_jax.composition.diffuse import _concentration_gradient_flux_4sp
        n = 20
        X = jnp.linspace(0.35, 0.70, n)
        Y = jnp.linspace(0.30, 0.28, n)
        Z = jnp.full(n, 0.02)
        # Sigma coefficients (6 total for He and Z responses)
        sig_He_H = jnp.ones(n) * 1e-7
        sig_He_He = jnp.ones(n) * (-1e-7)
        sig_He_Z = jnp.ones(n) * 5e-8
        sig_Z_H = jnp.ones(n) * 2e-7
        sig_Z_He = jnp.ones(n) * (-5e-8)
        sig_Z_Z = jnp.ones(n) * (-2e-7)
        cz_taper = jnp.ones(n)  # fully radiative
        from stellar_jax.config.constants import Rsun
        r_prof = jnp.linspace(0.01 * Rsun, 0.9 * Rsun, n)
        flux_area = jnp.ones(n) * 1e20
        dm = jnp.ones(n) * 1e30
        dt = 1e13
        v_max = 1e-6
        dY_conc, dZ_conc = _concentration_gradient_flux_4sp(
            X, Y, Z, sig_He_H, sig_He_He, sig_He_Z,
            sig_Z_H, sig_Z_He, sig_Z_Z,
            cz_taper, r_prof, flux_area, dm, dt, v_max)
        assert dY_conc.shape == (n,)
        assert dZ_conc.shape == (n,)
        # Should be non-zero since X/Y/Z have gradients
        assert float(jnp.max(jnp.abs(dY_conc))) > 0
        assert float(jnp.max(jnp.abs(dZ_conc))) > 0

    def test_concentration_gradient_flux_4sp_suppressed_in_cz(self):
        """4-species concentration gradient flux is zero in fully convective regions."""
        from stellar_jax.composition.diffuse import _concentration_gradient_flux_4sp
        n = 20
        X = jnp.linspace(0.35, 0.70, n)
        Y = jnp.linspace(0.30, 0.28, n)
        Z = jnp.full(n, 0.02)
        sig_He_H = jnp.ones(n) * 1e-7
        sig_He_He = jnp.ones(n) * (-1e-7)
        sig_He_Z = jnp.ones(n) * 5e-8
        sig_Z_H = jnp.ones(n) * 2e-7
        sig_Z_He = jnp.ones(n) * (-5e-8)
        sig_Z_Z = jnp.ones(n) * (-2e-7)
        # ALL zones convective (taper = 0 → fully suppressed)
        cz_taper = jnp.zeros(n)
        from stellar_jax.config.constants import Rsun
        r_prof = jnp.linspace(0.01 * Rsun, 0.9 * Rsun, n)
        flux_area = jnp.ones(n) * 1e20
        dm = jnp.ones(n) * 1e30
        dt = 1e13
        v_max = 1e-6
        dY_conc, dZ_conc = _concentration_gradient_flux_4sp(
            X, Y, Z, sig_He_H, sig_He_He, sig_He_Z,
            sig_Z_H, sig_Z_He, sig_Z_Z,
            cz_taper, r_prof, flux_area, dm, dt, v_max)
        # Fully convective → suppressed → zero
        np.testing.assert_allclose(dY_conc, 0.0, atol=1e-30)
        np.testing.assert_allclose(dZ_conc, 0.0, atol=1e-30)

    def test_concentration_gradient_flux_4sp_differentiable(self):
        """jax.grad flows through _concentration_gradient_flux_4sp w.r.t. Y."""
        from stellar_jax.composition.diffuse import _concentration_gradient_flux_4sp
        n = 20
        X = jnp.linspace(0.35, 0.70, n)
        Z = jnp.full(n, 0.02)
        sig_He_H = jnp.ones(n) * 1e-7
        sig_He_He = jnp.ones(n) * (-1e-7)
        sig_He_Z = jnp.ones(n) * 5e-8
        sig_Z_H = jnp.ones(n) * 2e-7
        sig_Z_He = jnp.ones(n) * (-5e-8)
        sig_Z_Z = jnp.ones(n) * (-2e-7)
        cz_taper = jnp.ones(n)  # fully radiative
        from stellar_jax.config.constants import Rsun
        r_prof = jnp.linspace(0.01 * Rsun, 0.9 * Rsun, n)
        flux_area = jnp.ones(n) * 1e20
        dm = jnp.ones(n) * 1e30
        dt = 1e13
        v_max = 1e-6

        def _sum_fluxes(Y):
            dY_conc, dZ_conc = _concentration_gradient_flux_4sp(
                X, Y, Z, sig_He_H, sig_He_He, sig_He_Z,
                sig_Z_H, sig_Z_He, sig_Z_Z,
                cz_taper, r_prof, flux_area, dm, dt, v_max)
            return jnp.sum(dY_conc**2 + dZ_conc**2)

        Y = jnp.linspace(0.30, 0.28, n)
        grad = jax.grad(_sum_fluxes)(Y)
        assert grad.shape == (n,)
        assert float(jnp.max(jnp.abs(grad))) > 0


# ===========================================================================
# §3. Mix helpers
# ===========================================================================

@pytest.mark.fast
class TestMixHelpers:
    """Tests for composition.mix decomposition helpers."""

    def test_core_conv_mask_fully_convective(self, small_comp_mfracs):
        """When all zones are convective, core mask covers everything."""
        from stellar_jax.composition.mix import _core_conv_mask
        n = 20
        # All zones strongly convective
        is_conv_comp = jnp.ones(n) * 0.99
        hp_comp = jnp.ones(n) * 0.1
        dmdr_comp = jnp.ones(n) * 3.0
        mask = _core_conv_mask(is_conv_comp, hp_comp, dmdr_comp,
                               small_comp_mfracs, 1.5, 0.016)
        # Should be ~1 everywhere (fully convective)
        assert float(jnp.min(mask)) > 0.9

    def test_core_conv_mask_fully_radiative(self, small_comp_mfracs):
        """When no zones are convective, core mask is zero everywhere."""
        from stellar_jax.composition.mix import _core_conv_mask
        n = 20
        # All zones strongly radiative
        is_conv_comp = jnp.ones(n) * 0.01
        hp_comp = jnp.ones(n) * 0.1
        dmdr_comp = jnp.ones(n) * 3.0
        mask = _core_conv_mask(is_conv_comp, hp_comp, dmdr_comp,
                               small_comp_mfracs, 1.0, 0.016)
        # Should be ~0 everywhere
        assert float(jnp.max(mask)) < 0.1

    def test_core_conv_mask_overshoot_extends(self, small_comp_mfracs):
        """With f_ov > 0, the mixed region extends beyond pure convection."""
        from stellar_jax.composition.mix import _core_conv_mask
        n = 20
        # Convective in inner 30% (first 6 zones of 20)
        is_conv_comp = jnp.where(small_comp_mfracs < 0.30, 0.99, 0.01)
        # Large Hp and dm/dr so that overshoot delta_m is larger than grid spacing
        hp_comp = jnp.ones(n) * 0.5
        dmdr_comp = jnp.ones(n) * 5.0

        mask_no_ov = _core_conv_mask(is_conv_comp, hp_comp, dmdr_comp,
                                     small_comp_mfracs, 1.5, 0.0)
        mask_with_ov = _core_conv_mask(is_conv_comp, hp_comp, dmdr_comp,
                                       small_comp_mfracs, 1.5, 0.5)
        # Overshoot should extend the mask (large f_ov=0.5 on large Hp ensures this)
        assert float(jnp.sum(mask_with_ov)) > float(jnp.sum(mask_no_ov))

    def test_envelope_zone_mixing_homogenizes(self, small_comp_mfracs, small_shell_data):
        """Envelope mixing homogenizes composition within surface-connected CZ.

        The mixing operator only homogenizes the surface-connected envelope CZ
        (MESA do_mix_envelope, mix_info.f90:1344-1381). Interior CZs that are
        disconnected from the surface are left untouched. This test creates a
        convective envelope (surface side) with a composition gradient and
        verifies that mixing reduces the variance.
        """
        from stellar_jax.composition.mix import _envelope_zone_mixing
        n = 20
        # Create a profile with a gradient
        X = jnp.linspace(0.35, 0.70, n)
        # Build shell data with convection at the SURFACE (mf > 0.60) so
        # the CZ is surface-connected (walk inward from surface finds conv).
        # comp_mfracs goes 0 (center) to 1 (surface); mf_shells goes 0.95
        # (surface) to 0.01 (center) in the fixture. We need nabla_rad > nad
        # at HIGH mf (surface side).
        mf_shells = small_shell_data[:, 1]  # descending: 0.95 → 0.01
        nad = jnp.full(n, 0.4)
        # Convective where mf > 0.60 (surface side = first ~8 shell indices)
        nrad = jnp.where(mf_shells > 0.60, 0.8, 0.25)
        # Use zero core mask so everything is "envelope"
        core_mix_mask = jnp.zeros(n)
        zone_weights = jnp.ones(n) / n

        X_out = _envelope_zone_mixing(X, small_comp_mfracs, mf_shells,
                                      nrad, nad, core_mix_mask, zone_weights)
        # In the surface-connected convective zones, the output should be
        # more homogeneous than input.
        # The surface-connected CZ corresponds to comp_mfracs > ~0.60
        # (high indices = near the surface).
        conv_zone = small_comp_mfracs > 0.60
        if float(jnp.sum(conv_zone)) > 1:
            var_in = float(jnp.var(X[conv_zone]))
            var_out = float(jnp.var(X_out[conv_zone]))
            assert var_out < var_in

    def test_mix_composition_no_change_radiative(self, small_comp_mfracs):
        """A fully radiative star returns the input unchanged."""
        from stellar_jax.composition.mix import mix_composition
        n = 20
        X = jnp.linspace(0.35, 0.70, n)
        # Create shell_data with nabla_rad < nabla_ad everywhere (radiative)
        mf = jnp.linspace(0.95, 0.01, n)
        shell_data = jnp.column_stack([
            jnp.zeros(n),  # eps
            mf,  # m/M
            jnp.full(n, 0.25),  # nabla_rad < nabla_ad everywhere
            jnp.full(n, 0.40),  # nabla_ad
            jnp.full(n, 0.1),   # Hp/R
            jnp.full(n, 3.0),   # dm_dr_norm
            jnp.linspace(3.78, 7.17, n),  # logT
            jnp.linspace(-7, 2.18, n),    # logrho
        ])
        X_out = mix_composition(X, shell_data, M_solar=1.0, f_ov=0.016,
                                comp_mfracs_in=small_comp_mfracs)
        # Fully radiative → no mixing → output = input
        np.testing.assert_allclose(X_out, X, atol=1e-10)


# ===========================================================================
# §4. Gradient checks (AD through helpers)
# ===========================================================================

@pytest.mark.fast
class TestHelperGradients:
    """Test that gradients flow through the decomposed helpers."""

    def test_core_conv_mask_differentiable(self, small_comp_mfracs):
        """jax.grad flows through _core_conv_mask w.r.t. is_conv_comp."""
        from stellar_jax.composition.mix import _core_conv_mask
        n = 20
        is_conv_comp = jnp.where(small_comp_mfracs < 0.30, 0.9, 0.1)
        hp_comp = jnp.ones(n) * 0.1
        dmdr_comp = jnp.ones(n) * 3.0

        def _sum_mask(is_conv):
            mask = _core_conv_mask(is_conv, hp_comp, dmdr_comp,
                                   small_comp_mfracs, 1.5, 0.016)
            return jnp.sum(mask)

        grad = jax.grad(_sum_mask)(is_conv_comp)
        assert grad.shape == (n,)
        # Gradient should be non-zero (sigmoid provides finite gradient)
        assert float(jnp.max(jnp.abs(grad))) > 0

    def test_advective_settling_flux_differentiable(self):
        """jax.grad flows through _advective_settling_flux w.r.t. Y."""
        from stellar_jax.composition.diffuse import _advective_settling_flux
        n = 20
        v_He = jnp.full(n, -5e-7)
        cz_taper = jnp.ones(n)  # fully radiative (no suppression)
        flux_area = jnp.ones(n) * 1e20
        dm = jnp.ones(n) * 1e30
        dt = 1e13
        v_max = 1e-6

        def _sum_dY(Y):
            dY = _advective_settling_flux(v_He, Y, cz_taper, flux_area, dm, dt, v_max)
            return jnp.sum(dY**2)

        Y = jnp.full(n, 0.28)
        grad = jax.grad(_sum_dY)(Y)
        assert grad.shape == (n,)
        assert float(jnp.max(jnp.abs(grad))) > 0


# ===========================================================================
# §5. Burgers solver tests (composition/burgers.py — extracted)
# ===========================================================================

@pytest.mark.fast
class TestBurgersSolvers:
    """Tests for composition.burgers (extracted from diffuse.py, issue #530)."""

    def test_solve_3species_ap_positive(self):
        """AP_He should be positive (He settles inward under gravity)."""
        from stellar_jax.composition.burgers import solve_3species
        # Solar center conditions: X=0.34, Y=0.64, T=1.5e7, rho=150
        AP_He, AT_He, AX_He_H, AX_He_He = solve_3species(
            jnp.float64(0.34), jnp.float64(0.64),
            jnp.float64(1.5e7), jnp.float64(150.0))
        # AP_He > 0 means positive dlnP/dr contribution → inward settling
        assert float(AP_He) > 0

    def test_solve_3species_matches_wrapper(self):
        """solve_3species matches the diffuse.py wrapper _thoul_burgers_coefficients."""
        from stellar_jax.composition.burgers import solve_3species
        from stellar_jax.composition.diffuse import _thoul_burgers_coefficients
        X, Y, T, rho = jnp.float64(0.70), jnp.float64(0.28), jnp.float64(5e6), jnp.float64(10.0)
        r1 = solve_3species(X, Y, T, rho)
        r2 = _thoul_burgers_coefficients(X, Y, T, rho)
        for a, b in zip(r1, r2):
            np.testing.assert_allclose(float(a), float(b), rtol=0, atol=0)

    def test_solve_4species_z_settles_faster(self):
        """Metals (Z_eff=12) should settle faster than He (Z=2): |AP_Z| > |AP_He|."""
        from stellar_jax.composition.burgers import solve_4species
        AP_He, AT_He, AP_Z, AT_Z, *_ = solve_4species(
            jnp.float64(0.70), jnp.float64(0.28), jnp.float64(0.02),
            jnp.float64(1.5e7), jnp.float64(150.0))
        # Metals (Z²/A much larger) settle faster
        assert abs(float(AP_Z)) > abs(float(AP_He))

    def test_solve_4species_matches_wrapper(self):
        """solve_4species matches the diffuse.py wrapper _thoul_burgers_4species."""
        from stellar_jax.composition.burgers import solve_4species
        from stellar_jax.composition.diffuse import _thoul_burgers_4species
        X, Y, Z, T, rho = (jnp.float64(0.70), jnp.float64(0.28),
                            jnp.float64(0.02), jnp.float64(1e7), jnp.float64(50.0))
        r1 = solve_4species(X, Y, Z, T, rho)
        r2 = _thoul_burgers_4species(X, Y, Z, T, rho)
        for a, b in zip(r1, r2):
            np.testing.assert_allclose(float(a), float(b), rtol=0, atol=0)

    def test_solve_3species_differentiable(self):
        """jax.grad flows through solve_3species."""
        from stellar_jax.composition.burgers import solve_3species

        def _ap_he(Y):
            AP_He, _, _, _ = solve_3species(jnp.float64(0.70), Y,
                                            jnp.float64(1e7), jnp.float64(50.0))
            return AP_He

        grad = jax.grad(_ap_he)(jnp.float64(0.28))
        assert jnp.isfinite(grad)
        assert float(grad) != 0.0


# ===========================================================================
# §6. Settling velocity computation (_compute_settling_velocities — extracted)
# ===========================================================================

@pytest.mark.fast
class TestSettlingVelocities:
    """Tests for composition.diffuse._compute_settling_velocities."""

    def test_3sp_returns_correct_shapes(self):
        """3-species velocity computation returns (v_He, sig_He_H, sig_He_He)."""
        from stellar_jax.composition.diffuse import _compute_settling_velocities
        n = 10
        X = jnp.full(n, 0.70)
        Y = jnp.full(n, 0.28)
        Z = jnp.full(n, 0.02)
        T = jnp.full(n, 1e7)
        rho = jnp.full(n, 50.0)
        dlnP = jnp.full(n, -10.0)
        dlnT = jnp.full(n, -3.0)
        result = _compute_settling_velocities(X, Y, Z, T, rho, dlnP, dlnT,
                                              use_4species=False)
        v_He, sig_He_H, sig_He_He = result
        assert v_He.shape == (n,)
        assert sig_He_H.shape == (n,)
        assert sig_He_He.shape == (n,)
        # He should settle inward (negative velocity in surface-to-center convention)
        assert float(jnp.mean(v_He)) < 0

    def test_4sp_returns_correct_shapes(self):
        """4-species velocity computation returns 8 arrays."""
        from stellar_jax.composition.diffuse import _compute_settling_velocities
        n = 10
        X = jnp.full(n, 0.70)
        Y = jnp.full(n, 0.28)
        Z = jnp.full(n, 0.02)
        T = jnp.full(n, 1e7)
        rho = jnp.full(n, 50.0)
        dlnP = jnp.full(n, -10.0)
        dlnT = jnp.full(n, -3.0)
        result = _compute_settling_velocities(X, Y, Z, T, rho, dlnP, dlnT,
                                              use_4species=True)
        assert len(result) == 8
        v_He, v_Z = result[0], result[1]
        assert v_He.shape == (n,)
        assert v_Z.shape == (n,)

    def test_settling_velocity_differentiable(self):
        """jax.grad flows through _compute_settling_velocities."""
        from stellar_jax.composition.diffuse import _compute_settling_velocities
        n = 5

        def _sum_v_he(Y):
            X = jnp.full(n, 0.70)
            Z = jnp.full(n, 0.02)
            T = jnp.full(n, 1e7)
            rho = jnp.full(n, 50.0)
            dlnP = jnp.full(n, -10.0)
            dlnT = jnp.full(n, -3.0)
            v_He, _, _ = _compute_settling_velocities(
                X, Y, Z, T, rho, dlnP, dlnT, use_4species=False)
            return jnp.sum(v_He)

        Y = jnp.full(n, 0.28)
        grad = jax.grad(_sum_v_he)(Y)
        assert grad.shape == (n,)
        assert float(jnp.max(jnp.abs(grad))) > 0
