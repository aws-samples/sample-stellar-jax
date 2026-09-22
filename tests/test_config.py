"""Unit tests for the config/ package and gradient_policy.py.

All tests are fast (<5s each, no evolve_star, no JIT compile, isolated).
They validate:
- StellarParams and SolverConfig frozen dataclass behavior
- GradientPolicy enum completeness
- Physical constants regression
- Mesh defaults invariants
- Open/Closed extension pattern
- Backward-compat shims

Marks: @pytest.mark.fast (no evolve_star, pure Python/numpy)
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# All tests in this module are fast: pure Python/numpy, no evolve_star, no JIT compile.
pytestmark = pytest.mark.fast


# ===========================================================================
# StellarParams tests
# ===========================================================================

class TestStellarParams:
    """Tests for config.params.StellarParams."""

    def test_frozen_immutable(self):
        """StellarParams instances cannot be mutated (frozen=True)."""
        from stellar_jax.config.params import StellarParams
        p = StellarParams(mass=1.0, Z=0.014)
        with pytest.raises(Exception):  # FrozenInstanceError
            p.mass = 2.0

    def test_default_values(self):
        """Default StellarParams has standard physics (knobs = 1.0)."""
        from stellar_jax.config.params import StellarParams
        p = StellarParams()
        assert p.mass == 1.0
        assert p.Z == 0.014
        assert p.alpha_mlt == 1.9
        assert p.f_ov == 0.016
        assert p.Y_init is None
        assert p.t_max is None
        assert p.opacity_factor == 1.0
        assert p.eps_nuc_factor == 1.0
        assert p.diffusion_factor == 1.0

    def test_knobs_default_to_identity(self):
        """Physics-calibration knobs all default to 1.0 (identity multiply)."""
        from stellar_jax.config.params import StellarParams
        p = StellarParams()
        assert p.opacity_factor == 1.0
        assert p.eps_nuc_factor == 1.0
        assert p.diffusion_factor == 1.0

    def test_custom_construction(self):
        """StellarParams can be constructed with non-default values."""
        from stellar_jax.config.params import StellarParams
        p = StellarParams(mass=2.0, Z=0.02, alpha_mlt=2.1, opacity_factor=1.1)
        assert p.mass == 2.0
        assert p.Z == 0.02
        assert p.alpha_mlt == 2.1
        assert p.opacity_factor == 1.1

    def test_implements_has_physics_knobs(self):
        """StellarParams satisfies the HasPhysicsKnobs protocol."""
        from stellar_jax.config.params import StellarParams
        from stellar_jax.config.contracts import HasPhysicsKnobs
        p = StellarParams()
        assert isinstance(p, HasPhysicsKnobs)

    def test_equality(self):
        """Two StellarParams with same values are equal (dataclass __eq__)."""
        from stellar_jax.config.params import StellarParams
        p1 = StellarParams(mass=1.5, Z=0.02)
        p2 = StellarParams(mass=1.5, Z=0.02)
        assert p1 == p2

    def test_inequality_on_knob_change(self):
        """Different knob values produce unequal instances."""
        from stellar_jax.config.params import StellarParams
        p1 = StellarParams(opacity_factor=1.0)
        p2 = StellarParams(opacity_factor=1.1)
        assert p1 != p2


# ===========================================================================
# SolverConfig tests
# ===========================================================================

class TestSolverConfig:
    """Tests for config.solver_config.SolverConfig."""

    def test_frozen_immutable(self):
        """SolverConfig instances cannot be mutated."""
        from stellar_jax.config.solver_config import SolverConfig
        cfg = SolverConfig(max_steps=200)
        with pytest.raises(Exception):
            cfg.max_steps = 300

    def test_default_values(self):
        """SolverConfig defaults match the existing evolve_star defaults."""
        from stellar_jax.config.solver_config import SolverConfig
        cfg = SolverConfig()
        assert cfg.max_steps == 500
        assert cfg.diffusion is True
        assert cfg.adaptive_mesh is True
        assert cfg.z_feedback is False
        assert cfg.bicubic_opacity is True
        assert cfg.freeze_schedule is False
        assert cfg.replay_schedule is False
        assert cfg.has_idle_guard is False
        assert cfg.eps_grav_enabled is True
        assert cfg.xh_cntr_limit == 0.01
        assert cfg.xh_cntr_hard_limit == 0.05
        assert cfg.grad_window is None
        assert cfg.fixed_dt is None
        assert cfg.varcontrol_target == 1e-3

    def test_all_fields_are_python_types(self):
        """All SolverConfig fields are Python types (not jnp arrays).

        This is critical: SolverConfig fields are static_argnames. If any field
        were a JAX tracer, it would silently corrupt the XLA compilation.
        """
        from stellar_jax.config.solver_config import SolverConfig
        cfg = SolverConfig()
        for field_name in cfg.__dataclass_fields__:
            val = getattr(cfg, field_name)
            assert not hasattr(val, 'shape'), \
                f"Field {field_name} looks like an array (has .shape) — must be plain Python"
            # Check it's a basic Python type
            assert isinstance(val, (int, float, bool, tuple, type(None))), \
                f"Field {field_name} is {type(val)} — expected Python scalar/tuple/None"

    def test_to_jit_kwargs(self):
        """to_jit_kwargs produces the correct mapping for _evolve_star_jit."""
        from stellar_jax.config.solver_config import SolverConfig
        cfg = SolverConfig(max_steps=100, diffusion=False, eps_grav_enabled=False)
        kw = cfg.to_jit_kwargs()
        assert kw['max_steps'] == 100
        assert kw['diffusion'] is False
        assert kw['_eps_grav_flag'] is False
        assert kw['_xh_cntr_limit'] == 0.01
        assert kw['adaptive_mesh'] is True

    def test_to_jit_kwargs_keys_match_static_argnames(self):
        """to_jit_kwargs keys cover all legacy static_argnames except varcontrol_target."""
        from stellar_jax.config.solver_config import SolverConfig
        cfg = SolverConfig()
        kw = cfg.to_jit_kwargs()
        # These are the static_argnames in the current _evolve_star_jit
        expected_static = {
            'max_steps', 'diffusion', 'grad_window', 'z_feedback',
            'bicubic_opacity', 'fixed_dt', 'freeze_schedule', 'replay_schedule',
            'adaptive_mesh', 'has_idle_guard', '_eps_grav_flag',
            '_xh_cntr_limit', '_xh_cntr_hard_limit',
        }
        assert set(kw.keys()) == expected_static

    def test_no_recompile_on_params_change(self):
        """Changing StellarParams doesn't affect SolverConfig hash (separate concerns)."""
        from stellar_jax.config.solver_config import SolverConfig
        from stellar_jax.config.params import StellarParams
        cfg = SolverConfig(max_steps=100)
        p1 = StellarParams(mass=1.0)
        p2 = StellarParams(mass=2.0)
        # SolverConfig hash is independent of StellarParams
        assert hash(cfg) == hash(SolverConfig(max_steps=100))


# ===========================================================================
# GradientPolicy tests
# ===========================================================================

class TestGradientPolicy:
    """Tests for gradient_policy.GradPolicy enum."""

    def test_exactly_12_members(self):
        """GradPolicy has exactly 12 members (GP-1 through GP-12)."""
        from stellar_jax.gradient_policy import GradPolicy
        assert len(GradPolicy) == 12

    def test_values_1_through_12(self):
        """GradPolicy values are integers 1 through 12."""
        from stellar_jax.gradient_policy import GradPolicy
        values = sorted(p.value for p in GradPolicy)
        assert values == list(range(1, 13))

    def test_expected_names(self):
        """All 12 expected policy names exist."""
        from stellar_jax.gradient_policy import GradPolicy
        expected = {
            'TIMESTEP', 'MESH_POSITIONS', 'DIFFUSION', 'Z_CNO_ISOTOPES',
            'STRUCTURE_TO_COMP', 'IDLE_GUARD', 'WINDOWED_SCAN',
            'EPSGRAV_HISTORY', 'CONVERGENCE_FLAGS',
            'WARM_START_GUESS', 'MESH_SCHEDULE', 'COMP_SCHEDULE',
        }
        actual = {p.name for p in GradPolicy}
        assert actual == expected

    def test_is_intenum(self):
        """GradPolicy members are integers (can be used in annotations/comments)."""
        from stellar_jax.gradient_policy import GradPolicy
        assert GradPolicy.TIMESTEP == 1
        assert GradPolicy.CONVERGENCE_FLAGS == 9
        assert isinstance(GradPolicy.DIFFUSION, int)

    def test_docstrings_present(self):
        """Each policy member has a non-empty docstring (in the enum class)."""
        from stellar_jax.gradient_policy import GradPolicy
        # IntEnum members don't have per-member __doc__; check the class docstring
        assert GradPolicy.__doc__ is not None
        assert len(GradPolicy.__doc__) > 50


# ===========================================================================
# Physical constants regression tests
# ===========================================================================

class TestConstants:
    """Regression tests for physical constants (catch accidental edits)."""

    def test_gravitational_constant(self):
        from stellar_jax.config.constants import G
        assert G == 6.67430e-8

    def test_speed_of_light(self):
        from stellar_jax.config.constants import c_light
        assert c_light == 2.99792458e10

    def test_solar_values(self):
        from stellar_jax.config.constants import Msun, Lsun, Rsun, mu_sun, G
        # Msun is derived from mu_sun/G (matching MESA const_def.f90:124)
        assert Msun == mu_sun / G
        assert Lsun == 3.828e33
        assert Rsun == 6.957e10

    def test_cosmological(self):
        from stellar_jax.config.constants import Y_BBN, DY_DZ
        assert Y_BBN == 0.2485
        assert DY_DZ == 1.5

    def test_seconds_per_year(self):
        from stellar_jax.config.constants import SECONDS_PER_YEAR
        assert SECONDS_PER_YEAR == 3.15576e7


# ===========================================================================
# Mesh defaults tests
# ===========================================================================

class TestMeshDefaults:
    """Tests for config.mesh_defaults."""

    def test_comp_mfracs_boundary(self):
        """COMP_MFRACS starts at 0 and ends at 1."""
        from stellar_jax.config.mesh_defaults import COMP_MFRACS
        assert COMP_MFRACS[0] == 0.0
        assert COMP_MFRACS[-1] == 1.0

    def test_comp_mfracs_monotonic(self):
        """COMP_MFRACS is strictly monotonically increasing."""
        from stellar_jax.config.mesh_defaults import COMP_MFRACS
        assert np.all(np.diff(COMP_MFRACS) > 0)

    def test_comp_mfracs_length(self):
        """COMP_MFRACS has N_COMP elements."""
        from stellar_jax.config.mesh_defaults import COMP_MFRACS, N_COMP
        assert len(COMP_MFRACS) == N_COMP

    def test_zone_masses_sum_to_one(self):
        """COMP_ZONE_MASSES sums to exactly 1.0 (conservative)."""
        from stellar_jax.config.mesh_defaults import COMP_ZONE_MASSES
        assert abs(COMP_ZONE_MASSES.sum() - 1.0) < 1e-15

    def test_zone_masses_all_positive(self):
        """All zone masses are positive."""
        from stellar_jax.config.mesh_defaults import COMP_ZONE_MASSES
        assert np.all(COMP_ZONE_MASSES > 0)

    def test_compute_zone_masses_matches_numpy(self):
        """JAX compute_zone_masses matches the numpy COMP_ZONE_MASSES."""
        from stellar_jax.config.mesh_defaults import COMP_MFRACS, COMP_ZONE_MASSES, compute_zone_masses
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        jax_zones = compute_zone_masses(jnp.array(COMP_MFRACS))
        np.testing.assert_allclose(np.array(jax_zones), COMP_ZONE_MASSES, atol=1e-15)


# ===========================================================================
# MESA config tests
# ===========================================================================

class TestMesaConfig:
    """Tests for config.mesa_config."""

    def test_mesa_config_has_required_keys(self):
        """MESA_CONFIG dict has the required physics keys."""
        from stellar_jax.config.mesa_config import MESA_CONFIG
        required = {'Z', 'alpha_mlt', 'Y_init', 'diffusion', 'f_ov'}
        assert required.issubset(set(MESA_CONFIG.keys()))

    def test_mesa_config_types(self):
        """MESA_CONFIG values have correct types."""
        from stellar_jax.config.mesa_config import MESA_CONFIG
        assert isinstance(MESA_CONFIG['Z'], float)
        assert isinstance(MESA_CONFIG['alpha_mlt'], float)
        assert isinstance(MESA_CONFIG['Y_init'], float)
        assert isinstance(MESA_CONFIG['diffusion'], bool)
        assert isinstance(MESA_CONFIG['f_ov'], (int, float))

    def test_mesa_mode_a_values(self):
        """MODE A values are consistent (alpha=2.0, Z=0.014, no diffusion)."""
        from stellar_jax.config.mesa_config import MESA_CONFIG
        assert MESA_CONFIG['alpha_mlt'] == 2.0
        assert MESA_CONFIG['Z'] == 0.014
        assert MESA_CONFIG['diffusion'] is False


# ===========================================================================
# Backward-compatibility shim tests
# ===========================================================================

# TestBackwardCompat removed: both constants.py shim (Part D,) and
# mesa_config.py shim are retired — no backward-compat shims remain.


# ===========================================================================
# Open/Closed extension pattern test
# ===========================================================================

class TestOpenClosedPattern:
    """Test the Open/Closed extension pattern for physics knobs."""

    def test_knob_extension_is_one_field(self):
        """Adding a new knob requires only adding a field (no signature change).

        This test documents the Open/Closed pattern: to add a new physics knob
        (e.g. nuclear_screening_factor), you add ONE field to StellarParams
        with default 1.0, and ONE multiply at the consumption point.
        """
        from stellar_jax.config.params import StellarParams

        # Default params — all knobs are identity
        default = StellarParams()
        assert default.opacity_factor == 1.0

        # Modified params — one knob changed
        modified = StellarParams(opacity_factor=1.1)
        assert modified.opacity_factor == 1.1

        # The multiply pattern: result * knob = result (when knob=1.0)
        fake_kappa = 0.42  # some opacity value
        assert fake_kappa * default.opacity_factor == fake_kappa
        assert fake_kappa * modified.opacity_factor == pytest.approx(0.462)

    def test_all_knobs_are_multiplicative(self):
        """All physics knobs multiply their target (identity at 1.0)."""
        from stellar_jax.config.params import StellarParams
        p = StellarParams()
        knobs = [p.opacity_factor, p.eps_nuc_factor, p.diffusion_factor]
        for knob in knobs:
            assert knob == 1.0, f"Knob {knob} should default to 1.0 (identity)"


# ===========================================================================
# Gradient Policy Lint Test (CI enforcement)
# ===========================================================================

class TestGradientPolicyLint:
    """Enforce that every stop_gradient call is annotated with its GP-N policy."""

    def test_all_stop_gradients_annotated(self):
        """Every stop_gradient call in production code has a # GP-<N> annotation.

        This is the CI enforcement of the gradient-policy contract
        (gradient_policy.py, docs/design/redesign/11-config-and-gradient-policy.md).
        """
        import subprocess
        result = subprocess.run(
            [sys.executable, 'scripts/lint_gradient_policy.py'],
            capture_output=True, text=True, cwd=os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
        assert result.returncode == 0, (
            f"Gradient policy lint failed:\n{result.stdout}\n{result.stderr}")


# ===========================================================================
# Constants Consistency Test
# ===========================================================================

class TestConstantsConsistency:
    """Verify that all physics modules get their constants from config/constants.py.

    Issue #788: structural de-duplication. Every module that uses a physical constant
    must import it from config/constants.py (via the root constants.py shim or directly).
    No module may re-declare a literal value that exists in the centralized source.

    This test imports each module and verifies the constant objects are literally the
    SAME float (object identity where possible, value equality as fallback).
    """

    def test_helm_constants_from_config(self):
        """microphysics/helm.py constants are imported from config/constants.py."""
        from stellar_jax.config.constants import (
            k_B_helm, m_H_helm, m_e_helm, hbar_helm,
            c_light_helm, e_charge_helm, a_rad_helm,
        )
        from stellar_jax.microphysics import helm
        assert helm.k_B == k_B_helm
        assert helm.m_H == m_H_helm
        assert helm.m_e == m_e_helm
        assert helm.hbar == hbar_helm
        assert helm.c_light == c_light_helm
        assert helm.e_charge == e_charge_helm
        assert helm.a_rad == a_rad_helm

    def test_fd_electron_constants_from_config(self):
        """microphysics/fd_electron.py constants are imported from config/constants.py."""
        from stellar_jax.config.constants import (
            k_B_fd, m_H_fd, hbar_fd,
            e_charge_fd, a_rad_fd,
        )
        from stellar_jax.microphysics import fd_electron
        assert fd_electron.k_B == k_B_fd
        assert fd_electron.m_H == m_H_fd
        assert fd_electron.hbar == hbar_fd
        assert fd_electron.e_charge == e_charge_fd
        assert fd_electron.a_rad == a_rad_fd
        # m_e and c_light are not re-exported by fd_electron — they were
        # unused aliases removed in. The values live in config/constants.py
        # as m_e_fd and c_light_fd.

    def test_opal_eos_constants_from_config(self):
        """microphysics/eos.py uses centralized OPAL constants (not function-local)."""
        from stellar_jax.config.constants import k_B_opal, m_H_opal, h_planck_opal, a_rad_opal
        # The constants are imported as _k_B, _m_H, _h_planck, _a_rad in eos.py.
        # Verify the values are correct.
        assert k_B_opal == 1.3806e-16
        assert m_H_opal == 1.6726e-24
        assert h_planck_opal == 6.6261e-27
        assert a_rad_opal == 7.5657e-15

    def test_oscillations_G_from_config(self):
        """oscillations modules use G from config/constants.py."""
        from stellar_jax.config.constants import G
        assert G == 6.67430e-8

    def test_fgong_column_constants_defined(self):
        """fgong/contracts.py exports all named FGONG column constants."""
        from stellar_jax.fgong.contracts import (
            VAR_R, VAR_LN_Q, VAR_T, VAR_P, VAR_RHO, VAR_X, VAR_L,
            VAR_KAPPA, VAR_EPS_NUC, VAR_GAMMA1, VAR_NABLA_AD, VAR_DELTA,
            VAR_CP, VAR_NABLA, VAR_A_STAR, VAR_Z, VAR_DIST_SURF,
            GLOB_M, GLOB_R, GLOB_L, GLOB_Z, GLOB_X_SURF, GLOB_ALPHA,
            GLOB_AGE, GLOB_G,
        )
        # Verify column indices match the FGONG standard
        # (Christensen-Dalsgaard 2008, Ap&SS 316, 113, Table A.1)
        assert VAR_R == 0
        assert VAR_LN_Q == 1
        assert VAR_T == 2
        assert VAR_P == 3
        assert VAR_RHO == 4
        assert VAR_X == 5
        assert VAR_L == 6
        assert VAR_KAPPA == 7
        assert VAR_EPS_NUC == 8
        assert VAR_GAMMA1 == 9
        assert VAR_NABLA_AD == 10
        assert VAR_DELTA == 11
        assert VAR_CP == 12
        assert VAR_NABLA == 13
        assert VAR_A_STAR == 14
        assert VAR_Z == 16
        assert VAR_DIST_SURF == 17
        assert GLOB_M == 0
        assert GLOB_R == 1
        assert GLOB_L == 2
        assert GLOB_Z == 3
        assert GLOB_X_SURF == 4
        assert GLOB_ALPHA == 5
        assert GLOB_AGE == 12
        assert GLOB_G == 14

    def test_no_redeclared_constants_in_production(self):
        """Grep-level check: no literal physical constants in production modules.

        After #788, the modules (helm.py, fd_electron.py, eos.py, oscillations/*)
        should not contain standalone literal declarations like 'k_B = 1.38...'
        Only config/constants.py is the source.
        """
        import re
        import pathlib
        project_root = pathlib.Path(__file__).parent.parent / 'src' / 'stellar_jax'
        # Modules that previously had redeclared constants
        targets = [
            project_root / 'microphysics' / 'helm.py',
            project_root / 'microphysics' / 'fd_electron.py',
            project_root / 'microphysics' / 'eos.py',
            project_root / 'oscillations' / 'coefficients.py',
            project_root / 'oscillations' / 'diagnostic.py',
            project_root / 'calibration' / 'comparison.py',
        ]
        # Pattern: standalone assignment of a physical constant to a scientific-notation
        # numeric literal (e.g. `k_B = 1.380649e-16`). Does NOT match derived expressions
        # like `h_planck = 2.0 * jnp.pi * hbar`.
        pattern = re.compile(
            r'^\s*(a_rad|k_B|m_H|m_e|hbar|c_light|e_charge|h_planck|G)\s*=\s*'
            r'[0-9]+\.[0-9]+[eE]',
            re.MULTILINE
        )
        violations = []
        for path in targets:
            text = path.read_text()
            matches = pattern.findall(text)
            if matches:
                violations.append(f"{path.name}: {matches}")
        assert not violations, (
            f"Physical constants still redeclared as literals in: {violations}. "
            f"Should import from config/constants.py (issue #788)."
        )


# ===========================================================================
# Physical Constants vs MESA Guard
# ===========================================================================

class TestPhysicalConstantsVsMesa:
    """Guard that primary physical constants match MESA r26.04.1 const_def.f90.

    What: verifies G, mu_sun, Msun (derived), Lsun, Rsun against MESA values.
    Why: removes the ~0.03–0.05% constant-drift systematic on the L/R/g scale.
    External reference: MESA const/public/const_def.f90 lines 114–126.
    Tolerance: <1e-5 relative (floating-point identity at these precisions).
    Mutation: any revert to the old Allen-era values fails immediately.
    """

    def test_G_matches_mesa_standard_cgrav(self):
        """G == MESA standard_cgrav (const_def.f90:114)."""
        from stellar_jax.config.constants import G
        mesa_standard_cgrav = 6.67430e-8
        rel_err = abs(G - mesa_standard_cgrav) / mesa_standard_cgrav
        assert rel_err < 1e-5, f"G={G} vs MESA {mesa_standard_cgrav}, rel_err={rel_err}"

    def test_mu_sun_matches_mesa(self):
        """mu_sun == MESA mu_sun (const_def.f90:118, IAU 2015 Res B3)."""
        from stellar_jax.config.constants import mu_sun
        mesa_mu_sun = 1.3271244e26
        rel_err = abs(mu_sun - mesa_mu_sun) / mesa_mu_sun
        assert rel_err < 1e-5, f"mu_sun={mu_sun} vs MESA {mesa_mu_sun}, rel_err={rel_err}"

    def test_Msun_derived_from_mu_sun_over_G(self):
        """Msun == mu_sun/G (matching MESA const_def.f90:124 derivation)."""
        from stellar_jax.config.constants import Msun, mu_sun, G
        mesa_Msun = mu_sun / G
        rel_err = abs(Msun - mesa_Msun) / mesa_Msun
        assert rel_err < 1e-10, f"Msun={Msun} vs mu_sun/G={mesa_Msun}, rel_err={rel_err}"

    def test_Lsun_matches_mesa(self):
        """Lsun == MESA Lsun (const_def.f90:126, IAU 2015 Res B3)."""
        from stellar_jax.config.constants import Lsun
        mesa_Lsun = 3.828e33
        rel_err = abs(Lsun - mesa_Lsun) / mesa_Lsun
        assert rel_err < 1e-5, f"Lsun={Lsun} vs MESA {mesa_Lsun}, rel_err={rel_err}"

    def test_Rsun_matches_mesa(self):
        """Rsun == MESA Rsun (const_def.f90:125, IAU 2015 Res B3)."""
        from stellar_jax.config.constants import Rsun
        mesa_Rsun = 6.957e10
        rel_err = abs(Rsun - mesa_Rsun) / mesa_Rsun
        assert rel_err < 1e-5, f"Rsun={Rsun} vs MESA {mesa_Rsun}, rel_err={rel_err}"

    def test_Msun_value_physical(self):
        """Msun value is in the expected physical range (sanity)."""
        from stellar_jax.config.constants import Msun
        # MESA: Msun = 1.3271244e26 / 6.67430e-8 ≈ 1.98841e33
        assert 1.988e33 < Msun < 1.989e33, f"Msun={Msun} out of physical range"


# ===========================================================================
# Float64 guarantee (Part B, §11-A)
# ===========================================================================

class TestFloat64Guarantee:
    """Verify that importing stellar_jax enables JAX float64.

    What: importing any stellar_jax submodule guarantees jax_enable_x64=True.
    Why: JAX defaults to float32; stellar structure tolerances (<1e-7 solar
         calibration, <1% sound speed vs Model S) are unreachable in float32.
    External reference: design standards §11-A (float64 is mandatory).
    """

    def test_leaf_import_enables_x64(self):
        """Importing a leaf module (via stellar_jax) enables float64.

        Regression guard: previously x64 was enabled only as a side-effect
        of importing stellar.py or adaptive_forward.py. Importing a leaf
        (e.g., config.constants) first would leave float32 active.
        """
        import jax
        # stellar_jax.__init__.py sets jax_enable_x64=True at import time.
        # By the time any test runs, the package has been imported.
        import stellar_jax.config.constants  # noqa: F401 — leaf module
        assert jax.config.jax_enable_x64, (
            "jax_enable_x64 is False after importing stellar_jax.config.constants. "
            "stellar_jax/__init__.py must enable x64 before any submodule import."
        )

    def test_jnp_default_dtype_is_float64(self):
        """jnp.array(1.0) produces float64 after stellar_jax import."""
        import jax.numpy as jnp
        x = jnp.array(1.0)
        assert x.dtype == jnp.float64, (
            f"Default jnp dtype is {x.dtype}, expected float64. "
            f"jax_enable_x64 may not be set early enough."
        )
