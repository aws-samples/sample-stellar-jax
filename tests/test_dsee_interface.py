"""Tests for dsee_interface.py — parameter mapping and normalization.

These tests validate the PURE MAPPING functions (no PyTorch/DSEE required).
The DSEEInterface class (which loads the emulator) is tested only when
CONF1DENCE is available (marked with pytest.importorskip).
"""

import numpy as np
import pytest

from stellar_jax.inference import (
    z_to_feh,
    feh_to_z,
    stellar_jax_to_dsee_labels,
    _normalize_labels,
    _denormalize_outputs,
    _norm_linear_rescale,
    _norm_normal,
    _norm_mantissa_exponent,
    dsee_outputs_to_observables,
    warmstart_params_from_observables,
    DSEE_LABEL_DEFAULT,
    DSEE_LABEL_HEADER,
    DSEE_LABEL_SCALER,
    DSEE_LABEL_SCALE,
    Z_SOLAR,
    ZX_SOLAR,
)
from stellar_jax.config.constants import Y_BBN, DY_DZ
from stellar_jax.config.mesh_defaults import ALPHA_MLT, F_OV


# ─── [Fe/H] ↔ Z round-trip tests ────────────────────────────────────────────

class TestFeHZConversion:
    """Test [Fe/H] ↔ Z conversion against known reference values."""

    def test_solar_feh_is_zero(self):
        """[Fe/H]=0 at solar metallicity Z=Z_solar."""
        # With Y from enrichment law at Z_solar
        Y_solar = Y_BBN + DY_DZ * Z_SOLAR
        feh = z_to_feh(Z_SOLAR, Y=Y_solar)
        assert abs(feh) < 0.01, f"Solar [Fe/H] should be ~0, got {feh}"

    def test_subsolar_feh_negative(self):
        """Sub-solar metallicity → negative [Fe/H]."""
        feh = z_to_feh(0.004)  # ~1/5 solar
        assert feh < -0.5, f"Expected [Fe/H]<-0.5 for Z=0.004, got {feh}"

    def test_supersolar_feh_positive(self):
        """Super-solar metallicity → positive [Fe/H]."""
        feh = z_to_feh(0.03)
        assert feh > 0.1, f"Expected [Fe/H]>0.1 for Z=0.03, got {feh}"

    def test_roundtrip_z_feh_z(self):
        """Z → [Fe/H] → Z should be identity."""
        for Z_test in [0.001, 0.005, 0.010, 0.014, 0.020, 0.030, 0.040]:
            feh = z_to_feh(Z_test)
            Z_recovered = feh_to_z(feh)
            assert abs(Z_recovered - Z_test) / Z_test < 1e-10, (
                f"Round-trip failed: Z={Z_test} → [Fe/H]={feh} → Z={Z_recovered}"
            )

    def test_roundtrip_feh_z_feh(self):
        """[Fe/H] → Z → [Fe/H] should be identity."""
        for feh_test in [-2.0, -1.0, -0.5, 0.0, 0.2, 0.5]:
            Z = feh_to_z(feh_test)
            feh_recovered = z_to_feh(Z)
            assert abs(feh_recovered - feh_test) < 1e-10, (
                f"Round-trip failed: [Fe/H]={feh_test} → Z={Z} → [Fe/H]={feh_recovered}"
            )

    def test_known_value_m13(self):
        """M13 globular cluster: [Fe/H]≈-1.53 → Z≈0.0005.

        Reference: Harris 1996 catalog (2010 edition), W.E. Harris, AJ 112, 1487.
        """
        Z = feh_to_z(-1.53)
        # At [Fe/H]=-1.53, Z should be ~0.0005-0.0006
        assert 0.0003 < Z < 0.001, f"M13 Z={Z} out of expected range"

    def test_invalid_z_raises(self):
        """Unphysical Z should raise ValueError."""
        with pytest.raises(ValueError):
            z_to_feh(0.0)  # Z=0 is unphysical
        with pytest.raises(ValueError):
            z_to_feh(0.8)  # Would make X<0

    def test_feh_with_fixed_Y(self):
        """[Fe/H] conversion with explicit Y (not from enrichment law)."""
        # With a fixed Y=0.25 (e.g., old metal-poor population)
        Z = 0.001
        Y = 0.25
        feh = z_to_feh(Z, Y=Y)
        Z_back = feh_to_z(feh, Y=Y)
        assert abs(Z_back - Z) / Z < 1e-10


# ─── Normalization tests ─────────────────────────────────────────────────────

class TestNormalization:
    """Test normalization utilities match CONF1DENCE's implementation."""

    def test_linear_rescale(self):
        """linear_rescale: (val - lo) / (hi - lo)."""
        # Mixing Length: scale [1.75, 2.5]
        assert abs(_norm_linear_rescale(1.75, [1.75, 2.5]) - 0.0) < 1e-10
        assert abs(_norm_linear_rescale(2.5, [1.75, 2.5]) - 1.0) < 1e-10
        assert abs(_norm_linear_rescale(2.125, [1.75, 2.5]) - 0.5) < 1e-10

    def test_normal_scaler(self):
        """normal: (val - mean) / std."""
        # PP rate: scale [0.997543, 0.0098]
        mean, std = 0.997543, 0.0098
        val = mean  # At the mean, normalized = 0
        assert abs(_norm_normal(val, [mean, std])) < 1e-10
        # One sigma away
        assert abs(_norm_normal(mean + std, [mean, std]) - 1.0) < 1e-10

    def test_mantissa_exponent(self):
        """mantissa_exponent: splits age into 2 dimensions."""
        # Age 4.603e9: mantissa=4.603, exponent=9
        age = 4.603e9
        m, e = _norm_mantissa_exponent(age, [3.0, 10.0])
        # Verify inverse: mantissa = ((m+5)/10)*9 + 1
        mantissa_recovered = ((m + 5.0) / 10.0) * 9.0 + 1.0
        # exponent = ((e+5)/10)*(exp_max-exp_min) + exp_min
        exponent_recovered = ((e + 5.0) / 10.0) * (10.0 - 3.0) + 3.0
        age_recovered = mantissa_recovered * 10.0 ** exponent_recovered
        assert abs(age_recovered - age) / age < 1e-6, (
            f"mantissa_exponent round-trip: {age} → {age_recovered}"
        )

    def test_normalize_labels_length(self):
        """Normalized labels should be 25-dim (24 labels, Age expands to 2)."""
        labels = np.array(DSEE_LABEL_DEFAULT, dtype=np.float64)
        normed = _normalize_labels(labels)
        assert normed.shape == (25,), f"Expected 25 dims, got {normed.shape}"

    def test_normalize_default_labels_finite(self):
        """Default labels should produce finite normalized values."""
        labels = np.array(DSEE_LABEL_DEFAULT, dtype=np.float64)
        normed = _normalize_labels(labels)
        assert np.all(np.isfinite(normed)), f"Non-finite in normalized defaults: {normed}"

    def test_denormalize_outputs_linear_rescale(self):
        """Denormalize outputs: inverse of linear_rescale."""
        # Log_T scale [3.0, 5.0]: if normed=0.5, physical = 0.5*(5-3)+3 = 4.0
        normed = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        phys = _denormalize_outputs(normed)
        assert abs(phys[0] - 4.0) < 1e-10  # Log_T
        assert abs(phys[1] - 2.25) < 1e-10  # Log_G: 0.5*(5.5-(-1))+(-1) = 2.25
        assert abs(phys[2] - 1.0) < 1e-10  # Log_L: 0.5*(5-(-3))+(-3) = 1.0
        assert abs(phys[3] - 1.0) < 1e-10  # Log_R: 0.5*(4-(-2))+(-2) = 1.0
        # Y_Core uses log_rescale: 10^(0.5*(0-(-2))+(-2)) = 10^(-1) = 0.1
        assert abs(phys[4] - 0.1) < 1e-6

    def test_denormalize_batch(self):
        """Denormalize works for batch inputs."""
        normed = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [1.0, 1.0, 1.0, 1.0, 1.0]])
        phys = _denormalize_outputs(normed)
        assert phys.shape == (2, 5)
        # Row 0: all at lower bounds
        assert abs(phys[0, 0] - 3.0) < 1e-10  # Log_T min
        assert abs(phys[0, 2] - (-3.0)) < 1e-10  # Log_L min
        # Row 1: all at upper bounds
        assert abs(phys[1, 0] - 5.0) < 1e-10  # Log_T max
        assert abs(phys[1, 2] - 5.0) < 1e-10  # Log_L max


# ─── Parameter mapping tests ─────────────────────────────────────────────────

class TestParameterMapping:
    """Test stellar-jax → DSEE label mapping."""

    def test_default_params_produce_valid_labels(self):
        """Default stellar-jax params should produce a valid DSEE label vector."""
        labels = stellar_jax_to_dsee_labels(mass=1.0)
        assert labels.shape == (24,)
        assert np.all(np.isfinite(labels))
        # Mass should be 1.0
        assert labels[23] == 1.0
        # alpha_mlt should be our default
        assert abs(labels[3] - ALPHA_MLT) < 1e-10

    def test_solar_case_produces_feh_near_zero(self):
        """Solar Z=0.0188 should give [Fe/H]≈0."""
        labels = stellar_jax_to_dsee_labels(mass=1.0, Z=Z_SOLAR)
        assert abs(labels[0]) < 0.02  # [Fe/H] ≈ 0

    def test_mass_threaded(self):
        """Mass is correctly placed at position 23."""
        for m in [0.8, 1.0, 1.5, 2.0]:
            labels = stellar_jax_to_dsee_labels(mass=m)
            assert labels[23] == m

    def test_alpha_mlt_threaded(self):
        """α_MLT is correctly placed at position 3."""
        for alpha in [1.8, 1.9, 2.0, 2.3]:
            labels = stellar_jax_to_dsee_labels(mass=1.0, alpha_mlt=alpha)
            assert abs(labels[3] - alpha) < 1e-10

    def test_f_ov_threaded(self):
        """f_ov maps to core overshooting (position 8)."""
        labels = stellar_jax_to_dsee_labels(mass=1.5, f_ov=0.05)
        assert abs(labels[8] - 0.05) < 1e-10
        # Envelope overshooting stays at default
        assert abs(labels[7] - 0.01) < 1e-10

    def test_diffusion_flag_mapping(self):
        """diffusion=True→1.0, False→0.5 for He/heavy diffusion factors."""
        labels_on = stellar_jax_to_dsee_labels(mass=1.0, diffusion=True)
        labels_off = stellar_jax_to_dsee_labels(mass=1.0, diffusion=False)
        assert labels_on[4] == 1.0   # He Diffusion
        assert labels_on[5] == 1.0   # Heavy Element Diffusion
        assert labels_off[4] == 0.5
        assert labels_off[5] == 0.5

    def test_age_threaded(self):
        """Age is correctly placed at position 22."""
        labels = stellar_jax_to_dsee_labels(mass=1.0, age_yr=10.0e9)
        assert abs(labels[22] - 10.0e9) < 1.0  # float precision

    def test_opacity_factor_threaded(self):
        """opacity_factor maps to both Low Temp and High Temp opacity."""
        labels = stellar_jax_to_dsee_labels(mass=1.0, opacity_factor=1.1)
        assert abs(labels[17] - 1.1) < 1e-10  # Low Temp Opacity
        assert abs(labels[18] - 1.1) < 1e-10  # High Temp Opacity

    def test_surface_bc_krishna_swamy(self):
        """Surface BC is set to 1.0 (Krishna-Swamy, matching our atmosphere)."""
        labels = stellar_jax_to_dsee_labels(mass=1.0)
        assert labels[6] == 1.0

    def test_nuclear_rates_at_defaults(self):
        """Nuclear rate factors stay at DSEE defaults."""
        labels = stellar_jax_to_dsee_labels(mass=1.0)
        # Positions 9–16 are nuclear rates
        for i in range(9, 17):
            assert abs(labels[i] - DSEE_LABEL_DEFAULT[i]) < 1e-10, (
                f"Nuclear rate {DSEE_LABEL_HEADER[i]} at position {i} "
                f"should be {DSEE_LABEL_DEFAULT[i]}, got {labels[i]}"
            )

    def test_full_normalization_roundtrip_sanity(self):
        """Normalized labels from default params are within expected range.

        DSEE's normalizing flow expects inputs roughly in [-5, 5] range
        (the mantissa_exponent scaler produces values in this range by design).
        """
        labels = stellar_jax_to_dsee_labels(mass=1.0, Z=0.014, age_yr=4.603e9)
        normed = _normalize_labels(labels)
        # None-scaled values pass through raw (FeH≈-0.16, He≈0.27, AlphaFe=0, SBC=1)
        # linear_rescale values should be in [0, 1] for in-range inputs
        # normal values should be within a few sigma (say [-10, 10])
        # mantissa_exponent values should be in [-5, 5]
        assert np.all(np.isfinite(normed))
        # Check alpha_mlt normalization: (1.9 - 1.75)/(2.5-1.75) = 0.2
        alpha_idx = 3  # linear_rescale
        assert abs(normed[alpha_idx] - 0.2) < 1e-6


# ─── Observable output tests ─────────────────────────────────────────────────

class TestDSEEOutputs:
    """Test output conversion."""

    def test_single_output_dict(self):
        """Single output → dict with correct keys."""
        raw = np.array([3.76, 4.44, 0.0, 0.0, 0.28])
        obs = dsee_outputs_to_observables(raw)
        assert 'log_Teff' in obs
        assert 'log_g' in obs
        assert 'log_L' in obs
        assert 'log_R' in obs
        assert 'Y_core' in obs
        assert obs['log_Teff'] == 3.76
        assert obs['Y_core'] == 0.28

    def test_batch_output_arrays(self):
        """Batch output → dict with arrays."""
        raw = np.array([[3.76, 4.44, 0.0, 0.0, 0.28],
                        [3.80, 4.00, 1.0, 0.3, 0.50]])
        obs = dsee_outputs_to_observables(raw)
        assert obs['log_Teff'].shape == (2,)
        assert obs['log_L'][1] == 1.0


# ─── Warm-start from observables (no emulator) ──────────────────────────────

class TestWarmstartFromObservables:
    """Test the pure-mapping warmstart_params_from_observables function."""

    def test_solar_case(self):
        """A solar-like star should return mass≈1, Z≈solar, alpha≈ALPHA_MLT."""
        result = warmstart_params_from_observables(
            log_Teff_obs=np.log10(5778.0),
            log_L_obs=0.0,  # L=L_sun
            log_g_obs=4.438,  # g=g_sun
            feh_obs=0.0,
        )
        assert 0.8 < result['mass'] < 1.3, f"Solar mass guess: {result['mass']}"
        assert abs(result['Z'] - Z_SOLAR) / Z_SOLAR < 0.05
        assert result['alpha_mlt'] == ALPHA_MLT
        assert result['diffusion'] is True

    def test_metal_poor_maps_low_z(self):
        """[Fe/H]=-1.0 should give Z<<Z_solar."""
        result = warmstart_params_from_observables(
            log_Teff_obs=3.80, log_L_obs=0.5, feh_obs=-1.0
        )
        assert result['Z'] < 0.003, f"Metal-poor Z={result['Z']} too high"

    def test_luminous_star_higher_mass(self):
        """A luminous star should get a higher mass guess."""
        result = warmstart_params_from_observables(
            log_Teff_obs=3.85, log_L_obs=1.5  # ~30 L_sun
        )
        assert result['mass'] > 1.5, f"Luminous star mass={result['mass']}"

    def test_returns_expected_keys(self):
        """Output dict has all required keys for evolve_star."""
        result = warmstart_params_from_observables(
            log_Teff_obs=3.76, log_L_obs=0.0
        )
        expected_keys = {'mass', 'Z', 'alpha_mlt', 'f_ov', 'Y_init', 't_max', 'diffusion'}
        assert expected_keys <= set(result.keys())

    def test_mass_clamped_to_physical_range(self):
        """Mass should be clamped to [0.5, 3.0]."""
        # Very luminous → would give M>3 from L relation
        result = warmstart_params_from_observables(
            log_Teff_obs=4.2, log_L_obs=4.0  # ~10000 L_sun
        )
        assert result['mass'] <= 3.0

    def test_no_feh_defaults_to_solar_neighborhood(self):
        """Without [Fe/H], Z defaults to 0.014."""
        result = warmstart_params_from_observables(
            log_Teff_obs=3.76, log_L_obs=0.0
        )
        assert result['Z'] == 0.014


# ─── Integration: full pipeline (requires torch + zuko + CONF1DENCE) ────────

def _can_import_torch():
    """Check if PyTorch is importable."""
    try:
        import torch
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _can_import_torch(),
    reason="PyTorch not installed — skipping DSEE emulator tests"
)
class TestDSEEEmulator:
    """Integration tests requiring the actual DSEE model.

    These only run when torch, zuko, and the DSEE model weights are available.
    """

    @pytest.fixture
    def dsee(self):
        """Load the DSEE interface (skip if model not found)."""
        try:
            from stellar_jax.inference import DSEEInterface
            return DSEEInterface()
        except (ImportError, FileNotFoundError) as e:
            pytest.skip(f"DSEE model not available: {e}")

    def test_solar_prediction_reasonable(self, dsee):
        """DSEE prediction for a solar-like star should be reasonable."""
        obs = dsee.predict(mass=1.0, Z=Z_SOLAR, age_yr=4.57e9)
        # Solar: log_Teff ≈ 3.76, log_L ≈ 0, log_R ≈ 0
        assert 3.70 < obs['log_Teff'] < 3.82
        assert -0.5 < obs['log_L'] < 0.5
        assert -0.3 < obs['log_R'] < 0.3

    def test_warmstart_returns_all_keys(self, dsee):
        """sample_warmstart returns a complete parameter dict."""
        result = dsee.sample_warmstart(mass=1.0, Z=0.014, age_yr=4.57e9)
        expected = {'mass', 'Z', 'alpha_mlt', 'f_ov', 'Y_init', 't_max',
                    'diffusion', 'log_L_pred', 'log_Teff_pred',
                    'log_R_pred', 'log_g_pred', 'Y_core_pred'}
        assert expected <= set(result.keys())

    def test_mass_luminosity_trend(self, dsee):
        """Higher mass → higher luminosity (MS physical trend)."""
        obs_1 = dsee.predict(mass=1.0, Z=0.014, age_yr=1.0e9)
        obs_2 = dsee.predict(mass=2.0, Z=0.014, age_yr=1.0e9)
        assert obs_2['log_L'] > obs_1['log_L'], (
            f"Expected L(2M☉) > L(1M☉): {obs_2['log_L']} vs {obs_1['log_L']}"
        )
