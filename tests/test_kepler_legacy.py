"""Residual report: Kepler LEGACY observed vs scaling-relation predictions.

Validates the ingested data (Lund et al. 2017 + Silva Aguirre et al. 2017)
by comparing observed seismic quantities to standard asteroseismic scaling
relation predictions using the published stellar parameters {M, R, Teff}.

This is a DATA-VALIDATION test, not a forward-model test. It verifies:
1. Data loads correctly for all selected targets
2. Internal consistency: observed Δν from frequencies matches the catalog value
3. Scaling-relation residuals are within the known systematic floor (~2-5%)

The full forward-model comparison (evolve to {M, age} and compute oscillation
frequencies) is the domain of #526 / #677 — this test provides the observed
reference data those tests will compare against.

References:
    Lund et al. 2017, ApJ 835, 172 (frequencies, ratios)
    Silva Aguirre et al. 2017, ApJ 835, 173 (M, R, age)
    Chaplin & Miglio 2013, ARA&A 51, 353 (scaling relations, arXiv:1303.1957)
    Kjeldsen & Bedding 1995, A&A 293, 87 (original Δν scaling)

Scaling relations used:
    Δν/Δν☉ = (M/M☉)^{0.5} (R/R☉)^{-1.5}         [Kjeldsen & Bedding 1995]
    ν_max/ν_max☉ = (M/M☉) (R/R☉)^{-2} (T/T☉)^{-0.5}  [Brown et al. 1991]

Solar reference values:
    Δν☉ = 135.1 μHz (Huber et al. 2011)
    ν_max☉ = 3090 μHz (Huber et al. 2011)
    T_eff☉ = 5777 K
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Solar reference values for scaling relations
DNU_SUN = 135.1    # μHz, Huber et al. 2011
NUMAX_SUN = 3090.0  # μHz, Huber et al. 2011
TEFF_SUN = 5777.0   # K


def _load_legacy():
    """Load the LEGACY data module."""
    legacy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'data', 'kepler_legacy')
    sys.path.insert(0, legacy_dir)
    import loader
    return loader


@pytest.mark.smoke
class TestKeplerLegacyDataIngestion:
    """Validate Kepler LEGACY data ingestion and internal consistency.

    WHAT: verifies that the LEGACY data files are correctly parsed and
    internally consistent (observed Δν from individual frequencies matches
    the catalog large separation).

    WHY: this is the real-star reference data for {M,age} recovery validation.
    If the loader is broken, all downstream validation is meaningless.

    EXTERNAL REFERENCE: Lund et al. 2017 (ApJ 835, 172); Silva Aguirre et al.
    2017 (ApJ 835, 173). Observed data, not our model output.

    TOLERANCE: Δν from individual modes vs catalog should agree to <1%
    (measurement precision); scaling-relation predictions agree with observed
    to ~2-5% (known systematic floor of scaling relations; Chaplin & Miglio 2013
    §3.2 quote ~2% systematic for Δν, ~5% for ν_max).
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.loader = _load_legacy()
        self.targets = {
            12069424: "16 Cyg A",
            12069449: "16 Cyg B",
            6106415: "Perky",
            8379927: "Arthur",
        }

    def test_loader_all_targets(self):
        """All 4 target stars load completely: frequencies, ratios, params."""
        for kic, name in self.targets.items():
            freqs = self.loader.load_frequencies(kic)
            ratios = self.loader.load_ratios(kic)
            params = self.loader.load_stellar_params(kic)
            glob = self.loader.load_global_params(kic)

            # Frequencies: must have l=0 and l=1 modes (the minimum for Δν)
            assert len(freqs[0]) >= 10, (
                f"{name}: only {len(freqs[0])} l=0 modes (expected >=10)")
            assert len(freqs[1]) >= 10, (
                f"{name}: only {len(freqs[1])} l=1 modes (expected >=10)")
            assert len(freqs[2]) >= 5, (
                f"{name}: only {len(freqs[2])} l=2 modes (expected >=5)")

            # Ratios: r_02 must exist for δν₀₂ validation
            assert len(ratios["r_02"]) >= 5, (
                f"{name}: only {len(ratios['r_02'])} r_02 values (expected >=5)")

            # Stellar params: mass and age must be physically reasonable
            assert 0.7 < params["mass"] < 1.5, (
                f"{name}: mass={params['mass']} outside 0.7-1.5 M☉")
            assert 0.5 < params["age"] < 14.0, (
                f"{name}: age={params['age']} Gyr outside 0.5-14")
            assert 0.7 < params["radius"] < 2.5, (
                f"{name}: radius={params['radius']} outside 0.7-2.5 R☉")

            # Global params sanity
            assert 800 < glob["numax"] < 5000, (
                f"{name}: numax={glob['numax']} outside MS range")
            assert 40 < glob["Dnu"] < 180, (
                f"{name}: Dnu={glob['Dnu']} outside MS range")

    def test_delta_nu_internal_consistency(self):
        """Δν from individual l=0 frequencies matches catalog Δν to <1%.

        This is a measurement-precision check: the catalog Δν (Table 3) should
        be consistent with the median consecutive l=0 spacing from Table 6.
        """
        for kic, name in self.targets.items():
            freqs = self.loader.load_frequencies(kic)
            glob = self.loader.load_global_params(kic)

            dnu_modes, _ = self.loader.compute_observed_delta_nu(freqs)
            dnu_catalog = glob["Dnu"]

            residual_pct = abs(dnu_modes - dnu_catalog) / dnu_catalog * 100
            assert residual_pct < 1.0, (
                f"{name}: Δν from modes ({dnu_modes:.3f}) vs catalog "
                f"({dnu_catalog:.3f}) differ by {residual_pct:.2f}% (>1%)")

    def test_scaling_relation_residuals(self):
        """Report Δν and ν_max scaling-relation predictions vs observed.

        The scaling relations (Kjeldsen & Bedding 1995; Brown et al. 1991) predict:
            Δν/Δν☉ = (M/M☉)^{0.5} (R/R☉)^{-1.5}
            ν_max/ν_max☉ = (M/M☉) (R/R☉)^{-2} (T/T☉)^{-0.5}

        Systematic floor is ~2% for Δν, ~5% for ν_max (Chaplin & Miglio 2013).
        We assert <8% (generous, since averaging pipeline M,R introduces scatter).
        """
        print("\n" + "=" * 72)
        print("RESIDUAL REPORT: Kepler LEGACY — Scaling Relations vs Observed")
        print("=" * 72)
        print(f"{'Star':<12} {'Δν_obs':>8} {'Δν_scl':>8} {'res%':>6} "
              f"{'νmax_obs':>9} {'νmax_scl':>9} {'res%':>6}")
        print("-" * 72)

        for kic, name in self.targets.items():
            params = self.loader.load_stellar_params(kic)
            glob = self.loader.load_global_params(kic)

            M = params["mass"]
            R = params["radius"]
            Teff = glob["Teff"]

            # Scaling predictions
            dnu_predicted = DNU_SUN * (M**0.5) * (R**(-1.5))
            numax_predicted = NUMAX_SUN * M * (R**(-2.0)) * (Teff / TEFF_SUN)**(-0.5)

            dnu_observed = glob["Dnu"]
            numax_observed = glob["numax"]

            res_dnu = (dnu_predicted - dnu_observed) / dnu_observed * 100
            res_numax = (numax_predicted - numax_observed) / numax_observed * 100

            print(f"{name:<12} {dnu_observed:8.2f} {dnu_predicted:8.2f} "
                  f"{res_dnu:+6.2f}  {numax_observed:9.1f} "
                  f"{numax_predicted:9.1f} {res_numax:+6.2f}")

            # Scaling relations have ~2-5% systematic error (Chaplin & Miglio 2013)
            # plus pipeline-averaged M/R have ~4%/2% uncertainties → total ~8%
            assert abs(res_dnu) < 8.0, (
                f"{name}: Δν scaling residual {res_dnu:.1f}% exceeds 8%")
            assert abs(res_numax) < 8.0, (
                f"{name}: ν_max scaling residual {res_numax:.1f}% exceeds 8%")

        print("-" * 72)
        print("Tolerances: <8% (scaling-relation systematic + pipeline M,R scatter)")
        print("Reference: Chaplin & Miglio 2013, ARA&A 51, 353 (§3.2)")
        print()

    def test_r02_ratio_report(self):
        """Report observed r₀₂ ratios — the age-diagnostic observable.

        The r₀₂(n) ratios are the most age-sensitive seismic diagnostic
        (sensitive to the sound-speed gradient near the core, hence to X_c).
        This test reports them for downstream use by #526/#677.

        No model prediction here (that requires evolve_star) — just verify
        the data is sane (positive, <0.15, decreasing mean trend with n).
        """
        print("\n" + "=" * 72)
        print("OBSERVED r₀₂ RATIOS (Lund et al. 2017, erratum-corrected)")
        print("=" * 72)

        for kic, name in self.targets.items():
            ratios = self.loader.load_ratios(kic)
            r02 = ratios["r_02"]

            print(f"\n{name} (KIC {kic}): {len(r02)} r₀₂ values")
            print(f"  {'n':>3}  {'r_02':>8}  {'σ_lo':>7}  {'σ_hi':>7}")
            for row in r02:
                print(f"  {row['n']:3d}  {row['ratio']:8.5f}  "
                      f"{row['e_ratio_lo']:7.5f}  {row['e_ratio_hi']:7.5f}")

            # Sanity checks on the ratio values
            assert np.all(r02["ratio"] > 0), (
                f"{name}: negative r_02 values found")
            assert np.all(r02["ratio"] < 0.15), (
                f"{name}: r_02 > 0.15 (unphysical for MS solar-like)")

            # r_02 should broadly decrease with n (core contraction lowers it)
            # — check that the last 5 are on average lower than the first 5
            if len(r02) >= 10:
                first5 = np.mean(r02["ratio"][:5])
                last5 = np.mean(r02["ratio"][-5:])
                assert last5 < first5, (
                    f"{name}: r_02 does not decrease with n "
                    f"(first5={first5:.5f}, last5={last5:.5f})")

        print("\n" + "-" * 72)
        print("These ratios are the age diagnostic for #526/{M,age} recovery.")
        print()

    def test_delta_nu_02_report(self):
        """Report observed δν₀₂ (small separation) — secondary age diagnostic.

        δν₀₂(n) = ν(n,l=0) - ν(n-1,l=2) is sensitive to the core hydrogen
        content (Christensen-Dalsgaard 1984; Roxburgh & Vorontsov 2003).
        """
        print("\n" + "=" * 72)
        print("OBSERVED δν₀₂ (small separation, computed from individual modes)")
        print("=" * 72)

        for kic, name in self.targets.items():
            freqs = self.loader.load_frequencies(kic)
            dnu02 = self.loader.compute_observed_delta_nu_02(freqs)

            dnu, _ = self.loader.compute_observed_delta_nu(freqs)
            mean_dnu02 = np.mean(dnu02["delta_nu_02"]) if len(dnu02) > 0 else 0

            print(f"\n{name} (KIC {kic}): {len(dnu02)} values, "
                  f"mean δν₀₂ = {mean_dnu02:.2f} μHz, "
                  f"δν₀₂/Δν = {mean_dnu02/dnu:.4f}")

            # δν₀₂ should be positive and much smaller than Δν (~5-10 μHz for solar-like)
            assert len(dnu02) >= 5, (
                f"{name}: only {len(dnu02)} δν₀₂ values (expected >=5)")
            assert np.all(dnu02["delta_nu_02"] > 0), (
                f"{name}: negative δν₀₂ values (non-physical)")
            assert mean_dnu02 < 0.15 * dnu, (
                f"{name}: mean δν₀₂ = {mean_dnu02:.1f} > 15% of Δν "
                f"(too large for MS)")
            assert mean_dnu02 > 0.01 * dnu, (
                f"{name}: mean δν₀₂ = {mean_dnu02:.1f} < 1% of Δν "
                f"(too small — evolved subgiant, not MS)")

        print()

    def test_pipeline_spread(self):
        """Report the inter-pipeline spread in {M, age} for each target.

        This quantifies the systematic uncertainty floor in the reference
        parameters. Our forward-model residuals cannot be better than this.
        """
        print("\n" + "=" * 72)
        print("PIPELINE SPREAD: {M, age} from Silva Aguirre et al. 2017")
        print("=" * 72)
        print(f"{'Star':<12} {'N':>3} {'M_mean':>7} {'M_std':>6} {'M_rng':>7} "
              f"{'age_mean':>8} {'age_std':>7} {'age_rng':>8}")
        print("-" * 72)

        for kic, name in self.targets.items():
            params = self.loader.load_stellar_params(kic)
            pipes = params["pipelines"]

            masses = np.array([p["mass"] for p in pipes])
            ages = np.array([p["age"] for p in pipes])

            print(f"{name:<12} {len(pipes):3d} {masses.mean():7.4f} "
                  f"{masses.std():6.4f} {masses.max()-masses.min():7.4f} "
                  f"{ages.mean():8.3f} {ages.std():7.3f} "
                  f"{ages.max()-ages.min():8.3f}")

            # Pipeline spread gives the systematic floor
            # Mass spread should be <10% of mean (well-determined)
            assert masses.std() / masses.mean() < 0.10, (
                f"{name}: mass std/mean = {masses.std()/masses.mean():.3f} > 10%")
            # Age spread can be larger (~10-15% for older stars)
            assert ages.std() / ages.mean() < 0.20, (
                f"{name}: age std/mean = {ages.std()/ages.mean():.3f} > 20%")

        print("-" * 72)
        print("Pipeline spread sets the systematic floor for any model comparison.")
        print()
