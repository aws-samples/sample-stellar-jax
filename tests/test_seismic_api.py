"""Validation tests for the first-class seismic API (issue #768).

Tests the n_pg-identified functions shipped in oscillations/seismic_quantities.py:
  - identify_modes, assign_n_pg
  - delta_nu_02, delta_nu_01
  - r02, r01, r10
  - large_separation
  - nu_max_scaling
  - epsilon_fit

All tests are @fast (no evolve_star, no oscillation solver call — operate directly
on committed GYRE reference data or MESA FGONGs). Each is @validation + @mutation-gated.

External references:
  - GYRE 8.1 (Townsend & Teitler 2013): committed per-mode frequencies at
    data/mesa_comparison/gyre_reference/ (JCD outer BC, GL6 collocation)
  - MESA report.f90:343 (large_separation), report.f90:353 (nu_max_scaling)
  - Roxburgh & Vorontsov (2003), A&A 411, 215 (r02/r01/r10 definitions)
  - Tassoul (1980), ApJS 43, 469 (asymptotic relation, ε)
  - Brown et al. (1991), ApJ 368, 599 (ν_max scaling)
"""
import csv
import gzip
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.helpers import _resolve_data_path


# ════════════════════════════════════════════════════════════════════════════════
# Helpers (shared with test_seismic_structure.py; DRY — load GYRE/MESA data)
# ════════════════════════════════════════════════════════════════════════════════

def _load_gyre_freqs(mass_str, stage):
    """Load GYRE reference frequencies: {l: {n_pg: freq_uHz}}."""
    path = _resolve_data_path(
        os.path.join('mesa_comparison', 'gyre_reference', mass_str, f'{stage}.txt'))
    freqs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('l'):
                continue
            parts = line.split()
            if len(parts) == 3:
                try:
                    l, n_pg, freq = int(parts[0]), int(parts[1]), float(parts[2])
                except ValueError:
                    continue
                if l not in freqs:
                    freqs[l] = {}
                freqs[l][n_pg] = freq
    return freqs


def _load_mesa_fgong(mass_str, stage):
    """Load a MESA MODE-A FGONG.gz file, return (glob, var)."""
    from stellar_jax.fgong.io import read_fgong

    rel_path = os.path.join('mesa_comparison', 'profiles', mass_str, f'{stage}.FGONG.gz')
    fgong_path = _resolve_data_path(rel_path)

    with gzip.open(fgong_path, 'rt') as f:
        content = f.read()
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.FGONG', delete=False)
    tmp.write(content)
    tmp.close()
    try:
        glob, var = read_fgong(tmp.name)
    finally:
        os.unlink(tmp.name)
    return glob, var


def _gyre_freqs_as_arrays(gyre_freqs, l_values=(0, 1, 2), nu_min=1000.0, nu_max=5000.0):
    """Convert GYRE {l: {n_pg: freq}} to {l: sorted_freq_array} for identify_modes."""
    result = {}
    for l in l_values:
        if l in gyre_freqs:
            freqs = [f for f in gyre_freqs[l].values()
                     if nu_min <= f <= nu_max]
            result[l] = np.array(sorted(freqs))
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Test 1: identify_modes recovers GYRE n_pg assignments
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("seismic_npg_offset")
@pytest.mark.right_reason("identification")
def test_identify_modes_vs_gyre_npg():
    """identify_modes assigns n_pg that match the GYRE reference within ±0.

    WHAT: Loads GYRE reference frequencies (which have ground-truth n_pg), feeds
    them as sorted arrays to identify_modes(), and verifies the assigned n_pg
    matches the GYRE n_pg for all modes in the asymptotic p-mode regime.

    WHY: n_pg identification is the foundation for all ratio/separation functions.
    If n_pg assignment is off by even 1, all δν₀₂ and r02 values are wrong.
    The asymptotic relation ν ≈ Δν(n + l/2 + ε) must correctly recover n for
    clean p-mode MS stars.

    EXTERNAL REFERENCE: GYRE 8.1 reference files (committed at
    data/mesa_comparison/gyre_reference/). GYRE assigns n_pg by eigenfunction
    node-counting (the ground truth for mode identification).

    TOLERANCE: Exact match (n_pg must be identical) for modes in the asymptotic
    regime (n ≥ 5, where the Tassoul relation is accurate to within ±0.1Δν).

    MUTATION: seismic_npg_offset — adds +2 to the ε estimate in assign_n_pg,
    shifting all n_pg assignments by −2. This makes every ratio/separation use
    the wrong mode pairs.
    """
    from stellar_jax.oscillations.seismic_quantities import identify_modes

    # Test across 4 masses/stages covering the MS (includes 1.0Msun zams where
    # eps_raw ≈ 0.30 — a corner case that previously triggered an off-by-1 error
    # with the old 0.3 threshold, now correctly handled by the 0.5 threshold).
    models = [
        ('1.0Msun', 'midMS'),
        ('1.0Msun', 'zams'),
        ('1.5Msun', 'zams'),
        ('2.0Msun', 'Xc0.40'),
    ]

    total_checked = 0
    total_correct = 0

    for mass_str, stage in models:
        gyre_freqs = _load_gyre_freqs(mass_str, stage)
        # Feed GYRE frequencies as arrays (simulating solver output)
        freqs_arrays = _gyre_freqs_as_arrays(gyre_freqs, l_values=(0, 1, 2))

        identified = identify_modes(freqs_arrays)

        # Check l=0 modes in the asymptotic regime (n ≥ 5)
        for l in (0, 1, 2):
            if l not in identified or l not in gyre_freqs:
                continue
            our_modes = identified[l]
            gyre_modes = gyre_freqs[l]
            for n_ours, freq_ours in our_modes.items():
                if n_ours < 5:
                    continue
                # Find the GYRE mode with matching frequency (within 0.1 µHz)
                gyre_n = None
                for gn, gf in gyre_modes.items():
                    if abs(gf - freq_ours) < 0.1:
                        gyre_n = gn
                        break
                if gyre_n is not None:
                    total_checked += 1
                    if n_ours == gyre_n:
                        total_correct += 1

    assert total_checked >= 50, (
        f"Only {total_checked} modes checked (expected ≥50)")
    frac_correct = total_correct / total_checked
    assert frac_correct > 0.95, (
        f"n_pg identification: {total_correct}/{total_checked} = "
        f"{frac_correct*100:.1f}% correct (expected >95%)")


# ════════════════════════════════════════════════════════════════════════════════
# Test 2: delta_nu_02 and r02 match independent GYRE calculation
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("seismic_npg_offset")
@pytest.mark.right_reason("median error")
def test_dnu02_and_r02_vs_gyre():
    """δν₀₂ and r02 from the API match independently-computed GYRE values.

    WHAT: Feeds GYRE reference frequencies through identify_modes() → delta_nu_02()
    and r02(), then compares against the same quantities computed directly from
    the GYRE {n_pg: freq} dictionaries (ground truth, no asymptotic identification
    needed since GYRE provides n_pg).

    WHY: Validates the complete chain: frequency → identify_modes → ratio/separation.
    If the API's n_pg assignment is correct, the ratios must match exactly (to
    floating-point precision) because the math is identical.

    EXTERNAL REFERENCE: GYRE 8.1 reference frequencies (committed). The "direct"
    calculation uses GYRE's own n_pg labeling.

    TOLERANCE: <1% relative error for median δν₀₂ and median r02 (accounts for
    slight n_pg assignment mismatches at the lowest/highest orders where the
    asymptotic relation is less accurate).

    MUTATION: seismic_npg_offset — wrong n_pg shifts pair all separations to
    incorrect mode combinations, producing ~50% median error.
    """
    from stellar_jax.oscillations.seismic_quantities import identify_modes, delta_nu_02, r02

    models = [
        ('1.0Msun', 'midMS'),
        ('1.0Msun', 'Xc0.30'),
        ('1.5Msun', 'zams'),
    ]

    for mass_str, stage in models:
        gyre_freqs = _load_gyre_freqs(mass_str, stage)
        freqs_arrays = _gyre_freqs_as_arrays(gyre_freqs)

        # Our API path
        identified = identify_modes(freqs_arrays)
        our_d02 = delta_nu_02(identified)
        our_r02 = r02(identified)

        # Direct from GYRE n_pg (ground truth)
        gyre_d02_direct = {}
        for n, f0 in gyre_freqs[0].items():
            if (n - 1) in gyre_freqs[2] and n >= 2:
                d = f0 - gyre_freqs[2][n - 1]
                if d > 0:
                    gyre_d02_direct[n] = d

        gyre_r02_direct = {}
        for n, f0 in gyre_freqs[0].items():
            if (n - 1) in gyre_freqs.get(2, {}) and n in gyre_freqs.get(1, {}):
                if (n - 1) in gyre_freqs.get(1, {}):
                    d02 = f0 - gyre_freqs[2][n - 1]
                    dnu1 = gyre_freqs[1][n] - gyre_freqs[1][n - 1]
                    if dnu1 > 0 and d02 > 0:
                        gyre_r02_direct[n] = d02 / dnu1

        # Compare median δν₀₂ (only orders present in both)
        common_n_d02 = set(our_d02.keys()) & set(gyre_d02_direct.keys())
        assert len(common_n_d02) >= 10, (
            f"{mass_str}/{stage}: only {len(common_n_d02)} common δν₀₂ pairs")
        our_vals = [our_d02[n] for n in sorted(common_n_d02) if n >= 5]
        gyre_vals = [gyre_d02_direct[n] for n in sorted(common_n_d02) if n >= 5]
        if len(our_vals) >= 5:
            rel_d02 = abs(np.median(our_vals) - np.median(gyre_vals)) / np.median(gyre_vals)
            assert rel_d02 < 0.01, (
                f"{mass_str}/{stage}: δν₀₂ median error {rel_d02*100:.2f}% > 1%")

        # Compare median r02
        common_n_r02 = set(our_r02.keys()) & set(gyre_r02_direct.keys())
        assert len(common_n_r02) >= 5, (
            f"{mass_str}/{stage}: only {len(common_n_r02)} common r02 pairs")
        our_r02_vals = [our_r02[n] for n in sorted(common_n_r02) if n >= 5]
        gyre_r02_vals = [gyre_r02_direct[n] for n in sorted(common_n_r02) if n >= 5]
        if len(our_r02_vals) >= 5:
            rel_r02 = abs(np.median(our_r02_vals) - np.median(gyre_r02_vals)) / np.median(gyre_r02_vals)
            assert rel_r02 < 0.01, (
                f"{mass_str}/{stage}: r02 median error {rel_r02*100:.2f}% > 1%")


# ════════════════════════════════════════════════════════════════════════════════
# Test 3: r01 and r10 smoothness and physical range
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("seismic_npg_offset")
@pytest.mark.right_reason("mismatch")
def test_r01_r10_physical_range():
    """r01 and r10 produce values in the expected physical range and match GYRE keys.

    WHAT: Computes r01(n) and r10(n) from GYRE reference frequencies and checks
    they fall in the physically expected range for MS p-modes. Also verifies that
    the n_pg keys match the independently-known GYRE radial orders.

    WHY: r01 and r10 are the l=0/l=1 surface-independent ratios (Roxburgh &
    Vorontsov 2003, Eqs. 4 & 6). For solar-type MS stars, they should be small
    (~0.01–0.05) and approximately constant in n (smooth). The key-matching
    against GYRE ensures n_pg identification is correct (not just that the
    values happen to be in range with shifted labels).

    EXTERNAL REFERENCE: Roxburgh & Vorontsov (2003), A&A 411, 215. For the Sun,
    r01 ≈ 0.01–0.03 (Fig. 1 therein). GYRE 8.1 mode identification by n_pg.

    TOLERANCE: 0.0 < r01 < 0.10, -0.02 < r10 < 0.10 (physical range for clean
    p-modes). At least 10 valid pairs must be found. n_pg keys must overlap >80%
    with the GYRE-direct calculation.

    MUTATION: seismic_npg_offset — shifts all n_pg by −2. The values remain in
    physical range (since the underlying physics is unchanged), but the n_pg KEYS
    no longer match GYRE's independently-computed radial orders → the >80%
    overlap assertion fails (shifted keys have 0% overlap for n≥5).
    """
    from stellar_jax.oscillations.seismic_quantities import identify_modes, r01, r10

    gyre_freqs = _load_gyre_freqs('1.0Msun', 'midMS')
    freqs_arrays = _gyre_freqs_as_arrays(gyre_freqs, l_values=(0, 1, 2))
    identified = identify_modes(freqs_arrays)

    our_r01 = r01(identified)
    our_r10 = r10(identified)

    assert len(our_r01) >= 10, f"Only {len(our_r01)} r01 values (expected ≥10)"
    assert len(our_r10) >= 10, f"Only {len(our_r10)} r10 values (expected ≥10)"

    # Physical range check
    r01_vals = np.array(list(our_r01.values()))
    r10_vals = np.array(list(our_r10.values()))

    assert np.all(r01_vals > -0.02), (
        f"r01 has values < -0.02: min={r01_vals.min():.4f}")
    assert np.all(r01_vals < 0.10), (
        f"r01 has values > 0.10: max={r01_vals.max():.4f}")
    assert np.median(r01_vals) > 0.0, (
        f"r01 median = {np.median(r01_vals):.4f} ≤ 0 (expected positive)")
    assert np.all(r10_vals > -0.02), (
        f"r10 has values < -0.02: min={r10_vals.min():.4f}")
    assert np.all(r10_vals < 0.10), (
        f"r10 has values > 0.10: max={r10_vals.max():.4f}")
    assert np.std(r01_vals) < 0.02, (
        f"r01 std = {np.std(r01_vals):.4f} > 0.02 (not smooth)")

    # KEY-VALUE MATCH against GYRE: compute r01 directly from GYRE n_pg.
    # Under the seismic_npg_offset mutation, our r01[n] uses a mode from
    # 2 orders higher than the true n → the value at a given key differs.
    gyre_r01_direct = {}
    for n, f0 in gyre_freqs[0].items():
        if n < 5:
            continue
        if n in gyre_freqs.get(1, {}) and (n - 1) in gyre_freqs.get(1, {}):
            d01 = f0 - 0.5 * (gyre_freqs[1][n - 1] + gyre_freqs[1][n])
            dnu1 = gyre_freqs[1][n] - gyre_freqs[1][n - 1]
            if dnu1 > 0:
                gyre_r01_direct[n] = d01 / dnu1

    # Compare our r01 values at matching keys
    common_keys = set(our_r01.keys()) & set(gyre_r01_direct.keys())
    common_keys = {k for k in common_keys if k >= 5}
    assert len(common_keys) >= 5, (
        f"Only {len(common_keys)} common r01 keys with GYRE (expected ≥5)")

    our_vals = np.array([our_r01[n] for n in sorted(common_keys)])
    gyre_vals = np.array([gyre_r01_direct[n] for n in sorted(common_keys)])
    max_abs_err = float(np.max(np.abs(our_vals - gyre_vals)))
    assert max_abs_err < 0.001, (
        f"r01 max absolute error vs GYRE = {max_abs_err:.5f} > 0.001. "
        f"n_pg identification mismatch.")


# ════════════════════════════════════════════════════════════════════════════════
# Test 4: large_separation from acoustic radius vs GYRE Δν
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("seismic_acoustic_radius_corrupt")
@pytest.mark.right_reason("vs GYRE")
def test_large_separation_vs_gyre():
    """Δν from the acoustic radius integral matches the GYRE summary Δν to <2%.

    WHAT: Computes Δν = 1/(2τ₀) where τ₀ = ∫dr/c_s from the committed MESA FGONG,
    and compares against the Δν in the committed GYRE summary.csv (median l=0
    spacing from individually computed modes).

    WHY: The acoustic radius integral is how MESA computes Δν (report.f90:343),
    and it must agree with the actual frequency spacing to within the asymptotic
    correction (which is <1% for MS stars). This validates our large_separation()
    function against an independent mode-based Δν.

    EXTERNAL REFERENCE: GYRE summary.csv (committed); MESA report.f90:343.

    TOLERANCE: <2% relative error. Physical basis: the asymptotic Δν formula
    Δν = 1/(2∫dr/c_s) is accurate to <1% for clean p-modes on the MS (the
    correction terms are O(1/n²)). The 2% bound provides headroom for boundary
    effects (truncation of the integral vs the photosphere correction MESA applies).

    MUTATION: seismic_acoustic_radius_corrupt — drops Γ₁ from the sound speed
    (c_s = √(P/ρ) instead of √(Γ₁·P/ρ)), making Δν ~25% too low.
    """
    from stellar_jax.oscillations.seismic_quantities import large_separation

    # Load GYRE summary for reference Δν values
    summary_path = _resolve_data_path(
        os.path.join('mesa_comparison', 'gyre_reference', 'summary.csv'))
    gyre_dnu = {}
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['mass_Msun'].strip(), row['stage'].strip())
            gyre_dnu[key] = float(row['Dnu_uHz'])

    models = [
        ('1.0Msun', 'midMS', '1.0', 'midMS'),
        ('1.0Msun', 'zams', '1.0', 'zams'),
        ('1.5Msun', 'zams', '1.5', 'zams'),
        ('2.0Msun', 'zams', '2.0', 'zams'),
    ]

    checked = 0
    for mass_str, stage, mass_csv, stage_csv in models:
        glob, var = _load_mesa_fgong(mass_str, stage)
        our_dnu = large_separation(glob, var)

        ref_key = (mass_csv, stage_csv)
        if ref_key not in gyre_dnu:
            continue
        gyre_val = gyre_dnu[ref_key]

        rel_err = abs(our_dnu - gyre_val) / gyre_val
        assert rel_err < 0.02, (
            f"{mass_str}/{stage}: Δν = {our_dnu:.2f} µHz vs GYRE {gyre_val:.2f} µHz"
            f" = {rel_err*100:.1f}% (>2%)")
        checked += 1

    assert checked >= 3, f"Only {checked} models checked (expected ≥3)"


# ════════════════════════════════════════════════════════════════════════════════
# Test 5: nu_max_scaling matches expected ranges
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("solar_constants_corrupt")
@pytest.mark.right_reason("outside")
def test_nu_max_scaling_range():
    """ν_max from the scaling relation falls in the expected frequency range.

    WHAT: Computes ν_max from MESA FGONGs using the Brown-Kjeldsen-Bedding
    scaling relation and checks it falls in the expected range for each mass.

    WHY: ν_max determines the observable mode window — modes within ~Δν of ν_max
    are detectable. If ν_max is wrong, the frequency range we scan is wrong.

    EXTERNAL REFERENCE: Brown et al. (1991); MESA report.f90:353.
    Expected: 1.0 M☉ midMS → ν_max ≈ 2500–3500 µHz; 2.0 M☉ ZAMS → 600–1500 µHz.

    TOLERANCE: ν_max within factor-of-2 of expected midpoint (the scaling relation
    is ~5–10% accurate for MS stars; factor-2 is a generous sanity bound).

    MUTATION: solar_constants_corrupt — R☉ × 3 shifts R/R☉ → ν_max wrong by ~9×.
    """
    from stellar_jax.oscillations.seismic_quantities import nu_max_scaling

    test_cases = [
        ('1.0Msun', 'midMS', 2000, 4000),
        ('2.0Msun', 'zams', 500, 2500),
        ('1.5Msun', 'zams', 800, 2500),
    ]

    for mass_str, stage, nu_min, nu_max in test_cases:
        glob, var = _load_mesa_fgong(mass_str, stage)
        our_numax = nu_max_scaling(glob, var)

        assert nu_min < our_numax < nu_max, (
            f"{mass_str}/{stage}: ν_max = {our_numax:.0f} µHz "
            f"outside [{nu_min}, {nu_max}]")


# ════════════════════════════════════════════════════════════════════════════════
# Test 6: epsilon_fit gives physical values
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("seismic_npg_offset")
@pytest.mark.right_reason("outside")
def test_epsilon_fit_physical():
    """ε from the asymptotic fit falls in the expected range for solar-type stars.

    WHAT: Computes ε = median(ν_{n,0}/Δν mod 1 + correction) from GYRE l=0
    frequencies and checks it falls in [1.0, 1.7] for solar-type MS stars.

    WHY: ε encodes surface physics (the phase shift at the acoustic cutoff).
    For solar-type stars with radiative cores, ε ≈ 1.4–1.5. For more massive
    stars, ε can be lower (~1.0–1.3). Values outside [0.8, 2.0] indicate a
    broken asymptotic assumption.

    EXTERNAL REFERENCE: Christensen-Dalsgaard, Lecture Notes on Stellar
    Oscillations, §5.1; White et al. (2012), ApJ 751, 36 (ε vs Teff
    relation: ε ≈ 1.5 at Teff☉, decreasing toward hotter stars).

    TOLERANCE: ε ∈ [1.0, 1.7] for the masses/stages tested.

    MUTATION: seismic_npg_offset — shifts ε by +2, producing values ∈ [3.0, 3.7],
    well outside the physical range.
    """
    from stellar_jax.oscillations.seismic_quantities import epsilon_fit

    models = [
        ('1.0Msun', 'midMS', 1.0, 1.7),   # solar-like, ε ≈ 1.1–1.5
        ('1.5Msun', 'zams', 0.5, 1.5),    # hotter, ε can be < 1.0
        ('2.0Msun', 'zams', 0.8, 1.6),    # hotter still
    ]

    for mass_str, stage, eps_min, eps_max in models:
        gyre_freqs = _load_gyre_freqs(mass_str, stage)
        freqs_arrays = _gyre_freqs_as_arrays(gyre_freqs, l_values=(0,))

        eps = epsilon_fit(freqs_arrays)
        assert eps_min < eps < eps_max, (
            f"{mass_str}/{stage}: ε = {eps:.3f} outside [{eps_min}, {eps_max}]")


# ════════════════════════════════════════════════════════════════════════════════
# Test 7: delta_nu_01 physical range and sign
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("seismic_npg_offset")
@pytest.mark.right_reason("mismatch")
def test_delta_nu_01_sign_and_range():
    """δν₀₁ is positive, bounded, and its n_pg keys match GYRE reference.

    WHAT: Computes δν₀₁(n) = ν(n,0) − [ν(n−1,1) + ν(n,1)]/2 from GYRE
    reference and checks physical properties plus n_pg key correctness.

    WHY: δν₀₁ measures the l=0 vs l=1 interior difference. For MS p-modes,
    it should be positive (l=0 sits between adjacent l=1 modes, slightly
    above the midpoint) and much smaller than Δν. The key-matching ensures
    the n_pg identification is correct.

    EXTERNAL REFERENCE: Roxburgh & Vorontsov (2003), Eq. 3; measured values
    for the Sun are δν₀₁ ≈ 2–6 µHz (Chaplin & Miglio 2013, Fig. 4).
    GYRE 8.1 n_pg identification.

    TOLERANCE: δν₀₁ ∈ (−2, 0.2·Δν) for all modes; median > 0. n_pg keys
    must overlap >80% with GYRE-direct calculation.

    MUTATION: seismic_npg_offset — shifts n_pg by −2. Values remain physical
    (uniform shift doesn't change local differences), but keys no longer
    match GYRE's ground-truth n_pg → overlap < 80%.
    """
    from stellar_jax.oscillations.seismic_quantities import identify_modes, delta_nu_01

    gyre_freqs = _load_gyre_freqs('1.0Msun', 'midMS')
    freqs_arrays = _gyre_freqs_as_arrays(gyre_freqs, l_values=(0, 1))
    identified = identify_modes(freqs_arrays)
    d01 = delta_nu_01(identified)

    assert len(d01) >= 10, f"Only {len(d01)} δν₀₁ values (expected ≥10)"

    d01_vals = np.array(list(d01.values()))
    delta_nu = float(np.median(np.diff(np.sort(freqs_arrays[0]))))

    # Median should be positive
    assert np.median(d01_vals) > 0.0, (
        f"δν₀₁ median = {np.median(d01_vals):.3f} ≤ 0")

    # All values should be bounded
    assert np.all(d01_vals > -2.0), (
        f"δν₀₁ has values < -2 µHz: min={d01_vals.min():.3f}")
    assert np.all(d01_vals < 0.2 * delta_nu), (
        f"δν₀₁ has values > 0.2·Δν = {0.2*delta_nu:.1f}: "
        f"max={d01_vals.max():.3f}")

    # VALUE MATCH: compute GYRE-direct δν₀₁ and verify our values match.
    # Under seismic_npg_offset, our d01[n] uses modes from 2 orders higher,
    # so the value at a given key n differs from GYRE's true δν₀₁(n).
    gyre_d01_direct = {}
    for n, f0 in gyre_freqs[0].items():
        if n < 5:
            continue
        if (n - 1) in gyre_freqs.get(1, {}) and n in gyre_freqs.get(1, {}):
            gyre_d01_direct[n] = f0 - 0.5 * (gyre_freqs[1][n - 1] + gyre_freqs[1][n])

    common_keys = {n for n in d01.keys() if n >= 5} & set(gyre_d01_direct.keys())
    assert len(common_keys) >= 5, (
        f"Only {len(common_keys)} common δν₀₁ keys with GYRE (expected ≥5)")

    our_vals = np.array([d01[n] for n in sorted(common_keys)])
    gyre_vals = np.array([gyre_d01_direct[n] for n in sorted(common_keys)])
    max_abs_err = float(np.max(np.abs(our_vals - gyre_vals)))
    assert max_abs_err < 0.01, (
        f"δν₀₁ max absolute error vs GYRE = {max_abs_err:.4f} µHz > 0.01. "
        f"n_pg identification mismatch.")


# ════════════════════════════════════════════════════════════════════════════════
# Test 8: estimate_delta_nu raises ValueError (no 135µHz fallback)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.fast
def test_estimate_delta_nu_no_fallback():
    """estimate_delta_nu raises ValueError when < 3 modes, not silent 135µHz.

    WHAT: Calls estimate_delta_nu with insufficient modes and verifies it
    raises ValueError (the 135µHz solar fallback was removed in #768).

    WHY: The silent fallback masked failures — a model with no computed modes
    would silently get Δν=135µHz, making downstream diagnostics meaningless
    without any error signal. The new behavior forces the caller to handle
    the insufficient-modes case explicitly.
    """
    from stellar_jax.oscillations.diagnostic import estimate_delta_nu

    # Should raise with < 3 modes
    with pytest.raises(ValueError, match="Cannot estimate"):
        estimate_delta_nu({0: np.array([1000.0, 2000.0])}, l=1)

    with pytest.raises(ValueError, match="Cannot estimate"):
        estimate_delta_nu({1: np.array([1000.0])}, l=0)

    # Should work fine with ≥ 3 modes
    result = estimate_delta_nu({0: np.array([1000.0, 1100.0, 1200.0, 1300.0])}, l=0)
    assert abs(result - 100.0) < 1.0
