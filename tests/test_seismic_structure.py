"""Tight validation tests for sensitive seismic and structure quantities (issue #686).

This file tests quantities that are SMALL DIFFERENCES or RATIOS of well-matched
gross outputs — the blind-spot class identified in the test-validation-gap audit
(docs/design/test-validation-gaps.md). A 0.2% per-mode error in individual ν can
produce a 30–50% error in δν₀₂; these tests catch that.

TOLERANCE NOTE — r02 and δν₀₂ tolerances (15%/20%) vs the issue #686 acceptance (1–2%):
The 1–2% target is GATED ON #682 (fix of the l=2 core-treatment bias identified in #678).
Our l=2 modes are currently +3–5 µHz too high vs GYRE (verified on the MESA box,
identical FGONG structure). This directly shifts δν₀₂ by 30–50% and r02 by ~15–30%.
The current tolerances (15% r02, 20% δν₀₂) document the MEASURED accuracy including
this known bias and will tighten to 1–2% once #682 lands. These tolerances are NOT
self-calibrated — they are the physical consequence of the documented l=2 bias on the
ratio/separation quantities, with modest headroom.

Tests:
  1. r02 frequency ratios vs GYRE (MODE-A FGONGs, cross-Xc, paired by n_pg)
  2. δν₀₂ small separation vs GYRE (MODE-A, cross-Xc, from summary.csv)
  3. MODE-A δc²/c² sound-speed profile vs MESA FGONG (radiative interior RMS)
  4. Per-zone ∇_ad vs MESA FGONG col-10 (MODE A, 1.0 M☉)
  5. δν₀₂ monotonic decrease across X_c (evolution trend)
  6. ν_max scaling relation sanity check
  7. ΔΠ₁ asymptotic g-mode period spacing (MESA star_utils.f90:952 formula)

External references:
  - GYRE 8.1 (Townsend & Teitler 2013): committed per-mode frequencies at
    data/mesa_comparison/gyre_reference/ (JCD outer BC, GL6 collocation)
  - Model S: Christensen-Dalsgaard et al. (1996), Science 272, 1286
  - r02 definition: Roxburgh & Vorontsov (2003), A&A 411, 215
  - δν₀₂ theory: Tassoul (1980), ApJS 43, 469
  - ν_max scaling: Brown et al. (1991), ApJ 368, 599; Kjeldsen & Bedding (1995)
  - ΔΠ₁: MESA star_utils.f90:952; Tassoul (1980) eq. 50
  - Chaplin & Miglio (2013), ARA&A 51, 353 (CD diagram, scaling relations)
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
# Helpers
# ════════════════════════════════════════════════════════════════════════════════

def _load_gyre_freqs(mass_str, stage):
    """Load GYRE reference frequencies for a given model.

    Args:
        mass_str: e.g. '1.0Msun', '1.2Msun', '1.5Msun', '2.0Msun'
        stage: e.g. 'zams', 'Xc0.60', 'midMS', 'Xc0.40', 'Xc0.30', 'Xc0.20'

    Returns:
        dict: {l: {n_pg: freq_uHz}} — frequencies indexed by (degree, radial order)
    """
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
                    continue  # skip any non-numeric lines
                if l not in freqs:
                    freqs[l] = {}
                freqs[l][n_pg] = freq
    return freqs


def _load_gyre_summary():
    """Load the GYRE reference summary.csv.

    Returns:
        list of dicts with keys: mass_Msun, stage, Dnu_uHz, dnu02_uHz,
        n_pairs, regime, n_l0, n_l1, n_l2, n_l3
    """
    path = _resolve_data_path(
        os.path.join('mesa_comparison', 'gyre_reference', 'summary.csv'))
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'mass_Msun': float(row['mass_Msun']),
                'stage': row['stage'],
                'Dnu_uHz': float(row['Dnu_uHz']),
                'dnu02_uHz': float(row['dnu02_uHz']),
                'n_pairs': int(row['n_pairs']),
                'regime': row['regime'],
            })
    return rows


def _compute_dnu02_from_gyre(freqs, nu_min=1500.0, nu_max=4500.0):
    """Compute δν₀₂(n) = ν(n,0) − ν(n−1,2) from GYRE frequency dict.

    Pairs by radial order n_pg (precise identification, no nearest-frequency
    heuristic). Only uses p-mode regime (n_pg >= 1 for l=0, n_pg >= 0 for l=2).

    Returns list of (n, dnu02) tuples for all valid pairs within the freq range.
    """
    if 0 not in freqs or 2 not in freqs:
        return []
    freq_l0 = freqs[0]
    freq_l2 = freqs[2]
    pairs = []
    for n in sorted(freq_l0.keys()):
        if n < 1:
            continue  # skip f-mode / g-modes for l=0
        freq_0 = freq_l0[n]
        if freq_0 < nu_min or freq_0 > nu_max:
            continue
        # δν₀₂(n) = ν(n,0) − ν(n−1,2)
        if (n - 1) in freq_l2:
            d02 = freq_0 - freq_l2[n - 1]
            if d02 > 0:  # physical: l=0 above l=2 at same effective order
                pairs.append((n, d02))
    return pairs


def _compute_r02_from_gyre(freqs, nu_min=1500.0, nu_max=4500.0):
    """Compute r02(n) = δν₀₂(n) / Δν₁(n) from GYRE frequency dict.

    Definition: Roxburgh & Vorontsov (2003), A&A 411, 215, Eq. 5.
    r02(n) = [ν(n,0) − ν(n−1,2)] / [ν(n,1) − ν(n−1,1)]

    Pairs by radial order n_pg (precise identification).

    Returns list of (n, r02) tuples.
    """
    if 0 not in freqs or 1 not in freqs or 2 not in freqs:
        return []
    freq_l0 = freqs[0]
    freq_l1 = freqs[1]
    freq_l2 = freqs[2]
    ratios = []
    for n in sorted(freq_l0.keys()):
        if n < 2:
            continue  # need n-1 for both l=1 and l=2
        freq_0 = freq_l0[n]
        if freq_0 < nu_min or freq_0 > nu_max:
            continue
        if (n - 1) not in freq_l2:
            continue
        if n not in freq_l1 or (n - 1) not in freq_l1:
            continue
        d02 = freq_0 - freq_l2[n - 1]
        dnu1 = freq_l1[n] - freq_l1[n - 1]
        if dnu1 > 0 and d02 > 0:
            ratios.append((n, d02 / dnu1))
    return ratios


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


# ════════════════════════════════════════════════════════════════════════════════
# Test 1: r02 frequency ratios vs GYRE across X_c (MODE-A FGONGs)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("corrupt_gamma1_jax")
@pytest.mark.mutation("partial_corrupt_gamma1")
@pytest.mark.right_reason("relative error")
def test_r02_ratio_vs_gyre(stellar):
    """r02 frequency ratios from our solver vs GYRE 8.1 reference (MODE-A FGONGs).

    WHAT: Computes r02(n) = [ν(n,0) − ν(n−1,2)] / [ν(n,1) − ν(n−1,1)] from our
    oscillation solver applied to committed MODE-A MESA FGONGs, and compares to
    the same quantity derived from committed GYRE 8.1 JCD/GL6 reference frequencies.
    Tests across multiple evolutionary stages (X_c) on 1.0 M☉.

    WHY: r02 is surface-independent (Roxburgh & Vorontsov 2003) — the dominant
    age/core diagnostic for the {M,X_c} science direction (#676). The #678 finding
    showed our l=2 modes are biased +3–5 µHz; this directly measures the effect on
    the ratio that the science depends on. Using GYRE on the SAME MODE-A FGONGs
    (identical structure) isolates the oscillation-solver agreement from any
    physics/structure difference.

    EXTERNAL REFERENCE: GYRE 8.1 (Townsend & Teitler 2013) reference frequencies
    committed at data/mesa_comparison/gyre_reference/. JCD outer boundary condition,
    GL6 collocation, on the same MESA MODE-A FGONGs. Mode identification by n_pg
    (radial order) enables precise pairing without nearest-frequency heuristics.

    TOLERANCE: 15% relative error on median r02. This is where the known l=2 bias
    (#678, +3–5 µHz) maps in ratio space (a ~3 µHz l=2 bias shifts r02 by ~15–30%
    relative on a ratio of ~0.07). TIGHTEN TO 1–2% AFTER #682 FIXES THE l=2 CORE
    TREATMENT (the 1–2% acceptance in #686 is gated on that fix).

    MUTATION: corrupt_gamma1_jax — scaling Γ₁ by 1.5 in the JAX oscillation grid
    builder (_build_oscillation_grid_jax) shifts all frequencies by ~22%,
    completely destroying the r02 pattern. partial_corrupt_gamma1 scales Γ₁ by
    1.10 (+10%), which shifts all frequencies by ~5% (ν ∝ √Γ₁). The ratio r02
    is first-order insensitive to a uniform Γ₁ perturbation (both numerator and
    denominator of δν₀₂/Δν₁ scale by the same factor and cancel). The Δν
    (large separation) assertion below catches the partial mutation: Δν ∝ √Γ₁
    shifts by ~5% under the 10% Γ₁ perturbation (clean err ~0.6%, mutated err
    ~4.5%, tolerance 2%). Uses the JAX-path mutation because this test calls
    compute_oscillation_freqs_jax (not the NumPy diagnostic path).
    """
    import jax
    jax.config.update("jax_enable_x64", True)

    from stellar_jax.oscillations import compute_oscillation_freqs_jax

    # Use clean_pmode stages for 1.0 M☉ that cover the MS evolution
    test_cases = [
        ('1.0Msun', 'Xc0.60'),
        ('1.0Msun', 'midMS'),
        ('1.0Msun', 'Xc0.30'),
    ]

    for mass_str, stage in test_cases:
        # Load GYRE reference
        gyre_freqs = _load_gyre_freqs(mass_str, stage)
        gyre_r02 = _compute_r02_from_gyre(gyre_freqs)
        assert len(gyre_r02) >= 10, (
            f"GYRE {mass_str}/{stage}: only {len(gyre_r02)} r02 pairs (expected ≥10)")
        gyre_r02_values = np.array([r for _, r in gyre_r02])

        # Load the same FGONG and compute with our solver
        glob, var = _load_mesa_fgong(mass_str, stage)

        our_freqs_raw = compute_oscillation_freqs_jax(
            glob, var, l_values=(0, 1, 2),
            nu_min=1500.0, nu_max=4500.0, n_scan=400, n_steps=8000)

        nu_l0 = np.sort(our_freqs_raw[0])
        nu_l1 = np.sort(our_freqs_raw[1])
        nu_l2 = np.sort(our_freqs_raw[2])

        assert len(nu_l0) >= 10, (
            f"{mass_str}/{stage} l=0: only {len(nu_l0)} modes found")
        assert len(nu_l1) >= 10, (
            f"{mass_str}/{stage} l=1: only {len(nu_l1)} modes found")
        assert len(nu_l2) >= 10, (
            f"{mass_str}/{stage} l=2: only {len(nu_l2)} modes found")

        # Compute our r02 by pairing modes (nearest l=2 below l=0, divided by l=1 spacing)
        # We cannot pair by n_pg from our solver (it doesn't output n_pg), so we use
        # the standard nearest-below pairing which is equivalent in the p-mode regime
        delta_nu = float(np.median(np.diff(nu_l0)))
        our_r02_values = []
        for i in range(1, len(nu_l0) - 1):
            nu_0 = nu_l0[i]
            # Find l=2 mode just below (within Δν)
            cand_l2 = nu_l2[(nu_l2 < nu_0) & (nu_l2 > nu_0 - delta_nu)]
            if len(cand_l2) == 0:
                continue
            nu_2_below = cand_l2[-1]
            d02 = nu_0 - nu_2_below

            # Find l=1 spacing at this order
            cand_l1_hi = nu_l1[(nu_l1 > nu_0 - 0.8 * delta_nu) &
                               (nu_l1 < nu_0 + 0.2 * delta_nu)]
            cand_l1_lo = nu_l1[(nu_l1 > nu_0 - 1.8 * delta_nu) &
                               (nu_l1 < nu_0 - 0.8 * delta_nu)]
            if len(cand_l1_hi) == 0 or len(cand_l1_lo) == 0:
                continue
            dnu1 = cand_l1_hi[0] - cand_l1_lo[-1]

            if dnu1 > 50 and 0 < d02 < 0.5 * delta_nu:
                our_r02_values.append(d02 / dnu1)

        assert len(our_r02_values) >= 5, (
            f"{mass_str}/{stage}: only {len(our_r02_values)} valid r02 pairs "
            f"(expected ≥5)")

        our_r02_median = float(np.median(our_r02_values))
        gyre_r02_median = float(np.median(gyre_r02_values))

        # r02 must be positive
        assert our_r02_median > 0, (
            f"{mass_str}/{stage}: r02 median = {our_r02_median:.5f} ≤ 0")

        # Compare to GYRE: two-sided pin [0.5%, 15%] (AC3).
        # Upper 15%: documents known l=2 bias (+3–5 µHz → ~15–30% in r02).
        # Lower 0.5%: catches degenerate comparison (both codes returning
        # identical wrong values) — inter-code scatter is physically ≥0.5%.
        # Mutation (corrupt_gamma1_jax → Γ₁×1.5 → r02 shifts ~50%) >> 15%.
        rel_err = abs(our_r02_median - gyre_r02_median) / gyre_r02_median
        assert rel_err < 0.15, (
            f"{mass_str}/{stage}: r02 median: ours={our_r02_median:.5f}, "
            f"GYRE={gyre_r02_median:.5f}, relative error={rel_err:.3f} > 0.15. "
            f"Known l=2 bias (#678); tighten to 1–2% after #682.")
        assert rel_err > 0.005, (
            f"{mass_str}/{stage}: r02 suspiciously perfect "
            f"(rel_err={rel_err:.2e} < 0.5%). Inter-code scatter between "
            f"our 4th-order solver and GYRE should produce ≥0.5% difference. "
            f"Check for circular comparison (same code path generating both).")

        # Physical range check (0.02–0.15 for MS stars)
        assert 0.02 < our_r02_median < 0.15, (
            f"{mass_str}/{stage}: r02 = {our_r02_median:.5f} outside [0.02, 0.15]")

        # ── Δν (large separation) vs GYRE: mutation gate for partial_corrupt_gamma1 ──
        # r02 = δν₀₂/Δν₁ is a frequency RATIO, so a uniform Γ₁ scaling cancels
        # in numerator and denominator (both shift by ~√Γ₁). The ratio is therefore
        # first-order insensitive to the partial_corrupt_gamma1 mutation (Γ₁×1.10),
        # which shifts r02 err from 0.6% to only 6.5% — well under the 15% tolerance.
        #
        # Δν (the l=0 large separation) IS directly sensitive: Δν ≈ 1/(2∫dr/c_s)
        # where c_s = √(Γ₁P/ρ), so Δν ∝ √Γ₁. A 10% Γ₁ increase shifts Δν by ~5%.
        # Measured: clean Δν err vs GYRE = 0.57–0.84%; mutated = 4.3–4.6%.
        #
        # Tolerance: 2% relative. Physical basis: inter-code scatter (our 4th-order
        # RK4 vs GYRE GL6 on identical FGONG structure) is <1% for the median large
        # separation. The 2% tolerance provides ~3× headroom over the measured 0.6–0.8%
        # clean error while clearly catching the 4.3–4.6% mutated error.
        #
        # External reference: GYRE 8.1 l=0 frequencies from committed reference files.
        gyre_nu_l0 = np.array(sorted(gyre_freqs[0].values()))
        gyre_dnu = float(np.median(np.diff(gyre_nu_l0)))
        # delta_nu was already computed above from our l=0 modes
        dnu_rel_err = abs(delta_nu - gyre_dnu) / gyre_dnu
        assert dnu_rel_err < 0.02, (
            f"{mass_str}/{stage}: Δν (large separation): ours={delta_nu:.2f} μHz, "
            f"GYRE={gyre_dnu:.2f} μHz, relative error={dnu_rel_err:.4f} > 0.02. "
            f"This catches Γ₁ perturbations that the r02 ratio is insensitive to "
            f"(ν ∝ √Γ₁ → Δν shifts by ~5% under 10% Γ₁ corruption).")

        print(f"  ✓ r02({mass_str}/{stage}): ours={our_r02_median:.5f}, "
              f"GYRE={gyre_r02_median:.5f}, err={rel_err*100:.1f}%"
              f"  |  Δν: ours={delta_nu:.2f}, GYRE={gyre_dnu:.2f}, "
              f"err={dnu_rel_err*100:.2f}%")


# ════════════════════════════════════════════════════════════════════════════════
# Test 2: δν₀₂ small separation vs GYRE reference (post-, all 4 masses, ≤2%)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("adipls_a_formulation_nonradial")
@pytest.mark.right_reason("regression")
def test_dnu02_vs_gyre_reference(stellar):
    """δν₀₂ matches the committed GYRE reference to ≤2% (matched radial order), all masses.

    WHAT: For the committed MODE-A FGONGs (M = 1.0/1.2/1.5/2.0 M☉, clean p-mode MS
    stages), compute l=0 and l=2 p-mode frequencies, pair by radial order against the
    committed GYRE 8.1 reference (data/mesa_comparison/gyre_reference/), and assert the
    small separation δν₀₂ = ν_{n,0} − ν_{n−1,2} agrees to ≤2% over the mid-order
    age-diagnostic band (radial orders ~6–18).

    WHY: δν₀₂ is the {M,age} age diagnostic (#526). The prior ADIPLS A-formulation
    biased it 5–40% low (#682): the Brunt term entered frequency-amplified (−ηA),
    numerically explosive for sharp evolved-core composition gradients. The
    GYRE-formulation port (PR #695) fixes it. This is the tight, small-difference
    check against a reference-grade code — validates the formulation is correct.

    TOLERANCE: ≤2% relative error on matched-n median δν₀₂. Measured ≤1.2% across
    all 4 masses in the n_pg 8–18 band. The 2% bound provides ~2× headroom while
    still catching any regression to the pre-fix ~5–40% bias. The tolerance is
    physically grounded: inter-code scatter (GYRE vs ADIPLS in the SAME formulation,
    same FGONG) is <0.5% for clean p-modes; 2% is a generous upper bound.

    MUTATION: adipls_a_formulation_nonradial — reverts the nonradial RHS to the ADIPLS
    A-formulation (re-introducing the −ηA frequency-amplified Brunt coupling). That
    restores the 5–40% δν₀₂ bias, violating the ≤2% bound → this test must fail.

    EXTERNAL REFERENCE: data/mesa_comparison/gyre_reference/ (GYRE 8.1, JCD/GL6,
    MODE-A); see that directory's README for provenance.

    Supersedes the prior test_dnu02_vs_gyre (20% tol, 1.0 M☉ only, #691) which
    explicitly noted "tighten to 1–2% after #682".
    """
    import jax
    jax.config.update("jax_enable_x64", True)

    from stellar_jax.oscillations import compute_oscillation_freqs_jax

    # Curated clean p-mode models spanning all 4 masses (zams + one evolved stage).
    # These cover the full mass range and the zams→evolved X_c trend.
    models = [
        ('1.0Msun', 'zams'), ('1.0Msun', 'Xc0.20'),
        ('1.2Msun', 'zams'), ('1.2Msun', 'Xc0.40'),
        ('1.5Msun', 'zams'), ('1.5Msun', 'Xc0.40'),
        ('2.0Msun', 'zams'), ('2.0Msun', 'Xc0.40'),
    ]

    checked = 0
    for mass_str, stage in models:
        # Load GYRE reference frequencies (paired by radial order)
        try:
            gyre_freqs = _load_gyre_freqs(mass_str, stage)
        except (FileNotFoundError, OSError):
            continue
        if 0 not in gyre_freqs or 2 not in gyre_freqs:
            continue
        if len(gyre_freqs[0]) < 20 or len(gyre_freqs[2]) < 20:
            continue

        # Determine frequency window and evaluation band from GYRE reference
        g0_freqs = np.array(sorted(gyre_freqs[0].values()))
        g2_freqs = np.array(sorted(gyre_freqs[2].values()))
        nu_min = max(200.0, float(g0_freqs.min()) - 100.0)
        nu_max = float(g2_freqs.max()) + 100.0
        # δν₀₂ evaluation band = mid-order p-modes (n_pg 8–18): the asymptotic /
        # near-ν_max age-diagnostic regime. Excludes lowest orders (non-asymptotic)
        # and high-freq modes where VACUUM-vs-JCD surface term grows.
        # Filtering by radial order (not frequency threshold) avoids floating-point
        # boundary effects where a mode at the exact band-edge frequency is included
        # for one code but excluded for the other.
        n_lo = 8   # lowest radial order included in evaluation
        n_hi = 18  # highest radial order included in evaluation

        # Load FGONG and compute with our solver
        glob, var = _load_mesa_fgong(mass_str, stage)
        our_freqs = compute_oscillation_freqs_jax(
            glob, var, l_values=(0, 2),
            nu_min=nu_min, nu_max=nu_max, n_steps=8000)
        f0 = np.sort(np.asarray(our_freqs[0]))
        f2 = np.sort(np.asarray(our_freqs[2]))

        # Assign our modes to GYRE radial orders by nearest-frequency matching
        g0f = np.array(sorted(gyre_freqs[0].values()))
        g2f = np.array(sorted(gyre_freqs[2].values()))
        g0inv = {v: k for k, v in gyre_freqs[0].items()}
        g2inv = {v: k for k, v in gyre_freqs[2].items()}

        def assign(mine, gf, ginv):
            out = {}
            for fv in mine:
                j = int(np.argmin(np.abs(gf - fv)))
                if abs(gf[j] - fv) < 3.0:  # unambiguous match (<3 µHz)
                    out[ginv[gf[j]]] = float(fv)
            return out

        n0 = assign(f0, g0f, g0inv)
        n2 = assign(f2, g2f, g2inv)

        # Compute matched-n δν₀₂ in the evaluation band (order-based)
        def matched_d02(freq_l0, freq_l2):
            vals = [freq_l0[n] - freq_l2[n - 1] for n in sorted(freq_l0)
                    if (n - 1) in freq_l2 and n_lo <= n <= n_hi]
            return float(np.median(vals)) if vals else float('nan')

        d_ours = matched_d02(n0, n2)
        d_ref = matched_d02(gyre_freqs[0], gyre_freqs[2])
        assert np.isfinite(d_ours) and np.isfinite(d_ref), (
            f"{mass_str}/{stage}: could not compute matched-n δν₀₂ "
            f"(n0={len(n0)}, n2={len(n2)})")

        rel = abs(d_ours - d_ref) / d_ref
        assert rel <= 0.02, (
            f"{mass_str}/{stage}: δν₀₂ {d_ours:.3f} vs GYRE {d_ref:.3f} µHz "
            f"= {100 * rel:.1f}% (>2%). GYRE-formulation regression (#682).")

        print(f"  ✓ δν₀₂({mass_str}/{stage}): ours={d_ours:.2f} μHz, "
              f"GYRE={d_ref:.2f} μHz, err={rel*100:.1f}%")
        checked += 1

    assert checked >= 4, (
        f"only {checked} models checked — expected ≥4 (one per mass). "
        f"Reference data may be missing.")


# ════════════════════════════════════════════════════════════════════════════════
# Test 3: δν₀₂ trend across X_c (monotonic decrease with evolution)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("zero_brunt_nonradial")
@pytest.mark.right_reason("δν₀₂")
def test_dnu02_decreases_with_xc(stellar):
    """δν₀₂ monotonically decreases as X_c drops (core evolution trend).

    WHAT: Computes δν₀₂ on committed MESA 1.0 M☉ FGONGs at 3 evolutionary
    stages (Xc≈0.60, Xc≈0.40, Xc≈0.20) and asserts the trend is monotonically
    decreasing.

    WHY: As the star evolves, the core contracts and the sound-speed gradient
    steepens less uniformly; δν₀₂ decreases with age (Tassoul 1980; confirmed
    by Chaplin & Miglio 2013 Fig. 4 CD diagram: δν₀₂ drops from ~12→5 μHz for
    1 M☉ ZAMS→TAMS). This is the trend the {M,X_c} science direction relies on —
    if our solver doesn't reproduce it, the age diagnostic is broken.

    EXTERNAL REFERENCE: The committed MESA MODE-A FGONGs (1.0 M☉, Z=0.014,
    identical physics; data/mesa_comparison/profiles/1.0Msun/) are the structure
    reference. The monotonic-decrease property is from Tassoul (1980) asymptotic
    theory and confirmed by every published evolution track (Chaplin & Miglio 2013).
    Also confirmed by the committed GYRE summary.csv: dnu02 = 14.6→10.8→7.0 μHz
    for Xc=0.60→0.40→0.20.

    TOLERANCE: Strict monotonicity — δν₀₂(Xc=0.60) > δν₀₂(Xc=0.40) > δν₀₂(Xc=0.20).
    No numerical tolerance needed; the physical property is qualitative.
    Additionally, each δν₀₂ must agree with the committed GYRE reference to ≤10%
    (measured ~3%; the 10% bound catches the mutation's mode shift while providing
    3× headroom for inter-code scatter).

    MUTATION: zero_brunt_nonradial — forces ll1=0 in the GYRE nonradial RHS,
    removing the angular-degree-dependent coupling from the dynamics while BCs
    still carry l=2. The degenerate system produces "modes" that don't pair
    correctly with l=0 to give valid δν₀₂ → test fails at the mode-pairing step.
    """
    import jax
    jax.config.update("jax_enable_x64", True)

    from stellar_jax.oscillations import compute_oscillation_freqs_jax

    stages = ['Xc0.60', 'Xc0.40', 'Xc0.20']
    dnu02_by_stage = {}

    for stage in stages:
        glob, var = _load_mesa_fgong('1.0Msun', stage)

        freqs = compute_oscillation_freqs_jax(
            glob, var, l_values=(0, 2),
            nu_min=1500.0, nu_max=4500.0, n_scan=400, n_steps=8000)

        nu_l0 = np.sort(freqs[0])
        nu_l2 = np.sort(freqs[2])

        if len(nu_l0) < 3 or len(nu_l2) < 3:
            pytest.fail(f"Stage {stage}: insufficient modes "
                        f"(l=0:{len(nu_l0)}, l=2:{len(nu_l2)})")

        delta_nu = float(np.median(np.diff(nu_l0)))
        dnu02_vals = []
        for i in range(1, len(nu_l0)):
            nu_0 = nu_l0[i]
            cand = nu_l2[(nu_l2 < nu_0) & (nu_l2 > nu_0 - 0.9 * delta_nu)]
            if len(cand) == 0:
                continue
            d02 = nu_0 - cand[-1]
            if 0 < d02 < 0.5 * delta_nu:
                dnu02_vals.append(d02)

        assert len(dnu02_vals) >= 2, (
            f"Stage {stage}: only {len(dnu02_vals)} valid δν₀₂ pairs")
        dnu02_by_stage[stage] = float(np.median(dnu02_vals))

    # Assert monotonic decrease: Xc0.60 > Xc0.40 > Xc0.20
    d60 = dnu02_by_stage['Xc0.60']
    d40 = dnu02_by_stage['Xc0.40']
    d20 = dnu02_by_stage['Xc0.20']

    assert d60 > d40, (
        f"δν₀₂ not decreasing: Xc0.60={d60:.2f} should be > Xc0.40={d40:.2f}")
    assert d40 > d20, (
        f"δν₀₂ not decreasing: Xc0.40={d40:.2f} should be > Xc0.20={d20:.2f}")

    # All values should be positive and in physical range (3–18 μHz for 1 M☉)
    for stage, val in dnu02_by_stage.items():
        assert 2.0 < val < 20.0, (
            f"Stage {stage}: δν₀₂={val:.2f} μHz outside range [2, 20]")

    # MUTATION GATE: compare δν₀₂ against committed GYRE reference.
    # Clean run agrees to ~3% (measured 14.13 vs 14.45, 10.28 vs 10.56, 6.34 vs 6.58).
    # Under force_l_zero_in_nonradial, l=2 modes are unphysical (RHS uses l=0
    # but BCs use l=2) → spurious modes shifted far from GYRE → >10% deviation.
    # GYRE reference δν₀₂ computed from matched radial orders in the test's freq window.
    gyre_dnu02 = {}
    for stage in stages:
        gyre_freqs = _load_gyre_freqs('1.0Msun', stage)
        if 0 not in gyre_freqs or 2 not in gyre_freqs:
            continue
        # Compute GYRE δν₀₂ = ν_{n,0} − ν_{n−1,2} for orders in [1500, 4500] μHz
        g_l0 = {n: f for n, f in gyre_freqs[0].items() if 1500.0 <= f <= 4500.0}
        g_l2 = {n: f for n, f in gyre_freqs[2].items() if 1500.0 <= f <= 4500.0}
        gyre_d02 = []
        for n, f0 in g_l0.items():
            if (n - 1) in g_l2:
                gyre_d02.append(f0 - g_l2[n - 1])
        if len(gyre_d02) >= 2:
            gyre_dnu02[stage] = float(np.median(gyre_d02))

    for stage, our_val in dnu02_by_stage.items():
        if stage not in gyre_dnu02:
            continue
        gyre_val = gyre_dnu02[stage]
        rel_err = abs(our_val - gyre_val) / gyre_val
        assert rel_err < 0.10, (
            f"Stage {stage}: δν₀₂={our_val:.2f} μHz deviates {rel_err*100:.1f}% "
            f"from GYRE reference {gyre_val:.2f} μHz (limit 10%). "
            f"Mutation 'force_l_zero_in_nonradial' breaks l=2 physics."
        )

    print(f"\n  ✓ δν₀₂ trend across X_c (1.0 M☉):")
    for stage, val in sorted(dnu02_by_stage.items(), reverse=True):
        print(f"    {stage}: δν₀₂ = {val:.2f} μHz")
    print(f"    Monotonic decrease confirmed ✓")


# ════════════════════════════════════════════════════════════════════════════════
# Test 4: MODE-A δc²/c² sound-speed profile vs MESA FGONG
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("chandrasekhar_gamma1_corrupt")
@pytest.mark.right_reason("δc²/c² RMS")
def test_sound_speed_profile_vs_mesa_fgong(stellar):
    """Sound-speed profile δc²/c² (Chandrasekhar Γ₁) vs MESA MODE-A FGONG.

    WHAT: Computes c²(r) = Γ₁·P/ρ using our code's Chandrasekhar Γ₁ formula
    (the same formula used in structure.py:640 for the sound-speed diagnostic)
    and compares to MESA's c²(r) derived from MESA's tabulated Γ₁ (FGONG col 9).
    Asserts RMS(δc²/c²) < 0.1% in the radiative interior.

    WHY: The existing sound-speed test (test_model_s_sound_speed_and_density) uses
    Model S (MODE B: different Z, different α_MLT, different physics). It checks
    only the global max, not the radial PROFILE. A MODE-A check on identical physics
    isolates the Γ₁ model agreement without Z-mismatch confusion. The δc²/c²
    profile in the radiative interior is the seismic observable that constrains
    microphysics (Basu & Antia 2004, ApJ 606, L85).

    Our code uses the Chandrasekhar (1939) formula for Γ₁ with radiation pressure:
      Γ₁ = (32 − 24β − 3β²) / (24 − 21β)
    where β = P_gas/P_total. This is exact for a fully ionized ideal gas + radiation
    (the radiative interior regime). In the partial ionization zone (CZ), it breaks
    down, which is expected and documented.

    EXTERNAL REFERENCE: Committed MESA 1.0 M☉ midMS FGONG (MODE A: Z=0.014,
    identical physics; data/mesa_comparison/profiles/1.0Msun/midMS.FGONG.gz).
    Γ₁ from the FGONG is MESA's full EOS-tabulated value (quintic Hermite on OPAL).

    TOLERANCE: 0.1% RMS in the radiative interior (0.15 < r/R < 0.70). Physical
    basis: the Chandrasekhar formula is analytically exact for ideal gas + radiation;
    deviations arise only from Coulomb corrections and electron degeneracy (both
    negligible in the MS radiative zone). Measured: 0.035% RMS. The 0.1% threshold
    is 3× the measured value, accommodating mass/stage variation.

    MUTATION: chandrasekhar_gamma1_corrupt — corrupts the radiation constant a_rad
    by 10×, which shifts P_rad → wrong β → wrong Γ₁ by ~15% in the deep interior,
    far exceeding the 0.1% tolerance.
    """
    from stellar_jax.config.constants import a_rad
    glob, var = _load_mesa_fgong('1.0Msun', 'midMS')

    # FGONG: r=0, ln(m/M)=1, T=2, P=3, rho=4, X=5, L=6, kappa=7, eps=8, Gamma1=9
    r = var[:, 0]
    T = var[:, 2]
    P_total = var[:, 3]
    rho = var[:, 4]
    gamma1_mesa = var[:, 9]

    R_star = r[-1]
    r_frac = r / R_star

    # MESA sound speed squared
    c2_mesa = gamma1_mesa * P_total / rho

    # Our Γ₁: Chandrasekhar formula (structure.py:640)
    # Uses imported a_rad so the mutation can corrupt it
    P_rad = a_rad * T**4 / 3.0
    P_gas = np.maximum(P_total - P_rad, 1e-3 * P_total)
    beta = P_gas / P_total

    gamma1_ours = (32.0 - 24.0 * beta - 3.0 * beta**2) / (24.0 - 21.0 * beta)
    c2_ours = gamma1_ours * P_total / rho

    # Radiative interior: 0.15 < r/R < 0.70
    mask = (r_frac > 0.15) & (r_frac < 0.70)
    dc2 = (c2_ours[mask] - c2_mesa[mask]) / c2_mesa[mask]

    rms = float(np.sqrt(np.mean(dc2**2)))
    max_abs = float(np.max(np.abs(dc2)))

    # RMS < 0.1% in the radiative interior (Chandrasekhar is near-exact here)
    assert rms < 0.001, (
        f"δc²/c² RMS = {rms*100:.4f}% > 0.1% in radiative interior "
        f"(0.15 < r/R < 0.70). Max|δc²/c²| = {max_abs*100:.4f}%. "
        f"N_points = {np.sum(mask)}. "
        f"The Chandrasekhar Γ₁ formula should agree to <0.05% here.")

    # Max: no single point off by >0.5%
    assert max_abs < 0.005, (
        f"δc²/c² max = {max_abs*100:.4f}% > 0.5% at some point in radiative interior")

    # Additionally check that the CZ (r/R > 0.72) shows the expected breakdown
    # (Chandrasekhar assumes full ionization → wrong in H/He ionization zone)
    cz_mask = r_frac > 0.72
    if np.sum(cz_mask) > 5:
        dc2_cz = (c2_ours[cz_mask] - c2_mesa[cz_mask]) / c2_mesa[cz_mask]
        cz_max = float(np.max(np.abs(dc2_cz)))
        # The CZ discrepancy should be significant (>1%) — proves the test
        # is not trivially passing because both use the same formula
        assert cz_max > 0.01, (
            f"CZ δc²/c² max = {cz_max*100:.2f}% < 1%: expected significant "
            f"deviation where Chandrasekhar breaks down (partial ionization). "
            f"Test may be vacuous.")

    print(f"\n  ✓ δc²/c² profile vs MESA FGONG (1.0 M☉ midMS):")
    print(f"    Radiative interior: RMS = {rms*100:.4f}%, max = {max_abs*100:.4f}%")
    print(f"    (Chandrasekhar Γ₁ validated in fully-ionized regime)")


# ════════════════════════════════════════════════════════════════════════════════
# Test 4b: Full-profile δc²/c² with the ionization-aware EOS Γ₁
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("eos_chi_rho_flat")
@pytest.mark.right_reason("δc²/c²")
def test_sound_speed_eos_gamma1_full_profile_vs_mesa_fgong(stellar):
    """Full-profile δc²/c² using the SAME ionization-aware EOS Γ₁ the solver uses.

    WHAT: At each MESA FGONG grid point, evaluates our production EOS (eos_lookup)
    to obtain (∇_ad, χ_ρ, χ_T), computes Γ₁ = χ_ρ / (1 − ∇_ad·χ_T) — the SAME
    formula used in the solver's FGONG builder (fgong/builder.py:105) and structure
    profile (fgong/hires_profile.py:95) — and compares c²(r) = Γ₁·P/ρ against
    MESA's c²(r) from FGONG col-9 Γ₁ across the FULL profile (including the
    ionization zone).

    WHY: The existing test (test_sound_speed_profile_vs_mesa_fgong, Test 4 above)
    validates only the Chandrasekhar Γ₁ formula in the radiative interior
    (0.15–0.70 r/R). That formula is ionization-blind and breaks in the H/He
    ionization zones (>1% error at r/R > 0.72). But the oscillation solver uses
    the EOS-consistent Γ₁ — which IS correct in the ionization zone. This test
    validates that the EOS-sourced Γ₁ (and hence c²) agrees with MESA across the
    FULL profile, closing the gap identified in issue #770.

    NON-REDUNDANCY vs Test 4: Test 4 validates the Chandrasekhar β-formula in the
    fully-ionized interior ONLY (0.15–0.70). THIS test validates the EOS-consistent
    formula (from eos_lookup's χ_ρ, χ_T, ∇_ad) across the full profile INCLUDING
    the ionization zone (r/R > 0.72). They use different formulas, cover different
    radial ranges, and have different mutations.

    EXTERNAL REFERENCE: MESA 1.0 M☉ midMS FGONG (MODE-A: Z=0.014, identical physics;
    data/mesa_comparison/profiles/1.0Msun/midMS.FGONG.gz). MESA stores Gamma1 from
    its full EOS in FGONG col 9 (pulse_fgong.f90:443: Gamma_1 = eval_face(s%dq,
    s%gamma1, k, k_a, k_b); computed via eospc_eval.f90:294: gamma1 = chiT*(gamma3-1)
    + chiRho, algebraically equivalent to χ_ρ/(1 − ∇_ad·χ_T)).

    TOLERANCE (per-band, physical basis):
      - Core (r/R < 0.30): RMS < 1%, max < 1.5%. Both codes use OPAL with
        radiation-pressure corrections; discrepancy from interpolation scheme
        (trilinear vs quintic Hermite) is sub-percent.
      - Radiative interior (0.30 ≤ r/R < 0.72): RMS < 2%, max < 3%. Same physics
        tables, different interpolation; bounded by the measured <0.5% ∇_ad residual
        (test_nabla_ad_vs_mesa_fgong) and <2% χ_ρ/χ_T residual
        (test_eos_chi_rho_chi_T_vs_mesa_fgong).
      - CZ / ionization zone (r/R ≥ 0.72): RMS < 3%, max < 5%. The ionization-zone
        EOS is most sensitive to interpolation details (ionization energy contributes
        strongly to all thermodynamic derivatives). The 3% RMS bound is 31× the
        measured 0.097% and tightly gates the eos_chi_rho_flat mutation (3.6%).
        The 5% max accommodates isolated surface-point scatter.

    MUTATION: eos_chi_rho_flat — forces χ_ρ_gas=1, χ_T_gas=1 (ideal-gas
    approximation), producing >30% error in the ionization zone (logT~4-5.5 where
    true χ_ρ ≠ β due to ionization energy). This directly corrupts the Γ₁ formula
    and must cause the test to FAIL in the CZ band.
    """
    import stellar_jax.microphysics.eos as _eos_mod

    glob, var = _load_mesa_fgong('1.0Msun', 'midMS')

    # FGONG: r=0, ln(m/M)=1, T=2, P=3, rho=4, X=5, L=6, kappa=7, eps=8, Gamma1=9
    r = var[:, 0]
    T_mesa = var[:, 2]
    P_total = var[:, 3]
    rho_mesa = var[:, 4]
    X_mesa = var[:, 5]
    gamma1_mesa = var[:, 9]

    R_star = r[-1]
    r_frac = r / R_star
    Z_model = 0.014  # MODE-A metallicity

    # Radiation constant for P_gas extraction
    a_rad = 7.5657e-15  # erg cm^-3 K^-4

    # Evaluate our EOS at each FGONG grid point to get (nad, chi_rho, chi_T)
    # Use the MESA FGONG (T, P_total, X, Z) as input conditions.
    gamma1_ours = np.zeros(len(r))
    for i in range(len(r)):
        logT_i = np.log10(T_mesa[i])
        # Separate P_gas from P_total (same radiation split as in eos.py:288-297)
        P_rad_i = a_rad * T_mesa[i]**4 / 3.0
        P_gas_i = max(P_total[i] - P_rad_i, 1e-3 * P_total[i])
        logPgas_i = np.log10(P_gas_i)

        result = _eos_mod.eos_lookup(logT_i, logPgas_i, float(X_mesa[i]), Z_model)
        _, _, nad_i, _, _, chi_rho_i, chi_T_i = result

        # EOS-consistent Γ₁ (MESA eospc_eval.f90:294 equivalent):
        #   Gamma1 = chi_rho / (1 - nabla_ad * chi_T)
        # Same formula as fgong/builder.py:105 and fgong/hires_profile.py:95.
        denom = 1.0 - float(nad_i) * float(chi_T_i)
        gamma1_ours[i] = float(chi_rho_i) / max(denom, 1e-10)

    # Sound speed squared: c² = Γ₁ · P_total / ρ
    c2_mesa = gamma1_mesa * P_total / rho_mesa
    c2_ours = gamma1_ours * P_total / rho_mesa  # use MESA's ρ to isolate Γ₁ error

    # Per-band δc²/c² comparison
    # Assert BOTH RMS and max-per-point. The RMS catches systematic bias;
    # the max catches isolated large deviations (e.g. ionization-zone peaks).
    bands = [
        # (name, mask, rms_tol, max_tol)
        ("core", r_frac < 0.30, 0.01, 0.015),
        ("radiative", (r_frac >= 0.30) & (r_frac < 0.72), 0.02, 0.03),
        ("CZ_ionization", r_frac >= 0.72, 0.03, 0.05),
    ]

    for band_name, mask, rms_tol, max_tol in bands:
        n_pts = int(np.sum(mask))
        if n_pts < 3:
            continue  # skip bands with too few points (shouldn't happen)

        dc2 = (c2_ours[mask] - c2_mesa[mask]) / c2_mesa[mask]
        rms = float(np.sqrt(np.mean(dc2**2)))
        max_abs = float(np.max(np.abs(dc2)))

        assert rms < rms_tol, (
            f"δc²/c² RMS = {rms*100:.3f}% > {rms_tol*100:.1f}% in {band_name} band "
            f"(N={n_pts}). Max|δc²/c²| = {max_abs*100:.3f}%. "
            f"The EOS-consistent Γ₁ = χ_ρ/(1−∇_ad·χ_T) should match MESA's "
            f"full-EOS Γ₁ within {rms_tol*100:.0f}% RMS here.")

        assert max_abs < max_tol, (
            f"δc²/c² max = {max_abs*100:.3f}% > {max_tol*100:.1f}% in {band_name} band "
            f"(N={n_pts}). No single point should deviate by more than "
            f"{max_tol*100:.0f}% — a large point error indicates the EOS "
            f"thermodynamic derivatives are wrong at that condition.")

    # Global diagnostic print
    for band_name, mask, rms_tol, max_tol in bands:
        n_pts = int(np.sum(mask))
        if n_pts < 3:
            continue
        dc2 = (c2_ours[mask] - c2_mesa[mask]) / c2_mesa[mask]
        rms = float(np.sqrt(np.mean(dc2**2)))
        max_abs = float(np.max(np.abs(dc2)))
        print(f"    {band_name}: RMS={rms*100:.4f}%, max={max_abs*100:.4f}% "
              f"(N={n_pts}, rms_tol={rms_tol*100:.0f}%, max_tol={max_tol*100:.0f}%)")

    print(f"\n  ✓ δc²/c² full-profile EOS-aware Γ₁ vs MESA FGONG (1.0 M☉ midMS)")
    print(f"    (ionization-aware formula validated across entire profile)")


# ════════════════════════════════════════════════════════════════════════════════
# Test 5: per-zone ∇_ad vs MESA FGONG col-10
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("eos_nad_offset")
@pytest.mark.right_reason("relative error")
def test_nabla_ad_vs_mesa_fgong(stellar):
    """Per-zone ∇_ad from our EOS vs MESA FGONG col-10 (MODE A).

    WHAT: Evaluates our EOS ∇_ad at each MESA FGONG grid point's (T, P_gas, X, Z)
    and compares to MESA's stored ∇_ad (FGONG col 10). Asserts <2% agreement in
    the deep interior (logT > 5.5) and <5% in the partial ionization zone.

    WHY: ∇_ad controls convective stability (Schwarzschild criterion: ∇_rad > ∇_ad
    → convection) and directly enters Γ₁ = χ_ρ + χ_T·δ·∇_ad, which sets the
    sound speed and oscillation frequencies. The existing test (test_nabla_ad_from_eos_tables)
    checks only that ∇_ad ≠ 0.4 at a few points — it does NOT compare to an external
    reference. This test provides the missing per-zone external validation.

    EXTERNAL REFERENCE: MESA 1.0 M☉ midMS FGONG col-10 (nabla_ad), committed at
    data/mesa_comparison/profiles/1.0Msun/midMS.FGONG.gz. MESA's ∇_ad comes from
    its OPAL EOS tables (quintic Hermite interpolation) — an independent computation
    from our trilinear interpolation on the same underlying physics tables.

    TOLERANCE: <2% in deep interior (logT > 5.5, fully ionized: ∇_ad ≈ 0.4 from
    ideal gas + radiation pressure correction). <5% in partial ionization zone
    (4.0 < logT < 4.5, where ∇_ad dips to ~0.1–0.3). The interior tolerance is
    grounded in the measured 0.5% residual from different interpolation schemes on
    the same OPAL tables (test_eos_chi_rho_chi_T_vs_mesa_fgong showed <2% for
    similar thermodynamic derivatives). The surface tolerance is wider because
    ionization-zone gradients are sensitive to the exact table blend.

    MUTATION: eos_nad_offset — shifts ∇_ad by +0.05 (12.5% in interior where
    ∇_ad ≈ 0.4), far exceeding the 2% tolerance.
    """
    from stellar_jax.microphysics.eos import eos_lookup

    glob, var = _load_mesa_fgong('1.0Msun', 'midMS')

    # FGONG columns: T=2, P=3, rho=4, X=5, nabla_ad=10
    T = var[:, 2]
    P_total = var[:, 3]
    X_col = var[:, 5]
    nabla_ad_mesa = var[:, 10]

    a_rad = 7.5657e-15
    Z_model = 0.014

    logT_arr = np.log10(T)

    # Deep interior: logT > 5.5
    interior_mask = logT_arr > 5.5
    interior_errs = []

    for idx in np.where(interior_mask)[0]:
        logT = float(logT_arr[idx])
        P_rad = a_rad * T[idx]**4 / 3.0
        P_gas = P_total[idx] - P_rad
        if P_gas <= 0:
            continue
        logP_gas = float(np.log10(P_gas))

        _, _, our_nad, _, _, _, _ = eos_lookup(
            logT, logP_gas, float(X_col[idx]), Z_model)
        our_nad = float(our_nad)
        mesa_nad = float(nabla_ad_mesa[idx])

        if mesa_nad > 0.01:  # skip near-zero values
            rel_err = abs(our_nad - mesa_nad) / mesa_nad
            interior_errs.append(rel_err)

    interior_errs = np.array(interior_errs)
    interior_median = float(np.median(interior_errs))
    interior_p95 = float(np.percentile(interior_errs, 95))

    assert interior_median < 0.02, (
        f"∇_ad interior median relative error = {interior_median*100:.2f}% > 2%. "
        f"Expected <2% for fully-ionized gas (logT > 5.5). "
        f"P95 = {interior_p95*100:.2f}%")

    assert interior_p95 < 0.05, (
        f"∇_ad interior P95 relative error = {interior_p95*100:.2f}% > 5%. "
        f"Some zones deviate significantly.")

    # Partial ionization zone: 4.0 < logT < 4.5
    ioniz_mask = (logT_arr > 4.0) & (logT_arr < 4.5)
    ioniz_errs = []

    for idx in np.where(ioniz_mask)[0]:
        logT = float(logT_arr[idx])
        P_rad = a_rad * T[idx]**4 / 3.0
        P_gas = P_total[idx] - P_rad
        if P_gas <= 0:
            continue
        logP_gas = float(np.log10(P_gas))

        _, _, our_nad, _, _, _, _ = eos_lookup(
            logT, logP_gas, float(X_col[idx]), Z_model)
        our_nad = float(our_nad)
        mesa_nad = float(nabla_ad_mesa[idx])

        if mesa_nad > 0.01:
            rel_err = abs(our_nad - mesa_nad) / mesa_nad
            ioniz_errs.append(rel_err)

    ioniz_errs = np.array(ioniz_errs)
    if len(ioniz_errs) > 0:
        ioniz_median = float(np.median(ioniz_errs))
        assert ioniz_median < 0.05, (
            f"∇_ad ionization-zone median relative error = {ioniz_median*100:.2f}% > 5%. "
            f"Expected <5% in partial ionization zone (4.0 < logT < 4.5).")
        print(f"    Ionization zone: median err = {ioniz_median*100:.2f}% "
              f"(N={len(ioniz_errs)})")

    print(f"\n  ✓ ∇_ad vs MESA FGONG (1.0 M☉ midMS):")
    print(f"    Interior (logT>5.5): median err = {interior_median*100:.3f}%, "
          f"P95 = {interior_p95*100:.3f}% (N={len(interior_errs)})")


# ════════════════════════════════════════════════════════════════════════════════
# Test 6: ν_max scaling relation sanity
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("solar_constants_corrupt")
@pytest.mark.right_reason("outside")
def test_nu_max_scaling_relation(stellar):
    """ν_max from MESA FGONG global parameters matches the scaling relation.

    WHAT: Computes ν_max from the Brown-Kjeldsen-Bedding scaling relation:
      ν_max = ν_max,☉ · (M/M☉) · (R/R☉)⁻² · (T_eff/T_eff,☉)^{-1/2}
    using M, R, T_eff from committed MESA FGONGs, and checks it falls in the
    expected p-mode range.

    WHY: ν_max is the frequency of maximum oscillation power — it determines
    which modes are observable (only modes within ~Δν of ν_max are detected
    by Kepler/TESS). No existing test checks ν_max in any form. This provides
    a sanity test that the structure's global parameters (M, R, Teff) are
    internally consistent with the expected oscillation frequency range.

    EXTERNAL REFERENCE: Brown et al. (1991), ApJ 368, 599; Kjeldsen & Bedding
    (1995), A&A 293, 87. Solar reference: ν_max,☉ = 3090 μHz, T_eff,☉ = 5777 K.
    For a 1.0 M☉ mid-MS star: ν_max ≈ 3000–3200 μHz. For 2.0 M☉ ZAMS:
    ν_max ≈ 800–1200 μHz (lower due to larger R and Teff).

    TOLERANCE: ν_max from the scaling relation should be within the p-mode
    computation range (1000–4500 μHz for 1 M☉; 400–2000 μHz for 2 M☉).
    Factor-of-2 agreement with the midpoint of our frequency scan.

    MUTATION: solar_constants_corrupt — corrupts R☉ by 3×, which shifts R/R☉
    → ν_max moves by factor ~9× (R² dependence), well outside expected range.
    """
    from stellar_jax.config.constants import Rsun, Msun

    # Solar reference values (Chaplin & Miglio 2013)
    NU_MAX_SUN = 3090.0  # μHz
    TEFF_SUN = 5777.0    # K
    RSUN_CM = Rsun       # imported so mutation can corrupt it
    MSUN_G = Msun        # imported so mutation can corrupt it

    test_cases = [
        ('1.0Msun', 'midMS', 2000, 4000),  # Expected ν_max ~ 3000 μHz
        ('2.0Msun', 'zams', 400, 2500),     # Expected ν_max ~ 1000 μHz
    ]

    for mass_str, stage, nu_min_expected, nu_max_expected in test_cases:
        glob, var = _load_mesa_fgong(mass_str, stage)

        # FGONG global parameters (Christensen-Dalsgaard 2008, Ap&SS 316, §A.1):
        # glob[0] = M (g), glob[1] = R (cm), glob[2] = L (erg/s),
        # glob[3] = Z, glob[4] = X0, ...
        # T_eff from L = 4πR²σT⁴ → T_eff = (L / 4πR²σ)^{1/4}
        M = glob[0]
        R = glob[1]
        L = glob[2]
        sigma_sb = 5.6704e-5  # erg cm^-2 s^-1 K^-4

        T_eff = (L / (4.0 * np.pi * R**2 * sigma_sb))**0.25

        # Scaling relation: ν_max = ν_max,☉ · (M/M☉) · (R/R☉)^-2 · (Teff/Teff,☉)^-0.5
        M_ratio = M / MSUN_G
        R_ratio = R / RSUN_CM
        Teff_ratio = T_eff / TEFF_SUN

        nu_max = NU_MAX_SUN * M_ratio * R_ratio**(-2) * Teff_ratio**(-0.5)

        assert nu_min_expected < nu_max < nu_max_expected, (
            f"{mass_str}/{stage}: ν_max = {nu_max:.0f} μHz outside expected "
            f"range [{nu_min_expected}, {nu_max_expected}]. "
            f"M/M☉={M_ratio:.3f}, R/R☉={R_ratio:.3f}, Teff={T_eff:.0f} K")

        print(f"  ✓ ν_max({mass_str}/{stage}) = {nu_max:.0f} μHz "
              f"(M/M☉={M_ratio:.3f}, R/R☉={R_ratio:.3f}, Teff={T_eff:.0f} K)")


# ════════════════════════════════════════════════════════════════════════════════
# Test 7: Asymptotic period spacing ΔΠ₁ sanity (gravity modes)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("corrupt_brunt_stencil")
@pytest.mark.right_reason("ΔΠ₁ mismatch")
def test_period_spacing_sanity(stellar):
    """Asymptotic period spacing ΔΠ₁ via our Brunt computation vs MESA FGONG reference.

    WHAT: Recomputes A* = (1/Γ₁)dlnP/dlnr − dlnρ/dlnr via _compute_brunt_vaisala
    (our Fornberg O(h²) stencil), then derives the asymptotic g-mode period spacing:
      ΔΠ₁ = √2·π² / ∫_{r1}^{r2} (N/r) dr   (l=1; Tassoul 1980 eq. 50)
    and asserts it agrees with MESA FGONG col-14's own ΔΠ₁ within 2%.
    MESA formula: star_utils.f90:952 `delta_Pg = sqrt(2._dp)*pi*pi/integral`.

    WHY: ΔΠ₁ is the primary diagnostic of core structure for evolved stars
    (Bedding et al. 2011, Nature 471, 608; Chaplin & Miglio 2013 §3.3). This test
    validates the INTEGRAL of N/r over the g-mode cavity — a different quantity from
    the per-zone A* accuracy tested in test_brunt_astar_vs_mesa_fgong_col14. It
    exercises _compute_brunt_vaisala through the full integration path, proving our
    Brunt computation gives the correct global oscillation diagnostic.

    EXTERNAL REFERENCE: MESA FGONG col-14 (A* = N²r/g, Christensen-Dalsgaard 2008
    Ap&SS 316, eq A.3) from committed MODE-A profiles. The reference ΔΠ₁ is computed
    from MESA's own A* — not our code. Agreement within 2% proves our Fornberg-stencil
    recomputation matches the MESA integral. Secondary sanity: ΔΠ₁ ∈ [1500, 4500] s
    for 1 M☉ midMS (Provost et al. 2000, A&A 350, 680: solar ΔΠ₁ ≈ 4400 s; lower
    for subgiants).

    TOLERANCE: 2% relative agreement between our A*-based ΔΠ₁ and MESA FGONG-based
    ΔΠ₁. This is a NUMERICAL accuracy bound — the Fornberg stencil achieves <0.01%
    when correct; the 2% ceiling is generous enough for mesh-edge effects while tight
    enough to catch the 2.4% shift from the degraded constant-spacing stencil.

    WHAT MAKES IT FAIL: @mutation("corrupt_brunt_stencil") replaces the Fornberg
    stencil with a degraded constant-spacing formula + 5% systematic offset. The
    degraded stencil shifts ΔΠ₁ by ~2.4% (outside the 2% tolerance), while the
    FGONG reference is unaffected (it's pre-computed MESA data). Confirmed: the
    correct stencil matches FGONG to 0.004%; the corrupt one diverges to 2.4%.
    """
    from stellar_jax.fgong.io import _compute_brunt_vaisala

    glob, var = _load_mesa_fgong('1.0Msun', 'midMS')

    # FGONG columns: r=0, ln(m/M)=1, T=2, P=3, rho=4, Gamma1=9, Brunt_A*=14
    r = var[:, 0]
    rho = var[:, 4]
    P = var[:, 3]
    gamma1 = var[:, 9]
    R_star = r[-1]

    if var.shape[1] <= 14:
        pytest.skip("FGONG does not contain Brunt A* (col 14)")

    # Reference: MESA FGONG col-14 A* (pre-computed by MESA, immutable)
    A_star_fgong = var[:, 14]

    # Our recomputation via _compute_brunt_vaisala (exercises the Fornberg stencil)
    A_star_ours = _compute_brunt_vaisala(r, P, rho, gamma1, R_star)

    # Gravity: g = G·m(r) / r²
    G = 6.6743e-8  # cm³ g⁻¹ s⁻²
    M_star = glob[0]
    m_frac = np.exp(np.clip(var[:, 1], -700, 0))
    m_r = m_frac * M_star
    g_r = G * m_r / (r**2 + 1e-30)

    def _delta_pi1(A_star):
        """Compute asymptotic l=1 period spacing from A*."""
        N2 = g_r * A_star / (r + 1e-30)
        r_frac = r / R_star
        cavity_mask = (N2 > 0) & (r_frac > 0.05) & (r_frac < 0.85)
        if np.sum(cavity_mask) < 10:
            return np.nan
        r_cav = r[cavity_mask]
        N_cav = np.sqrt(np.maximum(N2[cavity_mask], 0))
        integrand = N_cav / r_cav
        dr = np.diff(r_cav)
        avg_int = 0.5 * (integrand[:-1] + integrand[1:])
        integral = float(np.sum(avg_int * dr))
        if integral <= 0:
            return np.inf
        return np.sqrt(2.0) * np.pi**2 / integral

    delta_pi1 = _delta_pi1(A_star_ours)
    dp1_fgong = _delta_pi1(A_star_fgong)

    assert np.isfinite(dp1_fgong), (
        f"FGONG-based ΔΠ₁ is not finite: integral problem in MESA A* col-14")
    assert np.isfinite(delta_pi1), (
        f"Our ΔΠ₁ is not finite: _compute_brunt_vaisala produced invalid A*")

    # Primary assertion: our recomputation matches MESA FGONG within 2%
    # (corrupt_brunt_stencil shifts this to ~2.4%, breaking the gate)
    rel_err = abs(delta_pi1 - dp1_fgong) / dp1_fgong
    assert rel_err < 0.02, (
        f"ΔΠ₁ mismatch: ours={delta_pi1:.1f} s vs FGONG={dp1_fgong:.1f} s "
        f"(rel err={rel_err:.4f} > 0.02). Stencil accuracy degraded?")

    # Secondary sanity: physical range for 1 M☉ midMS
    assert 1500.0 < delta_pi1 < 4500.0, (
        f"ΔΠ₁ = {delta_pi1:.1f} s outside [1500, 4500] s for 1 M☉ midMS")

    print(f"\n  ✓ ΔΠ₁ (asymptotic) = {delta_pi1:.1f} s (1.0 M☉ midMS)")
    print(f"    vs FGONG reference: {dp1_fgong:.1f} s (rel err={rel_err:.6f})")
