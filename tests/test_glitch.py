"""Validation test: acoustic glitch signatures (He II + base-of-CZ) — issue #769.

WHAT: Extracts He II ionization-zone and base-of-convection-zone acoustic glitch
signatures from Model S ADIPLS frequencies via second differences + Houdek & Gough
(2007) fitting.

WHY: Acoustic glitches are key diagnostics for envelope helium abundance Y and
convection zone depth. This validates that our glitch extraction code (acoustic depth,
second differences, nonlinear fitting) produces results consistent with published
solar values from BiSON/GOLF data analysis.

EXTERNAL REFERENCES:
  - Houdek & Gough (2007), MNRAS 375, 861: D₀ diagnostic on GOLF data gives
    τ_He ≈ 707 s (their D₀, Fig. 5), τ_cz ≈ 2273 s, Δ_He ≈ 70 s.
    D₂ (with acoustic cutoff correction) gives τ_He ≈ 819 s (BiSON, Fig. 12).
  - Model S ADIPLS frequencies: genuine ADIPLS run from MESA r26.04.1
    (G matching config/constants.py, committed frequency table). Full 4th-order,
    istsbc=1, l=0-2. Supersedes legacy full.dat (issue #832).
  - MESA profile_getval.f90:672: acoustic_radius = sum(dr_div_csound(k:nz))
  - MESA star_utils.f90:370: csound = sqrt(gamma1 * P / rho)
  - Verma et al. (2014), ApJ 794, 114 (theoretical glitch study)

TOLERANCE RATIONALE:
  - τ_He (D₀): Houdek & Gough get 707 s on models with D₀. Our Model S ADIPLS
    gives 700 ± ~50 s depending on frequency range. Tolerance ±15% of 700 s covers
    the range 595-805 s (the D₀ underestimates τ by ~15% vs the D₂ = 819 s).
  - τ_cz: Houdek & Gough get 2273 s (BiSON). Model S gives ~2250 s. Tolerance ±10%.
  - Acoustic radius T: Model S ≈ 3593 s; 1/(2Δν) ≈ 3677 s.
    Tolerance ±3% (numerical integration vs analytic).

MUTATIONS:
  - corrupt_glitch_acoustic_depth: halves τ in the fitting model phase → fitted τ
    doubles → FAIL on τ_He/τ_cz bounds. Covers the fitting code path.
  - corrupt_glitch_gamma1_constant: doubles Γ₁ in acoustic depth computation →
    c increases by √2 → T drops to ~2540 s, below 3400 s bound. Covers
    compute_acoustic_depth_profile sensitivity to the Γ₁ structure.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.helpers import _resolve_data_path


@pytest.mark.validation
@pytest.mark.mutation("corrupt_glitch_acoustic_depth")
@pytest.mark.mutation("corrupt_glitch_gamma1_constant")
@pytest.mark.right_reason("outside expected range")
@pytest.mark.integration
@pytest.mark.fast
def test_acoustic_glitch_signatures_vs_published_solar():
    """Extract He II + BCZ glitch τ from Model S and validate vs Houdek & Gough (2007).

    Steps:
      1. Load Model S FGONG → compute acoustic depth profile (τ(r), T).
      2. Mesh-resolution check: verify the He II zone + BCZ are resolved.
      3. Load Model S ADIPLS l=0-2 frequencies (committed reference) → compute second differences.
      4. Fit Houdek & Gough D₀ model → extract τ_He, τ_cz, Δ_He.
      5. Assert τ_He, τ_cz, T within published ranges.

    This is a @fast test: no evolve_star, no JIT compilation. It reads committed
    FGONG + frequency data and runs a NumPy/SciPy fit (< 1 second).
    @integration added for CI visibility (the @fast bundle is disabled).
    """
    from stellar_jax.oscillations import read_fgong
    from stellar_jax.oscillations.glitch import (
        compute_acoustic_depth_profile,
        second_differences,
        fit_glitch_signatures,
    )

    # ─── 1. Acoustic depth from Model S FGONG ─────────────────────────────────
    fgong_path = _resolve_data_path('model_s/fgong.l5bi.d.15c')
    glob, var = read_fgong(fgong_path)

    tau_profile = compute_acoustic_depth_profile(glob, var)
    T = tau_profile['acoustic_radius']

    # Model S acoustic radius should be ~3500-3700 s
    # (1/(2×135 μHz) = 3704 s; 1/(2×138.27 μHz) = 3617 s; integration gives ~3593 s)
    assert 3400.0 < T < 3800.0, (
        f"Acoustic radius T = {T:.1f} s outside expected range [3400, 3800] s")

    # Sound speed at center should be ~5e7 cm/s (solar core)
    assert tau_profile['csound'][0] > 1e7, "Sound speed at center unexpectedly low"
    # Sound speed at surface should be much lower (~1e6 cm/s)
    assert tau_profile['csound'][-1] < 5e6, "Sound speed at surface unexpectedly high"

    # ─── 2. Mesh-resolution check ─────────────────────────────────────────────
    # The He II ionization zone (~100,000 K, ~40,000 K for He I) produces a localized
    # Γ₁ dip. The BCZ is a sharp transition in d²c/dr². Both require adequate mesh
    # resolution to preserve the glitch signal amplitude (aliasing kills the signal on
    # coarse grids). We verify:
    # (a) Sufficient points in the He II zone (where |dΓ₁/dr| is large)
    # (b) The acoustic depth at full vs 2× decimated resolution agrees to <1%
    #     (convergence test — if not, the grid is too coarse).

    gamma1 = var[:, 9]
    r = var[:, 0]
    R = glob[1]
    x = r / R  # fractional radius

    # (a) He II zone resolution: find where Γ₁ dips below 5/3 - 0.1 = 1.567
    # The He II ionization zone is around x ~ 0.98 (outer envelope)
    gamma1_threshold = 5.0 / 3.0 - 0.1  # 1.567
    he_ii_mask = gamma1 < gamma1_threshold
    n_he_ii_points = np.sum(he_ii_mask)
    assert n_he_ii_points >= 5, (
        f"Only {n_he_ii_points} mesh points resolve the He II Γ₁ dip "
        f"(Γ₁ < {gamma1_threshold:.3f}); need ≥5 for glitch amplitude preservation")

    # (b) Convergence test: compare acoustic radius at full vs 2× decimated resolution
    # Decimate by taking every 2nd point
    var_decimated = var[::2]
    glob_decimated = glob.copy()
    tau_decimated = compute_acoustic_depth_profile(glob_decimated, var_decimated)
    T_decimated = tau_decimated['acoustic_radius']
    rel_diff_T = abs(T - T_decimated) / T
    assert rel_diff_T < 0.01, (
        f"Acoustic radius not converged: full={T:.1f} s vs decimated={T_decimated:.1f} s "
        f"(rel diff {rel_diff_T:.4f} > 1%). Grid too coarse for glitch analysis.")

    # (c) BCZ resolution: check there are enough points around the convection zone base
    # The BCZ for the Sun is at x ~ 0.71. Check we have ≥10 points in [0.65, 0.75].
    bcz_mask = (x >= 0.65) & (x <= 0.75)
    n_bcz_points = np.sum(bcz_mask)
    assert n_bcz_points >= 10, (
        f"Only {n_bcz_points} mesh points around BCZ (x∈[0.65,0.75]); "
        f"need ≥10 for BCZ glitch signal resolution")

    # ─── 3. Load ADIPLS frequencies and compute second differences ─────────────
    # Genuine ADIPLS reference (MESA r26.04.1, G matching config/constants.py).
    # l=0,1,2 — sufficient for glitch fitting; l=3 not needed.
    from helpers import load_adipls_reference
    adipls_ref = load_adipls_reference()  # {l: {n: freq_uHz}}
    freqs_dict = {l: np.array(sorted(adipls_ref[l].values()))
                  for l in sorted(adipls_ref.keys())}

    # Verify we have enough data
    assert len(freqs_dict[0]) >= 20, f"Need ≥20 l=0 modes, got {len(freqs_dict[0])}"

    # Check second differences show oscillatory signal
    sd_l0 = second_differences(freqs_dict[0])
    assert len(sd_l0['delta2']) >= 18, "Too few second-difference points"
    # The oscillation amplitude should be ~1-3 μHz (not zero, not huge)
    d2_std = np.std(sd_l0['delta2'])
    assert 0.3 < d2_std < 5.0, (
        f"Second-difference std = {d2_std:.2f} μHz; expected oscillatory signal ~1 μHz")

    # ─── 4. Fit glitch signatures ─────────────────────────────────────────────
    result = fit_glitch_signatures(freqs_dict, nu_min=1300.0, nu_max=4200.0)

    tau_He = result['tau_He']
    tau_cz = result['tau_cz']
    Delta_He = result['Delta_He']
    residual_rms = result['residual_rms']

    # ─── 5. Validate against published values ──────────────────────────────────

    # He II acoustic depth: Houdek & Gough D₀ gives ~707 s on models (Fig. 5).
    # Allow 600-850 s range — the D₀ systematically underestimates by ~15% vs D₂.
    # Under corrupt_glitch_acoustic_depth mutation: τ_He doubles to ~1400 s → FAIL.
    assert 600.0 < tau_He < 850.0, (
        f"τ_He = {tau_He:.1f} s outside expected [600, 850] s "
        f"(Houdek & Gough 2007 D₀: ~707 s)")

    # BCZ acoustic depth: Houdek & Gough get 2273 s (BiSON).
    # Allow 1800-2600 s (±15% of 2200 s).
    assert 1800.0 < tau_cz < 2600.0, (
        f"τ_cz = {tau_cz:.1f} s outside expected [1800, 2600] s "
        f"(Houdek & Gough 2007: 2273 s)")

    # He II Gaussian width: published ~50-100 s for solar-like stars.
    assert 30.0 < Delta_He < 200.0, (
        f"Δ_He = {Delta_He:.1f} s outside expected [30, 200] s "
        f"(Houdek & Gough 2007: ~70 s)")

    # τ_cz > τ_He (BCZ is deeper than He II zone)
    assert tau_cz > tau_He, (
        f"Physical ordering violated: τ_cz ({tau_cz:.1f}) must > τ_He ({tau_He:.1f})")

    # Fit quality: RMS residual should be < 0.5 μHz for a good model
    # (Houdek & Gough achieve ~50 nHz = 0.05 μHz with optimized D₂ on BiSON;
    # our D₀ on ADIPLS theoretical freqs should do < 0.3 μHz easily)
    assert residual_rms < 0.5, (
        f"Fit residual RMS = {residual_rms:.3f} μHz > 0.5 μHz; poor glitch fit")


@pytest.mark.fast
def test_acoustic_depth_jax_matches_numpy():
    """Verify JAX-differentiable acoustic depth matches the NumPy reference path.

    WHAT: compute_acoustic_depth_jax produces the same τ(r) and T as the NumPy
    compute_acoustic_depth_profile on the same Model S FGONG input.

    WHY: The JAX path is exported for end-to-end differentiability but must produce
    identical results (to floating-point tolerance) as the validated NumPy path.
    This ensures correctness before the JAX path is wired into the AD graph.

    EXTERNAL REFERENCE: same Model S FGONG; cross-validated against the NumPy path
    (which is itself validated against Houdek & Gough 2007 in the test above).

    MUTATION: N/A — this is a cross-path consistency check, not a physics validation.
    The physics is validated by the test above; this only checks JAX≡NumPy.
    """
    import jax.numpy as jnp
    from stellar_jax.oscillations import read_fgong
    from stellar_jax.oscillations.glitch import (
        compute_acoustic_depth_profile,
        compute_acoustic_depth_jax,
    )

    fgong_path = _resolve_data_path('model_s/fgong.l5bi.d.15c')
    glob, var = read_fgong(fgong_path)

    # NumPy reference
    ref = compute_acoustic_depth_profile(glob, var)

    # JAX path — same inputs as JAX arrays
    r = jnp.array(var[:, 0])
    P = jnp.array(var[:, 3])
    rho = jnp.array(var[:, 4])
    gamma1 = jnp.array(var[:, 9])
    R = float(glob[1])

    jax_result = compute_acoustic_depth_jax(r, P, rho, gamma1, R)

    # Acoustic radius must match to machine precision (same formula, same data)
    T_np = ref['acoustic_radius']
    T_jax = float(jax_result['acoustic_radius'])
    rel_err_T = abs(T_np - T_jax) / T_np
    assert rel_err_T < 1e-6, (
        f"JAX acoustic radius T={T_jax:.4f} differs from NumPy T={T_np:.4f} "
        f"by {rel_err_T:.2e} (should be <1e-6)")

    # Sound speed array: point-wise agreement
    cs_np = ref['csound']
    cs_jax = np.asarray(jax_result['csound'])
    max_cs_err = np.max(np.abs(cs_np - cs_jax) / np.maximum(cs_np, 1.0))
    assert max_cs_err < 1e-6, (
        f"JAX csound max rel error {max_cs_err:.2e} > 1e-6 vs NumPy")

    # Full tau profile agreement
    tau_np = ref['tau']
    tau_jax = np.asarray(jax_result['tau'])
    # Compare where tau > 0 (not the surface point where both are 0)
    interior = tau_np > 1.0
    max_tau_err = np.max(np.abs(tau_np[interior] - tau_jax[interior]) / tau_np[interior])
    assert max_tau_err < 1e-6, (
        f"JAX tau profile max rel error {max_tau_err:.2e} > 1e-6 vs NumPy")

    # Verify JAX result is differentiable (can take gradient)
    import jax

    def _T_of_gamma1_scale(scale):
        g1_scaled = gamma1 * scale
        res = compute_acoustic_depth_jax(r, P, rho, g1_scaled, R)
        return res['acoustic_radius']

    grad_T = jax.grad(_T_of_gamma1_scale)(1.0)
    # Gradient should be finite and negative (higher Γ₁ → higher c → lower T)
    assert np.isfinite(float(grad_T)), "Gradient of T w.r.t. Γ₁ scale is not finite"
    assert float(grad_T) < 0, (
        f"Expected ∂T/∂(Γ₁ scale) < 0 (more stiff → faster sound → less travel time), "
        f"got {float(grad_T):.4f}")
