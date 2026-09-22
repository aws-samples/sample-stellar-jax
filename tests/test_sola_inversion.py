"""SOLA structure inversion demo — recover known δc²/c² from AD kernels (#782).

End-to-end test: perturb Γ₁ (hence c²) in Model S by a KNOWN profile, compute
the resulting δν/ν by re-solving the perturbed model, then feed those δν/ν and
the AD kernels into a SOLA inversion to RECOVER the input perturbation. The
inversion result must match the known input to within a physically justified
tolerance.

This demonstrates the full pipeline: AD kernels → SOLA inversion → δc²/c²(r),
proving the tool is inversion-ready (acceptance criterion #2 of issue #782).

INVERSE-CRIME AVOIDANCE: the δν/ν are computed by independently re-solving the
eigenvalue problem on the perturbed model (forward recomputation), NOT from the
kernel integral. The kernel is only used inside the SOLA inversion. This makes
the test a genuine inversion exercise: the data (δν/ν) come from a different
code path than the inversion tool (kernels + SOLA coefficients).

The input perturbation is a Gaussian bump in δΓ₁/Γ₁ centered at r/R = 0.5 with
width 0.04 and amplitude 0.5%. At fixed ρ, this is identical to δc²/c² = 0.5%.
We use ~56 modes (l=0,1,2,3, n_pg in [10,23]) — enough for the SOLA averaging
kernel to localize at r/R=0.5 and recover the bump amplitude to within 40%.

MODE COUNT JUSTIFICATION: with only ~11 modes (l=0,2 only), the SOLA averaging
kernel integrates to ~0.22 and cannot localize — amplitude recovery is impossible.
With ~56 modes spanning l=0,1,2,3, the averaging kernel integrates to ~0.7–0.9
and resolves the bump. This is still far fewer than the ~2000+ modes in full
helioseismic inversions (Basu & CD 1997), but sufficient for a demo-level test.

TOLERANCE: the recovered δc²/c² at the bump center must agree with the input
within 40% relative. This accounts for the convolution dilution: a Gaussian
bump of width 0.04 convolved with a SOLA averaging kernel of target_width 0.06
gives an effective resolution of √(0.04² + 0.06²) ≈ 0.072 — the recovered
peak amplitude is diluted to ~55% of the true value (i.e. ~45% error). The 40%
bound is therefore close to the physical floor; the real test is that the
inversion sees the bump at the right location with the right sign and order
of magnitude.

External reference: Model S FGONG (Christensen-Dalsgaard et al. 1996).
Method: Pijpers & Thompson (1994), A&A 281, 231; Basu & CD (1997).

Mutation: corrupt_gamma1 — scales Γ₁ by 1.5 in _build_oscillation_grid_jax,
shifting all eigenfrequencies by ~22%. The corrupt_gamma1 mutation patches
_build_oscillation_grid_jax at ALL import sites including
oscillations.kernels (fix #1024 — the kernels import site was originally
missing, making the module-scoped fixture immune to the mutation). The
eigenfrequency consistency check in test_sola_averaging_kernel_localized
computes a FRESH eigenfrequency (after the function-scoped mutation
monkeypatch activates) and compares against the clean fixture value.
Under corruption, frequencies shift by ~3%, failing the 1% tolerance.
test_gamma1_fidelity_vs_model_s provides a complementary check comparing
the oscillation grid Vg against raw FGONG columns (independent of the
sola_kernels fixture).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('JAX_ENABLE_X64', '1')


# ─── Perturbation parameters (shared by all tests) ─────────────────────────────

_X_CENTER = 0.5
_WIDTH = 0.04
_AMPLITUDE = 0.005  # 0.5% in δΓ₁/Γ₁ = δc²/c²


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_s_fgong():
    """Load the Model S FGONG file (committed reference data)."""
    fgong_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "model_s", "fgong.l5bi.d.15c"
    )
    assert os.path.isfile(fgong_path), (
        f"Model S FGONG not found: {fgong_path}")
    from stellar_jax.fgong.io import read_fgong
    glob, var = read_fgong(fgong_path)
    return glob, var


@pytest.fixture(scope="module")
def sola_kernels(model_s_fgong):
    """Compute AD kernels for ~56 modes (l=0,1,2,3, n_pg 10-23).

    Module-scoped so the expensive kernel computation (each mode requires
    a full eigenfrequency solve + AD gradient) is done once and shared by
    all tests in the class.
    """
    from stellar_jax.oscillations.kernels import compute_structure_kernels

    glob, var = model_s_fgong

    mode_specs = []
    for l_val in (0, 1, 2, 3):
        for n in range(10, 24):
            mode_specs.append((l_val, n))

    kernel_results = []
    for l_val, n_pg in mode_specs:
        kr = compute_structure_kernels(
            glob, var, l=l_val, n_pg=n_pg,
            nu_min=1000.0, nu_max=5000.0,
            n_scan=500, n_steps=8000)
        kernel_results.append(kr)

    return mode_specs, kernel_results


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Γ₁ fidelity check (shared by assertion 5 and test_gamma1_fidelity)
# ═══════════════════════════════════════════════════════════════════════════════


def _check_gamma1_fidelity(glob, var):
    """Compare Vg from _build_oscillation_grid_jax against raw FGONG columns.

    Calls _build_oscillation_grid_jax via the MODULE ATTRIBUTE (so
    monkeypatch.setattr is visible) and compares the resulting Vg against
    a reference Vg computed directly from raw FGONG columns (not through
    the grid builder). Under corrupt_gamma1 (Γ₁×1.5), Vg drops by 33%.

    Returns:
        median_err: median |ΔVg/Vg| in the radiative interior (x in [0.05, 0.90])
    """
    import stellar_jax.oscillations.coefficients as _osc_coefficients
    from stellar_jax.config.constants import G as G_cgs

    # Call via MODULE ATTRIBUTE so monkeypatch.setattr is visible
    grid_data = _osc_coefficients._build_oscillation_grid_jax(glob, var)

    # Reference Vg from raw FGONG columns (NOT through the grid builder)
    R = float(glob[1])
    M = float(glob[0])
    r_fgong = var[:, 0]
    x_fgong = r_fgong / R
    m_fgong = np.exp(var[:, 1]) * M
    m_fgong[0] = max(m_fgong[0], 1e-10 * M)
    P_fgong = var[:, 3]
    rho_fgong = var[:, 4]
    gamma1_fgong = var[:, 9]  # The REAL Γ₁ from Model S

    Vg_ref = (G_cgs * m_fgong * rho_fgong
              / (np.maximum(r_fgong, 1.0) * gamma1_fgong
                 * np.maximum(P_fgong, 1e-30)))

    mask = x_fgong > 1e-4
    Vg_ref_grid = np.nan_to_num(Vg_ref[mask], nan=0.0,
                                 posinf=0.0, neginf=0.0)

    x_builder = grid_data['x_grid']
    Vg_builder = grid_data['Vg']
    interior = (x_builder > 0.05) & (x_builder < 0.90)

    rel_err_vg = np.abs(
        (Vg_builder[interior] - Vg_ref_grid[interior])
        / np.maximum(np.abs(Vg_ref_grid[interior]), 1e-30)
    )
    return float(np.median(rel_err_vg))


# ═══════════════════════════════════════════════════════════════════════════════
# Test: SOLA inversion recovers a known δc²/c² perturbation
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.validation
@pytest.mark.mutation("corrupt_gamma1")
@pytest.mark.integration
class TestSOLAInversionDemo:
    """SOLA inversion recovers a known δc²/c² from AD kernels (#782).

    WHAT: end-to-end structure inversion demo — perturb Model S by a known
    δΓ₁/Γ₁ profile, compute δν/ν via forward recomputation, invert with
    SOLA using AD kernels, and verify the recovered profile matches.

    WHY: this is the deliverable that makes the AD kernel tool useful for
    asteroseismology. Without a working inversion, the kernels are just
    numbers. This proves they can drive a real SOLA inversion and recover
    physically meaningful structure differences. The variational consistency
    check (kernel integral ≈ forward δν/ν) further proves the AD chain is
    correct end-to-end for ALL modes in the inversion, not just spot-checked
    modes.

    EXTERNAL REFERENCE: Model S FGONG + forward eigenvalue recomputation.
    Method: Pijpers & Thompson (1994, A&A 281, 231); Basu & CD (1997,
    astro-ph/9702162). ADIPLS gm1ker.n.d.f (Christensen-Dalsgaard 1993)
    for analytic kernel reference. MESA does NOT contain a SOLA routine —
    inversions are external tools.

    TOLERANCE: 40% relative agreement at the bump center. This is close to
    the physical floor: a bump_width=0.04 convolved with a SOLA averaging
    kernel of target_width=0.06 dilutes the peak to ~55% of the true value
    (effective width √(0.04²+0.06²) ≈ 0.072). The sign and location must
    be correct — this is a necessary condition for any scientifically useful
    inversion.

    VARIATIONAL CONSISTENCY: 3% per-mode relative agreement between kernel-
    predicted and forward-recomputed δν/ν. Clean agreement is ~0.25%
    (measured); under corrupt_gamma1 it degrades to ~8–670% (measured),
    failing the gate. The 3% bound is generous for a 0.5% perturbation
    (linearization error is O(2.5e-5)), but allows for Brent root-finding
    tolerance and grid discretization. This extends the 2-mode spot check
    in test_structure_kernels.py to the full ~56-mode set.

    MUTATION: corrupt_gamma1 invalidates the kernels → variational
    consistency fails (kernel integral diverges from forward δν/ν), AND
    the SOLA inversion produces garbage (wrong amplitude/location/sign).
    """

    def test_sola_recovers_gaussian_bump(self, model_s_fgong, sola_kernels):
        """SOLA inversion recovers a 0.5% Gaussian δc²/c² bump at r/R=0.5.

        Computes δν/ν via forward recomputation on the perturbed model,
        verifies variational consistency (kernel integral ≈ forward δν/ν
        to <3% per mode), then runs the SOLA inversion and checks:
        (a) amplitude recovery within 40% of the true peak
        (b) peak location within ±0.15 of r/R=0.5
        (c) localization: far-from-bump values < 80% of peak value
        """
        from stellar_jax.oscillations.kernels import kernel_integral
        from stellar_jax.oscillations.eigenvalue import compute_oscillation_freqs_jax
        from stellar_jax.oscillations.inversions import sola_inversion

        glob, var = model_s_fgong
        mode_specs, kernel_results = sola_kernels
        R = glob[1]
        N_modes = len(kernel_results)

        # Local aliases from module constants (the module constants are the
        # single source of truth, shared with the sola_kernels fixture)
        x_center = _X_CENTER
        width = _WIDTH
        amplitude = _AMPLITUDE

        # ─── Compute δν/ν from forward recomputation ────────────────────
        # Perturb Γ₁ in the FGONG (independent from the kernel computation)
        var_perturbed = var.copy()
        x_fgong = var[:, 0] / R
        perturbation = amplitude * np.exp(
            -0.5 * ((x_fgong - x_center) / width) ** 2)
        var_perturbed[:, 9] = var[:, 9] * (1.0 + perturbation)

        dnu_over_nu_forward = np.zeros(N_modes)
        for i, kr in enumerate(kernel_results):
            nu_ref = kr['nu']
            l_val = kr['l']

            # Re-solve on the perturbed model to find the shifted frequency
            freqs_pert = compute_oscillation_freqs_jax(
                glob, var_perturbed, l_values=(l_val,),
                nu_min=nu_ref - 30.0, nu_max=nu_ref + 30.0,
                n_scan=200, n_steps=8000)

            pert_freqs = freqs_pert[l_val]
            assert len(pert_freqs) > 0, (
                f"No modes found near ν={nu_ref:.1f} µHz for l={l_val} "
                f"on the perturbed model")

            idx = np.argmin(np.abs(pert_freqs - nu_ref))
            nu_pert = pert_freqs[idx]
            dnu_over_nu_forward[i] = (nu_pert - nu_ref) / nu_ref

        # ─── Compute δν/ν from kernel integral (perturbation theory) ─────
        x_kernel = kernel_results[0]['x']
        bump_on_kernel_grid = amplitude * np.exp(
            -0.5 * ((x_kernel - x_center) / width) ** 2)

        dnu_over_nu_kernel = np.array([
            kernel_integral(kr, bump_on_kernel_grid)
            for kr in kernel_results
        ])

        # ─── ASSERTION 1: Variational consistency across all modes ────────
        # The kernel integral must match the forward-recomputed δν/ν for
        # EVERY mode to <3%. This proves the AD chain is correct end-to-end
        # for all ~56 modes used in the inversion.
        # Clean: ~0.25% (measured). Corrupt_gamma1: ~8–670% (measured).
        for i, (l_val, n_pg) in enumerate(mode_specs):
            ratio = dnu_over_nu_kernel[i] / dnu_over_nu_forward[i]
            assert abs(1.0 - ratio) < 0.03, (
                f"Kernel integral ≠ forward δν/ν for l={l_val} n={n_pg} "
                f"(ν={kernel_results[i]['nu']:.1f} µHz):\n"
                f"  Kernel integral δν/ν = {dnu_over_nu_kernel[i]:.6e}\n"
                f"  Forward recomp δν/ν  = {dnu_over_nu_forward[i]:.6e}\n"
                f"  Ratio = {ratio:.4f} (tolerance: 0.97–1.03)")

        # ─── Run the SOLA inversion ──────────────────────────────────────
        # Use small artificial errors (the forward model has no noise)
        sigma = np.ones(N_modes) * 1e-6  # σ(δν/ν) = 1e-6 (well below signal)

        # Target points spanning the radiative interior
        target_x = np.array([0.3, 0.4, 0.5, 0.6, 0.7])

        result = sola_inversion(
            kernel_results, dnu_over_nu_forward, sigma, target_x,
            target_width=0.06, mu=1e-6)

        # ─── Validate the inversion result ───────────────────────────────
        dc2_recovered = result['dc2_over_c2']

        # ASSERTION 2: The recovered profile must be POSITIVE at the bump
        # center (the perturbation increases c²)
        idx_center = np.argmin(np.abs(target_x - x_center))
        assert dc2_recovered[idx_center] > 0, (
            f"SOLA inversion has wrong sign at x={x_center}: "
            f"recovered δc²/c² = {dc2_recovered[idx_center]:.6e}, expected > 0")

        # ASSERTION 3: The recovered amplitude at the bump center must be
        # within 40% of the true peak value. Physical floor: a bump of
        # width 0.04 convolved with averaging kernel of target_width 0.06
        # dilutes to ~55% (effective width √(0.04²+0.06²) ≈ 0.072).
        true_peak = amplitude  # 0.005 = 0.5%
        recovered_peak = dc2_recovered[idx_center]
        rel_error = abs(recovered_peak - true_peak) / true_peak

        assert rel_error < 0.40, (
            f"SOLA inversion amplitude mismatch at x={x_center}:\n"
            f"  True δc²/c² = {true_peak:.6f}\n"
            f"  Recovered   = {recovered_peak:.6f}\n"
            f"  Rel error   = {rel_error:.1%} (tolerance: 40%)")

        # ASSERTION 4: The peak of the recovered profile should be near
        # x=0.5 (±0.15 in fractional radius — accounting for the coarse
        # 5-point target grid)
        peak_x = target_x[np.argmax(dc2_recovered)]
        assert abs(peak_x - x_center) < 0.15, (
            f"SOLA inversion peak at wrong location: "
            f"peak at x={peak_x:.2f}, expected near {x_center}")

        # ASSERTION 5: Away from the bump, the recovered profile should be
        # much smaller (the perturbation is localized)
        far_from_bump = np.abs(target_x - x_center) > 0.15
        if np.any(far_from_bump):
            max_far = np.max(np.abs(dc2_recovered[far_from_bump]))
            assert max_far < 0.8 * recovered_peak, (
                f"SOLA inversion not localized: "
                f"max |δc²/c²| away from bump = {max_far:.6e}, "
                f"but peak = {recovered_peak:.6e}")

    def test_sola_averaging_kernel_localized(self, model_s_fgong, sola_kernels):
        """The SOLA averaging kernel concentrates weight near the target.

        WHAT: (a) the averaging kernel K_avg(r; r₀) = Σ c_i · K_i(r)
        concentrates its integrated weight near x₀ (assertions 1-4, using
        the module-scoped sola_kernels fixture); (b) the eigenfrequency
        solver produces consistent results when called fresh vs from the
        fixture (assertion 5, calling compute_oscillation_freqs_jax FRESH
        so the function-scoped mutation monkeypatch is active on all
        import sites including oscillations.kernels).

        WHY: a well-localized averaging kernel means the inversion
        recovers a local average of δc²/c² near the target, not a global
        or surface-dominated average (Pijpers & Thompson 1994, §3). The
        eigenfrequency consistency check (assertion 5) ensures the
        oscillation grid is built from correct physics — since
        c² = Γ₁·P/ρ, corrupted Γ₁ shifts all eigenfrequencies,
        invalidating both the kernels and the SOLA inversion.

        NOTE on kernel density vs weight: the raw kernel density K_avg(r)
        always peaks near the stellar surface on a non-uniform FGONG grid
        because the grid spacing Δr/R is ~1e-5 near r/R=1 — the density
        K = (Γ₁/ν)·∂ν/∂Γ₁·R/Δr grows with 1/Δr. The physically
        meaningful localization metric is the WEIGHT per cell, K_avg·Δr/R,
        which removes the grid-spacing artifact (Basu & CD 1997, §2.1).

        MODE SET: ~56 modes (l=0,1,2,3, n_pg 10-23) via the module-scoped
        sola_kernels fixture.

        EXTERNAL REFERENCE: Model S FGONG (Christensen-Dalsgaard et al.
        1996) for the structure model. Assertion 5 compares a fresh
        eigenfrequency against the fixture's eigenfrequency — under clean
        code they are identical; under corrupt_gamma1, the fresh solve
        uses corrupted Γ₁ (via the now-patched _build_oscillation_grid_jax
        in oscillations.kernels and oscillations.eigenvalue), shifting
        frequencies by ~3%.
        Method: Pijpers & Thompson (1994), A&A 281, 231.

        TOLERANCE: weighted-peak tolerance ±0.30 is generous — with ~56
        modes, the peak should be much closer to x=0.5. Interior weight
        fraction (x < 0.8) > 0.25 catches surface-dominated degradation.
        Per-mode kernel integrals in [0.1, 1.0] catch corrupted eigenvalues.
        Eigenfrequency consistency: 1% relative (clean: <1e-10;
        corrupt_gamma1: ~3%).

        MUTATION: corrupt_gamma1 — patches _build_oscillation_grid_jax at
        ALL import sites (including oscillations.kernels, #1024 fix). The
        module-scoped sola_kernels fixture is computed before the function-
        scoped mutation activates (clean data), but assertion 5 computes a
        FRESH eigenfrequency (after mutation) and compares against the
        fixture value. Under corruption (Γ₁×1.5), frequencies shift ~3%,
        exceeding the 1% tolerance.
        """
        from stellar_jax.oscillations.inversions import sola_inversion

        mode_specs, kernel_results = sola_kernels
        N_modes = len(kernel_results)

        # Dummy data (we only care about the averaging kernel shape)
        dnu_over_nu = np.zeros(N_modes)
        sigma = np.ones(N_modes) * 1e-5
        target_x = np.array([0.5])

        result = sola_inversion(
            kernel_results, dnu_over_nu, sigma, target_x,
            target_width=0.06, mu=1e-5)

        avg_kernel = result['averaging_kernels'][0]
        x_grid = result['x_grid']
        dr_over_R = kernel_results[0]['dr_over_R']

        # ── 1. Weighted-kernel peak should be in the target region ────────
        weighted_kernel = avg_kernel * dr_over_R
        peak_idx = np.argmax(weighted_kernel)
        x_peak = x_grid[peak_idx]
        assert abs(x_peak - 0.5) < 0.30, (
            f"Averaging kernel weighted peak at x={x_peak:.3f}, "
            f"expected within 0.30 of target 0.5")

        # ── 2. Interior weight fraction ───────────────────────────────────
        interior_mask = x_grid < 0.8
        total_weight = float(np.sum(np.abs(weighted_kernel)))
        interior_weight = float(np.sum(np.abs(weighted_kernel[interior_mask])))
        interior_frac = interior_weight / max(total_weight, 1e-30)
        assert interior_frac > 0.25, (
            f"Interior weight fraction = {interior_frac:.3f}, "
            f"expected > 0.25 (kernel is surface-dominated)")

        # ── 3. Kernel integral should be positive and substantial ─────────
        integral = float(np.sum(avg_kernel * dr_over_R))
        assert 0.5 < integral < 1.5, (
            f"Averaging kernel integral = {integral:.3f}, "
            f"expected in [0.5, 1.5]")

        # ── 4. Individual kernel integrals must be physical ───────────────
        for i, kr in enumerate(kernel_results):
            ki = float(np.sum(kr['K_gamma1_rho'] * kr['dr_over_R']))
            assert 0.1 < ki < 1.0, (
                f"Kernel integral for mode {i} (l={kr['l']}, n={kr['n_pg']}) "
                f"= {ki:.4f}, expected in [0.1, 1.0] — physical p-mode "
                f"kernel should be positive and moderate")

        # ── 5. Eigenfrequency consistency: fresh solve reproduces fixture ──
        # The module-scoped sola_kernels fixture is computed before the
        # function-scoped mutation monkeypatch activates, so assertions 1-4
        # always use clean data. This assertion computes a FRESH
        # eigenfrequency (after mutation is active) and compares it against
        # the eigenfrequency stored in the fixture's kernel result. Under
        # clean code, they match exactly (same FGONG input, same solver).
        # Under corrupt_gamma1, the fresh eigenfrequency shifts by ~3%
        # (Γ₁×1.5 → c_s×√1.5 → ν shifts non-uniformly across the
        # spectrum, measured ~3% for n_pg=15) while the fixture value is
        # clean.
        # This proves the full chain _build_oscillation_grid_jax →
        # eigenvalue → SOLA is sensitive to Γ₁ corruption.
        # Tolerance: 1% relative — clean agreement is <1e-10 (identical
        # code path); corrupt_gamma1 gives ~3% disagreement.
        from stellar_jax.oscillations.eigenvalue import compute_oscillation_freqs_jax

        glob, var = model_s_fgong
        # Pick a reference mode from the fixture (l=0, first mode)
        ref_kr = kernel_results[0]
        ref_nu = ref_kr['nu']
        ref_l = ref_kr['l']

        fresh_freqs = compute_oscillation_freqs_jax(
            glob, var, l_values=(ref_l,),
            nu_min=ref_nu - 100.0, nu_max=ref_nu + 100.0,
            n_scan=100, n_steps=8000)

        assert len(fresh_freqs[ref_l]) > 0, (
            f"No modes found near ν={ref_nu:.1f} µHz for l={ref_l} "
            f"on fresh solve (Γ₁ may be corrupted)")
        fresh_nu = fresh_freqs[ref_l][
            np.argmin(np.abs(fresh_freqs[ref_l] - ref_nu))]
        rel_freq_err = abs(fresh_nu - ref_nu) / ref_nu
        assert rel_freq_err < 0.01, (
            f"Fresh eigenfrequency disagrees with fixture value:\n"
            f"  Fixture ν = {ref_nu:.4f} µHz (clean, module-scoped)\n"
            f"  Fresh ν   = {fresh_nu:.4f} µHz (function-scoped)\n"
            f"  Rel error = {rel_freq_err:.4%} (tolerance: 1%)\n"
            f"  This indicates _build_oscillation_grid_jax is using "
            f"corrupted Γ₁ — the SOLA kernels and inversion depend on "
            f"correct sound speed c² = Γ₁·P/ρ.")

    def test_gamma1_fidelity_vs_model_s(self, model_s_fgong):
        """Oscillation grid builder faithfully reproduces Model S Γ₁.

        WHAT: a FRESH call to _build_oscillation_grid_jax validates that
        the structure coefficients (specifically Vg, which has Γ₁ in the
        denominator) faithfully reproduce the Model S Γ₁ profile — the
        EXTERNAL reference (FGONG file, Christensen-Dalsgaard et al. 1996).

        WHY: independent mutation gate for corrupt_gamma1 in this class —
        no dependency on the module-scoped sola_kernels fixture. Since
        c² = Γ₁·P/ρ, corrupted Γ₁ means corrupted sound speed, so the
        SOLA inversion recovers a meaningless δc²/c² profile.

        EXTERNAL REFERENCE: Model S FGONG (Christensen-Dalsgaard et al.
        1996). Vg = G·m·ρ / (r·Γ₁·P) — both sides use the same FGONG
        file, but only the grid builder passes through
        _build_oscillation_grid_jax (patched under mutation).

        TOLERANCE: 1% median |ΔVg/Vg| in the radiative interior
        (x in [0.05, 0.90]). Clean: <1e-12. Corrupt_gamma1 (Γ₁×1.5):
        33% — 33× above tolerance. Pure NumPy, no JIT dependency.

        MUTATION: corrupt_gamma1 — Vg_builder = Vg_ref / 1.5 → 33%
        median error >> 1% tolerance.
        """
        glob, var = model_s_fgong
        median_err = _check_gamma1_fidelity(glob, var)

        assert median_err < 0.01, (
            f"Oscillation grid Vg mismatch vs Model S FGONG:\n"
            f"  Median |ΔVg/Vg| = {median_err:.4%} (tolerance: 1%)\n"
            f"  This indicates _build_oscillation_grid_jax is using "
            f"corrupted Γ₁ — the SOLA kernels and inversion are "
            f"physically meaningless.\n"
            f"  Vg = G·m·ρ/(r·Γ₁·P); under corrupt_gamma1 (Γ₁×1.5), "
            f"Vg drops by 33%.")
