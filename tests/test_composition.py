"""Stellar-jax validation tests — composition module.

Auto-split from tests/validate.py (issue #523). Each test is independently
callable. CI discovers and runs each @smoke/@integration test as its own task.
"""
import importlib.util
import gzip
import os
import sys
import tempfile
from pathlib import Path
import subprocess
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.helpers import _resolve_data_path, _load_mesa_zams_fgong, _SIGMA_SB, _LSUN, _RSUN




@pytest.mark.fast
@pytest.mark.smoke
@pytest.mark.timeout(300)
@pytest.mark.validation
@pytest.mark.mutation("burn_composition_identity")
@pytest.mark.right_reason("Core X should decrease")
@pytest.mark.parametrize("mass_dir,profile", [
    ("1.0Msun", "Xc0.60"),
    ("2.0Msun", "Xc0.60"),
])
def test_composition_nonuniform_depletion_vs_mesa_fgong(mass_dir, profile):
    """Tier-1: burn_composition depletes X non-uniformly (not prescribed steps).

    Re-leveled from test_composition_evolves (issue #589). The original test
    called evolve_star and verified X_c decreases non-uniformly. This validates
    the SAME claim (burn is non-uniform across zones) at the component level:
    load a FGONG with per-zone eps_nuc, run burn_composition for a short dt,
    verify the resulting dX varies by zone (because eps_nuc varies by zone).

    The claim: hydrogen depletion rate is set by eps_nuc(r), which is non-uniform
    (high in the core, zero in the envelope) → dX must be non-uniform.

    External reference: MESA FGONG eps_nuc (MODE A, identical physics).
    MESA struct_burn_mix.f90:82-100 — op_split_burn depletes per-zone.
    """
    import gzip, tempfile
    import jax.numpy as jnp
    from stellar_jax.oscillations import read_fgong, fgong_components
    from stellar_jax.evolution import burn_composition
    from stellar_jax.config.constants import Q_PER_G
    from stellar_jax.config.mesh_defaults import N_COMP, COMP_MFRACS

    # Load MESA FGONG
    path = os.path.join(os.path.dirname(__file__), "..",
                        "data", "mesa_comparison", "profiles", mass_dir,
                        f"{profile}.FGONG.gz")
    assert os.path.exists(path), f"FGONG not found: {path}"
    with gzip.open(path) as fz:
        with tempfile.NamedTemporaryFile("wb", suffix=".FGONG", delete=False) as t:
            t.write(fz.read())
            tmp = t.name
    try:
        glob, var = read_fgong(tmp)
        comp = fgong_components(glob, var)
    finally:
        os.unlink(tmp)

    # Extract MESA eps_nuc and X on the FGONG grid
    m_frac_fgong = comp["m_frac"]
    eps_fgong = comp["eps_nuc"]
    X_fgong = comp["X"]

    # Interpolate onto COMP_MFRACS grid
    comp_mfracs = np.array(COMP_MFRACS)
    valid = m_frac_fgong < (1.0 - 1e-10)
    X_prof = np.interp(comp_mfracs, m_frac_fgong[valid], X_fgong[valid])
    eps_prof = np.interp(comp_mfracs, m_frac_fgong[valid], eps_fgong[valid])

    # Verify MESA has non-uniform eps_nuc (test prerequisite)
    core_eps = eps_prof[comp_mfracs < 0.2]
    surf_eps = eps_prof[comp_mfracs > 0.8]
    assert np.mean(core_eps) > 1.0, (
        f"FGONG core eps_nuc too low: {np.mean(core_eps):.3e} — bad reference")
    assert np.mean(surf_eps) < 1e-3 * np.mean(core_eps), (
        "FGONG should show eps_nuc concentrated in core, not surface")

    # Build shell_data for burn_composition (columns 0=eps, 1=m_frac)
    n_shell = 300
    idx = np.linspace(0, np.sum(valid) - 1, n_shell, dtype=int)
    shell_data = np.zeros((n_shell, 10))
    shell_data[:, 0] = eps_fgong[valid][idx][::-1]  # eps (surface→center order)
    shell_data[:, 1] = m_frac_fgong[valid][idx][::-1]  # m_frac

    # Use a short dt (1 Myr) — enough to see non-uniform depletion
    dt = 1e6 * 3.156e7  # 1 Myr in seconds

    X_profile_jnp = jnp.array(X_prof)
    # Y from FGONG: MODE A has no diffusion → Z=const=0.014, so Y = 1-X-Z
    Y_prof = 1.0 - X_prof - 0.014
    Y_profile_jnp = jnp.array(Y_prof)
    shell_data_jnp = jnp.array(shell_data)
    X_burned, _Y_burned = burn_composition(X_profile_jnp, Y_profile_jnp, shell_data_jnp, dt)
    X_burned_np = np.array(X_burned)

    # --- Key assertions (preserved from test_composition_evolves) ---
    # 1. X_c should decrease (burn is active)
    core_mask = comp_mfracs < 0.1
    X_core_init = np.mean(X_prof[core_mask])
    X_core_burned = np.mean(X_burned_np[core_mask])
    assert X_core_init > X_core_burned, (
        f"Core X should decrease: init={X_core_init:.5f}, after={X_core_burned:.5f}")

    # 2. dX steps are non-uniform (not prescribed tanh/linear schedule)
    dX = X_prof - X_burned_np
    dX_nonzero = dX[dX > 1e-15]  # zones where burning occurred
    assert len(dX_nonzero) > 5, "Should have multiple zones with active burning"
    assert not np.allclose(dX_nonzero, dX_nonzero[0], rtol=0.01), (
        "dX steps are uniform — burn_composition should produce non-uniform "
        "depletion because eps_nuc varies spatially (MESA FGONG confirms this)")


@pytest.mark.integration
def test_cno_passive_model_adequate(stellar, bundle_2p0):
    """2.0 Msun CNO-dominated HR track matches MESA (tracked N14 + CF88/NACRE rate).

    WHAT: HR track of the CNO-dominated mass (2.0 Msun, L_cno/L_nuc = 44-96%)
    matches MESA to < 0.03 dex Teff and < 0.10 dex L — the general inter-code
    scatter level for identical-physics comparisons (Paxton+ 2011 §6.2).

    WHY: 2.0 Msun is the most stringent test of the CNO energy generation because
    CNO dominates the total luminosity. Any error in the rate constant, catalyst
    abundance, or Gamow peak propagates directly into the HR track.

    EXTERNAL REFERENCE: data/mesa_comparison/results/2.0Msun/history.data
    (MESA r26.04.1, pp_cno_extras_o18_ne22.net — full CNO isotope tracking,
    Chugunov screening default since r15140).

    PHYSICS: our code uses the CF88/NACRE rate constant (8.67e27, matching
    MESA ratelib.f90:1506 rate_n14pg_nacre) with tracked N14 from burn_cno as
    the catalyst (matching MESA net_approx21.f90:1116 y(in14)*y(ih1)*rate).
    The fallback (0.251*Z for the shooting solver/Jacobian) approximates MESA's
    ZAMS X_N14 after PMS CN+ON equilibration. CONSTRAINT: the ON-cycle N14 gain
    (~5-10% of X_N14 at mid-MS) is not tracked, bounded by the 0.03/0.10
    inter-code scatter tolerance.

    TOLERANCES: 0.03 dex Teff, 0.10 dex L — same as test_mesa_comparison.
    The prior tighter values (0.025/0.06) were calibrated for the old physics
    (4.10e27 rate + X_CNO=0.69*Z fallback) where compensating errors (2x low
    rate x ~2x high catalyst) produced fortuitously small residuals. With the
    corrected rate constant and tracked N14, the residual is dominated by
    mesh/timestep/operator-split differences — the same sources as the general
    inter-code scatter.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from stellar_jax.config.mesa_config import MESA_CONFIG

    # Read MESA 2.0Msun history — same path pattern as test_mesa_comparison (proven in CI)
    mesa_file = os.path.join(os.path.dirname(__file__), "..",
                             "data", "mesa_comparison", "results",
                             "2.0Msun", "history.data")
    assert os.path.exists(mesa_file), (
        f"MESA 2.0 Msun comparison data missing: {mesa_file}")

    with open(mesa_file) as f:
        lines = f.readlines()
    for i, l in enumerate(lines):
        if l.strip().startswith("model_number"):
            header = lines[i].split()
            data_start = i + 1
            break
    data = np.loadtxt(lines[data_start:])
    col = {name: idx for idx, name in enumerate(header)}

    ref_xc = data[:, col["center_h1"]]
    ref_logT = data[:, col["log_Teff"]]
    ref_logL = data[:, col["log_L"]]
    ref_logLnuc = data[:, col["log_Lnuc"]]
    cno_col = data[:, col["cno"]]

    # Filter to settled MS
    ms = (ref_xc > 0.01) & (np.abs(ref_logL - ref_logLnuc) < 0.01)
    ref_xc, ref_logT, ref_logL = ref_xc[ms], ref_logT[ms], ref_logL[ms]

    # Confirm this mass is CNO-dominated (test prerequisite)
    L_cno = 10**cno_col[ms]
    L_nuc = 10**ref_logLnuc[ms]
    max_cno_frac = (L_cno / L_nuc).max()
    assert max_cno_frac > 0.90, (
        f"2.0 Msun should be >90% CNO at TAMS, got {max_cno_frac:.3f}")

    # Run our passive model
    #: read from session-scoped MODE-A bundle instead of running
    # evolve_star — same result (2.0 M☉, max_steps=500, **MESA_CONFIG),
    # compiled once per CI session instead of once per test.
    r = bundle_2p0
    our_xc = np.array(r["center_h1"])
    our_logT = np.array(r["log_Teff"])
    our_logL = np.array(r["log_L"])

    # Compare on common Xc grid
    xc_min = max(ref_xc.min(), our_xc.min(), 0.05)
    xc_max = min(ref_xc.max(), our_xc.max()) - 0.005
    xc_common = np.linspace(xc_max, xc_min, 30)

    dT = np.max(np.abs(
        np.interp(xc_common, our_xc[::-1], our_logT[::-1]) -
        np.interp(xc_common, ref_xc[::-1], ref_logT[::-1])
    ))
    dL = np.max(np.abs(
        np.interp(xc_common, our_xc[::-1], our_logL[::-1]) -
        np.interp(xc_common, ref_xc[::-1], ref_logL[::-1])
    ))

    # Tolerances: same as test_mesa_comparison (0.03 dex Teff, 0.10 dex L).
    # These reflect the general inter-code scatter for identical-physics
    # comparisons (Paxton+ 2011 §6.2: 0.01-0.05 dex typical). The prior tighter
    # values (0.025/0.06) were calibrated for the old physics (4.10e27 + 0.69*Z)
    # where compensating errors produced fortuitously small residuals.
    # With CF88/NACRE rate (8.67e27) + tracked N14 (matching MESA's method),
    # the residual is driven by mesh/timestep/operator-split differences.
    assert dT < 0.03, (
        f"2.0 Msun (96% CNO): dTeff={dT:.4f} dex > 0.03 — "
        f"CNO track diverges from MESA")
    assert dL < 0.10, (
        f"2.0 Msun (96% CNO): dL={dL:.4f} dex > 0.10 — "
        f"CNO track diverges from MESA")



@pytest.mark.smoke
@pytest.mark.validation
@pytest.mark.mutation("burn_cno_stop_gradient_x")
@pytest.mark.right_reason("mismatch")
def test_burn_cno_gradient_ad_vs_fd(stellar):
    """AD gradient through burn_cno matches FD — validates no stop_gradient bias.

    Issue #393 removed stop_gradient on X_profile in burn_cno's rate computation
    (the CN equilibration timescale depends on X_H). This test validates that the
    AD gradient of N14_out w.r.t. X_H agrees with central-difference FD to <5%.

    The gradient path is: X_H → rate_cn → f (relaxation factor) → N14_out.
    At CN equilibrium (f≈1), ∂N14_out/∂X_H ≈ 0 (equilibrium fractions are
    independent of X_H — they depend only on T-dependent rate ratios).
    At partial equilibration (f<1), the sensitivity is non-zero.

    Tests at TWO regimes:
    1. Partial equilibration (T=12 MK, dt=1e12 s): f < 1, gradient non-zero
    2. Full equilibration (T=20 MK, dt=1e13 s): f ≈ 1, gradient ≈ 0

    MESA reference: net_approx21.f90:1116 — eps_cno uses y(in14)*y(ih1)*rate,
    so the burn sub-step has full X_H dependence in the rate. Our AD preserves
    this sensitivity.

    The mutation 'burn_cno_stop_gradient_x' reintroduces stop_gradient on
    X_profile inside burn_cno's rate, blocking the ∂N14/∂X_H path → AD returns
    0 while FD is non-zero → assertion 3 (AD≈FD) FAILS under mutation ✓.
    """
    import jax
    import jax.numpy as jnp
    from stellar_jax.evolution import burn_cno
    from stellar_jax.config.mesh_defaults import N_COMP

    # --- Set up composition and shell data for a single test case ---
    Z = 0.014
    X_H = 0.50  # Mid-MS hydrogen

    # Shell data layout: shell_data[:, 1]=mass_frac, [:, 6]=logT, [:, 7]=logRho
    # Use T=20 MK, dt=2e11 s: partial CN equilibration (f≈0.29) where
    # ∂N14/∂X_H is measurably non-zero.
    # At T=20 MK: rate_cn ≈ 1.7e-12 s⁻¹, τ_CN ≈ 18 kyr.
    # dt = 2e11 s (6.3 kyr) → dt/τ_CN ≈ 0.35 → f ≈ 0.29
    logT_partial = np.log10(2.0e7)  # 20 MK: partial equilibration
    logRho = np.log10(100.0)  # g/cc

    shell_data = np.zeros((N_COMP, 10))
    shell_data[:, 1] = np.linspace(0.0, 1.0, N_COMP)  # mass fractions
    shell_data[:, 6] = logT_partial  # All zones at 20 MK
    shell_data[:, 7] = logRho
    shell_data = jnp.array(shell_data)

    # Initial CN composition: primordial (NOT yet equilibrated)
    C12_init = jnp.full(N_COMP, 0.170 * Z)
    C13_init = jnp.full(N_COMP, 0.002 * Z)
    N14_init = jnp.full(N_COMP, 0.049 * Z)

    # Timestep: gives f ≈ 0.29 at T=20 MK → measurable gradient
    dt = jnp.float64(2e11)  # ~6.3 kyr

    def n14_core_from_X(X_val):
        """N14 in the core zone after one burn_cno step."""
        X_prof = jnp.full(N_COMP, X_val)
        _, _, N14_out = burn_cno(C12_init, C13_init, N14_init, X_prof, shell_data, dt)
        return N14_out[0]  # Core zone 0

    # AD gradient
    X0 = jnp.float64(X_H)
    ad_grad = float(jax.grad(n14_core_from_X)(X0))

    # FD gradient (central difference)
    dX = 1e-7
    n14_plus = float(n14_core_from_X(jnp.float64(X_H + dX)))
    n14_minus = float(n14_core_from_X(jnp.float64(X_H - dX)))
    fd_grad = (n14_plus - n14_minus) / (2 * dX)

    # --- Validation ---
    # 1. AD is non-NaN (gradient flows through burn_cno without overflow)
    assert not np.isnan(ad_grad), (
        f"AD gradient through burn_cno is NaN — exponential clip not working")

    # 2. FD is non-zero (the test is measuring something real)
    assert abs(fd_grad) > 1e-15, (
        f"FD gradient is too small ({fd_grad:.3e}) — test conditions "
        f"should produce measurable ∂N14/∂X_H")

    # 3. AD matches FD to <5% (no bias from stop_gradient or numerical artifacts)
    if abs(fd_grad) > 1e-12:
        rel_err = abs(ad_grad - fd_grad) / abs(fd_grad)
        assert rel_err < 0.05, (
            f"AD-vs-FD mismatch in burn_cno gradient: "
            f"AD={ad_grad:.6e}, FD={fd_grad:.6e}, rel_err={rel_err:.4f} (>5%)")
    else:
        # If FD is very small, check absolute agreement
        assert abs(ad_grad - fd_grad) < 1e-12, (
            f"AD-vs-FD absolute mismatch: AD={ad_grad:.6e}, FD={fd_grad:.6e}")

    # 4. Sign check: higher X_H → faster equilibration → N14 moves toward N14_eq.
    # N14_eq > N14_init (0.049*Z < 0.221*Z), so ∂N14/∂X should be positive
    # (more H → faster approach to equilibrium → higher N14 after one step).
    assert ad_grad > 0, (
        f"Gradient sign wrong: ∂N14/∂X_H should be positive "
        f"(more H → faster equilibration toward N14_eq > N14_init), "
        f"got {ad_grad:.6e}")



# ═══════════════════════════════════════════════════════════════
#: Conservative composition zone masses
# ═══════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.smoke
@pytest.mark.validation
@pytest.mark.mutation("non_conservative_zone_mass")
@pytest.mark.right_reason("sum")
def test_zone_masses_conservative_sum_to_one(stellar):
    """Zone masses used for composition mixing/diffusion must sum to exactly 1.

    The finite-volume cell masses are defined by midpoint boundaries:
      bnd[0]=0, bnd[k]=0.5*(node[k-1]+node[k]) for k=1..N-1, bnd[N]=1
      zone_mass[k] = bnd[k+1] - bnd[k]

    This guarantees sum = bnd[N]-bnd[0] = 1 (telescoping).

    Matches MESA star_utils.f90:672-713 (normalize_dqs: "rescale dq's so that
    add to 1.000") and L635-648 (set_m_and_dm: dm(k) = dq(k)*xmstar).

    The test also verifies:
    - Center zone (k=0) gets half the first node gap (boundary pinned at 0).
    - Surface zone (k=N-1) gets half the last node gap (boundary pinned at 1).
    - Interior zones (k=1..N-2) get the full midpoint-to-midpoint width.
    - The fix is LIVE in mix_composition (not a dead constant).
    """
    import jax.numpy as jnp
    from stellar_jax.config.mesh_defaults import COMP_MFRACS, COMP_ZONE_MASSES, compute_zone_masses, N_COMP

    # 1. Static constant must sum to 1 within machine epsilon
    total = float(np.sum(COMP_ZONE_MASSES))
    assert abs(total - 1.0) < 1e-14, (
        f"COMP_ZONE_MASSES sum = {total} (expect 1.0 within 1e-14)")

    # 2. JAX function must give the same result for the static grid
    jax_masses = compute_zone_masses(jnp.array(COMP_MFRACS))
    jax_total = float(jnp.sum(jax_masses))
    assert abs(jax_total - 1.0) < 1e-12, (
        f"compute_zone_masses sum = {jax_total} (expect 1.0 within 1e-12)")

    # 3. Boundary zone widths: center = half first gap, surface = half last gap
    first_gap = COMP_MFRACS[1] - COMP_MFRACS[0]
    last_gap = COMP_MFRACS[-1] - COMP_MFRACS[-2]
    expected_center = first_gap / 2.0
    expected_surface = last_gap / 2.0
    assert abs(COMP_ZONE_MASSES[0] - expected_center) < 1e-15, (
        f"Center zone mass {COMP_ZONE_MASSES[0]:.6e} != half first gap {expected_center:.6e}")
    assert abs(COMP_ZONE_MASSES[-1] - expected_surface) < 1e-15, (
        f"Surface zone mass {COMP_ZONE_MASSES[-1]:.6e} != half last gap {expected_surface:.6e}")

    # 4. Interior zone (k=100): midpoint-to-midpoint = full width
    k = 100
    left_bnd = 0.5 * (COMP_MFRACS[k-1] + COMP_MFRACS[k])
    right_bnd = 0.5 * (COMP_MFRACS[k] + COMP_MFRACS[k+1])
    expected_interior = right_bnd - left_bnd
    assert abs(COMP_ZONE_MASSES[k] - expected_interior) < 1e-15, (
        f"Interior zone mass[{k}] {COMP_ZONE_MASSES[k]:.6e} != "
        f"midpoint width {expected_interior:.6e}")

    # 5. Verify the conservative weights are LIVE in mix_composition
    # (not a dead constant). Build minimal shell_data and verify the
    # mass-weighted average uses weights summing to 1.
    N_SHOOT = 300
    mf_shells = np.linspace(1.0, 0.0, N_SHOOT)
    nad = np.full(N_SHOOT, 0.4)
    nrad = np.full(N_SHOOT, 0.6)  # Everything convective
    shell_data = np.zeros((N_SHOOT, 8))
    shell_data[:, 1] = mf_shells
    shell_data[:, 2] = nrad
    shell_data[:, 3] = nad
    shell_data[:, 4] = 0.1
    shell_data[:, 5] = 1.0
    shell_data[:, 6] = 7.0
    shell_data[:, 7] = 2.0

    # Non-uniform X profile — if weights don't sum to 1, the average
    # will be biased (numerator/denominator don't cancel properly).
    comp_mfracs = np.array(COMP_MFRACS)
    X_profile = jnp.array(0.7 - 0.3 * comp_mfracs)

    # With all zones convective, mixed X should be the mass-weighted average
    X_mixed = stellar.mix_composition(X_profile, jnp.array(shell_data), M_solar=1.0)
    # The mass-weighted average with conservative weights:
    expected_avg = float(np.sum(COMP_ZONE_MASSES * np.array(X_profile)))
    # After mixing, all zones should be at this average
    X_mixed_arr = np.array(X_mixed)
    # Core is fully mixed (the entire star is convective here)
    max_dev = np.max(np.abs(X_mixed_arr - expected_avg))
    assert max_dev < 0.02, (
        f"After full mixing, max deviation from conservative average = {max_dev:.4f}; "
        f"expected < 0.02 (would be biased if weights don't sum to 1)")

    # 6. Verify on a DIFFERENT grid (adaptive mesh case) — the function still sums to 1
    custom_grid = jnp.linspace(0.0, 1.0, 50)**2  # Different shape
    custom_masses = compute_zone_masses(custom_grid)
    custom_total = float(jnp.sum(custom_masses))
    assert abs(custom_total - 1.0) < 1e-12, (
        f"compute_zone_masses on custom grid: sum = {custom_total} (expect 1.0)")



# ═══════════════════════════════════════════════════════════════
#: Soft-boundary mixing regression tests
# ═══════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.smoke
def test_mix_composition_core_homogenization(stellar):
    """Core convective zones must be FULLY homogenized (no partial mixing).

    Regression test for issue #39: the soft sigmoid implementation must not
    produce artificial composition gradients within a convective zone.
    Physically, instantaneous convective mixing homogenizes completely —
    mask ≈ 0.5 giving partial blending is unphysical.

    Uses a SMOOTH transition in nabla_rad (tanh over ~0.02 in m_frac) mimicking
    realistic stellar structure. Tests up to m_frac < 0.39 (one grid spacing
    inside the 0.4 boundary) to verify the transition zone is handled correctly.
    """
    import jax.numpy as jnp

    N_SHOOT = 300

    # Build synthetic shell_data with smooth boundary at m_frac = 0.4
    # Smooth tanh transition over ~0.05 in m_frac (realistic for a 1-1.5 Msun
    # star where nrad approaches nad gradually near the CZ boundary)
    mf_shells = np.linspace(1.0, 0.0, N_SHOOT)
    nad = np.full(N_SHOOT, 0.4)
    # nrad: 0.6 deep in core, transitions smoothly to 0.2 at m_frac=0.4
    nrad = 0.4 + 0.2 * np.tanh((0.4 - mf_shells) / 0.05)

    shell_data = np.zeros((N_SHOOT, 8))
    shell_data[:, 0] = 1e-3
    shell_data[:, 1] = mf_shells
    shell_data[:, 2] = nrad
    shell_data[:, 3] = nad
    shell_data[:, 4] = 0.1
    shell_data[:, 5] = 1.0
    shell_data[:, 6] = 7.0
    shell_data[:, 7] = 2.0

    shell_data_jnp = jnp.array(shell_data)

    # Non-uniform X profile (gradient across core)
    comp_mfracs = np.array(stellar.COMP_MFRACS)
    X_profile = jnp.array(0.7 - 0.3 * comp_mfracs)

    X_mixed = stellar.mix_composition(X_profile, shell_data_jnp, M_solar=1.5, f_ov=0.0)
    X_mixed = np.array(X_mixed)

    # Test up to m_frac < 0.39: one grid spacing inside the 0.4 boundary.
    # With cubic spacing xi^3, xi=0.73 → m_frac=0.389, so this tests zones
    # right up to the convective boundary transition.
    core_mask = comp_mfracs < 0.39
    X_core = X_mixed[core_mask]

    assert len(X_core) > 10, "Need sufficient core zones for test"
    core_spread = X_core.max() - X_core.min()
    assert core_spread < 1e-6, (
        f"Core not fully homogenized: spread = {core_spread:.2e}. "
        f"Partial mixing (mask<1 blending) is unphysical."
    )



@pytest.mark.fast
@pytest.mark.smoke
def test_mix_composition_gradient_flows(stellar):
    """jax.grad through mix_composition returns finite, non-zero values.

    Re-leveled from @integration to @fast (issue #589): this test does NOT
    use evolve_star — it calls jax.grad through mix_composition on synthetic
    shell_data. It is a pure component-level gradient test.

    This is the entire point of issue #39: replacing hard thresholds with
    smooth ops so that gradients flow through convective boundary changes.
    Tests that ∂X_mixed/∂(nabla_rad at boundary) is non-zero — meaning the
    gradient propagates through the soft Schwarzschild criterion.

    The test uses a smooth transition (tanh profile) for nabla_rad near the
    boundary, mimicking realistic stellar structure where nrad-nad passes
    through zero over a few zones.
    """
    import jax
    import jax.numpy as jnp

    N_SHOOT = 300

    # Synthetic shell_data with core CZ at m_frac < 0.3
    # Use a smooth tanh transition so the boundary has nrad≈nad over ~2 zones
    mf_shells = jnp.linspace(1.0, 0.0, N_SHOOT)
    nad = jnp.full(N_SHOOT, 0.4)
    # Smooth nrad: tanh transition from 0.2 (radiative) to 0.6 (convective) at m_frac=0.3
    # Width ~0.01 gives a ~3-zone transition region where gradient can flow
    nrad_base = 0.4 + 0.2 * jnp.tanh((0.3 - mf_shells) / 0.01)

    hp = jnp.full(N_SHOOT, 0.1)
    dmdr = jnp.full(N_SHOOT, 1.0)
    eps = jnp.full(N_SHOOT, 1e-3)
    logT = jnp.full(N_SHOOT, 7.0)
    logrho = jnp.full(N_SHOOT, 2.0)

    X_profile = jnp.array(0.7 - 0.3 * np.array(stellar.COMP_MFRACS))

    def objective(nrad_shift):
        """Scalar function: shift nabla_rad uniformly and observe composition change."""
        nrad = nrad_base + nrad_shift
        shell_data = jnp.stack([eps, mf_shells, nrad, nad, hp, dmdr, logT, logrho], axis=1)
        X_mixed = stellar.mix_composition(X_profile, shell_data, M_solar=1.5, f_ov=0.0)
        return jnp.sum(X_mixed)

    grad_val = jax.grad(objective)(0.0)
    grad_val = float(grad_val)

    assert np.isfinite(grad_val), f"Gradient is not finite: {grad_val}"
    assert abs(grad_val) > 1e-10, (
        f"Gradient is zero ({grad_val:.2e}): no gradient flow through "
        f"convective boundary. Soft ops are not working."
    )



@pytest.mark.fast
@pytest.mark.smoke
def test_mix_composition_envelope_cz_isolation(stellar):
    """Surface-connected envelope CZ is mixed; interior CZ is NOT.

    MESA mix_info.f90:1344-1381 (do_mix_envelope): only the envelope CZ that
    connects to the surface is mixed. Disconnected interior convective zones
    (e.g. from H-burning shell on the RGB) are NOT part of the envelope mixing.
    This prevents over-depletion of C12 during the first dredge-up.

    Setup: shell_data with a radiative core, two separate envelope CZs:
    CZ1 (m_frac 0.50-0.60, interior) and CZ2 (m_frac 0.85-1.00, surface-
    connected). After mixing, CZ2 is homogenized (surface CZ) and CZ1 is
    NOT mixed (interior, disconnected from surface).
    """
    import jax.numpy as jnp

    N_SHOOT = 300

    # Shell structure: radiative core + two separate envelope CZs
    mf_shells = np.linspace(1.0, 0.0, N_SHOOT)  # surface to center
    nrad = np.full(N_SHOOT, 0.2)  # default: radiative (nrad < nad)
    nad = np.full(N_SHOOT, 0.4)

    # Make two distinct envelope CZs (nrad > nad):
    # CZ1: m_frac in [0.50, 0.60] — interior, NOT surface-connected
    # CZ2: m_frac in [0.85, 1.00] — surface-connected (touches surface)
    for i in range(N_SHOOT):
        mf = mf_shells[i]
        if 0.50 <= mf <= 0.60:
            nrad[i] = 0.6  # convective (interior)
        elif 0.85 <= mf <= 1.00:
            nrad[i] = 0.6  # convective (surface)

    shell_data = np.zeros((N_SHOOT, 8))
    shell_data[:, 0] = 1e-3
    shell_data[:, 1] = mf_shells
    shell_data[:, 2] = nrad
    shell_data[:, 3] = nad
    shell_data[:, 4] = 0.1
    shell_data[:, 5] = 1.0
    shell_data[:, 6] = 7.0
    shell_data[:, 7] = 2.0

    shell_data_jnp = jnp.array(shell_data)

    # Non-uniform X: different values in each CZ region
    comp_mfracs = np.array(stellar.COMP_MFRACS)
    X_init = 0.7 - 0.5 * comp_mfracs  # 0.7 at center, 0.2 at surface
    X_profile = jnp.array(X_init)

    X_mixed = stellar.mix_composition(X_profile, shell_data_jnp, M_solar=1.0, f_ov=0.0)
    X_mixed = np.array(X_mixed)

    # Identify zones in each CZ
    cz1_mask = (comp_mfracs >= 0.50) & (comp_mfracs <= 0.60)
    cz2_mask = (comp_mfracs >= 0.85) & (comp_mfracs <= 1.00)

    X_cz1 = X_mixed[cz1_mask]
    X_cz2 = X_mixed[cz2_mask]

    # CZ2 (surface-connected) should be homogenized.
    # CONSTRAINT: smooth mixing (anti-ratchet) uses env_conv as a mixing weight
    # rather than binary homogenization. For strongly convective zones
    # (sigmoid(30*0.2) ≈ 0.9975), the residual spread is ~0.25% of the
    # original gradient — effectively homogenized but not bit-identical.
    assert len(X_cz2) > 3, f"Need zones in CZ2, got {len(X_cz2)}"
    cz2_spread = X_cz2.max() - X_cz2.min()
    assert cz2_spread < 5e-4, (
        f"Surface CZ2 not homogenized: spread = {cz2_spread:.2e}"
    )

    # CZ1 (interior, disconnected) should NOT be mixed — it retains its gradient
    assert len(X_cz1) > 3, f"Need zones in CZ1, got {len(X_cz1)}"
    cz1_spread = X_cz1.max() - X_cz1.min()
    assert cz1_spread > 1e-3, (
        f"Interior CZ1 should NOT be mixed (it's not surface-connected): "
        f"spread = {cz1_spread:.2e} — expected gradient preserved"
    )

    # CRITICAL: no cross-contamination — even though CZ2 is mixed,
    # CZ1's mean X should NOT have changed from its initial value
    X_cz1_init = np.array(X_init)[cz1_mask]
    assert np.allclose(X_cz1, X_cz1_init, atol=1e-10), (
        f"Interior CZ1 composition was modified despite being disconnected "
        f"from the surface! This violates MESA do_mix_envelope behavior."
    )

        # else: both < 0.015 dex → well-converged, no attribution check needed


@pytest.mark.fast
@pytest.mark.smoke
@pytest.mark.timeout(300)
@pytest.mark.validation
@pytest.mark.mutation("mix_composition_identity")
@pytest.mark.right_reason("overshoot should extend the mixed core")
@pytest.mark.parametrize("mass_dir,M_solar", [
    ("1.5Msun", 1.5),
    ("2.0Msun", 2.0),
])
def test_overshoot_extends_mixed_core_vs_mesa_fgong(mass_dir, M_solar):
    """Tier-1: Convective overshoot extends the mixed core for M > 1.5 M☉.

    Re-leveled from test_overshoot_material_effect_high_mass (issue #589).
    The original test called evolve_star with/without overshoot and checked they
    differ. This validates the SAME claim at the component level: load a FGONG
    with a convective core, build shell_data from it, run mix_composition with
    f_ov=0.2 vs f_ov=0.0, verify the mixed core mass fraction is LARGER with
    overshoot.

    The claim: step overshoot (f_ov > 0) extends convective mixing beyond the
    Schwarzschild boundary, increasing the mixed core mass. For M > 1.5 M☉,
    this has a "massive effect on evolution" (external physics review).

    External reference: MESA FGONG structure (MODE A, identical physics).
    MESA mix_info.f90: CZ classification + boundary detection.
    Herwig (2000), A&A 360, 952: exponential overshoot calibration.
    Claret & Torres (2016), A&A 592, A15: eclipsing binary calibration.
    Farmer et al. (2015), ApJS 220, 15: f_ov=0.016 ↔ α_ov≈0.2.
    """
    import gzip, tempfile
    import jax.numpy as jnp
    from stellar_jax.oscillations import read_fgong, fgong_components
    from stellar_jax.evolution import mix_composition
    from stellar_jax.config.mesh_defaults import N_COMP, COMP_MFRACS

    # Load MESA FGONG (midMS has an established convective core)
    path = os.path.join(os.path.dirname(__file__), "..",
                        "data", "mesa_comparison", "profiles", mass_dir,
                        "midMS.FGONG.gz")
    assert os.path.exists(path), f"FGONG not found: {path}"
    with gzip.open(path) as fz:
        with tempfile.NamedTemporaryFile("wb", suffix=".FGONG", delete=False) as t:
            t.write(fz.read())
            tmp = t.name
    try:
        glob, var = read_fgong(tmp)
        comp = fgong_components(glob, var)
    finally:
        os.unlink(tmp)

    m_frac_fgong = comp["m_frac"]
    T_fgong = comp["T"]
    rho_fgong = comp["rho"]
    X_fgong = comp["X"]
    P_fgong = comp["P"]
    gamma1_fgong = comp["gamma1"]

    # Build shell_data from FGONG structure
    # Identify convective core from Brunt A* ≈ 0 or nabla_ad (col 10) vs structure
    # The standard approach: in the FGONG, use nabla_ad (var[:,10]) and compute
    # nabla_rad from L, kappa, P, T, m/M to determine convective regions.
    # Simpler: use the X profile discontinuity — the convective core has uniform X,
    # the radiative envelope has a gradient. For FGONG midMS, we can identify the
    # convective boundary as where X starts to deviate from the core value.
    nabla_ad = var[:, 10]  # FGONG col 10 = nabla_ad
    L_fgong = comp["L"]
    kappa_fgong = comp["kappa"]
    R_fgong = comp["r"]

    # Compute nabla_rad = 3/(16*pi*ac) * kappa * L * P / (G * m * T^4)
    # In FGONG units (cgs), using the radiative gradient formula
    from stellar_jax.config.constants import G, Msun
    M_star = glob[0]  # grams
    m = m_frac_fgong * M_star
    # nabla_rad = 3 * kappa * L * P / (16 * pi * a * c * G * m * T^4)
    a_rad = 7.5657e-15  # radiation constant erg cm^-3 K^-4
    c_light = 2.998e10  # cm/s
    fac = 3.0 / (16.0 * np.pi * a_rad * c_light)
    # Avoid division by zero at center
    m_safe = np.maximum(m, 1e10)
    nabla_rad = fac * kappa_fgong * L_fgong * P_fgong / (G * m_safe * T_fgong**4)
    # At center, nabla_rad is ~nabla_ad for a convective core
    nabla_rad[0] = nabla_rad[1]

    # Build shell_data ordered surface→center (reversed from FGONG center→surface)
    valid = (m_frac_fgong > 1e-6) & (m_frac_fgong < 0.999)
    idx_valid = np.where(valid)[0]
    n_shell = min(300, len(idx_valid))
    idx_sub = idx_valid[np.linspace(0, len(idx_valid) - 1, n_shell, dtype=int)]

    # Compute Hp/R = P / (rho * g * R) where g = G*m/r^2
    R_star = R_fgong[-1]
    m_cgs = m_frac_fgong * M_star
    r_safe = np.maximum(R_fgong, 1e5)
    g_local = G * np.maximum(m_cgs, 1e10) / r_safe**2
    Hp = P_fgong / (rho_fgong * g_local)
    Hp_over_R = Hp / R_star

    # Compute dm_dr normalized: dm/dr * R/M = rho * 4*pi*r^2 * R / M
    dm_dr_norm = rho_fgong * 4.0 * np.pi * r_safe**2 / M_star * R_star

    shell_data = np.zeros((n_shell, 10))
    # Reverse to surface→center ordering
    shell_data[:, 0] = comp["eps_nuc"][idx_sub][::-1]  # eps
    shell_data[:, 1] = m_frac_fgong[idx_sub][::-1]  # m_frac
    shell_data[:, 2] = nabla_rad[idx_sub][::-1]  # nabla_rad
    shell_data[:, 3] = nabla_ad[idx_sub][::-1]  # nabla_ad
    shell_data[:, 4] = Hp_over_R[idx_sub][::-1]  # Hp/R from FGONG structure
    shell_data[:, 5] = dm_dr_norm[idx_sub][::-1]  # dm/dr normalized
    shell_data[:, 6] = np.log10(T_fgong[idx_sub][::-1])  # logT
    shell_data[:, 7] = np.log10(rho_fgong[idx_sub][::-1])  # logrho

    # Build non-uniform X profile on COMP_MFRACS grid
    comp_mfracs = np.array(COMP_MFRACS)
    X_prof = np.interp(comp_mfracs, m_frac_fgong[m_frac_fgong < 1.0],
                       X_fgong[m_frac_fgong < 1.0])
    X_profile_jnp = jnp.array(X_prof)
    shell_data_jnp = jnp.array(shell_data)

    # Run mix_composition WITHOUT overshoot
    X_no_ov = np.array(mix_composition(X_profile_jnp, shell_data_jnp,
                                       M_solar=M_solar, f_ov=0.0))

    # Run mix_composition WITH overshoot (f_ov=0.2)
    X_with_ov = np.array(mix_composition(X_profile_jnp, shell_data_jnp,
                                         M_solar=M_solar, f_ov=0.2))

    # --- Key assertions (preserved from test_overshoot_material_effect_high_mass) ---
    # 1. The mixed core region is LARGER with overshoot
    # Identify mixed region as zones where X is homogenized (uniform)
    def _mixed_mass_fraction(X_mixed):
        """Find the outer boundary of the homogenized (mixed) region."""
        core_val = X_mixed[0]  # Core value after mixing
        # Mixed region = zones where X is within 1e-4 of core value
        is_mixed = np.abs(X_mixed - core_val) < 1e-4
        if not np.any(is_mixed):
            return 0.0
        mixed_indices = np.where(is_mixed)[0]
        return comp_mfracs[mixed_indices[-1]]

    m_mixed_no_ov = _mixed_mass_fraction(X_no_ov)
    m_mixed_with_ov = _mixed_mass_fraction(X_with_ov)

    # Overshoot must extend the mixed region
    assert m_mixed_with_ov > m_mixed_no_ov, (
        f"M={M_solar}: overshoot should extend the mixed core. "
        f"m_mixed(f_ov=0.0)={m_mixed_no_ov:.4f}, "
        f"m_mixed(f_ov=0.2)={m_mixed_with_ov:.4f}")

    # 2. The difference should be material (>2% effect in core mass fraction)
    dm_mixed = m_mixed_with_ov - m_mixed_no_ov
    assert dm_mixed > 0.02, (
        f"M={M_solar}: overshoot has negligible effect on mixed core mass. "
        f"dm_mixed={dm_mixed:.4f}. "
        f"Overshoot should materially change the mixed region for M>1.5 M☉. "
        f"(Herwig 2000; Claret & Torres 2016)")




# ═══════════════════════════════════════════════════════════════
# CNO diagnostic snapshot: validates burn_cno + mix_composition in isolation
# ═══════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.smoke
@pytest.mark.timeout(300)
@pytest.mark.parametrize("mass_dir", ["1.5Msun", "2.0Msun"])
def test_cno_burn_equilibrium_vs_mesa_fgong(mass_dir):
    """Tier-1: burn_cno drives toward CN equilibrium at MESA FGONG T/rho.

    Re-leveled from test_cno_isotope_tracking Parts 1+2 (issue #589). The
    original test validated burn_cno + mix_composition on synthetic data.
    This validates the SAME claims using real MESA FGONG T/rho from committed
    MODE-A profiles, making it a genuine external-reference test.

    Tests:
    1. burn_cno drives isotopes toward CN equilibrium (12C/13C → 3-5) in core
       zones where T > 15 MK and dt >> τ_CN (CN timescale)
    2. Surface zones (T < 5 MK) remain unburned (12C/13C ≈ 89)
    3. Total C+N is conserved to machine precision

    External reference: MESA FGONG T/rho profiles (MODE A, identical physics).
    MESA net_approx21.f90: CN equilibration via exponential relaxation.
    Clayton (1968), §5.4: CN-cycle equilibrium ratios.
    Caughlan & Fowler (1988), Table V.a: reaction rates.
    """
    import gzip, tempfile
    import jax.numpy as jnp
    from stellar_jax.oscillations import read_fgong, fgong_components
    from stellar_jax.composition.burn import burn_cno
    from stellar_jax.config.mesh_defaults import N_COMP, COMP_MFRACS

    # Load MESA FGONG
    path = os.path.join(os.path.dirname(__file__), "..",
                        "data", "mesa_comparison", "profiles", mass_dir,
                        "midMS.FGONG.gz")
    assert os.path.exists(path), f"FGONG not found: {path}"
    with gzip.open(path) as fz:
        with tempfile.NamedTemporaryFile("wb", suffix=".FGONG", delete=False) as t:
            t.write(fz.read())
            tmp = t.name
    try:
        glob, var = read_fgong(tmp)
        comp = fgong_components(glob, var)
    finally:
        os.unlink(tmp)

    m_frac_fgong = comp["m_frac"]
    T_fgong = comp["T"]
    rho_fgong = comp["rho"]

    # Build shell_data from FGONG (burn_cno uses cols 1=m_frac, 6=logT, 7=logrho)
    valid = (m_frac_fgong > 1e-6) & (m_frac_fgong < 0.999)
    idx_valid = np.where(valid)[0]
    n_shell = min(300, len(idx_valid))
    idx_sub = idx_valid[np.linspace(0, len(idx_valid) - 1, n_shell, dtype=int)]

    shell_data = np.zeros((n_shell, 10))
    # Surface→center ordering (reversed from FGONG center→surface)
    shell_data[:, 1] = m_frac_fgong[idx_sub][::-1]
    shell_data[:, 6] = np.log10(T_fgong[idx_sub][::-1])
    shell_data[:, 7] = np.log10(rho_fgong[idx_sub][::-1])
    shell_data_j = jnp.array(shell_data)

    # Initial CN composition: primordial (NOT yet equilibrated)
    Z_val = 0.014
    C12_init = 0.170 * Z_val
    C13_init = C12_init / 89.0  # solar 12C/13C = 89
    N14_init = 0.049 * Z_val
    CN_total = C12_init + C13_init + N14_init

    C12_prof = jnp.full(N_COMP, C12_init)
    C13_prof = jnp.full(N_COMP, C13_init)
    N14_prof = jnp.full(N_COMP, N14_init)
    X_prof = jnp.full(N_COMP, 0.7)

    # Apply CN burning with large dt (10 Gyr) — should reach equilibrium in core
    # (CN timescale at T=20MK, ρ=100: τ ≈ 20 kyr; 10 Gyr ≫ τ → f≈1)
    dt = jnp.float64(1e10 * 3.156e7)  # 10 Gyr in seconds
    C12_burned, C13_burned, N14_burned = burn_cno(
        C12_prof, C13_prof, N14_prof, X_prof, shell_data_j, dt)

    C12_b = np.asarray(C12_burned)
    C13_b = np.asarray(C13_burned)
    N14_b = np.asarray(N14_burned)

    # --- Assertions (preserved from test_cno_isotope_tracking Parts 1+2) ---
    # 1. Core (index 0) should be near CN equilibrium: 12C/13C ≈ 3.0-3.5
    #    (T-dependent: at T=20 MK, CF88 rates give ~3.1; Clayton 1968 §5.4)
    ratio_core = C12_b[0] / (C13_b[0] + 1e-30)
    assert 2.0 < ratio_core < 5.0, (
        f"Core 12C/13C should be ~3-4 (CN equil at T={T_fgong[0]/1e6:.1f} MK): "
        f"got {ratio_core:.2f}")

    # 2. Core 14N should have increased (most of C → N at equilibrium)
    assert N14_b[0] > 1.5 * N14_init, (
        f"Core 14N should increase at CN equil: got {N14_b[0]:.4e}, init {N14_init:.4e}")

    # 3. Surface (index -1) should be nearly unchanged (T too low for CN cycle)
    ratio_surf = C12_b[-1] / (C13_b[-1] + 1e-30)
    assert ratio_surf > 80.0, (
        f"Surface 12C/13C should stay ~89 (no CN burning at surface T): "
        f"got {ratio_surf:.1f}")

    # 4. Total C+N conserved everywhere (to machine precision)
    CN_total_final = C12_b + C13_b + N14_b
    assert np.allclose(CN_total_final, CN_total, rtol=1e-10), (
        f"C+N not conserved: max diff = "
        f"{np.max(np.abs(CN_total_final - CN_total)):.2e}")


@pytest.mark.fast
@pytest.mark.smoke
@pytest.mark.timeout(300)
@pytest.mark.parametrize("mass_dir", ["2.0Msun"])
def test_cno_mixing_homogenizes_in_cz_vs_mesa_fgong(mass_dir):
    """Tier-1: mix_composition homogenizes CN-processed profiles in convective zones.

    Re-leveled from test_cno_isotope_tracking Part 2 (issue #589). Uses FGONG
    nabla_rad/nabla_ad to build realistic shell_data for the mixing operator.

    Tests:
    1. Convective core is homogenized after mixing (std/mean < 5%)
    2. Surface remains near-pristine outside the CZ

    External reference: MESA FGONG structure (MODE A, identical physics).
    MESA adjust_xyz.f90:do_uniform_mix_section: homogenization within CZs.
    """
    import gzip, tempfile
    import jax.numpy as jnp
    from stellar_jax.oscillations import read_fgong, fgong_components
    from stellar_jax.composition.burn import burn_cno
    from stellar_jax.composition.mix import mix_composition
    from stellar_jax.config.constants import G
    from stellar_jax.config.mesh_defaults import N_COMP, COMP_MFRACS

    # Load MESA FGONG
    path = os.path.join(os.path.dirname(__file__), "..",
                        "data", "mesa_comparison", "profiles", mass_dir,
                        "midMS.FGONG.gz")
    assert os.path.exists(path), f"FGONG not found: {path}"
    with gzip.open(path) as fz:
        with tempfile.NamedTemporaryFile("wb", suffix=".FGONG", delete=False) as t:
            t.write(fz.read())
            tmp = t.name
    try:
        glob, var = read_fgong(tmp)
        comp = fgong_components(glob, var)
    finally:
        os.unlink(tmp)

    m_frac_fgong = comp["m_frac"]
    T_fgong = comp["T"]
    rho_fgong = comp["rho"]
    X_fgong = comp["X"]
    kappa_fgong = comp["kappa"]
    L_fgong = comp["L"]
    P_fgong = comp["P"]
    nabla_ad = var[:, 10]

    # Compute nabla_rad from FGONG structure
    M_star = glob[0]
    m = m_frac_fgong * M_star
    a_rad = 7.5657e-15
    c_light = 2.998e10
    fac = 3.0 / (16.0 * np.pi * a_rad * c_light)
    m_safe = np.maximum(m, 1e10)
    nabla_rad = fac * kappa_fgong * L_fgong * P_fgong / (G * m_safe * T_fgong**4)
    nabla_rad[0] = nabla_rad[1]

    # First: burn a CN-processed profile (like test_cno_isotope_tracking Part 1)
    Z_val = 0.014
    C12_init = 0.170 * Z_val
    C13_init = C12_init / 89.0
    N14_init = 0.049 * Z_val

    # Build shell_data for burn_cno
    valid = (m_frac_fgong > 1e-6) & (m_frac_fgong < 0.999)
    idx_valid = np.where(valid)[0]
    n_shell = min(300, len(idx_valid))
    idx_sub = idx_valid[np.linspace(0, len(idx_valid) - 1, n_shell, dtype=int)]

    shell_data = np.zeros((n_shell, 10))
    shell_data[:, 1] = m_frac_fgong[idx_sub][::-1]
    shell_data[:, 2] = nabla_rad[idx_sub][::-1]
    shell_data[:, 3] = nabla_ad[idx_sub][::-1]
    shell_data[:, 4] = 0.1  # Hp/R
    shell_data[:, 5] = 1.0  # dm/dr
    shell_data[:, 6] = np.log10(T_fgong[idx_sub][::-1])
    shell_data[:, 7] = np.log10(rho_fgong[idx_sub][::-1])

    # Burn with long dt to produce CN-processed core
    C12_prof = jnp.full(N_COMP, C12_init)
    C13_prof = jnp.full(N_COMP, C13_init)
    N14_prof = jnp.full(N_COMP, N14_init)
    X_prof = jnp.full(N_COMP, 0.7)
    dt = jnp.float64(1e10 * 3.156e7)

    shell_data_burn = np.zeros((n_shell, 10))
    shell_data_burn[:, 1] = shell_data[:, 1]
    shell_data_burn[:, 6] = shell_data[:, 6]
    shell_data_burn[:, 7] = shell_data[:, 7]

    C12_burned, _, _ = burn_cno(
        C12_prof, C13_prof, N14_prof, X_prof,
        jnp.array(shell_data_burn), dt)

    # Now mix the CN-processed C12 profile
    shell_data_jnp = jnp.array(shell_data)
    M_solar = float(mass_dir.replace("Msun", ""))
    C12_mixed = np.array(mix_composition(
        C12_burned, shell_data_jnp, M_solar=M_solar, f_ov=0.0))

    comp_mfracs = np.array(COMP_MFRACS)

    # --- Assertions (preserved from test_cno_isotope_tracking Part 2) ---
    # 1. In the convective core, C12 should be homogenized
    # Use a conservative mask well inside the convective core
    # (2.0 Msun convective core extends to ~0.20 in m/M; stay inside at 0.15)
    core_mask = comp_mfracs < 0.15  # well inside convective region
    C12_core = C12_mixed[core_mask]
    assert np.std(C12_core) / (np.mean(C12_core) + 1e-30) < 0.05, (
        f"Core CZ should be homogenized: std/mean = "
        f"{np.std(C12_core)/np.mean(C12_core):.4f}")

    # 2. Surface should still be unprocessed
    surf_mask = comp_mfracs > 0.8
    C12_surface = C12_mixed[surf_mask]
    assert np.mean(C12_surface) > 0.9 * C12_init, (
        f"Surface C12 should be ~initial outside CZ: "
        f"got {np.mean(C12_surface):.4e}, init {C12_init:.4e}")


@pytest.mark.smoke
def test_cno_isotope_carry_in_scan(stellar):
    """Emergent: CNO profiles are carried step-by-step inside lax.scan.

    This is Part 3 of the original test_cno_isotope_tracking (issue #589),
    kept at @smoke because it validates the genuinely emergent behavior of
    burn_cno + mix_composition carried inside the solver's lax.scan loop —
    something that cannot be validated from a single FGONG snapshot.

    Tests:
    (a) evolve_star returns CNO profile keys
    (b) Core 12C/13C decreases from 89 (CN processing in action)
    (c) C+N conserved
    (d) Surface still near-pristine for short MS evolution

    References:
      - Clayton (1968), §5.4: CN-cycle equilibrium ratios
      - MESA struct_burn_mix.f90: operator-split architecture
    """
    import jax.numpy as jnp

    r = stellar.evolve_star(1.5, Z=0.014, max_steps=50, alpha_mlt=1.9)
    assert 'C12_profile' in r, "evolve_star must return C12_profile"
    assert 'C13_profile' in r, "evolve_star must return C13_profile"
    assert 'N14_profile' in r, "evolve_star must return N14_profile"

    C12_ev = np.asarray(r['C12_profile'])
    C13_ev = np.asarray(r['C13_profile'])
    N14_ev = np.asarray(r['N14_profile'])

    Z_val = 0.014
    C12_init = 0.170 * Z_val
    C13_init = C12_init / 89.0
    N14_init = 0.079 * Z_val  # CN+ON equilibrium (matching _core.py ZAMS init)

    # Core should show CN processing (ratio decreasing from 89)
    core_ratio = C12_ev[0] / (C13_ev[0] + 1e-30)
    assert core_ratio < 89.0, (
        f"After evolution, core 12C/13C should decrease from 89: got {core_ratio:.1f}")

    # C+N conserved (CN_total = 0.251*Z, matching MESA ZAMS X_N14;
    # MESA ref: FGONG col 23, pulse_fgong.f90)
    CN_init = C12_init + C13_init + N14_init
    CN_final = C12_ev + C13_ev + N14_ev
    assert np.allclose(CN_final, CN_init, rtol=1e-6), "C+N not conserved in evolution"

    # Surface should remain near-pristine (1.5 Msun MS, no deep envelope CZ yet)
    surf_ratio = C12_ev[-1] / (C13_ev[-1] + 1e-30)
    assert surf_ratio > 50.0, (
        f"Surface 12C/13C should stay near 89 on MS: got {surf_ratio:.1f}")



# ═══════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.smoke
def test_cno_mixing_operator_isolation(stellar):
    """Issue #98: Validates mix_composition correctly homogenizes a CN-processed
    profile within a deep convective envelope — tested in isolation.

    This does NOT validate first-dredge-up physics (which requires step-by-step
    evolution with a moving CZ boundary — blocked by #90, #96). It verifies
    that given a synthetic CN-processed interior and a deep envelope CZ,
    mix_composition:
    1. Detects a deep envelope CZ (nrad > nad from surface to m/M ~ 0.35)
    2. Homogenizes composition across the entire envelope CZ
    3. Produces surface signatures consistent with dilution of processed
       material into the envelope (12C/13C drops, N14 increases)

    References:
      - Iben (1967), ApJ 147, 624: first dredge-up theory
      - Charbonnel (1994), A&A 282, 811: observed 12C/13C ~ 20-25 after FDU
      - Kippenhahn, Weigert & Weiss (2012), §32.2: envelope deepening on RGB
    """
    import jax.numpy as jnp

    Z_val = 0.014
    C12_init = 0.170 * Z_val  # 2.38e-3
    C13_init = C12_init / 89.0  # 2.67e-5 (solar 12C/13C = 89)
    N14_init = 0.049 * Z_val  # 6.86e-4
    CN_total = C12_init + C13_init + N14_init

    # CN equilibrium values at ~15-20 MK (Clayton 1968, §5.4; CF88 T-dependent
    # rates give C12_eq/CN ~ 0.03-0.04 in this regime). Used here as approximate
    # synthetic "processed" composition for testing the mixing mechanism.
    C12_eq = 0.035 * CN_total
    C13_eq = 0.010 * CN_total
    N14_eq = 0.955 * CN_total

    comp_mfracs = np.asarray(stellar.COMP_MFRACS)
    N_COMP = len(comp_mfracs)

    # --- Build a CN-processed interior profile ---
    # Mimicking a 2 M_sun star at first dredge-up (Iben 1967, ApJ 147, 624):
    # - MS convective core + H-shell burning processed to m/M ~ 0.50
    # - Transition zone 0.50 < m/M < 0.55
    # - Pristine envelope above m/M > 0.55
    # First dredge-up mixes the deep envelope (CZ base at m/M ~ 0.20) into
    # this processed region, diluting the surface composition.
    # Expected post-FDU 12C/13C ~ 20-30 (Charbonnel 1994, A&A 282, 811;
    # Charbonnel & Lagarde 2010, A&A 522, A10).
    C12_prof = np.full(N_COMP, C12_init)
    C13_prof = np.full(N_COMP, C13_init)
    N14_prof = np.full(N_COMP, N14_init)

    for i in range(N_COMP):
        mf = comp_mfracs[i]
        if mf < 0.50:
            w = 1.0  # fully processed
        elif mf < 0.55:
            w = (0.55 - mf) / 0.05  # linear transition
        else:
            w = 0.0  # pristine
        C12_prof[i] = C12_init * (1 - w) + C12_eq * w
        C13_prof[i] = C13_init * (1 - w) + C13_eq * w
        N14_prof[i] = N14_init * (1 - w) + N14_eq * w

    # Verify initial state: surface is pristine, core is processed
    assert abs(C12_prof[-1] / C13_prof[-1] - 89.0) < 1.0
    assert C12_prof[0] / C13_prof[0] < 5.0  # core at CN equilibrium

    # --- Build synthetic shell_data for deep envelope CZ ---
    # RGB structure: deep convective envelope from surface to m/M ~ 0.35
    # (Schwarzschild criterion: nrad > nad throughout the envelope)
    n_shells = 300
    shell_data = np.zeros((n_shells, 10))
    # Column 1: m/M decreases from surface (1.0) to center (0.0)
    shell_data[:, 1] = np.linspace(1.0, 0.0, n_shells)
    # Column 6: logT from surface (3.7) to center (7.5)
    shell_data[:, 6] = np.linspace(3.7, 7.5, n_shells)
    # Column 7: logrho from surface (-8) to center (3)
    shell_data[:, 7] = np.linspace(-8.0, 3.0, n_shells)
    # Column 4: Hp/R ~ 0.1
    shell_data[:, 4] = 0.1
    # Column 5: dm/dr ~ 1
    shell_data[:, 5] = 1.0

    # Deep envelope convection: nrad > nad for m/M > 0.15 (surface→0.15)
    # First dredge-up: envelope CZ deepens to m/M ~ 0.15 for 2 M_sun on
    # the upper RGB (Iben 1967; Kippenhahn & Weigert §32.2).
    # Below 0.15: radiative (He core + H-burning shell)
    for i in range(n_shells):
        mf = shell_data[i, 1]
        if mf > 0.15:
            # Convective envelope: nrad = 1.0 >> nad = 0.4
            shell_data[i, 2] = 1.0  # nabla_rad (high — superadiabatic)
            shell_data[i, 3] = 0.4  # nabla_ad
        else:
            # Radiative interior: nrad = 0.2 < nad = 0.4
            shell_data[i, 2] = 0.2  # nabla_rad
            shell_data[i, 3] = 0.4  # nabla_ad

    shell_data_j = jnp.array(shell_data)

    # --- Apply mix_composition ---
    C12_mixed = np.asarray(stellar.mix_composition(
        jnp.array(C12_prof), shell_data_j, M_solar=1.5))
    C13_mixed = np.asarray(stellar.mix_composition(
        jnp.array(C13_prof), shell_data_j, M_solar=1.5))
    N14_mixed = np.asarray(stellar.mix_composition(
        jnp.array(N14_prof), shell_data_j, M_solar=1.5))

    # --- Verify first dredge-up signature at the surface ---
    # Surface = last shells (m/M → 1.0, index 199 in COMP_MFRACS)
    surf_idx = comp_mfracs > 0.8
    surf_C12 = np.mean(C12_mixed[surf_idx])
    surf_C13 = np.mean(C13_mixed[surf_idx])
    surf_N14 = np.mean(N14_mixed[surf_idx])
    surf_ratio = surf_C12 / (surf_C13 + 1e-30)

    # After first dredge-up, surface 12C/13C should drop from 89 to ~20-30
    # (exact value depends on depth of CZ; Charbonnel 1994)
    assert surf_ratio < 50.0, (
        f"After FDU, surface 12C/13C should drop below 50: got {surf_ratio:.1f}")
    assert surf_ratio > 5.0, (
        f"Surface 12C/13C should not drop to full equilibrium (~3.5): got {surf_ratio:.1f}")

    # Surface N14 should be enriched (CN cycle converts C → N)
    assert surf_N14 > 1.3 * N14_init, (
        f"After FDU, surface N14 should increase by >30%: "
        f"got {surf_N14:.4e}, init {N14_init:.4e} (ratio {surf_N14/N14_init:.2f})")

    # Surface C12 should be depleted
    assert surf_C12 < 0.8 * C12_init, (
        f"After FDU, surface C12 should decrease: "
        f"got {surf_C12:.4e}, init {C12_init:.4e}")

    # C+N total must be conserved (mass-weighted average unchanged)
    CN_mixed = C12_mixed + C13_mixed + N14_mixed
    assert np.allclose(CN_mixed, CN_total, rtol=1e-6), (
        f"C+N not conserved after mixing: max diff = "
        f"{np.max(np.abs(CN_mixed - CN_total)):.2e}")

    # Envelope should be homogenized (m/M > 0.25 all same composition)
    env_mask = comp_mfracs > 0.30
    assert np.std(C12_mixed[env_mask]) / (np.mean(C12_mixed[env_mask]) + 1e-30) < 0.02, (
        f"Envelope CZ should be homogenized after mixing: "
        f"std/mean = {np.std(C12_mixed[env_mask]) / np.mean(C12_mixed[env_mask]):.4f}")



@pytest.mark.fast
@pytest.mark.smoke
@pytest.mark.timeout(300)
@pytest.mark.validation
@pytest.mark.mutation("burn_composition_identity")
@pytest.mark.right_reason("burn")
@pytest.mark.parametrize("mass_dir,profile_a,profile_b", [
    ("2.0Msun", "Xc0.40", "Xc0.30"),
    ("1.0Msun", "Xc0.40", "Xc0.30"),
])
def test_burn_composition_rate_vs_mesa_fgong(mass_dir, profile_a, profile_b):
    """Tier-1: burn_composition rate vs MESA FGONG hydrogen depletion.

    Loads two consecutive FGONG snapshots (earlier Xc and later Xc), estimates
    the time interval from the hydrogen depletion rate, runs burn_composition
    for that dt, and verifies the resulting core X matches the later snapshot.

    External reference: MESA FGONG hydrogen profiles (MODE A, identical physics).
    """
    import gzip, tempfile
    import jax.numpy as jnp
    from stellar_jax.oscillations import read_fgong, fgong_components
    from stellar_jax.evolution import burn_composition
    from stellar_jax.config.constants import Q_PER_G, Msun, Lsun
    from stellar_jax.config.mesh_defaults import N_COMP, COMP_MFRACS

    def _load(profile_name):
        path = os.path.join(os.path.dirname(__file__), "..",
                            "data", "mesa_comparison", "profiles", mass_dir,
                            f"{profile_name}.FGONG.gz")
        assert os.path.exists(path), f"FGONG not found: {path}"
        with gzip.open(path) as fz:
            with tempfile.NamedTemporaryFile("wb", suffix=".FGONG", delete=False) as t:
                t.write(fz.read())
                tmp = t.name
        try:
            glob, var = read_fgong(tmp)
            comp = fgong_components(glob, var)
        finally:
            os.unlink(tmp)
        return glob, comp

    glob_a, comp_a = _load(profile_a)
    glob_b, comp_b = _load(profile_b)

    M_star = glob_a[0]  # grams

    # Build X profiles on COMP_MFRACS grid
    comp_mfracs = np.array(COMP_MFRACS)
    m_a = comp_a["m_frac"]
    X_a = comp_a["X"]
    m_b = comp_b["m_frac"]
    X_b = comp_b["X"]

    valid_a = m_a < (1.0 - 1e-10)
    valid_b = m_b < (1.0 - 1e-10)
    X_prof_a = np.interp(comp_mfracs, m_a[valid_a], X_a[valid_a])
    X_prof_b = np.interp(comp_mfracs, m_b[valid_b], X_b[valid_b])

    # Core X: average over innermost zones (m/M < 0.1)
    core_mask = comp_mfracs < 0.1
    X_core_a = np.mean(X_prof_a[core_mask])
    X_core_b = np.mean(X_prof_b[core_mask])
    dX_core = X_core_a - X_core_b  # positive (hydrogen depleted)
    assert dX_core > 0.01, f"Core depletion too small: {dX_core}"

    # Estimate dt from MESA luminosity and eps_nuc:
    # dt = dX_core * Q_PER_G / eps_avg_core
    # where eps_avg_core is the average nuclear energy generation in the core
    eps_a = comp_a["eps_nuc"]
    core_mf_limit = 0.1  # same as core_mask above
    core_zone_mask_fgong = m_a < core_mf_limit
    eps_core_avg = np.mean(eps_a[core_zone_mask_fgong]) if np.sum(core_zone_mask_fgong) > 0 else 1.0
    dt = float(dX_core * Q_PER_G / eps_core_avg)

    print(f"\n  {mass_dir} burn rate ({profile_a} → {profile_b}):")
    print(f"    X_core_a = {X_core_a:.4f}, X_core_b = {X_core_b:.4f}, dX = {dX_core:.4f}")
    print(f"    eps_core_avg = {eps_core_avg:.3e} erg/g/s")
    print(f"    dt estimate = {dt:.3e} s = {dt/3.156e7:.3e} yr")

    # Build shell_data from profile_a for burn_composition
    # burn_composition only uses columns 0 (eps) and 1 (m_frac)
    valid_fgong = (m_a > 1e-6) & (m_a < 0.99)
    idx_fgong = np.where(valid_fgong)[0]
    n_shell = min(300, len(idx_fgong))
    idx_sub = idx_fgong[np.linspace(0, len(idx_fgong) - 1, n_shell, dtype=int)]

    # Shell_data ordered surface→center (reversed from FGONG center→surface)
    shell_data = np.zeros((n_shell, 10))
    shell_data[:, 0] = eps_a[idx_sub][::-1]  # eps
    shell_data[:, 1] = m_a[idx_sub][::-1]    # m_frac

    X_profile_jnp = jnp.array(X_prof_a)
    # Y from FGONG: no diffusion in MODE A → Z=const=0.014, so Y = 1-X-Z
    Y_prof_a = 1.0 - X_prof_a - 0.014
    Y_profile_jnp = jnp.array(Y_prof_a)
    shell_data_jnp = jnp.array(shell_data)
    X_burned, _Y_burned = burn_composition(X_profile_jnp, Y_profile_jnp, shell_data_jnp, dt)
    X_burned_np = np.array(X_burned)

    # Compare core X after burning to MESA's later snapshot
    X_core_burned = np.mean(X_burned_np[core_mask])
    err = abs(X_core_burned - X_core_b) / dX_core

    print(f"    X_core after burn = {X_core_burned:.4f}")
    print(f"    MESA X_core target = {X_core_b:.4f}")
    print(f"    |error| / |dX| = {err:.4f} ({err*100:.1f}%)")

    # Assert: within 15% of the expected depletion
    assert err < 0.15, (
        f"{mass_dir}: burn_composition core X after dt gives {X_core_burned:.4f}, "
        f"expected {X_core_b:.4f} (MESA {profile_b}). "
        f"Relative error {err*100:.1f}% exceeds 15% tolerance. "
        f"Burn rate implementation disagrees with MESA hydrogen depletion.")





@pytest.mark.fast
@pytest.mark.smoke
@pytest.mark.timeout(300)
@pytest.mark.validation
@pytest.mark.mutation("burn_composition_identity")
@pytest.mark.right_reason("burn")
@pytest.mark.parametrize("mass_dir,profile", [
    ("1.0Msun", "Xc0.60"),
    ("2.0Msun", "Xc0.60"),
])
def test_burn_baryon_conservation_vs_mesa_fgong(mass_dir, profile):
    """Tier-1: burn_composition conserves baryons per zone (ΔY = −ΔX).

    WHAT: After burning, X+Y+Z = 1 per zone (Z unchanged by pp/CNO H→He
    burn), AND Y increases in the core (He produced from H).

    WHY: Issue #1001 — the burn step previously decremented X without
    depositing into Y, so X+Y+Z drifted below 1 in burning zones. This
    broke mean molecular weight μ (→ EOS, opacity, structure) and He-core
    growth. MESA conserves baryons: net_eval.f90:274 computes
    dxdt(i) = Z_plus_N(ci) * dydt(i) where sum(dxdt)=0 by molar baryon
    conservation. MESA also renormalizes per zone: hydro_vars.f90:751-754.

    EXTERNAL REFERENCE: MESA FGONG eps_nuc profile (MODE A, identical
    physics). The burning rate dX = eps*dt/Q_PER_G and the baryon
    conservation ΔY = −ΔX are both grounded in MESA's nuclear network.

    TOLERANCE: |X+Y+Z − 1| < 1e-10 per zone — this is exact to machine
    precision because Z is unchanged and ΔY = −ΔX by construction. The
    1e-10 margin covers floating-point accumulation only.

    MUTATION: burn_composition_identity — if burning is disabled, Y does
    not increase in the core → the core ΔY > 1e-4 assertion fails.
    """
    import gzip, tempfile
    import jax.numpy as jnp
    from stellar_jax.oscillations import read_fgong, fgong_components
    from stellar_jax.evolution import burn_composition
    from stellar_jax.config.constants import Q_PER_G
    from stellar_jax.config.mesh_defaults import N_COMP, COMP_MFRACS

    # Load MESA FGONG
    path = os.path.join(os.path.dirname(__file__), "..",
                        "data", "mesa_comparison", "profiles", mass_dir,
                        f"{profile}.FGONG.gz")
    assert os.path.exists(path), f"FGONG not found: {path}"
    with gzip.open(path) as fz:
        with tempfile.NamedTemporaryFile("wb", suffix=".FGONG", delete=False) as t:
            t.write(fz.read())
            tmp = t.name
    try:
        glob, var = read_fgong(tmp)
        comp = fgong_components(glob, var)
    finally:
        os.unlink(tmp)

    # Extract MESA eps_nuc and X on the FGONG grid
    m_frac_fgong = comp["m_frac"]
    eps_fgong = comp["eps_nuc"]
    X_fgong = comp["X"]

    # Interpolate onto COMP_MFRACS grid
    comp_mfracs = np.array(COMP_MFRACS)
    valid = m_frac_fgong < (1.0 - 1e-10)
    X_prof = np.interp(comp_mfracs, m_frac_fgong[valid], X_fgong[valid])

    # Y from FGONG: MODE A has no diffusion → Z=const=0.014, so Y = 1-X-Z
    Z_val = 0.014
    Y_prof = 1.0 - X_prof - Z_val
    Z_prof = np.full_like(X_prof, Z_val)

    # Build shell_data from FGONG
    eps_prof_fgong = np.interp(comp_mfracs, m_frac_fgong[valid], eps_fgong[valid])
    n_shell = 300
    valid_fgong = (m_frac_fgong > 1e-6) & (m_frac_fgong < 0.99)
    idx_fgong = np.where(valid_fgong)[0]
    idx_sub = idx_fgong[np.linspace(0, len(idx_fgong) - 1, n_shell, dtype=int)]
    shell_data = np.zeros((n_shell, 10))
    shell_data[:, 0] = eps_fgong[idx_sub][::-1]  # eps (surface→center)
    shell_data[:, 1] = m_frac_fgong[idx_sub][::-1]  # m_frac

    # Short dt (1 Myr) — enough for meaningful burn, not enough to exhaust core H
    dt = 1e6 * 3.156e7  # 1 Myr in seconds

    X_profile_jnp = jnp.array(X_prof)
    Y_profile_jnp = jnp.array(Y_prof)
    shell_data_jnp = jnp.array(shell_data)
    X_burned, Y_burned = burn_composition(X_profile_jnp, Y_profile_jnp,
                                           shell_data_jnp, dt)
    X_b = np.array(X_burned)
    Y_b = np.array(Y_burned)

    # --- Assertion 1: per-zone X+Y+Z = 1 (baryon conservation) ---
    xa_sum = X_b + Y_b + Z_prof
    max_drift = np.max(np.abs(xa_sum - 1.0))
    print(f"\n  {mass_dir}/{profile} baryon conservation:")
    print(f"    max |X+Y+Z - 1| = {max_drift:.2e}")
    assert max_drift < 1e-10, (
        f"{mass_dir}: X+Y+Z drifts from 1 after burning: max |sum-1| = {max_drift:.2e}. "
        f"Baryon conservation violated (MESA net_eval.f90:274: sum(dxdt)=0).")

    # --- Assertion 2: Y increases in the core (He produced from H) ---
    # This is the mutation-kill assertion: burn_composition_identity
    # would leave Y unchanged → this fails.
    core_mask = comp_mfracs < 0.1
    dY_core = np.mean(Y_b[core_mask]) - np.mean(Y_prof[core_mask])
    print(f"    core ΔY = {dY_core:.6f}")
    assert dY_core > 1e-5, (
        f"{mass_dir}: Core Y did not increase after burning: ΔY = {dY_core:.2e}. "
        f"Burn must deposit H into He (4H → He4, baryon conservation).")

    # --- Assertion 3: ΔY = −ΔX per zone (exact to float64) ---
    dX = X_b - np.array(X_prof)
    dY = Y_b - np.array(Y_prof)
    max_mismatch = np.max(np.abs(dY + dX))
    print(f"    max |ΔY + ΔX| = {max_mismatch:.2e}")
    assert max_mismatch < 1e-14, (
        f"{mass_dir}: ΔY ≠ −ΔX after burning: max |ΔY+ΔX| = {max_mismatch:.2e}. "
        f"Baryon conservation requires ΔY = −ΔX exactly.")
# =============================================================================
# END-TO-END FIRST DREDGE-UP (FDU) VALIDATION vs MESA —
# =============================================================================
#
# Physics: The first dredge-up occurs as a low-mass star ascends the RGB and
# its convective envelope deepens into regions processed by CN-cycling during
# the MS. This brings CN-equilibrium material to the surface:
#   - surface C12 DECREASES (depleted by CN cycle → C13 → N14)
#   - surface N14 INCREASES (end product of CN cycle)
#   - surface He4 INCREASES slightly (envelope penetrates He-enriched zone)
# The magnitude depends on mass (deeper dredge-up at higher M; convective core
# stars have more processed material). Standard references:
#   Iben (1967), ApJ 147, 624 (theory);
#   Charbonnel (1994), A&A 282, 811 (observations);
#   Karakas & Lattanzio (2014), PASA 31, 30 (review of yields).
#
# MESA implementation: mix_info.f90:488-530 classifies convective zones via
# mixing_type == convective_mixing. Composition is homogenized within each CZ
# by the time-dependent diffusive mixing operator (D_mix × sigma → mix.f90:
# get_convection_sigmas). The MESA reference tracks use Schwarzschild
# (use_Ledoux_criterion=.false., the default in controls.defaults:2182).
#
# CONSTRAINT (labeled): Our adaptive_forward path uses the Ledoux criterion
# (gradL_composition_term ≠ 0), while the MESA reference uses
# Schwarzschild. The ∇μ stabilization makes our convective envelope slightly
# SHALLOWER → our C12 depletion may be SMALLER than MESA's. The tolerance band
# [40%, 180%] accounts for this known, bounded deviation.
# =============================================================================


def _fdu_surface_composition(traj, species='C12'):
    """Extract surface composition history from a Trajectory.

    Surface = outermost 10% by mass (q > 0.90 on the composition grid).
    MESA's surface_c12 is the outermost cell; averaging the outer 10% is
    conservative (the convective envelope homogenizes the outer ~30-70%
    during FDU anyway).

    IMPORTANT: the composition profiles in evolve_star_adaptive are stored
    on the ADAPTIVE mesh (q_cells = cell centers of s.q_mesh), NOT on a
    uniform grid. With max_concentration=500, most zones cluster at the
    H-burning shell (q~0.15-0.25). Using np.linspace(0,1,N) would map
    naive index 540/600=0.90 to a mass coordinate deep in the interior,
    reading CN-processed material as "surface". We use s.q_mesh to compute
    the actual mass coordinate of each zone.
    """
    accepted_steps = [s for s in traj.steps if s.accepted]
    surface_vals = []
    logL_vals = []
    for s in accepted_steps:
        if species == 'C12':
            profile = s.C12_profile
        elif species == 'C13':
            profile = s.C13_profile
        elif species == 'N14':
            profile = s.N14_profile
        elif species == 'Y':
            profile = s.Y_profile
        else:
            raise ValueError(f"Unknown species: {species}")
        # Use the actual adaptive mesh coordinates (cell centers from face positions)
        q_cells = 0.5 * (s.q_mesh[:-1] + s.q_mesh[1:])
        surf_mask = q_cells > 0.90
        if np.sum(surf_mask) > 0:
            val = float(np.mean(profile[surf_mask]))
        else:
            # Fallback: use the outermost zone (highest q)
            val = float(profile[np.argmax(q_cells)])
        surface_vals.append(val)
        logL_vals.append(s.logL)
    return np.array(surface_vals), np.array(logL_vals)


# MESA FDU C12 depletion fractions (measured from the committed reference tracks):
#   1.0 Msun: 2.384e-3 → 2.141e-3 = 10.2% (shallow FDU, no MS convective core)
#   1.5 Msun: 2.384e-3 → 1.809e-3 = 24.1% (deeper FDU, MS convective core)
#   2.0 Msun: 2.384e-3 → 1.842e-3 = 22.7% (deep FDU, large MS convective core)
_FDU_PARAMS = {
    1.5: {'mesa_depletion': 0.241, 'min_logL_rgb': 2.0, 'mesa_fdu_logL': 2.51},
    2.0: {'mesa_depletion': 0.227, 'min_logL_rgb': 2.0, 'mesa_fdu_logL': 2.78},
}


@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("disable_envelope_mixing")
@pytest.mark.right_reason("surface")
@pytest.mark.timeout(18000)  # 5h: full MS→RGB evolution (bimodal wall-time + cold-compile headroom)
@pytest.mark.parametrize("mass", [
    pytest.param(1.5, id="1p5"),
    pytest.param(2.0, id="2p0",
                 marks=pytest.mark.suspended),  #: nightly-demote — 1p5 stays per-wave
])
def test_first_dredge_up_surface_c12_vs_mesa(mass):
    """End-to-end FDU: surface C12 depletion agrees with MESA (MODE A).

    WHAT: evolve_star_adaptive evolves a star through the RGB; the deepening
    convective envelope dredges CN-processed material to the surface, depleting
    surface C12 by 10-24% (mass-dependent). This test compares the fractional
    C12 depletion against the MESA RGB track value.

    WHY (issue #162): The first dredge-up is the definitive end-to-end
    validation that: (a) the RGB convective envelope deepens to the correct
    depth, (b) the composition mixing machinery (_envelope_zone_mixing)
    correctly homogenizes the CZ, and (c) the CNO burning that established
    the processed interior during the MS produced a C12 gradient for the FDU
    to mix. The FDU is mass-dependent and observable — it sets the 12C/13C
    ratio in first-ascent giants (Charbonnel 1994, A&A 282, 811). Having
    BOTH 1.5 and 2.0 M☉ pass confirms the effect is mass-dependent (not a
    hardcoded depletion).

    EXTERNAL REFERENCE: MESA f12c70cf (r26.4.1) identical-physics RGB track
    (data/mesa_comparison/rgb/<M>Msun/history.data.gz), column surface_c12.
    Tolerance: our fractional depletion must be within [40%, 180%] of MESA's.
    This is generous because the convective boundary differs (our code uses
    Ledoux [#409] while MESA reference uses Schwarzschild — the ∇μ barrier
    makes our envelope shallower) and mesh resolution affects the boundary
    position. The key validation is that FDU OCCURS at the right order of
    magnitude, not bit-level agreement.

    WHAT MAKES IT FAIL (@mutation "disable_envelope_mixing"): patches
    _envelope_zone_mixing to return input unchanged → no envelope mixing →
    surface C12 stays at its initial value → depletion ≈ 0 → assertion fails.

    MESA reference for the mixing mechanism:
      - mix_info.f90:488-530: CZ classification (mixing_type == convective_mixing)
      - mix_info.f90:1344-1381: do_mix_envelope (extend CZ to surface)
      - Iben (1967), ApJ 147, 624: first dredge-up theory
    """
    import mesa_rgb

    params = _FDU_PARAMS[mass]

    # --- Load MESA reference ---
    mesa_hist = mesa_rgb.load_rgb_history(mass)
    mesa_surface_c12 = mesa_hist['surface_c12']
    mesa_log_L = mesa_hist['log_L']

    mesa_c12_initial = float(mesa_surface_c12[0])
    mesa_c12_min = float(np.min(mesa_surface_c12))
    mesa_depletion_frac = 1.0 - mesa_c12_min / mesa_c12_initial
    mesa_fdu_logL = float(mesa_log_L[np.argmin(mesa_surface_c12)])

    print(f"\n  MESA reference ({mass} Msun):")
    print(f"    Initial surface C12  = {mesa_c12_initial:.6e}")
    print(f"    Min surface C12      = {mesa_c12_min:.6e}")
    print(f"    Depletion fraction   = {mesa_depletion_frac*100:.1f}%")
    print(f"    FDU complete at logL = {mesa_fdu_logL:.3f}")

    # --- Run evolve_star_adaptive (MODE A: identical physics from MESA_CONFIG) ---
    from stellar_jax.adaptive_forward import evolve_star_adaptive
    from stellar_jax.config.mesa_config import MESA_CONFIG

    traj, result = evolve_star_adaptive(
        mass=mass, N_zones=600, max_steps=5000, verbose=True,
        **MESA_CONFIG)

    # --- Extract surface compositions ---
    accepted_steps = [s for s in traj.steps if s.accepted]
    assert len(accepted_steps) > 10, (
        f"Too few accepted steps ({len(accepted_steps)}) — evolution failed")

    surface_c12, logL_arr = _fdu_surface_composition(traj, 'C12')
    surface_n14, _ = _fdu_surface_composition(traj, 'N14')

    # Initial surface C12 (first 5 steps, before any mixing)
    our_c12_initial = float(np.mean(surface_c12[:5]))
    our_c12_min = float(np.min(surface_c12))
    our_depletion_frac = 1.0 - our_c12_min / our_c12_initial
    min_idx = int(np.argmin(surface_c12))
    our_fdu_logL = float(logL_arr[min_idx])
    max_logL = float(np.max(logL_arr))

    # N14 enrichment (CN equilibrium: C12 decreases → N14 increases)
    our_n14_initial = float(np.mean(surface_n14[:5]))
    our_n14_at_fdu = float(surface_n14[min_idx])
    n14_enrichment = (our_n14_at_fdu - our_n14_initial) / our_n14_initial

    print(f"\n  Our result ({mass} Msun):")
    print(f"    Initial surface C12  = {our_c12_initial:.6e}")
    print(f"    Min surface C12      = {our_c12_min:.6e}")
    print(f"    Depletion fraction   = {our_depletion_frac*100:.1f}%")
    print(f"    FDU at logL          = {our_fdu_logL:.3f}")
    print(f"    Max logL reached     = {max_logL:.3f}")
    print(f"    N14 enrichment       = {n14_enrichment*100:.1f}%")

    # --- Acceptance 0: Initial C12 sanity check ---
    # Both codes start from the same physics (Z=0.014, solar-scaled CNO).
    # Our initial C12 = 0.170*Z = 2.38e-3. MESA's = 2.384e-3.
    # If these disagree by >5%, the depletion comparison is meaningless.
    initial_c12_ratio = our_c12_initial / mesa_c12_initial
    assert 0.95 <= initial_c12_ratio <= 1.05, (
        f"{mass} Msun: initial surface C12 mismatch — ours {our_c12_initial:.6e} vs "
        f"MESA {mesa_c12_initial:.6e} (ratio {initial_c12_ratio:.4f}). "
        f"Cannot compare depletions from different starting compositions.")

    # --- Acceptance 1: Must reach the RGB ---
    assert max_logL > params['min_logL_rgb'], (
        f"{mass} Msun did not reach the RGB: max logL = {max_logL:.3f} < "
        f"{params['min_logL_rgb']}. Cannot validate FDU without RGB ascent.")

    # --- Acceptance 2: Non-trivial C12 depletion (FDU occurred) ---
    assert our_depletion_frac > 0.05, (
        f"{mass} Msun surface C12 depletion is trivial "
        f"({our_depletion_frac*100:.1f}% < 5%). FDU did not occur — envelope "
        f"mixing may not be reaching the CN-processed interior. "
        f"MESA expects {mesa_depletion_frac*100:.1f}% depletion.")

    # --- Acceptance 3: Depletion within [40%, 180%] of MESA ---
    ratio = our_depletion_frac / mesa_depletion_frac
    print(f"    C12 depletion ratio (ours/MESA) = {ratio:.3f}")
    assert 0.40 <= ratio <= 1.80, (
        f"{mass} Msun surface C12 depletion {our_depletion_frac*100:.1f}% is "
        f"outside [40%, 180%] of MESA's {mesa_depletion_frac*100:.1f}% "
        f"(ratio={ratio:.3f}). Expected "
        f"{mesa_depletion_frac*0.40*100:.1f}%–{mesa_depletion_frac*1.80*100:.1f}%. "
        f"The convective envelope depth or interior CN processing may be wrong.")

    # --- Acceptance 4: N14 enrichment has the correct SIGN ---
    # CN cycle: C12 → C13 → N14, so when surface C12 decreases, N14 MUST
    # increase. This is a necessary consistency check — if N14 doesn't rise,
    # the "depletion" may be numerical noise rather than real dredge-up.
    assert n14_enrichment > 0.01, (
        f"{mass} Msun: surface N14 did not increase during FDU "
        f"(enrichment = {n14_enrichment*100:.1f}%, need > 1%). "
        f"The CN-cycle signature is absent — mixing may not be reaching the "
        f"CN-processed interior, or burn_cno is not producing the N14 gradient.")
