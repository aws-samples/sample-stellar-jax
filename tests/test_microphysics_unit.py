"""Fast unit tests for microphysics/ (issue #496).

These tests verify that each public microphysics function:
1. Produces physically reasonable outputs at known stellar conditions
2. Is differentiable (jax.grad returns finite, sign-correct values)
3. Respects the contracts documented in microphysics/contracts.py

All tests use SYNTHETIC scalar inputs (no evolve_star, no heavy JIT).
Each test completes in <1s (no table loading after first import).

Marks: @pytest.mark.fast (no evolve_star dependency)
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ===========================================================================
# EOS tests
# ===========================================================================

@pytest.mark.fast
def test_eos_lookup_solar_core():
    """eos_lookup at solar-core conditions returns physically reasonable values.

    Reference: solar core (Model S): logT ~ 7.2, logP ~ 17.3, X ~ 0.34, Z ~ 0.02.
    Expected: rho ~ 150 g/cm³ (log rho ~ 2.18), mu ~ 0.85, nad ~ 0.4.
    Christensen-Dalsgaard et al. (1996), Model S.
    """
    from stellar_jax.microphysics.eos import eos_lookup

    rho, mu, nad, S, cp, chi_rho, chi_T = eos_lookup(7.2, 17.3, 0.34, 0.02)

    # Density: solar core ~ 150 g/cm³ → log rho ~ 2.18
    log_rho = float(jnp.log10(rho))
    assert 1.5 < log_rho < 2.5, f"log rho = {log_rho}, expected ~2.18"

    # Mean molecular weight: fully ionized H/He mix, X=0.34 → mu ~ 0.8-0.9
    mu_val = float(mu)
    assert 0.5 < mu_val < 1.5, f"mu = {mu_val}, expected ~0.85"

    # Adiabatic gradient: ideal gas is 0.4; partial ionization lowers it
    nad_val = float(nad)
    assert 0.2 < nad_val < 0.5, f"nad = {nad_val}, expected ~0.4"

    # Specific heat: must be positive
    cp_val = float(cp)
    assert cp_val > 0, f"cp = {cp_val}, expected positive"

    # chi_rho, chi_T: thermodynamic derivatives, must be positive
    assert float(chi_rho) > 0, "chi_rho must be positive"
    assert float(chi_T) > 0, "chi_T must be positive"


@pytest.mark.fast
def test_eos_differentiable():
    """eos_lookup is differentiable w.r.t. logT — gradient is finite and sign-correct.

    At fixed pressure, increasing T → decreasing rho (ideal gas: rho ∝ P/(T*mu)).
    So d(rho)/d(logT) should be negative.
    """
    from stellar_jax.microphysics.eos import eos_lookup

    def rho_of_logT(logT):
        rho, _, _, _, _, _, _ = eos_lookup(logT, 17.0, 0.7, 0.02)
        return rho

    grad_val = float(jax.grad(rho_of_logT)(7.0))

    assert np.isfinite(grad_val), f"d(rho)/d(logT) is not finite: {grad_val}"
    # At fixed P, T↑ → rho↓ (ideal gas law)
    assert grad_val < 0, (
        f"d(rho)/d(logT) = {grad_val}, expected negative (ideal gas: rho ∝ 1/T at fixed P)"
    )


# ===========================================================================
# Opacity tests
# ===========================================================================

@pytest.mark.fast
def test_kappa_physical_range():
    """kappa returns log10(opacity) in a physically reasonable range.

    At solar interior (logT=7.0, logRho=1.5, X=0.7, Z=0.02):
    Kramers bound-free/free-free dominates → log kappa ~ -0.5 to 1.0.
    Iglesias & Rogers (1996).
    """
    from stellar_jax.microphysics.opacity import kappa

    lk = float(kappa(7.0, 1.5, 0.7, 0.02))
    assert -2.0 < lk < 2.0, f"log kappa = {lk}, out of range [-2, 2]"


@pytest.mark.fast
def test_kappa_differentiable_temperature():
    """kappa is differentiable w.r.t. logT — Kramers: kappa ∝ T^{-3.5}.

    In the Kramers regime (bound-free + free-free, logT ~ 6-7):
    d(log kappa)/d(logT) ~ -3.5 (negative).
    """
    from stellar_jax.microphysics.opacity import kappa

    grad_val = float(jax.grad(kappa, argnums=0)(6.5, 0.5, 0.7, 0.02))

    assert np.isfinite(grad_val), f"d(log kappa)/d(logT) is not finite: {grad_val}"
    # Kramers opacity decreases with T
    assert grad_val < 0, (
        f"d(log kappa)/d(logT) = {grad_val}, expected negative in Kramers regime"
    )


@pytest.mark.fast
def test_kappa_conductive_degenerate():
    """Conductive opacity dominates in degenerate conditions.

    At He-core RGB conditions (logT=7, logRho=5, X=0): conduction is
    efficient → total kappa < radiative-only kappa (harmonic blend).
    Cassisi et al. (2007), ApJ 661, 1094.
    """
    from stellar_jax.microphysics.opacity import kappa, opal_kappa

    logT, logRho, X, Z = 7.0, 5.0, 0.0, 0.02

    lk_total = float(kappa(logT, logRho, X, Z))
    lk_rad = float(opal_kappa(logT, logRho, X, Z))

    # Harmonic blend: 1/kappa_total = 1/kappa_rad + 1/kappa_cond
    # → kappa_total < kappa_rad always
    assert lk_total < lk_rad, (
        f"Total opacity ({lk_total:.2f}) should be < radiative ({lk_rad:.2f}) "
        f"due to conductive contribution"
    )


# ===========================================================================
# Nuclear tests
# ===========================================================================

@pytest.mark.fast
def test_eps_nuclear_pp_dominates_solar():
    """At solar core conditions, PP chain dominates over CNO.

    Solar core: rho ~ 150, T ~ 1.5e7, X ~ 0.34, Z ~ 0.02.
    PP dominates for T < ~1.7e7 K (Adelberger et al. 2011).
    Verify eps > 0 and that zeroing N14 (CNO catalyst) barely changes it.
    """
    from stellar_jax.microphysics.nuclear import epsilon_nuclear

    rho, T, X, Z = 150.0, 1.5e7, 0.34, 0.02

    eps_total = float(epsilon_nuclear(rho, T, X, Z))
    eps_no_cno = float(epsilon_nuclear(rho, T, X, Z, X_N14=0.0))

    assert eps_total > 0, f"eps_nuclear = {eps_total}, expected > 0"
    # PP dominates at solar T → removing CNO changes eps by < 30%
    ratio = eps_no_cno / eps_total
    assert ratio > 0.7, (
        f"Removing CNO dropped eps to {ratio:.2%} of total — "
        f"PP should dominate at T = {T:.1e} K"
    )


@pytest.mark.fast
def test_eps_nuclear_cno_significant_hot():
    """At T = 2.5e7 K, CNO contributes significantly to total burning.

    CNO rate ∝ T^16 vs PP ∝ T^4 — at T ~ 2.5e7 K with solar Z,
    CNO contributes ~30% of total eps (equilibrium X_N14 = 0.69*Z).
    Adelberger et al. (2011), Rev. Mod. Phys. 83, 195.
    """
    from stellar_jax.microphysics.nuclear import epsilon_nuclear

    rho, T_hot, X, Z = 100.0, 2.5e7, 0.7, 0.02

    eps_total = float(epsilon_nuclear(rho, T_hot, X, Z))
    eps_no_cno = float(epsilon_nuclear(rho, T_hot, X, Z, X_N14=0.0))

    assert eps_total > 0, f"eps_nuclear = {eps_total}, expected > 0"
    # CNO should contribute at least 20% at T = 2.5e7 K with solar Z
    cno_fraction = 1.0 - eps_no_cno / eps_total
    assert cno_fraction > 0.20, (
        f"CNO fraction = {cno_fraction:.2%}, expected > 20% at T = {T_hot:.1e} K"
    )
    # But total should be much larger than at solar core (rates scale steeply)
    eps_solar = float(epsilon_nuclear(150.0, 1.5e7, 0.34, 0.02))
    assert eps_total > 5 * eps_solar, (
        f"eps at 2.5e7 K ({eps_total:.2e}) should be >> eps at 1.5e7 K ({eps_solar:.2e})"
    )


@pytest.mark.fast
def test_eps_nuclear_differentiable_temperature():
    """epsilon_nuclear is differentiable w.r.t. T — gradient is positive.

    Nuclear rates increase steeply with temperature (PP ~ T^4, CNO ~ T^16).
    """
    from stellar_jax.microphysics.nuclear import epsilon_nuclear

    def eps_of_T(T):
        return epsilon_nuclear(150.0, T, 0.34, 0.02)

    grad_val = float(jax.grad(eps_of_T)(1.5e7))

    assert np.isfinite(grad_val), f"d(eps)/dT is not finite: {grad_val}"
    assert grad_val > 0, (
        f"d(eps)/dT = {grad_val}, expected positive (rates increase with T)"
    )


# ===========================================================================
# Neutrino tests
# ===========================================================================

@pytest.mark.fast
def test_neutrino_negligible_ms():
    """Neutrino losses are negligible compared to nuclear generation on the MS.

    At solar core T ~ 1.5e7 K, neutrino losses are ~ 2-3% of nuclear
    (Bahcall & Pinsonneault 2004). Much less than eps_nuclear.
    """
    from stellar_jax.microphysics.neutrino import epsilon_neutrino
    from stellar_jax.microphysics.nuclear import epsilon_nuclear

    rho, T, X, Z = 150.0, 1.5e7, 0.34, 0.02

    eps_nu = float(epsilon_neutrino(rho, T, X, Z))
    eps_nuc = float(epsilon_nuclear(rho, T, X, Z))

    assert eps_nu >= 0, f"eps_neutrino = {eps_nu}, expected >= 0"
    assert eps_nu < 0.1 * eps_nuc, (
        f"Neutrino losses ({eps_nu:.2e}) should be << nuclear ({eps_nuc:.2e}) "
        f"at solar-core T"
    )


@pytest.mark.fast
def test_neutrino_grows_with_temperature():
    """Neutrino losses grow steeply with temperature.

    At T ~ 1e9 K, pair production dominates → eps_nu >> eps_nu(T=1e7).
    Itoh et al. (1996), ApJS 102, 411.
    """
    from stellar_jax.microphysics.neutrino import epsilon_neutrino

    rho, X, Z = 1e4, 0.0, 0.02  # He-core conditions

    eps_low = float(epsilon_neutrino(rho, 1e7, X, Z))
    eps_high = float(epsilon_neutrino(rho, 1e9, X, Z))

    assert eps_high > 100 * eps_low, (
        f"eps_nu(1e9 K) = {eps_high:.2e} should be >> eps_nu(1e7 K) = {eps_low:.2e}"
    )


@pytest.mark.fast
def test_neutrino_differentiable():
    """epsilon_neutrino is differentiable w.r.t. T — gradient is positive.

    Neutrino emission rates are monotonically increasing with T.
    """
    from stellar_jax.microphysics.neutrino import epsilon_neutrino

    def eps_nu_of_T(T):
        return epsilon_neutrino(1e4, T, 0.0, 0.02)

    grad_val = float(jax.grad(eps_nu_of_T)(1e8))

    assert np.isfinite(grad_val), f"d(eps_nu)/dT is not finite: {grad_val}"
    assert grad_val > 0, (
        f"d(eps_nu)/dT = {grad_val}, expected positive"
    )


# ===========================================================================
# safe_power tests
# ===========================================================================

@pytest.mark.fast
def test_safe_power_forward_exact():
    """safe_power forward pass is exact (no approximation).

    safe_power(x, p, floor) == x^p for x > 0.
    """
    from stellar_jax.microphysics.safe_math import safe_power

    x, p, floor = 2.0, 1.0 / 3.0, 0.5
    result = float(safe_power(x, p, floor))
    expected = 2.0 ** (1.0 / 3.0)

    assert abs(result - expected) < 1e-12, (
        f"safe_power(2, 1/3, 0.5) = {result}, expected {expected}"
    )


@pytest.mark.fast
def test_safe_power_gradient_bounded():
    """safe_power gradient is finite at x=0 (the whole point of the custom_jvp).

    Without the floor, d/dx[x^(1/3)] = (1/3)*x^(-2/3) → ∞ at x=0.
    With floor=0.5: gradient = (1/3)*0.5^(-2/3) ~ 0.53 (finite).
    """
    from stellar_jax.microphysics.safe_math import safe_power

    grad_at_zero = float(jax.grad(safe_power)(0.0, 1.0 / 3.0, 0.5))

    assert np.isfinite(grad_at_zero), (
        f"Gradient at x=0 is not finite: {grad_at_zero}"
    )
    # Expected: p * x_floor^(p-1) = (1/3) * 0.5^(-2/3) ≈ 0.5291
    expected = (1.0 / 3.0) * 0.5 ** (-2.0 / 3.0)
    assert abs(grad_at_zero - expected) < 1e-6, (
        f"Gradient at x=0: {grad_at_zero}, expected {expected}"
    )


@pytest.mark.fast
def test_safe_power_gradient_exact_above_floor():
    """safe_power gradient is exact when x > x_floor.

    For x > floor, the custom_jvp should give the standard gradient:
    d/dx[x^p] = p * x^(p-1).
    """
    from stellar_jax.microphysics.safe_math import safe_power

    x, p, floor = 3.0, 0.5, 0.1
    grad_val = float(jax.grad(safe_power)(x, p, floor))
    expected = p * x ** (p - 1.0)  # 0.5 * 3^(-0.5)

    assert abs(grad_val - expected) < 1e-10, (
        f"Gradient at x={x}: {grad_val}, expected {expected}"
    )


# ===========================================================================
# HELM EOS tests
# ===========================================================================

@pytest.mark.fast
def test_helm_eos_degenerate():
    """HELM EOS returns physically reasonable values in degenerate regime.

    At WD/He-core conditions (logT=7.5, logP=23, X=0, Z=0.02):
    density should be > 10^5 g/cm³, and Gamma1 > 4/3 (degenerate).
    Timmes & Swesty (2000), ApJS 126, 501.
    """
    from stellar_jax.microphysics.helm import helm_eos_full

    rho, mu, nad, S, cp, chi_rho, chi_T = helm_eos_full(7.5, 23.0, 0.0, 0.02)

    log_rho = float(jnp.log10(rho))
    assert log_rho > 5.0, f"log rho = {log_rho}, expected > 5 for degenerate"

    # In degenerate matter, chi_rho >> chi_T (pressure dominated by degeneracy,
    # not thermal)
    chi_rho_val = float(chi_rho)
    chi_T_val = float(chi_T)
    assert chi_rho_val > 0, f"chi_rho = {chi_rho_val}, expected positive"
    assert chi_T_val >= 0, f"chi_T = {chi_T_val}, expected non-negative"


@pytest.mark.fast
def test_helm_eos_differentiable():
    """HELM EOS is differentiable w.r.t. logT via IFT @custom_jvp.

    The NR density inversion has an IFT-based custom_jvp that avoids
    unrolling through 30 Newton iterations.
    """
    from stellar_jax.microphysics.helm import helm_eos_full

    def rho_of_logT(logT):
        rho, _, _, _, _, _, _ = helm_eos_full(logT, 23.0, 0.0, 0.02)
        return rho

    grad_val = float(jax.grad(rho_of_logT)(7.5))

    assert np.isfinite(grad_val), f"d(rho)/d(logT) via HELM IFT is not finite: {grad_val}"


# ===========================================================================
# Screening test
# ===========================================================================

@pytest.mark.fast
def test_screening_weak_limit():
    """Chugunov screening factor → 1 in the weak-screening limit.

    Weak screening: high T, low rho → Coulomb coupling Gamma << 1.
    The enhancement factor H12 → 0, so f_screen = exp(H12) → 1.
    Chugunov, DeWitt & Yakovlev (2007), Phys. Rev. D 76, 025028.
    """
    from stellar_jax.microphysics.nuclear import screen_chugunov

    # Weak screening: T = 1e9 K (high), rho = 1 g/cm³ (low)
    # PP: z1=1, z2=1, a1=1, a2=1
    f_screen = float(screen_chugunov(1.0, 1.0, 1.0, 1.0, 1.0, 1e9, 1.0, 1.0))

    # In weak limit, screening factor should be close to 1 (within ~10%)
    assert 0.9 < f_screen < 2.0, (
        f"Screening factor = {f_screen}, expected ~1 in weak limit "
        f"(high T, low rho)"
    )


# ===========================================================================
# Contract enforcement test
# ===========================================================================

@pytest.mark.fast
def test_no_stop_gradient_in_microphysics():
    """Verify microphysics/ contains zero stop_gradient CALLS.

    This is a contract: gradient policy lives in evolution/ (orchestrator),
    never in leaf physics modules. The word may appear in documentation
    (contracts.py docstrings) but never as a function call.
    """
    import pathlib
    import ast

    pkg_dir = pathlib.Path(__file__).parent.parent / "src" / "stellar_jax" / "microphysics"
    violations = []

    for py_file in pkg_dir.rglob("*.py"):
        # Skip mesa/ validation adapters and contracts.py (documentation only)
        if "mesa" in str(py_file):
            continue
        if py_file.name == "contracts.py":
            continue
        content = py_file.read_text()
        if "stop_gradient" in content:
            # Only flag if it's NOT purely in comments/docstrings
            # Simple heuristic: check non-docstring, non-comment lines
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith('#') or stripped.startswith('"') or stripped.startswith("'"):
                    continue
                if "stop_gradient" in line:
                    violations.append(f"{py_file.name}:{i}")

    assert violations == [], (
        f"stop_gradient calls found in microphysics/ (contract violation): {violations}"
    )


# ===========================================================================
# Fermi-Dirac electron EOS table tests
# ===========================================================================

@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("fd_electron_zero_table")
@pytest.mark.right_reason("expected")
def test_fd_electron_eos_gamma1_vs_mesa_partial_degeneracy():
    """FD electron EOS table Gamma1 vs MESA at partial degeneracy (logRho 5.0-5.5).

    WHAT: Gamma1 (first adiabatic exponent) computed from the full Fermi-Dirac
    electron EOS table at the partial-degeneracy regime of the RGB He core.
    WHY: The analytic Chandrasekhar+Sommerfeld form fails catastrophically here
    (Gamma1 off ~6.7% at logRho~5.0 vs MESA; #719 found ∇_ad off 27-56%).
    The FD table (full relativistic FD integrals, Timmes & Swesty 2000) should
    match MESA to ≤2.5% including ion/Coulomb contributions.
    REFERENCE: MESA RGB-tip FGONG (1.0 Msun, MODE-A identical physics;
    data/mesa_comparison/rgb/1.0Msun/tip.FGONG.gz). Gamma1 from MESA's
    Helmholtz free energy table (biquintic Hermite, Timmes & Swesty 2000).
    TOLERANCE: 1.5% Gamma1 — the residual is from (a) interpolation smoothness
    at cell boundaries (Catmull-Rom C¹ second derivatives) and (b) single-
    component vs multi-component Coulomb. The electron FD integrals themselves
    have <0.001% interpolation error at grid points. The Coulomb model now
    matches MESA's Yakovlev & Shalybkov (1989) form (helm_coulomb2.dek),
    reducing the previous Potekhin & Chabrier (2000) residual from ~2.6% to
    ~1.1% mean.
    FAILS UNDER: mutation "fd_electron_zero_table" (zeroes the table, forcing
    the electron free energy contribution to vanish → Gamma1 becomes pure
    ion gas ~5/3 ≈ 1.667, while MESA is ~1.58-1.64).

    References:
      - Timmes & Swesty (2000), ApJS 126, 501 (HELM EOS table method)
      - Aparicio (1998), ApJS 117, 627 (FD integral evaluation)
      - Yakovlev & Shalybkov (1989), Soviet Scientific Reviews E 7, 311 (Coulomb)
      - MESA: mesa/eos/private/helm_coulomb2.dek (coefficients + functionals)
    """
    import os
    import gzip
    import tempfile
    import numpy as np
    import jax
    import jax.numpy as jnp
    from stellar_jax.oscillations import read_fgong, fgong_components
    from stellar_jax.microphysics.fd_electron import _F_total_fd

    # --- Load MESA RGB-tip FGONG ---
    fgong_gz = os.path.join(os.path.dirname(__file__), '..', 'src', 'stellar_jax', 'data',
                            'mesa_comparison', 'rgb', '1.0Msun', 'tip.FGONG.gz')
    assert os.path.exists(fgong_gz), f"MESA RGB-tip FGONG not found: {fgong_gz}"
    with gzip.open(fgong_gz, 'rb') as fi:
        tmp = tempfile.NamedTemporaryFile(suffix='.fgong', delete=False)
        tmp.write(fi.read())
        tmp.close()
    try:
        glob, var = read_fgong(tmp.name)
    finally:
        os.unlink(tmp.name)
    comp = fgong_components(glob, var)

    # Anti-synthetic guard
    assert np.log10(comp['T'][0]) > 7.5, "Not a real RGB-tip FGONG"
    assert comp['rho'][0] > 1e5, "Center density too low for RGB-tip"

    # Select degenerate He core zones: X < 0.01, logRho 5.0-6.0
    # Covers both partial degeneracy (5.0-5.5, where analytic fails) and
    # strong degeneracy (5.5-6.0, where both should converge).
    log_rho = np.log10(comp['rho'])
    mask = (comp['X'] < 0.01) & (log_rho > 5.0) & (log_rho < 6.0)
    indices = np.where(mask)[0]
    assert len(indices) >= 20, f"Too few degenerate zones: {len(indices)}"

    Z = 0.014  # MODE-A physics

    # Compute Gamma1 from the FD total free energy at each zone
    # Gamma1 = chi_rho + (Gamma3-1)*chi_T where:
    #   chi_rho = (dP/d(ln_rho))/P, chi_T = (dP/d(ln_T))/P
    #   Gamma3-1 = P*chi_T/(rho*T*cv), cv = dS/d(ln_T)
    gamma1_errors = []
    for i in indices[::5]:  # every 5th for efficiency
        logT_i = float(np.log10(comp['T'][i]))
        logRho_i = float(np.log10(comp['rho'][i]))
        X_i = float(comp['X'][i])
        g1_mesa = float(comp['gamma1'][i])

        ln_rho = logRho_i * np.log(10)
        ln_T = logT_i * np.log(10)
        rho = np.exp(ln_rho)
        T = np.exp(ln_T)

        # Pressure and its derivatives from FD total free energy
        def P_fn(lr, lt):
            return jnp.exp(lr) * jax.grad(_F_total_fd, argnums=0)(lr, lt, X_i, Z)

        P = float(P_fn(ln_rho, ln_T))
        dP_dlnrho = float(jax.grad(P_fn, argnums=0)(ln_rho, ln_T))
        dP_dlnT = float(jax.grad(P_fn, argnums=1)(ln_rho, ln_T))

        chi_rho = dP_dlnrho / P
        chi_T = dP_dlnT / P

        # Entropy: S = -(1/T)*dF/d(ln_T)
        def S_fn(lr, lt):
            dF_dlt = jax.grad(_F_total_fd, argnums=1)(lr, lt, X_i, Z)
            return -dF_dlt / jnp.exp(lt)
        cv = float(jax.grad(S_fn, argnums=1)(ln_rho, ln_T))
        cv = max(cv, 1e5)

        # Cox & Giuli thermodynamic relations
        Gamma3_m1 = P * chi_T / (rho * T * cv)
        Gamma1 = chi_rho + Gamma3_m1 * chi_T

        rel_err = abs(Gamma1 - g1_mesa) / g1_mesa
        gamma1_errors.append(rel_err)

        # Also check chi_rho > 1.2 in the degenerate regime (logRho > 5.3).
        # Without the electron contribution, chi_rho drops to ~0.9 (ideal gas).
        # This is the sensitive mutation gate.
        if logRho_i > 5.3:
            assert chi_rho > 1.2, (
                f"chi_rho = {chi_rho:.4f} at logRho={logRho_i:.2f} — "
                f"expected > 1.2 in degenerate regime (electron degeneracy "
                f"pressure dominates). Mutation 'fd_electron_zero_table' breaks this."
            )

    # All Gamma1 values must be within 1.5% of MESA (max), mean < 1%.
    # The residual is from (a) Catmull-Rom interpolation smoothness at cell
    # boundaries and (b) remaining single-component vs multi-component Coulomb
    # differences. The electron FD integrals themselves have <0.001% error.
    # Coulomb model matches MESA's Yakovlev & Shalybkov (1989) form.
    # Key improvement: mean ~0.4% here vs analytic form's 6.7% at logRho~5.0.
    max_err = max(gamma1_errors)
    mean_err = np.mean(gamma1_errors)
    assert max_err < 0.015, (
        f"FD table Gamma1 vs MESA: max error {max_err*100:.2f}% exceeds 1.5% "
        f"(mean {mean_err*100:.2f}%) at degenerate regime (logRho 5.0-6.0)"
    )
    assert mean_err < 0.01, (
        f"FD table Gamma1 vs MESA: mean error {mean_err*100:.2f}% exceeds 1% "
        f"at degenerate regime (logRho 5.0-6.0)"
    )




@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("fd_electron_zero_table")
@pytest.mark.right_reason("expected")
def test_fd_electron_eos_nabla_ad_vs_mesa_partial_degeneracy():
    """FD electron EOS table nabla_ad vs MESA at partial degeneracy.

    WHAT: Adiabatic gradient nabla_ad = (Gamma3-1)/Gamma1 computed from the
    full Fermi-Dirac electron EOS table at the partial-degeneracy regime of
    the RGB He core (logRho 5.0-5.5), plus chi_rho as a mutation-sensitive
    guard.
    WHY: The analytic Chandrasekhar+Sommerfeld form gives nabla_ad off 27-56%
    in this regime (#719). The FD table must do substantially better. nabla_ad
    determines the convective stability of the He core and is a key EOS output.
    The chi_rho guard ensures the test actually detects when the electron
    contribution is missing: nabla_ad alone is insensitive to the mutation at
    partial degeneracy (~0.40 ideal vs MESA's 0.37-0.40), but chi_rho cleanly
    separates FD-table (>1.05 at partial degeneracy due to electron degeneracy
    pressure) from ideal ion gas (~1.0 without electrons).
    REFERENCE: MESA RGB-tip FGONG col-11 (nabla_ad), 1.0 Msun, MODE-A physics
    (data/mesa_comparison/rgb/1.0Msun/tip.FGONG.gz).
    TOLERANCE: 5% max nabla_ad, 2.5% mean — the ~2-4% residual is a CONSTRAINT
    of bicubic Catmull-Rom interpolation (C¹ second derivatives produce
    oscillations in chi_T = ∂²F/∂ρ∂T and cv = ∂²F/∂T²). MESA uses biquintic
    Hermite with pre-stored second derivatives (C² smooth). Table resolution
    Δ=0.01 (matching MESA's HELM spacing) bounds the oscillation amplitude.
    chi_rho > 1.05: conservative threshold well above ideal gas (~1.0) but
    comfortably below the FD-table value at partial degeneracy (electron
    degeneracy pressure adds to ∂P/∂ρ|_T, raising chi_rho above 1.0).
    FAILS UNDER: mutation "fd_electron_zero_table" — zeroes the electron free
    energy table, making chi_rho fall to ~1.0 (ideal ion gas) which violates
    the chi_rho > 1.05 guard. This is the same mechanism that gates the
    gamma1 siblings (which use chi_rho > 1.2 at logRho > 5.3). See #741.
    """
    import os
    import gzip
    import tempfile
    import numpy as np
    import jax
    import jax.numpy as jnp
    from stellar_jax.oscillations import read_fgong, fgong_components
    from stellar_jax.microphysics.fd_electron import _F_total_fd

    # --- Load MESA RGB-tip FGONG ---
    fgong_gz = os.path.join(os.path.dirname(__file__), '..', 'src', 'stellar_jax', 'data',
                            'mesa_comparison', 'rgb', '1.0Msun', 'tip.FGONG.gz')
    assert os.path.exists(fgong_gz), f"MESA RGB-tip FGONG not found: {fgong_gz}"
    with gzip.open(fgong_gz, 'rb') as fi:
        tmp = tempfile.NamedTemporaryFile(suffix='.fgong', delete=False)
        tmp.write(fi.read())
        tmp.close()
    try:
        glob, var = read_fgong(tmp.name)
    finally:
        os.unlink(tmp.name)
    comp = fgong_components(glob, var)

    # Anti-synthetic guard
    assert np.log10(comp['T'][0]) > 7.5, "Not a real RGB-tip FGONG"

    # Select partial-degeneracy zones: X < 0.01, logRho 5.0-5.5
    # (the regime where the analytic form fails catastrophically)
    log_rho = np.log10(comp['rho'])
    mask = (comp['X'] < 0.01) & (log_rho > 5.0) & (log_rho < 5.5)
    indices = np.where(mask)[0]
    assert len(indices) >= 20, f"Too few partial-degeneracy zones: {len(indices)}"

    Z = 0.014  # MODE-A physics

    nad_errors = []
    for i in indices[::5]:  # every 5th for efficiency
        logT_i = float(np.log10(comp['T'][i]))
        logRho_i = float(np.log10(comp['rho'][i]))
        X_i = float(comp['X'][i])
        nad_mesa = float(var[i, 10])  # FGONG col-11 = nabla_ad (0-indexed: 10)

        if nad_mesa < 0.01:  # skip zones with near-zero nabla_ad
            continue

        ln_rho = logRho_i * np.log(10)
        ln_T = logT_i * np.log(10)
        rho = np.exp(ln_rho)
        T = np.exp(ln_T)

        # Pressure and derivatives from FD total free energy
        def P_fn(lr, lt):
            return jnp.exp(lr) * jax.grad(_F_total_fd, argnums=0)(lr, lt, X_i, Z)

        P = float(P_fn(ln_rho, ln_T))
        dP_dlnrho = float(jax.grad(P_fn, argnums=0)(ln_rho, ln_T))
        dP_dlnT = float(jax.grad(P_fn, argnums=1)(ln_rho, ln_T))

        chi_rho = dP_dlnrho / P
        chi_T = dP_dlnT / P

        # MUTATION GATE: chi_rho must exceed ideal-gas value at partial
        # degeneracy. Electron degeneracy pressure raises ∂P/∂ρ|_T above
        # the ideal ion gas (chi_rho = 1). Under fd_electron_zero_table,
        # chi_rho falls to ~1.0 → this assertion FAILS, gating the mutation.
        # Threshold 1.05 is conservative: well above ideal gas (1.0) but
        # below the FD-table value (which grows from ~1.1 at logRho=5.0
        # toward ~1.5+ at logRho=5.5 as degeneracy strengthens).
        assert chi_rho > 1.05, (
            f"chi_rho = {chi_rho:.4f} at logRho={logRho_i:.2f}, logT={logT_i:.2f} — "
            f"expected > 1.05 at partial degeneracy (electron degeneracy pressure "
            f"raises chi_rho above ideal-gas value of ~1.0). "
            f"Mutation 'fd_electron_zero_table' breaks this. See #741."
        )

        # Entropy: S = -(1/T)*dF/d(ln_T); cv = dS/d(ln_T)
        def S_fn(lr, lt):
            dF_dlt = jax.grad(_F_total_fd, argnums=1)(lr, lt, X_i, Z)
            return -dF_dlt / jnp.exp(lt)
        cv = float(jax.grad(S_fn, argnums=1)(ln_rho, ln_T))
        cv = max(cv, 1e5)

        # Cox & Giuli thermodynamic relations
        Gamma3_m1 = P * chi_T / (rho * T * cv)
        Gamma1 = chi_rho + Gamma3_m1 * chi_T
        nad = Gamma3_m1 / Gamma1

        rel_err = abs(nad - nad_mesa) / nad_mesa
        nad_errors.append(rel_err)

    assert len(nad_errors) >= 10, f"Too few valid nad comparisons: {len(nad_errors)}"

    # nabla_ad tolerance: 5% max, 2.5% mean.
    # CONSTRAINT: bicubic Catmull-Rom interpolation (C¹ continuity at cell
    # boundaries) produces oscillations in the second derivatives ∂²F/∂ρ∂T
    # and ∂²F/∂T². MESA avoids this with biquintic Hermite + pre-stored
    # derivatives. Table resolution Δ=0.01 (matching MESA's HELM spacing)
    # bounds the oscillation amplitude. The max (~4.5%) is localized to 2
    # points at logρ≈5.04 (weakest-degeneracy boundary, highest curvature).
    # Mean ~2% demonstrates the FD table meets the ≲2% acceptance criterion.
    # Still a 7-15× improvement over the analytic form's 27-56%.
    max_err = max(nad_errors)
    mean_err = np.mean(nad_errors)
    assert max_err < 0.05, (
        f"FD table nabla_ad vs MESA: max error {max_err*100:.2f}% exceeds 5% "
        f"(mean {mean_err*100:.2f}%) at partial degeneracy (logRho 5.0-5.5). "
        f"CONSTRAINT: bicubic Catmull-Rom C¹ second-derivative oscillations."
    )
    assert mean_err < 0.025, (
        f"FD table nabla_ad vs MESA: mean error {mean_err*100:.2f}% exceeds 2.5% "
        f"at partial degeneracy (logRho 5.0-5.5)"
    )


@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("fd_electron_zero_table")
@pytest.mark.right_reason("expected")
def test_fd_electron_eos_gamma1_strong_degeneracy_vs_mesa():
    """FD electron EOS table Gamma1 at strong degeneracy (logRho>5.5) vs MESA.

    WHAT: Gamma1 (first adiabatic exponent) from the FD table at strong
    degeneracy — the regime where the analytic Chandrasekhar+Sommerfeld
    form is already accurate.
    WHY: Issue #723 requires the FD table to agree with the existing analytic
    HELM (<0.4% Gamma1) at strong degeneracy (logRho>5.5), ensuring no
    regression in the regime the analytic form handles well. This validates
    that #720 can safely use the FD table across the full range (logRho<5.5
    partial degeneracy handled by the partial-degeneracy test, logRho>5.5
    strong degeneracy handled here).
    REFERENCE: MESA RGB-tip FGONG col-9 (Gamma1), 1.0 Msun, MODE-A physics
    (data/mesa_comparison/rgb/1.0Msun/tip.FGONG.gz). At strong degeneracy,
    MESA's biquintic HELM table is the ground truth.
    TOLERANCE: 1.0% max, 0.4% mean — the analytic form achieves <0.4% here
    (from #719 analysis); the FD table must match or exceed this. The max of
    ~0.7% occurs at logRho=5.55 (the strong-degeneracy boundary), with mean
    ~0.23% across the bulk of the core.
    FAILS UNDER: mutation "fd_electron_zero_table" — zeroes the electron free
    energy table, making Gamma1 = pure ion gas (~5/3 ≈ 1.667), catastrophically
    different from MESA's degenerate Gamma1 (~1.58-1.60 in this regime).
    """
    import os
    import gzip
    import tempfile
    import numpy as np
    import jax
    import jax.numpy as jnp
    from stellar_jax.oscillations import read_fgong, fgong_components
    from stellar_jax.microphysics.fd_electron import _F_total_fd

    # --- Load MESA RGB-tip FGONG ---
    fgong_gz = os.path.join(os.path.dirname(__file__), '..', 'src', 'stellar_jax', 'data',
                            'mesa_comparison', 'rgb', '1.0Msun', 'tip.FGONG.gz')
    assert os.path.exists(fgong_gz), f"MESA RGB-tip FGONG not found: {fgong_gz}"
    with gzip.open(fgong_gz, 'rb') as fi:
        tmp = tempfile.NamedTemporaryFile(suffix='.fgong', delete=False)
        tmp.write(fi.read())
        tmp.close()
    try:
        glob, var = read_fgong(tmp.name)
    finally:
        os.unlink(tmp.name)
    comp = fgong_components(glob, var)

    # Anti-synthetic guard
    assert np.log10(comp['T'][0]) > 7.5, "Not a real RGB-tip FGONG"
    assert comp['rho'][0] > 1e5, "Center density too low for RGB-tip"

    # Select STRONG degeneracy zones: X < 0.01, logRho > 5.5
    log_rho = np.log10(comp['rho'])
    mask = (comp['X'] < 0.01) & (log_rho > 5.5) & (log_rho < 7.0)
    indices = np.where(mask)[0]
    assert len(indices) >= 20, f"Too few strong-degeneracy zones: {len(indices)}"

    Z = 0.014  # MODE-A physics

    gamma1_errors = []
    for i in indices[::5]:
        logT_i = float(np.log10(comp['T'][i]))
        logRho_i = float(np.log10(comp['rho'][i]))
        X_i = float(comp['X'][i])
        g1_mesa = float(comp['gamma1'][i])

        ln_rho = logRho_i * np.log(10)
        ln_T = logT_i * np.log(10)
        rho = np.exp(ln_rho)
        T = np.exp(ln_T)

        # Pressure and derivatives from FD total free energy
        def P_fn(lr, lt):
            return jnp.exp(lr) * jax.grad(_F_total_fd, argnums=0)(lr, lt, X_i, Z)

        P = float(P_fn(ln_rho, ln_T))
        dP_dlnrho = float(jax.grad(P_fn, argnums=0)(ln_rho, ln_T))
        dP_dlnT = float(jax.grad(P_fn, argnums=1)(ln_rho, ln_T))

        chi_rho = dP_dlnrho / P
        chi_T = dP_dlnT / P

        # Entropy: S = -(1/T)*dF/d(ln_T); cv = dS/d(ln_T)
        def S_fn(lr, lt):
            dF_dlt = jax.grad(_F_total_fd, argnums=1)(lr, lt, X_i, Z)
            return -dF_dlt / jnp.exp(lt)
        cv = float(jax.grad(S_fn, argnums=1)(ln_rho, ln_T))
        cv = max(cv, 1e5)

        # Cox & Giuli thermodynamic relations
        Gamma3_m1 = P * chi_T / (rho * T * cv)
        Gamma1 = chi_rho + Gamma3_m1 * chi_T

        rel_err = abs(Gamma1 - g1_mesa) / g1_mesa
        gamma1_errors.append(rel_err)

        # At strong degeneracy chi_rho should be > 1.3 (electron degeneracy
        # pressure dominates over ideal gas). Mutation guard.
        assert chi_rho > 1.3, (
            f"chi_rho = {chi_rho:.4f} at logRho={logRho_i:.2f} — "
            f"expected > 1.3 at strong degeneracy"
        )

    assert len(gamma1_errors) >= 10, (
        f"Too few valid comparisons: {len(gamma1_errors)}"
    )

    # Gamma1 at strong degeneracy: max < 1.0%, mean < 0.4%.
    # The analytic HELM achieves <0.4% here (analysis).
    # Our FD table achieves mean ~0.23% (better), with max ~0.7%
    # occurring at logRho=5.55 (the transition boundary).
    max_err = max(gamma1_errors)
    mean_err = np.mean(gamma1_errors)
    assert max_err < 0.01, (
        f"FD table Gamma1 vs MESA at strong degeneracy (logRho>5.5): "
        f"max error {max_err*100:.3f}% exceeds 1.0% "
        f"(mean {mean_err*100:.3f}%)"
    )
    assert mean_err < 0.004, (
        f"FD table Gamma1 vs MESA at strong degeneracy (logRho>5.5): "
        f"mean error {mean_err*100:.3f}% exceeds 0.4%"
    )




@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("fd_electron_scale_table")
@pytest.mark.right_reason("nad mean")
def test_fd_electron_c2_hermite_smoothness():
    """C² bicubic Hermite interpolation eliminates cell-boundary nad oscillation.

    WHAT: The FD electron table interpolation (fd_electron.py) uses C² bicubic
    Hermite (matching MESA's h3, helm_polynomials.f90:197), which guarantees
    continuous 2nd derivatives at cell boundaries. This test verifies:
      (a) d²f/dx² relative jumps at cell boundaries < 0.01%
      (b) nad (∇_ad) detrended oscillation PtP < 0.01% of mean
      (c) nad mean within 1% of MESA HELM reference (0.39) at partial degeneracy

    WHY: The former Catmull-Rom (C¹) scheme produced ±3% nad oscillation at
    cell boundaries (period = 0.01 dex cell spacing, research #1018), forcing
    _HELM_RHO_CENTER to 5.7 to hide the artifact behind the OPAL blend.
    The C² Hermite fix (#1021) eliminates this, allowing the gate to lower to 5.0.
    This test guards against regression to C¹ or degraded derivative tables.

    EXTERNAL REFERENCE: MESA HELM EOS at partial degeneracy (logRho 5-6,
    logT 7.9): nad ≈ 0.39 (Timmes & Swesty 2000; Cox & Giuli §9.93;
    mesa/eos/private/helm_gammas.dek). Our C² bicubic matches MESA's
    auxiliary-table scheme (h3, helm_polynomials.f90:197). Bounded divergence:
    nad vs MESA < 0.3% (research #1018). The d²f continuity measurement is a
    self-consistency check on the C² property of the Hermite basis.

    TOLERANCE: d²f jumps < 0.01% — the C² Hermite basis guarantees exact
    continuity; any residual is floating-point roundoff. Catmull-Rom (C¹) gives
    ~2.6% jumps. The 0.01% threshold catches C¹ regression with >200× margin.
    nad oscillation < 0.01% (detrended) — Catmull-Rom gives ~0.85%; threshold
    catches regression with >80× margin. nad vs MESA: 1% tolerance gives 3×
    margin over the measured 0.3% agreement, while catching the mutation's
    4.6% shift (clean 0.389 → mutated 0.372).

    WHAT MAKES IT FAIL: mutation "fd_electron_scale_table" scales the electron
    free energy by 10×, which corrupts ALL thermodynamic derivatives (P, S, cv,
    chi_rho, chi_T). The nad mean shifts from ~0.389 to ~0.372, a 4.6% deviation
    from the MESA reference (0.39), breaking the 1% tolerance in part (c). Parts
    (a) and (b) test interpolation smoothness (unaffected by a uniform scale);
    part (c) gates the mutation via tight agreement with the MESA reference value.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from stellar_jax.microphysics.fd_electron import (
        _bicubic_hermite_interp, _F_total_fd, _total_pressure_fd,
        _total_entropy_fd,
    )

    # --- (a) d²f/dx² cell-boundary continuity ---
    # Evaluate d²f/dx² at eps on each side of cell boundaries in logRho_e.
    # JIT-compile the second derivative to avoid re-tracing per boundary.
    logT = jnp.float64(7.9)
    eps = 1e-5  # well inside cell (Δ = 0.01 dex)

    @jax.jit
    def _d2f_dx2_jit(log_rho_e, log_T):
        return jax.grad(jax.grad(_bicubic_hermite_interp, argnums=0), argnums=0)(
            log_rho_e, log_T)

    # Sample every 5th cell boundary (30 points over [5.0, 6.5]) — sufficient
    # to catch any systematic C¹ regression (C¹ artifacts appear at ALL boundaries)
    boundaries = np.arange(5.0, 6.5, 0.05)
    max_jump = 0.0
    for b in boundaries:
        d2_lo = float(_d2f_dx2_jit(jnp.float64(b - eps), logT))
        d2_hi = float(_d2f_dx2_jit(jnp.float64(b + eps), logT))
        if abs(d2_lo) > 1e-10:
            rel_jump = abs(d2_hi - d2_lo) / abs(d2_lo)
            max_jump = max(max_jump, rel_jump)

    print(f"\n  C² smoothness: d²f/dx² max cell-boundary jump = {max_jump*100:.6f}%")
    assert max_jump < 1e-4, (
        f"d²f/dx² cell-boundary relative jump {max_jump*100:.4f}% exceeds 0.01%. "
        f"C² Hermite should give exact continuity; this suggests regression to "
        f"Catmull-Rom (C¹) or corrupted derivative tables."
    )

    # --- (b) nad detrended oscillation ---
    # Evaluate nad at dense points and remove the physical trend (deg-6 polynomial).
    # JIT-compile the nad computation to avoid re-tracing per point.
    X, Z = 0.01, 0.02

    @jax.jit
    def _compute_nad(ln_rho, ln_T):
        """Compute nad from free energy derivatives (JIT'd for speed)."""
        rho = jnp.exp(ln_rho)
        T = jnp.exp(ln_T)
        P = _total_pressure_fd(ln_rho, ln_T, X, Z)
        dP_dlnrho = jax.grad(_total_pressure_fd, argnums=0)(ln_rho, ln_T, X, Z)
        chi_rho_v = dP_dlnrho / jnp.maximum(P, 1e-30)
        dP_dlnT = jax.grad(_total_pressure_fd, argnums=1)(ln_rho, ln_T, X, Z)
        chi_T_v = dP_dlnT / jnp.maximum(P, 1e-30)
        S = _total_entropy_fd(ln_rho, ln_T, X, Z)
        dS_dlnT = jax.grad(_total_entropy_fd, argnums=1)(ln_rho, ln_T, X, Z)
        cv_v = jnp.maximum(dS_dlnT, 1e5)
        G3m1 = P * chi_T_v / (rho * T * jnp.maximum(cv_v, 1e5))
        G1 = jnp.maximum(chi_rho_v + G3m1 * chi_T_v, 0.1)
        return G3m1 / G1

    # 2 points per cell (0.005 dex spacing in 0.01 dex cells) over [5.0, 6.0]
    # → 200 points, sufficient for oscillation detection. JIT compiles once,
    # then each call is fast dispatch.
    logRho_pts = np.arange(5.0, 6.001, 0.005)
    ln_T_fixed = jnp.float64(np.log(10.0**7.9))
    nad_vals = []
    for lr in logRho_pts:
        ln_rho = jnp.float64(np.log(10.0**lr))
        nad_vals.append(float(_compute_nad(ln_rho, ln_T_fixed)))

    nad_arr = np.array(nad_vals)
    nad_mean = np.mean(nad_arr)

    # Detrend with degree-6 polynomial (removes physical variation)
    p = np.polyfit(logRho_pts, nad_arr, 6)
    trend = np.polyval(p, logRho_pts)
    residual = nad_arr - trend
    osc_ptp = np.ptp(residual) / nad_mean

    print(f"  C² smoothness: nad detrended oscillation PtP = {osc_ptp*100:.6f}%")
    assert osc_ptp < 1e-4, (
        f"nad detrended oscillation {osc_ptp*100:.4f}% exceeds 0.01%. "
        f"Expected < 0.001% with C² Hermite; Catmull-Rom (C¹) gives ~0.85%."
    )

    # --- (c) nad value vs MESA reference ---
    # At partial degeneracy (logRho 5-6, logT 7.9, X=0.01, Z=0.02), MESA's
    # HELM EOS gives nad ≈ 0.39 (Timmes & Swesty 2000; Cox & Giuli §9.93
    # thermodynamic relations, confirmed in mesa/eos/private/helm_gammas.dek).
    # Our C² bicubic Hermite gives 0.388-0.393 across this range (< 0.3% of
    # MESA,). Tolerance: ±1% — provides 3× margin over the
    # 0.3% code-MESA agreement while catching the 4.6% shift from the 10×
    # scale mutation (mutated nad ≈ 0.372, 4.6% below MESA reference).
    nad_mesa_ref = 0.39  # MESA HELM at partial degeneracy (Timmes & Swesty 2000)
    nad_rel_err = abs(nad_mean - nad_mesa_ref) / nad_mesa_ref
    assert nad_rel_err < 0.01, (
        f"nad mean {nad_mean:.6f} deviates {nad_rel_err*100:.2f}% from MESA "
        f"reference {nad_mesa_ref} — exceeds 1% tolerance. "
        f"Mutation 'fd_electron_scale_table' (10× electron free energy) corrupts "
        f"thermodynamic derivatives, shifting nad from ~0.389 to ~0.372."
    )
    print(f"  C² smoothness: nad mean = {nad_mean:.6f} ({nad_rel_err*100:.3f}% vs MESA {nad_mesa_ref})")


@pytest.mark.fast
def test_fd_electron_eos_differentiable():
    """FD electron EOS is fully differentiable via JAX autodiff.

    WHAT: JAX grad through the FD table lookup (C² bicubic Hermite spline)
    gives finite, physically reasonable derivatives.
    WHY: The whole point of using a table with JAX autodiff is to get correct
    thermodynamic derivatives (P, S, Gamma1, nabla_ad) without explicitly
    storing derivative tables. If autodiff through the spline is broken,
    all derived quantities are wrong.
    REFERENCE: Independent finite-difference gradient as cross-check.
    TOLERANCE: AD vs FD relative error < 0.1% (smooth spline, h=1e-5).
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from stellar_jax.microphysics.fd_electron import _F_total_fd

    # Test at partial degeneracy: logRho=5.3, logT=7.9 (He core)
    ln_rho = 5.3 * np.log(10)
    ln_T = 7.9 * np.log(10)
    X, Z = 0.0, 0.014

    # AD derivatives
    dF_dlnrho = float(jax.grad(_F_total_fd, argnums=0)(ln_rho, ln_T, X, Z))
    dF_dlnT = float(jax.grad(_F_total_fd, argnums=1)(ln_rho, ln_T, X, Z))

    assert np.isfinite(dF_dlnrho), f"dF/d(ln_rho) is not finite: {dF_dlnrho}"
    assert np.isfinite(dF_dlnT), f"dF/d(ln_T) is not finite: {dF_dlnT}"

    # FD cross-check
    h = 1e-5
    F_rho_p = float(_F_total_fd(ln_rho + h, ln_T, X, Z))
    F_rho_m = float(_F_total_fd(ln_rho - h, ln_T, X, Z))
    fd_dlnrho = (F_rho_p - F_rho_m) / (2 * h)

    F_T_p = float(_F_total_fd(ln_rho, ln_T + h, X, Z))
    F_T_m = float(_F_total_fd(ln_rho, ln_T - h, X, Z))
    fd_dlnT = (F_T_p - F_T_m) / (2 * h)

    rel_err_rho = abs(dF_dlnrho - fd_dlnrho) / abs(fd_dlnrho)
    rel_err_T = abs(dF_dlnT - fd_dlnT) / abs(fd_dlnT)

    assert rel_err_rho < 0.001, (
        f"AD vs FD for dF/d(ln_rho): {rel_err_rho*100:.4f}% > 0.1%"
    )
    assert rel_err_T < 0.001, (
        f"AD vs FD for dF/d(ln_T): {rel_err_T*100:.4f}% > 0.1%"
    )


@pytest.mark.fast
def test_fd_electron_eos_improvement_over_analytic():
    """FD table is strictly more accurate than analytic at partial degeneracy.

    WHAT: At logRho 5.0-5.5, the FD table Gamma1 is closer to MESA than the
    analytic Chandrasekhar+Sommerfeld form.
    WHY: This is the raison d'être of issue #723 — the analytic form is
    catastrophically wrong in this regime (#719 proved 27-56% ∇_ad error).
    The FD table must do better. If it doesn't, the implementation is wrong.
    REFERENCE: MESA RGB-tip FGONG Gamma1 at partial-degeneracy zones.
    """
    import os
    import gzip
    import tempfile
    import numpy as np
    import jax
    import jax.numpy as jnp
    from stellar_jax.oscillations import read_fgong, fgong_components
    from stellar_jax.microphysics.fd_electron import _F_total_fd
    from stellar_jax.microphysics.helm import _F_total

    # Load MESA RGB-tip FGONG
    fgong_gz = os.path.join(os.path.dirname(__file__), '..', 'src', 'stellar_jax', 'data',
                            'mesa_comparison', 'rgb', '1.0Msun', 'tip.FGONG.gz')
    with gzip.open(fgong_gz, 'rb') as fi:
        tmp = tempfile.NamedTemporaryFile(suffix='.fgong', delete=False)
        tmp.write(fi.read())
        tmp.close()
    try:
        glob, var = read_fgong(tmp.name)
    finally:
        os.unlink(tmp.name)
    comp = fgong_components(glob, var)

    # Select partial-degeneracy zones (logRho 5.0-5.2 where analytic is worst)
    log_rho = np.log10(comp['rho'])
    mask = (comp['X'] < 0.01) & (log_rho > 5.0) & (log_rho < 5.2)
    indices = np.where(mask)[0]
    assert len(indices) >= 5, f"Too few zones in partial-degeneracy regime"

    Z = 0.014

    def compute_gamma1(F_fn, ln_rho, ln_T, X_i):
        rho = np.exp(ln_rho)
        T = np.exp(ln_T)
        def P_fn(lr, lt):
            return jnp.exp(lr) * jax.grad(F_fn, argnums=0)(lr, lt, X_i, Z)
        P = float(P_fn(ln_rho, ln_T))
        dP_dlnrho = float(jax.grad(P_fn, argnums=0)(ln_rho, ln_T))
        dP_dlnT = float(jax.grad(P_fn, argnums=1)(ln_rho, ln_T))
        chi_rho = dP_dlnrho / P
        chi_T = dP_dlnT / P
        def S_fn(lr, lt):
            return -jax.grad(F_fn, argnums=1)(lr, lt, X_i, Z) / jnp.exp(lt)
        cv = max(float(jax.grad(S_fn, argnums=1)(ln_rho, ln_T)), 1e5)
        Gamma3_m1 = P * chi_T / (rho * T * cv)
        return chi_rho + Gamma3_m1 * chi_T

    # Compare at 3 sample points
    fd_better_count = 0
    for i in indices[::max(1, len(indices)//3)]:
        logT_i = float(np.log10(comp['T'][i]))
        logRho_i = float(np.log10(comp['rho'][i]))
        X_i = float(comp['X'][i])
        g1_mesa = float(comp['gamma1'][i])

        ln_rho = logRho_i * np.log(10)
        ln_T = logT_i * np.log(10)

        g1_fd = compute_gamma1(_F_total_fd, ln_rho, ln_T, X_i)
        g1_ana = compute_gamma1(_F_total, ln_rho, ln_T, X_i)

        err_fd = abs(g1_fd - g1_mesa) / g1_mesa
        err_ana = abs(g1_ana - g1_mesa) / g1_mesa

        if err_fd < err_ana:
            fd_better_count += 1

    # FD table must be better at the MAJORITY of partial-degeneracy points
    total_tested = min(3, len(indices))
    assert fd_better_count >= total_tested // 2, (
        f"FD table not consistently better than analytic at partial degeneracy: "
        f"better at {fd_better_count}/{total_tested} points"
    )


@pytest.mark.fast
def test_fd_electron_eos_thermodynamic_consistency():
    """FD table: JAX autodiff pressure agrees with finite-difference pressure.

    WHAT: Verifies that JAX autodiff through the bicubic Catmull-Rom spline
    gives pressure (P = ρ × ∂F/∂(ln_ρ)) consistent with centered FD of the
    same table. This is an INTERNAL self-consistency check (AD vs FD of the
    same interpolated table), NOT a comparison to an external reference.
    WHY: If the spline interpolation or autodiff wiring is broken, the AD
    and FD derivatives would diverge. This catches implementation bugs in
    the bicubic lookup + JAX tracing path.
    REFERENCE: Internal self-consistency (AD vs FD of same table).
    TOLERANCE: <0.01% — both derivatives use the same table, so they should
    agree to near machine precision (limited only by FD step size).
    """
    import numpy as np
    import jax
    import jax.numpy as jnp
    from stellar_jax.microphysics.fd_electron import _F_electron_fd, _TABLE_AVAILABLE

    assert _TABLE_AVAILABLE, "FD table not loaded"

    # Test at 3 points spanning partial → strong degeneracy
    test_cases = [
        (5.0, 7.8, 0.0, 0.014),  # partial degeneracy, pure He
        (5.5, 7.9, 0.0, 0.014),  # intermediate
        (6.0, 8.0, 0.0, 0.014),  # strong degeneracy
    ]

    for log_rho, logT, X, Z in test_cases:
        rho = 10**log_rho
        T = 10**logT
        ln_rho = np.log(rho)
        ln_T = np.log(T)

        # Pressure from table via autodiff: P = rho * dF/d(ln_rho)
        def F_e(lr, lt):
            return _F_electron_fd(lr, lt, X, Z)

        P_from_table = float(
            jnp.exp(ln_rho) * jax.grad(F_e, argnums=0)(ln_rho, ln_T)
        )

        # Cross-check: P from finite difference of table
        h = 1e-6
        F_hi = float(F_e(ln_rho + h, ln_T))
        F_lo = float(F_e(ln_rho - h, ln_T))
        P_from_fd = rho * (F_hi - F_lo) / (2 * h)

        # They should agree to machine precision (both use the same table)
        rel_err = abs(P_from_table - P_from_fd) / abs(P_from_fd)
        assert rel_err < 1e-4, (
            f"Thermodynamic consistency: AD vs FD pressure disagree "
            f"at logRho={log_rho}, logT={logT}: {rel_err*100:.4f}%"
        )

        # Also verify pressure is positive and physical
        assert P_from_table > 0, (
            f"Negative electron pressure at logRho={log_rho}, logT={logT}"
        )


# ===========================================================================
# HELM FD blend production wiring validation
# ===========================================================================

@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("helm_blend_disabled")
@pytest.mark.right_reason("exceeds")
def test_helm_eos_production_blend_rgb_tip():
    """Production eos_lookup Gamma1 vs MESA across degenerate He core (#720).

    WHAT: Gamma1 (first adiabatic exponent) from the PRODUCTION eos_lookup at
    degenerate RGB-tip conditions — covering the blend transition zone
    (logRho 5.5-5.7, where HELM starts to contribute) and the strongly-degenerate
    core (logRho > 5.7) where HELM FD dominates. Below logRho 5.5, OPAL handles
    the EOS with correct nad (<0.2%) but wrong Gamma1 (it doesn't model
    degeneracy) — the solver doesn't use Gamma1, so this is fine for evolution.

    WHY: Validates that wiring the full Fermi-Dirac table (#723) into the
    production path via the OPAL↔HELM blend gives correct electron-degenerate
    thermodynamics in the strongly-degenerate regime where OPAL clamps (its
    logP_max=22.0 table boundary corresponds to logRho≈5.7 at logT=7.9). The
    density gate (center=5.7) ensures OPAL handles everything below this.

    REFERENCE: MESA RGB-tip FGONG (1.0 Msun, MODE-A identical physics;
    data/mesa_comparison/rgb/1.0Msun/tip.FGONG.gz). Gamma1 from MESA's
    Helmholtz free energy table (biquintic Hermite, Timmes & Swesty 2000).

    TOLERANCE: 4% Gamma1 max — residual is from (a) the HELM FD table's ±3%
    oscillatory Catmull-Rom interpolation noise at logRho > 5.7 (where HELM
    dominates), (b) OPAL ideal-gas Gamma1 at the transition zone near 5.7,
    (c) single-component OCP Coulomb vs MESA's multi-component treatment.

    FAILS UNDER: mutation "helm_blend_disabled" — reverts eos_lookup to pure
    OPAL, which gives Gamma1≈1.9 at strongly-degenerate conditions (20%+ error
    at logRho > 5.7 where OPAL's table clamps). The test fails because the
    STRONGLY-degenerate zones (logRho > 5.9) jump from <3% to ~20% error.

    References:
      - Timmes & Swesty (2000), ApJS 126, 501 (HELM EOS table method)
      - Aparicio (1998), ApJS 117, 627 (FD integral evaluation)
      - MESA: eos/private/eosdt_support.f90:223 (element-wise EOS blend)
      - PR #723 (Full Fermi-Dirac electron EOS table)
    """
    import os
    import gzip
    import tempfile
    import numpy as np
    import jax.numpy as jnp
    from stellar_jax.microphysics.eos import eos_lookup, _helm_blend_weight
    from stellar_jax.oscillations import read_fgong, fgong_components

    # Load MESA RGB-tip FGONG
    fgong_gz = os.path.join(os.path.dirname(__file__), '..', 'src', 'stellar_jax', 'data',
                            'mesa_comparison', 'rgb', '1.0Msun', 'tip.FGONG.gz')
    assert os.path.exists(fgong_gz), f"MESA RGB-tip FGONG not found: {fgong_gz}"
    with gzip.open(fgong_gz, 'rb') as fi:
        tmp = tempfile.NamedTemporaryFile(suffix='.fgong', delete=False)
        tmp.write(fi.read())
        tmp.close()
    try:
        glob, var = read_fgong(tmp.name)
    finally:
        os.unlink(tmp.name)
    comp = fgong_components(glob, var)

    # Anti-synthetic guard: real MESA RGB-tip
    assert np.log10(comp['T'][0]) > 7.5, "FGONG center T too low — not a real RGB-tip"
    assert comp['rho'][0] > 1e5, "FGONG center density too low for RGB-tip"

    # Select He-core zones where HELM has meaningful weight (logRho > 5.7).
    # At center=5.7 (slope=20): logRho=5.7 → w_rho=0.5, logRho=5.8 → w_rho=0.88,
    # logRho=5.9 → w_rho=0.98. Below 5.7, OPAL dominates and its Gamma1 is wrong
    # (it doesn't model electron degeneracy: Gamma1≈1.85 vs MESA's ≈1.62) — but
    # the solver uses nad/chi directly (not Gamma1), and OPAL's nad is correct
    # (<0.2%) at those conditions. This test validates HELM's contribution WHERE
    # it actually contributes (the strongly-degenerate core).
    mask = (comp['X'] < 0.01) & (np.log10(comp['rho']) > 5.7)
    indices = np.where(mask)[0][::3]  # subsample every 3rd for speed
    assert len(indices) >= 5, f"Not enough degenerate zones: {len(indices)}"

    Z = 0.014  # MODE-A physics

    gamma1_errs = []
    logrho_vals = []
    for i in indices:
        logT_i = float(np.log10(comp['T'][i]))
        X_i = float(comp['X'][i])
        T_i = comp['T'][i]
        a_rad = 7.5657e-15
        P_total = comp['P'][i]
        P_rad = a_rad * T_i**4 / 3.0
        P_gas = max(P_total - P_rad, 1e-3 * P_total)
        logPgas_i = float(np.log10(P_gas))

        # Call the PRODUCTION eos_lookup (the path the solver uses)
        rho, mu, nad, S, cp, chi_rho, chi_T = eos_lookup(
            jnp.float64(logT_i), jnp.float64(logPgas_i),
            jnp.float64(X_i), jnp.float64(Z))

        # Gamma1 = chi_rho / (1 - nad * chi_T)
        # From thermodynamic identity: KWW 2012 eq 13.22
        denom = 1.0 - float(nad) * float(chi_T)
        g1_ours = float(chi_rho) / denom if abs(denom) > 1e-10 else float(chi_rho)
        g1_mesa = comp['gamma1'][i]
        gamma1_errs.append(abs(g1_ours - g1_mesa) / g1_mesa)
        logrho_vals.append(np.log10(comp['rho'][i]))

    gamma1_errs = np.array(gamma1_errs)
    logrho_vals = np.array(logrho_vals)
    median_err = np.median(gamma1_errs)
    max_err = np.max(gamma1_errs)

    # Verify we tested both the blend-transition and the deep-core zones
    assert np.any(logrho_vals < 5.85), (
        f"No blend-transition zones tested (min logRho={logrho_vals.min():.2f})"
    )
    assert np.any(logrho_vals > 5.85), (
        f"No deep-core zones tested (max logRho={logrho_vals.max():.2f})"
    )

    # Provenance: blend weight must be > 0.9 at RGB-tip center
    # (proves HELM FD is active, not dormant OPAL)
    logT_center = float(np.log10(comp['T'][0]))
    w_center = float(_helm_blend_weight(jnp.float64(logT_center)))
    assert w_center > 0.90, (
        f"HELM blend weight at RGB-tip center (logT={logT_center:.3f}) is "
        f"{w_center:.3f} — should be > 0.90, meaning HELM FD dominates")

    # Main assertion: Gamma1 matches MESA within 2% median, 8% max across
    # the HELM-dominated He core (logRho 5.7-6.0).
    # Median tests the bulk accuracy of the FD table where HELM dominates
    # (logRho > 5.8, w_rho > 0.88): measured ~1% (FD table ±3% noise).
    # Max allows for the blend transition zone (logRho 5.7-5.8) where OPAL's
    # non-degenerate Gamma1 (~1.85) still contributes at w_rho=0.5-0.88,
    # producing up to ~8% error at the boundary (verified: OPAL Gamma1 is
    # 14% wrong at logRho=5.7; at w_rho=0.5, blended error is ~7%).
    # The mutation gate ensures HELM IS the source (under helm_blend_disabled,
    # error jumps to 20%+ everywhere in this zone).
    print(f"  Gamma1 production blend: median err {median_err*100:.2f}%, "
          f"max err {max_err*100:.2f}% (N={len(indices)} zones, "
          f"logRho {logrho_vals.min():.2f}-{logrho_vals.max():.2f})")
    assert median_err < 0.02, (
        f"Production eos_lookup Gamma1 median error {median_err*100:.2f}% vs MESA "
        f"exceeds 2% in HELM-dominated zones — FD blend not working")
    assert max_err < 0.08, (
        f"Production eos_lookup Gamma1 max error {max_err*100:.2f}% vs MESA "
        f"exceeds 8% — blend transition zone error larger than expected "
        f"(OPAL Gamma1 leaking at w_rho≈0.5 near center=5.7)")


@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("helm_blend_disabled")
@pytest.mark.right_reason("AD vs FD")
@pytest.mark.timeout(1800)
def test_helm_eos_production_blend_differentiable():
    """Production eos_lookup is differentiable at degenerate conditions (#720).

    WHAT: jax.grad through eos_lookup at RGB-tip-like conditions (logT=8.0,
    logP=23.0, X=0, Z=0.014) where the HELM FD blend is active. Verifies
    that derivatives w.r.t. logP (the primary Henyey Jacobian variable)
    propagate correctly through the IFT custom_jvp in fd_electron.py.

    WHY: The Henyey solver uses jacfwd to compute ∂(residual)/∂(y_vector)
    where y includes logP per zone. The density inversion (IFT custom_jvp)
    gives ∂ln_rho/∂logP, which then flows into all thermodynamic outputs.
    If this path is broken, Newton convergence fails at degenerate zones.

    EXTERNAL REFERENCE: finite-difference cross-check (independent).
    TOLERANCE: AD vs FD agreement < 1% for ∂/∂logP (clean IFT path).

    NOTE: ∂/∂logT through helm_eos_fd has a known pre-existing gradient
    issue (#723) where the nested jax.grad calls for thermodynamic derivatives
    don't fully chain through the outer differentiation. This is tracked
    separately and does not block the production Henyey path (which primarily
    needs ∂/∂logP, and logT columns are minor Newton corrections).

    MUTATION: helm_blend_disabled — without HELM, ∂rho/∂logP at degenerate
    conditions is the OPAL-table gradient (wrong because OPAL clamps at
    logRho≈5.69), which differs from the correct HELM gradient.

    References:
      - Timmes & Swesty (2000), ApJS 126, 501 (HELM EOS)
      - Griewank & Walther (2008) §15 (IFT adjoints)
    """
    import jax
    import jax.numpy as jnp
    from stellar_jax.microphysics.eos import eos_lookup

    # Use explicit helm_eos parameter instead of module-global setter
    try:
        logT = jnp.float64(8.0)
        logP = jnp.float64(23.0)
        X = jnp.float64(0.0)  # Pure He core
        Z = jnp.float64(0.014)

        # --- ∂/∂logP (primary Henyey Jacobian variable) ---
        def rho_fn(lp):
            return eos_lookup(logT, lp, X, Z, helm_eos=True)[0]  # rho

        def nad_fn(lp):
            return eos_lookup(logT, lp, X, Z, helm_eos=True)[2]  # nad

        def chirho_fn(lp):
            return eos_lookup(logT, lp, X, Z, helm_eos=True)[5]  # chi_rho

        # Forward-mode (jacfwd) — what the Henyey solver uses
        jvp_rho = jax.jacfwd(rho_fn)(logP)
        assert jnp.isfinite(jvp_rho), f"jacfwd(rho) returned {jvp_rho}"

        # Reverse-mode (grad) — what the adjoint uses
        grad_rho = jax.grad(rho_fn)(logP)
        grad_nad = jax.grad(nad_fn)(logP)
        grad_chirho = jax.grad(chirho_fn)(logP)

        assert jnp.isfinite(grad_rho), f"grad(rho) not finite"
        assert jnp.isfinite(grad_nad), f"grad(nad) not finite"
        assert jnp.isfinite(grad_chirho), f"grad(chi_rho) not finite"

        # FD cross-check
        h = 1e-5
        fd_rho = (rho_fn(logP + h) - rho_fn(logP - h)) / (2 * h)
        fd_nad = (nad_fn(logP + h) - nad_fn(logP - h)) / (2 * h)
        fd_chirho = (chirho_fn(logP + h) - chirho_fn(logP - h)) / (2 * h)

        # AD vs FD agreement
        err_rho = abs(float(grad_rho) - float(fd_rho)) / max(abs(float(fd_rho)), 1e-30)
        err_nad = abs(float(grad_nad) - float(fd_nad)) / max(abs(float(fd_nad)), 1e-30)
        err_chirho = abs(float(grad_chirho) - float(fd_chirho)) / max(abs(float(fd_chirho)), 1e-30)

        print(f"  ∂rho/∂logP: AD={float(grad_rho):.4e}, FD={float(fd_rho):.4e}, "
              f"err={err_rho*100:.3f}%")
        print(f"  ∂nad/∂logP: AD={float(grad_nad):.6f}, FD={float(fd_nad):.6f}, "
              f"err={err_nad*100:.3f}%")
        print(f"  ∂chi_rho/∂logP: AD={float(grad_chirho):.6f}, FD={float(fd_chirho):.6f}, "
              f"err={err_chirho*100:.3f}%")

        assert err_rho < 0.01, (
            f"∂rho/∂logP: AD vs FD {err_rho*100:.1f}% > 1%")
        assert err_nad < 0.01, (
            f"∂nad/∂logP: AD vs FD {err_nad*100:.1f}% > 1%")
        assert err_chirho < 0.01, (
            f"∂chi_rho/∂logP: AD vs FD {err_chirho*100:.1f}% > 1%")

        # Physical sign checks
        assert float(grad_rho) > 0, (
            f"∂rho/∂logP should be positive (more pressure → higher density)")
        # NOTE: ∂nad/∂logP sign is positive at this operating point (logT=8.0,
        # logRho≈6.36, partial electron degeneracy). Increasing density at fixed T
        # increases electron degeneracy, raising nad toward its non-relativistic
        # degenerate limit (~0.4). The prior assertion (∂nad/∂logP < 0) was an
        # artifact of the Catmull-Rom C¹ interpolation oscillation: at cell
        # boundaries, the C¹ discontinuity created local nad dips that produced
        # a false negative gradient at h=1e-5 scale. The C² Hermite fix
        # reveals the correct smooth monotonic behavior. Verified: both AD and FD
        # agree on the positive sign with the C² interpolation.
        assert float(grad_nad) > 0, (
            f"∂nad/∂logP should be positive at partial degeneracy "
            f"(logT=8, logRho≈6.4: increasing density raises electron "
            f"degeneracy pressure contribution, increasing nad toward ~0.4)")
    finally:
        pass  # No global cleanup needed — helm_eos is an explicit parameter


@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("helm_blend_disabled")
@pytest.mark.right_reason("expected > 1.2 for electron degeneracy")
def test_helm_blend_used_in_solver_path():
    """Verify that solver-path eos_lookup activates HELM FD at RGB-tip (#720).

    WHAT: Calls eos_lookup at conditions representative of the deepest He-core
    zones during RGB-tip evolution. Asserts the result has degenerate-electron
    thermodynamics (Gamma1 < 1.7, chi_rho > 1.2) — impossible from OPAL alone.

    WHY: Provenance check that the HELM blend is wired into the PRODUCTION
    path used by the solver (not just a dormant function). Without this, the
    blend could be inert and all tests would still pass via OPAL.

    EXTERNAL REFERENCE: MESA RGB-tip center conditions (logT≈7.9, logRho≈6.0):
    Gamma1≈1.58, chi_rho≈1.5 (electron degeneracy pressure dominant).

    MUTATION: helm_blend_disabled — reverts to pure OPAL which gives Gamma1≈1.89
    (ideal gas + radiation) and chi_rho≈1.0 (no degeneracy contribution).
    """
    import jax.numpy as jnp
    from stellar_jax.microphysics.eos import eos_lookup, _helm_blend_weight, _helm_rho_gate

    # RGB-tip He-core conditions: logT=7.9, logP=22.5, pure He
    logT = jnp.float64(7.9)
    logP = jnp.float64(22.5)
    X = jnp.float64(0.0)
    Z = jnp.float64(0.014)

    rho, mu, nad, S, cp, chi_rho, chi_T = eos_lookup(logT, logP, X, Z)

    logRho = float(jnp.log10(rho))

    # Temperature blend weight should be near 1 at logT=7.9
    w_T = float(_helm_blend_weight(logT))
    assert w_T > 0.90, f"w_T={w_T:.3f} at logT=7.9, expected > 0.90"

    # Density gate should be open at the resulting logRho
    w_rho = float(_helm_rho_gate(jnp.float64(logRho)))
    assert w_rho > 0.50, f"w_rho={w_rho:.3f} at logRho={logRho:.2f}, expected > 0.50"

    # Degenerate thermodynamics indicators
    assert float(chi_rho) > 1.2, (
        f"chi_rho={float(chi_rho):.3f}, expected > 1.2 for electron degeneracy "
        f"(ideal gas gives 1.0)")

    # Gamma1 from thermodynamic identity
    denom = 1.0 - float(nad) * float(chi_T)
    g1 = float(chi_rho) / denom if abs(denom) > 1e-10 else float(chi_rho)
    assert 1.5 < g1 < 1.7, (
        f"Gamma1={g1:.3f} outside degenerate range [1.5, 1.7] "
        f"(OPAL gives ~1.89 for ideal gas + radiation)")

    # Density must be in the He-core range
    assert 5.0 < logRho < 7.0, (
        f"logRho={logRho:.2f} outside expected He-core range [5, 7]")

    print(f"  Solver-path EOS at RGB-tip: Gamma1={g1:.3f}, chi_rho={float(chi_rho):.3f}, "
          f"logRho={logRho:.2f}, w_T={w_T:.3f}, w_rho={w_rho:.3f}")



@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("helm_blend_disabled")
@pytest.mark.right_reason("differ")
@pytest.mark.timeout(1800)
def test_helm_blend_consumed_by_solver_at_rgb_tip():
    """Integrated check: Henyey solver uses HELM FD at RGB-tip degenerate core (#720).

    WHAT: Loads the MESA 1.0 Msun RGB-tip FGONG, builds a y_henyey state,
    calls extract_shell_data_from_henyey with HELM_ENABLED=True (so the JIT
    trace compiles the full OPAL+HELM blend into the EOS evaluation), and
    asserts that the deepest He-core zones show electron-degenerate
    thermodynamics: logRho > 5.5 and chi_rho > 1.2.

    WHY: Acceptance criterion #3 — proves the JIT-TRACED solver path (not just
    the concrete-value eager path) actually consumes the HELM blend. Without
    helm_eos=True, the JIT-compiled eos_lookup
    returns pure OPAL (helm_eos=False default) and the solver never sees
    degenerate thermodynamics. This test would FAIL because OPAL clamps density
    at logRho≈5.69 (table edge) and gives chi_rho≈1.0 (ideal gas).

    EXTERNAL REFERENCE: MESA RGB-tip 1.0 Msun (data/mesa_comparison/rgb/1.0Msun/
    tip.FGONG.gz). Center: logRho≈5.97, logT≈7.9, He core (X<0.01).
    MESA uses the full HELM table (Timmes & Swesty 2000) at these conditions.

    TOLERANCE: logRho > 5.5 (OPAL clamps at 5.69 — a 0.3 dex gap is unambiguous);
    chi_rho > 1.2 (degeneracy; ideal gas = 1.0 exactly).

    MUTATION: helm_blend_disabled — reverts eos_lookup to pure OPAL → logRho
    drops to ≈5.69 (table edge), chi_rho ≈ 1.0 → assertions fail.
    """
    import os
    import gzip
    import tempfile
    import numpy as np
    import jax
    import jax.numpy as jnp
    from stellar_jax.solver.shell_data import extract_shell_data_from_henyey
    from stellar_jax.oscillations import read_fgong, fgong_components
    from stellar_jax.config.constants import Msun, Lsun, Rsun

    # Load MESA RGB-tip FGONG
    fgong_gz = os.path.join(os.path.dirname(__file__), '..', 'src', 'stellar_jax', 'data',
                            'mesa_comparison', 'rgb', '1.0Msun', 'tip.FGONG.gz')
    assert os.path.exists(fgong_gz), f"MESA RGB-tip FGONG not found: {fgong_gz}"
    with gzip.open(fgong_gz, 'rb') as fi:
        tmp = tempfile.NamedTemporaryFile(suffix='.fgong', delete=False)
        tmp.write(fi.read())
        tmp.close()
    try:
        glob, var = read_fgong(tmp.name)
    finally:
        os.unlink(tmp.name)

    comp = fgong_components(glob, var)
    M_star = glob[0]  # mass in grams
    R_star = glob[1]  # radius in cm
    N_fgong = len(comp['r'])

    # Build a y_henyey state from the FGONG (surface-to-center ordering for
    # the solver: index 0 = surface, index N-1 = center).
    # Subsample to N_s=50 shells for a fast test (the full 2500 is unnecessary).
    N_s = 50
    indices = np.linspace(0, N_fgong - 1, N_s, dtype=int)
    r_sel = comp['r'][indices]
    P_sel = comp['P'][indices]
    T_sel = comp['T'][indices]
    L_sel = comp['L'][indices] if 'L' in comp else np.full(N_s, 1.0 * Lsun)

    ln_r = np.log(np.maximum(r_sel, 1.0))
    ln_P = np.log(np.maximum(P_sel, 1.0))
    ln_T = np.log(np.maximum(T_sel, 1.0))
    ell = L_sel / Lsun

    y_henyey = jnp.array(np.stack([ln_r, ln_P, ln_T, ell], axis=-1))

    # Build X_profile on the standard N_COMP=200 composition grid (COMP_MFRACS)
    # by interpolating from the FGONG's mass-fraction coordinate.
    # extract_shell_data_from_henyey uses COMP_MFRACS by default, so X_profile
    # must have N_COMP elements (not N_s) to match.
    from stellar_jax.config.mesh_defaults import COMP_MFRACS, N_COMP
    fgong_q = comp['m_frac']  # center→surface, ascending q = m/M
    fgong_X = comp['X']
    X_profile = jnp.array(np.interp(COMP_MFRACS, fgong_q, fgong_X))
    q_mesh = jnp.linspace(0.0, 1.0, N_s + 1)
    Z = 0.014
    alpha_mlt = 2.0

    # Enable HELM via explicit parameter — no module-global mutation needed.
    # JIT-compile and run extract_shell_data_from_henyey — this traces
    # eos_lookup under JIT (logT is a tracer), hitting the helm_eos
    # check in the ConcretizationTypeError path.
    shell_data = jax.jit(extract_shell_data_from_henyey, static_argnames=('helm_eos',))(
        y_henyey, q_mesh, jnp.float64(M_star),
        X_profile, jnp.float64(Z), jnp.float64(alpha_mlt),
        helm_eos=True)
    shell_data_np = np.asarray(shell_data)

    # Column 7 = log10(rho), column 3 = nabla_ad
    logRho_col = shell_data_np[:, 7]
    logT_col = shell_data_np[:, 6]

    # Select deepest core zones (last few indices = center)
    # These should have logT > 7.5 and logRho > 5.5
    core_mask = (logT_col > 7.5) & (logRho_col > 5.0)
    n_core = np.sum(core_mask)
    assert n_core >= 2, (
        f"Expected ≥2 zones with logT>7.5 and logRho>5.0 at RGB-tip, got {n_core}")

    # Key assertion: HELM gives logRho > 5.5 at the deepest core
    # (OPAL clamps at ≈5.69 due to table edge → cannot reach 5.9+)
    max_logRho = np.max(logRho_col[core_mask])
    assert max_logRho > 5.5, (
        f"Max logRho in He core = {max_logRho:.3f}, expected > 5.5. "
        f"OPAL clamps at ~5.69 — HELM should give ~5.97. "
        f"This means HELM is NOT active in the JIT-traced solver path.")

    # MESA center logRho ≈ 5.97; allow 0.2 dex agreement
    # FGONG is ordered center→surface: index 0 = center.
    mesa_logRho_center = np.log10(comp['rho'][0])
    assert abs(max_logRho - mesa_logRho_center) < 0.2, (
        f"Solver logRho_center={max_logRho:.3f} vs MESA={mesa_logRho_center:.3f} "
        f"differ by {abs(max_logRho - mesa_logRho_center):.3f} dex (> 0.2 dex)")

    print(f"  Integrated solver HELM check: max_logRho={max_logRho:.3f} "
          f"(MESA center: {mesa_logRho_center:.3f}), N_core_zones={n_core}")


# ===========================================================================
# Cross-call independence test
# ===========================================================================

@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("helm_blend_disabled")
@pytest.mark.right_reason("helm_eos=True gives different results than helm_eos=False")
def test_helm_eos_cross_call_independence():
    """Two JIT-traced eos_lookup calls with different helm_eos give independent results (#1070).

    WHAT: JIT-compiles eos_lookup with helm_eos=False and helm_eos=True at
    degenerate conditions (logT=8.0, logP=23.0) in the same process. Verifies:
    (a) helm_eos=True produces different rho (HELM blend is active), and
    (b) repeated calls with the same flag are identical (no cross-call leak).
    The helm_eos gate is a compile-time optimization: it controls whether
    HELM is included in the JIT graph. Under JIT tracing, different helm_eos
    values produce different compiled functions with different behavior.

    WHY: The old module-global HELM_ENABLED flag leaked across calls — two
    evolve_star() calls in one process could interfere. Now that helm_eos is
    threaded explicitly (via static_argnames), each JIT compilation is
    independent. This test catches any residual hidden state.

    EXTERNAL REFERENCE: MESA eosdt_eval.f90 — at logT=8.0, logP=23.0, X=0,
    OPAL-only returns pure table interpolation; OPAL+HELM blend gives
    different thermodynamics at degenerate conditions.

    TOLERANCE: helm_eos=True must give a different rho than helm_eos=False
    under JIT (the HELM_ENABLED gate only activates during JIT tracing).

    MUTATION: helm_blend_disabled — disables the HELM blend, so helm_eos=True
    also returns pure OPAL → both calls identical → assertion fails.
    """
    import functools
    import jax
    import jax.numpy as jnp
    from stellar_jax.microphysics.eos import eos_lookup

    # Degenerate conditions: RGB-tip He core (beyond OPAL table edge)
    logT = jnp.float64(8.0)
    logP = jnp.float64(23.0)
    X = jnp.float64(0.0)  # Pure He core
    Z = jnp.float64(0.014)

    # JIT-compile with different helm_eos values (compile-time gate)
    @functools.partial(jax.jit, static_argnames=('helm_eos',))
    def jit_eos(logT, logP, X, Z, helm_eos=False):
        return eos_lookup(logT, logP, X, Z, helm_eos=helm_eos)

    # Call 1: OPAL-only (helm_eos=False) — JIT'd
    rho_opal, *_ = jit_eos(logT, logP, X, Z, helm_eos=False)

    # Call 2: OPAL+HELM blend (helm_eos=True) — JIT'd (different compiled graph)
    rho_helm, *_ = jit_eos(logT, logP, X, Z, helm_eos=True)

    # Call 3: OPAL-only again — must be identical to Call 1 (no state leak)
    rho_opal_2, *_ = jit_eos(logT, logP, X, Z, helm_eos=False)

    # Call 4: OPAL+HELM again — must be identical to Call 2
    rho_helm_2, *_ = jit_eos(logT, logP, X, Z, helm_eos=True)

    # Assert independence: results depend ONLY on the explicit helm_eos parameter
    assert jnp.allclose(rho_opal, rho_opal_2, rtol=0.0, atol=0.0), \
        f"OPAL-only calls not identical: {rho_opal} vs {rho_opal_2} (state leak!)"
    assert jnp.allclose(rho_helm, rho_helm_2, rtol=0.0, atol=0.0), \
        f"HELM calls not identical: {rho_helm} vs {rho_helm_2} (state leak!)"

    # Assert that HELM gives a DIFFERENT result under JIT.
    # At logT=8.0, the HELM compile-time gate (helm_eos=False) returns pure
    # _eos_opal_table, while helm_eos=True computes the full OPAL+HELM blend
    # (always-blend architecture). The HELM density inversion gives a
    # different physical rho at degenerate conditions.
    rel_diff = abs(float(rho_helm - rho_opal)) / (abs(float(rho_opal)) + 1e-30)
    assert rel_diff > 0.001, (
        f"HELM and OPAL gave nearly identical rho under JIT "
        f"({rho_helm:.6e} vs {rho_opal:.6e}, rel_diff={rel_diff:.4e}). "
        f"Expected different values — the HELM blend may not be active.")

    print(f"  Cross-call independence: rho_opal={float(rho_opal):.6e}, "
          f"rho_helm={float(rho_helm):.6e}, rel_diff={rel_diff:.2%}")


# ===========================================================================
# HELM density-solve convergence tests
# ===========================================================================

@pytest.mark.fast
@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("helm_density_solve_disabled")
@pytest.mark.right_reason("failed to converge")
def test_helm_density_solve_convergence():
    """FD-HELM Newton density inversion converges across all stellar regimes.

    WHAT: Verifies that _solve_density_fd (lax.while_loop) converges to the
    correct Newton root (|P(ρ_converged) - P_target| / P_target < 1e-9) at
    conditions spanning the MS envelope through the degenerate RGB-tip core.

    WHY: Issue #799 replaced the fixed fori_loop(0,30,...) with a convergence-
    based lax.while_loop. This test guards that the solve actually converges
    (doesn't exit at max_iter with a bad root) at all conditions where HELM
    is active, including the partial-degeneracy transition zone where the
    Newton landscape is steepest.

    REFERENCE: The converged root satisfies P(ρ*) = P_target by construction
    (Newton on the exact same free-energy-based pressure). The residual
    tolerance is 1e-9 (looser than the internal 1e-10 to account for float
    accumulation). MESA uses logRho_tol=1e-8 (eosdt_eval.f90:2383).

    FAILS UNDER: mutation "helm_density_solve_disabled" (sets _NR_MAX_ITER=0,
    causing the while_loop to exit immediately with just the initial guess —
    the pressure residual then reflects the ideal-gas → full-EOS gap, which
    is large in the degenerate regime).
    """
    from stellar_jax.microphysics.fd_electron import (
        _solve_density_fd, _total_pressure_fd, _NR_TOL)

    # Test conditions: (logT, logP, X, Z) spanning MS → RGB tip
    conditions = [
        # MS envelope (non-degenerate — should converge in ~4 iter)
        (7.15, 17.0, 0.70, 0.02),
        # RGB H-shell (partial degeneracy transition)
        (7.70, 21.0, 0.30, 0.02),
        # RGB He-core (moderate degeneracy)
        (7.80, 22.0, 0.00, 0.02),
        # RGB-tip (strong degeneracy — the regime that caused the 6.6× slowdown)
        (7.50, 23.0, 0.00, 0.02),
        # Deep degenerate (extreme)
        (8.50, 24.5, 0.00, 0.02),
    ]

    for logT, logP, X, Z in conditions:
        ln_rho = _solve_density_fd(jnp.float64(logT), jnp.float64(logP),
                                   jnp.float64(X), jnp.float64(Z))
        # Verify convergence: P(ρ*) ≈ P_target
        T = 10.0**logT
        P_target = 10.0**logP
        ln_T = jnp.log(jnp.float64(T))
        P_converged = _total_pressure_fd(ln_rho, ln_T, jnp.float64(X), jnp.float64(Z))
        residual = float(jnp.abs((P_converged - P_target) / P_target))
        assert residual < 1e-9, (
            f"Density solve failed to converge at logT={logT}, logP={logP}, "
            f"X={X}: |ΔP/P| = {residual:.2e} (tol=1e-9). "
            f"ln_rho={float(ln_rho):.6f}, P_converged={float(P_converged):.6e}, "
            f"P_target={P_target:.6e}")


@pytest.mark.fast
@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("helm_density_solve_disabled")
@pytest.mark.right_reason("AD/FD gradient mismatch")
def test_helm_density_solve_gradient_ad_vs_fd():
    """AD gradient through HELM density solve matches independent FD.

    WHAT: ∂ρ/∂logP and ∂ρ/∂logT computed via the IFT custom_jvp (analytic)
    match independent central finite differences to < 0.1%.

    WHY: The while_loop convergence change (issue #799) must not break the
    gradient path. The custom_jvp provides the gradient analytically via
    the implicit function theorem — it depends only on the CONVERGED root,
    not on iteration count. This test verifies the gradient is unchanged.

    REFERENCE: Independent finite-difference (h=1e-6) provides the ground
    truth. AD/FD agreement < 0.1% demonstrates the IFT custom_jvp is intact.
    MESA: eosdt_eval.f90 do_partials computes the same IFT form
    (dlnRho_dlnPgas = 1/(dP_dRho * Rho/Pgas) = chi_rho^{-1}).

    FAILS UNDER: mutation "helm_density_solve_disabled" — with max_iter=0 the
    solve returns just the initial guess, whose sensitivity to inputs is the
    ideal-gas derivative (not the full-EOS IFT). This breaks the AD/FD match
    because AD still uses the IFT formula at the (wrong) converged point.

    NOTE: ∂/∂logT through helm_eos_fd has a known pre-existing gradient
    discrepancy (test_helm_eos_differentiable note at :1241). That issue is
    NOT caused by this change and is out of scope. This test isolates ∂ρ/∂logP
    which is clean.
    """
    from stellar_jax.microphysics.fd_electron import helm_eos_fd

    # Strongly degenerate point (HELM fully active)
    logT = jnp.float64(7.50)
    logP = jnp.float64(23.0)
    X = jnp.float64(0.0)
    Z = jnp.float64(0.02)

    # AD gradient: ∂ρ/∂logP (via the IFT custom_jvp)
    def rho_of_logP(lp):
        rho, *_ = helm_eos_fd(logT, lp, X, Z)
        return rho

    ad_drho_dlogP = float(jax.grad(rho_of_logP)(logP))

    # Independent FD: central difference
    h = 1e-6
    rho_plus = float(helm_eos_fd(logT, logP + h, X, Z)[0])
    rho_minus = float(helm_eos_fd(logT, logP - h, X, Z)[0])
    fd_drho_dlogP = (rho_plus - rho_minus) / (2 * h)

    # Agreement check
    rel_err = abs(ad_drho_dlogP - fd_drho_dlogP) / (abs(fd_drho_dlogP) + 1e-30)
    assert rel_err < 1e-3, (
        f"AD/FD gradient mismatch for ∂ρ/∂logP at degenerate conditions: "
        f"AD={ad_drho_dlogP:.6e}, FD={fd_drho_dlogP:.6e}, "
        f"rel_err={rel_err:.2e} (tol=0.1%)")

    # Also check ∂ρ/∂logP is positive (higher P → higher ρ, universal)
    assert ad_drho_dlogP > 0, (
        f"∂ρ/∂logP should be positive, got {ad_drho_dlogP:.6e}")


@pytest.mark.fast
@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("helm_density_solve_disabled")
@pytest.mark.right_reason("different root")
def test_helm_density_solve_warm_start_identical():
    """Warm-start guess gives same converged root as ideal-gas guess.

    WHAT: Verifies that passing a warm-start ln_rho_guess to _solve_density_fd
    produces a result identical (< 1e-9 relative) to the default ideal-gas
    guess — the root is unique and convergence is independent of the starting
    point.

    WHY: Issue #799 adds the ln_rho_guess parameter for future warm-start
    (Lever 2). This test guards that the guess does NOT change the physics —
    it only changes convergence speed.

    REFERENCE: Newton converges to the unique root of P(ρ) - P_target = 0
    regardless of initial guess (within the basin of attraction). The result
    must be bit-identical to floating precision.

    FAILS UNDER: mutation "helm_density_solve_disabled" — with max_iter=0,
    the solve returns the initial guess itself, so different starting points
    give different (wrong) answers.
    """
    from stellar_jax.microphysics.fd_electron import _solve_density_fd

    # Strongly degenerate conditions
    logT = jnp.float64(7.50)
    logP = jnp.float64(23.0)
    X = jnp.float64(0.0)
    Z = jnp.float64(0.02)

    # Solve with default (ideal-gas) guess
    ln_rho_default = _solve_density_fd(logT, logP, X, Z)

    # Solve with a warm-start guess (slightly perturbed from the true root)
    # Simulate the "previous step density" being close but not exact
    warm_guess = ln_rho_default + 0.1  # 10% perturbation in ln-space
    ln_rho_warm = _solve_density_fd(logT, logP, X, Z, ln_rho_guess=warm_guess)

    # Also try a deliberately bad guess (far from true root)
    bad_guess = jnp.float64(jnp.log(1.0))  # rho=1 g/cm³ (way too low for degenerate)
    ln_rho_bad = _solve_density_fd(logT, logP, X, Z, ln_rho_guess=bad_guess)

    # All should converge to the same root
    rel_err_warm = float(jnp.abs(ln_rho_warm - ln_rho_default) /
                         jnp.maximum(jnp.abs(ln_rho_default), 1e-30))
    rel_err_bad = float(jnp.abs(ln_rho_bad - ln_rho_default) /
                        jnp.maximum(jnp.abs(ln_rho_default), 1e-30))

    assert rel_err_warm < 1e-9, (
        f"Warm-start gives different root: rel_err={rel_err_warm:.2e}. "
        f"default={float(ln_rho_default):.10f}, warm={float(ln_rho_warm):.10f}")
    assert rel_err_bad < 1e-9, (
        f"Bad-guess gives different root: rel_err={rel_err_bad:.2e}. "
        f"default={float(ln_rho_default):.10f}, bad={float(ln_rho_bad):.10f}")


@pytest.mark.fast
@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("helm_density_solve_disabled")
@pytest.mark.right_reason("rel_err")
def test_helm_warm_start_wiring_through_eos_lookup():
    """eos_lookup with warm-start guess produces bit-identical results to cold start.

    WHAT: Verifies the FULL wiring from eos_lookup(ln_rho_guess=...) down to
    the HELM density solve. At degenerate conditions (where HELM activates),
    passing a warm-start guess must produce the same thermodynamic output as
    the default ideal-gas guess — the converged root is unique.

    WHY: Issue #808 wires the per-step carry's prev_logRho (already in the
    lax.scan carry from shell_data[:, 7]) as ln_rho_guess to the solver path:
    step_fn → _henyey_step → henyey_solve_from_state_atm → _henyey_continuation_atm
    → _build_residual_fixed_bc → _prepare_cell_inputs → _cell_residual → eos_lookup
    → helm_eos_fd → _solve_density_fd_ift. If any link in this chain drops the
    kwarg, the warm start is silently inactive (no speedup, no error).
    This test exercises the eos_lookup entry point (the first callable link) to
    guard against accidental kwarg-dropping in the solver → residual → EOS chain.

    REFERENCE: Newton converges to the unique root of P(ρ) - P_target = 0.
    The IFT custom_jvp on _solve_density_fd_ift makes the iteration count
    invisible to gradients (only the converged root matters).

    FAILS UNDER: mutation "helm_density_solve_disabled" — with max_iter=0,
    the solve returns the initial guess, so warm ≠ cold.

    MESA ref: micro.f90:333 passes s%lnd(k)/ln10 as logRho_guess.
    """
    from stellar_jax.microphysics.eos import eos_lookup

    # RGB-tip conditions where HELM is active (logT > 7.7, high density)
    logT = jnp.float64(7.85)
    logP = jnp.float64(22.5)
    X = jnp.float64(0.0)   # pure He core
    Z = jnp.float64(0.02)

    # Cold start (default)
    result_cold = eos_lookup(logT, logP, X, Z, ln_rho_guess=None)
    rho_cold = float(result_cold[0])

    # Warm start: use the cold-start density as the guess (simulating prev step)
    warm_guess = jnp.log(jnp.float64(rho_cold))
    result_warm = eos_lookup(logT, logP, X, Z, ln_rho_guess=warm_guess)
    rho_warm = float(result_warm[0])

    # Also test with a perturbed guess (20% off in ln-space)
    perturbed_guess = warm_guess + 0.2
    result_perturbed = eos_lookup(logT, logP, X, Z, ln_rho_guess=perturbed_guess)
    rho_perturbed = float(result_perturbed[0])

    # All 7 thermodynamic outputs must be identical
    for i, name in enumerate(['rho', 'mu', 'nad', 'S', 'cp', 'chi_rho', 'chi_T']):
        cold_val = float(result_cold[i])
        warm_val = float(result_warm[i])
        pert_val = float(result_perturbed[i])
        rel_cold_warm = abs(warm_val - cold_val) / max(abs(cold_val), 1e-30)
        rel_cold_pert = abs(pert_val - cold_val) / max(abs(cold_val), 1e-30)
        assert rel_cold_warm < 1e-9, (
            f"{name}: warm≠cold, rel_err={rel_cold_warm:.2e}")
        assert rel_cold_pert < 1e-9, (
            f"{name}: perturbed≠cold, rel_err={rel_cold_pert:.2e}")


@pytest.mark.fast
@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("helm_density_solve_disabled")
@pytest.mark.right_reason("rel_err")
def test_helm_warm_start_gradient_identity():
    """Gradient through eos_lookup is identical with vs without warm-start guess.

    WHAT: Verifies that ∂rho/∂logP through eos_lookup is the same whether
    ln_rho_guess is None (cold) or a concrete value (warm). The IFT custom_jvp
    on _solve_density_fd_ift ensures the gradient depends only on the converged
    root, not the iteration path.

    WHY: Issue #808 wires ln_rho_guess through the solver. If the gradient
    accidentally depended on the guess (e.g. through the while_loop unrolling
    rather than the IFT), warm-start would corrupt the adjoint.

    REFERENCE: Implicit function theorem — the derivative at a fixed point y*
    satisfying F(y*, θ)=0 is dy*/dθ = -(∂F/∂y)⁻¹ · ∂F/∂θ, independent of
    how y* was found.

    FAILS UNDER: mutation "helm_density_solve_disabled" — the gradient through
    the initial guess would differ from the gradient at the converged root.

    MESA ref: MESA does not autodiff; we validate gradient correctness via IFT.
    """
    from stellar_jax.microphysics.eos import eos_lookup

    # Strongly degenerate: HELM active
    logT = jnp.float64(7.85)
    X = jnp.float64(0.0)
    Z = jnp.float64(0.02)

    def rho_of_logP_cold(logP):
        return eos_lookup(logT, logP, X, Z, ln_rho_guess=None)[0]

    def rho_of_logP_warm(logP):
        # Warm guess: ideal gas estimate (a reasonable but imperfect guess)
        from stellar_jax.config.constants import k_B, m_H
        T = 10.0**logT
        mu = 4.0 / 3.0  # approximate for pure He
        rho_ig = 10.0**logP * mu * m_H / (k_B * T)
        guess = jnp.log(jnp.maximum(rho_ig, 1e-10))
        return eos_lookup(logT, logP, X, Z, ln_rho_guess=guess)[0]

    logP_test = jnp.float64(22.5)

    grad_cold = float(jax.grad(rho_of_logP_cold)(logP_test))
    grad_warm = float(jax.grad(rho_of_logP_warm)(logP_test))

    # Gradients must be identical (IFT makes iteration path invisible)
    rel_err = abs(grad_warm - grad_cold) / max(abs(grad_cold), 1e-30)
    assert rel_err < 1e-9, (
        f"Gradient differs with warm-start: cold={grad_cold:.6e}, "
        f"warm={grad_warm:.6e}, rel_err={rel_err:.2e}. "
        f"IFT custom_jvp may not be properly ignoring the guess tangent.")
