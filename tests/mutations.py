"""O2 mutation registry — deliberate physics breaks for the CI mutation gate.

Purpose (orchestrator-integrity, O2): a test only counts as *validation* if it can be WRONG.
Each entry is  name -> fn(monkeypatch)  that BREAKS a physical quantity (flip a sign, zero a term,
scale a coefficient) at its USAGE site. A `@validation` test must declare, via
`@pytest.mark.mutation("name")`, a mutation that makes it FAIL. CI then runs the test:
  - clean  -> must PASS
  - with STELLAR_MUTATION=<name> (applied via the conftest autouse fixture) -> must FAIL
A validation test that still passes under a real physics break is theater and the gate rejects it.

CONTRACT for adding a mutation:
  - Patch the attribute on the module that USES it at runtime (e.g. `evolution.epsilon_nuclear`),
    not only the definition site -- otherwise a `from x import y` binding won't be affected.
  - Keep the break unambiguous and physical (sign flip / zeroing / large scaling), not a tiny nudge.
  - Verify the usage-site path on the dev box (JAX) before relying on it.
  - NO STRAWMAN MUTATIONS: the mutation must be a REALISTIC break — a plausibly-wrong value, or the
    removal of a term with observable effect IN THE TEST'S REGIME. Specifically prohibited:
      (a) Gross constants (>100× or <0.01× physical value) that trip a convergence/residual gate
          by brute force without discriminating "feature correct" from "feature absent".
      (b) Disabling a quantity that is INERT in the test's regime (e.g. a preconditioner whose
          scaling is solution-invariant; an eps_grav term where eps_grav ≈ 0).
      (c) Breaking physics only in a regime the test does NOT exercise.
    The correct response when a legitimate mutation fails O2 is to move the test to the regime
    where the feature has observable effect — NOT to invent a grosser mutation.
    Sound examples: opacity_bump +0.3 dex (2×), cno_rate_x10 (10×), eos_mu_offset (+0.1 dex),
    neutrino_sinw_revert (wrong constant), disable_jacobian_conditioning (excessive Levenberg).

Example (commented until its usage-site path is verified during O2 CI wiring):

    @register("eps_nuc_zero")
    def _eps_nuc_zero(mp):
        import evolution, jax.numpy as jnp
        mp.setattr(evolution, "epsilon_nuclear", lambda *a, **k: jnp.zeros_like(a[0]), raising=True)
"""

MUTATIONS = {}

# ─── Module-level mutation flags (monkeypatched by the mutation function) ───
# These are read by test code that defines test-local forward models (JAX closures)
# where there is no module function to monkeypatch. The mutation sets the flag to
# True via mp.setattr; the test reads it to decide whether to apply stop_gradient.
# Default: False (clean run).
DETACH_RGB_MASS_IN_DNU_ACTIVE = False

# ─── Package aliases for bare setattr targets (src-layout fix) ──────────────────────────────────
# Many mutation fns reference bare package names as mp.setattr targets (e.g. `solver.residual`,
# `microphysics.eos`, `composition.mix`). Pre-src-layout those were bound by `import solver.residual`
# (which binds `solver`). The mechanical rewrite to `import stellar_jax.solver.residual` binds only
# `stellar_jax`, NOT `solver`, so every such reference raised NameError when the mutation activated
# (breaking the O2 gate). Bind the package names ONCE here at module level; the function-local
# `import stellar_jax.<pkg>.<sub>` statements still load and populate the submodule attribute
# (e.g. `solver.residual`), so `mp.setattr(solver.residual, ...)` resolves and patches the runtime
# usage-site module exactly as the mutation contract requires. (Functions that locally do
# `import ... as <name>` simply shadow these harmlessly.)
import stellar_jax.solver as solver              # noqa: E402
import stellar_jax.composition as composition    # noqa: E402
import stellar_jax.microphysics as microphysics  # noqa: E402
import stellar_jax.transport as transport        # noqa: E402
import stellar_jax.config as config              # noqa: E402
import stellar_jax.fgong as fgong                # noqa: E402
import stellar_jax.evolution as evolution        # noqa: E402
import stellar_jax.oscillations as oscillations  # noqa: E402


def register(name):
    """Decorator: register a mutation fn(monkeypatch) under `name`."""
    def deco(fn):
        if name in MUTATIONS:
            raise ValueError(f"duplicate mutation name: {name}")
        MUTATIONS[name] = fn
        return fn
    return deco


# ─── corrupt_gamma1: break the adiabatic exponent used in oscillation frequencies ───
# The Cowling solver computes c_s = sqrt(Γ₁ P / ρ); corrupting Γ₁ shifts all frequencies
# by >> 2 μHz, so the detrended-residual assertion in test_oscillation_vs_published_frequencies
# must FAIL under this mutation.
# Patches BOTH the NumPy/diagnostic path (oscillations._build_oscillation_grid) AND
# the JAX production path (oscillations.coefficients._build_oscillation_grid_jax) so that
# tests using compute_oscillation_freqs_jax (e.g. test_r02_ratio_vs_gyre) are also gated.
@register("corrupt_gamma1")
def _corrupt_gamma1(mp, stellar_mod=None):
    import stellar_jax.oscillations as oscillations
    import stellar_jax.oscillations.diagnostic
    import stellar_jax.oscillations.coefficients
    import stellar_jax.oscillations.eigenvalue
    import stellar_jax.oscillations.eigenfunction

    # --- NumPy/diagnostic path ---
    _orig_build = oscillations._build_oscillation_grid

    def _patched_build(glob, var):
        grid = _orig_build(glob, var)
        # Scale Γ₁ by 1.5 → c_s increases ~22%, frequencies shift by ~22%
        grid['gamma1'] = grid['gamma1'] * 1.5
        return grid

    mp.setattr(oscillations, "_build_oscillation_grid", _patched_build)
    mp.setattr(oscillations.diagnostic, "_build_oscillation_grid", _patched_build)

    # --- JAX production path ---
    # compute_oscillation_freqs_jax calls _build_oscillation_grid_jax which
    # consumes Γ₁ (var col 9) to build Vg = G*m*rho/(r*gamma1*P). Corrupt
    # by scaling the FGONG Γ₁ column before grid extraction (same approach as
    # corrupt_gamma1_jax).
    _orig_build_jax = oscillations.coefficients._build_oscillation_grid_jax

    def _patched_build_jax(glob, var):
        import numpy as np
        var_corrupted = var.copy()
        var_corrupted[:, 9] = var_corrupted[:, 9] * 1.5
        return _orig_build_jax(glob, var_corrupted)

    mp.setattr(oscillations.coefficients, "_build_oscillation_grid_jax", _patched_build_jax)
    mp.setattr(oscillations.eigenvalue, "_build_oscillation_grid_jax", _patched_build_jax)
    mp.setattr(oscillations.eigenfunction, "_build_oscillation_grid_jax", _patched_build_jax)

    # kernels.py imports _build_oscillation_grid_jax from coefficients via
    # `from .coefficients import _build_oscillation_grid_jax`, creating a LOCAL
    # binding. Patching the coefficients module attribute does NOT affect the
    # local binding — the kernels module must be patched explicitly.
    # Without this, compute_structure_kernels uses CLEAN Γ₁ even under mutation,
    # making any test that depends on the kernel computation insensitive to
    # corrupt_gamma1.
    import stellar_jax.oscillations.kernels
    mp.setattr(oscillations.kernels, "_build_oscillation_grid_jax", _patched_build_jax)

    # Eigenfunction path — eigenfunction.py imports _build_oscillation_grid_jax
    # directly from coefficients, creating its own binding that must also be patched.
    import stellar_jax.oscillations.eigenfunction
    mp.setattr(oscillations.eigenfunction, "_build_oscillation_grid_jax", _patched_build_jax)


# Concrete physics mutations are added alongside each @validation test (their usage-site paths
# verified on the dev box). The registry intentionally starts empty so the gate has a single,
# explicit source of truth rather than guessed/broken patch targets.


@register("eps_nuc_zero")
def _eps_nuc_zero(mp, stellar_mod):
    """Zero out nuclear energy generation — breaks any test that relies on burning.

    Patches epsilon_nuclear at ALL usage sites:
      - microphysics/nuclear.py (definition site — covers any test that
        imports from the definition module, e.g. test_burn_rate_*)
      - structure.py (shooting solver / diagnostic path)
      - solver/residual.py, solver/jacobian.py, solver/shell_data.py
        (production Henyey solver, called via evolution.driver)
    Patching only the stellar facade has no effect because each module holds
    its own import binding.
    """
    import jax.numpy as jnp
    import stellar_jax.microphysics.nuclear as nuc_mod
    import stellar_jax.structure as structure
    import stellar_jax.solver.residual
    import stellar_jax.solver.jacobian
    import stellar_jax.solver.shell_data
    # *args absorbs extra positional args (t_age passed as 5th positional by
    # structure.py, solver/residual.py, solver/jacobian.py, solver/shell_data.py);
    # **kwargs absorbs keyword args (X3, X3_eq_frozen, X_N14, eps_nuc_factor, etc.).
    # Together they prevent signature drift from re-breaking O2 coverage.
    _zero = lambda rho, T, X, Z, *args, **kwargs: jnp.float64(0.0)
    mp.setattr(nuc_mod, "epsilon_nuclear", _zero, raising=True)
    mp.setattr(structure, "epsilon_nuclear", _zero, raising=True)
    mp.setattr(solver.residual, "epsilon_nuclear", _zero, raising=True)
    mp.setattr(solver.jacobian, "epsilon_nuclear", _zero, raising=True)
    mp.setattr(solver.shell_data, "epsilon_nuclear", _zero, raising=True)


# ─── opacity_bump: +0.3 dex opacity shifts HR tracks beyond tolerance (M1) ───
# A 0.3 dex (2×) opacity increase shifts effective temperature by ~0.05+ dex
# (Kramers' κ∝ρT^{-3.5} → higher κ = deeper photosphere = cooler Teff).
# test_mesa_comparison tolerance is 0.03 dex → guaranteed fail.
# Ref: Rogers & Iglesias (1992), ApJS 79, 507 — opacity determines HRD location.
@register("opacity_bump")
def _opacity_bump(mp, stellar_mod):
    """Add +0.3 dex to radiative opacity — breaks HR track agreement with MESA."""
    import stellar_jax.structure as structure
    _orig_kappa = structure.kappa

    def _bumped_kappa(logT, logRho, X, Z):
        return _orig_kappa(logT, logRho, X, Z) + 0.3

    mp.setattr(structure, "kappa", _bumped_kappa, raising=True)


# ─── cno_rate_x10: 10× CNO boost prevents solar calibration (M2a) ───
# The solar luminosity is ~98% pp + ~2% CNO at the center (Bahcall 2005).
# Boosting CNO by 10× adds ~20% extra L_nuc. The Newton solver for
# (α, Y₀) → (L=L☉, R=R☉) cannot compensate: no physical (α, Y₀) pair
# yields zero residual when the nuclear source is grossly wrong.
# Ref: Bahcall, Serenelli & Basu (2005), ApJ 621, L85.
@register("cno_rate_x10")
def _cno_rate_x10(mp, stellar_mod):
    """Multiply CNO energy generation by 10× — prevents solar calibration convergence."""
    import stellar_jax.structure as structure
    import jax.numpy as jnp
    _orig_eps = structure.epsilon_nuclear

    def _boosted_eps(rho, T, X, Z, t_age):
        # Recompute with CNO amplified: eps = eps_pp + 10*eps_cno
        rho = jnp.maximum(rho, 1e-10)
        T6 = jnp.maximum(T * 1e-6, 0.5)
        T6_13 = T6 ** (1.0 / 3.0)
        T6_23 = T6_13 * T6_13
        fx = 0.133 * X * jnp.sqrt(jnp.maximum((3.0 + X) * rho, 1e-20)) / T6 ** 1.5
        fpp = 1.0 + fx * X
        psipp = 1.0 + 1.412e8 * (1.0 / jnp.maximum(X, 1e-3) - 1.0) * jnp.exp(-49.98 / T6_13)
        Cpp = 1.0 + 0.0123 * T6_13 + 0.0109 * T6_23 + 0.000938 * T6
        eps_pp = 2.38e6 * rho * X * X * fpp * psipp * Cpp / T6_23 * jnp.exp(-33.80 / T6_13)
        phi = 1.0 - 0.3 * jnp.exp(-t_age / 5.0e6)
        g_cno = 1.0 + 0.0027 * T6_13 - 0.00778 * T6_23 - 0.000149 * T6
        X_CNO = 0.69 * Z
        eps_cno = 8.67e27 * rho * X * X_CNO / T6_23 * jnp.exp(-152.28 / T6_13) * g_cno
        return eps_pp * phi + 10.0 * eps_cno  # 10× CNO

    mp.setattr(structure, "epsilon_nuclear", _boosted_eps, raising=True)


# ─── eos_mu_offset: shift mean molecular weight, breaks central conditions (M5) ───
# The ideal-gas EOS gives ρ = Pμm_H/(kT). A +0.15 offset on μ (from ~0.62 to ~0.77)
# raises ρ_c by ~24%, shifting log_rhoc by ~0.09 dex — well above the 0.03-0.05 tol
# in test_zams_properties_consistency.
# Ref: Kippenhahn, Weigert & Weiss (2012), §13.1 (ideal gas EOS).
@register("eos_mu_offset")
def _eos_mu_offset(mp, stellar_mod):
    """Inflate EOS density by +0.1 dex (26%) — breaks ZAMS log_rhoc agreement.

    The ideal-gas EOS gives ρ = Pμm_H/(kT). A +0.1 dex density bias shifts
    log_rhoc by > 0.03 dex, exceeding the tolerance for convective-core masses
    (1.5, 2.0 M☉). This simulates a systematic EOS table error.
    Patches at ALL usage sites: structure.eos_lookup, structure.eos_opal_only
    (the shooting solver's direct OPAL path), and evolution.eos_lookup.
    Ref: Kippenhahn, Weigert & Weiss (2012), §13.1 (ideal gas EOS).
    """
    import stellar_jax.structure as structure
    import stellar_jax.evolution as evolution
    import jax.numpy as jnp
    _orig_eos = structure.eos_opal_only

    def _shifted_eos(logT, logP, X, Z):
        result = _orig_eos(logT, logP, X, Z)
        # result = (rho, mu, nad, S, cp, chi_rho, chi_T)
        rho, mu, nad, S, cp, chi_rho, chi_T = result
        # Inflate density by +0.1 dex (factor 1.26)
        rho_inflated = rho * 1.2589  # 10^0.1
        return (rho_inflated, mu, nad, S, cp, chi_rho, chi_T)

    mp.setattr(structure, "eos_lookup", _shifted_eos, raising=True)
    mp.setattr(structure, "eos_opal_only", _shifted_eos, raising=True)
    mp.setattr(evolution, "eos_lookup", _shifted_eos, raising=True)


@register("zero_eps_grav")
def _zero_eps_grav(mp, stellar_mod):
    """Zero out eps_grav_frac in evolve_star output — validates that the test catches a dead formula.

    Patches stellar.evolve_star to wrap the real function and zero the eps_grav_frac key.
    This simulates a broken gravothermal calculation (eps_grav always = 0).
    """
    import numpy as np
    real_evolve = stellar_mod.evolve_star

    def wrapped(*args, **kwargs):
        r = real_evolve(*args, **kwargs)
        r['eps_grav_frac'] = np.zeros_like(np.array(r['eps_grav_frac']))
        return r

    mp.setattr(stellar_mod, "evolve_star", wrapped)


# ─── zero_center_eps_grav: disable eps_grav in the center boundary condition ───
# Patches _center_eps_grav in solver.residual to always return 0, simulating the
# pre-fix state where eps_grav was absent from the center BC.
# Reference: MESA hydro_energy.f90:108 — eps_grav is included at ALL cells
# (k==nz is center). Our _center_eps_grav implements this for BC2.
@register("zero_center_eps_grav")
def _zero_center_eps_grav(mp, stellar_mod=None):
    """Zero eps_grav at center BC — the pre-fix behavior."""
    import jax.numpy as jnp
    import stellar_jax.solver.residual

    mp.setattr(solver.residual, "_center_eps_grav",
               lambda *a, **kw: jnp.float64(0.0), raising=True)


# ─── no_convective_core: suppress Schwarzschild convection (M4) ───
# The Schwarzschild criterion triggers convection when ∇_rad > ∇_ad.
# evolve_star_diagnostic determines is_convective from shell_data columns
# (nabla_rad vs nabla_ad). By wrapping shoot_xprofile to cap nabla_rad ≤ nad
# in the returned shell_data, we force every zone to appear radiative — the
# 2 M☉ convective core (CNO-driven, KWW §22.3) never forms. Additionally,
# mlt_nabla is replaced to always return nabla_rad (no convective transport),
# which changes the structural integration itself.
# This breaks assertions (1) conv core exists, (2) boundary matches MESA,
# (3) >70% burning inside core, and (6) 2 M☉ >> 1 M☉.
# Ref: Kippenhahn, Weigert & Weiss (2012), §5.2 (Schwarzschild criterion).
@register("no_convective_core")
def _no_convective_core(mp, stellar_mod):
    """Kill all convective zones by forcing ∇_rad ≤ ∇_ad everywhere."""
    import stellar_jax.structure as structure
    import stellar_jax.evolution as evolution
    import jax.numpy as jnp

    # 1) Replace mlt_nabla to suppress convective energy transport
    def _always_radiative(nabla_rad, nad, T, P, rho, kappa, g, mu, alpha_mlt):
        return nabla_rad

    import stellar_jax.solver.residual as _sres
    import stellar_jax.solver.shell_data as _ssd
    import stellar_jax.solver.surface_bc as _ssbc
    mp.setattr(structure, "mlt_nabla", _always_radiative, raising=True)
    # production Henyey solver reads mlt_nabla via `from stellar_jax.transport import mlt_nabla`
    mp.setattr(_sres, "mlt_nabla", _always_radiative, raising=True)
    mp.setattr(_ssd, "mlt_nabla", _always_radiative, raising=True)
    mp.setattr(_ssbc, "mlt_nabla", _always_radiative, raising=True)

    # 2) Wrap shoot_xprofile to cap nabla_rad ≤ nad in returned shell_data
    #    (column 2 = nabla_rad, column 3 = nad). This ensures the diagnostic
    #    is_convective flag (nabla_rad > nad) is always False.
    #    Must patch at BOTH usage sites: structure (definition) and evolution
    #    (which holds its own binding via `from structure import shoot_xprofile`).
    _orig_shoot = structure.shoot_xprofile

    def _capped_shoot(*args, **kwargs):
        residual, shell_data, final_state = _orig_shoot(*args, **kwargs)
        nabla_rad = shell_data[:, 2]
        nad = shell_data[:, 3]
        capped_nrad = jnp.minimum(nabla_rad, nad)
        shell_data = shell_data.at[:, 2].set(capped_nrad)
        return residual, shell_data, final_state

    import stellar_jax.evolution._core as _ecore
    mp.setattr(structure, "shoot_xprofile", _capped_shoot, raising=True)
    # evolution._core holds its own `from stellar_jax.structure import shoot_xprofile` binding
    mp.setattr(_ecore, "shoot_xprofile", _capped_shoot, raising=True)


@register("corrupt_eps_grav")
def _corrupt_eps_grav(mp, stellar_mod):
    """Add +0.05 bias to eps_grav_frac — breaks energy conservation closure.

    A physically meaningful mutation: a systematic +5% bias in the
    gravothermal energy rate (e.g., from wrong cp or nabla_ad in Form C)
    adds ~5% of E_rad to E_grav, clearly violating the energy budget
    |E_rad - E_nuc - E_grav|/E_rad < 0.5%.

    Why additive (not multiplicative): on the settled MS, eps_grav is tiny
    and can be NEGATIVE (core contraction). A multiplicative scaling (e.g.,
    10×) can accidentally REDUCE the residual if the amplified E_grav
    compensates the existing E_nuc > E_rad imbalance. An additive offset
    is sign-independent and reliably detectable regardless of the true
    eps_grav magnitude.

    Ref: KWW (2012) eq. 4.18 — Form C eps_grav depends on cp and nabla_ad;
    a ~5% systematic error in either produces this magnitude of bias.
    """
    import numpy as np
    real_evolve = stellar_mod.evolve_star

    def wrapped(*args, **kwargs):
        r = real_evolve(*args, **kwargs)
        r['eps_grav_frac'] = np.array(r['eps_grav_frac']) + 0.05
        return r

    mp.setattr(stellar_mod, "evolve_star", wrapped)


# ─── corrupt_burgers_coefficients: break He-settling diffusion coefficients ───
# The Thoul+ (1994) Burgers-equation solver returns (AP_He, AT_He, AX_He_H, AX_He_He)
# — the pressure, thermal, and concentration-gradient diffusion coefficients that
# determine the He settling velocity. Zeroing AP_He removes gravitational settling
# entirely → v_He ≈ 0, which fails the test's assertion that |v_He|/v_ref > 0.15.
# Ref: Thoul, Bahcall & Loeb (1994, ApJ 421, 828), Table 3.
@register("corrupt_burgers_coefficients")
def _corrupt_burgers_coefficients(mp, stellar_mod):
    """Zero the pressure-diffusion coefficient AP_He — kills He settling."""
    import stellar_jax.evolution as evolution
    _orig = evolution._thoul_burgers_coefficients

    def _broken(X, Y, T, rho):
        AP_He, AT_He, AX_He_H, AX_He_He = _orig(X, Y, T, rho)
        return AP_He * 0.0, AT_He, AX_He_H, AX_He_He  # zero AP → no gravitational settling

    mp.setattr(evolution, "_thoul_burgers_coefficients", _broken, raising=True)


# ─── opacity_bump_source: +0.3 dex at the microphysics source module ───
# The CZ-base opacity test imports kappa directly from microphysics.opacity (the
# source module), not via structure.kappa. A +0.3 dex bump (2×) pushes the relative
# error from ~3% to >90%, well beyond the 6% tolerance.
# Ref: Rogers & Iglesias (1992), ApJS 79, 507.
@register("opacity_bump_source")
def _opacity_bump_source(mp, stellar_mod):
    """Add +0.3 dex to opacity at the microphysics source — breaks CZ-base κ test."""
    from stellar_jax.microphysics import opacity as opacity_mod
    _orig_kappa = opacity_mod.kappa

    def _bumped(logT, logRho, X, Z):
        return _orig_kappa(logT, logRho, X, Z) + 0.3

    mp.setattr(opacity_mod, "kappa", _bumped, raising=True)


# ─── force_shooting_provenance: make evolve_star report from_shooting=True ───
# The provenance test (acceptance) guards against hollow-close: it checks that
# evolve_star reports from_henyey=True and from_shooting=False. This mutation
# patches the return dict to falsely claim shooting drove the solution — if the
# provenance test is non-vacuous it MUST fail under this mutation.
# Ref: (the original hollow-close); acceptance criterion.
@register("force_shooting_provenance")
def _force_shooting_provenance(mp, stellar_mod):
    """Patch _evolve_star_jit's return to claim shooting drove observables."""
    import stellar_jax.evolution._core as _core_mod

    _orig_jit = _core_mod._evolve_star_jit

    def _patched(*args, **kwargs):
        result = _orig_jit(*args, **kwargs)
        # Override provenance flags — provenance test must catch this
        result = dict(result)
        result['from_shooting'] = True
        result['from_henyey'] = False
        return result

    # Patch _evolve_star_jit on the _core module. evolve_star() in _core.py
    # resolves _evolve_star_jit via its module globals (evolution._core.__dict__),
    # so patching the module attribute intercepts the call at the result site.
    mp.setattr(_core_mod, "_evolve_star_jit", _patched, raising=True)


# ─── compton_opacity_zero: zero out Compton opacity — breaks high-T/low-ρ test ───
# The Compton opacity (Poutanen 2017) dominates at high T / low ρ (logR < -4.5).
# Zeroing it forces the kappa function to use OPAL table values at the table edge
# (extrapolated, not physical), breaking the test's assertion that opacity matches
# the known Thomson/Klein-Nishina value.
# Ref: Poutanen (2017), ApJ 835, 119; MESA kap/private/kap_eval.f90.
@register("compton_opacity_zero")
def _compton_opacity_zero(mp, stellar_mod):
    """Zero out Compton opacity — forces OPAL-only at high-T/low-ρ."""
    from stellar_jax.microphysics import opacity as opacity_mod
    import jax.numpy as jnp

    # Replace compton_kappa with a function that returns a very low value
    # (log10(1e-30) = -30), effectively removing Compton contribution
    def _zero_compton(logT, logRho, X, Z):
        return jnp.float64(-30.0)

    mp.setattr(opacity_mod, "compton_kappa", _zero_compton, raising=True)


# ─── neutrino_sinw_revert: revert sin²θ_W to wrong 0.2319 value ───
# The correct value is 0.2229 (MESA const_def.f90, Itoh et al. 1996).
# Reverting to 0.2319 changes tfac4 from 0.8252 → 0.8408 (+1.85%) and
# tfac3 from 0.0911 → 0.1080 (+18.5%). The test asserts the MESA values,
# so it must FAIL under this mutation.
# Ref: MESA const_def.f90 weinberg_theta = 0.22290d0; Itoh et al. (1996).
@register("neutrino_sinw_revert")
def _neutrino_sinw_revert(mp, stellar_mod):
    """Revert sin²θ_W to the wrong 0.2319 — breaks MESA coupling constant agreement."""
    from stellar_jax.microphysics import neutrino as neu_mod

    # Recompute all derived constants with the WRONG sin²θ_W
    wrong_sin2tw = 0.2319
    wrong_cv = 0.5 + 2.0 * wrong_sin2tw
    wrong_cvp = 1.0 - wrong_cv
    ca = 0.5
    cap = 0.5
    wrong_tfac1 = wrong_cv**2 + ca**2 + 2.0*(wrong_cvp**2 + cap**2)
    wrong_tfac2 = wrong_cv**2 - ca**2 + 2.0*(wrong_cvp**2 - cap**2)
    wrong_tfac3 = wrong_tfac2 / wrong_tfac1
    wrong_tfac4 = 0.5 * wrong_tfac1

    mp.setattr(neu_mod, "_SIN2_TW", wrong_sin2tw, raising=True)
    mp.setattr(neu_mod, "_CV", wrong_cv, raising=True)
    mp.setattr(neu_mod, "_CVP", wrong_cvp, raising=True)
    mp.setattr(neu_mod, "_TFAC1", wrong_tfac1, raising=True)
    mp.setattr(neu_mod, "_TFAC2", wrong_tfac2, raising=True)
    mp.setattr(neu_mod, "_TFAC3", wrong_tfac3, raising=True)
    mp.setattr(neu_mod, "_TFAC4", wrong_tfac4, raising=True)


# ─── neutrino_rate_x3: 3× neutrino rate at microphysics source — breaks grid parity ───
# The Itoh et al. (1996) neutrino fitting formulas include a normalization factor
# from electroweak coupling constants (0.93153 for plasma, others for photo/pair).
# A 3× error simulates getting this normalization wrong — plausible for a reimplementation.
# Unlike neutrino_sinw_revert (which changes a 2nd-order coupling ratio by <2%), this
# mutation produces a clear 200% shift in the total rate, unambiguously exceeding the
# 5% tolerance. The factor 3 is within the sound-mutation band (0.01×..100×).
# Ref: Itoh et al. (1996), ApJS 102, 411; Haft, Raffelt & Weiss (1994), ApJ 425, 222.
@register("neutrino_rate_x3")
def _neutrino_rate_x3(mp, stellar_mod=None):
    """Scale neutrino loss rate by 3× — breaks grid parity and any test asserting MESA-level rates.

    Patches epsilon_neutrino at the microphysics.neutrino source module to return
    3× the true rate. This simulates a normalization constant error in the Itoh et al.
    fitting formulas (e.g. getting the 0.93153 electroweak factor wrong by ~3×).
    """
    from stellar_jax.microphysics import neutrino as neu_mod
    import jax.numpy as jnp
    _orig_eps = neu_mod.epsilon_neutrino

    def _scaled_eps(rho, T, X, Z):
        return _orig_eps(rho, T, X, Z) * 3.0

    mp.setattr(neu_mod, "epsilon_neutrino", _scaled_eps, raising=True)


# ─── disable_jacobian_conditioning: bypass Jacobian conditioning ───
@register("disable_jacobian_conditioning")
def _disable_jacobian_conditioning(mp, stellar_mod=None):
    """Break: inject EXCESSIVE Levenberg regularization (λ=1.0).

    With λ=1.0 (massive diagonal loading), the Newton step direction is
    dominated by the regularization rather than the true Jacobian, causing
    slow convergence or convergence to the wrong fixed point. Tests that
    validate the conditioning MUST fail under this mutation.

    Reference: Nocedal & Wright (2006), §10.2 — excessive damping prevents
    convergence in the iteration budget.
    """
    import stellar_jax.solver.conditioning_diagnostic as henyey_conditioning
    import stellar_jax.solver.conditioning as solver_conditioning
    import jax.numpy as jnp

    def _excessive_conditioning(A, B, C, rhs, R_norm, levenberg=True):
        """Excessive Levenberg: λ=1.0 → corrupts Newton direction."""
        row_max = jnp.max(jnp.abs(B), axis=-1)
        row_scale = 1.0 / jnp.maximum(row_max, 1e-30)
        D = row_scale[:, :, None]
        A_s = A * D
        B_s = B * D
        C_s = C * D
        rhs_s = rhs * row_scale
        # Excessive: λ = 1.0 regardless of R_norm → corrupts ALL paths
        eye4 = jnp.eye(4)
        B_reg = B_s + 1.0 * eye4[None, :, :]
        return A_s, B_reg, C_s, rhs_s

    # Patch the source module (solver.conditioning) — conditioning_diagnostic
    # imports from there, so conditioned_newton_armijo picks up the patch.
    mp.setattr(solver_conditioning, "_condition_system", _excessive_conditioning)


# ─── disable_armijo_line_search: bypass backtracking (Stage 2) ───
@register("disable_armijo_line_search")
def _disable_armijo_line_search(mp, stellar_mod=None):
    """Break: armijo_line_search always returns α=1 (no backtracking).

    Without backtracking, the full Newton step overshoots when the Jacobian
    linearization is invalid (cold start with O(1) residual). This causes
    oscillation/divergence, so cold-start convergence tests MUST fail.

    Reference: Nocedal & Wright (2006), §3.1 — Armijo condition guarantees
    descent; without it, Newton is only locally convergent.
    """
    import stellar_jax.solver.conditioning as solver_conditioning
    import jax.lax as lax
    import jax.numpy as jnp

    def _no_backtrack(y, dy, R, residual_fn, R_norm):
        """Always return α=1 — no line-search."""
        return lax.stop_gradient(jnp.float64(1.0))

    # Patch the source module (solver.conditioning) — conditioning_diagnostic
    # imports from there, so conditioned_newton_armijo picks up the patch.
    mp.setattr(solver_conditioning, "armijo_line_search", _no_backtrack)



# ─── disable_adaptive_mesh: bypass the reparameterization (return original mesh) ───
# The adaptive mesh works by computing a density ρ(q) ∝ |dg/dq| and concentrating
# points where g changes rapidly. Disabling it (returning y and q_mesh unchanged)
# means the mesh stays static — max gval change per cell is NOT reduced.
# This is the mutation for test_adaptive_mesh_resolution_improvement.
@register("disable_adaptive_mesh")
def _disable_adaptive_mesh(mp, stellar_mod=None):
    import stellar_jax.mesh as mesh
    import jax
    import jax.numpy as jnp

    def _noop_reparameterize(y, q_mesh, w_P=40.0, w_T=110.0, smooth_iters=3, relax_factor=0.005):
        """No-op: return state and mesh unchanged (defeats equidistribution)."""
        return y, jax.lax.stop_gradient(q_mesh)

    mp.setattr(mesh, "adaptive_mesh_reparameterize", _noop_reparameterize)


# ─── eps_nuc_zero_all: zero nuclear energy in ALL code paths (structure + solver) ───
# The Henyey-driven evolution (now production on main) uses epsilon_nuclear
# from solver/residual.py, solver/jacobian.py, and solver/shell_data.py (each holds
# its own import from microphysics.nuclear). Patching all ensures the mutation breaks
# gradient tests that run through the production Henyey path.
# Used by: test_gradient_ladder_rung*.
# ─── corrupt_gamma1_jax: break Γ₁ in the JAX oscillation solver (OSC-1) ───
# The JAX oscillation solver (oscillations/coefficients.py) uses _build_oscillation_grid_jax
# to extract structure coefficients. Corrupting Γ₁ at source (in the FGONG var array
# processing) shifts all frequencies by >>0.05 μHz, so the parity test must FAIL.
# This is the JAX-specific counterpart of corrupt_gamma1 (which targets oscillations.py).
@register("corrupt_gamma1_jax")
def _corrupt_gamma1_jax(mp, stellar_mod=None):
    """Scale Γ₁ by 1.5 in the JAX oscillation grid — shifts all frequencies ~22%."""
    import stellar_jax.oscillations.coefficients
    import stellar_jax.oscillations.eigenvalue
    _orig_build = stellar_jax.oscillations.coefficients._build_oscillation_grid_jax

    def _patched_build(glob, var):
        import numpy as np
        # Corrupt Γ₁ (column 9) before building the grid
        var_corrupted = var.copy()
        var_corrupted[:, 9] = var_corrupted[:, 9] * 1.5
        return _orig_build(glob, var_corrupted)

    # Patch the real modules (actual call sites)
    mp.setattr(stellar_jax.oscillations.coefficients, "_build_oscillation_grid_jax", _patched_build)
    mp.setattr(stellar_jax.oscillations.eigenvalue, "_build_oscillation_grid_jax", _patched_build)


@register("eps_nuc_zero_all")
def _eps_nuc_zero_all(mp, stellar_mod=None):
    """Zero nuclear energy in all code paths — breaks any gradient through burning.

    Patches epsilon_nuclear at ALL usage sites:
      - microphysics/nuclear.py (definition site — covers any test that
        imports from the definition module)
      - structure.py (shooting solver, diagnostic path)
      - solver/residual.py, solver/jacobian.py, solver/shell_data.py
        (production Henyey solver, evolution via IFT)
    This ensures that ∂logL/∂M vanishes under mutation regardless of which
    solver path is active.
    """
    import jax.numpy as jnp
    import stellar_jax.microphysics.nuclear as nuc_mod
    import stellar_jax.structure as structure
    import stellar_jax.solver.residual
    import stellar_jax.solver.jacobian
    import stellar_jax.solver.shell_data
    # *args absorbs extra positional args (t_age passed as 5th positional by
    # structure.py, solver/residual.py, solver/jacobian.py, solver/shell_data.py);
    # **kwargs absorbs keyword args (X3, X3_eq_frozen, X_N14, eps_nuc_factor, etc.).
    # Together they prevent signature drift from re-breaking O2 coverage.
    _zero = lambda rho, T, X, Z, *args, **kwargs: jnp.float64(0.0)
    mp.setattr(nuc_mod, "epsilon_nuclear", _zero, raising=True)
    mp.setattr(structure, "epsilon_nuclear", _zero, raising=True)
    mp.setattr(solver.residual, "epsilon_nuclear", _zero, raising=True)
    mp.setattr(solver.jacobian, "epsilon_nuclear", _zero, raising=True)
    mp.setattr(solver.shell_data, "epsilon_nuclear", _zero, raising=True)


# ─── unfreeze_schedule: remove the frozen-schedule adjoint (GRAD-1) ───
# The frozen-schedule adjoint (freeze_schedule=True) applies stop_gradient to
# the reject flag and dt_next computation, treating the adaptive-timestep
# schedule as constant in the backward pass. Removing this freeze (putting dt
# back on the tape) causes the AD gradient to include the discretization-error
# sensitivity ∂observable/∂schedule · ∂schedule/∂param — a spurious term that
# does NOT match frozen-schedule FD (which also uses the same fixed schedule).
#
# Mutation mechanism: monkeypatch evolve_star to force freeze_schedule=False
# even when the test requests freeze_schedule=True. The test computes AD with
# freeze_schedule=True and FD along the frozen schedule — removing the freeze
# makes AD diverge from the frozen-schedule FD reference.
#
# Reference: Griewank & Walther (2008) §15.4 — algorithmic decisions treated
# as fixed vs live in the adjoint.
@register("unfreeze_schedule")
def _unfreeze_schedule(mp, stellar_mod=None):
    """Disable the frozen-schedule adjoint — puts dt back on the gradient tape.

    Under this mutation, AD includes ∂observable/∂(dt_schedule) which does not
    match the frozen-schedule FD reference → test_adaptive_gradient_frozen_schedule
    must FAIL (AD-vs-FD > tolerance).
    """
    import stellar_jax.evolution as evolution
    _orig_evolve = evolution._evolve_star_jit

    # We need to intercept evolve_star to override freeze_schedule=False.
    # Since freeze_schedule is a static arg (changes JIT trace), we wrap the
    # public evolve_star function to always pass freeze_schedule=False.
    _orig_pub = evolution.evolve_star

    def _unfreeze_wrapper(*args, **kwargs):
        kwargs['freeze_schedule'] = False
        return _orig_pub(*args, **kwargs)

    mp.setattr(evolution, "evolve_star", _unfreeze_wrapper)
    # Also patch the stellar facade (tests import from stellar)
    if stellar_mod is not None:
        mp.setattr(stellar_mod, "evolve_star", _unfreeze_wrapper)


# ─── zero_ift_eigenfreq: break the IFT adjoint for oscillation frequencies (OSC-2) ───
# The IFT adjoint returns ∂σ²*/∂coeffs = −(∂D/∂coeffs)/(∂D/∂σ²). If we zero
# the backward pass, the AD gradient will be zero while the FD gradient is non-zero,
# so any AD-vs-FD test must FAIL.
@register("zero_ift_eigenfreq")
def _zero_ift_eigenfreq(mp, stellar_mod=None):
    """Zero the IFT backward pass for eigenfrequencies — AD gradient becomes zero."""
    import jax.numpy as jnp
    import stellar_jax.oscillations.adjoint

    # Replace the backward pass functions with versions that return zero gradients
    def _zero_radial_bwd(x_steps, h_steps, factor, res, g_sigma2):
        coeffs, x_grid, sigma2_converged = res
        return jnp.zeros_like(coeffs), jnp.zeros_like(x_grid), jnp.zeros_like(sigma2_converged)

    def _zero_nonradial_bwd(l, x_steps, h_steps, factor, res, g_sigma2):
        coeffs, x_grid, sigma2_converged = res
        return jnp.zeros_like(coeffs), jnp.zeros_like(x_grid), jnp.zeros_like(sigma2_converged)

    # Monkeypatch the backward functions on the real module
    mp.setattr(stellar_jax.oscillations.adjoint, "_eigenfreq_radial_bwd", _zero_radial_bwd)
    mp.setattr(stellar_jax.oscillations.adjoint, "_eigenfreq_nonradial_bwd", _zero_nonradial_bwd)

    # Re-register the vjp with zeroed backward pass
    import functools
    import jax

    # For the radial case
    @functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
    def eigenfreq_radial_zeroed(coeffs, x_grid, sigma2_converged, x_steps, h_steps, factor):
        return sigma2_converged

    def _fwd_r(coeffs, x_grid, sigma2_converged, x_steps, h_steps, factor):
        return sigma2_converged, (coeffs, x_grid, sigma2_converged)

    eigenfreq_radial_zeroed.defvjp(_fwd_r, _zero_radial_bwd)
    # Patch the real module (actual call site)
    mp.setattr(stellar_jax.oscillations.adjoint, "eigenfreq_radial", eigenfreq_radial_zeroed)

    # For the nonradial case
    @functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
    def eigenfreq_nonradial_zeroed(coeffs, x_grid, sigma2_converged, l, x_steps, h_steps, factor):
        return sigma2_converged

    def _fwd_nr(coeffs, x_grid, sigma2_converged, l, x_steps, h_steps, factor):
        return sigma2_converged, (coeffs, x_grid, sigma2_converged)

    eigenfreq_nonradial_zeroed.defvjp(_fwd_nr, _zero_nonradial_bwd)
    # Patch the real module (actual call site)
    mp.setattr(stellar_jax.oscillations.adjoint, "eigenfreq_nonradial", eigenfreq_nonradial_zeroed)


# ─── corrupt_target_readout: detach physical-target interpolation (GRAD-1) ───
# The smoothness scan validates that gradients across a mass sweep are smooth
# AND positive (mass-luminosity relation). The smoothness comes from the
# differentiable physical-target readout (observable_at_target uses jnp.interp
# onto a fixed age) combined with the frozen-schedule adjoint.
#
# Mutation mechanism: monkeypatch observable_at_target to DETACH its output
# via jax.lax.stop_gradient. This cuts the gradient path: jax.grad returns 0
# for all masses. The test asserts all(grads > 0) → FAILS under mutation.
#
# Why this is the right mutation for the smoothness test:
# The test validates that observable_at_target provides a DIFFERENTIABLE,
# smooth readout. If the readout is non-differentiable (gradient detached),
# no physical gradient flows → the test catches it immediately via the
# positivity check. This is what would happen if observable_at_target were
# implemented with a non-differentiable operation (e.g., argmin-based step
# selection, or an accidental stop_gradient).
#
# Note: the SEPARATE test_adaptive_gradient_frozen_schedule uses the
# 'unfreeze_schedule' mutation to demonstrate that the freeze mechanism
# specifically is needed (Part A: frozen AD ≠ unfrozen AD; Part B: frozen
# AD agrees with FD <1%). The smoothness scan uses this different mutation
# because it validates the readout mechanism, not the freeze in isolation.
#
# Reference: the test's "all positive" assertion requires d(logL)/dM > 0
# at every sweep point (mass-luminosity relation, Eddington 1924). A
# detached readout gives gradient = 0 → violates this physical constraint.
@register("corrupt_target_readout")
def _corrupt_target_readout(mp, stellar_mod=None):
    """Detach observable_at_target output — gradient becomes zero.

    Under this mutation:
    - observable_at_target wraps its output in jax.lax.stop_gradient
    - The gradient of the readout w.r.t. all inputs is therefore 0
    - The test asserts all(grads > 0) → FAILS (grads are all 0)

    This tests that the smoothness scan requires a functioning differentiable
    readout mechanism — without it, no physical gradient is produced.
    """
    import jax
    import stellar_jax.evolution as evolution

    _orig_target = evolution.observable_at_target

    def _detached_readout(result, observable='log_L', target_key='star_age',
                          target_value=None):
        """Detached readout: correct value but zero gradient."""
        val = _orig_target(result, observable=observable,
                           target_key=target_key, target_value=target_value)
        return jax.lax.stop_gradient(val)

    mp.setattr(evolution, "observable_at_target", _detached_readout)

    if stellar_mod is not None:
        mp.setattr(stellar_mod, "observable_at_target", _detached_readout)


# ─── OSC-3: detach structure→oscillation gradient link ──────────────────────────
# When this mutation is active, the gradient from eigenfrequency back through the
# structure coefficients is severed (stop_gradient on var). The forward computation
# is unchanged (frequencies still correct), but ∂ν/∂M collapses to ~0 because
# the in-memory structure→oscillation link is cut.
@register("detach_structure_oscillation")
def _detach_structure_oscillation(mp, stellar_mod=None):
    """Sever the differentiable structure→oscillation link via stop_gradient."""
    import jax
    import jax.numpy as jnp
    import stellar_jax.oscillations
    import stellar_jax.oscillations.coefficients
    import stellar_jax.oscillations.eigenvalue

    _orig_build = stellar_jax.oscillations.coefficients.build_oscillation_coeffs_jax

    def _detached_build(glob_jax, var_jax):
        # Apply stop_gradient to sever the backward pass from structure to frequencies
        return _orig_build(jax.lax.stop_gradient(glob_jax),
                           jax.lax.stop_gradient(var_jax))

    # Patch ALL bindings — the canonical definition, the eigenvalue module's
    # import, AND the package-level re-export in __init__.py.  Tests that do
    # `from stellar_jax.oscillations import build_oscillation_coeffs_jax`
    # bind to the __init__ attribute; without this third patch the test's
    # local reference points to the ORIGINAL, un-stop_gradient'd function
    # and the mutation is invisible (root cause of).
    mp.setattr(stellar_jax.oscillations.coefficients, "build_oscillation_coeffs_jax", _detached_build)
    mp.setattr(stellar_jax.oscillations.eigenvalue, "build_oscillation_coeffs_jax", _detached_build)
    mp.setattr(stellar_jax.oscillations, "build_oscillation_coeffs_jax", _detached_build)


# ─── INV-1: corrupt inverse-solver gradient path ────────────────────────────────
# The inverse solver (infer_parameters) relies on analytic Jacobians via jax.jacrev
# through the forward model. This mutation severs the gradient by stop_gradient-ing
# the evolve_star output arrays, so the Jacobian is identically zero. The forward
# values are still correct (residuals computed correctly), but the LM step direction
# is all-zeros → no convergence.
@register("detach_forward_model_gradient")
def _detach_forward_model_gradient(mp, stellar_mod=None):
    """Sever the gradient from parameters → evolution output via stop_gradient.

    The inverse solver computes J = jax.jacrev(forward_fn)(θ) where forward_fn
    reads observables from evolve_star output. Under this mutation, evolve_star
    wraps its output arrays in stop_gradient, making J = 0 for all entries.
    The LM step δθ = -(J^T J + λI)^{-1} J^T r becomes δθ ≈ 0 (since J^T r = 0
    when J = 0), so the optimizer is stuck at the initial guess and cannot
    converge to the true parameters.

    This mutation targets the _evolve_star_jit function's output arrays,
    which carry the gradient from the Henyey IFT backward pass to the user.
    """
    import jax
    import jax.numpy as jnp
    import stellar_jax.evolution as evolution

    _orig_evolve = evolution.evolve_star

    def _detached_evolve(*args, **kwargs):
        """Forward values correct, backward pass severed on output arrays."""
        result = _orig_evolve(*args, **kwargs)
        # Apply stop_gradient to the observable arrays — this kills
        # jax.jacrev through any function that reads from these arrays.
        detached = {}
        for k, v in result.items():
            if isinstance(v, jnp.ndarray) or (hasattr(v, 'shape') and hasattr(v, 'dtype')):
                detached[k] = jax.lax.stop_gradient(v)
            else:
                detached[k] = v
        return detached

    mp.setattr(evolution, "evolve_star", _detached_evolve)
    if stellar_mod is not None:
        mp.setattr(stellar_mod, "evolve_star", _detached_evolve)


# ─── disable_z_mixing: skip CZ Z-homogenization in the Henyey evolution loop ───
# When active, _mix_z_in_cz returns its input unchanged (no CZ Z-mixing).
# Patches BOTH evolution._mix_z_in_cz (for the lax.scan Henyey path) AND
# composition.mix._mix_z_in_cz (for component tests that import directly).
# This causes the CZ Z profile to become non-uniform: gravitational settling
# depletes Z at the CZ base (zone ~198 on the 200-zone grid) without
# convective replenishment, producing std(Z_CZ) >> 0.001.
# MESA does mix ALL species in CZ (mix_info.f90:set_dxdt_mix, j=1..species).
# Without Z-mixing, our code violates this fundamental convective mixing
# assumption (Böhm-Vitense 1958; Paxton+ 2011 §4).
@register("disable_z_mixing")
def _disable_z_mixing(mp, stellar_mod=None):
    """Disable CZ Z-mixing — _mix_z_in_cz becomes identity (returns Z unchanged)."""
    import stellar_jax.composition.mix
    import stellar_jax.evolution._core

    def _no_z_mix(Z_profile, shell_data, M_solar, f_ov, **kwargs):
        return Z_profile  # skip mixing, return Z as-is

    # Patch the canonical source (where component tests import from)
    mp.setattr(composition.mix, "_mix_z_in_cz", _no_z_mix, raising=True)
    # Patch the local binding in evolution._core (where production calls it)
    mp.setattr(evolution._core, "_mix_z_in_cz", _no_z_mix, raising=True)


# ─── eos_chi_rho_flat: force chi_rho_gas=1, chi_T_gas=1 (ideal-gas assumption) ───
# This reverts the table-interpolated χ_ρ/χ_T to the old ideal-gas approximation
# where chi_rho_gas = 1 and chi_T_gas = 1 everywhere. In the partial ionization
# zone (logT ~ 4-5.5) the true values deviate significantly from 1 due to
# ionization energy. Forcing them to 1 produces ~40-55% error in delta = χ_T/χ_ρ
# at logT ≈ 4.2-4.5, which must cause test_eos_chi_rho_chi_T_vs_mesa_fgong to FAIL.
# Ref: Rogers & Nayfonov 2002, ApJ 576, 1064 (OPAL EOS: true derivatives from
# free energy); KWW 2012 §13.2 (ideal-gas approximation: χ_ρ=β, χ_T=4-3β).
@register("eos_chi_rho_flat")
def _eos_chi_rho_flat(mp, stellar_mod=None):
    """Force EOS chi_rho_gas=1, chi_T_gas=1 — reverts to ideal-gas approximation.

    Patches eos_lookup to override the table-interpolated gas-pressure derivatives
    with the ideal-gas values (chi_rho_gas=1, chi_T_gas=1), making the total-pressure
    derivatives revert to chi_rho=beta, chi_T=4-3*beta. This produces large errors
    (>30%) in the partial ionization zone.
    """
    import stellar_jax.structure as structure
    import jax.numpy as jnp
    from stellar_jax.microphysics.eos import eos_lookup as _real_eos_lookup

    def _flat_chi_eos(logT, logP, X, Z):
        rho, mu, nad, S, cp, chi_rho, chi_T = _real_eos_lookup(logT, logP, X, Z)
        # Recompute with ideal-gas assumption: chi_rho_gas=1, chi_T_gas=1
        T = 10.0**logT
        P_gas = 10.0**logP
        a_rad = 7.5657e-15
        P_rad = a_rad * T**4 / 3.0
        P_total = P_gas + P_rad
        beta = P_gas / P_total
        chi_rho_flat = beta       # ideal gas: chi_rho = beta
        chi_T_flat = 4.0 - 3.0 * beta  # ideal gas: chi_T = 4 - 3*beta
        # Recompute cp with the flat values
        delta_flat = chi_T_flat / jnp.maximum(chi_rho_flat, 1e-6)
        cp_flat = P_total * delta_flat / (T * rho * jnp.maximum(nad, 0.05))
        return (rho, mu, nad, S, cp_flat, chi_rho_flat, chi_T_flat)

    mp.setattr(structure, "eos_lookup", _flat_chi_eos, raising=True)
    # Also patch microphysics.eos for direct imports
    import stellar_jax.microphysics.eos
    mp.setattr(microphysics.eos, "eos_lookup", _flat_chi_eos, raising=True)


# ─── disable_structural_rejection: raise hard limits to infinity ───
# The structural rejection (MESA delta_lgL_limit analog, timestep.f90:1961-1987)
# rejects steps with |ΔlogL| > 0.20 dex or |ΔlogTeff| > 0.05 dex. This mutation
# effectively disables those limits by setting them to 100.0 dex — far larger than
# any physical step could produce. When varcontrol is also relaxed (large target),
# the evolution can produce arbitrarily large structural jumps per step, which the
# test must catch.
# Ref: MESA controls.defaults:10674 (delta_lgL_limit); timestep.f90:732-766 (check_change).
@register("disable_structural_rejection")
def _disable_structural_rejection(mp, stellar_mod=None):
    """Disable structural hard limits — allows arbitrarily large ΔlogL/ΔlogTeff steps."""
    import stellar_jax.evolution._core as _core_mod
    import stellar_jax.evolution as _evo_pkg
    mp.setattr(_core_mod, "DELTA_LGL_HARD_LIMIT", 100.0, raising=True)
    mp.setattr(_core_mod, "DELTA_LGTE_HARD_LIMIT", 100.0, raising=True)
    # Also patch the evolution package namespace (the __init__.py setattr loop
    # copies values, not references — the test reads from the package).
    mp.setattr(_evo_pkg, "DELTA_LGL_HARD_LIMIT", 100.0, raising=True)
    mp.setattr(_evo_pkg, "DELTA_LGTE_HARD_LIMIT", 100.0, raising=True)


# ─── corrupt_concentration_gradient: zero AX coefficients → no concentration diffusion ───
# The concentration-gradient term (Σ AX(i,j)·dlnC_j/dr) provides negative feedback
# on gravitational settling: as a composition gradient develops, the AX term opposes
# further settling, establishing diffusive equilibrium. Zeroing AX removes this
# feedback, changing the settling rate significantly.
# Ref: TBL 1994, ApJ 421, 828, eq. 21; MESA diffusion_support.f90:700-704.
@register("corrupt_concentration_gradient")
def _corrupt_concentration_gradient(mp, stellar_mod):
    """Zero AX coefficients in the 3-species Burgers solver — removes concentration diffusion."""
    import stellar_jax.evolution as evolution
    _orig = evolution._thoul_burgers_coefficients

    def _broken(X, Y, T, rho):
        AP_He, AT_He, AX_He_H, AX_He_He = _orig(X, Y, T, rho)
        # Zero the AX coefficients → no concentration-gradient flux
        return AP_He, AT_He, AX_He_H * 0.0, AX_He_He * 0.0

    mp.setattr(evolution, "_thoul_burgers_coefficients", _broken, raising=True)


@register("n14_feedback_zero")
def _n14_feedback_zero(mp, stellar_mod=None):
    """Zero the N14 feedback by making epsilon_nuclear ignore X_N14.

    This forces the equilibrium fallback (X_CNO = 0.69*Z) regardless of the
    tracked N14 value. A test that compares tracked-N14 eps vs equilibrium eps
    will see no difference under this mutation → FAIL (as required).
    """
    import stellar_jax.microphysics.nuclear as nuc_mod
    import stellar_jax.solver.residual
    import stellar_jax.solver.jacobian
    import stellar_jax.solver.shell_data
    _orig = nuc_mod.epsilon_nuclear

    def _no_n14_eps(rho, T, X, Z, t_age=1e9, X_N14=None):
        # Ignore X_N14, always use equilibrium
        return _orig(rho, T, X, Z, t_age, X_N14=None)

    mp.setattr(nuc_mod, "epsilon_nuclear", _no_n14_eps, raising=True)
    mp.setattr(solver.residual, "epsilon_nuclear", _no_n14_eps, raising=True)
    mp.setattr(solver.jacobian, "epsilon_nuclear", _no_n14_eps, raising=True)
    mp.setattr(solver.shell_data, "epsilon_nuclear", _no_n14_eps, raising=True)


@register("burn_cno_stop_gradient_x")
def _burn_cno_stop_gradient_x(mp, stellar_mod=None):
    """Reintroduce stop_gradient on X_profile in burn_cno's rate computation.

    Issue #393 removed the stop_gradient on X_profile in the CN equilibration
    rate (rate_cn ∝ ρ * X_H * ...). Without it, the gradient ∂N14/∂X_H flows
    through the rate → relaxation factor → N14_out. This mutation blocks that
    path by detaching X_profile from the rate, making AD return ∂N14/∂X_H = 0
    while FD still sees the real sensitivity.

    A test that checks AD ≈ FD for ∂N14/∂X_H through burn_cno will FAIL under
    this mutation (AD=0, FD≠0) — confirming the test is sensitive to the
    stop_gradient being present vs absent.

    Physics: the X_H → rate_cn path is real (MESA net_approx21.f90:1116 uses
    y(ih1) in the rate). Detaching it is the specific physics break this
    mutation simulates.
    """
    import jax
    import stellar_jax.evolution as evolution
    _orig_burn = evolution.burn_cno

    def _burn_cno_detached_x(C12_profile, C13_profile, N14_profile, X_profile,
                             shell_data, dt):
        # Detach X_profile from the gradient tape before passing to burn_cno.
        # Forward: unchanged (uses real X values).
        # Backward: ∂(output)/∂X_profile = 0 through the rate path.
        X_detached = jax.lax.stop_gradient(X_profile)
        return _orig_burn(C12_profile, C13_profile, N14_profile, X_detached,
                          shell_data, dt)

    mp.setattr(evolution, "burn_cno", _burn_cno_detached_x, raising=True)


@register("disable_comp_equidistribution")
def _disable_comp_equidistribution(mp, stellar_mod=None):
    """Disable composition mesh equidistribution by making it return a uniform grid.

    When equidistribution is disabled, the mesh stays uniform (equal spacing)
    regardless of the composition gradient. A test that asserts zone concentration
    at a sharp gradient will FAIL because no concentration occurs.

    This mutation replaces equidistribute_comp_mesh (JAX/lax.scan path) AND
    equidistribute_mesh_numpy (adaptive_forward python outer loop) with versions
    that return uniform grids, disabling the MESA log-gval-driven adaptation.

    TWO mechanisms for robustness (the monkeypatch alone failed on CI due to
    an undiagnosed import/cache issue — the env-var persists across reimports):
      1. monkeypatch.setattr on both functions (standard mechanism)
      2. env-var _STELLAR_MUTATION_COMP_EQUIDIST_DISABLED=1 (checked by the
         function itself — survives reimport/subprocess/cache boundaries)
    """
    # Env-var signal: the function itself checks this and returns uniform.
    # This is the ROBUST mechanism — cannot be defeated by import caching.
    mp.setenv("_STELLAR_MUTATION_COMP_EQUIDIST_DISABLED", "1")

    import stellar_jax.mesh.composition_mesh as comp_mesh_mod
    import jax.numpy as jnp

    def _uniform_grid(X_prof, comp_mfracs):
        # Return uniform spacing instead of equidistributed
        N = X_prof.shape[0]
        return jnp.linspace(0.0, 1.0, N)

    mp.setattr(comp_mesh_mod, "equidistribute_comp_mesh", _uniform_grid, raising=True)

    # Also patch the numpy version used by evolution/adaptive/mesh_numpy.py
    import stellar_jax.evolution.adaptive.mesh_numpy as adaptive_mesh
    import numpy as np

    def _uniform_grid_numpy(gval_old, q_old, N_new, min_dq=1e-5):
        # Return uniform spacing — no equidistribution
        return np.linspace(0.0, 1.0, N_new)

    mp.setattr(adaptive_mesh, "equidistribute_mesh_numpy", _uniform_grid_numpy, raising=True)


# ─── disable_eps_grav_time_centering: revert to θ=1 (no time-centering,) ───
# Patches _eps_grav_form_c to IGNORE cp_start/nad_start — always returns the θ=1
# (end-of-step only) result regardless of arguments. This ensures the mutation is
# visible even when a test explicitly passes cp_start/nad_start to validate the
# formula. Without time-centering, eps_grav uses ONLY end-of-step thermo quantities.
# Ref: MESA eps_grav.f90:130 (use_time_centered_eps_grav), line 145-175.
@register("disable_eps_grav_time_centering")
def _disable_eps_grav_time_centering(mp, stellar_mod):
    """Disable eps_grav time-centering — patches _eps_grav_form_c to ignore start args."""
    import stellar_jax.solver.eps_grav as eps_grav_mod

    _original = eps_grav_mod._eps_grav_form_c

    def _no_time_centering(T_mid, P_mid, cp, nad, ln_T_prev_mid, ln_P_prev_mid, inv_dt,
                           cp_start=None, nad_start=None):
        # Always call original WITHOUT cp_start/nad_start → forces θ=1 path
        return _original(T_mid, P_mid, cp, nad, ln_T_prev_mid, ln_P_prev_mid, inv_dt)

    mp.setattr(eps_grav_mod, "_eps_grav_form_c", _no_time_centering)
    mp.setattr(eps_grav_mod, "EPS_GRAV_TIME_CENTERED", False, raising=True)



# ─── disable_eps_grav_composition: zero the composition eps_grav term ───
# Patches _eps_grav_composition to always return 0, simulating a missing
# composition term in eps_grav. Tests validating the composition contribution
# (e.g. comparing eps_grav_composition at the H-shell against MESA's expected
# value) must FAIL under this mutation.
# Ref: MESA eps_grav.f90:187-189 (include_composition_in_eps_grav), line 262
#      (eval_eps_grav_composition).
@register("disable_eps_grav_composition")
def _disable_eps_grav_composition(mp, stellar_mod):
    """Disable eps_grav composition term — patches _eps_grav_composition to return 0."""
    import stellar_jax.solver.eps_grav as eps_grav_mod

    def _zero_composition(T_mid, X_mid, X_prev_mid, Z, inv_dt):
        import jax.numpy as jnp
        return jnp.float64(0.0)

    mp.setattr(eps_grav_mod, "_eps_grav_composition", _zero_composition)

# ─── disable_forward_eps_grav: disable eps_grav in the forward Henyey solve ───
# Reverts evolution.py to the pre- state (inv_dt_h = 0, eps_grav ≡ 0 in the
# structure equations). Without eps_grav, at core H-exhaustion eps_nuc→0 and the
# Newton solve drives L→0 (logL→−63.58 collapse). The test must FAIL under this
# mutation by detecting the collapse (logL < −10 on a post-TAMS step).
# Ref: Paxton+2011 eq. 12, MESA eps_grav.f90:do_std_eps_grav.
@register("disable_forward_eps_grav")
def _disable_forward_eps_grav(mp, stellar_mod):
    """Disable eps_grav in the forward Henyey solve — restores the logL collapse."""
    import stellar_jax.evolution._core as _core_mod
    mp.setattr(_core_mod, "EPS_GRAV_IN_STRUCTURE", False, raising=True)


# ─── detach_eps_grav_carry: sever cross-step eps_grav gradient ───
# Severs the cross-step gradient through the gravothermal energy history terms
# (ln_T_prev_h, ln_P_prev_h from y_henyey_prev). The forward solve is UNCHANGED
# (eps_grav still contributes to L), but the backward pass no longer accumulates
# ∂L/∂(state_{k-1}) through eps_grav across steps. This degrades the post-MS
# gradient where eps_grav dominates (RGB).
# Ref: MESA eps_grav.f90 — lnT_start is a constant in the Newton solve; our
# adjoint propagates ∂F/∂(ln_T_prev) back through the carry when live.
@register("detach_eps_grav_carry")
def _detach_eps_grav_carry(mp, stellar_mod):
    """Detach the cross-step eps_grav history terms — biases post-MS gradients."""
    import stellar_jax.evolution._core as _core
    mp.setattr(_core, "DETACH_EPS_GRAV_CARRY", True, raising=True)


# ─── disable_xh_cntr_limiter: remove central-abundance timestep limiter ───
# Sets _DELTA_XH_CNTR_LIMIT and _DELTA_XH_CNTR_HARD_LIMIT to very large values,
# effectively disabling the limiter. Without it, the adaptive dt grows unbounded
# as Xc→0, causing either dt→nan (2.0 M☉) or a Xc-bounce from over-large steps
# breaking Newton convergence (1.0 M☉). The test must FAIL under this mutation
# by detecting non-monotonic Xc or dt=nan.
# Ref: MESA timestep.f90:check_XH_cntr (line 1740), controls.defaults:11185.
@register("disable_xh_cntr_limiter")
def _disable_xh_cntr_limiter(mp, stellar_mod):
    """Remove the central-abundance timestep limiter — restores unbounded dt at exhaustion."""
    import stellar_jax.evolution._core as _core_mod
    mp.setattr(_core_mod, "_DELTA_XH_CNTR_LIMIT", 1.0, raising=True)
    mp.setattr(_core_mod, "_DELTA_XH_CNTR_HARD_LIMIT", 1.0, raising=True)


# ─── opacity_linear_z_only: revert to MESA-default linear Z interpolation ───
# Replaces _interp4d_logkappa_bicubic with _interp4d_logkappa so that both the
# dispatch path (via opal_kappa) AND direct calls return the linear result.
# The test_alpha_mlt_opacity_attribution_quadratic_z test calls both functions
# directly and measures their difference. Under this mutation, _bicubic returns
# the same as _linear → Δlogκ ≈ 0 → the assertion Δlogκ > 0.003 FAILS.
# Ref: Rogers & Iglesias (1996); MESA kap.defaults:268.
@register("opacity_linear_z_only")
def _opacity_linear_z_only(mp, stellar_mod=None):
    """Replace bicubic opacity with linear — removes the quadratic Z correction."""
    from stellar_jax.microphysics import opacity as opacity_mod
    # Make the bicubic function return the linear result
    mp.setattr(opacity_mod, "_interp4d_logkappa_bicubic",
               opacity_mod._interp4d_logkappa, raising=True)
    # The dispatch flag is now a parameter (bicubic_opacity) on opal_kappa,
    # not a module global. The function swap above is sufficient to break
    # the bicubic path even when bicubic_opacity=True is passed.


# ─── opacity_quadrilinear_xz: revert 4D Hermite to quadrilinear ────────
# The 4D Steffen-Hermite interpolation guarantees C1-continuous derivatives at
# grid points, eliminating the VJP noise that corrupts ∂logL/∂M at the 1.5 M☉
# convective-core boundary. Under this mutation, opal_kappa uses the old
# quadrilinear (with jnp.clip fractions), which has C0-discontinuous derivatives
# at every grid point. The test asserting AD-vs-FD agreement < 1% at X=0.7
# must FAIL because the quadrilinear's derivative disagrees with its own FD by
# 37-56% near grid boundaries.
# Ref: Steffen M. (1990), A&A 239, 443;
#      MESA kap/private/kap_eval_fixed.f90:475 (Get_Kap_for_X_cubic)
@register("opacity_quadrilinear_xz")
def _opacity_quadrilinear_xz(mp, stellar_mod=None):
    """Revert opal_kappa to quadrilinear (C0-discontinuous derivatives)."""
    from stellar_jax.microphysics import opacity as opacity_mod

    def _quadrilinear_opal(logT, logRho, X, Z):
        """Quadrilinear OPAL with clip fractions (the broken path)."""
        logR = logRho - 3.0 * logT + 18.0
        return opacity_mod._interp4d_logkappa(
            opacity_mod.OPAL_X, opacity_mod.OPAL_Z,
            opacity_mod.OPAL_LOGT, opacity_mod.OPAL_LOGR, opacity_mod.OPAL_LK,
            logT, logR, X, Z)

    mp.setattr(opacity_mod, "opal_kappa", _quadrilinear_opal, raising=True)

# ─── non_conservative_zone_mass: revert to the legacy non-conservative weights ───
# The legacy `concat([diff, diff[-1:]])` sums to ~1.015, not 1.0 — it duplicates
# the surface zone's mass contribution. Under this mutation, the test asserting
# sum(zone_masses)==1.0 and correct boundary half-widths must FAIL.
# The mutation patches COMP_ZONE_MASSES to the legacy non-conservative values
# AND patches compute_zone_masses to return them, so both the static and dynamic
# paths are broken.
# Ref: MESA star_utils.f90:672-713 (normalize_dqs) — correct weights sum to 1.
@register("non_conservative_zone_mass")
def _non_conservative_zone_mass(mp, stellar_mod):
    """Revert zone masses to the legacy non-conservative concat([diff, diff[-1:]]) form."""
    import numpy as np
    import jax.numpy as jnp
    import stellar_jax.config.mesh_defaults as constants
    import stellar_jax.evolution as evolution

    # Legacy weights: diff of node positions, with last diff duplicated
    legacy_dm = np.diff(constants.COMP_MFRACS)
    legacy_weights = np.concatenate([legacy_dm, legacy_dm[-1:]])
    # Sum is ~1.015, not 1.0

    import stellar_jax.composition.mix as comp_mix  # runtime usage site (from-import binding)
    mp.setattr(constants, "COMP_ZONE_MASSES", legacy_weights, raising=True)
    mp.setattr(comp_mix, "COMP_ZONE_MASSES", legacy_weights, raising=True)

    def _legacy_compute(comp_mfracs):
        dm = jnp.diff(comp_mfracs)
        return jnp.concatenate([dm, dm[-1:]])

    mp.setattr(constants, "compute_zone_masses", _legacy_compute, raising=True)
    mp.setattr(comp_mix, "compute_zone_masses", _legacy_compute, raising=True)


# ─── disable_chugunov_screening: zero out screening entirely ───
# Returns f=1.0 (no screening enhancement) for all reactions.
# test_chugunov_screening_regime_aware asserts f_pp > 1.01 and f_cno > 1.03,
# so zero screening (f=1.0) cleanly breaks both tests using this mutation.
# NOTE: a Salpeter replacement was tried but its values (~1.03 pp, ~1.24 CNO)
# still satisfy the test's generous bounds — only full disabling is unambiguous.
# Ref: Chugunov+ (2007), PhRvD 76, 025028 (the formula being disabled).
@register("disable_chugunov_screening")
def _disable_chugunov_screening(mp, stellar_mod=None):
    """Disable all screening (return f=1.0) — breaks any test asserting f > 1."""
    import stellar_jax.microphysics.nuclear as nuc_mod

    def _no_screening(z1, z2, a1, a2, rho, T, zbar, abar):
        """No screening: f = 1.0 always."""
        return 1.0

    mp.setattr(nuc_mod, "screen_chugunov", _no_screening, raising=True)


# ─── env_cz_boundary_hard_step: revert envelope CZ boundary to hard step ───
# The smooth envelope mixing uses sigmoid(_K_env * delta_nab) for
# zone membership, giving non-zero AD gradient through the boundary position.
# This mutation replaces the smooth _K_env=30 sigmoid with _K_env=1e6 (effectively
# a hard step), then stop_gradients the result. AD sees zero derivative through
# the boundary; FD correctly measures the discrete jump. This is the exact
# failure mode of the pre-fix code (zone_labels == zid).
# Ref: Bengio et al. (2013), arXiv:1308.3432 (smooth relaxation of discrete ops).
@register("env_cz_boundary_hard_step")
def _env_cz_boundary_hard_step(mp, stellar_mod=None):
    """Replace smooth envelope CZ indicator with a hard step — kills AD gradient."""
    import jax
    import jax.numpy as jnp
    import stellar_jax.evolution as evolution

    _orig_mix = evolution.mix_composition

    def _hardened_mix(X_profile, shell_data, M_solar=1.0, f_ov=None, comp_mfracs_in=None):
        """Wrap mix_composition: use a hard step for env_conv (no gradient)."""
        from stellar_jax.config.mesh_defaults import COMP_MFRACS, F_OV, COMP_ZONE_MASSES, compute_zone_masses
        import jax.numpy as jnp

        if f_ov is None:
            f_ov = F_OV

        _K = 200.0
        nrad = shell_data[:, 2]
        nad = shell_data[:, 3]
        mf_shells = shell_data[:, 1]
        hp_over_R = shell_data[:, 4]
        dm_dr_norm = shell_data[:, 5]

        is_conv = jax.nn.sigmoid(_K * (nrad - nad))
        comp_mfracs = comp_mfracs_in if comp_mfracs_in is not None else jnp.array(COMP_MFRACS)
        mf_sorted = mf_shells[::-1]
        conv_sorted = is_conv[::-1]
        hp_sorted = hp_over_R[::-1]
        dmdr_sorted = dm_dr_norm[::-1]

        is_conv_comp = jnp.interp(comp_mfracs, mf_sorted, conv_sorted,
                                  left=conv_sorted[0], right=conv_sorted[-1])
        hp_comp = jnp.interp(comp_mfracs, mf_sorted, hp_sorted,
                             left=hp_sorted[0], right=hp_sorted[-1])
        dmdr_comp = jnp.interp(comp_mfracs, mf_sorted, dmdr_sorted,
                               left=dmdr_sorted[0], right=dmdr_sorted[-1])

        # Core logic (unchanged from original)
        log_conv = jnp.log(jnp.clip(is_conv_comp, 1e-10, 1.0))
        soft_core_mask = jnp.clip(jnp.exp(jnp.cumsum(log_conv)), 0.0, 1.0)
        hard_core_mask = (soft_core_mask > 0.5).astype(jnp.float64)
        core_conv_mask = jax.lax.stop_gradient(hard_core_mask - soft_core_mask) + soft_core_mask

        n_comp = comp_mfracs.shape[0]
        n_core_conv = jnp.sum(soft_core_mask)
        bdy_pos = jnp.clip(n_core_conv - 1.0, 0.0, n_comp - 1.0)
        indices = jnp.arange(n_comp, dtype=jnp.float64)
        bdy_weights = jnp.exp(-0.5 * ((indices - bdy_pos) / 0.5) ** 2)
        bdy_weights = bdy_weights / (jnp.sum(bdy_weights) + 1e-30)

        hp_at_bdy = jnp.sum(bdy_weights * hp_comp)
        dmdr_at_bdy = jnp.sum(bdy_weights * dmdr_comp)
        m_bdy = jnp.sum(bdy_weights * comp_mfracs)

        f_ov_effective = f_ov / (1.0 + jnp.exp(-(M_solar - 1.2) / 0.1))
        delta_m_ov = f_ov_effective * hp_at_bdy * dmdr_at_bdy

        soft_ov_mask = (jax.nn.sigmoid(_K * (comp_mfracs - m_bdy)) *
                        jax.nn.sigmoid(_K * (m_bdy + delta_m_ov - comp_mfracs)))
        has_core = jax.nn.sigmoid(_K * (n_core_conv - 1.0))
        soft_ov_mask = soft_ov_mask * has_core
        hard_ov_mask = (soft_ov_mask > 0.5).astype(jnp.float64)

        soft_combined = jnp.clip(jnp.maximum(soft_core_mask, soft_ov_mask), 0.0, 1.0)
        hard_combined = jnp.maximum(hard_core_mask, hard_ov_mask)
        core_mix_mask = jax.lax.stop_gradient(hard_combined - soft_combined) + soft_combined

        zone_weights = compute_zone_masses(comp_mfracs) if comp_mfracs_in is not None else jnp.array(COMP_ZONE_MASSES)

        core_mass_total = jnp.sum(core_mix_mask * zone_weights) + 1e-30
        core_avg_X = jnp.sum(core_mix_mask * X_profile * zone_weights) / core_mass_total
        X_out = core_mix_mask * core_avg_X + (1.0 - core_mix_mask) * X_profile

        # MUTATION: envelope CZ uses HARD step (stop_gradient) instead of smooth sigmoid
        non_core = 1.0 - core_mix_mask
        delta_nab_sorted = (nrad - nad)[::-1]
        delta_nab_comp = jnp.interp(comp_mfracs, mf_sorted, delta_nab_sorted,
                                    left=delta_nab_sorted[0], right=delta_nab_sorted[-1])
        # Hard step: no gradient through the boundary position
        env_conv_soft = jax.nn.sigmoid(30.0 * delta_nab_comp) * non_core
        env_conv = jax.lax.stop_gradient((env_conv_soft > 0.5).astype(jnp.float64))

        # Apply hard homogenization (no smooth zone_avg, just hard mask averaging)
        zone_mass_conv = env_conv * zone_weights
        total_conv_mass = jnp.sum(zone_mass_conv) + 1e-30
        env_avg_X = jnp.sum(env_conv * X_out * zone_weights) / total_conv_mass
        X_out = env_conv * env_avg_X + (1.0 - env_conv) * X_out

        return X_out

    mp.setattr(evolution, "mix_composition", _hardened_mix, raising=True)
    # Also patch on the stellar facade (tests import via `stellar.mix_composition`)
    if stellar_mod is not None:
        mp.setattr(stellar_mod, "mix_composition", _hardened_mix, raising=True)


# ─── opacity_factor_detach: sever the opacity_factor → kappa gradient ───
# The opacity_factor enters kappa() as: log10(kappa_total * opacity_factor)
# = log_kappa_total + log10(opacity_factor). If we stop_gradient the
# opacity_factor inside kappa, the forward result is unchanged but the
# backward ∂kappa/∂opacity_factor = 0 → ∂ν/∂opacity_factor ≈ 0.
# Any test that asserts a nonzero ∂ν/∂opacity_factor gradient will FAIL.
# MESA ref: micro.f90:622 (opacity_factor applied multiplicatively).
@register("opacity_factor_detach")
def _opacity_factor_detach(mp, stellar_mod=None):
    """Detach opacity_factor from the gradient tape — ∂kappa/∂opf = 0.

    Wraps kappa() to apply stop_gradient on opacity_factor before the log10.
    The forward value is unchanged (opacity_factor still multiplies kappa),
    but the AD gradient ∂kappa/∂opacity_factor = 0.

    This breaks any test that asserts a nonzero gradient w.r.t. opacity_factor.
    """
    import jax
    import jax.numpy as jnp
    import stellar_jax.solver.residual
    import stellar_jax.solver.surface_bc
    import stellar_jax.solver.shell_data
    from stellar_jax.microphysics import opacity as opacity_mod
    _orig_kappa = opacity_mod.kappa

    def _detached_kappa(logT, logRho, X, Z, opacity_factor=None):
        # Apply stop_gradient to opacity_factor: forward is unchanged, backward is zero
        if opacity_factor is not None:
            opacity_factor = jax.lax.stop_gradient(opacity_factor)
        return _orig_kappa(logT, logRho, X, Z, opacity_factor=opacity_factor)

    mp.setattr(opacity_mod, "kappa", _detached_kappa, raising=True)
    # Patch at all solver usage sites (each holds its own import binding)
    mp.setattr(solver.residual, "kappa", _detached_kappa, raising=True)
    mp.setattr(solver.surface_bc, "kappa", _detached_kappa, raising=True)
    mp.setattr(solver.shell_data, "kappa", _detached_kappa, raising=True)


# ─── eps_nuc_factor_detach: sever the eps_nuc_factor → epsilon_nuclear gradient ───
# The eps_nuc_factor enters epsilon_nuclear() as: eps_total * eps_nuc_factor.
# If we stop_gradient the eps_nuc_factor inside epsilon_nuclear, the forward
# result is unchanged but the backward ∂eps/∂eps_nuc_factor = 0 → any downstream
# gradient w.r.t. eps_nuc_factor ≈ 0.
# MESA ref: star/private/net.f90:365-372 (eps_nuc_factor applied multiplicatively).
@register("eps_nuc_factor_detach")
def _eps_nuc_factor_detach(mp, stellar_mod=None):
    """Detach eps_nuc_factor from the gradient tape — ∂eps/∂enf = 0.

    The solver path (solver/residual.py:_cell_residual) calls epsilon_nuclear
    WITHOUT eps_nuc_factor, then multiplies externally:
        eps = epsilon_nuclear(rho, T, X, Z, t_age, X_N14=...)
        eps = eps * eps_nuc_factor   # <-- gradient flows HERE
    So patching epsilon_nuclear alone does NOT sever the gradient in the solver.

    This mutation patches at TWO levels:
    1. epsilon_nuclear itself (for direct callers like the @fast test)
    2. _cell_residual (wraps it to stop_gradient eps_nuc_factor before the
       external multiplication, severing the Henyey solver gradient path)

    This breaks any test that asserts a nonzero gradient w.r.t. eps_nuc_factor.
    """
    import jax
    import jax.numpy as jnp
    import stellar_jax.solver.residual
    import stellar_jax.solver.jacobian
    import stellar_jax.solver.shell_data
    from stellar_jax.microphysics import nuclear as nuclear_mod

    _orig_eps_nuc = nuclear_mod.epsilon_nuclear

    def _detached_eps_nuc(rho, T, X, Z, *args, **kwargs):
        # Apply stop_gradient to eps_nuc_factor: forward is unchanged, backward is zero.
        # *args absorbs positional overflow (t_age is passed as 5th positional by
        # structure.py:291, solver/shell_data.py:94, solver/residual.py, solver/jacobian.py).
        # **kwargs absorbs keyword args (X3, X3_eq_frozen, etc.) —.
        eps_nuc_factor = kwargs.pop("eps_nuc_factor", None)
        if eps_nuc_factor is not None:
            eps_nuc_factor = jax.lax.stop_gradient(eps_nuc_factor)
        return _orig_eps_nuc(rho, T, X, Z, *args, eps_nuc_factor=eps_nuc_factor, **kwargs)

    mp.setattr(nuclear_mod, "epsilon_nuclear", _detached_eps_nuc, raising=True)
    # Patch at all solver usage sites (each holds its own import binding)
    mp.setattr(solver.residual, "epsilon_nuclear", _detached_eps_nuc, raising=True)
    mp.setattr(solver.jacobian, "epsilon_nuclear", _detached_eps_nuc, raising=True)
    mp.setattr(solver.shell_data, "epsilon_nuclear", _detached_eps_nuc, raising=True)

    # Patch _cell_residual to detach eps_nuc_factor at the EXTERNAL multiplication
    # site (solver/residual.py lines 165-166). The solver calls epsilon_nuclear
    # without eps_nuc_factor kwarg, then does `eps = eps * eps_nuc_factor` — so
    # the epsilon_nuclear patch above has no effect on the solver gradient path.
    _orig_cell_residual = solver.residual._cell_residual

    def _detached_cell_residual(y_k, y_kp1, dm, m_mid, M_star, X_mid, Z, alpha_mlt,
                                *args, **kwargs):
        # Detach eps_nuc_factor before passing to the real _cell_residual.
        # *args absorbs positional overflow (ln_T_prev_mid, ln_P_prev_mid, inv_dt
        # are passed as 9th/10th/11th positional args by solver/jacobian.py).
        # **kwargs absorbs keyword args (X_prev_mid, X3_mid, etc.) —.
        eps_nuc_factor = kwargs.pop("eps_nuc_factor", None)
        if eps_nuc_factor is not None:
            eps_nuc_factor = jax.lax.stop_gradient(eps_nuc_factor)
        return _orig_cell_residual(y_k, y_kp1, dm, m_mid, M_star, X_mid, Z, alpha_mlt,
                                   *args, eps_nuc_factor=eps_nuc_factor, **kwargs)

    mp.setattr(solver.residual, "_cell_residual", _detached_cell_residual, raising=True)

    # Also patch _cell_residual_raw (used by jacfwd in the Jacobian computation).
    # Same external multiplication pattern: epsilon_nuclear() without eps_nuc_factor,
    # then `eps = eps * eps_nuc_factor`.
    _orig_cell_residual_raw = solver.residual._cell_residual_raw

    def _detached_cell_residual_raw(y_k, y_kp1, dm, m_mid, M_star, X_mid, Z, alpha_mlt,
                                    *args, **kwargs):
        # *args absorbs positional overflow (ln_T_prev_mid, ln_P_prev_mid, inv_dt).
        # **kwargs absorbs keyword args (X_prev_mid, X3_mid, etc.) —.
        eps_nuc_factor = kwargs.pop("eps_nuc_factor", None)
        if eps_nuc_factor is not None:
            eps_nuc_factor = jax.lax.stop_gradient(eps_nuc_factor)
        return _orig_cell_residual_raw(y_k, y_kp1, dm, m_mid, M_star, X_mid, Z, alpha_mlt,
                                       *args, eps_nuc_factor=eps_nuc_factor, **kwargs)

    mp.setattr(solver.residual, "_cell_residual_raw", _detached_cell_residual_raw, raising=True)

    # Patch solver.jacobian's LOCAL bindings of _cell_residual and _cell_residual_raw.
    # solver/jacobian.py does `from solver.residual import _cell_residual` which creates
    # a LOCAL binding that is NOT affected by setattr on the source module. The Jacobian
    # computation feeds into the IFT backward pass, so if the jacobian's _cell_residual
    # is unpatched, the gradient through the Jacobian still flows w.r.t. eps_nuc_factor.
    mp.setattr(solver.jacobian, "_cell_residual", _detached_cell_residual, raising=True)
    mp.setattr(solver.jacobian, "_cell_residual_raw", _detached_cell_residual_raw, raising=True)

    # Patch _build_residual_fixed_bc to detach eps_nuc_factor at the CENTER BC
    # external multiplication (solver/residual.py lines 326-327):
    #   eps_c = epsilon_nuclear(rho_c, T_c, ...)
    #   eps_c = eps_c * eps_nuc_factor
    # This path is NOT inside _cell_residual — it's in the center luminosity BC
    # (BC2 = ell_0 - eps_c * m1 / Lsun). Without this patch, a nonzero gradient
    # leaks through BC2 even when _cell_residual is patched.
    _orig_build_residual_fixed_bc = solver.residual._build_residual_fixed_bc

    def _detached_build_residual_fixed_bc(y, q_mesh, M_star, X_profile, Z, alpha_mlt,
                                          ln_P_atm, ln_T_atm, atm_jac, y_surf_ref,
                                          *args, **kwargs):
        # Detach eps_nuc_factor before it reaches the center BC multiplication.
        # *args, **kwargs absorb current and future params (X_prev_profile, X3_profile, etc.) —.
        eps_nuc_factor = kwargs.pop("eps_nuc_factor", None)
        if eps_nuc_factor is not None:
            eps_nuc_factor = jax.lax.stop_gradient(eps_nuc_factor)
        return _orig_build_residual_fixed_bc(
            y, q_mesh, M_star, X_profile, Z, alpha_mlt,
            ln_P_atm, ln_T_atm, atm_jac, y_surf_ref,
            *args, eps_nuc_factor=eps_nuc_factor, **kwargs)

    mp.setattr(solver.residual, "_build_residual_fixed_bc", _detached_build_residual_fixed_bc, raising=True)

    # Patch solver.continuation's LOCAL binding of _build_residual_fixed_bc.
    # solver/continuation.py does `from solver.residual import _build_residual_fixed_bc`
    # which creates a LOCAL binding unaffected by setattr on solver.residual. The IFT
    # backward pass (custom_vjp _bwd) calls _build_residual_fixed_bc through this local
    # binding — if unpatched, the center BC's `eps_c * eps_nuc_factor` multiplication
    # leaks gradient through the backward pass.
    import stellar_jax.solver.continuation
    mp.setattr(solver.continuation, "_build_residual_fixed_bc", _detached_build_residual_fixed_bc, raising=True)


# ─── diffusion_factor_detach: sever the diffusion_factor → composition gradient ───
# The diffusion_factor enters the scan step body as a multiplicative scale on the
# diffusion composition delta: X_diff = X_mixed + diffusion_factor * (X_diffused - X_mixed).
# If we stop_gradient the diffusion_factor, the forward result is unchanged but the
# backward ∂X_diff/∂diffusion_factor = 0 → ∂(observable)/∂diffusion_factor ≈ 0.
# Any test that asserts a nonzero ∂(observable)/∂diffusion_factor gradient will FAIL.
# MESA ref: diffusion_support.f90:770 (SIG_factor applied multiplicatively to c).
# ─── eos_density_scale: +8% density error in microphysics.eos — breaks EOS-vs-MESA test ───
# The test_eos_vs_mesa_ms_fgong test imports eos_lookup directly from microphysics.eos,
# evaluates it at MESA FGONG (T, P_gas, X, Z), and asserts density matches MESA to <1-3%.
# A 1.08× (+8%, 0.033 dex) density inflation exceeds the parity tolerances while staying
# inside the 0.1-dex round-trip validity window. The old 2× (0.3 dex) scaling pushed
# round-trip error to 0.3 dex, causing the validity filter to reject most points before
# the parity assertion could fire — over-destructive.
# Ref: Rogers & Nayfonov (2002), ApJ 576, 1064 — OPAL EOS determines ρ(T,P,X,Z).
@register("eos_density_scale")
def _eos_density_scale(mp, stellar_mod=None):
    """Scale EOS density by 1.08× (+0.033 dex) — breaks density-vs-MESA parity.

    Patches eos_lookup at the microphysics.eos source module (where the
    test imports from) to return density * 1.08. The +8% density error
    exceeds the 0.8-3% parity tolerances while staying inside the 0.1-dex
    round-trip validity window (0.033 dex < 0.1 dex). This prevents the
    test's domain filter from rejecting most points (the old 2× / 0.3 dex
    scaling pushed round-trip error to 0.3 dex, causing the validity filter
    to reject 2382/2400 points before the parity assertion could fire).
    """
    from stellar_jax.microphysics import eos as eos_mod
    _orig_eos = eos_mod.eos_lookup

    def _scaled_eos(logT, logP, X, Z):
        rho, mu, nad, S, cp, chi_rho, chi_T = _orig_eos(logT, logP, X, Z)
        return (rho * 1.08, mu, nad, S, cp, chi_rho, chi_T)

    mp.setattr(eos_mod, "eos_lookup", _scaled_eos, raising=True)


# ─── mlt_nabla_offset: add +0.1 to the MLT gradient — breaks MLT-vs-MESA test ───
# The test_mlt_nabla_vs_mesa_fgong test imports mlt_nabla from transport, evaluates
# it at each zone with FGONG-derived inputs, and asserts the gradient matches MESA's
# actual d(lnT)/d(lnP) to within 5%. A +0.1 additive offset shifts the gradient
# far beyond tolerance (MESA's nabla ~ 0.1-0.4 in convective zones).
# Ref: Böhm-Vitense (1958), ZAp 46, 108 — MLT theory; Cox & Giuli (1968).
@register("mlt_nabla_offset")
def _mlt_nabla_offset(mp, stellar_mod=None):
    """Add +0.1 to mlt_nabla output — breaks MLT gradient vs MESA comparison.

    Patches mlt_nabla at the transport module (where the test imports from).
    The +0.1 offset is large compared to the superadiabaticity (~0.001-0.01
    in the deep CZ) and shifts the total gradient well beyond the 5% tolerance.
    """
    import stellar_jax.transport as transport
    import stellar_jax.transport.mlt as mlt_mod
    _orig_mlt = mlt_mod.mlt_nabla

    def _offset_mlt(*args, **kwargs):
        result = _orig_mlt(*args, **kwargs)
        return result + 0.1

    mp.setattr(mlt_mod, "mlt_nabla", _offset_mlt, raising=True)
    mp.setattr(transport, "mlt_nabla", _offset_mlt, raising=True)


# ─── eps_nuc_scale_5x: multiply nuclear energy by 5× — breaks eps-vs-MESA test ───
# The test_nuclear_eps_vs_mesa_fgong test calls stellar.epsilon_nuclear at each FGONG
# zone and asserts median relative error < 10%. A 5× scale gives 400% error.
# Ref: Adelberger+ (2011), Rev. Mod. Phys. 83, 195 — nuclear rate compilations.
@register("eps_nuc_scale_5x")
def _eps_nuc_scale_5x(mp, stellar_mod=None):
    """Multiply epsilon_nuclear by 5× — breaks any eps_nuc vs MESA comparison.

    Patches at microphysics.nuclear (the definition site) AND on the stellar
    facade module (where the test accesses it via the fixture).
    """
    from stellar_jax.microphysics import nuclear as nuc_mod
    _orig_eps = nuc_mod.epsilon_nuclear

    def _scaled_eps(*args, **kwargs):
        return _orig_eps(*args, **kwargs) * 5.0

    mp.setattr(nuc_mod, "epsilon_nuclear", _scaled_eps, raising=True)
    # Also patch on the stellar facade if available (test uses stellar.epsilon_nuclear)
    if stellar_mod is not None:
        mp.setattr(stellar_mod, "epsilon_nuclear", _scaled_eps, raising=True)


# ─── mix_composition_identity: disable mixing entirely — breaks convective boundary test ───
# The test_convective_boundary_vs_mesa_fgong test calls mix_composition from evolution
# and asserts that a convective core region is mixed (boundary matches Schwarzschild).
# If mix_composition returns X unchanged, no mixing occurs → dX == 0 everywhere →
# the assertion "np.any(mixed_noov)" FAILS immediately.
# Ref: Böhm-Vitense (1958); MESA mix_info.f90 — convective mixing homogenizes CZ.
@register("mix_composition_identity")
def _mix_composition_identity(mp, stellar_mod=None):
    """Make mix_composition return X unchanged — kills all convective mixing.

    Patches at the evolution module (where the test imports from).
    """
    import stellar_jax.evolution as evolution

    def _no_mix(X_profile, shell_data, M_solar=1.0, f_ov=None, comp_mfracs_in=None):
        return X_profile  # identity — no mixing

    mp.setattr(evolution, "mix_composition", _no_mix, raising=True)


# ─── burn_composition_identity: disable burning entirely — breaks burn rate test ───
# The test_burn_composition_rate_vs_mesa_fgong test calls burn_composition from
# evolution and asserts that core X after burning matches MESA's later snapshot
# (X_core depletion within 15%). If burn_composition returns X unchanged, the
# core doesn't deplete → error = dX_core/dX_core = 100% >> 15%.
# Ref: MESA struct_burn_mix.f90 — operator-split burn reduces X via eps_nuc.
@register("burn_composition_identity")
def _burn_composition_identity(mp, stellar_mod=None):
    """Make burn_composition return X,Y unchanged — kills hydrogen burning.

    Patches at the evolution module (where the test imports from via
    `from evolution import burn_composition`).
    """
    import stellar_jax.evolution as evolution
    from stellar_jax.composition import burn as burn_mod

    def _no_burn(X_profile, Y_profile, shell_data, dt):
        return X_profile, Y_profile  # identity — no burning

    mp.setattr(evolution, "burn_composition", _no_burn, raising=True)
    mp.setattr(burn_mod, "burn_composition", _no_burn, raising=True)


@register("diffusion_factor_detach")
def _diffusion_factor_detach(mp, stellar_mod=None):
    """Detach diffusion_factor from the gradient tape — ∂X_diff/∂diffusion_factor = 0.

    Wraps evolve_star to apply stop_gradient on diffusion_factor before passing it
    into the JIT function. The forward value is unchanged (diffusion_factor still
    scales the diffusion delta), but the AD gradient ∂(observable)/∂diffusion_factor = 0.

    This breaks any test that asserts a nonzero gradient w.r.t. diffusion_factor.
    """
    import jax
    import jax.numpy as jnp
    import stellar_jax.evolution as evolution

    _orig_evolve = evolution.evolve_star

    def _detached_evolve(*args, **kwargs):
        """Forward values correct, backward pass severed on diffusion_factor."""
        # Detach diffusion_factor: forward unchanged, backward zeroed
        if 'diffusion_factor' in kwargs:
            kwargs['diffusion_factor'] = jax.lax.stop_gradient(
                jnp.asarray(kwargs['diffusion_factor'], dtype=jnp.float64))
        return _orig_evolve(*args, **kwargs)

    mp.setattr(evolution, "evolve_star", _detached_evolve)
    if stellar_mod is not None:
        mp.setattr(stellar_mod, "evolve_star", _detached_evolve)





# ─── disable_remap_slopes: force order-0 (piecewise-constant) composition remap ───
# Disables the linear reconstruction slopes in BOTH the numpy (evolution/adaptive/mesh_numpy.py)
# and JAX (mesh/remap.py) conservative remap implementations, reverting to
# piecewise-constant behavior. Tests that assert order-1 accuracy (e.g. exact
# preservation of a linear profile) must FAIL under this mutation.
# MESA ref: mesh_adjust.f90:get1_lpp (line 1720) — order-1 slopes for composition.
@register("disable_remap_slopes")
def _disable_remap_slopes(mp, stellar_mod=None):
    """Zero all slopes in composition remap — reduces to piecewise-constant (order-0)."""
    import stellar_jax.evolution.adaptive.mesh_numpy as adaptive_mesh
    import numpy as np

    def _zero_slopes_numpy(profile, dq, q_nodes=None):
        """Return zero slopes (piecewise-constant reconstruction)."""
        return np.zeros(len(profile))

    mp.setattr(adaptive_mesh, "_compute_slopes_mesa", _zero_slopes_numpy, raising=True)

    # Also disable slopes in the JAX version (mesh/remap.py:_compute_slopes_jax)
    import jax.numpy as jnp
    from stellar_jax.mesh import remap as mesh_remap

    def _zero_slopes_jax(profile, dq, grid):
        """Return zero slopes (piecewise-constant reconstruction) — JAX version."""
        return jnp.zeros_like(profile)

    mp.setattr(mesh_remap, "_compute_slopes_jax", _zero_slopes_jax, raising=True)


# ─── detach_rgb_mass_in_dnu: corrupt ∂Δν/∂M for INV-3 RGB recovery ──────────
# The INV-3 test recovers an RGB star's mass + evolutionary state from
# seismic (Δν) + photometric (logL, logTeff) observables using infer_parameters
# (LM + jax.jacrev). The PRIMARY mass diagnostic on the RGB is the Δν scaling:
#   Δν ∝ (M/R³)^{1/2}  [Chaplin & Miglio 2013, arXiv:1303.1957, Eq. 1]
#
# This mutation applies jax.lax.stop_gradient(M) inside the Δν computation,
# killing ∂Δν/∂M in the Jacobian. Under this mutation:
# - The forward Δν value is UNCHANGED (forward pass still correct)
# - But the Jacobian row J[0,:] (Δν row) loses its mass sensitivity: ∂Δν/∂M → 0
# - The LM optimizer can still use ∂logL/∂M and ∂logTeff/∂M (photometry) for
#   mass, but these are DEGENERATE on the RGB (tracks of different masses
#   overlap in the HR diagram) → mass recovery degrades
# - The mutation gate assertion (|∂Δν/∂M| > 0.5) FAILS directly
#
# Implementation: the mutation is detected via STELLAR_MUTATION env var inside
# the test's forward_fn (the forward model is test-local, not a module-level
# function). This registry entry exists to satisfy the O2 gate's mutation lookup
# and to document the physical rationale.
#
# Ref: Chaplin & Miglio (2013), ARA&A 51, 353, arXiv:1303.1957, §2:
# "The large frequency separation Δν is related to the mean density of the star"
# Δν/Δν☉ ≈ (M/M☉)^{1/2} (R/R☉)^{-3/2}. Without ∂Δν/∂M, mass is degenerate.
@register("detach_rgb_mass_in_dnu")
def _detach_rgb_mass_in_dnu(mp, stellar_mod=None):
    """Apply stop_gradient(M) in the Δν scaling, killing ∂Δν/∂M.

    The forward model for the RGB inversion test is test-local (a JAX closure),
    so there is no module function to monkeypatch. Instead, this mutation sets
    a flag on the mutations module that the test reads at call time to decide
    whether to detach M. The flag is a module attribute patched via monkeypatch,
    so it is centrally auditable and reverted automatically after the test.

    Issue #824: moved the mutation signal from a raw STELLAR_MUTATION env-var
    check (registry-bypass) to a monkeypatched module flag (centrally audited).
    """
    import mutations as mut_mod
    mp.setattr(mut_mod, "DETACH_RGB_MASS_IN_DNU_ACTIVE", True, raising=False)


# ─── he3_phi_global: strip local ³He, reverting to global-age phi ───
# When X3 is passed to epsilon_nuclear, phi is computed locally from ³He
# equilibrium. This mutation strips X3 so phi falls back to the global-age
# placeholder (spatially uniform). Tests gated on this mutation MUST fail
# under mutation because their assertion depends on phi varying with local
# (T, ρ) — which only happens when X3 is in the call path.
#
# Implementation: wrap epsilon_nuclear at all solver-path sites to strip the
# X3 kwarg. The forward eps_pp uses the legacy phi=1-0.3*exp(-t/5Myr),
# making phi uniform. Any test asserting spatial phi variation will fail.
#
# Ref: Clayton (1968) §5.6,.
@register("he3_phi_global")
def _he3_phi_global(mp, stellar_mod=None):
    """Strip X3 from epsilon_nuclear → phi reverts to global-age placeholder.

    Patches ALL usage sites (microphysics.nuclear, solver.residual, solver.jacobian)
    so both unit tests (importing from microphysics.nuclear) and integration tests
    (going through the solver) see the stripped X3.
    """
    import stellar_jax.microphysics.nuclear
    import stellar_jax.solver.residual
    import stellar_jax.solver.jacobian
    _real_eps = microphysics.nuclear.epsilon_nuclear

    def _eps_no_x3(*args, **kwargs):
        kwargs.pop('X3', None)
        return _real_eps(*args, **kwargs)

    mp.setattr(microphysics.nuclear, "epsilon_nuclear", _eps_no_x3, raising=True)
    mp.setattr(solver.residual, "epsilon_nuclear", _eps_no_x3, raising=True)
    mp.setattr(solver.jacobian, "epsilon_nuclear", _eps_no_x3, raising=True)

# ─── ideal_gas_mlt: revert MLT to ideal-gas cp and Q=1 (Part A mutation) ───
# The production path (_cell_residual) passes real EOS cp and Q=chi_T/chi_rho
# to mlt_nabla. This mutation patches mlt_nabla to ignore the real cp/Q and
# use the ideal-gas fallback, breaking the Part A physics improvement.
# Usage site: solver/residual.py → mlt_nabla (from transport import mlt_nabla).
@register("ideal_gas_mlt")
def _ideal_gas_mlt(mp, stellar_mod=None):
    import stellar_jax.solver.residual
    import stellar_jax.transport as transport
    import stellar_jax.transport.mlt
    from stellar_jax.transport.mlt import _mlt_cubic_solve
    from stellar_jax.transport.switch import _mlt_switch
    from stellar_jax.config.constants import k_B, m_H
    import jax.numpy as jnp

    def _mlt_nabla_ideal(nabla_rad, nad, T, P, rho, kappa, g, mu, alpha_mlt,
                         cp=None, Q=None, gradL_composition_term=0.0):
        """MLT with forced ideal-gas cp and Q=1 (mutation for #409 Part A)."""
        # Force ideal-gas regardless of what cp/Q were passed
        Q_forced = 1.0
        cp_forced = (k_B / (mu * m_H)) / jnp.maximum(nad, 1e-2)
        grad_conv = _mlt_cubic_solve(nabla_rad, nad, T, P, rho, kappa, g, mu,
                                     alpha_mlt, cp_forced, Q_forced,
                                     gradL_composition_term)
        if isinstance(gradL_composition_term, (int, float)) and gradL_composition_term == 0:
            gradL = nad
        else:
            gradL = nad + gradL_composition_term
        nabla = _mlt_switch(nabla_rad, gradL, grad_conv)
        return nabla

    mp.setattr(transport, "mlt_nabla", _mlt_nabla_ideal)
    mp.setattr(transport.mlt, "mlt_nabla", _mlt_nabla_ideal)
    mp.setattr(solver.residual, "mlt_nabla", _mlt_nabla_ideal)


# ─── schwarzschild_only: zero the Ledoux composition-gradient term (Part B mutation) ───
# The production solver computes gradL_composition_term = max(d ln μ / d ln P, 0)
# and passes it to mlt_nabla. This mutation patches BOTH:
# (1) The helper that computes the term (for in-solver integration tests)
# (2) mlt_nabla itself to ignore any gradL_composition_term passed to it
# This reverts to pure Schwarzschild (no mu-barrier stabilization).
@register("schwarzschild_only")
def _schwarzschild_only(mp, stellar_mod=None):
    import stellar_jax.solver.residual
    import stellar_jax.transport as transport
    import stellar_jax.transport.mlt
    import jax.numpy as jnp
    from stellar_jax.transport.mlt import _mlt_cubic_solve
    from stellar_jax.transport.switch import _mlt_switch

    # (1) Patch the computation in the solver
    def _zero_gradL(y, X_profile, Z, q_mesh, M_star, comp_mfracs):
        """Always return zero composition gradient (Schwarzschild only)."""
        N_s = y.shape[0]
        N_c = N_s - 1
        return jnp.zeros(N_c)

    mp.setattr(solver.residual, "_compute_gradL_composition_term", _zero_gradL)

    # (2) Patch mlt_nabla to force gradL_composition_term=0 regardless of input
    _orig_mlt_nabla = transport.mlt.mlt_nabla

    def _mlt_nabla_schwarz(nabla_rad, nad, T, P, rho, kappa, g, mu, alpha_mlt,
                           cp=None, Q=None, gradL_composition_term=0.0):
        """Force Schwarzschild (ignore any composition gradient term)."""
        return _orig_mlt_nabla(nabla_rad, nad, T, P, rho, kappa, g, mu, alpha_mlt,
                               cp=cp, Q=Q, gradL_composition_term=0.0)

    mp.setattr(transport, "mlt_nabla", _mlt_nabla_schwarz)
    mp.setattr(transport.mlt, "mlt_nabla", _mlt_nabla_schwarz)
    mp.setattr(solver.residual, "mlt_nabla", _mlt_nabla_schwarz)


# ─── disable_warmstart_fn: bypass the warmstart_fn in infer_parameters ───
# The warmstart_fn provides a better initial guess for the LM optimizer, reducing
# iteration count. This mutation makes infer_parameters ignore the warmstart_fn
# (always use the explicit theta_init or a far-from-truth default), so the
# iteration-count advantage disappears. A test asserting "warm-start uses fewer
# iterations" must FAIL under this mutation.
# Ref:, epic Arm B (emulator warm-start).
@register("disable_warmstart_fn")
def _disable_warmstart_fn(mp, stellar_mod=None):
    """Disable warmstart_fn in infer_parameters — forces cold start always.

    Patches infer_parameters to ignore warmstart_fn (sets it to None before
    calling the real implementation). A test that asserts fewer iterations
    with warm-start will FAIL because both paths use the same cold init.
    """
    import stellar_jax.inference.solver as solver_mod
    _orig_infer = solver_mod.infer_parameters

    def _no_warmstart(*args, **kwargs):
        # Strip warmstart_fn — force cold start
        kwargs.pop('warmstart_fn', None)
        return _orig_infer(*args, **kwargs)

    mp.setattr(solver_mod, "infer_parameters", _no_warmstart, raising=True)
    # Also patch the package-level re-export
    import stellar_jax.inference as inference
    mp.setattr(inference, "infer_parameters", _no_warmstart, raising=True)


# ─── mlt_alpha_insensitive: clamp alpha_mlt to 2.0 inside mlt_nabla ───
# The alpha-sensitivity tests evaluate mlt_nabla at different alpha values and assert
# the output gradient changes. Under this mutation, mlt_nabla ignores the alpha_mlt
# argument (clamps to 2.0 internally), so the sensitivity d(nabla)/d(alpha) ≈ 0.
# This breaks: test_alpha_sensitivity_mass_dependent_vs_mesa_fgong,
#              test_alpha_affects_mlt_gradient_vs_mesa_fgong.
# Ref: Cox & Giuli (1968) §14 — alpha controls the mixing length Lambda = alpha * H_p;
#      different alpha → different convective efficiency → different nabla.
@register("mlt_alpha_insensitive")
def _mlt_alpha_insensitive(mp, stellar_mod=None):
    """Clamp alpha_mlt to 2.0 inside mlt_nabla, killing alpha sensitivity.

    Patches mlt_nabla and mlt_nabla_raw in the source module, the transport
    re-export, AND every solver-site import-time local binding that the hot
    path uses.  Without the solver-site patches the monkeypatch does not reach
    the Henyey solver (Python's ``from X import Y`` creates a local name at
    import time; patching X.Y later does NOT update the local name).

    Pattern follows ideal_gas_mlt (line ~1821) which correctly patches
    solver.residual.mlt_nabla.

    Forward value at alpha=2.0 is unchanged; but calling with alpha != 2.0
    now returns the alpha=2.0 result → d(nabla)/d(alpha) = 0.
    """
    import stellar_jax.transport as transport
    import stellar_jax.transport.mlt as mlt_mod
    import stellar_jax.solver.residual as solver_residual
    import stellar_jax.solver.shell_data as solver_shell_data
    import stellar_jax.solver.surface_bc as solver_surface_bc
    import stellar_jax.structure as structure_mod
    import jax.numpy as jnp

    _orig_mlt_nabla = mlt_mod.mlt_nabla
    _orig_mlt_nabla_raw = mlt_mod.mlt_nabla_raw

    def _clamped_mlt_nabla(nabla_rad, nad, T, P, rho, kappa, g, mu,
                           alpha_mlt, cp=None, Q=None,
                           gradL_composition_term=0.0):
        """mlt_nabla with alpha clamped to 2.0 — kills sensitivity."""
        alpha_fixed = jnp.float64(2.0)
        return _orig_mlt_nabla(nabla_rad, nad, T, P, rho, kappa, g, mu,
                               alpha_fixed, cp=cp, Q=Q,
                               gradL_composition_term=gradL_composition_term)

    def _clamped_mlt_nabla_raw(nabla_rad, nad, T, P, rho, kappa, g, mu,
                               alpha_mlt, cp=None, Q=None):
        """mlt_nabla_raw with alpha clamped to 2.0."""
        alpha_fixed = jnp.float64(2.0)
        return _orig_mlt_nabla_raw(nabla_rad, nad, T, P, rho, kappa, g, mu,
                                   alpha_fixed, cp=cp, Q=Q)

    # Source module + transport re-export
    mp.setattr(mlt_mod, "mlt_nabla", _clamped_mlt_nabla, raising=True)
    mp.setattr(mlt_mod, "mlt_nabla_raw", _clamped_mlt_nabla_raw, raising=True)
    mp.setattr(transport, "mlt_nabla", _clamped_mlt_nabla, raising=True)
    mp.setattr(transport, "mlt_nabla_raw", _clamped_mlt_nabla_raw, raising=True)

    # Solver-site import-time local bindings (the hot path)
    mp.setattr(solver_residual, "mlt_nabla", _clamped_mlt_nabla, raising=True)
    mp.setattr(solver_residual, "mlt_nabla_raw", _clamped_mlt_nabla_raw, raising=True)
    mp.setattr(solver_shell_data, "mlt_nabla", _clamped_mlt_nabla, raising=True)
    mp.setattr(solver_surface_bc, "mlt_nabla", _clamped_mlt_nabla, raising=True)
    mp.setattr(structure_mod, "mlt_nabla", _clamped_mlt_nabla, raising=True)


# ─── disable_envelope_mixing: no-op the envelope CZ homogenization (FDU mutation) ───
# During the first dredge-up, the deepening convective envelope mixes CN-processed
# material from the interior to the surface. _envelope_zone_mixing (composition/mix.py)
# identifies contiguous envelope CZs and homogenizes composition within them.
# This mutation disables it via a module-level flag — surface composition stays at
# its pristine initial value, and the C12 depletion assertion fails.
#
# The flag is read at runtime and passed as a JAX-traced boolean to mix_composition,
# which uses jnp.where (not a Python 'if') to select between the mixed and unmixed
# result. This guarantees the mutation works regardless of JAX's persistent
# compilation cache state: both paths are always compiled into the XLA graph, and
# the flag selects at runtime. A Python 'if' (the previous approach) gets baked in
# at trace time and can be defeated by a stale cached executable.
#
# Ref: MESA mix_info.f90:488-530 (CZ classification),
#      adjust_xyz.f90:do_adjust_xyz_for_mixing (homogenization);
#      Iben (1967), ApJ 147, 624; Charbonnel (1994), A&A 282, 811.
@register("disable_envelope_mixing")
def _disable_envelope_mixing(mp, stellar_mod=None):
    """Disable envelope CZ mixing — surface C12 stays pristine (FDU fails).

    Direct function patch → _envelope_zone_mixing replaced with an identity
    function that returns its input unchanged. This operates at TRACE TIME:
    when JAX traces _jit_composition_update in the O2 subprocess (persistent
    cache disabled by conftest), it traces the PATCHED function — the XLA graph
    never contains the mixing logic.

    The former module-level _ENVELOPE_MIXING_ENABLED flag was removed (#1070)
    as part of the module-global-switch cleanup. The function patch is the
    sole mechanism: it is robust against JIT cache state because conftest
    disables the persistent cache before mutation tests run.
    """
    import stellar_jax.composition.mix as mix_mod

    # Direct function patch → identity at trace time.
    # The patched function has the same signature as the original but returns
    # its first argument unchanged (no-op mixing). Since the conftest disables
    # the persistent compilation cache before this runs, the JIT will trace fresh
    # and bake the no-op into the XLA graph.
    def _noop_envelope_mixing(X_after_core, comp_mfracs, mf_shells, nrad, nad,
                              core_mix_mask, zone_weights, _K_env=30.0,
                              gradL_composition_term=None):
        return X_after_core

    mp.setattr(mix_mod, "_envelope_zone_mixing", _noop_envelope_mixing)


# ─── disable_energy_row_scaling: break the forward Newton energy-row scaling ───
# MESA always scales the energy equation row by dt/energy_start(k) in the Newton
# solve (star_utils.f90:3678 set_energy_eqn_scal). Without this scaling, the energy
# equation is O(cp*T/dt) while other equations are O(1), causing poor conditioning
# and under-converged eps_grav-dominated steps (the 0.166 dex logTeff shift on
# 2.0 M☉ RGB observed in when scaling was gated to replay-only).
# Ref: MESA star_utils.f90:3678 (set_energy_eqn_scal)
@register("disable_energy_row_scaling")
def _disable_energy_row_scaling(mp, stellar_mod=None):
    """Disable the forward energy-row scaling → convergence degrades on stiff steps.

    Patches _energy_row_scale_factors in solver.continuation (the import binding
    used in the forward Newton loop) to always return ones — effectively disabling
    the MESA set_energy_eqn_scal conditioning.
    """
    import jax.numpy as jnp
    import stellar_jax.solver.continuation

    def _ones_scale(y, q_mesh, X_profile, Z, inv_dt, comp_mfracs=None):
        N_s = y.shape[0]
        return jnp.ones(N_s)

    mp.setattr(solver.continuation, "_energy_row_scale_factors", _ones_scale)


# ─── [RETIRED] corrupt_energy_row_scaling ───
# Removed per: this was a strawman mutation that merely inflated the
# convergence-gate residual by 1e4× (brute-force gate trip). The honest mutation
# is disable_energy_row_scaling (returns all-ones), tested in the stiff regime by
# test_gradient_energy_row_scaling_stiff_regime. The corrupt variant is banned under
# the O2-hardening policy (/) which rejects gross-constant strawmen.



# ─── center_grid_include: include center (x=0) in integration grid → nan eigenvalues ───
# The oscillation ODE has 1/x terms, so integrating from x=0 produces nan.
# This mutation reverts the center-exclusion fix by patching _make_integration_grid
# to accept the full x_grid (including x=0) without filtering.
# Reference: ADIPLS adipls.c.d.f:1640-1641 starts at nibc=2 when x(1)=0.
@register("center_grid_include")
def _center_grid_include(mp, stellar_mod=None):
    """Revert center-exclusion: pass unfiltered x_grid to _make_integration_grid."""
    import stellar_jax.oscillations as oscillations
    import stellar_jax.oscillations.eigenvalue
    import numpy as np
    from stellar_jax.oscillations.integrator import _make_integration_grid as _orig_make_grid

    _orig_eigenfreq = oscillations.eigenvalue.compute_eigenfreq_from_structure_jax

    def _broken_eigenfreq(glob_jax, var_jax, l, nu_min, nu_max,
                          n_scan=200, n_steps=8000, mode_index=0):
        """Same as compute_eigenfreq_from_structure_jax but WITHOUT center filtering."""
        from stellar_jax.oscillations.coefficients import build_oscillation_coeffs_jax
        from stellar_jax.oscillations.determinant import radial_determinant, nonradial_determinant
        import jax
        import jax.numpy as jnp

        grid_data = build_oscillation_coeffs_jax(glob_jax, var_jax)
        x_grid_jax = grid_data['x_grid']
        coeffs_jax = grid_data['coeffs']
        factor = float(grid_data['factor'])

        # BUG: no center filtering — integration starts at x=0
        x_steps, h_steps = _orig_make_grid(np.asarray(x_grid_jax), n_steps=n_steps)

        if l == 0:
            @jax.jit
            def det_fn(s2):
                return radial_determinant(s2, x_steps, h_steps, x_grid_jax, coeffs_jax)
        else:
            l_float = float(l)
            @jax.jit
            def det_fn(s2):
                return nonradial_determinant(s2, jnp.float64(l_float),
                                             x_steps, h_steps, x_grid_jax, coeffs_jax)

        nu_arr = np.linspace(nu_min, nu_max, n_scan)
        roots_nu = oscillations.eigenvalue._bracket_and_refine(
            det_fn, nu_arr, factor, rtol=1e-10, maxiter=100)

        if len(roots_nu) <= mode_index:
            raise ValueError(
                f"Only {len(roots_nu)} modes found in [{nu_min}, {nu_max}] μHz "
                f"for l={l}, but mode_index={mode_index} requested")

        nu_converged = roots_nu[mode_index]
        sigma2_converged = jnp.float64(nu_converged**2 * factor)
        return {
            'nu': nu_converged,
            'sigma2': sigma2_converged,
            'factor': factor,
            'x_grid': x_grid_jax,
            'coeffs': coeffs_jax,
            'x_steps': x_steps,
            'h_steps': h_steps,
            'l': l,
        }

    import stellar_jax.oscillations as oscillations
    mp.setattr(oscillations.eigenvalue, "compute_eigenfreq_from_structure_jax",
               _broken_eigenfreq)
    # Also patch the package-level export (from oscillations import ... uses this)
    mp.setattr(oscillations, "compute_eigenfreq_from_structure_jax",
               _broken_eigenfreq)


# ─── disable_center_extension: revert the center extension fix → nonradial nan ───────
# The center extension prepends synthetic center points to the oscillation coefficient
# grid when the Henyey mesh starts far from center (x_min > 0.01). Without it, the
# nonradial integrator applies center regularity BCs at x≈0.03 where U≈5e7 (instead
# of 3 at x→0), causing RK4 overflow → nan for all l≥1 modes.
# Reference:, ADIPLS setbcs.n.d.f (truncated-model BCs).
@register("disable_center_extension")
def _disable_center_extension(mp, stellar_mod=None):
    """Disable center extension: return grid unchanged regardless of x_min."""
    import stellar_jax.oscillations.coefficients

    def _noop_extend(x_grid, coeffs, A1_grid):
        """No-op: always return grid unchanged (skip center extension)."""
        return x_grid, coeffs, 0

    mp.setattr(oscillations.coefficients, "_extend_to_center", _noop_extend)


# ─── force_l_zero_in_nonradial: zero the angular-degree coupling in nonradial ODE ────
# The nonradial oscillation equations (GYRE formulation, post-) depend on l
# through ll1 = l*(l+1) in the RHS matrix AND through bare l in the diagonal shifts.
# Forcing BOTH ll1=0 AND l=0 makes the nonradial (l=2) RHS identical to the l=0
# (radial) case — collapsing the l=2 modes onto l=0 frequencies. With BCs still
# expecting l=2, the system is inconsistent and no valid l=2 sign changes are found
# (or they coincide with l=0 frequencies → δν₀₂ ≈ 0 or undefined).
# This specifically tests the l-dependent physics measured by δν₀₂.
# Updated 2026-08-25 to match the GYRE formulation (ported the RHS from
# ADIPLS to GYRE; the prior version used the old ADIPLS equations which no longer
# matched the production code's BCs → modes leaked through → theater).
# Reference: (standalone δν₀₂ forward capability test);
#            Townsend & Teitler (2013) MNRAS 435, 3406 (GYRE formulation).
@register("force_l_zero_in_nonradial")
def _force_l_zero_in_nonradial(mp, stellar_mod=None):
    """Force l=0 and ll1=0 in the nonradial GYRE RHS, destroying angular-degree dependence."""
    import jax.numpy as jnp
    import stellar_jax.oscillations.integrator

    def _nonradial_rhs_l_zero(x_val, y, sigma2, l, x_grid, coeffs):
        """Nonradial RHS (GYRE form) with forced l=0, ll1=0 — collapses l=2 onto l=0."""
        from stellar_jax.oscillations.integrator import _interp_coeffs
        Vg, A1, A_bv, U, _ = _interp_coeffs(x_val, x_grid, coeffs)
        As = A_bv
        # MUTATION: force l=0 AND ll1=0 — removes ALL angular-degree dependence
        ll1 = 0.0
        lam_c1w2 = 0.0  # ll1 * A1 / sigma2 = 0
        c1w2 = sigma2 / jnp.maximum(A1, 1e-30)  # c₁σ² (unchanged — frequency term)

        y1, y2, y3, y4 = y[0], y[1], y[2], y[3]

        # GYRE xA matrix with l=0, ll1=0 (all -l terms become 0):
        dy1 = ((Vg - 1.0) * y1
               + (- Vg) * y2
               + 0.0) / x_val
        dy2 = ((c1w2 - As) * y1
               + (As - U + 3.0) * y2
               - y4) / x_val
        dy3 = ((3.0 - U) * y3
               + y4) / x_val
        dy4 = (U * As * y1
               + U * Vg * y2
               + 0.0
               + (2.0 - U) * y4) / x_val
        return jnp.array([dy1, dy2, dy3, dy4])

    mp.setattr(oscillations.integrator, "_nonradial_rhs", _nonradial_rhs_l_zero)


# ─── adipls_a_formulation_nonradial: revert nonradial RHS to the ADIPLS A-form ──────
# The GYRE formulation (this repo, post-) carries the Brunt term As UN-amplified.
# The ADIPLS A-formulation instead multiplies A by η = l(l+1)/(c₁σ²) — frequency-
# amplified, numerically explosive for sharp evolved-core composition gradients — which
# biased δν₀₂ (the age signal) 5–40% low. This mutation restores that A-formulation RHS
# (a REALISTIC break — it is exactly the pre-fix code), so the δν₀₂-vs-GYRE test must
# fail (δν₀₂ regresses well beyond the 2% bound). Reference:.
@register("adipls_a_formulation_nonradial")
def _adipls_a_formulation_nonradial(mp, stellar_mod=None):
    """Revert _nonradial_rhs to the ADIPLS A-formulation (re-introduces the δν₀₂ bias)."""
    import jax.numpy as jnp
    import stellar_jax.oscillations.integrator

    def _nonradial_rhs_adipls(x_val, y, sigma2, l, x_grid, coeffs):
        from stellar_jax.oscillations.integrator import _interp_coeffs
        Vg, A1, A_bv, U, _ = _interp_coeffs(x_val, x_grid, coeffs)
        ll1 = l * (l + 1.0)
        eta = ll1 * A1 / sigma2  # frequency-amplified Brunt coupling (the bug)
        y1, y2, y3, y4 = y[0], y[1], y[2], y[3]
        dy1 = ((Vg - 2.0) * y1
               + (1.0 - Vg / jnp.maximum(eta, 1e-30)) * y2
               - Vg * y3) / x_val
        dy2 = ((ll1 - eta * A_bv) * y1
               + (A_bv - 1.0) * y2
               + eta * A_bv * y3) / x_val
        dy3 = (y3 + y4) / x_val
        dy4 = (-A_bv * U * y1
               - U * Vg / jnp.maximum(eta, 1e-30) * y2
               + (ll1 + U * (A_bv - 2.0) + U * Vg) * y3
               + 2.0 * (1.0 - U) * y4) / x_val
        return jnp.array([dy1, dy2, dy3, dy4])

    mp.setattr(oscillations.integrator, "_nonradial_rhs", _nonradial_rhs_adipls)


# ─── chandrasekhar_gamma1_corrupt: corrupt radiation constant → wrong β → wrong Γ₁ ────
# The Chandrasekhar formula Γ₁ = (32 - 24β - 3β²)/(24 - 21β) depends on
# β = P_gas/P_total = (P - P_rad)/P where P_rad = a_rad·T⁴/3.
# Corrupting a_rad by 10× shifts P_rad → wrong β → wrong Γ₁ by ~15% in the
# deep interior (where radiation pressure contributes), far exceeding the 0.1% tol.
# This tests that the sound-speed diagnostic uses the CORRECT radiation constant.
# Reference: Chandrasekhar (1939), "An Introduction to the Study of Stellar Structure".
@register("chandrasekhar_gamma1_corrupt")
def _chandrasekhar_gamma1_corrupt(mp, stellar_mod=None):
    """Corrupt the radiation constant a_rad by 10× — breaks Chandrasekhar Γ₁."""
    import stellar_jax.config.constants
    import stellar_jax.config.constants as constants
    # Corrupt at both the source and re-export locations
    original = config.constants.a_rad
    mp.setattr(config.constants, "a_rad", original * 10.0)
    mp.setattr(constants, "a_rad", original * 10.0)


# ─── eos_nad_offset: shift nabla_ad by +0.05 (12%), breaks per-zone ∇_ad validation ────
# Our EOS returns ∇_ad (the adiabatic temperature gradient) from table interpolation.
# Shifting it by +0.05 (from ~0.4 to ~0.45 in the interior, a 12.5% change) exceeds
# the 2% tolerance in test_nabla_ad_vs_mesa_fgong.
# Patches at the microphysics.eos module level (where the test imports from).
# Reference: Kippenhahn, Weigert & Weiss (2012), §13.1 (adiabatic gradient).
@register("eos_nad_offset")
def _eos_nad_offset(mp, stellar_mod=None):
    """Shift ∇_ad by +0.05 — breaks per-zone ∇_ad agreement with MESA FGONG."""
    import stellar_jax.microphysics.eos
    _orig_eos = microphysics.eos.eos_lookup

    def _shifted_eos(logT, logP, X, Z):
        result = _orig_eos(logT, logP, X, Z)
        rho, mu, nad, S, cp, chi_rho, chi_T = result
        # Shift ∇_ad by +0.05 (12.5% for interior where nad≈0.4)
        nad_shifted = nad + 0.05
        return (rho, mu, nad_shifted, S, cp, chi_rho, chi_T)

    mp.setattr(microphysics.eos, "eos_lookup", _shifted_eos, raising=True)


# ─── solar_constants_corrupt: corrupt solar reference values → wrong ν_max scaling ────
# The ν_max scaling relation uses ν_max☉ = 3090 μHz, Teff☉ = 5777 K, and
# R☉ = 6.957e10 cm. Corrupting R☉ by 3× shifts M/M☉ or R/R☉ such that ν_max
# moves outside the expected range. This tests that the scaling relation uses
# correct solar constants.
# Reference: Brown et al. (1991), ApJ 368, 599; Kjeldsen & Bedding (1995), A&A 293, 87.
@register("solar_constants_corrupt")
def _solar_constants_corrupt(mp, stellar_mod=None):
    """Corrupt Rsun by 3× — shifts ν_max scaling outside expected range."""
    import stellar_jax.config.constants
    import stellar_jax.config.constants as constants
    original_Rsun = config.constants.Rsun
    mp.setattr(config.constants, "Rsun", original_Rsun * 3.0)
    mp.setattr(constants, "Rsun", original_Rsun * 3.0)


# ─── corrupt_brunt_stencil: degrade the Fornberg stencil in A* computation ───
# The _compute_brunt_vaisala uses proper Fornberg unequal-spacing FD weights.
# This mutation replaces it with a zeroth-order "constant spacing" formula that
# is O(h) on non-uniform meshes — introducing ~1-5% systematic errors on the
# quadratic-stretched FGONG mesh. A test asserting <0.5% RMS agreement with
# MESA col-14 must FAIL under this mutation.
@register("corrupt_brunt_stencil")
def _corrupt_brunt_stencil(mp, stellar_mod=None):
    """Replace the Fornberg O(h²) stencil with a degraded constant-spacing formula."""
    import stellar_jax.fgong.io
    import numpy as np

    def _brunt_bad_stencil(r, P, rho, Gamma1, R_star, mu=None,
                            chi_rho=None, chi_T=None):
        """A* with deliberately degraded stencil (constant-spacing on non-uniform mesh)."""
        nn = len(r)
        r_min_brunt = 1e-4 * R_star
        AA = np.zeros(nn)
        lnP = np.log(np.maximum(P, 1.0))
        lnrho = np.log(np.maximum(rho, 1e-30))
        for i in range(1, nn - 1):
            if r[i] <= r_min_brunt:
                continue
            # BAD: use constant-spacing formula (ignores non-uniform mesh)
            # This is what the code used before — treats all spacing as equal
            dr = r[i+1] - r[i-1]
            if abs(dr) < 1e-30:
                continue
            dlnP_dlnr = (lnP[i+1] - lnP[i-1]) / dr * r[i]
            dlnrho_dlnr = (lnrho[i+1] - lnrho[i-1]) / dr * r[i]
            AA[i] = dlnP_dlnr / max(Gamma1[i], 0.1) - dlnrho_dlnr
            # MUTATION: also add a 5% systematic offset to make failure clear
            AA[i] *= 1.05
        return AA

    mp.setattr(fgong.io, "_compute_brunt_vaisala", _brunt_bad_stencil)


# ─── hires_gamma1_cp_form: restore the buggy Γ₁ formula that uses cp instead of cv ────
#: hires_profile.py and builder.py computed Gamma3-1 = P*δ/(ρ*T*cp) — using
# cp where cv belongs. This gives Γ₁ ≈ 1.4 (ideal gas) instead of the correct 5/3.
# The correct form is Γ₁ = χ_ρ / (1 - ∇_ad·χ_T), equivalent to MESA's
# gamma1 = chiT*(gamma3-1) + chiRho with gamma3-1 = P*chiT/(rho*T*Cv).
# This mutation patches eos_lookup to return a modified chi_rho that, when fed through
# the correct formula, reproduces the old buggy result: Γ₁ = χ_ρ + ∇_ad·χ_T ≈ 1.4.
# Since cp = P*δ/(ρ*T*∇_ad) (Maxwell relation), the cp-form Gamma3-1 = ∇_ad, so
# the buggy Gamma1 = chi_rho + nad*chi_T. To make chi_rho/(1-nad*chi_T) yield
# chi_rho + nad*chi_T, we set chi_rho_fake = (chi_rho + nad*chi_T)*(1 - nad*chi_T).
# Reference: MESA eospc_eval.f90:293-294.
@register("hires_gamma1_cp_form")
def _hires_gamma1_cp_form(mp, stellar_mod=None):
    """Restore the buggy cp-based Γ₁ formula — gives ~1.4 instead of 5/3."""
    import stellar_jax.microphysics.eos
    import jax.numpy as jnp

    _orig_eos = microphysics.eos.eos_lookup

    def _eos_buggy_gamma1(logT, logP, X, Z):
        rho, mu, nad, S, cp, chi_rho, chi_T = _orig_eos(logT, logP, X, Z)
        # The buggy formula gives: Gamma1 = chi_rho + nad * chi_T (≈1.4 for ideal gas)
        # To make the CORRECT formula chi_rho/(1 - nad*chi_T) reproduce this, we modify
        # chi_rho so that: chi_rho_fake/(1 - nad*chi_T) = chi_rho + nad*chi_T
        buggy_g1 = chi_rho + nad * chi_T
        chi_rho_fake = buggy_g1 * jnp.maximum(1.0 - nad * chi_T, 1e-10)
        return (rho, mu, nad, S, cp, chi_rho_fake, chi_T)

    mp.setattr(microphysics.eos, "eos_lookup", _eos_buggy_gamma1, raising=True)


# ─── revert_diagnostic_to_adipls: undo the GYRE formulation in diagnostic.py ───
# The GYRE formulation enters the Brunt A* un-amplified in the nonradial RHS.
# The old ADIPLS A-formulation multiplied A* by η = l(l+1)/(c₁σ²) — a frequency-
# amplified factor that biased δν₀₂ low by 5–40% for evolved cores.
# This mutation reverts the nonradial RHS to the ADIPLS η-amplified form, so the
# GYRE-reference validation test must FAIL (frequencies shift by ~3-5 µHz for l=2).
# Reference: Christensen-Dalsgaard (2008), Ap&SS 316, 113, §4 (A-formulation caveat).
@register("revert_diagnostic_to_adipls")
def _revert_diagnostic_to_adipls(mp, stellar_mod=None):
    """Revert diagnostic.py nonradial RHS from GYRE → ADIPLS A-formulation."""
    import numpy as np
    import stellar_jax.oscillations.diagnostic

    _orig_full = oscillations.diagnostic.compute_oscillation_freqs_full

    def _patched_full(glob, var, l_values=(0, 1, 2, 3),
                      nu_min=1000.0, nu_max=4500.0, n_scan=500):
        """Hijack the full solver: replace nonradial shooting with ADIPLS-style."""
        from scipy.integrate import solve_ivp
        from scipy.optimize import brentq

        grid = oscillations.diagnostic._build_oscillation_grid(glob, var)
        x = grid['x']
        M, R, G_val = grid['M'], grid['R'], grid['G']
        r = x * R
        m = grid['m_frac'] * M
        P = grid['P']
        rho = grid['rho']
        gamma1 = grid['gamma1']
        g = grid['g']

        Vg = G_val * m * rho / (np.maximum(r, 1.0) * gamma1 * np.maximum(P, 1e-30))
        c1 = (r / R)**3 * M / np.maximum(m, 1e-10 * M)
        A_bv = grid['N2'] * np.maximum(r, 1.0) / np.maximum(g, 1e-10)
        U = 4.0 * np.pi * r**3 * rho / np.maximum(m, 1e-10 * M)

        mask = (x > 1e-4)
        x_grid = x[mask]
        Vg_grid = np.nan_to_num(Vg[mask], nan=0.0, posinf=0.0, neginf=0.0)
        c1_grid = np.nan_to_num(c1[mask], nan=1.0, posinf=1e10, neginf=1.0)
        A_grid = np.nan_to_num(A_bv[mask], nan=0.0, posinf=0.0, neginf=0.0)
        U_grid = np.nan_to_num(U[mask], nan=3.0, posinf=3.0, neginf=3.0)
        factor = (2.0 * np.pi * 1e-6)**2 * R**3 / (G_val * M)

        def _nonradial_adipls(sigma2, l):
            """ADIPLS A-formulation nonradial RHS (η-amplified A*)."""
            ll1 = l * (l + 1)

            def rhs(xv, y):
                Vg_x = np.interp(xv, x_grid, Vg_grid)
                c1_x = np.interp(xv, x_grid, c1_grid)
                As_x = np.interp(xv, x_grid, A_grid)
                U_x = np.interp(xv, x_grid, U_grid)
                # ADIPLS η = l(l+1)/(c₁σ²) — the frequency-amplified factor
                eta = ll1 / (c1_x * sigma2)
                y1, y2, y3, y4 = y
                dy1 = ((Vg_x - 2.0) * y1
                       + (1.0 - Vg_x / max(eta, 1e-30)) * y2
                       - Vg_x * y3) / xv
                dy2 = ((ll1 - eta * As_x) * y1
                       + (As_x - 1.0) * y2
                       + eta * As_x * y3) / xv
                dy3 = (y3 + y4) / xv
                dy4 = (-As_x * U_x * y1
                       - U_x * Vg_x / max(eta, 1e-30) * y2
                       + (ll1 + U_x * (As_x - 2.0) + U_x * Vg_x) * y3
                       + 2.0 * (1.0 - U_x) * y4) / xv
                return [dy1, dy2, dy3, dy4]

            x_in = x_grid[0]
            c1_c = c1_grid[0]
            ic1 = [1.0, sigma2 / ll1, 0.0, 0.0]
            ic2 = [0.0, 0.0, 1.0, float(l)]
            sol1 = solve_ivp(rhs, [x_in, x_grid[-1]], ic1,
                             method='DOP853', rtol=1e-8, atol=1e-10, max_step=0.05)
            sol2 = solve_ivp(rhs, [x_in, x_grid[-1]], ic2,
                             method='DOP853', rtol=1e-8, atol=1e-10, max_step=0.05)
            if not sol1.success or not sol2.success:
                return np.nan
            y1_1, y2_1, y3_1, y4_1 = sol1.y[:, -1]
            y1_2, y2_2, y3_2, y4_2 = sol2.y[:, -1]
            x_s = x_grid[-1]
            c1_s = np.interp(x_s, x_grid, c1_grid)
            eta_s = ll1 / (c1_s * sigma2)
            bc1_1 = y2_1 - (eta_s / x_s) * y1_1 - eta_s * y3_1
            bc1_2 = y2_2 - (eta_s / x_s) * y1_2 - eta_s * y3_2
            bc2_1 = y4_1 + (l + 1) * y3_1
            bc2_2 = y4_2 + (l + 1) * y3_2
            return bc1_1 * bc2_2 - bc1_2 * bc2_1

        # Only patch l≥1; keep radial (l=0) from the GYRE formulation (unchanged)
        results = {}
        for l in l_values:
            nu_arr = np.linspace(nu_min, nu_max, n_scan)
            sigma2_arr = nu_arr**2 * factor
            if l == 0:
                # Delegate to the real (GYRE) radial solver
                r0 = _orig_full(glob, var, l_values=(0,),
                                nu_min=nu_min, nu_max=nu_max, n_scan=n_scan)
                results[0] = r0[0]
                continue
            det_values = np.array([_nonradial_adipls(s2, l) for s2 in sigma2_arr])
            freqs = []
            for i in range(len(det_values) - 1):
                if np.isnan(det_values[i]) or np.isnan(det_values[i + 1]):
                    continue
                if det_values[i] * det_values[i + 1] < 0:
                    def f_root(nu, _l=l):
                        return _nonradial_adipls(nu**2 * factor, _l)
                    try:
                        nu_root = brentq(f_root, nu_arr[i], nu_arr[i + 1],
                                         rtol=1e-8, maxiter=50)
                        freqs.append(nu_root)
                    except (ValueError, RuntimeError):
                        continue
            results[l] = np.array(sorted(freqs))
        return results

    mp.setattr(oscillations.diagnostic, "compute_oscillation_freqs_full", _patched_full)
    # Also patch the re-export in oscillations/__init__.py — the test does
    # `from oscillations import compute_oscillation_freqs_full` which resolves
    # through the package namespace, not oscillations.diagnostic directly.
    import stellar_jax.oscillations as _osc_pkg
    mp.setattr(_osc_pkg, "compute_oscillation_freqs_full", _patched_full)



# ─── detach_y_init_gradient: sever Y_init → composition → logL/logTeff path ───
# Y_init is the initial helium mass fraction passed to evolve_star. It determines
# the initial composition (X = 1 - Y - Z), which feeds into the EOS, opacity,
# and nuclear burning throughout the evolution. Detaching Y_init with stop_gradient
# severs the AD path Y→X_init→composition→structure→observables, making
# ∂logL/∂Y = ∂logTeff/∂Y = 0 by AD while the forward model is unaffected.
# Any test asserting non-zero ∂/∂Y will fail under this mutation.
# Physical basis: Y determines the mean molecular weight μ = 1/(2X + 3Y/4 + Z/2),
# which controls the EOS (ρ, T profiles) and opacity (H/He ratio) — the entire
# thermodynamic structure depends on Y. A zero gradient is unphysical.
@register("detach_y_init_gradient")
def _detach_y_init_gradient(mp, stellar_mod=None):
    """stop_gradient on Y_init inside evolve_star — makes ∂/∂Y = 0."""
    import jax
    import stellar_jax.evolution as evolution

    _orig_evolve = evolution.evolve_star

    def _detached_y_evolve(*args, **kwargs):
        """Forward correct, backward path through Y_init severed."""
        if 'Y_init' in kwargs and kwargs['Y_init'] is not None:
            kwargs['Y_init'] = jax.lax.stop_gradient(kwargs['Y_init'])
        return _orig_evolve(*args, **kwargs)

    mp.setattr(evolution, "evolve_star", _detached_y_evolve)
    if stellar_mod is not None:
        mp.setattr(stellar_mod, "evolve_star", _detached_y_evolve)


# ─── detach_z_gradient: sever Z → composition/opacity/EOS → logL/logTeff/σ² path ───
# Z (metallicity) enters the pipeline as a traced jnp.float64 through TWO paths:
#   A. evolve_star(Z=z_val) — Z determines composition, opacity, EOS, CNO rate
#   B. structure_to_fgong_jax(..., Z=z_val, ...) — Z enters the FGONG builder's
#      EOS lookup (eos_lookup(logT, logPgas, X, Z) → ρ, Γ₁), creating a live
#      gradient path from Z → oscillation coefficients → eigenfrequency → σ²
# Both paths must be severed. Patching ONLY evolve_star (path A) leaves path B
# live, which keeps the AD gradient non-zero under mutation and makes tests
# insensitive — the exact O2 theater identified.
# Detaching Z with stop_gradient in BOTH paths severs ALL AD paths while the
# forward model is unaffected. Any test asserting non-zero ∂/∂Z will fail.
@register("detach_z_gradient")
def _detach_z_gradient(mp, stellar_mod=None):
    """stop_gradient on Z inside evolve_star AND structure_to_fgong_jax — makes ∂/∂Z = 0.

    Z enters the seismic gradient chain through TWO independent paths:
      1. Z → evolve_star → y_henyey → FGONG → σ²  (evolution path)
      2. Z → structure_to_fgong_jax → EOS(Z) → ρ, Γ₁ → coeffs → σ²  (FGONG builder path)
    Both paths must be severed for the mutation to make AD ≈ 0. Patching only
    evolve_star leaves path #2 live (the FGONG builder uses Z in its EOS lookup:
    eos_lookup(logT, logPgas, X, Z) → ρ, Γ₁), which keeps the AD gradient non-zero
    and makes the test pass under mutation — the exact theater issue #1159 found.

    Patches at ALL usage sites:
      - evolution.evolve_star (the evolution path)
      - fgong.builder.structure_to_fgong_jax (definition site)
      - evolution._core.structure_to_fgong_jax (intermediate import binding)
      - evolution.structure_to_fgong_jax (package-level re-export)
    """
    import jax
    import stellar_jax.evolution as evolution

    _orig_evolve = evolution.evolve_star

    def _detached_z_evolve(*args, **kwargs):
        """Forward correct, backward path through Z severed."""
        # Z is the second positional arg or keyword 'Z'
        if len(args) >= 2:
            args = list(args)
            args[1] = jax.lax.stop_gradient(jax.numpy.asarray(args[1], dtype=jax.numpy.float64))
            args = tuple(args)
        elif 'Z' in kwargs:
            kwargs['Z'] = jax.lax.stop_gradient(jax.numpy.asarray(kwargs['Z'], dtype=jax.numpy.float64))
        return _orig_evolve(*args, **kwargs)

    mp.setattr(evolution, "evolve_star", _detached_z_evolve)
    if stellar_mod is not None:
        mp.setattr(stellar_mod, "evolve_star", _detached_z_evolve)

    # ── Also sever Z in structure_to_fgong_jax (the FGONG builder path) ──
    # The test's sigma2_of_Z passes the traced z_val directly to
    # structure_to_fgong_jax as the 5th positional arg (Z). The builder uses
    # Z in eos_lookup(logT, logPgas, X, Z) to compute ρ and Γ₁ — a live
    # gradient path that bypasses the evolve_star stop_gradient. Wrapping
    # structure_to_fgong_jax to detach its Z argument severs this path.
    import stellar_jax.fgong.builder as fgong_builder
    import stellar_jax.evolution._core as evolution_core

    _orig_fgong = fgong_builder.structure_to_fgong_jax

    def _detached_z_fgong(M_solar, log_L, log_Te, X_profile, Z, *args, **kwargs):
        """Forward correct, backward path through Z in FGONG builder severed."""
        Z_detached = jax.lax.stop_gradient(jax.numpy.asarray(Z, dtype=jax.numpy.float64))
        return _orig_fgong(M_solar, log_L, log_Te, X_profile, Z_detached, *args, **kwargs)

    # Patch at all import binding sites:
    # 1. Definition site (fgong.builder)
    mp.setattr(fgong_builder, "structure_to_fgong_jax", _detached_z_fgong, raising=True)
    # 2. Intermediate binding (evolution._core imports from fgong.builder)
    mp.setattr(evolution_core, "structure_to_fgong_jax", _detached_z_fgong, raising=True)
    # 3. Package-level re-export (evolution.__init__ copies from _core)
    mp.setattr(evolution, "structure_to_fgong_jax", _detached_z_fgong, raising=True)


# ─── detach_mass_gradient: sever mass → structure → logL/logTeff path ───
# Mass is the first positional argument to evolve_star. It determines:
#   1. Gravitational structure: L_shell ∝ M_shell, P ∝ GM/R⁴ (hydrostatic eq)
#   2. Nuclear burning rate: L_nuc ∝ ρ T^ν (both ρ and T scale with M)
#   3. Convective structure: M→T_c→∇_rad/∇_ad→convective boundary position
#   4. Mass-luminosity relation: L ∝ M^{3-5} (homology, KWW §20.3)
# Detaching mass with stop_gradient severs ALL these AD paths while the
# forward model is unaffected. Any test asserting non-zero ∂/∂M will fail.
# Physical basis: the mass-luminosity relation is the most fundamental
# stellar property — a zero gradient is unphysical.
@register("detach_mass_gradient")
def _detach_mass_gradient(mp, stellar_mod=None):
    """stop_gradient on mass inside evolve_star — makes ∂/∂M = 0."""
    import jax
    import stellar_jax.evolution as evolution

    _orig_evolve = evolution.evolve_star

    def _detached_mass_evolve(*args, **kwargs):
        """Forward correct, backward path through mass severed."""
        # mass is the first positional arg
        if len(args) >= 1:
            args = list(args)
            args[0] = jax.lax.stop_gradient(jax.numpy.asarray(args[0], dtype=jax.numpy.float64))
            args = tuple(args)
        elif 'mass' in kwargs:
            kwargs['mass'] = jax.lax.stop_gradient(jax.numpy.asarray(kwargs['mass'], dtype=jax.numpy.float64))
        return _orig_evolve(*args, **kwargs)

    mp.setattr(evolution, "evolve_star", _detached_mass_evolve)
    if stellar_mod is not None:
        mp.setattr(stellar_mod, "evolve_star", _detached_mass_evolve)


# ─── detach_alpha_gradient: sever alpha_mlt → MLT → logL/logTeff path ───
# alpha_mlt is the mixing-length parameter passed to evolve_star. It determines:
#   1. Convective efficiency: higher α → larger mixing length → more efficient
#      energy transport in convective zones (Böhm-Vitense 1958)
#   2. Temperature gradient: in the superadiabatic layer, ∇ = ∇_ad + f(α)
#   3. Envelope structure: T(τ) atmosphere BC + MLT in the convective envelope
#      → Teff (the dominant α→observable channel)
#   4. Luminosity (indirect): via boundary conditions (L ∝ R² T_eff⁴)
# Detaching alpha_mlt with stop_gradient severs the AD path while the
# forward model is unaffected. Any test asserting non-zero ∂/∂α will fail.
# Physical basis: α_MLT sensitivity is the calibration knob for convection —
# a zero gradient would make solar calibration impossible.
@register("detach_alpha_gradient")
def _detach_alpha_gradient(mp, stellar_mod=None):
    """stop_gradient on alpha_mlt inside evolve_star — makes ∂/∂α = 0."""
    import jax
    import stellar_jax.evolution as evolution

    _orig_evolve = evolution.evolve_star

    def _detached_alpha_evolve(*args, **kwargs):
        """Forward correct, backward path through alpha_mlt severed."""
        # alpha_mlt is the 4th positional arg (mass, Z, max_steps, alpha_mlt)
        # or keyword 'alpha_mlt'
        if 'alpha_mlt' in kwargs:
            kwargs['alpha_mlt'] = jax.lax.stop_gradient(jax.numpy.asarray(kwargs['alpha_mlt'], dtype=jax.numpy.float64))
        elif len(args) >= 4:
            args = list(args)
            args[3] = jax.lax.stop_gradient(jax.numpy.asarray(args[3], dtype=jax.numpy.float64))
            args = tuple(args)
        return _orig_evolve(*args, **kwargs)

    mp.setattr(evolution, "evolve_star", _detached_alpha_evolve)
    if stellar_mod is not None:
        mp.setattr(stellar_mod, "evolve_star", _detached_alpha_evolve)


# ─── disable_jcd_atmosphere: force JCD BC to fall back to vacuum ───
# The JCD isothermal-atmosphere BC (Christensen-Dalsgaard 2008) shifts absolute
# frequencies down by ~1-5 µHz vs vacuum. Disabling it (forcing chi→vacuum fallback)
# means the JCD validation test must FAIL because frequencies will not match GYRE-JCD.
@register("disable_jcd_atmosphere")
def _disable_jcd_atmosphere(mp, stellar_mod=None):
    """Force the JCD BC to degenerate to vacuum by zeroing the atmospheric eigenvalue chi.

    When chi = b_11 = Vg-3, the first BC coefficient becomes chi-b_11 = 0, and the BC
    degenerates to -b_12*y₂ = 0 (trivial) rather than the isothermal-atmosphere matching.
    We patch _jcd_surface_coeffs to return vacuum-equivalent coefficients [1, -1, 0].
    """
    import stellar_jax.oscillations.determinant as det_mod
    import jax.numpy as jnp

    def _patched_jcd_coeffs(sigma2, l, Vg_s, A1_s, gamma1_s):
        # Return vacuum-equivalent: coeff_y1=1, coeff_y2=-1, coeff_y3=0
        # This makes the JCD BC identical to vacuum: y₁ - y₂ = 0
        return jnp.float64(1.0), jnp.float64(-1.0), jnp.float64(0.0)

    mp.setattr(det_mod, "_jcd_surface_coeffs", _patched_jcd_coeffs)


# ─── fd_electron_zero_table: zero the FD electron free energy table ───
# Zeroing the electron free energy table makes the electron contribution vanish,
# leaving only the ideal ion gas + radiation + Coulomb. This shifts Gamma1 toward
# the pure ideal gas value (~5/3 ≈ 1.667 for monatomic) and makes chi_rho ≈ 1,
# chi_T ≈ 1, which is drastically different from the degenerate regime where
# chi_rho > 1.5 and chi_T << 1. Any test checking Gamma1 against MESA at
# degenerate conditions will fail under this mutation.
# Reference: (full FD electron EOS table).
@register("fd_electron_zero_table")
def _fd_electron_zero_table(mp, stellar_mod=None):
    """Zero the FD electron free energy table, disabling the electron contribution."""
    import jax.numpy as jnp
    import stellar_jax.microphysics.fd_electron as fd_mod

    # Replace _F_electron_fd with a function that returns zero
    mp.setattr(fd_mod, "_F_electron_fd",
               lambda ln_rho, ln_T, X, Z: jnp.float64(0.0), raising=True)


# ─── disable_h211b_controller: bypass the H211b low-pass filter ───
# The H211b controller (Söderlind & Wang 2006) smooths dt changes via a 2nd-order
# digital filter with atan-limiter bandwidth κ=10. Disabling it reverts to the bare
# 1st-order controller: dt_next = dt * target/ratio — which has no memory and can
# oscillate/overshoot when limiters alternate between binding and not binding.
# Any test asserting the H211b smoothing property MUST FAIL under this mutation.
# MESA ref: timestep.f90:2362-2430 (filter_dt_next subroutine, enabled by default).
@register("disable_h211b_controller")
def _disable_h211b_controller(mp, stellar_mod=None):
    """Replace H211b controller with bare 1st-order ratio — removes smoothing."""
    import jax.numpy as jnp
    import stellar_jax.evolution.timestep as ts_mod

    def _first_order_only(dt, dt_old, dt_limit_ratio, dt_limit_ratio_old, order=1.0):
        """Always use 1st-order: dt_next = dt * 1.0 / max(ratio, 1e-10)."""
        ratio_safe = jnp.maximum(dt_limit_ratio, 1e-10)
        return dt * 1.0 / ratio_safe

    mp.setattr(ts_mod, "h211b_filter_dt_next", _first_order_only, raising=True)


# ─── disable_central_limiters: zero out central δlgT/δlgRho limiters ───
# The central limiters restrict dt based on changes in log10(T) and log10(ρ) at the
# stellar center. Disabling them (returning ratio=0 always) removes the constraint
# on dt from rapid central changes — material on the RGB where central ρ,T evolve
# fast. Any test asserting that central limiters bind MUST FAIL under this mutation.
# MESA ref: timestep.f90:1367-1440 (check_dlgT_cntr_change, check_dlgRho_cntr_change);
# MESA defaults: delta_lgT_cntr_limit=0.01, delta_lgRho_cntr_limit=0.05.
@register("disable_central_limiters")
def _disable_central_limiters(mp, stellar_mod=None):
    """Zero out central T/ρ limiter ratios — removes central-change dt constraint."""
    import jax.numpy as jnp
    import stellar_jax.evolution.timestep as ts_mod

    def _no_central(logT_cntr, logT_cntr_old, logRho_cntr, logRho_cntr_old,
                    delta_lgT_cntr_limit, delta_lgRho_cntr_limit):
        """Always return (0, 0) — central limiters never bind."""
        return jnp.float64(0.0), jnp.float64(0.0)

    mp.setattr(ts_mod, "compute_central_limiters", _no_central, raising=True)


# ─── fd_electron_scale_table: scale the FD electron free energy by 10× ───
# At partial degeneracy (logRho 5.0-5.5), zeroing the electron contribution
# leaves nabla_ad ≈ 0.4 (ideal ion gas), which is close enough to MESA's
# 0.37-0.40 to pass a 5% tolerance. MULTIPLYING by 10× instead grossly
# corrupts the thermodynamic derivatives: chi_rho and chi_T become dominated
# by the wrong electron contribution, pushing nabla_ad far from the physical
# regime. This discriminates nabla_ad tests where zero-table does not.
# Reference: (full FD electron EOS table), (CI fix).
@register("fd_electron_scale_table")
def _fd_electron_scale_table(mp, stellar_mod=None):
    """Scale FD electron free energy by 10×, corrupting all thermodynamic derivatives."""
    import jax.numpy as jnp
    import stellar_jax.microphysics.fd_electron as fd_mod

    # Save the real function
    _real_F_electron = fd_mod._F_electron_fd

    # Replace with 10× scaled version — grossly wrong P_e, chi_rho, chi_T, nabla_ad
    def _scaled_F_electron(ln_rho, ln_T, X, Z):
        return 10.0 * _real_F_electron(ln_rho, ln_T, X, Z)

    mp.setattr(fd_mod, "_F_electron_fd", _scaled_F_electron, raising=True)


# ─── zero_brunt_nonradial: force ll1=0 in the GYRE nonradial RHS ────────────────────
# The GYRE nonradial equations (ad_eqns_m.fypp) depend on the angular degree via
# ll1 = l*(l+1) in two critical terms: λ/(c₁σ²) = ll1·A1/σ² (which couples y1↔y2↔y3)
# and the ll1·y3 term in dy4 (gravitational restoring force).  Forcing ll1=0 removes
# the angular-degree-dependent physics from the RHS WHILE the BCs still carry l=2 →
# the resulting degenerate system produces "modes" that don't pair correctly with l=0
# modes to give valid δν₀₂ values → the test fails at the mode-pairing assertion.
# This mutation patches the CURRENT GYRE-formulation RHS (post-) and correctly
# breaks the δν₀₂ trend test. It supersedes force_l_zero_in_nonradial, which was
# written for the pre- ADIPLS A-formulation and is now stale (the old ADIPLS RHS
# with ll1=0 accidentally creates a hybrid system that still produces monotonic modes).
# Reference: Tassoul (1980) — δν₀₂ ∝ ∫(dc/dr)/r dr; Townsend & Teitler (2013) GYRE.
@register("zero_brunt_nonradial")
def _zero_brunt_nonradial(mp, stellar_mod=None):
    """Force ll1=0 in the GYRE nonradial RHS — destroys l-dependent mode coupling."""
    import jax.numpy as jnp
    import stellar_jax.oscillations.integrator

    def _nonradial_rhs_ll1_zero(x_val, y, sigma2, l, x_grid, coeffs):
        """GYRE nonradial RHS with forced ll1=0 — removes angular degree dependence."""
        from stellar_jax.oscillations.integrator import _interp_coeffs
        Vg, A1, A_bv, U, _ = _interp_coeffs(x_val, x_grid, coeffs)
        # MUTATION: force ll1=0 regardless of actual l
        ll1 = 0.0
        As = A_bv
        lam_c1w2 = 0.0  # ll1 * A1 / sigma2 = 0
        c1w2 = sigma2 / jnp.maximum(A1, 1e-30)

        y1, y2, y3, y4 = y[0], y[1], y[2], y[3]

        dy1 = ((Vg - 1.0 - l) * y1
               + (lam_c1w2 - Vg) * y2
               + lam_c1w2 * y3) / x_val
        dy2 = ((c1w2 - As) * y1
               + (As - U + 3.0 - l) * y2
               - y4) / x_val
        dy3 = ((3.0 - U - l) * y3
               + y4) / x_val
        dy4 = (U * As * y1
               + U * Vg * y2
               + ll1 * y3
               + (2.0 - U - l) * y4) / x_val
        return jnp.array([dy1, dy2, dy3, dy4])

    mp.setattr(oscillations.integrator, "_nonradial_rhs", _nonradial_rhs_ll1_zero)


# ─── detach_z_in_kappa: sever the Z → kappa gradient path ───────────────
# Z enters kappa() through both OPAL and Ferguson interpolation tables. If we
# stop_gradient Z before it reaches the opacity lookup, ∂kappa/∂Z = 0 in the
# backward pass, but the forward value is unchanged. This severs the dominant
# path Z → κ → ∇_rad → T_eff that contributes most of ∂logTeff/∂Z. The remaining
# Z → ε_CNO path contributes <10% of the total at N=50 (opacity dominates).
# Any test that asserts a robustly large ∂logTeff/∂Z (AD magnitude > 0.01 and
# AD-vs-FD agreement < 30%) will FAIL because AD underestimates by >>50%.
# Ref: MESA kap/private/kap_eval_fixed.f90:143 — Z enters κ interpolation.
@register("detach_z_in_kappa")
def _detach_z_in_kappa(mp, stellar_mod=None):
    """Detach Z from the gradient tape inside kappa — ∂kappa/∂Z = 0.

    Wraps kappa() to apply stop_gradient on Z before the opacity interpolation.
    The forward value is unchanged (Z still indexes the correct table), but the
    AD gradient ∂kappa/∂Z = 0, severing the dominant Z→Teff path.
    """
    import jax
    import jax.numpy as jnp
    import stellar_jax.solver.residual
    import stellar_jax.solver.surface_bc
    import stellar_jax.solver.shell_data
    from stellar_jax.microphysics import opacity as opacity_mod
    _orig_kappa = opacity_mod.kappa

    def _z_detached_kappa(logT, logRho, X, Z, opacity_factor=None):
        # Apply stop_gradient to Z: forward unchanged, backward ∂kappa/∂Z = 0
        Z_sg = jax.lax.stop_gradient(Z)
        return _orig_kappa(logT, logRho, X, Z_sg, opacity_factor=opacity_factor)

    mp.setattr(opacity_mod, "kappa", _z_detached_kappa, raising=True)
    # Patch at all solver usage sites
    mp.setattr(solver.residual, "kappa", _z_detached_kappa, raising=True)
    mp.setattr(solver.surface_bc, "kappa", _z_detached_kappa, raising=True)
    mp.setattr(solver.shell_data, "kappa", _z_detached_kappa, raising=True)


# ─── helm_blend_disabled: revert eos_lookup to pure OPAL (no HELM blend) ───
# The production eos_lookup blends OPAL with the full Fermi-Dirac HELM table
# at strongly-degenerate conditions (logT>7.7, logRho>5.5). This mutation replaces
# eos_lookup with eos_opal_only at ALL usage sites (the source module, the
# solver modules that hold their own import bindings, and the structure module).
# Under this mutation, the EOS at RGB-tip degenerate conditions returns the
# OPAL-extrapolated values (clamped, ideal-gas Gamma1≈1.9 instead of the
# correct ~1.58), causing any test that asserts correct degenerate
# thermodynamics to FAIL (20%+ Gamma1 error).
# Ref: MESA eosdt_eval.f90:749 (get_HELM_alfa — MESA uses HELM at degeneracy);
# (wire HELM into production), (analytic form inadequacy).
@register("helm_blend_disabled")
def _helm_blend_disabled(mp, stellar_mod=None):
    """Revert eos_lookup to pure OPAL (no HELM blend) — breaks degenerate EOS tests.

    Patches eos_lookup at all usage sites with eos_opal_only (the OPAL-only
    table interpolation, no HELM contribution). At degenerate conditions
    (logT>7.7, logRho>5.5), this gives Gamma1≈1.9 instead of ~1.58 (20% error),
    chi_rho≈1.0 instead of ~1.5, and density clamped at logRho≈5.69 instead of
    the correct ~5.97. Any test asserting correct degenerate thermodynamics FAILS.
    """
    from stellar_jax.microphysics.eos import eos_opal_only
    import stellar_jax.microphysics.eos
    import stellar_jax.solver.residual
    import stellar_jax.solver.jacobian
    import stellar_jax.solver.shell_data
    import stellar_jax.solver.surface_bc
    import stellar_jax.solver.eps_grav

    # Patch the source module
    mp.setattr(microphysics.eos, "eos_lookup", eos_opal_only, raising=True)
    # Patch all solver modules that hold their own import binding
    mp.setattr(solver.residual, "eos_lookup", eos_opal_only, raising=True)
    mp.setattr(solver.jacobian, "eos_lookup", eos_opal_only, raising=True)
    mp.setattr(solver.shell_data, "eos_lookup", eos_opal_only, raising=True)
    mp.setattr(solver.surface_bc, "eos_lookup", eos_opal_only, raising=True)
    mp.setattr(solver.eps_grav, "eos_lookup", eos_opal_only, raising=True)


# ═══════════════════════════════════════════════════════════════
# Atmosphere BC mutations
# ═══════════════════════════════════════════════════════════════

# ─── corrupt_ks_atmosphere_seed: corrupt KS T(τ) coefficients ───
# Krishna Swamy (1966, ApJ 145, 174) q(τ) = 1.39 - 0.815·exp(-2.54·τ) - 0.025·exp(-30·τ).
# Corrupting Q1=1.39 to 0.0 shifts the seed T by ~10% and propagates through the integration.
# MESA: atm/private/atm_t_tau_relations.f90:148-152 (same coefficients).
#
# NOTE: a disable_atm_pextra mutation was considered but NOT included because:
# the Pextra term (P_rad_surf = g*L/(6π*c*G*M)) has <1e-7 relative effect on
# atmosphere_bc OUTPUT at tau=100 (P grows ~1e6× during integration, washing it
# out completely). No output-comparing test can detect its removal. The formula
# correctness is verified algebraically in test_atmosphere_bc_radiation_pressure_pextra
# (a @fast @smoke code-review guard). See docs/design/atmosphere-bc-method-audit.md.
@register("corrupt_ks_atmosphere_seed")
def _corrupt_ks_atmosphere_seed(mp, stellar_mod=None):
    """Corrupt the KS q(τ) Q1 constant from 1.39 to 1.0 — a bounded perturbation."""
    import stellar_jax.structure as struct_mod
    import jax.numpy as jnp
    from jax import lax
    from stellar_jax.microphysics.eos import eos_opal_only
    from stellar_jax.microphysics.opacity import kappa
    from stellar_jax.transport import mlt_nabla
    from stellar_jax.config.constants import G, a_rad, c_light

    def _atm_bad_seed(Te, g_surf, X, Z, alpha_mlt, L_star, M_star, tau_base=100.0):
        """atmosphere_bc with corrupted KS Q1 constant (mutation)."""
        N_PHASE1 = 20
        N_PHASE2 = 180
        tau_start = 1e-4
        tau_mid = 1.0
        ln_tau_start = jnp.log(tau_start)
        ln_tau_mid = jnp.log(tau_mid)
        ln_tau_end = jnp.log(tau_base)
        d_ln_tau_1 = (ln_tau_mid - ln_tau_start) / N_PHASE1
        d_ln_tau_2 = (ln_tau_end - ln_tau_mid) / N_PHASE2

        # CORRUPTED: Q1=1.0 instead of 1.39 — a bounded perturbation that keeps
        # q(τ) positive (no NaN) but shifts the seed T by ~26% at τ_start,
        # propagating through the full atmosphere ODE integration to exceed
        # the 1% test tolerance on T(τ=100).
        # (Q1=0.0 makes q negative at small τ → NaN → crashes before the
        # physics assertion can fire — that was the over-destructive bug.)
        q_init = 1.0 - 0.815 * jnp.exp(-2.54 * tau_start) - 0.025 * jnp.exp(-30.0 * tau_start)
        T_init = Te * (0.75 * (tau_start + q_init)) ** 0.25
        logT_init = jnp.log10(T_init)
        rho_guess = 1e-9
        log_kap_init = kappa(logT_init, jnp.log10(jnp.float64(rho_guess)), X, Z)
        kap_init = 10.0 ** log_kap_init
        P_hydro = tau_start * g_surf / kap_init
        P_rad_surf = g_surf * L_star / (6.0 * jnp.pi * c_light * G * M_star)
        P_init = jnp.maximum(P_hydro + P_rad_surf, 1.0)

        def _atm_step(carry, _, d_ln_tau):
            ln_P, ln_T, ln_tau = carry
            tau = jnp.exp(ln_tau)
            P = jnp.exp(ln_P)
            T = jnp.exp(ln_T)
            logT = jnp.log10(T)
            logP = jnp.log10(P)
            rho, mu_l, nad, *_ = eos_opal_only(logT, logP, X, Z)
            log_kap = kappa(logT, jnp.log10(rho), X, Z)
            kap = 10.0 ** log_kap
            dlnP_dlntau = tau * g_surf / (kap * P + 1e-30)
            nabla_rad = 3.0 * kap * L_star * P / \
                        (16.0 * jnp.pi * a_rad * c_light * G * M_star * T ** 4 + 1e-30)
            nabla = mlt_nabla(nabla_rad, nad, T, P, rho, kap, g_surf, mu_l, alpha_mlt)
            dlnT_dlntau = nabla * dlnP_dlntau
            ln_P_new = ln_P + dlnP_dlntau * d_ln_tau
            ln_T_new = ln_T + dlnT_dlntau * d_ln_tau
            ln_tau_new = ln_tau + d_ln_tau
            return (ln_P_new, ln_T_new, ln_tau_new), None

        step1 = lambda carry, x: _atm_step(carry, x, d_ln_tau_1)
        init = (jnp.log(P_init), jnp.log(T_init), ln_tau_start)
        (ln_P_mid, ln_T_mid, ln_tau_at_mid), _ = lax.scan(step1, init, None, length=N_PHASE1)
        step2 = lambda carry, x: _atm_step(carry, x, d_ln_tau_2)
        (ln_P_final, ln_T_final, _), _ = lax.scan(
            step2, (ln_P_mid, ln_T_mid, ln_tau_at_mid), None, length=N_PHASE2)
        P_base = jnp.maximum(jnp.exp(ln_P_final), 1.0)
        T_base = jnp.exp(ln_T_final)
        return P_base, T_base

    mp.setattr(struct_mod, "atmosphere_bc", _atm_bad_seed, raising=True)
    import stellar_jax.solver.surface_bc
    mp.setattr(solver.surface_bc, "atmosphere_bc", _atm_bad_seed, raising=True)


# ─── seismic_npg_offset: break n_pg identification in the seismic API ──────
# The assign_n_pg function uses ε to determine radial orders. Adding +2 to ε
# shifts all n_pg assignments by −2, making every ratio/separation pair the wrong
# modes. This breaks all n_pg-dependent diagnostics (r02, r01, r10, δν₀₂, δν₀₁)
# because the numerator uses ν(n,0) − ν(n−1,2) but with wrong n assignments.
# Reference: Tassoul (1980), ApJS 43, 469 (the asymptotic relation being corrupted).
@register("seismic_npg_offset")
def _seismic_npg_offset(mp, stellar_mod=None):
    """Corrupt ε estimation by +2 → all n_pg assignments shift by −2."""
    import stellar_jax.oscillations.seismic_quantities

    _orig_estimate = oscillations.seismic_quantities._estimate_epsilon_from_l0

    def _patched_estimate(freqs_l0, delta_nu):
        return _orig_estimate(freqs_l0, delta_nu) + 2.0

    mp.setattr(oscillations.seismic_quantities, "_estimate_epsilon_from_l0",
               _patched_estimate, raising=True)


# ─── seismic_acoustic_radius_corrupt: break c_s in large_separation ────────
# large_separation computes c_s = sqrt(Γ₁·P/ρ), then τ = ∫dr/c_s, then Δν = 1e6/(2τ).
# This mutation patches it to use c_s = sqrt(P/ρ) (omits Γ₁ ≈ 5/3), which makes c_s
# ~25% too low → τ ~25% too high → Δν ~25% too low, far exceeding the 2% tolerance.
# This is a realistic break: it tests that Γ₁ is correctly included in the sound speed.
# Reference: MESA report.f90:343 (the formula being validated).
@register("seismic_acoustic_radius_corrupt")
def _seismic_acoustic_radius_corrupt(mp, stellar_mod=None):
    """Drop Γ₁ from the sound-speed in large_separation → Δν wrong by ~25%."""
    import stellar_jax.oscillations.seismic_quantities
    import numpy as np

    def _large_separation_no_gamma1(glob, var):
        """Deliberately wrong: c_s = sqrt(P/ρ) instead of sqrt(Γ₁·P/ρ)."""
        r = var[:, 0]
        P = var[:, 3]
        rho = var[:, 4]
        # MUTATION: omit gamma1 (var[:,9]) — c_s is ~sqrt(5/3) ≈ 1.29× too low
        c_s = np.sqrt(P / rho)
        dr = np.diff(r)
        inv_cs = 1.0 / c_s
        avg_inv_cs = 0.5 * (inv_cs[:-1] + inv_cs[1:])
        acoustic_radius = float(np.sum(avg_inv_cs * dr))
        if acoustic_radius <= 0:
            raise ValueError("acoustic_radius ≤ 0")
        return 1.0e6 / (2.0 * acoustic_radius)

    mp.setattr(oscillations.seismic_quantities, "large_separation",
               _large_separation_no_gamma1, raising=True)

# ─── disable_surface_correction: zero out BG14 surface correction coefficients ───
# The Ball & Gizon (2014) two-term surface correction fits (a1, a3) by weighted
# least-squares and applies δν = (a1·(ν/ν_ac)⁻¹ + a3·(ν/ν_ac)³)/I. This mutation
# replaces surface_correction_bg14 with a stub that returns zero correction,
# making "corrected" frequencies identical to uncorrected. Any test that asserts
# the correction reduces bias or produces specific coefficients will FAIL.
# Ref: Ball & Gizon (2014), A&A 568, A123, Eq. 4;.
@register("disable_surface_correction")
def _disable_surface_correction(mp, stellar_mod=None):
    """Zero the BG14 surface correction — corrected freqs = uncorrected.

    Patches surface_correction_bg14 to return a1=a3=0 and delta_nu=0,
    so the correction has no effect. Tests that rely on the correction
    absorbing a surface term will FAIL (chi2 remains at uncorrected level).
    """
    import jax.numpy as jnp
    import stellar_jax.oscillations.surface_correction

    def _stub_correction(nu_model, nu_obs, sigma, nu_ac, inertia=None):
        nu_model = jnp.asarray(nu_model, dtype=jnp.float64)
        return {
            'nu_corrected': nu_model,  # NO correction applied
            'a1': jnp.float64(0.0),
            'a3': jnp.float64(0.0),
            'delta_nu': jnp.zeros_like(nu_model),
        }

    mp.setattr(oscillations.surface_correction, "surface_correction_bg14",
               _stub_correction, raising=True)

    # Also patch chi2_with_surface_correction since it calls surface_correction_bg14
    # internally — if it has its own import binding, it won't see the patch above.
    # Re-import the module-level function reference:
    def _stub_chi2(nu_model, nu_obs, sigma, nu_ac, inertia=None):
        nu_model = jnp.asarray(nu_model, dtype=jnp.float64)
        nu_obs = jnp.asarray(nu_obs, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        residuals = (nu_model - nu_obs) / sigma
        return jnp.sum(residuals**2)

    mp.setattr(oscillations.surface_correction, "chi2_with_surface_correction",
               _stub_chi2, raising=True)


# ─── corrupt_glitch_acoustic_depth: break the sound speed in acoustic depth calculation ───
# A realistic break: if Γ₁ is replaced by a constant (5/3) in the acoustic depth
# calculation, the He II ionization zone depression is invisible in τ(r), and
# the fitted glitch parameters shift significantly. Also breaks the absolute T.
@register("corrupt_glitch_acoustic_depth")
def _corrupt_glitch_acoustic_depth(mp, stellar_mod=None):
    """Corrupt sound speed in glitch acoustic depth by using constant Γ₁ = 5/3.

    The He II ionization zone is characterized by a Γ₁ dip (from 5/3 down to ~1.19).
    If we compute acoustic depth with constant Γ₁ = 5/3, the local variation is lost,
    the acoustic radius T changes by ~1-2%, and the second-difference fitting finds
    wrong τ_He (the phase of the He II signal no longer aligns with the true 700 s).

    More directly: we corrupt the _glitch_model_D0 fitting function to use a WRONG
    phase convention (multiply tau by 0.5), which makes the fitted tau_He and tau_cz
    land far outside the published solar values.
    """
    import stellar_jax.oscillations.glitch
    import numpy as np

    _orig_model = oscillations.glitch._glitch_model_D0

    def _patched_model(nu, params):
        # Corrupt: halve the acoustic depth in the phase → fitted tau doubles
        params_corrupt = dict(params)
        params_corrupt['tau_He'] = params['tau_He'] * 0.5
        params_corrupt['tau_cz'] = params['tau_cz'] * 0.5
        return _orig_model(nu, params_corrupt)

    mp.setattr(oscillations.glitch, "_glitch_model_D0", _patched_model)


# ─── corrupt_glitch_gamma1_constant: replace Γ₁ with 5/3 in acoustic depth computation ───
# Tests that the acoustic depth calculation is sensitive to the actual Γ₁ structure.
# With constant Γ₁=5/3, the He II dip is erased and T changes; with a 2× scale the
# acoustic radius drops by 1/sqrt(2) → T ≈ 2540 s, well below the 3400 s lower bound.
@register("corrupt_glitch_gamma1_constant")
def _corrupt_glitch_gamma1_constant(mp, stellar_mod=None):
    """Replace Γ₁ with 2×Γ₁ in compute_acoustic_depth_profile, erasing the He II signal.

    Physically: doubling Γ₁ raises c by √2 everywhere, reducing acoustic radius T by
    a factor of 1/√2 (~0.71). For Model S T≈3593 s, this gives T≈2540 s — well below
    the [3400, 3800] acceptance bound. Demonstrates that the T assertion is sensitive to
    the actual Γ₁ profile, not merely "any positive number".
    """
    import stellar_jax.oscillations.glitch
    import numpy as np

    _orig_fn = oscillations.glitch.compute_acoustic_depth_profile

    def _patched_fn(glob, var):
        # Double gamma1 (col 9) → csound up by √2 → T down by 1/√2
        var_patched = var.copy()
        var_patched[:, 9] = var_patched[:, 9] * 2.0
        return _orig_fn(glob, var_patched)

    mp.setattr(oscillations.glitch, "compute_acoustic_depth_profile", _patched_fn)



# ─── helm_density_solve_disabled: disable Newton convergence in HELM density solve ───
# Sets _NR_MAX_ITER=0 in the HELM density inversion (fd_electron.py), causing
# the lax.while_loop to exit immediately without any Newton iteration. The
# returned density is just the initial guess (ideal gas or warm-start), which
# is far from the true EOS root in the degenerate regime. Any test that
# checks convergence, gradient accuracy, or degenerate thermodynamics will
# fail because the density is ~10-100× wrong in the He core.
# Ref: (convergence-based while_loop); MESA eosdt_eval.f90:2359
# do_safe_get_Rho_T (max_iter=20 with logRho_tol convergence).
@register("helm_density_solve_disabled")
def _helm_density_solve_disabled(mp, stellar_mod=None):
    """Set _NR_MAX_ITER=0 in the HELM density Newton solve, disabling convergence."""
    import stellar_jax.microphysics.fd_electron as fd_mod
    mp.setattr(fd_mod, "_NR_MAX_ITER", 0)


# ─── newton_cap_one_iteration: force Newton while_loop to exit after 1 iteration ───
# Caps the maximum Newton iterations in the production Henyey continuation solver
# (_henyey_continuation_atm) to 1. With only one iteration, the Newton solve
# cannot converge (typical convergence requires ~5–15 iterations at tol=1e-4),
# so the returned y_final is far from the fixed point. Any test that checks
# bit-identity between scan and while_loop, or Newton convergence, will FAIL
# because the while_loop exits after 1 iteration while a scan(length=100)
# runs the full 100 (the scan freezes dy=0 after convergence, so it converges
# correctly; 1 iteration cannot).
# Ref: (convergence-gated while_loop); MESA star_solver.f90:325
# iter_loop: do while (.not. passed_tol_tests) — the safety cap is max_tries.
@register("newton_cap_one_iteration")
def _newton_cap_one_iteration(mp, stellar_mod=None):
    """Patch henyey_solve_from_state_atm to pass n_iter=1, capping Newton at 1 iteration."""
    from stellar_jax.solver import continuation as cont_mod
    _original = cont_mod.henyey_solve_from_state_atm.__wrapped__ if hasattr(
        cont_mod.henyey_solve_from_state_atm, '__wrapped__') else cont_mod.henyey_solve_from_state_atm

    def _capped(*args, **kwargs):
        kwargs['n_iter'] = 1
        return _original(*args, **kwargs)

    mp.setattr(cont_mod, "henyey_solve_from_state_atm", _capped)
    # Also patch the henyey.py re-export
    try:
        import stellar_jax.henyey as henyey
        mp.setattr(henyey, "henyey_solve_from_state_atm", _capped)
    except (ImportError, AttributeError):
        pass

# ═══════════════════════════════════════════════════════════════════════════════
# PARTIAL-MUTATION TIER (AC3)
#
# The standard mutations are total breaks (sign-flip, zeroing, 10×).  These
# partial mutations are SCALED/DEGRADATION breaks (×0.5, +50%, ±30%) that test
# whether the widest-band gates catch realistic DEGRADATIONS, not just total
# failures.  The partial-mutation tier proves that even a halved-strength
# physics error is detectable by the validation suite.
#
# Each partial mutation targets one of the widest-tolerance tests:
#   - Z-gradient (60-75% tolerance) — partial_detach_z_gradient
#   - r02 ratio (15% tolerance) — partial_corrupt_gamma1
#   - eps_nuc (10% median tolerance) — partial_eps_nuc_scale
# ═══════════════════════════════════════════════════════════════════════════════

@register("partial_detach_z_gradient")
def _partial_detach_z_gradient(mp, stellar_mod=None):
    """Partially detach Z from the gradient tape — scale Z sensitivity by 0.10×.

    Unlike detach_z_gradient (which fully severs Z→AD=0), this mutation
    retains only 10% of the Z gradient by mixing the true Z with a
    stop_gradient copy:
      Z_effective = 0.10 * Z + 0.90 * stop_gradient(Z)
    This gives AD = 0.10 × true_AD, a 90% degradation.

    Detection arithmetic (structurally airtight at tol ≤ 0.75):
      Let α = clean AD/FD ratio.  Clean passes ⟹ |1−α| < tol, so
      α ∈ [1−tol, 1+tol].
      Mutated rel_err = |1 − 0.10·α|.
      Worst case (α = 1+tol = 1.75 at tol=0.75):
        |1 − 0.10·1.75| = |1 − 0.175| = 0.825 > 0.75.
      So for ANY α in the clean-passing range, the mutated rel_err
      exceeds tol — the rel_err gate catches the mutation unconditionally.

      The prior 0.25× factor was theater (#1189): with grad_window, the
      windowed AD can OVERSHOOT FD (α > 1), and at α ∈ (1.0, 1.75) the
      mutated rel_err |1−0.25α| fell BELOW 0.75, letting the test pass.
      0.10× eliminates this loophole because 0.10·1.75 = 0.175 ≪ 1.

    Tests gated by this mutation:
      - N=50 full-backward (tol_L=0.60): α ≤ 1.60, worst-case mutated
        rel_err = |1−0.16| = 0.84 > 0.60.  Caught.
      - N=100 grad_window (tol_L=0.75): α ≤ 1.75, worst-case mutated
        rel_err = 0.825 > 0.75.  Caught.

    The sign/trip-wire assertions (|AD| > 1e-6, AD*FD > 0) still pass
    — the gradient is non-zero and same-sign, just severely attenuated.
    The test fails on the TOLERANCE assertion (the physics gate), not
    an incidental guard.

    The 0.10× factor is within the sound-mutation band (0.01×..100×)
    and represents a 90% gradient loss — an unambiguous physics break
    (a plausible scenario: 90% of Z's gradient path severed by a
    misplaced stop_gradient or dead code).

    Issue #824 AC3: partial-mutation tier.  Fixed #1189 (theater).
    """
    import jax
    import jax.numpy as jnp
    import stellar_jax.evolution as evolution

    _orig_evolve = evolution.evolve_star

    def _partial_z_evolve(*args, **kwargs):
        """Forward correct, backward path through Z ×0.10."""
        if len(args) >= 2:
            args = list(args)
            Z_real = jnp.asarray(args[1], dtype=jnp.float64)
            # Mix: 0.10 * live_Z + 0.90 * dead_Z → gradient is 10% of true
            args[1] = 0.10 * Z_real + 0.90 * jax.lax.stop_gradient(Z_real)
            args = tuple(args)
        elif 'Z' in kwargs:
            Z_real = jnp.asarray(kwargs['Z'], dtype=jnp.float64)
            kwargs['Z'] = 0.10 * Z_real + 0.90 * jax.lax.stop_gradient(Z_real)
        return _orig_evolve(*args, **kwargs)

    mp.setattr(evolution, "evolve_star", _partial_z_evolve)
    if stellar_mod is not None:
        mp.setattr(stellar_mod, "evolve_star", _partial_z_evolve)


@register("partial_corrupt_gamma1")
def _partial_corrupt_gamma1(mp, stellar_mod=None):
    """Scale Γ₁ by +10% (1.10×) in the JAX oscillation grid — a partial break.

    The full corrupt_gamma1_jax scales Γ₁ by 1.5 (50%), shifting frequencies
    by ~22%. This partial mutation scales by only 1.10 (10%), shifting
    frequencies by ~5%. The r02 test (15% tolerance) should catch this
    because the frequency shift propagates differently for l=0 vs l=2,
    changing r02 by more than 15%.

    For l=2 modes closer to the core (higher sensitivity to Γ₁ in the
    Brunt-Väisälä region), a 10% Γ₁ change shifts δν₀₂ by ~20-30%
    (the small separation is a DIFFERENCE of frequencies, amplifying
    relative errors). The r02 ratio, being δν₀₂/Δν₁, inherits this.

    Issue #824 AC3: partial-mutation tier.
    """
    import stellar_jax.oscillations.coefficients
    import stellar_jax.oscillations.eigenvalue

    _orig_build_jax = oscillations.coefficients._build_oscillation_grid_jax

    def _patched_build_jax(glob, var):
        import numpy as np
        var_corrupted = var.copy()
        # +10% Γ₁ corruption (column 9)
        var_corrupted[:, 9] = var_corrupted[:, 9] * 1.10
        return _orig_build_jax(glob, var_corrupted)

    mp.setattr(oscillations.coefficients, "_build_oscillation_grid_jax", _patched_build_jax)
    mp.setattr(oscillations.eigenvalue, "_build_oscillation_grid_jax", _patched_build_jax)


@register("partial_eps_nuc_scale")
def _partial_eps_nuc_scale(mp, stellar_mod=None):
    """Multiply epsilon_nuclear by 1.5× — a 50% error, partial break.

    The full eps_nuc_scale_5x uses 5× (400% error), which overwhelms the 10%
    tolerance. This 1.5× (50% error) tests whether the median tolerance <10%
    catches a more realistic microphysics error (e.g. a wrong screening
    correction or rate coefficient at the 50% level).

    50% > 10% tolerance → test should still fail, proving the gate is not
    set so wide that only catastrophic errors are caught.

    Patches epsilon_nuclear at ALL usage sites:
      - microphysics.nuclear (definition site)
      - structure.py (shooting solver / diagnostic path)
      - solver/residual.py, solver/jacobian.py, solver/shell_data.py
        (production Henyey solver, called via evolution.driver)
    Patching only the source module has no effect on the Henyey solver
    because each solver module holds its own import binding via
    ``from stellar_jax.microphysics.nuclear import epsilon_nuclear``.
    Pattern matches eps_nuc_zero_all (#1077).

    Issue #824 AC3: partial-mutation tier.
    """
    from stellar_jax.microphysics import nuclear as nuc_mod
    import stellar_jax.structure as structure
    import stellar_jax.solver.residual
    import stellar_jax.solver.jacobian
    import stellar_jax.solver.shell_data
    _orig_eps = nuc_mod.epsilon_nuclear

    # *args absorbs extra positional args (t_age passed as 5th positional by
    # structure.py, solver/residual.py, solver/jacobian.py, solver/shell_data.py);
    # **kwargs absorbs keyword args (X3, X3_eq_frozen, X_N14, eps_nuc_factor, etc.).
    # Together they prevent signature drift from re-breaking O2 coverage.
    def _scaled_eps(*args, **kwargs):
        return _orig_eps(*args, **kwargs) * 1.5

    mp.setattr(nuc_mod, "epsilon_nuclear", _scaled_eps, raising=True)
    mp.setattr(structure, "epsilon_nuclear", _scaled_eps, raising=True)
    mp.setattr(solver.residual, "epsilon_nuclear", _scaled_eps, raising=True)
    mp.setattr(solver.jacobian, "epsilon_nuclear", _scaled_eps, raising=True)
    mp.setattr(solver.shell_data, "epsilon_nuclear", _scaled_eps, raising=True)
    if stellar_mod is not None:
        mp.setattr(stellar_mod, "epsilon_nuclear", _scaled_eps, raising=True)


# ─── f_ov_detach: sever the f_ov → composition gradient ──────────────
# f_ov enters mix_composition (and _mix_z_in_cz) through _core_conv_mask, where
# it controls the overshoot extent: f_ov_effective * Hp * dm/dr = delta_m_ov.
# If we stop_gradient f_ov before it enters mix_composition, the forward result
# is unchanged but the backward ∂composition/∂f_ov = 0 → any downstream
# gradient w.r.t. f_ov ≈ 0.
# MESA ref: overshoot_step.f90:eval_overshoot_step (f = s%overshoot_f(j));
#           overshoot_utils.f90:eval_conv_bdy_Hp.
@register("f_ov_detach")
def _f_ov_detach(mp, stellar_mod=None):
    """Detach f_ov from the gradient tape — ∂composition/∂f_ov = 0.

    Wraps mix_composition and _mix_z_in_cz to apply stop_gradient on f_ov
    before it is used. The forward value is unchanged (overshoot still operates),
    but the AD gradient ∂composition/∂f_ov = 0.

    This breaks any test that asserts a nonzero gradient w.r.t. f_ov.
    """
    import jax
    from stellar_jax.composition import mix as mix_mod
    from stellar_jax.config.mesh_defaults import F_OV

    _orig_mix = mix_mod.mix_composition
    _orig_mix_z = mix_mod._mix_z_in_cz

    def _detached_mix(X_profile, shell_data, M_solar=1.0, f_ov=None, **kwargs):
        if f_ov is None:
            f_ov = F_OV
        f_ov = jax.lax.stop_gradient(f_ov)
        return _orig_mix(X_profile, shell_data, M_solar=M_solar, f_ov=f_ov, **kwargs)

    def _detached_mix_z(Z_profile, shell_data, M_solar, f_ov, **kwargs):
        f_ov = jax.lax.stop_gradient(f_ov)
        return _orig_mix_z(Z_profile, shell_data, M_solar, f_ov, **kwargs)

    mp.setattr(mix_mod, "mix_composition", _detached_mix, raising=True)
    mp.setattr(mix_mod, "_mix_z_in_cz", _detached_mix_z, raising=True)
    mp.setattr(mix_mod, "mix_z_in_cz", _detached_mix_z, raising=True)
    # Patch at the evolution usage site (holds its own import binding)
    import stellar_jax.evolution._core as core_mod
    mp.setattr(core_mod, "mix_composition", _detached_mix, raising=True)
    mp.setattr(core_mod, "_mix_z_in_cz", _detached_mix_z, raising=True)
    # Patch at the lax.scan step-body call site: step.py has its own
    # `from stellar_jax.composition.mix import mix_composition, _mix_z_in_cz`
    # binding (line 33). Without this patch, the stop_gradient on f_ov never
    # takes effect in the actual evolution loop — the mutation was a no-op on
    # the real code path, making every f_ov_detach-gated test pass vacuously.
    import stellar_jax.evolution.step as step_mod
    mp.setattr(step_mod, "mix_composition", _detached_mix, raising=True)
    mp.setattr(step_mod, "_mix_z_in_cz", _detached_mix_z, raising=True)


# ─── disable_nan_safe_lm: revert _lm_step NaN guard to dead try/except ───
# The NaN-safe guard in _lm_step detects when jnp.linalg.solve returns NaN on a
# singular (J^T J + λI) and returns status='singular' (reject step, raise λ).
# Before this fix, a dead try/except caught Python exceptions that JAX never raises,
# so NaN silently propagated through theta and covariance.
# This mutation restores the dead-code behavior: _lm_step skips the NaN check on
# delta, allowing NaN to propagate. A test asserting 'singular' status and finite
# theta/cov MUST FAIL under this mutation.
# Ref: Press NR §15.5 (singular-matrix branch);.
@register("disable_nan_safe_lm")
def _disable_nan_safe_lm(mp, stellar_mod=None):
    """Disable the NaN-safe guard in _lm_step — NaN from singular solve propagates.

    Replaces _lm_step with a version that skips the jnp.isnan(delta) check.
    On a rank-deficient Jacobian, jnp.linalg.solve returns NaN for delta, and
    without the guard, that NaN flows into theta_trial → chi2_trial → the
    solver returns NaN theta/cov instead of a clean 'singular' rejection.
    """
    import stellar_jax.inference.solver as solver_mod
    import jax.numpy as jnp
    import numpy as np

    def _lm_step_no_nan_guard(theta, r, chi2, lam, jacobian_fn, residuals_fn,
                              bounds_lo, bounds_hi, lambda_up, lambda_down):
        """_lm_step WITHOUT the NaN guard — reproduces the pre-fix dead try/except."""
        J = jacobian_fn(theta)
        JtJ = J.T @ J
        Jtr = J.T @ r
        diag_JtJ = jnp.diag(jnp.maximum(jnp.diag(JtJ), 1e-30))
        A = JtJ + lam * diag_JtJ
        try:
            delta = jnp.linalg.solve(A, -Jtr)
        except Exception:
            # Dead code — JAX never raises on singular matrices
            return theta, r, chi2, lam * lambda_up, None, 'singular'
        # NO NaN check — NaN propagates through
        theta_trial = jnp.clip(theta + delta, bounds_lo, bounds_hi)
        r_trial = residuals_fn(theta_trial)
        chi2_trial = float(jnp.sum(r_trial ** 2))
        # NO chi2_trial non-finite guard
        predicted = float(-2.0 * delta @ Jtr - delta @ JtJ @ delta)
        rho = ((chi2 - chi2_trial) / predicted) if predicted > 0 else (
            0.0 if chi2_trial < chi2 else -1.0)
        rel_step = float(jnp.max(jnp.abs(delta) / jnp.maximum(jnp.abs(theta), 1e-30)))
        info = {'chi2_trial': chi2_trial, 'rho': rho, 'rel_step': rel_step}
        if chi2_trial < chi2:
            if rho > 0.75:
                lam *= lambda_down
            elif rho < 0.25:
                lam *= lambda_up
            return theta_trial, r_trial, chi2_trial, lam, info, 'accept'
        return theta, r, chi2, lam * lambda_up, info, 'reject'

    mp.setattr(solver_mod, "_lm_step", _lm_step_no_nan_guard, raising=True)


@register("disable_envelope_mesh_functions")
def _disable_envelope_mesh_functions(mp, stellar_mod=None):
    """Disable the T_function1 and P_function mesh weights (issue #623).

    Sets MESH_T_FUNCTION1_WEIGHT and MESH_P_FUNCTION_WEIGHT to zero, reverting
    the mesh to composition-only equidistribution. Without the T and P terms,
    the envelope gets poor resolution: zones are concentrated only at the
    H-burning shell (xa_function), not in the envelope where l≥1 modes propagate.

    A test that asserts envelope zone concentration or l≥1 frequency accuracy
    on our own evolved structure will FAIL because the envelope resolution
    degrades to the pre-#623 state.

    MESA ref: controls.defaults:6089 (P_function_weight=40), :6102 (T_function1_weight=110).
    """
    import stellar_jax.evolution.adaptive.mesh_numpy as af_mod
    mp.setattr(af_mod, "MESH_T_FUNCTION1_WEIGHT", 0.0, raising=True)
    mp.setattr(af_mod, "MESH_P_FUNCTION_WEIGHT", 0.0, raising=True)


# ─── corrupt_fisher_inversion: break the Fisher matrix inversion ───
# Replaces _safe_invert_fisher with a function that returns an IDENTITY matrix
# instead of the true inverse. This makes forecast_sigma produce σ_i that are
# determined by the identity diagonal (not by the actual Fisher information),
# so the Fisher-vs-MCMC agreement test must FAIL.
@register("corrupt_fisher_inversion")
def _corrupt_fisher_inversion(mp, stellar_mod=None):
    """Replace Fisher inversion with identity → forecast σ ≠ true posterior σ."""
    import numpy as _np
    import stellar_jax.inference.fisher as fisher_mod

    def _identity_invert(F):
        """Return identity instead of true inverse — breaks Cramér-Rao."""
        return _np.eye(F.shape[0]), 'corrupted'

    mp.setattr(fisher_mod, "_safe_invert_fisher", _identity_invert, raising=True)


# ─── corrupt_diffusion_normalization: break the 4-species X+Y normalization ───
# After diffuse_composition evolves X, Y, Z independently, Step 6 pins
# X+Y to 1-Z_input (operator-split CONSTRAINT — see diffuse.py Step 6
# comment).  This mutation patches diffuse_composition to corrupt the
# returned X_new by scaling it up, breaking the X+Y = 1-Z_input invariant.
# A test asserting that invariant must FAIL under this mutation.
@register("corrupt_diffusion_normalization")
def _corrupt_diffusion_normalization(mp, stellar_mod=None):
    """Corrupt the X+Y normalization in diffuse_composition."""
    import stellar_jax.composition.diffuse as diffuse_mod
    _orig = diffuse_mod.diffuse_composition

    def _corrupt_normalization(*args, **kwargs):
        import jax.numpy as _jnp
        # Call original
        X_new, Y_new, Z_new = _orig(*args, **kwargs)
        # Corrupt: scale X by 1.02 to break X+Y = 1-Z_input invariant
        # (a realistic break: if normalization were skipped or wrong,
        # the X+Y sum would deviate from the normalization target)
        X_new = X_new * 1.02
        return X_new, Y_new, Z_new

    mp.setattr(diffuse_mod, "diffuse_composition", _corrupt_normalization, raising=True)
    # Also patch every module that re-exports diffuse_composition
    import stellar_jax.evolution as evolution_mod
    mp.setattr(evolution_mod, "diffuse_composition", _corrupt_normalization, raising=True)
    import stellar_jax.stellar as stellar_mod
    mp.setattr(stellar_mod, "diffuse_composition", _corrupt_normalization, raising=True)


# ─── flip_sign_jacobian_<param>: per-column sign-flip mutations ───
# Each mutation negates the AD gradient for ONE Jacobian column by wrapping
# _make_seismic_fn in the fisher module. When the wrapped function is
# jax.grad'd, the result is −∂σ²/∂θ_j instead of +∂σ²/∂θ_j. The FD
# gradient is unaffected, so the sign-check in _evaluate_mode_param must
# FAIL for exactly that column's test. This proves each column test is
# genuinely gated by its own mutation (not a shared machinery mutation).
#
# Design: wrapping the returned scalar function with a sign-flip is cleaner
# than patching deep inside the adjoint, because (a) it only affects the
# specific parameter column and (b) it does not interfere with the FD path.


def _make_flip_sign_mutation(target_param):
    """Factory: create a mutation that flips the AD gradient for one param."""

    def _mutation_fn(mp, stellar_mod=None):
        import stellar_jax.inference.fisher as fisher_mod

        _orig_make = fisher_mod._make_seismic_fn

        def _flipped_make(param_names, fiducial, param_idx, **kwargs):
            fn = _orig_make(param_names, fiducial, param_idx, **kwargs)
            pname = param_names[param_idx]
            if pname == target_param:
                # Wrap: negate σ² so that jax.grad gives −∂σ²/∂θ_j.
                def neg_fn(theta_j):
                    return -fn(theta_j)
                return neg_fn
            return fn

        mp.setattr(fisher_mod, "_make_seismic_fn", _flipped_make, raising=True)

    _mutation_fn.__doc__ = (
        f"Flip the sign of the AD gradient for ∂σ²/∂{target_param}. "
        f"The per-column test for {target_param} must FAIL under this mutation."
    )
    return _mutation_fn


for _param in ('Y_init', 'Z', 'alpha_mlt', 'f_ov'):
    _name = f"flip_sign_jacobian_{_param}"
    register(_name)(_make_flip_sign_mutation(_param))


# ─── corrupt_age_jacobian: break the age observable pathway ───
# Replaces the f_age scaling constant in plato_forecast with 0.0, making
# σ(age) = 0 regardless of the Fisher σ(M). Any test asserting σ(age) > 0
# or σ(age) within a physical range must FAIL.
@register("corrupt_age_jacobian")
def _corrupt_age_jacobian(mp, stellar_mod=None):
    """Zero the f_age scaling → σ(age) = 0, breaking the age forecast."""
    import stellar_jax.inference.plato_forecast as pf_mod
    mp.setattr(pf_mod, "F_AGE_WITH_L2", 0.0, raising=True)
    mp.setattr(pf_mod, "F_AGE_WITHOUT_L2", 0.0, raising=True)


# ─── corrupt_cz_taper_consistency: break the shared He/Z CZ-base taper ───
# The smooth cosine taper at the CZ base is applied identically to He and Z
# settling velocities (matching MESA's limit_coeffs_face, which multiplies
# ALL species uniformly — diffusion_procs.f90:get_limit_coeffs L985-1039).
# This mutation patches _compute_diffusion_structure to replace the smooth
# cz_taper with a hard 0/1 mask (the old approach), AND patches
# diffuse_composition to use a different CZ-base stencil for Z than for He
# (the ~1.5× additive upwind that documented). A test asserting
# shape-consistency of He and Z drain profiles must FAIL under this mutation.
@register("corrupt_cz_taper_consistency")
def _corrupt_cz_taper_consistency(mp, stellar_mod=None):
    """Replace smooth taper with hard mask + inconsistent Z stencil."""
    import stellar_jax.composition.diffuse as diffuse_mod
    import jax.numpy as _jnp

    _orig_structure = diffuse_mod._compute_diffusion_structure

    def _hardmask_structure(*args, **kwargs):
        """Replace smooth cz_taper with a hard 0/1 mask."""
        struct = _orig_structure(*args, **kwargs)
        # Convert smooth taper to a hard binary: taper > 0.5 → 1.0, else → 0.0
        hard_mask = _jnp.where(struct['cz_taper'] > 0.5, 1.0, 0.0)
        struct['cz_taper'] = hard_mask
        return struct

    mp.setattr(diffuse_mod, "_compute_diffusion_structure",
               _hardmask_structure, raising=True)


# ─── inflate_file_size_cap: defeat the soft file-size cap lint ───
# The file-size lint (scripts/lint_file_size.py) checks that no source file
# exceeds ~1000 logical lines. This mutation inflates SOFT_CAP to a huge value
# so that scan() with default cap never reports any file as over-cap. A test
# that asserts "over-cap files are detected" using the default SOFT_CAP must
# FAIL under this mutation — proving the cap value is load-bearing.
# This is a code-quality mutation (no physics), gating the lint meta-test.
@register("inflate_file_size_cap")
def _inflate_file_size_cap(mp, stellar_mod=None):
    """Inflate SOFT_CAP to 999999 — no file is ever flagged as over-cap."""
    import sys
    from pathlib import Path as _Path
    _scripts_dir = str(_Path(__file__).resolve().parent.parent / "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    import lint_file_size
    mp.setattr(lint_file_size, "SOFT_CAP", 999999, raising=True)


# ─── inflate_body_line_cap: defeat the soft function-length cap lint ───
# The function-length lint (scripts/lint_signatures.py) checks that no function
# body exceeds ~100 logical lines. This mutation inflates MAX_BODY_LINES to a
# huge value so that check_function_body_length() with the default cap never
# reports any function as over-cap. A test that asserts "over-cap functions are
# detected" using lint.MAX_BODY_LINES must FAIL under this mutation — proving
# the cap value is load-bearing.
# This is a code-quality mutation (no physics), gating the lint meta-test.
@register("inflate_body_line_cap")
def _inflate_body_line_cap(mp, stellar_mod=None):
    """Inflate MAX_BODY_LINES to 999999 — no function is ever flagged as over-cap."""
    import sys
    from pathlib import Path as _Path
    _scripts_dir = str(_Path(__file__).resolve().parent.parent / "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    import lint_signatures
    mp.setattr(lint_signatures, "MAX_BODY_LINES", 999999, raising=True)


# ─── overshoot_default_split: revert ONE path to the old ALPHA_OV=0.2 default ───
# unified all code paths to F_OV=0.016. This mutation reverts the
# evolution path's f_ov=None fallback to 0.2 (the old ALPHA_OV parametrization
# error), reintroducing the 12.5× split between evolution and inference.
# A test asserting that the default f_ov is the same across all code paths
# MUST FAIL under this mutation.
# Ref: Herwig (2000), A&A 360, 952; MESA overshoot_step.f90:131 (dr < f*Hp_cb).
@register("overshoot_default_split")
def _overshoot_default_split(mp, stellar_mod=None):
    """Revert the evolution path's f_ov default to the old ALPHA_OV=0.2.

    Patches mix_composition's internal default so that f_ov=None resolves to
    0.2 instead of F_OV=0.016 — the exact bug #1124 fixed. Any test that
    asserts consistency between evolution-path and inference-path f_ov
    defaults must FAIL.
    """
    from stellar_jax.composition import mix as mix_mod

    _orig_mix = mix_mod.mix_composition

    def _split_mix(X_profile, shell_data, M_solar=1.0, f_ov=None, **kwargs):
        if f_ov is None:
            f_ov = 0.2  # Reintroduce the old ALPHA_OV=0.2 parametrization error
        return _orig_mix(X_profile, shell_data, M_solar=M_solar, f_ov=f_ov, **kwargs)

    mp.setattr(mix_mod, "mix_composition", _split_mix)


# ─── solar_dnu_mode_target: restore the hardcoded solar Δν=135/ε=1.5 selector ───
# The production code uses per-star Δν + ε from the acoustic radius integral and
# l=0 mode spacing, selecting modes via assign_n_pg (Tassoul relation). This mutation
# reverts the mode selector to the old hardcoded solar values (Δν=135 µHz, ε=1.5),
# which gives the WRONG mode for non-solar stars (2.0 M☉ has Δν≈85 µHz).
# Any test asserting correct non-solar mode selection (e.g. l=0 n=20 on 2.0 M☉
# midMS at 1781.7 µHz per ADIPLS) must FAIL under this mutation — the old selector
# returns ~2903 µHz (the solar target 135*(20+0+1.5)=2902.5).
# Reference: Tassoul (1980), ApJS 43, 469;.
# ─── detach_r02_ratio_gradient: sever the gradient path through r02 ratio ───
# r02_differentiable computes r₀₂(n) = [ν(n,0) − ν(n−1,2)] / [ν(n,1) − ν(n−1,1)]
# from differentiable eigenfrequencies (IFT adjoint). This mutation patches
# r02_differentiable to apply stop_gradient to its output, severing the gradient
# from the ratio back to the structure. The forward value is unchanged, but
# jax.grad returns 0. Any test asserting a nonzero ∂r₀₂/∂θ will FAIL.
# Ref: Roxburgh & Vorontsov (2003), A&A 411, 215;.
@register("detach_r02_ratio_gradient")
def _detach_r02_ratio_gradient(mp, stellar_mod=None):
    """stop_gradient on r02_differentiable output — kills ∂r₀₂/∂θ.

    Patches r02_differentiable at the seismic_quantities source module AND
    the oscillations package re-export to apply stop_gradient on the returned
    ratio value. The forward value is unchanged (r02 is correct), but the AD
    gradient ∂r₀₂/∂(any upstream parameter) = 0.
    """
    import jax
    import stellar_jax.oscillations.seismic_quantities as sq_mod
    import stellar_jax.oscillations as osc_mod

    _orig_r02_diff = sq_mod.r02_differentiable

    def _detached_r02_diff(*args, **kwargs):
        r02_val = _orig_r02_diff(*args, **kwargs)
        return jax.lax.stop_gradient(r02_val)

    mp.setattr(sq_mod, "r02_differentiable", _detached_r02_diff, raising=True)
    mp.setattr(osc_mod, "r02_differentiable", _detached_r02_diff, raising=True)


@register("solar_dnu_mode_target")
def _solar_dnu_mode_target(mp, stellar_mod=None):
    """Revert mode selection to hardcoded solar Δν=135, ε=1.5 — wrong for non-solar stars.

    Patches _select_mode_by_npg (the per-star selector introduced by #1122) in BOTH
    eigenfunction.py and analytic_kernels.py to use the old argmin(|roots − 135*(n+l/2+1.5)|)
    formula. The forward eigenvalue solver is unchanged — the roots are still correct —
    but the SELECTION picks the wrong root for non-solar stars.
    """
    import numpy as np
    import stellar_jax.oscillations.eigenfunction as ef_mod
    import stellar_jax.oscillations.analytic_kernels as ak_mod

    def _solar_select(roots_nu, glob, var, l, n_pg):
        """Old solar-hardcoded selector: argmin(|roots − 135*(n+l/2+1.5)|)."""
        dnu_est = 135.0
        nu_target = dnu_est * (n_pg + l / 2.0 + 1.5)
        mode_index = int(np.argmin(np.abs(np.array(roots_nu) - nu_target)))
        return mode_index

    mp.setattr(ef_mod, "_select_mode_by_npg", _solar_select, raising=True)
    mp.setattr(ak_mod, "_select_mode_by_npg", _solar_select, raising=True)


# ─── replay_mesh_schedule_detach: corrupt mesh schedule to prove per-step mesh matters ───
# When replay_mesh=True, the per-step q_mesh from the mesh_schedule is fed into
# the Henyey solve and eps computation. This mutation DISCARDS the per-step
# mesh override, forcing both paths to fall back to tc.q_mesh (the static ZAMS
# mesh). This proves that the per-step mesh variation recorded by the adaptive
# forward is necessary for the SGB replay — the ZAMS mesh poorly resolves the
# thin H-burning shell, causing the replay to diverge from the adaptive forward
# (logL_diff gate or oscillation failures).
# DISCRIMINATING: the gradient chain M → Henyey IFT → σ² is PRESERVED (not
# killed) — only the mesh VALUES are wrong, so the test fails on physics
# (model divergence), not on a blanket gradient kill.
#
# MESA ref: evolve.f90:1882-1886 — do_mesh() per step; the mesh affects the solve.
@register("replay_mesh_schedule_detach")
def _replay_mesh_schedule_detach(mp, stellar_mod=None):
    """Discard per-step adapted mesh, forcing ZAMS mesh in replay.

    Monkeypatches _step_solve_and_eps to null out q_mesh_override so
    both the Henyey solve and the eps grid use tc.q_mesh (the static
    ZAMS mesh) instead of the recorded per-step adapted mesh. The
    gradient chain M → Henyey IFT → structure → σ² is preserved —
    the test fails because the WRONG mesh produces a divergent model,
    not because the gradient was killed.

    PATCH SITE: patches at the _core module level (evolution/_core.py),
    where the local binding lives. The `from stellar_jax.evolution.step
    import _step_solve_and_eps` in _core.py creates a module-level name
    binding — patching the step module attribute does NOT reach this
    binding. We must patch _core's own attribute.
    """
    import stellar_jax.evolution._core as core_mod

    _orig_step_solve_and_eps = core_mod._step_solve_and_eps

    def _corrupted_step_solve_and_eps(y_henyey_prev, comp_s, step_in, tc,
                                       henyey_step_fn):
        # Discard the per-step mesh override — forces fallback to tc.q_mesh.
        corrupted_step_in = dict(step_in)
        corrupted_step_in['q_mesh_override'] = None
        return _orig_step_solve_and_eps(
            y_henyey_prev, comp_s, corrupted_step_in, tc,
            henyey_step_fn)

    mp.setattr(core_mod, "_step_solve_and_eps", _corrupted_step_solve_and_eps, raising=True)


# ─── break_fgong_self_consistency: strip atm_ratio + inject stale logL ──────
# structure_to_fgong_jax with atm_ratio derives R_star and L_star directly from
# y_henyey (MESA pulse_fgong.f90:168 — r_outer = photosphere_r), making the FGONG
# self-consistent by construction.  Without atm_ratio, the legacy Stefan-Boltzmann
# path re-derives R_star from log_L/log_Te.
#
# This mutation strips the atm_ratio kwarg AND offsets log_L by +2 dex, simulating
# the pathological case where the carry's log_L is stale/inconsistent with y_henyey.
# Result: R_star(SB) >> max(r_henyey) → all x = r/R < 1e-4 → empty FGONG grid → crash.
#
# The +2 dex offset is chosen to be non-vacuous even at near-ZAMS: a 100× luminosity
# boost makes R_star = sqrt(100*L/(4πσ))/Te² ≈ 10 × R_real → x_max ≈ 0.1, which
# pushes R_star/r_max well outside physical bounds and the oscillation grid fails.
#
# Any test asserting self-consistent FGONG structure (x_max near 1/atm_ratio, or a
# successful oscillation eigenfrequency computation) must FAIL under this mutation.
# Reference: MESA pulse_fgong.f90:168 (r_outer = photosphere_r);
# star_utils.f90:1093 (set_phot_info). Note: atm_ratio is a stellar-jax concept,
# not a MESA term.
@register("break_fgong_self_consistency")
def _break_fgong_self_consistency(mp, stellar_mod=None):
    """Strip atm_ratio from structure_to_fgong_jax + offset logL by +2 dex.

    Patches structure_to_fgong_jax at the definition site (fgong.builder),
    the package re-export (fgong.__init__), and the evolution re-export
    (evolution._core, evolution package). All import paths resolve to the
    broken version.

    Under this mutation the self-consistent path is disabled and the SB path
    receives a +2 dex logL → R_star >> max(r_henyey) → empty FGONG grid.
    """
    import stellar_jax.fgong.builder as fgong_builder
    import stellar_jax.fgong as fgong_pkg
    import stellar_jax.evolution._core as evolution_core
    import stellar_jax.evolution as evolution_pkg

    _orig = fgong_builder.structure_to_fgong_jax

    def _broken_fgong(*args, **kwargs):
        # Strip atm_ratio to force the legacy SB path.
        kwargs.pop('atm_ratio', None)
        # Offset log_L by +2 dex to make R_star >> max(r).
        # log_L is the 2nd positional arg (index 1).
        args = list(args)
        import jax.numpy as jnp
        args[1] = jnp.float64(args[1]) + 2.0
        return _orig(*args, **kwargs)

    mp.setattr(fgong_builder, "structure_to_fgong_jax", _broken_fgong, raising=True)
    mp.setattr(fgong_pkg, "structure_to_fgong_jax", _broken_fgong, raising=True)
    mp.setattr(evolution_core, "structure_to_fgong_jax", _broken_fgong, raising=True)
    mp.setattr(evolution_pkg, "structure_to_fgong_jax", _broken_fgong, raising=True)


# ─── detach_r01_ratio_gradient: sever the gradient path through r01 ratio ───
# r01_differentiable computes the 5-point smoothed r₀₁(n) ratio
# (MESA astero_support.f90:396: d01 = (l0(i0-1) - 4*l1(i1-1) + 6*l0(i0)
# - 4*l1(i1) + l0(i0+1))/8, r01 = d01/dnu) from differentiable
# eigenfrequencies (IFT adjoint). This mutation patches r01_differentiable
# to apply stop_gradient to its output, severing the gradient from the ratio
# back to the structure. The forward value is unchanged, but jax.grad returns 0.
# Any test asserting a nonzero ∂r₀₁/∂θ will FAIL.
# Ref: Roxburgh & Vorontsov (2003), A&A 411, 215;.
@register("detach_r01_ratio_gradient")
def _detach_r01_ratio_gradient(mp, stellar_mod=None):
    """stop_gradient on r01_differentiable output — kills ∂r₀₁/∂θ.

    Patches r01_differentiable at the seismic_quantities source module AND
    the oscillations package re-export to apply stop_gradient on the returned
    ratio value. The forward value is unchanged (r01 is correct), but the AD
    gradient ∂r₀₁/∂(any upstream parameter) = 0.
    """
    import jax
    import stellar_jax.oscillations.seismic_quantities as sq_mod
    import stellar_jax.oscillations as osc_mod

    _orig_r01_diff = sq_mod.r01_differentiable

    def _detached_r01_diff(*args, **kwargs):
        r01_val = _orig_r01_diff(*args, **kwargs)
        return jax.lax.stop_gradient(r01_val)

    mp.setattr(sq_mod, "r01_differentiable", _detached_r01_diff, raising=True)
    mp.setattr(osc_mod, "r01_differentiable", _detached_r01_diff, raising=True)


# ─── comp_nan_recovery_readonly: revert fix (NaN recovery on read-only arrays) ──────────
# The fix introduced _writable_copy() which calls np.array() to make writable copies of the
# four CNO arrays (C12, C13, N14, Z) before the NaN-recovery loop mutates them in-place.
# This mutation patches _writable_copy to return np.asarray (a read-only view), so the
# in-place NaN repair crashes with "assignment destination is read-only".
@register("comp_nan_recovery_readonly")
def _comp_nan_recovery_readonly(mp, stellar_mod=None):
    """Revert _writable_copy to np.asarray — NaN recovery must crash."""
    import numpy as np
    import stellar_jax.evolution.adaptive.forward as fwd_mod

    mp.setattr(fwd_mod, "_writable_copy", lambda arr: np.asarray(arr), raising=True)


# ─── freeze_composition_kernel: zero out the EOS composition sensitivity of Γ₁ ────────────
# This mutation makes the OPAL EOS table interpolation of Γ₁ insensitive to Z by
# zeroing ∂Γ₁/∂Z. We do this by patching _eos_opal_table to stop_gradient on Z
# before interpolation, so that jax.grad(Γ₁, Z) = 0 and the composition kernel
# K_{Z,ρ} vanishes. The forward Γ₁ value is unchanged (Z is still used for the
# primal interpolation), but the AD derivative through Z → Γ₁ is killed.
# Validates: test_composition_kernel_ad_vs_fd.
@register("freeze_composition_kernel")
def _freeze_composition_kernel(mp, stellar_mod=None):
    """Kill ∂Γ₁/∂Z by stop_gradient on Z in the OPAL EOS lookup.

    Forward Γ₁ values are unchanged (Z still enters the primal interpolation
    because stop_gradient only affects the gradient, not the value). But
    jax.grad(Γ₁, wrt Z) = 0, so the composition kernel K_{Z,ρ} vanishes.
    """
    import jax
    import stellar_jax.microphysics.eos as eos_mod

    _orig_opal = eos_mod._eos_opal_table

    def _frozen_opal(logT, logP, X, Z):
        # stop_gradient on Z: forward value unchanged, but dΓ₁/dZ = 0
        Z_frozen = jax.lax.stop_gradient(Z)
        return _orig_opal(logT, logP, X, Z_frozen)

    mp.setattr(eos_mod, "_eos_opal_table", _frozen_opal, raising=True)


# ─── opacity_hermite_z_tangent: revert Z tangent to Hermite (pre- bug) ───
# The fix in changed the Z tangent in the custom_jvp from Hermite (cubic)
# to AD through the quadrilinear (linear), matching MESA default
# (cubic_interpolation_in_Z = .false., kap.defaults:268). This mutation reverts
# to the old Hermite Z tangent, reintroducing the 0.3-37% primal-tangent mismatch.
# A test asserting AD-JVP vs AD-forward consistency on ∂κ/∂Z must FAIL.
# Ref: MESA kap/private/kap_eval_fixed.f90:267-289 (Get_Kap_for_Z_linear).
@register("opacity_hermite_z_tangent")
def _opacity_hermite_z_tangent(mp, stellar_mod=None):
    """Revert Z tangent to Hermite (cubic) — reintroduces primal-tangent mismatch.

    Patches _opal_kappa_smooth_xz_jvp_wrapper's JVP to use _interp4d_hermite
    for the Z tangent (the pre-#1213 behavior). The forward (quadrilinear) is
    unchanged, but the Z tangent comes from a DIFFERENT interpolant (cubic
    Hermite), creating a 0.3-37% mismatch depending on Z position in its
    bracket.

    This mutation re-registers the JVP on _opal_kappa_smooth_xz_jvp_wrapper
    with the old Hermite-based Z tangent. Because JAX custom_jvp decorators
    modify the function in-place, we replace the entire wrapper function.
    """
    import jax
    import jax.numpy as jnp
    from stellar_jax.microphysics import opacity as opacity_mod

    # Create a new version of the wrapper with the OLD (Hermite) Z tangent
    @jax.custom_jvp
    def _old_wrapper(logT, logRho, X, Z):
        logR = logRho - 3.0 * logT + 18.0
        return opacity_mod._interp4d_logkappa(
            opacity_mod.OPAL_X, opacity_mod.OPAL_Z,
            opacity_mod.OPAL_LOGT, opacity_mod.OPAL_LOGR, opacity_mod.OPAL_LK,
            logT, logR, X, Z)

    @_old_wrapper.defjvp
    def _old_jvp(primals, tangents):
        logT, logRho, X, Z = primals
        d_logT, d_logRho, d_X, d_Z = tangents
        logR = logRho - 3.0 * logT + 18.0
        primal_out = opacity_mod._interp4d_logkappa(
            opacity_mod.OPAL_X, opacity_mod.OPAL_Z,
            opacity_mod.OPAL_LOGT, opacity_mod.OPAL_LOGR, opacity_mod.OPAL_LK,
            logT, logR, X, Z)
        d_logR = d_logRho - 3.0 * d_logT
        _, tangent_TR = jax.jvp(
            lambda lt, lr: opacity_mod._interp4d_logkappa(
                opacity_mod.OPAL_X, opacity_mod.OPAL_Z,
                opacity_mod.OPAL_LOGT, opacity_mod.OPAL_LOGR, opacity_mod.OPAL_LK,
                lt, lr, X, Z),
            (logT, logR), (d_logT, d_logR))
        _, tangent_X = jax.jvp(
            lambda x: opacity_mod._interp4d_hermite(
                opacity_mod.OPAL_X, opacity_mod.OPAL_Z,
                opacity_mod.OPAL_LOGT, opacity_mod.OPAL_LOGR, opacity_mod.OPAL_LK,
                logT, logR, x, Z,
                opacity_mod.OPAL_SLOPES_X, opacity_mod.OPAL_SLOPES_Z,
                opacity_mod.OPAL_SLOPES_T, opacity_mod.OPAL_SLOPES_R),
            (X,), (d_X,))
        # OLD (pre-): Z tangent via HERMITE, not quadrilinear
        _, tangent_Z = jax.jvp(
            lambda z: opacity_mod._interp4d_hermite(
                opacity_mod.OPAL_X, opacity_mod.OPAL_Z,
                opacity_mod.OPAL_LOGT, opacity_mod.OPAL_LOGR, opacity_mod.OPAL_LK,
                logT, logR, X, z,
                opacity_mod.OPAL_SLOPES_X, opacity_mod.OPAL_SLOPES_Z,
                opacity_mod.OPAL_SLOPES_T, opacity_mod.OPAL_SLOPES_R),
            (Z,), (d_Z,))
        return primal_out, tangent_TR + tangent_X + tangent_Z

    # Replace the wrapper at the module level. opal_kappa() calls this directly.
    mp.setattr(opacity_mod, "_opal_kappa_smooth_xz_jvp_wrapper",
               _old_wrapper, raising=True)


# ─── drop_atm_dZ: zero the Z atmosphere correction term (Z-b) ──────────
# The Z-b fix in added ∂(ln_P_atm,ln_T_atm)/∂Z to the atmosphere correction
# in _atmosphere_correction (adjoint.py). This mutation zeros the Z atmosphere
# partials (indices 3 and 8 of the 10-element atm_param_grads), removing the Z→κ→
# P_atm,T_atm contribution to g_Zp. The forward model is unchanged — only the
# backward (gradient) path is affected.
# MESA ref: hydro_eqns.f90:930-940 (dlnP_bc_dlnkap * dlnkap_d...).
@register("drop_atm_dZ")
def _drop_atm_dZ(mp, stellar_mod=None):
    """Zero the Z atmosphere partials in atm_param_grads — removes g_Zp correction.

    Patches _atmosphere_correction in adjoint.py to zero the dlnP_dZ and dlnT_dZ
    entries (indices 3 and 8) of the atm_param_grads vector. This is equivalent
    to the pre-#1211 state where Z was closed over in _atm_of_params and the
    atmosphere Z sensitivity was lost.
    """
    import jax.numpy as jnp
    import stellar_jax.solver.adjoint as adj_mod
    import stellar_jax.solver.continuation as cont_mod

    _orig_atm_correction = adj_mod._atmosphere_correction

    def _zeroed_dZ_atm_correction(lam, N_s, g_Ms, g_ap, g_ar, g_Zp, g_Xp, atm_param_grads):
        # Zero the Z partials (indices 3 and 8) — removes the Z atmosphere term
        atm_param_grads_no_Z = atm_param_grads.at[3].set(0.0).at[8].set(0.0)
        return _orig_atm_correction(lam, N_s, g_Ms, g_ap, g_ar, g_Zp, g_Xp, atm_param_grads_no_Z)

    mp.setattr(adj_mod, "_atmosphere_correction", _zeroed_dZ_atm_correction, raising=True)
    mp.setattr(cont_mod, "_atmosphere_correction", _zeroed_dZ_atm_correction, raising=True)


# ─── drop_atm_dX: zero the X_surf atmosphere correction term ─────────
# The fix added ∂(ln_P_atm, ln_T_atm)/∂X_surf to the atmosphere correction
# in _atmosphere_correction (adjoint.py). This mutation zeros the X_surf atmosphere
# partials (indices 4 and 9 of the 10-element atm_param_grads), removing the
# X_surf→κ,EOS→P_atm,T_atm contribution to g_Xp. The forward model is unchanged —
# only the backward (gradient) path is affected.
# Since Y_init→X via X=1-Y-Z, the missing X_surf term biased ∂σ²/∂Y_init by ~25%.
# MESA ref: hydro_eqns.f90:930-940 (dlnP_bc_dlnkap * dlnkap_d... applied to X).
@register("drop_atm_dX")
def _drop_atm_dX(mp, stellar_mod=None):
    """Zero the X_surf atmosphere partials in atm_param_grads — removes g_Xp correction.

    Patches _atmosphere_correction in adjoint.py to zero the dlnP_dX and dlnT_dX
    entries (indices 4 and 9) of the atm_param_grads vector. This is equivalent
    to the pre-#1119 state where X_surf was closed over in _atm_of_params and the
    atmosphere X sensitivity was lost — biasing ∂σ²/∂Y_init by ~25%.
    """
    import jax.numpy as jnp
    import stellar_jax.solver.adjoint as adj_mod
    import stellar_jax.solver.continuation as cont_mod

    _orig_atm_correction = adj_mod._atmosphere_correction

    def _zeroed_dX_atm_correction(lam, N_s, g_Ms, g_ap, g_ar, g_Zp, g_Xp, atm_param_grads):
        # Zero the X partials (indices 4 and 9) — removes the X_surf atmosphere term
        atm_param_grads_no_X = atm_param_grads.at[4].set(0.0).at[9].set(0.0)
        return _orig_atm_correction(lam, N_s, g_Ms, g_ap, g_ar, g_Zp, g_Xp, atm_param_grads_no_X)

    mp.setattr(adj_mod, "_atmosphere_correction", _zeroed_dX_atm_correction, raising=True)
    mp.setattr(cont_mod, "_atmosphere_correction", _zeroed_dX_atm_correction, raising=True)


# ─── sever_cno_isotopes: re-apply stop_gradient on C12_m/C13_m (Z-a) ──
# The Z-a fix in relaxed the GP-4 stop_gradient on C12_m/C13_m in step.py.
# This mutation re-applies stop_gradient to simulate the pre- state where
# the C12/C13 gradient path through burn_cno was severed. The forward model is
# unchanged.
@register("sever_cno_isotopes")
def _sever_cno_isotopes(mp, stellar_mod=None):
    """Re-apply stop_gradient on C12/C13 after mix — severs ∂CN_total/∂Z via C12/C13.

    Patches mix_composition calls for C12/C13 in _step_composition_update by
    wrapping the composition update to stop_gradient the C12_m and C13_m outputs.
    """
    import jax
    import stellar_jax.evolution.step as step_mod

    _orig_comp_update = step_mod._step_composition_update

    def _severed_comp_update(comp, step_data, tc, physics):
        X_diff, Y_mixed, Z_mixed, C12_m, C13_m, N14_m, X3_m = _orig_comp_update(
            comp, step_data, tc, physics)
        return X_diff, Y_mixed, Z_mixed, jax.lax.stop_gradient(C12_m), \
            jax.lax.stop_gradient(C13_m), N14_m, X3_m

    mp.setattr(step_mod, "_step_composition_update", _severed_comp_update, raising=True)
    # Patch at the lax.scan call site: _core.py has its own import binding
    # `from stellar_jax.evolution.step import _step_composition_update` (line 118).
    # Without this patch, the stop_gradient on C12_m/C13_m never takes effect in
    # the actual evolution loop — the mutation would be inert on the real code path.
    # Mirrors the f_ov_detach fix.
    import stellar_jax.evolution._core as core_mod
    mp.setattr(core_mod, "_step_composition_update", _severed_comp_update, raising=True)


# ─── freeze_factor: re-freeze the σ²→ν factor to drop ∂factor/∂θ ──────────
# The bug fixed by was: factor = float(stop_gradient(grid_data['factor'])) in the
# σ²→ν conversion, which drops the dominant scaling term ν²·∂factor/∂θ. This mutation
# re-introduces that freeze by wrapping nu_from_sigma2 to apply stop_gradient on the
# factor argument. Under this mutation, test_dsigma2_dM_fresh_factor must FAIL (sign
# error or huge magnitude error) because the ∂factor/∂M term is removed.
# Note: test_dsigma2_dZ_fresh_factor was removed because in the isolated FGONG
# architecture (frozen y_henyey/logL/logTe), factor = R³/(GM) is Z-independent — the
# mutation is a no-op. The Z → EOS → σ² path is covered by test_seismic_gradient_Z
# with mutation detach_z_gradient.
@register("freeze_factor")
def _freeze_factor(mp, stellar_mod=None):
    """Re-freeze factor in nu_from_sigma2 — drops the ν²·∂factor/∂θ scaling term."""
    import jax
    import jax.numpy as jnp
    import stellar_jax.oscillations.seismic_conversion as conv_mod
    import stellar_jax.oscillations.seismic_quantities as sq_mod

    _orig_nu_from_sigma2 = conv_mod.nu_from_sigma2

    def _frozen_nu_from_sigma2(sigma2, factor):
        """nu_from_sigma2 with factor frozen via stop_gradient."""
        return jnp.sqrt(sigma2 / jax.lax.stop_gradient(factor))

    # Patch the canonical module
    mp.setattr(conv_mod, "nu_from_sigma2", _frozen_nu_from_sigma2, raising=True)

    # Patch the re-export in oscillations __init__ (callers may import from there)
    import stellar_jax.oscillations as osc_mod
    mp.setattr(osc_mod, "nu_from_sigma2", _frozen_nu_from_sigma2, raising=True)

    # Patch the local binding in seismic_quantities — the function-level import
    # `from .seismic_conversion import nu_from_sigma2` creates a local binding
    # inside r02_differentiable/r01_differentiable. Since those are function-level
    # imports (inside the function body), they re-import each call, so patching
    # the module attribute is sufficient.


# ─── disable_adjoint_iterative_refinement: skip refinement in the adjoint Thomas solve ──
# The production _solve_adjoint_system performs 1 refinement step (normal mode) or 2 (replay)
# to correct O(N) rounding errors in the Thomas forward/backward sweeps (Higham 2002 §9.4).
# This mutation sets refinement iterations to 0, leaving the raw Thomas solve result — which has
# ~1e-8 residual for N=200 interior-directed cotangents vs ~1e-15 with refinement.
@register("disable_adjoint_iterative_refinement")
def _disable_adjoint_iterative_refinement(mp, stellar_mod=None):
    """Break: zero iterative refinement iterations in _solve_adjoint_system."""
    import stellar_jax.solver.adjoint as adjoint_mod
    import jax.numpy as jnp
    from stellar_jax.solver.thomas import _adjoint_thomas_solve, _transpose_block_tridiag

    def _solve_adjoint_no_refine(A, B, C, g_y, energy_scale, bypass_conv_gate):
        """_solve_adjoint_system with 0 refinement iterations."""
        scale_row = energy_scale[1:, None]
        A = A.at[1:, 0, :].multiply(scale_row)
        B = B.at[1:, 0, :].multiply(scale_row)
        C = C.at[1:, 0, :].multiply(scale_row)
        A_T, B_T, C_T = _transpose_block_tridiag(A, B, C)
        mu = _adjoint_thomas_solve(A_T, B_T, C_T, g_y)
        mu = jnp.where(~jnp.isfinite(mu), 0.0, mu)
        # NO iterative refinement — the mutation
        lam = mu.at[1:, 0].multiply(energy_scale[1:])
        lam = jnp.where(~jnp.isfinite(lam), 0.0, lam)
        return lam

    mp.setattr(adjoint_mod, "_solve_adjoint_system", _solve_adjoint_no_refine, raising=True)

    # Also patch the continuation module's import binding (the production call site)
    import stellar_jax.solver.continuation as cont_mod
    mp.setattr(cont_mod, "_solve_adjoint_system", _solve_adjoint_no_refine, raising=True)

    # Patch the solver package re-export
    import stellar_jax.solver as solver_pkg
    if hasattr(solver_pkg, '_solve_adjoint_system'):
        mp.setattr(solver_pkg, "_solve_adjoint_system", _solve_adjoint_no_refine, raising=True)

    # Patch the henyey facade re-export
    import stellar_jax.henyey as henyey_mod
    if hasattr(henyey_mod, '_solve_adjoint_system'):
        mp.setattr(henyey_mod, "_solve_adjoint_system", _solve_adjoint_no_refine, raising=True)
