"""Per-mass gradient bundle infrastructure for CI consolidation (#1221).

Session-scoped fixtures that run ONE evolve_star per mass at
{1.0, 1.2, 1.5, 2.0} M☉ using MODE-A (MESA_CONFIG). Tests that previously
ran their own evolve_star read structure/gradient artifacts from these
bundles instead — collapsing per-wave evolve_star invocations.

Architecture:
  The bundle forward is a REAL evolve_star call on the production code path,
  not a pre-baked artifact file. This ensures:
  1. The bundle always reflects HEAD physics.
  2. Under STELLAR_MUTATION, the mutation is applied at the MODULE level
     BEFORE evolve_star runs → bundle artifacts carry mutated physics →
     downstream assertions fail (correctly). This is critical because
     session-scoped fixtures run before function-scoped _o2_mutation, so
     the bundle must apply mutations itself.
  3. The bundle is regenerated every CI session — no stale artifacts.

  Each bundle stores the evolve_star result dict with keys:
    star_age, log_L, log_Teff, log_R, center_h1, y_henyey_final,
    atm_ratio, converged, mass, Z, alpha_mlt, etc.

  Tests consume the bundle via session-scoped fixtures:
    bundle_1p0, bundle_1p2, bundle_1p5, bundle_2p0, all_bundles

CI grouping:
  The CI orchestrator groups all tests that request the same bundle_NpM
  fixture into a SINGLE CI task. The bundle forward compiles once
  (~10-16 min XLA), and all downstream tests share the compiled result.
  Each bundle task replaces the N individual tasks that were each compiling
  the same evolve_star independently.

  CI routing: tests requesting bundle fixtures are listed in
  tests/ci_bundle_tests.txt (one line per test).

MESA ref: MODE-A config from stellar_jax.config.mesa_config.MESA_CONFIG,
parsed from data/mesa_comparison/inlist_1.0Msun.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

# Ensure tests/ is importable
_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)


# ---------------------------------------------------------------------------
# Bundle configuration
# ---------------------------------------------------------------------------

# Max steps per mass: tuned to reach key evolutionary stages within the
# CI time budget while keeping compile cost to ONE XLA compile per mass.
# MODE-A: Z=0.014, α=2.0, Y=0.2695, diffusion=False, f_ov=0.0.
#
# The forward goes to mid-MS or TAMS depending on mass:
# - 1.0 M☉: N=500 → ~2 Gyr, mid-MS (MS lifetime ~10 Gyr)
# - 1.2 M☉: N=500 → ~1.5 Gyr, mid-MS (MS lifetime ~5 Gyr)
# - 1.5 M☉: N=500 → past TAMS (~2 Gyr MS)
# - 2.0 M☉: N=500 → past TAMS (~1 Gyr MS)
BUNDLE_MAX_STEPS = {
    1.0: 500,
    1.2: 500,
    1.5: 500,
    2.0: 500,
}

# Cache: applied once per session, guards against double-apply
_mutation_applied = False


def _apply_mutation_if_set():
    """Apply STELLAR_MUTATION at the module level, if the env var is set.

    Session-scoped fixtures run BEFORE function-scoped fixtures (including
    conftest._o2_mutation). The O2 mutation gate must break the bundle's
    evolve_star call so downstream assertions correctly fail. We apply
    the mutation here — once per session — via direct module attribute
    patching (the same mechanism mutations.py uses, but without pytest's
    monkeypatch since we're in session scope).

    The _o2_mutation function-scoped fixture will also try to apply the
    same mutation later, but since the module attributes are already
    patched, the double-apply is harmless (monkeypatch.setattr on an
    already-patched attribute is a no-op for the physics).
    """
    global _mutation_applied
    if _mutation_applied:
        return

    name = os.environ.get("STELLAR_MUTATION", "").strip()
    if not name:
        return

    # Disable JAX compilation cache under mutation (same as conftest._o2_mutation)
    import jax
    try:
        jax.config.update("jax_enable_compilation_cache", False)
    except Exception:
        pass
    try:
        jax.config.update("jax_compilation_cache_dir", "")
    except Exception:
        pass

    # Load mutations registry
    spec = importlib.util.spec_from_file_location(
        "stellar_mutations",
        os.path.join(os.path.dirname(__file__), "mutations.py"))
    muts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(muts)

    if name not in muts.MUTATIONS:
        pytest.fail(
            f"Bundle mutation gate: unknown mutation '{name}' "
            f"(not in tests/mutations.py registry)")

    # Create a minimal monkeypatch-like object for the mutation function.
    # mutations.py functions expect (monkeypatch, stellar_module) args.
    # We use _SessionMonkeypatch which records setattr calls so they
    # persist for the entire session.
    import importlib
    stellar = importlib.import_module("stellar_jax.stellar")

    mp = _SessionMonkeypatch()
    muts.MUTATIONS[name](mp, stellar)
    _mutation_applied = True

    print(f"\n[BUNDLE] Applied STELLAR_MUTATION='{name}' at session scope")


class _SessionMonkeypatch:
    """Minimal monkeypatch that persists for the session (no undo).

    Unlike pytest's monkeypatch (which undoes at fixture teardown), this
    patches module attributes permanently for the session. This is correct
    for O2 mutation runs where the entire session runs under one mutation.
    """

    def __init__(self):
        self._patches = []

    def setattr(self, target, name_or_value, value=None, raising=True):
        """Match pytest monkeypatch.setattr signature."""
        if value is None:
            # 2-arg form: setattr(target_string, value)
            # Parse "module.attr" → (module, attr)
            # But mutations.py always uses 3-arg form
            raise TypeError(
                "Bundle _SessionMonkeypatch only supports 3-arg setattr "
                "(target, name, value)")
        if raising and not hasattr(target, name_or_value):
            raise AttributeError(
                f"{target!r} has no attribute {name_or_value!r}")
        old = getattr(target, name_or_value, None)
        setattr(target, name_or_value, value)
        self._patches.append((target, name_or_value, old))


def _run_bundle(mass):
    """Run the MODE-A evolve_star for a single mass.

    Returns the full evolve_star result dict with bundle metadata attached.
    This is the SAME code path as what individual tests were calling — no
    shortcut. Under STELLAR_MUTATION, the mutation is applied before this
    call (see _apply_mutation_if_set).

    The call uses MESA_CONFIG exclusively (O4 rule: any test reading
    data/mesa_comparison/ MUST use MESA_CONFIG, never hardcoded values).
    """
    # Ensure mutation is applied before evolve_star runs
    _apply_mutation_if_set()

    from stellar_jax.stellar import evolve_star
    from stellar_jax.config.mesa_config import MESA_CONFIG

    max_steps = BUNDLE_MAX_STEPS[mass]

    print(f"\n{'='*60}")
    print(f"[BUNDLE] Running MODE-A evolve_star for {mass} M☉ "
          f"(max_steps={max_steps})")
    mutation_name = os.environ.get("STELLAR_MUTATION", "").strip()
    if mutation_name:
        print(f"[BUNDLE] ⚠ MUTATION ACTIVE: {mutation_name}")
    print(f"{'='*60}")

    result = evolve_star(
        mass,
        **MESA_CONFIG,
        max_steps=max_steps,
    )

    # Attach bundle metadata for downstream assertions
    result['_bundle_mass'] = mass
    result['_bundle_config'] = dict(MESA_CONFIG)
    result['_bundle_max_steps'] = max_steps

    # Validate that the bundle produced a usable result
    assert 'log_L' in result, f"Bundle {mass} M☉: missing log_L"
    assert 'log_Teff' in result, f"Bundle {mass} M☉: missing log_Teff"
    assert len(result['log_L']) == max_steps, (
        f"Bundle {mass} M☉: log_L has {len(result['log_L'])} entries, "
        f"expected {max_steps}")
    assert np.all(np.isfinite(result['log_L'][:10])), (
        f"Bundle {mass} M☉: first 10 log_L entries are not finite")

    print(f"[BUNDLE] {mass} M☉ complete: logL[-1]={result['log_L'][-1]:.4f}, "
          f"logTeff[-1]={result['log_Teff'][-1]:.4f}")

    return result


# ---------------------------------------------------------------------------
# Session-scoped bundle fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def bundle_1p0():
    """Per-mass gradient bundle: 1.0 M☉ MODE-A (MESA_CONFIG).

    Session-scoped: compiles evolve_star ONCE for this mass per CI session.
    All tests reading this fixture share the compiled result.
    Under STELLAR_MUTATION, the mutation is applied before evolve_star runs.

    Evolutionary stage: mid-MS (N=500, ~2 Gyr of a ~10 Gyr MS).
    """
    return _run_bundle(1.0)


@pytest.fixture(scope="session")
def bundle_1p2():
    """Per-mass gradient bundle: 1.2 M☉ MODE-A (MESA_CONFIG).

    Session-scoped: compiles evolve_star ONCE for this mass per CI session.

    Evolutionary stage: mid-MS (N=500, ~1.5 Gyr of a ~5 Gyr MS).
    """
    return _run_bundle(1.2)


@pytest.fixture(scope="session")
def bundle_1p5():
    """Per-mass gradient bundle: 1.5 M☉ MODE-A (MESA_CONFIG).

    Session-scoped: compiles evolve_star ONCE for this mass per CI session.

    Evolutionary stage: near TAMS or past TAMS (N=500, ~2 Gyr MS).
    """
    return _run_bundle(1.5)


@pytest.fixture(scope="session")
def bundle_2p0():
    """Per-mass gradient bundle: 2.0 M☉ MODE-A (MESA_CONFIG).

    Session-scoped: compiles evolve_star ONCE for this mass per CI session.

    Evolutionary stage: past TAMS (N=500, ~1 Gyr MS).
    """
    return _run_bundle(2.0)


@pytest.fixture(scope="session")
def all_bundles(bundle_1p0, bundle_1p2, bundle_1p5, bundle_2p0):
    """All four per-mass MODE-A bundles, keyed by mass.

    Convenience fixture for multi-mass tests (e.g., test_mesa_comparison)
    that iterate over several masses.
    """
    return {
        1.0: bundle_1p0,
        1.2: bundle_1p2,
        1.5: bundle_1p5,
        2.0: bundle_2p0,
    }
