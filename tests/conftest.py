"""Pytest configuration — register custom marks and shared fixtures."""
import gc
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time

# ═══════════════════════════════════════════════════════════════
# Ensure tests/ directory is on sys.path so that helper modules living in
# tests/ (mesa_rgb.py, mutations.py) are importable as top-level names.
# With tests/__init__.py present, pytest treats tests/ as a package and no
# longer adds it to sys.path independently — breaking `import mesa_rgb`
# inside test function bodies.
# ═══════════════════════════════════════════════════════════════
_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

# ═══════════════════════════════════════════════════════════════
# Data directory fixup for CI. The Docker image's baked ci_entrypoint.sh
# runs 'ln -sf /app/data data' which on Docker overlay2 may either:
# - Create data/data→/app/data (symlink inside existing dir)
# - Replace data/ with a symlink (on some overlay2 configurations)
# - Corrupt directory state (partial whiteouts)
# Fix: UNCONDITIONALLY verify data integrity and restore from git if needed.
# This is cheap (~3 MB, <1s) and guarantees all tests see correct data.
# ═══════════════════════════════════════════════════════════════
_repo_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_data_dir = os.path.join(_repo_dir, "src", "stellar_jax", "data")

def _data_integrity_ok():
    """Check that critical data subdirectories exist and are populated."""
    checks = [
        os.path.isfile(os.path.join(_data_dir, "model_s", "model_s_cs.dat")),
        os.path.isdir(os.path.join(_data_dir, "mesa_comparison", "results")),
        os.path.isfile(os.path.join(_data_dir, "mesa_comparison", "results",
                                     "1.0Msun", "history.data")),
        os.path.isfile(os.path.join(_data_dir, "mesa_comparison", "results",
                                     "2.0Msun", "history.data")),
        # FGONG profiles (used by multiple tests: henyey convergence, MESA MLT)
        os.path.isdir(os.path.join(_data_dir, "mesa_comparison", "profiles")),
        os.path.isfile(os.path.join(_data_dir, "mesa_comparison", "profiles",
                                     "1.0Msun", "midMS.FGONG.gz")),
    ]
    return all(checks)

if os.path.islink(_data_dir) or not _data_integrity_ok():
    try:
        if os.path.islink(_data_dir):
            os.unlink(_data_dir)
        # Remove any stale symlink INSIDE data/ (from ln -sf /app/data data)
        _inner_link = os.path.join(_data_dir, "data")
        if os.path.islink(_inner_link):
            os.unlink(_inner_link)
        subprocess.run(
            ["git", "checkout", "HEAD", "--", "src/stellar_jax/data"],
            cwd=_repo_dir, capture_output=True, timeout=30
        )
    except Exception:
        pass

import jax
jax.config.update('jax_enable_x64', True)

# ═══════════════════════════════════════════════════════════════
# CI cache-metric capture (fix: compiles/cache_hits were DARK on JAX 0.10.2).
# JAX emits, via the stdlib `logging` module under JAX_LOG_COMPILES=1:
#   "Finished XLA compilation of <fn> in <s> sec"                  → a compile (cache MISS)
#   "Persistent compilation cache hit for '<fn>' with key '<key>'" → a warm HIT (served from EFS)
# pytest's logging plugin captures these records and only surfaces them for FAILING tests, so
# for the (passing) heavy tests the metric cares about they never reach /tmp/pytest.out —
# leaving ci_emf.py's compiles=0/cache_hits=0 permanently dark. Attach a dedicated FileHandler
# to the `jax` logger so ci_entrypoint parses the true hit/miss counts from /tmp/jax_compiles.log
# regardless of pytest capture. Gated on JAX_LOG_COMPILES=1 (only CI sets it → no-op locally),
# message-filtered so the file stays tiny and parse-clean.
# ═══════════════════════════════════════════════════════════════
if os.environ.get("JAX_LOG_COMPILES") == "1":
    import logging as _logging

    class _JaxCompileFilter(_logging.Filter):
        def filter(self, rec):
            m = rec.getMessage()
            return ("XLA compilation of" in m) or ("compilation cache hit" in m)

    try:
        _fh = _logging.FileHandler(os.environ.get("JAX_COMPILE_LOG", "/tmp/jax_compiles.log"))
        _fh.setLevel(_logging.INFO)
        _fh.addFilter(_JaxCompileFilter())
        _fh.setFormatter(_logging.Formatter("%(message)s"))
        _jaxlogger = _logging.getLogger("jax")
        _jaxlogger.addHandler(_fh)
        if _jaxlogger.level == _logging.NOTSET or _jaxlogger.level > _logging.INFO:
            _jaxlogger.setLevel(_logging.INFO)
    except Exception:
        pass

import pytest

# ═══════════════════════════════════════════════════════════════
# Per-mass gradient bundles (— CI consolidation).
# Session-scoped fixtures that run ONE evolve_star per mass at
# {1.0, 1.2, 1.5, 2.0} M☉ using MODE-A (MESA_CONFIG).
# Defined in bundle_conftest.py; imported here for fixture registration.
# ═══════════════════════════════════════════════════════════════
from tests.bundle_conftest import (  # noqa: F401
    bundle_1p0, bundle_1p2, bundle_1p5, bundle_2p0, all_bundles,
)

# ═══════════════════════════════════════════════════════════════
# Per-call timing instrumentation for evolve_star.
# Autouse fixture monkeypatches stellar.evolve_star to record
# wall time per call. Written to /tmp/test_timing.json at session end.
# ═══════════════════════════════════════════════════════════════

_timing_log = []


# ═══════════════════════════════════════════════════════════════
# MESA backend skip: when STELLAR_MICROPHYSICS=mesa but the bind(C)
# wrapper libraries are unavailable, skip the entire test session gracefully.
# ═══════════════════════════════════════════════════════════════

def _mesa_backend_requested():
    return os.environ.get('STELLAR_MICROPHYSICS', '').lower() == 'mesa'


def _mesa_available():
    """Check if MESA bind(C) wrapper libraries are usable."""
    try:
        from stellar_jax.microphysics.mesa.bindings import _load_lib
        _load_lib('eos_wrapper')
        return True
    except (ImportError, ValueError, OSError):
        return False


def _apply_mesa_skip(items):
    """Skip all tests when MESA backend is requested but unavailable."""
    if not _mesa_backend_requested():
        return
    if _mesa_available():
        return
    skip_mark = pytest.mark.skip(
        reason="STELLAR_MICROPHYSICS=mesa but MESA wrapper libraries unavailable"
    )
    for item in items:
        item.add_marker(skip_mark)

# NOTE: CI runs one task per smoke/integration test (orchestrator policy.parse_test_groups), PLUS a
# SINGLE bundled 'fast' task for the compile-LIGHT @fast validation tests (no evolve_star / jax.grad /
# oscillation-solver) — they compile ~nothing, so bundling does NOT serialize expensive compiles, and
# the bundle runs on Fargate. Compile-HEAVY tests must be @integration (own task), NOT @fast. The
# `fast` marker is NOT auto-applied — set it explicitly per that classification.



def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: fast sanity checks (<1 min each, no heavy JIT)")
    config.addinivalue_line("markers", "integration: expensive correctness validation, run in parallel on CI")
    config.addinivalue_line("markers", "suspended: temporarily disabled, not run in CI")
    config.addinivalue_line("markers", "relnotes: produces data for release notes (charts, tables, metrics)")
    config.addinivalue_line("markers", "fast: auto-applied — pure sanity test with no heavy compile; CI-bundled into one task")
    config.addinivalue_line("markers", "mesa_validation: MESA parity test — not collected by per-wave CI, runs nightly")
    # Per-mass bundle markers: tests requesting a bundle fixture are
    # grouped into per-mass CI tasks (one XLA compile per mass, shared across
    # all tests in the group).
    config.addinivalue_line("markers", "bundle_1p0: shares the 1.0 M☉ MODE-A bundle (session-scoped evolve_star)")
    config.addinivalue_line("markers", "bundle_1p2: shares the 1.2 M☉ MODE-A bundle (session-scoped evolve_star)")
    config.addinivalue_line("markers", "bundle_1p5: shares the 1.5 M☉ MODE-A bundle (session-scoped evolve_star)")
    config.addinivalue_line("markers", "bundle_2p0: shares the 2.0 M☉ MODE-A bundle (session-scoped evolve_star)")
    # O2 mutation gate
    config.addinivalue_line("markers", "validation: external-reference validation test (subject to the O2 mutation gate)")
    config.addinivalue_line("markers", "mutation(name): a named physics mutation that MUST break this validation test")
    # O2 right-reason check: tag the PHYSICS assertion that should break
    config.addinivalue_line("markers",
        "right_reason(pattern): substring that the failure message must contain under "
        "mutation, proving the mutation broke the PHYSICS assertion (not an incidental guard)")


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    """Free JIT compilation caches after each test to prevent OOM in long suites.

    Runs jax.clear_caches() in a 30s-bounded daemon thread: clear_caches() can
    hang indefinitely deallocating large compiled XLA graphs (heavy gradient /
    windowed-backward tests), and pytest-timeout counts teardown — so a hung
    cache-clear would fail an otherwise-passing test during teardown (this is
    what ratcheted grad_window_adaptive_regime's timeout to 7200s on the closed
    #160 branch). CI runs one test per task, so abandoning a hung clear is
    harmless — the process exits immediately after. Restores commit 568104ca,
    lost when #160 was closed.
    """
    import threading

    yield

    def _clear():
        try:
            jax.clear_caches()
        except Exception:
            pass
        gc.collect()

    t = threading.Thread(target=_clear, daemon=True)
    t.start()
    t.join(timeout=30)


@pytest.fixture(autouse=True)
def _timing_instrumentation(request, monkeypatch):
    """Wrap evolve_star AND jax.grad to log wall time per call (issues #303, #433).

    Forward calls: wraps stellar.evolve_star with block_until_ready + perf_counter.
    Backward calls: wraps jax.grad so the returned grad function, when called,
    logs the full gradient evaluation time (compile + execute) with phase='grad'.

    Transparent: tests behave identically. Results written to
    /tmp/test_timing.json at session end.
    """
    import sys as _sys
    import functools as _functools
    import importlib as _il
    mod = _il.import_module("stellar_jax.stellar")

    original = mod.evolve_star
    fwd_call_count = [0]

    def _is_tracer(x):
        """Check if x is a JAX tracer (not a concrete value).

        During jax.grad tracing, arguments become GradTracers. We must not
        call float() or other concretizing ops on them — it raises
        ConcretizationTypeError. Skip timing metadata when tracing; the
        outer timed_jax_grad wrapper captures the overall gradient time.
        """
        return isinstance(x, jax.core.Tracer)

    def timed_evolve_star(*args, **kwargs):
        # If any positional arg is a tracer, we're inside jax.grad tracing.
        # Just call the original — the grad wrapper handles timing at the
        # outer level. Logging side-effects can't run during tracing anyway.
        if args and _is_tracer(args[0]):
            return original(*args, **kwargs)

        t0 = time.perf_counter()
        r = original(*args, **kwargs)
        jax.block_until_ready(r['log_L'])
        dt = time.perf_counter() - t0
        fwd_call_count[0] += 1
        _timing_log.append({
            'test': request.node.nodeid,
            'call': fwd_call_count[0],
            'phase': 'forward',
            'mass': float(args[0]) if args else float(kwargs.get('mass', 0)),
            'max_steps': int(kwargs.get('max_steps', 500)),
            'wall_s': round(dt, 2),
        })
        return r

    monkeypatch.setattr(mod, 'evolve_star', timed_evolve_star)

    # --- Backward-pass (jax.grad) instrumentation ---
    original_grad = jax.grad
    grad_call_count = [0]
    grad_seen_sigs = set()

    def timed_jax_grad(fun, argnums=0, has_aux=False, **grad_kwargs):
        """Drop-in replacement for jax.grad that records per-call timing."""
        grad_fn = original_grad(fun, argnums=argnums, has_aux=has_aux, **grad_kwargs)

        @_functools.wraps(grad_fn)
        def timed_grad_wrapper(*args, **kwargs):
            # Build signature for compile-vs-execute detection
            sig_parts = [id(fun)]
            for a in args:
                if hasattr(a, 'shape') and hasattr(a, 'dtype'):
                    sig_parts.append((a.shape, str(a.dtype)))
                else:
                    sig_parts.append(type(a).__name__)
            sig = tuple(sig_parts)
            cached = sig in grad_seen_sigs

            t0 = time.perf_counter()
            result = grad_fn(*args, **kwargs)
            # Block until result is materialized (JAX async dispatch)
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
            elif isinstance(result, tuple):
                for r in result:
                    if hasattr(r, 'block_until_ready'):
                        r.block_until_ready()
            dt = time.perf_counter() - t0

            grad_seen_sigs.add(sig)
            grad_call_count[0] += 1

            entry = {
                'test': request.node.nodeid,
                'call': grad_call_count[0],
                'phase': 'grad',
                'wall_s': round(dt, 2),
                'cached': cached,
            }
            if not cached:
                entry['compile_plus_execute_s'] = entry['wall_s']
            else:
                entry['execute_s'] = entry['wall_s']

            _timing_log.append(entry)
            return result

        return timed_grad_wrapper

    monkeypatch.setattr(jax, 'grad', timed_jax_grad)
    yield


@pytest.fixture(autouse=True)
def _o2_mutation(request, monkeypatch):
    """O2 mutation gate. If STELLAR_MUTATION=<name> is set, apply that deliberate physics break
    (from tests/mutations.py) before the test runs. CI runs each @validation test clean (must PASS)
    AND under its declared mutation (must FAIL); a validation test that still passes under a real
    physics break is theater. With no env var set this is a no-op (normal runs unaffected)."""
    name = os.environ.get("STELLAR_MUTATION", "").strip()
    if not name:
        yield
        return
    # Disable JAX persistent compilation cache entirely during mutation runs.
    # Mutations monkeypatch @custom_vjp functions (e.g. zero_ift_eigenfreq replaces
    # eigenfreq_radial with eigenfreq_radial_zeroed). The forward jaxpr of both is
    # IDENTICAL (both just return sigma2_converged), so JAX's persistent cache — keyed
    # on jaxpr + input shapes — serves the stale XLA executable that includes the REAL
    # backward pass, making the mutation invisible. We disable the cache via TWO
    # mechanisms to be robust against initialization-order edge cases:
    #   1. Set enable_compilation_cache=False (checked by is_persistent_cache_enabled())
    #   2. Clear compilation_cache_dir to None (no dir → no file-based cache backend)
    # Together these ensure the EFS warm cache is never consulted, forcing JAX to trace
    # and compile from scratch with the monkeypatched functions.
    try:
        jax.config.update("jax_enable_compilation_cache", False)
    except Exception:
        pass
    try:
        jax.config.update("jax_compilation_cache_dir", "")
    except Exception:
        pass
    spec = importlib.util.spec_from_file_location(
        "stellar_mutations", os.path.join(os.path.dirname(__file__), "mutations.py"))
    muts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(muts)
    if name not in muts.MUTATIONS:
        pytest.fail(f"O2 mutation gate: unknown mutation '{name}' (not in tests/mutations.py registry)")
    stellar = request.getfixturevalue("stellar")
    muts.MUTATIONS[name](monkeypatch, stellar)
    yield


# ═══════════════════════════════════════════════════════════════
# O2 "right-reason" check
#
# When running under mutation (STELLAR_MUTATION set), the test MUST fail. The
# right-reason check goes further: if the test declares @right_reason("pattern"),
# the failure message must CONTAIN that pattern — proving the mutation broke the
# PHYSICS assertion, not an incidental guard (e.g. len(modes) >= N, converged==True).
#
# Implementation: a pytest_runtest_makereport hook captures the failure message
# and writes a JSON verdict to /tmp/o2_right_reason.json. The CI entrypoint
# (ci_entrypoint.sh) reads it after the mutation run to distinguish right-reason
# failures from incidental ones.
#
# Tests without @right_reason are still checked pass/fail only (backward compatible).
# ═══════════════════════════════════════════════════════════════

_right_reason_results = {}


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Capture failure details for the O2 right-reason check.

    When a test fails during CALL phase under mutation, record whether the
    failure message matches the declared @right_reason pattern.
    """
    mutation_name = os.environ.get("STELLAR_MUTATION", "").strip()
    if not mutation_name:
        return  # Not a mutation run — no right-reason check

    if call.when != "call":
        return  # Only check the test body, not setup/teardown

    rr_marker = item.get_closest_marker("right_reason")
    if rr_marker is None:
        return  # No right_reason declared — backward compatible

    if call.excinfo is None:
        return  # Test PASSED — the existing O2 pass/fail check handles this

    expected_pattern = rr_marker.args[0] if rr_marker.args else ""
    if not expected_pattern:
        return

    failure_msg = str(call.excinfo.value)
    matched = expected_pattern.lower() in failure_msg.lower()

    _right_reason_results[item.nodeid] = {
        'mutation': mutation_name,
        'expected_pattern': expected_pattern,
        'failure_msg': failure_msg[:500],
        'matched': matched,
    }

    if not matched:
        print(f"\nO2 RIGHT-REASON WARNING: {item.nodeid} failed under "
              f"mutation '{mutation_name}' but the failure message does NOT "
              f"contain the expected physics assertion pattern "
              f"'{expected_pattern}'.\n"
              f"  Actual failure: {failure_msg[:200]}\n"
              f"  This may indicate the mutation broke an incidental guard, "
              f"not the physics assertion.\n")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """O2 helper + MESA backend skip.

    1. When STELLAR_MICROPHYSICS=mesa but bind(C) wrapper libs unavailable, skip all tests gracefully.
    2. When O2_LIST_VALIDATION=1, write validation test list for the mutation gate.

    NOTE: trylast=True ensures this runs AFTER pytest's internal mark deselection
    (deselect_by_mark), so `items` only contains tests matching the active -m filter.
    This prevents the O2 list from including tests outside the bundle's scope."""
    _apply_mesa_skip(items)

    if os.environ.get("O2_LIST_VALIDATION") != "1":
        return
    out = os.environ.get("O2_LIST_FILE", "/tmp/o2_validation.tsv")
    lines = []
    for it in items:
        if it.get_closest_marker("validation") is None:
            continue
        # Skip @suspended tests — they are excluded from per-wave CI and their
        # mutation gates may be known-theater (e.g.). Including them in the
        # O2 list would fail the gate for tests that aren't actually running.
        if it.get_closest_marker("suspended") is not None:
            continue
        # Support multiple @mutation marks on one test — emit one line per mutation
        mutations = list(it.iter_markers("mutation"))
        if mutations:
            for m in mutations:
                mut = m.args[0] if m.args else ""
                lines.append(f"{it.nodeid}\t{mut}")
        else:
            lines.append(f"{it.nodeid}\t")
    with open(out, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def pytest_sessionfinish(session, exitstatus):
    """Write per-call timing log and right-reason results at end of session."""
    if _timing_log:
        out = os.environ.get('TIMING_OUTPUT', '/tmp/test_timing.json')
        with open(out, 'w') as f:
            json.dump(_timing_log, f, indent=2)

    # Write right-reason results for CI post-check
    if _right_reason_results:
        rr_out = os.environ.get('O2_RIGHT_REASON_FILE', '/tmp/o2_right_reason.json')
        with open(rr_out, 'w') as f:
            json.dump(_right_reason_results, f, indent=2)


@pytest.fixture(scope="session")
def stellar():
    """Load the stellar module."""
    return importlib.import_module("stellar_jax.stellar")
