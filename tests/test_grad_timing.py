"""Verify backward-pass (jax.grad) timing instrumentation (issue #433).

The conftest autouse _timing_instrumentation fixture wraps jax.grad so that
each gradient evaluation logs wall time and compile-vs-execute status (via the
'cached' field). This test confirms that calling jax.grad(f)(x) produces a
timing entry with phase='grad' in the _timing_log.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


@pytest.mark.fast
def test_grad_timing_recorded():
    """Calling jax.grad(f)(x) produces a timing entry with phase='grad'."""
    from tests import conftest

    initial_len = len(conftest._timing_log)

    # A simple differentiable function — no evolve_star needed; we just need
    # to verify the jax.grad wrapper logs timing.
    f = lambda x: x ** 2 + 3.0 * x
    x = jnp.float64(2.0)

    g = jax.grad(f)(x)

    # The grad wrapper should have appended an entry
    grad_entries = [e for e in conftest._timing_log[initial_len:]
                    if e.get('phase') == 'grad']
    assert len(grad_entries) >= 1, (
        f"No 'grad' phase entry found in _timing_log after jax.grad call. "
        f"Log entries since start: {conftest._timing_log[initial_len:]}"
    )
    entry = grad_entries[-1]
    assert entry['phase'] == 'grad'
    assert entry['wall_s'] >= 0
    assert 'cached' in entry
    assert 'test' in entry
    assert 'call' in entry
    # First call reports compile_plus_execute_s; cached calls report execute_s
    assert 'compile_plus_execute_s' in entry or 'execute_s' in entry, (
        f"Expected compile_plus_execute_s or execute_s in entry: {entry}"
    )

    # Verify the gradient is correct (sanity: f'(2) = 2*2+3 = 7)
    assert abs(float(g) - 7.0) < 1e-10


@pytest.mark.fast
def test_grad_timing_cached_detection():
    """Second call with same signature shows cached=True."""
    from tests import conftest

    f = lambda x: jnp.sin(x) * x
    x = jnp.float64(1.0)

    # First call — should be cached=False (first time this signature seen)
    initial_len = len(conftest._timing_log)
    _ = jax.grad(f)(x)
    first_entries = [e for e in conftest._timing_log[initial_len:]
                     if e.get('phase') == 'grad']
    assert len(first_entries) >= 1
    # Note: cached may be True if a prior test already called grad with this
    # exact signature. We verify the field exists and is boolean.
    assert isinstance(first_entries[-1]['cached'], bool)

    # Second call with same function and same-shaped arg
    second_len = len(conftest._timing_log)
    _ = jax.grad(f)(x)
    second_entries = [e for e in conftest._timing_log[second_len:]
                      if e.get('phase') == 'grad']
    assert len(second_entries) >= 1
    # Same function id + same shape/dtype → cached should be True
    assert second_entries[-1]['cached'] is True


@pytest.mark.smoke
@pytest.mark.timeout(5400)
def test_grad_timing_with_evolve_star():
    """jax.grad through evolve_star logs a grad timing entry.

    When jax.grad(f)(x) is called, JAX traces f (evolve_star receives a tracer,
    so the forward wrapper correctly skips logging), then compiles and runs the
    combined forward+backward pass as a single XLA program. The timed_jax_grad
    wrapper captures the total wall time for this compiled gradient evaluation.

    A separate 'forward' entry only appears if the user makes an explicit f(x)
    call outside of jax.grad — the forward pass INSIDE the gradient is part of
    the compiled XLA graph, not a separate Python-level call.
    """
    from tests import conftest
    import stellar_jax.stellar as stellar

    initial_len = len(conftest._timing_log)

    # A very short evolution — still triggers XLA compilation (~10-16 min cold).
    # This test verifies the wiring: jax.grad(evolve_star) produces a grad entry.
    f = lambda m: stellar.evolve_star(m, Z=0.014, max_steps=2)['log_L'][-1]
    x = jnp.float64(1.0)
    g = jax.grad(f)(x)

    # The grad wrapper should have logged the overall gradient timing
    new_entries = conftest._timing_log[initial_len:]
    grad_entries = [e for e in new_entries if e.get('phase') == 'grad']

    assert len(grad_entries) >= 1, (
        f"No 'grad' entry after jax.grad(evolve_star). Entries: {new_entries}"
    )
    # The grad wall time should be positive (compile + execute)
    assert grad_entries[-1]['wall_s'] > 0
    # First call should be uncached (triggers compilation)
    assert grad_entries[-1]['cached'] is False
    # Should report compile_plus_execute_s for the first call
    assert 'compile_plus_execute_s' in grad_entries[-1]
    # The gradient should be a finite number
    assert jnp.isfinite(g)


@pytest.mark.fast
def test_grad_timing_json_includes_grad_phase(tmp_path, monkeypatch):
    """pytest_sessionfinish output includes grad-phase entries."""
    from tests import conftest
    import json

    # Ensure at least one grad entry exists
    f = lambda x: x ** 3
    _ = jax.grad(f)(jnp.float64(1.0))

    out_file = str(tmp_path / "timing.json")
    monkeypatch.setenv("TIMING_OUTPUT", out_file)
    conftest.pytest_sessionfinish(session=None, exitstatus=0)

    assert os.path.exists(out_file)
    with open(out_file) as fp:
        data = json.load(fp)
    grad_entries = [e for e in data if e.get('phase') == 'grad']
    assert len(grad_entries) >= 1, (
        f"No grad-phase entries in timing JSON. All entries: {data}"
    )
