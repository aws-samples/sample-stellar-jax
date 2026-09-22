"""Verify the per-call timing instrumentation fixture (issue #303).

This test confirms that the autouse _timing_instrumentation fixture in conftest.py
correctly records evolve_star calls with wall time, mass, and max_steps metadata,
and that pytest_sessionfinish writes the JSON file.
"""
import json
import os

import pytest


@pytest.mark.smoke
def test_timing_records_evolve_star_call():
    """Calling evolve_star produces a timing record in _timing_log."""
    # Access the _timing_log from the actual conftest module (loaded by pytest)
    from tests import conftest
    initial_len = len(conftest._timing_log)

    import stellar_jax.stellar as stellar
    stellar.evolve_star(1.0, max_steps=2)

    assert len(conftest._timing_log) > initial_len, \
        "No timing record was appended after evolve_star call"
    entry = conftest._timing_log[-1]
    assert entry['mass'] == 1.0
    assert entry['max_steps'] == 2
    assert entry['wall_s'] >= 0
    assert 'test' in entry
    assert 'call' in entry


@pytest.mark.smoke
def test_timing_json_written_on_session_end(tmp_path, monkeypatch):
    """pytest_sessionfinish writes timing data to the configured output file."""
    from tests import conftest

    # Ensure at least one forward entry exists
    if not any(e.get('phase') == 'forward' or 'mass' in e for e in conftest._timing_log):
        import stellar_jax.stellar as stellar
        stellar.evolve_star(1.0, max_steps=2)

    out_file = str(tmp_path / "timing.json")
    monkeypatch.setenv("TIMING_OUTPUT", out_file)
    conftest.pytest_sessionfinish(session=None, exitstatus=0)

    assert os.path.exists(out_file)
    with open(out_file) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) >= 1
    # Check that at least one forward entry has the expected fields
    fwd_entries = [e for e in data if e.get('phase') == 'forward' or 'mass' in e]
    assert len(fwd_entries) >= 1, "No forward timing entries found"
    assert 'wall_s' in fwd_entries[-1]
    assert 'mass' in fwd_entries[-1]
