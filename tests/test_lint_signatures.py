"""Tests for scripts/lint_signatures.py — function-body-length soft cap.

Tests the function-length lint check (§11-G) in isolation with synthetic inputs:
  - _count_logical_lines: correct counting of non-blank, non-comment lines.
  - _has_exception_marker: scan-body exception marker detection.
  - check_function_body_length: end-to-end with temp source trees.
  - Integration test running the script via subprocess.

Pattern follows test_lint_gradient_policy.py — tempfile-based fixtures, no
project source dependency (tests pass on a bare checkout).
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the lint module's internal functions for unit testing.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import lint_signatures as lint


# ===========================================================================
# _count_logical_lines
# ===========================================================================

class TestCountLogicalLines:
    """Unit tests for logical-line counting."""

    def test_all_code_lines(self):
        lines = ['a = 1', 'b = 2', 'c = 3']
        assert lint._count_logical_lines(lines, 1, 3) == 3

    def test_blank_lines_excluded(self):
        lines = ['a = 1', '', 'b = 2', '', 'c = 3']
        assert lint._count_logical_lines(lines, 1, 5) == 3

    def test_comment_lines_excluded(self):
        lines = ['a = 1', '# comment', 'b = 2', '    # indented comment', 'c = 3']
        assert lint._count_logical_lines(lines, 1, 5) == 3

    def test_mixed_blank_and_comment(self):
        lines = ['a = 1', '', '# comment', '', '    # another', 'b = 2']
        assert lint._count_logical_lines(lines, 1, 6) == 2

    def test_single_line(self):
        lines = ['return x']
        assert lint._count_logical_lines(lines, 1, 1) == 1

    def test_single_blank_line(self):
        lines = ['']
        assert lint._count_logical_lines(lines, 1, 1) == 0

    def test_single_comment_line(self):
        lines = ['# just a comment']
        assert lint._count_logical_lines(lines, 1, 1) == 0

    def test_subrange(self):
        """Count only lines 2-4 of a 5-line source."""
        lines = ['a = 1', 'b = 2', '# skip', 'c = 3', 'd = 4']
        assert lint._count_logical_lines(lines, 2, 4) == 2

    def test_end_beyond_source(self):
        """end_line past source length doesn't crash."""
        lines = ['a = 1', 'b = 2']
        assert lint._count_logical_lines(lines, 1, 10) == 2

    def test_inline_comment_counts_as_code(self):
        """A line with code + trailing comment IS a logical line."""
        lines = ['x = 1  # inline comment']
        assert lint._count_logical_lines(lines, 1, 1) == 1

    def test_docstring_triple_quotes_count_as_code(self):
        """Triple-quote lines are NOT comments (they're code/docstrings)."""
        lines = ['"""Module doc."""', 'x = 1']
        assert lint._count_logical_lines(lines, 1, 2) == 2

    def test_docstring_excluded_with_body_nodes(self):
        """When body_nodes is passed, standalone string Expr lines are excluded."""
        import ast
        source = 'def f():\n    """A docstring\n    spanning lines.\n    """\n    x = 1\n    y = 2\n'
        tree = ast.parse(source)
        func = tree.body[0]
        source_lines = source.splitlines()
        body_start = func.body[0].lineno
        body_end = func.end_lineno
        # Without body_nodes: docstring lines count as code (3 docstring lines + 2 code = 5)
        without = lint._count_logical_lines(source_lines, body_start, body_end)
        assert without == 5
        # With body_nodes: docstring lines excluded (only 2 code lines)
        with_nodes = lint._count_logical_lines(source_lines, body_start, body_end, func.body)
        assert with_nodes == 2

    def test_multiline_string_expr_excluded(self):
        """A bare multi-line string constant (not just the first docstring) is excluded."""
        import ast
        source = (
            'def f():\n'
            '    """Docstring."""\n'
            '    x = 1\n'
            '    """Another bare string\n'
            '    that spans.\n'
            '    """\n'
            '    y = 2\n'
        )
        tree = ast.parse(source)
        func = tree.body[0]
        source_lines = source.splitlines()
        body_start = func.body[0].lineno
        body_end = func.end_lineno
        with_nodes = lint._count_logical_lines(source_lines, body_start, body_end, func.body)
        assert with_nodes == 2  # only x = 1, y = 2


# ===========================================================================
# _has_exception_marker
# ===========================================================================

class TestHasExceptionMarker:
    """Unit tests for scan-body exception marker detection."""

    def test_marker_on_def_line(self):
        lines = [
            'def step_fn(carry):  # lint: scan-body exception — lax.scan body',
        ]
        assert lint._has_exception_marker(lines, 1) is True

    def test_marker_one_line_before_def(self):
        lines = [
            '# lint: scan-body exception — fused kernel for XLA',
            'def step_fn(carry):',
        ]
        assert lint._has_exception_marker(lines, 2) is True

    def test_marker_three_lines_before_def(self):
        lines = [
            '# lint: scan-body exception — reason',
            '@jax.jit',
            '@functools.partial',
            'def step_fn(carry):',
        ]
        assert lint._has_exception_marker(lines, 4) is True

    def test_marker_four_lines_before_too_far(self):
        lines = [
            '# lint: scan-body exception — reason',
            '',
            '@jax.jit',
            '@functools.partial',
            'def step_fn(carry):',
        ]
        # 4 lines before def (line 5) is line 1 — outside the 3-line window
        assert lint._has_exception_marker(lines, 5) is False

    def test_no_marker(self):
        lines = [
            '# A regular comment',
            'def step_fn(carry):',
        ]
        assert lint._has_exception_marker(lines, 2) is False

    def test_partial_marker_not_matched(self):
        """A marker missing the 'exception' keyword should not match."""
        lines = [
            '# lint: scan-body — missing the required keyword',
            'def step_fn(carry):',
        ]
        assert lint._has_exception_marker(lines, 2) is False

    def test_marker_at_start_of_file(self):
        lines = [
            '# lint: scan-body exception — first line',
            'def f():',
        ]
        assert lint._has_exception_marker(lines, 2) is True

    def test_jit_boundary_exception_marker(self):
        """The generalized regex matches '# lint: jit-boundary exception'."""
        lines = [
            '# lint: jit-boundary exception — must stay single fn for XLA trace',
            'def _evolve_star_jit():',
        ]
        assert lint._has_exception_marker(lines, 2) is True

    def test_limiter_chain_exception_marker(self):
        """The generalized regex matches '# lint: limiter-chain exception'."""
        lines = [
            '    # lint: limiter-chain exception — 6 coupled dt limiters',
            '    def _step_dt_limiters():',
        ]
        assert lint._has_exception_marker(lines, 2) is True

    def test_sequential_setup_exception_marker(self):
        """The generalized regex matches '# lint: sequential-setup exception'."""
        lines = [
            '    # lint: sequential-setup exception — 2-pass ZAMS init',
            '    def _init_zams_and_warmup():',
        ]
        assert lint._has_exception_marker(lines, 2) is True


# ===========================================================================
# check_function_body_length — end-to-end with synthetic source trees
# ===========================================================================

def _make_src_tree(tmpdir, files):
    """Create a fake src/pkg/ tree inside tmpdir.

    files: dict of {relative_path: source_code}
    Returns the Path to the 'pkg' directory (the src_root to pass to the checker).
    """
    src_root = Path(tmpdir) / 'src' / 'pkg'
    for rel_path, code in files.items():
        fpath = src_root / rel_path
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(code)
    return src_root


def _make_long_function(name, n_lines, marker=None):
    """Generate source for a function with exactly n_lines logical body lines.

    Args:
        name: function name.
        n_lines: number of logical (non-blank, non-comment) body lines.
        marker: if set, prepend this as a comment before the def.
    """
    parts = []
    if marker:
        parts.append(marker)
    parts.append(f'def {name}():')
    for i in range(n_lines):
        parts.append(f'    x_{i} = {i}')
    parts.append('')  # trailing newline
    return '\n'.join(parts)


class TestCheckFunctionBodyLength:
    """End-to-end tests for check_function_body_length with temp source trees."""

    def test_under_cap_no_violations(self):
        """A function with exactly 100 logical lines is fine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = _make_src_tree(tmpdir, {
                'module.py': _make_long_function('short_fn', 100),
            })
            new_v, exempt_w = lint.check_function_body_length(src_root)
            assert new_v == []
            assert exempt_w == []

    def test_over_cap_flagged(self):
        """A function with 101 logical lines is flagged as a new violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = _make_src_tree(tmpdir, {
                'module.py': _make_long_function('big_fn', 101),
            })
            new_v, exempt_w = lint.check_function_body_length(src_root)
            assert len(new_v) == 1
            assert new_v[0][2] == 'big_fn'
            assert new_v[0][3] == 101
            assert exempt_w == []

    def test_exception_marker_suppresses(self):
        """A function with the scan-body exception marker is not flagged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = _make_src_tree(tmpdir, {
                'module.py': _make_long_function(
                    'scan_body', 120,
                    marker='# lint: scan-body exception — lax.scan step body'
                ),
            })
            new_v, exempt_w = lint.check_function_body_length(src_root)
            assert new_v == []
            assert exempt_w == []

    def test_exempt_function_is_warned_not_failed(self):
        """A function in BODY_LENGTH_EXEMPT lands in exempt_warnings, not violations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = _make_src_tree(tmpdir, {
                'module.py': _make_long_function('exempt_fn', 120),
            })
            # Temporarily add a matching entry to the exempt set
            fake_key = ('src/pkg/module.py', 'exempt_fn')
            lint.BODY_LENGTH_EXEMPT.add(fake_key)
            try:
                new_v, exempt_w = lint.check_function_body_length(src_root)
                assert new_v == []
                assert len(exempt_w) == 1
                assert exempt_w[0][2] == 'exempt_fn'
                assert exempt_w[0][3] == 120
            finally:
                lint.BODY_LENGTH_EXEMPT.discard(fake_key)

    def test_blank_and_comment_lines_not_counted(self):
        """Blank lines and comments in a function body don't count."""
        code = (
            'def padded_fn():\n'
            '    x = 1\n'
            '\n'
            '    # comment\n'
            '    y = 2\n'
            '\n'
            '    # another comment\n'
            '\n'
            '    z = 3\n'
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = _make_src_tree(tmpdir, {'module.py': code})
            new_v, exempt_w = lint.check_function_body_length(src_root)
            # Only 3 logical lines — well under the cap
            assert new_v == []
            assert exempt_w == []

    def test_multiple_functions_mixed(self):
        """Multiple functions: one over, one under, one excepted."""
        small_fn = _make_long_function('small_fn', 40)
        big_fn = _make_long_function('big_fn', 110)
        excepted_fn = _make_long_function(
            'scan_fn', 100,
            marker='# lint: scan-body exception — XLA fusion'
        )
        code = small_fn + '\n' + big_fn + '\n' + excepted_fn
        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = _make_src_tree(tmpdir, {'module.py': code})
            new_v, exempt_w = lint.check_function_body_length(src_root)
            assert len(new_v) == 1
            assert new_v[0][2] == 'big_fn'
            assert exempt_w == []

    def test_nested_function_counted_separately(self):
        """An inner function is counted independently from the outer."""
        # Outer: 5 logical lines (including the inner def + its body counted
        # at the outer level). Inner: 105 logical lines (over cap).
        inner_body = '\n'.join(f'        x_{i} = {i}' for i in range(105))
        code = (
            'def outer():\n'
            '    a = 1\n'
            '    def inner():\n'
            + inner_body + '\n'
            '    return a\n'
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = _make_src_tree(tmpdir, {'module.py': code})
            new_v, exempt_w = lint.check_function_body_length(src_root)
            # inner is over the cap; outer may or may not be depending on
            # how its body lines are counted (the inner's lines are part of
            # outer's body range too). We only need to confirm inner IS flagged.
            inner_violations = [v for v in new_v if v[2] == 'inner']
            assert len(inner_violations) == 1
            assert inner_violations[0][3] == 105

    def test_empty_dir_no_crash(self):
        """An empty source tree produces no violations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = _make_src_tree(tmpdir, {})
            new_v, exempt_w = lint.check_function_body_length(src_root)
            assert new_v == []
            assert exempt_w == []

    def test_syntax_error_file_skipped(self):
        """A file with a syntax error is skipped without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = _make_src_tree(tmpdir, {
                'bad.py': 'def broken(\n',
            })
            new_v, exempt_w = lint.check_function_body_length(src_root)
            assert new_v == []
            assert exempt_w == []

    def test_async_function_counted(self):
        """Async functions are counted the same as sync functions."""
        parts = ['async def big_async():']
        for i in range(105):
            parts.append(f'    x_{i} = {i}')
        code = '\n'.join(parts) + '\n'
        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = _make_src_tree(tmpdir, {'module.py': code})
            new_v, exempt_w = lint.check_function_body_length(src_root)
            assert len(new_v) == 1
            assert new_v[0][2] == 'big_async'

    def test_exactly_at_cap_passes(self):
        """A function with exactly MAX_BODY_LINES (100) logical lines passes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = _make_src_tree(tmpdir, {
                'module.py': _make_long_function('boundary_fn', 100),
            })
            new_v, _ = lint.check_function_body_length(src_root)
            assert new_v == []

    def test_one_over_cap_fails(self):
        """A function with MAX_BODY_LINES+1 (101) logical lines fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = _make_src_tree(tmpdir, {
                'module.py': _make_long_function('boundary_fn', 101),
            })
            new_v, _ = lint.check_function_body_length(src_root)
            assert len(new_v) == 1


# ===========================================================================
# Integration: main() via subprocess
# ===========================================================================

class TestMainIntegration:
    """Test the lint script's main() exit code on the real project."""

    def test_passes_on_project(self):
        """The project itself must pass the lint (all over-cap are exempted)."""
        result = subprocess.run(
            [sys.executable, os.path.join(
                os.path.dirname(__file__), '..', 'scripts',
                'lint_signatures.py')],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), '..'))
        assert result.returncode == 0, (
            f"lint_signatures.py failed:\n{result.stdout}\n{result.stderr}")

    def test_no_exempt_functions_remain(self):
        """All pre-existing over-cap functions have been split.

        BODY_LENGTH_EXEMPT is now empty — no exemptions remain.
        """
        result = subprocess.run(
            [sys.executable, os.path.join(
                os.path.dirname(__file__), '..', 'scripts',
                'lint_signatures.py')],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), '..'))
        # No exemptions should remain — zero WARN lines expected.
        import re
        exempt_lines = re.findall(
            r'^\s+\S+\.py:\d+:.*—\s*\d+\s*logical lines',
            result.stdout, re.MULTILINE)
        assert len(exempt_lines) == 0, (
            f"BODY_LENGTH_EXEMPT should be empty but found "
            f"{len(exempt_lines)} exempted functions:\n"
            + '\n'.join(exempt_lines))


# ===========================================================================
# Real-tree CI gate — the function-length lint must pass on src/stellar_jax/
# ===========================================================================

@pytest.mark.fast
@pytest.mark.validation
class TestRealTree:
    """CI-gated: the real source tree must pass the function-length lint.

    What: runs check_function_body_length() against the real src/stellar_jax/
    tree and asserts zero new violations (all over-cap functions are either
    exempted or marked with the scan-body exception).

    Why: issue #1108 — the function-length lint existed but had no per-wave CI
    gate, so violations could slip onto main without blocking a merge. This
    class, marked @validation @fast, ensures the orchestrator runs it every
    wave and a violation FAILS CI.

    External reference: design-standard §11-G (function ≤100 logical lines).
    No physics reference — this is a code-quality meta-test.
    """

    def test_real_tree_is_green(self):
        """The function-length lint must report zero new violations on the
        current source tree."""
        new_violations, _exempt = lint.check_function_body_length()
        if new_violations:
            msg_lines = [
                "Function-length lint FAILED — new over-cap functions "
                "(not in BODY_LENGTH_EXEMPT):"
            ]
            for rel_path, lineno, name, logical in sorted(
                new_violations, key=lambda x: -x[3]
            ):
                msg_lines.append(
                    f"  {rel_path}:{lineno}: {name}() — {logical} logical lines"
                )
            pytest.fail("\n".join(msg_lines))

    def test_exempt_functions_still_exist_and_over_cap(self):
        """Every entry in BODY_LENGTH_EXEMPT must (a) still exist in its
        file and (b) still be over the cap — otherwise the exemption is
        stale and should be retired.

        This catches two kinds of staleness:
        - A file was removed or renamed (file no longer exists).
        - A function was decomposed below the cap (the exemption is now
          unnecessary — remove it so the lint catches future growth).
        """
        import ast as _ast

        repo_root = lint.SRC_ROOT.parent.parent
        for rel_path, func_name in sorted(lint.BODY_LENGTH_EXEMPT):
            full = repo_root / rel_path
            assert full.exists(), (
                f"Exempt file {rel_path} no longer exists — remove "
                f"({rel_path!r}, {func_name!r}) from BODY_LENGTH_EXEMPT"
            )

            # Parse the file and find the function
            source = full.read_text()
            source_lines = source.splitlines()
            tree = _ast.parse(source, filename=str(full))
            found = False
            for node in _ast.walk(tree):
                if (isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                        and node.name == func_name):
                    found = True
                    body_start = node.body[0].lineno
                    body_end = getattr(node, 'end_lineno', None) or body_start
                    logical = lint._count_logical_lines(
                        source_lines, body_start, body_end, node.body
                    )
                    assert logical > lint.MAX_BODY_LINES, (
                        f"Exempt function {func_name} in {rel_path} is now "
                        f"{logical} logical lines (≤ {lint.MAX_BODY_LINES}) "
                        f"— retire ({rel_path!r}, {func_name!r}) from "
                        f"BODY_LENGTH_EXEMPT"
                    )
                    break
            assert found, (
                f"Exempt function {func_name!r} not found in {rel_path} "
                f"— remove ({rel_path!r}, {func_name!r}) from "
                f"BODY_LENGTH_EXEMPT"
            )


# ===========================================================================
# Mutation-gated meta-test — the acceptance criterion for
# ===========================================================================

@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("inflate_body_line_cap")
class TestBodyLineCap:
    """Mutation-gated meta-test for the function-body-length soft cap.

    What: verifies that check_function_body_length() using the MODULE-LEVEL
    MAX_BODY_LINES (not a hardcoded constant) detects over-cap functions,
    passes under-cap functions, and honors the exemption list.

    Why: issue #1108 — a ≤100-line soft cap prevents god-functions. This class
    exercises the DEFAULT cap path (the module-level MAX_BODY_LINES that the
    mutation inflates) so that the mutation proves the cap is load-bearing.

    External reference: design-standard §11-G (function ≤100 logical lines).
    No physics reference — code-quality meta-test.

    Mutation: ``inflate_body_line_cap`` — inflates ``lint.MAX_BODY_LINES``
    from 100 to 999999. Under this mutation, a 101-line fixture is UNDER the
    inflated cap, so the violation assertion fails (proving the cap value is
    load-bearing and the test is non-vacuous).
    """

    def test_over_cap_detected_via_default_cap(self):
        """A function with 101 logical lines is flagged as a violation when
        the cap is lint.MAX_BODY_LINES (the module-level value the mutation
        targets).

        Under clean run: MAX_BODY_LINES=100, 101 > 100 → 1 violation.
        Under mutation:  MAX_BODY_LINES=999999, 101 < 999999 → 0 → FAIL.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = _make_src_tree(tmpdir, {
                'module.py': _make_long_function('big_fn', 101),
            })
            new_v, exempt_w = lint.check_function_body_length(src_root)
            assert len(new_v) == 1, (
                f"Expected 1 violation for a 101-line function "
                f"at cap={lint.MAX_BODY_LINES}, got {len(new_v)}"
            )
            assert new_v[0][2] == 'big_fn'
            assert new_v[0][3] == 101

    def test_under_cap_clean_via_default_cap(self):
        """A function with exactly 100 lines is clean at MAX_BODY_LINES=100."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = _make_src_tree(tmpdir, {
                'module.py': _make_long_function('ok_fn', 100),
            })
            new_v, exempt_w = lint.check_function_body_length(src_root)
            assert new_v == []

    def test_exemption_via_default_cap(self):
        """An over-cap function in BODY_LENGTH_EXEMPT is exempt, not a violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = _make_src_tree(tmpdir, {
                'module.py': _make_long_function('exempt_fn', 120),
            })
            fake_key = ('src/pkg/module.py', 'exempt_fn')
            lint.BODY_LENGTH_EXEMPT.add(fake_key)
            try:
                new_v, exempt_w = lint.check_function_body_length(src_root)
                assert new_v == []
                assert len(exempt_w) == 1
                assert exempt_w[0][2] == 'exempt_fn'
            finally:
                lint.BODY_LENGTH_EXEMPT.discard(fake_key)