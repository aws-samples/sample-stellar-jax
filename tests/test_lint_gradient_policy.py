"""Unit tests for scripts/lint_gradient_policy.py parsing logic.

Tests the lint script's internal functions in isolation with synthetic inputs,
covering edge cases for stop_gradient detection, GP-N tag matching, exclusion
logic, and docstring/comment handling.
"""

import os
import sys
import tempfile

import pytest

# Import the lint module's functions directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import lint_gradient_policy as lint


# ===========================================================================
# _is_excluded
# ===========================================================================

class TestIsExcluded:
    """Test file-path exclusion logic."""

    def test_tests_dir_excluded(self):
        assert lint._is_excluded('tests/test_foo.py')

    def test_tests_subdir_excluded(self):
        assert lint._is_excluded('tests/unit/test_bar.py')

    def test_docs_dir_excluded(self):
        assert lint._is_excluded('docs/design/example.py')

    def test_gradient_policy_file_excluded(self):
        assert lint._is_excluded('gradient_policy.py')

    def test_contracts_suffix_excluded(self):
        assert lint._is_excluded('solver/contracts.py')
        assert lint._is_excluded('composition/contracts.py')
        assert lint._is_excluded('mesh/contracts.py')

    def test_production_file_not_excluded(self):
        assert not lint._is_excluded('evolution/step.py')
        assert not lint._is_excluded('solver/newton.py')
        assert not lint._is_excluded('mesh/structure_mesh.py')

    def test_nested_production_not_excluded(self):
        assert not lint._is_excluded('evolution/_core.py')
        assert not lint._is_excluded('calibration/solar.py')

    def test_root_py_file_not_excluded(self):
        assert not lint._is_excluded('solver/conditioning_diagnostic.py')


# ===========================================================================
# _is_comment_or_docstring
# ===========================================================================

class TestIsCommentOrDocstring:
    """Test comment/docstring line detection."""

    def test_comment_line(self):
        assert lint._is_comment_or_docstring('# this is a comment')

    def test_indented_comment(self):
        assert lint._is_comment_or_docstring('    # indented comment')

    def test_triple_double_quote(self):
        assert lint._is_comment_or_docstring('"""docstring"""')

    def test_triple_single_quote(self):
        assert lint._is_comment_or_docstring("'''docstring'''")

    def test_code_line(self):
        assert not lint._is_comment_or_docstring('x = stop_gradient(y)')

    def test_code_with_trailing_comment(self):
        # The line IS code (not a comment line), even though it has a comment
        assert not lint._is_comment_or_docstring('x = 1  # GP-1')

    def test_empty_line(self):
        assert not lint._is_comment_or_docstring('')

    def test_whitespace_only(self):
        assert not lint._is_comment_or_docstring('    ')


# ===========================================================================
# _has_gp_tag_in_context
# ===========================================================================

class TestHasGpTagInContext:
    """Test GP-N tag detection in context window."""

    def test_tag_on_same_line(self):
        lines = ['x = jax.lax.stop_gradient(dt)  # GP-1: TIMESTEP\n']
        assert lint._has_gp_tag_in_context(lines, 0)

    def test_tag_one_line_before(self):
        lines = [
            '# GP-3: DIFFUSION\n',
            'result = jax.lax.stop_gradient(diffusion_out)\n',
        ]
        assert lint._has_gp_tag_in_context(lines, 1)

    def test_tag_three_lines_before(self):
        lines = [
            '# GP-5: STRUCTURE_TO_COMP\n',
            '# Additional context\n',
            '# More context\n',
            'shell_data_sg = jax.lax.stop_gradient(shell_data)\n',
        ]
        assert lint._has_gp_tag_in_context(lines, 3)

    def test_tag_four_lines_before_out_of_window(self):
        lines = [
            '# GP-5: STRUCTURE_TO_COMP\n',
            '# line 1\n',
            '# line 2\n',
            '# line 3\n',
            'shell_data_sg = jax.lax.stop_gradient(shell_data)\n',
        ]
        # Default window is 3, so 4 lines back is out of range
        assert not lint._has_gp_tag_in_context(lines, 4)

    def test_no_tag_anywhere(self):
        lines = [
            'x = 1\n',
            'y = 2\n',
            'z = jax.lax.stop_gradient(w)\n',
        ]
        assert not lint._has_gp_tag_in_context(lines, 2)

    def test_tag_with_varied_spacing(self):
        lines = ['x = stop_gradient(y)  #GP-9\n']
        assert lint._has_gp_tag_in_context(lines, 0)

    def test_tag_with_extra_space(self):
        lines = ['x = stop_gradient(y)  #  GP-2: MESH\n']
        assert lint._has_gp_tag_in_context(lines, 0)

    def test_first_line_of_file(self):
        """No preceding lines to check — must have tag on the line itself."""
        lines = ['x = stop_gradient(y)\n']
        assert not lint._has_gp_tag_in_context(lines, 0)

    def test_first_line_with_tag(self):
        lines = ['x = stop_gradient(y)  # GP-1\n']
        assert lint._has_gp_tag_in_context(lines, 0)


# ===========================================================================
# _SG_CALL_RE (regex matching)
# ===========================================================================

class TestStopGradientRegex:
    """Test the stop_gradient call regex pattern."""

    def test_matches_jax_lax_stop_gradient(self):
        assert lint._SG_CALL_RE.search('x = jax.lax.stop_gradient(y)')

    def test_matches_lax_stop_gradient(self):
        assert lint._SG_CALL_RE.search('x = lax.stop_gradient(y)')

    def test_matches_bare_stop_gradient(self):
        assert lint._SG_CALL_RE.search('from jax.lax import stop_gradient; stop_gradient(x)')

    def test_matches_with_space_before_paren(self):
        assert lint._SG_CALL_RE.search('stop_gradient (x)')

    def test_no_match_in_string_mention(self):
        # The regex matches the call pattern, so even in a string it would match
        # This is by design — the docstring/comment filter handles exclusion
        assert lint._SG_CALL_RE.search('"stop_gradient(x)"')

    def test_no_match_without_paren(self):
        assert not lint._SG_CALL_RE.search('# stop_gradient is used here')

    def test_no_match_partial_name(self):
        assert not lint._SG_CALL_RE.search('my_stop_gradient_fn(x)')  # has the word but different


# ===========================================================================
# _GP_TAG_RE (tag regex matching)
# ===========================================================================

class TestGpTagRegex:
    """Test the GP-N tag regex pattern."""

    def test_matches_gp_1(self):
        assert lint._GP_TAG_RE.search('# GP-1')

    def test_matches_gp_9(self):
        assert lint._GP_TAG_RE.search('# GP-9: CONVERGENCE_FLAGS')

    def test_matches_with_extra_space(self):
        assert lint._GP_TAG_RE.search('#  GP-3')

    def test_no_match_without_hash(self):
        assert not lint._GP_TAG_RE.search('GP-1')

    def test_no_match_gp_without_number(self):
        assert not lint._GP_TAG_RE.search('# GP-')

    def test_matches_double_digit(self):
        # Future-proofing: if we ever have GP-10+
        assert lint._GP_TAG_RE.search('# GP-10')


# ===========================================================================
# lint_file (integration of the above, with real temp files)
# ===========================================================================

class TestLintFile:
    """Test lint_file with synthetic files."""

    def _write_temp(self, content: str) -> str:
        """Write content to a temp .py file and return its path."""
        fd, path = tempfile.mkstemp(suffix='.py')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path

    def test_annotated_call_no_violation(self):
        path = self._write_temp(
            'import jax\n'
            'x = jax.lax.stop_gradient(dt)  # GP-1: TIMESTEP\n'
        )
        try:
            violations = lint.lint_file(path)
            assert violations == []
        finally:
            os.unlink(path)

    def test_unannotated_call_is_violation(self):
        path = self._write_temp(
            'import jax\n'
            'x = jax.lax.stop_gradient(dt)\n'
        )
        try:
            violations = lint.lint_file(path)
            assert len(violations) == 1
            assert violations[0][0] == 2  # line number
            assert 'stop_gradient' in violations[0][1]
        finally:
            os.unlink(path)

    def test_tag_on_preceding_line_ok(self):
        path = self._write_temp(
            '# GP-7: WINDOWED_SCAN\n'
            'carry = jax.lax.stop_gradient(carry_p1)\n'
        )
        try:
            violations = lint.lint_file(path)
            assert violations == []
        finally:
            os.unlink(path)

    def test_call_in_comment_not_flagged(self):
        path = self._write_temp(
            '# This uses stop_gradient(x) for policy 1\n'
            'y = 42\n'
        )
        try:
            violations = lint.lint_file(path)
            assert violations == []
        finally:
            os.unlink(path)

    def test_call_in_docstring_not_flagged(self):
        path = self._write_temp(
            '"""This module uses stop_gradient(x) for timestep."""\n'
            'y = 42\n'
        )
        try:
            violations = lint.lint_file(path)
            assert violations == []
        finally:
            os.unlink(path)

    def test_call_in_multiline_docstring_not_flagged(self):
        path = self._write_temp(
            '"""\n'
            'Example: stop_gradient(x) detaches x.\n'
            '"""\n'
            'y = 42\n'
        )
        try:
            violations = lint.lint_file(path)
            assert violations == []
        finally:
            os.unlink(path)

    def test_multiple_calls_mixed(self):
        """An unannotated call far enough from any GP tag is flagged."""
        path = self._write_temp(
            'a = lax.stop_gradient(x)  # GP-1: TIMESTEP\n'
            'b = 1\n'
            'c = 2\n'
            'd = 3\n'
            'e = lax.stop_gradient(y)\n'  # >3 lines from GP-1, no own tag
            'f = lax.stop_gradient(z)  # GP-3: DIFFUSION\n'
        )
        try:
            violations = lint.lint_file(path)
            assert len(violations) == 1
            assert violations[0][0] == 5  # line 5 is unannotated
        finally:
            os.unlink(path)

    def test_nearby_tag_covers_within_window(self):
        """A GP tag within 3 preceding lines covers the call (by design)."""
        path = self._write_temp(
            'a = lax.stop_gradient(x)  # GP-1: TIMESTEP\n'
            'b = lax.stop_gradient(y)\n'  # line 1's GP-1 is within 3 lines
        )
        try:
            violations = lint.lint_file(path)
            # The lint accepts this because GP-1 on line 1 is within window
            assert violations == []
        finally:
            os.unlink(path)

    def test_empty_file_no_violations(self):
        path = self._write_temp('')
        try:
            violations = lint.lint_file(path)
            assert violations == []
        finally:
            os.unlink(path)

    def test_nonexistent_file_no_violations(self):
        violations = lint.lint_file('/tmp/nonexistent_8374983.py')
        assert violations == []

    def test_multiline_stop_gradient_with_tag_above(self):
        """A stop_gradient call split across lines with tag above."""
        path = self._write_temp(
            '# GP-9: CONVERGENCE_FLAGS\n'
            'result = jax.lax.stop_gradient(\n'
            '    some_complex_expression\n'
            ')\n'
        )
        try:
            violations = lint.lint_file(path)
            assert violations == []
        finally:
            os.unlink(path)

    def test_straight_through_estimator_pattern(self):
        """The STE pattern: stop_gradient(hard - soft) + soft."""
        path = self._write_temp(
            'mask = jax.lax.stop_gradient(hard - soft) + soft  # GP-5: STRUCTURE_TO_COMP\n'
        )
        try:
            violations = lint.lint_file(path)
            assert violations == []
        finally:
            os.unlink(path)

    def test_indented_code_with_tag(self):
        """Indented code (inside a function) with annotation."""
        path = self._write_temp(
            'def foo():\n'
            '    dt = jax.lax.stop_gradient(dt_raw)  # GP-1: TIMESTEP\n'
            '    return dt\n'
        )
        try:
            violations = lint.lint_file(path)
            assert violations == []
        finally:
            os.unlink(path)

    def test_indented_code_without_tag(self):
        """Indented code (inside a function) missing annotation."""
        path = self._write_temp(
            'def foo():\n'
            '    dt = jax.lax.stop_gradient(dt_raw)\n'
            '    return dt\n'
        )
        try:
            violations = lint.lint_file(path)
            assert len(violations) == 1
            assert violations[0][0] == 2
        finally:
            os.unlink(path)

    def test_single_line_docstring_then_code(self):
        """Single-line docstring followed by actual code with stop_gradient."""
        path = self._write_temp(
            '"""Module doc."""\n'
            'x = lax.stop_gradient(y)  # GP-2: MESH_POSITIONS\n'
        )
        try:
            violations = lint.lint_file(path)
            assert violations == []
        finally:
            os.unlink(path)

    def test_triple_single_quote_docstring(self):
        """Triple single-quote docstring containing stop_gradient reference."""
        path = self._write_temp(
            "'''\n"
            "This uses stop_gradient(x) as an example.\n"
            "'''\n"
            "y = 42\n"
        )
        try:
            violations = lint.lint_file(path)
            assert violations == []
        finally:
            os.unlink(path)


# ===========================================================================
# End-to-end: main() via subprocess (integration sanity)
# ===========================================================================

class TestMainIntegration:
    """Test the lint script's main() function exit codes."""

    def test_passes_on_project(self):
        """The project itself must pass the lint (enforced in CI anyway)."""
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(
                os.path.dirname(__file__), '..', 'scripts',
                'lint_gradient_policy.py')],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), '..'))
        assert result.returncode == 0, f"Lint failed:\n{result.stdout}"

    def test_fails_on_unannotated(self):
        """A temporary file with an unannotated stop_gradient fails the lint."""
        # Create a temporary Python file in a non-excluded location
        tmpdir = tempfile.mkdtemp()
        tmpfile = os.path.join(tmpdir, 'bad_module.py')
        with open(tmpfile, 'w') as f:
            f.write('import jax\nx = jax.lax.stop_gradient(y)\n')

        import subprocess
        # Run lint from the tmpdir (so relative paths work)
        result = subprocess.run(
            [sys.executable, os.path.join(
                os.path.dirname(__file__), '..', 'scripts',
                'lint_gradient_policy.py')],
            capture_output=True, text=True,
            cwd=tmpdir)

        # Clean up
        os.unlink(tmpfile)
        os.rmdir(tmpdir)

        # The script should find the unannotated call and fail
        assert result.returncode == 1
        assert 'missing GP-N annotation' in result.stdout
