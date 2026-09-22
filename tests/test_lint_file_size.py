"""Unit tests for scripts/lint_file_size.py.

What: tests the file-size soft-cap lint (count_logical_lines, scan, and the
exemption mechanism) using synthetic temporary files — never real source files
whose line counts could change.

Why: issue #1091 adds a ~1000-logical-line soft cap so modules can't silently
grow into monoliths.  This test proves the lint correctly:
  - counts only non-blank, non-comment lines (count_logical_lines);
  - flags a file over the cap as a violation;
  - passes a file under the cap;
  - honors the exemption list (an over-cap file with a linked issue is
    reported as "exempted", not as a violation);
  - reports a NEW over-cap file (not in the exemption list) as a violation.

External reference: the lint implements design-standard §G and §10 (file
caps); no MESA/physics reference — this is a code-quality meta-test.

Mutation: "inflate_file_size_cap" — sets SOFT_CAP to a huge value so no file
is ever over the cap.  Under this mutation, the default-cap scan never reports
violations, so test_file_size_cap FAILS (proving the cap value is load-bearing).
"""

from __future__ import annotations

import os
import sys
import textwrap
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the lint module from scripts/
# ---------------------------------------------------------------------------
_scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _scripts_dir)

import lint_file_size as lint  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — tiny, deterministic temp trees
# ---------------------------------------------------------------------------

def _write_file(directory: Path, name: str, content: str) -> Path:
    """Write *content* to *directory/name* and return the Path."""
    p = directory / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _make_lines(n_logical: int, *, blanks: int = 5, comments: int = 5) -> str:
    """Generate a Python file body with exactly *n_logical* logical lines.

    Logical lines = non-blank, non-comment.  We pad with *blanks* blank lines
    and *comments* comment-only lines to verify they are NOT counted.
    """
    parts: list[str] = []
    parts.append("# This is a comment — should NOT count as a logical line")
    for _ in range(comments - 1):
        parts.append("# padding comment")
    for _ in range(blanks):
        parts.append("")
    for i in range(n_logical):
        parts.append(f"x_{i} = {i}")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# count_logical_lines
# ---------------------------------------------------------------------------

class TestCountLogicalLines:
    """Verify the counting logic on small synthetic files."""

    def test_empty_file(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "empty.py", "")
        assert lint.count_logical_lines(p) == 0

    def test_only_comments(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "comments.py", "# comment\n# another\n")
        assert lint.count_logical_lines(p) == 0

    def test_only_blanks(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "blanks.py", "\n\n\n")
        assert lint.count_logical_lines(p) == 0

    def test_mixed_content(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            # header comment
            import os

            def foo():
                # inline comment
                return 1
        """)
        p = _write_file(tmp_path, "mixed.py", content)
        # Logical lines: "import os", "def foo():", "return 1" = 3
        assert lint.count_logical_lines(p) == 3

    def test_docstring_body_counts(self, tmp_path: Path) -> None:
        content = textwrap.dedent('''\
            def foo():
                """Docstring line 1.

                Docstring line 3.
                """
                return 1
        ''')
        p = _write_file(tmp_path, "docstring.py", content)
        # Logical lines: "def foo():", '"""Docstring line 1.',
        # "Docstring line 3.", '"""', "return 1" = 5.
        # The blank line inside the docstring does not count.
        assert lint.count_logical_lines(p) == 5

    def test_exact_count_generated(self, tmp_path: Path) -> None:
        """_make_lines(N) produces exactly N logical lines."""
        for n in (0, 1, 50, 1001):
            content = _make_lines(n)
            p = _write_file(tmp_path, f"gen_{n}.py", content)
            assert lint.count_logical_lines(p) == n, f"expected {n} logical lines"


# ---------------------------------------------------------------------------
# scan — the core testable API
# ---------------------------------------------------------------------------

class TestScan:
    """Verify the scan() function on temporary directory trees."""

    def test_under_cap_clean(self, tmp_path: Path) -> None:
        """A tree where every file is under the cap → 0 violations, 0 exempt."""
        _write_file(tmp_path, "small.py", _make_lines(500))
        _write_file(tmp_path, "medium.py", _make_lines(999))
        violations, exempted = lint.scan(root=tmp_path, cap=1000, exemptions={})
        assert violations == []
        assert exempted == []

    def test_over_cap_violation(self, tmp_path: Path) -> None:
        """A file over the cap with no exemption → 1 violation."""
        _write_file(tmp_path, "big.py", _make_lines(1001))
        violations, exempted = lint.scan(root=tmp_path, cap=1000, exemptions={})
        assert len(violations) == 1
        assert violations[0][1] == 1001
        assert exempted == []

    def test_exactly_at_cap_is_clean(self, tmp_path: Path) -> None:
        """A file with exactly 1000 logical lines → no violation."""
        _write_file(tmp_path, "exact.py", _make_lines(1000))
        violations, exempted = lint.scan(root=tmp_path, cap=1000, exemptions={})
        assert violations == []
        assert exempted == []

    def test_exemption_honoured(self, tmp_path: Path) -> None:
        """An over-cap file IN the exemption list → exempted, not a violation."""
        _write_file(tmp_path, "big.py", _make_lines(1200))
        exemptions = {"big.py": "#9999"}
        violations, exempted = lint.scan(
            root=tmp_path, cap=1000, exemptions=exemptions,
        )
        assert violations == []
        assert len(exempted) == 1
        assert exempted[0][1] == 1200
        assert exempted[0][2] == "#9999"

    def test_new_over_cap_not_exempt(self, tmp_path: Path) -> None:
        """An over-cap file NOT in the exemption list → violation even if
        other files are exempt."""
        _write_file(tmp_path, "known.py", _make_lines(1500))
        _write_file(tmp_path, "new_big.py", _make_lines(1100))
        exemptions = {"known.py": "#1089"}
        violations, exempted = lint.scan(
            root=tmp_path, cap=1000, exemptions=exemptions,
        )
        assert len(violations) == 1
        assert "new_big" in str(violations[0][0])
        assert len(exempted) == 1

    def test_subdirectory_files(self, tmp_path: Path) -> None:
        """Files in subdirectories are found and checked."""
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_file(sub, "deep.py", _make_lines(1050))
        violations, exempted = lint.scan(root=tmp_path, cap=1000, exemptions={})
        assert len(violations) == 1
        assert violations[0][1] == 1050

    def test_subdirectory_exemption_key(self, tmp_path: Path) -> None:
        """Exemption keys use paths relative to root, including subdirs."""
        sub = tmp_path / "evolution"
        sub.mkdir()
        _write_file(sub, "_core.py", _make_lines(1600))
        exemptions = {"evolution/_core.py": "#1089"}
        violations, exempted = lint.scan(
            root=tmp_path, cap=1000, exemptions=exemptions,
        )
        assert violations == []
        assert len(exempted) == 1

    def test_data_dir_excluded(self, tmp_path: Path) -> None:
        """Files in a 'data' subdirectory are not scanned."""
        data = tmp_path / "data"
        data.mkdir()
        _write_file(data, "huge.py", _make_lines(5000))
        violations, exempted = lint.scan(root=tmp_path, cap=1000, exemptions={})
        assert violations == []
        assert exempted == []


# ---------------------------------------------------------------------------
# Integration: run the lint on the real tree
# ---------------------------------------------------------------------------

@pytest.mark.fast
@pytest.mark.validation
class TestRealTree:
    """Smoke-test the lint on the actual src/stellar_jax/ tree.

    This does NOT pin exact counts (they change as the codebase evolves); it
    only checks that the lint PASSES (exit-code 0) — i.e. all over-cap files
    are accounted for in the exemption list.
    """

    def test_real_tree_is_green(self) -> None:
        """The lint should be green on the current tree (all over-cap files
        are in the exemption list)."""
        violations, _exempted = lint.scan()
        if violations:
            msg_lines = ["Unexpected over-cap files (not in exemption list):"]
            for path, n in violations:
                rel = path.relative_to(lint.SRC_ROOT)
                msg_lines.append(f"  {rel}: {n} logical lines")
            pytest.fail("\n".join(msg_lines))

    def test_exempted_files_still_exist(self) -> None:
        """Every file in the exemption list must still exist (otherwise the
        exemption is stale and should be removed)."""
        for rel_path in lint.EXEMPTIONS:
            full = lint.SRC_ROOT / rel_path
            assert full.exists(), (
                f"Exempted file {rel_path} no longer exists — remove it from "
                f"EXEMPTIONS in scripts/lint_file_size.py"
            )

    def test_exempted_files_are_actually_over_cap(self) -> None:
        """Every exempted file should actually be over the cap (otherwise the
        decomposition is done and the exemption should be retired)."""
        for rel_path in lint.EXEMPTIONS:
            full = lint.SRC_ROOT / rel_path
            if not full.exists():
                continue  # caught by the test above
            n = lint.count_logical_lines(full)
            assert n > lint.SOFT_CAP, (
                f"Exempted file {rel_path} is now {n} logical lines "
                f"(≤ {lint.SOFT_CAP}) — retire its exemption!"
            )


# ---------------------------------------------------------------------------
# Mutation-gated meta-test — the acceptance criterion for
# ---------------------------------------------------------------------------

@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("inflate_file_size_cap")
class TestFileSizeCap:
    """Mutation-gated meta-test for the file-size soft cap.

    What: verifies that scan() using the MODULE-LEVEL SOFT_CAP (not a
    hardcoded constant) detects over-cap files, passes under-cap files,
    and honors the exemption list.

    Why: issue #1091 — a ~1000-line soft cap prevents monolith growth.
    This class exercises the DEFAULT cap path (``scan(root, cap=lint.SOFT_CAP)``)
    so that the mutation (which inflates SOFT_CAP from 1000 to 999999) defeats
    the check.

    External reference: design-standard §G (file-size soft cap); no physics
    reference — code-quality meta-test.

    Mutation: ``inflate_file_size_cap`` — inflates ``lint.SOFT_CAP`` from
    1000 to 999999.  Under this mutation, a 1001-line fixture is UNDER the
    inflated cap, so the violation assertion fails (proving the cap is
    load-bearing and the test is non-vacuous).
    """

    def test_over_cap_detected_via_default_cap(self, tmp_path: Path) -> None:
        """A file with 1001 logical lines is flagged as a violation when
        the cap is lint.SOFT_CAP (the module-level value the mutation targets).

        Under clean run: SOFT_CAP=1000, 1001 > 1000 → 1 violation.
        Under mutation:  SOFT_CAP=999999, 1001 < 999999 → 0 violations → FAIL.
        """
        _write_file(tmp_path, "big.py", _make_lines(1001))
        violations, exempted = lint.scan(
            root=tmp_path, cap=lint.SOFT_CAP, exemptions={},
        )
        assert len(violations) == 1, (
            f"Expected 1 violation for a 1001-line file "
            f"at cap={lint.SOFT_CAP}, got {len(violations)}"
        )
        assert violations[0][1] == 1001

    def test_under_cap_clean_via_default_cap(self, tmp_path: Path) -> None:
        """A file with exactly 1000 lines is clean at SOFT_CAP=1000."""
        _write_file(tmp_path, "ok.py", _make_lines(1000))
        violations, exempted = lint.scan(
            root=tmp_path, cap=lint.SOFT_CAP, exemptions={},
        )
        assert violations == []

    def test_exemption_via_default_cap(self, tmp_path: Path) -> None:
        """An over-cap file in the exemption list is exempt, not a violation."""
        _write_file(tmp_path, "known.py", _make_lines(1500))
        exemptions = {"known.py": "#9999"}
        violations, exempted = lint.scan(
            root=tmp_path, cap=lint.SOFT_CAP, exemptions=exemptions,
        )
        assert violations == []
        assert len(exempted) == 1
        assert exempted[0][2] == "#9999"