#!/usr/bin/env python3
"""Lint guard: soft file-size cap (~1000 logical lines per source file).

Scans ``src/stellar_jax/**/*.py`` and reports every file whose *logical* line
count (non-blank, non-comment) exceeds the cap.  Pre-existing over-cap files
are tracked in an exemption list (``scripts/file_size_exemptions.yml``) with a
linked decomposition issue; they are reported as *exempt* rather than as
violations, so the check is green today and exemptions retire as files are
split.

"Logical line" = a line that is neither blank nor a full-line ``#``-comment
(including shebangs).  Docstring body lines ARE counted — they carry real
content that a reviewer must hold in their head.

Usage:
    python scripts/lint_file_size.py          # exits 0 if clean, 1 if violations
    python scripts/lint_file_size.py --show   # also list exempt files + counts

Scope: ``src/stellar_jax/**/*.py`` (library source).  Test files, data files,
generated code, and everything outside the package are excluded.

Design standard §G / issue #1091: per-file soft cap (~1000 logical lines).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOFT_CAP = 1000  # logical lines

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "stellar_jax"

# Paths that are always excluded from the scan (relative to SRC_ROOT).
_EXCLUDED_DIRS = {"data", "__pycache__"}

# ---------------------------------------------------------------------------
# Exemption list — pre-existing over-cap files with linked decomposition
# issues.  Each entry is (path-relative-to-SRC_ROOT, issue-URL-or-number).
# When a file is split below the cap, REMOVE its entry here so the lint
# catches future growth.
# ---------------------------------------------------------------------------
EXEMPTIONS: dict[str, str] = {
    "evolution/adaptive/forward.py": "#994 eps_nuc sourcing adds ~4 lines over cap",
}


# ---------------------------------------------------------------------------
# Core helpers (importable by the test suite)
# ---------------------------------------------------------------------------

def count_logical_lines(path: Path) -> int:
    """Return the number of non-blank, non-comment lines in *path*."""
    count = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                continue
            count += 1
    return count


def collect_py_files(root: Path) -> list[Path]:
    """Return all ``*.py`` files under *root*, excluding data/cache dirs."""
    results: list[Path] = []
    for child in sorted(root.rglob("*.py")):
        # Skip excluded directory trees
        parts = child.relative_to(root).parts
        if any(p in _EXCLUDED_DIRS for p in parts):
            continue
        results.append(child)
    return results


def scan(
    root: Path | None = None,
    cap: int = SOFT_CAP,
    exemptions: dict[str, str] | None = None,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int, str]]]:
    """Scan library source and partition over-cap files.

    Returns
    -------
    violations : list of (path, count)
        Files over the cap that are NOT in the exemption list.
    exempted : list of (path, count, issue)
        Files over the cap that ARE in the exemption list.
    """
    if root is None:
        root = SRC_ROOT
    if exemptions is None:
        exemptions = EXEMPTIONS

    violations: list[tuple[Path, int]] = []
    exempted: list[tuple[Path, int, str]] = []

    for py_file in collect_py_files(root):
        n = count_logical_lines(py_file)
        if n <= cap:
            continue
        rel = str(py_file.relative_to(root))
        issue = exemptions.get(rel)
        if issue is not None:
            exempted.append((py_file, n, issue))
        else:
            violations.append((py_file, n))

    return violations, exempted


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    show = "--show" in argv

    violations, exempted = scan()

    if show and exempted:
        print(f"Exempt ({len(exempted)} file(s), tracked for decomposition):")
        for path, n, issue in exempted:
            rel = path.relative_to(SRC_ROOT)
            print(f"  {rel}: {n} logical lines (decomposition: {issue})")
        print()

    if violations:
        print(
            f"FAIL: {len(violations)} file(s) exceed the ~{SOFT_CAP}-line "
            f"soft cap (logical lines = non-blank, non-comment):\n"
        )
        for path, n in violations:
            rel = path.relative_to(SRC_ROOT)
            print(f"  {rel}: {n} logical lines")
        print(
            f"\nEither decompose the file (preferred) or add it to the "
            f"exemption list in scripts/lint_file_size.py with a linked "
            f"decomposition issue."
        )
        sys.exit(1)

    exempt_note = f" ({len(exempted)} exempt)" if exempted else ""
    print(f"OK: all src/stellar_jax/ files <= {SOFT_CAP} logical lines{exempt_note}.")
    sys.exit(0)


if __name__ == "__main__":
    main()
