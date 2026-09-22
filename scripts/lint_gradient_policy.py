#!/usr/bin/env python3
"""CI lint: enforce GP-N policy annotation on every stop_gradient call.

Every jax.lax.stop_gradient (or lax.stop_gradient) call in production code
MUST be annotated with its gradient-policy tag: # GP-<N>: <POLICY_NAME>

The annotation can appear:
  1. On the same line as the stop_gradient call, OR
  2. In a comment within the 3 lines preceding the call.

Excluded from enforcement:
  - tests/ (test code may reference stop_gradient in assertions/mocks)
  - docs/ (documentation examples)
  - Lines that ARE comments or docstrings (not actual calls)
  - gradient_policy.py itself (the definition/examples)
  - **/contracts.py (policy documentation in contracts)

Usage:
    python scripts/lint_gradient_policy.py          # exit 0 if clean, 1 if violations
    python scripts/lint_gradient_policy.py --verbose # show all annotated calls too

Reference: docs/design/redesign/11-config-and-gradient-policy.md §3.3
"""

import os
import re
import sys
from pathlib import Path

# Regex matching a stop_gradient CALL (not just a mention in a string/comment)
_SG_CALL_RE = re.compile(r'\bstop_gradient\s*\(')

# Regex matching a valid GP-N annotation
_GP_TAG_RE = re.compile(r'#\s*GP-\d+')

# Directories/files excluded from enforcement
_EXCLUDED_PATHS = {
    'tests/',
    'docs/',
    'dev/', 'tools/', 'infra/', 'scripts/', 'paper/', 'bench/',
}

# File suffixes excluded
_EXCLUDED_SUFFIXES = (
    'contracts.py',
    'gradient_policy.py',
)


def _is_excluded(filepath: str) -> bool:
    """Check if a file is excluded from lint enforcement."""
    for exc in _EXCLUDED_PATHS:
        if filepath.startswith(exc) or filepath == exc:
            return True
    if filepath.endswith(_EXCLUDED_SUFFIXES):
        return True
    return False


def _is_comment_or_docstring(line: str) -> bool:
    """Check if a line is a comment or inside a docstring."""
    stripped = line.lstrip()
    if stripped.startswith('#'):
        return True
    if stripped.startswith('"""') or stripped.startswith("'''"):
        return True
    return False


def _has_gp_tag_in_context(lines: list, idx: int, window: int = 3) -> bool:
    """Check if a GP-N tag exists on the line or within `window` preceding lines."""
    # Check the line itself
    if _GP_TAG_RE.search(lines[idx]):
        return True
    # Check preceding lines (up to `window` lines back)
    for i in range(max(0, idx - window), idx):
        if _GP_TAG_RE.search(lines[i]):
            return True
    return False


def lint_file(filepath: str) -> list:
    """Lint a single file. Returns list of (lineno, line) violations."""
    violations = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return violations

    in_docstring = False
    docstring_char = None

    for idx, line in enumerate(lines):
        stripped = line.lstrip()

        # Track docstring state (simplified — handles triple-quote blocks)
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                # Single-line docstring
                if stripped.count(docstring_char) >= 2:
                    continue
                in_docstring = True
                continue
        else:
            if docstring_char and docstring_char in stripped:
                in_docstring = False
            continue

        # Skip comment-only lines
        if stripped.startswith('#'):
            continue

        # Check for stop_gradient call on this line
        if _SG_CALL_RE.search(line):
            if not _has_gp_tag_in_context(lines, idx):
                violations.append((idx + 1, line.rstrip()))

    return violations


def main():
    verbose = '--verbose' in sys.argv
    root = Path('.')

    # Find all .py files in the project (excluding .git, venv, etc.)
    py_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden dirs and common non-source dirs
        dirnames[:] = [d for d in dirnames if not d.startswith('.')
                       and d not in ('__pycache__', 'node_modules', '.git',
                                     'venv', '.venv', 'env')]
        for fname in filenames:
            if fname.endswith('.py'):
                filepath = os.path.join(dirpath, fname)
                # Normalize path
                filepath = os.path.relpath(filepath, root)
                py_files.append(filepath)

    total_calls = 0
    total_annotated = 0
    all_violations = []

    for filepath in sorted(py_files):
        if _is_excluded(filepath):
            continue

        violations = lint_file(filepath)
        if violations:
            all_violations.extend([(filepath, lineno, line)
                                   for lineno, line in violations])

        # Count for summary
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            calls_in_file = len(_SG_CALL_RE.findall(content))
            # Subtract calls that are in comments/docstrings (approximate)
            for line in content.splitlines():
                stripped = line.lstrip()
                if stripped.startswith('#') and _SG_CALL_RE.search(line):
                    calls_in_file -= 1
            total_calls += calls_in_file
            annotated = len(_GP_TAG_RE.findall(content))
            total_annotated += min(annotated, calls_in_file)
        except (OSError, UnicodeDecodeError):
            pass

    # Report
    if all_violations:
        print(f"\n❌ GRADIENT POLICY LINT FAILED: {len(all_violations)} "
              f"stop_gradient call(s) missing GP-N annotation\n")
        for filepath, lineno, line in all_violations:
            print(f"  {filepath}:{lineno}: {line}")
        print(f"\nEvery stop_gradient call must be annotated with its policy:")
        print(f"  # GP-<N>: <POLICY_NAME>")
        print(f"\nSee gradient_policy.py for the 12 policies (GP-1 through GP-12).")
        print(f"Ref: docs/design/redesign/11-config-and-gradient-policy.md\n")
        sys.exit(1)
    else:
        print(f"✓ Gradient policy lint passed: all stop_gradient calls annotated "
              f"({total_calls} calls in production code)")
        if verbose:
            print(f"  Total annotated: {total_annotated}")
        sys.exit(0)


if __name__ == '__main__':
    main()
