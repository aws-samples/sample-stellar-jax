#!/usr/bin/env python3
"""Lint guard: no inline tracker refs (#NNN/GP-N/OSC-N) in evolution/_core.py.

Run as part of preflight / CI to prevent reintroduction of internal issue
numbers and gradient-policy codes in shipped production source.

Usage:
    python scripts/lint_tracker_refs.py          # exits 0 if clean, 1 if violations
    python scripts/lint_tracker_refs.py --fix    # (future) auto-strip

Design standard §9: "No internal-tracker refs in shipped source — keep #NNN
issue numbers and GP-/O2 codes out of physics modules."
"""
import re
import sys
from pathlib import Path

PATTERN = re.compile(r'#[0-9]{3,}|OSC-[0-9]')
# GP-N tags are EXCLUDED — they are required by the gradient-policy lint
# (lint_gradient_policy.py). Only issue numbers (#NNN) and OSC-N codes
# are considered tracker refs.
GP_PATTERN = re.compile(r'GP-[0-9]+')
TARGET = Path(__file__).resolve().parent.parent / 'src' / 'stellar_jax' / 'evolution' / '_core.py'


def lint(path: Path) -> list[tuple[int, str]]:
    """Return (line_number, line_text) for each violation."""
    violations = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            # Check for #NNN or OSC-N
            if PATTERN.search(line):
                violations.append((i, line.rstrip()))
            # Check for GP-N that is NOT on a stop_gradient annotation line
            elif GP_PATTERN.search(line) and 'stop_gradient' not in line:
                violations.append((i, line.rstrip()))
    return violations


def main():
    if not TARGET.exists():
        print(f"SKIP: {TARGET} not found")
        sys.exit(0)
    violations = lint(TARGET)
    if violations:
        print(f"FAIL: {len(violations)} inline tracker ref(s) in {TARGET}:")
        for lineno, text in violations:
            print(f"  L{lineno}: {text}")
        print("\nStrip #NNN/GP-N/OSC-N refs before committing.")
        sys.exit(1)
    print(f"OK: {TARGET} has 0 inline tracker refs.")
    sys.exit(0)


if __name__ == '__main__':
    main()
