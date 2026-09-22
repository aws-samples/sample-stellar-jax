#!/usr/bin/env python3
"""Lint guard: bare carry[N] indices, function length, (advisory) arg counts.

Run as part of preflight / CI to prevent reintroduction of magic-index carry
accesses and over-length functions.

Usage:
    python scripts/lint_signatures.py          # exits 0 if clean, 1 if violations

Checks:
  1. Bare carry[N] / final_carry[N] magic indices (hard fail).
  2. Arg counts > 5 (ADVISORY — warn, never fail; issue #1117 dropped enforcement).
  3. Function body > 100 logical lines (SOFT cap — §11-G):
     - Pre-existing violations in BODY_LENGTH_EXEMPT are warned, not failed.
     - Any ``# lint: ... exception — <reason>`` marker on or near the def
       suppresses a function (scan-body, jit-boundary, limiter-chain, etc.).
     - A NEW over-cap function not in the exemption list => hard fail.
     - Logical lines exclude blank lines, ``#``-comments, AND lines belonging
       to docstrings / standalone string-literal ``Expr`` nodes (AST-based).

Design standard §2 (no god-signatures), §4 (≤3-4 args), §11-G (function length).
"""
import ast
import os
import re
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / 'src' / 'stellar_jax'

# Files to scan for bare carry indices
CARRY_INDEX_FILES = [
    SRC_ROOT / 'evolution' / '_core.py',
    SRC_ROOT / 'calibration' / 'solar.py',
    SRC_ROOT / 'calibration' / 'diagnostic.py',
]

# Pattern matching carry[N] or final_carry[N] with bare integer indices
CARRY_MAGIC_PATTERN = re.compile(r'(?:carry|final_carry)\s*\[\s*\d+\s*\]')

# Functions exempt from the arg-count limit (PUBLIC APIs / special cases only)
EXEMPT_FUNCTIONS = {
    # Public entry points — keep kwargs for backward-compat
    'evolve_star',
    'evolve_star_adaptive',
    'evolve_star_comp',
    'evolve_star_diagnostic',
    'evolve_solar',
    'observable_at_target',
    'zams_properties',
    # JIT boundary functions — MUST keep positional-arg signatures for XLA
    # bit-identity. Dict-based args change the JAX pytree structure, producing
    # a different XLA HLO graph that accumulates FP differences over 400+ RGB
    # steps (measured: 0.287 dex shift, from logL=3.237 to 2.950). The Python
    # helpers that CALL these JIT functions do the dict→positional unpacking.
    '_evolve_star_jit',
    '_jit_henyey_step',
    '_jit_extract_shell_data',
    '_jit_composition_update',
    # _fourier_chebyshev (neutrino.py) — 36 Fourier-Chebyshev coefficients
    # passed as individual scalars inside the JIT-traced Henyey Newton solver.
    # Packing into a (3,12) jnp.array changes the XLA HLO graph (array
    # construction + indexing vs direct scalar ops), causing FP accumulation
    # drift over 400+ RGB steps. Same root cause as the JIT boundary issue.
    '_fourier_chebyshev',
    # Calibration utilities (separate scope, own scan loops)
    'solar_calibrate',
    'solar_calibrate_henyey',
    'solar_calibrate_with_zprofile',
    'solar_residual',
    'solar_residual_henyey',
    'solar_residual_with_zprofile',
    '_newton_raphson_2d',
    # Comparison/diagnostic utilities (numpy, not hot-path)
    'compare_mesa',
    'compare_model_s',
    'kippenhahn_diagram',
    'plot_kippenhahn',
    'read_mesa_history',
    # Constructors
    '__init__',
}

# Files to scan for arg counts
ARG_COUNT_FILES = [
    SRC_ROOT / 'evolution' / '_core.py',
    SRC_ROOT / 'evolution' / 'adaptive' / 'forward.py',
    SRC_ROOT / 'microphysics' / 'neutrino.py',
]

MAX_ARGS = 5

# ---------------------------------------------------------------------------
# Check 3: function body length — SOFT cap (§11-G)
# ---------------------------------------------------------------------------

MAX_BODY_LINES = 100

# Marker that exempts a single function from the body-length cap.
# Matches any ``# lint: <label> exception`` variant (scan-body, jit-boundary,
# limiter-chain, sequential-setup, etc.).  "exception" must appear as a
# standalone word (not embedded in other text like "missing exception keyword").
# Must appear on the def line or within 3 preceding lines.
EXCEPTION_MARKER_RE = re.compile(r'#\s*lint:\s*\S+(?:-\S+)*\s+exception\b')

# Pre-existing over-cap functions — documented exemption referencing the
# decompose issue.  Each entry is (relative_path_from_repo_root, function_name).
# These are WARNED (printed) but do not cause exit(1).
BODY_LENGTH_EXEMPT: set[tuple[str, str]] = set()


def _count_logical_lines(source_lines, start_line, end_line, body_nodes=None):
    """Count non-blank, non-comment, non-docstring lines in [start_line, end_line] (1-indexed).

    When *body_nodes* (the ``node.body`` list from ast.parse) is provided,
    lines belonging to standalone string-literal ``Expr`` statements (i.e.
    docstrings and bare string constants) are excluded.  This prevents
    well-documented functions from being penalised for their docstrings.
    """
    # Build set of line numbers to exclude (docstring / string-literal Expr).
    exclude_lines: set[int] = set()
    if body_nodes is not None:
        for stmt in body_nodes:
            if (isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)):
                stmt_end = getattr(stmt, 'end_lineno', stmt.lineno) or stmt.lineno
                for ln in range(stmt.lineno, stmt_end + 1):
                    exclude_lines.add(ln)

    count = 0
    for i in range(start_line - 1, min(end_line, len(source_lines))):
        lineno = i + 1  # 1-indexed
        if lineno in exclude_lines:
            continue
        stripped = source_lines[i].strip()
        if stripped == '' or stripped.startswith('#'):
            continue
        count += 1
    return count


def _has_exception_marker(source_lines, def_lineno):
    """Check for a ``# lint: ... exception`` marker on/before a function def (1-indexed)."""
    # Check the def line itself and up to 3 lines preceding it.
    for idx in range(max(0, def_lineno - 4), def_lineno):
        if idx < len(source_lines) and EXCEPTION_MARKER_RE.search(source_lines[idx]):
            return True
    return False


def check_function_body_length(src_root=None):
    """Check function body length across all .py files under src_root.

    Returns:
        (new_violations, exempt_warnings): two lists.
        - new_violations: functions over the cap that are NOT exempted/excepted.
          Each: (rel_path, lineno, func_name, logical_line_count).
        - exempt_warnings: functions over the cap that ARE in the exemption list.
          Same tuple format.
    """
    if src_root is None:
        src_root = SRC_ROOT
    repo_root = src_root.parent.parent

    new_violations = []
    exempt_warnings = []

    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        for fname in sorted(filenames):
            if not fname.endswith('.py'):
                continue
            fpath = Path(dirpath) / fname
            try:
                with open(fpath) as f:
                    source = f.read()
                source_lines = source.splitlines()
                tree = ast.parse(source, filename=str(fpath))
            except (SyntaxError, OSError):
                continue

            rel_path = str(fpath.relative_to(repo_root))

            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not node.body:
                    continue

                body_start = node.body[0].lineno
                body_end = getattr(node, 'end_lineno', None) or body_start
                logical = _count_logical_lines(source_lines, body_start, body_end, node.body)

                if logical <= MAX_BODY_LINES:
                    continue

                # Check for exception marker (scan-body / fused kernel)
                if _has_exception_marker(source_lines, node.lineno):
                    continue

                entry = (rel_path, node.lineno, node.name, logical)

                if (rel_path, node.name) in BODY_LENGTH_EXEMPT:
                    exempt_warnings.append(entry)
                else:
                    new_violations.append(entry)

    return new_violations, exempt_warnings


def check_carry_magic_indices():
    """Check for bare carry[N]/final_carry[N] in target files."""
    violations = []
    for path in CARRY_INDEX_FILES:
        if not path.exists():
            continue
        with open(path) as f:
            for i, line in enumerate(f, 1):
                stripped = line.lstrip()
                if stripped.startswith('#'):
                    continue
                if CARRY_MAGIC_PATTERN.search(line):
                    violations.append((path.relative_to(SRC_ROOT.parent.parent), i, line.rstrip()))
    return violations


def check_arg_counts():
    """Check that no function exceeds MAX_ARGS (config objects count as 1)."""
    violations = []
    for path in ARG_COUNT_FILES:
        if not path.exists():
            continue
        with open(path) as f:
            source = f.read()
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = node.name
            if name in EXEMPT_FUNCTIONS:
                continue
            # Count args (positional + keyword-only), excluding 'self'/'cls'
            args = node.args
            all_args = list(args.args) + list(args.kwonlyargs)
            arg_names = [a.arg for a in all_args if a.arg not in ('self', 'cls')]
            n_args = len(arg_names)
            if n_args > MAX_ARGS:
                rel_path = path.relative_to(SRC_ROOT.parent.parent)
                violations.append((rel_path, node.lineno, name, n_args, arg_names))
    return violations


def main():
    rc = 0

    # Check 1: carry magic indices
    carry_violations = check_carry_magic_indices()
    if carry_violations:
        print(f"FAIL: {len(carry_violations)} bare carry[N]/final_carry[N] magic index(es):")
        for path, lineno, text in carry_violations:
            print(f"  {path}:{lineno}: {text}")
        print("\nReplace with named constants from evolution/contracts.py.")
        rc = 1
    else:
        print("OK: 0 bare carry magic indices.")

    # Check 2: arg counts (ADVISORY — warn, never fail; dropped enforcement)
    arg_violations = check_arg_counts()
    if arg_violations:
        print(f"\nINFO: {len(arg_violations)} function(s) exceed {MAX_ARGS} args (advisory, not enforced):")
        for path, lineno, name, n_args, arg_names in arg_violations:
            print(f"  {path}:{lineno}: {name}() has {n_args} args")
    else:
        print(f"OK: all scanned functions <= {MAX_ARGS} args")

    # Check 3: function body length (SOFT cap)
    new_violations, exempt_warnings = check_function_body_length()
    if exempt_warnings:
        print(f"\nWARN: {len(exempt_warnings)} pre-existing function(s) exceed "
              f"{MAX_BODY_LINES} logical lines (exempted, tracked for decomposition):")
        for rel_path, lineno, name, logical in sorted(exempt_warnings, key=lambda x: -x[3]):
            print(f"  {rel_path}:{lineno}: {name}() — {logical} logical lines")
    if new_violations:
        print(f"\nFAIL: {len(new_violations)} NEW function(s) exceed "
              f"{MAX_BODY_LINES} logical lines (not exempted):")
        for rel_path, lineno, name, logical in sorted(new_violations, key=lambda x: -x[3]):
            print(f"  {rel_path}:{lineno}: {name}() — {logical} logical lines")
        print(f"\nRefactor to <={MAX_BODY_LINES} logical lines, or add "
              f"'# lint: <label> exception — <reason>' for a justified case.")
        rc = 1
    elif not exempt_warnings:
        print(f"\nOK: all functions <= {MAX_BODY_LINES} logical lines.")
    else:
        print(f"  (No new violations — soft cap is green.)")

    sys.exit(rc)


if __name__ == '__main__':
    main()
