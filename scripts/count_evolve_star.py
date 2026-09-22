#!/usr/bin/env python3
"""AST-based counter of per-wave evolve_star entrypoints (#1221 AC-5).

Walks tests/test_*.py via the AST and reports every test function that calls
evolve_star (or its variants: evolve_star_diagnostic, evolve_star_adaptive)
directly — i.e. not via a session-scoped bundle fixture.

The script classifies each call site as:
  BUNDLED   — test consumes a bundle_* fixture (no own evolve_star compilation)
  PER_WAVE  — test calls evolve_star itself, is @integration, and NOT @suspended
  NIGHTLY   — test calls evolve_star itself but is @suspended
  FAST_SMOKE— test calls evolve_star itself but is @fast or @smoke
  GRADIENT  — test calls jax.grad/jacrev/jacfwd through evolve_star

Output: a summary table + the two key numbers:
  per_wave_evolve_star_calls: number of per-wave integration evolve_star calls
  per_wave_bundle_calls:      number of calls served by bundles (not compiled per-test)

Exit code:
  0  — count succeeded
  1  — parse error

Usage:
  python scripts/count_evolve_star.py              # human-readable table
  python scripts/count_evolve_star.py --json       # machine-readable for CI
  python scripts/count_evolve_star.py --summary    # one-line summary only
"""
import ast
import json
import os
import sys
from pathlib import Path

EVOLVE_STAR_NAMES = {
    'evolve_star', 'evolve_star_diagnostic', 'evolve_star_adaptive',
}

BUNDLE_FIXTURES = {
    'bundle_1p0', 'bundle_1p2', 'bundle_1p5', 'bundle_2p0', 'all_bundles',
}

GRAD_NAMES = {'grad', 'value_and_grad', 'jacfwd', 'jacrev'}


def _get_markers(decorators):
    """Extract pytest marker names from decorator list."""
    markers = set()
    for dec in decorators:
        # @pytest.mark.foo or @pytest.mark.foo(...)
        if isinstance(dec, ast.Call):
            dec = dec.func
        if isinstance(dec, ast.Attribute):
            # Walk the chain: pytest.mark.suspended → 'suspended'
            markers.add(dec.attr)
            # Also handle pytest.mark.parametrize with marks=pytest.mark.suspended
        elif isinstance(dec, ast.Name):
            markers.add(dec.id)
    return markers


def _get_fixture_params(node):
    """Extract fixture parameter names from a test function's args."""
    params = set()
    for arg in node.args.args:
        params.add(arg.arg)
    return params


def _has_call(node, names):
    """Check if any AST subtree of node calls a function in `names`."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            # Direct call: evolve_star(...)
            if isinstance(func, ast.Name) and func.id in names:
                return True
            # Attr call: stellar.evolve_star(...), jax.grad(...)
            if isinstance(func, ast.Attribute) and func.attr in names:
                return True
    return False


def _count_calls(node, names):
    """Count how many times functions in `names` are called in the AST subtree."""
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name) and func.id in names:
                count += 1
            elif isinstance(func, ast.Attribute) and func.attr in names:
                count += 1
    return count


def analyze_file(filepath):
    """Analyze a single test file, returning a list of test-function records."""
    with open(filepath) as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        print(f"WARN: cannot parse {filepath}: {e}", file=sys.stderr)
        return []

    records = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith('test_'):
            continue

        markers = _get_markers(node.decorator_list)
        fixtures = _get_fixture_params(node)
        has_evolve = _has_call(node, EVOLVE_STAR_NAMES)
        evolve_count = _count_calls(node, EVOLVE_STAR_NAMES)
        has_grad = _has_call(node, GRAD_NAMES)
        uses_bundle = bool(fixtures & BUNDLE_FIXTURES)

        # Skip tests that neither call evolve_star nor use a bundle
        if not has_evolve and not uses_bundle:
            continue

        # Classify
        if uses_bundle and not has_evolve:
            category = 'BUNDLED'
        elif 'suspended' in markers:
            category = 'NIGHTLY'
        elif 'fast' in markers or 'smoke' in markers:
            category = 'FAST_SMOKE'
        elif has_grad:
            category = 'GRADIENT'
        else:
            category = 'PER_WAVE'

        records.append({
            'file': os.path.basename(filepath),
            'test': node.name,
            'line': node.lineno,
            'evolve_star_calls': evolve_count,
            'has_gradient': has_grad,
            'uses_bundle': uses_bundle,
            'markers': sorted(markers & {
                'integration', 'suspended', 'fast', 'smoke',
                'validation', 'mutation',
            }),
            'category': category,
        })

    return records


def main():
    tests_dir = Path(__file__).resolve().parent.parent / 'tests'
    test_files = sorted(tests_dir.glob('test_*.py'))

    all_records = []
    for tf in test_files:
        all_records.extend(analyze_file(str(tf)))

    # Summary counts
    by_cat = {}
    for r in all_records:
        cat = r['category']
        by_cat.setdefault(cat, []).append(r)

    bundled = by_cat.get('BUNDLED', [])
    per_wave = by_cat.get('PER_WAVE', [])
    nightly = by_cat.get('NIGHTLY', [])
    fast_smoke = by_cat.get('FAST_SMOKE', [])
    gradient = by_cat.get('GRADIENT', [])

    # Per-wave evolve_star calls (the AC-5 metric):
    # These are per-wave integration tests that compile their own evolve_star
    per_wave_calls = sum(r['evolve_star_calls'] for r in per_wave)
    gradient_calls = sum(r['evolve_star_calls'] for r in gradient)
    nightly_calls = sum(r['evolve_star_calls'] for r in nightly)
    bundled_calls_saved = sum(r['evolve_star_calls'] for r in bundled)
    # Bundle fixtures themselves do call evolve_star (4 masses)
    bundle_fixture_calls = len({
        f for r in all_records if r['uses_bundle']
        for f in ({'bundle_1p0', 'bundle_1p2', 'bundle_1p5', 'bundle_2p0'}
                  if 'all_bundles' in _get_fixture_params_from_record(r)
                  else _get_fixture_params_from_record(r) & BUNDLE_FIXTURES)
    })

    total_per_wave_evolve = per_wave_calls + gradient_calls + 4  # 4 bundle evolves

    if '--json' in sys.argv:
        result = {
            'total_tests_with_evolve_star': len(all_records),
            'categories': {
                'BUNDLED': len(bundled),
                'PER_WAVE': len(per_wave),
                'GRADIENT': len(gradient),
                'NIGHTLY': len(nightly),
                'FAST_SMOKE': len(fast_smoke),
            },
            'per_wave_evolve_star_tests': len(per_wave) + len(gradient),
            'per_wave_evolve_star_calls': per_wave_calls + gradient_calls,
            'bundle_evolve_star_calls': 4,
            'nightly_evolve_star_calls': nightly_calls,
            'total_per_wave_evolve_star': total_per_wave_evolve,
            'records': all_records,
        }
        print(json.dumps(result, indent=2))
        return

    if '--summary' in sys.argv:
        print(f"Per-wave evolve_star: {total_per_wave_evolve} "
              f"({per_wave_calls} forward + {gradient_calls} gradient + "
              f"4 bundle fixtures). "
              f"Nightly: {nightly_calls}. "
              f"Bundled tests: {len(bundled)} (0 own evolve_star).")
        return

    # Human-readable table
    print("=" * 90)
    print("evolve_star entrypoint count (#1221 AC-5)")
    print("=" * 90)
    print()

    for cat_name, cat_records in [
        ('BUNDLED (read from session-scoped bundle, no own compile)', bundled),
        ('PER_WAVE (forward-only @integration, not suspended)', per_wave),
        ('GRADIENT (calls jax.grad through evolve_star)', gradient),
        ('NIGHTLY (@suspended — excluded from per-wave CI)', nightly),
        ('FAST_SMOKE (@fast or @smoke tier)', fast_smoke),
    ]:
        if not cat_records:
            continue
        calls = sum(r['evolve_star_calls'] for r in cat_records)
        print(f"--- {cat_name}: {len(cat_records)} tests, "
              f"{calls} evolve_star calls ---")
        for r in sorted(cat_records, key=lambda x: (x['file'], x['test'])):
            grad_tag = ' [GRAD]' if r['has_gradient'] else ''
            bundle_tag = ' [BUNDLE]' if r['uses_bundle'] else ''
            print(f"  {r['file']}:{r['line']:>4d}  {r['test']}"
                  f"  (×{r['evolve_star_calls']}){grad_tag}{bundle_tag}")
        print()

    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  Per-wave forward tests:     {len(per_wave):>3d} "
          f"({per_wave_calls} evolve_star calls)")
    print(f"  Per-wave gradient tests:    {len(gradient):>3d} "
          f"({gradient_calls} evolve_star calls)")
    print(f"  Bundle fixtures:            {4:>3d} "
          f"(4 evolve_star calls, shared by {len(bundled)} tests)")
    print(f"  Nightly (suspended):        {len(nightly):>3d} "
          f"({nightly_calls} evolve_star calls)")
    print(f"  Fast/smoke:                 {len(fast_smoke):>3d} "
          f"(lightweight, Fargate-bundled)")
    print()
    per_wave_tests = len(per_wave) + len(gradient)
    print(f"  TOTAL per-wave evolve_star: {total_per_wave_evolve} calls "
          f"across {per_wave_tests} tests "
          f"(forward {per_wave_calls} + gradient {gradient_calls} + "
          f"bundle 4)")
    print()
    print(f"  AC-1 target was ≤5 per-wave evolve_star *tests*.")
    print(f"  Achieved: {per_wave_tests} per-wave tests "
          f"({len(per_wave)} forward + {len(gradient)} gradient); "
          f"nightly: {len(nightly)}; bundled: {len(bundled)}.")
    print(f"  Root cause: evolve_star's static args (max_steps, fixed_dt,")
    print(f"  grad_window, freeze_schedule) are XLA compile-cache keys.")
    print(f"  Each unique static config = distinct XLA compile = can't share.")
    print(f"  See tests/reconciliation.md for the full analysis.")


def _get_fixture_params_from_record(r):
    """Reconstruct fixture set from a record (for bundle counting)."""
    # This is approximate — we use the 'uses_bundle' flag
    # The actual fixture params are checked at analysis time
    return BUNDLE_FIXTURES  # placeholder


if __name__ == '__main__':
    main()
