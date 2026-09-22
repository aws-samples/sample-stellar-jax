"""Self-tests for issue #824 machinery: mutation lint, right-reason check, partial-mutation tier.

These tests verify the new validation/audit infrastructure itself:
  - lint_mutations.py correctly classifies mutations
  - The right-reason conftest hook fires under mutation
  - Partial mutations are registered and callable

Runs as part of the @fast bundle (no evolve_star, no JIT).
"""
import ast
import importlib.util
import os
import sys

import pytest

# ═══════════════════════════════════════════════════════════════
# Test 1: Mutation registry lint
# ═══════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.smoke
def test_lint_mutations_no_pass_body():
    """The mutation lint must find zero pass-body mutations (AC1 gate).

    A pass-body mutation has its physics break in the TEST, not the
    centrally-audited registry. The lint (tests/lint_mutations.py)
    must report zero such mutations after the #824 fix.
    """
    # Import the lint module
    lint_path = os.path.join(os.path.dirname(__file__), "lint_mutations.py")
    spec = importlib.util.spec_from_file_location("lint_mutations", lint_path)
    lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lint)

    results = lint.audit_mutations()
    assert len(results['pass_body']) == 0, (
        f"Found {len(results['pass_body'])} pass-body mutations — "
        f"each must move its break into the registry function body: "
        f"{[name for name, _ in results['pass_body']]}")
    assert len(results['no_effect']) == 0, (
        f"Found {len(results['no_effect'])} no-effect mutations — "
        f"each must call mp.setattr or mp.setenv: "
        f"{[name for name, _ in results['no_effect']]}")


@pytest.mark.fast
@pytest.mark.smoke
def test_lint_mutations_counts():
    """The mutation lint reports consistent totals."""
    lint_path = os.path.join(os.path.dirname(__file__), "lint_mutations.py")
    spec = importlib.util.spec_from_file_location("lint_mutations", lint_path)
    lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lint)

    results = lint.audit_mutations()
    total_classified = (len(results['pass_body']) + len(results['env_var_only'])
                        + len(results['no_effect']) + len(results['env_and_patch'])
                        + len(results['proper_patch']))
    assert total_classified == results['total'], (
        f"Classification mismatch: {total_classified} classified vs "
        f"{results['total']} total")
    # There should be at least 90 mutations (known count: 96 after)
    assert results['total'] >= 90, (
        f"Only {results['total']} mutations found — expected ≥90")


# ═══════════════════════════════════════════════════════════════
# Test 2: Partial-mutation tier registration
# ═══════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.smoke
def test_partial_mutations_registered():
    """All three partial mutations from #824 AC3 are registered and callable."""
    mutations_path = os.path.join(os.path.dirname(__file__), "mutations.py")
    spec = importlib.util.spec_from_file_location("mutations_check", mutations_path)
    muts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(muts)

    expected = [
        "partial_detach_z_gradient",
        "partial_corrupt_gamma1",
        "partial_eps_nuc_scale",
    ]
    for name in expected:
        assert name in muts.MUTATIONS, (
            f"Partial mutation '{name}' not found in registry")
        # Verify it's callable
        assert callable(muts.MUTATIONS[name]), (
            f"Partial mutation '{name}' is not callable")


@pytest.mark.fast
@pytest.mark.smoke
def test_partial_mutations_wired_to_tests():
    """Each partial mutation is declared on at least one test function.

    Issue #824 AC3: partial mutations must be exercised by O2, not just
    registered. This verifies that each partial mutation name appears
    as a @mutation("partial_*") decorator on at least one test function.
    """
    test_dir = os.path.dirname(__file__)
    partial_names = [
        "partial_detach_z_gradient",
        "partial_corrupt_gamma1",
        "partial_eps_nuc_scale",
    ]
    found = {name: [] for name in partial_names}

    for fname in sorted(os.listdir(test_dir)):
        if not fname.startswith('test_') or not fname.endswith('.py'):
            continue
        if fname == 'test_validation_audit.py':
            continue  # Skip this file (self-reference)
        fpath = os.path.join(test_dir, fname)
        with open(fpath) as f:
            source = f.read()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith('test_'):
                continue
            for deco in node.decorator_list:
                if (isinstance(deco, ast.Call)
                        and isinstance(deco.func, ast.Attribute)
                        and deco.func.attr == 'mutation'
                        and deco.args
                        and isinstance(deco.args[0], ast.Constant)):
                    mut_name = deco.args[0].value
                    if mut_name in found:
                        found[mut_name].append(f"{fname}:{node.name}")

    missing = [name for name, tests in found.items() if not tests]
    assert not missing, (
        f"Partial mutations NOT wired to any test: {missing}. "
        f"Each must appear as @mutation('name') on at least one "
        f"@validation test so O2 exercises it."
    )
def test_detach_rgb_mass_flag_default():
    """The DETACH_RGB_MASS_IN_DNU_ACTIVE flag defaults to False."""
    mutations_path = os.path.join(os.path.dirname(__file__), "mutations.py")
    spec = importlib.util.spec_from_file_location("mutations_flag", mutations_path)
    muts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(muts)

    assert hasattr(muts, 'DETACH_RGB_MASS_IN_DNU_ACTIVE'), (
        "Module flag DETACH_RGB_MASS_IN_DNU_ACTIVE missing from mutations.py")
    assert muts.DETACH_RGB_MASS_IN_DNU_ACTIVE is False, (
        "DETACH_RGB_MASS_IN_DNU_ACTIVE should default to False")


# ═══════════════════════════════════════════════════════════════
# Test 3: Right-reason marker registration
# ═══════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.smoke
def test_right_reason_marker_registered():
    """The right_reason marker is registered in conftest.py."""
    conftest_path = os.path.join(os.path.dirname(__file__), "conftest.py")
    with open(conftest_path) as f:
        content = f.read()
    assert "right_reason" in content, (
        "right_reason marker not found in conftest.py — "
        "the O2 right-reason check infrastructure is missing")
    # Verify the marker is used in at least one test file
    import glob as glob_mod
    test_files = glob_mod.glob(os.path.join(os.path.dirname(__file__), "test_*.py"))
    found = False
    for tf in test_files:
        with open(tf) as f:
            if "right_reason" in f.read():
                found = True
                break
    assert found, (
        "No test file uses @right_reason — at least the exemplar tests "
        "should have it (test_gradient_policy.py, test_seismic_structure.py, "
        "test_microphysics.py)")


# ═══════════════════════════════════════════════════════════════
# Test 4: (removed in the repo split) — validated the acceptance-criteria
# completeness procedure, maintained with the project tooling.
# The test moved with the tooling.
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# Test 5: Right-reason coverage (AC2 completeness gate)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.smoke
def test_right_reason_full_coverage():
    """Every @validation + @mutation test must have @right_reason (AC2).

    The right_reason marker tags the PHYSICS assertion that should break
    under mutation. Without it, a mutation could break an incidental guard
    (e.g. len(modes) >= N) and O2 wouldn't know. This test enforces 100%
    coverage of validation+mutation tests.

    Issue #824 AC2: "tag each @validation test's designated PHYSICS assertion."
    """
    test_dir = os.path.dirname(__file__)
    missing = []

    for fname in sorted(os.listdir(test_dir)):
        if not fname.startswith('test_') or not fname.endswith('.py'):
            continue
        fpath = os.path.join(test_dir, fname)
        with open(fpath) as f:
            source = f.read()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith('test_'):
                continue

            has_validation = False
            has_right_reason = False
            has_mutation = False

            for deco in node.decorator_list:
                deco_str = ast.dump(deco)
                if ('validation' in deco_str and 'mark' in deco_str
                        and 'mutation' not in deco_str
                        and 'right_reason' not in deco_str):
                    has_validation = True
                if 'right_reason' in deco_str:
                    has_right_reason = True
                if 'mutation' in deco_str and isinstance(deco, ast.Call):
                    has_mutation = True

            if has_validation and has_mutation and not has_right_reason:
                missing.append(f"{fname}:{node.lineno} {node.name}")

    assert len(missing) == 0, (
        f"Found {len(missing)} @validation + @mutation tests WITHOUT "
        f"@right_reason — each must tag its physics assertion. "
        f"Missing:\n  " + "\n  ".join(missing[:20])
        + (f"\n  ... and {len(missing)-20} more" if len(missing) > 20 else ""))


@pytest.mark.fast
@pytest.mark.smoke
def test_right_reason_patterns_nonempty():
    """Every @right_reason marker must have a non-empty pattern string.

    An empty pattern defeats the right-reason check (any failure message
    matches an empty string).
    """
    test_dir = os.path.dirname(__file__)
    empty_patterns = []

    for fname in sorted(os.listdir(test_dir)):
        if not fname.startswith('test_') or not fname.endswith('.py'):
            continue
        fpath = os.path.join(test_dir, fname)
        with open(fpath) as f:
            source = f.read()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith('test_'):
                continue

            for deco in node.decorator_list:
                deco_str = ast.dump(deco)
                if 'right_reason' in deco_str and isinstance(deco, ast.Call):
                    if deco.args:
                        try:
                            pattern = ast.literal_eval(deco.args[0])
                            if not pattern or not pattern.strip():
                                empty_patterns.append(
                                    f"{fname}:{node.lineno} {node.name}")
                        except Exception:
                            pass
                    else:
                        empty_patterns.append(
                            f"{fname}:{node.lineno} {node.name}")

    assert len(empty_patterns) == 0, (
        f"Found {len(empty_patterns)} @right_reason markers with EMPTY "
        f"patterns — each must contain a meaningful substring:\n  "
        + "\n  ".join(empty_patterns))


# ═══════════════════════════════════════════════════════════════
# Test 6: No dangling file-path references in docstrings/contracts
# ═══════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.smoke
def test_no_dangling_location_refs():
    """Docstring/contract file-path references must point at files that exist.

    WHAT: scans all .py files under src/stellar_jax/ for inline references to
    module paths (e.g. "evolution/_core.py", "mesh/structure_mesh.py") and
    verifies each referenced .py file exists on disk.

    WHY: the #504/#525 decomposition created then removed evolution/step.py,
    but the gradient_policy.py + contracts kept citing it. This lint prevents
    that class of stale reference from recurring. (Issue #1092.)

    PATTERN: matches `subpackage/filename.py` references in comments and
    docstrings. Excludes test paths, URLs, import statements, and
    non-package references (e.g. MESA source paths like `evolve.f90`).
    """
    import re
    from pathlib import Path

    src_root = Path(__file__).resolve().parent.parent / 'src' / 'stellar_jax'
    assert src_root.is_dir(), f"Package root not found: {src_root}"

    # Match patterns like "evolution/_core.py", "mesh/structure_mesh.py",
    # "composition/contracts.py" — a subpackage slash filename.py reference.
    # Anchored to known subpackages to avoid false positives on MESA refs.
    _KNOWN_SUBPKGS = (
        'evolution', 'mesh', 'transport', 'composition', 'fgong',
        'oscillations', 'solver', 'microphysics', 'config', 'calibration',
    )
    _subpkg_alt = '|'.join(_KNOWN_SUBPKGS)
    # Captures: (subpackage/filename.py) — allows _underscores and nested /
    _REF_RE = re.compile(
        rf'\b((?:{_subpkg_alt})/[a-z_][a-z0-9_]*\.py)\b'
    )

    dangling: list[tuple[str, int, str, str]] = []  # (file, lineno, ref, line)

    for py_file in sorted(src_root.rglob('*.py')):
        rel = py_file.relative_to(src_root)
        # Skip __pycache__ and data/
        if '__pycache__' in str(rel) or str(rel).startswith('data'):
            continue
        try:
            lines = py_file.read_text(encoding='utf-8').splitlines()
        except (OSError, UnicodeDecodeError):
            continue

        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            # Only check comments and docstring lines — skip import statements
            # and actual code (which may construct paths dynamically).
            if stripped.startswith(('import ', 'from ')):
                continue
            for m in _REF_RE.finditer(line):
                ref_path = m.group(1)
                # Check if the referenced file exists under src/stellar_jax/
                if not (src_root / ref_path).is_file():
                    dangling.append((str(rel), i, ref_path, line.rstrip()))

    assert len(dangling) == 0, (
        f"Found {len(dangling)} dangling file-path reference(s) in "
        f"src/stellar_jax/ docstrings/comments — each must point at a "
        f"file that exists:\n"
        + "\n".join(
            f"  {f}:{ln}: references '{ref}' (does not exist)\n"
            f"    {text}"
            for f, ln, ref, text in dangling
        )
    )


# ═══════════════════════════════════════════════════════════════
# Test-count reconciliation (AC-5)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.fast
@pytest.mark.smoke
def test_bundle_tests_manifest_matches_fixtures():
    """ci_bundle_tests.txt is consistent with bundle fixture usage.

    WHAT: every test listed in ci_bundle_tests.txt must request a bundle_*
    fixture in its signature, and every test that requests a bundle_* fixture
    must be listed in ci_bundle_tests.txt.

    WHY: the CI orchestrator routes bundled tests into per-mass tasks using
    this manifest. A mismatch means a test either runs outside its bundle
    (compiling its own evolve_star — defeating consolidation) or is listed
    but doesn't use the fixture (dead entry).
    """
    import ast as _ast
    from pathlib import Path

    tests_dir = Path(__file__).resolve().parent
    manifest = tests_dir / 'ci_bundle_tests.txt'
    assert manifest.is_file(), "ci_bundle_tests.txt missing"

    # Parse manifest: non-comment, non-blank lines
    manifest_tests = set()
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            manifest_tests.add(line)

    # Find all test functions that request bundle fixtures
    _BUNDLE_FIXTURES = {
        'bundle_1p0', 'bundle_1p2', 'bundle_1p5', 'bundle_2p0', 'all_bundles',
    }
    fixture_tests = set()
    for tf in sorted(tests_dir.glob('test_*.py')):
        source = tf.read_text()
        try:
            tree = _ast.parse(source)
        except SyntaxError:
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith('test_'):
                continue
            params = {arg.arg for arg in node.args.args}
            if params & _BUNDLE_FIXTURES:
                fixture_tests.add(node.name)

    # Check both directions
    in_manifest_not_fixture = manifest_tests - fixture_tests
    in_fixture_not_manifest = fixture_tests - manifest_tests

    errors = []
    if in_manifest_not_fixture:
        errors.append(
            f"In ci_bundle_tests.txt but no bundle fixture in signature: "
            f"{sorted(in_manifest_not_fixture)}")
    if in_fixture_not_manifest:
        errors.append(
            f"Has bundle fixture but not in ci_bundle_tests.txt: "
            f"{sorted(in_fixture_not_manifest)}")

    assert not errors, "\n".join(errors)


@pytest.mark.fast
@pytest.mark.smoke
def test_suspended_tests_excluded_from_validation_manifest():
    """Fully-suspended @validation tests must not appear in ci_validation_tests.txt.

    WHAT: every test in ci_validation_tests.txt must have at least one
    non-suspended parametrization (or be non-parametrized and not @suspended).

    WHY: the O2 mutation gate fans out one audit task per line in this
    manifest. If a FULLY-suspended test is listed, the gate will try to
    run it and fail. A partially-suspended parametrized test (e.g.
    test_foo[1p5] per-wave, test_foo[2p0] suspended) is fine — the
    non-suspended variant still runs per-wave.
    """
    from pathlib import Path

    tests_dir = Path(__file__).resolve().parent
    manifest = tests_dir / 'ci_validation_tests.txt'
    assert manifest.is_file(), "ci_validation_tests.txt missing"

    # Parse manifest: non-comment, non-blank lines
    manifest_tests = set()
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            manifest_tests.add(line)

    # Collect FULLY-suspended tests: those where ALL parametrizations are
    # @suspended (no non-suspended variant exists to run per-wave).
    import subprocess

    # Get all @suspended test nodeids
    result_susp = subprocess.run(
        [sys.executable, '-m', 'pytest', str(tests_dir),
         '--collect-only', '-q', '-m', 'suspended'],
        capture_output=True, text=True, timeout=30)
    suspended_nodeids = set()
    for line in result_susp.stdout.splitlines():
        if '::' in line:
            suspended_nodeids.add(line.strip())

    # Get all test nodeids (no filter)
    result_all = subprocess.run(
        [sys.executable, '-m', 'pytest', str(tests_dir),
         '--collect-only', '-q'],
        capture_output=True, text=True, timeout=60)
    all_nodeids = set()
    for line in result_all.stdout.splitlines():
        if '::' in line:
            all_nodeids.add(line.strip())

    # Group by base test name (strip parametrize brackets)
    def _base_name(nodeid):
        name = nodeid.split('::')[-1]
        if '[' in name:
            name = name[:name.index('[')]
        return name

    # A test is fully-suspended if ALL its nodeids are in suspended_nodeids
    from collections import defaultdict
    base_to_nodeids = defaultdict(set)
    for nid in all_nodeids:
        base_to_nodeids[_base_name(nid)].add(nid)

    fully_suspended = set()
    for base, nids in base_to_nodeids.items():
        if nids and nids <= suspended_nodeids:
            fully_suspended.add(base)

    overlap = manifest_tests & fully_suspended
    assert not overlap, (
        f"Fully-suspended tests in ci_validation_tests.txt (should be "
        f"excluded): {sorted(overlap)}")


@pytest.mark.fast
@pytest.mark.smoke
def test_evolve_star_entrypoint_count_script_runs():
    """The evolve_star entrypoint counting script (AC-5) runs without error.

    WHAT: scripts/count_evolve_star.py produces valid JSON output.

    WHY: this script is the AC-5 verification tool — it must be kept
    working so the test-count reconciliation stays machine-readable.
    """
    import subprocess
    import json as _json
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / 'scripts' / 'count_evolve_star.py'
    assert script.is_file(), f"count_evolve_star.py missing at {script}"

    result = subprocess.run(
        [sys.executable, str(script), '--json'],
        capture_output=True, text=True, timeout=30)

    assert result.returncode == 0, (
        f"count_evolve_star.py failed:\nstdout: {result.stdout[:500]}\n"
        f"stderr: {result.stderr[:500]}")

    data = _json.loads(result.stdout)
    assert 'total_tests_with_evolve_star' in data
    assert 'categories' in data
    assert data['categories']['BUNDLED'] >= 2, (
        f"Expected ≥2 bundled tests, got {data['categories']['BUNDLED']}")


@pytest.mark.fast
@pytest.mark.smoke
def test_suspended_tests_not_in_ci_routing_manifests():
    """Suspended tests must not appear in per-wave CI routing manifests.

    WHAT: Checks that no @suspended test name appears in ci_heavy_tests.txt,
    ci_huge_tests.txt, ci_mega_tests.txt, or ci_long_tests.txt.

    WHY: The CI orchestrator launches one task per manifest entry. A suspended
    test in a routing manifest wastes a CI task slot — the task starts, pytest
    collects the test, the -m filter skips it, and the slot is wasted. This
    was found during #1221: 132 wasted task entries across 4 manifests.

    WHAT MAKES IT FAIL: Adding a test to @suspended without removing it from
    the CI routing manifests.
    """
    import subprocess
    from pathlib import Path

    tests_dir = Path(__file__).resolve().parent

    # Collect all suspended test names (final component after ::)
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', str(tests_dir),
         '--collect-only', '-q', '-m', 'suspended'],
        capture_output=True, text=True, timeout=30)

    suspended = set()
    for line in result.stdout.splitlines():
        if '::' in line:
            suspended.add(line.strip().split('::')[-1])

    assert len(suspended) > 0, "Expected at least one suspended test"

    # Check each CI routing manifest
    manifest_files = [
        'ci_heavy_tests.txt',
        'ci_huge_tests.txt',
        'ci_mega_tests.txt',
        'ci_long_tests.txt',
    ]

    violations = {}
    for mf in manifest_files:
        manifest_path = tests_dir / mf
        if not manifest_path.is_file():
            continue

        with open(manifest_path) as f:
            entries = set()
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Strip mesa:: prefix for comparison
                name = line
                if name.startswith('mesa::'):
                    name = name[6:]
                entries.add(name)

        overlap = suspended & entries
        if overlap:
            violations[mf] = sorted(overlap)

    assert not violations, (
        f"Suspended tests found in CI routing manifests (wasted task slots):\n"
        + "\n".join(
            f"  {mf}: {tests}"
            for mf, tests in sorted(violations.items())
        )
        + "\n\nFix: remove suspended test names from the manifest files."
    )
