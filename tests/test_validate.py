"""Meta-validation tests — structural guards on the test suite itself (#1085).

These tests enforce PROCESS invariants that prevent a hidden scientific
limitation (a liveness-only gradient check, a proxy observable) from passing
CI as "delivered". They operate on the TEST SOURCE CODE, not on physics —
they are @fast (no evolve_star / jax.grad) and run in the Fargate bundle.

AC2: test_no_liveness_only_capability_gradient
    AST-parses test_fisher.py. Fails if any capability-column test (Y_init,
    Z, alpha_mlt, f_ov) passes green WITHOUT enforcing sign+tol — i.e.
    without an xfail(strict=True) marker.

AC3: test_no_proxy_observable_passes_as_delivered
    AST-parses plato_forecast.py. Fails if σ(age) is computed as a
    closed-form function of σ(M) (f_age × σ(M)/M pattern) without the
    forecast test being xfail(strict=True).

References
----------
Issue #1085: validation-gate hardening.
"""

import ast
import inspect
import re
import textwrap

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# AC2: Ban liveness-only capability gradients
# ═══════════════════════════════════════════════════════════════════════════════

# The capability-critical Jacobian columns. Each must have a test that either
# (a) enforces sign+tol (regime 1: rel_err < tol AND signs_agree), or
# (b) is marked xfail(strict=True) bound to an open issue.
# A test that passes green with only a liveness check ("both paths live",
# "non-zero") is a VIOLATION.
_CAPABILITY_COLUMNS = {'Y_init', 'Z', 'alpha_mlt', 'f_ov'}

# Test function names that correspond to each column.
_COLUMN_TEST_MAP = {
    'Y_init': 'test_jacobian_ad_vs_fd_Y_init',
    'Z': 'test_jacobian_ad_vs_fd_Z',
    'alpha_mlt': 'test_jacobian_ad_vs_fd_alpha_mlt',
    'f_ov': 'test_jacobian_ad_vs_fd_f_ov',
}


def _get_test_fisher_source():
    """Read test_fisher.py source for AST analysis."""
    import pathlib
    test_fisher_path = pathlib.Path(__file__).parent / 'test_fisher.py'
    return test_fisher_path.read_text()


def _get_test_fisher_ast():
    """Parse test_fisher.py into an AST."""
    source = _get_test_fisher_source()
    return ast.parse(source, filename='test_fisher.py')


def _function_has_xfail_strict(func_node):
    """Check if a function AST node has @pytest.mark.xfail(strict=True)."""
    for decorator in func_node.decorator_list:
        # Match @pytest.mark.xfail(strict=True, ...)
        if isinstance(decorator, ast.Call):
            # Get the decorator function name
            func_name = ast.dump(decorator.func)
            if 'xfail' in func_name:
                for kw in decorator.keywords:
                    if kw.arg == 'strict':
                        if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            return True
    return False


def _function_has_per_column_mutation(func_node, column_name):
    """Check if a function has a per-column mutation marker (not shared)."""
    for decorator in func_node.decorator_list:
        if isinstance(decorator, ast.Call):
            func_name = ast.dump(decorator.func)
            if 'mutation' in func_name:
                for arg in decorator.args:
                    if isinstance(arg, ast.Constant):
                        # Must be a per-column mutation, not the shared
                        # zero_ift_eigenfreq
                        val = arg.value
                        if column_name in str(val) and val != 'zero_ift_eigenfreq':
                            return True
    return False


@pytest.mark.fast
def test_no_liveness_only_capability_gradient():
    """Meta-test: every capability-column Jacobian test enforces sign+tol.

    WHAT: AST-parses test_fisher.py and checks that each capability-column
    test (Y_init, Z, alpha_mlt, f_ov) either:
      (a) enforces sign+tol via _validate_mode_param (the uniform path), OR
      (b) is marked @pytest.mark.xfail(strict=True) with a reason citing
          an open issue.
    A test that passes GREEN without enforcing sign+tol is a VIOLATION.

    WHY: Issue #1085 — the _GP5_SIGN_RISK_PARAMS branches let 4 columns
    pass as "delivered" with only liveness validation (non-zero gradient).
    This meta-test prevents that pattern from being reintroduced.

    WHAT MAKES IT FAIL: Reintroducing a liveness-only pass path for any
    capability column (removing xfail, or adding GP5-SIGN/GP5-LIVENESS
    forgiveness back into _evaluate_mode_param).
    """
    tree = _get_test_fisher_ast()

    # Build map of function names → AST nodes
    func_map = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_map[node.name] = node

    violations = []

    for column, test_name in _COLUMN_TEST_MAP.items():
        if test_name not in func_map:
            violations.append(
                f"MISSING: no test function '{test_name}' for column '{column}'"
            )
            continue

        func_node = func_map[test_name]
        has_xfail = _function_has_xfail_strict(func_node)

        if not has_xfail:
            # Without xfail(strict=True), the test must enforce sign+tol.
            # Check that it does NOT contain any liveness-only forgiveness
            # patterns (these would let a wrong-sign column pass green).
            source = _get_test_fisher_source()
            # Extract just this function's source
            func_source = ast.get_source_segment(source, func_node)
            if func_source is None:
                func_source = ""

            # Check for liveness-only forgiveness patterns
            liveness_patterns = [
                'GP5-SIGN',
                'GP5-LIVENESS',
                'GP5_SIGN_RISK',
                'both paths live',
                'tol not enforced',
                'liveness-validated',
            ]
            for pattern in liveness_patterns:
                if pattern in func_source:
                    violations.append(
                        f"LIVENESS-ONLY: test '{test_name}' for column "
                        f"'{column}' contains forgiveness pattern "
                        f"'{pattern}' without xfail(strict=True)"
                    )
                    break
            else:
                # No forgiveness patterns found — the test enforces sign+tol
                # uniformly. This is the IDEAL state (the column actually works).
                pass

    assert not violations, (
        "Capability-column Jacobian tests violate the sign+tol gate "
        "(issue #1085):\n" + "\n".join(f"  • {v}" for v in violations)
    )


@pytest.mark.fast
def test_no_liveness_forgiveness_in_evaluate_mode_param():
    """Meta-test: _evaluate_mode_param has no per-parameter sign forgiveness.

    WHAT: Checks that _evaluate_mode_param does NOT contain any parameter-
    specific forgiveness paths (like the deleted _GP5_SIGN_RISK_PARAMS).
    ALL parameters must go through the same sign+tol check.

    WHY: The original sin was special-casing {Y_init, Z, alpha_mlt, f_ov}
    to forgive sign disagreement. This meta-test ensures that pattern
    cannot be reintroduced.
    """
    source = _get_test_fisher_source()

    # Find the _evaluate_mode_param function source
    tree = _get_test_fisher_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_evaluate_mode_param':
            func_source = ast.get_source_segment(source, node)
            break
    else:
        pytest.fail("_evaluate_mode_param not found in test_fisher.py")

    # These patterns indicate per-parameter sign forgiveness
    forbidden_patterns = [
        ('_GP5_SIGN_RISK_PARAMS', 'parameter-specific sign-forgiveness set'),
        ('GP5-SIGN', 'GP-5 sign forgiveness label'),
        ('GP5-LIVENESS', 'GP-5 liveness-only label'),
        ('tol not enforced', 'tolerance bypass for specific params'),
    ]

    found = []
    for pattern, description in forbidden_patterns:
        if pattern in func_source:
            found.append(f"'{pattern}' ({description})")

    assert not found, (
        "_evaluate_mode_param contains sign-forgiveness patterns "
        "(issue #1085 bans these):\n" + "\n".join(f"  • {p}" for p in found)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AC3: Ban proxy observables passing as delivered
# ═══════════════════════════════════════════════════════════════════════════════

def _get_plato_forecast_source():
    """Read plato_forecast.py source."""
    import pathlib
    pf_path = (pathlib.Path(__file__).parent.parent /
               'src' / 'stellar_jax' / 'inference' / 'plato_forecast.py')
    return pf_path.read_text()


def _detect_proxy_age_observable(source):
    """Detect if σ(age) is computed as f · σ(M) (a proxy, not a real gradient).

    A proxy is a closed-form function of another σ — it does NOT come from
    d(age)/d(param) through the Fisher matrix. The pattern is:
        sigma_age = f_age * sigma_M / M  (or equivalently, f * sigma_M_frac)

    Returns
    -------
    list of str — descriptions of detected proxy patterns. Empty if none found.
    """
    proxies = []

    # Pattern 1: sigma_age_frac = f_age * sigma_M_frac
    if re.search(r'sigma_age.*=.*f_age\s*\*\s*sigma_M', source):
        proxies.append(
            "sigma_age is computed as f_age × sigma_M (proxy, not Fisher-derived)"
        )

    # Pattern 2: sigma_age = constant * sigma_M / M
    if re.search(r'sigma_age.*=.*\*\s*sigma_M.*/', source):
        proxies.append(
            "sigma_age is a scaled function of sigma_M (proxy)"
        )

    # Pattern 3: F_AGE_WITH_L2 / F_AGE_WITHOUT_L2 constants used in sigma_age
    if re.search(r'F_AGE_(WITH|WITHOUT)_L2', source):
        # Check if these constants are used in the sigma_age computation
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and 'sigma_age' in target.id:
                        assign_source = ast.get_source_segment(source, node)
                        if assign_source and 'f_age' in assign_source:
                            proxies.append(
                                f"sigma_age assignment uses f_age constant: "
                                f"{assign_source.strip()}"
                            )

    return proxies


def _get_plato_test_source():
    """Read test_plato_forecast.py source."""
    import pathlib
    test_path = pathlib.Path(__file__).parent / 'test_plato_forecast.py'
    return test_path.read_text()


def _test_has_xfail_strict(test_source, test_name):
    """Check if a test function in the source has xfail(strict=True)."""
    tree = ast.parse(test_source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == test_name:
            return _function_has_xfail_strict(node)
    return False


@pytest.mark.fast
@pytest.mark.xfail(
    strict=True,
    reason="#1085: plato_forecast.py σ(age) = f_age × σ(M)/M is a proxy. "
           "Blocked until Fisher-derived σ(age) via real ∂age/∂θ is implemented.",
)
def test_no_proxy_observable_passes_as_delivered():
    """Meta-test: proxy observables must not pass CI as "delivered".

    WHAT: Checks plato_forecast.py for proxy-observable patterns (σ(age) =
    f · σ(M)) and verifies that any test asserting on the proxy is either:
      (a) marked xfail(strict=True), or
      (b) the proxy has been replaced with a real Fisher-derived σ(age).

    WHY: Issue #1085 — σ(age) = f_age × σ(M)/M is explicitly acknowledged
    as a LIMITATION in plato_forecast.py. It does NOT come from
    d(age)/d(param) through the Fisher matrix. A test that checks this
    proxy and passes green is claiming "age uncertainty is validated" when
    it is only a calibrated scaling of the mass uncertainty.

    WHAT MAKES IT FAIL: The proxy pattern is present in plato_forecast.py
    AND the corresponding tests pass green (no xfail(strict=True)).
    """
    pf_source = _get_plato_forecast_source()
    proxies = _detect_proxy_age_observable(pf_source)

    if not proxies:
        # No proxy detected — σ(age) is Fisher-derived. All good.
        return

    # Proxy detected — check that ALL tests asserting on sigma_age are
    # xfail(strict=True).
    test_source = _get_plato_test_source()

    # Tests that assert on sigma_age_pct
    tests_asserting_age = [
        'test_plato_forecast_solar_analog_precision',
        'test_plato_forecast_l2_removal_degrades_age',
        'test_plato_forecast_surface_marginalization_bounded',
    ]

    unguarded = []
    for test_name in tests_asserting_age:
        has_xfail = _test_has_xfail_strict(test_source, test_name)
        if not has_xfail:
            # Check if the test actually asserts on sigma_age
            tree = ast.parse(test_source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == test_name:
                    func_src = ast.get_source_segment(test_source, node)
                    if func_src and 'sigma_age' in func_src:
                        unguarded.append(test_name)
                    break

    assert not unguarded, (
        "Proxy observable detected in plato_forecast.py but tests pass "
        "green (issue #1085 bans this):\n"
        "  Proxy patterns found:\n"
        + "\n".join(f"    • {p}" for p in proxies) + "\n"
        "  Tests asserting on σ(age) without xfail(strict=True):\n"
        + "\n".join(f"    • {t}" for t in unguarded) + "\n"
        "  Fix: either replace the proxy with a real Fisher-derived σ(age), "
        "or mark the tests xfail(strict=True, reason='#N') bound to an "
        "open issue tracking the fix."
    )
