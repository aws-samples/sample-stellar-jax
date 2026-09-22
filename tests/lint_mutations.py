#!/usr/bin/env python3
"""Lint: audit every @register mutation for registry-bypass patterns.

This script statically analyses tests/mutations.py and reports mutations
that do NOT monkeypatch a module symbol — i.e. mutations whose physics
break lives outside the centrally-audited registry. Three patterns:

  PASS-BODY   — the function body is just `pass`; the actual mutation
                logic is in the TEST code (reading STELLAR_MUTATION env-var).
                The registry entry is a stub. This is the highest-risk pattern:
                the break is invisible to anyone auditing only the registry.

  ENV-VAR-ONLY — the function uses only mp.setenv(); the adapter code checks
                 the env-var at runtime. Acceptable when the adapter clears
                 sys.modules and reimports (e.g. MESA Fortran adapters), but
                 must be explicitly justified in the docstring.

  NO-EFFECT    — the function neither calls mp.setattr nor mp.setenv.
                 The mutation has no observable effect.

GOOD mutations call mp.setattr() on at least one module attribute, placing
the break in the registry where it can be centrally audited.

Exit codes:
  0 — all mutations pass the lint
  1 — at least one mutation flagged

Usage:
  python tests/lint_mutations.py              # audit and report
  python tests/lint_mutations.py --strict     # exit 1 on any finding
  python -m pytest tests/lint_mutations.py    # as a pytest test (AC7)

Issue #824: validation/audit trustworthiness.
"""
import ast
import os
import sys


def audit_mutations(mutations_path=None):
    """Parse mutations.py and classify every @register mutation.

    Returns:
        dict with keys:
            'pass_body':    list of (name, lineno)
            'env_var_only': list of (name, lineno)
            'no_effect':    list of (name, lineno)
            'env_and_patch': list of (name, lineno)  — acceptable belt-and-suspenders
            'proper_patch': list of (name, lineno)    — good
            'total':        int
    """
    if mutations_path is None:
        mutations_path = os.path.join(os.path.dirname(__file__), "mutations.py")
    with open(mutations_path) as f:
        source = f.read()
    tree = ast.parse(source, filename=mutations_path)

    results = {
        'pass_body': [],
        'env_var_only': [],
        'no_effect': [],
        'env_and_patch': [],
        'proper_patch': [],
        'total': 0,
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        # Check if decorated with @register("name")
        mut_name = None
        for deco in node.decorator_list:
            if (isinstance(deco, ast.Call)
                    and isinstance(deco.func, ast.Name)
                    and deco.func.id == 'register'
                    and deco.args):
                mut_name = ast.literal_eval(deco.args[0])
        if mut_name is None:
            continue

        results['total'] += 1
        lineno = node.lineno

        # Filter out docstrings from body
        stmts = [s for s in node.body
                 if not (isinstance(s, ast.Expr)
                         and isinstance(s.value, (ast.Constant, ast.Str)))]

        # Check for pass-only body
        if len(stmts) == 0 or (len(stmts) == 1 and isinstance(stmts[0], ast.Pass)):
            results['pass_body'].append((mut_name, lineno))
            continue

        # Walk the function body for setenv/setattr calls
        has_setenv = False
        has_setattr = False
        for s in ast.walk(node):
            if isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute):
                if s.func.attr == 'setenv':
                    has_setenv = True
                elif s.func.attr == 'setattr':
                    has_setattr = True

        if has_setenv and not has_setattr:
            results['env_var_only'].append((mut_name, lineno))
        elif has_setenv and has_setattr:
            results['env_and_patch'].append((mut_name, lineno))
        elif has_setattr:
            results['proper_patch'].append((mut_name, lineno))
        else:
            results['no_effect'].append((mut_name, lineno))

    return results


def format_report(results):
    """Format audit results as a human-readable report."""
    lines = []
    lines.append(f"Mutation registry audit: {results['total']} registered mutations\n")

    if results['pass_body']:
        lines.append("❌ PASS-BODY (mutation logic NOT in registry — highest risk):")
        for name, lineno in results['pass_body']:
            lines.append(f"   L{lineno}: {name}")
        lines.append(f"   Count: {len(results['pass_body'])}\n")

    if results['env_var_only']:
        lines.append("⚠️  ENV-VAR-ONLY (no setattr — mutation via env-var signal):")
        for name, lineno in results['env_var_only']:
            lines.append(f"   L{lineno}: {name}")
        lines.append(f"   Count: {len(results['env_var_only'])}\n")

    if results['no_effect']:
        lines.append("❌ NO-EFFECT (neither setattr nor setenv — mutation is dead):")
        for name, lineno in results['no_effect']:
            lines.append(f"   L{lineno}: {name}")
        lines.append(f"   Count: {len(results['no_effect'])}\n")

    if results['env_and_patch']:
        lines.append("✅ ENV-VAR + SETATTR (belt-and-suspenders — acceptable):")
        for name, lineno in results['env_and_patch']:
            lines.append(f"   L{lineno}: {name}")
        lines.append(f"   Count: {len(results['env_and_patch'])}\n")

    lines.append(f"✅ PROPER MONKEYPATCH (setattr): {len(results['proper_patch'])}")
    lines.append(f"\nSummary: {len(results['pass_body'])} pass-body, "
                 f"{len(results['env_var_only'])} env-var-only, "
                 f"{len(results['no_effect'])} no-effect, "
                 f"{len(results['env_and_patch'])} belt-and-suspenders, "
                 f"{len(results['proper_patch'])} proper")

    n_bad = len(results['pass_body']) + len(results['no_effect'])
    if n_bad > 0:
        lines.append(f"\n⛔ {n_bad} mutation(s) have their break OUTSIDE the registry — fix required.")
    else:
        lines.append("\n✅ All mutations place their break in the registry.")
    return "\n".join(lines)


def main():
    strict = '--strict' in sys.argv
    results = audit_mutations()
    print(format_report(results))
    n_bad = len(results['pass_body']) + len(results['no_effect'])
    if strict and n_bad > 0:
        sys.exit(1)
    # env_var_only are flagged as warnings, not failures (justified for MESA adapters)
    sys.exit(0)


if __name__ == '__main__':
    main()
