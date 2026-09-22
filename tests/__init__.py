# tests/ is a package — required for cross-module imports (e.g. from tests.helpers)
# and for the validate.py compat shim's relative imports.
#
# Belt-and-suspenders: ensure the repository root (parent of tests/) is always
# on sys.path so that top-level project modules (stellar, evolution/, solver/,
# constants, etc.) are importable regardless of how pytest was invoked or what
# the working directory is.  This fixes a CI environment issue where the old
# baked ci_entrypoint.sh runs `pytest tests/validate.py` from /w and the shim's
# wildcard imports trigger test_*.py module bodies that do `import stellar` or
# `from calibration.solar import ...` before conftest.py's sys.path setup has
# taken effect.
import os as _os
import sys as _sys

_repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)
