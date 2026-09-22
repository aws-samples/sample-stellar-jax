"""Structural audit: named physics floors are correctly defined and substituted.

Checks:
  (a) Constant values match expectations (regression).
  (b) No bare literals remain at sites that should reference the named constants
      — a grep-based structural check that prevents reintroducing unnamed floors.

Marks: @pytest.mark.fast — pure Python, no evolve_star, no JIT.
"""

import ast
import os
import re

import pytest

# All tests are fast: structural audits, no evolve_star / JIT.
pytestmark = pytest.mark.fast

SRC = os.path.join(os.path.dirname(__file__), os.pardir, "src", "stellar_jax")


# ======================================================================
# (a) Constant-value regression
# ======================================================================

class TestPhysicsFloorValues:
    """Assert named physics-floor constants have their documented values."""

    def test_nad_cp_floor(self):
        from stellar_jax.config.physics_floors import NAD_CP_FLOOR
        assert NAD_CP_FLOOR == 0.05, f"NAD_CP_FLOOR changed to {NAD_CP_FLOOR}"

    def test_nad_upper(self):
        from stellar_jax.config.physics_floors import NAD_UPPER
        assert NAD_UPPER == 0.45, f"NAD_UPPER changed to {NAD_UPPER}"

    def test_pgas_frac_floor(self):
        from stellar_jax.config.physics_floors import PGAS_FRAC_FLOOR
        assert PGAS_FRAC_FLOOR == 1e-3, f"PGAS_FRAC_FLOOR changed to {PGAS_FRAC_FLOOR}"

    def test_x_min_pp(self):
        from stellar_jax.config.physics_floors import X_MIN_PP
        assert X_MIN_PP == 1e-3, f"X_MIN_PP changed to {X_MIN_PP}"


# ======================================================================
# (b) Structural audit — no bare literals at named-floor sites
# ======================================================================

# Files that MUST reference PGAS_FRAC_FLOOR (not bare 1e-3 * P).
_PGAS_FILES = [
    "structure.py",
    "solver/residual.py",
    "solver/jacobian.py",
    "solver/eps_grav.py",
    "solver/shell_data.py",
    "evolution/_core.py",
    "evolution/zams.py",
    "fgong/builder.py",
    "fgong/hires_profile.py",
    "calibration/comparison.py",
    "evolution/adaptive/forward.py",
]

# Files that MUST reference NAD_CP_FLOOR (not bare 0.05 in nad context).
_NAD_FILES = [
    "transport/mlt.py",
    "microphysics/eos.py",
    "evolution/step.py",
    "calibration/diagnostic.py",
]

# Pattern: bare `1e-3 * P` in a gas-pressure floor context.
# Matches "1e-3 * P" or "1e-3*P" with various P suffixes.
_BARE_PGAS_RE = re.compile(r"\b1e-3\s*\*\s*P")

# Pattern: bare `0.05` used with nad in a maximum/clip context.
# Matches "maximum(nad..., 0.05)" or "clip(..., 0.05, ...)".
_BARE_NAD_RE = re.compile(r"(?:maximum|clip)\([^)]*\bnad\b[^)]*,\s*0\.05")


class TestNoBareFloorLiterals:
    """Ensure named-floor sites use the constant, not the bare literal.

    These are structural (grep-level) checks that guard against accidentally
    reintroducing an unnamed literal at a site that should use the constant.
    """

    @pytest.mark.parametrize("relpath", _PGAS_FILES)
    def test_no_bare_pgas_literal(self, relpath):
        """No bare '1e-3 * P...' in gas-pressure floor sites."""
        fpath = os.path.join(SRC, relpath)
        assert os.path.exists(fpath), f"File not found: {fpath}"
        with open(fpath) as f:
            content = f.read()
        # Exclude comment lines from the search.
        code_lines = [
            line for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        code = "\n".join(code_lines)
        matches = _BARE_PGAS_RE.findall(code)
        assert not matches, (
            f"{relpath} has bare '1e-3 * P' literal(s) — use PGAS_FRAC_FLOOR instead. "
            f"Found: {matches}"
        )

    @pytest.mark.parametrize("relpath", _NAD_FILES)
    def test_no_bare_nad_literal(self, relpath):
        """No bare '0.05' in nad floor/clip sites."""
        fpath = os.path.join(SRC, relpath)
        assert os.path.exists(fpath), f"File not found: {fpath}"
        with open(fpath) as f:
            content = f.read()
        code_lines = [
            line for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        code = "\n".join(code_lines)
        matches = _BARE_NAD_RE.findall(code)
        assert not matches, (
            f"{relpath} has bare '0.05' in nad floor context — use NAD_CP_FLOOR instead. "
            f"Found: {matches}"
        )

    def test_nuclear_uses_x_min_pp(self):
        """nuclear.py psipp line uses X_MIN_PP, not bare 1e-3."""
        fpath = os.path.join(SRC, "microphysics", "nuclear.py")
        with open(fpath) as f:
            content = f.read()
        # The psipp line should reference X_MIN_PP.
        assert "X_MIN_PP" in content, "nuclear.py does not reference X_MIN_PP"
        # And should NOT have bare 'maximum(X, 1e-3)' in a non-comment line.
        code_lines = [
            line for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        bare = [l for l in code_lines if re.search(r"maximum\(X,\s*1e-3\)", l)]
        assert not bare, (
            f"nuclear.py has bare 'maximum(X, 1e-3)' — use X_MIN_PP. Found: {bare}"
        )

    def test_pgas_files_import_constant(self):
        """Every PGAS file imports PGAS_FRAC_FLOOR."""
        for relpath in _PGAS_FILES:
            fpath = os.path.join(SRC, relpath)
            with open(fpath) as f:
                content = f.read()
            assert "PGAS_FRAC_FLOOR" in content, (
                f"{relpath} does not import/reference PGAS_FRAC_FLOOR"
            )

    def test_nad_files_import_constant(self):
        """Every NAD file imports NAD_CP_FLOOR."""
        for relpath in _NAD_FILES:
            fpath = os.path.join(SRC, relpath)
            with open(fpath) as f:
                content = f.read()
            assert "NAD_CP_FLOOR" in content, (
                f"{relpath} does not import/reference NAD_CP_FLOOR"
            )
