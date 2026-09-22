#!/usr/bin/env bash
#
# preflight.sh — FAST pre-push sanity for stellar-jax.
#
# Purpose: give a quick, local signal BEFORE pushing, so the slow (~40–60 min)
# compiled CI wave is reserved for the definitive verdict — not the dev loop.
#
# Golden rule: "fast" == do NOT run the full solver. Reducing max_steps does NOT
# cut the ~10–16 min XLA compile (everything is lax.scan, trip-count-independent).
# So we either isolate the changed function (the @fast component tests, which call
# ONE physics function on a fixed FGONG / fixed input — no evolve_star) or run the
# solver eagerly (JAX_DISABLE_JIT=1) for a handful of steps just to catch crashes.
#
# CI (per-test rows in DynamoDB) remains the compiled source of truth. Preflight
# passing does NOT mean CI passes; preflight FAILING means do not bother pushing.
#
# Usage:
#   scripts/preflight.sh                # Tier0 + Tier1 (full @fast set)
#   scripts/preflight.sh -k opacity     # scope Tier1 to matching tests (RECOMMENDED)
#   scripts/preflight.sh --eager        # also run an eager 5-step evolve_star crash check
#   scripts/preflight.sh -f a.py -f b.py # py_compile these files (default: changed vs origin/main)
#
# Exit code: 0 = all tiers passed, non-zero = a tier failed.

set -uo pipefail

# ---- locate repo root ----
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
KFILTER=""
EAGER=0
FILES=()

while [ $# -gt 0 ]; do
  case "$1" in
    -k) KFILTER="${2:-}"; shift 2 ;;
    --eager) EAGER=1; shift ;;
    -f) FILES+=("${2:-}"); shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# timeout wrapper (present on Linux agent/CI; may be absent on macOS dev)
if command -v timeout >/dev/null 2>&1; then TO() { timeout "$@"; }; else TO() { shift; "$@"; }; fi

hr() { printf '%s\n' "------------------------------------------------------------"; }
step() { printf '\n\033[1m[preflight] %s\033[0m\n' "$1"; }

# ---- determine changed python files (for Tier 0 compile) ----
if [ "${#FILES[@]}" -eq 0 ]; then
  mapfile -t FILES < <(git diff --name-only origin/main...HEAD 2>/dev/null; git diff --name-only 2>/dev/null; git diff --name-only --cached 2>/dev/null)
fi
# unique .py only, that exist
PYFILES=()
for f in "${FILES[@]:-}"; do
  [ -n "$f" ] || continue
  case "$f" in *.py) [ -f "$f" ] && PYFILES+=("$f") ;; esac
done
# de-dup
if [ "${#PYFILES[@]}" -gt 0 ]; then
  mapfile -t PYFILES < <(printf '%s\n' "${PYFILES[@]}" | sort -u)
fi

rc=0

# =====================================================================
step "Tier 0 — syntax, import, collection (seconds)"
hr
if [ "${#PYFILES[@]}" -gt 0 ]; then
  echo "py_compile: ${PYFILES[*]}"
  TO 120 "$PY" -m py_compile "${PYFILES[@]}" || { echo "FAIL: py_compile"; rc=1; }
else
  echo "no changed .py files detected — skipping py_compile"
fi

echo "lint: gradient_policy annotations ..."
TO 30 "$PY" scripts/lint_gradient_policy.py || { echo "FAIL: gradient_policy lint"; rc=1; }
TO 30 "$PY" scripts/lint_tracker_refs.py || { echo "FAIL: tracker_refs lint"; rc=1; }
TO 30 "$PY" scripts/lint_signatures.py || { echo "FAIL: signatures lint"; rc=1; }
TO 30 "$PY" scripts/lint_file_size.py || { echo "FAIL: file_size lint"; rc=1; }

echo "import stellar_jax ..."
TO 300 "$PY" -c "import stellar_jax; print('  import OK')" || { echo "FAIL: import stellar_jax"; rc=1; }

COLLECT_ARGS=(tests/ -m fast --collect-only -q -p no:cacheprovider)
[ -n "$KFILTER" ] && COLLECT_ARGS+=(-k "$KFILTER")
echo "pytest --collect-only -m fast ${KFILTER:+-k $KFILTER} ..."
TO 300 "$PY" -m pytest "${COLLECT_ARGS[@]}" 2>&1 | tail -3 || { echo "FAIL: collection"; rc=1; }

if [ "$rc" -ne 0 ]; then
  hr; echo "Tier 0 FAILED — fix syntax/import before anything else."; exit "$rc"
fi

# =====================================================================
step "Tier 1 — @fast component tests (isolated physics, no evolve_star)"
echo "NOTE: full @fast is ~5–8 min COLD (per-function JIT compiles). Scope with -k <area>"
echo "      to your change for a seconds-to-1min signal."
hr
T1_ARGS=(tests/ -m fast -q -p no:cacheprovider --durations=10)
[ -n "$KFILTER" ] && T1_ARGS+=(-k "$KFILTER")
TO 1200 "$PY" -m pytest "${T1_ARGS[@]}"
t1=$?
[ "$t1" -ne 0 ] && { echo "FAIL: @fast tests (rc=$t1)"; rc=1; }

# =====================================================================
if [ "$EAGER" -eq 1 ]; then
  step "Tier 1b — eager 5-step evolve_star (crash check ONLY, JAX_DISABLE_JIT=1)"
  echo "Not a physics verdict — just proves the scan body runs without crashing."
  hr
  TO 600 env JAX_DISABLE_JIT=1 "$PY" -c "
from stellar import evolve_star
r = evolve_star(1.0, Z=0.014, max_steps=5, fixed_dt=1e7)
import numpy as np
ll = float(np.asarray(r['log_L'])[-1])
assert np.isfinite(ll), 'log_L not finite'
print(f'  eager 5-step OK: log_L[-1]={ll:.4f}')
" || { echo "FAIL: eager evolve_star crashed / NaN"; rc=1; }
fi

# =====================================================================
hr
if [ "$rc" -eq 0 ]; then
  echo "PREFLIGHT PASSED. Push a DRAFT PR — CI (compiled, per-test DynamoDB rows) is the"
  echo "definitive verdict. Do NOT run the full heavy/integration test locally (40-min"
  echo "compile trap), and do NOT push no-op 'CI retry' commits (they supersede the wave)."
else
  echo "PREFLIGHT FAILED (rc=$rc). Fix locally before pushing."
fi
exit "$rc"
