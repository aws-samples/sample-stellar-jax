# MESA Microphysics Backend (Forward-Only Validation)

Use MESA's battle-tested Fortran microphysics (EOS, opacity, neutrino, MLT)
as a **forward-only validation backend** for stellar-jax. This runs MESA physics through
compiled `bind(C)` Fortran shims loaded via plain `ctypes` — no gfort2py, no pyMesa.

**Validated modules: 4** (EOS, opacity/kap, neutrino/neu, MLT). Nuclear stays JAX in
both backends — see "Nuclear limitation" below.

**Purpose:** Validate that the JAX structure/evolution solver reproduces the same star
when fed MESA microphysics instead of the JAX reimplementations. This is a validation
configuration, not a production backend.

**Architecture:** Each MESA module is wrapped by a thin Fortran shim that:
1. Exposes a C-compatible interface (`bind(C)`, plain `real(c_double)` / `integer(c_int)`)
2. Self-initializes its own MESA runtime on first call (const_init, math_init, chem_init, etc.)
3. Internally constructs any MESA derived types needed
4. Calls the MESA routine and returns plain scalar outputs

This eliminates the gfort2py memory leak (per-call ctypes.Structure subclass creation),
the 2D-array marshalling bugs, and the pyMesa/gfModParser dependency chain.

## Status

| Module | Wrapper | Shared Library | Status |
|--------|---------|---------------|--------|
| EOS (`eosDT_get`) | `eos_wrapper.f90` | `libeos_wrapper.so` | ✅ validated |
| Opacity (`kap_get`) | `kap_wrapper.f90` | `libkap_wrapper.so` | ✅ validated |
| Neutrino (`neu_get`) | `neu_wrapper.f90` | `libneu_wrapper.so` | ✅ validated |
| MLT (`set_mlt`) | `mlt_wrapper.f90` | `libmlt_wrapper.so` | ✅ validated |
| Nuclear (`net_get`) | `net_wrapper.f90` | `libnet_wrapper.so` | ❌ NOT called — stays JAX |

## Measured JAX-vs-MESA agreement

Same inputs (T, P_gas, X, Z); 1.0 & 2.0 M☉ midMS FGONG + synthetic core→surface grid:

| Quantity | Interior (logT ≳ 5.5) | Surface / ionization zone |
|----------|----------------------|---------------------------|
| ρ (density) | ≤ 0.4% (|dex| ≤ 0.002) | ≤ 0.4% |
| ∇_ad | ≤ 0.5% (0.85% worst) | — |
| μ (mean mol. weight) | ≤ 1.4% | — |
| χ_ρ (= d ln P / d ln ρ) | ≤ 1% | — |
| cp, χ_T | ≤ 2% | 15–30% (expected: OPAL-only vs OPAL+FreeEOS+SCVH+Coulomb) |
| Entropy S | 25–45% offset (additive zero-point convention; flat to ~2% across 7 dex of P in ionized interior) |

**Entropy note:** The solver never consumes absolute S — eps_grav uses Form C
(cp/∇_ad, `solver/eps_grav.py`). The physical content (cp, ∇_ad) is asserted and agrees.

**Surface cp/χ_T note:** The gap is the expected completeness difference between
JAX's pure OPAL tables and MESA's multi-source blend. It does not affect the solver
because the Henyey Newton loop uses interior thermodynamics for the energy equation;
the surface BC depends on opacity and T(τ), not cp.

## Quick start

```bash
export STELLAR_MICROPHYSICS=mesa
export MESA_DIR=/path/to/mesa_libs   # wherever MESA libs + data are installed
export LD_LIBRARY_PATH=$MESA_DIR/lib:$MESA_DIR/sdk_lib:$LD_LIBRARY_PATH
python3 -c "from microphysics.eos import eos_lookup; print(eos_lookup(7.2, 17.0, 0.7, 0.02))"
```

## How it works

```
stellar-jax solver (JAX)
    │
    └── microphysics/__init__.py dispatch (checks STELLAR_MICROPHYSICS env var)
            │
            ├── 'jax' (default): pure JAX table interpolation (differentiable, production)
            │
            └── 'mesa': forward-only validation via bind(C) shims
                    │
                    ├── jax.pure_callback → ctypes → lib*_wrapper.so → MESA Fortran
                    │   No gfort2py, no pyMesa, no runtime compiler needed
                    │
                    └── NOT differentiable (forward-only validation backend)
```

## Building the wrapper libraries

Each shim must be compiled against the MESA installation:

```bash
# Requires MESA SDK 26.6.1 (GFortran 15.2.0) — must match .mod file format
export MESASDK_ROOT=/path/to/mesasdk
source $MESASDK_ROOT/bin/mesasdk_init.sh
export FC=$MESASDK_ROOT/bin/gfortran
export MESA_DIR=/path/to/mesa

# Common flags
FFLAGS="-shared -fPIC -I$MESA_DIR/include -L$MESA_DIR/lib"
LIBS="-lchem -lconst -lmath -lutils -lnum -lauto_diff"

# EOS wrapper
$FC $FFLAGS -o libeos_wrapper.so eos_wrapper.f90 -leos $LIBS

# Opacity wrapper
$FC $FFLAGS -o libkap_wrapper.so kap_wrapper.f90 -lkap $LIBS

# Neutrino wrapper
$FC $FFLAGS -o libneu_wrapper.so neu_wrapper.f90 -lneu $LIBS

# Nuclear wrapper
$FC $FFLAGS -o libnet_wrapper.so net_wrapper.f90 -lnet -lrates $LIBS

# MLT wrapper (existing pattern)
$FC $FFLAGS -o libmlt_wrapper.so mlt_wrapper.f90 -lturb -leos -lkap $LIBS
```

Place the resulting `.so` files in `$MESA_DIR/lib/` (alongside the MESA libraries themselves).

## Key design decisions

### Why bind(C) shims instead of gfort2py?

1. **No memory leak:** gfort2py 3.x creates fresh `ctypes.Structure` subclasses on every call
   to `ftype_assumed_shape.ctype()` (~16 KB per pointer arg per call). Over a full evolution
   loop this OOMs at ~60 GB. The bind(C) shims use fixed C types — zero per-call allocation.

2. **No derived-type marshalling bugs:** MESA uses `auto_diff_real_star_order1` types internally.
   gfort2py cannot reliably marshal these. The shim handles the conversion in Fortran.

3. **No runtime compiler needed:** gfort2py compiles small Fortran helpers at runtime for
   allocatable args. The bind(C) shims are pre-compiled — only the `.so` files are needed.

4. **Self-contained initialization:** Each shim calls `const_init`, `math_init`, `chem_init`,
   and its module-specific init internally. No shared global state to get out of sync.

### Why forward-only (no custom_vjp)?

Gradients are a property of the JAX production path. The MESA backend exists solely to validate
that the JAX reimplementations produce the same physics. Requiring differentiability added:
- Large `custom_vjp` machinery (hand-written backward passes through non-differentiable Fortran)
- Dependency on MESA's partial derivatives being correct for all regimes
- The impossible task of differentiating through `net_get` (complex network with workspace arrays)

Forward-only eliminates all of this complexity.

### Why is the nuclear network NOT called at runtime?

MESA's `net_get` requires a fully-initialized `Net_Info` workspace — a complex
derived type containing species arrays, rate tables, and screening factors that are
populated by a chain of init calls (`net_start_def → read_net_file → net_finish_def
→ net_setup_tables`). The `net_wrapper.f90` shim attempts this initialization, but
standalone (outside MESA's `star` module) the workspace cannot be fully populated:
calling `net_get_plain` with the incomplete workspace segfaults.

The wrapper is retained (compiles and loads → validates the build toolchain) but is
**never called**. Nuclear energy generation uses the JAX implementation
(`microphysics.nuclear.epsilon_nuclear`) in both backends. The parity test validates
that the 4 MESA modules (EOS, kap, neu, MLT) produce the same star — nuclear is
identical in both runs (JAX PP+CNO), so it cancels out of the comparison.

## Failure mode

If `STELLAR_MICROPHYSICS=mesa` is set but the wrapper libraries are unavailable:

```
RuntimeError: STELLAR_MICROPHYSICS=mesa requested but MESA backend is unavailable.
Reason: libeos_wrapper.so not found. Searched: ...
Either install the MESA wrapper libraries or unset STELLAR_MICROPHYSICS.
There is NO silent fallback to JAX — set STELLAR_MICROPHYSICS=jax explicitly if you want the JAX backend.
```

This is intentional — the prior silent fallback to JAX allowed hollow-green test results
where tests claimed to validate MESA physics but were actually running JAX.

## Version pinning

| Component | Version | Notes |
|-----------|---------|-------|
| MESA | `f12c70cf` | Pinned; all shims built against this |
| MESA SDK | `mesasdk-26.6.1` | GFortran 15.2.0 |
| Nuclear network | `pp_cno_extras_o18_ne22.net` | Valid at solar T (~1.5e7 K) |
| Zbase | 0.014 | Solar metallicity (configurable via `MESA_ZBASE`) |

## Limitations

- **Linux only** — shared library constraint.
- **CPU only** — Fortran calls can't run on GPU.
- **Not thread-safe** — MESA uses Fortran module-level state. `pure_callback` uses `vmap_method="sequential"`.
- **~10.5 GB data required** — must be accessible at `$MESA_DIR/data/`.
- **Forward-only** — NOT differentiable. Gradients use the JAX production path.
- **Slower than JAX** — each microphysics call goes through ctypes + Fortran. Acceptable for occasional validation runs.

**NOTE:** `test_mesa_microphysics_parity` was **removed** — the MESA .so backend's
`pure_callback` overhead (~3h/step) made evolved-track comparison infeasible. All module-level
parity coverage was superseded by precomputed grid tests (`test_{eos,opacity,neutrino,mlt}_parity_vs_mesa_grid`).
Evolved-track parity is validated by `test_mesa_comparison` (committed MESA `history.data`) and
`test_interior_structure_vs_mesa_fgong` (per-zone). See `MESA_MODULE_GRID_RECIPE.md`.