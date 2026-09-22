# MESA microphysics parity via a precomputed reference grid

**Status:** reference data generated + committed (2026-08-26); per-module grid tests live and
running per-wave (@fast/@smoke). **The former live-MESA parity test (`test_mesa_microphysics_parity`)
was removed** (2026-09-01): the MESA .so backend's `pure_callback` overhead (~3h/step, ~2M
ctypes round-trips per step) made evolved-track comparison infeasible, and all its module-level
coverage was superseded by the grid tests below. Evolved-track parity is gated separately by
`test_mesa_comparison` (4-mass MS HR tracks vs committed MESA `history.data`),
`test_eps_nuc_rgb_shell_vs_mesa_fgong` (RGB), and `test_interior_structure_vs_mesa_fgong` (per-zone).

## Problem with the live test

`test_mesa_microphysics_parity` validates our JAX microphysics (EOS, opacity, neutrino, MLT) against
MESA by calling the **live Fortran `bind(C)` shims at test time**. Four problems:

1. **Slow / non-portable** — needs a MESA build + SDK; ~33 min wall.
2. **Monolithic** — one test carries 4 module mutation gates → poor failure attribution.
3. **Skip-gated → vacuous green** — it `pytest.skip`s unless a change touched MESA-relevant code
   (`_mesa_code_changed()`), so it usually skipped (passed without executing) — passing did
   **not** mean the physics was validated. It genuinely ran only once (2026-08-18, `n_success=1`).
4. **Overfit sampling** — it probes only the zones of a single 2 M☉ model.

## Design: precompute MESA outputs on a grid, compare offline

MESA's module functions are deterministic (input → output), so we run them **once** over a fixed
grid, commit the outputs, and the tests compare our JAX modules against the saved arrays — **no live
MESA at test time**. This is the `*_vs_mesa_fgong` approach generalized from "one track's zones" to a
systematic grid.

**Mass is not an input** — microphysics is a function of local `(T, ρ, X, Z)` only, so **one grid
covers all masses**. Systematic grid coverage is *less* overfit than the single-model live test.

### Committed reference data (`data/`)
- **`mesa_module_grid.npz`** — EOS / opacity / neutrino, **2400 pts** on `(logT, logRho, X, Z)`:
  - `eos` (N,8): rho, mu, nabla_ad, S, cp, chi_rho, chi_T, lnPgas — MESA `eosDT_get`
  - `kap` (N,3): log10 κ, dlnκ/dlnρ, dlnκ/dlnT — MESA `kap_get`
  - `neu` (N,1): ε_ν [erg/g/s] — MESA `neu_get`
- **`mesa_mlt_grid.npz`** — MLT, **2400-pt self-consistent convective point set** (NOT a dense grid —
  `set_mlt` takes ~9 coupled inputs). T,P,ρ,μ,∇_ad from the EOS shim + κ from the opacity shim; sweep
  g and ∇_rad>∇_ad. `mlt_inputs` (N,9)=[nabla_rad,nad,T,P,rho,kappa,g,mu,alpha_mlt]; `mlt_gradT` (N,).

Generators: `data/mesa_comparison/generate/gen_module_grid.py` (eos/kap/neu) and `gen_mlt_grid.py`
(mlt). Both call the **raw ctypes shims** (`microphysics.mesa.bindings`, and `mlt_adapter._mlt_callback`
for MLT) — no jax, so they are jax-version-independent.

### Config alignment (MODE-A — verified)
The grid must match the committed FGONG/track reference physics. Verified: MESA r26.04.1 default
`kap_file_prefix='gs98'` == the committed inlist; `MESA_ZBASE=0.014`; net `pp_cno_extras_o18_ne22`.
Regenerate ONLY if MODE-A physics changes (same provenance rule as the FGONG data).

## The 5 modules — how each is covered
| Module | Coverage | Note |
|--------|----------|------|
| EOS | `mesa_module_grid.npz` | eosDT_get, 8 outputs |
| opacity | `mesa_module_grid.npz` | kap_get |
| neutrino | `mesa_module_grid.npz` | neu_get |
| MLT | `mesa_mlt_grid.npz` | set_mlt (Henyey); sampled point set |
| **nuclear** | **FGONG** (`test_nuclear_eps_vs_mesa_fgong`) | `net_get` shim **segfaults** standalone → never a shim-parity module; nuclear stays JAX and is validated vs MESA's per-zone `eps_nuc` in the FGONG files |

## Per-module tests
Each loads the grid with `STELLAR_MICROPHYSICS=jax`, evaluates our JAX module at the points, asserts
within the documented tolerances, and is **independently `@mutation`-gated**:
- `test_eos_parity_vs_mesa_grid` (eos_density_scale) — ρ ≤1%, ∇_ad ≤1%
- `test_opacity_parity_vs_mesa_grid` (opacity_bump_source) — ≤0.05 dex interior / 0.15 surface
- `test_neutrino_parity_vs_mesa_grid` (neutrino_rate_x3) — ≤5% above the taper
- `test_mlt_parity_vs_mesa_grid` (mlt_nabla_offset) — ∇ ≤5% in convective zones

**No skip-gate** — they need no MESA at test time, so they run **every wave** (real coverage, unlike
the former skip-gated live test). **EOS parametrization note:** our `eos_lookup` takes `(logT, logP)` while
the grid is keyed on `logRho` → reconcile via MESA's `lnPgas` column (feed our eos the point's P,
expect rho≈10^logRho + matching nabla_ad).

The live `test_mesa_microphysics_parity` was removed — a no-coverage-loss
swap. The 4 grid tests cover exactly its 4 mutation-gated modules; nuclear stays FGONG-covered.
The associated dead infrastructure (MESA path-gate config, 4 `mesa_*_dispatch` mutations, and
adapter env-var mutation hooks) was removed in the same change.

## Generation
The eos/kap/neu/mlt `bind(C)` shims are built against a **shared** MESA r26.04.1 build (a shared build
is required — a static MESA install cannot link a shared wrapper). Run the generators in
`data/mesa_comparison/generate/` (`gen_module_grid.py`, `gen_mlt_grid.py`) — they call the raw ctypes
shims (`microphysics.mesa.bindings`, `mlt_adapter._mlt_callback`) and are jax-version-independent.
