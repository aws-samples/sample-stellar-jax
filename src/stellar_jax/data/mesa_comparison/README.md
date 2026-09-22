# MESA Comparison

Identical-physics MESA runs for validating stellar-jax.

## MESA Version

- **Release:** `r26.04.1` — used for the `results/` history tracks.
- **Build for the FGONG profiles:** the per-zone `profiles/` (and `rgb/`) FGONG library was
  generated with MESA **git build `f12c70cf`** (which reports version `r26.4.1`) — identical
  MODE-A physics, only the build differs. See `profiles/README.md` for that recipe.
- **Source:** Zenodo (https://doi.org/10.5281/zenodo.2602941 → record 19722306)
- **SDK:** `mesasdk-x86_64-linux-26.6.1`
- **Reproduce on:** any x86-64 Linux host with the above MESA build + SDK installed (an 8-core
  machine runs each track in ~1–3 h).

> **Cross-build consistency (verified):** the `results/` history tracks (release install) and the
> `profiles/`/`rgb/` FGONG library (git build `f12c70cf`) are the **same MESA version `r26.4.1`** with
> **identical MODE-A physics**. Comparing the two runs' history tracks at matched central-hydrogen
> `X_c` (1.0 & 2.0 M☉), `log_L`/`log_Teff`/`log_R` agree to 5 decimals — bit-identical where both
> sample the same `X_c`. So the two datasets are mutually consistent; the only difference is the
> install method (release vs git checkout of the same version).

## Physics (matching stellar-jax)

| Setting | Value |
|---------|-------|
| EOS | OPAL (MESA default) |
| Opacity | OPAL, gs98 tables |
| MLT | Böhm-Vitense (`MLT_option = 'Cox'`), α=2.0 |
| Nuclear | PP + CNO (`pp_cno_extras_o18_ne22.net`) |
| Atmosphere | Krishna Swamy T(τ) |
| Diffusion | OFF |
| Overshooting | OFF |
| Rotation | OFF |
| Z | 0.014 |
| Y | 0.2695 (= 0.2485 + 1.5×Z) |

## Runs

| Mass | File | Stop condition |
|------|------|---------------|
| 1.0 M☉ | `results/1.0Msun/history.data` | X_c < 0.01 (TAMS) |
| 1.2 M☉ | `results/1.2Msun/history.data` | X_c < 0.01 |
| 1.5 M☉ | `results/1.5Msun/history.data` | X_c < 0.01 |
| 2.0 M☉ | `results/2.0Msun/history.data` | X_c < 0.01 |

## How to reproduce

All generation scripts and inlists are committed in the repo — nothing lives only on
an internal host.

### Quick path (committed scripts)

The `generate/` subdirectory has the exact scripts used to produce this data:

- `generate/inlist_comparison` — the MODE-A physics inlist (matches `inlist_1.0Msun`).
- `generate/run_dense.sh` — dense profile run → `profiles/` FGONG library.
- `generate/run_rgb.sh` — RGB-tip run → `rgb/`.
- `generate/run_dense_resume.sh` — resume a dense run for remaining masses.
- `generate/extract_composition_profiles.sh` — re-extract composition profiles from
  saved `.mod` files.
- `generate/gen_module_grid.py` / `generate/gen_mlt_grid.py` — microphysics reference
  grids (`mesa_module_grid.npz`, `mesa_mlt_grid.npz`).

Each script sets `MESA_DIR`, `MESASDK_ROOT`, and runs the MESA `star/work` binary
with the MODE-A inlist. Adjust these paths for your MESA install location.

### Manual single-mass run

```bash
# On any x86-64 Linux host with MESA r26.04.1 + mesasdk 26.6.1 installed:
export MESASDK_ROOT=$HOME/mesasdk
source $MESASDK_ROOT/bin/mesasdk_init.sh
export MESA_DIR=$HOME/mesa-26.04.1
export OMP_NUM_THREADS=8

cd $MESA_DIR/star/work
cp <repo>/data/mesa_comparison/generate/inlist_comparison inlist_project
sed -i "s/initial_mass = 1.0/initial_mass = <MASS>/" inlist_project
rm -rf LOGS photos
./rn
# Output: LOGS/history.data
```

### Verification

After regeneration, verify against `CHECKSUMS.md5` (decompressed md5 for `.FGONG.gz`
files; raw md5 for everything else).

## Key columns in history.data

- `star_age` — age in years
- `log_L` — log10(L/L☉)
- `log_Teff` — log10(Teff/K)
- `log_R` — log10(R/R☉)
- `center_h1` — core hydrogen mass fraction (X_c)

## Notes on comparison

- The identical-physics comparison uses α=2.0 (from inlist) for BOTH codes (MODE A).
  This validates that stellar-jax reproduces MESA's physics; residuals should be <0.02 dex.
- The solar-calibrated comparison (MODE B) uses each code's OWN α_solar.
  stellar-jax's α_solar = 2.30371896 (see `config/calibration.py`); MESA's differs. Never cross-compare.
- The grid-interpolated "calibrated" MESA comparison set (α≈2.10) that existed on
  release/v1.0 is RETIRED — it was not a true iterative 1e-7 calibration and mixing
  it with the identical-physics set violated the never-cross-compare rule.
- Remaining differences in MODE A are attributable to: mesh resolution, timestepping,
  atmosphere integration details.
- MESA starts from PMS; our code starts from ZAMS. Compare only MS phase (X_c < X_init).

## Microphysics module reference grids (`data/mesa_module_grid.npz`, `data/mesa_mlt_grid.npz`)

Precomputed MESA **module outputs** (EOS / opacity / neutrino / MLT) over a fixed input grid, so the
per-module parity tests can compare our JAX microphysics against MESA numbers **without calling MESA
at test time** (fast, portable, deterministic). MODE-A physics (gs98, Z=0.014, net
`pp_cno_extras_o18_ne22`) — same as the FGONG/track data above.

- `mesa_module_grid.npz` — EOS/opacity/neutrino, 2400 pts on `(logT, logRho, X, Z)`.
  `eos` (rho,mu,nabla_ad,S,cp,chi_rho,chi_T,lnPgas), `kap` (log10κ + derivs), `neu` (ε_ν).
- `mesa_mlt_grid.npz` — MLT, 2400-pt self-consistent convective point set (`set_mlt`, Henyey).
- Regenerate ONLY if MODE-A physics changes:
  `STELLAR_MICROPHYSICS=mesa python data/mesa_comparison/generate/gen_module_grid.py` (+ `gen_mlt_grid.py`),
  on a host with the MESA `bind(C)` shims built. Nuclear is NOT gridded (the `net` shim segfaults
  standalone) — nuclear-vs-MESA stays validated via the FGONG `eps_nuc` tests.

Design + rationale: `MESA_MODULE_GRID_RECIPE.md` (this directory).

## Date produced

2026-06-09 (tracks/FGONG); 2026-08-26 (module reference grids)
