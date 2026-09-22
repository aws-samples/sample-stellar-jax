# Regenerating the MESA reference data

The exact scripts used to generate `data/mesa_comparison/`. They assume a MESA install at
`$HOME/mesa` and the SDK at `$HOME/mesasdk` — adjust `MESA_DIR` / `MESASDK_ROOT` for your
environment. MESA build: git `f12c70cf` (reports version `r26.4.1`) for the `profiles/` FGONG
library + `rgb/`; release `r26.04.1` for the `results/` history tracks — identical MODE-A
physics (see `../README.md`, "Cross-build consistency").

- `inlist_comparison` — the MODE-A physics inlist (α=2.0 Cox, Z=0.014, Y=0.2695, gs98,
  Krishna-Swamy T(τ), PP+CNO, diffusion/overshoot/rotation off). `config/mesa_config.py` parses
  `../inlist_1.0Msun`, which is this same physics per mass.
- `run_dense.sh` / `run_dense_resume.sh` — evolve {1.0,1.2,1.5,2.0} M☉ ZAMS→RGB-tip with dense
  profile output → the `profiles/` FGONG library.
- `run_rgb.sh` — the RGB-tip run → `rgb/`.
- `extract_composition_profiles.sh` — re-extract full-composition profiles from saved models.

After a run, select snapshots by central-hydrogen `X_c` and gzip to `<stage>.FGONG.gz` as
described in `../profiles/README.md`. Verify the committed data against `../CHECKSUMS.md5`.
