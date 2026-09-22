# Third-party reference data — attribution, usage, and license scope

`stellar-jax`'s own source code is licensed under Apache-2.0 (see [`LICENSE`](LICENSE)). To make the
code **runnable out of the box** and reproducible, this repository also bundles reference data produced
by third parties, under `src/stellar_jax/data/`.

> **License scope.** The bundled reference data is **NOT covered by the Apache-2.0 license.** It was
> copied into this repository solely to make the code functional; each dataset remains under the terms
> of its **original source** (below). Please cite the original works, and consult their terms before
> redistributing or reusing the data.

## Microphysics tables (loaded by the solver at runtime)

- **OPAL opacities & equation of state** — `opal_4d.npz`, `eos_compact.npz`. OPAL project, Lawrence
  Livermore National Laboratory; Iglesias & Rogers (1996), ApJ 464, 943. Used by
  `microphysics/opacity.py` and `microphysics/eos.py`.
- **Low-temperature opacities** — `ferguson_4d.npz`. Ferguson et al. (2005), ApJ 623, 585.
- **Conductive (electron) opacities** — `potekhin_cond.npz`. A. Y. Potekhin and collaborators.
- **Fermi–Dirac electron EOS** — `fd_electron_eos.npz`. Computed following Timmes & Swesty (2000),
  ApJS 126, 501.
- **MESA microphysics grids** — `mesa_module_grid.npz`, `mesa_mlt_grid.npz`. Sampled from MESA modules.

## Reference models & validation data

- **Model S** — `model_s/`. Reference solar model; Christensen-Dalsgaard et al. (1996), Science 272,
  1286. Source: https://users-phys.au.dk/~jcd/solar_models/ . Used by the seismic-gradients demo.
- **MESA / GYRE / ADIPLS reference outputs** — `mesa_comparison/` (tracks, profiles, RGB, GYRE
  references) and `model_s/adipls_kernels/`. These are **our own runs' outputs**, generated under
  identical-physics ("MODE-A") settings with MESA (https://mesastar.org), GYRE (Townsend & Teitler
  2013, MNRAS 435, 3406), and ADIPLS (Christensen-Dalsgaard 2008, Ap&SS 316, 113); provided as a
  validation reference. Cite those tools.
- **Kepler LEGACY sample** — `kepler_legacy/`. Lund et al. (2017), ApJ 835, 172 (+ erratum 2017, ApJ
  850, 110); Silva Aguirre et al. (2017), ApJ 835, 173. Via CDS/VizieR (J/ApJ/835/172, J/ApJ/835/173).

## Regenerating the tables (optional)

Several tables can be rebuilt from source with the scripts under `scripts/` and
`src/stellar_jax/data/mesa_comparison/generate/` (e.g. `tools/build_ferguson_4d.py`,
`scripts/generate_fd_electron_table.py`, `gen_module_grid.py`). The OPAL and Potekhin tables must be
obtained from their original providers. Regenerated `.npz` files are validated by the test suite rather
than by exact checksum (numpy `.npz` archives are not byte-reproducible).
