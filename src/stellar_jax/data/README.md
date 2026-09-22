# Reference data

This directory bundles the reference data `stellar-jax` uses — both our own generated outputs and
third-party tables. The third-party data is included **to make the code runnable** and is **not
covered by the repository's Apache-2.0 license**; it remains under its original terms. See
[`../../../THIRD_PARTY.md`](../../../THIRD_PARTY.md) for per-dataset sources, usage, and citations.

Contents:
- `model_s/` — Model S reference solar model + our ADIPLS kernels/frequencies (used by the demo).
- `mesa_comparison/` — our MESA (MODE-A) tracks/profiles, GYRE references, 2.0 M☉ ADIPLS kernels.
- `kepler_legacy/` — published Kepler LEGACY tables.
- `opal_4d.npz`, `eos_compact.npz`, `ferguson_4d.npz`, `potekhin_cond.npz`, `fd_electron_eos.npz`,
  `mesa_module_grid.npz`, `mesa_mlt_grid.npz` — microphysics tables the solver loads at runtime.

Some tables can be regenerated from source (see `scripts/` and `mesa_comparison/generate/`); OPAL and
Potekhin must be obtained from their original providers. See `THIRD_PARTY.md`.
