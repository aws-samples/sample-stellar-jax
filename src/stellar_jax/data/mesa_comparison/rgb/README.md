# RGB MESA reference tracks

Real MESA red-giant-branch reference tracks for validating stellar-jax's
post-main-sequence evolution.
**These are genuine MESA outputs, not synthetic data** — the sanity test
`tests/test_evolution.py::test_rgb_reference_sanity` enforces anti-synthetic guards.

## Provenance

| | |
|---|---|
| Code | MESA `f12c70cf` (MESA SDK, `pp_cno_extras_o18_ne22.net`) |
| Generated | 2026-06-17 from the pinned inlist below (swept over `initial_mass`) |
| Pinned inlist | [`inlist_rgb`](./inlist_rgb) (`initial_mass` is the swept parameter) |

## Physics — identical-physics MODE A

α<sub>MLT</sub> = 2.0 (Cox MLT), Z = 0.014, Y = 0.2695, gs98 opacities,
Krishna-Swamy T(τ) atmosphere (varying opacity), **no** element diffusion,
**no** convective overshoot, **no** rotation. Evolved from a pre-main-sequence
model to the **tip of the RGB (pre-helium-flash)**, terminated by
`power_he_burn_upper_limit = 1d3`.

> MODE A is the **only** mode valid for cross-code (MESA-vs-stellar-jax)
> comparison: both codes use the *same* α. Never compare against a track that
> used each code's own calibrated α (MODE B is for the Model S solar
> comparison only).

## Contents (per mass)

Stored under `<M>Msun/`:

- `history.data.gz` — the **full** MESA history (every model, all columns),
  gzipped (~3 MB vs ~24 MB raw). The loader reads `.gz` transparently.
- `tip.FGONG.gz` — the FGONG stellar-structure file at the **final (RGB-tip)**
  model (`iconst=15`, `ivar=40`).

Intermediate profiles (~580 MB/mass) are **not** committed; only the tip FGONG,
which is what the RGB structure tests need. Regenerate from `inlist_rgb` if more
are required.

## Tip summary (measured)

| Mass (M☉) | history rows | He-core mass @ tip (M☉) | log L @ tip | log ρ_c (post-MS → tip) | FGONG mesh |
|-----------|-------------:|------------------------:|------------:|-------------------------|-----------:|
| 1.0 | 10205 | 0.4765 | 3.446 | 2.77 → 5.98 | 2191 |
| 1.2 | 10045 | 0.4759 | 3.443 | 2.90 → 5.97 | 2177 |
| 1.5 | 10015 | 0.4758 | 3.441 | 2.84 → 5.97 | 2173 |
| 2.0 | 6939 | 0.4553 | 3.330 | 2.61 → 5.90 | 2130 |

The ~0.476 M☉ tip He-core for 1.0–1.5 M☉ is the expected near-universal
degenerate-flash core mass. The 2.0 M☉ track sits near the degenerate/
non-degenerate transition; its tip core is slightly lower (0.455 M☉) but still a
degenerate flash. Expected ranges are set in `_RGB_HE_CORE_TIP_RANGE`.

## Loading

```python
import mesa_rgb  # tests/mesa_rgb.py

h = mesa_rgb.load_rgb_history(1.0)        # {column_name: ndarray}, reads .gz
fg = mesa_rgb.load_rgb_tip_fgong(1.0)     # {nn, iconst, ivar, iversion, glob, data}
```
