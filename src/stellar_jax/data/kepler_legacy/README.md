# Kepler LEGACY Sample — Observed Asteroseismic Data

Public oscillation frequencies and stellar parameters from the Kepler LEGACY project.

## Sources

- **Lund et al. 2017** (Paper I: oscillation mode parameters)
  ApJ 835, 172. arXiv: [1612.00436](https://arxiv.org/abs/1612.00436)
  CDS catalog: J/ApJ/835/172 (erratum-corrected tables from 2017 ApJ 850, 110)

- **Silva Aguirre et al. 2017** (Paper II: radii, masses, and ages)
  ApJ 835, 173. arXiv: [1611.08776](https://arxiv.org/abs/1611.08776)
  CDS catalog: J/ApJ/835/173

## Files

```
raw/
  lund2017_table1.dat          — 66-star target list (KIC, name, Kpmag, numax, Dnu, Teff, [Fe/H])
  lund2017_table6_frequencies.dat — Individual mode frequencies (KIC, n, l, freq±, amp±, width±)
  lund2017_table7_ratios.dat   — Frequency difference ratios r_01, r_10, r_02 (KIC, type, n, ratio±)
  silva2017_table3_global.dat  — Global asteroseismic + atmospheric properties (66 stars)
  silva2017_table4_params.dat  — Stellar params from 6 pipelines (mass, radius, age, logg, ...)
```

## Selected Targets

For initial validation, we use 4 well-characterized main-sequence targets (all "Simple"
oscillation spectra, high S/N, well-constrained parameters):

| KIC       | Name     | M/M☉        | Age/Gyr      | Δν/μHz  | Teff/K | [Fe/H] |
|-----------|----------|-------------|--------------|---------|--------|--------|
| 12069424  | 16 Cyg A | 1.07 ± 0.02 | 6.9 ± 0.4   | 103.3   | 5825   | +0.10  |
| 12069449  | 16 Cyg B | 1.01 ± 0.02 | 7.1 ± 0.4   | 116.9   | 5750   | +0.05  |
| 6106415   | Perky    | 1.04 ± 0.03 | 5.0 ± 0.8   | 104.1   | 6037   | −0.04  |
| 8379927   | Arthur   | 1.12 ± 0.03 | 2.2 ± 0.5   | 120.3   | 6067   | −0.10  |

Mass and age values are approximate pipeline averages from Silva Aguirre et al. (2017) Table 4.

## What this enables

Real-star validation: given a star's published {M, age, Z}, evolve our forward model and
compare the predicted oscillation frequencies (Δν, δν₀₂, ratios r₀₂) to the observed values.
This is the test synthetic self-recovery cannot provide — it catches physics gaps in the
forward model.

## Loader

`data/kepler_legacy/loader.py` provides:
- `load_frequencies(kic)` → dict with l=0,1,2,3 arrays of (n, freq, e_freq_lo, e_freq_hi)
- `load_ratios(kic)` → dict of ratio_type → array of (n, ratio, e_lo, e_hi)
- `load_stellar_params(kic)` → dict with pipeline-averaged mass, radius, age, and per-pipeline values
- `load_global_params(kic)` → dict with numax, Dnu, Teff, [Fe/H]
- `TARGETS` — dict mapping KIC to metadata for the selected targets

## Date downloaded

2026-08-23, from CDS archive (cdsarc.cds.unistra.fr). Erratum-corrected tables (2017 ApJ 850, 110).
