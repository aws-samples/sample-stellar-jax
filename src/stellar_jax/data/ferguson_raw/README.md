# Ferguson et al. (2005) low-temperature opacity tables

Source: MESA r26.04.1 `$MESA_DIR/data/kap_data/lowT_fa05_gs98_z*_x*.data`.

These are the EXACT low-T opacity tables MESA used to compute the reference
tracks in `data/mesa_comparison/` (MESA default `kap_lowT_prefix=lowT_fa05_gs98`
when `kap_file_prefix=gs98`). Using them guarantees consistency with our
validation reference.

## Format (per file)
- Header: form, version, X, Z, logRs, logR_min/max, logTs, logT_min/max
- Grid: logT 2.700–4.500 (105 pts), logR -8.0..1.0 (46 pts)
- logR = logRho - 3*logT + 18 (SAME convention as our opal_4d.npz)
- Values: log10(Rosseland mean kappa) in cm^2/g, INCLUDING H-/molecular/grains

## Coverage
- X: 0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98
- Z: 0.0, 0.0001, 0.0003, 0.001, 0.002, 0.004, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1
  (matches our OPAL Z grid)

## Usage
Parse these into `data/ferguson_4d.npz` with axes (X, Z, logT, logR) to match
opal_4d.npz, for blending below logT~4.5.
