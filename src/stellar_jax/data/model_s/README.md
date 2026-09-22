# Model S Data

Reference solar model from Christensen-Dalsgaard et al. (1996), Science 272, 1286.

## Files

- `fgong.l5bi.d.15c` — FGONG format (standard stellar model interchange), 2482 mesh
  points, md5 `8a076ec1810e9871cdf21f6ecdf72016`.
- `model_s_cs.dat` — Sound speed profile (r/R, c_s in cm/s), 2482 points.
- `model_s_rho.dat` — Density profile (r/R, ρ in g/cm³), 2482 points.
- `model_s_adipls_freqs_l0123.dat` — **Primary ADIPLS frequency reference (l=0-3).**
  Genuine, reproducible run from ADIPLS bundled with MESA 26.04.1. 84 modes: 21 per
  degree, l=0,1,2,3. Supersedes the l=0-2-only `mesa2604` table for any use that
  needs l=3 (e.g. the l=3 échelle overlay). Recipe below.
- `model_s_adipls_freqs_mesa2604.dat` — **ADIPLS frequency reference (l=0-2 only).**
  Same ADIPLS version, model, and physics as the l=0-3 table above (the l=0-2 modes
  are bit-identical between the two files). Retained for backward compatibility —
  tests that only need l=0-2 can use either file.
- `adipls_kernels/` — ADIPLS Γ₁ structure kernels + the ADIPLS control file from
  the same genuine run. See `adipls_kernels/README.md`.

## Source

Downloaded from: https://users-phys.au.dk/~jcd/solar_models/

## What it contains

2482 mesh points with 15 global parameters and 25 variables per point:
- r/R, ln(m/M), T, P, ρ, X, L_r, κ, ε_nuc, Γ₁, ∇_ad, etc.

## Physics

Model S includes:
- OPAL EOS + opacity
- Nuclear burning (pp + CNO + screening)
- Helium + heavy element diffusion
- MLT convection (α calibrated to Sun)
- Age: 4.6 Gyr, Z=0.0196, Y_init=0.2713

Our code includes element diffusion (Thoul/Burgers coefficients), so dc_s/c_s
differences relative to Model S arise primarily from remaining resolution and
calibration differences.

---

## ADIPLS frequency generation recipe

The ADIPLS adiabatic eigenfrequency tables committed here are the **independent
external reference** our oscillation solver is validated against (the "ADIPLS"
side of the échelle diagram and the per-mode frequency accuracy test).

### Provenance

| | |
|---|---|
| Code | **ADIPLS** (the Aarhus adiabatic pulsation package, bundled with MESA 26.04.1) |
| Equations | Full adiabatic (non-Cowling, `icow=0`); **variational** frequencies (`mdintg=5`, `iriche=0`); surface BC `istsbc=1` |
| Constant | G = 6.67430×10⁻⁸ cgs (== repo `config/constants.py`; MESA `standard_cgrav`) |
| Model | `fgong.l5bi.d.15c` (this directory; md5 `8a076ec1810e9871cdf21f6ecdf72016`) → `fgong-amdl.d` → `model_s.amdl` (2482 mesh points, no redistribution) |
| Control | `adipls_kernels/adipls_ms.in` (committed) |
| Degrees | l=0-3 (set `nsel=4` in the control file's `osc:` block; the committed file has `nsel=3` for l=0-2) |

### How to reproduce

On any host with gfortran + MESA source (the ADIPLS Fortran code in
`$MESA_DIR/adipls/adipack.c/`):

```bash
# 1. Compile ADIPLS from the MESA source tree.
#    ADIPLS is standard Fortran 77 in adipack.c/{adipls,gensr}/.
#    With a built MESA install, the wrapper scripts at adipack.c/bin/
#    call the compiled .x binaries; set aprgdir accordingly:
export aprgdir=$MESA_DIR/adipls/adipack.c

# 2. Convert Model S FGONG to ADIPLS binary format.
$aprgdir/bin/fgong-amdl.d  fgong.l5bi.d.15c  model_s.amdl

# 3. Copy and adjust the committed control file for l=0-3, no kernels.
cp adipls_kernels/adipls_ms.in adipls_l03.in
#    In the osc: block, change  el,nsel = 0,3  ->  0,4  (4 degrees: l=0,1,2,3).
#    In the out: block, change  igm1kr = 1  ->  0  (skip kernel output, faster).
sed -e 's/0,3,0,1,/0,4,0,1,/' \
    -e 's/0,,1,,,,,,,,/0,,0,,,,,,,,/' adipls_kernels/adipls_ms.in > adipls_l03.in

# 4. Run ADIPLS (takes ~10s for 84 modes).
$aprgdir/bin/adipls.c.d adipls_l03.in
#    -> ms_agsm.dat (grand summary, binary)

# 5. Extract frequencies to ASCII.
$aprgdir/bin/set-obs.d 1 ms_agsm.dat obs_ms_l03.txt
#    -> obs_ms_l03.txt: columns  l  n  nu_uHz

# 6. Verify: the l=0-2 modes must be bit-identical to the committed
#    model_s_adipls_freqs_mesa2604.dat.  Spot-check:
#    l=0 n=8 = 1263.6908 µHz,  l=0 n=9 = 1407.8177 µHz.
```

### Control file knobs (the two settings that matter)

The committed `adipls_ms.in` has:

- **`el,nsel,els1,dels = 0,3,0,1`** in the `osc:` block — this means `nsel`
  degrees starting at `els1`, stepping by `dels`. So `0,3,0,1` = l=0,1,2 (3
  degrees). For l=0-3, change to **`0,4,0,1`**.
- **`igm1kr = 1`** in the `out:` block — writes the Γ₁ structure kernel per
  mode (slow, ~minutes). For frequencies only, set to **`0`**.

### Verification (mandatory before committing a new table)

The l=0-2 overlap with the existing `model_s_adipls_freqs_mesa2604.dat` must
be **bit-identical**. If any frequency differs, the run used different physics
or a different model — do not commit.

Reference values for spot-checking: Δν(l=0, n≈17-25) ≈ 136 µHz. Our own
forward oscillation model agrees to ~5 significant figures (l=0 n=20: ours
2902.63 vs ADIPLS 2902.68 µHz).

---

## Usage (sound speed comparison)

```python
import numpy as np

# Parse FGONG (skip 4 header lines + 1 global params line)
data = np.loadtxt('data/model_s/fgong.l5bi.d.15c', skiprows=5)
# Reshape: 2482 points × 25 variables
model = data.reshape(2482, 25)

# Key columns (0-indexed):
# 0: r/R
# 1: ln(m/M)
# 2: T (K)
# 3: P (dyn/cm²)
# 4: ρ (g/cm³)
# 5: X (hydrogen mass fraction)
# 9: Γ₁ (first adiabatic exponent)
# 14: ∇_ad

# Sound speed:
r = model[:, 0]  # fractional radius
P = model[:, 3]
rho = model[:, 4]
gamma1 = model[:, 9]
c_s = np.sqrt(gamma1 * P / rho)  # cm/s
```
