# ADIPLS Reference Kernel Generation — Recipe

How the committed ADIPLS analytic reference eigenfrequencies + Γ₁ kernels under this directory
(`data/model_s/adipls_kernels/`, `data/model_s/model_s_adipls_freqs_mesa2604.dat`) were generated, so
they can be reproduced/verified. They are the ground truth for the oscillation-kernel validation.

**Toolchain:** MESA **r26.04.1** (which bundles ADIPLS) + its SDK. Set `MESA_DIR` to the MESA install
root; ADIPLS lives at `$MESA_DIR/adipls/adipack.c`.

---

## Verified working recipe (2026-08-31)

Two easy mistakes to avoid: (1) the adipack programs are driven by the **`.d` wrapper scripts with the
control file as an ARGUMENT** (`adipls.c.d ctl.in`), NOT `adipls.c.d.x < ctl.in` (raw stdin only prints
the usage prompt); (2) no separate `gm1ker` compile is needed — set `igm1kr=1` and `adipls.c.d` writes
the Γ₁ kernel to unit `idsgkr` (default 13) directly. Redistribution is not needed (Model S amdl is
already 2482 points).

```bash
export aprgdir="$MESA_DIR/adipls/adipack.c"
export PATH="$aprgdir/bin:$PATH"
mkdir -p kernel_ref && cd kernel_ref
# Model S FGONG == the committed data/model_s/fgong.l5bi.d.15c (md5 8a076ec1...)
cp "$MESA_DIR/gyre/gyre/models/fgong/fgong.l5bi.d.15c" model_s.fgong
fgong-amdl.d model_s.fgong model_s.amdl          # -> model_s.amdl (2482 pts)
adipls.c.d adipls_ms.in                           # control file committed here as adipls_ms.in
# Outputs: ms_agsm.dat (grand summary / freqs), ms_gm1ker.dat (Γ1 kernels), ms.prt (ASCII log)
set-obs.d 1 ms_agsm.dat obs_ms.txt                # freqs -> ASCII (l n nu_uHz)
```

Key control-file (`adipls_ms.in`) settings: file-header assigns unit 2=model, 11=agsm,
13=`ms_gm1ker.dat` (kernel), **4=`'0'` (route to /dev/null — REQUIRED or `openfc` STOPs)**;
`osc` `el,nsel = 0,3` (l=0,1,2); `cst cgrav = 6.67430e-8` (our repo G); `int istsbc=1, mdintg=5,
icow=0`; `out igm1kr=1`, **`nfmode` BLANK** (setting `nfmode=1` needs a unit-4 eigenfunction file or it
crashes).

### Parse `ms_gm1ker.dat` (Fortran unformatted, one record per mode)

Each record = `cs(1:50)` [float64] + `nnw` [int32] + interleaved `(x, K)` [`2·nnw` float64]. Mode
identity is in `cs`: `cs[17]=l`, `cs[18]=order n`, `cs[26]=ν in mHz`. The committed kernel subset lives
at `adipls_gm1ker_l{l}_n{n}.dat` (cols `x  K_gamma1_rho`).

### Normalization / comparison finding (verified 2026-08-31)

Our AD kernel and the ADIPLS `gm1ker` differ by a **constant factor ≈ 1.61** (identical for l=0 and
l=2 — a normalization convention, not a shape error; shape correlation 0.988). After removing it,
agreement is ~5–6 % over 0.1<r/R<0.9, growing to ~18 % only at r/R>0.9 (the surface-BC gap). A valid
AD-vs-ADIPLS test compares **normalized** kernels with a ~5–8 % bulk tolerance and excludes the
surface. See `README.md` in this directory.

## Method note

The ADIPLS variational kernel and our AD kernel are computed by fundamentally different methods:
ADIPLS uses an analytic formula from the eigenfunction (`gm1ker.n.d.f`), while we use `jax.grad`
through the implicit function theorem. Integral agreement to ~5% is expected; pointwise differences of
2–10× occur in the deep interior (where ADIPLS's acoustic-limit approximation diverges from the full
physics captured by our AD). The ADIPLS kernel formula (from `$MESA_DIR/adipls/adipack.c/adipls/gm1ker.n.d.f`):

```
K_{Γ₁}(r) = x² · q · u · A₁ · (x · d₁)² / (2σ² · ∫wrk₂)
```

## References

- Christensen-Dalsgaard (2008), Ap&SS 316, 113 (ADIPLS documentation)
- Christensen-Dalsgaard, Lecture Notes on Stellar Oscillations, Ch. 5 (variational kernel derivation)
- Gough & Thompson (1991), Solar Interior and Atmosphere (the original kernel formalism)
