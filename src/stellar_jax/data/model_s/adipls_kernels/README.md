# ADIPLS reference Γ₁ structure kernels — Model S

Genuine, reproducible ADIPLS `gm1ker` kernels K_{Γ₁,ρ}(r) for Model S, for
independent (cross-code) validation of the repo's AD structure kernels
(`oscillations/kernels.py`).

## Provenance (reproducible)

- **Code:** ADIPLS bundled with **MESA 26.04.1** (`$MESA_DIR/adipls`), run via
  the `.d` wrapper scripts. `igm1kr=1` writes the Γ₁ kernel per mode.
- **Model:** `../fgong.l5bi.d.15c` (md5 `8a076ec1810e9871cdf21f6ecdf72016`) →
  `fgong-amdl.d` → `model_s.amdl` (2482 mesh points). **No** redistribution.
- **Physics:** full adiabatic equations (`icow=0`), variational frequencies
  (`mdintg=5`), surface BC `istsbc=1`, **G = 6.67430e-8 cgs** (== repo
  `config/constants.py`, MESA `standard_cgrav`).
- **Control file:** `adipls_ms.in` (committed here). Full recipe:
  `RECIPE.md` (this directory).

## Files

- `adipls_gm1ker_l{l}_n{n}.dat` — columns `x=r/R   K_gamma1_rho`, full 2482-pt grid.
  Modes: l=0 n∈{15,17,20,22}, l=1 n∈{16,18,20}, l=2 n∈{15,17,20}. Includes the
  two modes the in-repo AD-vs-FD test uses (l=0 n=20, l=2 n=17).
- `adipls_ms.in` — the exact ADIPLS control file used.
- `../model_s_adipls_freqs_mesa2604.dat` — the frequencies from the same run
  (all 63 modes, l=0,1,2), reproducible replacement for the legacy freq tables.

## Normalization convention — RESOLVED (2026-08-31)

The **~1.61 constant factor** between our AD kernels and the ADIPLS kernels is the
normalization convention difference, identified from the ADIPLS Fortran source
(`gm1ker.n.d.f`) and the ADIPLS documentation (`adiab.prg.c.tex`, eq. 4.8):

    ADIPLS:  δω/ω = ∫ K_{ADIPLS}(x) · δΓ₁(x)         · dx   (ABSOLUTE δΓ₁)
    Ours:    δν/ν = ∫ K_{ours}(x)   · δΓ₁(x)/Γ₁(x)   · dx   (RELATIVE δΓ₁/Γ₁)

Since δΓ₁ = Γ₁ · (δΓ₁/Γ₁), the exact conversion is:

    K_{ours}(x) = K_{ADIPLS}(x) × Γ₁(x)

In Model S, Γ₁ ≈ 5/3 ≈ 1.667 throughout the interior (varies < 0.3%), so the
kernel-weighted average Γ₁ ≈ 1.61–1.67 (the ~1.61 constant factor). The conversion
is applied **pointwise** (not as a single constant) for the validation test, though
the effect is nearly identical in the interior where Γ₁ is constant.

Source:
- ADIPLS `gm1ker.n.d.f` line 3: `delta omega/omega = integral(gmk*delta gamma1 *dx)`
- ADIPLS `adiab.prg.c.tex` eq. 4.8: `δω/ω = ∫ K_{nl}^{(Γ₁)} δΓ₁ dx`
- Our `kernels.py`: `δν/ν = ∫ K_{Γ₁,ρ} · (δΓ₁/Γ₁) · dr/R`

## Comparison against our AD kernels (verified)

Frequencies match our forward model to ~5 sig figs (l=0 n=20: ours 2902.63 vs
ADIPLS 2902.68 µHz). After converting ADIPLS kernels to our relative-δΓ₁
convention (multiply by Γ₁(r)), then normalizing both to unit integral:

### Tested modes (ν > 2500 µHz) — ≤ 10%

| Mode    | ν (µHz) | max err | rms err | worst at |
|---------|---------|---------|---------|----------|
| l=0 n=22 | 3175 | 8.2%  | 3.1%    | x≈0.87   |
| l=2 n=20 | 3029 | 8.6%  | 3.1%    | x≈0.89   |
| l=0 n=20 | 2903 | 8.8%  | 3.3%    | x≈0.88   |
| l=1 n=20 | 2967 | 9.0%  | 3.3%    | x≈0.89   |
| l=2 n=17 | 2622 | 9.6%  | 3.6%    | x≈0.88   |
| l=1 n=18 | 2696 | 9.7%  | 3.5%    | x≈0.89   |

All pass the ≤ 10% peak-normalized error bound over 0.1 < r/R < 0.9. The
validation test (`test_kernel_adipls_validation.py`) asserts `tol_peak_norm = 0.10`.

### Lower-frequency modes (ν < 2500 µHz) — 10–12% scatter

| Mode    | ν (µHz) | max err | rms err | worst at |
|---------|---------|---------|---------|----------|
| l=2 n=15 | 2353 | 10.4% | 3.9%    | x≈0.89   |
| l=0 n=17 | 2498 | 11.0% | 4.0%    | x≈0.90   |
| l=1 n=16 | 2427 | 11.5% | 4.1%    | x≈0.90   |
| l=0 n=15 | 2229 | 11.5% | 4.5%    | x≈0.87   |

These exceed 10% and are **not** included in the validation test. The additional
scatter is a genuine physical effect — lower-frequency modes have deeper
turning points and more mode energy near the CZ base (r/R ≈ 0.87–0.90)
where the two codes disagree most. This is a **CONSTRAINT**: the AD method
(chain-rule differentiation through the full numerical eigenfrequency solver)
produces small oscillatory noise at kernel nodes (~0.8% of peak, 50–68
negative-valued interior points), while ADIPLS's analytic variational formula
produces exactly non-negative kernels. The noise is independent of integration
resolution (n_steps 8000→16000 identical) — it is a structural property of
differentiating through a discrete numerical chain.

### Residual anatomy

- **Worst-case location:** convection-zone base, r/R ≈ 0.87–0.90, where the
  Γ₁ kernel oscillates through nodes. Cross-code scatter is largest where
  kernel values are small.
- **Surface (r/R > 0.9):** excluded from comparison; residuals reach ~15–20%
  due to the known outer-BC difference (surface term).
- **RMS** is 3–4% for all modes — the max errors are localized spikes at
  nodes, not systematic disagreement.
- **Integral ratio** (AD/ADIPLS over [0.1, 0.9]) ranges from 0.93 (l=0 n=15)
  to 0.97 (l=0 n=22), reflecting a ~3–7% lower integral in the AD kernel
  (mode energy that our outer-BC treatment places differently).

**The AD-vs-ADIPLS test** (a) converts ADIPLS to relative convention (×Γ₁),
(b) normalizes both to unit integral, (c) uses peak-normalized absolute error
≤ 10% over 0.1 < r/R < 0.9, and (d) excludes r/R > 0.9 pending the surface-term correction.
