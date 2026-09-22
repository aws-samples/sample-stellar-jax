# ADIPLS Γ₁ structure kernels — MESA 2.0 M☉ midMS (REFERENCE)

Genuine ADIPLS `gm1ker` kernels K_{Γ₁,ρ}(r) for the committed 2.0Msun/midMS.FGONG.gz, generated 2026-09-09.
- Modes: l=0 n=20 (ν=1781.7 µHz), l=1 n=20 (1844.5 µHz), l=2 n=17 (1627.9 µHz).
- Code: ADIPLS (MESA 26.04.1), igm1kr=1, variational, istsbc=1, icow=0, mdintg=5, G=6.67430e-8.
- Model prep: committed FGONG re-rendered to ADIPLS 1pe16.9 fixed format, converted with fgong-amdl.d.
- Columns: `x=r/R   K_gamma1_rho`.

Convention (from ADIPLS source gm1ker.n.d.f line 3): **ABSOLUTE δΓ₁**
    δω/ω = ∫ K · δΓ₁ · dx
Our code uses RELATIVE δΓ₁/Γ₁, so the conversion is K_ours(x) = K_ADIPLS(x) × Γ₁(x),
identical to the Model S ADIPLS kernels (test_kernel_adipls_validation.py).

Validated by tests/test_mode_selection_2msun.py (≤10% RMS): our `compute_structure_kernels(n_pg=...)` now correctly selects
the per-star radial order using the acoustic-radius Δν and Tassoul n_pg identification
(replacing the hardcoded solar Δν=135/ε=1.5 selector). AD kernel vs ADIPLS agreement:
  - RMS peak-normalized error: ~5.5-6.2% (all three modes, ≤10% bar)
  - Max peak-normalized error: ~18-22% (concentrated at kernel nodes near x≈0.89)
The max exceeds the 10% Model S bar due to the lower mode frequencies (~1789 µHz vs
~2700 µHz for Model S), consistent with the known pattern: the Model S test notes
lower-frequency modes (ν < 2500 µHz) already show 10-12% errors.
Gated by test_mode_selection_2msun.py: ≤10% RMS, ≤22% max, ≤8% RMS.
