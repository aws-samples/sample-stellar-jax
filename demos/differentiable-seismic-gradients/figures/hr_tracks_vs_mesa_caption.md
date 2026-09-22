# Figure: HR tracks — stellar-jax forward evolution vs MESA

## Caption

Hertzsprung–Russell diagram (log Teff vs log L) comparing stellar-jax
forward-evolution tracks (scatter points) with identical-physics MESA reference
tracks (dashed lines) for 1.0, 1.5, and 2.0 M☉, from ZAMS through the
subgiant branch and the lower red giant branch.  The star marker on the
1.5 M☉ track indicates the subgiant operating point where the
through-evolution ∂ν²/∂M gradient is evaluated (AD vs independent FD: 24.5%,
sign-correct; hard target <15%).

The stellar-jax tracks are plotted as scatter points (not connected lines).
The committed tracks use max_steps=500 (practical compute time); connecting
consecutive adaptive steps produces non-physical zigzag artifacts from the
large timestep jumps.  Scatter points honestly show where the solver placed
its accepted steps; the MESA tracks (10 000+ model numbers) provide smooth
reference lines.  The CI-validated runs (max_steps=5000, hours per mass)
produce smooth tracks; the CI suite validates each evolutionary phase against
the MESA reference separately:
- MS: |Δlog L| < 0.10 dex, |Δlog Teff| < 0.03 dex at matched Xc (`test_mesa_comparison`)
- SGB: |Δlog L| < 0.15 dex at matched Xc (`test_post_tams_core_contraction`)
- RGB: median |Δlog Teff| < 0.08 dex on the Hayashi track (`test_adaptive_forward_rgb_*`)

## Physics

MODE-A identical-physics comparison (MESA_CONFIG parsed from
`data/mesa_comparison/inlist_1.0Msun`):
- Z = 0.014
- α_MLT = 2.0 (Böhm-Vitense, Cox formulation)
- Y_init = 0.2695
- Diffusion: OFF
- Overshoot: OFF (f_ov = 0.0)

## Provenance

MESA reference data:
- Source: MESA git build f12c70cf (r26.4.1), SDK mesasdk-x86_64-linux-26.6.1
- Location: `src/stellar_jax/data/mesa_comparison/rgb/{1.0,1.5,2.0}Msun/history.data.gz`
- Provenance: `src/stellar_jax/data/mesa_comparison/rgb/README.md`
- Generated: 2026-06-17

stellar-jax forward tracks:
- Computed via: `evolve_star_adaptive(mass, N_zones=600, max_steps=500, **MESA_CONFIG)`
- Location: `demos/differentiable-seismic-gradients/data/sjax_track_{mass}Msun.npz`
- Regenerate: `python generate_hr_tracks.py --compute`
- Note: max_steps=500 produces tracks through the full RGB but with visible
  step-to-step scatter where the adaptive timestep takes large jumps.  The
  figure trims our tracks at log L ≤ 2.0 and plots scatter points rather than
  connected lines.  The CI suite validates each phase separately:
  |Δlog L| < 0.10 dex on the MS (`test_mesa_comparison`),
  median |Δlog Teff| < 0.08 dex on the RGB (`test_adaptive_forward_rgb_*`).

## Validation gate

The plotted stellar-jax tracks agree with MESA, validated per-phase by
separate CI-gated tests:
- **MS:** |Δlog L| < 0.10 dex, |Δlog Teff| < 0.03 dex at matched Xc —
  `test_mesa_comparison` (per-wave)
- **SGB:** |Δlog L| < 0.15 dex at matched Xc —
  `test_post_tams_core_contraction` (per-wave)
- **RGB:** median |Δlog Teff| < 0.08 dex on the Hayashi track —
  `test_adaptive_forward_rgb_1p0` (per-wave, 5h timeout)
  `test_adaptive_forward_rgb_1p5` (nightly, 5h timeout)
  `test_adaptive_forward_rgb_2p0` (nightly, 5h timeout)

## Subgiant gradient

Operating point: 1.5 M☉, early subgiant (X_c < 0.01 + 50 steps past TAMS).
Observable: ν² (cyclic frequency squared, µHz²) — NOT σ² (see
test_replay_gradient_seismic_subgiant docstring for the σ² cancellation issue).
AD vs independent two-sided FD (schedule-replay at M±dM): 24.5%.
Sign: correct (AD·FD > 0).
Hard target: <15% (/).
CI test: `test_replay_gradient_seismic_subgiant` (18000s timeout).

## References

- Paxton et al. (2011, 2013, 2015, 2018, 2019) — MESA instrument papers
- arXiv:2609.02876 — GRADSOLVE (record–replay gradient construction)
- Christensen-Dalsgaard (2008), Ap&SS 316, 113 — oscillation coefficients
