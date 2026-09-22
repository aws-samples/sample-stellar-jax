# Capabilities — Stellar-JAX

What this package delivers, the physics it includes (and deliberately excludes), the starting model,
and who it's for. (The development acceptance-criteria / "done" contract lives with the automation, not
in the public package.)

## What this project delivers

A differentiable 1D stellar evolution code in JAX that:
1. Solves the stellar structure equations with correct physics
2. Produces analytic gradients of observables (L, Teff, R) w.r.t. inputs (M, Z, α, Y)
3. Is validated against MESA and helioseismic data (Model S)

## Physics included

| Feature | Status | Notes |
|---------|--------|-------|
| OPAL EOS | ✓ | 4D interpolation |
| OPAL opacity | ✓ | 4D interpolation (clamps below ~5623 K) |
| Ferguson 2005 low-T opacity | ✓ | H⁻/molecular, `data/ferguson_4d.npz`, blended below logT~4.5 |
| PP + CNO nuclear | ✓ | Caughlan & Fowler rates |
| MLT convection | ✓ | Böhm-Vitense, α threaded as parameter |
| Krishna Swamy T(τ) atmosphere | ✓ | Extended to τ=100 with MLT |
| SAL correction | ✓ | Mass/gravity-dependent, inside atmosphere_bc |
| Per-shell composition | ✓ | 200 zones, local burning |
| Convective mixing | ✓ | Schwarzschild criterion |
| Step overshooting | ✓ | f_ov=0.016, mass-dependent |
| Element diffusion | ✓ | Thoul et al. 1994, He settling |
| Adaptive timestepping | ✓ | varcontrol-based, inside lax.scan |
| IFT gradients | ✓ | Through full evolution via lax.scan |
| Conductive opacity (Potekhin) | ✓ | Degenerate regime (post-MS) |

## Physics NOT included (known limitations)

- No pre-main-sequence (start from supplied ZAMS — see §Starting Model below)
- No deuterium/lithium burning (not needed for supplied-ZAMS approach)
- No rotation
- No mass loss
- No radiation turbulence
- He flash / core flash out of scope

The source of truth for what is validated is the green CI test suite; see the `README.md` validation
section.

## Starting Model

The code adopts a **supplied ZAMS** (Zero-Age Main Sequence) starting model:

- **Definition:** A chemically homogeneous model in thermal equilibrium at the onset of
  core hydrogen burning — the unique stellar structure satisfying the four structure equations for
  given (M, X, Y, Z) with L_nuc = L_surface (Kippenhahn, Weigert & Weiss 2012, §22.1).
- **Approach:** Newton-Raphson iteration solves for (log L, log T_eff) satisfying the shooting
  equations at t=0, from a mass-dependent initial guess. Composition is uniform X = 1−Y−Z across all
  shells, with Y = Y_BBN + (ΔY/ΔZ)·Z.
- **Why not PMS:** Pre-main-sequence contraction requires lithium (⁷Li) and deuterium (D) burning
  networks, which are not implemented (Iben 1965).
- **Validity:** For main-sequence evolution the ZAMS starting point introduces negligible error; the
  thermal structure at ZAMS is uniquely determined by (M, X, Y, Z) independent of contraction history
  (Hayashi 1961). MESA's own supplied-ZAMS mode produces identical MS tracks to full PMS models
  (Paxton et al. 2011, §5).
- **Validation:** ZAMS properties (L, T_eff, T_c, ρ_c) validated against MESA r26.04.1 at 1.0, 1.2, 1.5,
  and 2.0 M☉ (see `test_zams_vs_mesa_f12` in `tests/test_evolution.py`).

## Target audience

Asteroseismologists, stellar modelers, and ML researchers needing differentiable stellar models. The
code is intended to be publishable as a methods paper.
