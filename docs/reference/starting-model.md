# Starting Model — stellar-jax

## Design choice: constructed ZAMS, no PMS

stellar-jax begins evolution from a **constructed Zero-Age Main Sequence (ZAMS)**
model rather than from pre-main-sequence (PMS) contraction.

**Why not PMS?** Pre-main-sequence contraction requires active deuterium (D) and
lithium (⁷Li) burning networks (D burns at ~10⁶ K, Li at ~3×10⁶ K). Without
these networks, a PMS model would be unphysical: the luminosity during Hayashi
contraction and the ZAMS arrival time would both be wrong (Iben 1965, ApJ 141,
993). These networks are not implemented, so PMS is skipped entirely.

**Validity for MS evolution:** For the goal of main-sequence evolution and
analytic gradients, the PMS omission introduces negligible error:
- D burns in ~0.1 Myr and Li in ~1–3 Myr, leaving the bulk composition
  unchanged to <10⁻⁴ in mass fraction.
- The thermal structure at ZAMS is uniquely determined by (M, X, Y, Z),
  independent of the specific contraction history (Hayashi 1961, PASJ 13, 450).
- MESA's own supplied-ZAMS mode produces main-sequence tracks identical to full
  PMS+ZAMS models (Paxton et al. 2011, ApJS 192, 3, §5).

## ZAMS construction

The ZAMS is the chemically homogeneous model in thermal equilibrium at the onset
of core hydrogen burning: the four stellar structure equations are satisfied for
given (M, X, Y, Z) with L_nuc = L_surface (Kippenhahn, Weigert & Weiss 2012,
§22.1).

**Algorithm (`zams_initial_model` in `structure.py`):**

1. **Initial composition:** uniform X = 1 − Y − Z across all N_COMP = 200 shells,
   with Y = Y_BBN + (ΔY/ΔZ)·Z (primordial helium Y_BBN = 0.2484,
   ΔY/ΔZ = 1.54; Peimbert et al. 2007, ApJ 667, 636; Casagrande et al.
   2007, MNRAS 382, 1516).

2. **Initial guess:** mass-scaled (logL, logTeff) from the ZAMS mass-luminosity
   relation (`initial_guess` in `structure.py`; KWW §22.1).

3. **Newton-Raphson ZAMS solve** (`newton_solve_xprofile`): iterates on
   (logL, logTeff) to satisfy the shooting boundary conditions at the stellar
   centre (P_c, T_c residuals < 10⁻⁶). Typically converges in 20 iterations
   from the initial guess.

4. **Structure on Lagrangian mesh:** once (logL, logTeff) converge, the full
   structure (P, T, ρ, κ, ε_nuc profiles on N_MESH = 600 shells) is computed
   by `shoot_xprofile` and placed on the quadratic-stretched Lagrangian mass mesh
   (`build_model_on_mesh`).

5. **Central boundary extrapolation:** central conditions (P_c, T_c, ρ_c) are
   extrapolated from the innermost shell via iterative Henyey boundary conditions
   (KWW §11.1).

**IFT gradient path:** the Newton solve uses `@custom_vjp` (implicit function
theorem) so `jax.grad` flows through the ZAMS construction to (logL, logTeff)
without unrolling the Newton iterations (Krantz & Parks 2013, §1.2).

## Validation against MESA r26.04.1

ZAMS properties (logL, logTeff, log_rhoc) are validated against MESA r26.04.1
at **1.0, 1.2, 1.5, and 2.0 M☉** (test `test_zams_vs_mesa_f12` in
`tests/test_evolution.py`).

MESA reference config (identical physics, MODE A per project convention):
- Krishna Swamy T(τ) atmosphere
- Böhm-Vitense MLT, α = 2.0
- No diffusion, no overshooting
- pp + CNO network, OPAL opacities
- Z = 0.014, Y = 0.2695

The MESA reference point is the **minimum-radius model** on each track
(the canonical ZAMS; X_c > 0.71).

### Tolerances

| Quantity   | Tolerance |
|------------|-----------|
| |Δlog L|   | < 0.05    |
| |Δlog Teff|| < 0.01    |
| |Δlog ρ_c| | < 0.10    |

### Known limitation: 1 M☉ radius

The shooting solver has a residual radius error at 1 M☉ ZAMS (~2–5% under
MESA). This arises because the superadiabatic layer (SAL) sets the entropy of
the deep convection zone and therefore the stellar radius
(Christensen-Dalsgaard 2008, ApSS 316, 13; Magic et al. 2015, A&A 573, A89).
The Henyey relaxation solver — the production per-step structure solver — eliminates this by solving
all mesh points simultaneously. The logL, logTeff, and log_rhoc are still within
tolerance; only the radius is affected.

## References

- Hayashi, C. 1961, PASJ 13, 450 (Hayashi track; thermal equilibrium uniqueness)
- Iben, I. 1965, ApJ 141, 993 (D and Li burning on PMS)
- Kippenhahn, R., Weigert, A. & Weiss, A. 2012, *Stellar Structure and Evolution* (KWW), §11.1, §22.1
- Paxton, B. et al. 2011, ApJS 192, 3, §5 (supplied-ZAMS validation in MESA)
- Peimbert, M. et al. 2007, ApJ 667, 636 (primordial Y_BBN)
- Christensen-Dalsgaard, J. 2008, Ap&SS 316, 13 (inter-code ZAMS comparison)
- Magic, Z. et al. 2015, A&A 573, A89 (SAL entropy → radius)
