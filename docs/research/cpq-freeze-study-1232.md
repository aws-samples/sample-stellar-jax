# Research: Frozen cp/Q Approximation in the Replay Adjoint (#1232)

## Summary

The frozen-cp/Q approximation in the Henyey IFT adjoint introduces a **forward
physics error** (up to 31% in nabla at the He I ionization zone of a 1.5 M☉
subgiant) but does **NOT** contribute to the AD-vs-FD gradient discrepancy.
The IFT adjoint is self-consistent: the same ideal-gas cp/Q is used in the
forward residual, Jacobian, and VJP. The freeze should be kept as a documented
CONSTRAINT.

## Background

The `` replay adjoint freezes EOS cp and Q = χ_T/χ_ρ in the MLT nabla
computation on the differentiable lax.scan path. Specifically:

- `residual.py::_cell_residual` (on_adaptive_path=False): passes cp=None, Q=None
  to `mlt_nabla` → ideal-gas defaults: cp = k_B/(μ·m_H·∇_ad), Q = 1.0
- `jacobian.py::_jacobian_blocks_fixed_bc`: uses `_cell_residual_raw` with the
  same ideal-gas defaults
- `adjoint.py::_vjp_residual_params`: VJP through `_build_residual_fixed_bc`
  which also uses ideal-gas cp/Q

This is forced by memory: `jax.jacfwd` through the 4D EOS to get live ∂cp/∂y
and ∂Q/∂y would OOM (residual.py:327–333).

**MESA comparison:** MESA's `turb/private/mlt.f90:61` uses
`auto_diff_real_star_order1` for chiT, chiRho, Cp — carrying full AD partials
through the Newton Jacobian. Our ideal-gas approximation is a deliberate
divergence labeled CONSTRAINT (GP-9: IFT_CONSISTENCY).

## Results

### AC1: Per-zone cp/Q departure (1.5 M☉ SGB, MESA FGONG)

| Metric | All zones | Convective zones |
|--------|-----------|-----------------|
| N zones | 1386 | 538 (39%) |
| max cp departure from ideal gas | 68.7% | 57.7% |
| max Q departure from ideal gas | 182.9% | 57.4% |
| max nabla departure (real vs ideal cp/Q) | — | **31.17%** |
| median nabla departure | — | **0.011%** |

**Worst zones:** T ≈ 11000 K (He I partial ionization), r/R ≈ 0.998 (near
surface). At these conditions, cp_real ≈ 1.2×10⁹ erg/g/K vs cp_ideal ≈ 6.4×10⁸,
Q_real ≈ 1.84 vs Q_ideal = 1.0. The 31% nabla error is concentrated in a thin
shell (~5 zones) near the surface.

**Median nabla departure is only 0.011%** — the freeze is negligible for the
vast majority of convective zones. The large departure is confined to the He I
ionization zone where radiation pressure and partial ionization make the ideal-gas
approximation poorest.

### AC2: Is the freeze the limiting error term?

**No.** The frozen cp/Q does NOT contribute to the AD-vs-FD gradient discrepancy.

The key insight is that on the differentiable lax.scan path, the forward
residual, the Jacobian, and the VJP **all** use the same ideal-gas cp/Q. The
central-difference FD reference also evaluates the same forward model (with
ideal-gas cp/Q). Therefore:

1. The IFT adjoint computes `J⁻¹ · ∂R/∂θ` where both J and ∂R/∂θ use
   ideal-gas cp/Q → **self-consistent**
2. The FD reference computes `[f(θ+δ) - f(θ-δ)] / 2δ` where f uses the
   same ideal-gas cp/Q forward → **same mathematical function**
3. AD and FD should agree on the derivative of this (approximate) function

The observed AD-vs-FD discrepancy comes from **other** stop_gradient terms:
- **GP-5** (shell_data composition detachment): ~0.56% at N=200 schedule replay
- **GP-4** (X3 equilibrium stop_gradient): ~2.36% at N=3 fixed_dt

The frozen cp/Q contributes **0%** to the AD-vs-FD error by construction.

### Existing CI measurements (for reference)

| Config | AD-vs-FD rel_err | Source |
|--------|-------------------|--------|
| 1.0 M☉, N=3, fixed_dt | 2.36% | test_replay_gradient_dsigma2_dM |
| 1.0 M☉, N=200, schedule replay | 0.56% | test_seismic_gradient_adaptive_frozen_schedule |

### AC3: Recommendation

**Keep the freeze as a documented CONSTRAINT.**

Rationale:

1. **Gradient accuracy is unaffected.** The freeze is IFT-self-consistent: the
   gradient is exact for the (approximate) forward model. There is no gradient
   accuracy to be gained by unfreezing cp/Q.

2. **Forward physics impact is localized.** The 31% nabla error is confined to
   ~5 zones in the He I ionization zone (T ≈ 11,000 K, r/R > 0.997). The median
   error across all 538 CZ zones is 0.011%. For main-sequence stars where the
   differentiable path (lax.scan) operates, the ionization zone is even thinner
   relative to the star.

3. **Memory constraint is real.** Unfreezing requires `jax.jacfwd` through the
   4D EOS interpolation (4 nested Catmull-Rom evaluations) to get ∂cp/∂y and
   ∂Q/∂y — this OOMs. A custom_jvp for cp/Q with a low-rank sensitivity
   approximation is feasible but complex, and the benefit is purely in forward
   physics (not gradient accuracy).

4. **Gradient improvement priority is elsewhere.** The composition adjoint
 is the limiting factor for AD-vs-FD ∂σ²/∂M accuracy. The GP-5
   shell_data composition detachment dominates the gradient error budget.
   Addressing that will reduce the ~0.56% (N=200) to near-zero, without
   touching cp/Q.

5. **The forward physics error is bounded by other approximations.** The 200-zone
   Lagrangian mesh, fixed composition grid (N_COMP=50), and no dynamic remeshing
   each introduce comparable or larger forward errors than the cp/Q ideal-gas
   approximation.

## Reproducing

```bash
# Part 1 only (fast, ~30 sec):
python tests/_studies/cpq_freeze_study.py

# Full study including end-to-end AD vs FD (requires ~30 min JIT):
python tests/_studies/cpq_freeze_study.py --with-part2
```

## References

- MESA `turb/private/mlt.f90:61,76,114` — auto_diff cp, Q in MLT
- Kippenhahn, Weigert & Weiss (2012), §14.1 (MLT cubic)
- Cox & Giuli (1968), §14.24 (Q = χ_T/χ_ρ)
- Griewank & Walther (2008), §15 (IFT for implicit solvers)
- `residual.py:327–333` (GP-9: IFT_CONSISTENCY comment)
