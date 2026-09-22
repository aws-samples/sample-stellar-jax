# Architecture

This document explains **what stellar-jax is, how the code is organized, and how it works** — the
big-picture map for anyone reading or extending the package. It is the *explanation/orientation*
companion to the factual [`reference/`](reference/) pages (which state what is validated and what to
trust). Line references (`file:line`) are anchors to help you jump into the source; they drift as the
code changes — treat them as a starting point, not a contract.

## Contents

1. [Purpose & design philosophy](#1-purpose--design-philosophy)
2. [Bird's-eye view](#2-birds-eye-view)
3. [Package map](#3-package-map)
4. [The forward evolution loop](#4-the-forward-evolution-loop)
5. [The structure solver](#5-the-structure-solver)
6. [Differentiability architecture](#6-differentiability-architecture)
7. [Microphysics](#7-microphysics)
8. [Mesh, transport & composition](#8-mesh-transport--composition)
9. [Oscillations (asteroseismology)](#9-oscillations-asteroseismology)
10. [FGONG bridge](-fgong-bridge)
11. [Inference & calibration](-inference--calibration)
12. [Configuration](-configuration)
13. [Cross-cutting concerns](-cross-cutting-concerns)
14. [Extending the code](-extending-the-code)

---

## 1. Purpose & design philosophy

stellar-jax evolves a 1D stellar model from the zero-age main sequence (ZAMS) through the main
sequence (and, on a dedicated path, onto the red-giant branch), and computes its adiabatic
oscillation frequencies. It is written entirely in [JAX](https://github.com/google/jax).

**The reason it exists is differentiability.** The whole pipeline —
`stellar parameters → evolution → structure → observables → oscillation frequencies` — is one
differentiable function. You can call `jax.grad` through it to get *analytic* derivatives of an
observable (luminosity, effective temperature, an oscillation frequency) with respect to an input
(mass, metallicity, mixing-length, helium, and physics-calibration knobs). Traditional stellar codes
give you the forward model only; getting gradients means finite-differencing the whole pipeline (slow,
noisy, resolution-limited) or training a separate surrogate. Here the gradient is exact and comes for
free from the same code that does the forward evolution. That is what enables gradient-based parameter
inference and optimization *through the physics* rather than around it.

Design principles that recur throughout the code:

- **One code path.** There is a single production solver of record. Parallel/legacy solvers exist for
  bootstrapping and validation but are deliberately *not* exposed as alternative public entry points,
  so callers cannot accidentally mix physics.
- **Gradients via the implicit function theorem (IFT), not by unrolling.** Every internal iterative
  solve (Newton structure solve, density inversion, oscillation eigenvalue) is wrapped as a custom
  differentiation boundary: the backward pass is a single linear solve at the converged solution, not
  backprop through the iterations. This is exact, cheap, and independent of iteration count.
- **Explicit differentiability boundaries.** Which quantities carry gradients and which are detached
  is a documented policy (§6), not an accident. Detachments are algorithmic choices (grid placement,
  step-size control) that are not physical derivatives.
- **Pure, leaf physics.** The microphysics modules are pure functions with no knowledge of the solver
  or evolution loop; the gradient policy lives in the orchestrator, not the leaves.

**Out of scope by design** (see [`reference/capabilities.md`](reference/capabilities.md)): the helium
flash, pre-main-sequence contraction, rotation, and mass loss.

---

## 2. Bird's-eye view

```
                    stellar parameters θ = (mass, Z, α_mlt, f_ov, Y_init, physics knobs)
                                              │
                                              ▼
                              ┌───────────────────────────────┐
                              │  evolve_star  (evolution/)     │   the public entry point
                              │                                │
   ZAMS initial model ───────►│  for each timestep (lax.scan): │
   (structure.py shooting)    │    1. structure solve  ────────┼──►  solver/  (Henyey Newton
                              │    2. composition update       │          + IFT adjoint)
                              │       (composition/)           │              │ uses
                              │    3. mesh adapt (mesh/)        │              ▼
                              │    4. adaptive timestep         │        microphysics/
                              │       (evolution/timestep.py)   │   (EOS, opacity, nuclear,
                              └───────────────────────────────┘    neutrino, MLT via transport/)
                                              │
                                              ▼
                          observables: log L, log Teff, log R, age, composition
                                              │
                                              ▼   (fgong/ builds a differentiable structure record)
                              ┌───────────────────────────────┐
                              │  oscillations/                 │   frequencies ν(θ)
                              │  coeffs → RK4 integrate →       │
                              │  determinant → root → IFT       │
                              └───────────────────────────────┘

   The ENTIRE chain is one differentiable function:  jax.grad(observable)(θ)  →  analytic ∂observable/∂θ
```

Two consumers sit on top of this forward model:

- **inference/** — gradient-based fitting of θ to observed data (the inverse problem).
- **calibration/** — solar calibration and comparison against reference models.

---

## 3. Package map

All source lives under `src/stellar_jax/`. Responsibilities:

| Package / module | Role |
|---|---|
| `stellar.py` | Thin public facade. Re-exports the public API (`evolve_star`, constants, microphysics, …) so imports stay flat. Defines nothing itself. |
| `evolution/` | The **live evolution driver**. `_core.py` owns `evolve_star`, the JIT core, and the 3-phase windowed `lax.scan`; `step.py` / `scan.py` hold the per-step and scan-body helpers; `timestep.py` is the adaptive step-size controller; `adaptive/` is the free-mesh subgiant/RGB evolver + differentiable record-replay (§4). |
| `solver/` | The **production structure solver**: block-tridiagonal Henyey/Newton relaxation + the IFT adjoint. This is the *structure differentiability boundary*. |
| `microphysics/` | Pure, differentiable physics leaves: equation of state, opacity, nuclear burning, neutrino losses. Plus an optional `mesa/` validation backend. |
| `transport/` | Mixing-length-theory (MLT) convective transport and the radiative/convective switch. |
| `composition/` | Per-shell composition evolution: nuclear burning, convective mixing, element diffusion. |
| `mesh/` | The Lagrangian mass grid: initial mesh, zone re-placement (equidistribution), and conservative remapping between grids. |
| `oscillations/` | Adiabatic oscillation (asteroseismology) solver + sensitivity kernels, inversions, and seismic diagnostics. |
| `fgong/` | The differentiable bridge from a converged structure to the FGONG structure format the oscillation solver consumes; plus file I/O. |
| `inference/` | Gradient-based parameter fitting (Levenberg–Marquardt) + optional warm-start/emulator seeding. |
| `calibration/` | Solar calibration and comparison against MESA / Model S references. |
| `config/` | Constants and the parameter/config objects (the traced-vs-static split, §12). |
| `data/` | Committed physics tables (EOS, opacity, conduction, Fermi–Dirac) and reference models. |

**Top-level modules.** The only compatibility shims at the top level are `henyey.py` (re-exports the
`solver/` package), `stellar.py`, and `structure.py`. Constants are imported from `config/constants.py`
and the oscillation code lives in the `oscillations/` package (there are no `constants.py` /
`oscillations_jax.py` top-level shims). `structure.py` is a legacy shooting solver retained for
bootstrapping and calibration (§5).
`evolution/adaptive/` (`forward.py` = `evolve_star_adaptive`, plus `trajectory.py`) is the free-mesh
subgiant/RGB evolver with differentiable frozen-schedule record-replay (§4).
`gradient_policy.py` enumerates the differentiability policy (§6).

---

## 4. The forward evolution loop

**Entry point:** `evolve_star(mass, Z=0.014, max_steps=500, alpha_mlt=1.9, ...)`
(`evolution/_core.py:380`). It takes the physical parameters plus solver flags and returns a dict of
tracks and final state (`log_L`, `log_Teff`, `log_R`, `star_age`, central conditions, final
composition profiles, and the converged structure record used for oscillations).

`stellar.py` re-exports it; the real implementation is in `evolution/_core.py`.

**The loop is a `jax.lax.scan` over timesteps.** Each step (`step_fn`, a closure inside the JIT core):

1. Remap the composition onto the fixed-size grid.
2. Solve the **structure** at the current composition (the Henyey Newton solve, §5) — this calls the
   microphysics and MLT internally.
3. Update the **composition** (burn → mix → diffuse, §8).
4. Adapt the **mesh** so zones track sharpening features (§8).
5. Compute diagnostics and choose the **next timestep** (`evolution/timestep.py`): a step is rejected
   and halved if the model changes too much in one step or Newton fails; otherwise the next step is
   set by an H211b digital-filter controller and capped by physical timescales.

**Windowed scan (three phases).** For gradient efficiency the scan is split into three
`lax.scan` calls in `_core.py`: a forward-only prefix `[0, k0)`, a
gradient-carrying window `[k0, k1)`, and a forward-only tail `[k1, max_steps)`. Only the middle window
participates in the backward pass, so the cost of `jax.grad` scales with the window size, not the full
trajectory. The window defaults to the whole run and is set via the `grad_window` argument (§6).

**Why the loop body is written as inlined module-level helpers.** An earlier refactor that split the
step into a `CarryState` NamedTuple changed the compiled XLA graph enough to alter floating-point
scheduling and cause numerical drift over long (1000+ step) runs. The helpers are therefore ordinary
module-level functions (Python calls are invisible to XLA tracing, so the compiled graph is identical
to a hand-inlined monolith), and the scan carry is a plain tuple whose length differs by mode rather
than a fixed-width struct. This is a deliberate, load-bearing choice — see the docstrings in
`evolution/__init__.py` and `evolution/contracts.py`.

**`evolution/adaptive/` — the subgiant/red-giant path + differentiable record-replay.** `lax.scan`
requires fixed array shapes, which caps how aggressively the mesh can be re-zoned mid-run. So
`evolution/adaptive/forward.py::evolve_star_adaptive` runs a plain Python loop around the **same**
JIT-compiled Henyey structure solve, freely re-meshing every step (concentrating zones on the thin
hydrogen-burning shell) to reach the subgiant and the red-giant branch, and records the full
`Trajectory` — per-step dt, mesh, and composition (`trajectory.py`). That recording is then **replayed
differentiably** with the dt + mesh + composition **schedules frozen** (`stop_gradient` at source —
GP-11 `MESH_SCHEDULE`, GP-12 `COMP_SCHEDULE`) but the `y_henyey` state carry kept **live**, yielding the
exact discrete adjoint (GRADSOLVE, arXiv:2609.02876). This record-replay is what delivers the first
through-evolution seismic gradient **past the main sequence** — the subgiant ∂ν²/∂M (see §9 and
`docs/reference/gradient-accuracy.md`). In short: `_core.py` is differentiable, fixed-mesh,
main-sequence-focused; `evolution/adaptive/` is free-mesh, subgiant/RGB-capable, non-differentiable in
the forward loop but differentiable via frozen-schedule replay.

---

## 5. The structure solver

`solver/` computes a single converged stellar structure and is the **structure differentiability
boundary** — the point where structure gradients are produced by the IFT.

**State vector.** The solver works on `y` of shape `(N_shells, 4)`, columns
`[ln r, ln P, ln T, ℓ]` (log radius, log pressure, log temperature, luminosity `L/L_sun`) on a
fractional-mass mesh.

**Production entry point:** `henyey_solve_from_state_atm(...)` (`solver/continuation.py:325`). It
re-converges the structure from the previous timestep's converged state — the natural continuation
(homotopy) of quasi-static evolution. Underneath it is the custom-differentiation boundary
`_henyey_continuation_atm` (`solver/continuation.py:49`).

**The solve pipeline** (per Newton iteration, in a `lax.while_loop`):

1. **Residual** (`residual.py`) — the four coupled structure equations per cell (mass–radius,
   hydrostatic equilibrium, energy, temperature gradient), each evaluated with a full
   EOS → opacity → nuclear → neutrino → MLT lookup at the cell.
2. **Jacobian** (`jacobian.py`) — a block-tridiagonal `∂F/∂y` built by forward-mode AD of 4×4 blocks.
3. **Linear solve** (`thomas.py`, `conditioning.py`) — an O(N) block-Thomas elimination for the Newton
   correction, with row equilibration and Levenberg regularization for ill-conditioned cores.
4. **Damping + line search** (`damping.py`, `conditioning.py`) — a residual-gated per-zone damping and
   an Armijo backtracking line search.
5. **Convergence** — declared when both the residual norm and the max correction fall below tolerance.

**Atmosphere boundary.** The outer boundary uses an atmosphere bridge (`surface_bc.py`) integrated from
the photosphere inward once per step; during Newton the surface condition is *linearized* about that
bridge (avoiding a repeated expensive integration).

**Energy equation & the gravothermal term.** `eps_grav.py` computes the gravothermal energy rate
(energy released/absorbed as a shell contracts and heats between timesteps) that enters the energy
equation. It is its own module because it is shared by the residual, the forward conditioning, and the
adjoint, and because it is the one place the previous-step history couples in — which is what turns a
static structure solve into a time-dependent evolution step.

**Legacy shooting solver.** `structure.py` (top level) is an older shooting-method BVP solver
(RK4 integration + a 2×2 Newton match). It is retained because it builds the ZAMS initial model that
seeds the Henyey warm start and because the calibration code paths still use it. It is *not* the
production per-step solver (it over-estimates the ZAMS radius of a 1 M_sun model by a few percent,
which is precisely why the Henyey solver replaced it).

---

## 6. Differentiability architecture

This is the defining feature, so it gets its own section.

**How gradients flow.** `evolve_star` is a JIT function whose body is the windowed `lax.scan`.
`jax.grad` differentiates it by reverse-mode through the scan. The expensive, ill-conditioned inner
solves are **not** unrolled — each is a custom differentiation boundary that applies the **implicit
function theorem**: at a converged solution `F(y*, θ) = 0`, the gradient satisfies
`∂y*/∂θ = −(∂F/∂y)⁻¹ (∂F/∂θ)`, so the backward pass is a *single transposed linear solve* at `y*`
rather than backprop through the iterations. This makes the gradient exact (no finite-difference bias),
cheap (one linear solve), and independent of how many Newton steps were taken.

Custom-VJP boundaries in the codebase:

- **Structure solve** — `_henyey_continuation_atm` (`solver/continuation.py:49`,
  `nondiff_argnums=(7, 8, 9, 21, 22)`; adjoint helpers in `solver/adjoint.py`).
- **Oscillation eigenfrequency** — `eigenfreq_radial` / `eigenfreq_nonradial`
  (`oscillations/adjoint.py:36`, `:111`).
- **Density inversion** (pressure→density Newton) inside the EOS — custom-JVP via the IFT.

**The gradient policy (what is detached, and why).** `gradient_policy.py` defines a `GradPolicy` enum
of twelve numbered policies. Every `stop_gradient` in the codebase cites its policy number in a comment,
and the rule is that detachment lives in the *orchestrator*, not in leaf physics. Differentiable
inputs (mass, Z, α_mlt, f_ov, Y_init, and the opacity/eps_nuc/diffusion calibration knobs) flow to the
observable outputs and oscillation frequencies. The twelve detached categories are algorithmic, not
physical derivatives:

| # | Policy | What is detached |
|---|---|---|
| 1 | `TIMESTEP` | step size / accept–reject / step-control algebra |
| 2 | `MESH_POSITIONS` | grid node locations (the one policy owned by `mesh/`, not the orchestrator) |
| 3 | `DIFFUSION` | element diffusion (operator-split, straight-through) |
| 4 | `Z_CNO_ISOTOPES` | secondary CNO isotope carries |
| 5 | `STRUCTURE_TO_COMP` | one-way structure→composition coupling |
| 6 | `IDLE_GUARD` | outputs on idle steps |
| 7 | `WINDOWED_SCAN` | scan phases outside the gradient window |
| 8 | `EPSGRAV_HISTORY` | a targeted numerical guard on the gravothermal-history extraction (the live history values themselves stay differentiable) |
| 9 | `CONVERGENCE_FLAGS` | boolean converged/rejected flags |
| 10 | `WARM_START_GUESS` | the ρ warm-start initial guess fed to the Henyey solve (guess-independent at convergence, so detaching it is exact) |
| 11 | `MESH_SCHEDULE` | the per-step recorded mesh replayed in the differentiable RGB/subgiant record-replay (frozen at source; §4) |
| 12 | `COMP_SCHEDULE` | the per-step recorded composition replayed in the differentiable record-replay (frozen at source; §4) |

**Adjoint trust gate.** The IFT backward (policy 9, `CONVERGENCE_FLAGS`) zeros a step's gradient
contribution only when its converged (scaled) residual exceeds a *fixed* threshold
`_ADJOINT_GATE_TOL = 0.03` (`solver/continuation.py`), **decoupled** from the Newton solve tolerance —
so tightening the solve (Newton `tol=1e-4`) never spuriously gates a converged-enough step. (See
`docs/reference/gradient-accuracy.md`.)

**The trust budget.** `GRAD_TRUST_STEPS = 100` (`evolution/_core.py:14`) is the number of timesteps for
which autodiff gradients are validated against finite differences. `evolve_star` emits a
`GradientTrustWarning` when `max_steps` exceeds it. To differentiate a longer trajectory while staying
within budget, use `grad_window=(k0, k1)` to differentiate only a sub-window (policy 7). The accuracy
tiers and bounds are documented in [`reference/gradient-accuracy.md`](reference/gradient-accuracy.md).

**Smoothness under adaptivity.** Adaptive timestepping and hard switches would otherwise inject kinks
into gradients. Two techniques address this: observables are interpolated onto a fixed physical
coordinate (age or central hydrogen) so the step-index has no derivative kink, and hard physical
switches (convective/radiative, EOS regime blends) use smooth surrogates on the backward pass (e.g. the
MLT switch in `transport/switch.py` is exact on the forward pass but has a sigmoid-blended derivative).

---

## 7. Microphysics

`microphysics/` is the pure, differentiable, leaf physics layer (no imports from the solver or
evolution). Each module exposes one primary entry point:

- **Equation of state** — `eos_lookup(logT, logP, X, Z)` (`eos.py:544`) returns density, mean molecular
  weight, ∇_ad, entropy, cp, and the pressure derivatives. It is a smooth blend of an OPAL table
  (non-degenerate regime) and a Fermi–Dirac / Helmholtz electron EOS (degenerate cores), gated by a 2-D
  sigmoid so the degenerate branch activates only where needed and accurate. Both branches are always
  computed and blended arithmetically (rather than a `lax.cond`) to keep a fixed-size differentiable
  graph.
- **Opacity** — `kappa(logT, logRho, X, Z)` (`opacity.py:770`) blends four sources: Ferguson (low-T
  molecular/H⁻), OPAL (high-T radiative), Compton scattering, and Potekhin electron conduction
  (combined with the radiative part by a harmonic mean).
- **Nuclear burning** — `epsilon_nuclear(rho, T, X, Z, ...)` (`nuclear.py:342`): the PP chain + CNO
  cycle with electron screening, plus a ³He equilibrium solver.
- **Neutrino losses** — `epsilon_neutrino(rho, T, X, Z)` (`neutrino.py:77`): plasma, photo, and pair
  thermal-neutrino processes.

**Tables** are loaded from `data/` (module-relative, each path overridable by an environment variable)
and interpolated with shared grid-location helpers in `interp_utils.py`. Interpolation schemes range
from quadrilinear to monotone-Hermite to bicubic depending on the smoothness the gradient needs.

**Differentiability-safe math.** `safe_math.py` provides `safe_power`, a custom-JVP power function
whose forward value is exact but whose derivative floors the base away from zero (fractional powers
have derivatives that blow up as the argument → 0, which would produce NaNs at surface/low-density
zones). Regime boundaries use smooth blends (sigmoids, smoothsteps, cosine tapers) rather than step
functions wherever a gradient must cross them. The package uses **no** `stop_gradient` internally —
that is the caller's job.

**Optional MESA backend.** `microphysics/mesa/` is an optional, forward-only *validation* backend that
calls MESA's real Fortran microphysics (EOS, opacity, neutrino, MLT) so the JAX reimplementations can
be checked against ground truth. It is selected by the environment variable `STELLAR_MICROPHYSICS=mesa`
(default `jax`), which swaps the leaf functions at import time. The mechanism is thin Fortran
`bind(C)` shims (compiled to shared libraries) called through `ctypes`, exposed into JAX via
`jax.pure_callback`; each adapter attaches a custom-JVP whose primal comes from MESA and whose tangent
uses the JAX implementation as a Jacobian proxy. It is not differentiable and not the production path,
and it fails loudly (never silently falls back) if the shim libraries are missing.

---

## 8. Mesh, transport & composition

**Mesh (`mesh/`).** The grid is Lagrangian — the coordinate is enclosed mass fraction `q = m/M ∈ [0,1]`,
not radius. Two grids exist: the structure solver's variable-resolution mesh, and a **fixed-size**
composition grid of `N_COMP = 200` zones (`config/mesh_defaults.py:23`). A shared equidistribution
kernel (`equidistribute.py`) re-places zones so each spans an equal change in a "mesh function"
(driven by pressure/temperature gradients for structure, by hydrogen depletion for composition, so
zones concentrate on the burning shell). `remap.py` moves field values between grids: a mass-conserving
(conservative) remap for composition and a linear remap for structure. Per the differentiability policy,
**grid positions are detached** (node placement is an algorithmic choice) while the field *values* stay
differentiable.

**Transport (`transport/`).** `mlt.py` implements mixing-length-theory convective transport (solving
the MLT cubic for convective efficiency). `switch.py` selects radiative vs convective transport; its
forward pass is the exact Schwarzschild/Ledoux hard switch, but its derivative is a sigmoid value-blend
so a nonzero mixing-length gradient keeps flowing across the convective boundary.

**Composition (`composition/`).** Per-shell species (H, He, metals, and the CNO isotopes) are evolved
by operator splitting each timestep, in order:

1. **Burn** (`burn.py`) — deplete hydrogen from the local nuclear rate; relax the CNO isotopes and ³He
   toward equilibrium.
2. **Mix** (`mix.py`) — instantaneously homogenize convective regions (core convection + overshoot, and
   the surface-connected envelope), with a differentiable convective-boundary position.
3. **Diffuse** (`diffuse.py`, `burgers.py`) — gravitational settling / element diffusion by solving the
   Burgers equations for the diffusion coefficients, then advecting/diffusing species. This step is
   operator-split and detached on the backward pass (policy 3).

These modules are pure; they consume a detached copy of the structure (`STRUCTURE_TO_COMP`, policy 5:
structure drives composition one-way), and the caller applies the detachment.

---

## 9. Oscillations (asteroseismology)

`oscillations/` turns a converged structure into adiabatic oscillation (p-mode) frequencies and
provides analytic gradients of those frequencies. It follows the ADIPLS and GYRE reference
formulations and has two parallel backends sharing the same physics: a differentiable JAX path
(fixed-step RK4 via `lax.scan`, in the autodiff graph) and a diagnostic NumPy/SciPy path (adaptive
integrator, for validation).

**End-to-end differentiable entry:** `compute_eigenfreq_from_structure_jax(...)`
(`oscillations/eigenvalue.py:230`). The chain is:

```
structure → coefficients → RK4 integrate (GYRE variables) → surface-BC determinant D(σ²)
          → scan σ² grid, bracket sign changes, refine root (Brent) → IFT adjoint → differentiable σ²
```

1. **Coefficients** (`coefficients.py`) — build the dimensionless structure coefficients on a
   fractional-radius grid, extending synthetic points toward the center for the regularity boundary
   condition.
2. **Integrator** (`integrator.py`) — classical RK4 in the GYRE variable set. (The port from the older
   ADIPLS formulation to GYRE was made because the ADIPLS form amplified the buoyancy term by
   frequency, which is numerically explosive for the sharp composition gradients of evolved cores.)
3. **Determinant** (`determinant.py`) — integrate from the center and evaluate the surface boundary
   condition; the eigenfrequencies are where the determinant crosses zero. Vacuum and isothermal-
   atmosphere outer conditions are both available.
4. **Eigenvalue search** (`eigenvalue.py`) — scan trial frequencies, bracket sign changes, refine with
   Brent. This root-find is forward-only.
5. **Adjoint** (`adjoint.py`) — the differentiability boundary: the root is found non-differentiably,
   but gradients are supplied analytically by the IFT wrapped in `@custom_vjp`. The backward pass
   differentiates with respect to *both* the coefficients and the radius grid (a mass change shifts the
   radii, which shifts where coefficients are sampled — a large part of `∂ν/∂mass`).

**Through-evolution seismic gradients.** Composing this eigenfrequency adjoint with the frozen-schedule
evolution replay (§4) gives the gradient of a seismic observable w.r.t. a physics input *through the
whole evolution* — including the delivered **subgiant ∂ν²/∂M** (`test_replay_gradient_seismic_subgiant`,
1.5 M☉, sign-correct, ~24.5%). The observable is the cyclic frequency ν (ν = σ/2π); `seismic_conversion.py`
converts σ²↔ν², and ν² is used at the subgiant because σ² is ill-conditioned there (a near-cancellation
in the σ²=ν²·factor(M) split). See `docs/reference/gradient-accuracy.md`.

**Beyond frequencies:** `eigenfunction.py` recovers displacement eigenfunctions and mode inertia;
`kernels.py` / `analytic_kernels.py` compute structure sensitivity kernels by AD and by finite
difference (each other's cross-check) for `inversions.py` (SOLA inversions); `seismic_quantities.py`
computes large/small separations, frequency ratios, Δν, and ν_max; `glitch.py` fits acoustic-glitch
signatures; `surface_correction.py` applies the near-surface-term correction.

---

## 10. FGONG bridge

FGONG is the standard interchange format for a stellar model's radial structure. `fgong/` connects the
evolution output to the oscillation solver in two modes:

- **Differentiable, in-memory** — `structure_to_fgong_jax(...)` (`fgong/builder.py:37`) turns the
  converged structure record from `evolve_star` into FGONG-layout JAX arrays, keeping the whole
  `mass → frequency` chain in one differentiable graph (no file round-trip). This is the path used for
  gradient-based seismic fitting.
- **File I/O** — `io.py` reads/writes FGONG (and GYRE) files for interoperability with external
  oscillation codes; `hires_profile.py` produces a high-resolution profile for those files. These are
  off the gradient path.

---

## 11. Inference & calibration

**Inference (`inference/`)** is the inverse problem: given observed stellar quantities and their
uncertainties, find the parameters θ that reproduce them. `infer_parameters(...)`
(`inference/solver.py:91`) runs Levenberg–Marquardt, using `jax.jacrev` to get the exact Jacobian of
the residuals with respect to θ in one reverse-mode pass — the payoff of the differentiable forward
model (no finite-difference Jacobian, no surrogate). It is a pure "leaf consumer": it contains no
`lax.scan`, no `stop_gradient`, and no custom VJP of its own; it simply differentiates a user-supplied
forward function (typically wrapping `evolve_star`). Optional `warmstart.py` / `emulator.py` seed the
initial guess (a fast surrogate); these run outside the JAX graph and are never on the gradient path.

**Calibration (`calibration/`)** solves for the mixing-length and initial helium that make a 1 M_sun
model reproduce the Sun (`solar_calibrate(...)`, `calibration/solar.py`), and compares evolved models
against MESA history tracks and the Model S sound-speed/density profile (`comparison.py`). The
differentiable residuals live here; the outer calibration loop (a Newton solve over the two calibration
parameters) is an ordinary non-differentiable optimizer. The calibrated constants it produces become
defaults in `config/`.

---

## 12. Configuration

`config/` holds constants and the two configuration objects, split by a rule that matters for
performance:

- **`StellarParams`** (`config/params.py`) — the *traced / differentiable* parameters (mass, Z, α_mlt,
  f_ov, Y_init, and the physics knobs). Changing these does **not** trigger recompilation and they can
  be differentiated.
- **`SolverConfig`** (`config/solver_config.py`) — the *static* configuration (max steps, feature
  flags). These are passed as static arguments to `jax.jit`; changing one triggers a fresh XLA
  compilation. The invariant is that `SolverConfig` is unpacked into individual static arguments at the
  JIT boundary, never passed as one traced struct.

The distinction between these two is the core configuration contract: it keeps the compilation cache
small (few distinct `SolverConfig` values) while letting the physically interesting parameters be
differentiated freely. `constants.py` holds CGS physics constants; `mesh_defaults.py` holds
`N_COMP = 200` and the composition mesh; `mesa_config.py` parses the identical-physics comparison
configuration from a committed reference input file.

---

## 13. Cross-cutting concerns

- **JAX/XLA compilation.** The forward and backward passes each compile once (single-threaded, on the
  order of minutes for a full evolution) and then execute fast; a persistent compilation cache makes
  this a one-time cost per distinct function signature. More CPU cores speed execution, not
  compilation.
- **Compiled-graph identity.** Several modules are deliberately *not* refactored into shared helpers
  (kept inlined) so the differentiable forward pass produces a bit-identical XLA graph — required so
  the IFT adjoint's Jacobian matches the forward residual, and to avoid memory blowups when compiling
  the backward pass. Docstrings flag these "keep whole / inline" invariants.
- **Numerical conditioning.** The block-tridiagonal solves use row equilibration and Levenberg
  regularization; the energy equation uses a diagonal preconditioner that cancels algebraically in both
  the Newton step and the adjoint (so it improves conditioning without biasing the gradient).
- **Custom-VJP fragility.** Custom differentiation boundaries are sensitive to argument order and must
  keep their `defvjp`/`defjvp` registration in the same module as the function (JAX registers by
  function identity). The solver enforces its signature with a compile-time assertion, because a
  one-position argument shift would silently corrupt every gradient with no error.

---

## 14. Extending the code

- **Add a physics calibration knob.** Add a field (default 1.0) to `StellarParams`, thread it to the
  point in the microphysics where it multiplies, and it is automatically differentiable with no change
  to `evolve_star`'s signature or the compilation behavior.
- **Add or change physics in a leaf.** Microphysics modules are pure and independent; change one
  without touching the solver. Keep new regime boundaries smooth (blends, not steps) so gradients
  survive, and use `safe_math` for fractional powers near zero.
- **Respect the differentiability policy.** Any new `stop_gradient` must cite a `GradPolicy` number and
  live in the orchestrator, not a leaf. If you add an iterative solve, wrap it as an IFT custom-VJP
  boundary rather than unrolling it.
- **Validate gradients.** New physics on the gradient path should be checked with an autodiff-vs-finite-
  difference test within the trust budget (see [`reference/gradient-accuracy.md`](reference/gradient-accuracy.md)).
- **Where to look first.** `evolution/_core.py` for the loop, `solver/continuation.py` for the structure
  solve and its adjoint, `gradient_policy.py` for the differentiability contract, and each package's
  `contracts.py` for its interface and gradient boundary.

---

*This document describes the architecture at a system level. For precise, validated numbers (accuracy
bounds, physics scope, validation regimes) see the [`reference/`](reference/) pages. For the exact
behavior of any component, the source and its docstrings are authoritative.*
