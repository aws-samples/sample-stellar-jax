# stellar-jax

Differentiable 1D stellar evolution in JAX. Analytic gradients through the structure equations via `lax.scan` + implicit function theorem.

## Background

stellar-jax is the output of an experiment: **how far can autonomous AI agents carry a real codebase in a domain its authors have no background in?** The domain we chose is stellar astrophysics — specifically, 1D stellar evolution and asteroseismology (modelling how stars evolve over time and how they oscillate). The code in this repository is that experiment's result. We hope it proves useful to the community —
feedback, questions, and contributions are welcome at [stellar-jax@amazon.com](mailto:stellar-jax@amazon.com).

The established reference in the field is [MESA](https://mesastar.org) (Modules for Experiments in Stellar Astrophysics), a mature open-source 1D stellar-evolution code written in Fortran; stellar-jax is validated against MESA under matched physics. Our goal was something MESA does not set out to be: **differentiable** — able to compute exact analytic gradients of its outputs with respect to stellar parameters and physics, straight through the simulation. We built it in [JAX](https://github.com/jax-ml/jax), a Python library for automatic differentiation and accelerated array computing. Python is the common language of the scientific-computing and machine-learning communities, which keeps the code approachable to that audience, while JAX provides the automatic differentiation the goal requires.

An initial version was produced with [AWS Transform](https://aws.amazon.com/transform/), AWS's agentic code-transformation service. From there the code was iterated by a fleet of autonomous agents running on [Amazon ECS](https://aws.amazon.com/ecs/), collaborating through an ordinary GitHub workflow:

- **Issues** — each piece of work (a feature, a bug fix, a physics validation, a research probe) is filed as a GitHub issue with explicit, testable acceptance criteria.
- **Pull requests** — an agent picks up an open issue, works on its own branch, and opens a pull request with the change.
- **Review** — every pull request is reviewed before it can merge; any change to physics or gradients must add or update a test.
- **CI** — once a pull request is approved, the automated test suite runs. A claim counts as validated only when its test is green on `main`.
- **Merge** — approved, passing pull requests merge automatically, and the cycle repeats on the next issue.

The whole loop is supervised by human maintainers together with [Kiro](https://kiro.dev/), AWS's agentic development environment.

## What it does

Evolves stars from ZAMS through the main sequence and onto the subgiant / red-giant branch, with:
- Per-shell composition tracking (200 zones, burning + mixing)
- Convective overshooting (mass-dependent)
- Element diffusion (He settling)
- Adaptive timestepping
- Full differentiability via JAX (`jax.grad` through the entire evolution)

## Quick start

Requires **Python ≥ 3.11**. `pip install -e .` installs everything the package and the demos need
(JAX + jaxlib pinned to 0.10.2 for reproducible float results, NumPy, and matplotlib for the demo figures):

```bash
pip install -e .              # from a local clone; pulls jax==0.10.2, jaxlib==0.10.2, numpy, matplotlib
# equivalently, the exact pinned set is in requirements.txt:
#   pip install -r requirements.txt && pip install -e . --no-deps
```

```python
from stellar_jax import evolve_star
import jax

# Evolve a 1 solar mass star through the main sequence
r = evolve_star(1.0, Z=0.014, max_steps=100)

# Analytic gradient: ∂(log L at TAMS) / ∂M
g = jax.grad(lambda m: evolve_star(m, Z=0.014, max_steps=100)['log_L'][-1])(1.0)
```

## Demos

Runnable, collaborator-facing demonstrations are in [`demos/`](demos/):

- **[`demos/differentiable-seismic-gradients/`](demos/differentiable-seismic-gradients/)** — analytic
  gradients of oscillation frequencies with respect to stellar physics, computed end-to-end in one
  reverse-mode pass, plus sound-speed structure kernels validated against ADIPLS, and a multi-mass
  HR diagram (1.0/1.5/2.0 M☉) showing forward-evolution tracks vs MESA with the subgiant ∂ν²/∂M
  operating point.
  Runnable: `python demos/differentiable-seismic-gradients/demo_seismic_gradients.py` (~1–3 min).

**Seismic gradients at a glance** (analytic reverse-mode AD vs independent finite difference; each row is a committed CI test — full table and tests in the demo README):

| Gradient (AD vs FD) | Regime | AD vs FD |
|---|---|---|
| ∂ν²/∂M through evolution → subgiant | evolved subgiant | **24.5%** (sign ✓, \|Δlog L\|<0.1) |
| ∂ν²/∂opacity, ∂ν²/∂eps_nuc | through evolution | < 5%, < 1% |
| ∂ν²/∂Y_init (M–Y degeneracy) | through evolution | 3.4% |
| ∂ν²/∂α_MLT | near-ZAMS / evolved (X_c≈0.22) | < 25% / 3.9% |
| ∂r₀₂/∂θ, ∂r₀₁/∂θ (surface-independent) | direct | < 5% |
| ∂ν/∂c²(r) kernels vs ADIPLS | Sun + 2 M☉ | ~3.4% / ~6% RMS |

The subgiant row carries ∂ν²/∂M through post-main-sequence evolution — sign-correct and reproduction-verified (|Δlog L| < 0.1). The kernel row is an independent check of the structure kernels against ADIPLS.

A ready-to-run container is provided: see [`demos/Dockerfile`](demos/Dockerfile).

## Physics

- OPAL EOS + opacity tables (4D interpolation)
- MLT convective energy transport (Böhm-Vitense, α threaded as parameter)
- Krishna Swamy T(τ) atmosphere (extended to τ=100 with MLT + SAL correction)
- PP + CNO nuclear burning (per-shell, self-consistent)
- Schwarzschild convection + step overshooting (f_ov=0.016)
- Element diffusion (Thoul et al. 1994)
- Adaptive timestepping (varcontrol-based)

## Validation status

**Experimental — under active validation.** The source of truth for what's proven is the **CI test suite** — a claim is validated iff its test is green on `main` (the exact tolerances live in the test assertions under `tests/`). Physics scope and coverage: [`docs/reference/capabilities.md`](docs/reference/capabilities.md).

At a glance:
- **Main-sequence structure vs MESA** (identical physics): log L < 0.10 dex, log Teff < 0.03 dex; internal sound speed vs Model S < 1% — **CI-gated**.
- **Post-MS (RGB, pre-flash)** ascent vs MESA + the gravothermal (ε_grav) gradient — **CI-gated** (He flash is out of scope).
- **Analytic gradients (AD vs finite-difference):** ∂/∂M and ∂/∂α validated < 5% (< 1% at fixed timestep); ∂/∂Y and ∂/∂Z validated. **Seismic** ∂ν²/∂physics are CI-gated across M / opacity / eps_nuc / Y / α (incl. an evolved late-MS model) / Z / f_ov and the surface-independent ratios — now including the first **through-evolution ∂ν²/∂M into the subgiant** (sign-correct, 24.5%); see the demos table. Note gradient accuracy is validated at specific operating points, not yet swept across the full parameter grid.
- **Asteroseismology (Δν, δν₀₂, r02, …):** the oscillation forward model's δν₀₂ age signal was fixed (≤2% vs GYRE across 4 masses) along with Γ₁ and the outer-BC / absolute-frequency correction. The {M, age} seismic inversion is not yet passing: seismic frequencies **alone** cannot break the M↔age degeneracy (intrinsic to acoustic modes), so recovery requires **seismic modes + L + Teff** (in progress). Do not treat seismic outputs as validated yet.

Microphysics (EOS / opacity / nuclear rates) is reimplemented in JAX for end-to-end differentiability — MESA-*matching* on the main sequence (ρ ≤ 0.4%, ∇_ad ≤ 0.5% interior), not MESA's Fortran tables.

## Structure

```
src/stellar_jax/        # The package: unified solver (stellar.py) + microphysics/, solver/, oscillations/, evolution/, structure.py, …
src/stellar_jax/data/   # OPAL/Ferguson/Potekhin tables, Model S, MESA comparison tracks (+ generation recipes)
tests/test_*.py         # Per-module validation tests (microphysics, solver, evolution, oscillations, …)
tools/                  # Data-build tooling (Ferguson table build, MESA comparison-data recipe)
scripts/                # Physics lint + data-provenance generators
docs/reference/         # User reference: capabilities, gradient accuracy, validation regimes, starting model
docs/ARCHITECTURE.md    # How the code works: package map, evolution loop, solver, differentiability
```

## Development

Validation is CI-gated: a claim counts as validated only when its test is green on `main`, and any
physics/gradient change must add or update a test under `tests/`.

## Known limitations

- **He flash / core flash, pre-main-sequence, rotation, mass loss** — out of v1 scope by design.
- **Asteroseismology** is under active repair (see *Validation status* above).
- The physics-scope summary (included / not included) lives in [`docs/reference/capabilities.md`](docs/reference/capabilities.md).

## License & bundled data

stellar-jax's source is licensed under Apache-2.0 (see [`LICENSE`](LICENSE)). To make the code runnable,
this repository also bundles third-party reference data under [`src/stellar_jax/data/`](src/stellar_jax/data/)
(OPAL, Ferguson, Potekhin, Model S, Kepler LEGACY, and MESA/GYRE/ADIPLS-derived references). That data is
**not** covered by the Apache-2.0 license and remains under its original terms — it was copied in only to
make the code functional. See [`THIRD_PARTY.md`](THIRD_PARTY.md) for sources and citations.
