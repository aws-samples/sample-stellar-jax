# Tests

Validation tests are split into per-module files under `tests/`,
mirroring the module architecture:

| File | Module | Tests |
|------|--------|-------|
| `test_microphysics.py` | EOS, opacity, nuclear, neutrino, conduction | 61 |
| `test_gradient_policy.py` | Gradients, gradient integrity, windowed grad | 38 |
| `test_evolution.py` | evolve_star, timestep, post-MS, RGB forward | 39 |
| `test_transport_validate.py` | MLT, convection, diffusion, α-sensitivity | 35 |
| `test_solver.py` | Henyey solver, block-Thomas, IFT adjoint, eps_grav | 28 |
| `test_oscillations.py` | Pulsation eigensolver, eigenfreq adjoint | 17 |
| `test_calibration.py` | Solar calibration, MESA comparison, Model S | 20 |
| `test_mesh.py` | Adaptive mesh, equidistribution, remesh, remap | 13 |
| `test_composition.py` | Burning, mixing, composition mesh | 12 |
| `test_golden_master.py` | Refactor characterization | 8 |
| `test_fgong.py` | FGONG/GYRE I/O, publication-grade output | 4 |
| `test_inference.py` | Inverse solver, seismic + RGB recovery | 4 |
| `test_config_validate.py` | MESA config parity, mode-A/B separation | 2 |

Additional unit test files from module refactors:

| File | Module | Tests |
|------|--------|-------|
| `test_lint_gradient_policy.py` | GP-N annotation lint | 56 |
| `test_evolution_unit.py` | CarryState, timestep, warmup | 43 |
| `test_config.py` | StellarParams, SolverConfig, constants | 38 |
| `test_dsee_interface.py` | DSEE emulator interface | 37 |
| `test_oscillations_unit.py` | Adjoint, integrator, contracts | 29 |
| `test_composition_unit.py` | Composition ops (burn/mix/diffuse) | 26 |
| `test_composition_helpers.py` | Composition DRY helpers | 25 |
| `test_solver_units.py` | Solver DRY helpers, cell inputs | 25 |
| `test_microphysics_unit.py` | Microphysics contracts, single-fn | 18 |
| `test_inference_unit.py` | Inference contracts, metallicity | 18 |
| `test_calibration_unit.py` | Calibration helpers, newton_2d | 16 |
| `test_fgong_unit.py` | FGONG builder, Brunt-Väisälä | 15 |
| `test_mesh_package.py` | Mesh package DRY, equidistribute | 14 |
| `test_interp_utils.py` | _locate_1d/_locate_4d grid helpers | 13 |
| `test_transport.py` | Transport package, cubic solve | 9 |
| `test_gradients.py` | Gradient smoke tests | 7 |
| `test_grad_timing.py` | Timing instrumentation | 4 |
| `test_timing_instrumentation.py` | Forward timing wrapper | 2 |

Organized by pytest marks (registered in `conftest.py`):

- `@pytest.mark.smoke` — fast sanity + cheap physics checks. Run before opening a PR.
- `@pytest.mark.integration` — expensive correctness checks needing JIT/gradients (>1 min each). Run in parallel in CI.
- `@pytest.mark.fast` — single-function component tests (no evolve_star). The dev-loop fast layer.
- `@pytest.mark.suspended` — temporarily disabled, not run in CI.
- `@pytest.mark.validation` + `@pytest.mark.mutation("name")` — mutation gate.

## Running

```bash
# Fast checks — run before opening a PR
scripts/preflight.sh -k <area>

# Smoke tests
python -m pytest tests/ -m smoke -v

# Expensive gradient checks (need ~16 GB)
python -m pytest tests/ -m integration -v --timeout=1200

# Single test
python -m pytest tests/test_calibration.py::test_alpha_sensitivity_mass_dependent -v

# All fast component tests (isolated physics, no evolve_star)
python -m pytest tests/ -m fast -v
```

## CI

Tests run in CI on every PR. New `@pytest.mark.integration` tests added in any
`test_*.py` file are auto-discovered — no workflow or registry changes needed.

## Other files

| File | Purpose |
|------|---------|
| `conftest.py` | Mark registration, fixtures (stellar, timing, mutation gate) |
| `helpers.py` | Shared utilities: data path resolution, FGONG loader |
| `mutations.py` | Mutation registry (physics breaks for validation tests) |
| `run_tests.py` | Legacy acceptance runner (archived) |

## Validation data

| Reference | Location | Purpose |
|-----------|----------|---------|
| MESA tracks | `data/mesa_comparison/results/` | Apples-to-apples track comparison |
| Model S | `data/model_s/fgong.l5bi.d.15c` | Sound speed profile validation |
| Ferguson 2005 low-T opacity | `data/ferguson_4d.npz` | H⁻/molecular opacity below logT~4.5 |
| MESA FGONG profiles | `data/mesa_comparison/profiles/` | Per-function MESA reference values |

See `data/mesa_comparison/README.md` for how MESA data was generated.
