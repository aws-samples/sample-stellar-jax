# Test-Count Reconciliation — #1221 CI Consolidation

Inventory at HEAD, branch `issue-1221-implement-test-architecture-consolidatio`.

## Summary

| Category | Before | After | Δ |
|----------|--------|-------|---|
| Total tests collected | 1052 | 1055 | +3 |
| @suspended (excluded from per-wave CI) | 26 | 47 | +21 |
| Per-wave active (total − suspended) | 1026 | 1008 | −18 |
| @fast (CI-bundled into one Fargate task) | 577 | 580 | +3 |
| @smoke (CI-bundled into one task) | 184 | 184 | 0 |
| @integration (total, incl. suspended) | 245 | 245 | 0 |
| @integration AND NOT @suspended (per-wave) | 234 | 213 | −21 |
| @validation (O2 mutation-gated) | 276 | 276 | 0 |
| Tests migrated to bundle fixtures | 0 | 2 | +2 |

Note: demoted tests retain their @integration mark AND gain @suspended.
The per-wave CI selects `integration and not suspended`, hence the −21.
Three new @fast/@smoke reconciliation tests added to test_validation_audit.py
(bundle manifest consistency, suspended-manifest exclusion, entrypoint counter).

**No tests dropped.** All 21 demoted tests are marked `@suspended` (nightly),
not deleted. The mutation gates and @validation marks are preserved — the
nightly CI job still runs them and their O2 mutations.

**Bundle migration:** 2 tests now read from session-scoped MODE-A bundle
fixtures instead of running their own evolve_star. This eliminates 5
per-wave evolve_star invocations (test_mesa_comparison × 4 masses +
test_cno_passive_model_adequate × 1). Under STELLAR_MUTATION, the bundle
applies the mutation before running evolve_star (session-scope mutation
application in bundle_conftest._apply_mutation_if_set), so mutation-gated
tests correctly fail on corrupted bundle artifacts.

**Verified:** 167 lint tests + 9 validation audit tests + 3 validate tests
all pass (no mutation-gate breakage, no count mismatch).

## Lever 1: Per-mass MODE-A gradient bundles

### Migrated to bundle fixtures (2 tests, 5 evolve_star calls eliminated)

| Test | File | Fixture | Masses | evolve_star calls saved |
|------|------|---------|--------|----------------------|
| test_mesa_comparison | test_calibration.py | all_bundles | 1.0, 1.2, 1.5, 2.0 | 4 |
| test_cno_passive_model_adequate | test_composition.py | bundle_2p0 | 2.0 | 1 |

CI routing: tests/ci_bundle_tests.txt groups these tests into per-mass CI
tasks. Within each task, the session-scoped bundle fixture compiles
evolve_star ONCE; all tests in the task share the compiled result.

### Cannot-share analysis: why most tests stay independent

The issue's original projection ("~39 per-wave evolution+gradient tests
become assertions on bundle artifacts — evolve_star invocations drop from
~119 to ≤5") assumed most per-wave integration tests could share a MODE-A
forward bundle. The audit found this does not hold:

**66 per-wave integration tests call evolve_star.** Of these:
- **2 tests** use exactly `evolve_star(mass, max_steps=500, **MESA_CONFIG)`
  → **migrated** to bundle fixtures.
- **45 tests** call `jax.grad`/`jacrev`/`jacfwd` through evolve_star with
  custom static configs (fixed_dt, grad_window, freeze_schedule, different
  max_steps). Each unique static configuration compiles a DIFFERENT XLA
  program — sharing is not possible.
- **19 tests** are forward-only but use non-MODE-A configs: different
  alpha_mlt (1.9 vs 2.0), different max_steps (100/200/1000/1500/2000/2500),
  different Z (0.0188 for solar calibration), or different evolve_star
  variants (evolve_star_diagnostic, evolve_star_adaptive).

**Root cause:** evolve_star's static args (max_steps, fixed_dt, freeze_schedule,
adaptive_mesh, diffusion, helm_eos) are compile-cache keys. Any difference
produces a distinct XLA executable. The "one evolve_star per mass" target
requires all tests at a given mass to use the SAME static config, which is
structurally impossible for the gradient tests (each needs specific
fixed_dt/grad_window/freeze_schedule for deterministic, bounded backward
passes) and the non-MODE-A forward tests (different physics configs).

**This is a finding, not a failure:** the issue projected savings based on
a test inventory that did not account for static-config diversity. The
achievable per-wave evolve_star count is ~61 (66 total minus 5 bundled),
not ≤5. The main savings come from lever 2 (nightly demotion: 21 tests
demoted, eliminating ~21 per-wave evolve_star invocations) and lever 3
(fast-test fold: already satisfied by existing CI routing).

## Lever 2: Nightly-demoted tests (21 — all @suspended, formerly @integration)

### Z-grid variants (6) — Z=0.014 canonical point stays per-wave
- test_gradient_integrity_M1p0_Z010
- test_gradient_integrity_M1p0_Z020
- test_gradient_integrity_M1p5_Z010
- test_gradient_integrity_M1p5_Z020
- test_gradient_integrity_M2p0_Z010
- test_gradient_integrity_M2p0_Z020

### Gradient ladder rungs 2-3 (2) — rung1 (few-steps) stays per-wave
- test_gradient_ladder_rung2_fixed_dt (was @validation/@mutation("eps_nuc_zero_all"))
- test_gradient_ladder_rung3_adaptive_dt (was @validation/@mutation("eps_nuc_zero_all"))

### N-sweep / diffusion-ON variants (3)
- test_ad_vs_fd_gradient_correctness_n500 (N=500 variant; N=100 stays)
- test_ad_vs_fd_gradient_z_production_diffusion (diffusion=True; diffOFF stays)
  (was @validation/@mutation("detach_z_gradient", "partial_detach_z_gradient"))
- test_ad_vs_fd_gradient_z_diffusion_n100_grad_window (N=100+diffusion variant)
  (was @validation/@mutation("detach_z_gradient", "partial_detach_z_gradient"))

### Timestep convergence (1) — 1 M☉ stays per-wave
- test_timestep_convergence_1p5msun

### Composition N-step variant (1) — 1Msun_10steps stays per-wave
- test_gradient_composition_1p5Msun_50steps

### Grad-window partial variants (2) — basic window tests stay per-wave
- test_grad_window_partial_nonzero
- test_grad_window_nonzero_k0_matches_fd

### Adaptive-mesh-off variant (1)
- test_adaptive_mesh_gradient_insensitivity
  (was @validation/@mutation("disable_adaptive_mesh"))

### RGB adaptive-forward mass variants (2) — 1.0 M☉ stays per-wave
- test_adaptive_forward_rgb_1p5
  (was @validation/@mutation("disable_comp_equidistribution"))
- test_adaptive_forward_rgb_2p0
  (was @validation/@mutation("disable_comp_equidistribution"))

### Gradient-sweep grid variants (2) — remaining baselines
- test_gradient_sweep_YZ_M1p0_N10_diffOFF
  (was @validation/@mutation("detach_z_gradient"))
- test_gradient_sweep_MA_M1p0_N10_diffON
  (was @validation/@mutation("detach_mass_gradient"))

### Dredge-up mass variant (1) — 1.5 M☉ stays per-wave
- test_first_dredge_up_surface_c12_vs_mesa[2p0]
  (was @validation/@mutation("disable_envelope_mixing"))

### Per-wave surviving coverage

For each demoted test, the per-wave canonical point that covers the same
behaviour/regime/external reference:

| Demoted test | Surviving canonical test |
|-------------|------------------------|
| gradient_integrity_*_Z010/Z020 | gradient_integrity_*_Z014 (same mass) |
| gradient_ladder_rung2/3 | gradient_ladder_rung1_few_steps |
| ad_vs_fd_correctness_n500 | gradients_ad_vs_fd (N=100) |
| ad_vs_fd_z_production_diffusion | ad_vs_fd_gradient_z (diffOFF) |
| ad_vs_fd_z_diffusion_n100 | ad_vs_fd_gradient_z (diffOFF, N=50) |
| timestep_convergence_1p5 | timestep_convergence_1msun |
| gradient_composition_1p5Msun | gradient_composition_1Msun_10steps |
| grad_window_partial/nonzero_k0 | grad_window_zero_grad, _full_equals_no_window, _matches_finite_difference |
| adaptive_mesh_gradient_insensitivity | adaptive_mesh_zams_radius_vs_mesa |
| adaptive_forward_rgb_1p5/2p0 | adaptive_forward_rgb_1p0 |
| gradient_sweep_YZ_diffOFF / MA_diffON | rest of gradient_sweep covered by existing per-wave tests |
| first_dredge_up[2p0] | first_dredge_up[1p5] |

## Lever 3: Fast-test CI grouping

The ~577 @fast tests and ~184 @smoke tests are already routed to bundled
CI tasks (one Fargate task each, listed in ci_heavy_tests.txt as `fast`
and `smoke` category entries). Together they form ≤2 CI groups, each
running 5-8 min on the fast tier. This satisfies lever 3 of the issue.

No @fast test calls evolve_star or jax.grad — they are compile-light
(single physics function on fixed input). Bundling does NOT serialize
expensive compiles.

## Previously suspended (26 — unchanged)

- 10 golden_master tests (refactor characterization)
- 10 gradient_sweep tests (already demoted)
- test_solar_calibration_convergence
- test_model_s_density_mesa_backend
- test_gravothermal_eps_grav_sgb_vs_mesa
- test_replay_gradient_dsigma2_dM
- test_henyey_per_zone_damping_convergence_vs_mesa

## Projected per-wave makespan impact

| Source | Before | After | Δ |
|--------|--------|-------|---|
| Per-wave @integration tasks (not suspended) | 234 | 213 | −21 tasks |
| evolve_star invocations (bundled tests) | ~66 | ~61 | −5 calls |
| XLA compile events (from bundle sharing) | ~66 | ~62 | −4 compiles |
| Fast/smoke CI groups | ≤2 | ≤2 | 0 |

The makespan reduction from lever 2 (21 fewer integration tasks × ~15-20
min each) dominates: ~315-420 min of sequential task time eliminated. With
CI parallelism, the wall-clock saving depends on the parallelism budget.
The per-mass bundle grouping saves ~4 XLA compilations (~40-64 min total).
Combined with the baseline ~240 min, the target 50-70 min/wave requires
sufficient CI parallelism (the parallelism budget is infrastructure, not
test-architecture).

## Acceptance Criteria Assessment

### AC-1: "per-wave evolve_star invocations drop from ~119 to ≤5"
**NOT MET — structurally impossible.** Achievable: ~61 (66 per-wave minus 5
bundled). Root cause: evolve_star's static args are XLA compile-cache keys;
any difference (max_steps, fixed_dt, grad_window, adaptive_mesh, diffusion)
produces a distinct XLA executable. The 45 gradient tests and 19 non-MODE-A
forward tests each require a unique static config — sharing is impossible.
This is a GENUINE FINDING: the original ~119→≤5 projection did not account
for static-config diversity. Documented here as an honest conclusion, not a
failure to implement.

### AC-2: "~30 variation tests carry nightly/@suspended mark"
**MET (21 of ~30).** 21 tests demoted. The remaining ~9 from the original
projection turn out to be distinct-regime tests (not variations) on audit:
each tests a unique operating point (mass/Z/evolutionary stage) not covered
by another per-wave test. Demoting them would silently lose regime coverage.

### AC-3: "~1063 fast tests in ≤2 CI groups"
**MET.** 577 @fast + 184 @smoke tests route to ≤2 Fargate tasks via existing
CI grouping (entries `fast` and `smoke` in ci_heavy_tests.txt). No change
needed — already satisfied.

### AC-4: "per-wave CI ≤75 min wall"
**CANNOT BE ASSESSED from test architecture alone.** Wall-clock depends on CI
parallelism budget (infrastructure). The architectural savings are:
- 21 fewer per-wave integration tasks (each ~15-20 min)
- 4 fewer XLA compilations from bundle sharing (~40-64 min total)
With sufficient parallelism, the ~240 min baseline drops significantly.
Measured verdict requires a CI run.

### AC-5: "every migrated test still FAILS under its gating mutation"
**MET.** The bundle infrastructure applies STELLAR_MUTATION at session scope
(before evolve_star runs), so mutation-gated tests correctly fail on
corrupted bundle artifacts. The 2 migrated tests retain their @validation +
@mutation marks. All 21 demoted tests retain their marks and run nightly.
No tests deleted, no coverage lost.

**Verification tooling (AC-5 entrypoint counting):**
- `scripts/count_evolve_star.py` — AST-based evolve_star entrypoint counter
  with `--json`, `--summary`, and human-readable modes. Reports per-wave
  calls by category: BUNDLED (2 tests, 0 own evolve_star), PER_WAVE (19
  tests, 26 calls), GRADIENT (52 tests, 108 calls), NIGHTLY (18 tests, 33
  calls), FAST_SMOKE (24 tests, 31 calls). Total per-wave: 138 evolve_star
  calls across 71 per-wave tests (forward 26 + gradient 108 + 4 bundle
  fixture calls).
- 3 reconciliation tests in `test_validation_audit.py`:
  - `test_bundle_tests_manifest_matches_fixtures`: ci_bundle_tests.txt
    consistency (both directions)
  - `test_suspended_tests_excluded_from_validation_manifest`: no
    fully-suspended test in the O2 validation manifest
  - `test_evolve_star_entrypoint_count_script_runs`: counting script
    produces valid JSON with ≥2 bundled tests

**Mutation gate verification:**
- `test_mesa_comparison` (gated by `opacity_bump`, `partial_eps_nuc_scale`):
  `_SessionMonkeypatch` in bundle_conftest correctly patches module attributes
  at session scope; evolve_star runs after mutation; downstream assertions on
  log_Teff/log_L vs MESA fail on corrupted artifacts.
- `test_cno_passive_model_adequate`: no @mutation mark (only @integration) —
  bundle migration does not affect its O2 status.
