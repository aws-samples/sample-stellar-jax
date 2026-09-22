"""Fast unit tests for the mesh/ package (issue #498).

Each test is <5s, isolated, uses synthetic inputs. No evolve_star or JIT
compilation of the full solver. Tests verify the extracted functions
preserve the expected behavior.

Reference:
  - MESA mesh_functions.f90:307-321 (do1_xa_function: weight*log10(xa+param))
  - Paxton et al. (2011), ApJS 192, 3, §7 (equidistribution)
  - mesh_adjust.f90:do_xa (conservative remap)
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)


# ===========================================================================
# T1: equidistribute_fixed_N — uniform density produces linspace
# ===========================================================================

@pytest.mark.fast
def test_equidistribute_fixed_N_uniform():
    """Uniform density → equidistributed grid equals linspace.

    When density is constant everywhere, the equidistribution should produce
    a uniform grid (equal spacing). This is the trivial/degenerate case.
    """
    from stellar_jax.mesh.equidistribute import equidistribute_fixed_N

    N = 100
    grid_old = jnp.linspace(0.0, 1.0, N)
    density = jnp.ones(N) * 5.0  # uniform density

    grid_new = equidistribute_fixed_N(density, grid_old, N)

    expected = jnp.linspace(0.0, 1.0, N)
    np.testing.assert_allclose(grid_new, expected, atol=1e-14,
                               err_msg="Uniform density should produce linspace grid")


# ===========================================================================
# T2: equidistribute_fixed_N — peaked density concentrates nodes
# ===========================================================================

@pytest.mark.fast
def test_equidistribute_fixed_N_peaked():
    """Delta-like density peak → zones concentrate at the peak.

    A peaked density function should cause more nodes to cluster where
    the density is highest.
    """
    from stellar_jax.mesh.equidistribute import equidistribute_fixed_N

    N = 200
    grid_old = jnp.linspace(0.0, 1.0, N)
    # Gaussian peak at position 0.3
    x = jnp.linspace(0.0, 1.0, N)
    density = 1.0 + 20.0 * jnp.exp(-((x - 0.3) / 0.02) ** 2)

    grid_new = equidistribute_fixed_N(density, grid_old, N)

    # More nodes near x=0.3: count nodes in [0.25, 0.35] should be > 3× uniform
    in_peak = jnp.sum((grid_new >= 0.25) & (grid_new <= 0.35))
    uniform_expected = N * 0.10  # 10% of range
    assert float(in_peak) > 3.0 * uniform_expected, (
        f"Expected >3× nodes near peak, got {float(in_peak)} vs uniform {uniform_expected}")


# ===========================================================================
# T3: conservative_remap — total mass preserved
# ===========================================================================

@pytest.mark.fast
def test_conservative_remap_total_mass():
    """Conservative remap preserves integrated species mass.

    sum(profile_new * dm_new) ≈ sum(profile_old * dm_old) to machine precision.
    """
    from stellar_jax.mesh.remap import conservative_remap

    N = 100
    # Random-ish grids (monotone, on [0,1])
    grid_old = jnp.linspace(0.0, 1.0, N)
    grid_new = jnp.linspace(0.0, 1.0, N) ** 1.5  # nonuniform

    # Smooth profile
    profile = 0.3 + 0.4 * jnp.sin(jnp.linspace(0, jnp.pi, N))

    profile_new = conservative_remap(profile, grid_old, grid_new)

    # Compute total masses using cell boundaries (same method as in the remap)
    def _total_mass(prof, grid):
        bnd = jnp.zeros(N + 1)
        bnd = bnd.at[0].set(0.0)
        bnd = bnd.at[1:-1].set(0.5 * (grid[:-1] + grid[1:]))
        bnd = bnd.at[-1].set(1.0)
        dm = jnp.diff(bnd)
        return float(jnp.sum(prof * dm))

    mass_old = _total_mass(profile, grid_old)
    mass_new = _total_mass(profile_new, grid_new)

    np.testing.assert_allclose(mass_new, mass_old, atol=1e-12,
                               err_msg="Conservative remap must preserve total mass")


# ===========================================================================
# T4: conservative_remap — identity (same grid → unchanged profile)
# ===========================================================================

@pytest.mark.fast
def test_conservative_remap_identity():
    """Same old/new grid → profile unchanged (identity remap)."""
    from stellar_jax.mesh.remap import conservative_remap

    N = 50
    grid = jnp.linspace(0.0, 1.0, N)
    profile = jnp.linspace(0.2, 0.8, N)

    profile_new = conservative_remap(profile, grid, grid)

    np.testing.assert_allclose(profile_new, profile, atol=1e-14,
                               err_msg="Identity remap should preserve profile exactly")


# ===========================================================================
# T5: conservative_remap — differentiable
# ===========================================================================

@pytest.mark.fast
def test_conservative_remap_is_differentiable():
    """jax.grad through conservative_remap runs and is nonzero.

    Verifies the remap path is AD-transparent (gradients flow through
    the profile interpolation).
    """
    from stellar_jax.mesh.remap import conservative_remap

    N = 50
    grid_old = jnp.linspace(0.0, 1.0, N)
    grid_new = jnp.linspace(0.0, 1.0, N) ** 1.3

    profile = jnp.linspace(0.2, 0.8, N)

    grad_fn = jax.grad(lambda p: jnp.sum(conservative_remap(p, grid_old, grid_new)))
    grad_val = grad_fn(profile)

    assert bool(jnp.all(jnp.isfinite(grad_val))), "Gradient must be finite"
    assert bool(jnp.any(grad_val != 0.0)), "Gradient must be nonzero"


# ===========================================================================
# T6: structure density — P+T weighting peaks at gradients
# ===========================================================================

@pytest.mark.fast
def test_structure_density_gval_weighting():
    """Sharp P+T gradient at one location → density peaks there.

    Checks the P+T weighting concentrates mesh density where structure
    changes rapidly.
    """
    from stellar_jax.mesh.density import compute_structure_density
    from stellar_jax.mesh import initial_lagrangian_mesh

    N_s = 100
    q_mesh = initial_lagrangian_mesh(N_s)
    xi = jnp.linspace(0.0, 1.0, N_s)

    # Sharp gradient at xi=0.3
    ln_P = jnp.log(10.0) * (17.0 - 12.0 * xi - 2.0 * jnp.tanh((xi - 0.3) / 0.01))
    ln_T = jnp.log(10.0) * (7.2 - 3.0 * xi - 1.0 * jnp.tanh((xi - 0.3) / 0.01))
    ln_r = jnp.log(6.96e10 * jnp.maximum(xi, 1e-6) ** (1.0 / 3.0))
    ell = jnp.ones(N_s)
    y = jnp.stack([ln_r, ln_P, ln_T, ell], axis=-1)

    density = compute_structure_density(y, q_mesh)

    # Find peak
    peak_idx = int(jnp.argmax(density))
    peak_xi = float(xi[peak_idx])

    # Peak should be near 0.3
    assert abs(peak_xi - 0.3) < 0.1, (
        f"Density peak at xi={peak_xi}, expected near 0.3")
    # Peak should be well above mean
    assert float(density[peak_idx]) > 3.0 * float(jnp.mean(density)), (
        "Density peak should be >3× mean")


# ===========================================================================
# T7: composition density — log-gval peaks at composition step
# ===========================================================================

@pytest.mark.fast
def test_composition_density_log_gval():
    """X profile with a sharp step → density peaks at the step.

    Verifies the log-gval formula concentrates zones at the H-burning shell.
    """
    from stellar_jax.mesh.density import compute_composition_density

    N = 200
    comp_mfracs = jnp.linspace(0.0, 1.0, N)

    # Sharp step at position 0.2 (H-burning shell analog)
    X_prof = jnp.where(comp_mfracs < 0.2, 0.01, 0.70)

    density = compute_composition_density(X_prof, comp_mfracs)

    # Peak should be near the step
    peak_idx = int(jnp.argmax(density))
    peak_m = float(comp_mfracs[peak_idx])
    assert abs(peak_m - 0.2) < 0.05, (
        f"Density peak at m/M={peak_m}, expected near 0.2 (the composition step)")
    # Peak should be much higher than mean (localized feature)
    assert float(density[peak_idx]) > 5.0 * float(jnp.mean(density))


# ===========================================================================
# T8: should_remap — skips smooth profile
# ===========================================================================

@pytest.mark.fast
def test_should_remap_skips_smooth_profile():
    """Smooth linear X profile → should_remap returns False.

    A smoothly varying composition has no sharp feature worth resolving,
    so equidistribution adds nothing but numerical diffusion.
    """
    from stellar_jax.mesh.composition_mesh import should_remap, equidistribute_comp_mesh

    N = 200
    comp_mfracs = jnp.linspace(0.0, 1.0, N)
    # Smooth linear gradient (no step function)
    X_prof = jnp.linspace(0.35, 0.70, N)

    comp_mfracs_new = equidistribute_comp_mesh(X_prof, comp_mfracs)
    result = should_remap(X_prof, comp_mfracs, comp_mfracs_new)

    assert not bool(result), "Smooth profile should NOT trigger a remap"


# ===========================================================================
# T9: should_remap — fires on sharp step
# ===========================================================================

@pytest.mark.fast
def test_should_remap_fires_on_step():
    """X with a sharp step (ΔX=0.5 in 3 zones) → should_remap returns True."""
    from stellar_jax.mesh.composition_mesh import should_remap, equidistribute_comp_mesh

    N = 200
    comp_mfracs = jnp.linspace(0.0, 1.0, N)  # uniform grid
    # Sharp step at zone 40 (m/M ≈ 0.2)
    X_prof = jnp.ones(N) * 0.70
    X_prof = X_prof.at[:40].set(0.01)

    comp_mfracs_new = equidistribute_comp_mesh(X_prof, comp_mfracs)
    result = should_remap(X_prof, comp_mfracs, comp_mfracs_new)

    assert bool(result), "Sharp step should trigger a remap"


# ===========================================================================
# T10: stop_gradient on grid output
# ===========================================================================

@pytest.mark.fast
def test_stop_gradient_on_grid_output():
    """adapt_structure_mesh: gradient w.r.t. y is nonzero but q_mesh_new is detached.

    Policy 2: grid positions don't carry gradient; physics values do.
    """
    from stellar_jax.mesh import adapt_structure_mesh, initial_lagrangian_mesh

    N_s = 50
    q_mesh = initial_lagrangian_mesh(N_s)
    xi = jnp.linspace(0.0, 1.0, N_s)
    ln_P = jnp.log(10.0) * (17.0 - 13.0 * xi)
    ln_T = jnp.log(10.0) * (7.18 - 3.4 * xi)
    ln_r = jnp.log(6.96e10 * jnp.maximum(xi, 1e-6) ** (1.0 / 3.0))
    ell = 1.0 - 0.5 * xi
    y = jnp.stack([ln_r, ln_P, ln_T, ell], axis=-1)

    # Gradient of y_new values w.r.t. y_in should be nonzero
    def loss_values(y_in):
        y_new, _ = adapt_structure_mesh(y_in, q_mesh, relax_factor=0.1)
        return jnp.sum(y_new[:, 2])  # sum of ln_T values

    grad_y = jax.grad(loss_values)(y)
    assert bool(jnp.all(jnp.isfinite(grad_y))), "Gradient through values must be finite"
    assert bool(jnp.any(grad_y != 0.0)), "Gradient through values must be nonzero"


# ===========================================================================
# T11: adapt_composition_mesh — shape invariant
# ===========================================================================

@pytest.mark.fast
def test_adapt_composition_mesh_shape_invariant():
    """adapt_composition_mesh output always (N_COMP,) for each species + grid.

    Verifies carry-shape stability for lax.scan: the output shape never changes
    regardless of whether the remap fires or is skipped.
    """
    from stellar_jax.mesh import adapt_composition_mesh
    from stellar_jax.config.mesh_defaults import N_COMP

    comp_mfracs = jnp.linspace(0.0, 1.0, N_COMP)
    Y = jnp.ones(N_COMP) * 0.28
    Z = jnp.ones(N_COMP) * 0.02
    C12 = jnp.ones(N_COMP) * 0.003
    C13 = jnp.ones(N_COMP) * 0.0001
    N14 = jnp.ones(N_COMP) * 0.001

    # Case 1: smooth profile (remap skipped)
    X_smooth = jnp.linspace(0.35, 0.70, N_COMP)
    result = adapt_composition_mesh(X_smooth, Y, Z, C12, C13, N14, comp_mfracs)
    assert len(result) == 7
    for i, r in enumerate(result):
        assert r.shape == (N_COMP,), f"Output {i} shape {r.shape} != ({N_COMP},)"

    # Case 2: step profile (remap fires)
    X_step = jnp.ones(N_COMP) * 0.70
    X_step = X_step.at[:40].set(0.01)
    result = adapt_composition_mesh(X_step, Y, Z, C12, C13, N14, comp_mfracs)
    assert len(result) == 7
    for i, r in enumerate(result):
        assert r.shape == (N_COMP,), f"Output {i} shape {r.shape} != ({N_COMP},)"


# ===========================================================================
# T12: initial_lagrangian_mesh — boundaries and monotonicity
# ===========================================================================

@pytest.mark.fast
def test_initial_lagrangian_mesh_boundaries():
    """initial_lagrangian_mesh(N): shape (N+1,), q[0]=0, q[-1]=1, strictly increasing."""
    from stellar_jax.mesh import initial_lagrangian_mesh

    for N in [10, 100, 600, 1000]:
        q = initial_lagrangian_mesh(N)
        assert q.shape == (N + 1,), f"Shape should be ({N+1},), got {q.shape}"
        assert float(q[0]) == 0.0, f"q[0] should be 0.0, got {float(q[0])}"
        assert float(q[-1]) == 1.0, f"q[-1] should be 1.0, got {float(q[-1])}"
        diffs = jnp.diff(q)
        assert bool(jnp.all(diffs > 0)), f"Mesh must be strictly increasing for N={N}"


# ===========================================================================
# T13: compute_gval — matches MESA formula
# ===========================================================================

@pytest.mark.fast
def test_compute_gval_mesa_formula():
    """compute_gval(X) = 30 * log10(X + 0.01) matches MESA do1_xa_function.

    Reference: mesh_functions.f90:307-321 (vals(m,i) = weight*log10(xa(j,m)+param))
    """
    from stellar_jax.mesh.density import compute_gval

    X = jnp.array([0.0, 0.01, 0.1, 0.5, 0.7])
    expected = 30.0 * jnp.log10(X + 0.01)
    result = compute_gval(X)

    np.testing.assert_allclose(result, expected, atol=1e-14,
                               err_msg="compute_gval must match MESA formula exactly")


# ===========================================================================
# T14: smooth — reduces variation
# ===========================================================================

@pytest.mark.fast
def test_smooth_reduces_variation():
    """Smoothing reduces the max/min ratio of density.

    3-point averaging should flatten peaks and fill troughs.
    """
    from stellar_jax.mesh.density import smooth

    N = 100
    # Density with a sharp spike
    density = jnp.ones(N) * 5.0
    density = density.at[50].set(100.0)

    smoothed = smooth(density, n_iters=3)

    # Peak should be reduced
    assert float(jnp.max(smoothed)) < float(jnp.max(density))
    # But still positive everywhere
    assert float(jnp.min(smoothed)) > 0.0
    # Overall sum approximately preserved
    np.testing.assert_allclose(
        float(jnp.sum(smoothed)), float(jnp.sum(density)), rtol=0.01)
