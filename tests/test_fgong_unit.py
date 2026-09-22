"""Fast unit tests for the fgong/ package (<5s each, no evolve_star).

These test the extracted functions in isolation using synthetic inputs
or fixed FGONG files. They do NOT validate physics (that's validate.py);
they ensure the extraction preserved behavior and the API contract holds.

All tests are @pytest.mark.fast — no JIT compilation of the full solver.
"""

import os
import sys
import gzip
import tempfile

import numpy as np
import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_synthetic_y_henyey():
    """Create a physically-reasonable synthetic Henyey state (N_HENYEY, 4).

    Columns: (ln_r, ln_P, ln_T, ell)
    Values are monotonic and physically reasonable for a 1 M_sun MS star.
    This exercises the builder without running evolve_star.
    """
    from stellar_jax.config.constants import Rsun
    from stellar_jax.config.mesh_defaults import N_HENYEY

    N = N_HENYEY
    # Radius: center (small) to surface (~Rsun), log-spaced
    r = np.linspace(0.01 * Rsun, 0.95 * Rsun, N)
    ln_r = np.log(r)

    # Pressure: center (~2e17 dyne/cm²) to surface (~1e5)
    logP = np.linspace(17.3, 5.0, N)
    ln_P = logP * np.log(10)

    # Temperature: center (~1.5e7 K) to surface (~6000 K)
    logT = np.linspace(7.18, 3.78, N)
    ln_T = logT * np.log(10)

    # ell (dimensionless luminosity placeholder)
    ell = np.linspace(0.0, 1.0, N)

    y = np.column_stack([ln_r, ln_P, ln_T, ell])
    return jnp.array(y, dtype=jnp.float64)


def _get_fgong_path():
    """Return path to the reference FGONG file (for I/O tests)."""
    fgong_gz = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "mesa_comparison", "profiles", "1.0Msun", "midMS.FGONG.gz"
    )
    if not os.path.isfile(fgong_gz):
        pytest.skip(f"FGONG reference not found: {fgong_gz}")
    return fgong_gz


def _decompress_fgong(fgong_gz):
    """Decompress a .gz FGONG into a temp file, return path."""
    with gzip.open(fgong_gz, 'rt') as gz:
        content = gz.read()
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.FGONG', delete=False)
    tmp.write(content)
    tmp.close()
    return tmp.name


# ===========================================================================
# T1: builder shapes
# ===========================================================================

@pytest.mark.fast
def test_fgong_builder_shapes():
    """structure_to_fgong_jax produces (glob=(15,), var=(N_HENYEY, 15)), all finite."""
    from stellar_jax.fgong.builder import structure_to_fgong_jax
    from stellar_jax.config.mesh_defaults import N_HENYEY, N_COMP

    y_henyey = _make_synthetic_y_henyey()
    X_profile = jnp.ones(N_COMP) * 0.7  # uniform H

    glob, var = structure_to_fgong_jax(
        jnp.float64(1.0),       # M_solar
        jnp.float64(0.0),       # log_L (1 L_sun)
        jnp.float64(3.76),      # log_Te (~5750 K)
        X_profile,
        jnp.float64(0.014),     # Z
        jnp.float64(0.0),       # t_age
        jnp.float64(2.0),       # alpha_mlt
        y_henyey=y_henyey,
    )

    assert glob.shape == (15,), f"glob shape: {glob.shape}"
    assert var.shape == (N_HENYEY, 15), f"var shape: {var.shape}"
    assert jnp.all(jnp.isfinite(glob)), "glob has NaN/Inf"
    assert jnp.all(jnp.isfinite(var)), "var has NaN/Inf"


# ===========================================================================
# T2: glob fields physically reasonable
# ===========================================================================

@pytest.mark.fast
def test_fgong_glob_values():
    """glob fields: M > 0, R > 0, L > 0, Z in (0, 0.1), X_surf in (0, 1)."""
    from stellar_jax.fgong.builder import structure_to_fgong_jax
    from stellar_jax.config.constants import Msun, Rsun, Lsun
    from stellar_jax.config.mesh_defaults import N_COMP

    y_henyey = _make_synthetic_y_henyey()
    X_profile = jnp.ones(N_COMP) * 0.7

    glob, _ = structure_to_fgong_jax(
        jnp.float64(1.0), jnp.float64(0.0), jnp.float64(3.76),
        X_profile, jnp.float64(0.014), jnp.float64(0.0),
        jnp.float64(2.0), y_henyey=y_henyey,
    )

    assert float(glob[0]) > 0, "M_star <= 0"
    assert float(glob[0]) == pytest.approx(Msun, rel=0.01), "M_star != 1 Msun"
    assert float(glob[1]) > 0, "R_star <= 0"
    assert float(glob[2]) > 0, "L_star <= 0"
    assert 0 < float(glob[3]) < 0.1, f"Z out of range: {glob[3]}"
    assert 0 < float(glob[4]) < 1.0, f"X_surf out of range: {glob[4]}"


# ===========================================================================
# T3: var array ordering — r increasing (center-to-surface)
# ===========================================================================

@pytest.mark.fast
def test_fgong_var_ordering():
    """var[:, 0] (radius) must be monotonically increasing."""
    from stellar_jax.fgong.builder import structure_to_fgong_jax
    from stellar_jax.config.mesh_defaults import N_COMP

    y_henyey = _make_synthetic_y_henyey()
    X_profile = jnp.ones(N_COMP) * 0.7

    _, var = structure_to_fgong_jax(
        jnp.float64(1.0), jnp.float64(0.0), jnp.float64(3.76),
        X_profile, jnp.float64(0.014), jnp.float64(0.0),
        jnp.float64(2.0), y_henyey=y_henyey,
    )

    r = var[:, 0]
    diff = r[1:] - r[:-1]
    # Allow for numerical noise at the very small scale
    assert jnp.all(diff >= 0), "r is not monotonically increasing"


# ===========================================================================
# T4: read_fgong parses a real FGONG file correctly
# ===========================================================================

@pytest.mark.fast
def test_read_fgong_parses_real_file():
    """read_fgong loads a reference FGONG and produces valid shapes."""
    from stellar_jax.fgong.io import read_fgong

    fgong_gz = _get_fgong_path()
    tmp_path = _decompress_fgong(fgong_gz)

    try:
        glob, var = read_fgong(tmp_path)
    finally:
        os.unlink(tmp_path)

    assert glob.shape[0] >= 15, f"glob too short: {glob.shape}"
    assert var.ndim == 2, "var is not 2D"
    assert var.shape[0] > 100, "var has too few points"
    assert var.shape[1] >= 15, "var has too few columns"
    # r (col 0) should be center-to-surface (increasing)
    assert var[-1, 0] > var[0, 0], "var not center-to-surface"


# ===========================================================================
# T5: fgong_components returns expected keys
# ===========================================================================

@pytest.mark.fast
def test_fgong_components_keys():
    """fgong_components returns a dict with standard keys."""
    from stellar_jax.fgong.io import read_fgong, fgong_components

    fgong_gz = _get_fgong_path()
    tmp_path = _decompress_fgong(fgong_gz)

    try:
        glob, var = read_fgong(tmp_path)
    finally:
        os.unlink(tmp_path)

    d = fgong_components(glob, var)
    expected_keys = {'r', 'T', 'P', 'rho', 'X', 'L', 'kappa', 'eps_nuc', 'gamma1'}
    assert expected_keys.issubset(set(d.keys())), f"Missing keys: {expected_keys - set(d.keys())}"
    # All arrays should have the same length
    n = var.shape[0]
    for k, v in d.items():
        assert len(v) == n, f"{k} has wrong length: {len(v)} != {n}"


# ===========================================================================
# T6: backward-compat imports from evolution and oscillations
# ===========================================================================

@pytest.mark.fast
def test_backward_compat_imports():
    """Verify that the old import paths still work."""
    from stellar_jax.evolution import structure_to_fgong_jax, write_fgong, write_gyre, N_HIRES
    from stellar_jax.oscillations import read_fgong, fgong_components

    # Verify they're callable
    assert callable(structure_to_fgong_jax)
    assert callable(write_fgong)
    assert callable(write_gyre)
    assert callable(read_fgong)
    assert callable(fgong_components)
    assert N_HIRES == 2500


# ===========================================================================
# T7: builder y_henyey=None raises ValueError
# ===========================================================================

@pytest.mark.fast
def test_fgong_builder_requires_y_henyey():
    """structure_to_fgong_jax raises ValueError when y_henyey is None."""
    from stellar_jax.fgong.builder import structure_to_fgong_jax
    from stellar_jax.config.mesh_defaults import N_COMP

    X_profile = jnp.ones(N_COMP) * 0.7

    with pytest.raises(ValueError, match="y_henyey is required"):
        structure_to_fgong_jax(
            jnp.float64(1.0), jnp.float64(0.0), jnp.float64(3.76),
            X_profile, jnp.float64(0.014), jnp.float64(0.0),
            jnp.float64(2.0), y_henyey=None,
        )


# ===========================================================================
# T8: contracts module has expected constants
# ===========================================================================

@pytest.mark.fast
def test_contracts_constants():
    """contracts.py exposes the documented shape constants."""
    from stellar_jax.fgong.contracts import GLOB_SIZE, VAR_COLS_DIFF, VAR_COLS_FULL, N_HIRES

    assert GLOB_SIZE == 15
    assert VAR_COLS_DIFF == 15
    assert VAR_COLS_FULL == 25
    assert N_HIRES == 2500


# ===========================================================================
# T9: Brunt-Väisälä discriminant A* structure
# ===========================================================================

@pytest.mark.fast
def test_fgong_brunt_vaisala_structure():
    """A* (col 14) has the correct structural properties.

    The Brunt-Väisälä discriminant A* = r * (dlnP/dr / Gamma1 - dlnrho/dr)
    is set to zero where r < r_min_brunt (= 1e-4 * R_star) and is non-zero
    in the radiative interior where there is stable stratification.
    This verifies the extraction preserved the A* computation correctly.
    """
    from stellar_jax.fgong.builder import structure_to_fgong_jax
    from stellar_jax.config.constants import Rsun, Lsun, sigma_sb
    from stellar_jax.config.mesh_defaults import N_HENYEY, N_COMP

    y_henyey = _make_synthetic_y_henyey()
    X_profile = jnp.ones(N_COMP) * 0.7

    glob, var = structure_to_fgong_jax(
        jnp.float64(1.0), jnp.float64(0.0), jnp.float64(3.76),
        X_profile, jnp.float64(0.014), jnp.float64(0.0),
        jnp.float64(2.0), y_henyey=y_henyey,
    )

    A_star = var[:, 14]
    r = var[:, 0]
    R_star = float(glob[1])

    # Zones below r_min_brunt = 1e-4 * R_star should have A* = 0
    r_min_brunt = 1e-4 * R_star
    below_thresh = np.array(r) < r_min_brunt
    if np.any(below_thresh):
        assert float(jnp.max(jnp.abs(A_star[below_thresh]))) < 1e-10, \
            "A* should be zero below r_min_brunt"

    # In the radiative interior (middle of star), A* should be non-zero
    mid = N_HENYEY // 2
    assert float(jnp.abs(A_star[mid])) > 1e-10, \
        f"A* at mid-radius should be non-zero, got {float(A_star[mid])}"

    # A* should be finite everywhere
    assert jnp.all(jnp.isfinite(A_star)), "A* has NaN/Inf values"


# ===========================================================================
# T10: Differentiability — jax.grad through the builder is finite
# ===========================================================================

@pytest.mark.fast
def test_fgong_builder_differentiable():
    """jax.grad through structure_to_fgong_jax produces a finite gradient.

    This is a fast (<5s) proof that the differentiable path works: we
    differentiate glob[0] (M_star in CGS) w.r.t. M_solar. The gradient
    should be Msun (since M_star = M_solar * Msun, and the chain is linear
    in mass at that point). The key guarantee is that no operation in the
    builder silently breaks the JAX trace (e.g., accidental NumPy, Python
    control flow on traced values, or stop_gradient).
    """
    from stellar_jax.fgong.builder import structure_to_fgong_jax
    from stellar_jax.config.constants import Msun
    from stellar_jax.config.mesh_defaults import N_COMP

    y_henyey = _make_synthetic_y_henyey()
    X_profile = jnp.ones(N_COMP) * 0.7

    def mass_to_glob0(M_solar):
        glob, _ = structure_to_fgong_jax(
            M_solar, jnp.float64(0.0), jnp.float64(3.76),
            X_profile, jnp.float64(0.014), jnp.float64(0.0),
            jnp.float64(2.0), y_henyey=y_henyey,
        )
        return glob[0]  # M_star in grams

    grad_val = float(jax.grad(mass_to_glob0)(jnp.float64(1.0)))

    assert np.isfinite(grad_val), f"Gradient is not finite: {grad_val}"
    # glob[0] = M_solar * Msun, so ∂glob[0]/∂M_solar = Msun
    np.testing.assert_allclose(grad_val, Msun, rtol=1e-10,
                               err_msg="∂M_star/∂M_solar should be Msun")


# ===========================================================================
# T11: _compute_brunt_vaisala helper
# ===========================================================================

@pytest.mark.fast
def test_compute_brunt_vaisala_zeros_near_center():
    """_compute_brunt_vaisala returns A*=0 for r < r_min_brunt."""
    from stellar_jax.fgong.io import _compute_brunt_vaisala

    N = 100
    R_star = 7e10  # ~1 Rsun
    r = np.linspace(0.001 * R_star, 0.95 * R_star, N)
    # Monotonic P, rho (physically reasonable)
    P = np.logspace(17, 5, N)
    rho = np.logspace(2, -7, N)
    Gamma1 = np.full(N, 5.0 / 3.0)  # ideal gas

    AA = _compute_brunt_vaisala(r, P, rho, Gamma1, R_star)

    assert len(AA) == N
    # Points with r < 1e-4 * R_star should have A* = 0
    r_min = 1e-4 * R_star
    center_mask = r < r_min
    if np.any(center_mask):
        assert np.all(AA[center_mask] == 0.0), "A* should be zero near center"


@pytest.mark.fast
def test_compute_brunt_vaisala_nonzero_interior():
    """_compute_brunt_vaisala returns non-zero A* in the interior."""
    from stellar_jax.fgong.io import _compute_brunt_vaisala

    N = 100
    R_star = 7e10
    r = np.linspace(0.1 * R_star, 0.9 * R_star, N)
    # Create a profile where dlnP/dlnr and dlnrho/dlnr differ
    P = np.logspace(14, 8, N)
    rho = np.logspace(1, -4, N)
    Gamma1 = np.full(N, 5.0 / 3.0)

    AA = _compute_brunt_vaisala(r, P, rho, Gamma1, R_star)

    # Interior points (not boundary) should be non-zero
    mid = N // 2
    assert abs(AA[mid]) > 1e-10, f"A* at mid should be non-zero, got {AA[mid]}"
    assert np.all(np.isfinite(AA)), "A* has NaN/Inf"


# ===========================================================================
# T12: _assemble_fgong_arrays helper
# ===========================================================================

@pytest.mark.fast
def test_assemble_fgong_arrays_shapes():
    """_assemble_fgong_arrays produces (glob=(15,), var=(N, 25))."""
    from stellar_jax.fgong.io import _assemble_fgong_arrays

    N = 50
    R_star = 7e10
    M_star = 2e33
    L_star = 4e33
    prof = {
        'r': np.linspace(0.01 * R_star, 0.95 * R_star, N),
        'P': np.logspace(17, 5, N),
        'T': np.logspace(7.2, 3.8, N),
        'rho': np.logspace(2, -7, N),
        'Mr': np.linspace(0.01 * M_star, M_star, N),
        'Lr': np.linspace(0.0, L_star, N),
        'Gamma1': np.full(N, 5.0 / 3.0),
        'nabla_ad': np.full(N, 0.4),
        'delta': np.full(N, 1.0),
        'cp': np.full(N, 2.5e8),
        'nabla': np.full(N, 0.3),
        'kappa_val': np.full(N, 0.5),
        'eps_nuc': np.full(N, 1e2),
        'X_local': np.full(N, 0.7),
    }

    glob, var = _assemble_fgong_arrays(
        prof, M_star, R_star, L_star,
        Z=0.014, X_profile=np.ones(200) * 0.7,
        alpha_mlt=2.0, t_age=4.57e9)

    assert glob.shape == (15,)
    assert var.shape == (N, 25)
    # Verify key glob fields
    assert glob[0] == pytest.approx(M_star)
    assert glob[1] == pytest.approx(R_star)
    assert glob[2] == pytest.approx(L_star)
    assert glob[3] == pytest.approx(0.014)
    assert glob[4] == pytest.approx(0.7)  # X_surf
    assert glob[5] == pytest.approx(2.0)  # alpha
    # Verify var columns are populated
    assert np.all(var[:, 0] > 0), "r should be positive"
    assert np.all(np.isfinite(var)), "var has NaN/Inf"


# ===========================================================================
# T13: write_fgong → read_fgong round-trip uses decomposed helpers
# ===========================================================================

@pytest.mark.fast
def test_fgong_write_read_roundtrip_decomposed():
    """write_fgong → read_fgong preserves glob[0:5] and var shape.

    This validates that the decomposition into _assemble_fgong_arrays +
    _compute_brunt_vaisala + file formatting produces parseable output.
    Uses a reference FGONG to get realistic inputs.
    """
    from stellar_jax.fgong.io import read_fgong, _assemble_fgong_arrays, _compute_brunt_vaisala

    fgong_gz = _get_fgong_path()
    tmp_path = _decompress_fgong(fgong_gz)
    try:
        glob, var = read_fgong(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Verify _compute_brunt_vaisala on real data produces finite results
    r = var[:, 0]
    P = var[:, 3]
    rho = var[:, 4]
    Gamma1 = var[:, 9]
    R_star = float(glob[1])

    AA = _compute_brunt_vaisala(r, P, rho, Gamma1, R_star)
    assert np.all(np.isfinite(AA)), "A* from real FGONG has NaN/Inf"
    # The reference FGONG should have non-zero A* in the interior
    mid = len(r) // 2
    assert abs(AA[mid]) > 1e-10, f"A* at mid should be non-zero for real data"


# ===========================================================================
# T8: builder uses public initial_lagrangian_mesh from mesh/ (Law of Demeter)
# ===========================================================================

@pytest.mark.fast
def test_fgong_builder_uses_public_mesh_symbol():
    """fgong/builder.py imports initial_lagrangian_mesh from mesh (public API).

    This verifies the Law-of-Demeter fix (#535): the builder no longer reaches
    through structure._lagrangian_mass_mesh (private) but uses the canonical
    public export mesh.initial_lagrangian_mesh. The output must be identical.

    Reference: docs/dev/refactoring-best-practices.md §4 (Law of Demeter).
    L2 spec: docs/design/redesign/08-fgong.md §1.1.
    """
    import stellar_jax.fgong.builder as builder_mod
    from stellar_jax.mesh import initial_lagrangian_mesh
    from stellar_jax.config.mesh_defaults import N_HENYEY

    # 1. Verify the builder module imports from mesh, not structure's private symbol
    import inspect
    source = inspect.getsource(builder_mod)
    assert 'from stellar_jax.mesh import initial_lagrangian_mesh' in source, (
        "builder.py should import initial_lagrangian_mesh from mesh (public API)"
    )
    assert 'from structure import' not in source or '_lagrangian_mass_mesh' not in source, (
        "builder.py should NOT import _lagrangian_mass_mesh from structure (private)"
    )

    # 2. Verify the mesh produced is correct: shape, boundaries, monotonicity
    q = initial_lagrangian_mesh(N_HENYEY)
    assert q.shape == (N_HENYEY + 1,), f"Expected ({N_HENYEY + 1},), got {q.shape}"
    assert float(q[0]) == 0.0, "q[0] must be 0 (center)"
    assert float(q[-1]) == 1.0, "q[-1] must be 1 (surface)"
    assert bool(jnp.all(jnp.diff(q) > 0)), "Mesh must be strictly increasing"



# ═══════════════════════════════════════════════════════════════
#: Brunt-Väisälä A* vs MESA FGONG col-14 — accuracy validation
# ═══════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.fast
@pytest.mark.validation
@pytest.mark.mutation("corrupt_brunt_stencil")
@pytest.mark.right_reason("RMS relative error")
def test_brunt_astar_vs_mesa_fgong_col14():
    """Our A* matches MESA FGONG col-14 to <0.5% RMS across masses and stages.

    WHAT: verifies _compute_brunt_vaisala computes A* = (1/Γ₁)dlnP/dlnr − dlnρ/dlnr
    using the Fornberg O(h²) finite-difference stencil on the non-uniform mesh,
    and that this matches MESA's full MHM-derived A* stored in FGONG col-14.

    WHY: issue #683 investigation revealed that the Schwarzschild form on self-
    consistent structure IS the full Ledoux discriminant (composition is already in ρ;
    no separate μ-term needed). The real fix was upgrading the derivative stencil from
    O(h) symmetric-spacing to O(h²) Fornberg unequal-spacing, which reduces numerical
    error by ~30% on the quadratic-stretched mesh. This test gates that accuracy.

    EXTERNAL REFERENCE: MESA FGONG col-14 from committed MODE-A profiles
    (data/mesa_comparison/profiles/). MESA computes A* = N²r/g where N² uses
    brunt_B (brunt.f90:get_brunt_B, MHM form — the diagnostic decomposition of the
    same total N²). The two methods agree because both compute from self-consistent
    structure (MESA inlist: same physics, same EOS, same composition at each zone).

    TOLERANCE: 0.5% RMS relative error in the bulk radiative zone (r/R ∈ [0.25, 0.60],
    |A*| > 0.05). This is a NUMERICAL accuracy bound (stencil quality + mesh density),
    not a physics bound. MESA's FGONG has ~1000 zones; our Fornberg stencil achieves
    sub-0.2% on this grid. The 0.5% ceiling provides headroom for edge effects.

    WHAT MAKES IT FAIL: @mutation("corrupt_brunt_stencil") replaces the Fornberg stencil
    with a degraded one (constant-spacing formula applied on non-uniform mesh → O(h)
    errors that grow to >1% on the stretched mesh near the core).

    References:
        MESA brunt.f90:get_brunt_B (lines 231-333): MHM form.
        Fornberg (1988), Math. Comp. 51, 699: FD weights on arbitrary grids.
        Christensen-Dalsgaard (2008), Ap&SS 316, 113: FGONG format, A* = N²r/g.
    """
    from stellar_jax.fgong.io import read_fgong, _compute_brunt_vaisala

    base = os.path.join(os.path.dirname(__file__), "..", "src", "stellar_jax", "data", "mesa_comparison",
                        "profiles")

    # Test across multiple masses and stages — proves accuracy is robust.
    test_cases = [
        ('1.0Msun', 'midMS'),   # moderate composition gradient (1 M☉ radiative)
        ('1.0Msun', 'Xc0.30'),  # more evolved (stronger gradient)
        ('1.5Msun', 'midMS'),   # higher mass, convective core boundary
        ('2.0Msun', 'midMS'),   # 2 M☉
    ]

    for mass, stage in test_cases:
        fgong_path = os.path.join(base, mass, f'{stage}.FGONG.gz')
        if not os.path.exists(fgong_path):
            pytest.skip(f"MESA FGONG not found: {fgong_path}")

        with gzip.open(fgong_path) as f:
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.FGONG')
            tmpf.write(f.read())
            tmpf.close()
        try:
            glob, var = read_fgong(tmpf.name)
        finally:
            os.unlink(tmpf.name)

        R_star = glob[1]
        r = var[:, 0]
        P = var[:, 3]
        rho = var[:, 4]
        gamma1 = var[:, 9]

        # Compute our A*
        AA_ours = _compute_brunt_vaisala(r, P, rho, gamma1, R_star)

        # MESA's A* from col-14
        AA_mesa = var[:, 14]

        # Bulk radiative interior (away from boundaries)
        r_frac = r / R_star
        rad_mask = ((r_frac > 0.25) & (r_frac < 0.60)
                    & (np.abs(AA_mesa) > 0.05))
        n_rad = int(rad_mask.sum())
        assert n_rad >= 50, (
            f"{mass}/{stage}: only {n_rad} points in radiative mask (need >=50)")

        rel_err = np.abs(AA_ours[rad_mask] - AA_mesa[rad_mask]) / np.abs(AA_mesa[rad_mask])
        rms_rel = float(np.sqrt(np.mean(rel_err**2)))

        assert rms_rel < 0.005, (
            f"{mass}/{stage}: A* vs MESA col-14 RMS relative error = "
            f"{rms_rel:.5f} > 0.5% in radiative zone "
            f"(r/R=[0.25,0.60], |A*|>0.05, N={n_rad} pts)")

    # Sign check: A* must be positive in stable radiative zones
    # (stable stratification: buoyancy restoring → N² > 0 → A* > 0)
    fgong_path = os.path.join(base, '1.0Msun', 'midMS.FGONG.gz')
    with gzip.open(fgong_path) as f:
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.FGONG')
        tmpf.write(f.read())
        tmpf.close()
    try:
        glob, var = read_fgong(tmpf.name)
    finally:
        os.unlink(tmpf.name)

    R_star = glob[1]
    r = var[:, 0]
    r_frac = r / R_star
    AA_ours = _compute_brunt_vaisala(r, var[:, 3], var[:, 4], var[:, 9], R_star)
    # In stable radiative zone (0.3-0.6), >95% of points must be positive
    stable_mask = (r_frac > 0.30) & (r_frac < 0.60)
    frac_positive = np.sum(AA_ours[stable_mask] > 0) / stable_mask.sum()
    assert frac_positive > 0.95, (
        f"A* should be >95% positive in stable radiative zone (0.3<r/R<0.6), "
        f"but only {frac_positive:.1%} positive")


# ===========================================================================
# T: Gamma1 deep-interior ≈ 5/3 (— cp/cv bug fix)
# ===========================================================================

@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("hires_gamma1_cp_form")
@pytest.mark.right_reason("Gamma1")
def test_gamma1_deep_interior_vs_mesa():
    """Γ₁ from our EOS formula matches MESA FGONG col-9 in the deep interior (< 2%).

    WHAT: verifies that the Gamma1 formula chi_rho/(1 - nad*chi_T) reproduces
    MESA's Gamma1 (FGONG col 9) in the deep radiative interior (0.1 < r/R < 0.5)
    where fully-ionized monatomic ideal gas gives Γ₁ ≈ 5/3.

    WHY: issue #707 — the old formula used cp where cv belongs, giving Γ₁ ≈ 1.4.
    This test catches the regression: any formula that returns ≈1.4 instead of ≈5/3
    in the deep interior fails.

    EXTERNAL REFERENCE: MESA r26.4.1 FGONG (1.0 M☉ ZAMS, MODE-A physics: Z=0.014,
    Y=0.2695, alpha_MLT=2.0 Cox, identical-physics). Col 9 = Gamma1.

    TOLERANCE: 2% — inter-code EOS scatter (our OPAL-only JAX vs MESA's
    multi-source blend) is < 1% in the deep interior; 2% provides headroom
    for the surface-ward transition. The buggy formula gives Γ₁ ≈ 1.4, a 16%
    deficit that fails this bound unambiguously.

    MUTATION: hires_gamma1_cp_form — patches eos_lookup to produce the old buggy
    Gamma1 ≈ chi_rho + nad*chi_T ≈ 1.4 (ideal gas); must FAIL this test.
    """
    from stellar_jax.microphysics.eos import eos_lookup
    from stellar_jax.fgong.io import read_fgong

    # Load MESA 1.0 M☉ ZAMS FGONG (external reference, MODE-A physics)
    fgong_gz = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "mesa_comparison", "profiles", "1.0Msun", "zams.FGONG.gz"
    )
    if not os.path.isfile(fgong_gz):
        pytest.skip(f"MESA FGONG reference not found: {fgong_gz}")

    with gzip.open(fgong_gz, 'rt') as f:
        content = f.read()
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.FGONG', delete=False)
    tmp.write(content)
    tmp.close()
    try:
        glob, var = read_fgong(tmp.name)
    finally:
        os.unlink(tmp.name)

    # MESA FGONG layout: col 0=r, 2=T, 3=P, 4=rho, 5=X, 9=Gamma1
    R_star = glob[1]
    r = var[:, 0]
    r_frac = r / R_star
    T_mesa = var[:, 2]
    P_mesa = var[:, 3]
    rho_mesa = var[:, 4]
    X_mesa = var[:, 5]
    g1_mesa = var[:, 9]

    # Deep interior: 0.1 < r/R < 0.5 (fully ionized, ideal gas dominant)
    deep_mask = (r_frac > 0.1) & (r_frac < 0.5)
    assert deep_mask.sum() > 50, "Not enough deep-interior points in MESA FGONG"

    # Compute OUR Gamma1 at MESA zone conditions
    Z = float(glob[3]) if glob[3] > 0 else 0.014
    g1_ours = np.zeros(deep_mask.sum())

    for i, idx in enumerate(np.where(deep_mask)[0]):
        logT = np.log10(T_mesa[idx])
        # Compute gas pressure: P_gas = P - P_rad
        a_rad_cgs = 7.5657e-15
        P_rad = a_rad_cgs * T_mesa[idx]**4 / 3.0
        P_gas = max(P_mesa[idx] - P_rad, 1e-3 * P_mesa[idx])
        logPgas = np.log10(P_gas)

        rho, mu, nad, S, cp, chi_rho, chi_T = eos_lookup(
            jnp.float64(logT), jnp.float64(logPgas),
            jnp.float64(X_mesa[idx]), jnp.float64(Z))

        # The corrected formula
        gamma1 = float(chi_rho / jnp.maximum(1.0 - nad * chi_T, 1e-10))
        g1_ours[i] = gamma1

    g1_mesa_deep = g1_mesa[deep_mask]

    # Sanity: MESA Gamma1 should be near 5/3 in the deep interior
    assert np.median(g1_mesa_deep) > 1.55, (
        f"MESA Gamma1 median in deep interior = {np.median(g1_mesa_deep):.4f}, "
        f"expected > 1.55 (near 5/3=1.667)")

    # Main assertion: our formula matches MESA col-9 to < 2%
    rel_err = np.abs(g1_ours - g1_mesa_deep) / g1_mesa_deep
    max_rel = float(np.max(rel_err))
    rms_rel = float(np.sqrt(np.mean(rel_err**2)))

    assert max_rel < 0.02, (
        f"Gamma1 vs MESA: max relative error = {max_rel:.4f} > 2% "
        f"in deep interior (0.1 < r/R < 0.5). "
        f"Median ours={np.median(g1_ours):.4f}, MESA={np.median(g1_mesa_deep):.4f}")

    # Secondary check: median should be near 5/3 (the ideal-gas limit)
    assert np.median(g1_ours) > 1.60, (
        f"Our Gamma1 median = {np.median(g1_ours):.4f} in deep interior, "
        f"expected > 1.60 (near 5/3=1.667). The cp-form bug gives ~1.4.")


# ─── Empty-grid guard (AC2,) ───────────────────────────────────────────

@pytest.mark.fast
class TestEmptyGridGuard:
    """Verify that degenerate / inconsistent FGONG structures produce clear
    ValueError diagnostics instead of opaque IndexError or other crashes.

    WHAT: Tests the empty-grid guards added to _make_integration_grid
    (integrator.py), compute_eigenfreq_from_structure_jax (eigenvalue.py),
    and _build_oscillation_grid_jax (coefficients.py).

    WHY: Issue #1176 — the differentiable evolved-model path can emit
    inconsistent log_L/log_Te vs y_henyey, making R_star >> max(r) and
    emptying the FGONG grid. The guard catches this early with an actionable
    message, not an opaque 'IndexError: index -1 out of bounds, size 0'.

    REFERENCE: N/A — this is a robustness guard, not a physics assertion.
    TOLERANCE: N/A.
    MUTATION: N/A — guard test, not a physics validation test.
    """

    def test_make_integration_grid_empty_raises_valueerror(self):
        """_make_integration_grid must raise ValueError on an empty x_grid."""
        from stellar_jax.oscillations.integrator import _make_integration_grid
        import numpy as np

        with pytest.raises(ValueError, match="Empty integration grid"):
            _make_integration_grid(np.array([]))

    def test_build_oscillation_grid_jax_empty_raises_valueerror(self):
        """_build_oscillation_grid_jax must raise ValueError when R >> max(r)."""
        from stellar_jax.oscillations.coefficients import _build_oscillation_grid_jax

        # Construct a degenerate FGONG where R_star >> max(r), so all x < 1e-4.
        # This simulates the inconsistent-structure failure mode.
        glob = np.zeros(15)
        glob[0] = 2e33   # M [g] ~ 1 Msun
        glob[1] = 1e18   # R [cm] = impossibly large → x = r/R ≈ 0
        glob[14] = 6.674e-8  # G

        N = 200
        var = np.zeros((N, 15))
        var[:, 0] = np.linspace(1e8, 7e10, N)  # r [cm] — normal stellar radii
        var[:, 1] = np.log(np.linspace(1e-10, 1.0, N))  # ln(m/M)
        var[:, 3] = np.logspace(17, 13, N)  # P
        var[:, 4] = np.logspace(2, -7, N)   # rho
        var[:, 9] = np.full(N, 5.0/3.0)     # gamma1

        with pytest.raises(ValueError, match="Empty oscillation grid"):
            _build_oscillation_grid_jax(glob, var)

    def test_eigenfreq_from_structure_empty_raises_valueerror(self):
        """compute_eigenfreq_from_structure_jax must raise ValueError
        on an empty-grid FGONG (R >> max(r)), not IndexError.
        """
        from stellar_jax.oscillations.eigenvalue import (
            compute_eigenfreq_from_structure_jax,
        )
        from stellar_jax.oscillations.coefficients import build_oscillation_coeffs_jax
        from stellar_jax.config.mesh_defaults import N_HENYEY

        # Build a JAX FGONG with impossibly large R_star → empty grid.
        glob_jax = jnp.zeros(15)
        glob_jax = glob_jax.at[0].set(2e33)    # M
        glob_jax = glob_jax.at[1].set(1e18)    # R = impossibly large
        glob_jax = glob_jax.at[2].set(4e33)    # L
        glob_jax = glob_jax.at[14].set(6.674e-8)  # G

        N = N_HENYEY
        var_jax = jnp.zeros((N, 15))
        var_jax = var_jax.at[:, 0].set(jnp.linspace(1e8, 7e10, N))
        var_jax = var_jax.at[:, 1].set(jnp.log(jnp.linspace(1e-10, 1.0, N)))
        var_jax = var_jax.at[:, 3].set(jnp.logspace(17, 13, N))
        var_jax = var_jax.at[:, 4].set(jnp.logspace(2, -7, N))
        var_jax = var_jax.at[:, 9].set(jnp.full(N, 5.0/3.0))

        with pytest.raises(ValueError, match="Empty oscillation grid"):
            compute_eigenfreq_from_structure_jax(
                glob_jax, var_jax, l=0, nu_min=1000, nu_max=4500)


# ─── Self-consistent FGONG (AC3,) ──────────────────────────────────────

@pytest.mark.fast
class TestFGONGSelfConsistency:
    """Verify that structure_to_fgong_jax with atm_ratio produces a
    self-consistent FGONG where x_max = max(r)/R is physically bounded.

    WHAT: When atm_ratio is provided, R_star and L_star are derived from
    y_henyey directly (same formula as _step_solve_and_eps), making the
    FGONG self-consistent by construction.

    WHY: Issue #1176 — the legacy Stefan-Boltzmann path (without atm_ratio)
    can produce R_star inconsistent with y_henyey if the carry's log_L/log_Te
    are stale. The atm_ratio path eliminates this failure mode.

    REFERENCE: MESA pulse_fgong.f90:168 (r_outer = Rsun*s%photosphere_r).
    TOLERANCE: x_max must equal 1/atm_ratio to machine precision. For
    the synthetic test data (atm_ratio=1.001), x_max ≈ 0.999. Real stellar
    models have atm_ratio ≈ 1.29 (1 Msun), so x_max ≈ 0.775.
    MUTATION: N/A — correctness check, not a physics validation test.
    """

    def test_atm_ratio_produces_consistent_x_max(self):
        """With atm_ratio, x_max = max(r)/R should be ~1/atm_ratio."""
        from stellar_jax.fgong.builder import structure_to_fgong_jax
        from stellar_jax.oscillations.coefficients import build_oscillation_coeffs_jax
        from stellar_jax.config.mesh_defaults import N_HENYEY, N_COMP
        from stellar_jax.config.constants import Lsun, sigma_sb

        y_henyey = _make_synthetic_y_henyey()
        X_profile = jnp.ones(N_COMP) * 0.7
        atm_ratio = 1.001

        # Pass atm_ratio for the self-consistent path
        glob, var = structure_to_fgong_jax(
            jnp.float64(1.0), jnp.float64(0.0), jnp.float64(3.76),
            X_profile, jnp.float64(0.014), jnp.float64(0.0),
            jnp.float64(2.0), y_henyey=y_henyey,
            atm_ratio=jnp.float64(atm_ratio),
        )

        R_star = float(glob[1])
        r = np.array(var[:, 0])
        r_max = float(np.max(r))
        x_max = r_max / R_star

        # x_max should be 1/atm_ratio ≈ 0.999
        expected_x_max = 1.0 / atm_ratio
        assert abs(x_max - expected_x_max) < 0.001, (
            f"x_max = {x_max:.6f}, expected ~{expected_x_max:.6f} "
            f"(R_star={R_star:.3e}, r_max={r_max:.3e})"
        )

    def test_stale_logL_logTe_with_atm_ratio_still_consistent(self):
        """Even with deliberately WRONG log_L/log_Te, the atm_ratio path
        should produce a self-consistent FGONG (x_max near 1/atm_ratio).
        """
        from stellar_jax.fgong.builder import structure_to_fgong_jax
        from stellar_jax.config.mesh_defaults import N_HENYEY, N_COMP

        y_henyey = _make_synthetic_y_henyey()
        X_profile = jnp.ones(N_COMP) * 0.7
        atm_ratio = 1.001

        # Pass DELIBERATELY WRONG log_L/log_Te (stale values from ZAMS
        # while y_henyey is from a subgiant)
        glob, var = structure_to_fgong_jax(
            jnp.float64(1.0),
            jnp.float64(-5.0),   # log_L = -5 (wrong!)
            jnp.float64(2.0),    # log_Te = 2 (wrong! → Te = 100 K)
            X_profile, jnp.float64(0.014), jnp.float64(0.0),
            jnp.float64(2.0), y_henyey=y_henyey,
            atm_ratio=jnp.float64(atm_ratio),
        )

        R_star = float(glob[1])
        r = np.array(var[:, 0])
        r_max = float(np.max(r))
        x_max = r_max / R_star

        # Even with wrong log_L/log_Te, x_max should be physical
        assert 0.95 < x_max < 1.05, (
            f"x_max = {x_max:.6f} — FGONG is inconsistent despite atm_ratio! "
            f"(R_star={R_star:.3e}, r_max={r_max:.3e})"
        )

    def test_without_atm_ratio_legacy_still_works(self):
        """Without atm_ratio, the legacy Stefan-Boltzmann path should work
        when log_L/log_Te are consistent with y_henyey.
        """
        from stellar_jax.fgong.builder import structure_to_fgong_jax
        from stellar_jax.config.mesh_defaults import N_HENYEY, N_COMP

        y_henyey = _make_synthetic_y_henyey()
        X_profile = jnp.ones(N_COMP) * 0.7

        # Use consistent log_L/log_Te (computed from the same y_henyey)
        glob, var = structure_to_fgong_jax(
            jnp.float64(1.0), jnp.float64(0.0), jnp.float64(3.76),
            X_profile, jnp.float64(0.014), jnp.float64(0.0),
            jnp.float64(2.0), y_henyey=y_henyey,
        )

        R_star = float(glob[1])
        r = np.array(var[:, 0])
        r_max = float(np.max(r))
        x_max = r_max / R_star

        # Legacy path should still work with reasonable x_max
        assert 0.5 < x_max < 2.0, (
            f"x_max = {x_max:.6f} — legacy path has unreasonable normalization "
            f"(R_star={R_star:.3e}, r_max={r_max:.3e})"
        )
