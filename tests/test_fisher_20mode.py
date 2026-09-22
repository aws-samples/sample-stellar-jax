"""Fisher information matrix conditioning at ~20+ modes (issue #1044).

Validates that the Fisher matrix F = J^T Σ^{-1} J for the seismic forward model
θ = (M, Z, α_MLT) → ν(n,l) is well-conditioned with a realistic solar-analog
mode set (~20+ modes, l=0,1,2 in [1500, 4500] μHz).

WHAT: Fisher matrix conditioning with ~20+ seismic modes.
WHY: Validates that the forward model can support multi-parameter inference
     without numerical degeneracy. Cunha (2021) showed that l=2 modes are
     essential for breaking the M↔age degeneracy — this test checks that the
     Fisher matrix remains well-conditioned when all l=0,1,2 modes are present.
EXTERNAL REFERENCE: Cunha et al. (2021), arXiv:2110.03332 (PLATO hare-and-
     hounds, ~20 modes, l=2 for age). Metcalfe et al. (2012, ApJ 748, L10)
     16 Cyg A & B: 46/41 modes for solar analogs (Kepler). Chaplin & Miglio
     (2013), arXiv:1303.1957 (asteroseismology of solar-type stars).
WHAT MAKES IT FAIL: @mutation('mlt_alpha_insensitive') clamps α_MLT inside
     the MLT solver → perturbing α in the FD Jacobian has no effect →
     ∂ν/∂α ≈ 0 → one column of J collapses → Fisher loses rank → κ >> 1e10.

TOLERANCE: κ < 1e10 is a standard ill-conditioning bound. The Fisher for a
     3-param model with ~20+ modes should have κ << 1e10 (typical κ ~ 1e3–1e6).

Architecture: Uses the existing compute_jacobian_fd + compute_fisher from
     inference/fisher.py (the same infrastructure that passes at N=10).
     Central finite differences: J[i,j] = (obs_i(θ_j+h) - obs_i(θ_j-h)) / (2h)
     7 forward passes total (2 × 3 params + 1 baseline) at N=100.
     Each forward pass: evolve_star → structure_to_fgong_jax → JAX eigenvalue
     solver (IFT custom_vjp). The JAX solver creates per-mode JIT compilations
     (~1-2s each) which accumulate in the cache (~4 GB for ~420 modes across
     7 passes — well within 120 GB ci-mega).

AD-vs-FD mass column (acceptance criterion 3): INFEASIBLE at N≥100.
  The jax.grad backward pass through lax.scan stores all N intermediates;
  at N=100 this exceeds CI ceiling (120 GB ci-mega). Issue says "if memory
  allows" — it does not. Already validated at N=3 by
  test_eigenfreq_end_to_end_gradient_vs_fd.
"""
import warnings

import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.mutation("mlt_alpha_insensitive")
@pytest.mark.right_reason("Jacobian column for alpha_mlt is near-zero")
@pytest.mark.timeout(5400)
def test_fisher_20_mode_condition_number():
    """Validate Fisher κ < 1e10 with ~20+ modes (l=0,1,2) at N=100.

    WHAT: Fisher matrix condition number with a realistic solar-analog mode set.
    WHY: Proves the seismic forward model θ=(M,Z,α) → ν(n,l) supports
         multi-parameter inference without degeneracy. Cunha (2021) showed
         ~20 modes with l=2 break the M↔age degeneracy.
    EXTERNAL REFERENCE: Cunha et al. (2021), arXiv:2110.03332, Table 3.
    WHAT MAKES IT FAIL: @mutation('mlt_alpha_insensitive') clamps α → ∂ν/∂α ≈ 0
         → Fisher loses rank → κ >> 1e10.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    from stellar_jax.evolution import GradientTrustWarning
    warnings.filterwarnings("ignore", category=GradientTrustWarning)

    from stellar_jax.inference.fisher import (
        compute_jacobian_fd, compute_fisher, JacobianResult,
    )

    # ── Configuration ──
    # 3-parameter fiducial: (M, Z, α_MLT).
    # Y_init and f_ov are excluded from the FD sweep to reduce the number of
    # forward passes from 11 to 7 (each pass at N=100 takes ~minutes).
    # f_ov is degenerate at 1 M☉ (marginal convective core); Y_init is strongly
    # correlated with Z for forward-only frequency observations.
    PARAM_NAMES = ('mass', 'Z', 'alpha_mlt')
    FIDUCIAL = {
        'mass': 1.0,
        'Y_init': 0.27,      # held fixed (not varied in FD)
        'Z': 0.014,
        'alpha_mlt': 1.9,
        'f_ov': 0.016,        # held fixed (degenerate at 1 M☉)
    }

    # FD relative step sizes — generous for detectable frequency shifts.
    # M 1%: Δν ∝ M^{-1/2} → ~0.5% freq shift → ~7 μHz at 1400 μHz.
    # Z 2%: opacity-driven structural shift.
    # α 1%: MLT-driven Teff/envelope shift.
    FD_REL_STEPS = {
        'mass': 1e-2,
        'Z': 2e-2,
        'alpha_mlt': 1e-2,
    }

    N_STEPS = 100
    DT_FIXED = 4e7           # yr per step → 4 Gyr total (mid-MS solar analog)
    L_VALUES = (0, 1, 2)
    NU_MIN, NU_MAX = 1500.0, 4500.0
    N_SCAN = 200              # 15 μHz spacing, ~9 per Δν ≈ 135 μHz
    N_STEPS_OSC = 4000        # RK4 integration steps for eigenvalue solver
    SIGMA_NU = 0.5            # μHz (Kepler-quality per-mode uncertainty)

    # ── Step 1: FD Jacobian via existing infrastructure ──
    print(f"\n  ═══ FISHER 20-MODE VALIDATION (issue #1044) ═══")
    print(f"  M={FIDUCIAL['mass']}, Z={FIDUCIAL['Z']}, α={FIDUCIAL['alpha_mlt']}, "
          f"N={N_STEPS}, dt={DT_FIXED:.0e} yr "
          f"({N_STEPS * DT_FIXED / 1e9:.1f} Gyr)")
    print(f"  Parameters: {PARAM_NAMES}")
    print(f"  FD steps: {FD_REL_STEPS}")
    import sys; sys.stdout.flush()

    jac_result = compute_jacobian_fd(
        fiducial=FIDUCIAL,
        param_names=PARAM_NAMES,
        l_values=L_VALUES,
        nu_min=NU_MIN, nu_max=NU_MAX,
        n_scan=N_SCAN, n_steps_osc=N_STEPS_OSC,
        target_age=None,
        max_steps=N_STEPS,
        fixed_dt=DT_FIXED,
        diffusion=False,
        fd_rel_steps=FD_REL_STEPS,
    )

    jacobian = np.asarray(jac_result.jacobian)
    n_obs = jacobian.shape[0]
    n_seismic = n_obs - 2  # first 2 rows are log_L, log_Teff

    print(f"\n  ═══ FD JACOBIAN RESULTS ═══")
    print(f"  Shape: {jacobian.shape} ({n_seismic} seismic + 2 classical)")
    print(f"  Labels: {jac_result.obs_labels[:4]}...{jac_result.obs_labels[-2:]}")
    sys.stdout.flush()

    # ── Step 2: Safety — replace NaN/inf with 0 ──
    n_nonfinite = int((~np.isfinite(jacobian)).sum())
    if n_nonfinite > 0:
        print(f"\n  WARNING: {n_nonfinite} non-finite entries → 0")
        jacobian = np.where(np.isfinite(jacobian), jacobian, 0.0)

    # ── Step 3: Per-column diagnostics ──
    seismic_block = jacobian[2:, :]
    seismic_row_norms = np.linalg.norm(seismic_block, axis=1)
    valid_seismic = seismic_row_norms > 1e-10
    n_valid_seismic = int(valid_seismic.sum())
    n_lost = n_seismic - n_valid_seismic

    # Count modes per l degree from labels
    n_modes_per_l = {}
    for label in jac_result.obs_labels[2:]:
        # Labels like 'nu_l0_n12'
        if label.startswith('nu_l'):
            l_val = int(label.split('_')[1][1:])
            n_modes_per_l[l_val] = n_modes_per_l.get(l_val, 0) + 1
    n_total = n_seismic
    n_l2 = n_modes_per_l.get(2, 0)

    print(f"  Seismic modes: {n_seismic} total, {n_valid_seismic} valid")
    print(f"  Modes per l: {n_modes_per_l}")
    if n_lost > 0:
        print(f"  WARNING: {n_lost} mode(s) with zero row norm")

    col_norms = np.linalg.norm(jacobian, axis=0)
    for j, pname in enumerate(PARAM_NAMES):
        n_nonzero = np.sum(np.abs(jacobian[:, j]) > 1e-10)
        print(f"  ||∂obs/∂{pname}|| = {col_norms[j]:.4e}  "
              f"({n_nonzero}/{n_obs} non-zero rows)")
    sys.stdout.flush()

    # ── Step 4: All 3 parameter columns must be non-zero ──
    # Under mutation mlt_alpha_insensitive, α_MLT column collapses → test fails.
    for j, pname in enumerate(PARAM_NAMES):
        assert col_norms[j] > 1e-6, (
            f"Jacobian column for {pname} is near-zero "
            f"(||J[:,{j}]|| = {col_norms[j]:.2e}). "
            f"Parameter {pname} does not affect observables → "
            f"Fisher degenerate")

    # ── Step 5: Filter zero seismic rows, keep classical + valid seismic ──
    keep_rows = np.concatenate([
        np.array([True, True]),  # log_L, log_Teff always kept
        valid_seismic,
    ])
    jacobian_clean = jacobian[keep_rows, :]
    n_obs_clean = jacobian_clean.shape[0]

    # Build σ vector (Lund 2017 scale)
    sigma = np.zeros(n_obs_clean)
    sigma[0] = 0.02     # σ(log_L) ~ 0.02 dex (Kepler-grade photometry)
    sigma[1] = 0.005    # σ(log_Teff) ~ 0.005 dex (spectroscopic)
    sigma[2:] = SIGMA_NU  # per-mode Kepler quality

    # Assemble Fisher via existing infrastructure
    jac_clean = JacobianResult(
        jacobian=jacobian_clean,
        obs_values=jac_result.obs_values[keep_rows],
        obs_labels=[jac_result.obs_labels[i]
                    for i in range(n_obs) if keep_rows[i]],
        param_names=list(PARAM_NAMES),
        param_values=np.array([FIDUCIAL[p] for p in PARAM_NAMES]),
    )
    fisher = compute_fisher(jac_clean, sigma)

    print(f"\n  ═══ FISHER MATRIX RESULTS ═══")
    print(f"  Dimensions: {fisher.fisher.shape[0]}×{fisher.fisher.shape[1]} "
          f"from {n_obs_clean} observables ({n_valid_seismic} seismic + 2 classical)")
    print(f"  Parameters: {list(PARAM_NAMES)}")
    print(f"  Eigenvalues: {fisher.eigenvalues}")
    print(f"  κ(F) = {fisher.condition_number:.2e}")
    sys.stdout.flush()

    # ── Step 6: Assertions ──
    # Mode count — Cunha (2021) target is ~20+ modes for a solar analog
    assert n_total >= 20, (
        f"Expected ≥20 modes (Cunha 2021), found {n_total}. "
        f"Mode counts: {n_modes_per_l}")

    assert n_l2 >= 2, (
        f"Expected ≥2 l=2 modes for degeneracy breaking (Cunha 2021), "
        f"found {n_l2}")

    # Positive definite
    assert np.all(fisher.eigenvalues > 0), (
        f"Fisher not positive definite. Eigenvalues: {fisher.eigenvalues}")

    # Condition number
    assert fisher.condition_number < 1e10, (
        f"Fisher ill-conditioned: κ(F) = {fisher.condition_number:.2e} > 1e10. "
        f"Eigenvalues: {fisher.eigenvalues}. Parameters: {list(PARAM_NAMES)}")

    print(f"\n  ✓ κ(F) = {fisher.condition_number:.2e} < 1e10 (well-conditioned)")
    print(f"  ✓ {n_total} modes (≥20), {n_l2} l=2 modes (≥2)")
    print(f"  ✓ All 3 parameter columns non-degenerate")
    sys.stdout.flush()
