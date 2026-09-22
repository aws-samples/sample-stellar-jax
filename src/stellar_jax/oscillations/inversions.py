"""SOLA structure inversion using AD kernels.

Implements the Subtractive Optimally Localized Averages (SOLA) method
(Pijpers & Thompson 1992, 1994) for recovering the relative sound-speed
perturbation δc²/c²(r) from observed frequency differences, using the
AD structure kernels from compute_structure_kernels.

The SOLA method finds inversion coefficients c_i such that the averaging
kernel K_avg(r; r₀) = Σ_i c_i · K_i(r) approximates a target localization
function T(r; r₀) (a Gaussian peaked at target radius r₀). The inverted
sound-speed difference at r₀ is then:

    <δc²/c²>(r₀) = Σ_i c_i · (δν_i / ν_i)

The coefficients are found by minimizing:

    χ² = ∫ [K_avg(r; r₀) - T(r; r₀)]² dr/R
         + μ · Σ_{ij} c_i · c_j · E_ij
         + β · [∫ C_avg(r; r₀)² dr/R]

where:
    - μ is the error-suppression (Tikhonov) parameter
    - E_ij = σ_i · σ_j · δ_ij (diagonal error covariance for uncorrelated errors)
    - β penalizes the cross-term kernel C_avg = Σ c_i · C_i(r) where C_i is the
      density kernel K_{ρ,c²} (minimizes contamination from density errors)

The linear system to solve is:

    (A + μ·E + β·B) · c = t

where:
    A_ij = ∫ K_i(r) · K_j(r) · dr/R            (kernel cross-products)
    B_ij = ∫ C_i(r) · C_j(r) · dr/R            (cross-term kernel products)
    t_i  = ∫ K_i(r) · T(r; r₀) · dr/R          (target overlap)
    E_ij = σ_i² · δ_ij                          (error covariance, diagonal)

References:
    Pijpers & Thompson (1992), A&A 262, L33
    Pijpers & Thompson (1994), A&A 281, 231
    Basu & Christensen-Dalsgaard (1997), astro-ph/9702162
    Rabello-Soares, Basu & CD (1999), MNRAS 309, 35
    Buldgen, Reese & Dupret (2016), A&A 583, A62 (arXiv:1610.01460)
"""
import numpy as np


def sola_inversion(kernel_results, dnu_over_nu, sigma, target_x,
                   target_width=0.04, mu=1e-4, beta=0.0):
    """Perform a SOLA inversion at a set of target radial points.

    Given a set of mode kernels, observed relative frequency differences,
    and their uncertainties, recovers the sound-speed perturbation profile
    δc²/c²(r) at specified target radii using the SOLA method.

    Args:
        kernel_results: list of dicts from compute_structure_kernels, one per mode.
            Each must contain 'K_gamma1_rho', 'K_c2_rho', 'x', 'dr_over_R', 'nu'.
            All kernels must be on the SAME radial grid (same FGONG input).
        dnu_over_nu: (N_modes,) array — observed δν/ν for each mode.
        sigma: (N_modes,) array — 1σ uncertainty on each δν/ν.
        target_x: (N_targets,) array — target fractional radii r₀/R for inversion.
        target_width: float — Gaussian target width Δ in fractional radius
            (default 0.04, suitable for ~20 modes spanning l=0,1,2).
        mu: float — Tikhonov regularization parameter for error suppression
            (default 1e-4; larger = smoother but more biased).
        beta: float — cross-term (density contamination) suppression weight
            (default 0.0 = no cross-term suppression; the (c²,ρ) and (Γ₁,ρ)
            kernels are identical in our formulation, so cross-term is moot
            unless a separate density kernel is available).

    Returns:
        dict with:
            'x_target': (N_targets,) target radii
            'dc2_over_c2': (N_targets,) recovered δc²/c² at each target
            'error': (N_targets,) propagated 1σ error on δc²/c² at each target
            'averaging_kernels': (N_targets, N_grid) averaging kernels
            'target_functions': (N_targets, N_grid) target Gaussians
            'coefficients': (N_targets, N_modes) inversion coefficients
            'x_grid': (N_grid,) the common radial grid

    Raises:
        ValueError: if kernels are on different grids or inputs are inconsistent.

    References:
        Pijpers & Thompson (1994), A&A 281, 231, §2
        Basu & Christensen-Dalsgaard (1997), astro-ph/9702162
    """
    N_modes = len(kernel_results)
    if N_modes == 0:
        raise ValueError("Need at least one kernel for inversion")

    dnu_over_nu = np.asarray(dnu_over_nu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    target_x = np.asarray(target_x, dtype=np.float64)

    if dnu_over_nu.shape != (N_modes,):
        raise ValueError(f"dnu_over_nu shape {dnu_over_nu.shape} != ({N_modes},)")
    if sigma.shape != (N_modes,):
        raise ValueError(f"sigma shape {sigma.shape} != ({N_modes},)")

    # Extract the common grid from the first kernel
    x_grid = kernel_results[0]['x']
    dr_over_R = kernel_results[0]['dr_over_R']
    N_grid = len(x_grid)

    # Stack all kernels: K[i, j] = K_i(r_j)
    K = np.zeros((N_modes, N_grid))
    for i, kr in enumerate(kernel_results):
        if len(kr['x']) != N_grid:
            raise ValueError(
                f"Kernel {i} has {len(kr['x'])} grid points, "
                f"expected {N_grid} (all must use the same FGONG)")
        K[i, :] = kr['K_gamma1_rho']

    # ─── Build the SOLA matrices ──────────────────────────────────────────
    # A_ij = ∫ K_i(r) · K_j(r) · dr/R = Σ_k K_i(r_k) · K_j(r_k) · Δr_k/R
    # This is K · diag(dr/R) · K^T
    W = np.diag(dr_over_R)  # (N_grid, N_grid) weight matrix
    A = K @ W @ K.T  # (N_modes, N_modes)

    # Error covariance: E_ij = σ_i² δ_ij
    E = np.diag(sigma**2)

    # For each target point, solve the linear system
    N_targets = len(target_x)
    dc2_result = np.zeros(N_targets)
    error_result = np.zeros(N_targets)
    avg_kernels = np.zeros((N_targets, N_grid))
    target_fns = np.zeros((N_targets, N_grid))
    coefficients = np.zeros((N_targets, N_modes))

    for t_idx in range(N_targets):
        x0 = target_x[t_idx]

        # Target function: Gaussian centered at x0, width target_width
        T = np.exp(-0.5 * ((x_grid - x0) / target_width)**2)
        # Normalize so ∫ T · dr/R = 1
        T_integral = float(np.sum(T * dr_over_R))
        if T_integral > 1e-30:
            T = T / T_integral
        target_fns[t_idx, :] = T

        # Target overlap: t_i = ∫ K_i(r) · T(r) · dr/R
        t_vec = K @ (T * dr_over_R)  # (N_modes,)

        # Solve: (A + μ·E) · c = t
        # (cross-term B omitted when beta=0)
        M = A + mu * E
        if beta > 0:
            # Would need separate density kernels; for (c²,ρ)=(Γ₁,ρ), skip
            pass

        # Solve the linear system with regularization
        try:
            c = np.linalg.solve(M, t_vec)
        except np.linalg.LinAlgError:
            # Fallback: pseudo-inverse
            c = np.linalg.lstsq(M, t_vec, rcond=None)[0]

        coefficients[t_idx, :] = c

        # Inverted value: <δc²/c²>(r₀) = Σ c_i · (δν_i/ν_i)
        dc2_result[t_idx] = float(c @ dnu_over_nu)

        # Propagated error: σ² = Σ c_i² σ_i²
        error_result[t_idx] = float(np.sqrt(np.sum(c**2 * sigma**2)))

        # Averaging kernel: K_avg(r) = Σ c_i · K_i(r)
        avg_kernels[t_idx, :] = c @ K

    return {
        'x_target': target_x,
        'dc2_over_c2': dc2_result,
        'error': error_result,
        'averaging_kernels': avg_kernels,
        'target_functions': target_fns,
        'coefficients': coefficients,
        'x_grid': x_grid,
    }
