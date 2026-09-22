"""PLATO per-target precision forecast via the autodiff Fisher matrix.

Given a solar-analog target's expected PLATO mode set + noise, forecast
σ(M), σ(R), σ(age) via the FD Jacobian → Fisher → Cramér-Rao bound, and
compare against published hare-and-hounds / grid baselines.

Physics
-------
The Fisher matrix F = Jᵀ Σ⁻¹ J gives the expected information content of
the observable set {ν_{n,l}, log_L, log_Teff} about the stellar parameters
θ = {M, Y_init, Z, α_MLT, f_ov}. The Cramér-Rao bound σ(θ_j) = √[F⁻¹]_jj
is the minimum-variance achievable uncertainty.

For derived quantities (radius, age), we compute ∂g/∂θ via FD from
evolve_star and propagate: σ²(g) = (∇g)^T F⁻¹ (∇g). This is the standard
Fisher error propagation for functions of parameters (Tegmark, Taylor &
Heavens 1997, ApJ 480, 22, §II).

Derived quantities:
- R: from evolve_star's log_R output. ∂R/∂θ_j via central FD.
- age: the final model age (star_age) at the end of the evolution track.
  ∂age/∂θ_j via central FD.

Note: age is a peculiar derived quantity here. At fixed max_steps and
fixed_dt, the age is IDENTICAL for all parameter perturbations (it is
just max_steps × fixed_dt). To get meaningful ∂age/∂θ, one must evolve
to a fixed EVOLUTIONARY STATE (e.g. fixed Xc, or a target_age) and
measure how the age at that state changes with parameters. With
fixed_dt and max_steps, the model always reaches the same age, so
∂age/∂θ = 0 trivially.

For the PLATO demo at N=10 / fixed_dt = 2 Myr, we are at a near-ZAMS
state where the model age is literally max_steps × fixed_dt ≈ 20 Myr
for ALL parameter perturbations. In this regime, the Fisher-propagated
σ(age) is effectively zero because ∂age/∂θ = 0 (the age is an input,
not an output that depends on the stellar model). This means a pure
Fisher-propagated σ(age) is not meaningful without evolving to a
target evolutionary state.

To give a physically meaningful age uncertainty, we use a HYBRID approach:
1. σ(M) comes directly from the Fisher matrix (genuine).
2. σ(age)/age is estimated from σ(M)/M via the MS lifetime scaling,
   σ(age)/age ≈ f_age × σ(M)/M, where f_age encodes the M↔age
   degeneracy strength calibrated from the H&H exercises.
3. σ(R)/R is Fisher-propagated when ∂R/∂θ is available; otherwise
   from the mass-radius homology R ∝ M^0.8.

The f_age approach is physically motivated: for MS solar-type stars,
τ_MS ∝ M^{1-η} with η ≈ 4, so ∂(ln age)/∂(ln M) ≈ 1-η ≈ -3.
Combined with the M↔age degeneracy (l=2 modes break it via δν₀₂),
the empirical H&H factor is f_age ≈ 2.5 with l=2 (from Cunha 2021).

IMPORTANT: The f_age scaling is a LIMITATION, not a virtue. A genuinely
Fisher-derived σ(age) would require evolving to a target Xc state and
computing ∂age/∂θ at that state, which is infeasible at the CI test
scale (N=10, ZAMS). The scaling is grounded in published H&H numbers.

For σ(R), we attempt Fisher propagation first. At N=10 with fixed_dt,
∂R/∂θ is meaningful (the radius genuinely changes when M/Y/Z/α change),
so σ(R) from error propagation is a genuine Fisher result.

References
----------
Tegmark, Taylor & Heavens (1997), ApJ 480, 22 — Fisher error propagation.
Rauer et al. (2024), arXiv:2406.05447 — PLATO mission: 15%/2%/10% stellar.
Cunha et al. (2021), arXiv:2110.03332 — PLATO H&H: mass ~2%, age ~6%.
Bétrisey et al. (2023), arXiv:2306.04509 — mean density + ratios: 1.9%/0.7%/4.1%.
Lund et al. (2017), ApJ 835, 172 — LEGACY per-mode uncertainties.
Kippenhahn, Weigert & Weiss (2012), "Stellar Structure and Evolution", §22.1.
Demircan & Kahraman (1991), Ap&SS 181, 313 — empirical MS mass-radius.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from stellar_jax.inference.fisher import (
    compute_jacobian_fd,
    compute_fisher,
    forecast_sigma,
    JacobianResult,
    DEFAULT_FIDUCIAL,
    DEFAULT_PARAM_NAMES,
    DEFAULT_FD_REL_STEPS,
)
# Import the module for attribute access (mutation-compatible).
# The monkeypatch sets fisher_mod._safe_invert_fisher; a local
# `from ... import _safe_invert_fisher` binding would NOT be affected.
import stellar_jax.inference.fisher as _fisher_mod


# ═══════════════════════════════════════════════════════════════════════════════
# PLATO-specific constants
# ═══════════════════════════════════════════════════════════════════════════════

# PLATO noise floor for bright solar-like targets (V < 11).
# Lund et al. (2017, ApJ 835, 172) Table 1: median σ_ν ≈ 0.15–0.30 μHz
# for LEGACY targets with 4-year Kepler data. PLATO expects comparable
# per-mode precision for its P1 sample (Rauer 2024, §4.2).
PLATO_SIGMA_NU: float = 0.2  # μHz

# Classical constraint uncertainties (spectroscopic).
# Typical PLATO input catalog precision (Rauer 2024, §3.3):
#   - Teff: ~80 K → σ(log_Teff) ≈ 80/(Teff·ln10) ≈ 0.006 dex at 5777 K.
#   - L: Gaia parallax + bolometric correction → σ(log_L) ≈ 0.02 dex.
PLATO_SIGMA_LOG_L: float = 0.02    # dex
PLATO_SIGMA_LOG_TEFF: float = 0.006  # dex

# Mass-radius exponent for MS solar-type stars (FALLBACK for σ(R) when
# Fisher propagation is not available).
# Kippenhahn, Weigert & Weiss (2012), §22.1: R ∝ M^0.8 for 0.8–1.2 M☉.
# Demircan & Kahraman (1991): α_MR = 0.78 ± 0.03 from eclipsing binaries.
ALPHA_MR: float = 0.8

# MS lifetime scaling for age estimate (HYBRID approach).
# τ_MS ∝ M / L ∝ M^{1-η} where L ∝ M^η with η ≈ 4 (Kippenhahn §22.1).
# Age sensitivity: σ(age)/age ≈ f_age × σ(M)/M, where f_age encodes
# the M↔age degeneracy strength.
# From Cunha (2021) Table 2: best mass ~1.6–2.6%, best age ~5–6% → f_age ~ 2.5.
# From Bétrisey (2023) Table 3: mass 1.9%, age 4.1% → f_age ~ 2.2.
# We use f_age = 2.5 (conservative, Cunha mid-range).
# With l=2 modes: f_age ~ 2.5. Without l=2: f_age ≳ 5 (Cunha §7).
F_AGE_WITH_L2: float = 2.5
F_AGE_WITHOUT_L2: float = 6.0


# ═══════════════════════════════════════════════════════════════════════════════
# Derived-quantity gradient computation
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_derived_gradients_fd(
    fiducial: Dict[str, float],
    param_names: Sequence[str],
    max_steps: int,
    fixed_dt: Optional[float],
    diffusion: bool,
    fd_rel_steps: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute ∂R/∂θ and ∂age/∂θ via central FD from evolve_star.

    No oscillation solving needed — this is much cheaper than the seismic
    Jacobian (1 + 2×N_params = 11 evolve_star calls, no mode-finding).

    R is extracted from log_R at the final step. age from star_age[-1].

    Parameters
    ----------
    fiducial : dict — stellar parameters at the fiducial point.
    param_names : sequence of str — parameters to differentiate w.r.t.
    max_steps, fixed_dt, diffusion : evolve_star configuration.
    fd_rel_steps : dict or None — relative FD step per parameter.

    Returns
    -------
    grad_R : (N_params,) array — ∂R/∂θ_j in solar radii.
    grad_age : (N_params,) array — ∂age/∂θ_j in years.
    R_fid : float — fiducial radius (R_sun).
    age_fid : float — fiducial age (years).

    References
    ----------
    Tegmark, Taylor & Heavens (1997), ApJ 480, 22, §II.B — ∂g/∂θ for
    derived-quantity error propagation through the Fisher covariance.
    """
    from stellar_jax.stellar import evolve_star

    if fd_rel_steps is None:
        fd_rel_steps = dict(DEFAULT_FD_REL_STEPS)

    n_params = len(param_names)

    def _run(params):
        r = evolve_star(
            params['mass'], Z=params['Z'], Y_init=params['Y_init'],
            alpha_mlt=params['alpha_mlt'], f_ov=params['f_ov'],
            max_steps=max_steps, fixed_dt=fixed_dt, diffusion=diffusion,
        )
        log_R = float(r['log_R'][-1])
        age = float(r['star_age'][-1])
        # log_R is log10(R/Rsun), so R_solar = 10^log_R.
        R_solar = 10.0**log_R
        return R_solar, age

    # Fiducial
    R_fid, age_fid = _run(fiducial)

    # Central FD for each parameter
    grad_R = np.zeros(n_params)
    grad_age = np.zeros(n_params)

    for j, pname in enumerate(param_names):
        delta = abs(fiducial[pname]) * fd_rel_steps.get(pname, 1e-4)

        fid_plus = dict(fiducial)
        fid_plus[pname] = fiducial[pname] + delta

        fid_minus = dict(fiducial)
        fid_minus[pname] = fiducial[pname] - delta

        R_p, age_p = _run(fid_plus)
        R_m, age_m = _run(fid_minus)

        grad_R[j] = (R_p - R_m) / (2 * delta)
        grad_age[j] = (age_p - age_m) / (2 * delta)

    return grad_R, grad_age, R_fid, age_fid


def _propagate_derived_sigma(
    C_theta: np.ndarray,
    grad_g: np.ndarray,
) -> float:
    """Propagate σ(g) = sqrt(grad_g^T C_theta grad_g).

    Standard Fisher error propagation for a scalar function g(θ).
    σ²(g) = (∇g)^T F⁻¹ (∇g) where C_theta = F⁻¹.

    Parameters
    ----------
    C_theta : (N, N) array — parameter covariance matrix (F⁻¹).
    grad_g : (N,) array — gradient ∂g/∂θ.

    Returns
    -------
    sigma_g : float — 1-σ uncertainty of g.

    References
    ----------
    Tegmark, Taylor & Heavens (1997), ApJ 480, 22, Eq. 2 — the standard
    Fisher error propagation formula.
    """
    var_g = float(grad_g @ C_theta @ grad_g)
    return float(np.sqrt(max(var_g, 0.0)))


def _filter_jacobian_and_assemble_fisher(
        jac_result, jacobian, n_obs, sigma_log_L, sigma_log_Teff,
        sigma_nu, param_names, fiducial):
    """Build sigma_obs, filter degenerate rows/columns, assemble Fisher matrix.

    Returns (fisher_result, jac_final, all_sigmas, C_theta, obs_labels_clean,
             valid_seismic, sigma_obs_clean, param_names_final).
    """
    # Build observational uncertainty vector
    sigma_obs = np.zeros(n_obs)
    sigma_obs[0] = sigma_log_L
    sigma_obs[1] = sigma_log_Teff
    sigma_obs[2:] = sigma_nu

    # Filter degenerate rows/columns
    seismic_block = jacobian[2:, :]
    seismic_row_norms = np.linalg.norm(seismic_block, axis=1)
    valid_seismic = seismic_row_norms > 1e-10

    keep_rows = np.concatenate([np.array([True, True]), valid_seismic])
    jacobian_clean = jacobian[keep_rows, :]
    sigma_obs_clean = sigma_obs[keep_rows]
    obs_labels_clean = [jac_result.obs_labels[i]
                        for i in range(n_obs) if keep_rows[i]]

    col_norms = np.linalg.norm(jacobian_clean, axis=0)
    valid_cols = col_norms > 1e-10
    jacobian_final = jacobian_clean[:, valid_cols]
    param_names_final = tuple(p for p, v in zip(param_names, valid_cols) if v)
    param_values_final = np.array([fiducial[p] for p in param_names_final])

    # Assemble Fisher and forecast σ
    jac_final = JacobianResult(
        jacobian=jacobian_final,
        obs_values=jac_result.obs_values[keep_rows],
        obs_labels=obs_labels_clean,
        param_names=list(param_names_final),
        param_values=param_values_final,
    )
    fisher_result = compute_fisher(jac_final, sigma_obs_clean)
    all_sigmas = forecast_sigma(fisher_result)

    # Get the Fisher covariance C = F⁻¹
    F = np.asarray(fisher_result.fisher)
    # Use module-level attribute access so the mutation monkeypatch
    # on _fisher_mod._safe_invert_fisher takes effect.
    C_theta, inv_method = _fisher_mod._safe_invert_fisher(F)

    return (fisher_result, jac_final, all_sigmas, C_theta, obs_labels_clean,
            valid_seismic, sigma_obs_clean, param_names_final)


def _compute_sigma_M_R_age(
        fisher_result, jac_final, all_sigmas, C_theta, obs_labels_clean,
        valid_seismic, param_names_final, fiducial, max_steps, fixed_dt,
        diffusion, fd_rel_steps):
    """Compute σ(M), σ(R), σ(age) from Fisher covariance and build result dict."""
    # σ(M)/M
    sigma_M = all_sigmas.get('mass', np.nan)
    M_fid = fiducial.get('mass', 1.0)
    sigma_M_frac = sigma_M / M_fid

    # σ(R)/R via Fisher error propagation
    grad_R, grad_age, R_fid, age_fid = _compute_derived_gradients_fd(
        fiducial=fiducial,
        param_names=param_names_final,
        max_steps=max_steps,
        fixed_dt=fixed_dt,
        diffusion=diffusion,
        fd_rel_steps=fd_rel_steps,
    )

    # Fisher-propagated σ(R)
    sigma_R = _propagate_derived_sigma(C_theta, grad_R)
    sigma_R_frac = sigma_R / R_fid if R_fid > 0 else np.nan

    if np.linalg.norm(grad_R) < 1e-15 or not np.isfinite(sigma_R_frac):
        sigma_R_frac = ALPHA_MR * sigma_M_frac
        sigma_R_method = 'homology_fallback'
    else:
        sigma_R_method = 'fisher_propagation'

    # σ(age)/age — HYBRID: f_age × σ(M)/M
    has_l2 = any(lab.startswith('nu_l2') for lab in obs_labels_clean)
    f_age = F_AGE_WITH_L2 if has_l2 else F_AGE_WITHOUT_L2
    sigma_age_frac = f_age * sigma_M_frac

    # Mode count per angular degree
    n_modes_per_l: Dict[int, int] = {}
    for lab in obs_labels_clean:
        if lab.startswith('nu_l'):
            l_str = lab.split('_')[1]  # 'l0', 'l1', 'l2'
            l_val = int(l_str[1:])
            n_modes_per_l[l_val] = n_modes_per_l.get(l_val, 0) + 1

    return {
        'sigma_M_frac': float(sigma_M_frac),
        'sigma_R_frac': float(sigma_R_frac),
        'sigma_age_frac': float(sigma_age_frac),
        'sigma_M_pct': float(sigma_M_frac * 100),
        'sigma_R_pct': float(sigma_R_frac * 100),
        'sigma_age_pct': float(sigma_age_frac * 100),
        'n_modes': int(valid_seismic.sum()),
        'n_modes_per_l': n_modes_per_l,
        'has_l2': has_l2,
        'fisher_result': fisher_result,
        'jacobian_result': jac_final,
        'all_sigmas': all_sigmas,
        'condition_number': float(fisher_result.condition_number),
        'f_age': float(f_age),
        'sigma_R_method': sigma_R_method,
        'fisher_covariance': C_theta,
        'grad_R': grad_R,
        'R_fid': float(R_fid),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Core forecast function
# ═══════════════════════════════════════════════════════════════════════════════

def plato_per_target_forecast(
    fiducial: Optional[Dict[str, float]] = None,
    param_names: Sequence[str] = DEFAULT_PARAM_NAMES,
    l_values: Sequence[int] = (0, 1, 2),
    sigma_nu: float = PLATO_SIGMA_NU,
    sigma_log_L: float = PLATO_SIGMA_LOG_L,
    sigma_log_Teff: float = PLATO_SIGMA_LOG_TEFF,
    nu_min: float = 1500.0,
    nu_max: float = 4500.0,
    n_scan: int = 300,
    n_steps_osc: int = 8000,
    target_age: Optional[float] = None,
    max_steps: int = 100,
    fixed_dt: Optional[float] = None,
    diffusion: bool = False,
    fd_rel_steps: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Forecast PLATO per-target precision for a solar-like oscillator.

    Computes σ(M), σ(R), σ(age) using the Fisher information matrix
    assembled from the FD Jacobian of individual oscillation frequencies
    + classical observables (log_L, log_Teff).

    σ(M): directly from the Fisher inverse (Cramér-Rao bound).
    σ(R): Fisher error propagation via ∂R/∂θ from FD evolve_star calls.
    σ(age): HYBRID — f_age × σ(M)/M scaling from Cunha (2021) H&H.
        (See module docstring for why pure Fisher propagation of age is
        not meaningful at the CI test scale.)

    Parameters
    ----------
    fiducial : dict or None
        Stellar parameters for the target. Defaults to DEFAULT_FIDUCIAL
        (1 M☉ solar analog: M=1.0, Y=0.27, Z=0.014, α=1.9, f_ov=0.016).
    param_names : sequence of str
        Model parameters to include in the Fisher. Default: 5 physical params.
    l_values : sequence of int
        Angular degrees for the mode set. Default (0, 1, 2) — the full
        PLATO seismic set. Use (0, 1) to test the l=2 degradation.
    sigma_nu : float
        Per-mode frequency uncertainty (μHz). Default 0.2 μHz (PLATO P1).
    sigma_log_L, sigma_log_Teff : float
        Classical observable uncertainties (dex).
    nu_min, nu_max : float
        Frequency range for mode-finding (μHz).
    n_scan : int
        Scan points for Brent bracketing in mode-finder.
    n_steps_osc : int
        RK4 integration steps for the oscillation solver.
    target_age : float or None
        Interpolation target age (yr). None = use final step.
    max_steps : int
        Evolution steps for evolve_star.
    fixed_dt : float or None
        Fixed timestep (yr). None = adaptive.
    diffusion : bool
        Enable element diffusion.
    fd_rel_steps : dict or None
        Relative FD step per parameter.

    Returns
    -------
    dict with keys:
        'sigma_M_frac': σ(M)/M as fractional uncertainty (0–1 scale).
        'sigma_R_frac': σ(R)/R — Fisher-propagated.
        'sigma_age_frac': σ(age)/age — hybrid (f_age × σ(M)/M).
        'sigma_M_pct', 'sigma_R_pct', 'sigma_age_pct': same in percent.
        'n_modes': total number of modes found.
        'n_modes_per_l': dict l → count.
        'has_l2': bool, whether l=2 modes are present.
        'fisher_result': FisherResult from the assembly.
        'jacobian_result': JacobianResult from the FD computation.
        'all_sigmas': dict of all parameter σ values from forecast_sigma.
        'condition_number': Fisher matrix condition number.
        'sigma_R_method': 'fisher_propagation' or 'homology_fallback'.
        'fisher_covariance': (N_params, N_params) F⁻¹ matrix.
        'grad_R': (N_params,) ∂R/∂θ vector (when Fisher-propagated).
        'R_fid': fiducial radius in R_sun.

    References
    ----------
    See module docstring for full references.
    """
    if fiducial is None:
        fiducial = dict(DEFAULT_FIDUCIAL)

    # ── Step 1: Compute the FD Jacobian ──
    jac_result = compute_jacobian_fd(
        fiducial=fiducial,
        param_names=param_names,
        l_values=l_values,
        nu_min=nu_min,
        nu_max=nu_max,
        n_scan=n_scan,
        n_steps_osc=n_steps_osc,
        target_age=target_age,
        max_steps=max_steps,
        fixed_dt=fixed_dt,
        diffusion=diffusion,
        fd_rel_steps=fd_rel_steps,
    )

    jacobian = np.asarray(jac_result.jacobian)
    n_obs = jacobian.shape[0]

    # ── Steps 2-5: Build sigma, filter, assemble Fisher ──
    fisher_result, jac_final, all_sigmas, C_theta, obs_labels_clean, \
        valid_seismic, sigma_obs_clean, param_names_final = \
        _filter_jacobian_and_assemble_fisher(
            jac_result, jacobian, n_obs, sigma_log_L, sigma_log_Teff,
            sigma_nu, param_names, fiducial)

    # ── Steps 6-8: σ(M), σ(R), σ(age) + mode counts → result dict ──
    return _compute_sigma_M_R_age(
        fisher_result, jac_final, all_sigmas, C_theta, obs_labels_clean,
        valid_seismic, param_names_final, fiducial, max_steps, fixed_dt,
        diffusion, fd_rel_steps)
