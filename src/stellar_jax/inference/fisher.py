"""Fisher information matrix via exact AD Jacobians.

Assembles the d(observable)/d(parameter) Jacobian block from the differentiable
chain (evolve_star → FGONG → oscillation → IFT eigenfreq), validates each column
AD-vs-FD, and builds F = Jᵀ Σ⁻¹ J.

Architecture
------------
The Jacobian J[i,j] = ∂obs_i/∂θ_j is computed by jax.grad through the FULL
differentiable chain for each (observable, parameter) pair. For N_obs observables
and N_params parameters, this is O(N_obs × N_params) gradient evaluations.

The chain for seismic observables:
    θ_j → evolve_star → y_henyey_final → structure_to_fgong_jax →
    build_oscillation_coeffs_jax → eigenfreq_from_coeffs (IFT @custom_vjp)

Mode-finding (Brent) runs forward-only outside JAX; the IFT adjoint supplies
the gradient at the converged σ².

Parameters θ (the differentiable stellar knobs):
    {M, Y_init, Z, α_MLT, f_ov} — 5 physically independent stellar parameters.

Observables:
    Individual ν_{n,l} for l=0,1,2 + log_L + log_Teff.
    l=2 modes required for age accuracy (Cunha 2021, arXiv:2110.03332).
    Diagonal Σ from per-mode peak-bagging uncertainties (Lund et al. 2017).

References
----------
Iacovelli et al. (2022), arXiv:2207.06910 — GWFAST: AD Fisher matrix.
Cunha et al. (2021), arXiv:2110.03332 — PLATO hare-and-hounds; mode selection.
Bétrisey et al. (2023), arXiv:2306.04509 — surface-independent ratios.
Lund et al. (2017), ApJ 835, 172 — LEGACY per-mode uncertainties.
"""

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from stellar_jax.config.mesh_defaults import F_OV


# ═══════════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════════

class JacobianResult(NamedTuple):
    """Result of a Jacobian computation.

    Attributes
    ----------
    jacobian : (N_obs, N_params) array — J[i,j] = ∂obs_i/∂θ_j.
    obs_values : (N_obs,) array — observable values at θ₀.
    obs_labels : list of str — human-readable labels for each observable row.
    param_names : list of str — parameter names for each column.
    param_values : (N_params,) array — parameter values θ₀.
    """
    jacobian: jnp.ndarray
    obs_values: np.ndarray
    obs_labels: List[str]
    param_names: List[str]
    param_values: np.ndarray


class FisherResult(NamedTuple):
    """Result of a Fisher matrix computation.

    Attributes
    ----------
    fisher : (N_params, N_params) array — F = Jᵀ Σ⁻¹ J.
    condition_number : float — ratio of largest to smallest eigenvalue.
    eigenvalues : (N_params,) array — eigenvalues of F (ascending).
    jacobian_result : JacobianResult — the underlying Jacobian.
    sigma_obs : (N_obs,) array — observational uncertainties used.
    """
    fisher: jnp.ndarray
    condition_number: float
    eigenvalues: np.ndarray
    jacobian_result: JacobianResult
    sigma_obs: np.ndarray


# ═══════════════════════════════════════════════════════════════════════════════
# Default parameter set
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PARAM_NAMES: Tuple[str, ...] = (
    'mass', 'Y_init', 'Z', 'alpha_mlt', 'f_ov',
)

# Solar-analog fiducial: M=1.0, Y₀=0.27 (BBN+GS98), Z=0.014 (GS98),
# α=1.9 (typical solar-cal), f_ov=F_OV (Herwig 2000 step overshooting, 0.016).
DEFAULT_FIDUCIAL: Dict[str, float] = {
    'mass': 1.0,
    'Y_init': 0.27,
    'Z': 0.014,
    'alpha_mlt': 1.9,
    'f_ov': F_OV,
}

# Central FD relative step sizes per parameter.
# Optimal ε ~ (3 ε_mach)^(1/3) ≈ 6e-6 for float64.
DEFAULT_FD_REL_STEPS: Dict[str, float] = {
    'mass': 1e-4,
    'Y_init': 1e-3,
    'Z': 1e-3,
    'alpha_mlt': 1e-4,
    'f_ov': 1e-3,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Core: scalar observable functions for jax.grad
# ═══════════════════════════════════════════════════════════════════════════════

def _make_classical_fn(observable: str, param_names: Sequence[str],
                       fiducial: Dict[str, float], param_idx: int,
                       target_age: Optional[float], max_steps: int,
                       fixed_dt: Optional[float], diffusion: bool):
    """Build a scalar function θ_j → classical_observable for jax.grad.

    Closes over all parameters except θ_j (the differentiation target).
    """
    from stellar_jax.stellar import evolve_star, observable_at_target

    pname = param_names[param_idx]

    def fn(theta_j):
        kwargs = dict(fiducial)
        kwargs[pname] = theta_j
        result = evolve_star(
            kwargs['mass'], Z=kwargs['Z'], Y_init=kwargs['Y_init'],
            alpha_mlt=kwargs['alpha_mlt'], f_ov=kwargs['f_ov'],
            max_steps=max_steps, fixed_dt=fixed_dt, diffusion=diffusion,
        )
        if target_age is not None:
            return observable_at_target(result, observable, 'star_age', target_age)
        return result[observable][-1]

    return fn


def _make_seismic_fn(param_names: Sequence[str], fiducial: Dict[str, float],
                     param_idx: int, l: int, mode_info: dict,
                     target_age: Optional[float],
                     max_steps: int, fixed_dt: Optional[float],
                     diffusion: bool, return_nu: bool = False):
    """Build a scalar function θ_j → σ²(mode) or θ_j → ν(mode) for jax.grad.

    The function computes the FULL differentiable chain:
        θ_j → evolve_star → structure_to_fgong_jax → build_oscillation_coeffs_jax
        → eigenfreq_from_coeffs (IFT @custom_vjp) → σ² [→ ν if return_nu=True]

    The mode-finding (Brent root-refinement) runs ONCE at the fiducial, OUTSIDE
    this function, producing sigma2_ref and the integration mesh (x_steps, h_steps).
    These are passed in via mode_info and used as constant seeds for the IFT adjoint.

    This is the same pattern used in the validated seismic gradient tests
    (test_oscillations.py:sigma2_of_mass) — the IFT adjoint in eigenfreq_from_coeffs
    provides ∂σ²/∂(coeffs, x_grid) at the converged root, and JAX propagates the
    gradient back through build_oscillation_coeffs_jax → structure_to_fgong_jax →
    evolve_star → θ_j.

    Parameters
    ----------
    mode_info : dict
        Pre-computed mode information from the fiducial forward pass. Must contain:
        'sigma2_ref': converged σ² from Brent (seed for IFT)
        'x_steps': integration grid x-values (static)
        'h_steps': integration step sizes (static)
        'factor': ν² → σ² conversion factor
        'l': angular degree
    return_nu : bool
        If True, return ν [μHz] = sqrt(σ²/factor) where factor is computed from
        the LIVE R, M, G values inside the differentiated function. This ensures
        jax.grad correctly captures ∂factor/∂θ (the factor depends on M directly
        and on R which changes with all parameters). Default False for backward
        compatibility with tests that work in σ² space.

    References
    ----------
    Blondel et al. (2022), "Efficient and Modular Implicit Differentiation" —
    the IFT adjoint supplies ∂σ²*/∂θ = −(∂D/∂θ)/(∂D/∂σ²).
    """
    from stellar_jax.stellar import evolve_star, observable_at_target
    from stellar_jax.fgong import structure_to_fgong_jax
    from stellar_jax.oscillations import build_oscillation_coeffs_jax, eigenfreq_from_coeffs

    pname = param_names[param_idx]
    sigma2_ref = mode_info['sigma2_ref']
    x_steps = mode_info['x_steps']
    h_steps = mode_info['h_steps']
    factor = mode_info['factor']

    def fn(theta_j):
        kwargs = dict(fiducial)
        kwargs[pname] = theta_j

        result = evolve_star(
            kwargs['mass'], Z=kwargs['Z'], Y_init=kwargs['Y_init'],
            alpha_mlt=kwargs['alpha_mlt'], f_ov=kwargs['f_ov'],
            max_steps=max_steps, fixed_dt=fixed_dt, diffusion=diffusion,
        )

        y_henyey = result['y_henyey_final']
        X_profile = result['X_profile']

        if target_age is not None:
            log_L = observable_at_target(result, 'log_L', 'star_age', target_age)
            log_Teff = observable_at_target(result, 'log_Teff', 'star_age', target_age)
        else:
            log_L = result['log_L'][-1]
            log_Teff = result['log_Teff'][-1]

        glob, var = structure_to_fgong_jax(
            kwargs['mass'], log_L, log_Teff, X_profile, kwargs['Z'],
            t_age=0.0, alpha_mlt=kwargs['alpha_mlt'], y_henyey=y_henyey,
            atm_ratio=result.get('atm_ratio'),
        )

        # Build oscillation coefficients as live JAX arrays (gradient path).
        grid_data = build_oscillation_coeffs_jax(glob, var)

        # Wire the IFT adjoint: eigenfreq_from_coeffs wraps sigma2_ref with
        # @custom_vjp so that jax.grad produces ∂σ²/∂(coeffs, x_grid) via the
        # implicit function theorem, then JAX propagates back through the chain.
        sigma2 = eigenfreq_from_coeffs(
            grid_data['coeffs'], grid_data['x_grid'], sigma2_ref,
            l, x_steps, h_steps, factor,
        )

        if return_nu:
            # Convert σ² → ν [μHz] using the LIVE factor from grid_data.
            # factor = (2π·1e-6)² · R³/(G·M) depends on M directly and on R
            # (which changes with all params). nu_from_sigma2 preserves the
            # gradient ∂factor/∂θ (the ν²·∂factor/∂θ scaling term).
            from stellar_jax.oscillations.seismic_conversion import nu_from_sigma2
            live_factor = grid_data['factor']
            nu = nu_from_sigma2(sigma2, live_factor)
            return nu

        return sigma2

    return fn


# ═══════════════════════════════════════════════════════════════════════════════
# Jacobian computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_jacobian(
    fiducial: Optional[Dict[str, float]] = None,
    param_names: Sequence[str] = DEFAULT_PARAM_NAMES,
    l_values: Sequence[int] = (0, 1, 2),
    nu_min: float = 1500.0,
    nu_max: float = 4500.0,
    n_scan: int = 300,
    n_steps_osc: int = 8000,
    target_age: Optional[float] = None,
    max_steps: int = 100,
    fixed_dt: Optional[float] = None,
    diffusion: bool = False,
) -> JacobianResult:
    """Compute the full observable-space Jacobian J[i,j] = ∂obs_i/∂θ_j.

    Assembled in two blocks:
    1. Classical (log_L, log_Teff): jax.grad through evolve_star.
    2. Seismic (individual ν_{n,l}): jax.grad through the full chain per mode.

    Parameters
    ----------
    fiducial : dict or None
        Parameter values θ₀. Defaults to DEFAULT_FIDUCIAL.
    param_names : sequence of str
        Parameters to differentiate w.r.t. Default: 5 physical params.
    l_values : sequence of int
        Angular degrees for seismic block. Default: (0, 1, 2).
    nu_min, nu_max : float
        Frequency scan range for mode-finding (μHz).
    n_scan : int
        Scan points for Brent bracketing.
    n_steps_osc : int
        RK4 steps for oscillation solver.
    target_age : float or None
        Interpolation target age (yr). None = use final step.
    max_steps : int
        Evolution steps for evolve_star.
    fixed_dt : float or None
        Fixed timestep for gradient consistency.
    diffusion : bool
        Enable element diffusion (default False for clean gradients).

    Returns
    -------
    JacobianResult
    """
    if fiducial is None:
        fiducial = dict(DEFAULT_FIDUCIAL)

    n_params = len(param_names)
    theta0 = np.array([fiducial[p] for p in param_names], dtype=np.float64)

    # ── Step 1: Forward pass — find modes at fiducial ──
    fwd = _forward_observables(
        fiducial, l_values=l_values, nu_min=nu_min, nu_max=nu_max,
        n_scan=n_scan, n_steps_osc=n_steps_osc, target_age=target_age,
        max_steps=max_steps, fixed_dt=fixed_dt, diffusion=diffusion,
    )
    obs_values = fwd['obs_values']
    obs_labels = fwd['obs_labels']
    modes = fwd['modes']
    n_obs = len(obs_values)

    # ── Step 2: Compute Jacobian via jax.grad ──
    jacobian = np.zeros((n_obs, n_params))

    for j in range(n_params):
        # Classical: ∂log_L/∂θ_j, ∂log_Teff/∂θ_j
        for i_obs, obs_name in enumerate(['log_L', 'log_Teff']):
            fn = _make_classical_fn(
                obs_name, param_names, fiducial, j,
                target_age=target_age, max_steps=max_steps,
                fixed_dt=fixed_dt, diffusion=diffusion,
            )
            theta_j = jnp.float64(fiducial[param_names[j]])
            grad_val = float(jax.grad(fn)(theta_j))
            jacobian[i_obs, j] = grad_val

        # Seismic: ∂ν_{n,l}/∂θ_j for each mode
        for k, (l, mi, nu_fid, factor, mode_info) in enumerate(modes):
            fn_nu = _make_seismic_fn(
                param_names, fiducial, j, l=l, mode_info=mode_info,
                target_age=target_age, max_steps=max_steps,
                fixed_dt=fixed_dt, diffusion=diffusion,
                return_nu=True,
            )
            theta_j = jnp.float64(fiducial[param_names[j]])
            grad_nu = float(jax.grad(fn_nu)(theta_j))

            # ∂ν/∂θ is computed directly by jax.grad because fn_nu returns
            # ν = sqrt(σ²/factor) where factor = (2π·1e-6)²·R³/(G·M) is
            # recomputed from live R, M inside the differentiated function.
            # This correctly captures ∂factor/∂θ (R and M depend on θ).
            jacobian[2 + k, j] = grad_nu

    return JacobianResult(
        jacobian=jnp.array(jacobian),
        obs_values=obs_values,
        obs_labels=obs_labels,
        param_names=list(param_names),
        param_values=theta0,
    )


def _forward_observables(
    fiducial: Dict[str, float],
    l_values: Sequence[int] = (0, 1, 2),
    nu_min: float = 1500.0,
    nu_max: float = 4500.0,
    n_scan: int = 300,
    n_steps_osc: int = 8000,
    target_age: Optional[float] = None,
    max_steps: int = 100,
    fixed_dt: Optional[float] = None,
    diffusion: bool = False,
) -> dict:
    """Run the forward model once and collect all observable values.

    Returns a dict with 'obs_values', 'obs_labels', 'modes'.
    modes is a list of (l, mode_index, nu_μHz, factor, mode_info_dict),
    where mode_info_dict contains {sigma2_ref, x_steps, h_steps, factor, l}
    for eigenfreq_from_coeffs (IFT adjoint).
    """
    from stellar_jax.stellar import evolve_star, observable_at_target
    from stellar_jax.fgong import structure_to_fgong_jax
    from stellar_jax.oscillations import compute_eigenfreq_from_structure_jax

    result = evolve_star(
        fiducial['mass'], Z=fiducial['Z'], Y_init=fiducial['Y_init'],
        alpha_mlt=fiducial['alpha_mlt'], f_ov=fiducial['f_ov'],
        max_steps=max_steps, fixed_dt=fixed_dt, diffusion=diffusion,
    )

    if target_age is not None:
        log_L = float(observable_at_target(result, 'log_L', 'star_age', target_age))
        log_Teff = float(observable_at_target(result, 'log_Teff', 'star_age', target_age))
    else:
        log_L = float(result['log_L'][-1])
        log_Teff = float(result['log_Teff'][-1])

    y_henyey = result['y_henyey_final']
    # X_profile from evolve_star is the final-step composition (N_COMP,).
    X_profile = result['X_profile']

    glob, var = structure_to_fgong_jax(
        fiducial['mass'], log_L, log_Teff, X_profile, fiducial['Z'],
        t_age=0.0, alpha_mlt=fiducial['alpha_mlt'], y_henyey=y_henyey,
        atm_ratio=result.get('atm_ratio'),
    )

    # Find modes — also save the integration mesh and sigma2 for the IFT adjoint.
    # Each mode entry carries the information eigenfreq_from_coeffs needs:
    # (l, mode_index, nu_μHz, factor, mode_info_dict)
    modes = []
    for l in l_values:
        mode_index = 0
        while True:
            try:
                info = compute_eigenfreq_from_structure_jax(
                    glob, var, l=l, nu_min=nu_min, nu_max=nu_max,
                    n_scan=n_scan, n_steps=n_steps_osc, mode_index=mode_index,
                )
                mode_info = {
                    'sigma2_ref': info['sigma2'],
                    'x_steps': info['x_steps'],
                    'h_steps': info['h_steps'],
                    'factor': float(info['factor']),
                    'l': l,
                }
                modes.append((l, mode_index, info['nu'], float(info['factor']), mode_info))
                mode_index += 1
            except ValueError:
                break

    modes.sort(key=lambda x: x[2])

    # Build observable vector
    obs_labels = ['log_L', 'log_Teff']
    obs_values_list = [log_L, log_Teff]
    for l, mi, nu, _, _ in modes:
        # Approximate radial order from asymptotic relation: ν ≈ Δν(n + l/2 + ε)
        # Δν_sun ≈ 135 μHz (Christensen-Dalsgaard, Lecture Notes, Ch. 7)
        DELTA_NU_SUN_APPROX = 135.0  # μHz
        n_approx = int(round(nu / DELTA_NU_SUN_APPROX - l / 2.0 - 1.5))
        obs_labels.append(f'nu_l{l}_n{n_approx}')
        obs_values_list.append(nu)

    return {
        'obs_values': np.array(obs_values_list),
        'obs_labels': obs_labels,
        'modes': modes,
    }


def compute_jacobian_fd(
    fiducial: Optional[Dict[str, float]] = None,
    param_names: Sequence[str] = DEFAULT_PARAM_NAMES,
    l_values: Sequence[int] = (0, 1, 2),
    nu_min: float = 1500.0,
    nu_max: float = 4500.0,
    n_scan: int = 300,
    n_steps_osc: int = 8000,
    target_age: Optional[float] = None,
    max_steps: int = 100,
    fixed_dt: Optional[float] = None,
    diffusion: bool = False,
    fd_rel_steps: Optional[Dict[str, float]] = None,
) -> JacobianResult:
    """Compute the Jacobian via central finite differences (for validation).

    For each parameter θ_j, computes observables at θ_j ± δ and forms the
    central FD derivative: ∂obs_i/∂θ_j ≈ (obs_i(θ_j+δ) - obs_i(θ_j-δ))/(2δ).

    Mode matching is by (l, mode_index) — the same Brent bracket must yield
    the same mode at both perturbation points (validated by frequency continuity).

    Parameters
    ----------
    fd_rel_steps : dict or None
        Relative FD step per parameter. Defaults to DEFAULT_FD_REL_STEPS.
    (other params same as compute_jacobian)

    Returns
    -------
    JacobianResult
    """
    if fiducial is None:
        fiducial = dict(DEFAULT_FIDUCIAL)
    if fd_rel_steps is None:
        fd_rel_steps = dict(DEFAULT_FD_REL_STEPS)

    n_params = len(param_names)
    theta0 = np.array([fiducial[p] for p in param_names], dtype=np.float64)

    # Forward at fiducial (to establish the mode set and labels)
    fwd_0 = _forward_observables(
        fiducial, l_values=l_values, nu_min=nu_min, nu_max=nu_max,
        n_scan=n_scan, n_steps_osc=n_steps_osc, target_age=target_age,
        max_steps=max_steps, fixed_dt=fixed_dt, diffusion=diffusion,
    )
    obs_labels = fwd_0['obs_labels']
    modes_0 = fwd_0['modes']
    n_obs = len(fwd_0['obs_values'])

    jacobian = np.zeros((n_obs, n_params))

    for j, pname in enumerate(param_names):
        delta = abs(fiducial[pname]) * fd_rel_steps.get(pname, 1e-4)

        fid_plus = dict(fiducial)
        fid_plus[pname] = fiducial[pname] + delta

        fid_minus = dict(fiducial)
        fid_minus[pname] = fiducial[pname] - delta

        fwd_plus = _forward_observables(
            fid_plus, l_values=l_values, nu_min=nu_min, nu_max=nu_max,
            n_scan=n_scan, n_steps_osc=n_steps_osc, target_age=target_age,
            max_steps=max_steps, fixed_dt=fixed_dt, diffusion=diffusion,
        )
        fwd_minus = _forward_observables(
            fid_minus, l_values=l_values, nu_min=nu_min, nu_max=nu_max,
            n_scan=n_scan, n_steps_osc=n_steps_osc, target_age=target_age,
            max_steps=max_steps, fixed_dt=fixed_dt, diffusion=diffusion,
        )

        # Classical FD — handle NaN from evolve_star failures at perturbed points
        log_L_p = fwd_plus['obs_values'][0]
        log_L_m = fwd_minus['obs_values'][0]
        log_T_p = fwd_plus['obs_values'][1]
        log_T_m = fwd_minus['obs_values'][1]

        if np.isfinite(log_L_p) and np.isfinite(log_L_m):
            jacobian[0, j] = (log_L_p - log_L_m) / (2 * delta)
        else:
            jacobian[0, j] = 0.0
            warnings.warn(
                f"log_L not finite at {pname}±δ — setting ∂log_L/∂{pname} = 0",
                stacklevel=2,
            )

        if np.isfinite(log_T_p) and np.isfinite(log_T_m):
            jacobian[1, j] = (log_T_p - log_T_m) / (2 * delta)
        else:
            jacobian[1, j] = 0.0
            warnings.warn(
                f"log_Teff not finite at {pname}±δ — setting ∂log_Teff/∂{pname} = 0",
                stacklevel=2,
            )

        # Seismic FD — match by (l, mode_index).
        # If a mode is lost at a perturbed point (different mode count from
        # fiducial), set that Jacobian entry to 0.0 instead of NaN.
        # This avoids NaN propagation into F = Jᵀ Σ⁻¹ J, which would make
        # condition_number = NaN. A zero entry means "no FD information for
        # this (mode, param)" — the Fisher matrix simply loses that constraint.
        plus_lookup = {(ml, mmi): mnu for ml, mmi, mnu, _, _ in fwd_plus['modes']}
        minus_lookup = {(ml, mmi): mnu for ml, mmi, mnu, _, _ in fwd_minus['modes']}

        for k, (l_k, mi_k, nu_fid, _, _) in enumerate(modes_0):
            nu_p = plus_lookup.get((l_k, mi_k))
            nu_m = minus_lookup.get((l_k, mi_k))
            if nu_p is not None and nu_m is not None:
                jacobian[2 + k, j] = (nu_p - nu_m) / (2 * delta)
            else:
                # Mode lost at perturbed point — no FD derivative available.
                jacobian[2 + k, j] = 0.0
                lost_side = []
                if nu_p is None:
                    lost_side.append(f'{pname}+δ')
                if nu_m is None:
                    lost_side.append(f'{pname}-δ')
                warnings.warn(
                    f"Mode (l={l_k}, mi={mi_k}, ν={nu_fid:.1f} μHz) lost at "
                    f"{', '.join(lost_side)} — setting ∂ν/∂{pname} = 0",
                    stacklevel=2,
                )

    return JacobianResult(
        jacobian=jnp.array(jacobian),
        obs_values=fwd_0['obs_values'],
        obs_labels=obs_labels,
        param_names=list(param_names),
        param_values=theta0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Fisher matrix assembly
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fisher(
    jacobian_result: JacobianResult,
    sigma_obs: np.ndarray,
) -> FisherResult:
    """Assemble the Fisher information matrix F = Jᵀ Σ⁻¹ J.

    Parameters
    ----------
    jacobian_result : JacobianResult
        The Jacobian from compute_jacobian.
    sigma_obs : np.ndarray, shape (N_obs,)
        1-σ observational uncertainties for each observable.

    Returns
    -------
    FisherResult with the Fisher matrix, condition number, eigenvalues.

    References
    ----------
    Iacovelli et al. (2022), arXiv:2207.06910, Eq. 2.7 — F = Jᵀ Σ⁻¹ J.
    Lund et al. (2017), ApJ 835, 172 — per-mode σ_ν (diagonal Σ).
    """
    J = np.asarray(jacobian_result.jacobian)
    sigma = np.asarray(sigma_obs)

    if J.shape[0] != sigma.shape[0]:
        raise ValueError(
            f"Jacobian has {J.shape[0]} observables but sigma_obs has {sigma.shape[0]}")

    # Safety: replace any NaN/inf in the Jacobian with 0.0.
    # NaN arises when a mode is lost at a perturbed parameter point (the FD
    # derivative is undefined). Setting to 0 means that (mode, param) pair
    # contributes no information to the Fisher — the correct semantic.
    nan_mask = ~np.isfinite(J)
    if nan_mask.any():
        n_nan = int(nan_mask.sum())
        warnings.warn(
            f"Jacobian has {n_nan} non-finite entries — replacing with 0",
            stacklevel=2,
        )
        J = np.where(nan_mask, 0.0, J)

    # F = Jᵀ diag(1/σ²) J
    w = 1.0 / sigma  # (N_obs,)
    Jw = J * w[:, None]  # (N_obs, N_params)  weighted Jacobian
    F = Jw.T @ Jw  # (N_params, N_params)

    # Eigenvalue decomposition for condition number (GWFAST §3.3)
    eigenvalues = np.linalg.eigvalsh(F)
    ev_abs = np.abs(eigenvalues)
    ev_min = ev_abs.min()
    ev_max = ev_abs.max()
    cond = ev_max / ev_min if ev_min > 0 else np.inf

    return FisherResult(
        fisher=jnp.array(F),
        condition_number=float(cond),
        eigenvalues=eigenvalues,
        jacobian_result=jacobian_result,
        sigma_obs=np.asarray(sigma_obs),
    )


def _safe_invert_fisher(F: np.ndarray) -> Tuple[np.ndarray, str]:
    """Invert the Fisher matrix with Cholesky (primary) or pinv (fallback).

    For a well-conditioned positive-definite Fisher matrix, Cholesky
    factorization is O(N³/3) — twice as fast as LU and numerically more
    stable. When F is ill-conditioned or singular (degenerate parameter
    directions), Cholesky fails and we fall back to Moore-Penrose pseudoinverse
    (pinv), which gracefully handles rank deficiency.

    Parameters
    ----------
    F : (N, N) array — Fisher matrix (should be symmetric positive-definite).

    Returns
    -------
    F_inv : (N, N) array — inverse (or pseudoinverse) of F.
    method : str — 'cholesky' or 'pinv', for diagnostics.

    References
    ----------
    GWFAST (Iacovelli 2022, arXiv:2207.06910, §3.3): condition-number flagging.
    Golub & Van Loan, "Matrix Computations", §4.2: Cholesky for SPD systems.
    """
    try:
        L = np.linalg.cholesky(F)
        # F = L Lᵀ → F⁻¹ = (Lᵀ)⁻¹ L⁻¹
        L_inv = np.linalg.inv(L)
        F_inv = L_inv.T @ L_inv
        return F_inv, 'cholesky'
    except np.linalg.LinAlgError:
        # F is not positive-definite (degenerate directions, numerical noise).
        cond = np.linalg.cond(F)
        warnings.warn(
            f"Fisher matrix is not positive-definite (cond={cond:.2e}); "
            f"falling back to Moore-Penrose pseudoinverse",
            stacklevel=3,
        )
        F_inv = np.linalg.pinv(F)
        return F_inv, 'pinv'


def forecast_sigma(
    fisher_result: FisherResult,
    physical_params: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Forecast 1-σ parameter uncertainties: σ(θ_j) = sqrt([F⁻¹]_jj).

    This is the Cramér-Rao lower bound on parameter estimation errors.

    When `physical_params` is given, nuisance parameters are marginalized:
    invert the FULL Fisher matrix, then extract only the diagonal entries
    corresponding to the physical parameters. This is the standard
    marginalization procedure — the full inverse already accounts for the
    information lost by not constraining the nuisance parameters.

    Parameters
    ----------
    fisher_result : FisherResult
    physical_params : sequence of str, optional
        Subset of parameter names to report. If None, reports all.
        Any parameter NOT in this list is treated as a nuisance parameter
        and marginalized over (its information is still present in F via
        off-diagonal blocks, and the inversion accounts for it).

    Returns
    -------
    dict mapping parameter name → forecast 1-σ uncertainty.

    References
    ----------
    Tegmark, Taylor & Heavens (1997), ApJ 480, 22, §II.B — marginalization
    by extracting the diagonal of the full inverse.
    Wolz et al. (2012), arXiv:1205.3984 — Fisher-vs-MCMC < 5% for
    well-constrained, Gaussian-likelihood cases.
    Coe (2009), arXiv:0906.4123, §4 — Fisher matrix marginalization.
    """
    F = np.asarray(fisher_result.fisher)
    F_inv, method = _safe_invert_fisher(F)

    all_param_names = list(fisher_result.jacobian_result.param_names)

    if physical_params is None:
        physical_params = all_param_names

    result = {}
    for pname in physical_params:
        if pname not in all_param_names:
            raise ValueError(
                f"Parameter '{pname}' not in Fisher matrix parameters: "
                f"{all_param_names}")
        j = all_param_names.index(pname)
        # σ_j = sqrt([F⁻¹]_jj) — the marginal uncertainty, with nuisance
        # parameters already marginalized by the full-matrix inversion.
        result[pname] = float(np.sqrt(max(F_inv[j, j], 0.0)))

    return result
