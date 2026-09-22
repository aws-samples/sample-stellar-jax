"""Age-bias attribution via the X_c composition clock.

Computes ∂age/∂θ for physics parameters θ ∈ {opacity_factor, eps_nuc_factor,
alpha_mlt} — the physics-systematic age budget that PLATO's grids/MCMC cannot
produce. Assembles three validated ingredients via an ordinary least-squares
(OLS) projection in frequency space (unit weights — all modes weighted equally):

    δX_c/δθ = [Σ_i (∂ν_i/∂X_c)(∂ν_i/∂θ)] / [Σ_i (∂ν_i/∂X_c)²]
    δage/δθ = (dage/dX_c) · (δX_c/δθ)

Physics
-------
X_c (central hydrogen) is the evolutionary clock: X_c decreases monotonically
during main-sequence burning as H → He. Each physics perturbation δθ shifts
the p-mode frequencies δν_i = (∂ν_i/∂θ)·δθ. The projection asks: what change
in X_c (evolutionary state) would produce the same frequency pattern? That
X_c shift maps to an age shift via dage/dX_c.

This is an INFERENCE SENSITIVITY (age-bias attribution), NOT a forward
∂age/∂θ. It measures how much a 1% physics perturbation biases inferred age
when the observer fits to X_c-parameterized tracks. Validated at the near-ZAMS
operating point (N=3, 1.0 Msun, MODE-A: alpha_mlt=2.0, Z=0.014) — a method
demonstration.

Ingredients
-----------
1. ∂ν/∂X_c — from the IFT chain, validated to machine precision (#1222:
   ∂σ²/∂X_c = −0.599, rel_err = 1.7e-9).
2. Σ(∂ν/∂X_c · ∂ν/∂θ) — OLS numerator, computed via a single weighted-sum
   backward pass per parameter through evolve_star → IFT chain.
3. dage/dX_c — scalar slope from evolve_star (FD), the age-X_c relationship.

Memory optimization
-------------------
The OLS numerator Σ_i w_i · ∂ν_i/∂θ (where w_i = ∂ν_i/∂X_c) is computed
via ONE backward pass per parameter by defining the scalar objective
f(θ) = Σ_i w_i · ν_i(θ) and taking jax.grad(f). This gives
df/dθ = Σ_i w_i · ∂ν_i/∂θ = the numerator. Without this, each mode would
require a separate backward pass through evolve_star (~30+ GB each), causing
OOM on ci-mega (120 GB) for N_modes × 3 parameters.

References
----------
Cunha et al. (2021), arXiv:2110.03332 — PLATO H&H: largest age bias 8.66%
    from unaccounted diffusion physics (gravitational settling).
MESA star/private/history.f90:2973 — center_h1 = s%xa(h1,nz).
MESA star/private/history.f90:1397 — star_age.
Tegmark, Taylor & Heavens (1997), ApJ 480, 22 — error propagation formalism.
"""

from typing import Dict, Sequence

import numpy as np

from stellar_jax.config.mesa_config import MESA_CONFIG


_HONESTY = (
    "INFERENCE SENSITIVITY (age-bias attribution), NOT a forward ∂age/∂θ. "
    "Measures how much a unit physics perturbation biases inferred age when "
    "the observer fits to X_c-parameterized tracks. Validated at the near-ZAMS "
    "operating point (N=3, 1.0 Msun, MODE-A) — a method demonstration. "
    "The formula δage/δθ = (dage/dX_c) · Σ(∂ν/∂X_c · ∂ν/∂θ) / Σ(∂ν/∂X_c)² "
    "is an OLS projection of the physics perturbation onto the evolutionary "
    "direction in frequency space (unit weights — all modes weighted equally)."
)

# Fiducial values for the physics perturbation parameters at MODE-A
# (neutral perturbation = no modification to the underlying physics).
# alpha_mlt is sourced from MESA_CONFIG (the canonical MODE-A inlist).
_FIDUCIAL_VALUES: Dict[str, float] = {
    'opacity_factor': 1.0,
    'eps_nuc_factor': 1.0,
    'alpha_mlt': MESA_CONFIG['alpha_mlt'],
}


def _find_modes(glob, var, l_value, nu_min, nu_max, n_scan, n_steps_osc):
    """Find all eigenfrequency modes in the frequency bracket.

    Returns a list of mode_info dicts from compute_eigenfreq_from_structure_jax.
    """
    from stellar_jax.oscillations import compute_eigenfreq_from_structure_jax

    modes = []
    for idx in range(50):  # safety cap
        try:
            mode_info = compute_eigenfreq_from_structure_jax(
                glob, var, l=l_value, nu_min=nu_min, nu_max=nu_max,
                n_scan=n_scan, n_steps=n_steps_osc, mode_index=idx,
            )
            modes.append(mode_info)
        except ValueError:
            break
    return modes


def _compute_dnu_dXc(modes, X_fixed, mass, logL_fixed, logTe_fixed,
                     Z, alpha_mlt, y_fixed, atm_ratio):
    """Compute ∂ν/∂X_c for each mode via AD (fixed structure).

    This is the #1222-validated path: X_profile[0] → EOS → ρ, Γ₁ →
    coeffs → IFT eigenfreq → σ² → ν.
    """
    import jax
    import jax.numpy as jnp
    from stellar_jax.evolution import structure_to_fgong_jax
    from stellar_jax.oscillations import (
        build_oscillation_coeffs_jax, eigenfreq_from_coeffs,
    )
    from stellar_jax.oscillations.seismic_conversion import nu_from_sigma2

    dnu_dXc = np.zeros(len(modes))
    for i, mode_info in enumerate(modes):

        def _nu_of_Xc(xc_val, _m=mode_info):
            X_perturbed = X_fixed.at[0].set(xc_val)
            glob, var = structure_to_fgong_jax(
                jnp.float64(mass), logL_fixed, logTe_fixed, X_perturbed,
                jnp.float64(Z), jnp.float64(0.0), jnp.float64(alpha_mlt),
                y_henyey=y_fixed, atm_ratio=atm_ratio,
            )
            gd = build_oscillation_coeffs_jax(glob, var)
            s2 = eigenfreq_from_coeffs(
                gd['coeffs'], gd['x_grid'], _m['sigma2'],
                _m['l'], _m['x_steps'], _m['h_steps'], _m['factor'],
            )
            return nu_from_sigma2(s2, gd['factor'])

        dnu_dXc[i] = float(jax.grad(_nu_of_Xc)(X_fixed[0]))
    return dnu_dXc


def _compute_ols_numerator(pname, theta0, modes, weights, mass, Z,
                           alpha_mlt, n_steps, dt_fixed, Y_init, f_ov):
    """Compute OLS numerator Σ_i w_i · ∂ν_i/∂θ via a single backward pass.

    Defines the scalar objective f(θ) = Σ_i w_i · ν_i(θ) and computes
    df/dθ = Σ_i w_i · ∂ν_i/∂θ in ONE jax.grad call through evolve_star.

    This is memory-efficient: one backward pass instead of N_modes separate
    backward passes. Each backward pass through evolve_star requires ~30+ GB;
    the per-mode approach would need N_modes × 30 GB, causing OOM on ci-mega
    (120 GB) for even 4 modes.

    Parameters
    ----------
    pname : str
        Physics parameter name.
    theta0 : jax scalar
        Fiducial value.
    modes : list of dict
        Mode info from _find_modes.
    weights : ndarray
        Weights w_i = ∂ν_i/∂X_c (from _compute_dnu_dXc).
    mass, Z, alpha_mlt : float
        Stellar parameters.
    n_steps, dt_fixed : int, float
        Evolution config.
    Y_init, f_ov : float
        Initial helium, overshoot.

    Returns
    -------
    float
        Σ_i w_i · ∂ν_i/∂θ — the OLS numerator.
    """
    import warnings
    import jax
    import jax.numpy as jnp
    from stellar_jax.evolution import evolve_star, structure_to_fgong_jax
    from stellar_jax.oscillations import (
        build_oscillation_coeffs_jax, eigenfreq_from_coeffs,
    )
    from stellar_jax.oscillations.seismic_conversion import nu_from_sigma2

    # Convert weights to JAX array (constant — not differentiated)
    w = jnp.array(weights)

    def _weighted_nu_sum(theta_val):
        """f(θ) = Σ_i w_i · ν_i(θ) — scalar objective for one backward pass."""
        kw = dict(Z=Z, max_steps=n_steps, alpha_mlt=alpha_mlt,
                  Y_init=Y_init, f_ov=f_ov,
                  diffusion=False, fixed_dt=dt_fixed, adaptive_mesh=False)
        if pname == 'alpha_mlt':
            kw['alpha_mlt'] = theta_val
        else:
            kw[pname] = theta_val
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = evolve_star(jnp.float64(mass), **kw)
        glob, var = structure_to_fgong_jax(
            jnp.float64(mass), r['log_L_final'], r['log_Teff_final'],
            r['X_profile'], jnp.float64(Z), jnp.float64(0.0),
            jnp.float64(kw['alpha_mlt']),
            y_henyey=r['y_henyey_final'], atm_ratio=r.get('atm_ratio'),
        )
        gd = build_oscillation_coeffs_jax(glob, var)

        # Sum over modes: f = Σ_i w_i · ν_i(θ)
        total = jnp.float64(0.0)
        for j, mode_info in enumerate(modes):
            s2 = eigenfreq_from_coeffs(
                gd['coeffs'], gd['x_grid'], mode_info['sigma2'],
                mode_info['l'], mode_info['x_steps'], mode_info['h_steps'],
                mode_info['factor'],
            )
            nu_j = nu_from_sigma2(s2, gd['factor'])
            total = total + w[j] * nu_j
        return total

    return float(jax.grad(_weighted_nu_sum)(theta0))


def _compute_dage_dXc(r_ref):
    """Compute dage/dX_c from the evolution track (FD slope).

    Uses the last two valid points of the age vs center_h1 arrays.
    X_c decreases with age, so dage/dX_c < 0.

    NOTE: Two-point FD is adequate for the near-ZAMS operating point (N=3)
    where the age(X_c) relationship is nearly linear. For longer evolutions
    (mid-MS/RGB), a linear fit over more points would be more robust.
    """
    age_arr = np.array([float(a) for a in r_ref['star_age']])
    xc_arr = np.array([float(x) for x in r_ref['center_h1']])
    valid = np.isfinite(age_arr) & np.isfinite(xc_arr) & (age_arr > 0)
    age_v = age_arr[valid]
    xc_v = xc_arr[valid]
    if len(age_v) < 2:
        raise RuntimeError("Not enough valid points for dage/dXc estimation")
    return float((age_v[-1] - age_v[-2]) / (xc_v[-1] - xc_v[-2]))


def compute_age_bias_attribution(
    physics_params: Sequence[str] = ('opacity_factor', 'eps_nuc_factor', 'alpha_mlt'),
    mass: float = 1.0,
    Z: float = MESA_CONFIG['Z'],
    alpha_mlt: float = MESA_CONFIG['alpha_mlt'],
    Y_init: float = MESA_CONFIG['Y_init'],
    f_ov: float = MESA_CONFIG['f_ov'],
    n_steps: int = 3,
    dt_fixed: float = 1e7,
    l_value: int = 0,
    nu_min: float = 2000.0,
    nu_max: float = 4500.0,
    n_scan: int = 250,
    n_steps_osc: int = 8000,
) -> Dict:
    """Compute the physics-systematic age budget via the X_c composition clock.

    For each physics parameter θ, reports δage/δθ — the age bias (in years)
    per unit perturbation of θ. See module docstring for physics and caveats.

    Parameters
    ----------
    physics_params : sequence of str
        Physics parameters to attribute. Default: opacity_factor, eps_nuc_factor,
        alpha_mlt — the three parameters with validated seismic gradients.
    mass, Z, alpha_mlt : float
        Stellar parameters. Defaults from MESA_CONFIG (MODE-A): 1.0 Msun,
        Z=0.014, alpha_mlt=2.0.
    Y_init, f_ov : float
        Initial helium and overshoot. Defaults from MESA_CONFIG (MODE-A):
        Y_init=0.2695, f_ov=0.0.
    n_steps, dt_fixed : int, float
        Evolution steps and timestep [yr]. Defaults: 3, 1e7 (near-ZAMS).
    l_value : int
        Angular degree for modes. Default: 0 (radial).
    nu_min, nu_max, n_scan, n_steps_osc : float, float, int, int
        Mode search parameters.

    Returns
    -------
    dict with keys:
        dage_dtheta : dict[str, float] — δage/δθ [yr] per parameter
        dage_dtheta_pct : dict[str, float] — δage/δθ / age × 100 [%]
        dXc_dtheta : dict[str, float] — δX_c/δθ per parameter
        ols_numerator : dict[str, float] — Σ(∂ν/∂X_c · ∂ν/∂θ) per parameter
        dnu_dXc : ndarray — ∂ν_i/∂X_c per mode
        dage_dXc : float — dage/dX_c from evolution track
        age_ref, Xc_ref : float — reference age and X_c
        nu_ref : ndarray — reference mode frequencies [μHz]
        modes : list — mode info dicts
        norm_sq : float — Σ(∂ν/∂X_c)²
        honesty : str — disclaimer
    """
    import gc
    import warnings
    import jax
    import jax.numpy as jnp
    from stellar_jax.evolution import evolve_star, structure_to_fgong_jax

    # ── Step 1: Reference evolution ──
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_ref = evolve_star(
            jnp.float64(mass), Z=Z, max_steps=n_steps, alpha_mlt=alpha_mlt,
            Y_init=Y_init, f_ov=f_ov,
            diffusion=False, fixed_dt=dt_fixed, adaptive_mesh=False,
        )
    age_ref = float(r_ref['star_age'][-1])
    Xc_ref = float(r_ref['center_h1'][-1])

    # GP-4: COMPOSITION_STOP_GRADIENT — fix structure at fiducial, differentiate
    # only through X_profile[0] for the ∂ν/∂X_c path (same as validated).
    y_fixed = jax.lax.stop_gradient(r_ref['y_henyey_final'])  # GP-4: COMPOSITION_STOP_GRADIENT
    logL_fixed = jax.lax.stop_gradient(r_ref['log_L_final'])  # GP-4: COMPOSITION_STOP_GRADIENT
    logTe_fixed = jax.lax.stop_gradient(r_ref['log_Teff_final'])  # GP-4: COMPOSITION_STOP_GRADIENT
    X_fixed = jax.lax.stop_gradient(r_ref['X_profile'])  # GP-4: COMPOSITION_STOP_GRADIENT
    atm_ratio = r_ref.get('atm_ratio')

    # ── Step 2: Find modes ──
    glob_ref, var_ref = structure_to_fgong_jax(
        jnp.float64(mass), logL_fixed, logTe_fixed, X_fixed,
        jnp.float64(Z), jnp.float64(0.0), jnp.float64(alpha_mlt),
        y_henyey=y_fixed, atm_ratio=atm_ratio,
    )
    modes = _find_modes(glob_ref, var_ref, l_value, nu_min, nu_max,
                        n_scan, n_steps_osc)
    if not modes:
        raise RuntimeError(
            f"No modes found for l={l_value} in [{nu_min}, {nu_max}] μHz")
    nu_refs = np.array([float(m['nu']) for m in modes])

    # ── Step 3: ∂ν/∂X_c (AD, fixed structure — path) ──
    dnu_dXc = _compute_dnu_dXc(
        modes, X_fixed, mass, logL_fixed, logTe_fixed,
        Z, alpha_mlt, y_fixed, atm_ratio)

    # ── Step 4: OLS numerator per parameter (single backward pass each) ──
    # Memory optimization: instead of N_modes × N_params backward passes through
    # evolve_star (~30+ GB each), compute f(θ) = Σ_i w_i · ν_i(θ) where
    # w_i = ∂ν_i/∂X_c. Then df/dθ = Σ_i w_i · ∂ν_i/∂θ = the OLS numerator.
    # Total: 3 backward passes (one per parameter) instead of N_modes × 3.
    ols_num = {}
    for pname in physics_params:
        theta0 = jnp.float64(
            alpha_mlt if pname == 'alpha_mlt' else _FIDUCIAL_VALUES[pname])
        ols_num[pname] = _compute_ols_numerator(
            pname, theta0, modes, dnu_dXc, mass, Z, alpha_mlt,
            n_steps, dt_fixed, Y_init, f_ov)
        # Release XLA buffers between backward passes to reduce peak memory
        gc.collect()

    # ── Step 5: dage/dX_c (from evolution track) ──
    dage_dXc = _compute_dage_dXc(r_ref)

    # ── Step 6: OLS composition ──
    norm_sq = float(np.sum(dnu_dXc**2))
    dXc_dtheta, dage_dtheta, dage_dtheta_pct = {}, {}, {}
    for pname in physics_params:
        proj = ols_num[pname] / norm_sq
        dXc_dtheta[pname] = proj
        dage_dt = dage_dXc * proj
        dage_dtheta[pname] = dage_dt
        dage_dtheta_pct[pname] = (dage_dt / age_ref * 100.0) if age_ref > 0 else 0.0

    return {
        'dage_dtheta': dage_dtheta, 'dage_dtheta_pct': dage_dtheta_pct,
        'dXc_dtheta': dXc_dtheta, 'ols_numerator': ols_num,
        'dnu_dXc': dnu_dXc, 'dage_dXc': dage_dXc,
        'age_ref': age_ref, 'Xc_ref': Xc_ref, 'nu_ref': nu_refs,
        'modes': modes, 'norm_sq': norm_sq, 'honesty': _HONESTY,
    }
