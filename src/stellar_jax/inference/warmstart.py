"""Warm-start factories for infer_parameters: observables → parameter guess.

Two modes, both producing a warmstart_fn(observed, sigma, param_names) → theta_init:

1. make_warmstart_fn (pure-mapping): standard astrophysical relations
   (mass–luminosity, log g → mass, enrichment law). No external deps.
2. make_dsee_warmstart_fn (emulator-based): uses DSEEInterface to run a
   fast (~0.1ms/eval) emulator optimization against the observed quantities,
   producing a tighter initial guess than the pure mapping. Falls back to
   pure-mapping if DSEE/PyTorch is unavailable.

Neither is in the gradient path — used only for initialization before the
exact LM loop (infer_parameters).
"""

import numpy as np

from stellar_jax.config.constants import Y_BBN, DY_DZ
from stellar_jax.config.mesh_defaults import ALPHA_MLT, F_OV
from stellar_jax.inference.metallicity import feh_to_z

# DSEE ↔ stellar_jax parameter/observable name mappings (module-level constants)
_PARAM_TO_DSEE_KWARG = {
    'mass': 'mass', 'M': 'mass', 'Z': 'Z',
    'alpha_mlt': 'alpha_mlt', 'alpha': 'alpha_mlt',
    'f_ov': 'f_ov', 'Y_init': 'Y_init',
    'opacity_factor': 'opacity_factor', 'opf': 'opacity_factor',
    't_max': 'age_yr',
}

_OBS_TO_DSEE_KEY = {
    'logl': 'log_L', 'l': 'log_L',
    'logteff': 'log_Teff', 'teff': 'log_Teff',
    'logr': 'log_R', 'r': 'log_R',
    'logg': 'log_g', 'g': 'log_g',
}

_PARAM_DEFAULTS = {
    'mass': 1.0, 'Z': 0.014,
    'alpha_mlt': float(ALPHA_MLT), 'f_ov': float(F_OV),
    'Y_init': float(Y_BBN + DY_DZ * 0.014),
    'opacity_factor': 1.0, 't_max': 4.603e9,
}


def warmstart_params_from_observables(
    log_Teff_obs, log_L_obs, log_g_obs=None, feh_obs=None,
    mass_guess=1.0, age_guess_yr=4.603e9,
):
    """Map observed classical observables to plausible stellar-jax parameters.

    This is a PURE MAPPING function (no emulator needed). Given observed
    surface properties, it returns a reasonable starting point for the
    stellar-jax parameter search. The mapping uses standard astrophysical
    relations — it does NOT run the emulator.

    This is useful when DSEE is not installed but you still want a physically
    motivated initial guess.

    Parameters
    ----------
    log_Teff_obs : float
        Observed log10(T_eff/K).
    log_L_obs : float
        Observed log10(L/L_sun).
    log_g_obs : float or None
        Observed log10(g). If given, refines mass estimate.
    feh_obs : float or None
        Observed [Fe/H]. If given, converts to Z.

    Returns
    -------
    dict
        Initial guess: mass, Z, alpha_mlt, Y_init, t_max.
    """
    # Z from [Fe/H]
    if feh_obs is not None:
        Z = feh_to_z(feh_obs)
    else:
        Z = 0.014  # solar neighborhood default

    Y_init = Y_BBN + DY_DZ * Z

    # Mass estimate from log g + log R (if log_g available)
    # log g = log(GM/R^2) → log M = log g + 2*log R - log(G*M_sun/R_sun^2)
    # For a rough estimate, use the mass-luminosity relation instead:
    # L/L_sun ≈ (M/M_sun)^3.5 for MS → M ≈ (L/L_sun)^(1/3.5)
    L_ratio = 10.0 ** log_L_obs
    mass_from_L = L_ratio ** (1.0 / 3.5) if L_ratio > 0 else mass_guess

    # Refine with log_g if available (log g gives M/R^2)
    if log_g_obs is not None:
        # R from L and Teff: L = 4πR²σT⁴ → R/R_sun = (L/L_sun)^0.5 * (T_sun/T)^2
        from stellar_jax.config.constants import T_sun, log_g_sun
        T_eff = 10.0 ** log_Teff_obs
        R_ratio = np.sqrt(L_ratio) * (T_sun / T_eff) ** 2
        # g/g_sun = M/R^2 (in solar units)
        g_ratio = 10.0 ** (log_g_obs - log_g_sun)
        mass_from_g = g_ratio * R_ratio ** 2
        # Average the two estimates (both approximate for evolved stars)
        mass = 0.5 * (mass_from_L + mass_from_g)
    else:
        mass = mass_from_L

    # Clamp mass to physical range
    mass = float(np.clip(mass, 0.5, 3.0))

    # α_MLT: start at the solar-calibrated value (code-dependent, ~1.9)
    alpha_mlt = float(ALPHA_MLT)

    # Age: crude estimate from MS lifetime scaling τ_MS ≈ 10 Gyr * (M/M_sun)^{-2.5}
    # but prefer the user's guess if provided
    age = age_guess_yr

    return {
        'mass': mass,
        'Z': float(Z),
        'alpha_mlt': alpha_mlt,
        'f_ov': float(F_OV),
        'Y_init': float(Y_init),
        't_max': age,
        'diffusion': True,
    }


def make_warmstart_fn(param_names, observed_names=None, **fixed_kwargs):
    """Create a warmstart_fn callable for use with infer_parameters.

    Returns a function with signature (observed, sigma, param_names) → theta_init
    that maps observed photometric quantities through warmstart_params_from_observables
    (or DSEEInterface if available) to produce a physically motivated initial guess.

    Parameters
    ----------
    param_names : list of str
        Names of the parameters being inferred, in the order they appear in theta.
        Supported names: 'mass', 'Z', 'alpha_mlt', 'opacity_factor', 'f_ov',
        'Y_init', 't_max'.
    observed_names : list of str or None
        Names of the observed quantities (same order as the observed vector).
        Used to extract log_Teff_obs, log_L_obs, log_g_obs for the warmstart.
        If None, assumes the first element is logL and second is logTeff
        (the convention used by infer_parameters tests).
    **fixed_kwargs
        Fixed keyword arguments passed to warmstart_params_from_observables
        (e.g. mass_guess, age_guess_yr, feh_obs).

    Returns
    -------
    callable
        A function (observed, sigma, param_names) → np.ndarray of shape (n_params,)
        suitable for passing as warmstart_fn to infer_parameters.

    Example
    -------
    >>> warmstart = make_warmstart_fn(
    ...     param_names=['mass', 'opacity_factor'],
    ...     observed_names=['logL', 'logTeff'],
    ... )
    >>> result = infer_parameters(forward_fn, observed, sigma,
    ...                           warmstart_fn=warmstart, param_names=['mass', 'opacity_factor'])
    """
    # Default mapping: param_name → key in warmstart dict
    _PARAM_KEY_MAP = {
        'mass': 'mass',
        'M': 'mass',
        'Z': 'Z',
        'alpha_mlt': 'alpha_mlt',
        'alpha': 'alpha_mlt',
        'opacity_factor': None,  # no direct warmstart (default 1.0)
        'opf': None,
        'eps_nuc_factor': None,  # no direct warmstart (default 1.0)
        'diffusion_factor': None,  # no direct warmstart (default 1.0)
        'f_ov': 'f_ov',
        'Y_init': 'Y_init',
        't_max': 't_max',
    }

    # Default values for params not in the warmstart dict
    _PARAM_DEFAULTS = {
        'opacity_factor': 1.0,
        'opf': 1.0,
        'eps_nuc_factor': 1.0,
        'diffusion_factor': 1.0,
    }

    def _warmstart_fn(observed_vec, sigma_vec, pnames):
        """Map observed quantities to a theta_init vector."""
        observed_vec = np.asarray(observed_vec)

        # Extract photometric observables from the observed vector
        log_Teff_obs = None
        log_L_obs = None
        log_g_obs = None

        if observed_names is not None:
            for i, name in enumerate(observed_names):
                name_lower = name.lower().replace('_', '')
                if i < len(observed_vec):
                    if 'teff' in name_lower or 'logteff' in name_lower:
                        log_Teff_obs = float(observed_vec[i])
                    elif ('logl' in name_lower or name_lower == 'l'
                          or name_lower == 'logl'):
                        log_L_obs = float(observed_vec[i])
                    elif 'logg' in name_lower:
                        log_g_obs = float(observed_vec[i])
        else:
            # Default convention: last two elements are [logL, logTeff]
            if len(observed_vec) >= 2:
                log_L_obs = float(observed_vec[-2])
                log_Teff_obs = float(observed_vec[-1])

        # Fall back to defaults if observables not identified
        if log_Teff_obs is None:
            log_Teff_obs = 3.76  # ~solar
        if log_L_obs is None:
            log_L_obs = 0.0  # ~solar

        # Call the pure-mapping warmstart
        guess = warmstart_params_from_observables(
            log_Teff_obs=log_Teff_obs,
            log_L_obs=log_L_obs,
            log_g_obs=log_g_obs,
            **fixed_kwargs,
        )

        # Build theta vector from the guess dict
        pnames_to_use = pnames if pnames is not None else param_names
        theta = np.zeros(len(pnames_to_use))
        for i, pname in enumerate(pnames_to_use):
            key = _PARAM_KEY_MAP.get(pname)
            if key is not None and key in guess:
                theta[i] = guess[key]
            elif pname in _PARAM_DEFAULTS:
                theta[i] = _PARAM_DEFAULTS[pname]
            elif pname in guess:
                theta[i] = guess[pname]
            else:
                # Unknown param — use 1.0 as a safe default for multiplicative factors
                theta[i] = 1.0

        return theta

    return _warmstart_fn


def _dsee_refine_guess(theta_rough, dsee, pnames_to_use, observed_vec,
                      sigma_vec, dsee_obs_indices, n_refine,
                      param_defaults, param_to_dsee_kwarg):
    """Refine a rough guess via Nelder-Mead against the DSEE emulator."""
    from scipy.optimize import minimize

    def _cost(theta_vec):
        kwargs = dict(param_defaults)
        for i, pname in enumerate(pnames_to_use):
            dsee_kwarg = param_to_dsee_kwarg.get(pname)
            if dsee_kwarg is not None:
                kwargs[dsee_kwarg] = float(theta_vec[i])
        if 'age_yr' not in kwargs and 't_max' in kwargs:
            kwargs['age_yr'] = kwargs.pop('t_max')
        try:
            pred = dsee.predict(**kwargs)
        except Exception:
            return 1e10
        chi2 = 0.0
        for i, key in dsee_obs_indices:
            if key in pred:
                chi2 += ((pred[key] - float(observed_vec[i])) / float(sigma_vec[i])) ** 2
        return chi2

    result = minimize(
        _cost, theta_rough, method='Nelder-Mead',
        options={'maxiter': n_refine, 'xatol': 1e-3, 'fatol': 1e-3})
    if result.success or result.fun < _cost(theta_rough):
        return result.x
    return theta_rough


def _resolve_dsee_interface(dsee_interface, model_path, fallback_to_pure):
    """Resolve the DSEE emulator interface, with optional fallback.

    Returns (dsee_instance, dsee_available).
    """
    if dsee_interface is not None:
        return dsee_interface, True
    try:
        from stellar_jax.inference.emulator import DSEEInterface
        return DSEEInterface(model_path=model_path), True
    except (ImportError, FileNotFoundError) as e:
        if not fallback_to_pure:
            raise ImportError(
                f"DSEE emulator not available: {e}. "
                "Install PyTorch + zuko + CONF1DENCE, or set "
                "fallback_to_pure=True to use pure-mapping warmstart."
            ) from e
        return None, False


def make_dsee_warmstart_fn(
    param_names,
    observed_names=None,
    dsee_interface=None,
    model_path=None,
    n_refine=5,
    fallback_to_pure=True,
    **fixed_kwargs,
):
    """Create a warmstart_fn that uses the DSEE emulator for a tighter initial guess.

    The DSEE emulator (arXiv:2604.06348, Ying et al. 2025) predicts classical
    stellar observables (log_Teff, log_L, log_R, log_g) from parameters in ~0.1 ms.
    This factory produces a warmstart_fn that:
    1. Gets a rough estimate from the pure-mapping (mass-luminosity relation etc.)
    2. Refines it via a lightweight Nelder-Mead optimization against the emulator
       as a fast proxy for the full stellar evolution model.
    3. Returns the refined estimate as theta_init for infer_parameters.

    The emulator is NOT in the gradient path — this runs once, before the LM loop,
    to produce a better starting point. The exact gradients then refine.

    Parameters
    ----------
    param_names : list of str
        Names of the parameters being inferred, in the order they appear in theta.
        Supported: 'mass', 'Z', 'alpha_mlt', 'opacity_factor', 'f_ov', 'Y_init', 't_max'.
    observed_names : list of str or None
        Names of the observed quantities. Used to match observations to DSEE outputs.
        Supported: 'logL', 'logTeff', 'logR', 'logg'. If None, assumes the last two
        elements are [logL, logTeff].
    dsee_interface : DSEEInterface or None
        Pre-constructed DSEEInterface instance. If None, one will be created
        from model_path (or the default search path).
    model_path : str or None
        Path to the DSEE.model file. Used only if dsee_interface is None.
    n_refine : int
        Number of Nelder-Mead iterations for emulator-level refinement (default 5).
        More iterations = tighter initial guess, but diminishing returns past ~10
        (the exact solver will refine anyway).
    fallback_to_pure : bool
        If True (default), falls back to pure-mapping warmstart when DSEE/PyTorch
        is not available (no error). If False, raises ImportError.
    **fixed_kwargs
        Fixed keyword arguments passed to warmstart_params_from_observables for
        the initial rough estimate (e.g. age_guess_yr, feh_obs).

    Returns
    -------
    callable
        A function (observed, sigma, param_names) → np.ndarray of shape (n_params,)
        suitable for passing as warmstart_fn to infer_parameters.

    Notes
    -----
    The DSEE emulator requires PyTorch + zuko. If these are not installed,
    behavior depends on fallback_to_pure:
    - True: silently falls back to make_warmstart_fn (pure mapping).
    - False: raises ImportError immediately.

    The emulator-level optimization uses scipy.optimize.minimize (Nelder-Mead)
    with the DSEE forward model as a fast proxy. This is ~1000x faster than
    running the full stellar evolution, so even 50 Nelder-Mead evals are
    negligible compared to one infer_parameters iteration.

    References
    ----------
    Ying et al. (2025), "DSEE: Dartmouth Stellar Evolution Emulator",
        arXiv:2604.06348.
    Issue #473, epic #466 Arm B (emulator warm-start).
    """
    # Try to set up the DSEE interface
    _dsee, _dsee_available = _resolve_dsee_interface(
        dsee_interface, model_path, fallback_to_pure)

    # If DSEE is not available, fall back to pure-mapping
    if not _dsee_available:
        return make_warmstart_fn(
            param_names=param_names,
            observed_names=observed_names,
            **fixed_kwargs,
        )

    # Build the pure-mapping factory as a fallback for the initial rough guess
    _pure_warmstart = make_warmstart_fn(
        param_names=param_names,
        observed_names=observed_names,
        **fixed_kwargs,
    )

    def _dsee_warmstart_fn(observed_vec, sigma_vec, pnames):
        """DSEE-refined warm-start: pure-mapping + emulator optimization."""
        observed_vec = np.asarray(observed_vec)
        sigma_vec = np.asarray(sigma_vec)

        # Step 1: Get a rough initial guess from the pure mapping
        theta_rough = _pure_warmstart(observed_vec, sigma_vec, pnames)

        # Step 2: Identify which observed quantities map to DSEE outputs
        obs_keys = []
        if observed_names is not None:
            for name in observed_names:
                key = _OBS_TO_DSEE_KEY.get(name.lower().replace('_', ''), None)
                obs_keys.append(key)
        else:
            # Default: last two are [logL, logTeff]
            n_obs = len(observed_vec)
            obs_keys = [None] * n_obs
            if n_obs >= 2:
                obs_keys[-2] = 'log_L'
                obs_keys[-1] = 'log_Teff'

        # Only refine if we have at least one observable that DSEE predicts
        dsee_obs_indices = [(i, key) for i, key in enumerate(obs_keys)
                           if key is not None]

        if not dsee_obs_indices:
            return theta_rough

        # Refine via DSEE emulator (fast Nelder-Mead, ~0.1ms/eval).
        pnames_to_use = pnames if pnames is not None else param_names
        return _dsee_refine_guess(
            theta_rough, _dsee, pnames_to_use, observed_vec,
            sigma_vec, dsee_obs_indices, n_refine,
            _PARAM_DEFAULTS, _PARAM_TO_DSEE_KWARG)

    return _dsee_warmstart_fn
