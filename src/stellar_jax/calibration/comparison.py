"""MESA / Model S comparison functions.

NOT differentiable — uses numpy I/O and Python control flow.

Reference: Christensen-Dalsgaard et al. (1996), Science 272, 1286.
"""
import logging
import os

import jax
import jax.numpy as jnp
import numpy as np

from stellar_jax.config.constants import (
    SECONDS_PER_YEAR,
)
from stellar_jax.config.mesh_defaults import (
    N_SHOOT, N_NEWTON_WARM, COMP_MFRACS,
)
from stellar_jax.config.calibration import ALPHA_SOLAR, Y0_SOLAR
from stellar_jax.config.physics_floors import PGAS_FRAC_FLOOR
from stellar_jax.config.constants import a_rad_opal as _a_rad_opal
from stellar_jax.microphysics.eos import eos_lookup_bicubic
from stellar_jax.structure import (
    interp_X_at_mass, shoot_xprofile, newton_solve_xprofile,
    get_structure_profile,
)

from stellar_jax.calibration.interpolation import lagrange_interp_3pt

logger = logging.getLogger(__name__)


# ======================================================================
# Private helpers decomposed from compare_model_s
# ======================================================================

def _resolve_model_s_data_dir():
    """Locate the Model S data directory.

    Searches relative to this module's project root and the cwd.
    Returns the directory path or None if not found.

    The data directory is expected at ``data/model_s/`` relative to the
    project root. Override with the ``STELLAR_MODEL_S_DIR`` environment
    variable if the data lives elsewhere (e.g. in a container).
    """
    env_override = os.environ.get('STELLAR_MODEL_S_DIR')
    if env_override and os.path.isfile(os.path.join(env_override, 'model_s_cs.dat')):
        return env_override

    _module_dir = os.path.dirname(os.path.realpath(__file__))
    _project_dir = os.path.dirname(_module_dir)
    _candidates = [
        os.path.join(_project_dir, 'data', 'model_s'),
        os.path.join(os.getcwd(), 'data', 'model_s'),
    ]
    for _d in _candidates:
        _cs_file = os.path.join(_d, 'model_s_cs.dat')
        if os.path.isfile(_cs_file):
            return _d
    return None


def _load_model_s_data(data_dir):
    """Load Model S sound-speed and density reference data.

    Args:
        data_dir: Path to the model_s data directory.

    Returns:
        Tuple of (cs_data, rho_data) where each is a numpy array with
        columns [r/R, value], or (cs_data, None) if density file is missing.
        Both are sorted ascending in r/R for interpolation.

    Reference: Christensen-Dalsgaard et al. (1996), Science 272, 1286.
        Data from https://users-phys.au.dk/~jcd/solar_models/
    """
    cs_path = os.path.join(data_dir, 'model_s_cs.dat')
    rho_path = os.path.join(data_dir, 'model_s_rho.dat')

    cs_data = np.loadtxt(cs_path)
    # Sort ascending in r/R for np.interp
    cs_r_asc = cs_data[:, 0][::-1]
    cs_val_asc = cs_data[:, 1][::-1]

    rho_data = None
    if os.path.exists(rho_path):
        rho_raw = np.loadtxt(rho_path)
        rho_r_asc = rho_raw[:, 0][::-1]
        rho_val_asc = rho_raw[:, 1][::-1]
        rho_data = (rho_r_asc, rho_val_asc)

    return (cs_r_asc, cs_val_asc), rho_data


def _solar_model_at_age(M_solar, Z, alpha_mlt, max_steps):
    """Evolve a solar model and return reconverged structure at 4.57 Gyr.

    This is the shared sub-step that:
      (1) Evolves 1 M☉ with z_feedback via evolve_solar
      (2) Interpolates (logL, logTe) to exactly 4.57 Gyr (Lagrange quadratic)
      (3) Reconverges Newton at solar age
      (4) Computes Z-profile via CZ homogenization
      (5) Re-shoots with Z_profile for self-consistent structure

    Args:
        M_solar: Stellar mass in solar masses (normally 1.0).
        Z: Metallicity.
        alpha_mlt: MLT mixing-length parameter.
        max_steps: Maximum evolution steps for evolve_solar.

    Returns:
        Tuple of (shell_data_zp, X_final, Z_profile, logL_f, logTe_f, r_over_R)
        where shell_data_zp is the reconverged structure with Z_profile, and
        r_over_R is the radial coordinate array from get_structure_profile.
    """
    from stellar_jax.calibration.solar import evolve_solar  # lazy import to avoid circular dependency

    ages_yr, log_L_arr, log_R_arr, log_Te_arr, X_final, Y_final, Z_profile_evolved = evolve_solar(
        jnp.float64(alpha_mlt), jnp.float64(Y0_SOLAR), Z,
        max_steps=max_steps, t_max=4.57e9, z_feedback=True)

    # Interpolate logL and logTe to exactly 4.57 Gyr (quadratic, 3-point Lagrange)
    ages = ages_yr * SECONDS_PER_YEAR
    t_target = 4.57e9 * SECONDS_PER_YEAR
    idx = int(jnp.searchsorted(ages, t_target)) - 1
    idx = max(1, min(idx, max_steps - 2))
    t0, t1, t2 = float(ages[idx-1]), float(ages[idx]), float(ages[idx+1])
    tt = float(t_target)

    logL_final = lagrange_interp_3pt(t0, t1, t2,
                                      float(log_L_arr[idx-1]), float(log_L_arr[idx]), float(log_L_arr[idx+1]),
                                      tt)
    logTe_final = lagrange_interp_3pt(t0, t1, t2,
                                       float(log_Te_arr[idx-1]), float(log_Te_arr[idx]), float(log_Te_arr[idx+1]),
                                       tt)
    logL_final = jnp.float64(float(logL_final))
    logTe_final = jnp.float64(float(logTe_final))

    M_j = jnp.asarray(float(M_solar), dtype=jnp.float64)
    Z_j = jnp.asarray(float(Z), dtype=jnp.float64)
    alpha_mlt_j = jnp.asarray(float(alpha_mlt), dtype=jnp.float64)
    t_age_final = jnp.float64(4.57e9)

    # Reconverge Newton
    logL_f, logTe_f = newton_solve_xprofile(
        M_j, X_final, Z_j, t_age_final, logL_final, logTe_final, alpha_mlt_j,
        N_NEWTON_WARM)
    _, shell_data_final, _ = shoot_xprofile(
        M_j, logL_f, logTe_f, X_final, Z_j, t_age_final, alpha_mlt_j)

    # CZ Z-homogenization (see evolution.py comment block for physics rationale)
    nrad_final = shell_data_final[:, 2]
    nad_final = shell_data_final[:, 3]
    mf_shells_final = shell_data_final[:, 1]
    mf_sorted_final = mf_shells_final[::-1]
    delta_nab_final = (nrad_final - nad_final)[::-1]
    comp_mfracs = jnp.array(COMP_MFRACS)
    delta_nab_comp = jnp.interp(comp_mfracs, mf_sorted_final, delta_nab_final,
                                left=delta_nab_final[0], right=delta_nab_final[-1])
    env_cz_mask = jax.nn.sigmoid(30.0 * delta_nab_comp)
    log_conv = jnp.log(jnp.clip(env_cz_mask, 1e-10, 1.0))
    soft_core_mask = jnp.clip(jnp.exp(jnp.cumsum(log_conv)), 0.0, 1.0)
    non_core = 1.0 - (soft_core_mask > 0.5).astype(jnp.float64)
    env_cz_hard = (env_cz_mask * non_core > 0.5).astype(jnp.float64)
    Z_cz_surface = Z_profile_evolved[-1]
    Z_profile = jnp.where(
        (env_cz_hard > 0.5) & (Z_profile_evolved < Z_cz_surface),
        Z_cz_surface,
        Z_profile_evolved)

    # Evaluate structure with Z_profile
    _, shell_data_zp, _ = shoot_xprofile(
        M_j, logL_f, logTe_f, X_final, Z_j, t_age_final, alpha_mlt_j,
        Z_profile=Z_profile)

    # Get r/R from get_structure_profile
    profile = get_structure_profile(M_j, logL_f, logTe_f,
                                    X_final, Z_j, t_age_final, alpha_mlt_j,
                                    Z_profile=Z_profile)
    r_over_R = profile[:, 0]

    return shell_data_zp, X_final, Z_profile, logL_f, logTe_f, r_over_R


def _compute_sound_speed_and_density(shell_data_zp, X_final, Z_profile,
                                     r_over_R, use_structure_density):
    """Compute sound speed and density from reconverged shell data.

    Sound speed: c_s = sqrt(gamma1 * P / rho), where gamma1 uses the
    radiation-pressure-corrected formula (Chandrasekhar 1939, §56).

    Args:
        shell_data_zp: Shell data array from the Z-profile reconverged model.
        X_final: Final hydrogen composition profile.
        Z_profile: Metal composition profile.
        r_over_R: Radial coordinate array (r/R_star).
        use_structure_density: If True, use density from the structure solve
            directly (physically self-consistent).

    Returns:
        Tuple of (r_valid, c_valid, rho_valid) — arrays restricted to
        0.05 < r/R < 0.90 (excluding center and near-surface).
    """
    # Extract T, P from shell_data for sound speed calculation
    logT_arr = shell_data_zp[:, 6]
    logP_arr = shell_data_zp[:, 9]

    # Compute Pgas = P_total - P_rad
    a_rad_c = _a_rad_opal
    T_arr = 10.0**logT_arr
    P_total_arr = 10.0**logP_arr
    P_rad_arr = a_rad_c * T_arr**4 / 3.0
    P_gas_arr = jnp.maximum(P_total_arr - P_rad_arr, PGAS_FRAC_FLOOR * P_total_arr)
    logPgas_arr = jnp.log10(P_gas_arr)

    # Density
    if use_structure_density:
        rho_ours = 10.0**shell_data_zp[:, 7]
    else:
        m_frac_arr = shell_data_zp[:, 1]

        def _bicubic_rho_at_shell(i):
            mf = m_frac_arr[i]
            X_l = interp_X_at_mass(X_final, mf)
            Z_l = interp_X_at_mass(Z_profile, mf)
            rho_bc, *_ = eos_lookup_bicubic(logT_arr[i], logPgas_arr[i], X_l, Z_l)
            return rho_bc

        rho_ours = jax.vmap(lambda i: _bicubic_rho_at_shell(i))(jnp.arange(N_SHOOT))

    # Sound speed: c_s = sqrt(gamma1 * P / rho)
    beta_arr = P_gas_arr / P_total_arr
    # Eddington Γ₁ for an ideal gas + radiation mixture, β = P_gas/P_total
    # (Chandrasekhar 1939, §56; Cox & Giuli 1968, §9.17). Analytic (exact), not fitted.
    gamma1_arr = (32.0 - 24.0 * beta_arr - 3.0 * beta_arr**2) / (24.0 - 21.0 * beta_arr)
    c_s = jnp.sqrt(gamma1_arr * P_total_arr / rho_ours)

    # Restrict to interior (exclude center and near-surface)
    r_np = np.asarray(r_over_R)
    c_np = np.asarray(c_s)
    rho_np = np.asarray(rho_ours)

    valid = (r_np > 0.05) & (r_np < 0.90)
    return r_np[valid], c_np[valid], rho_np[valid]


def _compare_sound_speed(r_valid, c_valid, cs_ref):
    """Compare our sound speed against Model S reference.

    Args:
        r_valid: Radial coordinate array (already filtered to 0.05-0.90).
        c_valid: Our computed sound speed at those radii.
        cs_ref: Tuple of (r_asc, c_asc) from Model S, sorted ascending.

    Returns:
        Dict with keys: dc_over_c, max_abs_dc, rms_dc.
    """
    cs_r_asc, cs_val_asc = cs_ref
    ms_c_interp = np.interp(r_valid, cs_r_asc, cs_val_asc)
    dc_over_c = (c_valid - ms_c_interp) / ms_c_interp

    return {
        'dc_over_c': dc_over_c,
        'max_abs_dc': np.max(np.abs(dc_over_c)),
        'rms_dc': np.sqrt(np.mean(dc_over_c**2)),
    }


def _compare_density(r_valid, rho_valid, rho_ref):
    """Compare our density against Model S reference.

    Args:
        r_valid: Radial coordinate array (already filtered to 0.05-0.90).
        rho_valid: Our computed density at those radii.
        rho_ref: Tuple of (r_asc, rho_asc) from Model S, sorted ascending.

    Returns:
        Dict with keys: drho_over_rho, max_abs_drho, rms_drho.
    """
    rho_r_asc, rho_val_asc = rho_ref
    ms_rho_interp = np.interp(r_valid, rho_r_asc, rho_val_asc)
    drho_over_rho = (rho_valid - ms_rho_interp) / ms_rho_interp

    return {
        'drho_over_rho': drho_over_rho,
        'max_abs_drho': np.max(np.abs(drho_over_rho)),
        'rms_drho': np.sqrt(np.mean(drho_over_rho**2)),
    }


# ======================================================================
# Public API
# ======================================================================

def compare_model_s(M_solar=1.0, Z=None, alpha_mlt=None, max_steps=500,
                    use_structure_density=True):
    """Compare our solar model sound speed and density against Model S.

    Decomposes into:
      1. _resolve_model_s_data_dir — locate reference data
      2. _solar_model_at_age — evolve + Z-profile reconvergence at 4.57 Gyr
      3. _compute_sound_speed_and_density — physics from shell data
      4. _compare_sound_speed / _compare_density — residuals vs Model S

    Reference: Christensen-Dalsgaard et al. (1996), Science 272, 1286.

    Args:
        use_structure_density: If True (default), use density from the structure
            solve directly (shell_data logrho). This is physically self-consistent.

    Current accuracy: max|dc_s/c_s| < 1%, max|dρ/ρ| ~ 2-4%.
    """
    if Z is None:
        Z = 0.0188
    if alpha_mlt is None:
        alpha_mlt = ALPHA_SOLAR

    # Step 1: Locate Model S data
    data_dir = _resolve_model_s_data_dir()
    if data_dir is None:
        _module_dir = os.path.dirname(os.path.realpath(__file__))
        _project_dir = os.path.dirname(_module_dir)
        return {'error': (
            f'Model S data not found. Expected at data/model_s/ relative to '
            f'project root ({_project_dir}). Set STELLAR_MODEL_S_DIR to override.'
        )}

    # Step 2: Evolve + reconverge at solar age
    shell_data_zp, X_final, Z_profile, logL_f, logTe_f, r_over_R = \
        _solar_model_at_age(M_solar, Z, alpha_mlt, max_steps)

    # Step 3: Compute sound speed and density
    r_valid, c_valid, rho_valid = _compute_sound_speed_and_density(
        shell_data_zp, X_final, Z_profile, r_over_R, use_structure_density)

    # Step 4: Load Model S reference data
    cs_ref, rho_ref = _load_model_s_data(data_dir)

    # Step 5: Compare sound speed
    cs_result = _compare_sound_speed(r_valid, c_valid, cs_ref)

    out = {
        'r_over_R': r_valid,
        'c_s_ours': c_valid,
        'c_s_model_s': np.interp(r_valid, cs_ref[0], cs_ref[1]),
        'dc_over_c': cs_result['dc_over_c'],
        'max_abs_dc': cs_result['max_abs_dc'],
        'rms_dc': cs_result['rms_dc'],
        'logL': float(logL_f),
        'logTe': float(logTe_f),
        'X_c': float(X_final[0]),
    }

    # Step 6: Compare density (if reference data available)
    if rho_ref is not None:
        rho_result = _compare_density(r_valid, rho_valid, rho_ref)
        out['drho_over_rho'] = rho_result['drho_over_rho']
        out['max_abs_drho'] = rho_result['max_abs_drho']
        out['rms_drho'] = rho_result['rms_drho']

    return out


def read_mesa_history(filepath):
    """Parse MESA history.data file."""
    with open(filepath) as f:
        lines = f.readlines()
    col_names = lines[5].split()
    data_lines = lines[6:]
    data = np.array([[float(x) for x in line.split()] for line in data_lines if line.strip()])
    cols = {name: data[:, i] for i, name in enumerate(col_names)}
    return cols


def compare_mesa(masses=(1.0, 1.2, 1.5, 2.0), Z=0.014, alpha_mlt=None,
                 mesa_dir=None, max_steps=1000, xc_grid=None):
    """Compare our tracks against MESA."""
    if alpha_mlt is None:
        alpha_mlt = ALPHA_SOLAR
    if mesa_dir is None:
        _module_dir = os.path.dirname(os.path.realpath(__file__))
        _project_dir = os.path.dirname(_module_dir)
        mesa_dir = os.path.join(_project_dir, 'data', 'mesa_comparison', 'results')
    if xc_grid is None:
        xc_grid = np.linspace(0.6, 0.1, 50)

    results = {}
    for mass in masses:
        mass_str = f"{mass:.1f}Msun"
        mesa_path = os.path.join(mesa_dir, mass_str, 'history.data')
        if not os.path.exists(mesa_path):
            logger.warning(f"{mesa_path} not found, skipping {mass_str}")
            continue

        mesa = read_mesa_history(mesa_path)
        mesa_xc = mesa['center_h1']
        mesa_logL = mesa['log_L']
        mesa_logTeff = mesa['log_Teff']
        mesa_logR = mesa['log_R']

        mask = (mesa_xc <= xc_grid[0]) & (mesa_xc >= xc_grid[-1])
        if mask.sum() < 3:
            logger.warning(f"{mass_str} MESA has <3 points in X_c range, skipping")
            continue

        mesa_xc_ms = mesa_xc[mask]
        mesa_logL_ms = mesa_logL[mask]
        mesa_logTeff_ms = mesa_logTeff[mask]
        mesa_logR_ms = mesa_logR[mask]

        mesa_logL_interp = np.interp(xc_grid, mesa_xc_ms[::-1], mesa_logL_ms[::-1])
        mesa_logTeff_interp = np.interp(xc_grid, mesa_xc_ms[::-1], mesa_logTeff_ms[::-1])
        mesa_logR_interp = np.interp(xc_grid, mesa_xc_ms[::-1], mesa_logR_ms[::-1])

        logger.info(f"Running {mass_str} (α={alpha_mlt:.4f}, max_steps={max_steps})...")
        from stellar_jax.evolution import evolve_star  # lazy import to avoid circular dependency
        r = evolve_star(mass, Z=Z, max_steps=max_steps, alpha_mlt=alpha_mlt)
        our_xc = np.asarray(r['center_h1'])
        our_logL = np.asarray(r['log_L'])
        our_logTeff = np.asarray(r['log_Teff'])
        our_logR = np.asarray(r['log_R'])

        our_mask = (our_xc <= xc_grid[0]) & (our_xc >= xc_grid[-1])
        if our_mask.sum() < 3:
            logger.warning(f"{mass_str} our code has <3 points in range (X_c min={our_xc.min():.3f})")
            results[mass] = {'status': 'insufficient_evolution'}
            continue

        our_xc_ms = our_xc[our_mask]
        our_logL_ms = our_logL[our_mask]
        our_logTeff_ms = our_logTeff[our_mask]
        our_logR_ms = our_logR[our_mask]

        our_logL_interp = np.interp(xc_grid, our_xc_ms[::-1], our_logL_ms[::-1])
        our_logTeff_interp = np.interp(xc_grid, our_xc_ms[::-1], our_logTeff_ms[::-1])
        our_logR_interp = np.interp(xc_grid, our_xc_ms[::-1], our_logR_ms[::-1])

        dlogL = our_logL_interp - mesa_logL_interp
        dlogTeff = our_logTeff_interp - mesa_logTeff_interp
        dlogR = our_logR_interp - mesa_logR_interp

        results[mass] = {
            'max_dlogL': float(np.max(np.abs(dlogL))),
            'max_dlogTeff': float(np.max(np.abs(dlogTeff))),
            'max_dlogR': float(np.max(np.abs(dlogR))),
            'rms_dlogL': float(np.sqrt(np.mean(dlogL**2))),
            'rms_dlogTeff': float(np.sqrt(np.mean(dlogTeff**2))),
            'mean_dlogL': float(np.mean(dlogL)),
            'mean_dlogTeff': float(np.mean(dlogTeff)),
            'our_steps': max_steps,
            'mesa_points_in_range': int(mask.sum()),
            'status': 'ok',
            'xc_grid': xc_grid.tolist(),
            'dlogL': dlogL.tolist(),
            'dlogTeff': dlogTeff.tolist(),
            'dlogR': dlogR.tolist(),
            'our_logL': our_logL_interp.tolist(),
            'mesa_logL': mesa_logL_interp.tolist(),
            'our_logTeff': our_logTeff_interp.tolist(),
            'mesa_logTeff': mesa_logTeff_interp.tolist(),
        }
        logger.info(f"  {mass_str}: max|Δlog_L|={results[mass]['max_dlogL']:.4f}, "
                    f"max|Δlog_Teff|={results[mass]['max_dlogTeff']:.4f}, "
                    f"max|Δlog_R|={results[mass]['max_dlogR']:.4f}")

    return results
