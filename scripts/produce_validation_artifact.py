#!/usr/bin/env python3
"""Produce a full-resolution validation artifact for a given commit.

Generates: HRD tracks, solar sound speed/density comparisons, MESA residuals,
oscillation frequencies, gradient accuracy data, Kippenhahn diagrams, and
publication-quality figures. Packages everything into a tarball suitable for
upload as a GitHub release asset.

Usage:
    python scripts/produce_validation_artifact.py [--output-dir DIR] [--sha SHA]

Environment:
    STELLAR_MICROPHYSICS=mesa  — required for the MESA backend comparison section.
    JAX_ENABLE_X64=1           — set automatically for reproducibility.

References:
    Christensen-Dalsgaard et al. (1996), Science 272, 1286 (Model S)
    Paxton et al. (2011), ApJS 192, 3 (MESA)
"""
import argparse
import json
import os
import platform
import sys
import tarfile
import tempfile
import time

# Deterministic JAX: 64-bit floats, deterministic ops
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_serializable(obj):
    """Convert numpy arrays and JAX arrays to JSON-serializable lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'tolist'):  # JAX arrays
        return np.asarray(obj).tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def _write_json(data, path):
    """Write dict to JSON with 8 significant digits for floats."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(_to_serializable(data), f, indent=2)


def _setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 1.5,
        'font.family': 'serif',
    })
    # Use LaTeX-style labels if available
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.serif'] = ['Computer Modern Roman']
    except Exception:
        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.fontset'] = 'cm'
    return plt


# ---------------------------------------------------------------------------
# Data producers
# ---------------------------------------------------------------------------

MASSES = (1.0, 1.2, 1.5, 2.0)
Z_DEFAULT = 0.014
MAX_STEPS_TRACK = 500
MAX_STEPS_MESA = 1000


def produce_tracks(out_dir):
    """HRD tracks for 4 masses × 500 steps."""
    from evolution import evolve_star, ALPHA_SOLAR
    tracks_dir = os.path.join(out_dir, 'tracks')
    os.makedirs(tracks_dir, exist_ok=True)

    config = {
        'Z': Z_DEFAULT, 'alpha_mlt': float(ALPHA_SOLAR),
        'Y_init': None,  # default from DY_DZ
        'diffusion': True, 'f_ov': 0.016,
    }
    _write_json(config, os.path.join(tracks_dir, 'mesa_config.json'))

    for mass in MASSES:
        print(f"  tracks: {mass} Msun ...")
        r = evolve_star(float(mass), Z=Z_DEFAULT, max_steps=MAX_STEPS_TRACK,
                        alpha_mlt=ALPHA_SOLAR)
        data = {
            'mass': mass,
            'star_age': r['star_age'],
            'log_L': r['log_L'],
            'log_Teff': r['log_Teff'],
            'log_R': r['log_R'],
            'center_h1': r['center_h1'],
            'log_Tc': r['log_Tc'],
            'log_rhoc': r['log_rhoc'],
        }
        _write_json(data, os.path.join(tracks_dir, f'{mass}Msun_500steps.json'))
    return tracks_dir


def produce_solar_model(out_dir):
    """Solar sound speed + density vs Model S, FGONG profile, oscillations."""
    from evolution import compare_model_s, write_fgong, ALPHA_SOLAR, evolve_star
    from constants import Y0_SOLAR
    import oscillations
    import stellar

    solar_dir = os.path.join(out_dir, 'solar_model')
    os.makedirs(solar_dir, exist_ok=True)

    # Sound speed and density comparison
    print("  solar: compare_model_s ...")
    ms_result = compare_model_s(max_steps=MAX_STEPS_TRACK)
    cs_data = {
        'r_over_R': ms_result['r_over_R'],
        'c_s_ours': ms_result['c_s_ours'],
        'c_s_model_s': ms_result['c_s_model_s'],
        'fractional_diff': ms_result['dc_over_c'],
        'max_abs_dc': ms_result['max_abs_dc'],
        'rms_dc': ms_result['rms_dc'],
    }
    _write_json(cs_data, os.path.join(solar_dir, 'sound_speed_vs_model_s.json'))

    if 'drho_over_rho' in ms_result:
        rho_data = {
            'r_over_R': ms_result['r_over_R'],
            'rho_ours': None,  # not separately stored; difference is the key data
            'rho_model_s': None,
            'fractional_diff': ms_result['drho_over_rho'],
            'max_abs_drho': ms_result['max_abs_drho'],
            'rms_drho': ms_result['rms_drho'],
        }
        _write_json(rho_data, os.path.join(solar_dir, 'density_vs_model_s.json'))

    # FGONG profile (using single solar calibration, Z=0.0188)
    print("  solar: FGONG profile ...")
    Z_solar = 0.0188
    r = evolve_star(1.0, Z=Z_solar, max_steps=MAX_STEPS_TRACK,
                    alpha_mlt=ALPHA_SOLAR, Y_init=Y0_SOLAR,
                    t_max=4.57e9)
    fgong_path = os.path.join(solar_dir, 'profile.FGONG')
    write_fgong(fgong_path, 1.0, float(np.asarray(r['log_L_final'])),
                float(np.asarray(r['log_Teff_final'])),
                r['X_profile'], Z_solar, 4.57e9, float(ALPHA_SOLAR))

    # Oscillation frequencies
    print("  solar: oscillation frequencies ...")
    osc_result = oscillations.compute_evolved_solar_model_freqs(
        stellar, Z=Z_solar, t_max=4.57e9, max_steps=MAX_STEPS_TRACK)
    osc_data = {
        'delta_nu': osc_result['delta_nu'],
        'R_over_Rsun': osc_result['R_over_Rsun'],
        'age_gyr': osc_result['age_gyr'],
        'freqs': {str(l): list(np.asarray(v)) for l, v in osc_result['freqs'].items()},
    }
    _write_json(osc_data, os.path.join(solar_dir, 'oscillations.json'))
    return solar_dir


def produce_gradients(out_dir):
    """AD vs FD gradient accuracy at N=100 and N=500, plus dlogTeff/dalpha sensitivity."""
    import jax
    import jax.numpy as jnp
    from evolution import evolve_star

    grad_dir = os.path.join(out_dir, 'gradients')
    os.makedirs(grad_dir, exist_ok=True)

    results = {}
    for N in (100, 500):
        print(f"  gradients: N={N} ...")
        # Mass gradient: dlogL/dM
        M, dM = 1.0, 1e-7
        f_L = lambda m, n=N: evolve_star(m, Z=0.014, max_steps=n)["log_L"][-1]
        ad_L = float(jax.grad(f_L)(jnp.float64(M)))
        fd_L = (float(f_L(M + dM)) - float(f_L(M - dM))) / (2 * dM)
        rel_err_L = abs(ad_L - fd_L) / (abs(fd_L) + 1e-30)

        # Alpha gradient: dlogTeff/dalpha
        alpha, da = 1.9, 1e-4
        f_T = lambda a, n=N: evolve_star(1.0, Z=0.014, max_steps=n, alpha_mlt=a)["log_Teff"][0]
        ad_T = float(jax.grad(f_T)(jnp.float64(alpha)))
        fd_T = (float(f_T(alpha + da)) - float(f_T(alpha - da))) / (2 * da)
        rel_err_T = abs(ad_T - fd_T) / (abs(fd_T) + 1e-30)

        data = {
            'max_steps': N,
            'mass_gradient': {
                'param': 'mass', 'mass': M,
                'ad_value': ad_L, 'fd_value': fd_L, 'rel_error': rel_err_L,
            },
            'alpha_gradient': {
                'param': 'alpha_mlt', 'mass': 1.0,
                'ad_value': ad_T, 'fd_value': fd_T, 'rel_error': rel_err_T,
            },
        }
        _write_json(data, os.path.join(grad_dir, f'ad_vs_fd_N{N}.json'))
        results[N] = data

    # dlogTeff/dalpha sensitivity across masses with published bounds.
    # Published range: Magic et al. (2015, A&A 573, A89) Figure 12 gives
    # dlogTeff/dalpha ~ 0.03-0.06 for solar-type (1 Msun) and decreasing
    # for higher masses (radiative envelopes less sensitive to MLT).
    # Salaris & Cassisi (2015, A&A 577, A60) Section 4: ~0.04-0.08 at 1 Msun.
    print("  gradients: dlogTeff/dalpha sensitivity ...")
    alpha_ref = 1.9
    da = 1e-4
    published_bounds = {
        1.0: (0.02, 0.10),   # solar-type: significant MLT sensitivity
        1.2: (0.015, 0.08),  # slightly less sensitive
        1.5: (0.005, 0.05),  # partially radiative envelope
        2.0: (0.001, 0.03),  # mostly radiative, weak MLT sensitivity
    }
    sensitivity_data = []
    for mass in MASSES:
        f_T = lambda a, m=mass: evolve_star(
            m, Z=0.014, max_steps=100, alpha_mlt=a)["log_Teff"][0]
        value = float(jax.grad(f_T)(jnp.float64(alpha_ref)))
        pmin, pmax = published_bounds[mass]
        sensitivity_data.append({
            'mass': mass,
            'value': value,
            'published_min': pmin,
            'published_max': pmax,
        })
    _write_json(sensitivity_data, os.path.join(grad_dir, 'dlogTeff_dalpha.json'))

    return grad_dir


def produce_mesa_comparison(out_dir):
    """MESA backend comparison: run microphysics with STELLAR_MICROPHYSICS=mesa and diff vs JAX.

    Fails hard if MESA backend is unavailable (issue acceptance: never skip).
    Gated on #292 — this will only pass once MESA support is deployed.
    """
    mesa_dir = os.path.join(out_dir, 'mesa_backend')
    os.makedirs(mesa_dir, exist_ok=True)

    # --- Verify MESA backend is available; fail hard if not ---
    print("  mesa_backend: verifying MESA backend availability ...")
    try:
        from microphysics.mesa.bindings import _ensure_init
        _ensure_init()
    except (ImportError, ValueError, OSError, FileNotFoundError) as e:
        raise RuntimeError(
            f"MESA backend unavailable — validation artifact requires "
            f"STELLAR_MICROPHYSICS=mesa to be functional (gated on #292). "
            f"Error: {e}"
        ) from e

    from microphysics.mesa.eos_adapter import eos_lookup as mesa_eos_lookup
    from microphysics.mesa.opacity_adapter import kappa as mesa_kappa
    from microphysics.eos import eos_lookup as jax_eos_lookup
    from microphysics.opacity import kappa as jax_kappa

    # --- Compare EOS outputs at a grid of conditions ---
    print("  mesa_backend: EOS comparison ...")
    import jax.numpy as jnp

    # Test conditions spanning solar interior (logT ~ 6.2-7.2, logP ~ 15-17)
    test_conditions = [
        {'logT': 6.2, 'logP': 15.0, 'X': 0.70, 'Z': 0.02, 'label': 'envelope_base'},
        {'logT': 6.8, 'logP': 16.5, 'X': 0.50, 'Z': 0.02, 'label': 'mid_radiative'},
        {'logT': 7.1, 'logP': 17.0, 'X': 0.35, 'Z': 0.02, 'label': 'core_edge'},
        {'logT': 7.2, 'logP': 17.2, 'X': 0.10, 'Z': 0.02, 'label': 'core_center'},
    ]

    eos_results = []
    for cond in test_conditions:
        logT = jnp.float64(cond['logT'])
        logP = jnp.float64(cond['logP'])
        X = jnp.float64(cond['X'])
        Z = jnp.float64(cond['Z'])

        jax_out = jax_eos_lookup(logT, logP, X, Z)
        mesa_out = mesa_eos_lookup(logT, logP, X, Z)

        # eos_lookup returns (rho, mu, nad, S, cp, chi_rho, chi_T)
        fields = ['rho', 'mu', 'nad', 'S', 'cp', 'chi_rho', 'chi_T']
        for i, field in enumerate(fields):
            jv = float(np.asarray(jax_out[i]))
            mv = float(np.asarray(mesa_out[i]))
            rel_diff = abs(jv - mv) / (abs(mv) + 1e-30)
            eos_results.append({
                'module': 'eos', 'field': field,
                'condition': cond['label'],
                'jax_value': jv, 'mesa_value': mv, 'rel_diff': rel_diff,
            })

    # --- Compare opacity outputs ---
    print("  mesa_backend: opacity comparison ...")
    opacity_results = []
    for cond in test_conditions:
        logT = jnp.float64(cond['logT'])
        logP = jnp.float64(cond['logP'])
        X = jnp.float64(cond['X'])
        Z = jnp.float64(cond['Z'])

        # Get rho from JAX EOS for kappa call
        rho = jax_eos_lookup(logT, logP, X, Z)[0]
        logRho = jnp.log10(rho)

        jv = float(np.asarray(jax_kappa(logT, logRho, X, Z)))
        mv = float(np.asarray(mesa_kappa(logT, logRho, X, Z)))
        rel_diff = abs(jv - mv) / (abs(mv) + 1e-30)
        opacity_results.append({
            'module': 'opacity', 'field': 'log_kappa',
            'condition': cond['label'],
            'jax_value': jv, 'mesa_value': mv, 'rel_diff': rel_diff,
        })

    # --- Write results ---
    _write_json(eos_results + opacity_results,
                os.path.join(mesa_dir, 'component_comparison.json'))

    # Per-mass track comparison using MESA backend for a short evolution
    print("  mesa_backend: 1 Msun short evolution with MESA backend ...")
    os.environ['STELLAR_MICROPHYSICS'] = 'mesa'
    # Force reimport to pick up the new backend
    import importlib
    import microphysics
    importlib.reload(microphysics)
    from evolution import evolve_star
    r_mesa = evolve_star(1.0, Z=Z_DEFAULT, max_steps=50)

    os.environ['STELLAR_MICROPHYSICS'] = 'jax'
    importlib.reload(microphysics)
    r_jax = evolve_star(1.0, Z=Z_DEFAULT, max_steps=50)

    steps = int(np.sum(np.asarray(r_mesa['star_age']) > 0))
    residuals = {
        'mass': 1.0, 'max_steps': 50, 'valid_steps': steps,
        'dlog_L': (np.asarray(r_mesa['log_L'][:steps])
                   - np.asarray(r_jax['log_L'][:steps])).tolist(),
        'dlog_Teff': (np.asarray(r_mesa['log_Teff'][:steps])
                      - np.asarray(r_jax['log_Teff'][:steps])).tolist(),
        'star_age': np.asarray(r_mesa['star_age'][:steps]).tolist(),
    }
    _write_json(residuals, os.path.join(mesa_dir, 'residuals_1Msun.json'))

    return mesa_dir


def produce_kippenhahn(out_dir):
    """Kippenhahn diagrams for 1.0 and 2.0 Msun."""
    from evolution import evolve_star_diagnostic, ALPHA_SOLAR

    kipp_dir = os.path.join(out_dir, 'kippenhahn')
    os.makedirs(kipp_dir, exist_ok=True)

    for mass in (1.0, 2.0):
        print(f"  kippenhahn: {mass} Msun ...")
        diag = evolve_star_diagnostic(float(mass), Z=Z_DEFAULT,
                                      max_steps=MAX_STEPS_TRACK,
                                      alpha_mlt=float(ALPHA_SOLAR))
        data = {
            'mass': mass,
            'star_age': diag['star_age'],
            'center_h1': diag['center_h1'],
            'eps_nuc': diag['eps_nuc'],
            'is_convective': diag['is_convective'],
            'mass_coords': diag['mass_coords'],
        }
        _write_json(data, os.path.join(kipp_dir, f'{mass}Msun.json'))
    return kipp_dir


# ---------------------------------------------------------------------------
# Figure producers
# ---------------------------------------------------------------------------

def produce_figures(out_dir):
    """Generate all publication-quality figures from the serialized data."""
    plt = _setup_matplotlib()
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # HRD tracks
    _fig_hrd(out_dir, fig_dir, plt)
    # Sound speed residual
    _fig_sound_speed(out_dir, fig_dir, plt)
    # Density residual
    _fig_density(out_dir, fig_dir, plt)
    # Echelle diagram
    _fig_echelle(out_dir, fig_dir, plt)
    # Gradient accuracy
    _fig_gradients(out_dir, fig_dir, plt)
    # Kippenhahn
    _fig_kippenhahn(out_dir, fig_dir, plt)
    # MESA residuals
    _fig_mesa(out_dir, fig_dir, plt)

    return fig_dir


def _fig_hrd(out_dir, fig_dir, plt):
    """HR diagram with 4 mass tracks."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, mass in enumerate(MASSES):
        path = os.path.join(out_dir, 'tracks', f'{mass}Msun_500steps.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        teff = np.array(d['log_Teff'])
        lum = np.array(d['log_L'])
        valid = teff != 0
        ax.plot(teff[valid], lum[valid], color=colors[i],
                label=rf'{mass} M$_\odot$')
    ax.set_xlabel(r'$\log\,T_{\rm eff}$ [K]')
    ax.set_ylabel(r'$\log\,L/L_\odot$')
    ax.set_title('HR Diagram — stellar-jax')
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, 'hrd_tracks.png'))
    plt.close()


def _fig_sound_speed(out_dir, fig_dir, plt):
    """Sound speed residual vs Model S."""
    path = os.path.join(out_dir, 'solar_model', 'sound_speed_vs_model_s.json')
    if not os.path.exists(path):
        return
    with open(path) as f:
        d = json.load(f)
    fig, ax = plt.subplots(figsize=(7, 4))
    r = np.array(d['r_over_R'])
    dc = np.array(d['fractional_diff'])
    ax.plot(r, dc * 100, 'b-', linewidth=1.2)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.fill_between(r, -1, 1, alpha=0.1, color='green', label=r'$\pm1\%$ target')
    ax.set_xlabel(r'$r/R$')
    ax.set_ylabel(r'$\delta c_s / c_s$ [\%]')
    ax.set_title('Sound speed: stellar-jax vs Model S')
    ax.set_xlim(0.05, 0.90)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, 'sound_speed_residual.png'))
    plt.close()


def _fig_density(out_dir, fig_dir, plt):
    """Density residual vs Model S."""
    path = os.path.join(out_dir, 'solar_model', 'density_vs_model_s.json')
    if not os.path.exists(path):
        return
    with open(path) as f:
        d = json.load(f)
    if d.get('fractional_diff') is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    r = np.array(d['r_over_R'])
    drho = np.array(d['fractional_diff'])
    ax.plot(r, drho * 100, 'r-', linewidth=1.2)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.fill_between(r, -1, 1, alpha=0.1, color='green', label=r'$\pm1\%$ target')
    ax.set_xlabel(r'$r/R$')
    ax.set_ylabel(r'$\delta\rho / \rho$ [\%]')
    ax.set_title('Density: stellar-jax vs Model S')
    ax.set_xlim(0.05, 0.90)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, 'density_residual.png'))
    plt.close()


def _fig_echelle(out_dir, fig_dir, plt):
    """Echelle diagram from oscillation data."""
    path = os.path.join(out_dir, 'solar_model', 'oscillations.json')
    if not os.path.exists(path):
        return
    with open(path) as f:
        d = json.load(f)
    delta_nu = d['delta_nu']
    freqs = d['freqs']
    if not delta_nu or not freqs:
        return

    fig, ax = plt.subplots(figsize=(6, 8))
    colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728'}
    markers = {0: 'o', 1: 's', 2: '^', 3: 'v'}
    for l_str, nu_list in freqs.items():
        l = int(l_str)
        nu = np.array(nu_list)
        if len(nu) == 0:
            continue
        x = nu % delta_nu
        ax.scatter(x, nu, c=colors.get(l, 'gray'), marker=markers.get(l, 'o'),
                   s=30, label=rf'$\ell={l}$', alpha=0.8)
    ax.set_xlabel(rf'$\nu$ mod $\Delta\nu$ ($\mu$Hz), $\Delta\nu={delta_nu:.1f}$')
    ax.set_ylabel(r'$\nu$ ($\mu$Hz)')
    ax.set_title(r'Echelle diagram — evolved solar model')
    ax.set_xlim(0, delta_nu)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, 'echelle_solar.png'))
    plt.close()


def _fig_gradients(out_dir, fig_dir, plt):
    """Gradient accuracy: AD vs FD at N=100 and N=500."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, N in enumerate((100, 500)):
        path = os.path.join(out_dir, 'gradients', f'ad_vs_fd_N{N}.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        ax = axes[i]
        params = ['mass_gradient', 'alpha_gradient']
        labels = [r'$\partial\log L/\partial M$', r'$\partial\log T_{\rm eff}/\partial\alpha$']
        ad_vals = [d[p]['ad_value'] for p in params]
        fd_vals = [d[p]['fd_value'] for p in params]
        x = np.arange(len(params))
        w = 0.35
        ax.bar(x - w/2, ad_vals, w, label='AD (analytic)', color='#1f77b4')
        ax.bar(x + w/2, fd_vals, w, label='FD (finite diff)', color='#ff7f0e')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f'N = {N} steps')
        ax.legend()
        ax.set_ylabel('Gradient value')
        ax.grid(True, alpha=0.3, axis='y')
    plt.suptitle('Gradient accuracy: AD vs FD')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'gradient_accuracy.png'))
    plt.close()


def _fig_kippenhahn(out_dir, fig_dir, plt):
    """Kippenhahn diagram for 1.0 Msun."""
    path = os.path.join(out_dir, 'kippenhahn', '1.0Msun.json')
    if not os.path.exists(path):
        return
    with open(path) as f:
        d = json.load(f)
    from evolution import plot_kippenhahn
    # Re-use the built-in plotting function
    diag = {k: np.array(v) for k, v in d.items() if k != 'mass'}
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_kippenhahn(diag, title=r'1.0 M$_\odot$ Kippenhahn', ax=ax)
        plt.savefig(os.path.join(fig_dir, 'kippenhahn_1Msun.png'))
        plt.close()
    except Exception as e:
        print(f"  WARNING: kippenhahn figure failed: {e}")


def _fig_mesa(out_dir, fig_dir, plt):
    """MESA backend comparison: component-level residuals + track residuals."""
    mesa_dir = os.path.join(out_dir, 'mesa_backend')
    if not os.path.isdir(mesa_dir):
        return

    # Component comparison (EOS + opacity per-condition)
    comp_path = os.path.join(mesa_dir, 'component_comparison.json')
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            comps = json.load(f)
        if comps:
            fig, ax = plt.subplots(figsize=(10, 5))
            labels = [f"{c['module']}/{c['field']}\n{c['condition']}" for c in comps]
            rel_diffs = [c['rel_diff'] * 100 for c in comps]
            x = np.arange(len(labels))
            colors = ['#1f77b4' if c['module'] == 'eos' else '#d62728' for c in comps]
            ax.bar(x, rel_diffs, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel(r'Relative difference [\%]')
            ax.set_title('MESA backend vs JAX: module-level comparison')
            ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label=r'1\% threshold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'mesa_backend_residuals.png'))
            plt.close()

    # Track residuals (1 Msun evolution)
    res_path = os.path.join(mesa_dir, 'residuals_1Msun.json')
    if os.path.exists(res_path):
        with open(res_path) as f:
            d = json.load(f)
        if d.get('star_age') and d.get('dlog_L'):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            age = np.array(d['star_age']) / 1e9  # Gyr
            ax1.plot(age, d['dlog_L'], 'b-')
            ax1.set_ylabel(r'$\Delta\log L$ (MESA$-$JAX)')
            ax1.axhline(0, color='k', linewidth=0.5)
            ax1.grid(True, alpha=0.3)
            ax2.plot(age, d['dlog_Teff'], 'r-')
            ax2.set_ylabel(r'$\Delta\log T_{\rm eff}$ (MESA$-$JAX)')
            ax2.set_xlabel('Age [Gyr]')
            ax2.axhline(0, color='k', linewidth=0.5)
            ax2.grid(True, alpha=0.3)
            plt.suptitle(r'1.0 M$_\odot$ evolution: MESA backend $-$ JAX backend')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'mesa_backend_tracks.png'))
            plt.close()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def produce_metadata(out_dir, sha, wall_seconds):
    """Write metadata.json with commit SHA, versions, and timing."""
    import jax
    meta = {
        'sha': sha,
        'date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'python_version': platform.python_version(),
        'jax_version': jax.__version__,
        'numpy_version': np.__version__,
        'platform': platform.platform(),
        'wall_seconds': round(wall_seconds, 1),
    }
    # MESA version if available
    try:
        from microphysics.mesa import bindings
        meta['mesa_version'] = getattr(bindings, 'MESA_VERSION', 'unknown')
    except Exception:
        meta['mesa_version'] = 'not_available'
    _write_json(meta, os.path.join(out_dir, 'metadata.json'))
    return meta


# ---------------------------------------------------------------------------
# Package into tarball
# ---------------------------------------------------------------------------

def package_tarball(out_dir, sha):
    """Create the final tarball from the output directory."""
    sha7 = sha[:7] if sha else 'unknown'
    tarball_name = f'stellar-jax-validation-{sha7}.tar.gz'
    tarball_path = os.path.join(os.path.dirname(out_dir), tarball_name)
    with tarfile.open(tarball_path, 'w:gz') as tar:
        tar.add(out_dir, arcname=f'stellar-jax-validation-{sha7}')
    return tarball_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Produce validation artifact')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: tempdir)')
    parser.add_argument('--sha', default=None,
                        help='Git commit SHA (default: read from git)')
    args = parser.parse_args()

    # Determine SHA
    sha = args.sha
    if not sha:
        try:
            import subprocess
            sha = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], text=True).strip()
        except Exception:
            sha = 'unknown'

    print(f"=== Validation artifact for {sha[:8]} ===")
    t0 = time.time()

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = tempfile.mkdtemp(prefix='validation_')

    print(f"Output: {out_dir}")

    # Produce all data sections
    print("\n[1/6] HRD tracks ...")
    produce_tracks(out_dir)

    print("\n[2/6] Solar model (sound speed, density, FGONG, oscillations) ...")
    produce_solar_model(out_dir)

    print("\n[3/6] Gradients (AD vs FD) ...")
    produce_gradients(out_dir)

    print("\n[4/6] MESA track comparison ...")
    produce_mesa_comparison(out_dir)

    print("\n[5/6] Kippenhahn diagrams ...")
    produce_kippenhahn(out_dir)

    print("\n[6/6] Figures ...")
    produce_figures(out_dir)

    wall_seconds = time.time() - t0
    print(f"\n[meta] Writing metadata (wall time: {wall_seconds:.0f}s) ...")
    produce_metadata(out_dir, sha, wall_seconds)

    # Package
    tarball = package_tarball(out_dir, sha)
    print(f"\n=== Done: {tarball} ({os.path.getsize(tarball) / 1024 / 1024:.1f} MB) ===")
    return tarball


if __name__ == '__main__':
    main()
