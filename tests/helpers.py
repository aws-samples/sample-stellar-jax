"""Shared test helpers — utilities used across per-module test files.

Extracted from the original tests/validate.py preamble (issue #523).
"""
import gzip
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Physical constants (CGS)
# ═══════════════════════════════════════════════════════════════

_SIGMA_SB = 5.6704e-5   # Stefan-Boltzmann constant, erg cm^-2 s^-1 K^-4
_LSUN = 3.828e33        # Solar luminosity, erg/s
_RSUN = 6.957e10        # Solar radius, cm


# ═══════════════════════════════════════════════════════════════
# Data path resolution — handles both local and CI environments
# ═══════════════════════════════════════════════════════════════

def _resolve_data_path(rel_path):
    """Resolve a path relative to the data/ directory.

    In CI, the Docker image bakes data into /app/data/ and the git clone has
    data/ at the repo root. The ``ln -sf /app/data data`` in ci_entrypoint.sh
    may or may not replace the git-tracked directory depending on coreutils
    behavior. This helper tries multiple candidate roots to find the file.

    Args:
        rel_path: path relative to data/, e.g. "mesa_comparison/profiles/1.0Msun/midMS.FGONG.gz"

    Returns:
        Resolved absolute path (first existing candidate).

    Raises:
        FileNotFoundError if not found in any candidate location.
    """
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "src", "stellar_jax", "data", rel_path),
        os.path.join("/app", "data", rel_path),
        os.path.join(os.getcwd(), "src", "stellar_jax", "data", rel_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Data file not found: data/{rel_path}\n  Searched: {candidates}")


# ═══════════════════════════════════════════════════════════════
# FGONG runtime loader — replaces hardcoded MESA reference values
# ═══════════════════════════════════════════════════════════════

def _load_mesa_zams_fgong(mass):
    """Load MESA ZAMS FGONG profile and extract reference quantities at runtime.

    References loaded at runtime from MODE-A FGONG library (MESA r26.04.1,
    identical physics: Z=0.014, Y=0.2695, alpha_MLT=2.0 Cox, Krishna-Swamy
    T(tau), diffusion/overshoot/rotation OFF).

    Returns dict with: log_Tc, log_rhoc, log_Teff, log_L, L_solar, R_solar.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from stellar_jax.oscillations import read_fgong, fgong_components

    path = _resolve_data_path(
        os.path.join("mesa_comparison", "profiles", f"{mass}Msun", "zams.FGONG.gz"))
    with gzip.open(path) as fz:
        raw = fz.read()
    with tempfile.NamedTemporaryFile("wb", suffix=".FGONG", delete=False) as t:
        t.write(raw)
        tmp = t.name
    try:
        glob, var = read_fgong(tmp)
        comp = fgong_components(glob, var)
    finally:
        os.unlink(tmp)

    # Central conditions (index 0 = center in center-to-surface ordering)
    log_Tc = float(np.log10(comp['T'][0]))
    log_rhoc = float(np.log10(comp['rho'][0]))

    # Surface (index -1 = outermost zone)
    R = comp['r'][-1]
    L = comp['L'][-1]
    log_Teff = float(np.log10((L / (4 * np.pi * R**2 * _SIGMA_SB))**0.25))
    log_L = float(np.log10(L / _LSUN))
    L_solar = float(L / _LSUN)
    R_solar = float(R / _RSUN)

    return dict(log_Tc=log_Tc, log_rhoc=log_rhoc, log_Teff=log_Teff,
                log_L=log_L, L_solar=L_solar, R_solar=R_solar)


# ═══════════════════════════════════════════════════════════════
# ADIPLS frequency reference loader (DRY —)
# ═══════════════════════════════════════════════════════════════

def load_adipls_reference(filename='model_s_adipls_freqs_mesa2604.dat'):
    """Load an ADIPLS frequency reference file (l, n, freq_uHz columns).

    Returns a dict ``{l: {n: freq_uHz}}`` for all (l, n) pairs in the file.
    The canonical reference is ``model_s_adipls_freqs_mesa2604.dat`` — a
    genuine, reproducible ADIPLS run from MESA 26.04.1 with G=6.67430e-8
    on the byte-identical Model S FGONG (recipe:
    ``data/model_s/adipls_kernels/RECIPE.md``; control file:
    ``data/model_s/adipls_kernels/adipls_ms.in``).

    File format (mesa2604 and full.dat): ``l  n  freq_uHz`` (l-first).

    Args:
        filename: name of the .dat file under ``data/model_s/``.

    Returns:
        dict mapping ``l -> {n: freq_uHz}``.
    """
    path = _resolve_data_path(os.path.join('model_s', filename))
    freqs = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            l_deg, n_order, freq = int(parts[0]), int(parts[1]), float(parts[2])
            if l_deg not in freqs:
                freqs[l_deg] = {}
            freqs[l_deg][n_order] = freq
    return freqs


def _load_fgong_profile(mass, stage):
    """Load a committed MESA FGONG profile at a given evolutionary stage.

    Loads MODE-A (Z=0.014, Y=0.2695, alpha_MLT=2.0, diffusion OFF) profiles
    from data/mesa_comparison/profiles/<mass>/<stage>.FGONG.gz.

    Args:
        mass: e.g. "1.0Msun", "1.5Msun", "2.0Msun"
        stage: e.g. "zams", "midMS", "Xc0.60", "Xc0.40", "TAMS", "SGB"

    Returns:
        (glob, comp) where glob is the FGONG global array and comp is the
        fgong_components dict (keys: r, T, P, rho, X, L, kappa, eps_nuc,
        gamma1, m_frac, brunt_A).
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from stellar_jax.oscillations import read_fgong, fgong_components

    path = _resolve_data_path(
        os.path.join("mesa_comparison", "profiles", mass, f"{stage}.FGONG.gz"))
    with gzip.open(path) as fz:
        raw = fz.read()
    with tempfile.NamedTemporaryFile("wb", suffix=".FGONG", delete=False) as t:
        t.write(raw)
        tmp = t.name
    try:
        glob, var = read_fgong(tmp)
        comp = fgong_components(glob, var)
    finally:
        os.unlink(tmp)
    return glob, comp
