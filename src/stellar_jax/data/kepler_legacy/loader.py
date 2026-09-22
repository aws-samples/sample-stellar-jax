"""Loader for Kepler LEGACY asteroseismic data.

Sources:
    Lund et al. 2017, ApJ 835, 172 (Paper I — frequencies, ratios)
    Silva Aguirre et al. 2017, ApJ 835, 173 (Paper II — M, R, age)

CDS catalogs: J/ApJ/835/172, J/ApJ/835/173
Erratum-corrected (2017 ApJ 850, 110) tables for ratios.
"""
import os
from pathlib import Path

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Target catalog
# ═══════════════════════════════════════════════════════════════

TARGETS = {
    12069424: {"name": "16 Cyg A", "category": "Simple"},
    12069449: {"name": "16 Cyg B", "category": "Simple"},
    6106415: {"name": "Perky", "category": "Simple"},
    8379927: {"name": "Arthur", "category": "Simple"},
}

# All 66 LEGACY KICs (from table1)
ALL_KICS = [
    1435467, 2837475, 3427720, 3456181, 3632418, 3656476, 3735871,
    4914923, 5184732, 5773345, 5950854, 6106415, 6116048, 6225718,
    6508366, 6603624, 6679371, 6933899, 7103006, 7106245, 7206837,
    7296438, 7510397, 7680114, 7771282, 7871531, 7940546, 7970740,
    8006161, 8150065, 8179536, 8228742, 8379927, 8394589, 8424992,
    8694723, 8760414, 8938364, 9025370, 9098294, 9139151, 9139163,
    9206432, 9353712, 9410862, 9414417, 9812850, 9955598, 9965715,
    10068307, 10079226, 10162436, 10454113, 10516096, 10644253, 10730618,
    10963065, 11081729, 11253226, 11772920, 12009504, 12069127, 12069424,
    12069449, 12258514, 12317678,
]


def _data_dir():
    """Return the path to this data directory."""
    return Path(os.path.dirname(os.path.abspath(__file__)))


def _raw_dir():
    """Return the path to the raw/ subdirectory."""
    return _data_dir() / "raw"


# ═══════════════════════════════════════════════════════════════
# Frequency loader (Lund 2017 Table 6)
# ═══════════════════════════════════════════════════════════════

def load_frequencies(kic):
    """Load individual oscillation mode frequencies for a given KIC.

    Returns:
        dict keyed by angular degree l (0, 1, 2, 3), each value is a
        structured array with fields: n, freq, e_freq_lo, e_freq_hi (all μHz).
        Only modes with a valid frequency are returned.

    Source: Lund et al. 2017, Table 6 (erratum-corrected).
    Format: fixed-width, cols 1-8=KIC, 10-11=n, 13=l, 15-24=freq, 26-32=e_lo, 34-40=e_hi
    """
    path = _raw_dir() / "lund2017_table6_frequencies.dat"

    result = {0: [], 1: [], 2: [], 3: []}

    with open(path, "r") as f:
        for line in f:
            if line[:8].strip() == str(kic):
                n = int(line[9:11])
                l = int(line[12:13])
                freq = float(line[14:24])
                e_lo = float(line[25:32])
                e_hi = float(line[33:40])
                result[l].append((n, freq, e_lo, e_hi))

    # Convert to structured numpy arrays
    dtype = np.dtype([("n", int), ("freq", float),
                      ("e_freq_lo", float), ("e_freq_hi", float)])
    for l in result:
        if result[l]:
            result[l] = np.array(result[l], dtype=dtype)
        else:
            result[l] = np.array([], dtype=dtype)

    return result


# ═══════════════════════════════════════════════════════════════
# Ratio loader (Lund 2017 Table 7)
# ═══════════════════════════════════════════════════════════════

def load_ratios(kic):
    """Load frequency difference ratios r_01, r_10, r_02 for a given KIC.

    Returns:
        dict keyed by ratio type ("r_01", "r_10", "r_02"), each value is a
        structured array with fields: n, ratio, e_ratio_lo, e_ratio_hi.

    Source: Lund et al. 2017, Table 7 (erratum-corrected).
    Format: fixed-width, cols 1-8=KIC, 10-14=type, 16-17=n, 19-26=ratio, 28-34=e_lo, 36-42=e_hi
    """
    path = _raw_dir() / "lund2017_table7_ratios.dat"

    result = {"r_01": [], "r_10": [], "r_02": []}

    with open(path, "r") as f:
        for line in f:
            if line[:8].strip() == str(kic):
                rtype = line[9:14].strip()
                # Normalize the type string
                key = rtype.replace("_", "_")  # r_01_, r_10_, r_02_
                key = key.rstrip("_")  # r_01, r_10, r_02
                n = int(line[15:17])
                ratio = float(line[18:26])
                e_lo = float(line[27:34])
                e_hi = float(line[35:42])
                if key in result:
                    result[key].append((n, ratio, e_lo, e_hi))

    dtype = np.dtype([("n", int), ("ratio", float),
                      ("e_ratio_lo", float), ("e_ratio_hi", float)])
    for key in result:
        if result[key]:
            result[key] = np.array(result[key], dtype=dtype)
        else:
            result[key] = np.array([], dtype=dtype)

    return result


# ═══════════════════════════════════════════════════════════════
# Global asteroseismic parameters (Silva Aguirre 2017 Table 3)
# ═══════════════════════════════════════════════════════════════

def load_global_params(kic):
    """Load global asteroseismic and atmospheric properties for a KIC.

    Returns:
        dict with keys: numax, e_numax_hi, e_numax_lo, Dnu, e_Dnu_hi, e_Dnu_lo,
        Teff, e_Teff, FeH, e_FeH

    Source: Silva Aguirre et al. 2017, Table 3.
    Format: fixed-width per ReadMe byte descriptions.
    """
    path = _raw_dir() / "silva2017_table3_global.dat"

    with open(path, "r") as f:
        for line in f:
            if line[:8].strip() == str(kic):
                return {
                    "numax": float(line[9:15]),
                    "e_numax_hi": float(line[16:20]),
                    "e_numax_lo": float(line[21:25]),
                    "Dnu": float(line[26:33]),
                    "e_Dnu_hi": float(line[34:39]),
                    "e_Dnu_lo": float(line[40:45]),
                    "Teff": int(line[46:50]),
                    "e_Teff": int(line[51:54]),
                    "FeH": float(line[55:60]),
                    "e_FeH": float(line[61:65]),
                }

    raise ValueError(f"KIC {kic} not found in silva2017_table3_global.dat")


# ═══════════════════════════════════════════════════════════════
# Stellar parameters from modeling (Silva Aguirre 2017 Table 4)
# ═══════════════════════════════════════════════════════════════

# The 6 pipelines in Silva Aguirre 2017
PIPELINES = ["AIMS", "AST", "ASTFIT", "BASTA", "C2kSMO", "GOE", "V&A", "YMCM"]
# Note: not all pipelines produce results for every star; 396 rows / 66 stars ≈ 6 per star


def load_stellar_params(kic):
    """Load stellar parameters from all modeling pipelines for a given KIC.

    Returns:
        dict with:
          - "pipelines": list of dicts, one per pipeline, each with keys:
            pipe, mass, e_mass_hi, e_mass_lo, radius, e_radius_hi, e_radius_lo,
            age, e_age_hi, e_age_lo, logg, e_logg_hi, e_logg_lo
          - "mass": pipeline-averaged mass (M☉)
          - "e_mass": average uncertainty in mass
          - "radius": pipeline-averaged radius (R☉)
          - "age": pipeline-averaged age (Gyr)
          - "e_age": average uncertainty in age
          - "Xini": average initial H fraction
          - "Yini": average initial He fraction

    Source: Silva Aguirre et al. 2017, Table 4.
    """
    path = _raw_dir() / "silva2017_table4_params.dat"

    pipelines = []
    with open(path, "r") as f:
        for line in f:
            if line[7:15].strip() == str(kic):
                pipe = line[0:6].strip()
                mass = float(line[16:22])
                e_mass_hi = float(line[23:29])
                e_mass_lo = float(line[30:37])
                rad = float(line[38:44])
                e_rad_hi = float(line[45:51])
                e_rad_lo = float(line[52:59])
                logg = float(line[60:66])
                age = float(line[82:89])
                e_age_hi = float(line[90:96])
                e_age_lo = float(line[97:104])

                # Initial composition (if available)
                xini_str = line[178:184].strip()
                yini_str = line[201:207].strip()
                xini = float(xini_str) if xini_str and float(xini_str) > -9 else None
                yini = float(yini_str) if yini_str and float(yini_str) > -9 else None

                pipelines.append({
                    "pipe": pipe,
                    "mass": mass,
                    "e_mass_hi": e_mass_hi,
                    "e_mass_lo": abs(e_mass_lo),
                    "radius": rad,
                    "e_radius_hi": e_rad_hi,
                    "e_radius_lo": abs(e_rad_lo),
                    "age": age,
                    "e_age_hi": e_age_hi,
                    "e_age_lo": abs(e_age_lo),
                    "logg": logg,
                    "Xini": xini,
                    "Yini": yini,
                })

    if not pipelines:
        raise ValueError(f"KIC {kic} not found in silva2017_table4_params.dat")

    # Pipeline-averaged values (simple mean)
    masses = [p["mass"] for p in pipelines]
    radii = [p["radius"] for p in pipelines]
    ages = [p["age"] for p in pipelines]
    e_masses = [(p["e_mass_hi"] + p["e_mass_lo"]) / 2 for p in pipelines]
    e_ages = [(p["e_age_hi"] + p["e_age_lo"]) / 2 for p in pipelines]
    xinis = [p["Xini"] for p in pipelines if p["Xini"] is not None]
    yinis = [p["Yini"] for p in pipelines if p["Yini"] is not None]

    return {
        "pipelines": pipelines,
        "mass": float(np.mean(masses)),
        "e_mass": float(np.mean(e_masses)),
        "radius": float(np.mean(radii)),
        "age": float(np.mean(ages)),
        "e_age": float(np.mean(e_ages)),
        "Xini": float(np.mean(xinis)) if xinis else None,
        "Yini": float(np.mean(yinis)) if yinis else None,
    }


# ═══════════════════════════════════════════════════════════════
# Convenience: compute observed Δν and δν₀₂ directly from frequencies
# ═══════════════════════════════════════════════════════════════

def compute_observed_delta_nu(freqs):
    """Compute the large frequency separation from l=0 modes.

    Args:
        freqs: dict from load_frequencies()

    Returns:
        (delta_nu, e_delta_nu) in μHz — median of consecutive l=0 spacings.
    """
    f0 = freqs[0]
    if len(f0) < 3:
        return None, None
    diffs = np.diff(f0["freq"])
    return float(np.median(diffs)), float(np.std(diffs) / np.sqrt(len(diffs)))


def compute_observed_delta_nu_02(freqs):
    """Compute the small frequency separation δν₀₂ from l=0 and l=2 modes.

    δν₀₂(n) = ν(n,0) - ν(n-1,2)

    Args:
        freqs: dict from load_frequencies()

    Returns:
        structured array with fields: n, delta_nu_02, e_lo, e_hi (μHz)
    """
    f0 = freqs[0]
    f2 = freqs[2]
    if len(f0) == 0 or len(f2) == 0:
        return np.array([], dtype=[("n", int), ("delta_nu_02", float),
                                   ("e_lo", float), ("e_hi", float)])

    # Match by n: δν₀₂(n) = ν(n,l=0) - ν(n-1,l=2)
    results = []
    n0_set = {int(row["n"]): i for i, row in enumerate(f0)}
    n2_set = {int(row["n"]): i for i, row in enumerate(f2)}

    for n in sorted(n0_set.keys()):
        if (n - 1) in n2_set:
            i0 = n0_set[n]
            i2 = n2_set[n - 1]
            dnu02 = f0[i0]["freq"] - f2[i2]["freq"]
            # Propagate errors in quadrature
            e_lo = np.sqrt(f0[i0]["e_freq_lo"]**2 + f2[i2]["e_freq_hi"]**2)
            e_hi = np.sqrt(f0[i0]["e_freq_hi"]**2 + f2[i2]["e_freq_lo"]**2)
            results.append((n, dnu02, e_lo, e_hi))

    dtype = np.dtype([("n", int), ("delta_nu_02", float),
                      ("e_lo", float), ("e_hi", float)])
    return np.array(results, dtype=dtype) if results else np.array([], dtype=dtype)
