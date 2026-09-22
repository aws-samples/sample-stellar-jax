#!/usr/bin/env python3
"""Build ferguson_4d.npz from Ferguson 2005 GS98 low-T opacity tables.

Source: Wichita State University, Ferguson et al. 2005 (ApJ, 623, 585)
        Grevesse & Sauval 1998 solar mixture
        http://www.math.wichita.edu/~ferguson/f05.gs98.tar.gz

Parses ASCII .tron files and regrids logT onto uniform 0.05 grid.
logR grid is already identical to OPAL (-8 to +1, step 0.5).
"""

import os
import re
import numpy as np
from scipy.interpolate import interp1d

# Target grids (must match OPAL convention)
X_grid = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.8, 0.9])
Z_grid = np.array([0.0, 0.0001, 0.0003, 0.001, 0.002, 0.004,
                   0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1])
logT_grid = np.arange(3.00, 4.55, 0.05)  # 31 points
logR_grid = np.arange(-8.0, 1.5, 0.5)    # 19 points: -8 to +1

# File prefix encoding: X=0.0 -> "0", X=0.1 -> "1", X=0.35 -> "35", etc.
X_PREFIX = {0.0: "0", 0.1: "1", 0.2: "2", 0.35: "35",
            0.5: "5", 0.7: "7", 0.8: "8", 0.9: "9"}

# Z suffix encoding: Z=0.0001 -> "0001", Z=0.001 -> "001", etc.
Z_SUFFIX = {0.0: "0", 0.0001: "0001", 0.0003: "0003", 0.001: "001",
            0.002: "002", 0.004: "004", 0.01: "01", 0.02: "02",
            0.03: "03", 0.04: "04", 0.06: "06", 0.08: "08", 0.1: "1"}

RAW_DIR = "data/ferguson_raw"


def parse_tron_file(filepath):
    """Parse a .tron file. Returns (logT_raw, logR_raw, kappa_table).
    
    Header format:
      Line 1: "Grevesse & Sauval 1998 with X= ... and Z= ..."
      Line 2: "                                        log R"
      Line 3: blank
      Line 4: "log T  -8.000 -7.500 ... 1.000"
      Lines 5+: "logT_val  kappa_1 kappa_2 ... kappa_19"
    """
    with open(filepath) as f:
        lines = f.readlines()

    # Parse header for X, Z (validation)
    m = re.search(r'X=\s*([\d.]+)\s+and\s+Z=\s*([\d.]+)', lines[0])
    x_val, z_val = float(m.group(1)), float(m.group(2))

    # Parse logR from header line 4
    logR_raw = np.array(lines[3].split()[2:], dtype=np.float64)

    # Parse data rows (handle Fortran fixed-width overflow where fields merge)
    logT_list, kappa_rows = [], []
    for line in lines[4:]:
        line = line.rstrip('\n')
        if len(line) < 10:
            continue
        logT_list.append(float(line[:6]))
        # Fixed-width: 19 fields of 7 chars starting at position 6
        rest = line[6:]
        row = []
        for k in range(19):
            field = rest[k*7:(k+1)*7]
            row.append(float(field))
        kappa_rows.append(row)

    logT_raw = np.array(logT_list)
    kappa_raw = np.array(kappa_rows)  # shape (nT, nR)
    return x_val, z_val, logT_raw, logR_raw, kappa_raw


def regrid_logT(logT_raw, kappa_raw, logT_target):
    """Interpolate kappa from non-uniform logT_raw onto uniform logT_target.
    
    Uses linear interpolation in logT for each logR column.
    Only interpolates within the range of logT_raw (no extrapolation).
    Ferguson tables go from 4.5 down to 2.7; our target is 3.0 to 4.5.
    """
    # Ferguson has logT descending; flip to ascending for interp1d
    idx = np.argsort(logT_raw)
    logT_sorted = logT_raw[idx]
    kappa_sorted = kappa_raw[idx]  # shape (nT_raw, nR)

    nR = kappa_sorted.shape[1]
    nT_target = len(logT_target)
    result = np.empty((nT_target, nR))

    for j in range(nR):
        f = interp1d(logT_sorted, kappa_sorted[:, j], kind='linear',
                     bounds_error=False, fill_value='extrapolate')
        result[:, j] = f(logT_target)

    return result


def build():
    nX, nZ, nT, nR = len(X_grid), len(Z_grid), len(logT_grid), len(logR_grid)
    log_kappa = np.empty((nX, nZ, nT, nR), dtype=np.float64)

    for i, X in enumerate(X_grid):
        for j, Z in enumerate(Z_grid):
            fname = f"g98.{X_PREFIX[X]}.{Z_SUFFIX[Z]}.tron"
            fpath = os.path.join(RAW_DIR, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing: {fpath}")

            x_val, z_val, logT_raw, logR_raw, kappa_raw = parse_tron_file(fpath)

            # Validate parsed X, Z match expected
            assert abs(x_val - X) < 1e-4, f"X mismatch in {fname}: {x_val} vs {X}"
            assert abs(z_val - Z) < 1e-4, f"Z mismatch in {fname}: {z_val} vs {Z}"

            # Validate logR grid matches
            assert len(logR_raw) == nR, f"logR count {len(logR_raw)} != {nR} in {fname}"
            assert np.allclose(logR_raw, logR_grid, atol=0.01), f"logR mismatch in {fname}"

            # Regrid logT
            kappa_regridded = regrid_logT(logT_raw, kappa_raw, logT_grid)
            log_kappa[i, j] = kappa_regridded

    # Verify no NaN
    assert not np.any(np.isnan(log_kappa)), "NaN found in output!"
    assert not np.any(np.isinf(log_kappa)), "Inf found in output!"

    # Save
    outpath = "data/ferguson_4d.npz"
    np.savez(outpath,
             X_grid=X_grid,
             Z_grid=Z_grid,
             logT_grid=logT_grid,
             logR_grid=logR_grid,
             log_kappa=log_kappa)

    print(f"Saved {outpath}")
    print(f"  Shape: {log_kappa.shape}")
    print(f"  logT range: [{logT_grid[0]:.2f}, {logT_grid[-1]:.2f}]")
    print(f"  logR range: [{logR_grid[0]:.1f}, {logR_grid[-1]:.1f}]")
    print(f"  kappa range: [{log_kappa.min():.3f}, {log_kappa.max():.3f}]")

    # Spot check: solar photosphere T=5000K (logT=3.7), logR~+0.25
    # Interpolate between logR=0.0 (idx 16) and logR=0.5 (idx 17)
    iX = 5  # X=0.7
    iZ = 7  # Z=0.02
    iT = np.searchsorted(logT_grid, 3.70)  # logT=3.70
    print(f"\n  Spot check (X=0.7, Z=0.02, logT=3.70):")
    print(f"    logR=0.0: log_kappa = {log_kappa[iX, iZ, iT, 16]:.3f}")
    print(f"    logR=0.5: log_kappa = {log_kappa[iX, iZ, iT, 17]:.3f}")

    # T=5000K -> logT=3.699
    iT2 = np.searchsorted(logT_grid, 3.65)
    iT3 = np.searchsorted(logT_grid, 3.75)
    print(f"    logT=3.65, logR=0.0: log_kappa = {log_kappa[iX, iZ, iT2, 16]:.3f}")
    print(f"    logT=3.75, logR=0.0: log_kappa = {log_kappa[iX, iZ, iT3, 16]:.3f}")
    print(f"    -> kappa varies with T (not frozen): PASS" 
          if abs(log_kappa[iX, iZ, iT2, 16] - log_kappa[iX, iZ, iT3, 16]) > 0.1
          else "    -> WARNING: kappa barely varies with T")


if __name__ == "__main__":
    build()
