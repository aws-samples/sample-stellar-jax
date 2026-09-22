"""Metallicity conversions: Z ↔ [Fe/H].

Standard astronomical conversions between metal mass fraction Z and the
logarithmic iron abundance [Fe/H]. These are reusable utilities, independent
of the DSEE emulator.

References:
    Asplund et al. 2009, ARAA 47, 481 (solar reference values).
"""

import numpy as np

from stellar_jax.config.constants import Y_BBN, DY_DZ

# Present-day photospheric Z (Asplund et al. 2009, ARAA 47, 481; our calibration).
Z_SOLAR = 0.0188
# Y_solar from enrichment law at Z_solar.
_Y_SOLAR_ENRICH = Y_BBN + DY_DZ * Z_SOLAR  # 0.2767
X_SOLAR = 1.0 - _Y_SOLAR_ENRICH - Z_SOLAR  # X at solar Z via enrichment law
# (Z/X)_solar reference for [Fe/H] conversion
ZX_SOLAR = Z_SOLAR / X_SOLAR


def z_to_feh(Z, Y=None):
    """Convert metal mass fraction Z to [Fe/H].

    Standard astronomical convention:
        [Fe/H] = log10(Z/X) - log10(Z_sun/X_sun)

    where X = 1 - Y - Z. If Y is not given, uses the helium enrichment law:
        Y = Y_BBN + (ΔY/ΔZ) * Z

    Parameters
    ----------
    Z : float
        Metal mass fraction.
    Y : float, optional
        Helium mass fraction. If None, computed from enrichment law.

    Returns
    -------
    float
        [Fe/H] in dex.

    References
    ----------
    Asplund et al. 2009, ARAA 47, 481 (solar reference values).
    """
    if Y is None:
        Y = Y_BBN + DY_DZ * Z
    X = 1.0 - Y - Z
    if X <= 0 or Z <= 0:
        raise ValueError(f"Unphysical composition: Z={Z}, Y={Y}, X={X}")
    return np.log10(Z / X) - np.log10(ZX_SOLAR)


def feh_to_z(feh, Y=None):
    """Convert [Fe/H] to metal mass fraction Z.

    Inverts the standard relation:
        [Fe/H] = log10(Z/X) - log10(Z_sun/X_sun)

    with X = 1 - Y - Z and Y from the helium enrichment law (if not given).

    Parameters
    ----------
    feh : float
        [Fe/H] in dex.
    Y : float, optional
        If given, use this fixed Y. Otherwise solve self-consistently with
        Y = Y_BBN + DY_DZ * Z.

    Returns
    -------
    float
        Metal mass fraction Z.
    """
    # Z/X = (Z/X)_sun * 10^[Fe/H]
    zx_target = ZX_SOLAR * 10.0 ** feh

    if Y is not None:
        # Fixed Y: Z = zx_target * X = zx_target * (1 - Y - Z)
        # Z * (1 + zx_target) = zx_target * (1 - Y)
        Z = zx_target * (1.0 - Y) / (1.0 + zx_target)
    else:
        # Self-consistent: Y = Y_BBN + DY_DZ * Z, X = 1 - Y - Z
        # Z = zx_target * (1 - Y_BBN - DY_DZ*Z - Z)
        # Z * (1 + zx_target + DY_DZ * zx_target) = zx_target * (1 - Y_BBN)
        Z = zx_target * (1.0 - Y_BBN) / (1.0 + zx_target * (1.0 + DY_DZ))

    if Z <= 0 or Z >= 1:
        raise ValueError(f"Unphysical Z={Z} from [Fe/H]={feh}")
    return Z
