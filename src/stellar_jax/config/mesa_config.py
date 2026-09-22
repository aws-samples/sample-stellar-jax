"""Canonical MESA identical-physics comparison config (MODE A).

Single source of truth for all tests that compare stellar-jax against
data/mesa_comparison/ tracks. Parsed from data/mesa_comparison/inlist_1.0Msun.

Usage:
    from stellar_jax.config.mesa_config import MESA_CONFIG
    r = stellar.evolve_star(mass, **MESA_CONFIG, max_steps=500)

O4 reviewer rule: any test reading data/mesa_comparison/ MUST use MESA_CONFIG.
Never use these values for Model S comparison (that is MODE B; see
docs/reference/validation-regimes.md).
"""

import os

_INLIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'data', 'mesa_comparison')
_INLIST_PATH = os.path.join(_INLIST_DIR, 'inlist_1.0Msun')


def _parse_inlist(path):
    """Extract key physics parameters from a MESA inlist file."""
    params = {}
    with open(path) as f:
        for line in f:
            line = line.split('!')[0].strip()  # strip comments
            if '=' not in line:
                continue
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip().rstrip('/')
            # Parse Fortran values
            if val in ('.true.', '.True.'):
                params[key] = True
            elif val in ('.false.', '.False.'):
                params[key] = False
            else:
                try:
                    params[key] = float(val)
                except ValueError:
                    params[key] = val.strip("'\"")
    return params


_raw = _parse_inlist(_INLIST_PATH) if os.path.exists(_INLIST_PATH) else {}

# Canonical config — kwargs for evolve_star() in MODE A tests
MESA_CONFIG = {
    'Z': _raw.get('initial_z', 0.014),
    'alpha_mlt': _raw.get('mixing_length_alpha', 2.0),
    'Y_init': _raw.get('initial_y', 0.2695),
    'diffusion': _raw.get('do_element_diffusion', False),
    'f_ov': 0.0,  # overshooting off (MESA default when no overshoot_scheme set)
}

# Expose individual values for documentation/assertions
Z_MESA = MESA_CONFIG['Z']
ALPHA_MLT_MESA = MESA_CONFIG['alpha_mlt']
Y_MESA = MESA_CONFIG['Y_init']
DIFFUSION_MESA = MESA_CONFIG['diffusion']
F_OV_MESA = MESA_CONFIG['f_ov']
