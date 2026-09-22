"""DSEE emulator interface — maps DSEE↔stellar-jax parameter spaces.

This module provides a thin adapter between the CONF1DENCE/DSEE stellar evolution
emulator (arXiv:2604.06348, Ying et al. 2025) and stellar-jax's parameter space.
It runs OUTSIDE the JAX graph (PyTorch forward-only) and is used ONLY as an optional
warm-start for parameter inference — never in the gradient path.

DSEE (Dartmouth Stellar Evolution Emulator) is a conditional normalizing flow trained
on >8 million DSEP stellar evolution tracks. Given (M, Z, α_MLT, overshooting, age, ...),
it predicts (T_eff, L, R, log g, Y_core) in ~0.1 ms per star on CPU.

Repository: https://github.com/200k33p3r/CONF1DENCE (MIT license)
Weights: included in repo at data/SEM/DSEE.model

NON-DIFFERENTIABLE: This module uses PyTorch and NumPy only. It is explicitly
outside the JAX computational graph. Outputs are used only for warm-start
initialization, never in the gradient path.
"""

import numpy as np

from stellar_jax.config.constants import Y_BBN, DY_DZ
from stellar_jax.config.mesh_defaults import ALPHA_MLT, F_OV
from stellar_jax.inference.metallicity import z_to_feh


# ─── DSEE format specification ────────────────────────────────────────────────
# Extracted from CONF1DENCE/data/format/DSEE_iso_format.json
# (Ying et al. 2025, arXiv:2604.06348). Embedded inline to avoid requiring
# CONF1DENCE to be installed just for the parameter mapping.

# The 24 label dimensions and their normalization:
DSEE_LABEL_HEADER = [
    "FeH", "He Abundance", "AlphaFe", "Mixing Length",
    "He Diffusion", "Heavy Element Diffusion", "Surface Boundary Condition",
    "Envelope Overshooting", "Core Overshooting",
    "PP", "He3+He3", "He3+He4", "P+C12", "P+C13", "P+N14", "P+O16",
    "C12+He4", "Low Temp Opacity", "High Temp Opacity",
    "Triple-Alpha", "Plasma Neutrino Loss", "Conductive Opacity",
    "Age", "mass",
]

DSEE_LABEL_SCALER = [
    "None", "None", "None", "linear_rescale",
    "linear_rescale", "linear_rescale", "None",
    "linear_rescale", "linear_rescale",
    "normal", "normal", "normal", "normal", "normal", "normal", "normal",
    "normal", "linear_rescale", "normal",
    "normal", "normal", "normal",
    "mantissa_exponent", "linear_rescale",
]

DSEE_LABEL_SCALE = [
    None, None, None, [1.75, 2.5],
    [0.5, 1.3], [0.5, 1.3], None,
    [0.0, 0.2], [0.0, 0.2],
    [0.997543, 0.0098], [1.02913, 0.0971], [1.03704, 0.0556],
    [0.965517, 0.3448], [1.47273, 0.2182], [0.478916, 0.0331],
    [1.12766, 0.0851], [1.0, 0.15], [0.7, 1.3], [1.0, 0.03],
    [1.0, 0.15], [1.0, 0.05], [1.0, 0.20],
    [3.0, 10.0], [0.1, 4.0],
]

# Default label values (physical units, before normalization)
DSEE_LABEL_DEFAULT = [
    0.0,      # [Fe/H]
    0.275,    # He Abundance (Y)
    0.0,      # [α/Fe]
    1.9258,   # Mixing Length (α_MLT)
    1.0,      # He Diffusion factor
    1.0,      # Heavy Element Diffusion factor
    5.0,      # Surface BC (5=PHOENIX, 1=Krishna-Swamy, 0=Eddington)
    0.01,     # Envelope Overshooting
    0.01,     # Core Overshooting
    0.997543, # PP rate factor (mean of normal distribution)
    1.02913,  # He3+He3 rate factor
    1.03704,  # He3+He4 rate factor
    0.965517, # P+C12 rate factor
    1.47273,  # P+C13 rate factor
    0.478916, # P+N14 rate factor
    1.12766,  # P+O16 rate factor
    1.0,      # C12+He4 (triple-alpha product) rate factor
    1.0,      # Low Temp Opacity factor
    1.0,      # High Temp Opacity factor
    1.0,      # Triple-Alpha rate factor
    1.0,      # Plasma Neutrino Loss factor
    0.0,      # Conductive Opacity factor
    4.603e9,  # Age (years)
    1.0,      # Mass (M_sun)
]

# Output header for gen_theory_models
DSEE_DATA_HEADER = ["Log_T", "Log_G", "Log_L", "Log_R", "Y_Core"]

DSEE_DATA_SCALER = [
    "linear_rescale", "linear_rescale", "linear_rescale",
    "linear_rescale", "log_rescale",
]

DSEE_DATA_SCALE = [
    [3.0, 5.0], [-1.0, 5.5], [-3.0, 5.0], [-2.0, 4.0], [-2.0, 0.0],
]


# ─── Normalization utilities ─────────────────────────────────────────────────
# These replicate CONF1DENCE's data_transform.norm_data exactly so we can
# construct properly normalized input tensors without importing CONF1DENCE.

def _norm_linear_rescale(val, scale):
    """linear_rescale: (val - lo) / (hi - lo)"""
    return (val - scale[0]) / (scale[1] - scale[0])


def _norm_normal(val, scale):
    """normal: (val - mean) / std"""
    return (val - scale[0]) / scale[1]


def _norm_mantissa_exponent(val, scale):
    """mantissa_exponent: split into scaled mantissa + scaled exponent.

    Returns TWO values (the normalized column expands from 1 to 2 dimensions).
    Scale = [exp_min, exp_max].
    """
    exp_min, exp_max = scale[0], scale[1]
    exponent = np.floor(np.log10(np.clip(val, 1e-10, None)))
    mantissa = val / (10.0 ** exponent)
    mantissa_scaled = ((mantissa - 1.0) / 9.0) * 10.0 - 5.0
    exponent_scaled = ((exponent - exp_min) / (exp_max - exp_min)) * 10.0 - 5.0
    return mantissa_scaled, exponent_scaled


def _normalize_labels(labels_physical):
    """Normalize a 1D array of 24 physical-space labels into DSEE input space.

    Parameters
    ----------
    labels_physical : array-like, shape (24,)
        Physical-unit label values in DSEE_LABEL_HEADER order.

    Returns
    -------
    np.ndarray, shape (25,)
        Normalized labels (25 because Age uses mantissa_exponent → 2 dims).
    """
    labels_physical = np.asarray(labels_physical, dtype=np.float64)
    normed = []
    for i, (scaler, scale) in enumerate(zip(DSEE_LABEL_SCALER, DSEE_LABEL_SCALE)):
        val = labels_physical[i]
        if scaler == "None":
            normed.append(val)
        elif scaler == "linear_rescale":
            normed.append(_norm_linear_rescale(val, scale))
        elif scaler == "normal":
            normed.append(_norm_normal(val, scale))
        elif scaler == "mantissa_exponent":
            m, e = _norm_mantissa_exponent(val, scale)
            normed.append(m)
            normed.append(e)
        else:
            raise ValueError(f"Unknown scaler: {scaler}")
    return np.array(normed, dtype=np.float32)


def _denormalize_outputs(normed_data):
    """Inverse-normalize DSEE output (5 dims) back to physical space.

    Parameters
    ----------
    normed_data : array-like, shape (5,) or (N, 5)
        Normalized DSEE output.

    Returns
    -------
    np.ndarray, same shape
        Physical-space outputs: [Log_T, Log_G, Log_L, Log_R, Y_Core].
    """
    normed_data = np.asarray(normed_data, dtype=np.float64)
    single = normed_data.ndim == 1
    if single:
        normed_data = normed_data[None, :]

    result = np.empty_like(normed_data)
    idx = 0
    for i, (scaler, scale) in enumerate(zip(DSEE_DATA_SCALER, DSEE_DATA_SCALE)):
        col = normed_data[:, idx]
        if scaler == "linear_rescale":
            # inverse: val = normed * (hi - lo) + lo
            result[:, i] = col * (scale[1] - scale[0]) + scale[0]
        elif scaler == "log_rescale":
            # inverse: val = 10^(normed * (hi - lo) + lo)
            result[:, i] = 10.0 ** (col * (scale[1] - scale[0]) + scale[0])
        else:
            raise ValueError(f"Unknown data scaler: {scaler}")
        idx += 1

    if single:
        return result[0]
    return result


# ─── Parameter mapping: stellar-jax ↔ DSEE ──────────────────────────────────

def stellar_jax_to_dsee_labels(
    mass,
    Z=0.014,
    alpha_mlt=ALPHA_MLT,
    f_ov=None,
    Y_init=None,
    age_yr=4.603e9,
    diffusion=True,
    opacity_factor=1.0,
):
    """Map stellar-jax parameters to DSEE's 24-dim physical label space.

    This builds a full DSEE label vector from the stellar-jax parameters we
    control, filling uncontrolled dimensions with DSEE defaults.

    Parameters
    ----------
    mass : float
        Stellar mass in solar masses.
    Z : float
        Metal mass fraction (default 0.014).
    alpha_mlt : float
        Mixing-length parameter (default 1.9).
    f_ov : float or None
        Core overshooting parameter. If None, uses stellar-jax default F_OV=0.016.
    Y_init : float or None
        Initial helium abundance. If None, computed from enrichment law.
    age_yr : float
        Stellar age in years (default 4.603e9, solar age).
    diffusion : bool
        Whether diffusion is enabled. Maps to DSEE's He/heavy diffusion factors
        (1.0 if True, 0.5 — minimum — if False, effectively suppressing it).
    opacity_factor : float
        Multiplicative opacity scaling (default 1.0). Maps to DSEE's
        "High Temp Opacity" (our main regime is OPAL high-T).

    Returns
    -------
    np.ndarray, shape (24,)
        Physical-space DSEE labels (before normalization).

    Mapping notes
    -------------
    - [Fe/H] ← Z via z_to_feh() (standard astronomical conversion).
    - Mixing Length ← alpha_mlt (direct, same definition).
    - Core Overshooting ← f_ov (DSEE range 0–0.2; our default 0.016).
    - Envelope Overshooting ← 0.01 (DSEE default; we don't vary envelope ov).
    - He Diffusion / Heavy Element Diffusion ← 1.0 if diffusion else 0.5.
    - High Temp Opacity ← opacity_factor.
    - Nuclear rates, neutrino loss, conductive opacity ← DSEE defaults (1.0).
    - Surface BC ← 1.0 (Krishna-Swamy, matching our atmosphere_bc implementation).
    - Age ← age_yr.
    - Mass ← mass.
    """
    if f_ov is None:
        f_ov = F_OV  # 0.016

    if Y_init is None:
        Y_init = Y_BBN + DY_DZ * Z

    feh = z_to_feh(Z, Y=Y_init)

    # Diffusion factor: DSEE uses a continuous scale [0.5, 1.3].
    # Our boolean maps to: True→1.0 (standard), False→0.5 (minimum, effectively off)
    diff_factor = 1.0 if diffusion else 0.5

    labels = np.array(DSEE_LABEL_DEFAULT, dtype=np.float64)

    # Overwrite the dimensions we control:
    labels[0] = feh                 # [Fe/H]
    labels[1] = Y_init              # He Abundance
    labels[2] = 0.0                 # [α/Fe] (solar-scaled, no enhancement)
    labels[3] = alpha_mlt           # Mixing Length
    labels[4] = diff_factor         # He Diffusion
    labels[5] = diff_factor         # Heavy Element Diffusion
    labels[6] = 1.0                 # Surface BC = Krishna-Swamy (matches our atm)
    labels[7] = 0.01                # Envelope Overshooting (DSEE default)
    labels[8] = f_ov                # Core Overshooting
    # Nuclear rates [9:17]: keep DSEE defaults (mean of their training distribution)
    labels[17] = opacity_factor     # Low Temp Opacity factor (scale with same factor)
    labels[18] = opacity_factor     # High Temp Opacity factor
    # Triple-Alpha [19], Neutrino [20], Conductive [21]: keep defaults
    labels[22] = age_yr             # Age in years
    labels[23] = mass               # Mass in solar masses

    return labels


def dsee_outputs_to_observables(dsee_output):
    """Convert raw DSEE output to physical observables.

    Parameters
    ----------
    dsee_output : array-like, shape (5,) or (N, 5)
        Denormalized DSEE output: [Log_T, Log_G, Log_L, Log_R, Y_Core].

    Returns
    -------
    dict
        'log_Teff': log10(T_eff/K)
        'log_g': log10(g / cm s^-2)
        'log_L': log10(L/L_sun)
        'log_R': log10(R/R_sun)
        'Y_core': core helium mass fraction
    """
    dsee_output = np.asarray(dsee_output, dtype=np.float64)
    if dsee_output.ndim == 1:
        return {
            'log_Teff': float(dsee_output[0]),
            'log_g': float(dsee_output[1]),
            'log_L': float(dsee_output[2]),
            'log_R': float(dsee_output[3]),
            'Y_core': float(dsee_output[4]),
        }
    return {
        'log_Teff': dsee_output[:, 0],
        'log_g': dsee_output[:, 1],
        'log_L': dsee_output[:, 2],
        'log_R': dsee_output[:, 3],
        'Y_core': dsee_output[:, 4],
    }


# ─── Main interface class ────────────────────────────────────────────────────

class DSEEInterface:
    """Thin adapter to load the DSEE model and query it for warm-start guesses.

    This class wraps the CONF1DENCE emulator, handling:
    1. Loading the pretrained PyTorch model (forward-only, CPU).
    2. Converting stellar-jax parameters → DSEE normalized input.
    3. Running the emulator and converting output → physical observables.
    4. Packaging the result as a warm-start guess for stellar-jax's
       infer_parameters (inverse.py).

    The emulator runs OUTSIDE the JAX computational graph. It is used ONLY
    to provide initial guesses — never in the gradient path.

    Parameters
    ----------
    model_path : str or None
        Path to the DSEE.model file. If None, attempts to find it in the
        CONF1DENCE package's default location.
    device : str
        PyTorch device ('cpu' recommended — GPU not needed for warm-start).

    Raises
    ------
    ImportError
        If torch or CONF1DENCE is not installed.
    FileNotFoundError
        If the model weights cannot be found.
    """

    def __init__(self, model_path=None, device='cpu'):
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for the DSEE interface. "
                "Install with: pip install torch --index-url "
                "https://download.pytorch.org/whl/cpu"
            )

        self._torch = torch
        self._device = torch.device(device)

        # Try to load the model
        if model_path is None:
            model_path = self._find_model_path()

        if not model_path:
            raise FileNotFoundError(
                "DSEE model weights not found. Either:\n"
                "  1. Install CONF1DENCE: pip install -e /path/to/CONF1DENCE\n"
                "  2. Pass model_path='/path/to/DSEE.model' explicitly\n"
                "  3. Clone: git clone https://github.com/200k33p3r/CONF1DENCE"
            )

        self._load_model(model_path)

    def _find_model_path(self):
        """Attempt to locate the DSEE model file."""
        import os

        # Try CONF1DENCE package location
        try:
            import CONF1DENCE
            default_path = os.path.join(
                CONF1DENCE.__abspath__, 'data', 'SEM', 'DSEE.model'
            )
            if os.path.isfile(default_path):
                return default_path
        except ImportError:
            pass

        # Try common locations
        candidates = [
            os.path.expanduser('~/CONF1DENCE/data/SEM/DSEE.model'),
            '/tmp/CONF1DENCE/data/SEM/DSEE.model',
            'CONF1DENCE/data/SEM/DSEE.model',
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path

        return None

    def _load_model(self, model_path):
        """Load the pretrained DSEE normalizing flow model."""
        import os
        torch = self._torch

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            import zuko
        except ImportError:
            raise ImportError(
                "zuko is required for the DSEE normalizing flow. "
                "Install with: pip install zuko"
            )

        # DSEE format: 5 output features, 25 context dims (24 labels, but Age
        # uses mantissa_exponent → 2 dims, so 25 total context dimensions)
        n_features = 5  # Log_T, Log_G, Log_L, Log_R, Y_Core
        n_context = 25  # 24 labels with Age expanded to 2

        self._flow = zuko.flows.NSF(
            features=n_features,
            context=n_context,
            transforms=10,
            hidden_features=[256] * 10,
        ).to(self._device)

        # Load weights
        checkpoint = torch.load(
            model_path, map_location=self._device, weights_only=False
        )
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self._flow.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Old format: saved as model directly
            self._flow = checkpoint.to(self._device)

        self._flow.eval()
        self._model_path = model_path

    def predict(self, mass, Z=0.014, alpha_mlt=ALPHA_MLT, f_ov=None,
                Y_init=None, age_yr=4.603e9, diffusion=True,
                opacity_factor=1.0):
        """Query DSEE for predicted observables at the given parameters.

        Parameters
        ----------
        mass : float or array-like
            Stellar mass(es) in solar masses.
        Z : float
            Metal mass fraction.
        alpha_mlt : float
            Mixing-length parameter.
        f_ov : float or None
            Core overshooting (default: stellar-jax's F_OV=0.016).
        Y_init : float or None
            Initial helium (default: enrichment law).
        age_yr : float
            Age in years.
        diffusion : bool
            Diffusion enabled.
        opacity_factor : float
            Opacity scaling factor.

        Returns
        -------
        dict
            Physical observables: log_Teff, log_g, log_L, log_R, Y_core.
        """
        torch = self._torch

        # Build DSEE label vector
        labels_phys = stellar_jax_to_dsee_labels(
            mass=mass, Z=Z, alpha_mlt=alpha_mlt, f_ov=f_ov,
            Y_init=Y_init, age_yr=age_yr, diffusion=diffusion,
            opacity_factor=opacity_factor,
        )

        # Normalize
        labels_norm = _normalize_labels(labels_phys)

        # Convert to torch tensor
        labels_tensor = torch.tensor(
            labels_norm[None, :], dtype=torch.float32, device=self._device
        )

        # Run the emulator (forward only, no grad)
        with torch.no_grad():
            output_norm = self._flow(labels_tensor).sample()

        # Convert back to numpy and denormalize
        output_np = output_norm.cpu().numpy().squeeze()
        output_phys = _denormalize_outputs(output_np)

        return dsee_outputs_to_observables(output_phys)

    def sample_warmstart(self, mass, Z=0.014, alpha_mlt=ALPHA_MLT, f_ov=None,
                         Y_init=None, age_yr=4.603e9, diffusion=True,
                         opacity_factor=1.0):
        """Generate a warm-start parameter guess for stellar-jax inference.

        Queries DSEE for predicted observables, then packages the input
        parameters + predicted outputs as a dict suitable for initializing
        the infer_parameters optimizer (inverse.py).

        Parameters
        ----------
        (same as predict)

        Returns
        -------
        dict with keys:
            'mass' : float — stellar mass (M_sun)
            'Z' : float — metallicity
            'alpha_mlt' : float — mixing-length parameter
            'f_ov' : float — core overshooting
            'Y_init' : float — initial helium
            't_max' : float — target age (years), for evolve_star
            'diffusion' : bool — diffusion flag
            'log_L_pred' : float — DSEE-predicted log10(L/L_sun)
            'log_Teff_pred' : float — DSEE-predicted log10(T_eff/K)
            'log_R_pred' : float — DSEE-predicted log10(R/R_sun)
            'log_g_pred' : float — DSEE-predicted log10(g)
            'Y_core_pred' : float — DSEE-predicted core He fraction
        """
        if f_ov is None:
            f_ov = F_OV
        if Y_init is None:
            Y_init = Y_BBN + DY_DZ * Z

        obs = self.predict(
            mass=mass, Z=Z, alpha_mlt=alpha_mlt, f_ov=f_ov,
            Y_init=Y_init, age_yr=age_yr, diffusion=diffusion,
            opacity_factor=opacity_factor,
        )

        return {
            'mass': float(mass),
            'Z': float(Z),
            'alpha_mlt': float(alpha_mlt),
            'f_ov': float(f_ov),
            'Y_init': float(Y_init),
            't_max': float(age_yr),
            'diffusion': bool(diffusion),
            # DSEE predictions (warm-start targets)
            'log_L_pred': float(obs['log_L']),
            'log_Teff_pred': float(obs['log_Teff']),
            'log_R_pred': float(obs['log_R']),
            'log_g_pred': float(obs['log_g']),
            'Y_core_pred': float(obs['Y_core']),
        }
