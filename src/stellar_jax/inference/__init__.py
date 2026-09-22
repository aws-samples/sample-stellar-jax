"""Inference module: gradient-based inverse solver + DSEE emulator warm-start.

Extracted from inverse.py + dsee_interface.py (issue #501, refactor P1.5).
These are PURE functions / leaf consumers of the differentiable forward model —
they contain no @custom_vjp, no lax.scan, no stop_gradient policy decisions.

Structure:
- solver.py: infer_parameters (Levenberg-Marquardt, kept whole as a cohesive kernel)
- emulator.py: DSEEInterface + DSEE normalization/mapping (non-differentiable, PyTorch)
- metallicity.py: z_to_feh, feh_to_z (shared utility, no external deps)
- warmstart.py: warmstart_params_from_observables (pure mapping, no emulator)
- contracts.py: ForwardFn Protocol, InferenceResult/WarmstartGuess TypedDicts

Differentiability contract:
- forward_fn (user-provided to solver): MUST be differentiable via jax.jacrev
- Everything else in this module: NOT in the gradient path (used for initialization only)

See the module redesign spec for the authoritative design.
"""

from stellar_jax.inference.solver import infer_parameters
from stellar_jax.inference.emulator import (
    DSEEInterface,
    stellar_jax_to_dsee_labels,
    dsee_outputs_to_observables,
    # Normalization internals — re-exported for test access
    _normalize_labels,
    _denormalize_outputs,
    _norm_linear_rescale,
    _norm_normal,
    _norm_mantissa_exponent,
    DSEE_LABEL_DEFAULT,
    DSEE_LABEL_HEADER,
    DSEE_LABEL_SCALER,
    DSEE_LABEL_SCALE,
    DSEE_DATA_HEADER,
    DSEE_DATA_SCALER,
    DSEE_DATA_SCALE,
)
from stellar_jax.inference.metallicity import z_to_feh, feh_to_z, Z_SOLAR, ZX_SOLAR, X_SOLAR
from stellar_jax.inference.warmstart import (
    warmstart_params_from_observables,
    make_warmstart_fn,
    make_dsee_warmstart_fn,
)
from stellar_jax.inference.contracts import ForwardFn, InferenceResult, WarmstartGuess
from stellar_jax.inference.fisher import (
    compute_jacobian,
    compute_jacobian_fd,
    compute_fisher,
    forecast_sigma,
    _safe_invert_fisher,
    JacobianResult,
    FisherResult,
    DEFAULT_PARAM_NAMES,
    DEFAULT_FIDUCIAL,
    DEFAULT_FD_REL_STEPS,
)
from stellar_jax.inference.mcmc import run_mcmc, MCMCResult
from stellar_jax.inference.age_composition import compute_age_bias_attribution
from stellar_jax.inference.plato_forecast import (
    plato_per_target_forecast,
    PLATO_SIGMA_NU,
    PLATO_SIGMA_LOG_L,
    PLATO_SIGMA_LOG_TEFF,
    ALPHA_MR,
    F_AGE_WITH_L2,
    F_AGE_WITHOUT_L2,
)

__all__ = [
    # Solver
    "infer_parameters",
    # Emulator
    "DSEEInterface",
    "stellar_jax_to_dsee_labels",
    "dsee_outputs_to_observables",
    # Metallicity
    "z_to_feh",
    "feh_to_z",
    "Z_SOLAR",
    "ZX_SOLAR",
    "X_SOLAR",
    # Warmstart
    "warmstart_params_from_observables",
    "make_warmstart_fn",
    "make_dsee_warmstart_fn",
    # Contracts
    "ForwardFn",
    "InferenceResult",
    "WarmstartGuess",
    # Fisher matrix
    "compute_jacobian",
    "compute_jacobian_fd",
    "compute_fisher",
    "forecast_sigma",
    "_safe_invert_fisher",
    "JacobianResult",
    "FisherResult",
    "DEFAULT_PARAM_NAMES",
    "DEFAULT_FIDUCIAL",
    "DEFAULT_FD_REL_STEPS",
    # MCMC validation
    "run_mcmc",
    "MCMCResult",
    # Age-bias attribution
    "compute_age_bias_attribution",
    # PLATO forecast demo
    "plato_per_target_forecast",
    "PLATO_SIGMA_NU",
    "PLATO_SIGMA_LOG_L",
    "PLATO_SIGMA_LOG_TEFF",
    "ALPHA_MR",
    "F_AGE_WITH_L2",
    "F_AGE_WITHOUT_L2",
    # Normalization internals (for tests)
    "_normalize_labels",
    "_denormalize_outputs",
    "_norm_linear_rescale",
    "_norm_normal",
    "_norm_mantissa_exponent",
    "DSEE_LABEL_DEFAULT",
    "DSEE_LABEL_HEADER",
    "DSEE_LABEL_SCALER",
    "DSEE_LABEL_SCALE",
    "DSEE_DATA_HEADER",
    "DSEE_DATA_SCALER",
    "DSEE_DATA_SCALE",
]
