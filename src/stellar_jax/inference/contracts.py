"""Contracts for the inference module.

Defines the interfaces that the inference solver depends on:
- ForwardFn: the differentiable forward model contract
- InferenceResult: typed return of infer_parameters
- WarmstartGuess: output of DSEE or pure-mapping warm-start

The ForwardFn protocol is the extension point for physics-calibration knobs.
Adding a new knob (e.g. opacity_factor) requires ONLY adding it to θ and
threading it through evolve_star's StellarParams. The inference module's LM
solver needs NO change — it sees θ → y. jax.jacrev automatically computes
the Jacobian w.r.t. ALL elements of θ.

Differentiability boundary:
- forward_fn (user-provided): MUST be differentiable via jax.jacrev
  (unless jacobian_fn is provided externally)
- DSEEInterface.predict(): NOT differentiable (PyTorch, outside JAX graph)
- z_to_feh / feh_to_z: NOT differentiable (NumPy, parameter-space utility)
- infer_parameters: compile-once via jax.jit-cached Python loop when
  jacobian_fn=None (fully JAX-traceable); plain Python loop when jacobian_fn
  is provided (two-level forward: Brent + IFT).
"""

from typing import Protocol, TypedDict, List
import jax.numpy as jnp


class ForwardFn(Protocol):
    """The differentiable forward model contract.

    Maps a parameter vector θ (1D JAX array of shape (n_params,)) to a
    predicted observables vector y (1D JAX array of shape (n_obs,)).

    MUST be differentiable via jax.jacrev — i.e., the entire chain
    (evolve_star → extract observable) must have no stop_gradient on the
    observables that are being fit.

    Adding a new physics-calibration knob (e.g. opacity_factor) requires
    ONLY adding it to θ and threading it through evolve_star's StellarParams.
    The inference module's LM solver needs NO change — it sees θ → y.

    Example
    -------
    >>> # Mass-only forward model:
    >>> def forward_fn(theta: jnp.ndarray) -> jnp.ndarray:
    ...     result = evolve_star(mass=theta[0], Z=0.014, ...)
    ...     return jnp.array([result['log_L'][-1], result['log_Teff'][-1]])

    >>> # Mass + opacity_factor (1-line extension):
    >>> def forward_fn(theta: jnp.ndarray) -> jnp.ndarray:
    ...     result = evolve_star(mass=theta[0], opacity_factor=theta[1], ...)
    ...     return jnp.array([result['log_L'][-1], result['log_Teff'][-1]])
    """

    def __call__(self, theta: jnp.ndarray) -> jnp.ndarray: ...


class InferenceResult(TypedDict):
    """Return type of infer_parameters.

    All fields are populated by the LM solver upon completion.
    """

    theta: jnp.ndarray  # Final parameter estimates, shape (n_params,)
    chi2: float  # Final χ² value
    jacobian: jnp.ndarray  # Jacobian at solution, shape (n_obs, n_params)
    covariance: jnp.ndarray  # Parameter covariance, shape (n_params, n_params)
    sigma_params: jnp.ndarray  # 1-σ uncertainties, shape (n_params,)
    converged: bool
    n_iter: int
    history: List[dict]
    residuals: jnp.ndarray  # Final residuals (n_obs,)
    param_names: List[str]


class WarmstartGuess(TypedDict):
    """Output of the DSEE warm-start or pure-mapping warm-start.

    Provides a physically motivated initial guess for stellar-jax's
    infer_parameters optimizer. Can come from:
    1. DSEEInterface.sample_warmstart() — emulator-based (requires PyTorch)
    2. warmstart_params_from_observables() — pure-mapping (no dependencies)
    """

    mass: float
    Z: float
    alpha_mlt: float
    f_ov: float
    Y_init: float
    t_max: float
    diffusion: bool
