"""stellar-jax: differentiable 1D stellar evolution in JAX.

Float64 is mandatory for stellar structure (solar-calibration tolerance < 1e-7,
sound-speed agreement < 1% vs Model S — both unreachable in float32). Enable it
once here so that importing any submodule guarantees 64-bit arithmetic, regardless
of import order. Individual modules (stellar.py, adaptive_forward.py) also set it
for historical robustness; this is the canonical, earliest-possible enable.

References:
    JAX docs: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
    Design standards §11-A (float64 is mandatory)
"""
import jax
jax.config.update("jax_enable_x64", True)

from stellar_jax.stellar import evolve_star  # noqa: F401
