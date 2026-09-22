"""Public API contract for microphysics/.

All physics functions are PURE in their evaluated path (no side-effects during
JAX tracing or execution), DIFFERENTIABLE (JAX-traceable in both forward and
reverse mode), and LEAF (no imports from solver/evolution/composition/mesh).

Configuration — two mechanisms control physics dispatch:
  - bicubic_opacity (opacity.py) and helm_eos (eos.py): Python bool parameters
    threaded explicitly through the call chain (#1070). Different values
    produce different XLA graphs (via separate JIT cache entries).
  - Backend dispatch (__init__.py): when STELLAR_MICROPHYSICS=mesa, the
    __init__.py patches submodule attributes (eos_lookup, kappa, etc.) to
    the MESA bind(C) implementations. This is a load-time dispatch that
    swaps the entire backend, not a per-call mutation.
  Neither mechanism mutates state DURING JAX tracing or execution — they
  configure the static graph before JIT compilation. The traced functions
  themselves remain pure.

stop_gradient: NONE within this package. Gradient policy is owned by
evolution/ (the orchestrator), never by leaf physics modules.

@custom_jvp boundaries (3 total):
  - safe_math.safe_power: bounded gradient for x^p near x=0.
    p and x_floor are static Python floats (nondiff by contract).
  - nuclear._pow_3_2: bounded gradient for x^{3/2} near x=0.
  - helm._solve_density_ift: IFT at HELM density inversion.

Physics-calibration knobs (opacity_factor, eps_nuc_factor, etc.) are
applied by the CALLER (solver), NOT inside this package. This keeps
microphysics pure (Open/Closed principle: closed for modification,
open for extension via caller-side scaling).

Backend dispatch: set STELLAR_MICROPHYSICS=mesa for the forward-only
MESA validation backend. Default 'jax' is the production path.
"""

from typing import Optional, Tuple

# Type alias: all inputs/outputs are JAX-traced f64 scalars unless noted.
Scalar = float  # jax.Array scalar (float64) at runtime


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API — the functions the solver may call
# ═══════════════════════════════════════════════════════════════════════

def eos_lookup(logT: Scalar, logP: Scalar, X: Scalar, Z: Scalar
               ) -> Tuple[Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar]:
    """OPAL EOS: (logT, logP, X, Z) → (rho, mu, nad, S, cp, chi_rho, chi_T).

    Fully differentiable via trilinear 4D table interpolation.
    Inputs clamped to table bounds (smooth, no discontinuity).
    """
    ...


def eos_lookup_bicubic(logT: Scalar, logP: Scalar, X: Scalar, Z: Scalar
                       ) -> Tuple[Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar]:
    """Higher-order EOS lookup (Catmull-Rom in T,P). Same contract as eos_lookup."""
    ...


def kappa(logT: Scalar, logRho: Scalar, X: Scalar, Z: Scalar,
          opacity_factor: Optional[Scalar] = None) -> Scalar:
    """Blended opacity: Ferguson + OPAL + Compton + conduction → log10(kappa).

    Fully differentiable. opacity_factor (default 1.0) is the MESA
    control `opacity_factor` — applied here as a convenience but
    conceptually a caller-side knob.
    """
    ...


def epsilon_nuclear(rho: Scalar, T: Scalar, X: Scalar, Z: Scalar,
                    t_age: Scalar = 1e9, X_N14: Optional[Scalar] = None
                    ) -> Scalar:
    """Nuclear energy generation (PP + CNO) with Chugunov screening → erg/g/s.

    Differentiable via safe_power @custom_jvp for T6^(1/3) near zero.
    """
    ...


def epsilon_neutrino(rho: Scalar, T: Scalar, X: Scalar, Z: Scalar) -> Scalar:
    """Neutrino losses (plasma + photo + pair) → erg/g/s (always >= 0).

    Fully differentiable via safe_power @custom_jvp for near-zero args.
    """
    ...


def helm_eos_full(logT: Scalar, logP_target: Scalar, X: Scalar, Z: Scalar
                  ) -> Tuple[Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar]:
    """HELM EOS for degenerate regime. Same output contract as eos_lookup.

    Differentiable via IFT @custom_jvp at the NR density inversion.
    """
    ...


def safe_power(x: Scalar, p: float, x_floor: float) -> Scalar:
    """x^p with AD-safe gradient bounded near x=0.

    @custom_jvp: forward = exact x^p; backward = p*max(x, x_floor)^(p-1).
    p and x_floor are static Python floats (nondiff_argnums by contract —
    never pass JAX-traced values for these).
    """
    ...
