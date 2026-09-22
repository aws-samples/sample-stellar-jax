"""Transport module: MLT convective energy transport + Schwarzschild switch.

Public API (preserves existing `from transport import X` contract):
  - mlt_nabla: MLT gradient with smooth alpha_mlt adjoint (@custom_jvp switch)
  - mlt_nabla_raw: MLT gradient with exact hard-switch derivative (Newton Jacobian)
  - _mlt_switch: the @custom_jvp Schwarzschild switch (exported for direct testing)

Internal (not part of public API):
  - _mlt_cubic_solve: the shared cubic solver kernel (import from transport.mlt)
  - _constants: MLT geometry constants (FF1-FF4)
"""

from stellar_jax.transport.mlt import mlt_nabla, mlt_nabla_raw, _mlt_cubic_solve  # noqa: F401
from stellar_jax.transport.switch import _mlt_switch  # noqa: F401

__all__ = ['mlt_nabla', 'mlt_nabla_raw', '_mlt_switch', '_mlt_cubic_solve']
