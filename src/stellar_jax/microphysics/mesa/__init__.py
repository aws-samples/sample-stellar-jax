"""MESA microphysics backend — forward-only validation via bind(C) shims.

Four of five MESA modules (EOS, opacity, neutrino, MLT) are accessed
through compiled Fortran wrappers loaded via ctypes. No gfort2py, no pyMesa.

Nuclear stays JAX — MESA's net_get requires Net_Info workspace allocation
that cannot be done standalone (segfaults). See microphysics/mesa/README.md.
The net_wrapper compiles and loads (validates the build), but is not called.

This package is OPTIONAL. Set STELLAR_MICROPHYSICS=mesa to activate.
Default (jax) uses the repo's own reimplemented tables.

Architecture:
  - Each Fortran shim (eos_wrapper.f90, kap_wrapper.f90, neu_wrapper.f90,
    net_wrapper.f90, mlt_wrapper.f90) exposes a bind(C) interface.
  - bindings.py loads the .so files via ctypes.
  - Each adapter module (eos_adapter, opacity_adapter, etc.) wraps the
    ctypes call in jax.pure_callback for integration with the solver.

Status:
  - EOS (eosDT_get): via libeos_wrapper.so ✓
  - Opacity (kap_get): via libkap_wrapper.so ✓
  - Neutrino (neu_get): via libneu_wrapper.so ✓
  - MLT (set_mlt): via libmlt_wrapper.so ✓ (existing, proven)
  - Nuclear (net_get): BUILDS but not called (Net_Info issue; stays JAX)
"""
from stellar_jax.microphysics.mesa.eos_adapter import eos_lookup
from stellar_jax.microphysics.mesa.opacity_adapter import kappa
from stellar_jax.microphysics.nuclear import epsilon_nuclear  # stays JAX (net_get infeasible)
from stellar_jax.microphysics.mesa.neutrino_adapter import epsilon_neutrino
from stellar_jax.microphysics.mesa.mlt_adapter import mlt_nabla, mlt_nabla_raw

__all__ = ['eos_lookup', 'kappa', 'epsilon_nuclear', 'epsilon_neutrino',
           'mlt_nabla', 'mlt_nabla_raw']
