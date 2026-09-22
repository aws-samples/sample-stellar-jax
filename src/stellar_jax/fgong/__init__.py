"""fgong/ — Structure→FGONG conversion + FGONG/GYRE I/O.

The differentiable bridge between the Henyey solver (evolution/) and the
oscillation eigensolver (oscillations/).

Public API:
    structure_to_fgong_jax  — differentiable builder (pure JAX, gradient path)
    hires_profile           — high-res RK4 structure profile (for I/O only)
    write_fgong             — write publication-grade FGONG v3.00 file
    write_gyre              — write GYRE MESA v1.01 format file
    read_fgong              — parse FGONG file → (glob, var) NumPy arrays
    fgong_components        — named dict from FGONG arrays
"""

from stellar_jax.fgong.builder import structure_to_fgong_jax
from stellar_jax.fgong.hires_profile import hires_profile
from stellar_jax.fgong.io import write_fgong, write_gyre, read_fgong, fgong_components

__all__ = [
    'structure_to_fgong_jax',
    'hires_profile',
    'write_fgong',
    'write_gyre',
    'read_fgong',
    'fgong_components',
]
