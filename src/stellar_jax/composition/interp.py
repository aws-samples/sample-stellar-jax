"""Shell-to-composition grid interpolation utility.

DRY extraction: the idiom of sorting shell_data mass fractions and
interpolating quantities onto the composition grid was repeated ~11 times
across burn.py, mix.py, and diffuse.py.  This module provides the shared
helper.

The interpolation pattern is:
  1. Extract mf_shells from shell_data[:, 1]
  2. Sort to ascending order (required by jnp.interp)
  3. Reorder quantities alongside
  4. Call jnp.interp(comp_mfracs, mf_sorted, qty_sorted)

Two sorting approaches existed in the original code:
  - burn.py / mix.py: mf_shells[::-1] (simple reversal — assumes
    shell_data is surface-to-center, i.e. decreasing m_frac)
  - diffuse.py: jnp.argsort(mf_shells) (general, handles any ordering)

Both give identical results on surface-to-center data. This helper uses
argsort for generality; the `_interp_shell_to_comp_reversed` variant uses
[::-1] for callers that need exact floating-point bit-match with the
original mix.py / burn.py code path.
"""
import jax.numpy as jnp


def _interp_shell_to_comp(comp_mfracs, mf_shells, quantity):
    """Interpolate a shell-grid quantity onto the composition grid.

    Uses argsort for generality (matches diffuse.py's original approach).

    Parameters
    ----------
    comp_mfracs : array (N_COMP,)
        Composition grid mass-fraction coordinates (ascending, 0→1).
    mf_shells : array (N_SHELLS,)
        Shell mass fractions (any order — will be sorted).
    quantity : array (N_SHELLS,)
        Values to interpolate (same indexing as mf_shells).

    Returns
    -------
    array (N_COMP,)
        Quantity interpolated onto the composition grid.
    """
    sort_idx = jnp.argsort(mf_shells)
    mf_sorted = mf_shells[sort_idx]
    qty_sorted = quantity[sort_idx]
    return jnp.interp(comp_mfracs, mf_sorted, qty_sorted,
                      left=qty_sorted[0], right=qty_sorted[-1])


def _interp_shell_to_comp_reversed(comp_mfracs, mf_shells, quantity):
    """Interpolate a shell-grid quantity onto the composition grid.

    Uses [::-1] reversal (matches burn.py / mix.py's original approach).
    Assumes mf_shells is in DESCENDING order (surface-to-center).

    Parameters
    ----------
    comp_mfracs : array (N_COMP,)
        Composition grid mass-fraction coordinates (ascending, 0→1).
    mf_shells : array (N_SHELLS,)
        Shell mass fractions in DESCENDING order (surface → center).
    quantity : array (N_SHELLS,)
        Values to interpolate (same indexing as mf_shells).

    Returns
    -------
    array (N_COMP,)
        Quantity interpolated onto the composition grid.
    """
    mf_sorted = mf_shells[::-1]
    qty_sorted = quantity[::-1]
    return jnp.interp(comp_mfracs, mf_sorted, qty_sorted,
                      left=qty_sorted[0], right=qty_sorted[-1])
