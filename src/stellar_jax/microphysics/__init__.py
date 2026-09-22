"""Microphysics modules for stellar-jax.

Provides EOS, opacity, nuclear energy generation, and neutrino losses.

Backend dispatch: set STELLAR_MICROPHYSICS=mesa to use MESA Fortran modules
via compiled bind(C) shims (no gfort2py, no pyMesa). Requires the compiled
wrapper libraries (libeos_wrapper.so, libkap_wrapper.so, libneu_wrapper.so,
libnet_wrapper.so, libmlt_wrapper.so) and MESA data.

Default is 'jax' (pure JAX table interpolation — production path, differentiable).

The MESA backend is FORWARD-ONLY (not differentiable) — used purely for
validation that the JAX microphysics reproduces the same physics.
"""
import os

_BACKEND = os.environ.get('STELLAR_MICROPHYSICS', 'jax').lower()

if _BACKEND == 'mesa':
    # FAIL LOUDLY if MESA is requested but unavailable.
    # No silent fallback to JAX — that enabled hollow-green results in the past.
    try:
        from stellar_jax.microphysics.mesa import (eos_lookup, kappa, epsilon_nuclear,
                                       epsilon_neutrino, mlt_nabla, mlt_nabla_raw)
        from stellar_jax.microphysics.mesa.bindings import _ensure_mesa_init
        _ensure_mesa_init()
    except (ImportError, OSError, RuntimeError) as e:
        raise RuntimeError(
            f"STELLAR_MICROPHYSICS=mesa requested but MESA backend is unavailable.\n"
            f"Reason: {e}\n"
            f"Either install the MESA wrapper libraries or unset STELLAR_MICROPHYSICS.\n"
            f"There is NO silent fallback to JAX — set STELLAR_MICROPHYSICS=jax explicitly "
            f"if you want the JAX backend."
        ) from e

    # Patch submodule attributes so solver imports pick up the MESA version.
    import stellar_jax.microphysics.eos as _eos_mod
    import stellar_jax.microphysics.opacity as _opacity_mod
    import stellar_jax.microphysics.nuclear as _nuclear_mod
    import stellar_jax.microphysics.neutrino as _neutrino_mod
    _eos_mod.eos_lookup = eos_lookup
    _opacity_mod.kappa = kappa
    _nuclear_mod.epsilon_nuclear = epsilon_nuclear
    _neutrino_mod.epsilon_neutrino = epsilon_neutrino

    # Patch transport.mlt_nabla so the solver uses MESA's MLT
    import stellar_jax.transport as _transport_mod
    _transport_mod.mlt_nabla = mlt_nabla
    _transport_mod.mlt_nabla_raw = mlt_nabla_raw

    # Re-export everything solvers depend on
    from stellar_jax.microphysics.eos import (
        _build_eos_pgrid, EOS_X, EOS_Z, EOS_LOGT, EOS_LOGP,
        EOS_LOGRHO, EOS_MU, EOS_NAD, eos_lookup_bicubic,
    )
    from stellar_jax.microphysics.opacity import (
        OPAL_X, OPAL_Z, OPAL_LOGT, OPAL_LOGR, OPAL_LK,
        FERG_X, FERG_Z, FERG_LOGT, FERG_LOGR, FERG_LK,
        _interp4d_logkappa, opal_kappa, ferguson_kappa, _ferguson_weight,
    )
    from stellar_jax.microphysics.nuclear import _eps_nuclear_np

else:
    # Default JAX backend — production path, fully differentiable
    from stellar_jax.microphysics.eos import (
        _build_eos_pgrid, EOS_X, EOS_Z, EOS_LOGT, EOS_LOGP,
        EOS_LOGRHO, EOS_MU, EOS_NAD, eos_lookup, eos_lookup_bicubic,
    )
    from stellar_jax.microphysics.opacity import (
        OPAL_X, OPAL_Z, OPAL_LOGT, OPAL_LOGR, OPAL_LK,
        FERG_X, FERG_Z, FERG_LOGT, FERG_LOGR, FERG_LK,
        _interp4d_logkappa, opal_kappa, ferguson_kappa, _ferguson_weight, kappa,
    )
    from stellar_jax.microphysics.nuclear import epsilon_nuclear, _eps_nuclear_np
    from stellar_jax.microphysics.neutrino import epsilon_neutrino
