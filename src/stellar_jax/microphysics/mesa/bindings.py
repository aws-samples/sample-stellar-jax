"""Low-level ctypes bindings to MESA shared-library wrappers.

Architecture: each MESA module (EOS, kap, neu, net, MLT) is wrapped by a thin
Fortran shim (`*_wrapper.f90`) compiled into a shared library (`lib*_wrapper.so`).
The shims expose plain C-compatible interfaces (bind(C)) that take/return
real(c_double) and integer(c_int). This module loads those .so files via ctypes
and exposes Python callables.

This replaces the prior gfort2py/pyMesa approach, which leaked memory on every
call (per-call ctypes.Structure subclass creation) and couldn't handle MESA's
derived types or 2D allocatable arrays.

Requirements:
  - Compiled shim libraries on LD_LIBRARY_PATH or at MESA_LIBS_DIR/lib/:
    libeos_wrapper.so, libkap_wrapper.so, libneu_wrapper.so, libnet_wrapper.so,
    libmlt_wrapper.so
  - MESA data at MESA_LIBS_DIR/data/ (EOS tables, kap tables, etc.)
  - No gfort2py, no pyMesa, no runtime Fortran compiler needed.

References:
  - MESA "Just a Module" usage: Paxton et al. (2011), ApJS 192, 3, §8
  - bind(C) interop: Fortran 2003 ISO_C_BINDING standard
"""
import os
import ctypes as ct
import pathlib


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_MESA_DIR = os.environ.get('MESA_DIR',
             os.environ.get('MESA_LIBS_DIR', '/cache/mesa_libs'))

# Default nuclear network
_NET_NAME = os.environ.get('MESA_NET_NAME', 'pp_cno_extras_o18_ne22.net')

# Default Zbase for opacity (solar metallicity)
_ZBASE = float(os.environ.get('MESA_ZBASE', '0.014'))


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------
_libs = {}  # cache: name -> ctypes.CDLL


def _load_lib(name):
    """Load a shared library by name, searching standard paths."""
    if name in _libs:
        return _libs[name]

    candidates = [
        pathlib.Path(_MESA_DIR) / 'lib' / f'lib{name}.so',
        pathlib.Path(__file__).parent / f'lib{name}.so',
    ]
    for p in candidates:
        if p.exists():
            lib = ct.CDLL(str(p), mode=ct.RTLD_GLOBAL)
            _libs[name] = lib
            return lib

    # Fall back to system LD_LIBRARY_PATH search
    try:
        lib = ct.CDLL(f'lib{name}.so', mode=ct.RTLD_GLOBAL)
        _libs[name] = lib
        return lib
    except OSError:
        raise OSError(
            f"lib{name}.so not found. Searched:\n"
            + '\n'.join(f'  {p}' for p in candidates)
            + '\n  LD_LIBRARY_PATH (system search)\n'
            f"Build it from microphysics/mesa/{name.replace('_wrapper', '')}_wrapper.f90"
        )


# ---------------------------------------------------------------------------
# Initialization state
# ---------------------------------------------------------------------------
_eos_initialized = False
_kap_initialized = False
_neu_initialized = False
_net_initialized = False
_mlt_initialized = False  # MLT has no separate init (uses existing pattern)


def _mesa_dir_cstr():
    """Return (ctypes char array, length) for MESA_DIR."""
    b = _MESA_DIR.encode('ascii')
    return ct.create_string_buffer(b, len(b)), ct.c_int(len(b))


def _ensure_eos_init():
    """Initialize the EOS wrapper (idempotent)."""
    global _eos_initialized
    if _eos_initialized:
        return
    lib = _load_lib('eos_wrapper')
    mesa_buf, mesa_len = _mesa_dir_cstr()
    ierr = ct.c_int(0)
    lib.eos_wrapper_init(mesa_buf, mesa_len, ct.byref(ierr))
    if ierr.value != 0:
        raise RuntimeError(f"MESA EOS init failed (ierr={ierr.value}). "
                          f"Check MESA_DIR={_MESA_DIR} has data/eosDT_data/")
    _eos_initialized = True


def _ensure_kap_init():
    """Initialize the kap wrapper (idempotent)."""
    global _kap_initialized
    if _kap_initialized:
        return
    lib = _load_lib('kap_wrapper')
    mesa_buf, mesa_len = _mesa_dir_cstr()
    ierr = ct.c_int(0)
    lib.kap_wrapper_init(mesa_buf, mesa_len, ct.c_double(_ZBASE), ct.byref(ierr))
    if ierr.value != 0:
        raise RuntimeError(f"MESA kap init failed (ierr={ierr.value}). "
                          f"Check MESA_DIR={_MESA_DIR} has data/kap_data/")
    _kap_initialized = True


def _ensure_neu_init():
    """Initialize the neutrino wrapper (idempotent)."""
    global _neu_initialized
    if _neu_initialized:
        return
    lib = _load_lib('neu_wrapper')
    mesa_buf, mesa_len = _mesa_dir_cstr()
    ierr = ct.c_int(0)
    lib.neu_wrapper_init(mesa_buf, mesa_len, ct.byref(ierr))
    if ierr.value != 0:
        raise RuntimeError(f"MESA neutrino init failed (ierr={ierr.value}). "
                          f"Check MESA_DIR={_MESA_DIR}")
    _neu_initialized = True


def _ensure_net_init():
    """Initialize the nuclear network wrapper (idempotent)."""
    global _net_initialized
    if _net_initialized:
        return
    lib = _load_lib('net_wrapper')
    mesa_buf, mesa_len = _mesa_dir_cstr()
    net_bytes = _NET_NAME.encode('ascii')
    net_buf = ct.create_string_buffer(net_bytes, len(net_bytes))
    net_len = ct.c_int(len(net_bytes))
    ierr = ct.c_int(0)
    lib.net_wrapper_init(mesa_buf, mesa_len, net_buf, net_len, ct.byref(ierr))
    if ierr.value != 0:
        raise RuntimeError(
            f"MESA net init failed (ierr={ierr.value}). "
            f"Check MESA_DIR={_MESA_DIR} has data/net_data/ and data/rates_data/. "
            f"Network: {_NET_NAME}")
    _net_initialized = True


def _ensure_mesa_init():
    """Initialize ALL MESA wrappers. Called by the dispatch layer.

    Note: net (nuclear) init is SKIPPED — MESA's net_get requires Net_Info
    workspace allocation that cannot be done standalone (see README.md).
    Nuclear stays JAX even when STELLAR_MICROPHYSICS=mesa.
    """
    _ensure_eos_init()
    _ensure_kap_init()
    _ensure_neu_init()
    # _ensure_net_init() — intentionally skipped: MESA's net_get requires
    # Net_Info workspace that cannot be properly allocated standalone.
    # The net_wrapper compiles and loads (AC1/AC2) but calling net_get_plain
    # at runtime segfaults. Nuclear stays JAX by design.


# ---------------------------------------------------------------------------
# Public API: EOS
# ---------------------------------------------------------------------------

def call_mesa_eos(logT, logRho, X, Z):
    """Call MESA EOS via the bind(C) wrapper.

    Args:
        logT: log10(T/K)
        logRho: log10(rho / g/cm³)
        X: hydrogen mass fraction
        Z: metal mass fraction

    Returns:
        tuple: (rho, mu, nabla_ad, S, cp, chi_rho, chi_T, lnPgas)
          rho      — density [g/cm³]
          mu       — mean molecular weight
          nabla_ad — adiabatic temperature gradient
          S        — specific entropy [erg/g/K]
          cp       — specific heat at constant P [erg/g/K]
          chi_rho  — (d ln P / d ln rho)_T
          chi_T    — (d ln P / d ln T)_rho
          lnPgas   — ln(P_gas) from MESA (natural log, includes all EOS physics)
    """
    _ensure_eos_init()
    lib = _load_lib('eos_wrapper')

    # Output variables
    rho_out = ct.c_double()
    mu_out = ct.c_double()
    nad_out = ct.c_double()
    S_out = ct.c_double()
    cp_out = ct.c_double()
    chi_rho_out = ct.c_double()
    chi_T_out = ct.c_double()
    lnPgas_out = ct.c_double()
    ierr = ct.c_int(0)

    lib.eos_get_plain(
        ct.c_double(float(logT)),
        ct.c_double(float(logRho)),
        ct.c_double(float(X)),
        ct.c_double(float(Z)),
        ct.byref(rho_out),
        ct.byref(mu_out),
        ct.byref(nad_out),
        ct.byref(S_out),
        ct.byref(cp_out),
        ct.byref(chi_rho_out),
        ct.byref(chi_T_out),
        ct.byref(lnPgas_out),
        ct.byref(ierr),
    )

    if ierr.value != 0:
        raise RuntimeError(f"MESA eos_get_plain failed (ierr={ierr.value})")

    return (rho_out.value, mu_out.value, nad_out.value,
            S_out.value, cp_out.value, chi_rho_out.value, chi_T_out.value,
            lnPgas_out.value)


# ---------------------------------------------------------------------------
# Public API: Opacity
# ---------------------------------------------------------------------------

def call_mesa_kap(logT, logRho, X, Z):
    """Call MESA opacity via the bind(C) wrapper.

    Args:
        logT: log10(T/K)
        logRho: log10(rho / g/cm³)
        X: hydrogen mass fraction
        Z: metal mass fraction

    Returns:
        tuple: (log_kappa, dlnkap_dlnT, dlnkap_dlnRho)
          log_kappa       — log10(kappa / cm²/g)
          dlnkap_dlnT     — d(ln kappa)/d(ln T)|rho
          dlnkap_dlnRho   — d(ln kappa)/d(ln rho)|T
    """
    _ensure_kap_init()
    lib = _load_lib('kap_wrapper')

    log_kappa_out = ct.c_double()
    dlnkap_dlnT_out = ct.c_double()
    dlnkap_dlnRho_out = ct.c_double()
    ierr = ct.c_int(0)

    lib.kap_get_plain(
        ct.c_double(float(logT)),
        ct.c_double(float(logRho)),
        ct.c_double(float(X)),
        ct.c_double(float(Z)),
        ct.byref(log_kappa_out),
        ct.byref(dlnkap_dlnT_out),
        ct.byref(dlnkap_dlnRho_out),
        ct.byref(ierr),
    )

    if ierr.value != 0:
        raise RuntimeError(f"MESA kap_get_plain failed (ierr={ierr.value})")

    return (log_kappa_out.value, dlnkap_dlnT_out.value, dlnkap_dlnRho_out.value)


# ---------------------------------------------------------------------------
# Public API: Neutrinos
# ---------------------------------------------------------------------------

def call_mesa_neu(logT, logRho, X, Z):
    """Call MESA neutrino loss via the bind(C) wrapper.

    Args:
        logT: log10(T/K)
        logRho: log10(rho / g/cm³)
        X: hydrogen mass fraction
        Z: metal mass fraction

    Returns:
        float: eps_neu (erg/g/s, positive = energy lost from star)
    """
    _ensure_neu_init()
    lib = _load_lib('neu_wrapper')

    eps_neu_out = ct.c_double()
    ierr = ct.c_int(0)

    lib.neu_get_plain(
        ct.c_double(float(logT)),
        ct.c_double(float(logRho)),
        ct.c_double(float(X)),
        ct.c_double(float(Z)),
        ct.byref(eps_neu_out),
        ct.byref(ierr),
    )

    if ierr.value != 0:
        raise RuntimeError(f"MESA neu_get_plain failed (ierr={ierr.value})")

    return eps_neu_out.value


# ---------------------------------------------------------------------------
# Public API: Nuclear
# ---------------------------------------------------------------------------

def call_mesa_net(rho, T, X, Z):
    """Call MESA nuclear network via the bind(C) wrapper.

    Args:
        rho: density [g/cm³]
        T: temperature [K]
        X: hydrogen mass fraction
        Z: metal mass fraction

    Returns:
        float: eps_nuc (erg/g/s)
    """
    _ensure_net_init()
    lib = _load_lib('net_wrapper')

    eps_nuc_out = ct.c_double()
    ierr = ct.c_int(0)

    lib.net_get_plain(
        ct.c_double(float(rho)),
        ct.c_double(float(T)),
        ct.c_double(float(X)),
        ct.c_double(float(Z)),
        ct.byref(eps_nuc_out),
        ct.byref(ierr),
    )

    if ierr.value != 0:
        raise RuntimeError(f"MESA net_get_plain failed (ierr={ierr.value})")

    return eps_nuc_out.value


# ---------------------------------------------------------------------------
# Public API: MLT (uses existing libmlt_wrapper.so — unchanged from prior work)
# ---------------------------------------------------------------------------

_mlt_lib_handle = None


def _load_mlt_lib():
    """Load the MLT wrapper library."""
    global _mlt_lib_handle
    if _mlt_lib_handle is not None:
        return _mlt_lib_handle

    candidates = [
        pathlib.Path(__file__).parent / 'libmlt_wrapper.so',
        pathlib.Path(_MESA_DIR) / 'lib' / 'libmlt_wrapper.so',
    ]
    for p in candidates:
        if p.exists():
            _mlt_lib_handle = ct.CDLL(str(p), mode=ct.RTLD_GLOBAL)
            return _mlt_lib_handle

    try:
        _mlt_lib_handle = ct.CDLL('libmlt_wrapper.so', mode=ct.RTLD_GLOBAL)
    except OSError:
        raise OSError(
            "libmlt_wrapper.so not found. Build from mlt_wrapper.f90 or "
            "set LD_LIBRARY_PATH to include its directory.")
    return _mlt_lib_handle


def call_mesa_mlt(chiT, chiRho, Cp, grav, Lambda, rho, P, T, opacity,
                  gradr, grada, gradL,
                  mixing_length_alpha, mlt_option='Henyey',
                  Henyey_MLT_nu_param=8.0, Henyey_MLT_y_param=1.0/3.0,
                  max_conv_vel=1e99):
    """Call MESA set_mlt via the Fortran ctypes wrapper.

    All auto_diff_real_star_order1 inputs are passed as plain real(dp) scalars;
    the wrapper constructs the derived types, calls set_mlt, and extracts %val
    from the outputs.

    Returns:
        dict with keys: 'gradT', 'Gamma', 'Y_face', 'conv_vel', 'D', 'mixing_type'
    """
    lib = _load_mlt_lib()

    # String argument
    mlt_bytes = mlt_option.encode('ascii')
    mlt_len = ct.c_int(len(mlt_bytes))
    mlt_chars = ct.create_string_buffer(mlt_bytes, len(mlt_bytes))

    # Output variables
    Gamma_out = ct.c_double()
    gradT_out = ct.c_double()
    Y_face_out = ct.c_double()
    conv_vel_out = ct.c_double()
    D_out = ct.c_double()
    mixing_type_out = ct.c_int()
    ierr_out = ct.c_int()

    lib.set_mlt_plain(
        mlt_len, mlt_chars,
        ct.c_double(float(mixing_length_alpha)),
        ct.c_double(float(Henyey_MLT_nu_param)),
        ct.c_double(float(Henyey_MLT_y_param)),
        ct.c_double(float(max_conv_vel)),
        ct.c_double(float(chiT)),
        ct.c_double(float(chiRho)),
        ct.c_double(float(Cp)),
        ct.c_double(float(grav)),
        ct.c_double(float(Lambda)),
        ct.c_double(float(rho)),
        ct.c_double(float(P)),
        ct.c_double(float(T)),
        ct.c_double(float(opacity)),
        ct.c_double(float(gradr)),
        ct.c_double(float(grada)),
        ct.c_double(float(gradL)),
        ct.byref(Gamma_out),
        ct.byref(gradT_out),
        ct.byref(Y_face_out),
        ct.byref(conv_vel_out),
        ct.byref(D_out),
        ct.byref(mixing_type_out),
        ct.byref(ierr_out),
    )

    if ierr_out.value != 0:
        raise RuntimeError(f"MESA set_mlt failed (ierr={ierr_out.value})")

    return {
        'gradT': gradT_out.value,
        'Gamma': Gamma_out.value,
        'Y_face': Y_face_out.value,
        'conv_vel': conv_vel_out.value,
        'D': D_out.value,
        'mixing_type': mixing_type_out.value,
    }
