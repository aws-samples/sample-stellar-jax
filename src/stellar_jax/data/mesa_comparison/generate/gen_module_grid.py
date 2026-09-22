#!/usr/bin/env python3
"""Generate the MESA microphysics reference grid (EOS / opacity / neutrino).

PURPOSE
-------
Precompute MESA module outputs over a fixed (logT, logRho, X, Z) grid, committing them
(`data/mesa_module_grid.npz`), so the per-module parity tests can compare our JAX modules
against the saved arrays without needing live MESA at test time. See issue #756.

The former live-MESA `test_mesa_microphysics_parity` test was removed in #1002 — the
pure_callback overhead made evolved-track comparison infeasible in CI.

Mass is NOT an input — microphysics is a function of LOCAL conditions (T, ρ, X, Z), so ONE grid
covers all masses (and it's systematic coverage, not one track's zones).

WHERE / HOW TO RUN
------------------
On a host with the built bind(C) shims + a MESA install (with a SHARED
MESA build at ~/mesa; the static ~/mesa-26.04.1 will NOT link a shared wrapper). Calls the RAW
ctypes shims directly (`microphysics.mesa.bindings`) — NO jax, so it is jax-version-independent.

    # build the 3 shims (once) against the SHARED MESA:
    export MESASDK_ROOT=$HOME/mesasdk; source $MESASDK_ROOT/bin/mesasdk_init.sh
    make -f Makefile.mlt_wrapper FC=gfortran MESA_DIR=$HOME/mesa MESA_SDK_LIB=$MESASDK_ROOT/lib \
        microphysics/mesa/lib{eos,kap,neu}_wrapper.so
    # generate:
    export MESA_DIR=$HOME/mesa LD_LIBRARY_PATH=$HOME/mesa/lib:$HOME/mesasdk/lib
    export STELLAR_MICROPHYSICS=mesa MESA_ZBASE=0.014
    python3 data/mesa_comparison/generate/gen_module_grid.py --out data/mesa_module_grid.npz

Config alignment (MODE-A, verified): MESA r26.04.1 default `kap_file_prefix='gs98'` matches the
committed inlist; `MESA_ZBASE=0.014`; net `pp_cno_extras_o18_ne22`. Regenerate ONLY if MODE-A
physics changes (same provenance rule as the FGONG reference data).

OUTPUT (npz)
------------
points   (N,4): (logT, logRho, X, Z)
eos      (N,8): rho, mu, nabla_ad, S, cp, chi_rho, chi_T, lnPgas   (MESA eosDT_get)
kap      (N,3): log10 kappa, dlnkap/dlnRho, dlnkap/dlnT            (MESA kap_get)
neu      (N,1): epsilon_neutrino [erg/g/s]                          (MESA neu_get)
+ axes_*, eos_cols, and provenance (mode/mesa/kap_prefix/zbase).

MLT is intentionally NOT gridded (coupled convective signature, not a pure (T,ρ,X,Z) lookup).
"""
from __future__ import annotations
import argparse
import os
import numpy as np

LOGT = np.round(np.linspace(4.0, 7.8, 20), 4)      # MS interior; below the logT=7.9 degeneracy clamp
LOGRHO = np.round(np.linspace(-9.0, 2.0, 20), 4)    # envelope .. core density
XVALS = np.array([0.00, 0.35, 0.70])                # H mass fraction (core-burnt .. envelope)
ZVALS = np.array([0.014, 0.02])                     # MODE-A Z=0.014 + a second point


def _arr(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=np.float64)).ravel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mesa_module_grid.npz")
    args = ap.parse_args()
    if os.environ.get("STELLAR_MICROPHYSICS") != "mesa":
        raise SystemExit("Set STELLAR_MICROPHYSICS=mesa (this driver captures the MESA backend).")

    from stellar_jax.microphysics.mesa.bindings import (call_mesa_eos, call_mesa_kap, call_mesa_neu,
                                            _ensure_eos_init, _ensure_kap_init, _ensure_neu_init)
    _ensure_eos_init(); _ensure_kap_init(); _ensure_neu_init()

    pts, eos_o, kap_o, neu_o = [], [], [], []
    for lt in LOGT:
        for lr in LOGRHO:
            for x in XVALS:
                for z in ZVALS:
                    pts.append((lt, lr, x, z))
                    eos_o.append(_arr(call_mesa_eos(float(lt), float(lr), float(x), float(z))))
                    kap_o.append(_arr(call_mesa_kap(float(lt), float(lr), float(x), float(z))))
                    neu_o.append(_arr(call_mesa_neu(float(lt), float(lr), float(x), float(z))))

    pts = np.asarray(pts)
    eos_o, kap_o, neu_o = np.vstack(eos_o), np.vstack(kap_o), np.vstack(neu_o)
    np.savez_compressed(
        args.out,
        points=pts, eos=eos_o, kap=kap_o, neu=neu_o,
        axes_logT=LOGT, axes_logRho=LOGRHO, axes_X=XVALS, axes_Z=ZVALS,
        eos_cols=np.array(["rho", "mu", "nabla_ad", "S", "cp", "chi_rho", "chi_T", "lnPgas"][:eos_o.shape[1]]),
        mode="MODE-A", mesa="r26.04.1 shared", kap_prefix="gs98(default)", zbase="0.014",
    )
    print(f"[gen] wrote {args.out}: points={pts.shape} eos={eos_o.shape} kap={kap_o.shape} neu={neu_o.shape}")


if __name__ == "__main__":
    main()
