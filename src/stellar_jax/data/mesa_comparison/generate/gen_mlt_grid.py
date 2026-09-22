#!/usr/bin/env python3
"""Generate the MESA MLT reference point-set (issue #756, 4th module).

MLT (`set_mlt`) is a COUPLED convective calc (~9 inputs), not a pure (T,rho,X,Z) lookup, so a dense
grid is infeasible. Instead we sample a SELF-CONSISTENT convective point set: T, P, rho, mu,
nabla_ad from the MESA EOS shim and kappa from the MESA opacity shim (at (logT,logRho,X,Z) points),
then sweep g and nabla_rad>nabla_ad (convective). Mass is NOT an input (local physics only) → one
point set covers all masses.

Reference = `microphysics.mesa.mlt_adapter._mlt_callback` — the SAME numpy bridge the live test's
MESA side uses (chiT=chiRho=1, Cp=k_B/(mu m_H)/nad, Lambda=alpha*P/(rho g), gradL=nad → raw
call_mesa_mlt / set_mlt, Henyey option). Pure numpy → no jax dependency.

RUN (on a host with the built shims incl. libmlt_wrapper.so + shared MESA; see gen_module_grid.py):
    export MESA_DIR=$HOME/mesa LD_LIBRARY_PATH=$HOME/mesa/lib:$HOME/mesasdk/lib
    export STELLAR_MICROPHYSICS=mesa MESA_ZBASE=0.014
    python3 data/mesa_comparison/generate/gen_mlt_grid.py --out data/mesa_mlt_grid.npz

OUTPUT: mlt_inputs (N,9)=[nabla_rad,nad,T,P,rho,kappa,g,mu,alpha_mlt]; mlt_gradT (N,)=MESA gradT.
The per-module test (issue #756) evaluates our JAX transport.mlt.mlt_nabla at mlt_inputs and asserts
gradT agreement (~5% in convective zones), @mutation-gated on mlt_nabla_offset.
"""
from __future__ import annotations
import argparse, os
import numpy as np

LOGT = np.round(np.linspace(4.0, 7.0, 10), 4)
LOGRHO = np.round(np.linspace(-8.0, 1.0, 10), 4)
XVALS = [0.35, 0.70]
ZVALS = [0.014, 0.02]
GLOG = [3.0, 4.5]
NRAD_FACTORS = [1.5, 3.0, 10.0]
ALPHA = 2.0


def _arr(x): return np.atleast_1d(np.asarray(x, dtype=np.float64)).ravel()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/mesa_mlt_grid.npz")
    args = ap.parse_args()
    if os.environ.get("STELLAR_MICROPHYSICS") != "mesa":
        raise SystemExit("Set STELLAR_MICROPHYSICS=mesa.")
    from stellar_jax.microphysics.mesa.bindings import (call_mesa_eos, call_mesa_kap,
                                            _ensure_eos_init, _ensure_kap_init)
    from stellar_jax.microphysics.mesa.mlt_adapter import _mlt_callback
    _ensure_eos_init(); _ensure_kap_init()

    inp, grad, prov = [], [], []
    for lt in LOGT:
        for lr in LOGRHO:
            for x in XVALS:
                for z in ZVALS:
                    eos = _arr(call_mesa_eos(float(lt), float(lr), float(x), float(z)))
                    rho, mu, nad, lnPgas = eos[0], eos[1], eos[2], eos[7]
                    if not np.all(np.isfinite([rho, mu, nad, lnPgas])):
                        continue
                    P, T = float(np.exp(lnPgas)), 10.0 ** float(lt)
                    kappa = 10.0 ** _arr(call_mesa_kap(float(lt), float(lr), float(x), float(z)))[0]
                    for glog in GLOG:
                        g = 10.0 ** glog
                        for f in NRAD_FACTORS:
                            a = np.array([f * nad, nad, T, P, rho, kappa, g, mu, ALPHA], dtype=np.float64)
                            inp.append(a); grad.append(float(_mlt_callback(a))); prov.append((lt, lr, x, z, glog, f))

    inp, grad, prov = np.vstack(inp), np.asarray(grad), np.asarray(prov)
    np.savez_compressed(args.out, mlt_inputs=inp, mlt_gradT=grad, provenance=prov,
        input_cols=np.array(["nabla_rad", "nad", "T", "P", "rho", "kappa", "g", "mu", "alpha_mlt"]),
        prov_cols=np.array(["logT", "logRho", "X", "Z", "log_g", "nrad_factor"]),
        mode="MODE-A", mesa="r26.04.1 shared", mlt_option="Henyey")
    print(f"[gen-mlt] wrote {args.out}: inputs={inp.shape} gradT={grad.shape}")


if __name__ == "__main__":
    main()
