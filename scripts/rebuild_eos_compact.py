#!/usr/bin/env python3
"""Rebuild eos_compact.npz with chi_rho_gas and chi_T_gas computed from the logPgas table.

The OPAL EOS tables natively store chi_rho and chi_T (see MESA eosdt_load_tables.f90,
jchiRho=4, jchiT=5), but our compact table only stored logPgas, mu, grad_ad. We compute
the gas-pressure thermodynamic derivatives from the logPgas table via numerical differentiation:

    chi_rho_gas = d(logPgas)/d(logQ)|_logT     [logQ = logRho - 2*logT + 12]
    chi_T_gas = d(logPgas)/d(logT)|_logQ - 2 * chi_rho_gas

These are the GAS pressure derivatives (d ln P_gas / d ln rho)_T and (d ln P_gas / d ln T)_rho.
At runtime, eos_lookup converts to total-pressure derivatives using the radiation correction:
    chi_rho_total = beta * chi_rho_gas
    chi_T_total = beta * chi_T_gas + 4*(1 - beta)

Reference:
    Rogers & Nayfonov 2002, ApJ 576, 1064 (OPAL EOS tables)
    MESA eos/private/eosdt_load_tables.f90 lines 32-33, 513-514
"""
import os
import numpy as np

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    inpath = os.path.join(data_dir, 'eos_compact.npz')
    outpath = inpath  # overwrite in place

    print(f"Loading {inpath}...")
    e = np.load(inpath)
    X_grid = e['X_grid']
    Z_grid = e['Z_grid']
    logT_grid = e['logT_grid']
    logQ_grid = e['logQ_grid']
    logPgas = e['logPgas']    # shape (5, 3, 78, 150), axes (X, Z, logQ, logT)
    mu = e['mu']
    grad_ad = e['grad_ad']

    print(f"  logPgas shape: {logPgas.shape}")
    print(f"  logQ_grid: [{logQ_grid[0]:.3f}, {logQ_grid[-1]:.3f}], n={len(logQ_grid)}")
    print(f"  logT_grid: [{logT_grid[0]:.3f}, {logT_grid[-1]:.3f}], n={len(logT_grid)}")

    # Compute gas-pressure thermodynamic derivatives via central finite differences.
    # np.gradient uses 2nd-order central differences in the interior and 1st-order
    # one-sided differences at the boundaries.
    #
    # axis 2 = logQ (at constant logT)
    # axis 3 = logT (at constant logQ)
    print("Computing chi_rho_gas = d(logPgas)/d(logQ)|_logT ...")
    chi_rho_gas = np.gradient(logPgas, logQ_grid, axis=2)

    print("Computing chi_T_gas = d(logPgas)/d(logT)|_rho ...")
    dP_dT_constQ = np.gradient(logPgas, logT_grid, axis=3)
    # At constant rho: logQ = logRho - 2*logT + 12, so d(logQ)/d(logT)|_rho = -2
    # Chain rule: d(logP)/d(logT)|_rho = d(logP)/d(logT)|_Q + d(logP)/d(logQ)|_T * (-2)
    chi_T_gas = dP_dT_constQ - 2.0 * chi_rho_gas

    # Sanity checks
    # For ideal gas (high T, fully ionized): chi_rho_gas -> 1, chi_T_gas -> 1
    # Check at logT = 7.0 (index ~120), mid logQ
    it_deep = np.argmin(np.abs(logT_grid - 7.0))
    iq_mid = len(logQ_grid) // 2
    print(f"\nSanity check at logT=7.0 (fully ionized, should be ~1):")
    print(f"  chi_rho_gas[X=0.6, Z=0.02, logQ_mid, logT=7.0] = {chi_rho_gas[3, 1, iq_mid, it_deep]:.4f}")
    print(f"  chi_T_gas[X=0.6, Z=0.02, logQ_mid, logT=7.0] = {chi_T_gas[3, 1, iq_mid, it_deep]:.4f}")

    # Check at logT = 4.5 (partial ionization zone, should deviate from 1)
    it_surf = np.argmin(np.abs(logT_grid - 4.5))
    print(f"\nPartial ionization zone (logT=4.5, should deviate from 1):")
    print(f"  chi_rho_gas[X=0.6, Z=0.02, logQ_mid, logT=4.5] = {chi_rho_gas[3, 1, iq_mid, it_surf]:.4f}")
    print(f"  chi_T_gas[X=0.6, Z=0.02, logQ_mid, logT=4.5] = {chi_T_gas[3, 1, iq_mid, it_surf]:.4f}")

    # Physical bounds enforcement: chi_rho_gas should be in (0, ~2), chi_T_gas in (~0.5, ~3)
    # The boundaries of the logQ and logT grids can have numerical artifacts from one-sided differences
    chi_rho_gas = np.clip(chi_rho_gas, 0.3, 2.0)
    chi_T_gas = np.clip(chi_T_gas, 0.3, 4.0)

    print(f"\n  chi_rho_gas range: [{chi_rho_gas.min():.4f}, {chi_rho_gas.max():.4f}]")
    print(f"  chi_T_gas range: [{chi_T_gas.min():.4f}, {chi_T_gas.max():.4f}]")

    # Save
    print(f"\nSaving to {outpath}...")
    np.savez(outpath,
             X_grid=X_grid,
             Z_grid=Z_grid,
             logT_grid=logT_grid,
             logQ_grid=logQ_grid,
             logPgas=logPgas,
             mu=mu,
             grad_ad=grad_ad,
             chi_rho_gas=chi_rho_gas.astype(np.float64),
             chi_T_gas=chi_T_gas.astype(np.float64))
    print("Done.")

    # Verify the file
    check = np.load(outpath)
    print(f"\nVerification - keys in new file: {sorted(check.files)}")
    print(f"  chi_rho_gas shape: {check['chi_rho_gas'].shape}")
    print(f"  chi_T_gas shape: {check['chi_T_gas'].shape}")


if __name__ == '__main__':
    main()
