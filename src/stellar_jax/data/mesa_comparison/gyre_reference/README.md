# GYRE adiabatic oscillation reference frequencies

Reference p-mode frequencies computed with **GYRE** for the committed MODE-A MESA FGONG
profiles in `../profiles/`. This is the **converged reference oscillation code** used to
validate stellar-jax's own oscillation solver (the small separation δν₀₂ and ratios r02 —
the asteroseismic {M,age} age diagnostics).

## Why this exists
The repo previously had **no tight, reference-grade oscillation test** — δν₀₂ was only
checked against a ±60% physical-range window, and Δν against the observed Sun. That
looseness let a 5–40% δν₀₂ bias go undetected. These committed reference
frequencies let CI validate our solver's small separation to 1–2% without needing GYRE at
run time (GYRE is not required at run time).

## Contents
- `<mass>Msun/<stage>.txt` — per-model mode table, columns: `l n_pg freq_uHz`
  (l = degree, n_pg = radial order from GYRE's Scuflaire–Osaki classification, freq in µHz).
  **These raw per-mode frequencies are the authoritative artifact.**
- `summary.csv` — convenience derived quantities: `mass_Msun,stage,Dnu_uHz,dnu02_uHz,...`
  where Δν and δν₀₂ are computed by **matched radial order** over the clean p-mode regime.

## Generation (provenance)
- **Code:** GYRE 8.1 (Townsend & Teitler 2013), adiabatic.
- **Namelist:** `outer_bound='JCD'` (Christensen-Dalsgaard atmospheric boundary — the physical
  reference), `diff_scheme='COLLOC_GL6'` (6th-order Gauss-Legendre collocation),
  grid `w_osc=25, w_exp=3, w_ctr=20`, scan LINEAR 200–5500 µHz `n_freq=2500`, degrees l=0,1,2,3.
- **Input models:** the committed `../profiles/<mass>Msun/<stage>.FGONG.gz` (MODE-A physics:
  α_mlt=2.0, Z=0.014, Y=0.2695, gs98, Krishna-Swamy T(τ), diffusion/overshoot OFF).
- **Reproduce:** run `gyre` with the namelist above on each `profiles/*.FGONG.gz`
  (`summary_item_list='l,n_pg,freq'`, `summary_file_format='TXT'`).

## IMPORTANT caveat — p-mode regime only
δν₀₂ / the small separation is a clean diagnostic only in the **acoustic (p-mode) regime**
of the early–mid main sequence. Near TAMS and in evolved cores, l=1 and l=2 modes become
**mixed (p-g) modes**, and a naive "nearest l=2 below l=0" pairing yields non-physical
values. The `summary.csv` δν₀₂ is therefore computed by **matched radial order over p-modes
only** and omitted where mixed modes make it ill-defined. For any derived quantity, use the
raw per-mode `<stage>.txt` files and select modes by (l, n_pg) explicitly.
