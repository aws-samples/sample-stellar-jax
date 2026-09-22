# MESA per-zone structure references — dense MODE-A FGONG library

Per-zone **FGONG** stellar-structure snapshots spanning ZAMS → SGB for M ∈ {1.0, 1.2, 1.5,
2.0} M☉, used for **component-level** validation of stellar-jax against MESA.
A loaded FGONG is a **real external MODE-A reference**
(identical-physics) — never synthetic. **MODE A only**: use with `MESA_CONFIG`; never
cross-compare with Model S (which is MODE B).

> **Single provenance (2026-06-22):** every file in this directory was re-derived from **one**
> dense MESA run per mass. This replaces
> the earlier library, which mixed coarsely-sampled snapshots from the RGB run. There is now
> exactly one inlist and one recipe behind every file here, so the whole set is independently
> reproducible without ambiguity. The companion `../rgb/<mass>/tip.FGONG.gz` (RGB-tip) files come
> from the RGB run but use the **same MESA build and identical physics** (only the output cadence
> differs), so they remain mutually consistent with this library.

## Provenance

| | |
|---|---|
| Code | **MESA git build `f12c70cf`** (reports version `r26.4.1`; same build as `../rgb/`; *not* the `r26.04.1` release used for `../results/`) |
| SDK | `mesasdk-x86_64-linux-26.6.1` |
| Driver | a shell driver that writes the inlist below and sweeps `initial_mass` over {1.0, 1.2, 1.5, 2.0} |
| Output | `LOGS/profile<N>.data.FGONG` (FGONG pulse data, `iconst=15`, `ivar=40`, atmosphere appended) |
| Generated | 2026-06-21 |

## Physics — identical-physics MODE A

α<sub>MLT</sub> = 2.0 (Cox MLT), Z = 0.014, Y = 0.2695 (= 0.2485 + 1.5·Z), gs98 opacities,
Krishna-Swamy T(τ) atmosphere (varying opacity), PP+CNO net (`pp_cno_extras_o18_ne22.net`),
**no** element diffusion, **no** convective overshoot, **no** rotation. Evolved from a
pre-main-sequence model through the main sequence, subgiant branch, and on to the RGB tip
(terminated by `power_he_burn_upper_limit = 1d3`). This matches the physics stellar-jax targets
in MODE A; residuals are expected only from mesh/timestepping/atmosphere-integration details.

## Exact inlist (per run; `initial_mass` is swept over 1.0, 1.2, 1.5, 2.0)

```fortran
&star_job
  show_log_description_at_start = .false.
  create_pre_main_sequence_model = .true.
  save_model_when_terminate = .true.
  save_model_filename = "final_<M>.mod"
/
&eos
/
&kap
  Zbase = 0.014
  kap_file_prefix = "gs98"
/
&controls
  initial_mass = <M>            ! 1.0, 1.2, 1.5, 2.0
  initial_z = 0.014
  initial_y = 0.2695
  MLT_option = "Cox"
  mixing_length_alpha = 2.0
  default_net_name = "pp_cno_extras_o18_ne22.net"
  atm_option = "T_tau"
  atm_T_tau_relation = "Krishna_Swamy"
  atm_T_tau_opacity = "varying"
  do_element_diffusion = .false.
  power_he_burn_upper_limit = 1d3   ! terminate at the RGB tip, pre-flash
  max_model_number = 30000
  varcontrol_target = 1d-3
  history_interval = 1
  profile_interval = 10             ! dense output (RGB run used 50)
  max_num_profile_models = 2000
  write_pulse_data_with_profile = .true.
  pulse_data_format = "FGONG"
  add_atmosphere_to_pulse_data = .true.
/
```

## How the snapshots were selected (X_c → FGONG)

For each mass, `profiles.index` maps `model_number → profile_number` and `history.data`
gives the central hydrogen fraction `center_h1` (X_c) per model; FGONGs are
`profile<profile_number>.data.FGONG`. Snapshots are picked by X_c (nearest available profile;
near-duplicates de-duplicated against the named phase points):

- **`zams`** — the earliest post-ignition main-sequence snapshot (target X_c = X_init − 0.003 ≈
  0.7135, the standard ZAMS definition; the dense output cadence lands it at X_c ≈ 0.70). This is
  a true H-burning ZAMS model — **not** the pre-main-sequence initial model.
- **`midMS`** — mid-main-sequence (mass-dependent target: 0.511 / 0.542 / 0.607 / 0.662). This is
  the file the EOS-vs-MESA test pins, hence the stable name + X_c.
- **`Xc0.NN`** — uniform main-sequence grid at X_c ≈ {0.60, 0.50, 0.40, 0.30, 0.20, 0.10}
  (grid points coinciding with `zams`/`midMS` are omitted).
- **`TAMS`** — terminal-age main sequence (X_c nearest 0.01).
- **`SGB`** — subgiant branch: the luminosity minimum after central-H exhaustion (base of the
  giant branch). For 2.0 M☉ this is the distinct post-Hertzsprung-gap subgiant that the earlier
  library lacked.

## How to reproduce independently

```bash
# On a host with MESA build f12c70cf + mesasdk 26.6.1:
export MESASDK_ROOT=$HOME/mesasdk && source $MESASDK_ROOT/bin/mesasdk_init.sh
export MESA_DIR=$HOME/mesa && export OMP_NUM_THREADS=8
cd $MESA_DIR/star/work
# write the inlist above to inlist_project (set initial_mass + save_model_filename per mass), then:
rm -rf LOGS photos && ./rn
# FGONGs land in LOGS/profile<N>.data.FGONG; history in LOGS/history.data, index in LOGS/profiles.index.
# Select by X_c as described above (cross-reference profiles.index ↔ history.data center_h1),
# gzip the chosen profile<N>.data.FGONG to <label>.FGONG.gz.
```

## Contents (sorted by X_c; ages from `history.data`)

### 1.0 M☉
| file | X_c | age (Gyr) | log L | log Teff | log R |
|------|-----|-----------|-------|----------|-------|
| `zams.FGONG.gz`  | 0.6985 | 0.261 | −0.0645 | 3.7660 | −0.0417 |
| `Xc0.60.FGONG.gz`| 0.6121 | 1.263 | −0.0306 | 3.7687 | −0.0300 |
| `midMS.FGONG.gz` | 0.5107 | 2.391 |  0.0086 | 3.7715 | −0.0161 |
| `Xc0.40.FGONG.gz`| 0.4093 | 3.463 |  0.0488 | 3.7739 | −0.0008 |
| `Xc0.30.FGONG.gz`| 0.3078 | 4.475 |  0.0900 | 3.7759 |  0.0159 |
| `Xc0.20.FGONG.gz`| 0.2069 | 5.436 |  0.1326 | 3.7772 |  0.0345 |
| `Xc0.10.FGONG.gz`| 0.1038 | 6.348 |  0.1762 | 3.7778 |  0.0552 |
| `TAMS.FGONG.gz`  | 0.0073 | 7.292 |  0.2243 | 3.7769 |  0.0810 |
| `SGB.FGONG.gz`   | 0.0001 | 8.124 |  0.2809 | 3.7752 |  0.1127 |

### 1.2 M☉
| file | X_c | age (Gyr) | log L | log Teff | log R |
|------|-----|-----------|-------|----------|-------|
| `zams.FGONG.gz`  | 0.7074 | 0.086 | 0.3085 | 3.8086 | 0.0598 |
| `Xc0.60.FGONG.gz`| 0.6522 | 0.461 | 0.3321 | 3.8098 | 0.0690 |
| `midMS.FGONG.gz` | 0.5416 | 1.135 | 0.3726 | 3.8114 | 0.0862 |
| `Xc0.40.FGONG.gz`| 0.4377 | 1.670 | 0.4055 | 3.8119 | 0.1017 |
| `Xc0.30.FGONG.gz`| 0.3380 | 2.180 | 0.4359 | 3.8112 | 0.1182 |
| `Xc0.20.FGONG.gz`| 0.1660 | 3.206 | 0.4819 | 3.8022 | 0.1592 |
| `Xc0.10.FGONG.gz`| 0.0698 | 3.667 | 0.4998 | 3.7953 | 0.1820 |
| `TAMS.FGONG.gz`  | 0.0116 | 3.830 | 0.5423 | 3.7998 | 0.1942 |
| `SGB.FGONG.gz`   | 0.0000 | 4.988 | 0.6016 | 3.7213 | 0.3808 |

### 1.5 M☉
| file | X_c | age (Gyr) | log L | log Teff | log R |
|------|-----|-----------|-------|----------|-------|
| `zams.FGONG.gz`  | 0.6963 | 0.094 | 0.7529 | 3.8777 | 0.1438 |
| `midMS.FGONG.gz` | 0.6072 | 0.434 | 0.7846 | 3.8762 | 0.1625 |
| `Xc0.50.FGONG.gz`| 0.4966 | 0.820 | 0.8165 | 3.8686 | 0.1936 |
| `Xc0.40.FGONG.gz`| 0.3895 | 1.159 | 0.8421 | 3.8564 | 0.2310 |
| `Xc0.30.FGONG.gz`| 0.2761 | 1.438 | 0.8599 | 3.8414 | 0.2698 |
| `Xc0.20.FGONG.gz`| 0.1623 | 1.669 | 0.8711 | 3.8267 | 0.3048 |
| `Xc0.10.FGONG.gz`| 0.0558 | 1.847 | 0.8833 | 3.8167 | 0.3309 |
| `TAMS.FGONG.gz`  | 0.0083 | 1.912 | 0.9271 | 3.8266 | 0.3331 |
| `SGB.FGONG.gz`   | 0.0000 | 2.208 | 0.8255 | 3.7272 | 0.4810 |

### 2.0 M☉
| file | X_c | age (Gyr) | log L | log Teff | log R |
|------|-----|-----------|-------|----------|-------|
| `zams.FGONG.gz`  | 0.7080 | 0.023 | 1.2616 | 3.9835 | 0.1865 |
| `midMS.FGONG.gz` | 0.6619 | 0.116 | 1.2753 | 3.9777 | 0.2049 |
| `Xc0.60.FGONG.gz`| 0.5481 | 0.312 | 1.3082 | 3.9633 | 0.2501 |
| `Xc0.40.FGONG.gz`| 0.4438 | 0.461 | 1.3359 | 3.9496 | 0.2915 |
| `Xc0.30.FGONG.gz`| 0.3392 | 0.585 | 1.3586 | 3.9336 | 0.3347 |
| `Xc0.20.FGONG.gz`| 0.2343 | 0.688 | 1.3751 | 3.9149 | 0.3804 |
| `Xc0.10.FGONG.gz`| 0.1295 | 0.773 | 1.3859 | 3.8939 | 0.4279 |
| `TAMS.FGONG.gz`  | 0.0046 | 0.852 | 1.4522 | 3.9082 | 0.4324 |
| `SGB.FGONG.gz`   | 0.0000 | 0.913 | 1.2600 | 3.7122 | 0.7282 |

## FGONG columns (ivar=40, 0-indexed per-zone `var[:,j]`)

r=0, ln(m/M)=1, **T=2**, P=3, ρ=4, **X=5**, **L=6**, **κ=7**, **ε_nuc=8**, Γ1=9, … Brunt A*=14.
`oscillations.read_fgong` + `oscillations.fgong_components` expose r/T/P/ρ/X/L/κ/ε_nuc/Γ1/A*.
A loaded FGONG is an **external** reference → anti-theater compliant. `ε_nuc` is correctly ~0 in
the envelope; verify it is non-zero in-core before relying on it.

## Consumers

- `tests/test_fgong.py::test_fgong_reference_library_loads` — globs this library + the RGB tips,
  asserts every file loads with physical per-zone T/ρ/P/X/κ/ε_nuc (structural sanity).
- `tests/test_microphysics.py::test_eos_vs_mesa_ms_fgong` — pins `1.0Msun/midMS.FGONG.gz` (X_c≈0.51) to
  validate the production OPAL EOS density pointwise against MESA (MODE A).
