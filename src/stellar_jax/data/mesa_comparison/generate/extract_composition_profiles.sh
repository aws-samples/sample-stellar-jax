#!/usr/bin/env bash
# Extract full-composition profiles from saved .mod files.
# Outputs to mesa_results_dense_compositions/ (does NOT touch existing data).
set -eo pipefail
export MANPATH=${MANPATH:-}
export MESASDK_ROOT=$HOME/mesasdk
source $HOME/mesasdk/bin/mesasdk_init.sh
export MESA_DIR=$HOME/mesa
export OMP_NUM_THREADS=4

cd $HOME/mesa/star/work || exit 1
STAR_BIN=$HOME/mesa/star/work/build/bin/star
OUT=$HOME/mesa_results_dense_compositions
mkdir -p "$OUT"

# Create profile_columns file with all network species
cat > profile_columns_full.list << "COLS"
zone
mass
logR
logT
logRho
logP
luminosity
eps_nuc
non_nuc_neu
h1
h2
he3
he4
li7
be7
b8
c12
c13
n13
n14
n15
o14
o15
o16
o17
o18
f17
f18
f19
ne18
ne19
ne20
ne22
mg24
mg22
COLS

for M in 1.0 1.2 1.5 2.0; do
  MOD="$HOME/mesa_results_dense/${M}Msun/final_${M}.mod"
  if [ ! -f "$MOD" ]; then
    echo "SKIP ${M}Msun — no .mod file"
    continue
  fi
  echo "=== Extracting ${M}Msun $(date -u) ==="
  mkdir -p "$OUT/${M}Msun"

  cat > inlist_project << INLIST_EOF
&star_job
  show_log_description_at_start = .false.
  load_saved_model = .true.
  load_model_filename = 
  set_initial_age = .false.
  set_initial_model_number = .false.
  save_model_when_terminate = .false.
/
&eos
/
&kap
  Zbase = 0.014
  kap_file_prefix = "gs98"
/
&controls
  initial_mass = ${M}
  initial_z = 0.014
  initial_y = 0.2695
  MLT_option = "Cox"
  mixing_length_alpha = 2.0
  default_net_name = "pp_cno_extras_o18_ne22.net"
  atm_option = "T_tau"
  atm_T_tau_relation = "Krishna_Swamy"
  atm_T_tau_opacity = "varying"
  do_element_diffusion = .false.
  max_model_number = 2
  profile_interval = 1
  write_pulse_data_with_profile = .true.
  pulse_data_format = "FGONG"
  add_atmosphere_to_pulse_data = .true.
/
INLIST_EOF

  rm -rf LOGS photos
  mkdir -p LOGS
  $STAR_BIN 1>>"$OUT/${M}Msun/mesa_log.txt" 2>&1
  RC=$?
  
  if [ -f LOGS/profile1.data ]; then
    cp LOGS/profile1.data "$OUT/${M}Msun/profile_full_composition.data"
    echo "  Profile saved (rc=$RC)"
  else
    echo "  NO PROFILE (rc=$RC)"
  fi
  if [ -f LOGS/profile1.data.FGONG ]; then
    cp LOGS/profile1.data.FGONG "$OUT/${M}Msun/profile_full_composition.FGONG"
  fi
done
echo "=== DONE $(date -u) ==="
