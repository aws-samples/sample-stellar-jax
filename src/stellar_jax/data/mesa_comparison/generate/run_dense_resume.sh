#!/usr/bin/env bash
# Denser-sampling variant of run_rgb.sh: profile_interval 50->10 to populate the MS + SGB
# (clean mid-MS Xc~0.35, a real 2.0 Msun SGB, denser giant-branch). IDENTICAL MODE-A physics
# as the committed RGB tracks. Writes to a SEPARATE dir (mesa_results_dense) — does NOT touch
# mesa_results_rgb (the source of the committed FGONGs).
set -o pipefail
export MANPATH=${MANPATH:-}
export MESASDK_ROOT=$HOME/mesasdk
source $HOME/mesasdk/bin/mesasdk_init.sh
export MESA_DIR=$HOME/mesa
export OMP_NUM_THREADS=8
cd $HOME/mesa/star/work || exit 1
OUT=$HOME/mesa_results_dense
mkdir -p "$OUT"
for M in 2.0 1.2 1.5; do
  echo "=== START ${M} Msun $(date -u) ==="
  cat > inlist_project << INLIST_EOF
&star_job
  show_log_description_at_start = .false.
  create_pre_main_sequence_model = .true.
  save_model_when_terminate = .true.
  save_model_filename = "final_${M}.mod"
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
  power_he_burn_upper_limit = 1d3
  max_model_number = 30000
  varcontrol_target = 1d-3
  history_interval = 1
  profile_interval = 10
  max_num_profile_models = 2000
  write_pulse_data_with_profile = .true.
  pulse_data_format = "FGONG"
  add_atmosphere_to_pulse_data = .true.
/
INLIST_EOF
  rm -rf LOGS photos
  ./rn
  RC=$?
  mkdir -p "$OUT/${M}Msun"
  cp -r LOGS/* "$OUT/${M}Msun/" 2>/dev/null
  cp "final_${M}.mod" "$OUT/${M}Msun/" 2>/dev/null
  ROWS=$(grep -c "" LOGS/history.data 2>/dev/null)
  echo "=== DONE ${M} Msun rc=$RC rows=$ROWS $(date -u) ==="
done
echo "=== ALL DENSE RUNS COMPLETE $(date -u) ==="
