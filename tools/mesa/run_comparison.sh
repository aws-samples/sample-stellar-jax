#!/bin/bash
# Run MESA comparison tracks for stellar-jax validation.
# Run on a host with MESA installed.
#
# Usage: bash tools/mesa/run_comparison.sh
#
# Produces: results/{mass}/history.data for each mass

set -e

export MESASDK_ROOT=$HOME/mesasdk
source $MESASDK_ROOT/bin/mesasdk_init.sh
export MESA_DIR=$HOME/mesa
export OMP_NUM_THREADS=8

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

# Compile the work directory
cd $MESA_DIR/star/work
./mk

for MASS in 1.0 1.2 1.5 2.0; do
    echo "=== Running M=${MASS} Msun ==="
    
    # Set up inlist
    cp "$SCRIPT_DIR/inlist_comparison" inlist_project
    sed -i "s/initial_mass = 1.0/initial_mass = ${MASS}/" inlist_project
    
    # Clean and run
    rm -rf LOGS photos
    ./rn
    
    # Save results
    mkdir -p "$RESULTS_DIR/${MASS}Msun"
    cp LOGS/history.data "$RESULTS_DIR/${MASS}Msun/"
    echo "  Done: $RESULTS_DIR/${MASS}Msun/history.data"
done

echo ""
echo "=== All runs complete ==="
echo "Results in: $RESULTS_DIR/"
echo "Key columns in history.data: star_age, log_L, log_Teff, log_R, center_h1"
