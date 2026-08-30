#!/bin/bash
set -e

# Target directory in native Linux filesystem
SIM_DIR="$HOME/simulations/fowt-mooring-twin/openfast"
mkdir -p "$SIM_DIR"
cd "$SIM_DIR"

if [ ! -d "r-test" ]; then
    echo "Cloning OpenFAST r-test repository..."
    git clone https://github.com/OpenFAST/r-test.git
    cd r-test
    git checkout v3.5.3  # Match the conda openfast version
else
    echo "r-test repository already exists."
fi

# The working directory for the OC4 semi-submersible model is:
BASE_MODEL_DIR="$SIM_DIR/r-test/glue-codes/openfast/5MW_OC4Semi_WSt_WavesWN"

echo "========================================="
echo "Model downloaded."
echo "Base model directory is: $BASE_MODEL_DIR"
echo "========================================="
