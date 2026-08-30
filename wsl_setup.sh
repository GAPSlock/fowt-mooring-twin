#!/bin/bash
set -e

cd $HOME

if [ ! -d "$HOME/miniforge3" ]; then
    echo "Downloading and installing Miniforge3..."
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o Miniforge3.sh
    bash Miniforge3.sh -b -p $HOME/miniforge3
    rm Miniforge3.sh
else
    echo "Miniforge3 is already installed."
fi

# Source conda
source $HOME/miniforge3/bin/activate
conda init bash

# Check if fowt environment exists, if not create it
if ! conda info --envs | grep -q "^fowt "; then
    echo "Creating 'fowt' conda environment..."
    conda create -n fowt python=3.10 -c conda-forge -y
else
    echo "'fowt' environment already exists."
fi

# Activate environment
source $HOME/miniforge3/bin/activate fowt

echo "Installing OpenFAST..."
conda install -c conda-forge openfast -y

echo "Installing Python dependencies (PyTorch, openfast-toolbox, fatpack, etc.)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install openfast-toolbox fatpack numpy pandas matplotlib scipy h5py pyyaml plotly dash dash-bootstrap-components cdsapi

echo "Setting up native Linux simulation directories..."
mkdir -p $HOME/simulations/fowt-mooring-twin/openfast/base_model

echo "========================================="
echo "Setup Complete!"
echo "OpenFAST version:"
openfast -v
echo "========================================="
