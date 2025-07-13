#!/bin/bash

# Minimal setup script for Power Grid LSTM Environment
# Creates conda environment and installs essential packages

set -e

echo "Creating conda environment 'power-grid'..."
conda create -n power-grid python=3.11 -y

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate power-grid

echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio -c pytorch -y

echo "Installing core packages..."
conda install numpy pandas scikit-learn matplotlib seaborn -y
pip install tqdm wandb

echo "Done! Activate with: conda activate power-grid"
