#!/bin/bash

# Environment setup script for distributed training
# This script installs and configures the necessary dependencies

set -e

echo "================================="
echo "Setting up Distributed Training Environment"
echo "================================="

# ================================
# Check Python and pip
# ================================

echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found. Please install Python 3.7+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 not found. Please install pip first."
    exit 1
fi

# ================================
# Install Core Dependencies
# ================================

echo ""
echo "Installing core dependencies..."

# Create requirements file for distributed training
cat > requirements_distributed.txt << EOF
# Core PyTorch and ML libraries
torch>=2.0.0
torchvision
torchaudio

# HuggingFace Accelerate for distributed training
accelerate>=0.20.0

# Data processing and scientific computing
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Progress bars and utilities
tqdm>=4.62.0

# Logging and experiment tracking
wandb>=0.15.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Additional utilities
psutil  # For system monitoring
gpustat  # For GPU monitoring
EOF

echo "Installing Python packages..."
pip3 install -r requirements_distributed.txt

# ================================
# Verify Installation
# ================================

echo ""
echo "Verifying installation..."

# Check PyTorch installation
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('No CUDA GPUs detected - will use CPU training')
"

# Check Accelerate installation
python3 -c "
try:
    import accelerate
    print(f'Accelerate version: {accelerate.__version__}')
    print('Accelerate installed successfully')
except ImportError:
    print('Error: Accelerate not installed properly')
    exit(1)
"

# Check other dependencies
python3 -c "
dependencies = ['numpy', 'pandas', 'sklearn', 'tqdm', 'matplotlib', 'seaborn']
missing = []
for dep in dependencies:
    try:
        __import__(dep)
        print(f'✓ {dep}')
    except ImportError:
        print(f'✗ {dep}')
        missing.append(dep)

if missing:
    print(f'Missing dependencies: {missing}')
    exit(1)
else:
    print('All core dependencies installed successfully')
"

# ================================
# Configure Accelerate
# ================================

echo ""
echo "Configuring Accelerate..."

# Create a basic single-node configuration
mkdir -p ~/.cache/huggingface/accelerate/

cat > ~/.cache/huggingface/accelerate/default_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: $(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 1)")
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "Default Accelerate configuration created at ~/.cache/huggingface/accelerate/default_config.yaml"

# ================================
# Test Configuration
# ================================

echo ""
echo "Testing Accelerate configuration..."

# Create a simple test script
cat > test_accelerate.py << 'EOF'
import torch
from accelerate import Accelerator

def main():
    accelerator = Accelerator()
    
    print(f"Process {accelerator.process_index} of {accelerator.num_processes}")
    print(f"Device: {accelerator.device}")
    print(f"Mixed precision: {accelerator.mixed_precision}")
    
    # Test a simple tensor operation
    x = torch.randn(10, 10)
    x = x.to(accelerator.device)
    y = torch.matmul(x, x.T)
    
    print(f"Test tensor computation successful on {accelerator.device}")
    print("Accelerate setup is working correctly!")

if __name__ == "__main__":
    main()
EOF

echo "Running Accelerate test..."
accelerate launch test_accelerate.py

# Clean up test file
rm test_accelerate.py

# ================================
# System Information
# ================================

echo ""
echo "================================="
echo "System Information"
echo "================================="

# CPU info
echo "CPU Information:"
python3 -c "
import psutil
import os
print(f'  CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical')
print(f'  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB')
"

# GPU info (if available)
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU: %s (%.1f GB, Driver: %s)\n", $1, $2/1024, $3}'
fi

# Network info for multi-node setup
echo ""
echo "Network Information (for multi-node setup):"
echo "  Hostname: $(hostname)"
echo "  IP Address: $(hostname -I | awk '{print $1}')"

# ================================
# Usage Instructions
# ================================

echo ""
echo "================================="
echo "Setup Complete!"
echo "================================="
echo ""
echo "You can now run distributed training using:"
echo ""
echo "1. Single-node multi-GPU training:"
echo "   ./run_accelerated_training.sh"
echo ""
echo "2. Multi-node training:"
echo "   ./run_multinode_training.sh"
echo ""
echo "3. Custom configuration:"
echo "   accelerate config  # Interactive configuration"
echo "   accelerate launch your_script.py"
echo ""
echo "For more information about Accelerate:"
echo "   https://huggingface.co/docs/accelerate/"
echo ""
echo "Environment setup completed successfully!"

# Clean up
rm requirements_distributed.txt
