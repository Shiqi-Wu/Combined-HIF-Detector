#!/bin/bash

# Multi-node distributed training launcher for LSTM classifier using Accelerate
# This script uses HuggingFace Accelerate for simplified distributed training

set -e  # Exit on any error

# ================================
# Configuration - All parameters explicitly defined
# ================================

# Data parameters
DATA_DIR="data"
SAMPLE_STEP=1
WINDOW_SIZE=30
BATCH_SIZE=32  # Per device batch size

# Model parameters
HIDDEN_SIZE=128
NUM_LAYERS=2
DROPOUT=0.2
BIDIRECTIONAL=true

# Training parameters
NUM_EPOCHS=100
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-4
PATIENCE=20
GRAD_CLIP=1.0
GRADIENT_ACCUMULATION_STEPS=2
MIXED_PRECISION="fp16"  # Options: no, fp16, bf16
SEED=42

# Logging parameters
USE_WANDB=false  # Set to true to use wandb
WANDB_PROJECT="accelerated-lstm-training"
WANDB_ENTITY=""
EXPERIMENT_NAME="accelerated_lstm_$(date +%Y%m%d_%H%M%S)"

# Output parameters
RESULTS_DIR="./results_accelerated"

# Accelerate configuration
ACCELERATE_CONFIG_FILE="./accelerate_config.yaml"

# Script path
SCRIPT_PATH="./src/trainers/distributed_trainer.py"

# ================================
# Display Configuration
# ================================

echo "================================="
echo "Accelerated LSTM Training Script"
echo "================================="
echo "Data Configuration:"
echo "  Data Directory: $DATA_DIR"
echo "  Sample Step: $SAMPLE_STEP"
echo "  Window Size: $WINDOW_SIZE"
echo "  Batch Size (per device): $BATCH_SIZE"
echo ""
echo "Model Configuration:"
echo "  Hidden Size: $HIDDEN_SIZE"
echo "  Number of Layers: $NUM_LAYERS"
echo "  Dropout: $DROPOUT"
echo "  Bidirectional: $BIDIRECTIONAL"
echo ""
echo "Training Configuration:"
echo "  Number of Epochs: $NUM_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Patience: $PATIENCE"
echo "  Gradient Clipping: $GRAD_CLIP"
echo "  Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  Mixed Precision: $MIXED_PRECISION"
echo "  Random Seed: $SEED"
echo ""
echo "Logging Configuration:"
echo "  Use Wandb: $USE_WANDB"
if [ "$USE_WANDB" = true ]; then
    echo "  Wandb Project: $WANDB_PROJECT"
    echo "  Experiment Name: $EXPERIMENT_NAME"
else
    echo "  Training logs will be saved to local files in results directory"
fi
echo ""
echo "Output Configuration:"
echo "  Results Directory: $RESULTS_DIR"
echo "  Accelerate Config: $ACCELERATE_CONFIG_FILE"
echo "================================="

# ================================
# Environment Setup
# ================================

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check if accelerate is installed
if ! command -v accelerate &> /dev/null; then
    echo "Error: accelerate command not found. Please install with:"
    echo "pip install accelerate"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Training script not found at $SCRIPT_PATH"
    exit 1
fi

# ================================
# Generate Accelerate Configuration
# ================================

echo "Generating Accelerate configuration..."

# Create accelerate config if it doesn't exist
if [ ! -f "$ACCELERATE_CONFIG_FILE" ]; then
    cat > "$ACCELERATE_CONFIG_FILE" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: $MIXED_PRECISION
num_machines: 1
num_processes: $(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 1)")
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
    
    echo "Created Accelerate configuration at $ACCELERATE_CONFIG_FILE"
    echo "Configuration details:"
    cat "$ACCELERATE_CONFIG_FILE"
    echo ""
else
    echo "Using existing Accelerate configuration at $ACCELERATE_CONFIG_FILE"
fi

# ================================
# Build Arguments
# ================================

ARGS="--data_dir $DATA_DIR"
ARGS="$ARGS --sample_step $SAMPLE_STEP"
ARGS="$ARGS --window_size $WINDOW_SIZE"
ARGS="$ARGS --batch_size $BATCH_SIZE"
ARGS="$ARGS --hidden_size $HIDDEN_SIZE"
ARGS="$ARGS --num_layers $NUM_LAYERS"
ARGS="$ARGS --dropout $DROPOUT"
ARGS="$ARGS --num_epochs $NUM_EPOCHS"
ARGS="$ARGS --learning_rate $LEARNING_RATE"
ARGS="$ARGS --weight_decay $WEIGHT_DECAY"
ARGS="$ARGS --patience $PATIENCE"
ARGS="$ARGS --grad_clip $GRAD_CLIP"
ARGS="$ARGS --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"
ARGS="$ARGS --mixed_precision $MIXED_PRECISION"
ARGS="$ARGS --seed $SEED"
ARGS="$ARGS --results_dir $RESULTS_DIR"
ARGS="$ARGS --wandb_project $WANDB_PROJECT"

# Add conditional arguments
if [ "$BIDIRECTIONAL" = true ]; then
    ARGS="$ARGS --bidirectional"
fi

if [ "$USE_WANDB" = true ]; then
    ARGS="$ARGS --use_wandb"
fi

if [ -n "$WANDB_ENTITY" ]; then
    ARGS="$ARGS --wandb_entity $WANDB_ENTITY"
fi

# ================================
# Execute Training
# ================================

echo "Starting accelerated training..."
echo "Command: accelerate launch --config_file $ACCELERATE_CONFIG_FILE $SCRIPT_PATH $ARGS"
echo "================================="

# Launch training with accelerate
accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" "$SCRIPT_PATH" $ARGS

TRAINING_EXIT_CODE=$?

echo "================================="
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results saved in: $RESULTS_DIR"
    
    # Display some results if available
    if [ -f "$RESULTS_DIR/training_history.json" ]; then
        echo ""
        echo "Training Summary:"
        echo "Latest training history available at: $RESULTS_DIR/training_history.json"
    fi
    
    if [ -f "$RESULTS_DIR/best_model.pth" ]; then
        echo "Best model saved at: $RESULTS_DIR/best_model.pth"
    fi
    
    if [ -f "$RESULTS_DIR/performance_metrics.json" ]; then
        echo "Performance metrics available at: $RESULTS_DIR/performance_metrics.json"
    fi
else
    echo "Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "Check the error messages above for details."
fi
echo "================================="

exit $TRAINING_EXIT_CODE
