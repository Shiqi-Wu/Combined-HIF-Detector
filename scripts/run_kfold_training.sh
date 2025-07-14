#!/bin/bash

# K-Fold cross-validation training launcher for LSTM classifier using Accelerate
# This script uses HuggingFace Accelerate for distributed K-fold training

set -e  # Exit on any error

# ================================
# Configuration - All parameters explicitly defined
# ================================

# Data parameters
DATA_DIR="data"
SAMPLE_STEP=1
WINDOW_SIZE=1000
BATCH_SIZE=20  # Per device batch size
PCA_DIM=2

# Model parameters
HIDDEN_SIZE=128
NUM_LAYERS=2
DROPOUT=0.2
BIDIRECTIONAL=true

# Training parameters
NUM_EPOCHS=1000
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-4
PATIENCE=200
GRAD_CLIP=1.0
GRADIENT_ACCUMULATION_STEPS=2
SEED=42

# K-Fold parameters
K_FOLDS=5

# Logging parameters
USE_WANDB=true  # Set to true to use wandb
WANDB_PROJECT="kfold-lstm-training"
WANDB_ENTITY=""
EXPERIMENT_NAME="kfold_lstm_$(date +%Y%m%d_%H%M%S)"

# Output parameters
RESULTS_DIR="./results/results_kfold_2gpu_8192_1000"

# Accelerate configuration
ACCELERATE_CONFIG_FILE="./configs/accelerate_kfold_config.yaml"

# Script path
SCRIPT_PATH="./src/trainers/distributed_trainer.py"

# ================================
# Display Configuration
# ================================

echo "================================="
echo "K-Fold LSTM Training Script with Accelerate"
echo "================================="
echo "Data Configuration:"
echo "  Data Directory: $DATA_DIR"
echo "  Sample Step: $SAMPLE_STEP"
echo "  Window Size: $WINDOW_SIZE"
echo "  Batch Size (per device): $BATCH_SIZE"
echo "  PCA Dimension: $PCA_DIM"
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
echo "  Random Seed: $SEED"
echo ""
echo "K-Fold Configuration:"
echo "  Number of Folds: $K_FOLDS"
echo ""
echo "Logging Configuration:"
echo "  Use Wandb: $USE_WANDB"
if [ "$USE_WANDB" = true ]; then
    echo "  Wandb Project: $WANDB_PROJECT"
    echo "  Experiment Name: $EXPERIMENT_NAME"
else
    echo "  Training logs will be saved to local files in results directory"
    echo "  Individual fold logs: training_history_fold_X.json"
    echo "  Individual fold models: best_model_fold_X.pth"
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

echo "Generating Accelerate configuration for K-Fold training..."

# Create accelerate config if it doesn't exist
if [ ! -f "$ACCELERATE_CONFIG_FILE" ]; then
    cat > "$ACCELERATE_CONFIG_FILE" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: no
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
ARGS="$ARGS --pca_dim $PCA_DIM"
ARGS="$ARGS --hidden_size $HIDDEN_SIZE"
ARGS="$ARGS --num_layers $NUM_LAYERS"
ARGS="$ARGS --dropout $DROPOUT"
ARGS="$ARGS --num_epochs $NUM_EPOCHS"
ARGS="$ARGS --learning_rate $LEARNING_RATE"
ARGS="$ARGS --weight_decay $WEIGHT_DECAY"
ARGS="$ARGS --patience $PATIENCE"
ARGS="$ARGS --grad_clip $GRAD_CLIP"
ARGS="$ARGS --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"
ARGS="$ARGS --seed $SEED"
ARGS="$ARGS --k_folds $K_FOLDS"
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
# Execute K-Fold Training
# ================================

echo "Starting K-Fold training with accelerate..."
echo "Command: accelerate launch --config_file $ACCELERATE_CONFIG_FILE $SCRIPT_PATH $ARGS"
echo "================================="

# Launch K-fold training with accelerate
accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" "$SCRIPT_PATH" $ARGS

TRAINING_EXIT_CODE=$?

echo "================================="
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "K-Fold training completed successfully!"
    echo "Results saved in: $RESULTS_DIR"
    
    # Display results summary if available
    echo ""
    echo "K-Fold Results Summary:"
    echo "======================="
    
    # Check for individual fold results
    for ((i=1; i<=K_FOLDS; i++)); do
        FOLD_DIR="$RESULTS_DIR/fold_$i"
        if [ -d "$FOLD_DIR" ]; then
            echo "Fold $i:"
            if [ -f "$FOLD_DIR/best_model_fold_$i.pth" ]; then
                echo "  ✓ Best model saved"
            fi
            if [ -f "$FOLD_DIR/training_history_fold_$i.json" ]; then
                echo "  ✓ Training history saved"
            fi
            if [ -f "$FOLD_DIR/performance_metrics_fold_$i.json" ]; then
                echo "  ✓ Performance metrics saved"
            fi
        fi
    done
    
    echo ""
    echo "Individual fold results are stored in: $RESULTS_DIR/fold_*/"
    echo "Best models: $RESULTS_DIR/fold_*/best_model_fold_*.pth"
    echo "Training histories: $RESULTS_DIR/fold_*/training_history_fold_*.json"
    echo "Performance metrics: $RESULTS_DIR/fold_*/performance_metrics_fold_*.json"
    
    # Display GPU utilization summary if nvidia-smi is available
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "Current GPU Status:"
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU: %s | Util: %s%% | Memory: %s/%s MB\n", $1, $2, $3, $4}'
    fi
    
else
    echo "K-Fold training failed with exit code: $TRAINING_EXIT_CODE"
    echo "Check the error messages above for details."
    
    # Check for partial results
    echo ""
    echo "Checking for partial results..."
    COMPLETED_FOLDS=0
    for ((i=1; i<=K_FOLDS; i++)); do
        FOLD_DIR="$RESULTS_DIR/fold_$i"
        if [ -d "$FOLD_DIR" ] && [ -f "$FOLD_DIR/best_model_fold_$i.pth" ]; then
            COMPLETED_FOLDS=$((COMPLETED_FOLDS + 1))
        fi
    done
    
    if [ $COMPLETED_FOLDS -gt 0 ]; then
        echo "Partial results available for $COMPLETED_FOLDS out of $K_FOLDS folds"
        echo "Check individual fold directories for completed training"
    else
        echo "No completed folds found"
    fi
fi

echo "================================="

exit $TRAINING_EXIT_CODE
