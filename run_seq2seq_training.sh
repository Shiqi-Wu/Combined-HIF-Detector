#!/bin/bash

# K-fold Seq2Seq LSTM training launcher using Accelerate
# This script trains a sequence-to-sequence LSTM model for control signal generation

set -e  # Exit on any error

# ================================
# Configuration - All parameters explicitly defined
# ================================

# Data parameters
DATA_DIR="data"
SAMPLE_STEP=1
WINDOW_SIZE=30
PCA_DIM=2

# Model parameters
STATE_DIM=68      # Dimension of state signals x(t)
CONTROL_DIM=10    # Dimension of control signals u(t)
HIDDEN_SIZE=128
NUM_LAYERS=2
DROPOUT=0.2
BIDIRECTIONAL=true
USE_ATTENTION=false
TEACHER_FORCING_RATIO=0.5

# Training parameters
BATCH_SIZE=32
NUM_EPOCHS=100
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-4
PATIENCE=20
GRAD_CLIP=1.0
GRADIENT_ACCUMULATION_STEPS=2
SEED=42

# Loss function parameters
MSE_WEIGHT=1.0
L1_WEIGHT=0.1
SMOOTHNESS_WEIGHT=0.01

# Teacher forcing scheduling
TEACHER_FORCING_DECAY=0.95
MIN_TEACHER_FORCING_RATIO=0.1

# K-fold parameters
K_FOLDS=5

# Logging parameters
USE_WANDB=false  # Set to true to use wandb
WANDB_PROJECT="seq2seq-kfold-training"
EXPERIMENT_NAME="seq2seq_kfold_$(date +%Y%m%d_%H%M%S)"

# Output parameters
RESULTS_DIR="./results_seq2seq_kfold"

# Script path
SCRIPT_PATH="./src/trainers/seq2seq_trainer.py"

# ================================
# Display Configuration
# ================================

echo "================================="
echo "Seq2Seq LSTM K-Fold Training Script"
echo "================================="
echo "Data Configuration:"
echo "  Data Directory: $DATA_DIR"
echo "  Sample Step: $SAMPLE_STEP"
echo "  Window Size: $WINDOW_SIZE"
echo "  PCA Dimension: $PCA_DIM"
echo "  Batch Size: $BATCH_SIZE"
echo ""
echo "Model Configuration:"
echo "  State Dimension: $STATE_DIM"
echo "  Control Dimension: $CONTROL_DIM"
echo "  Hidden Size: $HIDDEN_SIZE"
echo "  Number of Layers: $NUM_LAYERS"
echo "  Dropout: $DROPOUT"
echo "  Bidirectional: $BIDIRECTIONAL"
echo "  Use Attention: $USE_ATTENTION"
echo "  Teacher Forcing Ratio: $TEACHER_FORCING_RATIO"
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
echo "Loss Function Configuration:"
echo "  MSE Weight: $MSE_WEIGHT"
echo "  L1 Weight: $L1_WEIGHT"
echo "  Smoothness Weight: $SMOOTHNESS_WEIGHT"
echo ""
echo "Teacher Forcing Scheduling:"
echo "  Decay Rate: $TEACHER_FORCING_DECAY"
echo "  Minimum Ratio: $MIN_TEACHER_FORCING_RATIO"
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
    echo "  Performance metrics: performance_metrics_fold_X.json"
fi
echo ""
echo "Output Configuration:"
echo "  Results Directory: $RESULTS_DIR"
echo "================================="

# ================================
# Display Configuration
# ================================

echo "================================="
echo "LSTM Sequence-to-Sequence Training"
echo "================================="
echo "Data Configuration:"
echo "  Data Directory: $DATA_DIR"
echo "  Input Sequence Length: $INPUT_LENGTH"
echo "  Target Sequence Length: $TARGET_LENGTH"
echo "  Window Size: $WINDOW_SIZE"
echo "  Sample Step: $SAMPLE_STEP"
echo ""
echo "Model Configuration:"
echo "  Hidden Size: $HIDDEN_SIZE"
echo "  Number of Layers: $NUM_LAYERS"
echo "  Dropout: $DROPOUT"
echo ""
echo "Training Configuration:"
echo "  Batch Size (per device): $BATCH_SIZE"
echo "  Number of Epochs: $NUM_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Patience: $PATIENCE"
echo "  Gradient Clipping: $GRAD_CLIP"
echo "  Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  Mixed Precision: $MIXED_PRECISION"
echo "  Teacher Forcing Ratio: $TEACHER_FORCING_RATIO"
echo "  Random Seed: $SEED"
echo ""
echo "Logging Configuration:"
echo "  Use Wandb: $USE_WANDB"
if [ "$USE_WANDB" = true ]; then
    echo "  Wandb Project: $WANDB_PROJECT"
    echo "  Experiment Name: $EXPERIMENT_NAME"
else
    echo "  Training logs will be saved to local files"
fi
echo ""
echo "Output Configuration:"
echo "  Results Directory: $RESULTS_DIR"
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

echo "Generating Accelerate configuration for Seq2Seq training..."

# Create accelerate config optimized for sequence generation
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

echo "Created Accelerate configuration:"
cat "$ACCELERATE_CONFIG_FILE"
echo ""

# ================================
# Build Training Arguments
# ================================

ARGS="--data_dir $DATA_DIR"
ARGS="$ARGS --input_length $INPUT_LENGTH"
ARGS="$ARGS --target_length $TARGET_LENGTH"
ARGS="$ARGS --window_size $WINDOW_SIZE"
ARGS="$ARGS --sample_step $SAMPLE_STEP"
ARGS="$ARGS --hidden_size $HIDDEN_SIZE"
ARGS="$ARGS --num_layers $NUM_LAYERS"
ARGS="$ARGS --dropout $DROPOUT"
ARGS="$ARGS --batch_size $BATCH_SIZE"
ARGS="$ARGS --num_epochs $NUM_EPOCHS"
ARGS="$ARGS --learning_rate $LEARNING_RATE"
ARGS="$ARGS --weight_decay $WEIGHT_DECAY"
ARGS="$ARGS --patience $PATIENCE"
ARGS="$ARGS --grad_clip $GRAD_CLIP"
ARGS="$ARGS --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"
ARGS="$ARGS --mixed_precision $MIXED_PRECISION"
ARGS="$ARGS --teacher_forcing_ratio $TEACHER_FORCING_RATIO"
ARGS="$ARGS --seed $SEED"
ARGS="$ARGS --results_dir $RESULTS_DIR"
ARGS="$ARGS --wandb_project $WANDB_PROJECT"

# Add conditional arguments
if [ "$USE_WANDB" = true ]; then
    ARGS="$ARGS --use_wandb"
fi

if [ -n "$WANDB_ENTITY" ]; then
    ARGS="$ARGS --wandb_entity $WANDB_ENTITY"
fi

# ================================
# Execute Training
# ================================

echo "Starting sequence-to-sequence training..."
echo "Command: accelerate launch --config_file $ACCELERATE_CONFIG_FILE $SCRIPT_PATH $ARGS"
echo "================================="

# Launch training with accelerate
accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" "$SCRIPT_PATH" $ARGS

TRAINING_EXIT_CODE=$?

echo "================================="
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Sequence-to-sequence training completed successfully!"
    echo "Results saved in: $RESULTS_DIR"
    
    # Display some results if available
    if [ -f "$RESULTS_DIR/training_history.json" ]; then
        echo ""
        echo "Training Summary:"
        echo "Training history available at: $RESULTS_DIR/training_history.json"
    fi
    
    if [ -f "$RESULTS_DIR/best_model.pth" ]; then
        echo "Best model saved at: $RESULTS_DIR/best_model.pth"
    fi
    
    if [ -f "$RESULTS_DIR/prediction_examples.png" ]; then
        echo "Prediction examples saved at: $RESULTS_DIR/prediction_examples.png"
    fi
    
    echo ""
    echo "Model Architecture Summary:"
    echo "  Input: State sequences x(0:t) + Control history u(0:t-1)"
    echo "  Output: Future control sequences u(t:t+$TARGET_LENGTH)"
    echo "  Method: Encoder-Decoder with Attention and Teacher Forcing"
    echo ""
    echo "You can now use the trained model for:"
    echo "  1. Generating control sequences from state observations"
    echo "  2. Analyzing attention patterns over input sequences" 
    echo "  3. Fine-tuning on specific control tasks"
    
else
    echo "Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "Check the error messages above for details."
    echo ""
    echo "Common issues and solutions:"
    echo "  - Out of memory: Reduce batch_size or sequence lengths"
    echo "  - CUDA errors: Check GPU availability and drivers"
    echo "  - Data loading errors: Verify data directory and format"
fi

echo "================================="

exit $TRAINING_EXIT_CODE
