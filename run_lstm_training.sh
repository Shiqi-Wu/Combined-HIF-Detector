#!/bin/bash

# Bash script to run LSTM classifier with k-fold cross validation
# Author: Generated for Power Grid Fault Detection Pro# Execute the Python script
echo "Starting LSTM training with k-fold cross validation..."
echo "Command: python3 $SCRIPT_PATH $ARGS"
echo "================================="

# Use eval to properly handle the arguments
eval "python3 \"$SCRIPT_PATH\" $ARGS"set -e  # Exit on any error

# ================================
# Configuration - All parameters explicitly defined
# ================================

# Data parameters
DATA_DIR="data"
SAMPLE_STEP=1
WINDOW_SIZE=30
BATCH_SIZE=512

# Model parameters
HIDDEN_SIZE=64
NUM_LAYERS=2
DROPOUT=0.1
BIDIRECTIONAL=true

# Training parameters
NUM_EPOCHS=2000
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-4
PATIENCE=20

# K-fold parameters
N_FOLDS=8
RANDOM_STATE=42

# Logging parameters
USE_WANDB=true  # Set to true to use wandb, false to save logs to CSV files
WANDB_PROJECT="power-grid-fault-detection"
WANDB_ENTITY=""
EXPERIMENT_NAME="lstm_kfold_$(date +%Y%m%d_%H%M%S)"

# Output parameters
RESULTS_DIR="./results"
SAVE_MODELS=true

# Script path
SCRIPT_PATH="./src/models/fault_lstm_classifier.py"

# ================================
# Display Configuration
# ================================

echo "================================="
echo "LSTM Classifier Training Script"
echo "================================="
echo "Data Configuration:"
echo "  Data Directory: $DATA_DIR"
echo "  Sample Step: $SAMPLE_STEP"
echo "  Window Size: $WINDOW_SIZE"
echo "  Batch Size: $BATCH_SIZE"
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
echo ""
echo "K-Fold Configuration:"
echo "  Number of Folds: $N_FOLDS"
echo "  Random State: $RANDOM_STATE"
echo ""
echo "Logging Configuration:"
echo "  Use Wandb: $USE_WANDB"
if [ "$USE_WANDB" = true ]; then
    echo "  Wandb Project: $WANDB_PROJECT"
    echo "  Experiment Name: $EXPERIMENT_NAME"
else
    echo "  Training logs will be saved to CSV files in results directory"
    echo "  Individual fold logs: training_log_fold_X.csv"
    echo "  Combined logs: combined_training_logs_TIMESTAMP.csv"
fi
echo ""
echo "Output Configuration:"
echo "  Results Directory: $RESULTS_DIR"
echo "  Save Models: $SAVE_MODELS"
echo "================================="

# Create results directory
mkdir -p "$RESULTS_DIR"

# ================================
# Execute Python Script
# ================================

# Build arguments string
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
ARGS="$ARGS --n_folds $N_FOLDS"
ARGS="$ARGS --random_state $RANDOM_STATE"
ARGS="$ARGS --wandb_project $WANDB_PROJECT"
ARGS="$ARGS --experiment_name $EXPERIMENT_NAME"
ARGS="$ARGS --results_dir $RESULTS_DIR"

# Add conditional arguments
if [ "$BIDIRECTIONAL" = true ]; then
    ARGS="$ARGS --bidirectional"
fi

if [ "$USE_WANDB" = true ]; then
    ARGS="$ARGS --use_wandb"
fi

if [ "$SAVE_MODELS" = true ]; then
    ARGS="$ARGS --save_models"
fi

if [ -n "$WANDB_ENTITY" ]; then
    ARGS="$ARGS --wandb_entity $WANDB_ENTITY"
fi

# Execute the Python script
echo "Starting LSTM training with k-fold cross validation..."
echo "Command: python3 $SCRIPT_PATH $ARGS"
echo "================================="

python3 $SCRIPT_PATH $ARGS

echo "================================="
echo "Training completed!"
echo "Results saved in: $RESULTS_DIR"
echo "================================="
