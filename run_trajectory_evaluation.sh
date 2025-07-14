#!/bin/bash

# LSTM Trajectory-Level Evaluation Script
set -e

# Basic Configuration
RESULTS_DIR="./results/results_kfold_1gpu_8192"
DATA_DIR="data"
SAVE_DIR="./evaluations/trajectory_lstm_evaluation"
EVAL_SCRIPT="./src/eval/lstm_trajectory_eval.py"
PREPROCESSING_PARAMS="/home/shiqi_w/code/Combined-HIF-detector/preprocessing_params_fold.pkl"

# Window parameters
WINDOW_SIZE=30
STRIDE=15  # 50% overlap

echo "Running LSTM trajectory-level evaluation..."
echo "Results directory: $RESULTS_DIR"
echo "Data directory: $DATA_DIR"
echo "Window size: $WINDOW_SIZE"
echo "Stride: $STRIDE"
echo "Save directory: $SAVE_DIR"

# Check required files exist
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: Evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi

if [ ! -f "$PREPROCESSING_PARAMS" ]; then
    echo "Error: Preprocessing parameters not found: $PREPROCESSING_PARAMS"
    exit 1
fi

# Create save directory
mkdir -p "$SAVE_DIR"

# Run trajectory-level evaluation
python "$EVAL_SCRIPT" \
    --results_dir "$RESULTS_DIR" \
    --data_dir "$DATA_DIR" \
    --preprocessing_params "$PREPROCESSING_PARAMS" \
    --save_dir "$SAVE_DIR" \
    --window_size "$WINDOW_SIZE" \
    --stride "$STRIDE" \
    --max_folds 5 \
    --device cuda

echo "Trajectory-level evaluation completed! Results saved in: $SAVE_DIR"
