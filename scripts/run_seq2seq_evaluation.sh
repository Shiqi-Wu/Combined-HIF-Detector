#!/bin/bash

# Simple Seq2Seq Evaluation Script
set -e

# Basic Configuration
RESULTS_DIR="./results/seq2seq_simple"
DATA_DIR="data"
SAVE_DIR="./evaluation_results_seq2seq"
EVAL_SCRIPT="./src/eval/dynamic_system_classifier_eval.py"
FOLD_IDX=1  # Which fold to evaluate (0-4 for 5-fold)

echo "Running seq2seq evaluation..."

# Find the best model file for the specified fold
MODEL_FILE="$RESULTS_DIR/fold_$((FOLD_IDX + 1))/best_model_fold_$((FOLD_IDX + 1)).pth"

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

# Check if model file exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found: $MODEL_FILE"
    echo "Available models:"
    ls -la "$RESULTS_DIR"/*/best_model_*.pth 2>/dev/null || echo "No models found"
    exit 1
fi

echo "Using model: $MODEL_FILE"

# Create save directory
mkdir -p "$SAVE_DIR"

# Run evaluation with default parameters
python "$EVAL_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --seq2seq_model "$MODEL_FILE" \
    --save_dir "$SAVE_DIR" \
    --fold_idx "$FOLD_IDX"

echo "Evaluation completed! Results saved in: $SAVE_DIR"
