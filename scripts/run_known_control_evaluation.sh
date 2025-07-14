#!/bin/bash

# Simple Known Control Classifier Evaluation
set -e

# Basic Configuration
DATA_DIR="data"
OUTPUT_DIR="./evaluations/evaluation_results_known_control_2000_300"
EVAL_SCRIPT="./src/eval/known_control_classifier_eval.py"
TRAIN_WINDOW_SIZE=2000  # Window size for fitting K and B matrices
TEST_WINDOW_SIZE=300   # Window size for classification

echo "Running known control classifier evaluation..."
echo "Training window size (K,B fitting): $TRAIN_WINDOW_SIZE"
echo "Testing window size (classification): $TEST_WINDOW_SIZE"

# Check required files exist
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: Evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
python "$EVAL_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --save_dir "$OUTPUT_DIR" \
    --train_window_size "$TRAIN_WINDOW_SIZE" \
    --test_window_size "$TEST_WINDOW_SIZE"

echo "Evaluation completed! Results saved in: $OUTPUT_DIR"
