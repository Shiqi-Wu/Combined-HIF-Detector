#!/bin/bash

# Simple State-Based Classifier Evaluation
set -e

# Basic Configuration
DATA_DIR="data"
OUTPUT_DIR="./evaluations/evaluation_results_state_based"
EVAL_SCRIPT="./src/eval/state_based_classifier_eval.py"

echo "Running state-based classifier evaluation..."

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
    --fold 0 \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Evaluation completed! Results saved in: $OUTPUT_DIR"
