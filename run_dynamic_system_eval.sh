#!/bin/bash

# Dynamic System Classifier Evaluation Script
# This script evaluates the hybrid dynamic system + seq2seq classifier

set -e  # Exit on any error

# ================================
# Configuration
# ================================

# Required paths
DATA_DIR="data"
SEQ2SEQ_MODEL="./results/seq2seq_results/best_model.pth"  # Change this to your seq2seq model path

# Optional paths
PREPROCESSING_PARAMS="/home/shiqi_w/code/Combined-HIF-detector/preprocessing_params_fold.pkl"
SAVE_DIR="./dynamic_system_evaluation_results"  # Where to save evaluation results

# Parameters (should match training configuration)
FOLD_IDX=0  # Which fold to use for evaluation
K_FOLDS=5
RANDOM_STATE=42
WINDOW_SIZE=30
SAMPLE_STEP=1
BATCH_SIZE=64

# Seq2seq model parameters (should match training configuration)
STATE_DIM=2  # After PCA
CONTROL_DIM=2
HIDDEN_SIZE=128
NUM_LAYERS=2
DROPOUT=0.2
BIDIRECTIONAL=true

# ================================
# Display Configuration
# ================================

echo "================================="
echo "Dynamic System Classifier Evaluation"
echo "================================="
echo "Paths:"
echo "  Data Directory: $DATA_DIR"
echo "  Seq2Seq Model: $SEQ2SEQ_MODEL"
echo "  Preprocessing Params: $PREPROCESSING_PARAMS"
echo "  Save Directory: $SAVE_DIR"
echo ""
echo "Data Configuration:"
echo "  Fold Index: $FOLD_IDX"
echo "  K-Folds: $K_FOLDS"
echo "  Random State: $RANDOM_STATE"
echo "  Window Size: $WINDOW_SIZE"
echo "  Sample Step: $SAMPLE_STEP"
echo "  Batch Size: $BATCH_SIZE"
echo ""
echo "Model Configuration:"
echo "  State Dimension: $STATE_DIM"
echo "  Control Dimension: $CONTROL_DIM"
echo "  Hidden Size: $HIDDEN_SIZE"
echo "  Number of Layers: $NUM_LAYERS"
echo "  Dropout: $DROPOUT"
echo "  Bidirectional: $BIDIRECTIONAL"
echo "================================="

# ================================
# Validation
# ================================

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

# Check if seq2seq model exists
if [ ! -f "$SEQ2SEQ_MODEL" ]; then
    echo "Error: Seq2seq model not found: $SEQ2SEQ_MODEL"
    echo "Please train a seq2seq model first or update the path."
    exit 1
fi

# Check if preprocessing params file exists
if [ ! -f "$PREPROCESSING_PARAMS" ]; then
    echo "Error: Preprocessing parameters file not found: $PREPROCESSING_PARAMS"
    echo "Please check the path or run training first to generate this file."
    exit 1
fi

# Check if evaluation script exists
EVAL_SCRIPT="./src/eval/dynamic_system_classifier_eval.py"
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: Evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi

# Create save directory
mkdir -p "$SAVE_DIR"

# ================================
# Run Evaluation
# ================================

echo ""
echo "Starting Dynamic System Classifier evaluation..."
echo ""

# Build command
CMD="python $EVAL_SCRIPT"
CMD="$CMD --data_dir '$DATA_DIR'"
CMD="$CMD --seq2seq_model '$SEQ2SEQ_MODEL'"
CMD="$CMD --preprocessing_params '$PREPROCESSING_PARAMS'"
CMD="$CMD --save_dir '$SAVE_DIR'"
CMD="$CMD --fold_idx $FOLD_IDX"
CMD="$CMD --k_folds $K_FOLDS"
CMD="$CMD --random_state $RANDOM_STATE"
CMD="$CMD --window_size $WINDOW_SIZE"
CMD="$CMD --sample_step $SAMPLE_STEP"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --state_dim $STATE_DIM"
CMD="$CMD --control_dim $CONTROL_DIM"
CMD="$CMD --hidden_size $HIDDEN_SIZE"
CMD="$CMD --num_layers $NUM_LAYERS"
CMD="$CMD --dropout $DROPOUT"

if [ "$BIDIRECTIONAL" = true ]; then
    CMD="$CMD --bidirectional"
fi

# Execute command
echo "Executing: $CMD"
echo ""
eval $CMD

# ================================
# Completion
# ================================

echo ""
echo "================================="
echo "EVALUATION COMPLETED"
echo "================================="
echo "Results saved in: $SAVE_DIR"
echo ""
echo "Generated files:"
echo "  - confusion_matrix_*.png: Confusion matrices for each dataset"
echo "  - error_analysis_*.png: Prediction error analysis plots"
echo "  - evaluation_results.json: Numerical results summary"
echo ""
echo "The evaluation shows:"
echo "  1. System matrices K (shared) and B (per-class) estimation"
echo "  2. Seq2seq model control generation performance"  
echo "  3. Classification accuracy based on prediction residuals"
echo "  4. Detailed error analysis and confusion matrices"
echo "================================="
