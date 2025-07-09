#!/bin/bash

# Simple Seq2Seq LSTM training script
# Usage: bash run_seq2seq_simple.sh

set -e

echo "Starting Seq2Seq LSTM training..."

# Basic parameters
DATA_DIR="data"
RESULTS_DIR="./results/seq2seq_simple"
SCRIPT_PATH="./src/trainers/seq2seq_trainer.py"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Run training directly with Python
python "$SCRIPT_PATH" \
    --data_dir "$DATA_DIR" \
    --state_dim 2 \
    --control_dim 2 \
    --hidden_size 128 \
    --num_layers 2 \
    --dropout 0.2 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --weight_decay 1e-4 \
    --patience 20 \
    --grad_clip 1.0 \
    --teacher_forcing_ratio 0.5 \
    --results_dir "$RESULTS_DIR" \
    --k_folds 5 \
    --bidirectional \
    --window_size 30 \
    --sample_step 1 \
    --seed 42 \
    --use_wandb \

echo "Training completed! Results saved in: $RESULTS_DIR"
