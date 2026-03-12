#!/bin/bash

set -e

CONFIG_FILE="./configs/koopman/residual.json"
CHECKPOINT="./checkpoints/koopman_residual/best_model.pt"
EVAL_SCRIPT="./src/koopman/residual_eval.py"
OUTPUT_PREFIX="./evaluations/koopman_residual"

python "$EVAL_SCRIPT" \
  --config "$CONFIG_FILE" \
  --checkpoint "$CHECKPOINT" \
  --output_prefix "$OUTPUT_PREFIX"
