#!/bin/bash

set -e

CONFIG_FILE="./configs/koopman/phi.json"
CHECKPOINT="./checkpoints/koopman_phi/best_model.pt"
EVAL_SCRIPT="./src/eval_runner.py"
SAVE_CSV="./evaluations/koopman_phi_eval.csv"

python "$EVAL_SCRIPT" \
  --method koopman \
  --config "./configs/lstm/classifier.json" \
  --koopman_config "$CONFIG_FILE" \
  --koopman_checkpoint "$CHECKPOINT" \
  --save_csv "$SAVE_CSV"
