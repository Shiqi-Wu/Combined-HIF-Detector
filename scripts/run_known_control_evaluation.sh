#!/bin/bash

# Simple Known Control Classifier Evaluation
set -e

# === CONFIG ===
CONFIG_FILE="./configs/lstm_classifier_config.json"
EVAL_SCRIPT="./src/eval/eval_runner.py"
SAVE_CSV="./checkpoints/known_control_classifier/2000/eval_results.csv"

python "$EVAL_SCRIPT" --method dynamic --config "$CONFIG_FILE" --save_csv "$SAVE_CSV"