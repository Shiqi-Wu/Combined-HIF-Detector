#!/bin/bash

# Simple Known Control Classifier Evaluation
set -e

# === CONFIG ===
CONFIG_FILE="./configs/dynamical_system_known_control.json"
EVAL_SCRIPT="./src/eval/known_control_classifier_eval.py"

python "$EVAL_SCRIPT" --config "$CONFIG_FILE"
