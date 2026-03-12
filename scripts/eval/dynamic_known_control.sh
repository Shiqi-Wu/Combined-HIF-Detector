#!/bin/bash

set -e

CONFIG_FILE="./configs/lstm/classifier.json"
EVAL_SCRIPT="./src/eval_runner.py"
SAVE_CSV="./checkpoints/known_control_classifier/2000/eval_results.csv"

python "$EVAL_SCRIPT" --method dynamic --config "$CONFIG_FILE" --save_csv "$SAVE_CSV"
