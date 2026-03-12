#!/bin/bash

set -e

CONFIG_FILE="configs/lstm/classifier.json"
MODEL_PATH="checkpoints/lstm_classifier/2000/best_model.pth"
OUTPUT_CSV="checkpoints/lstm_classifier/2000/eval_results.csv"

python src/lstm/eval.py \
    --config $CONFIG_FILE \
    --model_path $MODEL_PATH \
    --output_csv $OUTPUT_CSV
