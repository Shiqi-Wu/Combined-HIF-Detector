#!/bin/bash

# Simple LSTM Evaluation Script

CONFIG_FILE="configs/lstm_classifier_config.json"
MODEL_PATH="checkpoints/lstm_classifier/2000/best_model.pth"
OUTPUT_CSV="checkpoints/lstm_classifier/2000/eval_results.csv"

python src/eval/lstm_eval.py \
    --config $CONFIG_FILE \
    --model_path $MODEL_PATH \
    --output_csv $OUTPUT_CSV
