#!/bin/bash

# Simple LSTM Evaluation Script

python ./src/eval/lstm_eval.py \
    --results_dir "./results/results_kfold_1gpu_8192" \
    --data_dir "data" \
    --preprocessing_params "/home/shiqi_w/code/Combined-HIF-detector/preprocessing_params_fold.pkl" \
    --save_dir "./evaluations/evaluation_lstm" \
    --csv_file "./evaluation_results/fold_evaluation_results.csv" \
    --bidirectional
