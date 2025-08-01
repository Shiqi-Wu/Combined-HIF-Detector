#!/bin/bash

# === CONFIG ===
CONFIG_FILE="./configs/lstm_classifier_config.json"
TRAIN_SCRIPT="./src/trainers/lstm_classifier_trainer.py"
ACCELERATE_CONFIG_PATH="./configs/accelerate_config.yaml"

# === CREATE DIRECTORIES ===
mkdir -p "${SAVE_DIR}"
mkdir -p "${SAVE_DIR}/logs"

# === EXPORT ACCELERATE CONFIG PATH ===
export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_PATH}"

# === LAUNCH TRAINING ===
echo "Launching training with config at $ACCELERATE_CONFIG_FILE"
accelerate launch "${TRAIN_SCRIPT}" \
  --config "${CONFIG_FILE}" 
