#!/bin/bash

set -e

CONFIG_FILE="./configs/lstm/classifier.json"
TRAIN_SCRIPT="./src/lstm/train.py"
ACCELERATE_CONFIG_PATH="./configs/runtime/accelerate.yaml"
export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_PATH}"
echo "Launching training with config at $ACCELERATE_CONFIG_FILE"
accelerate launch "${TRAIN_SCRIPT}" \
  --config "${CONFIG_FILE}" 
