#!/bin/bash

set -e

CONFIG_FILE="./configs/koopman/residual.json"
TRAIN_SCRIPT="./src/koopman/residual_train.py"

python "$TRAIN_SCRIPT" --config "$CONFIG_FILE"
