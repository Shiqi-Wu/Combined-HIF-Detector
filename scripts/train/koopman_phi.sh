#!/bin/bash

set -e

CONFIG_FILE="./configs/koopman/phi.json"
TRAIN_SCRIPT="./src/koopman/phi_train.py"

python "$TRAIN_SCRIPT" --config "$CONFIG_FILE"
