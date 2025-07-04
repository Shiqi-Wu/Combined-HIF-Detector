#!/bin/bash

# Single-node multi-GPU training launcher for LSTM classifier
# Optimized for single machine with multiple GPUs using HuggingFace Accelerate

set -e  # Exit on any error

# ================================
# Configuration - Single-node Multi-GPU
# ================================

# Automatically detect GPU configuration
AVAILABLE_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
NUM_GPUS=${NUM_GPUS:-$AVAILABLE_GPUS}

echo "================================="
echo "Single-Node Multi-GPU LSTM Training"
echo "================================="
echo "GPU Configuration:"
echo "  Available GPUs: $AVAILABLE_GPUS"
echo "  Using GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "  Mode: CPU training (no GPUs detected)"
    DEVICE_TYPE="cpu"
else
    echo "  Mode: Multi-GPU training"
    DEVICE_TYPE="gpu"
    # Display GPU information
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -n $NUM_GPUS | while read line; do
        echo "    $line"
    done
fi

# ================================
# Training Parameters
# ================================

# Data parameters
DATA_DIR="data"
SAMPLE_STEP=1
WINDOW_SIZE=30

# Model parameters  
HIDDEN_SIZE=128
NUM_LAYERS=6
DROPOUT=0.2
BIDIRECTIONAL=true

# Training parameters optimized for multi-GPU
if [ "$NUM_GPUS" -gt 1 ]; then
    BATCH_SIZE=4096  # Per-GPU batch size for multi-GPU
    GRADIENT_ACCUMULATION_STEPS=2
else
    BATCH_SIZE=4096  # Larger batch for single GPU/CPU
    GRADIENT_ACCUMULATION_STEPS=1
fi

NUM_EPOCHS=1
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-4
PATIENCE=20
GRAD_CLIP=1.0
MIXED_PRECISION="no"
SEED=42

# Logging parameters
USE_WANDB=true
WANDB_PROJECT="single-node-lstm"
EXPERIMENT_NAME="single_node_$(date +%Y%m%d_%H%M%S)"

# Output
RESULTS_DIR="./results/results_6layers_lstm"
SCRIPT_PATH="./src/trainers/distributed_trainer.py"

# ================================
# Display Configuration
# ================================

EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))
if [ "$NUM_GPUS" -eq 0 ]; then
    EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
fi

echo ""
echo "Training Configuration:"
echo "  Batch Size per Device: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Effective Batch Size: $EFFECTIVE_BATCH_SIZE"
echo "  Mixed Precision: $MIXED_PRECISION"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Number of Epochs: $NUM_EPOCHS"
echo "  Early Stopping Patience: $PATIENCE"
echo ""
echo "Model Configuration:"
echo "  Hidden Size: $HIDDEN_SIZE"
echo "  LSTM Layers: $NUM_LAYERS"
echo "  Dropout: $DROPOUT"
echo "  Bidirectional: $BIDIRECTIONAL"
echo ""
echo "Data Configuration:"
echo "  Data Directory: $DATA_DIR"
echo "  Window Size: $WINDOW_SIZE"
echo "  Sample Step: $SAMPLE_STEP"
echo "================================="

# ================================
# Generate Accelerate Configuration
# ================================

ACCELERATE_CONFIG_FILE="./accelerate_config.yaml"

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU configuration
    cat > "$ACCELERATE_CONFIG_FILE" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: $NUM_GPUS
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
    echo "Generated multi-GPU configuration for $NUM_GPUS GPUs"
elif [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU configuration
    cat > "$ACCELERATE_CONFIG_FILE" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: NO
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 1
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
    echo "Generated single-GPU configuration"
else
    # CPU configuration
    cat > "$ACCELERATE_CONFIG_FILE" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: NO
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 1
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: true
EOF
    echo "Generated CPU configuration"
fi

# ================================
# Pre-flight Checks
# ================================

echo ""
echo "Running pre-flight checks..."

# Check if accelerate is installed
if ! command -v accelerate &> /dev/null; then
    echo "Error: accelerate command not found!"
    echo "Please install with: pip install accelerate"
    exit 1
fi

# Check if training script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Training script not found at $SCRIPT_PATH"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR not found!"
    echo "Please ensure your data is in the correct location."
    exit 1
fi

# Check data files
DATA_FILES=$(find "$DATA_DIR" -name "*.npy" | wc -l)
if [ "$DATA_FILES" -eq 0 ]; then
    echo "Error: No .npy data files found in $DATA_DIR"
    exit 1
fi
echo "Found $DATA_FILES data files in $DATA_DIR"

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "Pre-flight checks passed!"

# ================================
# Build Arguments
# ================================

ARGS="--data_dir $DATA_DIR"
ARGS="$ARGS --sample_step $SAMPLE_STEP"
ARGS="$ARGS --window_size $WINDOW_SIZE"
ARGS="$ARGS --batch_size $BATCH_SIZE"
ARGS="$ARGS --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"
ARGS="$ARGS --hidden_size $HIDDEN_SIZE"
ARGS="$ARGS --num_layers $NUM_LAYERS"
ARGS="$ARGS --dropout $DROPOUT"
ARGS="$ARGS --num_epochs $NUM_EPOCHS"
ARGS="$ARGS --learning_rate $LEARNING_RATE"
ARGS="$ARGS --weight_decay $WEIGHT_DECAY"
ARGS="$ARGS --patience $PATIENCE"
ARGS="$ARGS --grad_clip $GRAD_CLIP"
ARGS="$ARGS --seed $SEED"
ARGS="$ARGS --results_dir $RESULTS_DIR"
ARGS="$ARGS --wandb_project $WANDB_PROJECT"

# Conditional arguments
if [ "$BIDIRECTIONAL" = true ]; then
    ARGS="$ARGS --bidirectional"
fi

if [ "$USE_WANDB" = true ]; then
    ARGS="$ARGS --use_wandb"
fi

# ================================
# Launch Training
# ================================

echo ""
echo "================================="
echo "Starting Training..."
echo "================================="
echo "Command: accelerate launch --config_file $ACCELERATE_CONFIG_FILE $SCRIPT_PATH $ARGS"
echo ""

# Record start time
START_TIME=$(date +%s)

# Launch training
accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" "$SCRIPT_PATH" $ARGS

# Record end time and calculate duration
TRAINING_EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "================================="
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training Completed Successfully! 🎉"
    echo "Total training time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    echo "Results location: $RESULTS_DIR"
    echo ""
    
    # Show training results
    if [ -f "$RESULTS_DIR/training_history.json" ]; then
        echo "Training Summary:"
        python3 -c "
import json
try:
    with open('$RESULTS_DIR/training_history.json', 'r') as f:
        data = json.load(f)
    
    if 'val_acc' in data and data['val_acc']:
        best_acc = max(data['val_acc'])
        final_acc = data['val_acc'][-1]
        print(f'  Best validation accuracy: {best_acc:.4f}')
        print(f'  Final validation accuracy: {final_acc:.4f}')
        print(f'  Total epochs trained: {len(data[\"val_acc\"])}')
    
    if 'train_acc' in data and data['train_acc']:
        final_train = data['train_acc'][-1]
        print(f'  Final training accuracy: {final_train:.4f}')
        
except Exception as e:
    print(f'  Could not parse training history: {e}')
"
    fi
    
    echo ""
    echo "Available files:"
    ls -la "$RESULTS_DIR"/ | grep -E '\.(pth|json|png)$' || echo "  No result files found"
    
    echo ""
    echo "Next steps:"
    echo "  1. Check training curves: python3 -c \"import json, matplotlib.pyplot as plt; data=json.load(open('$RESULTS_DIR/training_history.json')); plt.subplot(121); plt.plot(data['train_acc'], label='Train'); plt.plot(data['val_acc'], label='Val'); plt.legend(); plt.title('Accuracy'); plt.subplot(122); plt.plot(data['train_loss'], label='Train'); plt.plot(data['val_loss'], label='Val'); plt.legend(); plt.title('Loss'); plt.tight_layout(); plt.show()\""
    echo "  2. Load best model: torch.load('$RESULTS_DIR/best_model.pth')"
    echo "  3. Run evaluation on test set"
    
else
    echo "Training Failed ❌"
    echo "Exit code: $TRAINING_EXIT_CODE"
    echo "Duration before failure: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    echo "Troubleshooting tips:"
    echo "  1. Check GPU memory: nvidia-smi"
    echo "  2. Reduce batch size if OOM error"
    echo "  3. Check data format and paths"
    echo "  4. Verify all dependencies are installed"
    echo "  5. Check the error messages above"
fi
echo "================================="

# Cleanup temporary config file
rm -f "$ACCELERATE_CONFIG_FILE"

exit $TRAINING_EXIT_CODE
