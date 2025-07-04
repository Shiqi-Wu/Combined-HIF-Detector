#!/bin/bash

# Single-node multi-GPU distributed training launcher
# This script sets up and launches training on multiple GPUs on a single machine

set -e

# ================================
# Single-node Multi-GPU Configuration
# ================================

# Automatically detect available GPUs
AVAILABLE_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 1)")
NUM_GPUS=${NUM_GPUS:-$AVAILABLE_GPUS}  # Use all available GPUs by default

echo "Detected $AVAILABLE_GPUS GPU(s), using $NUM_GPUS GPU(s) for training"

# Training parameters optimized for single-node multi-GPU
DATA_DIR="data"
SAMPLE_STEP=1
WINDOW_SIZE=30
BATCH_SIZE=32        # Per-GPU batch size
GRADIENT_ACCUMULATION_STEPS=2
MIXED_PRECISION="fp16"

# Model parameters
HIDDEN_SIZE=128
NUM_LAYERS=2
DROPOUT=0.2
BIDIRECTIONAL=true

# Training parameters
NUM_EPOCHS=100
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-4
PATIENCE=20
GRAD_CLIP=1.0
SEED=42

# Logging parameters
USE_WANDB=false
WANDB_PROJECT="single-node-lstm-training"
EXPERIMENT_NAME="single_node_$(date +%Y%m%d_%H%M%S)"

# Output parameters
RESULTS_DIR="./results_single_node"

# Script path
SCRIPT_PATH="./src/trainers/distributed_trainer.py"

echo "================================="
echo "Single-Node Multi-GPU Training"
echo "================================="
echo "Hardware Configuration:"
echo "  Available GPUs: $AVAILABLE_GPUS"
echo "  Using GPUs: $NUM_GPUS"
echo "  Total GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n $NUM_GPUS | awk '{sum+=$1} END {printf "%.1f GB\n", sum/1024}' 2>/dev/null || echo "N/A")"
echo ""
echo "Training Configuration:"
echo "  Batch Size per GPU: $BATCH_SIZE"
echo "  Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))"
echo "  Mixed Precision: $MIXED_PRECISION"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Number of Epochs: $NUM_EPOCHS"
echo ""
echo "Model Configuration:"
echo "  Hidden Size: $HIDDEN_SIZE"
echo "  Number of Layers: $NUM_LAYERS"
echo "  Dropout: $DROPOUT"
echo "  Bidirectional: $BIDIRECTIONAL"
echo "================================="

# ================================
# Generate Single-node Accelerate Config
# ================================

ACCELERATE_CONFIG_FILE="./accelerate_single_node_config.yaml"

cat > "$ACCELERATE_CONFIG_FILE" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: $MIXED_PRECISION
num_machines: 1
num_processes: $NUM_GPUS
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "Generated single-node Accelerate configuration:"
cat "$ACCELERATE_CONFIG_FILE"
echo ""

# ================================
# Build Training Arguments
# ================================

ARGS="--data_dir $DATA_DIR"
ARGS="$ARGS --sample_step $SAMPLE_STEP"
ARGS="$ARGS --window_size $WINDOW_SIZE"
ARGS="$ARGS --batch_size $BATCH_SIZE"
ARGS="$ARGS --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"
ARGS="$ARGS --mixed_precision $MIXED_PRECISION"
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

# Add conditional arguments
if [ "$BIDIRECTIONAL" = true ]; then
    ARGS="$ARGS --bidirectional"
fi

if [ "$USE_WANDB" = true ]; then
    ARGS="$ARGS --use_wandb"
fi

# ================================
# Pre-flight Checks
# ================================

echo "Running pre-flight checks..."

# Check if CUDA is available
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    echo "Warning: CUDA not available. Training will use CPU (much slower)."
    # Update config for CPU training
    sed -i 's/use_cpu: false/use_cpu: true/g' "$ACCELERATE_CONFIG_FILE"
    sed -i 's/distributed_type: MULTI_GPU/distributed_type: NO/g' "$ACCELERATE_CONFIG_FILE"
    sed -i "s/num_processes: $NUM_GPUS/num_processes: 1/g" "$ACCELERATE_CONFIG_FILE"
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' not found!"
    echo "Please make sure your data is in the correct location."
    exit 1
fi

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Training script not found at '$SCRIPT_PATH'"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "Pre-flight checks completed successfully!"
echo ""

# ================================
# Launch Training
# ================================

echo "================================="
echo "Starting Single-Node Multi-GPU Training..."
echo "================================="

echo "Command: accelerate launch --config_file $ACCELERATE_CONFIG_FILE $SCRIPT_PATH $ARGS"
echo ""

# Start training
START_TIME=$(date +%s)
accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" "$SCRIPT_PATH" $ARGS
TRAINING_EXIT_CODE=$?
END_TIME=$(date +%s)
TRAINING_DURATION=$((END_TIME - START_TIME))

echo ""
echo "================================="
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Training duration: $(($TRAINING_DURATION / 3600))h $(($TRAINING_DURATION % 3600 / 60))m $(($TRAINING_DURATION % 60))s"
    echo "Results saved in: $RESULTS_DIR"
    
    # Display some results if available
    echo ""
    echo "Training Summary:"
    if [ -f "$RESULTS_DIR/training_history.json" ]; then
        echo "  ✓ Training history: $RESULTS_DIR/training_history.json"
    fi
    
    if [ -f "$RESULTS_DIR/best_model.pth" ]; then
        echo "  ✓ Best model: $RESULTS_DIR/best_model.pth"
    fi
    
    if [ -f "$RESULTS_DIR/performance_metrics.json" ]; then
        echo "  ✓ Performance metrics: $RESULTS_DIR/performance_metrics.json"
        
        # Extract final accuracy if available
        FINAL_ACC=$(python3 -c "
import json
try:
    with open('$RESULTS_DIR/training_history.json', 'r') as f:
        data = json.load(f)
    if 'val_acc' in data and data['val_acc']:
        print(f'Final validation accuracy: {max(data[\"val_acc\"]):.4f}')
except:
    pass
" 2>/dev/null)
        
        if [ -n "$FINAL_ACC" ]; then
            echo "  ✓ $FINAL_ACC"
        fi
    fi
    
    echo ""
    echo "You can visualize training progress with:"
    echo "  python3 -c \"import json, matplotlib.pyplot as plt; data=json.load(open('$RESULTS_DIR/training_history.json')); plt.plot(data['train_acc'], label='Train'); plt.plot(data['val_acc'], label='Val'); plt.legend(); plt.show()\""
    
else
    echo "Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "Check the error messages above for details."
    echo ""
    echo "Common troubleshooting:"
    echo "  1. Check GPU memory: nvidia-smi"
    echo "  2. Reduce batch size if out of memory"
    echo "  3. Check data directory exists: ls -la $DATA_DIR"
    echo "  4. Verify dependencies: python3 -c 'import torch, accelerate'"
fi
echo "================================="

exit $TRAINING_EXIT_CODE
