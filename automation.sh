#!/bin/bash
# P-EAGLE Full Automation Script
# Handles data generation, feature extraction, and training

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Change to project directory
cd "$(dirname "$0")"
source venv/bin/activate

# Configuration
MODE=${1:-"single"}  # single or multi
MASTER_IP=${2:-""}

DRAFTER_MODEL="google/gemma-3-270m-it"
TARGET_MODEL="google/gemma-3-4b-it"
TARGET_HIDDEN_DIM=2560
SPECULATION_DEPTH=2
MAX_SEQ_LEN=1024

if [ "$MODE" == "single" ]; then
    BATCH_SIZE=1
    NUM_EPOCHS=3
    OUTPUT_DIR="checkpoints/automated_single"
    print_header "P-EAGLE Automation - Single GPU Mode"
elif [ "$MODE" == "multi" ]; then
    BATCH_SIZE=1
    NUM_EPOCHS=3
    OUTPUT_DIR="checkpoints/automated_multi"
    print_header "P-EAGLE Automation - Multi-GPU Mode"

    if [ -z "$MASTER_IP" ]; then
        print_error "Master IP required for multi-GPU mode"
        echo "Usage: $0 multi <master_ip>"
        exit 1
    fi
else
    print_error "Invalid mode: $MODE"
    echo "Usage: $0 [single|multi] [master_ip]"
    exit 1
fi

# Check prerequisites
print_header "Step 1: Checking Prerequisites"

if [ ! -d "data/features" ]; then
    print_warning "Feature directory not found. Need to generate data first."

    if [ ! -d "data/raw" ]; then
        print_warning "Raw data not found. Running data generation..."
        python3 scripts/generate_data.py \
            --output data/raw/train.jsonl \
            --num_samples 1000 \
            --min_tokens 100 \
            --max_tokens 4000
        print_success "Data generation complete"
    fi

    print_warning "Running feature extraction..."
    python3 -m p_eagle.training.feature_extractor \
        --dataset data/raw/train.jsonl \
        --target_model "$TARGET_MODEL" \
        --output_dir data/features \
        --layer_indices -1 -2 -3 \
        --batch_size 4
    print_success "Feature extraction complete"
else
    print_success "Features found at data/features"
fi

# Run preflight check
print_header "Step 2: Running Preflight Checks"
python3 scripts/preflight_check.py --feature_dir data/features

# Training
print_header "Step 3: Starting Training"

TRAIN_ARGS=(
    --drafter_model "$DRAFTER_MODEL"
    --target_hidden_dim "$TARGET_HIDDEN_DIM"
    --speculation_depth "$SPECULATION_DEPTH"
    --feature_dir data/features
    --output_dir "$OUTPUT_DIR"
    --use_lora
    --lora_rank 32
    --batch_size "$BATCH_SIZE"
    --num_epochs "$NUM_EPOCHS"
    --learning_rate 5e-05
    --warmup_steps 50
    --skip-hardware-check
    --skip-security-check
    --quantization 4bit
    --yes
    --gradient_accumulation_steps 1
    --max_seq_len "$MAX_SEQ_LEN"
    --gpu-safety-margin 2.0
)

if [ "$MODE" == "single" ]; then
    print_header "Training on Single GPU"
    python3 -m p_eagle.training.trainer "${TRAIN_ARGS[@]}"
else
    print_header "Training on Multiple GPUs"
    # Determine rank
    MY_IP=$(hostname -I | awk '{print $1}')
    if [[ "$MY_IP" == *"105"* ]]; then
        RANK=1
        NODE_RANK=1
        print_warning "Detected as Worker (Rank 1)"
    else
        RANK=0
        NODE_RANK=0
        print_success "Detected as Master (Rank 0)"
    fi

    export MASTER_ADDR="$MASTER_IP"
    export MASTER_PORT=29500
    export WORLD_SIZE=2
    export RANK="$RANK"

    torchrun \
        --nnodes=2 \
        --nproc_per_node=1 \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m p_eagle.training.trainer \
        "${TRAIN_ARGS[@]}"
fi

print_header "Training Complete!"
print_success "Model saved to: $OUTPUT_DIR/best_model"
print_success "Logs saved to: $OUTPUT_DIR/logs"

# Summary
print_header "Summary"
echo "Mode: $MODE"
echo "Output: $OUTPUT_DIR"
echo "To evaluate: python3 -m p_eagle.inference.generate --checkpoint $OUTPUT_DIR/best_model"
