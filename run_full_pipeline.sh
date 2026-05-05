#!/bin/bash
# P-EAGLE Full Pipeline Automation Script - FIXED VERSION
# Includes loss_mask_segments generation and verification steps
#
# Usage:
#   ./run_full_pipeline.sh                          # Run with defaults
#   ./run_full_pipeline.sh --target <model> --drafter <model>  # Custom models
#   ./run_full_pipeline.sh --skip-data-gen          # Skip data generation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================"
echo "P-EAGLE Full Pipeline (Fixed)"
echo "================================"

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

# DGX Sparx Configuration: Gemma 3 4B (target) + Gemma 3 270M (drafter)
# Same model family (Gemma 3) for vocab and hidden state compatibility
TARGET_MODEL="google/gemma-3-4b-it"
DRAFTER_MODEL="google/gemma-3-270m-it"

# Dimensions for Gemma 3 models
TARGET_HIDDEN_DIM="2560"   # gemma-3-4b: 2560
DRAFTER_HIDDEN_DIM="1920"  # gemma-3-270m: 1920

# Training parameters
SPECULATION_DEPTH="${SPECULATION_DEPTH:-6}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LORA_RANK="${LORA_RANK:-64}"
QUANTIZATION="${QUANTIZATION:-8bit}"

# Paths
DATA_DIR="${DATA_DIR:-./data}"
FEATURES_DIR="$DATA_DIR/features"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
PROCESSED_DIR="$DATA_DIR/processed"
EVAL_OUTPUT="${EVAL_OUTPUT:-evaluation_results.json}"

# ============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================

SKIP_DATA_GEN=false
SKIP_FEATURE_EXTRACTION=false
SKIP_TRAINING=false
SKIP_EVALUATION=false
SKIP_SECURITY_CHECK=false
RUN_DRY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET_MODEL="$2"
            shift 2
            ;;
        --drafter)
            DRAFTER_MODEL="$2"
            shift 2
            ;;
        --target-hidden-dim)
            TARGET_HIDDEN_DIM="$2"
            shift 2
            ;;
        --skip-data-gen)
            SKIP_DATA_GEN=true
            shift
            ;;
        --skip-feature-extraction)
            SKIP_FEATURE_EXTRACTION=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        --dry-run)
            RUN_DRY=true
            shift
            ;;
        --skip-security-check)
            SKIP_SECURITY_CHECK=true
            shift
            ;;
        --help|-h)
            echo "P-EAGLE Full Pipeline (Fixed)"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Model Selection:"
            echo "  --target MODEL          Target model (default: $TARGET_MODEL)"
            echo "  --drafter MODEL         Drafter model (default: $DRAFTER_MODEL)"
            echo "  --target-hidden-dim N   Target hidden dimension (default: $TARGET_HIDDEN_DIM)"
            echo ""
            echo "Stage Control:"
            echo "  --skip-data-gen           Skip data generation"
            echo "  --skip-feature-extraction Skip feature extraction"
            echo "  --skip-training           Skip training"
            echo "  --skip-evaluation         Skip evaluation"
            echo "  --dry-run                 Show commands without executing"
            echo ""
            echo "Examples:"
            echo "  $0                                              # Run all stages"
            echo "  $0 --skip-data-gen --skip-feature-extraction   # Retrain only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (use --help for usage)"
            exit 1
            ;;
    esac
done

# ============================================================================
# STEP 0: Environment Setup
# ============================================================================
echo ""
echo "Step 0: Environment Setup"
echo "-------------------------"

mkdir -p "$DATA_DIR"{"/raw","/processed","/features","/output"}
mkdir -p "$CHECKPOINT_DIR" "$OUTPUT_DIR" ./logs ./plot_scripts/plots

python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" || {
    echo "ERROR: PyTorch not available"
    exit 1
}

# Print configuration
echo ""
echo "Configuration:"
echo "  Target Model:   $TARGET_MODEL"
echo "  Drafter Model:  $DRAFTER_MODEL"
echo "  Hidden Dim:     $TARGET_HIDDEN_DIM"
echo "  Speculation K:  $SPECULATION_DEPTH"
echo "  Epochs:         $EPOCHS"
echo "  Batch Size:     $BATCH_SIZE"
echo "  Learning Rate:  $LEARNING_RATE"
echo "  LoRA Rank:      $LORA_RANK"

if [ "$RUN_DRY" = true ]; then
    echo ""
    echo "*** DRY RUN MODE - Commands will be shown but not executed ***"
fi

# ============================================================================
# STEP 1: Data Preparation (FIXED - includes loss_mask_segments)
# ============================================================================
if [ "$SKIP_DATA_GEN" = false ]; then
    echo ""
    echo "Step 1: Data Preparation (with loss_mask_segments)"
    echo "---------------------------------------------------"

    # Check for processed data
    PROCESSED_COUNT=$(find "$PROCESSED_DIR" -name "*.json" -type f 2>/dev/null | wc -l)

    if [ "$PROCESSED_COUNT" -eq 0 ]; then
        echo "ERROR: No processed JSON files found in $PROCESSED_DIR"
        echo "Please populate $PROCESSED_DIR with conversation JSON files"
        exit 1
    fi

    echo "Found $PROCESSED_COUNT processed files"

    # FIXED: generate_data.py now outputs loss_mask_segments (min-words default is 10)
    CMD="python3 scripts/generate_data.py --local --num-samples $NUM_SAMPLES --input-dir $PROCESSED_DIR --output $DATA_DIR/output --format openai --output-format jsonl --deduplicate"

    if [ "$RUN_DRY" = true ]; then
        echo "CMD: $CMD"
    else
        eval $CMD
    fi

    # Find the generated dataset file
    DATASET_FILE=$(find "$DATA_DIR/output" -name "dataset_*.jsonl" -type f 2>/dev/null | sort -t_ -k2,2n -k3 | tail -1)

    if [ -z "$DATASET_FILE" ] && [ "$RUN_DRY" = false ]; then
        echo "ERROR: No dataset file generated"
        exit 1
    fi

    echo "Dataset: $DATASET_FILE"

    # VERIFICATION: Check that loss_mask_segments exists
    if [ "$RUN_DRY" = false ]; then
        echo ""
        echo "Verifying dataset format..."
        python3 -c "
import json
with open('$DATASET_FILE', 'r') as f:
    sample = json.loads(f.readline())
    if 'loss_mask_segments' not in sample:
        print('ERROR: loss_mask_segments missing from dataset!')
        exit(1)
    lms = sample['loss_mask_segments']
    train_count = len(lms.get('train_indices', []))
    total = len(lms.get('segments', []))
    print(f'  ✓ loss_mask_segments present')
    print(f'  ✓ Trainable positions: {train_count}/{total}')
    if train_count == 0:
        print('  WARNING: No trainable positions found!')
"
    fi
else
    echo "Skipping data generation"
    DATASET_FILE=$(find "$DATA_DIR/output" -name "dataset_*.jsonl" -type f 2>/dev/null | sort -t_ -k2,2n -k3 | tail -1)
    echo "Using existing: $DATASET_FILE"
fi

# ============================================================================
# STEP 2: Feature Extraction
# ============================================================================
if [ "$SKIP_FEATURE_EXTRACTION" = false ]; then
    echo ""
    echo "Step 2: Feature Extraction"
    echo "--------------------------"

    # Clear old features
    if [ "$RUN_DRY" = false ]; then
        rm -f "$FEATURES_DIR"/*.pt
    fi

    CMD="python3 -m p_eagle.scripts.extract_features \
        --model_path $TARGET_MODEL \
        --tokenizer_path $DRAFTER_MODEL \
        --input_data $DATASET_FILE \
        --output_dir $FEATURES_DIR \
        --quantization $QUANTIZATION \
        --layers early,middle,final \
        --fusion mean \
        --batch_size 1 \
        --shard_size 5000 \
        --max_length 4096"

    if [ "$RUN_DRY" = true ]; then
        echo "CMD: $CMD"
    else
        eval $CMD
    fi

    if [ "$RUN_DRY" = false ]; then
        FEATURE_COUNT=$(find "$FEATURES_DIR" -name "*.pt" -type f | wc -l)
        echo "Extracted $FEATURE_COUNT feature shards"

        # VERIFICATION: Check that masks have non-zero values
        echo ""
        echo "Verifying feature masks..."
        python3 -c "
import torch
import glob
pt_files = glob.glob('$FEATURES_DIR/*_shard*.pt')
if pt_files:
    shard = torch.load(pt_files[0], map_location='cpu')
    mask = shard['loss_mask'][0]
    mask_sum = mask.sum().item()
    print(f'  ✓ Shard 0 mask sum: {mask_sum}')
    if mask_sum == 0:
        print('  ERROR: Mask is all zeros! Training will fail.')
        exit(1)
    print(f'  ✓ Non-zero masks confirmed')
"
    fi
else
    echo "Skipping feature extraction"
fi

# ============================================================================
# STEP 3: Training
# ============================================================================
if [ "$SKIP_TRAINING" = false ]; then
    echo ""
    echo "Step 3: Training Drafter"
    echo "------------------------"

    # Clear old checkpoints except logs
    if [ "$RUN_DRY" = false ]; then
        mkdir -p "$CHECKPOINT_DIR"
        find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 ! -name "logs" -type d -exec rm -rf {} + 2>/dev/null || true
    fi

    CMD="python3 -m p_eagle.scripts.train_drafter \
        --drafter_model $DRAFTER_MODEL \
        --target_hidden_dim $TARGET_HIDDEN_DIM \
        --feature_dir $FEATURES_DIR \
        --output_dir $CHECKPOINT_DIR \
        --num_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --warmup_steps 100 \
        --speculation_depth $SPECULATION_DEPTH \
        --use_lora \
        --lora_rank $LORA_RANK \
        --skip-hardware-check \
        --dataset-source $DATASET_FILE \
        $([ "$SKIP_SECURITY_CHECK" = true ] && echo "--skip-security-check")"

    if [ "$RUN_DRY" = true ]; then
        echo "CMD: $CMD"
    else
        eval $CMD
        echo "Training complete. Best model: $CHECKPOINT_DIR/best_model"
    fi
else
    echo "Skipping training"
fi

# ============================================================================
# STEP 4: Evaluation
# ============================================================================
if [ "$SKIP_EVALUATION" = false ]; then
    echo ""
    echo "Step 4: Evaluation"
    echo "------------------"

    CMD="python3 -m p_eagle.scripts.evaluate \
        --drafter_checkpoint $CHECKPOINT_DIR/best_model \
        --target_model $TARGET_MODEL \
        --baseline \
        --max_tokens 100 \
        --domain_test \
        --output $EVAL_OUTPUT"

    if [ "$RUN_DRY" = true ]; then
        echo "CMD: $CMD"
    else
        eval $CMD
        echo "Results: $EVAL_OUTPUT"

        # Display key metrics
        if [ -f "$EVAL_OUTPUT" ]; then
            echo ""
            echo "Key Metrics:"
            python3 -c "
import json
with open('$EVAL_OUTPUT', 'r') as f:
    data = json.load(f)
    if 'mean_acceptance_length' in data:
        print(f\"  Mean Acceptance Length (MAL): {data['mean_acceptance_length']:.2f}\")
    if 'speedup' in data:
        print(f\"  Speedup: {data['speedup']:.2f}x\")
"
        fi
    fi
else
    echo "Skipping evaluation"
fi

# ============================================================================
# STEP 5: Plotting
# ============================================================================
echo ""
echo "Step 5: Generating Plots"
echo "------------------------"

CMD="python3 -m plot_scripts.generate_plots --mode all --checkpoint_dirs $CHECKPOINT_DIR --eval_file $EVAL_OUTPUT --output_dir plot_scripts/plots"

if [ "$RUN_DRY" = true ]; then
    echo "CMD: $CMD"
else
    eval $CMD 2>/dev/null || echo "Plotting skipped (may require matplotlib)"
fi

# ============================================================================
# Done
# ============================================================================
echo ""
echo "================================"
if [ "$RUN_DRY" = true ]; then
    echo "DRY RUN COMPLETE"
else
    echo "PIPELINE COMPLETE!"
fi
echo "================================"
echo ""
echo "Quick Commands:"
echo "  Test inference:"
echo "    python3 -m p_eagle.scripts.run_inference \\"
echo "      --target_model $TARGET_MODEL \\"
echo "      --drafter_checkpoint $CHECKPOINT_DIR/best_model \\"
echo "      --prompt 'Your prompt here'"
echo ""
