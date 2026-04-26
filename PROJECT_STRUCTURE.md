# P-EAGLE Project Structure

This document describes the organization of the P-EAGLE codebase.

## Directory Overview

```
p_eagle/
├── p_eagle/                      # Main Python package
│   ├── __init__.py
│   ├── data_preparation/         # Data preprocessing & PII handling
│   ├── models/                   # Neural network architectures
│   ├── training/                 # Training loops & feature extraction
│   ├── inference/                # Speculative decoding engine
│   ├── utils/                    # Helper utilities
│   └── scripts/                  # CLI entry points
├── scripts/                      # Standalone utilities
├── data/                         # Data storage (gitignored)
├── checkpoints/                  # Model checkpoints (gitignored)
├── logs/                         # Training logs (gitignored)
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
└── README.md                     # Main documentation
```

## Core Modules

### 1. Data Preparation (`p_eagle/data_preparation/`)

| File | Class/Function | Purpose |
|------|----------------|---------|
| `data_manager.py` | `EAGLEDistiller` | Converts LiteLLM logs to training datasets |
| `data_manager.py` | `ConversationDataset` | PyTorch dataset for conversations |
| `secret_scanner.py` | `SmartSecretScanner` | Detects and masks PII/secrets |

**Usage:**
```python
from p_eagle.data_preparation import EAGLEDistiller

distiller = EAGLEDistiller(output_dir="data/output")
dataset_path = distiller.create_dataset_from_raw("data/raw/")
```

### 2. Models (`p_eagle/models/`)

| File | Class | Purpose |
|------|-------|---------|
| `eagle_drafter.py` | `EagleDrafterModel` | Main drafter with LoRA support |
| `eagle_drafter.py` | `EagleMTPHead` | Multi-token prediction head |
| `eagle_drafter.py` | `DimensionProjection` | Projects drafter dim → target dim |
| `tree_attention.py` | `TreeAttentionMask` | Tree-structured attention for verification |

**Architecture:**
```python
from p_eagle.models import EagleDrafterModel

model = EagleDrafterModel(
    base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    target_hidden_dim=3072,  # Gemma-7B
    speculation_depth=4,
    use_lora=True,
    lora_rank=64
)
```

### 3. Training (`p_eagle/training/`)

| File | Class/Function | Purpose |
|------|----------------|---------|
| `feature_extractor.py` | `FeatureExtractor` | Extracts tri-layer features from target |
| `feature_extractor.py` | `TriLayerConfig` | Configures which layers to extract |
| `trainer.py` | `EagleTrainer` | Main training loop |
| `trainer.py` | `main()` | CLI entry point |

**Training Loop:**
```python
from p_eagle.training import EagleTrainer

trainer = EagleTrainer(
    drafter_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    target_hidden_dim=3072,
    feature_dir="data/features",
    output_dir="checkpoints",
    use_lora=True,
    lora_rank=64
)
trainer.train()
```

### 4. Inference (`p_eagle/inference/`)

| File | Class | Purpose |
|------|-------|---------|
| `inference_engine.py` | `PEAGLEInference` | End-to-end speculative decoding |
| `inference_engine.py` | `SpeculationResult` | Result container with metrics |

**Inference:**
```python
from p_eagle.inference import PEAGLEInference

engine = PEAGLEInference(
    target_model_name="google/gemma-7b",
    drafter_checkpoint="checkpoints/best_model"
)
output, metrics = engine.generate("Hello, world!", max_new_tokens=50)
```

### 5. Utils (`p_eagle/utils/`)

| File | Class/Function | Purpose |
|------|----------------|---------|
| `feature_utils.py` | `EagleDataset` | Dataset for feature extraction |
| `feature_utils.py` | `EagleTrainingDataset` | Dataset for training |
| `feature_utils.py` | `align_segments_to_tokens()` | Aligns conversation segments |
| `feature_utils.py` | `fuse_tri_layer_features()` | Fuses early/middle/final layers |
| `loss_utils.py` | `masked_mse_loss()` | MSE loss with masking |
| `metrics.py` | `MetricsTracker` | Tracks training metrics |
| `metrics.py` | `GenerationMetrics` | Inference metrics (MAL, speedup) |

## CLI Scripts (`p_eagle/scripts/`)

These are entry points for the three-stage pipeline:

| Script | Command | Purpose |
|--------|---------|---------|
| `extract_features.py` | `python -m p_eagle.scripts.extract_features` | Stage 1: Feature extraction |
| `train_drafter.py` | `python -m p_eagle.scripts.train_drafter` | Stage 2: Drafter training |
| `run_inference.py` | `python -m p_eagle.scripts.run_inference` | Stage 3: Speculative decoding |

## Data Directories

### `data/raw/` - Raw Input
Contains original LiteLLM log files in JSON format.

```
data/raw/
├── litellm_logs_2024_01.json
└── litellm_logs_2024_02.json
```

### `data/processed/` - Intermediate
Preprocessed conversations, deduplicated and cleaned.

```
data/processed/
├── conversations_cleaned.jsonl
└── stats.json
```

### `data/features/` - Extracted Features
PyTorch tensors from target model forward passes.

```
data/features/
├── dataset_20240424_shard0000.pt
├── dataset_20240424_shard0001.pt
└── ...
```

Each `.pt` file contains:
- `input_ids`: `[batch, seq_len]` token IDs
- `fused_hidden_states`: `[batch, seq_len, hidden_dim]` features
- `loss_mask`: `[batch, seq_len]` binary mask
- `attention_mask`: `[batch, seq_len]` padding mask
- `model_name`: Source model name
- `layer_indices`: Which layers were extracted
- `fusion_mode`: Fusion method used
- `num_samples`: Number of samples

### `data/output/` - Generated Datasets
Final training datasets in JSONL format.

```
data/output/
└── dataset_20240424_184927.jsonl
```

## Checkpoints (`checkpoints/`)

Saved model states during and after training.

```
checkpoints/
├── best_model/                   # Best validation loss
│   ├── adapter_model/            # LoRA weights
│   ├── dimension_projection.pt
│   ├── mtp_heads/
│   └── config.json
├── checkpoint_step_1000/         # Periodic checkpoint
└── final_model/                  # Last epoch
```

## Logs (`logs/`)

TensorBoard logs and training metrics.

```
logs/
├── training_20240424/
│   └── events.out.tfevents.*
└── inference_20240424/
    └── results.json
```

View with: `tensorboard --logdir logs/`

## File Naming Conventions

### Dataset Files
- `dataset_YYYYMMDD_HHMMSS.jsonl` - Generated datasets
- `*_shard####.pt` - Feature shards (#### = zero-padded index)

### Checkpoint Directories
- `best_model/` - Lowest validation loss
- `checkpoint_step_NNNN/` - Every N steps
- `epoch_N/` - End of epoch N

### Log Files
- `events.out.tfevents.*` - TensorBoard events
- `training_*.log` - Text training logs
- `results_*.json` - Inference results

## Adding New Components

### Adding a New Model Architecture

1. Create file in `p_eagle/models/`
2. Import in `p_eagle/models/__init__.py`
3. Update `eagle_drafter.py` if extending base drafter

### Adding a New Loss Function

1. Add function to `p_eagle/utils/loss_utils.py`
2. Update `trainer.py` to use the new loss

### Adding a New CLI Command

1. Create script in `p_eagle/scripts/`
2. Make executable: `chmod +x p_eagle/scripts/my_script.py`
3. Run as: `python -m p_eagle.scripts.my_script`

## Testing

Test files mirror the package structure:

```
tests/
├── test_data_preparation/
├── test_models/
├── test_training/
├── test_inference/
└── test_utils/
```

Run tests:
```bash
pytest tests/
```

## Development Workflow

1. **Prepare data:** Place LiteLLM logs in `data/raw/`
2. **Generate dataset:** `python scripts/generate_data.py`
3. **Extract features:** `python -m p_eagle.scripts.extract_features`
4. **Train drafter:** `python -m p_eagle.scripts.train_drafter`
5. **Evaluate:** `python -m p_eagle.scripts.run_inference`
6. **Iterate:** Adjust hyperparameters, retrain

## Dependencies

Core dependencies by module:

| Module | Key Dependencies |
|--------|------------------|
| `models` | `torch`, `transformers`, `peft` |
| `training` | `torch`, `bitsandbytes`, `tqdm` |
| `inference` | `torch`, `transformers` |
| `data_preparation` | `presidio-analyzer`, `detect-secrets` |
| `utils` | `numpy`, `torch` |

See `requirements.txt` for full list.
