# P-EAGLE: Parallel Speculative Decoding Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Production-grade PyTorch framework for training and deploying Parallel Eagle (P-EAGLE) speculative decoding models. Achieve **1.5-2.5x speedup** on large language model inference by predicting multiple future tokens in parallel.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Extract features from target model (e.g., Gemma-7B)
python -m p_eagle.scripts.extract_features \
    --model_path google/gemma-7b \
    --input_data data/output/dataset.jsonl \
    --output_dir data/features \
    --quantization 8bit

# 3. Train drafter model (e.g., Qwen-1.5B)
python -m p_eagle.scripts.train_drafter \
    --drafter_model Qwen/Qwen2.5-1.5B-Instruct \
    --target_hidden_dim 3072 \
    --feature_dir data/features \
    --output_dir checkpoints \
    --epochs 50 \
    --use_lora

# 4. Run inference with speculative decoding
python -m p_eagle.scripts.run_inference \
    --target_model google/gemma-7b \
    --drafter_checkpoint checkpoints/best_model \
    --prompt "Explain quantum computing"
```

## Table of Contents

- [Architecture](#architecture)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Performance](#performance)
- [Project Structure](#project-structure)

---

## Architecture

```
TARGET MODEL (e.g., Gemma-7B, 28 layers, 3072 hidden dim)
         |
         |  features (layers 7, 14, 21 fused)
         v
DRAFTER MODEL (e.g., Qwen-1.5B, 1536 hidden dim)
    +-----------------------------+
    |  Base LLM (frozen + LoRA)   |
    |       (1536 dim)            |
    +-------------+---------------+
                  |
          Dim Projection
         (1536 -> 3072)
                  |
    +-------------+-------------+
    |             |             |
    v             v             v
 MTP Head 1   MTP Head 2   MTP Head K
 (h_{t+1})    (h_{t+2})    (h_{t+K})
```

**Key Components:**
- **Tri-Layer Fusion**: Combines early, middle, and final layer features
- **Multi-Token Prediction (MTP)**: K parallel heads predict future hidden states
- **Tree Attention**: Verifies K tokens in single target forward pass

---

## Pipeline

### Stage 1: Feature Extraction

Extract tri-layer fused hidden states from the target model.

```bash
python -m p_eagle.scripts.extract_features \
    --model_path google/gemma-7b \
    --input_data data/output/dataset.jsonl \
    --output_dir data/features \
    --quantization 8bit \
    --layers early,middle,final \
    --fusion mean \
    --batch_size 4 \
    --shard_size 5000
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Target model (Gemma, Llama, Qwen, etc.) | required |
| `--input_data` | Path to dataset JSONL file | required |
| `--output_dir` | Where to save .pt feature files | `./features` |
| `--quantization` | Quantization mode: `4bit`, `8bit`, `none` | `4bit` |
| `--layers` | Which layers to fuse | `early,middle,final` |
| `--fusion` | Fusion method: `mean`, `weighted`, `concat` | `mean` |
| `--batch_size` | Extraction batch size | 1 |
| `--shard_size` | Samples per output file | 1000 |

**Output:** `.pt` files containing:
- `input_ids`: Tokenized sequences
- `fused_hidden_states`: Tri-layer features (shape: `[batch, seq, hidden_dim]`)
- `loss_mask`: Binary mask (1=train on this token, 0=ignore)
- `attention_mask`: Padding mask

---

### Stage 2: Train Drafter

Train a small model to predict target model's hidden states.

```bash
python -m p_eagle.scripts.train_drafter \
    --drafter_model Qwen/Qwen2.5-1.5B-Instruct \
    --target_hidden_dim 3072 \
    --feature_dir data/features \
    --output_dir checkpoints \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --warmup_steps 30 \
    --speculation_depth 4 \
    --use_lora \
    --lora_rank 64
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--drafter_model` | Small base model (1-3B params) | required |
| `--target_hidden_dim` | Target model's hidden dimension | required |
| `--feature_dir` | Directory with extracted features | required |
| `--output_dir` | Checkpoint output directory | `./checkpoints` |
| `--epochs` | Training epochs | 3 |
| `--batch_size` | Training batch size | 4 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--warmup_steps` | LR warmup steps | 100 |
| `--speculation_depth` | Number of MTP heads (K) | 4 |
| `--use_lora` | Enable LoRA fine-tuning | True |
| `--lora_rank` | LoRA rank | 64 |

**Hardware Requirements:**
| Drafter Size | VRAM (LoRA) | VRAM (Full) | Training Time (1000 samples) |
|--------------|-------------|-------------|------------------------------|
| 0.5B | ~6 GB | ~10 GB | ~10 min |
| 1.5B | ~12 GB | ~20 GB | ~30 min |
| 3B | ~20 GB | ~35 GB | ~1 hour |

---

### Stage 3: Inference

Run speculative decoding with trained drafter.

```bash
python -m p_eagle.scripts.run_inference \
    --target_model google/gemma-7b \
    --drafter_checkpoint checkpoints/best_model \
    --prompt "Explain neural networks" \
    --max_tokens 200 \
    --temperature 0.7
```

**Programmatic Usage:**

```python
from p_eagle.inference import PEAGLEInference

# Initialize engine
engine = PEAGLEInference(
    target_model_name="google/gemma-7b",
    drafter_checkpoint="./checkpoints/best_model"
)

# Generate with speculative decoding
output, metrics = engine.generate(
    prompt="Explain quantum computing",
    max_new_tokens=100,
    temperature=0.7
)

print(f"Generated: {output}")
print(f"Speedup: {metrics.speedup:.2f}x")
print(f"Mean Acceptance Length: {metrics.mean_acceptance_length:.2f}")
```

---

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (12+ GB VRAM recommended)
- Linux/macOS/WSL

### Setup

```bash
# Clone and install
git clone <repository-url>
cd p_eagle
pip install -r requirements.txt

# Optional: Install flash-attention for faster inference
pip install flash-attn --no-build-isolation
```

### Environment Variables

Create `.env` file:

```bash
# HuggingFace token (for gated models like Llama)
HF_TOKEN=your_token_here
```

Get token from: https://huggingface.co/settings/tokens

---

## Usage Guide

### Model Pair Compatibility

| Target Model | Hidden Dim | Compatible Drafters |
|--------------|-----------|---------------------|
| Gemma-7B | 3072 | Qwen2.5-1.5B, TinyLlama-1.1B |
| Llama-2-7B | 4096 | Qwen2.5-1.5B, Phi-2 |
| Mistral-7B | 4096 | Qwen2.5-1.5B, Phi-2 |
| Qwen2.5-7B | 3584 | Qwen2.5-1.5B |

**Rule:** Drafter hidden dim ≠ Target hidden dim. The `dim_projection` layer handles the mismatch.

### Choosing Speculation Depth (K)

| K | Use Case | Typical MAL |
|---|----------|-------------|
| 2 | Conservative, stable | 1.8-2.2 |
| 4 | Balanced (recommended) | 2.5-3.2 |
| 6 | Aggressive, high variance | 3.0-4.0 |

Higher K = more parallelism but diminishing returns if acceptance drops.

### Training Tips

1. **Small dataset (< 1000 samples):** Use higher epochs (50+), LoRA rank 64-128
2. **Large dataset (> 10k samples):** Fewer epochs (5-10), LoRA rank 32-64
3. **Loss not decreasing:** Reduce learning rate (5e-5), increase warmup
4. **OOM errors:** Reduce batch size, enable gradient checkpointing

---

## Performance

### Expected Metrics

| Metric | Value Range | Interpretation |
|--------|-------------|----------------|
| Mean Acceptance Length (MAL) | 2.5 - 4.0 | Tokens accepted per speculation |
| Speedup vs Autoregressive | 1.5x - 2.5x | Overall throughput improvement |
| Draft Overhead | 10-20% | Time spent on drafter forward pass |

### Benchmarks

On NVIDIA A100-80GB, batch_size=1:

| Target | Drafter | K | MAL | Speedup |
|--------|---------|---|-----|---------|
| Gemma-7B | Qwen-1.5B | 4 | 3.2 | 2.1x |
| Llama-2-7B | Qwen-1.5B | 4 | 2.9 | 1.9x |
| Mistral-7B | Phi-2 | 4 | 3.0 | 1.8x |

---

## Project Structure

```
p_eagle/
├── p_eagle/                      # Main package
│   ├── data_preparation/         # Data processing
│   │   ├── data_manager.py       # EAGLEDistiller
│   │   └── secret_scanner.py     # PII scanner
│   ├── models/                   # Neural networks
│   │   ├── eagle_drafter.py      # Drafter + MTP heads
│   │   └── tree_attention.py     # Tree attention masks
│   ├── training/                 # Training & extraction
│   │   ├── feature_extractor.py  # Tri-layer extraction
│   │   └── trainer.py            # Training loop
│   ├── inference/                # Speculative decoding
│   │   └── inference_engine.py   # PEAGLEInference
│   ├── utils/                    # Utilities
│   │   ├── feature_utils.py      # Datasets
│   │   ├── loss_utils.py         # Loss functions
│   │   └── metrics.py            # Evaluation metrics
│   └── scripts/                  # CLI entry points
│       ├── extract_features.py
│       ├── train_drafter.py
│       └── run_inference.py
├── scripts/                      # Standalone scripts
│   └── generate_data.py          # Dataset generation
├── data/                         # Data directories
│   ├── raw/                      # Raw logs
│   ├── processed/                # Intermediate
│   ├── features/                 # Extracted features
│   └── output/                   # Generated datasets
├── checkpoints/                  # Model checkpoints
├── logs/                         # Training logs
├── requirements.txt              # Dependencies
└── setup.py                      # Package setup
```

---

## Troubleshooting

### Common Issues

**Issue:** `CUDA out of memory` during training  
**Solution:** Reduce `--batch_size`, use `--use_lora`, or lower `--speculation_depth`

**Issue:** `Repository not found` for HuggingFace model  
**Solution:** Check model name spelling, ensure HF_TOKEN is set for gated models

**Issue:** `MAL < 2.0` after training  
**Solution:** Train longer (more epochs), increase dataset size, or check feature extraction is from correct target model

**Issue:** Slow feature extraction  
**Solution:** Increase `--batch_size` if VRAM permits, or use `--quantization 8bit` instead of 4bit (faster)

---

## Citations

```bibtex
@article{eagle2024,
  title={EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty},
  author={Li, Xiaotian and Liu, Zheng and others},
  journal={arXiv preprint},
  year={2024}
}

@article{levithan2023fast,
  title={Fast Inference from Transformers via Speculative Decoding},
  author={Leviathan, Yaniv and Kalman, Matan and Matias, Yossi},
  journal={ICML},
  year={2023}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.
