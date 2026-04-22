# P-EAGLE: Parallel Speculative Decoding Framework

Production-grade PyTorch framework for training and deploying Parallel Eagle (P-EAGLE) speculative decoding models.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TARGET MODEL                           │
│                   (e.g., Gemma-7B)                          │
│                      3072 hidden dim                         │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ features (h_t from layers 6, 12, 18)
                              │
┌─────────────────────────────────────────────────────────────┐
│                      DRAFTER MODEL                          │
│                  (e.g., Qwen-1.5B)                          │
│                      1536 hidden dim                         │
│                                                              │
│  ┌─────────────┐    ┌──────────────────┐                    │
│  │ Base LLM    │───→│ Dim Projection   │───→ 3072 dim       │
│  │ (LoRA)      │    │ (1536 → 3072)    │                    │
│  └─────────────┘    └──────────────────┘                    │
│                              │                               │
│           ┌──────────────────┼──────────────────┐            │
│           ↓                  ↓                  ↓            │
│     ┌──────────┐      ┌──────────┐      ┌──────────┐        │
│     │ MTP h_1  │      │ MTP h_2  │      │ MTP h_K  │        │
│     │  (t+1)   │      │  (t+2)   │      │  (t+K)   │        │
│     └──────────┘      └──────────┘      └──────────┘        │
│          │                  │                  │             │
└──────────┼──────────────────┼──────────────────┼─────────────┘
           │                  │                  │
           ▼                  ▼                  ▼
      Predict h_{t+1}   Predict h_{t+2}   Predict h_{t+K}
      from h_t         from h_t          from h_t
```

## Pipeline

### Stage 1: Feature Extraction
```bash
python extract_features.py \
    --model_path google/gemma-7b-it \
    --input_data ./output/dataset_*.jsonl \
    --output_dir ./features \
    --quantization 4bit \
    --layers early,middle,final \
    --fusion mean
```

**Output:** `.pt` files containing:
- `input_ids`: Tokenized input
- `fused_hidden_states`: Tri-layer fused features from target
- `loss_mask`: Binary mask (1=assistant, 0=other)

### Stage 2: Train Drafter
```bash
python train_drafter.py \
    --drafter_model Qwen/Qwen2-1.5B-Instruct \
    --target_hidden_dim 3072 \
    --feature_dir ./features \
    --output_dir ./checkpoints \
    --speculation_depth 4 \
    --use_lora \
    --lora_rank 64 \
    --batch_size 4 \
    --num_epochs 3
```

**Key Features:**
- **MTP Heads:** K parallel prediction heads for h_{t+1}...h_{t+K}
- **Dim Projection:** Learnable adapter (drafter_dim → target_dim)
- **Masked MSE Loss:** Only on assistant tokens
- **Memory Efficient:** PagedAdamW8bit + LoRA + 4-bit base

### Stage 3: Inference
```bash
python inference.py \
    --target_model google/gemma-7b-it \
    --drafter_checkpoint ./checkpoints/best_model \
    --prompt "Explain quantum computing" \
    --max_tokens 200
```

**Features:**
- Parallel tree construction
- Tree-attention verification
- Mean Acceptance Length (MAL) tracking
- Speedup measurement

## Key Innovations

### 1. Tri-Layer Feature Fusion
Instead of using only the final layer, we fuse early, middle, and final layers:

```python
fused = mean([
    hidden_states[layer_6],   # Early semantics
    hidden_states[layer_12],  # Mid-level reasoning
    hidden_states[layer_18]   # Final representations
])
```

This captures multi-granular information for better distillation.

### 2. Multi-Token Prediction (MTP)
Standard EAGLE predicts one token ahead. P-EAGLE predicts K tokens in parallel:

```python
# Single forward pass, K predictions
mtp_predictions = [
    head_1(h_t) → h_{t+1},  # 1 token ahead
    head_2(h_t) → h_{t+2},  # 2 tokens ahead
    ...
    head_K(h_t) → h_{t+K}   # K tokens ahead
]
```

### 3. Tree Attention for Verification
Verifies K speculative tokens with single target forward pass:

```
Position:  0    1    2    3    4    5    6
Content:  [S]  [U]  [A]  [D1] [D2] [D3] [D4]
           │    │    │    │    │    │    │
Attention: └────┴────┘    │    │    │    │
               ↑          │    │    │    │
            verified    draft draft draft draft
            context     tok1  tok2  tok3  tok4
```

Each draft token attends to all verified tokens + previous drafts.

## Performance Expectations

| Metric | Typical Value |
|--------|--------------|
| Mean Acceptance Length (MAL) | 2.5 - 3.5 tokens |
| Speedup over autoregressive | 1.5x - 2.5x |
| Training VRAM (1.5B drafter) | ~16-20 GB |
| Inference VRAM (7B target + 1.5B drafter) | ~20-24 GB |

## Directory Structure

```
p_eagle/
├── extract_features.py      # Stage 1: Feature extraction
├── train_drafter.py         # Stage 2: Drafter training
├── inference.py             # Stage 3: Speculative decoding
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Installation

```bash
pip install -r requirements.txt

# For 4-bit quantization
pip install bitsandbytes

# For secret scanning (optional)
pip install detect-secrets presidio-analyzer
```

## Usage Example

```python
from p_eagle.inference import PEAGLEInference

# Initialize
engine = PEAGLEInference(
    target_model_name="google/gemma-7b-it",
    drafter_checkpoint="./checkpoints/best_model"
)

# Generate
output, metrics = engine.generate(
    prompt="Explain neural networks",
    max_new_tokens=100
)

print(f"Generated: {output}")
print(f"Speedup: {metrics.speedup:.2f}x")
print(f"Mean Acceptance Length: {metrics.mean_acceptance_length:.2f}")
```

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
