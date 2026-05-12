# P-EAGLE Architecture Deep Dive

## How It Actually Works

### Training Flow
```
1. Target Model (e.g., Gemma-7B) processes text
   └── Outputs hidden states from layers [7, 14, 21]
   └── These are fused into single vector per token

2. Drafter Model (e.g., Gemma-2B) is trained with EAGLE-3 architecture:
   └── Input: Concatenated [Token Embeddings; Target Hidden States]
   └── First layer modified to accept 2x hidden size input
   └── Separate norms for embeddings and hidden portions
   └── Output: Hidden state vectors matching target's fused hidden states

3. Loss Computation:
   └── KL divergence between token distributions
       ├── Pass predicted hidden through TARGET's lm_head → logits_pred
       ├── Pass target hidden through TARGET's lm_head → logits_target
       └── Minimize KL(logits_pred || logits_target)
```

### Inference Flow
```
1. Drafter sees current token sequence
   └── Generates K parallel hidden state predictions (h_{t+1}, ..., h_{t+K})

2. Convert hidden states to tokens:
   └── Pass each h_{t+i} through TARGET's lm_head
   └── Get token probabilities → sample token

3. Target verifies K drafts in parallel:
   └── Target model sees [context] + [draft_token_1, ..., draft_token_K]
   └── Outputs K logits (one per position)
   └── Accept tokens where target agrees with draft

4. Keep accepted tokens, reject + resample from target for rest
```

## Critical Requirements

### Vocabulary Compatibility
- **Training**: Drafter and target must use COMPATIBLE tokenizers
- **Inference**: Drafter outputs hidden states → target's lm_head → target tokens
- **Key Point**: Token IDs flow through drafter, but tokens come from target's lm_head

### Compatible Model Pairs
| Target | Drafter | Vocab | Works? |
|--------|---------|-------|--------|
| Gemma-7B | Gemma-2B | Same ✅ | Yes |
| Qwen-7B | Qwen-1.5B | Same ✅ | Yes |
| GLM-5.1 | GLM-1.5B | Same ✅ | Yes |
| Gemma-7B | Qwen-1.5B | Different ❌ | No (vocab mismatch) |

## Why Previous Training Failed

### The Problem
Training used MSE loss on hidden state vectors:
```python
loss = MSE(pred_hidden, target_hidden)
```

**BUT**: Two vectors can be very close in MSE but produce completely
different argmax tokens through lm_head!

Example:
- `pred_hidden` → lm_head → token_id=45 ("hello")
- `target_hidden` → lm_head → token_id=12345 (random garbage)
- MSE between vectors: 0.001 (very small)
- Token match: NO ❌

### The Fix
Use KL divergence on token distributions:
```python
pred_logits = target_lm_head(pred_hidden)
target_logits = target_lm_head(target_hidden)
loss = KL(pred_logits, target_logits)
```

This ensures predicted hidden states produce the SAME token distribution
as target hidden states when passed through the target's lm_head.

## For GLM Family Models

Since GLM-5.1 and GLM-1.5B share the same vocabulary:

1. **Feature Extraction**: Use GLM-5.1 to extract hidden states
   ```bash
   python extract_features --model_path THUDM/glm-5.1 --tokenizer_path THUDM/glm-1.5b
   ```

2. **Training**: Train GLM-1.5B to predict GLM-5.1's hidden states
   ```bash
   python train_drafter --drafter_model THUDM/glm-1.5b --target_hidden_dim 4096
   ```

3. **Inference**: GLM-1.5B hidden → GLM-5.1 lm_head → GLM tokens ✓

## Multi-Token Prediction (MTP) Heads

The drafter uses K parallel MTP heads to predict future hidden states:

```
projected_hidden ──┬─→ MTP Head 1 ──→ h_{t+2}
                   ├─→ MTP Head 2 ──→ h_{t+3}
                   ├─→ MTP Head 3 ──→ h_{t+4}
                   └─→ MTP Head K ──→ h_{t+K+1}
```

Each MTP head is a 2-layer MLP:
```python
EagleMTPHead(
    hidden_dim=target_hidden_dim,
    target_hidden_dim=target_hidden_dim,
    num_layers=2,  # Linear → LayerNorm → GELU → Dropout → Linear
    dtype=torch.bfloat16
)
```

**Training:** Each head k is trained to predict the hidden state at position t+k+1 from position t.
**Inference:** All K heads run in parallel in a single forward pass.

## Loss Mask Segments

Training uses segment-based loss masks to identify which tokens to train on:

```json
{
  "loss_mask_segments": {
    "train_indices": [2],      // Assistant message indices
    "ignore_indices": [0, 1],  // System/user indices
    "segments": [
      {"index": 0, "role": "system", "mask": 0},
      {"index": 1, "role": "user", "mask": 0},
      {"index": 2, "role": "assistant", "mask": 1}
    ]
  }
}
```

- `mask=1`: Train on these tokens (assistant responses)
- `mask=0`: Ignore these tokens (system, user, tool)

**Critical:** If `loss_mask_segments` is missing or empty, training loss will be 0 and MAL will be 1.0 (no learning).

## Content Format Handling

The feature extractor handles multiple content formats:
- Plain string: `"Hello world"`
- Nested list: `[{"type": "text", "text": "Hello"}]`
- String-encoded list: `"[{'type': 'text', 'text': 'Hello'}]"`

All formats are normalized to strings for tokenization.

## EAGLE-3 Architecture Details

### Hidden State Injection

The key innovation in EAGLE-3 is hidden state injection via concatenation:

```python
# First layer input: concatenate embeddings with target hidden states
input = torch.cat([token_embeddings, target_hidden_states], dim=-1)
# Shape: [batch, seq_len, 2 * hidden_size]
```

Inside the first layer:
1. Split input into embeddings and hidden portions
2. Apply separate LayerNorm to each
3. Re-concatenate for attention
4. Project attention output back to hidden_size for residual

```python
class Eagle3FirstLayer(nn.Module):
    def forward(self, hidden_states):
        # Split concatenated input
        embeds, hidden = hidden_states[..., :mid], hidden_states[..., mid:]

        # Store residual from hidden portion
        residual = hidden

        # Separate norms
        embeds = self.embed_norm(embeds)
        hidden = self.hidden_norm(hidden)

        # Re-concatenate for attention
        hidden_states = torch.cat([embeds, hidden], dim=-1)

        # Attention (outputs 2x hidden)
        attn_output = self.base_layer.self_attn(hidden_states)

        # Project back to hidden_size for residual
        attn_projected = self.output_proj(attn_output)
        hidden = residual + attn_projected
        ...
```

### Benefits
- Direct access to target model's representations
- Better alignment between drafter and target
- Improved token prediction accuracy

## Implementation Status

✅ Implemented: EAGLE-3 architecture with hidden state injection
✅ Fixed: Token-level loss during training (KL divergence)
✅ Fixed: Token alignment with target's lm_head
✅ Fixed: Vocab compatibility checking
✅ Fixed: Content parsing for nested formats
✅ Fixed: loss_mask_segments generation in dataset
✅ Fixed: Training/inference hidden injection alignment
⚠️  Note: Target lm_head is loaded from feature files during training if available

## File Structure

- `p_eagle/models/peagle_drafter.py` - Main drafter model with EAGLE-3 architecture
  - `Eagle3FirstLayer` - Modified first layer with concatenation handling
  - `EagleMTPHead` - Multi-token prediction heads
  - `EagleDrafterModel` - Complete drafter with LoRA support
